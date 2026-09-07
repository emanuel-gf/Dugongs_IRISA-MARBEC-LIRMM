"""
tile_export_pipeline.py
========================

Tiles full-resolution images from a FiftyOne dataset into fixed-size square
tiles (shift-inward edge rule, no resize/no padding) and exports them into
ONE output tree, split by a SINGLE shared area-ratio threshold:

    <output_dir>/
        positive/
            images/    <stem>__tile_<y>_<x>_t<tile>_o<overlap>.jpg
            labels/    <stem>__tile_<y>_<x>_t<tile>_o<overlap>.txt
            metadata/  <stem>__tile_<y>_<x>_t<tile>_o<overlap>.json
        negative/
            images/    ...
            labels/    ... (always empty .txt, kept for YOLO loader consistency)
            metadata/  ...

A tile lands in positive/ if AT LEAST ONE ground-truth detection has at
least --min-area-ratio of its ORIGINAL area surviving the clip into that
tile; otherwise it lands in negative/. This single split serves BOTH
downstream tasks:

  - Detection: every box in positive/labels/*.txt already cleared the
    same --min-area-ratio bar, so the YOLO label file only ever contains
    boxes that are "real enough" to be worth learning from. Train
    directly on positive/ (+ as much of negative/ as desired for
    background).

  - Classification: positive/ vs negative/ folder membership IS the
    binary label. No threshold to re-apply later -- if a tile is under
    positive/, it's class 1; under negative/, it's class 0.

Storing only ONE copy of each tile (instead of separate detection/ and
classification/ trees) avoids doubling disk usage for what would
otherwise be near-duplicate image files.

GEOMETRY (shift-inward edge rule)
----------------------------------
For images whose width/height is not a multiple of the stride, the last
tile in each row/column is shifted so its right/bottom edge aligns with
the image boundary, keeping every tile EXACTLY tile_size x tile_size in
native pixels -- no resize, no padding, ever.

    x_end   = min(x + tile_size, W)
    x_start = max(0, x_end - tile_size)

GROUND TRUTH FIELD
----------------------------------
--gt-field selects which FiftyOne Detections field to read boxes from:
    ground_truth    -> typically "dugong" only
    ground_truthv2  -> "dugong" + "calf" (after relabel_calves)
Whatever labels exist in that field are carried through to the YOLO label
lines as-is (multi-class capable, not hardcoded to a single class index).

NEGATIVE SUBSAMPLING
----------------------------------
--k-neg optionally subsamples negative/ tiles per source image (useful if
you want a smaller, more balanced set for quick detection experiments).
Omit it to keep ALL negative tiles -- recommended if you also intend to
use this same tree for the classification task, since classification
benefits from seeing the full variety of negative habitat/water types.

NOTE ON MANIFESTS
----------------------------------
This script does NOT generate a JSON/CSV manifest. It only writes files
into the positive/negative folder structure described above. Manifest
generation (e.g. for a classification training loader) is intentionally
left to a separate, later script -- keeping this script's scope limited
to geometry + export.

Usage
-----
  python tile_export_pipeline.py \\
      --dataset Domain-Shift-WP --port 44123 \\
      --gt-field ground_truthv2 \\
      --output-dir /share/home/e2406743/dataset/tiled/wp_1024_o250 \\
      --tile-size 1024 --overlap 250 \\
      --min-area-ratio 0.45 \\
      --k-neg 2 --seed 42
"""

import os
import json
import random
import argparse
from pathlib import Path

from PIL import Image


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="Tile export pipeline -- single positive/negative tree, "
                    "shared by detection and classification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset",    "-d", required=True,
                   help="Name of the FiftyOne dataset to process.")
    p.add_argument("--port",       default="44123",
                   help="MongoDB port. (default: 44123)")
    p.add_argument("--gt-field",   default="ground_truth",
                   help="FiftyOne Detections field to read boxes from "
                        "(e.g. 'ground_truth' or 'ground_truthv2'). "
                        "(default: ground_truth)")

    p.add_argument("--tile-size",  type=int, default=1024,
                   help="Tile size in pixels (square). (default: 1024)")
    p.add_argument("--overlap",    type=int, default=250,
                   help="Overlap between adjacent tiles in pixels. (default: 250)")

    p.add_argument("--min-area-ratio", type=float, default=0.45,
                   help="SHARED threshold: minimum fraction of a "
                        "detection's original area that must survive the "
                        "clip into a tile for (a) that box to be written "
                        "into the YOLO label file, and (b) the tile to be "
                        "placed in positive/ rather than negative/. "
                        "(default: 0.45)")

    p.add_argument("--k-neg",      type=int, default=None,
                   help="Negative tiles to sample per source image. Omit "
                        "to keep ALL negative tiles (recommended if this "
                        "tree will also be used for classification).")
    p.add_argument("--seed",       type=int, default=42,
                   help="Random seed for negative sampling. (default: 42)")

    p.add_argument("--output-dir", "-o", required=True,
                   help="Root folder. Creates positive/ and negative/ "
                        "subtrees here.")
    p.add_argument("--skip-grid",  action="store_true",
                   help="Skip storing the roi_grid field in FiftyOne "
                        "(visualisation only -- has no effect on export).")
    return p.parse_args()


# ── Tile geometry (shift-inward edge rule) ────────────────────────────────────

def compute_tiles(img_w: int, img_h: int, tile_size: int, overlap: int) -> list:
    """
    Returns a list of tiles as absolute pixel rectangles:
        [x_start, y_start, x_end, y_end]  (all integers)

    Shift-inward edge rule: the last tile in a row/column is shifted
    LEFT/UP so its right/bottom edge aligns with the image boundary,
    keeping its size EXACTLY tile_size x tile_size -- no resize, no pad.
    """
    stride = tile_size - overlap
    tiles = []

    for y in range(0, img_h, stride):
        for x in range(0, img_w, stride):
            x_end   = min(x + tile_size, img_w)
            y_end   = min(y + tile_size, img_h)
            x_start = max(0, x_end - tile_size)
            y_start = max(0, y_end - tile_size)
            tiles.append((x_start, y_start, x_end, y_end))

    return tiles


# ── Ground truth loader (from FiftyOne field, not disk) ──────────────────────

def load_gt_boxes_pixels(sample, gt_field: str, img_w: int, img_h: int) -> list:
    """
    Reads detections from sample[gt_field] and returns absolute pixel boxes:
        [{"x1","y1","x2","y2","label"}, ...]

    FiftyOne stores bounding_box as normalised [top_left_x, top_left_y, w, h].
    Whatever label string is on each detection ("dugong", "calf", ...) is
    carried straight through -- this script does not hardcode any class.
    """
    detections_obj = sample[gt_field]
    boxes = []
    if not detections_obj or not detections_obj.detections:
        return boxes

    for det in detections_obj.detections:
        bx, by, bw, bh = det.bounding_box   # normalised [0,1]
        x1 = max(0, int(round(bx * img_w)))
        y1 = max(0, int(round(by * img_h)))
        x2 = min(img_w, int(round((bx + bw) * img_w)))
        y2 = min(img_h, int(round((by + bh) * img_h)))

        if x2 > x1 and y2 > y1:
            boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "label": det.label})

    return boxes


# ── ROI grid visualisation (FiftyOne field, optional) ─────────────────────────

def add_roi_grid(dataset, tile_size: int, overlap: int):
    """Stores the tile grid as a Detections field for visual sanity-checking."""
    import fiftyone as fo

    print(f"\n[ROI grid] tile={tile_size}px overlap={overlap}px ...")
    updated = 0
    for sample in dataset.iter_samples(autosave=True, progress=True):
        if sample.metadata is None or not sample.metadata.width:
            continue
        W, H = sample.metadata.width, sample.metadata.height
        tiles = compute_tiles(W, H, tile_size, overlap)
        rois = [
            fo.Detection(
                label=f"tile_{y0}_{x0}",
                bounding_box=[x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H],
            )
            for (x0, y0, x1, y1) in tiles
        ]
        sample["roi_grid"] = fo.Detections(detections=rois)
        updated += 1
    print(f"  roi_grid added to {updated} samples.")


# ── Per-image tile/label computation (geometry computed ONCE per tile) ───────

def select_tiles_for_image(
    img_w: int,
    img_h: int,
    gt_boxes: list,
    tile_size: int,
    overlap: int,
    min_area_ratio: float,
) -> list:
    """
    Computes, for every tile in the grid, the clipped detection boxes plus
    a single derived "is_positive" bit, both driven by the SAME
    min_area_ratio threshold.

    Returns a list of dicts:
        {
          "x_start", "y_start", "x_end", "y_end",
          "clipped_boxes": [
              {"x1","y1","x2","y2","label","area_ratio"}, ...
              # only boxes with area_ratio >= min_area_ratio are included
          ],
          "is_positive": bool,   # True iff clipped_boxes is non-empty
        }
    """
    tiles  = compute_tiles(img_w, img_h, tile_size, overlap)
    result = []

    for (x_start, y_start, x_end, y_end) in tiles:
        clipped_boxes = []

        for box in gt_boxes:
            ix1 = max(box["x1"], x_start)
            iy1 = max(box["y1"], y_start)
            ix2 = min(box["x2"], x_end)
            iy2 = min(box["y2"], y_end)

            if ix1 >= ix2 or iy1 >= iy2:
                continue   # no overlap

            orig_area  = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])
            inter_area = (ix2 - ix1) * (iy2 - iy1)
            area_ratio = inter_area / orig_area if orig_area > 0 else 0.0

            # SINGLE shared threshold: a box only counts at all (for
            # either task) if it clears min_area_ratio here.
            if area_ratio < min_area_ratio:
                continue

            clipped_boxes.append({
                "x1": ix1, "y1": iy1, "x2": ix2, "y2": iy2,
                "label": box["label"], "area_ratio": area_ratio,
            })

        result.append({
            "x_start": x_start, "y_start": y_start,
            "x_end": x_end, "y_end": y_end,
            "clipped_boxes": clipped_boxes,
            "is_positive": len(clipped_boxes) > 0,
        })

    return result


# ── Box -> YOLO label line ─────────────────────────────────────────────────────

def _box_to_yolo_line(box: dict, tile: dict, tile_size: int, label_to_idx: dict) -> str | None:
    """
    Converts one (already-thresholded) clipped box into a YOLO label line,
    normalised to the tile. Returns None only if the box degenerates after
    clamping (should be rare given it already passed min_area_ratio).
    """
    tile_pw = tile["x_end"] - tile["x_start"]
    tile_ph = tile["y_end"] - tile["y_start"]

    lx1 = max(0, min(tile_pw, box["x1"] - tile["x_start"]))
    ly1 = max(0, min(tile_ph, box["y1"] - tile["y_start"]))
    lx2 = max(0, min(tile_pw, box["x2"] - tile["x_start"]))
    ly2 = max(0, min(tile_ph, box["y2"] - tile["y_start"]))

    if lx2 <= lx1 or ly2 <= ly1:
        return None

    # tile is always EXACTLY tile_size x tile_size (shift-inward rule),
    # so no rescale factor is needed here -- normalise directly.
    nx1, ny1 = lx1 / tile_size, ly1 / tile_size
    nx2, ny2 = lx2 / tile_size, ly2 / tile_size

    cx = max(0.0, min(1.0, (nx1 + nx2) / 2))
    cy = max(0.0, min(1.0, (ny1 + ny2) / 2))
    w  = max(0.0, min(1.0, nx2 - nx1))
    h  = max(0.0, min(1.0, ny2 - ny1))

    if w < 1e-6 or h < 1e-6:
        return None

    class_idx = label_to_idx[box["label"]]
    return f"{class_idx} {cx:.10f} {cy:.10f} {w:.10f} {h:.10f}"


# ── Export one tile ────────────────────────────────────────────────────────────

def export_tile_image(image_filepath: str, tile: dict, tile_size: int,
                       dest_images_dir: Path, final_name: str) -> Path | None:
    """Crops and saves the tile image. Asserts native tile_size x tile_size
    (shift-inward rule guarantees this -- no resize branch needed)."""
    dest_images_dir.mkdir(parents=True, exist_ok=True)
    img_path = dest_images_dir / f"{final_name}.jpg"

    try:
        with Image.open(image_filepath) as img:
            crop = img.crop((tile["x_start"], tile["y_start"],
                            tile["x_end"], tile["y_end"]))
            assert crop.size == (tile_size, tile_size), (
                f"Unexpected crop size {crop.size} for tile at "
                f"({tile['x_start']},{tile['y_start']}) -- shift-inward rule "
                f"should guarantee exact tile_size. Check geometry."
            )
            crop.save(img_path, quality=95)
    except Exception as e:
        print(f"  WARNING: crop failed for {image_filepath}: {e}")
        return None

    return img_path


def export_tile(image_filepath, tile, tile_size, min_area_ratio,
                label_to_idx, output_root, final_name, meta_extra):
    """
    Exports ONE tile to EITHER positive/ or negative/ (never both), writing
    images/, labels/, and metadata/ together. This single export serves
    both detection (via labels/*.txt) and classification (via folder
    membership) without duplicating the image file anywhere.
    """
    branch = "positive" if tile["is_positive"] else "negative"
    dest   = output_root / branch

    img_path = export_tile_image(image_filepath, tile, tile_size,
                                 dest / "images", final_name)
    if img_path is None:
        return False

    # labels/ -- non-empty only for positive tiles; empty .txt for
    # negatives so a YOLO-style loader always finds a label file present.
    lines = []
    if tile["is_positive"]:
        for box in tile["clipped_boxes"]:
            line = _box_to_yolo_line(box, tile, tile_size, label_to_idx)
            if line:
                lines.append(line)

    labels_dir = dest / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / f"{final_name}.txt").write_text("\n".join(lines))

    metadata_dir = dest / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "source_image":    image_filepath,
        "tile_name":       f"tile_{tile['y_start']}_{tile['x_start']}",
        "x_start":         tile["x_start"],
        "y_start":         tile["y_start"],
        "x_end":           tile["x_end"],
        "y_end":           tile["y_end"],
        "is_positive":     tile["is_positive"],
        "min_area_ratio":  min_area_ratio,
        "n_boxes_in_tile": len(tile["clipped_boxes"]),
        "n_boxes_written": len(lines),
        "boxes": [
            {"label": b["label"], "area_ratio": round(b["area_ratio"], 6)}
            for b in tile["clipped_boxes"]
        ],
        **meta_extra,
    }
    (metadata_dir / f"{final_name}.json").write_text(json.dumps(meta, indent=2))
    return True


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    args = get_args()

    os.environ["FIFTYONE_DATABASE_URI"] = f"mongodb://localhost:{args.port}"
    import fiftyone as fo

    try:
        print(f"Connected to MongoDB at localhost:{args.port}. "
              f"Existing datasets: {fo.list_datasets()}")
    except Exception as e:
        print(f"ERROR: Could not connect to MongoDB.\n{e}")
        return

    assert args.dataset in fo.list_datasets(), \
        f"Dataset '{args.dataset}' not found. Available: {fo.list_datasets()}"
    dataset = fo.load_dataset(args.dataset)
    print(f"Loaded dataset '{args.dataset}' -- {len(dataset)} samples.")
    print(f"Reading ground truth from field: '{args.gt_field}'")

    dataset.compute_metadata(skip_failures=True)

    if not args.skip_grid:
        add_roi_grid(dataset, tile_size=args.tile_size, overlap=args.overlap)
    else:
        print("\n[ROI grid] Skipped (--skip-grid).")

    # ── Discover label set for this gt_field, build a stable class index ─────
    print(f"\nScanning '{args.gt_field}' for label set ...")
    all_labels = set()
    for sample in dataset.iter_samples(progress=True):
        det_obj = sample[args.gt_field]
        if det_obj and det_obj.detections:
            for det in det_obj.detections:
                all_labels.add(det.label)
    label_to_idx = {lab: i for i, lab in enumerate(sorted(all_labels))}
    print(f"  Labels found: {label_to_idx}")

    output_root = Path(args.output_dir)
    rng = random.Random(args.seed)

    counts = {"positive": 0, "negative": 0, "skipped": 0}

    print(f"\n[Export] tile={args.tile_size}px overlap={args.overlap}px  "
          f"min_area_ratio={args.min_area_ratio}  "
          f"k_neg={'ALL' if args.k_neg is None else args.k_neg}")

    for sample in dataset.iter_samples(progress=True):
        if sample.metadata is None or not sample.metadata.width:
            counts["skipped"] += 1
            continue

        filepath = sample.filepath
        img_w, img_h = sample.metadata.width, sample.metadata.height
        stem = Path(filepath).stem

        meta_extra = {
            "region":       sample.get_field("region"),
            "mission_name": sample.get_field("mission_name"),
            "parent_name":  sample.get_field("parent_name"),
            "gt_field":     args.gt_field,
        }

        gt_boxes = load_gt_boxes_pixels(sample, args.gt_field, img_w, img_h)
        tiles = select_tiles_for_image(
            img_w, img_h, gt_boxes,
            args.tile_size, args.overlap, args.min_area_ratio,
        )

        pos_tiles = [t for t in tiles if t["is_positive"]]
        neg_tiles = [t for t in tiles if not t["is_positive"]]

        if args.k_neg is not None and len(neg_tiles) > args.k_neg:
            neg_tiles = rng.sample(neg_tiles, k=args.k_neg)

        for t in pos_tiles + neg_tiles:
            final_name = (f"{stem}__tile_{t['y_start']}_{t['x_start']}"
                          f"_t{args.tile_size}_o{args.overlap}")
            ok = export_tile(filepath, t, args.tile_size, args.min_area_ratio,
                            label_to_idx, output_root, final_name, meta_extra)
            if ok:
                counts["positive" if t["is_positive"] else "negative"] += 1

    print(f"\n{'─'*55}")
    print(f"  Positive tiles : {counts['positive']}")
    print(f"  Negative tiles : {counts['negative']}")
    print(f"  Skipped images : {counts['skipped']} (no metadata)")
    print(f"  Output root    : {output_root}")
    print(f"{'─'*55}")
    print("\nAll done mate, drink a tee")
    print("This single tree serves BOTH tasks: detection reads "
          "positive/labels/*.txt directly; classification uses "
          "positive/ vs negative/ folder membership as the binary label.")


if __name__ == "__main__":
    main()