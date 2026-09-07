"""
export_patches_pipeline.py
==========================

End-to-end pipeline that converts a FiftyOne dataset of full aerial images
into a folder of 640x640 patch images ready for object-detection training.

APPROACH
--------
All intermediate geometry is kept in ABSOLUTE PIXEL COORDINATES on the source
image.  Only at the very final step are coordinates converted to normalised
YOLO format [cx, cy, w, h] in [0,1].  This eliminates all compounding
normalisation errors from previous versions.

Ground-truth labels are read directly from the labels_yolo/ folder on disk
(same folder structure as the source images) rather than from FiftyOne fields.
This avoids any FiftyOne coordinate-space confusion entirely.

ROI GRID EDGE-TILE RULE
-----------------------
For images whose width/height is not a multiple of the stride, the last tile
in each row/column is positioned so its RIGHT/BOTTOM edge aligns with the
image edge, and its LEFT/TOP edge is pushed back accordingly:

    x_start = max(0, x_end - tile_size)   where x_end = min(x + tile_size, W)

This means the tile physically covers [x_start, x_start+tile_size] pixels.
The tile is saved with its ACTUAL pixel origin in the filename:
    tile_<y_start>_<x_start>
so the filename always encodes the true top-left pixel position.

The pipeline runs in four sequential stages:

  Stage 1 -- ROI Grid (stored in FiftyOne for visualisation only)
  Stage 2 -- Dugong detection (reads labels_yolo/ directly)
  Stage 3 -- Sample selection (positive + k_neg negatives per image)
  Stage 4 -- Patch export (crop + YOLO label + JSON metadata)

Output folder layout
--------------------
  <output-dir>/
      images/     <stem>__tile_<y>_<x>_<p|n>.jpg
      labels/     <stem>__tile_<y>_<x>_<p|n>.txt
      metadata/   <stem>__tile_<y>_<x>_<p|n>.json

Usage
-----
  python export_patches_pipeline.py \\
      --dataset dugong --port 44123 \\
      --output-dir /share/home/e2406743/dataset/exported_img/seed_42 \\
      --tile-size 640 --overlap 100 --seed 42 --k-neg 2
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
        description="ROI-grid tiling + patch export pipeline (absolute pixel coords).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset",    "-d", required=True,
                   help="Name of the FiftyOne dataset to process.")
    p.add_argument("--port",       default="44123",
                   help="MongoDB port. (default: 44123)")
    p.add_argument("--tile-size",  type=int, default=640,
                   help="Tile size in pixels (square). (default: 640)")
    p.add_argument("--overlap",    type=int, default=100,
                   help="Overlap between adjacent tiles in pixels. (default: 100)")
    p.add_argument("--k-neg",      type=int, default=None,
                   help="Negative tiles to sample per parent image. "
                        "Omit to export ALL negatives.")
    p.add_argument("--seed",       type=int, default=42,
                   help="Random seed for negative sampling. (default: 42)")
    p.add_argument("--output-dir", "-o", required=True,
                   help="Root folder for exported images/, labels/, metadata/.")
    p.add_argument("--skip-grid",  action="store_true",
                   help="Skip Stage 1 (roi_grid already stored in FiftyOne).")
    return p.parse_args()


# ── Label loader ──────────────────────────────────────────────────────────────

def load_gt_boxes_pixels(image_filepath: str, img_w: int, img_h: int) -> list:
    """
    Read the YOLO label file for a source image and return ground-truth boxes
    as absolute pixel coordinates: [[x1, y1, x2, y2], ...].

    The labels_yolo/ folder lives alongside the images/ folder:
        .../mission_name/images/IMG.jpeg  ->  .../mission_name/labels_yolo/IMG.txt

    YOLO format: class cx cy w h  (normalised [0,1] relative to full image)
    Returns list of [x1_px, y1_px, x2_px, y2_px] (integer pixels, clamped).
    """
    image_path = Path(image_filepath)
    label_path = image_path.parent.parent / "labels_yolo" / (image_path.stem + ".txt")

    boxes = []
    if not label_path.exists():
        return boxes

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            # class cx cy w h -- all normalised [0,1]
            cx = float(parts[1]) * img_w
            cy = float(parts[2]) * img_h
            bw = float(parts[3]) * img_w
            bh = float(parts[4]) * img_h

            x1 = max(0, int(round(cx - bw / 2)))
            y1 = max(0, int(round(cy - bh / 2)))
            x2 = min(img_w, int(round(cx + bw / 2)))
            y2 = min(img_h, int(round(cy + bh / 2)))

            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])

    return boxes


# ── ROI grid (pixel coordinates) ─────────────────────────────────────────────

def compute_tiles(img_w: int, img_h: int, tile_size: int, overlap: int) -> list:
    """
    Returns a list of tiles as absolute pixel rectangles:
        [x_start, y_start, x_end, y_end]  (all integers)

    Edge-tile rule: when the stride does not divide the image evenly, the last
    tile in a row/column is shifted LEFT/UP so its right/bottom edge aligns
    with the image boundary, keeping its size exactly tile_size x tile_size.

    x_end   = min(x + tile_size, img_w)
    x_start = max(0, x_end - tile_size)      <- may differ from x at edges
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


# ── Stage 1: store ROI grid in FiftyOne (for visualisation) ──────────────────

def add_roi_grid(dataset, tile_size: int = 640, overlap: int = 100):
    """
    Stores the tile grid in FiftyOne's roi_grid field for visualisation.
    The bounding_box is stored as normalised [x_start/W, y_start/H, tw/W, th/H].
    The label encodes the TRUE pixel origin: tile_{y_start}_{x_start}.
    """
    import fiftyone as fo

    print(f"\n[Stage 1] Adding ROI grid (tile={tile_size}px, overlap={overlap}px) ...")
    updated = 0

    for sample in dataset.iter_samples(autosave=True, progress=True):
        W = sample.metadata.width
        H = sample.metadata.height
        tiles = compute_tiles(W, H, tile_size, overlap)
        rois = []

        for (x_start, y_start, x_end, y_end) in tiles:
            rois.append(fo.Detection(
                label=f"tile_{y_start}_{x_start}",   # TRUE pixel origin
                bounding_box=[
                    x_start / W,
                    y_start / H,
                    (x_end - x_start) / W,
                    (y_end - y_start) / H,
                ],
            ))

        sample["roi_grid"] = fo.Detections(detections=rois)
        updated += 1

    print(f"  ROI grid added to {updated} samples.")


# ── Stage 2+3: select tiles per image ────────────────────────────────────────

def select_tiles_for_image(
    image_filepath: str,
    img_w: int,
    img_h: int,
    tile_size: int,
    overlap: int,
    k_neg: int | None,
    rng: random.Random,
) -> tuple[list, list]:
    """
    For one source image:
      1. Compute all tiles (absolute pixel coords).
      2. Load GT boxes from labels_yolo/ (absolute pixel coords).
      3. For each tile, find which GT boxes overlap and clip them.
      4. Return (positive_tiles, negative_tiles) where each entry is a dict:
            {
              "x_start", "y_start", "x_end", "y_end",   # tile pixel coords
              "dugongs": [                                # list of clipped boxes
                  {"x1","y1","x2","y2"}                  # pixel coords in full img
              ]
            }
      negative_tiles is already sub-sampled to k_neg (or all if k_neg is None).
    """
    tiles    = compute_tiles(img_w, img_h, tile_size, overlap)
    gt_boxes = load_gt_boxes_pixels(image_filepath, img_w, img_h)

    positive = []
    negative = []

    for (x_start, y_start, x_end, y_end) in tiles:
        clipped_dugongs = []

        for (gx1, gy1, gx2, gy2) in gt_boxes:
            # Intersection
            ix1 = max(gx1, x_start)
            iy1 = max(gy1, y_start)
            ix2 = min(gx2, x_end)
            iy2 = min(gy2, y_end)

            if ix1 < ix2 and iy1 < iy2:   # genuine overlap
                clipped_dugongs.append({
                    "x1": ix1, "y1": iy1,
                    "x2": ix2, "y2": iy2,
                })

        tile_info = {
            "x_start": x_start, "y_start": y_start,
            "x_end":   x_end,   "y_end":   y_end,
            "dugongs": clipped_dugongs,
        }

        if clipped_dugongs:
            positive.append(tile_info)
        else:
            negative.append(tile_info)

    # Sub-sample negatives
    if k_neg is not None and len(negative) > k_neg:
        negative = rng.sample(negative, k=k_neg)

    return positive, negative


# ── Stage 4: export one tile ──────────────────────────────────────────────────

def _clip_to_yolo(dugong_px: dict, tile: dict, tile_size: int) -> str | None:
    """
    Convert a clipped dugong bounding box (absolute pixel coords on the source
    image) to YOLO format normalised to the tile.

    Steps (all in pixel space):
      1. Translate box origin to tile-local coords.
      2. Clamp to [0, tile_pixel_w] x [0, tile_pixel_h].
      3. Scale to the RESIZED tile (always tile_size x tile_size).
      4. Normalise to [0, 1].
      5. Convert top-left [x,y,w,h] -> centre [cx,cy,w,h].
    """
    tile_pw = tile["x_end"] - tile["x_start"]   # actual crop width in pixels
    tile_ph = tile["y_end"] - tile["y_start"]   # actual crop height in pixels

    # Translate to tile-local pixel coords
    lx1 = dugong_px["x1"] - tile["x_start"]
    ly1 = dugong_px["y1"] - tile["y_start"]
    lx2 = dugong_px["x2"] - tile["x_start"]
    ly2 = dugong_px["y2"] - tile["y_start"]

    # Clamp to tile bounds
    lx1 = max(0, min(tile_pw, lx1))
    ly1 = max(0, min(tile_ph, ly1))
    lx2 = max(0, min(tile_pw, lx2))
    ly2 = max(0, min(tile_ph, ly2))

    if lx2 <= lx1 or ly2 <= ly1:
        return None   # degenerate box after clamping

    # Scale to resized tile (tile_size x tile_size)
    sx = tile_size / tile_pw
    sy = tile_size / tile_ph

    rx1 = lx1 * sx;  ry1 = ly1 * sy
    rx2 = lx2 * sx;  ry2 = ly2 * sy

    # Normalise to [0, 1]
    nx1 = rx1 / tile_size;  ny1 = ry1 / tile_size
    nx2 = rx2 / tile_size;  ny2 = ry2 / tile_size

    # YOLO centre format
    cx = (nx1 + nx2) / 2
    cy = (ny1 + ny2) / 2
    w  = nx2 - nx1
    h  = ny2 - ny1

    # Final clamp
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w  = max(0.0, min(1.0, w))
    h  = max(0.0, min(1.0, h))

    if w < 1e-6 or h < 1e-6:
        return None

    return f"0 {cx:.10f} {cy:.10f} {w:.10f} {h:.10f}"


def export_tile(
    image_filepath: str,
    tile: dict,
    is_positive: bool,
    output_dir: str,
    tile_size: int,
    region: str | None,
    mission_name: str | None,
    stratify_key: str | None,
) -> bool:
    """
    Crops one tile from the source image, writes image + label + metadata.
    Returns True on success, False on error.
    """
    stem       = Path(image_filepath).stem
    x_start    = tile["x_start"]
    y_start    = tile["y_start"]
    alias      = "p" if is_positive else "n"
    final_name = f"{stem}__tile_{y_start}_{x_start}_{alias}"

    img_path  = Path(output_dir) / "images"   / f"{final_name}.jpg"
    txt_path  = Path(output_dir) / "labels"   / f"{final_name}.txt"
    json_path = Path(output_dir) / "metadata" / f"{final_name}.json"

    for path in [img_path, txt_path, json_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    # ── crop ──────────────────────────────────────────────────────────────────
    try:
        with Image.open(image_filepath) as img:
            crop = img.crop((
                tile["x_start"], tile["y_start"],
                tile["x_end"],   tile["y_end"],
            ))
            # Resize only if needed (edge tiles that are smaller than tile_size)
            if crop.size != (tile_size, tile_size):
                crop = crop.resize((tile_size, tile_size), Image.LANCZOS)
            crop.save(img_path, quality=95)
    except Exception as e:
        print(f"  WARNING: crop failed for {image_filepath}: {e}")
        return False

    # ── YOLO label ────────────────────────────────────────────────────────────
    lines = []
    if is_positive:
        for dug in tile["dugongs"]:
            line = _clip_to_yolo(dug, tile, tile_size)
            if line:
                lines.append(line)

    txt_path.write_text("\n".join(lines))

    # ── metadata JSON ─────────────────────────────────────────────────────────
    meta = {
        "source_image":    image_filepath,
        "tile_name":       f"tile_{y_start}_{x_start}",
        "x_start":         x_start,
        "y_start":         y_start,
        "x_end":           tile["x_end"],
        "y_end":           tile["y_end"],
        "tile_size_px":    (tile["x_end"] - x_start, tile["y_end"] - y_start),
        "region":          region,
        "mission_name":    mission_name,
        "stratify_key":    stratify_key,
        "contains_dugong": "positive" if is_positive else "negative",
        "n_dugongs":       len(tile["dugongs"]),
        "dugong_boxes_px": tile["dugongs"],   # absolute pixel coords for debugging
    }
    json_path.write_text(json.dumps(meta, indent=2))
    return True


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    args = get_args()

    # CRITICAL: must be set before fiftyone is imported
    os.environ["FIFTYONE_DATABASE_URI"] = f"mongodb://localhost:{args.port}"

    import fiftyone as fo
    from fiftyone import ViewField as F

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

    # ── Stage 1: ROI grid (FiftyOne, optional) ────────────────────────────────
    if not args.skip_grid:
        dataset.compute_metadata()
        add_roi_grid(dataset, tile_size=args.tile_size, overlap=args.overlap)
    else:
        print("\n[Stage 1] Skipped (--skip-grid).")

    # ── Stages 2+3: scan all images and select tiles ──────────────────────────
    print(f"\n[Stage 2+3] Scanning images and selecting tiles "
          f"(k_neg={'ALL' if args.k_neg is None else args.k_neg}, "
          f"seed={args.seed}) ...")

    rng = random.Random(args.seed)
    all_positive = []   # list of (filepath, tile_dict, meta_dict)
    all_negative = []

    for sample in dataset.iter_samples(progress=True):
        filepath    = sample.filepath
        img_w       = sample.metadata.width
        img_h       = sample.metadata.height
        region      = getattr(sample, "region",       None)
        mission     = getattr(sample, "mission_name", None)
        strat_key   = getattr(sample, "stratify_key", None)

        pos_tiles, neg_tiles = select_tiles_for_image(
            image_filepath=filepath,
            img_w=img_w,
            img_h=img_h,
            tile_size=args.tile_size,
            overlap=args.overlap,
            k_neg=args.k_neg,
            rng=rng,
        )

        meta = {"region": region, "mission_name": mission, "stratify_key": strat_key}

        for tile in pos_tiles:
            all_positive.append((filepath, tile, meta))
        for tile in neg_tiles:
            all_negative.append((filepath, tile, meta))

    print(f"  Positive tiles : {len(all_positive)}")
    print(f"  Negative tiles : {len(all_negative)}")
    print(f"  Total          : {len(all_positive) + len(all_negative)}")

    # ── Confirmation ──────────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  Output directory : {args.output_dir}")
    print(f"{'─'*55}")
    answer = input("\nProceed with export? [y/N] ").strip().lower()
    if answer not in ("y", "yes"):
        print("Aborted.")
        return

    # ── Stage 4: export ───────────────────────────────────────────────────────
    print(f"\n[Stage 4] Exporting {len(all_positive)+len(all_negative)} patches ...")
    os.makedirs(args.output_dir, exist_ok=True)

    exported = 0
    skipped  = 0

    for (filepath, tile, meta) in all_positive + all_negative:
        is_positive = len(tile["dugongs"]) > 0
        ok = export_tile(
            image_filepath=filepath,
            tile=tile,
            is_positive=is_positive,
            output_dir=args.output_dir,
            tile_size=args.tile_size,
            region=meta["region"],
            mission_name=meta["mission_name"],
            stratify_key=meta["stratify_key"],
        )
        if ok:
            exported += 1
        else:
            skipped += 1

    print(f"\n  Exported : {exported} patches")
    if skipped:
        print(f"  Skipped  : {skipped} (errors)")
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()