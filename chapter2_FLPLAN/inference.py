"""
inference_flplan.py
====================
Run RT-DETR inference over a FiftyOne FLPLAN dataset view and write a single
consolidated JSON file that is directly compatible with reconstruct_from_json()
in reconstruct_flplan_predictions.py.

Output JSON format
------------------
One file per run:  {save_json_path}/{name_run}_test_predictions.json

[
  {
    "filepath":      "/abs/path/to/tile.jpg",
    "detections":    [
        {"label": "dugong", "bounding_box": [cx, cy, w, h], "confidence": 0.82},
        ...
    ],
    "tile_metadata": {"x_start": ..., "y_start": ..., "tile_w": ..., "tile_h": ...}
  },
  ...
]

Bounding boxes are in RT-DETR native format [cx, cy, w, h] normalised to the
tile dimensions — this matches what the model produces and what
reconstruct_from_json() expects.

Usage
-----
# Zero-shot baseline (no fine-tuning)
python inference_flplan.py \\
    --dataset        FLPLAN \\
    --port           44123 \\
    --view           test_0 \\
    --checkpoint-dir /share/.../checkpoints/NNN_NC_SEED63_augm_0510_1843/hf_export \\
    --tile-folder    /share/.../dataset/exported_img/new_dataset_flplan200/images \\
    --save-json-path /share/.../inference_flplan \\
    --name-run       baseline_seed0 \\
    --confidence     0.05

# Fine-tuned model
python inference_flplan.py \\
    --dataset        FLPLAN \\
    --view           test_0 \\
    --checkpoint-dir /share/.../checkpoints_flplan/NWW_p10_aclr_seed0_.../hf_export \\
    --tile-folder    /share/.../dataset/exported_img/new_dataset_flplan200/images \\
    --save-json-path /share/.../inference_flplan \\
    --name-run       NWW_p10_aclr_seed0_rtdetr_0607_1200

Notes
-----
- The script collects tile images by expanding each full-image filepath from
  the FiftyOne view into its corresponding tiles found in --tile-folder.
- Tile discovery uses glob on the full-image stem:
      {tile_folder}/{stem}__tile_*.jpg
- If --view is a tag name (e.g. 'test_0'), samples are filtered with
  dataset.match_tags(view). If --view is a saved view name, pass
  --view-type saved to use dataset.load_saved_view(view) instead.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor


# ─────────────────────────────────────────────────────────────────────────────
# Tile helpers
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def get_tile_metadata(
    filepath:  str | Path,
    img_w:     int,
    img_h:     int,
    tile_size: int = 640,
) -> dict | None:
    """
    Recompute tile pixel extent from the tile filename and image dimensions.

    Tile filename format:  {sample_stem}__tile_{y_stride}_{x_stride}_{type}.jpg

    Returns dict with x_start, y_start, tile_w, tile_h — or None if the
    filename does not match the expected pattern.
    """
    match = re.search(r"__tile_(\d+)_(\d+)", str(filepath))
    if not match:
        return None

    y_stride = int(match.group(1))
    x_stride = int(match.group(2))

    x_end   = min(x_stride + tile_size, img_w)
    y_end   = min(y_stride + tile_size, img_h)
    x_start = max(x_end - tile_size, 0)
    y_start = max(y_end - tile_size, 0)

    return {
        "x_start": x_start,
        "y_start": y_start,
        "tile_w":  x_end - x_start,
        "tile_h":  y_end - y_start,
    }


def expand_to_tiles(
    full_image_filepaths: list[str | Path],
    tile_folder:          str | Path,
    extensions:           set[str] = SUPPORTED_EXTENSIONS,
    verbose:              bool = True,
) -> list[Path]:
    """
    Expand a list of full-image filepaths into their corresponding tile paths
    found in tile_folder.

    For each full image with stem S, collects all files matching:
        {tile_folder}/{S}__tile_*.{ext}

    Parameters
    ----------
    full_image_filepaths : list of full-image paths from FiftyOne
    tile_folder          : directory containing the tile images
    extensions           : allowed file extensions
    verbose              : print summary stats

    Returns
    -------
    Sorted list of tile Path objects.
    """
    tile_folder = Path(tile_folder)
    if not tile_folder.exists():
        raise FileNotFoundError(f"tile_folder not found: {tile_folder}")

    tile_paths: list[Path] = []
    n_missing = 0

    for fp in full_image_filepaths:
        stem    = Path(fp).stem
        pattern = str(tile_folder / f"{stem}__tile_*")
        matches = [
            Path(p) for p in glob.glob(pattern)
            if Path(p).suffix.lower() in extensions
        ]

        if not matches:
            if verbose:
                print(f"  WARN no tiles found for stem '{stem}'")
            n_missing += 1
            continue

        tile_paths.extend(matches)

    tile_paths = sorted(set(tile_paths))

    if verbose:
        print(
            f"  expand_to_tiles: {len(full_image_filepaths)} full images → "
            f"{len(tile_paths)} tiles  "
            f"({n_missing} full images had no tiles)"
        )

    return tile_paths


# ─────────────────────────────────────────────────────────────────────────────
# Dataset view loader
# ─────────────────────────────────────────────────────────────────────────────

def load_view_filepaths(
    dataset_name: str,
    view:         str,
    view_type:    str = "tag",
    port:         int = 44123,
    verbose:      bool = True,
) -> list[str]:
    """
    Load a FiftyOne dataset and return the filepaths of the requested view.

    Parameters
    ----------
    dataset_name : FiftyOne dataset name (e.g. 'FLPLAN')
    view         : tag name (view_type='tag') or saved view name
                   (view_type='saved')
    view_type    : 'tag' — dataset.match_tags(view)
                   'saved' — dataset.load_saved_view(view)
    port         : MongoDB port (default 44123)
    verbose      : print sample count

    Returns
    -------
    list of filepath strings
    """
    os.environ["FIFTYONE_DATABASE_URI"] = f"mongodb://localhost:{port}"

    import fiftyone as fo

    assert dataset_name in fo.list_datasets(), (
        f"Dataset '{dataset_name}' not found. "
        f"Available: {fo.list_datasets()}"
    )

    dataset = fo.load_dataset(dataset_name)

    if view_type == "tag":
        ds_view = dataset.match_tags(view)
    elif view_type == "saved":
        ds_view = dataset.load_saved_view(view)
    else:
        raise ValueError(
            f"view_type='{view_type}' not supported. Choose 'tag' or 'saved'."
        )

    filepaths = ds_view.values("filepath")

    if verbose:
        print(
            f"  Dataset '{dataset_name}'  view='{view}' ({view_type})  "
            f"→ {len(filepaths)} samples"
        )

    return filepaths


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_model(
    checkpoint_dir: str | Path,
    device:         str | torch.device | None = None,
) -> tuple[RTDetrForObjectDetection, RTDetrImageProcessor, dict]:
    """
    Load an RT-DETR model and processor from a local hf_export/ checkpoint.

    Returns
    -------
    model     : in eval mode on device
    processor : RTDetrImageProcessor
    id2label  : {int: str}
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

    model     = RTDetrForObjectDetection.from_pretrained(str(checkpoint_dir))
    processor = RTDetrImageProcessor.from_pretrained(str(checkpoint_dir))
    model     = model.to(device).eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    print(f"  Model loaded from '{checkpoint_dir.name}'")
    print(f"  Labels: {id2label}")
    print(f"  Device: {device}")

    return model, processor, id2label


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    tile_filepaths:       list[str | Path],
    model:                RTDetrForObjectDetection,
    processor:            RTDetrImageProcessor,
    id2label:             dict[int, str],
    confidence_threshold: float = 0.05,
    tile_size:            int   = 640,
    device:               str | torch.device | None = None,
    verbose:              bool  = True,
) -> list[dict]:
    """
    Run inference on a list of tile image paths.

    Bounding boxes are converted from absolute xyxy (torchvision output)
    back to RT-DETR native format [cx, cy, w, h] normalised to the tile,
    which is what reconstruct_from_json() expects.

    Parameters
    ----------
    tile_filepaths       : list of tile image paths
    model / processor    : loaded RT-DETR model and processor
    id2label             : label mapping (only label_id=0 / 'dugong' used)
    confidence_threshold : discard detections below this score (default 0.05)
    tile_size            : nominal tile size in pixels (default 640)
    device               : inference device
    verbose              : print progress every 200 tiles

    Returns
    -------
    list of dicts — one per tile — in the consolidated JSON format:
        {filepath, detections: [{label, bounding_box: [cx,cy,w,h], confidence}],
         tile_metadata: {x_start, y_start, tile_w, tile_h}}
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    results: list[dict] = []
    n      = len(tile_filepaths)
    n_dets = 0

    print(f"  Running inference on {n} tiles  (threshold={confidence_threshold})")

    for i, path in enumerate(tile_filepaths):
        path         = Path(path)
        image        = Image.open(path).convert("RGB")
        w_img, h_img = image.size

        inputs  = processor(images=image, return_tensors="pt")
        outputs = model(pixel_values=inputs["pixel_values"].to(device))

        preds = processor.post_process_object_detection(
            outputs,
            target_sizes=[(h_img, w_img)],
            threshold=confidence_threshold,
        )[0]

        detections = []
        for score, label_id, box in zip(
            preds["scores"],
            preds["labels"],
            [b.tolist() for b in preds["boxes"]],
        ):
            if label_id.item() != 0:
                continue

            x1, y1, x2, y2 = box

            # Convert absolute xyxy → normalised [cx, cy, w, h] (RT-DETR format)
            cx = (x1 + x2) / 2 / w_img
            cy = (y1 + y2) / 2 / h_img
            bw = (x2 - x1) / w_img
            bh = (y2 - y1) / h_img

            detections.append({
                "label":        "dugong",
                "bounding_box": [
                    round(cx, 8),
                    round(cy, 8),
                    round(bw, 8),
                    round(bh, 8),
                ],
                "confidence": round(score.item(), 6),
            })

        n_dets += len(detections)

        record: dict = {
            "filepath":   str(path.resolve()),
            "detections": detections,
        }

        tile_meta = get_tile_metadata(path, w_img, h_img, tile_size)
        if tile_meta:
            record["tile_metadata"] = tile_meta

        results.append(record)

        if verbose and (i + 1) % 200 == 0:
            print(f"    {i+1}/{n} tiles processed  ({n_dets} detections so far)")

    print(
        f"  Inference complete — {n} tiles  "
        f"{n_dets} detections  "
        f"mean {n_dets/max(n,1):.2f} det/tile"
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run RT-DETR inference on a FLPLAN FiftyOne view.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── dataset ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--dataset", "-d",
        default="FLPLAN",
        help="FiftyOne dataset name (default: FLPLAN)",
    )
    p.add_argument(
        "--port",
        type=int, default=44123,
        help="MongoDB port for FiftyOne (default: 44123)",
    )
    p.add_argument(
        "--view", "-v",
        required=True,
        help="Tag name or saved-view name identifying the test set, "
             "e.g. 'test_0'",
    )
    p.add_argument(
        "--view-type",
        default="tag",
        choices=["tag", "saved"],
        help="How to interpret --view: 'tag' (match_tags) or "
             "'saved' (load_saved_view). Default: tag",
    )

    # ── model ─────────────────────────────────────────────────────────────────
    p.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Path to hf_export/ directory from a training run",
    )
    p.add_argument(
        "--device",
        default=None,
        help="'cuda', 'cpu', or blank for auto-detect",
    )
    p.add_argument(
        "--confidence",
        type=float, default=0.05,
        help="Detection confidence threshold (default: 0.05)",
    )
    p.add_argument(
        "--tile-size",
        type=int, default=640,
        help="Tile size used during tiling (default: 640)",
    )

    # ── data ──────────────────────────────────────────────────────────────────
    p.add_argument(
        "--tile-folder",
        required=True,
        help="Folder containing the tile .jpg images, e.g. "
             "/share/.../new_dataset_flplan200/images",
    )

    # ── output ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--save-json-path",
        required=True,
        help="Directory where the consolidated output JSON is saved",
    )
    p.add_argument(
        "--name-run",
        required=True,
        help="Run name used as the JSON filename prefix, e.g. "
             "'baseline_seed0' or 'NWW_p10_aclr_seed0_rtdetr_0607_1200'. "
             "Output file: {save_json_path}/{name_run}_test_predictions.json",
    )

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print("=" * 60)
    print(f"  dataset      : {args.dataset}")
    print(f"  view         : {args.view}  ({args.view_type})")
    print(f"  checkpoint   : {args.checkpoint_dir}")
    print(f"  tile_folder  : {args.tile_folder}")
    print(f"  name_run     : {args.name_run}")
    print(f"  confidence   : {args.confidence}")
    print("=" * 60)

    # ── 1. Collect full-image filepaths from FiftyOne ─────────────────────────
    print("\n[1/4] Loading dataset view ...")
    full_image_paths = load_view_filepaths(
        dataset_name = args.dataset,
        view         = args.view,
        view_type    = args.view_type,
        port         = args.port,
        verbose      = True,
    )

    # ── 2. Expand to tile paths ───────────────────────────────────────────────
    print("\n[2/4] Expanding full images → tiles ...")
    tile_paths = expand_to_tiles(
        full_image_filepaths = full_image_paths,
        tile_folder          = args.tile_folder,
        verbose              = True,
    )

    if not tile_paths:
        raise RuntimeError(
            "No tile images found. Check --tile-folder and that the dataset "
            "view contains samples whose stems match tile filenames."
        )

    # ── 3. Load model ─────────────────────────────────────────────────────────
    print("\n[3/4] Loading model ...")
    model, processor, id2label = load_model(
        checkpoint_dir = args.checkpoint_dir,
        device         = args.device,
    )

    # ── 4. Run inference ──────────────────────────────────────────────────────
    print("\n[4/4] Running inference ...")
    results = run_inference(
        tile_filepaths       = tile_paths,
        model                = model,
        processor            = processor,
        id2label             = id2label,
        confidence_threshold = args.confidence,
        tile_size            = args.tile_size,
        device               = args.device,
        verbose              = True,
    )

    # ── Save consolidated JSON ────────────────────────────────────────────────
    out_dir  = Path(args.save_json_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"{args.name_run}_test_predictions.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Saved → {out_file}")
    print(f"  Entries: {len(results)}")
    print(f"  Total detections: {sum(len(r['detections']) for r in results)}")
    print("Done.")


if __name__ == "__main__":
    main()