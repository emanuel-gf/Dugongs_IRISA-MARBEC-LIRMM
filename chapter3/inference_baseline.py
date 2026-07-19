"""
inference_tiles.py
==================
Run RT-DETR inference over a FiftyOne dataset view whose samples ARE tile
images (no full-image → tile expansion), and write a consolidated JSON
compatible with reconstruct_from_json() and with the training pipeline's
test-predictions format.

Output
------
{save_json_path}/{name_run}_test_predictions.json

[
  {
    "filepath":   "/abs/path/to/tile.jpg",
    "detections": [
        {"label": "dugong", "bounding_box": [cx, cy, w, h], "confidence": 0.82},
        ...
    ],
    "tile_metadata": {"x_start": ..., "y_start": ..., ...}
  },
  ...
]

Boxes are [cx, cy, w, h] normalised to tile dimensions (RT-DETR native).

Usage
-----
# Zero-shot NC baseline on a fold's test tiles
python inference_tiles.py \
    --dataset wp_final_test \
    --port 44123 \
    --tags test_2 \
    --checkpoint-dir /share/.../NNN_p100_random_seed0_rtdetr_0709_1910/hf_export \
    --save-json-path /share/.../inference_zeroshot \
    --name-run nc_baseline_fold2 \
    --image-size 1024 \
    --confidence 0.05 \
    --batch-size 8
"""

import os
import json
import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection


def get_args():
    p = argparse.ArgumentParser(
        description="RT-DETR inference on tile-level FiftyOne samples."
    )
    p.add_argument("--dataset", "-d", required=True)
    p.add_argument("--port", default="44123")
    p.add_argument("--tags", nargs="+", default=None,
                   help="Filter samples by tag(s), e.g. test_2. "
                        "Omit to run on the whole dataset.")
    p.add_argument("--checkpoint-dir", required=True,
                   help="hf_export dir (save_pretrained output).")
    p.add_argument("--save-json-path", required=True)
    p.add_argument("--name-run", required=True)
    p.add_argument("--image-size", type=int, default=1024,
                   help="MUST match training resolution (default 1024).")
    p.add_argument("--confidence", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--label", default="dugong")
    return p.parse_args()


def load_tile_metadata(img_path: Path) -> dict:
    """.../images/STEM.jpg → .../metadata/STEM.json (empty dict if absent)."""
    meta = img_path.parent.parent / "metadata" / f"{img_path.stem}.json"
    if meta.exists():
        try:
            with open(meta) as f:
                return json.load(f)
        except Exception as e:
            print(f"  WARNING: bad metadata {meta}: {e}")
    return {}


@torch.inference_mode()
def run_inference(model, processor, filepaths, args):
    results = []
    n = len(filepaths)

    for start in range(0, n, args.batch_size):
        batch_paths = filepaths[start:start + args.batch_size]
        images, valid_paths = [], []

        for fp in batch_paths:
            try:
                images.append(Image.open(fp).convert("RGB"))
                valid_paths.append(fp)
            except Exception as e:
                print(f"  WARNING: cannot read {fp}: {e}")

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt").to(args.device)

        with torch.autocast(device_type=args.device.split(":")[0],
                            dtype=torch.float16,
                            enabled=args.device.startswith("cuda")):
            outputs = model(**inputs)

        # Post-process per image at its ORIGINAL tile size
        target_sizes = torch.tensor(
            [img.size[::-1] for img in images], device=args.device
        )  # (h, w)
        processed = processor.post_process_object_detection(
            outputs, threshold=args.confidence, target_sizes=target_sizes
        )

        for fp, img, det in zip(valid_paths, images, processed):
            w, h = img.size
            detections = []
            for score, box in zip(det["scores"], det["boxes"]):
                # boxes come back as absolute xyxy → convert to normalised cxcywh
                x0, y0, x1, y1 = box.tolist()
                cx = ((x0 + x1) / 2) / w
                cy = ((y0 + y1) / 2) / h
                bw = (x1 - x0) / w
                bh = (y1 - y0) / h
                detections.append({
                    "label": args.label,
                    "bounding_box": [cx, cy, bw, bh],
                    "confidence": float(score),
                })

            results.append({
                "filepath": fp,
                "detections": detections,
                "tile_metadata": load_tile_metadata(Path(fp)),
            })

        done = min(start + args.batch_size, n)
        print(f"  [{done}/{n}] tiles processed", end="\r")

    print()
    return results


def main():
    args = get_args()

    os.environ["FIFTYONE_DATABASE_URI"] = f"mongodb://localhost:{args.port}"
    import fiftyone as fo

    # ── Dataset view (tiles directly — no stem expansion) ─────────────────
    assert args.dataset in fo.list_datasets(), (
        f"Dataset '{args.dataset}' not found. Available: {fo.list_datasets()}"
    )
    dataset = fo.load_dataset(args.dataset)
    view = dataset.match_tags(args.tags) if args.tags else dataset
    filepaths = view.values("filepath")
    print(f"Dataset '{args.dataset}' | tags={args.tags} → {len(filepaths)} tiles")
    assert len(filepaths) > 0, "View is empty — check the tag names."

    # ── Model + processor (size override = training consistency) ─────────
    ckpt = Path(args.checkpoint_dir)
    assert ckpt.is_dir(), f"Checkpoint dir not found: {ckpt}"

    processor = AutoImageProcessor.from_pretrained(
        str(ckpt),
        size={"height": args.image_size, "width": args.image_size},
    )
    model = AutoModelForObjectDetection.from_pretrained(str(ckpt))
    model.to(args.device).eval()
    print(f"Model loaded from {ckpt} | inference at "
          f"{args.image_size}x{args.image_size} | device={args.device}")

    # ── Run ───────────────────────────────────────────────────────────────
    results = run_inference(model, processor, filepaths, args)

    n_det = sum(len(r["detections"]) for r in results)
    print(f"Done: {len(results)} tiles, {n_det} detections "
          f"(conf ≥ {args.confidence})")

    # ── Save ──────────────────────────────────────────────────────────────
    out_dir = Path(args.save_json_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.name_run}_test_predictions.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {out_file}")


if __name__ == "__main__":
    main()