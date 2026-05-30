"""
Reconstructs a FiftyOne dataset from exported 640x640 patches.

Expected folder structure:
    <patches_root>/
        images/    *.jpg
        labels/    *.txt   (YOLO format, empty for negatives)
        metadata/  *.json

Each triplet shares the same stem:
    {source_stem}__{tile_name}_{p|n}

Usage:
    python import_patches_dataset.py \
        --patches-root /share/home/e2406743/dataset/exported_img/seed_42 \
        --port 44123 \
        --name dugong_patches_seed42
"""

import os
import json
import argparse
from pathlib import Path


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(
        description="Import exported 640x640 patches into FiftyOne."
    )
    parser.add_argument(
        "--patches-root", "-r", required=True,
        help="Root folder containing images/, labels/, metadata/ subfolders.",
    )
    parser.add_argument(
        "--port", default="44123",
        help="MongoDB port of the running FiftyOne instance. (default: 44123)",
    )
    parser.add_argument(
        "--name", "-n", default="dugong_patches",
        help="FiftyOne dataset name to create or load.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", default=False,
        help="Delete and recreate the dataset if it already exists.",
    )
    parser.add_argument(
        "--launch", "-l", action="store_true", default=False,
        help="Launch the FiftyOne App after import.",
    )
    return parser.parse_args()


# ── YOLO helpers ──────────────────────────────────────────────────────────────

def parse_yolo_label(txt_path: Path, fo):
    """
    Reads a YOLO .txt file and returns a list of fo.Detection objects.
    Returns an empty list for blank (negative) files.

    YOLO format: class cx cy w h  (all normalised [0,1])
    FiftyOne bbox: [top_left_x, top_left_y, w, h]
    """
    detections = []
    if not txt_path.exists():
        return detections

    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            # YOLO center → FiftyOne top-left
            x = cx - w / 2
            y = cy - h / 2
            label = "Dugong" if cls_id == 0 else str(cls_id)
            detections.append(
                fo.Detection(label=label, bounding_box=[x, y, w, h])
            )
    return detections


# ── Main import ───────────────────────────────────────────────────────────────

def build_dataset(patches_root: str, name: str, overwrite: bool, fo, F):
    images_dir   = Path(patches_root) / "images"
    labels_dir   = Path(patches_root) / "labels"
    metadata_dir = Path(patches_root) / "metadata"

    assert images_dir.exists(),   f"images/ folder not found: {images_dir}"
    assert labels_dir.exists(),   f"labels/ folder not found: {labels_dir}"
    assert metadata_dir.exists(), f"metadata/ folder not found: {metadata_dir}"

    # ── dataset creation / loading ────────────────────────────────────────────
    if name in fo.list_datasets():
        if overwrite:
            fo.delete_dataset(name)
            print(f"Deleted existing dataset '{name}'.")
        else:
            print(f"Dataset '{name}' already exists — loading.")
            return fo.load_dataset(name)

    dataset = fo.Dataset(name=name)
    dataset.persistent = True
    print(f"Created dataset '{name}'.")

    # ── iterate over all patch images ─────────────────────────────────────────
    image_paths = sorted(images_dir.glob("*.jpg"))
    print(f"Found {len(image_paths)} patch images.")

    samples      = []
    missing_meta  = 0
    missing_label = 0

    for img_path in image_paths:
        stem      = img_path.stem
        txt_path  = labels_dir   / f"{stem}.txt"
        json_path = metadata_dir / f"{stem}.json"

        # ── load metadata ─────────────────────────────────────────────────────
        meta = {}
        if json_path.exists():
            with open(json_path) as f:
                meta = json.load(f)
        else:
            missing_meta += 1

        # ── parse YOLO labels ─────────────────────────────────────────────────
        detections = parse_yolo_label(txt_path, fo)
        if not txt_path.exists():
            missing_label += 1

        # ── build sample ──────────────────────────────────────────────────────
        sample = fo.Sample(filepath=str(img_path))

        sample["ground_truth"]   = fo.Detections(detections=detections)
        sample["contains_dugong"] = len(detections) > 0

        # restore all metadata fields from JSON
        for field in [
            "region", "subregion", "mission_name",
            "stratify_key", "tile_name", "source_image",
            "tile_id", "sample_id", "dugong_id",
        ]:
            val = meta.get(field)
            if val is not None:
                sample[field] = val

        # keep the raw "positive"/"negative" string as well
        if "contains_dugong" in meta:
            sample["contains_dugong_str"] = meta["contains_dugong"]

        samples.append(sample)

    # ── batch add ─────────────────────────────────────────────────────────────
    print("Adding samples to dataset …")
    dataset.add_samples(samples, progress=True)
    dataset.compute_metadata()
    dataset.save()

    print(f"\nDone. {len(samples)} samples added.")
    if missing_meta:
        print(f"  Warning: {missing_meta} samples had no .json metadata file.")
    if missing_label:
        print(f"  Warning: {missing_label} samples had no .txt label file.")

    # ── summary stats ─────────────────────────────────────────────────────────
    n_pos = dataset.match(F("contains_dugong") == True).count()
    n_neg = dataset.match(F("contains_dugong") == False).count()
    print(f"\n  Positive patches (dugong present): {n_pos}")
    print(f"  Negative patches (no dugong):      {n_neg}")
    print(f"  Total:                             {len(dataset)}")

    return dataset


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = get_args()

    # CRITICAL: must be set before fiftyone is imported
    os.environ["FIFTYONE_DATABASE_URI"] = f"mongodb://localhost:{args.port}"
    print(f"Connecting to MongoDB at localhost:{args.port} …")

    # Import fiftyone here — AFTER the env var is set
    import fiftyone as fo
    from fiftyone import ViewField as F

    # Verify connection
    try:
        existing = fo.list_datasets()
        print(f"Connected. Existing datasets: {existing}")
    except Exception as e:
        print(f"ERROR: Could not connect to MongoDB at localhost:{args.port}.\n{e}")
        return

    dataset = build_dataset(
        patches_root=args.patches_root,
        name=args.name,
        overwrite=args.overwrite,
        fo=fo,
        F=F,
    )

    # ── save convenience views ────────────────────────────────────────────────
    pos_view = dataset.match(F("contains_dugong") == True)
    neg_view = dataset.match(F("contains_dugong") == False)
    dataset.save_view("positive_patches", pos_view)
    dataset.save_view("negative_patches", neg_view)
    print(f"  Saved views: positive_patches ({len(pos_view)}), "
          f"negative_patches ({len(neg_view)})")

    # ── per-region views if region field exists ───────────────────────────────
    if "region" in dataset.get_field_schema():
        for region in dataset.distinct("region"):
            v = dataset.match(F("region") == region)
            dataset.save_view(f"region_{region}", v)
            print(f"  Saved view 'region_{region}' ({len(v)} samples)")

    if args.launch:
        session = fo.launch_app(dataset, remote=True)
        session.wait()


if __name__ == "__main__":
    main()