"""
create_tile_dataset.py
========================

Loads a tiled dataset produced by tile_export_pipeline.py (the
positive/{images,labels,metadata}/ + negative/{images,labels,metadata}/
folder structure) into a FiftyOne dataset for visual inspection in the App.

For every tile:
  - filepath        -> positive|negative/images/<name>.jpg
  - ground_truth     <- parsed from the matching labels/<name>.txt
                        (YOLO format, normalised [0,1] center coords)
  - type_label       <- "positive" or "negative" (which folder it came from)
  - region, mission_name, parent_name, gt_field, min_area_ratio,
    n_boxes_in_tile, n_boxes_written  <- read from metadata/<name>.json

Usage
-----
    from create_tile_dataset import create_tile_dataset

    dataset = create_tile_dataset(
        path_root="/share/home/e2406743/dataset/tiles_all/nc765",
        name_dataset="nc765_T1024ov224",
    )
"""

import json
from pathlib import Path


def _parse_yolo_label_file(label_path: Path):
    """
    Reads a YOLO-format label file (class cx cy w h, normalised [0,1]) and
    returns a fo.Detections object, or None if the file is missing/empty.
    Class index is converted back to a label string using idx_to_label
    if available in the metadata; falls back to the raw class index as a
    string otherwise.
    """
    import fiftyone as fo

    if not label_path.exists():
        return None

    detections = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_idx = parts[0]
            cx, cy, w, h = (float(v) for v in parts[1:5])

            top_left_x = cx - w / 2
            top_left_y = cy - h / 2

            detections.append(
                fo.Detection(
                    label=class_idx,   # see note below re: idx_to_label
                    bounding_box=[top_left_x, top_left_y, w, h],
                )
            )

    return fo.Detections(detections=detections) if detections else None


def create_tile_dataset(path_root: str, name_dataset: str, overwrite: bool = True):
    """
    Builds (or loads, if it already exists) a FiftyOne dataset from a tiled
    positive/negative folder structure.

    Parameters
    ----------
    path_root    : str  - root folder containing positive/ and negative/
                            subfolders (the --output-dir from
                            tile_export_pipeline.py)
    name_dataset : str  - name for the FiftyOne dataset
    overwrite    : bool - if True and a dataset with this name already
                            exists, it is deleted and rebuilt from scratch.
                            If False and it exists, the existing dataset is
                            loaded and returned unchanged. (default: True)

    Returns
    -------
    fo.Dataset
    """
    import fiftyone as fo

    if name_dataset in fo.list_datasets():
        if not overwrite:
            print(f"Dataset '{name_dataset}' already exists -- loading as-is "
                  f"(overwrite=False).")
            return fo.load_dataset(name_dataset)
        print(f"Dataset '{name_dataset}' already exists -- deleting and rebuilding.")
        fo.delete_dataset(name_dataset)

    dataset = fo.Dataset(name=name_dataset, overwrite=True)
    dataset.persistent = True

    root = Path(path_root)
    samples_to_add = []

    for type_label in ["positive", "negative"]:
        images_dir   = root / type_label / "images"
        labels_dir   = root / type_label / "labels"
        metadata_dir = root / type_label / "metadata"

        if not images_dir.exists():
            print(f"  WARNING: '{images_dir}' does not exist -- skipping '{type_label}'.")
            continue

        image_paths = sorted(images_dir.glob("*.jpg"))
        print(f"  Found {len(image_paths)} images in {type_label}/images/")

        for img_path in image_paths:
            stem = img_path.stem
            sample = fo.Sample(filepath=str(img_path))

            sample["type_label"] = type_label

            # ── Ground truth from labels/<stem>.txt ──────────────────────
            label_path = labels_dir / f"{stem}.txt"
            yolo_labels = _parse_yolo_label_file(label_path)
            if yolo_labels:
                sample["ground_truth"] = yolo_labels

            # ── Metadata from metadata/<stem>.json ───────────────────────
            meta_path = metadata_dir / f"{stem}.json"
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)

                sample["region"]         = meta.get("region")
                sample["mission_name"]   = meta.get("mission_name")
                sample["parent_name"]    = meta.get("parent_name")
                sample["gt_field"]       = meta.get("gt_field")
                sample["min_area_ratio"] = meta.get("min_area_ratio")
                sample["n_boxes_in_tile"]  = meta.get("n_boxes_in_tile")
                sample["n_boxes_written"]  = meta.get("n_boxes_written")
                sample["source_image"]   = meta.get("source_image")
                sample["tile_name"]      = meta.get("tile_name")
                sample["x_start"]        = meta.get("x_start")
                sample["y_start"]        = meta.get("y_start")

            samples_to_add.append(sample)

    print(f"\nAdding {len(samples_to_add)} tile samples to '{name_dataset}' ...")
    dataset.add_samples(samples_to_add, progress=True)
    dataset.compute_metadata()
    dataset.save()

    print(f"Done. Dataset '{name_dataset}' created with {len(dataset)} samples.")
    print(f"  type_label counts: {dataset.count_values('type_label')}")

    return dataset