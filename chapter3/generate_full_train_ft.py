"""
generate_ids_json.py
====================
Export tagged samples from a FiftyOne dataset directly into the resolved
paths JSON consumed by DugongDataModule (no map_ids_to_paths step).

Positive tiles form the core of each split. Optionally, a fraction of
negative (background) tiles is added to train and val to reduce false
positives at inference time.

Structure
---------
{
  "0": {
    "train": {"p100": {"random": {"images": [...], "labels": [...], "metadata": [...]}}},
    "val":  {"images": [...], "labels": [...], "metadata": [...]},
    "test": {"images": [...], "labels": [...], "metadata": [...]}
  },
  ...
}

Usage
-----
  python generate_ids_json.py \
      --dataset my_source_dataset \
      --output_json /path/to/resolved_paths.json \
      --num_seeds 3 \
      --neg_frac 0.1 \
      --port 44123
"""

import os
import json
import random
import argparse
from pathlib import Path


def get_args():
    p = argparse.ArgumentParser(
        description="Export train/val(/test) tile paths per seed to resolved JSON."
    )
    p.add_argument("--dataset", "-d", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--num_seeds", type=int, nargs="+", default=[1])
    p.add_argument("--partition", default="p100")
    p.add_argument("--method", default="random")
    p.add_argument("--port", default="44123")
    return p.parse_args()


def resolve_tile_paths(image_filepaths: list) -> dict:
    """
    Derive label/metadata paths from tile image paths by substitution:
      .../images/STEM.jpg → .../labels/STEM.txt, .../metadata/STEM.json
    Missing label = negative tile (allowed). Missing metadata = empty dict.
    """
    images, labels, metadata = [], [], []

    for img_str in sorted(image_filepaths):
        img  = Path(img_str)
        lbl  = img.parent.parent / "labels"   / f"{img.stem}.txt"
        meta = img.parent.parent / "metadata" / f"{img.stem}.json"

        if not img.exists():
            print(f"  WARNING: image not found, skipping: {img}")
            continue

        meta_dict = {}
        if meta.exists():
            try:
                with open(meta) as f:
                    meta_dict = json.load(f)
            except Exception as e:
                print(f"  WARNING: bad metadata {meta}: {e}")

        images.append(str(img))
        labels.append(str(lbl))
        metadata.append(meta_dict)

    return {"images": images, "labels": labels, "metadata": metadata}



def main():
    args = get_args()

    # MUST happen before importing fiftyone
    os.environ["FIFTYONE_DATABASE_URI"] = f"mongodb://localhost:{args.port}"

    import fiftyone as fo
    from fiftyone import ViewField

    assert args.dataset in fo.list_datasets(), (
        f"Dataset '{args.dataset}' not found. "
        f"Available: {fo.list_datasets()}"
    )
    dataset = fo.load_dataset(args.dataset)
    print(f"Loaded dataset '{args.dataset}' ({len(dataset)} samples)")

    positive_view = dataset.match(ViewField("type_label").contains_str("positive"))

    output = {}

    for seed in args.num_seeds:
        pos_train = positive_view.match_tags(f"train_{seed}")
        pos_val   = positive_view.match_tags(f"test_{seed}")
        pos_test  = positive_view.match_tags(f"test_{seed}")

        assert len(pos_train) > 0, f"No samples tagged 'train_{seed}'."
        assert len(pos_val)   > 0, f"No samples tagged 'val_{seed}'."
        assert len(pos_test)   > 0, f"No samples tagged 'val_{seed}'."

        train_paths = list(pos_train.values("filepath"))
        val_paths   = list(pos_val.values("filepath"))
        test_paths = list(pos_test.values("filepath"))

        print(
            f"Seed {seed} | train={len(train_paths)} "
            f"val={len(val_paths)}"
            f"test={len(test_paths)}"
        )

        output[str(seed)] = {
            "train": {
                args.partition: {
                    args.method: resolve_tile_paths(train_paths)
                }
            },
            "val":  resolve_tile_paths(val_paths),
            "test": resolve_tile_paths(test_paths),
        }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved resolved paths JSON → {output_path}")


if __name__ == "__main__":
    main()