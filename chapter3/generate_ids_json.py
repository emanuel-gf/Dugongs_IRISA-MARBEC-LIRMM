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
    p.add_argument("--num_seeds", type=int, default=1)
    p.add_argument("--partition", default="p100")
    p.add_argument("--method", default="random")
    p.add_argument(
        "--neg_frac", type=float, default=0.0,
        help="Fraction of negative tiles to add to train and val, relative "
             "to the split's positive count. E.g. 0.1 adds 1 negative per "
             "10 positives. 0 = positives only (default)."
    )
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


def sample_negatives(neg_view, n_pos: int, neg_frac: float, rng: random.Random):
    """
    Randomly sample floor(neg_frac * n_pos) negative filepaths from neg_view.
    Caps at the number available and warns if short.
    """
    n_wanted = int(neg_frac * n_pos)
    if n_wanted == 0:
        return []

    neg_paths = neg_view.values("filepath")
    if len(neg_paths) < n_wanted:
        print(f"  WARNING: only {len(neg_paths)} negatives available, "
              f"wanted {n_wanted} — using all.")
        return list(neg_paths)

    return rng.sample(neg_paths, n_wanted)


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
    negative_view = dataset.match(~ViewField("type_label").contains_str("positive"))

    print(f"Positives: {len(positive_view)}  |  Negatives: {len(negative_view)}")

    output = {}

    for seed in range(args.num_seeds):
        rng = random.Random(seed)   # reproducible negative sampling per seed

        pos_train = positive_view.match_tags(f"train_{seed}")
        pos_val   = positive_view.match_tags(f"val_{seed}")
        pos_test  = positive_view.match_tags(f"test_{seed}")

        assert len(pos_train) > 0, f"No samples tagged 'train_{seed}'."
        assert len(pos_val)   > 0, f"No samples tagged 'val_{seed}'."

        train_paths = list(pos_train.values("filepath"))
        val_paths   = list(pos_val.values("filepath"))

        # ── Negatives: sampled from the SAME split's tags (no leakage) ──
        n_neg_train = n_neg_val = 0
        if args.neg_frac > 0:
            neg_train_pool = negative_view.match_tags(f"train_{seed}")
            neg_val_pool   = negative_view.match_tags(f"val_{seed}")

            neg_train = sample_negatives(neg_train_pool, len(train_paths),
                                         args.neg_frac, rng)
            neg_val   = sample_negatives(neg_val_pool, len(val_paths),
                                         args.neg_frac, rng)

            n_neg_train, n_neg_val = len(neg_train), len(neg_val)
            train_paths += neg_train
            val_paths   += neg_val

        # ── Test: from tags if present, else mirror val ──────────────────
        if len(pos_test) > 0:
            test_paths = list(pos_test.values("filepath"))
            if args.neg_frac > 0:
                neg_test_pool = negative_view.match_tags(f"test_{seed}")
                test_paths += sample_negatives(neg_test_pool, len(test_paths),
                                               args.neg_frac, rng)
            test_src = "test tags"
        else:
            test_paths = list(val_paths)   # already includes negatives
            test_src = "copied from val (no test tags found)"

        print(
            f"Seed {seed} | train={len(train_paths)} "
            f"(pos={len(pos_train)}, neg={n_neg_train})  "
            f"val={len(val_paths)} (pos={len(pos_val)}, neg={n_neg_val})  "
            f"test={len(test_paths)} [{test_src}]"
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