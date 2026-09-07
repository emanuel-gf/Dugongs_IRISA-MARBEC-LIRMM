"""
add_negatives_json.py
=====================
Step 2 of 2: enrich a merged resolved-paths JSON with negative (background)
tiles drawn from a (possibly different) FiftyOne dataset.

Design
------
- Negative count per split = ceil(neg_frac * n_positives_in_split).
  E.g. p5 train with 20 tiles and --neg_frac 0.2 → +4 negatives.
- The SAME negatives are added to all methods within a partition, so
  methods differ only in their positive selection.
- Val and test negatives are sampled FIRST and excluded from all train
  pools — no background tile appears in both train and evaluation.
- Sampling is seeded → re-running produces the same enriched JSON.

Usage
-----
  python add_negatives_json.py \
      --input_json  merged_methods.json \
      --output_json merged_methods_neg20.json \
      --neg_dataset my_negatives_dataset \
      --neg_frac 0.2 \
      --rng_seed 0 \
      --port 44123
"""

import os
import json
import math
import random
import argparse
from pathlib import Path


def get_args():
    p = argparse.ArgumentParser(
        description="Add negative tiles to a merged resolved-paths JSON."
    )
    p.add_argument("--input_json", required=True,
                   help="Merged resolved paths JSON (output of merge step).")
    p.add_argument("--output_json", required=True,
                   help="Where to save the enriched JSON.")
    p.add_argument("--neg_dataset", required=True,
                   help="FiftyOne dataset containing negative tiles.")
    p.add_argument("--neg_frac", type=float, required=True,
                   help="Negatives per split as a fraction of its positive "
                        "count, e.g. 0.2 = 1 negative per 5 positives.")
    p.add_argument("--rng_seed", type=int, default=0,
                   help="Seed for reproducible negative sampling.")
    p.add_argument("--port", default="44123")
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────

def resolve_tile_paths(image_filepaths: list, context: str) -> dict:
    """.../images/STEM.jpg → labels/STEM.txt + metadata/STEM.json (loaded)."""
    images, labels, metadata = [], [], []
    for img_str in sorted(image_filepaths):
        img = Path(img_str)
        if not img.exists():
            print(f"  WARNING [{context}]: image not found, skipping: {img}")
            continue
        lbl  = img.parent.parent / "labels"   / f"{img.stem}.txt"
        meta = img.parent.parent / "metadata" / f"{img.stem}.json"

        meta_dict = {}
        if meta.exists():
            try:
                with open(meta) as f:
                    meta_dict = json.load(f)
            except Exception as e:
                print(f"  WARNING [{context}]: bad metadata {meta}: {e}")

        images.append(str(img))
        labels.append(str(lbl))
        metadata.append(meta_dict)
    return {"images": images, "labels": labels, "metadata": metadata}


def n_negatives(n_pos: int, neg_frac: float) -> int:
    """ceil so small partitions get at least 1 negative when frac > 0."""
    return math.ceil(neg_frac * n_pos)


def merge_split(split: dict, neg_resolved: dict) -> dict:
    """Append negatives to a split and re-sort jointly by image path."""
    triples = list(zip(
        split["images"]        + neg_resolved["images"],
        split["labels"]        + neg_resolved["labels"],
        split["metadata"]      + neg_resolved["metadata"],
    ))
    triples.sort(key=lambda t: t[0])
    images, labels, metadata = map(list, zip(*triples)) if triples else ([], [], [])
    return {"images": images, "labels": labels, "metadata": metadata}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    os.environ["FIFTYONE_DATABASE_URI"] = f"mongodb://localhost:{args.port}"
    import fiftyone as fo
    from fiftyone import ViewField

    # ── Load inputs ───────────────────────────────────────────────────────
    input_path = Path(args.input_json)
    assert input_path.is_file(), f"Not found: {input_path}"
    with open(input_path) as f:
        data = json.load(f)

    assert args.neg_dataset in fo.list_datasets(), (
        f"Dataset '{args.neg_dataset}' not found. "
        f"Available: {fo.list_datasets()}"
    )
    neg_dataset = fo.load_dataset(args.neg_dataset)

    # ── Negative pool via type_label ──────────────────────────────────────
    negative_view = neg_dataset.match(
        ViewField("type_label").contains_str("negative")
    )
    neg_pool = list(negative_view.values("filepath"))
    print(f"Negative pool: {len(neg_pool)} tiles "
          f"(from '{args.neg_dataset}', {len(neg_dataset)} samples)")
    assert len(neg_pool) > 0, "No negative tiles found via type_label."

    rng = random.Random(args.rng_seed)

    # ── Enrich each seed ──────────────────────────────────────────────────
    for seed, seed_dict in data.items():
        print(f"\n── Seed {seed} ─────────────────────────────────")
        available = set(neg_pool)

        def draw(n: int, context: str) -> list:
            """Sample n negatives from the remaining pool, without replacement."""
            pool = sorted(available)          # sorted → deterministic given rng
            if len(pool) < n:
                print(f"  WARNING [{context}]: only {len(pool)} negatives "
                      f"left, wanted {n} — using all.")
                n = len(pool)
            picked = rng.sample(pool, n)
            available.difference_update(picked)
            return picked

        # ── 1. Val / test first (excluded from all train pools) ──────────
        for split_name in ("val", "test"):
            split = seed_dict[split_name]
            n_pos = len(split["images"])
            n_neg = n_negatives(n_pos, args.neg_frac)
            picked = draw(n_neg, split_name)
            resolved = resolve_tile_paths(picked, split_name)
            seed_dict[split_name] = merge_split(split, resolved)
            print(f"  {split_name:<5}: {n_pos} pos + {len(resolved['images'])} neg "
                  f"= {len(seed_dict[split_name]['images'])}")

        # ── 2. Train: same negatives for every method in a partition ─────
        def _pkey(p):
            return int(p.lstrip("p")) if p.lstrip("p").isdigit() else 999

        for partition in sorted(seed_dict["train"].keys(), key=_pkey):
            part_dict = seed_dict["train"][partition]

            # positive count should match across methods; use max to be safe
            n_pos = max(len(m["images"]) for m in part_dict.values())
            n_neg = n_negatives(n_pos, args.neg_frac)

            picked   = draw(n_neg, f"train/{partition}")
            resolved = resolve_tile_paths(picked, f"train/{partition}")

            for method in part_dict:
                part_dict[method] = merge_split(part_dict[method], resolved)

            counts = {m: len(part_dict[m]["images"]) for m in sorted(part_dict)}
            print(f"  {partition:<5}: +{len(resolved['images'])} neg → {counts}")

    # ── Write ─────────────────────────────────────────────────────────────
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved enriched JSON → {output_path}")


if __name__ == "__main__":
    main()