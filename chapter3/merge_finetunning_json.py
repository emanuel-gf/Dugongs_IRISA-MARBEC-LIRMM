"""
merge_finetune_json.py
======================
Merge per-method ACLR selection JSONs (+ random) into a single resolved
paths JSON ready for DugongDataModule fine-tuning runs.

Step 1 of 2: merges methods + resolves val/test from target dataset tags.
Step 2 (separate script): enrich with negatives from the negatives dataset.

Usage
-----
  python merge_finetune_json.py \
      --method_jsons centroid.json centroid_uniqueness.json ball_radius.json random.json \
      --dataset my_target_dataset \
      --output_json /path/to/target_resolved_paths.json \
      --port 44123
"""

import os
import json
import argparse
from pathlib import Path


def get_args():
    p = argparse.ArgumentParser(
        description="Merge per-method selection JSONs + tag-based val/test "
                    "into one resolved paths JSON."
    )
    p.add_argument("--method_jsons", nargs="+", required=True)
    p.add_argument("--dataset", "-d", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--port", default="44123")
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────

def load_metadata_entry(entry):
    """Metadata entries may be dicts (pass through) or .json filepaths (load)."""
    if isinstance(entry, dict):
        return entry
    if isinstance(entry, str):
        path = Path(entry)
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception as e:
                print(f"  WARNING: bad metadata {path}: {e}")
        else:
            print(f"  WARNING: metadata file not found: {path}")
    return {}


def normalise_split_dict(split: dict, context: str) -> dict:
    """
    Sort images/labels/metadata JOINTLY by image path (keeps alignment and
    matches the DataModule's later sorted() calls), verify images exist,
    load metadata filepaths into dicts.
    """
    images   = split["images"]
    labels   = split["labels"]
    metadata = split.get("metadata", [{}] * len(images))

    assert len(images) == len(labels) == len(metadata), (
        f"[{context}] length mismatch: images={len(images)} "
        f"labels={len(labels)} metadata={len(metadata)}"
    )

    triples = sorted(zip(images, labels, metadata), key=lambda t: t[0])

    out_images, out_labels, out_meta = [], [], []
    for img, lbl, meta in triples:
        if not Path(img).exists():
            print(f"  WARNING [{context}]: image missing, skipping: {img}")
            continue
        out_images.append(img)
        out_labels.append(lbl)
        out_meta.append(load_metadata_entry(meta))

    return {"images": out_images, "labels": out_labels, "metadata": out_meta}


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
        images.append(str(img))
        labels.append(str(lbl))
        metadata.append(load_metadata_entry(str(meta)))
    return {"images": images, "labels": labels, "metadata": metadata}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    os.environ["FIFTYONE_DATABASE_URI"] = f"mongodb://localhost:{args.port}"
    import fiftyone as fo

    assert args.dataset in fo.list_datasets(), (
        f"Dataset '{args.dataset}' not found. Available: {fo.list_datasets()}"
    )
    dataset = fo.load_dataset(args.dataset)
    print(f"Loaded target dataset '{args.dataset}' ({len(dataset)} samples)")

    # ── 1. Merge all method JSONs ─────────────────────────────────────────
    output = {}
    seen_methods = set()

    for json_path in args.method_jsons:
        json_path = Path(json_path)
        assert json_path.is_file(), f"Not found: {json_path}"

        with open(json_path) as f:
            data = json.load(f)

        print(f"\nMerging: {json_path.name}")

        for seed, seed_dict in data.items():
            output.setdefault(seed, {"train": {}})

            for partition, part_dict in seed_dict["train"].items():
                output[seed]["train"].setdefault(partition, {})

                for method, split in part_dict.items():
                    seen_methods.add(method)
                    context = f"{json_path.name} seed={seed} {partition}/{method}"

                    if method in output[seed]["train"][partition]:
                        print(f"  WARNING: duplicate {partition}/{method} "
                              f"(seed {seed}) — overwriting with {json_path.name}")

                    output[seed]["train"][partition][method] = \
                        normalise_split_dict(split, context)

    # ── 2. Val / test from target dataset tags ────────────────────────────
    for seed in sorted(output.keys()):
        val_view  = dataset.match_tags(f"val_{seed}")
        test_view = dataset.match_tags(f"test_{seed}")

        assert len(val_view)  > 0, f"No samples tagged 'val_{seed}'."
        assert len(test_view) > 0, f"No samples tagged 'test_{seed}'."

        output[seed]["val"]  = resolve_tile_paths(
            val_view.values("filepath"),  f"val_{seed}")
        output[seed]["test"] = resolve_tile_paths(
            test_view.values("filepath"), f"test_{seed}")

    # ── 3. Summary table ──────────────────────────────────────────────────
    methods = sorted(seen_methods)
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for seed in sorted(output.keys()):
        print(f"\nSeed {seed}")
        header = f"  {'partition':<12}" + "".join(f"{m:>22}" for m in methods)
        print(header)
        print("  " + "-" * (len(header) - 2))

        def _pkey(p):   # numeric sort: p5 < p10 < p100
            return int(p.lstrip("p")) if p.lstrip("p").isdigit() else 999

        for partition in sorted(output[seed]["train"].keys(), key=_pkey):
            part_dict = output[seed]["train"][partition]
            row = f"  {partition:<12}"
            for m in methods:
                n = len(part_dict[m]["images"]) if m in part_dict else "MISSING"
                row += f"{str(n):>22}"
            print(row)

        print(f"\n  val  = {len(output[seed]['val']['images'])} tiles")
        print(f"  test = {len(output[seed]['test']['images'])} tiles")

    # ── 4. Write ──────────────────────────────────────────────────────────
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved merged resolved paths JSON → {output_path}")


if __name__ == "__main__":
    main()