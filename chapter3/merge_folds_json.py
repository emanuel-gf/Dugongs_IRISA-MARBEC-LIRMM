"""
=======================
Merge different json containing the Active Learning proposition with respect the seed.
Build one resolved-paths training JSON per fold.

It does not incorporate the validation set. Therefore it expectes the fine-tunning to not be composed by a validation set.

Per fold F:
  test  = all UM tiles whose mission is in fold F
  train = per-method selection JSONs generated on fold F's pool
          (other folds' UM tiles + manual-flight tiles)

Usage
-----
  python merge_finetune_folds.py \
      --test_seed fold_A \
      --method_jsons ball_A.json centroid_A.json cent_uniq_A.json random_A.json \
      --dataset wp_final_test \
      --output_json resolved_fold_A.json \
      --port 44123
"""

import os
import json
import argparse
from pathlib import Path


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_seed", required=True, help="e.g. test_1")
    p.add_argument("--method_jsons", nargs="+", required=True)
    p.add_argument("--dataset", "-d", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--port", default="44123")
    return p.parse_args()


def load_metadata_entry(entry):
    if isinstance(entry, dict):
        return entry
    if isinstance(entry, str) and Path(entry).exists():
        try:
            with open(entry) as f:
                return json.load(f)
        except Exception as e:
            print(f"  WARNING: bad metadata {entry}: {e}")
    return {}


def normalise_split_dict(split: dict, context: str) -> dict:
    images   = split["images"]
    labels   = split["labels"]
    metadata = split.get("metadata", [{}] * len(images))
    assert len(images) == len(labels) == len(metadata), f"[{context}] length mismatch"

    triples = sorted(zip(images, labels, metadata), key=lambda t: t[0])
    out_i, out_l, out_m = [], [], []
    for img, lbl, meta in triples:
        if not Path(img).exists():
            print(f"  WARNING [{context}]: missing image, skipping: {img}")
            continue
        out_i.append(img); out_l.append(lbl); out_m.append(load_metadata_entry(meta))
    return {"images": out_i, "labels": out_l, "metadata": out_m}


def resolve_tile_paths(image_filepaths: list, context: str) -> dict:
    images, labels, metadata = [], [], []
    for img_str in sorted(image_filepaths):
        img = Path(img_str)
        if not img.exists():
            print(f"  WARNING [{context}]: image not found, skipping: {img}")
            continue
        lbl  = img.parent.parent / "labels"   / f"{img.stem}.txt"
        meta = img.parent.parent / "metadata" / f"{img.stem}.json"
        images.append(str(img)); labels.append(str(lbl))
        metadata.append(load_metadata_entry(str(meta)))
    return {"images": images, "labels": labels, "metadata": metadata}


def main():
    args = get_args()

    os.environ["FIFTYONE_DATABASE_URI"] = f"mongodb://localhost:{args.port}"
    import fiftyone as fo
    from fiftyone import ViewField

    # ── Dataset: test view by mission 
    dataset = fo.load_dataset(args.dataset)
    possible_seeds = list(dataset.count_values("tags").keys())
    assert args.test_seed in possible_seeds, f"Test seed not find"

    test_view = dataset.match_tags(args.test_seed)

    test_paths = test_view.values("filepath")
    print(f"Test tiles: {len(test_paths)}")
    assert len(test_paths) > 0, "Test view is empty — check mission_field values."

    # ── Merge method JSONs (train side) 
    output = {}
    test_set = set(test_paths)   # for leakage check

    for json_path in args.method_jsons:
        json_path = Path(json_path)
        with open(json_path) as f:
            data = json.load(f)
        print(f"\nMerging: {json_path.name}")

        for seed, seed_dict in data.items():
            output.setdefault(seed, {"train": {}})
            for partition, part_dict in seed_dict["train"].items():
                output[seed]["train"].setdefault(partition, {})
                for method, split in part_dict.items():
                    context = f"{json_path.name} {partition}/{method}"
                    norm = normalise_split_dict(split, context)

                    # ── LEAKAGE ASSERTION ─────────────────────────────
                    leaked = test_set.intersection(norm["images"])
                    assert not leaked, (
                        f"LEAKAGE [{context}]: {len(leaked)} train tiles are in "
                        f"the test fold! e.g. {sorted(leaked)[:2]}"
                    )
                    output[seed]["train"][partition][method] = norm
                    print(f"  seed={seed} {partition}/{method}: {len(norm['images'])}")

    # ── Attach test (and val = test view, or leave for your choice) ──────
    test_resolved = resolve_tile_paths(test_paths, f"test/{args.test_seed}")
    for seed in output:
        output[seed]["test"] = test_resolved
        # No-val design: point val at test so the pipeline runs; with fixed
        # epoch budget + final-checkpoint evaluation, val is never used for
        # decisions. Replace with a real val split if you change strategy.
        output[seed]["val"] = test_resolved

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()