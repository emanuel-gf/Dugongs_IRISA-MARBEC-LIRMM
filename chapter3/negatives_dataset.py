"""
sample_negatives.py
====================

Balanced budget sampling of images from a multi-folder dataset.

Iterates over all subfolders in round-robin order, picking one image per
folder per round until the budget is exhausted. This guarantees no single
folder dominates the selection even if its images/ subfolder is much larger
than others. A final shuffle ensures the output list is not folder-grouped.

Two modes:
  --output-txt   : write a text file with one selected filepath per line
  --output-dir   : recreate the source folder structure under a new root,
                   copying selected images/ and their matching labels/ files.
                   Both can be used together.

Usage
-----
  # Just a manifest (no copy)
  python sample_negatives.py \\
      --root /share/home/e2406743/dataset/NEGATIVES_UM \\
      --budget 1000 \\
      --output-txt selected_negatives.txt \\
      --seed 42

  # Recreate folder structure (images/ + labels/ per subfolder)
  python sample_negatives.py \\
      --root /share/home/e2406743/dataset/NEGATIVES_UM \\
      --budget 1000 \\
      --output-dir /share/home/e2406743/dataset/NEGATIVES_UM_1k \\
      --seed 42

  # Both at once
  python sample_negatives.py \\
      --root /share/home/e2406743/dataset/NEGATIVES_UM \\
      --budget 1000 \\
      --output-txt selected_negatives.txt \\
      --output-dir /share/home/e2406743/dataset/NEGATIVES_UM_1k \\
      --seed 42
"""

import argparse
import random
import shutil
from pathlib import Path


def get_args():
    p = argparse.ArgumentParser(
        description="Round-robin balanced image sampling across subfolders.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--root",       "-r", required=True,
                   help="Root directory containing one subfolder per mission.")
    p.add_argument("--budget",     "-b", type=int, default=1000,
                   help="Total number of images to select. (default: 1000)")
    p.add_argument("--output-txt", default=None,
                   help="Optional: write a text manifest of selected filepaths "
                        "(one absolute path per line).")
    p.add_argument("--output-dir", "-o", default=None,
                   help="Optional: recreate the source folder structure "
                        "(images/ + labels/) under this new root, copying "
                        "only the selected files.")
    p.add_argument("--ext", nargs="+",
                   default=["jpg", "jpeg", "JPG", "JPEG"],
                   help="Image file extensions to consider. "
                        "(default: jpg jpeg JPG JPEG)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for per-folder shuffle. (default: 42)")
    return p.parse_args()


def main():
    args = get_args()

    if args.output_txt is None and args.output_dir is None:
        print("ERROR: provide at least one of --output-txt or --output-dir.")
        return

    root = Path(args.root)
    rng  = random.Random(args.seed)

    # ── Collect candidate images per subfolder ─────────────────────────────
    exts = {f".{e.lstrip('.')}" for e in args.ext}

    pools = {}   # {folder_name: [shuffled list of image paths]}
    for subfolder in sorted(root.iterdir()):
        images_dir = subfolder / "images"
        if not subfolder.is_dir() or not images_dir.exists():
            continue

        candidates = [
            p for p in images_dir.iterdir()
            if p.is_file() and p.suffix in exts
        ]
        if not candidates:
            continue

        rng.shuffle(candidates)
        pools[subfolder.name] = candidates

    if not pools:
        print(f"ERROR: no images/ subfolders found under '{root}'.")
        return

    folder_names = sorted(pools.keys())
    print(f"Found {len(folder_names)} folder(s) with images:")
    for name in folder_names:
        print(f"  {name:60s}  {len(pools[name])} images")

    total_available = sum(len(v) for v in pools.values())
    if args.budget > total_available:
        print(f"\nWARNING: budget ({args.budget}) exceeds total available "
              f"images ({total_available}). Selecting all {total_available}.")
        budget = total_available
    else:
        budget = args.budget

    # ── Round-robin selection ──────────────────────────────────────────────
    pointers = {name: 0 for name in folder_names}
    selected = []   # list of Path objects
    active   = list(folder_names)

    while len(selected) < budget and active:
        exhausted = []
        for name in active:
            if len(selected) >= budget:
                break
            idx = pointers[name]
            if idx >= len(pools[name]):
                exhausted.append(name)
                continue
            selected.append(pools[name][idx])
            pointers[name] += 1

        for name in exhausted:
            active.remove(name)

        if not any(pointers[n] <= len(pools[n]) - 1 for n in active):
            break

    # Final shuffle so output is not folder-grouped
    rng.shuffle(selected)

    # ── Optional: write manifest ──────────────────────────────────────────
    if args.output_txt:
        txt_path = Path(args.output_txt)
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text("\n".join(str(p) for p in selected) + "\n")
        print(f"\nManifest -> {txt_path.resolve()}")

    # ── Optional: recreate folder structure ───────────────────────────────
    if args.output_dir:
        out_root = Path(args.output_dir)
        copied_images = 0
        copied_labels = 0
        missing_labels = 0

        for img_path in selected:
            # img_path is e.g. <root>/<subfolder>/images/<stem>.jpg
            subfolder_name = img_path.parent.parent.name   # e.g. FPLAN_M4_UM_M13_...
            stem           = img_path.stem

            # ── Copy image ─────────────────────────────────────────────────
            dest_img_dir = out_root / subfolder_name / "images"
            dest_img_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dest_img_dir / img_path.name)
            copied_images += 1

            # ── Copy matching label if it exists ───────────────────────────
            # Check the sibling labels/ folder (standard structure)
            src_label = img_path.parent.parent / "labels" / f"{stem}.txt"
            if src_label.exists():
                dest_lbl_dir = out_root / subfolder_name / "labels"
                dest_lbl_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_label, dest_lbl_dir / src_label.name)
                copied_labels += 1
            else:
                # Write an empty label file to keep images/ and labels/ in sync
                dest_lbl_dir = out_root / subfolder_name / "labels"
                dest_lbl_dir.mkdir(parents=True, exist_ok=True)
                (dest_lbl_dir / f"{stem}.txt").write_text("")
                missing_labels += 1

        print(f"\nFolder structure recreated -> {out_root.resolve()}")
        print(f"  Images copied   : {copied_images}")
        print(f"  Labels copied   : {copied_labels}")
        print(f"  Empty labels written (no source .txt found): {missing_labels}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\nSelected {len(selected)} images from {len(folder_names)} folders.")

    from_folder = {name: 0 for name in folder_names}
    for img_path in selected:
        folder = img_path.parent.parent.name
        if folder in from_folder:
            from_folder[folder] += 1

    print("\nPer-folder selection count:")
    for name in folder_names:
        n_sel = from_folder[name]
        n_tot = len(pools[name])
        bar   = "█" * (n_sel // max(1, budget // 40))
        print(f"  {name:60s}  {n_sel:>4} / {n_tot:>5}  {bar}")


if __name__ == "__main__":
    main()

