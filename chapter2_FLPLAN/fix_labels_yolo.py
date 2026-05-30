"""
fix_yolo_labels.py
==================
Recomputes YOLO label files from the original pixel-coordinate annotations,
using the TRUE image dimensions read by PIL for each image.

The original label format is:
    x1 y1 x2 y2 Dugong <extra_fields>
    (absolute pixel coordinates, top-left + bottom-right corners)

The YOLO format we write is:
    0 cx_n cy_n w_n h_n
    (class 0, all normalised [0,1] to true image dimensions)

Why recompute from scratch:
    The existing labels_yolo/ files were generated with a hardcoded (wrong)
    image height that varies per mission/camera. Rather than inferring and
    correcting the wrong height, we go back to the original pixel annotations
    and divide by the PIL-measured true dimensions. This is mission-agnostic.

Folder structure expected:
    <mission_root>/
        images/         *.jpeg  (or *.jpg)
        labels/         *.txt   (original x1 y1 x2 y2 format)
        labels_yolo/    *.txt   (YOLO format — will be overwritten)

Usage:
    # Dry run first (prints what would change, writes nothing)
    python fix_yolo_labels.py \\
        --root /share/home/e2406743/dataset/new_dataset/UM_DUGONG_2025 \\
        --dry-run

    # Apply fix
    python fix_yolo_labels.py \\
        --root /share/home/e2406743/dataset/new_dataset/UM_DUGONG_2025

    # Backup existing labels_yolo/ before fixing
    python fix_yolo_labels.py \\
        --root /share/home/e2406743/dataset/new_dataset/UM_DUGONG_2025 \\
        --backup
"""

import argparse
import shutil
from pathlib import Path
from PIL import Image


def get_args():
    p = argparse.ArgumentParser(
        description="Recompute labels_yolo/ from original pixel labels using PIL image dims.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--root", "-r", required=True,
                   help="Root folder to search recursively (e.g. UM_DUGONG_2025/).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would change but write nothing.")
    p.add_argument("--backup", action="store_true",
                   help="Copy each labels_yolo/ to labels_yolo_backup/ before overwriting.")
    p.add_argument("--label-folder",  default="labels",
                   help="Name of the original label folder. (default: labels)")
    p.add_argument("--yolo-folder",   default="labels_yolo",
                   help="Name of the YOLO output folder. (default: labels_yolo)")
    p.add_argument("--image-folder",  default="images",
                   help="Name of the images folder. (default: images)")
    p.add_argument("--img-ext",       default=".jpeg",
                   help="Image extension to look for. (default: .jpeg)")
    return p.parse_args()


def parse_original_label(label_path: Path) -> list:
    """
    Parse the original label file.
    Expected format per line:
        x1 y1 x2 y2 Dugong <extra>
    Returns list of (x1, y1, x2, y2) integer tuples.
    """
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                x1, y1, x2, y2 = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2, y2))
            except (ValueError, IndexError):
                # skip header lines or malformed entries
                continue
    return boxes


def boxes_to_yolo(boxes: list, img_w: int, img_h: int) -> list:
    """
    Convert list of (x1,y1,x2,y2) pixel boxes to YOLO format strings.
    YOLO: class cx_n cy_n w_n h_n  (all normalised to [0,1])
    """
    lines = []
    for (x1, y1, x2, y2) in boxes:
        cx_n = ((x1 + x2) / 2) / img_w
        cy_n = ((y1 + y2) / 2) / img_h
        w_n  = (x2 - x1) / img_w
        h_n  = (y2 - y1) / img_h

        # Clamp to [0,1] in case of any pixel-boundary annotation overflow
        cx_n = max(0.0, min(1.0, cx_n))
        cy_n = max(0.0, min(1.0, cy_n))
        w_n  = max(0.0, min(1.0, w_n))
        h_n  = max(0.0, min(1.0, h_n))

        lines.append(f"0 {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")
    return lines


def find_image(images_dir: Path, stem: str, img_ext: str) -> Path | None:
    """Try the configured extension, then common fallbacks."""
    for ext in [img_ext, ".jpeg", ".jpg", ".JPG", ".JPEG", ".png"]:
        candidate = images_dir / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def process_mission(
    mission_dir: Path,
    label_folder: str,
    yolo_folder: str,
    image_folder: str,
    img_ext: str,
    dry_run: bool,
    backup: bool,
) -> dict:
    """
    Process one mission directory.
    Returns stats dict: {processed, skipped_no_image, skipped_no_label, errors}
    """
    labels_dir      = mission_dir / label_folder
    yolo_dir        = mission_dir / yolo_folder
    images_dir      = mission_dir / image_folder

    if not labels_dir.exists() or not images_dir.exists():
        return None   # not a mission dir with the expected structure

    label_files = sorted(labels_dir.glob("*.txt"))
    if not label_files:
        return None

    # Backup if requested
    if backup and not dry_run:
        backup_dir = mission_dir / (yolo_folder + "_backup")
        if yolo_dir.exists() and not backup_dir.exists():
            shutil.copytree(yolo_dir, backup_dir)
            print(f"  Backed up → {backup_dir}")

    stats = {"processed": 0, "skipped_no_image": 0,
             "skipped_no_label": 0, "errors": 0,
             "changed": 0, "unchanged": 0}

    for lbl_path in label_files:
        stem      = lbl_path.stem
        img_path  = find_image(images_dir, stem, img_ext)
        yolo_path = yolo_dir / (stem + ".txt")

        if img_path is None:
            stats["skipped_no_image"] += 1
            continue

        # Parse original pixel labels
        boxes = parse_original_label(lbl_path)
        if not boxes:
            # Write empty YOLO file (negative image — no dugongs)
            if not dry_run:
                yolo_dir.mkdir(parents=True, exist_ok=True)
                yolo_path.write_text("")
            stats["processed"] += 1
            continue

        # Read true image dimensions with PIL
        try:
            with Image.open(img_path) as img:
                true_w, true_h = img.size
        except Exception as e:
            print(f"  ERROR reading {img_path.name}: {e}")
            stats["errors"] += 1
            continue

        # Compute correct YOLO lines
        new_lines = boxes_to_yolo(boxes, true_w, true_h)
        new_content = "\n".join(new_lines)

        # Compare with existing YOLO file
        old_content = yolo_path.read_text().strip() if yolo_path.exists() else ""
        changed = (new_content.strip() != old_content)

        if changed:
            stats["changed"] += 1
        else:
            stats["unchanged"] += 1

        if not dry_run:
            yolo_dir.mkdir(parents=True, exist_ok=True)
            yolo_path.write_text(new_content)

        stats["processed"] += 1

    return stats


def main():
    args = get_args()
    root = Path(args.root)
    assert root.exists(), f"Root not found: {root}"

    print(f"{'DRY RUN — ' if args.dry_run else ''}Scanning: {root}")
    print(f"  label_folder  : {args.label_folder}")
    print(f"  yolo_folder   : {args.yolo_folder}")
    print(f"  image_folder  : {args.image_folder}")
    print(f"  backup        : {args.backup}")
    print()

    # Find all mission directories (any dir that contains images/ and labels/)
    mission_dirs = set()
    for img_dir in root.rglob(args.image_folder):
        candidate = img_dir.parent
        if (candidate / args.label_folder).exists():
            mission_dirs.add(candidate)

    print(f"Found {len(mission_dirs)} mission directories\n")

    total = {"processed": 0, "skipped_no_image": 0,
             "skipped_no_label": 0, "errors": 0,
             "changed": 0, "unchanged": 0}

    for mission_dir in sorted(mission_dirs):
        stats = process_mission(
            mission_dir=mission_dir,
            label_folder=args.label_folder,
            yolo_folder=args.yolo_folder,
            image_folder=args.image_folder,
            img_ext=args.img_ext,
            dry_run=args.dry_run,
            backup=args.backup,
        )
        if stats is None:
            continue

        print(f"  {mission_dir.name}")
        print(f"    processed={stats['processed']}  "
              f"changed={stats['changed']}  "
              f"unchanged={stats['unchanged']}  "
              f"no_image={stats['skipped_no_image']}  "
              f"errors={stats['errors']}")

        for k in total:
            total[k] += stats[k]

    print(f"\n{'='*55}")
    print(f"{'DRY RUN SUMMARY' if args.dry_run else 'DONE'}")
    print(f"  Total processed  : {total['processed']}")
    print(f"  Labels changed   : {total['changed']}")
    print(f"  Labels unchanged : {total['unchanged']}")
    print(f"  Skipped (no img) : {total['skipped_no_image']}")
    print(f"  Errors           : {total['errors']}")
    if args.dry_run:
        print(f"\n  Run without --dry-run to apply changes.")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()