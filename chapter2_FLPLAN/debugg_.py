"""
diagnose_tile.py
================
Traces the full coordinate pipeline for a specific tile to find bbox shifts.

Usage:
    python diagnose_tile.py \\
        --image /path/to/mission/images/IMG.jpeg \\
        --tile-x 2160 \\
        --tile-y 0 \\
        --tile-size 640 \\
        --overlap 100 \\
        --save-debug /tmp/debug
"""

import argparse
from pathlib import Path
from PIL import Image, ImageDraw


def get_args():
    p = argparse.ArgumentParser(description="Diagnose bbox alignment for one tile.")
    p.add_argument("--image",      required=True, help="Full path to source .jpeg image.")
    p.add_argument("--tile-x",     type=int, required=True,
                   help="x_start of the tile in pixels (as in the patch filename).")
    p.add_argument("--tile-y",     type=int, required=True,
                   help="y_start of the tile in pixels (as in the patch filename).")
    p.add_argument("--tile-size",  type=int, default=640)
    p.add_argument("--overlap",    type=int, default=100)
    p.add_argument("--save-debug", default=None,
                   help="Path stem for debug images (e.g. /tmp/debug). "
                        "Writes <stem>_crop.jpg and <stem>_context.jpg.")
    return p.parse_args()


def load_gt_boxes_pixels(image_filepath, img_w, img_h):
    image_path = Path(image_filepath)
    label_path = image_path.parent.parent / "labels_yolo" / (image_path.stem + ".txt")

    print(f"\n--- GT label file ---")
    print(f"  Path : {label_path}")
    print(f"  Exists: {label_path.exists()}")

    boxes = []
    if not label_path.exists():
        print("  WARNING: label file not found — check the path above.")
        return boxes

    raw = label_path.read_text().strip().splitlines()
    print(f"  Lines: {len(raw)}")

    for i, line in enumerate(raw):
        parts = line.strip().split()
        if not parts:
            continue
        cls  = parts[0]
        cx_n = float(parts[1]);  cy_n = float(parts[2])
        bw_n = float(parts[3]);  bh_n = float(parts[4])

        cx_px = cx_n * img_w;  cy_px = cy_n * img_h
        bw_px = bw_n * img_w;  bh_px = bh_n * img_h

        x1 = max(0, int(round(cx_px - bw_px / 2)))
        y1 = max(0, int(round(cy_px - bh_px / 2)))
        x2 = min(img_w, int(round(cx_px + bw_px / 2)))
        y2 = min(img_h, int(round(cy_px + bh_px / 2)))

        print(f"  [{i}] cls={cls}  "
              f"cx_n={cx_n:.6f} cy_n={cy_n:.6f} w_n={bw_n:.6f} h_n={bh_n:.6f}")
        print(f"       cx_px={cx_px:.2f} cy_px={cy_px:.2f} "
              f"w_px={bw_px:.2f} h_px={bh_px:.2f}")
        print(f"       bbox: ({x1},{y1})->({x2},{y2})  size={x2-x1}x{y2-y1}")

        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])

    return boxes


def trace_tile(image_filepath, x_start, y_start, tile_size, overlap, save_debug):
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"IMAGE : {Path(image_filepath).name}")
    print(f"TILE  : x_start={x_start}  y_start={y_start}")
    print(f"PARAMS: tile_size={tile_size}  overlap={overlap}")
    print(sep)

    with Image.open(image_filepath) as img:
        img_w, img_h = img.size
    print(f"\n--- Source image dimensions ---")
    print(f"  {img_w} x {img_h} px")

    # Tile boundaries
    x_end  = min(x_start + tile_size, img_w)
    y_end  = min(y_start + tile_size, img_h)
    tile_pw = x_end - x_start
    tile_ph = y_end - y_start

    print(f"\n--- Tile pixel boundaries ---")
    print(f"  x: [{x_start}, {x_end}]  crop_w={tile_pw}px")
    print(f"  y: [{y_start}, {y_end}]  crop_h={tile_ph}px")
    is_edge = (tile_pw != tile_size or tile_ph != tile_size)
    if is_edge:
        sx = tile_size / tile_pw;  sy = tile_size / tile_ph
        print(f"  EDGE TILE: crop={tile_pw}x{tile_ph} -> resized to {tile_size}x{tile_size}")
        print(f"  Resize scale: sx={sx:.8f}  sy={sy:.8f}")
    else:
        sx = sy = 1.0
        print(f"  Interior tile: no resize needed")

    # GT boxes
    gt_boxes = load_gt_boxes_pixels(image_filepath, img_w, img_h)

    # Intersection
    print(f"\n--- Intersection analysis ---")
    overlapping = []

    for i, (gx1, gy1, gx2, gy2) in enumerate(gt_boxes):
        ix1 = max(gx1, x_start);  iy1 = max(gy1, y_start)
        ix2 = min(gx2, x_end);    iy2 = min(gy2, y_end)
        overlaps = ix1 < ix2 and iy1 < iy2

        print(f"\n  GT box [{i}]: source px ({gx1},{gy1})->({gx2},{gy2})")
        print(f"    overlaps tile: {overlaps}")

        if not overlaps:
            continue

        # Translate to tile-local pixel coords
        lx1 = ix1 - x_start;  ly1 = iy1 - y_start
        lx2 = ix2 - x_start;  ly2 = iy2 - y_start
        print(f"    clipped (source px):  ({ix1},{iy1})->({ix2},{iy2})")
        print(f"    tile-local px:        ({lx1},{ly1})->({lx2},{ly2})")

        # Apply resize scale
        rx1 = lx1 * sx;  ry1 = ly1 * sy
        rx2 = lx2 * sx;  ry2 = ly2 * sy
        if is_edge:
            print(f"    after resize:         ({rx1:.2f},{ry1:.2f})->({rx2:.2f},{ry2:.2f})")

        # Normalise to [0,1]
        nx1 = rx1 / tile_size;  ny1 = ry1 / tile_size
        nx2 = rx2 / tile_size;  ny2 = ry2 / tile_size

        # YOLO centre
        cx = (nx1 + nx2) / 2;  cy = (ny1 + ny2) / 2
        w  = nx2 - nx1;        h  = ny2 - ny1

        print(f"    YOLO: 0 {cx:.10f} {cy:.10f} {w:.10f} {h:.10f}")
        print(f"    centre in resized tile px: ({cx*tile_size:.2f}, {cy*tile_size:.2f})")
        print(f"    box size in resized tile px: {w*tile_size:.2f} x {h*tile_size:.2f}")

        overlapping.append({
            "idx":        i,
            "gt_src":     (gx1, gy1, gx2, gy2),
            "tile_local": (lx1, ly1, lx2, ly2),
            "yolo":       (cx, cy, w, h),
        })

    # Check if exported label file exists nearby
    stem       = Path(image_filepath).stem
    label_name = f"{stem}__tile_{y_start}_{x_start}_p.txt"
    print(f"\n--- Exported label file check ---")
    print(f"  Expected filename: {label_name}")
    for found in Path(image_filepath).parent.parent.parent.rglob(label_name):
        content = found.read_text().strip()
        print(f"  Found: {found}")
        print(f"  Content: '{content}'")
        for line in content.splitlines():
            parts = line.split()
            if len(parts) == 5:
                ex_cx = float(parts[1]);  ex_cy = float(parts[2])
                print(f"  Exported centre in px: ({ex_cx*tile_size:.2f}, {ex_cy*tile_size:.2f})")

    # Save debug images
    if save_debug:
        save_stem = Path(save_debug)
        save_stem.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(image_filepath) as img:
            # 1. Cropped tile with YOLO boxes drawn on it
            crop = img.crop((x_start, y_start, x_end, y_end))
            if is_edge:
                crop = crop.resize((tile_size, tile_size), Image.LANCZOS)
            draw = ImageDraw.Draw(crop)

            for ov in overlapping:
                cx, cy, w, h = ov["yolo"]
                bx1 = (cx - w/2) * tile_size
                by1 = (cy - h/2) * tile_size
                bx2 = (cx + w/2) * tile_size
                by2 = (cy + h/2) * tile_size
                draw.rectangle([bx1, by1, bx2, by2], outline="red", width=2)
                draw.text((bx1+2, max(0, by1-14)), f"box{ov['idx']}", fill="red")

            crop_path = str(save_stem) + "_crop.jpg"
            crop.save(crop_path, quality=95)
            print(f"\n  Saved: {crop_path}")
            print(f"    (red boxes = YOLO labels drawn back onto the crop)")

            # 2. Context view: zoomed-out source image showing tile + GT boxes
            pad = 300
            vx1 = max(0, x_start - pad);  vy1 = max(0, y_start - pad)
            vx2 = min(img_w, x_end + pad); vy2 = min(img_h, y_end + pad)
            ctx = img.crop((vx1, vy1, vx2, vy2)).copy()
            draw2 = ImageDraw.Draw(ctx)

            # Tile boundary in blue
            draw2.rectangle(
                [x_start-vx1, y_start-vy1, x_end-vx1, y_end-vy1],
                outline="blue", width=3
            )
            # All GT boxes in green
            for (gx1, gy1, gx2, gy2) in gt_boxes:
                draw2.rectangle(
                    [gx1-vx1, gy1-vy1, gx2-vx1, gy2-vy1],
                    outline="lime", width=2
                )
            ctx_path = str(save_stem) + "_context.jpg"
            ctx.save(ctx_path, quality=95)
            print(f"  Saved: {ctx_path}")
            print(f"    (blue = tile boundary, green = all GT boxes in source)")

    # Summary
    print(f"\n{sep}")
    print(f"SUMMARY: {len(overlapping)} dugong(s) in tile_y{y_start}_x{x_start}")
    for ov in overlapping:
        cx, cy, w, h = ov["yolo"]
        print(f"  GT[{ov['idx']}]: YOLO 0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    print(sep)


def main():
    args = get_args()
    trace_tile(
        image_filepath=args.image,
        x_start=args.tile_x,
        y_start=args.tile_y,
        tile_size=args.tile_size,
        overlap=args.overlap,
        save_debug=args.save_debug,
    )


if __name__ == "__main__":
    main()