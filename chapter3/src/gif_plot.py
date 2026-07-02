"""
failure_gif.py
================

Builds a "find the dugong" reveal-style GIF from a list of failed
classification tile filepaths: each tile is shown first as the raw
image (longer duration), then with its ground-truth bounding box(es)
overlaid (shorter duration), before moving to the next tile.

Bounding boxes are read directly from the tile's matching labels/*.txt
file (same naming convention as create_tile_dataset.py / tile_export_pipeline.py):

    <root>/positive/images/<stem>.jpg
    <root>/positive/labels/<stem>.txt   <- read from here

No titles, legends, or other text overlays are drawn -- pure image frames.

Usage
-----
    from failure_gif import create_failure_reveal_gif

    create_failure_reveal_gif(
        filepaths=failed_image_paths,   # list of image filepaths (strings)
        output_path="/tmp/dugong_failures.gif",
    )
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from PIL import Image, ImageDraw, ImageFont


def _get_font(img_w: int, img_h: int, scale: float = 0.05):
    """
    Tries to load a real TrueType font sized relative to the image
    (5% of the longer side by default); falls back to PIL's tiny
    built-in bitmap font if none is available.
    """
    size = max(14, int(max(img_w, img_h) * scale))
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_label(pil_img: Image.Image, text: str,
                text_color=(255, 255, 255),
                bg_color=(0, 0, 0),
                margin: int = 8,
                padding: int = 6) -> Image.Image:
    """
    Returns a COPY of pil_img with `text` drawn in the upper-left corner
    on a filled background rectangle (for readability over water/imagery).
    """
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    font = _get_font(*img.size)

    # measure text
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    x0, y0 = margin, margin
    draw.rectangle(
        [x0, y0, x0 + tw + 2 * padding, y0 + th + 2 * padding],
        fill=bg_color,
    )
    draw.text((x0 + padding - bbox[0], y0 + padding - bbox[1]),
              text, fill=text_color, font=font)
    return img

def _load_yolo_boxes(label_path: Path, img_w: int, img_h: int) -> list:
    """
    Reads a YOLO-format label file (class cx cy w h, normalised [0,1]) and
    returns a list of absolute pixel rectangles [x1, y1, x2, y2].
    Returns an empty list if the file is missing or empty.
    """
    if not label_path.exists():
        return []

    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cx, cy, w, h = (float(v) for v in parts[1:5])

            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append([x1, y1, x2, y2])

    return boxes


def _find_label_path(image_path: Path) -> Path:
    """
    Resolves the matching label file for a tile image, assuming the
    standard tiling-pipeline layout:
        <root>/<positive|negative>/images/<stem>.jpg
        <root>/<positive|negative>/labels/<stem>.txt
    i.e. labels/ is a SIBLING of images/, both directly under the
    positive/negative folder -- so label_path = image.parent.parent / "labels" / f"{stem}.txt"
    """
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def _draw_boxes(pil_img: Image.Image, boxes: list, color=(0, 255, 136), width: int = 4) -> Image.Image:
    """Returns a COPY of pil_img with rectangles drawn for each box."""
    img_with_boxes = pil_img.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    for (x1, y1, x2, y2) in boxes:
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return img_with_boxes


def create_failure_reveal_gif(
    filepaths: list,
    output_path: str,
    raw_duration_s: float = 1.5,
    reveal_duration_s: float = 1.0,
    labels: list | None = None,  
    max_size: int | None = 800,
    box_color: tuple = (0, 255, 136),
    box_width: int = 4,
    loop: int = 0,
):
    """
    Builds a reveal-style GIF: for each filepath, shows the raw tile image
    first (raw_duration_s), then the same tile with its ground-truth
    bounding box(es) overlaid (reveal_duration_s), then moves to the next
    tile. No titles, legends, or text are drawn anywhere.

    Parameters
    ----------
    filepaths        : list[str] -- image filepaths, in the order they
                        should appear in the GIF. Each must follow the
                        tiling-pipeline layout (labels/ sibling to images/)
                        so its bounding box(es) can be located automatically.
    output_path       : str -- where to save the .gif
    raw_duration_s    : float -- seconds to show the raw (un-annotated) frame
                        (default 1.5 -- longer, gives the viewer time to look)
    reveal_duration_s : float -- seconds to show the bbox-overlaid frame
                        (default 1.0 -- shorter, the "reveal")
    max_size          : int or None -- if set, each frame is resized so its
                        longer side is at most this many pixels (keeps file
                        size reasonable for many large tiles). None = no resize.
    box_color         : tuple (R,G,B) -- bounding box outline color
    box_width         : int -- bounding box outline thickness in pixels
    loop              : int -- 0 = infinite loop, N = loop N times

    Returns
    -------
    dict with keys: path, n_tiles, n_frames, size_kb, missing_labels (list
                     of filepaths whose label file was missing/empty --
                     those tiles still appear in the GIF, just with no
                     box drawn on the reveal frame)
    """
    frames = []
    durations = []
    missing_labels = []

    if labels is not None and len(labels) != len(filepaths):
        raise ValueError(
            f"`labels` must be the same length as `filepaths` "
            f"({len(labels)} != {len(filepaths)})"
        )
    
    for i,fp in enumerate(filepaths):
        image_path = Path(fp)

        with Image.open(image_path) as img:
            pil_img = img.convert("RGB").copy()

        img_w, img_h = pil_img.size

        if max_size is not None and max(img_w, img_h) > max_size:
            scale = max_size / max(img_w, img_h)
            new_size = (int(img_w * scale), int(img_h * scale))
            pil_img = pil_img.resize(new_size, Image.LANCZOS)
            img_w, img_h = new_size

        # ── NEW: stamp the per-tile label onto the base image ──────────
        if labels is not None and labels[i] is not None:
            lab = labels[i]
            text = f"{lab:.3f}" if isinstance(lab, float) else str(lab)
            pil_img = _draw_label(pil_img, text)

        label_path = _find_label_path(image_path)
        boxes = _load_yolo_boxes(label_path, img_w, img_h)
        if not boxes:
            missing_labels.append(str(image_path))

        # Frame 1: raw image, no annotations
        frames.append(np.array(pil_img))
        durations.append(raw_duration_s)

        # Frame 2: same image with bbox(es) overlaid
        pil_revealed = _draw_boxes(pil_img, boxes, color=box_color, width=box_width)
        frames.append(np.array(pil_revealed))
        durations.append(reveal_duration_s)

    if not frames:
        raise ValueError("No frames were generated -- 'filepaths' is empty?")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Write via Pillow directly, with explicit per-frame durations ────────
    # GIF stores delay in centiseconds (1/100s) internally; Pillow's
    # save(..., duration=...) wants MILLISECONDS per frame as a list (one
    # entry per frame) -- using imageio's duration= here has historically
    # been unreliable about units/seconds-vs-ms across versions and
    # backends, which is almost certainly why your 15s setting had no
    # visible effect. Writing through Pillow directly avoids that ambiguity.
    durations_ms = [int(round(d * 1000)) for d in durations]

    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=durations_ms,   # list, one entry per frame, in milliseconds
        loop=loop,
        disposal=2,              # replace each frame fully (avoids ghosting)
    )

    # ── Sanity check: read the durations back out of the saved file ─────────
    with Image.open(output_path) as check_img:
        actual_durations_ms = []
        try:
            for i in range(check_img.n_frames):
                check_img.seek(i)
                actual_durations_ms.append(check_img.info.get("duration", None))
        except Exception:
            pass

    size_kb = output_path.stat().st_size / 1024
    n_tiles = len(filepaths)
    n_frames = len(frames)

    print(f"GIF saved -> {output_path}")
    print(f"  Tiles: {n_tiles}  Frames: {n_frames} (2 per tile)")
    print(f"  Size: {size_kb:.1f} KB")
    print(f"  Requested durations (ms), first 4 frames: {durations_ms[:4]}")
    print(f"  Actual durations read back from file, first 4 frames: "
          f"{actual_durations_ms[:4]}")
    if actual_durations_ms[:4] != durations_ms[:4]:
        print("  NOTE: actual != requested -- your GIF VIEWER may still clamp "
              "or round long delays even though the file stores them correctly. "
              "Try opening the .gif directly in a browser tab (not a Slack/Discord "
              "preview or some IDE thumbnailers, which often hardcode a max delay).")
    if missing_labels:
        print(f"  WARNING: {len(missing_labels)} tile(s) had no bounding box "
              f"found (missing/empty label file) -- reveal frame shows no box "
              f"for these.")

    return {
        "path": str(output_path),
        "n_tiles": n_tiles,
        "n_frames": n_frames,
        "size_kb": size_kb,
        "missing_labels": missing_labels,
        "requested_durations_ms": durations_ms,
        "actual_durations_ms": actual_durations_ms,
    }