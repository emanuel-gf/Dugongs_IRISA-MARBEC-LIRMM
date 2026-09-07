"""
reconstruct_flplan_predictions.py
==================================
Utilities to reconstruct FLPLAN tile-level inference results (produced by
the RT-DETR fine-tuning pipeline) back into full-image FiftyOne Detections
fields, with optional Non-Maximum Suppression.

New JSON format (one file per run, all tiles inside)
-----------------------------------------------------
[
  {
    "filepath":      "/path/to/images/{sample_stem}__{tile_y}_{tile_x}_{type}.jpg",
    "detections":    [{"label": "dugong",
                       "bounding_box": [cx, cy, w, h],   ← RT-DETR centre format
                       "confidence": float}, ...],
    "tile_metadata": "/path/to/metadata/..."              ← ignored here
  },
  ...
]

Parent-image identification
----------------------------
Tile filepath stem follows the pattern:
    {sample_stem}__{tile_y}_{tile_x}_{type}
so  Path(filepath).stem.split('__')[0]  gives sample_stem.

Bounding-box conversion
------------------------
RT-DETR outputs [cx, cy, w, h] normalised to the tile.
We convert to top-left [x, y, w, h] before remapping to full-image coords:
    x = cx - w / 2
    y = cy - h / 2

Field naming
------------
Given a JSON path such as:
    .../NWW_p5_aclr_seed1_rtdetr_0606_1418_test_predictions.json
the FiftyOne field name is derived by:
    1. Taking the stem:  NWW_p5_aclr_seed1_rtdetr_0606_1418_test_predictions
    2. Splitting on '_rtdetr' and keeping part[0]: NWW_p5_aclr_seed1
This gives a short, readable field name that encodes schema / partition /
method / seed without the run timestamp.

Public API
----------
derive_field_name(json_path)
    → str   field name derived from the JSON filename

reconstruct_from_json(dataset, json_path, field_name=None,
                      tile_size=640, confidence_threshold=0.0,
                      verbose=True)
    → saves {field_name}_raw on matching samples

reconstruct_and_nms(dataset, json_path, field_name=None,
                    tile_size=640, iou_threshold=0.35,
                    confidence_threshold=0.0, verbose=True)
    → calls reconstruct_from_json, then applies per-image NMS
    → saves {field_name}_nms

reconstruct_batch(dataset, json_paths, tile_size=640,
                  iou_threshold=0.35, confidence_threshold=0.0,
                  run_nms=True, verbose=True)
    → loops over a list of JSON paths and calls reconstruct_and_nms
       (or reconstruct_from_json) for each one.
    → intended to be called directly from a notebook.
"""

from __future__ import annotations

import json
import warnings
from collections import defaultdict
from pathlib import Path

import torch
import torchvision.ops as tv_ops
import fiftyone as fo
from fiftyone import ViewField as F


# ─────────────────────────────────────────────────────────────────────────────
# Field name helper
# ─────────────────────────────────────────────────────────────────────────────

def derive_field_name(json_path: str | Path) -> str:
    """
    Derive a short FiftyOne field name from a JSON prediction file path.

    Rule
    ----
    Take the file stem, split on '_rtdetr', keep the left part.

    Examples
    --------
    >>> derive_field_name(
    ...   ".../NWW_p5_aclr_seed1_rtdetr_0606_1418_test_predictions.json"
    ... )
    'NWW_p5_aclr_seed1'

    >>> derive_field_name(
    ...   ".../NWW_p10_random_seed0_rtdetr_0607_0900_test_predictions.json"
    ... )
    'NWW_p10_random_seed0'
    """
    stem = Path(json_path).stem                  # strip .json
    return stem.split("_rtdetr")[0]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _log(msg: str, verbose: bool, level: str = "info") -> None:
    if not verbose:
        return
    prefix = {"info": "  ", "success": "OK  ", "warn": "WARN", "error": "ERR "}.get(
        level, "  "
    )
    print(f"[{prefix}] {msg}")


def _parse_tile_filepath(filepath: str) -> tuple[str, int, int, str] | None:
    """
    Extract (sample_stem, y_stride, x_stride, tile_type) from a tile filepath.

    Tile filename format:
        {sample_stem}__{y_stride}_{x_stride}_{type}.jpg

    Examples
    --------
    'FPLAN_M4_UM_M46_F1_2025_P_2-699847233596f_105__tile_2200_2200_n.jpg'
        → ('FPLAN_M4_UM_M46_F1_2025_P_2-699847233596f_105', 2200, 2200, 'n')

    Returns None if the filename cannot be parsed.
    """
    stem   = Path(filepath).stem          # drop extension
    parts  = stem.split("__")

    if len(parts) != 2:
        return None

    sample_stem = parts[0]
    tile_part   = parts[1]               # e.g. "tile_2200_2200_n"
    tile_tokens = tile_part.split("_")   # ['tile', '2200', '2200', 'n']

    if len(tile_tokens) < 4:
        return None

    try:
        y_stride  = int(tile_tokens[1])
        x_stride  = int(tile_tokens[2])
        tile_type = tile_tokens[3]        # 'n' or 'p'
    except (ValueError, IndexError):
        return None

    return sample_stem, y_stride, x_stride, tile_type


def _group_entries_by_sample(
    entries: list[dict],
    verbose: bool,
) -> dict[str, list[dict]]:
    """
    Group JSON entries by sample stem derived from each entry's filepath.

    Returns
    -------
    {
      'FPLAN_M4_UM_M46_F1_2025_P_2-699847233596f_105': [
          {'y_stride': 2200, 'x_stride': 2200, 'tile_type': 'n',
           'detections': [...]},
          ...
      ],
      ...
    }
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    n_skipped = 0

    for entry in entries:
        filepath = entry.get("filepath", "")
        parsed   = _parse_tile_filepath(filepath)

        if parsed is None:
            _log(f"Cannot parse tile filepath, skipping: {filepath}", verbose, "warn")
            n_skipped += 1
            continue

        sample_stem, y_stride, x_stride, tile_type = parsed
        grouped[sample_stem].append({
            "y_stride":   y_stride,
            "x_stride":   x_stride,
            "tile_type":  tile_type,
            "detections": entry.get("detections", []),
        })

    _log(
        f"Grouped {len(entries) - n_skipped} tile entries → "
        f"{len(grouped)} unique sample stems  "
        f"(skipped {n_skipped} unparseable)",
        verbose,
    )
    return grouped


def _tile_pixel_extent(
    y_stride: int,
    x_stride: int,
    img_w: int,
    img_h: int,
    tile_size: int = 640,
) -> tuple[int, int, int, int]:
    """
    Recompute (x_start, y_start, tile_w, tile_h) in pixels, mirroring
    add_roi_grid clamping logic exactly.
    """
    x_end   = min(x_stride + tile_size, img_w)
    y_end   = min(y_stride + tile_size, img_h)
    x_start = max(x_end - tile_size, 0)
    y_start = max(y_end - tile_size, 0)
    tile_w  = x_end - x_start
    tile_h  = y_end - y_start
    return x_start, y_start, tile_w, tile_h


def _cx_cy_to_xywh(cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float]:
    """
    Convert RT-DETR centre format [cx, cy, w, h] → top-left [x, y, w, h].
    All values remain normalised to [0, 1] within the tile.
    """
    x = cx - w / 2
    y = cy - h / 2
    return x, y, w, h


def _remap_to_full_image(
    tile_bbox: tuple[float, float, float, float],  # [x, y, w, h] top-left, normalised to tile
    x_start: int,
    y_start: int,
    tile_w: int,
    tile_h: int,
    img_w: int,
    img_h: int,
) -> list[float]:
    """
    Convert tile-local normalised [x, y, w, h]
    → full-image normalised [x, y, w, h].
    """
    xn, yn, wn, hn = tile_bbox

    x_full = (xn * tile_w + x_start) / img_w
    y_full = (yn * tile_h + y_start) / img_h
    w_full = wn * tile_w / img_w
    h_full = hn * tile_h / img_h

    # clamp to [0, 1]
    x_full = max(0.0, min(x_full, 1.0))
    y_full = max(0.0, min(y_full, 1.0))
    w_full = max(0.0, min(w_full, 1.0 - x_full))
    h_full = max(0.0, min(h_full, 1.0 - y_full))

    return [x_full, y_full, w_full, h_full]


def _xywh_norm_to_xyxy_abs(
    bboxes: list[list[float]],
    img_w: int,
    img_h: int,
) -> torch.Tensor:
    """
    Convert list of [x, y, w, h] normalised → Tensor[N, 4] xyxy absolute pixels.
    Required by torchvision.ops.nms.
    """
    if not bboxes:
        return torch.zeros((0, 4), dtype=torch.float32)

    t  = torch.tensor(bboxes, dtype=torch.float32)
    x1 = t[:, 0] * img_w
    y1 = t[:, 1] * img_h
    x2 = (t[:, 0] + t[:, 2]) * img_w
    y2 = (t[:, 1] + t[:, 3]) * img_h
    return torch.stack([x1, y1, x2, y2], dim=1)


def _build_fo_detections(
    tile_entries:           list[dict],
    img_w:                  int,
    img_h:                  int,
    tile_size:              int   = 640,
    confidence_threshold:   float = 0.0,
) -> list[fo.Detection]:
    """
    Build a flat list of full-image fo.Detection objects from all tile entries
    belonging to one sample.

    Bounding boxes in each entry are in RT-DETR format [cx, cy, w, h]
    normalised to the tile. They are converted to top-left [x, y, w, h]
    before remapping to full-image coordinates.
    """
    fo_detections = []

    for entry in tile_entries:
        raw_dets = entry.get("detections", [])
        if not raw_dets:
            continue

        x_start, y_start, tile_w, tile_h = _tile_pixel_extent(
            entry["y_stride"], entry["x_stride"],
            img_w, img_h, tile_size,
        )

        for det in raw_dets:
            conf = float(det["confidence"])
            if conf < confidence_threshold:
                continue

            cx, cy, w, h = det["bounding_box"]
            x, y, w, h   = _cx_cy_to_xywh(cx, cy, w, h)

            full_bbox = _remap_to_full_image(
                (x, y, w, h),
                x_start, y_start, tile_w, tile_h,
                img_w, img_h,
            )

            fo_detections.append(
                fo.Detection(
                    label      = "dugong",
                    confidence = conf,
                    bounding_box = full_bbox,
                    tile_source  = f"tile_{entry['y_stride']}_{entry['x_stride']}",
                    tile_type    = entry["tile_type"],
                )
            )

    return fo_detections


def _apply_nms(
    fo_detections: list[fo.Detection],
    img_w:         int,
    img_h:         int,
    iou_threshold: float,
) -> list[fo.Detection]:
    """Run torchvision NMS and return the surviving fo.Detection objects."""
    if not fo_detections:
        return []

    bboxes = [d.bounding_box for d in fo_detections]
    scores = torch.tensor(
        [d.confidence for d in fo_detections], dtype=torch.float32
    )
    boxes      = _xywh_norm_to_xyxy_abs(bboxes, img_w, img_h)
    keep       = tv_ops.nms(boxes, scores, iou_threshold)
    return [fo_detections[i] for i in keep.tolist()]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_from_json(
    dataset,
    json_path:            str | Path,
    field_name:           str   = None,
    tile_size:            int   = 640,
    confidence_threshold: float = 0.0,
    verbose:              bool  = True,
) -> str:
    """
    Reconstruct tile-level detections from a single inference JSON file into
    full-image FiftyOne Detections and save as {field_name}_raw.

    Parameters
    ----------
    dataset              : FiftyOne dataset or view (must have metadata loaded)
    json_path            : path to the consolidated inference JSON
    field_name           : FiftyOne field base name. If None, derived
                           automatically via derive_field_name().
    tile_size            : tile size used during inference (default 640)
    confidence_threshold : discard detections below this confidence before
                           saving (default 0.0 — keep all)
    verbose              : print progress

    Returns
    -------
    str — the raw field name written ('{field_name}_raw')
    """
    json_path  = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    if field_name is None:
        field_name = derive_field_name(json_path)

    raw_field = f"{field_name}_raw"

    _log(f"JSON       : {json_path.name}", verbose)
    _log(f"Field name : {raw_field}", verbose)

    # ── Load JSON ─────────────────────────────────────────────────────────────
    _log("Loading JSON ...", verbose)
    with open(json_path, "r") as f:
        entries = json.load(f)
    _log(f"Loaded {len(entries)} tile entries", verbose)

    # ── Group by sample stem ──────────────────────────────────────────────────
    grouped = _group_entries_by_sample(entries, verbose)

    # ── Build stem → sample lookup from dataset ───────────────────────────────
    _log("Building dataset stem lookup ...", verbose)
    stem_to_id: dict[str, str] = {}

    for sample in dataset.iter_samples(progress=verbose):
        # derive stem from filepath — same logic as tile parsing
        stem = Path(sample.filepath).stem
        stem_to_id[stem] = sample.id

    _log(f"Dataset has {len(stem_to_id)} samples", verbose)

    # ── Match grouped stems to dataset samples ────────────────────────────────
    matched   = set(grouped.keys()) & set(stem_to_id.keys())
    unmatched = set(grouped.keys()) - set(stem_to_id.keys())

    if unmatched:
        _log(
            f"{len(unmatched)} stems in JSON have no matching dataset sample "
            f"— first few: {sorted(unmatched)[:3]}",
            verbose, "warn",
        )
    _log(f"Matched {len(matched)} / {len(stem_to_id)} samples", verbose)

    # ── Write detections ──────────────────────────────────────────────────────
    n_det_total = 0

    view = dataset.select(
        [stem_to_id[s] for s in matched if s in stem_to_id]
    )

    for sample in view.iter_samples(autosave=True, progress=verbose):
        stem  = Path(sample.filepath).stem
        img_w = sample.metadata.width
        img_h = sample.metadata.height

        fo_dets = _build_fo_detections(
            grouped[stem], img_w, img_h,
            tile_size, confidence_threshold,
        )

        sample[raw_field] = fo.Detections(detections=fo_dets)
        n_det_total      += len(fo_dets)

    _log(
        f"reconstruct_from_json done — "
        f"field='{raw_field}'  total_detections={n_det_total}",
        verbose, "success",
    )
    return raw_field


def reconstruct_and_nms(
    dataset,
    json_path:            str | Path,
    field_name:           str   = None,
    tile_size:            int   = 640,
    iou_threshold:        float = 0.35,
    confidence_threshold: float = 0.0,
    verbose:              bool  = True,
) -> tuple[str, str]:
    """
    Reconstruct tile predictions AND apply Non-Maximum Suppression.

    Step 1 — reconstruct_from_json  → saves {field_name}_raw
    Step 2 — torchvision NMS        → saves {field_name}_nms

    Parameters
    ----------
    dataset              : FiftyOne dataset or view
    json_path            : path to the consolidated inference JSON
    field_name           : base field name (derived automatically if None)
    tile_size            : tile size used during inference (default 640)
    iou_threshold        : IoU threshold for NMS (default 0.35)
    confidence_threshold : pre-NMS confidence filter (default 0.0)
    verbose              : print progress

    Returns
    -------
    (raw_field, nms_field) — the two field names written
    """
    json_path = Path(json_path)

    if field_name is None:
        field_name = derive_field_name(json_path)

    raw_field = reconstruct_from_json(
        dataset              = dataset,
        json_path            = json_path,
        field_name           = field_name,
        tile_size            = tile_size,
        confidence_threshold = confidence_threshold,
        verbose              = verbose,
    )

    nms_field = f"{field_name}_nms"
    _log(
        f"Running NMS (iou_threshold={iou_threshold}) → '{nms_field}' ...",
        verbose,
    )

    n_before = 0
    n_after  = 0

    view = dataset.match(F(raw_field).exists())

    for sample in view.iter_samples(autosave=True, progress=verbose):
        img_w = sample.metadata.width
        img_h = sample.metadata.height

        raw = sample[raw_field]
        if raw is None or len(raw.detections) == 0:
            sample[nms_field] = fo.Detections(detections=[])
            continue

        before  = raw.detections
        after   = _apply_nms(before, img_w, img_h, iou_threshold)

        n_before += len(before)
        n_after  += len(after)

        sample[nms_field] = fo.Detections(detections=after)

    suppressed = n_before - n_after
    _log(
        f"NMS done — field='{nms_field}'  "
        f"before={n_before}  after={n_after}  "
        f"suppressed={suppressed} "
        f"({100 * suppressed / max(n_before, 1):.1f}%)",
        verbose, "success",
    )
    return raw_field, nms_field


def reconstruct_batch(
    dataset,
    json_paths:           list[str | Path],
    tile_size:            int   = 640,
    iou_threshold:        float = 0.35,
    confidence_threshold: float = 0.0,
    run_nms:              bool  = True,
    verbose:              bool  = True,
) -> list[dict]:
    """
    Reconstruct a list of inference JSON files into FiftyOne fields.

    Intended to be called directly from a notebook:

        results = reconstruct_batch(
            dataset,
            json_paths=[
                "/path/to/NWW_p5_aclr_seed0_rtdetr_0606_1045_test_predictions.json",
                "/path/to/NWW_p5_aclr_seed1_rtdetr_0606_1418_test_predictions.json",
                "/path/to/NWW_p10_random_seed0_rtdetr_0607_0900_test_predictions.json",
            ],
        )

    Parameters
    ----------
    dataset              : FiftyOne dataset or view
    json_paths           : list of paths to consolidated inference JSON files
    tile_size            : tile size used during inference (default 640)
    iou_threshold        : IoU threshold for NMS (default 0.35)
    confidence_threshold : pre-NMS confidence filter (default 0.0)
    run_nms              : if True (default) also runs NMS after reconstruction
    verbose              : print progress per file

    Returns
    -------
    list of dicts, one per JSON, with keys:
        json_path   : Path
        field_name  : str   (base name)
        raw_field   : str
        nms_field   : str | None   (None if run_nms=False)
        status      : "ok" | "error"
        error       : str | None
    """
    results = []

    for i, json_path in enumerate(json_paths):
        json_path  = Path(json_path)
        field_name = derive_field_name(json_path)

        print(f"\n{'─'*60}")
        print(f"[{i+1}/{len(json_paths)}]  {json_path.name}")
        print(f"  field_name : {field_name}")
        print(f"{'─'*60}")

        try:
            if run_nms:
                raw_field, nms_field = reconstruct_and_nms(
                    dataset              = dataset,
                    json_path            = json_path,
                    field_name           = field_name,
                    tile_size            = tile_size,
                    iou_threshold        = iou_threshold,
                    confidence_threshold = confidence_threshold,
                    verbose              = verbose,
                )
            else:
                raw_field = reconstruct_from_json(
                    dataset              = dataset,
                    json_path            = json_path,
                    field_name           = field_name,
                    tile_size            = tile_size,
                    confidence_threshold = confidence_threshold,
                    verbose              = verbose,
                )
                nms_field = None

            results.append({
                "json_path":  json_path,
                "field_name": field_name,
                "raw_field":  raw_field,
                "nms_field":  nms_field,
                "status":     "ok",
                "error":      None,
            })

        except Exception as exc:
            print(f"  [ERR ] {json_path.name} failed: {exc}")
            results.append({
                "json_path":  json_path,
                "field_name": field_name,
                "raw_field":  None,
                "nms_field":  None,
                "status":     "error",
                "error":      str(exc),
            })

    # ── Summary ───────────────────────────────────────────────────────────────
    n_ok  = sum(1 for r in results if r["status"] == "ok")
    n_err = sum(1 for r in results if r["status"] == "error")
    print(f"\n{'═'*60}")
    print(f"  Batch done — {n_ok} succeeded  {n_err} failed")
    print(f"{'═'*60}")

    for r in results:
        status = "✓" if r["status"] == "ok" else "✗"
        print(f"  {status}  {r['field_name']:<30}  raw={r['raw_field']}  "
              f"nms={r['nms_field']}")

    return results