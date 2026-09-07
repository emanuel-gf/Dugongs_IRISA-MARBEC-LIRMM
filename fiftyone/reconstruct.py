"""
reconstruct_predictions.py
──────────────────────────
Utilities to reconstruct tile-level inference results back into full-image
FiftyOne Detections fields, with optional Non-Maximum Suppression.

Two public functions
--------------------
reconstruct_predictions(dataset, predictions_dir, field_name)
    → saves {field_name}_raw  (all tile detections, no NMS)

reconstruct_and_nms(dataset, predictions_dir, field_name, iou_threshold=0.35)
    → calls reconstruct_predictions first
    → then runs per-image NMS
    → saves {field_name}_nms
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import torch
import torchvision.ops as tv_ops
import fiftyone as fo
from fiftyone import ViewField as F
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

TILE_TYPE_MAP = {
    "n": "background",
    "p": "positive_tile",
}


def _parse_tile_stem(stem: str) -> tuple[str, int, int, str] | None:
    """
    Parse a tile JSON stem into (sample_stem, y_stride, x_stride, tile_type).

    Example
    -------
    'GH024197-60e713a81ab1d_1223__tile_0_1620_n'
    → ('GH024197-60e713a81ab1d_1223', 0, 1620, 'background')
    """
    m = stem.split('__')
    if m is None:
        return None
    sample_stem = m[0]
    y_stride    = int(m[1].split('_')[1])
    x_stride    = int(m[1].split('_')[2])
    tile_type   = TILE_TYPE_MAP[m[1].split('_')[3]]
    return sample_stem, y_stride, x_stride, tile_type


def _group_jsons_by_sample(predictions_dir: Path) -> dict[str, list[dict]]:
    """
    Walk predictions_dir, parse every *.json filename, and group by
    sample stem_filepath.

    Returns
    -------
    {
      'GH024197-60e713a81ab1d_1223': [
          {'path': Path(...), 'y_stride': 0, 'x_stride': 1620,
           'tile_type': 'background', 'tile_label': 'tile_0_1620'},
          ...
      ],
      ...
    }
    """
    grouped: dict[str, list[dict]] = defaultdict(list)

    json_files = list(predictions_dir.glob("*.json"))
    if not json_files:
        logger.warning(f"No JSON files found in {predictions_dir}")
        return grouped

    for json_path in json_files:
        parsed = _parse_tile_stem(Path(json_path).stem)
        if parsed is None:
            logger.warning(f"Could not parse tile filename, skipping: {json_path.name}")
            continue
        sample_stem, y_stride, x_stride, tile_type = parsed
        grouped[sample_stem].append({
            "path":        json_path,
            "y_stride":    y_stride,
            "x_stride":    x_stride,
            "tile_type":   tile_type,
            "tile_label":  f"tile_{y_stride}_{x_stride}",
        })

    logger.info(f"Grouped {len(json_files)} JSONs → {len(grouped)} unique sample stems")
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
    add_roi_grid's clamping logic exactly.
    """
    x_end = min(x_stride + tile_size, img_w)
    y_end = min(y_stride + tile_size, img_h)

    x_start = max(x_end - tile_size, 0)
    y_start = max(y_end - tile_size, 0)

    tile_w = x_end - x_start
    tile_h = y_end - y_start

    return x_start, y_start, tile_w, tile_h


def _remap_to_full_image(
    tile_norm_bbox: list[float],   # [x, y, w, h] normalized inside the tile
    x_start: int,
    y_start: int,
    tile_w: int,
    tile_h: int,
    img_w: int,
    img_h: int,
) -> list[float]:
    """
    Convert tile-local normalized [x, y, w, h]
    → full-image normalized [x, y, w, h].
    """
    xn, yn, wn, hn = tile_norm_bbox

    x_full = (xn * tile_w + x_start) / img_w
    y_full = (yn * tile_h + y_start) / img_h
    w_full = wn * tile_w / img_w
    h_full = hn * tile_h / img_h

    # Clamp to [0, 1] for safety
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
    Convert list of [x, y, w, h] normalized → Tensor[N, 4] xyxy absolute pixels.
    Required format for torchvision.ops.nms.
    """
    if not bboxes:
        return torch.zeros((0, 4), dtype=torch.float32)

    t = torch.tensor(bboxes, dtype=torch.float32)  # [N, 4]
    x1 = t[:, 0] * img_w
    y1 = t[:, 1] * img_h
    x2 = (t[:, 0] + t[:, 2]) * img_w
    y2 = (t[:, 1] + t[:, 3]) * img_h
    return torch.stack([x1, y1, x2, y2], dim=1)


def _build_fo_detections(
    tile_entries: list[dict],
    img_w: int,
    img_h: int,
    tile_size: int = 640,
) -> list[fo.Detection]:
    """
    Read all tile JSONs for one sample and return a flat list of fo.Detection
    objects remapped to full-image normalized coords.
    """
    fo_detections = []

    for entry in tile_entries:
        json_data = json.loads(entry["path"].read_text())
        raw_detections = json_data.get("detections", [])

        if not raw_detections:
            continue

        x_start, y_start, tile_w, tile_h = _tile_pixel_extent(
            entry["y_stride"], entry["x_stride"],
            img_w, img_h, tile_size,
        )

        for det in raw_detections:
            full_bbox = _remap_to_full_image(
                det["bounding_box"],
                x_start, y_start, tile_w, tile_h,
                img_w, img_h,
            )
            fo_detections.append(
                fo.Detection(
                    label="Dugong",
                    confidence=det["confidence"],
                    bounding_box=full_bbox,
                    tile_source=entry["tile_label"],
                    tile_type=entry["tile_type"],
                )
            )

    return fo_detections


def _apply_nms(
    fo_detections: list[fo.Detection],
    img_w: int,
    img_h: int,
    iou_threshold: float,
) -> list[fo.Detection]:
    """
    Run torchvision NMS on a flat list of fo.Detection objects.
    Returns the subset of detections that survive NMS.
    """
    if not fo_detections:
        return []

    bboxes  = [d.bounding_box for d in fo_detections]
    scores  = torch.tensor([d.confidence for d in fo_detections], dtype=torch.float32)
    boxes   = _xywh_norm_to_xyxy_abs(bboxes, img_w, img_h)

    keep_indices = tv_ops.nms(boxes, scores, iou_threshold)
    return [fo_detections[i] for i in keep_indices.tolist()]


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_predictions(
    dataset: fo.Dataset,
    predictions_dir: str | Path,
    field_name: str,
    tile_size: int = 640,
) -> None:
    """
    Reconstruct tile-level inference JSONs into full-image FiftyOne Detections.

    Saves results into  {field_name}_raw  on each sample.
    Includes ALL detections from both positive (_s) and negative (_n) tiles.

    Parameters
    ----------
    dataset         : fo.Dataset
    predictions_dir : folder containing tile JSON files
    field_name      : base name for the output field (suffix _raw is added)
    tile_size       : tile size used during inference (default 640)
    """
    predictions_dir = Path(predictions_dir)
    raw_field       = f"{field_name}_raw"

    if not predictions_dir.exists():
        raise FileNotFoundError(f"predictions_dir not found: {predictions_dir}")

    # Build stem → sample lookup
    #stem_to_sample = {s.stem_filepath: s for s in dataset.iter_samples()}
    stem_list = dataset.values('stem_filepath')

    # Group JSONs by sample stem
    grouped = _group_jsons_by_sample(predictions_dir)

    # Warn about JSONs with no matching sample
    for stem in grouped:
        if stem not in stem_list:
            logger.warning(f"No matching sample found for stem: '{stem}' — skipping")

    matched_stems = set(grouped.keys()) & set(stem_list)
    logger.info(f"Matched {len(matched_stems)} / {len(stem_list)} samples to JSONs")

    # filter the dataset and iterate over the view
    view_dataset = dataset.match(F("stem_filepath").is_in(list(grouped.keys())))
    for sample in view_dataset.iter_samples():
        stem = sample['stem_filepath']
        img_w = sample.metadata.width
        img_h = sample.metadata.height
        has_gt = (
            sample.ground_truth is not None
            and len(sample.ground_truth.detections) > 0
        )

        if stem not in grouped:
            # No JSONs found for this sample
            if has_gt:
                logger.warning(
                    f"Sample '{stem}' has {len(sample.ground_truth.detections)} "
                    f"ground-truth dugong(s) but NO tile JSONs found — "
                    f"setting {raw_field}=fo.Detections(detections=[]) (counts as FN)"
                )
            sample[raw_field] = fo.Detections(detections=[])
            sample.save()
            continue

        fo_detections = _build_fo_detections(
            grouped[stem], img_w, img_h, tile_size,
        )

        sample[raw_field] = fo.Detections(detections=fo_detections)
        sample.save()

    logger.success(
        f"reconstruct_predictions complete → field '{raw_field}' "
    )


def reconstruct_and_nms(
    dataset: fo.Dataset,
    field_name: str,
    predictions_dir: str | Path = None,
    raw_field: str | None = None,  # NEW: Optional raw_field to skip reconstruction
    iou_threshold: float = 0.35,
    tile_size: int = 640,
) -> None:
    """
    Reconstruct tile predictions AND apply Non-Maximum Suppression.

    If `raw_field` is provided, it skips reconstruction and directly applies NMS to the existing field.
    Otherwise, it calls reconstruct_predictions to create the raw field first.

    Step 1 — (if raw_field is None) calls reconstruct_predictions → saves {field_name}_raw
    Step 2 — runs torchvision.ops.nms per image → saves {field_name}_nms

    NMS is run in full-image xyxy absolute pixel space, then results are
    stored back as normalized xywh in FiftyOne.

    Parameters
    ----------
    dataset         : fo.Dataset
    predictions_dir : folder containing tile JSON files (ignored if raw_field is provided)
    field_name      : base name; _raw and _nms suffixes are added automatically
    raw_field       : optional name of an existing raw field to use (skips reconstruction if provided)
    iou_threshold   : IoU threshold for NMS (default 0.35, tuned for small objects)
    tile_size       : tile size used during inference (default 640, ignored if raw_field is provided)
    """
    nms_field = f"{field_name}_nms"

    # Step 1: Reconstruct raw detections if raw_field is not provided
    if raw_field is None:
        assert predictions_dir is not None, f"Please encharge to pass the predictions"
        raw_field = f"{field_name}_raw"
        logger.info(f"Reconstructing raw detections from '{predictions_dir}'")
        reconstruct_predictions(
            dataset,
            predictions_dir,
            field_name,
            tile_size
        )
    else:
        logger.info(f"Using existing raw field: '{raw_field}' (skipping reconstruction)")
        # Validate that the raw_field exists in the dataset
        if raw_field not in dataset.get_field_schema():
            raise ValueError(f"raw_field '{raw_field}' does not exist in the dataset")

    logger.info(f"Running NMS (iou_threshold={iou_threshold}) → field '{nms_field}'")

    n_before = 0
    n_after  = 0

    # Create a view of the dataset that only includes samples with the raw_field
    # This is more efficient than iterating through all samples
    if raw_field in dataset.get_field_schema():
        view = dataset.match(F(raw_field).exists())
    else:
        view = dataset

    for sample in view.iter_samples(autosave=True, 
                                    progress=True):
        img_w = sample.metadata.width
        img_h = sample.metadata.height

        raw = sample[raw_field]
        if raw is None or len(raw.detections) == 0:
            sample[nms_field] = fo.Detections(detections=[])
            # Store iou threshold used at sample level for traceability
            #sample[f"{field_name}_iou_threshold"] = iou_threshold
            continue

        before = raw.detections
        after  = _apply_nms(before, img_w, img_h, iou_threshold)

        n_before += len(before)
        n_after  += len(after)

        sample[nms_field] = fo.Detections(detections=after)
        #sample[f"{field_name}_iou_threshold"] = iou_threshold

    logger.success(
        f"reconstruct_and_nms complete → field '{nms_field}' saved | "
        f"detections before NMS: {n_before} | after NMS: {n_after} | "
        f"suppressed: {n_before - n_after} "
        f"({100 * (n_before - n_after) / max(n_before, 1):.1f}%)"
    )