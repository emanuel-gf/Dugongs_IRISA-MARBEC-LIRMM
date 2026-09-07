"""
Utilities to query, add, and remove inference prediction fields
on a FiftyOne dataset.

Designed to be called from a notebook:

    # query all seed0 p10 jsons
    paths = query_inference_jsons(
        inference_dir,
        seed="seed0",
        partition="p10",
    )

    # add them to the dataset
    add_prediction_fields(dataset, paths)

    # remove them later to save memory
    remove_prediction_fields(dataset, seed="seed0", partition="p10")
"""

from __future__ import annotations
from pathlib import Path
import fiftyone as fo


# ─────────────────────────────────────────────────────────────────────────────
# 1. Query inference JSONs
# ─────────────────────────────────────────────────────────────────────────────

def query_inference_jsons(
    inference_dir:  str | Path,
    seed:           str | list[str] | None = None,
    partition:      str | list[str] | None = None,
    method:         str | list[str] | None = None,
    extra_terms:    str | list[str] | None = None,
    pattern:        str = "*_test_predictions.json",
    verbose:        bool = True,
) -> list[Path]:
    """
    Scan inference_dir for prediction JSON files and return those matching
    ALL supplied filter terms (AND logic within each filter, OR logic within
    a list).

    Parameters
    ----------
    inference_dir : directory containing *_test_predictions.json files
    seed          : seed filter, e.g. "seed0" or ["seed0", "seed1"]
                    matches if ANY of the listed values is in the filename
    partition     : partition filter, e.g. "p10" or ["p5", "p10"]
                    matches if ANY of the listed values is in the filename
    method        : method filter, e.g. "aclr" or ["aclr", "random"]
                    matches if ANY of the listed values is in the filename
    extra_terms   : any additional substring(s) that must be present
    pattern       : glob pattern for JSON files (default *_test_predictions.json)
    verbose       : print summary

    Returns
    -------
    sorted list of matching Path objects

    Examples
    --------
    # all seed0 files
    query_inference_jsons(d, seed="seed0")

    # all p10 files across all seeds
    query_inference_jsons(d, partition="p10")

    # seed0 AND p10 only
    query_inference_jsons(d, seed="seed0", partition="p10")

    # seed0 AND (p5 or p10)
    query_inference_jsons(d, seed="seed0", partition=["p5", "p10"])

    # aclr only, all seeds and partitions
    query_inference_jsons(d, method="aclr")

    # seed1, p20, aclr
    query_inference_jsons(d, seed="seed1", partition="p20", method="aclr")
    """
    inference_dir = Path(inference_dir)
    if not inference_dir.exists():
        raise FileNotFoundError(f"inference_dir not found: {inference_dir}")

    all_jsons = sorted(inference_dir.glob(pattern))
    if verbose:
        print(f"  Found {len(all_jsons)} JSON files in '{inference_dir.name}'")

    def _to_list(val):
        if val is None:
            return None
        return [val] if isinstance(val, str) else list(val)

    seeds       = _to_list(seed)
    partitions  = _to_list(partition)
    methods     = _to_list(method)
    extras      = _to_list(extra_terms)

    def _matches(path: Path) -> bool:
        name = path.stem   # e.g. NWW_p10_aclr_seed0_rtdetr_0606_1045_test_predictions
        # strip _test_predictions suffix for cleaner matching
        name = name.replace("_test_predictions", "")

        # seed filter — ANY of the listed seeds must appear
        if seeds and not any(s in name for s in seeds):
            return False

        # partition filter — ANY of the listed partitions must appear
        # use word-boundary style check: _p10_ to avoid p10 matching p100
        if partitions:
            def _part_match(p, n):
                # match _p10_ or _p10 at end of segment
                return f"_{p}_" in f"_{n}_"
            if not any(_part_match(p, name) for p in partitions):
                return False

        # method filter — ANY of the listed methods must appear
        if methods and not any(m in name for m in methods):
            return False

        # extra terms — ALL must appear
        if extras and not all(e in name for e in extras):
            return False

        return True

    matched = sorted([p for p in all_jsons if _matches(p)])

    if verbose:
        print(f"  Matched {len(matched)} / {len(all_jsons)} files")
        for p in matched:
            print(f"    {p.name}")

    return matched


# ─────────────────────────────────────────────────────────────────────────────
# 2. Add prediction fields
# ─────────────────────────────────────────────────────────────────────────────

def add_prediction_fields(
    dataset,
    json_paths:             list[str | Path],
    run_nms:                bool  = True,
    iou_threshold:          float = 0.35,
    confidence_threshold:   float = 0.05,
    tile_size:              int   = 640,
    skip_existing:          bool  = True,
    verbose:                bool  = True,
) -> list[dict]:
    """
    Add prediction fields to the dataset from a list of inference JSON paths.
    Thin wrapper around reconstruct_batch that adds a skip_existing guard.

    Parameters
    ----------
    dataset              : FiftyOne dataset
    json_paths           : list of JSON paths, e.g. from query_inference_jsons()
    run_nms              : also run NMS and store {field}_nms (default True)
    iou_threshold        : NMS IoU threshold (default 0.35)
    confidence_threshold : pre-NMS confidence filter (default 0.05)
    tile_size            : tile size used during inference (default 640)
    skip_existing        : if True, skip any JSON whose derived field name
                           already exists in the dataset schema (default True)
    verbose              : print progress

    Returns
    -------
    list of result dicts from reconstruct_batch
    """
    from src.reconstruct import reconstruct_batch, derive_field_name

    existing_fields = set(dataset.get_field_schema().keys())

    to_process = []
    skipped    = []

    for p in json_paths:
        p          = Path(p)
        field_name = derive_field_name(p)
        raw_field  = f"{field_name}_raw"
        nms_field  = f"{field_name}_nms"

        if skip_existing and (raw_field in existing_fields or nms_field in existing_fields):
            skipped.append(p.name)
            continue

        to_process.append(p)

    if skipped and verbose:
        print(f"\n  Skipping {len(skipped)} already-existing fields:")
        for s in skipped:
            print(f"    {s}")

    if not to_process:
        print("  Nothing to process — all fields already exist.")
        return []

    return reconstruct_batch(
        dataset              = dataset,
        json_paths           = to_process,
        run_nms              = run_nms,
        iou_threshold        = iou_threshold,
        confidence_threshold = confidence_threshold,
        tile_size            = tile_size,
        verbose              = verbose,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Remove prediction fields
# ─────────────────────────────────────────────────────────────────────────────

def remove_prediction_fields(
    dataset,
    seed:           str | list[str] | None = None,
    partition:      str | list[str] | None = None,
    method:         str | list[str] | None = None,
    extra_terms:    str | list[str] | None = None,
    include_raw:    bool = True,
    include_nms:    bool = True,
    include_eval:   bool = False,
    keep_baseline:   bool = True,
    dry_run:        bool = True,
    verbose:        bool = True,
) -> list[str]:
    """
    Remove prediction fields from the dataset schema matching the given filters.

    By default runs in dry_run=True mode — prints what would be deleted
    without actually deleting anything. Set dry_run=False to commit.

    Parameters
    ----------
    dataset      : FiftyOne dataset
    seed         : seed filter e.g. "seed0" or ["seed0", "seed1"]
    partition    : partition filter e.g. "p10" or ["p5", "p10"]
    method       : method filter e.g. "aclr" or ["aclr", "random"]
    extra_terms  : additional substring(s) that must all be present
    include_raw  : remove _raw fields (default True)
    include_nms  : remove _nms fields (default True)
    keep_baseline: where to keep this field of not
    include_eval : also remove FiftyOne eval bookkeeping fields
                   (_tp, _fp, _fn, _tn, _iou) for matching eval keys
                   (default False — be careful with this)
    dry_run      : if True (default), only print — do not delete
    verbose      : print progress

    Returns
    -------
    list of field names that were (or would be) deleted

    Examples
    --------
    # preview what seed0 fields would be removed
    remove_prediction_fields(dataset, seed="seed0", dry_run=True)

    # actually remove seed0 raw fields only
    remove_prediction_fields(
        dataset, seed="seed0",
        include_raw=True, include_nms=False,
        dry_run=False,
    )

    # remove all p5 fields across all seeds and methods
    remove_prediction_fields(dataset, partition="p5", dry_run=False)
    """
    def _to_list(val):
        if val is None:
            return None
        return [val] if isinstance(val, str) else list(val)

    seeds      = _to_list(seed)
    partitions = _to_list(partition)
    methods    = _to_list(method)
    extras     = _to_list(extra_terms)

    _EVAL_SUFFIXES = ("_tp", "_fp", "_fn", "_tn", "_iou")

    all_fields = list(dataset.get_field_schema().keys())

    # fields we must never delete regardless of filters
    _PROTECTED = {
        "id", "filepath", "tags", "metadata",
        "created_at", "last_modified_at",
        "ground_truth", "audit", "roi_grid",
        "region", "mission_name", "parent_name",
        "m_flight", "flight_mission", "name_plot",
        "full_embeddings", "full_emb_umap_dinov3", "v2_full_emb_umap_dinov3",
        "uniqueness_score", "representativeness_score",
        "cluster_label", "cluster_label_8",
        "uniqueness_score_per_cluster", "soft_coverage_score",
        "thumbnail_path",
    }

    def _field_matches(field_name: str) -> bool:
        # only touch prediction fields
        is_raw  = field_name.endswith("_raw")
        is_nms  = field_name.endswith("_nms")
        is_eval = any(field_name.endswith(s) for s in _EVAL_SUFFIXES)

        if not (is_raw or is_nms or is_eval):
            return False
        if is_raw  and not include_raw:
            return False
        if is_nms  and not include_nms:
            return False
        if is_eval and not include_eval:
            return False
        if field_name in _PROTECTED:
            return False

        name = field_name

        # seed filter — ANY must match
        if seeds and not any(s in name for s in seeds):
            return False

        # partition filter — word-boundary match
        if partitions:
            def _part_match(p, n):
                return f"_{p}_" in f"_{n}_"
            if not any(_part_match(p, name) for p in partitions):
                return False

        # method filter — ANY must match
        if methods and not any(m in name for m in methods):
            return False

        # extra terms — ALL must match
        if extras and not all(e in name for e in extras):
            return False

        return True

    to_delete = sorted([f for f in all_fields if _field_matches(f)])

    # whether keep the baseline metrics or not
    if keep_baseline:
        to_delete = [f for f in to_delete if "baseline" not in f]

    if not to_delete:
        print("  No matching fields found.")
        return []

    if verbose or dry_run:
        tag = "DRY RUN —" if dry_run else "DELETING"
        print(f"\n  {tag} {len(to_delete)} fields:")
        for f in to_delete:
            print(f"    {f}")

    if dry_run:
        print(f"\n  Dry run complete — pass dry_run=False to commit deletion.")
        return to_delete

    # ── Commit deletion ───────────────────────────────────────────────────────
    n_ok  = 0
    n_err = 0

    for field_name in to_delete:
        try:
            dataset.delete_sample_field(field_name)
            n_ok += 1
            if verbose:
                print(f"  deleted  {field_name}")
        except Exception as e:
            n_err += 1
            print(f"  ERROR deleting '{field_name}': {e}")

    dataset.save()
    print(f"\n  Done — deleted {n_ok} fields  ({n_err} errors)")
    return to_delete