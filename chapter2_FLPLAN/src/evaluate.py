"""
evaluate_flplan_predictions.py
===============================
Evaluation helpers for the FLPLAN fine-tuning experiment.

Designed to be called from a notebook in this order:

    1.  get_fields_for_seed(dataset, seed_term)
            → filtered field lists (raw / nms / clean)

    2.  select_evaluation_list(all_op, nms_or_raw, baseline_field, seed_term)
            → list of {pred_field, eval_key} dicts

    3.  run_evaluations(dataset_view, eval_list, gt_field, iou)
            → {eval_key: COCODetectionResults}

    4.  extract_all_metrics(results_dict, seed)
            → pd.DataFrame  one row per model

Supports both old-style field names (NWW_ACLR_partition_10_SEED63_...) and
new-style field names (NWW_p5_aclr_seed0_nms) transparently via
create_eval_key_name().
"""

from __future__ import annotations

from collections import Counter
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Field name → eval key
# ─────────────────────────────────────────────────────────────────────────────

def create_eval_key_name(field_name: str) -> str:
    """
    Derive a short, readable eval key from a FiftyOne prediction field name.

    Supports two field name formats:

    NEW format  (from reconstruct_flplan_predictions)
    --------------------------------------------------
    Pattern : NWW_{partition}_{method}_{seed}_{suffix}
    Examples:
        NWW_p5_aclr_seed0_nms    → aclr_p5_seed0_nms
        NWW_p10_random_seed1_raw → random_p10_seed1_raw
        NWW_p100_aclr_seed2_nms  → aclr_p100_seed2_nms

    OLD format  (legacy pipeline, kept for backward compatibility)
    --------------------------------------------------------------
    Pattern : NWW[_ACLR]_partition_{n}_{SEED}_{...}_{suffix}
    Examples:
        NWW_ACLR_partition_10_SEED63_augm_0510_2124_nms → ACLR_p10_SEED63_nms
        NWW_partition_10_SEED63_augm_0510_2120_nms       → p10_SEED63_nms

    Baseline fields (neither pattern matches) are returned unchanged.
    """
    splits = field_name.split("_")

    # ── NEW format detection ──────────────────────────────────────────────────
    # Heuristic: token[1] starts with 'p' followed by digits (e.g. 'p5', 'p10')
    # and token[2] is 'aclr' or 'random'
    if (
        len(splits) >= 5
        and splits[0] == "NWW"
        and splits[1].startswith("p")
        and splits[1][1:].isdigit()
        and splits[2] in ("aclr", "random")
    ):
        partition = splits[1]          # e.g. 'p5'
        method    = splits[2]          # 'aclr' | 'random'
        seed      = splits[3]          # e.g. 'seed0'
        suffix    = splits[-1]         # 'raw' | 'nms'
        return f"{method}_{partition}_{seed}_{suffix}"

    # ── OLD format detection ──────────────────────────────────────────────────
    # NWW_ACLR_partition_{n}_{SEED}_..._{suffix}
    if "ACLR" in splits and "partition" in splits:
        part_idx  = splits.index("partition")
        partition = splits[part_idx + 1]
        seed      = splits[part_idx + 2]   # e.g. 'SEED63'
        suffix    = splits[-1]
        return f"ACLR_p{partition}_{seed}_{suffix}"

    # NWW_partition_{n}_{SEED}_..._{suffix}
    if "partition" in splits:
        part_idx  = splits.index("partition")
        partition = splits[part_idx + 1]
        seed      = splits[part_idx + 2]
        suffix    = splits[-1]
        return f"p{partition}_{seed}_{suffix}"

    # ── Fallback — return as-is (baseline or unrecognised) ───────────────────
    return field_name


# ─────────────────────────────────────────────────────────────────────────────
# Field filtering
# ─────────────────────────────────────────────────────────────────────────────

_EVAL_SUFFIXES = ("tp", "fp", "fn", "tn", "iou", "eval")


def get_fields_for_seed(
    dataset,
    seed_term:  str,
    verbose:    bool = True,
) -> dict[str, list[str]]:
    """
    Return all prediction fields belonging to one seed, split into
    raw / nms / clean (excluding FiftyOne evaluation book-keeping fields).

    Parameters
    ----------
    dataset   : FiftyOne dataset or view
    seed_term : substring that identifies the seed in field names,
                e.g. 'seed0', 'seed1', 'SEED63'
    verbose   : print counts

    Returns
    -------
    dict with keys:
        'all'   — every field containing seed_term
        'raw'   — fields ending in _raw
        'nms'   — fields ending in _nms
        'clean' — all fields excluding FiftyOne eval book-keeping fields
                  (tp / fp / fn / tn / iou / eval suffixes)
    """
    all_fields = list(dataset.get_field_schema().keys())
    seed_fields = [f for f in all_fields if seed_term in f]

    raw   = [f for f in seed_fields if f.endswith("_raw")]
    nms   = [f for f in seed_fields if f.endswith("_nms")]
    clean = [
        f for f in seed_fields
        if not any(f.endswith(f"_{s}") for s in _EVAL_SUFFIXES)
    ]

    if verbose:
        print(f"seed_term='{seed_term}'  "
              f"total={len(seed_fields)}  "
              f"raw={len(raw)}  nms={len(nms)}  clean={len(clean)}")

    return {"all": seed_fields, "raw": raw, "nms": nms, "clean": clean}


def filter_ops(
    all_op:      list[str],
    nms_or_raw:  str  = None,
    seed_term:   str  = None,
) -> list[str]:
    """
    Filter a list of field names by suffix (raw/nms) and/or seed substring.
    """
    def _match(op: str) -> bool:
        tokens = op.split("_")
        if nms_or_raw and nms_or_raw not in tokens:
            return False
        if seed_term and seed_term not in op:
            return False
        return True

    return [op for op in all_op if _match(op)]


def select_evaluation_list(
    all_op:         list[str],
    nms_or_raw:     str  = "nms",
    baseline_field: str  = None,
    seed_term:      str  = None,
    deduplicate:    bool = True,
    verbose:        bool = True,
) -> list[dict]:
    """
    Build the list of {pred_field, eval_key} dicts ready to pass to
    run_evaluations().

    Parameters
    ----------
    all_op          : list of prediction field names (from get_fields_for_seed
                      or dataset.get_field_schema())
    nms_or_raw      : 'nms' or 'raw' — which suffix to select
    baseline_field  : optional explicit baseline field to append. The baseline
                      is appended last and given eval_key
                      'baseline_{seed_term}_{nms_or_raw}'.
    seed_term       : seed substring filter passed to filter_ops
    deduplicate     : remove duplicate pred_fields, preferring the entry
                      whose eval_key contains 'baseline' (same logic as the
                      notebook dedup block). Default True.
    verbose         : print the final list

    Returns
    -------
    list of {'pred_field': str, 'eval_key': str}
    """
    nms_or_raw = nms_or_raw.strip().lower()
    if nms_or_raw not in ("raw", "nms"):
        raise ValueError(f"nms_or_raw must be 'raw' or 'nms', got '{nms_or_raw}'")

    filtered = filter_ops(all_op, nms_or_raw=nms_or_raw, seed_term=seed_term)

    eval_list = [
        {"pred_field": f, "eval_key": create_eval_key_name(f)}
        for f in filtered
    ]

    # ── Append baseline ───────────────────────────────────────────────────────
    if baseline_field is not None:
        baseline_key = (
            f"baseline_{seed_term}_{nms_or_raw}"
            if seed_term
            else f"baseline_{nms_or_raw}"
        )
        eval_list.append({
            "pred_field": baseline_field,
            "eval_key":   baseline_key,
        })

    # ── Deduplication ─────────────────────────────────────────────────────────
    if deduplicate:
        seen:   dict[str, dict] = {}
        dups:   list[str]       = []

        counts = Counter(d["pred_field"] for d in eval_list)
        dups   = [k for k, v in counts.items() if v > 1]
        if dups and verbose:
            print(f"  WARNING: duplicate pred_fields detected: {dups}")

        for item in eval_list:
            key = item["pred_field"]
            if key not in seen:
                seen[key] = item
            else:
                # prefer the baseline entry if one of them is a baseline
                if "baseline" in item["eval_key"]:
                    seen[key] = item
                # else keep first occurrence

        eval_list = list(seen.values())

    if verbose:
        print(f"\nselect_evaluation_list → {len(eval_list)} entries "
              f"(nms_or_raw='{nms_or_raw}'  seed='{seed_term}')")
        for e in eval_list:
            print(f"  {e['pred_field']:<55} → {e['eval_key']}")

    return eval_list


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluations(
    dataset_view,
    eval_list:  list[dict],
    gt_field:   str   = "ground_truth",
    iou:        float = 0.5,
    verbose:    bool  = True,
) -> dict:
    """
    Run FiftyOne COCO-style evaluate_detections for each entry in eval_list.

    Parameters
    ----------
    dataset_view : FiftyOne dataset or view (typically the test split)
    eval_list    : list of {'pred_field': str, 'eval_key': str} dicts
                   as returned by select_evaluation_list()
    gt_field     : ground-truth detections field (default 'ground_truth')
    iou          : IoU threshold for matching (default 0.5)
    verbose      : print progress

    Returns
    -------
    dict  {eval_key: COCODetectionResults}
    """
    results = {}

    for i, m in enumerate(eval_list):
        pred_field = m["pred_field"]
        eval_key   = m["eval_key"]

        if verbose:
            print(f"  [{i+1}/{len(eval_list)}]  {pred_field}  →  {eval_key}")

        try:
            results[eval_key] = dataset_view.evaluate_detections(
                pred_field,
                gt_field    = gt_field,
                eval_key    = eval_key,
                method      = "coco",
                iou         = iou,
                classwise   = False,
                compute_mAP = True,
            )
        except Exception as exc:
            print(f"    ERROR evaluating '{pred_field}': {exc}")

    if verbose:
        print(f"\n  run_evaluations done — {len(results)} / {len(eval_list)} succeeded")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Threshold sweep + metrics extraction
# ─────────────────────────────────────────────────────────────────────────────

def sweep_thresholds(
    res,
    iou_thresh:    float = 0.5,
    n_thresholds:  int   = 200,
) -> dict:
    """
    Sweep confidence thresholds and return metrics at the F1-optimal point.

    Parameters
    ----------
    res           : COCODetectionResults from evaluate_detections()
    iou_thresh    : IoU threshold for TP matching (default 0.5)
    n_thresholds  : number of threshold steps in [0.01, 0.99] (default 200)

    Returns
    -------
    dict with keys: threshold, tp, fp, fn, precision, recall, f1, n_gt
    """
    ytrue = np.array(res.ytrue)
    ypred = np.array(res.ypred)
    confs = np.array(
        [v if v is not None else 0.0 for v in res.confs], dtype=float
    )
    ious  = np.array(
        [v if v is not None else 0.0 for v in res.ious],  dtype=float
    )

    n_gt = int(
        (ypred == "(none)").sum()
        + ((ytrue == "Dugong") & (ypred == "Dugong")).sum()
    )
    pred_mask  = ypred != "(none)"
    thresholds = np.linspace(0.01, 0.99, n_thresholds)

    best: dict = {"f1": -1}

    for thresh in thresholds:
        mask = pred_mask & (confs >= thresh)
        tp   = int(((ytrue[mask] == "Dugong") & (ious[mask] >= iou_thresh)).sum())
        fp   = int((ytrue[mask] == "(none)").sum())
        fn   = int(n_gt - tp)

        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)

        if f1 > best["f1"]:
            best = dict(
                threshold = thresh,
                tp        = tp,
                fp        = fp,
                fn        = fn,
                precision = precision,
                recall    = recall,
                f1        = f1,
                n_gt      = n_gt,
            )

    return best


def extract_all_metrics(
    results_dict: dict,
    seed,
    iou_thresh:   float = 0.5,
    n_thresholds: int   = 200,
) -> pd.DataFrame:
    """
    Compute F1-optimal metrics for every model in results_dict and return
    a tidy DataFrame with one row per model.

    Parameters
    ----------
    results_dict  : {eval_key: COCODetectionResults} as returned by
                    run_evaluations()
    seed          : seed identifier written into the 'seed' column
                    (int, str — whatever makes sense for your experiment)
    iou_thresh    : forwarded to sweep_thresholds (default 0.5)
    n_thresholds  : forwarded to sweep_thresholds (default 200)

    Returns
    -------
    pd.DataFrame with columns:
        seed, eval_key, mAP, mAR,
        threshold, f1, precision, recall, tp, fp, fn, n_gt
    """
    rows = []

    for eval_key, res in results_dict.items():
        try:
            best = sweep_thresholds(res, iou_thresh, n_thresholds)
            rows.append({
                "seed":      seed,
                "eval_key":  eval_key,
                "mAP":       res.mAP(),
                "mAR":       res.mAR(),
                "threshold": best["threshold"],
                "f1":        best["f1"],
                "precision": best["precision"],
                "recall":    best["recall"],
                "tp":        best["tp"],
                "fp":        best["fp"],
                "fn":        best["fn"],
                "n_gt":      best["n_gt"],
            })
        except Exception as exc:
            print(f"  WARNING: extract_all_metrics failed for '{eval_key}': {exc}")

    print(f"Seed {seed} done — {len(rows)} / {len(results_dict)} models processed")
    return pd.DataFrame(rows)