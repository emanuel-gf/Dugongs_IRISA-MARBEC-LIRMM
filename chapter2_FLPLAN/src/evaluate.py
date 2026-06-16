"""
evaluate_flplan_predictions.py
===============================
Standardised evaluation pipeline for the FLPLAN domain-adaptation experiment.

Public API
----------
    derive_field_name(json_path)
    get_fields_for_seed(dataset, seed_term)
    filter_ops(all_op, nms_or_raw, seed_term)
    select_evaluation_list(all_op, nms_or_raw, baseline_field, seed_term)
    run_evaluations(dataset_view, eval_list, gt_field, iou)

    sweep_thresholds(res, method, iou_thresh, custom_threshold, n_thresholds)
        → ThresholdSweepResult (dict)

    plot_sweep(res, method, iou_thresh, n_thresholds, save_path)
        → matplotlib Figure   [diagnostic: Prints 3 subplots each
                                    one of them with a f1-optimal threshold
                                     or equal error rate.
                                       — call before extract_all_metrics]

    extract_all_metrics(results_dict, seed, method, iou_thresh, n_thresholds)
        → pd.DataFrame        [one row per model]

Column contract
---------------
Every DataFrame produced by extract_all_metrics has these columns:

    Identification
        seed, eval_key

    COCO metrics (threshold-free)
        mAP, mAR

    F1-optimal operating point
        f1_threshold, f1, precision_f1, recall_f1, tp_f1, fp_f1, fn_f1

    EER operating point
        eer_threshold, fn_rate_eer, fp_rate_eer,
        precision_eer, recall_eer, tp_eer, fp_eer, fn_eer

    Shared
        n_gt, iou_thresh, positive_label

    Alias columns  (point to whichever method= was chosen)
        threshold, f1_score, precision, recall, tp, fp, fn
        (named f1_score to avoid shadowing the f1 column from the F1-opt block)

Existing plotting functions that read df["threshold"], df["f1"], df["precision"],
df["recall"], df["tp"], df["fp"], df["fn"] continue to work unchanged.
"""

from __future__ import annotations

import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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


def _infer_positive_labels(ytrue: np.ndarray) -> set:
    """
    Derive the set of positive class labels from a ytrue array.
    Excludes the FiftyOne unmatched-prediction sentinel '(none)'.
    Raises if no positive labels are found.
    """
    labels = set(np.unique(ytrue)) - {"(none)"}
    if not labels:
        raise ValueError(
            "No positive labels found in ytrue. "
            "Check that evaluate_detections() was run correctly."
        )
    return labels


def _is_positive(arr: np.ndarray, positive_labels: set) -> np.ndarray:
    """Boolean mask: True where arr contains a positive label."""
    return np.isin(arr, list(positive_labels))


def _read_iou_thresh(res, iou_thresh: float | None) -> float:
    """
    Return the IoU threshold to use for the confidence sweep.
    If iou_thresh is None, read from res.config.iou (the value used during
    evaluate_detections). This ensures the sweep always respects the actual
    evaluation that was run.
    """
    if iou_thresh is not None:
        return float(iou_thresh)
    try:
        return float(res.config.iou)
    except AttributeError:
        return 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Field name helpers
# ─────────────────────────────────────────────────────────────────────────────

def derive_field_name(json_path: str | Path) -> str:
    """
    Derive a short FiftyOne field name from a JSON prediction file path.

    Rule: take the file stem, split on '_rtdetr', keep the left part.

    Examples
    --------
    >>> derive_field_name(
    ...   ".../NWW_p5_aclr_seed1_rtdetr_0606_1418_test_predictions.json"
    ... )
    'NWW_p5_aclr_seed1'
    """
    stem = Path(json_path).stem
    return stem.split("_rtdetr")[0]


_EVAL_SUFFIXES = ("_tp", "_fp", "_fn", "_tn", "_iou")


def get_fields_for_seed(
    dataset,
    seed_term: str,
    verbose:   bool = True,
) -> dict[str, list[str]]:
    """
    Return all prediction fields belonging to one seed, split into
    raw / nms / clean (excluding FiftyOne eval book-keeping fields).

    Parameters
    ----------
    dataset   : FiftyOne dataset or view
    seed_term : substring that identifies the seed, e.g. 'seed0', 'SEED63'
    verbose   : print counts

    Returns
    -------
    dict with keys: 'all', 'raw', 'nms', 'clean'
    """
    all_fields  = list(dataset.get_field_schema().keys())
    seed_fields = [f for f in all_fields if seed_term in f]

    raw   = [f for f in seed_fields if f.endswith("_raw")]
    nms   = [f for f in seed_fields if f.endswith("_nms")]
    clean = [
        f for f in seed_fields
        if not any(f.endswith(f"_{s}") for s in ("tp", "fp", "fn", "tn", "iou"))
    ]

    if verbose:
        print(f"seed_term='{seed_term}'  "
              f"total={len(seed_fields)}  "
              f"raw={len(raw)}  nms={len(nms)}  clean={len(clean)}")

    return {"all": seed_fields, "raw": raw, "nms": nms, "clean": clean}


def filter_ops(
    all_op:     list[str],
    nms_or_raw: str  = None,
    seed_term:  str  = None,
) -> list[str]:
    """Filter a list of field names by suffix and/or seed substring."""
    def _match(op: str) -> bool:
        tokens = op.split("_")
        if nms_or_raw and nms_or_raw not in tokens:
            return False
        if seed_term and seed_term not in op:
            return False
        return True
    return [op for op in all_op if _match(op)]


def create_eval_key_name(field_name: str) -> str:
    """
    Derive a short eval key from a FiftyOne prediction field name.
    Supports both new-style (NWW_p5_aclr_seed0_nms) and old-style
    (NWW_ACLR_partition_10_SEED63_..._nms) field names.
    """
    splits = field_name.split("_")

    # ── New format: NWW_{partition}_{method}_{seed}_{suffix} ─────────────────
    if (
        len(splits) >= 5
        and splits[0] == "NWW"
        and splits[1].startswith("p")
        and splits[1][1:].isdigit()
        and splits[2] in ("aclr", "random")
    ):
        partition = splits[1]
        method    = splits[2]
        seed      = splits[3]
        suffix    = splits[-1]
        return f"{method}_{partition}_{seed}_{suffix}"

    # ── Old format: NWW_ACLR_partition_{n}_{SEED}_{suffix} ───────────────────
    if "ACLR" in splits and "partition" in splits:
        part_idx  = splits.index("partition")
        partition = splits[part_idx + 1]
        seed      = splits[part_idx + 2]
        suffix    = splits[-1]
        return f"ACLR_p{partition}_{seed}_{suffix}"

    if "partition" in splits:
        part_idx  = splits.index("partition")
        partition = splits[part_idx + 1]
        seed      = splits[part_idx + 2]
        suffix    = splits[-1]
        return f"p{partition}_{seed}_{suffix}"

    # ── Fallback ──────────────────────────────────────────────────────────────
    return field_name


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
    all_op          : list of prediction field names
    nms_or_raw      : 'nms' or 'raw'
    baseline_field  : optional explicit baseline field to append
    seed_term       : seed substring filter
    deduplicate     : remove duplicate pred_fields
    verbose         : print the final list
    """
    nms_or_raw = nms_or_raw.strip().lower()
    if nms_or_raw not in ("raw", "nms"):
        raise ValueError(f"nms_or_raw must be 'raw' or 'nms', got '{nms_or_raw}'")

    filtered  = filter_ops(all_op, nms_or_raw=nms_or_raw, seed_term=seed_term)
    eval_list = [
        {"pred_field": f, "eval_key": create_eval_key_name(f)}
        for f in filtered
    ]

    if baseline_field is not None:
        baseline_key = (
            f"baseline_{seed_term}_{nms_or_raw}"
            if seed_term else f"baseline_{nms_or_raw}"
        )
        eval_list.append({"pred_field": baseline_field, "eval_key": baseline_key})

    if deduplicate:
        counts = Counter(d["pred_field"] for d in eval_list)
        dups   = [k for k, v in counts.items() if v > 1]
        if dups and verbose:
            print(f"  WARNING: duplicate pred_fields: {dups}")

        seen: dict[str, dict] = {}
        for item in eval_list:
            key = item["pred_field"]
            if key not in seen:
                seen[key] = item
            elif "baseline" in item["eval_key"]:
                seen[key] = item
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
    eval_list    : list of {'pred_field': str, 'eval_key': str}
    gt_field     : ground-truth detections field
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
        print(f"\n  run_evaluations done — "
              f"{len(results)} / {len(eval_list)} succeeded")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Core sweep function
# ─────────────────────────────────────────────────────────────────────────────

def sweep_thresholds(
    res,
    method:           str   = "eer",
    iou_thresh:       float | None = None,
    custom_threshold: float | None = None,
    n_thresholds:     int   = 200,
) -> dict:
    """
    Sweep confidence thresholds over a COCODetectionResults object and return
    full curve arrays plus scalar metrics at the chosen operating point.

    Labels are inferred dynamically from res.ytrue — never hardcoded.
    IoU threshold defaults to res.config.iou if not overridden.

    Parameters
    ----------
    res              : COCODetectionResults from evaluate_detections()
    method           : operating point selection
                       'f1'     — argmax of F1 curve
                       'eer'    — argmin of |fn_rate - fp_rate|
                       'custom' — use custom_threshold directly
    iou_thresh       : IoU for TP matching. If None, reads res.config.iou.
    custom_threshold : required when method='custom'
    n_thresholds     : number of steps in [0.01, 0.99]

    Returns
    -------
    dict with keys:
        n_gt, positive_labels, iou_thresh, method
        thresholds, f1, precision, recall,
        tp, fp, fn, fn_rate, fp_rate          — np.ndarray (n_thresholds,)
        best   — scalar metrics at the chosen operating point
            threshold, f1, precision, recall,
            tp, fp, fn, fn_rate, fp_rate, n_gt
    """
    if method not in ("f1", "eer", "custom"):
        raise ValueError(f"method='{method}' not supported. "
                         "Choose 'f1', 'eer', or 'custom'.")
    if method == "custom" and custom_threshold is None:
        raise ValueError("method='custom' requires custom_threshold.")

    iou_t = _read_iou_thresh(res, iou_thresh)

    ytrue = np.array(res.ytrue)
    ypred = np.array(res.ypred)
    confs = np.array(
        [v if v is not None else 0.0 for v in res.confs], dtype=float
    )
    ious  = np.array(
        [v if v is not None else 0.0 for v in res.ious],  dtype=float
    )

    # ── Infer positive labels dynamically ────────────────────────────────────
    positive_labels = _infer_positive_labels(ytrue)
    if len(positive_labels) > 1:
        warnings.warn(
            f"Multiple positive labels found: {positive_labels}. "
            "All will be treated as positives.",
            UserWarning, stacklevel=2,
        )

    # ── n_gt: unmatched GTs + matched GTs ────────────────────────────────────
    # unmatched GT: ytrue=positive, ypred='(none)'
    # matched GT:   ytrue=positive, ypred=positive
    n_gt = int(
        (ypred == "(none)").sum()
        + (_is_positive(ytrue, positive_labels)
           & _is_positive(ypred, positive_labels)).sum()
    )

    pred_mask  = ypred != "(none)"
    thresholds = np.linspace(0.01, 0.99, n_thresholds)

    f1_arr        = np.zeros(n_thresholds)
    precision_arr = np.zeros(n_thresholds)
    recall_arr    = np.zeros(n_thresholds)
    tp_arr        = np.zeros(n_thresholds, dtype=int)
    fp_arr        = np.zeros(n_thresholds, dtype=int)
    fn_arr        = np.zeros(n_thresholds, dtype=int)
    fn_rate_arr   = np.zeros(n_thresholds)
    fp_rate_arr   = np.zeros(n_thresholds)

    for i, thresh in enumerate(thresholds):
        mask = pred_mask & (confs >= thresh)

        tp = int(
            (_is_positive(ytrue[mask], positive_labels)
             & (ious[mask] >= iou_t)).sum()
        )
        fp = int((ytrue[mask] == "(none)").sum())
        fn = int(n_gt - tp)

        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)
        fn_rate   = fn / (n_gt + 1e-9)
        fp_rate   = fp / (mask.sum() + 1e-9) if mask.sum() > 0 else 0.0

        f1_arr[i]        = f1
        precision_arr[i] = precision
        recall_arr[i]    = recall
        tp_arr[i]        = tp
        fp_arr[i]        = fp
        fn_arr[i]        = fn
        fn_rate_arr[i]   = fn_rate
        fp_rate_arr[i]   = fp_rate

    # ── Find operating point ──────────────────────────────────────────────────
    if method == "f1":
        best_idx = int(np.argmax(f1_arr))
    elif method == "eer":
        best_idx = int(np.argmin(np.abs(fn_rate_arr - fp_rate_arr)))
    else:  # custom
        best_idx = int(np.argmin(np.abs(thresholds - custom_threshold)))

    best = dict(
        threshold = float(thresholds[best_idx]),
        f1        = float(f1_arr[best_idx]),
        precision = float(precision_arr[best_idx]),
        recall    = float(recall_arr[best_idx]),
        tp        = int(tp_arr[best_idx]),
        fp        = int(fp_arr[best_idx]),
        fn        = int(fn_arr[best_idx]),
        fn_rate   = float(fn_rate_arr[best_idx]),
        fp_rate   = float(fp_rate_arr[best_idx]),
        n_gt      = n_gt,
    )

    return dict(
        n_gt            = n_gt,
        positive_labels = positive_labels,
        iou_thresh      = iou_t,
        method          = method,
        thresholds      = thresholds,
        f1              = f1_arr,
        precision       = precision_arr,
        recall          = recall_arr,
        tp              = tp_arr,
        fp              = fp_arr,
        fn              = fn_arr,
        fn_rate         = fn_rate_arr,
        fp_rate         = fp_rate_arr,
        best            = best,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_sweep(
    res,
    method:       str         = "eer",
    iou_thresh:   float|None  = None,
    n_thresholds: int         = 200,
    title:        str         = None,
    save_path:    str         = None,
) -> plt.Figure:
    """
    3-panel diagnostic plot for a single COCODetectionResults object.
    Shows both the F1-optimal and EER operating points so the user can
    decide which method to pass to extract_all_metrics.

    Panels
    ------
    Left   — F1 / Precision / Recall vs confidence threshold
    Centre — FN rate vs FP rate (EER curve)
    Right  — TP / FP / FN counts vs confidence threshold

    Parameters
    ----------
    res          : COCODetectionResults
    method       : which operating point to highlight ('f1' | 'eer' | 'custom')
    iou_thresh   : if None reads res.config.iou
    n_thresholds : sweep resolution
    title        : figure suptitle (auto-generated if None)
    save_path    : optional save path

    Returns
    -------
    matplotlib.figure.Figure
    """
    # compute both operating points regardless of chosen method
    sweep_f1  = sweep_thresholds(res, method="f1",  iou_thresh=iou_thresh,
                                  n_thresholds=n_thresholds)
    sweep_eer = sweep_thresholds(res, method="eer", iou_thresh=iou_thresh,
                                  n_thresholds=n_thresholds)

    # use the user-chosen method for the "primary" highlighted point
    sweep_primary = sweep_f1 if method == "f1" else sweep_eer

    threshs    = sweep_primary["thresholds"]
    f1s        = sweep_primary["f1"]
    precisions = sweep_primary["precision"]
    recalls    = sweep_primary["recall"]
    fn_rates   = sweep_primary["fn_rate"]
    fp_rates   = sweep_primary["fp_rate"]
    tps        = sweep_primary["tp"]
    fps        = sweep_primary["fp"]
    fns        = sweep_primary["fn"]

    t_f1  = sweep_f1["best"]["threshold"]
    t_eer = sweep_eer["best"]["threshold"]

    C_F1  = "#4e79a7"
    C_EER = "#f28e2b"
    C_P   = "#59a14f"
    C_R   = "#e15759"

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.patch.set_facecolor("white")

    def _style(ax):
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    # ── Panel 1 — F1 / P / R curves ──────────────────────────────────────────
    ax = axes[0]
    ax.plot(threshs, f1s,        color=C_F1, linewidth=2.2, label="F1")
    ax.plot(threshs, precisions, color=C_P,  linewidth=1.5,
            linestyle="--", label="Precision")
    ax.plot(threshs, recalls,    color=C_R,  linewidth=1.5,
            linestyle="--", label="Recall")
    ax.axvline(t_f1,  color=C_F1, linewidth=1.5, linestyle=":",
               label=f"F1-optimal  t={t_f1:.3f}  "
                     f"F1={sweep_f1['best']['f1']:.3f}")
    ax.axvline(t_eer, color=C_EER, linewidth=1.5, linestyle=":",
               label=f"EER         t={t_eer:.3f}  "
                     f"FNr={sweep_eer['best']['fn_rate']:.3f}")
    ax.set_xlabel("Confidence threshold", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("F1 / Precision / Recall", fontsize=11)
    ax.legend(fontsize=7.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    _style(ax)

    # ── Panel 2 — EER curve ───────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(threshs, fn_rates, color=C_R,   linewidth=2.2, label="FN rate")
    ax.plot(threshs, fp_rates, color=C_F1,  linewidth=2.2, label="FP rate")
    ax.axvline(t_eer, color=C_EER, linewidth=1.5, linestyle=":",
               label=f"EER  t={t_eer:.3f}  "
                     f"FNr={sweep_eer['best']['fn_rate']:.3f}  "
                     f"FPr={sweep_eer['best']['fp_rate']:.3f}")
    ax.axvline(t_f1,  color=C_F1, linewidth=1.5, linestyle=":",
               label=f"F1-opt  t={t_f1:.3f}")
    ax.set_xlabel("Confidence threshold", fontsize=10)
    ax.set_ylabel("Rate", fontsize=10)
    ax.set_title("FN rate vs FP rate — EER", fontsize=11)
    ax.legend(fontsize=7.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    _style(ax)

    # ── Panel 3 — TP / FP / FN counts ────────────────────────────────────────
    ax = axes[2]
    ax.plot(threshs, tps, color=C_F1, linewidth=2.2, label="TP")
    ax.plot(threshs, fps, color=C_R,  linewidth=2.2, label="FP")
    ax.plot(threshs, fns, color=C_EER,linewidth=2.2, label="FN")
    ax.axvline(t_f1,  color=C_F1, linewidth=1.5, linestyle=":",
               label=f"F1-opt  t={t_f1:.3f}  "
                     f"TP={sweep_f1['best']['tp']}  "
                     f"FP={sweep_f1['best']['fp']}  "
                     f"FN={sweep_f1['best']['fn']}")
    ax.axvline(t_eer, color=C_EER, linewidth=1.5, linestyle=":",
               label=f"EER     t={t_eer:.3f}  "
                     f"TP={sweep_eer['best']['tp']}  "
                     f"FP={sweep_eer['best']['fp']}  "
                     f"FN={sweep_eer['best']['fn']}")
    ax.set_xlabel("Confidence threshold", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("TP / FP / FN counts", fontsize=11)
    ax.legend(fontsize=7.5)
    ax.set_xlim(0, 1)
    _style(ax)

    # ── Metadata box ─────────────────────────────────────────────────────────
    iou_t  = sweep_primary["iou_thresh"]
    n_gt   = sweep_primary["n_gt"]
    labels = sweep_primary["positive_labels"]
    info   = (f"n_gt={n_gt}  |  IoU≥{iou_t}  |  "
              f"labels={labels}  |  highlighted: {method.upper()}")
    fig.text(0.5, 0.01, info, ha="center", fontsize=8,
             color="#555555", style="italic")

    _title = title or (
        f"Threshold sweep — "
        f"mAP={res.mAP():.3f}  mAR={res.mAR():.3f}"
    )
    fig.suptitle(_title, fontsize=12, fontweight="500", y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved → {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Metrics extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_metrics(
    results_dict: dict,
    seed,
    method:       str        = "eer",
    iou_thresh:   float|None = None,
    n_thresholds: int        = 200,
) -> pd.DataFrame:
    """
    Compute threshold-sweep metrics for every model in results_dict and
    return a tidy DataFrame with one row per model.

    Both F1-optimal and EER operating points are always stored. The
    method= argument controls which one is aliased into the short column
    names (threshold, f1_score, precision, recall, tp, fp, fn) so that
    existing plotting functions continue to work unchanged.

    Parameters
    ----------
    results_dict  : {eval_key: COCODetectionResults}
    seed          : seed identifier written into the 'seed' column
    method        : 'f1' | 'eer' — controls alias columns
    iou_thresh    : if None, reads res.config.iou per model
    n_thresholds  : sweep resolution

    Returns
    -------
    pd.DataFrame — see module docstring for column contract
    """
    if method not in ("f1", "eer"):
        raise ValueError(f"method='{method}' not supported here. "
                         "Choose 'f1' or 'eer'.")

    rows = []

    for eval_key, res in results_dict.items():
        try:
            sw_f1  = sweep_thresholds(res, method="f1",  iou_thresh=iou_thresh,
                                       n_thresholds=n_thresholds)
            sw_eer = sweep_thresholds(res, method="eer", iou_thresh=iou_thresh,
                                       n_thresholds=n_thresholds)

            b_f1  = sw_f1["best"]
            b_eer = sw_eer["best"]

            # chosen operating point for alias columns
            chosen = b_f1 if method == "f1" else b_eer

            row = {
                # ── identification ────────────────────────────────────────────
                "seed":           seed,
                "eval_key":       eval_key,

                # ── COCO threshold-free metrics ───────────────────────────────
                "mAP":            res.mAP(),
                "mAR":            res.mAR(),

                # ── F1-optimal operating point ────────────────────────────────
                "f1_threshold":   b_f1["threshold"],
                "f1":             b_f1["f1"],
                "precision_f1":   b_f1["precision"],
                "recall_f1":      b_f1["recall"],
                "tp_f1":          b_f1["tp"],
                "fp_f1":          b_f1["fp"],
                "fn_f1":          b_f1["fn"],

                # ── EER operating point ───────────────────────────────────────
                "eer_threshold":  b_eer["threshold"],
                "fn_rate_eer":    b_eer["fn_rate"],
                "fp_rate_eer":    b_eer["fp_rate"],
                "precision_eer":  b_eer["precision"],
                "recall_eer":     b_eer["recall"],
                "tp_eer":         b_eer["tp"],
                "fp_eer":         b_eer["fp"],
                "fn_eer":         b_eer["fn"],

                # ── shared ────────────────────────────────────────────────────
                "n_gt":           sw_f1["n_gt"],
                "iou_thresh":     sw_f1["iou_thresh"],
                "positive_label": str(sorted(sw_f1["positive_labels"])),

                # ── alias columns (backward-compatible with plotting fns) ─────
                "threshold":      chosen["threshold"],
                "f1_score":       chosen["f1"],
                "precision":      chosen["precision"],
                "recall":         chosen["recall"],
                "tp":             chosen["tp"],
                "fp":             chosen["fp"],
                "fn":             chosen["fn"],
            }
            rows.append(row)

        except Exception as exc:
            print(f"  WARNING: extract_all_metrics failed for '{eval_key}': {exc}")

    print(f"Seed {seed} done — {len(rows)} / {len(results_dict)} models processed "
          f"(method='{method}')")
    return pd.DataFrame(rows)