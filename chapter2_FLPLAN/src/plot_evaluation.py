"""
plot_evaluation.py
==================
All plotting functions for the FLPLAN domain-adaptation evaluation pipeline.

Intended to be imported into a notebook alongside evaluate_flplan_predictions.py:

    from evaluate_flplan_predictions import (
        sweep_thresholds, extract_all_metrics, run_evaluations,
    )
    from plot_evaluation import (
        plot_sweep,
        plot_pr_curves_partitions,
        plot_confidence_grid,
        plot_f1_vs_partition,
        plot_relative_improvement,
        plot_efficiency_curves,
    )

All plotting functions that accept a df_metrics DataFrame expect the column
contract defined in evaluate_flplan_predictions.py — specifically the alias
columns (threshold, f1_score, precision, recall, tp, fp, fn) which are always
present regardless of which method= was used in extract_all_metrics.

Functions
---------
plot_sweep(res, ...)
    3-panel diagnostic for one COCODetectionResults. Shows F1/P/R curve,
    EER curve, and TP/FP/FN counts with both operating points marked.
    Call this before extract_all_metrics to decide which method= to use.

plot_pr_curves_partitions(results, df_metrics, ...)
    One panel per partition showing the COCO PR curve (from res.precision/
    res.recall) for baseline vs random vs ACLR. Annotates mAP and F1 from
    df_metrics.

plot_confidence_grid(results, df_metrics, rows, ...)
    Grid of confidence histograms (TP vs FP) with F1-optimal threshold line.
    One row per partition, three columns: baseline / random / ACLR.

plot_f1_vs_partition(df_metrics, seed, metric, ...)
    Single plot: chosen metric vs partition size for random vs ACLR,
    with the baseline as a horizontal reference line.

plot_relative_improvement(df_seed0, df_seed63, df_seed72, df_baseline, ...)
    Mean relative improvement over zero-shot baseline vs partition size,
    aggregated across seeds, for random and ACLR.

plot_efficiency_curves(df_seed0, df_seed63, df_seed72, ...)
    3×3 grid: rows = seeds, columns = metrics. Each cell shows absolute
    performance vs partition size for random vs ACLR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── Shared colour palette ─────────────────────────────────────────────────────
C_BASELINE = "#888888"
C_RANDOM   = "#4e79a7"
C_ACLR     = "#e15759"
C_F1_LINE  = "#4e79a7"
C_EER_LINE = "#f28e2b"
C_TP       = "#4e79a7"
C_FP       = "#e15759"
C_FN       = "#f28e2b"
C_PREC     = "#59a14f"
C_REC      = "#e15759"


def _style_ax(ax: plt.Axes) -> None:
    """Apply consistent axis styling."""
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def _get_df_row(df_metrics: pd.DataFrame, eval_key: str) -> pd.Series | None:
    """Return the first matching row from df_metrics or None."""
    if df_metrics is None:
        return None
    mask = df_metrics["eval_key"] == eval_key
    if not mask.any():
        return None
    return df_metrics[mask].iloc[0]


def _infer_positive_labels(ytrue: np.ndarray) -> set:
    return set(np.unique(ytrue)) - {"(none)"}


def _is_positive(arr: np.ndarray, positive_labels: set) -> np.ndarray:
    return np.isin(arr, list(positive_labels))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. plot_sweep
# ═══════════════════════════════════════════════════════════════════════════════

def plot_sweep(
    res,
    method:       str        = "eer",
    iou_thresh:   float|None = None,
    n_thresholds: int        = 200,
    title:        str        = None,
    save_path:    str        = None,
) -> plt.Figure:
    """
    3-panel diagnostic plot for a single COCODetectionResults object.

    Shows both the F1-optimal and EER operating points on all three panels
    so the user can compare them visually before deciding which method= to
    pass to extract_all_metrics.

    Parameters
    ----------
    res          : COCODetectionResults from evaluate_detections()
    method       : operating point to highlight in the title ('f1' | 'eer')
    iou_thresh   : IoU for TP matching. If None reads res.config.iou.
    n_thresholds : sweep resolution
    title        : figure suptitle (auto-generated if None)
    save_path    : optional path to save the figure

    Returns
    -------
    matplotlib.figure.Figure
    """
    from evaluate_flplan_predictions import sweep_thresholds

    sw_f1  = sweep_thresholds(res, method="f1",  iou_thresh=iou_thresh,
                               n_thresholds=n_thresholds)
    sw_eer = sweep_thresholds(res, method="eer", iou_thresh=iou_thresh,
                               n_thresholds=n_thresholds)

    sw = sw_f1 if method == "f1" else sw_eer

    threshs    = sw["thresholds"]
    f1s        = sw["f1"]
    precisions = sw["precision"]
    recalls    = sw["recall"]
    fn_rates   = sw["fn_rate"]
    fp_rates   = sw["fp_rate"]
    tps        = sw["tp"]
    fps        = sw["fp"]
    fns        = sw["fn"]

    t_f1  = sw_f1["best"]["threshold"]
    t_eer = sw_eer["best"]["threshold"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.patch.set_facecolor("white")

    # ── Panel 1 — F1 / Precision / Recall ────────────────────────────────────
    ax = axes[0]
    ax.plot(threshs, f1s,        color=C_F1_LINE, linewidth=2.2, label="F1")
    ax.plot(threshs, precisions, color=C_PREC,    linewidth=1.5,
            linestyle="--", label="Precision")
    ax.plot(threshs, recalls,    color=C_REC,     linewidth=1.5,
            linestyle="--", label="Recall")
    ax.axvline(t_f1,  color=C_F1_LINE, linewidth=1.5, linestyle=":",
               label=f"F1-opt  t={t_f1:.3f}  "
                     f"F1={sw_f1['best']['f1']:.3f}  "
                     f"P={sw_f1['best']['precision']:.3f}  "
                     f"R={sw_f1['best']['recall']:.3f}")
    ax.axvline(t_eer, color=C_EER_LINE, linewidth=1.5, linestyle=":",
               label=f"EER     t={t_eer:.3f}  "
                     f"FNr={sw_eer['best']['fn_rate']:.3f}  "
                     f"FPr={sw_eer['best']['fp_rate']:.3f}")
    ax.set_xlabel("Confidence threshold", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("F1 / Precision / Recall vs Threshold", fontsize=11)
    ax.legend(fontsize=7)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    _style_ax(ax)

    # ── Panel 2 — EER curve ───────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(threshs, fn_rates, color=C_REC,     linewidth=2.2, label="FN rate")
    ax.plot(threshs, fp_rates, color=C_F1_LINE, linewidth=2.2, label="FP rate")
    ax.axvline(t_eer, color=C_EER_LINE, linewidth=1.5, linestyle=":",
               label=f"EER  t={t_eer:.3f}  "
                     f"FNr={sw_eer['best']['fn_rate']:.3f}  "
                     f"FPr={sw_eer['best']['fp_rate']:.3f}")
    ax.axvline(t_f1,  color=C_F1_LINE, linewidth=1.5, linestyle=":",
               label=f"F1-opt  t={t_f1:.3f}")
    ax.set_xlabel("Confidence threshold", fontsize=10)
    ax.set_ylabel("Rate", fontsize=10)
    ax.set_title("FN rate vs FP rate — Equal Error Rate", fontsize=11)
    ax.legend(fontsize=7)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    _style_ax(ax)

    # ── Panel 3 — TP / FP / FN counts ────────────────────────────────────────
    ax = axes[2]
    ax.plot(threshs, tps, color=C_TP,  linewidth=2.2, label="TP")
    ax.plot(threshs, fps, color=C_FP,  linewidth=2.2, label="FP")
    ax.plot(threshs, fns, color=C_FN,  linewidth=2.2, label="FN")
    ax.axvline(t_f1,  color=C_F1_LINE, linewidth=1.5, linestyle=":",
               label=f"F1-opt  t={t_f1:.3f}  "
                     f"TP={sw_f1['best']['tp']}  "
                     f"FP={sw_f1['best']['fp']}  "
                     f"FN={sw_f1['best']['fn']}")
    ax.axvline(t_eer, color=C_EER_LINE, linewidth=1.5, linestyle=":",
               label=f"EER     t={t_eer:.3f}  "
                     f"TP={sw_eer['best']['tp']}  "
                     f"FP={sw_eer['best']['fp']}  "
                     f"FN={sw_eer['best']['fn']}")
    ax.set_xlabel("Confidence threshold", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("TP / FP / FN counts vs Threshold", fontsize=11)
    ax.legend(fontsize=7)
    ax.set_xlim(0, 1)
    _style_ax(ax)

    # ── Metadata caption ──────────────────────────────────────────────────────
    n_gt   = sw["n_gt"]
    iou_t  = sw["iou_thresh"]
    labels = sw["positive_labels"]
    fig.text(
        0.5, 0.01,
        f"n_gt={n_gt}  |  IoU≥{iou_t}  |  labels={labels}  |  highlighted: {method.upper()}",
        ha="center", fontsize=8, color="#555555", style="italic",
    )

    _title = title or (
        f"Threshold sweep  —  mAP={res.mAP():.3f}  mAR={res.mAR():.3f}"
    )
    fig.suptitle(_title, fontsize=12, fontweight="500", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved → {save_path}")

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 2. plot_pr_curves_partitions
# ═══════════════════════════════════════════════════════════════════════════════

def plot_pr_curves_partitions(
    results:    dict,
    df_metrics: pd.DataFrame,
    partitions: list        = None,
    iou_idx:    int         = 0,
    seed:       int         = 0,
    title:      str         = None,
    save_path:  str         = None,
) -> plt.Figure:
    """
    One panel per partition showing the COCO PR curve for baseline vs
    random vs ACLR. Annotates mAP, F1, precision, recall, and optimal
    threshold from df_metrics.

    The PR curve is taken from res.precision[iou_idx, 0, :] and res.recall —
    the COCO-protocol 101-point interpolated curve, not the confidence sweep.

    Parameters
    ----------
    results    : {eval_key: COCODetectionResults}
    df_metrics : DataFrame from extract_all_metrics()
    partitions : list of partition sizes as strings, e.g. ["5", "10", "25"]
    iou_idx    : index into the IoU threshold axis (0 = IoU@0.50)
    seed       : seed number used when building eval_key strings
    title      : figure suptitle
    save_path  : optional save path
    """
    if partitions is None:
        partitions = ["5", "10", "25", "50", "75"]

    def _label(eval_key: str, base: str) -> str:
        row = _get_df_row(df_metrics, eval_key)
        if row is not None:
            return (
                f"{base}\n"
                f"mAP={row['mAP']:.2f}  F1={row['f1_score']:.2f}\n"
                f"P={row['precision']:.2f}  R={row['recall']:.2f}  "
                f"@t={row['threshold']:.2f}"
            )
        if eval_key in results:
            return f"{base}  (mAP={results[eval_key].mAP():.2f})"
        return base

    fig, axes = plt.subplots(
        1, len(partitions),
        figsize=(5 * len(partitions), 5),
        sharey=True,
    )
    if len(partitions) == 1:
        axes = [axes]

    for ax, p in zip(axes, partitions):
        # derive seed string for old-format eval keys
        seed_str = f"SEED{seed}" if seed in (63, 72) else f"seed{seed}"

        subset = [
            (f"baseline_{seed_str}_nms",      "Baseline", C_BASELINE),
            (f"random_p{p}_{seed_str}_nms",   f"Random {p}%", C_RANDOM),
            (f"aclr_p{p}_{seed_str}_nms",     f"ACLR {p}%",   C_ACLR),
            # old-format fallbacks
            (f"baseline_SEED{seed}_nms",       "Baseline",     C_BASELINE),
            (f"p{p}_SEED{seed}_nms",           f"Random {p}%", C_RANDOM),
            (f"ACLR_p{p}_SEED{seed}_nms",      f"ACLR {p}%",   C_ACLR),
        ]

        plotted_labels = set()
        for eval_key, base_label, color in subset:
            if base_label in plotted_labels:
                continue
            if eval_key not in results:
                continue
            res   = results[eval_key]
            prec  = res.precision[iou_idx, 0, :]
            rec   = res.recall
            valid = prec >= 0
            label = _label(eval_key, base_label)
            ax.plot(rec[valid], prec[valid], color=color,
                    linewidth=2.5, label=label)
            plotted_labels.add(base_label)

        ax.set_title(f"Partition {p}%", fontsize=11)
        ax.set_xlabel("Recall", fontsize=10)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(fontsize=6.5, loc="lower left",
                  framealpha=0.9, edgecolor="lightgrey")
        ax.set_aspect("equal")
        _style_ax(ax)

    axes[0].set_ylabel("Precision", fontsize=10)
    _title = title or f"PR Curves @ IoU=0.50 — seed={seed}"
    fig.suptitle(_title, fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved → {save_path}")

    plt.show()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 3. plot_confidence_grid
# ═══════════════════════════════════════════════════════════════════════════════

def plot_confidence_grid(
    results:      dict,
    df_metrics:   pd.DataFrame,
    rows:         list,
    baseline_key: str  = None,
    title:        str  = None,
    save_path:    str  = None,
) -> plt.Figure:
    """
    Grid of confidence histograms (TP vs FP distributions) with the
    F1-optimal threshold marked as a vertical line.

    Parameters
    ----------
    results      : {eval_key: COCODetectionResults}
    df_metrics   : DataFrame from extract_all_metrics()
    rows         : list of (label, random_key, aclr_key) tuples
                   e.g. [("5%", "random_p5_seed0_nms", "aclr_p5_seed0_nms")]
    baseline_key : eval key for the baseline model (shown in column 0)
    title        : figure suptitle
    save_path    : optional save path
    """
    n_rows = len(rows)
    bins   = np.linspace(0, 1, 20)

    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(13, 4 * n_rows),
        sharey=False, sharex=True,
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    def _plot_conf(ax: plt.Axes, eval_key: str, subtitle: str) -> None:
        if eval_key not in results:
            ax.text(0.5, 0.5, f"'{eval_key}'\nnot found",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="#888780", style="italic")
            ax.set_title(subtitle, fontsize=10)
            return

        res   = results[eval_key]
        ytrue = np.array(res.ytrue)
        confs = np.array([v if v is not None else np.nan
                          for v in res.confs], dtype=float)
        ious  = np.array([v if v is not None else 0.0
                          for v in res.ious], dtype=float)

        pos_labels = _infer_positive_labels(ytrue)
        pred_mask  = ~np.isnan(confs)

        row_df = _get_df_row(df_metrics, eval_key)

        # read threshold from df_metrics alias column
        iou_t = 0.5
        if row_df is not None and "iou_thresh" in row_df:
            iou_t = float(row_df["iou_thresh"])

        tp_confs = confs[pred_mask
                         & _is_positive(ytrue, pos_labels)
                         & (ious >= iou_t)]
        fp_confs = confs[pred_mask & (ytrue == "(none)")]

        ax.hist(tp_confs, bins=bins, color=C_TP, alpha=0.6,
                label=f"TP (n={len(tp_confs)})", density=True)
        ax.hist(fp_confs, bins=bins, color=C_FP, alpha=0.6,
                label=f"FP (n={len(fp_confs)})", density=True)

        if len(tp_confs) > 0:
            ax.axvline(np.median(tp_confs), color=C_TP,
                       linewidth=1.2, linestyle="--", alpha=0.7)
        if len(fp_confs) > 0:
            ax.axvline(np.median(fp_confs), color=C_FP,
                       linewidth=1.2, linestyle="--", alpha=0.7)

        if row_df is not None:
            thresh = float(row_df["threshold"])
            ax.axvline(
                thresh, color="#2ca02c", linewidth=2.0, linestyle="-",
                label=(f"thresh={thresh:.2f}\n"
                       f"F1={row_df['f1_score']:.2f}  "
                       f"P={row_df['precision']:.2f}  "
                       f"R={row_df['recall']:.2f}"),
            )

        ax.set_title(subtitle, fontsize=10)
        ax.legend(fontsize=6.5, loc="upper left", framealpha=0.9)
        _style_ax(ax)

    for row_idx, (row_label, random_key, aclr_key) in enumerate(rows):
        bk = baseline_key or ""
        _plot_conf(axes[row_idx, 0], bk,
                   "Baseline" if row_idx == 0 else "")
        _plot_conf(axes[row_idx, 1], random_key, f"Random {row_label}")
        _plot_conf(axes[row_idx, 2], aclr_key,   f"ACLR {row_label}")
        axes[row_idx, 0].set_ylabel(f"Partition {row_label}\nDensity",
                                    fontsize=10)

    for ax, header in zip(axes[0], ["Baseline", "Random", "ACLR"]):
        ax.set_title(header, fontsize=11, fontweight="bold", pad=12)

    for ax in axes[-1]:
        ax.set_xlabel("Confidence score", fontsize=10)

    _title = title or (
        "Confidence Distribution — TP vs FP\n"
        "green line = F1-optimal threshold"
    )
    fig.suptitle(_title, fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved → {save_path}")

    plt.show()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 4. plot_f1_vs_partition
# ═══════════════════════════════════════════════════════════════════════════════

def plot_f1_vs_partition(
    df_metrics:   pd.DataFrame,
    seed:         int   = 0,
    metric:       str   = "f1_score",
    partitions:   list  = None,
    baseline_key: str   = None,
    title:        str   = None,
    save_path:    str   = None,
) -> plt.Figure:
    """
    Single plot: chosen metric vs partition size for random vs ACLR,
    with the baseline as a horizontal reference line.

    Parameters
    ----------
    df_metrics   : DataFrame from extract_all_metrics()
    seed         : seed number
    metric       : column name to plot, e.g. 'f1_score', 'mAP', 'mAR', 'recall'
    partitions   : list of ints, default [5, 10, 25, 50, 75, 100]
    baseline_key : eval_key of the baseline row in df_metrics.
                   Auto-detected if None.
    title        : figure suptitle
    save_path    : optional save path
    """
    if partitions is None:
        partitions = [5, 10, 25, 50, 75, 100]

    seed_str = f"SEED{seed}" if seed in (63, 72) else f"seed{seed}"

    # ── Baseline reference ────────────────────────────────────────────────────
    if baseline_key is None:
        # try both new and old naming conventions
        for candidate in [
            f"baseline_{seed_str}_nms",
            f"baseline_SEED{seed}_nms",
        ]:
            row = _get_df_row(df_metrics, candidate)
            if row is not None:
                baseline_key = candidate
                break

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")

    if baseline_key:
        b_row = _get_df_row(df_metrics, baseline_key)
        if b_row is not None:
            b_val = float(b_row[metric])
            ax.axhline(b_val, color=C_BASELINE, linewidth=2,
                       linestyle="--",
                       label=f"Baseline ({metric}={b_val:.3f})", zorder=2)
            ax.fill_between([0, 108], b_val - 0.005, b_val + 0.005,
                            color=C_BASELINE, alpha=0.12)

    # ── Random and ACLR lines ─────────────────────────────────────────────────
    random_vals, aclr_vals = [], []
    for p in partitions:
        for keys, vals in [
            ([f"random_p{p}_{seed_str}_nms",
              f"p{p}_SEED{seed}_nms"],   random_vals),
            ([f"aclr_p{p}_{seed_str}_nms",
              f"ACLR_p{p}_SEED{seed}_nms"], aclr_vals),
        ]:
            found = False
            for k in keys:
                row = _get_df_row(df_metrics, k)
                if row is not None:
                    vals.append(float(row[metric]))
                    found = True
                    break
            if not found:
                vals.append(np.nan)

    x = np.array(partitions)
    ax.plot(x, random_vals, color=C_RANDOM, linewidth=2.5,
            marker="o", markersize=8, label="Random", zorder=3)
    ax.plot(x, aclr_vals,   color=C_ACLR,   linewidth=2.5,
            marker="s", markersize=8, label="ACLR",   zorder=3)

    # ── Point annotations ─────────────────────────────────────────────────────
    for p, rv, av in zip(partitions, random_vals, aclr_vals):
        if not np.isnan(rv):
            ax.annotate(f"{rv:.3f}", (p, rv),
                        textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=8, color=C_RANDOM)
        if not np.isnan(av):
            ax.annotate(f"{av:.3f}", (p, av),
                        textcoords="offset points", xytext=(0, -15),
                        ha="center", fontsize=8, color=C_ACLR)

    # ── Shade advantage region ────────────────────────────────────────────────
    r_arr = np.array(random_vals)
    a_arr = np.array(aclr_vals)
    valid = ~(np.isnan(r_arr) | np.isnan(a_arr))
    if valid.any():
        ax.fill_between(x[valid], r_arr[valid], a_arr[valid],
                        where=a_arr[valid] >= r_arr[valid],
                        alpha=0.12, color=C_ACLR)
        ax.fill_between(x[valid], r_arr[valid], a_arr[valid],
                        where=r_arr[valid] > a_arr[valid],
                        alpha=0.12, color=C_RANDOM)

    # ── Formatting ────────────────────────────────────────────────────────────
    all_vals = [v for v in list(random_vals) + list(aclr_vals)
                if not np.isnan(v)]
    ax.set_xlim(0, 108)
    if all_vals:
        ax.set_ylim(min(all_vals) * 0.96, max(all_vals) * 1.04)
    ax.set_xticks(partitions)
    ax.set_xticklabels([f"{p}%" for p in partitions])
    ax.set_xlabel("Training partition (%)", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    _title = title or (
        f"Data Efficiency — {metric} vs Partition Size\n"
        f"Random vs ACLR  |  seed={seed}  |  dashed = Baseline"
    )
    ax.set_title(_title, fontsize=13)
    ax.legend(fontsize=10, loc="lower right")
    _style_ax(ax)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved → {save_path}")

    plt.show()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 5. plot_relative_improvement
# ═══════════════════════════════════════════════════════════════════════════════
def plot_relative_improvement(
    df_concat:   pd.DataFrame,
    metric:      str  = "f1_score",
    partitions:  list = None,
    title:       str  = None,
    save_path:   str  = None,
) -> plt.Figure:
    """
    Mean relative improvement over zero-shot baseline vs partition size,
    aggregated across seeds, for random and ACLR.

    Parameters
    ----------
    df_concat   : concatenated DataFrame from all seeds, with columns:
                  seed, eval_key, method, partition, + metric columns.
                  method values: 'aclr', 'random', 'baseline'
                  partition values: 'p5', 'p10', 'p20', ... 'p100'
    metric      : column to compute improvement on, e.g. 'f1_score', 'mAP'
    partitions  : list of partition strings to include, e.g. ['p5','p10','p25']
                  if None, auto-detected from df_concat
    title       : figure suptitle
    save_path   : path stem (no extension); saves .png + .pdf if given

    Returns
    -------
    matplotlib.figure.Figure
    """
    # ── Auto-detect partitions if not given ───────────────────────────────────
    if partitions is None:
        # extract from rows where method is aclr or random
        p_vals = (
            df_concat[df_concat["method"].isin(["aclr", "random"])]
            ["partition"]
            .unique()
        )
        # sort numerically by the integer after 'p'
        partitions = sorted(
            p_vals,
            key=lambda p: int(p[1:]) if p[1:].isdigit() else 0,
        )

    # ── Per-seed baseline anchor ──────────────────────────────────────────────
    # baseline rows: method == 'baseline'
    baseline_rows = df_concat[df_concat["method"] == "baseline"]

    if baseline_rows.empty:
        raise ValueError(
            "No rows with method='baseline' found in df_concat. "
            "Make sure your baseline eval_key starts with 'baseline'."
        )

    # anchor per seed = mean of metric across all baseline rows for that seed
    anchors = (
        baseline_rows
        .groupby("seed")[metric]
        .mean()
        .to_dict()
    )
    seeds = sorted(df_concat["seed"].unique())

    print(f"  Seeds found     : {seeds}")
    print(f"  Baseline anchors:")
    for s, v in anchors.items():
        print(f"    seed={s}  anchor={v:.4f}")

    # ── Relative improvement per seed × partition × method ────────────────────
    improvements = {
        "aclr":   {p: [] for p in partitions},
        "random": {p: [] for p in partitions},
    }

    for seed in seeds:
        anchor   = anchors.get(seed, np.nan)
        seed_df  = df_concat[df_concat["seed"] == seed]

        for method in ("aclr", "random"):
            method_df = seed_df[seed_df["method"] == method]
            for p in partitions:
                part_df = method_df[method_df["partition"] == p]
                if not part_df.empty:
                    val = float(part_df.iloc[0][metric]) - anchor
                else:
                    val = np.nan
                improvements[method][p].append(val)

    # ── Aggregate across seeds: mean ± std ────────────────────────────────────
    def _agg(method: str):
        means, stds = [], []
        for p in partitions:
            vals = [v for v in improvements[method][p] if not np.isnan(v)]
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals, ddof=0) if len(vals) > 1 else 0.0)
        return np.array(means), np.array(stds)

    r_means, r_stds = _agg("random")
    a_means, a_stds = _agg("aclr")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")

    # x-axis: integer partition sizes for spacing
    x_ints = np.array([int(p[1:]) for p in partitions])

    for means, stds, color, label, marker in [
        (r_means, r_stds, C_RANDOM, "Random", "o"),
        (a_means, a_stds, C_ACLR,   "ACLR",   "s"),
    ]:
        valid = ~np.isnan(means)
        ax.plot(x_ints[valid], means[valid],
                color=color, linewidth=2.5,
                marker=marker, markersize=8, label=label, zorder=4)
        ax.errorbar(x_ints[valid], means[valid], yerr=stds[valid],
                    fmt="none", color=color,
                    capsize=5, capthick=1.8, linewidth=1.8, zorder=5)
        ax.fill_between(
            x_ints[valid],
            means[valid] - stds[valid],
            means[valid] + stds[valid],
            color=color, alpha=0.15, zorder=3,
        )

    # ── Value annotations ─────────────────────────────────────────────────────
    for xi, rm, am in zip(x_ints, r_means, a_means):
        if not np.isnan(rm):
            ax.annotate(
                f"{rm:+.3f}", (xi, rm),
                textcoords="offset points", xytext=(0, -18),
                ha="center", fontsize=8, color=C_RANDOM, fontweight="bold",
            )
        if not np.isnan(am):
            ax.annotate(
                f"{am:+.3f}", (xi, am),
                textcoords="offset points", xytext=(0, +10),
                ha="center", fontsize=8, color=C_ACLR, fontweight="bold",
            )

    # ── ACLR vs Random advantage shading ─────────────────────────────────────
    valid = ~(np.isnan(r_means) | np.isnan(a_means))
    if valid.any():
        ax.fill_between(x_ints[valid], r_means[valid], a_means[valid],
                        where=a_means[valid] >= r_means[valid],
                        alpha=0.08, color=C_ACLR)
        ax.fill_between(x_ints[valid], r_means[valid], a_means[valid],
                        where=r_means[valid] > a_means[valid],
                        alpha=0.08, color=C_RANDOM)

    # ── Zero reference ────────────────────────────────────────────────────────
    ax.axhline(0, color=C_BASELINE, linewidth=1.8, linestyle="--",
               label="Zero-shot baseline", zorder=2)

    # ── Formatting ────────────────────────────────────────────────────────────
    all_vals = np.concatenate([
        r_means - r_stds, r_means + r_stds,
        a_means - a_stds, a_means + a_stds,
    ])
    pad = 0.02
    ax.set_ylim(np.nanmin(all_vals) - pad, np.nanmax(all_vals) + pad)
    ax.set_xlim(x_ints[0] - 3, x_ints[-1] + 3)
    ax.set_xticks(x_ints)
    ax.set_xticklabels([f"{p}%" for p in partitions], fontsize=10)
    ax.set_xlabel("Training partition (%)", fontsize=12)
    ax.set_ylabel(f"Δ {metric}", fontsize=12)

    _title = title or (
        f"Relative Improvement over Zero-Shot Baseline\n"
        f"Metric: {metric}  |  {len(seeds)} seeds  |  "
        f"error bars = ±1 SD"
    )
    ax.set_title(_title, fontsize=13)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.85)
    _style_ax(ax)

    # ── Per-seed anchor caption ───────────────────────────────────────────────
    anchor_str = "  |  ".join(
        f"{s}: anchor={anchors.get(s, float('nan')):.3f}"
        for s in seeds
    )
    fig.text(0.5, -0.03, anchor_str,
             ha="center", fontsize=8, style="italic", color="#555555")

    plt.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"{save_path}.pdf", dpi=300, bbox_inches="tight")
        print(f"  Saved → {save_path}.png / .pdf")

    plt.show()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 6. plot_efficiency_curves
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_vals(
    df:            pd.DataFrame,
    method_prefix: str,
    seed:          int,
    partitions:    list,
    metric:        str,
) -> np.ndarray:
    """Return metric values for a given method/seed across partitions."""
    seed_str = f"SEED{seed}" if seed in (63, 72) else f"seed{seed}"
    vals = []
    for p in partitions:
        if method_prefix == "random":
            candidates = [f"random_p{p}_{seed_str}_nms",
                          f"p{p}_SEED{seed}_nms"]
        else:
            candidates = [f"aclr_p{p}_{seed_str}_nms",
                          f"ACLR_p{p}_SEED{seed}_nms"]

        found = False
        for k in candidates:
            row = _get_df_row(df, k)
            if row is not None:
                vals.append(float(row[metric]))
                found = True
                break
        if not found:
            vals.append(np.nan)

    return np.array(vals)


def plot_efficiency_curves(
    df_seed0:   pd.DataFrame,
    df_seed63:  pd.DataFrame,
    df_seed72:  pd.DataFrame,
    partitions: list = None,
    metrics:    list = None,
    title:      str  = None,
    save_path:  str  = None,
) -> plt.Figure:
    """
    3×3 grid: rows = seeds (0, 63, 72), columns = metrics.
    Each cell shows absolute performance vs partition size for random vs ACLR.

    Parameters
    ----------
    df_seed0/63/72 : DataFrames from extract_all_metrics()
    partitions     : list of ints, default [5, 10, 25, 50, 75]
    metrics        : list of column names, default ["mAP", "mAR", "f1_score"]
    title          : figure suptitle
    save_path      : path stem (no extension); saves .png + .pdf if given
    """
    if partitions is None:
        partitions = [5, 10, 25, 50, 75]
    if metrics is None:
        metrics = ["mAP", "mAR", "f1_score"]

    seed_map      = [(0, df_seed0), (63, df_seed63), (72, df_seed72)]
    metric_labels = {
        "mAP":     "mAP",
        "mAR":     "mAR",
        "f1_score":"F1",
        "f1":      "F1",
        "precision":"Precision",
        "recall":  "Recall",
    }

    fig, axes = plt.subplots(
        3, len(metrics),
        figsize=(5 * len(metrics), 12),
        sharey=False, sharex=True,
    )
    if len(metrics) == 1:
        axes = axes[:, np.newaxis]

    _title = title or (
        "Data Efficiency — Absolute Performance vs Partition Size\n"
        "Rows = Seeds  |  Columns = Metrics  |  Random vs ACLR"
    )
    fig.suptitle(_title, fontsize=14, fontweight="bold", y=1.01)

    x = np.array(partitions)

    for row_idx, (seed, df) in enumerate(seed_map):
        for col_idx, metric in enumerate(metrics):
            ax      = axes[row_idx, col_idx]
            m_label = metric_labels.get(metric, metric)

            r_vals = _extract_vals(df, "random", seed, partitions, metric)
            a_vals = _extract_vals(df, "aclr",   seed, partitions, metric)

            for vals, color, method, marker in [
                (r_vals, C_RANDOM, "Random", "o"),
                (a_vals, C_ACLR,   "ACLR",   "s"),
            ]:
                valid = ~np.isnan(vals)
                if not valid.any():
                    continue
                ax.plot(x[valid], vals[valid], color=color,
                        linewidth=2.5, marker=marker, markersize=7,
                        label=method, zorder=4)

                for xi, v in zip(x[valid], vals[valid]):
                    offset = +10 if method == "Random" else -14
                    ax.annotate(
                        f"{v:.3f}", (xi, v),
                        textcoords="offset points", xytext=(0, offset),
                        ha="center", fontsize=7, color=color, fontweight="bold",
                    )

            # shade advantage region
            valid_both = ~(np.isnan(r_vals) | np.isnan(a_vals))
            if valid_both.any():
                ax.fill_between(x[valid_both], r_vals[valid_both],
                                a_vals[valid_both],
                                where=a_vals[valid_both] >= r_vals[valid_both],
                                alpha=0.12, color=C_ACLR)
                ax.fill_between(x[valid_both], r_vals[valid_both],
                                a_vals[valid_both],
                                where=r_vals[valid_both] > a_vals[valid_both],
                                alpha=0.12, color=C_RANDOM)

            # formatting
            ax.set_xlim(x[0] - 3, x[-1] + 3)
            all_v = np.concatenate([r_vals, a_vals])
            all_v = all_v[~np.isnan(all_v)]
            if len(all_v):
                pad = (all_v.max() - all_v.min()) * 0.15 + 0.02
                ax.set_ylim(all_v.min() - pad, all_v.max() + pad)

            ax.set_xticks(partitions)
            _style_ax(ax)

            if row_idx == 0:
                ax.set_title(m_label, fontsize=12, fontweight="bold", pad=8)
            if col_idx == 0:
                ax.set_ylabel(f"seed={seed}\n{m_label}", fontsize=10)
            else:
                ax.set_ylabel(m_label, fontsize=9)
            if row_idx == len(seed_map) - 1:
                ax.set_xticklabels([f"{p}%" for p in partitions], fontsize=9)
                ax.set_xlabel("Training partition (%)", fontsize=10)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=8.5, loc="lower right", framealpha=0.85)

    plt.tight_layout()
    if save_path:
        fig.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"{save_path}.pdf", dpi=300, bbox_inches="tight")
        print(f"  Saved → {save_path}.png / .pdf")

    plt.show()
    return fig