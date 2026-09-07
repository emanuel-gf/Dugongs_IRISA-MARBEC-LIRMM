"""
plot_analysis.py
================
Three high-value visualisation functions for the FLPLAN domain-adaptation
experiment, all operating on a single concatenated DataFrame (df_concat).

Expected df_concat columns
--------------------------
    seed        : seed identifier, e.g. 's0', 's1', 's2'
    eval_key    : full evaluation key string
    method      : 'aclr' | 'random' | 'baseline'  (from eval_key.split('_')[0])
    partition   : 'p5' | 'p10' | ... | 'p100'      (from eval_key.split('_')[1])
    mAP, mAR
    f1_score, precision, recall
    tp, fp, fn, n_gt
    threshold, iou_thresh

Functions
---------
plot_box_strip(df_concat, metrics, ...)
    Box + strip plot: metric distribution across seeds per method × partition.
    One subplot per metric. Baseline shown as horizontal dashed line.

plot_delta_heatmap(df_concat, metric, ...)
    Two-panel heatmap:
        left  — absolute metric per method × partition (mean ± std)
        right — Δ(aclr − random) per seed × partition plus mean row

plot_map_vs_f1(df_concat, ...)
    Scatter plot: mAP vs f1_score for all rows.
    Colour = method, marker shape = seed, size = partition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D


# ── Shared palette ────────────────────────────────────────────────────────────
C_BASELINE = "#888888"
C_RANDOM   = "#4e79a7"
C_ACLR     = "#e15759"
METHOD_COLORS = {
    "aclr":     C_ACLR,
    "random":   C_RANDOM,
    "baseline": C_BASELINE,
}

SEED_MARKERS = ["o", "s", "D", "^", "v", "P", "*"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _sort_partitions(partitions: list) -> list:
    """Sort partition strings numerically: p5 < p10 < p20 ..."""
    return sorted(partitions, key=lambda p: int(p[1:]) if p[1:].isdigit() else 0)


def _get_partitions(df: pd.DataFrame) -> list:
    """Auto-detect sorted partition list from df_concat."""
    raw = df[df["method"].isin(["aclr", "random"])]["partition"].unique()
    return _sort_partitions(list(raw))


def _style_ax(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def _normalise_partitions(partitions) -> list[str]:
    """Accept both int [5, 10] and str ['p5', 'p10'] inputs."""
    if partitions is None:
        return None
    return [f"p{p}" if isinstance(p, int) else str(p) for p in partitions]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Box + strip plot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_box_strip(
    df_concat:  pd.DataFrame,
    metrics:    list        = None,
    partitions: list        = None,
    title:      str         = None,
    figsize:    tuple       = None,
    save_path:  str         = None,
) -> plt.Figure:
    """
    Box + strip plot showing metric distribution across seeds per
    method × partition. One subplot per metric.

    Each partition has two side-by-side boxes (aclr / random). Individual
    seed values are overlaid as coloured dots. The baseline mean is shown
    as a horizontal dashed line.

    Parameters
    ----------
    df_concat  : concatenated DataFrame with method, partition, seed columns
    metrics    : list of column names to plot, default ['mAP', 'f1_score']
    partitions : list of partition strings or ints, None = auto-detect
    title      : figure suptitle
    figsize    : override figure size
    save_path  : optional save path (.png + .pdf)
    """
    if metrics is None:
        metrics = ["mAP", "f1_score"]

    partitions = _normalise_partitions(partitions) or _get_partitions(df_concat)
    seeds      = sorted(df_concat["seed"].unique())
    n_metrics  = len(metrics)

    # seed → marker mapping
    seed_marker = {s: SEED_MARKERS[i % len(SEED_MARKERS)]
                   for i, s in enumerate(seeds)}
    seed_colors = plt.cm.Set2(np.linspace(0, 1, len(seeds)))
    seed_color  = {s: seed_colors[i] for i, s in enumerate(seeds)}

    fig_w = max(5 * n_metrics, 10)
    fig_h = 5
    fig, axes = plt.subplots(
        1, n_metrics,
        figsize=figsize or (fig_w, fig_h),
        sharey=False,
    )
    if n_metrics == 1:
        axes = [axes]

    fig.patch.set_facecolor("white")

    x_positions_all = np.arange(len(partitions))
    width           = 0.32
    offsets         = {"aclr": -width / 2, "random": +width / 2}

    for ax, metric in zip(axes, metrics):

        # ── Baseline reference line ───────────────────────────────────────────
        baseline_df = df_concat[df_concat["method"] == "baseline"]
        if not baseline_df.empty:
            b_mean = float(baseline_df[metric].mean())
            ax.axhline(
                b_mean, color=C_BASELINE, linewidth=1.8,
                linestyle="--", zorder=1,
                label=f"Baseline mean ({b_mean:.3f})",
            )
            ax.fill_between(
                [-0.5, len(partitions) - 0.5],
                b_mean - 0.005, b_mean + 0.005,
                color=C_BASELINE, alpha=0.10, zorder=1,
            )

        for method in ("aclr", "random"):
            color  = METHOD_COLORS[method]
            offset = offsets[method]
            method_df = df_concat[df_concat["method"] == method]

            for xi, part in enumerate(partitions):
                part_df = method_df[method_df["partition"] == part]
                vals    = part_df[metric].dropna().values

                if len(vals) == 0:
                    continue

                xc = xi + offset

                # ── Box (manual: Q1, median, Q3) ─────────────────────────────
                q1, med, q3 = np.percentile(vals, [25, 50, 75])
                iqr  = q3 - q1
                wlo  = max(vals.min(), q1 - 1.5 * iqr)
                whi  = min(vals.max(), q3 + 1.5 * iqr)
                bw   = width * 0.85

                # whiskers
                ax.plot([xc, xc], [wlo, q1],
                        color=color, linewidth=1.2, zorder=2)
                ax.plot([xc, xc], [q3, whi],
                        color=color, linewidth=1.2, zorder=2)
                # whisker caps
                ax.plot([xc - bw * 0.3, xc + bw * 0.3], [wlo, wlo],
                        color=color, linewidth=1.2, zorder=2)
                ax.plot([xc - bw * 0.3, xc + bw * 0.3], [whi, whi],
                        color=color, linewidth=1.2, zorder=2)
                # IQR box
                rect = plt.Rectangle(
                    (xc - bw / 2, q1), bw, iqr,
                    facecolor=color, alpha=0.25,
                    edgecolor=color, linewidth=1.2, zorder=3,
                )
                ax.add_patch(rect)
                # median line
                ax.plot([xc - bw / 2, xc + bw / 2], [med, med],
                        color=color, linewidth=2.2, zorder=4)

                # ── Strip: individual seed points ─────────────────────────────
                for _, row in part_df.iterrows():
                    v = row[metric]
                    if pd.isna(v):
                        continue
                    s  = row["seed"]
                    jitter = np.random.uniform(-0.035, 0.035)
                    ax.scatter(
                        xc + jitter, v,
                        color=seed_color[s],
                        marker=seed_marker[s],
                        s=55, zorder=5, linewidths=0.6,
                        edgecolors="white",
                    )

        # ── Formatting ────────────────────────────────────────────────────────
        ax.set_xticks(x_positions_all)
        ax.set_xticklabels(
            [f"{p}" for p in partitions], fontsize=9, rotation=30,
        )
        ax.set_xlabel("Partition", fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(metric, fontsize=12, fontweight="500")
        ax.set_xlim(-0.5, len(partitions) - 0.5)
        _style_ax(ax)

    # ── Shared legend ─────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(facecolor=C_ACLR,   alpha=0.6, label="ACLR"),
        mpatches.Patch(facecolor=C_RANDOM, alpha=0.6, label="Random"),
        Line2D([0], [0], color=C_BASELINE, linewidth=1.8,
               linestyle="--", label="Baseline"),
    ]
    for s in seeds:
        handles.append(
            Line2D([0], [0], marker=seed_marker[s], color="w",
                   markerfacecolor=seed_color[s],
                   markeredgecolor="grey", markersize=8,
                   label=f"seed {s}")
        )

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        fontsize=8.5,
        frameon=True,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.08),
    )

    _title = title or (
        f"Metric distribution across seeds — Random vs ACLR\n"
        f"({len(seeds)} seeds, boxes = IQR, dots = individual seeds)"
    )
    fig.suptitle(_title, fontsize=13, fontweight="500", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"{save_path}.pdf", dpi=300, bbox_inches="tight")
        print(f"  Saved → {save_path}.png / .pdf")

    plt.show()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Delta heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def plot_delta_heatmap(
    df_concat:  pd.DataFrame,
    metric:     str   = "f1_score",
    partitions: list  = None,
    title:      str   = None,
    fmt_abs:    str   = ".3f",
    fmt_delta:  str   = "+.3f",
    figsize:    tuple = None,
    save_path:  str   = None,
) -> plt.Figure:
    """
    Two-panel heatmap:
        Left  — absolute metric mean ± std per method × partition
        Right — Δ(aclr − random) per seed × partition + mean row
                diverging colormap centred at 0

    Parameters
    ----------
    df_concat  : concatenated DataFrame
    metric     : column to visualise
    partitions : list of partition strings or ints, None = auto-detect
    title      : figure suptitle
    fmt_abs    : format string for absolute values cells
    fmt_delta  : format string for delta cells
    figsize    : override figure size
    save_path  : optional save path stem
    """
    partitions = _normalise_partitions(partitions) or _get_partitions(df_concat)
    seeds      = sorted(df_concat["seed"].unique())
    n_parts    = len(partitions)

    # ── Build absolute table: rows = [aclr, random], cols = partitions ────────
    abs_means = np.full((2, n_parts), np.nan)
    abs_stds  = np.full((2, n_parts), np.nan)
    row_labels_abs = ["ACLR", "Random"]

    for mi, method in enumerate(("aclr", "random")):
        method_df = df_concat[df_concat["method"] == method]
        for pi, part in enumerate(partitions):
            vals = method_df[method_df["partition"] == part][metric].dropna().values
            if len(vals):
                abs_means[mi, pi] = vals.mean()
                abs_stds[mi, pi]  = vals.std(ddof=0) if len(vals) > 1 else 0.0

    # ── Build delta table: rows = [s0, s1, s2, mean], cols = partitions ───────
    n_delta_rows  = len(seeds) + 1           # seeds + mean row
    delta_mat     = np.full((n_delta_rows, n_parts), np.nan)
    row_labels_d  = [str(s) for s in seeds] + ["mean"]

    for si, seed in enumerate(seeds):
        seed_df = df_concat[df_concat["seed"] == seed]
        for pi, part in enumerate(partitions):
            aclr_row   = seed_df[
                (seed_df["method"] == "aclr") &
                (seed_df["partition"] == part)
            ][metric].values
            random_row = seed_df[
                (seed_df["method"] == "random") &
                (seed_df["partition"] == part)
            ][metric].values
            if len(aclr_row) and len(random_row):
                delta_mat[si, pi] = float(aclr_row[0]) - float(random_row[0])

    # mean row (last row)
    delta_mat[-1, :] = np.nanmean(delta_mat[:-1, :], axis=0)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        1, 2,
        figsize=figsize or (max(12, n_parts * 1.4), max(4, n_delta_rows * 0.9)),
        gridspec_kw={"width_ratios": [1, 1.4]},
    )
    fig.patch.set_facecolor("white")

    col_labels = [p for p in partitions]

    # ── Left panel — absolute ─────────────────────────────────────────────────
    ax = axes[0]
    vmin_abs = np.nanmin(abs_means)
    vmax_abs = np.nanmax(abs_means)

    im_abs = ax.imshow(
        abs_means,
        cmap="Blues",
        vmin=max(0, vmin_abs - 0.05),
        vmax=min(1, vmax_abs + 0.05),
        aspect="auto",
    )

    for ri in range(2):
        for ci in range(n_parts):
            v = abs_means[ri, ci]
            s = abs_stds[ri, ci]
            if not np.isnan(v):
                cell_text = f"{v:{fmt_abs}}"
                if not np.isnan(s) and s > 0:
                    cell_text += f"\n±{s:.3f}"
                brightness = (v - max(0, vmin_abs - 0.05)) / max(
                    (min(1, vmax_abs + 0.05) - max(0, vmin_abs - 0.05)), 1e-6
                )
                txt_color = "white" if brightness > 0.6 else "black"
                ax.text(ci, ri, cell_text,
                        ha="center", va="center",
                        fontsize=8.5, color=txt_color)

    ax.set_xticks(range(n_parts))
    ax.set_xticklabels(col_labels, fontsize=9, rotation=35, ha="right")
    ax.set_yticks(range(2))
    ax.set_yticklabels(row_labels_abs, fontsize=10)
    ax.set_title(f"Absolute {metric}\n(mean ± std across seeds)", fontsize=10)
    plt.colorbar(im_abs, ax=ax, fraction=0.046, pad=0.04,
                 label=metric)

    # ── Right panel — delta ───────────────────────────────────────────────────
    ax = axes[1]
    abs_max_delta = np.nanmax(np.abs(delta_mat))
    abs_max_delta = max(abs_max_delta, 0.01)   # avoid zero range

    im_d = ax.imshow(
        delta_mat,
        cmap="RdYlGn",
        vmin=-abs_max_delta,
        vmax=+abs_max_delta,
        aspect="auto",
    )

    for ri in range(n_delta_rows):
        for ci in range(n_parts):
            v = delta_mat[ri, ci]
            if not np.isnan(v):
                is_mean_row = (ri == n_delta_rows - 1)
                fw   = "bold" if is_mean_row else "normal"
                norm_v = (v + abs_max_delta) / (2 * abs_max_delta)
                # use black text for mid-range, white for extremes
                if 0.25 < norm_v < 0.75:
                    txt_color = "black"
                else:
                    txt_color = "white"
                ax.text(ci, ri, f"{v:{fmt_delta}}",
                        ha="center", va="center",
                        fontsize=8.5, color=txt_color,
                        fontweight=fw)

    # separator line between seed rows and mean row
    ax.axhline(len(seeds) - 0.5, color="white", linewidth=2)

    ax.set_xticks(range(n_parts))
    ax.set_xticklabels(col_labels, fontsize=9, rotation=35, ha="right")
    ax.set_yticks(range(n_delta_rows))
    ax.set_yticklabels(row_labels_d, fontsize=10)
    ax.set_title(f"Δ(ACLR − Random)  [{metric}]\nper seed + mean row",
                 fontsize=10)
    plt.colorbar(im_d, ax=ax, fraction=0.046, pad=0.04,
                 label=f"Δ {metric}  (green = ACLR wins)")

    _title = title or (
        f"Performance Heatmap — {metric}\n"
        f"Left: absolute  |  Right: ACLR − Random delta"
    )
    fig.suptitle(_title, fontsize=13, fontweight="500", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"{save_path}.pdf", dpi=300, bbox_inches="tight")
        print(f"  Saved → {save_path}.png / .pdf")

    plt.show()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 3. mAP vs F1 scatter
# ═══════════════════════════════════════════════════════════════════════════════

def plot_map_vs_f1(
    df_concat:      pd.DataFrame,
    x_metric:       str   = "mAP",
    y_metric:       str   = "f1_score",
    annotate:       bool  = True,
    annotate_label: str   = "partition",
    partitions:     list  = None,
    title:          str   = None,
    figsize:        tuple = (9, 7),
    save_path:      str   = None,
) -> plt.Figure:
    """
    Scatter plot: x_metric vs y_metric for all rows in df_concat.

    Encoding
    --------
    colour      : method (aclr / random / baseline)
    marker shape: seed
    size        : partition integer (larger = more training data)
    annotation  : partition label on each point (optional)

    Parameters
    ----------
    df_concat      : concatenated DataFrame
    x_metric       : column for x-axis, default 'mAP'
    y_metric       : column for y-axis, default 'f1_score'
    annotate       : whether to label each point with its partition
    annotate_label : column to use for annotation text
    partitions     : filter to specific partitions (None = all)
    title          : figure suptitle
    figsize        : figure size
    save_path      : optional save path stem
    """
    partitions = _normalise_partitions(partitions)
    seeds      = sorted(df_concat["seed"].unique())
    seed_marker = {s: SEED_MARKERS[i % len(SEED_MARKERS)]
                   for i, s in enumerate(seeds)}

    # filter partitions if requested
    plot_df = df_concat.copy()
    if partitions:
        plot_df = plot_df[plot_df["partition"].isin(partitions)]

    # partition size → marker size mapping
    unique_parts  = _sort_partitions(plot_df["partition"].dropna().unique().tolist())
    part_int      = {p: int(p[1:]) for p in unique_parts if p[1:].isdigit()}
    max_part      = max(part_int.values()) if part_int else 100
    min_size, max_size = 40, 260

    def _marker_size(part: str) -> float:
        n = part_int.get(part, 50)
        return min_size + (n / max_part) * (max_size - min_size)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")

    # ── Diagonal reference (x == y) ───────────────────────────────────────────
    all_x = plot_df[x_metric].dropna().values
    all_y = plot_df[y_metric].dropna().values
    if len(all_x) and len(all_y):
        lim_min = min(all_x.min(), all_y.min()) - 0.02
        lim_max = max(all_x.max(), all_y.max()) + 0.02
        ax.plot([lim_min, lim_max], [lim_min, lim_max],
                color="#aaaaaa", linewidth=1, linestyle="--",
                alpha=0.5, zorder=1, label=f"{x_metric} = {y_metric}")

    # ── Scatter per method × seed ─────────────────────────────────────────────
    plotted_method_labels = set()

    for method in ("baseline", "random", "aclr"):
        color    = METHOD_COLORS.get(method, "#333333")
        mdf      = plot_df[plot_df["method"] == method]

        for seed in seeds:
            sdf    = mdf[mdf["seed"] == seed]
            marker = seed_marker[seed]

            for _, row in sdf.iterrows():
                xv = row.get(x_metric, np.nan)
                yv = row.get(y_metric, np.nan)
                if pd.isna(xv) or pd.isna(yv):
                    continue
                part = row.get("partition", "")
                sz   = _marker_size(part)

                label = method if method not in plotted_method_labels else "_nolegend_"
                if method not in plotted_method_labels:
                    plotted_method_labels.add(method)

                ax.scatter(
                    xv, yv,
                    c=color, marker=marker,
                    s=sz, alpha=0.82,
                    edgecolors="white", linewidths=0.6,
                    zorder=3, label=label,
                )

                if annotate and annotate_label in row.index:
                    ann = str(row[annotate_label])
                    ax.annotate(
                        ann,
                        (xv, yv),
                        textcoords="offset points",
                        xytext=(5, 4),
                        fontsize=7,
                        color=color,
                        alpha=0.85,
                    )

    # ── Convex hull per method to show clustering ─────────────────────────────
    from scipy.spatial import ConvexHull

    for method in ("aclr", "random"):
        color = METHOD_COLORS[method]
        mdf   = plot_df[plot_df["method"] == method][[x_metric, y_metric]].dropna()
        if len(mdf) >= 3:
            try:
                pts  = mdf.values
                hull = ConvexHull(pts)
                for simplex in hull.simplices:
                    ax.plot(pts[simplex, 0], pts[simplex, 1],
                            color=color, linewidth=0.8,
                            alpha=0.25, zorder=2)
                ax.fill(pts[hull.vertices, 0], pts[hull.vertices, 1],
                        color=color, alpha=0.06, zorder=2)
            except Exception:
                pass   # not enough unique points for hull

    # ── Size legend ───────────────────────────────────────────────────────────
    size_handles = []
    for p in unique_parts[::max(1, len(unique_parts) // 4)]:
        sz = _marker_size(p)
        size_handles.append(
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#888888", markeredgecolor="#888888",
                   markersize=np.sqrt(sz) * 0.55,
                   label=p)
        )

    # ── Seed shape legend ─────────────────────────────────────────────────────
    seed_handles = [
        Line2D([0], [0], marker=seed_marker[s], color="w",
               markerfacecolor="#888888", markeredgecolor="#888888",
               markersize=8, label=f"seed {s}")
        for s in seeds
    ]

    # ── Method colour legend ──────────────────────────────────────────────────
    method_handles = [
        mpatches.Patch(facecolor=METHOD_COLORS[m], alpha=0.7, label=m.capitalize())
        for m in ("aclr", "random", "baseline")
        if m in plot_df["method"].unique()
    ]

    leg1 = ax.legend(handles=method_handles, title="Method",
                     loc="upper left", fontsize=8, title_fontsize=8,
                     framealpha=0.9)
    ax.add_artist(leg1)

    leg2 = ax.legend(handles=seed_handles, title="Seed",
                     loc="lower right", fontsize=8, title_fontsize=8,
                     framealpha=0.9)
    ax.add_artist(leg2)

    ax.legend(handles=size_handles, title="Partition",
              loc="lower left", fontsize=8, title_fontsize=8,
              framealpha=0.9)

    # ── Formatting ────────────────────────────────────────────────────────────
    ax.set_xlabel(x_metric, fontsize=12)
    ax.set_ylabel(y_metric, fontsize=12)
    _style_ax(ax)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    _title = title or (
        f"{x_metric} vs {y_metric}\n"
        f"colour = method  |  shape = seed  |  size = partition"
    )
    ax.set_title(_title, fontsize=12, fontweight="500")
    plt.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"{save_path}.pdf", dpi=300, bbox_inches="tight")
        print(f"  Saved → {save_path}.png / .pdf")

    plt.show()
    return fig