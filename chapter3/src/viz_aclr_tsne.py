"""
active_learning_viz.py
======================
Visualise and compare active-learning selections in t-SNE space.

Produces a 1-row x N-column figure (one column per strategy) for a SINGLE
partition. In every panel:
  * background = all embeddings, small + faded, coloured by cluster
  * foreground = the selected samples for that strategy, drawn as stars in
    their cluster colour with a black edge so they stand out

Key design points
-----------------
1. t-SNE is computed ONCE on the full pool and reused for every panel, so the
   three columns share one coordinate frame and are directly comparable.
   compute_tsne() returns the coords -- cache them and pass them back in, since
   t-SNE is the slow part.
2. metric="cosine" matches the chapter's argument that the cosine projection is
   the informative one for L2-normalised DINOv3 embeddings.
3. Selections are read back from the JSON files the strategies wrote, mapped to
   embedding rows by filepath, so the stars are exactly what RT-DETR trains on.

Typical use
-----------
    from active_learning_viz import compute_tsne, visualize_partition_from_json

    coords = compute_tsne(embeddings_norm, seed=42)        # once, reuse below

    visualize_partition_from_json(
        embeddings_norm = embeddings_norm,
        cluster_labels  = cluster_labels,
        sample_ids      = sample_ids,
        dataset         = train_view,
        json_paths      = {
            "Centroid":            "centroid.json",
            "Centroid-Uniqueness": "cent_uniq.json",
            "Ball-Radius (ACLR)":  "ball.json",
        },
        partition       = 0.5,
        coords          = coords,
        save_path       = "tsne_p50.png",
    )
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize


# ══════════════════════════════════════════════════════════════════════════════
#  t-SNE  (compute once, reuse across panels)
# ══════════════════════════════════════════════════════════════════════════════

def compute_tsne(
    embeddings_norm,
    perplexity=30,
    seed=42,
    metric="cosine",
    init="pca",
    learning_rate="auto",
    verbose=True,
):
    """
    2-D t-SNE of the full embedding pool. Returns (N, 2) coordinates.

    Cache the result and feed it back into the plotting functions -- recomputing
    per figure is wasteful and (with a different seed) would move every point.
    """
    emb = normalize(np.asarray(embeddings_norm, dtype=np.float32), norm="l2")
    N = len(emb)
    perplexity = min(perplexity, max(5, (N - 1) // 3))   # guard tiny pools

    if verbose:
        print(f"  t-SNE: N={N}  perplexity={perplexity}  metric={metric} ...")

    # init='pca' is not valid together with a precomputed/cosine metric in some
    # sklearn versions when the data isn't euclidean; 'pca' on the raw vectors is
    # fine here, but fall back to 'random' if the version complains.
    try:
        ts = TSNE(n_components=2, perplexity=perplexity, metric=metric,
                  init=init, learning_rate=learning_rate, random_state=seed)
        coords = ts.fit_transform(emb)
    except (ValueError, TypeError):
        ts = TSNE(n_components=2, perplexity=perplexity, metric=metric,
                  init="random", learning_rate=learning_rate, random_state=seed)
        coords = ts.fit_transform(emb)

    if verbose:
        print(f"  t-SNE done: coords shape {coords.shape}")
    return coords


# ══════════════════════════════════════════════════════════════════════════════
#  JSON  ->  embedding rows   (map by filepath)
# ══════════════════════════════════════════════════════════════════════════════

def build_fp_to_row(dataset, sample_ids):
    """
    Build {filepath: row_index} for the pool, where row_index indexes into
    embeddings_norm / cluster_labels (same order as sample_ids).
    """
    id_to_row = {sid: i for i, sid in enumerate(sample_ids)}
    fp_to_row = {}
    tuple = dataset.values(["id", "filepath"])
    for sid, fp in zip(tuple[0],tuple[1]):
        r = id_to_row.get(sid)
        if r is not None:
            fp_to_row[fp] = r
    return fp_to_row


def rows_from_json(json_path, partition, fp_to_row, seed="0", split="train",
                   verbose=True):
    """
    Read one selection JSON and return {method_name: [row indices]} for the
    requested partition. Each strategy file typically holds a single method key.
    """
    with open(json_path) as f:
        data = json.load(f)

    key = f"p{int(round(partition * 100))}"
    try:
        node = data[seed][split][key]
    except KeyError as e:
        raise KeyError(
            f"{json_path}: no entry for [{seed}][{split}][{key}]. "
            f"Available partitions: "
            f"{list(data.get(seed, {}).get(split, {}).keys())}"
        ) from e

    out = {}
    for method, payload in node.items():
        rows, missing = [], 0
        for fp in payload["images"]:
            r = fp_to_row.get(fp)
            if r is None:
                missing += 1
            else:
                rows.append(r)
        out[method] = rows
        if verbose:
            msg = f"  {Path(json_path).name} [{key}] {method}: {len(rows)} rows"
            if missing:
                msg += f"  ({missing} filepaths not matched to the pool)"
            print(msg)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  COLOURS
# ══════════════════════════════════════════════════════════════════════════════

def _distinct_colors(K):
    """K visually distinct RGBA colours."""
    if K <= 10:
        return plt.cm.tab10(np.arange(K) % 10)
    if K <= 20:
        return plt.cm.tab20(np.arange(K) % 20)
    if K <= 40:
        return np.vstack([plt.cm.tab20(np.arange(20)),
                          plt.cm.tab20b(np.arange(20))])[:K]
    return plt.cm.gist_ncar(np.linspace(0, 1, K, endpoint=False))


def _cluster_color_map(cluster_labels):
    uniq = np.unique(cluster_labels)
    colors = _distinct_colors(len(uniq))
    cid_to_color = {int(c): colors[i] for i, c in enumerate(uniq)}
    point_colors = np.array([cid_to_color[int(c)] for c in cluster_labels])
    return point_colors, cid_to_color, uniq


# ══════════════════════════════════════════════════════════════════════════════
#  THE PLOT
# ══════════════════════════════════════════════════════════════════════════════

def plot_selection_tsne(
    coords,
    cluster_labels,
    selected_rows_by_method,        # {column_title: [row indices]} (ordered)
    partition_label="",
    figsize_per_col=(6.0, 6.0),
    bg_size=6, bg_alpha=0.25,
    sel_size=150, sel_marker="*",
    sel_edgecolor="black", sel_edgewidth=0.6,
    bg_by_cluster=True,             # False -> grey background, colour only stars
    highlight_color=None,           # None -> star keeps cluster colour; else fixed
    show_legend=None,               # None -> auto (legend if <=20 clusters)
    save_path=None, dpi=200,
):
    """
    One row of panels (one per method) sharing the same t-SNE frame.

    Parameters
    ----------
    coords                   : (N, 2) t-SNE coordinates for the whole pool
    cluster_labels           : (N,) int cluster id per row
    selected_rows_by_method  : ordered dict {title: [row indices selected]}
    partition_label          : e.g. "50%" -> used in the suptitle
    bg_by_cluster            : colour the faded background by cluster (True) or
                               grey it out and colour only the stars (False)
    highlight_color          : None keeps each star its cluster colour; pass a
                               colour string to force one highlight colour
    show_legend              : force legend on/off; None auto-decides by K

    Returns
    -------
    fig, axes
    """
    coords = np.asarray(coords)
    labels = np.asarray(cluster_labels).astype(int)
    point_colors, cid_to_color, uniq = _cluster_color_map(labels)
    K = len(uniq)

    methods = list(selected_rows_by_method.keys())
    ncols = len(methods)
    fig, axes = plt.subplots(
        1, ncols,
        figsize=(figsize_per_col[0] * ncols, figsize_per_col[1]),
        sharex=True, sharey=True,
    )
    if ncols == 1:
        axes = [axes]

    bg_c = point_colors if bg_by_cluster else np.array([[0.75, 0.75, 0.75, 1.0]] * len(labels))

    for ax, title in zip(axes, methods):
        rows = np.asarray(selected_rows_by_method[title], dtype=int)

        # layer 1 -- all embeddings, faded
        ax.scatter(coords[:, 0], coords[:, 1],
                   c=bg_c, s=bg_size, alpha=bg_alpha,
                   linewidths=0, zorder=1)

        # layer 2 -- selected samples as stars
        if len(rows) > 0:
            star_c = (point_colors[rows] if highlight_color is None
                      else highlight_color)
            ax.scatter(coords[rows, 0], coords[rows, 1],
                       c=star_c, s=sel_size, marker=sel_marker,
                       edgecolors=sel_edgecolor, linewidths=sel_edgewidth,
                       zorder=3)

        ax.set_title(f"{title} | {len(rows)} selected samples", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_alpha(0.3)

    if partition_label:
        fig.suptitle(f"Partition {partition_label}",
                     fontsize=15, y=1.0)

    # optional discrete cluster legend
    auto_legend = (K <= 20) if show_legend is None else show_legend
    if auto_legend:
        handles = [Line2D([0], [0], marker="o", linestyle="",
                          markerfacecolor=cid_to_color[int(c)],
                          markeredgecolor="none", markersize=6,
                          label=f"cluster {int(c)}")
                   for c in uniq]
        handles.append(Line2D([0], [0], marker=sel_marker, linestyle="",
                              markerfacecolor="white",
                              markeredgecolor=sel_edgecolor, markersize=11,
                              label="selected"))
        fig.legend(handles=handles, loc="center left",
                   bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"  saved -> {save_path}")
    return fig, axes


# ══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE WRAPPER  (JSON files -> figure)
# ══════════════════════════════════════════════════════════════════════════════

def visualize_partition_from_json(
    embeddings_norm,
    cluster_labels,
    sample_ids,
    dataset,
    json_paths,                     # {column_title: json_path}  OR  [paths]
    partition,
    coords=None,                    # precomputed t-SNE; computed if None
    seed=42,
    save_path=None,
    verbose=True,
    **plot_kwargs,
):
    """
    Read the strategy JSONs, map each partition's selection to embedding rows,
    and draw the comparison figure.

    json_paths may be:
      * a dict {column_title: path} -- titles used as panel headers (order kept)
      * a list/tuple of paths       -- the method name inside each JSON is used

    coords: pass the array from compute_tsne() to avoid recomputing. If None,
    t-SNE is computed here (and returned alongside the figure).

    Returns
    -------
    fig, axes, coords
    """
    fp_to_row = build_fp_to_row(dataset, sample_ids)

    # normalise json_paths into an ordered {title: rows} dict
    selected_rows = {}
    if isinstance(json_paths, dict):
        for title, path in json_paths.items():
            per_method = rows_from_json(path, partition, fp_to_row, verbose=verbose)
            # each strategy file holds one method -> flatten, keep the given title
            rows = next(iter(per_method.values())) if per_method else []
            selected_rows[title] = rows
    else:
        for path in json_paths:
            per_method = rows_from_json(path, partition, fp_to_row, verbose=verbose)
            for method, rows in per_method.items():
                selected_rows[method] = rows

    if coords is None:
        coords = compute_tsne(embeddings_norm, seed=seed, verbose=verbose)

    fig, axes = plot_selection_tsne(
        coords=coords,
        cluster_labels=cluster_labels,
        selected_rows_by_method=selected_rows,
        partition_label=f"{int(round(partition * 100))}%",
        save_path=save_path,
        **plot_kwargs,
    )
    return fig, axes, coords


# ══════════════════════════════════════════════════════════════════════════════
#  (optional) direct-from-memory variant
# ══════════════════════════════════════════════════════════════════════════════

def visualize_partition_from_rows(
    embeddings_norm,
    cluster_labels,
    selected_rows_by_method,        # {title: [row indices]} you already have
    partition,
    coords=None,
    seed=42,
    save_path=None,
    verbose=True,
    **plot_kwargs,
):
    """
    Same figure as visualize_partition_from_json, but when you already hold the
    selected row indices per method in memory (skips JSON + filepath mapping).
    """
    if coords is None:
        coords = compute_tsne(embeddings_norm, seed=seed, verbose=verbose)
    fig, axes = plot_selection_tsne(
        coords=coords,
        cluster_labels=cluster_labels,
        selected_rows_by_method=selected_rows_by_method,
        partition_label=f"{int(round(partition * 100))}%",
        save_path=save_path,
        **plot_kwargs,
    )
    return fig, axes, coords