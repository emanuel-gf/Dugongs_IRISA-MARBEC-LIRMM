import numpy as np
import matplotlib.pyplot as plt 
import math
import random
from pathlib import Path
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from umap import UMAP
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

def plot_decay_family_comparison(
    embeddings_norm,
    k:          int   = 15,
    bins:       int   = 50,
    save_path:  str   = None,
    dpi:        int   = 180,
):
    """
    Computes uniqueness scores for different weight decay families on real
    embeddings and plots their distributions side by side.

    Parameters
    ----------
    embeddings_norm : np.ndarray (N, D) — L2-normalised embeddings
    k               : number of neighbours (same for all families)
    bins            : histogram bins
    save_path       : optional save path
    dpi             : output resolution
    """
    from sklearn.neighbors import NearestNeighbors

    BG    = "#ececf0"
    WHITE = "#0a0000"
    MUTED = "#2D2D2E"

    N = len(embeddings_norm)
    print(f"Computing uniqueness for N={N} embeddings, k={k} ...")

    # ── Fit kNN once — shared across all families ─────────────────────────────
    knn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=-1)
    knn.fit(embeddings_norm)
    distances, _ = knn.kneighbors(embeddings_norm)
    dists = distances[:, 1:]   # (N, k) — exclude self

    ranks = np.arange(1, k + 1, dtype=np.float64)

    # ── Define families ───────────────────────────────────────────────────────
    def _scores(weights):
        w = np.array(weights) / np.array(weights).sum()
        s = (dists * w).sum(axis=1)
        return s / s.max()

    families = [
        ("Linear",            "#02859f", _scores((k + 1 - ranks) / k)),
        ("Exponential λ=0.5", "#830404", _scores(np.exp(-0.5 * ranks))),
        ("Exponential λ=0.8", "#996a0c", _scores(np.exp(-0.8 * ranks))),
        ("Power p=1 (harmonic)", "#11af11", _scores(1.0 / ranks ** 1)),
        ("Power p=3",          "#a66ca6", _scores(1.0 / ranks ** 3)),
    ]

    for name, _, scores in families:
        print(f"  {name:<25} mean={scores.mean():.4f}  "
              f"std={scores.std():.4f}  "
              f"above 0.9: {(scores > 0.9).sum()}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), facecolor=BG)
    fig.suptitle(
        f"Uniqueness Score Distribution by Decay Family  "
        f"\n (N={N}, k={k})",
        color=WHITE, fontsize=13, fontweight="bold", y=1.02,
    )

    def _style(ax, title, xlabel, ylabel):
        ax.set_facecolor(BG)
        ax.set_title(title, color=WHITE, fontsize=10, pad=6)
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
        ax.tick_params(colors=MUTED)
        ax.legend(fontsize=8, facecolor="#d4d4e1",
                  edgecolor="#555577", labelcolor=WHITE)
        ax.grid(color="#333355", linewidth=0.5, alpha=0.6)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444466")

    # Panel 1 — overlapping histograms
    ax = axes[0]
    for name, color, scores in families:
        ax.hist(scores, bins=bins, alpha=0.50, color=color,
                label=name, edgecolor=BG, linewidth=0.3)
    _style(ax, "Score distributions per decay family",
           "Uniqueness score", "Count")
    ax.set_xlim(0, 1)

    # Panel 2 — KDE-style smooth curves (using histogram density)
    ax = axes[1]
    for name, color, scores in families:
        counts, bin_edges = np.histogram(scores, bins=bins, density=True)
        centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.plot(centres, counts, color=color, linewidth=2.2, label=name)
        ax.fill_between(centres, counts, alpha=0.10, color=color)
        # Vertical mean line
        ax.axvline(scores.mean(), color=color, linewidth=1,
                   linestyle="--", alpha=0.7)
    _style(ax, "Density curves",
           "Uniqueness score", "Density")
    ax.set_xlim(0, 1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=BG)
        print(f"Saved → {save_path}")
    plt.show()
    return fig

def plot_uniqueness_landscape(
    dataset,
    embeddings_field:  str   = "full_embeddings",
    uniqueness_field:  str   = "uniqueness_score",
    color_field:       str   = None,   # optional categorical field to colour by (e.g. "region")
    n_components:      int   = 2,
    umap_neighbors:    int   = 15,
    umap_min_dist:     float = 0.1,
    tsne_perplexity:   float = 30.0,
    seed:              int   = 42,
    percentile:         int = 90,
    save_path:         str   = None,
    dpi:               int   = 180,
    legend:            bool = True,
    size_points:       int = 45,
):
    """
    Plots uniqueness scores over three dimensionality reductions:
    PCA, UMAP, t-SNE.

    Each point is coloured by its uniqueness score (continuous colormap).
    Optionally a categorical field (e.g. "region") can be used instead.

    Parameters
    ----------
    dataset           : FiftyOne dataset with embeddings + uniqueness fields
    embeddings_field  : field containing L2-normalised DINOv3 embeddings
    uniqueness_field  : field containing pre-computed uniqueness scores [0,1]
    color_field       : optional string field to colour by instead of uniqueness
    n_components      : dimensionality of the projection (default 2)
    umap_neighbors    : UMAP n_neighbors
    umap_min_dist     : UMAP min_dist
    tsne_perplexity   : t-SNE perplexity
    seed              : random seed for reproducibility
    save_path         : optional path to save the figure
    dpi               : output resolution
    percentile        : which percentile to consider as Unique samples
    legend            : wether or not create legend  
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize
    from umap import UMAP
    from sklearn.manifold import TSNE

    BG    = "#ececf0"
    WHITE = "#0a0000"
    MUTED = "#2D2D2E"

    # ── Load data from FiftyOne ───────────────────────────────────────────────
    print("Loading embeddings and scores ...")
    embeddings  = []
    scores      = []
    color_vals  = []
    sample_ids  = []

    for sample in dataset.iter_samples(progress=True):
        emb   = sample.get_field(embeddings_field)
        score = sample.get_field(uniqueness_field)
        if emb is None or score is None:
            continue
        embeddings.append(np.array(emb, dtype=np.float32))
        scores.append(float(score))
        sample_ids.append(sample.id)
        if color_field:
            color_vals.append(sample.get_field(color_field))

    embs   = normalize(np.stack(embeddings), norm="l2")
    scores = np.array(scores)
    N      = len(embs)
    print(f"  N={N}  |  score range [{scores.min():.3f}, {scores.max():.3f}]")

    # ── Colour mapping ────────────────────────────────────────────────────────
    if color_field and color_vals:
        categories  = sorted(set(str(v) for v in color_vals))
        cat_palette = ["#00d4ff", "#ff4444", "#ffaa00", "#aaffaa",
                       "#ff88ff", "#ffffff", "#f4a261", "#e76f51"]
        cat_to_col  = {c: cat_palette[i % len(cat_palette)]
                       for i, c in enumerate(categories)}
        point_colors = [cat_to_col[str(v)] for v in color_vals]
        cmap_label   = color_field
        use_cmap     = False
    else:
        point_colors = scores
        cmap_label   = "Uniqueness score"
        use_cmap     = True

    # ── Dimensionality reductions ─────────────────────────────────────────────
    print("Computing PCA ...")
    pca_coords = PCA(n_components=2, random_state=seed).fit_transform(embs)

    print("Computing UMAP ...")
    umap_coords = UMAP(
        n_components=2,
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
        random_state=seed,
        n_jobs=1,
    ).fit_transform(embs)

    print("Computing t-SNE ...")
    tsne_coords = TSNE(
        n_components=2,
        perplexity=tsne_perplexity,
        random_state=seed,
    ).fit_transform(embs)

    coords = [pca_coords, umap_coords, tsne_coords]
    names  = ["PCA", "UMAP", "t-SNE"]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
    fig.suptitle(
        f"Embedding Manifold \n Coloured by {cmap_label}  (N={N})",
        color=WHITE, fontsize=14, fontweight="bold", y=1.02,
    )

    scatter_kwargs = dict(s=size_points, linewidths=0, alpha=0.75)
    cmap           = "YlOrRd"

    for ax, coord, name in zip(axes, coords, names):
        ax.set_facecolor(BG)

        if use_cmap:
            sc = ax.scatter(
                coord[:, 0], coord[:, 1],
                c=point_colors, cmap=cmap, vmin=0, vmax=1,
                **scatter_kwargs,
            )
        else:
            # categorical — plot each category separately for legend
            for cat in categories:
                mask = np.array([str(v) == cat for v in color_vals])
                ax.scatter(
                    coord[mask, 0], coord[mask, 1],
                    c=cat_to_col[cat], label=cat,
                    **scatter_kwargs,
                )

        # Highlight top-10% most unique samples with a ring
        top_mask = scores >= np.percentile(scores, percentile)
        ax.scatter(
            coord[top_mask, 0], coord[top_mask, 1], facecolors="none", edgecolors="#090009",marker='*',
            linewidths=0.8, alpha=0.9, label=f"Top {str(percentile)}% unique",
        )

        ax.set_title(name, color=WHITE, fontsize=12, pad=6)
        ax.tick_params(colors=MUTED, labelsize=7)
        ax.set_xlabel("Dim 1", color=MUTED, fontsize=8)
        ax.set_ylabel("Dim 2", color=MUTED, fontsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444466")
        ax.grid(color="#333355", linewidth=0.4, alpha=0.4)

        if not use_cmap:
            if legend:
                ax.legend(fontsize=7, facecolor="#2a2a4a",
                        edgecolor="#555577", labelcolor=WHITE,
                        markerscale=2, loc="best")
        else:
            if legend:
                ax.legend(
                    handles=[plt.scatter([], [], s=35, facecolors="none",
                                        edgecolors="#ffffff", linewidths=0.8)],
                    labels=["Top 10% unique"],
                    fontsize=7, facecolor="#2a2a4a",
                    edgecolor="#555577", labelcolor=WHITE,
                )

    # Shared colorbar for uniqueness
    if use_cmap:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        sm      = plt.cm.ScalarMappable(cmap=cmap,
                                        norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(cmap_label, color=MUTED, fontsize=9)
        cbar.ax.tick_params(colors=MUTED, labelsize=7)

    plt.tight_layout(rect=[0, 0, 0.91, 1] if use_cmap else [0, 0, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved → {save_path}")

    plt.show()
    return fig, {"pca": pca_coords, "umap": umap_coords, "tsne": tsne_coords}


def plot_patch_embedding_landscape(
    dataset,
    embeddings_field:  str   = "ground_truth.detections.patch_embeddings",
    color_field:       str   = None,   # sample-level field e.g. "region", "mission_name"
    umap_neighbors:    int   = 15,
    umap_min_dist:     float = 0.1,
    tsne_perplexity:   float = 30.0,
    seed:              int   = 42,
    save_path:         str   = None,
    dpi:               int   = 180,
):
    """
    Plots dugong patch embeddings (one per detection) over PCA, UMAP, t-SNE.

    Embeddings are stored at detection level (ground_truth.detections.patch_embeddings).
    The color_field is read from the SAMPLE level (e.g. "region", "mission_name").

    Parameters
    ----------
    dataset          : FiftyOne dataset
    embeddings_field : dot-path to detection-level embedding field
    color_field      : sample-level string field for colouring points.
                       If None, points are coloured by detection index (grey).
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize
    from sklearn.manifold import TSNE
    from umap import UMAP
    import numpy as np
    import matplotlib.pyplot as plt

    BG    = "#0d1117"
    PANEL = "#161b22"
    WHITE = "#f0f6fc"
    MUTED = "#8b949e"
    GRID  = "#21262d"

    # ── Collect embeddings + colour values ────────────────────────────────────
    print("Loading patch embeddings ...")
    embeddings  = []
    color_vals  = []

    for sample in dataset.iter_samples(progress=True):
        if sample.ground_truth is None:
            continue
        col_val = sample.get_field(color_field) if color_field else None

        for det in sample.ground_truth.detections:
            emb = det.get_field("patch_embeddings")
            if emb is None:
                continue
            embeddings.append(np.array(emb, dtype=np.float32))
            color_vals.append(str(col_val) if col_val is not None else "unknown")

    if not embeddings:
        raise ValueError("No patch embeddings found. Run compute_patch_embeddings first.")

    embs = normalize(np.stack(embeddings), norm="l2")
    N    = len(embs)
    print(f"  N={N} dugong patch embeddings  dim={embs.shape[1]}")

    # ── Colour mapping ────────────────────────────────────────────────────────
    palette = [
        "#00d4ff", "#ff4444", "#ffaa00", "#aaffaa",
        "#ff88ff", "#f4a261", "#e76f51", "#3fb950",
    ]

    if color_field:
        categories  = sorted(set(color_vals))
        cat_to_col  = {c: palette[i % len(palette)] for i, c in enumerate(categories)}
        point_colors = [cat_to_col[v] for v in color_vals]
        legend_info  = cat_to_col
        title_suffix = f"coloured by '{color_field}'"
    else:
        point_colors = ["#00d4ff"] * N
        legend_info  = None
        title_suffix = ""

    # ── Dimensionality reductions ─────────────────────────────────────────────
    print("Computing PCA ...")
    pca_coords  = PCA(n_components=2, random_state=seed).fit_transform(embs)

    print("Computing UMAP ...")
    umap_coords = UMAP(
        n_components=2,
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
        random_state=seed,
        n_jobs=1,
    ).fit_transform(embs)

    print("Computing t-SNE ...")
    tsne_coords = TSNE(
        n_components=2,
        perplexity=min(tsne_perplexity, N - 1),
        random_state=seed,
    ).fit_transform(embs)

    coords = [pca_coords, umap_coords, tsne_coords]
    names  = ["PCA", "UMAP", "t-SNE"]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
    fig.suptitle(
        f"Dugong Patch Embedding Space  (N={N})  {title_suffix}",
        color=WHITE, fontsize=13, fontweight="bold", y=1.02,
    )

    for ax, coord, name in zip(axes, coords, names):
        ax.set_facecolor(PANEL)

        if color_field and legend_info:
            for cat, col in legend_info.items():
                mask = np.array([v == cat for v in color_vals])
                ax.scatter(
                    coord[mask, 0], coord[mask, 1],
                    c=col, s=14, alpha=0.7, linewidths=0,
                    label=f"{cat} ({mask.sum()})",
                )
            ax.legend(
                fontsize=8, facecolor="#21262d",
                edgecolor="#30363d", labelcolor=WHITE,
                markerscale=2, loc="best",
            )
        else:
            ax.scatter(
                coord[:, 0], coord[:, 1],
                c=point_colors, s=14, alpha=0.7, linewidths=0,
            )

        ax.set_title(name, color=WHITE, fontsize=12, pad=6)
        ax.set_xlabel("Dim 1", color=MUTED, fontsize=8)
        ax.set_ylabel("Dim 2", color=MUTED, fontsize=8)
        ax.tick_params(colors=MUTED, labelsize=7)
        ax.grid(color=GRID, linewidth=0.4, alpha=0.6)
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved → {save_path}")
    plt.show()
    return fig, {"pca": pca_coords, "umap": umap_coords, "tsne": tsne_coords}



def compute_clustering_detections(
    dataset,
    embeddings_field: str   = "ground_truth.detections.patch_embeddings",
    cluster_field:    str   = "ground_truth.detections.patch_cluster_label",
    n_clusters:       int   = 13,
    n_init:           int   = 10,
    seed:             int   = 42,
    save:             bool  = True,
    verbose:          bool  = True,
    **kmeans_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Runs KMeans clustering in object (detection) embedding space.
    Reads embeddings from detection-level field and writes cluster labels
    back to each detection.

    Parameters
    ----------
    dataset           : FiftyOne dataset
    embeddings_field  : dot-path to detection-level embedding
                        e.g. "ground_truth.detections.patch_embeddings"
    cluster_field     : dot-path to write cluster label to each detection
                        e.g. "ground_truth.detections.patch_cluster_label"
    n_clusters        : number of KMeans clusters
    n_init            : KMeans n_init
    seed              : random seed
    save              : write cluster labels to dataset
    verbose           : print progress and stats
    **kmeans_kwargs   : forwarded to sklearn KMeans
                        (n_clusters, n_init, random_state are set explicitly)

    Returns
    -------
    cluster_labels  : np.ndarray (N_detections,)
    centroids       : np.ndarray (K, D)
    embeddings_norm : np.ndarray (N_detections, D) — L2-normalised
    det_ids         : list of N detection IDs (same order)
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize

    def _log(msg, level="info"):
        if not verbose:
            return
        prefix = {"info": "  ", "success": "✓ ", "warn": "⚠ "}.get(level, "  ")
        print(f"{prefix}{msg}")

    # ── Load detection-level embeddings ───────────────────────────────────────
    _log(f"Loading detection embeddings from '{embeddings_field}' ...")

    # Parse the detection field name (last part after detections.)
    # e.g. "ground_truth.detections.patch_embeddings" → "patch_embeddings"
    det_emb_key     = embeddings_field.split(".")[-1]
    det_cluster_key = cluster_field.split(".")[-1]

    embeddings = []
    det_ids    = []
    # Track (sample_id, det_idx) to write labels back efficiently
    det_locators = []

    for sample in dataset.iter_samples(progress=verbose):
        if sample.ground_truth is None:
            continue
        for det_idx, det in enumerate(sample.ground_truth.detections):
            emb = det.get_field(det_emb_key)
            if emb is None:
                continue
            embeddings.append(np.array(emb, dtype=np.float32))
            det_ids.append(det.id)
            det_locators.append((sample.id, det_idx))

    if not embeddings:
        raise ValueError(
            f"No embeddings found at '{embeddings_field}'. "
            "Run compute_patch_embeddings first."
        )

    embs_norm = normalize(np.stack(embeddings), norm="l2")
    N         = len(embs_norm)
    _log(f"Loaded {N} detection embeddings  dim={embs_norm.shape[1]}")

    # ── KMeans ────────────────────────────────────────────────────────────────
    kmeans_kwargs.pop("n_clusters",   None)
    kmeans_kwargs.pop("n_init",       None)
    kmeans_kwargs.pop("random_state", None)

    _log(
        f"Running KMeans (k={n_clusters}, n_init={n_init}, seed={seed}"
        + (f", extra={kmeans_kwargs}" if kmeans_kwargs else "")
        + ") ..."
    )

    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=n_init,
        random_state=seed,
        **kmeans_kwargs,
    )
    cluster_labels = kmeans.fit_predict(embs_norm)
    centroids      = kmeans.cluster_centers_

    # ── Cluster stats ─────────────────────────────────────────────────────────
    cluster_sizes = np.bincount(cluster_labels, minlength=n_clusters)
    _log(f"Cluster sizes: min={cluster_sizes.min()}  "
         f"mean={cluster_sizes.mean():.1f}  max={cluster_sizes.max()}")
    _log(f"Empty clusters: {(cluster_sizes == 0).sum()} / {n_clusters}")

    # ── Write labels back to detections ───────────────────────────────────────
    if save:
        _log(f"Writing cluster labels to '{cluster_field}' ...")

        # Build lookup: det_id → cluster_label
        id_to_label = dict(zip(det_ids, cluster_labels.tolist()))

        for sample in dataset.iter_samples(autosave=True, progress=verbose):
            if sample.ground_truth is None:
                continue
            changed = False
            for det in sample.ground_truth.detections:
                label = id_to_label.get(det.id)
                if label is not None:
                    det[det_cluster_key] = int(label)
                    changed = True

        _log(f"Cluster labels saved to '{cluster_field}'.", level="success")

    return cluster_labels, centroids, embs_norm, det_ids




def plot_clusters_full_images(
    dataset,
    field_cluster:      str  = "cluster_label",
    num_img_per_row:    int  = 5,
    size:               str  = "a4",
    format:             str  = "png",
    field_image_name:   str  = "name_field",
    output_path:        str  = None,
    seed:               int  = 42,
    verbose:            bool = True,
) -> Image.Image:
    """
    Plot one row per cluster label, each row showing a random sample of
    num_img_per_row images from that cluster. Saves to output_path if given
    and always returns the composed PIL Image.

    Layout
    ------
    - Page size: A4 (2480×3508 px at 300 dpi) or A5 (1748×2480 px at 300 dpi),
      portrait orientation.
    - Left margin: vertical cluster label, white text on a coloured band.
    - Each row: num_img_per_row thumbnail images with a short filename title
      below each.
    - Thumbnail size is computed automatically to fill the usable row width.
    - If a cluster has more images than num_img_per_row, a random sample is
      drawn (seeded for reproducibility). If it has fewer, all are shown and
      the remaining slots are left blank.

    Parameters
    ----------
    dataset          : FiftyOne dataset or view
    field_cluster    : sample field containing integer cluster labels
    num_img_per_row  : number of images per cluster row (default 5)
    size             : "a4" or "a5" (default "a4")
    format           : output file format — "png" or "jpg" (default "png")
    field_image_name : sample field containing a display name string.
                       Falls back to the filepath basename if the field is
                       None or missing on a sample.
    output_path      : file path to save the figure. If None the figure is
                       not saved but is still returned.
    seed             : random seed for the per-cluster image sampling
    verbose          : print progress

    Returns
    -------
    PIL.Image.Image — the fully composed figure
    """

    rng = random.Random(seed)

    # ── Page dimensions (300 dpi) ─────────────────────────────────────────────
    PAGE_SIZES = {
        "a4": (2480, 3508),
        "a5": (1748, 2480),
    }
    size_key = size.lower()
    if size_key not in PAGE_SIZES:
        raise ValueError(f"size='{size}' not supported. Choose 'a4' or 'a5'.")

    PAGE_W, PAGE_H = PAGE_SIZES[size_key]
    DPI = 300

    # ── Fixed layout constants ────────────────────────────────────────────────
    LABEL_BAND_W  = 90          # width of the left cluster-label band (px)
    H_PADDING     = 24          # horizontal gap between thumbnails
    V_PADDING     = 28          # vertical gap between rows
    TITLE_H       = 42          # height reserved below each thumbnail for name
    TOP_MARGIN    = 60          # top margin
    BOTTOM_MARGIN = 60          # bottom margin
    INNER_PAD     = 16          # padding inside the label band (top/bottom)

    # ── Compute optimal thumbnail size ────────────────────────────────────────
    usable_w   = PAGE_W - LABEL_BAND_W - (num_img_per_row + 1) * H_PADDING
    THUMB_W    = usable_w // num_img_per_row
    THUMB_W    = max(THUMB_W, 80)           # floor to avoid degenerate sizes
    THUMB_H    = int(THUMB_W * 0.75)        # 4:3 aspect ratio for thumbnails

    ROW_H = THUMB_H + TITLE_H + V_PADDING  # total height per cluster row

    if verbose:
        print(f"  Page: {PAGE_W}×{PAGE_H} px  ({size.upper()}  {DPI} dpi)")
        print(f"  Thumbnail: {THUMB_W}×{THUMB_H} px")
        print(f"  Row height: {ROW_H} px")

    # ── Load data from FiftyOne ───────────────────────────────────────────────
    if verbose:
        print(f"  Loading samples from field '{field_cluster}' ...")

    clusters: dict[int, list[dict]] = {}

    for sample in dataset.iter_samples(progress=verbose):
        label = sample.get_field(field_cluster)
        if label is None:
            continue
        label = int(label)

        # resolve display name
        name = sample.get_field(field_image_name)
        if not name:
            name = Path(sample.filepath).stem.split('-')[0]

        clusters.setdefault(label, []).append({
            "filepath": sample.filepath,
            "name":     str(name),
        })

    if not clusters:
        raise ValueError(
            f"No samples found with field '{field_cluster}' set. "
            "Run compute_clustering first."
        )

    sorted_labels = sorted(clusters.keys())
    n_rows        = len(sorted_labels)

    if verbose:
        print(f"  {n_rows} clusters found  |  "
              f"{sum(len(v) for v in clusters.values())} total samples")

    # ── Compute canvas height ─────────────────────────────────────────────────
    CANVAS_H = TOP_MARGIN + n_rows * ROW_H + BOTTOM_MARGIN
    # if content overflows page height, let it grow (multi-page not supported)
    CANVAS_H = max(CANVAS_H, PAGE_H)

    # ── Colour palette for cluster label bands ────────────────────────────────
    BAND_COLOURS = [
        (83,  74,  183),   # purple
        (29,  158, 117),   # teal
        (216, 90,  48),    # coral
        (212, 83,  126),   # pink
        (55,  95,  165),   # blue
        (99,  153, 34),    # green
        (186, 117, 23),    # amber
        (136, 135, 128),   # gray
    ]

    def band_color(label: int) -> tuple:
        return BAND_COLOURS[label % len(BAND_COLOURS)]

    # ── Font loading ──────────────────────────────────────────────────────────
    def _font(size_pt: int, bold: bool = False):
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
        for path in candidates:
            try:
                return ImageFont.truetype(path, size_pt)
            except (IOError, OSError):
                continue
        return ImageFont.load_default()

    FONT_TITLE  = _font(22)
    FONT_LABEL  = _font(28, bold=True)

    # ── Create canvas ─────────────────────────────────────────────────────────
    canvas = Image.new("RGB", (PAGE_W, CANVAS_H), color=(255, 255, 255))
    draw   = ImageDraw.Draw(canvas)

    # ── Draw each cluster row ─────────────────────────────────────────────────
    for row_idx, label in enumerate(sorted_labels):
        samples_in_cluster = clusters[label]

        # random sample if more than num_img_per_row
        if len(samples_in_cluster) > num_img_per_row:
            chosen = rng.sample(samples_in_cluster, num_img_per_row)
        else:
            chosen = samples_in_cluster   # use all, may be fewer

        row_y = TOP_MARGIN + row_idx * ROW_H

        # ── Cluster label band ────────────────────────────────────────────────
        bc = band_color(label)
        draw.rectangle(
            [0, row_y, LABEL_BAND_W, row_y + ROW_H - V_PADDING],
            fill=bc,
        )

        # vertical text — draw on a small rotated sub-image then paste
        label_text  = f"cluster {label} N={len(samples_in_cluster)}"
        band_height = ROW_H - V_PADDING

        # measure text size
        bbox  = FONT_LABEL.getbbox(label_text)
        txt_w = bbox[2] - bbox[0]
        txt_h = bbox[3] - bbox[1]

        # create a horizontal text image, rotate 90° CCW
        txt_img = Image.new("RGBA", (txt_w + 2 * INNER_PAD, txt_h + 2 * INNER_PAD),
                            color=(0, 0, 0, 0))
        txt_draw = ImageDraw.Draw(txt_img)
        txt_draw.text((INNER_PAD, INNER_PAD), label_text,
                      font=FONT_LABEL, fill=(255, 255, 255, 255))

        rotated = txt_img.rotate(90, expand=True)

        # centre the rotated label in the band
        paste_x = (LABEL_BAND_W - rotated.width)  // 2
        paste_y = row_y + (band_height - rotated.height) // 2

        canvas.paste(rotated, (paste_x, paste_y), rotated)

        # ── Thumbnails ────────────────────────────────────────────────────────
        for col_idx, sample_info in enumerate(chosen):
            thumb_x = (LABEL_BAND_W
                       + (col_idx + 1) * H_PADDING
                       + col_idx * THUMB_W)
            thumb_y = row_y + V_PADDING // 2

            # load and resize image
            try:
                img = Image.open(sample_info["filepath"]).convert("RGB")
                img.thumbnail((THUMB_W, THUMB_H), Image.LANCZOS)

                # centre on a fixed-size background to keep grid aligned
                bg = Image.new("RGB", (THUMB_W, THUMB_H), (230, 230, 230))
                offset_x = (THUMB_W - img.width)  // 2
                offset_y = (THUMB_H - img.height) // 2
                bg.paste(img, (offset_x, offset_y))
                canvas.paste(bg, (thumb_x, thumb_y))

                # thin border
                draw.rectangle(
                    [thumb_x, thumb_y,
                     thumb_x + THUMB_W - 1, thumb_y + THUMB_H - 1],
                    outline=(180, 180, 180), width=1,
                )

            except Exception as exc:
                # draw a placeholder if image fails to load
                draw.rectangle(
                    [thumb_x, thumb_y,
                     thumb_x + THUMB_W, thumb_y + THUMB_H],
                    fill=(210, 210, 210), outline=(180, 180, 180),
                )
                if verbose:
                    print(f"    WARN could not load {sample_info['filepath']}: {exc}")

            # image title below thumbnail
            title_text = sample_info["name"]
            # truncate if too long
            max_chars = THUMB_W // 10
            if len(title_text) > max_chars:
                title_text = title_text[:max_chars - 1] + "…"

            title_bbox = FONT_TITLE.getbbox(title_text)
            title_w    = title_bbox[2] - title_bbox[0]
            title_x    = thumb_x + (THUMB_W - title_w) // 2
            title_y    = thumb_y + THUMB_H + 6

            draw.text(
                (title_x, title_y),
                title_text,
                font=FONT_TITLE,
                fill=(60, 60, 60),
            )

        # ── Cluster size annotation (right of band) ───────────────────────────
        n_total = len(samples_in_cluster)
        ann_text = f"n={n_total}"
        ann_bbox = FONT_TITLE.getbbox(ann_text)
        ann_w    = ann_bbox[2] - ann_bbox[0]
        ann_x    = LABEL_BAND_W + H_PADDING
        ann_y    = row_y + (ROW_H - V_PADDING - (ann_bbox[3] - ann_bbox[1])) // 2 + THUMB_H + 6

        draw.text((ann_x, ann_y), ann_text, font=FONT_TITLE, fill=(120, 120, 120))

        if verbose:
            print(f"  cluster {label:>3}: {n_total:>4} samples  "
                  f"showing {len(chosen)}")

    # ── Save ──────────────────────────────────────────────────────────────────
    if output_path:
        fmt   = format.upper().replace("JPG", "JPEG")
        extra = {"dpi": (DPI, DPI)}
        if fmt == "JPEG":
            extra["quality"] = 92
        canvas.save(output_path, format=fmt, **extra)
        if verbose:
            print(f"  Saved → {output_path}")

    return canvas



def plot_cluster_uniqueness_overview(
    dataset,
 
    # embedding source
    embeddings_field:         str   = "full_embeddings",
 
    # score fields — one per row
    global_uniqueness_field:  str   = "uniqueness_score",
    cluster_uniqueness_field: str   = "cluster_uniqueness_score",
    soft_coverage_field:      str   = "soft_coverage_score",
 
    # cluster membership — drives marker shape in rows 1 & 2
    cluster_field:            str   = "cluster_label",
 
    # dimensionality reduction
    umap_neighbors:           int   = 15,
    umap_min_dist:            float = 0.1,
    tsne_perplexity:          float = 30.0,
    seed:                     int   = 42,
 
    # visual
    score_cmap:               str   = "YlOrRd",
    cluster_cmap:             str   = "tab20",   # used for legend markers only
    point_size:               int   = 22,
    alpha:                    float = 0.80,
 
    # output
    figsize:                  tuple = (18, 16),
    dpi:                      int   = 150,
    save_path:                str   = None,
    verbose:                  bool  = True,
) -> plt.Figure:
    """
    3 × 3 overview figure linking embedding geometry with three uniqueness
    signals. Cluster identity is encoded by marker shape — each cluster gets
    a unique marker (circle, square, diamond, star, …) so the score colormap
    fill is visually unambiguous.
 
    Parameters
    ----------
    dataset                  : FiftyOne dataset or view
    embeddings_field         : L2-normalised embedding field
    global_uniqueness_field  : per-sample global uniqueness score [0,1]
    cluster_uniqueness_field : per-sample within-cluster uniqueness [0,1]
    soft_coverage_field      : per-sample soft penalty coverage score [0,1]
    cluster_field            : integer cluster label field
    umap_neighbors           : UMAP n_neighbors
    umap_min_dist            : UMAP min_dist
    tsne_perplexity          : t-SNE perplexity
    seed                     : random seed for UMAP and t-SNE
    score_cmap               : matplotlib colormap name for score fill
                               (continuous, e.g. "YlOrRd", "viridis",
                                "plasma", "coolwarm")
    cluster_cmap             : matplotlib colormap name used only for
                               legend marker colours (categorical,
                               e.g. "tab20", "Set1", "Paired")
    point_size               : base scatter marker size (s= parameter).
                               Larger markers (*, P, X …) are automatically
                               scaled down by _LARGE_MARKER_SCALE.
    alpha                    : scatter marker opacity
    figsize                  : figure size in inches
    dpi                      : figure resolution
    save_path                : optional output path (.png / .pdf / .svg)
    verbose                  : print progress
 
    Returns
    -------
    matplotlib.figure.Figure
    """
    def _safe_float(val) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return float("nan")
    _MARKERS = [
    'o',   # circle
    's',   # square
    'D',   # diamond
    '^',   # triangle up
    'v',   # triangle down
    'P',   # plus (filled)
    '*',   # star
    'X',   # x (filled)
    'h',   # hexagon 1
    'p',   # pentagon
    '<',   # triangle left
    '>',   # triangle right
    'H',   # hexagon 2
    '8',   # octagon
    'd',   # thin diamond
    '1',   # tri down
    '2',   # tri up
    '3',   # tri left
    '4',   # tri right
    '+',   # plus
    ]
 
    # Markers that render visually larger at the same s= value — scale them down
    _LARGE_MARKERS = {'*', 'P', 'X', 'p', 'H', '8'}
    _LARGE_MARKER_SCALE = 0.65

    def _log(msg):
        if verbose:
            print(f"  {msg}")
 
    # ── Load data from FiftyOne ───────────────────────────────────────────────
    _log(f"Loading embeddings from '{embeddings_field}' ...")
 
    embeddings     = []
    global_uniq    = []
    cluster_uniq   = []
    soft_cov       = []
    cluster_labels = []
    sample_ids     = []
 
    for sample in dataset.iter_samples(progress=verbose):
        emb = sample.get_field(embeddings_field)
        if emb is None:
            continue
 
        embeddings.append(np.array(emb, dtype=np.float32))
        sample_ids.append(sample.id)
 
        global_uniq.append(  _safe_float(sample.get_field(global_uniqueness_field)))
        cluster_uniq.append( _safe_float(sample.get_field(cluster_uniqueness_field)))
        soft_cov.append(     _safe_float(sample.get_field(soft_coverage_field)))
 
        cl = sample.get_field(cluster_field)
        cluster_labels.append(int(cl) if cl is not None else -1)
 
    if not embeddings:
        raise ValueError(f"No embeddings found in field '{embeddings_field}'.")
 
    embs           = normalize(np.stack(embeddings), norm="l2")
    N              = len(embs)
    global_uniq    = np.array(global_uniq,    dtype=np.float64)
    cluster_uniq   = np.array(cluster_uniq,   dtype=np.float64)
    soft_cov       = np.array(soft_cov,       dtype=np.float64)
    cluster_labels = np.array(cluster_labels, dtype=int)
 
    _log(f"Loaded {N} samples")
 
    # ── Check score field availability ────────────────────────────────────────
    def _field_ok(arr, name):
        ok = not np.all(np.isnan(arr))
        if not ok:
            warnings.warn(
                f"Field '{name}' has no valid values — row will be empty. "
                "Run the corresponding compute_* function first.",
                UserWarning, stacklevel=2,
            )
        return ok
 
    row_ok = [
        _field_ok(global_uniq,  global_uniqueness_field),
        _field_ok(cluster_uniq, cluster_uniqueness_field),
        _field_ok(soft_cov,     soft_coverage_field),
    ]
 
    # ── Cluster / marker setup ────────────────────────────────────────────────
    unique_clusters = sorted(set(cluster_labels[cluster_labels >= 0]))
    n_clusters      = len(unique_clusters)
    cl_idx_map      = {c: i for i, c in enumerate(unique_clusters)}
 
    # marker shape per cluster index
    def _marker_for(cluster_id: int) -> str:
        return _MARKERS[cl_idx_map[cluster_id] % len(_MARKERS)]
 
    # effective point size per cluster (scale down visually large markers)
    def _size_for(cluster_id: int) -> float:
        m = _marker_for(cluster_id)
        return point_size * (_LARGE_MARKER_SCALE if m in _LARGE_MARKERS else 1.0)
 
    # cluster colour for legend only
    cl_cmap = plt.get_cmap(cluster_cmap)
 
    def _legend_color(cluster_id: int):
        return cl_cmap(cl_idx_map[cluster_id] / max(n_clusters - 1, 1))
 
    # ── Dimensionality reductions (computed once) ─────────────────────────────
    _log("Computing UMAP ...")
    umap_coords = UMAP(
        n_components=2,
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
        random_state=seed,
        n_jobs=1,
    ).fit_transform(embs)
 
    _log("Computing t-SNE ...")
    tsne_coords = TSNE(
        n_components=2,
        perplexity=min(tsne_perplexity, N - 1),
        random_state=seed,
    ).fit_transform(embs)
 
    _log("Computing PCA ...")
    pca_coords = PCA(n_components=2, random_state=seed).fit_transform(embs)
 
    all_coords = [umap_coords, tsne_coords, pca_coords]
    col_titles = ["UMAP", "t-SNE", "PCA"]
 
    # ── Score arrays and row metadata ─────────────────────────────────────────
    score_arrays = [global_uniq, cluster_uniq, soft_cov]
    row_labels   = [
        "Global\nuniqueness",
        "Cluster\nuniqueness",
        "Soft coverage\nscore",
    ]
 
    score_norm = mcolors.Normalize(vmin=0, vmax=1)
    s_cmap     = plt.get_cmap(score_cmap)
 
    # ── Figure / GridSpec ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("white")
 
    gs = GridSpec(
        3, 4,
        figure=fig,
        width_ratios=[1, 1, 1, 0.06],
        hspace=0.40,
        wspace=0.12,
        left=0.08, right=0.94,
        top=0.93,  bottom=0.06,
    )
 
    # ── Row labels (vertical, left margin) ───────────────────────────────────
    for rl, ry in zip(row_labels, [0.78, 0.50, 0.22]):
        fig.text(
            0.01, ry, rl,
            va="center", ha="left",
            fontsize=10, fontweight="500",
            color="#2C2C2A",
            rotation=90,
            transform=fig.transFigure,
        )
 
    axes_grid = []
 
    for row in range(3):
        row_axes    = []
        scores      = score_arrays[row]
        ok          = row_ok[row]
        plot_scores = np.where(np.isnan(scores), 0.0, scores)
 
        for col in range(3):
            ax     = fig.add_subplot(gs[row, col])
            coords = all_coords[col]
 
            # ── placeholder if field missing ──────────────────────────────────
            if not ok:
                ax.set_facecolor("#F1EFE8")
                ax.text(
                    0.5, 0.5,
                    "field not found\nrun compute_* first",
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=9, color="#888780", style="italic",
                )
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_edgecolor("#D3D1C7")
                row_axes.append(ax)
                continue
 
            ax.set_facecolor("#F8F7F5")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#D3D1C7")
                sp.set_linewidth(0.5)
 
            # ── Row 0 — global uniqueness, single marker shape ────────────────
            if row == 0:
                ax.scatter(
                    coords[:, 0], coords[:, 1],
                    c          = plot_scores,
                    cmap       = score_cmap,
                    norm       = score_norm,
                    s          = point_size,
                    alpha      = alpha,
                    marker     = "o",
                    linewidths = 0,
                    rasterized = True,
                )
 
            # ── Rows 1 & 2 — one scatter call per cluster (marker shape) ──────
            else:
                # draw unlabelled samples first (cluster_label == -1)
                mask_none = cluster_labels == -1
                if mask_none.any():
                    ax.scatter(
                        coords[mask_none, 0], coords[mask_none, 1],
                        c          = plot_scores[mask_none],
                        cmap       = score_cmap,
                        norm       = score_norm,
                        s          = point_size * 0.6,
                        alpha      = alpha * 0.5,
                        marker     = "o",
                        linewidths = 0,
                        rasterized = True,
                    )
 
                for c_id in unique_clusters:
                    mask   = cluster_labels == c_id
                    if not mask.any():
                        continue
                    marker = _marker_for(c_id)
                    sz     = _size_for(c_id)
 
                    ax.scatter(
                        coords[mask, 0], coords[mask, 1],
                        c          = plot_scores[mask],
                        cmap       = score_cmap,
                        norm       = score_norm,
                        s          = sz,
                        alpha      = alpha,
                        marker     = marker,
                        linewidths = 0,
                        rasterized = True,
                    )
 
            # stat annotation
            valid = plot_scores[~np.isnan(scores)]
            if len(valid):
                ax.set_title(
                    f"μ={valid.mean():.2f}  σ={valid.std():.2f}",
                    fontsize=8, color="#5F5E5A", pad=3,
                )
 
            # column header on top row only
            if row == 0:
                ax.set_xlabel(
                    col_titles[col],
                    fontsize=11, fontweight="500",
                    labelpad=5, color="#2C2C2A",
                )
 
            row_axes.append(ax)
 
        # ── Per-row colorbar ──────────────────────────────────────────────────
        cbar_ax = fig.add_subplot(gs[row, 3])
        sm = ScalarMappable(cmap=score_cmap, norm=score_norm)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cbar_ax)
        cb.set_label(
            row_labels[row].replace("\n", " "),
            fontsize=8, color="#5F5E5A",
        )
        cb.ax.tick_params(labelsize=7, colors="#5F5E5A")
        cb.outline.set_linewidth(0.5)
 
        axes_grid.append(row_axes)
 
    # ── Column headers ────────────────────────────────────────────────────────
    for col, title in enumerate(col_titles):
        axes_grid[0][col].set_xlabel(
            title, fontsize=11, fontweight="500",
            labelpad=5, color="#2C2C2A",
        )
 
    # ── Cluster legend — marker shape + neutral fill ──────────────────────────
    if n_clusters <= 20:
        handles = [
            Line2D(
                [0], [0],
                marker          = _marker_for(c),
                color           = "w",
                markerfacecolor = _legend_color(c),
                markeredgecolor = "none",
                markersize      = 8,
                label           = f"cluster {c}",
            )
            for c in unique_clusters
        ]
        fig.legend(
            handles          = handles,
            title            = "cluster  (shape = cluster id)",
            title_fontsize   = 8,
            fontsize         = 7,
            loc              = "lower center",
            ncol             = min(n_clusters, 10),
            frameon          = True,
            framealpha       = 0.92,
            edgecolor        = "#D3D1C7",
            bbox_to_anchor   = (0.47, 0.0),
        )
    else:
        _log(f"Skipping cluster legend — {n_clusters} clusters exceeds 20.")
 
    # ── Main title ────────────────────────────────────────────────────────────
    fig.suptitle(
        "Embedding space — uniqueness and coverage overview",
        fontsize=13, fontweight="500", color="#2C2C2A", y=0.97,
    )
 
    plt.tight_layout(rect=[0.05, 0.05, 1.0, 0.96])
 
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        _log(f"Saved → {save_path}")
 
    return fig


def _safe_float(val):
    """Return float or NaN for None/missing values."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")



def plot_cluster_cosine_similarity(
    dataset,

    # data fields
    embeddings_field:   str   = "full_embeddings",
    cluster_field:      str   = "cluster_label",
    field_image_name:   str   = "name_field",

    # layout
    n_compare:          int   = 4,
    size:               str   = "a4",
    format:             str   = "png",

    # sampling & ordering
    seed:               int   = 42,
    sort_by_similarity: bool  = True,

    # visual
    ref_border_color:   tuple = (83, 74, 183),
    sim_cmap:           str   = "RdYlGn",
    show_sim_bar:       bool  = True,
    sim_bar_h:          int   = 12,

    # output
    output_path:        str   = None,
    verbose:            bool  = True,
) -> "Image.Image":
    """
    Plot one row per cluster. Column 0 is a randomly chosen reference image.
    Columns 1..n_compare show the most cosine-similar images from the same
    cluster (or a random sample when sort_by_similarity=False), each titled
    with its name and cosine similarity to the reference.

    Since embeddings are L2-normalised, cosine similarity is computed as the
    dot product:  cos_sim = dot(ref_emb, cmp_emb).

    Layout
    ------
    - Left band  : vertical cluster label (same colour scheme as
                   plot_clusters_full_images).
    - Column 0   : reference image with a coloured border.
    - Columns 1+ : comparison images sorted high→low similarity,
                   title = "<name>  cos=<value>",
                   optional thin similarity colour bar below each image.
    - Blank cells: grey placeholder when cluster has fewer than n_compare
                   comparison candidates.

    Parameters
    ----------
    dataset            : FiftyOne dataset or view
    embeddings_field   : L2-normalised embedding field
    cluster_field      : integer cluster label field
    field_image_name   : display name field; falls back to filepath stem
    n_compare          : number of comparison images per row (default 4)
    size               : "a4" | "a5"
    format             : "png" | "jpg"
    seed               : random seed for reference and comparison sampling
    sort_by_similarity : if True (default), show the n_compare most similar
                         images; if False, sample randomly from the cluster
    ref_border_color   : RGB tuple for the reference image border
    sim_cmap           : matplotlib colormap name for the similarity bar
                         and score text colour (e.g. "RdYlGn", "coolwarm")
    show_sim_bar       : draw a thin coloured bar below each comparison image
                         encoding cosine similarity
    sim_bar_h          : height in px of the similarity bar (default 12)
    output_path        : optional save path
    verbose            : print progress

    Returns
    -------
    PIL.Image.Image — the composed figure
    """

    rng = random.Random(seed)

    # ── Page geometry ─────────────────────────────────────────────────────────
    PAGE_SIZES = {"a4": (2480, 3508), "a5": (1748, 2480)}
    size_key = size.lower()
    if size_key not in PAGE_SIZES:
        raise ValueError(f"size='{size}' not supported. Choose 'a4' or 'a5'.")
    PAGE_W, PAGE_H = PAGE_SIZES[size_key]

    # ── Layout constants ──────────────────────────────────────────────────────
    LABEL_BAND_W  = 90
    H_PAD         = 20
    V_PAD         = 32
    TITLE_H       = 48
    TOP_MARGIN    = 60
    BOTTOM_MARGIN = 60
    INNER_PAD     = 16
    REF_BORDER_W  = 6
    n_cols        = n_compare + 1          # reference + comparisons

    # thumbnail width from available horizontal space
    usable_w = PAGE_W - LABEL_BAND_W - (n_cols + 1) * H_PAD
    THUMB_W  = max(usable_w // n_cols, 80)
    THUMB_H  = int(THUMB_W * 0.75)
    SIM_BAR  = sim_bar_h if show_sim_bar else 0
    ROW_H    = THUMB_H + TITLE_H + SIM_BAR + V_PAD

    if verbose:
        print(f"  Page : {PAGE_W}×{PAGE_H} px  ({size.upper()})")
        print(f"  Thumb: {THUMB_W}×{THUMB_H} px  |  cols: {n_cols}")

    # ── Font loading ──────────────────────────────────────────────────────────
    def _font(pt, bold=False):
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
        for p in candidates:
            try:
                return ImageFont.truetype(p, pt)
            except (IOError, OSError):
                continue
        return ImageFont.load_default()

    FONT_TITLE = _font(20)
    FONT_SIM   = _font(22, bold=True)
    FONT_LABEL = _font(28, bold=True)
    FONT_REF   = _font(20, bold=True)

    # ── Similarity colourmap ──────────────────────────────────────────────────
    sim_cm   = plt.get_cmap(sim_cmap)
    sim_norm = mcolors.Normalize(vmin=0.5, vmax=1.0)

    def sim_to_rgb(val: float) -> tuple:
        """Map cosine similarity in [0,1] → RGB tuple (0-255)."""
        rgba = sim_cm(sim_norm(float(val)))
        return tuple(int(c * 255) for c in rgba[:3])

    # ── Cluster band colours ──────────────────────────────────────────────────
    BAND_COLOURS = [
        (83,  74,  183),
        (29,  158, 117),
        (216, 90,  48),
        (212, 83,  126),
        (55,  95,  165),
        (99,  153, 34),
        (186, 117, 23),
        (136, 135, 128),
    ]
    def band_color(label: int) -> tuple:
        return BAND_COLOURS[label % len(BAND_COLOURS)]

    # ── Load data from FiftyOne ───────────────────────────────────────────────
    if verbose:
        print(f"  Loading samples ...")

    clusters: dict[int, list[dict]] = {}

    for sample in dataset.iter_samples(progress=verbose):
        label = sample.get_field(cluster_field)
        if label is None:
            continue
        label = int(label)

        emb = sample.get_field(embeddings_field)
        if emb is None:
            continue

        name = sample.get_field(field_image_name)
        if not name:
            name = Path(sample.filepath).stem.split('-')[0]

        clusters.setdefault(label, []).append({
            "filepath": sample.filepath,
            "name":     str(name),
            "emb":      np.array(emb, dtype=np.float32),
        })

    if not clusters:
        raise ValueError(
            f"No samples found with both '{cluster_field}' and "
            f"'{embeddings_field}' set."
        )

    sorted_labels = sorted(clusters.keys())
    n_rows        = len(sorted_labels)

    if verbose:
        print(f"  {n_rows} clusters  |  "
              f"{sum(len(v) for v in clusters.values())} samples total")

    # ── Canvas ────────────────────────────────────────────────────────────────
    CANVAS_H = max(
        TOP_MARGIN + n_rows * ROW_H + BOTTOM_MARGIN,
        PAGE_H,
    )
    canvas = Image.new("RGB", (PAGE_W, CANVAS_H), color=(255, 255, 255))
    draw   = ImageDraw.Draw(canvas)

    # ── Helper: load and centre a thumbnail ───────────────────────────────────
    def _load_thumb(filepath: str) -> Image.Image:
        img = Image.open(filepath).convert("RGB")
        img.thumbnail((THUMB_W, THUMB_H), Image.LANCZOS)
        bg = Image.new("RGB", (THUMB_W, THUMB_H), (220, 220, 220))
        ox = (THUMB_W - img.width)  // 2
        oy = (THUMB_H - img.height) // 2
        bg.paste(img, (ox, oy))
        return bg

    def _blank_thumb() -> Image.Image:
        bg = Image.new("RGB", (THUMB_W, THUMB_H), (235, 234, 231))
        d  = ImageDraw.Draw(bg)
        d.rectangle([0, 0, THUMB_W - 1, THUMB_H - 1],
                    outline=(200, 200, 200), width=1)
        return bg

    # ── Draw helper: vertical label band ─────────────────────────────────────
    def _draw_label_band(row_y: int, label: int, band_h: int):
        bc = band_color(label)
        draw.rectangle([0, row_y, LABEL_BAND_W, row_y + band_h], fill=bc)

        label_text = f"cluster {label}"
        bbox  = FONT_LABEL.getbbox(label_text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        txt_img  = Image.new("RGBA", (tw + 2 * INNER_PAD, th + 2 * INNER_PAD),
                             (0, 0, 0, 0))
        td = ImageDraw.Draw(txt_img)
        td.text((INNER_PAD, INNER_PAD), label_text,
                font=FONT_LABEL, fill=(255, 255, 255, 255))
        rotated = txt_img.rotate(90, expand=True)

        px = (LABEL_BAND_W - rotated.width)  // 2
        py = row_y + (band_h - rotated.height) // 2
        canvas.paste(rotated, (px, py), rotated)

    # ── Draw helper: place one image cell ────────────────────────────────────
    def _draw_cell(thumb: Image.Image,
                   x: int, y: int,
                   title: str,
                   border_color: tuple = None,
                   border_w: int = 2,
                   sim_val: float = None):

        # border (reference gets thick coloured border)
        if border_color:
            draw.rectangle(
                [x - border_w, y - border_w,
                 x + THUMB_W + border_w - 1,
                 y + THUMB_H + border_w - 1],
                outline=border_color, width=border_w,
            )
        canvas.paste(thumb, (x, y))

        # thin neutral border for comparisons
        if not border_color:
            draw.rectangle(
                [x, y, x + THUMB_W - 1, y + THUMB_H - 1],
                outline=(180, 180, 180), width=1,
            )

        # similarity bar
        bar_y = y + THUMB_H
        if show_sim_bar and sim_val is not None:
            bar_color = sim_to_rgb(sim_val)
            # filled proportion encodes similarity within [0.5, 1.0]
            fill_w = int(THUMB_W * sim_norm(sim_val))
            draw.rectangle(
                [x, bar_y, x + fill_w, bar_y + SIM_BAR - 1],
                fill=bar_color,
            )
            draw.rectangle(
                [x + fill_w, bar_y, x + THUMB_W - 1, bar_y + SIM_BAR - 1],
                fill=(230, 229, 226),
            )
        elif show_sim_bar:
            draw.rectangle(
                [x, bar_y, x + THUMB_W - 1, bar_y + SIM_BAR - 1],
                fill=(230, 229, 226),
            )

        # title text
        title_y = y + THUMB_H + SIM_BAR + 5
        max_chars = THUMB_W // 10
        if len(title) > max_chars:
            title = title[:max_chars - 1] + "…"

        # choose font and colour based on whether this is a sim label
        if sim_val is not None:
            txt_color = sim_to_rgb(sim_val)
            font      = FONT_SIM
        else:
            txt_color = (60, 60, 60)
            font      = FONT_REF

        bbox  = font.getbbox(title)
        tw    = bbox[2] - bbox[0]
        tx    = x + (THUMB_W - tw) // 2
        draw.text((tx, title_y), title, font=font, fill=txt_color)

    # ── Main loop: one row per cluster ────────────────────────────────────────
    for row_idx, label in enumerate(sorted_labels):
        samples   = clusters[label]
        row_y     = TOP_MARGIN + row_idx * ROW_H
        band_h    = ROW_H - V_PAD

        _draw_label_band(row_y, label, band_h)

        if len(samples) < 2:
            # only one image — show it as reference, all compare slots blank
            ref  = samples[0]
            rest = []
        else:
            # pick reference at random
            ref_idx = rng.randrange(len(samples))
            ref     = samples[ref_idx]
            rest    = [s for i, s in enumerate(samples) if i != ref_idx]

        ref_emb = ref["emb"] / (np.linalg.norm(ref["emb"]) + 1e-12)

        # compute cosine similarities for all candidates
        for s in rest:
            e = s["emb"]
            e = e / (np.linalg.norm(e) + 1e-12)
            s["cos_sim"] = float(np.dot(ref_emb, e))

        if sort_by_similarity:
            rest = sorted(rest, key=lambda s: s["cos_sim"], reverse=True)

        # take at most n_compare
        comparisons = rest[:n_compare]

        if verbose:
            n_avail = len(rest)
            print(f"  cluster {label:>3}: "
                  f"total={len(samples):>4}  "
                  f"comparisons shown={len(comparisons)}/{n_compare}")

        # ── Reference cell ────────────────────────────────────────────────────
        ref_x = LABEL_BAND_W + H_PAD
        ref_y = row_y + V_PAD // 2

        try:
            ref_thumb = _load_thumb(ref["filepath"])
        except Exception as exc:
            ref_thumb = _blank_thumb()
            if verbose:
                print(f"    WARN ref image failed: {exc}")

        _draw_cell(
            thumb        = ref_thumb,
            x            = ref_x,
            y            = ref_y,
            title        = ref["name"],
            border_color = ref_border_color,
            border_w     = REF_BORDER_W,
            sim_val      = None,
        )

        # small "ref" tag above reference image
        tag_text = "reference"
        tb = FONT_TITLE.getbbox(tag_text)
        draw.rectangle(
            [ref_x, ref_y - 26,
             ref_x + tb[2] - tb[0] + 12, ref_y - 4],
            fill=ref_border_color,
        )
        draw.text(
            (ref_x + 6, ref_y - 24),
            tag_text,
            font=FONT_TITLE,
            fill=(255, 255, 255),
        )

        # ── Comparison cells ──────────────────────────────────────────────────
        for col_idx in range(n_compare):
            cx = ref_x + (col_idx + 1) * (THUMB_W + H_PAD)
            cy = ref_y

            if col_idx >= len(comparisons):
                # blank placeholder
                canvas.paste(_blank_thumb(), (cx, cy))
                continue

            cmp = comparisons[col_idx]
            cos = cmp["cos_sim"]

            try:
                cmp_thumb = _load_thumb(cmp["filepath"])
            except Exception as exc:
                cmp_thumb = _blank_thumb()
                if verbose:
                    print(f"    WARN cmp image failed: {exc}")

            ##title_str = f"{cmp['name']} \n cos={cos:.3f}"
            title_str = f"  cos={cos:.3f}"

            _draw_cell(
                thumb        = cmp_thumb,
                x            = cx,
                y            = cy,
                title        = title_str,
                border_color = None,
                sim_val      = cos,
            )

    # ── Save ──────────────────────────────────────────────────────────────────
    if output_path:
        fmt   = format.upper().replace("JPG", "JPEG")
        extra = {"dpi": (300, 300)}
        if fmt == "JPEG":
            extra["quality"] = 92
        canvas.save(output_path, format=fmt, **extra)
        if verbose:
            print(f"  Saved → {output_path}")

    return canvas