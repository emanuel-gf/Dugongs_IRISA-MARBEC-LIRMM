import numpy as np
import matplotlib.pyplot as plt 


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