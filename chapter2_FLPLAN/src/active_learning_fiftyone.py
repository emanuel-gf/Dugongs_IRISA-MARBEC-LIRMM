"""
TODO: NEEDS TO REWRITE THIS INFORMATION
active_learning_fiftyone.py
===========================
Two FiftyOne-native active learning functions for Jupyter notebooks.

    compute_uniqueness_field(dataset, embeddings_field, uniqueness_field, ...)
        Computes weighted kNN uniqueness scores and stores them on each sample.

    compute_clustering_representativeness(dataset, embeddings_field,
                                          representativeness_field, ...)
        Runs KMeans on the embeddings, assigns each sample its distance-to-centroid
        score (inverted so higher = more representative), and stores cluster label
        and representativeness score on each sample.

Both functions:
  - Read embeddings directly from a FiftyOne field
  - Write results back to the dataset as new fields
  - Return the numpy arrays for further use (e.g. ACLR pipeline)
  - Are fully compatible with the existing active_learning_pipeline()
  - Accept a verbose=True/False argument — no loguru dependency needed


"""

import math
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import math
import warnings
import numpy as np
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.preprocessing import normalize



# Tiny print helper 

def _log(msg: str, verbose: bool, level: str = "info"):
    if not verbose:
        return
    prefix = {"info": "  ", "success": "bum! ", "warn": "DANGER"}.get(level, "  ")
    print(f"{prefix}{msg}")


# Helpers

def _load_embeddings(dataset, embeddings_field: str, verbose: bool):
    """
    Load embeddings from a FiftyOne dataset field.
    Returns (embeddings_norm, sample_ids) — embeddings are L2-normalised.
    Skips samples where the field is None.
    """
    _log(f"Loading embeddings from field '{embeddings_field}' ...", verbose)

    sample_ids     = []
    raw_embeddings = []

    for sample in dataset.iter_samples(progress=verbose):
        emb = sample.get_field(embeddings_field)
        if emb is None:
            continue
        raw_embeddings.append(np.array(emb, dtype=np.float32))
        sample_ids.append(sample.id)

    if not raw_embeddings:
        raise ValueError(
            f"No embeddings found in field '{embeddings_field}'. "
            "Run compute_embeddings() first."
        )

    arr      = np.stack(raw_embeddings, axis=0)   # (N, D)
    arr_norm = normalize(arr, norm="l2")           # L2-normalise
    _log(f"Loaded {len(arr_norm)} embeddings, dim={arr_norm.shape[1]}", verbose)
    return arr_norm, sample_ids


def _write_field(dataset, sample_ids: list, values, field_name: str, verbose: bool):
    """Bulk-write scalar values back to a FiftyOne dataset field."""
    _log(f"Writing field '{field_name}' to dataset ...", verbose)
    id_to_value = dict(zip(sample_ids, values))

    for sample in dataset.iter_samples(autosave=True, progress=verbose):
        val = id_to_value.get(sample.id)
        if val is not None:
            sample[field_name] = float(val) if np.isscalar(val) else val


# ── Decay weight factories ────────────────────────────────────────────────────
 
def compute_decay_weights(k: int, decay: str = "exponential",
                          decay_param: float = 0.5) -> np.ndarray:
    """
    Compute a weight vector of length k for neighbour ranks 1..k.
 
    Parameters
    ----------
    k           : number of neighbours
    decay       : "exponential" | "linear" | "power"
    decay_param : lambda for exponential (higher = faster decay)
                  p for power (higher = faster decay)
                  ignored for linear
 
    Returns
    -------
    weights : np.ndarray shape (k,), NOT normalised to sum-to-1.
              Normalisation is done inside the uniqueness function so that
              the absolute scale of weights does not change scores.
    """
    ranks = np.arange(1, k + 1, dtype=np.float64)
 
    if decay == "exponential":
        # w(i) = exp(-lambda * i)
        # lambda=0   -> all weights = 1 (uniform)
        # lambda=0.5 -> moderate decay
        # lambda=1.0 -> aggressive decay, only nearest neighbour matters
        weights = np.exp(-decay_param * ranks)
 
    elif decay == "linear":
        # w(i) = (k + 1 - i) / k
        # nearest neighbour gets weight 1.0, furthest gets 1/k
        weights = (k + 1 - ranks) / k
 
    elif decay == "power":
        # w(i) = 1 / i^p
        # p=0.5 -> very slow decay (sqrt)
        # p=1   -> harmonic (1, 1/2, 1/3, ...)
        # p=2   -> squared (1, 1/4, 1/9, ...)
        weights = 1.0 / (ranks ** decay_param)
 
    else:
        raise ValueError(
            f"Unknown decay '{decay}'. Choose from: 'exponential', 'linear', 'power'."
        )
 
    return weights
 

# Weighted kNN Uniqueness 
def compute_uniqueness_field(
    dataset,
    embeddings_field:  str   = "full_embeddings",
    uniqueness_field:  str   = "uniqueness_score",
    k:                 int   = 10,
    decay:             str   = "exponential",
    decay_param:       float = 0.5,
    custom_weights:    list  = None,
    save:              bool  = True,
    verbose:           bool  = True,
) -> tuple[np.ndarray, list]:
    """
    Compute weighted kNN uniqueness scores and optionally store in the dataset.
 
    Uniqueness = weighted mean of cosine distances to the k nearest neighbours.
    Samples geometrically isolated in embedding space score high (→ 1.0).
    Scores are normalised to [0, 1].
 
    Parameters
    ----------
    dataset          : FiftyOne dataset or view
    embeddings_field : field containing pre-computed embeddings
    uniqueness_field : field name to write the scores to
    k                : number of neighbours.
                       Rule of thumb: keep k < sqrt(N) for meaningful local
                       structure (e.g. N=2755 → k < 52, recommended k=10-20).
    decay            : weight decay family — "exponential", "linear", "power"
    decay_param      : lambda for exponential, p for power, unused for linear
    custom_weights   : if given, overrides decay and uses these weights directly.
                       len(custom_weights) must equal k.
    save             : if True, writes scores back to the dataset
    verbose          : if True, prints progress and stats
 
    Returns
    -------
    uniqueness_scores : np.ndarray shape (N,) — normalised [0,1]
    sample_ids        : list of N sample IDs (same order as scores)
    """
    embeddings_norm, sample_ids = _load_embeddings(dataset, embeddings_field, verbose)
    N = len(embeddings_norm)
 
    # Warn if k >= sqrt(N)
    k_max_recommended = int(math.sqrt(N))
    if k >= k_max_recommended:
        _log(
            f"k={k} >= sqrt(N)={k_max_recommended:.0f}. "
            f"Consider reducing k to avoid neighbourhood overlap.",
            verbose, level="warn",
        )
 
    # Build weights
    if custom_weights is not None:
        if len(custom_weights) != k:
            raise ValueError(
                f"len(custom_weights)={len(custom_weights)} must equal k={k}."
            )
        weights = np.array(custom_weights, dtype=np.float64)
        _log(f"Using custom weights: {weights.tolist()}", verbose)
    else:
        weights = compute_decay_weights(k, decay, decay_param)
        _log(
            f"Decay: {decay}  param={decay_param}  k={k}  "
            f"weights=[{', '.join(f'{w:.3f}' for w in weights)}]",
            verbose,
        )
 
    # Normalise weights to sum-to-1 so the weighted mean is scale-invariant
    weights = weights / weights.sum()
 
    _log(f"Fitting kNN (k={k}, metric=cosine, N={N}) ...", verbose)
    knn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=-1)
    knn.fit(embeddings_norm)
    distances, _ = knn.kneighbors(embeddings_norm)
 
    # Exclude self (rank 0), apply weights
    relevant_dists = distances[:, 1:]                              # (N, k)
    weighted_dists = (relevant_dists * weights).sum(axis=1)        # (N,)
 
    # Normalise to [0, 1]
    max_val = weighted_dists.max()
    if max_val > 0:
        weighted_dists /= max_val
 
    _log(
        f"Uniqueness stats: "
        f"min={weighted_dists.min():.4f}  "
        f"mean={weighted_dists.mean():.4f}  "
        f"max={weighted_dists.max():.4f}",
        verbose,
    )
    _log(
        f"Above 0.5: {(weighted_dists > 0.5).sum()}  |  "
        f"above 0.9: {(weighted_dists > 0.9).sum()}  "
        f"(out of {N})",
        verbose,
    )
 
    if save:
        _write_field(dataset, sample_ids, weighted_dists, uniqueness_field, verbose)
        _log(f"Scores saved to '{uniqueness_field}'.", verbose, level="success")
 
    return weighted_dists, sample_ids

## Version 2 with uniqueness within  a given cluster
def compute_uniqueness_field_v2(
    dataset,
    embeddings_field:  str   = "full_embeddings",
    uniqueness_field:  str   = "uniqueness_score",
    cluster_field:     str   = None,   # if given, compute uniqueness per cluster
    k:                 int   = 10,
    decay:             str   = "exponential",
    decay_param:       float = 0.5,
    custom_weights:    list  = None,
    save:              bool  = True,
    verbose:           bool  = True,
) -> tuple[np.ndarray, list]:
    """
    Compute weighted kNN uniqueness scores and optionally store in the dataset.

    If cluster_field is provided, uniqueness is computed independently within
    each cluster — kNN neighbours are restricted to samples in the same cluster.
    This measures "how unique is this sample among its visual peers" rather than
    global isolation, which avoids large dense clusters suppressing the scores
    of smaller clusters globally.

    If cluster_field is None, standard global uniqueness is computed.

    Parameters
    ----------
    dataset          : FiftyOne dataset or view
    embeddings_field : field containing pre-computed embeddings
    uniqueness_field : field name to write the scores to
    cluster_field    : optional field containing integer cluster labels
                       (e.g. "cluster_label" from compute_clustering).
                       If given, kNN is run per-cluster.
    k                : number of neighbours within the cluster (or globally).
    decay            : "exponential", "linear", "power"
    decay_param      : lambda for exponential, p for power, unused for linear
    custom_weights   : optional manual weight list of length k
    save             : write scores back to dataset
    verbose          : print progress and stats

    Returns
    -------
    uniqueness_scores : np.ndarray (N,) — normalised [0,1]
    sample_ids        : list of N sample IDs (same order)
    """
    embeddings_norm, sample_ids = _load_embeddings(dataset, embeddings_field, verbose)
    N = len(embeddings_norm)

    # ── Build weights ─────────────────────────────────────────────────────────
    if custom_weights is not None:
        if len(custom_weights) != k:
            raise ValueError(
                f"len(custom_weights)={len(custom_weights)} must equal k={k}."
            )
        weights = np.array(custom_weights, dtype=np.float64)
        _log(f"Using custom weights: {weights.tolist()}", verbose)
    else:
        weights = compute_decay_weights(k, decay, decay_param)
        _log(
            f"Decay: {decay}  param={decay_param}  k={k}  "
            f"weights=[{', '.join(f'{w:.3f}' for w in weights)}]",
            verbose,
        )
    weights = weights / weights.sum()

    weighted_dists = np.zeros(N, dtype=np.float64)

    # ── Per-cluster or global ─────────────────────────────────────────────────
    if cluster_field is not None:
        # Load cluster labels aligned to sample_ids
        _log(f"Loading cluster labels from '{cluster_field}' ...", verbose)
        id_to_idx     = {sid: i for i, sid in enumerate(sample_ids)}
        cluster_labels = np.full(N, -1, dtype=int)

        for sample in dataset.iter_samples(progress=verbose):
            idx = id_to_idx.get(sample.id)
            if idx is None:
                continue
            label = sample.get_field(cluster_field)
            if label is not None:
                cluster_labels[idx] = int(label)

        unique_clusters = np.unique(cluster_labels[cluster_labels >= 0])
        _log(f"Found {len(unique_clusters)} clusters.", verbose)

        for cluster_id in unique_clusters:
            cluster_idx = np.where(cluster_labels == cluster_id)[0]
            n_in_cluster = len(cluster_idx)

            if n_in_cluster < 2:
                # Only one sample — uniqueness is undefined, set to 1.0
                # (it is maximally unique within its cluster by definition)
                weighted_dists[cluster_idx] = 1.0
                continue

            # Clamp k to cluster size — can't have more neighbours than members
            k_eff = min(k, n_in_cluster - 1)

            k_max_recommended = int(math.sqrt(n_in_cluster))
            if k_eff >= k_max_recommended and verbose:
                _log(
                    f"  cluster {cluster_id}: k_eff={k_eff} >= "
                    f"sqrt({n_in_cluster})={k_max_recommended}. "
                    f"Consider reducing k.",
                    verbose, level="warn",
                )

            # Recompute weights for this k_eff (in case k was clamped)
            if k_eff < k:
                w_local = compute_decay_weights(k_eff, decay, decay_param)
                w_local = w_local / w_local.sum()
            else:
                w_local = weights

            sub_embs = embeddings_norm[cluster_idx]   # (n_in_cluster, D)

            knn = NearestNeighbors(
                n_neighbors=k_eff + 1, metric="cosine", n_jobs=-1
            )
            knn.fit(sub_embs)
            distances, _ = knn.kneighbors(sub_embs)

            relevant_dists = distances[:, 1:]          # (n_in_cluster, k_eff)
            scores         = (relevant_dists * w_local).sum(axis=1)

            # Normalise within cluster to [0,1]
            max_val = scores.max()
            if max_val > 0:
                scores /= max_val

            weighted_dists[cluster_idx] = scores

            _log(
                f"  cluster {cluster_id:>3}: n={n_in_cluster:>4}  "
                f"k_eff={k_eff}  "
                f"mean={scores.mean():.4f}  max={scores.max():.4f}",
                verbose,
            )

    else:
        # ── Global uniqueness (original behaviour) ────────────────────────
        k_max_recommended = int(math.sqrt(N))
        if k >= k_max_recommended:
            _log(
                f"k={k} >= sqrt(N)={k_max_recommended}. "
                f"Consider reducing k.",
                verbose, level="warn",
            )

        _log(f"Fitting global kNN (k={k}, metric=cosine, N={N}) ...", verbose)
        knn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=-1)
        knn.fit(embeddings_norm)
        distances, _ = knn.kneighbors(embeddings_norm)

        relevant_dists = distances[:, 1:]
        weighted_dists = (relevant_dists * weights).sum(axis=1)

        max_val = weighted_dists.max()
        if max_val > 0:
            weighted_dists /= max_val

    # ── Stats and save ────────────────────────────────────────────────────────
    _log(
        f"Uniqueness stats: "
        f"min={weighted_dists.min():.4f}  "
        f"mean={weighted_dists.mean():.4f}  "
        f"max={weighted_dists.max():.4f}",
        verbose,
    )
    _log(
        f"Above 0.5: {(weighted_dists > 0.5).sum()}  |  "
        f"above 0.9: {(weighted_dists > 0.9).sum()}  "
        f"(out of {N})",
        verbose,
    )

    if save:
        _write_field(dataset, sample_ids, weighted_dists, uniqueness_field, verbose)
        _log(f"Scores saved to '{uniqueness_field}'.", verbose, level="success")

    return weighted_dists, sample_ids

    
#  compute_clustering
def compute_clustering(
    dataset,
    embeddings_field:  str   = "full_embeddings",
    cluster_field:     str   = "cluster_label",
    method:            str   = "kmeans",
    n_clusters:        int   = 13,
    n_init:            int   = 10,
    bandwidth:         float = None,
    bandwidth_quantile: float = 0.2,
    seed:              int   = 42,
    save:              bool  = True,
    verbose:           bool  = True,
    **method_kwargs,
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Cluster embeddings using KMeans or MeanShift and store cluster labels
    as a FiftyOne field.

    Parameters
    ----------
    dataset            : FiftyOne dataset or view
    embeddings_field   : field containing pre-computed L2-normalised embeddings
    cluster_field      : field name to write integer cluster labels to.
                         Default: "cluster_label".
                         Tip: use descriptive names when storing multiple
                         clusterings, e.g. "cluster_kmeans_13" or
                         "cluster_meanshift".
    method             : "kmeans" (default) | "meanshift"
    n_clusters         : number of clusters — KMeans only
    n_init             : number of KMeans initialisations (KMeans only)
    bandwidth          : MeanShift bandwidth. If None, estimated automatically
                         via sklearn estimate_bandwidth using bandwidth_quantile.
    bandwidth_quantile : quantile passed to estimate_bandwidth when bandwidth
                         is not provided (MeanShift only). Lower values → smaller
                         bandwidth → more clusters. Range (0, 1), default 0.2.
    seed               : random seed for reproducibility (KMeans only;
                         MeanShift is deterministic given bandwidth)
    save               : write cluster labels back to dataset
    verbose            : print progress and stats
    **method_kwargs    : extra kwargs forwarded to the sklearn estimator.
                         n_clusters / n_init / random_state (KMeans) and
                         bandwidth (MeanShift) are set explicitly and will
                         override duplicates in method_kwargs.

    Returns
    -------
    cluster_labels  : np.ndarray (N,)   — integer cluster assignments
    embeddings_norm : np.ndarray (N, D) — L2-normalised embeddings
                      (returned so callers can reuse without re-loading)
    sample_ids      : list of N FiftyOne sample ID strings (same order)

    Notes
    -----
    MeanShift does not take n_clusters — the number of clusters is inferred
    from the bandwidth. The function logs the discovered cluster count and
    warns if:
      - only 1 cluster is found (bandwidth likely too large)
      - more than 50 clusters are found (bandwidth likely too small)
    
    This function is called internally by compute_clustering_representativeness
    so that clustering logic is not duplicated.
    """
    if method not in ("kmeans", "meanshift"):
        raise ValueError(
            f"method='{method}' not supported. Choose 'kmeans' or 'meanshift'."
        )

    embeddings_norm, sample_ids = _load_embeddings(dataset, embeddings_field, verbose)
    N = len(embeddings_norm)

    # ── KMeans ────────────────────────────────────────────────────────────────
    if method == "kmeans":
        for key in ("n_clusters", "n_init", "random_state"):
            method_kwargs.pop(key, None)

        _log(
            f"KMeans: n_clusters={n_clusters}  n_init={n_init}  seed={seed}"
            + (f"  extra={method_kwargs}" if method_kwargs else ""),
            verbose,
        )

        km = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=n_init,
            random_state=seed,
            **method_kwargs,
        )
        cluster_labels = km.fit_predict(embeddings_norm)
        cluster_sizes  = np.bincount(cluster_labels, minlength=n_clusters)

        _log(
            f"KMeans done: "
            f"min_size={cluster_sizes.min()}  "
            f"mean_size={cluster_sizes.mean():.1f}  "
            f"max_size={cluster_sizes.max()}",
            verbose,
        )
        _log(
            f"Empty clusters: {(cluster_sizes == 0).sum()} / {n_clusters}",
            verbose,
        )

    # ── MeanShift ─────────────────────────────────────────────────────────────
    else:
        method_kwargs.pop("bandwidth", None)

        if bandwidth is None:
            _log(
                f"MeanShift: estimating bandwidth "
                f"(quantile={bandwidth_quantile}, N={N}) ...",
                verbose,
            )
            try:
                bandwidth = estimate_bandwidth(
                    embeddings_norm,
                    quantile=bandwidth_quantile,
                    random_state=seed,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"estimate_bandwidth failed: {exc}. "
                    "Try passing bandwidth explicitly."
                ) from exc

            if bandwidth <= 0:
                raise ValueError(
                    f"Estimated bandwidth={bandwidth:.6f} is non-positive. "
                    "Increase bandwidth_quantile or pass bandwidth explicitly."
                )
            _log(f"Estimated bandwidth={bandwidth:.6f}", verbose)
        else:
            _log(f"MeanShift: using provided bandwidth={bandwidth:.6f}", verbose)

        ms = MeanShift(bandwidth=bandwidth, **method_kwargs)

        _log("Fitting MeanShift ...", verbose)
        cluster_labels     = ms.fit_predict(embeddings_norm)
        n_found            = len(np.unique(cluster_labels))
        cluster_sizes      = np.bincount(cluster_labels)

        _log(
            f"MeanShift done: n_clusters={n_found}  "
            f"min_size={cluster_sizes.min()}  "
            f"mean_size={cluster_sizes.mean():.1f}  "
            f"max_size={cluster_sizes.max()}",
            verbose,
        )

        if n_found == 1:
            warnings.warn(
                f"MeanShift found only 1 cluster (bandwidth={bandwidth:.6f}). "
                "The bandwidth is likely too large — try a smaller value or "
                "a lower bandwidth_quantile.",
                UserWarning, stacklevel=2,
            )
        elif n_found > 50:
            warnings.warn(
                f"MeanShift found {n_found} clusters (bandwidth={bandwidth:.6f}). "
                "The bandwidth may be too small — try a larger value or "
                "a higher bandwidth_quantile.",
                UserWarning, stacklevel=2,
            )

    # ── Save ──────────────────────────────────────────────────────────────────
    if save:
        _write_field(
            dataset, sample_ids,
            [int(c) for c in cluster_labels],
            cluster_field, verbose,
        )
        _log(f"Cluster labels saved to '{cluster_field}'.", verbose, level="success")

    return cluster_labels, embeddings_norm, sample_ids


#
#  compute_clustering_representativeness
#   (now delegates to compute_clustering)
def compute_clustering_representativeness(
    dataset,
    embeddings_field:       str   = "full_embeddings",
    cluster_field:          str   = "cluster_label",
    representativeness_field: str = "representativeness_score",
    method:                 str   = "kmeans",
    n_clusters:             int   = 13,
    n_init:                 int   = 10,
    bandwidth:              float = None,
    bandwidth_quantile:     float = 0.2,
    seed:                   int   = 42,
    save:                   bool  = True,
    verbose:                bool  = True,
    **method_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Cluster embeddings and score each sample by proximity to its cluster
    centroid (representativeness). Higher score = closer to centroid = more
    representative of its cluster's visual mode.

    Clustering is delegated entirely to compute_clustering — all clustering
    parameters are forwarded. See compute_clustering for full parameter docs.

    Parameters
    ----------
    dataset                  : FiftyOne dataset or view
    embeddings_field         : field containing pre-computed embeddings
    cluster_field            : field to write cluster labels to
    representativeness_field : field to write per-sample representativeness
                               scores to. Scores are normalised to [0, 1]
                               within each cluster (1 = closest to centroid).
    method                   : "kmeans" | "meanshift" — forwarded to
                               compute_clustering
    n_clusters               : KMeans only — forwarded to compute_clustering
    n_init                   : KMeans only — forwarded to compute_clustering
    bandwidth                : MeanShift only — forwarded to compute_clustering
    bandwidth_quantile       : MeanShift only — forwarded to compute_clustering
    seed                     : forwarded to compute_clustering
    save                     : write both cluster labels and representativeness
                               scores back to dataset
    verbose                  : print progress and stats
    **method_kwargs          : forwarded to compute_clustering

    Returns
    -------
    representativeness_scores : np.ndarray (N,) normalised [0,1]
    cluster_labels            : np.ndarray (N,)
    embeddings_norm           : np.ndarray (N, D)
    sample_ids                : list of N sample ID strings
    """
    # ── Step 1: cluster via compute_clustering ────────────────────────────────
    cluster_labels, embeddings_norm, sample_ids = compute_clustering(
        dataset          = dataset,
        embeddings_field = embeddings_field,
        cluster_field    = cluster_field,
        method           = method,
        n_clusters       = n_clusters,
        n_init           = n_init,
        bandwidth        = bandwidth,
        bandwidth_quantile = bandwidth_quantile,
        seed             = seed,
        save             = save,
        verbose          = verbose,
        **method_kwargs,
    )

    N               = len(embeddings_norm)
    unique_clusters = np.unique(cluster_labels)
    rep_scores      = np.zeros(N, dtype=np.float64)

    # ── Step 2: per-cluster centroid distance → representativeness ────────────
    _log("Computing representativeness scores ...", verbose)

    for cid in unique_clusters:
        idx      = np.where(cluster_labels == cid)[0]
        sub_embs = embeddings_norm[idx]

        centroid = sub_embs.mean(axis=0)
        centroid /= (np.linalg.norm(centroid) + 1e-12)

        dists = np.linalg.norm(sub_embs - centroid, axis=1)

        max_d = dists.max()
        if max_d > 0:
            scores = 1.0 - dists / max_d
        else:
            scores = np.ones(len(idx))

        rep_scores[idx] = scores

        _log(
            f"  cluster {cid:>3}: n={len(idx):>4}  "
            f"mean_rep={scores.mean():.4f}  "
            f"min_rep={scores.min():.4f}",
            verbose,
        )

    _log(
        f"Representativeness stats: "
        f"min={rep_scores.min():.4f}  "
        f"mean={rep_scores.mean():.4f}  "
        f"max={rep_scores.max():.4f}",
        verbose,
    )

    if save:
        _write_field(
            dataset, sample_ids, rep_scores,
            representativeness_field, verbose,
        )
        _log(
            f"Scores saved to '{representativeness_field}'.",
            verbose, level="success",
        )

    return rep_scores, cluster_labels, embeddings_norm, sample_ids



#  _soft_penalty_propagation 
def _soft_penalty_propagation(
    embeddings_norm:   np.ndarray,
    uniqueness_scores: np.ndarray,
    ball_radius:       float,
    penalty:           float,
) -> np.ndarray:
    """
    Propagate ball-radius penalties across all embeddings without hard
    exclusion — every point stays in contention throughout.

    Visits points in descending uniqueness order. For each point, all
    neighbours within ball_radius (L2 on unit vectors) have their working
    score multiplied by penalty. A point inside k selected balls accumulates
    penalty^k, encoding how "explained" it is by its high-uniqueness
    neighbours geometrically.

    Note on ball_radius units
    -------------------------
    cKDTree operates in L2 distance on L2-normalised embeddings. For unit
    vectors the relationship to cosine similarity is:
        L2² = 2 * (1 - cosine_similarity)
    So ball_radius=0.5 corresponds to cosine_similarity >= 0.875.

    Parameters
    ----------
    embeddings_norm   : (N, D) L2-normalised embeddings
    uniqueness_scores : (N,)   pre-computed uniqueness scores in [0, 1]
    ball_radius       : L2 radius for neighbourhood penalisation
    penalty           : score multiplier per ball hit, in (0, 1).
                        Lower = more aggressive suppression of near-duplicates.

    Returns
    -------
    working_scores : np.ndarray (N,) raw degraded scores before normalisation.
                     Values are in [0, max(uniqueness_scores)].
                     Normalisation to [0,1] is done by the calling wrapper.
    """
    N       = len(embeddings_norm)
    tree    = cKDTree(embeddings_norm)
    working = uniqueness_scores.copy().astype(np.float64)
    order   = np.argsort(working)[::-1]

    for idx in order:
        neighbours = tree.query_ball_point(
            embeddings_norm[idx], ball_radius, return_sorted=False
        )
        for nb in neighbours:
            if nb != idx:
                working[nb] *= penalty

    return working


#
#  compute_soft_coverage_scores
#   (public FiftyOne wrapper)

def compute_soft_coverage_scores(
    dataset,
    embeddings_field:  str   = "full_embeddings",
    uniqueness_field:  str   = "uniqueness_score",
    cluster_field:     str   = None,
    ball_radius:       float = 0.5,
    penalty:           float = 0.7,
    coverage_field:    str   = "soft_coverage_score",
    save:              bool  = True,
    verbose:           bool  = True,
) -> tuple[np.ndarray, list]:
    """
    Compute soft coverage scores for all embeddings by propagating ball-radius
    penalties without hard exclusion.

    Unlike the greedy selection in _cluster_diverse_selection_from_arrays —
    where penalised points are added to visited and permanently skipped — this
    function keeps every point in contention. A point inside k penalty balls
    accumulates penalty^k, producing a continuous float score for every
    embedding that encodes how "explained" it is by its high-uniqueness
    neighbours.

    The result is a full-cluster coverage map: isolated points retain their
    original uniqueness score; points in dense, well-covered regions are
    suppressed toward zero.

    Scores are normalised to [0, 1]:
      - per cluster, if cluster_field is provided
      - globally otherwise

    Requires pre-computed uniqueness scores
    ----------------------------------------
    This function reads uniqueness scores from uniqueness_field. If that field
    is missing or contains no valid values, a ValueError is raised with
    instructions to run compute_uniqueness_field_v2 first:

        compute_uniqueness_field_v2(
            dataset,
            embeddings_field="{embeddings_field}",
            cluster_field="{cluster_field}",   # if clustering already done
        )

    Parameters
    ----------
    dataset          : FiftyOne dataset or view
    embeddings_field : field containing pre-computed L2-normalised embeddings
    uniqueness_field : field containing pre-computed uniqueness scores [0,1].
                       Must be populated before calling this function.
    cluster_field    : optional field containing integer cluster labels.
                       If given, penalty propagation runs independently per
                       cluster and normalisation is per-cluster, which avoids
                       large dense clusters suppressing isolated smaller ones.
    ball_radius      : L2 radius for neighbourhood penalisation.
                       ball_radius=0.5 → cosine_similarity >= 0.875.
                       ball_radius=0.3 → cosine_similarity >= 0.955.
    penalty          : score multiplier applied per ball hit, in (0, 1).
                       Lower values suppress near-duplicates more aggressively.
                       Recommended range: 0.5–0.8.
    coverage_field   : field name to write the output scores to.
                       Default: "soft_coverage_score".
    save             : write scores back to dataset
    verbose          : print progress and stats

    Returns
    -------
    coverage_scores : np.ndarray (N,) normalised [0,1]
    sample_ids      : list of N FiftyOne sample ID strings (same order)
    """
    embeddings_norm, sample_ids = _load_embeddings(dataset, embeddings_field, verbose)
    N = len(embeddings_norm)
    id_to_idx = {sid: i for i, sid in enumerate(sample_ids)}

    # ── Load uniqueness scores ────────────────────────────────────────────────
    _log(f"Loading uniqueness scores from '{uniqueness_field}' ...", verbose)

    uniqueness_scores = np.full(N, np.nan, dtype=np.float64)
    for sample in dataset.iter_samples(progress=verbose):
        idx = id_to_idx.get(sample.id)
        if idx is None:
            continue
        val = sample.get_field(uniqueness_field)
        if val is not None:
            uniqueness_scores[idx] = float(val)

    n_missing = np.isnan(uniqueness_scores).sum()
    if n_missing == N:
        raise ValueError(
            f"No values found in uniqueness field '{uniqueness_field}'. "
            f"Run compute_uniqueness_field_v2 first:\n\n"
            f"    compute_uniqueness_field_v2(\n"
            f"        dataset,\n"
            f"        embeddings_field='{embeddings_field}',\n"
            f"        cluster_field='{cluster_field}',  "
            f"# omit if no clustering\n"
            f"    )\n"
        )
    if n_missing > 0:
        _log(
            f"{n_missing} samples have no uniqueness score — "
            f"filling with cluster mean (or global mean).",
            verbose, level="warn",
        )

    coverage_scores = np.zeros(N, dtype=np.float64)

    # ── Per-cluster or global propagation ────────────────────────────────────
    if cluster_field is not None:
        _log(f"Loading cluster labels from '{cluster_field}' ...", verbose)
        cluster_labels = np.full(N, -1, dtype=int)

        for sample in dataset.iter_samples(progress=verbose):
            idx = id_to_idx.get(sample.id)
            if idx is None:
                continue
            label = sample.get_field(cluster_field)
            if label is not None:
                cluster_labels[idx] = int(label)

        unique_clusters = np.unique(cluster_labels[cluster_labels >= 0])
        _log(f"Running per-cluster propagation ({len(unique_clusters)} clusters) ...", verbose)

        for cid in unique_clusters:
            cluster_idx = np.where(cluster_labels == cid)[0]
            n_c         = len(cluster_idx)
            sub_embs    = embeddings_norm[cluster_idx]
            sub_uniq    = uniqueness_scores[cluster_idx]

            # Fill any NaN uniqueness within this cluster with cluster mean
            nan_mask = np.isnan(sub_uniq)
            if nan_mask.any():
                sub_uniq[nan_mask] = np.nanmean(sub_uniq)

            raw = _soft_penalty_propagation(
                sub_embs, sub_uniq, ball_radius, penalty
            )

            # Normalise per-cluster to [0, 1]
            max_val = raw.max()
            scores  = raw / max_val if max_val > 0 else raw

            coverage_scores[cluster_idx] = scores

            _log(
                f"  cluster {cid:>3}: n={n_c:>4}  "
                f"mean={scores.mean():.4f}  "
                f"min={scores.min():.4f}  "
                f"max={scores.max():.4f}",
                verbose,
            )

    else:
        _log("Running global propagation ...", verbose)

        # Fill any NaN with global mean
        nan_mask = np.isnan(uniqueness_scores)
        if nan_mask.any():
            uniqueness_scores[nan_mask] = np.nanmean(uniqueness_scores)

        raw = _soft_penalty_propagation(
            embeddings_norm, uniqueness_scores, ball_radius, penalty
        )

        max_val        = raw.max()
        coverage_scores = raw / max_val if max_val > 0 else raw

    # ── Stats and save ────────────────────────────────────────────────────────
    _log(
        f"Coverage stats: "
        f"min={coverage_scores.min():.4f}  "
        f"mean={coverage_scores.mean():.4f}  "
        f"max={coverage_scores.max():.4f}",
        verbose,
    )
    _log(
        f"Above 0.5: {(coverage_scores > 0.5).sum()}  |  "
        f"above 0.9: {(coverage_scores > 0.9).sum()}  "
        f"(out of {N})",
        verbose,
    )

    if save:
        _write_field(dataset, sample_ids, coverage_scores, coverage_field, verbose)
        _log(f"Scores saved to '{coverage_field}'.", verbose, level="success")

    return coverage_scores, sample_ids
