"""
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
    arr_norm = normalize(arr, norm="l2")           # L2-normalise before kNN
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

    
# KMeans Clustering Representativeness
def compute_clustering_representativeness(
    dataset,
    embeddings_field: str   = "full_embeddings",
    cluster_field:    str   = "cluster_label",
    n_clusters:       int   = 13,
    n_init:           int   = 10,
    seed:             int   = 42,
    save:             bool  = True,
    verbose:          bool  = True,
    **kmeans_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Run KMeans clustering on embeddings and store cluster labels per sample. The embeddings are L2 normalized .

    Parameters
    ----------
    dataset          : FiftyOne dataset or view
    embeddings_field : field containing pre-computed embeddings
    cluster_field    : field name to write cluster integer labels
    n_clusters       : number of KMeans clusters (K)
    n_init           : KMeans n_init (number of random initialisations)
    seed             : random seed for reproducibility
    save             : if True, writes cluster labels to dataset
    verbose          : if True, prints progress and stats
    **kmeans_kwargs  : any additional kwargs forwarded directly to sklearn KMeans
                       e.g. max_iter=500, tol=1e-5, algorithm="lloyd"
                       Note: n_clusters, n_init, random_state are set explicitly
                       above and will override any duplicate keys in kmeans_kwargs.

    Returns
    -------
    cluster_labels  : np.ndarray (N,)   — integer cluster assignments
    centroids       : np.ndarray (K, D) — cluster centroid coordinates
    embeddings_norm : np.ndarray (N, D) — L2-normalised embeddings
                      (returned so downstream functions like stochastic
                       pooling can reuse them without re-loading)
    sample_ids      : list of N sample IDs (same order as cluster_labels)
    """
    embeddings_norm, sample_ids = _load_embeddings(dataset, embeddings_field, verbose)

    # Explicit args take precedence over anything passed via kmeans_kwargs
    kmeans_kwargs.pop("n_clusters",    None)
    kmeans_kwargs.pop("n_init",        None)
    kmeans_kwargs.pop("random_state",  None)

    _log(
        f"Running KMeans (k={n_clusters}, n_init={n_init}, seed={seed}"
        + (f", extra={kmeans_kwargs}" if kmeans_kwargs else "")
        + ") ...",
        verbose,
    )

    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=n_init,
        random_state=seed,
        **kmeans_kwargs,
    )
    cluster_labels = kmeans.fit_predict(embeddings_norm)  # (N,)
    centroids      = kmeans.cluster_centers_              # (K, D)

    # ── Cluster stats ─────────────────────────────────────────────────────────
    cluster_sizes = np.bincount(cluster_labels, minlength=n_clusters)
    _log(f"Cluster sizes: min={cluster_sizes.min()}  "
         f"mean={cluster_sizes.mean():.1f}  "
         f"max={cluster_sizes.max()}",
         verbose)
    _log(f"Empty clusters: {(cluster_sizes == 0).sum()} / {n_clusters}", verbose)

    if save:
        _write_field(dataset, sample_ids,
                     [int(c) for c in cluster_labels],
                     cluster_field, verbose)
        _log(f"Cluster labels saved to '{cluster_field}'.",
             verbose, level="success")

    return cluster_labels, centroids, embeddings_norm, sample_ids


