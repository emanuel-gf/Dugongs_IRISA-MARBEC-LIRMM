"""
active_learning_patches.py
============================

Patch-aware counterparts to compute_clustering / compute_uniqueness_field_v2.

The sample-level versions write one scalar value per SAMPLE (e.g.
sample["cluster_label"] = 3). That breaks down for object/patch embeddings,
where a single sample (tile) can contain MULTIPLE detections (e.g. two
dugongs in one tile), each needing its OWN independent cluster/uniqueness
value. FiftyOne's patches-view embeddings visualization expects these
values to live ON EACH Detection object, not on the parent sample.

These functions take the same (sample_id, embedding) shape as before, PLUS
a parallel detection_ids array (the join key matching each row to a
specific fo.Detection within that sample's patches_field), and write
results back as an attribute on each individual Detection.

Expects the same npz shape produced by the object-embedding extraction
pipeline:
    np.load("all_objects_embeddings.npz")
        -> object_embeddings  (N, dim)
        -> detection_ids      (N,)  -- fo.Detection.id strings
        -> sample_ids         (N,)  -- parent fo.Sample.id strings

Usage
-----
    from active_learning_patches import compute_clustering_patches, compute_uniqueness_patches

    data = np.load("all_objects_embeddings.npz")

    cluster_labels, emb_norm, sample_ids, detection_ids = compute_clustering_patches(
        dataset=dataset,
        embeddings=data["object_embeddings"],
        sample_ids=data["sample_ids"],
        detection_ids=data["detection_ids"],
        patches_field="ground_truth",
        cluster_field="cluster_label",
        n_clusters=15,
    )

    uniqueness_scores, _ = compute_uniqueness_patches(
        dataset=dataset,
        embeddings=data["object_embeddings"],
        sample_ids=data["sample_ids"],
        detection_ids=data["detection_ids"],
        patches_field="ground_truth",
        uniqueness_field="uniqueness_score",
        cluster_field="cluster_label",
        k=10,
        decay="exponential",
        decay_param=0.5,
        save=True,
        verbose=True,
    )
"""

import numpy as np
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors


def _log(msg, verbose):
    if verbose:
        print(f"  {msg}")


# ── Shared writeback helper ───────────────────────────────────────────────────

def _write_patch_field(
    dataset,
    sample_ids: np.ndarray,
    detection_ids: np.ndarray,
    values: np.ndarray,
    patches_field: str,
    target_field: str,
    verbose: bool = True,
):
    """
    Writes `values` onto individual fo.Detection objects, one value per
    (sample_id, detection_id) row -- robust to ARBITRARY row order in the
    input arrays (does not assume rows are grouped/sorted by sample_id).

    Approach
    --------
    1. Build a single dict: detection_id -> value (a flat O(N) mapping
       pass over the input arrays, independent of their original order).
    2. Build a second dict: sample_id -> set of detection_ids that
       belong to it (also a flat O(N) pass), so we only fetch and save
       each sample ONCE regardless of how many detections it has or
       what order they appeared in the input arrays.
    3. Iterate sample_id groups, fetch that sample once, walk its
       patches_field detections, look up each one's value by
       detection.id in the first dict, set target_field, save once.

    Parameters
    ----------
    dataset        : fo.Dataset or fo.DatasetView
    sample_ids     : np.ndarray (N,) -- parent sample_id per row
    detection_ids  : np.ndarray (N,) -- fo.Detection.id per row, SAME
                      ROW ORDER as sample_ids/values (the only ordering
                      requirement -- across ROWS arbitrary order is fine)
    values         : np.ndarray (N,) -- value to write for each row
    patches_field  : str -- name of the Detections field to write into
    target_field   : str -- attribute name to set on each Detection
    verbose        : bool

    Returns
    -------
    n_written        : int -- detections successfully updated
    n_missing_sample : int -- rows whose sample_id wasn't found in dataset
    n_missing_det    : int -- rows whose detection_id wasn't found within
                        that sample's patches_field.detections
    """
    assert len(sample_ids) == len(detection_ids) == len(values), (
        f"sample_ids ({len(sample_ids)}), detection_ids ({len(detection_ids)}), "
        f"and values ({len(values)}) must all be the same length."
    )

    # Step 1: detection_id -> value (order-independent flat mapping)
    detid_to_value = {
        str(did): val for did, val in zip(detection_ids, values)
    }

    # Step 2: sample_id -> set of detection_ids belonging to it
    sampleid_to_detids = {}
    for sid, did in zip(sample_ids, detection_ids):
        sid = str(sid)
        sampleid_to_detids.setdefault(sid, set()).add(str(did))

    _log(f"Writing '{target_field}' onto {len(detid_to_value)} detections "
         f"across {len(sampleid_to_detids)} samples ...", verbose)

    n_written        = 0
    n_missing_sample  = 0
    n_missing_det     = 0

    all_sample_ids = list(sampleid_to_detids.keys())

    # ── Batched query instead of one dataset[sid] fetch per sample ──────────
    # dataset[sid] is effectively its own find_one query. Calling it
    # thousands of times sequentially (plus a separate sample.save() per
    # sample) sends thousands of individual round trips to MongoDB, which
    # can overload a manually-started mongod instance enough to make it
    # unresponsive. dataset.select(ids) issues ONE batched query (an $in
    # filter), and iter_samples(autosave=True) reuses a single cursor with
    # FiftyOne's own internal write batching -- a single MongoDB session
    # doing far fewer, larger operations instead of many tiny ones.
    #
    # chunk_size further bounds how large any single $in query/save batch
    # gets, in case the full sample set is itself very large.
    chunk_size = 2000

    for start in range(0, len(all_sample_ids), chunk_size):
        chunk_ids = all_sample_ids[start:start + chunk_size]
        found_in_chunk = set()

        view = dataset.select(chunk_ids)

        for sample in view.iter_samples(autosave=True, progress=verbose):
            sid = sample.id
            found_in_chunk.add(sid)
            expected_detids = sampleid_to_detids[sid]

            det_obj = sample[patches_field]
            if not det_obj or not det_obj.detections:
                n_missing_det += len(expected_detids)
                continue

            found_detids = set()
            for det in det_obj.detections:
                if det.id in expected_detids:
                    det[target_field] = detid_to_value[det.id]
                    found_detids.add(det.id)
                    n_written += 1

            missing_here = expected_detids - found_detids
            n_missing_det += len(missing_here)

        missing_sample_ids = set(chunk_ids) - found_in_chunk
        for sid in missing_sample_ids:
            n_missing_sample += len(sampleid_to_detids[sid])

    if n_missing_sample or n_missing_det:
        _log(f"  WARNING: {n_missing_sample} row(s) had an unresolvable "
             f"sample_id, {n_missing_det} row(s) had a detection_id not "
             f"found within that sample's '{patches_field}' field.", verbose)

    _log(f"  Done: {n_written} detections updated.", verbose)

    return n_written, n_missing_sample, n_missing_det


# ── Clustering ──────────────────────────────────────────────────────────────────

def compute_clustering_patches(
    dataset,
    embeddings: np.ndarray,
    sample_ids: np.ndarray,
    detection_ids: np.ndarray,
    patches_field: str = "ground_truth",
    cluster_field: str = "cluster_label",
    method: str = "kmeans",
    n_clusters: int = 15,
    n_init: int = 10,
    bandwidth=None,
    bandwidth_quantile: float = 0.3,
    seed: int = 42,
    save: bool = True,
    verbose: bool = True,
):
    """
    Clusters OBJECT/PATCH embeddings (pooled across every detection in the
    given arrays, regardless of which sample they came from) and writes
    the resulting cluster label onto each individual fo.Detection, not
    onto the parent sample.

    Mirrors compute_clustering's KMeans/MeanShift logic exactly -- the only
    difference is the writeback target (per-Detection instead of per-Sample).

    Parameters
    ----------
    dataset         : fo.Dataset or fo.DatasetView
    embeddings      : np.ndarray (N, dim) -- one row per detection
    sample_ids      : np.ndarray (N,) -- parent sample_id per row
    detection_ids   : np.ndarray (N,) -- fo.Detection.id per row (the join key)
    patches_field   : str -- name of the Detections field to write into
                       (default "ground_truth")
    cluster_field   : str -- attribute name to set on each Detection
                       (default "cluster_label")
    method          : "kmeans" | "meanshift"
    n_clusters      : int -- KMeans only
    n_init          : int -- KMeans only
    bandwidth       : float or None -- MeanShift only; estimated via
                       estimate_bandwidth(quantile=bandwidth_quantile) if None
    bandwidth_quantile : float -- MeanShift only, used when bandwidth is None
    seed            : int
    save            : bool -- write cluster labels back to the dataset's
                       Detection objects (default True)
    verbose         : bool

    Returns
    -------
    cluster_labels  : np.ndarray (N,) int -- SAME ROW ORDER as the input arrays
    embeddings_norm : np.ndarray (N, dim) -- L2-normalised embeddings actually clustered
    sample_ids      : np.ndarray (N,) -- echoed back unchanged, for convenience
    detection_ids   : np.ndarray (N,) -- echoed back unchanged, for convenience
    """
    assert len(embeddings) == len(sample_ids) == len(detection_ids), (
        "embeddings, sample_ids, and detection_ids must all be the same length."
    )

    embeddings_norm = normalize(embeddings, norm="l2")
    N = len(embeddings_norm)

    _log(f"Clustering {N} object/patch embeddings via '{method}' ...", verbose)

    if method == "kmeans":
        km = KMeans(n_clusters=n_clusters, init="k-means++",
                    n_init=n_init, random_state=seed)
        cluster_labels = km.fit_predict(embeddings_norm)

    elif method == "meanshift":
        bw = bandwidth
        if bw is None:
            bw = estimate_bandwidth(embeddings_norm, quantile=bandwidth_quantile,
                                    random_state=seed)
            _log(f"  Estimated bandwidth: {bw:.4f}", verbose)
        ms = MeanShift(bandwidth=bw, bin_seeding=True)
        cluster_labels = ms.fit_predict(embeddings_norm)

        n_found = len(np.unique(cluster_labels))
        if n_found == 1:
            _log(f"  WARNING: MeanShift found only 1 cluster -- bandwidth "
                 f"may be too large.", verbose)
        elif n_found > 50:
            _log(f"  WARNING: MeanShift found {n_found} clusters -- bandwidth "
                 f"may be too small.", verbose)

    else:
        raise ValueError(f"method must be 'kmeans' or 'meanshift', got '{method}'")

    sizes = {int(c): int((cluster_labels == c).sum()) for c in np.unique(cluster_labels)}
    _log(f"  Cluster sizes: {dict(sorted(sizes.items()))}", verbose)

    if save:
        _write_patch_field(
            dataset, sample_ids, detection_ids, cluster_labels,
            patches_field, cluster_field, verbose=verbose,
        )

    return cluster_labels, embeddings_norm, sample_ids, detection_ids


# ── Uniqueness ──────────────────────────────────────────────────────────────────

def _decay_weights(k, decay="exponential", decay_param=0.5):
    ranks = np.arange(1, k + 1, dtype=np.float64)
    if decay == "exponential":
        w = np.exp(-decay_param * ranks)
    elif decay == "linear":
        w = (k + 1 - ranks) / k
    elif decay == "power":
        w = 1.0 / (ranks ** decay_param)
    else:
        raise ValueError(f"Unknown decay '{decay}'.")
    return w / w.sum()


def compute_uniqueness_patches(
    dataset,
    embeddings: np.ndarray,
    sample_ids: np.ndarray,
    detection_ids: np.ndarray,
    patches_field: str = "ground_truth",
    uniqueness_field: str = "uniqueness_score",
    cluster_labels=None,
    cluster_field=None,
    k: int = 10,
    decay: str = "exponential",
    decay_param: float = 0.5,
    save: bool = True,
    verbose: bool = True,
):
    """
    Computes a kNN-based uniqueness score for each OBJECT/PATCH embedding,
    optionally WITHIN each cluster (if cluster_labels or cluster_field is
    given), and writes the score onto each individual fo.Detection.

    Mirrors compute_uniqueness_field_v2's scoring logic exactly -- the only
    difference is the writeback target (per-Detection instead of per-Sample).

    Parameters
    ----------
    dataset           : fo.Dataset or fo.DatasetView
    embeddings        : np.ndarray (N, dim) -- one row per detection
    sample_ids        : np.ndarray (N,) -- parent sample_id per row
    detection_ids     : np.ndarray (N,) -- fo.Detection.id per row (the join key)
    patches_field     : str -- name of the Detections field to write into
                         (default "ground_truth")
    uniqueness_field  : str -- attribute name to set on each Detection
                         (default "uniqueness_score")
    cluster_labels    : np.ndarray (N,) or None -- pass directly if you already
                         have them (e.g. from compute_clustering_patches),
                         SAME ROW ORDER as embeddings/sample_ids/detection_ids.
                         Takes precedence over cluster_field if both given.
    cluster_field     : str or None -- alternatively, read existing cluster
                         labels back off each Detection's attribute of this
                         name (must already be populated, e.g. by a prior
                         compute_clustering_patches(save=True) call).
    k                 : int -- kNN neighbours for uniqueness scoring
    decay             : "exponential" | "linear" | "power"
    decay_param       : float -- lambda (exponential) or p (power)
    save              : bool -- write scores back to the dataset (default True)
    verbose           : bool

    Returns
    -------
    uniqueness_scores : np.ndarray (N,) float, normalised to [0, 1]
                         (per-cluster if clustering is used, globally otherwise)
    sample_ids        : np.ndarray (N,) -- echoed back unchanged
    """
    assert len(embeddings) == len(sample_ids) == len(detection_ids), (
        "embeddings, sample_ids, and detection_ids must all be the same length."
    )

    embeddings_norm = normalize(embeddings, norm="l2")
    N = len(embeddings_norm)

    # ── Resolve cluster labels, if any ───────────────────────────────────────
    if cluster_labels is None and cluster_field is not None:
        _log(f"Reading cluster labels from Detection.{cluster_field} ...", verbose)
        lookup = {}

        # Same batching fix as _write_patch_field: one dataset.select(ids)
        # query instead of one dataset[sid] fetch per sample.
        unique_sids = [str(s) for s in np.unique(sample_ids)]
        chunk_size = 2000
        for start in range(0, len(unique_sids), chunk_size):
            chunk_ids = unique_sids[start:start + chunk_size]
            view = dataset.select(chunk_ids)
            for sample in view.iter_samples(progress=verbose):
                det_obj = sample[patches_field]
                if det_obj and det_obj.detections:
                    for det in det_obj.detections:
                        val = det.get_field(cluster_field) if hasattr(det, "get_field") else None
                        if val is not None:
                            lookup[det.id] = val

        cluster_labels = np.array([lookup.get(str(did), -1) for did in detection_ids])
        n_unresolved = (cluster_labels == -1).sum()
        if n_unresolved:
            _log(f"  WARNING: {n_unresolved} detections had no '{cluster_field}' "
                 f"value set -- treating them as their own single-member cluster (-1).",
                 verbose)

    uniqueness_scores = np.zeros(N, dtype=np.float64)

    if cluster_labels is not None:
        unique_clusters = np.unique(cluster_labels)
        _log(f"Computing per-cluster uniqueness across {len(unique_clusters)} "
             f"clusters ...", verbose)

        for cid in unique_clusters:
            idx = np.where(cluster_labels == cid)[0]
            n_c = len(idx)
            sub_embs = embeddings_norm[idx]

            k_eff = min(k, n_c - 1)
            if k_eff < 1:
                sub_scores = np.ones(n_c, dtype=np.float64)
            else:
                w = _decay_weights(k_eff, decay, decay_param)
                knn = NearestNeighbors(n_neighbors=k_eff + 1, metric="cosine", n_jobs=-1)
                knn.fit(sub_embs)
                dists, _ = knn.kneighbors(sub_embs)
                raw = (dists[:, 1:] * w).sum(axis=1)
                sub_scores = raw / raw.max() if raw.max() > 0 else raw

            uniqueness_scores[idx] = sub_scores
            _log(f"  cluster {cid}: n={n_c}  mean_uniq={sub_scores.mean():.4f}", verbose)

    else:
        _log(f"Computing global uniqueness across {N} embeddings ...", verbose)
        k_eff = min(k, N - 1)
        w = _decay_weights(k_eff, decay, decay_param)
        knn = NearestNeighbors(n_neighbors=k_eff + 1, metric="cosine", n_jobs=-1)
        knn.fit(embeddings_norm)
        dists, _ = knn.kneighbors(embeddings_norm)
        raw = (dists[:, 1:] * w).sum(axis=1)
        uniqueness_scores = raw / raw.max() if raw.max() > 0 else raw

    _log(f"Uniqueness stats: min={uniqueness_scores.min():.4f}  "
         f"mean={uniqueness_scores.mean():.4f}  max={uniqueness_scores.max():.4f}",
         verbose)

    if save:
        _write_patch_field(
            dataset, sample_ids, detection_ids, uniqueness_scores,
            patches_field, uniqueness_field, verbose=verbose,
        )

    return uniqueness_scores, sample_ids