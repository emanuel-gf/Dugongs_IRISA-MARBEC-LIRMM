"""
active_learning_selection.py
============================
Latent-space active-learning sample selection for the FLPLAN / dugong pipeline.

This module implements the three selection strategies described in the thesis
chapter (Latent Space Analysis for Domain Shift), plus a random baseline:

    active_learning_centroid(...)     # §4.7.1  Centroid Proximity
    aclr_centroid_uniqueness(...)     # §4.7.2  Centroid-Uniqueness (rho hybrid)
    aclr_ball_radius(...)             # §4.7.3  Ball-Radius Greedy (ACLR)
    random_baseline(...)              # uniform draw, no clustering

INTENDED PIPELINE
-----------------
    # 1) cluster the train pool ONCE (your existing function)
    cluster_labels, embeddings_norm, sample_ids = compute_clustering(
        train_view, embeddings=emb, sample_ids=ids, n_clusters=K)

    # 2) run a selection strategy -> writes its own JSON
    active_learning_centroid(
        train_view, cluster_labels, embeddings_norm, sample_ids,
        partitions=[0.05, 0.10, 0.20, 0.30], output_json="centroid.json")

DESIGN CONTRACT (why three aligned arrays)
------------------------------------------
Every selection function takes `embeddings_norm`, `sample_ids` and
`cluster_labels` as THREE ROW-ALIGNED arrays -- exactly the triple returned by
compute_clustering(). Passing them together is deliberate: it guarantees the
cluster label of row i always corresponds to embedding row i. If a function
re-read labels from a FiftyOne field or an .npz independently, row order could
silently drift and corrupt the selection.

`dataset` (a Dataset or DatasetView) is still passed, but used ONLY to look up
each selected sample's filepath BY ID (order-independent) when building JSON.

FILE-TRIPLE RESOLUTION (the JSON leaf)
--------------------------------------
Each sample's filepath is assumed to live at:  <root>/images/<stem>.<ext>
The label and metadata siblings are resolved as:
    label    = <root>/labels/<stem>.*
    metadata = <root>/metadata/<stem>.*
where <root> = Path(filepath).parent.parent. Extensions are auto-resolved by
globbing unless label_ext / meta_ext are pinned explicitly.

OUTPUT JSON STRUCTURE (one file per strategy)
---------------------------------------------
    {
      "0": {                         # single seed, keyed "0" (RT-DETR expects this)
        "train": {
          "p5":  {"<method>": {"images": [...], "labels": [...], "metadata": [...]}},
          "p10": {"<method>": {...}},
          ...
        }
      }
    }

NESTING GUARANTEE
-----------------
All strategies define a FIXED total order per cluster, then select the first
b_k samples for each partition. Per-cluster budgets are made monotonic across
ascending partitions (_nested_budgets), so the selection at p_i is a strict
subset of the selection at p_j for p_i < p_j:  p5 subset p10 subset p20 ...

NOTE ON UNIQUENESS NORMALISATION
--------------------------------
This module follows the chapter (Eq. uniqueness_normalised): per-cluster
MIN-MAX. Your earlier compute_uniqueness_field_v2 used max-only division;
the min-max form here is the one that matches the written text.
"""

import json
import math
import random
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


# ── tiny print helper ─────────────────────────────────────────────────────────

def _log(msg, verbose, level="info"):
    if not verbose:
        return
    prefix = {"info": "  ", "success": "OK   ", "warn": "WARN "}.get(level, "  ")
    print(f"{prefix}{msg}")


# ══════════════════════════════════════════════════════════════════════════════
#  DECAY WEIGHTS + WITHIN-CLUSTER UNIQUENESS  (pure numpy)
# ══════════════════════════════════════════════════════════════════════════════

def compute_decay_weights(k, decay="exponential", decay_param=0.5):
    """
    Weight vector of length k over neighbour ranks 1..k (NOT sum-normalised).

    decay : "exponential" (e^{-lambda j}) | "linear" ((k+1-j)/k) | "power" (j^-p)
    decay_param : lambda for exponential, p for power, ignored for linear.
    """
    ranks = np.arange(1, k + 1, dtype=np.float64)
    if decay == "exponential":
        w = np.exp(-decay_param * ranks)
    elif decay == "linear":
        w = (k + 1 - ranks) / k
    elif decay == "power":
        w = 1.0 / (ranks ** decay_param)
    else:
        raise ValueError(
            f"Unknown decay '{decay}'. Choose 'exponential', 'linear', 'power'."
        )
    return w


def _within_cluster_uniqueness(
    embeddings_norm, cluster_labels,
    k=5, decay="exponential", decay_param=0.5, verbose=True,
):
    """
    Within-cluster uniqueness score U~ (chapter Eq. uniqueness_score +
    uniqueness_normalised).

    For each cluster, the k nearest neighbours of every sample are restricted
    to that cluster. The score is a rank-weighted sum of cosine distances to
    those neighbours, then MIN-MAX normalised inside the cluster to [0, 1].

    Returns
    -------
    scores : np.ndarray (N,)  in [0, 1], aligned to embeddings_norm rows.
    """
    N = len(embeddings_norm)
    scores = np.zeros(N, dtype=np.float64)

    w_full = compute_decay_weights(k, decay, decay_param)
    w_full = w_full / w_full.sum()

    for cid in np.unique(cluster_labels):
        idx = np.where(cluster_labels == cid)[0]
        n_c = len(idx)

        if n_c < 2:
            # degenerate cluster: a lone sample is maximally distinctive
            scores[idx] = 1.0
            continue

        k_eff = min(k, n_c - 1)
        if k_eff < k:
            w = compute_decay_weights(k_eff, decay, decay_param)
            w = w / w.sum()
        else:
            w = w_full

        sub = embeddings_norm[idx]
        knn = NearestNeighbors(n_neighbors=k_eff + 1, metric="cosine", n_jobs=-1)
        knn.fit(sub)
        dists, _ = knn.kneighbors(sub)          # (n_c, k_eff+1), col 0 = self

        raw = (dists[:, 1:] * w).sum(axis=1)     # rank-weighted cosine distance

        mn, mx = raw.min(), raw.max()
        scores[idx] = (raw - mn) / (mx - mn + 1e-12)   # per-cluster min-max

        _log(f"  uniqueness cluster {int(cid):>3}: n={n_c:>4}  k_eff={k_eff}  "
             f"mean={scores[idx].mean():.4f}", verbose)

    return scores


# ══════════════════════════════════════════════════════════════════════════════
#  BUDGET ALLOCATION  (proportional-capped, then made monotonic across partitions)
# ══════════════════════════════════════════════════════════════════════════════

def _allocate_budget(cluster_sizes, total_budget, mode="proportional_capped"):
    """
    Allocate total_budget across clusters (single partition).

    proportional_capped: proportional base -> floor at 1 -> hard cap at size ->
    integer reconciliation -> redistribute cap surplus to largest-capacity
    clusters.  uniform: equal split (ablation).
    """
    cluster_ids = sorted(cluster_sizes.keys())
    total_N = sum(cluster_sizes.values())

    if mode == "uniform":
        base = total_budget // len(cluster_ids)
        budgets = {c: base for c in cluster_ids}
        leftover = total_budget - base * len(cluster_ids)
        for c in sorted(cluster_ids, key=lambda c: cluster_sizes[c], reverse=True):
            if leftover <= 0:
                break
            budgets[c] += 1
            leftover -= 1
        for c in cluster_ids:
            budgets[c] = min(budgets[c], cluster_sizes[c])
        return budgets

    if mode == "proportional_capped":
        raw = {c: total_budget * cluster_sizes[c] / total_N for c in cluster_ids}
        budgets = {c: max(1, round(raw[c])) for c in cluster_ids}
        for c in cluster_ids:
            budgets[c] = min(budgets[c], cluster_sizes[c])

        frac_order = sorted(
            cluster_ids,
            key=lambda c: (raw[c] - math.floor(raw[c]), cluster_sizes[c]),
            reverse=True,
        )
        drift = sum(budgets.values()) - total_budget
        if drift > 0:
            for c in reversed(frac_order):
                if drift == 0:
                    break
                if budgets[c] > 1:
                    budgets[c] -= 1
                    drift -= 1
        elif drift < 0:
            capacity_order = sorted(
                cluster_ids,
                key=lambda c: cluster_sizes[c] - budgets[c], reverse=True)
            for c in capacity_order:
                if drift == 0:
                    break
                if budgets[c] < cluster_sizes[c]:
                    budgets[c] += 1
                    drift += 1

        surplus = total_budget - sum(budgets.values())
        if surplus > 0:
            capacity_order = sorted(
                [c for c in cluster_ids if budgets[c] < cluster_sizes[c]],
                key=lambda c: cluster_sizes[c] - budgets[c], reverse=True)
            for c in capacity_order:
                if surplus == 0:
                    break
                give = min(cluster_sizes[c] - budgets[c], surplus)
                budgets[c] += give
                surplus -= give
        return budgets

    raise ValueError(f"budget_mode='{mode}' not supported.")


def _add_slots(budgets, cluster_sizes, need):
    """Add `need` slots to clusters with remaining capacity (monotonic: only adds)."""
    budgets = dict(budgets)
    cluster_ids = sorted(cluster_sizes)
    while need > 0:
        avail = [c for c in cluster_ids if budgets[c] < cluster_sizes[c]]
        if not avail:
            break
        remaining_cap = {c: cluster_sizes[c] - budgets[c] for c in avail}
        total_cap = sum(remaining_cap.values())
        added = 0
        for c in sorted(avail, key=lambda c: remaining_cap[c], reverse=True):
            if need - added <= 0:
                break
            give = max(1, int(round((need) * remaining_cap[c] / total_cap)))
            give = min(give, remaining_cap[c], need - added)
            budgets[c] += give
            added += give
        need -= added
        if added == 0:
            break
    return budgets


def _nested_budgets(cluster_sizes, partitions, N, budget_mode="proportional_capped",
                    verbose=True):
    """
    Per-cluster budgets for every partition, MONOTONIC across ascending
    partitions (guarantees strict nesting p5 subset p10 subset ...).

    The smallest partition is seeded with a standalone proportional-capped
    allocation. Each larger partition starts from the previous budgets and
    only ADDS the incremental slots needed to reach its target total.

    Returns
    -------
    budgets_by_p : dict {partition_float: {cluster_id: budget_int}}
    """
    parts = sorted(partitions)
    budgets_by_p = {}
    prev = None
    for p in parts:
        target = min(N, max(1, int(round(p * N))))
        if prev is None:
            b = _allocate_budget(cluster_sizes, target, mode=budget_mode)
        else:
            b = dict(prev)
            need = target - sum(b.values())
            if need > 0:
                b = _add_slots(b, cluster_sizes, need)
            # if need <= 0 (prev already >= target due to floor-at-1), keep prev
        budgets_by_p[p] = b
        prev = b
        _log(f"budget p{int(round(p*100))}: target={target}  "
             f"allocated={sum(b.values())}", verbose)
    return budgets_by_p


# ══════════════════════════════════════════════════════════════════════════════
#  BALL-RADIUS PENALTY  (soft mode -- reused from the original ACLR code)
# ══════════════════════════════════════════════════════════════════════════════

def _soft_penalty_propagation(embeddings_norm, uniqueness_scores, ball_radius, penalty):
    """
    Soft ball-radius degradation: visit points by descending uniqueness; each
    neighbour within ball_radius (L2 on unit vectors) has its working score
    multiplied by `penalty`. No point is ever removed.  L2^2 = 2(1 - cos), so
    ball_radius=0.5 <-> cosine >= 0.875.
    """
    tree = cKDTree(embeddings_norm)
    working = uniqueness_scores.copy().astype(np.float64)
    order = np.argsort(working, kind="stable")[::-1]
    for idx in order:
        for nb in tree.query_ball_point(embeddings_norm[idx], ball_radius):
            if nb != idx:
                working[nb] *= penalty
    return working


# ══════════════════════════════════════════════════════════════════════════════
#  PER-CLUSTER ORDER BUILDERS  (each returns GLOBAL indices in preference order)
# ══════════════════════════════════════════════════════════════════════════════

def _order_centroid(sub_embs, global_idx):
    """Ascending cosine distance to the L2-normalised cluster centroid."""
    centroid = sub_embs.mean(axis=0)
    centroid /= (np.linalg.norm(centroid) + 1e-12)
    cos_sim = sub_embs @ centroid                    # unit vectors -> cosine sim
    cos_dist = 1.0 - cos_sim
    local_order = np.argsort(cos_dist, kind="stable")   # closest first
    return global_idx[local_order]


def _select_centroid_uniqueness(emb, labels, uniq, sizes, partitions, N,
                                budget_mode, rho, verbose):
    """
    Direct centroid/uniqueness split (Algorithm 2), applied per cluster at each
    partition's own budget b_k:

        b_c = floor(rho * b_k)          # from centroid proximity
        b_u = b_k - b_c                 # from within-cluster uniqueness

    e.g. b_k = 10, rho = 0.6  ->  6 centroid + 4 uniqueness, regardless of how
    big the cluster is. The uniqueness picks exclude the centroid picks.

    Nesting is preserved because, per cluster, both quotas are monotonic in b_k
    (b_c and b_u never shrink as the budget grows), the centroid picks are a
    growing prefix of one fixed ordering, and the uniqueness picks are drawn in
    a fixed order from a candidate pool that only shrinks by items that are
    themselves promoted into the (still-selected) centroid set. So the p5
    selection stays a subset of p10, etc.

    Returns
    -------
    selections : dict {partition_float: [global row indices]}
    """
    # one fixed ordering per cluster for each criterion (global indices)
    c_orders, u_orders = {}, {}
    for cid in sorted(sizes):
        idx = np.where(labels == cid)[0]
        sub = emb[idx]
        centroid = sub.mean(axis=0)
        centroid /= (np.linalg.norm(centroid) + 1e-12)
        c_orders[cid] = idx[np.argsort(1.0 - (sub @ centroid), kind="stable")]  # closest first
        u_orders[cid] = idx[np.argsort(uniq[idx], kind="stable")[::-1]]          # most unique first

    budgets_by_p = _nested_budgets(sizes, partitions, N, budget_mode, verbose)

    selections = {}
    for p in sorted(partitions):
        b = budgets_by_p[p]
        sel, n_cent, n_uniq = [], 0, 0
        for cid in sorted(sizes):
            b_k = b[cid]
            b_c = int(math.floor(rho * b_k))      # centroid quota (chapter: floor)
            b_u = b_k - b_c                        # uniqueness quota

            centroid_pick = [int(x) for x in c_orders[cid][:b_c]]
            picked = set(centroid_pick)
            uniq_pick = [int(x) for x in u_orders[cid]
                         if int(x) not in picked][:b_u]

            sel.extend(centroid_pick)
            sel.extend(uniq_pick)
            n_cent += len(centroid_pick)
            n_uniq += len(uniq_pick)

        selections[p] = sel
        _log(f"partition p{int(round(p * 100))}: {len(sel)} selected "
             f"(centroid={n_cent}, uniqueness={n_uniq})", verbose)

    return selections


def _order_ball_radius_hard(sub_embs, sub_uniq, global_idx, ball_radius, backfill):
    """
    Algorithm 3 (ACLR, hard exclusion): greedily pick the highest-uniqueness
    sample, then permanently exclude every candidate within `ball_radius`.
    Repeat over survivors. The excluded ("near-duplicate") samples are then
    appended by descending uniqueness as a backfill tail so partitions can
    still reach their target count.

    Returns
    -------
    global_order : np.ndarray of global indices (permutation of the cluster)
    n_primary    : number of true greedy picks before backfill begins
    """
    n_c = len(sub_embs)
    tree = cKDTree(sub_embs)
    uniq_order = np.argsort(sub_uniq, kind="stable")[::-1]   # desc uniqueness

    excluded = np.zeros(n_c, dtype=bool)
    primary = []
    for i in uniq_order:
        if excluded[i]:
            continue
        primary.append(i)
        for nb in tree.query_ball_point(sub_embs[i], ball_radius):
            if nb != i:
                excluded[nb] = True

    n_primary = len(primary)
    if backfill:
        primary_set = set(primary)
        tail = [i for i in uniq_order if i not in primary_set]
        local_full = primary + tail
    else:
        local_full = primary                      # may under-fill large budgets

    return global_idx[np.array(local_full, dtype=int)], n_primary


def _order_ball_radius_soft(sub_embs, sub_uniq, global_idx, ball_radius, penalty):
    """Soft variant: degrade uniqueness by ball penalties, order by result desc."""
    degraded = _soft_penalty_propagation(sub_embs, sub_uniq, ball_radius, penalty)
    local_order = np.argsort(degraded, kind="stable")[::-1]
    return global_idx[local_order]


# ══════════════════════════════════════════════════════════════════════════════
#  SELECTION DRIVER  (fixed per-cluster orders -> nested per-partition subsets)
# ══════════════════════════════════════════════════════════════════════════════

def _select_from_orders(cluster_orders, cluster_sizes, partitions, N,
                        budget_mode, verbose):
    """
    Given a fixed preference order per cluster (global indices), produce the
    selected global-index list for every partition by taking the first b_k
    per cluster. Budgets are monotonic across partitions -> strict nesting.

    Returns
    -------
    selections : dict {partition_float: [global_idx, ...]}
    budgets_by_p : dict {partition_float: {cid: b_k}}
    """
    budgets_by_p = _nested_budgets(cluster_sizes, partitions, N, budget_mode, verbose)
    selections = {}
    for p in sorted(partitions):
        b = budgets_by_p[p]
        sel = []
        for cid in sorted(cluster_orders.keys()):
            order = cluster_orders[cid]
            sel.extend(int(x) for x in order[:b[cid]])
        selections[p] = sel
        _log(f"partition p{int(round(p*100))}: selected {len(sel)} "
             f"(target {min(N, max(1, int(round(p*N))))})", verbose)
    return selections, budgets_by_p


# ══════════════════════════════════════════════════════════════════════════════
#  FILE-TRIPLE RESOLUTION + JSON WRITER
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_sibling(sibling_dir, stem, ext, verbose):
    """Resolve <sibling_dir>/<stem>.<ext>; glob the extension if ext is None."""
    if ext is not None:
        return str(sibling_dir / f"{stem}{ext}")
    matches = sorted(sibling_dir.glob(f"{stem}.*"))
    if not matches:
        _log(f"no file matched {sibling_dir}/{stem}.*", verbose, level="warn")
        return None
    if len(matches) > 1:
        _log(f"multiple matches for {stem} in {sibling_dir} -> using {matches[0].name}",
             verbose, level="warn")
    return str(matches[0])


def _resolve_file_triple(filepath, label_ext, meta_ext, verbose):
    """
    filepath = <root>/images/<stem>.<ext>  ->  (image, label, metadata) paths.
    """
    p = Path(filepath)
    stem = p.stem
    root = p.parent.parent
    image = str(filepath)
    label = _resolve_sibling(root / "labels", stem, label_ext, verbose)
    meta = _resolve_sibling(root / "metadata", stem, meta_ext, verbose)
    return image, label, meta


def _build_and_write_json(dataset, method_name, selections_by_partition,
                          sample_ids, output_json, label_ext, meta_ext, verbose):
    """
    Turn {partition_float: [row_index,...]} into the nested JSON structure and
    write it. Row indices index into `sample_ids`; filepaths are looked up by
    ID (order-independent) from `dataset`.
    """
    # order-safe id -> filepath map (single aligned call)
    pairs = dataset.values(["id", "filepath"])
    id_to_fp = {sid: fp for sid, fp in zip(pairs[0],pairs[1])}

    result = {"0": {"train": {}}}
    for p in sorted(selections_by_partition.keys()):
        key = f"p{int(round(p * 100))}"
        images, labels, metadata = [], [], []
        for row in selections_by_partition[p]:
            sid = sample_ids[row]
            fp = id_to_fp.get(sid)
            if fp is None:
                _log(f"sample id {sid} not found in dataset -- skipped.",
                     verbose, level="warn")
                continue
            img, lab, met = _resolve_file_triple(fp, label_ext, meta_ext, verbose)
            images.append(img)
            labels.append(lab)
            metadata.append(met)
        result["0"]["train"][key] = {
            method_name: {"images": images, "labels": labels, "metadata": metadata}
        }
        _log(f"{key}: {len(images)} images resolved.", verbose)

    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    _log(f"wrote {out}", verbose, level="success")
    return result


# ── shared entry-point prep ────────────────────────────────────────────────────

def _prepare(embeddings_norm, sample_ids, cluster_labels):
    """Coerce inputs to aligned numpy arrays; re-normalise defensively."""
    emb = np.asarray(embeddings_norm, dtype=np.float32)
    emb = normalize(emb, norm="l2")                 # idempotent on unit vectors
    sids = list(sample_ids)
    labels = np.asarray(cluster_labels).astype(int)
    N = len(emb)
    if not (len(sids) == len(labels) == N):
        raise ValueError(
            f"length mismatch: embeddings={N}, sample_ids={len(sids)}, "
            f"cluster_labels={len(labels)} -- these must be row-aligned.")
    sizes = {int(c): int((labels == c).sum()) for c in np.unique(labels)}
    return emb, sids, labels, N, sizes


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC STRATEGY 1 -- CENTROID PROXIMITY  (§4.7.1)
# ══════════════════════════════════════════════════════════════════════════════

def active_learning_centroid(
    dataset,
    cluster_labels,
    embeddings_norm,
    sample_ids,
    partitions,
    output_json,
    budget_mode="proportional_capped",
    method_name="centroid",
    label_ext=None,
    meta_ext=None,
    verbose=True,
):
    """
    Centroid-proximity selection. Within each cluster, samples are ordered by
    ascending cosine distance to the L2-normalised centroid, and the first b_k
    are selected -- the modal, most representative core of each cluster.

    Parameters
    ----------
    dataset          : FiftyOne Dataset/View (used only for id->filepath lookup)
    cluster_labels   : (N,) int  cluster assignment per sample (from clustering)
    embeddings_norm  : (N, D)    L2-normalised embeddings (row-aligned)
    sample_ids       : (N,)      sample IDs (row-aligned)
    partitions       : list of floats, e.g. [0.05, 0.10, 0.20, 0.30]
    output_json      : path for the JSON file this function writes
    budget_mode      : "proportional_capped" | "uniform"
    label_ext/meta_ext : pin sibling extensions (e.g. ".txt"/".json") to skip
                         globbing; None -> auto-resolve by glob.

    Returns
    -------
    result : the nested dict that was written to output_json
    """
    emb, sids, labels, N, sizes = _prepare(embeddings_norm, sample_ids, cluster_labels)
    _log(f"[centroid] N={N}  clusters={len(sizes)}  partitions={partitions}", verbose)

    cluster_orders = {}
    for cid in sorted(sizes):
        idx = np.where(labels == cid)[0]
        cluster_orders[cid] = _order_centroid(emb[idx], idx)

    selections, _ = _select_from_orders(
        cluster_orders, sizes, partitions, N, budget_mode, verbose)

    return _build_and_write_json(
        dataset, method_name, selections, sids, output_json,
        label_ext, meta_ext, verbose)


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC STRATEGY 2 -- CENTROID-UNIQUENESS  (§4.7.2)
# ══════════════════════════════════════════════════════════════════════════════

def aclr_centroid_uniqueness(
    dataset,
    cluster_labels,
    embeddings_norm,
    sample_ids,
    partitions,
    output_json,
    rho=0.60,
    k=5,
    decay="exponential",
    decay_param=0.5,
    budget_mode="proportional_capped",
    method_name="centroid_uniqueness",
    label_ext=None,
    meta_ext=None,
    verbose=True,
):
    """
    Centroid-Uniqueness hybrid (Algorithm 2). Each cluster's budget is split by
    ratio rho: a fraction rho filled by centroid proximity (representative
    core), the remainder by descending within-cluster uniqueness U~
    (distinctive periphery), excluding already-picked centroid samples.

    Per cluster, at each partition's own budget b_k:
        b_c = floor(rho * b_k) centroid-proximity picks,
        b_u = b_k - b_c        within-cluster uniqueness picks.
    So b_k = 10, rho = 0.6 gives 6 centroid + 4 uniqueness in every cluster,
    independent of cluster size. See _select_centroid_uniqueness for the nesting
    argument.

    Extra parameters
    ----------------
    rho         : centroid fraction in (0, 1). Chapter uses 0.60.
    k           : neighbours for uniqueness (chapter: 5)
    decay       : uniqueness weight decay (chapter: "exponential")
    decay_param : lambda for exponential (chapter: 0.5)

    See active_learning_centroid for the shared parameters.
    """
    emb, sids, labels, N, sizes = _prepare(embeddings_norm, sample_ids, cluster_labels)
    _log(f"[centroid_uniqueness] N={N}  clusters={len(sizes)}  rho={rho}  k={k}",
         verbose)

    uniq = _within_cluster_uniqueness(emb, labels, k, decay, decay_param, verbose)

    selections = _select_centroid_uniqueness(
        emb, labels, uniq, sizes, partitions, N, budget_mode, rho, verbose)

    return _build_and_write_json(
        dataset, method_name, selections, sids, output_json,
        label_ext, meta_ext, verbose)


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC STRATEGY 3 -- BALL-RADIUS GREEDY (ACLR)  (§4.7.3)
# ══════════════════════════════════════════════════════════════════════════════

def aclr_ball_radius(
    dataset,
    cluster_labels,
    embeddings_norm,
    sample_ids,
    partitions,
    output_json,
    ball_radius=0.5,
    k=5,
    decay="exponential",
    decay_param=0.5,
    mode="hard",
    penalty=0.7,
    backfill=True,
    budget_mode="proportional_capped",
    method_name="ball_radius",
    label_ext=None,
    meta_ext=None,
    verbose=True,
):
    """
    Ball-Radius Greedy selection (Algorithm 3, ACLR). Within each cluster,
    samples are picked greedily by descending within-cluster uniqueness; after
    each pick, all candidates within L2 radius `ball_radius` are permanently
    excluded (hard mode). On the unit hypersphere L2^2 = 2(1 - cos), so
    ball_radius=0.5 <-> cosine >= 0.875.

    mode="hard" (default, matches Algorithm 3):
        permanent exclusion; excluded near-duplicates are appended as a
        backfill tail (backfill=True) so partitions still reach their target.
    mode="soft":
        no exclusion; uniqueness is multiplied by `penalty` per ball hit and
        samples are ordered by the degraded score (uses _soft_penalty_propagation).

    Extra parameters
    ----------------
    ball_radius : L2 exclusion radius (chapter: 0.5 -> cos 0.875)
    mode        : "hard" | "soft"
    penalty     : soft-mode multiplier per ball hit, in (0,1)
    backfill    : hard-mode only; append excluded samples so counts match the
                  other strategies. If False, large budgets may under-fill.

    See active_learning_centroid for the shared parameters.
    """
    if mode not in ("hard", "soft"):
        raise ValueError("mode must be 'hard' or 'soft'.")

    emb, sids, labels, N, sizes = _prepare(embeddings_norm, sample_ids, cluster_labels)
    cos_thr = 1.0 - (ball_radius ** 2) / 2.0
    _log(f"[ball_radius] N={N}  clusters={len(sizes)}  r={ball_radius} "
         f"(cos>={cos_thr:.4f})  mode={mode}  backfill={backfill}", verbose)

    uniq = _within_cluster_uniqueness(emb, labels, k, decay, decay_param, verbose)

    cluster_orders = {}
    for cid in sorted(sizes):
        idx = np.where(labels == cid)[0]
        if mode == "hard":
            order, n_primary = _order_ball_radius_hard(
                emb[idx], uniq[idx], idx, ball_radius, backfill)
            cluster_orders[cid] = order
            _log(f"  cluster {cid:>3}: n={len(idx):>4}  diverse_picks={n_primary}"
                 f"{'' if backfill else '  (no backfill)'}", verbose)
        else:
            cluster_orders[cid] = _order_ball_radius_soft(
                emb[idx], uniq[idx], idx, ball_radius, penalty)

    selections, _ = _select_from_orders(
        cluster_orders, sizes, partitions, N, budget_mode, verbose)

    return _build_and_write_json(
        dataset, method_name, selections, sids, output_json,
        label_ext, meta_ext, verbose)


# ══════════════════════════════════════════════════════════════════════════════
#  RANDOM BASELINE  (run once, no clustering)
# ══════════════════════════════════════════════════════════════════════════════

def random_baseline(
    dataset,
    sample_ids,
    partitions,
    output_json,
    seed=0,
    method_name="random",
    label_ext=None,
    meta_ext=None,
    verbose=True,
):
    """
    Uniform random selection from the full pool, no cluster stratification and
    no uniqueness weighting. Isolates the value of latent-space-guided
    selection over naive collection.

    The pool is shuffled ONCE and partitions take growing prefixes, so the
    random selection is itself nested (p5 subset p10 subset ...), consistent
    with the other strategies.

    Parameters
    ----------
    dataset     : FiftyOne Dataset/View (id->filepath lookup only)
    sample_ids  : (N,) sample IDs of the full train pool
    partitions  : list of floats
    output_json : path for the JSON file
    seed        : RNG seed (kept at 0 to match the single-seed convention)
    """
    sids = list(sample_ids)
    N = len(sids)
    _log(f"[random] N={N}  seed={seed}  partitions={partitions}", verbose)

    rng = random.Random(seed)
    shuffled = sids[:]
    rng.shuffle(shuffled)
    order = {sid: i for i, sid in enumerate(sids)}   # map back to row indices

    selections = {}
    for p in sorted(partitions):
        target = min(N, max(1, int(round(p * N))))
        chosen = shuffled[:target]
        selections[p] = [order[sid] for sid in chosen]
        _log(f"partition p{int(round(p*100))}: selected {len(chosen)}", verbose)

    return _build_and_write_json(
        dataset, method_name, selections, sids, output_json,
        label_ext, meta_ext, verbose)