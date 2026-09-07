"""
VERSION 4 - CENTROID PROXIMITY SELECTION (proportional-capped budget)
PROXIMITY TO THE CENTROID
run_aclr_pipeline_v4.py
=======================
Active Learning sample selection pipeline for the FLPLAN dataset.
Centroid-proximity variant — representative sampling instead of diversity.

Pipeline stages
---------------
  1.  Tag split  (skippable with --skip-tag)
        Tags each sample with train_{seed} / val_{seed} / test_{seed}.

  2.  Load embeddings
        Reads full_embeddings for all train_{seed} tagged samples.

  3.  Centroid-proximity selection
        Per seed:
          a) KMeans clustering on the FULL train pool — fixed for all partitions.
        Per partition:
          b) Proportional-capped budget allocation across clusters.
          c) Per-cluster selection of the budget_c samples CLOSEST to the
             cluster centroid (L2 distance, ascending sort).

        Because the cluster structure is frozen per seed, smaller partitions
        select a strict subset of what larger partitions select (nesting
        property): p5 ⊆ p10 ⊆ p20 ...

  4.  Random baseline
        Random sampling from the same train pool.

  5.  Save results
        Writes {seed: {partition: {method: [sample_ids]}}} to --output-json.

Comparison to V3
----------------
  V3 (ACLR / ball-radius):  within each cluster, selects the most DIVERSE
      samples — favouring unique, peripheral points via greedy ball-radius
      penalisation.

  V4 (centroid proximity):  within each cluster, selects the most
      REPRESENTATIVE samples — favouring the dense core via L2 distance to
      the centroid.

  Everything else is identical: same clustering (fixed per seed), same
  proportional-capped budget allocation, same random baseline.  This makes
  V3 vs V4 a clean ablation of the within-cluster selection criterion.

Budget allocation strategy
--------------------------
  "proportional_capped"  (default)
        Each cluster receives a budget proportional to its size.  Guardrails:
          1. Hard cap  : budget_c <= size_c
          2. Floor     : budget_c >= 1
          3. Redistribution of capped surplus to largest-capacity clusters.

  "uniform"
        Equal split — kept for ablation.

Usage
-----
  python run_aclr_pipeline_v4.py \\
      --dataset FLPLAN --port 44123 \\
      --seeds 0 1 2 \\
      --partitions 0.05 0.10 0.20 0.30 0.40 0.50 1.00 \\
      --embeddings-field full_embeddings \\
      --output-json /share/home/e2406743/results/v4_selections.json

  # Skip tagging if already done
  python run_aclr_pipeline_v4.py ... --skip-tag

  # Use legacy uniform budget split
  python run_aclr_pipeline_v4.py ... --budget-mode uniform
"""

import os
import json
import math
import random
import argparse
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="Centroid-proximity selection + random baseline for FLPLAN.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset",             "-d", required=True)
    p.add_argument("--port",                default="44123")
    p.add_argument("--seeds",               nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--partitions",          nargs="+", type=float,
                   default=[0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 1.00])
    p.add_argument("--embeddings-field",    default="full_embeddings")
    p.add_argument("--stratify-by",         default="m_flight")
    p.add_argument("--train-size",          type=float, default=0.85)
    p.add_argument("--val-size",            type=float, default=0.15)
    p.add_argument("--test-size",           type=float, default=0.15)
    # clustering — fixed once per seed on the full train pool
    p.add_argument("--n-clusters",          type=int, default=None,
                   help="Fixed number of KMeans clusters per seed.  "
                        "If omitted, adaptive_n_clusters() is used on the "
                        "full train pool. (default: None → adaptive)")
    p.add_argument("--target-cluster-size", type=int, default=10,
                   help="Target samples per cluster for adaptive n_clusters "
                        "when --n-clusters is not set. (default: 10)")
    # budget
    p.add_argument("--budget-mode",         default="proportional_capped",
                   choices=["uniform", "proportional_capped"],
                   help="Budget allocation across clusters. (default: proportional_capped)")
    # misc
    p.add_argument("--output-json",         "-o", required=True)
    p.add_argument("--skip-tag",            action="store_true",
                   help="Skip tagging — assume tags already exist.")
    return p.parse_args()


# ── Tag split ─────────────────────────────────────────────────────────────────

def tag_train_test_seeded_split(
    dataset, fo, F,
    stratify_by="m_flight",
    train_size=0.85,
    val_size=0.15,
    test_size=0.15,
    num_seeds=1,
    test_buffer=0.05,
    verbose=True,
):
    """
    Splits dataset into train/val/test by tagging samples, stratified by
    a categorical field (e.g. flight mission).  Entire flights are kept
    together — no flight is split across train and test.

    Tags written: train_{seed}, val_{seed}, test_{seed}, notin_TEST_{seed}
    """
    def _log(msg):
        if verbose:
            print(f"  {msg}")

    flight_counts = dict(
        sorted(dataset.count_values(stratify_by).items(), key=lambda x: x[1])
    )
    all_flights  = list(flight_counts.keys())
    total_images = dataset.count()
    _log(f"Dataset: {total_images} images  |  "
         f"{len(all_flights)} flights via '{stratify_by}'")

    used_test_flights = []

    for seed in range(num_seeds):
        rng = random.Random(seed)
        _log(f"\n── Seed {seed} ──────────────────────────────")

        target_test_n   = int(total_images * test_size)
        target_test_max = int(total_images * (test_size + test_buffer))

        available = [f for f in all_flights if f not in used_test_flights]
        if not available:
            available = all_flights.copy()

        shuffled         = rng.sample(available, len(available))
        test_flights     = []
        test_count       = 0
        remaining_budget = 0

        for flight in shuffled:
            n = flight_counts[flight]
            if test_count + n + remaining_budget <= target_test_max:
                test_flights.append(flight)
                test_count      += n
                remaining_budget = 0
            else:
                remaining_budget += max(0, target_test_n - test_count)
            if test_count >= target_test_n:
                break

        _log(f"Test flights: {test_flights}  "
             f"({test_count} images = {test_count/total_images:.1%})")
        used_test_flights.extend(test_flights)

        for flight in test_flights:
            ids = dataset.match(F(stratify_by) == flight).values("id")
            dataset.select(ids).tag_samples(f"test_{seed}")

        non_test = [f for f in all_flights if f not in test_flights]
        for flight in non_test:
            ids = dataset.match(F(stratify_by) == flight).values("id")
            dataset.select(ids).tag_samples(f"notin_TEST_{seed}")

        tv_view = dataset.match(F(stratify_by).is_in(non_test))
        ids_tv  = tv_view.values("id")
        strata  = tv_view.values(stratify_by)

        train_ids, val_ids = train_test_split(
            ids_tv, test_size=val_size,
            stratify=strata, random_state=seed, shuffle=True,
        )
        dataset.select(train_ids).tag_samples(f"train_{seed}")
        dataset.select(val_ids).tag_samples(f"val_{seed}")
        _log(f"train_{seed}={len(train_ids)}  val_{seed}={len(val_ids)}  "
             f"test_{seed}={test_count}")


# ── Embedding loader ──────────────────────────────────────────────────────────

def load_train_embeddings(dataset, seed, embeddings_field, verbose=True):
    """
    Load L2-normalised embeddings for all train_{seed} tagged samples.
    Returns (embeddings_norm, sample_ids).
    """
    print(f"\n  Loading embeddings for train_{seed} ...")
    embeddings = []
    sample_ids = []

    for sample in dataset.match_tags(f"train_{seed}").iter_samples(
            progress=verbose):
        emb = sample.get_field(embeddings_field)
        if emb is None:
            continue
        embeddings.append(np.array(emb, dtype=np.float32))
        sample_ids.append(sample.id)

    if not embeddings:
        raise ValueError(
            f"No embeddings in '{embeddings_field}' for train_{seed}."
        )

    arr      = np.stack(embeddings)
    arr_norm = normalize(arr, norm="l2")
    print(f"  {len(arr_norm)} embeddings  dim={arr_norm.shape[1]}")
    return arr_norm, sample_ids


# ── Adaptive n_clusters ───────────────────────────────────────────────────────

def adaptive_n_clusters(total_samples,
                        target_cluster_size=10,
                        min_clusters=3,
                        min_points_per_cluster=2):
    """
    Compute n_clusters from the FULL train pool size.

    Called once per seed (not per partition).  Uses target_cluster_size as
    the desired points-per-cluster.  Falls back to sqrt heuristic when the
    pool is too small to fill that many clusters.
    """
    ideal = total_samples // target_cluster_size
    if ideal >= min_points_per_cluster:
        return target_cluster_size
    return max(min_clusters, int(math.floor(math.sqrt(total_samples))))


# ── KMeans — fitted once per seed ────────────────────────────────────────────

def fit_kmeans(embeddings_norm, n_clusters, seed):
    """
    Fit KMeans on the full train pool for this seed.

    Returns
    -------
    cluster_labels : np.ndarray (N,)   integer cluster assignment per sample
    centroids      : np.ndarray (K, D) cluster centroid coordinates
    """
    print(f"  Fitting KMeans: n_clusters={n_clusters}  "
          f"N={len(embeddings_norm)}  seed={seed}")
    km             = KMeans(n_clusters=n_clusters, init="k-means++",
                            n_init=10, random_state=seed)
    cluster_labels = km.fit_predict(embeddings_norm)
    centroids      = km.cluster_centers_
    sizes          = {int(c): int((cluster_labels == c).sum())
                      for c in np.unique(cluster_labels)}
    print(f"  Cluster sizes: {dict(sorted(sizes.items()))}")
    return cluster_labels, centroids


# ── Budget allocation ─────────────────────────────────────────────────────────

def _allocate_budget(cluster_sizes, total_budget, mode="proportional_capped"):
    """
    Allocate total_budget across clusters.

    Parameters
    ----------
    cluster_sizes : dict  {cluster_id (int): size (int)}
    total_budget  : int   total number of samples to select
    mode          : str   "proportional_capped" | "uniform"

    Returns
    -------
    budgets : dict  {cluster_id (int): budget (int)}

    proportional_capped strategy
    ----------------------------
    Step 1 — Proportional base
        budget_c = round(total_budget * size_c / total_N)

    Step 2 — Floor
        budget_c = max(1, budget_c)  — every non-empty cluster contributes

    Step 3 — Hard cap
        budget_c = min(budget_c, size_c)  — can't select more than available

    Step 4 — Integer reconciliation
        After rounding, the sum may drift from total_budget.  Corrected by
        adding/removing 1 from clusters sorted by fractional remainder
        (ties broken by cluster size descending).

    Step 5 — Redistribute surplus freed by hard cap
        Surplus flows to uncapped clusters with the most remaining capacity
        (size_c - budget_c), largest first.
    """
    cluster_ids = sorted(cluster_sizes.keys())
    total_N     = sum(cluster_sizes.values())

    # ── Uniform ───────────────────────────────────────────────────────────
    if mode == "uniform":
        base     = total_budget // len(cluster_ids)
        budgets  = {c: base for c in cluster_ids}
        leftover = total_budget - base * len(cluster_ids)
        for c in sorted(cluster_ids,
                        key=lambda c: cluster_sizes[c], reverse=True):
            if leftover <= 0:
                break
            budgets[c] += 1
            leftover   -= 1
        for c in cluster_ids:
            budgets[c] = min(budgets[c], cluster_sizes[c])
        return budgets

    # ── Proportional capped ───────────────────────────────────────────────
    if mode == "proportional_capped":

        # Step 1: raw proportional (float)
        raw = {c: total_budget * cluster_sizes[c] / total_N
               for c in cluster_ids}

        # Step 2: round to int + floor at 1
        budgets = {c: max(1, round(raw[c])) for c in cluster_ids}

        # Step 3: hard cap at cluster size
        for c in cluster_ids:
            budgets[c] = min(budgets[c], cluster_sizes[c])

        # Step 4: integer reconciliation
        frac_order = sorted(cluster_ids,
                            key=lambda c: (raw[c] - math.floor(raw[c]),
                                          cluster_sizes[c]),
                            reverse=True)
        drift = sum(budgets.values()) - total_budget
        if drift > 0:
            for c in reversed(frac_order):
                if drift == 0:
                    break
                if budgets[c] > 1:
                    budgets[c] -= 1
                    drift      -= 1
        elif drift < 0:
            capacity_order = sorted(cluster_ids,
                                    key=lambda c: cluster_sizes[c] - budgets[c],
                                    reverse=True)
            for c in capacity_order:
                if drift == 0:
                    break
                if budgets[c] < cluster_sizes[c]:
                    budgets[c] += 1
                    drift      += 1

        # Step 5: redistribute surplus freed by the hard cap
        current_total = sum(budgets.values())
        surplus       = total_budget - current_total

        if surplus > 0:
            capacity_order = sorted(
                [c for c in cluster_ids if budgets[c] < cluster_sizes[c]],
                key=lambda c: cluster_sizes[c] - budgets[c],
                reverse=True,
            )
            for c in capacity_order:
                if surplus == 0:
                    break
                room        = cluster_sizes[c] - budgets[c]
                give        = min(room, surplus)
                budgets[c] += give
                surplus    -= give

        return budgets

    raise ValueError(f"budget_mode='{mode}' not supported.")


# ════════════════════════════════════════════════════════════════════════════
#  INNER FUNCTION — centroid-proximity selection, pure numpy
# ════════════════════════════════════════════════════════════════════════════

def _cluster_centroid_selection_from_arrays(
    embeddings_norm:   np.ndarray,      # (N, D) L2-normalised
    cluster_labels:    np.ndarray,      # (N,)  integer cluster assignments
                                        #        — FIXED, same for all partitions
    centroids:         np.ndarray,      # (K, D) cluster centroid coordinates
    partition_size:    float,
    budget_mode:       str   = "proportional_capped",
    seed:              int   = 42,
    verbose:           bool  = True,
) -> tuple[list, dict]:
    """
    Centroid-proximity selection — pure numpy inner function.

    For each cluster, computes the L2 distance from every sample to its
    cluster centroid and selects the budget_c closest samples.  The closest
    samples are the most representative of the cluster's dense core.

    cluster_labels and centroids are supplied externally and are FIXED for
    all partitions of a given seed, giving the nesting property
    p5 ⊆ p10 ⊆ p20 (approximately).

    Parameters
    ----------
    embeddings_norm  : (N, D) L2-normalised embedding matrix
    cluster_labels   : (N,)  integer cluster label per sample (fixed per seed)
    centroids        : (K, D) KMeans centroid coordinates
    partition_size   : fraction of N to select
    budget_mode      : "proportional_capped" | "uniform"
    seed             : random seed for tie-breaking on equal distances
    verbose          : print per-cluster stats

    Returns
    -------
    selected_indices : list of integer indices into embeddings_norm
    cluster_stats    : dict keyed by cluster_id with selection metadata
    """
    def _log(msg):
        if verbose:
            print(f"    {msg}")

    N               = len(embeddings_norm)
    total_budget    = max(1, int(round(partition_size * N)))
    unique_clusters = np.unique(cluster_labels)
    cluster_sizes   = {int(c): int((cluster_labels == c).sum())
                       for c in unique_clusters}

    # ── Budget allocation ─────────────────────────────────────────────────
    budgets = _allocate_budget(cluster_sizes, total_budget, mode=budget_mode)

    _log(f"N={N}  total_budget={total_budget}  "
         f"n_clusters={len(unique_clusters)}  budget_mode={budget_mode}")
    _log(f"cluster sizes: {dict(sorted(cluster_sizes.items()))}")
    _log(f"budgets:       {dict(sorted(budgets.items()))}")
    _log(f"allocated:     {sum(budgets.values())}")

    # ── Per-cluster centroid-proximity selection ───────────────────────────
    rng           = random.Random(seed)
    selected_idx  = []
    cluster_stats = {}

    for cluster_id in sorted(unique_clusters):
        cid         = int(cluster_id)
        cluster_idx = np.where(cluster_labels == cid)[0]   # global indices
        n_c         = len(cluster_idx)
        budget_c    = budgets[cid]

        sub_embs  = embeddings_norm[cluster_idx]            # (n_c, D)
        centroid  = centroids[cid]                          # (D,)

        # L2 distance to centroid — ascending sort → closest first
        distances    = np.linalg.norm(sub_embs - centroid, axis=1)
        sorted_local = np.argsort(distances)                # local indices

        # Select the budget_c closest samples
        chosen_local  = sorted_local[:budget_c].tolist()

        # Fallback: should never be needed since budget_c <= n_c by construction,
        # but guard against any edge-case rounding
        if len(chosen_local) < budget_c:
            remaining = [i for i in range(n_c) if i not in set(chosen_local)]
            rng.shuffle(remaining)
            chosen_local.extend(remaining[:budget_c - len(chosen_local)])

        # Map local → global indices
        chosen_global = [int(cluster_idx[i]) for i in chosen_local]
        selected_idx.extend(chosen_global)

        cluster_stats[cid] = {
            "n_in_cluster":  n_c,
            "budget":        budget_c,
            "n_selected":    len(chosen_global),
            "mean_dist":     float(distances.mean()),
            "min_dist":      float(distances.min()),
            "max_dist":      float(distances.max()),
            "selected_mean_dist": float(distances[sorted_local[:budget_c]].mean()),
        }

        _log(f"cluster {cid:>3}: "
             f"n={n_c:>4}  budget={budget_c:>3}  "
             f"selected={len(chosen_global):>3}  "
             f"mean_dist={distances.mean():.4f}  "
             f"sel_mean_dist={distances[sorted_local[:budget_c]].mean():.4f}")

    _log(f"→ total selected: {len(selected_idx)} / {total_budget} target")
    return selected_idx, cluster_stats


# ════════════════════════════════════════════════════════════════════════════
#  FIFTYONE WRAPPER — loads from fields, calls inner function
# ════════════════════════════════════════════════════════════════════════════

def compute_centroid_selection(
    dataset,
    embeddings_field:  str   = "full_embeddings",
    cluster_field:     str   = "cluster_label",
    centroid_field:    str   = "cluster_centroid",
    partition_size:    float = 0.10,
    budget_mode:       str   = "proportional_capped",
    seed:              int   = 42,
    verbose:           bool  = True,
) -> tuple[list, dict]:
    """
    FiftyOne wrapper around _cluster_centroid_selection_from_arrays.

    Loads embeddings, cluster labels, and centroids from FiftyOne fields,
    runs the inner function, and returns selected sample IDs (strings).

    Note: this wrapper is for notebook / standalone use where cluster labels
    and centroids are already stored as FiftyOne fields.  In the main
    pipeline the clustering is fitted via fit_kmeans() and passed directly.

    Parameters
    ----------
    dataset          : FiftyOne dataset or view
    embeddings_field : field with L2-normalised embeddings
    cluster_field    : field with integer cluster labels (pre-fitted)
    centroid_field   : field with centroid coordinates per sample
                       (each sample stores its cluster's centroid vector)
    partition_size   : fraction of dataset to select
    budget_mode      : "proportional_capped" | "uniform"
    seed             : random seed for tie-breaking
    verbose          : print per-cluster stats

    Returns
    -------
    selected_ids  : list of FiftyOne sample ID strings
    cluster_stats : dict with per-cluster metadata
    """
    def _log(msg):
        if verbose:
            print(f"  {msg}")

    _log(f"Loading embeddings='{embeddings_field}'  "
         f"cluster='{cluster_field}' ...")

    sample_ids     = []
    emb_list       = []
    cluster_list   = []
    centroid_map   = {}                     # cluster_id → centroid vector

    for sample in dataset.iter_samples(progress=verbose):
        emb     = sample.get_field(embeddings_field)
        clust   = sample.get_field(cluster_field)
        centroid = sample.get_field(centroid_field)
        if emb is None or clust is None or centroid is None:
            continue
        cid = int(clust)
        sample_ids.append(sample.id)
        emb_list.append(np.array(emb, dtype=np.float32))
        cluster_list.append(cid)
        if cid not in centroid_map:
            centroid_map[cid] = np.array(centroid, dtype=np.float32)

    if not emb_list:
        raise ValueError(
            f"No samples found with '{embeddings_field}', "
            f"'{cluster_field}', and '{centroid_field}' all set."
        )

    embeddings_norm = normalize(np.stack(emb_list), norm="l2")
    cluster_labels  = np.array(cluster_list)
    K               = len(centroid_map)
    D               = embeddings_norm.shape[1]
    centroids       = np.zeros((K, D), dtype=np.float32)
    for cid, vec in centroid_map.items():
        centroids[cid] = vec

    N = len(sample_ids)
    _log(f"Loaded {N} samples  dim={D}  clusters={K}")

    selected_idx, cluster_stats = _cluster_centroid_selection_from_arrays(
        embeddings_norm = embeddings_norm,
        cluster_labels  = cluster_labels,
        centroids       = centroids,
        partition_size  = partition_size,
        budget_mode     = budget_mode,
        seed            = seed,
        verbose         = verbose,
    )

    selected_ids = [sample_ids[i] for i in selected_idx]
    return selected_ids, cluster_stats


# ════════════════════════════════════════════════════════════════════════════
#  CENTROID PIPELINE — receives pre-fitted labels+centroids, iterates partitions
# ════════════════════════════════════════════════════════════════════════════

def centroid_pipeline(
    embeddings_norm,
    sample_ids,
    cluster_labels,         # pre-fitted — fixed for ALL partitions of this seed
    centroids,              # (K, D) — from fit_kmeans()
    partitions,             # list of floats, e.g. [0.05, 0.10, 0.20, ...]
    budget_mode  = "proportional_capped",
    seed         = 42,
):
    """
    Centroid-proximity selection for all partitions of one seed.

    cluster_labels and centroids are computed ONCE on the full train pool
    before calling this function (see fit_kmeans()).  The same cluster
    structure is reused for every partition, giving the nesting property
    p5 ⊆ p10 ⊆ p20.

    Parameters
    ----------
    embeddings_norm : (N, D) L2-normalised embeddings for the full train pool
    sample_ids      : list of N FiftyOne sample ID strings
    cluster_labels  : (N,) integer array — KMeans result on full train pool
    centroids       : (K, D) KMeans centroid coordinates
    partitions      : list of partition fractions to process
    budget_mode     : "proportional_capped" | "uniform"
    seed            : random seed for tie-breaking

    Returns
    -------
    results : dict  {partition_key: {"ids": [...], "stats": {...}}}
    """
    results = {}

    for partition_size in partitions:
        key = f"p{int(partition_size * 100)}"
        print(f"\n    [{key}] partition={partition_size:.0%}  "
              f"target={int(round(partition_size * len(embeddings_norm)))}")

        selected_idx, cluster_stats = _cluster_centroid_selection_from_arrays(
            embeddings_norm = embeddings_norm,
            cluster_labels  = cluster_labels,
            centroids       = centroids,
            partition_size  = partition_size,
            budget_mode     = budget_mode,
            seed            = seed,
            verbose         = True,
        )

        selected_ids = [sample_ids[i] for i in selected_idx]
        print(f"    [{key}] final selected: {len(selected_ids)}")

        results[key] = {
            "ids":   selected_ids,
            "stats": cluster_stats,
        }

    return results


# ── Random baseline ───────────────────────────────────────────────────────────

def random_baseline(sample_ids, partition_size, seed):
    """Random sample WITHOUT replacement from the full train pool."""
    N      = len(sample_ids)
    target = math.floor(partition_size * N)
    rng    = random.Random(seed)
    return rng.sample(sample_ids, k=min(target, N))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    os.environ["FIFTYONE_DATABASE_URI"] = f"mongodb://localhost:{args.port}"

    import fiftyone as fo
    from fiftyone import ViewField as F

    try:
        print(f"Connected to MongoDB at localhost:{args.port}. "
              f"Datasets: {fo.list_datasets()}")
    except Exception as e:
        print(f"ERROR: {e}"); return

    assert args.dataset in fo.list_datasets(), \
        f"Dataset '{args.dataset}' not found."
    dataset = fo.load_dataset(args.dataset)
    print(f"Loaded '{args.dataset}' — {len(dataset)} samples.")

    # ── Stage 1: Tag split ────────────────────────────────────────────────
    if not args.skip_tag:
        print("\n[Stage 1] Tagging splits ...")
        tag_train_test_seeded_split(
            dataset=dataset, fo=fo, F=F,
            stratify_by=args.stratify_by,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            num_seeds=len(args.seeds),
            verbose=True,
        )
    else:
        print("\n[Stage 1] Skipped (--skip-tag).")

    # ── Stages 2–4: centroid selection + random per seed ──────────────────
    results = {}

    for seed in args.seeds:
        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")

        # Stage 2: load embeddings for this seed's train pool
        embeddings_norm, sample_ids = load_train_embeddings(
            dataset, seed, args.embeddings_field, verbose=False
        )
        N = len(sample_ids)
        print(f"  Train pool: {N} samples")

        # Stage 3a: fit KMeans ONCE on the full train pool for this seed
        n_clusters = (
            args.n_clusters
            if args.n_clusters is not None
            else adaptive_n_clusters(N,
                                     target_cluster_size=args.target_cluster_size)
        )
        cluster_labels, centroids = fit_kmeans(embeddings_norm, n_clusters, seed)

        # Stage 3b: centroid-proximity selection for all partitions
        print(f"\n  [Centroid] running all partitions with fixed clustering ...")
        centroid_results = centroid_pipeline(
            embeddings_norm = embeddings_norm,
            sample_ids      = sample_ids,
            cluster_labels  = cluster_labels,
            centroids       = centroids,
            partitions      = args.partitions,
            budget_mode     = args.budget_mode,
            seed            = seed,
        )

        # Stage 4: random baseline + assemble results
        print(f"\n  [Random] running all partitions ...")
        results[str(seed)] = {}

        for partition in args.partitions:
            key = f"p{int(partition * 100)}"
            results[str(seed)][key] = {}

            # Centroid selection
            results[str(seed)][key]["centroid"]       = centroid_results[key]["ids"]
           # results[str(seed)][key]["centroid_stats"]  = centroid_results[key]["stats"]

            # Random
            rand_ids = random_baseline(sample_ids, partition, seed)
            results[str(seed)][key]["random"] = rand_ids

            print(f"  {key}  Centroid={len(centroid_results[key]['ids']):>5}  "
                  f"Random={len(rand_ids):>5}")

    # ── Save ──────────────────────────────────────────────────────────────
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {out}")

    print("\n── Summary ──────────────────────────────────────────────────")
    for seed in args.seeds:
        for partition in args.partitions:
            key = f"p{int(partition * 100)}"
            nc = len(results[str(seed)][key]["centroid"])
            nr = len(results[str(seed)][key]["random"])
            print(f"  SEED{seed}  {key:>4}  Centroid={nc:>5}  Random={nr:>5}")


if __name__ == "__main__":
    main()