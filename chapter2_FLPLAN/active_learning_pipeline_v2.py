"""
VERSION 2 - BALL RADIUS
run_aclr_pipeline.py
====================
Modified Active Learning sample selection pipeline for the FLPLAN dataset.

Diverse-within cluster sample selection based on latent space.

Pipeline stages
---------------
  1.  Tag split  (skippable with --skip-tag)
        Tags each sample with train_{seed} / val_{seed} / test_{seed}.

  2.  Load embeddings
        Reads full_embeddings for all train_{seed} tagged samples.

  3.  ACLR selection  (cluster-stratified diversity sampling)
        For each partition size:
         - Adaptative cluster size: Compute the size of the clustering given the partition.
         Followed by:
          a) KMeans clustering
          b) Per-cluster weighted kNN uniqueness scoring.
          c) Per-cluster ball-radius diversity selection.
       

  4.  Random baseline
        Random sampling from the same train pool.

  5.  Save results
        Writes {seed: {partition: {method: [sample_ids]}}} to --output-json.

Architecture
------------
  _cluster_diverse_selection_from_arrays()
      Pure-numpy inner function.  Operates entirely on arrays — no FiftyOne
      dependency.  Called directly by the pipeline for speed.

  compute_cluster_diverse_selection()
      FiftyOne wrapper.  Loads arrays from dataset fields, calls the inner
      function, optionally saves nothing (selection only).  Used from notebooks
      or standalone scripts where data lives in FiftyOne.

Usage
-----
  python run_aclr_pipeline.py \\
      --dataset FLPLAN --port 44123 \\
      --seeds 0 63 72 \\
      --partitions 0.05 0.10 0.25 0.50 0.75 \\
      --embeddings-field full_embeddings \\
      --output-json /share/home/e2406743/results/aclr_selections.json

  # Skip tagging if already done
  python run_aclr_pipeline.py ... --skip-tag
"""

import os
import json
import math
import random
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="Cluster-stratified ACLR + random baseline for FLPLAN.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset",            "-d", required=True)
    p.add_argument("--port",               default="44123")
    p.add_argument("--seeds",              nargs="+", type=int, default=[0, 63, 72])
    p.add_argument("--partitions",         nargs="+", type=float,
                   default=[0.05, 0.10, 0.25, 0.50, 0.75])
    p.add_argument("--embeddings-field",   default="full_embeddings")
    p.add_argument("--stratify-by",        default="m_flight")
    p.add_argument("--train-size",         type=float, default=0.85)
    p.add_argument("--val-size",           type=float, default=0.15)
    p.add_argument("--test-size",          type=float, default=0.15)
    # clustering
    p.add_argument("--target-cluster-size", type=int, default=10,
                   help="Target samples per cluster for adaptive n_clusters. "
                        "(default: 10)")
    # uniqueness
    p.add_argument("--k-uniq",             type=int,   default=5,
                   help="kNN neighbours for per-cluster uniqueness. (default: 5)")
    p.add_argument("--decay",              default="exponential",
                   choices=["exponential", "linear", "power"],
                   help="Decay family for uniqueness weights. (default: exponential)")
    p.add_argument("--decay-param",        type=float, default=0.5,
                   help="lambda (exponential) or p (power). (default: 0.5)")
    # diversity selection
    p.add_argument("--ball-radius",        type=float, default=0.5,
                   help="L2 ball radius for neighbour penalisation. (default: 0.5)")
    p.add_argument("--penalty",            type=float, default=0.7,
                   help="Score multiplier for penalised neighbours. (default: 0.7)")
    p.add_argument("--budget-mode",        default="uniform",
                   choices=["uniform", "proportional"],
                   help="Budget allocation across clusters. (default: uniform)")
    # misc
    p.add_argument("--output-json",        "-o", required=True)
    p.add_argument("--skip-tag",           action="store_true",
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

        tv_view  = dataset.match(F(stratify_by).is_in(non_test))
        ids_tv   = tv_view.values("id")
        strata   = tv_view.values(stratify_by)

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


# ── Decay weights ─────────────────────────────────────────────────────────────

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


# ── Adaptive n_clusters ───────────────────────────────────────────────────────

def adaptive_n_clusters(partition_size, total_samples,
                         target_cluster_size=10,
                         min_clusters=3,
                         min_points_per_cluster=2):
    n_samples = int(partition_size * total_samples)
    ideal     = n_samples // target_cluster_size
    if ideal >= min_points_per_cluster:
        return target_cluster_size
    return max(min_clusters, int(math.floor(math.sqrt(n_samples))))


# ════════════════════════════════════════════════════════════════════════════
#  INNER FUNCTION — pure numpy, no FiftyOne dependency
# ════════════════════════════════════════════════════════════════════════════

def _cluster_diverse_selection_from_arrays(
    embeddings_norm:   np.ndarray,          # (N, D) L2-normalised
    cluster_labels:    np.ndarray,          # (N,)  integer cluster assignments
    partition_size:    float,
    k_uniq:            int   = 5,
    decay:             str   = "exponential",
    decay_param:       float = 0.5,
    ball_radius:       float = 0.5,
    penalty:           float = 0.7,
    budget_mode:       str   = "uniform",
    seed:              int   = 42,
    verbose:           bool  = True,
) -> tuple[list, dict]:
    """
    Cluster-stratified diversity sampling — pure numpy inner function.

    Steps
    -----
    1. KMeans cluster assignments are supplied externally (cluster_labels).
    2. Per-cluster uniqueness: kNN within each cluster → weighted cosine distances.
    3. Per-cluster ball-radius selection: rank by uniqueness, penalise neighbours,
       greedily select until per-cluster budget is met.

    Parameters
    ----------
    embeddings_norm  : (N, D) L2-normalised embedding matrix
    cluster_labels   : (N,)  integer cluster label per sample
    partition_size   : fraction of N to select
    k_uniq           : kNN neighbours for uniqueness (within cluster)
    decay / decay_param : weight decay family for uniqueness
    ball_radius      : L2 distance threshold for neighbour penalisation
    penalty          : score multiplier for penalised neighbours
    budget_mode      : "uniform" | "proportional"
    seed             : random seed for tie-breaking
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
    n_clusters      = len(unique_clusters)
    cluster_sizes   = {int(c): int((cluster_labels == c).sum())
                       for c in unique_clusters}

    # ── Budget allocation ─────────────────────────────────────────────────
    if budget_mode == "uniform":
        base    = total_budget // n_clusters
        budgets = {int(c): base for c in unique_clusters}
        leftover = total_budget - base * n_clusters
        for c in sorted(unique_clusters,
                        key=lambda c: cluster_sizes[int(c)], reverse=True):
            if leftover <= 0:
                break
            budgets[int(c)] += 1
            leftover        -= 1

    elif budget_mode == "proportional":
        total_s = sum(cluster_sizes.values())
        budgets = {}
        allocated = 0
        for c in unique_clusters:
            b = max(1, int(round(total_budget * cluster_sizes[int(c)] / total_s)))
            budgets[int(c)] = b
            allocated += b
        drift = allocated - total_budget
        for c in sorted(unique_clusters,
                        key=lambda c: budgets[int(c)], reverse=(drift > 0)):
            if drift == 0:
                break
            if drift > 0 and budgets[int(c)] > 1:
                budgets[int(c)] -= 1; drift -= 1
            elif drift < 0:
                budgets[int(c)] += 1; drift += 1
    else:
        raise ValueError(f"budget_mode='{budget_mode}' not supported.")

    _log(f"N={N}  total_budget={total_budget}  "
         f"n_clusters={n_clusters}  budget_mode={budget_mode}")

    # ── Per-cluster selection ─────────────────────────────────────────────
    rng           = random.Random(seed)
    selected_idx  = []
    cluster_stats = {}

    for cluster_id in sorted(unique_clusters):
        cid          = int(cluster_id)
        cluster_idx  = np.where(cluster_labels == cid)[0]   # global indices
        n_c          = len(cluster_idx)
        budget_c     = min(budgets[cid], n_c)

        sub_embs     = embeddings_norm[cluster_idx]          # (n_c, D)

        # ── Per-cluster uniqueness ────────────────────────────────────────
        k_eff = min(k_uniq, n_c - 1)

        if k_eff < 1:
            # Single-sample cluster — uniqueness is trivially 1.0
            sub_scores = np.ones(n_c, dtype=np.float64)
        else:
            w = _decay_weights(k_eff, decay, decay_param)
            knn = NearestNeighbors(n_neighbors=k_eff + 1,
                                   metric="cosine", n_jobs=-1)
            knn.fit(sub_embs)
            dists, _ = knn.kneighbors(sub_embs)
            raw = (dists[:, 1:] * w).sum(axis=1)
            sub_scores = raw / raw.max() if raw.max() > 0 else raw

        # ── Ball-radius greedy selection ──────────────────────────────────
        tree         = cKDTree(sub_embs)
        working      = sub_scores.copy().astype(np.float64)
        visited      = set()
        chosen_local = []                                    # local indices

        order = np.argsort(working)[::-1]

        for local_idx in order:
            if len(chosen_local) >= budget_c:
                break
            if local_idx in visited:
                continue

            visited.add(local_idx)
            chosen_local.append(int(local_idx))

            neighbours = tree.query_ball_point(
                sub_embs[local_idx], ball_radius, return_sorted=True
            )
            to_penalise = [n for n in neighbours if n not in visited]
            visited.update(to_penalise)
            working[to_penalise] *= penalty

            # Re-rank after downweighting
            order = np.argsort(working)[::-1]

        # Fallback: fill if budget not met (tiny cluster, large ball_radius)
        if len(chosen_local) < budget_c:
            remaining = [i for i in range(n_c) if i not in chosen_local]
            rng.shuffle(remaining)
            chosen_local.extend(remaining[:budget_c - len(chosen_local)])

        # Map local → global indices
        chosen_global = [int(cluster_idx[i]) for i in chosen_local]
        selected_idx.extend(chosen_global)

        cluster_stats[cid] = {
            "n_in_cluster":    n_c,
            "budget":          budget_c,
            "n_selected":      len(chosen_global),
            "mean_uniqueness": float(sub_scores.mean()),
            "max_uniqueness":  float(sub_scores.max()),
        }

        _log(f"cluster {cid:>3}: "
             f"n={n_c:>4}  budget={budget_c:>3}  "
             f"selected={len(chosen_global):>3}  "
             f"mean_uniq={sub_scores.mean():.3f}")

    _log(f"→ total selected: {len(selected_idx)} / {total_budget} target")
    return selected_idx, cluster_stats


# ════════════════════════════════════════════════════════════════════════════
#  FIFTYONE WRAPPER — loads from fields, calls inner function
# ════════════════════════════════════════════════════════════════════════════

def compute_cluster_diverse_selection(
    dataset,
    embeddings_field:  str   = "full_embeddings",
    cluster_field:     str   = "cluster_label",
    uniqueness_field:  str   = None,   # optional — not used by inner fn
    partition_size:    float = 0.10,
    k_uniq:            int   = 5,
    decay:             str   = "exponential",
    decay_param:       float = 0.5,
    ball_radius:       float = 0.5,
    penalty:           float = 0.7,
    budget_mode:       str   = "uniform",
    seed:              int   = 42,
    verbose:           bool  = True,
) -> tuple[list, dict]:
    """
    FiftyOne wrapper around _cluster_diverse_selection_from_arrays.

    Loads embeddings and cluster labels from FiftyOne fields, runs the
    inner function, and returns selected sample IDs (strings).

    Parameters
    ----------
    dataset           : FiftyOne dataset or view
    embeddings_field  : field with L2-normalised embeddings
    cluster_field     : field with integer cluster labels
    uniqueness_field  : unused here (inner function recomputes per-cluster
                        uniqueness from scratch for correctness). Kept as
                        parameter for API consistency with other functions.
    partition_size    : fraction of dataset to select
    ...               : remaining params forwarded to inner function

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
    cluster_labels = []

    for sample in dataset.iter_samples(progress=verbose):
        emb   = sample.get_field(embeddings_field)
        clust = sample.get_field(cluster_field)
        if emb is None or clust is None:
            continue
        sample_ids.append(sample.id)
        emb_list.append(np.array(emb, dtype=np.float32))
        cluster_labels.append(int(clust))

    if not emb_list:
        raise ValueError(
            f"No samples found with both '{embeddings_field}' "
            f"and '{cluster_field}' set."
        )

    embeddings_norm = normalize(np.stack(emb_list), norm="l2")
    cluster_labels  = np.array(cluster_labels)
    N               = len(sample_ids)
    _log(f"Loaded {N} samples  dim={embeddings_norm.shape[1]}  "
         f"clusters={len(np.unique(cluster_labels))}")

    # Call pure-numpy inner function
    selected_idx, cluster_stats = _cluster_diverse_selection_from_arrays(
        embeddings_norm = embeddings_norm,
        cluster_labels  = cluster_labels,
        partition_size  = partition_size,
        k_uniq          = k_uniq,
        decay           = decay,
        decay_param     = decay_param,
        ball_radius     = ball_radius,
        penalty         = penalty,
        budget_mode     = budget_mode,
        seed            = seed,
        verbose         = verbose,
    )

    selected_ids = [sample_ids[i] for i in selected_idx]
    return selected_ids, cluster_stats


# ════════════════════════════════════════════════════════════════════════════
#  ACLR PIPELINE  (thin orchestrator — calls inner function directly)
# ════════════════════════════════════════════════════════════════════════════

def aclr_pipeline(
    embeddings_norm,
    sample_ids,
    partition_size,
    target_cluster_size = 10,
    k_uniq              = 5,
    decay               = "exponential",
    decay_param         = 0.5,
    ball_radius         = 0.5,
    penalty             = 0.7,
    budget_mode         = "uniform",
    seed                = 42,
):
    """
    Cluster-stratified ACLR for one partition size.

    Steps
    -----
    1. Adaptive KMeans — n_clusters scales with partition size.
    2. _cluster_diverse_selection_from_arrays — per-cluster uniqueness +
       ball-radius greedy selection.

    Returns
    -------
    selected_ids : list of sample ID strings
    cluster_stats: dict with per-cluster metadata
    """
    N          = len(embeddings_norm)
    n_clusters = adaptive_n_clusters(partition_size, N,
                                      target_cluster_size=target_cluster_size)

    print(f"    partition={partition_size:.0%}  "
          f"target={int(round(partition_size*N))}  "
          f"n_clusters={n_clusters}")

    # ── Step 1: KMeans ────────────────────────────────────────────────────
    km = KMeans(n_clusters=n_clusters, init="k-means++",
                n_init=10, random_state=seed)
    cluster_labels = km.fit_predict(embeddings_norm)

    # ── Step 2: cluster-stratified diversity selection ────────────────────
    selected_idx, cluster_stats = _cluster_diverse_selection_from_arrays(
        embeddings_norm = embeddings_norm,
        cluster_labels  = cluster_labels,
        partition_size  = partition_size,
        k_uniq          = k_uniq,
        decay           = decay,
        decay_param     = decay_param,
        ball_radius     = ball_radius,
        penalty         = penalty,
        budget_mode     = budget_mode,
        seed            = seed,
        verbose         = True,
    )

    selected_ids = [sample_ids[i] for i in selected_idx]
    print(f"    final selected: {len(selected_ids)}")
    return selected_ids, cluster_stats


# ── Random baseline ───────────────────────────────────────────────────────────

def random_baseline(sample_ids, partition_size, seed):
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

    # ── Stages 2–4: ACLR + random per seed ───────────────────────────────
    results = {}

    for seed in args.seeds:
        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")

        embeddings_norm, sample_ids = load_train_embeddings(
            dataset, seed, args.embeddings_field, verbose=False
        )
        N = len(sample_ids)
        print(f"  Train pool: {N} samples")

        results[str(seed)] = {}

        for partition in args.partitions:
            key = f"p{int(partition*100)}"
            print(f"\n  Partition {partition:.0%}  ({key})")
            results[str(seed)][key] = {}

            # ACLR
            print("  -- ACLR --")
            aclr_ids, stats = aclr_pipeline(
                embeddings_norm     = embeddings_norm,
                sample_ids          = sample_ids,
                partition_size      = partition,
                target_cluster_size = args.target_cluster_size,
                k_uniq              = args.k_uniq,
                decay               = args.decay,
                decay_param         = args.decay_param,
                ball_radius         = args.ball_radius,
                penalty             = args.penalty,
                budget_mode         = args.budget_mode,
                seed                = seed,
            )
            results[str(seed)][key]["aclr"]          = aclr_ids
           # results[str(seed)][key]["aclr_stats"]    = stats

            # Random baseline
            print("  -- Random --")
            rand_ids = random_baseline(sample_ids, partition, seed)
            results[str(seed)][key]["random"] = rand_ids

            print(f"  ACLR={len(aclr_ids)}  Random={len(rand_ids)}")

    # ── Save ──────────────────────────────────────────────────────────────
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {out}")

    print("\n── Summary ──────────────────────────────────────────────────")
    for seed in args.seeds:
        for partition in args.partitions:
            key = f"p{int(partition*100)}"
            na = len(results[str(seed)][key]["aclr"])
            nr = len(results[str(seed)][key]["random"])
            print(f"  SEED{seed}  {key:>4}  ACLR={na:>5}  Random={nr:>5}")


if __name__ == "__main__":
    main()