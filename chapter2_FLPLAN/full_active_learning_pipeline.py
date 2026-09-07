"""
run_aclr_pipeline.py
====================
Active Learning sample selection pipeline for the FLPLAN dataset.

Simplified from the original NC/WP dual-domain pipeline to a single-dataset
setting.  All NC/WP branching has been removed.

Pipeline stages
---------------
  1.  Tag split  (skippable with --skip-tag)
        Calls tag_train_test_seeded_split() to tag each sample with
        train_{seed} / val_{seed} / test_{seed}.
        Skipped if --skip-tag is passed (tags already exist).

  2.  Load embeddings
        Reads full_embeddings from the FiftyOne dataset for all train_{seed}
        tagged samples.

  3.  ACLR selection
        For each partition size:
          a) Weighted kNN uniqueness scoring  (geometric isolation)
          b) KMeans clustering                (representativeness)
          c) Stochastic softmax sampling from each pool
          d) Set union
          e) Ball-radius downweighting        (replaces random clipping)
        Returns a list of FiftyOne sample IDs for that partition.

  4.  Random baseline selection
        For each partition size, randomly sample without replacement from
        the same train pool (fixes the original random.choices bug).

  5.  Save results
        Writes a JSON file mapping
            {seed: {partition: {method: [sample_ids]}}}
        to --output-json.

Improvements over the original pipeline
----------------------------------------
  - No WP/NC split — single dataset throughout
  - random.sample (without replacement) replaces random.choices
  - Ball-radius downweighting replaces random clipping
  - Adaptive n_clusters based on partition size
  - --skip-tag flag to skip tagging when already done

Usage
-----
  # Full run (tag + ACLR + random)
  python run_aclr_pipeline.py \\
      --dataset FLPLAN \\
      --port 44123 \\
      --seeds 0 63 72 \\
      --partitions 0.05 0.10 0.25 0.50 0.75 \\
      --embeddings-field full_embeddings \\
      --output-json /share/home/e2406743/results/aclr_selections.json

  # Skip tagging (already done)
  python run_aclr_pipeline.py \\
      --dataset FLPLAN --port 44123 \\
      --skip-tag \\
      --output-json /share/home/e2406743/results/aclr_selections.json
"""

import os
import json
import math
import random
import argparse
from pathlib import Path
from copy import deepcopy

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from scipy.spatial import cKDTree


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="ACLR + random baseline selection pipeline for FLPLAN.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset",          "-d",  required=True)
    p.add_argument("--port",             default="44123")
    p.add_argument("--seeds",            nargs="+", type=int, default=[0, 63, 72])
    p.add_argument("--partitions",       nargs="+", type=float,
                   default=[0.05, 0.10, 0.25, 0.50, 0.75])
    p.add_argument("--embeddings-field", default="full_embeddings")
    p.add_argument("--stratify-by",      default="m_flight")
    p.add_argument("--train-size",       type=float, default=0.85)
    p.add_argument("--val-size",         type=float, default=0.15)
    p.add_argument("--test-size",        type=float, default=0.15)
    p.add_argument("--ratio-uniq",       type=float, default=0.5,
                   help="Fraction of budget from uniqueness pool. (default: 0.5)")
    p.add_argument("--temperature",      type=float, default=0.5,
                   help="KMeans softmax temperature. (default: 0.5)")
    p.add_argument("--temperature-uniq", type=float, default=0.5,
                   help="Uniqueness softmax temperature. (default: 0.5)")
    p.add_argument("--ball-radius",      type=float, default=0.5,
                   help="Ball radius for downweight deduplication. (default: 0.5)")
    p.add_argument("--penalty",          type=float, default=0.7,
                   help="Score multiplier for neighbours in ball. (default: 0.7)")
    p.add_argument("--target-cluster-size", type=int, default=10,
                   help="The ideal number of cluster to be considered "
                        "(default: 10)")
    p.add_argument("--output-json",      "-o", required=True,
                   help="Path to write selection results JSON.")
    p.add_argument("--skip-tag",         action="store_true",
                   help="Skip tagging — assume tags already exist in dataset.")
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

        shuffled    = rng.sample(available, len(available))
        test_flights = []
        test_count   = 0
        remaining_budget = 0

        for flight in shuffled:
            n = flight_counts[flight]
            contribution = n + remaining_budget
            if test_count + contribution <= target_test_max:
                test_flights.append(flight)
                test_count += n
                remaining_budget = 0
            else:
                remaining_budget += max(0, target_test_n - test_count)
            if test_count >= target_test_n:
                break

        _log(f"Test flights: {test_flights}  "
             f"({test_count} images = {test_count/total_images:.1%})")
        used_test_flights.extend(test_flights)

        # Tag test samples
        for flight in test_flights:
            flight_ids = dataset.match(F(stratify_by) == flight).values("id")
            dataset.select(flight_ids).tag_samples(f"test_{seed}")

        # Tag non-test as notin_TEST
        non_test_flights = [f for f in all_flights if f not in test_flights]
        for flight in non_test_flights:
            flight_ids = dataset.match(F(stratify_by) == flight).values("id")
            dataset.select(flight_ids).tag_samples(f"notin_TEST_{seed}")

        # Train/val split on remaining flights
        trainval_view    = dataset.match(F(stratify_by).is_in(non_test_flights))
        ids_trainval     = trainval_view.values("id")
        strata_trainval  = trainval_view.values(stratify_by)

        train_ids, val_ids = train_test_split(
            ids_trainval,
            test_size=val_size,
            stratify=strata_trainval,
            random_state=seed,
            shuffle=True,
        )

        dataset.select(train_ids).tag_samples(f"train_{seed}")
        dataset.select(val_ids).tag_samples(f"val_{seed}")

        _log(f"Tagged: train_{seed} ({len(train_ids)})  "
             f"val_{seed} ({len(val_ids)})  "
             f"test_{seed} ({test_count})")


# ── Embedding loader ──────────────────────────────────────────────────────────

def load_train_embeddings(dataset, seed, embeddings_field, verbose=True):
    """
    Load L2-normalised embeddings for all train_{seed} tagged samples.
    Returns (embeddings_norm, sample_ids).
    """
    print(f"\n  Loading embeddings for train_{seed} ...")
    embeddings = []
    sample_ids = []

    train_view = dataset.match_tags(f"train_{seed}")
    for sample in train_view.iter_samples(progress=verbose):
        emb = sample.get_field(embeddings_field)
        if emb is None:
            continue
        embeddings.append(np.array(emb, dtype=np.float32))
        sample_ids.append(sample.id)

    if not embeddings:
        raise ValueError(
            f"No embeddings found in field '{embeddings_field}' "
            f"for train_{seed} samples."
        )

    arr = np.stack(embeddings)
    arr_norm = normalize(arr, norm="l2")
    print(f"  Loaded {len(arr_norm)} embeddings, dim={arr_norm.shape[1]}")
    return arr_norm, sample_ids


# ── Uniqueness ────────────────────────────────────────────────────────────────

def compute_weighted_uniqueness(embeddings, 
                                k=5, 
                                decay="exponential",
                                decay_param=0.5):
    ranks   = np.arange(1, k + 1, dtype=np.float64)
    weights = np.exp(-decay_param * ranks)
    weights = weights / weights.sum()

    knn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=-1)
    knn.fit(embeddings)
    distances, _ = knn.kneighbors(embeddings)

    relevant_dists = distances[:, 1:]
    weighted_dists = (relevant_dists * weights).sum(axis=1)
    if weighted_dists.max() > 0:
        weighted_dists /= weighted_dists.max()
    return weighted_dists


# ── Stochastic samplers ───────────────────────────────────────────────────────

def get_stochastic_uniqueness_representatives(
    uniqueness_scores, 
    candidate_idx, 
    target_count,
    temperature=0.5, 
    seed=42
):
    rng = np.random.default_rng(seed)
    scores = uniqueness_scores[candidate_idx]
    weights = np.exp(scores / temperature)
    probs   = weights / weights.sum()
    n = min(target_count, len(candidate_idx))
    chosen = rng.choice(len(candidate_idx), size=n, replace=False, p=probs)
    return candidate_idx[chosen]


def get_diverse_stochastic_representatives(
    embeddings, labels, centroids, n_clusters,
    c_per_cluster, temperature=0.5, seed=42
):
    representative_indices = []
    rng = np.random.default_rng(seed)

    for i in range(n_clusters):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) == 0:
            continue
        dists   = np.linalg.norm(embeddings[cluster_indices] - centroids[i], axis=1)
        weights = np.exp(-dists / temperature)
        probs   = weights / weights.sum()
        n       = min(len(cluster_indices), c_per_cluster)
        chosen  = rng.choice(len(cluster_indices), size=n, replace=False, p=probs)
        representative_indices.extend(cluster_indices[chosen])

    return np.array(representative_indices)


# ── Ball-radius downweighting (replaces random clipping) ─────────────────────

def ball_radius_downweight(
    embeddings, combined_indices, scores,
    target_count, ball_radius=0.5, penalty=0.7, seed=42
):
    """
    Iterative greedy deduplication via ball-radius score downweighting.
    Replaces random clipping when len(combined_indices) > target_count.

    Starting from the highest-scoring sample, finds all neighbours within
    ball_radius and multiplies their scores by penalty.  Iterates until
    target_count samples are selected.

    This preserves the best samples from every region of the embedding space
    instead of randomly discarding them.

    Parameters
    ----------
    embeddings       : (N, D) L2-normalised embeddings for the FULL train pool
    combined_indices : array of indices into embeddings (the over-budget union)
    scores           : score per index in combined_indices (uniqueness or combined)
    target_count     : desired final number of samples
    ball_radius      : L2 radius for neighbour search
    penalty          : score multiplier for downweighted neighbours (< 1.0)
    seed             : random seed for tie-breaking

    Returns
    -------
    selected_indices : list of target_count indices into embeddings
    """
    if len(combined_indices) <= target_count:
        return list(combined_indices)

    rng = random.Random(seed)

    # Build KD-tree on the candidate subset only
    sub_embs = embeddings[combined_indices]
    tree     = cKDTree(sub_embs)

    working_scores = scores.copy().astype(np.float64)
    visited        = set()
    selected       = []

    # Rank by score descending
    ordered = np.argsort(working_scores)[::-1]

    for local_idx in ordered:
        if len(selected) >= target_count:
            break
        if local_idx in visited:
            continue

        visited.add(local_idx)
        selected.append(int(combined_indices[local_idx]))

        # Find neighbours within ball_radius and downweight them
        neighbours = tree.query_ball_point(
            sub_embs[local_idx], ball_radius, return_sorted=True
        )
        to_penalise = [n for n in neighbours if n not in visited]
        visited.update(to_penalise)
        working_scores[to_penalise] *= penalty

        # Re-sort after downweighting
        ordered = np.argsort(working_scores)[::-1]

    # If still short (all candidates were penalised), fill with remaining
    remaining = [int(combined_indices[i]) for i in range(len(combined_indices))
                 if combined_indices[i] not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[:target_count - len(selected)])

    return selected[:target_count]


# ── Adaptive n_clusters ───────────────────────────────────────────────────────

def adaptive_n_clusters(partition_size, 
                        total_samples,
                        target_cluster_size=10,
                        min_clusters=3, 
                        min_points_per_cluster=2):
    """
    COmpute the size of the cluster. First check if the given target cluster will contain
    at least min_points_per_cluster, if not, then returns a number of cluster smaller than the target.
    The max cluster is the target cluster size defined by the user
    """
    n_samples = int(partition_size * total_samples)
    adap_cluster = math.floor(math.sqrt(n_samples))
    ideal_target_cluster = n_samples // target_cluster_size
    if ideal_target_cluster >= min_points_per_cluster:
        return target_cluster_size
    else:
        n         = max(min_clusters, adap_cluster)
        return n


# ── ACLR pipeline ─────────────────────────────────────────────────────────────

def aclr_pipeline(
    embeddings_norm,
    sample_ids,
    partition_size,
    ratio_uniq=0.5,
    n_clusters=None,
    temperature=0.5,
    temperature_uniq=0.5,
    ball_radius=0.5,
    penalty=0.7,
    target_cluster_size=15,
    seed=42,
    buffer_percent=0.05,
):
    """
    Full ACLR selection for one partition size.

    Returns list of selected sample IDs (strings).
    """
    N             = len(embeddings_norm)
    target_total  = math.floor(partition_size * N)
    target_uniq   = math.floor(target_total * ratio_uniq)

    if n_clusters is None:
        n_clusters = adaptive_n_clusters(
            partition_size, 
            N,
            target_cluster_size=target_cluster_size,
        )

    print(f"    partition={partition_size:.0%}  target={target_total}  "
          f"uniq_budget={target_uniq}  k={n_clusters}")

    # ── 1. Uniqueness ──────────────────────────────────────────────────────
    k_uniq = min(3, int(np.sqrt(N)) - 1)
    uniqueness = compute_weighted_uniqueness(embeddings_norm, k=max(4, k_uniq))

    # Percentile gate: find threshold closest to target_uniq count
    percentiles = np.linspace(50, 99, 30)[::-1]
    best_pct    = 95.0
    best_diff   = 1e9
    for pct in percentiles:
        count = (uniqueness >= np.percentile(uniqueness, pct)).sum()
        diff  = abs(count - target_uniq)
        if diff < best_diff:
            best_diff = diff
            best_pct  = pct

    threshold     = np.percentile(uniqueness, best_pct)
    candidate_idx = np.where(uniqueness >= threshold)[0]

    uniq_indices = get_stochastic_uniqueness_representatives(
        uniqueness, 
        candidate_idx,
        target_count=target_uniq,
        temperature=temperature_uniq,
        seed=seed,
    )

    # ── 2. KMeans clustering ───────────────────────────────────────────────
    kmeans_budget     = (target_total - len(uniq_indices)) * (1 + buffer_percent)
    samples_per_cluster = max(1, math.ceil(kmeans_budget / n_clusters))

    km = KMeans(n_clusters=n_clusters, init="k-means++",
                n_init=10, random_state=seed)
    cluster_labels = km.fit_predict(embeddings_norm)
    centroids      = km.cluster_centers_

    km_indices = get_diverse_stochastic_representatives(
        embeddings_norm, 
        cluster_labels, 
        centroids,
        n_clusters, 
        samples_per_cluster,
        temperature=temperature, seed=seed,
    )

    # ── 3. Set union ───────────────────────────────────────────────────────
    combined = np.array(list(set(uniq_indices.tolist()) | set(km_indices.tolist())))
    overlap  = len(set(uniq_indices.tolist()) & set(km_indices.tolist()))
    print(f"    union={len(combined)}  overlap={overlap}  target={target_total}")

    # ── 4. Ball-radius downweighting instead of random clipping ───────────
    if len(combined) > target_total:
        # Score for the combined set = uniqueness + centerness proxy
        centerness = np.zeros(N)
        for i in range(n_clusters):
            cidx = np.where(cluster_labels == i)[0]
            if len(cidx) == 0:
                continue
            dists = np.linalg.norm(
                embeddings_norm[cidx] - centroids[i], axis=1
            )
            centerness[cidx] = 1 / (1 + dists)

        combined_scores = 0.5 * uniqueness[combined] + 0.5 * centerness[combined]

        selected = ball_radius_downweight(
            embeddings_norm, combined, combined_scores,
            target_count=target_total,
            ball_radius=ball_radius,
            penalty=penalty,
            seed=seed,
        )
    else:
        selected = combined.tolist()

    selected_ids = [sample_ids[i] for i in selected]
    print(f"    final selected: {len(selected_ids)}")
    return selected_ids


# ── Random baseline ───────────────────────────────────────────────────────────

def random_baseline(sample_ids, partition_size, seed):
    """
    Random sample WITHOUT replacement..
    """
    N      = len(sample_ids)
    target = math.floor(partition_size * N)
    rng    = random.Random(seed)
    return rng.sample(sample_ids, k=min(target, N))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    # CRITICAL: must be set before importing fiftyone
    os.environ["FIFTYONE_DATABASE_URI"] = f"mongodb://localhost:{args.port}"

    import fiftyone as fo
    from fiftyone import ViewField as F

    try:
        print(f"Connected to MongoDB at localhost:{args.port}. "
              f"Datasets: {fo.list_datasets()}")
    except Exception as e:
        print(f"ERROR: Could not connect.\n{e}")
        return

    assert args.dataset in fo.list_datasets(), \
        f"Dataset '{args.dataset}' not found."

    dataset = fo.load_dataset(args.dataset)
    print(f"Loaded '{args.dataset}' — {len(dataset)} samples.")

    # ── Stage 1: Tag split ────────────────────────────────────────────────
    if not args.skip_tag:
        print("\n[Stage 1] Tagging train/val/test splits ...")
        tag_train_test_seeded_split(
            dataset=dataset,
            fo=fo,
            F=F,
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
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")

        embeddings_norm, sample_ids = load_train_embeddings(
            dataset, seed, args.embeddings_field, verbose=False
        )
        N = len(sample_ids)
        print(f"  Train pool: {N} samples")

        results[str(seed)] = {}

        for partition in args.partitions:
            key = f"p{int(partition*100)}"
            print(f"\n  Partition {partition:.0%} ({key})")
            results[str(seed)][key] = {}

            # ACLR
            print(f"  -- ACLR --")
            aclr_ids = aclr_pipeline(
                embeddings_norm=embeddings_norm,
                sample_ids=sample_ids,
                partition_size=partition,
                ratio_uniq=args.ratio_uniq,
                temperature=args.temperature,
                temperature_uniq=args.temperature_uniq,
                ball_radius=args.ball_radius,
                penalty=args.penalty,
                target_cluster_size=args.target_cluster_size,
                seed=seed,
                buffer_percent=0.05,
            )
            results[str(seed)][key]["aclr"] = aclr_ids

            # Random baseline (without replacement)
            print(f"  -- Random --")
            rand_ids = random_baseline(sample_ids, partition, seed)
            results[str(seed)][key]["random"] = rand_ids

            print(f"  ACLR={len(aclr_ids)}  Random={len(rand_ids)}")

    # ── Save results ──────────────────────────────────────────────────────
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Summary
    print("\n── Summary ──────────────────────────────────────────────────")
    for seed in args.seeds:
        for partition in args.partitions:
            key = f"p{int(partition*100)}"
            n_aclr = len(results[str(seed)][key]["aclr"])
            n_rand = len(results[str(seed)][key]["random"])
            print(f"  SEED{seed}  {key:>4}  ACLR={n_aclr:>5}  Random={n_rand:>5}")


if __name__ == "__main__":
    main()