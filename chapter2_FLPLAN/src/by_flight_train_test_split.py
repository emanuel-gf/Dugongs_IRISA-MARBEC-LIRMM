"""
This script runs the partitioning of the dataset by a given matter. 
The first implementation is done at the Island level (flight mission) and it is instructed to split for a given ratio
stratified by the island or also called as flight mission. This provides that each flight mission holds a ratio of
70:15:15, spliting for both views (NC and WP)
"""


## init mongo db and fiftyone connection
import os

# Define the URI to point to your manual process
os.environ["FIFTYONE_DATABASE_URI"] = "mongodb://localhost:44123"

import fiftyone as fo

# Verify connection
print(fo.core.odm.database.get_db_conn()) 

from pathlib import Path
import random
import glob 
from fiftyone import ViewField as F
import pandas as pd
import numpy as np 
import math
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from typing import List, Dict, Optional
import matplotlib.gridspec as gridspec
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from umap import UMAP  
from sklearn.preprocessing import normalize
import numpy as np
import os
from pathlib import Path
from loguru import logger
import sys
from datetime import datetime

# LOGURU SETUP  — call once at startup; writes both to stderr and a dated file
def setup_logger(log_dir: str ="/share/home/e2406743/code/Dugongs_IRISA-MARBEC-LIRMM/logs_train_test_split/", 
                 run_name: str = "run"):
    """
    Configure loguru: coloured stderr + rotating file in log_dir.
    Returns the path of the log file so it can be passed to W&B.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{run_name}.log"
 
    logger.remove()                                         # drop default handler
    logger.add(sys.stderr, level="DEBUG", colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
    logger.add(log_file,   level="DEBUG", rotation="50 MB",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}")
 
    logger.info(f"Logger initialised → {log_file}")
    return log_file





import random
import numpy as np
from sklearn.model_selection import train_test_split
from fiftyone import ViewField as F


def tag_train_test_seeded_split(
    dataset,
    stratify_by:  str   = "m_flight",
    train_size:   float = 0.85,
    val_size:     float = 0.15,
    test_size:    float = 0.15,
    num_seeds:    int   = 1,
    test_buffer:  float = 0.05,
    verbose:      bool  = True,
):
    """
    Splits a FiftyOne dataset into train / val / test by tagging samples,
    stratified by a categorical field (e.g. flight mission).

    Entire flight missions are kept together — no flight is split across
    train and test.  The approach mirrors the original WP pipeline:

      1. Randomly select flight(s) as test candidates until the test size
         budget is met (with a carry-forward for small flights).
      2. Split remaining flights into train / val using stratified sampling.
      3. Tag all samples with  train_{seed} / val_{seed} / test_{seed}.
         Samples not selected for val are tagged  notin_val_{seed}.
         Samples not selected for test are tagged  notin_TEST_{seed}.

    Parameters
    ----------
    dataset      : FiftyOne dataset
    stratify_by  : sample field used for stratification (e.g. "m_flight")
    train_size   : fraction of non-test samples to use for training
    val_size     : fraction of non-test samples to use for validation
    test_size    : target fraction of full dataset to hold out for test
    num_seeds    : number of independent splits to generate
    test_buffer  : fuzzy tolerance on the test size target
                   (a flight is accepted if its cumulative ratio is within
                    test_size ± test_buffer)
    verbose      : print progress

    Tags written
    ------------
    train_{seed}, val_{seed}, test_{seed},
    notin_val_{seed}, notin_TEST_{seed}
    """

    def _log(msg):
        if verbose:
            print(f"  {msg}")

    # ── Build flight→count dict sorted ascending by size ─────────────────────
    flight_counts = dict(
        sorted(dataset.count_values(stratify_by).items(), key=lambda x: x[1])
    )
    all_flights  = list(flight_counts.keys())
    total_images = dataset.count()

    _log(f"Dataset: {total_images} images  |  "
         f"{len(all_flights)} flights via '{stratify_by}'")
    _log(f"Target split — train:{train_size:.0%}  "
         f"val:{val_size:.0%}  test:{test_size:.0%}")

    used_test_flights = []   # track across seeds to avoid reuse

    for seed in range(num_seeds):
        rng = random.Random(seed)
        _log(f"\n── Seed {seed} ──────────────────────────────")

        # ── Step 1: select test flights ───────────────────────────────────────
        target_test_n   = int(total_images * test_size)
        target_test_max = int(total_images * (test_size + test_buffer))

        # Prioritise unused flights; fall back to full pool if needed
        available = [f for f in all_flights if f not in used_test_flights]
        if not available:
            available = all_flights.copy()

        shuffled = rng.sample(available, len(available))

        test_flights   = []
        test_count     = 0
        remaining_budget = 0

        for flight in shuffled:
            n = flight_counts[flight]
            contribution = n + remaining_budget

            if test_count + contribution <= target_test_max:
                test_flights.append(flight)
                test_count += n
                remaining_budget = 0
            else:
                # flight is too large — carry budget and try next
                remaining_budget += max(0, target_test_n - test_count)

            if test_count >= target_test_n:
                break

        _log(f"Test flights : {test_flights}  "
             f"({test_count} images = {test_count/total_images:.1%})")
        used_test_flights.extend(test_flights)

        # ── Step 2: tag test samples ──────────────────────────────────────────
        for flight in test_flights:
            flight_ids = dataset.match(
                F(stratify_by) == flight
            ).values("id")

            dataset.select(flight_ids).tag_samples(f"test_{seed}")

            _log(f"  Tagged test: {flight}  ({len(flight_ids)} samples)")

        # Tag non-test flight samples as notin_TEST
        non_test_flights = [f for f in all_flights if f not in test_flights]
        for flight in non_test_flights:
            flight_ids = dataset.match(
                F(stratify_by) == flight
            ).values("id")
            dataset.select(flight_ids).tag_samples(f"notin_TEST_{seed}")

        # ── Step 3: train / val split on remaining flights ────────────────────
        trainval_view = dataset.match(
            F(stratify_by).is_in(non_test_flights)
        )
        ids_trainval   = trainval_view.values("id")
        strata_trainval = trainval_view.values(stratify_by)

        train_ids, val_ids = train_test_split(
            ids_trainval,
            test_size=val_size,
            stratify=strata_trainval,
            random_state=seed,
            shuffle=True,
        )

        _log(f"Train: {len(train_ids)}  Val: {len(val_ids)}")

        # ── Step 4: tag train / val ───────────────────────────────────────────
        dataset.select(train_ids).tag_samples(f"train_{seed}")
        dataset.select(val_ids).tag_samples(f"val_{seed}")

        # Tag val samples not selected as notin_val
        # (here all val_ids ARE selected — add notin_val for future subsampling)
        dataset.select(val_ids).tag_samples(f"val_{seed}")

        _log(f"Tagged: train_{seed} ({len(train_ids)})  "
             f"val_{seed} ({len(val_ids)})  "
             f"test_{seed} ({test_count})")

        # ── Summary ───────────────────────────────────────────────────────────
        if verbose:
            print(f"\n  Split summary for seed {seed}:")
            print(f"    train_{seed} : {len(train_ids):>5}  "
                  f"({len(train_ids)/total_images:.1%})")
            print(f"    val_{seed}   : {len(val_ids):>5}  "
                  f"({len(val_ids)/total_images:.1%})")
            print(f"    test_{seed}  : {test_count:>5}  "
                  f"({test_count/total_images:.1%})")

    print(f"\nDone. {num_seeds} seed(s) tagged.")
    

def suggest_candidates(
    dict_stratify:dict,
    used_flights,
    k_choices=(2, 3, 4),
    max_tries=100,
    min_test_size=0.1,
    test_buffer=0.05
):  
    #print(dict_stratify)
    total_size = np.array(list(dict_stratify.values())).reshape(-1).sum(axis=0)
    # prioritize unused flights
    unused = [k for k in dict_stratify if k not in used_flights]
    pool = unused if len(unused) >= min(k_choices) else list(dict_stratify.keys())

    for _ in range(max_tries):
        kk = min(random.choice(k_choices), len(pool))
        candidates = random.sample(pool, k=kk)

        candidates_sum = sum(dict_stratify[c] for c in candidates)
        ratio = (candidates_sum / total_size)

        if min_test_size <= ratio <= (min_test_size + test_buffer):
            logger.info(f"Found flight candidates: {candidates}")
            logger.info(f"Efective test size:{np.array([dict_stratify[cand] for cand in candidates]).sum()}")
            logger.info(f"Effective Ratio test size:{ratio}")
            return candidates

    return None

    
# check if they existed before 
def check_already_selected_candidates(
    old_list_candidates,
    new_candidates
    ):
    return sum(1 for flight in new_candidates if flight in old_list_candidates)


def tag_traintest_seeded_NC(dataset,
                            nc_flight_candidates:list,
                            nc_dict_stratify, 
                            seed_number:int, 
                            nc_val_size:float):
    """
    For a given random seed, find which flight mission should be used to test.
    Then, tag all these flights without subsetting the NC. 
    Split the remaining flights into train and val using a stratification given by flight mission.
    Tag everything.
    """
    
    dict_flights = nc_dict_stratify


    # GET THESE TEST ids
    ids_test_nc = dataset.match(F('stratify_key').is_in(nc_flight_candidates)).values('id')

    ## SELECT ALL OTHER FLIGHTS IN NC - to be TRAIN/VAL
    all_other_flights_NC = [flight for flight in list(dict_flights.keys()) if flight not in nc_flight_candidates]

    ## get ID and Strata
    ids_NC = dataset.match(F('stratify_key').is_in(all_other_flights_NC)).values('id')
    strata_NC = dataset.match(F('stratify_key').is_in(all_other_flights_NC)).values('stratify_key')


    #split
    train_ids, val_ids = train_test_split(
                    ids_NC, 
                    test_size= nc_val_size, 
                    stratify=strata_NC,
                    shuffle=True, 
                    random_state=seed_number
                )
    

    ## TAG TRAIN/VAL
    dataset.select(train_ids).tag_samples(f"train_{str(seed_number)}")
    dataset.select(val_ids).tag_samples(f"val_{str(seed_number)}")

    ## tag TEST NC
    dataset.select(ids_test_nc).tag_samples(f"test_{str(seed_number)}")
    logger.success(f"New Caledonia tagged! Seed:{seed_number}")
    logger.success(f"Length Train:{len(train_ids)}")
    logger.success(f"Length Val:{len(val_ids)}")
    logger.success(f"Length Test: {len(ids_test_nc)}")

    return seed_number
        


def return_list_filepath_train_test_val(seed_number):
    """
    Returns: train_wp, test_wp, val_wp, train_nc, test_nc, val_nc list of filepaths. 
    """
    train_filepath = wp_view.match_tags(f"train_{seed_number}").values("filepath")
    test_filepath = wp_view.match_tags(f"test_{seed_number}").values("filepath")
    val_filepath = wp_view.match_tags(f"val_{seed_number}").values("filepath")
    return train_filepath, test_filepath, val_filepath


def build_filepath_df(train_filepath, test_filepath, val_filepath):
    """
    Cstruct a simple df from the given entries.

    Returns: df
    """

    df = pd.DataFrame({
        "train": pd.Series(train_wp_filepath),
        "test": pd.Series(test_wp_filepath),
        "val": pd.Series(val_wp_filepath),
    })

    return df

## FUNCTIONS TO LOAD IT BACK 
def get_seed_from_filepath(csv_file):
    path = Path(csv_file).stem
    return path.split('_')[-1]


def return_list_from_csv(csv_file):
    dff = pd.read_csv(csv_file)
    wp_train_list = dff['train'].dropna().values
    wp_test_list = dff['test'].dropna().values
    wp_val_list = dff['val'].dropna().values
    return wp_train_list, wp_test_list, wp_val_list


## RANDOM PARTITION
def random_choice_train_list(train_list,
                             seed,
                             subset_size,
                             partitions:list=[0.05, 0.1,0.25,0.5,0.75,1.0]
                             ):
    """
    Select a randomly the paths files from the list

    Returns:
    Each key of the partition name where the values are a list of filepaths.
    """
    length = len(train_list)

    if subset_size is not None:
        length  = subset_size

    ## seed 
    random.seed(seed)

    dict_out ={}
    for p in partitions:
        num_images = int(math.floor(length*p))
        logger.info(f"Partition :{p} | size train: {num_images}")
        dict_out[f"partition_{str(int(p*100))}"] = random.choices(train_list, k=num_images)
    
    return dict_out


## MAPDICT
def get_files_by_stem(filepath_stem, patch_folder):
    dict_out = {}
    foolder_meta = os.path.join(patch_folder, 'metadata')
    list_meta = list(glob.glob(os.path.join(foolder_meta, f'{filepath_stem}__*.json')))
    foolder_meta = os.path.join(patch_folder, 'images')
    list_images = list(glob.glob(os.path.join(foolder_meta, f'{filepath_stem}__*.jpg')))
    foolder_meta = os.path.join(patch_folder, 'labels')
    list_labels = list(glob.glob(os.path.join(foolder_meta, f'{filepath_stem}__*.txt')))
    dict_out['metadata'] = list_meta
    dict_out['label'] = list_labels
    dict_out['images'] = list_images
    return dict_out


## use the map dict to create the final list of filepaths regarding the patches 
def mapdict_patches_filepath(list_paths, patch_folder):
    """
    Map each filepath into three tiles-filepaths respectively to images, labels and metadata. 
    """
    dict_map_filepath = {}
    for path in list_paths:
        stem = Path(path).stem
        dict_map_filepath[stem] = get_files_by_stem(stem, patch_folder)

    ## flat dict
    filepath_all_images   = [f for d in dict_map_filepath.values() for f in d.get('images', [])]
    filepath_all_labels   = [f for d in dict_map_filepath.values() for f in d.get('label', [])]
    filepath_all_metadata = [f for d in dict_map_filepath.values() for f in d.get('metadata', [])]

    ## -------------
    logger.info(f'images:{len(filepath_all_images)}')
    logger.info(f"labels:{len(filepath_all_labels)}")
    logger.info(f"metadata:{len(filepath_all_metadata)}")

    return filepath_all_images, filepath_all_labels, filepath_all_metadata


## ----------------------------------------------------
## ACTIVE LEARNING
## ---------------------------------------------------------------------------------
def compute_weighted_uniqueness(embeddings, 
                                weights=[0.9, 0.6, 0.3, 0.1]):
    """
    Computes the Uniqueness score based on a weighted nearest neighboors analysis.
    """
    k = len(weights)
    
    # knn
    #  look for k+1 because the 1st neighbor is always the point itself (dist=0)
    knn = NearestNeighbors(n_neighbors=k + 1,
                            metric='cosine')
    knn.fit(embeddings)
    
    # extract distances
    # distances shape: (num_samples, k+1)
    distances, _ = knn.kneighbors(embeddings)
    
    # apply the Weights [0.6, 0.3, 0.1]
    # here we slice dists[:, 1:] to ignore the 0-distance to self
    relevant_dists = distances[:, 1:]
    
    # weighted mean
    weighted_dists = np.mean(relevant_dists * weights,
                            axis=1
                            )
    
    # normalize to [0, 1]
    if weighted_dists.max() > 0:
        weighted_dists /= weighted_dists.max()
        
    return weighted_dists


def get_diverse_stochastic_representatives(embeddings, 
                                           labels, 
                                           centroids, 
                                           n_clusters, 
                                           c_per_cluster, 
                                           temperature=0.5, 
                                           seed=42):
    """
    Selects representative samples from clusters using a temperature-scaled 
    Softmax distribution to balance representation and diversity.

    Instead of a 'greedy' selection (always picking the closest sample), this 
    function assigns a probability to each sample in a cluster based on its 
    proximity to the centroid. A 'Temperature' parameter controls the randomness.

    Args:
        embeddings (np.ndarray): Normalized feature vectors (N, D).
        labels (np.ndarray): Cluster assignment for each sample (N,).
        centroids (np.ndarray): Coordinates of the cluster centers (K, D).
        n_clusters (int): Number of clusters (K).
        c_per_cluster (int): Number of samples to select from each cluster.
        temperature (float): Controls selection strictly. 
            - T -> 0: Approaches 'greedy' selection (closest to centroid).
            - T -> 1: Follows the natural distribution of distances.
            - T -> infinity: Approaches 'random' uniform sampling.
        seed (int): Random seed for reproducibility.

    Returns:
        np.ndarray: Indices of the selected representative samples.
    """
    representative_indices = []
    rng = np.random.default_rng(seed)

    for i in range(n_clusters):
        # 1. Isolate indices of samples belonging to the current cluster
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) == 0:
            continue
        
        # 2. Calculate L2 distances from the Centroid to all points in the cluster
        # Even with Cosine metric, Euclidean distance on normalized vectors works perfectly
        dists = np.linalg.norm(embeddings[cluster_indices] - centroids[i], axis=1)
        
        # 3. Compute Softmax-based probabilities
        # We use (-distances) because smaller distances should have higher probability.
        # Scaling by 'temperature' stretches or flattens the probability distribution.
        weights = np.exp(-dists / temperature)
        probs = weights / np.sum(weights)
        
        # 4. Stochastic Sampling
        # size: Number of samples to pick (cannot exceed cluster size)
        # replace=False: Ensures we pick unique images
        n_to_pick = min(len(cluster_indices), c_per_cluster)
        chosen_in_cluster = rng.choice(
            len(cluster_indices), 
            size=n_to_pick, 
            replace=False, 
            p=probs
        )
        
        # Map local cluster indices back to original dataset indices
        representative_indices.extend(cluster_indices[chosen_in_cluster])
    
    return np.array(representative_indices)


def selector_num_uniqueness(partition_size:float, 
                            wp_train_length:int , 
                            uniqueness:np.array,
                            ratio_cluster_uniqueness:float = 0.5
                            ):
    """
    Find the number of uniqueness subset to compound the best subset for the given training partition.
    It looks first at the number of images which should compose the training partition.
    As an example, if partition is 10% and the train size is 1000 images, given the ratio as 50% which means 
    the same amount compounding the cluster and the uniqueness, 500 images coming from cluster and 500 images coming 
    from uniqueness. 

    Args:
        ratio_cluster_uniqueness: float. The ratio of uniqueness:cluster If 1:1 so 0.5. If 1:2 so 0.33, if 1:3 so 0.25. 
    """
    import math
    len_uniq_images = math.floor(partition_size*wp_train_length*ratio_cluster_uniqueness)
    logger.info(f"Length of the uniqueness image size list for the given partition:{len_uniq_images}")

    # loop through this space checking if the amount is within the fuzzy_border
    diff_list = []
    percentiles = np.array([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.90,0.92,0.93,0.94,
                                0.95,0.96,0.97,0.98,0.99])[::-1] #reverse order
    for percentile in percentiles:
        num_images_percent = (uniqueness >np.percentile(uniqueness,percentile*100)).sum()
        diff = np.abs(len_uniq_images-num_images_percent)
        diff_list.append(diff)

    #print(np.argmin(np.array(diff_list)))
    perc_out = percentiles[np.argmin(np.array(diff_list))]
    logger.info(f"Percentile choosen:{perc_out*100}")
    logger.info(f"number of images:{(uniqueness >np.percentile(uniqueness,perc_out*100)).sum()}")
    
    return perc_out*100


def plot_uniqueness_to_ax(ax, uniqueness_scores):
    ax.hist(uniqueness_scores, bins=50, color='skyblue', edgecolor='black', alpha=0.8)
    ax.axvline(np.mean(uniqueness_scores), color='red', linestyle='dashed', label='Mean')
    ax.axvline(np.percentile(uniqueness_scores, q=75), color='green', linestyle='dashed', label='P.75')
    ax.axvline(np.percentile(uniqueness_scores, q=90), color='gray', linestyle='dashed', label='P.90')
    ax.axvline(np.percentile(uniqueness_scores, q=95), color='yellow', linestyle='dashed', label='P.95')
    ax.axvline(np.percentile(uniqueness_scores, q=98), color='purple', linestyle='dashed', label='P.98')
    ax.set_title("Uniqueness Distribution (Outlier Detection)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Uniqueness Score")
    ax.set_ylabel("Frequency")
    ax.legend()


def plot_landscape_to_axes(axes, coords_list, labels, uniq_idx, km_idx, n_clusters):
    names = ["PCA", "UMAP", "t-SNE"]
    cmap  = plt.cm.get_cmap("tab20", n_clusters)

    for i, ax in enumerate(axes):
        # Background: All potential candidates
        ax.scatter(coords_list[i][:, 0], 
                   coords_list[i][:, 1], 
                   c=labels, 
                   cmap=cmap, 
                   vmin=0, 
                   vmax=n_clusters-1,
                   s=8, 
                   alpha=0.2, 
                   zorder=1
                   )
        
        
        # 1. Cluster Representatives (The "Diversity" picks)
        ax.scatter(coords_list[i][km_idx, 0], coords_list[i][km_idx, 1], 
                    c=labels[km_idx], cmap=cmap, vmin=0, vmax=n_clusters-1,
                   #c='cyan', 
                   marker='o', s=35, edgecolor='black', linewidth=0.5, zorder=3)
        
        # 2. Uniqueness Outliers (The "Informative" picks)
        ax.scatter(coords_list[i][uniq_idx, 0], coords_list[i][uniq_idx, 1], 
                   c='magenta', 
                   marker='*', s=80, alpha=0.9, edgecolor='black', linewidth=0.5,
                   zorder=4)
        
        ax.set_title(names[i], fontsize=12)
        ax.set_xticks([]); ax.set_yticks([]) # Clean look for manifolds
        ax.grid(True, linestyle='--', alpha=0.2)


def save_consolidated_active_learning_report(uniqueness_scores, 
                                            coords_list, 
                                            labels, 
                                            uniq_idx, 
                                            km_idx, 
                                            n_clusters,
                                            partition_size:float,
                                            output_dir_image:str,
                                            filename="AL_Report.png"):
    
    # Calculate counts for the report header
    n_uniq = len(uniq_idx)
    n_km = len(km_idx)
    n_total = len(set(uniq_idx) | set(km_idx)) # Unique union count

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1.4]) 

    # --- TOP ROW: Distribution ---
    ax_top = fig.add_subplot(gs[0, :]) 
    plot_uniqueness_to_ax(ax_top, uniqueness_scores)

    # --- BOTTOM ROW: Projections ---
    axes_bottom = [fig.add_subplot(gs[1, i]) for i in range(3)]
    plot_landscape_to_axes(axes_bottom, coords_list, labels, uniq_idx, km_idx, n_clusters)

    # --- HEADER & METADATA ---
    fig.suptitle("Active Learning Selection Pipeline Report", fontsize=24, fontweight='bold', y=0.98)
    
    # Text box with selection statistics
    stats_text = (
        f"Selection Summary:\n"
        f"------------------\n"
        f"Partition on Run: {partition_size*100}%\n"
        f"Total Partition Size: {n_total} samples\n"
        f"K-Means (Diversity): {n_km} samples (●)\n"
        f"Uniqueness (Outliers): {n_uniq} samples (★)"
    )
    fig.text(0.02, 0.93, stats_text, fontsize=14, family='monospace', 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=1'))

    # --- CUSTOM LEGEND ---
    # Create manual legend elements
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='K-Means (Cluster Representatives)',
               markerfacecolor='gray', markeredgecolor='black', markersize=10),
        Line2D([0], [0], marker='*', color='w', label='Uniqueness (High-Density Outliers)',
               markerfacecolor='gray', markeredgecolor='black', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='Unselected Manifold',
               markerfacecolor='gray', alpha=0.3, markersize=8)
    ]
    
    # Place legend centrally below the manifolds
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=12, frameon=True, bbox_to_anchor=(0.5, 0.05))

    plt.tight_layout(rect=[0, 0.08, 1, 0.92]) 
    os.makedirs(Path(output_dir_image), exist_ok=True)
    plt.savefig(os.path.join(output_dir_image,filename), dpi=300, bbox_inches='tight')
    plt.close(fig)
    #plt.show()
    logger.success(f"Final report generated: {filename}")


def plot_consolidated_landscape(pca, 
                                umap, 
                                tsne, 
                                uniq_idx, 
                                km_idx, 
                                title):
    """
    Visualizes the entire manifold with two distinct colors for selection types.
    """
    fig, axes = plt.subplots(ncols=3, figsize=(18, 6))
    coords = [pca, umap, tsne]
    names = ["PCA", "UMAP", "t-SNE"]
    
    for i, ax in enumerate(axes):
        # Background
        ax.scatter(coords[i][:, 0], coords[i][:, 1], c='lightgrey', s=10, alpha=0.3)
        
        # Plot K-Means representatives (Cluster structure)
        ax.scatter(coords[i][km_idx, 0], coords[i][km_idx, 1], 
                   c='cyan', marker='o', s=40, edgecolor='black', label='Cluster Reps')
        
        # Plot Uniqueness representatives (Outliers)
        ax.scatter(coords[i][uniq_idx, 0], coords[i][uniq_idx, 1], 
                   c='magenta', marker='*', s=70, edgecolor='black', label='Uniqueness Outliers')
        
        ax.set_title(names[i])
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def get_stochastic_uniqueness_representatives(
    uniqueness_scores: np.ndarray,
    candidate_idx: np.ndarray,
    target_count: int,
    temperature_uniqueness: float = 0.5,
    seed: int = 42
) -> np.ndarray:
    """
    Stochastic sampling from the upper-tail uniqueness pool.
    
    Unlike a hard threshold selection, this applies a temperature-scaled
    softmax over the candidate pool so higher-uniqueness samples are 
    preferred but not deterministically chosen — spreading the selection
    across the uniqueness manifold rather than clustering at the tip.

    Args:
        uniqueness_scores: Full uniqueness array, shape (N,).
        candidate_idx: Indices of the upper-tail pool (from percentile gate).
        target_count: Number of samples to select.
        temperature_uniqueness: Controls spread within the upper tail.
            - Low  (0.05): near-deterministic, picks highest scores → re-clusters.
            - Med  (0.3-0.5): biased but spread, recommended.
            - High (2.0+): near-uniform over upper tail.
        seed: Random seed for reproducibility.

    Returns:
        np.ndarray: Selected indices into the original embeddings array.
    """
    rng = np.random.default_rng(seed)

    candidate_scores = uniqueness_scores[candidate_idx]

    # Softmax with +score (higher uniqueness = higher prob, opposite of kmeans distance)
    weights = np.exp(candidate_scores / temperature_uniqueness)
    probs   = weights / weights.sum()

    n_to_pick = min(target_count, len(candidate_idx))
    chosen    = rng.choice(len(candidate_idx), size=n_to_pick, replace=False, p=probs)

    return candidate_idx[chosen]


def active_learning_pipeline(
    embeddings_norm,
    partition_size: float,
    filename_image:str,
    output_dir_image:str ,
    subset_size:int = None,
    ratio_cluster_uniqueness: float = 0.5,
    n_clusters: int = 10,
    temperature: float = 0.5,
    temperature_uniq: float = 0.5,
    seed_number: int = 42,
    buffer_percent:float = 0.05
    ):
    """
    Active Learning Pipeline:
    1. Computes Weighted Uniqueness Scores.
    2. Determines optimal threshold to meet partition size.
    3. Selects 'Outlier' samples via Uniqueness.
    4. Selects 'Representative' samples via Stochastic K-Means.
    5. Visualizes the selection landscape.
    
    Returns:
        combined_indices (list): Final list of indices for FiftyOne selection.
    """
    total_samples = len(embeddings_norm)
    if subset_size is not None:
        total_samples = subset_size
    target_total_count = math.floor(partition_size * total_samples)
    
    logger.info(f"--- Starting Pipeline (Target Subset Size: {target_total_count}) ---")

    # 1. Compute Uniqueness
    uniqueness_scores = compute_weighted_uniqueness(embeddings_norm)
    #plot_uniqueness_distribution(uniqueness_scores)

    # 2. Determine Number of Uniqueness vs. Clustering images
    # budget_uniqueness = total_target * ratio (e.g. 500 = 1000 * 0.5)
    target_uniq_count = math.floor(target_total_count * ratio_cluster_uniqueness)
    
    # Use selector to find the best percentile threshold
    chosen_percentile = selector_num_uniqueness(
        partition_size=partition_size,
        wp_train_length=total_samples,
        uniqueness=uniqueness_scores,
        ratio_cluster_uniqueness=ratio_cluster_uniqueness
    )
    
    # Extract Uniqueness Indices
    threshold_value = np.percentile(uniqueness_scores, chosen_percentile)
    candidate_idx   = np.where(uniqueness_scores >= threshold_value)[0]

    # OLD
    #uniqueness_indices = np.where(uniqueness_scores >= threshold_value)[0]
    # NEW
    # Stochastic sampling within upper-tail pool
    uniqueness_indices = get_stochastic_uniqueness_representatives(
        uniqueness_scores,
        candidate_idx,
        target_count=target_uniq_count,
        temperature_uniqueness=temperature_uniq,
        seed=seed_number
    )
    # 3. Determine remaining budget for K-Means
    # We subtract what we actually got from uniqueness to fill the rest with clusters
    # we add a buffer here to avoid that this size is smaller than the target size
    kmeans_budget = (target_total_count - len(uniqueness_indices))*(1 + buffer_percent)
    samples_per_cluster = max(1, math.ceil(kmeans_budget / n_clusters))
    
    logger.info(f"Budget Allocation: Uniqueness={len(uniqueness_indices)}, KMeans={kmeans_budget}")

    # 4. Run K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, 
                    init='k-means++', 
                    n_init=10, 
                    random_state=seed_number)
    cluster_labels = kmeans.fit_predict(embeddings_norm)
    centroids = kmeans.cluster_centers_

    # 5. Stochastic K-Means Selection
    kmeans_indices = get_diverse_stochastic_representatives(
        embeddings_norm, 
        cluster_labels, 
        centroids, 
        n_clusters, # number of clusters 
        samples_per_cluster, # how many samples per cluster
        temperature=temperature,
        seed=seed_number
    )
    
    # 6. Consolidate Indices
    ## IMPORTANT - IF CLAUDE, READ THIS
    ## the combined indices that are being return from the function are being clip
    ## randomly to fit the max number of samples for the partition. 
    # E.g subset=1000, train_size=850, partition=0.1 -> return size of 85
    # BUT the active learning ideal has a bit more sample since each cluster has a criterion
    # of how many sample per clusters and  buffer percent, so the clip is done randomly
    # which may affect that some good samples from important clusters are being cut-off 
    combined_indices = list(set(uniqueness_indices) | set(kmeans_indices))
    
    overlap = len(set(uniqueness_indices) & set(kmeans_indices))
    logger.info(f"Overlap: {overlap} | Union size: {len(combined_indices)} | Target: {target_total_count}")

    if len(combined_indices) > target_total_count:
        rng = random.Random(seed_number)
        combined_indices = rng.sample(combined_indices, 
                                      k=target_total_count)
        logger.success(f"Clipped to {target_total_count} via random sampling")



    # Plot
    logger.info("Computing Projections for final visualization...")
    pca_coords = PCA(n_components=2, random_state=seed_number).fit_transform(embeddings_norm)
    umap_coords = UMAP(n_components=2, random_state=seed_number, n_jobs=1 ).fit_transform(embeddings_norm)
    tsne_coords = TSNE(n_components=2, random_state=seed_number).fit_transform(embeddings_norm)


    coords_list = [pca_coords, umap_coords, tsne_coords]

    save_consolidated_active_learning_report(
        uniqueness_scores, 
        coords_list, 
        cluster_labels, 
        uniqueness_indices, 
        kmeans_indices,
        n_clusters=n_clusters,
        partition_size = partition_size,
        filename= filename_image,
        output_dir_image = output_dir_image
    )
    return combined_indices




## --------------------------
def argparse():
    parse = ArgumentParser(description='Train Test and Val split')
    parse.add_argument('--dataset', default='dugong')
    parse.add_argument('--subset-size', default=750, help='Size of the subset to be selected. Exclusively for WP region, how much should be the subset.')
    parse.add_argument('--train-size', type=float,default=0.8)
    parse.add_argument('--test-size', type=float,default=0.15)
    parse.add_argument('--val-size', type=float,default=0.20)
    
    parse.add_argument('--stratify-key',type=str,default='stratify_key',help="The field name that contains the strategy" \
                                                                            "for stratification of the partitioning.")
    parse.add_argument('--output-folder',type=str,default="/share/home/e2406743/dataset/df_filepaths",
                       help="folder where to store the csv paths generated.")
    parse.add_argument('--patch-folder',type=str, default="/share/home/e2406743/dataset/exported_img/seed_42",
                       help='Folder where the images tiles are located.')
    parse.add_argument('--output-img-dir', type=str,default='/share/home/e2406743/code/Dugongs_IRISA-MARBEC-LIRMM/active_learning_images',
                        help='Where to store the plot of Active Learning selection')
    parse.add_argument('--num-clusters', type=int, default=13,
                                                                help='Num of clusters to cluster the dataset.')
    parse.add_argument('--ratio', type=float, default=0.25,
                       help='Ratio of uniqueness over clusters. If 0.5 so 1:1, if 0.33 so 1:2, which means two images of cluster for each one image of uniqueness.')
    parse.add_argument('--temperature', type=float,default=0.5, help= 'temperature-scaled Softmax distribution for selecting the samples closer to the cluster centroid.')
    return parse.parse_args()

def main():
    from datetime import datetime 
    datetime = datetime.now().strftime('%m%d_%H%M')
    
    args = argparse()
    logger.info(args)

    dataset_mongodb = args.dataset
    assert dataset_mongodb in fo.list_datasets(), f"Dataset not valid, should be one of these:{fo.list_dataset()}"
    assert os.path.isdir(args.output_folder), os.makedirs(args.output_folder, exist_ok=True)
    assert os.path.isdir(args.patch_folder), f"Patch folder does not exist"
    assert  os.path.isdir(args.output_img_dir), os.makedirs(args.output_img_dir, exist_ok=True)
    

    # set up logger
    run_name = f"{datetime}"
    setup_logger(run_name=run_name)
    
    ## Load dataset and views from mongodb 
    dataset = fo.load_dataset(dataset_mongodb)

    ## Unified strategy
    dict_strata = dataset.count_values('m_flights')
    dict_strata = dict(sorted(dict_strata.items()))
    size_dataset = dataset.count()

    PARTITIONS_ = np.array([0.01, 0.02, 0.03, 0.05,0.07, 0.1, 0.2, 0.25, 0.5, 1.0])
    
    partitions = PARTITIONS_*size_dataset

    seed_number_list = np.unique([k.split('_')[-1] for k in list(dataset.count_values('tags').keys())]).astype('int')
    csv_filename_list = []

    ## RUN ALL GIVEN SEEDS AND CREATES A CSV WITH THE PATHS REGARDING THE FULL IMAGE
    for ss in seed_number_list:
        logger.info(f"Running seed:{ss}")
        (train_filepath, test_filepath, val_filepath) = return_list_filepath_train_test_val(
            ss,
        )

        df_seed = build_filepath_df(
            train_filepath, 
            test_filepath, 
            val_filepath, 
        )

        ## save it keeping 
        output_filename = f"df_train_test_split_filepath_{str(ss)}.csv"
        logger.success(f"saving file:{output_filename}")
        output_folder = args.output_folder
        logger.success(f"saving at:{os.path.join(output_folder,output_filename)}")
        df_seed.to_csv(os.path.join(output_folder, output_filename))
        logger.success('done!')
        csv_filename_list.append(output_filename)
    
    ## -------------------------------------------------------------------------------------
    ## SECOND PART ---------------------------------------
    ## -------------------------------------------------------------
    logger.info(f"generating the fraction partition of train subsets for:{csv_filename_list}")

    ## LOAD IT BACK AND GENERATE THE MATCHING FOR PARTITIONING THE WP_TRAIN SET IN FRACTIONS
    ## do in a loop
    for csv_file in csv_filename_list:
        (train_list, test_list, val_list) = return_list_from_csv(os.path.join(args.output_folder,csv_file))
        
        ## get seed
        seed_number = int(get_seed_from_filepath(csv_file))
        logger.info(f"Running seed:{seed_number}")
        random.seed(seed_number)

        ##
        ## ---------------------------- RANDOM SELECTION --------------------------------------
        ## create a dict containing the filepath_stem with the keys containig the filepath for images, labels and metadata
        dictt_random_choice = random_choice_train_list(train_list = train_list,
                                            seed = seed_number,
                                            subset_size = size_dataset,
                                            partitions = PARTITIONS_
                                        )
        

        ## LOOP into each dictt key and return a full dataset
        ## return for each partition the paths associated for images, labels and metadata.

        new_dict = dictt_random_choice.copy()
        output_dict_partitions = dict()
        ## each key is a full filepath
        for key in new_dict.keys():
            logger.info(f"Running key:{key}")
            ## retriveves for each partition the associated patches, labels, metadata  - filepath 
            list_images, list_labels, list_metadata = mapdict_patches_filepath(dictt_random_choice[key],
                                                                                args.patch_folder)
            output_dict_partitions[key] = {'images':list_images , 
                                           'labels':list_labels, 
                                           'metadata': list_metadata}

        ## ----------- ACTIVE LEARNING ------------------------------------------------------------
        # loop into each partition, excluding partitions over 50%
        logger.info("Active Learning")

        # retrieve the tagged by the seed
        train_wp = dataset.match(F('tags').contains(f"train_{seed_number}"))
        assert len(train_wp)!=0, f"Error on finding the tagged train dataset for seed: {seed_number}"

        ## get the embeddings relative to the train set.
        train_emb = train_wp.values('full_embeddings')
        train_emb_norm = normalize(train_emb) #normalize 

        wp_train_ids = train_wp.values('filepath')
        logger.info(f"Active Pipeline,full train dataset size:{len(wp_train_ids)}")
        dict_actlr_full_filepath = {}
        output_actlr_tiles_filepath = {}
        for partition in PARTITIONS_:
            if partition > 0.75:
                pass
            else:
                logger.info(f"Running partition:{partition}")

                ## # Run the consolidate pipeline
                final_al_indices = active_learning_pipeline(
                                        train_emb_norm,
                                        subset_size = size_dataset,
                                        partition_size= partition,            # partition
                                        ratio_cluster_uniqueness= args.ratio,   # 1:2 66% cluster and 33% uniqueness
                                        n_clusters= args.num_clusters,
                                        seed_number= seed_number,
                                        temperature = args.temperature,
                                        filename_image = f'AC_LR_subset{str(wp_subset_size)}_seed{str(seed_number)}_partition_{str(int(partition*100))}_ratio{str(args.ratio)}.png',
                                        output_dir_image = args.output_img_dir
                )

                ## extract the ids filepath for the returned 
                final_selection_ids = [wp_train_ids[i] for i in final_al_indices] 

                ## save it into a csv for further verification
                key_name = f"ACLR_partition_{str(int(partition*100))}"
                dict_actlr_full_filepath[key_name] = final_selection_ids

                ## MAP this list of filepaths into the tile-filepaths 
                ## retriveves for each partition the associated patches, labels, metadata  - filepath 
                list_images, list_labels, list_metadata = mapdict_patches_filepath(final_selection_ids,
                                                                                    args.patch_folder
                                                                                    )
                output_actlr_tiles_filepath[key_name] = {'images':list_images , 
                                                        'labels':list_labels, 
                                                        'metadata': list_metadata
                                                        }

        ## TRAIN WP- SAVE all dicts into different dfs.
        logger.info("saving patches filepath with the partition and selected by the given seed")

        ##create df of Active Learning partitions for the full image path for verification
        ## pad before adding to a df
        aclr_df_full_filepath = pd.DataFrame({
                                k: pd.Series(v) 
                                for k, v in dict_actlr_full_filepath.items()
                            })
        output_aclr_filename = f"df_actlr_full_filepath_seed_{str(seed_number)}.parquet"
        aclr_df_full_filepath.to_parquet(os.path.join(output_folder, output_aclr_filename))

        ## create df of TILES 
        random_df_patches_filepath = pd.DataFrame().from_dict(output_dict_partitions)
        aclr_df_patches_filepath = pd.DataFrame().from_dict(output_actlr_tiles_filepath)

        ## concat 
        output_df = pd.concat([random_df_patches_filepath,
                                aclr_df_patches_filepath
                                ],
                                axis=1)
        output_filename = f"df_train_test_split_filepath_PATCHES_wpartitions_seed_{str(seed_number)}.parquet"
        logger.success(f"saving file:{output_filename}")
        os.makedirs(output_folder, exist_ok=True)
        logger.success(f"saving at:{os.path.join(output_folder,output_filename)}")
        output_df.to_parquet((os.path.join(output_folder, output_filename)))

        logger.success(f"done for seed:{seed_number}")

