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


def find_flight_candidates_WP(
    wp_view,
    wp_train_val_size: float,
    wp_test_size: float,
    subset_size: int,
    seed: int = 0,
    max_tries: int = 100,
    fuzzy: int = 3,
) -> Optional[Dict[str, str]]:
    """
    Find the flights mission candidates to be elected as TEST set only. This will be excluded from the whole
    downstream pipeline.

    Returns:
        dict: {island: selected_flight} if valid
        proportional_budget:dict. e.g: 'UM':54, 'FRIWEN': 32
        None: if no valid split found
    """

    rng = random.Random(seed)

    train_val_size = int(subset_size * wp_train_val_size)
    min_test_size = int(subset_size * wp_test_size)

    logger.info(f"Finding FLIGHT CANDIDATES = WEST PAPUA")
    logger.info(f"Total subset: {subset_size}")
    logger.info(f"Train/Val: {train_val_size}")
    logger.info(f"Test: {min_test_size}")

    # --- Stratify counts ---
    WP_dict_stratify_key = wp_view.count_values("stratify_key")
    proportionality = wp_view.count_values("subregion")

    # --- Remove MANTASANDY from proportionality ---
    full_set = len(wp_view) - proportionality["MANTASANDY"]

    out_prop = {
        k: v / full_set
        for k, v in proportionality.items()
        if k != "MANTASANDY"
    }

    # --- Budgets ---
    proportional_budget_test = {
        k: int(v * min_test_size) for k, v in out_prop.items()
    }

    proportional_budget_trainval = {
        k: int(v * train_val_size) for k, v in out_prop.items()
    }

    logger.info(f"Test budget: {proportional_budget_test}")
    logger.info(f"Train/Val budget: {proportional_budget_trainval}")

    # --- Flights per island ---
    islands_to_split = ["UM", "FRIWEN", "GAM"]

    dict_island = {}
    for island in islands_to_split:
        island_view = wp_view.match(F("subregion") == island)
        dict_island[island] = island_view.distinct("stratify_key")

    flight_keys = set(WP_dict_stratify_key.keys())

    # --- Try multiple times ---
    for attempt in range(max_tries):

        # 1. Sample one flight per island
        flight_candidates = {
            island: rng.choice(flights)
            for island, flights in dict_island.items()
        }

        # 2. Add fixed test flight
        test_flights = set(flight_candidates.values())
        test_flights.add("WP_MANTASANDY_MANTASANDY_M5")  # adjust if needed

        # --- 3. TEST VALIDATION ---
        valid = True

        for island, flight in flight_candidates.items():
            size = WP_dict_stratify_key[flight]
            min_required = proportional_budget_test[island] - fuzzy

            if size < min_required:
                valid = False
                break

        if not valid:
            continue

        # --- 4. TRAIN/VAL VALIDATION ---
        train_val_flights = flight_keys - test_flights

        train_val_by_island = {k: 0 for k in proportional_budget_trainval.keys()}

        for f in train_val_flights:
            island = f.split("_")[1]  # assumes format WP_<ISLAND>_...
            if island in train_val_by_island:
                train_val_by_island[island] += WP_dict_stratify_key[f]

        for island, total in train_val_by_island.items():
            min_required = proportional_budget_trainval[island] - fuzzy

            if total < min_required:
                valid = False
                break

        if valid:
            logger.success(f"[SUCCESS] Found split at attempt {attempt}")
            logger.success(f"Test flights: {test_flights}")
            #for flight in test_flights:
                #print(f"{flight}:{WP_dict_stratify_key[flight]}")
            return flight_candidates,proportional_budget_test

    logger.fail("[FAIL] No valid split found")
    return None,None


def tagged_traintest_seeded_split_subset_WP(
    dataset,
    wp_view,
    wp_train_size: float,
    wp_val_size:float,
    wp_test_size: float,
    subset_size: int,
    flight_candidates, 
    proportional_budget_test,
    seed: int = 0,
    ):
    """
    Retrives the subset to be used on training. It finds the Train and Val

    Randomly chooses the images into the partition for the budget size. 
    The seed controls are deterministic.

    1. Get the flight candidates to be tagged as TEST.
    2. Select the correct number of TESTing samples, tag it.
    3. Tag all the samples belonging to the flight candidates that weren't selected. 
    4. Split the rest of flight missions into Train/Vak
    5. Tag all these missions as train and val. It doesnt cut the train and val to the appropriate subset size!
    6. The cut of TrainVal subset should be done sequentially out of this loop. 
    7. Tag the subset of val here. Defining val_seed and notin_val_seed
    8. The train subset is full taggedd as train to be further subsampled.

    Args:
        wp_train_size and wp_val_size: are dependent, both should some to 1.
        wp_test_size: is independent. The first to be extracted. 
        flight_candidates: List or Dict. If dict 'UM':'flight_mission', ...
        subset_size: int. If no subset size is given, so the subset is the same sie as the full wp view. 
    """
    # set the seed
    rng = random.Random(seed)

    if subset_size is None:
        subset_size = len(wp_view)

    train_size = int(subset_size*wp_train_size)
    val_size =   int(subset_size*wp_val_size)
    test_size =  int(subset_size*wp_test_size)
    logger.info(f"Tag WP for the given flight candidates")
    logger.info(f"Full subset size:{subset_size}")
    logger.info(f"Train size:{train_size}")
    logger.info(f"Val size:{val_size}")
    logger.info(f"Test size:{test_size}")
    logger.info(f"Proportional budget:{proportional_budget_test}")
    
    # transform the dict into a list, if not comes as a list 
    if type(flight_candidates)==dict:
        flight_candidates = list(flight_candidates.values())

    ## TAG TEST
    for flight in flight_candidates:
        logger.info(f"Running flight:{flight}")
        retrieve_island = flight.split('_',2)[1] # get island
        proportion_test_size = proportional_budget_test[retrieve_island]
        subset_view = dataset.match(F("stratify_key")==flight)
        if len(subset_view) > proportion_test_size:
            logger.info(f"Subseting Test: {len(subset_view)} -> {proportion_test_size}")
            
            flight_mission_ids = subset_view.values('id')
            # retrieves the list of ids regarding the flight
            new_test_subset = rng.choices(flight_mission_ids, 
                                          k=proportion_test_size)

            not_in_subset = [ii for ii in flight_mission_ids if ii not in new_test_subset]
            
            # subset
            # TAG all these new samples. 
            dataset.select(new_test_subset).tag_samples(f"test_{str(seed)}")
            dataset.select(not_in_subset).tag_samples(f"notin_TEST_{str(seed)}")

    ## TAG AND SELECT TRAIN/VAL
    all_flights = wp_view.distinct('stratify_key')
    not_in_flights = [f for f in all_flights if f not in flight_candidates]

    ## subset view of trainval
    trainval_view = dataset.match(F("stratify_key").is_in(not_in_flights))
    ids_trainval = trainval_view.values('id')
    strata_trainval = trainval_view.values('stratify_key')

    train_ids, val_ids = train_test_split(
                ids_trainval, 
                test_size= wp_train_size, 
                stratify=strata_trainval, 
                random_state=seed,
                shuffle=True
            )
    # subset the val before tag eveything
    ## train is tagged full, because it can be choosen within the active learninng pipeline
    subset_val_ids = rng.choices(val_ids,
                                 k=int(wp_val_size*subset_size))
    notin_val_ids = [id for id in val_ids if id not in subset_val_ids]
     
    # tag all train and val samples. The subset will be done later on the chain. 
    dataset.select(train_ids).tag_samples(f"train_{str(seed)}")
    dataset.select(subset_val_ids).tag_samples(f"val_{str(seed)}")
    dataset.select(notin_val_ids).tag_samples(f"notin_val_{str(seed)}")
    logger.success(f"WP completed and tagged! On seed:{seed}")
    

# def find_flights_for_test_NC(nc_view, 
#                           dict_flights,
#                           seed_number:int, 
#                           test_size=0.15, 
#                           fuzzy=0.03,
#                           k_choices=(1, 2, 3, 4, 5), 
#                           max_tries=50):
#     logger.info(f"Running Finding Flights for NEW CALEDONIA")
#     ## seed
#     rng = random.Random(seed_number) 
#     rng.seed(seed_number)

#     budget = len(nc_view) * test_size
#     superior_limit = len(nc_view) * (test_size + fuzzy)

#     for attempt in range(max_tries):
#         k = random.choice(k_choices)  # random pick, not looping
#         flights_choice = random.choices(list(dict_flights.keys()), k=k)
#         sum_nc_train = np.array([dict_flights[f] for f in flights_choice]).sum()

#         if budget <= sum_nc_train <= superior_limit:
#             logger.success("Matched:")
#             logger.info(f"flights selected for test: {flights_choice}")
#             logger.info(f"k used: {k}")
#             logger.info(f"budget: {budget}, superior limit: {superior_limit}")
#             logger.info(f"number of images: {sum_nc_train}")
#             return flights_choice

#     logger.error("No match found within max tries.")
#     return None


def find_seed_for_valid_split_NC(nc_view, 
                                seed_range=range(5000),
                                test_size=0.15, 
                                fuzzy=0.03,
                                k_choices=(1, 2, 3, 4),
                                max_tries=100
                                ):
    """
    Find the seed that completes the flight mission split
    """
    logger.info(f"--------------------- \n")
    logger.info(f"New run")

    dict_flights = nc_view.count_values('stratify_key')

    for seed in seed_range:
        rng = random.Random(seed) 

        budget = len(nc_view) * test_size
        superior_limit = len(nc_view) * (test_size + fuzzy)

        for attempt in range(max_tries):
            k = random.choice(k_choices)  # random pick of how many independent flight missions
            flights_choice = rng.choices(list(dict_flights.keys()), 
                                            k=k
                                            )
            sum_nc_train = np.array([dict_flights[f] for f in flights_choice]).sum()

            if budget <= sum_nc_train <= superior_limit:
                logger.success("Matched:")
                logger.info(f"flights selected for test: {flights_choice}")
                logger.info(f"k used: {k}")
                logger.info(f"budget: {budget}, superior limit: {superior_limit}")
                logger.info(f"number of images: {sum_nc_train}")
                return seed, flights_choice


    

def tag_traintest_seeded_NC(dataset,
                            nc_view,
                            nc_flight_candidates:list,
                            seed_number:int, 
                            nc_train_size:float,
                            nc_val_size:float,
                            nc_test_size:float):
    """
    For a given random seed, find which flight mission should be used to test.
    Then, tag all these flights without subsetting the NC. 
    Split the remaining flights into train and val using a stratification given by flight mission.
    Tag everything.
    """
    
    dict_flights = nc_view.count_values('stratify_key')


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
        

def run_seeded_splits_and_TAG(dataset, 
                      nc_view, 
                      wp_view, 
                      runs: int, 
                      subset_size: int = 760,
                      nc_test_size:float = 0.15,
                      nc_train_size:float = 0.8,
                      nc_val_size:float = 0.2,
                      wp_train_val_size:float = 0.9,
                      wp_train_size:float = 0.85,
                      wp_val_size:float = 0.15,
                      wp_test_size: float = 0.1
                        ):
    """
    Run seeded splits for NC and WP, ensuring:
    - Each seed uses unique flights for test.
    - Seeds are consistent across NC and WP.
    - No flight is reused across seeds.

        # the task consist in find flights missions viable to be select as test, both WP and NC, separately
    # from these flights, tag test set.The remaining flights are split in train/val with stratification
    # given by flight mission. 
    # From this train/val flight mission, if the view needs to subset, so a set of is tagged on train/val
    # and the remaining are tagged as notin_train_ and notin_val 
    """
    used_flights_NC = []
    used_flights_WP = []
    seeds_list = []

    # loop to the number of runs to do this task.
    for run in range(runs + 1):
        # --- NC: Find seed and flights ---
        seed_number, selected_flights_NC_TEST = find_seed_for_valid_split_NC(
            nc_view,
            seed_range=range(run * 1000, (run + 1) * 1000),
            test_size=nc_test_size,
            fuzzy=0.03,
            max_tries=50
        )

        # Ensure flights are unique
        while any(flight in used_flights_NC for flight in selected_flights_NC_TEST):
            seed_number, selected_flights_NC_TEST = find_seed_for_valid_split_NC(
                nc_view,
                seed_range=range(seed_number + 1, (run + 2) * 1000),
                test_size=nc_test_size,
                fuzzy=0.03,
                max_tries=50
            )
        ## append to the list 
        used_flights_NC.extend(selected_flights_NC_TEST)
        seeds_list.append(seed_number)

        ## ----------------------------------------------
        # Tag NC
        tag_traintest_seeded_NC(
            dataset,
            nc_view,
            nc_flight_candidates=selected_flights_NC_TEST,
            seed_number=seed_number,
            nc_train_size=nc_train_size,
            nc_val_size=nc_val_size,
            nc_test_size=nc_test_size
        )

        ## WP WP WP WP WP  WP 
        ## ------------------------------------
        # --- WP: Find flights ---
        flight_candidates, proportional_budget_test = find_flight_candidates_WP(
            wp_view,
            wp_train_val_size=wp_train_val_size,
            wp_test_size=wp_test_size,
            subset_size=subset_size,
            seed=seed_number,
            max_tries=100,
            fuzzy=3
        )

        # Ensure flights are unique
        ## Here we allow that if 2 out of 3 flights are unique, so the test set is considered unique
        # Count how many flights are already used
        used_count = sum(1 for flight in flight_candidates.values() if flight in used_flights_WP)

        # Only retry if 2 or more flights are duplicates
        while used_count >= 2:
            flight_candidates, _ = find_flight_candidates_WP(
                wp_view,
                wp_train_val_size=wp_train_val_size,
                wp_test_size=wp_test_size,
                subset_size=subset_size,
                seed=seed_number + 1,
                max_tries=100,
                fuzzy=3
            )
            used_count = sum(1 for flight in flight_candidates.values() if flight in used_flights_WP)

        # Add the new flights to the used list (even if 1 is a duplicate)
        used_flights_WP.extend(flight_candidates.values())

        # Tag WP
        tagged_traintest_seeded_split_subset_WP(
            dataset,
            wp_view,
            wp_train_size=wp_train_size,  # 80% of the remaining 90% (after test)
            wp_val_size=wp_val_size,     # 20% of the remaining 90%
            wp_test_size=wp_test_size,
            subset_size=subset_size,
            flight_candidates=flight_candidates,
            seed=seed_number,
            proportional_budget_test = proportional_budget_test
        )

    return seeds_list


def return_list_filepath_train_test_val(seed_number, nc_view, wp_view):
    """
    Returns: train_wp, test_wp, val_wp, train_nc, test_nc, val_nc list of filepaths. 
    """
    train_wp_filepath = wp_view.match_tags(f"train_{seed_number}").values("filepath")
    test_wp_filepath = wp_view.match_tags(f"test_{seed_number}").values("filepath")
    val_wp_filepath = wp_view.match_tags(f"val_{seed_number}").values("filepath")
    train_nc_filepath = nc_view.match_tags(f"train_{seed_number}").values("filepath")
    test_nc_filepath = nc_view.match_tags(f"test_{seed_number}").values("filepath")
    val_nc_filepath = nc_view.match_tags(f"val_{seed_number}").values("filepath")
    return train_wp_filepath, test_wp_filepath, val_wp_filepath, train_nc_filepath, test_nc_filepath, val_nc_filepath


def build_filepath_df(train_wp_filepath, test_wp_filepath, val_wp_filepath,
                       train_nc_filepath, test_nc_filepath, val_nc_filepath):
    """
    Cstruct a simple df from the given entries.

    Returns: df
    """

    df = pd.DataFrame({
        "train_wp": pd.Series(train_wp_filepath),
        "test_wp": pd.Series(test_wp_filepath),
        "val_wp": pd.Series(val_wp_filepath),
        "train_nc": pd.Series(train_nc_filepath),
        "test_nc": pd.Series(test_nc_filepath),
        "val_nc": pd.Series(val_nc_filepath),
    })

    return df

## FUNCTIONS TO LOAD IT BACK 
def get_seed_from_filepath(csv_file):
    path = Path(csv_file).stem
    return path.split('_')[-1]


def return_list_from_csv(csv_file):
    dff = pd.read_csv(csv_file)
    wp_train_list = dff['train_wp'].dropna().values
    wp_test_list = dff['test_wp'].dropna().values
    wp_val_list = dff['val_wp'].dropna().values
    nc_train_list = dff['train_nc'].dropna().values
    nc_test_list =dff['test_nc'].dropna().values 
    nc_val_list = dff['val_nc'].dropna().values
    return wp_train_list, wp_test_list, wp_val_list, nc_train_list, nc_test_list, nc_val_list


## RANDOM PARTITION
def random_choice_train_list(train_list,
                             seed,
                             partitions:list=[0.1,0.25,0.5,0.75,1.0]
                             ):
    """
    Select a randomly the paths files from the list

    Returns:
    Each key of the partition name where the values are a list of filepaths.
    """
    length = len(train_list)

    ## seed 
    random.seed(seed)

    dict_out ={}
    for p in partitions:
        num_images = int(math.floor(length*p))
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


def plot_landscape_to_axes(axes, coords_list, labels, uniq_idx, km_idx):
    names = ["PCA", "UMAP", "t-SNE"]
    for i, ax in enumerate(axes):
        # Background: All potential candidates
        ax.scatter(coords_list[i][:, 0], 
                   coords_list[i][:, 1], 
                   c='red', s=8, alpha=0.15)
        
        # 1. Cluster Representatives (The "Diversity" picks)
        ax.scatter(coords_list[i][km_idx, 0], coords_list[i][km_idx, 1], 
                   c='cyan', marker='o', s=35, edgecolor='black', linewidth=0.5)
        
        # 2. Uniqueness Outliers (The "Informative" picks)
        ax.scatter(coords_list[i][uniq_idx, 0], coords_list[i][uniq_idx, 1], 
                   c='magenta', marker='*', s=80, edgecolor='black', linewidth=0.5)
        
        ax.set_title(names[i], fontsize=12)
        ax.set_xticks([]); ax.set_yticks([]) # Clean look for manifolds
        ax.grid(True, linestyle='--', alpha=0.2)


def save_consolidated_active_learning_report(uniqueness_scores, 
                                            coords_list, 
                                            labels, 
                                            uniq_idx, 
                                            km_idx, 
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
    plot_landscape_to_axes(axes_bottom, coords_list, labels, uniq_idx, km_idx)

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
        Line2D([0], [0], marker='o', color='w', label='K-Means (Cluster Medoids)',
               markerfacecolor='cyan', markeredgecolor='black', markersize=10),
        Line2D([0], [0], marker='*', color='w', label='Uniqueness (High-Density Outliers)',
               markerfacecolor='magenta', markeredgecolor='black', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='Unselected Manifold',
               markerfacecolor='lightgrey', alpha=0.4, markersize=8)
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


def active_learning_pipeline(
    embeddings_norm,
    partition_size: float,
    filename_image:str,
    output_dir_image:str ,
    ratio_cluster_uniqueness: float = 0.5,
    n_clusters: int = 10,
    temperature: float = 0.5,
    seed_number: int = 42,
    buffer_percent:float = 0.2
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
    target_total_count = math.floor(partition_size * total_samples)
    
    logger.info(f"--- Starting Pipeline (Target Subset Size: {target_total_count}) ---")

    # 1. Compute Uniqueness
    uniqueness_scores = compute_weighted_uniqueness(embeddings_norm)
    #plot_uniqueness_distribution(uniqueness_scores)

    # 2. Determine Number of Uniqueness vs. Clustering images
    # budget_uniqueness = total_target * ratio (e.g. 500 = 1000 * 0.5)
    target_uniq_count = math.floor(target_total_count * ratio_cluster_uniqueness)
    
    # Use your selector to find the best percentile threshold
    chosen_percentile = selector_num_uniqueness(
        partition_size=partition_size,
        wp_train_length=total_samples,
        uniqueness=uniqueness_scores,
        ratio_cluster_uniqueness=ratio_cluster_uniqueness
    )
    
    # Extract Uniqueness Indices
    threshold_value = np.percentile(uniqueness_scores, chosen_percentile)
    uniqueness_indices = np.where(uniqueness_scores >= threshold_value)[0]
    
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
        n_clusters, 
        samples_per_cluster, 
        temperature=temperature,
        seed=seed_number
    )
    
    # 6. Consolidate Indices
    # We use a set to ensure no duplicates if a unique point is also a cluster medoid
    overlap = len(set(uniqueness_indices) & set(kmeans_indices))
    logger.info(f"Overlap between uniqueness and clustering: {overlap}")
    combined_indices = list(set(uniqueness_indices) | set(kmeans_indices))

    # Clip to target_total_count
    combined_indices = combined_indices[:target_total_count]
    logger.success(f"Final Consolidated Subset Size: {len(combined_indices)}")


    # Plot
    logger.info("Computing Projections for final visualization...")
    pca_coords = PCA(n_components=2, random_state=seed_number).fit_transform(embeddings_norm)
    umap_coords = UMAP(n_components=2, random_state=seed_number, n_jobs=1 ).fit_transform(embeddings_norm)
    tsne_coords = TSNE(n_components=2, random_state=seed_number).fit_transform(embeddings_norm)

    # plot_consolidated_landscape(
    #     pca_coords, 
    #     umap_coords, 
    #     tsne_coords, 
    #     uniqueness_indices, 
    #     kmeans_indices,
    #     title=f" Pipeline: {partition_size*100}% Partition \n Uniq: {len(uniqueness_indices)} | Clust: {len(kmeans_indices)})"
    # )

    coords_list = [pca_coords, umap_coords, tsne_coords]

    save_consolidated_active_learning_report(
        uniqueness_scores, 
        coords_list, 
        cluster_labels, 
        uniqueness_indices, 
        kmeans_indices,
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
    parse.add_argument('--nc-train-size', type=float,default=0.8)
    parse.add_argument('--nc-test-size', type=float,default=0.15)
    parse.add_argument('--nc-val-size', type=float,default=0.20)
    parse.add_argument('--wp-train-val-size', type=float, default=0.9, help="Size of train + val. To be later as 80/20")
    parse.add_argument('--wp-test-size', type=float,default=0.1, help='Size of the test size')
    parse.add_argument('--num-seeds',type=int,default=2, help="Number of diferent seeds to tag the dataset." \
                                                " If 1, so 2 seeds are tagged. N+1")
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
    parse.add_argument('--ratio', type=float, default=0.33,
                       help='Ratio of uniqueness over clusters. If 0.5 so 1:1, if 0.33 so 1:2, which means two images of cluster for each one image of uniqueness.')
    parse.add_argument('--temperature', type=float,default=0.5, help= 'temperature-scaled Softmax distribution for selecting the samples closer to the cluster centroid.')
    return parse.parse_args()

def main():
    from datetime import datetime 
    datetime = datetime.now().strftime('%m%d_%H%M')
    PARTITIONS_ = [0.05,0.1,0.25,0.5,0.75,0.90,0.95, 1.0]
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
    ## load the views
    nc_view = dataset.match(F('region').starts_with('NC'))
    wp_view = dataset.match(F('region').starts_with('WP'))

    ## check if the dataset has the key used for stratification
    assert dataset._has_field(args.stratify_key), f"Field does not exist in the dataset. Please add it before run"

    # If a subset if given, so the whole pipeline process as a new size
    if args.subset_size is not None:
            wp_subset_size = int(args.subset_size)
            # multiply the ratio by the partitions to retrieve the partition per subset
            PARTITIONS = np.array(PARTITIONS_)*np.array((wp_subset_size/len(wp_view)))
            logger.info(f"SUBSET MODE | ON:")
            logger.info(f"old partitions:{PARTITIONS_}")
            logger.info(f"Adapted partitions:{PARTITIONS}")
            logger.info(f"Size of the total WP:{len(wp_view)}")
            logger.info(f"New size of subset:{wp_subset_size}")
    else:
            PARTITIONS = PARTITIONS_

    ## run the split and tag the dataset. 
    logger.info(f"tagging the dataset and split the train,test,val")
    seed_number_list = run_seeded_splits_and_TAG(dataset, 
                                                nc_view, 
                                                wp_view, 
                                                runs = args.num_seeds, 
                                                subset_size =  wp_subset_size,
                                                nc_test_size = args.nc_test_size,
                                                nc_train_size = args.nc_train_size,
                                                nc_val_size = args.nc_val_size,
                                                wp_train_val_size = args.wp_train_val_size,
                                                wp_test_size = args.wp_test_size
                                                )
    logger.success("tagged!")
    logger.info(f"------ nNEXT STAGE")

    ## AT THIS POINT
    ## THE FLIGHT MISSION IS NOT REDUCED YET, ALL FLIGHT MISSIONS FOR TRAIN WERE TAGGED
    ## unless VAL and TEST,which is tagged by notin_test and test_
    ## therefore, the create csv with filepath contains all files on it. 

    csv_filename_list = []
    ## IMPLEMENT LOOP HERE
    ## RUN ALL GIVEN SEEDS AND CREATES A CSV WITH THE PATHS REGARDING THE FULL IMAGE
    for ss in seed_number_list:
        logger.info(f"Running seed:{ss}")
        (train_wp_filepath, test_wp_filepath, val_wp_filepath, 
        train_nc_filepath, test_nc_filepath, val_nc_filepath) = return_list_filepath_train_test_val(
            ss,
            nc_view=nc_view,
            wp_view = wp_view
        )

        df_seed = build_filepath_df(
            train_wp_filepath, 
            test_wp_filepath, 
            val_wp_filepath, 
            train_nc_filepath, 
            test_nc_filepath, 
            val_nc_filepath
        )

        ## save it keeping 
        output_filename = f"df_train_test_split_filepath_wpsubset{wp_subset_size}_{str(ss)}.csv"
        logger.success(f"saving file:{output_filename}")
        output_folder = args.output_folder
        logger.success(f"saving at:{os.path.join(output_folder,output_filename)}")
        df_seed.to_csv(os.path.join(output_folder, output_filename))
        logger.success('done!')
        csv_filename_list.append(output_filename)
    
    ## -------------------------------------------------------------------------------------
    ## SECOND PART ---------------------------------------
    ## -------------------------------------------------------------
    logger.info(f"generating the fraction partition of WP train subsets for:{csv_filename_list}")

    ## LOAD IT BACK AND GENERATE THE MATCHING FOR PARTITIONING THE WP_TRAIN SET IN FRACTIONS
    ## do in a loop
    for csv_file in csv_filename_list:
        (wp_train_list, wp_test_list, wp_val_list, 
         nc_train_list, nc_test_list, nc_val_list) = return_list_from_csv(os.path.join(args.output_folder,csv_file))
        
        ## get seed
        seed_number = int(get_seed_from_filepath(csv_file))
        logger.info(f"Running seed:{seed_number}")
        random.seed(seed_number)

        ## TRAIN WP--------------------
        ## ---------------------------- RANDOM SELECTION --------------------------------------
        ## create a dict containing the filepath_stem with the keys containig the filepath for images, labels and metadata
        dictt_random_choice = random_choice_train_list(train_list = wp_train_list,
                                            seed = seed_number,
                                            partitions = PARTITIONS
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
        train_wp = wp_view.match(F('tags').contains(f"train_{seed_number}"))
        assert len(train_wp)!=0, f"Error on finding the tagged train dataset for seed: {seed_number}"

        ## get the embeddings relative to the train set.
        wp_train_emb = train_wp.values('full_embeddings')
        wp_train_emb_norm = normalize(wp_train_emb) #normalize 

        wp_train_ids = train_wp.values('filepath')
        
        dict_actlr_full_filepath = {}
        output_actlr_tiles_filepath = {}
        for partition in PARTITIONS:
            if partition > 0.5:
                pass
            else:
                logger.info(f"Running partition:{partition}")

                ## # Run the consolidate pipeline
                final_al_indices = active_learning_pipeline(
                                        wp_train_emb_norm,
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


if __name__ == "__main__":
    setup_logger()
    main()