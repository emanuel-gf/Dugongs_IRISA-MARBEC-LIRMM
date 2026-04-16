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



def tag_train_test_split_seeded(train_size:float,
                                test_size:float,
                                val_size:float,
                                runs:int,
                                dataset,
                                stratification_key = 'stratify_key',
                                island_to_split = ['NC','UM', 'GAM', 'FRIWEN']
                                ):
    """
    Creates the split for train, test and validation using a stratified pick given by the complexity. 
    Args:
        runs: Number of loop to pass and create the tag inside the dataset 
    
        Returns:
        Return the seed_number and the dataset get tagged.
    """
    # target islands (skipping MANTASANDY)
    islands_to_split = island_to_split
    seeds_list = []

    ## tags adding train_seed_number for west papua 
    for run in range(0,runs+1):
        seed_number  = random.randint(1,50)
        print(f"Seed number selected:{seed_number}")
        seeds_list.append(seed_number)
        for island in islands_to_split:
            # Get a view of just this island
            island_view = dataset.match(F("subregion") == island)
            
            ids = island_view.values("id")

            # stratification field presented in the dataset 
            strata = island_view.values(stratification_key)
            
            # Split off the TEST set (20%) ---
            # Stratify ensures the 'high complexity' ratio stays the same
            train_val_ids, test_ids = train_test_split(
                ids, 
                test_size= test_size, 
                stratify=strata,
                shuffle=True, 
                random_state=seed_number
            )
            
            # Get strata for the remaining 80% to split again
            train_val_strata = [s for i, s in zip(ids, strata) if i in train_val_ids]
            
            # Split remaining 80% into Train (70% total) and Val (10% total) ---
            # 0.125 * 0.8 = 0.1 (which is 10% of the original total)
            train_ids, val_ids = train_test_split(
                train_val_ids, 
                test_size= (val_size/(1-test_size)), 
                stratify=train_val_strata, 
                random_state=seed_number
            )
            
            # 3. Apply the tags in FiftyOne
            dataset.select(train_ids).tag_samples(f"train_{str(seed_number)}")
            dataset.select(val_ids).tag_samples(f"val_{str(seed_number)}")
            dataset.select(test_ids).tag_samples(f"test_{str(seed_number)}")
            
            print(f"Island {island}: Train={len(train_ids)}, Test={len(test_ids)}, Val={len(val_ids)}")

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
    wp_train_list = dff['train_seed'].dropna().values
    wp_test_list = dff['test_seed'].dropna().values
    wp_val_list = dff['val_seed'].dropna().values
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
    dict_map_filepath = {}
    for path in list_paths:
        stem = Path(path).stem
        dict_map_filepath[stem] = get_files_by_stem(stem, patch_folder)

    ## flat dict
    filepath_all_images   = [f for d in dict_map_filepath.values() for f in d.get('images', [])]
    filepath_all_labels   = [f for d in dict_map_filepath.values() for f in d.get('label', [])]
    filepath_all_metadata = [f for d in dict_map_filepath.values() for f in d.get('metadata', [])]

    ## -------------
    print(f'images:{len(filepath_all_images)}')
    print(f"labels:{len(filepath_all_labels)}")
    print(f"metadata:{len(filepath_all_metadata)}")

    return filepath_all_images, filepath_all_labels, filepath_all_metadata


def argparse():
    parse = ArgumentParser(description='Train Test and Val split')
    parse.add_argument('--dataset', default='dugong')
    parse.add_argument('--train-size', type=float,default=0.7)
    parse.add_argument('--test-size', type=float,default=0.15)
    parse.add_argument('--val-size', type=float,default=0.15)
    parse.add_argument('--num-seeds',type=int,default=1, help="Number of diferent seeds to tag the dataset." \
                                                " If 1, so 2 seeds are tagged. N+1")
    parse.add_argument('--stratify-key',type=str,default='stratify_key',help="The field name that contains the strategy" \
                                                                            "for stratification of the partitioning.")
    parse.add_argument('--output-folder',type=str,default="/share/home/e2406743/dataset/df_filepaths",
                       help="folder where to store the csv paths generated.")
    parse.add_argument('--patch-folder',type=str, 
                       help='Folder where the images tiles are located.')

def main():
    args = argparse()
    print(args)

    dataset_mongodb = args.dataset
    assert dataset_mongodb in fo.list_datasets(), f"Dataset not valid, should be one of these:{fo.list_dataset()}"
    assert os.path.isdir(args.output_folder), os.makedirs(args.output_folder, exist_ok=True)
    assert os.path.isdir(args.patch_folder), f"Patch folder does not exist"

    ## Load dataset and views from mongodb 
    dataset = fo.load_dataset(dataset_mongodb)
    ## load the views
    nc_view = dataset.match(F('region').starts_with('NC'))
    wp_view = dataset.match(F('region').starts_with('WP'))

    ## check if the dataset has the key used for stratification
    assert dataset._has_field(args.stratify_key), f"Field does not exist in the dataset. Please add it before run"

    ## run the split and tag the dataset. 
    print(f"tagging the dataset and split the train,test,val")
    seed_number_list = tag_train_test_split_seeded(args.train_size,
                                                    args.test_size,
                                                    args.val_size,
                                          runs=args.num_seeds,
                                          dataset= dataset,
                                          stratification_key=args.stratify_key
                                          )
    print("tagged!")

    csv_filename_list = []
    ## IMPLEMENT LOOP HERE
    ## RUN ALL GIVEN SEEDS AND CREATES A CSV WITH THE PATHS REGARDING THE FULL IMAGE
    for ss in seed_number_list:
        print(f"Running seed:{ss}")
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
        output_filename = f"df_train_test_split_filepath_{str(ss)}.csv"
        print(f"saving file:{output_filename}")
        output_folder = args.output_folder
        print(f"saving at:{os.path.join(output_folder,output_filename)}")
        df_seed.to_csv(os.path.join(output_folder, output_filename))
        print('done!')
        csv_filename_list.append(output_filename)
    
    ## SECOND PART
    print(f"generating the fraction partition of WP train subsets for:{csv_filename_list}")

    ## LOAD IT BACK AND GENERATE THE MATCHING FOR PARTITIONING THE WP_TRAIN SET IN FRACTIONS
    ## do in a loop
    for csv_file in csv_filename_list:
        (wp_train_list, wp_test_list, wp_val_list, 
         nc_train_list, nc_test_list, nc_val_list) = return_list_from_csv(csv_file)
        
        ## get seed
        seed_number = int(get_seed_from_filepath(csv_file))
        print(f"Running seed:{seed_number}")
        random.seed(seed_number)

        ## TRAIN WP--------------------
        ## create a dict containing the filepath_stem with the keys containig the filepath for images, labels and metadata
        dictt_random_choice = random_choice_train_list(train_list = wp_train_list,
                                            seed = seed_number,
                                            partitions = [0.05,0.1,0.25,0.5,0.75,1.0]
                                        )
        

        ## LOOP into each dictt key and return a full dataset
        ## return for each partition the paths associated for images, labels and metadata.

        new_dict = dictt_random_choice.copy()
        output_dict_partitions = dict()
        ## each key is a full filepath
        for key in new_dict.keys():
            print(f"Running key:{key}")
            ## retriveves for each partition the associated patches, labels, metadata  - filepath 
            list_images, list_labels, list_metadata = mapdict_patches_filepath(dictt_random_choice[key],
                                                                                args.patch_folder)
            output_dict_partitions[key] = {'images':list_images , 
                                           'labels':list_labels, 
                                           'metadata': list_metadata}

        ## TRAIN WP- SAVE DF 
        print("\n")
        print("saving patches filepath with the partition and selected by the given seed")
        df_patches_filepath = pd.DataFrame().from_dict(output_dict_partitions)
        output_filename = f"df_train_test_split_filepath_PATCHES_wpartitions_seed_{str(seed_number)}.parquet"
        print(f"saving file:{output_filename}")
        os.makedirs(output_folder, exist_ok=True)
        print(f"saving at:{os.path.join(output_folder,output_filename)}")
        df_patches_filepath.to_parquet((os.path.join(output_folder, output_filename)))
        print(f"done for seed:{seed_number}")
        print('-'*30)