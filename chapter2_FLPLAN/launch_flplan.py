import os
import pandas as pd 
from pathlib import Path
import numpy as np
import shutil
import argparse
import sys

# --- ARGPARSE SETUP ---
def get_args():
    parser = argparse.ArgumentParser(description="Create Dataset and Launch Session.")
    parser.add_argument("--root", '-r', help="Root directory of the dataset.", 
                        default="/share/home/e2406743/dataset")
    parser.add_argument("--path_database", help="Folder for MongoDB (e.g., /tmp/my_db)", default = None)
    parser.add_argument("--port_database", help="Port for MongoDB (0 for random)", default="44123")
    parser.add_argument("--launch", "-l", help="Launch the FiftyOne App")
    parser.add_argument("--name_dataset", '-n', help="Dataset name", default="Domain-Shift")
    return parser.parse_args()

def parse_yolo_file(ann_path, sample_root):
    """
    Reads a single .txt file and converts YOLO [x_c, y_c, w, h] 
    to FiftyOne [top-left-x, top-left-y, w, h].

    Args:
    ann_path: FOLDER str  - Path to the FOLDER containing YOLO annotation .txt files
    """
    import fiftyone as fo
    file_path = os.path.join(ann_path, f"{sample_root}.txt")
    detections = []

    if not os.path.exists(file_path):
        return None

    with open(file_path, 'r') as file:
        for line in file:
            ann = line.strip().split()
            if not ann: continue

            # Class mapping
            label = "Dugong" if ann[0] == "0" else str(ann[0])

            # Dimensions
            w, h = float(ann[3]), float(ann[4])
            
            # Math: Center to Top-Left
            top_left_x = float(ann[1]) - (w / 2)
            top_left_y = float(ann[2]) - (h / 2)

            detections.append(
                fo.Detection(label=label, bounding_box=[top_left_x, top_left_y, w, h])
            )

    return fo.Detections(detections=detections)


## launch
def convert_bool(value:str) -> bool :
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.strip().lower() in ['true', '1', 'yes']:
            return True
        else:
            return False




def find_sub_data_sources(base_dir):
    """
    Finds all 'images' folders and their sibling 'labels_yolo' folders 
    within a nested structure.

    It looks for parent folder and stored it as keys into the dic. 
    For further be processed as tags. 
    """
    sources = []
    base_path = Path(base_dir)
    
    # We look for all directories named 'images' regardless of how deep they are
    for img_dir in base_path.rglob("images"):
        # The parent is the mission folder (e.g., UM_M5)
        parent_dir = img_dir.parent
        label_dir = parent_dir / "labels_yolo"
        
        if label_dir.exists():
            sources.append({
                "images": str(img_dir),
                "labels": str(label_dir),
                "region":"WP",
                "mission_name": "UM", # e.g., 'UM_M5' # ALL THE IMAGES HERE ARE FROM UM
                "parent_name": Path(parent_dir).stem, # e.g., 'FLPLAN_M4_UM_M4_F1_2025'
            })
    return sources

 
def main():
    """ 
    Construct or load the dataset and launch the session if requested.
    The name_dataset will be used to either import and existent class 
    or create a new class with the given name. 

    Args:
        root_dir: str - The root directory where the dataset is located.
        launch: str - Whether to launch the FiftyOne App after dataset creation/loading.
        name_dataset: str - The name to use for the FiftyOne dataset.
        path_database: str - The path where the MongoDB database should be stored.
                                If None so defaults to the fiftyone default (e.g., /user/fiftyone/mongodb).
        path_port_db: str - The port for the MongoDB database (0 for random).
    """
    args = get_args()

    # --- STEP 1: DATABASE CONFIGURATION (MUST BE BEFORE IMPORT) ---
    # We use environment variables because FiftyOne reads these during initialization.
    
    # Use provided path or default to a user-specific tmp folder
    args = get_args()

    # Connect to the EXISTING MongoDB — do NOT start a new one
    os.environ["FIFTYONE_DATABASE_URI"] = f"mongodb://localhost:{args.port_database}"

    import fiftyone as fo

    try:
        existing = fo.list_datasets()
        print(f"Connected. Existing datasets: {existing}")
    except Exception as e:
        print(f"ERROR: Could not connect to MongoDB at localhost:{args.port_database}.\n{e}")
        return


    # FIFTYONE ---
    import fiftyone as fo
    print(f"FiftyOne is using database at: {fo.config.database_dir}")

    # Validation check: Ensure we are actually connected
    try:
        fo.list_datasets()
        print(f"Connected to MongoDB successfully!")
    except Exception as e:
        print(f"ERROR: Could not connect to manual MongoDB at localhost:44123. \n{e}")
        return
    
    # pass the rest of the args ---
    name_dataset = args.name_dataset
    root_dir = args.root
    launch = convert_bool(args.launch)
    assert isinstance(launch, bool), f"Expected boolean value for launch, got {type(launch)}"

    ## name of the dataset to be created or loaded from memory
    name_dataset = name_dataset
    
    ## construct map dict with paths and tags for each sub dataset. 
    sub_ds_list = find_sub_data_sources(root_dir)

    ## Try to load the dataset first instead of creating every time.
    if name_dataset in fo.list_datasets():
        # Load the existing one from the DB 
        dataset = fo.load_dataset(name_dataset)
        print("Dataset loaded from database.")
    else:
        ## Create the dataset in case does not exist 
        dataset = fo.Dataset(name=name_dataset, overwrite=True)
        dataset.persistent = True
        print(f"{name_dataset} - was created and saved as a persistent dataset.")

        ## Loop over all the regions and subregions 
        ## create a list for baching
        samples_to_add = []
        region_prev = None
        for sub_dataset in sub_ds_list:
            print(f"Processing sub-dataset: {sub_dataset['mission_name']} <> {sub_dataset['parent_name']}")

            image_dir_path = sub_dataset['images']
            label_dir_path = sub_dataset['labels']
            region = sub_dataset['region'] ##
            parent_name = sub_dataset['parent_name']

            ## SAMPLE CREATION
            ## loop through the img_dir 
            for image_filepath in Path(image_dir_path).rglob("*.jpeg"):
                sample = fo.Sample(filepath = str(image_filepath))

                ## tags
                sample['region'] = region
                sample['mission_name'] = sub_dataset['mission_name']
                sample['parent_name'] = parent_name
                
                ## add field columns
                sample_root = image_filepath.stem
        
                # Handle Labels (YOLO)
                yolo_labels = parse_yolo_file(label_dir_path, sample_root)
                if yolo_labels:
                    sample["ground_truth"] = yolo_labels
                
                ## save sample
                samples_to_add.append(sample)

        print("Batching all inside dataset")
        dataset.add_samples(samples_to_add, progress=True)
        print(f"All sub-datasets have been processed and added to the '{name_dataset}' dataset.")
        # ## compute metadata
        dataset.compute_metadata()
        dataset.save()


    if launch:
        # Ensures that the App processes are safely launched on Windows
        session = fo.launch_app(dataset, remote=True)
        # View the dataset's current App config
        print(dataset.app_config)
        session.wait()

if __name__ == "__main__":
    main()