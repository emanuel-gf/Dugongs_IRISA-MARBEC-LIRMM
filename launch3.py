import fiftyone as fo
import os
import pandas as pd 
from pathlib import Path
import numpy as np
import argparse 


def parse_yolo_file(ann_path, sample_root):
    """
    Reads a single .txt file and converts YOLO [x_c, y_c, w, h] 
    to FiftyOne [top-left-x, top-left-y, w, h].

    Args:
    ann_path: FOLDER str  - Path to the FOLDER containing YOLO annotation .txt files
    """
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

def find_proper_csv_environmental_variables(region,
                                            full_path_NC_csv="/home/camarada/Documents/CDE/thesis/dataset_raw/DATASET/NC/dugong_environmental_variables_NC.csv",
                                            full_path_WP_csv="/home/camarada/Documents/CDE/thesis/dataset_raw/DATASET/WP/dugong_environmental_variables_WP.xlsx"):
    """
    Finds the CSV file containing environmental and ecological data.
    It looks for files with specific names in the WP and NC directories.

    Returns:
    A dictionary with keys 'WP' and 'NC' mapping to their respective CSV file paths.
    """
    match region:
        case "WP":
            csv_path = full_path_WP_csv
        case "NC":
            csv_path = full_path_NC_csv
        case _:
            print(f"Warning: Unrecognized region '{region}'. Expected 'WP' or 'NC'.")
            csv_path = None
    
    return csv_path


def load_df_from_csv(filepath):
    "Process the csv containing environmental and ecological data."
    filepath = Path(filepath)
    match filepath.suffix:
        case ".csv":
            return pd.read_csv(filepath)
        case ".xlsx":
            return pd.read_excel(filepath)

def preprocess_csv(df, list_column_exhibit):
    """
    Preprocess the CSV data by creating a 'stem_filename' column for index matching
    Also filter the DataFrame to include one row per image with the specified columns.

    Args:
    df: pandas DataFrame - The original DataFrame loaded from the CSV file.
    list_column_exhibit: List - The list of columns to retain from the original csv. 

    Returns:
    A dictionary mapping 'stem_filename' to the selected metadata columns.
    """
    ## try to rename it
    try:
        df = df.rename(columns={'picture_name':'label_name'})
    except KeyError:
        pass  # If 'picture_name' doesn't exist, do nothing

    if 'label_name' not in df.columns:
        raise KeyError("The DataFrame must contain either 'label_name' or 'picture_name' column for matching.")

    ## Handle the cloud_reflection AND sun_glitter
    ## there are conflicting values that need to be fixed
    def fix_col_cloud(string):
        if string.strip().lower() == '0':
            return '0-0'
        elif string.strip() == '':
            return 'No-value'
        else:
            return string
    if 'cloud_reflection' in df.columns:
        df['cloud_reflection'] = df['cloud_reflection'].astype('str').apply(lambda x:fix_col_cloud(x))
    
    if 'sun_glitter' in df.columns:
        df['sun_glitter'] = df['sun_glitter'].astype('str').apply(lambda x:fix_col_cloud(x))
    
    # handle missing values globally before filtering
    # We use a loop to fill based on the current data type of the column
    for col in list_column_exhibit:
        if col not in df.columns:
            print(f"Col: {col} not found in df. Please check csv and list_column_exhibit")
            continue
            
        try:
            # Check the type of the column and provide an appropriate "Zero" value
            if pd.api.types.is_numeric_dtype(df[col]):
                if pd.api.types.is_integer_dtype(df[col]):
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna(0.0)
            elif pd.api.types.is_bool_dtype(df[col]):
                df[col] = df[col].fillna(False)
            else:
                # Default for objects/strings
                df[col] = df[col].fillna("Unknown")
        except Exception as e:
            print(f"Warning: Could not fill NaNs in column {col}: {e}")
            df[col] = df[col].fillna("None")     

    # Create the stem column for matching
    df['stem_filename'] = df['label_name'].apply(lambda x: Path(x).stem)
    
    # Ensure it doesn't have duplicates and set the index to the stem string
    df_mapped = df.drop_duplicates(subset=['stem_filename'], keep='first')
    
    # Select only the relevant columns for metadata
    df_mapped = df_mapped.set_index('stem_filename')[list_column_exhibit]
    
    # Convert to a dictionary for easy lookup
    field_columns_mapdict = df_mapped.to_dict(orient='index')
    
    return field_columns_mapdict


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
        parent_1 = img_dir.parent.parent
        parent_2 = img_dir.parent.parent.parent

        if parent_1.name == "NC":
            region = parent_1.name
            subregion = "None"

        elif parent_2.name == "WP":
            region = parent_2.name
            subregion = parent_1.name
        else:
            region = "Unknown"
            subregion = "Unknown"
        
        if label_dir.exists():
            sources.append({
                "images": str(img_dir),
                "labels": str(label_dir),
                "mission_name": parent_dir.name, # e.g., 'UM_M5'
                "subregion": subregion, # e.g., 'GAM'
                "region": region # WP or NC
            })
    return sources

 
def main(root_dir:str, launch:bool, path_NC_csv:str, path_WP_csv:str, name_dataset:str="Domain-Shift"):
    """ 
    Construct or load the dataset and launch the session if requested.
    The name_dataset will be used to either import and existent class 
    or create a new class with the given name. 

    Args:
        root_dir: str - The root directory where the dataset is located.
        launch: bool - Whether to launch the FiftyOne App after dataset creation/loading.
        path_NC_csv: str - Path to the New Caledonia CSV file.
        path_WP_csv: str - Path to the West Papua CSV file.
        name_dataset: str - The name to use for the FiftyOne dataset.
    """
    ## construct dict of paths based on the unzip structure.
    root_dir = root_dir

    ## csv paths
    full_path_NC_csv=path_NC_csv
    full_path_WP_csv=path_WP_csv
    if not os.path.exists(full_path_NC_csv):
        print(f"Warning: The specified NC CSV path does not exist: {full_path_NC_csv}")

    ## name of the dataset to be created or loaded from memory
    name_dataset = name_dataset

    list_columns_keep = ['sea_state', 'turbidity_global', 
                            'turbidity_local', 'sun_glitter', 'cloud_reflection',
                            'habitat_type', 'background_complexity', 'coral', 'sand', 'dense_seagrass',
                            'open_sea','sparse_seagrass'
                        ]
    
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
            print(f"Processing sub-dataset: {sub_dataset['mission_name']} <> {sub_dataset['subregion']} <> {sub_dataset['region']}")

            image_dir_path = sub_dataset['images']
            label_dir_path = sub_dataset['labels']
            region = sub_dataset['region'] ## region for csv matching
            tags = [sub_dataset['mission_name'], sub_dataset['subregion'], sub_dataset['region']] ## tags list

            ## find the proper csv
            csv_path = find_proper_csv_environmental_variables(region,
                                                                full_path_NC_csv=full_path_NC_csv,
                                                                full_path_WP_csv=full_path_WP_csv)

            ## check if the csv path was already loaded
            if region != region_prev:
                df = load_df_from_csv(csv_path)
                field_columns_mapdict = preprocess_csv(df, list_columns_keep)
                region_prev = region
                

            ## SAMPLE CREATION
            ## loop through the img_dir 
            for image_filepath in Path(image_dir_path).rglob("*.jpeg"):
                sample = fo.Sample(filepath = str(image_filepath))

                ## tags
                sample["tags"] = tags
                sample['region'] = region
                sample['subregion'] = sub_dataset['subregion']
                sample['mission_name'] = sub_dataset['mission_name']
                
                ## add field columns
                sample_root = image_filepath.stem
        
                if sample_root in field_columns_mapdict:
                    data = field_columns_mapdict[sample_root]

                    # add ecological variables columns
                    for col in list_columns_keep:
                        val = data.get(col)

                        #safety check: skip values that are still effectively null
                        if val is not None and not (isinstance(val, float) and np.isnan(val)):
                            sample[col] = val

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


    ## Create essencial views - WP and NC
    nc_view = dataset.match_tags("NC")
    wp_view = dataset.match_tags("WP")
    dataset.save_view("New_Caledonia", nc_view)
    dataset.save_view("West_Papua", wp_view)
    info = dict(
        name="West_Papua",
        description="West Papua subset samples with tags.",
        )
    dataset.update_saved_view_info("West_Papua", info)

    if launch:
        # Ensures that the App processes are safely launched on Windows
        session = fo.launch_app(dataset)
        # View the dataset's current App config
        print(dataset.app_config)
        session.wait()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Dataset and Launch Session.")
    parser.add_argument("--root",'-r', help="Root directory of the dataset.", default="/home/camarada/Documents/CDE/thesis/dataset_raw/DATASET")
    parser.add_argument("--NCcsv", help="Path to the New Caledonia CSV file.", default="/home/camarada/Documents/CDE/thesis/dataset_raw/DATASET/NC/dugong_environmental_variables_NC.csv")
    parser.add_argument("--WPcsv", help="Path to the West Papua CSV file.", default="/home/camarada/Documents/CDE/thesis/dataset_raw/DATASET/WP/dugong_environmental_variables_WP.xlsx")
    parser.add_argument("--launch","-l", help="Either launch or not.", default=False)
    parser.add_argument("--name_dataset",'-n', help="Name of the dataset to be created or loaded.", default="Domain-Shift")
    args = parser.parse_args()
    print(f"Args: {args}")
    ## create dataset and launch
    main(args.root, args.launch, args.NCcsv, args.WPcsv, name_dataset=args.name_dataset)