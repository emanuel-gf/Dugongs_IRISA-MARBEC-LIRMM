import fiftyone as fo
import os
import pandas as pd 
from pathlib import Path
import numpy as np

def parse_yolo_file(ann_path, sample_root):
    """
    Reads a single .txt file and converts YOLO [x_c, y_c, w, h] 
    to FiftyOne [top-left-x, top-left-y, w, h].
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

def create_dataset(dataset, ann_path,  csv_path, list_column_exhibit):
    """
    Orchestrates the metadata and label attachment.
    Allow stilizing the metadata from csv files

    Args:
    dataset: fiftyone Dataset instance to populate
    ann_path: str - Path to the folder containing YOLO annotation .txt files
    csv_path: str - Path to the CSV file containing metadata (with a 'label_name' column that matches image filenames)
    list_column_exhibit: List - Name of the columns to add at the dataset metadata. It is the filter of csv_path file. So columns should match.
    """
    # Load CSV data into memory
    df = pd.read_csv(csv_path)

    # Create the stem column for matching
    df['stem_filename'] = df['label_name'].apply(lambda x: Path(x).stem)
    
    # Ensure it don't have duplicates and set the index to the stem string
    # drop duplicates one row per image
    df_mapped = df.drop_duplicates(subset=['stem_filename'], keep='first')
    # select columns
    df_mapped = df_mapped.set_index('stem_filename')[list_column_exhibit]
    metadata_map = df_mapped.to_dict(orient='index')

    for sample in dataset:
        filename = Path(sample.filepath)
        sample_root = filename.stem
        # Handle Metadata (CSV)
        if sample_root in metadata_map:
            data = metadata_map[sample_root]

            # add ecological variables columns
            # Use .get() to avoid KeyErrors if a column is missing
            for col in list_column_exhibit:
                sample[col] = data.get(col)

        # Handle Labels (YOLO)
        yolo_labels = parse_yolo_file(ann_path, sample_root)
        if yolo_labels:
            sample["ground_truth"] = yolo_labels
        
        # Save once at the end of the loop
        sample.save()


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

 
     
if __name__ == "__main__":
    ## construct dict of paths based on the unzip structure.
    root_dir = "/home/camarada/Documents/CDE/thesis/dataset_raw/DATASET"

    ## construct map dict with paths and tags for each sub dataset. 
    map_dict = find_sub_data_sources(root_dir)

    images_path = "/home/camarada/Documents/CDE/thesis/dataset_raw/DATASET/NC/images"
    ann_path = "/home/camarada/Documents/CDE/thesis/dataset_raw/DATASET/NC/labels_yolo"
    csv_path = "/home/camarada/Documents/CDE/thesis/dataset_raw/DATASET/NC/dugong_environmental_variables_NC.csv"

    name = "New Caledonia"

    list_medatata_columns = ['altitude',
                            'resolution', 'sea_state', 'turbidity_global', 
                            'turbidity_local', 'sun_glitter', 'cloud_reflection',
                            'habitat_type', 'background_complexity', 'coral', 'sand', 'dense_seagrass',
                            'open_sea','sparse_seagrass'
                            ]
    
    if name in fo.list_datasets():
        # Load the existing one from the DB (much faster!)
        dataset = fo.load_dataset(name)
        print("Dataset loaded from database.")
    else:
        # Create the FiftyOne dataset
        dataset = fo.Dataset.from_dir(
            dataset_dir=images_path,
            dataset_type=fo.types.ImageDirectory,
            name=name,
            overwrite=True
            )
        
        create_dataset(dataset, ann_path, csv_path, list_medatata_columns)
        dataset.persistent = True
        print(f"{name} - was created and saved as a persistent dataset.")

        ## compute metadata
        dataset.compute_metadata()
        dataset.save()

    ## debugg
    # dataset = fo.Dataset.from_dir(
    #         dataset_dir=images_path,
    #         dataset_type=fo.types.ImageDirectory,
    #         name=name,
    #         overwrite=True
    #     )
        
    # create_dataset(dataset, ann_path, csv_path, list_medatata_columns)
    # dataset.persistent = True
    # print("done")

    # Ensures that the App processes are safely launched on Windows
    session = fo.launch_app(dataset)
    # View the dataset's current App config
    print(dataset.app_config)
    # session.wait()