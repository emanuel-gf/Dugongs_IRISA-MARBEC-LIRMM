# Dugong Dataset Preprocessing with FiftyOne

This project prepares a multi-mission dugong detection dataset for AI-engineered workflows using [FiftyOne](https://fiftyone.ai/), a powerful tool for dataset management and visualization.

## About Dataset

*NOT PUBLIC AVAILABLE* 

1. NC:

    The NC dataset is divided into independent flights conducted with a microlight airplane fitted with a GoPro camera. Flights occurred along pre-defined transects on different days and/or on different dates. You can find the trajectory of the flights on [Laura Mannocci’s github](https://github.com/LauraMannocci/dugong_mapping/tree/main/outputs). Successive frames are overlapping.

    The main author is [Laura Mannocci](https://github.com/LauraMannocci/). 

2. WP:

    The WP dataset is divided into independent missions that were conducted by drone (Phantom 4 and Phantom 4 Pro). Missions corresponds to flights that were conducted on different dates or that were tracking different individuals. Two types of flights were conducted: 

        • FPLAN flights: pre-defined transects assessing a given area. 
        • MAN flights: tracking of one or a couple of individuals on their trajectory.

    For the WP dataset, there are many authors from Indonesia as the data was collected as part of [SELAMAT International Laboratory](https://www.ird.fr/lmi-selamat-sentinel-laboratory-indonesian-marine-biodiversity).

## Overview

The `launch3.py` script processes raw YOLO-annotated dugong images from multiple missions and regions, enriches them with environmental metadata, and organizes them into a FiftyOne Dataset for analysis and training.

## What is FiftyOne Dataset?

FiftyOne's `Dataset` class is a structured container for managing computer vision data. It provides:

- **Centralized storage**: Organizes samples (images) and their associated metadata, annotations, and ground truth labels
- **Flexible metadata**: Stores custom fields per sample (environmental variables, tags, mission info, etc.)
- **Built-in tools**: Supports object detection, classification, segmentation, and more annotation formats
- **Persistence**: Can save datasets to disk for reproducibility and sharing
- **Interactive visualization**: Integrates with FiftyOne's App for easy data exploration

In this project, each `Sample` represents one dugong observation image, with:
- Image filepath
- YOLO bounding box annotations (converted to FiftyOne format)
- Environmental metadata (altitude, turbidity, habitat type, etc.)
- Organizational tags (mission name, region, subregion)

## Directory Structure & Organization

The script handles a nested folder hierarchy straight from the zip file.

```
dataset_raw/DATASET/
├── WP/                          # Western Pacific region
│   ├── GAM/                     # Subregion (Galapagos, Mariana, etc.)
│   │   ├── UM_M5/               # Mission folder
│   │   │   ├── images/          # .jpeg files
│   │   │   └── labels_yolo/     # .txt YOLO annotation files
│   │   └── ...
│   └── dugong_environmental_variables_WP.xlsx
│
└── NC/                          # New Caledonia
    ├── mission_folder/
    │   ├── images/
    │   └── labels_yolo/
    └── dugong_environmental_variables_NC.csv
```

### Processing Pipeline

1. **Discovery** (`find_sub_data_sources()`): Recursively finds all `images/` directories and pairs them with their sibling `labels_yolo/` folders, extracting region and mission metadata

2. **CSV Loading** (`load_df_from_csv()`): Loads environmental metadata from region-specific CSV/XLSX files

3. **CSV Preprocessing** (`preprocess_csv()`): Creates a lookup dictionary mapping image filenames to their environmental variables

4. **YOLO Conversion** (`parse_yolo_file()`): Converts YOLO format `[x_center, y_center, width, height]` (normalized) to FiftyOne format `[top-left-x, top-left-y, width, height]`

5. **Dataset Population**: For each image:
   - Create a FiftyOne `Sample` with the image filepath
   - Add organizational metadata (tags, region, subregion, mission)
   - Attach environmental variables as custom fields
   - Attach YOLO detections as ground truth bounding boxes

## Quick Start Example

### Dependencies

Use uv to manage dependencies:

    1. cd to the project folder
    ```
    cd your_project_folder
    ```

    2. Start a new venv inside the folder 
    ```
    uv venv
    ```

    Activate the env
    ```
    source .venv/bin/activate ##Linux
    ```

    3. Sync the project
    ```
    uv sync
    ```


### Launch

```bash
python launch3.py
```

## Additional information

Ecological paper that relied on this dataset to map the population of [dugongs](https://onlinelibrary.wiley.com/doi/full/10.1002/aqc.4237)
---

**BIG NOTE** The dataset is not currently public available. 
