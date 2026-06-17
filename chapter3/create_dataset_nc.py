import os
from pathlib import Path
import argparse


# --- ARGPARSE SETUP ---
def get_args():
    parser = argparse.ArgumentParser(description="Create NC Dataset and Launch Session.")
    parser.add_argument("--root", "-r", help="Root directory of the NC dataset.",
                        default="/share/home/e2406743/dataset/dataset/NC")
    parser.add_argument("--port_database", help="Port for MongoDB", default="44123")
    parser.add_argument("--launch", "-l", help="Launch the FiftyOne App", default="false")
    parser.add_argument("--name_dataset", "-n", help="Dataset name", default="Domain-Shift-NC")
    return parser.parse_args()


def convert_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in ["true", "1", "yes"]


def parse_yolo_file(ann_path: str, sample_root: str):
    """
    Reads a single .txt file in labels_yolo/ and converts YOLO
    [class, x_c, y_c, w, h] (all normalised [0,1]) to FiftyOne
    [top-left-x, top-left-y, w, h].

    Class mapping: "0" -> "dugong" (lowercase, matching the NC label convention).

    Args:
        ann_path:    FOLDER str - path to the FOLDER containing YOLO .txt files
        sample_root: str        - stem of the image (no extension)
    """
    import fiftyone as fo

    file_path = os.path.join(ann_path, f"{sample_root}.txt")
    if not os.path.exists(file_path):
        return None

    detections = []
    with open(file_path, "r") as f:
        for line in f:
            ann = line.strip().split()
            if not ann:
                continue

            # Class mapping
            label = "dugong" if ann[0] == "0" else str(ann[0])

            # Dimensions
            w, h = float(ann[3]), float(ann[4])

            # Center to top-left
            top_left_x = float(ann[1]) - (w / 2)
            top_left_y = float(ann[2]) - (h / 2)

            detections.append(
                fo.Detection(label=label, bounding_box=[top_left_x, top_left_y, w, h])
            )

    return fo.Detections(detections=detections) if detections else None


def find_sub_data_sources(base_dir: str) -> list[dict]:
    """
    Walks the NC structure:
        <base_dir>/<Flight_XXX>/images   (+ sibling labels_yolo/)

    Returns a list of dicts with keys:
        images, labels, region, mission_name, parent_name
    """
    sources = []
    base_path = Path(base_dir)

    for img_dir in base_path.rglob("images"):
        parent_dir = img_dir.parent              # e.g. .../Flight_195
        label_dir  = parent_dir / "labels_yolo"   # NC uses labels_yolo/, not labels/

        if not label_dir.exists():
            continue

        flight_name = parent_dir.stem             # e.g. "Flight_195"

        sources.append({
            "images":       str(img_dir),
            "labels":       str(label_dir),
            "region":       "NC",
            "mission_name": flight_name,
            "parent_name":  flight_name,           # NC has no deeper sub-mission level
        })

    return sources


def main():
    args = get_args()

    # --- DATABASE CONFIG (must happen before importing fiftyone) ---
    os.environ["FIFTYONE_DATABASE_URI"] = f"mongodb://localhost:{args.port_database}"

    import fiftyone as fo

    try:
        existing = fo.list_datasets()
        print(f"Connected to MongoDB. Existing datasets: {existing}")
    except Exception as e:
        print(f"ERROR: Could not connect to MongoDB at localhost:{args.port_database}.\n{e}")
        return

    name_dataset = args.name_dataset
    root_dir     = args.root
    launch       = convert_bool(args.launch)
    assert isinstance(launch, bool), f"Expected bool for --launch, got {type(launch)}"

    sub_ds_list = find_sub_data_sources(root_dir)
    print(f"Found {len(sub_ds_list)} flight folder(s) under '{root_dir}'.")

    # --- LOAD OR CREATE DATASET ---
    if name_dataset in fo.list_datasets():
        dataset = fo.load_dataset(name_dataset)
        print(f"Loaded existing dataset '{name_dataset}' from database.")
    else:
        dataset = fo.Dataset(name=name_dataset, overwrite=True)
        dataset.persistent = True
        print(f"Created new persistent dataset '{name_dataset}'.")

        samples_to_add = []

        for sub_ds in sub_ds_list:
            image_dir  = Path(sub_ds["images"])
            label_dir  = sub_ds["labels"]
            region     = sub_ds["region"]
            flight     = sub_ds["mission_name"]

            print(f"  Processing: {flight}")

            # NC images are .jpeg (matches the original create_dataset.py glob)
            for img_path in image_dir.rglob("*.jpeg"):
                sample = fo.Sample(filepath=str(img_path))

                sample["region"]       = region
                sample["mission_name"] = flight
                sample["parent_name"]  = flight

                sample_root = img_path.stem
                yolo_labels = parse_yolo_file(label_dir, sample_root)
                if yolo_labels:
                    sample["ground_truth"] = yolo_labels

                samples_to_add.append(sample)

        print(f"Batching {len(samples_to_add)} sample(s) into dataset...")
        dataset.add_samples(samples_to_add, progress=True)
        dataset.compute_metadata()
        dataset.save()
        print(f"Done. Dataset '{name_dataset}' saved with {len(dataset)} samples.")

    # --- OPTIONAL APP LAUNCH ---
    if launch:
        session = fo.launch_app(dataset, remote=True)
        print(dataset.app_config)
        session.wait()


if __name__ == "__main__":
    main()