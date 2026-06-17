import os
from pathlib import Path
import argparse
from PIL import Image


# --- ARGPARSE SETUP ---
def get_args():
    parser = argparse.ArgumentParser(description="Create Dataset and Launch Session.")
    parser.add_argument("--root", "-r", help="Root directory of the dataset.",
                        default="/share/home/e2406743/dataset/wp_improved/new_WP")
    parser.add_argument("--port_database", help="Port for MongoDB", default="44123")
    parser.add_argument("--launch", "-l", help="Launch the FiftyOne App", default="false")
    parser.add_argument("--name_dataset", "-n", help="Dataset name", default="Domain-Shift-WP")
    parser.add_argument("--thumbnails", "-t", help="Generate 400x400 thumbnails for the App grid",
                        default="false")
    parser.add_argument("--thumbnail_dir", help="Directory to store thumbnail images",
                        default="/tmp/thumbnails_wp")
    return parser.parse_args()


def convert_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in ["true", "1", "yes"]


def parse_pixel_label_file(label_path: Path, img_width: int, img_height: int):
    """
    Reads a single .txt annotation file whose format is:
        x1 y1 x2 y2 ClassName
    where coordinates are absolute pixel values (top-left + bottom-right corners).

    Converts to FiftyOne normalized [top_left_x, top_left_y, w, h] in [0, 1].

    Args:
        label_path:  Path to the .txt annotation file.
        img_width:   Width of the corresponding image in pixels.
        img_height:  Height of the corresponding image in pixels.

    Returns:
        fo.Detections or None if the file does not exist / is empty.
    """
    import fiftyone as fo

    if not label_path.exists():
        return None

    detections = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            x1, y1, x2, y2 = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
            class_name = " ".join(parts[4:])  # handles multi-word class names

            # Normalise to [0, 1] for FiftyOne
            norm_x = x1 / img_width
            norm_y = y1 / img_height
            norm_w = (x2 - x1) / img_width
            norm_h = (y2 - y1) / img_height

            detections.append(
                fo.Detection(
                    label=class_name,
                    bounding_box=[norm_x, norm_y, norm_w, norm_h],
                )
            )

    return fo.Detections(detections=detections) if detections else None


def find_sub_data_sources(base_dir: str) -> list[dict]:
    """
    Walks the new_WP structure:
        <base_dir>/<MISSION>/<MISSION_MX>/images   (+ sibling labels/)

    Returns a list of dicts with keys:
        images, labels, region, mission_name, parent_name
    """
    sources = []
    base_path = Path(base_dir)

    for img_dir in base_path.rglob("images"):
        parent_dir = img_dir.parent          # e.g. .../GAM/GAM_M1
        label_dir  = parent_dir / "labels"   # sibling labels/ folder

        if not label_dir.exists():
            continue

        # Infer mission and parent from the directory hierarchy
        mission_name = parent_dir.parent.stem   # e.g. "GAM"
        parent_name  = parent_dir.stem          # e.g. "GAM_M1"

        sources.append({
            "images":       str(img_dir),
            "labels":       str(label_dir),
            "region":       "NC",
            "mission_name": mission_name,
            "parent_name":  parent_name,
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

    name_dataset  = args.name_dataset
    root_dir      = args.root
    launch        = convert_bool(args.launch)
    thumbnails    = convert_bool(args.thumbnails)
    thumbnail_dir = args.thumbnail_dir
    assert isinstance(launch, bool),     f"Expected bool for --launch, got {type(launch)}"
    assert isinstance(thumbnails, bool), f"Expected bool for --thumbnails, got {type(thumbnails)}"

    sub_ds_list = find_sub_data_sources(root_dir)
    print(f"Found {len(sub_ds_list)} sub-dataset(s) under '{root_dir}'.")

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
            label_dir  = Path(sub_ds["labels"])
            region     = sub_ds["region"]
            mission    = sub_ds["mission_name"]
            parent     = sub_ds["parent_name"]

            print(f"  Processing: {mission} / {parent}")

            for img_path in image_dir.rglob("*.jpeg"):
                # --- Use PIL to get real image dimensions ---
                try:
                    with Image.open(img_path) as pil_img:
                        img_width, img_height = pil_img.size
                except Exception as e:
                    print(f"    WARNING: Could not open {img_path}: {e}. Skipping.")
                    continue

                sample = fo.Sample(filepath=str(img_path))

                # Metadata fields
                sample["region"]       = region
                sample["mission_name"] = mission
                sample["parent_name"]  = parent

                # Labels
                label_file = label_dir / f"{img_path.stem}.txt"
                detections = parse_pixel_label_file(label_file, img_width, img_height)
                if detections:
                    sample["ground_truth"] = detections

                samples_to_add.append(sample)

        print(f"Batching {len(samples_to_add)} sample(s) into dataset…")
        dataset.add_samples(samples_to_add, progress=True)
        dataset.compute_metadata()
        dataset.save()
        print(f"Done. Dataset '{name_dataset}' saved with {len(dataset)} samples.")

    # --- OPTIONAL THUMBNAIL GENERATION ---
    if thumbnails:
        import fiftyone.utils.image as foui

        print(f"Generating 400x400 thumbnails -> '{thumbnail_dir}' ...")
        foui.transform_images(
            dataset,
            size=(400, 400),              # fixed 400x400 grid thumbnails
            output_field="thumbnail_path",
            output_dir=thumbnail_dir,
            update_filepaths=True,
        )

        # Tell the App to use thumbnails in the grid, full image on click
        dataset.app_config.media_fields     = ["filepath", "thumbnail_path"]
        dataset.app_config.grid_media_field = "thumbnail_path"
        dataset.app_config.media_fallback   = True   # fall back to filepath if thumb missing
        dataset.save()
        print("Thumbnails ready. Grid view -> thumbnail_path | Modal -> filepath")

    # --- OPTIONAL APP LAUNCH ---
    if launch:
        session = fo.launch_app(dataset, remote=True)
        print(dataset.app_config)
        session.wait()


if __name__ == "__main__":
    main()