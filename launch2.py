import fiftyone as fo

images_path = "/home/camarada/Documents/CDE/thesis/dataset_raw/DATASET/NC/images"
ann_path = "/home/camarada/Documents/CDE/thesis/dataset_raw/DATASET/NC/labels_yolo"

name = "New Caledonia"

# Create the FiftyOne dataset
dataset = fo.Dataset.from_dir(
    dataset_dir=images_path,
    dataset_type=fo.types.ImageDirectory,
    name=name,
)

dataset.import_dir(
    dataset_dir="/path/to/dataset", # Should contain /images and /labels folders
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="ground_truth",
)

if __name__ == "__main__":
    # Ensures that the App processes are safely launched on Windows
    session = fo.launch_app(dataset)
    session.wait()