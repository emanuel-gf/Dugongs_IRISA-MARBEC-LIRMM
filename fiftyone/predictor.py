"""
Predicts a given Test set tag of the dataset. 
By the given model checkpoint. 
"""

from __future__ import annotations
import argparse
import json
import os
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
 

import datetime
import glob
 
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import pytorch_lightning as pl
# from loguru import logger
# from PIL import Image
# from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
# from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
# from pytorch_lightning.loggers import WandbLogger
# from dotenv import load_dotenv


# 5.  FIFTYONE INFERENCE HELPER
# ─────────────────────────────────────────────────────────────────────────────
def get_tile_metadata(filename: str, img_w: int, img_h: int, tile_size: int = 640, overlap: int = 100):
    """
    Recompute the actual tile start and size from the filename and image dimensions.
    """
    import re
    # Extract stride-based offsets from filename
    match = re.search(r"__tile_(\d+)_(\d+)", filename)
    if not match:
        return None  # Not a tiled image

    y_stride = int(match.group(1))
    x_stride = int(match.group(2))

    stride = tile_size - overlap

    # Recompute actual tile start and end
    x_end = min(x_stride + tile_size, img_w)
    y_end = min(y_stride + tile_size, img_h)

    x_start = x_end - tile_size if x_end - tile_size >= 0 else 0
    y_start = y_end - tile_size if y_end - tile_size >= 0 else 0

    tile_w = x_end - x_start
    tile_h = y_end - y_start

    return {
        "x_start": x_start,
        "y_start": y_start,
        "tile_w": tile_w,
        "tile_h": tile_h,
    }



def load_model(
    checkpoint_dir: str | Path,
    device: str | torch.device | None = None,
) -> tuple[RTDetrForObjectDetection, RTDetrImageProcessor, dict]:
    """
    Load an RT-DETR model and processor from a local checkpoint directory
    (the hf_export/ folder produced by a training run).
 
    Returns:
        model      – in eval mode, moved to *device*
        processor  – RTDetrImageProcessor
        id2label   – {int: str} label mapping from the model config
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
 
    checkpoint_dir = Path(checkpoint_dir)
    assert checkpoint_dir.exists(), f"Checkpoint dir not found: {checkpoint_dir}"
    
    ## ----------------------
    model = RTDetrForObjectDetection.from_pretrained(
        str(checkpoint_dir)
    )
    processor = RTDetrImageProcessor.from_pretrained(str(checkpoint_dir))
    model     = model.to(device).eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    print(f"Model loaded — labels: {id2label}")  # should print {0: 'dugong'}
    return model, processor, id2label


@torch.no_grad()
def run_inference(
    image_filepaths: list[str | Path],
    model: RTDetrForObjectDetection,
    processor: RTDetrImageProcessor,
    id2label: dict[int, str],
    confidence_threshold: float = 0.3,
    output_dir: str | Path | None = None,
    device: str | torch.device | None = None,
    tile_size: int = 640,
    overlap: int = 100,
) -> list[dict]:
    """
    Run inference on a list of image paths and write one JSON per image.
 
    Returns a list of result dicts:
        {
            "filepath":     str,
            "detections":   [{"label", "bounding_box", "confidence"}, ...],
            "tile_metadata": {...} | absent if not tiled
        }
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
 
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
 
    results: list[dict] = []
    print(f"Starting inference on {len(image_filepaths)} images  (device={device})")
 
    for path in image_filepaths:
        path         = Path(path)
        image        = Image.open(path).convert("RGB")
        w_img, h_img = image.size
 
        inputs  = processor(images=image, return_tensors="pt")
        outputs = model(pixel_values=inputs["pixel_values"].to(device))
 
        preds = processor.post_process_object_detection(
            outputs,
            target_sizes=[(h_img, w_img)],
            threshold=confidence_threshold,
        )[0]
 
        detections = [
            {
                "label":  "dugong",
                "bounding_box": [
                    x1 / w_img,
                    y1 / h_img,
                    (x2 - x1) / w_img,
                    (y2 - y1) / h_img,
                ],
                "confidence": round(score.item(), 6),
            }
            for score, label_id, (x1, y1, x2, y2) in zip(
                preds["scores"], preds["labels"],
                [b.tolist() for b in preds["boxes"]]
            )
            if label_id.item() == 0
        ]
 
        record: dict = {"filepath": str(path.resolve()), "detections": detections}
 
        tile_meta = get_tile_metadata(str(path), w_img, h_img, tile_size, overlap)
        if tile_meta:
            record["tile_metadata"] = tile_meta
 
        results.append(record)
 
        if output_dir is not None:
            json_path = (Path(output_dir) / path.name).with_suffix(".json")
            json_path.write_text(json.dumps(record, indent=2))
 
    print(f"Inference complete — {len(results)} images processed"
          + (f", JSONs → {output_dir}" if output_dir else ""))
    return results
 



SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
def collect_images(
    image_dir: str | Path | None = None,
    image_list: str | Path | None = None,
    max_images: int | None = None,
) -> list[Path]:
    """
    Build the list of image paths to run inference on.
 
    Priority:
      1. --image-list  (one filepath per line, lines starting with # ignored)
      2. --image-dir   (all supported images found recursively)
 
    Optionally cap with --max-images for quick subset runs.
    """
    if image_list is not None:
        paths = [
            Path(line.strip())
            for line in Path(image_list).read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    elif image_dir is not None:
        image_dir = Path(image_dir)
        paths = sorted(
            p for p in image_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    else:
        raise ValueError("Provide either --image-dir or --image-list")
 
    if max_images is not None:
        paths = paths[:max_images]
 
    assert paths, "No images found — check --image-dir / --image-list"
    return paths
 

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
    # print(f'images:{len(filepath_all_images)}')
    # print(f"labels:{len(filepath_all_labels)}")
    # print(f"metadata:{len(filepath_all_metadata)}")

    return filepath_all_images, filepath_all_labels, filepath_all_metadata


def collect_filepath_fiftyone(dataset,
                                tag_test_set,
                                region_set_field,
                                tile_images_folder,
                                )-> list[Path]:
    """
    Create a list of filepaths by filtering the dataset within sample_tags and for the given region, retrieving
    all filepaths who matches the query.
    """
    os.environ["FIFTYONE_DATABASE_URI"] = "mongodb://localhost:44123"
    import fiftyone as fo
    print(fo.core.odm.database.get_db_conn()) 
    from fiftyone import ViewField as F

    dataset = fo.load_dataset("dugong")

    assert len(dataset) != f"Didnt find the dataset. Availables:{fo.list_datasets()}"
    assert len(list(dataset.get_field_schema().keys())) > 30, f"Probably not loading the correct mongodb dataset. Actualy length = :{len(list(dataset.get_field_schema().keys()))} | \n {list(dataset.get_field_schema().keys())}"

    wp_test_set = (dataset
                .match_tags(tag_test_set)
               .match(F("region")==region_set_field)
               )
    list_filepath = wp_test_set.values('filepath')

    ## mapdict into the tiles
    test_list_images, test_list_labels, _ = mapdict_patches_filepath(
                list_filepath, tile_images_folder
                )
    return  test_list_images

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RT-DETR inference on a (sub)set of images."
    )
 
    # --- model ---
    parser.add_argument(
        "--checkpoint-dir", type=str, required=True,
        help="Local hf_export/ directory produced by a training run "
             "(contains config.json, model.safetensors, preprocessor_config.json, …)",
    )
    
    ## --fiftyone
    parser.add_argument('--dataset', type=str, help='name of the dataset')
    parser.add_argument('--tag-test', type=str, help='name of the tag present in the dataset')
    parser.add_argument('--region', type=str, help='Which region to filter the dataset. Either WP or NC ')
    # --- images ---
    # src = parser.add_mutually_exclusive_group(required=True)
    # src.add_argument(
    #     "--image-dir", type=str, default=None,
    #     help="Folder to search recursively for images (.jpg/.png/.tif).",
    # )
    # src.add_argument(
    #     "--image-list", type=str, default=None,
    #     help="Text file with one image filepath per line (lines starting "
    #          "with '#' are ignored).",
    # )
    # parser.add_argument(
    #     "--max-images", type=int, default=None,
    #     help="Cap the number of images (useful for a quick sanity check).",
    # )
 
    # --- inference settings ---
    parser.add_argument("--confidence", type=float, default=0.3,
                        help="Detection confidence threshold (default: 0.3).")
    parser.add_argument("--tile-size",  type=int,   default=640)
    parser.add_argument("--overlap",    type=int,   default=100)
 
    # --- output ---
    parser.add_argument("--output-dir", type=str, default="inference_results",
                        help="Directory where per-image JSON files are written.")
    parser.add_argument("--device",     type=str,   default=None,
                        help="'cuda', 'cpu', or leave blank for auto-detect.")
    parser.add_argument("--patch-folder", type=str, 
                        default="/share/home/e2406743/dataset/exported_img/seed_42",
                        help="Root folder containing images/, labels/, metadata/ subfolders"
                        )
    return parser.parse_args()

def get_run_name(checkpoint_path):
    return str(Path(checkpoint_path).parent.stem).strip()


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


def main() -> None:
    args = parse_args()
    print(args)
    # 1. Collect images
    image_paths = collect_filepath_fiftyone(
        dataset  = args.dataset,
        tag_test_set=args.tag_test,
        region_set_field=args.region,
        tile_images_folder = args.patch_folder 
    )
    
    print(f"Found {len(image_paths)} image(s) to process")
 
    # 2. Load model
    model, processor, id2label = load_model(
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )

    print(f"OUtput dir:{args.output_dir}")
    
    run_name = get_run_name(args.checkpoint_dir)
    run_name = str(run_name) + f'_conf_{int(args.confidence*100)}'
    # 3. Run inference
    run_inference(
        image_filepaths=image_paths,
        model=model,
        processor=processor,
        id2label=id2label,
        confidence_threshold=args.confidence,
        output_dir=Path(os.path.join(args.output_dir,run_name)),
        device=args.device,
        tile_size=args.tile_size,
        overlap=args.overlap,
    )
 
 
if __name__ == "__main__":
    main()