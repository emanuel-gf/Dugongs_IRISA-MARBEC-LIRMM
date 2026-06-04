"""
src/data/datamodule.py
======================
DugongDataModule — reads from resolved_paths.json
needed at training time.

JSON structure expected (from map_ids_to_paths.py output):
{
  "0": {
    "train": {
      "p5": {
        "aclr":   {"images": [...], "labels": [...], "metadata": [...]},
        "random": {"images": [...], "labels": [...], "metadata": [...]}
      },
      ...
    },
    "val":  {"images": [...], "labels": [...], "metadata": [...]},
    "test": {"images": [...], "labels": [...], "metadata": [...]}
  },
  ...
}

Usage
-----
    from datamodule import DugongDataModule

    dm = DugongDataModule(
        resolved_paths_json = "/path/to/resolved_paths.json",
        seed       = 0,
        partition  = "p10",
        method     = "aclr",          # "aclr" | "random"
        processor  = processor,
        batch_size = 8,
        augmentor  = None,            # optional albumentations Compose
        num_workers= 4,
    )
    dm.setup("fit")
    train_loader = dm.train_dataloader()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor

import logging
_log = logging.getLogger(__name__)
# ── Dataset ───────────────────────────────────────────────────────────────────

class DugongDataset(Dataset):
    """
    Patch-level dataset for RT-DETR fine-tuning.

    Follows the HuggingFace Transformers convention:
      - Images processed by AutoImageProcessor (returns pixel_values)
      - Labels in RT-DETR format: normalised [cx, cy, w, h] boxes
      - All fields required by the model are returned

    Augmentation
    ------------
    Augmentations are applied via an optional Albumentations Compose object.
    The compose must be configured with bboxes in COCO format:

        augmentor = A.Compose(
            [...],
            bbox_params=A.BboxParams(
                format="coco",
                label_fields=["class_labels"],
                min_visibility=0.1,
            )
        )

    Note: OT colour transfer will be added here as a custom Albumentations
    transform when implemented.

    Parameters
    ----------
    image_filepaths : list of image paths (sorted)
    label_filepaths : list of YOLO .txt label paths (sorted, 1:1 with images)
    processor       : HuggingFace AutoImageProcessor
    augmentor       : optional Albumentations Compose (COCO bbox format)
    """

    def __init__(
        self,
        image_filepaths: list,
        label_filepaths: list,
        processor: AutoImageProcessor,
        augmentor=None,
        metadata_list: list | None = None
    ):
        self.image_filepaths = [Path(p) for p in sorted(image_filepaths)]
        self.label_filepaths = [Path(p) for p in sorted(label_filepaths)]

        assert len(self.image_filepaths) == len(self.label_filepaths), (
            f"Image/label count mismatch: "
            f"{len(self.image_filepaths)} vs {len(self.label_filepaths)}"
        )

        self.processor = processor
        self.augmentor = augmentor

        # metadata_list: one dict per image with tile geometry (x_start, y_start,)
        # Indices are sorted in parallel with image_filepaths.
        self.metadata_list = metadata_list or [{}] * len(self.image_filepaths)

    def __len__(self) -> int:
        return len(self.image_filepaths)

    def __getitem__(self, idx: int):
        # ── Load image ────────────────────────────────────────────────────
        image_path = self.image_filepaths[idx]
        image      = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size

        # ── Load YOLO labels → COCO absolute pixels ───────────────────────
        label_path       = self.label_filepaths[idx]
        coco_annotations = []

        if label_path.exists():
            try:
                with open(label_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cls_id, xc, yc, w, h = map(float, parts)
                        x_min  = (xc - w / 2) * img_w
                        y_min  = (yc - h / 2) * img_h
                        abs_w  = w * img_w
                        abs_h  = h * img_h
                        coco_annotations.append({
                            "bbox":        [x_min, y_min, abs_w, abs_h],
                            "category_id": int(cls_id),
                        })
            except Exception as e:
                _log.info(f"  [DugongDataset] Warning: failed to read {label_path}: {e}")

        # ── Augmentation (Albumentations, COCO format) ────────────────────
        if self.augmentor and coco_annotations:
            try:
                image_np     = np.array(image)
                bboxes       = [ann["bbox"] for ann in coco_annotations]
                class_labels = [ann["category_id"] for ann in coco_annotations]

                result       = self.augmentor(
                    image=image_np,
                    bboxes=bboxes,
                    class_labels=class_labels,
                )
                image        = Image.fromarray(result["image"])
                bboxes       = result["bboxes"]
                class_labels = result["class_labels"]

            except Exception as e:
                _log.info(f"  [DugongDataset] Augmentation failed for {image_path}: {e}")
                bboxes       = [ann["bbox"] for ann in coco_annotations]
                class_labels = [ann["category_id"] for ann in coco_annotations]
        else:
            bboxes       = [ann["bbox"] for ann in coco_annotations]
            class_labels = [ann["category_id"] for ann in coco_annotations]

        # ── Processor: image only (no annotations argument) ───────────────
        # RT-DETR expects labels in its own format, not via the processor.
        encoding     = self.processor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)

        # ── Convert COCO [x_min, y_min, w, h] → RT-DETR [cx, cy, w, h] normalised
        boxes = []
        for (x_min, y_min, w, h) in bboxes:
            cx     = (x_min + w / 2) / img_w
            cy     = (y_min + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h
            boxes.append([cx, cy, w_norm, h_norm])

        num_boxes = len(boxes)

        labels = {
            # RT-DETR expects normalised cxcywh
            "boxes": (
                torch.tensor(boxes, dtype=torch.float32)
                if num_boxes > 0
                else torch.zeros((0, 4), dtype=torch.float32)
            ),
            "class_labels": (
                torch.tensor(class_labels, dtype=torch.int64)
                if num_boxes > 0
                else torch.zeros(0, dtype=torch.int64)
            ),
            # Required by the model
            "image_id":  torch.tensor([idx]),
            "orig_size": torch.tensor([img_h, img_w], dtype=torch.int64),
            "area": (
                torch.tensor(
                    [w * h for (_, _, w, h) in bboxes], dtype=torch.float32
                )
                if num_boxes > 0
                else torch.zeros(0, dtype=torch.float32)
            ),
            "iscrowd": torch.zeros(num_boxes, dtype=torch.int64),
        }

        return pixel_values, labels, str(image_path), self.metadata_list[idx]


# ── Collate ───────────────────────────────────────────────────────────────────

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([item[0] for item in batch]),
        "labels":       [item[1] for item in batch],
        "filepaths":    [item[2] for item in batch],
        "metadata":     [item[3] for item in batch],
    }


# ── DataModule ────────────────────────────────────────────────────────────────

class DugongDataModule(pl.LightningDataModule):
    """
    LightningDataModule for the FLPLAN dugong patch dataset.

    Reads file paths from a json containing the mapping of IDs to Full filepaths.
    (output of map_ids_to_paths.py).

    Parameters
    ----------
    resolved_paths_json : path to resolved_paths.json
    seed                : int — which seed split to use (e.g. 0, 63, 72)
    partition           : str — partition key e.g. "p5", "p10", "p100"
    method              : str — "aclr" or "random"
    processor           : HuggingFace AutoImageProcessor
    batch_size          : int
    augmentor           : optional Albumentations Compose (COCO bbox format)
    num_workers         : int
    pin_memory          : bool
    """

    def __init__(
        self,
        resolved_paths_json: str | Path,
        seed:       int,
        partition:  str,
        method:     str,
        processor:  AutoImageProcessor,
        batch_size: int  = 8,
        augmentor        = None,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()

        self.resolved_paths_json = Path(resolved_paths_json)
        self.seed        = seed
        self.partition   = partition
        self.method      = method
        self.processor   = processor
        self.batch_size  = batch_size
        self.augmentor   = augmentor
        self.num_workers = num_workers
        self.pin_memory  = pin_memory

        # Loaded in setup()
        self.train_dataset: Optional[DugongDataset] = None
        self.val_dataset:   Optional[DugongDataset] = None
        self.test_dataset:  Optional[DugongDataset] = None

    # ── Setup ─────────────────────────────────────────────────────────────

    def setup(self, stage: str | None = None) -> None:
        """
        Load file lists from resolved_paths.json and build datasets.
        Called by Lightning before each trainer stage.
        """
        with open(self.resolved_paths_json) as f:
            resolved = json.load(f)

        seed_str = str(self.seed)
        assert seed_str in resolved, (
            f"Seed '{seed_str}' not found in {self.resolved_paths_json}. "
            f"Available seeds: {list(resolved.keys())}"
        )
        seed_data = resolved[seed_str]

        # ── Train paths ───────────────────────────────────────────────────
        assert "train" in seed_data, (
            f"Key 'train' not found for seed {seed_str}. "
            f"Available keys: {list(seed_data.keys())}"
        )
        train_splits = seed_data["train"]

        assert self.partition in train_splits, (
            f"Partition '{self.partition}' not found for seed {seed_str}. "
            f"Available partitions: {list(train_splits.keys())}"
        )
        method_data = train_splits[self.partition]

        assert self.method in method_data, (
            f"Method '{self.method}' not found in partition '{self.partition}'. "
            f"Available methods: {list(method_data.keys())}"
        )
        train_data = method_data[self.method]

        train_images = train_data["images"]
        train_labels = train_data["labels"]
        train_metadata  = train_data.get("metadata", [{}] * len(train_images))

        # ── Val / test paths ──────────────────────────────────────────────
        val_data  = seed_data["val"]
        test_data = seed_data["test"]


        val_images    = val_data["images"]
        val_labels    = val_data["labels"]
        val_metadata  = val_data.get("metadata", [{}] * len(val_images))
        test_images   = test_data["images"]
        test_labels   = test_data["labels"]
        test_metadata = test_data.get("metadata", [{}] * len(test_images))

        # Validate counts
        assert len(train_images) == len(train_labels), (
            f"Train image/label mismatch: {len(train_images)} vs {len(train_labels)}"
        )
        assert len(val_images) == len(val_labels), (
            f"Val image/label mismatch: {len(val_images)} vs {len(val_labels)}"
        )
        assert len(test_images) == len(test_labels), (
            f"Test image/label mismatch: {len(test_images)} vs {len(test_labels)}"
        )

        print(
            f"  DataModule setup | seed={self.seed} | "
            f"partition={self.partition} | method={self.method}\n"
            f"    train={len(train_images)}  "
            f"val={len(val_images)}  "
            f"test={len(test_images)}"
        )

        if stage in ("fit", None):
            self.train_dataset = DugongDataset(
                train_images, train_labels,
                self.processor, augmentor=self.augmentor,
            )
            self.val_dataset = DugongDataset(
                val_images, val_labels,
                self.processor, augmentor=None,   # never augment val
            )

        if stage in ("validate", None):
            self.val_dataset = DugongDataset(
                val_images, val_labels,
                self.processor, augmentor=None,
                metadata_list=val_metadata,
            )
 
        if stage in ("test", None):
            self.test_dataset = DugongDataset(
                test_images, test_labels,
                self.processor, augmentor=None,   # never augment test
                metadata_list=test_metadata,
            )

    # ── DataLoaders ───────────────────────────────────────────────────────

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )