# rtdetr_lightning.py

from __future__ import annotations
 
import argparse
import datetime
import glob
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import pytorch_lightning as pl
import kornia.augmentation as K
from loguru import logger
from kornia.augmentation import AugmentationSequential, RandomHorizontalFlip, RandomVerticalFlip, RandomAffine
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb
from huggingface_hub import login as hf_login
from dotenv import load_dotenv

# LOGURU SETUP  — call once at startup; writes both to stderr and a dated file
def setup_logger(log_dir: str = "logs_logger", run_name: str = "run"):
    """
    Configure loguru: coloured stderr + rotating file in log_dir.
    Returns the path of the log file so it can be passed to W&B.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{run_name}.log"
 
    logger.remove()                                         # drop default handler
    logger.add(sys.stderr, level="DEBUG", colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
    logger.add(log_file,   level="DEBUG", rotation="50 MB",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}")
 
    logger.info(f"Logger initialised → {log_file}")
    return log_file


def check_hf_auth():
    """Validates Hugging Face token from environment variables."""
    hf_token = os.getenv("HUGGING_FACE_API")
    if not hf_token:
        logger.error("HUGGING_FACE_API not found in environment variables.")
        sys.exit(1)
    
    try:
        hf_login(token=hf_token)
        logger.success("Hugging Face authentication successful.")
    except Exception as e:
        logger.error(f"Hugging Face login failed: {e}")
        sys.exit(1)


def check_wandb_auth():
    """Validates Weights & Biases API key from environment variables."""
    wandb_key = os.getenv("WANDB_API_KEY")
    if not wandb_key:
        logger.error("WANDB_API_KEY not found in environment variables.")
        sys.exit(1)
    
    try:
        if wandb.login(key=wandb_key):
            logger.success("Weights & Biases authentication successful.")
        else:
            logger.error("Weights & Biases login failed (invalid key).")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Weights & Biases error: {e}")
        sys.exit(1)


### HELPERs -------------------------
def get_seed_from_filepath(csv_file):
    path = Path(csv_file).stem
    return path.split('_')[-1]


def return_list_from_csv(csv_file):
    dff = pd.read_csv(csv_file)
    wp_train_list = dff['train_wp'].dropna().values
    wp_test_list = dff['test_wp'].dropna().values
    wp_val_list = dff['val_wp'].dropna().values
    nc_train_list = dff['train_nc'].dropna().values
    nc_test_list =dff['test_nc'].dropna().values 
    nc_val_list = dff['val_nc'].dropna().values
    return wp_train_list, wp_test_list, wp_val_list, nc_train_list, nc_test_list, nc_val_list


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
    # print(f'images:{len(filepath_all_images)}')
    # print(f"labels:{len(filepath_all_labels)}")
    # print(f"metadata:{len(filepath_all_metadata)}")

    return filepath_all_images, filepath_all_labels, filepath_all_metadata


# ─────────────────────────────────────────────
# Dataset TORCH
class DugongDataset(Dataset):
    def __init__(self, list_image_filepath, list_label_filepath, processor):
        self.list_image_filepath = sorted(list_image_filepath)
        self.list_label_filepath = sorted(list_label_filepath)
        assert len(self.list_image_filepath) == len(self.list_label_filepath)
        self.processor = processor

    def __len__(self):
        return len(self.list_image_filepath)

    def __getitem__(self, idx):
        image = Image.open(self.list_image_filepath[idx]).convert("RGB")
        img_w, img_h = image.size

        annotations = []
        label_path = self.list_label_filepath[idx]
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls_id, xc, yc, w, h = map(float, parts)

                    ## convert from YOLO to normalized COCO
                    x_min = (xc - w / 2) * img_w
                    y_min = (yc - h / 2) * img_h
                    abs_w = w * img_w
                    abs_h = h * img_h

                    annotations.append({
                        "category_id": int(cls_id),
                        "bbox": [x_min, y_min, abs_w, abs_h], ## COCO ABSOLUTE PIXELS
                        "area": abs_w * abs_h,
                        "iscrowd": 0,
                    })

        encoding = self.processor(
            images=image,
            annotations={"image_id": idx, "annotations": annotations},
            return_tensors="pt",
        )
        return encoding["pixel_values"].squeeze(0), encoding["labels"][0]



# ─────────────────────────────────────────────
# DataModule
def collate_fn(batch):
        return {
            "pixel_values": torch.stack([item[0] for item in batch]),
            "labels": [item[1] for item in batch],
        }

class DugongDataModule(pl.LightningDataModule):
    """
    DataModule
 
    Params:
    ----------
    {train,val,test}_image_filepaths : sorted path lists for each split
    {train,val,test}_label_filepaths : matching YOLO .txt label lists
    processor                        : shared RTDetrImageProcessor
    batch_size                       : samples per GPU per step
    num_workers                      : DataLoader workers per split
    """
 
    def __init__(
        self,
        train_image_filepaths: list,
        train_label_filepaths: list,
        val_image_filepaths: list,
        val_label_filepaths: list,
        test_image_filepaths: list,
        test_label_filepaths: list,
        processor: RTDetrImageProcessor,
        batch_size: int = 8,
        num_workers: int = 4,
    ):
        super().__init__()
        self.train_image_filepaths = train_image_filepaths
        self.train_label_filepaths = train_label_filepaths
        self.val_image_filepaths   = val_image_filepaths ## the entrance path should be the test, name is switched
        self.val_label_filepaths   = val_label_filepaths
        self.test_image_filepaths  = test_image_filepaths
        self.test_label_filepaths  = test_label_filepaths
        self.processor             = processor
        self.batch_size            = batch_size
        self.num_workers           = num_workers
 
    def setup(self, stage: str | None = None):
        # 'stage' is set by Lightning: "fit", "validate", "test", or "predict"
        if stage in ("fit", None):
            self.train_dataset = DugongDataset(
                self.train_image_filepaths, self.train_label_filepaths, self.processor
            )
            self.val_dataset = DugongDataset(
                self.val_image_filepaths, self.val_label_filepaths, self.processor
            )
        if stage in ("validate", None):
            self.val_dataset = DugongDataset(
                self.val_image_filepaths, self.val_label_filepaths, self.processor
            )
        if stage in ("test", None):
            self.test_dataset = DugongDataset(
                self.test_image_filepaths, self.test_label_filepaths, self.processor
            )
 
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )
 
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,                # never shuffle val
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )
 
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,                # never shuffle test
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )

# ─────────────────────────────────────────────
# Box format contract
# ───────────────────
# RT-DETR labels store boxes as  [cx, cy, w, h]  normalised to [0,1].
# Kornia's "bbox" data_key expects [x1, y1, x2, y2] (xyxy, same scale).
# We convert → augment → convert back so neither side sees a wrong format.
 
def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """(..., 4) cxcywh → xyxy"""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2,
                         cx + w / 2, cy + h / 2], dim=-1)
 
def _xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """(..., 4) xyxy → cxcywh"""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2,
                         x2 - x1,       y2 - y1], dim=-1)


class DugongAugmentor(nn.Module):
    def __init__(self):
        super().__init__()
        self.augmentations = AugmentationSequential(
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            data_keys=["input", "bbox"],  # bbox expects (B, N, 4, 2) quadrilaterals
        )

    @torch.no_grad()
    def forward(self, images: torch.Tensor, boxes_cxcywh: torch.Tensor):
        """
        images      : (B, 3, H, W)
        boxes_cxcywh: (B, N, 4) normalised [0,1] cxcywh
        """
        _, _, H, W = images.shape
        scale = boxes_cxcywh.new_tensor([W, H, W, H])

        # cxcywh normalised → xyxy absolute pixels
        boxes_xyxy_abs = _cxcywh_to_xyxy(boxes_cxcywh) * scale  # (B, N, 4)

        # xyxy absolute → (B, N, 4, 2) quadrilaterals — what Kornia actually needs
        boxes_quad = self._xyxy_to_quad(boxes_xyxy_abs)          # (B, N, 4, 2)

        images_aug, boxes_quad_aug = self.augmentations(images, boxes_quad)

        # (B, N, 4, 2) → xyxy absolute → cxcywh normalised
        boxes_xyxy_abs_aug = self._quad_to_xyxy(boxes_quad_aug)  # (B, N, 4)
        boxes_aug = _xyxy_to_cxcywh(boxes_xyxy_abs_aug / scale).clamp(0.0, 1.0)

        return images_aug, boxes_aug

    @staticmethod
    def _xyxy_to_quad(boxes_xyxy: torch.Tensor) -> torch.Tensor:
        """(B, N, 4) xyxy → (B, N, 4, 2) quadrilateral corners: TL, TR, BR, BL"""
        x1, y1, x2, y2 = boxes_xyxy.unbind(dim=-1)
        return torch.stack([
            torch.stack([x1, y1], dim=-1),  # top-left
            torch.stack([x2, y1], dim=-1),  # top-right
            torch.stack([x2, y2], dim=-1),  # bottom-right
            torch.stack([x1, y2], dim=-1),  # bottom-left
        ], dim=-2)                           # (B, N, 4, 2)

    @staticmethod
    def _quad_to_xyxy(boxes_quad: torch.Tensor) -> torch.Tensor:
        """(B, N, 4, 2) quadrilateral → (B, N, 4) xyxy — fixed dim bug"""
        x1 = boxes_quad[..., 0].min(dim=-1)[0]  # min x across 4 corners
        y1 = boxes_quad[..., 1].min(dim=-1)[0]  # min y across 4 corners
        x2 = boxes_quad[..., 0].max(dim=-1)[0]  # max x across 4 corners
        y2 = boxes_quad[..., 1].max(dim=-1)[0]  # max y across 4 corners
        return torch.stack([x1, y1, x2, y2], dim=-1)
    
# ─────────────────────────────────────────────
# Lightning Module
class RTDETRLightningModule(pl.LightningModule):
    """
    training_step   → augment + forward + loss   (logged as train/*)
    validation_step → forward + loss             (logged as val/*)
    test_step       → forward + loss             (logged as test/*)
 
    val/loss drives ModelCheckpoint and EarlyStopping.
    test/* is computed once via trainer.test() after training.
    """
 
    def __init__(
        self,
        checkpoint: str = "PekingU/rtdetr_r50vd",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 50,
        id2label: dict | None = None,
        use_augmentation = False,
        early_stopping_patience: int = 10,
        confidence_threshold = 0.3,
         **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["id2label"])
 
        self.model = RTDetrForObjectDetection.from_pretrained(
            checkpoint)
        #self.model.train()
        self.augmentor = DugongAugmentor() if use_augmentation else None
        self.id2label  = id2label or {0: "dugong"}
        self._first_batch_done = False
        self.confidence_threshold = confidence_threshold
        # Assign any additional hyperparameters from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        ##MaP
        # one metric object per split — they accumulate across batches
        self.val_map  = MeanAveragePrecision(iou_type="bbox", 
                                             box_format="xyxy",
                                             max_detection_thresholds=[1,10,300],
                                             backend="faster_coco_eval",   # ← handles sparse area buckets correctly
                                             )
        self.test_map = MeanAveragePrecision(iou_type="bbox", box_format="xyxy",
                                             max_detection_thresholds=[1, 10, 300],
                                             backend="faster_coco_eval",   # ← handles sparse area buckets correctly
                                             )
    # ── box padding helpers ───────────────────────────────────────────────
 
    def _pad_boxes(self, labels):
        box_list = [lbl["boxes"] for lbl in labels]
        max_n = max(b.shape[0] for b in box_list)
        B = len(box_list)
        padded = torch.zeros(B, max_n, 4, device=self.device)
        mask   = torch.zeros(B, max_n, dtype=torch.bool, device=self.device)
        for i, boxes in enumerate(box_list):
            n = boxes.shape[0]
            padded[i, :n] = boxes
            mask[i, :n]   = True
        return padded, mask
 
    def _unpad_boxes(self, padded_boxes, mask, labels):
        for i, label in enumerate(labels):
            n = mask[i].sum().item()
            label["boxes"] = padded_boxes[i, :n]
 
    def _move_labels_to_device(self, raw_labels):
        return [{k: v.to(self.device) for k, v in lbl.items()} for lbl in raw_labels]
 
    def _eval_step(self, batch):
        pixel_values = batch["pixel_values"]
        labels       = self._move_labels_to_device(batch["labels"])
        outputs      = self(pixel_values, labels)
        ## return outputs.loss, outputs.loss_dict
        return outputs, labels
    
    # def _collect_map_inputs(self, outputs, labels, confidence_threshold
    #                         ):
    #     """
    #     Convert raw RT-DETR outputs + ground-truth labels into the
    #     list-of-dicts format that torchmetrics MeanAveragePrecision expects.

    #     torchmetrics preds format (one dict per image):
    #         boxes  : (M, 4) xyxy  absolute pixels  — FloatTensor
    #         scores : (M,)                           — FloatTensor
    #         labels : (M,)                           — IntTensor

    #     torchmetrics targets format (one dict per image):
    #         boxes  : (N, 4) xyxy  absolute pixels  — FloatTensor
    #         labels : (N,)                           — IntTensor
    #     """
    #     # logits: (B, num_queries, num_classes)
    #     # pred_boxes: (B, num_queries, 4) cxcywh normalised
    #     scores_all = outputs.logits.sigmoid()           # (B, Q, C)
    #     boxes_all  = outputs.pred_boxes                 # (B, Q, 4) cxcywh norm

    #     preds   = []
    #     targets = []

    #     for i, lbl in enumerate(labels):
    #         # ── image size from label (stored by processor as [h, w]) ──
    #         h, w = lbl["orig_size"].tolist()

    #         # ── predictions ──────────────────────────────────────────
    #         scores_i, classes_i = scores_all[i].max(dim=-1)   # (Q,), (Q,)
    #         keep = scores_i > confidence_threshold

    #         boxes_norm  = boxes_all[i][keep]                   # (M, 4) cxcywh norm
    #         boxes_xyxy  = _cxcywh_to_xyxy(boxes_norm)          # (M, 4) xyxy norm
    #         scale       = boxes_norm.new_tensor([w, h, w, h])
    #         boxes_abs   = (boxes_xyxy * scale).clamp(0)        # (M, 4) xyxy abs pixels

    #         preds.append({
    #             "boxes":  boxes_abs.cpu(),
    #             "scores": scores_i[keep].cpu(),
    #             "labels": classes_i[keep].int().cpu(),
    #         })

    #         # ── ground truth ─────────────────────────────────────────
    #         gt_boxes_norm = lbl["boxes"]                        # (N, 4) cxcywh norm
    #         gt_boxes_abs  = (_cxcywh_to_xyxy(gt_boxes_norm) * scale).clamp(0)

    #         targets.append({
    #             "boxes":  gt_boxes_abs.cpu(),
    #             "labels": lbl["class_labels"].int().cpu(),
    #         })

    #     return preds, targets

    def _collect_map_inputs(self, outputs, labels, confidence_threshold=0.01):
        scores_all = outputs.logits.sigmoid()
        boxes_all  = outputs.pred_boxes

        preds, targets = [], []

        for i, lbl in enumerate(labels):
            h, w = lbl["orig_size"].tolist()
            scale = boxes_all.new_tensor([w, h, w, h])

            scores_i, classes_i = scores_all[i].max(dim=-1)

            # low guard only — drops near-zero noise, doesn't truncate the PR curve
            keep = scores_i > 0.01
            boxes_abs = (_cxcywh_to_xyxy(boxes_all[i]) * scale).clamp(0)

            preds.append({
                "boxes":  boxes_abs[keep].cpu(),
                "scores": scores_i[keep].cpu(),
                "labels": classes_i[keep].int().cpu(),
            })

            gt_boxes_abs = (_cxcywh_to_xyxy(lbl["boxes"]) * scale).clamp(0)
            targets.append({
                "boxes":  gt_boxes_abs.cpu(),
                "labels": lbl["class_labels"].int().cpu(),
            })

        return preds, targets
    
        # # ── forward ──────────────────────────────────────────────────────────
    def forward(self, pixel_values, labels=None):
        return self.model(pixel_values=pixel_values, labels=labels)

    ## WEIGHT AND BIAS 
    def _log_loss_dict(self, prefix: str, loss: torch.Tensor, loss_dict: dict):
        """
        Log losses to W&B with a clean, epoch-only grouping:
 
        W&B panel layout
        ────────────────
        {prefix}/loss              ← total loss (main progress metric)
        {prefix}/main/loss_vfl     ← varifocal classification loss
        {prefix}/main/loss_bbox    ← L1 box regression loss
        {prefix}/main/loss_giou    ← GIoU box loss
        {prefix}/aux/loss_vfl      ← mean across all auxiliary heads
        {prefix}/aux/loss_bbox
        {prefix}/aux/loss_giou
 
        Auxiliary losses (loss_*_aux_N) are averaged across decoder layers
        so you get one clean line per loss type instead of 6 noisy ones.
        on_step=False everywhere → W&B x-axis is always epoch, never step.
        """
        batch_size = self.trainer.datamodule.batch_size
        sync = prefix == "train"   # only reduce across GPUs during training
 
        # total loss
        self.log(f"{prefix}/loss", loss,
                 on_step=False, on_epoch=True, prog_bar=True,
                 sync_dist=sync, batch_size=batch_size)
 
        # split main vs aux
        main_losses, aux_accum = {}, {}
        for k, v in loss_dict.items():
            if "_aux_" in k:
                # e.g. "loss_vfl_aux_3" → base key "loss_vfl"
                base = k.split("_aux_")[0]          # "loss_vfl"
                aux_accum.setdefault(base, []).append(v)
            else:
                main_losses[k] = v
 
        for k, v in main_losses.items():
            self.log(f"{prefix}/main/{k}", v,
                     on_step=False, on_epoch=True,
                     sync_dist=sync, batch_size=batch_size)
 
        for base, vals in aux_accum.items():
            mean_val = torch.stack(vals).mean()
            self.log(f"{prefix}/aux/{base}", mean_val,
                     on_step=False, on_epoch=True,
                     sync_dist=sync, batch_size=batch_size)
    # ── training ─────────────────────────────────────────────────────────
    def on_train_epoch_start(self):
        self.model.train()
        ## new inclusion trying to avoid the model to forget past stages during training.    
        # freeze backbone BatchNorm — prevents catastrophic forgetting
        for module in self.model.model.backbone.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()

    def training_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = self._move_labels_to_device(batch["labels"])
 
        # ── first-batch debug ─────────────────────────────────────────────
        if not self._first_batch_done:
            total_boxes = sum(lbl["boxes"].shape[0] for lbl in labels)
            logger.debug("─" * 60)
            logger.debug(f"First batch sanity check  (epoch {self.current_epoch})")
            logger.debug(f"  pixel_values : {list(pixel_values.shape)}  dtype={pixel_values.dtype}")
            logger.debug(f"  pixel range  : [{pixel_values.min():.3f}, {pixel_values.max():.3f}]")
            logger.debug(f"  batch size   : {len(labels)} images")
            logger.debug(f"  total boxes  : {total_boxes}  "
                         f"(avg {total_boxes / len(labels):.1f} per image)")
            for i, lbl in enumerate(labels):
                logger.debug(f"    img[{i}]  boxes={list(lbl['boxes'].shape)}  "
                             f"classes={lbl['class_labels'].tolist()}")
            logger.debug("─" * 60)
            self._first_batch_done = True


        if self.augmentor and any(lbl["boxes"].shape[0] > 0 for lbl in labels):
            padded_boxes, mask = self._pad_boxes(labels)          # (B, N, 4) cxcywh normalised
            pixel_values, aug_boxes = self.augmentor(pixel_values, padded_boxes)
            self._unpad_boxes(aug_boxes, mask, labels)

        ## output
        outputs = self(pixel_values, labels)

        # ── first-batch prediction debug ──────────────────────────────────
        if not self._first_batch_done:
            logger.debug("─" * 60)
            logger.debug("First batch RAW PREDICTIONS (before post-processing)")

            # ── only peek at 2 images, move immediately to CPU to avoid OOM ──
            n_debug = min(2, len(labels))
            logits_cpu    = outputs.logits[:n_debug].detach().float().cpu()   # float() in case of fp16
            pred_boxes_cpu = outputs.pred_boxes[:n_debug].detach().float().cpu()

            scores     = logits_cpu.sigmoid()                  # (n_debug, num_queries, num_classes)
            top_scores, top_classes = scores.max(dim=-1)       # (n_debug, num_queries)

            for i in range(n_debug):
                topk_scores, topk_idx = top_scores[i].topk(min(5, top_scores[i].shape[0]))
                topk_boxes   = pred_boxes_cpu[i][topk_idx]
                topk_classes = top_classes[i][topk_idx]

                logger.debug(f"  img[{i}] — top-5 queries:")
                for rank, (sc, cls, box) in enumerate(
                        zip(topk_scores.tolist(), topk_classes.tolist(), topk_boxes.tolist())):
                    logger.debug(f"    [{rank}] score={sc:.4f}  class={cls}  "
                                f"box(cxcywh)=[{box[0]:.3f}, {box[1]:.3f}, "
                                f"{box[2]:.3f}, {box[3]:.3f}]")

            flat_scores = top_scores.flatten()
            logger.debug(f"  Score stats (first {n_debug} imgs): "
                        f"min={flat_scores.min():.4f}  max={flat_scores.max():.4f}  "
                        f"mean={flat_scores.mean():.4f}  median={flat_scores.median():.4f}")
            logger.debug(f"  Queries > 0.3 : {(flat_scores > 0.3).sum()} / {flat_scores.numel()}")
            logger.debug(f"  Queries > 0.1 : {(flat_scores > 0.1).sum()} / {flat_scores.numel()}")
            logger.debug("─" * 60)

            self._first_batch_done = True

        self._log_loss_dict("train", outputs.loss, outputs.loss_dict)
        return outputs.loss
 
 
    #     return loss
    def validation_step(self, batch, batch_idx):
        outputs, labels = self._eval_step(batch)
        self._log_loss_dict("val", outputs.loss, outputs.loss_dict)

        # accumulate mAP inputs — detached, on CPU inside _collect_map_inputs
        with torch.no_grad():
            preds, targets = self._collect_map_inputs(
                outputs, labels, self.confidence_threshold
            )
        self.val_map.update(preds, targets)

        return outputs.loss
    
    def on_validation_epoch_end(self):
        map_result = self.val_map.compute()


        self.log("val/mAP",    map_result["map"],    prog_bar=True, sync_dist=False)
        self.log("val/mAP_50", map_result["map_50"], prog_bar=True, sync_dist=False)
        self.log("val/mAP_75", map_result["map_75"],               sync_dist=False)

        logger.info(
            f"Epoch {self.current_epoch} — "
            f"val/mAP={map_result['map']:.4f}  "
            f"val/mAP_50={map_result['map_50']:.4f}  "
            f"val/mAP_75={map_result['map_75']:.4f}"
        )

                # log all sub-metrics to diagnose the -1
        logger.info(f"  map_small  = {map_result['map_small']:.4f}")
        logger.info(f"  map_medium = {map_result['map_medium']:.4f}")
        logger.info(f"  map_large  = {map_result['map_large']:.4f}")
        logger.info(f"  mar_1      = {map_result['mar_1']:.4f}")
        logger.info(f"  mar_10     = {map_result['mar_10']:.4f}")
        logger.info(f"  mar_300    = {map_result['mar_300']:.4f}")
        self.val_map.reset()   # ← must reset after every epoch

# ── test ─────────────────────────────────────────────────────────────

    def test_step(self, batch, batch_idx):
        outputs, labels = self._eval_step(batch)
        self._log_loss_dict("test", outputs.loss, outputs.loss_dict)

        with torch.no_grad():
            preds, targets = self._collect_map_inputs(
                outputs, labels, self.confidence_threshold
            )
        self.test_map.update(preds, targets)

        return outputs.loss

    def on_test_epoch_end(self):
        map_result = self.test_map.compute()

        self.log("test/mAP",    map_result["map"],    sync_dist=False)
        self.log("test/mAP_50", map_result["map_50"], sync_dist=False)
        self.log("test/mAP_75", map_result["map_75"], sync_dist=False)

        logger.info(
            f"Test — "
            f"mAP={map_result['map']:.4f}  "
            f"mAP_50={map_result['map_50']:.4f}  "
            f"mAP_75={map_result['map_75']:.4f}"
        )
        self.test_map.reset()

    # ── optimizer ────────────────────────────────────────────────────────
 
    def configure_optimizers(self):
        # Separate backbone and head learning rates
        backbone_params = [p for n, p in self.model.named_parameters() 
                        if "backbone" in n]
        head_params = [p for n, p in self.model.named_parameters() 
                    if "backbone" not in n]

        optimizer = AdamW([
            {"params": backbone_params, "lr": self.hparams.lr * 0.1},  # 1e-5
            {"params": head_params,     "lr": self.hparams.lr},         # 1e-4
        ], weight_decay=self.hparams.weight_decay)

        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=3)
        cosine = CosineAnnealingLR(optimizer, 
                                   T_max=self.hparams.max_epochs // 2, 
                                   eta_min=1e-6
                                   )
        scheduler = SequentialLR(optimizer, 
                                 schedulers=[warmup, cosine], 
                                 milestones=[3])
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, 
                             "interval": "epoch"},
        }
 
 

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

@torch.no_grad()
def run_inference(
    image_filepaths: list,
    lightning_module: RTDETRLightningModule,
    processor: RTDetrImageProcessor,
    confidence_threshold: float = 0.3,
    output_dir: str | Path | None = None,
    device: str | torch.device | None = None,
    tile_size: int = 640,  # Default tile size
    overlap: int = 100,    # Default overlap
) -> list[dict]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir,exist_ok=True)
    model    = lightning_module.model.to(device).eval()
    id2label = lightning_module.id2label
    results  = []
 
    logger.info(f"Starting inference on {len(image_filepaths)} images  (device={device})")
 
    for path in image_filepaths:
        path         = Path(path)
        image        = Image.open(path).convert("RGB")
        w_img, h_img = image.size
 
        inputs  = processor(images=image, return_tensors="pt")
        outputs = model(pixel_values=inputs["pixel_values"].to(device))
 
        preds = processor.post_process_object_detection(
            outputs, target_sizes=[(h_img, w_img)], threshold=confidence_threshold)[0]
 
        detections = []
        for score, label_id, box in zip(preds["scores"], preds["labels"], preds["boxes"]):
            x1, y1, x2, y2 = box.tolist()
            detections.append({
                "label":        id2label.get(label_id.item(), str(label_id.item())),
                "bounding_box": [x1/w_img, y1/h_img, (x2-x1)/w_img, (y2-y1)/h_img],
                "confidence":   round(score.item(), 6),
            })

        # --- Extract tile metadata ---
        tile_metadata = get_tile_metadata(
            str(path), w_img, h_img, tile_size=tile_size, overlap=overlap
        )
 
        record = {"filepath": str(path.resolve()), "detections": detections}

        if tile_metadata:
            record["tile_metadata"] = tile_metadata
        results.append(record)
 
        json_path = (Path(output_dir) / path.name if output_dir else path).with_suffix(".json")
        json_path.write_text(json.dumps(record, indent=2))
 
    dest = str(output_dir) if output_dir else "image folders"
    logger.success(f"Inference complete — {len(results)} images processed, JSONs → {dest}")
    return results


## ---------------------------------------------
## TRAIN 
def train(
    train_images, train_labels,
    val_images,   val_labels,
    test_images,  test_labels,
    use_augmentation:bool,
    run_name: str,                              # unified ID for W&B + checkpoints
    checkpoint: str     = "PekingU/rtdetr_r50vd",
    id2label: dict      = None,
    batch_size: int     = 8,
    max_epochs: int     = 50,
    lr: float           = 1e-4,
    weight_decay: float = 1e-3,
    output_dir: str     = "checkpoints",
    early_stopping_patience: int = 10,
    wandb_project: str  = "rtdetr-dugong",
    wandb_tags: list    = None,
    **kwargs
):
    id2label  = id2label or {0: "dugong"}
    ckpt_dir  = os.path.join(output_dir, run_name)   # checkpoints/schema_partition_timestamp/
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Run: {run_name}")
    logger.info(f"NEW Checkpoint dir: {ckpt_dir}")
 
    processor = RTDetrImageProcessor.from_pretrained(checkpoint)
    logger.success(f"Processor LOADED from '{checkpoint}'")
 
    lit_model = RTDETRLightningModule(
        checkpoint=checkpoint, 
        lr=lr, weight_decay=weight_decay,
        max_epochs=max_epochs, id2label=id2label, use_augmentation=use_augmentation,
        **kwargs
        )
    logger.success("Lightning module instantiated")
 
    data_module = DugongDataModule(
        train_image_filepaths=train_images, train_label_filepaths=train_labels,
        val_image_filepaths=val_images,     val_label_filepaths=val_labels,
        test_image_filepaths=test_images,   test_label_filepaths=test_labels,
        processor=processor, batch_size=batch_size)
    logger.info(f"DataModule ready  "
                f"(train={len(train_images)}  val={len(val_images)}  test={len(test_images)})")
 
    ckpt_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"{run_name}-{{epoch:02d}}-{{val/mAP_50:.4f}}",
        monitor="val/mAP_50",   # ← best detector, not best loss
        mode="max",             # ← higher is better
        save_top_k=3,
        save_last=False,
        #every_n_epochs=10,
    )
 
    wandb_logger = WandbLogger(
        project=wandb_project,
        name=run_name,          # same unified ID
        tags=wandb_tags or [],
        log_model=False,
        config=dict(checkpoint=checkpoint, lr=lr, weight_decay=weight_decay,
                    batch_size=batch_size, max_epochs=max_epochs,
                      id2label=id2label, early_stopping_patience=early_stopping_patience),
    )
    logger.info(f"W&B run: project='{wandb_project}'  name='{run_name}'")
 
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        logger=wandb_logger,
        callbacks=[
            ckpt_callback,
            EarlyStopping(monitor="val/mAP_50",
                        patience=early_stopping_patience,
                        mode="max"),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        log_every_n_steps=100,
        gradient_clip_val=0.1,
        enable_progress_bar=True,
    )
 
    logger.info("Starting training …")
    trainer.fit(lit_model,
                 datamodule=data_module)
    logger.success("Training complete")
 
    #logger.info("Evaluating best checkpoint on held-out test set …")
    #trainer.test(lit_model, datamodule=data_module, ckpt_path="best")
    #logger.success("Test evaluation complete")
 
    return trainer, lit_model, processor


# ─────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────
PARTITIONS = ['NC','partition_5', 'partition_10', 'partition_25', 'partition_50',
                'partition_75', 'partition_100', 'ACLR_partition_5',
                'ACLR_partition_10', 'ACLR_partition_25', 'ACLR_partition_50'
                ]
 
def parse_args():
    parser = argparse.ArgumentParser(description="Train RT-DETR on dugong patches")
    parser.add_argument('--schema', type=str, required=True)
    parser.add_argument("--partition",    type=str,
                         required=True, choices=PARTITIONS,
                        help="Training partition strategy")
    parser.add_argument("--csvfile",      type=str, required=True,
                        help="FULL PATH - CSV with train_nc / test_nc / val_nc / train_wp, test_wp, val_wp in columns")
    parser.add_argument("--csvpatches",   type=str, default=None,
                        help="Parquet file with pre-split patch filepaths (required for partition)")
    parser.add_argument("--patch-folder", type=str, 
                        default="/share/home/e2406743/dataset/exported_img/seed_42",
                        help="Root folder containing images/, labels/, metadata/ subfolders"
                        )
    parser.add_argument("--output-dir",   type=str, default="checkpoints")
    parser.add_argument("--output-inference",   type=str, default="inference", 
                        help="folder to store the inferences")
    parser.add_argument("--batch-size",   type=int, default=8)
    parser.add_argument("--max-epochs",   type=int, default=50)
    parser.add_argument("--early-stopping", type=int, default=10, 
                        help="Patience for early stopping", 
                        dest="early_stopping_patience"
                        )
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--wandb-project",type=str, default="rtdetr-dugong")
    parser.add_argument("--augment", action="store_true", default=False,
                        help="Enable data augmentation during training")
    parser.add_argument("--hf-repo",      type=str, default=None,
                        help="HF Hub repo id (e.g. 'username/rtdetr-dugong-nc'). "
                             "NC run pushes here; partition runs load from here.")
    parser.add_argument("--hf-revision",  type=str, default="nc-best",
                        help="Git revision (branch/tag) used when pushing and loading. "
                             "Keeps multiple experiments in the same repo without overwriting. "
                             "Default: 'nc-best'")
    parser.add_argument("--nc-checkpoint-dir", type=str, default=None,
                        help="Local directory containing NC weights saved with save_pretrained() "
                             "(e.g. checkpoints/schema_NC_MMDD_HHMM/hf_export). "
                             "Required for partition_* runs.")
    parser.add_argument("--save-nc-local", action="store_true", default=False,
                        help="After NC training, save weights locally to "
                             "<output-dir>/<run-name>/hf_export/ "
                             "(use the printed path as --nc-checkpoint-dir for partition runs)")
    return parser.parse_args()


def _save_and_push(
    model: RTDETRLightningModule,
    processor: RTDetrImageProcessor,
    local_dir: str,
    hf_repo: str,
    #hf_revision: str,
):
    """
    Save the HF model + processor locally with save_pretrained(), then push
    to the Hub on a named revision (git branch/tag).
 
    Why save_pretrained() instead of push_to_hub() directly?
    ─────────────────────────────────────────────────────────
    • model.model is the raw HF model inside the Lightning wrapper —
      save_pretrained() is the correct, stable API for it.
    • Saving locally first gives you a backup and lets you inspect the files
      (config.json, model.safetensors) before they leave the machine.
    • Pushing to a *revision* means multiple experiments can live in the same
      repo without overwriting each other. Fine-tuning jobs then load from
      that exact revision, so there is zero ambiguity about which weights
      they receive.
 
    On the Hub, revisions are ordinary git branches:
        https://huggingface.co/{hf_repo}/tree/{hf_revision}
    """
    token = os.environ.get("HUGGING_FACE_API")
    assert token, "Provide --hf-token or set HF_TOKEN env var"
 
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
 
    # ── save locally ─────────────────────────────────────────────────────
    logger.info(f"Saving model locally → {local_dir}")
    model.model.save_pretrained(local_dir)      # writes config.json + model.safetensors
    processor.save_pretrained(local_dir)        # writes preprocessor_config.json
    logger.success(f"Model saved to {local_dir}")
 
    # ── push to Hub on a named revision ──────────────────────────────────
    logger.info(f"Pushing to HF Hub → {hf_repo} ")
    model.model.push_to_hub(
        hf_repo,
        #revision=hf_revision,   # creates/updates a branch with this name
        token=token,
    )
    processor.push_to_hub(
        hf_repo,
        #revision=hf_revision,
        token=token,
    )
    logger.success(
        f"Model available at "
        f"https://huggingface.co/{hf_repo}"
    )
    

# Save locally only (no HF push)
def _save_local(
    model: RTDETRLightningModule,
    processor: RTDetrImageProcessor,
    local_dir: str,
):
    """
    Save model weights and processor config locally using save_pretrained().
    Use this after an NC run so partition_* runs can load from the folder
    via --nc-checkpoint-dir without needing to push to or pull from HF Hub.
 
    Writes:
        <local_dir>/config.json
        <local_dir>/model.safetensors
        <local_dir>/preprocessor_config.json
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
 
    logger.info(f"Saving model locally → {local_dir}")
    model.model.save_pretrained(local_dir)
    processor.save_pretrained(local_dir)
    logger.success(f"NC weights saved to {local_dir}  "
                   f"(pass this path as --nc-checkpoint-dir for partition runs)")


def main():
    args = parse_args()
 
    # ── validate args ─────────────────────────────────────────────────────
    assert Path(args.csvfile).exists(),       f"csvfile not found: {args.csvfile}"
    assert Path(args.patch_folder).exists(),  f"patch-folder not found: {args.patch_folder}"
    if args.partition != 'NC':
        assert args.csvpatches is not None,   "--csvpatches is required for partition run."
        assert Path(args.csvpatches).exists(), f"csvpatches not found: {args.csvpatches}"

    augmentation_flag = args.augment

    is_finetune = args.partition != 'NC'
 
    if is_finetune:
        assert args.csvpatches is not None, \
            "--csvpatches is required for partition_* runs"
        assert Path(args.csvpatches).exists(), \
            f"csvpatches not found: {args.csvpatches}"
        assert args.nc_checkpoint_dir is not None, \
            "--nc-checkpoint-dir is required for partition_* runs " \
            "(point it to the hf_export/ folder from your NC run)"
        assert Path(args.nc_checkpoint_dir).exists(), \
            f"--nc-checkpoint-dir not found: {args.nc_checkpoint_dir}"

    # ── unified run name: schema_partition_MMDD_HHMM ─────────────────────
    now = datetime.datetime.now()
    seed_number = get_seed_from_filepath(args.csvfile)
    if augmentation_flag:
        run_name     = f"{args.schema}_{args.partition}_SEED{seed_number}_augm_{now.strftime('%m%d_%H%M')}"
    else:
        run_name     = f"{args.schema}_{args.partition}_SEED{seed_number}_{now.strftime('%m%d_%H%M')}"
    

    # ── logger setup (file written to logs/<run_name>.log) ────────────────
    setup_logger(log_dir='/share/home/e2406743/code/Dugongs_IRISA-MARBEC-LIRMM/logs_logger/',
                    run_name=run_name
                    )
    logger.info(f"Run name: {run_name}")
    logger.info(f"Args: {vars(args)}")
    logger.warning(f"AUGMENTATION: {str(augmentation_flag)}")

    ## huggingface or local path
    if is_finetune:
        # Load weights from the local NC export folder.
        # RTDetrForObjectDetection.from_pretrained() accepts a local path
        # exactly the same as a HF repo id — no code change needed in the model.
        checkpoint = args.nc_checkpoint_dir
        logger.success(f"Fine-tune run: loading NC weights from local dir → {checkpoint}")
    else:
        checkpoint = "PekingU/rtdetr_r50vd"
        logger.success(f"NC run: loading base weights from HF Hub → {checkpoint}")
    
    # ── load split lists from CSV ─────────────────────────────────────────
    (wp_train_list, wp_test_list, wp_val_list, 
         nc_train_list, nc_test_list, nc_val_list) = return_list_from_csv(args.csvfile)
    logger.info(f"CSV loaded: {args.csvfile}")
 
    # ── resolve train images by partition ─────────────────────────────────
    match args.partition:
        case p if p.startswith("ACLR_"):
            logger.info(f"ACTIVE LEARNING - Partition:{p}")

            df = pd.read_parquet(args.csvpatches)
            train_list_images = df.loc["images", p]
            train_list_labels = df.loc["labels", p]
 
        case p if p.startswith("partition_"):
            logger.info(f"RANDOM SELECTION - Partition: {p}")

            df = pd.read_parquet(args.csvpatches)
            train_list_images = df.loc["images", p]
            train_list_labels = df.loc["labels", p]
        
        case "NC":
            logger.info("Partition:NC - train on NC val and test on NC")
            logger.info(f"Training the model to be saved on checkpoint.")

            ## map to the patch filepath
            train_list_images, train_list_labels, _ = mapdict_patches_filepath(
                nc_train_list, args.patch_folder
            ) 
        case _:
            raise ValueError(f"Unknown partition: {args.partition}")
 
    assert len(train_list_images) > 0, "train_list_images is empty — check your CSV / parquet"
    assert len(train_list_labels) > 0, "train_list_labels is empty — check your CSV / parquet"
    assert len(train_list_images) == len(train_list_labels), \
        f"Train image/label count mismatch: {len(train_list_images)} vs {len(train_list_labels)}"
    logger.success(f"Train set: {len(train_list_images)} images")
 
    # ── test & val ───────────────────────────────────
    match args.schema:
        case "NNN":
            test_list_images, test_list_labels, _ = mapdict_patches_filepath(
                nc_test_list, args.patch_folder
            )
            val_list_images, val_list_labels, _ = mapdict_patches_filepath(
                nc_val_list, args.patch_folder
            )
        case "NWW":
            test_list_images, test_list_labels, _ = mapdict_patches_filepath(
                wp_test_list,args.patch_folder
                )
        
            val_list_images, val_list_labels, _ = mapdict_patches_filepath(
                wp_val_list, args.patch_folder
                )
        case _:
            raise ValueError(f"Unknown schema: {args.schema}")


    assert len(test_list_images) > 0, "test_list_images is empty"
    assert len(test_list_images) == len(test_list_labels)
    logger.success(f"Test Set:{len(test_list_images)}")
    assert len(val_list_images) > 0, "val_list_images is empty"
    assert len(val_list_images) == len(val_list_labels)
    logger.success(f"Val set:{len(val_list_images)}")
 
    # ── train ─────────────────────────────────────────────────────────────
    trainer, model, processor = train(
        train_images=train_list_images, train_labels=train_list_labels,
        val_images=val_list_images,     val_labels=val_list_labels,
        test_images=test_list_images,   test_labels=test_list_labels,
        use_augmentation = augmentation_flag,
        run_name=run_name,
        checkpoint=checkpoint,
        id2label={0: "dugong"},
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        lr=args.lr,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
         wandb_tags=[args.schema, args.partition, "NC-pretrained" if is_finetune else "from-hub"],
         early_stopping_patience= args.early_stopping_patience
         )
 
    ## save to hugging face
    if not is_finetune:
        local_hf_dir = os.path.join(args.output_dir, run_name, "hf_export")
 
        if args.save_nc_local or args.hf_repo:
            # Always save locally first (needed for HF push too)
            _save_local(model=model, processor=processor, local_dir=local_hf_dir)
 
        if args.hf_repo:
            _save_and_push(
                model=model,
                processor=processor,
                local_dir=local_hf_dir,
                hf_repo=args.hf_repo,
            )

    # ── inferensce on test set → JSON files ────────────────────────────────
    logger.info("Running inference on test set …")
    run_inference(
        image_filepaths=test_list_images,
        lightning_module=model,
        processor=processor,
        output_dir=Path(os.path.join(args.output_inference,run_name)),
        confidence_threshold=0.1,
        tile_size=640,  # Match tiling parameters
        overlap=100,
    )


if __name__ == "__main__":
        ## check connections
    load_dotenv()
    check_hf_auth()
    check_wandb_auth()
    main()