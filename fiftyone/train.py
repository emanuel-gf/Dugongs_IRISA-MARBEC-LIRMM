# rtdetr_lightning.py

from __future__ import annotations
 
import argparse
import datetime
import glob
import json
import os
import sys
from pathlib import Path
 
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
import kornia.augmentation as K
from loguru import logger
from kornia.augmentation import AugmentationSequential, RandomHorizontalFlip, RandomVerticalFlip, RandomAffine

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb
from huggingface_hub import login as hf_login
from dotenv import load_dotenv

# LOGURU SETUP  — call once at startup; writes both to stderr and a dated file
def setup_logger(log_dir: str = "logs", run_name: str = "run"):
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
    wp_train_list = dff['train_seed'].dropna().values
    test_list = dff['test_seed'].dropna().values
    val_list = dff['val_seed'].dropna().values
    nc_train_list = dff['train_nc'].dropna().values
    return wp_train_list, nc_train_list, test_list, val_list

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
# Dataset

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

        annotations = []
        label_path = self.list_label_filepath[idx]
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls_id, xc, yc, w, h = map(float, parts)
                    annotations.append({
                        "category_id": int(cls_id),
                        "bbox": [xc, yc, w, h],
                        "area": w * h,
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

def xyxy_to_quadrilateral(boxes_xyxy):
    """
    boxes_xyxy: [B, N, 4] or [N, 4] in [x1, y1, x2, y2] format
    Returns: [B, N, 4, 2] or [N, 4, 2] quadrilaterals
    """
    x1, y1, x2, y2 = boxes_xyxy.unbind(dim=-1)
    quadrilaterals = torch.stack([
        torch.stack([x1, y1], dim=-1),  # top-left
        torch.stack([x2, y1], dim=-1),  # top-right
        torch.stack([x2, y2], dim=-1),  # bottom-right
        torch.stack([x1, y2], dim=-1),  # bottom-left
    ], dim=-2)
    return quadrilaterals

def quadrilateral_to_xyxy(boxes_quad):
    """
    boxes_quad: [B, N, 4, 2] or [N, 4, 2] quadrilaterals
    Returns: [B, N, 4] or [N, 4] in [x1, y1, x2, y2] format
    """
    x_coords, _ = boxes_quad[..., 0], boxes_quad[..., 1]
    x1 = x_coords.min(dim=-1)[0]
    y1 = boxes_quad[..., 1].min(dim=-1)[0]
    x2 = x_coords.max(dim=-1)[0]
    y2 = boxes_quad[..., 1].max(dim=-1)[0]
    return torch.stack([x1, y1, x2, y2], dim=-1)

class DugongAugmentor(nn.Module):
    def __init__(self):
        super().__init__()
        self.augmentations = AugmentationSequential(
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            data_keys=["input", "bbox"],
        )

    @torch.no_grad()
    def forward(self, images: torch.Tensor, boxes_xyxy: torch.Tensor):
        """
        images    : (B, 3, H, W)
        boxes_xyxy: (B, N, 4)  in [x1, y1, x2, y2] format

        Returns images and boxes both back in their original formats.
        """
        boxes_quad = xyxy_to_quadrilateral(boxes_xyxy)
        images_aug, boxes_quad_aug = self.augmentations(images, boxes_quad)
        boxes_xyxy_aug = quadrilateral_to_xyxy(boxes_quad_aug)
        return images_aug, boxes_xyxy_aug


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
        use_augmentation = False
    ):
        super().__init__()
        self.save_hyperparameters("lr", "weight_decay", "max_epochs")
 
        self.model     = RTDetrForObjectDetection.from_pretrained(checkpoint)
        self.augmentor = DugongAugmentor() if use_augmentation else None
        self.id2label  = id2label or {0: "dugong"}
        self._first_batch_done = False
 
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
        return outputs.loss, outputs.loss_dict
 
    # ── forward ──────────────────────────────────────────────────────────
 
    def forward(self, pixel_values, labels=None):
        return self.model(pixel_values=pixel_values, labels=labels)
 
    # ── training ─────────────────────────────────────────────────────────
 
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

        # Conditionally apply augmentation
        if self.augmentor and any(lbl["boxes"].shape[0] > 0 for lbl in labels):
            padded_boxes, mask = self._pad_boxes(labels)
            # Convert boxes to xyxy format for augmentation
            padded_boxes_xyxy = _cxcywh_to_xyxy(padded_boxes)
            pixel_values, aug_boxes_xyxy = self.augmentor(pixel_values, padded_boxes_xyxy)
            # Convert back to cxcywh
            aug_boxes = _xyxy_to_cxcywh(aug_boxes_xyxy)
            self._unpad_boxes(aug_boxes, mask, labels)

 
        outputs = self(pixel_values, labels)
        loss    = outputs.loss
 
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        for k, v in outputs.loss_dict.items():
            self.log(f"train/{k}", v, on_step=False, on_epoch=True)
 
        return loss
 
    # ── validation ───────────────────────────────────────────────────────
 
    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self._eval_step(batch)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        for k, v in loss_dict.items():
            self.log(f"val/{k}", v, on_epoch=True)
        return loss
 
    # ── test ─────────────────────────────────────────────────────────────
 
    def test_step(self, batch, batch_idx):
        loss, loss_dict = self._eval_step(batch)
        self.log("test/loss", loss, on_epoch=True, prog_bar=True)
        for k, v in loss_dict.items():
            self.log(f"test/{k}", v, on_epoch=True)
        return loss
 
    # ── optimizer ────────────────────────────────────────────────────────
 
    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(),
                          lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
 
 

# 5.  FIFTYONE INFERENCE HELPER
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(
    image_filepaths: list,
    lightning_module: RTDETRLightningModule,
    processor: RTDetrImageProcessor,
    confidence_threshold: float = 0.3,
    output_dir: str | Path | None = None,
    device: str | torch.device | None = None,
) -> list[dict]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
 
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
 
        record = {"filepath": str(path.resolve()), "detections": detections}
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
    weight_decay: float = 1e-4,
    output_dir: str     = "checkpoints",
    early_stop_patience: int = 10,
    wandb_project: str  = "rtdetr-dugong",
    wandb_tags: list    = None,
):
    id2label  = id2label or {0: "dugong"}
    ckpt_dir  = os.path.join(output_dir, run_name)   # checkpoints/schema_partition_timestamp/
 
    logger.info(f"Run: {run_name}")
    logger.info(f"Checkpoint dir: {ckpt_dir}")
 
    processor = RTDetrImageProcessor.from_pretrained(checkpoint)
    logger.success(f"Processor loaded from '{checkpoint}'")
 
    lit_model = RTDETRLightningModule(
        checkpoint=checkpoint, lr=lr, weight_decay=weight_decay,
        max_epochs=max_epochs, id2label=id2label, use_augmentation=use_augmentation
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
        filename=f"{run_name}-{{epoch:02d}}-{{val/loss:.4f}}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=False,
        every_n_epochs=10,
    )
 
    wandb_logger = WandbLogger(
        project=wandb_project,
        name=run_name,          # same unified ID
        tags=wandb_tags or [],
        log_model=False,
        config=dict(checkpoint=checkpoint, lr=lr, weight_decay=weight_decay,
                    batch_size=batch_size, max_epochs=max_epochs, id2label=id2label),
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
            EarlyStopping(monitor="val/loss", patience=early_stop_patience, mode="min"),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        log_every_n_steps=100,
        gradient_clip_val=0.1,
        enable_progress_bar=True,
    )
 
    logger.info("Starting training …")
    trainer.fit(lit_model, datamodule=data_module)
    logger.success("Training complete")
 
    logger.info("Evaluating best checkpoint on held-out test set …")
    trainer.test(lit_model, datamodule=data_module, ckpt_path="best")
    logger.success("Test evaluation complete")
 
    return trainer, lit_model, processor


# ─────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train RT-DETR on dugong patches")
    parser.add_argument("--schema",       type=str, required=True,
                        help="Experiment schema label (e.g. 'v1', 'ablation')")
    parser.add_argument("--partition",    type=str, required=True,
                        choices=["NC", "partition_25"],
                        help="Training partition strategy")
    parser.add_argument("--csvfile",      type=str, required=True,
                        help="CSV with train_seed / test_seed / val_seed / train_nc columns")
    parser.add_argument("--csvpatches",   type=str, default=None,
                        help="Parquet file with pre-split patch filepaths (required for partition_25)")
    parser.add_argument("--patch-folder", type=str,
                        default="/share/home/e2406743/dataset/exported_img/seed_42",
                        help="Root folder containing images/, labels/, metadata/ subfolders")
    parser.add_argument("--output-dir",   type=str, default="checkpoints")
    parser.add_argument("--batch-size",   type=int, default=8)
    parser.add_argument("--max-epochs",   type=int, default=50)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--wandb-project",type=str, default="rtdetr-dugong")
    parser.add_argument("--augment", action="store_true", default=False,
                        help="Enable data augmentation during training")
    return parser.parse_args()

def main():
    args = parse_args()
 
    # ── validate args ─────────────────────────────────────────────────────
    assert Path(args.csvfile).exists(),       f"csvfile not found: {args.csvfile}"
    assert Path(args.patch_folder).exists(),  f"patch-folder not found: {args.patch_folder}"
    if args.partition == "partition_25":
        assert args.csvpatches is not None,   "--csvpatches is required for partition_25"
        assert Path(args.csvpatches).exists(), f"csvpatches not found: {args.csvpatches}"
    
    augmentation_flag = args.augment
    # ── unified run name: schema_partition_MMDD_HHMM ─────────────────────
    now          = datetime.datetime.now()
    run_name     = f"{args.schema}_{args.partition}_{now.strftime('%m%d_%H%M')}"
    

    # ── logger setup (file written to logs/<run_name>.log) ────────────────
    setup_logger(log_dir="logs", run_name=run_name)
    logger.info(f"Run name: {run_name}")
    logger.info(f"Args: {vars(args)}")
    logger.info(f"AUGMENTATION: {str(augmentation_flag)}")
    # ── load split lists from CSV ─────────────────────────────────────────
    wp_train_list, nc_train_list, test_list, val_list = return_list_from_csv(args.csvfile)
    logger.info(f"CSV loaded: {args.csvfile}")
 
    # ── resolve train images by partition ─────────────────────────────────
    match args.partition:
        case "NC":
            logger.info("Partition: NC — using nc_train_list")
            train_list_images, train_list_labels, _ = mapdict_patches_filepath(
                nc_train_list, args.patch_folder)
 
        case "partition_25":
            logger.info("Partition: partition_25 — loading from parquet")
            df = pd.read_parquet(args.csvpatches)
            train_list_images   = df.loc[df.index == "images",   args.partition].values.tolist()
            train_list_labels   = df.loc[df.index == "labels",   args.partition].values.tolist()
 
    assert len(train_list_images) > 0, "train_list_images is empty — check your CSV / parquet"
    assert len(train_list_labels) > 0, "train_list_labels is empty — check your CSV / parquet"
    assert len(train_list_images) == len(train_list_labels), \
        f"Train image/label count mismatch: {len(train_list_images)} vs {len(train_list_labels)}"
    logger.success(f"Train set: {len(train_list_images)} images")
 
    # ── test & val always come from CSV ───────────────────────────────────
    logger.info("Mapping test patches …")
    test_list_images, test_list_labels, _ = mapdict_patches_filepath(test_list, args.patch_folder)
    assert len(test_list_images) > 0, "test_list_images is empty"
 
    logger.info("Mapping val patches …")
    val_list_images, val_list_labels, _ = mapdict_patches_filepath(val_list, args.patch_folder)
    assert len(val_list_images) > 0, "val_list_images is empty"
 
    # ── train ─────────────────────────────────────────────────────────────
    trainer, model, processor = train(
        train_images=train_list_images, train_labels=train_list_labels,
        val_images=val_list_images,     val_labels=val_list_labels,
        test_images=test_list_images,   test_labels=test_list_labels,
        use_augmentation = augmentation_flag,
        run_name=run_name,
        id2label={0: "dugong"},
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        lr=args.lr,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        wandb_tags=[args.schema, args.partition],
    )
 
    # ── inference on test set → JSON files ────────────────────────────────
    logger.info("Running inference on test set …")
    run_inference(
        image_filepaths=test_list_images,
        lightning_module=model,
        processor=processor,
        confidence_threshold=0.3,
    )


if __name__ == "__main__":
        ## check connections
    load_dotenv()
    check_hf_auth()
    check_wandb_auth()
    main()