"""
src/training/module.py
======================
DetectorLightningModule — model-agnostic Lightning wrapper around any adapter.

Design
------
- This module knows nothing about model internals. All forward/decode logic
  lives in the adapter (RTDETRAdapter, YOLOAdapter, ...).
- W&B logging is on/off via cfg.logging.enabled — no code change needed.
- BN freeze is delegated to the adapter's freeze_backbone_bn() if it exists.
- mAP is computed at val and test epoch end via torchmetrics.

Loss logging layout (W&B panels)
---------------------------------
  {split}/loss              ← total loss (prog_bar=True for val)
  {split}/main/loss_*       ← main decoder head losses
  {split}/aux/loss_*        ← mean across aux decoder layers

Parameters (from Hydra cfg)
----------------------------
  cfg.model.*               → forwarded to adapter (already built)
  cfg.training.lr
  cfg.training.backbone_lr_factor
  cfg.training.weight_decay
  cfg.training.max_epochs
  cfg.training.warmup_epochs
  cfg.training.gradient_clip_val   (used by Trainer, not here)
  cfg.model.confidence_threshold
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from src.adapters.base import BaseDetectorAdapter


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """(..., 4) normalised cxcywh → xyxy (same scale)."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)


class DetectorLightningModule(pl.LightningModule):
    """
    Model-agnostic Lightning module.

    Parameters
    ----------
    adapter  : any BaseDetectorAdapter subclass (already built)
    cfg      : Hydra DictConfig with training.* and logging.* sections
    """

    def __init__(self, adapter: BaseDetectorAdapter, cfg,                  inference_dir=None):
        super().__init__()
        self.adapter = adapter
        self.cfg     = cfg
        self.inference_dir = inference_dir

        # REGISTER ADAPATER MODEL 
        self.model = adapter.model
        
        # Save hyperparameters (excluding non-serialisable objects)
        self.save_hyperparameters(ignore=["adapter","cfg","inference_dir"])

        self._first_batch_logged = cfg.get("logging.first_batch_log",False)
        self._test_predictions: list[dict] = []

        # ── mAP — one object per split, accumulates across batches 
        map_kwargs = dict(
            iou_type="bbox",
            box_format="xyxy",
            max_detection_thresholds=[1, 5, 100],
            backend="faster_coco_eval",
            ## case want to address the size of bounding box
            ## area_ranges = {"small":[0,2000],"medium":[2000,8000],"large":[8000,1e10]}
        )
        self.val_map  = MeanAveragePrecision(**map_kwargs)
        self.test_map = MeanAveragePrecision(**map_kwargs)

    #  forward 

    def forward(self, batch):
        return self.adapter.forward(batch)

    #  training 

    def on_train_epoch_start(self):
        self.adapter.train()
        # Freeze backbone BatchNorm if adapter supports it
        if hasattr(self.adapter, "freeze_backbone_bn"):
            self.adapter.freeze_backbone_bn()

    def training_step(self, batch, batch_idx):
        outputs = self(batch)

        # First-batch sanity log
        if not self._first_batch_logged and batch_idx == 0:
            self._log_first_batch(batch, outputs)
            self._first_batch_logged = True

        self._log_losses("train", outputs.loss, outputs.loss_dict)
        return outputs.loss

    # ── validation ────────────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        self._log_losses("val", outputs.loss, outputs.loss_dict)

        with torch.no_grad():
            preds, targets = self._collect_map_inputs(outputs, batch["labels"])
        self.val_map.update(preds, targets)

        return outputs.loss

    def on_validation_epoch_end(self):
        result = self.val_map.compute()
        self.log("val/mAP",    result["map"],    prog_bar=True,  sync_dist=False)
        self.log("val/mAP_50", result["map_50"], prog_bar=True,  sync_dist=False)
        self.log("val/mAP_75", result["map_75"],                 sync_dist=False)
        self._log_map_breakdown("val", result)
        self.val_map.reset()

    # ── test
    def test_step(self, batch, batch_idx):
        outputs = self(batch)
        self._log_losses("test", outputs.loss, outputs.loss_dict)
        preds, targets = self._collect_map_inputs(outputs, batch["labels"])
        self.test_map.update(preds, targets)
 
        # Accumulate predictions for JSON export
        filepaths = batch.get("filepaths", [""] * len(batch["labels"]))
        metadata  = batch.get("metadata",  [{}]  * len(batch["labels"]))
        thr       = self.cfg.model.confidence_threshold
 
        scores_all = outputs.logits.sigmoid()   # (B, Q, 80)
        boxes_all  = outputs.pred_boxes          # (B, Q, 4) normalised cxcywh
 
        for i in range(len(batch["labels"])):
            dugong_scores = scores_all[i, :, 0]
            keep          = dugong_scores > thr
 
            detections = []
            for box, score in zip(
                boxes_all[i][keep].cpu().tolist(),
                dugong_scores[keep].cpu().tolist(),
            ):
                detections.append({
                    "label":        "dugong",
                    "bounding_box": box,          # [cx, cy, w, h] normalised
                    "confidence":   round(score, 6),
                })
 
            self._test_predictions.append({
                "filepath":      filepaths[i],
                "detections":    detections,
                "tile_metadata": metadata[i],
            })
 
        return outputs.loss

    def on_test_epoch_end(self):
        result = self.test_map.compute()
        self.log("test/mAP",    result["map"],    sync_dist=False)
        self.log("test/mAP_50", result["map_50"], sync_dist=False)
        self.log("test/mAP_75", result["map_75"], sync_dist=False)
        self._log_map_breakdown("test", result)
        self.test_map.reset()
 
        # ── Write predictions JSON ────────────────────────────────────────
        if self._test_predictions and self.inference_dir:
            import json
            from pathlib import Path
            out_dir = Path(self.inference_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{self.cfg.run_name}_test_predictions.json"
            with open(out_path, "w") as f:
                json.dump(self._test_predictions, f, indent=2)
            import logging
            logging.getLogger(__name__).info(
                f"Saved {len(self._test_predictions)} test predictions → {out_path}"
            )
        self._test_predictions = []   # reset for potential re-use

    # ── optimizer & scheduler ─────────────────────────────────────────────

    def configure_optimizers(self):
        t = self.cfg.training

        # Differential LR: backbone gets lr * backbone_lr_factor
        if hasattr(self.adapter, "backbone_and_head_params"):
            backbone_params, head_params = self.adapter.backbone_and_head_params()
            param_groups = [
                {"params": backbone_params, "lr": t.lr * t.backbone_lr_factor},
                {"params": head_params,     "lr": t.lr},
            ]
        else:
            param_groups = [{"params": self.adapter.parameters(), "lr": t.lr}]

        optimizer = AdamW(param_groups, weight_decay=t.weight_decay)

        warmup = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=t.warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=t.max_epochs // 2,
            eta_min=t.lr * t.backbone_lr_factor * 0.01,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[t.warmup_epochs],
        )

        return {
            "optimizer":    optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    # ── private helpers ───────────────────────────────────────────────────

    def _log_losses(self, prefix: str, loss: torch.Tensor, loss_dict: dict):
        """
        Log total + split main/aux losses.
        on_step=False everywhere → W&B x-axis is epoch, never step.
        """
        bs   = self.trainer.datamodule.batch_size
        sync = prefix == "train"

        self.log(
            f"{prefix}/loss", loss,
            on_step=False, on_epoch=True,
            prog_bar=(prefix == "val"),
            sync_dist=sync, batch_size=bs,
        )

        main, aux_accum = {}, {}
        for k, v in loss_dict.items():
            if "_aux_" in k:
                base = k.split("_aux_")[0]
                aux_accum.setdefault(base, []).append(v)
            else:
                main[k] = v

        for k, v in main.items():
            self.log(
                f"{prefix}/main/{k}", v,
                on_step=False, on_epoch=True,
                sync_dist=sync, batch_size=bs,
            )

        for base, vals in aux_accum.items():
            self.log(
                f"{prefix}/aux/{base}", torch.stack(vals).mean(),
                on_step=False, on_epoch=True,
                sync_dist=sync, batch_size=bs,
            )

    def _collect_map_inputs(self, outputs, labels) -> tuple[list, list]:
        """Convert model outputs → torchmetrics-ready preds / targets."""
        thr        = self.cfg.model.confidence_threshold
        scores_all = outputs.logits.sigmoid()   # (B, Q, num_classes)
        boxes_all  = outputs.pred_boxes          # (B, Q, 4) norm cxcywh

        preds, targets = [], []

        for i, lbl in enumerate(labels):
            h, w   = lbl["orig_size"].tolist()
            device = boxes_all.device  # Get the device of the model output
            scale = torch.tensor([w, h, w, h], device=device)  # Move scale to the same device as boxes_all
            ##scale  = boxes_all.new_tensor([w, h, w, h])

            # Class-0 (dugong) scores only
            dugong_scores = scores_all[i, :, 0]
            keep          = dugong_scores > thr
            boxes_abs     = (_cxcywh_to_xyxy(boxes_all[i]) * scale).clamp(0)

            preds.append({
                "boxes":  boxes_abs[keep].cpu(),
                "scores": dugong_scores[keep].cpu(),
                "labels": torch.zeros(keep.sum(), dtype=torch.int32),
            })
            
             # Move ground truth boxes to the same device as scale
            gt_boxes = lbl["boxes"].to(device)  # Move to GPU if needed
            gt_abs = (_cxcywh_to_xyxy(gt_boxes) * scale).clamp(0)

            targets.append({
                "boxes":  gt_abs.cpu(),
                "labels": torch.zeros(len(lbl["boxes"]), dtype=torch.int32),
            })

        return preds, targets

    def _log_map_breakdown(self, prefix: str, result: dict):
        """Log size-specific mAP sub-metrics for diagnostics."""
        for key in ("map_small", "map_medium", "map_large",
                    "mar_1", "mar_10", "mar_300"):
            val = result.get(key, torch.tensor(-1.0))
            self.log(f"{prefix}/{key}", val, sync_dist=False)

    def _log_first_batch(self, batch, outputs):
        """One-shot debug log on the first training batch."""
        import logging
        log = logging.getLogger(__name__)

        pv     = batch["pixel_values"]
        labels = batch["labels"]
        total  = sum(lbl["boxes"].shape[0] for lbl in labels)

        log.debug("─" * 60)
        log.debug(f"First batch  epoch={self.current_epoch}")
        log.debug(f"  pixel_values : {list(pv.shape)}  dtype={pv.dtype}")
        log.debug(f"  pixel range  : [{pv.min():.3f}, {pv.max():.3f}]")
        log.debug(f"  batch size   : {len(labels)}")
        log.debug(f"  total boxes  : {total}  (avg {total/len(labels):.1f}/img)")

        # Top-5 predictions from first image
        logits_cpu = outputs.logits[0].detach().float().cpu()
        boxes_cpu  = outputs.pred_boxes[0].detach().float().cpu()
        scores     = logits_cpu.sigmoid().max(dim=-1)
        topk       = scores.values.topk(min(5, len(scores.values)))

        log.debug("  Top-5 queries (img 0):")
        for rank, idx in enumerate(topk.indices.tolist()):
            sc  = scores.values[idx].item()
            cls = scores.indices[idx].item()
            box = boxes_cpu[idx].tolist()
            log.debug(
                f"    [{rank}] score={sc:.4f}  class={cls}  "
                f"box=[{box[0]:.3f},{box[1]:.3f},{box[2]:.3f},{box[3]:.3f}]"
            )
        log.debug("─" * 60)

    def set_inference_dir(self, inference_dir) -> None:
        """Enable or disable prediction JSON export. Pass None to disable."""
        self.inference_dir = inference_dir