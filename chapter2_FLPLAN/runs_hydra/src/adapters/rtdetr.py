"""
adapters/adapters_rtdetr.py
===========================
RT-DETR adapter.  Wraps HuggingFace RTDetrForObjectDetection.

Key design decisions
--------------------
- 80-class head strategy: model loads with all 80 COCO classes intact.
  Only class index 0 is remapped to "dugong". The head is NEVER reinitialised
  to 1 class — doing so causes sigmoid saturation and zero gradients.
- Backbone BatchNorm frozen during training to prevent catastrophic forgetting
  when fine-tuning NC→WP. Called via freeze_backbone_bn() from the Lightning
  module's on_train_epoch_start.
- Differential LR: backbone vs head via backbone_and_head_params().
- Confidence threshold applied only to class-0 (dugong) at decode time.
- No Lightning import anywhere in this file.

Compatible checkpoints
----------------------
- "PekingU/rtdetr_r50vd"     (base HF weights, NC run)
- local save_pretrained() dir (NC export, fine-tune run)
- any HF Hub repo with the same architecture
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

from src.adapters.base import BaseDetectorAdapter

import logging
_log = logging.getLogger(__name__)

def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """(..., 4) normalised cxcywh → xyxy (same scale)."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)


class RTDETRAdapter(BaseDetectorAdapter):
    """
    Adapter for PekingU/rtdetr_r50vd.

    Expected cfg fields (model/rtdetr.yaml)
    ----------------------------------------
    name                : rtdetr
    checkpoint          : HF model id or local save_pretrained() dir
    confidence_threshold: 0.01
    num_classes         : 80     # keep 80-class head strategy
    id2label            :
      0: dugong
    """

    # ── Build ─────────────────────────────────────────────────────────────

    def _build_model(self, cfg: Any) -> None:
        """Load model + processor, remap class 0 → dugong."""
        self.processor = RTDetrImageProcessor.from_pretrained(cfg.checkpoint)
        self.model     = RTDetrForObjectDetection.from_pretrained(cfg.checkpoint)

        # 80-class head strategy: remap label at index 0, leave head intact
        self.model.config.id2label[0]    = "dugong"
        self.model.config.label2id["dugong"] = 0

        head_shape = self.model.model.enc_score_head.weight.shape
        assert head_shape[0] == cfg.num_classes, (
            f"Head has {head_shape[0]} rows, expected {cfg.num_classes}. "
            "Check num_classes in model/rtdetr.yaml — it must match the "
            "checkpoint's head (80 for PekingU/rtdetr_r50vd)."
        )
        _log.info(
            f"  [RTDETRAdapter] Loaded '{cfg.checkpoint}' "
            f"head={head_shape}  dugong@class0"
        )

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, batch: dict) -> Any:
        """
        Run RT-DETR forward pass.

        Parameters
        ----------
        batch : {
            "pixel_values" : (B, 3, H, W) tensor  — required
            "labels"       : list[dict]            — optional, training only
        }

        Returns
        -------
        RTDetrObjectDetectionOutput
            .loss       — scalar (None at inference if labels not provided)
            .loss_dict  — dict of component losses
            .logits     — (B, Q, num_classes) raw logits
            .pred_boxes — (B, Q, 4) normalised cxcywh
        """
        pixel_values = batch["pixel_values"]
        labels       = batch.get("labels", None)

        # When called from Lightning (training/val/test), the batch is already
        # on the correct device and dtype — Lightning's AMP casts model weights
        # to float16 on GPU and expects inputs to match. Do NOT call .to(device)
        # here or it sends pixel_values back to CPU float32, causing the
        # "FloatTensor vs cuda.HalfTensor" mismatch.
        # Only move tensors when calling standalone (adapter.predict()).
        if not pixel_values.is_cuda and self._device not in ("cpu", ""):
            pixel_values = pixel_values.to(self._device)
            if labels is not None:
                labels = [
                    {k: v.to(self._device) for k, v in lbl.items()}
                    for lbl in labels
                ]

        return self.model(pixel_values=pixel_values, labels=labels)

    # ── Decode ────────────────────────────────────────────────────────────

    def decode(
        self,
        outputs: Any,
        original_sizes: list[tuple[int, int]],
        threshold: float | None = None,
    ) -> list[dict]:
        """
        Convert model outputs to standard detection dicts.

        Applies threshold to class-0 (dugong) scores only.
        Converts normalised cxcywh → absolute pixel xyxy.

        Parameters
        ----------
        outputs        : RTDetrObjectDetectionOutput
        original_sizes : [(h, w), ...] one per image in the batch
        threshold      : overrides self.confidence_threshold if given

        Returns
        -------
        list of dicts, one per image:
            {
                "boxes":  np.ndarray (N, 4)  [x1, y1, x2, y2] absolute pixels
                "scores": np.ndarray (N,)
                "labels": np.ndarray (N,)    all zeros (dugong)
            }
        """
        thr        = threshold if threshold is not None else self.confidence_threshold
        scores_all = outputs.logits.sigmoid()   # (B, Q, 80)
        boxes_all  = outputs.pred_boxes          # (B, Q, 4) normalised cxcywh

        results = []
        for i, (h, w) in enumerate(original_sizes):
            scale = boxes_all.new_tensor([w, h, w, h])

            dugong_scores = scores_all[i, :, 0]             # (Q,) class-0 only
            keep          = dugong_scores > thr
            boxes_xyxy    = (_cxcywh_to_xyxy(boxes_all[i]) * scale).clamp(0)

            results.append({
                "boxes":  boxes_xyxy[keep].cpu().numpy(),
                "scores": dugong_scores[keep].cpu().numpy(),
                "labels": np.zeros(int(keep.sum()), dtype=np.int32),
            })

        return results

    # ── Load checkpoint ───────────────────────────────────────────────────

    def load_checkpoint(self, path: str | Path) -> None:
        """
        Load weights from three possible sources:

        (a) Lightning .ckpt  — extracts state_dict, strips "adapter.model." prefix
        (b) save_pretrained() dir  — loads via RTDetrForObjectDetection.from_pretrained
        (c) HF Hub model id  — same as (b)

        Always re-applies the dugong label remapping after loading.
        """
        path = str(path)

        if path.endswith(".ckpt"):
            # weights_only=False needed because Lightning checkpoints contain
            # OmegaConf DictConfig objects (PyTorch 2.6 changed the default).
            ckpt  = torch.load(path, map_location="cpu", weights_only=False)
            state = {
                k.removeprefix("adapter.model."): v
                for k, v in ckpt["state_dict"].items()
                if k.startswith("adapter.model.")
            }
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing:
                n = len(missing)
                _log.info(f"  [RTDETRAdapter] {n} missing keys: "
                      f"{missing[:3]}{'...' if n > 3 else ''}")
            if unexpected:
                n = len(unexpected)
                _log.info(f"  [RTDETRAdapter] {n} unexpected keys: "
                      f"{unexpected[:3]}{'...' if n > 3 else ''}")
            _log.info(f"  [RTDETRAdapter] Loaded from Lightning ckpt: {path}")

        else:
            # save_pretrained dir or HF Hub id — reload both model and processor
            self.model     = RTDetrForObjectDetection.from_pretrained(path)
            self.processor = RTDetrImageProcessor.from_pretrained(path)
            _log.info(f"  [RTDETRAdapter] Loaded from pretrained: {path}")

        # Always re-apply label remapping — it may not be in the saved config
        self.model.config.id2label[0]        = "dugong"
        self.model.config.label2id["dugong"] = 0

    # ── Preprocessing  (for standalone inference) ─────────────────────────

    def _preprocess(self, pil_images: list[Image.Image]) -> dict:
        """Convert PIL images → processor batch on self._device."""
        inputs = self.processor(images=pil_images, return_tensors="pt")
        return {k: v.to(self._device) for k, v in inputs.items()}

    # ── Training helpers ──────────────────────────────────────────────────

    def freeze_backbone_bn(self) -> None:
        """
        Set all backbone BatchNorm2d layers to eval mode.
        Called every epoch from DetectorLightningModule.on_train_epoch_start.
        Prevents running-mean/var drift that causes catastrophic forgetting
        when fine-tuning on a small target-domain dataset.
        """
        frozen = 0
        for module in self.model.model.backbone.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()
                frozen += 1
        # Only print once (epoch 0)
        if not getattr(self, "_bn_freeze_logged", False):
            _log.info(f"  [RTDETRAdapter] Frozen {frozen} backbone BN layers")
            self._bn_freeze_logged = True

    def backbone_and_head_params(self) -> tuple[list, list]:
        """
        Split parameters into backbone vs head for differential LR.

        Returns
        -------
        (backbone_params, head_params)

        Usage in configure_optimizers
        ------------------------------
            backbone_p, head_p = adapter.backbone_and_head_params()
            AdamW([
                {"params": backbone_p, "lr": lr * backbone_lr_factor},
                {"params": head_p,     "lr": lr},
            ])
        """
        backbone_params = [
            p for n, p in self.model.named_parameters() if "backbone" in n
        ]
        head_params = [
            p for n, p in self.model.named_parameters() if "backbone" not in n
        ]
        return backbone_params, head_params

    # ── Persistence ───────────────────────────────────────────────────────

    def save_pretrained(self, output_dir: str | Path) -> None:
        """
        Save model + processor using save_pretrained().
        Verifies the 80-class head and dugong label before writing.

        Used after NC training to create the checkpoint that fine-tune runs load.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Pre-save assertion: never save a 1-class head
        head_shape = self.model.model.enc_score_head.weight.shape
        assert head_shape[0] == self.num_classes, (
            f"Head has {head_shape[0]} rows — expected {self.num_classes}. "
            "Do not reinitialise the head to 1 class before saving."
        )

        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)

        # Post-save verification
        with open(output_dir / "config.json") as f:
            saved_cfg = json.load(f)

        assert "0" in saved_cfg.get("id2label", {}), (
            f"CRITICAL: 'dugong' not at key '0' in saved config.json "
            f"({output_dir / 'config.json'})"
        )
        _log.info(
            f"  [RTDETRAdapter] Saved and verified → {output_dir}\n"
            f"    head={head_shape}  dugong@class0=✓"
        )