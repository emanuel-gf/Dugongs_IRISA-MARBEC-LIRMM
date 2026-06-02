"""
BaseDetectorAdapter: Abstract base class for all object detection adapters.

Subclasses must implement:
- _build_model: Load model/processor and apply model-specific setup.
- forward: Run model forward pass (must return outputs with .loss, .loss_dict, .logits, .pred_boxes).
- decode: Convert raw outputs to standard detection dicts.
- load_checkpoint: Load weights from .ckpt, save_pretrained dir, or HF Hub.
- _preprocess: Convert PIL images to model input format.

Optional helpers (override if needed):
- freeze_backbone_bn: Freeze BatchNorm layers for fine-tuning stability.
- backbone_and_head_params: Split params for differential LR.
- save_pretrained: Save model/processor to disk.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

class BaseDetectorAdapter(ABC):
    """
    Abstract base class for detector adapters.

    Attributes
    ----------
    model : torch.nn.Module
        The underlying detection model (e.g., RTDetrForObjectDetection).
    processor : Any
        Model-specific processor (e.g., RTDetrImageProcessor).
    num_classes : int
        Number of classes (e.g., 80 for COCO).
    confidence_threshold : float
        Minimum confidence score for detections.
    _device : torch.device
        Device (cpu/cuda) where the model is located.
    """

    def __init__(self, cfg: Any):
        """
        Initialize the adapter.

        Parameters
        ----------
        cfg : OmegaConf DictConfig or dict
            Must contain:
            - name: str (e.g., "rtdetr")
            - checkpoint: str (HF Hub ID or local path)
            - confidence_threshold: float
            - num_classes: int
            - id2label: dict (class index to label name)
        """
        self.cfg = cfg
        self.num_classes = cfg.num_classes
        self.confidence_threshold = cfg.confidence_threshold
        self.id2label = cfg.id2label
        self._device = torch.device("cpu")  # Default; moved via .to(device)
        self.model = None
        self.processor = None

        # Build model/processor
        self._build_model(cfg)

    # ── Abstract Methods ─────────────────────────────────────────────────

    @abstractmethod
    def _build_model(self, cfg: Any) -> None:
        """Load model and processor, apply model-specific setup."""
        pass

    @abstractmethod
    def forward(self, batch: dict) -> Any:
        """
        Run forward pass.

        Parameters
        ----------
        batch : dict
            Must contain "pixel_values" (tensor). May contain "labels" (list[dict]).

        Returns
        -------
        Model-specific output with:
            - .loss: scalar tensor (total loss)
            - .loss_dict: dict of component losses (e.g., {"loss_cls": ..., "loss_box": ...})
            - .logits: (B, Q, num_classes) raw logits
            - .pred_boxes: (B, Q, 4) normalised cxcywh
        """
        pass

    @abstractmethod
    def decode(
        self,
        outputs: Any,
        original_sizes: list[tuple[int, int]],
        threshold: float | None = None,
    ) -> list[dict]:
        """
        Convert model outputs to standard detection dicts.

        Parameters
        ----------
        outputs : Model output (from forward())
        original_sizes : list of (height, width) tuples
        threshold : float (overrides self.confidence_threshold if provided)

        Returns
        -------
        list of dicts, one per image:
            {
                "boxes": np.ndarray (N, 4) [x1, y1, x2, y2] (absolute pixels),
                "scores": np.ndarray (N,),
                "labels": np.ndarray (N,) (class indices)
            }
        """
        pass

    @abstractmethod
    def load_checkpoint(self, path: str | Path) -> None:
        """
        Load weights from:
        - Lightning .ckpt (extracts state_dict, strips adapter prefix)
        - save_pretrained() dir (reloads model/processor)
        - HF Hub model ID (reloads model/processor)
        """
        pass

    @abstractmethod
    def _preprocess(self, pil_images: list[Image.Image]) -> dict:
        """
        Convert PIL images to model input format.

        Parameters
        ----------
        pil_images : list[PIL.Image.Image]

        Returns
        -------
        dict with keys like "pixel_values" (tensor on self._device).
        """
        pass

    # ── Device Management ─────────────────────────────────────────────────

    def to(self, device: str | torch.device) -> "BaseDetectorAdapter":
        """Move model and processor to device."""
        self._device = torch.device(device)
        if self.model is not None:
            self.model.to(self._device)
        return self

    # ── Training Helpers (Override if Needed) ─────────────────────────────

    def freeze_backbone_bn(self) -> None:
        """
        Freeze BatchNorm layers in the backbone (default: no-op).
        Override for models like RT-DETR where this is critical.
        """
        pass

    def backbone_and_head_params(self) -> tuple[list, list]:
        """
        Split parameters into backbone vs head for differential LR.
        Default: return all params in both lists (no splitting).

        Returns
        -------
        (backbone_params, head_params)
        """
        all_params = list(self.model.parameters())
        return all_params, all_params

    def save_pretrained(self, output_dir: str | Path) -> None:
        """
        Save model and processor to disk (default: raises NotImplementedError).
        Override for models that support save_pretrained().
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support save_pretrained()."
        )

    # ── Utility Methods ───────────────────────────────────────────────────

    def train(self) -> None:
        """Set model to train mode."""
        if self.model is not None:
            self.model.train()

    def eval(self) -> None:
        """Set model to eval mode."""
        if self.model is not None:
            self.model.eval()