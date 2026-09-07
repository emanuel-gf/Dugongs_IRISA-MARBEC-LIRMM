"""
adapters/adapters_registry.py
==============================
Model registry — maps model name strings to adapter classes.

Adding a new model
------------------
1. Create adapters_yolo.py with class YOLOAdapter(BaseDetectorAdapter)
2. Import it below and add to MODEL_REGISTRY
3. Add a config/model/yolov8n.yaml with at least: name, checkpoint,
   confidence_threshold, num_classes, id2label

Usage
-----
    from adapters_registry import build_adapter

    adapter = build_adapter(cfg.model)   # cfg.model.name = "rtdetr"
    adapter.to("cuda")
"""

from __future__ import annotations

from .base import BaseDetectorAdapter
from .rtdetr import RTDETRAdapter

# Uncomment when adapters_yolo.py is implemented:
# from adapters_yolo import YOLOAdapter

MODEL_REGISTRY: dict[str, type[BaseDetectorAdapter]] = {
    "rtdetr": RTDETRAdapter,
    # "yolo": YOLOAdapter,
}


def build_adapter(model_cfg) -> BaseDetectorAdapter:
    """
    Instantiate the correct adapter from a Hydra model config.

    Parameters
    ----------
    model_cfg : OmegaConf DictConfig
        Must have a `name` field matching a key in MODEL_REGISTRY.
        All other fields are forwarded to the adapter's __init__.

    Returns
    -------
    Instantiated adapter, not yet moved to a device.
    Call adapter.to("cuda") or let Lightning handle device placement.

    Raises
    ------
    ValueError  if model_cfg.name is not registered.
    """
    name = model_cfg.name.lower().strip()

    if name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available in MODEL_REGISTRY: {available}\n"
            f"To add a new model, implement BaseDetectorAdapter and register it."
        )

    adapter = MODEL_REGISTRY[name](model_cfg)
    print(f"  [registry] Built adapter: '{name}'")
    return adapter