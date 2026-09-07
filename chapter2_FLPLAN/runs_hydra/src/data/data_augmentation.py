"""
src/data/data_augmentation.py
=============================
Albumentations augmentation factory for dugong aerial patch imagery.

All pipelines share the same BboxParams (COCO format, min_visibility=0.1)
so they are drop-in compatible with DugongDataset.

Named pipelines
---------------
  none        → returns None (no augmentation)
  flip        → HorizontalFlip + VerticalFlip
  flip_color  → flips + RandomBrightnessContrast + HueSaturationValue
                + GaussianBlur

Design notes
------------
- bbox_params are defined once and shared across all pipelines.
- Each pipeline reads only the keys it needs from aug_cfg — unknown keys
  are ignored, so adding new config params is always backward compatible.
- All probability and magnitude params have safe defaults so the function
  works even if the config key is missing.
- GaussianBlur blur_limit is forced to an odd number (Albumentations
  requirement) by rounding up if needed.

Usage
-----
    from src.data.data_augmentation import build_augmentor

    augmentor = build_augmentor(cfg.get("augmentation"))
    # augmentor is either None or an A.Compose object ready for DugongDataset
"""

from __future__ import annotations

import logging
import math
import json
import random
import numpy as np
from pathlib import Path

try:
    import ot
except ImportError:
    ot = None

try:
    import albumentations as A
    from albumentations.core.transforms_interface import ImageOnlyTransform
except ImportError:
    ImageOnlyTransform = object   # fallback so the class definition doesn't crash

_log = logging.getLogger(__name__)


def build_augmentor(aug_cfg):
    """
    Build an Albumentations Compose pipeline from a Hydra augmentation config.

    Parameters
    ----------
    aug_cfg : OmegaConf DictConfig | None
        Expects at minimum a `name` key.  All other keys have safe defaults.

    Returns
    -------
    A.Compose  if a named pipeline is requested
    None       if aug_cfg is None or name == "none"
    """
    if aug_cfg is None:
        return None

    aug_name = aug_cfg.get("name", "none")

    if aug_name == "none":
        return None

    try:
        import albumentations as A
    except ImportError:
        _log.warning("albumentations not installed — skipping augmentation.")
        return None

    # ── Shared BboxParams — used by every pipeline ────────────────────────
    bbox_params = A.BboxParams(
        format         = "coco",
        label_fields   = ["class_labels"],
        min_visibility = 0.1,
    )

    # ── Flip transforms — shared by flip and flip_color ───────────────────
    flips = _build_flips(aug_cfg, A)

    # ── Pipeline dispatch ─────────────────────────────────────────────────
    if aug_name == "flip":
        pipeline = flips
        _log.info("[data_augmentation] Pipeline: flip")

    elif aug_name == "flip_color":
        pipeline = flips + _build_color(aug_cfg, A)
        _log.info("[data_augmentation] Pipeline: flip_color")

    elif aug_name == "flip_color_ot":
        pipeline = flips + _build_color(aug_cfg, A) + [
        OTColorTransfer(
            reference_json = aug_cfg.get("ot_reference_json"),
            n_samples      = aug_cfg.get("ot_n_samples",     500),
            reg_e          = aug_cfg.get("ot_reg_e",         0.1),
            max_pool_size  = aug_cfg.get("ot_max_pool_size",  50),
            p              = aug_cfg.get("ot_p",             0.5),
        )
        ]   
        _log.info("[data_augmentation] Pipeline: flip_color_ot")
    else:
        _log.warning(
            f"[data_augmentation] Unknown augmentation name '{aug_name}' "
            "— no augmentation applied."
        )
        return None

    _log_pipeline(pipeline)
    return A.Compose(pipeline, bbox_params=bbox_params)


# ── Sub-builders ──────────────────────────────────────────────────────────────

def _build_flips(aug_cfg, A):
    """HorizontalFlip + VerticalFlip."""
    return [
        A.HorizontalFlip(p=aug_cfg.get("horizontal_flip_p", 0.5)),
        A.VerticalFlip(p=aug_cfg.get("vertical_flip_p",     0.5)),
    ]


def _build_color(aug_cfg, A):
    """
    Color augmentations suited for aerial dugong imagery.

    RandomBrightnessContrast
        Simulates varying sun angle, cloud cover, and water turbidity
        across different flight missions.

    HueSaturationValue
        Simulates water colour variation between sites (Ningaloo clear
        turquoise vs Fitzroy/Lagrange more turbid greens).  Kept subtle
        to avoid producing unrealistic imagery.

    GaussianBlur
        Simulates varying altitude and lens focus.  Small kernels only
        (3×3 or 5×5) to avoid destroying fine dugong detail.
    """
    # Force blur_limit to odd (Albumentations requirement)
    blur_limit = aug_cfg.get("blur_limit", 5)
    if blur_limit % 2 == 0:
        blur_limit += 1
        _log.warning(
            f"[data_augmentation] blur_limit must be odd — "
            f"rounded up to {blur_limit}."
        )

    return [
        A.RandomBrightnessContrast(
            brightness_limit = aug_cfg.get("brightness_limit", 0.2),
            contrast_limit   = aug_cfg.get("contrast_limit",   0.2),
            p                = aug_cfg.get("brightness_contrast_p", 0.5),
        ),
        A.HueSaturationValue(
            hue_shift_limit = aug_cfg.get("hue_shift_limit", 10),
            sat_shift_limit = aug_cfg.get("sat_shift_limit", 20),
            val_shift_limit = aug_cfg.get("val_shift_limit", 10),
            p               = aug_cfg.get("hue_saturation_p", 0.3),
        ),
        A.GaussianBlur(
            blur_limit = (3, blur_limit),   # always at least 3×3
            p          = aug_cfg.get("blur_p", 0.2),
        ),
    ]



## OPTIMAL TRANSFORM CLASS
class OTColorTransfer(ImageOnlyTransform):
    """
    Per-image Optimal Transport color transfer as an Albumentations transform.

    Fits a SinkhornTransport on a random subset of pixels from the source
    image and a randomly drawn reference image, then applies the mapping to
    all source pixels.  Reference images are loaded once at init from a JSON
    file listing their paths.

    Parameters
    ----------
    reference_json : str
        Path to a JSON file with structure:
            {"references": ["/path/to/img1.jpg", "/path/to/img2.jpg", ...]}
    n_samples      : int
        Number of pixels sampled from source and reference for OT fit.
        300–500 is the sweet spot between quality and speed. (default: 500)
    reg_e          : float
        Sinkhorn entropy regularisation.  Lower = sharper transfer but
        slower convergence.  (default: 0.1)
    max_pool_size  : int
        Cap on how many reference images to load into memory. (default: 50)
    p              : float
        Probability of applying the transfer per image. (default: 0.5)
    """

    def __init__(
        self,
        reference_json: str,
        n_samples:      int   = 500,
        reg_e:          float = 0.1,
        max_pool_size:  int   = 50,
        p:              float = 0.5,
    ):
        super().__init__(p=p)

        if ot is None:
            raise ImportError(
                "POT (Python Optimal Transport) is not installed. "
                "Run: pip install POT"
            )

        self.reference_json = reference_json
        self.n_samples      = n_samples
        self.reg_e          = reg_e
        self.max_pool_size  = max_pool_size

        self.pool = self._load_pool(reference_json, max_pool_size)
        _log.info(
            f"[OTColorTransfer] Loaded {len(self.pool)} reference images "
            f"from {reference_json}"
        )

    # ── Pool loader ───────────────────────────────────────────────────────

    def _load_pool(self, reference_json: str, max_pool_size: int) -> list:
        """
        Load reference images from JSON into a list of float64 numpy arrays.

        Returns
        -------
        list of (N_pixels, 3) float64 arrays — one per reference image
        """
        with open(reference_json) as f:
            data = json.load(f)

        paths = data["references"][:max_pool_size]
        pool  = []

        for path_str in paths:
            path = Path(path_str)
            if not path.exists():
                _log.warning(f"[OTColorTransfer] Reference not found: {path}")
                continue
            try:
                from PIL import Image as PILImage
                img     = PILImage.open(path).convert("RGB")
                arr     = np.array(img).astype(np.float64) / 255.0  # (H, W, 3)
                pixels  = arr.reshape(-1, 3)                         # (N, 3)
                pool.append({"pixels": pixels, "path": str(path)})
            except Exception as e:
                _log.warning(f"[OTColorTransfer] Failed to load {path}: {e}")

        if not pool:
            raise ValueError(
                f"[OTColorTransfer] No valid reference images loaded "
                f"from {reference_json}."
            )

        return pool

    # ── Albumentations interface ──────────────────────────────────────────

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Apply OT color transfer to a single (H, W, 3) uint8 image.

        Steps
        -----
        1. Convert uint8 → float64 [0, 1]
        2. Draw a random reference from the pool (Option A — pure random)
        3. Sample n_samples pixels from source and reference
        4. Fit SinkhornTransport on sampled pixels
        5. Transform ALL source pixels
        6. Clip, convert back to uint8
        """
        # ── 1. Convert source to float64 ──────────────────────────────────
        H, W, C  = image.shape
        src_f64  = image.astype(np.float64) / 255.0     # (H, W, 3)
        src_flat = src_f64.reshape(-1, 3)               # (H*W, 3)

        # ── 2. Random reference (Option A) ────────────────────────────────
        ref_entry  = random.choice(self.pool)           # (N_ref, 3) float64
        ref_pixels = ref_entry["pixels"]

        # ── 3. Sample pixels for OT fit ───────────────────────────────────
        n_src = src_flat.shape[0]
        n_ref = ref_pixels.shape[0]

        idx_s = np.random.randint(0, n_src, size=self.n_samples)
        idx_t = np.random.randint(0, n_ref, size=self.n_samples)

        Xs = src_flat[idx_s]      # (n_samples, 3)
        Xt = ref_pixels[idx_t]    # (n_samples, 3)

        # ── 4. Fit SinkhornTransport ──────────────────────────────────────
        transport = ot.da.SinkhornTransport(reg_e=self.reg_e, verbose=False)
        transport.fit(Xs=Xs, Xt=Xt)

        # ── 5. Transform all source pixels ────────────────────────────────
        transported = transport.transform(Xs=src_flat)  # (H*W, 3)
        transported = np.clip(transported, 0.0, 1.0)

        # ── 6. Convert back to uint8 ──────────────────────────────────────
        result = (transported.reshape(H, W, 3) * 255).astype(np.uint8)
        return result

    def get_transform_init_args_names(self) -> tuple:
        """Required by Albumentations for serialisation."""
        return ("reference_json", "n_samples", "reg_e", "max_pool_size")


# ── Logging helper ────────────────────────────────────────────────────────────

def _log_pipeline(transforms):
    """Log each transform in the pipeline at INFO level."""
    for t in transforms:
        _log.info(f"  [data_augmentation]   {t.__class__.__name__}  p={t.p}")