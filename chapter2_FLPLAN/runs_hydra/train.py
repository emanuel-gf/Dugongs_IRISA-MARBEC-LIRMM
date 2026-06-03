"""
src/train.py
============
Hydra entry point for RT-DETR fine-tuning on FLPLAN dugong patches.

Usage
-----
  python train.py                                         # debug, no W&B
  python train.py logging=wandb                          # real run
  python train.py seed=63 partition=p10 method=aclr      # override split
  python train.py training.max_epochs=3                  # quick smoke-test
  python train.py training.max_epochs=3 profiler=simple  # with profiler
  python train.py run_zero_shot=false                    # skip zero-shot
"""

from __future__ import annotations

import logging
import os
import time
import typing
from datetime import datetime
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import torch.serialization
from omegaconf import DictConfig, OmegaConf
from transformers import AutoImageProcessor

from src.adapters.registry import build_adapter
from src.training.callbacks import build_callbacks
from src.data.datamodule import DugongDataModule
from src.training.module import DetectorLightningModule
from src.training.logger_factory import build_logger

log = logging.getLogger(__name__)


# ── PyTorch 2.6: register safe globals for weights_only=True ─────────────────
# Lightning calls torch.load(weights_only=True) internally when restoring
# checkpoints for trainer.test(). Any Python type in the checkpoint that is
# not pre-registered causes an UnpicklingError.
# def _register_safe_globals() -> None:
#     try:
#         import omegaconf.base
#         import omegaconf.dictconfig
#         import omegaconf.listconfig
#         import omegaconf.nodes
#         torch.serialization.add_safe_globals([
#             # OmegaConf
#             omegaconf.dictconfig.DictConfig,
#             omegaconf.listconfig.ListConfig,
#             omegaconf.base.ContainerMetadata,
#             omegaconf.nodes.AnyNode,
#             omegaconf.nodes.IntegerNode,
#             omegaconf.nodes.FloatNode,
#             omegaconf.nodes.StringNode,
#             omegaconf.nodes.BooleanNode,
#             # typing
#             typing.Any,
#             # Python builtins stored in Lightning checkpoint sections
#             # (lr_schedulers state, loops state, callbacks state)
#             dict,
#             list,
#             set,
#         ])
#     except Exception as e:
#         log.warning(f"Could not register safe globals: {e}")

# _register_safe_globals()


# ── Suppress faster_coco_eval INFO spam ──────────────────────────────────────
logging.getLogger("faster_coco_eval.core.cocoeval").setLevel(logging.WARNING)


# ── Entry point ───────────────────────────────────────────────────────────────

@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:

    # ── 1. Run name ───────────────────────────────────────────────────────
    now      = datetime.now().strftime("%m%d_%H%M")
    run_name = (
        f"{cfg.schema}_{cfg.partition}_{cfg.method}"
        f"_seed{cfg.seed}_{cfg.model.name}_{now}"
    )
    OmegaConf.update(cfg, "run_name", run_name, merge=True)
    log.info(f"Run: {run_name}")
    log.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # ── 2. Reproducibility + Tensor Core hint ─────────────────────────────
    pl.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    # ── 3. Logger ─────────────────────────────────────────────────────────
    logger = build_logger(cfg)

    # ── 4. Adapter ────────────────────────────────────────────────────────
    is_finetune = cfg.get("nc_checkpoint_dir") is not None

    if is_finetune:
        nc_dir = cfg.nc_checkpoint_dir
        assert Path(nc_dir).exists(), (
            f"nc_checkpoint_dir not found: {nc_dir}\n"
            "Run the NC baseline first with save_nc_local=true."
        )
        OmegaConf.update(cfg, "model.checkpoint", str(nc_dir), merge=True)
        log.info(f"Fine-tune run: loading NC weights from {nc_dir}")
    else:
        log.info(f"NC run: loading base weights from {cfg.model.checkpoint}")

    adapter = build_adapter(cfg.model)
    log.info(f"Adapter built: {cfg.model.name}")

    # ── 5. Lightning module ───────────────────────────────────────────────
    module = DetectorLightningModule(adapter=adapter, cfg=cfg)

    # ── 6. Processor + augmentor + DataModule ─────────────────────────────
    augmentor = _build_augmentor(cfg)
    processor = AutoImageProcessor.from_pretrained(cfg.model.checkpoint)

    datamodule = DugongDataModule(
        resolved_paths_json = cfg.paths.resolved_paths_json,
        seed        = cfg.seed,
        partition   = cfg.partition,
        method      = cfg.method,
        processor   = processor,
        batch_size  = cfg.training.batch_size,
        augmentor   = augmentor,
        num_workers = cfg.data.num_workers,
        pin_memory  = cfg.data.pin_memory,
    )

    # ── 7. Zero-shot evaluation (before any gradient update) ──────────────
    # Uses a separate Trainer so it never touches the training state.
    # DO NOT call datamodule.setup() here — validate() does it internally
    # and handles device/dtype (AMP half-precision) transfer correctly.
    if cfg.get("run_zero_shot", True):
        log.info("Zero-shot evaluation (loaded weights, no training) …")
        zs_trainer = pl.Trainer(
            accelerator         = "auto",
            devices             = "auto",
            precision           = cfg.training.precision,
            logger              = logger,
            enable_progress_bar = True,
            enable_checkpointing= False,
        )
        zs_results = zs_trainer.validate(module, datamodule=datamodule)
        log.info(
            f"Zero-shot | mAP={zs_results[0].get('val/mAP', -1):.4f}  "
            f"mAP_50={zs_results[0].get('val/mAP_50', -1):.4f}"
        )

        # test set
        zs_results = zs_trainer.test(module, datamodule=datamodule)
        log.info(
            f"Zero-shot | mAP={zs_results[0].get('val/mAP', -1):.4f}  "
            f"mAP_50={zs_results[0].get('val/mAP_50', -1):.4f}"
        )
    # ── 8. Callbacks + main Trainer ───────────────────────────────────────
    callbacks = build_callbacks(cfg, run_name)

    trainer = pl.Trainer(
        max_epochs           = cfg.training.max_epochs,
        accelerator          = "auto",
        devices              = "auto",
        precision            = cfg.training.precision,
        logger               = logger,
        callbacks            = callbacks,
        log_every_n_steps    = cfg.training.log_every_n_steps,
        gradient_clip_val    = cfg.training.gradient_clip_val,
        enable_progress_bar  = True,
        deterministic        = False,
        num_sanity_val_steps = 0,
        profiler             = cfg.get("profiler", None),
    )

    # ── 9. Fit ────────────────────────────────────────────────────────────
    log.info("Starting training …")
    t0 = time.perf_counter()
    trainer.fit(module, datamodule=datamodule)
    elapsed = time.perf_counter() - t0
    log.info(f"Training complete in {elapsed:.1f}s")

    # ── 10. Test on best checkpoint ───────────────────────────────────────
    log.info("Running test on best checkpoint …")
    trainer.test(module, datamodule=datamodule, ckpt_path="best", weights_only=False)

    # ── 11. Save NC weights locally / push to HF Hub ─────────────────────
    if not is_finetune:
        _save_nc_weights(cfg, module, processor, run_name)


# ── Augmentor factory ─────────────────────────────────────────────────────────

def _build_augmentor(cfg):
    """
    Build Albumentations Compose from cfg.augmentation.name.
    Supported: none | flip | flip_only | ot_color (stub → falls back to flip)
    """
    aug_cfg = cfg.get("augmentation")
    if aug_cfg is None:
        return None

    aug_name = aug_cfg.get("name", "none")
    if aug_name == "none":
        return None

    try:
        import albumentations as A
    except ImportError:
        log.warning("albumentations not installed — skipping augmentation.")
        return None

    bbox_params = A.BboxParams(
        format       = "coco",
        label_fields = ["class_labels"],
        min_visibility = 0.1,
    )
    flips = [
        A.HorizontalFlip(p=aug_cfg.get("horizontal_flip_p", 0.5)),
        A.VerticalFlip(p=aug_cfg.get("vertical_flip_p",   0.5)),
    ]

    if aug_name in ("flip", "flip_only"):
        return A.Compose(flips, bbox_params=bbox_params)

    if aug_name == "ot_color":
        # OTColorTransfer not yet implemented as Albumentations transform.
        # Falls back to flips until it is ready — plug in here when done.
        log.warning("ot_color not yet implemented — falling back to flip.")
        return A.Compose(flips, bbox_params=bbox_params)

    log.warning(f"Unknown augmentation '{aug_name}' — no augmentation applied.")
    return None


# ── NC weight saving ──────────────────────────────────────────────────────────

def _save_nc_weights(cfg, module, processor, run_name: str) -> None:
    """Save the NC baseline weights locally and/or push to HF Hub."""
    local_dir  = Path(cfg.paths.output_dir) / run_name / "hf_export"
    save_local = cfg.get("save_nc_local", False)
    hf_repo    = cfg.get("hf_repo")

    if not (save_local or hf_repo):
        return

    if hasattr(module.adapter, "save_pretrained"):
        module.adapter.save_pretrained(local_dir)
        log.info(f"NC weights saved locally → {local_dir}")

    if hf_repo:
        token = os.environ.get("HUGGING_FACE_API")
        if not token:
            log.error("HUGGING_FACE_API env var not set — skipping Hub push.")
            return
        module.adapter.model.push_to_hub(hf_repo, token=token)
        processor.push_to_hub(hf_repo, token=token)
        log.info(f"Model pushed to https://huggingface.co/{hf_repo}")


if __name__ == "__main__":
    main()