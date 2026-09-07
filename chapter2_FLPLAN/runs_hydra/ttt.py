"""
src/train.py
============
Hydra entry point for RT-DETR fine-tuning on FLPLAN dugong patches.

Pipeline
--------
  1. Build run_name from experiment config + timestamp
  2. Build logger   (WandbLogger | CSVLogger)
  3. Build adapter  (RTDETRAdapter via registry)
  4. Build Lightning module (DetectorLightningModule)
  5. Build DataModule (reads resolved_paths.json)
  6. Build callbacks (checkpoint, early stopping, LR monitor)
  7. Build Trainer
  8. trainer.fit()
  9. trainer.test() on best checkpoint
  10. Save NC weights locally / push to HF Hub if configured

Usage
-----
  # Debug run (no W&B, quick check)
  python train.py

  # Real run with W&B
  python train.py logging=wandb

  # Override seed/partition/method
  python train.py seed=63 partition=p10 method=aclr logging=wandb

  # Full NC baseline
  python train.py experiment=nc_baseline logging=wandb

  # Fine-tune WP with ACLR 10%
  python train.py experiment=wp_aclr_p10 logging=wandb

  # Skip checkpoint saving (fast debug)
  python train.py save_checkpoints=false logging=debug
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoImageProcessor

from adapters_registry import build_adapter
from callbacks import build_callbacks
from datamodule import DugongDataModule
from detector_module import DetectorLightningModule
from logger_factory import build_logger

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:

    # ── 1. Run name ───────────────────────────────────────────────────────
    now      = datetime.now().strftime("%m%d_%H%M")
    run_name = (
        f"{cfg.schema}_{cfg.partition}_{cfg.method}"
        f"_seed{cfg.seed}_{cfg.model.name}_{now}"
    )
    # Make run_name accessible throughout cfg
    OmegaConf.update(cfg, "run_name", run_name, merge=True)

    log.info(f"Run: {run_name}")
    log.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # ── 2. Reproducibility ────────────────────────────────────────────────
    pl.seed_everything(cfg.seed, workers=True)

    # ── 3. Logger ─────────────────────────────────────────────────────────
    logger = build_logger(cfg)

    # ── 4. Adapter ────────────────────────────────────────────────────────
    # Load from NC checkpoint dir for fine-tune runs, else from HF Hub
    is_finetune = cfg.get("nc_checkpoint_dir") is not None

    if is_finetune:
        nc_dir = cfg.nc_checkpoint_dir
        assert Path(nc_dir).exists(), (
            f"nc_checkpoint_dir not found: {nc_dir}\n"
            "Run the NC baseline first with save_nc_local=true."
        )
        # Override checkpoint in model config to point to local NC weights
        OmegaConf.update(cfg, "model.checkpoint", str(nc_dir), merge=True)
        log.info(f"Fine-tune run: loading NC weights from {nc_dir}")
    else:
        log.info(f"NC run: loading base weights from {cfg.model.checkpoint}")

    adapter = build_adapter(cfg.model)
    log.info(f"Adapter built: {cfg.model.name}")

    # ── 5. Lightning module ───────────────────────────────────────────────
    module = DetectorLightningModule(adapter=adapter, cfg=cfg)

    # ── 6. Augmentor (optional Albumentations) ────────────────────────────
    augmentor = _build_augmentor(cfg)

    # ── 7. Processor + DataModule ─────────────────────────────────────────
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

    # ── 8. Callbacks ──────────────────────────────────────────────────────
    callbacks = build_callbacks(cfg, run_name)

    # ── 9. Trainer ────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs     = cfg.training.max_epochs,
        accelerator    = "auto",
        devices        = "auto",
        precision      = cfg.training.precision,
        logger         = logger,
        callbacks      = callbacks,
        log_every_n_steps = cfg.training.log_every_n_steps,
        gradient_clip_val = cfg.training.gradient_clip_val,
        enable_progress_bar = True,
        deterministic  = False,   # true is slower; set true if full reproducibility needed
    )

    # ── 10. Fit ───────────────────────────────────────────────────────────
    log.info("Starting training …")
    trainer.fit(module, datamodule=datamodule)
    log.info("Training complete")

    # ── 11. Test on best checkpoint ───────────────────────────────────────
    # PyTorch 2.6 changed torch.load default to weights_only=True, which
    # rejects OmegaConf DictConfig objects stored in Lightning checkpoints.
    # Allowlisting them is safe — they come from our own training run.
    import torch.serialization
    from omegaconf import DictConfig as _DictConfig
    from omegaconf.listconfig import ListConfig as _ListConfig
    torch.serialization.add_safe_globals([_DictConfig, _ListConfig])

    log.info("Running test on best checkpoint …")
    trainer.test(module, datamodule=datamodule, ckpt_path="best")

    # ── 12. Save NC weights locally / push to HF Hub ─────────────────────
    if not is_finetune:
        _save_nc_weights(cfg, module, processor, run_name)


# ── Augmentor factory ─────────────────────────────────────────────────────────

def _build_augmentor(cfg):
    """
    Build an Albumentations Compose from cfg.augmentation.

    Currently supports:
        none       — returns None
        flip_only  — horizontal + vertical flip
        ot_color   — flip + OT colour transfer (stub, not yet implemented)

    Returns None if augmentation is disabled or not configured.
    """
    aug_cfg = cfg.get("augmentation")
    if aug_cfg is None:
        return None

    aug_name = aug_cfg.get("name", "none")

    if aug_name == "none":
        return None

    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2  # noqa: F401
    except ImportError:
        log.warning("albumentations not installed — skipping augmentation.")
        return None

    if aug_name == "flip_only":
        return A.Compose(
            [
                A.HorizontalFlip(p=aug_cfg.get("horizontal_flip_p", 0.5)),
                A.VerticalFlip(p=aug_cfg.get("vertical_flip_p", 0.5)),
            ],
            bbox_params=A.BboxParams(
                format="coco",
                label_fields=["class_labels"],
                min_visibility=0.1,
            ),
        )

    if aug_name == "ot_color":
        # OT colour transfer not yet implemented as Albumentations transform.
        # Flip-only is used as fallback until it is available.
        log.warning(
            "ot_color augmentation not yet implemented — falling back to flip_only."
        )
        return A.Compose(
            [
                A.HorizontalFlip(p=aug_cfg.get("horizontal_flip_p", 0.5)),
                A.VerticalFlip(p=aug_cfg.get("vertical_flip_p", 0.5)),
                # OTColorTransfer(p=aug_cfg.ot_color_transfer.p) ← add here
            ],
            bbox_params=A.BboxParams(
                format="coco",
                label_fields=["class_labels"],
                min_visibility=0.1,
            ),
        )

    log.warning(f"Unknown augmentation '{aug_name}' — no augmentation applied.")
    return None


# ── NC weight saving ──────────────────────────────────────────────────────────

def _save_nc_weights(cfg, module, processor, run_name: str):
    """
    After an NC run: save weights locally and optionally push to HF Hub.
    """
    local_dir = Path(cfg.paths.output_dir) / run_name / "hf_export"

    save_local = cfg.get("save_nc_local", False)
    hf_repo    = cfg.get("hf_repo")

    if not (save_local or hf_repo):
        return

    if hasattr(module.adapter, "save_pretrained"):
        module.adapter.save_pretrained(local_dir)
        log.info(f"NC weights saved locally → {local_dir}")

    if hf_repo:
        _push_to_hub(cfg, module, processor, local_dir, hf_repo)


def _push_to_hub(cfg, module, processor, local_dir: Path, hf_repo: str):
    """Push model + processor to HF Hub."""
    token = os.environ.get("HUGGING_FACE_API")
    if not token:
        log.error("HUGGING_FACE_API env var not set — skipping Hub push.")
        return

    log.info(f"Pushing to HF Hub → {hf_repo}")
    module.adapter.model.push_to_hub(hf_repo, token=token)
    processor.push_to_hub(hf_repo, token=token)
    log.info(f"Model available at https://huggingface.co/{hf_repo}")


if __name__ == "__main__":
    main()