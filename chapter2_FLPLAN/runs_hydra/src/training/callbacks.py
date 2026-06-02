"""
src/training/callbacks.py
==========================
Build standard training callbacks from Hydra config.

Callbacks returned
------------------
- ModelCheckpoint   — saves top-k by val/mAP_50 (or disables saving)
- EarlyStopping     — monitors val/mAP_50
- LearningRateMonitor

Usage in train.py
-----------------
    from callbacks import build_callbacks
    callbacks = build_callbacks(cfg, run_name)
    trainer   = pl.Trainer(..., callbacks=callbacks)
"""

from __future__ import annotations

from pathlib import Path

from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)


def build_callbacks(cfg, run_name: str) -> list:
    """
    Build Lightning callbacks from Hydra config.

    Parameters
    ----------
    cfg      : Hydra DictConfig — must have cfg.training and cfg.paths
    run_name : unified run identifier (used as checkpoint subdirectory)

    Returns
    -------
    list of Lightning Callback objects
    """
    t        = cfg.training
    ckpt_dir = Path(cfg.paths.output_dir) / run_name

    # ── ModelCheckpoint ───────────────────────────────────────────────────
    save_checkpoints = cfg.get("save_checkpoints", True)

    if save_checkpoints:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_cb = ModelCheckpoint(
            dirpath   = str(ckpt_dir),
            filename  = f"{run_name}-{{epoch:02d}}-{{val/mAP_50:.4f}}",
            monitor   = "val/mAP_50",
            mode      = "max",
            save_top_k= 3,
            save_last = False,
            verbose   = True,
            save_weights_only=True,
        )
        print(f"  [callbacks] Checkpoints → {ckpt_dir}")
    else:
        checkpoint_cb = ModelCheckpoint(save_top_k=0,
                                        save_weights_only=True
                                        )
        print("  [callbacks] Checkpoint saving DISABLED")

    # ── EarlyStopping ─────────────────────────────────────────────────────
    early_stop_cb = EarlyStopping(
        monitor  = "val/mAP_50",
        patience = t.early_stopping_patience,
        mode     = "max",
        verbose  = True,
    )

    # ── LearningRateMonitor ───────────────────────────────────────────────
    lr_monitor_cb = LearningRateMonitor(logging_interval="epoch")

    return [checkpoint_cb, early_stop_cb, lr_monitor_cb]