"""
src/training/callbacks.py
==========================
Build standard training callbacks from Hydra config.

Two training regimes are supported via cfg.training:

  checkpoint_mode: best   (default)
      ModelCheckpoint saves top-k by val/mAP_50 — for NC baseline runs
      where a real validation set exists.

  checkpoint_mode: last
      ModelCheckpoint saves only the final epoch — for fine-tune runs
      with a fixed epoch budget and no validation-based decisions
      (k-fold design where val is a mirror of test / disabled).

  use_early_stopping: true | false  (default: true)
      Must be false whenever no legitimate val set exists.

Callbacks returned
------------------
- ModelCheckpoint      — mode-dependent (see above), or disabled
- EarlyStopping        — only if use_early_stopping=true
- LearningRateMonitor  — always

Usage in train.py
-----------------
    from src.training.callbacks import build_callbacks
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
    t         = cfg.training
    ckpt_dir  = Path(cfg.paths.output_dir) / run_name
    ckpt_mode = t.get("checkpoint_mode", "best")   # "best" | "last"

    assert ckpt_mode in ("best", "last"), (
        f"training.checkpoint_mode must be 'best' or 'last', got '{ckpt_mode}'"
    )

    callbacks: list = []

    # ── ModelCheckpoint ───────────────────────────────────────────────────
    save_checkpoints = cfg.get("save_checkpoints", True)

    if not save_checkpoints:
        callbacks.append(
            ModelCheckpoint(save_top_k=0, save_weights_only=True)
        )
        print("  [callbacks] Checkpoint saving DISABLED")

    elif ckpt_mode == "best":
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            ModelCheckpoint(
                dirpath   = str(ckpt_dir),
                filename  = f"{run_name}-{{epoch:02d}}-{{val/mAP_50:.4f}}",
                monitor   = "val/mAP_50",
                mode      = "max",
                save_top_k= 3,
                save_last = False,
                verbose   = True,
                save_weights_only=True,
            )
        )
        print(f"  [callbacks] Checkpoints (best val/mAP_50, top-3) → {ckpt_dir}")

    else:  # ckpt_mode == "last"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            ModelCheckpoint(
                dirpath   = str(ckpt_dir),
                filename  = f"{run_name}-final-{{epoch:02d}}",
                monitor   = None,          # no metric — save unconditionally
                save_top_k= 1,             # keep only the most recent epoch
                every_n_epochs=1,
                save_last = True,          # also writes 'last.ckpt' symlink/file
                verbose   = False,
                save_weights_only=True,
            )
        )
        print(f"  [callbacks] Checkpoints (final epoch only) → {ckpt_dir}")

    # ── EarlyStopping ─────────────────────────────────────────────────────
    use_early_stopping = t.get("use_early_stopping", True)

    if use_early_stopping and ckpt_mode == "last":
        # Early stopping consumes val — invalid in the no-val regime.
        print("  [callbacks] WARNING: use_early_stopping=true is incompatible "
              "with checkpoint_mode=last (no-val design) — DISABLING it.")
        use_early_stopping = False

    if use_early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor  = "val/mAP_50",
                patience = t.early_stopping_patience,
                mode     = "max",
                verbose  = True,
            )
        )
        print(f"  [callbacks] EarlyStopping on val/mAP_50 "
              f"(patience={t.early_stopping_patience})")
    else:
        print("  [callbacks] EarlyStopping DISABLED — fixed epoch budget "
              f"({t.max_epochs} epochs)")

    # ── LearningRateMonitor ───────────────────────────────────────────────
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    return callbacks