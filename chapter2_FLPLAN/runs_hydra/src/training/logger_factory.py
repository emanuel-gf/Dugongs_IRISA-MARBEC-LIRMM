"""
src/training/logger_factory.py
================================
Build the experiment logger from Hydra config.

cfg.logging.enabled = false  →  CSVLogger (debug / local runs, no W&B)
cfg.logging.enabled = true   →  WandbLogger (real runs)

Config groups
-------------
config/logging/debug.yaml
    enabled: false
    ...

config/logging/wandb.yaml
    enabled: true
    project: rtdetr-dugong
    entity: null
    tags: []
    log_model: false
    watch_model: false

Usage in train.py
-----------------
    from logger_factory import build_logger
    logger = build_logger(cfg)
    trainer = pl.Trainer(..., logger=logger)
"""

from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
from omegaconf import OmegaConf


def build_logger(cfg) -> pl.loggers.Logger:
    """
    Build a Lightning logger from Hydra cfg.

    Parameters
    ----------
    cfg : Hydra DictConfig — must have cfg.logging and cfg.run_name

    Returns
    -------
    WandbLogger if cfg.logging.enabled else CSVLogger
    """
    log_cfg  = cfg.logging
    run_name = cfg.run_name

    if not log_cfg.enabled:
        save_dir = Path(cfg.paths.log_dir) / f"S{cfg.seed}_{cfg.schema}_{cfg.partition}_{cfg.method}"
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [logger] W&B disabled — writing CSV logs to {save_dir}/{run_name}")
        return pl.loggers.CSVLogger(
            save_dir=str(save_dir),
            name=run_name,
        )

    # ── W&B ───────────────────────────────────────────────────────────────
    try:
        from pytorch_lightning.loggers import WandbLogger
        import wandb
    except ImportError:
        raise ImportError(
            "wandb is not installed. "
            "Run `pip install wandb` or set logging.enabled=false."
        )

    # Resolve tags: merge static tags from YAML with dynamic experiment tags
    tags = list(log_cfg.get("tags", []) or [])

    # Auto-add seed / partition / method from experiment cfg if present
    for key in ("schema", "partition", "method", "seed"):
        val = cfg.get(key)
        if val is not None and str(val) not in tags:
            tags.append(str(val))

    # Dump full resolved config to W&B
    full_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)

    print(
        f"  [logger] W&B enabled — project='{log_cfg.project}'  "
        f"name='{run_name}'  tags={tags}"
    )

    logger = WandbLogger(
        project  = log_cfg.project,
        entity   = log_cfg.get("entity") or None,
        name     = run_name,
        tags     = tags,
        log_model= log_cfg.get("log_model", False),
        config   = full_config,
    )

    if log_cfg.get("watch_model", False):
        # watch_model is called after the trainer attaches the model
        # We store the flag; DetectorLightningModule.on_fit_start can call it
        logger._watch_model = True

    return logger