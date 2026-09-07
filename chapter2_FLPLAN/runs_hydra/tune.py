"""
tune.py
=======
Optuna hyperparameter search for RT-DETR fine-tuning.

Uses p10 / aclr / seed=0 as the proxy partition — fastest to train,
representative of the small-data regime where HP sensitivity is highest.

Usage
-----
  # Local / interactive (debug, 5 trials)
  python tune.py --n-trials 5 --max-epochs 5

  # Full search on SLURM (50 trials, parallel workers)
  python tune.py --n-trials 50 --max-epochs 15

  # Resume an interrupted study
  python tune.py --n-trials 50 --study-name rtdetr_flplan_v1

  # Analyse results without running
  python tune.py --analyse-only --study-name rtdetr_flplan_v1

Search space
------------
  lr                : log-uniform [1e-6, 1e-3]
  backbone_lr_factor: log-uniform [0.01, 0.5]
  weight_decay      : log-uniform [1e-5, 1e-2]
  gradient_clip_val : log-uniform [0.01, 1.0]
  warmup_epochs     : int         [1, 5]

Objective
---------
  Maximise val/mAP_50 at the end of training.
  MedianPruner kills trials that are below median at epoch 3.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import optuna
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from transformers import AutoImageProcessor

# ── make src importable when running from runs_hydra/ root ───────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.adapters.registry import build_adapter
from src.data.datamodule import DugongDataModule
from src.training.module import DetectorLightningModule

log = logging.getLogger(__name__)

# ── Suppress noisy loggers ────────────────────────────────────────────────────
logging.getLogger("faster_coco_eval.core.cocoeval").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger("lightning_fabric").setLevel(logging.WARNING)


# ── LightningModule with tuneable HP ─────────────────────────────────────────

class TunableDetectorModule(DetectorLightningModule):
    """
    Thin subclass that overrides configure_optimizers with Optuna-suggested HP.
    Everything else (forward, val/test steps, mAP logging) is inherited unchanged.
    """

    def __init__(self, adapter, cfg, trial_hp: dict):
        super().__init__(adapter=adapter, cfg=cfg, inference_dir=None)
        self.trial_hp = trial_hp   # overrides cfg.training values

    def configure_optimizers(self):
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        hp  = self.trial_hp
        lr  = hp["lr"]
        blf = hp["backbone_lr_factor"]
        wd  = hp["weight_decay"]
        me  = self.cfg.training.max_epochs
        we  = hp["warmup_epochs"]

        backbone_params, head_params = self.adapter.backbone_and_head_params()
        optimizer = AdamW(
            [
                {"params": backbone_params, "lr": lr * blf},
                {"params": head_params,     "lr": lr},
            ],
            weight_decay=wd,
        )

        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=we)
        cosine = CosineAnnealingLR(optimizer, T_max=max(me - we, 1), eta_min=lr * blf * 0.01)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[we])

        return {
            "optimizer":    optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ── Optuna PyTorch Lightning callback for pruning ─────────────────────────────

class OptunaPruningCallback(pl.Callback):
    """Reports val/mAP_50 to Optuna after each epoch and prunes if needed."""

    def __init__(self, trial: optuna.Trial, monitor: str = "val/mAP_50"):
        self.trial   = trial
        self.monitor = monitor

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch   = trainer.current_epoch
        value   = trainer.callback_metrics.get(self.monitor)
        if value is None:
            return

        self.trial.report(float(value), step=epoch)

        if self.trial.should_prune():
            raise optuna.TrialPruned(
                f"Trial pruned at epoch {epoch} "
                f"({self.monitor}={float(value):.4f})"
            )


# ── Objective ─────────────────────────────────────────────────────────────────

def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """
    One Optuna trial = one full training run with sampled HP.
    Returns val/mAP_50 of the best epoch (or last epoch if no improvement).
    """

    # ── Sample hyperparameters ────────────────────────────────────────────────
    trial_hp = {
        "lr":                 trial.suggest_float("lr",                 1e-6, 1e-3, log=True),
        "backbone_lr_factor": trial.suggest_float("backbone_lr_factor", 0.01, 0.5,  log=True),
        "weight_decay":       trial.suggest_float("weight_decay",       1e-5, 1e-2, log=True),
        "gradient_clip_val":  trial.suggest_float("gradient_clip_val",  0.01, 1.0,  log=True),
        "warmup_epochs":      trial.suggest_int(  "warmup_epochs",      1,    5),
    }

    log.info(
        f"Trial {trial.number} | "
        + "  ".join(f"{k}={v:.2e}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in trial_hp.items())
    )

    # ── Build config (minimal, no Hydra needed here) ──────────────────────────
    cfg = OmegaConf.create({
        "model": {
            "name":                 "rtdetr",
            "checkpoint":          args.nc_checkpoint_dir,
            "confidence_threshold": 0.01,
            "num_classes":          80,
            "id2label":            {0: "dugong"},
        },
        "training": {
            "lr":                   trial_hp["lr"],
            "backbone_lr_factor":   trial_hp["backbone_lr_factor"],
            "weight_decay":         trial_hp["weight_decay"],
            "batch_size":           args.batch_size,
            "max_epochs":           args.max_epochs,
            "early_stopping_patience": args.max_epochs,  # disable early stop during search
            "warmup_epochs":        trial_hp["warmup_epochs"],
            "gradient_clip_val":    trial_hp["gradient_clip_val"],
            "precision":           "16-mixed",
            "log_every_n_steps":    1,
            "num_workers":          args.num_workers,
        },
        "data": {
            "num_workers": args.num_workers,
            "pin_memory":  True,
        },
        "run_name": f"optuna_trial_{trial.number}",
    })

    # ── Build adapter + module ────────────────────────────────────────────────
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    adapter = build_adapter(cfg.model)
    module  = TunableDetectorModule(adapter=adapter, cfg=cfg, trial_hp=trial_hp)

    # ── DataModule ────────────────────────────────────────────────────────────
    processor  = AutoImageProcessor.from_pretrained(args.nc_checkpoint_dir)
    datamodule = DugongDataModule(
        resolved_paths_json = args.resolved_paths_json,
        seed        = args.seed,
        partition   = args.partition,
        method      = args.method,
        processor   = processor,
        batch_size  = args.batch_size,
        augmentor   = None,   # no augmentation during HP search — faster + less variance
        num_workers = args.num_workers,
        pin_memory  = True,
    )

    # ── Trainer (no logger, no checkpoints — pure speed) ─────────────────────
    pruning_cb = OptunaPruningCallback(trial, monitor="val/mAP_50")

    trainer = pl.Trainer(
        max_epochs           = args.max_epochs,
        accelerator          = "auto",
        devices              = "auto",
        precision            = "16-mixed",
        logger               = False,        # no W&B/CSV during search
        enable_checkpointing = False,        # no disk writes
        enable_progress_bar  = args.verbose,
        num_sanity_val_steps = 0,
        gradient_clip_val    = trial_hp["gradient_clip_val"],
        callbacks            = [pruning_cb],
    )

    try:
        trainer.fit(module, datamodule=datamodule)
    except optuna.TrialPruned:
        raise
    except Exception as e:
        log.warning(f"Trial {trial.number} failed with: {e}")
        return 0.0

    # Return best val/mAP_50 seen during training
    val_map_50 = trainer.callback_metrics.get("val/mAP_50", torch.tensor(0.0))
    return float(val_map_50)


# ── Analysis helper ───────────────────────────────────────────────────────────

def analyse_study(study: optuna.Study) -> None:
    """Print a clean summary of the completed study."""
    print("\n" + "="*60)
    print(f"  Study: {study.study_name}")
    print(f"  Trials completed : {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"  Trials pruned    : {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"  Trials failed    : {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
    print("="*60)

    best = study.best_trial
    print(f"\n  Best trial: #{best.number}  val/mAP_50={best.value:.4f}")
    print("\n  Best hyperparameters:")
    for k, v in best.params.items():
        if isinstance(v, float):
            print(f"    {k:30s}: {v:.2e}")
        else:
            print(f"    {k:30s}: {v}")

    print("\n  Top 5 trials:")
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    for t in sorted(completed, key=lambda x: x.value or 0, reverse=True)[:5]:
        print(
            f"    #{t.number:3d}  mAP_50={t.value:.4f}  "
            f"lr={t.params['lr']:.1e}  "
            f"blf={t.params['backbone_lr_factor']:.2f}  "
            f"wd={t.params['weight_decay']:.1e}  "
            f"clip={t.params['gradient_clip_val']:.2f}  "
            f"warmup={t.params['warmup_epochs']}"
        )

    print("\n  → Add to config.yaml:")
    print(f"    training:")
    print(f"      lr:                 {best.params['lr']:.2e}")
    print(f"      backbone_lr_factor: {best.params['backbone_lr_factor']:.4f}")
    print(f"      weight_decay:       {best.params['weight_decay']:.2e}")
    print(f"      gradient_clip_val:  {best.params['gradient_clip_val']:.4f}")
    print(f"      warmup_epochs:      {best.params['warmup_epochs']}")
    print("="*60 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Optuna HP search for RT-DETR fine-tuning")

    # Study config
    p.add_argument("--study-name",   default="rtdetr_flplan_v1")
    p.add_argument("--storage",      default=None,
                   help="Optuna DB URI. Default: sqlite in log_dir. "
                        "For parallel: 'sqlite:///optuna.db' or MySQL URI")
    p.add_argument("--n-trials",     type=int, default=40)
    p.add_argument("--n-startup",    type=int, default=10,
                   help="Random trials before TPE kicks in")
    p.add_argument("--n-warmup-pruning", type=int, default=3,
                   help="Epochs before pruner can prune a trial")
    p.add_argument("--analyse-only", action="store_true",
                   help="Load existing study and print analysis, no new trials")

    # Training config
    p.add_argument("--max-epochs",   type=int, default=15,
                   help="Epochs per trial. 10-20 is enough for HP ranking.")
    p.add_argument("--batch-size",   type=int, default=2)
    p.add_argument("--num-workers",  type=int, default=4)
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--partition",    default="p10")
    p.add_argument("--method",       default="aclr")
    p.add_argument("--verbose",      action="store_true")

    # Paths
    p.add_argument(
        "--nc-checkpoint-dir",
        default="/share/home/e2406743/code/Dugongs_IRISA-MARBEC-LIRMM/"
                "checkpoints/NNN_NC_SEED63_augm_0510_1843/hf_export",
    )
    p.add_argument(
        "--resolved-paths-json",
        default="/share/home/e2406743/code/Dugongs_IRISA-MARBEC-LIRMM/"
                "chapter2_FLPLAN/splits_json/splits_seeds012_mapped.json",
    )
    p.add_argument(
        "--log-dir",
        default="/share/home/e2406743/code/Dugongs_IRISA-MARBEC-LIRMM/"
                "logs_logger_flplan/optuna",
    )

    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s] %(message)s",
    )
    args = parse_args()

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # ── Storage ───────────────────────────────────────────────────────────────
    # Default: SQLite in log_dir so the study persists and can be resumed.
    storage = args.storage or f"sqlite:///{args.log_dir}/{args.study_name}.db"
    log.info(f"Study storage: {storage}")

    # ── Create or load study ──────────────────────────────────────────────────
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=args.n_startup,
        seed=args.seed,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials  = args.n_startup,
        n_warmup_steps    = args.n_warmup_pruning,   # don't prune before epoch 3
        interval_steps    = 1,
    )

    study = optuna.create_study(
        study_name     = args.study_name,
        storage        = storage,
        direction      = "maximize",
        sampler        = sampler,
        pruner         = pruner,
        load_if_exists = True,   # resume if study already exists
    )

    if args.analyse_only:
        analyse_study(study)
        return

    # ── Run ───────────────────────────────────────────────────────────────────
    log.info(
        f"Starting Optuna search | "
        f"study={args.study_name} | "
        f"n_trials={args.n_trials} | "
        f"max_epochs={args.max_epochs} | "
        f"partition={args.partition} | "
        f"method={args.method}"
    )

    study.optimize(
        lambda trial: objective(trial, args),
        n_trials          = args.n_trials,
        catch             = (RuntimeError,),   # catch CUDA OOM etc.
        show_progress_bar = True,
    )

    analyse_study(study)

    # ── Save best params as YAML snippet ─────────────────────────────────────
    best_yaml = Path(args.log_dir) / f"{args.study_name}_best_params.yaml"
    with open(best_yaml, "w") as f:
        f.write("# Best hyperparameters from Optuna study\n")
        f.write(f"# study: {args.study_name}\n")
        f.write(f"# val/mAP_50: {study.best_value:.4f}\n\n")
        f.write("training:\n")
        for k, v in study.best_params.items():
            if isinstance(v, float):
                f.write(f"  {k}: {v:.6f}\n")
            else:
                f.write(f"  {k}: {v}\n")
    log.info(f"Best params saved → {best_yaml}")


if __name__ == "__main__":
    main()