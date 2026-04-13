#!/bin/bash
# =============================================================================
# STEP 2 — Fine-tune from the NC checkpoint for all domain-shift partitions.
#           Each partition is an independent SLURM array task on its own GPU.
#
# IMPORTANT: run_NC.sh must have completed and pushed to HF Hub before
#            you submit this script.
#
# Submit : sbatch run_finetune.sh
# Monitor: squeue -u $USER
# Logs   : logs/<jobid>_<taskid>_ft_stdout.log
# =============================================================================

#SBATCH --job-name=rtdetr_ft
#SBATCH --partition=shortrun
#SBATCH --gres=gpu:1
#SBATCH --constraint="titan|a6000"
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-5                          # 6 partitions → task IDs 0..5
#SBATCH --output=logs/%j_%a_ft_stdout.log    # %j=job, %a=array index
#SBATCH --error=logs/%j_%a_ft_stderr.log

# ── environment ───────────────────────────────────────────────────────────────
. /share/common/anaconda/etc/profile.d/conda.sh
conda activate fifty
mkdir -p logs
which python

# ── config — must match run_NC.sh exactly ─────────────────────────────────────
SCHEMA="NWW"
CSVFILE="/share/home/e2406743/dataset/df_filepaths/df_train_test_split_filepath_38.csv"
CSVPATCHES="/share/home/e2406743/dataset/df_filepaths/df_train_test_split_filepath_PATCHES_wpartitions_seed_38.parquet"
PATCH_FOLDER="/share/home/e2406743/dataset/exported_img/seed_42"
OUTPUT_DIR="checkpoints"
OUTPUT_INFERENCE = "inference_files"
HF_REPO="manecomaneca/rtdetr"   # ← same repo NC pushed to
HF_REVISION="nc-best"                          # ← must match run_NC.sh exactly

# ── partition array (index → name) ───────────────────────────────────────────
PARTITIONS=("partition_5" "partition_10" "partition_25"
            "partition_50" "partition_75" "partition_100")
PARTITION="${PARTITIONS[$SLURM_ARRAY_TASK_ID]}"

echo "============================================"
echo "Array task : $SLURM_ARRAY_TASK_ID"
echo "Partition  : $PARTITION"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "GPU        : $CUDA_VISIBLE_DEVICES"
echo "Loading NC weights from: $HF_REPO @ revision '$HF_REVISION'"
echo "============================================"

# ── run ───────────────────────────────────────────────────────────────────────
python ./fiftyone/train.py \
    --schema        "$SCHEMA"       \
    --partition     "$PARTITION"    \
    --csvfile       "$CSVFILE"      \
    --csvpatches    "$CSVPATCHES"   \
    --patch-folder  "$PATCH_FOLDER" \
    --output-dir    "$OUTPUT_DIR"   \
    --batch-size    16              \
    --max-epochs    45              \
    --lr            5e-5            \
    --wandb-project "rtdetr-dugong" \
    --nc-checkpoint-dir "/share/home/e2406743/code/Dugongs_IRISA-MARBEC-LIRMM/checkpoints/NNN_NC_SEED38_augm_0412_1511/hf_export" \
    --output-inference "$OUTPUT_INFERENCE" \
    --early-stopping 20 \