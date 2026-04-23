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
which python

# ── config — must match run_NC.sh exactly ─────────────────────────────────────
SCHEMA="NWW"
CSVFILE="/share/home/e2406743/dataset/df_filepaths/df_train_test_split_filepath_wpsubset1000_10.csv"
CSVPATCHES="/share/home/e2406743/dataset/df_filepaths/df_train_test_split_filepath_PATCHES_wpartitions_seed_10.parquet"
PATCH_FOLDER="/share/home/e2406743/dataset/exported_img/seed_42"
OUTPUT_DIR="checkpoints"
OUTPUT_INFERENCE = "/share/home/e2406743/code/Dugongs_IRISA-MARBEC-LIRMM/output_inference/"
HF_REPO="manecomaneca/rtdetr"   # ← same repo NC pushed to
HF_REVISION="nc-best"                          # ← must match run_NC.sh exactly
NC_CHECKPOINT_DIR=/share/home/e2406743/code/Dugongs_IRISA-MARBEC-LIRMM/checkpoints/NNN_NC_SEED10_augm_0422_2254/hf_export
# ── partition array (index → name) ───────────────────────────────────────────
PARTITIONS=('partition_2', 'partition_4', 'partition_12', 'partition_24',
            'partition_36', 'partition_44', 'partition_46', 'partition_49')
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
    --batch-size    8              \
    --max-epochs    100              \
    --lr            1e-7            \
    --wandb-project "rtdetr-dugong" \
    --nc-checkpoint-dir "$NC_CHECKPOINT_DIR" \
    --output-inference "$OUTPUT_INFERENCE" \
    --early-stopping 25 \
    --augment 