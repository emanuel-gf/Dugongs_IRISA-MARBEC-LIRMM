#!/bin/sh

# --- SLURM Directives ---
#SBATCH --job-name=rtdetr_dugong
#SBATCH --partition=shortrun            # Use longrun if training takes > 2 days
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --constraint="titan|a6000"      # Prioritize Titan (24G) or A6000 (48G)
#SBATCH --cpus-per-task=4               # Matching your num_workers in Python
#SBATCH --mem=32G                       # Global RAM (CPU memory)
#SBATCH --time=12:00:00                 # 12 hours max
#SBATCH --output=logs/%j_stdout.log     # %j inserts the Job ID
#SBATCH --error=logs/%j_stderr.log

# --- Environment Setup ---
# Load your specific environment (adjust path as needed)
. /share/common/anaconda/etc/profile.d/conda.sh
conda activate fifty

which python

# Ensure the logs folder exists
mkdir -p logs

# --- Run the Python script ---
# We use the arguments defined in your script's parse_args()
python ./fiftyone/train.py \
    --schema "NWW" \
    --partition "NC" \
    --csvfile "/share/home/e2406743/dataset/df_filepaths/df_train_test_split_filepath_38.csv" \
    --patch-folder "/share/home/e2406743/dataset/exported_img/seed_42" \
    --batch-size 16 \
    --max-epochs 10 \
    --lr 1e-4                       