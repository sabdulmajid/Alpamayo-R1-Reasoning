#!/bin/bash
#SBATCH --job-name=mismatch_2k
#SBATCH --partition=midcard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=28G
#SBATCH --time=06:00:00
#SBATCH --chdir=/mnt/slurm_nfs/a6abdulm/projects/alpamayo-r1-research
#SBATCH --output=/mnt/slurm_nfs/a6abdulm/projects/alpamayo-r1-research/logs/mismatch_2k_%j.out
#SBATCH --error=/mnt/slurm_nfs/a6abdulm/projects/alpamayo-r1-research/logs/mismatch_2k_%j.err

echo "=========================================="
echo "MISMATCH 2K — Batch Job $SLURM_JOB_ID"
echo "Node: $(hostname)  Start: $(date)"
echo "=========================================="

export HF_TOKEN="${HF_TOKEN:-}"
export HF_HOME="/mnt/slurm_nfs/a6abdulm/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
export PYTORCH_NVML_BASED_CUDA_CHECK=0

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN is not set. Export your token before submitting this job."
    exit 1
fi

source /mnt/slurm_nfs/a6abdulm/miniconda3/etc/profile.d/conda.sh
conda activate alpamayo

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Process up to 400 clips (resumes from last checkpoint automatically)
python src/run_mismatch_batch.py --batch-size 400

# Check how many remain
DONE=$(python3 -c "
import pandas as pd
from pathlib import Path
csv = Path('results/mismatch_2k/all_results.csv')
total = len(pd.read_parquet('data/eval_clips_2k.parquet'))
done = len(pd.read_csv(csv)) if csv.exists() else 0
print(f'{done}/{total}')
remaining = total - done
print(remaining)
")
REMAINING=$(echo "$DONE" | tail -1)
echo "Progress: $(echo "$DONE" | head -1) clips done, $REMAINING remaining"

# Auto-submit next batch if work remains
if [ "$REMAINING" -gt 0 ]; then
    NEXT_JOB=$(sbatch --parsable slurm/run_mismatch_2k.sh)
    echo "Submitted next batch: job $NEXT_JOB"
else
    echo "ALL 2000 CLIPS COMPLETE!"
fi

echo "=========================================="
echo "Batch done: $(date)"
