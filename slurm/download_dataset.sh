#!/bin/bash
#SBATCH --job-name=alpamayo_data
#SBATCH --partition=midcard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --chdir=/mnt/slurm_nfs/a6abdulm/projects/alpamayo-r1-research
#SBATCH --output=/mnt/slurm_nfs/a6abdulm/projects/alpamayo-r1-research/logs/download_%j.out
#SBATCH --error=/mnt/slurm_nfs/a6abdulm/projects/alpamayo-r1-research/logs/download_%j.err

echo "=========================================="
echo "PHYSICALAI-AV DATASET DOWNLOAD"
echo "Downloading subset for mismatch experiments"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo ""

# HuggingFace authentication
# Expect HF_TOKEN to be provided by the user environment.
export HF_TOKEN="${HF_TOKEN:-}"
export HF_HOME="/mnt/slurm_nfs/a6abdulm/.cache/huggingface"

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN is not set. Export your token before submitting this job."
    exit 1
fi

# Activate conda environment
source /mnt/slurm_nfs/a6abdulm/miniconda3/etc/profile.d/conda.sh
conda activate alpamayo

echo "Python: $(which python)"
echo "Disk space available:"
df -h /mnt/slurm_nfs/a6abdulm/projects/alpamayo-r1-research/data
echo ""

# Run the download script
python src/download_dataset.py \
    --output-dir data/physicalai_av \
    --num-clips 500 \
    --scenarios rain,fog,night,clear,heavy_traffic,cut_in

echo ""
echo "=========================================="
echo "Download completed: $(date)"
