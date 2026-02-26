#!/bin/bash
#SBATCH --job-name=alpamayo_probe
#SBATCH --partition=midcard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=28G
#SBATCH --time=01:00:00
#SBATCH --chdir=/mnt/slurm_nfs/a6abdulm/projects/alpamayo-r1-research
#SBATCH --output=/mnt/slurm_nfs/a6abdulm/projects/alpamayo-r1-research/logs/probe_%j.out
#SBATCH --error=/mnt/slurm_nfs/a6abdulm/projects/alpamayo-r1-research/logs/probe_%j.err

echo "=========================================="
echo "ALPAMAYO-R1 MODEL FIT PROBE"
echo "Testing bfloat16 loading on A4500 (20GB)"
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

# Bypass NVML-dependent caching allocator (driver/library version mismatch on node)
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
export PYTORCH_NVML_BASED_CUDA_CHECK=0

# Activate conda environment (Python 3.12 with alpamayo_r1)
source /mnt/slurm_nfs/a6abdulm/miniconda3/etc/profile.d/conda.sh
conda activate alpamayo

# Show environment info
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Run the probe script
python src/probe_model_fit.py

echo ""
echo "=========================================="
echo "Probe completed: $(date)"
