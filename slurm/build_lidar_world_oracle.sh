#!/bin/bash
#SBATCH --job-name=lidar_oracle
#SBATCH --partition=dualcard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=0-7%2
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python3}
CLIP_PARQUET=${CLIP_PARQUET:-${PROJECT_DIR}/data/eval_clips_2k.parquet}
CANDIDATE_DIR=${CANDIDATE_DIR:-${PROJECT_DIR}/results/candidates_k6}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_DIR}/results/lidar_world_oracle}
DATASET_CACHE=${DATASET_CACHE:-}

if [[ ! -x "$(command -v "${PYTHON_BIN}" 2>/dev/null || true)" ]]; then
    echo "ERROR: PYTHON_BIN does not resolve to an executable: ${PYTHON_BIN}" >&2
    exit 2
fi
if [[ ! -f "${CLIP_PARQUET}" ]]; then
    echo "ERROR: clip parquet does not exist: ${CLIP_PARQUET}" >&2
    exit 2
fi
if [[ ! -d "${CANDIDATE_DIR}" ]]; then
    echo "ERROR: candidate output directory does not exist: ${CANDIDATE_DIR}" >&2
    exit 2
fi
if [[ ! -d "${CANDIDATE_DIR}/records" ]]; then
    echo "ERROR: candidate records directory does not exist: ${CANDIDATE_DIR}/records" >&2
    exit 2
fi
for name in SLURM_ARRAY_TASK_ID SLURM_ARRAY_TASK_COUNT SLURM_ARRAY_TASK_MIN SLURM_ARRAY_TASK_MAX SLURM_ARRAY_TASK_STEP; do
    if [[ -z "${!name:-}" ]]; then
        echo "ERROR: ${name} is unset; submit this script as a SLURM array" >&2
        exit 2
    fi
done
if (( SLURM_ARRAY_TASK_MIN != 0 || SLURM_ARRAY_TASK_STEP != 1 || SLURM_ARRAY_TASK_MAX + 1 != SLURM_ARRAY_TASK_COUNT )); then
    echo "ERROR: array indices must be contiguous and zero-based" >&2
    exit 2
fi

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_DIR}"

srun --ntasks=1 "${PYTHON_BIN}" -c \
    "import DracoPy, physical_ai_av; print('LiDAR dependencies available')"

ARGS=(
    --clip-parquet "${CLIP_PARQUET}"
    --candidate-dir "${CANDIDATE_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --num-shards "${SLURM_ARRAY_TASK_COUNT}"
    --shard-index "${SLURM_ARRAY_TASK_ID}"
    --resume
)
if [[ -n "${DATASET_CACHE}" ]]; then
    ARGS+=(--cache-dir "${DATASET_CACHE}")
fi

srun --ntasks=1 "${PYTHON_BIN}" src/lidar_world_oracle.py "${ARGS[@]}"
