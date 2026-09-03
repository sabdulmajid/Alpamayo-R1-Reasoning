#!/bin/bash
#SBATCH --job-name=bev_world_train
#SBATCH --partition=dualcard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00

set -euo pipefail

REPO_ROOT="${WORLD_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${WORLD_PYTHON_BIN:-python}"
TRAIN_MANIFEST="${WORLD_TRAIN_MANIFEST:?Set WORLD_TRAIN_MANIFEST to train.jsonl}"
VAL_MANIFEST="${WORLD_VAL_MANIFEST:?Set WORLD_VAL_MANIFEST to val.jsonl}"
OUTPUT_DIR="${WORLD_OUTPUT_DIR:?Set WORLD_OUTPUT_DIR to a checkpoint directory}"

cd "$REPO_ROOT"
test -r "$TRAIN_MANIFEST"
test -r "$VAL_MANIFEST"
mkdir -p "$OUTPUT_DIR"
srun --nodes=1 --ntasks=1 "$PYTHON_BIN" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory)
PY
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

COMMAND=(
    "$PYTHON_BIN" -m src.world_model.train
    --train-manifest "$TRAIN_MANIFEST"
    --val-manifest "$VAL_MANIFEST"
    --output-dir "$OUTPUT_DIR"
    --epochs "${WORLD_EPOCHS:-20}"
    --batch-size "${WORLD_BATCH_SIZE:-4}"
    --workers "${WORLD_WORKERS:-8}"
    --base-channels "${WORLD_BASE_CHANNELS:-16}"
    --seed "${WORLD_SEED:-2026}"
    --device cuda
    --amp
)
if [[ -n "${WORLD_RESUME:-}" ]]; then
    COMMAND+=(--resume "$WORLD_RESUME")
fi
srun --nodes=1 --ntasks=1 "${COMMAND[@]}"
