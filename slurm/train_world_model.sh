#!/bin/bash
#SBATCH --job-name=bev_world_train
#SBATCH --partition=dualcard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=28G
#SBATCH --time=20:00:00

set -euo pipefail

REPO_ROOT="${WORLD_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${WORLD_PYTHON_BIN:-python}"
TRAIN_MANIFEST="${WORLD_TRAIN_MANIFEST:?Set WORLD_TRAIN_MANIFEST to train.jsonl}"
VAL_MANIFEST="${WORLD_VAL_MANIFEST:?Set WORLD_VAL_MANIFEST to val.jsonl}"
PROTOCOL="${WORLD_PROTOCOL:?Set WORLD_PROTOCOL to protocol.json}"
OUTPUT_DIR="${WORLD_OUTPUT_DIR:?Set WORLD_OUTPUT_DIR to a checkpoint directory}"

cd "$REPO_ROOT"
test -r "$TRAIN_MANIFEST"
test -r "$VAL_MANIFEST"
test -r "$PROTOCOL"
mkdir -p "$OUTPUT_DIR"
srun --nodes=1 --ntasks=1 "$PYTHON_BIN" - <<'PY'
import os
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
if torch.cuda.device_count() != 1:
    raise SystemExit(f"Expected one visible GPU, found {torch.cuda.device_count()}")
name = torch.cuda.get_device_name(0)
expected = os.environ.get("WORLD_EXPECTED_GPU_NAME", "NVIDIA RTX A4500")
if name != expected:
    raise SystemExit(f"Expected GPU {expected!r}, found {name!r}")
print(name, torch.cuda.get_device_properties(0).total_memory)
PY
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

COMMAND=(
    "$PYTHON_BIN" -m src.world_model.train
    --protocol "$PROTOCOL"
    --train-manifest "$TRAIN_MANIFEST"
    --val-manifest "$VAL_MANIFEST"
    --output-dir "$OUTPUT_DIR"
    --epochs "${WORLD_EPOCHS:-30}"
    --batch-size "${WORLD_BATCH_SIZE:-4}"
    --workers "${WORLD_WORKERS:-4}"
    --base-channels "${WORLD_BASE_CHANNELS:-16}"
    --learning-rate "${WORLD_LEARNING_RATE:-0.0003}"
    --weight-decay "${WORLD_WEIGHT_DECAY:-0.0001}"
    --seed "${WORLD_SEED:-2026}"
    --device cuda
    --amp
)
if [[ -n "${WORLD_RESUME:-}" ]]; then
    COMMAND+=(--resume "$WORLD_RESUME")
fi
srun --nodes=1 --ntasks=1 "${COMMAND[@]}"
