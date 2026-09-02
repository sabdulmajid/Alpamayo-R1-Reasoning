#!/bin/bash
#SBATCH --job-name=bev_world_eval
#SBATCH --partition=dualcard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=04:00:00

set -euo pipefail

REPO_ROOT="${WORLD_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${WORLD_PYTHON_BIN:-python}"
MANIFEST="${WORLD_EVAL_MANIFEST:?Set WORLD_EVAL_MANIFEST to val.jsonl or test.jsonl}"
METHOD="${WORLD_METHOD:-learned}"
OUTPUT="${WORLD_METRICS_OUTPUT:?Set WORLD_METRICS_OUTPUT to a JSON path}"

cd "$REPO_ROOT"
test -r "$MANIFEST"
if [[ "$METHOD" == "learned" ]]; then
    CHECKPOINT="${WORLD_CHECKPOINT:?Set WORLD_CHECKPOINT for learned evaluation}"
    test -r "$CHECKPOINT"
fi
srun --nodes=1 --ntasks=1 "$PYTHON_BIN" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory)
PY

COMMAND=(
    "$PYTHON_BIN" -m src.world_model.evaluate
    --manifest "$MANIFEST"
    --method "$METHOD"
    --output "$OUTPUT"
    --batch-size "${WORLD_BATCH_SIZE:-4}"
    --workers "${WORLD_WORKERS:-8}"
    --device cuda
)
if [[ "$METHOD" == "learned" ]]; then
    COMMAND+=(--checkpoint "$CHECKPOINT" --amp)
fi
if [[ -n "${WORLD_PREDICTION_DIR:-}" ]]; then
    COMMAND+=(--prediction-dir "$WORLD_PREDICTION_DIR")
fi
srun --nodes=1 --ntasks=1 "${COMMAND[@]}"
