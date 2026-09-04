#!/bin/bash
#SBATCH --job-name=bev_world_eval
#SBATCH --partition=dualcard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=28G
#SBATCH --time=08:00:00

set -euo pipefail

REPO_ROOT="${WORLD_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${WORLD_PYTHON_BIN:-python}"
MANIFEST="${WORLD_EVAL_MANIFEST:?Set WORLD_EVAL_MANIFEST to val.jsonl or test.jsonl}"
METHOD="${WORLD_METHOD:-learned}"
DATA_ROLE="${WORLD_DATA_ROLE:-test}"
OUTPUT="${WORLD_METRICS_OUTPUT:?Set WORLD_METRICS_OUTPUT to a JSON path}"

cd "$REPO_ROOT"
test -r "$MANIFEST"
if [[ "$METHOD" == "learned" ]]; then
    CHECKPOINT="${WORLD_CHECKPOINT:?Set WORLD_CHECKPOINT for learned evaluation}"
    test -r "$CHECKPOINT"
    if [[ "$DATA_ROLE" == "test" ]]; then
        PROTOCOL="${WORLD_PROTOCOL:?Set WORLD_PROTOCOL for learned test evaluation}"
        SELECTION_RECORD="${WORLD_SELECTION_RECORD:?Set WORLD_SELECTION_RECORD for learned test evaluation}"
        EVALUATION_AUDIT="${WORLD_EVALUATION_AUDIT:?Set WORLD_EVALUATION_AUDIT for learned test evaluation}"
        test -r "$PROTOCOL"
        test -r "$SELECTION_RECORD"
        test -r "$EVALUATION_AUDIT"
    fi
fi
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

COMMAND=(
    "$PYTHON_BIN" -m src.world_model.evaluate
    --manifest "$MANIFEST"
    --method "$METHOD"
    --data-role "$DATA_ROLE"
    --output "$OUTPUT"
    --batch-size "${WORLD_BATCH_SIZE:-4}"
    --workers "${WORLD_WORKERS:-4}"
    --device cuda
)
if [[ "$METHOD" == "learned" ]]; then
    COMMAND+=(--checkpoint "$CHECKPOINT" --amp)
    if [[ "$DATA_ROLE" == "test" ]]; then
        COMMAND+=(
            --protocol "$PROTOCOL"
            --selection-record "$SELECTION_RECORD"
            --evaluation-audit "$EVALUATION_AUDIT"
        )
    fi
fi
if [[ -n "${WORLD_PREDICTION_DIR:-}" ]]; then
    COMMAND+=(--prediction-dir "$WORLD_PREDICTION_DIR")
fi
srun --nodes=1 --ntasks=1 "${COMMAND[@]}"
