#!/bin/bash
#SBATCH --job-name=world-benchmark-stage
#SBATCH --partition=dualcard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${BENCHMARK_REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
PYTHON_BIN=${BENCHMARK_PYTHON:-python3}
ACTION=${BENCHMARK_ACTION:?Set BENCHMARK_ACTION}
RUN_DIR=${BENCHMARK_RUN_DIR:?Set BENCHMARK_RUN_DIR}
CANDIDATE_DIR=${BENCHMARK_CANDIDATE_DIR:?Set BENCHMARK_CANDIDATE_DIR}
ORACLE_DIR=${BENCHMARK_ORACLE_DIR:?Set BENCHMARK_ORACLE_DIR}
CLIP_PARQUET=${BENCHMARK_CLIP_PARQUET:?Set BENCHMARK_CLIP_PARQUET}
EXPECTED_SHARDS=${BENCHMARK_EXPECTED_SHARDS:-8}
EXPECTED_ROWS=${BENCHMARK_EXPECTED_ROWS:-2000}
EXPECTED_CANDIDATES=${BENCHMARK_EXPECTED_CANDIDATES:-6}
SPLIT_SEED=${BENCHMARK_SPLIT_SEED:-2026}
SPLIT_RATIOS=${BENCHMARK_SPLIT_RATIOS:-0.8,0.1,0.1}
EPOCHS=${BENCHMARK_EPOCHS:-30}
BATCH_SIZE=${BENCHMARK_BATCH_SIZE:-4}
WORKERS=${BENCHMARK_WORKERS:-4}
BASE_CHANNELS=${BENCHMARK_BASE_CHANNELS:-16}
EXPECTED_REPOSITORY_COMMIT=${BENCHMARK_REPOSITORY_COMMIT:?Set BENCHMARK_REPOSITORY_COMMIT}

MERGED_MANIFEST=${RUN_DIR}/manifests/all.jsonl
MERGE_SUMMARY=${RUN_DIR}/manifests/merge_summary.json
SPLIT_DIR=${RUN_DIR}/splits
TRAIN_MANIFEST=${SPLIT_DIR}/train.jsonl
VAL_MANIFEST=${SPLIT_DIR}/val.jsonl
TEST_MANIFEST=${SPLIT_DIR}/test.jsonl
SELECTION=${RUN_DIR}/checkpoint_selection.json
EVALUATION_AUDIT=${RUN_DIR}/evaluation_partition.json

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: BENCHMARK_PYTHON does not resolve to an executable: ${PYTHON_BIN}" >&2
    exit 2
fi
if [[ ! -d "${REPO_ROOT}" ]]; then
    echo "ERROR: repository does not exist: ${REPO_ROOT}" >&2
    exit 2
fi
if [[ ! "${EXPECTED_REPOSITORY_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: BENCHMARK_REPOSITORY_COMMIT must be a full lowercase Git commit" >&2
    exit 2
fi
if ! command -v git >/dev/null 2>&1; then
    echo "ERROR: git is unavailable; cannot verify benchmark source" >&2
    exit 2
fi
ACTUAL_REPOSITORY_COMMIT=$(git -C "${REPO_ROOT}" rev-parse --verify HEAD^{commit})
if [[ "${ACTUAL_REPOSITORY_COMMIT}" != "${EXPECTED_REPOSITORY_COMMIT}" ]]; then
    echo "ERROR: repository HEAD ${ACTUAL_REPOSITORY_COMMIT} does not match expected commit ${EXPECTED_REPOSITORY_COMMIT}" >&2
    exit 2
fi
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=all)" ]]; then
    echo "ERROR: repository worktree is dirty: ${REPO_ROOT}" >&2
    exit 2
fi
mkdir -p "${RUN_DIR}"
cd "${REPO_ROOT}"

require_file() {
    if [[ ! -r "$1" ]]; then
        echo "ERROR: required file is not readable: $1" >&2
        exit 2
    fi
}

require_new_output() {
    if [[ -e "$1" ]]; then
        echo "ERROR: refusing to replace an existing benchmark output: $1" >&2
        exit 2
    fi
    mkdir -p "$(dirname -- "$1")"
}

prepare_prediction_directory() {
    if [[ -e "$1" && ! -d "$1" ]]; then
        echo "ERROR: prediction path exists and is not a directory: $1" >&2
        exit 2
    fi
    # The evaluator recomputes and compares every existing probability array.
    # This permits interrupted jobs to resume without accepting stale values.
    mkdir -p "$1"
}

selected_checkpoint() {
    "${PYTHON_BIN}" - "${SELECTION}" <<'PY'
import json
import sys
from pathlib import Path

selection = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(selection["selected_checkpoint"])
PY
}

verify_cuda() {
    srun --nodes=1 --ntasks=1 "${PYTHON_BIN}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory)
PY
}

case "${ACTION}" in
    oracle)
        export PROJECT_DIR="${REPO_ROOT}"
        export PYTHON_BIN
        export CLIP_PARQUET
        export CANDIDATE_DIR
        export OUTPUT_DIR="${ORACLE_DIR}"
        exec "${REPO_ROOT}/slurm/build_lidar_world_oracle.sh"
        ;;
    prepare)
        require_file "${CANDIDATE_DIR}/records"
        require_file "${ORACLE_DIR}/manifests/shard-00000-of-$(printf '%05d' "${EXPECTED_SHARDS}").jsonl"
        srun --nodes=1 --ntasks=1 "${PYTHON_BIN}" -m src.world_model.benchmark_prep merge \
            --manifest-dir "${ORACLE_DIR}/manifests" \
            --candidate-dir "${CANDIDATE_DIR}" \
            --output "${MERGED_MANIFEST}" \
            --summary "${MERGE_SUMMARY}" \
            --expected-shards "${EXPECTED_SHARDS}" \
            --expected-rows "${EXPECTED_ROWS}" \
            --expected-candidates "${EXPECTED_CANDIDATES}"
        srun --nodes=1 --ntasks=1 "${PYTHON_BIN}" -m src.world_model.benchmark_prep split \
            --manifest "${MERGED_MANIFEST}" \
            --output-dir "${SPLIT_DIR}" \
            --seed "${SPLIT_SEED}" \
            --ratios "${SPLIT_RATIOS}"
        srun --nodes=1 --ntasks=1 "${PYTHON_BIN}" -m src.world_model.benchmark_prep freeze-protocol \
            --split-summary "${SPLIT_DIR}/split_summary.json" \
            --source-parquet "${CLIP_PARQUET}" \
            --epochs "${EPOCHS}" \
            --batch-size "${BATCH_SIZE}" \
            --workers "${WORKERS}" \
            --base-channels "${BASE_CHANNELS}" \
            --output "${RUN_DIR}/protocol.json"
        ;;
    train)
        require_file "${TRAIN_MANIFEST}"
        require_file "${VAL_MANIFEST}"
        TRAIN_SEED=${BENCHMARK_TRAIN_SEED:?Set BENCHMARK_TRAIN_SEED for train}
        TRAIN_OUTPUT=${BENCHMARK_TRAIN_OUTPUT:?Set BENCHMARK_TRAIN_OUTPUT for train}
        mkdir -p "${TRAIN_OUTPUT}"
        verify_cuda
        COMMAND=(
            "${PYTHON_BIN}" -m src.world_model.train
            --train-manifest "${TRAIN_MANIFEST}"
            --val-manifest "${VAL_MANIFEST}"
            --output-dir "${TRAIN_OUTPUT}"
            --epochs "${EPOCHS}"
            --batch-size "${BATCH_SIZE}"
            --workers "${WORKERS}"
            --base-channels "${BASE_CHANNELS}"
            --seed "${TRAIN_SEED}"
            --device cuda
            --amp
        )
        if [[ -f "${TRAIN_OUTPUT}/latest.pt" ]]; then
            COMMAND+=(--resume auto)
        fi
        srun --nodes=1 --ntasks=1 "${COMMAND[@]}"
        require_file "${TRAIN_OUTPUT}/best.pt"
        ;;
    select)
        require_file "${RUN_DIR}/training/seed-2026/best.pt"
        require_file "${RUN_DIR}/training/seed-2027/best.pt"
        require_file "${TEST_MANIFEST}"
        srun --nodes=1 --ntasks=1 "${PYTHON_BIN}" -m src.world_model.benchmark_prep select-checkpoint \
            --run "seed-2026=${RUN_DIR}/training/seed-2026" \
            --run "seed-2027=${RUN_DIR}/training/seed-2027" \
            --output "${SELECTION}"
        srun --nodes=1 --ntasks=1 "${PYTHON_BIN}" -m src.world_model.benchmark_prep verify-evaluation \
            --selection "${SELECTION}" \
            --test-manifest "${TEST_MANIFEST}" \
            --output "${EVALUATION_AUDIT}"
        ;;
    evaluate-learned)
        require_file "${SELECTION}"
        require_file "${EVALUATION_AUDIT}"
        require_file "${TEST_MANIFEST}"
        CHECKPOINT=$(selected_checkpoint)
        require_file "${CHECKPOINT}"
        METRICS=${RUN_DIR}/evaluation/learned_metrics.json
        PREDICTIONS=${RUN_DIR}/predictions/learned
        require_new_output "${METRICS}"
        prepare_prediction_directory "${PREDICTIONS}"
        verify_cuda
        srun --nodes=1 --ntasks=1 "${PYTHON_BIN}" -m src.world_model.evaluate \
            --manifest "${TEST_MANIFEST}" \
            --method learned \
            --data-role test \
            --checkpoint "${CHECKPOINT}" \
            --output "${METRICS}" \
            --prediction-dir "${PREDICTIONS}" \
            --batch-size "${BATCH_SIZE}" \
            --workers "${WORKERS}" \
            --device cuda \
            --amp
        ;;
    evaluate-persistence)
        require_file "${TEST_MANIFEST}"
        METRICS=${RUN_DIR}/evaluation/persistence_metrics.json
        PREDICTIONS=${RUN_DIR}/predictions/persistence
        require_new_output "${METRICS}"
        prepare_prediction_directory "${PREDICTIONS}"
        srun --nodes=1 --ntasks=1 "${PYTHON_BIN}" -m src.world_model.evaluate \
            --manifest "${TEST_MANIFEST}" \
            --method persistence \
            --data-role test \
            --output "${METRICS}" \
            --prediction-dir "${PREDICTIONS}" \
            --batch-size "${BATCH_SIZE}" \
            --workers "${WORKERS}" \
            --device cpu
        ;;
    rerank-learned|rerank-persistence)
        require_file "${TEST_MANIFEST}"
        METHOD=${ACTION#rerank-}
        PREDICTIONS=${RUN_DIR}/predictions/${METHOD}
        OUTPUT=${RUN_DIR}/selections/${METHOD}.jsonl
        if [[ ! -d "${PREDICTIONS}" ]]; then
            echo "ERROR: prediction directory does not exist: ${PREDICTIONS}" >&2
            exit 2
        fi
        require_new_output "${OUTPUT}"
        srun --nodes=1 --ntasks=1 "${PYTHON_BIN}" -m src.world_model.rerank \
            --manifest "${TEST_MANIFEST}" \
            --prediction-dir "${PREDICTIONS}" \
            --output "${OUTPUT}" \
            --collision-weight 10.0 \
            --uncertainty-weight 1.0 \
            --out-of-bounds-weight 5.0 \
            --acceleration-weight 0.05 \
            --jerk-weight 0.01 \
            --curvature-weight 0.1 \
            --progress-weight 0.02
        ;;
    compare-forecasts)
        require_file "${RUN_DIR}/protocol.json"
        require_file "${TRAIN_MANIFEST}"
        require_file "${VAL_MANIFEST}"
        require_file "${TEST_MANIFEST}"
        OUTPUT=${RUN_DIR}/evaluation/forecast_comparison.json
        require_new_output "${OUTPUT}"
        srun --nodes=1 --ntasks=1 "${PYTHON_BIN}" -m src.world_model.compare_forecasts \
            --manifest "${TEST_MANIFEST}" \
            --learned-prediction-dir "${RUN_DIR}/predictions/learned" \
            --persistence-prediction-dir "${RUN_DIR}/predictions/persistence" \
            --output "${OUTPUT}" \
            --threshold 0.5 \
            --ap-bins 1000 \
            --bootstrap-replicates 10000 \
            --bootstrap-seed 2026
        ;;
    aggregate)
        require_file "${TRAIN_MANIFEST}"
        require_file "${VAL_MANIFEST}"
        require_file "${TEST_MANIFEST}"
        require_file "${MERGED_MANIFEST}"
        require_file "${RUN_DIR}/protocol.json"
        require_file "${RUN_DIR}/evaluation/forecast_comparison.json"
        require_file "${RUN_DIR}/selections/learned.jsonl"
        require_file "${RUN_DIR}/selections/persistence.jsonl"
        OUTPUT=${RUN_DIR}/evaluation/selection_test.json
        PER_CLIP_OUTPUT=${RUN_DIR}/evaluation/selection_test_per_clip.jsonl
        require_new_output "${OUTPUT}"
        require_new_output "${PER_CLIP_OUTPUT}"
        srun --nodes=1 --ntasks=1 "${PYTHON_BIN}" -m src.world_model.evaluate_selection \
            --test-manifest "${TEST_MANIFEST}" \
            --train-manifest "${TRAIN_MANIFEST}" \
            --validation-manifest "${VAL_MANIFEST}" \
            --oracle-manifest "${MERGED_MANIFEST}" \
            --learned-selections "${RUN_DIR}/selections/learned.jsonl" \
            --persistence-selections "${RUN_DIR}/selections/persistence.jsonl" \
            --candidate-dir "${CANDIDATE_DIR}" \
            --output "${OUTPUT}" \
            --per-clip-output "${PER_CLIP_OUTPUT}" \
            --bootstrap-replicates 10000 \
            --bootstrap-seed 2026
        ;;
    *)
        echo "ERROR: unsupported BENCHMARK_ACTION: ${ACTION}" >&2
        exit 2
        ;;
esac
