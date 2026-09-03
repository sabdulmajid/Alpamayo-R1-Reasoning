#!/usr/bin/env bash
#SBATCH --job-name=alp-candidates
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=28G
#SBATCH --time=06:00:00

set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="${ALPAMAYO_PROJECT_ROOT:-$(cd -- "${script_dir}/.." && pwd)}"
python_bin="${ALPAMAYO_PYTHON:-python}"
clip_parquet="${CLIP_PARQUET:-${project_root}/data/eval_clips_2k.parquet}"
output_dir="${OUTPUT_DIR:-${project_root}/results/candidates_k6}"

if [[ ! -d "${project_root}" || ! -f "${project_root}/src/generate_candidates.py" ]]; then
    echo "ERROR: project is not visible on $(hostname): ${project_root}" >&2
    exit 2
fi
if [[ ! -r "${clip_parquet}" ]]; then
    echo "ERROR: clip parquet is not readable: ${clip_parquet}" >&2
    exit 2
fi
if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found: ${python_bin}" >&2
    exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi is unavailable" >&2
    exit 2
fi
shard_index="${SLURM_ARRAY_TASK_ID:-0}"
num_shards="${SLURM_ARRAY_TASK_COUNT:-1}"
if [[ "${num_shards}" -gt 1 ]]; then
    if [[ "${SLURM_ARRAY_TASK_MIN:-0}" -ne 0 || "${SLURM_ARRAY_TASK_STEP:-1}" -ne 1 ]]; then
        echo "ERROR: the array must use contiguous zero-based indices (for example --array=0-5)" >&2
        exit 2
    fi
fi

mkdir -p "${output_dir}"
cd "${project_root}"

"${python_bin}" - <<'PY'
import os
import sys

import alpamayo_r1
import flash_attn
import physical_ai_av
import torch
import transformers
from huggingface_hub import get_token

expected = {"torch": "2.8.0", "transformers": "4.57.1"}
actual = {
    "torch": torch.__version__.split("+")[0],
    "transformers": transformers.__version__,
}
if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"Python 3.12 is required; found {sys.version.split()[0]}")
for package, version in expected.items():
    if actual[package] != version:
        raise SystemExit(f"{package}=={version} is required; found {actual[package]}")
if flash_attn.__version__ != "2.8.3":
    raise SystemExit(f"flash-attn==2.8.3 is required; found {flash_attn.__version__}")
visible_gpus = torch.cuda.device_count()
if visible_gpus != 1:
    raise SystemExit(
        "Exactly one GPU must be visible; Alpamayo rollout cache tensors cannot be "
        "split safely across CUDA devices"
    )
allocated_gpus = os.environ.get("SLURM_GPUS_ON_NODE", "")
if allocated_gpus.isdigit() and int(allocated_gpus) != 1:
    raise SystemExit(
        f"Expected a one-GPU SLURM allocation; SLURM_GPUS_ON_NODE={allocated_gpus}"
    )
if os.environ.get("HF_HUB_OFFLINE") != "1" and get_token() is None:
    raise SystemExit(
        "Hugging Face credentials not found; run `hf auth login`, inject HF_TOKEN, "
        "or set HF_HUB_OFFLINE=1 when all assets are cached"
    )
print(
    f"preflight: Python {sys.version.split()[0]}, Torch {torch.__version__}, "
    f"visible GPUs={visible_gpus}"
)
for index in range(visible_gpus):
    properties = torch.cuda.get_device_properties(index)
    print(
        f"  cuda:{index}: {properties.name}, "
        f"{properties.total_memory / 1024**3:.1f} GiB"
    )
PY

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

command=(
    "${python_bin}" src/generate_candidates.py
    --clip-parquet "${clip_parquet}"
    --output-dir "${output_dir}"
    --shard-index "${shard_index}"
    --num-shards "${num_shards}"
    --num-candidates "${NUM_CANDIDATES:-6}"
    --max-generation-length "${MAX_GENERATION_LENGTH:-256}"
    --camera-num-frames "${CAMERA_NUM_FRAMES:-4}"
    --base-seed "${BASE_SEED:-42}"
    --dataset-revision "${DATASET_REVISION:-2ae73f49ffd2b5db43b404201beb7b92889f7afc}"
    --model-revision "${MODEL_REVISION:-69f9e9ba94445c81d8d802b048883fc473326137}"
    --gpu-memory "${GPU_MEMORY:-18GiB}"
    --cpu-memory "${CPU_MEMORY:-8GiB}"
)
if [[ -n "${MAX_CLIPS:-}" ]]; then
    command+=(--max-clips "${MAX_CLIPS}")
fi
if [[ "${FAIL_FAST:-0}" == "1" ]]; then
    command+=(--fail-fast)
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
    command+=(--overwrite)
fi

echo "Starting candidate shard ${shard_index}/${num_shards} on $(hostname)"
srun "${command[@]}"
