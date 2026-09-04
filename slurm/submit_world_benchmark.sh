#!/bin/bash
# Submit the complete, dependency-locked world-model benchmark.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${BENCHMARK_REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
PYTHON_BIN=${BENCHMARK_PYTHON:-python3}
CANDIDATE_JOB_ID=${BENCHMARK_CANDIDATE_JOB_ID:-}
EXISTING_ORACLE_JOB_ID=${BENCHMARK_ORACLE_JOB_ID:-}
ORACLE_SUBMISSION_RECORD=${BENCHMARK_ORACLE_SUBMISSION_RECORD:-}
ORACLE_LINEAGE_RECORD=${BENCHMARK_ORACLE_LINEAGE_RECORD:-}
CLIP_PARQUET=${BENCHMARK_CLIP_PARQUET:?Set BENCHMARK_CLIP_PARQUET}
CANDIDATE_DIR=${BENCHMARK_CANDIDATE_DIR:?Set BENCHMARK_CANDIDATE_DIR}
ORACLE_DIR=${BENCHMARK_ORACLE_DIR:?Set BENCHMARK_ORACLE_DIR}
RUN_DIR=${BENCHMARK_RUN_DIR:?Set BENCHMARK_RUN_DIR}
EXPECTED_SHARDS=${BENCHMARK_EXPECTED_SHARDS:-8}
EXPECTED_ROWS=${BENCHMARK_EXPECTED_ROWS:-2000}
EXPECTED_CANDIDATES=${BENCHMARK_EXPECTED_CANDIDATES:-6}
EPOCHS=${BENCHMARK_EPOCHS:-30}
ORACLE_CONCURRENCY=${BENCHMARK_ORACLE_CONCURRENCY:-2}
ORACLE_CPUS=${BENCHMARK_ORACLE_CPUS:-4}
ORACLE_MEMORY=${BENCHMARK_ORACLE_MEMORY:-24G}

if [[ -n "${EXISTING_ORACLE_JOB_ID}" ]]; then
    if [[ ! "${EXISTING_ORACLE_JOB_ID}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: BENCHMARK_ORACLE_JOB_ID must be one numeric SLURM job ID" >&2
        exit 2
    fi
    if [[ -z "${ORACLE_SUBMISSION_RECORD}" ]]; then
        echo "ERROR: set BENCHMARK_ORACLE_SUBMISSION_RECORD when reusing an oracle job" >&2
        exit 2
    fi
    if [[ -z "${ORACLE_LINEAGE_RECORD}" ]]; then
        echo "ERROR: set BENCHMARK_ORACLE_LINEAGE_RECORD when reusing an oracle job" >&2
        exit 2
    fi
elif [[ ! "${CANDIDATE_JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: set a numeric BENCHMARK_CANDIDATE_JOB_ID or BENCHMARK_ORACLE_JOB_ID" >&2
    exit 2
fi
if [[ ! "${ORACLE_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: BENCHMARK_ORACLE_CONCURRENCY must be a positive integer" >&2
    exit 2
fi
if [[ ! "${ORACLE_CPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: BENCHMARK_ORACLE_CPUS must be a positive integer" >&2
    exit 2
fi
for value_name in EXPECTED_SHARDS EXPECTED_ROWS EXPECTED_CANDIDATES; do
    if [[ ! "${!value_name}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: BENCHMARK_${value_name} must be a positive integer" >&2
        exit 2
    fi
done
if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch is unavailable" >&2
    exit 2
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: BENCHMARK_PYTHON does not resolve to an executable: ${PYTHON_BIN}" >&2
    exit 2
fi
if [[ ! -r "${CLIP_PARQUET}" ]]; then
    echo "ERROR: clip parquet does not exist: ${CLIP_PARQUET}" >&2
    exit 2
fi
if [[ -e "${RUN_DIR}/submission.json" ]]; then
    echo "ERROR: this run directory already has a submission record: ${RUN_DIR}" >&2
    exit 2
fi

mkdir -p "${CANDIDATE_DIR}" "${ORACLE_DIR}" "${RUN_DIR}/logs"
REPO_ROOT=$(cd -- "${REPO_ROOT}" && pwd)
CANDIDATE_DIR=$(cd -- "${CANDIDATE_DIR}" && pwd)
ORACLE_DIR=$(cd -- "${ORACLE_DIR}" && pwd)
RUN_DIR=$(cd -- "${RUN_DIR}" && pwd)
CLIP_PARQUET=$(realpath "${CLIP_PARQUET}")
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
    echo "ERROR: benchmark repository must be clean before submission" >&2
    exit 2
fi
REPOSITORY_COMMIT=$(git -C "${REPO_ROOT}" rev-parse --verify HEAD)
STAGE_SCRIPT=${REPO_ROOT}/slurm/run_benchmark_stage.sh
ORACLE_PROVENANCE_JSON=

if [[ -n "${EXISTING_ORACLE_JOB_ID}" ]]; then
    ORACLE_PROVENANCE_JSON=$("${PYTHON_BIN}" - \
        "${ORACLE_SUBMISSION_RECORD}" \
        "${EXISTING_ORACLE_JOB_ID}" \
        "${CLIP_PARQUET}" \
        "${CANDIDATE_DIR}" \
        "${ORACLE_DIR}" \
        "${EXPECTED_SHARDS}" \
        "${EXPECTED_ROWS}" \
        "${EXPECTED_CANDIDATES}" <<'PY'
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


try:
    record_path = Path(sys.argv[1]).expanduser().resolve(strict=True)
except (OSError, RuntimeError) as error:
    raise SystemExit(f"ERROR: oracle submission record cannot be resolved: {error}") from error
if not record_path.is_file():
    raise SystemExit(f"ERROR: oracle submission record is not a file: {record_path}")
try:
    record_bytes = record_path.read_bytes()
except OSError as error:
    raise SystemExit(f"ERROR: oracle submission record cannot be read: {error}") from error
try:
    record = json.loads(record_bytes)
except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise SystemExit(f"ERROR: invalid oracle submission record: {error}") from error
if not isinstance(record, dict) or record.get("schema_version") != 2:
    raise SystemExit("ERROR: oracle submission record must use schema_version 2")

jobs = record.get("jobs")
paths = record.get("paths")
source = record.get("source")
configuration = record.get("configuration")
if not all(isinstance(value, dict) for value in (jobs, paths, source, configuration)):
    raise SystemExit("ERROR: oracle submission record is missing required objects")
if configuration.get("reused_oracle_job") is not False:
    raise SystemExit("ERROR: oracle submission record must describe the original oracle job")

expected_configuration = {
    "expected_oracle_shards": int(sys.argv[6]),
    "expected_rows": int(sys.argv[7]),
    "expected_candidates": int(sys.argv[8]),
}
for name, expected_value in expected_configuration.items():
    recorded_value = configuration.get(name)
    if isinstance(recorded_value, bool) or recorded_value != expected_value:
        raise SystemExit(
            f"ERROR: oracle submission record {name} {recorded_value!r} "
            f"does not match {expected_value}"
        )

oracle_configuration = {}
for name in ("oracle_concurrency", "oracle_cpus_per_task"):
    value = configuration.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SystemExit(f"ERROR: oracle submission record has invalid {name}")
    oracle_configuration[name] = value
oracle_memory = configuration.get("oracle_memory_per_task")
if not isinstance(oracle_memory, str) or re.fullmatch(r"[1-9][0-9]*[KMGTP]?", oracle_memory) is None:
    raise SystemExit("ERROR: oracle submission record has invalid oracle_memory_per_task")
oracle_configuration["oracle_memory_per_task"] = oracle_memory

expected_job = int(sys.argv[2])
recorded_job = jobs.get("oracle_job")
if isinstance(recorded_job, bool) or recorded_job != expected_job:
    raise SystemExit(
        f"ERROR: oracle submission record job {recorded_job!r} does not match {expected_job}"
    )


def canonical_record_path(name: str, *, directory: bool) -> Path:
    value = paths.get(name)
    if not isinstance(value, str) or not value:
        raise SystemExit(f"ERROR: oracle submission record has invalid path {name!r}")
    try:
        resolved = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise SystemExit(
            f"ERROR: oracle submission record path {name!r} cannot be resolved: {error}"
        ) from error
    valid_type = resolved.is_dir() if directory else resolved.is_file()
    if not valid_type:
        expected_type = "directory" if directory else "file"
        raise SystemExit(
            f"ERROR: oracle submission record path {name!r} is not a {expected_type}: {resolved}"
        )
    return resolved


expected_paths = {
    "clip_parquet": Path(sys.argv[3]).resolve(strict=True),
    "candidate_directory": Path(sys.argv[4]).resolve(strict=True),
    "oracle_directory": Path(sys.argv[5]).resolve(strict=True),
}
for name, expected_path in expected_paths.items():
    recorded_path = canonical_record_path(name, directory=name.endswith("directory"))
    if recorded_path != expected_path:
        raise SystemExit(
            f"ERROR: oracle submission record {name} {recorded_path} "
            f"does not match {expected_path}"
        )

oracle_repository = canonical_record_path("repository", directory=True)
oracle_commit = source.get("repository_commit")
if not isinstance(oracle_commit, str) or re.fullmatch(r"[0-9a-f]{40}", oracle_commit) is None:
    raise SystemExit("ERROR: oracle submission record has no full lowercase source commit")


def git_output(*arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(oracle_repository), *arguments],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "git command failed"
        raise SystemExit(f"ERROR: cannot validate oracle source repository: {detail}")
    return result.stdout


oracle_head = git_output("rev-parse", "--verify", "HEAD^{commit}").strip()
if oracle_head != oracle_commit:
    raise SystemExit(
        f"ERROR: oracle source HEAD {oracle_head} does not match recorded commit {oracle_commit}"
    )
if git_output("status", "--porcelain", "--untracked-files=all"):
    raise SystemExit(f"ERROR: oracle source worktree is dirty: {oracle_repository}")

print(
    json.dumps(
        {
            "oracle_repository_commit": oracle_commit,
            "oracle_configuration": oracle_configuration,
            "oracle_submission_record": str(record_path),
            "oracle_submission_record_sha256": hashlib.sha256(record_bytes).hexdigest(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
)
PY
    )
    ORACLE_LINEAGE_JSON=$("${PYTHON_BIN}" "${REPO_ROOT}/tools/oracle_lineage.py" validate \
        --record "${ORACLE_LINEAGE_RECORD}" \
        --oracle-submission-record "${ORACLE_SUBMISSION_RECORD}" \
        --oracle-job "${EXISTING_ORACLE_JOB_ID}" \
        --clip-parquet "${CLIP_PARQUET}" \
        --candidate-directory "${CANDIDATE_DIR}" \
        --oracle-directory "${ORACLE_DIR}")
else
    ORACLE_LINEAGE_JSON=
fi

normalize_job_id() {
    local value=$1
    value=${value%%;*}
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: sbatch returned an invalid job ID: $1" >&2
        exit 2
    fi
    printf '%s\n' "${value}"
}

if [[ -n "${EXISTING_ORACLE_JOB_ID}" ]]; then
    ORACLE_JOB=${EXISTING_ORACLE_JOB_ID}
else
    ORACLE_JOB=$(normalize_job_id "$(
        sbatch --parsable \
            --dependency="afterok:${CANDIDATE_JOB_ID}" \
            --array="0-$((EXPECTED_SHARDS - 1))%${ORACLE_CONCURRENCY}" \
            --time=20:00:00 \
            --mem="${ORACLE_MEMORY}" \
            --cpus-per-task="${ORACLE_CPUS}" \
            --job-name=world-oracle-2k \
            --output="${RUN_DIR}/logs/oracle-%A_%a.out" \
            --error="${RUN_DIR}/logs/oracle-%A_%a.err" \
            --export="ALL,BENCHMARK_REPO_ROOT=${REPO_ROOT},BENCHMARK_PYTHON=${PYTHON_BIN},BENCHMARK_RUN_DIR=${RUN_DIR},BENCHMARK_CANDIDATE_DIR=${CANDIDATE_DIR},BENCHMARK_ORACLE_DIR=${ORACLE_DIR},BENCHMARK_CLIP_PARQUET=${CLIP_PARQUET},BENCHMARK_REPOSITORY_COMMIT=${REPOSITORY_COMMIT},BENCHMARK_ACTION=oracle" \
            "${STAGE_SCRIPT}"
    )")
fi

COMMON_EXPORT="ALL,BENCHMARK_REPO_ROOT=${REPO_ROOT},BENCHMARK_PYTHON=${PYTHON_BIN},BENCHMARK_RUN_DIR=${RUN_DIR},BENCHMARK_CANDIDATE_DIR=${CANDIDATE_DIR},BENCHMARK_ORACLE_DIR=${ORACLE_DIR},BENCHMARK_CLIP_PARQUET=${CLIP_PARQUET},BENCHMARK_REPOSITORY_COMMIT=${REPOSITORY_COMMIT},BENCHMARK_EXPECTED_SHARDS=${EXPECTED_SHARDS},BENCHMARK_EXPECTED_ROWS=${EXPECTED_ROWS},BENCHMARK_EXPECTED_CANDIDATES=${EXPECTED_CANDIDATES},BENCHMARK_EPOCHS=${EPOCHS}"

PREP_JOB=$(normalize_job_id "$(
    sbatch --parsable \
        --dependency="afterok:${ORACLE_JOB}" \
        --time=04:00:00 \
        --mem=24G \
        --cpus-per-task=4 \
        --job-name=world-prepare-2k \
        --output="${RUN_DIR}/logs/prepare-%j.out" \
        --error="${RUN_DIR}/logs/prepare-%j.err" \
        --export="${COMMON_EXPORT},BENCHMARK_ACTION=prepare" \
        "${STAGE_SCRIPT}"
)")

TRAIN_2026_JOB=$(normalize_job_id "$(
    sbatch --parsable \
        --dependency="afterok:${PREP_JOB}" \
        --gres=gpu:1 \
        --time=20:00:00 \
        --mem=28G \
        --cpus-per-task=4 \
        --job-name=world-train-2026 \
        --output="${RUN_DIR}/logs/train-2026-%j.out" \
        --error="${RUN_DIR}/logs/train-2026-%j.err" \
        --export="${COMMON_EXPORT},BENCHMARK_ACTION=train,BENCHMARK_TRAIN_SEED=2026,BENCHMARK_TRAIN_OUTPUT=${RUN_DIR}/training/seed-2026" \
        "${STAGE_SCRIPT}"
)")

TRAIN_2027_JOB=$(normalize_job_id "$(
    sbatch --parsable \
        --dependency="afterok:${PREP_JOB}" \
        --gres=gpu:1 \
        --time=20:00:00 \
        --mem=28G \
        --cpus-per-task=4 \
        --job-name=world-train-2027 \
        --output="${RUN_DIR}/logs/train-2027-%j.out" \
        --error="${RUN_DIR}/logs/train-2027-%j.err" \
        --export="${COMMON_EXPORT},BENCHMARK_ACTION=train,BENCHMARK_TRAIN_SEED=2027,BENCHMARK_TRAIN_OUTPUT=${RUN_DIR}/training/seed-2027" \
        "${STAGE_SCRIPT}"
)")

SELECT_JOB=$(normalize_job_id "$(
    sbatch --parsable \
        --dependency="afterok:${TRAIN_2026_JOB}:${TRAIN_2027_JOB}" \
        --time=01:00:00 \
        --mem=12G \
        --cpus-per-task=2 \
        --job-name=world-select \
        --output="${RUN_DIR}/logs/select-%j.out" \
        --error="${RUN_DIR}/logs/select-%j.err" \
        --export="${COMMON_EXPORT},BENCHMARK_ACTION=select" \
        "${STAGE_SCRIPT}"
)")

PERSISTENCE_JOB=$(normalize_job_id "$(
    sbatch --parsable \
        --dependency="afterok:${SELECT_JOB}" \
        --time=08:00:00 \
        --mem=16G \
        --cpus-per-task=4 \
        --job-name=world-persistence \
        --output="${RUN_DIR}/logs/persistence-%j.out" \
        --error="${RUN_DIR}/logs/persistence-%j.err" \
        --export="${COMMON_EXPORT},BENCHMARK_ACTION=evaluate-persistence" \
        "${STAGE_SCRIPT}"
)")

LEARNED_JOB=$(normalize_job_id "$(
    sbatch --parsable \
        --dependency="afterok:${SELECT_JOB}" \
        --gres=gpu:1 \
        --time=08:00:00 \
        --mem=28G \
        --cpus-per-task=4 \
        --job-name=world-learned-eval \
        --output="${RUN_DIR}/logs/learned-%j.out" \
        --error="${RUN_DIR}/logs/learned-%j.err" \
        --export="${COMMON_EXPORT},BENCHMARK_ACTION=evaluate-learned" \
        "${STAGE_SCRIPT}"
)")

LEARNED_RERANK_JOB=$(normalize_job_id "$(
    sbatch --parsable \
        --dependency="afterok:${LEARNED_JOB}" \
        --time=04:00:00 \
        --mem=16G \
        --cpus-per-task=4 \
        --job-name=world-learned-rerank \
        --output="${RUN_DIR}/logs/learned-rerank-%j.out" \
        --error="${RUN_DIR}/logs/learned-rerank-%j.err" \
        --export="${COMMON_EXPORT},BENCHMARK_ACTION=rerank-learned" \
        "${STAGE_SCRIPT}"
)")

PERSISTENCE_RERANK_JOB=$(normalize_job_id "$(
    sbatch --parsable \
        --dependency="afterok:${PERSISTENCE_JOB}" \
        --time=04:00:00 \
        --mem=16G \
        --cpus-per-task=4 \
        --job-name=world-persistence-rerank \
        --output="${RUN_DIR}/logs/persistence-rerank-%j.out" \
        --error="${RUN_DIR}/logs/persistence-rerank-%j.err" \
        --export="${COMMON_EXPORT},BENCHMARK_ACTION=rerank-persistence" \
        "${STAGE_SCRIPT}"
)")

FORECAST_COMPARE_JOB=$(normalize_job_id "$(
    sbatch --parsable \
        --dependency="afterok:${LEARNED_JOB}:${PERSISTENCE_JOB}" \
        --time=04:00:00 \
        --mem=16G \
        --cpus-per-task=4 \
        --job-name=world-forecast-compare \
        --output="${RUN_DIR}/logs/forecast-compare-%j.out" \
        --error="${RUN_DIR}/logs/forecast-compare-%j.err" \
        --export="${COMMON_EXPORT},BENCHMARK_ACTION=compare-forecasts" \
        "${STAGE_SCRIPT}"
)")

AGGREGATE_JOB=$(normalize_job_id "$(
    sbatch --parsable \
        --dependency="afterok:${LEARNED_RERANK_JOB}:${PERSISTENCE_RERANK_JOB}:${FORECAST_COMPARE_JOB}" \
        --time=04:00:00 \
        --mem=16G \
        --cpus-per-task=4 \
        --job-name=world-final-eval \
        --output="${RUN_DIR}/logs/final-eval-%j.out" \
        --error="${RUN_DIR}/logs/final-eval-%j.err" \
        --export="${COMMON_EXPORT},BENCHMARK_ACTION=aggregate" \
        "${STAGE_SCRIPT}"
)")

export CANDIDATE_JOB_ID EXISTING_ORACLE_JOB_ID ORACLE_JOB PREP_JOB TRAIN_2026_JOB TRAIN_2027_JOB
export PERSISTENCE_JOB SELECT_JOB LEARNED_JOB LEARNED_RERANK_JOB
export PERSISTENCE_RERANK_JOB FORECAST_COMPARE_JOB AGGREGATE_JOB RUN_DIR REPO_ROOT CLIP_PARQUET
export CANDIDATE_DIR ORACLE_DIR EPOCHS EXPECTED_ROWS EXPECTED_CANDIDATES EXPECTED_SHARDS
export ORACLE_CONCURRENCY ORACLE_CPUS ORACLE_MEMORY REPOSITORY_COMMIT ORACLE_PROVENANCE_JSON
export ORACLE_LINEAGE_JSON
"${PYTHON_BIN}" - <<'PY'
import json
import os
import tempfile
from pathlib import Path

keys = (
    "ORACLE_JOB",
    "PREP_JOB",
    "TRAIN_2026_JOB",
    "TRAIN_2027_JOB",
    "PERSISTENCE_JOB",
    "SELECT_JOB",
    "LEARNED_JOB",
    "LEARNED_RERANK_JOB",
    "PERSISTENCE_RERANK_JOB",
    "FORECAST_COMPARE_JOB",
    "AGGREGATE_JOB",
)
oracle_provenance = (
    json.loads(os.environ["ORACLE_PROVENANCE_JSON"])
    if os.environ["ORACLE_PROVENANCE_JSON"]
    else {}
)
source = {
    "repository_commit": os.environ["REPOSITORY_COMMIT"],
    "oracle_repository_commit": oracle_provenance.get(
        "oracle_repository_commit", os.environ["REPOSITORY_COMMIT"]
    ),
}
lineage = (
    json.loads(os.environ["ORACLE_LINEAGE_JSON"])
    if os.environ["ORACLE_LINEAGE_JSON"]
    else {}
)
source.update(lineage)
for name in ("oracle_submission_record", "oracle_submission_record_sha256"):
    if name in oracle_provenance:
        source[name] = oracle_provenance[name]

configuration = {
    "epochs": int(os.environ["EPOCHS"]),
    "expected_rows": int(os.environ["EXPECTED_ROWS"]),
    "expected_candidates": int(os.environ["EXPECTED_CANDIDATES"]),
    "expected_oracle_shards": int(os.environ["EXPECTED_SHARDS"]),
    "oracle_concurrency": int(os.environ["ORACLE_CONCURRENCY"]),
    "oracle_cpus_per_task": int(os.environ["ORACLE_CPUS"]),
    "oracle_memory_per_task": os.environ["ORACLE_MEMORY"],
    "training_seeds": [2026, 2027],
    "maximum_concurrent_gpu_jobs": 2,
    "gpus_per_training_job": 1,
    "model_sharding": False,
    "reused_oracle_job": bool(os.environ["EXISTING_ORACLE_JOB_ID"]),
}
if oracle_provenance:
    configuration.update(oracle_provenance["oracle_configuration"])

payload = {
    "schema_version": 2,
    "jobs": {key.lower(): int(os.environ[key]) for key in keys},
    "paths": {
        "repository": os.environ["REPO_ROOT"],
        "clip_parquet": os.environ["CLIP_PARQUET"],
        "candidate_directory": os.environ["CANDIDATE_DIR"],
        "oracle_directory": os.environ["ORACLE_DIR"],
        "run_directory": os.environ["RUN_DIR"],
    },
    "configuration": configuration,
    "source": source,
}
payload["jobs"]["candidate_job_id"] = (
    int(os.environ["CANDIDATE_JOB_ID"])
    if os.environ["CANDIDATE_JOB_ID"]
    else None
)
output = Path(os.environ["RUN_DIR"]) / "submission.json"
with tempfile.NamedTemporaryFile(
    mode="w", encoding="utf-8", dir=output.parent, prefix=".submission.", delete=False
) as handle:
    temporary = Path(handle.name)
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, output)
print(output.read_text(encoding="utf-8"), end="")
PY
