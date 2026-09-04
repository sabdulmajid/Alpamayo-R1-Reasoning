#!/usr/bin/env python3
"""Create and validate immutable provenance for resumed oracle artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


SCHEMA_VERSION = 1
FULL_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
FILE_SHA_PATTERN = re.compile(r"[0-9a-f]{64}")
REQUIRED_IMPLEMENTATION_PATHS = (
    "slurm/build_lidar_world_oracle.sh",
    "src/lidar_world_oracle.py",
    "src/revision_pinned_dataset.py",
)


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def read_bytes(path: Path, description: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        fail(f"cannot read {description}: {error}")


def read_json(path: Path, description: str) -> tuple[dict[str, Any], bytes]:
    content = read_bytes(path, description)
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        fail(f"invalid {description}: {error}")
    if not isinstance(value, dict):
        fail(f"{description} must be a JSON object")
    return value, content


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def require_exact_object(value: Any, name: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{name} must be an object")
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        fail(f"{name} has invalid keys; missing={missing}, extra={extra}")
    return value


def require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        fail(f"{name} must be a positive integer")
    return value


def require_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        fail(f"{name} must be a nonnegative integer")
    return value


def require_full_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or FULL_SHA_PATTERN.fullmatch(value) is None:
        fail(f"{name} must be a full lowercase Git SHA-1")
    return value


def require_file_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or FILE_SHA_PATTERN.fullmatch(value) is None:
        fail(f"{name} must be a lowercase SHA-256")
    return value


def canonical_path(value: Any, name: str, *, directory: bool) -> Path:
    if not isinstance(value, str) or not value:
        fail(f"{name} must be a nonempty path string")
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        fail(f"cannot resolve {name}: {error}")
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        expected = "directory" if directory else "file"
        fail(f"{name} is not a {expected}: {path}")
    return path


def git_output(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "git command failed"
        fail(f"cannot validate implementation equivalence: {detail}")
    return result.stdout.strip()


def submission_objects(
    record: dict[str, Any], description: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    jobs = record.get("jobs")
    paths = record.get("paths")
    source = record.get("source")
    if not isinstance(jobs, dict) or not isinstance(paths, dict):
        fail(f"{description} is missing jobs or paths")
    if not isinstance(source, dict):
        source = {}
    return jobs, paths, source


def validate_recorded_paths(
    paths: dict[str, Any],
    *,
    clip_parquet: Path,
    candidate_directory: Path,
    oracle_directory: Path,
    description: str,
) -> None:
    expected = {
        "clip_parquet": (clip_parquet, False),
        "candidate_directory": (candidate_directory, True),
        "oracle_directory": (oracle_directory, True),
    }
    for name, (expected_path, directory) in expected.items():
        recorded = canonical_path(
            paths.get(name), f"{description} path {name!r}", directory=directory
        )
        if recorded != expected_path:
            fail(f"{description} path {name!r} {recorded} does not match {expected_path}")


def validate_lineage(
    *,
    lineage_record: Path,
    oracle_submission_record: Path,
    oracle_job: int,
    clip_parquet: Path,
    candidate_directory: Path,
    oracle_directory: Path,
) -> dict[str, Any]:
    lineage, lineage_bytes = read_json(lineage_record, "oracle lineage record")
    require_exact_object(
        lineage,
        "oracle lineage record",
        {
            "schema_version",
            "current_oracle",
            "resumed_artifact_producers",
            "resumed_artifacts",
            "prior_submission",
            "implementation_equivalence",
        },
    )
    if lineage["schema_version"] != SCHEMA_VERSION:
        fail(f"oracle lineage record must use schema_version {SCHEMA_VERSION}")

    original, original_bytes = read_json(
        oracle_submission_record, "oracle submission record"
    )
    original_jobs, original_paths, original_source = submission_objects(
        original, "oracle submission record"
    )
    original_job = require_positive_int(
        original_jobs.get("oracle_job"), "oracle submission record oracle job"
    )
    if original_job != oracle_job:
        fail(f"oracle submission record job {original_job} does not match {oracle_job}")
    validate_recorded_paths(
        original_paths,
        clip_parquet=clip_parquet,
        candidate_directory=candidate_directory,
        oracle_directory=oracle_directory,
        description="oracle submission record",
    )
    oracle_commit = require_full_sha(
        original_source.get("repository_commit"),
        "oracle submission record repository commit",
    )
    oracle_repository = canonical_path(
        original_paths.get("repository"),
        "oracle submission record repository",
        directory=True,
    )

    current = require_exact_object(
        lineage["current_oracle"],
        "current_oracle",
        {"job_id", "repository_commit", "submission_record", "submission_record_sha256"},
    )
    if require_positive_int(current["job_id"], "current_oracle.job_id") != oracle_job:
        fail("current_oracle.job_id does not match the reused oracle job")
    if (
        require_full_sha(current["repository_commit"], "current_oracle.repository_commit")
        != oracle_commit
    ):
        fail("current_oracle.repository_commit does not match the oracle submission record")
    current_record_path = canonical_path(
        current["submission_record"], "current_oracle.submission_record", directory=False
    )
    if current_record_path != oracle_submission_record:
        fail("current_oracle.submission_record does not match the supplied record")
    current_record_sha = require_file_sha(
        current["submission_record_sha256"], "current_oracle.submission_record_sha256"
    )
    if current_record_sha != sha256(original_bytes):
        fail("current_oracle submission record SHA-256 does not match its bytes")

    producers = lineage["resumed_artifact_producers"]
    if not isinstance(producers, list):
        fail("resumed_artifact_producers must be an array")
    producer_commits: set[str] = {oracle_commit}
    producer_array_jobs: set[int] = set()
    producer_task_jobs: set[int] = set()
    previous_sort_key: tuple[int, str] | None = None
    for index, raw_producer in enumerate(producers):
        producer = require_exact_object(
            raw_producer,
            f"resumed_artifact_producers[{index}]",
            {"array_job_id", "task_job_ids", "repository_commit"},
        )
        array_job = require_positive_int(
            producer["array_job_id"],
            f"resumed_artifact_producers[{index}].array_job_id",
        )
        commit = require_full_sha(
            producer["repository_commit"],
            f"resumed_artifact_producers[{index}].repository_commit",
        )
        sort_key = (array_job, commit)
        if previous_sort_key is not None and sort_key <= previous_sort_key:
            fail("resumed_artifact_producers must be unique and sorted")
        previous_sort_key = sort_key
        if array_job == oracle_job or array_job in producer_array_jobs:
            fail("resumed producer array jobs must be unique and differ from current job")
        producer_array_jobs.add(array_job)
        raw_tasks = producer["task_job_ids"]
        if not isinstance(raw_tasks, list) or not raw_tasks:
            fail(f"resumed_artifact_producers[{index}].task_job_ids must be nonempty")
        tasks = [
            require_positive_int(value, f"resumed_artifact_producers[{index}].task_job_ids")
            for value in raw_tasks
        ]
        if tasks != sorted(set(tasks)):
            fail("resumed producer task job IDs must be unique and sorted")
        if oracle_job in tasks or producer_task_jobs.intersection(tasks):
            fail("resumed producer task job IDs overlap another producer or current job")
        producer_task_jobs.update(tasks)
        producer_commits.add(commit)

    resumed = require_exact_object(
        lineage["resumed_artifacts"],
        "resumed_artifacts",
        {"count", "inventory"},
    )
    count = require_nonnegative_int(resumed["count"], "resumed_artifacts.count")
    inventory = resumed["inventory"]
    if not isinstance(inventory, list):
        fail("resumed_artifacts.inventory must be an array")
    if count != len(inventory):
        fail("resumed_artifacts.count does not match the inventory length")
    if bool(count) != bool(producers):
        fail("resumed artifact producers are required exactly when resumed artifacts exist")
    previous_path: str | None = None
    for index, raw_artifact in enumerate(inventory):
        artifact = require_exact_object(
            raw_artifact,
            f"resumed_artifacts.inventory[{index}]",
            {"path", "sha256"},
        )
        relative = artifact["path"]
        if not isinstance(relative, str) or not relative:
            fail(f"resumed_artifacts.inventory[{index}].path must be nonempty")
        pure_path = PurePosixPath(relative)
        if pure_path.is_absolute() or ".." in pure_path.parts or "." in pure_path.parts:
            fail(f"resumed artifact path must be normalized and relative: {relative!r}")
        if not relative.startswith("clips/") or not relative.endswith(".oracle.npz"):
            fail(f"resumed artifact path is outside the oracle clip contract: {relative!r}")
        if previous_path is not None and relative <= previous_path:
            fail("resumed artifact inventory paths must be unique and sorted")
        previous_path = relative
        artifact_path = canonical_path(
            str(oracle_directory / pure_path),
            f"resumed artifact {relative!r}",
            directory=False,
        )
        try:
            artifact_path.relative_to(oracle_directory)
        except ValueError:
            fail(f"resumed artifact escapes the oracle directory: {relative!r}")
        expected_sha = require_file_sha(
            artifact["sha256"], f"resumed_artifacts.inventory[{index}].sha256"
        )
        if sha256(read_bytes(artifact_path, f"resumed artifact {relative!r}")) != expected_sha:
            fail(f"resumed artifact SHA-256 does not match: {relative!r}")

    prior_value = lineage["prior_submission"]
    if count == 0:
        if prior_value is not None:
            fail("prior_submission must be null when no artifacts were resumed")
    else:
        prior = require_exact_object(
            prior_value,
            "prior_submission",
            {"path", "sha256"},
        )
        prior_path = canonical_path(
            prior["path"], "prior_submission.path", directory=False
        )
        prior_record, prior_bytes = read_json(prior_path, "prior submission record")
        if prior_record.get("schema_version") not in (1, 2):
            fail("prior submission record must use schema_version 1 or 2")
        prior_sha = require_file_sha(prior["sha256"], "prior_submission.sha256")
        if prior_sha != sha256(prior_bytes):
            fail("prior submission record SHA-256 does not match its bytes")
        prior_jobs, prior_paths, _ = submission_objects(
            prior_record, "prior submission record"
        )
        prior_repository = canonical_path(
            prior_paths.get("repository"),
            "prior submission record repository",
            directory=True,
        )
        if prior_repository != oracle_repository:
            fail("prior submission repository does not match the oracle repository")
        prior_job = require_positive_int(
            prior_jobs.get("oracle_job"), "prior submission record oracle job"
        )
        if prior_job not in producer_array_jobs:
            fail("prior submission oracle job is not a resumed producer array job")
        validate_recorded_paths(
            prior_paths,
            clip_parquet=clip_parquet,
            candidate_directory=candidate_directory,
            oracle_directory=oracle_directory,
            description="prior submission record",
        )

    equivalence = require_exact_object(
        lineage["implementation_equivalence"],
        "implementation_equivalence",
        {"repository", "commits", "paths"},
    )
    equivalence_repository = canonical_path(
        equivalence["repository"],
        "implementation_equivalence.repository",
        directory=True,
    )
    if equivalence_repository != oracle_repository:
        fail("implementation equivalence repository does not match oracle repository")
    commits = equivalence["commits"]
    if not isinstance(commits, list):
        fail("implementation_equivalence.commits must be an array")
    commits = [
        require_full_sha(value, "implementation_equivalence.commits") for value in commits
    ]
    if commits != sorted(producer_commits):
        fail("implementation equivalence commits must equal the sorted artifact producer commits")
    paths = equivalence["paths"]
    if not isinstance(paths, list):
        fail("implementation_equivalence.paths must be an array")
    if len(paths) != len(REQUIRED_IMPLEMENTATION_PATHS):
        fail("implementation equivalence must cover the complete oracle implementation closure")
    recorded_blobs: dict[str, str] = {}
    for index, raw_path in enumerate(paths):
        item = require_exact_object(
            raw_path,
            f"implementation_equivalence.paths[{index}]",
            {"path", "git_blob"},
        )
        path = item["path"]
        if not isinstance(path, str) or path not in REQUIRED_IMPLEMENTATION_PATHS:
            fail(f"unexpected implementation path: {path!r}")
        if path in recorded_blobs:
            fail(f"duplicate implementation path: {path}")
        recorded_blobs[path] = require_full_sha(
            item["git_blob"], f"implementation blob for {path}"
        )
    if list(recorded_blobs) != list(REQUIRED_IMPLEMENTATION_PATHS):
        fail("implementation paths must use the required canonical order")
    for commit in commits:
        resolved_commit = git_output(
            equivalence_repository, "rev-parse", "--verify", f"{commit}^{{commit}}"
        )
        if resolved_commit != commit:
            fail(f"implementation commit does not resolve exactly: {commit}")
        for path, expected_blob in recorded_blobs.items():
            actual_blob = git_output(equivalence_repository, "rev-parse", f"{commit}:{path}")
            if actual_blob != expected_blob:
                fail(f"implementation blob differs at {commit}:{path}")

    return {
        "oracle_artifact_producer_commits": sorted(producer_commits),
        "oracle_lineage_record": str(lineage_record),
        "oracle_lineage_record_sha256": sha256(lineage_bytes),
        "oracle_resumed_artifact_count": count,
    }


def artifact_inventory(artifact_list: Path, oracle_directory: Path) -> list[dict[str, str]]:
    try:
        lines = artifact_list.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        fail(f"cannot read resumed artifact list: {error}")
    inventory: dict[str, str] = {}
    for line_number, raw_value in enumerate(lines, 1):
        value = raw_value.strip()
        if not value or value.startswith("#"):
            continue
        raw_path = Path(value).expanduser()
        candidate = raw_path if raw_path.is_absolute() else oracle_directory / raw_path
        artifact = canonical_path(
            str(candidate), f"artifact list line {line_number}", directory=False
        )
        try:
            relative = artifact.relative_to(oracle_directory).as_posix()
        except ValueError:
            fail(f"artifact list line {line_number} escapes the oracle directory")
        if not relative.startswith("clips/") or not relative.endswith(".oracle.npz"):
            fail(f"artifact list line {line_number} is not an oracle clip artifact")
        if relative in inventory:
            fail(f"artifact list contains a duplicate path: {relative}")
        inventory[relative] = sha256(read_bytes(artifact, f"artifact {relative!r}"))
    return [{"path": path, "sha256": inventory[path]} for path in sorted(inventory)]


def parse_producer(value: str) -> dict[str, Any]:
    parts = value.split(":", 2)
    if len(parts) != 3:
        fail("--resumed-producer must be ARRAY_JOB_ID:COMMIT:TASK_JOB_ID[,TASK_JOB_ID]")
    try:
        array_job = int(parts[0])
        tasks = [int(item) for item in parts[2].split(",") if item]
    except ValueError:
        fail("--resumed-producer job IDs must be integers")
    require_positive_int(array_job, "--resumed-producer array job")
    require_full_sha(parts[1], "--resumed-producer commit")
    if not tasks or any(task <= 0 for task in tasks):
        fail("--resumed-producer task job IDs must be positive and nonempty")
    return {
        "array_job_id": array_job,
        "task_job_ids": sorted(set(tasks)),
        "repository_commit": parts[1],
    }


def create_lineage(arguments: argparse.Namespace) -> None:
    output = Path(arguments.output).expanduser()
    if output.exists():
        fail(f"refusing to replace existing lineage record: {output}")
    oracle_submission = canonical_path(
        arguments.oracle_submission_record,
        "oracle submission record",
        directory=False,
    )
    original, original_bytes = read_json(oracle_submission, "oracle submission record")
    jobs, paths, source = submission_objects(original, "oracle submission record")
    oracle_job = require_positive_int(jobs.get("oracle_job"), "oracle submission record job")
    oracle_commit = require_full_sha(
        source.get("repository_commit"), "oracle submission record repository commit"
    )
    repository = canonical_path(
        paths.get("repository"), "oracle submission record repository", directory=True
    )
    oracle_directory = canonical_path(
        paths.get("oracle_directory"),
        "oracle submission record oracle directory",
        directory=True,
    )
    producers = sorted(
        (parse_producer(value) for value in arguments.resumed_producer),
        key=lambda value: (value["array_job_id"], value["repository_commit"]),
    )
    inventory = artifact_inventory(
        canonical_path(arguments.artifact_list, "resumed artifact list", directory=False),
        oracle_directory,
    )
    if bool(inventory) != bool(producers):
        fail("resumed producers are required exactly when the artifact list is nonempty")
    prior: dict[str, str] | None = None
    if inventory:
        if not arguments.prior_submission_record:
            fail("--prior-submission-record is required when artifacts were resumed")
        prior_path = canonical_path(
            arguments.prior_submission_record, "prior submission record", directory=False
        )
        prior = {
            "path": str(prior_path),
            "sha256": sha256(read_bytes(prior_path, "prior submission record")),
        }
    elif arguments.prior_submission_record:
        fail("--prior-submission-record is invalid when no artifacts were resumed")

    commits = sorted({oracle_commit, *(producer["repository_commit"] for producer in producers)})
    implementation_paths = []
    for path in REQUIRED_IMPLEMENTATION_PATHS:
        blobs = {git_output(repository, "rev-parse", f"{commit}:{path}") for commit in commits}
        if len(blobs) != 1:
            fail(f"oracle implementation is not equivalent across commits at {path}")
        implementation_paths.append({"path": path, "git_blob": blobs.pop()})
    payload = {
        "schema_version": SCHEMA_VERSION,
        "current_oracle": {
            "job_id": oracle_job,
            "repository_commit": oracle_commit,
            "submission_record": str(oracle_submission),
            "submission_record_sha256": sha256(original_bytes),
        },
        "resumed_artifact_producers": producers,
        "resumed_artifacts": {"count": len(inventory), "inventory": inventory},
        "prior_submission": prior,
        "implementation_equivalence": {
            "repository": str(repository),
            "commits": commits,
            "paths": implementation_paths,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=".oracle-lineage.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        original_paths = submission_objects(original, "oracle submission record")[1]
        validate_lineage(
            lineage_record=temporary,
            oracle_submission_record=oracle_submission,
            oracle_job=oracle_job,
            clip_parquet=canonical_path(
                original_paths.get("clip_parquet"),
                "oracle submission record clip parquet",
                directory=False,
            ),
            candidate_directory=canonical_path(
                original_paths.get("candidate_directory"),
                "oracle submission record candidate directory",
                directory=True,
            ),
            oracle_directory=oracle_directory,
        )
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create", help="create a lineage sidecar")
    create.add_argument("--output", required=True, help="new sidecar path")
    create.add_argument(
        "--oracle-submission-record",
        required=True,
        help="submission.json for the current oracle job",
    )
    create.add_argument(
        "--prior-submission-record",
        help="submission.json associated with the resumed producer array",
    )
    create.add_argument(
        "--resumed-producer",
        action="append",
        default=[],
        metavar="ARRAY_JOB:COMMIT:TASK_JOB[,TASK_JOB]",
        help="resumed producer identity; repeat for each producer array",
    )
    create.add_argument(
        "--artifact-list",
        required=True,
        help="newline-delimited resumed artifact paths relative to the oracle directory",
    )

    validate = subparsers.add_parser("validate", help="validate and summarize a sidecar")
    validate.add_argument("--record", required=True)
    validate.add_argument("--oracle-submission-record", required=True)
    validate.add_argument("--oracle-job", type=int, required=True)
    validate.add_argument("--clip-parquet", required=True)
    validate.add_argument("--candidate-directory", required=True)
    validate.add_argument("--oracle-directory", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "create":
        create_lineage(arguments)
        return
    record = canonical_path(arguments.record, "oracle lineage record", directory=False)
    original = canonical_path(
        arguments.oracle_submission_record, "oracle submission record", directory=False
    )
    clip = canonical_path(arguments.clip_parquet, "clip parquet", directory=False)
    candidates = canonical_path(
        arguments.candidate_directory, "candidate directory", directory=True
    )
    oracle = canonical_path(arguments.oracle_directory, "oracle directory", directory=True)
    summary = validate_lineage(
        lineage_record=record,
        oracle_submission_record=original,
        oracle_job=require_positive_int(arguments.oracle_job, "oracle job"),
        clip_parquet=clip,
        candidate_directory=candidates,
        oracle_directory=oracle,
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
