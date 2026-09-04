from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STAGE_SCRIPT = REPOSITORY_ROOT / "slurm" / "run_benchmark_stage.sh"
SUBMIT_SCRIPT = REPOSITORY_ROOT / "slurm" / "submit_world_benchmark.sh"
LINEAGE_TOOL = REPOSITORY_ROOT / "tools" / "oracle_lineage.py"


def run_checked(*command: str | Path, **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(part) for part in command],
        check=True,
        capture_output=True,
        text=True,
        **kwargs,
    )


def initialize_repository(path: Path, files: dict[str, str] | None = None) -> str:
    path.mkdir()
    run_checked("git", "init", "-q", path)
    run_checked("git", "-C", path, "config", "user.name", "Test User")
    run_checked("git", "-C", path, "config", "user.email", "test@example.com")
    for relative, content in (files or {"tracked.txt": "clean\n"}).items():
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        if relative.endswith(".sh"):
            target.chmod(0o755)
    run_checked("git", "-C", path, "add", ".")
    run_checked("git", "-C", path, "commit", "-q", "-m", "fixture")
    return run_checked("git", "-C", path, "rev-parse", "HEAD").stdout.strip()


def initialize_submission_repository(path: Path) -> str:
    return initialize_repository(
        path,
        {
            "slurm/run_benchmark_stage.sh": STAGE_SCRIPT.read_text(encoding="utf-8"),
            "slurm/submit_world_benchmark.sh": SUBMIT_SCRIPT.read_text(encoding="utf-8"),
            "tools/oracle_lineage.py": LINEAGE_TOOL.read_text(encoding="utf-8"),
        },
    )


def initialize_oracle_repository(path: Path) -> tuple[str, str]:
    producer_commit = initialize_repository(
        path,
        {
            "slurm/build_lidar_world_oracle.sh": "#!/bin/sh\nexit 0\n",
            "src/lidar_world_oracle.py": "# oracle fixture\n",
            "src/revision_pinned_dataset.py": "# revision fixture\n",
        },
    )
    (path / "submission-change.txt").write_text("resource metadata only\n", encoding="utf-8")
    run_checked("git", "-C", path, "add", ".")
    run_checked("git", "-C", path, "commit", "-q", "-m", "submission only")
    current_commit = run_checked("git", "-C", path, "rev-parse", "HEAD").stdout.strip()
    return producer_commit, current_commit


def install_fake_sbatch(root: Path) -> tuple[Path, Path, Path]:
    fake_bin = root / "bin"
    fake_bin.mkdir()
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        "value=$(cat \"${FAKE_SBATCH_COUNTER}\")\n"
        "value=$((value + 1))\n"
        "printf '%s\\n' \"${value}\" > \"${FAKE_SBATCH_COUNTER}\"\n"
        "printf 'CALL\\n' >> \"${FAKE_SBATCH_LOG}\"\n"
        "printf '%s\\n' \"$@\" >> \"${FAKE_SBATCH_LOG}\"\n"
        "if env | grep '^BENCHMARK_' >/dev/null; then\n"
        "  echo 'inherited BENCHMARK_* variable reached sbatch' >&2\n"
        "  exit 99\n"
        "fi\n"
        "printf '%s\\n' \"${value}\"\n",
        encoding="utf-8",
    )
    fake_sbatch.chmod(0o755)
    counter = root / "counter"
    counter.write_text("20000\n", encoding="utf-8")
    return fake_bin, counter, root / "sbatch.log"


def submission_environment(
    *,
    repository: Path,
    root: Path,
    clip_parquet: Path,
    candidate_dir: Path,
    oracle_dir: Path,
    run_dir: Path,
    extra: dict[str, str],
) -> tuple[dict[str, str], Path]:
    fake_bin, counter, log = install_fake_sbatch(root)
    environment = os.environ.copy()
    environment.update(
        {
            "BENCHMARK_CANDIDATE_DIR": str(candidate_dir),
            "BENCHMARK_CLIP_PARQUET": str(clip_parquet),
            "BENCHMARK_ORACLE_DIR": str(oracle_dir),
            "BENCHMARK_PYTHON": sys.executable,
            "BENCHMARK_REPO_ROOT": str(repository),
            "BENCHMARK_RUN_DIR": str(run_dir),
            "FAKE_SBATCH_COUNTER": str(counter),
            "FAKE_SBATCH_LOG": str(log),
            "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
            **extra,
        }
    )
    return environment, log


def write_original_oracle_record(
    path: Path,
    *,
    job_id: int,
    repository: Path,
    repository_commit: str,
    clip_parquet: Path,
    candidate_dir: Path,
    oracle_dir: Path,
) -> None:
    payload = {
        "schema_version": 2,
        "jobs": {"oracle_job": job_id},
        "paths": {
            "repository": str(repository),
            "clip_parquet": str(clip_parquet),
            "candidate_directory": str(candidate_dir),
            "oracle_directory": str(oracle_dir),
        },
        "configuration": {
            "expected_candidates": 6,
            "expected_oracle_shards": 8,
            "expected_rows": 2000,
            "oracle_concurrency": 8,
            "oracle_cpus_per_task": 3,
            "oracle_memory_per_task": "7G",
            "reused_oracle_job": False,
        },
        "source": {"repository_commit": repository_commit},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_prior_oracle_record(
    path: Path,
    *,
    job_id: int,
    repository: Path,
    clip_parquet: Path,
    candidate_dir: Path,
    oracle_dir: Path,
) -> None:
    payload = {
        "schema_version": 1,
        "jobs": {"oracle_job": job_id},
        "paths": {
            "repository": str(repository),
            "clip_parquet": str(clip_parquet),
            "candidate_directory": str(candidate_dir),
            "oracle_directory": str(oracle_dir),
        },
        "configuration": {"reused_oracle_job": True},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def create_lineage_record(
    root: Path,
    *,
    oracle_submission_record: Path,
    oracle_repository: Path,
    producer_commit: str,
    clip_parquet: Path,
    candidate_dir: Path,
    oracle_dir: Path,
) -> tuple[Path, Path, Path]:
    clips = oracle_dir / "clips"
    clips.mkdir(exist_ok=True)
    first = clips / "first.oracle.npz"
    second = clips / "second.oracle.npz"
    first.write_bytes(b"first resumed artifact")
    second.write_bytes(b"second resumed artifact")
    artifact_list = root / "resumed-artifacts.txt"
    artifact_list.write_text(
        "clips/second.oracle.npz\nclips/first.oracle.npz\n", encoding="utf-8"
    )
    prior_record = root / "predecessor-submission.json"
    write_prior_oracle_record(
        prior_record,
        job_id=666,
        repository=oracle_repository,
        clip_parquet=clip_parquet,
        candidate_dir=candidate_dir,
        oracle_dir=oracle_dir,
    )
    lineage_record = root / "oracle-lineage.json"
    run_checked(
        sys.executable,
        LINEAGE_TOOL,
        "create",
        "--output",
        lineage_record,
        "--oracle-submission-record",
        oracle_submission_record,
        "--prior-submission-record",
        prior_record,
        "--resumed-producer",
        f"666:{producer_commit}:667,668",
        "--artifact-list",
        artifact_list,
    )
    return lineage_record, prior_record, first


class RuntimeSourceGuardTest(unittest.TestCase):
    def stage_environment(
        self, repository: Path, expected_commit: str, temporary_root: Path
    ) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(
            {
                "BENCHMARK_ACTION": "unsupported-test-action",
                "BENCHMARK_CANDIDATE_DIR": str(temporary_root / "candidates"),
                "BENCHMARK_CLIP_PARQUET": str(temporary_root / "clips.parquet"),
                "BENCHMARK_ORACLE_DIR": str(temporary_root / "oracle"),
                "BENCHMARK_PYTHON": sys.executable,
                "BENCHMARK_REPOSITORY_COMMIT": expected_commit,
                "BENCHMARK_REPO_ROOT": str(repository),
                "BENCHMARK_RUN_DIR": str(temporary_root / "run"),
            }
        )
        return environment

    def test_world_wrappers_match_dual_a4500_resource_and_authentication_plan(self) -> None:
        oracle = (REPOSITORY_ROOT / "slurm/build_lidar_world_oracle.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("#SBATCH --cpus-per-task=3", oracle)
        self.assertIn("#SBATCH --mem=7G", oracle)
        self.assertIn("#SBATCH --array=0-7%8", oracle)

        training = (REPOSITORY_ROOT / "slurm/train_world_model.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("#SBATCH --cpus-per-task=4", training)
        self.assertIn("#SBATCH --gres=gpu:1", training)
        self.assertIn("#SBATCH --mem=28G", training)
        for argument in (
            '--epochs "${WORLD_EPOCHS:-30}"',
            '--workers "${WORLD_WORKERS:-4}"',
            '--learning-rate "${WORLD_LEARNING_RATE:-0.0003}"',
            '--weight-decay "${WORLD_WEIGHT_DECAY:-0.0001}"',
        ):
            self.assertIn(argument, training)

        evaluation = (REPOSITORY_ROOT / "slurm/evaluate_world_model.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("#SBATCH --mem=28G", evaluation)
        for variable in (
            "WORLD_PROTOCOL",
            "WORLD_SELECTION_RECORD",
            "WORLD_EVALUATION_AUDIT",
        ):
            self.assertIn(f'${{{variable}:?', evaluation)
        for argument in ("--protocol", "--selection-record", "--evaluation-audit"):
            self.assertIn(argument, evaluation)

    def test_stage_accepts_only_clean_expected_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repository"
            commit = initialize_repository(repository)
            result = subprocess.run(
                [str(STAGE_SCRIPT)],
                capture_output=True,
                text=True,
                env=self.stage_environment(repository, commit, root),
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("unsupported BENCHMARK_ACTION", result.stderr)
            self.assertNotIn("repository HEAD", result.stderr)
            self.assertNotIn("worktree is dirty", result.stderr)

    def test_stage_rejects_head_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repository"
            commit = initialize_repository(repository)
            result = subprocess.run(
                [str(STAGE_SCRIPT)],
                capture_output=True,
                text=True,
                env=self.stage_environment(repository, "0" * 40, root),
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn(f"repository HEAD {commit} does not match expected commit", result.stderr)

    def test_stage_rejects_dirty_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repository"
            commit = initialize_repository(repository)
            (repository / "untracked.txt").write_text("dirty\n", encoding="utf-8")
            result = subprocess.run(
                [str(STAGE_SCRIPT)],
                capture_output=True,
                text=True,
                env=self.stage_environment(repository, commit, root),
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("repository worktree is dirty", result.stderr)

    def test_submission_exports_and_records_one_exact_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repository"
            commit = initialize_submission_repository(repository)
            clip_parquet = root / "clips.parquet"
            clip_parquet.write_bytes(b"fixture")
            candidate_dir = root / "candidates"
            oracle_dir = root / "oracle"
            run_dir = root / "run"
            environment, log = submission_environment(
                repository=repository,
                root=root,
                clip_parquet=clip_parquet,
                candidate_dir=candidate_dir,
                oracle_dir=oracle_dir,
                run_dir=run_dir,
                extra={
                    "BENCHMARK_CANDIDATE_JOB_ID": "12345",
                    "BENCHMARK_ACTION": "hostile-inherited-action",
                    "BENCHMARK_REPOSITORY_COMMIT": "0" * 40,
                    "BENCHMARK_TRAIN_OUTPUT": str(root / "hostile-output"),
                    "BENCHMARK_TRAIN_SEED": "999",
                },
            )
            run_checked(
                repository / "slurm" / "submit_world_benchmark.sh",
                env=environment,
            )

            submission = json.loads((run_dir / "submission.json").read_text(encoding="utf-8"))
            self.assertEqual(submission["source"]["repository_commit"], commit)
            self.assertEqual(submission["source"]["oracle_repository_commit"], commit)
            self.assertNotIn("oracle_submission_record", submission["source"])
            configuration = submission["configuration"]
            self.assertEqual(configuration["split_seed"], 2026)
            self.assertEqual(configuration["split_ratios"], [0.8, 0.1, 0.1])
            self.assertEqual(configuration["training_seeds"], [2026, 2027])
            self.assertEqual(configuration["epochs"], 30)
            self.assertEqual(configuration["batch_size"], 4)
            self.assertEqual(configuration["workers"], 4)
            self.assertEqual(configuration["base_channels"], 16)
            self.assertTrue(configuration["automatic_mixed_precision"])
            self.assertEqual(configuration["learning_rate"], 0.0003)
            self.assertEqual(configuration["weight_decay"], 0.0001)
            self.assertEqual(configuration["bootstrap_seed"], 2026)
            self.assertEqual(configuration["bootstrap_replicates"], 10000)
            self.assertEqual(configuration["maximum_concurrent_gpu_jobs"], 2)
            self.assertEqual(configuration["oracle_concurrency"], 8)
            self.assertEqual(configuration["oracle_cpus_per_task"], 3)
            self.assertEqual(configuration["oracle_memory_per_task"], "7G")
            resources = submission["resources"]
            self.assertEqual(resources["partition"], "dualcard")
            self.assertEqual(resources["expected_gpu_name"], "NVIDIA RTX A4500")
            self.assertEqual(resources["oracle"]["array_concurrency"], 8)
            self.assertEqual(resources["oracle"]["tasks_per_array_element"], 1)
            self.assertEqual(resources["train_seed_2026"]["nodes"], 1)
            self.assertEqual(resources["train_seed_2026"]["tasks_per_job"], 1)
            self.assertEqual(resources["train_seed_2026"]["memory"], "28G")
            self.assertEqual(resources["train_seed_2026"]["gpus_per_task"], 1)
            self.assertEqual(resources["train_seed_2027"], resources["train_seed_2026"])
            jobs = submission["jobs"]
            dependencies = submission["dependencies"]
            self.assertEqual(dependencies["type"], "afterok")
            self.assertEqual(
                dependencies["stages"]["select"],
                [jobs["train_2026_job"], jobs["train_2027_job"]],
            )
            self.assertEqual(
                dependencies["stages"]["aggregate"],
                [
                    jobs["learned_rerank_job"],
                    jobs["persistence_rerank_job"],
                    jobs["forecast_compare_job"],
                ],
            )
            runtime = submission["runtime"]
            self.assertEqual(runtime["python_executable"], str(Path(sys.executable).resolve()))
            self.assertEqual(len(runtime["stage_script_sha256"]), 64)
            calls = log.read_text(encoding="utf-8").split("CALL\n")[1:]
            self.assertEqual(len(calls), 11)
            expected_export = f"BENCHMARK_REPOSITORY_COMMIT={commit}"
            for call in calls:
                arguments = call.splitlines()
                self.assertTrue(any(expected_export in argument for argument in arguments))
                self.assertEqual(arguments[-1], str(repository / "slurm/run_benchmark_stage.sh"))
            self.assertTrue(
                any("BENCHMARK_ACTION=oracle" in argument for argument in calls[0].splitlines())
            )
            oracle_arguments = calls[0].splitlines()
            self.assertIn("--partition=dualcard", oracle_arguments)
            self.assertIn("--array=0-7%8", oracle_arguments)
            self.assertIn("--cpus-per-task=3", oracle_arguments)
            self.assertIn("--mem=7G", oracle_arguments)
            for call in calls[2:4]:
                arguments = call.splitlines()
                self.assertIn("--gres=gpu:1", arguments)
                self.assertIn("--mem=28G", arguments)
                self.assertIn(f"--dependency=afterok:{jobs['prep_job']}", arguments)
                export_argument = next(
                    argument for argument in arguments if argument.startswith("--export=")
                )
                for expected in (
                    "BENCHMARK_SPLIT_RATIOS=0.8:0.1:0.1",
                    "BENCHMARK_BATCH_SIZE=4",
                    "BENCHMARK_WORKERS=4",
                    "BENCHMARK_BASE_CHANNELS=16",
                    "BENCHMARK_BOOTSTRAP_REPLICATES=10000",
                    "BENCHMARK_EXPECTED_GPU_NAME=NVIDIA RTX A4500",
                ):
                    self.assertIn(expected, export_argument)

    def test_submission_records_and_exports_runtime_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repository"
            initialize_submission_repository(repository)
            clip_parquet = root / "clips.parquet"
            clip_parquet.write_bytes(b"fixture")
            run_dir = root / "run"
            environment, log = submission_environment(
                repository=repository,
                root=root,
                clip_parquet=clip_parquet,
                candidate_dir=root / "candidates",
                oracle_dir=root / "oracle",
                run_dir=run_dir,
                extra={
                    "BENCHMARK_CANDIDATE_JOB_ID": "12345",
                    "BENCHMARK_EPOCHS": "40",
                    "BENCHMARK_BATCH_SIZE": "5",
                    "BENCHMARK_WORKERS": "2",
                    "BENCHMARK_BASE_CHANNELS": "12",
                },
            )
            run_checked(repository / "slurm" / "submit_world_benchmark.sh", env=environment)
            submission = json.loads((run_dir / "submission.json").read_text(encoding="utf-8"))
            configuration = submission["configuration"]
            self.assertEqual(
                (
                    configuration["epochs"],
                    configuration["batch_size"],
                    configuration["workers"],
                    configuration["base_channels"],
                ),
                (40, 5, 2, 12),
            )
            calls = log.read_text(encoding="utf-8").split("CALL\n")[2:]
            for call in calls:
                export_argument = next(
                    argument for argument in call.splitlines() if argument.startswith("--export=")
                )
                for expected in (
                    "BENCHMARK_EPOCHS=40",
                    "BENCHMARK_BATCH_SIZE=5",
                    "BENCHMARK_WORKERS=2",
                    "BENCHMARK_BASE_CHANNELS=12",
                ):
                    self.assertIn(expected, export_argument)

    def test_submission_rejects_nonprotocol_split_or_bootstrap(self) -> None:
        cases = (
            ({"BENCHMARK_SPLIT_SEED": "17"}, "split seed 2026"),
            ({"BENCHMARK_SPLIT_RATIOS": "0.7,0.2,0.1"}, "ratios 0.8,0.1,0.1"),
            ({"BENCHMARK_BOOTSTRAP_SEED": "17"}, "bootstrap seed 2026"),
            ({"BENCHMARK_BOOTSTRAP_REPLICATES": "9999"}, "10000 replicates"),
        )
        for extra, expected_error in cases:
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                repository = root / "repository"
                initialize_submission_repository(repository)
                clip_parquet = root / "clips.parquet"
                clip_parquet.write_bytes(b"fixture")
                environment, log = submission_environment(
                    repository=repository,
                    root=root,
                    clip_parquet=clip_parquet,
                    candidate_dir=root / "candidates",
                    oracle_dir=root / "oracle",
                    run_dir=root / "run",
                    extra={"BENCHMARK_CANDIDATE_JOB_ID": "12345", **extra},
                )
                result = subprocess.run(
                    [str(repository / "slurm" / "submit_world_benchmark.sh")],
                    capture_output=True,
                    text=True,
                    env=environment,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(expected_error, result.stderr)
                self.assertFalse(log.exists())

    def test_reused_oracle_accepts_exact_original_submission_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repository"
            current_commit = initialize_submission_repository(repository)
            oracle_repository = root / "oracle-repository"
            producer_commit, oracle_commit = initialize_oracle_repository(
                oracle_repository
            )
            clip_parquet = root / "clips.parquet"
            clip_parquet.write_bytes(b"fixture")
            candidate_dir = root / "candidates"
            oracle_dir = root / "oracle"
            candidate_dir.mkdir()
            oracle_dir.mkdir()
            prior_record = root / "prior-submission.json"
            write_original_oracle_record(
                prior_record,
                job_id=777,
                repository=oracle_repository,
                repository_commit=oracle_commit,
                clip_parquet=clip_parquet,
                candidate_dir=candidate_dir,
                oracle_dir=oracle_dir,
            )
            prior_sha256 = hashlib.sha256(prior_record.read_bytes()).hexdigest()
            lineage_record, predecessor_record, _ = create_lineage_record(
                root,
                oracle_submission_record=prior_record,
                oracle_repository=oracle_repository,
                producer_commit=producer_commit,
                clip_parquet=clip_parquet,
                candidate_dir=candidate_dir,
                oracle_dir=oracle_dir,
            )
            lineage_sha256 = hashlib.sha256(lineage_record.read_bytes()).hexdigest()
            run_dir = root / "run"
            environment, log = submission_environment(
                repository=repository,
                root=root,
                clip_parquet=clip_parquet,
                candidate_dir=candidate_dir,
                oracle_dir=oracle_dir,
                run_dir=run_dir,
                extra={
                    "BENCHMARK_ORACLE_JOB_ID": "777",
                    "BENCHMARK_ORACLE_SUBMISSION_RECORD": str(prior_record),
                    "BENCHMARK_ORACLE_LINEAGE_RECORD": str(lineage_record),
                },
            )
            run_checked(
                repository / "slurm" / "submit_world_benchmark.sh",
                env=environment,
            )

            submission = json.loads((run_dir / "submission.json").read_text(encoding="utf-8"))
            self.assertEqual(submission["jobs"]["oracle_job"], 777)
            self.assertIsNone(submission["jobs"]["candidate_job_id"])
            self.assertTrue(submission["configuration"]["reused_oracle_job"])
            self.assertEqual(submission["configuration"]["oracle_concurrency"], 8)
            self.assertEqual(submission["configuration"]["oracle_cpus_per_task"], 3)
            self.assertEqual(submission["configuration"]["oracle_memory_per_task"], "7G")
            self.assertEqual(submission["source"]["repository_commit"], current_commit)
            self.assertEqual(
                submission["source"]["oracle_repository_commit"], oracle_commit
            )
            self.assertEqual(
                submission["source"]["oracle_submission_record"],
                str(prior_record.resolve()),
            )
            self.assertEqual(
                submission["source"]["oracle_submission_record_sha256"], prior_sha256
            )
            self.assertEqual(
                submission["source"]["oracle_lineage_record"],
                str(lineage_record.resolve()),
            )
            self.assertEqual(
                submission["source"]["oracle_lineage_record_sha256"], lineage_sha256
            )
            self.assertEqual(
                submission["source"]["oracle_artifact_producer_commits"],
                sorted([producer_commit, oracle_commit]),
            )
            self.assertEqual(submission["source"]["oracle_resumed_artifact_count"], 2)
            lineage = json.loads(lineage_record.read_text(encoding="utf-8"))
            self.assertEqual(
                lineage["prior_submission"]["path"], str(predecessor_record.resolve())
            )
            calls = log.read_text(encoding="utf-8").split("CALL\n")[1:]
            self.assertEqual(len(calls), 10)
            expected_export = f"BENCHMARK_REPOSITORY_COMMIT={current_commit}"
            self.assertTrue(
                all(
                    any(expected_export in argument for argument in call.splitlines())
                    for call in calls
                )
            )

    def test_reused_oracle_rejects_configuration_drift_before_sbatch(self) -> None:
        cases = (
            ("BENCHMARK_EXPECTED_SHARDS", "7", "expected_oracle_shards"),
            ("BENCHMARK_EXPECTED_ROWS", "1999", "expected_rows"),
            ("BENCHMARK_EXPECTED_CANDIDATES", "5", "expected_candidates"),
        )
        for variable, value, expected_error in cases:
            with self.subTest(variable=variable), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                repository = root / "repository"
                initialize_submission_repository(repository)
                oracle_repository = root / "oracle-repository"
                producer_commit, oracle_commit = initialize_oracle_repository(
                    oracle_repository
                )
                clip_parquet = root / "clips.parquet"
                clip_parquet.write_bytes(b"fixture")
                candidate_dir = root / "candidates"
                oracle_dir = root / "oracle"
                candidate_dir.mkdir()
                oracle_dir.mkdir()
                original_record = root / "original-submission.json"
                write_original_oracle_record(
                    original_record,
                    job_id=777,
                    repository=oracle_repository,
                    repository_commit=oracle_commit,
                    clip_parquet=clip_parquet,
                    candidate_dir=candidate_dir,
                    oracle_dir=oracle_dir,
                )
                lineage_record, _, _ = create_lineage_record(
                    root,
                    oracle_submission_record=original_record,
                    oracle_repository=oracle_repository,
                    producer_commit=producer_commit,
                    clip_parquet=clip_parquet,
                    candidate_dir=candidate_dir,
                    oracle_dir=oracle_dir,
                )
                environment, log = submission_environment(
                    repository=repository,
                    root=root,
                    clip_parquet=clip_parquet,
                    candidate_dir=candidate_dir,
                    oracle_dir=oracle_dir,
                    run_dir=root / "run",
                    extra={
                        "BENCHMARK_ORACLE_JOB_ID": "777",
                        "BENCHMARK_ORACLE_SUBMISSION_RECORD": str(original_record),
                        "BENCHMARK_ORACLE_LINEAGE_RECORD": str(lineage_record),
                        variable: value,
                    },
                )
                result = subprocess.run(
                    [str(repository / "slurm" / "submit_world_benchmark.sh")],
                    capture_output=True,
                    text=True,
                    env=environment,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected_error, result.stderr)
                self.assertFalse(log.exists())

    def test_reused_oracle_rejects_invalid_lineage_before_sbatch(self) -> None:
        cases = (
            ("artifact", "resumed artifact SHA-256 does not match"),
            ("prior_record", "prior submission record SHA-256 does not match"),
            ("implementation_blob", "implementation blob differs"),
        )
        for case, expected_error in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                repository = root / "repository"
                initialize_submission_repository(repository)
                oracle_repository = root / "oracle-repository"
                producer_commit, oracle_commit = initialize_oracle_repository(
                    oracle_repository
                )
                clip_parquet = root / "clips.parquet"
                clip_parquet.write_bytes(b"fixture")
                candidate_dir = root / "candidates"
                oracle_dir = root / "oracle"
                candidate_dir.mkdir()
                oracle_dir.mkdir()
                original_record = root / "original-submission.json"
                write_original_oracle_record(
                    original_record,
                    job_id=777,
                    repository=oracle_repository,
                    repository_commit=oracle_commit,
                    clip_parquet=clip_parquet,
                    candidate_dir=candidate_dir,
                    oracle_dir=oracle_dir,
                )
                lineage_record, prior_record, first_artifact = create_lineage_record(
                    root,
                    oracle_submission_record=original_record,
                    oracle_repository=oracle_repository,
                    producer_commit=producer_commit,
                    clip_parquet=clip_parquet,
                    candidate_dir=candidate_dir,
                    oracle_dir=oracle_dir,
                )
                if case == "artifact":
                    first_artifact.write_bytes(b"changed")
                elif case == "prior_record":
                    prior_record.write_text(
                        prior_record.read_text(encoding="utf-8") + "\n", encoding="utf-8"
                    )
                else:
                    lineage = json.loads(lineage_record.read_text(encoding="utf-8"))
                    lineage["implementation_equivalence"]["paths"][0]["git_blob"] = "0" * 40
                    lineage_record.write_text(
                        json.dumps(lineage, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                environment, log = submission_environment(
                    repository=repository,
                    root=root,
                    clip_parquet=clip_parquet,
                    candidate_dir=candidate_dir,
                    oracle_dir=oracle_dir,
                    run_dir=root / "run",
                    extra={
                        "BENCHMARK_ORACLE_JOB_ID": "777",
                        "BENCHMARK_ORACLE_SUBMISSION_RECORD": str(original_record),
                        "BENCHMARK_ORACLE_LINEAGE_RECORD": str(lineage_record),
                    },
                )
                result = subprocess.run(
                    [str(repository / "slurm" / "submit_world_benchmark.sh")],
                    capture_output=True,
                    text=True,
                    env=environment,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected_error, result.stderr)
                self.assertFalse(log.exists())

    def test_reused_oracle_rejects_unbound_submission_records(self) -> None:
        cases = (
            ("job", "oracle submission record job"),
            ("clip_parquet", "clip_parquet"),
            ("candidate_directory", "candidate_directory"),
            ("oracle_directory", "oracle_directory"),
            ("invalid_commit", "no full lowercase source commit"),
            ("head_mismatch", "oracle source HEAD"),
            ("dirty", "oracle source worktree is dirty"),
        )
        for case, expected_error in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                repository = root / "repository"
                initialize_submission_repository(repository)
                oracle_repository = root / "oracle-repository"
                oracle_commit = initialize_repository(oracle_repository)
                clip_parquet = root / "clips.parquet"
                clip_parquet.write_bytes(b"fixture")
                candidate_dir = root / "candidates"
                oracle_dir = root / "oracle"
                candidate_dir.mkdir()
                oracle_dir.mkdir()
                prior_record = root / "prior-submission.json"
                write_original_oracle_record(
                    prior_record,
                    job_id=777 if case != "job" else 778,
                    repository=oracle_repository,
                    repository_commit=(
                        "main" if case == "invalid_commit" else oracle_commit
                    ),
                    clip_parquet=(
                        root / "other.parquet"
                        if case == "clip_parquet"
                        else clip_parquet
                    ),
                    candidate_dir=(
                        root / "other-candidates"
                        if case == "candidate_directory"
                        else candidate_dir
                    ),
                    oracle_dir=(
                        root / "other-oracle"
                        if case == "oracle_directory"
                        else oracle_dir
                    ),
                )
                if case == "clip_parquet":
                    (root / "other.parquet").write_bytes(b"other")
                elif case == "candidate_directory":
                    (root / "other-candidates").mkdir()
                elif case == "oracle_directory":
                    (root / "other-oracle").mkdir()
                elif case == "head_mismatch":
                    (oracle_repository / "tracked.txt").write_text("next\n", encoding="utf-8")
                    run_checked("git", "-C", oracle_repository, "add", ".")
                    run_checked("git", "-C", oracle_repository, "commit", "-q", "-m", "next")
                elif case == "dirty":
                    (oracle_repository / "untracked.txt").write_text("dirty\n", encoding="utf-8")
                run_dir = root / "run"
                environment, log = submission_environment(
                    repository=repository,
                    root=root,
                    clip_parquet=clip_parquet,
                    candidate_dir=candidate_dir,
                    oracle_dir=oracle_dir,
                    run_dir=run_dir,
                    extra={
                        "BENCHMARK_ORACLE_JOB_ID": "777",
                        "BENCHMARK_ORACLE_SUBMISSION_RECORD": str(prior_record),
                        "BENCHMARK_ORACLE_LINEAGE_RECORD": str(
                            root / "unused-lineage.json"
                        ),
                    },
                )
                result = subprocess.run(
                    [str(repository / "slurm" / "submit_world_benchmark.sh")],
                    capture_output=True,
                    text=True,
                    env=environment,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected_error, result.stderr)
                self.assertFalse(log.exists())

    def test_reused_oracle_requires_submission_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repository"
            initialize_submission_repository(repository)
            clip_parquet = root / "clips.parquet"
            clip_parquet.write_bytes(b"fixture")
            environment, log = submission_environment(
                repository=repository,
                root=root,
                clip_parquet=clip_parquet,
                candidate_dir=root / "candidates",
                oracle_dir=root / "oracle",
                run_dir=root / "run",
                extra={"BENCHMARK_ORACLE_JOB_ID": "777"},
            )
            result = subprocess.run(
                [str(repository / "slurm" / "submit_world_benchmark.sh")],
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("BENCHMARK_ORACLE_SUBMISSION_RECORD", result.stderr)
            self.assertFalse(log.exists())

    def test_reused_oracle_requires_lineage_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repository"
            initialize_submission_repository(repository)
            clip_parquet = root / "clips.parquet"
            clip_parquet.write_bytes(b"fixture")
            environment, log = submission_environment(
                repository=repository,
                root=root,
                clip_parquet=clip_parquet,
                candidate_dir=root / "candidates",
                oracle_dir=root / "oracle",
                run_dir=root / "run",
                extra={
                    "BENCHMARK_ORACLE_JOB_ID": "777",
                    "BENCHMARK_ORACLE_SUBMISSION_RECORD": str(root / "submission.json"),
                },
            )
            result = subprocess.run(
                [str(repository / "slurm" / "submit_world_benchmark.sh")],
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("BENCHMARK_ORACLE_LINEAGE_RECORD", result.stderr)
            self.assertFalse(log.exists())

    def test_reused_oracle_rejects_missing_submission_record_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repository"
            initialize_submission_repository(repository)
            clip_parquet = root / "clips.parquet"
            clip_parquet.write_bytes(b"fixture")
            environment, log = submission_environment(
                repository=repository,
                root=root,
                clip_parquet=clip_parquet,
                candidate_dir=root / "candidates",
                oracle_dir=root / "oracle",
                run_dir=root / "run",
                extra={
                    "BENCHMARK_ORACLE_JOB_ID": "777",
                    "BENCHMARK_ORACLE_SUBMISSION_RECORD": str(root / "missing.json"),
                    "BENCHMARK_ORACLE_LINEAGE_RECORD": str(root / "unused-lineage.json"),
                },
            )
            result = subprocess.run(
                [str(repository / "slurm" / "submit_world_benchmark.sh")],
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("oracle submission record cannot be resolved", result.stderr)
            self.assertFalse(log.exists())


if __name__ == "__main__":
    unittest.main()
