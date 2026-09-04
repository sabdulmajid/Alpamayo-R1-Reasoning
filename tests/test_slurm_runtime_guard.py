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
        },
    )


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
        "configuration": {"reused_oracle_job": False},
        "source": {"repository_commit": repository_commit},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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

    def test_reused_oracle_accepts_exact_original_submission_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repository"
            current_commit = initialize_submission_repository(repository)
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
                job_id=777,
                repository=oracle_repository,
                repository_commit=oracle_commit,
                clip_parquet=clip_parquet,
                candidate_dir=candidate_dir,
                oracle_dir=oracle_dir,
            )
            prior_sha256 = hashlib.sha256(prior_record.read_bytes()).hexdigest()
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
            calls = log.read_text(encoding="utf-8").split("CALL\n")[1:]
            self.assertEqual(len(calls), 10)
            expected_export = f"BENCHMARK_REPOSITORY_COMMIT={current_commit}"
            self.assertTrue(
                all(
                    any(expected_export in argument for argument in call.splitlines())
                    for call in calls
                )
            )

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
