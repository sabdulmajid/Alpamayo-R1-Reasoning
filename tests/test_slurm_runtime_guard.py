from __future__ import annotations

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
            submit_text = SUBMIT_SCRIPT.read_text(encoding="utf-8")
            stage_text = STAGE_SCRIPT.read_text(encoding="utf-8")
            commit = initialize_repository(
                repository,
                {
                    "slurm/run_benchmark_stage.sh": stage_text,
                    "slurm/submit_world_benchmark.sh": submit_text,
                },
            )
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
            log = root / "sbatch.log"
            clip_parquet = root / "clips.parquet"
            clip_parquet.write_bytes(b"fixture")
            candidate_dir = root / "candidates"
            oracle_dir = root / "oracle"
            run_dir = root / "run"
            environment = os.environ.copy()
            environment.update(
                {
                    "BENCHMARK_CANDIDATE_DIR": str(candidate_dir),
                    "BENCHMARK_CANDIDATE_JOB_ID": "12345",
                    "BENCHMARK_CLIP_PARQUET": str(clip_parquet),
                    "BENCHMARK_ORACLE_DIR": str(oracle_dir),
                    "BENCHMARK_PYTHON": sys.executable,
                    "BENCHMARK_REPO_ROOT": str(repository),
                    "BENCHMARK_RUN_DIR": str(run_dir),
                    "FAKE_SBATCH_COUNTER": str(counter),
                    "FAKE_SBATCH_LOG": str(log),
                    "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
                }
            )
            run_checked(
                repository / "slurm" / "submit_world_benchmark.sh",
                env=environment,
            )

            submission = json.loads((run_dir / "submission.json").read_text(encoding="utf-8"))
            self.assertEqual(submission["source"]["repository_commit"], commit)
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


if __name__ == "__main__":
    unittest.main()
