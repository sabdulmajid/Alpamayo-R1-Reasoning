import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.revision_pinned_dataset import (
    pin_streaming_revision,
    revision_qualified_path,
)


REVISION = "2ae73f49ffd2b5db43b404201beb7b92889f7afc"


class FakeFilesystem:
    def __init__(self) -> None:
        self.paths: list[tuple[str, str]] = []

    @contextlib.contextmanager
    def open(self, path: str, mode: str):
        self.paths.append((path, mode))
        yield io.BytesIO(b"remote")


class FakeInterface:
    def __init__(self) -> None:
        self.repo_snapshot_info = {
            "repo_id": "owner/dataset",
            "repo_type": "dataset",
            "revision": REVISION,
        }
        self.cache_dir = None
        self.fs = FakeFilesystem()


class RevisionPinnedDatasetTest(unittest.TestCase):
    def test_command_line_entry_points_import_revision_helper(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        for script_name in ("generate_candidates.py", "lidar_world_oracle.py"):
            with self.subTest(script=script_name):
                completed = subprocess.run(
                    [sys.executable, str(project_root / "src" / script_name), "--help"],
                    cwd=project_root,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_revision_qualified_dataset_path(self) -> None:
        self.assertEqual(
            revision_qualified_path("owner/data", "dataset", REVISION, "x/y.zip"),
            f"datasets/owner/data@{REVISION}/x/y.zip",
        )

    def test_stream_path_contains_exact_revision(self) -> None:
        interface = FakeInterface()
        with mock.patch(
            "src.revision_pinned_dataset.try_to_load_from_cache", return_value=None
        ):
            pin_streaming_revision(interface, REVISION)
            with interface.open_file("feature/chunk.zip", maybe_stream=True) as handle:
                self.assertEqual(handle.read(), b"remote")
        self.assertEqual(
            interface.fs.paths,
            [(f"datasets/owner/dataset@{REVISION}/feature/chunk.zip", "rb")],
        )

    def test_cached_revision_file_is_used_without_streaming(self) -> None:
        interface = FakeInterface()
        with tempfile.TemporaryDirectory() as temporary:
            cached = Path(temporary) / "cached.zip"
            cached.write_bytes(b"cached")
            with mock.patch(
                "src.revision_pinned_dataset.try_to_load_from_cache",
                return_value=str(cached),
            ):
                pin_streaming_revision(interface, REVISION)
                with interface.open_file("feature/chunk.zip") as handle:
                    self.assertEqual(handle.read(), b"cached")
        self.assertEqual(interface.fs.paths, [])

    def test_symbolic_revision_is_rejected(self) -> None:
        interface = FakeInterface()
        interface.repo_snapshot_info["revision"] = "main"
        with self.assertRaisesRegex(ValueError, "full lowercase commit SHA"):
            pin_streaming_revision(interface, "main")

    def test_expected_revision_must_match_interface(self) -> None:
        interface = FakeInterface()
        with self.assertRaisesRegex(ValueError, "differs"):
            pin_streaming_revision(interface, "0" * 40)


if __name__ == "__main__":
    unittest.main()
