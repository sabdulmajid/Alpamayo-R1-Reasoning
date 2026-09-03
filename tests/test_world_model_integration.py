import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


REPOSITORY = Path(__file__).resolve().parents[1]


def run_module(*arguments: str) -> None:
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    subprocess.run(
        [sys.executable, *arguments],
        cwd=REPOSITORY,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def write_example(path: Path, clip_id: str, t0_us: int, occupied_y: int) -> None:
    past = np.zeros((3, 12, 12), dtype=np.uint8)
    past[:, 6, 2:5] = 1
    future = np.zeros((2, 12, 12), dtype=np.uint8)
    future[:, occupied_y, 3:6] = 1
    times = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    candidate_zero = np.stack(
        (np.array([0.0, 1.0, 2.0]), np.zeros(3), np.zeros(3)), axis=1
    )
    candidate_one = np.stack(
        (np.array([0.0, 1.0, 2.0]), np.full(3, 3.0), np.zeros(3)), axis=1
    )
    np.savez_compressed(
        path,
        schema_version=np.int16(4),
        clip_id=np.asarray(clip_id),
        t0_us=np.int64(t0_us),
        past_occupancy=past,
        past_observed=np.ones_like(past),
        occupancy=future,
        observed=np.ones_like(future),
        history_offsets_s=np.array([-1.0, -0.5, 0.0], dtype=np.float32),
        horizons_s=np.array([1.0, 2.0], dtype=np.float32),
        candidate_xyz=np.stack((candidate_zero, candidate_one)).astype(np.float32),
        candidate_yaw=np.zeros((2, 3), dtype=np.float32),
        candidate_times_s=times,
        vehicle_dimensions_m=np.array([2.0, 1.0, 1.5, 0.0], dtype=np.float32),
        oracle_safest_idx=np.int32(1),
        frame=np.asarray("ego_at_t0"),
        bev_x_min_m=np.float32(-2.0),
        bev_x_max_m=np.float32(10.0),
        bev_y_min_m=np.float32(-6.0),
        bev_y_max_m=np.float32(6.0),
        bev_resolution_m=np.float32(1.0),
    )


class SyntheticPipelineTest(unittest.TestCase):
    def test_train_resume_evaluate_and_rerank(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests: dict[str, Path] = {}
            for index, split in enumerate(("train", "val", "test"), start=1):
                artifact = root / f"{split}.npz"
                clip_id = f"clip-{split}"
                t0_us = 100 + index
                write_example(artifact, clip_id, t0_us, occupied_y=6 + index % 2)
                manifest = root / f"{split}.jsonl"
                manifest.write_text(
                    json.dumps(
                        {
                            "clip_id": clip_id,
                            "t0_us": t0_us,
                            "chunk_id": f"chunk-{split}",
                            "artifact_path": artifact.name,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                manifests[split] = manifest

            output_dir = root / "run"
            common_train = (
                "-m",
                "src.world_model.train",
                "--train-manifest",
                str(manifests["train"]),
                "--val-manifest",
                str(manifests["val"]),
                "--output-dir",
                str(output_dir),
                "--batch-size",
                "1",
                "--workers",
                "0",
                "--base-channels",
                "2",
                "--device",
                "cpu",
            )
            overlapping_train = list(common_train)
            validation_index = overlapping_train.index("--val-manifest") + 1
            overlapping_train[validation_index] = str(manifests["train"])
            output_index = overlapping_train.index("--output-dir") + 1
            overlapping_train[output_index] = str(root / "invalid-overlap-run")
            with self.assertRaises(subprocess.CalledProcessError) as split_error:
                run_module(*overlapping_train, "--epochs", "1")
            self.assertIn("share source chunks", split_error.exception.stderr)

            run_module(*common_train, "--epochs", "1")
            first = torch.load(output_dir / "latest.pt", map_location="cpu", weights_only=False)
            self.assertEqual(first["epoch"], 0)
            self.assertIn("scaler_state", first)
            self.assertIn("rng_state", first)
            self.assertEqual(first["checkpoint_schema_version"], 2)
            self.assertEqual(first["data_provenance"]["train"]["artifact_count"], 1)
            self.assertEqual(
                len(first["data_provenance"]["train"]["dataset_fingerprint"]), 64
            )

            run_module(*common_train, "--epochs", "2", "--resume", "auto")
            resumed = torch.load(
                output_dir / "latest.pt", map_location="cpu", weights_only=False
            )
            self.assertEqual(resumed["epoch"], 1)
            self.assertEqual(len(resumed["history"]), 2)

            prediction_dir = root / "predictions"
            learned_metrics = root / "learned.json"
            run_module(
                "-m",
                "src.world_model.evaluate",
                "--manifest",
                str(manifests["test"]),
                "--method",
                "learned",
                "--checkpoint",
                str(output_dir / "latest.pt"),
                "--output",
                str(learned_metrics),
                "--prediction-dir",
                str(prediction_dir),
                "--batch-size",
                "1",
                "--workers",
                "0",
                "--device",
                "cpu",
            )
            learned = json.loads(learned_metrics.read_text(encoding="utf-8"))
            self.assertEqual(learned["method"], "learned")
            self.assertEqual(len(learned["checkpoint_sha256"]), 64)
            prediction = next(prediction_dir.glob("*.prediction.npz"))
            with np.load(prediction, allow_pickle=False) as archive:
                self.assertEqual(str(archive["producer_method"]), "learned")
                self.assertEqual(int(archive["schema_version"]), 2)
                self.assertEqual(len(str(archive["prediction_run_fingerprint"])), 64)

            persistence_metrics = root / "persistence.json"
            run_module(
                "-m",
                "src.world_model.evaluate",
                "--manifest",
                str(manifests["test"]),
                "--method",
                "persistence",
                "--output",
                str(persistence_metrics),
                "--batch-size",
                "1",
                "--workers",
                "0",
                "--device",
                "cpu",
            )
            baseline = json.loads(persistence_metrics.read_text(encoding="utf-8"))
            self.assertEqual(baseline["method"], "persistence")
            self.assertIsNone(baseline["checkpoint_sha256"])

            reranked_path = root / "reranked.jsonl"
            run_module(
                "-m",
                "src.world_model.rerank",
                "--manifest",
                str(manifests["test"]),
                "--prediction-dir",
                str(prediction_dir),
                "--output",
                str(reranked_path),
            )
            reranked = json.loads(reranked_path.read_text(encoding="utf-8"))
            self.assertEqual(reranked["selection_policy"], "predicted_occupancy_v1")
            self.assertNotIn("oracle_selected_index", reranked)
            self.assertEqual(len(reranked["prediction_sha256"]), 64)
            self.assertEqual(len(reranked["candidate_sha256"]), 64)

            with self.assertRaises(subprocess.CalledProcessError) as overlap_error:
                run_module(
                    "-m",
                    "src.world_model.evaluate",
                    "--manifest",
                    str(manifests["train"]),
                    "--method",
                    "learned",
                    "--checkpoint",
                    str(output_dir / "latest.pt"),
                    "--output",
                    str(root / "invalid-overlap.json"),
                    "--workers",
                    "0",
                    "--device",
                    "cpu",
                )
            self.assertIn("share source chunks", overlap_error.exception.stderr)

            train_artifact = root / "train.npz"
            with np.load(train_artifact, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            arrays["past_occupancy"] = arrays["past_occupancy"].copy()
            arrays["past_occupancy"][0, 0, 0] = 1
            np.savez_compressed(train_artifact, **arrays)
            with self.assertRaises(subprocess.CalledProcessError) as mutation_error:
                run_module(*common_train, "--epochs", "3", "--resume", "auto")
            self.assertIn("provenance does not match", mutation_error.exception.stderr)


if __name__ == "__main__":
    unittest.main()
