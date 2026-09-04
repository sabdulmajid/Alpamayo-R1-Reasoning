import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.world_model.benchmark_prep import verify_evaluation_partition
from src.world_model.data import dataset_provenance, load_manifest
from src.world_model.protocol import checkpoint_training_configuration
from src.world_model.runtime import canonical_fingerprint, sha256_file


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


def write_protocol(root: Path, manifests: dict[str, Path]) -> Path:
    manifest_records = {}
    for name, path in manifests.items():
        provenance = dataset_provenance(load_manifest(path))
        manifest_records[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "clips": int(provenance["artifact_count"]),
            "chunks": len(provenance["chunk_ids"]),
            "dataset_fingerprint": provenance["dataset_fingerprint"],
        }
    protocol = {
        "schema_version": 1,
        "status": "frozen_before_test_evaluation",
        "source_data": {},
        "split": {"unit": "chunk_id", "manifests": manifest_records},
        "training": {
            "seeds": [2026, 2027],
            "epochs": 2,
            "batch_size": 1,
            "workers": 0,
            "base_channels": 2,
            "automatic_mixed_precision": False,
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "checkpoint_selection": "minimum_best_validation_loss",
            "test_metrics_used_for_selection": False,
        },
        "forecast_evaluation": {},
        "trajectory_selection": {},
        "uncertainty": {},
        "acceptance_gates": {},
    }
    protocol["protocol_fingerprint"] = canonical_fingerprint(protocol)
    path = root / "protocol.json"
    path.write_text(json.dumps(protocol, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_evaluation_binding(
    root: Path, checkpoint_path: Path, test_manifest: Path, protocol_path: Path
) -> tuple[Path, Path]:
    checkpoint_path = checkpoint_path.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    signature = checkpoint["resume_signature"]
    if signature["protocol"]["path"] != str(protocol_path.resolve()):
        raise AssertionError("Checkpoint is not bound to the fixture protocol")
    provenance = checkpoint["data_provenance"]
    selected_run = {
        "label": "seed-selected",
        "run_directory": str(checkpoint_path.parent),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "epoch": int(checkpoint["epoch"]),
        "seed": int(checkpoint["seed"]),
        "best_validation_loss": float(checkpoint["best_val_loss"]),
        "train_manifest_sha256": signature["train_manifest_sha256"],
        "validation_manifest_sha256": signature["val_manifest_sha256"],
        "train_dataset_fingerprint": provenance["train"]["dataset_fingerprint"],
        "validation_dataset_fingerprint": provenance["val"]["dataset_fingerprint"],
        "model_config": checkpoint["model_config"],
        "data_schema": checkpoint["data_schema"],
        "training_configuration": checkpoint_training_configuration(checkpoint),
        "protocol": signature["protocol"],
        "comparison_signature": {
            key: value for key, value in signature.items() if key != "seed"
        },
    }
    completion_path = checkpoint_path.parent / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    selected_run["training_completion"] = {
        "path": str(completion_path.resolve()),
        "sha256": sha256_file(completion_path),
        "completion_fingerprint": completion["completion_fingerprint"],
        "latest_checkpoint": completion["latest_checkpoint"],
        "latest_checkpoint_sha256": completion["latest_checkpoint_sha256"],
    }
    other_checkpoint = (root / "other-best.pt").resolve()
    other_checkpoint.write_bytes(b"other checkpoint")
    other_latest = (root / "other-latest.pt").resolve()
    other_latest.write_bytes(b"other latest checkpoint")
    other_run = {
        **selected_run,
        "label": "seed-other",
        "checkpoint": str(other_checkpoint),
        "checkpoint_sha256": sha256_file(other_checkpoint),
        "seed": int(checkpoint["seed"]) + 1,
        "best_validation_loss": float(checkpoint["best_val_loss"]) + 1.0,
    }
    other_completion = {
        "schema_version": 1,
        "status": "completed",
        "seed": other_run["seed"],
        "epochs": other_run["training_configuration"]["epochs"],
        "final_epoch": other_run["training_configuration"]["epochs"] - 1,
        "protocol": signature["protocol"],
        "latest_checkpoint": str(other_latest),
        "latest_checkpoint_sha256": sha256_file(other_latest),
        "best_checkpoint": str(other_checkpoint),
        "best_checkpoint_sha256": sha256_file(other_checkpoint),
    }
    other_completion["completion_fingerprint"] = canonical_fingerprint(
        other_completion
    )
    other_completion_path = root / "other-completion.json"
    other_completion_path.write_text(
        json.dumps(other_completion, sort_keys=True) + "\n", encoding="utf-8"
    )
    other_run["training_completion"] = {
        "path": str(other_completion_path.resolve()),
        "sha256": sha256_file(other_completion_path),
        "completion_fingerprint": other_completion["completion_fingerprint"],
        "latest_checkpoint": str(other_latest),
        "latest_checkpoint_sha256": sha256_file(other_latest),
    }
    selection = {
        "schema_version": 2,
        "selection_policy": "minimum_best_validation_loss",
        "test_metrics_used": False,
        "protocol": signature["protocol"],
        "selected_label": selected_run["label"],
        "selected_seed": selected_run["seed"],
        "selected_checkpoint": selected_run["checkpoint"],
        "selected_checkpoint_sha256": selected_run["checkpoint_sha256"],
        "selected_epoch": selected_run["epoch"],
        "selected_validation_loss": selected_run["best_validation_loss"],
        "runs": [selected_run, other_run],
    }
    selection["selection_fingerprint"] = canonical_fingerprint(selection)
    selection_path = root / "checkpoint_selection.json"
    selection_path.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit = verify_evaluation_partition(selection_path, test_manifest)
    audit["selection_fingerprint"] = selection["selection_fingerprint"]
    audit_path = root / "evaluation_partition.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return selection_path, audit_path


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

            protocol = write_protocol(root, manifests)

            output_dir = root / "run"
            common_train = (
                "-m",
                "src.world_model.train",
                "--protocol",
                str(protocol),
                "--train-manifest",
                str(manifests["train"]),
                "--val-manifest",
                str(manifests["val"]),
                "--output-dir",
                str(output_dir),
                "--epochs",
                "2",
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
                run_module(*overlapping_train)
            self.assertIn("share source chunks", split_error.exception.stderr)

            mismatched_train = list(common_train)
            base_index = mismatched_train.index("--base-channels") + 1
            mismatched_train[base_index] = "3"
            output_index = mismatched_train.index("--output-dir") + 1
            mismatched_train[output_index] = str(root / "invalid-protocol-run")
            with self.assertRaises(subprocess.CalledProcessError) as protocol_error:
                run_module(*mismatched_train)
            self.assertIn("Frozen protocol training", protocol_error.exception.stderr)

            run_module(*common_train)
            first = torch.load(
                output_dir / "latest.pt", map_location="cpu", weights_only=False
            )
            self.assertEqual(first["epoch"], 1)
            self.assertIn("scaler_state", first)
            self.assertIn("rng_state", first)
            self.assertEqual(first["checkpoint_schema_version"], 2)
            self.assertEqual(first["data_provenance"]["train"]["artifact_count"], 1)
            self.assertEqual(
                len(first["data_provenance"]["train"]["dataset_fingerprint"]), 64
            )

            run_module(*common_train, "--resume", "auto")
            resumed = torch.load(
                output_dir / "latest.pt", map_location="cpu", weights_only=False
            )
            self.assertEqual(resumed["epoch"], 1)
            self.assertEqual(len(resumed["history"]), 2)

            prediction_dir = root / "predictions"
            learned_metrics = root / "learned.json"
            selection_record, evaluation_audit = write_evaluation_binding(
                root, output_dir / "best.pt", manifests["test"], protocol
            )
            learned_binding_arguments = (
                "--protocol",
                str(protocol),
                "--selection-record",
                str(selection_record),
                "--evaluation-audit",
                str(evaluation_audit),
            )
            with self.assertRaises(subprocess.CalledProcessError) as binding_error:
                run_module(
                    "-m",
                    "src.world_model.evaluate",
                    "--manifest",
                    str(manifests["test"]),
                    "--method",
                    "learned",
                    "--checkpoint",
                    str(output_dir / "best.pt"),
                    "--output",
                    str(root / "unbound.json"),
                )
            self.assertIn(
                "requires --protocol",
                binding_error.exception.stderr,
            )
            run_module(
                "-m",
                "src.world_model.evaluate",
                "--manifest",
                str(manifests["test"]),
                "--method",
                "learned",
                "--checkpoint",
                str(output_dir / "best.pt"),
                *learned_binding_arguments,
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
                prediction_run = json.loads(str(archive["prediction_run_json"]))
                binding = prediction_run["evaluation_binding"]
                self.assertEqual(binding["protocol"]["path"], str(protocol.resolve()))
                self.assertEqual(
                    binding["checkpoint_selection"]["sha256"],
                    sha256_file(selection_record),
                )
                self.assertEqual(
                    binding["evaluation_audit"]["sha256"],
                    sha256_file(evaluation_audit),
                )

            audit_bytes = evaluation_audit.read_bytes()
            audit_payload = json.loads(audit_bytes)
            evaluation_audit.write_text(
                json.dumps(audit_payload, sort_keys=True), encoding="utf-8"
            )
            with self.assertRaises(subprocess.CalledProcessError) as record_error:
                run_module(
                    "-m",
                    "src.world_model.evaluate",
                    "--manifest",
                    str(manifests["test"]),
                    "--method",
                    "learned",
                    "--checkpoint",
                    str(output_dir / "best.pt"),
                    *learned_binding_arguments,
                    "--output",
                    str(root / "changed-record.json"),
                    "--prediction-dir",
                    str(prediction_dir),
                    "--batch-size",
                    "1",
                    "--workers",
                    "0",
                    "--device",
                    "cpu",
                )
            self.assertIn("different prediction_run", record_error.exception.stderr)
            evaluation_audit.write_bytes(audit_bytes)

            with np.load(prediction, allow_pickle=False) as archive:
                prediction_arrays = {name: archive[name] for name in archive.files}
            prediction_arrays["occupancy_prob"] = prediction_arrays[
                "occupancy_prob"
            ].copy()
            original_probability = float(prediction_arrays["occupancy_prob"].flat[0])
            prediction_arrays["occupancy_prob"].flat[0] = (
                1.0 if original_probability < 0.5 else 0.0
            )
            np.savez_compressed(prediction, **prediction_arrays)
            with self.assertRaises(subprocess.CalledProcessError) as stale_error:
                run_module(
                    "-m",
                    "src.world_model.evaluate",
                    "--manifest",
                    str(manifests["test"]),
                    "--method",
                    "learned",
                    "--checkpoint",
                    str(output_dir / "best.pt"),
                    *learned_binding_arguments,
                    "--output",
                    str(root / "stale.json"),
                    "--prediction-dir",
                    str(prediction_dir),
                    "--batch-size",
                    "1",
                    "--workers",
                    "0",
                    "--device",
                    "cpu",
                )
            self.assertIn(
                "different occupancy_prob values", stale_error.exception.stderr
            )
            run_module(
                "-m",
                "src.world_model.evaluate",
                "--manifest",
                str(manifests["test"]),
                "--method",
                "learned",
                "--checkpoint",
                str(output_dir / "best.pt"),
                *learned_binding_arguments,
                "--output",
                str(learned_metrics),
                "--prediction-dir",
                str(prediction_dir),
                "--overwrite-predictions",
                "--batch-size",
                "1",
                "--workers",
                "0",
                "--device",
                "cpu",
            )

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
            self.assertIsNone(baseline["evaluation_binding"]["checkpoint"])
            self.assertIsNone(baseline["evaluation_binding"]["protocol"])
            self.assertIsNone(baseline["evaluation_binding"]["checkpoint_selection"])
            self.assertIsNone(baseline["evaluation_binding"]["evaluation_audit"])

            reranked_path = root / "reranked.jsonl"
            run_module(
                "-m",
                "src.world_model.rerank",
                "--manifest",
                str(manifests["test"]),
                "--method",
                "learned",
                "--prediction-dir",
                str(prediction_dir),
                *learned_binding_arguments,
                "--output",
                str(reranked_path),
            )
            reranked = json.loads(reranked_path.read_text(encoding="utf-8"))
            self.assertEqual(reranked["selection_policy"], "predicted_occupancy_v1")
            self.assertNotIn("oracle_selected_index", reranked)
            self.assertEqual(len(reranked["prediction_sha256"]), 64)
            self.assertEqual(len(reranked["candidate_sha256"]), 64)
            authenticated_run = reranked["prediction_run_provenance"]
            self.assertEqual(authenticated_run["method"], "learned")
            self.assertEqual(
                authenticated_run["checkpoint_selection"]["sha256"],
                sha256_file(selection_record),
            )
            self.assertEqual(
                authenticated_run["evaluation_audit"]["sha256"],
                sha256_file(evaluation_audit),
            )
            self.assertEqual(
                authenticated_run["partition_isolation"],
                {
                    "train_evaluation_chunk_overlap": 0,
                    "validation_evaluation_chunk_overlap": 0,
                },
            )

            with self.assertRaises(subprocess.CalledProcessError) as overlap_error:
                run_module(
                    "-m",
                    "src.world_model.evaluate",
                    "--manifest",
                    str(manifests["train"]),
                    "--method",
                    "learned",
                    "--checkpoint",
                    str(output_dir / "best.pt"),
                    *learned_binding_arguments,
                    "--output",
                    str(root / "invalid-overlap.json"),
                    "--workers",
                    "0",
                    "--device",
                    "cpu",
                )
            self.assertIn("share source chunks", overlap_error.exception.stderr)

            with self.assertRaises(subprocess.CalledProcessError) as val_overlap_error:
                run_module(
                    "-m",
                    "src.world_model.evaluate",
                    "--manifest",
                    str(manifests["val"]),
                    "--method",
                    "learned",
                    "--checkpoint",
                    str(output_dir / "best.pt"),
                    *learned_binding_arguments,
                    "--output",
                    str(root / "invalid-val-overlap.json"),
                    "--workers",
                    "0",
                    "--device",
                    "cpu",
                )
            self.assertIn("share source chunks", val_overlap_error.exception.stderr)

            validation_metrics = root / "validation.json"
            run_module(
                "-m",
                "src.world_model.evaluate",
                "--manifest",
                str(manifests["val"]),
                "--method",
                "learned",
                "--data-role",
                "validation",
                "--checkpoint",
                str(output_dir / "latest.pt"),
                "--output",
                str(validation_metrics),
                "--workers",
                "0",
                "--device",
                "cpu",
            )
            validation_result = json.loads(
                validation_metrics.read_text(encoding="utf-8")
            )
            self.assertEqual(validation_result["data_role"], "validation")

            train_artifact = root / "train.npz"
            with np.load(train_artifact, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            arrays["past_occupancy"] = arrays["past_occupancy"].copy()
            arrays["past_occupancy"][0, 0, 0] = 1
            np.savez_compressed(train_artifact, **arrays)
            with self.assertRaises(subprocess.CalledProcessError) as mutation_error:
                run_module(*common_train, "--resume", "auto")
            self.assertIn("dataset fingerprint differs", mutation_error.exception.stderr)


if __name__ == "__main__":
    unittest.main()
