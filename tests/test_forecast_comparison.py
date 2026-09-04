import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.world_model.compare_forecasts import compare_forecasts
from src.world_model.runtime import canonical_fingerprint, prediction_filename, sha256_file


HORIZONS = np.array([0.5, 1.0, 2.0], dtype=np.float32)
ORIGIN = np.array([-1.0, -1.0], dtype=np.float32)
REPOSITORY = Path(__file__).resolve().parents[1]
TEST_AP_BINS = 10
TEST_BOOTSTRAP_REPLICATES = 20
TEST_BOOTSTRAP_SEED = 19


def write_oracle(path: Path, clip_id: str, t0_us: int, positive: tuple[int, int]) -> None:
    past = np.zeros((2, 2, 2), dtype=np.uint8)
    future = np.zeros((3, 2, 2), dtype=np.uint8)
    future[:, positive[0], positive[1]] = 1
    np.savez_compressed(
        path,
        schema_version=np.int16(4),
        clip_id=np.asarray(clip_id),
        t0_us=np.int64(t0_us),
        past_occupancy=past,
        past_observed=np.ones_like(past),
        occupancy=future,
        observed=np.ones_like(future),
        history_offsets_s=np.array([-0.5, 0.0], dtype=np.float32),
        horizons_s=HORIZONS,
        resolution_m=np.float32(1.0),
        grid_origin_xy_m=ORIGIN,
        coordinate_frame=np.asarray("ego_at_t0"),
    )


def write_prediction(
    directory: Path,
    oracle_path: Path,
    clip_id: str,
    t0_us: int,
    *,
    method: str,
    probability: np.ndarray,
    prediction_run: dict,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / prediction_filename(clip_id, t0_us)
    np.savez_compressed(
        path,
        schema_version=np.int16(2),
        clip_id=np.asarray(clip_id),
        t0_us=np.int64(t0_us),
        occupancy_prob=probability.astype(np.float32),
        horizons_s=HORIZONS,
        resolution_m=np.float32(1.0),
        grid_origin_xy_m=ORIGIN,
        coordinate_frame=np.asarray("ego_at_t0"),
        source_artifact_sha256=np.asarray(sha256_file(oracle_path)),
        producer_method=np.asarray(method),
        checkpoint_sha256=np.asarray(prediction_run["checkpoint_sha256"]),
        prediction_run_json=np.asarray(
            json.dumps(
                prediction_run, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        ),
        prediction_run_fingerprint=np.asarray(canonical_fingerprint(prediction_run)),
    )
    return path


def replace_array(path: Path, name: str, value: np.ndarray) -> None:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays[name] = value
    np.savez_compressed(path, **arrays)


def remove_array(path: Path, name: str) -> None:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files if key != name}
    np.savez_compressed(path, **arrays)


def replace_prediction_run(path: Path, mutate) -> None:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    run = json.loads(str(arrays["prediction_run_json"]))
    mutate(run)
    binding = run["evaluation_binding"]
    binding.pop("binding_fingerprint", None)
    binding["binding_fingerprint"] = canonical_fingerprint(binding)
    arrays["prediction_run_json"] = np.asarray(
        json.dumps(run, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    arrays["prediction_run_fingerprint"] = np.asarray(canonical_fingerprint(run))
    np.savez_compressed(path, **arrays)


def write_binding_records(
    root: Path, protocol: Path, manifest: Path, *, clips: int, chunks: int,
    dataset_fingerprint: str
) -> tuple[Path, Path, dict, dict]:
    protocol_payload = json.loads(protocol.read_text(encoding="utf-8"))
    protocol_provenance = {
        "path": str(protocol.resolve()),
        "sha256": sha256_file(protocol),
        "protocol_fingerprint": protocol_payload["protocol_fingerprint"],
    }
    training_configuration = {
        "epochs": 1,
        "batch_size": 1,
        "workers": 0,
        "base_channels": 2,
        "automatic_mixed_precision": False,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
    }
    resume_signature = {
        "seed": 2026,
        "train_manifest_sha256": "1" * 64,
        "val_manifest_sha256": "2" * 64,
        "batch_size": 1,
        "base_channels": 2,
        "amp": False,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "protocol": protocol_provenance,
    }
    checkpoint = {
        "epoch": 0,
        "seed": 2026,
        "best_val_loss": 0.1,
        "resume_signature": resume_signature,
        "run_config": {
            "epochs": 1,
            "batch_size": 1,
            "workers": 0,
            "base_channels": 2,
            "amp": False,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
        },
        "model_config": {"base_channels": 2},
        "data_schema": {},
        "data_provenance": {
            "train": {"dataset_fingerprint": "3" * 64},
            "val": {"dataset_fingerprint": "4" * 64},
        },
    }
    checkpoint_path = (root / "selected.pt").resolve()
    torch.save(checkpoint, checkpoint_path)
    other_checkpoint = (root / "other.pt").resolve()
    other_checkpoint.write_bytes(b"other checkpoint")

    def completion(path: Path, seed: int, name: str) -> tuple[Path, dict]:
        record = {
            "schema_version": 1,
            "status": "completed",
            "seed": seed,
            "epochs": 1,
            "final_epoch": 0,
            "protocol": protocol_provenance,
            "latest_checkpoint": str(path),
            "latest_checkpoint_sha256": sha256_file(path),
            "best_checkpoint": str(path),
            "best_checkpoint_sha256": sha256_file(path),
        }
        record["completion_fingerprint"] = canonical_fingerprint(record)
        record_path = root / f"{name}-completion.json"
        record_path.write_text(
            json.dumps(record, sort_keys=True) + "\n", encoding="utf-8"
        )
        return record_path.resolve(), record

    selected_completion_path, selected_completion = completion(
        checkpoint_path, 2026, "selected"
    )
    other_completion_path, other_completion = completion(
        other_checkpoint, 2027, "other"
    )
    common_run = {
        "epoch": 0,
        "train_manifest_sha256": resume_signature["train_manifest_sha256"],
        "validation_manifest_sha256": resume_signature["val_manifest_sha256"],
        "train_dataset_fingerprint": "3" * 64,
        "validation_dataset_fingerprint": "4" * 64,
        "model_config": checkpoint["model_config"],
        "data_schema": checkpoint["data_schema"],
        "training_configuration": training_configuration,
        "protocol": protocol_provenance,
        "comparison_signature": {
            key: value for key, value in resume_signature.items() if key != "seed"
        },
    }
    selected_run = {
        "label": "selected",
        "seed": 2026,
        "best_validation_loss": 0.1,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_completion": {
            "path": str(selected_completion_path),
            "sha256": sha256_file(selected_completion_path),
            "completion_fingerprint": selected_completion["completion_fingerprint"],
            "latest_checkpoint": str(checkpoint_path),
            "latest_checkpoint_sha256": sha256_file(checkpoint_path),
        },
        **common_run,
    }
    other_run = {
        "label": "other",
        "seed": 2027,
        "best_validation_loss": 0.2,
        "checkpoint": str(other_checkpoint),
        "checkpoint_sha256": sha256_file(other_checkpoint),
        "training_completion": {
            "path": str(other_completion_path),
            "sha256": sha256_file(other_completion_path),
            "completion_fingerprint": other_completion["completion_fingerprint"],
            "latest_checkpoint": str(other_checkpoint),
            "latest_checkpoint_sha256": sha256_file(other_checkpoint),
        },
        **common_run,
    }
    selection = {
        "schema_version": 2,
        "selection_policy": "minimum_best_validation_loss",
        "test_metrics_used": False,
        "protocol": protocol_provenance,
        "selected_label": "selected",
        "selected_seed": 2026,
        "selected_checkpoint": str(checkpoint_path),
        "selected_checkpoint_sha256": sha256_file(checkpoint_path),
        "selected_epoch": 0,
        "selected_validation_loss": 0.1,
        "runs": [selected_run, other_run],
    }
    selection["selection_fingerprint"] = canonical_fingerprint(selection)
    selection_path = (root / "checkpoint_selection.json").resolve()
    selection_path.write_text(
        json.dumps(selection, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit = {
        "selection": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "selection_fingerprint": selection["selection_fingerprint"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "test_manifest": str(manifest.resolve()),
        "test_manifest_sha256": sha256_file(manifest),
        "test_clips": clips,
        "test_chunks": chunks,
        "train_test_chunk_overlap": 0,
        "validation_test_chunk_overlap": 0,
        "protocol": protocol_provenance,
    }
    audit_path = (root / "evaluation_partition.json").resolve()
    audit_path.write_text(json.dumps(audit, sort_keys=True) + "\n", encoding="utf-8")
    record_binding = {
        "protocol": protocol_provenance,
        "checkpoint_selection": {
            "path": str(selection_path),
            "sha256": sha256_file(selection_path),
            "selection_fingerprint": selection["selection_fingerprint"],
            "selected_label": "selected",
            "selected_seed": 2026,
            "selected_epoch": 0,
            "selected_validation_loss": 0.1,
        },
        "evaluation_audit": {
            "path": str(audit_path),
            "sha256": sha256_file(audit_path),
            "record_fingerprint": canonical_fingerprint(audit),
            "selection_fingerprint": selection["selection_fingerprint"],
            "train_test_chunk_overlap": 0,
            "validation_test_chunk_overlap": 0,
        },
    }
    evaluation_manifest = {
        "path": str(manifest.resolve()),
        "sha256": sha256_file(manifest),
        "dataset_fingerprint": dataset_fingerprint,
        "clips": clips,
        "chunks": chunks,
    }
    learned_binding = {
        "evaluation_manifest": evaluation_manifest,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "epoch": 0,
            "seed": 2026,
            "train_manifest_sha256": "1" * 64,
            "validation_manifest_sha256": "2" * 64,
            "train_dataset_fingerprint": "3" * 64,
            "validation_dataset_fingerprint": "4" * 64,
        },
        **record_binding,
        "partition_isolation": {
            "train_evaluation_chunk_overlap": 0,
            "validation_evaluation_chunk_overlap": 0,
        },
    }
    learned_binding["binding_fingerprint"] = canonical_fingerprint(learned_binding)
    persistence_binding = {
        "evaluation_manifest": evaluation_manifest,
        "checkpoint": None,
        "protocol": None,
        "checkpoint_selection": None,
        "evaluation_audit": None,
        "partition_isolation": {
            "train_evaluation_chunk_overlap": None,
            "validation_evaluation_chunk_overlap": None,
        },
    }
    persistence_binding["binding_fingerprint"] = canonical_fingerprint(
        persistence_binding
    )
    common_prediction_run = {
        "data_role": "test",
        "amp": False,
        "data_schema": {},
    }
    learned_prediction_run = {
        **common_prediction_run,
        "method": "learned",
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "evaluation_binding": learned_binding,
    }
    persistence_prediction_run = {
        **common_prediction_run,
        "method": "persistence",
        "checkpoint_sha256": "",
        "evaluation_binding": persistence_binding,
    }
    return selection_path, audit_path, learned_prediction_run, persistence_prediction_run


def make_tree(root: Path) -> tuple[Path, Path, Path, list[Path], Path]:
    oracle_dir = root / "oracle"
    learned_dir = root / "learned"
    persistence_dir = root / "persistence"
    oracle_dir.mkdir()
    rows = []
    learned_paths = []
    examples = []
    for index in range(4):
        clip_id = f"clip-{index}"
        t0_us = 1000 + index
        oracle_path = oracle_dir / f"{clip_id}.npz"
        positive = (index % 2, (index // 2) % 2)
        write_oracle(oracle_path, clip_id, t0_us, positive)
        with np.load(oracle_path, allow_pickle=False) as archive:
            target = np.asarray(archive["occupancy"], dtype=np.float32)
        learned_probability = np.where(target > 0, 0.9, 0.1).astype(np.float32)
        persistence_probability = np.zeros_like(target)
        examples.append(
            (
                oracle_path,
                clip_id,
                t0_us,
                learned_probability,
                persistence_probability,
            )
        )
        rows.append(
            {
                "clip_id": clip_id,
                "t0_us": t0_us,
                "chunk_id": f"chunk-{index // 2}",
                "artifact_path": oracle_path.name,
                "artifact_sha256": sha256_file(oracle_path),
            }
        )
    manifest = oracle_dir / "test.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    dataset_index = [
        {
            "artifact_sha256": row["artifact_sha256"],
            "chunk_id": row["chunk_id"],
            "clip_id": row["clip_id"],
            "t0_us": row["t0_us"],
        }
        for row in rows
    ]
    dataset_fingerprint = canonical_fingerprint(dataset_index)
    protocol_payload = {
        "schema_version": 1,
        "status": "frozen_before_test_evaluation",
        "source_data": {},
        "split": {
            "unit": "chunk_id",
            "manifests": {
                "test": {
                    "path": str(manifest.resolve()),
                    "sha256": sha256_file(manifest),
                    "clips": len(rows),
                    "chunks": len({row["chunk_id"] for row in rows}),
                    "dataset_fingerprint": dataset_fingerprint,
                }
            },
        },
        "training": {
            "seeds": [2026, 2027],
            "epochs": 1,
            "batch_size": 1,
            "workers": 0,
            "base_channels": 2,
            "automatic_mixed_precision": False,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "checkpoint_selection": "minimum_best_validation_loss",
            "test_metrics_used_for_selection": False,
        },
        "forecast_evaluation": {
            "metrics": ["average_precision", "brier", "iou"],
            "short_horizons_s": HORIZONS.tolist(),
            "probability_threshold": 0.5,
            "average_precision_bins": TEST_AP_BINS,
            "post_test_threshold_tuning": False,
            "comparison": "learned_minus_persistence",
        },
        "trajectory_selection": {},
        "uncertainty": {
            "method": "paired_cluster_bootstrap_percentile",
            "cluster_unit": "chunk_id",
            "confidence_level": 0.95,
            "bootstrap_seed": TEST_BOOTSTRAP_SEED,
            "bootstrap_replicates": TEST_BOOTSTRAP_REPLICATES,
        },
        "acceptance_gates": {
            "forecast": {
                "average_precision": "learned_higher_than_persistence",
                "brier": "learned_lower_than_persistence",
                "iou": "learned_higher_than_persistence",
            },
            "selection": {},
        },
    }
    protocol_payload["protocol_fingerprint"] = canonical_fingerprint(protocol_payload)
    protocol = root / "protocol.json"
    protocol.write_text(json.dumps(protocol_payload, sort_keys=True) + "\n", encoding="utf-8")
    _, _, learned_run, persistence_run = write_binding_records(
        root,
        protocol,
        manifest,
        clips=len(rows),
        chunks=len({row["chunk_id"] for row in rows}),
        dataset_fingerprint=dataset_fingerprint,
    )
    for oracle_path, clip_id, t0_us, learned_probability, persistence_probability in examples:
        learned_paths.append(
            write_prediction(
                learned_dir,
                oracle_path,
                clip_id,
                t0_us,
                method="learned",
                probability=learned_probability,
                prediction_run=learned_run,
            )
        )
        write_prediction(
            persistence_dir,
            oracle_path,
            clip_id,
            t0_us,
            method="persistence",
            probability=persistence_probability,
            prediction_run=persistence_run,
        )
    return manifest, learned_dir, persistence_dir, learned_paths, protocol


def compare_fixture(
    manifest: Path,
    learned: Path,
    persistence: Path,
    protocol: Path,
    **overrides: object,
) -> dict:
    arguments = {
        "protocol": protocol,
        "manifest": manifest,
        "learned_prediction_dir": learned,
        "persistence_prediction_dir": persistence,
        "selection_record": protocol.parent / "checkpoint_selection.json",
        "evaluation_audit": protocol.parent / "evaluation_partition.json",
        "ap_bins": TEST_AP_BINS,
        "bootstrap_replicates": TEST_BOOTSTRAP_REPLICATES,
        "bootstrap_seed": TEST_BOOTSTRAP_SEED,
        **overrides,
    }
    return compare_forecasts(**arguments)


class ForecastComparisonTest(unittest.TestCase):
    def test_reports_clip_macro_metrics_and_deterministic_cluster_intervals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, learned, persistence, _, protocol = make_tree(Path(temporary))
            arguments = {
                "protocol": protocol,
                "manifest": manifest,
                "learned_prediction_dir": learned,
                "persistence_prediction_dir": persistence,
                "selection_record": protocol.parent / "checkpoint_selection.json",
                "evaluation_audit": protocol.parent / "evaluation_partition.json",
                "ap_bins": TEST_AP_BINS,
                "bootstrap_replicates": TEST_BOOTSTRAP_REPLICATES,
                "bootstrap_seed": TEST_BOOTSTRAP_SEED,
            }
            result = compare_forecasts(**arguments)
            repeated = compare_forecasts(**arguments)

            self.assertEqual(result["paired_differences"], repeated["paired_differences"])
            self.assertEqual(result["inputs"]["manifest"]["clips"], 4)
            self.assertEqual(result["inputs"]["manifest"]["chunks"], 2)
            self.assertEqual(len(result["inputs"]["manifest"]["manifest_sha256"]), 64)
            self.assertEqual(
                result["inputs"]["predictions"]["learned"]["checkpoint_sha256"],
                json.loads(
                    (protocol.parent / "checkpoint_selection.json").read_text(
                        encoding="utf-8"
                    )
                )["selected_checkpoint_sha256"],
            )
            self.assertFalse(
                result["configuration"]["average_precision"]["trapezoidal_pr_auc"]
            )
            self.assertFalse(result["configuration"]["bootstrap"]["cell_level_resampling"])

            learned_short = result["metrics"]["learned"]["short_horizon"]
            persistence_short = result["metrics"]["persistence"]["short_horizon"]
            self.assertAlmostEqual(learned_short["iou"], 1.0)
            self.assertAlmostEqual(learned_short["average_precision"], 1.0)
            self.assertAlmostEqual(learned_short["brier"], 0.01, places=6)
            self.assertAlmostEqual(persistence_short["iou"], 0.0)
            self.assertAlmostEqual(persistence_short["average_precision"], 0.25)
            self.assertAlmostEqual(persistence_short["brier"], 0.25)
            difference = result["paired_differences"]["short_horizon"]
            self.assertAlmostEqual(difference["iou"]["estimate"], 1.0)
            self.assertEqual(difference["iou"]["percentile_95_ci"], [1.0, 1.0])
            self.assertAlmostEqual(difference["positive_prevalence"]["estimate"], 0.0)
            self.assertTrue(result["acceptance_gates"]["all_passed"])
            self.assertEqual(result["protocol"]["sha256"], sha256_file(protocol))
            self.assertEqual(
                result["inputs"]["predictions"]["learned"]["authenticated_run"][
                    "partition_isolation"
                ],
                {
                    "train_evaluation_chunk_overlap": 0,
                    "validation_evaluation_chunk_overlap": 0,
                },
            )
            self.assertIsNone(
                result["inputs"]["predictions"]["persistence"]["authenticated_run"][
                    "checkpoint_selection"
                ]
            )
            self.assertFalse(
                result["configuration"]["uncertainty_scope"][
                    "training_seed_variability_included"
                ]
            )

    def test_cli_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, learned, persistence, _, protocol = make_tree(root)
            output = root / "comparison.json"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.world_model.compare_forecasts",
                    "--protocol",
                    str(protocol),
                    "--manifest",
                    str(manifest),
                    "--learned-prediction-dir",
                    str(learned),
                    "--persistence-prediction-dir",
                    str(persistence),
                    "--selection-record",
                    str(root / "checkpoint_selection.json"),
                    "--evaluation-audit",
                    str(root / "evaluation_partition.json"),
                    "--output",
                    str(output),
                    "--bootstrap-replicates",
                    str(TEST_BOOTSTRAP_REPLICATES),
                    "--bootstrap-seed",
                    str(TEST_BOOTSTRAP_SEED),
                    "--ap-bins",
                    str(TEST_AP_BINS),
                ],
                cwd=REPOSITORY,
                env={**os.environ, "OMP_NUM_THREADS": "1"},
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["comparison"], "learned_minus_persistence")
            self.assertEqual(len(payload["metrics"]["learned"]["per_horizon"]), 3)

    def test_protocol_fingerprint_and_configuration_drift_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, learned, persistence, _, protocol = make_tree(Path(temporary))
            payload = json.loads(protocol.read_text(encoding="utf-8"))
            payload["forecast_evaluation"]["probability_threshold"] = 0.4
            protocol.write_text(
                json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                compare_fixture(manifest, learned, persistence, protocol)

        with tempfile.TemporaryDirectory() as temporary:
            manifest, learned, persistence, _, protocol = make_tree(Path(temporary))
            with self.assertRaisesRegex(ValueError, "forecast_evaluation"):
                compare_fixture(
                    manifest, learned, persistence, protocol, threshold=0.4
                )
            with self.assertRaisesRegex(ValueError, "Bootstrap bootstrap_seed"):
                compare_fixture(
                    manifest,
                    learned,
                    persistence,
                    protocol,
                    bootstrap_seed=TEST_BOOTSTRAP_SEED + 1,
                )

        with tempfile.TemporaryDirectory() as temporary:
            manifest, learned, persistence, _, protocol = make_tree(Path(temporary))
            payload = json.loads(protocol.read_text(encoding="utf-8"))
            payload["uncertainty"]["unplanned_option"] = True
            payload.pop("protocol_fingerprint")
            payload["protocol_fingerprint"] = canonical_fingerprint(payload)
            protocol.write_text(
                json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "uncertainty"):
                compare_fixture(manifest, learned, persistence, protocol)

    def test_directional_gates_fail_when_learned_does_not_improve(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, learned, persistence, learned_paths, protocol = make_tree(
                Path(temporary)
            )
            for path in learned_paths:
                replace_array(
                    path, "occupancy_prob", np.zeros((3, 2, 2), dtype=np.float32)
                )
            result = compare_fixture(manifest, learned, persistence, protocol)
        self.assertFalse(result["acceptance_gates"]["all_passed"])
        self.assertFalse(
            result["acceptance_gates"]["decisions"]["average_precision"]["passed"]
        )
        self.assertFalse(result["acceptance_gates"]["decisions"]["brier"]["passed"])
        self.assertFalse(result["acceptance_gates"]["decisions"]["iou"]["passed"])

    def test_protocol_binds_exact_manifest_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, learned, persistence, _, protocol = make_tree(root)
            copied = manifest.parent / "copied-test.jsonl"
            copied.write_bytes(manifest.read_bytes())
            with self.assertRaisesRegex(ValueError, "test manifest path differs"):
                compare_fixture(copied, learned, persistence, protocol)

    def test_rejects_mixed_learned_run_or_checkpoint(self) -> None:
        for field, replacement, message in (
            ("prediction_run_fingerprint", np.asarray("d" * 64), "run fingerprint"),
            ("checkpoint_sha256", np.asarray("e" * 64), "checkpoint"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                manifest, learned, persistence, learned_paths, protocol = make_tree(
                    Path(temporary)
                )
                replace_array(learned_paths[0], field, replacement)
                with self.assertRaisesRegex(ValueError, message):
                    compare_fixture(manifest, learned, persistence, protocol)

    def test_rejects_missing_or_forged_prediction_run_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, learned, persistence, learned_paths, protocol = make_tree(
                Path(temporary)
            )
            remove_array(learned_paths[0], "prediction_run_json")
            with self.assertRaisesRegex(ValueError, "prediction_run_json"):
                compare_fixture(manifest, learned, persistence, protocol)

        with tempfile.TemporaryDirectory() as temporary:
            manifest, learned, persistence, learned_paths, protocol = make_tree(
                Path(temporary)
            )
            path = learned_paths[0]
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            forged = json.loads(str(arrays["prediction_run_json"]))
            forged["amp"] = True
            arrays["prediction_run_json"] = np.asarray(
                json.dumps(forged, sort_keys=True, separators=(",", ":"))
            )
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(ValueError, "fingerprint does not match"):
                compare_fixture(manifest, learned, persistence, protocol)

    def test_rejects_noncanonical_or_mixed_prediction_run_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, learned, persistence, learned_paths, protocol = make_tree(
                Path(temporary)
            )
            path = learned_paths[0]
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            run = json.loads(str(arrays["prediction_run_json"]))
            arrays["prediction_run_json"] = np.asarray(json.dumps(run, sort_keys=True))
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(ValueError, "not canonically encoded"):
                compare_fixture(manifest, learned, persistence, protocol)

        with tempfile.TemporaryDirectory() as temporary:
            manifest, learned, persistence, learned_paths, protocol = make_tree(
                Path(temporary)
            )
            replace_prediction_run(learned_paths[0], lambda run: run.update(amp=True))
            with self.assertRaisesRegex(ValueError, "one canonical run payload"):
                compare_fixture(manifest, learned, persistence, protocol)

    def test_rejects_wrong_checkpoint_audit_or_protocol_in_run(self) -> None:
        mutations = (
            (
                lambda run: run["evaluation_binding"]["checkpoint"].update(
                    sha256="5" * 64
                ),
                "checkpoint",
            ),
            (
                lambda run: run["evaluation_binding"]["evaluation_audit"].update(
                    sha256="6" * 64
                ),
                "audit",
            ),
            (
                lambda run: run["evaluation_binding"]["protocol"].update(
                    sha256="7" * 64
                ),
                "protocol",
            ),
        )
        for mutate, label in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                manifest, learned, persistence, learned_paths, protocol = make_tree(
                    Path(temporary)
                )
                for path in learned_paths:
                    replace_prediction_run(path, mutate)
                with self.assertRaisesRegex(ValueError, "evaluation binding"):
                    compare_fixture(manifest, learned, persistence, protocol)

    def test_rejects_source_method_geometry_and_nonfinite_probability(self) -> None:
        mutations = (
            (
                "source_artifact_sha256",
                np.asarray("f" * 64),
                "source oracle artifact SHA-256 differs",
            ),
            ("producer_method", np.asarray("persistence"), "producer_method"),
            ("horizons_s", np.array([0.5, 1.0, 3.0], dtype=np.float32), "horizons differ"),
            (
                "occupancy_prob",
                np.full((3, 2, 2), np.nan, dtype=np.float32),
                "finite values",
            ),
        )
        for field, replacement, message in mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                manifest, learned, persistence, learned_paths, protocol = make_tree(
                    Path(temporary)
                )
                replace_array(learned_paths[0], field, replacement)
                with self.assertRaisesRegex(ValueError, message):
                    compare_fixture(manifest, learned, persistence, protocol)

    def test_rejects_duplicate_or_extra_prediction_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, learned, persistence, learned_paths, protocol = make_tree(
                Path(temporary)
            )
            shutil.copyfile(learned_paths[0], learned / "duplicate.prediction.npz")
            with self.assertRaisesRegex(ValueError, "prediction file set differs"):
                compare_fixture(manifest, learned, persistence, protocol)

    def test_rejects_oracle_bytes_changed_after_freeze(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, learned, persistence, _, protocol = make_tree(Path(temporary))
            first_row = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
            oracle = manifest.parent / first_row["artifact_path"]
            with oracle.open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256 differs from the manifest"):
                compare_fixture(manifest, learned, persistence, protocol)

    def test_rejects_one_source_chunk_for_cluster_uncertainty(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, learned, persistence, _, protocol = make_tree(Path(temporary))
            rows = [json.loads(line) for line in manifest.read_text().splitlines()]
            for row in rows:
                row["chunk_id"] = "only-chunk"
            manifest.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            payload = json.loads(protocol.read_text(encoding="utf-8"))
            test_record = payload["split"]["manifests"]["test"]
            test_record["sha256"] = sha256_file(manifest)
            test_record["chunks"] = 1
            test_record["dataset_fingerprint"] = canonical_fingerprint(
                [
                    {
                        "artifact_sha256": row["artifact_sha256"],
                        "chunk_id": row["chunk_id"],
                        "clip_id": row["clip_id"],
                        "t0_us": row["t0_us"],
                    }
                    for row in rows
                ]
            )
            payload.pop("protocol_fingerprint")
            payload["protocol_fingerprint"] = canonical_fingerprint(payload)
            protocol.write_text(
                json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "at least two source chunks"):
                compare_fixture(manifest, learned, persistence, protocol)


if __name__ == "__main__":
    unittest.main()
