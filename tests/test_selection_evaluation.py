import json
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from unittest import mock

import numpy as np

from src.world_model.evaluate_selection import (
    cluster_bootstrap_interval,
    evaluate_selection,
    main,
)
from src.world_model.rerank import RerankWeights, rerank_artifacts
from src.world_model.runtime import canonical_fingerprint, sha256_file


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class SelectionFixture:
    def __init__(
        self,
        root: Path,
        *,
        collision_exposure: tuple[float, float] = (0.5, 0.1),
    ) -> None:
        self.root = root
        self.candidate_dir = root / "candidates"
        (self.candidate_dir / "records").mkdir(parents=True)
        (self.candidate_dir / "artifacts").mkdir()
        self.oracle_dir = root / "oracles"
        self.oracle_dir.mkdir()
        self.prediction_dir = root / "predictions"
        self.prediction_dir.mkdir()
        self.test_manifest = root / "test.jsonl"
        self.train_manifest = root / "train.jsonl"
        self.validation_manifest = root / "validation.jsonl"
        self.oracle_manifest = root / "all_oracles.jsonl"
        self.learned_selections = root / "learned_selections.jsonl"
        self.persistence_selections = root / "persistence_selections.jsonl"
        self.forecast_comparison = root / "forecast_comparison.json"
        self.test_rows: list[dict] = []
        self.oracle_rows: list[dict] = []
        self.rerank_rows: list[dict] = []
        self.persistence_rows: list[dict] = []
        self.collision_exposure = collision_exposure

        for index in range(4):
            self._add_clip(
                clip_id=f"clip-{index}",
                t0_us=100 + index,
                chunk_id=f"chunk-{index // 2}",
            )
        _write_jsonl(self.test_manifest, self.test_rows)
        _write_jsonl(self.oracle_manifest, self.oracle_rows)
        _write_jsonl(
            self.train_manifest,
            [
                {
                    "clip_id": "clip-train",
                    "t0_us": 1,
                    "chunk_id": "chunk-train",
                    "artifact_path": str(self.oracle_dir / "unused-train.npz"),
                }
            ],
        )
        _write_jsonl(
            self.validation_manifest,
            [
                {
                    "clip_id": "clip-validation",
                    "t0_us": 2,
                    "chunk_id": "chunk-validation",
                    "artifact_path": str(self.oracle_dir / "unused-validation.npz"),
                }
            ],
        )
        self.protocol = root / "protocol.json"
        split_rows = {
            "test": self.test_rows,
            "train": [
                json.loads(self.train_manifest.read_text(encoding="utf-8").strip())
            ],
            "val": [
                json.loads(self.validation_manifest.read_text(encoding="utf-8").strip())
            ],
        }
        split_paths = {
            "test": self.test_manifest,
            "train": self.train_manifest,
            "val": self.validation_manifest,
        }
        protocol_payload = {
            "schema_version": 1,
            "status": "frozen_before_test_evaluation",
            "source_data": {
                "merged_oracle_manifest": str(self.oracle_manifest.resolve()),
                "merged_oracle_manifest_sha256": sha256_file(self.oracle_manifest),
            },
            "split": {
                "unit": "chunk_id",
                "manifests": {
                    name: {
                        "path": str(split_paths[name].resolve()),
                        "sha256": sha256_file(split_paths[name]),
                        "clips": len(rows),
                        "chunks": len({str(row["chunk_id"]) for row in rows}),
                        "dataset_fingerprint": canonical_fingerprint(rows),
                    }
                    for name, rows in split_rows.items()
                },
            },
            "training": {},
            "forecast_evaluation": {},
            "trajectory_selection": {
                "primary_outcome": "collision_exposure",
                "baseline": "candidate_0",
                "learned_comparator": "persistence_selected",
                "reranker_weights": {
                    name: getattr(RerankWeights(), name)
                    for name in RerankWeights.__dataclass_fields__
                },
            },
            "uncertainty": {
                "method": "paired_cluster_bootstrap_percentile",
                "cluster_unit": "chunk_id",
                "confidence_level": 0.95,
                "bootstrap_seed": 17,
                "bootstrap_replicates": 200,
            },
            "acceptance_gates": {
                "forecast": {
                    "average_precision": "learned_higher_than_persistence",
                    "brier": "learned_lower_than_persistence",
                    "iou": "learned_higher_than_persistence",
                },
                "selection": {
                    "candidate_0_collision_exposure_relative_reduction_minimum": 0.15,
                    "learned_collision_exposure_no_worse_than_persistence": True,
                    "ade_degradation_m_paired_ci_95_upper_maximum": 0.2,
                    "out_of_bounds_fraction_no_higher_than_candidate_0": True,
                    "observed_fraction_no_lower_than_candidate_0": True,
                },
            },
        }
        protocol_payload["protocol_fingerprint"] = canonical_fingerprint(
            protocol_payload
        )
        self.protocol.write_text(
            json.dumps(protocol_payload, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._write_predictions_and_selections(protocol_payload)
        _write_jsonl(self.learned_selections, self.rerank_rows)
        _write_jsonl(self.persistence_selections, self.persistence_rows)
        self._write_forecast_comparison(protocol_payload)

    def _write_forecast_comparison(self, protocol_payload: dict) -> None:
        learned = {"average_precision": 0.8, "brier": 0.1, "iou": 0.6}
        persistence = {"average_precision": 0.7, "brier": 0.2, "iou": 0.5}
        differences = {
            metric: learned[metric] - persistence[metric] for metric in learned
        }
        rules = protocol_payload["acceptance_gates"]["forecast"]
        report = {
            "schema_version": 1,
            "comparison": "learned_minus_persistence",
            "protocol": {
                "path": str(self.protocol.resolve()),
                "sha256": sha256_file(self.protocol),
                "protocol_fingerprint": protocol_payload["protocol_fingerprint"],
            },
            "inputs": {
                "manifest": {
                    "manifest_path": str(self.test_manifest.resolve()),
                    "manifest_sha256": sha256_file(self.test_manifest),
                    "dataset_fingerprint": canonical_fingerprint(
                        [
                            {
                                "artifact_sha256": row["artifact_sha256"],
                                "chunk_id": row["chunk_id"],
                                "clip_id": row["clip_id"],
                                "t0_us": row["t0_us"],
                            }
                            for row in self.test_rows
                        ]
                    ),
                    "clips": len(self.test_rows),
                    "chunks": len({row["chunk_id"] for row in self.test_rows}),
                }
            },
            "metrics": {
                "learned": {"short_horizon": learned},
                "persistence": {"short_horizon": persistence},
            },
            "paired_differences": {
                "short_horizon": {
                    metric: {"estimate": difference}
                    for metric, difference in differences.items()
                }
            },
            "acceptance_gates": {
                "all_passed": True,
                "decisions": {
                    metric: {
                        "rule": rules[metric],
                        "passed": True,
                        "learned": learned[metric],
                        "persistence": persistence[metric],
                        "learned_minus_persistence": differences[metric],
                    }
                    for metric in rules
                },
            },
        }
        self.forecast_comparison.write_text(
            json.dumps(report, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _add_clip(self, clip_id: str, t0_us: int, chunk_id: str) -> None:
        candidate_path = self.candidate_dir / "artifacts" / f"{clip_id}.npz"
        gt_xyz = np.stack(
            (
                np.array([1.0, 2.0, 3.0], dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
            ),
            axis=1,
        )
        candidate_zero = gt_xyz.copy()
        candidate_zero[:, 1] = np.array([2.0, -2.0, 2.0], dtype=np.float32)
        candidate_one = gt_xyz.copy()
        candidate_one[:, 1] += 1.0
        pred_xyz = np.stack((candidate_zero, candidate_one))
        candidate_config = "a" * 64
        np.savez_compressed(
            candidate_path,
            schema_version=np.int32(1),
            clip_id=np.asarray(clip_id),
            t0_us=np.int64(t0_us),
            config_fingerprint=np.asarray(candidate_config),
            pred_xyz=pred_xyz,
            gt_xyz=gt_xyz,
        )
        candidate_sha256 = sha256_file(candidate_path)
        record_path = self.candidate_dir / "records" / f"{clip_id}.json"
        record_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "clip_id": clip_id,
                    "t0_us": t0_us,
                    "artifact_path": f"artifacts/{clip_id}.npz",
                    "artifact_sha256": candidate_sha256,
                    "config_fingerprint": candidate_config,
                    "candidates": [
                        {
                            "candidate_index": 0,
                            "metrics": {"ade_m": 2.0, "fde_m": 2.0},
                        },
                        {
                            "candidate_index": 1,
                            "metrics": {"ade_m": 1.0, "fde_m": 1.0},
                        },
                    ],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        oracle_path = self.oracle_dir / f"{clip_id}.npz"
        past = np.zeros((2, 6, 6), dtype=np.uint8)
        future = np.zeros((2, 6, 6), dtype=np.uint8)
        np.savez_compressed(
            oracle_path,
            schema_version=np.int16(4),
            clip_id=np.asarray(clip_id),
            t0_us=np.int64(t0_us),
            past_occupancy=past,
            past_observed=np.ones_like(past),
            occupancy=future,
            observed=np.ones_like(future),
            history_offsets_s=np.array([-1.0, 0.0], dtype=np.float32),
            horizons_s=np.array([1.0, 2.0], dtype=np.float32),
            bev_x_min_m=np.float32(-2.0),
            bev_x_max_m=np.float32(4.0),
            bev_y_min_m=np.float32(-3.0),
            bev_y_max_m=np.float32(3.0),
            bev_resolution_m=np.float32(1.0),
            frame=np.asarray("ego_at_t0"),
            candidate_times_s=np.array([0.0, 1.0, 2.0], dtype=np.float32),
            vehicle_dimensions_m=np.array([0.1, 0.1, 1.0, 0.0], dtype=np.float32),
            candidate_config_fingerprint=np.asarray(candidate_config),
            candidate_artifact_sha256=np.asarray(candidate_sha256),
            oracle_config_fingerprint=np.asarray("b" * 64),
            dataset_revision=np.asarray("dataset-revision"),
            candidate_xyz=pred_xyz,
            candidate_collision_exposure=np.array(
                self.collision_exposure, dtype=np.float32
            ),
            candidate_collision_cells=np.array([6, 1], dtype=np.int32),
            candidate_conflict_horizons=np.array([2, 1], dtype=np.int16),
            candidate_out_of_bounds_horizons=np.array([0, 0], dtype=np.int16),
            candidate_out_of_bounds_fraction=np.array([0.0, 0.0], dtype=np.float32),
            candidate_observed_fraction=np.array([1.0, 1.0], dtype=np.float32),
            candidate_unobserved_horizons=np.array([0, 0], dtype=np.int16),
            candidate_oracle_rank=np.array([1, 0], dtype=np.int32),
            oracle_safest_idx=np.int32(1),
        )
        oracle_sha256 = sha256_file(oracle_path)
        oracle_row = {
            "clip_id": clip_id,
            "t0_us": t0_us,
            "chunk_id": chunk_id,
            "artifact_path": str(oracle_path),
            "artifact_sha256": oracle_sha256,
            "candidate_artifact_sha256": candidate_sha256,
            "candidate_count": 2,
            "oracle_safest_idx": 1,
        }
        self.test_rows.append(dict(oracle_row))
        self.oracle_rows.append(dict(oracle_row))

    def _write_predictions_and_selections(self, protocol_payload: dict) -> None:
        checkpoint_sha256 = "c" * 64
        selection_record = {
            "schema_version": 2,
            "selection_policy": "minimum_best_validation_loss",
            "test_metrics_used": False,
            "selected_label": "seed-17",
            "selected_seed": 17,
            "selected_epoch": 3,
            "selected_validation_loss": 0.25,
            "selected_checkpoint": str((self.root / "best.pt").resolve()),
            "selected_checkpoint_sha256": checkpoint_sha256,
            "runs": [],
        }
        selection_record["selection_fingerprint"] = canonical_fingerprint(
            selection_record
        )
        selection_path = self.root / "checkpoint_selection.json"
        selection_path.write_text(
            json.dumps(selection_record, sort_keys=True) + "\n", encoding="utf-8"
        )
        audit_record = {
            "selection": str(selection_path.resolve()),
            "selection_sha256": sha256_file(selection_path),
            "selection_fingerprint": selection_record["selection_fingerprint"],
            "checkpoint": str((self.root / "best.pt").resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "test_manifest": str(self.test_manifest.resolve()),
            "test_manifest_sha256": sha256_file(self.test_manifest),
            "test_clips": len(self.test_rows),
            "test_chunks": len({row["chunk_id"] for row in self.test_rows}),
            "train_test_chunk_overlap": 0,
            "validation_test_chunk_overlap": 0,
        }
        audit_path = self.root / "evaluation_partition.json"
        audit_path.write_text(
            json.dumps(audit_record, sort_keys=True) + "\n", encoding="utf-8"
        )
        entries = [
            {
                "artifact_sha256": row["artifact_sha256"],
                "chunk_id": row["chunk_id"],
                "clip_id": row["clip_id"],
                "t0_us": row["t0_us"],
            }
            for row in sorted(
                self.test_rows,
                key=lambda row: (row["chunk_id"], row["clip_id"], row["t0_us"]),
            )
        ]
        evaluation_manifest = {
            "path": str(self.test_manifest.resolve()),
            "sha256": sha256_file(self.test_manifest),
            "dataset_fingerprint": canonical_fingerprint(entries),
            "clips": len(self.test_rows),
            "chunks": len({row["chunk_id"] for row in self.test_rows}),
        }
        protocol_provenance = {
            "path": str(self.protocol.resolve()),
            "sha256": sha256_file(self.protocol),
            "protocol_fingerprint": protocol_payload["protocol_fingerprint"],
        }
        selection_binding = {
            "path": str(selection_path.resolve()),
            "sha256": sha256_file(selection_path),
            "selection_fingerprint": selection_record["selection_fingerprint"],
            "selected_label": selection_record["selected_label"],
            "selected_seed": selection_record["selected_seed"],
            "selected_epoch": selection_record["selected_epoch"],
            "selected_validation_loss": selection_record["selected_validation_loss"],
        }
        audit_binding = {
            "path": str(audit_path.resolve()),
            "sha256": sha256_file(audit_path),
            "record_fingerprint": canonical_fingerprint(audit_record),
            "selection_fingerprint": selection_record["selection_fingerprint"],
            "train_test_chunk_overlap": 0,
            "validation_test_chunk_overlap": 0,
        }
        data_schema = {
            "input_shape": [2, 6, 6],
            "target_shape": [2, 6, 6],
            "past_horizons_s": [-1.0, 0.0],
            "horizons_s": [1.0, 2.0],
            "grid_origin_xy_m": [-2.0, -3.0],
            "resolution_m": 1.0,
            "coordinate_frame": "ego_at_t0",
        }
        weights = RerankWeights()
        learned_probability = np.zeros((2, 6, 6), dtype=np.float32)
        learned_probability[0, 0, 3] = 0.9
        learned_probability[1, 4, 4] = 0.9
        persistence_probability = np.zeros((2, 6, 6), dtype=np.float32)
        persistence_probability[0, 3, 3] = 0.9
        persistence_probability[1, 3, 4] = 0.9
        for row in self.test_rows:
            clip_id = row["clip_id"]
            t0_us = row["t0_us"]
            oracle_path = Path(row["artifact_path"])
            oracle_sha256 = row["artifact_sha256"]
            for method, probability, checkpoint, destination in (
                ("learned", learned_probability, checkpoint_sha256, self.rerank_rows),
                ("persistence", persistence_probability, "", self.persistence_rows),
            ):
                learned = method == "learned"
                evaluation_binding = {
                    "evaluation_manifest": evaluation_manifest,
                    "checkpoint": (
                        {
                            "path": str((self.root / "best.pt").resolve()),
                            "sha256": checkpoint,
                            "epoch": 3,
                            "seed": 17,
                            "train_manifest_sha256": "1" * 64,
                            "validation_manifest_sha256": "2" * 64,
                            "train_dataset_fingerprint": "3" * 64,
                            "validation_dataset_fingerprint": "4" * 64,
                        }
                        if learned
                        else None
                    ),
                    "protocol": protocol_provenance if learned else None,
                    "checkpoint_selection": selection_binding if learned else None,
                    "evaluation_audit": audit_binding if learned else None,
                    "partition_isolation": {
                        "train_evaluation_chunk_overlap": 0 if learned else None,
                        "validation_evaluation_chunk_overlap": 0 if learned else None,
                    },
                }
                evaluation_binding["binding_fingerprint"] = canonical_fingerprint(
                    evaluation_binding
                )
                prediction_run = {
                    "method": method,
                    "data_role": "test",
                    "checkpoint_sha256": checkpoint,
                    "amp": False,
                    "evaluation_binding": evaluation_binding,
                    "data_schema": data_schema,
                }
                run_fingerprint = canonical_fingerprint(prediction_run)
                prediction_path = (
                    self.prediction_dir / f"{clip_id}.{method}.prediction.npz"
                )
                np.savez_compressed(
                    prediction_path,
                    schema_version=np.int16(2),
                    clip_id=np.asarray(clip_id),
                    t0_us=np.int64(t0_us),
                    occupancy_prob=probability,
                    horizons_s=np.array([1.0, 2.0], dtype=np.float32),
                    grid_origin_xy_m=np.array([-2.0, -3.0], dtype=np.float32),
                    resolution_m=np.float32(1.0),
                    coordinate_frame=np.asarray("ego_at_t0"),
                    source_artifact_sha256=np.asarray(oracle_sha256),
                    producer_method=np.asarray(method),
                    checkpoint_sha256=np.asarray(checkpoint),
                    prediction_run_json=np.asarray(
                        json.dumps(
                            prediction_run,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                    ),
                    prediction_run_fingerprint=np.asarray(run_fingerprint),
                )
                reranked = rerank_artifacts(
                    prediction_path,
                    oracle_path,
                    weights,
                    expected_clip_id=clip_id,
                    expected_t0_us=t0_us,
                )
                destination.append(
                    {
                        "prediction_path": str(prediction_path.resolve()),
                        "prediction_sha256": sha256_file(prediction_path),
                        "candidate_path": str(oracle_path.resolve()),
                        **reranked,
                    }
                )

    def mutate_prediction_run(
        self, method: str, mutate: Callable[[dict], None]
    ) -> None:
        rows = self.rerank_rows if method == "learned" else self.persistence_rows
        row = rows[0]
        prediction_path = Path(row["prediction_path"])
        with np.load(prediction_path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        prediction_run = json.loads(str(arrays["prediction_run_json"].reshape(())))
        mutate(prediction_run)
        binding = prediction_run.get("evaluation_binding")
        if isinstance(binding, dict):
            binding_payload = dict(binding)
            binding_payload.pop("binding_fingerprint", None)
            binding["binding_fingerprint"] = canonical_fingerprint(binding_payload)
        run_fingerprint = canonical_fingerprint(prediction_run)
        arrays["prediction_run_json"] = np.asarray(
            json.dumps(
                prediction_run,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        arrays["prediction_run_fingerprint"] = np.asarray(run_fingerprint)
        np.savez_compressed(prediction_path, **arrays)
        row["prediction_run_fingerprint"] = run_fingerprint
        row["prediction_sha256"] = sha256_file(prediction_path)
        destination = (
            self.learned_selections
            if method == "learned"
            else self.persistence_selections
        )
        _write_jsonl(destination, rows)

    def evaluate(self) -> tuple[dict, list[dict]]:
        return evaluate_selection(
            protocol=self.protocol,
            forecast_comparison=self.forecast_comparison,
            test_manifest=self.test_manifest,
            train_manifest=self.train_manifest,
            validation_manifest=self.validation_manifest,
            oracle_manifest=self.oracle_manifest,
            learned_selections=self.learned_selections,
            persistence_selections=self.persistence_selections,
            candidate_dir=self.candidate_dir,
            bootstrap_replicates=200,
            bootstrap_seed=17,
        )


class ClusterBootstrapTest(unittest.TestCase):
    def test_interval_is_deterministic_and_resamples_whole_clusters(self) -> None:
        values = np.array([0.0, 0.0, 10.0])
        clusters = ["large", "large", "small"]
        first = cluster_bootstrap_interval(values, clusters, replicates=500, seed=9)
        second = cluster_bootstrap_interval(values, clusters, replicates=500, seed=9)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["estimate"], 10.0 / 3.0)
        self.assertEqual(first["ci_95_lower"], 0.0)
        self.assertEqual(first["ci_95_upper"], 10.0)

    def test_interval_rejects_one_cluster(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two source chunks"):
            cluster_bootstrap_interval([1.0, 2.0], ["same", "same"])


class HeldOutSelectionEvaluationTest(unittest.TestCase):
    def test_evaluation_joins_artifacts_and_reports_paired_intervals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SelectionFixture(Path(temporary))
            summary, per_clip = fixture.evaluate()
            protocol_sha256 = sha256_file(fixture.protocol)
            forecast_sha256 = sha256_file(fixture.forecast_comparison)
        self.assertEqual(summary["evaluation"]["clips"], 4)
        self.assertEqual(summary["evaluation"]["source_chunks"], 2)
        self.assertEqual(summary["evaluation"]["candidate_count"], 2)
        self.assertEqual(len(per_clip), 4)
        self.assertEqual(summary["policy_macro_means"]["candidate_0"]["ade_m"], 2.0)
        self.assertEqual(
            summary["policy_macro_means"]["learned_selected"]["ade_m"], 1.0
        )
        self.assertEqual(
            summary["policy_macro_means"]["persistence_selected"]["ade_m"], 2.0
        )
        self.assertEqual(
            summary["policy_macro_means"]["comfort_only_selected"]["ade_m"],
            1.0,
        )
        self.assertEqual(
            summary["policy_macro_means"]["random_candidate_expectation"]["ade_m"],
            1.5,
        )
        difference = summary["paired_differences"][
            "learned_selected_minus_candidate_0"
        ]["ade_m"]
        self.assertEqual(difference["estimate"], -1.0)
        self.assertEqual(difference["ci_95_lower"], -1.0)
        self.assertEqual(difference["ci_95_upper"], -1.0)
        self.assertEqual(
            summary["selection_rates"]["learned_change_from_candidate_0"]["estimate"],
            1.0,
        )
        self.assertEqual(
            summary["selection_rates"]["learned_oracle_agreement"]["estimate"],
            1.0,
        )
        self.assertEqual(
            summary["selection_rates"]["comfort_only_oracle_agreement"]["estimate"],
            1.0,
        )
        self.assertEqual(
            summary["paired_differences"][
                "learned_selected_minus_persistence_selected"
            ]["ade_m"]["estimate"],
            -1.0,
        )
        self.assertEqual(
            summary["candidate_diversity"]["macro_means"]["unique_trajectory_count"],
            2.0,
        )
        self.assertEqual(
            summary["candidate_diversity"]["definition"][
                "trajectory_unique_tolerance_m"
            ],
            1e-4,
        )
        self.assertEqual(
            summary["candidate_diversity"]["macro_means"]["endpoint_spread_m"],
            1.0,
        )
        self.assertEqual(
            summary["candidate_diversity"]["nonzero_selection_opportunity_rate"][
                "estimate"
            ],
            1.0,
        )
        self.assertTrue(summary["acceptance_gates"]["all_passed"])
        reduction = summary["acceptance_gates"]["decisions"][
            "candidate_0_collision_exposure_relative_reduction_minimum"
        ]
        self.assertAlmostEqual(reduction["relative_reduction"], 0.8)
        self.assertEqual(summary["protocol"]["sha256"], protocol_sha256)
        learned_run = summary["run"]["learned"]["prediction_run"]
        persistence_run = summary["run"]["persistence"]["prediction_run"]
        self.assertEqual(
            learned_run["evaluation_binding"]["protocol"], summary["protocol"]
        )
        self.assertEqual(learned_run["method"], "learned")
        self.assertIsNone(persistence_run["evaluation_binding"]["checkpoint_selection"])
        self.assertIsNone(persistence_run["evaluation_binding"]["checkpoint"])
        self.assertEqual(
            summary["provenance"]["forecast_comparison"]["sha256"],
            forecast_sha256,
        )
        self.assertTrue(summary["benchmark_acceptance"]["all_passed"])
        self.assertTrue(
            summary["benchmark_acceptance"]["components"]["forecast"]["all_passed"]
        )
        self.assertTrue(
            summary["benchmark_acceptance"]["components"]["selection"]["all_passed"]
        )
        self.assertFalse(
            summary["evaluation"]["uncertainty_scope"][
                "training_seed_variability_included"
            ]
        )

    def test_zero_baseline_exposure_fails_relative_reduction_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SelectionFixture(Path(temporary), collision_exposure=(0.0, 0.1))
            summary, _ = fixture.evaluate()
        decision = summary["acceptance_gates"]["decisions"][
            "candidate_0_collision_exposure_relative_reduction_minimum"
        ]
        self.assertFalse(decision["passed"])
        self.assertIsNone(decision["relative_reduction"])
        self.assertIn("zero mean collision exposure", decision["reason"])
        self.assertFalse(summary["acceptance_gates"]["all_passed"])

    def test_protocol_fingerprint_and_configuration_drift_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SelectionFixture(Path(temporary))
            payload = json.loads(fixture.protocol.read_text(encoding="utf-8"))
            payload["uncertainty"]["bootstrap_seed"] = 18
            fixture.protocol.write_text(
                json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                fixture.evaluate()

        with tempfile.TemporaryDirectory() as temporary:
            fixture = SelectionFixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "Bootstrap bootstrap_seed"):
                evaluate_selection(
                    protocol=fixture.protocol,
                    forecast_comparison=fixture.forecast_comparison,
                    test_manifest=fixture.test_manifest,
                    train_manifest=fixture.train_manifest,
                    validation_manifest=fixture.validation_manifest,
                    oracle_manifest=fixture.oracle_manifest,
                    learned_selections=fixture.learned_selections,
                    persistence_selections=fixture.persistence_selections,
                    candidate_dir=fixture.candidate_dir,
                    bootstrap_replicates=200,
                    bootstrap_seed=18,
                )

    def test_protocol_binds_exact_test_manifest_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SelectionFixture(Path(temporary))
            copy = fixture.root / "copied-test.jsonl"
            copy.write_bytes(fixture.test_manifest.read_bytes())
            with self.assertRaisesRegex(ValueError, "test manifest path differs"):
                evaluate_selection(
                    protocol=fixture.protocol,
                    forecast_comparison=fixture.forecast_comparison,
                    test_manifest=copy,
                    train_manifest=fixture.train_manifest,
                    validation_manifest=fixture.validation_manifest,
                    oracle_manifest=fixture.oracle_manifest,
                    learned_selections=fixture.learned_selections,
                    persistence_selections=fixture.persistence_selections,
                    candidate_dir=fixture.candidate_dir,
                    bootstrap_replicates=200,
                    bootstrap_seed=17,
                )
            copied_oracle = fixture.root / "copied-oracle.jsonl"
            copied_oracle.write_bytes(fixture.oracle_manifest.read_bytes())
            with self.assertRaisesRegex(ValueError, "Oracle manifest path differs"):
                evaluate_selection(
                    protocol=fixture.protocol,
                    forecast_comparison=fixture.forecast_comparison,
                    test_manifest=fixture.test_manifest,
                    train_manifest=fixture.train_manifest,
                    validation_manifest=fixture.validation_manifest,
                    oracle_manifest=copied_oracle,
                    learned_selections=fixture.learned_selections,
                    persistence_selections=fixture.persistence_selections,
                    candidate_dir=fixture.candidate_dir,
                    bootstrap_replicates=200,
                    bootstrap_seed=17,
                )

    def test_protocol_binds_exact_reranker_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SelectionFixture(Path(temporary))
            for row in fixture.rerank_rows + fixture.persistence_rows:
                row["weights"]["collision"] = 9.0
            _write_jsonl(fixture.learned_selections, fixture.rerank_rows)
            _write_jsonl(fixture.persistence_selections, fixture.persistence_rows)
            with self.assertRaisesRegex(ValueError, "reranker weights"):
                fixture.evaluate()

    def test_rejects_forged_reranker_decision_and_score(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SelectionFixture(Path(temporary))
            for row in fixture.rerank_rows:
                row["world_selected_index"] = 0
                row["candidate_scores"][0]["score"] = -1000.0
                row["candidate_scores"][1]["score"] = 1000.0
            _write_jsonl(fixture.learned_selections, fixture.rerank_rows)
            with self.assertRaisesRegex(
                ValueError, "world_selected_index does not match recomputation"
            ):
                fixture.evaluate()

        with tempfile.TemporaryDirectory() as temporary:
            fixture = SelectionFixture(Path(temporary))
            fixture.rerank_rows[0]["candidate_scores"][0]["score"] += 0.5
            _write_jsonl(fixture.learned_selections, fixture.rerank_rows)
            with self.assertRaisesRegex(
                ValueError, "candidate 0 score differs from recomputation"
            ):
                fixture.evaluate()

    def test_rejects_reranker_metadata_that_differs_from_prediction(self) -> None:
        for field in ("checkpoint_sha256", "prediction_run_fingerprint"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                fixture = SelectionFixture(Path(temporary))
                fixture.rerank_rows[0][field] = "f" * 64
                _write_jsonl(fixture.learned_selections, fixture.rerank_rows)
                with self.assertRaisesRegex(
                    ValueError,
                    (
                        "Prediction run checkpoint does not match the artifact"
                        if field == "checkpoint_sha256"
                        else rf"{field} does not match recomputation"
                    ),
                ):
                    fixture.evaluate()

    def test_prediction_run_binding_rejects_self_consistent_forgery(self) -> None:
        cases = (
            (
                "method",
                "learned",
                lambda run: run.__setitem__("method", "persistence"),
                "run method does not match",
            ),
            (
                "test manifest",
                "learned",
                lambda run: run["evaluation_binding"][
                    "evaluation_manifest"
                ].__setitem__("sha256", "f" * 64),
                "different test-manifest binding",
            ),
            (
                "protocol",
                "learned",
                lambda run: run["evaluation_binding"]["protocol"].__setitem__(
                    "sha256", "f" * 64
                ),
                "protocol provenance does not match",
            ),
            (
                "persistence selection",
                "persistence",
                lambda run: run["evaluation_binding"].__setitem__(
                    "checkpoint_selection", {"forged": True}
                ),
                "must not carry checkpoint selection metadata",
            ),
        )
        for name, method, mutate, expected_error in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = SelectionFixture(Path(temporary))
                fixture.mutate_prediction_run(method, mutate)
                with self.assertRaisesRegex(ValueError, expected_error):
                    fixture.evaluate()

    def test_prediction_run_json_and_fingerprint_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SelectionFixture(Path(temporary))
            row = fixture.rerank_rows[0]
            prediction_path = Path(row["prediction_path"])
            with np.load(prediction_path, allow_pickle=False) as archive:
                arrays = {
                    name: np.asarray(archive[name])
                    for name in archive.files
                    if name != "prediction_run_json"
                }
            np.savez_compressed(prediction_path, **arrays)
            row["prediction_sha256"] = sha256_file(prediction_path)
            _write_jsonl(fixture.learned_selections, fixture.rerank_rows)
            with self.assertRaisesRegex(ValueError, "missing prediction_run_json"):
                fixture.evaluate()

        with tempfile.TemporaryDirectory() as temporary:
            fixture = SelectionFixture(Path(temporary))
            row = fixture.rerank_rows[0]
            prediction_path = Path(row["prediction_path"])
            with np.load(prediction_path, allow_pickle=False) as archive:
                arrays = {name: np.asarray(archive[name]) for name in archive.files}
            arrays["prediction_run_fingerprint"] = np.asarray("f" * 64)
            np.savez_compressed(prediction_path, **arrays)
            row["prediction_sha256"] = sha256_file(prediction_path)
            row["prediction_run_fingerprint"] = "f" * 64
            _write_jsonl(fixture.learned_selections, fixture.rerank_rows)
            with self.assertRaisesRegex(ValueError, "run fingerprint does not match"):
                fixture.evaluate()

    def test_prediction_run_rejects_changed_selection_record_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SelectionFixture(Path(temporary))
            binding = fixture.rerank_rows[0]
            with np.load(binding["prediction_path"], allow_pickle=False) as archive:
                prediction_run = json.loads(str(archive["prediction_run_json"]))
            selection_path = Path(
                prediction_run["evaluation_binding"]["checkpoint_selection"]["path"]
            )
            selection_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "selection SHA-256 does not match"):
                fixture.evaluate()

    def test_forecast_comparison_provenance_is_required_exactly(self) -> None:
        cases = (
            (
                "schema",
                lambda report: report.__setitem__("schema_version", 2),
                "unsupported schema version",
            ),
            (
                "protocol path",
                lambda report: report["protocol"].__setitem__(
                    "path", "/different/protocol.json"
                ),
                "different protocol path",
            ),
            (
                "protocol SHA",
                lambda report: report["protocol"].__setitem__("sha256", "f" * 64),
                "protocol sha256 does not match",
            ),
            (
                "protocol fingerprint",
                lambda report: report["protocol"].__setitem__(
                    "protocol_fingerprint", "f" * 64
                ),
                "protocol protocol_fingerprint does not match",
            ),
            (
                "test manifest path",
                lambda report: report["inputs"]["manifest"].__setitem__(
                    "manifest_path", "/different/test.jsonl"
                ),
                "different test-manifest path",
            ),
            (
                "test manifest SHA",
                lambda report: report["inputs"]["manifest"].__setitem__(
                    "manifest_sha256", "f" * 64
                ),
                "test-manifest manifest_sha256 does not match",
            ),
            (
                "test dataset fingerprint",
                lambda report: report["inputs"]["manifest"].__setitem__(
                    "dataset_fingerprint", "f" * 64
                ),
                "test-manifest dataset_fingerprint does not match",
            ),
            (
                "test clip count",
                lambda report: report["inputs"]["manifest"].__setitem__("clips", 999),
                "test-manifest clips does not match",
            ),
            (
                "gate decision",
                lambda report: report["acceptance_gates"]["decisions"][
                    "average_precision"
                ].__setitem__("passed", False),
                "average_precision gate decision is inconsistent",
            ),
        )
        for name, mutate, expected_error in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = SelectionFixture(Path(temporary))
                report = json.loads(
                    fixture.forecast_comparison.read_text(encoding="utf-8")
                )
                mutate(report)
                fixture.forecast_comparison.write_text(
                    json.dumps(report, sort_keys=True) + "\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, expected_error):
                    fixture.evaluate()

    def test_combined_gate_fails_when_forecast_gate_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SelectionFixture(Path(temporary))
            report = json.loads(fixture.forecast_comparison.read_text(encoding="utf-8"))
            report["metrics"]["learned"]["short_horizon"]["average_precision"] = 0.6
            report["paired_differences"]["short_horizon"]["average_precision"][
                "estimate"
            ] = -0.1
            decision = report["acceptance_gates"]["decisions"]["average_precision"]
            decision["learned"] = 0.6
            decision["learned_minus_persistence"] = -0.1
            decision["passed"] = False
            report["acceptance_gates"]["all_passed"] = False
            fixture.forecast_comparison.write_text(
                json.dumps(report, sort_keys=True) + "\n", encoding="utf-8"
            )
            summary, _ = fixture.evaluate()
        self.assertFalse(summary["benchmark_acceptance"]["all_passed"])
        self.assertFalse(
            summary["benchmark_acceptance"]["components"]["forecast"]["all_passed"]
        )
        self.assertTrue(
            summary["benchmark_acceptance"]["components"]["selection"]["all_passed"]
        )

    def test_test_train_chunk_leakage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SelectionFixture(Path(temporary))
            _write_jsonl(
                fixture.train_manifest,
                [
                    {
                        "clip_id": "different",
                        "t0_us": 999,
                        "chunk_id": "chunk-0",
                        "artifact_path": "unused.npz",
                    }
                ],
            )
            with self.assertRaisesRegex(ValueError, "share source chunks"):
                fixture.evaluate()

    def test_missing_and_duplicate_rerank_rows_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SelectionFixture(Path(temporary))
            _write_jsonl(fixture.learned_selections, fixture.rerank_rows[:-1])
            with self.assertRaisesRegex(ValueError, "exactly match"):
                fixture.evaluate()
            _write_jsonl(
                fixture.learned_selections,
                fixture.rerank_rows + [fixture.rerank_rows[0]],
            )
            with self.assertRaisesRegex(ValueError, "Duplicate clip identity"):
                fixture.evaluate()

    def test_rerank_artifact_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SelectionFixture(Path(temporary))
            fixture.rerank_rows[0]["candidate_sha256"] = "e" * 64
            _write_jsonl(fixture.learned_selections, fixture.rerank_rows)
            with self.assertRaisesRegex(ValueError, "artifact SHA-256 mismatch"):
                fixture.evaluate()

    def test_learned_and_persistence_weights_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SelectionFixture(Path(temporary))
            fixture.persistence_rows[0]["weights"]["collision"] = 9.0
            _write_jsonl(fixture.persistence_selections, fixture.persistence_rows)
            with self.assertRaisesRegex(ValueError, "reranker weights"):
                fixture.evaluate()

    def test_learned_and_persistence_comfort_components_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SelectionFixture(Path(temporary))
            fixture.persistence_rows[0]["candidate_scores"][0][
                "mean_acceleration_mps2"
            ] = 2.0
            _write_jsonl(fixture.persistence_selections, fixture.persistence_rows)
            with self.assertRaisesRegex(ValueError, "differs from recomputation"):
                fixture.evaluate()

    def test_missing_candidate_record_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SelectionFixture(Path(temporary))
            (fixture.candidate_dir / "records" / "clip-3.json").unlink()
            with self.assertRaisesRegex(ValueError, "missing test clips"):
                fixture.evaluate()

    def test_cli_writes_audit_rows_before_summary_completion_marker(self) -> None:
        summary = {"evaluation": {"clips": 2, "source_chunks": 2}}
        per_clip = [{"clip_id": "clip-0"}, {"clip_id": "clip-1"}]
        events: list[tuple[str, object, str]] = []
        argv = [
            "evaluate_selection",
            "--protocol",
            "protocol.json",
            "--forecast-comparison",
            "forecast.json",
            "--test-manifest",
            "test.jsonl",
            "--train-manifest",
            "train.jsonl",
            "--validation-manifest",
            "validation.jsonl",
            "--oracle-manifest",
            "oracle.jsonl",
            "--learned-selections",
            "learned.jsonl",
            "--persistence-selections",
            "persistence.jsonl",
            "--candidate-dir",
            "candidates",
            "--output",
            "summary.json",
            "--per-clip-output",
            "audit.jsonl",
        ]
        with (
            mock.patch("sys.argv", argv),
            mock.patch(
                "src.world_model.evaluate_selection.evaluate_selection",
                return_value=(summary, per_clip),
            ),
            mock.patch(
                "src.world_model.evaluate_selection.write_jsonl",
                side_effect=lambda rows, path: events.append(("jsonl", rows, path)),
            ),
            mock.patch(
                "src.world_model.evaluate_selection.write_json",
                side_effect=lambda payload, path: events.append(
                    ("json", payload, path)
                ),
            ),
        ):
            main()

        self.assertEqual(
            events,
            [
                ("jsonl", per_clip, "audit.jsonl"),
                ("json", summary, "summary.json"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
