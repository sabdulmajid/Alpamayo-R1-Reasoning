import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.world_model.evaluate_selection import (
    cluster_bootstrap_interval,
    evaluate_selection,
)
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
        _write_jsonl(self.learned_selections, self.rerank_rows)
        _write_jsonl(self.persistence_selections, self.persistence_rows)
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
                "reranker_weights": self.rerank_rows[0]["weights"],
            },
            "uncertainty": {
                "method": "paired_cluster_bootstrap_percentile",
                "cluster_unit": "chunk_id",
                "confidence_level": 0.95,
                "bootstrap_seed": 17,
                "bootstrap_replicates": 200,
            },
            "acceptance_gates": {
                "forecast": {},
                "selection": {
                    "candidate_0_collision_exposure_relative_reduction_minimum": 0.15,
                    "learned_collision_exposure_no_worse_than_persistence": True,
                    "ade_degradation_m_paired_ci_95_upper_maximum": 0.2,
                    "out_of_bounds_fraction_no_higher_than_candidate_0": True,
                    "observed_fraction_no_lower_than_candidate_0": True,
                },
            },
        }
        protocol_payload["protocol_fingerprint"] = canonical_fingerprint(protocol_payload)
        self.protocol.write_text(
            json.dumps(protocol_payload, sort_keys=True) + "\n", encoding="utf-8"
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
        candidate_zero[:, 1] += 2.0
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
        past = np.zeros((2, 4, 5), dtype=np.uint8)
        future = np.zeros((2, 4, 5), dtype=np.uint8)
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
            bev_y_min_m=np.float32(-1.0),
            bev_resolution_m=np.float32(1.0),
            frame=np.asarray("ego_at_t0"),
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
            "candidate_artifact_sha256": candidate_sha256,
            "candidate_count": 2,
            "oracle_safest_idx": 1,
        }
        self.test_rows.append(dict(oracle_row))
        self.oracle_rows.append(dict(oracle_row))

        prediction_path = self.prediction_dir / f"{clip_id}.prediction"
        prediction_path.write_bytes(f"prediction:{clip_id}".encode())
        self.rerank_rows.append(
            {
                "clip_id": clip_id,
                "t0_us": t0_us,
                "candidate_count": 2,
                "first_candidate_index": 0,
                "world_selected_index": 1,
                "producer_method": "learned",
                "checkpoint_sha256": "c" * 64,
                "prediction_run_fingerprint": "d" * 64,
                "selection_policy": "predicted_occupancy_v1",
                "candidate_path": str(oracle_path),
                "candidate_sha256": oracle_sha256,
                "prediction_path": str(prediction_path),
                "prediction_sha256": sha256_file(prediction_path),
                "weights": {
                    "collision": 10.0,
                    "uncertainty": 1.0,
                    "out_of_bounds": 5.0,
                    "acceleration": 0.05,
                    "jerk": 0.01,
                    "curvature": 0.1,
                    "progress": 0.02,
                },
                "candidate_scores": [
                    {
                        "score": 2.0,
                        "mean_acceleration_mps2": 1.0,
                        "mean_jerk_mps3": 0.0,
                        "max_curvature_inv_m": 0.0,
                        "progress_m": 1.0,
                    },
                    {
                        "score": 1.0,
                        "mean_acceleration_mps2": 0.0,
                        "mean_jerk_mps3": 0.0,
                        "max_curvature_inv_m": 0.0,
                        "progress_m": 4.0,
                    },
                ],
            }
        )
        persistence_prediction_path = (
            self.prediction_dir / f"{clip_id}.persistence.prediction"
        )
        persistence_prediction_path.write_bytes(f"persistence:{clip_id}".encode())
        self.persistence_rows.append(
            {
                "clip_id": clip_id,
                "t0_us": t0_us,
                "candidate_count": 2,
                "first_candidate_index": 0,
                "world_selected_index": 0,
                "producer_method": "persistence",
                "checkpoint_sha256": None,
                "prediction_run_fingerprint": "e" * 64,
                "selection_policy": "predicted_occupancy_v1",
                "candidate_path": str(oracle_path),
                "candidate_sha256": oracle_sha256,
                "prediction_path": str(persistence_prediction_path),
                "prediction_sha256": sha256_file(persistence_prediction_path),
                "weights": {
                    "collision": 10.0,
                    "uncertainty": 1.0,
                    "out_of_bounds": 5.0,
                    "acceleration": 0.05,
                    "jerk": 0.01,
                    "curvature": 0.1,
                    "progress": 0.02,
                },
                "candidate_scores": [
                    {
                        "score": 1.0,
                        "mean_acceleration_mps2": 1.0,
                        "mean_jerk_mps3": 0.0,
                        "max_curvature_inv_m": 0.0,
                        "progress_m": 1.0,
                    },
                    {
                        "score": 2.0,
                        "mean_acceleration_mps2": 0.0,
                        "mean_jerk_mps3": 0.0,
                        "max_curvature_inv_m": 0.0,
                        "progress_m": 4.0,
                    },
                ],
            }
        )

    def evaluate(self) -> tuple[dict, list[dict]]:
        return evaluate_selection(
            protocol=self.protocol,
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
        self.assertEqual(summary["evaluation"]["clips"], 4)
        self.assertEqual(summary["evaluation"]["source_chunks"], 2)
        self.assertEqual(summary["evaluation"]["candidate_count"], 2)
        self.assertEqual(len(per_clip), 4)
        self.assertEqual(
            summary["policy_macro_means"]["candidate_0"]["ade_m"], 2.0
        )
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
            summary["selection_rates"]["learned_change_from_candidate_0"][
                "estimate"
            ],
            1.0,
        )
        self.assertEqual(
            summary["selection_rates"]["learned_oracle_agreement"]["estimate"],
            1.0,
        )
        self.assertEqual(
            summary["selection_rates"]["comfort_only_oracle_agreement"][
                "estimate"
            ],
            1.0,
        )
        self.assertEqual(
            summary["paired_differences"][
                "learned_selected_minus_persistence_selected"
            ]["ade_m"]["estimate"],
            -1.0,
        )
        self.assertEqual(
            summary["candidate_diversity"]["macro_means"][
                "unique_trajectory_count"
            ],
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
        self.assertFalse(
            summary["evaluation"]["uncertainty_scope"][
                "training_seed_variability_included"
            ]
        )

    def test_zero_baseline_exposure_fails_relative_reduction_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SelectionFixture(
                Path(temporary), collision_exposure=(0.0, 0.1)
            )
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
            with self.assertRaisesRegex(ValueError, "trajectory_selection"):
                fixture.evaluate()

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
            with self.assertRaisesRegex(ValueError, "comfort components differ"):
                fixture.evaluate()

    def test_missing_candidate_record_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SelectionFixture(Path(temporary))
            (fixture.candidate_dir / "records" / "clip-3.json").unlink()
            with self.assertRaisesRegex(ValueError, "missing test clips"):
                fixture.evaluate()


if __name__ == "__main__":
    unittest.main()
