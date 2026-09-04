import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.world_model.compare_forecasts import compare_forecasts
from src.world_model.runtime import canonical_fingerprint, prediction_filename, sha256_file


HORIZONS = np.array([0.5, 1.0, 2.0], dtype=np.float32)
ORIGIN = np.array([-1.0, -1.0], dtype=np.float32)
CHECKPOINT = "a" * 64
LEARNED_RUN = "b" * 64
PERSISTENCE_RUN = "c" * 64
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
        checkpoint_sha256=np.asarray(CHECKPOINT if method == "learned" else ""),
        prediction_run_fingerprint=np.asarray(
            LEARNED_RUN if method == "learned" else PERSISTENCE_RUN
        ),
    )
    return path


def replace_array(path: Path, name: str, value: np.ndarray) -> None:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays[name] = value
    np.savez_compressed(path, **arrays)


def make_tree(root: Path) -> tuple[Path, Path, Path, list[Path], Path]:
    oracle_dir = root / "oracle"
    learned_dir = root / "learned"
    persistence_dir = root / "persistence"
    oracle_dir.mkdir()
    rows = []
    learned_paths = []
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
        learned_paths.append(
            write_prediction(
                learned_dir,
                oracle_path,
                clip_id,
                t0_us,
                method="learned",
                probability=learned_probability,
            )
        )
        write_prediction(
            persistence_dir,
            oracle_path,
            clip_id,
            t0_us,
            method="persistence",
            probability=persistence_probability,
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
                    "dataset_fingerprint": canonical_fingerprint(dataset_index),
                }
            },
        },
        "training": {},
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
                CHECKPOINT,
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
            ("checkpoint_sha256", np.asarray("e" * 64), "checkpoint identity"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                manifest, learned, persistence, learned_paths, protocol = make_tree(
                    Path(temporary)
                )
                replace_array(learned_paths[0], field, replacement)
                with self.assertRaisesRegex(ValueError, message):
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
