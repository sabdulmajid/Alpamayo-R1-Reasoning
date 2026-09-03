import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.world_model.baselines import ego_compensated_persistence, persistence
from src.world_model.data import (
    SUPPORTED_ORACLE_SCHEMA_VERSIONS,
    OccupancyDataset,
    dataset_provenance,
    load_occupancy_artifact,
    reject_chunk_overlap,
)
from src.world_model.losses import occupancy_loss
from src.world_model.metrics import OccupancyMetricAccumulator
from src.world_model.model import ModelConfig, TemporalOccupancyNet
from src.world_model.rerank import RerankWeights, rerank_artifacts, trajectory_comfort
from src.world_model.runtime import sha256_file
from src.world_model.split_manifest import split_rows


def write_oracle_artifact(path: Path, *, clip_id: str = "clip", t0_us: int = 10) -> None:
    np.savez_compressed(
        path,
        schema_version=np.int16(4),
        clip_id=np.asarray(clip_id),
        t0_us=np.int64(t0_us),
        past_occupancy=np.zeros((3, 8, 10), dtype=np.uint8),
        past_observed=np.ones((3, 8, 10), dtype=np.uint8),
        occupancy=np.zeros((2, 8, 10), dtype=np.uint8),
        observed=np.ones((2, 8, 10), dtype=np.uint8),
        history_offsets_s=np.array([-1.0, -0.5, 0.0], dtype=np.float32),
        horizons_s=np.array([1.0, 2.0], dtype=np.float32),
        bev_x_min_m=np.float32(-2.0),
        bev_x_max_m=np.float32(3.0),
        bev_y_min_m=np.float32(-3.0),
        bev_y_max_m=np.float32(1.0),
        bev_resolution_m=np.float32(0.5),
        frame=np.asarray("ego_at_t0"),
    )


class OccupancyDataTest(unittest.TestCase):
    def test_oracle_contract_loads_without_enrichment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "clip.npz"
            write_oracle_artifact(artifact)
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "clip_id": "clip",
                        "t0_us": 10,
                        "chunk_id": "chunk-a",
                        "artifact_path": "clip.npz",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            item = OccupancyDataset(manifest)[0]
            self.assertEqual(item["inputs"].shape, (3, 2, 8, 10))
            self.assertEqual(item["target"].shape, (2, 8, 10))
            np.testing.assert_allclose(item["grid_origin_xy_m"], [-2.0, -3.0])
            np.testing.assert_allclose(item["past_horizons_s"], [-1.0, -0.5, 0.0])

    def test_manifest_artifact_identity_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "clip.npz"
            write_oracle_artifact(artifact, clip_id="different")
            rows = [
                {
                    "clip_id": "clip",
                    "t0_us": 10,
                    "chunk_id": "chunk-a",
                    "artifact_path": str(artifact),
                }
            ]
            with self.assertRaisesRegex(ValueError, "identity differs"):
                OccupancyDataset(rows)[0]

    def test_missing_past_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "future_only.npz"
            np.savez_compressed(
                artifact,
                schema_version=np.int16(4),
                clip_id=np.asarray("clip"),
                t0_us=np.int64(10),
                occupancy=np.zeros((2, 4, 4), dtype=np.uint8),
                observed=np.ones((2, 4, 4), dtype=np.uint8),
            )
            with self.assertRaisesRegex(ValueError, "past_occupancy"):
                load_occupancy_artifact(artifact)

    def test_only_current_oracle_schema_is_supported(self) -> None:
        self.assertEqual(SUPPORTED_ORACLE_SCHEMA_VERSIONS, frozenset({4}))
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "old.npz"
            write_oracle_artifact(artifact)
            with np.load(artifact, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            arrays["schema_version"] = np.int16(3)
            np.savez_compressed(artifact, **arrays)
            with self.assertRaisesRegex(ValueError, "Unsupported oracle schema_version 3"):
                load_occupancy_artifact(artifact)

    def test_dataset_fingerprint_binds_artifact_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "clip.npz"
            write_oracle_artifact(artifact)
            rows = [
                {
                    "clip_id": "clip",
                    "t0_us": 10,
                    "chunk_id": "chunk-a",
                    "artifact_path": str(artifact),
                }
            ]
            first = dataset_provenance(rows)
            with np.load(artifact, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            arrays["past_occupancy"] = arrays["past_occupancy"].copy()
            arrays["past_occupancy"][0, 0, 0] = 1
            np.savez_compressed(artifact, **arrays)
            second = dataset_provenance(rows)
            self.assertNotEqual(first["dataset_fingerprint"], second["dataset_fingerprint"])
            self.assertNotEqual(
                first["artifacts"][0]["artifact_sha256"],
                second["artifacts"][0]["artifact_sha256"],
            )


class SplitTest(unittest.TestCase):
    def test_chunk_groups_are_disjoint_and_order_invariant(self) -> None:
        rows = [
            {
                "clip_id": f"clip-{index // 2}",
                "t0_us": index,
                "chunk_id": f"chunk-{index // 4}",
                "artifact_path": "x",
            }
            for index in range(60)
        ]
        first = split_rows(rows, seed=7, ratios=(0.7, 0.2, 0.1))
        second = split_rows(list(reversed(rows)), seed=7, ratios=(0.7, 0.2, 0.1))
        self.assertEqual(first, second)
        groups = [{row["chunk_id"] for row in first[name]} for name in ("train", "val", "test")]
        self.assertFalse(groups[0] & groups[1])
        self.assertFalse(groups[0] & groups[2])
        self.assertFalse(groups[1] & groups[2])
        self.assertEqual(sum(len(split) for split in first.values()), len(rows))

    def test_chunk_overlap_is_rejected(self) -> None:
        left = {"chunk_ids": ["1", "2"]}
        right = {"chunk_ids": ["2", "3"]}
        with self.assertRaisesRegex(ValueError, "share source chunks: 2"):
            reject_chunk_overlap(left, right, left_name="Train", right_name="validation")


class ModelAndMetricTest(unittest.TestCase):
    def test_model_loss_backward(self) -> None:
        model = TemporalOccupancyNet(ModelConfig(in_channels=2, base_channels=4, num_horizons=3))
        inputs = torch.rand(2, 3, 2, 15, 17)
        target = (torch.rand(2, 3, 15, 17) > 0.8).float()
        visibility = torch.ones_like(target)
        logits = model(inputs)
        self.assertEqual(logits.shape, target.shape)
        loss, components = occupancy_loss(logits, target, visibility)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(components["loss"], 0)
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_metrics_ignore_unobserved_cells(self) -> None:
        target = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
        probability = target.clone()
        probability[0, 0, 0, 0] = 1.0
        visibility = torch.ones_like(target)
        visibility[0, 0, 0, 0] = 0.0
        accumulator = OccupancyMetricAccumulator(1, bins=20)
        accumulator.update(probability, target, visibility)
        metrics = accumulator.compute([1.0])
        self.assertEqual(metrics["per_horizon"][0]["valid_cells"], 3)
        self.assertAlmostEqual(metrics["mean"]["iou"], 1.0)
        self.assertAlmostEqual(metrics["mean"]["average_precision"], 1.0)
        self.assertAlmostEqual(metrics["mean"]["brier"], 0.0)

    def test_persistence_baselines_identity(self) -> None:
        inputs = torch.zeros(1, 2, 1, 5, 5)
        inputs[0, -1, 0, 2, 3] = 1
        repeated = persistence(inputs, 2)
        warped = ego_compensated_persistence(
            inputs,
            torch.zeros(1, 2, 3),
            torch.tensor([[0.0, 0.0]]),
            torch.tensor([1.0]),
        )
        torch.testing.assert_close(repeated, warped)


class RerankerTest(unittest.TestCase):
    def test_comfort_includes_origin_before_first_positive_sample(self) -> None:
        xy = np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
        times = np.array([1.0, 2.0], dtype=np.float32)
        metrics = trajectory_comfort(xy, times)
        self.assertAlmostEqual(metrics["mean_speed_mps"], 1.0)
        self.assertAlmostEqual(metrics["progress_m"], 2.0)

    def test_predicted_occupancy_changes_selection_without_oracle_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            probability = np.zeros((2, 20, 30), dtype=np.float32)
            probability[:, 9:12, 13:23] = 0.99
            prediction_path = root / "prediction.npz"
            np.savez_compressed(
                prediction_path,
                schema_version=np.int16(2),
                clip_id=np.asarray("clip"),
                t0_us=np.int64(10),
                occupancy_prob=probability,
                horizons_s=np.array([1.0, 2.0], dtype=np.float32),
                grid_origin_xy_m=np.array([-2.0, -5.0], dtype=np.float32),
                resolution_m=np.float32(0.5),
                coordinate_frame=np.asarray("ego_at_t0"),
                source_artifact_sha256=np.asarray("a" * 64),
                producer_method=np.asarray("learned"),
                checkpoint_sha256=np.asarray("b" * 64),
                prediction_run_fingerprint=np.asarray("c" * 64),
            )
            times = np.array([0.0, 1.0, 2.0], dtype=np.float32)
            candidate_zero = np.stack(
                (np.array([0.0, 4.0, 8.0]), np.zeros(3), np.zeros(3)), axis=1
            )
            candidate_one = np.stack(
                (np.array([0.0, 4.0, 8.0]), np.full(3, 4.0), np.zeros(3)), axis=1
            )
            candidate_path = root / "candidates.npz"
            np.savez_compressed(
                candidate_path,
                clip_id=np.asarray("clip"),
                t0_us=np.int64(10),
                candidate_xyz=np.stack((candidate_zero, candidate_one)),
                candidate_times_s=times,
                candidate_yaw=np.zeros((2, 3), dtype=np.float32),
                vehicle_dimensions_m=np.array([2.0, 1.0, 1.5, 0.0], dtype=np.float32),
                # Evaluation-only arrays may coexist, but reranking never loads them.
                occupancy=np.ones((2, 20, 30), dtype=np.uint8),
                oracle_safest_idx=np.int32(0),
                frame=np.asarray("ego_at_t0"),
                bev_x_min_m=np.float32(-2.0),
                bev_x_max_m=np.float32(13.0),
                bev_y_min_m=np.float32(-5.0),
                bev_y_max_m=np.float32(5.0),
                bev_resolution_m=np.float32(0.5),
            )
            with np.load(prediction_path, allow_pickle=False) as prediction:
                prediction_arrays = {name: prediction[name] for name in prediction.files}
            prediction_arrays["source_artifact_sha256"] = np.asarray(
                sha256_file(candidate_path)
            )
            np.savez_compressed(prediction_path, **prediction_arrays)
            result = rerank_artifacts(prediction_path, candidate_path, RerankWeights())
            self.assertEqual(result["world_selected_index"], 1)
            self.assertNotIn("oracle_selected_index", result)
            self.assertEqual(result["selection_policy"], "predicted_occupancy_v1")

            with np.load(candidate_path, allow_pickle=False) as artifact:
                arrays = {name: artifact[name] for name in artifact.files}
            arrays["oracle_safest_idx"] = np.int32(1)
            arrays["occupancy"] = np.zeros((2, 20, 30), dtype=np.uint8)
            np.savez_compressed(candidate_path, **arrays)
            prediction_arrays["source_artifact_sha256"] = np.asarray(
                sha256_file(candidate_path)
            )
            np.savez_compressed(prediction_path, **prediction_arrays)
            repeated = rerank_artifacts(prediction_path, candidate_path, RerankWeights())
            self.assertEqual(repeated["world_selected_index"], 1)


if __name__ == "__main__":
    unittest.main()
