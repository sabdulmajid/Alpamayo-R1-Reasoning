import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

from src.lidar_world_oracle import (
    BEVConfig,
    CandidateArtifactRef,
    ClipOracleResult,
    REFERENCE_TIMESTAMP_MODE,
    PointFilterConfig,
    VehicleDimensions,
    atomic_save_npz,
    config_fingerprint,
    decode_lidar_spin,
    evaluate_candidate_risks,
    file_sha256,
    filter_ego_returns,
    footprint_mask,
    footprint_within_bev,
    infer_tangent_yaw,
    load_candidate_record_index,
    oracle_config_payload,
    rasterize_spin,
    result_arrays,
    select_spin_rows,
    transform_points_to_t0,
    validate_candidate_artifact,
    validate_clip_frame,
    validate_existing_output,
)


def transform(*, yaw_rad: float = 0.0, translation=(0.0, 0.0, 0.0)) -> np.ndarray:
    cosine, sine = np.cos(yaw_rad), np.sin(yaw_rad)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:2, :2] = [[cosine, -sine], [sine, cosine]]
    matrix[:3, 3] = translation
    return matrix


class TransformTests(unittest.TestCase):
    def test_sensor_extrinsic_and_timestamped_egomotion_are_composed(self):
        points_sensor = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        transform_rig_sensor = transform(translation=(1.0, 0.0, 0.0))
        point_poses = np.stack(
            [
                transform(translation=(3.0, 0.0, 0.0)),
                transform(yaw_rad=np.pi / 2, translation=(3.0, 0.0, 0.0)),
            ]
        )
        t0_pose = transform(translation=(1.0, 0.0, 0.0))

        points_t0, origins_t0, points_rig = transform_points_to_t0(
            points_sensor, transform_rig_sensor, point_poses, t0_pose
        )

        np.testing.assert_allclose(points_rig, [[2.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        np.testing.assert_allclose(
            points_t0, [[4.0, 0.0, 0.0], [1.0, 1.0, 0.0]], atol=1e-7
        )
        np.testing.assert_allclose(
            origins_t0, [[3.0, 0.0, 0.0], [2.0, 1.0, 0.0]], atol=1e-7
        )

    def test_t0_rotation_is_inverted(self):
        points = np.array([[1.0, 0.0, 0.0]])
        points_t0, _, _ = transform_points_to_t0(
            points,
            np.eye(4),
            np.eye(4)[None],
            transform(yaw_rad=np.pi / 2),
        )
        np.testing.assert_allclose(points_t0, [[0.0, -1.0, 0.0]], atol=1e-7)


class RasterTests(unittest.TestCase):
    def setUp(self):
        self.config = BEVConfig(-5.0, 5.0, -5.0, 5.0, 1.0)

    def test_occupancy_counts_hits_and_ground_only_affects_occupancy(self):
        points = np.array(
            [
                [1.2, 1.2, 1.0],
                [1.3, 1.2, 2.0],
                [2.2, -1.2, 0.0],
            ]
        )
        origins = np.zeros_like(points)
        occupancy, observed, counts = rasterize_spin(
            points,
            origins,
            self.config,
            PointFilterConfig(ground_min_z_m=0.2, max_z_m=3.0, max_range_m=10.0),
            ray_stride=1,
        )

        row, col = 6, 6
        self.assertEqual(counts[row, col], 2)
        self.assertEqual(occupancy[row, col], 1)
        ground_row, ground_col = 3, 7
        self.assertEqual(occupancy[ground_row, ground_col], 0)
        self.assertEqual(observed[ground_row, ground_col], 1)
        self.assertGreater(observed.sum(), occupancy.sum())

    def test_ego_filter_uses_rear_axle_offset_and_height(self):
        vehicle = VehicleDimensions(4.0, 2.0, 2.0, 1.0)
        points = np.array(
            [
                [1.0, 0.0, 1.0],
                [-2.0, 0.0, 1.0],
                [1.0, 2.0, 1.0],
                [1.0, 0.0, 3.0],
            ]
        )
        np.testing.assert_array_equal(
            filter_ego_returns(points, vehicle, padding_m=0.0),
            [False, True, True, True],
        )

    def test_spin_selection_uses_midpoint_and_enforces_tolerance(self):
        frame = pd.DataFrame(
            {
                "spin_start_timestamp": [900_000, 1_900_000, 2_900_000],
                "spin_end_timestamp": [1_100_000, 2_100_000, 3_100_000],
            }
        )
        selected = select_spin_rows(
            frame, t0_us=0, horizons_s=[1.0, 3.0], tolerance_s=0.01
        )
        self.assertEqual(
            [int(row.spin_start_timestamp) for row in selected], [900_000, 2_900_000]
        )
        with self.assertRaisesRegex(ValueError, "No LiDAR spin"):
            select_spin_rows(frame, t0_us=0, horizons_s=[1.4], tolerance_s=0.1)

    def test_reference_timestamp_schema_uses_rigid_spin_pose(self):
        frame = pd.DataFrame(
            {
                "reference_timestamp": [900_000, 1_900_000, 2_900_000],
                "draco_encoded_pointcloud": [b"first", b"second", b"third"],
            }
        )
        selected = select_spin_rows(
            frame,
            t0_us=0,
            horizons_s=[1.0, 3.0],
            tolerance_s=0.11,
            timestamp_mode=REFERENCE_TIMESTAMP_MODE,
        )
        self.assertEqual(
            [int(row.reference_timestamp) for row in selected],
            [900_000, 2_900_000],
        )
        cloud = SimpleNamespace(
            points=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            attributes=[],
        )
        with mock.patch("DracoPy.decode", return_value=cloud):
            spin = decode_lidar_spin(
                selected[0], timestamp_mode=REFERENCE_TIMESTAMP_MODE
            )
        np.testing.assert_array_equal(spin.point_timestamps_us, [900_000, 900_000])
        self.assertEqual(spin.midpoint_timestamp_us, 900_000)


class CollisionTests(unittest.TestCase):
    def setUp(self):
        self.config = BEVConfig(0.0, 10.0, -5.0, 5.0, 0.5)
        self.vehicle = VehicleDimensions(2.0, 1.0, 1.5, 0.0)

    def test_footprint_is_oriented_about_rear_axle_pose(self):
        longitudinal = footprint_mask(5.0, 0.0, 0.0, self.vehicle, self.config)
        lateral = footprint_mask(5.0, 0.0, np.pi / 2, self.vehicle, self.config)
        self.assertEqual(longitudinal.sum(), lateral.sum())
        self.assertFalse(np.array_equal(longitudinal, lateral))

    def test_collision_metrics_rank_clear_candidate_first(self):
        horizons = np.array([1.0, 2.0], dtype=np.float32)
        occupancy = np.zeros((2, *self.config.shape), dtype=np.uint8)
        obstacle_row = int(
            np.floor((0.0 - self.config.y_min_m) / self.config.resolution_m)
        )
        obstacle_col = int(
            np.floor((2.0 - self.config.x_min_m) / self.config.resolution_m)
        )
        occupancy[:, obstacle_row, obstacle_col] = 1
        candidate_xyz = np.array(
            [
                [[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[2.0, 3.0, 0.0], [2.0, 3.0, 0.0]],
            ],
            dtype=np.float32,
        )
        yaw = np.zeros((2, 2), dtype=np.float32)

        risks, safest_idx = evaluate_candidate_risks(
            occupancy,
            np.ones_like(occupancy),
            horizons,
            candidate_xyz,
            yaw,
            np.array([1.0, 2.0], dtype=np.float32),
            self.vehicle,
            self.config,
        )

        self.assertEqual(safest_idx, 1)
        self.assertEqual(risks[0].conflict_horizons, 2)
        self.assertEqual(risks[0].first_conflict_s, 1.0)
        self.assertEqual(risks[0].min_clearance_m, 0.0)
        self.assertGreater(risks[0].collision_exposure, 0.0)
        self.assertEqual(risks[1].conflict_horizons, 0)
        self.assertGreater(risks[1].min_clearance_m, 0.0)
        self.assertEqual([risk.oracle_rank for risk in risks], [1, 0])

    def test_out_of_bounds_candidate_cannot_win_on_unknown_space(self):
        horizons = np.array([1.0, 2.0], dtype=np.float32)
        occupancy = np.zeros((2, *self.config.shape), dtype=np.uint8)
        obstacle_row = int((0.0 - self.config.y_min_m) / self.config.resolution_m)
        obstacle_col = int((2.0 - self.config.x_min_m) / self.config.resolution_m)
        occupancy[:, obstacle_row, obstacle_col] = 1
        candidate_xyz = np.array(
            [
                [[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[20.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
            ],
            dtype=np.float32,
        )

        risks, safest_idx = evaluate_candidate_risks(
            occupancy,
            np.ones_like(occupancy),
            horizons,
            candidate_xyz,
            np.zeros((2, 2), dtype=np.float32),
            horizons,
            self.vehicle,
            self.config,
        )

        self.assertEqual(safest_idx, 0)
        self.assertEqual(risks[0].out_of_bounds_horizons, 0)
        self.assertEqual(risks[1].out_of_bounds_horizons, 2)
        self.assertEqual(risks[1].out_of_bounds_fraction, 1.0)
        self.assertFalse(
            footprint_within_bev(20.0, 0.0, 0.0, self.vehicle, self.config)
        )

    def test_unobserved_candidate_cannot_win_on_unknown_space(self):
        horizons = np.array([1.0, 2.0], dtype=np.float32)
        occupancy = np.zeros((2, *self.config.shape), dtype=np.uint8)
        observed = np.ones_like(occupancy)
        unknown = footprint_mask(7.0, 0.0, 0.0, self.vehicle, self.config)
        observed[:, unknown] = 0
        candidate_xyz = np.array(
            [
                [[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[7.0, 0.0, 0.0], [7.0, 0.0, 0.0]],
            ],
            dtype=np.float32,
        )

        risks, safest_idx = evaluate_candidate_risks(
            occupancy,
            observed,
            horizons,
            candidate_xyz,
            np.zeros((2, 2), dtype=np.float32),
            horizons,
            self.vehicle,
            self.config,
        )

        self.assertEqual(safest_idx, 0)
        self.assertEqual(risks[0].unobserved_horizons, 0)
        self.assertEqual(risks[0].observed_fraction, 1.0)
        self.assertEqual(risks[1].unobserved_horizons, 2)
        self.assertEqual(risks[1].observed_fraction, 0.0)

    def test_future_horizon_must_have_candidate_support(self):
        occupancy = np.zeros((1, *self.config.shape), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "within the candidate timestamp range"):
            evaluate_candidate_risks(
                occupancy,
                np.ones_like(occupancy),
                np.array([2.0], dtype=np.float32),
                np.zeros((1, 1, 3), dtype=np.float32),
                np.zeros((1, 1), dtype=np.float32),
                np.array([1.0], dtype=np.float32),
                self.vehicle,
                self.config,
            )

    def test_tangent_yaw_carries_heading_through_stationary_points(self):
        xyz = np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]])
        yaw = infer_tangent_yaw(xyz)
        np.testing.assert_allclose(yaw, np.pi / 2, atol=1e-6)


class ClipFrameTests(unittest.TestCase):
    def test_null_and_duplicate_clip_identities_fail(self):
        with self.assertRaisesRegex(ValueError, "must not contain null"):
            validate_clip_frame(pd.DataFrame({"clip_id": [None], "t0_us": [1]}))
        duplicate = pd.DataFrame(
            {"clip_id": ["clip", "clip"], "t0_us": [1, 1]}
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_clip_frame(duplicate)


class SerializationTests(unittest.TestCase):
    def _config(self, bev_config):
        return oracle_config_payload(
            horizons_s=[1.0],
            history_offsets_s=[-1.0, 0.0],
            bev_config=bev_config,
            filter_config=PointFilterConfig(),
            spin_tolerance_s=0.2,
            candidate_dt_s=0.1,
            observed_ray_stride=16,
            footprint_padding_m=0.0,
        )

    def test_npz_schema_is_pickle_free_and_atomic(self):
        config = BEVConfig(0.0, 2.0, -1.0, 1.0, 1.0)
        vehicle = VehicleDimensions(1.0, 1.0, 1.5, 0.0)
        occupancy = np.zeros((1, 2, 2), dtype=np.uint8)
        observed = np.ones((1, 2, 2), dtype=np.uint8)
        candidate_xyz = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
        candidate_yaw = np.zeros((1, 1), dtype=np.float32)
        candidate_times = np.array([1.0], dtype=np.float32)
        risks, safest_idx = evaluate_candidate_risks(
            occupancy,
            observed,
            candidate_times,
            candidate_xyz,
            candidate_yaw,
            candidate_times,
            vehicle,
            config,
        )
        result = ClipOracleResult(
            past_occupancy=np.zeros((2, 2, 2), dtype=np.uint8),
            past_observed=np.ones((2, 2, 2), dtype=np.uint8),
            past_occupancy_count=np.zeros((2, 2, 2), dtype=np.uint16),
            history_offsets_s=np.array([-1.0, 0.0], dtype=np.float32),
            history_spin_timestamps_us=np.array([0, 1_000_000], dtype=np.int64),
            occupancy=occupancy,
            observed=observed,
            occupancy_count=np.zeros((1, 2, 2), dtype=np.uint16),
            horizons_s=np.array([1.0], dtype=np.float32),
            spin_timestamps_us=np.array([2_000_000], dtype=np.int64),
            candidate_xyz=candidate_xyz,
            candidate_yaw=candidate_yaw,
            candidate_times_s=candidate_times,
            risks=risks,
            oracle_safest_idx=safest_idx,
            vehicle_dimensions=vehicle,
            yaw_source="pred_yaw",
            candidate_config_fingerprint="candidate-config",
            candidate_artifact_sha256="",
        )
        oracle_config = self._config(config)
        with tempfile.TemporaryDirectory() as directory:
            candidate_path = Path(directory) / "candidate.npz"
            np.savez_compressed(
                candidate_path,
                clip_id=np.asarray("clip"),
                t0_us=np.asarray(1_000_000, dtype=np.int64),
                config_fingerprint=np.asarray("candidate-config"),
                pred_xyz=candidate_xyz,
                pred_yaw=candidate_yaw,
                candidate_times_s=candidate_times,
            )
            candidate_ref = CandidateArtifactRef(
                candidate_path,
                "candidate-config",
                file_sha256(candidate_path),
            )
            result = ClipOracleResult(
                **{
                    **result.__dict__,
                    "candidate_artifact_sha256": candidate_ref.content_sha256,
                }
            )
            arrays = result_arrays(
                result, "clip", 1_000_000, config, "revision", oracle_config
            )
            path = Path(directory) / "clip.oracle.npz"
            atomic_save_npz(path, arrays)
            self.assertTrue(path.is_file())
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])
            with np.load(path, allow_pickle=False) as saved:
                self.assertEqual(str(saved["frame"]), "ego_at_t0")
                self.assertEqual(saved["occupancy"].dtype, np.uint8)
                self.assertEqual(saved["past_occupancy"].shape, (2, 2, 2))
                self.assertEqual(
                    str(saved["oracle_config_fingerprint"]),
                    config_fingerprint(oracle_config),
                )
                self.assertEqual(saved["candidate_xyz"].shape, (1, 1, 3))

            resumed = validate_existing_output(
                path,
                "clip",
                1_000_000,
                "revision",
                oracle_config,
                candidate_ref,
            )
            self.assertTrue(resumed["resumed"])

            rounded = dict(arrays)
            rounded["candidate_yaw"] = rounded["candidate_yaw"].copy()
            rounded["candidate_yaw"][0, 0] = np.nextafter(
                rounded["candidate_yaw"][0, 0], np.float32(np.inf)
            )
            atomic_save_npz(path, rounded)
            self.assertTrue(
                validate_existing_output(
                    path,
                    "clip",
                    1_000_000,
                    "revision",
                    oracle_config,
                    candidate_ref,
                )["resumed"]
            )

            changed = dict(oracle_config)
            changed["history_offsets_s"] = [-0.5, 0.0]
            with self.assertRaisesRegex(ValueError, "configuration"):
                validate_existing_output(
                    path,
                    "clip",
                    1_000_000,
                    "revision",
                    changed,
                    candidate_ref,
                )

            with self.assertRaisesRegex(ValueError, "changed after indexing"):
                wrong_ref = CandidateArtifactRef(
                    candidate_path,
                    "candidate-config",
                    "0" * 64,
                )
                validate_existing_output(
                    path,
                    "clip",
                    1_000_000,
                    "revision",
                    oracle_config,
                    wrong_ref,
                )

            corrupted = dict(arrays)
            corrupted["occupancy"] = corrupted["occupancy"].copy()
            corrupted["occupancy"][0, 0, 0] = 2
            atomic_save_npz(path, corrupted)
            with self.assertRaisesRegex(ValueError, "non-binary"):
                validate_existing_output(
                    path,
                    "clip",
                    1_000_000,
                    "revision",
                    oracle_config,
                    candidate_ref,
                )

    def test_candidate_records_resolve_hash_named_artifacts_and_check_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "records").mkdir()
            (root / "artifacts").mkdir()
            artifact_path = root / "artifacts" / "1a2b3c.npz"
            np.savez_compressed(
                artifact_path,
                clip_id=np.asarray("clip-a"),
                t0_us=np.asarray(42, dtype=np.int64),
                config_fingerprint=np.asarray("candidate-config"),
                pred_xyz=np.zeros((1, 2, 3), dtype=np.float32),
            )
            (root / "records" / "1a2b3c.json").write_text(
                '{"artifact_path":"artifacts/1a2b3c.npz",'
                '"clip_id":"clip-a","config_fingerprint":"candidate-config",'
                '"t0_us":42}',
                encoding="utf-8",
            )

            index = load_candidate_record_index(root)
            reference = index[("clip-a", 42)]
            self.assertEqual(
                reference,
                CandidateArtifactRef(
                    artifact_path, "candidate-config", file_sha256(artifact_path)
                ),
            )
            validate_candidate_artifact(reference, "clip-a", 42)
            with self.assertRaisesRegex(ValueError, "does not match"):
                validate_candidate_artifact(reference, "clip-b", 42)
            np.savez_compressed(
                artifact_path,
                clip_id=np.asarray("clip-a"),
                t0_us=np.asarray(42, dtype=np.int64),
                config_fingerprint=np.asarray("candidate-config"),
                pred_xyz=np.ones((1, 2, 3), dtype=np.float32),
            )
            with self.assertRaisesRegex(ValueError, "changed after indexing"):
                validate_candidate_artifact(reference, "clip-a", 42)


if __name__ == "__main__":
    unittest.main()
