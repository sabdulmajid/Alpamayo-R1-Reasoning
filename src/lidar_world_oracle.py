"""Recorded-future LiDAR occupancy oracle for trajectory evaluation.

All geometry is expressed in the ego rig frame at ``t0``. PhysicalAI-AV
egomotion provides ``T_world_rig(t)`` and sensor extrinsics provide the static
``T_rig_lidar`` transform. A LiDAR return measured at time ``t`` is transformed
as::

    p_ego@t0 = inv(T_world_rig(t0)) @ T_world_rig(t) @ T_rig_lidar @ p_lidar

The oracle replays recorded future observations. It does not predict how the
world would react to a counterfactual ego trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation

if __package__:
    from .revision_pinned_dataset import pin_streaming_revision
else:
    from revision_pinned_dataset import pin_streaming_revision


DATASET_REVISION = "2ae73f49ffd2b5db43b404201beb7b92889f7afc"
LIDAR_FEATURE = "lidar_top_360fov"
REFERENCE_TIMESTAMP_MODE = "reference-timestamp-rigid-spin"
PER_POINT_TIMESTAMP_MODE = "spin-interval-per-point"
OUTPUT_SCHEMA_VERSION = 4


@dataclass(frozen=True)
class BEVConfig:
    """Metric extent and raster resolution for a t0-frame BEV grid."""

    x_min_m: float = -20.0
    x_max_m: float = 80.0
    y_min_m: float = -40.0
    y_max_m: float = 40.0
    resolution_m: float = 0.5

    def __post_init__(self) -> None:
        if self.x_max_m <= self.x_min_m or self.y_max_m <= self.y_min_m:
            raise ValueError("BEV maxima must be greater than minima")
        if self.resolution_m <= 0:
            raise ValueError("BEV resolution must be positive")
        for extent in (self.x_max_m - self.x_min_m, self.y_max_m - self.y_min_m):
            cells = extent / self.resolution_m
            if not np.isclose(cells, round(cells), atol=1e-8):
                raise ValueError(
                    "BEV extents must be integer multiples of the resolution"
                )

    @property
    def shape(self) -> tuple[int, int]:
        return (
            int(round((self.y_max_m - self.y_min_m) / self.resolution_m)),
            int(round((self.x_max_m - self.x_min_m) / self.resolution_m)),
        )


@dataclass(frozen=True)
class PointFilterConfig:
    """Geometric filters applied after ego-motion compensation."""

    ground_min_z_m: float | None = 0.15
    max_z_m: float | None = 3.5
    max_range_m: float | None = 100.0
    ego_padding_m: float = 0.5

    def __post_init__(self) -> None:
        if self.max_range_m is not None and self.max_range_m <= 0:
            raise ValueError("max_range_m must be positive or None")
        if self.ego_padding_m < 0:
            raise ValueError("ego_padding_m cannot be negative")
        if (
            self.ground_min_z_m is not None
            and self.max_z_m is not None
            and self.ground_min_z_m >= self.max_z_m
        ):
            raise ValueError("ground_min_z_m must be below max_z_m")


@dataclass(frozen=True)
class VehicleDimensions:
    length_m: float
    width_m: float
    height_m: float
    rear_axle_to_center_m: float

    def __post_init__(self) -> None:
        if min(self.length_m, self.width_m, self.height_m) <= 0:
            raise ValueError("Vehicle dimensions must be positive")


@dataclass(frozen=True)
class DecodedSpin:
    points_lidar_m: np.ndarray
    point_timestamps_us: np.ndarray
    start_timestamp_us: int
    end_timestamp_us: int

    @property
    def midpoint_timestamp_us(self) -> int:
        return (self.start_timestamp_us + self.end_timestamp_us) // 2


@dataclass(frozen=True)
class CandidateRisk:
    candidate_idx: int
    collision_exposure: float
    collision_cells: int
    conflict_horizons: int
    min_clearance_m: float
    first_conflict_s: float | None
    oracle_rank: int = -1
    out_of_bounds_horizons: int = 0
    out_of_bounds_fraction: float = 0.0
    observed_fraction: float = 0.0
    unobserved_horizons: int = 0


@dataclass(frozen=True)
class ClipOracleResult:
    past_occupancy: np.ndarray
    past_observed: np.ndarray
    past_occupancy_count: np.ndarray
    history_offsets_s: np.ndarray
    history_spin_timestamps_us: np.ndarray
    occupancy: np.ndarray
    observed: np.ndarray
    occupancy_count: np.ndarray
    horizons_s: np.ndarray
    spin_timestamps_us: np.ndarray
    candidate_xyz: np.ndarray
    candidate_yaw: np.ndarray
    candidate_times_s: np.ndarray
    risks: tuple[CandidateRisk, ...]
    oracle_safest_idx: int
    vehicle_dimensions: VehicleDimensions
    yaw_source: str
    candidate_config_fingerprint: str
    candidate_artifact_sha256: str


@dataclass(frozen=True)
class CandidateArtifactRef:
    """A candidate artifact resolved from its authoritative per-clip record."""

    path: Path
    config_fingerprint: str
    content_sha256: str


def quaternion_transform_matrix(values: pd.Series | dict[str, Any]) -> np.ndarray:
    """Build a homogeneous transform from ``qx,qy,qz,qw,x,y,z`` fields."""

    rotation = Rotation.from_quat(
        [float(values[name]) for name in ("qx", "qy", "qz", "qw")]
    ).as_matrix()
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = [float(values[name]) for name in ("x", "y", "z")]
    return transform


def transform_points_to_t0(
    points_sensor_m: np.ndarray,
    transform_rig_sensor: np.ndarray,
    transforms_world_rig: np.ndarray,
    transform_world_rig_t0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform timestamped sensor points and sensor origins into ego@t0.

    ``transforms_world_rig`` has one pose per input point. The returned tuple is
    ``(points_t0, sensor_origins_t0, points_rig_at_measurement)``.
    """

    points = np.asarray(points_sensor_m, dtype=np.float64)
    point_poses = np.asarray(transforms_world_rig, dtype=np.float64)
    extrinsic = np.asarray(transform_rig_sensor, dtype=np.float64)
    t0_pose = np.asarray(transform_world_rig_t0, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape [N,3], received {points.shape}")
    if point_poses.shape != (len(points), 4, 4):
        raise ValueError(
            "Expected one [4,4] world-from-rig pose per point, received "
            f"{point_poses.shape} for {len(points)} points"
        )
    if extrinsic.shape != (4, 4) or t0_pose.shape != (4, 4):
        raise ValueError("Transforms must have shape [4,4]")

    points_rig = points @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    points_world = np.einsum("nij,nj->ni", point_poses[:, :3, :3], points_rig)
    points_world += point_poses[:, :3, 3]

    sensor_origin_rig = extrinsic[:3, 3]
    origins_world = (
        np.einsum("nij,j->ni", point_poses[:, :3, :3], sensor_origin_rig)
        + point_poses[:, :3, 3]
    )

    world_to_t0_rotation = t0_pose[:3, :3].T
    points_t0 = (points_world - t0_pose[:3, 3]) @ world_to_t0_rotation.T
    origins_t0 = (origins_world - t0_pose[:3, 3]) @ world_to_t0_rotation.T
    return points_t0, origins_t0, points_rig


def rigid_transform_to_matrices(transform: Any) -> np.ndarray:
    """Convert a scipy ``RigidTransform`` (scalar or batched) to matrices."""

    rotation = np.asarray(transform.rotation.as_matrix(), dtype=np.float64)
    translation = np.asarray(transform.translation, dtype=np.float64)
    if rotation.ndim == 2:
        rotation = rotation[None]
        translation = translation.reshape(1, 3)
    matrices = np.broadcast_to(
        np.eye(4, dtype=np.float64), (len(rotation), 4, 4)
    ).copy()
    matrices[:, :3, :3] = rotation
    matrices[:, :3, 3] = translation
    return matrices


def filter_ego_returns(
    points_rig_m: np.ndarray,
    vehicle: VehicleDimensions,
    padding_m: float,
) -> np.ndarray:
    """Return a mask excluding points inside the padded ego bounding box."""

    points = np.asarray(points_rig_m)
    half_length = vehicle.length_m / 2.0 + padding_m
    half_width = vehicle.width_m / 2.0 + padding_m
    half_height = vehicle.height_m / 2.0 + padding_m
    center_x = vehicle.rear_axle_to_center_m
    center_z = vehicle.height_m / 2.0
    inside = (
        (np.abs(points[:, 0] - center_x) <= half_length)
        & (np.abs(points[:, 1]) <= half_width)
        & (np.abs(points[:, 2] - center_z) <= half_height)
    )
    return ~inside


def metric_to_grid(
    points_xy_m: np.ndarray, config: BEVConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map metric xy coordinates to ``(row=y, col=x)`` raster coordinates."""

    points = np.asarray(points_xy_m, dtype=np.float64)
    cols = np.floor((points[:, 0] - config.x_min_m) / config.resolution_m).astype(
        np.int64
    )
    rows = np.floor((points[:, 1] - config.y_min_m) / config.resolution_m).astype(
        np.int64
    )
    height, width = config.shape
    valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    return rows, cols, valid


def rasterize_observed_rays(
    sensor_origins_xy_m: np.ndarray,
    endpoints_xy_m: np.ndarray,
    config: BEVConfig,
    ray_stride: int = 16,
) -> np.ndarray:
    """Rasterize approximate line-of-sight coverage for a subsample of rays."""

    if ray_stride < 1:
        raise ValueError("ray_stride must be at least 1")
    origins = np.asarray(sensor_origins_xy_m, dtype=np.float64)[::ray_stride]
    endpoints = np.asarray(endpoints_xy_m, dtype=np.float64)[::ray_stride]
    observed = np.zeros(config.shape, dtype=np.uint8)
    if len(endpoints) == 0:
        return observed

    start_rows, start_cols, _ = metric_to_grid(origins, config)
    end_rows, end_cols, _ = metric_to_grid(endpoints, config)
    height, width = config.shape
    for row0, col0, row1, col1 in zip(start_rows, start_cols, end_rows, end_cols):
        steps = int(max(abs(row1 - row0), abs(col1 - col0))) + 1
        rows = np.rint(np.linspace(row0, row1, steps)).astype(np.int64)
        cols = np.rint(np.linspace(col0, col1, steps)).astype(np.int64)
        valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        observed[rows[valid], cols[valid]] = 1
    return observed


def rasterize_spin(
    points_t0_m: np.ndarray,
    sensor_origins_t0_m: np.ndarray,
    config: BEVConfig,
    filters: PointFilterConfig,
    ray_stride: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create endpoint occupancy, line-of-sight observation, and hit-count grids."""

    points = np.asarray(points_t0_m, dtype=np.float64)
    origins = np.asarray(sensor_origins_t0_m, dtype=np.float64)
    if points.shape != origins.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points and sensor origins must both have shape [N,3]")

    valid = np.isfinite(points).all(axis=1) & np.isfinite(origins).all(axis=1)
    if filters.max_range_m is not None:
        valid &= (
            np.linalg.norm(points[:, :2] - origins[:, :2], axis=1)
            <= filters.max_range_m
        )
    observed = rasterize_observed_rays(
        origins[valid, :2], points[valid, :2], config, ray_stride=ray_stride
    )

    occupied = valid.copy()
    if filters.ground_min_z_m is not None:
        occupied &= points[:, 2] >= filters.ground_min_z_m
    if filters.max_z_m is not None:
        occupied &= points[:, 2] <= filters.max_z_m

    rows, cols, in_bounds = metric_to_grid(points[occupied, :2], config)
    rows = rows[in_bounds]
    cols = cols[in_bounds]
    counts = np.zeros(config.shape, dtype=np.uint16)
    if len(rows):
        flat = rows * config.shape[1] + cols
        unique, cell_counts = np.unique(flat, return_counts=True)
        count_values = np.minimum(cell_counts, np.iinfo(np.uint16).max).astype(
            np.uint16
        )
        counts.ravel()[unique] = count_values
    occupancy = (counts > 0).astype(np.uint8)
    observed |= occupancy
    return occupancy, observed, counts


def detect_lidar_timestamp_mode(columns: Iterable[str]) -> str:
    """Identify the supported LiDAR timestamp schema without guessing."""

    names = set(columns)
    if {"spin_start_timestamp", "spin_end_timestamp"} <= names:
        return PER_POINT_TIMESTAMP_MODE
    if "reference_timestamp" in names:
        return REFERENCE_TIMESTAMP_MODE
    raise ValueError(
        "LiDAR parquet has neither a reference timestamp nor a spin interval"
    )


def decode_lidar_spin(
    row: pd.Series, timestamp_mode: str | None = None
) -> DecodedSpin:
    """Decode one PhysicalAI-AV Draco point-cloud row with absolute timestamps."""

    try:
        import DracoPy
    except ImportError as exc:  # pragma: no cover - depends on production environment
        raise RuntimeError(
            "DracoPy is required to decode PhysicalAI-AV LiDAR; install "
            "requirements-world-oracle.txt"
        ) from exc

    mode = timestamp_mode or detect_lidar_timestamp_mode(row.index)
    required = {"draco_encoded_pointcloud"}
    if mode == PER_POINT_TIMESTAMP_MODE:
        required.update({"spin_start_timestamp", "spin_end_timestamp"})
    elif mode == REFERENCE_TIMESTAMP_MODE:
        required.add("reference_timestamp")
    else:
        raise ValueError(f"Unsupported LiDAR timestamp mode: {mode}")
    missing = sorted(required - set(row.index))
    if missing:
        raise ValueError(f"LiDAR row is missing columns: {missing}")
    cloud = DracoPy.decode(row["draco_encoded_pointcloud"])
    points = np.asarray(cloud.points, dtype=np.float64)
    if mode == REFERENCE_TIMESTAMP_MODE:
        reference = int(row["reference_timestamp"])
        timestamps = np.full(len(points), reference, dtype=np.int64)
        return DecodedSpin(points, timestamps, reference, reference)

    attributes = {
        attribute["name"]: attribute["data"] for attribute in cloud.attributes
    }
    if "timestamp" not in attributes:
        raise ValueError("Decoded LiDAR cloud has no per-point timestamp attribute")
    timestamps = np.asarray(attributes["timestamp"]).reshape(-1).astype(np.int64)
    if len(timestamps) != len(points):
        raise ValueError(
            f"Per-point timestamp count {len(timestamps)} does not match point count {len(points)}"
        )
    start = int(row["spin_start_timestamp"])
    end = int(row["spin_end_timestamp"])
    if np.any(timestamps < start) or np.any(timestamps > end):
        raise ValueError("Decoded point timestamps fall outside the spin interval")
    return DecodedSpin(points, timestamps, start, end)


def select_spin_rows(
    lidar_df: pd.DataFrame,
    t0_us: int,
    horizons_s: Sequence[float],
    tolerance_s: float,
    timestamp_mode: str | None = None,
) -> list[pd.Series]:
    """Select the nearest complete spin to each requested future horizon."""

    mode = timestamp_mode or detect_lidar_timestamp_mode(lidar_df.columns)
    if mode == PER_POINT_TIMESTAMP_MODE:
        required = {"spin_start_timestamp", "spin_end_timestamp"}
    elif mode == REFERENCE_TIMESTAMP_MODE:
        required = {"reference_timestamp"}
    else:
        raise ValueError(f"Unsupported LiDAR timestamp mode: {mode}")
    missing = sorted(required - set(lidar_df.columns))
    if missing:
        raise ValueError(f"LiDAR parquet is missing columns: {missing}")
    if lidar_df.empty:
        raise ValueError("LiDAR parquet contains no spins")
    if tolerance_s <= 0:
        raise ValueError("Spin tolerance must be positive")
    if mode == PER_POINT_TIMESTAMP_MODE:
        midpoints = (
            lidar_df["spin_start_timestamp"].to_numpy(dtype=np.int64)
            + lidar_df["spin_end_timestamp"].to_numpy(dtype=np.int64)
        ) // 2
    else:
        midpoints = lidar_df["reference_timestamp"].to_numpy(dtype=np.int64)
    selected: list[pd.Series] = []
    tolerance_us = int(round(tolerance_s * 1_000_000))
    for horizon_s in horizons_s:
        target_us = t0_us + int(round(horizon_s * 1_000_000))
        index = int(np.argmin(np.abs(midpoints - target_us)))
        delta_us = abs(int(midpoints[index]) - target_us)
        if delta_us > tolerance_us:
            raise ValueError(
                f"No LiDAR spin within {tolerance_s:.3f}s of horizon {horizon_s:.3f}s "
                f"(nearest delta={delta_us / 1e6:.3f}s)"
            )
        selected.append(lidar_df.iloc[index])
    return selected


def infer_tangent_yaw(candidate_xyz: np.ndarray) -> np.ndarray:
    """Infer heading from path tangents, carrying heading through stationary samples."""

    xyz = np.asarray(candidate_xyz, dtype=np.float64)
    if xyz.ndim != 3 or xyz.shape[-1] < 2:
        raise ValueError(f"Expected candidate_xyz [K,T,3], received {xyz.shape}")
    yaw = np.zeros(xyz.shape[:2], dtype=np.float64)
    for candidate_idx, xy in enumerate(xyz[:, :, :2]):
        if len(xy) == 1:
            continue
        derivative = np.gradient(xy, axis=0)
        speed = np.linalg.norm(derivative, axis=1)
        valid = speed > 1e-4
        if not valid.any():
            continue
        raw = np.arctan2(derivative[:, 1], derivative[:, 0])
        valid_indices = np.flatnonzero(valid)
        nearest = np.abs(np.arange(len(xy))[:, None] - valid_indices[None, :]).argmin(
            axis=1
        )
        yaw[candidate_idx] = np.unwrap(raw[valid_indices[nearest]])
    return yaw.astype(np.float32)


def normalize_candidate_xyz(values: np.ndarray) -> np.ndarray:
    xyz = np.asarray(values)
    if xyz.ndim == 2:
        xyz = xyz[None]
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"Expected pred_xyz [K,T,3], received {xyz.shape}")
    if xyz.shape[0] < 1 or xyz.shape[1] < 1:
        raise ValueError("Candidate trajectories must contain candidates and samples")
    if not np.isfinite(xyz).all():
        raise ValueError("Candidate trajectories contain non-finite values")
    return xyz.astype(np.float32)


def candidate_yaw_from_artifact(
    artifact: Any, candidate_xyz: np.ndarray
) -> tuple[np.ndarray, str]:
    """Load candidate heading from yaw/rotation, or use a documented tangent fallback."""

    candidate_count, steps = candidate_xyz.shape[:2]
    if "pred_yaw" in artifact.files:
        yaw = np.asarray(artifact["pred_yaw"], dtype=np.float64).reshape(
            candidate_count, steps
        )
        return yaw.astype(np.float32), "pred_yaw"
    if "pred_rot" in artifact.files:
        rotations = np.asarray(artifact["pred_rot"])
        if rotations.shape[-2:] == (3, 3):
            rotations = rotations.reshape(candidate_count, steps, 3, 3)
            yaw = np.arctan2(rotations[..., 1, 0], rotations[..., 0, 0])
        elif rotations.shape[-1] == 4:
            quaternions = rotations.reshape(-1, 4)
            yaw = Rotation.from_quat(quaternions).as_euler("xyz")[:, 2]
            yaw = yaw.reshape(candidate_count, steps)
        else:
            raise ValueError(f"Unsupported pred_rot shape {rotations.shape}")
        return np.unwrap(yaw, axis=1).astype(np.float32), "pred_rot"
    return infer_tangent_yaw(candidate_xyz), "path_tangent_fallback"


def footprint_corners(
    x_m: float,
    y_m: float,
    yaw_rad: float,
    vehicle: VehicleDimensions,
    padding_m: float = 0.0,
) -> np.ndarray:
    """Return oriented footprint corners for a rear-axle pose."""

    if padding_m < 0:
        raise ValueError("Footprint padding cannot be negative")
    half_length = vehicle.length_m / 2.0 + padding_m
    half_width = vehicle.width_m / 2.0 + padding_m
    cosine, sine = np.cos(yaw_rad), np.sin(yaw_rad)
    center_x = x_m + cosine * vehicle.rear_axle_to_center_m
    center_y = y_m + sine * vehicle.rear_axle_to_center_m

    corners_local = np.array(
        [
            [-half_length, -half_width],
            [-half_length, half_width],
            [half_length, -half_width],
            [half_length, half_width],
        ]
    )
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return corners_local @ rotation.T + [center_x, center_y]


def footprint_mask(
    x_m: float,
    y_m: float,
    yaw_rad: float,
    vehicle: VehicleDimensions,
    config: BEVConfig,
    padding_m: float = 0.0,
) -> np.ndarray:
    """Rasterize an oriented vehicle footprint whose pose is at the rear axle."""

    corners = footprint_corners(x_m, y_m, yaw_rad, vehicle, padding_m=padding_m)
    cosine, sine = np.cos(yaw_rad), np.sin(yaw_rad)
    center_x = x_m + cosine * vehicle.rear_axle_to_center_m
    center_y = y_m + sine * vehicle.rear_axle_to_center_m
    half_length = vehicle.length_m / 2.0 + padding_m
    half_width = vehicle.width_m / 2.0 + padding_m
    min_x, min_y = corners.min(axis=0)
    max_x, max_y = corners.max(axis=0)

    col0 = max(0, int(np.floor((min_x - config.x_min_m) / config.resolution_m)))
    col1 = min(
        config.shape[1], int(np.ceil((max_x - config.x_min_m) / config.resolution_m))
    )
    row0 = max(0, int(np.floor((min_y - config.y_min_m) / config.resolution_m)))
    row1 = min(
        config.shape[0], int(np.ceil((max_y - config.y_min_m) / config.resolution_m))
    )
    mask = np.zeros(config.shape, dtype=bool)
    if row0 >= row1 or col0 >= col1:
        return mask

    rows, cols = np.mgrid[row0:row1, col0:col1]
    cell_x = config.x_min_m + (cols + 0.5) * config.resolution_m
    cell_y = config.y_min_m + (rows + 0.5) * config.resolution_m
    dx, dy = cell_x - center_x, cell_y - center_y
    local_x = cosine * dx + sine * dy
    local_y = -sine * dx + cosine * dy
    inside = (np.abs(local_x) <= half_length) & (np.abs(local_y) <= half_width)
    mask[rows[inside], cols[inside]] = True
    return mask


def footprint_within_bev(
    x_m: float,
    y_m: float,
    yaw_rad: float,
    vehicle: VehicleDimensions,
    config: BEVConfig,
    padding_m: float = 0.0,
) -> bool:
    """Return whether the complete oriented footprint lies inside the BEV."""

    corners = footprint_corners(x_m, y_m, yaw_rad, vehicle, padding_m=padding_m)
    return bool(
        np.all(corners[:, 0] >= config.x_min_m)
        and np.all(corners[:, 0] <= config.x_max_m)
        and np.all(corners[:, 1] >= config.y_min_m)
        and np.all(corners[:, 1] <= config.y_max_m)
    )


def evaluate_candidate_risks(
    occupancy: np.ndarray,
    observed: np.ndarray,
    horizons_s: np.ndarray,
    candidate_xyz: np.ndarray,
    candidate_yaw: np.ndarray,
    candidate_times_s: np.ndarray,
    vehicle: VehicleDimensions,
    config: BEVConfig,
    footprint_padding_m: float = 0.0,
) -> tuple[tuple[CandidateRisk, ...], int]:
    """Evaluate time-indexed future occupancy under each candidate footprint.

    The exposure is the fraction of observed footprint cells that contain an
    occupied endpoint. ``observed_fraction`` measures coverage over all in-grid
    footprint cells. A horizon is unobserved when at least one in-grid footprint
    cell has no ray observation. A horizon is out of bounds when any part of the
    vehicle footprint leaves the BEV. Candidates with either form of unknown
    exposure rank after candidates with complete coverage. Remaining ties use
    collision and clearance metrics.
    """

    occupancy_values = np.asarray(occupancy)
    observed_values = np.asarray(observed)
    horizons = np.asarray(horizons_s, dtype=np.float64)
    xyz = normalize_candidate_xyz(candidate_xyz)
    yaw = np.asarray(candidate_yaw, dtype=np.float64)
    times = np.asarray(candidate_times_s, dtype=np.float64)
    expected_grid_shape = (len(horizons), *config.shape)
    if occupancy_values.shape != expected_grid_shape:
        raise ValueError(
            f"Expected occupancy [{len(horizons)},{config.shape[0]},{config.shape[1]}], "
            f"received {occupancy_values.shape}"
        )
    if observed_values.shape != expected_grid_shape:
        raise ValueError(
            f"Expected observed [{len(horizons)},{config.shape[0]},{config.shape[1]}], "
            f"received {observed_values.shape}"
        )
    if not np.isin(occupancy_values, (0, 1)).all() or not np.isin(
        observed_values, (0, 1)
    ).all():
        raise ValueError("Occupancy and observed grids must be binary")
    grids = occupancy_values.astype(bool, copy=False)
    observed_grids = observed_values.astype(bool, copy=False)
    if np.any(grids & ~observed_grids):
        raise ValueError("Occupied cells must also be marked observed")
    if yaw.shape != xyz.shape[:2] or times.shape != (xyz.shape[1],):
        raise ValueError(
            "Candidate yaw/timestamp shapes do not match candidate trajectories"
        )
    if (
        not len(horizons)
        or not np.isfinite(horizons).all()
        or np.any(horizons <= 0)
        or np.any(np.diff(horizons) <= 0)
    ):
        raise ValueError("Future horizons must be finite, positive, and increasing")
    if (
        not np.isfinite(yaw).all()
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("Candidate yaw and timestamps must be finite and valid")
    if horizons[0] < times[0] or horizons[-1] > times[-1]:
        raise ValueError(
            "Future horizons must be within the candidate timestamp range "
            f"[{times[0]:.6g}, {times[-1]:.6g}] seconds"
        )

    trajectory_indices = np.abs(times[None, :] - horizons[:, None]).argmin(axis=1)
    distance_grids = []
    for grid in grids:
        if grid.any():
            distance_grids.append(distance_transform_edt(~grid) * config.resolution_m)
        else:
            distance_grids.append(np.full(config.shape, np.inf, dtype=np.float64))

    provisional: list[CandidateRisk] = []
    for candidate_idx in range(len(xyz)):
        collision_cells = 0
        conflict_horizons = 0
        footprint_cells = 0
        observed_footprint_cells = 0
        min_clearance = np.inf
        first_conflict: float | None = None
        out_of_bounds_horizons = 0
        unobserved_horizons = 0
        for horizon_idx, trajectory_idx in enumerate(trajectory_indices):
            pose = (
                float(xyz[candidate_idx, trajectory_idx, 0]),
                float(xyz[candidate_idx, trajectory_idx, 1]),
                float(yaw[candidate_idx, trajectory_idx]),
            )
            if not footprint_within_bev(
                *pose,
                vehicle,
                config,
                padding_m=footprint_padding_m,
            ):
                out_of_bounds_horizons += 1
            footprint = footprint_mask(
                *pose,
                vehicle,
                config,
                padding_m=footprint_padding_m,
            )
            cell_count = int(footprint.sum())
            footprint_cells += cell_count
            if not cell_count:
                continue
            observed_cell_count = int(
                np.count_nonzero(observed_grids[horizon_idx] & footprint)
            )
            observed_footprint_cells += observed_cell_count
            if observed_cell_count < cell_count:
                unobserved_horizons += 1
            overlaps = int(np.count_nonzero(grids[horizon_idx] & footprint))
            collision_cells += overlaps
            if overlaps:
                conflict_horizons += 1
                if first_conflict is None:
                    first_conflict = float(horizons[horizon_idx])
            min_clearance = min(
                min_clearance,
                float(np.min(distance_grids[horizon_idx][footprint])),
            )
        exposure = (
            collision_cells / observed_footprint_cells
            if observed_footprint_cells
            else 0.0
        )
        observed_fraction = (
            observed_footprint_cells / footprint_cells if footprint_cells else 0.0
        )
        provisional.append(
            CandidateRisk(
                candidate_idx=candidate_idx,
                collision_exposure=float(exposure),
                collision_cells=collision_cells,
                conflict_horizons=conflict_horizons,
                min_clearance_m=float(min_clearance),
                first_conflict_s=first_conflict,
                out_of_bounds_horizons=out_of_bounds_horizons,
                out_of_bounds_fraction=out_of_bounds_horizons / len(horizons),
                observed_fraction=float(observed_fraction),
                unobserved_horizons=unobserved_horizons,
            )
        )

    order = sorted(
        range(len(provisional)),
        key=lambda index: (
            provisional[index].out_of_bounds_horizons > 0,
            provisional[index].out_of_bounds_horizons,
            provisional[index].unobserved_horizons > 0,
            provisional[index].unobserved_horizons,
            -provisional[index].observed_fraction,
            provisional[index].conflict_horizons,
            provisional[index].collision_cells,
            provisional[index].collision_exposure,
            -provisional[index].min_clearance_m,
            provisional[index].candidate_idx,
        ),
    )
    ranks = np.empty(len(order), dtype=np.int32)
    ranks[order] = np.arange(len(order), dtype=np.int32)
    risks = tuple(
        CandidateRisk(**{**asdict(risk), "oracle_rank": int(ranks[index])})
        for index, risk in enumerate(provisional)
    )
    return risks, int(order[0])


def _vehicle_dimensions(values: pd.Series) -> VehicleDimensions:
    return VehicleDimensions(
        length_m=float(values["length"]),
        width_m=float(values["width"]),
        height_m=float(values["height"]),
        rear_axle_to_center_m=float(values["rear_axle_to_bbox_center"]),
    )


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for an artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def load_candidate_record_index(
    candidate_dir: Path,
) -> dict[tuple[str, int], CandidateArtifactRef]:
    """Resolve foundation artifacts through ``records/*.json`` metadata.

    Candidate filenames are deliberately hash-derived, so identity is read from
    the record and then checked again inside the referenced NPZ.
    """

    root = candidate_dir.expanduser().resolve()
    record_dir = root / "records"
    if not record_dir.is_dir():
        raise FileNotFoundError(
            f"Candidate output root has no records directory: {record_dir}"
        )
    records = sorted(record_dir.glob("*.json"))
    if not records:
        raise FileNotFoundError(f"No candidate records found in {record_dir}")

    index: dict[tuple[str, int], CandidateArtifactRef] = {}
    for record_path in records:
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            identity = (str(record["clip_id"]), int(record["t0_us"]))
            artifact_value = Path(str(record["artifact_path"]))
            fingerprint = str(record["config_fingerprint"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Invalid candidate record {record_path}: {error}"
            ) from error
        if identity in index:
            raise ValueError(
                f"Duplicate candidate record for {identity[0]}@{identity[1]}"
            )
        if artifact_value.is_absolute():
            raise ValueError(
                f"Candidate record {record_path} must use an output-root-relative artifact_path"
            )
        artifact_path = (root / artifact_value).resolve()
        if not artifact_path.is_relative_to(root):
            raise ValueError(
                f"Candidate record {record_path} references a path outside {root}"
            )
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"Candidate record {record_path} references missing artifact {artifact_path}"
            )
        index[identity] = CandidateArtifactRef(
            artifact_path, fingerprint, file_sha256(artifact_path)
        )
    return index


def validate_candidate_artifact(
    reference: CandidateArtifactRef, clip_id: str, t0_us: int
) -> None:
    """Check record metadata against the embedded, pickle-free NPZ identity."""

    current_sha256 = file_sha256(reference.path)
    if current_sha256 != reference.content_sha256:
        raise ValueError(f"Candidate artifact changed after indexing: {reference.path}")

    with np.load(reference.path, allow_pickle=False) as artifact:
        required = {"clip_id", "t0_us", "config_fingerprint", "pred_xyz"}
        missing = sorted(required - set(artifact.files))
        if missing:
            raise ValueError(
                f"Candidate artifact {reference.path} is missing keys: {missing}"
            )
        embedded_identity = (
            str(np.asarray(artifact["clip_id"]).reshape(())),
            int(np.asarray(artifact["t0_us"]).reshape(())),
        )
        if embedded_identity != (clip_id, t0_us):
            raise ValueError(
                f"Candidate artifact identity {embedded_identity[0]}@{embedded_identity[1]} "
                f"does not match {clip_id}@{t0_us}"
            )
        embedded_fingerprint = str(
            np.asarray(artifact["config_fingerprint"]).reshape(())
        )
        if embedded_fingerprint != reference.config_fingerprint:
            raise ValueError(
                f"Candidate artifact configuration does not match its record: {reference.path}"
            )


def load_candidate_geometry(
    reference: CandidateArtifactRef,
    candidate_dt_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Load validated candidate positions, headings, and timestamps."""

    if candidate_dt_s <= 0:
        raise ValueError("Candidate timestep must be positive")
    with np.load(reference.path, allow_pickle=False) as artifact:
        if "pred_xyz" not in artifact.files:
            raise ValueError(f"Candidate artifact {reference.path} has no pred_xyz")
        candidate_xyz = normalize_candidate_xyz(artifact["pred_xyz"])
        candidate_yaw, yaw_source = candidate_yaw_from_artifact(
            artifact, candidate_xyz
        )
        if "candidate_times_s" in artifact.files:
            candidate_times_s = np.asarray(
                artifact["candidate_times_s"], dtype=np.float32
            )
        else:
            candidate_times_s = (
                np.arange(candidate_xyz.shape[1], dtype=np.float32) + 1.0
            ) * candidate_dt_s
    if candidate_times_s.shape != (candidate_xyz.shape[1],):
        raise ValueError(
            f"Expected candidate_times_s [{candidate_xyz.shape[1]}], "
            f"received {candidate_times_s.shape}"
        )
    if not np.isfinite(candidate_times_s).all() or np.any(
        np.diff(candidate_times_s) <= 0
    ):
        raise ValueError("Candidate timestamps must be finite and strictly increasing")
    return candidate_xyz, candidate_yaw, candidate_times_s, yaw_source


def oracle_config_payload(
    *,
    horizons_s: Sequence[float],
    history_offsets_s: Sequence[float],
    bev_config: BEVConfig,
    filter_config: PointFilterConfig,
    spin_tolerance_s: float,
    candidate_dt_s: float,
    observed_ray_stride: int,
    footprint_padding_m: float,
    lidar_timestamp_mode: str = REFERENCE_TIMESTAMP_MODE,
) -> dict[str, Any]:
    """Return every parameter that changes an oracle artifact."""

    return {
        "dataset_access_mode": "revision-qualified-streaming",
        "lidar_timestamp_mode": lidar_timestamp_mode,
        "horizons_s": [float(value) for value in horizons_s],
        "history_offsets_s": [float(value) for value in history_offsets_s],
        "bev": asdict(bev_config),
        "point_filter": asdict(filter_config),
        "spin_tolerance_s": float(spin_tolerance_s),
        "candidate_dt_s": float(candidate_dt_s),
        "observed_ray_stride": int(observed_ray_stride),
        "footprint_padding_m": float(footprint_padding_m),
    }


def config_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_clip_oracle(
    avdi: Any,
    clip_id: str,
    t0_us: int,
    candidate_artifact: CandidateArtifactRef,
    horizons_s: Sequence[float],
    history_offsets_s: Sequence[float],
    bev_config: BEVConfig,
    filter_config: PointFilterConfig,
    spin_tolerance_s: float = 0.2,
    candidate_dt_s: float = 0.1,
    observed_ray_stride: int = 16,
    footprint_padding_m: float = 0.0,
    lidar_timestamp_mode: str = REFERENCE_TIMESTAMP_MODE,
) -> ClipOracleResult:
    """Load one clip, replay future LiDAR, and rank its trajectory candidates."""

    if not candidate_artifact.path.is_file():
        raise FileNotFoundError(candidate_artifact.path)
    if not horizons_s or any(horizon <= 0 for horizon in horizons_s):
        raise ValueError("Future horizons must be non-empty and positive")
    if any(right <= left for left, right in zip(horizons_s, horizons_s[1:])):
        raise ValueError("Future horizons must be strictly increasing")
    if candidate_dt_s <= 0:
        raise ValueError("Candidate timestep must be positive")
    if not history_offsets_s or any(offset > 0 for offset in history_offsets_s):
        raise ValueError("History offsets must be non-empty and non-positive")
    if any(
        right <= left for left, right in zip(history_offsets_s, history_offsets_s[1:])
    ):
        raise ValueError("History offsets must be strictly increasing")
    validate_candidate_artifact(candidate_artifact, clip_id, t0_us)

    lidar_data = avdi.get_clip_feature(clip_id, LIDAR_FEATURE, maybe_stream=True)
    if "pointclouds" not in lidar_data or not isinstance(
        lidar_data["pointclouds"], pd.DataFrame
    ):
        raise ValueError(
            "PhysicalAI-AV LiDAR feature did not contain a pointclouds DataFrame"
        )
    detected_timestamp_mode = detect_lidar_timestamp_mode(
        lidar_data["pointclouds"].columns
    )
    if detected_timestamp_mode != lidar_timestamp_mode:
        raise ValueError(
            f"LiDAR timestamp mode {detected_timestamp_mode!r} does not match "
            f"the configured mode {lidar_timestamp_mode!r}"
        )
    history_rows = select_spin_rows(
        lidar_data["pointclouds"],
        t0_us,
        history_offsets_s,
        tolerance_s=spin_tolerance_s,
        timestamp_mode=lidar_timestamp_mode,
    )
    future_rows = select_spin_rows(
        lidar_data["pointclouds"],
        t0_us,
        horizons_s,
        tolerance_s=spin_tolerance_s,
        timestamp_mode=lidar_timestamp_mode,
    )
    egomotion = avdi.get_clip_feature(clip_id, "egomotion", maybe_stream=True)
    extrinsics = avdi.get_clip_feature(clip_id, "sensor_extrinsics", maybe_stream=True)
    dimensions = avdi.get_clip_feature(clip_id, "vehicle_dimensions", maybe_stream=True)
    if LIDAR_FEATURE not in extrinsics.index:
        raise ValueError(f"No {LIDAR_FEATURE} row in sensor extrinsics")
    transform_rig_lidar = quaternion_transform_matrix(extrinsics.loc[LIDAR_FEATURE])
    vehicle = _vehicle_dimensions(dimensions)
    transform_world_rig_t0 = rigid_transform_to_matrices(
        egomotion(np.int64(t0_us)).pose
    )[0]

    def rasterize_rows(
        rows: Sequence[pd.Series],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        occupancy_grids = []
        observed_grids = []
        count_grids = []
        spin_timestamps = []
        for row in rows:
            spin = decode_lidar_spin(row, timestamp_mode=lidar_timestamp_mode)
            point_poses = rigid_transform_to_matrices(
                egomotion(spin.point_timestamps_us).pose
            )
            points_t0, origins_t0, points_rig = transform_points_to_t0(
                spin.points_lidar_m,
                transform_rig_lidar,
                point_poses,
                transform_world_rig_t0,
            )
            non_ego = filter_ego_returns(
                points_rig, vehicle, filter_config.ego_padding_m
            )
            occupancy, observed, counts = rasterize_spin(
                points_t0[non_ego],
                origins_t0[non_ego],
                bev_config,
                filter_config,
                ray_stride=observed_ray_stride,
            )
            occupancy_grids.append(occupancy)
            observed_grids.append(observed)
            count_grids.append(counts)
            spin_timestamps.append(spin.midpoint_timestamp_us)
        return (
            np.stack(occupancy_grids),
            np.stack(observed_grids),
            np.stack(count_grids),
            np.asarray(spin_timestamps, dtype=np.int64),
        )

    past_occupancy, past_observed, past_counts, history_spin_timestamps = (
        rasterize_rows(history_rows)
    )
    occupancy_array, observed_array, count_array, future_spin_timestamps = (
        rasterize_rows(future_rows)
    )

    candidate_xyz, candidate_yaw, candidate_times_s, yaw_source = (
        load_candidate_geometry(candidate_artifact, candidate_dt_s)
    )
    risks, safest_idx = evaluate_candidate_risks(
        occupancy_array,
        observed_array,
        np.asarray(horizons_s, dtype=np.float32),
        candidate_xyz,
        candidate_yaw,
        candidate_times_s,
        vehicle,
        bev_config,
        footprint_padding_m=footprint_padding_m,
    )
    return ClipOracleResult(
        past_occupancy=past_occupancy,
        past_observed=past_observed,
        past_occupancy_count=past_counts,
        history_offsets_s=np.asarray(history_offsets_s, dtype=np.float32),
        history_spin_timestamps_us=history_spin_timestamps,
        occupancy=occupancy_array,
        observed=observed_array,
        occupancy_count=count_array,
        horizons_s=np.asarray(horizons_s, dtype=np.float32),
        spin_timestamps_us=future_spin_timestamps,
        candidate_xyz=candidate_xyz,
        candidate_yaw=candidate_yaw,
        candidate_times_s=candidate_times_s,
        risks=risks,
        oracle_safest_idx=safest_idx,
        vehicle_dimensions=vehicle,
        yaw_source=yaw_source,
        candidate_config_fingerprint=candidate_artifact.config_fingerprint,
        candidate_artifact_sha256=candidate_artifact.content_sha256,
    )


def result_arrays(
    result: ClipOracleResult,
    clip_id: str,
    t0_us: int,
    bev_config: BEVConfig,
    dataset_revision: str,
    oracle_config: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Serialize the stable, pickle-free oracle NPZ schema."""

    return {
        "schema_version": np.asarray(OUTPUT_SCHEMA_VERSION, dtype=np.int16),
        "clip_id": np.asarray(clip_id),
        "t0_us": np.asarray(t0_us, dtype=np.int64),
        "dataset_revision": np.asarray(dataset_revision),
        "frame": np.asarray("ego_at_t0"),
        "oracle_config_json": np.asarray(
            json.dumps(oracle_config, sort_keys=True, separators=(",", ":"))
        ),
        "oracle_config_fingerprint": np.asarray(config_fingerprint(oracle_config)),
        "candidate_config_fingerprint": np.asarray(result.candidate_config_fingerprint),
        "candidate_artifact_sha256": np.asarray(result.candidate_artifact_sha256),
        "past_occupancy": result.past_occupancy.astype(np.uint8, copy=False),
        "past_observed": result.past_observed.astype(np.uint8, copy=False),
        "past_occupancy_count": result.past_occupancy_count.astype(
            np.uint16, copy=False
        ),
        "history_offsets_s": result.history_offsets_s.astype(np.float32, copy=False),
        "history_spin_timestamps_us": result.history_spin_timestamps_us.astype(
            np.int64, copy=False
        ),
        "occupancy": result.occupancy.astype(np.uint8, copy=False),
        "observed": result.observed.astype(np.uint8, copy=False),
        "occupancy_count": result.occupancy_count.astype(np.uint16, copy=False),
        "horizons_s": result.horizons_s.astype(np.float32, copy=False),
        "spin_timestamps_us": result.spin_timestamps_us.astype(np.int64, copy=False),
        "candidate_xyz": result.candidate_xyz.astype(np.float32, copy=False),
        "candidate_yaw": result.candidate_yaw.astype(np.float32, copy=False),
        "candidate_times_s": result.candidate_times_s.astype(np.float32, copy=False),
        "candidate_collision_exposure": np.asarray(
            [risk.collision_exposure for risk in result.risks], dtype=np.float32
        ),
        "candidate_collision_cells": np.asarray(
            [risk.collision_cells for risk in result.risks], dtype=np.int32
        ),
        "candidate_conflict_horizons": np.asarray(
            [risk.conflict_horizons for risk in result.risks], dtype=np.int16
        ),
        "candidate_out_of_bounds_horizons": np.asarray(
            [risk.out_of_bounds_horizons for risk in result.risks], dtype=np.int16
        ),
        "candidate_out_of_bounds_fraction": np.asarray(
            [risk.out_of_bounds_fraction for risk in result.risks], dtype=np.float32
        ),
        "candidate_observed_fraction": np.asarray(
            [risk.observed_fraction for risk in result.risks], dtype=np.float32
        ),
        "candidate_unobserved_horizons": np.asarray(
            [risk.unobserved_horizons for risk in result.risks], dtype=np.int16
        ),
        "candidate_min_clearance_m": np.asarray(
            [risk.min_clearance_m for risk in result.risks], dtype=np.float32
        ),
        "candidate_first_conflict_s": np.asarray(
            [
                np.nan if risk.first_conflict_s is None else risk.first_conflict_s
                for risk in result.risks
            ],
            dtype=np.float32,
        ),
        "candidate_oracle_rank": np.asarray(
            [risk.oracle_rank for risk in result.risks], dtype=np.int32
        ),
        "oracle_safest_idx": np.asarray(result.oracle_safest_idx, dtype=np.int32),
        "vehicle_dimensions_m": np.asarray(
            [
                result.vehicle_dimensions.length_m,
                result.vehicle_dimensions.width_m,
                result.vehicle_dimensions.height_m,
                result.vehicle_dimensions.rear_axle_to_center_m,
            ],
            dtype=np.float32,
        ),
        "candidate_yaw_source": np.asarray(result.yaw_source),
        "bev_x_min_m": np.asarray(bev_config.x_min_m, dtype=np.float32),
        "bev_x_max_m": np.asarray(bev_config.x_max_m, dtype=np.float32),
        "bev_y_min_m": np.asarray(bev_config.y_min_m, dtype=np.float32),
        "bev_y_max_m": np.asarray(bev_config.y_max_m, dtype=np.float32),
        "bev_resolution_m": np.asarray(bev_config.resolution_m, dtype=np.float32),
    }


def atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write a compressed NPZ in the target directory and atomically replace it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def validate_existing_output(
    path: Path,
    clip_id: str,
    t0_us: int,
    dataset_revision: str,
    expected_config: dict[str, Any],
    candidate_artifact: CandidateArtifactRef,
) -> dict[str, Any]:
    validate_candidate_artifact(candidate_artifact, clip_id, t0_us)
    with np.load(path, allow_pickle=False) as artifact:
        required = {
            "schema_version",
            "clip_id",
            "t0_us",
            "dataset_revision",
            "frame",
            "oracle_config_json",
            "oracle_config_fingerprint",
            "candidate_config_fingerprint",
            "candidate_artifact_sha256",
            "history_offsets_s",
            "history_spin_timestamps_us",
            "horizons_s",
            "spin_timestamps_us",
            "past_occupancy",
            "past_observed",
            "past_occupancy_count",
            "occupancy",
            "observed",
            "occupancy_count",
            "candidate_xyz",
            "candidate_yaw",
            "candidate_times_s",
            "candidate_collision_exposure",
            "candidate_collision_cells",
            "candidate_conflict_horizons",
            "candidate_out_of_bounds_horizons",
            "candidate_out_of_bounds_fraction",
            "candidate_observed_fraction",
            "candidate_unobserved_horizons",
            "candidate_min_clearance_m",
            "candidate_first_conflict_s",
            "candidate_oracle_rank",
            "oracle_safest_idx",
            "candidate_yaw_source",
            "vehicle_dimensions_m",
            "bev_x_min_m",
            "bev_x_max_m",
            "bev_y_min_m",
            "bev_y_max_m",
            "bev_resolution_m",
        }
        missing = sorted(required - set(artifact.files))
        if missing:
            raise ValueError(f"Existing output {path} is missing keys: {missing}")

        def scalar(name: str) -> Any:
            value = np.asarray(artifact[name])
            if value.shape != ():
                raise ValueError(f"Existing output {name} is not scalar: {path}")
            return value.item()

        if (
            artifact["schema_version"].dtype != np.dtype(np.int16)
            or int(scalar("schema_version")) != OUTPUT_SCHEMA_VERSION
        ):
            raise ValueError(f"Unsupported schema in existing output {path}")
        if (
            str(scalar("clip_id")) != clip_id
            or artifact["t0_us"].dtype != np.dtype(np.int64)
            or int(scalar("t0_us")) != t0_us
        ):
            raise ValueError(f"Existing output identity does not match {path}")
        if str(scalar("dataset_revision")) != dataset_revision:
            raise ValueError(
                f"Existing output dataset revision does not match requested revision: {path}"
            )
        if str(scalar("frame")) != "ego_at_t0":
            raise ValueError(f"Existing output coordinate frame is invalid: {path}")
        expected_fingerprint = config_fingerprint(expected_config)
        expected_config_json = json.dumps(
            expected_config, sort_keys=True, separators=(",", ":")
        )
        if str(scalar("oracle_config_json")) != expected_config_json:
            raise ValueError(
                f"Existing output oracle configuration payload does not match request: {path}"
            )
        saved_fingerprint = str(scalar("oracle_config_fingerprint"))
        if saved_fingerprint != expected_fingerprint:
            raise ValueError(
                f"Existing output oracle configuration does not match request: {path}"
            )
        saved_candidate_fingerprint = str(scalar("candidate_config_fingerprint"))
        if saved_candidate_fingerprint != candidate_artifact.config_fingerprint:
            raise ValueError(
                f"Existing output candidate configuration does not match source: {path}"
            )
        if (
            str(scalar("candidate_artifact_sha256"))
            != candidate_artifact.content_sha256
        ):
            raise ValueError(
                f"Existing output candidate artifact content does not match source: {path}"
            )
        history_offsets = np.asarray(artifact["history_offsets_s"])
        horizons = np.asarray(artifact["horizons_s"])
        if history_offsets.dtype != np.dtype(np.float32) or horizons.dtype != np.dtype(
            np.float32
        ):
            raise ValueError(f"Existing output temporal axis dtype is invalid: {path}")
        if not np.array_equal(
            history_offsets,
            np.asarray(expected_config["history_offsets_s"], dtype=np.float32),
        ):
            raise ValueError(f"Existing output history offsets do not match: {path}")
        if not np.array_equal(
            horizons, np.asarray(expected_config["horizons_s"], dtype=np.float32)
        ):
            raise ValueError(f"Existing output future horizons do not match: {path}")
        expected_shape = BEVConfig(**expected_config["bev"]).shape
        temporal_groups = (
            (
                len(history_offsets),
                "past_occupancy",
                "past_observed",
                "past_occupancy_count",
                "history_spin_timestamps_us",
            ),
            (
                len(horizons),
                "occupancy",
                "observed",
                "occupancy_count",
                "spin_timestamps_us",
            ),
        )
        future_occupancy: np.ndarray | None = None
        future_observed: np.ndarray | None = None
        for (
            size,
            occupied_name,
            observed_name,
            count_name,
            timestamps_name,
        ) in temporal_groups:
            grid_shape = (size, *expected_shape)
            occupied = np.asarray(artifact[occupied_name])
            observed = np.asarray(artifact[observed_name])
            counts = np.asarray(artifact[count_name])
            timestamps = np.asarray(artifact[timestamps_name])
            if (
                occupied.shape != grid_shape
                or observed.shape != grid_shape
                or counts.shape != grid_shape
                or occupied.dtype != np.dtype(np.uint8)
                or observed.dtype != np.dtype(np.uint8)
                or counts.dtype != np.dtype(np.uint16)
                or timestamps.shape != (size,)
                or timestamps.dtype != np.dtype(np.int64)
            ):
                raise ValueError(f"Existing output grid schema is invalid: {path}")
            if (
                not np.isin(occupied, (0, 1)).all()
                or not np.isin(observed, (0, 1)).all()
            ):
                raise ValueError(f"Existing output contains non-binary grids: {path}")
            if not np.array_equal(counts > 0, occupied.astype(bool)):
                raise ValueError(
                    f"Existing output occupancy and hit counts disagree: {path}"
                )
            if np.any(occupied > observed):
                raise ValueError(
                    f"Existing output marks occupied cells as unobserved: {path}"
                )
            if occupied_name == "occupancy":
                future_occupancy = occupied
                future_observed = observed

        candidate_xyz = np.asarray(artifact["candidate_xyz"])
        candidate_yaw = np.asarray(artifact["candidate_yaw"])
        candidate_times = np.asarray(artifact["candidate_times_s"])
        if candidate_xyz.ndim != 3 or candidate_xyz.shape[-1] != 3:
            raise ValueError(f"Existing output candidate geometry is invalid: {path}")
        candidate_count, candidate_steps = candidate_xyz.shape[:2]
        if candidate_count < 1 or candidate_steps < 1:
            raise ValueError(f"Existing output has no candidate samples: {path}")
        if (
            candidate_xyz.dtype != np.dtype(np.float32)
            or candidate_yaw.dtype != np.dtype(np.float32)
            or candidate_times.dtype != np.dtype(np.float32)
            or candidate_yaw.shape != (candidate_count, candidate_steps)
            or candidate_times.shape != (candidate_steps,)
            or not np.isfinite(candidate_xyz).all()
            or not np.isfinite(candidate_yaw).all()
            or not np.isfinite(candidate_times).all()
            or np.any(np.diff(candidate_times) <= 0)
        ):
            raise ValueError(f"Existing output candidate arrays are invalid: {path}")
        (
            source_candidate_xyz,
            source_candidate_yaw,
            source_candidate_times,
            source_yaw_source,
        ) = load_candidate_geometry(
            candidate_artifact, float(expected_config["candidate_dt_s"])
        )
        if (
            not np.array_equal(candidate_xyz, source_candidate_xyz)
            or not np.allclose(
                candidate_yaw,
                source_candidate_yaw,
                rtol=0.0,
                atol=1e-7,
            )
            or not np.array_equal(candidate_times, source_candidate_times)
            or str(scalar("candidate_yaw_source")) != source_yaw_source
        ):
            raise ValueError(
                f"Existing output candidate geometry does not match its exact source: {path}"
            )

        risk_specs = {
            "candidate_collision_exposure": np.float32,
            "candidate_collision_cells": np.int32,
            "candidate_conflict_horizons": np.int16,
            "candidate_out_of_bounds_horizons": np.int16,
            "candidate_out_of_bounds_fraction": np.float32,
            "candidate_observed_fraction": np.float32,
            "candidate_unobserved_horizons": np.int16,
            "candidate_min_clearance_m": np.float32,
            "candidate_first_conflict_s": np.float32,
            "candidate_oracle_rank": np.int32,
        }
        risks: dict[str, np.ndarray] = {}
        for name, dtype in risk_specs.items():
            values = np.asarray(artifact[name])
            if values.shape != (candidate_count,) or values.dtype != np.dtype(dtype):
                raise ValueError(f"Existing output {name} schema is invalid: {path}")
            risks[name] = values
        exposure = risks["candidate_collision_exposure"]
        cells = risks["candidate_collision_cells"]
        conflicts = risks["candidate_conflict_horizons"]
        out_of_bounds = risks["candidate_out_of_bounds_horizons"]
        out_of_bounds_fraction = risks["candidate_out_of_bounds_fraction"]
        observed_fraction = risks["candidate_observed_fraction"]
        unobserved_horizons = risks["candidate_unobserved_horizons"]
        clearance = risks["candidate_min_clearance_m"]
        first_conflict = risks["candidate_first_conflict_s"]
        ranks = risks["candidate_oracle_rank"]
        if (
            not np.isfinite(exposure).all()
            or np.any((exposure < 0) | (exposure > 1))
            or np.any(cells < 0)
            or np.any((conflicts < 0) | (conflicts > len(horizons)))
            or np.any((out_of_bounds < 0) | (out_of_bounds > len(horizons)))
            or not np.isfinite(observed_fraction).all()
            or np.any((observed_fraction < 0) | (observed_fraction > 1))
            or np.any(
                (unobserved_horizons < 0)
                | (unobserved_horizons > len(horizons))
            )
            or not np.allclose(
                out_of_bounds_fraction,
                out_of_bounds / len(horizons),
                rtol=0.0,
                atol=1e-7,
            )
            or np.isnan(clearance).any()
            or np.any(clearance < 0)
            or np.isinf(first_conflict).any()
        ):
            raise ValueError(
                f"Existing output candidate risk values are invalid: {path}"
            )
        if not np.array_equal(np.sort(ranks), np.arange(candidate_count)):
            raise ValueError(f"Existing output candidate ranks are invalid: {path}")
        safest_idx = int(scalar("oracle_safest_idx"))
        if (
            artifact["oracle_safest_idx"].dtype != np.dtype(np.int32)
            or not 0 <= safest_idx < candidate_count
            or ranks[safest_idx] != 0
        ):
            raise ValueError(f"Existing output safest candidate is invalid: {path}")

        vehicle = np.asarray(artifact["vehicle_dimensions_m"])
        if (
            vehicle.dtype != np.dtype(np.float32)
            or vehicle.shape != (4,)
            or not np.isfinite(vehicle).all()
            or np.any(vehicle[:3] <= 0)
        ):
            raise ValueError(f"Existing output vehicle dimensions are invalid: {path}")
        bev_config = BEVConfig(**expected_config["bev"])
        bev_scalars = {
            "bev_x_min_m": bev_config.x_min_m,
            "bev_x_max_m": bev_config.x_max_m,
            "bev_y_min_m": bev_config.y_min_m,
            "bev_y_max_m": bev_config.y_max_m,
            "bev_resolution_m": bev_config.resolution_m,
        }
        for name, expected_value in bev_scalars.items():
            if artifact[name].dtype != np.dtype(np.float32) or not np.isclose(
                float(scalar(name)), expected_value, rtol=0.0, atol=1e-6
            ):
                raise ValueError(f"Existing output BEV metadata is invalid: {path}")

        if future_occupancy is None or future_observed is None:
            raise AssertionError("Future grids were not loaded")
        vehicle_config = VehicleDimensions(*[float(value) for value in vehicle])
        recomputed_risks, recomputed_safest_idx = evaluate_candidate_risks(
            future_occupancy,
            future_observed,
            horizons,
            candidate_xyz,
            candidate_yaw,
            candidate_times,
            vehicle_config,
            bev_config,
            footprint_padding_m=float(expected_config["footprint_padding_m"]),
        )
        expected_risk_arrays = {
            "candidate_collision_exposure": np.asarray(
                [risk.collision_exposure for risk in recomputed_risks],
                dtype=np.float32,
            ),
            "candidate_collision_cells": np.asarray(
                [risk.collision_cells for risk in recomputed_risks], dtype=np.int32
            ),
            "candidate_conflict_horizons": np.asarray(
                [risk.conflict_horizons for risk in recomputed_risks], dtype=np.int16
            ),
            "candidate_out_of_bounds_horizons": np.asarray(
                [risk.out_of_bounds_horizons for risk in recomputed_risks],
                dtype=np.int16,
            ),
            "candidate_out_of_bounds_fraction": np.asarray(
                [risk.out_of_bounds_fraction for risk in recomputed_risks],
                dtype=np.float32,
            ),
            "candidate_observed_fraction": np.asarray(
                [risk.observed_fraction for risk in recomputed_risks],
                dtype=np.float32,
            ),
            "candidate_unobserved_horizons": np.asarray(
                [risk.unobserved_horizons for risk in recomputed_risks],
                dtype=np.int16,
            ),
            "candidate_min_clearance_m": np.asarray(
                [risk.min_clearance_m for risk in recomputed_risks],
                dtype=np.float32,
            ),
            "candidate_first_conflict_s": np.asarray(
                [
                    np.nan if risk.first_conflict_s is None else risk.first_conflict_s
                    for risk in recomputed_risks
                ],
                dtype=np.float32,
            ),
            "candidate_oracle_rank": np.asarray(
                [risk.oracle_rank for risk in recomputed_risks], dtype=np.int32
            ),
        }
        for name, expected_values in expected_risk_arrays.items():
            if not np.allclose(
                risks[name], expected_values, rtol=0.0, atol=1e-7, equal_nan=True
            ):
                raise ValueError(
                    f"Existing output {name} is inconsistent with its grids and candidates: {path}"
                )
        if safest_idx != recomputed_safest_idx:
            raise ValueError(
                f"Existing output safest candidate is inconsistent with its risk values: {path}"
            )

        return {
            "clip_id": clip_id,
            "t0_us": t0_us,
            "candidate_count": candidate_count,
            "oracle_safest_idx": safest_idx,
            "yaw_source": str(scalar("candidate_yaw_source")),
            "resumed": True,
        }


def stable_shard_for_chunk(chunk_id: int, num_shards: int) -> int:
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")
    return int(chunk_id) % num_shards


def validate_clip_frame(clips: pd.DataFrame) -> pd.DataFrame:
    """Validate unique clip identities before any output is written."""

    required = ["clip_id", "t0_us"]
    missing = sorted(set(required) - set(clips.columns))
    if missing:
        raise ValueError(f"Clip parquet is missing columns: {missing}")
    if clips[required].isna().any().any():
        raise ValueError("clip_id and t0_us must not contain null values")
    duplicate = clips.duplicated(required)
    if duplicate.any():
        raise ValueError(
            "Clip parquet contains "
            f"{int(duplicate.sum())} duplicate clip timestamp identities"
        )
    return clips.copy()


def _identity_digest(clip_id: str, t0_us: int) -> str:
    return hashlib.sha256(f"{clip_id}\0{t0_us}".encode()).hexdigest()[:20]


def run_shard(args: argparse.Namespace) -> Path:
    try:
        import physical_ai_av
    except ImportError as exc:  # pragma: no cover - production dependency
        raise RuntimeError(
            "physical_ai_av is required; install requirements-world-oracle.txt"
        ) from exc

    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must satisfy 0 <= index < num-shards")
    horizons = tuple(float(value) for value in args.horizons)
    history_offsets = tuple(float(value) for value in args.history_offsets)
    bev_config = BEVConfig(
        x_min_m=args.x_min,
        x_max_m=args.x_max,
        y_min_m=args.y_min,
        y_max_m=args.y_max,
        resolution_m=args.resolution,
    )
    filter_config = PointFilterConfig(
        ground_min_z_m=None if args.disable_ground_filter else args.ground_min_z,
        max_z_m=args.max_z,
        max_range_m=args.max_range,
        ego_padding_m=args.ego_padding,
    )
    oracle_config = oracle_config_payload(
        horizons_s=horizons,
        history_offsets_s=history_offsets,
        bev_config=bev_config,
        filter_config=filter_config,
        spin_tolerance_s=args.spin_tolerance,
        candidate_dt_s=args.candidate_dt,
        observed_ray_stride=args.observed_ray_stride,
        footprint_padding_m=args.footprint_padding,
        lidar_timestamp_mode=args.lidar_timestamp_mode,
    )
    oracle_fingerprint = config_fingerprint(oracle_config)
    candidate_index = load_candidate_record_index(Path(args.candidate_dir))
    avdi = pin_streaming_revision(
        physical_ai_av.PhysicalAIAVDatasetInterface(
            revision=args.dataset_revision,
            cache_dir=args.cache_dir,
            confirm_download_threshold_gb=float("inf"),
        ),
        args.dataset_revision,
    )
    clips = validate_clip_frame(pd.read_parquet(args.clip_parquet))
    clips["chunk_id"] = [avdi.get_clip_chunk(clip_id) for clip_id in clips["clip_id"]]
    clips = clips[
        clips["chunk_id"].map(
            lambda chunk_id: stable_shard_for_chunk(chunk_id, args.num_shards)
        )
        == args.shard_index
    ].sort_values(["chunk_id", "clip_id", "t0_us"])
    if args.limit is not None:
        clips = clips.head(args.limit)

    output_dir = Path(args.output_dir)
    clip_output_dir = output_dir / "clips"
    manifest_path = (
        output_dir
        / "manifests"
        / (f"shard-{args.shard_index:05d}-of-{args.num_shards:05d}.jsonl")
    )
    manifest_rows: list[dict[str, Any]] = []
    for sequence_idx, row in enumerate(clips.itertuples(index=False), start=1):
        clip_id = str(row.clip_id)
        t0_us = int(row.t0_us)
        identity = (clip_id, t0_us)
        if identity not in candidate_index:
            raise FileNotFoundError(
                f"No candidate record for {clip_id}@{t0_us} in {args.candidate_dir}"
            )
        candidate_artifact = candidate_index[identity]
        validate_candidate_artifact(candidate_artifact, clip_id, t0_us)
        output_path = clip_output_dir / f"{_identity_digest(clip_id, t0_us)}.oracle.npz"
        relative_artifact_path = os.path.relpath(output_path, manifest_path.parent)
        print(
            f"[{sequence_idx}/{len(clips)}] chunk={row.chunk_id} clip={clip_id} t0_us={t0_us}",
            flush=True,
        )
        if args.resume and output_path.is_file():
            manifest_row = validate_existing_output(
                output_path,
                clip_id,
                t0_us,
                args.dataset_revision,
                oracle_config,
                candidate_artifact,
            )
            manifest_row.update(
                {
                    "artifact_path": relative_artifact_path,
                    "candidate_artifact_sha256": candidate_artifact.content_sha256,
                    "chunk_id": int(row.chunk_id),
                    "dataset_revision": args.dataset_revision,
                    "horizons_s": list(horizons),
                    "history_offsets_s": list(history_offsets),
                    "identity_digest": _identity_digest(clip_id, t0_us),
                    "oracle_config_fingerprint": oracle_fingerprint,
                }
            )
            manifest_rows.append(manifest_row)
            continue

        result = build_clip_oracle(
            avdi=avdi,
            clip_id=clip_id,
            t0_us=t0_us,
            candidate_artifact=candidate_artifact,
            horizons_s=horizons,
            history_offsets_s=history_offsets,
            bev_config=bev_config,
            filter_config=filter_config,
            spin_tolerance_s=args.spin_tolerance,
            candidate_dt_s=args.candidate_dt,
            observed_ray_stride=args.observed_ray_stride,
            footprint_padding_m=args.footprint_padding,
            lidar_timestamp_mode=args.lidar_timestamp_mode,
        )
        atomic_save_npz(
            output_path,
            result_arrays(
                result,
                clip_id,
                t0_us,
                bev_config,
                args.dataset_revision,
                oracle_config,
            ),
        )
        manifest_rows.append(
            {
                "artifact_path": relative_artifact_path,
                "candidate_artifact_sha256": candidate_artifact.content_sha256,
                "candidate_count": len(result.risks),
                "chunk_id": int(row.chunk_id),
                "clip_id": clip_id,
                "dataset_revision": args.dataset_revision,
                "horizons_s": list(horizons),
                "history_offsets_s": list(history_offsets),
                "identity_digest": _identity_digest(clip_id, t0_us),
                "oracle_config_fingerprint": oracle_fingerprint,
                "oracle_safest_idx": result.oracle_safest_idx,
                "resumed": False,
                "t0_us": t0_us,
                "yaw_source": result.yaw_source,
            }
        )

    atomic_write_jsonl(manifest_path, manifest_rows)
    print(f"Wrote {len(manifest_rows)} clips and manifest {manifest_path}", flush=True)
    return manifest_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build recorded-future LiDAR BEV oracle artifacts for one chunk-aware shard."
    )
    parser.add_argument("--clip-parquet", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--horizons", nargs="+", type=float, default=[0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
    )
    parser.add_argument(
        "--history-offsets", nargs="+", type=float, default=[-1.0, -0.5, 0.0]
    )
    parser.add_argument("--spin-tolerance", type=float, default=0.2)
    parser.add_argument("--candidate-dt", type=float, default=0.1)
    parser.add_argument("--x-min", type=float, default=-20.0)
    parser.add_argument("--x-max", type=float, default=80.0)
    parser.add_argument("--y-min", type=float, default=-40.0)
    parser.add_argument("--y-max", type=float, default=40.0)
    parser.add_argument("--resolution", type=float, default=0.5)
    parser.add_argument("--ground-min-z", type=float, default=0.15)
    parser.add_argument("--disable-ground-filter", action="store_true")
    parser.add_argument("--max-z", type=float, default=3.5)
    parser.add_argument("--max-range", type=float, default=100.0)
    parser.add_argument("--ego-padding", type=float, default=0.5)
    parser.add_argument("--observed-ray-stride", type=int, default=16)
    parser.add_argument("--footprint-padding", type=float, default=0.0)
    parser.add_argument(
        "--lidar-timestamp-mode",
        choices=[REFERENCE_TIMESTAMP_MODE, PER_POINT_TIMESTAMP_MODE],
        default=REFERENCE_TIMESTAMP_MODE,
    )
    return parser.parse_args(argv)


def main() -> None:
    run_shard(parse_args())


if __name__ == "__main__":
    main()
