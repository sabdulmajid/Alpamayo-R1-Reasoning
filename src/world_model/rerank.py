"""Rerank trajectory candidates using predicted occupancy only."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data import write_jsonl
from .runtime import prediction_filename, sha256_file


@dataclass(frozen=True)
class RerankWeights:
    collision: float = 10.0
    uncertainty: float = 1.0
    out_of_bounds: float = 5.0
    acceleration: float = 0.05
    jerk: float = 0.01
    curvature: float = 0.1
    progress: float = 0.02


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    manifest = Path(path).resolve()
    if manifest.suffix.lower() == ".csv":
        with manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    else:
        with manifest.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    identities: set[tuple[str, int]] = set()
    for row_number, row in enumerate(rows, start=1):
        if "clip_id" not in row or "t0_us" not in row:
            raise ValueError(f"Manifest row {row_number} needs clip_id and t0_us")
        if "artifact_path" not in row and "artifact" not in row:
            raise ValueError(f"Manifest row {row_number} needs artifact_path")
        row["clip_id"] = str(row["clip_id"])
        row["t0_us"] = int(row["t0_us"])
        identity = (row["clip_id"], row["t0_us"])
        if identity in identities:
            raise ValueError(f"Duplicate clip identity in manifest: {identity}")
        identities.add(identity)
        artifact = Path(str(row.get("artifact_path", row.get("artifact"))))
        if not artifact.is_absolute():
            artifact = manifest.parent / artifact
        row["artifact_path"] = str(artifact.resolve())
        if "prediction_path" in row:
            prediction = Path(str(row["prediction_path"]))
            if not prediction.is_absolute():
                row["prediction_path"] = str((manifest.parent / prediction).resolve())
    return rows


def _candidate_headings(xy: np.ndarray) -> np.ndarray:
    difference = np.gradient(xy, axis=0)
    return np.arctan2(difference[:, 1], difference[:, 0]).astype(np.float32)


def _sample_footprint_cells(
    center_xy: np.ndarray,
    yaw: float,
    vehicle_dimensions_m: np.ndarray,
    origin_xy_m: np.ndarray,
    resolution_m: float,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, float]:
    length, width, _, rear_axle_to_center = vehicle_dimensions_m
    center = center_xy + rear_axle_to_center * np.array([np.cos(yaw), np.sin(yaw)])
    longitudinal = np.arange(-length / 2, length / 2 + resolution_m * 0.5, resolution_m)
    lateral = np.arange(-width / 2, width / 2 + resolution_m * 0.5, resolution_m)
    forward, left = np.meshgrid(longitudinal, lateral, indexing="xy")
    cosine, sine = np.cos(yaw), np.sin(yaw)
    x = center[0] + cosine * forward - sine * left
    y = center[1] + sine * forward + cosine * left
    columns = np.floor((x - origin_xy_m[0]) / resolution_m).astype(np.int64)
    rows = np.floor((y - origin_xy_m[1]) / resolution_m).astype(np.int64)
    valid = (rows >= 0) & (rows < shape[0]) & (columns >= 0) & (columns < shape[1])
    valid_fraction = float(valid.mean())
    if not valid.any():
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), 1.0
    cells = np.unique(np.stack((rows[valid], columns[valid]), axis=1), axis=0)
    return cells[:, 0], cells[:, 1], 1.0 - valid_fraction


def trajectory_comfort(xy: np.ndarray, times_s: np.ndarray) -> dict[str, float]:
    if xy.shape[0] < 2 or times_s.shape != (xy.shape[0],):
        raise ValueError("Candidates need at least two positions and one timestamp per position")
    delta_t = np.diff(times_s)
    if np.any(delta_t <= 0):
        raise ValueError("candidate_times_s must be strictly increasing")
    if times_s[0] < 0:
        raise ValueError("candidate_times_s cannot start before zero")
    if times_s[0] > 0:
        xy = np.concatenate((np.zeros((1, 2), dtype=xy.dtype), xy), axis=0)
        times_s = np.concatenate((np.zeros(1, dtype=times_s.dtype), times_s))
        delta_t = np.diff(times_s)
    velocity = np.diff(xy, axis=0) / delta_t[:, None]
    speed = np.linalg.norm(velocity, axis=1)
    if velocity.shape[0] > 1:
        acceleration_vector = np.diff(velocity, axis=0) / delta_t[1:, None]
        acceleration = np.linalg.norm(acceleration_vector, axis=1)
    else:
        acceleration = np.zeros(1)
    if acceleration.shape[0] > 1:
        jerk = np.abs(np.diff(acceleration) / delta_t[2:])
    else:
        jerk = np.zeros(1)
    heading = np.unwrap(np.arctan2(velocity[:, 1], velocity[:, 0]))
    distance = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    if heading.shape[0] > 1:
        curvature = np.abs(np.diff(heading) / np.maximum(distance[1:], 1e-3))
    else:
        curvature = np.zeros(1)
    return {
        "mean_acceleration_mps2": float(np.mean(acceleration)),
        "mean_jerk_mps3": float(np.mean(jerk)),
        "max_curvature_inv_m": float(np.max(curvature)),
        "progress_m": float(xy[-1, 0] - xy[0, 0]),
        "mean_speed_mps": float(np.mean(speed)),
    }


def score_candidate(
    occupancy_probability: np.ndarray,
    horizons_s: np.ndarray,
    origin_xy_m: np.ndarray,
    resolution_m: float,
    candidate_xy: np.ndarray,
    candidate_times_s: np.ndarray,
    candidate_yaw: np.ndarray | None,
    vehicle_dimensions_m: np.ndarray,
    weights: RerankWeights,
) -> dict[str, float]:
    """Compute risk and utility without accessing recorded future occupancy."""

    if occupancy_probability.ndim != 3 or occupancy_probability.shape[0] != horizons_s.size:
        raise ValueError("occupancy_probability must be [horizon,y,x]")
    if candidate_xy.ndim != 2 or candidate_xy.shape[1] != 2:
        raise ValueError("candidate_xy must have shape [time,2]")
    if candidate_times_s.shape != (candidate_xy.shape[0],):
        raise ValueError("candidate_times_s must have one value per trajectory point")
    if np.any(np.diff(candidate_times_s) <= 0):
        raise ValueError("candidate_times_s must be strictly increasing")
    if candidate_times_s[0] > horizons_s[0] or candidate_times_s[-1] < horizons_s[-1]:
        raise ValueError("Candidate time range must cover every occupancy horizon")
    horizon_xy = np.stack(
        [np.interp(horizons_s, candidate_times_s, candidate_xy[:, axis]) for axis in range(2)],
        axis=1,
    )
    if candidate_yaw is None:
        candidate_yaw = _candidate_headings(candidate_xy)
    if candidate_yaw.shape != (candidate_xy.shape[0],):
        raise ValueError("candidate_yaw must have one value per trajectory point")
    horizon_yaw = np.interp(horizons_s, candidate_times_s, np.unwrap(candidate_yaw))
    horizon_risk = np.zeros(horizons_s.size, dtype=np.float64)
    horizon_uncertainty = np.zeros(horizons_s.size, dtype=np.float64)
    horizon_oob = np.ones(horizons_s.size, dtype=np.float64)
    for horizon_index in range(horizons_s.size):
        rows, columns, outside = _sample_footprint_cells(
            horizon_xy[horizon_index],
            float(horizon_yaw[horizon_index]),
            vehicle_dimensions_m,
            origin_xy_m,
            resolution_m,
            occupancy_probability.shape[1:],
        )
        horizon_oob[horizon_index] = outside
        if rows.size:
            probabilities = np.clip(
                occupancy_probability[horizon_index, rows, columns].astype(np.float64), 0.0, 1.0
            )
            horizon_risk[horizon_index] = max(
                horizon_risk[horizon_index], float(probabilities.max())
            )
            clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
            entropy = -(clipped * np.log(clipped) + (1 - clipped) * np.log(1 - clipped)) / np.log(2)
            horizon_uncertainty[horizon_index] = float(entropy.mean())
    collision_risk_proxy = float(1.0 - np.prod(1.0 - horizon_risk))
    uncertainty = float(np.mean(horizon_uncertainty))
    out_of_bounds = float(np.mean(horizon_oob))
    comfort = trajectory_comfort(candidate_xy, candidate_times_s)
    total = (
        weights.collision * collision_risk_proxy
        + weights.uncertainty * uncertainty
        + weights.out_of_bounds * out_of_bounds
        + weights.acceleration * comfort["mean_acceleration_mps2"]
        + weights.jerk * comfort["mean_jerk_mps3"]
        + weights.curvature * comfort["max_curvature_inv_m"]
        - weights.progress * comfort["progress_m"]
    )
    return {
        "score": float(total),
        "collision_risk_proxy": collision_risk_proxy,
        "uncertainty": uncertainty,
        "out_of_bounds_fraction": out_of_bounds,
        **comfort,
    }


def rerank_artifacts(
    prediction_path: str | Path,
    candidate_path: str | Path,
    weights: RerankWeights,
    *,
    expected_clip_id: str | None = None,
    expected_t0_us: int | None = None,
) -> dict[str, Any]:
    candidate_sha256 = sha256_file(candidate_path)
    with np.load(prediction_path, allow_pickle=False) as prediction:
        required = {
            "schema_version",
            "clip_id",
            "t0_us",
            "occupancy_prob",
            "horizons_s",
            "grid_origin_xy_m",
            "resolution_m",
            "coordinate_frame",
            "source_artifact_sha256",
            "producer_method",
            "checkpoint_sha256",
            "prediction_run_fingerprint",
        }
        missing = sorted(required - set(prediction.files))
        if missing:
            raise ValueError(f"Prediction artifact is missing {missing}")
        prediction_clip_id = str(np.asarray(prediction["clip_id"]).reshape(()))
        prediction_t0_us = int(np.asarray(prediction["t0_us"]).reshape(()))
        if int(np.asarray(prediction["schema_version"]).reshape(())) != 2:
            raise ValueError("Unsupported prediction schema version")
        probability = np.asarray(prediction["occupancy_prob"], dtype=np.float32)
        horizons_s = np.asarray(prediction["horizons_s"], dtype=np.float32)
        origin = np.asarray(prediction["grid_origin_xy_m"], dtype=np.float32)
        resolution = float(np.asarray(prediction["resolution_m"]).reshape(()))
        prediction_frame = str(np.asarray(prediction["coordinate_frame"]).reshape(()))
        source_artifact_sha256 = str(
            np.asarray(prediction["source_artifact_sha256"]).reshape(())
        )
        producer_method = str(np.asarray(prediction["producer_method"]).reshape(()))
        checkpoint_sha256 = str(np.asarray(prediction["checkpoint_sha256"]).reshape(()))
        prediction_run_fingerprint = str(
            np.asarray(prediction["prediction_run_fingerprint"]).reshape(())
        )
    if (
        probability.ndim != 3
        or not np.isfinite(probability).all()
        or probability.min() < 0
        or probability.max() > 1
    ):
        raise ValueError("occupancy_prob must be finite [horizon,y,x] values in [0,1]")
    if (
        horizons_s.shape != (probability.shape[0],)
        or not np.isfinite(horizons_s).all()
        or np.any(np.diff(horizons_s) <= 0)
        or np.any(horizons_s <= 0)
    ):
        raise ValueError("Prediction horizons must be strictly increasing")
    if origin.shape != (2,) or not np.isfinite(origin).all() or not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("Prediction grid geometry is invalid")
    with np.load(candidate_path, allow_pickle=False) as candidate:
        required = {
            "clip_id",
            "t0_us",
            "candidate_xyz",
            "candidate_times_s",
            "vehicle_dimensions_m",
            "frame",
            "bev_x_min_m",
            "bev_x_max_m",
            "bev_y_min_m",
            "bev_y_max_m",
            "bev_resolution_m",
        }
        missing = sorted(required - set(candidate.files))
        if missing:
            raise ValueError(f"Candidate artifact is missing {missing}")
        candidate_clip_id = str(np.asarray(candidate["clip_id"]).reshape(()))
        candidate_t0_us = int(np.asarray(candidate["t0_us"]).reshape(()))
        trajectories = np.asarray(candidate["candidate_xyz"], dtype=np.float32)
        if trajectories.ndim != 3 or trajectories.shape[0] == 0 or trajectories.shape[2] < 2:
            raise ValueError("candidate_xyz must have shape [candidate,time,>=2]")
        times_s = np.asarray(candidate["candidate_times_s"], dtype=np.float32)
        yaw = (
            np.asarray(candidate["candidate_yaw"], dtype=np.float32)
            if "candidate_yaw" in candidate
            else None
        )
        dimensions = np.asarray(candidate["vehicle_dimensions_m"], dtype=np.float32)
        candidate_frame = str(np.asarray(candidate["frame"]).reshape(()))
        candidate_origin = np.asarray(
            [candidate["bev_x_min_m"], candidate["bev_y_min_m"]], dtype=np.float32
        ).reshape(2)
        candidate_max = np.asarray(
            [candidate["bev_x_max_m"], candidate["bev_y_max_m"]], dtype=np.float32
        ).reshape(2)
        candidate_resolution = float(np.asarray(candidate["bev_resolution_m"]).reshape(()))
    if prediction_clip_id != candidate_clip_id or prediction_t0_us != candidate_t0_us:
        raise ValueError("Prediction and candidate clip identities differ")
    hex_characters = set("0123456789abcdef")
    for name, value in (
        ("source_artifact_sha256", source_artifact_sha256),
        ("prediction_run_fingerprint", prediction_run_fingerprint),
    ):
        if len(value) != 64 or not set(value).issubset(hex_characters):
            raise ValueError(f"Prediction {name} is not a SHA-256 digest")
    if producer_method not in {"learned", "persistence"}:
        raise ValueError(f"Unsupported prediction producer_method: {producer_method!r}")
    if producer_method == "learned" and (
        len(checkpoint_sha256) != 64 or not set(checkpoint_sha256).issubset(hex_characters)
    ):
        raise ValueError("Learned prediction has no valid checkpoint SHA-256")
    if producer_method == "persistence" and checkpoint_sha256:
        raise ValueError("Persistence prediction must not claim a checkpoint")
    if source_artifact_sha256 != candidate_sha256:
        raise ValueError(
            "Prediction source_artifact_sha256 does not match the candidate artifact bytes"
        )
    if expected_clip_id is not None and prediction_clip_id != expected_clip_id:
        raise ValueError("Prediction clip_id does not match the manifest row")
    if expected_t0_us is not None and prediction_t0_us != expected_t0_us:
        raise ValueError("Prediction t0_us does not match the manifest row")
    frame_aliases = {"t0_ego": "ego_at_t0", "ego_at_t0": "ego_at_t0"}
    if frame_aliases.get(prediction_frame, prediction_frame) != frame_aliases.get(
        candidate_frame, candidate_frame
    ):
        raise ValueError(
            f"Prediction frame '{prediction_frame}' and candidate frame '{candidate_frame}' differ"
        )
    if frame_aliases.get(candidate_frame, candidate_frame) != "ego_at_t0":
        raise ValueError("Reranking currently requires trajectories and occupancy in ego_at_t0")
    expected_max = origin + resolution * np.asarray(
        [probability.shape[2], probability.shape[1]], dtype=np.float32
    )
    if (
        not np.allclose(candidate_origin, origin, rtol=0, atol=1e-5)
        or not np.isclose(candidate_resolution, resolution, rtol=0, atol=1e-6)
        or not np.allclose(candidate_max, expected_max, rtol=0, atol=1e-4)
    ):
        raise ValueError("Prediction and candidate BEV geometry differ")
    if dimensions.shape != (4,):
        raise ValueError("vehicle_dimensions_m must be [length,width,height,rear_axle_to_center]")
    if (
        not np.isfinite(dimensions).all()
        or np.any(dimensions[:3] <= 0)
        or abs(dimensions[3]) > dimensions[0]
    ):
        raise ValueError("Vehicle length, width, and height must be finite and positive")
    if not np.isfinite(trajectories).all() or not np.isfinite(times_s).all():
        raise ValueError("Candidate trajectories and timestamps must be finite")
    if yaw is not None and not np.isfinite(yaw).all():
        raise ValueError("Candidate yaw must be finite")
    if times_s.shape != (trajectories.shape[1],) or np.any(np.diff(times_s) <= 0):
        raise ValueError("candidate_times_s must be strictly increasing with one value per point")
    if yaw is not None and yaw.shape != trajectories.shape[:2]:
        raise ValueError("candidate_yaw must have shape [candidate,time]")
    scores = [
        score_candidate(
            probability,
            horizons_s,
            origin,
            resolution,
            trajectories[index, :, :2],
            times_s,
            yaw[index] if yaw is not None else None,
            dimensions,
            weights,
        )
        for index in range(trajectories.shape[0])
    ]
    world_index = int(np.argmin([score["score"] for score in scores]))
    return {
        "candidate_count": len(scores),
        "clip_id": prediction_clip_id,
        "t0_us": prediction_t0_us,
        "first_candidate_index": 0,
        "world_selected_index": world_index,
        "selection_policy": "predicted_occupancy_v1",
        "producer_method": producer_method,
        "checkpoint_sha256": checkpoint_sha256 or None,
        "prediction_run_fingerprint": prediction_run_fingerprint,
        "candidate_sha256": candidate_sha256,
        "weights": asdict(weights),
        "candidate_scores": scores,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Oracle candidate manifest")
    parser.add_argument("--prediction-dir", help="Defaults to prediction_path in each manifest row")
    parser.add_argument("--output", required=True, help="Output JSONL")
    for field, default in asdict(RerankWeights()).items():
        parser.add_argument(f"--{field.replace('_', '-')}-weight", type=float, default=default)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    weights = RerankWeights(
        **{
            field: getattr(args, f"{field}_weight")
            for field in asdict(RerankWeights())
        }
    )
    if any(not np.isfinite(value) or value < 0 for value in asdict(weights).values()):
        raise ValueError("Reranking weights must be finite and non-negative")
    rows = _load_rows(args.manifest)
    if not rows:
        raise ValueError("Candidate manifest is empty")
    prediction_dir = Path(args.prediction_dir) if args.prediction_dir else None
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        if prediction_dir is not None:
            prediction_path = prediction_dir / prediction_filename(
                row["clip_id"], row["t0_us"]
            )
        elif "prediction_path" in row:
            prediction_path = Path(str(row["prediction_path"]))
        else:
            raise ValueError("Provide --prediction-dir or prediction_path in every row")
        result = rerank_artifacts(
            prediction_path,
            row["artifact_path"],
            weights,
            expected_clip_id=row["clip_id"],
            expected_t0_us=row["t0_us"],
        )
        output_rows.append(
            {
                "prediction_path": str(prediction_path.resolve()),
                "prediction_sha256": sha256_file(prediction_path),
                "candidate_path": str(Path(row["artifact_path"]).resolve()),
                **result,
            }
        )
    write_jsonl(output_rows, args.output)
    print(f"Wrote world-aware selections for {len(rows)} clips to {args.output}")


if __name__ == "__main__":
    main()
