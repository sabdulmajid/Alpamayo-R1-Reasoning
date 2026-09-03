"""Dataset utilities for cached BEV occupancy artifacts.

Each manifest row must contain ``clip_id``, ``t0_us``, ``chunk_id``, and
``artifact_path``. Paths may be absolute or relative to the manifest. The NPZ
schema is validated at load time so preprocessing errors fail before training.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset


TARGET_OCCUPANCY_KEYS = ("occupancy", "future_occupancy")
TARGET_VISIBILITY_KEYS = ("observed", "future_visibility")
PAST_VISIBILITY_KEYS = ("past_observed", "past_visibility")
PAST_HORIZON_KEYS = ("history_offsets_s", "past_horizons_s")
SUPPORTED_ORACLE_SCHEMA_VERSIONS = frozenset({4})


@dataclass(frozen=True)
class OccupancyMetadata:
    """Geometry shared by one cached occupancy example."""

    horizons_s: np.ndarray
    past_horizons_s: np.ndarray
    resolution_m: float
    grid_origin_xy_m: np.ndarray
    coordinate_frame: str


@dataclass(frozen=True)
class ArtifactIdentity:
    """Stable identity embedded in one oracle artifact."""

    clip_id: str
    t0_us: int
    schema_version: int


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL or CSV manifest and resolve relative artifact paths."""

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if manifest_path.suffix.lower() in {".jsonl", ".json"}:
        with manifest_path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    elif manifest_path.suffix.lower() == ".csv":
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError("Manifest must be .jsonl, .json, or .csv")

    normalized: list[dict[str, Any]] = []
    seen_examples: set[tuple[str, int]] = set()
    for row_number, row in enumerate(rows, start=1):
        missing = {"clip_id", "t0_us"}.difference(row)
        if missing:
            raise ValueError(f"Manifest row {row_number} is missing {sorted(missing)}")
        if "artifact_path" not in row and "artifact" not in row:
            raise ValueError(
                f"Manifest row {row_number} needs artifact_path (legacy alias: artifact)"
            )
        if "chunk_id" not in row and "source_chunk" not in row:
            raise ValueError(
                f"Manifest row {row_number} needs chunk_id (or source_chunk) "
                "for leakage-safe splits"
            )
        clip_id = str(row["clip_id"])
        try:
            t0_us = int(row["t0_us"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Manifest row {row_number} has invalid t0_us") from exc
        identity = (clip_id, t0_us)
        if identity in seen_examples:
            raise ValueError(f"Duplicate clip identity in manifest: {identity}")
        seen_examples.add(identity)
        artifact = Path(str(row.get("artifact_path", row.get("artifact")))).expanduser()
        if not artifact.is_absolute():
            artifact = manifest_path.parent / artifact
        normalized.append(
            {
                **row,
                "clip_id": clip_id,
                "t0_us": t0_us,
                "chunk_id": str(row["chunk_id"] if "chunk_id" in row else row["source_chunk"]),
                "artifact_path": str(artifact.resolve()),
            }
        )
    if not normalized:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return normalized


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    """Write manifest rows deterministically with parent directories created."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _require_array(archive: Mapping[str, np.ndarray], name: str, ndim: int) -> np.ndarray:
    if name not in archive:
        raise ValueError(f"Occupancy artifact is missing required array '{name}'")
    array = np.asarray(archive[name])
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {array.shape}")
    return array


def _require_one_of(
    archive: Mapping[str, np.ndarray], names: tuple[str, ...], ndim: int
) -> np.ndarray:
    for name in names:
        if name in archive:
            return _require_array(archive, name, ndim)
    raise ValueError(f"Occupancy artifact is missing one of the required arrays {names}")


def load_occupancy_artifact(path: str | Path) -> dict[str, Any]:
    """Load and validate one occupancy artifact.

    Occupancy and visibility arrays use ``[time, y, x]``. Occupancy is binary;
    visibility is one where target occupancy is observable and should contribute
    to the training loss.
    """

    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Occupancy artifact not found: {artifact_path}")
    with np.load(artifact_path, allow_pickle=False) as archive:
        for identity_key in ("schema_version", "clip_id", "t0_us"):
            if identity_key not in archive:
                raise ValueError(
                    f"Occupancy artifact is missing required identity '{identity_key}'"
                )
        schema_version = int(np.asarray(archive["schema_version"]).reshape(()))
        clip_id = str(np.asarray(archive["clip_id"]).reshape(()))
        t0_us = int(np.asarray(archive["t0_us"]).reshape(()))
        if schema_version not in SUPPORTED_ORACLE_SCHEMA_VERSIONS:
            supported = ", ".join(str(version) for version in sorted(SUPPORTED_ORACLE_SCHEMA_VERSIONS))
            raise ValueError(
                f"Unsupported oracle schema_version {schema_version}; supported: {supported}"
            )
        if not clip_id:
            raise ValueError("Artifact clip_id cannot be empty")
        past = _require_array(archive, "past_occupancy", 3).astype(np.float32)
        future = _require_one_of(archive, TARGET_OCCUPANCY_KEYS, 3).astype(np.float32)
        visibility = _require_one_of(archive, TARGET_VISIBILITY_KEYS, 3).astype(np.float32)
        past_visibility = _require_one_of(archive, PAST_VISIBILITY_KEYS, 3).astype(np.float32)
        if future.shape != visibility.shape:
            raise ValueError(
                f"future_occupancy {future.shape} and future_visibility "
                f"{visibility.shape} must match"
            )
        if past.shape[1:] != future.shape[1:]:
            raise ValueError(
                f"Past grid {past.shape[1:]} and future grid {future.shape[1:]} must match"
            )
        if past_visibility.shape != past.shape:
            raise ValueError("past_observed must match past_occupancy")
        for name, array in (
            ("past_occupancy", past),
            ("occupancy", future),
            ("past_observed", past_visibility),
            ("observed", visibility),
        ):
            if not np.isfinite(array).all() or not np.logical_or(array == 0, array == 1).all():
                raise ValueError(f"{name} must be binary and finite")

        if "horizons_s" not in archive:
            raise ValueError("Occupancy artifact is missing required array 'horizons_s'")
        horizons = np.asarray(archive["horizons_s"], dtype=np.float32)
        if horizons.shape != (future.shape[0],) or not np.all(np.diff(horizons) > 0):
            raise ValueError("horizons_s must be strictly increasing with one value per target")
        if np.any(horizons <= 0):
            raise ValueError("horizons_s must contain future times greater than zero")
        past_horizons = np.asarray(
            _require_one_of(archive, PAST_HORIZON_KEYS, 1), dtype=np.float32
        )
        if past_horizons.shape != (past.shape[0],) or not np.all(np.diff(past_horizons) > 0):
            raise ValueError(
                "past_horizons_s must be strictly increasing with one value per history grid"
            )
        if np.any(past_horizons > 0):
            raise ValueError("past_horizons_s must contain history times at or before zero")

        if "resolution_m" in archive:
            resolution_value = archive["resolution_m"]
        elif "bev_resolution_m" in archive:
            resolution_value = archive["bev_resolution_m"]
        else:
            raise ValueError("Occupancy artifact needs resolution_m or bev_resolution_m")
        resolution = float(np.asarray(resolution_value).reshape(()))
        if "grid_origin_xy_m" in archive:
            origin_value = archive["grid_origin_xy_m"]
        elif "bev_x_min_m" in archive and "bev_y_min_m" in archive:
            origin_value = np.asarray([archive["bev_x_min_m"], archive["bev_y_min_m"]]).reshape(2)
        else:
            raise ValueError(
                "Occupancy artifact needs grid_origin_xy_m or both BEV minimum coordinates"
            )
        origin = np.asarray(origin_value, dtype=np.float32)
        if resolution <= 0 or not np.isfinite(resolution):
            raise ValueError("resolution_m must be finite and positive")
        if origin.shape != (2,) or not np.isfinite(origin).all():
            raise ValueError("grid_origin_xy_m must have shape [2] and be finite")
        if "coordinate_frame" in archive:
            frame_value = archive["coordinate_frame"]
        elif "frame" in archive:
            frame_value = archive["frame"]
        else:
            raise ValueError("Occupancy artifact needs coordinate_frame or frame")
        coordinate_frame = str(np.asarray(frame_value).reshape(()))
        if coordinate_frame != "ego_at_t0":
            raise ValueError(
                f"World-model artifacts must use coordinate_frame='ego_at_t0', got {coordinate_frame!r}"
            )

        result: dict[str, Any] = {
            "past_occupancy": past,
            "future_occupancy": future,
            "future_visibility": visibility,
            "past_visibility": past_visibility,
            "identity": ArtifactIdentity(clip_id, t0_us, schema_version),
            "metadata": OccupancyMetadata(
                horizons, past_horizons, resolution, origin, coordinate_frame
            ),
        }
        if "future_ego_motion" in archive:
            ego_motion = np.asarray(archive["future_ego_motion"], dtype=np.float32)
            if ego_motion.shape != (future.shape[0], 3):
                raise ValueError("future_ego_motion must have shape [horizon, 3]")
            result["future_ego_motion"] = ego_motion
    return result


def dataset_provenance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return canonical chunk and artifact-byte provenance for a dataset."""

    entries: list[dict[str, Any]] = []
    for row in sorted(
        rows,
        key=lambda item: (str(item["chunk_id"]), str(item["clip_id"]), int(item["t0_us"])),
    ):
        artifact_path = Path(str(row["artifact_path"]))
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Occupancy artifact not found: {artifact_path}")
        digest = hashlib.sha256()
        with artifact_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        entries.append(
            {
                "artifact_sha256": digest.hexdigest(),
                "chunk_id": str(row["chunk_id"]),
                "clip_id": str(row["clip_id"]),
                "t0_us": int(row["t0_us"]),
            }
        )
    encoded = json.dumps(
        entries, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    chunk_ids = sorted({entry["chunk_id"] for entry in entries})
    return {
        "artifact_count": len(entries),
        "chunk_ids": chunk_ids,
        "dataset_fingerprint": hashlib.sha256(encoded).hexdigest(),
        "artifacts": entries,
    }


def reject_chunk_overlap(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    left_name: str,
    right_name: str,
) -> None:
    """Reject source chunks shared by two dataset provenance records."""

    overlap = sorted(set(left["chunk_ids"]) & set(right["chunk_ids"]))
    if overlap:
        preview = ", ".join(overlap[:10])
        suffix = "" if len(overlap) <= 10 else f" (and {len(overlap) - 10} more)"
        raise ValueError(
            f"{left_name} and {right_name} share source chunks: {preview}{suffix}"
        )


class OccupancyDataset(Dataset[dict[str, Any]]):
    """Lazy dataset backed by per-clip compressed NPZ files."""

    def __init__(
        self,
        manifest: str | Path | list[dict[str, Any]],
        *,
        include_visibility_channel: bool = True,
    ) -> None:
        self.rows = load_manifest(manifest) if not isinstance(manifest, list) else manifest
        self.include_visibility_channel = include_visibility_channel
        if not self.rows:
            raise ValueError("Occupancy dataset cannot be empty")
        reference = load_occupancy_artifact(self.rows[0]["artifact_path"])
        metadata: OccupancyMetadata = reference["metadata"]
        self._schema = {
            "past_shape": reference["past_occupancy"].shape,
            "future_shape": reference["future_occupancy"].shape,
            "horizons_s": metadata.horizons_s,
            "past_horizons_s": metadata.past_horizons_s,
            "resolution_m": metadata.resolution_m,
            "grid_origin_xy_m": metadata.grid_origin_xy_m,
            "coordinate_frame": metadata.coordinate_frame,
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        artifact = load_occupancy_artifact(row["artifact_path"])
        identity: ArtifactIdentity = artifact["identity"]
        if identity.clip_id != row["clip_id"] or identity.t0_us != row["t0_us"]:
            raise ValueError(f"Artifact identity differs for {row['artifact_path']}")
        metadata: OccupancyMetadata = artifact["metadata"]
        if artifact["past_occupancy"].shape != self._schema["past_shape"]:
            raise ValueError(f"Past occupancy shape differs for {row['artifact_path']}")
        if artifact["future_occupancy"].shape != self._schema["future_shape"]:
            raise ValueError(f"Future occupancy shape differs for {row['artifact_path']}")
        for name in ("horizons_s", "past_horizons_s", "grid_origin_xy_m"):
            if not np.allclose(getattr(metadata, name), self._schema[name], rtol=0, atol=1e-6):
                raise ValueError(f"{name} differs for {row['artifact_path']}")
        if not np.isclose(metadata.resolution_m, self._schema["resolution_m"], rtol=0, atol=1e-6):
            raise ValueError(f"resolution_m differs for {row['artifact_path']}")
        if metadata.coordinate_frame != self._schema["coordinate_frame"]:
            raise ValueError(f"coordinate_frame differs for {row['artifact_path']}")
        past = torch.from_numpy(artifact["past_occupancy"])
        if self.include_visibility_channel:
            inputs = torch.stack((past, torch.from_numpy(artifact["past_visibility"])), dim=1)
        else:
            inputs = past.unsqueeze(1)
        item: dict[str, Any] = {
            "inputs": inputs,
            "target": torch.from_numpy(artifact["future_occupancy"]),
            "visibility": torch.from_numpy(artifact["future_visibility"]),
            "horizons_s": torch.from_numpy(metadata.horizons_s),
            "past_horizons_s": torch.from_numpy(metadata.past_horizons_s),
            "grid_origin_xy_m": torch.from_numpy(metadata.grid_origin_xy_m),
            "resolution_m": torch.tensor(metadata.resolution_m, dtype=torch.float32),
            "coordinate_frame": metadata.coordinate_frame,
            "clip_id": row["clip_id"],
            "t0_us": row["t0_us"],
            "chunk_id": row["chunk_id"],
            "artifact_path": row["artifact_path"],
        }
        if "future_ego_motion" in artifact:
            item["future_ego_motion"] = torch.from_numpy(artifact["future_ego_motion"])
        return item
