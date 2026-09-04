"""Freeze and verify the data inputs for a world-model benchmark.

The merge step treats the oracle shard manifests as untrusted indexes. It
checks their shard assignment, each oracle artifact, and the corresponding
candidate record before it writes one canonical manifest. The split step then
creates deterministic, chunk-disjoint train, validation, and test manifests.

The checkpoint step selects between completed training runs by validation loss
only. All outputs are deterministic and idempotent: an existing output is
accepted only when its bytes match the requested result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .data import load_manifest, load_occupancy_artifact
from .protocol import (
    checkpoint_training_configuration,
    load_frozen_protocol,
    validate_protocol_provenance,
    validate_training_run_configuration,
)
from .runtime import canonical_fingerprint, sha256_file
from .split_manifest import SPLIT_NAMES, parse_ratios, split_rows


ORACLE_SCHEMA_VERSION = 4
CHECKPOINT_SCHEMA_VERSION = 2
PREPARATION_SCHEMA_VERSION = 1
SELECTION_SCHEMA_VERSION = 2
SOURCE_PARQUET_SHA256 = (
    "c5be3fd1f45574739e051f48d406968bec2495d76bb44d9e46451fcd5a0be3b8"
)
FIXED_RERANKER_WEIGHTS = {
    "collision": 10.0,
    "uncertainty": 1.0,
    "out_of_bounds": 5.0,
    "acceleration": 0.05,
    "jerk": 0.01,
    "curvature": 0.1,
    "progress": 0.02,
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SHARD_PATTERN = re.compile(r"^shard-(\d{5})-of-(\d{5})\.jsonl$")


def _canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _canonical_jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                dict(row), sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            + "\n"
        ).encode()
        for row in rows
    )


def _write_idempotent(path: Path, content: bytes) -> None:
    """Write bytes atomically, or accept an identical existing output."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise FileExistsError(
                f"Refusing to replace non-identical benchmark output: {path}"
            )
        return
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path}:{line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(
                    f"Manifest row must be an object: {path}:{line_number}"
                )
            rows.append(row)
    return rows


def _scalar(archive: Mapping[str, np.ndarray], name: str) -> Any:
    if name not in archive:
        raise ValueError(f"Oracle artifact is missing {name}")
    value = np.asarray(archive[name])
    if value.shape != ():
        raise ValueError(f"Oracle artifact field {name} must be scalar")
    return value.item()


def _identity_digest(clip_id: str, t0_us: int) -> str:
    return hashlib.sha256(f"{clip_id}\0{t0_us}".encode()).hexdigest()[:20]


def _resolve_contained(path_value: Any, *, base: Path, root: Path, label: str) -> Path:
    raw = Path(str(path_value)).expanduser()
    if raw.is_absolute():
        raise ValueError(f"{label} must be a relative path")
    resolved = (base / raw).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes its output root: {path_value}") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _load_candidate_records(
    candidate_dir: Path,
) -> dict[tuple[str, int], dict[str, Any]]:
    candidate_root = candidate_dir.expanduser().resolve()
    record_dir = candidate_root / "records"
    if not record_dir.is_dir():
        raise FileNotFoundError(f"Candidate records directory not found: {record_dir}")
    paths = sorted(record_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"No candidate records found in {record_dir}")

    records: dict[tuple[str, int], dict[str, Any]] = {}
    artifact_paths: set[Path] = set()
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            schema_version = int(record["schema_version"])
            clip_id = str(record["clip_id"])
            t0_us = int(record["t0_us"])
            artifact_sha256 = str(record["artifact_sha256"])
            config_fingerprint = str(record["config_fingerprint"])
            config = record["config"]
            candidates = record["candidates"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid candidate record {path}: {error}") from error
        identity = (clip_id, t0_us)
        if not clip_id or identity in records:
            raise ValueError(f"Invalid or duplicate candidate identity: {identity}")
        if schema_version != 1:
            raise ValueError(f"Unsupported candidate record schema in {path}")
        if not _SHA256_PATTERN.fullmatch(artifact_sha256):
            raise ValueError(f"Candidate record has invalid artifact SHA-256: {path}")
        if not _SHA256_PATTERN.fullmatch(config_fingerprint):
            raise ValueError(
                f"Candidate record has invalid configuration fingerprint: {path}"
            )
        if (
            not isinstance(config, dict)
            or canonical_fingerprint(config) != config_fingerprint
        ):
            raise ValueError(
                f"Candidate record configuration fingerprint does not match: {path}"
            )
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"Candidate record has no candidates: {path}")
        if [candidate.get("candidate_index") for candidate in candidates] != list(
            range(len(candidates))
        ):
            raise ValueError(f"Candidate record indices are invalid: {path}")
        artifact_path = _resolve_contained(
            record.get("artifact_path"),
            base=candidate_root,
            root=candidate_root,
            label=f"Candidate artifact in {path}",
        )
        identity_key = f"{_identity_digest(clip_id, t0_us)}_{t0_us}"
        if (
            path.name != f"{identity_key}.json"
            or artifact_path.name != f"{identity_key}.npz"
        ):
            raise ValueError(
                f"Candidate record or artifact filename differs from {identity}"
            )
        if artifact_path in artifact_paths:
            raise ValueError(
                f"Candidate artifact is referenced more than once: {artifact_path}"
            )
        artifact_paths.add(artifact_path)
        if sha256_file(artifact_path) != artifact_sha256:
            raise ValueError(
                f"Candidate artifact SHA-256 differs from its record: {artifact_path}"
            )
        with np.load(artifact_path, allow_pickle=False) as archive:
            embedded_identity = (
                str(_scalar(archive, "clip_id")),
                int(_scalar(archive, "t0_us")),
            )
            embedded_fingerprint = str(_scalar(archive, "config_fingerprint"))
            if "pred_xyz" not in archive:
                raise ValueError(f"Candidate artifact has no pred_xyz: {artifact_path}")
            candidate_array = np.asarray(archive["pred_xyz"])
            if candidate_array.ndim < 1:
                raise ValueError(
                    f"Candidate artifact pred_xyz is invalid: {artifact_path}"
                )
            candidate_count = int(candidate_array.shape[0])
        if embedded_identity != identity:
            raise ValueError(
                f"Candidate artifact identity differs from its record: {artifact_path}"
            )
        if embedded_fingerprint != config_fingerprint:
            raise ValueError(
                f"Candidate artifact configuration differs from its record: {artifact_path}"
            )
        if candidate_count != len(candidates):
            raise ValueError(
                f"Candidate count differs between record and artifact: {path}"
            )
        records[identity] = {
            "artifact_path": artifact_path,
            "artifact_sha256": artifact_sha256,
            "candidate_count": candidate_count,
            "config_fingerprint": config_fingerprint,
            "record_path": path.resolve(),
            "record_sha256": sha256_file(path),
        }
    return records


def _validate_oracle_row(
    row: dict[str, Any],
    *,
    manifest_path: Path,
    oracle_root: Path,
    shard_index: int,
    shard_count: int,
    candidate_records: Mapping[tuple[str, int], dict[str, Any]],
    expected_candidates: int,
) -> dict[str, Any]:
    required = {
        "artifact_path",
        "candidate_artifact_sha256",
        "candidate_count",
        "chunk_id",
        "clip_id",
        "dataset_revision",
        "horizons_s",
        "history_offsets_s",
        "identity_digest",
        "oracle_config_fingerprint",
        "oracle_safest_idx",
        "t0_us",
        "yaw_source",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"Oracle manifest row in {manifest_path} is missing {missing}")
    try:
        clip_id = str(row["clip_id"])
        t0_us = int(row["t0_us"])
        chunk_id = int(row["chunk_id"])
        candidate_count = int(row["candidate_count"])
        oracle_safest_idx = int(row["oracle_safest_idx"])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Oracle manifest row has invalid numeric fields: {row}"
        ) from error
    identity = (clip_id, t0_us)
    if not clip_id:
        raise ValueError("Oracle manifest clip_id cannot be empty")
    if chunk_id % shard_count != shard_index:
        raise ValueError(
            f"Chunk {chunk_id} is in shard {shard_index}, expected {chunk_id % shard_count}"
        )
    if str(row["identity_digest"]) != _identity_digest(*identity):
        raise ValueError(f"Oracle manifest identity digest is invalid for {identity}")
    if candidate_count != expected_candidates:
        raise ValueError(
            f"Oracle row {identity} has {candidate_count} candidates, expected {expected_candidates}"
        )
    if not 0 <= oracle_safest_idx < candidate_count:
        raise ValueError(f"Oracle safest index is out of range for {identity}")
    candidate_record = candidate_records.get(identity)
    if candidate_record is None:
        raise ValueError(f"Oracle row has no candidate record: {identity}")
    candidate_sha256 = str(row["candidate_artifact_sha256"])
    if candidate_sha256 != candidate_record["artifact_sha256"]:
        raise ValueError(
            f"Oracle row candidate SHA-256 differs from its source: {identity}"
        )
    if candidate_record["candidate_count"] != expected_candidates:
        raise ValueError(f"Candidate source has the wrong candidate count: {identity}")

    artifact_path = _resolve_contained(
        row["artifact_path"],
        base=manifest_path.parent,
        root=oracle_root,
        label=f"Oracle artifact for {identity}",
    )
    clips_root = (oracle_root / "clips").resolve()
    if artifact_path.parent != clips_root:
        raise ValueError(
            f"Oracle artifact must be directly inside {clips_root}: {artifact_path}"
        )
    expected_name = f"{_identity_digest(*identity)}.oracle.npz"
    if artifact_path.name != expected_name:
        raise ValueError(
            f"Oracle artifact filename differs from its identity: {artifact_path}"
        )

    # The shared loader checks the occupancy tensor contract and schema version.
    loaded = load_occupancy_artifact(artifact_path)
    if (loaded["identity"].clip_id, loaded["identity"].t0_us) != identity:
        raise ValueError(
            f"Oracle artifact identity differs from its manifest: {artifact_path}"
        )

    with np.load(artifact_path, allow_pickle=False) as archive:
        embedded_schema = int(_scalar(archive, "schema_version"))
        embedded_revision = str(_scalar(archive, "dataset_revision"))
        embedded_oracle_fingerprint = str(_scalar(archive, "oracle_config_fingerprint"))
        embedded_candidate_fingerprint = str(
            _scalar(archive, "candidate_config_fingerprint")
        )
        embedded_candidate_sha256 = str(_scalar(archive, "candidate_artifact_sha256"))
        embedded_safest = int(_scalar(archive, "oracle_safest_idx"))
        embedded_horizons = np.asarray(archive["horizons_s"], dtype=np.float64)
        embedded_history = np.asarray(archive["history_offsets_s"], dtype=np.float64)
        if "candidate_xyz" not in archive:
            raise ValueError(f"Oracle artifact has no candidate_xyz: {artifact_path}")
        embedded_candidate_count = int(np.asarray(archive["candidate_xyz"]).shape[0])
    if embedded_schema != ORACLE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported oracle schema {embedded_schema}: {artifact_path}"
        )
    scalar_pairs = (
        ("dataset revision", embedded_revision, str(row["dataset_revision"])),
        (
            "oracle configuration fingerprint",
            embedded_oracle_fingerprint,
            str(row["oracle_config_fingerprint"]),
        ),
        ("candidate SHA-256", embedded_candidate_sha256, candidate_sha256),
        (
            "candidate configuration fingerprint",
            embedded_candidate_fingerprint,
            candidate_record["config_fingerprint"],
        ),
    )
    for label, embedded, expected in scalar_pairs:
        if embedded != expected:
            raise ValueError(
                f"Oracle artifact {label} differs from its source: {artifact_path}"
            )
    if not _SHA256_PATTERN.fullmatch(embedded_oracle_fingerprint):
        raise ValueError(
            f"Oracle configuration fingerprint is invalid: {artifact_path}"
        )
    if (
        embedded_candidate_count != candidate_count
        or embedded_safest != oracle_safest_idx
    ):
        raise ValueError(
            f"Oracle candidate metadata differs from its manifest: {artifact_path}"
        )
    for label, embedded, manifest_values in (
        ("horizons", embedded_horizons, row["horizons_s"]),
        ("history offsets", embedded_history, row["history_offsets_s"]),
    ):
        values = np.asarray(manifest_values, dtype=np.float64)
        if embedded.shape != values.shape or not np.array_equal(embedded, values):
            raise ValueError(
                f"Oracle artifact {label} differ from its manifest: {artifact_path}"
            )

    normalized = dict(row)
    normalized.update(
        {
            "artifact_path": str(artifact_path),
            "artifact_sha256": sha256_file(artifact_path),
            "candidate_artifact_path": str(candidate_record["artifact_path"]),
            "candidate_record_path": str(candidate_record["record_path"]),
            "candidate_record_sha256": candidate_record["record_sha256"],
            "chunk_id": str(chunk_id),
            "clip_id": clip_id,
            "t0_us": t0_us,
        }
    )
    return normalized


def merge_oracle_manifests(
    *,
    manifest_dir: Path,
    candidate_dir: Path,
    expected_shards: int,
    expected_rows: int,
    expected_candidates: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate and merge an exact set of sharded oracle manifests."""

    if min(expected_shards, expected_rows, expected_candidates) < 1:
        raise ValueError("Expected shard, row, and candidate counts must be positive")
    manifest_root = manifest_dir.expanduser().resolve()
    if not manifest_root.is_dir():
        raise FileNotFoundError(f"Oracle manifest directory not found: {manifest_root}")
    oracle_root = manifest_root.parent.resolve()
    discovered = sorted(manifest_root.glob("shard-*-of-*.jsonl"))
    expected_paths = [
        manifest_root / f"shard-{index:05d}-of-{expected_shards:05d}.jsonl"
        for index in range(expected_shards)
    ]
    if discovered != expected_paths:
        missing = [str(path) for path in expected_paths if path not in discovered]
        unexpected = [str(path) for path in discovered if path not in expected_paths]
        raise ValueError(
            f"Oracle shard set is not exact; missing={missing}, unexpected={unexpected}"
        )
    candidate_records = _load_candidate_records(candidate_dir)
    if len(candidate_records) != expected_rows:
        raise ValueError(
            f"Found {len(candidate_records)} candidate records, expected {expected_rows}"
        )

    rows: list[dict[str, Any]] = []
    source_manifests: list[dict[str, Any]] = []
    identities: set[tuple[str, int]] = set()
    artifact_paths: set[str] = set()
    for shard_index, path in enumerate(expected_paths):
        match = _SHARD_PATTERN.fullmatch(path.name)
        if match is None or (int(match.group(1)), int(match.group(2))) != (
            shard_index,
            expected_shards,
        ):
            raise AssertionError(f"Internal shard-name mismatch: {path}")
        shard_rows = _read_jsonl(path)
        source_manifests.append(
            {
                "path": str(path),
                "rows": len(shard_rows),
                "sha256": sha256_file(path),
                "shard_index": shard_index,
            }
        )
        for row in shard_rows:
            normalized = _validate_oracle_row(
                row,
                manifest_path=path,
                oracle_root=oracle_root,
                shard_index=shard_index,
                shard_count=expected_shards,
                candidate_records=candidate_records,
                expected_candidates=expected_candidates,
            )
            identity = (normalized["clip_id"], normalized["t0_us"])
            if identity in identities:
                raise ValueError(f"Duplicate oracle clip identity: {identity}")
            identities.add(identity)
            artifact_path = normalized["artifact_path"]
            if artifact_path in artifact_paths:
                raise ValueError(f"Duplicate oracle artifact path: {artifact_path}")
            artifact_paths.add(artifact_path)
            rows.append(normalized)

    if len(rows) != expected_rows:
        raise ValueError(f"Merged {len(rows)} oracle rows, expected {expected_rows}")
    candidate_identities = set(candidate_records)
    if identities != candidate_identities:
        missing = sorted(candidate_identities - identities)[:10]
        unexpected = sorted(identities - candidate_identities)[:10]
        raise ValueError(
            "Oracle and candidate identity sets differ; "
            f"missing={missing}, unexpected={unexpected}"
        )
    rows.sort(key=lambda row: (int(row["chunk_id"]), row["clip_id"], row["t0_us"]))

    invariant_fields = (
        "dataset_revision",
        "oracle_config_fingerprint",
        "horizons_s",
        "history_offsets_s",
        "candidate_count",
    )
    invariants: dict[str, Any] = {}
    for field in invariant_fields:
        encoded_values = {
            json.dumps(
                row[field], sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            for row in rows
        }
        if len(encoded_values) != 1:
            raise ValueError(f"Oracle rows do not share one {field}")
        invariants[field] = rows[0][field]
    candidate_fingerprints = sorted(
        {record["config_fingerprint"] for record in candidate_records.values()}
    )
    if len(candidate_fingerprints) != 1:
        raise ValueError("Candidate records do not share one generation configuration")
    artifact_index = [
        {
            "artifact_sha256": row["artifact_sha256"],
            "candidate_artifact_sha256": row["candidate_artifact_sha256"],
            "chunk_id": row["chunk_id"],
            "clip_id": row["clip_id"],
            "t0_us": row["t0_us"],
        }
        for row in rows
    ]
    summary = {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "stage": "merge_oracle_manifests",
        "expected_shards": expected_shards,
        "expected_rows": expected_rows,
        "expected_candidates": expected_candidates,
        "clips": len(rows),
        "chunks": len({row["chunk_id"] for row in rows}),
        "source_manifests": source_manifests,
        "candidate_directory": str(candidate_dir.expanduser().resolve()),
        "candidate_config_fingerprint": candidate_fingerprints[0],
        "invariants": invariants,
        "dataset_fingerprint": canonical_fingerprint(artifact_index),
    }
    return rows, summary


def _manifest_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(_canonical_jsonl(rows)).hexdigest()


def freeze_splits(
    rows: list[dict[str, Any]], *, seed: int, ratios: Sequence[float]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Return deterministic non-empty chunk-disjoint splits and provenance."""

    normalized_ratios = parse_ratios(ratios)
    splits = split_rows(rows, seed, normalized_ratios)
    identity_sets: dict[str, set[tuple[str, int]]] = {}
    chunk_sets: dict[str, set[str]] = {}
    for name in SPLIT_NAMES:
        if not splits[name]:
            raise ValueError(f"Deterministic split '{name}' is empty")
        identity_sets[name] = {
            (str(row["clip_id"]), int(row["t0_us"])) for row in splits[name]
        }
        chunk_sets[name] = {str(row["chunk_id"]) for row in splits[name]}
    for index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[index + 1 :]:
            if identity_sets[left] & identity_sets[right]:
                raise AssertionError(
                    f"Clip identity leakage between {left} and {right}"
                )
            if chunk_sets[left] & chunk_sets[right]:
                raise AssertionError(f"Chunk leakage between {left} and {right}")
    all_identities = {(str(row["clip_id"]), int(row["t0_us"])) for row in rows}
    if set().union(*identity_sets.values()) != all_identities:
        raise AssertionError("Split manifests do not cover the merged manifest exactly")
    split_summary = {
        name: {
            "clips": len(splits[name]),
            "chunks": len(chunk_sets[name]),
            "chunk_ids": sorted(chunk_sets[name]),
            "manifest_sha256": _manifest_sha256(splits[name]),
            "dataset_fingerprint": canonical_fingerprint(
                [
                    {
                        "artifact_sha256": row["artifact_sha256"],
                        "chunk_id": row["chunk_id"],
                        "clip_id": row["clip_id"],
                        "t0_us": row["t0_us"],
                    }
                    for row in splits[name]
                ]
            ),
        }
        for name in SPLIT_NAMES
    }
    summary: dict[str, Any] = {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "stage": "freeze_chunk_disjoint_splits",
        "algorithm": "sha256(seed:chunk_id), first_64_bits_uniform",
        "seed": seed,
        "ratios": dict(zip(SPLIT_NAMES, normalized_ratios)),
        "source_manifest_sha256": _manifest_sha256(rows),
        "total_clips": len(rows),
        "total_chunks": len(set().union(*chunk_sets.values())),
        "splits": split_summary,
    }
    summary["split_plan_fingerprint"] = canonical_fingerprint(summary)
    return splits, summary


def _checkpoint_record(label: str, run_dir: Path) -> dict[str, Any]:
    checkpoint_path = (run_dir / "best.pt").resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Best checkpoint not found for {label}: {checkpoint_path}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported checkpoint schema for {label}")
    try:
        epoch = int(checkpoint["epoch"])
        best_val_loss = float(checkpoint["best_val_loss"])
        seed = int(checkpoint["seed"])
        history = checkpoint["history"]
        resume_signature = checkpoint["resume_signature"]
        data_provenance = checkpoint["data_provenance"]
        model_config = checkpoint["model_config"]
        data_schema = checkpoint["data_schema"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Checkpoint metadata is incomplete for {label}: {error}"
        ) from error
    if not np.isfinite(best_val_loss) or best_val_loss < 0:
        raise ValueError(f"Checkpoint validation loss is invalid for {label}")
    if not isinstance(history, list) or not history:
        raise ValueError(f"Checkpoint history is empty for {label}")
    matching = [record for record in history if int(record.get("epoch", -1)) == epoch]
    if len(matching) != 1:
        raise ValueError(
            f"Checkpoint epoch is absent or duplicated in history for {label}"
        )
    try:
        epoch_val_loss = float(matching[0]["val"]["loss"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Checkpoint validation history is invalid for {label}"
        ) from error
    if not np.isclose(epoch_val_loss, best_val_loss, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"Best checkpoint does not contain its best validation epoch: {label}"
        )
    if int(resume_signature.get("seed", -1)) != seed:
        raise ValueError(f"Checkpoint seed and resume signature differ for {label}")
    training_configuration = checkpoint_training_configuration(checkpoint)
    try:
        protocol_path = Path(resume_signature["protocol"]["path"])
    except (KeyError, TypeError) as error:
        raise ValueError(f"Checkpoint protocol provenance is missing for {label}") from error
    frozen_protocol = load_frozen_protocol(protocol_path)
    protocol_provenance = validate_protocol_provenance(
        frozen_protocol,
        resume_signature.get("protocol"),
        label=f"Checkpoint {label}",
    )
    validate_training_run_configuration(
        frozen_protocol,
        seed=seed,
        configuration=training_configuration,
    )

    completion_path = (run_dir / "completion.json").resolve()
    try:
        completion_bytes = completion_path.read_bytes()
        completion = json.loads(completion_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid training completion record for {label}") from error
    if not isinstance(completion, dict):
        raise ValueError(f"Training completion record must be an object for {label}")
    completion_fingerprint = str(completion.get("completion_fingerprint", ""))
    completion_payload = dict(completion)
    completion_payload.pop("completion_fingerprint", None)
    if (
        not _SHA256_PATTERN.fullmatch(completion_fingerprint)
        or canonical_fingerprint(completion_payload) != completion_fingerprint
    ):
        raise ValueError(f"Training completion fingerprint is invalid for {label}")
    latest_path = (run_dir / "latest.pt").resolve()
    expected_completion = {
        "schema_version": 1,
        "status": "completed",
        "seed": seed,
        "epochs": training_configuration["epochs"],
        "final_epoch": training_configuration["epochs"] - 1,
        "protocol": protocol_provenance,
        "latest_checkpoint": str(latest_path),
        "latest_checkpoint_sha256": sha256_file(latest_path),
        "best_checkpoint": str(checkpoint_path),
        "best_checkpoint_sha256": sha256_file(checkpoint_path),
    }
    if completion_payload != expected_completion:
        raise ValueError(f"Training completion record does not match run {label}")
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    latest_history = latest.get("history")
    try:
        completed_epochs = [int(record["epoch"]) for record in latest_history]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Latest checkpoint history is invalid for {label}") from error
    if (
        latest.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION
        or int(latest.get("epoch", -1)) != training_configuration["epochs"] - 1
        or completed_epochs != list(range(training_configuration["epochs"]))
        or latest.get("seed") != seed
        or latest.get("resume_signature") != resume_signature
        or latest.get("data_provenance") != data_provenance
        or latest.get("model_config") != model_config
        or latest.get("data_schema") != data_schema
        or checkpoint_training_configuration(latest) != training_configuration
    ):
        raise ValueError(f"Latest checkpoint does not prove completed training for {label}")
    return {
        "label": label,
        "run_directory": str(run_dir.resolve()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "epoch": epoch,
        "seed": seed,
        "best_validation_loss": best_val_loss,
        "train_manifest_sha256": resume_signature.get("train_manifest_sha256"),
        "validation_manifest_sha256": resume_signature.get("val_manifest_sha256"),
        "train_dataset_fingerprint": data_provenance["train"]["dataset_fingerprint"],
        "validation_dataset_fingerprint": data_provenance["val"]["dataset_fingerprint"],
        "model_config": model_config,
        "data_schema": data_schema,
        "training_configuration": training_configuration,
        "protocol": protocol_provenance,
        "training_completion": {
            "path": str(completion_path),
            "sha256": hashlib.sha256(completion_bytes).hexdigest(),
            "completion_fingerprint": completion_fingerprint,
            "latest_checkpoint": str(latest_path),
            "latest_checkpoint_sha256": expected_completion[
                "latest_checkpoint_sha256"
            ],
        },
        "comparison_signature": {
            key: value for key, value in resume_signature.items() if key != "seed"
        },
    }


def select_checkpoint(runs: Sequence[tuple[str, Path]]) -> dict[str, Any]:
    """Select one best checkpoint using validation loss and a stable tie break."""

    if len(runs) < 2:
        raise ValueError("Checkpoint selection needs at least two independent runs")
    labels = [label for label, _ in runs]
    if any(not label for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("Training-run labels must be non-empty and unique")
    records = [
        _checkpoint_record(label, directory.expanduser()) for label, directory in runs
    ]
    reference = records[0]
    comparison_fields = (
        "train_manifest_sha256",
        "validation_manifest_sha256",
        "train_dataset_fingerprint",
        "validation_dataset_fingerprint",
        "model_config",
        "data_schema",
        "training_configuration",
        "protocol",
        "comparison_signature",
    )
    for record in records[1:]:
        for field in comparison_fields:
            if record[field] != reference[field]:
                raise ValueError(f"Training runs differ in {field}")
    seeds = [record["seed"] for record in records]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Training runs must use distinct random seeds")
    records.sort(
        key=lambda record: (
            record["best_validation_loss"],
            record["seed"],
            record["label"],
        )
    )
    selected = records[0]
    payload: dict[str, Any] = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "selection_policy": "minimum_best_validation_loss",
        "test_metrics_used": False,
        "protocol": selected["protocol"],
        "selected_label": selected["label"],
        "selected_seed": selected["seed"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "selected_epoch": selected["epoch"],
        "selected_validation_loss": selected["best_validation_loss"],
        "runs": records,
    }
    payload["selection_fingerprint"] = canonical_fingerprint(payload)
    return payload


def verify_evaluation_partition(
    selection_path: Path, test_manifest: Path
) -> dict[str, Any]:
    """Prove that test chunks occur in neither checkpoint training partition."""

    selection_file = selection_path.expanduser().resolve()
    try:
        selection = json.loads(selection_file.read_text(encoding="utf-8"))
        if selection["schema_version"] != SELECTION_SCHEMA_VERSION:
            raise ValueError("unsupported selection schema")
        checkpoint_path = Path(selection["selected_checkpoint"]).resolve()
        expected_checkpoint_sha256 = str(selection["selected_checkpoint_sha256"])
        fingerprint = str(selection["selection_fingerprint"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Invalid checkpoint selection record: {selection_file}"
        ) from error
    fingerprint_payload = dict(selection)
    fingerprint_payload.pop("selection_fingerprint", None)
    if canonical_fingerprint(fingerprint_payload) != fingerprint:
        raise ValueError("Checkpoint selection fingerprint does not match its payload")
    try:
        protocol_path = Path(selection["protocol"]["path"])
    except (KeyError, TypeError) as error:
        raise ValueError("Checkpoint selection has no protocol provenance") from error
    frozen_protocol = load_frozen_protocol(protocol_path)
    protocol_provenance = validate_protocol_provenance(
        frozen_protocol,
        selection.get("protocol"),
        label="Checkpoint selection record",
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Selected checkpoint not found: {checkpoint_path}")
    if sha256_file(checkpoint_path) != expected_checkpoint_sha256:
        raise ValueError("Selected checkpoint bytes differ from the selection record")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Selected checkpoint has an unsupported schema")
    provenance = checkpoint.get("data_provenance")
    if not isinstance(provenance, dict) or not {"train", "val"}.issubset(provenance):
        raise ValueError(
            "Selected checkpoint has incomplete train and validation provenance"
        )
    rows = load_manifest(test_manifest)
    test_chunks = {str(row["chunk_id"]) for row in rows}
    for partition in ("train", "val"):
        source = provenance[partition]
        if not isinstance(source, dict) or not isinstance(
            source.get("chunk_ids"), list
        ):
            raise ValueError(f"Checkpoint {partition} provenance has no chunk list")
        overlap = sorted(test_chunks & {str(value) for value in source["chunk_ids"]})
        if overlap:
            preview = ", ".join(overlap[:10])
            partition_name = "validation" if partition == "val" else partition
            raise ValueError(
                f"Test data and checkpoint {partition_name} data share source chunks: {preview}"
            )
    return {
        "selection": str(selection_file),
        "selection_sha256": sha256_file(selection_file),
        "selection_fingerprint": fingerprint,
        "protocol": protocol_provenance,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": expected_checkpoint_sha256,
        "test_manifest": str(test_manifest.expanduser().resolve()),
        "test_manifest_sha256": sha256_file(test_manifest),
        "test_clips": len(rows),
        "test_chunks": len(test_chunks),
        "train_test_chunk_overlap": 0,
        "validation_test_chunk_overlap": 0,
    }


def freeze_benchmark_protocol(
    *,
    split_summary_path: Path,
    source_parquet: Path,
    expected_source_parquet_sha256: str = SOURCE_PARQUET_SHA256,
    epochs: int = 30,
    batch_size: int = 4,
    workers: int = 4,
    base_channels: int = 16,
) -> dict[str, Any]:
    """Build the predeclared analysis protocol from verified split outputs."""

    split_summary_file = split_summary_path.expanduser().resolve()
    try:
        split_summary = json.loads(split_summary_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid split summary: {split_summary_file}") from error
    if (
        split_summary.get("schema_version") != PREPARATION_SCHEMA_VERSION
        or split_summary.get("stage") != "freeze_chunk_disjoint_splits"
    ):
        raise ValueError("Split summary has an unsupported schema or stage")
    split_plan_payload = {
        key: split_summary[key]
        for key in (
            "schema_version",
            "stage",
            "algorithm",
            "seed",
            "ratios",
            "source_manifest_sha256",
            "total_clips",
            "total_chunks",
            "splits",
        )
    }
    if canonical_fingerprint(split_plan_payload) != split_summary.get(
        "split_plan_fingerprint"
    ):
        raise ValueError("Split plan fingerprint does not match its summary")
    if split_summary.get("seed") != 2026:
        raise ValueError("Benchmark protocol requires split seed 2026")
    expected_ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
    if split_summary.get("ratios") != expected_ratios:
        raise ValueError("Benchmark protocol requires 0.8,0.1,0.1 split ratios")
    output_directory = Path(str(split_summary.get("output_directory", ""))).resolve()
    manifest_records: dict[str, dict[str, Any]] = {}
    for name in SPLIT_NAMES:
        manifest_path = output_directory / f"{name}.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Frozen {name} manifest not found: {manifest_path}"
            )
        actual_sha256 = sha256_file(manifest_path)
        expected_sha256 = (
            split_summary.get("splits", {}).get(name, {}).get("manifest_sha256")
        )
        if actual_sha256 != expected_sha256:
            raise ValueError(f"Frozen {name} manifest differs from its split summary")
        manifest_records[name] = {
            "path": str(manifest_path),
            "sha256": actual_sha256,
            "clips": int(split_summary["splits"][name]["clips"]),
            "chunks": int(split_summary["splits"][name]["chunks"]),
            "dataset_fingerprint": split_summary["splits"][name]["dataset_fingerprint"],
        }
    source_manifest = Path(str(split_summary.get("source_manifest", ""))).resolve()
    if not source_manifest.is_file():
        raise FileNotFoundError(f"Merged source manifest not found: {source_manifest}")
    if sha256_file(source_manifest) != split_summary.get("source_file_sha256"):
        raise ValueError("Merged source manifest differs from its split summary")

    parquet_path = source_parquet.expanduser().resolve()
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Source clip parquet not found: {parquet_path}")
    actual_parquet_sha256 = sha256_file(parquet_path)
    if actual_parquet_sha256 != expected_source_parquet_sha256:
        raise ValueError(
            "Source clip parquet SHA-256 differs from the predeclared benchmark input"
        )
    try:
        source_clips = pd.read_parquet(parquet_path, columns=["clip_id", "t0_us"])
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"Invalid source clip parquet: {parquet_path}") from error
    parquet_identities = [
        (str(row.clip_id), int(row.t0_us))
        for row in source_clips.itertuples(index=False)
    ]
    if any(not clip_id for clip_id, _ in parquet_identities):
        raise ValueError("Source clip parquet contains an empty clip_id")
    if len(set(parquet_identities)) != len(parquet_identities):
        raise ValueError("Source clip parquet contains duplicate clip identities")
    source_rows = _read_jsonl(source_manifest)
    manifest_identities = {
        (str(row["clip_id"]), int(row["t0_us"])) for row in source_rows
    }
    if set(parquet_identities) != manifest_identities:
        raise ValueError(
            "Source clip parquet and merged oracle manifest contain different clips"
        )
    if len(parquet_identities) != int(split_summary["total_clips"]):
        raise ValueError("Source clip parquet count differs from the split summary")
    if min(epochs, batch_size, base_channels) < 1 or workers < 0:
        raise ValueError("Training protocol values are invalid")

    protocol: dict[str, Any] = {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "status": "frozen_before_test_evaluation",
        "source_data": {
            "clip_parquet": str(parquet_path),
            "clip_parquet_sha256": actual_parquet_sha256,
            "merged_oracle_manifest": str(source_manifest),
            "merged_oracle_manifest_sha256": split_summary["source_file_sha256"],
        },
        "split": {
            "unit": "chunk_id",
            "algorithm": split_summary["algorithm"],
            "seed": split_summary["seed"],
            "ratios": split_summary["ratios"],
            "split_plan_fingerprint": split_summary["split_plan_fingerprint"],
            "manifests": manifest_records,
        },
        "training": {
            "seeds": [2026, 2027],
            "epochs": epochs,
            "batch_size": batch_size,
            "workers": workers,
            "base_channels": base_channels,
            "automatic_mixed_precision": True,
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "checkpoint_selection": "minimum_best_validation_loss",
            "test_metrics_used_for_selection": False,
        },
        "forecast_evaluation": {
            "metrics": ["average_precision", "brier", "iou"],
            "short_horizons_s": [0.5, 1.0, 2.0],
            "probability_threshold": 0.5,
            "average_precision_bins": 1000,
            "post_test_threshold_tuning": False,
            "comparison": "learned_minus_persistence",
        },
        "trajectory_selection": {
            "primary_outcome": "collision_exposure",
            "baseline": "candidate_0",
            "learned_comparator": "persistence_selected",
            "reranker_weights": FIXED_RERANKER_WEIGHTS,
        },
        "uncertainty": {
            "method": "paired_cluster_bootstrap_percentile",
            "cluster_unit": "chunk_id",
            "confidence_level": 0.95,
            "bootstrap_seed": 2026,
            "bootstrap_replicates": 10000,
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
    protocol["protocol_fingerprint"] = canonical_fingerprint(protocol)
    return protocol


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Training run must use LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("Training run must use non-empty LABEL=PATH")
    return label, Path(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    merge = subparsers.add_parser(
        "merge", help="Verify and merge oracle shard manifests"
    )
    merge.add_argument("--manifest-dir", type=Path, required=True)
    merge.add_argument("--candidate-dir", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--summary", type=Path, required=True)
    merge.add_argument("--expected-shards", type=int, required=True)
    merge.add_argument("--expected-rows", type=int, required=True)
    merge.add_argument("--expected-candidates", type=int, required=True)

    split = subparsers.add_parser("split", help="Freeze chunk-disjoint data splits")
    split.add_argument("--manifest", type=Path, required=True)
    split.add_argument("--output-dir", type=Path, required=True)
    split.add_argument("--summary", type=Path)
    split.add_argument("--seed", type=int, default=2026)
    split.add_argument("--ratios", default="0.8,0.1,0.1")

    select = subparsers.add_parser(
        "select-checkpoint", help="Choose a completed training run by validation loss"
    )
    select.add_argument("--run", action="append", type=_parse_run, required=True)
    select.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser(
        "verify-evaluation", help="Verify test isolation from a selected checkpoint"
    )
    verify.add_argument("--selection", type=Path, required=True)
    verify.add_argument("--test-manifest", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)

    protocol = subparsers.add_parser(
        "freeze-protocol",
        help="Write the benchmark analysis protocol before evaluation",
    )
    protocol.add_argument("--split-summary", type=Path, required=True)
    protocol.add_argument("--source-parquet", type=Path, required=True)
    protocol.add_argument(
        "--expected-source-parquet-sha256", default=SOURCE_PARQUET_SHA256
    )
    protocol.add_argument("--epochs", type=int, default=30)
    protocol.add_argument("--batch-size", type=int, default=4)
    protocol.add_argument("--workers", type=int, default=4)
    protocol.add_argument("--base-channels", type=int, default=16)
    protocol.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "merge":
        rows, summary = merge_oracle_manifests(
            manifest_dir=args.manifest_dir,
            candidate_dir=args.candidate_dir,
            expected_shards=args.expected_shards,
            expected_rows=args.expected_rows,
            expected_candidates=args.expected_candidates,
        )
        manifest_content = _canonical_jsonl(rows)
        summary.update(
            {
                "output_manifest": str(args.output.resolve()),
                "output_manifest_sha256": hashlib.sha256(manifest_content).hexdigest(),
            }
        )
        summary["preparation_fingerprint"] = canonical_fingerprint(summary)
        _write_idempotent(args.output, manifest_content)
        _write_idempotent(args.summary, _canonical_json(summary))
        print(f"Verified and merged {len(rows)} oracle rows into {args.output}")
    elif args.command == "split":
        # The merged manifest uses absolute, validated artifact paths. It is read
        # directly so its exact rows and byte hash remain part of the split record.
        rows = _read_jsonl(args.manifest.resolve())
        if not rows:
            raise ValueError(f"Merged manifest is empty: {args.manifest}")
        splits, summary = freeze_splits(
            rows, seed=args.seed, ratios=parse_ratios(args.ratios)
        )
        output_dir = args.output_dir.resolve()
        summary_path = (
            args.summary.resolve()
            if args.summary is not None
            else output_dir / "split_summary.json"
        )
        summary["source_manifest"] = str(args.manifest.resolve())
        summary["source_file_sha256"] = sha256_file(args.manifest)
        summary["output_directory"] = str(output_dir)
        for name in SPLIT_NAMES:
            _write_idempotent(
                output_dir / f"{name}.jsonl", _canonical_jsonl(splits[name])
            )
        _write_idempotent(summary_path, _canonical_json(summary))
        print(f"Froze {len(rows)} clips into chunk-disjoint splits at {output_dir}")
    elif args.command == "select-checkpoint":
        payload = select_checkpoint(args.run)
        _write_idempotent(args.output, _canonical_json(payload))
        print(
            f"Selected {payload['selected_label']} at validation loss "
            f"{payload['selected_validation_loss']:.8f}"
        )
    elif args.command == "verify-evaluation":
        payload = verify_evaluation_partition(args.selection, args.test_manifest)
        _write_idempotent(args.output, _canonical_json(payload))
        print(
            "Verified that the test chunks do not occur in checkpoint training "
            "or validation data"
        )
    else:
        payload = freeze_benchmark_protocol(
            split_summary_path=args.split_summary,
            source_parquet=args.source_parquet,
            expected_source_parquet_sha256=args.expected_source_parquet_sha256,
            epochs=args.epochs,
            batch_size=args.batch_size,
            workers=args.workers,
            base_channels=args.base_channels,
        )
        _write_idempotent(args.output, _canonical_json(payload))
        print(f"Froze the predeclared benchmark protocol at {args.output}")


if __name__ == "__main__":
    main()
