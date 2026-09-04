"""Authenticate prediction-run records before downstream benchmark use."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .data import dataset_provenance, load_manifest
from .evaluate import validate_learned_test_binding
from .runtime import canonical_fingerprint, sha256_file


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_KEYS = {
    "method",
    "data_role",
    "checkpoint_sha256",
    "amp",
    "evaluation_binding",
    "data_schema",
}


@dataclass(frozen=True)
class PredictionRun:
    """A canonical prediction-run payload verified against its fingerprint."""

    canonical_json: str
    fingerprint: str
    payload: dict[str, Any]


def _scalar(archive: Mapping[str, np.ndarray], name: str) -> Any:
    if name not in archive:
        raise ValueError(f"Prediction artifact is missing {name}")
    value = np.asarray(archive[name])
    if value.shape != ():
        raise ValueError(f"Prediction field {name} must be scalar")
    return value.item()


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value)
    if not _SHA256.fullmatch(digest):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return digest


def _require_exact(actual: Any, expected: Any, label: str) -> None:
    if canonical_fingerprint(actual) != canonical_fingerprint(expected):
        raise ValueError(f"Prediction run {label} does not match authenticated inputs")


def load_prediction_run(
    archive: Mapping[str, np.ndarray], *, artifact_path: str | Path
) -> PredictionRun:
    """Parse one canonical run record and recompute its declared fingerprint."""

    label = str(artifact_path)
    raw = _scalar(archive, "prediction_run_json")
    if not isinstance(raw, str):
        raise ValueError(f"Prediction run JSON must be text: {label}")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"Prediction run JSON is invalid: {label}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Prediction run JSON must contain an object: {label}")
    try:
        canonical_json = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"Prediction run JSON is not canonical JSON: {label}") from error
    if raw != canonical_json:
        raise ValueError(f"Prediction run JSON is not canonically encoded: {label}")
    fingerprint = _require_sha256(
        _scalar(archive, "prediction_run_fingerprint"),
        f"Prediction run fingerprint in {label}",
    )
    if canonical_fingerprint(payload) != fingerprint:
        raise ValueError(f"Prediction run fingerprint does not match its JSON: {label}")
    return PredictionRun(
        canonical_json=canonical_json,
        fingerprint=fingerprint,
        payload=payload,
    )


def _manifest_identity(
    manifest: str | Path, provenance: Mapping[str, Any] | None
) -> tuple[Path, str, dict[str, Any]]:
    manifest_path = Path(manifest).expanduser().resolve()
    if provenance is None:
        actual = dataset_provenance(load_manifest(manifest_path))
    else:
        try:
            actual = {
                "artifact_count": int(provenance["artifact_count"]),
                "chunk_ids": sorted(str(value) for value in provenance["chunk_ids"]),
                "dataset_fingerprint": str(provenance["dataset_fingerprint"]),
            }
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Test manifest provenance is incomplete") from error
    return manifest_path, sha256_file(manifest_path), actual


def authenticate_prediction_run(
    run: PredictionRun,
    *,
    expected_method: str,
    manifest: str | Path,
    protocol: str | Path | None = None,
    selection_record: str | Path | None = None,
    evaluation_audit: str | Path | None = None,
    manifest_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind a run record to the exact test inputs and learned-model records."""

    if expected_method not in {"learned", "persistence"}:
        raise ValueError(f"Unsupported prediction method: {expected_method!r}")
    payload = run.payload
    if set(payload) != _RUN_KEYS:
        raise ValueError("Prediction run fields do not match prediction schema version 2")
    if payload.get("method") != expected_method:
        raise ValueError("Prediction run method does not match the artifact method")
    if payload.get("data_role") != "test":
        raise ValueError("Downstream benchmark predictions must have data_role='test'")
    if not isinstance(payload.get("amp"), bool):
        raise ValueError("Prediction run amp must be Boolean")
    if not isinstance(payload.get("data_schema"), dict):
        raise ValueError("Prediction run data_schema must be an object")
    binding = payload.get("evaluation_binding")
    if not isinstance(binding, dict):
        raise ValueError("Prediction run evaluation_binding must be an object")
    binding_fingerprint = _require_sha256(
        binding.get("binding_fingerprint"), "Prediction evaluation binding fingerprint"
    )
    fingerprint_payload = dict(binding)
    fingerprint_payload.pop("binding_fingerprint", None)
    if canonical_fingerprint(fingerprint_payload) != binding_fingerprint:
        raise ValueError("Prediction evaluation binding fingerprint does not match")

    manifest_path, manifest_sha256, evaluation_provenance = _manifest_identity(
        manifest, manifest_provenance
    )
    evaluation_manifest = {
        "path": str(manifest_path),
        "sha256": manifest_sha256,
        "dataset_fingerprint": evaluation_provenance["dataset_fingerprint"],
        "clips": int(evaluation_provenance["artifact_count"]),
        "chunks": len(evaluation_provenance["chunk_ids"]),
    }

    checkpoint_sha256 = str(payload.get("checkpoint_sha256", ""))
    if expected_method == "learned":
        missing = [
            flag
            for flag, value in (
                ("--protocol", protocol),
                ("--selection-record", selection_record),
                ("--evaluation-audit", evaluation_audit),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "Learned prediction authentication requires " + " and ".join(missing)
            )
        checkpoint_sha256 = _require_sha256(
            checkpoint_sha256, "Prediction run checkpoint SHA-256"
        )
        selection_path = Path(str(selection_record)).expanduser().resolve()
        try:
            selection = json.loads(selection_path.read_bytes())
            checkpoint_path = Path(str(selection["selected_checkpoint"])).expanduser().resolve()
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"Invalid checkpoint selection record: {selection_path}") from error
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Selected checkpoint not found: {checkpoint_path}")
        actual_checkpoint_sha256 = sha256_file(checkpoint_path)
        if actual_checkpoint_sha256 != checkpoint_sha256:
            raise ValueError("Prediction run does not bind the selected checkpoint bytes")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("Selected checkpoint must contain an object")
        record_binding = validate_learned_test_binding(
            protocol=str(protocol),
            selection_record=selection_path,
            evaluation_audit=str(evaluation_audit),
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=actual_checkpoint_sha256,
            checkpoint=checkpoint,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            evaluation_provenance=evaluation_provenance,
        )
        try:
            checkpoint_provenance = checkpoint["data_provenance"]
            checkpoint_signature = checkpoint["resume_signature"]
            checkpoint_record = {
                "path": str(checkpoint_path),
                "sha256": actual_checkpoint_sha256,
                "epoch": int(checkpoint["epoch"]),
                "seed": int(checkpoint["seed"]),
                "train_manifest_sha256": checkpoint_signature.get(
                    "train_manifest_sha256"
                ),
                "validation_manifest_sha256": checkpoint_signature.get(
                    "val_manifest_sha256"
                ),
                "train_dataset_fingerprint": checkpoint_provenance["train"][
                    "dataset_fingerprint"
                ],
                "validation_dataset_fingerprint": checkpoint_provenance["val"][
                    "dataset_fingerprint"
                ],
            }
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Selected checkpoint has incomplete prediction provenance") from error
        expected_binding = {
            "evaluation_manifest": evaluation_manifest,
            "checkpoint": checkpoint_record,
            **record_binding,
            "partition_isolation": {
                "train_evaluation_chunk_overlap": 0,
                "validation_evaluation_chunk_overlap": 0,
            },
        }
    else:
        if protocol is not None or selection_record is not None or evaluation_audit is not None:
            raise ValueError(
                "Persistence authentication does not accept protocol, selection, or audit records"
            )
        if checkpoint_sha256:
            raise ValueError("Persistence prediction run must not claim a checkpoint")
        expected_binding = {
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
    expected_binding["binding_fingerprint"] = canonical_fingerprint(expected_binding)
    _require_exact(binding, expected_binding, "evaluation binding")

    return {
        "method": expected_method,
        "data_role": "test",
        "prediction_run_fingerprint": run.fingerprint,
        "prediction_run_json_sha256": hashlib.sha256(
            run.canonical_json.encode("utf-8")
        ).hexdigest(),
        "checkpoint_sha256": checkpoint_sha256 or None,
        "evaluation_binding_fingerprint": binding_fingerprint,
        "evaluation_manifest": evaluation_manifest,
        "protocol": expected_binding["protocol"],
        "checkpoint_selection": expected_binding["checkpoint_selection"],
        "evaluation_audit": expected_binding["evaluation_audit"],
        "partition_isolation": expected_binding["partition_isolation"],
    }
