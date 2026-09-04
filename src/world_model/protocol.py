"""Strict loading and input binding for a frozen benchmark protocol."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .runtime import canonical_fingerprint, sha256_file


PROTOCOL_SCHEMA_VERSION = 1
PROTOCOL_STATUS = "frozen_before_test_evaluation"


@dataclass(frozen=True)
class FrozenProtocol:
    """A verified protocol payload and its immutable file provenance."""

    path: Path
    sha256: str
    fingerprint: str
    payload: dict[str, Any]

    def provenance(self) -> dict[str, str]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "protocol_fingerprint": self.fingerprint,
        }


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Frozen protocol {label} must be an object")
    return value


def _require_sha256(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"Frozen protocol {label} is not a lowercase SHA-256 digest")
    return text


def load_frozen_protocol(path: str | Path) -> FrozenProtocol:
    """Load a protocol and verify its canonical fingerprint."""

    protocol_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid frozen protocol: {protocol_path}") from error
    if not isinstance(payload, dict):
        raise ValueError("Frozen protocol must be a JSON object")
    if payload.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("Frozen protocol has an unsupported schema version")
    if payload.get("status") != PROTOCOL_STATUS:
        raise ValueError("Frozen protocol does not have frozen pre-test status")
    fingerprint = _require_sha256(
        payload.get("protocol_fingerprint"), "protocol_fingerprint"
    )
    fingerprint_payload = dict(payload)
    fingerprint_payload.pop("protocol_fingerprint", None)
    if canonical_fingerprint(fingerprint_payload) != fingerprint:
        raise ValueError("Frozen protocol fingerprint does not match its payload")
    for section in (
        "source_data",
        "split",
        "training",
        "forecast_evaluation",
        "trajectory_selection",
        "uncertainty",
        "acceptance_gates",
    ):
        _mapping(payload.get(section), section)
    return FrozenProtocol(
        path=protocol_path,
        sha256=sha256_file(protocol_path),
        fingerprint=fingerprint,
        payload=payload,
    )


def validate_manifest_binding(
    protocol: FrozenProtocol,
    split_name: str,
    manifest: str | Path,
    *,
    actual_sha256: str | None = None,
    clips: int | None = None,
    chunks: int | None = None,
    dataset_fingerprint: str | None = None,
) -> None:
    """Require one manifest to be the exact split input named by the protocol."""

    split = _mapping(protocol.payload["split"], "split")
    if split.get("unit") != "chunk_id":
        raise ValueError("Frozen protocol split unit must be chunk_id")
    manifests = _mapping(split.get("manifests"), "split.manifests")
    record = _mapping(manifests.get(split_name), f"split.manifests.{split_name}")
    actual_path = Path(manifest).expanduser().resolve()
    expected_path = Path(str(record.get("path", ""))).expanduser().resolve()
    if actual_path != expected_path:
        raise ValueError(
            f"{split_name} manifest path differs from the frozen protocol: "
            f"{actual_path} != {expected_path}"
        )
    digest = actual_sha256 or sha256_file(actual_path)
    expected_digest = _require_sha256(
        record.get("sha256"), f"split.manifests.{split_name}.sha256"
    )
    if digest != expected_digest:
        raise ValueError(
            f"{split_name} manifest SHA-256 differs from the frozen protocol"
        )
    for name, actual in (("clips", clips), ("chunks", chunks)):
        if actual is not None and record.get(name) != actual:
            raise ValueError(
                f"{split_name} manifest {name} differ from the frozen protocol"
            )
    if dataset_fingerprint is not None:
        expected_fingerprint = _require_sha256(
            record.get("dataset_fingerprint"),
            f"split.manifests.{split_name}.dataset_fingerprint",
        )
        if dataset_fingerprint != expected_fingerprint:
            raise ValueError(
                f"{split_name} dataset fingerprint differs from the frozen protocol"
            )


def validate_source_manifest_binding(
    protocol: FrozenProtocol, manifest: str | Path
) -> None:
    """Require the exact merged oracle manifest frozen in the protocol."""

    source = _mapping(protocol.payload["source_data"], "source_data")
    actual_path = Path(manifest).expanduser().resolve()
    expected_path = Path(
        str(source.get("merged_oracle_manifest", ""))
    ).expanduser().resolve()
    if actual_path != expected_path:
        raise ValueError(
            "Oracle manifest path differs from the frozen protocol: "
            f"{actual_path} != {expected_path}"
        )
    expected_sha256 = _require_sha256(
        source.get("merged_oracle_manifest_sha256"),
        "source_data.merged_oracle_manifest_sha256",
    )
    if sha256_file(actual_path) != expected_sha256:
        raise ValueError("Oracle manifest SHA-256 differs from the frozen protocol")


def validate_uncertainty_configuration(
    protocol: FrozenProtocol, *, replicates: int, seed: int
) -> None:
    """Require the predeclared source-chunk bootstrap configuration."""

    expected = {
        "method": "paired_cluster_bootstrap_percentile",
        "cluster_unit": "chunk_id",
        "confidence_level": 0.95,
        "bootstrap_seed": seed,
        "bootstrap_replicates": replicates,
    }
    uncertainty = _mapping(protocol.payload["uncertainty"], "uncertainty")
    for name, actual in expected.items():
        value = uncertainty.get(name)
        if isinstance(actual, float):
            matches = (
                isinstance(value, (int, float))
                and math.isfinite(float(value))
                and float(value) == actual
            )
        else:
            matches = value == actual
        if not matches:
            raise ValueError(
                f"Bootstrap {name}={actual!r} differs from frozen protocol value {value!r}"
            )
    require_exact_mapping(uncertainty, expected, "uncertainty")


def require_exact_mapping(
    actual: Any, expected: Mapping[str, Any], label: str
) -> Mapping[str, Any]:
    """Require an exact JSON-compatible mapping."""

    mapping = _mapping(actual, label)
    if canonical_fingerprint(mapping) != canonical_fingerprint(dict(expected)):
        raise ValueError(f"Frozen protocol {label} is not supported exactly")
    return mapping
