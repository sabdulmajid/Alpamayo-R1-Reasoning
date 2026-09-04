"""Strict loading and input binding for a frozen benchmark protocol."""

from __future__ import annotations

import hashlib
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
        content = protocol_path.read_bytes()
        payload = json.loads(content)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
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
        sha256=hashlib.sha256(content).hexdigest(),
        fingerprint=fingerprint,
        payload=payload,
    )


def normalize_training_configuration(
    value: Any, *, label: str
) -> dict[str, Any]:
    """Validate and normalize the shared training-configuration fields."""

    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")

    def integer(name: str, *, minimum: int) -> int:
        item = value.get(name)
        if isinstance(item, bool) or not isinstance(item, int) or item < minimum:
            raise ValueError(f"{label} {name} is invalid")
        return item

    def finite_number(name: str, *, minimum: float, inclusive: bool) -> float:
        item = value.get(name)
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{label} {name} is invalid")
        result = float(item)
        valid_bound = result >= minimum if inclusive else result > minimum
        if not math.isfinite(result) or not valid_bound:
            raise ValueError(f"{label} {name} is invalid")
        return result

    amp = value.get("automatic_mixed_precision")
    if not isinstance(amp, bool):
        raise ValueError(f"{label} automatic_mixed_precision is invalid")
    normalized = {
        "epochs": integer("epochs", minimum=1),
        "batch_size": integer("batch_size", minimum=1),
        "workers": integer("workers", minimum=0),
        "base_channels": integer("base_channels", minimum=1),
        "automatic_mixed_precision": amp,
        "learning_rate": finite_number(
            "learning_rate", minimum=0.0, inclusive=False
        ),
        "weight_decay": finite_number(
            "weight_decay", minimum=0.0, inclusive=True
        ),
    }
    if canonical_fingerprint(value) != canonical_fingerprint(normalized):
        raise ValueError(f"{label} is not supported exactly")
    return normalized


def checkpoint_training_configuration(
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the protocol-visible training configuration in a checkpoint."""

    run_config = checkpoint.get("run_config")
    resume_signature = checkpoint.get("resume_signature")
    model_config = checkpoint.get("model_config")
    if not all(
        isinstance(item, dict)
        for item in (run_config, resume_signature, model_config)
    ):
        raise ValueError("Checkpoint has incomplete training configuration")

    def integer(name: str, *, minimum: int) -> int:
        value = run_config.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"Checkpoint training {name} is invalid")
        return value

    def finite_number(name: str, *, minimum: float, inclusive: bool) -> float:
        value = run_config.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Checkpoint training {name} is invalid")
        result = float(value)
        valid_bound = result >= minimum if inclusive else result > minimum
        if not math.isfinite(result) or not valid_bound:
            raise ValueError(f"Checkpoint training {name} is invalid")
        return result

    amp = run_config.get("amp")
    if not isinstance(amp, bool):
        raise ValueError("Checkpoint training amp is invalid")
    configuration = normalize_training_configuration({
        "epochs": integer("epochs", minimum=1),
        "batch_size": integer("batch_size", minimum=1),
        "workers": integer("workers", minimum=0),
        "base_channels": integer("base_channels", minimum=1),
        "automatic_mixed_precision": amp,
        "learning_rate": finite_number(
            "learning_rate", minimum=0.0, inclusive=False
        ),
        "weight_decay": finite_number(
            "weight_decay", minimum=0.0, inclusive=True
        ),
    }, label="Checkpoint training configuration")
    resume_bindings = {
        "batch_size": configuration["batch_size"],
        "base_channels": configuration["base_channels"],
        "amp": configuration["automatic_mixed_precision"],
        "learning_rate": configuration["learning_rate"],
        "weight_decay": configuration["weight_decay"],
    }
    for name, expected in resume_bindings.items():
        if resume_signature.get(name) != expected:
            raise ValueError(
                f"Checkpoint run configuration and resume signature differ in {name}"
            )
    if model_config.get("base_channels") != configuration["base_channels"]:
        raise ValueError(
            "Checkpoint run configuration and model configuration differ in base_channels"
        )
    return configuration


def validate_training_binding(
    protocol: FrozenProtocol,
    *,
    seeds: list[int],
    configuration: Mapping[str, Any],
    selection_policy: str,
    test_metrics_used: bool,
) -> None:
    """Bind all recorded training runs to the frozen training protocol."""

    expected = {
        "seeds": seeds,
        **dict(configuration),
        "checkpoint_selection": selection_policy,
        "test_metrics_used_for_selection": test_metrics_used,
    }
    require_exact_mapping(protocol.payload["training"], expected, "training")


def validate_training_run_configuration(
    protocol: FrozenProtocol,
    *,
    seed: int,
    configuration: Mapping[str, Any],
) -> None:
    """Require one training invocation to conform to the frozen protocol."""

    training = _mapping(protocol.payload["training"], "training")
    seeds = training.get("seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) < 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Frozen protocol training seeds are invalid")
    if seed not in seeds:
        raise ValueError(f"Training seed {seed} is absent from the frozen protocol")
    validate_training_binding(
        protocol,
        seeds=seeds,
        configuration=configuration,
        selection_policy="minimum_best_validation_loss",
        test_metrics_used=False,
    )


def validate_protocol_provenance(
    protocol: FrozenProtocol, value: Any, *, label: str
) -> dict[str, str]:
    """Require an embedded record to identify these exact protocol bytes."""

    expected = protocol.provenance()
    if not isinstance(value, dict):
        raise ValueError(f"{label} protocol provenance must be an object")
    if canonical_fingerprint(value) != canonical_fingerprint(expected):
        raise ValueError(f"{label} does not bind the requested frozen protocol")
    return expected


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
