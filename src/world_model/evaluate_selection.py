"""Evaluate held-out trajectory selection against recorded-future LiDAR risk.

The evaluator joins five immutable data products by ``(clip_id, t0_us)``:

* a frozen test split;
* the source oracle manifest and schema-4 oracle artifacts;
* learned and persistence reranker outputs; and
* the original candidate records and trajectory artifacts.

It reports macro averages over clips. Confidence intervals use a paired cluster
bootstrap that resamples source ``chunk_id`` groups. This preserves the pairing
between policies and does not treat clips from one source archive as independent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from .compare_forecasts import compare_forecasts
from .data import load_manifest, load_occupancy_artifact, write_jsonl
from .protocol import (
    FrozenProtocol,
    load_frozen_protocol,
    require_exact_mapping,
    validate_manifest_binding,
    validate_source_manifest_binding,
    validate_uncertainty_configuration,
)
from .rerank import RerankWeights, rerank_artifacts
from .runtime import canonical_fingerprint, sha256_file, write_json


OUTPUT_SCHEMA_VERSION = 1
FORECAST_COMPARISON_SCHEMA_VERSION = 1
ORACLE_SCHEMA_VERSION = 4
HEX_DIGITS = frozenset("0123456789abcdef")
POLICIES = (
    "candidate_0",
    "learned_selected",
    "persistence_selected",
    "comfort_only_selected",
    "random_candidate_expectation",
    "recorded_future_oracle",
)
TRAJECTORY_UNIQUE_TOLERANCE_M = 1e-4
COMFORT_COMPONENTS = {
    "mean_acceleration_mps2": ("acceleration", 1.0),
    "mean_jerk_mps3": ("jerk", 1.0),
    "max_curvature_inv_m": ("curvature", 1.0),
    "progress_m": ("progress", -1.0),
}
METRIC_METADATA: dict[str, dict[str, str]] = {
    "ade_m": {"unit": "m", "preferred_direction": "lower"},
    "fde_m": {"unit": "m", "preferred_direction": "lower"},
    "collision_exposure": {"unit": "fraction", "preferred_direction": "lower"},
    "collision_cells": {"unit": "cells", "preferred_direction": "lower"},
    "conflict_horizons": {"unit": "horizons", "preferred_direction": "lower"},
    "collision_clip_rate": {"unit": "fraction", "preferred_direction": "lower"},
    "out_of_bounds_horizons": {
        "unit": "horizons",
        "preferred_direction": "lower",
    },
    "out_of_bounds_fraction": {
        "unit": "fraction",
        "preferred_direction": "lower",
    },
    "observed_fraction": {"unit": "fraction", "preferred_direction": "higher"},
    "unobserved_horizons": {
        "unit": "horizons",
        "preferred_direction": "lower",
    },
}
RISK_ARRAYS = {
    "collision_exposure": "candidate_collision_exposure",
    "collision_cells": "candidate_collision_cells",
    "conflict_horizons": "candidate_conflict_horizons",
    "out_of_bounds_horizons": "candidate_out_of_bounds_horizons",
    "out_of_bounds_fraction": "candidate_out_of_bounds_fraction",
    "observed_fraction": "candidate_observed_fraction",
    "unobserved_horizons": "candidate_unobserved_horizons",
}


def _protocol_rerank_weights(protocol: FrozenProtocol) -> RerankWeights:
    """Load the exact reranker weights declared by the frozen protocol."""

    selection = protocol.payload["trajectory_selection"]
    if not isinstance(selection, dict):
        raise ValueError("Frozen protocol trajectory_selection must be an object")
    values = selection.get("reranker_weights")
    if not isinstance(values, dict):
        raise ValueError("Frozen protocol reranker_weights must be an object")
    names = set(RerankWeights.__dataclass_fields__)
    if set(values) != names:
        raise ValueError(
            "Frozen protocol reranker_weights fields are not supported exactly"
        )
    normalized = {
        name: _finite_nonnegative(values[name], f"protocol reranker weight {name}")
        for name in sorted(names)
    }
    return RerankWeights(**normalized)


def _identity(row: Mapping[str, Any], source: str) -> tuple[str, int]:
    try:
        clip_id = str(row["clip_id"])
        t0_us = int(row["t0_us"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{source} has an invalid clip identity") from error
    if not clip_id:
        raise ValueError(f"{source} has an empty clip_id")
    return clip_id, t0_us


def _valid_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and set(text).issubset(HEX_DIGITS)


def _finite_nonnegative(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _index_rows(
    rows: Sequence[dict[str, Any]], source: str
) -> dict[tuple[str, int], dict[str, Any]]:
    index: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        identity = _identity(row, source)
        if identity in index:
            raise ValueError(
                f"Duplicate clip identity in {source}: {identity[0]}@{identity[1]}"
            )
        index[identity] = row
    return index


def _load_jsonl(path: str | Path, source: str) -> list[dict[str, Any]]:
    input_path = Path(path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"{source} not found: {input_path}")
    rows: list[dict[str, Any]] = []
    with input_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{source} line {line_number} is not valid JSON"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(f"{source} line {line_number} must be a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{source} is empty: {input_path}")
    return rows


def _load_forecast_comparison(
    path: str | Path,
    *,
    protocol: FrozenProtocol,
    test_manifest: Path,
    test_rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Validate the forecast report bound to this protocol and test split."""

    report_path = Path(path).expanduser().resolve()
    try:
        report_bytes = report_path.read_bytes()
        report = json.loads(report_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid forecast comparison: {report_path}") from error
    if not isinstance(report, dict):
        raise ValueError("Forecast comparison must be a JSON object")
    if report.get("schema_version") != FORECAST_COMPARISON_SCHEMA_VERSION:
        raise ValueError("Forecast comparison has an unsupported schema version")
    if report.get("comparison") != "learned_minus_persistence":
        raise ValueError("Forecast comparison has an unsupported comparison")

    report_protocol = report.get("protocol")
    if not isinstance(report_protocol, dict):
        raise ValueError("Forecast comparison has no protocol provenance")
    try:
        report_protocol_path = Path(str(report_protocol["path"])).expanduser().resolve()
    except KeyError as error:
        raise ValueError(
            "Forecast comparison protocol provenance is incomplete"
        ) from error
    expected_protocol = protocol.provenance()
    if report_protocol_path != protocol.path:
        raise ValueError("Forecast comparison uses a different protocol path")
    for name in ("sha256", "protocol_fingerprint"):
        if report_protocol.get(name) != expected_protocol[name]:
            raise ValueError(f"Forecast comparison protocol {name} does not match")

    inputs = report.get("inputs")
    manifest = inputs.get("manifest") if isinstance(inputs, dict) else None
    if not isinstance(manifest, dict):
        raise ValueError("Forecast comparison has no test-manifest provenance")
    actual_manifest_sha256 = sha256_file(test_manifest)
    actual_dataset_fingerprint = canonical_fingerprint(
        [
            {
                "artifact_sha256": row["artifact_sha256"],
                "chunk_id": str(row["chunk_id"]),
                "clip_id": str(row["clip_id"]),
                "t0_us": int(row["t0_us"]),
            }
            for row in test_rows
        ]
    )
    try:
        report_manifest_path = (
            Path(str(manifest["manifest_path"])).expanduser().resolve()
        )
    except KeyError as error:
        raise ValueError(
            "Forecast comparison test-manifest provenance is incomplete"
        ) from error
    expected_manifest_values = {
        "manifest_sha256": actual_manifest_sha256,
        "dataset_fingerprint": actual_dataset_fingerprint,
        "clips": len(test_rows),
        "chunks": len({str(row["chunk_id"]) for row in test_rows}),
    }
    if report_manifest_path != test_manifest:
        raise ValueError("Forecast comparison uses a different test-manifest path")
    for name, expected in expected_manifest_values.items():
        if manifest.get(name) != expected:
            raise ValueError(f"Forecast comparison test-manifest {name} does not match")

    gates = report.get("acceptance_gates")
    rules = protocol.payload["acceptance_gates"].get("forecast")
    if not isinstance(gates, dict) or not isinstance(rules, dict):
        raise ValueError("Forecast comparison acceptance gates are incomplete")
    require_exact_mapping(
        rules,
        {
            "average_precision": "learned_higher_than_persistence",
            "brier": "learned_lower_than_persistence",
            "iou": "learned_higher_than_persistence",
        },
        "acceptance_gates.forecast",
    )
    decisions = gates.get("decisions")
    if not isinstance(decisions, dict) or set(decisions) != set(rules):
        raise ValueError("Forecast comparison gate decisions do not match the protocol")
    metrics = report.get("metrics")
    paired = report.get("paired_differences")
    try:
        learned = metrics["learned"]["short_horizon"]
        persistence = metrics["persistence"]["short_horizon"]
        paired_short = paired["short_horizon"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Forecast comparison metric summaries are incomplete"
        ) from error
    passed_values: list[bool] = []
    for metric, rule in rules.items():
        decision = decisions[metric]
        if not isinstance(decision, dict) or decision.get("rule") != rule:
            raise ValueError(f"Forecast comparison {metric} gate rule does not match")
        try:
            learned_value = learned[metric]
            persistence_value = persistence[metric]
            difference = paired_short[metric]["estimate"]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"Forecast comparison {metric} values are incomplete"
            ) from error
        expected_values = {
            "learned": learned_value,
            "persistence": persistence_value,
            "learned_minus_persistence": difference,
        }
        if any(decision.get(name) != value for name, value in expected_values.items()):
            raise ValueError(f"Forecast comparison {metric} gate values do not match")
        values = (learned_value, persistence_value, difference)
        if any(value is not None and not _is_finite_number(value) for value in values):
            raise ValueError(f"Forecast comparison {metric} gate values are invalid")
        defined = all(value is not None for value in values)
        if rule == "learned_higher_than_persistence":
            expected_passed = defined and float(difference) > 0.0
        elif rule == "learned_lower_than_persistence":
            expected_passed = defined and float(difference) < 0.0
        else:
            raise ValueError(f"Unsupported forecast comparison gate rule: {rule}")
        if decision.get("passed") is not expected_passed:
            raise ValueError(
                f"Forecast comparison {metric} gate decision is inconsistent"
            )
        passed_values.append(expected_passed)
    expected_all_passed = all(passed_values)
    if gates.get("all_passed") is not expected_all_passed:
        raise ValueError("Forecast comparison aggregate gate decision is inconsistent")

    return report, {
        "path": str(report_path),
        "sha256": hashlib.sha256(report_bytes).hexdigest(),
    }


def _require_recomputed_forecast_comparison(
    report: Mapping[str, Any],
    *,
    protocol: FrozenProtocol,
    test_manifest: Path,
    learned_run: Mapping[str, Any],
    prediction_inputs: Mapping[str, Sequence[Mapping[str, Any]]],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> None:
    """Recompute the forecast report from the exact reranking predictions."""

    prediction_roots: dict[str, Path] = {}
    for method in ("learned", "persistence"):
        entries = prediction_inputs.get(method)
        if not entries:
            raise ValueError(f"No authenticated {method} reranking predictions")
        paths: set[Path] = set()
        for entry in entries:
            path = Path(str(entry["path"])).expanduser().resolve()
            if sha256_file(path) != entry["sha256"]:
                raise ValueError(
                    f"Authenticated {method} prediction changed before forecast "
                    "recomputation"
                )
            paths.add(path)
        parents = {path.parent for path in paths}
        if len(parents) != 1:
            raise ValueError(
                f"Authenticated {method} reranking predictions span directories"
            )
        root = next(iter(parents))
        actual_paths = {path.resolve() for path in root.glob("*.prediction.npz")}
        if actual_paths != paths:
            raise ValueError(
                f"Authenticated {method} reranking predictions are not an exact "
                "prediction directory"
            )
        prediction_roots[method] = root

    binding = learned_run["evaluation_binding"]
    selection = binding["checkpoint_selection"]
    audit = binding["evaluation_audit"]
    forecast_configuration = protocol.payload["forecast_evaluation"]
    recomputed = compare_forecasts(
        protocol=protocol.path,
        manifest=test_manifest,
        selection_record=selection["path"],
        evaluation_audit=audit["path"],
        learned_prediction_dir=prediction_roots["learned"],
        persistence_prediction_dir=prediction_roots["persistence"],
        threshold=forecast_configuration["probability_threshold"],
        ap_bins=forecast_configuration["average_precision_bins"],
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    authenticated_run = recomputed["inputs"]["predictions"]["learned"][
        "authenticated_run"
    ]
    immutable_inputs = (
        (protocol.path, recomputed["protocol"]["sha256"], "protocol"),
        (
            test_manifest,
            recomputed["inputs"]["manifest"]["manifest_sha256"],
            "test manifest",
        ),
        (
            Path(str(selection["path"])).expanduser().resolve(),
            authenticated_run["checkpoint_selection"]["sha256"],
            "checkpoint selection",
        ),
        (
            Path(str(audit["path"])).expanduser().resolve(),
            authenticated_run["evaluation_audit"]["sha256"],
            "evaluation audit",
        ),
        (
            Path(str(binding["checkpoint"]["path"])).expanduser().resolve(),
            authenticated_run["checkpoint_sha256"],
            "selected checkpoint",
        ),
    )
    for path, expected_sha256, name in immutable_inputs:
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"Authenticated {name} changed during recomputation")
    for method, entries in prediction_inputs.items():
        for entry in entries:
            if sha256_file(entry["path"]) != entry["sha256"]:
                raise ValueError(
                    f"Authenticated {method} prediction changed during forecast "
                    "recomputation"
                )
    try:
        supplied_json = json.dumps(
            report, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        recomputed_json = json.dumps(
            recomputed, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Forecast comparison is not canonical JSON data") from error
    if supplied_json != recomputed_json:
        raise ValueError(
            "Forecast comparison does not match deterministic recomputation from "
            "the authenticated reranking predictions"
        )


def _evaluation_manifest_binding(
    manifest_path: Path, rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Recompute the evaluation provenance encoded by prediction producers."""

    entries = [
        {
            "artifact_sha256": sha256_file(row["artifact_path"]),
            "chunk_id": str(row["chunk_id"]),
            "clip_id": str(row["clip_id"]),
            "t0_us": int(row["t0_us"]),
        }
        for row in sorted(
            rows,
            key=lambda item: (
                str(item["chunk_id"]),
                str(item["clip_id"]),
                int(item["t0_us"]),
            ),
        )
    ]
    return {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "dataset_fingerprint": canonical_fingerprint(entries),
        "clips": len(entries),
        "chunks": len({entry["chunk_id"] for entry in entries}),
    }


def _resolve_row_path(row: Mapping[str, Any], key: str, parent: Path) -> Path:
    try:
        path = Path(str(row[key])).expanduser()
    except KeyError as error:
        raise ValueError(f"Rerank row is missing {key}") from error
    if not path.is_absolute():
        path = parent / path
    return path.resolve()


def _validate_prediction_run(
    prediction_path: Path,
    *,
    expected_method: str,
    expected_checkpoint_sha256: Any,
    protocol: FrozenProtocol,
    evaluation_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate embedded run metadata and its held-out evaluation binding."""

    with np.load(prediction_path, allow_pickle=False) as archive:
        for name in ("prediction_run_json", "prediction_run_fingerprint"):
            if name not in archive.files:
                raise ValueError(
                    f"Prediction artifact is missing {name}: {prediction_path}"
                )
            if np.asarray(archive[name]).shape != ():
                raise ValueError(
                    f"Prediction artifact {name} must be scalar: {prediction_path}"
                )
        run_json = str(np.asarray(archive["prediction_run_json"]).reshape(()))
        embedded_fingerprint = str(
            np.asarray(archive["prediction_run_fingerprint"]).reshape(())
        )
        try:
            run = json.loads(run_json)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Prediction run metadata is not valid JSON: {prediction_path}"
            ) from error
        if not isinstance(run, dict):
            raise ValueError("Prediction run metadata must be a JSON object")
        try:
            canonical_json = json.dumps(
                run, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Prediction run metadata is not canonical JSON data"
            ) from error
        if run_json != canonical_json:
            raise ValueError("Prediction run JSON is not canonically encoded")
        if (
            not _valid_sha256(embedded_fingerprint)
            or canonical_fingerprint(run) != embedded_fingerprint
        ):
            raise ValueError("Prediction run fingerprint does not match its metadata")

        required_run_fields = {
            "method",
            "data_role",
            "checkpoint_sha256",
            "amp",
            "evaluation_binding",
            "data_schema",
        }
        if set(run) != required_run_fields:
            raise ValueError("Prediction run metadata fields are not supported exactly")
        if run["method"] != expected_method:
            raise ValueError("Prediction run method does not match the reranker input")
        if run["data_role"] != "test":
            raise ValueError("Prediction run is not bound to the held-out test role")
        expected_checkpoint = (
            ""
            if expected_checkpoint_sha256 in (None, "")
            else str(expected_checkpoint_sha256)
        )
        if run["checkpoint_sha256"] != expected_checkpoint:
            raise ValueError("Prediction run checkpoint does not match the artifact")
        if not isinstance(run["amp"], bool):
            raise ValueError("Prediction run amp field must be boolean")

        data_schema = run["data_schema"]
        schema_fields = {
            "input_shape",
            "target_shape",
            "past_horizons_s",
            "horizons_s",
            "grid_origin_xy_m",
            "resolution_m",
            "coordinate_frame",
        }
        if not isinstance(data_schema, dict) or set(data_schema) != schema_fields:
            raise ValueError(
                "Prediction run data schema fields are not supported exactly"
            )
        probability_shape = list(np.asarray(archive["occupancy_prob"]).shape)
        if data_schema["target_shape"] != probability_shape:
            raise ValueError("Prediction run target shape does not match the artifact")
        embedded_schema = {
            "horizons_s": np.asarray(archive["horizons_s"], dtype=np.float64),
            "grid_origin_xy_m": np.asarray(
                archive["grid_origin_xy_m"], dtype=np.float64
            ),
        }
        for name, embedded in embedded_schema.items():
            declared = np.asarray(data_schema[name], dtype=np.float64)
            if declared.shape != embedded.shape or not np.array_equal(
                declared, embedded
            ):
                raise ValueError(f"Prediction run {name} does not match the artifact")
        if float(data_schema["resolution_m"]) != float(
            np.asarray(archive["resolution_m"]).reshape(())
        ):
            raise ValueError("Prediction run resolution does not match the artifact")
        if str(data_schema["coordinate_frame"]) != str(
            np.asarray(archive["coordinate_frame"]).reshape(())
        ):
            raise ValueError(
                "Prediction run coordinate frame does not match the artifact"
            )
        for name in ("input_shape", "past_horizons_s"):
            values = data_schema[name]
            if not isinstance(values, list) or not values:
                raise ValueError(f"Prediction run {name} must be a non-empty list")

    binding = run["evaluation_binding"]
    common_binding_fields = {
        "evaluation_manifest",
        "checkpoint",
        "checkpoint_selection",
        "evaluation_audit",
        "partition_isolation",
        "binding_fingerprint",
    }
    supported_binding_fields = common_binding_fields | {"protocol"}
    if not isinstance(binding, dict) or set(binding) not in (
        common_binding_fields,
        supported_binding_fields,
    ):
        raise ValueError(
            "Prediction evaluation-binding fields are not supported exactly"
        )
    binding_payload = dict(binding)
    binding_fingerprint = binding_payload.pop("binding_fingerprint")
    if (
        not _valid_sha256(binding_fingerprint)
        or canonical_fingerprint(binding_payload) != binding_fingerprint
    ):
        raise ValueError("Prediction evaluation-binding fingerprint does not match")
    if canonical_fingerprint(
        binding.get("evaluation_manifest")
    ) != canonical_fingerprint(dict(evaluation_manifest)):
        raise ValueError("Prediction run uses a different test-manifest binding")

    checkpoint = binding.get("checkpoint")
    selection = binding.get("checkpoint_selection")
    audit = binding.get("evaluation_audit")
    isolation = binding.get("partition_isolation")
    if not isinstance(isolation, dict) or set(isolation) != {
        "train_evaluation_chunk_overlap",
        "validation_evaluation_chunk_overlap",
    }:
        raise ValueError("Prediction partition-isolation record is incomplete")
    if expected_method == "learned":
        if set(binding) != supported_binding_fields:
            raise ValueError("Learned prediction has no protocol provenance")
        protocol_record = binding["protocol"]
        if canonical_fingerprint(protocol_record) != canonical_fingerprint(
            protocol.provenance()
        ):
            raise ValueError("Prediction run protocol provenance does not match")
        checkpoint_fields = {
            "path",
            "sha256",
            "epoch",
            "seed",
            "train_manifest_sha256",
            "validation_manifest_sha256",
            "train_dataset_fingerprint",
            "validation_dataset_fingerprint",
        }
        if not isinstance(checkpoint, dict) or set(checkpoint) != checkpoint_fields:
            raise ValueError("Learned prediction checkpoint binding is incomplete")
        if checkpoint.get("sha256") != expected_checkpoint:
            raise ValueError("Learned prediction checkpoint binding does not match")
        for name in checkpoint_fields - {"path", "epoch", "seed"}:
            if not _valid_sha256(checkpoint.get(name)):
                raise ValueError(f"Learned prediction checkpoint {name} is invalid")
        for name in ("epoch", "seed"):
            if isinstance(checkpoint[name], bool) or not isinstance(
                checkpoint[name], int
            ):
                raise ValueError(f"Learned prediction checkpoint {name} is invalid")
        if not isinstance(selection, dict) or not isinstance(audit, dict):
            raise ValueError(
                "Learned prediction has no checkpoint-selection audit binding"
            )
        if isolation != {
            "train_evaluation_chunk_overlap": 0,
            "validation_evaluation_chunk_overlap": 0,
        }:
            raise ValueError(
                "Learned prediction partition isolation is not demonstrated"
            )
    else:
        if checkpoint is not None or selection is not None or audit is not None:
            raise ValueError(
                "Persistence prediction must not carry checkpoint selection metadata"
            )
        if isolation != {
            "train_evaluation_chunk_overlap": None,
            "validation_evaluation_chunk_overlap": None,
        }:
            raise ValueError("Persistence prediction has invalid partition metadata")
    return run


def _reject_split_leakage(
    test_rows: Sequence[dict[str, Any]],
    other_rows: Sequence[dict[str, Any]],
    other_name: str,
) -> None:
    test_chunks = {str(row["chunk_id"]) for row in test_rows}
    other_chunks = {str(row["chunk_id"]) for row in other_rows}
    overlap = sorted(test_chunks & other_chunks)
    if overlap:
        preview = ", ".join(overlap[:10])
        suffix = "" if len(overlap) <= 10 else f" (and {len(overlap) - 10} more)"
        raise ValueError(
            f"Test and {other_name} manifests share source chunks: {preview}{suffix}"
        )
    test_identities = {_identity(row, "test manifest") for row in test_rows}
    other_identities = {_identity(row, f"{other_name} manifest") for row in other_rows}
    identity_overlap = sorted(test_identities & other_identities)
    if identity_overlap:
        preview = ", ".join(
            f"{clip_id}@{t0_us}" for clip_id, t0_us in identity_overlap[:10]
        )
        raise ValueError(
            f"Test and {other_name} manifests share clip identities: {preview}"
        )


def cluster_bootstrap_interval(
    values: Sequence[float] | np.ndarray,
    cluster_ids: Sequence[str],
    *,
    replicates: int = 10_000,
    seed: int = 2026,
) -> dict[str, float | int]:
    """Return a deterministic percentile interval for one clip-level statistic.

    Each replicate samples the observed clusters with replacement. All clips in
    a sampled cluster are included, and repeated clusters repeat all their clips.
    """

    array = np.asarray(values, dtype=np.float64)
    clusters = np.asarray([str(value) for value in cluster_ids], dtype=np.str_)
    if array.ndim != 1 or clusters.shape != array.shape or array.size == 0:
        raise ValueError(
            "Bootstrap values and cluster_ids must be non-empty 1-D arrays"
        )
    if not np.isfinite(array).all():
        raise ValueError("Bootstrap values must be finite")
    if replicates < 1:
        raise ValueError("Bootstrap replicates must be positive")
    unique_clusters, inverse = np.unique(clusters, return_inverse=True)
    if unique_clusters.size < 2:
        raise ValueError("Cluster bootstrap needs at least two source chunks")

    cluster_sums = np.bincount(inverse, weights=array).astype(np.float64)
    cluster_counts = np.bincount(inverse).astype(np.float64)
    generator = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    batch_size = 1024
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        selected = generator.integers(
            0, unique_clusters.size, size=(stop - start, unique_clusters.size)
        )
        numerator = cluster_sums[selected].sum(axis=1)
        denominator = cluster_counts[selected].sum(axis=1)
        samples[start:stop] = numerator / denominator
    lower, upper = np.percentile(samples, [2.5, 97.5])
    return {
        "estimate": float(array.mean()),
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
        "bootstrap_replicates": int(replicates),
    }


def _load_candidate_records(
    candidate_dir: str | Path,
    required_identities: set[tuple[str, int]],
) -> dict[tuple[str, int], dict[str, Any]]:
    root = Path(candidate_dir).expanduser().resolve()
    record_dir = root / "records"
    if not record_dir.is_dir():
        raise FileNotFoundError(f"Candidate records directory not found: {record_dir}")
    record_paths = sorted(record_dir.glob("*.json"))
    if not record_paths:
        raise FileNotFoundError(f"No candidate records found in {record_dir}")

    index: dict[tuple[str, int], dict[str, Any]] = {}
    for record_path in record_paths:
        try:
            record_bytes = record_path.read_bytes()
            record = json.loads(record_bytes)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid candidate record JSON: {record_path}") from error
        if not isinstance(record, dict) or record.get("schema_version") != 1:
            raise ValueError(f"Unsupported candidate record schema: {record_path}")
        identity = _identity(record, f"candidate record {record_path}")
        if identity in index:
            raise ValueError(
                f"Duplicate candidate record for {identity[0]}@{identity[1]}"
            )
        index[identity] = {
            **record,
            "record_path": str(record_path.resolve()),
            "record_sha256": hashlib.sha256(record_bytes).hexdigest(),
        }

    missing = sorted(required_identities - set(index))
    if missing:
        preview = ", ".join(f"{clip_id}@{t0_us}" for clip_id, t0_us in missing[:10])
        raise ValueError(f"Candidate records are missing test clips: {preview}")

    validated: dict[tuple[str, int], dict[str, Any]] = {}
    for identity in sorted(required_identities):
        record = index[identity]
        record_path = Path(str(record["record_path"]))
        if sha256_file(record_path) != record["record_sha256"]:
            raise ValueError(
                f"Candidate record changed while evaluating: {record_path}"
            )
        artifact_value = Path(str(record.get("artifact_path", "")))
        if not str(artifact_value) or artifact_value.is_absolute():
            raise ValueError(
                f"Candidate record {record_path} must use a root-relative artifact_path"
            )
        artifact_path = (root / artifact_value).resolve()
        if not artifact_path.is_relative_to(root):
            raise ValueError(
                f"Candidate record {record_path} escapes the candidate root"
            )
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Candidate artifact not found: {artifact_path}")
        artifact_sha256 = str(record.get("artifact_sha256", ""))
        if not _valid_sha256(artifact_sha256):
            raise ValueError(
                f"Candidate record {record_path} has an invalid artifact SHA-256"
            )
        if sha256_file(artifact_path) != artifact_sha256:
            raise ValueError(f"Candidate artifact SHA-256 mismatch: {artifact_path}")
        config_fingerprint = str(record.get("config_fingerprint", ""))
        if not _valid_sha256(config_fingerprint):
            raise ValueError(
                f"Candidate record {record_path} has an invalid configuration fingerprint"
            )
        candidates = record.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"Candidate record {record_path} has no candidates")
        indices = [candidate.get("candidate_index") for candidate in candidates]
        if indices != list(range(len(candidates))):
            raise ValueError(f"Candidate indices are not contiguous in {record_path}")

        with np.load(artifact_path, allow_pickle=False) as artifact:
            required = {
                "schema_version",
                "clip_id",
                "t0_us",
                "config_fingerprint",
                "pred_xyz",
                "gt_xyz",
            }
            missing_arrays = sorted(required - set(artifact.files))
            if missing_arrays:
                raise ValueError(
                    f"Candidate artifact {artifact_path} is missing {missing_arrays}"
                )
            embedded_identity = (
                str(np.asarray(artifact["clip_id"]).reshape(())),
                int(np.asarray(artifact["t0_us"]).reshape(())),
            )
            if embedded_identity != identity:
                raise ValueError(
                    f"Candidate artifact identity mismatch: {artifact_path}"
                )
            if int(np.asarray(artifact["schema_version"]).reshape(())) != 1:
                raise ValueError(
                    f"Unsupported candidate artifact schema: {artifact_path}"
                )
            embedded_fingerprint = str(
                np.asarray(artifact["config_fingerprint"]).reshape(())
            )
            if embedded_fingerprint != config_fingerprint:
                raise ValueError(
                    f"Candidate configuration fingerprint mismatch: {artifact_path}"
                )
            pred_xyz = np.asarray(artifact["pred_xyz"], dtype=np.float64)
            gt_xyz = np.asarray(artifact["gt_xyz"], dtype=np.float64)
        if (
            pred_xyz.ndim != 3
            or pred_xyz.shape[0] != len(candidates)
            or pred_xyz.shape[2] != 3
            or gt_xyz.shape != pred_xyz.shape[1:]
            or not np.isfinite(pred_xyz).all()
            or not np.isfinite(gt_xyz).all()
        ):
            raise ValueError(
                f"Candidate trajectory geometry is invalid: {artifact_path}"
            )

        displacement = np.linalg.norm(pred_xyz[:, :, :2] - gt_xyz[None, :, :2], axis=2)
        metrics: list[dict[str, float]] = []
        for candidate_index, candidate in enumerate(candidates):
            saved_metrics = candidate.get("metrics")
            if not isinstance(saved_metrics, dict):
                raise ValueError(
                    f"Candidate {candidate_index} has no metrics in {record_path}"
                )
            recomputed = {
                "ade_m": float(displacement[candidate_index].mean()),
                "fde_m": float(displacement[candidate_index, -1]),
            }
            for metric_name, expected in recomputed.items():
                saved = _finite_nonnegative(
                    saved_metrics.get(metric_name),
                    f"candidate {candidate_index} {metric_name}",
                )
                if not np.isclose(saved, expected, rtol=1e-10, atol=1e-10):
                    raise ValueError(
                        f"Candidate {candidate_index} {metric_name} does not match "
                        f"artifact geometry in {record_path}"
                    )
            metrics.append(recomputed)
        validated[identity] = {
            **record,
            "artifact_path": str(artifact_path),
            "artifact_sha256": artifact_sha256,
            "candidate_count": len(candidates),
            "metrics": metrics,
            "pred_xyz": pred_xyz.astype(np.float32),
        }
    return validated


def _load_oracle_metrics(
    path: Path,
    identity: tuple[str, int],
    candidate_record: Mapping[str, Any],
) -> dict[str, Any]:
    loaded = load_occupancy_artifact(path)
    artifact_identity = loaded["identity"]
    if (
        artifact_identity.clip_id,
        artifact_identity.t0_us,
        artifact_identity.schema_version,
    ) != (identity[0], identity[1], ORACLE_SCHEMA_VERSION):
        raise ValueError(f"Oracle artifact identity or schema mismatch: {path}")

    with np.load(path, allow_pickle=False) as artifact:
        required = {
            "candidate_config_fingerprint",
            "candidate_artifact_sha256",
            "candidate_xyz",
            "candidate_collision_exposure",
            "candidate_collision_cells",
            "candidate_conflict_horizons",
            "candidate_out_of_bounds_horizons",
            "candidate_out_of_bounds_fraction",
            "candidate_observed_fraction",
            "candidate_unobserved_horizons",
            "candidate_oracle_rank",
            "oracle_safest_idx",
            "oracle_config_fingerprint",
            "dataset_revision",
        }
        missing = sorted(required - set(artifact.files))
        if missing:
            raise ValueError(f"Oracle artifact {path} is missing {missing}")
        candidate_sha256 = str(
            np.asarray(artifact["candidate_artifact_sha256"]).reshape(())
        )
        candidate_config = str(
            np.asarray(artifact["candidate_config_fingerprint"]).reshape(())
        )
        oracle_config = str(
            np.asarray(artifact["oracle_config_fingerprint"]).reshape(())
        )
        dataset_revision = str(np.asarray(artifact["dataset_revision"]).reshape(()))
        candidate_xyz = np.asarray(artifact["candidate_xyz"], dtype=np.float32)
        safest_index = int(np.asarray(artifact["oracle_safest_idx"]).reshape(()))
        ranks = np.asarray(artifact["candidate_oracle_rank"], dtype=np.int64)
        risks = {
            metric: np.asarray(artifact[array_name])
            for metric, array_name in RISK_ARRAYS.items()
        }
        horizon_count = int(np.asarray(artifact["horizons_s"]).size)

    candidate_count = int(candidate_record["candidate_count"])
    if candidate_sha256 != candidate_record["artifact_sha256"]:
        raise ValueError(f"Oracle source candidate SHA-256 mismatch: {path}")
    if candidate_config != candidate_record["config_fingerprint"]:
        raise ValueError(f"Oracle source candidate configuration mismatch: {path}")
    if not _valid_sha256(oracle_config):
        raise ValueError(f"Oracle configuration fingerprint is invalid: {path}")
    if not dataset_revision:
        raise ValueError(f"Oracle dataset revision is empty: {path}")
    if candidate_xyz.shape != candidate_record["pred_xyz"].shape or not np.array_equal(
        candidate_xyz, candidate_record["pred_xyz"]
    ):
        raise ValueError(f"Oracle trajectories do not match source candidates: {path}")
    if not 0 <= safest_index < candidate_count:
        raise ValueError(f"Oracle safest index is invalid: {path}")
    if ranks.shape != (candidate_count,) or not np.array_equal(
        np.sort(ranks), np.arange(candidate_count)
    ):
        raise ValueError(f"Oracle candidate ranks are invalid: {path}")
    if ranks[safest_index] != 0:
        raise ValueError(f"Oracle safest index and candidate ranks disagree: {path}")

    for name, values in risks.items():
        if values.shape != (candidate_count,) or not np.isfinite(values).all():
            raise ValueError(f"Oracle {name} values are invalid: {path}")
    for name in ("collision_exposure", "out_of_bounds_fraction", "observed_fraction"):
        if np.any((risks[name] < 0) | (risks[name] > 1)):
            raise ValueError(f"Oracle {name} values must be in [0,1]: {path}")
    for name in (
        "collision_cells",
        "conflict_horizons",
        "out_of_bounds_horizons",
        "unobserved_horizons",
    ):
        if np.any(risks[name] < 0):
            raise ValueError(f"Oracle {name} values must be non-negative: {path}")
    for name in ("conflict_horizons", "out_of_bounds_horizons", "unobserved_horizons"):
        if np.any(risks[name] > horizon_count):
            raise ValueError(f"Oracle {name} exceeds the horizon count: {path}")
    if not np.allclose(
        risks["out_of_bounds_fraction"],
        risks["out_of_bounds_horizons"] / horizon_count,
        rtol=0,
        atol=1e-7,
    ):
        raise ValueError(f"Oracle out-of-bounds metrics disagree: {path}")
    return {
        "candidate_count": candidate_count,
        "oracle_safest_idx": safest_index,
        "oracle_config_fingerprint": oracle_config,
        "dataset_revision": dataset_revision,
        "risks": risks,
    }


def _validate_rerank_row(
    row: Mapping[str, Any],
    rerank_parent: Path,
    identity: tuple[str, int],
    oracle_path: Path,
    oracle_sha256: str,
    candidate_count: int,
    expected_producer_method: str,
    expected_weights: RerankWeights,
    protocol: FrozenProtocol,
    evaluation_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if _identity(row, "rerank output") != identity:
        raise AssertionError("Rerank index returned the wrong identity")
    if int(row.get("candidate_count", -1)) != candidate_count:
        raise ValueError(
            f"Rerank candidate count mismatch for {identity[0]}@{identity[1]}"
        )
    if int(row.get("first_candidate_index", -1)) != 0:
        raise ValueError(
            f"Rerank baseline index must be zero for {identity[0]}@{identity[1]}"
        )
    selected_index = int(row.get("world_selected_index", -1))
    if not 0 <= selected_index < candidate_count:
        raise ValueError(
            f"Rerank selected index is invalid for {identity[0]}@{identity[1]}"
        )
    producer_method = str(row.get("producer_method", ""))
    if producer_method != expected_producer_method:
        raise ValueError(
            f"Expected rerank producer_method={expected_producer_method!r}, "
            f"got {producer_method!r} for {identity[0]}@{identity[1]}"
        )
    checkpoint_sha256 = row.get("checkpoint_sha256")
    if expected_producer_method == "learned" and not _valid_sha256(checkpoint_sha256):
        raise ValueError(
            f"Learned rerank has no valid checkpoint for {identity[0]}@{identity[1]}"
        )
    run_fingerprint = str(row.get("prediction_run_fingerprint", ""))
    if not _valid_sha256(run_fingerprint):
        raise ValueError(
            f"Rerank run fingerprint is invalid for {identity[0]}@{identity[1]}"
        )
    candidate_sha256 = str(row.get("candidate_sha256", ""))
    if candidate_sha256 != oracle_sha256:
        raise ValueError(
            f"Rerank oracle artifact SHA-256 mismatch for {identity[0]}@{identity[1]}"
        )
    candidate_path = _resolve_row_path(row, "candidate_path", rerank_parent)
    candidate_content_sha256 = (
        oracle_sha256
        if candidate_path == oracle_path
        else sha256_file(candidate_path)
        if candidate_path.is_file()
        else ""
    )
    if candidate_content_sha256 != oracle_sha256:
        raise ValueError(
            f"Rerank candidate_path content mismatch for {identity[0]}@{identity[1]}"
        )

    prediction_sha256 = str(row.get("prediction_sha256", ""))
    if not _valid_sha256(prediction_sha256):
        raise ValueError(
            f"Rerank prediction SHA-256 is invalid for {identity[0]}@{identity[1]}"
        )
    prediction_path = _resolve_row_path(row, "prediction_path", rerank_parent)
    if (
        not prediction_path.is_file()
        or sha256_file(prediction_path) != prediction_sha256
    ):
        raise ValueError(
            f"Rerank prediction content mismatch for {identity[0]}@{identity[1]}"
        )
    if expected_producer_method == "persistence" and checkpoint_sha256 not in (
        None,
        "",
    ):
        raise ValueError(
            f"Persistence rerank claims a checkpoint for {identity[0]}@{identity[1]}"
        )
    recomputed = rerank_artifacts(
        prediction_path,
        candidate_path,
        expected_weights,
        expected_clip_id=identity[0],
        expected_t0_us=identity[1],
    )
    prediction_run = _validate_prediction_run(
        prediction_path,
        expected_method=expected_producer_method,
        expected_checkpoint_sha256=checkpoint_sha256,
        protocol=protocol,
        evaluation_manifest=evaluation_manifest,
    )
    if sha256_file(prediction_path) != prediction_sha256:
        raise ValueError(
            f"Rerank prediction changed during recomputation for "
            f"{identity[0]}@{identity[1]}"
        )
    for name in (
        "candidate_count",
        "first_candidate_index",
        "world_selected_index",
        "selection_policy",
        "producer_method",
        "checkpoint_sha256",
        "prediction_run_fingerprint",
        "candidate_sha256",
    ):
        if row.get(name) != recomputed[name]:
            raise ValueError(
                f"Rerank {name} does not match recomputation for "
                f"{identity[0]}@{identity[1]}"
            )
    weights = row.get("weights")
    require_exact_mapping(weights, recomputed["weights"], "reranker weights")
    scores = row.get("candidate_scores")
    expected_scores = recomputed["candidate_scores"]
    if not isinstance(scores, list) or len(scores) != len(expected_scores):
        raise ValueError(f"Rerank scores are invalid for {identity[0]}@{identity[1]}")
    comfort_components: list[dict[str, float]] = []
    for candidate_index, (score, expected_score) in enumerate(
        zip(scores, expected_scores, strict=True)
    ):
        if not isinstance(score, dict) or set(score) != set(expected_score):
            raise ValueError(
                f"Rerank candidate score fields differ from recomputation for "
                f"{identity[0]}@{identity[1]}"
            )
        for name, expected in expected_score.items():
            actual = _finite(
                score.get(name), f"rerank candidate {candidate_index} {name}"
            )
            if not np.isclose(actual, expected, rtol=1e-7, atol=1e-9):
                raise ValueError(
                    f"Rerank candidate {candidate_index} {name} differs from "
                    f"recomputation for {identity[0]}@{identity[1]}"
                )
        components = {
            component: float(expected_score[component])
            for component in COMFORT_COMPONENTS
        }
        comfort_components.append(components)
    return {
        "selected_index": recomputed["world_selected_index"],
        "producer_method": recomputed["producer_method"],
        "checkpoint_sha256": recomputed["checkpoint_sha256"],
        "prediction_run_fingerprint": recomputed["prediction_run_fingerprint"],
        "selection_policy": recomputed["selection_policy"],
        "weights": recomputed["weights"],
        "comfort_components": comfort_components,
        "prediction_path": str(prediction_path),
        "prediction_sha256": prediction_sha256,
        "prediction_run": prediction_run,
    }


def _comfort_only_selection(
    learned_rerank: Mapping[str, Any],
    persistence_rerank: Mapping[str, Any],
    identity: tuple[str, int],
) -> tuple[int, list[float]]:
    learned_components = learned_rerank["comfort_components"]
    persistence_components = persistence_rerank["comfort_components"]
    if canonical_fingerprint(learned_components) != canonical_fingerprint(
        persistence_components
    ):
        raise ValueError(
            f"Learned and persistence comfort components differ for "
            f"{identity[0]}@{identity[1]}"
        )
    learned_weights = learned_rerank["weights"]
    persistence_weights = persistence_rerank["weights"]
    comfort_weight_names = {
        weight_name for weight_name, _ in COMFORT_COMPONENTS.values()
    }
    learned_comfort_weights = {
        name: learned_weights[name] for name in sorted(comfort_weight_names)
    }
    persistence_comfort_weights = {
        name: persistence_weights[name] for name in sorted(comfort_weight_names)
    }
    if canonical_fingerprint(learned_comfort_weights) != canonical_fingerprint(
        persistence_comfort_weights
    ):
        raise ValueError(
            f"Learned and persistence comfort weights differ for "
            f"{identity[0]}@{identity[1]}"
        )
    comfort_scores = [
        float(
            sum(
                direction
                * float(learned_weights[weight_name])
                * float(components[component])
                for component, (weight_name, direction) in COMFORT_COMPONENTS.items()
            )
        )
        for components in learned_components
    ]
    if not np.isfinite(comfort_scores).all():
        raise ValueError(
            f"Comfort-only scores are not finite for {identity[0]}@{identity[1]}"
        )
    return int(np.argmin(comfort_scores)), comfort_scores


def _candidate_diversity(
    candidate_xyz: np.ndarray,
    ade_values: Sequence[float],
    collision_exposure: np.ndarray,
    *,
    tolerance_m: float = TRAJECTORY_UNIQUE_TOLERANCE_M,
) -> dict[str, float | int | bool]:
    """Summarize planar diversity for one ordered candidate set."""

    trajectories = np.asarray(candidate_xyz, dtype=np.float64)
    if (
        trajectories.ndim != 3
        or trajectories.shape[0] == 0
        or trajectories.shape[2] < 2
        or not np.isfinite(trajectories).all()
    ):
        raise ValueError(
            "Candidate diversity needs finite [candidate,time,>=2] geometry"
        )
    if tolerance_m <= 0:
        raise ValueError("Trajectory uniqueness tolerance must be positive")
    representatives: list[np.ndarray] = []
    for trajectory in trajectories[:, :, :2]:
        if all(
            float(np.max(np.linalg.norm(trajectory - representative, axis=1)))
            > tolerance_m
            for representative in representatives
        ):
            representatives.append(trajectory)
    endpoints = trajectories[:, -1, :2]
    endpoint_spread = 0.0
    for left in range(len(endpoints)):
        for right in range(left + 1, len(endpoints)):
            endpoint_spread = max(
                endpoint_spread,
                float(np.linalg.norm(endpoints[left] - endpoints[right])),
            )
    ade = np.asarray(ade_values, dtype=np.float64)
    exposure = np.asarray(collision_exposure, dtype=np.float64)
    if (
        ade.shape != (len(trajectories),)
        or exposure.shape != ade.shape
        or not np.isfinite(ade).all()
        or not np.isfinite(exposure).all()
    ):
        raise ValueError("Candidate diversity metrics must match the candidate count")
    unique_count = len(representatives)
    return {
        "unique_trajectory_count": unique_count,
        "endpoint_spread_m": endpoint_spread,
        "ade_spread_m": float(ade.max() - ade.min()),
        "recorded_collision_exposure_spread": float(exposure.max() - exposure.min()),
        "nonzero_selection_opportunity": unique_count > 1,
    }


def _require_uniform(values: Sequence[Any], name: str) -> Any:
    fingerprints = {canonical_fingerprint(value) for value in values}
    if len(fingerprints) != 1:
        raise ValueError(f"Test rows mix different {name} values")
    return values[0]


def _selection_gate_configuration(protocol: FrozenProtocol) -> Mapping[str, Any]:
    gates = protocol.payload["acceptance_gates"].get("selection")
    if not isinstance(gates, dict):
        raise ValueError("Frozen protocol acceptance_gates.selection must be an object")
    expected_names = {
        "candidate_0_collision_exposure_relative_reduction_minimum",
        "learned_collision_exposure_no_worse_than_persistence",
        "ade_degradation_m_paired_ci_95_upper_maximum",
        "out_of_bounds_fraction_no_higher_than_candidate_0",
        "observed_fraction_no_lower_than_candidate_0",
    }
    if set(gates) != expected_names:
        raise ValueError(
            "Frozen protocol selection acceptance gates are not supported exactly"
        )
    for name in (
        "candidate_0_collision_exposure_relative_reduction_minimum",
        "ade_degradation_m_paired_ci_95_upper_maximum",
    ):
        value = gates[name]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError(
                f"Frozen protocol selection gate {name} must be non-negative"
            )
    for name in expected_names - {
        "candidate_0_collision_exposure_relative_reduction_minimum",
        "ade_degradation_m_paired_ci_95_upper_maximum",
    }:
        if gates[name] is not True:
            raise ValueError(f"Frozen protocol selection gate {name} must be true")
    return gates


def _selection_gate_decisions(
    protocol: FrozenProtocol,
    policy_aggregates: Mapping[str, Mapping[str, float]],
    comparisons: Mapping[str, Mapping[str, Mapping[str, float | int]]],
) -> dict[str, Any]:
    gates = _selection_gate_configuration(protocol)
    candidate_0 = policy_aggregates["candidate_0"]
    learned = policy_aggregates["learned_selected"]
    persistence = policy_aggregates["persistence_selected"]
    baseline_exposure = candidate_0["collision_exposure"]
    learned_exposure = learned["collision_exposure"]
    minimum_reduction = float(
        gates["candidate_0_collision_exposure_relative_reduction_minimum"]
    )
    if baseline_exposure == 0.0:
        relative_reduction: float | None = None
        reduction_passed = False
        reduction_reason = (
            "Candidate zero has zero mean collision exposure. The benchmark cannot "
            "demonstrate a relative reduction from a zero baseline."
        )
    else:
        relative_reduction = (baseline_exposure - learned_exposure) / baseline_exposure
        reduction_passed = relative_reduction >= minimum_reduction
        reduction_reason = (
            "The relative reduction meets the predeclared minimum."
            if reduction_passed
            else "The relative reduction does not meet the predeclared minimum."
        )

    exposure_no_worse = learned_exposure <= persistence["collision_exposure"]
    ade_interval = comparisons["learned_selected_minus_candidate_0"]["ade_m"]
    ade_upper = float(ade_interval["ci_95_upper"])
    ade_maximum = float(gates["ade_degradation_m_paired_ci_95_upper_maximum"])
    ade_passed = ade_upper <= ade_maximum
    oob_no_higher = (
        learned["out_of_bounds_fraction"] <= candidate_0["out_of_bounds_fraction"]
    )
    observed_no_lower = learned["observed_fraction"] >= candidate_0["observed_fraction"]
    decisions = {
        "candidate_0_collision_exposure_relative_reduction_minimum": {
            "threshold": minimum_reduction,
            "passed": reduction_passed,
            "reason": reduction_reason,
            "candidate_0": baseline_exposure,
            "learned_selected": learned_exposure,
            "relative_reduction": relative_reduction,
        },
        "learned_collision_exposure_no_worse_than_persistence": {
            "required": True,
            "passed": exposure_no_worse,
            "reason": (
                "Learned selection has no higher mean collision exposure."
                if exposure_no_worse
                else "Learned selection has higher mean collision exposure."
            ),
            "learned_selected": learned_exposure,
            "persistence_selected": persistence["collision_exposure"],
        },
        "ade_degradation_m_paired_ci_95_upper_maximum": {
            "threshold_m": ade_maximum,
            "passed": ade_passed,
            "reason": (
                "The paired 95% interval upper bound meets the predeclared maximum."
                if ade_passed
                else "The paired 95% interval upper bound exceeds the predeclared maximum."
            ),
            "paired_difference_m": ade_interval["estimate"],
            "ci_95_lower_m": ade_interval["ci_95_lower"],
            "ci_95_upper_m": ade_upper,
        },
        "out_of_bounds_fraction_no_higher_than_candidate_0": {
            "required": True,
            "passed": oob_no_higher,
            "reason": (
                "Learned selection has no higher mean out-of-bounds fraction."
                if oob_no_higher
                else "Learned selection has a higher mean out-of-bounds fraction."
            ),
            "candidate_0": candidate_0["out_of_bounds_fraction"],
            "learned_selected": learned["out_of_bounds_fraction"],
        },
        "observed_fraction_no_lower_than_candidate_0": {
            "required": True,
            "passed": observed_no_lower,
            "reason": (
                "Learned selection has no lower mean observed fraction."
                if observed_no_lower
                else "Learned selection has a lower mean observed fraction."
            ),
            "candidate_0": candidate_0["observed_fraction"],
            "learned_selected": learned["observed_fraction"],
        },
    }
    return {
        "all_passed": all(decision["passed"] for decision in decisions.values()),
        "decisions": decisions,
    }


def evaluate_selection(
    *,
    protocol: str | Path,
    forecast_comparison: str | Path,
    test_manifest: str | Path,
    train_manifest: str | Path,
    oracle_manifest: str | Path,
    learned_selections: str | Path,
    persistence_selections: str | Path,
    candidate_dir: str | Path,
    validation_manifest: str | Path,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 2026,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate the benchmark join and return summary and per-clip records."""

    test_path = Path(test_manifest).expanduser().resolve()
    train_path = Path(train_manifest).expanduser().resolve()
    oracle_manifest_path = Path(oracle_manifest).expanduser().resolve()
    learned_path = Path(learned_selections).expanduser().resolve()
    persistence_path = Path(persistence_selections).expanduser().resolve()
    frozen_protocol = load_frozen_protocol(protocol)
    protocol_weights = _protocol_rerank_weights(frozen_protocol)
    validate_uncertainty_configuration(
        frozen_protocol, replicates=bootstrap_replicates, seed=bootstrap_seed
    )
    _selection_gate_configuration(frozen_protocol)
    validation_path = Path(validation_manifest).expanduser().resolve()
    test_rows = load_manifest(test_path)
    expected_evaluation_manifest = _evaluation_manifest_binding(test_path, test_rows)
    train_rows = load_manifest(train_path)
    _reject_split_leakage(test_rows, train_rows, "train")
    validation_rows = load_manifest(validation_path)
    _reject_split_leakage(test_rows, validation_rows, "validation")
    for name, path, rows in (
        ("test", test_path, test_rows),
        ("train", train_path, train_rows),
        ("val", validation_path, validation_rows),
    ):
        validate_manifest_binding(
            frozen_protocol,
            name,
            path,
            clips=len(rows),
            chunks=len({str(row["chunk_id"]) for row in rows}),
        )
    validate_source_manifest_binding(frozen_protocol, oracle_manifest_path)
    forecast_report, forecast_provenance = _load_forecast_comparison(
        forecast_comparison,
        protocol=frozen_protocol,
        test_manifest=test_path,
        test_rows=test_rows,
    )

    test_index = _index_rows(test_rows, "test manifest")
    oracle_index = _index_rows(load_manifest(oracle_manifest_path), "oracle manifest")
    test_identities = set(test_index)
    missing_oracle = sorted(test_identities - set(oracle_index))
    if missing_oracle:
        preview = ", ".join(
            f"{clip_id}@{t0_us}" for clip_id, t0_us in missing_oracle[:10]
        )
        raise ValueError(f"Oracle manifest is missing test clips: {preview}")

    selection_indexes: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for method, path in (
        ("learned", learned_path),
        ("persistence", persistence_path),
    ):
        rows = _load_jsonl(path, f"{method} selections")
        index = _index_rows(rows, f"{method} selections")
        missing = sorted(test_identities - set(index))
        extra = sorted(set(index) - test_identities)
        if missing or extra:
            raise ValueError(
                f"{method.capitalize()} selection identities must exactly match the "
                f"frozen test split: missing={len(missing)}, extra={len(extra)}"
            )
        selection_indexes[method] = index
    candidates = _load_candidate_records(candidate_dir, test_identities)

    per_clip: list[dict[str, Any]] = []
    oracle_configs: list[str] = []
    dataset_revisions: list[str] = []
    candidate_configs: list[str] = []
    checkpoint_hashes: dict[str, list[str | None]] = {
        "learned": [],
        "persistence": [],
    }
    run_fingerprints: dict[str, list[str]] = {"learned": [], "persistence": []}
    prediction_runs: dict[str, list[dict[str, Any]]] = {
        "learned": [],
        "persistence": [],
    }
    selection_policies: dict[str, list[str]] = {
        "learned": [],
        "persistence": [],
    }
    rerank_weights: dict[str, list[dict[str, Any]]] = {
        "learned": [],
        "persistence": [],
    }
    candidate_counts: list[int] = []
    candidate_provenance: list[dict[str, Any]] = []
    oracle_provenance: list[dict[str, Any]] = []
    prediction_provenance: dict[str, list[dict[str, Any]]] = {
        "learned": [],
        "persistence": [],
    }
    forecast_prediction_inputs: dict[str, list[dict[str, Any]]] = {
        "learned": [],
        "persistence": [],
    }

    for identity in sorted(test_identities):
        test_row = test_index[identity]
        oracle_row = oracle_index[identity]
        if str(test_row["chunk_id"]) != str(oracle_row["chunk_id"]):
            raise ValueError(
                f"Test and oracle chunk_id differ for {identity[0]}@{identity[1]}"
            )
        test_oracle_path = Path(str(test_row["artifact_path"])).resolve()
        source_oracle_path = Path(str(oracle_row["artifact_path"])).resolve()
        if not test_oracle_path.is_file() or not source_oracle_path.is_file():
            raise FileNotFoundError(
                f"Oracle artifact not found for {identity[0]}@{identity[1]}"
            )
        source_oracle_sha = sha256_file(source_oracle_path)
        test_oracle_sha = (
            source_oracle_sha
            if test_oracle_path == source_oracle_path
            else sha256_file(test_oracle_path)
        )
        if test_oracle_sha != source_oracle_sha:
            raise ValueError(
                f"Frozen test and oracle manifests reference different bytes for "
                f"{identity[0]}@{identity[1]}"
            )
        candidate_record = candidates[identity]
        oracle = _load_oracle_metrics(source_oracle_path, identity, candidate_record)
        for manifest_name, manifest_row in (
            ("test", test_row),
            ("oracle", oracle_row),
        ):
            if (
                "candidate_artifact_sha256" in manifest_row
                and str(manifest_row["candidate_artifact_sha256"])
                != candidate_record["artifact_sha256"]
            ):
                raise ValueError(
                    f"{manifest_name.capitalize()} manifest candidate SHA-256 mismatch "
                    f"for {identity[0]}@{identity[1]}"
                )
            if (
                "oracle_safest_idx" in manifest_row
                and int(manifest_row["oracle_safest_idx"])
                != oracle["oracle_safest_idx"]
            ):
                raise ValueError(
                    f"{manifest_name.capitalize()} manifest safest index mismatch for "
                    f"{identity[0]}@{identity[1]}"
                )
            if (
                "candidate_count" in manifest_row
                and int(manifest_row["candidate_count"]) != oracle["candidate_count"]
            ):
                raise ValueError(
                    f"{manifest_name.capitalize()} manifest candidate count mismatch for "
                    f"{identity[0]}@{identity[1]}"
                )
            if (
                "oracle_config_fingerprint" in manifest_row
                and str(manifest_row["oracle_config_fingerprint"])
                != oracle["oracle_config_fingerprint"]
            ):
                raise ValueError(
                    f"{manifest_name.capitalize()} manifest oracle configuration mismatch "
                    f"for {identity[0]}@{identity[1]}"
                )
            if (
                "dataset_revision" in manifest_row
                and str(manifest_row["dataset_revision"]) != oracle["dataset_revision"]
            ):
                raise ValueError(
                    f"{manifest_name.capitalize()} manifest dataset revision mismatch for "
                    f"{identity[0]}@{identity[1]}"
                )
        reranks = {
            method: _validate_rerank_row(
                selection_indexes[method][identity],
                path.parent,
                identity,
                source_oracle_path,
                source_oracle_sha,
                oracle["candidate_count"],
                method,
                protocol_weights,
                frozen_protocol,
                expected_evaluation_manifest,
            )
            for method, path in (
                ("learned", learned_path),
                ("persistence", persistence_path),
            )
        }
        comfort_index, comfort_scores = _comfort_only_selection(
            reranks["learned"], reranks["persistence"], identity
        )
        indices = {
            "candidate_0": 0,
            "learned_selected": reranks["learned"]["selected_index"],
            "persistence_selected": reranks["persistence"]["selected_index"],
            "comfort_only_selected": comfort_index,
            "recorded_future_oracle": oracle["oracle_safest_idx"],
        }
        policy_metrics: dict[str, dict[str, float]] = {}
        for policy, candidate_index in indices.items():
            trajectory_metrics = candidate_record["metrics"][candidate_index]
            risk_metrics = {
                name: float(values[candidate_index])
                for name, values in oracle["risks"].items()
            }
            policy_metrics[policy] = {
                **trajectory_metrics,
                **risk_metrics,
                "collision_clip_rate": float(risk_metrics["conflict_horizons"] > 0),
            }
        policy_metrics["random_candidate_expectation"] = {
            metric: float(
                np.mean(
                    [
                        (
                            candidate_record["metrics"][candidate_index][metric]
                            if metric in ("ade_m", "fde_m")
                            else (
                                float(
                                    oracle["risks"]["conflict_horizons"][
                                        candidate_index
                                    ]
                                    > 0
                                )
                                if metric == "collision_clip_rate"
                                else float(oracle["risks"][metric][candidate_index])
                            )
                        )
                        for candidate_index in range(oracle["candidate_count"])
                    ]
                )
            )
            for metric in METRIC_METADATA
        }
        diversity = _candidate_diversity(
            candidate_record["pred_xyz"],
            [metrics["ade_m"] for metrics in candidate_record["metrics"]],
            oracle["risks"]["collision_exposure"],
        )
        per_clip.append(
            {
                "clip_id": identity[0],
                "t0_us": identity[1],
                "chunk_id": str(test_row["chunk_id"]),
                "candidate_count": oracle["candidate_count"],
                "selected_indices": indices,
                "comfort_only_scores": comfort_scores,
                "learned_changed_from_candidate_0": bool(
                    indices["learned_selected"] != 0
                ),
                "learned_oracle_agreement": bool(
                    indices["learned_selected"] == indices["recorded_future_oracle"]
                ),
                "persistence_changed_from_candidate_0": bool(
                    indices["persistence_selected"] != 0
                ),
                "persistence_oracle_agreement": bool(
                    indices["persistence_selected"] == indices["recorded_future_oracle"]
                ),
                "learned_persistence_agreement": bool(
                    indices["learned_selected"] == indices["persistence_selected"]
                ),
                "comfort_only_changed_from_candidate_0": bool(
                    indices["comfort_only_selected"] != 0
                ),
                "comfort_only_oracle_agreement": bool(
                    indices["comfort_only_selected"]
                    == indices["recorded_future_oracle"]
                ),
                "oracle_changed_from_candidate_0": bool(
                    indices["recorded_future_oracle"] != 0
                ),
                "candidate_diversity": diversity,
                "metrics": policy_metrics,
            }
        )
        oracle_configs.append(oracle["oracle_config_fingerprint"])
        dataset_revisions.append(oracle["dataset_revision"])
        candidate_configs.append(str(candidate_record["config_fingerprint"]))
        for method, rerank in reranks.items():
            checkpoint_hashes[method].append(rerank["checkpoint_sha256"])
            run_fingerprints[method].append(rerank["prediction_run_fingerprint"])
            prediction_runs[method].append(rerank["prediction_run"])
            selection_policies[method].append(rerank["selection_policy"])
            rerank_weights[method].append(rerank["weights"])
            prediction_provenance[method].append(
                {
                    "clip_id": identity[0],
                    "t0_us": identity[1],
                    "prediction_sha256": rerank["prediction_sha256"],
                }
            )
            forecast_prediction_inputs[method].append(
                {
                    "clip_id": identity[0],
                    "t0_us": identity[1],
                    "path": rerank["prediction_path"],
                    "sha256": rerank["prediction_sha256"],
                }
            )
        candidate_counts.append(oracle["candidate_count"])
        candidate_provenance.append(
            {
                "clip_id": identity[0],
                "t0_us": identity[1],
                "record_sha256": candidate_record["record_sha256"],
                "artifact_sha256": candidate_record["artifact_sha256"],
            }
        )
        oracle_provenance.append(
            {
                "clip_id": identity[0],
                "t0_us": identity[1],
                "artifact_sha256": source_oracle_sha,
            }
        )

    uniform_oracle_config = _require_uniform(oracle_configs, "oracle configurations")
    uniform_dataset_revision = _require_uniform(
        dataset_revisions, "oracle dataset revisions"
    )
    uniform_candidate_config = _require_uniform(
        candidate_configs, "candidate configurations"
    )
    uniform_checkpoints = {
        method: _require_uniform(values, f"{method} model checkpoints")
        for method, values in checkpoint_hashes.items()
    }
    uniform_runs = {
        method: _require_uniform(values, f"{method} prediction runs")
        for method, values in run_fingerprints.items()
    }
    uniform_prediction_runs = {
        method: _require_uniform(values, f"{method} prediction-run metadata")
        for method, values in prediction_runs.items()
    }
    _require_recomputed_forecast_comparison(
        forecast_report,
        protocol=frozen_protocol,
        test_manifest=test_path,
        learned_run=uniform_prediction_runs["learned"],
        prediction_inputs=forecast_prediction_inputs,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    uniform_policy = _require_uniform(
        selection_policies["learned"] + selection_policies["persistence"],
        "selection policies",
    )
    uniform_weights = _require_uniform(
        rerank_weights["learned"] + rerank_weights["persistence"],
        "reranker weights",
    )
    require_exact_mapping(
        frozen_protocol.payload["trajectory_selection"],
        {
            "primary_outcome": "collision_exposure",
            "baseline": "candidate_0",
            "learned_comparator": "persistence_selected",
            "reranker_weights": uniform_weights,
        },
        "trajectory_selection",
    )
    uniform_candidate_count = _require_uniform(candidate_counts, "candidate counts")

    cluster_ids = [str(row["chunk_id"]) for row in per_clip]
    policy_aggregates: dict[str, dict[str, float]] = {}
    for policy in POLICIES:
        policy_aggregates[policy] = {
            metric: float(np.mean([row["metrics"][policy][metric] for row in per_clip]))
            for metric in METRIC_METADATA
        }

    comparisons: dict[str, dict[str, dict[str, float | int]]] = {}
    for comparison_name, policy in (
        ("learned_selected_minus_candidate_0", "learned_selected"),
        ("persistence_selected_minus_candidate_0", "persistence_selected"),
        ("comfort_only_selected_minus_candidate_0", "comfort_only_selected"),
        (
            "random_candidate_expectation_minus_candidate_0",
            "random_candidate_expectation",
        ),
        ("recorded_future_oracle_minus_candidate_0", "recorded_future_oracle"),
    ):
        comparisons[comparison_name] = {}
        for metric in METRIC_METADATA:
            differences = np.asarray(
                [
                    row["metrics"][policy][metric]
                    - row["metrics"]["candidate_0"][metric]
                    for row in per_clip
                ],
                dtype=np.float64,
            )
            comparisons[comparison_name][metric] = cluster_bootstrap_interval(
                differences,
                cluster_ids,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
            )
    comparisons["learned_selected_minus_persistence_selected"] = {}
    for metric in METRIC_METADATA:
        differences = np.asarray(
            [
                row["metrics"]["learned_selected"][metric]
                - row["metrics"]["persistence_selected"][metric]
                for row in per_clip
            ],
            dtype=np.float64,
        )
        comparisons["learned_selected_minus_persistence_selected"][metric] = (
            cluster_bootstrap_interval(
                differences,
                cluster_ids,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
            )
        )

    rates = {
        "learned_change_from_candidate_0": cluster_bootstrap_interval(
            [float(row["learned_changed_from_candidate_0"]) for row in per_clip],
            cluster_ids,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
        "learned_oracle_agreement": cluster_bootstrap_interval(
            [float(row["learned_oracle_agreement"]) for row in per_clip],
            cluster_ids,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
        "persistence_change_from_candidate_0": cluster_bootstrap_interval(
            [float(row["persistence_changed_from_candidate_0"]) for row in per_clip],
            cluster_ids,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
        "persistence_oracle_agreement": cluster_bootstrap_interval(
            [float(row["persistence_oracle_agreement"]) for row in per_clip],
            cluster_ids,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
        "learned_persistence_agreement": cluster_bootstrap_interval(
            [float(row["learned_persistence_agreement"]) for row in per_clip],
            cluster_ids,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
        "comfort_only_change_from_candidate_0": cluster_bootstrap_interval(
            [float(row["comfort_only_changed_from_candidate_0"]) for row in per_clip],
            cluster_ids,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
        "comfort_only_oracle_agreement": cluster_bootstrap_interval(
            [float(row["comfort_only_oracle_agreement"]) for row in per_clip],
            cluster_ids,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
        "random_change_from_candidate_0_expectation": cluster_bootstrap_interval(
            [1.0 - 1.0 / row["candidate_count"] for row in per_clip],
            cluster_ids,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
        "random_oracle_agreement_expectation": cluster_bootstrap_interval(
            [1.0 / row["candidate_count"] for row in per_clip],
            cluster_ids,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
        "oracle_change_from_candidate_0": cluster_bootstrap_interval(
            [float(row["oracle_changed_from_candidate_0"]) for row in per_clip],
            cluster_ids,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
    }
    diversity_metrics = (
        "unique_trajectory_count",
        "endpoint_spread_m",
        "ade_spread_m",
        "recorded_collision_exposure_spread",
    )
    diversity_summary = {
        "definition": {
            "trajectory_unique_tolerance_m": TRAJECTORY_UNIQUE_TOLERANCE_M,
            "unique_trajectory_rule": (
                "A candidate is new when its maximum planar waypoint distance from "
                "every earlier representative exceeds the tolerance."
            ),
            "endpoint_spread": "Maximum pairwise planar endpoint distance per clip.",
            "ade_spread": "Maximum candidate ADE minus minimum candidate ADE per clip.",
            "recorded_collision_exposure_spread": (
                "Maximum candidate exposure minus minimum candidate exposure per clip."
            ),
            "nonzero_selection_opportunity": (
                "The candidate set contains more than one unique planar trajectory."
            ),
        },
        "macro_means": {
            metric: float(
                np.mean([row["candidate_diversity"][metric] for row in per_clip])
            )
            for metric in diversity_metrics
        },
        "minimums": {
            metric: float(
                np.min([row["candidate_diversity"][metric] for row in per_clip])
            )
            for metric in diversity_metrics
        },
        "maximums": {
            metric: float(
                np.max([row["candidate_diversity"][metric] for row in per_clip])
            )
            for metric in diversity_metrics
        },
        "nonzero_selection_opportunity_rate": cluster_bootstrap_interval(
            [
                float(row["candidate_diversity"]["nonzero_selection_opportunity"])
                for row in per_clip
            ],
            cluster_ids,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
    }
    selection_gates = _selection_gate_decisions(
        frozen_protocol, policy_aggregates, comparisons
    )
    forecast_gates = forecast_report["acceptance_gates"]
    benchmark_acceptance = {
        "all_passed": bool(
            forecast_gates["all_passed"] and selection_gates["all_passed"]
        ),
        "rule": "forecast.all_passed and selection.all_passed",
        "components": {
            "forecast": {"all_passed": forecast_gates["all_passed"]},
            "selection": {"all_passed": selection_gates["all_passed"]},
        },
    }
    if sha256_file(forecast_provenance["path"]) != forecast_provenance["sha256"]:
        raise ValueError("Forecast comparison changed while selection was evaluated")
    summary: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "protocol": frozen_protocol.provenance(),
        "evaluation": {
            "clips": len(per_clip),
            "source_chunks": len(set(cluster_ids)),
            "candidate_count": uniform_candidate_count,
            "unit_of_analysis": "clip",
            "confidence_interval": "paired source-chunk cluster bootstrap percentile",
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
            "uncertainty_scope": {
                "held_out_source_chunk_sampling": True,
                "training_seed_variability_included": False,
                "statement": (
                    "Intervals cover held-out source-chunk sampling only. They do not "
                    "cover optimization-seed variability after checkpoint selection."
                ),
            },
        },
        "run": {
            "learned": {
                "checkpoint_sha256": uniform_checkpoints["learned"],
                "prediction_run_fingerprint": uniform_runs["learned"],
                "prediction_run": uniform_prediction_runs["learned"],
            },
            "persistence": {
                "checkpoint_sha256": uniform_checkpoints["persistence"],
                "prediction_run_fingerprint": uniform_runs["persistence"],
                "prediction_run": uniform_prediction_runs["persistence"],
            },
            "selection_policy": uniform_policy,
            "reranker_weights": uniform_weights,
            "comfort_only_formula": (
                "acceleration_weight*mean_acceleration_mps2 + "
                "jerk_weight*mean_jerk_mps3 + "
                "curvature_weight*max_curvature_inv_m - progress_weight*progress_m"
            ),
            "comfort_only_tie_break": "lowest candidate index",
            "candidate_config_fingerprint": uniform_candidate_config,
            "oracle_config_fingerprint": uniform_oracle_config,
            "dataset_revision": uniform_dataset_revision,
        },
        "metric_metadata": METRIC_METADATA,
        "policy_macro_means": policy_aggregates,
        "paired_differences": comparisons,
        "acceptance_gates": selection_gates,
        "benchmark_acceptance": benchmark_acceptance,
        "selection_rates": rates,
        "candidate_diversity": diversity_summary,
        "provenance": {
            "test_manifest": str(test_path),
            "test_manifest_sha256": sha256_file(test_path),
            "train_manifest": str(train_path),
            "train_manifest_sha256": sha256_file(train_path),
            "validation_manifest": str(validation_path),
            "validation_manifest_sha256": sha256_file(validation_path),
            "oracle_manifest": str(oracle_manifest_path),
            "oracle_manifest_sha256": sha256_file(oracle_manifest_path),
            "learned_selections": str(learned_path),
            "learned_selections_sha256": sha256_file(learned_path),
            "persistence_selections": str(persistence_path),
            "persistence_selections_sha256": sha256_file(persistence_path),
            "forecast_comparison": forecast_provenance,
            "candidate_dir": str(Path(candidate_dir).expanduser().resolve()),
            "candidate_subset_fingerprint": canonical_fingerprint(candidate_provenance),
            "oracle_subset_fingerprint": canonical_fingerprint(oracle_provenance),
            "learned_prediction_subset_fingerprint": canonical_fingerprint(
                prediction_provenance["learned"]
            ),
            "persistence_prediction_subset_fingerprint": canonical_fingerprint(
                prediction_provenance["persistence"]
            ),
        },
    }
    return summary, per_clip


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--forecast-comparison", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--oracle-manifest", required=True)
    parser.add_argument("--learned-selections", required=True)
    parser.add_argument("--persistence-selections", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output", required=True, help="Summary JSON path")
    parser.add_argument("--per-clip-output", help="Optional audit JSONL path")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary, per_clip = evaluate_selection(
        protocol=args.protocol,
        forecast_comparison=args.forecast_comparison,
        test_manifest=args.test_manifest,
        train_manifest=args.train_manifest,
        validation_manifest=args.validation_manifest,
        oracle_manifest=args.oracle_manifest,
        learned_selections=args.learned_selections,
        persistence_selections=args.persistence_selections,
        candidate_dir=args.candidate_dir,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    if args.per_clip_output:
        write_jsonl(per_clip, args.per_clip_output)
    # The summary is the completion marker for this two-file report. Write it
    # only after the optional audit rows are durable, so an interrupted stage
    # can safely replace an uncommitted audit file and try again.
    write_json(summary, args.output)
    print(
        f"Evaluated {summary['evaluation']['clips']} held-out clips across "
        f"{summary['evaluation']['source_chunks']} source chunks"
    )


if __name__ == "__main__":
    main()
