"""Compare learned and persistence occupancy forecasts on a frozen test set.

This module treats the manifest and prediction directories as immutable benchmark
inputs. It validates their identities, byte provenance, tensor contracts, and run
metadata before it reports equally weighted per-clip metrics. Uncertainty comes
from paired source-chunk cluster resampling, never from treating BEV cells as
independent observations.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .data import load_manifest, load_occupancy_artifact
from .protocol import (
    FrozenProtocol,
    load_frozen_protocol,
    require_exact_mapping,
    validate_manifest_binding,
    validate_uncertainty_configuration,
)
from .prediction_auth import authenticate_prediction_run, load_prediction_run
from .runtime import (
    canonical_fingerprint,
    prediction_filename,
    sha256_file,
    write_json,
)


PREDICTION_SCHEMA_VERSION = 2
COMPARISON_SCHEMA_VERSION = 1
SHORT_HORIZONS_S = (0.5, 1.0, 2.0)
METRIC_NAMES = ("iou", "average_precision", "brier", "positive_prevalence")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, help="Frozen benchmark protocol JSON")
    parser.add_argument("--manifest", required=True, help="Frozen test JSONL manifest")
    parser.add_argument(
        "--selection-record",
        required=True,
        help="Checkpoint-selection JSON used to produce the learned predictions",
    )
    parser.add_argument(
        "--evaluation-audit",
        required=True,
        help="Zero-overlap evaluation-partition audit for the learned predictions",
    )
    parser.add_argument("--learned-prediction-dir", required=True)
    parser.add_argument("--persistence-prediction-dir", required=True)
    parser.add_argument("--output", required=True, help="Comparison JSON path")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--ap-bins", type=int, default=1000)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    return parser


def _validate_forecast_protocol(
    protocol: FrozenProtocol,
    *,
    threshold: float,
    ap_bins: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> None:
    configuration = protocol.payload["forecast_evaluation"]
    expected_configuration = {
        "metrics": ["average_precision", "brier", "iou"],
        "short_horizons_s": list(SHORT_HORIZONS_S),
        "probability_threshold": threshold,
        "average_precision_bins": ap_bins,
        "post_test_threshold_tuning": False,
        "comparison": "learned_minus_persistence",
    }
    require_exact_mapping(
        configuration, expected_configuration, "forecast_evaluation"
    )
    expected_gates = {
        "average_precision": "learned_higher_than_persistence",
        "brier": "learned_lower_than_persistence",
        "iou": "learned_higher_than_persistence",
    }
    require_exact_mapping(
        protocol.payload["acceptance_gates"].get("forecast"),
        expected_gates,
        "acceptance_gates.forecast",
    )
    validate_uncertainty_configuration(
        protocol, replicates=bootstrap_replicates, seed=bootstrap_seed
    )


def _forecast_gate_decisions(
    protocol: FrozenProtocol,
    methods: Mapping[str, Any],
    paired: Mapping[str, Any],
) -> dict[str, Any]:
    rules = protocol.payload["acceptance_gates"]["forecast"]
    learned = methods["learned"]["short_horizon"]
    persistence = methods["persistence"]["short_horizon"]
    paired_short = paired["short_horizon"]
    decisions: dict[str, Any] = {}
    for metric, rule in rules.items():
        learned_value = learned[metric]
        persistence_value = persistence[metric]
        difference = paired_short[metric]["estimate"]
        values_defined = all(
            value is not None and np.isfinite(float(value))
            for value in (learned_value, persistence_value, difference)
        )
        if not values_defined:
            passed = False
            reason = "The short-horizon point estimates are not all defined."
        elif rule == "learned_higher_than_persistence":
            passed = float(difference) > 0.0
            reason = (
                "The learned short-horizon point estimate is higher."
                if passed
                else "The learned short-horizon point estimate is not higher."
            )
        elif rule == "learned_lower_than_persistence":
            passed = float(difference) < 0.0
            reason = (
                "The learned short-horizon point estimate is lower."
                if passed
                else "The learned short-horizon point estimate is not lower."
            )
        else:  # Guarded by _validate_forecast_protocol.
            raise AssertionError(f"Unsupported forecast gate rule: {rule}")
        decisions[metric] = {
            "rule": rule,
            "passed": passed,
            "reason": reason,
            "learned": learned_value,
            "persistence": persistence_value,
            "learned_minus_persistence": difference,
            "paired_percentile_95_ci": paired_short[metric]["percentile_95_ci"],
        }
    return {
        "all_passed": all(decision["passed"] for decision in decisions.values()),
        "basis": (
            "Predeclared short-horizon clip-macro point estimates. Confidence "
            "intervals are reported but do not change these directional gates."
        ),
        "decisions": decisions,
    }


def _scalar(archive: Mapping[str, np.ndarray], name: str) -> Any:
    if name not in archive:
        raise ValueError(f"Prediction artifact is missing {name}")
    value = np.asarray(archive[name])
    if value.shape != ():
        raise ValueError(f"Prediction field {name} must be scalar")
    return value.item()


def _require_sha256(value: str, label: str) -> str:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _mean_or_none(values: np.ndarray) -> tuple[float | None, int]:
    finite = np.isfinite(values)
    count = int(finite.sum())
    return (float(values[finite].mean()) if count else None, count)


def _average_precision(
    scores: np.ndarray, labels: np.ndarray, *, bins: int
) -> float | None:
    """Return step-integrated AP from fixed score histograms.

    The implementation matches the repository's streaming metric: bins are
    traversed from high to low score, and precision is integrated against each
    positive recall increment. It is not trapezoidal PR-AUC.
    """

    positives_total = int(labels.sum())
    if positives_total == 0:
        return None
    positive_hist = np.histogram(scores[labels], bins=bins, range=(0.0, 1.0))[0]
    negative_hist = np.histogram(scores[~labels], bins=bins, range=(0.0, 1.0))[0]
    positives = positive_hist[::-1].cumsum(dtype=np.float64)
    negatives = negative_hist[::-1].cumsum(dtype=np.float64)
    recall = positives / positives_total
    precision = positives / np.maximum(positives + negatives, 1.0)
    recall_step = np.diff(np.concatenate(([0.0], recall)))
    return float(np.sum(recall_step * precision))


def _clip_metrics(
    probability: np.ndarray,
    target: np.ndarray,
    visibility: np.ndarray,
    *,
    threshold: float,
    ap_bins: int,
) -> np.ndarray:
    """Return [horizon, metric] values for one clip."""

    values = np.full((target.shape[0], len(METRIC_NAMES)), np.nan, dtype=np.float64)
    for horizon in range(target.shape[0]):
        mask = visibility[horizon]
        scores = probability[horizon][mask]
        labels = target[horizon][mask]
        if scores.size == 0:
            continue
        predicted = scores >= threshold
        union = int(np.logical_or(predicted, labels).sum())
        if union:
            values[horizon, 0] = float(np.logical_and(predicted, labels).sum() / union)
        average_precision = _average_precision(scores, labels, bins=ap_bins)
        if average_precision is not None:
            values[horizon, 1] = average_precision
        values[horizon, 2] = float(
            np.mean(np.square(scores.astype(np.float64) - labels.astype(np.float64)))
        )
        values[horizon, 3] = float(labels.mean())
    return values


def _manifest_inputs(manifest: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_manifest(manifest)
    artifact_paths: set[Path] = set()
    records: list[dict[str, Any]] = []
    reference_geometry: dict[str, Any] | None = None
    for row in rows:
        identity = (str(row["clip_id"]), int(row["t0_us"]))
        chunk_id = str(row["chunk_id"])
        if not chunk_id:
            raise ValueError(f"Manifest chunk_id is empty for {identity}")
        if "artifact_sha256" not in row:
            raise ValueError(f"Frozen manifest has no artifact_sha256 for {identity}")
        declared_sha256 = _require_sha256(
            str(row["artifact_sha256"]), f"Manifest artifact_sha256 for {identity}"
        )
        artifact_path = Path(str(row["artifact_path"])).resolve()
        if artifact_path in artifact_paths:
            raise ValueError(f"Manifest references an oracle artifact more than once: {artifact_path}")
        artifact_paths.add(artifact_path)
        actual_sha256 = sha256_file(artifact_path)
        if actual_sha256 != declared_sha256:
            raise ValueError(f"Oracle artifact SHA-256 differs from the manifest for {identity}")
        artifact = load_occupancy_artifact(artifact_path)
        embedded_identity = (artifact["identity"].clip_id, artifact["identity"].t0_us)
        if embedded_identity != identity:
            raise ValueError(f"Oracle artifact identity differs from the manifest for {identity}")
        metadata = artifact["metadata"]
        geometry = {
            "target_shape": list(artifact["future_occupancy"].shape),
            "horizons_s": metadata.horizons_s.tolist(),
            "resolution_m": metadata.resolution_m,
            "grid_origin_xy_m": metadata.grid_origin_xy_m.tolist(),
            "coordinate_frame": metadata.coordinate_frame,
        }
        if reference_geometry is None:
            reference_geometry = geometry
        elif geometry != reference_geometry:
            raise ValueError(f"Oracle geometry or horizons differ for {identity}")
        records.append(
            {
                "clip_id": identity[0],
                "t0_us": identity[1],
                "chunk_id": chunk_id,
                "artifact_path": artifact_path,
                "artifact_sha256": actual_sha256,
            }
        )
    assert reference_geometry is not None
    horizons = np.asarray(reference_geometry["horizons_s"], dtype=np.float64)
    for required in SHORT_HORIZONS_S:
        matches = np.flatnonzero(np.isclose(horizons, required, rtol=0.0, atol=1e-6))
        if matches.size != 1:
            raise ValueError(
                f"Frozen benchmark must contain exactly one {required:g}-second horizon"
            )
    index = [
        {
            "artifact_sha256": record["artifact_sha256"],
            "chunk_id": record["chunk_id"],
            "clip_id": record["clip_id"],
            "t0_us": record["t0_us"],
        }
        for record in records
    ]
    provenance = {
        "manifest_path": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "dataset_fingerprint": canonical_fingerprint(index),
        "clips": len(records),
        "chunks": len({record["chunk_id"] for record in records}),
        "geometry": reference_geometry,
        "oracle_artifacts": [
            {
                "clip_id": record["clip_id"],
                "t0_us": record["t0_us"],
                "chunk_id": record["chunk_id"],
                "path": str(record["artifact_path"]),
                "sha256": record["artifact_sha256"],
            }
            for record in records
        ],
    }
    return records, provenance


def _prediction_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {directory}")
    paths = sorted(directory.glob("*.prediction.npz"))
    if not paths:
        raise ValueError(f"No prediction artifacts found in {directory}")
    symbolic_links = [str(path) for path in paths if path.is_symlink()]
    if symbolic_links:
        raise ValueError(f"Prediction artifacts must not be symbolic links: {symbolic_links[:10]}")
    return [path.resolve() for path in paths]


def _load_prediction(
    path: Path,
    *,
    expected_method: str,
    expected_record: Mapping[str, Any],
    oracle: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    required = {
        "schema_version",
        "clip_id",
        "t0_us",
        "occupancy_prob",
        "horizons_s",
        "resolution_m",
        "grid_origin_xy_m",
        "coordinate_frame",
        "source_artifact_sha256",
        "producer_method",
        "checkpoint_sha256",
        "prediction_run_json",
        "prediction_run_fingerprint",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"Prediction artifact {path} is missing {missing}")
        schema_version = int(_scalar(archive, "schema_version"))
        clip_id = str(_scalar(archive, "clip_id"))
        t0_us = int(_scalar(archive, "t0_us"))
        producer_method = str(_scalar(archive, "producer_method"))
        checkpoint_sha256 = str(_scalar(archive, "checkpoint_sha256"))
        prediction_run = load_prediction_run(archive, artifact_path=path)
        run_fingerprint = prediction_run.fingerprint
        source_sha256 = str(_scalar(archive, "source_artifact_sha256"))
        coordinate_frame = str(_scalar(archive, "coordinate_frame"))
        resolution_m = float(_scalar(archive, "resolution_m"))
        probability_raw = np.asarray(archive["occupancy_prob"])
        if probability_raw.dtype.kind != "f":
            raise ValueError(f"occupancy_prob must use a floating-point dtype: {path}")
        probability = probability_raw.astype(np.float64)
        horizons_s = np.asarray(archive["horizons_s"], dtype=np.float32)
        grid_origin = np.asarray(archive["grid_origin_xy_m"], dtype=np.float32)
    identity = (clip_id, t0_us)
    expected_identity = (str(expected_record["clip_id"]), int(expected_record["t0_us"]))
    if identity != expected_identity:
        raise ValueError(f"Prediction identity differs from its expected file: {path}")
    if schema_version != PREDICTION_SCHEMA_VERSION:
        raise ValueError(f"Unsupported prediction schema version in {path}")
    if producer_method != expected_method:
        raise ValueError(
            f"Expected producer_method={expected_method!r}, got {producer_method!r}: {path}"
        )
    _require_sha256(run_fingerprint, f"Prediction run fingerprint in {path}")
    _require_sha256(source_sha256, f"Prediction source artifact SHA-256 in {path}")
    if expected_method == "learned":
        _require_sha256(checkpoint_sha256, f"Learned checkpoint SHA-256 in {path}")
    elif checkpoint_sha256:
        raise ValueError(f"Persistence prediction claims a checkpoint: {path}")
    if prediction_run.payload.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError(f"Prediction checkpoint differs from prediction_run_json: {path}")
    if source_sha256 != expected_record["artifact_sha256"]:
        raise ValueError(f"Prediction source oracle artifact SHA-256 differs for {identity}")
    target = oracle["future_occupancy"]
    metadata = oracle["metadata"]
    if probability.shape != target.shape:
        raise ValueError(f"Prediction shape differs from the oracle target for {identity}")
    if not np.isfinite(probability).all() or np.any(probability < 0) or np.any(probability > 1):
        raise ValueError(f"Prediction probabilities must be finite values in [0,1] for {identity}")
    if not np.array_equal(horizons_s, metadata.horizons_s):
        raise ValueError(f"Prediction horizons differ from the oracle for {identity}")
    if grid_origin.shape != (2,) or not np.array_equal(grid_origin, metadata.grid_origin_xy_m):
        raise ValueError(f"Prediction grid origin differs from the oracle for {identity}")
    if not np.isfinite(resolution_m) or resolution_m != metadata.resolution_m:
        raise ValueError(f"Prediction resolution differs from the oracle for {identity}")
    if coordinate_frame != metadata.coordinate_frame:
        raise ValueError(f"Prediction coordinate frame differs from the oracle for {identity}")
    return probability, {
        "clip_id": clip_id,
        "t0_us": t0_us,
        "path": str(path),
        "sha256": sha256_file(path),
        "source_artifact_sha256": source_sha256,
        "checkpoint_sha256": checkpoint_sha256 or None,
        "prediction_run_fingerprint": run_fingerprint,
        "prediction_run_json": prediction_run.canonical_json,
        "_prediction_run": prediction_run,
        "producer_method": producer_method,
    }


def _group_clip_values(values: np.ndarray, horizon_indices: Sequence[int]) -> np.ndarray:
    selected = values[:, horizon_indices, :]
    grouped = np.full((values.shape[0], values.shape[2]), np.nan, dtype=np.float64)
    for clip in range(values.shape[0]):
        for metric in range(values.shape[2]):
            finite = selected[clip, :, metric]
            finite = finite[np.isfinite(finite)]
            if finite.size:
                grouped[clip, metric] = finite.mean()
    return grouped


def _metric_summary(values: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric_index, metric in enumerate(METRIC_NAMES):
        value, count = _mean_or_none(values[:, metric_index])
        result[metric] = value
        result[f"{metric}_contributing_clips"] = count
    return result


def _method_summary(values: np.ndarray, horizons_s: np.ndarray) -> dict[str, Any]:
    short_indices = [
        int(np.flatnonzero(np.isclose(horizons_s, value, rtol=0.0, atol=1e-6))[0])
        for value in SHORT_HORIZONS_S
    ]
    return {
        "per_horizon": [
            {
                "horizon_s": float(horizon),
                **_metric_summary(values[:, index, :]),
            }
            for index, horizon in enumerate(horizons_s)
        ],
        "short_horizon": {
            "horizons_s": list(SHORT_HORIZONS_S),
            **_metric_summary(_group_clip_values(values, short_indices)),
        },
        "all_horizon": {
            "horizons_s": horizons_s.tolist(),
            **_metric_summary(_group_clip_values(values, list(range(horizons_s.size)))),
        },
    }


def _bootstrap_intervals(
    differences: np.ndarray,
    chunk_ids: Sequence[str],
    *,
    replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Bootstrap columns of paired clip differences by source chunk."""

    chunks = sorted(set(chunk_ids))
    chunk_lookup = {chunk: index for index, chunk in enumerate(chunks)}
    cluster_sum = np.zeros((differences.shape[1], len(chunks)), dtype=np.float64)
    cluster_count = np.zeros_like(cluster_sum)
    for clip, chunk_id in enumerate(chunk_ids):
        finite = np.isfinite(differences[clip])
        cluster_sum[finite, chunk_lookup[chunk_id]] += differences[clip, finite]
        cluster_count[finite, chunk_lookup[chunk_id]] += 1.0
    distributions = np.full((replicates, differences.shape[1]), np.nan, dtype=np.float64)
    rng = np.random.default_rng(seed)
    probability = np.full(len(chunks), 1.0 / len(chunks), dtype=np.float64)
    batch_size = 256
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        draws = rng.multinomial(len(chunks), probability, size=stop - start)
        numerator = draws @ cluster_sum.T
        denominator = draws @ cluster_count.T
        np.divide(
            numerator,
            denominator,
            out=distributions[start:stop],
            where=denominator > 0,
        )
    results: list[dict[str, Any]] = []
    for index in range(differences.shape[1]):
        observed = differences[:, index]
        estimate, paired_clips = _mean_or_none(observed)
        distribution = distributions[:, index]
        distribution = distribution[np.isfinite(distribution)]
        interval = (
            [float(value) for value in np.quantile(distribution, [0.025, 0.975])]
            if distribution.size
            else [None, None]
        )
        results.append(
            {
                "estimate": estimate,
                "percentile_95_ci": interval,
                "paired_clips": paired_clips,
                "bootstrap_valid_replicates": int(distribution.size),
            }
        )
    return results


def _paired_summary(
    learned: np.ndarray,
    persistence: np.ndarray,
    horizons_s: np.ndarray,
    chunk_ids: Sequence[str],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    groups: list[tuple[str, float | list[float], np.ndarray, np.ndarray]] = []
    for index, horizon in enumerate(horizons_s):
        groups.append(
            ("per_horizon", float(horizon), learned[:, index, :], persistence[:, index, :])
        )
    short_indices = [
        int(np.flatnonzero(np.isclose(horizons_s, value, rtol=0.0, atol=1e-6))[0])
        for value in SHORT_HORIZONS_S
    ]
    groups.extend(
        (
            ("short_horizon", list(SHORT_HORIZONS_S), _group_clip_values(learned, short_indices), _group_clip_values(persistence, short_indices)),
            ("all_horizon", horizons_s.tolist(), _group_clip_values(learned, list(range(horizons_s.size))), _group_clip_values(persistence, list(range(horizons_s.size)))),
        )
    )
    difference_columns: list[np.ndarray] = []
    for _, _, learned_values, persistence_values in groups:
        paired = np.full_like(learned_values, np.nan)
        finite = np.isfinite(learned_values) & np.isfinite(persistence_values)
        paired[finite] = learned_values[finite] - persistence_values[finite]
        difference_columns.extend(paired[:, index] for index in range(len(METRIC_NAMES)))
    intervals = _bootstrap_intervals(
        np.stack(difference_columns, axis=1),
        chunk_ids,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    cursor = 0
    per_horizon: list[dict[str, Any]] = []
    aggregate: dict[str, Any] = {}
    for group_name, horizon_values, _, _ in groups:
        metrics = {
            metric: intervals[cursor + index]
            for index, metric in enumerate(METRIC_NAMES)
        }
        cursor += len(METRIC_NAMES)
        if group_name == "per_horizon":
            per_horizon.append({"horizon_s": horizon_values, **metrics})
        else:
            aggregate[group_name] = {"horizons_s": horizon_values, **metrics}
    return {"per_horizon": per_horizon, **aggregate}


def compare_forecasts(
    *,
    protocol: str | Path,
    manifest: str | Path,
    learned_prediction_dir: str | Path,
    persistence_prediction_dir: str | Path,
    selection_record: str | Path,
    evaluation_audit: str | Path,
    threshold: float = 0.5,
    ap_bins: int = 1000,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 2026,
) -> dict[str, Any]:
    """Validate benchmark inputs and return a paired forecast comparison."""

    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and in [0,1]")
    if ap_bins < 2:
        raise ValueError("ap_bins must be at least 2")
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    frozen_protocol = load_frozen_protocol(protocol)
    _validate_forecast_protocol(
        frozen_protocol,
        threshold=threshold,
        ap_bins=ap_bins,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    manifest_path = Path(manifest).expanduser().resolve()
    records, manifest_provenance = _manifest_inputs(manifest_path)
    validate_manifest_binding(
        frozen_protocol,
        "test",
        manifest_path,
        actual_sha256=manifest_provenance["manifest_sha256"],
        clips=manifest_provenance["clips"],
        chunks=manifest_provenance["chunks"],
        dataset_fingerprint=manifest_provenance["dataset_fingerprint"],
    )
    if manifest_provenance["chunks"] < 2:
        raise ValueError("Chunk-cluster uncertainty requires at least two source chunks")
    identity_to_record = {
        (record["clip_id"], record["t0_us"]): record for record in records
    }
    expected_paths: dict[str, dict[tuple[str, int], Path]] = {}
    prediction_roots = {
        "learned": Path(learned_prediction_dir).expanduser().resolve(),
        "persistence": Path(persistence_prediction_dir).expanduser().resolve(),
    }
    for method, root in prediction_roots.items():
        actual = set(_prediction_files(root))
        paths = {
            identity: (root / prediction_filename(*identity)).resolve()
            for identity in identity_to_record
        }
        expected = set(paths.values())
        if actual != expected:
            missing = sorted(str(path) for path in expected - actual)
            unexpected = sorted(str(path) for path in actual - expected)
            raise ValueError(
                f"{method} prediction file set differs from the frozen manifest; "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}"
            )
        expected_paths[method] = paths

    horizons_s = np.asarray(manifest_provenance["geometry"]["horizons_s"], dtype=np.float64)
    values = {
        method: np.full(
            (len(records), horizons_s.size, len(METRIC_NAMES)), np.nan, dtype=np.float64
        )
        for method in prediction_roots
    }
    prediction_provenance: dict[str, list[dict[str, Any]]] = {
        method: [] for method in prediction_roots
    }
    for clip_index, record in enumerate(records):
        identity = (record["clip_id"], record["t0_us"])
        oracle = load_occupancy_artifact(record["artifact_path"])
        target = oracle["future_occupancy"].astype(bool)
        visibility = oracle["future_visibility"].astype(bool)
        for method in prediction_roots:
            probability, provenance = _load_prediction(
                expected_paths[method][identity],
                expected_method=method,
                expected_record=record,
                oracle=oracle,
            )
            values[method][clip_index] = _clip_metrics(
                probability,
                target,
                visibility,
                threshold=threshold,
                ap_bins=ap_bins,
            )
            prediction_provenance[method].append(provenance)

    run_provenance: dict[str, Any] = {}
    authentication_manifest_provenance = {
        "artifact_count": manifest_provenance["clips"],
        "chunk_ids": sorted({record["chunk_id"] for record in records}),
        "dataset_fingerprint": manifest_provenance["dataset_fingerprint"],
    }
    for method, entries in prediction_provenance.items():
        run_fingerprints = {entry["prediction_run_fingerprint"] for entry in entries}
        run_json_records = {entry["prediction_run_json"] for entry in entries}
        checkpoints = {entry["checkpoint_sha256"] for entry in entries}
        if len(run_json_records) != 1:
            raise ValueError(f"{method} predictions do not share one canonical run payload")
        if len(run_fingerprints) != 1:
            raise ValueError(f"{method} predictions do not share one run fingerprint")
        if len(checkpoints) != 1:
            raise ValueError(f"{method} predictions do not share one checkpoint identity")
        checkpoint = next(iter(checkpoints))
        if method == "learned" and checkpoint is None:
            raise AssertionError("Learned checkpoint validation was bypassed")
        authenticated_run = authenticate_prediction_run(
            entries[0]["_prediction_run"],
            expected_method=method,
            manifest=manifest_path,
            protocol=protocol if method == "learned" else None,
            selection_record=selection_record if method == "learned" else None,
            evaluation_audit=evaluation_audit if method == "learned" else None,
            manifest_provenance=authentication_manifest_provenance,
        )
        if authenticated_run["checkpoint_sha256"] != checkpoint:
            raise ValueError(f"{method} artifact checkpoint differs from authenticated run")
        public_entries = [
            {
                key: value
                for key, value in entry.items()
                if key not in {"_prediction_run", "prediction_run_json"}
            }
            for entry in entries
        ]
        run_provenance[method] = {
            "directory": str(prediction_roots[method]),
            "prediction_run_fingerprint": next(iter(run_fingerprints)),
            "checkpoint_sha256": checkpoint,
            "authenticated_run": authenticated_run,
            "prediction_set_fingerprint": canonical_fingerprint(
                [
                    {
                        "clip_id": entry["clip_id"],
                        "t0_us": entry["t0_us"],
                        "sha256": entry["sha256"],
                    }
                    for entry in public_entries
                ]
            ),
            "artifacts": public_entries,
        }

    chunks = [record["chunk_id"] for record in records]
    paired = _paired_summary(
        values["learned"],
        values["persistence"],
        horizons_s,
        chunks,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    method_summaries = {
        "learned": _method_summary(values["learned"], horizons_s),
        "persistence": _method_summary(values["persistence"], horizons_s),
    }
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "comparison": "learned_minus_persistence",
        "protocol": frozen_protocol.provenance(),
        "configuration": {
            "threshold": threshold,
            "average_precision": {
                "estimator": "fixed_histogram_step_integral",
                "bins": ap_bins,
                "trapezoidal_pr_auc": False,
            },
            "short_horizons_s": list(SHORT_HORIZONS_S),
            "bootstrap": {
                "method": "paired_source_chunk_cluster_percentile",
                "confidence_level": 0.95,
                "replicates": bootstrap_replicates,
                "seed": bootstrap_seed,
                "resampling_unit": "source_chunk",
                "cell_level_resampling": False,
            },
            "uncertainty_scope": {
                "held_out_source_chunk_sampling": True,
                "training_seed_variability_included": False,
                "statement": (
                    "Intervals cover held-out source-chunk sampling only. They do not "
                    "cover optimization-seed variability after checkpoint selection."
                ),
            },
        },
        "estimand": (
            "Each metric is computed for each visible clip-horizon. Selected horizons are "
            "averaged equally within each clip, and clip values are then averaged equally. "
            "IoU is undefined when its union is empty. Average precision is undefined when "
            "a clip-horizon has no positive target. Undefined values do not contribute. "
            "Differences pair learned and persistence values within clips. Percentile "
            "intervals resample source chunks and carry every clip in each sampled chunk. "
            "No interval treats BEV cells as independent observations."
        ),
        "inputs": {
            "manifest": manifest_provenance,
            "predictions": run_provenance,
        },
        "metrics": method_summaries,
        "paired_differences": paired,
        "acceptance_gates": _forecast_gate_decisions(
            frozen_protocol, method_summaries, paired
        ),
    }


def main() -> None:
    args = build_parser().parse_args()
    result = compare_forecasts(
        protocol=args.protocol,
        manifest=args.manifest,
        selection_record=args.selection_record,
        evaluation_audit=args.evaluation_audit,
        learned_prediction_dir=args.learned_prediction_dir,
        persistence_prediction_dir=args.persistence_prediction_dir,
        threshold=args.threshold,
        ap_bins=args.ap_bins,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    write_json(result, args.output)
    print(
        f"Compared {result['inputs']['manifest']['clips']} clips across "
        f"{result['inputs']['manifest']['chunks']} source chunks; wrote {args.output}"
    )


if __name__ == "__main__":
    main()
