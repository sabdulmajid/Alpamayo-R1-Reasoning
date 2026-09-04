"""Evaluate learned and persistence occupancy forecasts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .baselines import persistence
from .data import OccupancyDataset, dataset_provenance, reject_chunk_overlap
from .metrics import OccupancyMetricAccumulator
from .model import ModelConfig, TemporalOccupancyNet
from .runtime import (
    atomic_save_npz,
    canonical_fingerprint,
    prediction_filename,
    seed_everything,
    select_device,
    sha256_file,
    write_json,
)


PREDICTION_SCHEMA_VERSION = 2
CHECKPOINT_SCHEMA_VERSION = 2
SELECTION_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--method", choices=("learned", "persistence"), required=True)
    parser.add_argument("--checkpoint", help="Required for method=learned")
    parser.add_argument(
        "--selection-record",
        help="Checkpoint-selection JSON required for learned test evaluation",
    )
    parser.add_argument(
        "--evaluation-audit",
        help="Evaluation-partition audit JSON required for learned test evaluation",
    )
    parser.add_argument("--output", required=True, help="Metrics JSON path")
    parser.add_argument(
        "--data-role",
        choices=("test", "validation"),
        default="test",
        help="Test rejects train and validation overlap; validation must match the checkpoint validation set",
    )
    parser.add_argument(
        "--prediction-dir", help="Optional per-clip probability NPZ directory"
    )
    parser.add_argument(
        "--overwrite-predictions",
        action="store_true",
        help="Replace existing prediction files instead of requiring identical provenance",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-visibility-channel", action="store_true")
    return parser


def _read_json_record(path: str | Path, label: str) -> tuple[Path, dict[str, Any], str]:
    """Read one JSON record and hash the exact bytes that were parsed."""

    record_path = Path(path).expanduser().resolve()
    try:
        content = record_path.read_bytes()
        payload = json.loads(content)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid {label}: {record_path}") from error
    if not isinstance(payload, dict):
        raise ValueError(
            f"{label.capitalize()} must contain a JSON object: {record_path}"
        )
    return record_path, payload, hashlib.sha256(content).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value)
    if not _SHA256_PATTERN.fullmatch(digest):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return digest


def _require_exact_path(value: Any, expected: Path, label: str) -> None:
    try:
        recorded = Path(str(value)).expanduser().resolve()
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"{label} path is invalid") from error
    if recorded != expected:
        raise ValueError(
            f"{label} path {recorded} does not match requested path {expected}"
        )


def _selected_run_record(selection: dict[str, Any]) -> dict[str, Any]:
    runs = selection.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("Checkpoint selection record has no compared training runs")
    selected_label = selection.get("selected_label")
    matches = [
        run
        for run in runs
        if isinstance(run, dict) and run.get("label") == selected_label
    ]
    if len(matches) != 1:
        raise ValueError(
            "Checkpoint selection record does not identify one selected training run"
        )
    return matches[0]


def _validate_selection_record(
    selection_path: Path,
    selection: dict[str, Any],
    selection_sha256: str,
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    """Bind a checkpoint to its validation-only selection record."""

    if selection.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise ValueError("Checkpoint selection record has an unsupported schema")
    if selection.get("selection_policy") != "minimum_best_validation_loss":
        raise ValueError("Checkpoint selection record has an unsupported policy")
    if selection.get("test_metrics_used") is not False:
        raise ValueError("Checkpoint selection record does not exclude test metrics")

    try:
        selection_fingerprint = _require_sha256(
            selection["selection_fingerprint"], "Checkpoint selection fingerprint"
        )
    except KeyError as error:
        raise ValueError(
            "Checkpoint selection record is missing 'selection_fingerprint'"
        ) from error
    fingerprint_payload = dict(selection)
    fingerprint_payload.pop("selection_fingerprint", None)
    if canonical_fingerprint(fingerprint_payload) != selection_fingerprint:
        raise ValueError("Checkpoint selection fingerprint does not match its payload")

    try:
        _require_exact_path(
            selection["selected_checkpoint"],
            checkpoint_path,
            "Selected checkpoint",
        )
        selected_sha256 = _require_sha256(
            selection["selected_checkpoint_sha256"],
            "Selected checkpoint SHA-256",
        )
        selected_epoch = int(selection["selected_epoch"])
        selected_seed = int(selection["selected_seed"])
        selected_validation_loss = float(selection["selected_validation_loss"])
    except KeyError as error:
        raise ValueError(
            f"Checkpoint selection record is missing {error.args[0]!r}"
        ) from error
    if selected_sha256 != checkpoint_sha256:
        raise ValueError("Selected checkpoint SHA-256 does not match checkpoint bytes")
    try:
        checkpoint_epoch = int(checkpoint["epoch"])
        checkpoint_seed = int(checkpoint["seed"])
        checkpoint_validation_loss = float(checkpoint["best_val_loss"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Checkpoint has incomplete selection metadata") from error
    if selected_epoch != checkpoint_epoch or selected_seed != checkpoint_seed:
        raise ValueError("Selected checkpoint epoch or seed differs from checkpoint")
    if not np.isfinite(selected_validation_loss) or not np.isclose(
        selected_validation_loss,
        checkpoint_validation_loss,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Selected validation loss differs from checkpoint")

    selected_run = _selected_run_record(selection)
    selected_label = str(selection["selected_label"])
    try:
        _require_exact_path(
            selected_run["checkpoint"], checkpoint_path, "Selected run checkpoint"
        )
        run_sha256 = _require_sha256(
            selected_run["checkpoint_sha256"], "Selected run checkpoint SHA-256"
        )
        run_epoch = int(selected_run["epoch"])
        run_seed = int(selected_run["seed"])
        run_validation_loss = float(selected_run["best_validation_loss"])
    except KeyError as error:
        raise ValueError(f"Selected run record is missing {error.args[0]!r}") from error
    if (
        run_sha256 != checkpoint_sha256
        or run_epoch != selected_epoch
        or run_seed != selected_seed
        or not np.isclose(
            run_validation_loss,
            selected_validation_loss,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise ValueError("Selected run record does not bind to the selected checkpoint")

    provenance = checkpoint.get("data_provenance")
    signature = checkpoint.get("resume_signature")
    if not isinstance(provenance, dict) or not isinstance(signature, dict):
        raise ValueError("Checkpoint has incomplete training provenance")
    try:
        bindings = {
            "train_manifest_sha256": signature["train_manifest_sha256"],
            "validation_manifest_sha256": signature["val_manifest_sha256"],
            "train_dataset_fingerprint": provenance["train"]["dataset_fingerprint"],
            "validation_dataset_fingerprint": provenance["val"]["dataset_fingerprint"],
            "model_config": checkpoint["model_config"],
            "data_schema": checkpoint["data_schema"],
        }
    except (KeyError, TypeError) as error:
        raise ValueError("Checkpoint has incomplete training provenance") from error
    for field, expected in bindings.items():
        if selected_run.get(field) != expected:
            raise ValueError(
                f"Selected run {field} does not match checkpoint provenance"
            )

    return {
        "path": str(selection_path),
        "sha256": selection_sha256,
        "selection_fingerprint": selection_fingerprint,
        "selected_label": selected_label,
        "selected_seed": selected_seed,
        "selected_epoch": selected_epoch,
        "selected_validation_loss": selected_validation_loss,
    }


def validate_learned_test_binding(
    *,
    selection_record: str | Path,
    evaluation_audit: str | Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    checkpoint: dict[str, Any],
    manifest_path: Path,
    manifest_sha256: str,
    evaluation_provenance: dict[str, Any],
) -> dict[str, Any]:
    """Validate immutable selection and held-out partition records."""

    selection_path, selection, selection_sha256 = _read_json_record(
        selection_record, "checkpoint selection record"
    )
    selection_metadata = _validate_selection_record(
        selection_path,
        selection,
        selection_sha256,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint=checkpoint,
    )
    audit_path, audit, audit_sha256 = _read_json_record(
        evaluation_audit, "evaluation partition audit"
    )
    try:
        _require_exact_path(
            audit["selection"], selection_path, "Audit selection record"
        )
        _require_exact_path(audit["checkpoint"], checkpoint_path, "Audit checkpoint")
        _require_exact_path(
            audit["test_manifest"], manifest_path, "Audit test manifest"
        )
        audit_selection_sha256 = _require_sha256(
            audit["selection_sha256"], "Audit selection SHA-256"
        )
        audit_checkpoint_sha256 = _require_sha256(
            audit["checkpoint_sha256"], "Audit checkpoint SHA-256"
        )
        audit_manifest_sha256 = _require_sha256(
            audit["test_manifest_sha256"], "Audit test manifest SHA-256"
        )
        test_clips = int(audit["test_clips"])
        test_chunks = int(audit["test_chunks"])
    except KeyError as error:
        raise ValueError(
            f"Evaluation partition audit is missing {error.args[0]!r}"
        ) from error
    if audit_selection_sha256 != selection_sha256:
        raise ValueError("Evaluation audit does not bind the selection record bytes")
    if audit_checkpoint_sha256 != checkpoint_sha256:
        raise ValueError("Evaluation audit does not bind the checkpoint bytes")
    if audit_manifest_sha256 != manifest_sha256:
        raise ValueError("Evaluation audit does not bind the test manifest bytes")
    if test_clips != int(evaluation_provenance["artifact_count"]) or test_chunks != len(
        evaluation_provenance["chunk_ids"]
    ):
        raise ValueError("Evaluation audit test counts differ from the test manifest")
    for key in ("train_test_chunk_overlap", "validation_test_chunk_overlap"):
        value = audit.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value != 0:
            raise ValueError(f"Evaluation audit requires {key}=0")
    try:
        audit_selection_fingerprint = _require_sha256(
            audit["selection_fingerprint"], "Audit selection fingerprint"
        )
    except KeyError as error:
        raise ValueError(
            "Evaluation partition audit is missing 'selection_fingerprint'"
        ) from error
    if audit_selection_fingerprint != selection_metadata["selection_fingerprint"]:
        raise ValueError("Evaluation audit selection fingerprint does not match")

    return {
        "checkpoint_selection": selection_metadata,
        "evaluation_audit": {
            "path": str(audit_path),
            "sha256": audit_sha256,
            "record_fingerprint": canonical_fingerprint(audit),
            "selection_fingerprint": audit_selection_fingerprint,
            "train_test_chunk_overlap": 0,
            "validation_test_chunk_overlap": 0,
        },
    }


def _save_predictions(
    directory: Path,
    batch: dict[str, Any],
    probabilities: torch.Tensor,
    *,
    method: str,
    checkpoint_sha256: str,
    run_metadata: dict[str, Any],
    run_fingerprint: str,
    overwrite: bool,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    values = probabilities.detach().float().cpu().numpy()
    for index, clip_id in enumerate(batch["clip_id"]):
        t0_us = int(batch["t0_us"][index])
        source_sha256 = sha256_file(batch["artifact_path"][index])
        output = directory / prediction_filename(str(clip_id), t0_us)
        arrays = {
            "schema_version": np.asarray(PREDICTION_SCHEMA_VERSION, dtype=np.int16),
            "clip_id": np.asarray(str(clip_id)),
            "t0_us": np.asarray(t0_us, dtype=np.int64),
            "occupancy_prob": values[index].astype(np.float16),
            "horizons_s": batch["horizons_s"][index].numpy().astype(np.float32),
            "resolution_m": np.asarray(
                float(batch["resolution_m"][index]), dtype=np.float32
            ),
            "grid_origin_xy_m": batch["grid_origin_xy_m"][index]
            .numpy()
            .astype(np.float32),
            "coordinate_frame": np.asarray(batch["coordinate_frame"][index]),
            "source_artifact_sha256": np.asarray(source_sha256),
            "producer_method": np.asarray(method),
            "checkpoint_sha256": np.asarray(checkpoint_sha256),
            "prediction_run_json": np.asarray(
                json.dumps(
                    run_metadata,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            ),
            "prediction_run_fingerprint": np.asarray(run_fingerprint),
        }
        if output.exists() and not overwrite:
            _validate_existing_prediction(output, arrays)
            continue
        atomic_save_npz(
            output,
            **arrays,
        )


def _validate_existing_prediction(path: Path, expected: dict[str, np.ndarray]) -> None:
    """Accept a resumable output only when identity, geometry, and provenance match."""

    metadata_keys = set(expected) - {"occupancy_prob"}
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(expected) - set(archive.files))
        if missing:
            raise ValueError(f"Existing prediction {path} is missing {missing}")
        actual_probability = np.asarray(archive["occupancy_prob"])
        expected_probability = expected["occupancy_prob"]
        if (
            actual_probability.shape != expected_probability.shape
            or not np.isfinite(actual_probability).all()
            or actual_probability.min(initial=0.0) < 0
            or actual_probability.max(initial=1.0) > 1
        ):
            raise ValueError(f"Existing prediction {path} has invalid probabilities")
        if not np.array_equal(actual_probability, expected_probability):
            raise ValueError(
                f"Existing prediction {path} has different occupancy_prob values; "
                "use --overwrite-predictions to replace it"
            )
        for key in metadata_keys:
            if not np.array_equal(np.asarray(archive[key]), expected[key]):
                raise ValueError(
                    f"Existing prediction {path} has different {key}; "
                    "use --overwrite-predictions to replace it"
                )


def main() -> None:
    args = build_parser().parse_args()
    if args.method == "learned" and not args.checkpoint:
        raise ValueError("--checkpoint is required for method=learned")
    if args.method == "learned" and args.data_role == "test":
        missing_records = [
            flag
            for flag, value in (
                ("--selection-record", args.selection_record),
                ("--evaluation-audit", args.evaluation_audit),
            )
            if not value
        ]
        if missing_records:
            raise ValueError(
                "Learned test evaluation requires " + " and ".join(missing_records)
            )
    elif args.selection_record or args.evaluation_audit:
        raise ValueError(
            "--selection-record and --evaluation-audit are only valid for learned test evaluation"
        )
    if args.method == "persistence" and args.checkpoint:
        raise ValueError("--checkpoint is only valid for method=learned")
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("batch-size must be positive and workers cannot be negative")
    seed_everything(args.seed)
    device = select_device(args.device)
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest_sha256 = sha256_file(manifest_path)
    dataset = OccupancyDataset(
        manifest_path, include_visibility_channel=not args.no_visibility_channel
    )
    evaluation_provenance = dataset_provenance(dataset.rows)
    if sha256_file(manifest_path) != manifest_sha256:
        raise ValueError(
            "Evaluation manifest changed while its provenance was computed"
        )
    loader: DataLoader[dict[str, Any]] = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    model: TemporalOccupancyNet | None = None
    checkpoint_epoch: int | None = None
    checkpoint_schema: dict[str, Any] | None = None
    checkpoint_sha256 = ""
    checkpoint_path: Path | None = None
    checkpoint: dict[str, Any] | None = None
    record_binding: dict[str, Any] = {
        "checkpoint_selection": None,
        "evaluation_audit": None,
    }
    partition_isolation: dict[str, int | None] = {
        "train_evaluation_chunk_overlap": None,
        "validation_evaluation_chunk_overlap": None,
    }
    if args.method == "learned":
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        checkpoint_sha256 = sha256_file(checkpoint_path)
        loaded_checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if not isinstance(loaded_checkpoint, dict):
            raise ValueError("Checkpoint payload must be a dictionary")
        checkpoint = loaded_checkpoint
        if sha256_file(checkpoint_path) != checkpoint_sha256:
            raise ValueError("Checkpoint changed while it was loaded")
        if checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("Unsupported or missing checkpoint schema version")
        checkpoint_provenance = checkpoint.get("data_provenance")
        if (
            not isinstance(checkpoint_provenance, dict)
            or "train" not in checkpoint_provenance
        ):
            raise ValueError("Checkpoint has no training-data provenance")
        reject_chunk_overlap(
            checkpoint_provenance["train"],
            evaluation_provenance,
            left_name="Checkpoint training data",
            right_name="evaluation data",
        )
        partition_isolation["train_evaluation_chunk_overlap"] = 0
        if "val" not in checkpoint_provenance:
            raise ValueError("Checkpoint has no validation-data provenance")
        if args.data_role == "test":
            reject_chunk_overlap(
                checkpoint_provenance["val"],
                evaluation_provenance,
                left_name="Checkpoint validation data",
                right_name="evaluation data",
            )
            partition_isolation["validation_evaluation_chunk_overlap"] = 0
            assert args.selection_record is not None
            assert args.evaluation_audit is not None
            record_binding = validate_learned_test_binding(
                selection_record=args.selection_record,
                evaluation_audit=args.evaluation_audit,
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_sha256,
                checkpoint=checkpoint,
                manifest_path=manifest_path,
                manifest_sha256=manifest_sha256,
                evaluation_provenance=evaluation_provenance,
            )
        elif evaluation_provenance["dataset_fingerprint"] != checkpoint_provenance[
            "val"
        ].get("dataset_fingerprint"):
            raise ValueError(
                "Validation evaluation data does not match checkpoint validation data"
            )
        model = TemporalOccupancyNet(ModelConfig(**checkpoint["model_config"]))
        model.load_state_dict(checkpoint["model_state"])
        model.to(device).eval()
        checkpoint_epoch = int(checkpoint["epoch"])
        checkpoint_schema = checkpoint.get("data_schema")
        if checkpoint_schema is None:
            raise ValueError("Checkpoint has no occupancy geometry metadata")

    example = dataset[0]
    if model is not None:
        if model.config.in_channels != int(example["inputs"].shape[1]):
            raise ValueError(
                "Checkpoint input channels do not match the evaluation artifacts"
            )
        if model.config.num_horizons != int(example["target"].shape[0]):
            raise ValueError(
                "Checkpoint horizons do not match the evaluation artifacts"
            )
        evaluation_schema = {
            "input_shape": list(example["inputs"].shape),
            "target_shape": list(example["target"].shape),
            "past_horizons_s": example["past_horizons_s"].tolist(),
            "horizons_s": example["horizons_s"].tolist(),
            "grid_origin_xy_m": example["grid_origin_xy_m"].tolist(),
            "resolution_m": float(example["resolution_m"]),
            "coordinate_frame": example["coordinate_frame"],
        }
        if checkpoint_schema != evaluation_schema:
            raise ValueError(
                "Checkpoint geometry does not match the evaluation artifacts"
            )
    accumulator = OccupancyMetricAccumulator(
        int(example["target"].shape[0]), threshold=args.threshold
    )
    checkpoint_record: dict[str, Any] | None = None
    if checkpoint is not None:
        checkpoint_provenance = checkpoint["data_provenance"]
        checkpoint_signature = checkpoint.get("resume_signature", {})
        checkpoint_record = {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "epoch": checkpoint_epoch,
            "seed": int(checkpoint["seed"]),
            "train_manifest_sha256": checkpoint_signature.get("train_manifest_sha256"),
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
    evaluation_binding = {
        "evaluation_manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha256,
            "dataset_fingerprint": evaluation_provenance["dataset_fingerprint"],
            "clips": int(evaluation_provenance["artifact_count"]),
            "chunks": len(evaluation_provenance["chunk_ids"]),
        },
        "checkpoint": checkpoint_record,
        **record_binding,
        "partition_isolation": partition_isolation,
    }
    evaluation_binding["binding_fingerprint"] = canonical_fingerprint(
        evaluation_binding
    )
    prediction_run = {
        "method": args.method,
        "data_role": args.data_role,
        "checkpoint_sha256": checkpoint_sha256,
        "amp": bool(args.amp and device.type == "cuda"),
        "evaluation_binding": evaluation_binding,
        "data_schema": {
            "input_shape": list(example["inputs"].shape),
            "target_shape": list(example["target"].shape),
            "past_horizons_s": example["past_horizons_s"].tolist(),
            "horizons_s": example["horizons_s"].tolist(),
            "grid_origin_xy_m": example["grid_origin_xy_m"].tolist(),
            "resolution_m": float(example["resolution_m"]),
            "coordinate_frame": example["coordinate_frame"],
        },
    }
    prediction_run_fingerprint = canonical_fingerprint(prediction_run)
    prediction_dir = Path(args.prediction_dir) if args.prediction_dir else None
    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device, non_blocking=True)
            if args.method == "learned":
                assert model is not None
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=args.amp and device.type == "cuda",
                ):
                    probabilities = model(inputs).sigmoid()
            else:
                probabilities = persistence(inputs, batch["target"].shape[1])
            target = batch["target"].to(device, non_blocking=True)
            visibility = batch["visibility"].to(device, non_blocking=True)
            accumulator.update(probabilities, target, visibility)
            if prediction_dir is not None:
                _save_predictions(
                    prediction_dir,
                    batch,
                    probabilities,
                    method=args.method,
                    checkpoint_sha256=checkpoint_sha256,
                    run_metadata=prediction_run,
                    run_fingerprint=prediction_run_fingerprint,
                    overwrite=args.overwrite_predictions,
                )

    result = {
        "method": args.method,
        "data_role": args.data_role,
        "manifest": str(manifest_path),
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_sha256": checkpoint_sha256 or None,
        "manifest_sha256": manifest_sha256,
        "evaluation_binding": evaluation_binding,
        "prediction_run": prediction_run,
        "prediction_run_fingerprint": prediction_run_fingerprint,
        "evaluation_dataset_fingerprint": evaluation_provenance["dataset_fingerprint"],
        "clips": len(dataset),
        "metrics": accumulator.compute(example["horizons_s"].numpy()),
    }
    write_json(result, args.output)
    print(f"Wrote {args.method} metrics for {len(dataset)} clips to {args.output}")


if __name__ == "__main__":
    main()
