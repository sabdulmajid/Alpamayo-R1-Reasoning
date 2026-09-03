"""Evaluate learned and persistence occupancy forecasts."""

from __future__ import annotations

import argparse
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--method", choices=("learned", "persistence"), required=True
    )
    parser.add_argument("--checkpoint", help="Required for method=learned")
    parser.add_argument("--output", required=True, help="Metrics JSON path")
    parser.add_argument("--prediction-dir", help="Optional per-clip probability NPZ directory")
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


def _save_predictions(
    directory: Path,
    batch: dict[str, Any],
    probabilities: torch.Tensor,
    *,
    method: str,
    checkpoint_sha256: str,
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
            "prediction_run_fingerprint": np.asarray(run_fingerprint),
        }
        if output.exists() and not overwrite:
            _validate_existing_prediction(output, arrays)
            continue
        atomic_save_npz(
            output,
            **arrays,
        )


def _validate_existing_prediction(
    path: Path, expected: dict[str, np.ndarray]
) -> None:
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
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("batch-size must be positive and workers cannot be negative")
    seed_everything(args.seed)
    device = select_device(args.device)
    dataset = OccupancyDataset(
        args.manifest, include_visibility_channel=not args.no_visibility_channel
    )
    evaluation_provenance = dataset_provenance(dataset.rows)
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
    if args.method == "learned":
        checkpoint_sha256 = sha256_file(args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("Unsupported or missing checkpoint schema version")
        checkpoint_provenance = checkpoint.get("data_provenance")
        if not isinstance(checkpoint_provenance, dict) or "train" not in checkpoint_provenance:
            raise ValueError("Checkpoint has no training-data provenance")
        reject_chunk_overlap(
            checkpoint_provenance["train"],
            evaluation_provenance,
            left_name="Checkpoint training data",
            right_name="evaluation data",
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
            raise ValueError("Checkpoint input channels do not match the evaluation artifacts")
        if model.config.num_horizons != int(example["target"].shape[0]):
            raise ValueError("Checkpoint horizons do not match the evaluation artifacts")
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
            raise ValueError("Checkpoint geometry does not match the evaluation artifacts")
    accumulator = OccupancyMetricAccumulator(
        int(example["target"].shape[0]), threshold=args.threshold
    )
    prediction_run = {
        "method": args.method,
        "checkpoint_sha256": checkpoint_sha256,
        "amp": bool(args.amp and device.type == "cuda"),
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
                    run_fingerprint=prediction_run_fingerprint,
                    overwrite=args.overwrite_predictions,
                )

    result = {
        "method": args.method,
        "manifest": str(Path(args.manifest).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_sha256": checkpoint_sha256 or None,
        "manifest_sha256": sha256_file(args.manifest),
        "prediction_run_fingerprint": prediction_run_fingerprint,
        "evaluation_dataset_fingerprint": evaluation_provenance["dataset_fingerprint"],
        "clips": len(dataset),
        "metrics": accumulator.compute(example["horizons_s"].numpy()),
    }
    write_json(result, args.output)
    print(f"Wrote {args.method} metrics for {len(dataset)} clips to {args.output}")


if __name__ == "__main__":
    main()
