"""Train the compact temporal BEV occupancy model."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .data import OccupancyDataset, dataset_provenance, reject_chunk_overlap
from .losses import occupancy_loss
from .metrics import OccupancyMetricAccumulator
from .model import ModelConfig, TemporalOccupancyNet
from .protocol import (
    load_frozen_protocol,
    normalize_training_configuration,
    validate_manifest_binding,
    validate_training_run_configuration,
)
from .runtime import (
    atomic_torch_save,
    canonical_fingerprint,
    capture_rng_state,
    restore_rng_state,
    seed_everything,
    select_device,
    sha256_file,
    write_json,
)


CHECKPOINT_SCHEMA_VERSION = 2


def make_loader(
    dataset: OccupancyDataset,
    batch_size: int,
    workers: int,
    *,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader[dict[str, Any]]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        generator=generator,
    )


def run_epoch(
    model: TemporalOccupancyNet,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    amp: bool,
    dice_weight: float,
    max_positive_weight: float,
    gradient_clip: float,
) -> tuple[dict[str, float], dict[str, object] | None]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "bce": 0.0, "dice": 0.0}
    examples = 0
    metric_accumulator: OccupancyMetricAccumulator | None = None
    for batch in loader:
        inputs = batch["inputs"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        visibility = batch["visibility"].to(device, non_blocking=True)
        if metric_accumulator is None and not training:
            metric_accumulator = OccupancyMetricAccumulator(target.shape[1])
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            logits = model(inputs)
            loss, components = occupancy_loss(
                logits,
                target,
                visibility,
                dice_weight=dice_weight,
                max_positive_weight=max_positive_weight,
            )
        if training:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            if gradient_clip > 0:
                clip_grad_norm_(model.parameters(), gradient_clip)
            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        batch_size = inputs.shape[0]
        examples += batch_size
        for name in totals:
            totals[name] += components[name] * batch_size
        if metric_accumulator is not None:
            metric_accumulator.update(logits.sigmoid(), target, visibility)
    if examples == 0:
        raise RuntimeError("DataLoader produced no examples")
    losses = {name: value / examples for name, value in totals.items()}
    metrics = None
    if metric_accumulator is not None:
        metrics = metric_accumulator.compute(loader.dataset[0]["horizons_s"].numpy())
    return losses, metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--dice-weight", type=float, default=0.25)
    parser.add_argument("--max-positive-weight", type=float, default=30.0)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", help="Checkpoint path, or 'auto' for output-dir/latest.pt")
    parser.add_argument("--no-visibility-channel", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("epochs and batch-size must be positive; workers cannot be negative")
    seed_everything(args.seed)
    device = select_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    include_visibility = not args.no_visibility_channel
    train_dataset = OccupancyDataset(
        args.train_manifest, include_visibility_channel=include_visibility
    )
    val_dataset = OccupancyDataset(args.val_manifest, include_visibility_channel=include_visibility)
    train_provenance = dataset_provenance(train_dataset.rows)
    val_provenance = dataset_provenance(val_dataset.rows)
    reject_chunk_overlap(
        train_provenance,
        val_provenance,
        left_name="Training data",
        right_name="validation data",
    )
    data_provenance = {"train": train_provenance, "val": val_provenance}
    frozen_protocol = load_frozen_protocol(args.protocol)
    for name, manifest, provenance in (
        ("train", args.train_manifest, train_provenance),
        ("val", args.val_manifest, val_provenance),
    ):
        validate_manifest_binding(
            frozen_protocol,
            name,
            manifest,
            actual_sha256=sha256_file(manifest),
            clips=int(provenance["artifact_count"]),
            chunks=len(provenance["chunk_ids"]),
            dataset_fingerprint=str(provenance["dataset_fingerprint"]),
        )
    training_configuration = normalize_training_configuration(
        {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "base_channels": args.base_channels,
            "automatic_mixed_precision": args.amp,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        label="Requested training configuration",
    )
    validate_training_run_configuration(
        frozen_protocol,
        seed=args.seed,
        configuration=training_configuration,
    )
    example = train_dataset[0]
    val_example = val_dataset[0]
    if (
        example["inputs"].shape != val_example["inputs"].shape
        or example["target"].shape != val_example["target"].shape
        or not torch.equal(example["horizons_s"], val_example["horizons_s"])
        or not torch.equal(example["past_horizons_s"], val_example["past_horizons_s"])
        or not torch.equal(example["grid_origin_xy_m"], val_example["grid_origin_xy_m"])
        or example["resolution_m"] != val_example["resolution_m"]
        or example["coordinate_frame"] != val_example["coordinate_frame"]
    ):
        raise ValueError("Train and validation occupancy schemas do not match")
    model_config = ModelConfig(
        in_channels=int(example["inputs"].shape[1]),
        base_channels=args.base_channels,
        num_horizons=int(example["target"].shape[0]),
    )
    data_schema = {
        "input_shape": list(example["inputs"].shape),
        "target_shape": list(example["target"].shape),
        "past_horizons_s": example["past_horizons_s"].tolist(),
        "horizons_s": example["horizons_s"].tolist(),
        "grid_origin_xy_m": example["grid_origin_xy_m"].tolist(),
        "resolution_m": float(example["resolution_m"]),
        "coordinate_frame": example["coordinate_frame"],
    }
    model = TemporalOccupancyNet(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    val_loader = make_loader(
        val_dataset, args.batch_size, args.workers, shuffle=False, seed=args.seed, device=device
    )

    resume_signature = {
        "protocol": frozen_protocol.provenance(),
        "train_manifest": str(Path(args.train_manifest).resolve()),
        "train_manifest_sha256": sha256_file(args.train_manifest),
        "train_dataset_fingerprint": train_provenance["dataset_fingerprint"],
        "val_manifest": str(Path(args.val_manifest).resolve()),
        "val_manifest_sha256": sha256_file(args.val_manifest),
        "val_dataset_fingerprint": val_provenance["dataset_fingerprint"],
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "base_channels": args.base_channels,
        "dice_weight": args.dice_weight,
        "max_positive_weight": args.max_positive_weight,
        "gradient_clip": args.gradient_clip,
        "visibility_channel": include_visibility,
        "amp": args.amp,
        "seed": args.seed,
    }
    start_epoch, best_val, history = 0, float("inf"), []
    if args.resume:
        checkpoint_path = output_dir / "latest.pt" if args.resume == "auto" else Path(args.resume)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("Unsupported or missing checkpoint schema version")
        if checkpoint["model_config"] != model_config.to_dict():
            raise ValueError("Checkpoint model configuration does not match this dataset/run")
        if checkpoint.get("data_schema") != data_schema:
            raise ValueError("Checkpoint occupancy geometry does not match this dataset/run")
        if checkpoint.get("data_provenance") != data_provenance:
            raise ValueError("Checkpoint source artifact provenance does not match this run")
        if checkpoint.get("resume_signature") != resume_signature:
            raise ValueError(
                "Checkpoint data or optimization configuration differs from the requested run"
            )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scaler_state" not in checkpoint:
            raise ValueError("Checkpoint has no GradScaler state and cannot be resumed")
        scaler.load_state_dict(checkpoint["scaler_state"])
        if "rng_state" not in checkpoint:
            raise ValueError("Checkpoint has no RNG state and cannot be reproducibly resumed")
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint["best_val_loss"])
        history = list(checkpoint.get("history", []))

    run_config = {
        **vars(args),
        "model": model_config.to_dict(),
        "data_schema": data_schema,
        "device_resolved": str(device),
    }
    write_json(run_config, output_dir / "config.json")
    for epoch in range(start_epoch, args.epochs):
        started = time.monotonic()
        # Epoch-specific sampling preserves the exact order across interrupted runs.
        train_loader = make_loader(
            train_dataset,
            args.batch_size,
            args.workers,
            shuffle=True,
            seed=args.seed + epoch,
            device=device,
        )
        train_losses, _ = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            scaler=scaler,
            amp=args.amp,
            dice_weight=args.dice_weight,
            max_positive_weight=args.max_positive_weight,
            gradient_clip=args.gradient_clip,
        )
        with torch.no_grad():
            val_losses, val_metrics = run_epoch(
                model,
                val_loader,
                device,
                optimizer=None,
                scaler=None,
                amp=args.amp,
                dice_weight=args.dice_weight,
                max_positive_weight=args.max_positive_weight,
                gradient_clip=args.gradient_clip,
            )
        epoch_record = {
            "epoch": epoch,
            "elapsed_s": time.monotonic() - started,
            "train": train_losses,
            "val": val_losses,
            "val_metrics": val_metrics,
        }
        history.append(epoch_record)
        is_best = val_losses["loss"] < best_val
        best_val = min(best_val, val_losses["loss"])
        checkpoint = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "epoch": epoch,
            "best_val_loss": best_val,
            "model_config": model_config.to_dict(),
            "data_schema": data_schema,
            "data_provenance": data_provenance,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "rng_state": capture_rng_state(),
            "run_config": run_config,
            "seed": args.seed,
            "resume_signature": resume_signature,
            "history": history,
        }
        atomic_torch_save(checkpoint, output_dir / "latest.pt")
        if is_best:
            atomic_torch_save(checkpoint, output_dir / "best.pt")
        write_json(history, output_dir / "history.json")
        print(
            f"epoch={epoch} train_loss={train_losses['loss']:.5f} "
            f"val_loss={val_losses['loss']:.5f} best={best_val:.5f}"
        )

    completed_epochs = [int(record["epoch"]) for record in history]
    if completed_epochs != list(range(args.epochs)):
        raise ValueError(
            "Training history does not prove completion of every protocol epoch"
        )
    latest_path = (output_dir / "latest.pt").resolve()
    best_path = (output_dir / "best.pt").resolve()
    if not latest_path.is_file() or not best_path.is_file():
        raise FileNotFoundError("Completed training has no latest or best checkpoint")
    completion = {
        "schema_version": 1,
        "status": "completed",
        "seed": args.seed,
        "epochs": args.epochs,
        "final_epoch": args.epochs - 1,
        "protocol": frozen_protocol.provenance(),
        "latest_checkpoint": str(latest_path),
        "latest_checkpoint_sha256": sha256_file(latest_path),
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": sha256_file(best_path),
    }
    completion["completion_fingerprint"] = canonical_fingerprint(completion)
    write_json(completion, output_dir / "completion.json")


if __name__ == "__main__":
    main()
