#!/usr/bin/env python3
"""Train and evaluate a small occupancy forecast on generated motion data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from src.world_model.losses import occupancy_loss
from src.world_model.metrics import OccupancyMetricAccumulator
from src.world_model.model import ModelConfig, TemporalOccupancyNet


SEED = 7
GRID_SIZE = 24
EXAMPLES = 128
TRAIN_EXAMPLES = 96
TRAIN_STEPS = 100
HISTORY_TIMES = (-2, -1, 0)
FUTURE_TIMES = (1, 2, 3)


def _square(grid: torch.Tensor, center_x: int, center_y: int, radius: int = 3) -> None:
    y_start = max(0, center_y - radius)
    y_stop = min(grid.shape[-2], center_y + radius + 1)
    x_start = max(0, center_x - radius)
    x_stop = min(grid.shape[-1], center_x + radius + 1)
    grid[..., y_start:y_stop, x_start:x_stop] = 1.0


def make_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Make deterministic squares that move with constant velocity."""

    inputs = torch.zeros(EXAMPLES, 3, 2, GRID_SIZE, GRID_SIZE)
    targets = torch.zeros(EXAMPLES, 3, GRID_SIZE, GRID_SIZE)
    observed = torch.ones_like(targets)
    generator = np.random.default_rng(SEED)

    for example in range(EXAMPLES):
        velocity_x = int(generator.choice((-2, -1, 1, 2)))
        velocity_y = int(generator.choice((-1, 0, 1)))
        start_x = int(generator.integers(9, 15))
        start_y = int(generator.integers(8, 16))

        for index, time in enumerate(HISTORY_TIMES):
            _square(
                inputs[example, index, 0],
                start_x + velocity_x * time,
                start_y + velocity_y * time,
            )
            inputs[example, index, 1] = 1.0

        for index, time in enumerate(FUTURE_TIMES):
            _square(
                targets[example, index],
                start_x + velocity_x * time,
                start_y + velocity_y * time,
            )

    return inputs, targets, observed


def _metrics(
    probability: torch.Tensor, target: torch.Tensor, observed: torch.Tensor
) -> dict[str, float]:
    accumulator = OccupancyMetricAccumulator(num_horizons=3, bins=100)
    accumulator.update(probability, target, observed)
    result = accumulator.compute(FUTURE_TIMES)["mean"]
    return {"brier": float(result["brier"]), "iou": float(result["iou"])}


def _font(
    size: int, *, bold: bool = False
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size=size)
    except OSError:
        return ImageFont.load_default()


def _binary_image(grid: np.ndarray) -> Image.Image:
    colors = np.full((*grid.shape, 3), 247, dtype=np.uint8)
    colors[grid >= 0.5] = (15, 76, 129)
    return Image.fromarray(colors).resize((216, 216), Image.Resampling.NEAREST)


def _probability_image(grid: np.ndarray) -> Image.Image:
    probability = np.clip(grid.astype(np.float32), 0.0, 1.0)[..., None]
    empty = np.asarray([247, 250, 252], dtype=np.float32)
    occupied = np.asarray([16, 185, 129], dtype=np.float32)
    colors = empty + probability * (occupied - empty)
    return Image.fromarray(colors.astype(np.uint8)).resize(
        (216, 216), Image.Resampling.NEAREST
    )


def render_example(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    learned: torch.Tensor,
    persistence: torch.Tensor,
    output: Path,
) -> None:
    panels = (
        ("Past -2", _binary_image(inputs[0, 0, 0].numpy())),
        ("Past -1", _binary_image(inputs[0, 1, 0].numpy())),
        ("Current", _binary_image(inputs[0, 2, 0].numpy())),
        ("Recorded +3", _binary_image(targets[0, 2].numpy())),
        ("Learned +3", _probability_image(learned[0, 2].numpy())),
        ("Copy-current +3", _binary_image(persistence[0, 2].numpy())),
    )
    canvas = Image.new("RGB", (780, 650), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (32, 24),
        "Small occupancy-forecast demo",
        font=_font(30, bold=True),
        fill=(15, 23, 42),
    )
    draw.text(
        (32, 64),
        "The square moves across three past frames. The model predicts its future position.",
        font=_font(17),
        fill=(71, 85, 105),
    )
    for index, (label, panel) in enumerate(panels):
        row, column = divmod(index, 3)
        x = 32 + column * 250
        y = 112 + row * 260
        canvas.paste(panel, (x, y + 30))
        draw.rectangle((x, y + 30, x + 216, y + 246), outline=(148, 163, 184), width=2)
        draw.text((x, y), label, font=_font(18, bold=True), fill=(30, 41, 59))
    draw.text(
        (32, 625),
        "Blue: occupied cell   Green: predicted occupancy probability",
        font=_font(16),
        fill=(71, 85, 105),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def main() -> None:
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    inputs, targets, observed = make_data()

    model = TemporalOccupancyNet(
        ModelConfig(in_channels=2, base_channels=4, num_horizons=3)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model.train()
    for _ in range(TRAIN_STEPS):
        indices = torch.randint(0, TRAIN_EXAMPLES, (24,))
        logits = model(inputs[indices])
        loss, _ = occupancy_loss(logits, targets[indices], observed[indices])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_inputs = inputs[TRAIN_EXAMPLES:]
        test_targets = targets[TRAIN_EXAMPLES:]
        test_observed = observed[TRAIN_EXAMPLES:]
        learned = model(test_inputs).sigmoid()
        persistence = test_inputs[:, -1, 0].unsqueeze(1).expand(-1, 3, -1, -1)
        learned_metrics = _metrics(learned, test_targets, test_observed)
        persistence_metrics = _metrics(persistence, test_targets, test_observed)

    if not (
        learned_metrics["brier"] < persistence_metrics["brier"]
        and learned_metrics["iou"] > persistence_metrics["iou"]
    ):
        raise RuntimeError(
            "The learned forecast did not beat the copy-current baseline"
        )

    output_directory = Path("results/quick_demo")
    figure_path = output_directory / "forecast.png"
    report_path = output_directory / "metrics.json"
    render_example(
        test_inputs,
        test_targets,
        learned,
        persistence,
        figure_path,
    )
    report_path.write_text(
        json.dumps(
            {
                "data": "deterministic generated constant-velocity squares",
                "learned": learned_metrics,
                "copy_current": persistence_metrics,
                "seed": SEED,
                "train_examples": TRAIN_EXAMPLES,
                "test_examples": EXAMPLES - TRAIN_EXAMPLES,
                "train_steps": TRAIN_STEPS,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("Quick demo complete")
    print(
        f"Brier: learned {learned_metrics['brier']:.4f}, "
        f"copy-current {persistence_metrics['brier']:.4f} (lower is better)"
    )
    print(
        f"IoU:   learned {learned_metrics['iou']:.4f}, "
        f"copy-current {persistence_metrics['iou']:.4f} (higher is better)"
    )
    print(f"Figure: {figure_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
