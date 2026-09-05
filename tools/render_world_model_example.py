#!/usr/bin/env python3
"""Render a deterministic held-out occupancy-forecast example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PANEL_WIDTH = 320
PANEL_HEIGHT = 400
PANEL_GAP = 32
LEFT_MARGIN = 150
TOP_MARGIN = 190
ROW_GAP = 74
HORIZONS_TO_RENDER = (0.5, 2.0, 6.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select the held-out clip nearest the median short-horizon Brier improvement "
            "and render its forecasts."
        )
    )
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--learned-predictions", type=Path, required=True)
    parser.add_argument("--persistence-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _font(
    size: int, *, bold: bool = False
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size=size)
    except OSError:
        return ImageFont.load_default()


def _prediction_path(directory: Path, row: dict[str, Any]) -> Path:
    pattern = f"{row['clip_id']}_{int(row['t0_us'])}_*.prediction.npz"
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one prediction for {pattern}, found {len(matches)}")
    return matches[0]


def _masked_brier(
    probability: np.ndarray, target: np.ndarray, observed: np.ndarray
) -> float:
    mask = observed.astype(bool)
    if not mask.any():
        raise ValueError("Cannot calculate Brier score without observed cells")
    error = probability[mask].astype(np.float32) - target[mask].astype(np.float32)
    return float(np.mean(np.square(error)))


def _select_median_example(
    rows: list[dict[str, Any]], learned_dir: Path, persistence_dir: Path
) -> tuple[dict[str, Any], Path, Path, float, float]:
    scored: list[tuple[float, dict[str, Any], Path, Path]] = []
    for row in rows:
        learned_path = _prediction_path(learned_dir, row)
        persistence_path = _prediction_path(persistence_dir, row)
        with (
            np.load(row["artifact_path"], allow_pickle=False) as target_archive,
            np.load(learned_path, allow_pickle=False) as learned_archive,
            np.load(persistence_path, allow_pickle=False) as persistence_archive,
        ):
            target = target_archive["occupancy"][:3]
            observed = target_archive["observed"][:3]
            learned = learned_archive["occupancy_prob"][:3]
            persistence = persistence_archive["occupancy_prob"][:3]
            learned_brier = np.mean(
                [
                    _masked_brier(learned[index], target[index], observed[index])
                    for index in range(3)
                ]
            )
            persistence_brier = np.mean(
                [
                    _masked_brier(persistence[index], target[index], observed[index])
                    for index in range(3)
                ]
            )
        scored.append(
            (
                float(persistence_brier - learned_brier),
                row,
                learned_path,
                persistence_path,
            )
        )

    improvements = np.asarray([item[0] for item in scored], dtype=np.float64)
    median = float(np.median(improvements))
    selected = min(scored, key=lambda item: (abs(item[0] - median), item[1]["clip_id"]))
    return selected[1], selected[2], selected[3], selected[0], median


def _resolve_artifact_paths(
    rows: list[dict[str, Any]], manifest_path: Path
) -> list[dict[str, Any]]:
    manifest_directory = manifest_path.expanduser().resolve().parent
    resolved = []
    for row in rows:
        artifact = Path(row["artifact_path"]).expanduser()
        if not artifact.is_absolute():
            artifact = manifest_directory / artifact
        resolved.append({**row, "artifact_path": str(artifact.resolve())})
    return resolved


def _orient(array: np.ndarray) -> np.ndarray:
    """Put ego-forward at the top and ego-left at the left of the image."""

    return np.fliplr(np.flipud(array.T))


def _probability_colors(probability: np.ndarray, observed: np.ndarray) -> np.ndarray:
    probability = np.clip(_orient(probability.astype(np.float32)), 0.0, 1.0)
    mask = _orient(observed.astype(bool))
    low = np.asarray([47, 20, 74], dtype=np.float32)
    middle = np.asarray([24, 139, 139], dtype=np.float32)
    high = np.asarray([253, 231, 37], dtype=np.float32)
    lower_weight = np.minimum(probability * 2.0, 1.0)[..., None]
    upper_weight = np.maximum(probability * 2.0 - 1.0, 0.0)[..., None]
    colors = low + lower_weight * (middle - low)
    colors = colors + upper_weight * (high - middle)
    colors[~mask] = np.asarray([12, 15, 20], dtype=np.float32)
    return colors.astype(np.uint8)


def _target_colors(target: np.ndarray, observed: np.ndarray) -> np.ndarray:
    target = _orient(target.astype(bool))
    mask = _orient(observed.astype(bool))
    colors = np.zeros((*target.shape, 3), dtype=np.uint8)
    colors[:] = (12, 15, 20)
    colors[mask] = (43, 49, 58)
    colors[target & mask] = (244, 247, 250)
    return colors


def _panel(colors: np.ndarray) -> Image.Image:
    image = Image.fromarray(colors)
    return image.resize((PANEL_WIDTH, PANEL_HEIGHT), resample=Image.Resampling.NEAREST)


def _draw_centered(
    draw: ImageDraw.ImageDraw,
    center_x: int,
    y: int,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    left, _, right, _ = draw.textbbox((0, 0), text, font=font)
    draw.text((center_x - (right - left) / 2, y), text, font=font, fill=fill)


def _draw_legend(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    width = 270
    for offset in range(width):
        probability = offset / (width - 1)
        color = _probability_colors(
            np.asarray([[probability]], dtype=np.float32),
            np.asarray([[1]], dtype=np.uint8),
        )[0, 0]
        draw.line(
            (x + offset, y, x + offset, y + 18),
            fill=tuple(int(value) for value in color),
        )
    draw.rectangle((x, y, x + width, y + 18), outline=(132, 141, 153), width=1)
    draw.text((x, y + 24), "0: empty", font=_font(18), fill=(177, 185, 196))
    right_text = "1: occupied"
    right_box = draw.textbbox((0, 0), right_text, font=_font(18))
    draw.text(
        (x + width - (right_box[2] - right_box[0]), y + 24),
        right_text,
        font=_font(18),
        fill=(177, 185, 196),
    )


def render(
    row: dict[str, Any],
    learned_path: Path,
    persistence_path: Path,
    improvement: float,
    median_improvement: float,
    output: Path,
) -> None:
    with (
        np.load(row["artifact_path"], allow_pickle=False) as target_archive,
        np.load(learned_path, allow_pickle=False) as learned_archive,
        np.load(persistence_path, allow_pickle=False) as persistence_archive,
    ):
        horizons = target_archive["horizons_s"].astype(np.float32)
        indices = [
            int(np.argmin(np.abs(horizons - value))) for value in HORIZONS_TO_RENDER
        ]
        if any(
            abs(float(horizons[index]) - value) > 1e-4
            for index, value in zip(indices, HORIZONS_TO_RENDER)
        ):
            raise ValueError("The required display horizons are not present")
        target = target_archive["occupancy"]
        observed = target_archive["observed"]
        learned = learned_archive["occupancy_prob"].astype(np.float32)
        persistence = persistence_archive["occupancy_prob"].astype(np.float32)

        rows_data = []
        for index in indices:
            rows_data.append(
                {
                    "horizon": float(horizons[index]),
                    "target": target[index],
                    "observed": observed[index],
                    "learned": learned[index],
                    "persistence": persistence[index],
                    "learned_brier": _masked_brier(
                        learned[index], target[index], observed[index]
                    ),
                    "persistence_brier": _masked_brier(
                        persistence[index], target[index], observed[index]
                    ),
                }
            )

    canvas_width = LEFT_MARGIN + 3 * PANEL_WIDTH + 2 * PANEL_GAP + 70
    canvas_height = TOP_MARGIN + 3 * PANEL_HEIGHT + 2 * ROW_GAP + 150
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(8, 11, 16))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (48, 30),
        "Held-out occupancy forecast: median test example",
        font=_font(36, bold=True),
        fill=(244, 247, 250),
    )
    draw.text(
        (48, 82),
        "Selected before plotting: closest clip to the median 0.5-2.0 s Brier improvement",
        font=_font(22),
        fill=(177, 185, 196),
    )
    draw.text(
        (48, 118),
        f"Clip {row['clip_id']}  |  improvement {improvement:.4f}  |  test median {median_improvement:.4f}",
        font=_font(19),
        fill=(132, 199, 255),
    )

    headers = ("Recorded future", "Learned forecast", "Copy-current baseline")
    for column, header in enumerate(headers):
        x = LEFT_MARGIN + column * (PANEL_WIDTH + PANEL_GAP)
        _draw_centered(
            draw,
            x + PANEL_WIDTH // 2,
            154,
            header,
            _font(23, bold=True),
            (231, 235, 240),
        )

    for row_index, values in enumerate(rows_data):
        y = TOP_MARGIN + row_index * (PANEL_HEIGHT + ROW_GAP)
        draw.text(
            (34, y + PANEL_HEIGHT // 2 - 30),
            f"{values['horizon']:g} s",
            font=_font(28, bold=True),
            fill=(244, 247, 250),
        )
        panels = (
            _panel(_target_colors(values["target"], values["observed"])),
            _panel(_probability_colors(values["learned"], values["observed"])),
            _panel(_probability_colors(values["persistence"], values["observed"])),
        )
        for column, panel in enumerate(panels):
            x = LEFT_MARGIN + column * (PANEL_WIDTH + PANEL_GAP)
            canvas.paste(panel, (x, y))
            draw.rectangle(
                (x, y, x + PANEL_WIDTH, y + PANEL_HEIGHT),
                outline=(85, 94, 108),
                width=2,
            )

        metric_y = y + PANEL_HEIGHT + 10
        learned_x = LEFT_MARGIN + PANEL_WIDTH + PANEL_GAP
        persistence_x = learned_x + PANEL_WIDTH + PANEL_GAP
        _draw_centered(
            draw,
            learned_x + PANEL_WIDTH // 2,
            metric_y,
            f"Brier {values['learned_brier']:.3f}",
            _font(19),
            (121, 211, 173),
        )
        _draw_centered(
            draw,
            persistence_x + PANEL_WIDTH // 2,
            metric_y,
            f"Brier {values['persistence_brier']:.3f}",
            _font(19),
            (255, 174, 102),
        )

    legend_y = canvas_height - 105
    draw.text(
        (48, legend_y),
        "Forecast probability",
        font=_font(20, bold=True),
        fill=(231, 235, 240),
    )
    _draw_legend(draw, 250, legend_y + 2)
    draw.rectangle((575, legend_y + 2, 597, legend_y + 24), fill=(244, 247, 250))
    draw.text(
        (608, legend_y), "recorded occupied cell", font=_font(18), fill=(177, 185, 196)
    )
    draw.rectangle(
        (860, legend_y + 2, 882, legend_y + 24),
        fill=(12, 15, 20),
        outline=(85, 94, 108),
    )
    draw.text(
        (893, legend_y),
        "not observed; not scored",
        font=_font(18),
        fill=(177, 185, 196),
    )
    draw.text(
        (48, canvas_height - 48),
        "Map view: forward is up. Each pixel represents 0.5 m. Lower Brier is better.",
        font=_font(19),
        fill=(177, 185, 196),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def main() -> None:
    args = parse_args()
    rows = _resolve_artifact_paths(
        [
            json.loads(line)
            for line in args.test_manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ],
        args.test_manifest,
    )
    if not rows:
        raise ValueError("Test manifest is empty")
    selected = _select_median_example(
        rows, args.learned_predictions, args.persistence_predictions
    )
    render(*selected, args.output)
    print(
        json.dumps(
            {
                "clip_id": selected[0]["clip_id"],
                "t0_us": int(selected[0]["t0_us"]),
                "short_horizon_brier_improvement": selected[3],
                "test_median_improvement": selected[4],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
