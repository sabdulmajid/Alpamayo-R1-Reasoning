"""Streaming, dependency-light occupancy metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class _HorizonState:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0
    brier_sum: float = 0.0
    count: int = 0


class OccupancyMetricAccumulator:
    """Accumulate threshold metrics, Brier score, and histogram AUPRC."""

    def __init__(self, num_horizons: int, threshold: float = 0.5, bins: int = 1000) -> None:
        if bins < 2:
            raise ValueError("bins must be at least 2")
        self.threshold = threshold
        self.bins = bins
        self.states = [_HorizonState() for _ in range(num_horizons)]
        self.positive_hist = np.zeros((num_horizons, bins), dtype=np.int64)
        self.negative_hist = np.zeros((num_horizons, bins), dtype=np.int64)

    def update(
        self, probabilities: torch.Tensor, target: torch.Tensor, visibility: torch.Tensor
    ) -> None:
        if probabilities.shape != target.shape or target.shape != visibility.shape:
            raise ValueError("probabilities, target, and visibility must have identical shapes")
        if probabilities.ndim != 4 or probabilities.shape[1] != len(self.states):
            raise ValueError("Expected [batch,horizon,y,x] matching configured horizons")
        probabilities_np = probabilities.detach().float().cpu().numpy()
        target_np = target.detach().cpu().numpy() > 0.5
        visibility_np = visibility.detach().cpu().numpy() > 0.5
        for horizon, state in enumerate(self.states):
            mask = visibility_np[:, horizon]
            scores = np.clip(probabilities_np[:, horizon][mask], 0.0, 1.0)
            labels = target_np[:, horizon][mask]
            if scores.size == 0:
                continue
            predicted = scores >= self.threshold
            state.true_positive += int(np.logical_and(predicted, labels).sum())
            state.false_positive += int(np.logical_and(predicted, ~labels).sum())
            state.false_negative += int(np.logical_and(~predicted, labels).sum())
            state.true_negative += int(np.logical_and(~predicted, ~labels).sum())
            state.brier_sum += float(np.square(scores - labels.astype(np.float32)).sum())
            state.count += int(scores.size)
            positive_scores = scores[labels]
            negative_scores = scores[~labels]
            self.positive_hist[horizon] += np.histogram(
                positive_scores, bins=self.bins, range=(0.0, 1.0)
            )[0]
            self.negative_hist[horizon] += np.histogram(
                negative_scores, bins=self.bins, range=(0.0, 1.0)
            )[0]

    def compute(self, horizons_s: list[float] | np.ndarray | None = None) -> dict[str, object]:
        results: list[dict[str, float | int | None]] = []
        for index, state in enumerate(self.states):
            tp, fp, fn = state.true_positive, state.false_positive, state.false_negative
            union = tp + fp + fn
            precision = tp / (tp + fp) if tp + fp else None
            recall = tp / (tp + fn) if tp + fn else None
            auprc = self._auprc(index)
            results.append(
                {
                    "horizon_s": float(horizons_s[index]) if horizons_s is not None else index,
                    "valid_cells": state.count,
                    "iou": tp / union if union else None,
                    "precision": precision,
                    "recall": recall,
                    "auprc": auprc,
                    "brier": state.brier_sum / state.count if state.count else None,
                }
            )
        valid = [result for result in results if result["valid_cells"]]
        mean_metrics: dict[str, float | None] = {}
        for name in ("iou", "precision", "recall", "auprc", "brier"):
            values = [float(result[name]) for result in valid if result[name] is not None]
            mean_metrics[name] = float(np.mean(values)) if values else None
        return {"per_horizon": results, "mean": mean_metrics, "threshold": self.threshold}

    def _auprc(self, horizon: int) -> float | None:
        positives = self.positive_hist[horizon][::-1].cumsum().astype(np.float64)
        negatives = self.negative_hist[horizon][::-1].cumsum().astype(np.float64)
        total_positives = positives[-1]
        if total_positives == 0:
            return None
        recall = positives / total_positives
        precision = positives / np.maximum(positives + negatives, 1.0)
        recall = np.concatenate(([0.0], recall))
        precision = np.concatenate(([1.0], precision))
        trapezoid = getattr(np, "trapezoid", None)
        if trapezoid is None:  # NumPy < 2.0
            trapezoid = np.trapz
        return float(trapezoid(precision, recall))
