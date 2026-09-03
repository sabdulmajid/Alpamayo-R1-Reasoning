"""Visibility-masked, class-balanced occupancy losses."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def masked_balanced_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    visibility: torch.Tensor,
    *,
    max_positive_weight: float = 30.0,
) -> torch.Tensor:
    """Binary cross entropy with per-batch occupancy balancing."""

    mask = visibility.to(dtype=logits.dtype)
    positives = (target * mask).sum()
    valid = mask.sum()
    negatives = (valid - positives).clamp_min(0)
    positive_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, max_positive_weight)
    weights = torch.where(target > 0.5, positive_weight, torch.ones_like(target)) * mask
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (raw * weights).sum() / weights.sum().clamp_min(1.0)


def masked_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    visibility: torch.Tensor,
    *,
    epsilon: float = 1.0,
) -> torch.Tensor:
    probabilities = logits.sigmoid() * visibility
    masked_target = target * visibility
    dimensions = tuple(range(2, logits.ndim))
    intersection = (probabilities * masked_target).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + masked_target.sum(dim=dimensions)
    return (1.0 - (2.0 * intersection + epsilon) / (denominator + epsilon)).mean()


def occupancy_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    visibility: torch.Tensor,
    *,
    dice_weight: float = 0.25,
    max_positive_weight: float = 30.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    bce = masked_balanced_bce(
        logits, target, visibility, max_positive_weight=max_positive_weight
    )
    dice = masked_dice_loss(logits, target, visibility)
    total = bce + dice_weight * dice
    components = {
        "loss": float(total.detach()),
        "bce": float(bce.detach()),
        "dice": float(dice.detach()),
    }
    return total, components
