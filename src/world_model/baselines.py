"""Non-learned occupancy forecasting baselines."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def persistence(inputs: torch.Tensor, num_horizons: int) -> torch.Tensor:
    """Repeat the last observed occupancy at every forecast horizon."""

    if inputs.ndim != 5:
        raise ValueError("inputs must have shape [batch,time,channel,y,x]")
    return inputs[:, -1, 0].unsqueeze(1).expand(-1, num_horizons, -1, -1)


def ego_compensated_persistence(
    inputs: torch.Tensor,
    future_ego_motion: torch.Tensor,
    grid_origin_xy_m: torch.Tensor,
    resolution_m: torch.Tensor,
) -> torch.Tensor:
    """Warp the last observation into future ego frames.

    ``future_ego_motion[b,h] = (x, y, yaw)`` is the future ego pose in the
    current ego frame. This baseline is only valid when target grids are each
    expressed in their corresponding future ego frame.
    """

    if inputs.ndim != 5 or future_ego_motion.ndim != 3:
        raise ValueError("Invalid input or future_ego_motion dimensions")
    occupancy = inputs[:, -1, 0]
    batch, height, width = occupancy.shape
    horizons = future_ego_motion.shape[1]
    device, dtype = occupancy.device, occupancy.dtype
    row = torch.arange(height, device=device, dtype=dtype)
    column = torch.arange(width, device=device, dtype=dtype)
    rows, columns = torch.meshgrid(row, column, indexing="ij")
    outputs: list[torch.Tensor] = []
    for horizon in range(horizons):
        pose = future_ego_motion[:, horizon].to(device=device, dtype=dtype)
        origin = grid_origin_xy_m.to(device=device, dtype=dtype)
        resolution = resolution_m.to(device=device, dtype=dtype)
        target_x = origin[:, 0, None, None] + (columns + 0.5) * resolution[:, None, None]
        target_y = origin[:, 1, None, None] + (rows + 0.5) * resolution[:, None, None]
        cosine, sine = pose[:, 2].cos()[:, None, None], pose[:, 2].sin()[:, None, None]
        source_x = cosine * target_x - sine * target_y + pose[:, 0, None, None]
        source_y = sine * target_x + cosine * target_y + pose[:, 1, None, None]
        source_column = (source_x - origin[:, 0, None, None]) / resolution[:, None, None] - 0.5
        source_row = (source_y - origin[:, 1, None, None]) / resolution[:, None, None] - 0.5
        grid_x = 2.0 * (source_column + 0.5) / width - 1.0
        grid_y = 2.0 * (source_row + 0.5) / height - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1)
        outputs.append(
            F.grid_sample(
                occupancy.unsqueeze(1),
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            ).squeeze(1)
        )
    return torch.stack(outputs, dim=1)
