"""Small recurrent convolutional model for multi-horizon occupancy."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class ConvGRUCell(nn.Module):
    """Convolutional GRU cell operating at the encoder bottleneck."""

    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        joint = input_channels + hidden_channels
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(joint, 2 * hidden_channels, 3, padding=1)
        self.candidate = nn.Conv2d(joint, hidden_channels, 3, padding=1)

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor | None) -> torch.Tensor:
        if hidden is None:
            hidden = inputs.new_zeros(
                inputs.shape[0], self.hidden_channels, inputs.shape[2], inputs.shape[3]
            )
        reset, update = self.gates(torch.cat((inputs, hidden), dim=1)).sigmoid().chunk(2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat((inputs, reset * hidden), dim=1)))
        return (1.0 - update) * hidden + update * candidate


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 2
    base_channels: int = 16
    num_horizons: int = 6

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class TemporalOccupancyNet(nn.Module):
    """Encode each past grid, aggregate with ConvGRU, and decode all horizons."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.in_channels < 1 or config.base_channels < 1 or config.num_horizons < 1:
            raise ValueError("All model channel and horizon counts must be positive")
        self.config = config
        base = config.base_channels
        self.encoder_1 = ConvBlock(config.in_channels, base)
        self.encoder_2 = ConvBlock(base, 2 * base, stride=2)
        self.encoder_3 = ConvBlock(2 * base, 4 * base, stride=2)
        self.temporal = ConvGRUCell(4 * base, 4 * base)
        self.decoder_2 = ConvBlock(6 * base, 2 * base)
        self.decoder_1 = ConvBlock(3 * base, base)
        self.output = nn.Conv2d(base, config.num_horizons, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return logits shaped ``[batch, horizon, y, x]``."""

        if inputs.ndim != 5:
            raise ValueError(f"Expected [batch,time,channel,y,x], got {tuple(inputs.shape)}")
        if inputs.shape[2] != self.config.in_channels:
            raise ValueError(
                f"Expected {self.config.in_channels} input channels, got {inputs.shape[2]}"
            )
        hidden: torch.Tensor | None = None
        last_skip_1: torch.Tensor | None = None
        last_skip_2: torch.Tensor | None = None
        for time_index in range(inputs.shape[1]):
            skip_1 = self.encoder_1(inputs[:, time_index])
            skip_2 = self.encoder_2(skip_1)
            encoded = self.encoder_3(skip_2)
            hidden = self.temporal(encoded, hidden)
            last_skip_1, last_skip_2 = skip_1, skip_2
        if hidden is None or last_skip_1 is None or last_skip_2 is None:
            raise ValueError("At least one history frame is required")
        decoded = F.interpolate(
            hidden, size=last_skip_2.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded = self.decoder_2(torch.cat((decoded, last_skip_2), dim=1))
        decoded = F.interpolate(
            decoded, size=last_skip_1.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded = self.decoder_1(torch.cat((decoded, last_skip_1), dim=1))
        return self.output(decoded)
