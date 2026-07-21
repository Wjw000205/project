"""Universal target-free reliability gate for a frozen periodic Amp expert.

The gate owns only eight bounded p12 confidence values.  It deliberately has
no dataset, horizon, native-period, channel-cluster, Amp-expert, or free
residual state; native execution remains the responsibility of the frozen Amp
expert and its Q-space decoder.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class UniversalPeriodicAmpReliabilityGate(nn.Module):
    """Fixed H96/p12 confidence kernel for suppressing frozen Amp actions."""

    input_lanes = 17
    summary_lanes = 7
    history_steps = 96
    patch_steps = 12
    patches = 8
    parameter_count = 4_065
    initial_confidence = 0.9

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(17, 24, kernel_size=5, padding=2),
            nn.GroupNorm(4, 24),
            nn.GELU(),
            nn.Conv1d(24, 24, kernel_size=3, padding=1),
            nn.GroupNorm(4, 24),
            nn.GELU(),
        )
        self.pool = nn.AvgPool1d(12, stride=12)
        self.summary = nn.Sequential(nn.Linear(14, 8), nn.GELU())
        self.head = nn.Conv1d(32, 1, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(
            self.head.bias,
            math.log(self.initial_confidence / (1.0 - self.initial_confidence)),
        )

        actual_parameters = sum(parameter.numel() for parameter in self.parameters())
        if actual_parameters != self.parameter_count:
            raise RuntimeError("Amp reliability Gate parameter-count drift")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if (
            features.ndim != 3
            or int(features.shape[1]) != self.input_lanes
            or int(features.shape[2]) != self.history_steps
        ):
            raise ValueError("Amp reliability feature shape must be [B, 17, 96]")

        encoded = self.pool(self.encoder(features))
        dynamic = features[:, : self.summary_lanes]
        summary = self.summary(
            torch.cat((dynamic.mean(dim=2), dynamic.std(dim=2)), dim=1)
        )
        combined = torch.cat(
            (encoded, summary[:, :, None].expand(-1, -1, self.patches)), dim=1
        )
        return torch.sigmoid(self.head(combined).squeeze(1))


__all__ = ["UniversalPeriodicAmpReliabilityGate"]
