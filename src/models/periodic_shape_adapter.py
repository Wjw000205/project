"""Universal physical-period Shape adapter with isolated parameters."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class PeriodicShapeOutput:
    mixture_weights: torch.Tensor
    seasonal_anchor: torch.Tensor
    residual: torch.Tensor
    action_strength: torch.Tensor
    raw_correction: torch.Tensor


class UniversalPeriodicShapeAdapter(nn.Module):
    """Decode one canonical P96 waveform from causal seasonal memory.

    The learned kernel is independent of the native period, dataset, channel
    count, forecast horizon, and number of physical-period instances.  A
    parameter-free caller maps P96 to the native clock and projects the raw
    correction into the complement of Level, Trend, and the causal Amp basis.
    """

    feature_width = 591
    canonical_steps = 96
    anchor_count = 4
    parameter_count = 85253

    def __init__(
        self,
        *,
        max_residual: float = 0.50,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if max_residual <= 0.0 or not 0.0 <= dropout < 1.0:
            raise ValueError("invalid universal Shape configuration")
        self.max_residual = float(max_residual)
        self.register_buffer("feature_center", torch.zeros(self.feature_width))
        self.register_buffer("feature_scale", torch.ones(self.feature_width))
        self.history_base_encoder = nn.Linear(192, 64)
        self.memory_encoder = nn.Linear(384, 96)
        self.context_encoder = nn.Linear(15, 16)
        self.body = nn.Sequential(
            nn.Linear(176, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.mixture_head = nn.Linear(128, self.anchor_count)
        self.residual_head = nn.Linear(128, self.canonical_steps)
        self.action_head = nn.Linear(128, 1)
        for layer in (
            self.history_base_encoder,
            self.memory_encoder,
            self.context_encoder,
            self.body[0],
        ):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.mixture_head.weight)
        nn.init.zeros_(self.mixture_head.bias)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)

    @torch.no_grad()
    def set_normalization(
        self,
        center: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        center = torch.as_tensor(center, dtype=self.feature_center.dtype)
        scale = torch.as_tensor(scale, dtype=self.feature_scale.dtype)
        if center.shape != self.feature_center.shape or scale.shape != self.feature_scale.shape:
            raise ValueError("Shape normalization shape mismatch")
        if not torch.isfinite(center).all() or not torch.isfinite(scale).all():
            raise ValueError("Shape normalization must be finite")
        self.feature_center.copy_(center)
        self.feature_scale.copy_(scale.clamp_min(1.0e-6))

    def forward(
        self,
        features: torch.Tensor,
        anchors: torch.Tensor,
    ) -> PeriodicShapeOutput:
        if features.ndim != 2 or int(features.shape[1]) != self.feature_width:
            raise ValueError("Shape feature shape mismatch")
        expected_anchor = (
            int(features.shape[0]),
            self.anchor_count,
            self.canonical_steps,
        )
        if tuple(anchors.shape) != expected_anchor:
            raise ValueError("Shape anchor shape mismatch")
        normalized = ((features - self.feature_center) / self.feature_scale).clamp(
            -6.0, 6.0
        )
        normalized = torch.nan_to_num(
            normalized, nan=0.0, posinf=6.0, neginf=-6.0
        )
        history_base = F.silu(self.history_base_encoder(normalized[:, :192]))
        memory = F.silu(self.memory_encoder(normalized[:, 192:576]))
        context = F.silu(self.context_encoder(normalized[:, 576:]))
        hidden = self.body(torch.cat([history_base, memory, context], dim=1))
        weights = torch.softmax(self.mixture_head(hidden), dim=1)
        seasonal = torch.sum(weights[:, :, None] * anchors, dim=1)
        residual = self.max_residual * torch.tanh(
            self.residual_head(hidden) / self.max_residual
        )
        action = torch.tanh(self.action_head(hidden).squeeze(1))
        return PeriodicShapeOutput(
            mixture_weights=weights,
            seasonal_anchor=seasonal,
            residual=residual,
            action_strength=action,
            raw_correction=action[:, None] * (seasonal + residual),
        )


__all__ = ["PeriodicShapeOutput", "UniversalPeriodicShapeAdapter"]
