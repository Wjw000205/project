"""Lightweight MLP adapter for an additive orthogonal Trend residual.

The network consumes only Trend-oriented coordinates from the current input,
frozen-base patch and aligned historical residual decomposition, plus known
channel/forecast-phase context.  It emits one signed coefficient on a fixed
zero-mean linear basis; it cannot emit Level, curvature or Amp shape.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from src.models.trend_basis_adapter import TrendAdapterOutput, trend_patch_features, zero_mean_linear_basis


CONTROLLER_FEATURE_WIDTH = 118
HISTORY_LENGTH = 96
BASE_PATCH_START = 100
BASE_PATCH_STOP = 112


def _linear_coefficient(values: torch.Tensor) -> torch.Tensor:
    basis = zero_mean_linear_basis(
        int(values.shape[-1]), device=values.device, dtype=values.dtype
    )
    centered = values - values.mean(dim=-1, keepdim=True)
    return (centered * basis).sum(dim=-1) / basis.square().sum().clamp_min(1.0e-12)


def trend_semantic_features(
    aligned_trend_residuals: torch.Tensor,
    controller_features: torch.Tensor,
    phase_context: torch.Tensor,
) -> torch.Tensor:
    """Build compact Trend-only coordinates for the MLP.

    ``controller_features`` is the frozen generic ``phi(x, y_base)`` tensor:
    its first 96 values are the ordered input normalized relative to the latest
    observation, and values 100:112 are the matching frozen-base patch.  Means,
    raw amplitude, curvature and free waveform values are not passed through.
    """

    if aligned_trend_residuals.ndim != 3:
        raise ValueError("aligned_trend_residuals must have shape [B,L,H]")
    if controller_features.ndim != 2 or int(controller_features.shape[1]) != CONTROLLER_FEATURE_WIDTH:
        raise ValueError("controller_features must have shape [B,118]")
    if phase_context.ndim != 2 or phase_context.shape[0] != controller_features.shape[0]:
        raise ValueError("phase_context must have shape [B,C]")
    if aligned_trend_residuals.shape[0] != controller_features.shape[0]:
        raise ValueError("Trend input batch mismatch")

    historical = trend_patch_features(aligned_trend_residuals).flatten(start_dim=1)
    input_path = controller_features[:, :HISTORY_LENGTH]
    input_slopes = torch.stack(
        [_linear_coefficient(input_path[:, -width:]) for width in (12, 24, 48, 96)],
        dim=1,
    )
    base_patch = controller_features[:, BASE_PATCH_START:BASE_PATCH_STOP]
    base_slopes = torch.stack(
        [
            _linear_coefficient(base_patch),
            0.5 * (base_patch[:, -1] - base_patch[:, 0]),
            _linear_coefficient(base_patch[:, :6]),
            _linear_coefficient(base_patch[:, 6:]),
        ],
        dim=1,
    )
    signed = torch.cat(
        [
            input_slopes,
            base_slopes,
            (base_slopes[:, :1] - input_slopes[:, :1]),
        ],
        dim=1,
    )
    trend_state = torch.cat([historical, signed, signed.abs()], dim=1)
    features = torch.cat([trend_state, phase_context], dim=1)
    return torch.nan_to_num(features, nan=0.0, posinf=8.0, neginf=-8.0)


def trend_phase_context(
    label_start: torch.Tensor,
    channel: torch.Tensor,
    patch: torch.Tensor,
    *,
    channel_count: int = 7,
    patch_count: int = 8,
    daily_period: int = 96,
    harmonics: int = 4,
) -> torch.Tensor:
    """Known channel/forecast-position/daily-phase context."""

    if label_start.ndim != 1 or channel.shape != label_start.shape or patch.shape != label_start.shape:
        raise ValueError("label_start, channel and patch must have shape [B]")
    dtype = torch.float32 if not label_start.is_floating_point() else label_start.dtype
    center = label_start.to(dtype) + patch.to(dtype) * 12.0 + 5.5
    angle = 2.0 * math.pi * center / float(daily_period)
    periodic = torch.stack(
        [
            function(harmonic * angle)
            for harmonic in range(1, harmonics + 1)
            for function in (torch.sin, torch.cos)
        ],
        dim=1,
    )
    channel_onehot = F.one_hot(channel.long(), num_classes=channel_count).to(dtype)
    patch_onehot = F.one_hot(patch.long(), num_classes=patch_count).to(dtype)
    channel_interaction = (
        channel_onehot.unsqueeze(-1) * periodic.unsqueeze(1)
    ).flatten(start_dim=1)
    patch_interaction = (
        patch_onehot.unsqueeze(-1) * periodic.unsqueeze(1)
    ).flatten(start_dim=1)
    return torch.cat(
        [channel_onehot, patch_onehot, periodic, channel_interaction, patch_interaction],
        dim=1,
    )


class TrendMLPAdapter(nn.Module):
    """One small shared MLP with coefficient, uncertainty and need outputs."""

    def __init__(
        self,
        feature_width: int,
        hidden: int = 32,
        max_abs_coefficient: float = 2.0,
        prior_probability: float = 0.25,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if feature_width <= 0 or hidden <= 0 or max_abs_coefficient <= 0.0:
            raise ValueError("invalid Trend MLP dimensions or coefficient bound")
        self.feature_width = int(feature_width)
        self.max_abs_coefficient = float(max_abs_coefficient)
        self.register_buffer("feature_center", torch.zeros(feature_width))
        self.register_buffer("feature_scale", torch.ones(feature_width))
        self.body = nn.Sequential(
            nn.Linear(feature_width, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.mean = nn.Linear(hidden, 1)
        self.log_scale = nn.Linear(hidden, 1)
        self.need = nn.Linear(hidden, 1)
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.log_scale.weight)
        nn.init.constant_(self.log_scale.bias, -2.0)
        nn.init.zeros_(self.need.weight)
        prior = min(max(float(prior_probability), 1.0e-4), 1.0 - 1.0e-4)
        nn.init.constant_(self.need.bias, math.log(prior / (1.0 - prior)))

    def set_normalization(self, center: torch.Tensor, scale: torch.Tensor) -> None:
        if center.shape != (self.feature_width,) or scale.shape != center.shape:
            raise ValueError("Trend MLP normalization shape mismatch")
        if not bool(torch.isfinite(center).all() and torch.isfinite(scale).all()):
            raise ValueError("nonfinite Trend MLP normalization")
        with torch.no_grad():
            self.feature_center.copy_(center)
            self.feature_scale.copy_(scale.clamp_min(1.0e-4))

    def forward(self, features: torch.Tensor, horizon: int = 12) -> TrendAdapterOutput:
        if features.ndim != 2 or int(features.shape[1]) != self.feature_width:
            raise ValueError("Trend MLP feature shape mismatch")
        standardized = ((features - self.feature_center) / self.feature_scale).clamp(-6.0, 6.0)
        hidden = self.body(standardized)
        coefficient_mean = self.max_abs_coefficient * torch.tanh(
            self.mean(hidden).squeeze(-1) / self.max_abs_coefficient
        )
        coefficient_scale = F.softplus(self.log_scale(hidden).squeeze(-1)) + 1.0e-4
        uncertainty_shrink = 1.0 / (1.0 + coefficient_scale)
        coefficient = uncertainty_shrink * coefficient_mean
        need_logit = self.need(hidden).squeeze(-1)
        need_probability = torch.sigmoid(need_logit)
        basis = zero_mean_linear_basis(horizon, device=features.device, dtype=features.dtype)
        correction = coefficient.unsqueeze(-1) * basis.unsqueeze(0)
        return TrendAdapterOutput(
            correction=correction,
            coefficient=coefficient,
            coefficient_mean=coefficient_mean,
            coefficient_scale=coefficient_scale,
            uncertainty_shrink=uncertainty_shrink,
            need_logit=need_logit,
            need_probability=need_probability,
        )


__all__ = [
    "TrendMLPAdapter",
    "trend_phase_context",
    "trend_semantic_features",
]
