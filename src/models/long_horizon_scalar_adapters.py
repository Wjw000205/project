"""Cell-local scalar residual adapters for long-horizon frozen forecasts.

The models in this module share an interface, not parameters.  A Level model
emits a sign times a non-negative magnitude.  A Trend model emits one signed
coefficient on a fixed zero-mean linear basis.  Neither model can emit an Amp
waveform or replace the frozen backbone forecast.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ScalarAdapterOutput:
    coordinate: torch.Tensor
    uncertainty: torch.Tensor
    sign_logit: torch.Tensor | None = None
    magnitude: torch.Tensor | None = None


@dataclass(frozen=True)
class FixedBlockLevelTrendOutput:
    level_coordinate: torch.Tensor
    trend_coordinate: torch.Tensor
    level_correction: torch.Tensor
    trend_correction: torch.Tensor
    level_uncertainty: torch.Tensor
    trend_uncertainty: torch.Tensor


def fixed_block_level_trend_decomposition(
    values: torch.Tensor,
    *,
    block_steps: int = 96,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Orthogonally decompose [B,H] into repeated fixed-block coordinates."""

    if values.ndim != 2:
        raise ValueError("fixed-block decomposition expects [B,H]")
    if block_steps <= 1:
        raise ValueError("block_steps must exceed one")
    levels = []
    trends = []
    level_correction = torch.empty_like(values)
    trend_correction = torch.empty_like(values)
    for left in range(0, values.shape[1], block_steps):
        right = min(left + block_steps, values.shape[1])
        block = values[:, left:right]
        basis = torch.linspace(
            -1.0, 1.0, block.shape[1], device=values.device, dtype=values.dtype
        )
        basis = basis - basis.mean()
        level = block.mean(dim=1)
        centered = block - level[:, None]
        trend = (centered * basis).sum(dim=1) / basis.square().sum().clamp_min(1.0e-12)
        levels.append(level)
        trends.append(trend)
        level_correction[:, left:right] = level[:, None]
        trend_correction[:, left:right] = trend[:, None] * basis
    return (
        torch.stack(levels, dim=1),
        torch.stack(trends, dim=1),
        level_correction,
        trend_correction,
    )


def fixed_block_reconstruction(
    level: torch.Tensor,
    trend: torch.Tensor,
    horizon: int,
    *,
    block_steps: int = 96,
) -> tuple[torch.Tensor, torch.Tensor]:
    if level.ndim != 2 or trend.shape != level.shape:
        raise ValueError("fixed-block coordinates must have matching [B,S] shape")
    segment_count = (int(horizon) + block_steps - 1) // block_steps
    if level.shape[1] != segment_count:
        raise ValueError("fixed-block coordinate count mismatch")
    level_correction = level.new_empty((level.shape[0], horizon))
    trend_correction = trend.new_empty((trend.shape[0], horizon))
    for segment, left in enumerate(range(0, horizon, block_steps)):
        right = min(left + block_steps, horizon)
        basis = torch.linspace(
            -1.0, 1.0, right - left, device=level.device, dtype=level.dtype
        )
        basis = basis - basis.mean()
        level_correction[:, left:right] = level[:, segment, None]
        trend_correction[:, left:right] = trend[:, segment, None] * basis
    return level_correction, trend_correction


class FixedBlockLevelTrendAdapter(nn.Module):
    """Causal shared segment network for repeated 96-step Level/Trend targets."""

    def __init__(
        self,
        *,
        patch_len: int = 12,
        block_steps: int = 96,
        carrier_count: int = 8,
        context_width: int = 11,
        hidden: int = 64,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if block_steps % patch_len != 0 or block_steps <= patch_len:
            raise ValueError("block_steps must contain complete patches")
        if hidden <= 0 or hidden % heads != 0 or layers <= 0:
            raise ValueError("invalid fixed-block adapter dimensions")
        self.patch_len = int(patch_len)
        self.block_steps = int(block_steps)
        self.block_patches = self.block_steps // self.patch_len
        self.carrier_count = int(carrier_count)
        self.context_width = int(context_width)
        self.hidden = int(hidden)
        self.carrier_encoder = nn.Sequential(
            nn.Linear(4, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.carrier_age_encoder = nn.Linear(4, hidden)
        self.base_encoder = nn.Sequential(
            nn.Linear(4, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.position_encoder = nn.Linear(8, hidden)
        self.context_encoder = nn.Sequential(
            nn.Linear(context_width, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=2 * hidden,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.segment_encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden)
        self.level_head = nn.Linear(hidden, 1)
        self.trend_head = nn.Linear(hidden, 1)
        self.level_uncertainty_head = nn.Linear(hidden, 1)
        self.trend_uncertainty_head = nn.Linear(hidden, 1)
        for head in (self.level_head, self.trend_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        for head in (self.level_uncertainty_head, self.trend_uncertainty_head):
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, -1.5)

    @staticmethod
    def _coordinates(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        basis = torch.linspace(
            -1.0, 1.0, values.shape[-1], device=values.device, dtype=values.dtype
        )
        basis = basis - basis.mean()
        level = values.mean(dim=-1)
        centered = values - level.unsqueeze(-1)
        trend = (centered * basis).sum(dim=-1) / basis.square().sum().clamp_min(1.0e-12)
        amp = centered - trend.unsqueeze(-1) * basis
        amp_rms = torch.sqrt(amp.square().mean(dim=-1) + 1.0e-8)
        return level, trend, amp_rms

    def forward(
        self,
        carriers: torch.Tensor,
        base_patch: torch.Tensor,
        context: torch.Tensor,
    ) -> FixedBlockLevelTrendOutput:
        if carriers.ndim != 4:
            raise ValueError("carriers must have shape [B,P,K,L]")
        batch, patch_count, carrier_count, patch_len = carriers.shape
        if carrier_count != self.carrier_count or patch_len != self.patch_len:
            raise ValueError("fixed-block carrier dimensions mismatch")
        if base_patch.shape != (batch, patch_count, patch_len):
            raise ValueError("fixed-block base dimensions mismatch")
        if context.shape != (batch, self.context_width):
            raise ValueError("fixed-block context dimensions mismatch")
        horizon = patch_count * patch_len
        tokens = []
        scales = []
        lengths = []
        age = torch.linspace(
            -1.0, 1.0, carrier_count, device=carriers.device, dtype=carriers.dtype
        )
        age_feature = torch.stack(
            [age, age.square(), torch.sin(math.pi * age), torch.cos(math.pi * age)],
            dim=1,
        )
        age_token = self.carrier_age_encoder(age_feature).view(1, carrier_count, self.hidden)
        for left in range(0, patch_count, self.block_patches):
            right = min(left + self.block_patches, patch_count)
            width = (right - left) * patch_len
            carrier_block = carriers[:, left:right].permute(0, 2, 1, 3).reshape(
                batch, carrier_count, width
            )
            carrier_level, carrier_trend, carrier_amp_rms = self._coordinates(carrier_block)
            carrier_rms = torch.sqrt(carrier_block.square().mean(dim=-1) + 1.0e-8)
            segment_scale = torch.sqrt(carrier_block.square().mean(dim=(1, 2)) + 1.0e-8)
            scale = segment_scale[:, None].clamp_min(1.0e-5)
            carrier_feature = torch.stack(
                [
                    carrier_level / scale,
                    carrier_trend / scale,
                    carrier_amp_rms / scale,
                    torch.log(carrier_rms / scale + 1.0e-6),
                ],
                dim=-1,
            )
            carrier_token = (self.carrier_encoder(carrier_feature) + age_token).mean(dim=1)

            base_block = base_patch[:, left:right].reshape(batch, width)
            base_level, base_trend, base_amp_rms = self._coordinates(base_block)
            base_rms = torch.sqrt(base_block.square().mean(dim=1) + 1.0e-8).clamp_min(1.0e-5)
            base_feature = torch.stack(
                [
                    base_level / base_rms,
                    base_trend / base_rms,
                    base_amp_rms / base_rms,
                    torch.log(base_rms / segment_scale.clamp_min(1.0e-5) + 1.0e-6),
                ],
                dim=1,
            )
            tokens.append(carrier_token + self.base_encoder(base_feature))
            scales.append(segment_scale)
            lengths.append(width)
        token = torch.stack(tokens, dim=1)
        segment_count = token.shape[1]
        midpoint_steps = torch.as_tensor(
            [
                min(left + self.block_steps, horizon) / 2.0 + left / 2.0
                for left in range(0, horizon, self.block_steps)
            ],
            device=token.device,
            dtype=token.dtype,
        )
        lead_days = midpoint_steps / 96.0
        daily = 2.0 * math.pi * midpoint_steps / 96.0
        weekly = 2.0 * math.pi * midpoint_steps / (7.0 * 96.0)
        length_fraction = torch.as_tensor(
            lengths, device=token.device, dtype=token.dtype
        ) / self.block_steps
        position = torch.stack(
            [
                torch.sin(daily),
                torch.cos(daily),
                torch.sin(weekly),
                torch.cos(weekly),
                lead_days,
                torch.log1p(lead_days),
                1.0 / (1.0 + lead_days),
                length_fraction,
            ],
            dim=1,
        )
        token = (
            token
            + self.position_encoder(position).unsqueeze(0)
            + self.context_encoder(context).unsqueeze(1)
        )
        mask = torch.ones(
            segment_count, segment_count, device=token.device, dtype=torch.bool
        ).triu(diagonal=1)
        encoded = self.norm(self.segment_encoder(token, mask=mask))
        segment_scale = torch.stack(scales, dim=1)
        level = self.level_head(encoded).squeeze(-1) * segment_scale
        trend = self.trend_head(encoded).squeeze(-1) * segment_scale
        level_correction, trend_correction = fixed_block_reconstruction(
            level, trend, horizon, block_steps=self.block_steps
        )
        return FixedBlockLevelTrendOutput(
            level_coordinate=level,
            trend_coordinate=trend,
            level_correction=level_correction,
            trend_correction=trend_correction,
            level_uncertainty=F.softplus(self.level_uncertainty_head(encoded).squeeze(-1)) + 1.0e-4,
            trend_uncertainty=F.softplus(self.trend_uncertainty_head(encoded).squeeze(-1)) + 1.0e-4,
        )


class _NormalizedMLP(nn.Module):
    def __init__(
        self,
        feature_width: int,
        *,
        hidden: int = 64,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if feature_width <= 0 or hidden <= 0:
            raise ValueError("invalid scalar-adapter dimensions")
        self.feature_width = int(feature_width)
        self.register_buffer("feature_center", torch.zeros(feature_width))
        self.register_buffer("feature_scale", torch.ones(feature_width))
        self.body = nn.Sequential(
            nn.Linear(feature_width, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )

    def set_normalization(self, center: torch.Tensor, scale: torch.Tensor) -> None:
        if center.shape != (self.feature_width,) or scale.shape != center.shape:
            raise ValueError("scalar-adapter normalization shape mismatch")
        if not bool(torch.isfinite(center).all() and torch.isfinite(scale).all()):
            raise ValueError("nonfinite scalar-adapter normalization")
        with torch.no_grad():
            self.feature_center.copy_(center)
            self.feature_scale.copy_(scale.clamp_min(1.0e-5))

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or int(features.shape[1]) != self.feature_width:
            raise ValueError("scalar-adapter feature shape mismatch")
        normalized = ((features - self.feature_center) / self.feature_scale).clamp(-8.0, 8.0)
        return self.body(torch.nan_to_num(normalized, nan=0.0, posinf=8.0, neginf=-8.0))


class LevelSignMagnitudeAdapter(_NormalizedMLP):
    """Level-specific sign x magnitude model with heteroscedastic uncertainty."""

    def __init__(
        self,
        feature_width: int,
        *,
        hidden: int = 64,
        dropout: float = 0.10,
    ) -> None:
        super().__init__(feature_width, hidden=hidden, dropout=dropout)
        self.sign = nn.Linear(hidden, 1)
        self.log_magnitude = nn.Linear(hidden, 1)
        self.log_uncertainty = nn.Linear(hidden, 1)
        nn.init.zeros_(self.sign.weight)
        nn.init.zeros_(self.sign.bias)
        nn.init.zeros_(self.log_magnitude.weight)
        nn.init.constant_(self.log_magnitude.bias, -0.5)
        nn.init.zeros_(self.log_uncertainty.weight)
        nn.init.constant_(self.log_uncertainty.bias, -1.5)

    def forward(self, features: torch.Tensor) -> ScalarAdapterOutput:
        hidden = self.encode(features)
        sign_logit = self.sign(hidden).squeeze(-1)
        magnitude = F.softplus(self.log_magnitude(hidden).squeeze(-1))
        uncertainty = F.softplus(self.log_uncertainty(hidden).squeeze(-1)) + 1.0e-4
        coordinate = torch.tanh(sign_logit) * magnitude
        return ScalarAdapterOutput(
            coordinate=coordinate,
            uncertainty=uncertainty,
            sign_logit=sign_logit,
            magnitude=magnitude,
        )


class TrendCoefficientAdapter(_NormalizedMLP):
    """Trend-specific direct signed coefficient with no free waveform output."""

    def __init__(
        self,
        feature_width: int,
        *,
        hidden: int = 64,
        dropout: float = 0.10,
    ) -> None:
        super().__init__(feature_width, hidden=hidden, dropout=dropout)
        self.mean = nn.Linear(hidden, 1)
        self.log_uncertainty = nn.Linear(hidden, 1)
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.log_uncertainty.weight)
        nn.init.constant_(self.log_uncertainty.bias, -1.5)

    def forward(self, features: torch.Tensor) -> ScalarAdapterOutput:
        hidden = self.encode(features)
        coordinate = self.mean(hidden).squeeze(-1)
        uncertainty = F.softplus(self.log_uncertainty(hidden).squeeze(-1)) + 1.0e-4
        return ScalarAdapterOutput(coordinate=coordinate, uncertainty=uncertainty)


class AmpPhaseScaleAdapter(_NormalizedMLP):
    """Amp-specific signed phase/scale coefficient for an external shape candidate.

    Unlike Level, this coefficient is never itself expanded as a constant
    correction.  The caller multiplies it by a separately generated zero-affine
    Amp shape.  The class is intentionally separate so no Level parameter or
    output semantics can be reused accidentally.
    """

    def __init__(
        self,
        feature_width: int,
        *,
        hidden: int = 96,
        dropout: float = 0.10,
    ) -> None:
        super().__init__(feature_width, hidden=hidden, dropout=dropout)
        self.sign = nn.Linear(hidden, 1)
        self.log_magnitude = nn.Linear(hidden, 1)
        self.log_uncertainty = nn.Linear(hidden, 1)
        nn.init.zeros_(self.sign.weight)
        nn.init.zeros_(self.sign.bias)
        nn.init.zeros_(self.log_magnitude.weight)
        nn.init.constant_(self.log_magnitude.bias, -0.25)
        nn.init.zeros_(self.log_uncertainty.weight)
        nn.init.constant_(self.log_uncertainty.bias, -1.25)

    def forward(self, features: torch.Tensor) -> ScalarAdapterOutput:
        hidden = self.encode(features)
        sign_logit = self.sign(hidden).squeeze(-1)
        magnitude = F.softplus(self.log_magnitude(hidden).squeeze(-1))
        uncertainty = F.softplus(self.log_uncertainty(hidden).squeeze(-1)) + 1.0e-4
        coordinate = torch.tanh(sign_logit) * magnitude
        return ScalarAdapterOutput(
            coordinate=coordinate,
            uncertainty=uncertainty,
            sign_logit=sign_logit,
            magnitude=magnitude,
        )


def zero_mean_linear_basis(
    horizon: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if horizon <= 1:
        raise ValueError("Trend horizon must exceed one")
    basis = torch.linspace(-1.0, 1.0, horizon, device=device, dtype=dtype)
    return basis - basis.mean()


__all__ = [
    "AmpPhaseScaleAdapter",
    "FixedBlockLevelTrendAdapter",
    "FixedBlockLevelTrendOutput",
    "LevelSignMagnitudeAdapter",
    "ScalarAdapterOutput",
    "TrendCoefficientAdapter",
    "fixed_block_level_trend_decomposition",
    "fixed_block_reconstruction",
    "zero_mean_linear_basis",
]
