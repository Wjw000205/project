"""Penalty-specific linear-basis adapter for frozen-backbone trend errors.

The adapter never emits a free waveform.  It predicts one signed coefficient
on an exactly zero-mean linear basis, so its correction cannot duplicate Level
or nonlinear Amp/d2 shape.  Inputs are fully observed, horizon-aligned historic
frozen-forecast residual patches plus the current frozen-base patch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class TrendAdapterOutput:
    correction: torch.Tensor
    coefficient: torch.Tensor
    coefficient_mean: torch.Tensor
    coefficient_scale: torch.Tensor
    uncertainty_shrink: torch.Tensor
    need_logit: torch.Tensor
    need_probability: torch.Tensor


def zero_mean_linear_basis(
    horizon: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if horizon < 2:
        raise ValueError("trend basis requires horizon >= 2")
    basis = torch.linspace(-1.0, 1.0, horizon, device=device, dtype=dtype)
    return basis - basis.mean()


def trend_patch_features(patches: torch.Tensor) -> torch.Tensor:
    """Return only coordinates inside the residual trend subspace.

    Args:
        patches: [..., H] historic residual or frozen-base patches.
    Returns:
        [..., 2]: signed LS trend coefficient and its absolute magnitude.

    Mean/Level, endpoint-only shape, curvature and diff-amplitude coordinates
    are deliberately excluded from the Trend body.
    """

    if patches.ndim < 2 or int(patches.shape[-1]) < 2:
        raise ValueError("patches must have shape [..., H>=2]")
    horizon = int(patches.shape[-1])
    basis = zero_mean_linear_basis(
        horizon, device=patches.device, dtype=patches.dtype
    )
    centered = patches - patches.mean(dim=-1, keepdim=True)
    denom = basis.square().sum().clamp_min(1.0e-12)
    ls_coefficient = (centered * basis).sum(dim=-1, keepdim=True) / denom
    features = torch.cat([ls_coefficient, ls_coefficient.abs()], dim=-1)
    return torch.nan_to_num(features, nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0)


class _TrendStateEncoder(nn.Module):
    """Small ordered encoder; separate instances keep training stages disjoint."""

    def __init__(
        self,
        history_patches: int,
        hidden: int,
        channel_count: int,
        patch_count: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.history_patches = int(history_patches)
        self.token_projection = nn.Sequential(
            nn.Linear(2, hidden),
            nn.SiLU(),
        )
        self.lag_embedding = nn.Embedding(history_patches, hidden)
        self.sequence = nn.GRU(hidden, hidden, batch_first=True)
        self.base_projection = nn.Sequential(nn.Linear(2, hidden), nn.SiLU())
        self.channel_embedding = nn.Embedding(channel_count, embedding_dim)
        self.patch_embedding = nn.Embedding(patch_count, embedding_dim)
        self.output_dim = 2 * hidden + 2 * embedding_dim

    def forward(
        self,
        aligned_residuals: torch.Tensor,
        base: torch.Tensor,
        channel: torch.Tensor,
        patch: torch.Tensor,
    ) -> torch.Tensor:
        if aligned_residuals.ndim != 3:
            raise ValueError("aligned_residuals must have shape [B,L,H]")
        if int(aligned_residuals.shape[1]) != self.history_patches:
            raise ValueError("history patch count drift")
        if base.ndim != 2 or base.shape != aligned_residuals[:, 0].shape:
            raise ValueError("base must have shape [B,H]")
        batch = int(base.shape[0])
        if channel.shape != (batch,) or patch.shape != (batch,):
            raise ValueError("channel and patch must have shape [B]")
        token = self.token_projection(trend_patch_features(aligned_residuals))
        lag = torch.arange(
            self.history_patches,
            device=aligned_residuals.device,
            dtype=torch.long,
        )
        token = token + self.lag_embedding(lag).unsqueeze(0)
        _, state = self.sequence(token)
        base_state = self.base_projection(trend_patch_features(base))
        return torch.cat(
            [
                state[-1],
                base_state,
                self.channel_embedding(channel),
                self.patch_embedding(patch),
            ],
            dim=-1,
        )


class TrendNeedHead(nn.Module):
    def __init__(
        self,
        history_patches: int = 8,
        hidden: int = 16,
        channel_count: int = 7,
        patch_count: int = 8,
        embedding_dim: int = 4,
        prior_probability: float = 0.25,
    ) -> None:
        super().__init__()
        self.encoder = _TrendStateEncoder(
            history_patches, hidden, channel_count, patch_count, embedding_dim
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.classifier[-1].weight)
        prior = min(max(float(prior_probability), 1.0e-4), 1.0 - 1.0e-4)
        nn.init.constant_(self.classifier[-1].bias, math.log(prior / (1.0 - prior)))

    def forward(
        self,
        aligned_residuals: torch.Tensor,
        base: torch.Tensor,
        channel: torch.Tensor,
        patch: torch.Tensor,
    ) -> torch.Tensor:
        state = self.encoder(aligned_residuals, base, channel, patch)
        return self.classifier(state).squeeze(-1)


class TrendCoefficientHead(nn.Module):
    def __init__(
        self,
        history_patches: int = 8,
        hidden: int = 16,
        channel_count: int = 7,
        patch_count: int = 8,
        embedding_dim: int = 4,
        max_abs_coefficient: float = 2.0,
    ) -> None:
        super().__init__()
        if max_abs_coefficient <= 0.0:
            raise ValueError("max_abs_coefficient must be positive")
        self.max_abs_coefficient = float(max_abs_coefficient)
        self.encoder = _TrendStateEncoder(
            history_patches, hidden, channel_count, patch_count, embedding_dim
        )
        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden),
            nn.SiLU(),
        )
        self.mean = nn.Linear(hidden, 1)
        self.log_scale = nn.Linear(hidden, 1)
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.log_scale.weight)
        nn.init.constant_(self.log_scale.bias, -2.0)

    def forward(
        self,
        aligned_residuals: torch.Tensor,
        base: torch.Tensor,
        channel: torch.Tensor,
        patch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.encoder(aligned_residuals, base, channel, patch)
        hidden = self.trunk(state)
        mean = self.max_abs_coefficient * torch.tanh(
            self.mean(hidden).squeeze(-1) / self.max_abs_coefficient
        )
        scale = F.softplus(self.log_scale(hidden).squeeze(-1)) + 1.0e-4
        return mean, scale


class TrendBasisAdapter(nn.Module):
    """Compose disjoint need and signed-coefficient heads."""

    def __init__(
        self,
        history_patches: int = 8,
        hidden: int = 16,
        channel_count: int = 7,
        patch_count: int = 8,
        embedding_dim: int = 4,
        max_abs_coefficient: float = 2.0,
    ) -> None:
        super().__init__()
        common = dict(
            history_patches=history_patches,
            hidden=hidden,
            channel_count=channel_count,
            patch_count=patch_count,
            embedding_dim=embedding_dim,
        )
        self.need_head = TrendNeedHead(**common)
        self.coefficient_head = TrendCoefficientHead(
            **common, max_abs_coefficient=max_abs_coefficient
        )

    def need_parameters(self):
        return self.need_head.parameters()

    def coefficient_parameters(self):
        return self.coefficient_head.parameters()

    def freeze_need(self) -> None:
        self.need_head.requires_grad_(False)
        self.need_head.eval()

    def forward(
        self,
        aligned_residuals: torch.Tensor,
        base: torch.Tensor,
        channel: torch.Tensor,
        patch: torch.Tensor,
        *,
        need_probability_override: torch.Tensor | None = None,
    ) -> TrendAdapterOutput:
        need_logit = self.need_head(aligned_residuals, base, channel, patch)
        need_probability = torch.sigmoid(need_logit)
        if need_probability_override is not None:
            if need_probability_override.shape != need_probability.shape:
                raise ValueError("need_probability_override shape mismatch")
            need_probability = need_probability_override
        coefficient_mean, coefficient_scale = self.coefficient_head(
            aligned_residuals, base, channel, patch
        )
        uncertainty_shrink = 1.0 / (1.0 + coefficient_scale)
        coefficient = need_probability * uncertainty_shrink * coefficient_mean
        basis = zero_mean_linear_basis(
            int(base.shape[-1]), device=base.device, dtype=base.dtype
        )
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


def trend_coefficient_targets(
    base: torch.Tensor, target: torch.Tensor
) -> Dict[str, torch.Tensor]:
    if base.ndim != 2 or target.shape != base.shape or int(base.shape[-1]) < 2:
        raise ValueError("base and target must have shape [B,H>=2]")
    residual = target - base
    basis = zero_mean_linear_basis(
        int(base.shape[-1]), device=base.device, dtype=base.dtype
    )
    ls = (residual * basis).sum(dim=-1) / basis.square().sum().clamp_min(1.0e-12)
    endpoint = (residual[..., -1] - residual[..., 0]) / (
        basis[-1] - basis[0]
    ).clamp_min(1.0e-12)
    return {"least_squares": ls, "endpoint": endpoint}


def decompose_residual_trend(
    base: torch.Tensor, target: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Additively decompose ``target-base`` into Level, Trend and Other."""

    targets = trend_coefficient_targets(base, target)
    basis = zero_mean_linear_basis(
        int(base.shape[-1]), device=base.device, dtype=base.dtype
    )
    component = targets["least_squares"].unsqueeze(-1) * basis.unsqueeze(0)
    residual = target - base
    level = residual.mean(dim=-1, keepdim=True).expand_as(residual)
    other = residual - level - component
    return {
        "residual": residual,
        "level": level,
        "coefficient": targets["least_squares"],
        "component": component,
        "trend": component,
        "other": other,
        "reconstruction": level + component + other,
    }


def trend_adapter_loss(
    base: torch.Tensor,
    target: torch.Tensor,
    output: TrendAdapterOutput,
    *,
    need_coefficient_sq_threshold: float,
    component_weight: float = 1.0,
    coefficient_nll_weight: float = 0.25,
    need_weight: float = 0.10,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if need_coefficient_sq_threshold <= 0.0:
        raise ValueError("need coefficient threshold must be positive")
    decomposition = decompose_residual_trend(base, target)
    coefficient_target = decomposition["coefficient"]
    component_mse = F.mse_loss(output.correction, decomposition["component"])
    need = (
        coefficient_target.square() >= need_coefficient_sq_threshold
    ).to(base.dtype)
    need_bce = F.binary_cross_entropy_with_logits(output.need_logit, need)
    normalized_error = (
        coefficient_target - output.coefficient_mean
    ) / output.coefficient_scale
    coefficient_nll = (
        0.5 * normalized_error.square() + output.coefficient_scale.log()
    ).mean()
    total = (
        component_weight * component_mse
        + coefficient_nll_weight * coefficient_nll
        + need_weight * need_bce
    )
    return total, {
        "component_mse": component_mse,
        "coefficient_nll": coefficient_nll,
        "need_bce": need_bce,
        "need_fraction": need.mean(),
    }


__all__ = [
    "TrendAdapterOutput",
    "TrendBasisAdapter",
    "TrendCoefficientHead",
    "TrendNeedHead",
    "decompose_residual_trend",
    "trend_adapter_loss",
    "trend_coefficient_targets",
    "trend_patch_features",
    "zero_mean_linear_basis",
]
