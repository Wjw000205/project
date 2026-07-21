"""Conservative scalar coefficient head for a frozen Amp waveform shape.

The carrier network supplies a target-free unit waveform.  This module does
not recreate or classify carriers: it only predicts the nonnegative distance
to travel along that frozen direction.  Its zero-initialized residual starts
from the frozen coefficient exactly, while the additive parameterization can
move to zero or to coefficients much larger than the frozen value.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class AmpProjectedCoefficientOutput:
    signed_location: torch.Tensor
    coefficient: torch.Tensor
    correction: torch.Tensor
    candidate: torch.Tensor


class AmpProjectedCoefficientHead(nn.Module):
    """Predict a coefficient along an externally frozen unit Amp shape."""

    def __init__(
        self,
        feature_dim: int,
        coefficient_scale: float,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if coefficient_scale <= 0.0:
            raise ValueError("coefficient_scale must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.feature_dim = int(feature_dim)
        self.coefficient_scale = float(coefficient_scale)
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
        )
        self.location_residual = nn.Linear(hidden_dim // 2, 1)
        nn.init.zeros_(self.location_residual.weight)
        nn.init.zeros_(self.location_residual.bias)

    def forward(
        self,
        features: torch.Tensor,
        current_coefficient: torch.Tensor,
        unit_shape: torch.Tensor,
        base: torch.Tensor,
    ) -> AmpProjectedCoefficientOutput:
        if features.ndim != 2 or features.shape[-1] != self.feature_dim:
            raise ValueError("features must have shape [batch, feature_dim]")
        if current_coefficient.shape != (features.shape[0],):
            raise ValueError("current_coefficient must align with features")
        if unit_shape.ndim != 2 or base.shape != unit_shape.shape:
            raise ValueError("base and unit_shape must share [batch,horizon] shape")
        if unit_shape.shape[0] != features.shape[0]:
            raise ValueError("unit_shape must align with features")
        residual = self.location_residual(self.network(features)).squeeze(-1)
        signed_location = current_coefficient + self.coefficient_scale * residual
        coefficient = torch.relu(signed_location)
        correction = coefficient.unsqueeze(-1) * unit_shape
        return AmpProjectedCoefficientOutput(
            signed_location=signed_location,
            coefficient=coefficient,
            correction=correction,
            candidate=base + correction,
        )


def signed_projection_target(
    base: torch.Tensor,
    target: torch.Tensor,
    unit_shape: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the signed sufficient statistic for MSE along ``unit_shape``."""

    if base.ndim != 2 or target.shape != base.shape or unit_shape.shape != base.shape:
        raise ValueError("base, target and unit_shape must share [batch,horizon] shape")
    energy = unit_shape.square().sum(dim=-1)
    coefficient = torch.where(
        energy > 1.0e-12,
        ((target - base) * unit_shape).sum(dim=-1) / energy.clamp_min(1.0e-12),
        torch.zeros_like(energy),
    )
    return coefficient, energy


def projection_coefficient_loss(
    predicted_coefficient: torch.Tensor,
    signed_target: torch.Tensor,
    shape_energy: torch.Tensor,
    coefficient_scale: float,
) -> torch.Tensor:
    """Final-MSE-equivalent coefficient loss, up to one positive constant.

    The signed target is deliberately *not* clipped.  Applying the nonnegative
    constraint to the model output instead of the noisy sample label avoids the
    positive bias from regressing ``max(sample_projection, 0)``.
    """

    if predicted_coefficient.shape != signed_target.shape:
        raise ValueError("predicted and target coefficients must align")
    if shape_energy.shape != signed_target.shape:
        raise ValueError("shape_energy must align with coefficients")
    if coefficient_scale <= 0.0:
        raise ValueError("coefficient_scale must be positive")
    energy_scale = shape_energy.detach().mean().clamp_min(1.0e-12)
    error = (predicted_coefficient - signed_target) / float(coefficient_scale)
    return (error.square() * shape_energy / energy_scale).mean()


def epistemic_reliability_shrink(
    coefficients: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shrink an ensemble mean only by between-model disagreement."""

    if coefficients.ndim != 2 or coefficients.shape[0] < 2:
        raise ValueError("coefficients must have shape [model>=2,batch]")
    mean = coefficients.mean(dim=0)
    variance = coefficients.var(dim=0, unbiased=False)
    signal = mean.square()
    denominator = signal + variance
    gamma = torch.where(
        denominator > 1.0e-16,
        signal / denominator.clamp_min(1.0e-16),
        torch.ones_like(denominator),
    )
    return gamma * mean, gamma, variance.sqrt()
