"""Amp adapter with an explicit differentiable amplitude geometry decoder.

The network predicts a signed carrier mixture and a requested log amplitude
ratio.  It never interprets that ratio as an additive correction coefficient.
Instead, a per-row quadratic decoder solves

    A(base + a * unit_shape)^2 = A(base)^2 + 2*a*cov + a^2

for the signed coefficient with minimum absolute value.  The resulting
correction is constrained to zero level and zero linear trend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from src.models.amp_mixture_adapter import (
    TemporalCarrierEncoder,
    carrier_entropy,
    difference_std,
    normalize_carrier,
    remove_affine,
)


@dataclass
class AmpGeometricOutput:
    carrier_weights: torch.Tensor
    carrier_signs: torch.Tensor
    shape: torch.Tensor
    shape_available: torch.Tensor
    base_amplitude: torch.Tensor
    log_amplitude_ratio: torch.Tensor
    requested_amplitude: torch.Tensor
    base_shape_covariance: torch.Tensor
    discriminant: torch.Tensor
    amplitude_reachable: torch.Tensor
    coefficient: torch.Tensor
    correction: torch.Tensor
    decoded_amplitude: torch.Tensor
    candidate: torch.Tensor


def centered_difference_covariance(
    base: torch.Tensor, unit_shape: torch.Tensor
) -> torch.Tensor:
    """Covariance matching unbiased first-difference amplitude geometry."""

    if base.ndim != 2 or unit_shape.shape != base.shape:
        raise ValueError("base and unit_shape must share [batch,horizon] shape")
    if base.shape[-1] < 3:
        raise ValueError("amplitude geometry needs horizon at least three")
    base_difference = torch.diff(remove_affine(base), dim=-1)
    base_difference = base_difference - base_difference.mean(dim=-1, keepdim=True)
    shape_difference = torch.diff(unit_shape, dim=-1)
    shape_difference = shape_difference - shape_difference.mean(
        dim=-1, keepdim=True
    )
    degrees = float(base.shape[-1] - 2)
    return (base_difference * shape_difference).sum(dim=-1) / degrees


def signed_geometric_amplitude_coefficient(
    base: torch.Tensor,
    unit_shape: torch.Tensor,
    requested_amplitude: torch.Tensor,
    shape_available: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode the minimum-absolute signed quadratic root per row.

    When a requested smaller radius does not intersect the shape line, the
    constrained closest point is the quadratic vertex ``a=-cov``.  A request
    equal to the base amplitude is made an exact no-op.
    """

    if base.ndim != 2 or unit_shape.shape != base.shape:
        raise ValueError("base and unit_shape must share [batch,horizon] shape")
    if requested_amplitude.shape != (base.shape[0],):
        raise ValueError("requested_amplitude must align with rows")
    if shape_available is None:
        shape_available = difference_std(unit_shape) > 1.0e-6
    if shape_available.shape != requested_amplitude.shape:
        raise ValueError("shape_available must align with rows")
    base_amplitude = difference_std(base)
    covariance = centered_difference_covariance(base, unit_shape)
    discriminant = (
        covariance.square()
        + requested_amplitude.square()
        - base_amplitude.square()
    )
    reachable = (discriminant >= -1.0e-8) & shape_available
    root_scale = torch.sqrt(discriminant.clamp_min(1.0e-12))
    # Rationalized minimum-absolute root.  In contrast to selecting between
    # ``-cov +/- sqrt``, this is exactly zero and retains a useful gradient when
    # requested_amplitude equals base_amplitude.
    radial_delta = requested_amplitude.square() - base_amplitude.square()
    covariance_sign = torch.where(
        covariance >= 0.0,
        torch.ones_like(covariance),
        -torch.ones_like(covariance),
    )
    denominator = covariance + covariance_sign * root_scale
    safe_denominator = covariance_sign * denominator.abs().clamp_min(1.0e-6)
    root = radial_delta / safe_denominator
    vertex = -covariance
    coefficient = torch.where(reachable, root, vertex)
    coefficient = torch.where(
        shape_available, coefficient, torch.zeros_like(coefficient)
    )
    return coefficient, covariance, discriminant, reachable


class AmpGeometricAdapter(nn.Module):
    """Predict waveform shape and decode its coefficient in amplitude space."""

    def __init__(
        self,
        context_dim: int,
        horizon: int = 12,
        history_carrier_count: int = 8,
        carrier_hidden_dim: int = 48,
        context_hidden_dim: int = 128,
        trunk_hidden_dim: int = 128,
        maximum_log_amplitude_ratio: float = 6.0,
        maximum_correction_abs: Optional[float] = None,
    ) -> None:
        super().__init__()
        if context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if maximum_log_amplitude_ratio <= 0.0:
            raise ValueError("maximum_log_amplitude_ratio must be positive")
        self.context_dim = int(context_dim)
        self.horizon = int(horizon)
        self.history_carrier_count = int(history_carrier_count)
        self.maximum_log_amplitude_ratio = float(maximum_log_amplitude_ratio)
        self.maximum_correction_abs = maximum_correction_abs
        self.history_encoder = TemporalCarrierEncoder(horizon, carrier_hidden_dim)
        self.base_encoder = TemporalCarrierEncoder(horizon, carrier_hidden_dim)
        self.history_lag_embedding = nn.Parameter(
            torch.empty(1, history_carrier_count, carrier_hidden_dim)
        )
        nn.init.normal_(self.history_lag_embedding, mean=0.0, std=0.02)
        self.carrier_weight_bias = nn.Parameter(
            torch.zeros(1, history_carrier_count + 1)
        )
        self.carrier_sign_bias = nn.Parameter(
            torch.zeros(1, history_carrier_count + 1)
        )
        self.context_stem = nn.Sequential(
            nn.Linear(context_dim, context_hidden_dim),
            nn.LayerNorm(context_hidden_dim),
            nn.SiLU(),
        )
        self.context_query = nn.Sequential(
            nn.Linear(context_hidden_dim, carrier_hidden_dim), nn.SiLU()
        )
        pair_dim = carrier_hidden_dim * 2
        self.mixture_score = nn.Sequential(
            nn.Linear(pair_dim, carrier_hidden_dim),
            nn.SiLU(),
            nn.Linear(carrier_hidden_dim, 1),
        )
        self.mixture_sign = nn.Sequential(
            nn.Linear(pair_dim, carrier_hidden_dim),
            nn.SiLU(),
            nn.Linear(carrier_hidden_dim, 1),
        )
        global_dim = context_hidden_dim + carrier_hidden_dim * 2 + 3
        self.global_trunk = nn.Sequential(
            nn.Linear(global_dim, trunk_hidden_dim),
            nn.LayerNorm(trunk_hidden_dim),
            nn.SiLU(),
            nn.Linear(trunk_hidden_dim, trunk_hidden_dim),
            nn.SiLU(),
        )
        self.log_amplitude_ratio_head = nn.Linear(trunk_hidden_dim, 1)
        nn.init.zeros_(self.log_amplitude_ratio_head.weight)
        nn.init.zeros_(self.log_amplitude_ratio_head.bias)

    def forward(
        self,
        context: torch.Tensor,
        history_carriers: torch.Tensor,
        base: torch.Tensor,
    ) -> AmpGeometricOutput:
        if context.ndim != 2 or context.shape[-1] != self.context_dim:
            raise ValueError("context must have shape [batch,context_dim]")
        if history_carriers.ndim != 3:
            raise ValueError("history_carriers must have [batch,carrier,horizon]")
        if base.ndim != 2 or base.shape[-1] != self.horizon:
            raise ValueError("base must have shape [batch,horizon]")
        if context.shape[0] != base.shape[0] or history_carriers.shape[0] != base.shape[0]:
            raise ValueError("batch dimensions do not match")
        if history_carriers.shape[1] != self.history_carrier_count:
            raise ValueError("history carrier count does not match")
        if history_carriers.shape[-1] != self.horizon:
            raise ValueError("history horizon does not match")
        history_unit, history_available = normalize_carrier(history_carriers)
        base_unit, base_available = normalize_carrier(base)
        history_embedding = (
            self.history_encoder(history_unit) + self.history_lag_embedding
        )
        base_embedding = self.base_encoder(base_unit)
        carrier_unit = torch.cat([history_unit, base_unit.unsqueeze(1)], dim=1)
        carrier_available = torch.cat(
            [history_available, base_available.unsqueeze(1)], dim=1
        )
        carrier_embedding = torch.cat(
            [history_embedding, base_embedding.unsqueeze(1)], dim=1
        )
        context_global = self.context_stem(context)
        query = (
            base_embedding + self.context_query(context_global)
        ).unsqueeze(1).expand_as(carrier_embedding)
        pair = torch.cat([carrier_embedding, query], dim=-1)
        logits = self.mixture_score(pair).squeeze(-1) + self.carrier_weight_bias
        logits = logits.masked_fill(~carrier_available, -1.0e9)
        weights = torch.softmax(logits, dim=1) * carrier_available.to(logits.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        signs = torch.tanh(
            self.mixture_sign(pair).squeeze(-1) + self.carrier_sign_bias
        )
        mixed = ((weights * signs).unsqueeze(-1) * carrier_unit).sum(dim=1)
        shape, shape_available = normalize_carrier(mixed)
        shape = torch.where(shape_available.unsqueeze(-1), shape, 0.0)
        mixture_embedding = (weights.unsqueeze(-1) * carrier_embedding).sum(dim=1)
        base_amplitude = difference_std(base)
        covariance = centered_difference_covariance(base, shape)
        base_shape_rms = remove_affine(base).square().mean(dim=-1).sqrt()
        geometry = torch.stack(
            [base_amplitude, covariance, base_shape_rms], dim=-1
        )
        hidden = self.global_trunk(
            torch.cat(
                [context_global, base_embedding, mixture_embedding, geometry],
                dim=-1,
            )
        )
        raw_log_ratio = self.log_amplitude_ratio_head(hidden).squeeze(-1)
        log_ratio = self.maximum_log_amplitude_ratio * torch.tanh(
            raw_log_ratio / self.maximum_log_amplitude_ratio
        )
        requested_amplitude = base_amplitude * torch.exp(log_ratio)
        coefficient, covariance, discriminant, reachable = (
            signed_geometric_amplitude_coefficient(
                base,
                shape,
                requested_amplitude,
                shape_available,
            )
        )
        correction = coefficient.unsqueeze(-1) * shape
        if self.maximum_correction_abs is not None:
            maximum = correction.abs().amax(dim=-1, keepdim=True)
            clip_scale = torch.clamp(
                float(self.maximum_correction_abs) / maximum.clamp_min(1.0e-8),
                max=1.0,
            )
            correction = correction * clip_scale
        correction = remove_affine(correction)
        candidate = base + correction
        return AmpGeometricOutput(
            carrier_weights=weights,
            carrier_signs=signs,
            shape=shape,
            shape_available=shape_available,
            base_amplitude=base_amplitude,
            log_amplitude_ratio=log_ratio,
            requested_amplitude=requested_amplitude,
            base_shape_covariance=covariance,
            discriminant=discriminant,
            amplitude_reachable=reachable,
            coefficient=coefficient,
            correction=correction,
            decoded_amplitude=difference_std(candidate),
            candidate=candidate,
        )


def amp_geometric_loss(
    output: AmpGeometricOutput,
    target: torch.Tensor,
    point_scale: float,
    amplitude_scale: float,
    sparse_weight: float = 1.0e-3,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Equal relative-error point and Amp objective on the final prediction."""

    if point_scale <= 0.0 or amplitude_scale <= 0.0:
        raise ValueError("loss scales must be positive")
    point_mse = (output.candidate - target).square().mean()
    amplitude_mse = (
        difference_std(output.candidate) - difference_std(target)
    ).square().mean()
    sparse = carrier_entropy(output.carrier_weights)
    total = point_mse / float(point_scale) + amplitude_mse / float(
        amplitude_scale
    ) + sparse_weight * sparse
    return total, {
        "point_mse": point_mse,
        "amplitude_mse": amplitude_mse,
        "carrier_entropy": sparse,
    }
