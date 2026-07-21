"""Full-span Amp adapter with nine direct signed carrier coefficients.

The adapter keeps all eight normalized history carriers plus the normalized
frozen-base carrier until the final residual is constructed.  It therefore
does not collapse the carrier bank to one shape line before amplitude decoding.
Each coefficient is expressed relative to the row's base diff-amplitude.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.models.amp_mixture_adapter import (
    TemporalCarrierEncoder,
    difference_std,
    normalize_carrier,
    remove_affine,
)


@dataclass
class AmpSpanResidualOutput:
    carrier_unit: torch.Tensor
    carrier_available: torch.Tensor
    base_amplitude: torch.Tensor
    base_carrier_covariance: torch.Tensor
    normalized_coefficients: torch.Tensor
    coefficients: torch.Tensor
    correction: torch.Tensor
    candidate: torch.Tensor
    decoded_amplitude: torch.Tensor
    decoded_log_amplitude_ratio: torch.Tensor


def decode_signed_span_correction(
    carrier_unit: torch.Tensor,
    base_amplitude: torch.Tensor,
    normalized_coefficients: torch.Tensor,
    carrier_available: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode a base-scaled coefficient vector without rank-one factorization."""

    if carrier_unit.ndim != 3:
        raise ValueError("carrier_unit must have [batch,carrier,horizon]")
    if normalized_coefficients.shape != carrier_unit.shape[:2]:
        raise ValueError("normalized coefficients must align with carriers")
    if base_amplitude.shape != (carrier_unit.shape[0],):
        raise ValueError("base amplitude must align with rows")
    coefficients = base_amplitude.unsqueeze(-1) * normalized_coefficients
    if carrier_available is not None:
        if carrier_available.shape != coefficients.shape:
            raise ValueError("carrier availability must align with coefficients")
        coefficients = coefficients * carrier_available.to(coefficients.dtype)
    correction = (coefficients.unsqueeze(-1) * carrier_unit).sum(dim=1)
    return coefficients, remove_affine(correction)


class AmpSpanResidualAdapter(nn.Module):
    """Predict all nine continuous signed coefficients directly."""

    def __init__(
        self,
        context_dim: int,
        horizon: int = 12,
        history_carrier_count: int = 8,
        carrier_hidden_dim: int = 48,
        context_hidden_dim: int = 128,
        coefficient_hidden_dim: int = 128,
        maximum_normalized_coefficient: float = 12.0,
        maximum_correction_abs: float | None = None,
    ) -> None:
        super().__init__()
        if context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if maximum_normalized_coefficient <= 0.0:
            raise ValueError("maximum_normalized_coefficient must be positive")
        self.context_dim = int(context_dim)
        self.horizon = int(horizon)
        self.history_carrier_count = int(history_carrier_count)
        self.maximum_normalized_coefficient = float(
            maximum_normalized_coefficient
        )
        self.maximum_correction_abs = maximum_correction_abs
        self.history_encoder = TemporalCarrierEncoder(horizon, carrier_hidden_dim)
        self.base_encoder = TemporalCarrierEncoder(horizon, carrier_hidden_dim)
        self.history_lag_embedding = nn.Parameter(
            torch.empty(1, history_carrier_count, carrier_hidden_dim)
        )
        nn.init.normal_(self.history_lag_embedding, mean=0.0, std=0.02)
        self.base_carrier_embedding = nn.Parameter(
            torch.empty(1, 1, carrier_hidden_dim)
        )
        nn.init.normal_(self.base_carrier_embedding, mean=0.0, std=0.02)
        self.context_stem = nn.Sequential(
            nn.Linear(context_dim, context_hidden_dim),
            nn.LayerNorm(context_hidden_dim),
            nn.SiLU(),
        )
        coefficient_input_dim = (
            context_hidden_dim + 2 * carrier_hidden_dim + 3
        )
        self.coefficient_trunk = nn.Sequential(
            nn.Linear(coefficient_input_dim, coefficient_hidden_dim),
            nn.LayerNorm(coefficient_hidden_dim),
            nn.SiLU(),
            nn.Linear(coefficient_hidden_dim, coefficient_hidden_dim),
            nn.SiLU(),
        )
        self.coefficient_head = nn.Linear(coefficient_hidden_dim, 1)
        nn.init.zeros_(self.coefficient_head.weight)
        nn.init.zeros_(self.coefficient_head.bias)

    def forward(
        self,
        context: torch.Tensor,
        history_carriers: torch.Tensor,
        base: torch.Tensor,
    ) -> AmpSpanResidualOutput:
        if context.ndim != 2 or context.shape[-1] != self.context_dim:
            raise ValueError("context must have shape [batch,context_dim]")
        if history_carriers.ndim != 3:
            raise ValueError("history carriers must have [batch,carrier,horizon]")
        if base.ndim != 2 or base.shape[-1] != self.horizon:
            raise ValueError("base must have shape [batch,horizon]")
        if history_carriers.shape[1:] != (
            self.history_carrier_count,
            self.horizon,
        ):
            raise ValueError("history carrier dimensions do not match")
        if context.shape[0] != base.shape[0] or history_carriers.shape[0] != base.shape[0]:
            raise ValueError("batch dimensions do not match")

        history_unit, history_available = normalize_carrier(history_carriers)
        base_unit, base_available = normalize_carrier(base)
        carrier_unit = torch.cat([history_unit, base_unit.unsqueeze(1)], dim=1)
        carrier_available = torch.cat(
            [history_available, base_available.unsqueeze(1)], dim=1
        )
        history_embedding = (
            self.history_encoder(history_unit) + self.history_lag_embedding
        )
        base_embedding = self.base_encoder(base_unit)
        carrier_embedding = torch.cat(
            [
                history_embedding,
                base_embedding.unsqueeze(1) + self.base_carrier_embedding,
            ],
            dim=1,
        )
        context_embedding = self.context_stem(context)
        carrier_count = carrier_unit.shape[1]
        base_expanded = base_embedding.unsqueeze(1).expand(
            -1, carrier_count, -1
        )
        context_expanded = context_embedding.unsqueeze(1).expand(
            -1, carrier_count, -1
        )
        base_amplitude = difference_std(base)
        base_difference = torch.diff(remove_affine(base), dim=-1)
        base_difference = base_difference - base_difference.mean(
            dim=-1, keepdim=True
        )
        carrier_difference = torch.diff(carrier_unit, dim=-1)
        carrier_difference = carrier_difference - carrier_difference.mean(
            dim=-1, keepdim=True
        )
        covariance = (
            base_difference.unsqueeze(1) * carrier_difference
        ).sum(dim=-1) / float(self.horizon - 2)
        base_rms = remove_affine(base).square().mean(dim=-1).sqrt()
        geometry = torch.stack(
            [
                base_amplitude.unsqueeze(1).expand(-1, carrier_count),
                covariance,
                base_rms.unsqueeze(1).expand(-1, carrier_count),
            ],
            dim=-1,
        )
        hidden = self.coefficient_trunk(
            torch.cat(
                [
                    context_expanded,
                    carrier_embedding,
                    base_expanded,
                    geometry,
                ],
                dim=-1,
            )
        )
        raw = self.coefficient_head(hidden).squeeze(-1)
        maximum = self.maximum_normalized_coefficient
        normalized_coefficients = maximum * torch.tanh(raw / maximum)
        coefficients, correction = decode_signed_span_correction(
            carrier_unit,
            base_amplitude,
            normalized_coefficients,
            carrier_available,
        )
        if self.maximum_correction_abs is not None:
            correction_maximum = correction.abs().amax(dim=-1, keepdim=True)
            correction = correction * torch.clamp(
                float(self.maximum_correction_abs)
                / correction_maximum.clamp_min(1.0e-8),
                max=1.0,
            )
        candidate = base + correction
        decoded_amplitude = difference_std(candidate)
        decoded_log_ratio = torch.log(
            decoded_amplitude.clamp_min(1.0e-8)
            / base_amplitude.clamp_min(1.0e-8)
        )
        return AmpSpanResidualOutput(
            carrier_unit=carrier_unit,
            carrier_available=carrier_available,
            base_amplitude=base_amplitude,
            base_carrier_covariance=covariance,
            normalized_coefficients=normalized_coefficients,
            coefficients=coefficients,
            correction=correction,
            candidate=candidate,
            decoded_amplitude=decoded_amplitude,
            decoded_log_amplitude_ratio=decoded_log_ratio,
        )
