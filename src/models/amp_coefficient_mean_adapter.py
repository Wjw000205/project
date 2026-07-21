"""Deterministic amp conditional-mean adapter over causal waveform carriers.

Unlike ``AmpMixtureAdapter``, this module has no unit-normalized mixed shape,
positive point-amplitude head, q-amplitude target, or future-need multiplier.
It predicts an unconstrained signed coefficient mean for each normalized causal
carrier.  The zero-initialized coefficient output makes the initial point path
an exact no-op; coefficient cancellation and magnitude remain meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from src.models.amp_mixture_adapter import (
    TemporalCarrierEncoder,
    normalize_carrier,
    remove_affine,
)


@dataclass
class AmpCoefficientMeanOutput:
    high_need_logit: torch.Tensor
    high_need_probability: torch.Tensor
    coefficients: torch.Tensor
    carrier_weights: torch.Tensor
    carrier_signs: torch.Tensor
    correction: torch.Tensor
    candidate: torch.Tensor


class AmpCoefficientMeanAdapter(nn.Module):
    """Predict the point conditional mean as signed causal-carrier coefficients."""

    def __init__(
        self,
        context_dim: int,
        horizon: int = 12,
        history_carrier_count: int = 8,
        carrier_hidden_dim: int = 48,
        context_hidden_dim: int = 128,
        coefficient_hidden_dim: int = 96,
        maximum_correction_abs: Optional[float] = None,
    ) -> None:
        super().__init__()
        if context_dim <= 0:
            raise ValueError("context_dim must be positive")
        self.context_dim = int(context_dim)
        self.horizon = int(horizon)
        self.history_carrier_count = int(history_carrier_count)
        self.maximum_correction_abs = maximum_correction_abs
        self.history_encoder = TemporalCarrierEncoder(horizon, carrier_hidden_dim)
        self.base_encoder = TemporalCarrierEncoder(horizon, carrier_hidden_dim)
        self.history_lag_embedding = nn.Parameter(
            torch.empty(1, history_carrier_count, carrier_hidden_dim)
        )
        nn.init.normal_(self.history_lag_embedding, mean=0.0, std=0.02)
        self.need_network = nn.Sequential(
            nn.Linear(context_dim, context_hidden_dim),
            nn.LayerNorm(context_hidden_dim),
            nn.SiLU(),
            nn.Linear(context_hidden_dim, context_hidden_dim),
            nn.SiLU(),
            nn.Linear(context_hidden_dim, 1),
        )
        self.point_context = nn.Sequential(
            nn.Linear(context_dim, context_hidden_dim),
            nn.LayerNorm(context_hidden_dim),
            nn.SiLU(),
            nn.Linear(context_hidden_dim, carrier_hidden_dim),
            nn.SiLU(),
        )
        pair_dim = carrier_hidden_dim * 2
        self.coefficient_hidden = nn.Sequential(
            nn.Linear(pair_dim, coefficient_hidden_dim),
            nn.LayerNorm(coefficient_hidden_dim),
            nn.SiLU(),
            nn.Linear(coefficient_hidden_dim, coefficient_hidden_dim),
            nn.SiLU(),
        )
        self.coefficient_output = nn.Linear(coefficient_hidden_dim, 1)
        nn.init.zeros_(self.coefficient_output.weight)
        nn.init.zeros_(self.coefficient_output.bias)

    def need_parameters(self):
        return self.need_network.parameters()

    def point_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith("need_network."):
                yield parameter

    def predict_need_logit(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 2 or context.shape[-1] != self.context_dim:
            raise ValueError("context must have shape [batch, context_dim]")
        return self.need_network(context).squeeze(-1)

    def forward(
        self,
        context: torch.Tensor,
        history_carriers: torch.Tensor,
        base: torch.Tensor,
    ) -> AmpCoefficientMeanOutput:
        if context.ndim != 2 or context.shape[-1] != self.context_dim:
            raise ValueError("context must have shape [batch, context_dim]")
        if history_carriers.ndim != 3:
            raise ValueError("history_carriers must have shape [batch, carrier, horizon]")
        if base.ndim != 2 or base.shape[-1] != self.horizon:
            raise ValueError("base must have shape [batch, horizon]")
        if history_carriers.shape[0] != base.shape[0] or context.shape[0] != base.shape[0]:
            raise ValueError("batch dimensions do not match")
        if history_carriers.shape[1] != self.history_carrier_count:
            raise ValueError("history carrier count does not match configured lag embeddings")
        if history_carriers.shape[-1] != self.horizon:
            raise ValueError("history carrier horizon does not match base")

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
        query = self.point_context(context).unsqueeze(1).expand_as(carrier_embedding)
        hidden = self.coefficient_hidden(torch.cat([carrier_embedding, query], dim=-1))
        coefficients = self.coefficient_output(hidden).squeeze(-1)
        coefficients = coefficients * carrier_available.to(coefficients.dtype)
        correction = (coefficients.unsqueeze(-1) * carrier_unit).sum(dim=1)
        correction = remove_affine(correction)
        if self.maximum_correction_abs is not None:
            maximum = correction.abs().amax(dim=-1, keepdim=True)
            clip_scale = torch.clamp(
                float(self.maximum_correction_abs) / maximum.clamp_min(1.0e-8),
                max=1.0,
            )
            correction = remove_affine(correction * clip_scale)
        candidate = base + correction
        absolute = coefficients.abs()
        denominator = absolute.sum(dim=1, keepdim=True)
        uniform = carrier_available.to(coefficients.dtype)
        uniform = uniform / uniform.sum(dim=1, keepdim=True).clamp_min(1.0)
        weights = torch.where(
            denominator > 1.0e-8,
            absolute / denominator.clamp_min(1.0e-8),
            uniform,
        )
        need_logit = self.predict_need_logit(context)
        return AmpCoefficientMeanOutput(
            high_need_logit=need_logit,
            high_need_probability=torch.sigmoid(need_logit),
            coefficients=coefficients,
            carrier_weights=weights,
            carrier_signs=torch.tanh(coefficients),
            correction=correction,
            candidate=candidate,
        )
