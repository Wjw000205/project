"""Parameter-disjoint two-stage Amp shape and geometric-amplitude adapter."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.models.amp_geometric_adapter import (
    AmpGeometricAdapter,
    AmpGeometricOutput,
    centered_difference_covariance,
    signed_geometric_amplitude_coefficient,
)
from src.models.amp_mixture_adapter import (
    carrier_entropy,
    difference_std,
    normalize_carrier,
    remove_affine,
)


@dataclass
class AmpShapeProjection:
    coefficient: torch.Tensor
    correction: torch.Tensor
    candidate: torch.Tensor
    energy: torch.Tensor


class AmpGeometricTwoStageAdapter(AmpGeometricAdapter):
    """Train waveform direction before the independent amplitude coordinate."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        first = self.context_stem[0]
        context_hidden_dim = int(first.out_features)
        self.amplitude_context_stem = nn.Sequential(
            nn.Linear(self.context_dim, context_hidden_dim),
            nn.LayerNorm(context_hidden_dim),
            nn.SiLU(),
        )

    def shape_parameters(self):
        amplitude_prefixes = (
            "amplitude_context_stem.",
            "global_trunk.",
            "log_amplitude_ratio_head.",
        )
        for name, parameter in self.named_parameters():
            if not name.startswith(amplitude_prefixes):
                yield parameter

    def amplitude_parameters(self):
        amplitude_prefixes = (
            "amplitude_context_stem.",
            "global_trunk.",
            "log_amplitude_ratio_head.",
        )
        for name, parameter in self.named_parameters():
            if name.startswith(amplitude_prefixes):
                yield parameter

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
        shape_context = self.context_stem(context)
        query = (
            base_embedding + self.context_query(shape_context)
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
        amplitude_context = self.amplitude_context_stem(context)
        # Amplitude is intentionally unable to update carrier direction.
        hidden = self.global_trunk(
            torch.cat(
                [
                    amplitude_context,
                    base_embedding.detach(),
                    mixture_embedding.detach(),
                    geometry.detach(),
                ],
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
                base, shape, requested_amplitude, shape_available
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


def shape_variable_projection(
    output: AmpGeometricOutput,
    base: torch.Tensor,
    target: torch.Tensor,
) -> AmpShapeProjection:
    """Best continuous training-time MSE projection along predicted shape."""

    if base.shape != target.shape or output.shape.shape != base.shape:
        raise ValueError("shape projection inputs must align")
    residual = target - base
    energy = output.shape.square().sum(dim=-1)
    coefficient = torch.where(
        energy > 1.0e-12,
        (residual * output.shape).sum(dim=-1) / energy.clamp_min(1.0e-12),
        torch.zeros_like(energy),
    )
    correction = coefficient.unsqueeze(-1) * output.shape
    return AmpShapeProjection(
        coefficient=coefficient,
        correction=correction,
        candidate=base + correction,
        energy=energy,
    )


def amp_shape_projection_loss(
    output: AmpGeometricOutput,
    base: torch.Tensor,
    target: torch.Tensor,
    point_scale: float,
    sparse_weight: float = 1.0e-3,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train only which continuous carrier mixture captures point residual."""

    if point_scale <= 0.0:
        raise ValueError("point_scale must be positive")
    projection = shape_variable_projection(output, base, target)
    point_mse = (projection.candidate - target).square().mean()
    sparse = carrier_entropy(output.carrier_weights)
    total = point_mse / float(point_scale) + sparse_weight * sparse
    return total, {
        "projection_point_mse": point_mse,
        "carrier_entropy": sparse,
        "projection_coefficient_abs_mean": projection.coefficient.abs().mean(),
    }
