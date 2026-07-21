"""Amp carrier-mixture adapter with a directly learned signed residual radius.

Unlike :mod:`amp_geometric_adapter`, this module does not assume that a
requested forecast amplitude identifies one of the two quadratic roots.  It
predicts a continuous signed coefficient in units of the base first-difference
amplitude instead:

    correction = A(base) * z(context, base, shape) * unit_shape

``z`` is bounded, zero initialized, and may have either sign.  Consequently
the coefficient is multiplicative in the row's residual geometry rather than
an equal additive amount across rows.  The resulting amplitude is measured
after constructing the candidate; it is not used as a root label.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch

from src.models.amp_geometric_adapter import centered_difference_covariance
from src.models.amp_geometric_two_stage_adapter import AmpGeometricTwoStageAdapter
from src.models.amp_mixture_adapter import (
    difference_std,
    normalize_carrier,
    remove_affine,
)


@dataclass
class AmpSignedResidualOutput:
    carrier_weights: torch.Tensor
    carrier_signs: torch.Tensor
    shape: torch.Tensor
    shape_available: torch.Tensor
    base_amplitude: torch.Tensor
    base_shape_covariance: torch.Tensor
    normalized_coefficient: torch.Tensor
    coefficient: torch.Tensor
    correction: torch.Tensor
    decoded_amplitude: torch.Tensor
    decoded_log_amplitude_ratio: torch.Tensor
    candidate: torch.Tensor


def conservative_signed_ensemble_coefficient(
    coefficients: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ensemble mean, epistemic sigma and a one-sigma signed LCB.

    ``coefficients`` has shape ``[member, ...]``.  The lower bound preserves
    the ensemble-mean sign and shrinks its magnitude toward exact no-op; it
    never reverses an action.
    """

    if coefficients.ndim < 1 or coefficients.shape[0] < 2:
        raise ValueError("at least two ensemble members are required")
    mean = coefficients.mean(dim=0)
    sigma = coefficients.std(dim=0, unbiased=False)
    lower_bound = torch.sign(mean) * torch.relu(mean.abs() - sigma)
    return mean, sigma, lower_bound


def amp_non_regression_hinge_loss(
    candidate: torch.Tensor,
    base: torch.Tensor,
    target: torch.Tensor,
    point_scale: float,
    amplitude_scale: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Final-MSE loss plus a one-sided per-row Amp non-regression constraint."""

    if candidate.shape != base.shape or target.shape != base.shape:
        raise ValueError("candidate, base and target must align")
    if point_scale <= 0.0 or amplitude_scale <= 0.0:
        raise ValueError("loss scales must be positive")
    point_mse = (candidate - target).square().mean()
    target_amplitude = difference_std(target)
    candidate_error = (difference_std(candidate) - target_amplitude).square()
    base_error = (difference_std(base) - target_amplitude).square()
    amp_hinge = torch.relu(candidate_error - base_error).mean()
    total = point_mse / float(point_scale) + amp_hinge / float(amplitude_scale)
    return total, {
        "point_mse": point_mse,
        "amp_non_regression_hinge": amp_hinge,
        "candidate_amplitude_mse": candidate_error.mean(),
        "base_amplitude_mse": base_error.mean(),
    }


class AmpSignedResidualAdapter(AmpGeometricTwoStageAdapter):
    """Predict a signed, base-amplitude-scaled coefficient on a carrier shape."""

    def __init__(
        self,
        *args,
        maximum_normalized_coefficient: float = 12.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if maximum_normalized_coefficient <= 0.0:
            raise ValueError("maximum_normalized_coefficient must be positive")
        self.maximum_normalized_coefficient = float(
            maximum_normalized_coefficient
        )

    def forward(
        self,
        context: torch.Tensor,
        history_carriers: torch.Tensor,
        base: torch.Tensor,
    ) -> AmpSignedResidualOutput:
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
        raw = self.log_amplitude_ratio_head(hidden).squeeze(-1)
        maximum = self.maximum_normalized_coefficient
        normalized_coefficient = maximum * torch.tanh(raw / maximum)
        coefficient = base_amplitude * normalized_coefficient
        coefficient = torch.where(
            shape_available, coefficient, torch.zeros_like(coefficient)
        )
        correction = remove_affine(coefficient.unsqueeze(-1) * shape)
        if self.maximum_correction_abs is not None:
            correction_maximum = correction.abs().amax(dim=-1, keepdim=True)
            clip_scale = torch.clamp(
                float(self.maximum_correction_abs)
                / correction_maximum.clamp_min(1.0e-8),
                max=1.0,
            )
            correction = correction * clip_scale
        candidate = base + correction
        decoded_amplitude = difference_std(candidate)
        decoded_log_ratio = torch.log(
            decoded_amplitude.clamp_min(1.0e-8)
            / base_amplitude.clamp_min(1.0e-8)
        )
        return AmpSignedResidualOutput(
            carrier_weights=weights,
            carrier_signs=signs,
            shape=shape,
            shape_available=shape_available,
            base_amplitude=base_amplitude,
            base_shape_covariance=covariance,
            normalized_coefficient=normalized_coefficient,
            coefficient=coefficient,
            correction=correction,
            decoded_amplitude=decoded_amplitude,
            decoded_log_amplitude_ratio=decoded_log_ratio,
            candidate=candidate,
        )
