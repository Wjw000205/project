"""Amp-specific frozen-backbone adapter and conservative OOF utility gate.

This module deliberately does not reuse the Level adapter.  Level correction is
one-dimensional; amp correction must choose a waveform as well as a magnitude.
The candidate model therefore keeps the recent patches as separate temporal
carriers, forms a continuous signed sparse mixture, and predicts a positive
conditional amplitude with heteroscedastic uncertainty.  The correction is
projected onto the zero-level/zero-linear-trend subspace by construction.

It implements the common adapter contract ``prediction = base + correction``;
the internal architecture is intentionally penalty-specific rather than shared
with Level or Trend.

The utility gate is a separate module.  It must be fit on realized gains from
strictly out-of-fold candidates; fitting it on in-sample candidates leaks the
candidate learner's target information into SKIP decisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F


def remove_affine(values: torch.Tensor) -> torch.Tensor:
    """Remove the per-row constant and least-squares linear components."""

    if values.ndim < 2:
        raise ValueError("values must have a horizon dimension")
    horizon = values.shape[-1]
    if horizon < 2:
        return torch.zeros_like(values)
    time = torch.linspace(
        -1.0, 1.0, horizon, dtype=values.dtype, device=values.device
    )
    time = time - time.mean()
    centered = values - values.mean(dim=-1, keepdim=True)
    slope = (centered * time).sum(dim=-1, keepdim=True) / time.square().sum()
    return centered - slope * time


def difference_std(values: torch.Tensor) -> torch.Tensor:
    """Match ``penalty_diff_amp``'s unbiased first-difference deviation."""

    if values.shape[-1] < 3:
        raise ValueError("diff-amp needs a horizon of at least three")
    return torch.diff(values, dim=-1).std(dim=-1)


def normalize_carrier(
    values: torch.Tensor, eps: float = 1.0e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return affine-free unit-diff-amp carriers and an availability mask."""

    projected = remove_affine(values)
    scale = difference_std(projected)
    available = scale > eps
    normalized = projected / scale.clamp_min(eps).unsqueeze(-1)
    normalized = torch.where(available.unsqueeze(-1), normalized, 0.0)
    return normalized, available


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparse probability transform from Martins & Astudillo (2016)."""

    if logits.numel() == 0:
        return logits
    shifted = logits - logits.amax(dim=dim, keepdim=True)
    sorted_logits, _ = torch.sort(shifted, dim=dim, descending=True)
    size = logits.shape[dim]
    rank_shape = [1] * logits.ndim
    rank_shape[dim] = size
    ranks = torch.arange(
        1, size + 1, dtype=logits.dtype, device=logits.device
    ).view(rank_shape)
    cumulative = sorted_logits.cumsum(dim)
    support = 1.0 + ranks * sorted_logits > cumulative
    support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
    threshold = (
        cumulative.gather(dim, support_size - 1) - 1.0
    ) / support_size.to(logits.dtype)
    return torch.clamp(shifted - threshold, min=0.0)


class TemporalCarrierEncoder(nn.Module):
    """Shared local temporal encoder; the horizon is never flattened into context."""

    def __init__(self, horizon: int, hidden_dim: int) -> None:
        super().__init__()
        if horizon < 3:
            raise ValueError("horizon must be at least three")
        self.horizon = int(horizon)
        self.input = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
        self.local = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
        )
        self.position = nn.Parameter(torch.zeros(1, hidden_dim, horizon))
        self.pool_score = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, carrier: torch.Tensor) -> torch.Tensor:
        if carrier.shape[-1] != self.horizon:
            raise ValueError(
                f"carrier horizon {carrier.shape[-1]} != configured {self.horizon}"
            )
        flat = carrier.reshape(-1, 1, self.horizon)
        initial = self.input(flat) + self.position
        encoded = initial + self.local(initial)
        attention = torch.softmax(self.pool_score(encoded), dim=-1)
        pooled = (encoded * attention).sum(dim=-1)
        return self.norm(pooled).reshape(*carrier.shape[:-1], -1)


@dataclass
class AmpCandidateOutput:
    high_need_logit: torch.Tensor
    high_need_probability: torch.Tensor
    carrier_weights: torch.Tensor
    carrier_signs: torch.Tensor
    shape: torch.Tensor
    amplitude: torch.Tensor
    amplitude_scale: torch.Tensor
    uncertainty_shrink: torch.Tensor
    correction: torch.Tensor
    candidate: torch.Tensor


class AmpMixtureAdapter(nn.Module):
    """Generate an amp correction from continuous signed carrier mixtures.

    ``history_carriers`` has shape ``[batch, carrier, horizon]`` and ``base``
    has shape ``[batch, horizon]``.  Channel/patch samples may be folded into
    the batch dimension.  ``context`` contains only causal scalar/cross-channel
    state; the carrier waveforms should not be flattened into it.
    """

    def __init__(
        self,
        context_dim: int,
        horizon: int = 12,
        history_carrier_count: int = 8,
        carrier_hidden_dim: int = 48,
        context_hidden_dim: int = 128,
        trunk_hidden_dim: int = 128,
        minimum_scale: float = 1.0e-4,
        maximum_correction_abs: Optional[float] = None,
        normalize_mixture_shape: bool = True,
    ) -> None:
        super().__init__()
        if context_dim <= 0:
            raise ValueError("context_dim must be positive")
        self.context_dim = int(context_dim)
        self.horizon = int(horizon)
        self.history_carrier_count = int(history_carrier_count)
        self.minimum_scale = float(minimum_scale)
        self.maximum_correction_abs = maximum_correction_abs
        self.normalize_mixture_shape = bool(normalize_mixture_shape)

        # History and frozen-base shapes have different semantics and therefore
        # use separate encoders even though they share the same temporal layout.
        self.history_encoder = TemporalCarrierEncoder(horizon, carrier_hidden_dim)
        self.base_encoder = TemporalCarrierEncoder(horizon, carrier_hidden_dim)
        self.history_lag_embedding = nn.Parameter(
            torch.empty(1, history_carrier_count, carrier_hidden_dim)
        )
        nn.init.normal_(self.history_lag_embedding, mean=0.0, std=0.02)
        # Global lag priors are explicit residual biases; sample-conditioned
        # attention still decides the final continuous mixture.  This avoids
        # asking a shared content encoder to rediscover ordered lag identity.
        self.carrier_weight_bias = nn.Parameter(
            torch.zeros(1, history_carrier_count + 1)
        )
        self.carrier_sign_bias = nn.Parameter(
            torch.zeros(1, history_carrier_count + 1)
        )
        self.need_network = nn.Sequential(
            nn.Linear(context_dim, context_hidden_dim),
            nn.LayerNorm(context_hidden_dim),
            nn.SiLU(),
            nn.Linear(context_hidden_dim, trunk_hidden_dim),
            nn.SiLU(),
            nn.Linear(trunk_hidden_dim, 1),
        )
        self.candidate_context_stem = nn.Sequential(
            nn.Linear(context_dim, context_hidden_dim),
            nn.LayerNorm(context_hidden_dim),
            nn.SiLU(),
        )
        self.context_query = nn.Sequential(
            nn.Linear(context_hidden_dim, carrier_hidden_dim),
            nn.SiLU(),
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
        global_dim = context_hidden_dim + carrier_hidden_dim * 2
        self.global_trunk = nn.Sequential(
            nn.Linear(global_dim, trunk_hidden_dim),
            nn.LayerNorm(trunk_hidden_dim),
            nn.SiLU(),
            nn.Linear(trunk_hidden_dim, trunk_hidden_dim),
            nn.SiLU(),
        )
        self.amplitude_location_head = nn.Linear(trunk_hidden_dim, 1)
        self.amplitude_scale_head = nn.Linear(trunk_hidden_dim, 1)

    def need_parameters(self):
        """Parameters trained on the target-independent full TRAIN population."""

        return self.need_network.parameters()

    def candidate_parameters(self):
        """Parameters trained on the broader high-need candidate population."""

        for name, parameter in self.named_parameters():
            if not name.startswith("need_network."):
                yield parameter

    def predict_need_logit(self, context: torch.Tensor) -> torch.Tensor:
        """Evaluate the parameter-disjoint high-need head only."""

        if context.ndim != 2 or context.shape[-1] != self.context_dim:
            raise ValueError("context must have shape [batch, context_dim]")
        return self.need_network(context).squeeze(-1)

    @staticmethod
    def relative_uncertainty_shrink(
        amplitude: torch.Tensor,
        scale: torch.Tensor,
        eps: float = 1.0e-6,
    ) -> torch.Tensor:
        """Shrink toward zero as uncertainty dominates conditional amplitude."""

        return amplitude / (amplitude + scale + eps)

    def forward(
        self,
        context: torch.Tensor,
        history_carriers: torch.Tensor,
        base: torch.Tensor,
    ) -> AmpCandidateOutput:
        if context.ndim != 2 or context.shape[-1] != self.context_dim:
            raise ValueError("context must have shape [batch, context_dim]")
        if history_carriers.ndim != 3:
            raise ValueError("history_carriers must have shape [batch, carrier, horizon]")
        if base.ndim != 2 or base.shape[-1] != self.horizon:
            raise ValueError("base must have shape [batch, horizon]")
        if history_carriers.shape[0] != base.shape[0] or context.shape[0] != base.shape[0]:
            raise ValueError("batch dimensions do not match")
        if history_carriers.shape[-1] != self.horizon:
            raise ValueError("history carrier horizon does not match base")
        if history_carriers.shape[1] != self.history_carrier_count:
            raise ValueError(
                "history carrier count does not match configured lag embeddings"
            )

        history_unit, history_available = normalize_carrier(history_carriers)
        base_unit, base_available = normalize_carrier(base)
        history_embedding = (
            self.history_encoder(history_unit) + self.history_lag_embedding
        )
        base_embedding = self.base_encoder(base_unit)
        context_global = self.candidate_context_stem(context)
        context_embedding = self.context_query(context_global)
        query = base_embedding + context_embedding

        carrier_unit = torch.cat([history_unit, base_unit.unsqueeze(1)], dim=1)
        carrier_available = torch.cat(
            [history_available, base_available.unsqueeze(1)], dim=1
        )
        carrier_embedding = torch.cat(
            [history_embedding, base_embedding.unsqueeze(1)], dim=1
        )
        expanded_query = query.unsqueeze(1).expand_as(carrier_embedding)
        pair = torch.cat([carrier_embedding, expanded_query], dim=-1)
        logits = self.mixture_score(pair).squeeze(-1) + self.carrier_weight_bias
        logits = logits.masked_fill(~carrier_available, -1.0e9)
        # Soft attention keeps every available carrier trainable.  Sparsity is
        # induced by the explicit entropy loss; a hard sparsemax support here
        # can permanently starve initially unselected lags of gradient.
        weights = torch.softmax(logits, dim=1) * carrier_available.to(logits.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        signs = torch.tanh(
            self.mixture_sign(pair).squeeze(-1) + self.carrier_sign_bias
        )
        signed_weights = weights * signs
        mixed = (signed_weights.unsqueeze(-1) * carrier_unit).sum(dim=1)
        if self.normalize_mixture_shape:
            shape, shape_available = normalize_carrier(mixed)
        else:
            # Preserve the norm of the signed mixture.  Carrier cancellation is
            # target-free directional uncertainty and must be allowed to shrink
            # the emitted correction toward a genuine no-op.
            shape = remove_affine(mixed)
            shape_available = difference_std(shape) > 1.0e-6
        shape = torch.where(shape_available.unsqueeze(-1), shape, 0.0)

        mixture_embedding = (
            weights.unsqueeze(-1) * carrier_embedding
        ).sum(dim=1)
        # Use the pre-projection context representation as well as the base and
        # mixture embeddings.  This lets need/amplitude use broad causal state
        # without destroying the carrier axis used to form the waveform.
        hidden = self.global_trunk(
            torch.cat([context_global, base_embedding, mixture_embedding], dim=-1)
        )
        # This path is parameter-disjoint from candidate generation, so it can
        # be fit on the full population and then frozen exactly as required by
        # the staged amp protocol.
        high_need_logit = self.predict_need_logit(context)
        high_need_probability = torch.sigmoid(high_need_logit)
        amplitude = F.softplus(self.amplitude_location_head(hidden).squeeze(-1))
        amplitude_scale = (
            F.softplus(self.amplitude_scale_head(hidden).squeeze(-1))
            + self.minimum_scale
        )
        uncertainty_shrink = high_need_probability * self.relative_uncertainty_shrink(
            amplitude, amplitude_scale
        )
        correction = uncertainty_shrink.unsqueeze(-1) * amplitude.unsqueeze(-1) * shape
        if self.maximum_correction_abs is not None:
            maximum = correction.abs().amax(dim=-1, keepdim=True)
            clip_scale = torch.clamp(
                float(self.maximum_correction_abs) / maximum.clamp_min(1.0e-8),
                max=1.0,
            )
            correction = correction * clip_scale
        # Reproject after clipping to make the Level-isolation contract explicit
        # even under finite precision.
        correction = remove_affine(correction)
        candidate = base + correction
        return AmpCandidateOutput(
            high_need_logit=high_need_logit,
            high_need_probability=high_need_probability,
            carrier_weights=weights,
            carrier_signs=signs,
            shape=shape,
            amplitude=amplitude,
            amplitude_scale=amplitude_scale,
            uncertainty_shrink=uncertainty_shrink,
            correction=correction,
            candidate=candidate,
        )


def student_t_nll(
    target: torch.Tensor,
    location: torch.Tensor,
    scale: torch.Tensor,
    degrees_of_freedom: float = 3.0,
) -> torch.Tensor:
    """Elementwise robust heteroscedastic negative log likelihood."""

    if degrees_of_freedom <= 2.0:
        raise ValueError("degrees_of_freedom must exceed two")
    df = torch.as_tensor(
        degrees_of_freedom, dtype=target.dtype, device=target.device
    )
    standardized = (target - location) / scale
    constant = (
        torch.lgamma((df + 1.0) / 2.0)
        - torch.lgamma(df / 2.0)
        - 0.5 * torch.log(df * math.pi)
    )
    return -constant + torch.log(scale) + 0.5 * (df + 1.0) * torch.log1p(
        standardized.square() / df
    )


def carrier_entropy(weights: torch.Tensor) -> torch.Tensor:
    """Normalized entropy used as a differentiable sparse-mixture penalty."""

    if weights.shape[-1] <= 1:
        return weights.new_zeros(())
    entropy = -(weights * weights.clamp_min(1.0e-8).log()).sum(dim=-1)
    return (entropy / math.log(weights.shape[-1])).mean()


def amp_candidate_loss(
    output: AmpCandidateOutput,
    target: torch.Tensor,
    high_need_target: torch.Tensor,
    amplitude_target: Optional[torch.Tensor] = None,
    *,
    mse_weight: float = 1.0,
    diff_amp_weight: float = 0.25,
    need_weight: float = 0.25,
    amplitude_nll_weight: float = 0.10,
    sparse_weight: float = 1.0e-3,
    shape_alignment_weight: float = 0.0,
    degrees_of_freedom: float = 3.0,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Joint final-prediction objective without carrier/root class labels."""

    need = high_need_target.to(dtype=torch.bool)
    point_mse = F.mse_loss(output.candidate, target)
    diff_amp = (difference_std(output.candidate) - difference_std(target)).square().mean()
    need_bce = F.binary_cross_entropy_with_logits(
        output.high_need_logit, high_need_target.to(output.high_need_logit.dtype)
    )
    sparse = carrier_entropy(output.carrier_weights)
    base = output.candidate - output.correction
    target_direction = remove_affine(target - base)
    direction_numerator = (output.shape * target_direction).sum(dim=-1)
    direction_denominator = (
        output.shape.square().sum(dim=-1).sqrt()
        * target_direction.square().sum(dim=-1).sqrt()
    )
    direction_valid = direction_denominator > 1.0e-8
    shape_alignment = output.shape.new_zeros(())
    if bool(direction_valid.any().item()):
        shape_alignment = (
            1.0
            - direction_numerator[direction_valid]
            / direction_denominator[direction_valid].clamp_min(1.0e-8)
        ).mean()
    amplitude_nll = output.amplitude.new_zeros(())
    if amplitude_target is not None and bool(need.any().item()):
        amplitude_nll = student_t_nll(
            amplitude_target[need].clamp_min(0.0),
            output.amplitude[need],
            output.amplitude_scale[need],
            degrees_of_freedom,
        ).mean()
    total = (
        mse_weight * point_mse
        + diff_amp_weight * diff_amp
        + need_weight * need_bce
        + amplitude_nll_weight * amplitude_nll
        + sparse_weight * sparse
        + shape_alignment_weight * shape_alignment
    )
    return total, {
        "point_mse": point_mse,
        "diff_amp": diff_amp,
        "need_bce": need_bce,
        "amplitude_nll": amplitude_nll,
        "carrier_entropy": sparse,
        "shape_alignment": shape_alignment,
    }


def realized_candidate_utility(
    base: torch.Tensor, candidate: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Per-sample point-MSE benefit used as the rolling-OOF utility target."""

    return (base - target).square().mean(dim=-1) - (
        candidate - target
    ).square().mean(dim=-1)


def candidate_utility_features(output: AmpCandidateOutput) -> torch.Tensor:
    """Target-free summaries of a frozen OOF candidate for the utility head."""

    correction_rms = output.correction.square().mean(dim=-1).sqrt()
    correction_peak = output.correction.abs().amax(dim=-1)
    entropy = -(
        output.carrier_weights
        * output.carrier_weights.clamp_min(1.0e-8).log()
    ).sum(dim=-1)
    summary = torch.stack(
        [
            output.high_need_probability,
            output.amplitude,
            output.amplitude_scale,
            output.uncertainty_shrink,
            correction_rms,
            correction_peak,
            entropy,
        ],
        dim=-1,
    )
    return torch.cat(
        [summary, output.carrier_weights, output.carrier_signs], dim=-1
    )


@dataclass
class AmpUtilityOutput:
    mean: torch.Tensor
    scale: torch.Tensor
    lower_confidence_bound: torch.Tensor


class AmpUtilityGate(nn.Module):
    """Estimate OOF candidate benefit and expose a conservative lower bound."""

    def __init__(
        self,
        context_dim: int,
        carrier_count: int,
        hidden_dim: int = 128,
        minimum_scale: float = 1.0e-5,
    ) -> None:
        super().__init__()
        self.context_dim = int(context_dim)
        self.carrier_count = int(carrier_count)
        self.minimum_scale = float(minimum_scale)
        candidate_dim = 7 + 2 * carrier_count
        self.network = nn.Sequential(
            nn.Linear(context_dim + candidate_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
        )
        self.mean_head = nn.Linear(hidden_dim // 2, 1)
        self.scale_head = nn.Linear(hidden_dim // 2, 1)

    def forward_features(
        self,
        context: torch.Tensor,
        candidate_features: torch.Tensor,
        *,
        kappa: float = 1.0,
    ) -> AmpUtilityOutput:
        """Evaluate precomputed target-free candidate summaries."""

        expected = 7 + 2 * self.carrier_count
        if candidate_features.ndim != 2 or candidate_features.shape[-1] != expected:
            raise ValueError(f"candidate_features must have width {expected}")
        hidden = self.network(torch.cat([context, candidate_features], dim=-1))
        mean = self.mean_head(hidden).squeeze(-1)
        scale = F.softplus(self.scale_head(hidden).squeeze(-1)) + self.minimum_scale
        lower = mean - float(kappa) * scale
        return AmpUtilityOutput(mean=mean, scale=scale, lower_confidence_bound=lower)

    def forward(
        self,
        context: torch.Tensor,
        candidate: AmpCandidateOutput,
        *,
        kappa: float = 1.0,
        detach_candidate: bool = True,
    ) -> AmpUtilityOutput:
        candidate_features = candidate_utility_features(candidate)
        if detach_candidate:
            candidate_features = candidate_features.detach()
        return self.forward_features(context, candidate_features, kappa=kappa)


def utility_nll_loss(
    output: AmpUtilityOutput, realized_utility: torch.Tensor
) -> torch.Tensor:
    """Gaussian heteroscedastic loss for rolling-OOF realized benefit."""

    standardized = (realized_utility - output.mean) / output.scale
    return (torch.log(output.scale) + 0.5 * standardized.square()).mean()


@dataclass
class AmpDecisionOutput:
    execute: torch.Tensor
    correction: torch.Tensor
    prediction: torch.Tensor


def apply_amp_decision(
    candidate: AmpCandidateOutput,
    utility: AmpUtilityOutput,
    base: torch.Tensor,
    *,
    minimum_need_probability: float = 0.5,
) -> AmpDecisionOutput:
    """Execute only when need is high and the utility lower bound is positive."""

    execute = (
        candidate.high_need_probability >= float(minimum_need_probability)
    ) & (utility.lower_confidence_bound > 0.0)
    correction = candidate.correction * execute.to(base.dtype).unsqueeze(-1)
    return AmpDecisionOutput(
        execute=execute,
        correction=correction,
        prediction=base + correction,
    )
