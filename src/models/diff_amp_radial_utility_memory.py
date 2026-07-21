"""Isolated radial utility-memory hurdle adapter for DiffAmp research.

This module deliberately does not participate in any production or universal
adapter registry.  The learned action is a signed scalar times the current
target-free DiffAmp carrier, so both its input target and its output remain in
the named DiffAmp residual space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F


HORIZON = 96
MEMORY_COUNT = 28
PERIOD = 96
QUERY_WIDTH = 28
KEY_WIDTH = 29
LATENT_WIDTH = 16
CONTEXT_WIDTH = 36


@dataclass(frozen=True)
class DiffAmpRadialOutput:
    """All quantities emitted by one radial hurdle forward pass."""

    action: torch.Tensor
    action_scalar: torch.Tensor
    support_logit: torch.Tensor
    support_probability: torch.Tensor
    coefficient_raw: torch.Tensor
    coefficient: torch.Tensor
    attention: torch.Tensor
    retrieval_stats: torch.Tensor


class DiffAmpRadialUtilityMemoryHurdleAdapter(nn.Module):
    """One fixed 2,706-parameter DiffAmp-only radial memory kernel.

    ``memory_support`` and ``memory_beta_normalized`` are source-local memory
    values.  They affect only the four analytic retrieval statistics; the
    learned memory key is target-free.  At execution the output is always
    ``scalar * current_carrier`` and therefore cannot leave the current
    DiffAmp carrier line.
    """

    def __init__(self) -> None:
        super().__init__()
        self.query_encoder = nn.Sequential(
            nn.Linear(QUERY_WIDTH, LATENT_WIDTH),
            nn.SiLU(),
            nn.Linear(LATENT_WIDTH, LATENT_WIDTH),
            nn.SiLU(),
        )
        self.key_encoder = nn.Sequential(
            nn.Linear(KEY_WIDTH, LATENT_WIDTH),
            nn.SiLU(),
            nn.Linear(LATENT_WIDTH, LATENT_WIDTH),
            nn.SiLU(),
        )
        self.support_head = nn.Sequential(
            nn.Linear(CONTEXT_WIDTH, LATENT_WIDTH),
            nn.SiLU(),
            nn.Linear(LATENT_WIDTH, 1),
        )
        self.coefficient_head = nn.Sequential(
            nn.Linear(CONTEXT_WIDTH, LATENT_WIDTH),
            nn.SiLU(),
            nn.Linear(LATENT_WIDTH, 1),
        )
        support_final = self.support_head[-1]
        coefficient_final = self.coefficient_head[-1]
        assert isinstance(support_final, nn.Linear)
        assert isinstance(coefficient_final, nn.Linear)
        nn.init.zeros_(support_final.weight)
        nn.init.constant_(support_final.bias, -1.0)
        nn.init.zeros_(coefficient_final.weight)
        nn.init.zeros_(coefficient_final.bias)

    def forward(
        self,
        query: torch.Tensor,
        memory_key: torch.Tensor,
        memory_support: torch.Tensor,
        memory_beta_normalized: torch.Tensor,
        current_carrier: torch.Tensor,
        beta_scale: torch.Tensor | float,
    ) -> DiffAmpRadialOutput:
        if query.ndim != 2 or query.shape[1] != QUERY_WIDTH:
            raise ValueError("query must have shape [N,28]")
        batch = query.shape[0]
        if memory_key.shape != (batch, MEMORY_COUNT, KEY_WIDTH):
            raise ValueError("memory_key must have shape [N,28,29]")
        if memory_support.shape != (batch, MEMORY_COUNT):
            raise ValueError("memory_support must have shape [N,28]")
        if memory_beta_normalized.shape != (batch, MEMORY_COUNT):
            raise ValueError("memory_beta_normalized must have shape [N,28]")
        if current_carrier.shape != (batch, HORIZON):
            raise ValueError("current_carrier must have shape [N,96]")

        query_latent = self.query_encoder(query)
        key_latent = self.key_encoder(memory_key)
        attention_logits = torch.einsum("nd,nkd->nk", query_latent, key_latent) / 4.0
        attention = torch.softmax(attention_logits, dim=1)
        key_context = torch.einsum("nk,nkd->nd", attention, key_latent)

        active_weight = attention * memory_support.to(attention)
        active_mass = active_weight.sum(dim=1)
        safe_mass = active_mass.clamp_min(1.0e-12)
        beta_mean = (active_weight * memory_beta_normalized).sum(dim=1) / safe_mass
        beta_centered = memory_beta_normalized - beta_mean.unsqueeze(1)
        beta_variance = (
            active_weight * beta_centered.square()
        ).sum(dim=1) / safe_mass
        # ``sqrt(0)`` has an infinite derivative and can turn the query/key
        # gradient into NaN for none/one/equal active memories.  The epsilon
        # keeps that path finite; the no-active branch is still exact zero.
        beta_std = (beta_variance.clamp_min(0.0) + 1.0e-12).sqrt()
        beta_abs_mean = (
            active_weight * memory_beta_normalized.abs()
        ).sum(dim=1) / safe_mass
        has_active = active_mass > 0.0
        beta_mean = torch.where(has_active, beta_mean, torch.zeros_like(beta_mean))
        beta_std = torch.where(has_active, beta_std, torch.zeros_like(beta_std))
        beta_abs_mean = torch.where(
            has_active, beta_abs_mean, torch.zeros_like(beta_abs_mean)
        )
        retrieval_stats = torch.stack(
            [active_mass, beta_mean, beta_std, beta_abs_mean], dim=1
        )
        context = torch.cat([query_latent, key_context, retrieval_stats], dim=1)
        if context.shape != (batch, CONTEXT_WIDTH):
            raise RuntimeError("DiffAmp radial context contract drift")

        support_logit = self.support_head(context).squeeze(1)
        coefficient_raw = self.coefficient_head(context).squeeze(1)
        support_probability = torch.sigmoid(support_logit)
        scale = torch.as_tensor(
            beta_scale, device=coefficient_raw.device, dtype=coefficient_raw.dtype
        )
        if scale.numel() != 1:
            raise ValueError("beta_scale must be scalar")
        coefficient = scale.reshape(()) * coefficient_raw
        action_scalar = support_probability * coefficient
        action = action_scalar.unsqueeze(1) * current_carrier
        return DiffAmpRadialOutput(
            action=action,
            action_scalar=action_scalar,
            support_logit=support_logit,
            support_probability=support_probability,
            coefficient_raw=coefficient_raw,
            coefficient=coefficient,
            attention=attention,
            retrieval_stats=retrieval_stats,
        )


@dataclass(frozen=True)
class DiffAmpRadialLoss:
    total: torch.Tensor
    support_bce: torch.Tensor
    active_coefficient_mse: torch.Tensor


def diff_amp_radial_utility_loss(
    output: DiffAmpRadialOutput,
    target_support: torch.Tensor,
    target_beta_normalized: torch.Tensor,
    carrier_energy: torch.Tensor,
) -> DiffAmpRadialLoss:
    """Fixed unweighted hurdle loss, with no point-risk optimizer term."""

    if target_support.shape != output.support_logit.shape:
        raise ValueError("target_support shape drift")
    if target_beta_normalized.shape != output.coefficient_raw.shape:
        raise ValueError("target_beta_normalized shape drift")
    if carrier_energy.shape != output.coefficient_raw.shape:
        raise ValueError("carrier_energy shape drift")
    support = target_support.to(output.support_logit)
    support_bce = F.binary_cross_entropy_with_logits(
        output.support_logit, support, reduction="mean"
    )
    active_energy = support * carrier_energy.to(output.coefficient_raw).clamp_min(0.0)
    denominator = active_energy.sum()
    numerator = (
        active_energy
        * (output.coefficient_raw - target_beta_normalized.to(output.coefficient_raw)).square()
    ).sum()
    active_coefficient_mse = torch.where(
        denominator > 0.0,
        numerator / denominator.clamp_min(1.0e-12),
        numerator * 0.0,
    )
    return DiffAmpRadialLoss(
        total=support_bce + active_coefficient_mse,
        support_bce=support_bce,
        active_coefficient_mse=active_coefficient_mse,
    )


@torch.no_grad()
def radial_beta_and_support(
    carrier: torch.Tensor,
    target_component: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the source-local radial coefficient, support, and carrier SSE."""

    if carrier.ndim != 2 or carrier.shape[1] != HORIZON:
        raise ValueError("carrier must have shape [N,96]")
    if target_component.shape != carrier.shape:
        raise ValueError("target_component must match carrier")
    carrier_energy = carrier.square().sum(dim=1)
    beta = (carrier * target_component).sum(dim=1) / carrier_energy.clamp_min(1.0e-12)
    valid_carrier = carrier_energy > 1.0e-12
    beta = torch.where(valid_carrier, beta, torch.zeros_like(beta))
    radial_energy = beta.square() * carrier_energy
    support = valid_carrier & (radial_energy > 1.0e-16)
    beta = torch.where(support, beta, torch.zeros_like(beta))
    return beta, support, carrier_energy


def causal_memory_origins(query_origins: torch.Tensor) -> torch.Tensor:
    """Return the exact 28 matured physical-period source origins."""

    if query_origins.ndim != 1:
        raise ValueError("query_origins must be one-dimensional")
    ages = torch.arange(
        1, MEMORY_COUNT + 1, device=query_origins.device, dtype=query_origins.dtype
    )
    result = query_origins.unsqueeze(1) - PERIOD * ages.unsqueeze(0)
    if bool(torch.any(result + HORIZON - 1 > query_origins.unsqueeze(1) - 1)):
        raise RuntimeError("DiffAmp memory maturity contract violated")
    return result


T = TypeVar("T")


def stage_b_only_if_unlocked(unlocked: bool, callback: Callable[[], T]) -> T | None:
    """Make audit/cache construction unreachable after a failed Stage-A gate."""

    if not bool(unlocked):
        return None
    return callback()


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
