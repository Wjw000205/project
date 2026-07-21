"""Variable-horizon patch-token Amp adapter for frozen forecasts.

One parameter-shared network handles any horizon divisible by the patch size.
For each future patch it attends over eight fully observed aligned forecast-
residual carriers, predicts signed mixture weights, amplitude and uncertainty,
then projects the complete correction to zero Level and zero linear Trend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.models.periodic_adapter_config import LOCKED_KERNEL_GEOMETRY
from torch import nn


@dataclass(frozen=True)
class AmpPatchMixtureOutput:
    correction: torch.Tensor
    mixture_weight: torch.Tensor
    carrier_sign: torch.Tensor
    amplitude: torch.Tensor
    uncertainty: torch.Tensor
    uncertainty_shrink: torch.Tensor


@dataclass(frozen=True)
class AmpPatchUtilitySelectorOutput:
    """Continuous all-carrier actions with a target-free utility selection."""

    correction: torch.Tensor
    mixture_weight: torch.Tensor
    carrier_sign: torch.Tensor
    amplitude: torch.Tensor
    uncertainty: torch.Tensor
    uncertainty_shrink: torch.Tensor
    candidate_correction: torch.Tensor
    candidate_unit: torch.Tensor
    coefficient: torch.Tensor
    utility_score: torch.Tensor


def remove_affine(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError("Amp affine projection expects [B,H]")
    basis = torch.linspace(-1.0, 1.0, values.shape[1], device=values.device, dtype=values.dtype)
    basis = basis - basis.mean()
    centered = values - values.mean(dim=1, keepdim=True)
    coefficient = (centered * basis).sum(dim=1, keepdim=True) / basis.square().sum().clamp_min(1.0e-12)
    return centered - coefficient * basis


def project_carriers_to_amp(carriers: torch.Tensor) -> torch.Tensor:
    """Project every full-horizon historical carrier into Amp space."""
    if carriers.ndim != 4:
        raise ValueError("carriers must have shape [B,P,K,L]")
    batch, patch_count, carrier_count, patch_len = carriers.shape
    full = carriers.permute(0, 2, 1, 3).reshape(
        batch * carrier_count, patch_count * patch_len
    )
    projected = remove_affine(full)
    return projected.reshape(batch, carrier_count, patch_count, patch_len).permute(
        0, 2, 1, 3
    )


def project_carriers_to_patch_amp(carriers: torch.Tensor) -> torch.Tensor:
    """Remove each carrier patch mean using no horizon/period constant."""

    if carriers.ndim != 4:
        raise ValueError("carriers must have shape [B,P,K,L]")
    return carriers - carriers.mean(dim=-1, keepdim=True)


def project_to_patch_amp(values: torch.Tensor, *, patch_len: int = 12) -> torch.Tensor:
    """Project a flat action into the within-patch zero-mean Amp space."""

    if values.ndim != 2 or patch_len <= 1 or values.shape[1] % patch_len != 0:
        raise ValueError("patch Amp projection expects [B,H] divisible by patch_len")
    patches = values.reshape(values.shape[0], -1, patch_len)
    return (patches - patches.mean(dim=2, keepdim=True)).reshape_as(values)


def project_carriers_to_fixed_block_amp(
    carriers: torch.Tensor,
    *,
    block_steps: int = 96,
) -> torch.Tensor:
    """Remove affine coordinates in repeated fixed physical-time blocks.

    Unlike complete-horizon projection, an already complete block is unchanged
    when later forecast patches are appended.  H96 is exactly the legacy Amp
    carrier definition; longer horizons repeat that one-day coordinate system.
    """

    if carriers.ndim != 4:
        raise ValueError("carriers must have shape [B,P,K,L]")
    batch, patch_count, carrier_count, patch_len = carriers.shape
    if block_steps <= 1 or block_steps % patch_len != 0:
        raise ValueError("block_steps must be divisible by carrier patch length")
    block_patches = block_steps // patch_len
    result = torch.empty_like(carriers)
    for left in range(0, patch_count, block_patches):
        right = min(left + block_patches, patch_count)
        width = right - left
        full = carriers[:, left:right].permute(0, 2, 1, 3).reshape(
            batch * carrier_count, width * patch_len
        )
        projected = remove_affine(full)
        result[:, left:right] = projected.reshape(
            batch, carrier_count, width, patch_len
        ).permute(0, 2, 1, 3)
    return result


def project_to_fixed_block_amp(
    values: torch.Tensor,
    *,
    block_steps: int = 96,
) -> torch.Tensor:
    """Project a [B,H] action independently in fixed physical-time blocks."""

    if values.ndim != 2:
        raise ValueError("fixed-block Amp projection expects [B,H]")
    if block_steps <= 1:
        raise ValueError("block_steps must exceed one")
    result = torch.empty_like(values)
    for left in range(0, values.shape[1], block_steps):
        right = min(left + block_steps, values.shape[1])
        result[:, left:right] = remove_affine(values[:, left:right])
    return result


def patch_position_features(
    patch_count: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    harmonics: int = 4,
    patch_len: int = 12,
    period_patches: int = 8,
    position_mode: str = "normalized",
) -> torch.Tensor:
    if patch_count <= 0:
        raise ValueError("patch_count must be positive")
    if patch_len <= 0 or period_patches <= 0:
        raise ValueError("patch_len and period_patches must be positive")
    position = (torch.arange(patch_count, device=device, dtype=dtype) + 0.5) / patch_count
    normalized_angle = 2.0 * math.pi * position
    if position_mode == "normalized":
        return torch.stack(
            [
                function(harmonic * normalized_angle)
                for harmonic in range(1, harmonics + 1)
                for function in (torch.sin, torch.cos)
            ],
            dim=1,
        )
    if position_mode not in {"physical_lead", "physical_only", "periodic_only"}:
        raise ValueError(
            "position_mode must be normalized, physical_lead, or periodic_only"
        )
    if harmonics != 4:
        raise ValueError("physical_lead position mode expects four normalized harmonics")
    lead_steps = (
        torch.arange(patch_count, device=device, dtype=dtype) + 0.5
    ) * patch_len
    canonical_period_steps = float(period_patches * patch_len)
    daily_angle = 2.0 * math.pi * lead_steps / canonical_period_steps
    weekly_angle = 2.0 * math.pi * lead_steps / (7.0 * canonical_period_steps)
    if position_mode in {"physical_only", "periodic_only"}:
        lead_days = lead_steps / canonical_period_steps
        return torch.stack(
            [
                torch.sin(daily_angle),
                torch.cos(daily_angle),
                torch.sin(weekly_angle),
                torch.cos(weekly_angle),
                lead_days,
                torch.log1p(lead_days),
                1.0 / (1.0 + lead_days),
            ],
            dim=1,
        )
    normalized = torch.stack(
        [
            function(harmonic * normalized_angle)
            for harmonic in range(1, harmonics + 1)
            for function in (torch.sin, torch.cos)
        ],
        dim=1,
    )
    physical = torch.stack(
        [
            torch.sin(daily_angle),
            torch.cos(daily_angle),
            torch.sin(weekly_angle),
            torch.cos(weekly_angle),
        ],
        dim=1,
    )
    return torch.cat([normalized, physical], dim=1)


def carrier_age_summary_features(
    carrier_unit: torch.Tensor,
    carrier_rms: torch.Tensor,
) -> torch.Tensor:
    """Fixed recent-minus-old carrier shape and log-energy evolution."""

    if carrier_unit.ndim != 4 or carrier_rms.ndim != 4:
        raise ValueError("carrier age summary expects [B,P,K,L] and [B,P,K,1]")
    if carrier_unit.shape[:3] != carrier_rms.shape[:3] or carrier_rms.shape[-1] != 1:
        raise ValueError("carrier age summary dimensions do not align")
    carrier_count = int(carrier_unit.shape[2])
    if carrier_count < 2:
        raise ValueError("carrier age summary requires at least two carriers")
    split = carrier_count // 2
    old_shape = carrier_unit[:, :, :split].mean(dim=2)
    recent_shape = carrier_unit[:, :, split:].mean(dim=2)
    old_rms = carrier_rms[:, :, :split].mean(dim=2)
    recent_rms = carrier_rms[:, :, split:].mean(dim=2)
    log_energy_ratio = torch.log((recent_rms + 1.0e-6) / (old_rms + 1.0e-6))
    return torch.cat([recent_shape - old_shape, log_energy_ratio], dim=-1)


def sparsemax(values: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Differentiable Euclidean projection onto the probability simplex."""
    if values.ndim == 0:
        raise ValueError("sparsemax expects at least one dimension")
    dim = dim if dim >= 0 else values.ndim + dim
    if dim < 0 or dim >= values.ndim:
        raise ValueError("invalid sparsemax dimension")
    sorted_values, _ = torch.sort(values, dim=dim, descending=True)
    cumulative = torch.cumsum(sorted_values, dim=dim)
    count = values.shape[dim]
    rank_shape = [1] * values.ndim
    rank_shape[dim] = count
    rank = torch.arange(1, count + 1, device=values.device, dtype=values.dtype).view(
        rank_shape
    )
    support = 1.0 + rank * sorted_values > cumulative
    support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
    threshold_sum = torch.gather(cumulative, dim, support_size - 1)
    threshold = (threshold_sum - 1.0) / support_size.to(values.dtype)
    return torch.clamp(values - threshold, min=0.0)


class AmpPatchMixtureAdapter(nn.Module):
    """Shared-token carrier mixture whose parameter count is horizon invariant."""

    def __init__(
        self,
        *,
        patch_len: int = 12,
        carrier_count: int = 8,
        period_patches: int = 8,
        context_width: int = 11,
        hidden: int = 64,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.10,
        carrier_space: str = "raw",
        ordered_history: bool = False,
        history_summary: bool = False,
        sparse_mixture: bool = False,
        position_mode: str = "normalized",
        scale_invariant_inputs: bool = False,
        causal_attention: bool = False,
        output_projection: str = "global",
    ) -> None:
        super().__init__()
        if (
            patch_len <= 1
            or carrier_count <= 0
            or period_patches <= 0
            or context_width <= 0
        ):
            raise ValueError("invalid Amp patch-mixture dimensions")
        if hidden <= 0 or heads <= 0 or hidden % heads != 0 or layers <= 0:
            raise ValueError("invalid Amp transformer dimensions")
        if carrier_space not in {
            "raw",
            "amp",
            "amp_block96",
            "amp_periodic",
            "amp_patch",
        }:
            raise ValueError("carrier_space must be raw, amp, or amp_periodic")
        if position_mode not in {
            "normalized",
            "physical_lead",
            "physical_only",
            "periodic_only",
        }:
            raise ValueError(
                "position_mode must be normalized, physical_lead, or periodic_only"
            )
        if ordered_history and history_summary:
            raise ValueError("choose either learned ordered history or fixed history summary")
        if history_summary and carrier_count < 2:
            raise ValueError("history_summary requires at least two carriers")
        if output_projection not in {"global", "block96", "periodic", "patch"}:
            raise ValueError("output_projection must be global or periodic")
        self.patch_len = int(patch_len)
        self.carrier_count = int(carrier_count)
        self.period_patches = int(period_patches)
        self.context_width = int(context_width)
        self.hidden = int(hidden)
        self.carrier_space = str(carrier_space)
        self.ordered_history = bool(ordered_history)
        self.history_summary = bool(history_summary)
        self.sparse_mixture = bool(sparse_mixture)
        self.position_mode = str(position_mode)
        self.scale_invariant_inputs = bool(scale_invariant_inputs)
        self.causal_attention = bool(causal_attention)
        self.output_projection = str(output_projection)
        self.carrier_encoder = nn.Sequential(
            nn.Linear(patch_len, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.carrier_scale_encoder = nn.Linear(1, hidden)
        if self.ordered_history or self.history_summary:
            self.carrier_age_encoder = nn.Linear(4, hidden)
        if self.ordered_history:
            history_layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=heads,
                dim_feedforward=2 * hidden,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.history_encoder = nn.TransformerEncoder(history_layer, num_layers=1)
        if self.history_summary:
            self.history_delta_encoder = nn.Sequential(
                nn.Linear(patch_len + 1, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
            )
        self.base_encoder = nn.Sequential(
            nn.Linear(patch_len, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        position_width = {
            "normalized": 8,
            "physical_lead": 12,
            "physical_only": 7,
            "periodic_only": 7,
        }[self.position_mode]
        self.position_encoder = nn.Linear(position_width, hidden)
        self.context_encoder = nn.Sequential(
            nn.Linear(context_width, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
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
        self.patch_encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.query_norm = nn.LayerNorm(hidden)
        self.sign_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.amplitude_head = nn.Linear(hidden, 1)
        self.uncertainty_head = nn.Linear(hidden, 1)
        self.residual_decoder = nn.Linear(hidden, patch_len)
        nn.init.zeros_(self.sign_head[-1].weight)
        nn.init.zeros_(self.sign_head[-1].bias)
        nn.init.zeros_(self.amplitude_head.weight)
        nn.init.constant_(self.amplitude_head.bias, -0.5)
        nn.init.zeros_(self.uncertainty_head.weight)
        nn.init.constant_(self.uncertainty_head.bias, -1.5)
        nn.init.zeros_(self.residual_decoder.weight)
        nn.init.zeros_(self.residual_decoder.bias)

    def forward(
        self,
        carriers: torch.Tensor,
        base_patch: torch.Tensor,
        context: torch.Tensor,
    ) -> AmpPatchMixtureOutput:
        if carriers.ndim != 4:
            raise ValueError("carriers must have shape [B,P,K,L]")
        batch, patch_count, carrier_count, patch_len = carriers.shape
        if carrier_count != self.carrier_count or patch_len != self.patch_len:
            raise ValueError("Amp carrier dimensions mismatch")
        if base_patch.shape != (batch, patch_count, patch_len):
            raise ValueError("Amp base-patch dimensions mismatch")
        if context.shape != (batch, self.context_width):
            raise ValueError("Amp context dimensions mismatch")

        if self.carrier_space == "amp":
            carriers = project_carriers_to_amp(carriers)
        elif self.carrier_space in {"amp_block96", "amp_periodic"}:
            carriers = project_carriers_to_fixed_block_amp(
                carriers,
                block_steps=self.period_patches * self.patch_len,
            )
        elif self.carrier_space == "amp_patch":
            carriers = project_carriers_to_patch_amp(carriers)

        carrier_rms = torch.sqrt(carriers.square().mean(dim=-1, keepdim=True) + 1.0e-8)
        carrier_unit = carriers / carrier_rms
        if self.scale_invariant_inputs:
            # Amp is orthogonal to complete-window Level and Trend.  Remove
            # those nuisance coordinates from the frozen forecast and expose
            # only within-example relative carrier energy.  The physical RMS
            # below still restores the correction to the original units.
            carrier_global_rms = torch.sqrt(
                carriers.square().mean(dim=(1, 2, 3), keepdim=True) + 1.0e-8
            )
            carrier_scale_feature = torch.log(
                carrier_rms / carrier_global_rms + 1.0e-6
            )
            base_amp = remove_affine(base_patch.reshape(batch, patch_count * patch_len))
            base_rms = torch.sqrt(
                base_amp.square().mean(dim=1, keepdim=True) + 1.0e-8
            )
            base_feature = (base_amp / base_rms).reshape(
                batch, patch_count, patch_len
            )
        else:
            carrier_scale_feature = torch.log(carrier_rms + 1.0e-6)
            base_feature = base_patch
        carrier_token = self.carrier_encoder(carrier_unit)
        carrier_token = carrier_token + self.carrier_scale_encoder(carrier_scale_feature)
        if self.ordered_history or self.history_summary:
            age = torch.linspace(
                -1.0,
                1.0,
                carrier_count,
                device=carriers.device,
                dtype=carriers.dtype,
            )
            age_feature = torch.stack(
                [age, age.square(), torch.sin(math.pi * age), torch.cos(math.pi * age)],
                dim=1,
            )
            carrier_token = carrier_token + self.carrier_age_encoder(age_feature).view(
                1, 1, carrier_count, self.hidden
            )
        if self.ordered_history:
            carrier_token = self.history_encoder(
                carrier_token.reshape(batch * patch_count, carrier_count, self.hidden)
            ).reshape(batch, patch_count, carrier_count, self.hidden)
        history_delta_token: torch.Tensor | float = 0.0
        if self.history_summary:
            history_delta_token = self.history_delta_encoder(
                carrier_age_summary_features(carrier_unit, carrier_rms)
            )
        pooled_carrier = carrier_token.mean(dim=2)
        position = patch_position_features(
            patch_count,
            device=carriers.device,
            dtype=carriers.dtype,
            patch_len=self.patch_len,
            period_patches=self.period_patches,
            position_mode=self.position_mode,
        )
        token = (
            self.base_encoder(base_feature)
            + pooled_carrier
            + history_delta_token
            + self.position_encoder(position).unsqueeze(0)
            + self.context_encoder(context).unsqueeze(1)
        )
        if self.causal_attention:
            causal_mask = torch.ones(
                patch_count,
                patch_count,
                device=token.device,
                dtype=torch.bool,
            ).triu(diagonal=1)
            encoded = self.patch_encoder(token, mask=causal_mask)
        else:
            encoded = self.patch_encoder(token)
        query = self.query_norm(encoded)
        logits = (query.unsqueeze(2) * carrier_token).sum(dim=-1) / math.sqrt(self.hidden)
        mixture_weight = (
            sparsemax(logits, dim=2)
            if self.sparse_mixture
            else torch.softmax(logits, dim=2)
        )
        query_expanded = query.unsqueeze(2).expand_as(carrier_token)
        carrier_sign = torch.tanh(
            self.sign_head(torch.cat([query_expanded, carrier_token], dim=-1)).squeeze(-1)
        )
        mixture = (
            mixture_weight.unsqueeze(-1)
            * carrier_sign.unsqueeze(-1)
            * carrier_unit
        ).sum(dim=2)
        carrier_scale = carrier_rms.mean(dim=2).squeeze(-1)
        amplitude = F.softplus(self.amplitude_head(query).squeeze(-1)) * carrier_scale
        uncertainty = F.softplus(self.uncertainty_head(query).squeeze(-1)) + 1.0e-4
        uncertainty_shrink = 1.0 / (1.0 + uncertainty)
        free_residual = 0.25 * carrier_scale.unsqueeze(-1) * torch.tanh(
            self.residual_decoder(query)
        )
        raw_patch = uncertainty_shrink.unsqueeze(-1) * (
            amplitude.unsqueeze(-1) * mixture + free_residual
        )
        raw_correction = raw_patch.reshape(batch, patch_count * patch_len)
        correction = (
            project_to_patch_amp(raw_correction, patch_len=self.patch_len)
            if self.output_projection == "patch"
            else
            project_to_fixed_block_amp(
                raw_correction,
                block_steps=self.period_patches * self.patch_len,
            )
            if self.output_projection in {"block96", "periodic"}
            else remove_affine(raw_correction)
        )
        return AmpPatchMixtureOutput(
            correction=correction,
            mixture_weight=mixture_weight,
            carrier_sign=carrier_sign,
            amplitude=amplitude,
            uncertainty=uncertainty,
            uncertainty_shrink=uncertainty_shrink,
        )


class UniversalPeriodicAmpAdapter(AmpPatchMixtureAdapter):
    """One fixed Amp network for every dataset and forecast horizon.

    All native clocks are converted to the same eight-token canonical period
    and the same 28 complete historical periods before this module.
    Consequently this constructor exposes no dataset, physical period, patch
    size, horizon, carrier recipe, or projection mode.  A fixed recency summary
    exposes residual evolution without making the parameter count depend on
    dataset or forecast length.
    """

    def __init__(
        self,
        *,
        hidden: int = 64,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.10,
    ) -> None:
        super().__init__(
            patch_len=12,
            carrier_count=LOCKED_KERNEL_GEOMETRY.carrier_periods,
            period_patches=8,
            context_width=11,
            hidden=hidden,
            heads=heads,
            layers=layers,
            dropout=dropout,
            carrier_space="amp_patch",
            ordered_history=False,
            history_summary=True,
            sparse_mixture=False,
            position_mode="periodic_only",
            scale_invariant_inputs=False,
            causal_attention=True,
            output_projection="patch",
        )


class AmpPatchUtilitySelectorAdapter(nn.Module):
    """Select a patch carrier by continuously supervised expected utility.

    Every carrier receives a signed scalar action and a continuous utility
    score.  Training supervises all actions and utilities; it never supplies a
    hard oracle-carrier class.  Evaluation uses the highest predicted utility,
    avoiding the phase cancellation of a conditional-mean carrier mixture.
    """

    def __init__(
        self,
        *,
        patch_len: int = 12,
        carrier_count: int = 8,
        context_width: int = 11,
        hidden: int = 64,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.10,
        signed_phase_only: bool = False,
        carrier_age_summary: bool = False,
        ordered_carrier_gru: bool = False,
        carrier_space: str = "amp",
        position_mode: str = "normalized",
        scale_invariant_inputs: bool = False,
        causal_attention: bool = False,
        output_projection: str = "global",
    ) -> None:
        super().__init__()
        if patch_len <= 1 or carrier_count <= 0 or context_width <= 0:
            raise ValueError("invalid Amp utility-selector dimensions")
        if hidden <= 0 or heads <= 0 or hidden % heads != 0 or layers <= 0:
            raise ValueError("invalid Amp utility-selector transformer dimensions")
        self.patch_len = int(patch_len)
        self.carrier_count = int(carrier_count)
        self.context_width = int(context_width)
        self.hidden = int(hidden)
        self.signed_phase_only = bool(signed_phase_only)
        self.signed_phase_candidates = self.signed_phase_only
        self.carrier_age_summary = bool(carrier_age_summary)
        self.ordered_carrier_gru = bool(ordered_carrier_gru)
        if carrier_space not in {"amp", "amp_block96"}:
            raise ValueError("utility-selector carrier_space must be amp or amp_block96")
        if position_mode not in {"normalized", "physical_lead", "physical_only"}:
            raise ValueError("invalid utility-selector position mode")
        if output_projection not in {"global", "block96"}:
            raise ValueError("utility-selector output projection must be global or block96")
        self.carrier_space = str(carrier_space)
        self.position_mode = str(position_mode)
        self.scale_invariant_inputs = bool(scale_invariant_inputs)
        self.causal_attention = bool(causal_attention)
        self.output_projection = str(output_projection)
        if self.carrier_age_summary and self.ordered_carrier_gru:
            raise ValueError("choose carrier age summary or ordered carrier GRU")
        self.carrier_encoder = nn.Sequential(
            nn.Linear(patch_len, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.carrier_scale_encoder = nn.Linear(1, hidden)
        self.carrier_age_encoder = nn.Linear(4, hidden)
        self.base_encoder = nn.Sequential(
            nn.Linear(patch_len, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        position_width = {
            "normalized": 8,
            "physical_lead": 12,
            "physical_only": 7,
        }[self.position_mode]
        self.position_encoder = nn.Linear(position_width, hidden)
        self.context_encoder = nn.Sequential(
            nn.Linear(context_width, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        if self.carrier_age_summary:
            self.history_delta_encoder = nn.Sequential(
                nn.Linear(patch_len + 1, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
            )
        if self.ordered_carrier_gru:
            self.carrier_history_gru = nn.GRU(hidden, hidden, batch_first=True)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=2 * hidden,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.patch_encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.query_norm = nn.LayerNorm(hidden)
        self.pair_encoder = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.coefficient_head = nn.Linear(hidden, 1)
        self.utility_head = nn.Linear(hidden, 1)
        self.uncertainty_head = nn.Linear(hidden, 1)
        if self.signed_phase_only:
            self.global_amplitude = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.coefficient_head.weight)
        nn.init.zeros_(self.coefficient_head.bias)
        nn.init.zeros_(self.utility_head.weight)
        nn.init.zeros_(self.utility_head.bias)
        nn.init.zeros_(self.uncertainty_head.weight)
        nn.init.constant_(self.uncertainty_head.bias, -3.0)

    def forward(
        self,
        carriers: torch.Tensor,
        base_patch: torch.Tensor,
        context: torch.Tensor,
    ) -> AmpPatchUtilitySelectorOutput:
        if carriers.ndim != 4:
            raise ValueError("carriers must have shape [B,P,K,L]")
        batch, patch_count, carrier_count, patch_len = carriers.shape
        if carrier_count != self.carrier_count or patch_len != self.patch_len:
            raise ValueError("Amp utility-selector carrier dimensions mismatch")
        if base_patch.shape != (batch, patch_count, patch_len):
            raise ValueError("Amp utility-selector base-patch dimensions mismatch")
        if context.shape != (batch, self.context_width):
            raise ValueError("Amp utility-selector context dimensions mismatch")

        carriers = (
            project_carriers_to_fixed_block_amp(carriers)
            if self.carrier_space == "amp_block96"
            else project_carriers_to_amp(carriers)
        )
        carrier_rms = torch.sqrt(carriers.square().mean(dim=-1, keepdim=True) + 1.0e-8)
        carrier_unit = carriers / carrier_rms
        if self.scale_invariant_inputs:
            carrier_global_rms = torch.sqrt(
                carriers.square().mean(dim=(1, 2, 3), keepdim=True) + 1.0e-8
            )
            carrier_scale_feature = torch.log(
                carrier_rms / carrier_global_rms + 1.0e-6
            )
            base_amp = (
                project_to_fixed_block_amp(
                    base_patch.reshape(batch, patch_count * patch_len)
                )
                if self.output_projection == "block96"
                else remove_affine(base_patch.reshape(batch, patch_count * patch_len))
            )
            base_rms = torch.sqrt(
                base_amp.square().mean(dim=1, keepdim=True) + 1.0e-8
            )
            base_feature = (base_amp / base_rms).reshape(
                batch, patch_count, patch_len
            )
        else:
            carrier_scale_feature = torch.log(carrier_rms + 1.0e-6)
            base_feature = base_patch
        carrier_token = self.carrier_encoder(carrier_unit)
        carrier_token = carrier_token + self.carrier_scale_encoder(
            carrier_scale_feature
        )
        age = torch.linspace(
            -1.0,
            1.0,
            carrier_count,
            device=carriers.device,
            dtype=carriers.dtype,
        )
        age_feature = torch.stack(
            [age, age.square(), torch.sin(math.pi * age), torch.cos(math.pi * age)],
            dim=1,
        )
        carrier_token = carrier_token + self.carrier_age_encoder(age_feature).view(
            1, 1, carrier_count, self.hidden
        )
        position = patch_position_features(
            patch_count,
            device=carriers.device,
            dtype=carriers.dtype,
            patch_len=self.patch_len,
            position_mode=self.position_mode,
        )
        history_delta_token: torch.Tensor | float = 0.0
        if self.carrier_age_summary:
            history_delta_token = self.history_delta_encoder(
                carrier_age_summary_features(carrier_unit, carrier_rms)
            )
        if self.ordered_carrier_gru:
            _history_output, history_hidden = self.carrier_history_gru(
                carrier_token.reshape(
                    batch * patch_count, carrier_count, self.hidden
                )
            )
            history_delta_token = history_hidden[-1].reshape(
                batch, patch_count, self.hidden
            )
        token = (
            self.base_encoder(base_feature)
            + carrier_token.mean(dim=2)
            + history_delta_token
            + self.position_encoder(position).unsqueeze(0)
            + self.context_encoder(context).unsqueeze(1)
        )
        if self.causal_attention:
            causal_mask = torch.ones(
                patch_count,
                patch_count,
                device=token.device,
                dtype=torch.bool,
            ).triu(diagonal=1)
            encoded = self.patch_encoder(token, mask=causal_mask)
        else:
            encoded = self.patch_encoder(token)
        query = self.query_norm(encoded)
        pair = self.pair_encoder(
            torch.cat(
                [query.unsqueeze(2).expand_as(carrier_token), carrier_token], dim=-1
            )
        )
        coefficient = self.coefficient_head(pair).squeeze(-1) * carrier_rms.squeeze(-1)
        utility_score = self.utility_head(pair).squeeze(-1)
        if self.signed_phase_only:
            signed_unit = torch.cat([carrier_unit, -carrier_unit], dim=2)
            utility_score = torch.cat([utility_score, -utility_score], dim=2)
            mixture_weight = torch.softmax(utility_score, dim=2)
            hard_index = utility_score.argmax(dim=2)
            hard_weight = F.one_hot(
                hard_index, 2 * carrier_count
            ).to(utility_score.dtype)
            selection_weight = (
                hard_weight
                if not self.training
                else hard_weight + mixture_weight - mixture_weight.detach()
            )
            candidate_correction = self.global_amplitude * signed_unit
            selected_patch = (
                selection_weight.unsqueeze(-1) * candidate_correction
            ).sum(dim=2)
            raw_correction = selected_patch.reshape(batch, patch_count * patch_len)
            correction = (
                project_to_fixed_block_amp(raw_correction)
                if self.output_projection == "block96"
                else remove_affine(raw_correction)
            )
            amplitude = self.global_amplitude.abs().expand(batch, patch_count)
            uncertainty = torch.ones_like(amplitude)
            uncertainty_shrink = torch.ones_like(amplitude)
            return AmpPatchUtilitySelectorOutput(
                correction=correction,
                mixture_weight=mixture_weight,
                carrier_sign=torch.tanh(utility_score[:, :, :carrier_count]),
                amplitude=amplitude,
                uncertainty=uncertainty,
                uncertainty_shrink=uncertainty_shrink,
                candidate_correction=candidate_correction,
                candidate_unit=signed_unit,
                coefficient=self.global_amplitude.expand(
                    batch, patch_count, 2 * carrier_count
                ),
                utility_score=utility_score,
            )
        mixture_weight = torch.softmax(utility_score, dim=2)
        hard_index = utility_score.argmax(dim=2)
        hard_weight = F.one_hot(hard_index, carrier_count).to(utility_score.dtype)
        selection_weight = (
            hard_weight
            if not self.training
            else hard_weight + mixture_weight - mixture_weight.detach()
        )
        candidate_correction = coefficient.unsqueeze(-1) * carrier_unit
        selected_patch = (
            selection_weight.unsqueeze(-1) * candidate_correction
        ).sum(dim=2)
        uncertainty = F.softplus(self.uncertainty_head(query).squeeze(-1)) + 1.0e-4
        uncertainty_shrink = 1.0 / (1.0 + uncertainty)
        selected_patch = uncertainty_shrink.unsqueeze(-1) * selected_patch
        raw_correction = selected_patch.reshape(batch, patch_count * patch_len)
        correction = (
            project_to_fixed_block_amp(raw_correction)
            if self.output_projection == "block96"
            else remove_affine(raw_correction)
        )
        amplitude = torch.sqrt(selected_patch.square().mean(dim=-1) + 1.0e-8)
        return AmpPatchUtilitySelectorOutput(
            correction=correction,
            mixture_weight=mixture_weight,
            carrier_sign=torch.tanh(
                coefficient / carrier_rms.squeeze(-1).clamp_min(1.0e-6)
            ),
            amplitude=amplitude,
            uncertainty=uncertainty,
            uncertainty_shrink=uncertainty_shrink,
            candidate_correction=candidate_correction,
            candidate_unit=carrier_unit,
            coefficient=coefficient,
            utility_score=utility_score,
        )


class AmpLowRankMixtureAdapter(nn.Module):
    """Whole-horizon carrier mixture with fixed-rank local modulation.

    The encoder may read an arbitrary number of future patches, but all local
    decisions are decoded from a fixed number of latent tokens.  Routing
    degrees of freedom therefore do not grow with the forecast horizon.
    """

    def __init__(
        self,
        *,
        patch_len: int = 12,
        carrier_count: int = 8,
        context_width: int = 11,
        hidden: int = 64,
        heads: int = 4,
        layers: int = 2,
        latent_rank: int = 4,
        dropout: float = 0.10,
        carrier_space: str = "raw",
    ) -> None:
        super().__init__()
        if patch_len <= 1 or carrier_count <= 0 or context_width <= 0:
            raise ValueError("invalid Amp low-rank dimensions")
        if hidden <= 0 or heads <= 0 or hidden % heads != 0 or layers <= 0:
            raise ValueError("invalid Amp low-rank transformer dimensions")
        if latent_rank <= 0:
            raise ValueError("latent_rank must be positive")
        if carrier_space not in {"raw", "amp"}:
            raise ValueError("carrier_space must be raw or amp")
        self.patch_len = int(patch_len)
        self.carrier_count = int(carrier_count)
        self.context_width = int(context_width)
        self.hidden = int(hidden)
        self.latent_rank = int(latent_rank)
        self.carrier_space = str(carrier_space)

        self.carrier_encoder = nn.Sequential(
            nn.Linear(patch_len, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.carrier_scale_encoder = nn.Linear(1, hidden)
        self.base_encoder = nn.Sequential(
            nn.Linear(patch_len, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.position_encoder = nn.Linear(8, hidden)
        self.context_encoder = nn.Sequential(
            nn.Linear(context_width, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=3 * hidden,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.patch_encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.latent_query = nn.Parameter(torch.empty(latent_rank, hidden))
        nn.init.normal_(self.latent_query, std=0.02)
        self.latent_attention = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True
        )
        self.latent_norm = nn.LayerNorm(hidden)
        self.latent_feedforward = nn.Sequential(
            nn.Linear(hidden, 2 * hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, hidden),
        )
        self.position_to_rank = nn.Linear(8, latent_rank)
        self.decode_norm = nn.LayerNorm(hidden)

        self.global_logit_head = nn.Linear(2 * hidden, 1)
        self.global_sign_head = nn.Linear(2 * hidden, 1)
        self.local_logit_head = nn.Linear(hidden, carrier_count)
        self.local_sign_head = nn.Linear(hidden, carrier_count)
        self.amplitude_head = nn.Linear(hidden, 1)
        self.local_amplitude_head = nn.Linear(hidden, 1)
        self.uncertainty_head = nn.Linear(hidden, 1)
        self.residual_decoder = nn.Linear(hidden, patch_len)

        for layer_to_zero in (
            self.global_sign_head,
            self.local_sign_head,
            self.residual_decoder,
        ):
            nn.init.zeros_(layer_to_zero.weight)
            nn.init.zeros_(layer_to_zero.bias)
        nn.init.zeros_(self.local_logit_head.weight)
        nn.init.zeros_(self.local_logit_head.bias)
        nn.init.zeros_(self.amplitude_head.weight)
        nn.init.constant_(self.amplitude_head.bias, -0.5)
        nn.init.zeros_(self.local_amplitude_head.weight)
        nn.init.zeros_(self.local_amplitude_head.bias)
        nn.init.zeros_(self.uncertainty_head.weight)
        nn.init.constant_(self.uncertainty_head.bias, -1.5)

    def forward(
        self,
        carriers: torch.Tensor,
        base_patch: torch.Tensor,
        context: torch.Tensor,
    ) -> AmpPatchMixtureOutput:
        if carriers.ndim != 4:
            raise ValueError("carriers must have shape [B,P,K,L]")
        batch, patch_count, carrier_count, patch_len = carriers.shape
        if carrier_count != self.carrier_count or patch_len != self.patch_len:
            raise ValueError("Amp low-rank carrier dimensions mismatch")
        if base_patch.shape != (batch, patch_count, patch_len):
            raise ValueError("Amp low-rank base-patch dimensions mismatch")
        if context.shape != (batch, self.context_width):
            raise ValueError("Amp low-rank context dimensions mismatch")

        if self.carrier_space == "amp":
            carriers = project_carriers_to_amp(carriers)

        carrier_rms = torch.sqrt(carriers.square().mean(dim=-1, keepdim=True) + 1.0e-8)
        carrier_unit = carriers / carrier_rms
        carrier_token = self.carrier_encoder(carrier_unit)
        carrier_token = carrier_token + self.carrier_scale_encoder(
            torch.log(carrier_rms + 1.0e-6)
        )
        position = patch_position_features(
            patch_count, device=carriers.device, dtype=carriers.dtype
        )
        context_token = self.context_encoder(context)
        patch_token = (
            self.base_encoder(base_patch)
            + carrier_token.mean(dim=2)
            + self.position_encoder(position).unsqueeze(0)
            + context_token.unsqueeze(1)
        )
        encoded_patch = self.patch_encoder(patch_token)
        latent_query = self.latent_query.unsqueeze(0).expand(batch, -1, -1)
        attended, _ = self.latent_attention(
            latent_query, encoded_patch, encoded_patch, need_weights=False
        )
        latent = self.latent_norm(latent_query + attended)
        latent = self.latent_norm(latent + self.latent_feedforward(latent))
        global_state = latent.mean(dim=1) + context_token

        rank_weight = torch.softmax(self.position_to_rank(position), dim=1)
        decoded = torch.einsum("pr,brh->bph", rank_weight, latent)
        decoded = self.decode_norm(
            decoded
            + self.position_encoder(position).unsqueeze(0)
            + global_state.unsqueeze(1)
        )

        global_carrier = carrier_token.mean(dim=1)
        global_expanded = global_state.unsqueeze(1).expand_as(global_carrier)
        global_pair = torch.cat([global_expanded, global_carrier], dim=-1)
        global_logits = self.global_logit_head(global_pair).squeeze(-1)
        local_logits = 0.35 * torch.tanh(self.local_logit_head(decoded))
        mixture_weight = torch.softmax(global_logits.unsqueeze(1) + local_logits, dim=2)

        global_sign_raw = self.global_sign_head(global_pair).squeeze(-1)
        local_sign_raw = 0.35 * self.local_sign_head(decoded)
        carrier_sign = torch.tanh(global_sign_raw.unsqueeze(1) + local_sign_raw)
        mixture = (
            mixture_weight.unsqueeze(-1)
            * carrier_sign.unsqueeze(-1)
            * carrier_unit
        ).sum(dim=2)

        carrier_scale = carrier_rms.mean(dim=2).squeeze(-1)
        amplitude_raw = self.amplitude_head(global_state).unsqueeze(1)
        amplitude_raw = amplitude_raw + 0.50 * torch.tanh(
            self.local_amplitude_head(decoded)
        )
        amplitude = F.softplus(amplitude_raw.squeeze(-1)) * carrier_scale
        uncertainty = F.softplus(self.uncertainty_head(decoded).squeeze(-1)) + 1.0e-4
        uncertainty_shrink = 1.0 / (1.0 + uncertainty)
        free_residual = 0.15 * carrier_scale.unsqueeze(-1) * torch.tanh(
            self.residual_decoder(decoded)
        )
        raw_patch = uncertainty_shrink.unsqueeze(-1) * (
            amplitude.unsqueeze(-1) * mixture + free_residual
        )
        correction = remove_affine(raw_patch.reshape(batch, patch_count * patch_len))
        return AmpPatchMixtureOutput(
            correction=correction,
            mixture_weight=mixture_weight,
            carrier_sign=carrier_sign,
            amplitude=amplitude,
            uncertainty=uncertainty,
            uncertainty_shrink=uncertainty_shrink,
        )


__all__ = [
    "AmpLowRankMixtureAdapter",
    "AmpPatchMixtureAdapter",
    "AmpPatchMixtureOutput",
    "UniversalPeriodicAmpAdapter",
    "patch_position_features",
    "project_carriers_to_amp",
    "project_carriers_to_patch_amp",
    "project_carriers_to_fixed_block_amp",
    "project_to_patch_amp",
    "project_to_fixed_block_amp",
    "remove_affine",
    "sparsemax",
]
