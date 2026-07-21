"""Horizon-invariant Level and Trend adapters at their native physical scales.

Level is an H96-scoped adapter whose output consists of eight patch-local
(12-step) constant coordinates.  Its learned magnitude body intentionally
mirrors the accepted H96 Level specialist: separate input history and
frozen-base-patch encoders followed by a non-negative head.  Every complete
H96 lead block owns an independent fitted sign/magnitude state; sign and
execution routing are fitted separately by the caller.

Trend is a 96-step zero-mean linear correction.  Its compact normalized MLP
mirrors the accepted H96 absolute-state Trend candidate.  Long forecasts reuse
the same architecture for every physical 96-step block, while every complete
block owns an independently fitted parameter/optimizer/checkpoint state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class PhysicalScalarComponents:
    coordinate: torch.Tensor
    correction: torch.Tensor


@dataclass(frozen=True)
class DecomposedLevelOutput:
    sign_logit: torch.Tensor
    sign: torch.Tensor
    magnitude: torch.Tensor
    coordinate: torch.Tensor


def fixed_patch_level_component(
    values: torch.Tensor,
    *,
    patch_steps: int = 12,
) -> PhysicalScalarComponents:
    """Return independent constant Level coordinates for fixed-size patches."""

    if values.ndim != 2 or int(values.shape[1]) % int(patch_steps) != 0:
        raise ValueError("Level values must have shape [B,H] with patch_steps dividing H")
    if patch_steps <= 0:
        raise ValueError("patch_steps must be positive")
    patches = values.reshape(values.shape[0], -1, patch_steps)
    coordinate = patches.mean(dim=-1)
    correction = coordinate.unsqueeze(-1).expand_as(patches).reshape_as(values)
    return PhysicalScalarComponents(coordinate=coordinate, correction=correction)


def fixed_block_trend_component(
    values: torch.Tensor,
    *,
    block_steps: int = 96,
) -> PhysicalScalarComponents:
    """Return one zero-mean linear Trend coordinate per physical-time block."""

    if values.ndim != 2:
        raise ValueError("Trend values must have shape [B,H]")
    if block_steps <= 1:
        raise ValueError("block_steps must exceed one")
    coordinates = []
    correction = torch.empty_like(values)
    for left in range(0, int(values.shape[1]), block_steps):
        right = min(left + block_steps, int(values.shape[1]))
        block = values[:, left:right]
        basis = torch.linspace(
            -1.0, 1.0, right - left, device=values.device, dtype=values.dtype
        )
        basis = basis - basis.mean()
        centered = block - block.mean(dim=-1, keepdim=True)
        coordinate = (centered * basis).sum(dim=-1) / basis.square().sum().clamp_min(1.0e-12)
        coordinates.append(coordinate)
        correction[:, left:right] = coordinate.unsqueeze(-1) * basis.unsqueeze(0)
    return PhysicalScalarComponents(
        coordinate=torch.stack(coordinates, dim=1), correction=correction
    )


def reconstruct_patch_level(
    coordinate: torch.Tensor,
    *,
    patch_steps: int = 12,
) -> torch.Tensor:
    if coordinate.ndim != 2 or patch_steps <= 0:
        raise ValueError("Level coordinate must have shape [B,P]")
    return coordinate.unsqueeze(-1).expand(-1, -1, patch_steps).reshape(
        coordinate.shape[0], coordinate.shape[1] * patch_steps
    )


def reconstruct_block_trend(
    coordinate: torch.Tensor,
    horizon: int,
    *,
    block_steps: int = 96,
) -> torch.Tensor:
    if coordinate.ndim != 2 or horizon <= 1 or block_steps <= 1:
        raise ValueError("invalid Trend reconstruction dimensions")
    expected = (int(horizon) + block_steps - 1) // block_steps
    if int(coordinate.shape[1]) != expected:
        raise ValueError("Trend block count mismatch")
    result = coordinate.new_empty((coordinate.shape[0], int(horizon)))
    for block, left in enumerate(range(0, int(horizon), block_steps)):
        right = min(left + block_steps, int(horizon))
        basis = torch.linspace(
            -1.0, 1.0, right - left, device=coordinate.device, dtype=coordinate.dtype
        )
        basis = basis - basis.mean()
        result[:, left:right] = coordinate[:, block, None] * basis[None]
    return result


class FixedPatchLevelMagnitudeAdapter(nn.Module):
    """H96-compatible non-negative Level magnitude body for one p12 patch."""

    def __init__(
        self,
        initial_magnitude: float,
        *,
        input_steps: int = 96,
        patch_steps: int = 12,
        hidden: int = 32,
    ) -> None:
        super().__init__()
        if initial_magnitude <= 0.0 or min(input_steps, patch_steps, hidden) <= 0:
            raise ValueError("invalid Level magnitude dimensions")
        self.input_steps = int(input_steps)
        self.patch_steps = int(patch_steps)
        self.feature_width = self.input_steps + 4 + self.patch_steps + 6
        self.history_encoder = nn.Linear(self.input_steps + 4, hidden)
        self.patch_encoder = nn.Linear(self.patch_steps + 6, hidden)
        self.joint_encoder = nn.Linear(2 * hidden, hidden)
        self.magnitude_head = nn.Linear(hidden, 1)
        for layer in (self.history_encoder, self.patch_encoder, self.joint_encoder):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.magnitude_head.weight)
        nn.init.constant_(
            self.magnitude_head.bias, math.log(math.expm1(float(initial_magnitude)))
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or int(features.shape[1]) != self.feature_width:
            raise ValueError("Level magnitude feature shape mismatch")
        split = self.input_steps + 4
        history = F.gelu(self.history_encoder(features[:, :split]))
        patch = F.gelu(self.patch_encoder(features[:, split:]))
        joint = F.gelu(self.joint_encoder(torch.cat([history, patch], dim=-1)))
        return F.softplus(self.magnitude_head(joint).squeeze(-1))


class UniversalPeriodicLevelMagnitudeAdapter(FixedPatchLevelMagnitudeAdapter):
    """Locked canonical Level magnitude kernel for every physical clock.

    Native P96/p12 and P24/p3 observations are converted outside this module
    to the same 96-history/12-patch feature contract.  Keeping the canonical
    dimensions private prevents a dataset or forecast horizon from silently
    changing the learned architecture while preserving exact state-dict
    compatibility with the accepted ETTm1 Level magnitude body.
    """

    def __init__(self, initial_magnitude: float) -> None:
        super().__init__(
            initial_magnitude,
            input_steps=96,
            patch_steps=12,
            hidden=32,
        )


class UniversalPeriodicLevelSignAdapter(nn.Module):
    """Fixed cross-dataset signed-confidence head for one physical period.

    The input is the universal 601D :func:`period_level_features` contract.
    Its internal grouping is fixed: one canonical history, eight shared patch
    views, semantic moments, context, feedback moments, and an eight-slot mask.
    Native period length, horizon, channel count, dataset identity, and output
    block count never enter the constructor or learned shapes.

    The bounded output is a signed confidence in ``[-1, 1]``.  The enclosing
    Level expert multiplies it by its independently frozen magnitude.  The
    final head is zero initialized, making an unfitted/rejected sign adapter an
    exact Level NOOP instead of an arbitrary 0.5 classifier probability.
    """

    feature_width = 601
    history_width = 100
    patch_views = 8
    patch_width = 18
    semantic_width = 72
    context_width = 147
    feedback_width = 130
    mask_width = 8

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("feature_center", torch.zeros(self.feature_width))
        self.register_buffer("feature_scale", torch.ones(self.feature_width))
        self.history_encoder = nn.Linear(self.history_width, 32)
        self.patch_encoder = nn.Linear(self.patch_width, 16)
        self.semantic_encoder = nn.Linear(self.semantic_width, 16)
        self.context_encoder = nn.Linear(self.context_width, 16)
        self.feedback_encoder = nn.Linear(self.feedback_width, 16)
        self.body = nn.Sequential(
            nn.Linear(96, 48),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(48, 24),
            nn.SiLU(),
        )
        self.sign_head = nn.Linear(24, 1)
        for layer in (
            self.history_encoder,
            self.patch_encoder,
            self.semantic_encoder,
            self.context_encoder,
            self.feedback_encoder,
            self.body[0],
            self.body[3],
        ):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.sign_head.weight)
        nn.init.zeros_(self.sign_head.bias)

    @torch.no_grad()
    def set_normalization(
        self,
        center: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        center = torch.as_tensor(center, dtype=self.feature_center.dtype)
        scale = torch.as_tensor(scale, dtype=self.feature_scale.dtype)
        if center.shape != self.feature_center.shape or scale.shape != self.feature_scale.shape:
            raise ValueError("Level sign normalization shape mismatch")
        if not torch.isfinite(center).all() or not torch.isfinite(scale).all():
            raise ValueError("Level sign normalization must be finite")
        self.feature_center.copy_(center)
        self.feature_scale.copy_(scale.clamp_min(1.0e-6))

    def forward_logit(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or int(features.shape[1]) != self.feature_width:
            raise ValueError("Level sign feature shape mismatch")
        value = ((features - self.feature_center) / self.feature_scale).clamp(-6.0, 6.0)
        value = torch.nan_to_num(value, nan=0.0, posinf=6.0, neginf=-6.0)
        history_end = self.history_width
        patch_end = history_end + self.patch_views * self.patch_width
        semantic_end = patch_end + self.semantic_width
        context_end = semantic_end + self.context_width
        feedback_end = context_end + self.feedback_width
        if feedback_end + self.mask_width != self.feature_width:
            raise RuntimeError("Level sign feature partition drift")

        history = F.silu(self.history_encoder(value[:, :history_end]))
        patch = value[:, history_end:patch_end].reshape(
            -1, self.patch_views, self.patch_width
        )
        mask = features[:, feedback_end:].clamp(0.0, 1.0)
        patch = F.silu(self.patch_encoder(patch))
        patch = (patch * mask[:, :, None]).sum(dim=1)
        patch = patch / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        semantic = F.silu(
            self.semantic_encoder(value[:, patch_end:semantic_end])
        )
        context = F.silu(self.context_encoder(value[:, semantic_end:context_end]))
        feedback = F.silu(
            self.feedback_encoder(value[:, context_end:feedback_end])
        )
        hidden = self.body(
            torch.cat([history, patch, semantic, context, feedback], dim=-1)
        )
        return self.sign_head(hidden).squeeze(-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Exact identity with ``2 * sigmoid(logit) - 1`` gives the same signed
        # confidence semantics as the historical HGB sign probability.
        return torch.tanh(0.5 * self.forward_logit(features))


class UniversalPeriodicLevelSignResidualAdapter(nn.Module):
    """Bounded universal correction around a frozen Level-sign prior.

    ``base_logit`` is produced by the independently fitted universal 601D HGB
    sign model.  This module owns only a bounded residual logit; it cannot
    replace the prior or change Level magnitude.  The residual head is zero
    initialized, so an unfitted or rejected checkpoint is an exact identity
    mapping of the frozen HGB signed confidence.

    Native sampling clock, forecast horizon, dataset identity, channel count,
    and physical-period count never enter this fixed kernel.
    """

    feature_width = 601
    history_width = 100
    patch_views = 8
    patch_width = 18
    semantic_width = 72
    context_width = 147
    feedback_width = 130
    mask_width = 8
    max_abs_logit_delta = 1.0

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("feature_center", torch.zeros(self.feature_width))
        self.register_buffer("feature_scale", torch.ones(self.feature_width))
        self.history_encoder = nn.Linear(self.history_width, 16)
        self.patch_encoder = nn.Linear(self.patch_width, 8)
        self.semantic_encoder = nn.Linear(self.semantic_width, 8)
        self.context_encoder = nn.Linear(self.context_width, 8)
        self.feedback_encoder = nn.Linear(self.feedback_width, 8)
        self.prior_encoder = nn.Linear(3, 8)
        self.body = nn.Sequential(
            nn.Linear(56, 32),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(32, 16),
            nn.SiLU(),
        )
        self.residual_head = nn.Linear(16, 1)
        for layer in (
            self.history_encoder,
            self.patch_encoder,
            self.semantic_encoder,
            self.context_encoder,
            self.feedback_encoder,
            self.prior_encoder,
            self.body[0],
            self.body[3],
        ):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    @torch.no_grad()
    def set_normalization(
        self,
        center: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        center = torch.as_tensor(center, dtype=self.feature_center.dtype)
        scale = torch.as_tensor(scale, dtype=self.feature_scale.dtype)
        if center.shape != self.feature_center.shape or scale.shape != self.feature_scale.shape:
            raise ValueError("Level sign residual normalization shape mismatch")
        if not torch.isfinite(center).all() or not torch.isfinite(scale).all():
            raise ValueError("Level sign residual normalization must be finite")
        self.feature_center.copy_(center)
        self.feature_scale.copy_(scale.clamp_min(1.0e-6))

    def _validate_inputs(
        self,
        features: torch.Tensor,
        base_logit: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 2 or int(features.shape[1]) != self.feature_width:
            raise ValueError("Level sign residual feature shape mismatch")
        if base_logit.ndim != 1 or int(base_logit.shape[0]) != int(features.shape[0]):
            raise ValueError("Level sign residual base logit shape mismatch")
        return torch.nan_to_num(base_logit, nan=0.0, posinf=12.0, neginf=-12.0).clamp(
            -12.0, 12.0
        )

    def forward_delta(
        self,
        features: torch.Tensor,
        base_logit: torch.Tensor,
    ) -> torch.Tensor:
        prior_logit = self._validate_inputs(features, base_logit)
        value = ((features - self.feature_center) / self.feature_scale).clamp(-6.0, 6.0)
        value = torch.nan_to_num(value, nan=0.0, posinf=6.0, neginf=-6.0)
        history_end = self.history_width
        patch_end = history_end + self.patch_views * self.patch_width
        semantic_end = patch_end + self.semantic_width
        context_end = semantic_end + self.context_width
        feedback_end = context_end + self.feedback_width
        if feedback_end + self.mask_width != self.feature_width:
            raise RuntimeError("Level sign residual feature partition drift")

        history = F.silu(self.history_encoder(value[:, :history_end]))
        patch = value[:, history_end:patch_end].reshape(
            -1, self.patch_views, self.patch_width
        )
        mask = features[:, feedback_end:].clamp(0.0, 1.0)
        patch = F.silu(self.patch_encoder(patch))
        patch = (patch * mask[:, :, None]).sum(dim=1)
        patch = patch / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        semantic = F.silu(self.semantic_encoder(value[:, patch_end:semantic_end]))
        context = F.silu(self.context_encoder(value[:, semantic_end:context_end]))
        feedback = F.silu(self.feedback_encoder(value[:, context_end:feedback_end]))
        prior = torch.stack(
            [prior_logit / 4.0, prior_logit.abs() / 4.0, torch.tanh(0.5 * prior_logit)],
            dim=-1,
        )
        prior = F.silu(self.prior_encoder(prior))
        hidden = self.body(
            torch.cat([history, patch, semantic, context, feedback, prior], dim=-1)
        )
        return self.max_abs_logit_delta * torch.tanh(
            self.residual_head(hidden).squeeze(-1)
        )

    def forward_logit(
        self,
        features: torch.Tensor,
        base_logit: torch.Tensor,
    ) -> torch.Tensor:
        prior_logit = self._validate_inputs(features, base_logit)
        return prior_logit + self.forward_delta(features, prior_logit)

    def forward(
        self,
        features: torch.Tensor,
        base_logit: torch.Tensor,
    ) -> torch.Tensor:
        return torch.tanh(0.5 * self.forward_logit(features, base_logit))


class UniversalPeriodicLevelRevisionSignResidualAdapter(nn.Module):
    """Universal HGB-logit residual conditioned on forecast revisions.

    This v2 kernel is deliberately separate from
    :class:`UniversalPeriodicLevelSignResidualAdapter`.  It adds one fixed 24D
    target-free state built from seven overlapping frozen-backbone forecast
    revisions, while preserving the accepted 601D Level contract and frozen
    HGB prior.  A zero residual head gives a bit-exact HGB identity fallback.
    """

    feature_width = 601
    revision_width = 24
    history_width = 100
    patch_views = 8
    patch_width = 18
    semantic_width = 72
    context_width = 147
    feedback_width = 130
    mask_width = 8
    max_abs_logit_delta = 1.0

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("feature_center", torch.zeros(self.feature_width))
        self.register_buffer("feature_scale", torch.ones(self.feature_width))
        self.register_buffer("revision_center", torch.zeros(self.revision_width))
        self.register_buffer("revision_scale", torch.ones(self.revision_width))
        self.history_encoder = nn.Linear(self.history_width, 16)
        self.patch_encoder = nn.Linear(self.patch_width, 8)
        self.semantic_encoder = nn.Linear(self.semantic_width, 8)
        self.context_encoder = nn.Linear(self.context_width, 8)
        self.feedback_encoder = nn.Linear(self.feedback_width, 8)
        self.prior_encoder = nn.Linear(3, 8)
        self.revision_encoder = nn.Linear(self.revision_width, 8)
        self.body = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(32, 16),
            nn.SiLU(),
        )
        self.residual_head = nn.Linear(16, 1)
        for layer in (
            self.history_encoder,
            self.patch_encoder,
            self.semantic_encoder,
            self.context_encoder,
            self.feedback_encoder,
            self.prior_encoder,
            self.revision_encoder,
            self.body[0],
            self.body[3],
        ):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    @torch.no_grad()
    def set_normalization(
        self,
        feature_center: torch.Tensor,
        feature_scale: torch.Tensor,
        revision_center: torch.Tensor,
        revision_scale: torch.Tensor,
    ) -> None:
        feature_center = torch.as_tensor(
            feature_center, dtype=self.feature_center.dtype
        )
        feature_scale = torch.as_tensor(
            feature_scale, dtype=self.feature_scale.dtype
        )
        revision_center = torch.as_tensor(
            revision_center, dtype=self.revision_center.dtype
        )
        revision_scale = torch.as_tensor(
            revision_scale, dtype=self.revision_scale.dtype
        )
        if (
            feature_center.shape != self.feature_center.shape
            or feature_scale.shape != self.feature_scale.shape
            or revision_center.shape != self.revision_center.shape
            or revision_scale.shape != self.revision_scale.shape
        ):
            raise ValueError("Level revision residual normalization shape mismatch")
        if not all(
            torch.isfinite(value).all()
            for value in (
                feature_center,
                feature_scale,
                revision_center,
                revision_scale,
            )
        ):
            raise ValueError("Level revision residual normalization must be finite")
        self.feature_center.copy_(feature_center)
        self.feature_scale.copy_(feature_scale.clamp_min(1.0e-6))
        self.revision_center.copy_(revision_center)
        self.revision_scale.copy_(revision_scale.clamp_min(1.0e-6))

    def _validate_inputs(
        self,
        features: torch.Tensor,
        revision_state: torch.Tensor,
        base_logit: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 2 or int(features.shape[1]) != self.feature_width:
            raise ValueError("Level revision residual feature shape mismatch")
        if (
            revision_state.ndim != 2
            or int(revision_state.shape[0]) != int(features.shape[0])
            or int(revision_state.shape[1]) != self.revision_width
        ):
            raise ValueError("Level revision residual state shape mismatch")
        if base_logit.ndim != 1 or int(base_logit.shape[0]) != int(features.shape[0]):
            raise ValueError("Level revision residual base logit shape mismatch")
        return torch.nan_to_num(base_logit, nan=0.0, posinf=12.0, neginf=-12.0).clamp(
            -12.0, 12.0
        )

    def forward_delta(
        self,
        features: torch.Tensor,
        revision_state: torch.Tensor,
        base_logit: torch.Tensor,
    ) -> torch.Tensor:
        prior_logit = self._validate_inputs(features, revision_state, base_logit)
        value = ((features - self.feature_center) / self.feature_scale).clamp(-6.0, 6.0)
        value = torch.nan_to_num(value, nan=0.0, posinf=6.0, neginf=-6.0)
        revision = (
            (revision_state - self.revision_center) / self.revision_scale
        ).clamp(-6.0, 6.0)
        revision = torch.nan_to_num(revision, nan=0.0, posinf=6.0, neginf=-6.0)
        history_end = self.history_width
        patch_end = history_end + self.patch_views * self.patch_width
        semantic_end = patch_end + self.semantic_width
        context_end = semantic_end + self.context_width
        feedback_end = context_end + self.feedback_width
        if feedback_end + self.mask_width != self.feature_width:
            raise RuntimeError("Level revision residual feature partition drift")

        history = F.silu(self.history_encoder(value[:, :history_end]))
        patch = value[:, history_end:patch_end].reshape(
            -1, self.patch_views, self.patch_width
        )
        mask = features[:, feedback_end:].clamp(0.0, 1.0)
        patch = F.silu(self.patch_encoder(patch))
        patch = (patch * mask[:, :, None]).sum(dim=1)
        patch = patch / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        semantic = F.silu(self.semantic_encoder(value[:, patch_end:semantic_end]))
        context = F.silu(self.context_encoder(value[:, semantic_end:context_end]))
        feedback = F.silu(self.feedback_encoder(value[:, context_end:feedback_end]))
        prior = torch.stack(
            [prior_logit / 4.0, prior_logit.abs() / 4.0, torch.tanh(0.5 * prior_logit)],
            dim=-1,
        )
        prior = F.silu(self.prior_encoder(prior))
        revision = F.silu(self.revision_encoder(revision))
        hidden = self.body(
            torch.cat(
                [history, patch, semantic, context, feedback, prior, revision],
                dim=-1,
            )
        )
        return self.max_abs_logit_delta * torch.tanh(
            self.residual_head(hidden).squeeze(-1)
        )

    def forward_logit(
        self,
        features: torch.Tensor,
        revision_state: torch.Tensor,
        base_logit: torch.Tensor,
    ) -> torch.Tensor:
        prior_logit = self._validate_inputs(features, revision_state, base_logit)
        return prior_logit + self.forward_delta(
            features, revision_state, prior_logit
        )

    def forward(
        self,
        features: torch.Tensor,
        revision_state: torch.Tensor,
        base_logit: torch.Tensor,
    ) -> torch.Tensor:
        return torch.tanh(
            0.5 * self.forward_logit(features, revision_state, base_logit)
        )


class UniversalPeriodicLevelOrderedSignAdapter(nn.Module):
    """Fixed Level-sign head that retains eight-view matured-feedback order.

    The 991D input is produced by
    :func:`period_level_ordered_sign_features` for every dataset and horizon.
    One shared 65D feedback encoder is applied to each canonical phase view;
    the eight encoded views are then kept in order for the joint body.  Native
    clock, horizon, channel count, dataset identity, and physical-period count
    never enter the constructor or learned shapes.
    """

    feature_width = 991
    history_width = 100
    patch_views = 8
    patch_width = 18
    semantic_width = 72
    context_width = 147
    feedback_views = 8
    feedback_width = 65
    mask_width = 8

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("feature_center", torch.zeros(self.feature_width))
        self.register_buffer("feature_scale", torch.ones(self.feature_width))
        self.history_encoder = nn.Linear(self.history_width, 32)
        self.patch_encoder = nn.Linear(self.patch_width, 16)
        self.semantic_encoder = nn.Linear(self.semantic_width, 16)
        self.context_encoder = nn.Linear(self.context_width, 16)
        self.feedback_encoder = nn.Linear(self.feedback_width, 16)
        self.body = nn.Sequential(
            nn.Linear(208, 64),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 32),
            nn.SiLU(),
        )
        self.sign_head = nn.Linear(32, 1)
        for layer in (
            self.history_encoder,
            self.patch_encoder,
            self.semantic_encoder,
            self.context_encoder,
            self.feedback_encoder,
            self.body[0],
            self.body[3],
        ):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.sign_head.weight)
        nn.init.zeros_(self.sign_head.bias)

    @torch.no_grad()
    def set_normalization(
        self,
        center: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        center = torch.as_tensor(center, dtype=self.feature_center.dtype)
        scale = torch.as_tensor(scale, dtype=self.feature_scale.dtype)
        if center.shape != self.feature_center.shape or scale.shape != self.feature_scale.shape:
            raise ValueError("ordered Level sign normalization shape mismatch")
        if not torch.isfinite(center).all() or not torch.isfinite(scale).all():
            raise ValueError("ordered Level sign normalization must be finite")
        self.feature_center.copy_(center)
        self.feature_scale.copy_(scale.clamp_min(1.0e-6))

    def forward_logit(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or int(features.shape[1]) != self.feature_width:
            raise ValueError("ordered Level sign feature shape mismatch")
        value = ((features - self.feature_center) / self.feature_scale).clamp(-6.0, 6.0)
        value = torch.nan_to_num(value, nan=0.0, posinf=6.0, neginf=-6.0)
        history_end = self.history_width
        patch_end = history_end + self.patch_views * self.patch_width
        semantic_end = patch_end + self.semantic_width
        context_end = semantic_end + self.context_width
        feedback_end = context_end + self.feedback_views * self.feedback_width
        if feedback_end + self.mask_width != self.feature_width:
            raise RuntimeError("ordered Level sign feature partition drift")

        mask = features[:, feedback_end:].clamp(0.0, 1.0)
        history = F.silu(self.history_encoder(value[:, :history_end]))
        patch = value[:, history_end:patch_end].reshape(
            -1, self.patch_views, self.patch_width
        )
        patch = F.silu(self.patch_encoder(patch))
        patch = (patch * mask[:, :, None]).sum(dim=1)
        patch = patch / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        semantic = F.silu(self.semantic_encoder(value[:, patch_end:semantic_end]))
        context = F.silu(self.context_encoder(value[:, semantic_end:context_end]))
        feedback = value[:, context_end:feedback_end].reshape(
            -1, self.feedback_views, self.feedback_width
        )
        feedback = F.silu(self.feedback_encoder(feedback))
        feedback = (feedback * mask[:, :, None]).reshape(features.shape[0], -1)
        hidden = self.body(
            torch.cat([history, patch, semantic, context, feedback], dim=-1)
        )
        return self.sign_head(hidden).squeeze(-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tanh(0.5 * self.forward_logit(features))


class UniversalPeriodicLevelPhaseConvSignAdapter(nn.Module):
    """Translation-shared local phase-coherence Level-sign head.

    This consumes the same fixed 991D ordered feature contract as
    :class:`UniversalPeriodicLevelOrderedSignAdapter`, but the joint body never
    receives position-specific phase lanes.  One shared kernel-3 convolution
    is reduced by masked mean/max pooling, providing a dataset-free local-phase
    inductive bias while preventing an eight-position shortcut table.
    """

    feature_width = 991
    history_width = 100
    patch_views = 8
    patch_width = 18
    semantic_width = 72
    context_width = 147
    feedback_views = 8
    feedback_width = 65
    mask_width = 8

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("feature_center", torch.zeros(self.feature_width))
        self.register_buffer("feature_scale", torch.ones(self.feature_width))
        self.history_encoder = nn.Linear(self.history_width, 32)
        self.patch_encoder = nn.Linear(self.patch_width, 16)
        self.semantic_encoder = nn.Linear(self.semantic_width, 16)
        self.context_encoder = nn.Linear(self.context_width, 16)
        self.feedback_encoder = nn.Linear(self.feedback_width, 16)
        self.feedback_phase_conv = nn.Conv1d(16, 16, kernel_size=3, padding=1)
        self.body = nn.Sequential(
            nn.Linear(112, 48),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(48, 24),
            nn.SiLU(),
        )
        self.sign_head = nn.Linear(24, 1)
        for layer in (
            self.history_encoder,
            self.patch_encoder,
            self.semantic_encoder,
            self.context_encoder,
            self.feedback_encoder,
            self.body[0],
            self.body[3],
        ):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.feedback_phase_conv.weight)
        nn.init.zeros_(self.feedback_phase_conv.bias)
        nn.init.zeros_(self.sign_head.weight)
        nn.init.zeros_(self.sign_head.bias)

    @torch.no_grad()
    def set_normalization(
        self,
        center: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        center = torch.as_tensor(center, dtype=self.feature_center.dtype)
        scale = torch.as_tensor(scale, dtype=self.feature_scale.dtype)
        if center.shape != self.feature_center.shape or scale.shape != self.feature_scale.shape:
            raise ValueError("phase-conv Level sign normalization shape mismatch")
        if not torch.isfinite(center).all() or not torch.isfinite(scale).all():
            raise ValueError("phase-conv Level sign normalization must be finite")
        self.feature_center.copy_(center)
        self.feature_scale.copy_(scale.clamp_min(1.0e-6))

    def forward_logit(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or int(features.shape[1]) != self.feature_width:
            raise ValueError("phase-conv Level sign feature shape mismatch")
        value = ((features - self.feature_center) / self.feature_scale).clamp(-6.0, 6.0)
        value = torch.nan_to_num(value, nan=0.0, posinf=6.0, neginf=-6.0)
        history_end = self.history_width
        patch_end = history_end + self.patch_views * self.patch_width
        semantic_end = patch_end + self.semantic_width
        context_end = semantic_end + self.context_width
        feedback_end = context_end + self.feedback_views * self.feedback_width
        if feedback_end + self.mask_width != self.feature_width:
            raise RuntimeError("phase-conv Level sign feature partition drift")

        mask = features[:, feedback_end:].clamp(0.0, 1.0)
        history = F.silu(self.history_encoder(value[:, :history_end]))
        patch = value[:, history_end:patch_end].reshape(
            -1, self.patch_views, self.patch_width
        )
        patch = F.silu(self.patch_encoder(patch))
        patch = (patch * mask[:, :, None]).sum(dim=1)
        patch = patch / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        semantic = F.silu(self.semantic_encoder(value[:, patch_end:semantic_end]))
        context = F.silu(self.context_encoder(value[:, semantic_end:context_end]))
        feedback = value[:, context_end:feedback_end].reshape(
            -1, self.feedback_views, self.feedback_width
        )
        feedback = F.silu(self.feedback_encoder(feedback)) * mask[:, :, None]
        feedback = F.silu(self.feedback_phase_conv(feedback.transpose(1, 2))).transpose(1, 2)
        feedback_mean = (feedback * mask[:, :, None]).sum(dim=1)
        feedback_mean = feedback_mean / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        feedback_max = feedback.masked_fill(mask[:, :, None] <= 0.0, -torch.inf).amax(dim=1)
        feedback_max = torch.nan_to_num(feedback_max, neginf=0.0)
        hidden = self.body(
            torch.cat(
                [history, patch, semantic, context, feedback_mean, feedback_max],
                dim=-1,
            )
        )
        return self.sign_head(hidden).squeeze(-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tanh(0.5 * self.forward_logit(features))


class UniversalPeriodicLevelReliabilityGate(nn.Module):
    """Dataset/horizon-free local suppression gate for frozen Level output.

    The 608D input is the fixed 601D physical-period Level state followed by
    seven target-free frozen-candidate summaries.  Native period length and
    forecast block count are intentionally absent from the constructor.  The
    gate owns only a bounded ``[0, 1]`` reliability coefficient and therefore
    cannot amplify or modify the isolated Level/Amp experts.
    """

    feature_width = 608

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("feature_center", torch.zeros(self.feature_width))
        self.register_buffer("feature_scale", torch.ones(self.feature_width))
        self.encoder = nn.Linear(self.feature_width, 32)
        self.body = nn.Linear(32, 16)
        self.reliability_head = nn.Linear(16, 1)
        for layer in (self.encoder, self.body):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.reliability_head.weight)
        nn.init.constant_(self.reliability_head.bias, math.log(9.0))

    @torch.no_grad()
    def set_normalization(
        self,
        center: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        center = torch.as_tensor(center, dtype=self.feature_center.dtype)
        scale = torch.as_tensor(scale, dtype=self.feature_scale.dtype)
        if center.shape != self.feature_center.shape or scale.shape != self.feature_scale.shape:
            raise ValueError("Level reliability normalization shape mismatch")
        if not torch.isfinite(center).all() or not torch.isfinite(scale).all():
            raise ValueError("Level reliability normalization must be finite")
        self.feature_center.copy_(center)
        self.feature_scale.copy_(scale.clamp_min(1.0e-6))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or int(features.shape[1]) != self.feature_width:
            raise ValueError("Level reliability feature shape mismatch")
        normalized = (features - self.feature_center) / self.feature_scale
        hidden = F.silu(self.encoder(normalized))
        hidden = F.silu(self.body(hidden))
        return torch.sigmoid(self.reliability_head(hidden).squeeze(-1))


class DecomposedFeedbackLevelAdapter(nn.Module):
    """Self-amplifying p12 Level adapter with causal decomposed feedback.

    The adapter consumes the existing 118D observed-history/base-patch contract
    plus a strictly matured Level-residual feedback vector.  It owns both sign
    and bounded magnitude; a downstream gate may select the correction but must
    not rescale it.
    """

    def __init__(
        self,
        feedback_width: int,
        initial_magnitude: float,
        max_abs_coordinate: float,
        *,
        input_steps: int = 96,
        patch_steps: int = 12,
        hidden: int = 48,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if (
            feedback_width <= 0
            or initial_magnitude <= 0.0
            or max_abs_coordinate <= initial_magnitude
            or min(input_steps, patch_steps, hidden) <= 0
        ):
            raise ValueError("invalid decomposed Level adapter dimensions")
        self.input_steps = int(input_steps)
        self.patch_steps = int(patch_steps)
        self.raw_width = self.input_steps + 4 + self.patch_steps + 6
        self.feedback_width = int(feedback_width)
        self.feature_width = self.raw_width + self.feedback_width
        self.max_abs_coordinate = float(max_abs_coordinate)
        self.register_buffer("feature_center", torch.zeros(self.feature_width))
        self.register_buffer("feature_scale", torch.ones(self.feature_width))
        self.history_encoder = nn.Linear(self.input_steps + 4, hidden)
        self.patch_encoder = nn.Linear(self.patch_steps + 6, hidden)
        self.feedback_encoder = nn.Linear(self.feedback_width, hidden)
        self.body = nn.Sequential(
            nn.Linear(3 * hidden, 2 * hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
        )
        self.sign_head = nn.Linear(hidden, 1)
        self.magnitude_head = nn.Linear(hidden, 1)
        for layer in (
            self.history_encoder,
            self.patch_encoder,
            self.feedback_encoder,
            self.body[0],
            self.body[3],
        ):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.sign_head.weight)
        nn.init.zeros_(self.sign_head.bias)
        nn.init.zeros_(self.magnitude_head.weight)
        fraction = float(initial_magnitude) / float(max_abs_coordinate)
        fraction = min(max(fraction, 1.0e-4), 1.0 - 1.0e-4)
        nn.init.constant_(self.magnitude_head.bias, math.log(fraction / (1.0 - fraction)))

    def set_normalization(self, center: torch.Tensor, scale: torch.Tensor) -> None:
        if center.shape != (self.feature_width,) or scale.shape != center.shape:
            raise ValueError("Level normalization shape mismatch")
        with torch.no_grad():
            self.feature_center.copy_(center)
            self.feature_scale.copy_(scale.clamp_min(1.0e-5))

    def forward_components(self, features: torch.Tensor) -> DecomposedLevelOutput:
        if features.ndim != 2 or int(features.shape[1]) != self.feature_width:
            raise ValueError("decomposed Level feature shape mismatch")
        value = ((features - self.feature_center) / self.feature_scale).clamp(-6.0, 6.0)
        value = torch.nan_to_num(value, nan=0.0, posinf=6.0, neginf=-6.0)
        history_end = self.input_steps + 4
        patch_end = self.raw_width
        history = F.gelu(self.history_encoder(value[:, :history_end]))
        patch = F.gelu(self.patch_encoder(value[:, history_end:patch_end]))
        feedback = F.gelu(self.feedback_encoder(value[:, patch_end:]))
        hidden = self.body(torch.cat([history, patch, feedback], dim=-1))
        sign_logit = self.sign_head(hidden).squeeze(-1)
        sign = torch.tanh(sign_logit)
        magnitude = self.max_abs_coordinate * torch.sigmoid(
            self.magnitude_head(hidden).squeeze(-1)
        )
        return DecomposedLevelOutput(
            sign_logit=sign_logit,
            sign=sign,
            magnitude=magnitude,
            coordinate=sign * magnitude,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.forward_components(features).coordinate


class DecomposedFeedbackLevelMagnitudeAdapter(nn.Module):
    """Bounded Level magnitude specialist over raw118 plus matured feedback.

    This is a parameter-isolated magnitude branch.  It predicts neither sign
    nor gate strength; the enclosing Level adapter composes it with its own
    separately fitted sign probability and fixed internal output scale.
    """

    def __init__(
        self,
        feedback_width: int,
        initial_magnitude: float,
        max_magnitude: float,
        *,
        input_steps: int = 96,
        patch_steps: int = 12,
        hidden: int = 48,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if (
            feedback_width <= 0
            or initial_magnitude <= 0.0
            or max_magnitude <= initial_magnitude
            or min(input_steps, patch_steps, hidden) <= 0
        ):
            raise ValueError("invalid decomposed Level magnitude dimensions")
        self.input_steps = int(input_steps)
        self.patch_steps = int(patch_steps)
        self.raw_width = self.input_steps + 4 + self.patch_steps + 6
        self.feedback_width = int(feedback_width)
        self.feature_width = self.raw_width + self.feedback_width
        self.max_magnitude = float(max_magnitude)
        self.register_buffer("feature_center", torch.zeros(self.feature_width))
        self.register_buffer("feature_scale", torch.ones(self.feature_width))
        self.history_encoder = nn.Linear(self.input_steps + 4, hidden)
        self.patch_encoder = nn.Linear(self.patch_steps + 6, hidden)
        self.feedback_encoder = nn.Linear(self.feedback_width, hidden)
        self.body = nn.Sequential(
            nn.Linear(3 * hidden, 2 * hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        for layer in (
            self.history_encoder,
            self.patch_encoder,
            self.feedback_encoder,
            self.body[0],
            self.body[3],
        ):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.body[-1].weight)
        fraction = float(initial_magnitude) / float(max_magnitude)
        fraction = min(max(fraction, 1.0e-4), 1.0 - 1.0e-4)
        nn.init.constant_(self.body[-1].bias, math.log(fraction / (1.0 - fraction)))

    def set_normalization(self, center: torch.Tensor, scale: torch.Tensor) -> None:
        if center.shape != (self.feature_width,) or scale.shape != center.shape:
            raise ValueError("Level magnitude normalization shape mismatch")
        with torch.no_grad():
            self.feature_center.copy_(center)
            self.feature_scale.copy_(scale.clamp_min(1.0e-5))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or int(features.shape[1]) != self.feature_width:
            raise ValueError("decomposed Level magnitude feature shape mismatch")
        value = ((features - self.feature_center) / self.feature_scale).clamp(-6.0, 6.0)
        value = torch.nan_to_num(value, nan=0.0, posinf=6.0, neginf=-6.0)
        history_end = self.input_steps + 4
        patch_end = self.raw_width
        history = F.gelu(self.history_encoder(value[:, :history_end]))
        patch = F.gelu(self.patch_encoder(value[:, history_end:patch_end]))
        feedback = F.gelu(self.feedback_encoder(value[:, patch_end:]))
        raw = self.body(torch.cat([history, patch, feedback], dim=-1)).squeeze(-1)
        return self.max_magnitude * torch.sigmoid(raw)


class AnchoredFeedbackLevelMagnitudeResidualAdapter(nn.Module):
    """Zero-initialized feedback delta around a frozen Level magnitude anchor.

    The raw-only magnitude is supplied by the enclosing Level adapter and is
    never replaced.  This branch sees raw118 plus strictly matured feedback and
    can change the anchor only within ``max_relative_delta``.  It predicts no
    sign and contains no routing/strength parameters.
    """

    def __init__(
        self,
        feedback_width: int,
        *,
        max_relative_delta: float = 0.25,
        input_steps: int = 96,
        patch_steps: int = 12,
        hidden: int = 48,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if (
            feedback_width <= 0
            or not 0.0 < max_relative_delta < 1.0
            or min(input_steps, patch_steps, hidden) <= 0
        ):
            raise ValueError("invalid anchored Level residual dimensions")
        self.input_steps = int(input_steps)
        self.patch_steps = int(patch_steps)
        self.raw_width = self.input_steps + 4 + self.patch_steps + 6
        self.feedback_width = int(feedback_width)
        self.feature_width = self.raw_width + self.feedback_width
        self.max_relative_delta = float(max_relative_delta)
        self.register_buffer("feature_center", torch.zeros(self.feature_width))
        self.register_buffer("feature_scale", torch.ones(self.feature_width))
        self.history_encoder = nn.Linear(self.input_steps + 4, hidden)
        self.patch_encoder = nn.Linear(self.patch_steps + 6, hidden)
        self.feedback_encoder = nn.Linear(self.feedback_width, hidden)
        self.body = nn.Sequential(
            nn.Linear(3 * hidden, 2 * hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        for layer in (
            self.history_encoder,
            self.patch_encoder,
            self.feedback_encoder,
            self.body[0],
            self.body[3],
        ):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        # Exact anchor fallback before any training and for a zero checkpoint.
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def set_normalization(self, center: torch.Tensor, scale: torch.Tensor) -> None:
        if center.shape != (self.feature_width,) or scale.shape != center.shape:
            raise ValueError("anchored Level residual normalization shape mismatch")
        with torch.no_grad():
            self.feature_center.copy_(center)
            self.feature_scale.copy_(scale.clamp_min(1.0e-5))

    def forward(
        self,
        features: torch.Tensor,
        anchor_magnitude: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 2 or int(features.shape[1]) != self.feature_width:
            raise ValueError("anchored Level residual feature shape mismatch")
        if anchor_magnitude.shape != (features.shape[0],):
            raise ValueError("anchored Level magnitude shape mismatch")
        if bool(torch.any(anchor_magnitude < 0.0)):
            raise ValueError("anchored Level magnitude must be nonnegative")
        value = ((features - self.feature_center) / self.feature_scale).clamp(-6.0, 6.0)
        value = torch.nan_to_num(value, nan=0.0, posinf=6.0, neginf=-6.0)
        history_end = self.input_steps + 4
        patch_end = self.raw_width
        history = F.gelu(self.history_encoder(value[:, :history_end]))
        patch = F.gelu(self.patch_encoder(value[:, history_end:patch_end]))
        feedback = F.gelu(self.feedback_encoder(value[:, patch_end:]))
        relative = self.max_relative_delta * torch.tanh(
            self.body(torch.cat([history, patch, feedback], dim=-1)).squeeze(-1)
        )
        return anchor_magnitude * (1.0 + relative)


class AnchoredPhysicalBlockTrendResidualAdapter(nn.Module):
    """Bounded Trend residual around a history-minus-backbone physical anchor.

    The adapter owns no Level, Amp, or gate state.  Its zero-initialized path is
    an exact fixed-strength physical Trend correction, and learning may only
    attenuate or amplify that correction without reversing its sign.
    """

    SEMANTIC_WIDTH = 26
    CALENDAR_WIDTH = 71
    STATE_WIDTH = 9

    def __init__(
        self,
        feature_width: int = 106,
        *,
        anchor_strength: float = 0.125,
        max_relative_delta: float = 1.0,
        hidden: int = 32,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        expected_width = self.SEMANTIC_WIDTH + self.CALENDAR_WIDTH + self.STATE_WIDTH
        if feature_width != expected_width:
            raise ValueError("anchored Trend feature width must be 106")
        if (
            anchor_strength <= 0.0
            or not 0.0 < max_relative_delta <= 1.0
            or hidden <= 0
            or hidden % 2 != 0
        ):
            raise ValueError("invalid anchored Trend adapter configuration")
        self.feature_width = int(feature_width)
        self.anchor_strength = float(anchor_strength)
        self.max_relative_delta = float(max_relative_delta)
        self.register_buffer("feature_center", torch.zeros(feature_width))
        self.register_buffer("feature_scale", torch.ones(feature_width))
        self.semantic_encoder = nn.Linear(self.SEMANTIC_WIDTH, hidden)
        self.calendar_encoder = nn.Linear(self.CALENDAR_WIDTH, hidden // 2)
        self.state_encoder = nn.Linear(self.STATE_WIDTH, hidden // 2)
        self.body = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def set_normalization(self, center: torch.Tensor, scale: torch.Tensor) -> None:
        if center.shape != (self.feature_width,) or scale.shape != center.shape:
            raise ValueError("anchored Trend normalization shape mismatch")
        with torch.no_grad():
            self.feature_center.copy_(center)
            self.feature_scale.copy_(scale.clamp_min(1.0e-5))

    def forward(self, features: torch.Tensor, anchor_coordinate: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or int(features.shape[1]) != self.feature_width:
            raise ValueError("anchored Trend feature shape mismatch")
        if anchor_coordinate.shape != (features.shape[0],):
            raise ValueError("anchored Trend coordinate shape mismatch")
        value = ((features - self.feature_center) / self.feature_scale).clamp(-6.0, 6.0)
        value = torch.nan_to_num(value, nan=0.0, posinf=6.0, neginf=-6.0)
        semantic_end = self.SEMANTIC_WIDTH
        calendar_end = semantic_end + self.CALENDAR_WIDTH
        semantic = F.silu(self.semantic_encoder(value[:, :semantic_end]))
        calendar = F.silu(self.calendar_encoder(value[:, semantic_end:calendar_end]))
        state = F.silu(self.state_encoder(value[:, calendar_end:]))
        relative = self.max_relative_delta * torch.tanh(
            self.body(torch.cat([semantic, calendar, state], dim=-1)).squeeze(-1)
        )
        multiplier = self.anchor_strength * (1.0 + relative)
        return anchor_coordinate * multiplier


class UniversalPeriodicTrendAdapter(nn.Module):
    """Dataset/horizon-free direct Trend-coordinate kernel.

    The historical ETTm1 adapter above is deliberately an anchored multiplier.
    That is a valid locked production implementation, but it cannot transfer to
    a physical clock where the history-minus-backbone anchor has the wrong
    sign.  The universal kernel therefore predicts a signed coordinate directly
    from the same fixed 106D representation.  ``coordinate_scale`` is a causal
    physical scale (normally the history-period standard deviation), so the
    learned action remains bounded and unit-equivariant.

    Native history/base periods are converted outside this module to the shared
    26 semantic + 71 descriptor/calendar + 9 state contract.  Physical periods
    and forecast horizons are batch axes and never constructor fields.  The
    final head is zero-initialized, making the unfitted adapter an exact NOOP.
    """

    SEMANTIC_WIDTH = 26
    CALENDAR_WIDTH = 71
    STATE_WIDTH = 9
    feature_width = 106
    parameter_count = 4289

    def __init__(
        self,
        *,
        max_abs_coordinate: float = 0.40,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if max_abs_coordinate <= 0.0 or not 0.0 <= dropout < 1.0:
            raise ValueError("invalid universal Trend adapter configuration")
        self.max_abs_coordinate = float(max_abs_coordinate)
        self.register_buffer("feature_center", torch.zeros(self.feature_width))
        self.register_buffer("feature_scale", torch.ones(self.feature_width))
        self.semantic_encoder = nn.Linear(self.SEMANTIC_WIDTH, 32)
        self.calendar_encoder = nn.Linear(self.CALENDAR_WIDTH, 16)
        self.state_encoder = nn.Linear(self.STATE_WIDTH, 16)
        self.body = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    @torch.no_grad()
    def set_normalization(self, center: torch.Tensor, scale: torch.Tensor) -> None:
        center = torch.as_tensor(center, dtype=self.feature_center.dtype)
        scale = torch.as_tensor(scale, dtype=self.feature_scale.dtype)
        if center.shape != self.feature_center.shape or scale.shape != center.shape:
            raise ValueError("universal Trend normalization shape mismatch")
        if not torch.isfinite(center).all() or not torch.isfinite(scale).all():
            raise ValueError("universal Trend normalization must be finite")
        self.feature_center.copy_(center)
        self.feature_scale.copy_(scale.clamp_min(1.0e-6))

    def forward(
        self,
        features: torch.Tensor,
        coordinate_scale: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 2 or int(features.shape[1]) != self.feature_width:
            raise ValueError("universal Trend feature shape mismatch")
        if coordinate_scale.shape != (features.shape[0],):
            raise ValueError("universal Trend coordinate-scale shape mismatch")
        if bool(torch.any(coordinate_scale < 0.0)):
            raise ValueError("universal Trend coordinate scale must be nonnegative")
        value = ((features - self.feature_center) / self.feature_scale).clamp(-6.0, 6.0)
        value = torch.nan_to_num(value, nan=0.0, posinf=6.0, neginf=-6.0)
        semantic_end = self.SEMANTIC_WIDTH
        calendar_end = semantic_end + self.CALENDAR_WIDTH
        semantic = F.silu(self.semantic_encoder(value[:, :semantic_end]))
        calendar = F.silu(self.calendar_encoder(value[:, semantic_end:calendar_end]))
        state = F.silu(self.state_encoder(value[:, calendar_end:]))
        raw = self.body(torch.cat([semantic, calendar, state], dim=-1)).squeeze(-1)
        return coordinate_scale * self.max_abs_coordinate * torch.tanh(raw)


class PhysicalBlockTrendAdapter(nn.Module):
    """Compact direct coefficient MLP instantiated independently per H96 block."""

    def __init__(
        self,
        feature_width: int,
        *,
        hidden: int = 32,
        dropout: float = 0.10,
        max_abs_coefficient: float = 3.0,
    ) -> None:
        super().__init__()
        if feature_width <= 0 or hidden <= 0 or max_abs_coefficient <= 0.0:
            raise ValueError("invalid Trend adapter dimensions")
        self.feature_width = int(feature_width)
        self.max_abs_coefficient = float(max_abs_coefficient)
        self.register_buffer("feature_center", torch.zeros(feature_width))
        self.register_buffer("feature_scale", torch.ones(feature_width))
        self.body = nn.Sequential(
            nn.Linear(feature_width, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def set_normalization(self, center: torch.Tensor, scale: torch.Tensor) -> None:
        if center.shape != (self.feature_width,) or scale.shape != center.shape:
            raise ValueError("Trend normalization shape mismatch")
        with torch.no_grad():
            self.feature_center.copy_(center)
            self.feature_scale.copy_(scale.clamp_min(1.0e-5))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or int(features.shape[1]) != self.feature_width:
            raise ValueError("Trend feature shape mismatch")
        value = ((features - self.feature_center) / self.feature_scale).clamp(-6.0, 6.0)
        raw = self.body(torch.nan_to_num(value, nan=0.0, posinf=6.0, neginf=-6.0)).squeeze(-1)
        return self.max_abs_coefficient * torch.tanh(raw / self.max_abs_coefficient)


__all__ = [
    "AnchoredFeedbackLevelMagnitudeResidualAdapter",
    "AnchoredPhysicalBlockTrendResidualAdapter",
    "DecomposedFeedbackLevelAdapter",
    "DecomposedFeedbackLevelMagnitudeAdapter",
    "DecomposedLevelOutput",
    "FixedPatchLevelMagnitudeAdapter",
    "UniversalPeriodicLevelMagnitudeAdapter",
    "UniversalPeriodicLevelRevisionSignResidualAdapter",
    "UniversalPeriodicLevelSignResidualAdapter",
    "UniversalPeriodicLevelReliabilityGate",
    "UniversalPeriodicTrendAdapter",
    "PhysicalBlockTrendAdapter",
    "PhysicalScalarComponents",
    "fixed_block_trend_component",
    "fixed_patch_level_component",
    "reconstruct_block_trend",
    "reconstruct_patch_level",
]
