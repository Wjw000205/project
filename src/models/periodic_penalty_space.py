"""Shared physical-period residual spaces for reusable penalty adapters.

The learned adapter kernels do not own this decomposition.  A dataset supplies
only its physical period/patch geometry and causal carrier bank; the projector
formula is otherwise identical for every dataset and horizon.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.models.period_geometry import PeriodGeometry


def _require_horizon(values: torch.Tensor, geometry: PeriodGeometry) -> int:
    if values.ndim != 2:
        raise ValueError("periodic penalty values must have shape [B,H]")
    horizon = int(values.shape[1])
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    return horizon


def project_period_level(
    values: torch.Tensor,
    geometry: PeriodGeometry,
) -> torch.Tensor:
    """Project onto one constant coordinate per physical period."""

    horizon = _require_horizon(values, geometry)
    result = torch.empty_like(values)
    for left in range(0, horizon, geometry.period_steps):
        right = min(left + geometry.period_steps, horizon)
        result[:, left:right] = values[:, left:right].mean(dim=1, keepdim=True)
    return result


def project_period_patch_level(
    values: torch.Tensor,
    geometry: PeriodGeometry,
) -> torch.Tensor:
    """Project onto one constant coordinate per native phase patch.

    This is the fine-grained Level space used by the accepted ETTm1 H96
    expert: a physical period contains ``patches_per_period`` independently
    predicted constant coordinates.  Patch geometry is only the external
    decoder; it does not prescribe a correction sign, magnitude, or template.
    """

    horizon = _require_horizon(values, geometry)
    result = torch.empty_like(values)
    for left in range(0, horizon, geometry.patch_steps):
        right = min(left + geometry.patch_steps, horizon)
        result[:, left:right] = values[:, left:right].mean(dim=1, keepdim=True)
    return result


def project_period_local_level(
    values: torch.Tensor,
    geometry: PeriodGeometry,
) -> torch.Tensor:
    """Return the zero-period-mean part of patch-local Level.

    ``project_period_patch_level(values)`` is exactly the orthogonal sum of
    the one-coordinate physical-period Level and this within-period local
    component.  A complete physical period therefore owns one global and
    ``patches_per_period - 1`` relative Level degrees of freedom.
    """

    return project_period_patch_level(values, geometry) - project_period_level(
        values, geometry
    )


def project_period_patch_centered_remainder(
    values: torch.Tensor,
    geometry: PeriodGeometry,
) -> torch.Tensor:
    """Remove every native phase-patch Level coordinate from ``values``."""

    _require_horizon(values, geometry)
    return values - project_period_patch_level(values, geometry)


def project_period_trend(
    values: torch.Tensor,
    geometry: PeriodGeometry,
) -> torch.Tensor:
    """Project onto one zero-mean continuous linear physical-period coordinate."""

    horizon = _require_horizon(values, geometry)
    result = torch.empty_like(values)
    for left in range(0, horizon, geometry.period_steps):
        right = min(left + geometry.period_steps, horizon)
        block = values[:, left:right]
        basis = torch.linspace(
            -1.0,
            1.0,
            right - left,
            device=values.device,
            dtype=values.dtype,
        )
        basis = basis - basis.mean()
        centered = block - block.mean(dim=1, keepdim=True)
        coefficient = (centered * basis).sum(dim=1, keepdim=True)
        coefficient = coefficient / basis.square().sum().clamp_min(1.0e-12)
        result[:, left:right] = coefficient * basis
    return result


def project_period_amp_envelope(
    values: torch.Tensor,
    geometry: PeriodGeometry,
) -> torch.Tensor:
    """Affine-free residual envelope inside every physical period."""

    _require_horizon(values, geometry)
    return values - project_period_level(values, geometry) - project_period_trend(
        values, geometry
    )


def project_period_patch_level_conditional_amp_envelope(
    values: torch.Tensor,
    geometry: PeriodGeometry,
) -> torch.Tensor:
    """Remove patch-local Level and its orthogonal conditional Trend.

    For each physical period, let ``P_patch`` project onto the independently
    constant native patches and let ``b`` be the centered linear period basis.
    This returns the orthogonal projection

    ``(I - P_patch - P_((I - P_patch)b)) values``.

    The learned adapter remains outside this geometry-only operation.  In
    particular, this projector neither owns nor changes any Amp parameters.
    """

    horizon = _require_horizon(values, geometry)
    parts: list[torch.Tensor] = []
    for left in range(0, horizon, geometry.period_steps):
        right = min(left + geometry.period_steps, horizon)
        block = values[:, left:right]
        patch_centered = block - project_period_patch_level(block, geometry)

        basis = torch.linspace(
            -1.0,
            1.0,
            right - left,
            device=values.device,
            dtype=values.dtype,
        )
        basis = basis - basis.mean()
        conditional_basis = basis - project_period_patch_level(
            basis.unsqueeze(0), geometry
        ).squeeze(0)
        coefficient = (patch_centered * conditional_basis).sum(
            dim=1, keepdim=True
        )
        coefficient = coefficient / conditional_basis.square().sum().clamp_min(
            1.0e-12
        )
        parts.append(patch_centered - coefficient * conditional_basis)
    return torch.cat(parts, dim=1)


def affine_free_carrier_curves(
    carriers: torch.Tensor,
    geometry: PeriodGeometry,
) -> torch.Tensor:
    """Return causal carrier curves as [B,K,H] in physical-period Amp space."""

    if carriers.ndim != 4:
        raise ValueError("carriers must have shape [B,P,K,L]")
    batch, patch_count, carrier_count, patch_steps = carriers.shape
    if int(patch_steps) != geometry.patch_steps:
        raise ValueError("carrier patch width differs from native geometry")
    horizon = int(patch_count) * int(patch_steps)
    full = carriers.permute(0, 2, 1, 3).reshape(batch * carrier_count, horizon)
    projected = project_period_amp_envelope(full, geometry)
    return projected.reshape(batch, carrier_count, horizon)


def project_period_carrier_amp(
    values: torch.Tensor,
    carriers: torch.Tensor,
    geometry: PeriodGeometry,
    *,
    rtol: float = 1.0e-5,
) -> torch.Tensor:
    """Project each physical period into its causal affine-free carrier span.

    ``carriers`` is target-free.  The target-visible coefficient is used only
    to materialize a supervised Amp coordinate during fitting or to constrain
    a predicted action at execution.  ``pinv`` gives the orthogonal projection
    even when causal carriers are rank deficient.
    """

    horizon = _require_horizon(values, geometry)
    curves = affine_free_carrier_curves(carriers, geometry)
    if curves.shape[0] != values.shape[0] or curves.shape[2] != horizon:
        raise ValueError("carrier/value batch or horizon mismatch")
    parts: list[torch.Tensor] = []
    for left in range(0, horizon, geometry.period_steps):
        right = min(left + geometry.period_steps, horizon)
        design = curves[:, :, left:right].transpose(1, 2)
        response = values[:, left:right].unsqueeze(2)
        coefficient = torch.linalg.pinv(design, rtol=float(rtol)) @ response
        parts.append((design @ coefficient).squeeze(2))
    return torch.cat(parts, dim=1)


def project_period_amp_reference(
    values: torch.Tensor,
    amp_reference: torch.Tensor,
    geometry: PeriodGeometry,
) -> torch.Tensor:
    """Project affine-free values onto one causal Amp direction per period.

    ``amp_reference`` is target-free and may be produced by any reusable Amp
    kernel or by a deterministic seasonal-memory anchor.  Only its direction
    defines the Amp coordinate; a zero reference yields exact NOOP.  The
    operation is differentiable with respect to both inputs.
    """

    horizon = _require_horizon(values, geometry)
    if amp_reference.shape != values.shape:
        raise ValueError("Amp reference must match periodic values")
    response = project_period_amp_envelope(values, geometry)
    reference = project_period_amp_envelope(amp_reference, geometry)
    result = torch.zeros_like(response)
    for left in range(0, horizon, geometry.period_steps):
        right = min(left + geometry.period_steps, horizon)
        current = reference[:, left:right]
        coefficient = (response[:, left:right] * current).sum(dim=1, keepdim=True)
        coefficient = coefficient / current.square().sum(dim=1, keepdim=True).clamp_min(
            1.0e-12
        )
        result[:, left:right] = coefficient * current
    return result


def project_period_patch_centered_amp_reference(
    values: torch.Tensor,
    amp_reference: torch.Tensor,
    geometry: PeriodGeometry,
) -> torch.Tensor:
    """Project onto a causal Amp direction after removing p12-Level space.

    Both the supervised response and target-free reference are first placed
    in the exact complement of the patch-local Level projector.  The resulting
    Amp action therefore has zero mean in every native patch and cannot
    duplicate any of the Level expert's independently learned coordinates.
    One coefficient is solved per physical period; a zero reference is exact
    NOOP.  The operation remains differentiable with respect to both inputs.
    """

    horizon = _require_horizon(values, geometry)
    if amp_reference.shape != values.shape:
        raise ValueError("Amp reference must match periodic values")
    response = project_period_patch_centered_remainder(values, geometry)
    reference = project_period_patch_centered_remainder(amp_reference, geometry)
    result = torch.zeros_like(response)
    for left in range(0, horizon, geometry.period_steps):
        right = min(left + geometry.period_steps, horizon)
        current = reference[:, left:right]
        coefficient = (response[:, left:right] * current).sum(dim=1, keepdim=True)
        coefficient = coefficient / current.square().sum(dim=1, keepdim=True).clamp_min(
            1.0e-12
        )
        result[:, left:right] = coefficient * current
    return result


def project_period_shape_remainder(
    values: torch.Tensor,
    amp_reference: torch.Tensor,
    geometry: PeriodGeometry,
) -> torch.Tensor:
    """Project into the complement of Level, Trend, and causal Amp direction."""

    envelope = project_period_amp_envelope(values, geometry)
    amp = project_period_amp_reference(values, amp_reference, geometry)
    return envelope - amp


@dataclass(frozen=True)
class PeriodicPenaltyComponents:
    level: torch.Tensor
    trend: torch.Tensor
    amp: torch.Tensor
    shape: torch.Tensor


def decompose_periodic_penalties(
    values: torch.Tensor,
    carriers: torch.Tensor,
    geometry: PeriodGeometry,
    *,
    rtol: float = 1.0e-5,
) -> PeriodicPenaltyComponents:
    """Return the complete mutually exclusive Level/Trend/Amp/Shape split.

    The deterministic mean of the supplied target-free causal carrier curves
    defines one Amp direction per physical period.  Shape owns the remaining
    waveform after Level, Trend, and that Amp direction.  Changing the native
    clock changes only ``geometry``; the formula and action spaces are shared.
    """

    level = project_period_level(values, geometry)
    trend = project_period_trend(values, geometry)
    curves = affine_free_carrier_curves(carriers, geometry)
    amp_reference = curves.mean(dim=1)
    amp = project_period_amp_reference(values, amp_reference, geometry)
    shape = project_period_shape_remainder(values, amp_reference, geometry)
    return PeriodicPenaltyComponents(level=level, trend=trend, amp=amp, shape=shape)


__all__ = [
    "PeriodicPenaltyComponents",
    "affine_free_carrier_curves",
    "decompose_periodic_penalties",
    "project_period_amp_envelope",
    "project_period_amp_reference",
    "project_period_carrier_amp",
    "project_period_level",
    "project_period_local_level",
    "project_period_patch_centered_amp_reference",
    "project_period_patch_centered_remainder",
    "project_period_patch_level",
    "project_period_patch_level_conditional_amp_envelope",
    "project_period_shape_remainder",
    "project_period_trend",
]
