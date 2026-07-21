"""Parameter-free canonical alignment for periodic Shape coordinates.

Shape is defined in the complement of the current frozen Amp direction.  Two
independently fitted Amp instances may rotate that complement even though the
Shape network and physical residual are unchanged.  This module supplies one
deterministic Householder frame on the fixed canonical P96 lattice.  It owns no
parameters and the same transform is its own inverse.
"""

from __future__ import annotations

import math

import torch


CANONICAL_SHAPE_STEPS = 96


def _remove_affine(values: torch.Tensor) -> torch.Tensor:
    steps = int(values.shape[-1])
    centered = values - values.mean(dim=-1, keepdim=True)
    basis = torch.linspace(
        -1.0,
        1.0,
        steps,
        dtype=values.dtype,
        device=values.device,
    )
    basis = basis - basis.mean()
    coefficient = torch.sum(centered * basis, dim=-1, keepdim=True)
    coefficient = coefficient / torch.sum(torch.square(basis)).clamp_min(1.0e-12)
    return centered - coefficient * basis


def canonical_shape_reference(
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the fixed unit affine-free reference direction on canonical P96."""

    phase = torch.arange(
        CANONICAL_SHAPE_STEPS,
        dtype=dtype,
        device=device,
    )
    reference = torch.cos(
        (2.0 * math.pi / CANONICAL_SHAPE_STEPS) * phase
    )
    reference = _remove_affine(reference)
    return reference / torch.linalg.vector_norm(reference).clamp_min(1.0e-12)


def canonical_shape_householder_vector(
    amp_reference: torch.Tensor,
    *,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Return a deterministic reflector mapping each Amp line to one axis.

    Amp is an unoriented one-dimensional subspace.  Its sign is therefore
    chosen target-free so that the unit direction has nonnegative inner product
    with the fixed reference.  A zero Amp row or an already aligned row returns
    the zero reflector, i.e. exact identity.
    """

    if amp_reference.ndim != 2 or amp_reference.shape[1] != CANONICAL_SHAPE_STEPS:
        raise ValueError("canonical Shape Amp reference must have shape [B,96]")
    amp = _remove_affine(amp_reference)
    amp_norm = torch.linalg.vector_norm(amp, dim=1, keepdim=True)
    unit = amp / amp_norm.clamp_min(float(epsilon))
    reference = canonical_shape_reference(
        device=amp_reference.device, dtype=amp_reference.dtype
    )[None]
    orientation = torch.where(
        torch.sum(unit * reference, dim=1, keepdim=True) < 0.0,
        -torch.ones_like(amp_norm),
        torch.ones_like(amp_norm),
    )
    oriented = orientation * unit
    difference = oriented - reference
    difference_norm = torch.linalg.vector_norm(difference, dim=1, keepdim=True)
    active = (amp_norm > float(epsilon)) & (difference_norm > float(epsilon))
    vector = difference / difference_norm.clamp_min(float(epsilon))
    return torch.where(active, vector, torch.zeros_like(vector))


def apply_canonical_shape_alignment(
    values: torch.Tensor,
    amp_reference: torch.Tensor,
    *,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Apply the per-row P96 Householder transform on the final axis.

    ``values`` may be ``[B,96]`` or contain any number of middle axes such as
    the 28 causal memory curves.  Householder reflection is symmetric and
    orthogonal, so calling this function twice with the same Amp reference is
    the exact inverse up to floating-point error.
    """

    if values.ndim < 2 or int(values.shape[0]) != int(amp_reference.shape[0]):
        raise ValueError("canonical Shape values and Amp batch must align")
    if int(values.shape[-1]) != CANONICAL_SHAPE_STEPS:
        raise ValueError("canonical Shape values must end in width 96")
    vector = canonical_shape_householder_vector(
        amp_reference, epsilon=epsilon
    )
    view_shape = (int(values.shape[0]),) + (1,) * (values.ndim - 2) + (
        CANONICAL_SHAPE_STEPS,
    )
    vector = vector.reshape(view_shape)
    coefficient = torch.sum(values * vector, dim=-1, keepdim=True)
    return values - 2.0 * coefficient * vector


__all__ = [
    "CANONICAL_SHAPE_STEPS",
    "apply_canonical_shape_alignment",
    "canonical_shape_householder_vector",
    "canonical_shape_reference",
]
