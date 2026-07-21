"""Semantically isolated residual-space adapters for existing penalties.

The existing functions in :mod:`src.models.penalties` remain the authoritative
objectives.  This module supplies target-free feature contracts and mutually
orthogonal correction spaces.  Every adapter owns its network and may only
predict coordinates inside the space named by its penalty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.penalties import build_penalty_compute


HORIZON = 96
PATCHES = 8
PATCH_STEPS = 12
SPACE_ORDER = (
    "amp_under",
    "range",
    "diff_amp",
    "delta",
    "direction",
    "d2_match",
    "corr",
)


@dataclass(frozen=True)
class SemanticPenaltyContract:
    name: str
    feature_width: int
    coordinates: int
    feature_semantics: str
    residual_space: str
    positive_coordinates: bool = False


CONTRACTS = {
    "amp_under": SemanticPenaltyContract(
        "amp_under",
        51,
        1,
        "horizon/p12 standard-deviation state plus 24D amplitude-only forecast revisions",
        "centered frozen-base amplitude ray",
        True,
    ),
    "range": SemanticPenaltyContract(
        "range",
        35,
        1,
        "horizon/p12 peak-to-trough extent and extrema phase",
        "frozen-base max-minus-min endpoint coordinate",
    ),
    "diff_amp": SemanticPenaltyContract(
        "diff_amp",
        27,
        4,
        "first-difference volatility at horizon and p12 scales",
        "four causal high-pass history carriers",
    ),
    "delta": SemanticPenaltyContract(
        "delta",
        48,
        8,
        "signed and absolute first-difference p12 summaries",
        "eight integrated local first-difference coordinates",
    ),
    "direction": SemanticPenaltyContract(
        "direction",
        40,
        8,
        "p12 movement sign, agreement, and movement energy",
        "eight causal sign-vote integrated movement rays",
        True,
    ),
    "d2_match": SemanticPenaltyContract(
        "d2_match",
        72,
        8,
        "signed, absolute, and scale curvature p12 summaries",
        "eight affine-free double-integrated curvature coordinates",
    ),
    "corr": SemanticPenaltyContract(
        "corr",
        49,
        4,
        "centered unit-shape p12 profile, spread, and correlation",
        "four causal centered-shape carriers after other spaces",
    ),
}

PENALTY_NATIVE_RESIDUAL_SPACES = {
    "amp_under": "actual positive horizon-standard-deviation error",
    "range": "actual horizon peak-to-trough range error",
    "diff_amp": "actual first-difference standard-deviation error",
    "delta": "actual pointwise first-difference residual",
    "direction": "actual one-sided directional under-fit residual",
    "d2_match": "actual pointwise second-difference residual",
    "corr": "actual centered unit-shape residual",
}

PENALTY_NATIVE_CONTRACTS = {
    name: SemanticPenaltyContract(
        name=contract.name,
        feature_width=contract.feature_width,
        coordinates=contract.coordinates,
        feature_semantics=contract.feature_semantics,
        residual_space=PENALTY_NATIVE_RESIDUAL_SPACES[name],
        # QR may flip any target-free basis sign. One-sided semantics belong to
        # the actual error encoder, never to the arbitrary coordinate sign.
        positive_coordinates=False,
    )
    for name, contract in CONTRACTS.items()
}

DELTA_MATURED_RESIDUAL_CONTRACT = SemanticPenaltyContract(
    name="delta",
    feature_width=112,
    coordinates=8,
    feature_semantics=(
        "48D current D1 summaries plus 64D fully matured historical forecast "
        "residual D1 state"
    ),
    residual_space="actual pointwise first-difference residual",
    positive_coordinates=False,
)

DIFF_AMP_MATURED_RESIDUAL_CONTRACT = SemanticPenaltyContract(
    name="diff_amp",
    feature_width=63,
    coordinates=4,
    feature_semantics=(
        "27D current D1-volatility summaries plus 36D fully matured historical "
        "forecast D1-volatility error state"
    ),
    residual_space="actual first-difference standard-deviation error",
    positive_coordinates=False,
)

DELTA_OPERATOR_SEQUENCE_CONTRACT = SemanticPenaltyContract(
    name="delta",
    feature_width=8 * HORIZON,
    coordinates=8,
    feature_semantics=(
        "eight complete fully matured D1 forecast-residual curves plus current "
        "history/base D1 sequences"
    ),
    residual_space="actual pointwise first-difference residual",
    positive_coordinates=False,
)

DELTA_PATCH_SIGNED_MEMORY_CONTRACT = SemanticPenaltyContract(
    name="delta",
    feature_width=8 * HORIZON,
    coordinates=8,
    feature_semantics=(
        "eight complete matured D1 residual memories; shared p12 attention "
        "retrieval with one bounded signed gate per patch"
    ),
    residual_space="actual pointwise first-difference residual",
    positive_coordinates=False,
)

DELTA_PHYSICAL_MEMORY_CONTRACT = SemanticPenaltyContract(
    name="delta",
    feature_width=28 * HORIZON,
    coordinates=8,
    feature_semantics=(
        "28 complete same-phase unseparated D1(target-base) operator-error "
        "memories with shared p12 cross-attention and one bounded signed gate"
    ),
    residual_space="actual pointwise first-difference residual",
    positive_coordinates=False,
)

DELTA_PHYSICAL_COMPONENT_MEMORY_CONTRACT = SemanticPenaltyContract(
    name="delta",
    feature_width=28 * HORIZON,
    coordinates=8,
    feature_semantics=(
        "28 complete same-phase source-local penalty-native Delta-component "
        "D1 residual memories after orthogonal decomposition and risk guard"
    ),
    residual_space="actual pointwise first-difference residual",
    positive_coordinates=False,
)

DELTA_WHERE_WHAT_COMPONENT_MEMORY_CONTRACT = SemanticPenaltyContract(
    name="delta",
    feature_width=42 * HORIZON,
    coordinates=8,
    feature_semantics=(
        "28 source-local penalty-native Delta-component D1 residual values; "
        "seven aligned target-free D1 forecast revisions and masks feed only "
        "an independent local [0,1] position suppressor"
    ),
    residual_space="actual pointwise first-difference residual",
    positive_coordinates=False,
)

DELTA_CAUSAL_TRANSITION_CONTRACT = SemanticPenaltyContract(
    name="delta",
    feature_width=42 * HORIZON,
    coordinates=8,
    feature_semantics=(
        "28 source-local penalty-native Delta-component D1 memories plus seven "
        "aligned target-free backbone D1 revisions and masks; one shared causal "
        "state-transition kernel emits the next-step D1 action at every lead"
    ),
    residual_space="actual pointwise first-difference residual",
    positive_coordinates=False,
)

DELTA_PHYSICAL_KEY_VALUE_CONTRACT = SemanticPenaltyContract(
    name="delta",
    feature_width=28 * 3 * HORIZON,
    coordinates=8,
    feature_semantics=(
        "28 same-phase key/value records: source D1(history/base) keys paired "
        "with fully matured actual D1(target-base) residual values"
    ),
    residual_space="actual pointwise first-difference residual",
    positive_coordinates=False,
)

DIRECTION_PHYSICAL_MEMORY_CONTRACT = SemanticPenaltyContract(
    name="direction",
    feature_width=28 * HORIZON,
    coordinates=8,
    feature_semantics=(
        "28 complete same-phase actual one-sided Direction residual memories "
        "with shared p12 cross-attention and a bounded signed gate"
    ),
    residual_space="actual one-sided directional under-fit residual",
    positive_coordinates=False,
)

AMP_UNDER_PHYSICAL_SCALAR_CONTRACT = SemanticPenaltyContract(
    name="amp_under",
    feature_width=52,
    coordinates=1,
    feature_semantics=(
        "28 fully matured actual positive horizon-standard-deviation deficits "
        "plus 24 target-free aligned amplitude forecast revisions"
    ),
    residual_space="actual positive horizon-standard-deviation error",
    positive_coordinates=False,
)


def _require_panel(history: torch.Tensor, base: torch.Tensor) -> None:
    if history.shape != base.shape or history.ndim != 2:
        raise ValueError("history and base must both have shape [N,96]")
    if history.shape[1] != HORIZON:
        raise ValueError("semantic penalty adapters require canonical H96")


def _patch(values: torch.Tensor) -> torch.Tensor:
    return values.reshape(values.shape[0], PATCHES, PATCH_STEPS)


def _patch_mean(values: torch.Tensor) -> torch.Tensor:
    return _patch(values).mean(dim=2)


def _patch_abs_mean(values: torch.Tensor) -> torch.Tensor:
    return _patch(values).abs().mean(dim=2)


def _patch_std(values: torch.Tensor) -> torch.Tensor:
    return _patch(values).std(dim=2, unbiased=False)


def _patch_range(values: torch.Tensor) -> torch.Tensor:
    panel = _patch(values)
    return panel.amax(dim=2) - panel.amin(dim=2)


def _d1(values: torch.Tensor) -> torch.Tensor:
    difference = values[:, 1:] - values[:, :-1]
    return F.pad(difference, (1, 0))


def _d2(values: torch.Tensor) -> torch.Tensor:
    curvature = values[:, 2:] - 2.0 * values[:, 1:-1] + values[:, :-2]
    return F.pad(curvature, (2, 0))


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return numerator / denominator.clamp_min(1.0e-6)


def _phase(index: torch.Tensor) -> torch.Tensor:
    angle = index.to(dtype=torch.float32) * (2.0 * math.pi / HORIZON)
    return torch.stack([angle.sin(), angle.cos()], dim=1)


def _center_unit(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean(dim=1, keepdim=True)
    return centered / centered.square().mean(dim=1, keepdim=True).sqrt().clamp_min(
        1.0e-6
    )


def _amp_under_features(history: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    history_std = history.std(dim=1, unbiased=False, keepdim=True)
    base_std = base.std(dim=1, unbiased=False, keepdim=True)
    return torch.cat(
        [
            base_std,
            history_std,
            _safe_ratio(base_std, history_std),
            _patch_std(base),
            _patch_std(history),
            _patch_std(base) - _patch_std(history),
        ],
        dim=1,
    )


def _range_features(history: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    history_range = history.amax(dim=1, keepdim=True) - history.amin(
        dim=1, keepdim=True
    )
    base_range = base.amax(dim=1, keepdim=True) - base.amin(dim=1, keepdim=True)
    return torch.cat(
        [
            base_range,
            history_range,
            _safe_ratio(base_range, history_range),
            _patch_range(base),
            _patch_range(history),
            _patch_range(base) - _patch_range(history),
            _phase(base.argmax(dim=1)),
            _phase(base.argmin(dim=1)),
            _phase(history.argmax(dim=1)),
            _phase(history.argmin(dim=1)),
        ],
        dim=1,
    )


def _diff_amp_features(history: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    history_d1 = _d1(history)
    base_d1 = _d1(base)
    history_std = history_d1.std(dim=1, unbiased=False, keepdim=True)
    base_std = base_d1.std(dim=1, unbiased=False, keepdim=True)
    return torch.cat(
        [
            base_std,
            history_std,
            _safe_ratio(base_std, history_std),
            _patch_std(base_d1),
            _patch_std(history_d1),
            _patch_std(base_d1) - _patch_std(history_d1),
        ],
        dim=1,
    )


def _delta_features(history: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    history_d1 = _d1(history)
    base_d1 = _d1(base)
    return torch.cat(
        [
            _patch_mean(base_d1),
            _patch_mean(history_d1),
            _patch_mean(base_d1 - history_d1),
            _patch_abs_mean(base_d1),
            _patch_abs_mean(history_d1),
            _patch_abs_mean(base_d1 - history_d1),
        ],
        dim=1,
    )


def _direction_features(history: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    history_d1 = _d1(history)
    base_d1 = _d1(base)
    agreement = history_d1.sign() * base_d1.sign()
    return torch.cat(
        [
            _patch_mean(base_d1.sign()),
            _patch_mean(history_d1.sign()),
            _patch_mean(agreement),
            _patch_abs_mean(base_d1),
            _patch_abs_mean(history_d1),
        ],
        dim=1,
    )


def _d2_features(history: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    history_d2 = _d2(history)
    base_d2 = _d2(base)
    difference = base_d2 - history_d2
    return torch.cat(
        [
            _patch_mean(base_d2),
            _patch_mean(history_d2),
            _patch_mean(difference),
            _patch_abs_mean(base_d2),
            _patch_abs_mean(history_d2),
            _patch_abs_mean(difference),
            _patch_std(base_d2),
            _patch_std(history_d2),
            _patch_std(difference),
        ],
        dim=1,
    )


def _corr_features(history: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    history_shape = _center_unit(history)
    base_shape = _center_unit(base)
    correlation = (history_shape * base_shape).mean(dim=1, keepdim=True)
    return torch.cat(
        [
            _patch_mean(base_shape),
            _patch_mean(history_shape),
            _patch_mean(base_shape - history_shape),
            _patch_std(base_shape),
            _patch_std(history_shape),
            _patch_std(base_shape - history_shape),
            correlation,
        ],
        dim=1,
    )


FEATURE_BUILDERS: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "amp_under": _amp_under_features,
    "range": _range_features,
    "diff_amp": _diff_amp_features,
    "delta": _delta_features,
    "direction": _direction_features,
    "d2_match": _d2_features,
    "corr": _corr_features,
}


def semantic_penalty_features(
    name: str,
    history: torch.Tensor,
    base: torch.Tensor,
    *,
    causal_state: torch.Tensor | None = None,
) -> torch.Tensor:
    _require_panel(history, base)
    if name not in FEATURE_BUILDERS:
        raise ValueError(f"unsupported semantic penalty adapter: {name}")
    result = FEATURE_BUILDERS[name](history, base)
    if name == "amp_under":
        if causal_state is None:
            causal_state = torch.zeros(
                history.shape[0], 24, device=history.device, dtype=history.dtype
            )
        if causal_state.shape != (history.shape[0], 24):
            raise ValueError("amp_under causal revision state must have shape [N,24]")
        result = torch.cat([result, causal_state.to(result)], dim=1)
    elif causal_state is not None:
        raise ValueError(f"{name} does not accept amp_under causal state")
    expected = CONTRACTS[name].feature_width
    if result.shape != (history.shape[0], expected):
        raise RuntimeError(
            f"{name} feature contract drift: {tuple(result.shape)} != "
            f"({history.shape[0]}, {expected})"
        )
    return result


def _remove_affine(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean(dim=1, keepdim=True)
    axis = torch.linspace(
        -1.0, 1.0, HORIZON, device=values.device, dtype=values.dtype
    )
    axis = axis - axis.mean()
    coefficient = (centered * axis).sum(dim=1, keepdim=True)
    coefficient = coefficient / axis.square().sum().clamp_min(1.0e-12)
    return centered - coefficient * axis


def _smooth(values: torch.Tensor, width: int) -> torch.Tensor:
    if width <= 1:
        return values
    padded = F.pad(values.unsqueeze(1), (width - 1, 0), mode="replicate")
    return F.avg_pool1d(padded, kernel_size=width, stride=1).squeeze(1)


def _cumulative(values: torch.Tensor) -> torch.Tensor:
    """Deterministic inclusive cumulative sum on the fixed H96 axis."""

    matrix = torch.ones(
        HORIZON, HORIZON, device=values.device, dtype=values.dtype
    ).triu()
    return values @ matrix


def _raw_spaces(history: torch.Tensor, base: torch.Tensor) -> dict[str, torch.Tensor]:
    count = history.shape[0]
    centered_base = base - base.mean(dim=1, keepdim=True)
    amp = centered_base.unsqueeze(1)

    range_basis = torch.zeros_like(base)
    range_basis.scatter_(1, base.argmax(dim=1, keepdim=True), 1.0)
    range_basis.scatter_add_(
        1,
        base.argmin(dim=1, keepdim=True),
        -torch.ones(count, 1, device=base.device, dtype=base.dtype),
    )

    diff_amp = torch.stack(
        [history - _smooth(history, width) for width in (3, 6, 12, 24)],
        dim=1,
    )

    delta_parts = []
    direction_parts = []
    history_d1 = _d1(history)
    base_d1 = _d1(base)
    sign_vote = (history_d1 + base_d1).sign()
    for patch in range(PATCHES):
        left = patch * PATCH_STEPS
        right = left + PATCH_STEPS
        gradient = torch.zeros_like(base)
        gradient[:, left:right] = 1.0
        delta_parts.append(_cumulative(gradient))
        signed_gradient = torch.zeros_like(base)
        signed_gradient[:, left:right] = sign_vote[:, left:right]
        direction_parts.append(_cumulative(signed_gradient))
    delta = torch.stack(delta_parts, dim=1)
    direction = torch.stack(direction_parts, dim=1)

    d2_parts = []
    for patch in range(PATCHES):
        left = patch * PATCH_STEPS
        right = left + PATCH_STEPS
        curvature = torch.zeros_like(base)
        curvature[:, left:right] = 1.0
        d2_parts.append(_remove_affine(_cumulative(_cumulative(curvature))))
    d2_match = torch.stack(d2_parts, dim=1)

    history_shape = _center_unit(history)
    corr = torch.stack(
        [_remove_affine(_smooth(history_shape, width)) for width in (1, 3, 6, 12)],
        dim=1,
    )
    return {
        "amp_under": amp,
        "range": range_basis.unsqueeze(1),
        "diff_amp": diff_amp,
        "delta": delta,
        "direction": direction,
        "d2_match": d2_match,
        "corr": corr,
    }


def semantic_penalty_bases(
    history: torch.Tensor,
    base: torch.Tensor,
    *,
    space_order: tuple[str, ...] = SPACE_ORDER,
    eps: float = 1.0e-5,
) -> dict[str, torch.Tensor]:
    """Return mutually orthogonal target-free bases as ``[N,K,96]``.

    ``space_order`` is the frozen dataset/cluster-owned penalty pool. Omitting
    a rejected penalty prevents that unused direction from consuming residual
    energy during decomposition. The raw named spaces and adapters stay fixed.
    """

    _require_panel(history, base)
    if not space_order or len(set(space_order)) != len(space_order):
        raise ValueError("space_order must contain unique penalty names")
    unknown = set(space_order).difference(SPACE_ORDER)
    if unknown:
        raise ValueError(f"unsupported semantic spaces: {sorted(unknown)}")
    raw = _raw_spaces(history, base)
    joined = torch.cat([raw[name] for name in space_order], dim=1)
    joined = joined - joined.mean(dim=2, keepdim=True)
    q, r = torch.linalg.qr(joined.transpose(1, 2), mode="reduced")
    diagonal = r.diagonal(dim1=1, dim2=2).abs()
    valid = diagonal > float(eps)
    orthogonal = q.transpose(1, 2) * math.sqrt(HORIZON)
    orthogonal = torch.where(valid.unsqueeze(2), orthogonal, torch.zeros_like(orthogonal))
    result: dict[str, torch.Tensor] = {}
    offset = 0
    for name in space_order:
        width = raw[name].shape[1]
        result[name] = orthogonal[:, offset : offset + width]
        offset += width
    return result


def project_residual_to_penalty_space(
    name: str,
    residual: torch.Tensor,
    history: torch.Tensor,
    base: torch.Tensor,
    *,
    space_order: tuple[str, ...] = SPACE_ORDER,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project ``residual`` into one named target-free semantic space."""

    _require_panel(history, base)
    if residual.shape != base.shape:
        raise ValueError("residual must match history/base shape")
    if name not in space_order:
        raise ValueError(f"{name} is absent from the requested decomposition")
    basis = semantic_penalty_bases(
        history, base, space_order=space_order
    )[name]
    coordinates = torch.einsum("nh,nkh->nk", residual, basis) / HORIZON
    if CONTRACTS[name].positive_coordinates:
        coordinates = coordinates.clamp_min(0.0)
    projection = torch.einsum("nk,nkh->nh", coordinates, basis)
    return projection, coordinates, basis


@torch.no_grad()
def penalty_aligned_residual_component(
    name: str,
    residual: torch.Tensor,
    history: torch.Tensor,
    base: torch.Tensor,
    *,
    space_order: tuple[str, ...] = SPACE_ORDER,
    line_search_steps: int = 17,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a named residual component that cannot worsen its penalty.

    The raw Euclidean residual projection supplies the named target-free
    action direction. A per-row line search over ``[0,1]`` then reuses the
    exact existing penalty and includes exact NOOP. Consequently the returned
    training label remains inside the named space, cannot increase MSE, and
    cannot increase the penalty whose name the adapter carries.
    """

    if int(line_search_steps) < 2:
        raise ValueError("penalty alignment needs at least two line-search steps")
    projection, coordinates, basis = project_residual_to_penalty_space(
        name,
        residual,
        history,
        base,
        space_order=space_order,
    )
    target = base + residual
    alpha = torch.linspace(
        0.0,
        1.0,
        int(line_search_steps),
        device=base.device,
        dtype=base.dtype,
    )
    candidate = base.unsqueeze(0) + alpha[:, None, None] * projection.unsqueeze(0)
    expanded_target = target.unsqueeze(0).expand_as(candidate)
    penalty_compute = build_penalty_compute([name], jump_thr=0.6)
    penalties = penalty_compute(
        candidate.reshape(-1, 1, HORIZON),
        expanded_target.reshape(-1, 1, HORIZON),
    ).reshape(alpha.numel(), residual.shape[0])
    selected = alpha[penalties.argmin(dim=0)]
    aligned_coordinates = coordinates * selected.unsqueeze(1)
    aligned_projection = projection * selected.unsqueeze(1)
    return aligned_projection, aligned_coordinates, basis


def _solve_operator_residual(
    operator_basis: torch.Tensor,
    operator_residual: torch.Tensor,
) -> torch.Tensor:
    """Minimum-norm coordinates fitting an actual penalty-domain residual."""

    design = operator_basis.transpose(1, 2)
    return (
        torch.linalg.pinv(design, rtol=1.0e-5)
        @ operator_residual.unsqueeze(2)
    ).squeeze(2)


def _std_observable_coordinates(
    values: torch.Tensor,
    target_values: torch.Tensor,
    basis: torch.Tensor,
    *,
    one_sided: bool,
) -> torch.Tensor:
    residual = target_values.std(dim=1, unbiased=True) - values.std(
        dim=1, unbiased=True
    )
    if one_sided:
        residual = residual.clamp_min(0.0)
    return _std_observable_coordinates_from_residual(values, basis, residual)


def _std_observable_coordinates_from_residual(
    values: torch.Tensor,
    basis: torch.Tensor,
    operator_residual: torch.Tensor,
) -> torch.Tensor:
    """Decode one actual standard-deviation error through its Jacobian."""

    count = int(values.shape[1])
    centered = values - values.mean(dim=1, keepdim=True)
    deviation = values.std(dim=1, unbiased=True).clamp_min(1.0e-6)
    centered_basis = basis - basis.mean(dim=2, keepdim=True)
    gradient = centered / (float(count - 1) * deviation.unsqueeze(1))
    jacobian = torch.einsum("nh,nkh->nk", gradient, centered_basis)
    denominator = jacobian.square().sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    return jacobian * operator_residual.unsqueeze(1) / denominator


def diff_amp_operator_residual(
    base: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Actual signed error of the existing ``diff_amp`` penalty operator."""

    if base.shape != target.shape or base.ndim != 2:
        raise ValueError("diff_amp base/target must share shape [N,H]")
    if int(base.shape[1]) < 2:
        raise ValueError("diff_amp requires at least two forecast steps")
    return torch.diff(target, dim=1).std(
        dim=1, unbiased=True
    ) - torch.diff(base, dim=1).std(dim=1, unbiased=True)


def amp_under_operator_residual(
    base: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Actual nonnegative error of the existing AmpUnder operator."""

    if base.shape != target.shape or base.ndim != 2:
        raise ValueError("amp_under base/target must share shape [N,H]")
    return (
        target.std(dim=1, unbiased=True) - base.std(dim=1, unbiased=True)
    ).clamp_min(0.0)


def decode_amp_under_operator_residual(
    history: torch.Tensor,
    base: torch.Tensor,
    operator_residual: torch.Tensor,
    *,
    basis: torch.Tensor | None = None,
    space_order: tuple[str, ...] = SPACE_ORDER,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode a predicted AmpUnder deficit into its current named space."""

    _require_panel(history, base)
    if operator_residual.shape != (base.shape[0],):
        raise ValueError("amp_under operator residual must have shape [N]")
    if basis is None:
        if "amp_under" not in space_order:
            raise ValueError("amp_under is absent from the requested decomposition")
        basis = semantic_penalty_bases(
            history, base, space_order=space_order
        )["amp_under"]
    if basis.shape != (base.shape[0], 1, HORIZON):
        raise ValueError("amp_under basis must be [N,1,96]")
    coordinates = _std_observable_coordinates_from_residual(
        base, basis, operator_residual
    )
    correction = torch.einsum("nk,nkh->nh", coordinates, basis)
    return correction, coordinates, basis


def decode_diff_amp_operator_residual(
    history: torch.Tensor,
    base: torch.Tensor,
    operator_residual: torch.Tensor,
    *,
    space_order: tuple[str, ...] = SPACE_ORDER,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode a predicted DiffAmp error into a DiffAmp-space residual curve.

    This path is target-free at inference: ``operator_residual`` is supplied by
    the isolated estimator, while the current basis/Jacobian uses only causal
    history and the frozen backbone forecast.
    """

    _require_panel(history, base)
    if operator_residual.shape != (base.shape[0],):
        raise ValueError("diff_amp operator residual must have shape [N]")
    if "diff_amp" not in space_order:
        raise ValueError("diff_amp is absent from the requested decomposition")
    basis = semantic_penalty_bases(
        history, base, space_order=space_order
    )["diff_amp"]
    coordinates = _std_observable_coordinates_from_residual(
        torch.diff(base, dim=1),
        torch.diff(basis, dim=2),
        operator_residual,
    )
    correction = torch.einsum("nk,nkh->nh", coordinates, basis)
    return correction, coordinates, basis


@torch.no_grad()
def penalty_native_residual_component(
    name: str,
    history: torch.Tensor,
    base: torch.Tensor,
    target: torch.Tensor,
    *,
    space_order: tuple[str, ...] = SPACE_ORDER,
    line_search_steps: int = 17,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode the *actual named error residual* and decode a residual curve.

    Unlike :func:`penalty_aligned_residual_component`, coordinates are not an
    Euclidean projection of ``target-base``. They are fitted in the original
    penalty's observable domain (D1, D2, direction hinge, volatility, range,
    amplitude, or centered unit shape). The fixed target-free named basis then
    decodes those coordinates into a time-domain residual. A final scalar risk
    guard selects the MSE-best point that does not worsen the exact penalty;
    exact NOOP is always available.
    """

    _require_panel(history, base)
    if target.shape != base.shape:
        raise ValueError("penalty-native target must match history/base")
    if name not in space_order:
        raise ValueError(f"{name} is absent from the requested decomposition")
    if int(line_search_steps) < 2:
        raise ValueError("penalty-native alignment needs at least two steps")
    basis = semantic_penalty_bases(
        history, base, space_order=space_order
    )[name]

    if name == "delta":
        operator_basis = torch.diff(basis, dim=2)
        operator_residual = torch.diff(target, dim=1) - torch.diff(base, dim=1)
        coordinates = _solve_operator_residual(operator_basis, operator_residual)
    elif name == "d2_match":
        operator_basis = torch.diff(basis, n=2, dim=2)
        operator_residual = torch.diff(target, n=2, dim=1) - torch.diff(
            base, n=2, dim=1
        )
        coordinates = _solve_operator_residual(operator_basis, operator_residual)
    elif name == "direction":
        operator_basis = torch.diff(basis, dim=2)
        target_delta = torch.diff(target, dim=1)
        base_delta = torch.diff(base, dim=1)
        deficit = (
            target_delta.abs() - base_delta * target_delta.sign()
        ).clamp_min(0.0)
        operator_residual = target_delta.sign() * deficit
        coordinates = _solve_operator_residual(operator_basis, operator_residual)
    elif name == "diff_amp":
        coordinates = _std_observable_coordinates(
            torch.diff(base, dim=1),
            torch.diff(target, dim=1),
            torch.diff(basis, dim=2),
            one_sided=False,
        )
    elif name == "amp_under":
        coordinates = _std_observable_coordinates(
            base, target, basis, one_sided=True
        )
    elif name == "range":
        maximum = base.argmax(dim=1)
        minimum = base.argmin(dim=1)
        rows = torch.arange(base.shape[0], device=base.device)
        jacobian = basis[rows, :, maximum] - basis[rows, :, minimum]
        residual = (
            target.amax(dim=1)
            - target.amin(dim=1)
            - base.amax(dim=1)
            + base.amin(dim=1)
        )
        coordinates = jacobian * residual.unsqueeze(1) / jacobian.square().sum(
            dim=1, keepdim=True
        ).clamp_min(1.0e-8)
    elif name == "corr":
        base_shape = _center_unit(base)
        target_shape = _center_unit(target)
        centered_basis = basis - basis.mean(dim=2, keepdim=True)
        scale = (
            (base - base.mean(dim=1, keepdim=True))
            .square()
            .mean(dim=1, keepdim=True)
            .sqrt()
            .clamp_min(1.0e-6)
        )
        radial = torch.einsum(
            "nh,nkh->nk", base_shape, centered_basis
        ) / float(HORIZON)
        operator_basis = (
            centered_basis - radial.unsqueeze(2) * base_shape.unsqueeze(1)
        ) / scale.unsqueeze(2)
        coordinates = _solve_operator_residual(
            operator_basis, target_shape - base_shape
        )
    else:
        raise ValueError(f"unsupported penalty-native residual: {name}")

    raw_action = torch.einsum("nk,nkh->nh", coordinates, basis)
    alpha = torch.linspace(
        0.0,
        1.0,
        int(line_search_steps),
        device=base.device,
        dtype=base.dtype,
    )
    candidate = base.unsqueeze(0) + alpha[:, None, None] * raw_action.unsqueeze(0)
    expanded_target = target.unsqueeze(0).expand_as(candidate)
    compute = build_penalty_compute([name], jump_thr=0.6)
    penalties = compute(
        candidate.reshape(-1, 1, HORIZON),
        expanded_target.reshape(-1, 1, HORIZON),
    ).reshape(alpha.numel(), base.shape[0])
    squared_error = (candidate - expanded_target).square().mean(dim=2)
    allowed = penalties <= penalties[0:1] + 1.0e-7
    guarded_error = torch.where(
        allowed, squared_error, torch.full_like(squared_error, torch.inf)
    )
    selected = alpha[guarded_error.argmin(dim=0)]
    coordinates = coordinates * selected.unsqueeze(1)
    action = raw_action * selected.unsqueeze(1)
    return action, coordinates, basis


class SemanticPenaltyAdapter(nn.Module):
    """One isolated coordinate predictor for one existing penalty."""

    penalty_name: str

    def __init__(
        self,
        penalty_name: str,
        *,
        contract: SemanticPenaltyContract | None = None,
    ) -> None:
        super().__init__()
        if penalty_name not in CONTRACTS:
            raise ValueError(f"unsupported semantic penalty adapter: {penalty_name}")
        self.penalty_name = penalty_name
        self._contract = CONTRACTS[penalty_name] if contract is None else contract
        if self._contract.name != penalty_name:
            raise ValueError("semantic adapter contract/name mismatch")
        contract = self._contract
        self.network = nn.Sequential(
            nn.LayerNorm(contract.feature_width),
            nn.Linear(contract.feature_width, 32),
            nn.SiLU(),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, contract.coordinates),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    @property
    def contract(self) -> SemanticPenaltyContract:
        return self._contract

    def forward(
        self,
        history: torch.Tensor,
        base: torch.Tensor,
        *,
        features: torch.Tensor | None = None,
        basis: torch.Tensor | None = None,
        causal_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _require_panel(history, base)
        if features is None:
            features = semantic_penalty_features(
                self.penalty_name,
                history,
                base,
                causal_state=causal_state,
            )
        if basis is None:
            basis = semantic_penalty_bases(history, base)[self.penalty_name]
        if features.shape != (history.shape[0], self.contract.feature_width):
            raise ValueError("precomputed semantic features violate adapter contract")
        if basis.shape != (
            history.shape[0],
            self.contract.coordinates,
            HORIZON,
        ):
            raise ValueError("precomputed semantic basis violates adapter contract")
        raw = self.network(features)
        scale = history.std(dim=1, unbiased=False, keepdim=True).clamp_min(1.0e-4)
        scale = 0.5 * scale
        if self.contract.positive_coordinates:
            centered_softplus = F.softplus(raw) - F.softplus(torch.zeros_like(raw))
            coordinates = centered_softplus.clamp_min(0.0) * scale
        else:
            coordinates = raw.tanh() * scale
        correction = torch.einsum("nk,nkh->nh", coordinates, basis)
        return correction, coordinates


class AmpUnderSemanticAdapter(SemanticPenaltyAdapter):
    def __init__(self) -> None:
        super().__init__("amp_under")


class RangeSemanticAdapter(SemanticPenaltyAdapter):
    def __init__(self) -> None:
        super().__init__("range")


class DiffAmpSemanticAdapter(SemanticPenaltyAdapter):
    def __init__(self) -> None:
        super().__init__("diff_amp")


class DeltaSemanticAdapter(SemanticPenaltyAdapter):
    def __init__(self) -> None:
        super().__init__("delta")


class DirectionSemanticAdapter(SemanticPenaltyAdapter):
    def __init__(self) -> None:
        super().__init__("direction")


class D2MatchSemanticAdapter(SemanticPenaltyAdapter):
    def __init__(self) -> None:
        super().__init__("d2_match")


class CorrSemanticAdapter(SemanticPenaltyAdapter):
    def __init__(self) -> None:
        super().__init__("corr")


class PenaltyNativeResidualAdapter(SemanticPenaltyAdapter):
    """v2 signed-coordinate head for one actual penalty-domain residual."""

    def __init__(self, penalty_name: str) -> None:
        super().__init__(
            penalty_name, contract=PENALTY_NATIVE_CONTRACTS[penalty_name]
        )


class DeltaMaturedResidualPenaltyNativeAdapter(SemanticPenaltyAdapter):
    """Delta v2 candidate with a Delta-only matured causal residual state."""

    def __init__(self) -> None:
        super().__init__("delta", contract=DELTA_MATURED_RESIDUAL_CONTRACT)


class DeltaOperatorSequencePenaltyNativeAdapter(nn.Module):
    """Delta-specific sequence estimator with an analytic D1-space decoder."""

    penalty_name = "delta"

    def __init__(self) -> None:
        super().__init__()
        self._contract = DELTA_OPERATOR_SEQUENCE_CONTRACT
        self.input_projection = nn.Conv1d(12, 32, kernel_size=5, padding=2)
        self.residual_block = nn.Sequential(
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2, groups=32),
            nn.Conv1d(32, 32, kernel_size=1),
        )
        self.output_head = nn.Sequential(
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv1d(32, 1, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)
        position = torch.arange(HORIZON, dtype=torch.float32)
        angle = 2.0 * math.pi * position / float(HORIZON)
        self.register_buffer(
            "phase", torch.stack([angle.sin(), angle.cos()])[None], persistent=False
        )

    @property
    def contract(self) -> SemanticPenaltyContract:
        return self._contract

    def forward(
        self,
        history: torch.Tensor,
        base: torch.Tensor,
        *,
        features: torch.Tensor | None = None,
        basis: torch.Tensor | None = None,
        causal_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _require_panel(history, base)
        carrier = features if features is not None else causal_state
        if carrier is None or carrier.shape != (history.shape[0], 8, HORIZON):
            raise ValueError("Delta operator sequence carrier must be [N,8,96]")
        if basis is None:
            basis = semantic_penalty_bases(history, base)["delta"]
        if basis.shape != (history.shape[0], 8, HORIZON):
            raise ValueError("Delta operator sequence basis must be [N,8,96]")

        current = torch.stack([_d1(history), _d1(base)], dim=1)
        carrier = carrier.to(device=base.device, dtype=base.dtype)
        scale = torch.cat([current, carrier], dim=1).square().mean(
            dim=(1, 2), keepdim=True
        ).sqrt().clamp_min(1.0e-4)
        phase = self.phase.to(device=base.device, dtype=base.dtype).expand(
            history.shape[0], -1, -1
        )
        sequence = torch.cat([current / scale, carrier / scale, phase], dim=1)
        hidden = self.input_projection(sequence)
        hidden = hidden + self.residual_block(hidden)
        predicted_d1 = self.output_head(hidden).squeeze(1).tanh()
        predicted_d1 = 0.5 * scale.squeeze(2) * predicted_d1

        operator_basis = torch.diff(basis, dim=2)
        target_operator = predicted_d1[:, 1:]
        gram = torch.einsum("nkh,njh->nkj", operator_basis, operator_basis)
        cross = torch.einsum("nkh,nh->nk", operator_basis, target_operator)
        identity = torch.eye(8, device=base.device, dtype=base.dtype)[None]
        coordinates = torch.linalg.solve(gram + 1.0e-4 * identity, cross.unsqueeze(2))
        coordinates = coordinates.squeeze(2)
        correction = torch.einsum("nk,nkh->nh", coordinates, basis)
        return correction, coordinates


class DeltaPatchSignedMemoryPenaltyNativeAdapter(nn.Module):
    """Patch-local signed mixture of actual matured Delta residual memories."""

    penalty_name = "delta"

    def __init__(self) -> None:
        super().__init__()
        self._contract = DELTA_PATCH_SIGNED_MEMORY_CONTRACT
        self.query_encoder = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.memory_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.signed_gate = nn.Sequential(
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.signed_gate[-1].weight)
        nn.init.zeros_(self.signed_gate[-1].bias)

    @property
    def contract(self) -> SemanticPenaltyContract:
        return self._contract

    def forward(
        self,
        history: torch.Tensor,
        base: torch.Tensor,
        *,
        features: torch.Tensor | None = None,
        basis: torch.Tensor | None = None,
        causal_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _require_panel(history, base)
        carrier = features if features is not None else causal_state
        if carrier is None or carrier.shape != (history.shape[0], 8, HORIZON):
            raise ValueError("Delta patch memory carrier must be [N,8,96]")
        if basis is None:
            basis = semantic_penalty_bases(history, base)["delta"]
        if basis.shape != (history.shape[0], 8, HORIZON):
            raise ValueError("Delta patch memory basis must be [N,8,96]")

        carrier = carrier.to(device=base.device, dtype=base.dtype)
        history_d1 = _d1(history).reshape(-1, PATCHES, PATCH_STEPS)
        base_d1 = _d1(base).reshape(-1, PATCHES, PATCH_STEPS)
        memories = carrier.reshape(
            -1, 8, PATCHES, PATCH_STEPS
        ).permute(0, 2, 1, 3)
        scale = torch.cat(
            [history_d1[:, :, None], base_d1[:, :, None], memories], dim=2
        ).square().mean(dim=(2, 3), keepdim=True).sqrt().clamp_min(1.0e-4)

        query_input = torch.stack([history_d1, base_d1], dim=2) / scale
        query = self.query_encoder(
            query_input.reshape(-1, 2, PATCH_STEPS)
        ).squeeze(2).reshape(-1, PATCHES, 16)
        key = self.memory_encoder(
            (memories / scale).reshape(-1, 1, PATCH_STEPS)
        ).squeeze(2).reshape(-1, PATCHES, 8, 16)
        logits = torch.einsum("npd,npvd->npv", query, key) / math.sqrt(16.0)
        attention = logits.softmax(dim=2)
        mixture = torch.einsum("npv,npvh->nph", attention, memories)
        context = torch.einsum("npv,npvd->npd", attention, key)
        gate = self.signed_gate(torch.cat([query, context], dim=2)).tanh()
        predicted_d1 = (gate * mixture).reshape(-1, HORIZON)

        operator_basis = torch.diff(basis, dim=2)
        gram = torch.einsum("nkh,njh->nkj", operator_basis, operator_basis)
        cross = torch.einsum(
            "nkh,nh->nk", operator_basis, predicted_d1[:, 1:]
        )
        identity = torch.eye(8, device=base.device, dtype=base.dtype)[None]
        coordinates = torch.linalg.solve(
            gram + 1.0e-4 * identity, cross.unsqueeze(2)
        ).squeeze(2)
        correction = torch.einsum("nk,nkh->nh", coordinates, basis)
        return correction, coordinates


class DeltaPhysicalMemoryPenaltyNativeAdapter(nn.Module):
    """Same-phase raw-D1 memory retrieval with no free waveform output."""

    penalty_name = "delta"

    def __init__(self) -> None:
        super().__init__()
        self._contract = DELTA_PHYSICAL_MEMORY_CONTRACT
        self.query_encoder = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.memory_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.signed_gate = nn.Sequential(
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.signed_gate[-1].weight)
        nn.init.zeros_(self.signed_gate[-1].bias)

    @property
    def contract(self) -> SemanticPenaltyContract:
        return self._contract

    def forward(
        self,
        history: torch.Tensor,
        base: torch.Tensor,
        *,
        features: torch.Tensor | None = None,
        basis: torch.Tensor | None = None,
        causal_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _require_panel(history, base)
        memory = features if features is not None else causal_state
        if memory is None or memory.shape != (history.shape[0], 28, HORIZON):
            raise ValueError("Delta physical memory must be [N,28,96]")
        if basis is None:
            basis = semantic_penalty_bases(history, base)["delta"]
        if basis.shape != (history.shape[0], 8, HORIZON):
            raise ValueError("Delta physical-memory basis must be [N,8,96]")

        memory = memory.to(device=base.device, dtype=base.dtype)
        history_d1 = _d1(history).reshape(-1, PATCHES, PATCH_STEPS)
        base_d1 = _d1(base).reshape(-1, PATCHES, PATCH_STEPS)
        memories = memory.reshape(
            -1, 28, PATCHES, PATCH_STEPS
        ).permute(0, 2, 1, 3)
        scale = torch.cat(
            [history_d1[:, :, None], base_d1[:, :, None], memories], dim=2
        ).square().mean(dim=(2, 3), keepdim=True).sqrt().clamp_min(1.0e-4)
        query_input = torch.stack([history_d1, base_d1], dim=2) / scale
        query = self.query_encoder(
            query_input.reshape(-1, 2, PATCH_STEPS)
        ).squeeze(2).reshape(-1, PATCHES, 16)
        key = self.memory_encoder(
            (memories / scale).reshape(-1, 1, PATCH_STEPS)
        ).squeeze(2).reshape(-1, PATCHES, 28, 16)
        logits = torch.einsum("npd,npvd->npv", query, key) / math.sqrt(16.0)
        attention = logits.softmax(dim=2)
        mixture = torch.einsum("npv,npvh->nph", attention, memories)
        context = torch.einsum("npv,npvd->npd", attention, key)
        gate = self.signed_gate(torch.cat([query, context], dim=2)).tanh()
        predicted_d1 = (gate * mixture).reshape(-1, HORIZON)

        operator_basis = torch.diff(basis, dim=2)
        gram = torch.einsum("nkh,njh->nkj", operator_basis, operator_basis)
        cross = torch.einsum(
            "nkh,nh->nk", operator_basis, predicted_d1[:, 1:]
        )
        identity = torch.eye(8, device=base.device, dtype=base.dtype)[None]
        coordinates = torch.linalg.solve(
            gram + 1.0e-4 * identity, cross.unsqueeze(2)
        ).squeeze(2)
        correction = torch.einsum("nk,nkh->nh", coordinates, basis)
        return correction, coordinates


class DeltaPhysicalComponentMemoryPenaltyNativeAdapter(
    DeltaPhysicalMemoryPenaltyNativeAdapter
):
    """Same network using only source-local decomposed Delta values."""

    def __init__(self) -> None:
        super().__init__()
        self._contract = DELTA_PHYSICAL_COMPONENT_MEMORY_CONTRACT


class DeltaWhereWhatComponentMemoryPenaltyNativeAdapter(nn.Module):
    """Local where->what Delta retrieval with revision/value isolation."""

    penalty_name = "delta"

    def __init__(self) -> None:
        super().__init__()
        self._contract = DELTA_WHERE_WHAT_COMPONENT_MEMORY_CONTRACT
        self.what_current_encoder = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.what_memory_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.what_signed_gate = nn.Sequential(
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )
        self.where_current_encoder = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.where_revision_encoder = nn.Sequential(
            nn.Conv1d(14, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.where_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.what_signed_gate[-1].weight)
        nn.init.zeros_(self.what_signed_gate[-1].bias)
        nn.init.zeros_(self.where_head[-1].weight)
        nn.init.constant_(self.where_head[-1].bias, 4.0)

    @property
    def contract(self) -> SemanticPenaltyContract:
        return self._contract

    def _split_state(
        self, state: torch.Tensor, count: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if state.shape != (count, 42, HORIZON):
            raise ValueError("Delta where/what state must be [N,42,96]")
        memory = state[:, :28].reshape(
            count, 28, PATCHES, PATCH_STEPS
        ).permute(0, 2, 1, 3)
        revisions = state[:, 28:35].reshape(
            count, 7, PATCHES, PATCH_STEPS
        ).permute(0, 2, 1, 3)
        masks = state[:, 35:42].reshape(
            count, 7, PATCHES, PATCH_STEPS
        ).permute(0, 2, 1, 3)
        return memory, revisions, masks

    def retrieve_residual_value(
        self,
        history: torch.Tensor,
        base: torch.Tensor,
        memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return what-mixture/gate/context without any revision input."""

        count = history.shape[0]
        current = torch.stack([_d1(history), _d1(base)], dim=1).reshape(
            count, 2, PATCHES, PATCH_STEPS
        ).permute(0, 2, 1, 3)
        scale = torch.cat(
            [current.reshape(count, PATCHES, -1), memory.reshape(count, PATCHES, -1)],
            dim=2,
        ).square().mean(dim=2, keepdim=True).sqrt().clamp_min(1.0e-4)
        query = self.what_current_encoder(
            (current / scale[:, :, None, :]).reshape(-1, 2, PATCH_STEPS)
        ).squeeze(2).reshape(count, PATCHES, 16)
        key = self.what_memory_encoder(
            (memory / scale[:, :, None, :]).reshape(-1, 1, PATCH_STEPS)
        ).squeeze(2).reshape(count, PATCHES, 28, 16)
        attention = (
            torch.einsum("npd,npvd->npv", query, key) / math.sqrt(16.0)
        ).softmax(dim=2)
        mixture = torch.einsum("npv,npvh->nph", attention, memory)
        context = torch.einsum("npv,npvd->npd", attention, key)
        gate = self.what_signed_gate(torch.cat([query, context], dim=2)).tanh()
        return mixture, gate, context

    def locate(
        self,
        history: torch.Tensor,
        base: torch.Tensor,
        revisions: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Predict only local [0,1] suppression from current/revision state."""

        count = history.shape[0]
        current = torch.stack([_d1(history), _d1(base)], dim=1).reshape(
            count, 2, PATCHES, PATCH_STEPS
        ).permute(0, 2, 1, 3)
        scale = torch.cat(
            [current.reshape(count, PATCHES, -1), revisions.reshape(count, PATCHES, -1)],
            dim=2,
        ).square().mean(dim=2, keepdim=True).sqrt().clamp_min(1.0e-4)
        current_code = self.where_current_encoder(
            (current / scale[:, :, None, :]).reshape(-1, 2, PATCH_STEPS)
        ).squeeze(2).reshape(count, PATCHES, 16)
        revision_input = torch.cat(
            [revisions / scale[:, :, None, :], masks], dim=2
        )
        revision_code = self.where_revision_encoder(
            revision_input.reshape(-1, 14, PATCH_STEPS)
        ).squeeze(2).reshape(count, PATCHES, 16)
        return self.where_head(
            torch.cat([current_code, revision_code], dim=2)
        ).squeeze(2).sigmoid()

    def forward_with_location(
        self,
        history: torch.Tensor,
        base: torch.Tensor,
        *,
        features: torch.Tensor,
        basis: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _require_panel(history, base)
        state = features.to(device=base.device, dtype=base.dtype)
        memory, revisions, masks = self._split_state(state, history.shape[0])
        mixture, gate, _context = self.retrieve_residual_value(
            history, base, memory
        )
        location = self.locate(history, base, revisions, masks)
        predicted_d1 = (location.unsqueeze(2) * gate * mixture).reshape(
            -1, HORIZON
        )
        if basis is None:
            basis = semantic_penalty_bases(history, base)["delta"]
        if basis.shape != (history.shape[0], 8, HORIZON):
            raise ValueError("Delta where/what basis must be [N,8,96]")
        operator_basis = torch.diff(basis, dim=2)
        gram = torch.einsum("nkh,njh->nkj", operator_basis, operator_basis)
        cross = torch.einsum(
            "nkh,nh->nk", operator_basis, predicted_d1[:, 1:]
        )
        identity = torch.eye(8, device=base.device, dtype=base.dtype)[None]
        coordinates = torch.linalg.solve(
            gram + 1.0e-4 * identity, cross.unsqueeze(2)
        ).squeeze(2)
        correction = torch.einsum("nk,nkh->nh", coordinates, basis)
        return correction, coordinates, location

    def forward(
        self,
        history: torch.Tensor,
        base: torch.Tensor,
        *,
        features: torch.Tensor | None = None,
        basis: torch.Tensor | None = None,
        causal_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = features if features is not None else causal_state
        if state is None:
            raise ValueError("Delta where/what adapter requires its isolated state")
        correction, coordinates, _location = self.forward_with_location(
            history, base, features=state, basis=basis
        )
        return correction, coordinates


class DeltaCausalTransitionPenaltyNativeAdapter(nn.Module):
    """Shared next-step Delta dynamics with a fixed Level-free decoder.

    The network does not own eight independent position heads.  A history
    encoder initializes one recurrent state and the same transition kernel is
    rolled over all 96 forecast leads.  Its padded D1 action is finally
    projected into the current target-free orthogonal Delta basis, so the
    emitted time-domain correction cannot leave the named residual space.
    """

    penalty_name = "delta"

    def __init__(self) -> None:
        super().__init__()
        self._contract = DELTA_CAUSAL_TRANSITION_CONTRACT
        self.history_encoder = nn.GRU(
            input_size=4, hidden_size=32, batch_first=True
        )
        self.initial_state = nn.Linear(32, 48)
        self.transition = nn.GRU(
            input_size=24, hidden_size=48, batch_first=True
        )
        self.output_norm = nn.LayerNorm(48)
        self.output_head = nn.Linear(48, 1)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)
        position = torch.arange(HORIZON, dtype=torch.float32)
        angle = 2.0 * math.pi * position / float(HORIZON)
        self.register_buffer(
            "relative_phase",
            torch.stack([angle.sin(), angle.cos()], dim=1),
            persistent=False,
        )

    @property
    def contract(self) -> SemanticPenaltyContract:
        return self._contract

    def _split_state(
        self, state: torch.Tensor, count: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if state.shape != (count, 42, HORIZON):
            raise ValueError("Delta causal-transition state must be [N,42,96]")
        return state[:, :28], state[:, 28:35], state[:, 35:42]

    def predict_d1(
        self,
        history: torch.Tensor,
        base: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Predict one Level-free padded D1 curve by recurrent rollout."""

        _require_panel(history, base)
        state = state.to(device=base.device, dtype=base.dtype)
        memory, revisions, masks = self._split_state(state, history.shape[0])
        revisions = revisions * masks
        history_d1 = _d1(history)
        base_d1 = _d1(base)
        motion_scale = torch.cat(
            [history_d1[:, None], base_d1[:, None], memory, revisions], dim=1
        ).square().mean(dim=(1, 2)).sqrt().clamp_min(1.0e-4)
        value_scale = history.std(
            dim=1, unbiased=False, keepdim=True
        ).clamp_min(1.0e-4)
        history_centered = (history - history[:, -1:]) / value_scale
        phase = self.relative_phase.to(device=base.device, dtype=base.dtype)
        phase = phase.unsqueeze(0).expand(history.shape[0], -1, -1)
        history_tokens = torch.cat(
            [
                history_centered.unsqueeze(2),
                (history_d1 / motion_scale[:, None]).unsqueeze(2),
                phase,
            ],
            dim=2,
        )
        _encoded, history_state = self.history_encoder(history_tokens)
        initial = torch.tanh(self.initial_state(history_state[-1])).unsqueeze(0)

        memory_summary = torch.stack(
            [
                memory[:, 0],
                memory[:, :3].mean(dim=1),
                memory[:, :7].mean(dim=1),
                memory.mean(dim=1),
            ],
            dim=1,
        )
        base_centered = (base - history[:, -1:]) / value_scale
        decoder_tokens = torch.cat(
            [
                base_centered.unsqueeze(2),
                (base_d1 / motion_scale[:, None]).unsqueeze(2),
                history_centered.unsqueeze(2),
                (history_d1 / motion_scale[:, None]).unsqueeze(2),
                (memory_summary / motion_scale[:, None, None]).transpose(1, 2),
                (revisions / motion_scale[:, None, None]).transpose(1, 2),
                masks.transpose(1, 2),
                phase,
            ],
            dim=2,
        )
        hidden, _state = self.transition(decoder_tokens, initial)
        raw = self.output_head(self.output_norm(hidden)).squeeze(2).tanh()
        predicted = 0.5 * motion_scale[:, None] * raw
        # The integration constant belongs to Level.  Lane zero is therefore
        # a fixed gauge value and only the 95 within-horizon transitions are
        # decoded by Delta.
        return torch.cat([torch.zeros_like(predicted[:, :1]), predicted[:, 1:]], dim=1)

    def forward(
        self,
        history: torch.Tensor,
        base: torch.Tensor,
        *,
        features: torch.Tensor | None = None,
        basis: torch.Tensor | None = None,
        causal_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = features if features is not None else causal_state
        if state is None:
            raise ValueError("Delta causal-transition adapter requires its isolated state")
        if basis is None:
            basis = semantic_penalty_bases(history, base)["delta"]
        if basis.shape != (history.shape[0], 8, HORIZON):
            raise ValueError("Delta causal-transition basis must be [N,8,96]")
        predicted_d1 = self.predict_d1(history, base, state)
        operator_basis = torch.diff(basis, dim=2)
        gram = torch.einsum("nkh,njh->nkj", operator_basis, operator_basis)
        cross = torch.einsum(
            "nkh,nh->nk", operator_basis, predicted_d1[:, 1:]
        )
        identity = torch.eye(8, device=base.device, dtype=base.dtype)[None]
        coordinates = torch.linalg.solve(
            gram + 1.0e-4 * identity, cross.unsqueeze(2)
        ).squeeze(2)
        correction = torch.einsum("nk,nkh->nh", coordinates, basis)
        return correction, coordinates


class DeltaPhysicalKeyValuePenaltyNativeAdapter(nn.Module):
    """Causal state-matched Delta residual retrieval with analytic decoding."""

    penalty_name = "delta"

    def __init__(self) -> None:
        super().__init__()
        self._contract = DELTA_PHYSICAL_KEY_VALUE_CONTRACT
        self.query_encoder = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.key_encoder = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.signed_gate = nn.Sequential(
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.signed_gate[-1].weight)
        nn.init.zeros_(self.signed_gate[-1].bias)

    @property
    def contract(self) -> SemanticPenaltyContract:
        return self._contract

    def forward(
        self,
        history: torch.Tensor,
        base: torch.Tensor,
        *,
        features: torch.Tensor | None = None,
        basis: torch.Tensor | None = None,
        causal_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _require_panel(history, base)
        memory = features if features is not None else causal_state
        expected = (history.shape[0], 28, 3, HORIZON)
        if memory is None or memory.shape != expected:
            raise ValueError("Delta physical key/value memory must be [N,28,3,96]")
        if basis is None:
            basis = semantic_penalty_bases(history, base)["delta"]
        if basis.shape != (history.shape[0], 8, HORIZON):
            raise ValueError("Delta key/value basis must be [N,8,96]")

        memory = memory.to(device=base.device, dtype=base.dtype)
        count = history.shape[0]
        current = torch.stack([_d1(history), _d1(base)], dim=1).reshape(
            count, 2, PATCHES, PATCH_STEPS
        ).permute(0, 2, 1, 3)
        records = memory.reshape(
            count, 28, 3, PATCHES, PATCH_STEPS
        ).permute(0, 3, 1, 2, 4)
        keys = records[:, :, :, :2]
        values = records[:, :, :, 2]
        scale = torch.cat(
            [current.reshape(count, PATCHES, -1), records.reshape(count, PATCHES, -1)],
            dim=2,
        ).square().mean(dim=2, keepdim=True).sqrt().clamp_min(1.0e-4)
        query = self.query_encoder(
            (current / scale[:, :, None, :]).reshape(-1, 2, PATCH_STEPS)
        ).squeeze(2).reshape(count, PATCHES, 16)
        key = self.key_encoder(
            (keys / scale[:, :, None, None, :]).reshape(-1, 2, PATCH_STEPS)
        ).squeeze(2).reshape(count, PATCHES, 28, 16)
        logits = torch.einsum("npd,npvd->npv", query, key) / math.sqrt(16.0)
        attention = logits.softmax(dim=2)
        mixture = torch.einsum("npv,npvh->nph", attention, values)
        context = torch.einsum("npv,npvd->npd", attention, key)
        gate = self.signed_gate(torch.cat([query, context], dim=2)).tanh()
        predicted_d1 = (gate * mixture).reshape(count, HORIZON)

        operator_basis = torch.diff(basis, dim=2)
        gram = torch.einsum("nkh,njh->nkj", operator_basis, operator_basis)
        cross = torch.einsum(
            "nkh,nh->nk", operator_basis, predicted_d1[:, 1:]
        )
        identity = torch.eye(8, device=base.device, dtype=base.dtype)[None]
        coordinates = torch.linalg.solve(
            gram + 1.0e-4 * identity, cross.unsqueeze(2)
        ).squeeze(2)
        correction = torch.einsum("nk,nkh->nh", coordinates, basis)
        return correction, coordinates


class AmpUnderPhysicalScalarPenaltyNativeAdapter(nn.Module):
    """Positive scalar AmpUnder estimator with analytic residual decoding."""

    penalty_name = "amp_under"

    def __init__(self) -> None:
        super().__init__()
        self._contract = AMP_UNDER_PHYSICAL_SCALAR_CONTRACT
        self.memory_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.context_encoder = nn.Sequential(
            nn.LayerNorm(51),
            nn.Linear(51, 16),
            nn.SiLU(),
        )
        self.fuse = nn.Sequential(nn.Linear(32, 16), nn.SiLU())
        self.magnitude_head = nn.Linear(16, 1)
        nn.init.zeros_(self.magnitude_head.weight)
        nn.init.zeros_(self.magnitude_head.bias)

    @property
    def contract(self) -> SemanticPenaltyContract:
        return self._contract

    def predict_operator_residual(
        self,
        history: torch.Tensor,
        base: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        _require_panel(history, base)
        if features.shape != (history.shape[0], 52):
            raise ValueError("AmpUnder physical scalar state must be [N,52]")
        features = features.to(device=base.device, dtype=base.dtype)
        scale = torch.maximum(
            history.std(dim=1, unbiased=True),
            base.std(dim=1, unbiased=True),
        ).clamp_min(1.0e-4)
        memory = features[:, :28] / scale.unsqueeze(1)
        memory_code = self.memory_encoder(memory.unsqueeze(1)).squeeze(2)
        context = torch.cat([_amp_under_features(history, base), features[:, 28:]], dim=1)
        context_code = self.context_encoder(context)
        latent = self.fuse(torch.cat([memory_code, context_code], dim=1))
        raw = self.magnitude_head(latent).squeeze(1)
        centered = torch.where(
            raw >= 0.0,
            F.softplus(raw) - math.log(2.0),
            torch.zeros_like(raw),
        )
        return scale * centered.tanh()

    def forward(
        self,
        history: torch.Tensor,
        base: torch.Tensor,
        *,
        features: torch.Tensor | None = None,
        basis: torch.Tensor | None = None,
        causal_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = features if features is not None else causal_state
        if state is None:
            raise ValueError("AmpUnder physical scalar adapter requires its state")
        operator_residual = self.predict_operator_residual(history, base, state)
        correction, coordinates, _basis = decode_amp_under_operator_residual(
            history, base, operator_residual, basis=basis
        )
        return correction, coordinates


class DirectionPhysicalMemoryPenaltyNativeAdapter(nn.Module):
    """Direction-only same-phase residual memory and analytic D1 decoder."""

    penalty_name = "direction"

    def __init__(self) -> None:
        super().__init__()
        self._contract = DIRECTION_PHYSICAL_MEMORY_CONTRACT
        self.query_encoder = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.memory_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.signed_gate = nn.Sequential(
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.signed_gate[-1].weight)
        nn.init.zeros_(self.signed_gate[-1].bias)

    @property
    def contract(self) -> SemanticPenaltyContract:
        return self._contract

    def forward(
        self,
        history: torch.Tensor,
        base: torch.Tensor,
        *,
        features: torch.Tensor | None = None,
        basis: torch.Tensor | None = None,
        causal_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _require_panel(history, base)
        memory = features if features is not None else causal_state
        if memory is None or memory.shape != (history.shape[0], 28, HORIZON):
            raise ValueError("Direction physical memory must be [N,28,96]")
        if basis is None:
            basis = semantic_penalty_bases(history, base)["direction"]
        if basis.shape != (history.shape[0], 8, HORIZON):
            raise ValueError("Direction physical-memory basis must be [N,8,96]")

        memory = memory.to(device=base.device, dtype=base.dtype)
        history_d1 = _d1(history).reshape(-1, PATCHES, PATCH_STEPS)
        base_d1 = _d1(base).reshape(-1, PATCHES, PATCH_STEPS)
        memories = memory.reshape(
            -1, 28, PATCHES, PATCH_STEPS
        ).permute(0, 2, 1, 3)
        scale = torch.cat(
            [history_d1[:, :, None], base_d1[:, :, None], memories], dim=2
        ).square().mean(dim=(2, 3), keepdim=True).sqrt().clamp_min(1.0e-4)
        query_input = torch.stack([history_d1, base_d1], dim=2) / scale
        query = self.query_encoder(
            query_input.reshape(-1, 2, PATCH_STEPS)
        ).squeeze(2).reshape(-1, PATCHES, 16)
        key = self.memory_encoder(
            (memories / scale).reshape(-1, 1, PATCH_STEPS)
        ).squeeze(2).reshape(-1, PATCHES, 28, 16)
        logits = torch.einsum("npd,npvd->npv", query, key) / math.sqrt(16.0)
        attention = logits.softmax(dim=2)
        mixture = torch.einsum("npv,npvh->nph", attention, memories)
        context = torch.einsum("npv,npvd->npd", attention, key)
        gate = self.signed_gate(torch.cat([query, context], dim=2)).tanh()
        predicted_d1 = (gate * mixture).reshape(-1, HORIZON)

        operator_basis = torch.diff(basis, dim=2)
        gram = torch.einsum("nkh,njh->nkj", operator_basis, operator_basis)
        cross = torch.einsum(
            "nkh,nh->nk", operator_basis, predicted_d1[:, 1:]
        )
        identity = torch.eye(8, device=base.device, dtype=base.dtype)[None]
        coordinates = torch.linalg.solve(
            gram + 1.0e-4 * identity, cross.unsqueeze(2)
        ).squeeze(2)
        correction = torch.einsum("nk,nkh->nh", coordinates, basis)
        return correction, coordinates


class DiffAmpMaturedResidualPenaltyNativeAdapter(SemanticPenaltyAdapter):
    """DiffAmp v2 candidate with a DiffAmp-only matured causal error state."""

    def __init__(self) -> None:
        super().__init__(
            "diff_amp", contract=DIFF_AMP_MATURED_RESIDUAL_CONTRACT
        )


ADAPTER_CLASSES: dict[str, type[SemanticPenaltyAdapter]] = {
    "amp_under": AmpUnderSemanticAdapter,
    "range": RangeSemanticAdapter,
    "diff_amp": DiffAmpSemanticAdapter,
    "delta": DeltaSemanticAdapter,
    "direction": DirectionSemanticAdapter,
    "d2_match": D2MatchSemanticAdapter,
    "corr": CorrSemanticAdapter,
}


def build_semantic_penalty_adapter(name: str) -> SemanticPenaltyAdapter:
    if name not in ADAPTER_CLASSES:
        raise ValueError(f"unsupported semantic penalty adapter: {name}")
    return ADAPTER_CLASSES[name]()


def build_penalty_native_residual_adapter(
    name: str, *, feature_variant: str = "current"
) -> SemanticPenaltyAdapter:
    if feature_variant == "amp_under_physical_scalar":
        if name != "amp_under":
            raise ValueError("amp_under_physical_scalar is AmpUnder-only")
        return AmpUnderPhysicalScalarPenaltyNativeAdapter()
    if feature_variant == "delta_matured_residual":
        if name != "delta":
            raise ValueError("delta_matured_residual feature variant is Delta-only")
        return DeltaMaturedResidualPenaltyNativeAdapter()
    if feature_variant == "delta_operator_sequence":
        if name != "delta":
            raise ValueError("delta_operator_sequence feature variant is Delta-only")
        return DeltaOperatorSequencePenaltyNativeAdapter()
    if feature_variant == "delta_patch_signed_memory":
        if name != "delta":
            raise ValueError("delta_patch_signed_memory feature variant is Delta-only")
        return DeltaPatchSignedMemoryPenaltyNativeAdapter()
    if feature_variant == "delta_physical_memory":
        if name != "delta":
            raise ValueError("delta_physical_memory feature variant is Delta-only")
        return DeltaPhysicalMemoryPenaltyNativeAdapter()
    if feature_variant == "delta_physical_component_memory":
        if name != "delta":
            raise ValueError("delta_physical_component_memory is Delta-only")
        return DeltaPhysicalComponentMemoryPenaltyNativeAdapter()
    if feature_variant == "delta_where_what_component_memory":
        if name != "delta":
            raise ValueError("delta_where_what_component_memory is Delta-only")
        return DeltaWhereWhatComponentMemoryPenaltyNativeAdapter()
    if feature_variant in {
        "delta_causal_transition",
        "delta_causal_transition_time_domain",
    }:
        if name != "delta":
            raise ValueError("Delta causal-transition variants are Delta-only")
        return DeltaCausalTransitionPenaltyNativeAdapter()
    if feature_variant == "delta_physical_key_value":
        if name != "delta":
            raise ValueError("delta_physical_key_value feature variant is Delta-only")
        return DeltaPhysicalKeyValuePenaltyNativeAdapter()
    if feature_variant == "direction_physical_memory":
        if name != "direction":
            raise ValueError("direction_physical_memory feature variant is Direction-only")
        return DirectionPhysicalMemoryPenaltyNativeAdapter()
    if feature_variant == "diff_amp_matured_residual":
        if name != "diff_amp":
            raise ValueError("diff_amp_matured_residual feature variant is DiffAmp-only")
        return DiffAmpMaturedResidualPenaltyNativeAdapter()
    if feature_variant != "current":
        raise ValueError(f"unsupported penalty-native feature variant: {feature_variant}")
    if name not in PENALTY_NATIVE_CONTRACTS:
        raise ValueError(f"unsupported penalty-native residual adapter: {name}")
    return PenaltyNativeResidualAdapter(name)


__all__ = [
    "ADAPTER_CLASSES",
    "CONTRACTS",
    "PENALTY_NATIVE_CONTRACTS",
    "PENALTY_NATIVE_RESIDUAL_SPACES",
    "DELTA_MATURED_RESIDUAL_CONTRACT",
    "DELTA_OPERATOR_SEQUENCE_CONTRACT",
    "DELTA_PATCH_SIGNED_MEMORY_CONTRACT",
    "DELTA_PHYSICAL_MEMORY_CONTRACT",
    "DELTA_PHYSICAL_COMPONENT_MEMORY_CONTRACT",
    "DELTA_WHERE_WHAT_COMPONENT_MEMORY_CONTRACT",
    "DELTA_CAUSAL_TRANSITION_CONTRACT",
    "DELTA_PHYSICAL_KEY_VALUE_CONTRACT",
    "DIRECTION_PHYSICAL_MEMORY_CONTRACT",
    "AMP_UNDER_PHYSICAL_SCALAR_CONTRACT",
    "DIFF_AMP_MATURED_RESIDUAL_CONTRACT",
    "HORIZON",
    "SPACE_ORDER",
    "AmpUnderSemanticAdapter",
    "CorrSemanticAdapter",
    "D2MatchSemanticAdapter",
    "DeltaSemanticAdapter",
    "DiffAmpSemanticAdapter",
    "DirectionSemanticAdapter",
    "RangeSemanticAdapter",
    "SemanticPenaltyAdapter",
    "SemanticPenaltyContract",
    "PenaltyNativeResidualAdapter",
    "DeltaMaturedResidualPenaltyNativeAdapter",
    "DeltaOperatorSequencePenaltyNativeAdapter",
    "DeltaPatchSignedMemoryPenaltyNativeAdapter",
    "DeltaPhysicalMemoryPenaltyNativeAdapter",
    "DeltaPhysicalComponentMemoryPenaltyNativeAdapter",
    "DeltaWhereWhatComponentMemoryPenaltyNativeAdapter",
    "DeltaCausalTransitionPenaltyNativeAdapter",
    "DeltaPhysicalKeyValuePenaltyNativeAdapter",
    "DirectionPhysicalMemoryPenaltyNativeAdapter",
    "AmpUnderPhysicalScalarPenaltyNativeAdapter",
    "DiffAmpMaturedResidualPenaltyNativeAdapter",
    "build_penalty_native_residual_adapter",
    "build_semantic_penalty_adapter",
    "decode_diff_amp_operator_residual",
    "decode_amp_under_operator_residual",
    "diff_amp_operator_residual",
    "amp_under_operator_residual",
    "project_residual_to_penalty_space",
    "penalty_native_residual_component",
    "semantic_penalty_bases",
    "semantic_penalty_features",
]
