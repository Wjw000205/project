"""One periodic residual-memory Amp adapter for every dataset and horizon.

The adapter has the accepted ETTm1-v5 computation graph:

1. a phase-vintage residual-state KNN conditional mean;
2. a 28-period (four groups of seven) analytic residual-memory action; and
3. a causal non-negative blend fitted only from fully matured past errors.

All arrays entering this module use the fixed canonical ``8 x 12`` period.
Native P96/p12 or P24/p3 conversion and causal indexing are parameter-free and
live in ``periodic_adapter_io``/``periodic_adapter_features``.  Consequently no
dataset name, native period, forecast horizon, or output block count appears in
the learned/retrieval kernel or its hyperparameter schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.models.periodic_adapter_config import LOCKED_KERNEL_GEOMETRY


PERIOD_STEPS = LOCKED_KERNEL_GEOMETRY.period_steps
PATCH_STEPS = LOCKED_KERNEL_GEOMETRY.patch_steps
PATCHES_PER_PERIOD = LOCKED_KERNEL_GEOMETRY.patches_per_period
MEMORY_PERIODS = LOCKED_KERNEL_GEOMETRY.carrier_periods
PERIODS_PER_GROUP = 7
MEMORY_GROUPS = MEMORY_PERIODS // PERIODS_PER_GROUP


@dataclass(frozen=True)
class UniversalPeriodicKnnAmpConfig:
    """Dataset/horizon-independent Amp retrieval hyperparameters."""

    pca_components: int = 32
    neighbors: int = 64
    reference_periods: int = 28
    calibration_periods: int = 7
    ridge_alpha: float = 10.0
    minimum_matured_rows: int = 256
    blend_history_periods: int | None = None
    seed: int = 2026

    def __post_init__(self) -> None:
        if self.pca_components <= 0 or self.neighbors <= 0:
            raise ValueError("KNN dimensions must be positive")
        if self.reference_periods <= 0 or self.calibration_periods <= 0:
            raise ValueError("physical reference durations must be positive")
        if self.ridge_alpha < 0.0 or self.minimum_matured_rows <= 0:
            raise ValueError("invalid analytic-memory regularization")
        if self.blend_history_periods is not None and self.blend_history_periods <= 0:
            raise ValueError("blend history must be expanding or positive")


@dataclass
class PeriodicKnnAmpState:
    """Fitted state for one independently calibrated channel/penalty."""

    scaler: StandardScaler
    reducer: PCA
    nearest: NearestNeighbors
    target: np.ndarray
    inner: np.ndarray
    gram: np.ndarray
    cross: np.ndarray
    count: int
    outer: np.ndarray


def remove_affine(values: np.ndarray) -> np.ndarray:
    """Remove one row-wise constant and linear phase coordinate."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("affine removal expects [row,step]")
    centered = values - values.mean(axis=1, keepdims=True)
    phase = np.linspace(-1.0, 1.0, values.shape[1], dtype=np.float64)
    phase -= phase.mean()
    coefficient = (centered * phase[None]).sum(axis=1, keepdims=True)
    coefficient /= np.square(phase).sum()
    return centered - coefficient * phase[None]


def phase_vintage_state_features(carrier: np.ndarray) -> np.ndarray:
    """Exact v5 KNN features on the canonical eight-phase vintage bank."""

    carrier = np.asarray(carrier, dtype=np.float64)
    expected = (PATCHES_PER_PERIOD, PATCHES_PER_PERIOD, PATCH_STEPS)
    if carrier.ndim != 4 or tuple(carrier.shape[1:]) != expected:
        raise ValueError(f"phase-vintage carrier must end in {expected}")
    epsilon = 1.0e-6
    scale = np.sqrt(np.square(carrier).mean(axis=(1, 2, 3))) + epsilon
    unit = carrier / scale[:, None, None, None]
    patch_rms = np.sqrt(np.square(unit).mean(axis=3)).reshape(
        -1, PATCHES_PER_PERIOD * PATCHES_PER_PERIOD
    )
    horizon_rms = patch_rms.reshape(
        -1, PATCHES_PER_PERIOD, PATCHES_PER_PERIOD
    ).mean(axis=2)
    history_rms = patch_rms.reshape(
        -1, PATCHES_PER_PERIOD, PATCHES_PER_PERIOD
    ).mean(axis=1)
    return np.concatenate(
        [
            unit.reshape(unit.shape[0], -1),
            patch_rms,
            horizon_rms,
            history_rms,
            np.log(scale)[:, None],
        ],
        axis=1,
    ).astype(np.float32)


def period_memory_curves(memory: np.ndarray) -> np.ndarray:
    """Convert old-to-recent patch memory to recent-to-old period curves."""

    memory = np.asarray(memory, dtype=np.float64)
    expected = (PATCHES_PER_PERIOD, MEMORY_PERIODS, PATCH_STEPS)
    if memory.ndim != 4 or tuple(memory.shape[1:]) != expected:
        raise ValueError(f"period memory must end in {expected}")
    curves = memory.transpose(0, 2, 1, 3).reshape(
        memory.shape[0] * MEMORY_PERIODS, PERIOD_STEPS
    )
    curves = remove_affine(curves).reshape(
        memory.shape[0], MEMORY_PERIODS, PERIOD_STEPS
    )
    # The v5 four-week equations number lags 1..28 (recent to old).  Indexing
    # is stored old-to-recent so every temporal encoder has one public order.
    return curves[:, ::-1].copy()


def fit_four_group_inner(
    curves: np.ndarray,
    target: np.ndarray,
    ridge_alpha: float,
) -> np.ndarray:
    curves = np.asarray(curves, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if curves.shape != (target.shape[0], MEMORY_PERIODS, PERIOD_STEPS):
        raise ValueError("period-memory/target shape mismatch")
    weights = np.zeros((MEMORY_GROUPS, PERIODS_PER_GROUP), dtype=np.float64)
    ridge = float(ridge_alpha) * np.eye(PERIODS_PER_GROUP, dtype=np.float64)
    flat_target = target.reshape(-1)
    for group in range(MEMORY_GROUPS):
        left = group * PERIODS_PER_GROUP
        right = left + PERIODS_PER_GROUP
        design = curves[:, left:right].transpose(0, 2, 1).reshape(
            -1, PERIODS_PER_GROUP
        )
        weights[group] = np.linalg.solve(
            design.T @ design + ridge,
            design.T @ flat_target,
        )
    return weights


def four_group_shapes(curves: np.ndarray, inner: np.ndarray) -> np.ndarray:
    curves = np.asarray(curves, dtype=np.float64)
    inner = np.asarray(inner, dtype=np.float64)
    if curves.ndim != 3 or curves.shape[1:] != (MEMORY_PERIODS, PERIOD_STEPS):
        raise ValueError("invalid period-memory curves")
    if inner.shape != (MEMORY_GROUPS, PERIODS_PER_GROUP):
        raise ValueError("invalid inner memory coefficients")
    result = np.empty((curves.shape[0], MEMORY_GROUPS, PERIOD_STEPS), dtype=np.float64)
    for group in range(MEMORY_GROUPS):
        left = group * PERIODS_PER_GROUP
        right = left + PERIODS_PER_GROUP
        result[:, group] = np.einsum(
            "ndp,d->np", curves[:, left:right], inner[group]
        )
    return result


def four_group_statistics(
    shapes: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    shapes = np.asarray(shapes, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    design = shapes.transpose(0, 2, 1).reshape(-1, MEMORY_GROUPS)
    flat_target = target.reshape(-1)
    return design.T @ design, design.T @ flat_target, int(shapes.shape[0])


def solve_four_group_outer(
    gram: np.ndarray,
    cross: np.ndarray,
    ridge_alpha: float,
) -> np.ndarray:
    return np.linalg.solve(
        np.asarray(gram, dtype=np.float64)
        + float(ridge_alpha) * np.eye(MEMORY_GROUPS, dtype=np.float64),
        np.asarray(cross, dtype=np.float64),
    )


def predict_four_group(shapes: np.ndarray, outer: np.ndarray) -> np.ndarray:
    return remove_affine(
        np.einsum(
            "ngp,g->np",
            np.asarray(shapes, dtype=np.float64),
            np.asarray(outer, dtype=np.float64),
        )
    )


def optimal_nonnegative_scale(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    denominator = float(np.square(prediction).sum())
    if denominator <= 1.0e-12:
        return 0.0
    return float(np.clip((target * prediction).sum() / denominator, 0.0, 1.0))


def box_least_squares_two(
    gram: np.ndarray,
    cross: np.ndarray,
    lower: float = 0.0,
    upper: float = 1.0,
) -> np.ndarray:
    """Exact two-variable least squares under a closed coefficient box."""

    gram = np.asarray(gram, dtype=np.float64)
    cross = np.asarray(cross, dtype=np.float64)
    candidates: list[np.ndarray] = []
    ridge = max(float(np.trace(gram)) / 2.0, 1.0) * 1.0e-12
    unconstrained = np.linalg.solve(gram + ridge * np.eye(2), cross)
    if np.all(unconstrained >= lower) and np.all(unconstrained <= upper):
        candidates.append(unconstrained)
    for first in (lower, upper):
        second = (cross[1] - gram[1, 0] * first) / max(gram[1, 1], 1.0e-30)
        candidates.append(np.asarray([first, np.clip(second, lower, upper)]))
    for second in (lower, upper):
        first = (cross[0] - gram[0, 1] * second) / max(gram[0, 0], 1.0e-30)
        candidates.append(np.asarray([np.clip(first, lower, upper), second]))
    objective = [float(weight @ gram @ weight - 2.0 * weight @ cross) for weight in candidates]
    return candidates[int(np.argmin(objective))]


def blend_row_statistics(
    residual: np.ndarray,
    knn: np.ndarray,
    four: np.ndarray,
) -> np.ndarray:
    """Sufficient statistics for bounded KNN/four-period-memory blending."""

    residual = np.asarray(residual, dtype=np.float64)
    knn = np.asarray(knn, dtype=np.float64)
    four = np.asarray(four, dtype=np.float64)
    if residual.shape != knn.shape or residual.shape != four.shape:
        raise ValueError("Amp blend arrays must have identical shapes")
    return np.stack(
        [
            np.square(knn).sum(axis=1),
            (knn * four).sum(axis=1),
            np.square(four).sum(axis=1),
            (knn * residual).sum(axis=1),
            (four * residual).sum(axis=1),
            np.ones(residual.shape[0], dtype=np.float64),
        ],
        axis=1,
    )


def weights_from_statistics(
    statistics: np.ndarray,
    minimum_rows: int,
    initial: tuple[float, float] = (1.0, 0.0),
) -> np.ndarray:
    statistics = np.asarray(statistics, dtype=np.float64)
    count = int(round(float(statistics[5])))
    if count < int(minimum_rows):
        return np.asarray(initial, dtype=np.float64)
    gram = np.asarray(
        [[statistics[0], statistics[1]], [statistics[1], statistics[2]]],
        dtype=np.float64,
    )
    return box_least_squares_two(gram, statistics[3:5])


def causal_blend(
    origins: np.ndarray,
    residual: np.ndarray,
    knn: np.ndarray,
    four: np.ndarray,
    *,
    native_period_steps: int,
    forecast_period_index: int = 0,
    minimum_matured_rows: int = 256,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Blend candidates using only earlier fully matured origin/channel rows."""

    origins = np.asarray(origins, dtype=np.int64)
    residual = np.asarray(residual, dtype=np.float64)
    knn = np.asarray(knn, dtype=np.float64)
    four = np.asarray(four, dtype=np.float64)
    if residual.ndim != 3 or residual.shape != knn.shape or residual.shape != four.shape:
        raise ValueError("causal blend expects aligned [origin,channel,period]")
    if origins.shape != (residual.shape[0],):
        raise ValueError("causal blend origin count mismatch")
    delay = (int(forecast_period_index) + 1) * int(native_period_steps)
    origin_stat = np.stack(
        [
            blend_row_statistics(residual[index], knn[index], four[index]).sum(axis=0)
            for index in range(origins.size)
        ]
    )
    prefix = np.concatenate(
        [np.zeros((1, origin_stat.shape[1])), np.cumsum(origin_stat, axis=0)],
        axis=0,
    )
    weight = np.empty((origins.size, 2), dtype=np.float64)
    matured = np.empty(origins.size, dtype=np.int64)
    for index, origin in enumerate(origins):
        right = int(np.searchsorted(origins, int(origin) - delay, side="right"))
        statistics = prefix[right]
        matured[index] = int(round(float(statistics[5])))
        weight[index] = weights_from_statistics(
            statistics, minimum_matured_rows
        )
    correction = weight[:, None, :1] * knn + weight[:, None, 1:] * four
    return correction, weight, {
        "feedback_delay_steps": delay,
        "minimum_matured_rows": int(minimum_matured_rows),
        "fallback_origins": int(np.sum(matured < int(minimum_matured_rows))),
        "mean_weights": weight.mean(axis=0).tolist(),
        "latest_weights": weight[-1].tolist(),
        "latest_matured_rows": int(matured[-1]),
    }


def causal_blend_panel(
    origins: np.ndarray,
    residual: np.ndarray,
    knn: np.ndarray,
    four: np.ndarray,
    *,
    native_period_steps: int,
    minimum_matured_rows: int = 256,
    maturity_periods: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Causally blend a variable number of physical-period instances.

    Arrays use ``[origin, instance, physical_period]``.  Instance ``s`` is
    mature only after ``(s + 1) * native_period_steps``.  The instance axis is
    data, not architecture: H96/P96 has one instance while H96/P24 has four,
    and both use one shared two-coefficient blend.
    """

    origins = np.asarray(origins, dtype=np.int64)
    residual = np.asarray(residual, dtype=np.float64)
    knn = np.asarray(knn, dtype=np.float64)
    four = np.asarray(four, dtype=np.float64)
    if residual.ndim != 3 or residual.shape != knn.shape or residual.shape != four.shape:
        raise ValueError("causal panel blend expects aligned [origin,instance,period]")
    if origins.shape != (residual.shape[0],) or residual.shape[2] != PERIOD_STEPS:
        raise ValueError("causal panel origin/period shape mismatch")
    instance_count = int(residual.shape[1])
    if maturity_periods is None:
        maturity_periods = np.arange(1, instance_count + 1, dtype=np.int64)
    maturity_periods = np.asarray(maturity_periods, dtype=np.int64)
    if maturity_periods.shape != (instance_count,) or np.any(maturity_periods <= 0):
        raise ValueError("panel maturity periods must be positive per instance")
    row_stat = np.empty((origins.size, instance_count, 6), dtype=np.float64)
    for instance in range(instance_count):
        row_stat[:, instance] = blend_row_statistics(
            residual[:, instance], knn[:, instance], four[:, instance]
        )
    weights, weight_summary = causal_panel_weights_from_row_statistics(
        origins,
        row_stat,
        native_period_steps=native_period_steps,
        minimum_matured_rows=minimum_matured_rows,
        maturity_periods=maturity_periods,
    )
    correction = (
        weights[:, None, :1] * knn + weights[:, None, 1:] * four
    )
    return correction, weights, weight_summary


def causal_panel_weights_from_row_statistics(
    origins: np.ndarray,
    row_statistics: np.ndarray,
    *,
    native_period_steps: int,
    minimum_matured_rows: int = 256,
    maturity_periods: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Derive shared causal blend weights from additive instance statistics."""

    origins = np.asarray(origins, dtype=np.int64)
    row_statistics = np.asarray(row_statistics, dtype=np.float64)
    if row_statistics.ndim != 3 or row_statistics.shape[2] != 6:
        raise ValueError("panel statistics must have shape [origin,instance,6]")
    if origins.shape != (row_statistics.shape[0],):
        raise ValueError("panel statistics origin mismatch")
    instance_count = int(row_statistics.shape[1])
    if maturity_periods is None:
        maturity_periods = np.arange(1, instance_count + 1, dtype=np.int64)
    maturity_periods = np.asarray(maturity_periods, dtype=np.int64)
    if maturity_periods.shape != (instance_count,) or np.any(maturity_periods <= 0):
        raise ValueError("panel maturity periods must be positive per instance")
    prefix = np.concatenate(
        [
            np.zeros((1, instance_count, 6), dtype=np.float64),
            np.cumsum(row_statistics, axis=0),
        ],
        axis=0,
    )
    weights = np.empty((origins.size, 2), dtype=np.float64)
    matured = np.empty(origins.size, dtype=np.int64)
    for index, origin in enumerate(origins):
        statistics = np.zeros(6, dtype=np.float64)
        for instance in range(instance_count):
            delay = int(maturity_periods[instance]) * int(native_period_steps)
            right = int(np.searchsorted(origins, int(origin) - delay, side="right"))
            statistics += prefix[right, instance]
        matured[index] = int(round(float(statistics[5])))
        weights[index] = weights_from_statistics(statistics, minimum_matured_rows)
    return weights, {
        "instance_count": instance_count,
        "feedback_delay_steps_by_instance": [
            int(periods) * int(native_period_steps) for periods in maturity_periods
        ],
        "minimum_matured_rows": int(minimum_matured_rows),
        "fallback_origins": int(np.sum(matured < int(minimum_matured_rows))),
        "mean_weights": weights.mean(axis=0).tolist(),
        "latest_weights": weights[-1].tolist(),
        "latest_matured_rows": int(matured[-1]),
    }


class UniversalPeriodicKnnAmpAdapter:
    """Fixed retrieval/analytic Amp kernel with no dataset or horizon field."""

    def __init__(self, config: UniversalPeriodicKnnAmpConfig | None = None) -> None:
        self.config = config or UniversalPeriodicKnnAmpConfig()

    def fit(
        self,
        phase_vintage: np.ndarray,
        period_memory: np.ndarray,
        target: np.ndarray,
    ) -> PeriodicKnnAmpState:
        # Target projection belongs to the shared residual decomposition before
        # this kernel.  Do not project a second time: on the ETTm1 canonical
        # clock this preserves the accepted v5 KNN arithmetic bit for bit.
        target = np.asarray(target, dtype=np.float64)
        feature = phase_vintage_state_features(phase_vintage)
        if feature.shape[0] != target.shape[0] or target.shape[1] != PERIOD_STEPS:
            raise ValueError("Amp reference state/target mismatch")
        if feature.shape[0] < self.config.neighbors:
            raise ValueError("insufficient Amp KNN reference rows")
        scaler = StandardScaler()
        scaled = scaler.fit_transform(feature).astype(np.float32)
        components = min(
            self.config.pca_components,
            scaled.shape[0] - 1,
            scaled.shape[1],
        )
        reducer = PCA(
            n_components=components,
            whiten=True,
            svd_solver="randomized",
            random_state=self.config.seed,
        )
        reduced = reducer.fit_transform(scaled).astype(np.float32)
        nearest = NearestNeighbors(
            n_neighbors=self.config.neighbors,
            metric="euclidean",
            algorithm="brute",
            n_jobs=-1,
        ).fit(reduced)

        curves = period_memory_curves(period_memory)
        inner = fit_four_group_inner(curves, target, self.config.ridge_alpha)
        shapes = four_group_shapes(curves, inner)
        gram, cross, count = four_group_statistics(shapes, target)
        outer = solve_four_group_outer(gram, cross, self.config.ridge_alpha)
        return PeriodicKnnAmpState(
            scaler=scaler,
            reducer=reducer,
            nearest=nearest,
            target=target,
            inner=inner,
            gram=gram,
            cross=cross,
            count=count,
            outer=outer,
        )

    def predict_candidates(
        self,
        state: PeriodicKnnAmpState,
        phase_vintage: np.ndarray,
        period_memory: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        feature = phase_vintage_state_features(phase_vintage)
        reduced = state.reducer.transform(
            state.scaler.transform(feature).astype(np.float32)
        ).astype(np.float32)
        distance, index = state.nearest.kneighbors(reduced, return_distance=True)
        knn = state.target[index].mean(axis=1)
        curves = period_memory_curves(period_memory)
        four = predict_four_group(four_group_shapes(curves, state.inner), state.outer)
        return knn, four, {
            "reference_rows": int(state.target.shape[0]),
            "pca_components": int(state.reducer.n_components_),
            "neighbors": int(self.config.neighbors),
            "mean_nearest_distance": float(distance[:, 0].mean()),
            "mean_kth_distance": float(distance[:, -1].mean()),
        }

    def predict_candidates_panel(
        self,
        state: PeriodicKnnAmpState,
        phase_vintage: np.ndarray,
        period_memory: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Apply one state to every physical-period instance in a horizon."""

        phase_vintage = np.asarray(phase_vintage)
        period_memory = np.asarray(period_memory)
        if phase_vintage.ndim != 5 or period_memory.ndim != 5:
            raise ValueError("panel inputs must include origin and instance axes")
        if phase_vintage.shape[:2] != period_memory.shape[:2]:
            raise ValueError("panel vintage/memory axes do not align")
        origin_count, instance_count = phase_vintage.shape[:2]
        knn, four, summary = self.predict_candidates(
            state,
            phase_vintage.reshape(-1, *phase_vintage.shape[2:]),
            period_memory.reshape(-1, *period_memory.shape[2:]),
        )
        summary.update(
            {
                "origin_count": int(origin_count),
                "physical_period_instances": int(instance_count),
                "shared_state_across_instances": True,
            }
        )
        return (
            knn.reshape(origin_count, instance_count, PERIOD_STEPS),
            four.reshape(origin_count, instance_count, PERIOD_STEPS),
            summary,
        )

    def predict_candidates_online_panel(
        self,
        state: PeriodicKnnAmpState,
        phase_vintage: np.ndarray,
        period_memory: np.ndarray,
        origins: np.ndarray,
        matured_target: np.ndarray,
        *,
        native_period_steps: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Online four-group update shared across all physical-period blocks."""

        origins = np.asarray(origins, dtype=np.int64)
        target = np.asarray(matured_target, dtype=np.float64)
        knn, _static_four, summary = self.predict_candidates_panel(
            state, phase_vintage, period_memory
        )
        if target.shape != knn.shape or origins.shape != (target.shape[0],):
            raise ValueError("online panel target/origin shape mismatch")
        origin_count, instance_count = target.shape[:2]
        four, four_summary = self.predict_four_online_panel(
            state,
            period_memory,
            origins,
            target,
            native_period_steps=native_period_steps,
        )
        summary.update(four_summary)
        return knn, four, summary

    def predict_four_online_panel(
        self,
        state: PeriodicKnnAmpState,
        period_memory: np.ndarray,
        origins: np.ndarray,
        matured_target: np.ndarray,
        *,
        native_period_steps: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Run only the shared 28-period analytic branch after KNN is gated off."""

        period_memory = np.asarray(period_memory)
        origins = np.asarray(origins, dtype=np.int64)
        target = np.asarray(matured_target, dtype=np.float64)
        if period_memory.ndim != 5 or target.ndim != 3:
            raise ValueError("online four panel expects origin/instance axes")
        if period_memory.shape[:2] != target.shape[:2]:
            raise ValueError("online four panel axes do not align")
        if origins.shape != (target.shape[0],) or target.shape[2] != PERIOD_STEPS:
            raise ValueError("online four panel target/origin shape mismatch")
        origin_count, instance_count = target.shape[:2]
        curves = period_memory_curves(
            period_memory.reshape(-1, *period_memory.shape[2:])
        )
        shapes = four_group_shapes(curves, state.inner).reshape(
            origin_count, instance_count, MEMORY_GROUPS, PERIOD_STEPS
        )
        gram = state.gram.copy()
        cross = state.cross.copy()
        count = int(state.count)
        next_matured = np.zeros(instance_count, dtype=np.int64)
        update_events = 0
        if instance_count == 1:
            # Preserve the accepted ETTm1-v5 accumulation order bit for bit.
            four = np.empty_like(target, dtype=np.float64)
            for index, origin in enumerate(origins):
                delay = int(native_period_steps)
                mature_right = int(
                    np.searchsorted(origins, int(origin) - delay, side="right")
                )
                left = int(next_matured[0])
                if mature_right > left:
                    batch_gram, batch_cross, added = four_group_statistics(
                        shapes[left:mature_right, 0], target[left:mature_right, 0]
                    )
                    gram += batch_gram
                    cross += batch_cross
                    count += added
                    next_matured[0] = mature_right
                    update_events += 1
                outer = solve_four_group_outer(
                    gram, cross, self.config.ridge_alpha
                )
                four[index, 0] = predict_four_group(
                    shapes[index : index + 1, 0], outer
                )[0]
        else:
            # The same sufficient statistics can be accumulated from prefix
            # sums.  This removes millions of tiny Python calls on wide
            # datasets without changing the ridge equations or maturity rule.
            row_gram = np.einsum(
                "nsgp,nshp->nsgh", shapes, shapes, optimize=True
            )
            row_cross = np.einsum(
                "nsgp,nsp->nsg", shapes, target, optimize=True
            )
            gram_prefix = np.concatenate(
                [
                    np.zeros((1, instance_count, MEMORY_GROUPS, MEMORY_GROUPS)),
                    np.cumsum(row_gram, axis=0),
                ],
                axis=0,
            )
            cross_prefix = np.concatenate(
                [
                    np.zeros((1, instance_count, MEMORY_GROUPS)),
                    np.cumsum(row_cross, axis=0),
                ],
                axis=0,
            )
            outer_rows = np.empty(
                (origin_count, MEMORY_GROUPS), dtype=np.float64
            )
            for index, origin in enumerate(origins):
                for instance in range(instance_count):
                    delay = (instance + 1) * int(native_period_steps)
                    mature_right = int(
                        np.searchsorted(origins, int(origin) - delay, side="right")
                    )
                    left = int(next_matured[instance])
                    if mature_right > left:
                        gram += (
                            gram_prefix[mature_right, instance]
                            - gram_prefix[left, instance]
                        )
                        cross += (
                            cross_prefix[mature_right, instance]
                            - cross_prefix[left, instance]
                        )
                        count += mature_right - left
                        next_matured[instance] = mature_right
                        update_events += 1
                outer_rows[index] = solve_four_group_outer(
                    gram, cross, self.config.ridge_alpha
                )
            four = remove_affine(
                np.einsum(
                    "nsgp,ng->nsp", shapes, outer_rows, optimize=True
                ).reshape(-1, PERIOD_STEPS)
            ).reshape(origin_count, instance_count, PERIOD_STEPS)
        return four, {
            "online_four_period_updates": int(update_events),
            "online_four_period_added_origins_by_instance": next_matured.tolist(),
            "online_four_period_final_fit_count": int(count),
            "feedback_delay_steps_by_instance": [
                (instance + 1) * int(native_period_steps)
                for instance in range(instance_count)
            ],
        }

    def predict_candidates_online(
        self,
        state: PeriodicKnnAmpState,
        phase_vintage: np.ndarray,
        period_memory: np.ndarray,
        origins: np.ndarray,
        matured_target: np.ndarray,
        *,
        native_period_steps: int,
        forecast_period_index: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Predict with the v5 analytic branch updated by matured prior rows."""

        origins = np.asarray(origins, dtype=np.int64)
        target = np.asarray(matured_target, dtype=np.float64)
        knn, _static_four, summary = self.predict_candidates(
            state, phase_vintage, period_memory
        )
        curves = period_memory_curves(period_memory)
        shapes = four_group_shapes(curves, state.inner)
        if origins.shape != (shapes.shape[0],) or target.shape != (
            shapes.shape[0],
            PERIOD_STEPS,
        ):
            raise ValueError("online four-period target/origin shape mismatch")
        gram = state.gram.copy()
        cross = state.cross.copy()
        count = int(state.count)
        delay = (int(forecast_period_index) + 1) * int(native_period_steps)
        next_matured = 0
        four = np.empty((origins.size, PERIOD_STEPS), dtype=np.float64)
        update_events = 0
        for index, origin in enumerate(origins):
            mature_right = int(np.searchsorted(origins, int(origin) - delay, side="right"))
            if mature_right > next_matured:
                batch_gram, batch_cross, added = four_group_statistics(
                    shapes[next_matured:mature_right],
                    target[next_matured:mature_right],
                )
                gram += batch_gram
                cross += batch_cross
                count += added
                next_matured = mature_right
                update_events += 1
            outer = solve_four_group_outer(gram, cross, self.config.ridge_alpha)
            four[index] = predict_four_group(shapes[index : index + 1], outer)[0]
        summary.update(
            {
                "online_four_period_updates": update_events,
                "online_four_period_added_origins": next_matured,
                "online_four_period_final_fit_count": count,
            }
        )
        return knn, four, summary

    def calibrate_candidates(
        self,
        target: np.ndarray,
        knn: np.ndarray,
        four: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        target = remove_affine(np.asarray(target, dtype=np.float64))
        knn_scale = optimal_nonnegative_scale(target, knn)
        calibrated_knn = knn_scale * np.asarray(knn, dtype=np.float64)
        design = np.stack([calibrated_knn, four], axis=2).reshape(-1, 2)
        flat_target = target.reshape(-1)
        weight = box_least_squares_two(design.T @ design, design.T @ flat_target)
        return weight, {
            "knn_scale": float(knn_scale),
            "blend_weights": weight.tolist(),
        }

    @staticmethod
    def blend(knn: np.ndarray, four: np.ndarray, weight: np.ndarray) -> np.ndarray:
        weight = np.asarray(weight, dtype=np.float64)
        if weight.shape != (2,):
            raise ValueError("Amp blend requires two coefficients")
        return remove_affine(weight[0] * knn + weight[1] * four)


__all__ = [
    "MEMORY_GROUPS",
    "MEMORY_PERIODS",
    "PATCHES_PER_PERIOD",
    "PATCH_STEPS",
    "PERIOD_STEPS",
    "PeriodicKnnAmpState",
    "UniversalPeriodicKnnAmpAdapter",
    "UniversalPeriodicKnnAmpConfig",
    "box_least_squares_two",
    "blend_row_statistics",
    "causal_blend",
    "causal_blend_panel",
    "causal_panel_weights_from_row_statistics",
    "fit_four_group_inner",
    "four_group_shapes",
    "four_group_statistics",
    "optimal_nonnegative_scale",
    "period_memory_curves",
    "phase_vintage_state_features",
    "predict_four_group",
    "remove_affine",
    "solve_four_group_outer",
    "weights_from_statistics",
]
