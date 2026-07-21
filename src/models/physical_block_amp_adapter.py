"""Legacy P96 compatibility shim for the universal periodic KNN Amp.

The dataset/horizon-independent kernel now lives in
``periodic_knn_amp_adapter.py``.  This module retains the historical P96 names
and physical-stream helpers needed to replay frozen ETTm1 artifacts.

Every block removes its own affine level/trend component.  Later blocks use
older daily residual vintages and longer feedback delays so that no residual
whose target endpoint is still in the future can enter the action.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.models.periodic_knn_amp_adapter import (
    blend_row_statistics as universal_blend_row_statistics,
    box_least_squares_two as universal_box_least_squares_two,
    fit_four_group_inner as universal_fit_four_group_inner,
    four_group_shapes as universal_four_group_shapes,
    four_group_statistics as universal_four_group_statistics,
    phase_vintage_state_features,
    predict_four_group as universal_predict_four_group,
    remove_affine as universal_remove_affine,
    solve_four_group_outer as universal_solve_four_group_outer,
    weights_from_statistics as universal_weights_from_statistics,
)


BLOCK_STEPS = 96
PATCH_STEPS = 12
PATCHES_PER_BLOCK = BLOCK_STEPS // PATCH_STEPS
HISTORY_PATCHES = 8
DAYS_PER_WEEK = 7
WEEK_COUNT = 4
DAY_COUNT = DAYS_PER_WEEK * WEEK_COUNT
ROBUST_CURVE_TRIM_FRACTION = 0.20


@dataclass(frozen=True)
class PhysicalBlockAmpConfig:
    """Hyperparameters shared by the same Amp algorithm across horizons."""

    block_steps: int = BLOCK_STEPS
    patch_steps: int = PATCH_STEPS
    history_patches: int = HISTORY_PATCHES
    pca_components: int = 32
    neighbors: int = 64
    reference_origins: int = 28 * BLOCK_STEPS
    calibration_origins: int = 7 * BLOCK_STEPS
    ridge_alpha: float = 10.0
    minimum_matured_rows: int = 256
    seed: int = 2026

    def __post_init__(self) -> None:
        if self.block_steps != BLOCK_STEPS:
            raise ValueError("Amp physical block must remain 96 steps")
        if self.patch_steps != PATCH_STEPS:
            raise ValueError("Amp patch interface must remain 12 steps")
        if self.history_patches != HISTORY_PATCHES:
            raise ValueError("Amp aligned history must remain eight p12 patches")
        if self.neighbors <= 0 or self.pca_components <= 0:
            raise ValueError("KNN dimensions must be positive")


def block_count(horizon: int) -> int:
    """Number of complete/partial physical H96 blocks in a horizon."""

    horizon = int(horizon)
    if horizon <= 0 or horizon % PATCH_STEPS:
        raise ValueError("Amp horizon must be a positive multiple of p12")
    return (horizon + BLOCK_STEPS - 1) // BLOCK_STEPS


def block_slice(horizon: int, block_index: int) -> slice:
    """Step-coordinate slice for one physical block."""

    count = block_count(horizon)
    block_index = int(block_index)
    if block_index < 0 or block_index >= count:
        raise IndexError("Amp physical block index out of range")
    left = block_index * BLOCK_STEPS
    return slice(left, min(left + BLOCK_STEPS, int(horizon)))


def patch_block_slice(horizon: int, block_index: int) -> slice:
    steps = block_slice(horizon, block_index)
    return slice(steps.start // PATCH_STEPS, steps.stop // PATCH_STEPS)


def remove_affine(values: np.ndarray) -> np.ndarray:
    """Remove a row-wise constant and linear coordinate."""

    return universal_remove_affine(values)


def remove_affine_by_block(values: np.ndarray) -> np.ndarray:
    """Remove affine content independently inside fixed physical H96 blocks."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("block affine removal expects [row, horizon]")
    result = np.empty_like(values, dtype=np.float64)
    for index in range(block_count(values.shape[1])):
        current = block_slice(values.shape[1], index)
        result[:, current] = remove_affine(values[:, current])
    return result


def block_state_features(carrier: np.ndarray) -> np.ndarray:
    """Exact native H96 residual-state features for one complete block.

    ``carrier`` is ``[row, 8 future patches, 8 aligned histories, 12]``.
    """

    return phase_vintage_state_features(carrier)


def fit_predict_knn(
    fit_carrier: np.ndarray,
    fit_target: np.ndarray,
    query_carrier: np.ndarray,
    config: PhysicalBlockAmpConfig,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Fit one channel/block native KNN and predict complete H96 actions."""

    fit_target = np.asarray(fit_target, dtype=np.float64)
    if fit_target.ndim != 2 or fit_target.shape[1] != BLOCK_STEPS:
        raise ValueError("native Amp KNN target must be a complete H96 block")
    fit_feature = block_state_features(fit_carrier)
    query_feature = block_state_features(query_carrier)
    if fit_feature.shape[0] != fit_target.shape[0]:
        raise ValueError("Amp KNN carrier/target row mismatch")
    if fit_feature.shape[0] < config.neighbors:
        raise ValueError("insufficient Amp KNN reference rows")
    scaler = StandardScaler()
    fit_scaled = scaler.fit_transform(fit_feature).astype(np.float32)
    query_scaled = scaler.transform(query_feature).astype(np.float32)
    components = min(
        config.pca_components,
        fit_scaled.shape[0] - 1,
        fit_scaled.shape[1],
    )
    reducer = PCA(
        n_components=components,
        whiten=True,
        svd_solver="randomized",
        random_state=config.seed,
    )
    fit_state = reducer.fit_transform(fit_scaled).astype(np.float32)
    query_state = reducer.transform(query_scaled).astype(np.float32)
    nearest = NearestNeighbors(
        n_neighbors=config.neighbors,
        metric="euclidean",
        algorithm="brute",
        n_jobs=-1,
    ).fit(fit_state)
    distance, index = nearest.kneighbors(query_state, return_distance=True)
    prediction = fit_target[index].mean(axis=1)
    return prediction, {
        "fit_rows": int(fit_target.shape[0]),
        "pca_components": int(components),
        "pca_explained_variance_fraction": float(
            reducer.explained_variance_ratio_.sum()
        ),
        "mean_nearest_distance": float(distance[:, 0].mean()),
        "mean_kth_distance": float(distance[:, -1].mean()),
    }


def robust_curve_trimmed_mean(
    neighbor_target: np.ndarray,
    trim_fraction: float = ROBUST_CURVE_TRIM_FRACTION,
) -> np.ndarray:
    """Average the central whole-curve neighbors, preserving Amp isolation.

    A coordinate-wise trimmed mean can mix a different set of neighbors at
    every forecast step and thereby reintroduce affine Level/Trend content.
    Here each complete H96 residual curve is either retained or discarded as a
    unit.  Curves are ranked by squared distance to the coordinate-wise median
    curve and the most distant fixed fraction is removed before averaging.
    Since the retained inputs are complete Amp residual curves, their average
    remains in the same affine-free subspace.
    """

    neighbor_target = np.asarray(neighbor_target, dtype=np.float64)
    if neighbor_target.ndim != 3 or neighbor_target.shape[2] != BLOCK_STEPS:
        raise ValueError("robust Amp aggregation expects [query, neighbor, 96]")
    trim_fraction = float(trim_fraction)
    if not 0.0 <= trim_fraction < 1.0:
        raise ValueError("robust Amp trim fraction must be in [0, 1)")
    neighbor_count = int(neighbor_target.shape[1])
    remove_count = int(np.floor(neighbor_count * trim_fraction))
    keep_count = neighbor_count - remove_count
    if keep_count <= 0:
        raise ValueError("robust Amp aggregation must retain a neighbor")
    center = np.median(neighbor_target, axis=1)
    distance = np.square(neighbor_target - center[:, None]).mean(axis=2)
    keep = np.argpartition(distance, keep_count - 1, axis=1)[:, :keep_count]
    retained = np.take_along_axis(
        neighbor_target,
        keep[:, :, None],
        axis=1,
    )
    return retained.mean(axis=1)


def fit_predict_knn_robust(
    fit_carrier: np.ndarray,
    fit_target: np.ndarray,
    query_carrier: np.ndarray,
    config: PhysicalBlockAmpConfig,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Native KNN geometry with one predeclared 20% whole-curve trim."""

    fit_target = np.asarray(fit_target, dtype=np.float64)
    if fit_target.ndim != 2 or fit_target.shape[1] != BLOCK_STEPS:
        raise ValueError("native Amp KNN target must be a complete H96 block")
    fit_feature = block_state_features(fit_carrier)
    query_feature = block_state_features(query_carrier)
    if fit_feature.shape[0] != fit_target.shape[0]:
        raise ValueError("Amp KNN carrier/target row mismatch")
    if fit_feature.shape[0] < config.neighbors:
        raise ValueError("insufficient Amp KNN reference rows")
    scaler = StandardScaler()
    fit_scaled = scaler.fit_transform(fit_feature).astype(np.float32)
    query_scaled = scaler.transform(query_feature).astype(np.float32)
    components = min(
        config.pca_components,
        fit_scaled.shape[0] - 1,
        fit_scaled.shape[1],
    )
    reducer = PCA(
        n_components=components,
        whiten=True,
        svd_solver="randomized",
        random_state=config.seed,
    )
    fit_state = reducer.fit_transform(fit_scaled).astype(np.float32)
    query_state = reducer.transform(query_scaled).astype(np.float32)
    nearest = NearestNeighbors(
        n_neighbors=config.neighbors,
        metric="euclidean",
        algorithm="brute",
        n_jobs=-1,
    ).fit(fit_state)
    distance, index = nearest.kneighbors(query_state, return_distance=True)
    prediction = robust_curve_trimmed_mean(fit_target[index])
    return prediction, {
        "fit_rows": int(fit_target.shape[0]),
        "pca_components": int(components),
        "pca_explained_variance_fraction": float(
            reducer.explained_variance_ratio_.sum()
        ),
        "mean_nearest_distance": float(distance[:, 0].mean()),
        "mean_kth_distance": float(distance[:, -1].mean()),
        "aggregation": "whole_curve_distance_trimmed_mean",
        "trim_fraction": float(ROBUST_CURVE_TRIM_FRACTION),
        "retained_neighbors": int(
            config.neighbors
            - np.floor(config.neighbors * ROBUST_CURVE_TRIM_FRACTION)
        ),
    }


def empirical_bayes_curve_mean(
    neighbor_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Positive-part finite-sample reliability for a whole Amp mean curve.

    The observed squared norm of a K-neighbor sample mean contains both the
    latent conditional-mean energy and approximately ``noise_energy / K``.
    Subtract that finite-sample term, clamp the signal estimate at zero, and
    use its fraction of the observed mean energy as one scalar reliability for
    the complete curve.  Scalar shrinkage preserves the affine-free Amp space.
    """

    neighbor_target = np.asarray(neighbor_target, dtype=np.float64)
    if neighbor_target.ndim != 3 or neighbor_target.shape[2] != BLOCK_STEPS:
        raise ValueError("empirical-Bayes Amp aggregation expects [query, neighbor, 96]")
    neighbor_count = int(neighbor_target.shape[1])
    if neighbor_count <= 0:
        raise ValueError("empirical-Bayes Amp aggregation requires neighbors")
    mean = neighbor_target.mean(axis=1)
    mean_energy = np.square(mean).mean(axis=1)
    noise_energy = np.square(neighbor_target - mean[:, None]).mean(axis=(1, 2))
    finite_sample_variance = noise_energy / float(neighbor_count)
    reliability = np.zeros_like(mean_energy)
    nonzero = mean_energy > 1.0e-15
    reliability[nonzero] = np.clip(
        1.0 - finite_sample_variance[nonzero] / mean_energy[nonzero],
        0.0,
        1.0,
    )
    return mean * reliability[:, None], reliability


def fit_predict_knn_empirical_bayes(
    fit_carrier: np.ndarray,
    fit_target: np.ndarray,
    query_carrier: np.ndarray,
    config: PhysicalBlockAmpConfig,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Native KNN geometry with parameter-free whole-curve reliability."""

    fit_target = np.asarray(fit_target, dtype=np.float64)
    if fit_target.ndim != 2 or fit_target.shape[1] != BLOCK_STEPS:
        raise ValueError("native Amp KNN target must be a complete H96 block")
    fit_feature = block_state_features(fit_carrier)
    query_feature = block_state_features(query_carrier)
    if fit_feature.shape[0] != fit_target.shape[0]:
        raise ValueError("Amp KNN carrier/target row mismatch")
    if fit_feature.shape[0] < config.neighbors:
        raise ValueError("insufficient Amp KNN reference rows")
    scaler = StandardScaler()
    fit_scaled = scaler.fit_transform(fit_feature).astype(np.float32)
    query_scaled = scaler.transform(query_feature).astype(np.float32)
    components = min(
        config.pca_components,
        fit_scaled.shape[0] - 1,
        fit_scaled.shape[1],
    )
    reducer = PCA(
        n_components=components,
        whiten=True,
        svd_solver="randomized",
        random_state=config.seed,
    )
    fit_state = reducer.fit_transform(fit_scaled).astype(np.float32)
    query_state = reducer.transform(query_scaled).astype(np.float32)
    nearest = NearestNeighbors(
        n_neighbors=config.neighbors,
        metric="euclidean",
        algorithm="brute",
        n_jobs=-1,
    ).fit(fit_state)
    distance, index = nearest.kneighbors(query_state, return_distance=True)
    prediction, reliability = empirical_bayes_curve_mean(fit_target[index])
    return prediction, {
        "fit_rows": int(fit_target.shape[0]),
        "pca_components": int(components),
        "pca_explained_variance_fraction": float(
            reducer.explained_variance_ratio_.sum()
        ),
        "mean_nearest_distance": float(distance[:, 0].mean()),
        "mean_kth_distance": float(distance[:, -1].mean()),
        "aggregation": "positive_part_empirical_bayes_curve_mean",
        "mean_reliability": float(reliability.mean()),
        "minimum_reliability": float(reliability.min()),
        "maximum_reliability": float(reliability.max()),
    }


def fit_predict_knn_with_support(
    fit_carrier: np.ndarray,
    fit_target: np.ndarray,
    query_carrier: np.ndarray,
    config: PhysicalBlockAmpConfig,
) -> tuple[np.ndarray, dict[str, float | int], dict[str, np.ndarray]]:
    """Identical native KNN prediction plus target-free per-query support.

    Neighbor targets are strictly historical/matured fit rows.  Their spread
    and the query-to-reference distances are therefore available at forecast
    time.  This diagnostic API does not change the fitted KNN or prediction.
    """

    fit_target = np.asarray(fit_target, dtype=np.float64)
    if fit_target.ndim != 2 or fit_target.shape[1] != BLOCK_STEPS:
        raise ValueError("native Amp KNN target must be a complete H96 block")
    fit_feature = block_state_features(fit_carrier)
    query_feature = block_state_features(query_carrier)
    if fit_feature.shape[0] != fit_target.shape[0]:
        raise ValueError("Amp KNN carrier/target row mismatch")
    if fit_feature.shape[0] < config.neighbors:
        raise ValueError("insufficient Amp KNN reference rows")
    scaler = StandardScaler()
    fit_scaled = scaler.fit_transform(fit_feature).astype(np.float32)
    query_scaled = scaler.transform(query_feature).astype(np.float32)
    components = min(config.pca_components, fit_scaled.shape[0] - 1, fit_scaled.shape[1])
    reducer = PCA(
        n_components=components,
        whiten=True,
        svd_solver="randomized",
        random_state=config.seed,
    )
    fit_state = reducer.fit_transform(fit_scaled).astype(np.float32)
    query_state = reducer.transform(query_scaled).astype(np.float32)
    nearest = NearestNeighbors(
        n_neighbors=config.neighbors,
        metric="euclidean",
        algorithm="brute",
        n_jobs=-1,
    ).fit(fit_state)
    distance, index = nearest.kneighbors(query_state, return_distance=True)
    neighbor_target = fit_target[index]
    prediction = neighbor_target.mean(axis=1)
    target_spread = np.sqrt(np.square(neighbor_target - prediction[:, None]).mean(axis=(1, 2)))
    summary = {
        "fit_rows": int(fit_target.shape[0]),
        "pca_components": int(components),
        "pca_explained_variance_fraction": float(reducer.explained_variance_ratio_.sum()),
        "mean_nearest_distance": float(distance[:, 0].mean()),
        "mean_kth_distance": float(distance[:, -1].mean()),
    }
    support = {
        "nearest_distance": distance[:, 0].astype(np.float32),
        "mean_distance": distance.mean(axis=1).astype(np.float32),
        "kth_distance": distance[:, -1].astype(np.float32),
        "neighbor_target_spread": target_spread.astype(np.float32),
    }
    return prediction, summary, support


def daily_lag_days(block_index: int) -> np.ndarray:
    """Twenty-eight fully observed daily vintages for a physical block.

    Block 0 is exactly the native H96 lags 1..28.  Block ``b`` starts at
    ``b+1`` days because an H96 block at that lead is not observed sooner.
    """

    block_index = int(block_index)
    if block_index < 0:
        raise ValueError("Amp block index cannot be negative")
    return np.arange(block_index + 1, block_index + 1 + DAY_COUNT, dtype=np.int64)


def feedback_delay_steps(block_index: int) -> int:
    """First time at which a block target is completely mature."""

    block_index = int(block_index)
    if block_index < 0:
        raise ValueError("Amp block index cannot be negative")
    return (block_index + 1) * BLOCK_STEPS


def daily_block_carriers(
    stream: np.ndarray,
    stream_origin0: int,
    origins: np.ndarray,
    channel: int,
    block_index: int,
) -> np.ndarray:
    """Load causal daily residual vintages for one complete physical block."""

    stream = np.asarray(stream)
    origins = np.asarray(origins, dtype=np.int64)
    patch_slice = patch_block_slice(stream.shape[2] * PATCH_STEPS, block_index)
    if patch_slice.stop - patch_slice.start != PATCHES_PER_BLOCK:
        raise ValueError("four-week Amp branch requires a complete H96 block")
    offsets = BLOCK_STEPS * daily_lag_days(block_index)
    index = origins[:, None] - offsets[None] - int(stream_origin0)
    if int(index.min()) < 0 or int(index.max()) >= stream.shape[0]:
        raise IndexError("four-week Amp carrier source is outside the residual stream")
    raw = stream[index, int(channel), patch_slice, :].reshape(
        origins.size * DAY_COUNT, BLOCK_STEPS
    )
    return remove_affine(raw.astype(np.float64)).reshape(
        origins.size, DAY_COUNT, BLOCK_STEPS
    )


def block_targets(
    stream: np.ndarray,
    stream_origin0: int,
    origins: np.ndarray,
    channel: int,
    block_index: int,
) -> np.ndarray:
    stream = np.asarray(stream)
    origins = np.asarray(origins, dtype=np.int64)
    patch_slice = patch_block_slice(stream.shape[2] * PATCH_STEPS, block_index)
    if patch_slice.stop - patch_slice.start != PATCHES_PER_BLOCK:
        raise ValueError("native Amp target requires a complete H96 block")
    index = origins - int(stream_origin0)
    if int(index.min()) < 0 or int(index.max()) >= stream.shape[0]:
        raise IndexError("Amp target is outside the residual stream")
    raw = stream[index, int(channel), patch_slice, :].reshape(-1, BLOCK_STEPS)
    return remove_affine(raw.astype(np.float64))


def fit_four_week_inner(
    carriers: np.ndarray,
    target: np.ndarray,
    ridge_alpha: float,
) -> np.ndarray:
    return universal_fit_four_group_inner(carriers, target, ridge_alpha)


def four_week_shapes(carriers: np.ndarray, inner: np.ndarray) -> np.ndarray:
    return universal_four_group_shapes(carriers, inner)


def four_week_statistics(
    shapes: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    return universal_four_group_statistics(shapes, target)


def solve_four_week_outer(
    gram: np.ndarray, cross: np.ndarray, ridge_alpha: float
) -> np.ndarray:
    return universal_solve_four_group_outer(gram, cross, ridge_alpha)


def predict_four_week(shapes: np.ndarray, outer: np.ndarray) -> np.ndarray:
    return universal_predict_four_group(shapes, outer)


def box_least_squares_two(
    gram: np.ndarray,
    cross: np.ndarray,
    lower: float = 0.0,
    upper: float = 1.0,
) -> np.ndarray:
    """Exact two-action least squares under a box constraint."""

    return universal_box_least_squares_two(gram, cross, lower, upper)


def blend_row_statistics(
    residual: np.ndarray, knn: np.ndarray, four_week: np.ndarray
) -> np.ndarray:
    """Sufficient statistics for bounded KNN/four-week blending."""

    return universal_blend_row_statistics(residual, knn, four_week)


def weights_from_statistics(
    statistics: np.ndarray,
    minimum_rows: int,
    initial: tuple[float, float] = (1.0, 0.0),
) -> np.ndarray:
    return universal_weights_from_statistics(statistics, minimum_rows, initial)
