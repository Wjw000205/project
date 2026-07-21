"""Isolated KNN estimator for the actual DiffAmp operator residual.

This is deliberately not an alias of :class:`UniversalPeriodicKnnAmpAdapter`.
It reuses the successful retrieval inductive bias, while changing both sides
of the contract to DiffAmp semantics:

* the query is built only from first differences of fully matured forecast
  residual carriers; and
* the retrieved target is the signed error
  ``std(D1(target)) - std(D1(backbone))``.

The scalar estimate is decoded by ``decode_diff_amp_operator_residual`` into
the current target-free DiffAmp basis.  Thus the deployed action is a forecast
residual in the named space, never a complete forecast and never an Amp wave.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.models.periodic_knn_amp_adapter import phase_vintage_state_features


@dataclass(frozen=True)
class PeriodicKnnDiffAmpConfig:
    pca_components: int = 32
    neighbors: int = 64
    reference_periods: int = 28
    seed: int = 2026

    def __post_init__(self) -> None:
        if self.pca_components <= 0 or self.neighbors <= 0:
            raise ValueError("DiffAmp KNN dimensions must be positive")
        if self.reference_periods != 28:
            raise ValueError("DiffAmp KNN must retain 28 physical reference periods")


@dataclass
class PeriodicKnnDiffAmpState:
    scaler: StandardScaler
    reducer: PCA
    nearest: NearestNeighbors
    operator_residual: np.ndarray


def diff_amp_phase_vintage_features(carrier: np.ndarray) -> np.ndarray:
    """D1-only features from eight fully matured phase-vintage curves."""

    carrier = np.asarray(carrier, dtype=np.float64)
    if carrier.ndim != 4 or tuple(carrier.shape[1:]) != (8, 8, 12):
        raise ValueError("DiffAmp phase vintage must have shape [N,8,8,12]")
    curves = carrier.reshape(carrier.shape[0], 8, 96)
    difference = np.diff(curves, axis=2, prepend=curves[:, :, :1])
    return phase_vintage_state_features(difference.reshape(-1, 8, 8, 12))


class PeriodicKnnDiffAmpResidualAdapter:
    """One fixed D1-state KNN that predicts a signed DiffAmp error scalar."""

    def __init__(self, config: PeriodicKnnDiffAmpConfig | None = None) -> None:
        self.config = config or PeriodicKnnDiffAmpConfig()

    def fit(
        self,
        phase_vintage: np.ndarray,
        operator_residual: np.ndarray,
    ) -> PeriodicKnnDiffAmpState:
        feature = diff_amp_phase_vintage_features(phase_vintage)
        target = np.asarray(operator_residual, dtype=np.float64)
        if target.shape != (feature.shape[0],):
            raise ValueError("DiffAmp KNN target must have shape [N]")
        if feature.shape[0] < self.config.neighbors:
            raise ValueError("insufficient DiffAmp KNN reference rows")
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
        return PeriodicKnnDiffAmpState(
            scaler=scaler,
            reducer=reducer,
            nearest=nearest,
            operator_residual=target.copy(),
        )

    def predict_operator_residual(
        self,
        state: PeriodicKnnDiffAmpState,
        phase_vintage: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        feature = diff_amp_phase_vintage_features(phase_vintage)
        reduced = state.reducer.transform(
            state.scaler.transform(feature).astype(np.float32)
        ).astype(np.float32)
        distance, index = state.nearest.kneighbors(reduced, return_distance=True)
        prediction = state.operator_residual[index].mean(axis=1)
        return prediction, {
            "reference_rows": int(state.operator_residual.shape[0]),
            "pca_components": int(state.reducer.n_components_),
            "neighbors": int(self.config.neighbors),
            "mean_nearest_distance": float(distance[:, 0].mean()),
            "mean_kth_distance": float(distance[:, -1].mean()),
        }


__all__ = [
    "PeriodicKnnDiffAmpConfig",
    "PeriodicKnnDiffAmpResidualAdapter",
    "PeriodicKnnDiffAmpState",
    "diff_amp_phase_vintage_features",
]
