from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import compute_channel_shape_features
from src.utils.clustering import cluster_channels_by_corr


def test_feature_aware_distance_can_split_corr_identical_channels() -> None:
    corr = torch.ones(4, 4)
    data = torch.randn(32, 4)
    extra = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
            [5.1, 5.0],
        ]
    )

    ids_plain, _ = cluster_channels_by_corr(
        corr_cc=corr,
        data_tc=data,
        n_clusters=None,
        distance_threshold=0.1,
        method="agglomerative",
        min_cluster_size=1,
        merge_small_clusters=False,
        extra_features_cf=None,
        feature_weight=0.0,
    )
    ids_feature, _ = cluster_channels_by_corr(
        corr_cc=corr,
        data_tc=data,
        n_clusters=None,
        distance_threshold=0.1,
        method="agglomerative",
        min_cluster_size=1,
        merge_small_clusters=False,
        extra_features_cf=extra,
        feature_weight=1.0,
    )

    assert int(ids_plain.max().item()) + 1 == 1
    assert int(ids_feature.max().item()) + 1 == 2
    assert ids_feature[0].item() == ids_feature[1].item()
    assert ids_feature[2].item() == ids_feature[3].item()
    assert ids_feature[0].item() != ids_feature[2].item()


def test_channel_shape_features_are_per_channel_and_finite() -> None:
    t = torch.linspace(0.0, 1.0, steps=64)
    data = torch.stack(
        [
            t,
            torch.sin(2.0 * torch.pi * t),
            torch.sign(torch.sin(8.0 * torch.pi * t)),
        ],
        dim=1,
    )
    feat = compute_channel_shape_features(data, acf_lags=[1, 4])

    assert feat.shape == (3, 12)
    assert torch.isfinite(feat).all()
    assert not torch.allclose(feat[0], feat[2])


def test_singleton_merge_strategy_can_keep_unrelated_singletons() -> None:
    corr = torch.tensor(
        [
            [1.0, 0.95, 0.05, -0.10],
            [0.95, 1.0, 0.02, -0.15],
            [0.05, 0.02, 1.0, -0.30],
            [-0.10, -0.15, -0.30, 1.0],
        ]
    )
    data = torch.randn(64, 4)

    ids_pool, clusters_pool = cluster_channels_by_corr(
        corr_cc=corr,
        data_tc=data,
        n_clusters=None,
        distance_threshold=0.2,
        method="agglomerative",
        min_cluster_size=2,
        merge_small_clusters=True,
        no_merge_if_channels_lt=1,
        singleton_merge_strategy="pool",
    )
    ids_keep, clusters_keep = cluster_channels_by_corr(
        corr_cc=corr,
        data_tc=data,
        n_clusters=None,
        distance_threshold=0.2,
        method="agglomerative",
        min_cluster_size=2,
        merge_small_clusters=True,
        no_merge_if_channels_lt=1,
        singleton_merge_strategy="keep",
    )

    assert len(clusters_pool) == 2
    assert sorted(len(v) for v in clusters_pool.values()) == [2, 2]
    assert ids_pool[2].item() == ids_pool[3].item()
    assert len(clusters_keep) == 3
    assert sorted(len(v) for v in clusters_keep.values()) == [1, 1, 2]
    assert ids_keep[2].item() != ids_keep[3].item()


def test_guarded_singleton_merge_only_pools_nearby_singletons() -> None:
    corr = torch.tensor(
        [
            [1.0, 0.95, 0.05, -0.10, -0.20],
            [0.95, 1.0, 0.02, -0.15, -0.18],
            [0.05, 0.02, 1.0, 0.82, -0.30],
            [-0.10, -0.15, 0.82, 1.0, -0.35],
            [-0.20, -0.18, -0.30, -0.35, 1.0],
        ]
    )
    data = torch.randn(64, 5)

    ids_guarded, clusters_guarded = cluster_channels_by_corr(
        corr_cc=corr,
        data_tc=data,
        n_clusters=None,
        distance_threshold=0.1,
        method="agglomerative",
        min_cluster_size=2,
        merge_small_clusters=True,
        no_merge_if_channels_lt=1,
        singleton_merge_strategy="guarded_pool",
        singleton_merge_distance_threshold=0.25,
    )

    assert sorted(len(v) for v in clusters_guarded.values()) == [1, 2, 2]
    assert ids_guarded[2].item() == ids_guarded[3].item()
    assert ids_guarded[4].item() != ids_guarded[2].item()
    assert ids_guarded[4].item() != ids_guarded[0].item()


if __name__ == "__main__":
    test_feature_aware_distance_can_split_corr_identical_channels()
    test_channel_shape_features_are_per_channel_and_finite()
    test_singleton_merge_strategy_can_keep_unrelated_singletons()
    test_guarded_singleton_merge_only_pools_nearby_singletons()
