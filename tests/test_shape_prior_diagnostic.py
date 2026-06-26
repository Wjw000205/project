import inspect
import subprocess
import sys

import torch

from scripts import shape_prior_diagnostic as spd


def test_shape_prior_diagnostic_script_help_imports_from_repo_root() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/shape_prior_diagnostic.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--checkpoint" in result.stdout


def test_shape_features_do_not_accept_y_true_and_are_target_free() -> None:
    sig = inspect.signature(spd.compute_shape_features)
    assert "y_true" not in sig.parameters
    assert "y" not in sig.parameters

    x = torch.tensor(
        [
            [
                [0.0, 1.0, 2.0, 4.0],
                [1.0, 1.0, 1.0, 1.0],
            ]
        ]
    )
    y_base = torch.tensor(
        [
            [
                [4.0, 5.0, 6.0, 7.0],
                [1.0, 1.5, 1.0, 0.5],
            ]
        ]
    )
    cluster_id = torch.tensor([0, 1])

    features_a, names_a = spd.compute_shape_features(x, y_base, cluster_id, K=2)
    features_b, names_b = spd.compute_shape_features(x, y_base, cluster_id, K=2)

    assert names_a == names_b
    assert features_a.shape == (1, 2, len(names_a))
    torch.testing.assert_close(features_a, features_b)
    assert torch.isfinite(features_a).all()


def test_allowed_mask_respects_filtered_branch_policy() -> None:
    penalty_names = ["jump", "amp_under", "level", "delta"]
    mask = spd.build_allowed_mask(
        penalty_names=penalty_names,
        K=2,
        allowed_by_cluster={0: ["jump", "delta"], 1: ["amp_under", "delta", "jump"]},
    )

    expected = torch.tensor(
        [
            [True, False, False, True],
            [True, True, False, True],
        ]
    )
    assert torch.equal(mask.cpu(), expected)


def test_shape_bucket_edges_are_fit_from_train_features_only() -> None:
    train_fit = torch.tensor(
        [
            [[0.0, 10.0]],
            [[1.0, 11.0]],
            [[2.0, 12.0]],
            [[3.0, 13.0]],
        ]
    )
    train_holdout = torch.tensor(
        [
            [[100.0, -100.0]],
            [[101.0, -101.0]],
        ]
    )

    edges = spd.fit_quantile_bucket_edges(train_fit, feature_names=["a", "b"], q=4)
    buckets = spd.apply_quantile_bucket_edges(train_holdout, edges)

    assert edges["feature_names"] == ["a", "b"]
    assert edges["q"] == 4
    assert edges["edges"][0][0] == [0.75, 1.5, 2.25]
    assert edges["edges"][0][1] == [10.75, 11.5, 12.25]
    assert buckets.shape == (2, 1, 2)
    assert buckets[:, 0, 0].tolist() == [3, 3]
    assert buckets[:, 0, 1].tolist() == [0, 0]


def test_bucket_gain_stats_masks_disallowed_penalties() -> None:
    bucket_ids = torch.tensor(
        [
            [[0]],
            [[0]],
            [[1]],
            [[1]],
        ]
    )
    gains = torch.tensor(
        [
            [[[0.1, 0.9, 5.0]]],
            [[[0.2, 0.8, 6.0]]],
            [[[-0.1, 0.7, 7.0]]],
            [[[0.0, 0.6, 8.0]]],
        ]
    )
    allowed = torch.tensor([[True, False, False]])

    stats = spd.compute_bucket_gain_stats(
        bucket_ids=bucket_ids,
        gains_bkp=gains.squeeze(2),
        allowed_mask_kp=allowed,
        feature_names=["shape"],
        penalty_names=["jump", "amp_under", "level"],
        split_name="train_fit",
        q=2,
    )

    penalties = {row["penalty"] for row in stats}
    assert penalties == {"jump"}
    bucket0 = next(row for row in stats if row["bucket"] == 0)
    assert bucket0["support_count"] == 2
    assert bucket0["mean_gain"] == 0.15000000596046448
    assert bucket0["positive_rate"] == 1.0
