import pytest
import torch

from scripts.next11d_candidate_utility_stability import (
    _candidate_channel_stability_rows,
    _candidate_gain_by_channel_penalty,
    _candidate_gain_by_cluster_penalty,
    _candidate_stability_rows,
    _normalize_requested_splits,
    _split_channel_candidate_stats,
    _split_candidate_stats,
    _static_channel_guard_metrics,
)


def test_candidate_gain_by_cluster_penalty_averages_channels_per_cluster() -> None:
    y = torch.zeros(1, 3, 2)
    base = torch.ones(1, 3, 2)
    cand = torch.stack(
        [
            torch.zeros_like(base),
            torch.full_like(base, 2.0),
        ],
        dim=2,
    )
    cluster_id = torch.tensor([0, 0, 1], dtype=torch.long)

    gain = _candidate_gain_by_cluster_penalty(
        base_bch=base,
        cand_bcpH=cand,
        y_bch=y,
        cluster_id_c=cluster_id,
        cluster_count=2,
    )

    assert gain.shape == (1, 2, 2)
    assert gain[0, 0, 0].item() == pytest.approx(1.0)
    assert gain[0, 0, 1].item() == pytest.approx(-3.0)
    assert gain[0, 1, 0].item() == pytest.approx(1.0)


def test_candidate_gain_by_channel_penalty_preserves_channel_dimension() -> None:
    y = torch.zeros(1, 2, 2)
    base = torch.ones(1, 2, 2)
    cand = torch.stack([torch.zeros_like(base), torch.full_like(base, 2.0)], dim=2)

    gain = _candidate_gain_by_channel_penalty(base_bch=base, cand_bcpH=cand, y_bch=y)

    assert gain.shape == (1, 2, 2)
    assert gain[0, 0, 0].item() == pytest.approx(1.0)
    assert gain[0, 0, 1].item() == pytest.approx(-3.0)
    assert gain[0, 1, 0].item() == pytest.approx(1.0)


def test_candidate_stability_rows_require_train_fit_and_holdout_positive_utility() -> None:
    penalty_names = ["trend", "direction"]
    fit_stats = _split_candidate_stats(
        gain_bkp=torch.tensor(
            [
                [[0.2, -0.1]],
                [[0.3, 0.2]],
                [[0.4, -0.2]],
            ],
            dtype=torch.float32,
        ),
        split="train_fit",
        penalty_names=penalty_names,
    )
    holdout_stats = _split_candidate_stats(
        gain_bkp=torch.tensor(
            [
                [[0.1, -0.1]],
                [[0.2, -0.2]],
                [[0.3, 0.3]],
            ],
            dtype=torch.float32,
        ),
        split="train_holdout",
        penalty_names=penalty_names,
    )
    val_stats = _split_candidate_stats(
        gain_bkp=torch.tensor(
            [
                [[-0.1, -0.1]],
                [[0.2, -0.2]],
                [[0.3, -0.3]],
            ],
            dtype=torch.float32,
        ),
        split="val",
        penalty_names=penalty_names,
    )

    rows = _candidate_stability_rows(
        fit_stats=fit_stats,
        holdout_stats=holdout_stats,
        val_stats=val_stats,
        min_support=3,
        margin=0.0,
        positive_rate_threshold=0.60,
    )

    trend = rows[0]
    direction = rows[1]
    assert trend["penalty"] == "trend"
    assert trend["stable_train_splits"] is True
    assert trend["val_mean_gain"] == pytest.approx(0.13333333)
    assert direction["stable_train_splits"] is False


def test_channel_candidate_stability_rows_can_find_channel_specific_signal() -> None:
    penalty_names = ["trend"]
    fit_stats = _split_channel_candidate_stats(
        gain_bcp=torch.tensor(
            [
                [[0.3], [-0.3]],
                [[0.2], [-0.2]],
                [[0.1], [0.4]],
            ],
            dtype=torch.float32,
        ),
        split="train_fit",
        penalty_names=penalty_names,
    )
    holdout_stats = _split_channel_candidate_stats(
        gain_bcp=torch.tensor(
            [
                [[0.1], [-0.1]],
                [[0.2], [-0.2]],
                [[0.3], [0.5]],
            ],
            dtype=torch.float32,
        ),
        split="train_holdout",
        penalty_names=penalty_names,
    )

    rows = _candidate_channel_stability_rows(
        fit_stats=fit_stats,
        holdout_stats=holdout_stats,
        val_stats=None,
        min_support=3,
        margin=0.0,
        positive_rate_threshold=0.60,
    )

    by_channel = {int(row["channel"]): row for row in rows}
    assert by_channel[0]["stable_train_splits"] is True
    assert by_channel[1]["stable_train_splits"] is False


def test_static_channel_guard_metrics_uses_only_stable_channel_penalty_rows() -> None:
    y = torch.zeros(1, 2, 2)
    base = torch.ones(1, 2, 2)
    cand = torch.stack([torch.zeros_like(base), torch.full_like(base, 3.0)], dim=2)
    rows = [
        {
            "channel": 0,
            "penalty_idx": 0,
            "penalty": "trend",
            "stable_train_splits": True,
            "holdout_mean_gain": 1.0,
        },
        {
            "channel": 1,
            "penalty_idx": 1,
            "penalty": "direction",
            "stable_train_splits": False,
            "holdout_mean_gain": 9.0,
        },
    ]

    metrics = _static_channel_guard_metrics(
        base_bch=base,
        cand_bcpH=cand,
        y_bch=y,
        channel_stability_rows=rows,
    )

    assert metrics["selected_mse"] == pytest.approx(0.5)
    assert metrics["selected_mae"] == pytest.approx(0.5)
    assert metrics["candidate_use_rate_channel"] == pytest.approx(0.5)
    assert metrics["selected_channels"] == {"0": "trend"}


def test_normalize_requested_splits_refuses_test() -> None:
    with pytest.raises(ValueError, match="refuses to read test"):
        _normalize_requested_splits(["train_fit", "test"])
