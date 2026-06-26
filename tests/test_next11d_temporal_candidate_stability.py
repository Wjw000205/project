import pytest
import torch

from scripts.next11d_temporal_candidate_stability import (
    _classify_temporal_stability,
    _segment_candidate_stats,
    _segment_slices,
    _temporal_stability_rows,
)


def test_segment_slices_cover_samples_without_empty_segments() -> None:
    assert _segment_slices(10, 4) == [(0, 3), (3, 6), (6, 8), (8, 10)]
    assert _segment_slices(3, 5) == [(0, 1), (1, 2), (2, 3)]

    with pytest.raises(ValueError, match="segments must be positive"):
        _segment_slices(4, 0)


def test_segment_candidate_stats_reports_cluster_segment_gain_signs() -> None:
    rows = _segment_candidate_stats(
        gain_bep=torch.tensor(
            [
                [[0.3]],
                [[0.1]],
                [[-0.2]],
                [[-0.4]],
            ],
            dtype=torch.float32,
        ),
        split="train_fit",
        entity_name="cluster",
        penalty_names=["trend"],
        segments=2,
    )

    assert len(rows) == 2
    assert rows[0]["segment"] == 0
    assert rows[0]["segment_start"] == 0
    assert rows[0]["segment_end"] == 2
    assert rows[0]["cluster"] == 0
    assert rows[0]["mean_gain"] == pytest.approx(0.2)
    assert rows[0]["positive_rate"] == pytest.approx(1.0)
    assert rows[1]["mean_gain"] == pytest.approx(-0.3)
    assert rows[1]["positive_rate"] == pytest.approx(0.0)


def test_temporal_stability_rows_detect_train_stable_candidate_that_flips_on_val() -> None:
    penalty_names = ["trend"]
    fit_stats = _segment_candidate_stats(
        gain_bep=torch.tensor([[[0.2]], [[0.1]], [[0.3]], [[0.4]]], dtype=torch.float32),
        split="train_fit",
        entity_name="cluster",
        penalty_names=penalty_names,
        segments=2,
    )
    holdout_stats = _segment_candidate_stats(
        gain_bep=torch.tensor([[[0.4]], [[0.2]], [[0.1]], [[0.2]]], dtype=torch.float32),
        split="train_holdout",
        entity_name="cluster",
        penalty_names=penalty_names,
        segments=2,
    )
    val_stats = _segment_candidate_stats(
        gain_bep=torch.tensor([[[0.2]], [[0.1]], [[-0.3]], [[-0.2]]], dtype=torch.float32),
        split="val",
        entity_name="cluster",
        penalty_names=penalty_names,
        segments=2,
    )

    rows = _temporal_stability_rows(
        fit_segment_stats=fit_stats,
        holdout_segment_stats=holdout_stats,
        val_segment_stats=val_stats,
        entity_name="cluster",
        min_segment_support=2,
        margin=0.0,
        positive_rate_threshold=0.50,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["cluster"] == 0
    assert row["stable_train_segments"] is True
    assert row["fit_positive_segments"] == 2
    assert row["holdout_positive_segments"] == 2
    assert row["val_positive_segments"] == 1
    assert row["val_segment_sign_agrees"] is False


def test_temporal_classifier_distinguishes_no_stable_candidate_from_val_shift() -> None:
    no_stable = [
        {
            "stable_train_segments": False,
            "val_segment_sign_agrees": True,
        }
    ]
    shifted = [
        {
            "stable_train_segments": True,
            "val_segment_sign_agrees": False,
        }
    ]

    assert _classify_temporal_stability(no_stable)["failure_layer"] == "adapter candidate quality"
    verdict = _classify_temporal_stability(shifted)
    assert verdict["failure_layer"] == "train-val utility shift"
    assert verdict["decision"] == "temporal_candidate_train_stable_but_val_flips"
