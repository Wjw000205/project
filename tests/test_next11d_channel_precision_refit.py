import pytest
import torch

from scripts.next11d_channel_precision_refit import (
    _channel_feature_table,
    _select_precision_thresholds,
)


def test_channel_feature_table_puts_skip_and_penalty_features_in_class_slots() -> None:
    skip_feat = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    cand_feat = torch.tensor([[[[10.0, 20.0], [30.0, 40.0]], [[50.0, 60.0], [70.0, 80.0]]]])

    table = _channel_feature_table(skip_feat_bcf=skip_feat, cand_feat_bcpf=cand_feat)

    assert table.shape == (2, 3, 2)
    assert table[0, 0].tolist() == [1.0, 2.0]
    assert table[0, 1].tolist() == [10.0, 20.0]
    assert table[0, 2].tolist() == [30.0, 40.0]
    assert table[1, 0].tolist() == [3.0, 4.0]
    assert table[1, 1].tolist() == [50.0, 60.0]
    assert table[1, 2].tolist() == [70.0, 80.0]


def test_select_precision_thresholds_prefers_high_recall_under_precision_floor() -> None:
    scores = torch.tensor(
        [
            [0.95],
            [0.90],
            [0.80],
            [0.70],
            [0.60],
            [0.40],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([1, 1, 0, 1, 0, 0], dtype=torch.long)

    thresholds, summary = _select_precision_thresholds(
        scores_np=scores,
        labels=labels,
        label_names=["skip", "trend"],
        precision_floor=0.80,
        min_recall=0.0,
        max_thresholds_per_penalty=6,
        max_threshold_combinations=100,
        min_apply_rate=0.0,
    )

    assert thresholds.shape == (1,)
    assert summary["best_metrics"]["positive_precision_any"] >= 0.80
    assert summary["best_metrics"]["positive_recall_any"] == pytest.approx(2.0 / 3.0)
    assert summary["best_metrics"]["prediction_counts"]["trend"] == 2


def test_select_precision_thresholds_can_prioritize_positive_utility() -> None:
    scores = torch.tensor(
        [
            [0.95],
            [0.90],
            [0.80],
            [0.70],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([1, 1, 0, 1], dtype=torch.long)
    gain = torch.tensor([[0.5], [0.4], [-2.0], [0.01]], dtype=torch.float32)

    thresholds, summary = _select_precision_thresholds(
        scores_np=scores,
        labels=labels,
        label_names=["skip", "trend"],
        precision_floor=0.70,
        min_recall=0.0,
        max_thresholds_per_penalty=4,
        max_threshold_combinations=100,
        min_apply_rate=0.0,
        gain_np=gain,
        selection_objective="utility_gain",
    )

    assert thresholds.shape == (1,)
    assert summary["best_gain_mean"] == pytest.approx(0.225)
    assert summary["best_metrics"]["positive_recall_any"] == pytest.approx(2.0 / 3.0)
