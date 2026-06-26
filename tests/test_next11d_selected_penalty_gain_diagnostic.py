import pytest
import torch

from scripts.next11d_selected_penalty_gain_diagnostic import (
    _normalize_requested_splits,
    _selected_penalty_gain_summary,
)


def test_selected_penalty_gain_summary_counts_effective_and_harmful_routes() -> None:
    gain = torch.tensor(
        [
            [[0.20, -0.10], [0.05, 0.30]],
            [[0.40, 0.10], [-0.20, 0.25]],
            [[-0.10, 0.20], [0.00, -0.30]],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [
            [1, 2],
            [0, 2],
            [2, 0],
        ],
        dtype=torch.long,
    )
    current = torch.tensor(
        [
            [1, 0],
            [1, 2],
            [2, 1],
        ],
        dtype=torch.long,
    )

    summary = _selected_penalty_gain_summary(
        split="val",
        gain_bkp=gain,
        labels_bk=labels,
        current_pred_bk=current,
        penalty_names=["trend", "direction"],
        action_margin=0.001,
    )

    assert summary["route_count"] == 6
    assert summary["selected_count"] == 5
    assert summary["selected_rate"] == pytest.approx(5 / 6)
    assert summary["oracle_positive_count"] == 4
    assert summary["selected_label_precision"] == pytest.approx(3 / 5)
    assert summary["selected_false_adopt_rate"] == pytest.approx(2 / 5)
    assert summary["missed_positive_count"] == 1
    assert summary["missed_positive_rate_on_oracle_positive"] == pytest.approx(1 / 4)
    assert summary["selected_gain_mean_on_selected"] == pytest.approx((0.20 + 0.40 + 0.25 + 0.20 + 0.0) / 5)
    assert summary["selected_gain_positive_rate"] == pytest.approx(4 / 5)
    assert summary["selected_gain_above_margin_rate"] == pytest.approx(4 / 5)
    assert summary["selected_gain_nonpositive_count"] == 1

    trend = summary["per_penalty"]["trend"]
    assert trend["selected_count"] == 3
    assert trend["label_precision"] == pytest.approx(1 / 3)
    assert trend["false_adopt_rate"] == pytest.approx(2 / 3)
    assert trend["gain_mean"] == pytest.approx((0.20 + 0.40 + 0.0) / 3)

    direction = summary["per_penalty"]["direction"]
    assert direction["selected_count"] == 2
    assert direction["label_precision"] == pytest.approx(2 / 2)
    assert direction["false_adopt_rate"] == pytest.approx(0.0)
    assert direction["gain_mean"] == pytest.approx((0.25 + 0.20) / 2)


def test_selected_penalty_gain_summary_rejects_shape_mismatch() -> None:
    gain = torch.zeros(2, 1, 2)
    labels = torch.zeros(2, 1, dtype=torch.long)
    current = torch.zeros(2, 2, dtype=torch.long)

    with pytest.raises(ValueError, match="labels_bk and current_pred_bk must share"):
        _selected_penalty_gain_summary(
            split="val",
            gain_bkp=gain,
            labels_bk=labels,
            current_pred_bk=current,
            penalty_names=["trend", "direction"],
        )


def test_selected_penalty_gain_diagnostic_refuses_test_split() -> None:
    with pytest.raises(ValueError, match="refuses to read test"):
        _normalize_requested_splits(["train_fit", "test"])
