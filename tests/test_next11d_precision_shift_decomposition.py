import pytest
import torch

from scripts.next11d_precision_shift_decomposition import (
    _classify_shift,
    _decomposition_rows,
    _shift_rows,
)


def test_decomposition_rows_measure_channel_penalty_precision_and_gain() -> None:
    labels = torch.tensor(
        [
            [1, 0],
            [0, 1],
            [2, 0],
        ],
        dtype=torch.long,
    )
    pred = torch.tensor(
        [
            [1, 1],
            [1, 0],
            [2, 2],
        ],
        dtype=torch.long,
    )
    gain = torch.tensor(
        [
            [[0.5, 0.0], [-0.2, 0.0]],
            [[-1.0, 0.0], [0.4, 0.0]],
            [[0.0, 0.3], [0.0, -0.4]],
        ],
        dtype=torch.float32,
    )

    rows = _decomposition_rows(
        labels_bc=labels,
        pred_bc=pred,
        gain_bcp=gain,
        penalty_names=["trend", "direction"],
    )

    by_key = {(row["channel"], row["penalty"]): row for row in rows}
    c0_trend = by_key[(0, "trend")]
    assert c0_trend["pred_count"] == 2
    assert c0_trend["exact_precision"] == pytest.approx(0.5)
    assert c0_trend["any_positive_precision"] == pytest.approx(0.5)
    assert c0_trend["false_skip_apply_count"] == 1
    assert c0_trend["mean_gain"] == pytest.approx(-0.25)
    assert c0_trend["negative_gain_rate"] == pytest.approx(0.5)

    c0_direction = by_key[(0, "direction")]
    assert c0_direction["exact_precision"] == pytest.approx(1.0)
    assert c0_direction["mean_gain"] == pytest.approx(0.3)

    c1_direction = by_key[(1, "direction")]
    assert c1_direction["false_skip_apply_count"] == 1
    assert c1_direction["mean_gain"] == pytest.approx(-0.4)


def test_shift_rows_and_classifier_flag_concentrated_val_contamination() -> None:
    holdout_rows = [
        {
            "channel": 0,
            "penalty": "trend",
            "pred_count": 10,
            "exact_precision": 0.9,
            "any_positive_precision": 0.9,
            "mean_gain": 0.3,
            "false_skip_apply_count": 1,
        },
        {
            "channel": 1,
            "penalty": "trend",
            "pred_count": 2,
            "exact_precision": 0.5,
            "any_positive_precision": 0.5,
            "mean_gain": 0.0,
            "false_skip_apply_count": 1,
        },
    ]
    val_rows = [
        {
            "channel": 0,
            "penalty": "trend",
            "pred_count": 20,
            "exact_precision": 0.2,
            "any_positive_precision": 0.2,
            "mean_gain": -0.4,
            "false_skip_apply_count": 16,
        },
        {
            "channel": 1,
            "penalty": "trend",
            "pred_count": 2,
            "exact_precision": 0.5,
            "any_positive_precision": 0.5,
            "mean_gain": 0.0,
            "false_skip_apply_count": 1,
        },
    ]

    shifts = _shift_rows(holdout_rows=holdout_rows, val_rows=val_rows)
    verdict = _classify_shift(shifts)

    by_channel = {row["channel"]: row for row in shifts}
    assert by_channel[0]["exact_precision_delta"] == pytest.approx(-0.7)
    assert by_channel[0]["mean_gain_delta"] == pytest.approx(-0.7)
    assert verdict["failure_layer"] == "train-val utility shift"
    assert verdict["decision"] == "precision_shift_concentrated_by_channel_penalty"
