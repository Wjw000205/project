import pytest
import torch

from scripts.next11d_candidate_objective_alignment import (
    _gain_hinge_pressure_metrics,
    _split_branch_pressure_rows,
)


def test_gain_hinge_pressure_separates_any_positive_from_all_positive_requirement() -> None:
    gain = torch.tensor(
        [
            [[0.20, -0.10], [0.30, 0.40]],
            [[-0.20, -0.30], [0.01, -0.02]],
        ],
        dtype=torch.float32,
    )

    metrics = _gain_hinge_pressure_metrics(
        gain_bcp=gain,
        allowed_cp=None,
        margin=0.0,
    )

    assert metrics["sample_count"] == 4
    assert metrics["allowed_branch_count"] == 8
    assert metrics["any_positive_rate"] == pytest.approx(0.75)
    assert metrics["all_allowed_positive_rate"] == pytest.approx(0.25)
    assert metrics["skip_target_rate"] == pytest.approx(0.25)
    assert metrics["single_positive_rate"] == pytest.approx(0.50)
    assert metrics["multi_positive_rate"] == pytest.approx(0.25)
    assert metrics["active_branch_loss_rate"] == pytest.approx(0.5)
    assert metrics["zero_loss_all_branch_sample_rate"] == pytest.approx(0.25)
    assert metrics["loss_share_from_skip_target_samples"] == pytest.approx(0.5 / 0.62)


def test_gain_hinge_pressure_respects_channel_penalty_allowed_mask() -> None:
    gain = torch.tensor(
        [
            [[0.20, -10.0], [-10.0, 0.40]],
            [[0.10, -10.0], [-10.0, -0.20]],
        ],
        dtype=torch.float32,
    )
    allowed = torch.tensor([[True, False], [False, True]])

    metrics = _gain_hinge_pressure_metrics(
        gain_bcp=gain,
        allowed_cp=allowed,
        margin=0.0,
    )

    assert metrics["sample_count"] == 4
    assert metrics["allowed_branch_count"] == 4
    assert metrics["any_positive_rate"] == pytest.approx(0.75)
    assert metrics["all_allowed_positive_rate"] == pytest.approx(0.75)
    assert metrics["active_branch_loss_rate"] == pytest.approx(0.25)


def test_split_branch_pressure_rows_report_loss_share_per_channel_penalty() -> None:
    gain = torch.tensor(
        [
            [[0.20, -0.10]],
            [[-0.30, -0.20]],
        ],
        dtype=torch.float32,
    )

    rows = _split_branch_pressure_rows(
        gain_bcp=gain,
        split="train_fit",
        penalty_names=["trend", "direction"],
        allowed_cp=None,
        margin=0.0,
    )

    by_penalty = {row["penalty"]: row for row in rows}
    assert by_penalty["trend"]["positive_rate"] == pytest.approx(0.5)
    assert by_penalty["trend"]["mean_hinge_loss"] == pytest.approx(0.15)
    assert by_penalty["trend"]["loss_share"] == pytest.approx(0.5)
    assert by_penalty["direction"]["positive_rate"] == pytest.approx(0.0)
    assert by_penalty["direction"]["mean_hinge_loss"] == pytest.approx(0.15)
    assert by_penalty["direction"]["loss_share"] == pytest.approx(0.5)
