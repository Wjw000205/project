import pytest
import torch

from scripts.next11d_skip_zero_diagnostic import (
    _best_penalty_gain_and_labels_from_penalty_tensors,
    _cluster_best_penalty_gain_and_labels,
    _cluster_penalty_gain_and_delta,
    _per_penalty_support_summary,
    _reject_test_splits,
    _skip_margin_summary,
)


def test_cluster_best_penalty_gain_and_labels_respect_allowed_mask() -> None:
    base_err = torch.tensor(
        [
            [1.0, 1.0, 4.0],
            [1.0, 1.0, 4.0],
        ]
    )
    cand_err = torch.tensor(
        [
            [[0.5, 2.0], [0.5, 2.0], [5.0, 3.0]],
            [[1.2, 0.0], [1.2, 0.0], [4.5, 1.0]],
        ]
    )
    cluster_id = torch.tensor([0, 0, 1])
    allowed = torch.tensor([[1, 0], [0, 1]], dtype=torch.bool)

    gain, labels = _cluster_best_penalty_gain_and_labels(
        base_err_bc=base_err,
        cand_err_bcp=cand_err,
        cluster_id_c=cluster_id,
        K=2,
        allowed_mask_kp=allowed,
    )

    torch.testing.assert_close(gain, torch.tensor([[0.5, 1.0], [-0.2, 3.0]]))
    assert labels.tolist() == [[1, 2], [0, 2]]


def test_skip_margin_summary_reports_near_zero_and_confusion() -> None:
    gain = torch.tensor([[-0.0001, 0.0002], [-0.5, 0.6]])
    labels = torch.tensor([[0, 1], [0, 2]])
    current = torch.tensor([[0, 0], [1, 2]])

    summary = _skip_margin_summary(
        best_penalty_gain_bk=gain,
        labels_bk=labels,
        current_pred_bk=current,
        near_zero_thresholds=[0.001, 0.1],
    )

    assert summary["samples"] == 4
    assert summary["oracle_skip_rate"] == pytest.approx(0.5)
    assert summary["actual_skip_rate"] == pytest.approx(0.5)
    assert summary["skip_recall"] == pytest.approx(0.5)
    assert summary["skip_precision"] == pytest.approx(0.5)
    assert summary["near_zero_abs_gain_rates"]["0.001"] == pytest.approx(0.5)
    assert summary["near_zero_abs_gain_rates"]["0.1"] == pytest.approx(0.5)
    assert summary["oracle_skip_gain_mean"] == pytest.approx(-0.25005)
    assert summary["oracle_penalty_gain_mean"] == pytest.approx(0.3001)


def test_cluster_penalty_gain_and_delta_respect_allowed_mask() -> None:
    base_err = torch.tensor([[1.0, 1.0, 4.0]])
    cand_err = torch.tensor([[[0.5, 2.0], [0.25, 3.0], [5.0, 3.0]]])
    delta_rms = torch.tensor([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]])
    cluster_id = torch.tensor([0, 0, 1])
    allowed = torch.tensor([[1, 0], [0, 1]], dtype=torch.bool)

    gain, delta = _cluster_penalty_gain_and_delta(
        base_err_bc=base_err,
        cand_err_bcp=cand_err,
        delta_rms_bcp=delta_rms,
        cluster_id_c=cluster_id,
        K=2,
        allowed_mask_kp=allowed,
    )

    torch.testing.assert_close(gain, torch.tensor([[[0.625, float("nan")], [float("nan"), 1.0]]]), equal_nan=True)
    torch.testing.assert_close(delta, torch.tensor([[[0.2, float("nan")], [float("nan"), 0.6]]]), equal_nan=True)


def test_per_penalty_support_summary_reports_action_size_and_gain_signs() -> None:
    gain = torch.tensor(
        [
            [[0.002, -0.003], [0.0, 0.004]],
            [[0.0002, -0.0002], [float("nan"), -0.005]],
        ]
    )
    delta = torch.tensor(
        [
            [[0.10, 0.20], [0.00, 0.40]],
            [[0.05, 0.15], [float("nan"), 0.30]],
        ]
    )

    summary = _per_penalty_support_summary(
        penalty_gain_bkp=gain,
        penalty_delta_rms_bkp=delta,
        penalty_names=["trend", "direction"],
        strong_threshold=0.001,
        near_zero_threshold=0.001,
    )

    trend = summary["trend"]
    direction = summary["direction"]
    assert trend["samples"] == 3
    assert trend["positive_rate"] == pytest.approx(2 / 3)
    assert trend["strong_positive_rate"] == pytest.approx(1 / 3)
    assert trend["strong_negative_rate"] == pytest.approx(0.0)
    assert trend["near_zero_rate"] == pytest.approx(2 / 3)
    assert trend["delta_rms_mean"] == pytest.approx(0.05)
    assert direction["samples"] == 4
    assert direction["strong_negative_rate"] == pytest.approx(2 / 4)
    assert direction["delta_rms_mean"] == pytest.approx(0.2625)


def test_action_floor_oracle_ignores_near_noop_penalty_candidates() -> None:
    gain = torch.tensor(
        [
            [[-0.0030, 0.0001]],
            [[0.0040, 0.0001]],
            [[0.0002, 0.0003]],
        ]
    )
    delta = torch.tensor(
        [
            [[0.0200, 0.0002]],
            [[0.0200, 0.0002]],
            [[0.0004, 0.0003]],
        ]
    )

    no_floor_gain, no_floor_labels, no_floor_has_action = _best_penalty_gain_and_labels_from_penalty_tensors(
        penalty_gain_bkp=gain,
        penalty_delta_rms_bkp=delta,
        min_delta_rms=0.0,
    )
    floor_gain, floor_labels, floor_has_action = _best_penalty_gain_and_labels_from_penalty_tensors(
        penalty_gain_bkp=gain,
        penalty_delta_rms_bkp=delta,
        min_delta_rms=0.001,
    )

    torch.testing.assert_close(no_floor_gain, torch.tensor([[0.0001], [0.0040], [0.0003]]))
    assert no_floor_labels.tolist() == [[2], [1], [2]]
    assert no_floor_has_action.tolist() == [[True], [True], [True]]
    torch.testing.assert_close(floor_gain, torch.tensor([[-0.0030], [0.0040], [0.0]]))
    assert floor_labels.tolist() == [[0], [1], [0]]
    assert floor_has_action.tolist() == [[True], [True], [False]]


def test_skip_zero_diagnostic_rejects_test_split() -> None:
    _reject_test_splits(["train_fit", "train_holdout", "val"])

    with pytest.raises(ValueError, match="No test read"):
        _reject_test_splits(["train_fit", "test"])
