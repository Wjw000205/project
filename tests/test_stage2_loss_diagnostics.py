import math

import pytest
import torch

from src.train import (
    _parameter_grad_l2_norm,
    _stage2_loss_epoch_summary,
    _stage2_route_epoch_summary,
)


def test_stage2_loss_epoch_summary_reports_weighted_total_and_components() -> None:
    cluster_weight = torch.tensor([0.25, 0.75])
    summary = _stage2_loss_epoch_summary(
        epoch=3,
        count=2,
        cluster_weight_k=cluster_weight,
        total_loss_sum_k=torch.tensor([4.0, 8.0]),
        forecast_loss_sum_k=torch.tensor([1.0, 3.0]),
        penalty_loss_sum_k=torch.tensor([0.5, 1.5]),
        pred_residual_aux_loss_sum_k=torch.tensor([0.25, 0.75]),
        candidate_supervision_loss_sum_k=torch.tensor([0.1, 0.3]),
        gate_utility_loss_sum_k=torch.tensor([0.2, 0.4]),
        skip_noop_loss_sum_k=torch.tensor([0.0, 0.2]),
        intervention_supervision_loss_sum_k=torch.tensor([0.05, 0.15]),
        other_aux_loss_sum_k=torch.tensor([0.9, 0.7]),
        train_mse_sum_k=torch.tensor([2.0, 6.0]),
        train_mae_sum_k=torch.tensor([1.0, 5.0]),
    )

    assert summary["epoch"] == 3
    assert summary["total_train_loss"] == pytest.approx(3.5)
    assert summary["forecast_loss_only"] == pytest.approx(1.25)
    assert summary["aux_penalty_loss"] == pytest.approx(0.625)
    assert summary["pred_residual_aux_loss"] == pytest.approx(0.3125)
    assert summary["candidate_supervision_loss"] == pytest.approx(0.125)
    assert summary["gate_utility_loss"] == pytest.approx(0.175)
    assert summary["skip_noop_loss"] == pytest.approx(0.075)
    assert summary["intervention_supervision_loss"] == pytest.approx(0.0625)
    assert summary["other_aux_loss"] == pytest.approx(0.375)
    assert summary["train_mse"] == pytest.approx(2.5)
    assert summary["train_mae"] == pytest.approx(2.0)


def test_stage2_route_epoch_summary_reports_entropy_distribution_and_skip() -> None:
    cluster_weight = torch.tensor([0.25, 0.75])
    summary = _stage2_route_epoch_summary(
        penalty_names=["jump", "delta"],
        cluster_weight_k=cluster_weight,
        route_count_k=torch.tensor([2.0, 4.0]),
        route_prob_sum_kp=torch.tensor([[1.5, 0.5], [1.0, 3.0]]),
        route_actual_sum_kp=torch.tensor([[2.0, 0.0], [1.0, 3.0]]),
        route_entropy_sum_k=torch.tensor([1.0, 2.0]),
        skip_prob_sum_k=torch.tensor([0.4, 2.0]),
        skip_active_sum_k=torch.tensor([0.0, 1.0]),
    )

    assert summary["route_entropy"] == pytest.approx(0.5)
    assert summary["skip_prob"] == pytest.approx(0.425)
    assert summary["skip_noop_rate"] == pytest.approx(0.1875)
    assert summary["actual_route_distribution"]["jump"] == pytest.approx(0.25 * 1.0 + 0.75 * 0.25)
    assert summary["actual_route_distribution"]["delta"] == pytest.approx(0.25 * 0.0 + 0.75 * 0.75)
    assert summary["per_cluster"][0]["prob_distribution"]["jump"] == pytest.approx(0.75)
    assert summary["per_cluster"][1]["actual_distribution"]["delta"] == pytest.approx(0.75)


def test_parameter_grad_l2_norm_ignores_missing_grads() -> None:
    p1 = torch.nn.Parameter(torch.zeros(2))
    p2 = torch.nn.Parameter(torch.zeros(1))
    p3 = torch.nn.Parameter(torch.zeros(1))
    p1.grad = torch.tensor([3.0, 4.0])
    p2.grad = torch.tensor([12.0])

    assert _parameter_grad_l2_norm([p1, p2, p3]) == pytest.approx(13.0)

    p1.grad = None
    p2.grad = None
    assert math.isclose(_parameter_grad_l2_norm([p1, p2]), 0.0)
