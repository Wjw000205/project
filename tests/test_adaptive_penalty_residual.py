from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.moe_gate import ClusterwiseMoEGate
from src.models.residual_moe import (
    ChannelPatchPenaltyRouter,
    ClusterwisePredResidualMoE,
)
from src.train import (
    _freeze_module_params_except_prefixes,
    _causal_patch_regime_descriptor,
    _causal_patch_scale_features,
    _patch_router_expected_mse_loss_bk,
    _patch_router_mixture_mse_loss_bk,
    _patch_router_hierarchical_recall_loss_terms,
    _patch_router_oracle_batch_stats,
    _patch_router_oracle_ce_loss_bk,
    _pred_residual_candidate_predictions,
    _pred_residual_candidates_on_eval_path,
    _loss_gradient_overlap_summary,
    _risk_score_threshold_curve_summary,
    _temporal_group_dro_incremental_loss,
    _select_recall_constrained_risk_threshold,
    _select_recall_constrained_risk_threshold_by_penalty,
    _walk_forward_patch_reliability_metrics,
    _causal_expert_feedback_ridge_metrics,
    _walk_forward_expert_reliability_rerank_metrics,
)


def test_recall_constrained_risk_threshold_uses_temporal_net_gain() -> None:
    result = _select_recall_constrained_risk_threshold(
        score_n=torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5]),
        gain_n=torch.tensor([1.0, -0.2, 0.8, -2.0, 1.0]),
        block_n=torch.tensor([0, 0, 1, 1, 1]),
        min_gain_cost_ratio=1.0,
        min_block_net_gain=0.0,
    )
    assert result["status"] == "ok"
    assert result["threshold"] == pytest.approx(0.65)
    assert result["selected_count"] == 3
    assert result["selected_positive_count"] == 2
    assert result["positive_recall"] == pytest.approx(2.0 / 3.0)
    assert result["gain_cost_ratio"] == pytest.approx(9.0)
    assert result["block_net_gain"] == pytest.approx([0.8, 0.8])


def test_recall_constrained_risk_threshold_calibrates_each_penalty() -> None:
    result = _select_recall_constrained_risk_threshold_by_penalty(
        score_n=torch.tensor([0.9, 0.8, 0.7, 0.95, 0.85, 0.75]),
        gain_n=torch.tensor([1.0, -0.2, 0.8, -2.0, 1.0, 1.0]),
        block_n=torch.tensor([0, 0, 1, 0, 0, 1]),
        penalty_n=torch.tensor([0, 0, 0, 1, 1, 1]),
        penalty_names=["stable", "unstable"],
        min_gain_cost_ratio=1.0,
        min_block_net_gain=0.0,
    )

    assert result["threshold_by_penalty"]["stable"] < 0.7
    assert result["threshold_by_penalty"]["unstable"] > 0.95
    assert result["selected_count"] == 3
    assert result["selected_positive_count"] == 2
    assert result["positive_gain"] == pytest.approx(1.8)
    assert result["negative_cost"] == pytest.approx(0.2)
    assert result["block_net_gain"] == pytest.approx([0.8, 0.8])


def test_risk_score_threshold_curve_separates_ranking_from_fixed_cutoff() -> None:
    result = _risk_score_threshold_curve_summary(
        score_n=torch.tensor([0.9, 0.8, 0.7, 0.6]),
        gain_n=torch.tensor([1.0, -0.2, 0.8, -2.0]),
        fixed_threshold=0.75,
    )

    assert result["fixed"]["selected_count"] == 2
    assert result["fixed"]["positive_recall"] == pytest.approx(0.5)
    assert result["fixed"]["net_gain"] == pytest.approx(0.8)
    assert result["max_net_gain"]["selected_count"] == 3
    assert result["max_net_gain"]["positive_recall"] == pytest.approx(1.0)
    assert result["max_net_gain"]["net_gain"] == pytest.approx(1.6)
    assert result["max_recall_nonnegative"]["positive_recall"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("score", "expected_auc", "expected_top_precision", "correlation_sign"),
    [
        ([0.9, 0.8, 0.2, 0.1], 1.0, 1.0, 1.0),
        ([0.1, 0.2, 0.8, 0.9], 0.0, 0.0, -1.0),
    ],
)
def test_risk_score_threshold_curve_reports_target_overlap(
    score: list[float],
    expected_auc: float,
    expected_top_precision: float,
    correlation_sign: float,
) -> None:
    result = _risk_score_threshold_curve_summary(
        score_n=torch.tensor(score),
        gain_n=torch.tensor([2.0, 1.0, -1.0, -2.0]),
        fixed_threshold=0.5,
    )

    assert result["benefit_auroc"] == pytest.approx(expected_auc)
    assert result["benefit_average_precision"] is not None
    assert result["top_prevalence"]["positive_precision"] == pytest.approx(
        expected_top_precision
    )
    assert correlation_sign * result["score_gain_pearson"] > 0.0


def test_risk_score_threshold_curve_treats_constant_scores_as_ties() -> None:
    result = _risk_score_threshold_curve_summary(
        score_n=torch.ones(4),
        gain_n=torch.tensor([2.0, 1.0, -1.0, -2.0]),
        fixed_threshold=0.5,
    )

    assert result["benefit_auroc"] == pytest.approx(0.5)
    assert result["benefit_average_precision"] == pytest.approx(0.5)
    assert result["score_gain_pearson"] is None


def test_loss_gradient_overlap_reports_aligned_opposed_and_disjoint_terms() -> None:
    x = torch.nn.Parameter(torch.tensor(1.0))
    y = torch.nn.Parameter(torch.tensor(2.0))
    z = torch.nn.Parameter(torch.tensor(3.0))
    reference = x.square() + y.square()
    result = _loss_gradient_overlap_summary(
        reference_loss=reference,
        term_losses={
            "aligned": 3.0 * reference,
            "opposed": -reference,
            "disjoint": z.square(),
        },
        named_parameters=[("x", x), ("y", y), ("z", z)],
        parameter_groups={
            "xy": ("x", "y"),
            "all": ("",),
        },
    )

    xy_terms = result["groups"]["xy"]["terms"]
    assert xy_terms["aligned"]["cosine_with_reference"] == pytest.approx(1.0)
    assert xy_terms["opposed"]["cosine_with_reference"] == pytest.approx(-1.0)
    assert xy_terms["disjoint"]["cosine_with_reference"] is None
    assert result["groups"]["all"]["terms"]["disjoint"][
        "cosine_with_reference"
    ] == pytest.approx(0.0)


def test_temporal_group_dro_backpropagates_through_worst_incremental_domain() -> None:
    incremental = torch.nn.Parameter(
        torch.tensor([[-2.0], [-1.0], [1.0], [3.0]])
    )
    loss, domain_ids, domain_losses = _temporal_group_dro_incremental_loss(
        incremental_loss_bk=incremental,
        query_index_b=torch.tensor([0, 1, 8, 9]),
        train_window_count=10,
        cluster_weight_k=torch.ones(1),
        num_domains=2,
        temperature=0.0,
    )
    loss.backward()

    assert domain_ids.tolist() == [0, 1]
    assert domain_losses.detach().tolist() == pytest.approx([-1.5, 2.0])
    assert loss.item() == pytest.approx(2.0)
    assert incremental.grad[:2].abs().sum().item() == pytest.approx(0.0)
    assert incremental.grad[2:].reshape(-1).tolist() == pytest.approx([0.5, 0.5])


def test_temporal_group_dro_smoothmax_is_normalized_for_equal_domains() -> None:
    loss, _, domain_losses = _temporal_group_dro_incremental_loss(
        incremental_loss_bk=torch.ones(4, 1),
        query_index_b=torch.tensor([0, 1, 8, 9]),
        train_window_count=10,
        cluster_weight_k=torch.ones(1),
        num_domains=2,
        temperature=0.1,
    )

    assert domain_losses.tolist() == pytest.approx([1.0, 1.0])
    assert loss.item() == pytest.approx(1.0)


def test_walk_forward_patch_reliability_waits_for_matured_labels() -> None:
    train_time = torch.tensor([0, 1, 2, 3])
    train_gain = -0.25 * torch.ones(4, 1, 1)
    eval_time = torch.tensor([5, 6, 7, 8])
    base_mse = torch.ones(4, 1, 1)
    candidate_mse = torch.tensor([0.0, 0.0, 0.0, 0.0]).view(4, 1, 1)
    base_mae = torch.ones_like(base_mse)
    candidate_mae = torch.zeros_like(base_mse)

    result = _walk_forward_patch_reliability_metrics(
        train_time_n=train_time,
        train_gain_ncq=train_gain,
        eval_time_n=eval_time,
        eval_base_mse_ncq=base_mse,
        eval_candidate_mse_ncq=candidate_mse,
        eval_base_mae_ncq=base_mae,
        eval_candidate_mae_ncq=candidate_mae,
        active_channel_mask_c=torch.tensor([True]),
        label_delay=2,
        lookback_windows=3,
        min_history_windows=1,
        temporal_blocks=4,
    )

    rates = [row["adoption_rate"] for row in result["temporal_blocks"]]
    assert rates[:2] == [0.0, 0.0]
    assert rates[2:] == [1.0, 1.0]
    assert result["selected_mse"] == pytest.approx(0.5)


def test_walk_forward_patch_reliability_rejects_zero_label_delay() -> None:
    values = torch.ones(1, 1, 1)
    with pytest.raises(ValueError, match="label_delay"):
        _walk_forward_patch_reliability_metrics(
            train_time_n=torch.tensor([0]),
            train_gain_ncq=values,
            eval_time_n=torch.tensor([1]),
            eval_base_mse_ncq=values,
            eval_candidate_mse_ncq=torch.zeros_like(values),
            eval_base_mae_ncq=values,
            eval_candidate_mae_ncq=torch.zeros_like(values),
            active_channel_mask_c=torch.tensor([True]),
            label_delay=0,
            lookback_windows=1,
            min_history_windows=1,
        )


def test_causal_patch_regime_support_abstains_on_ood_input() -> None:
    train_x = torch.tensor(
        [
            [[[0.0, 0.1, 0.0, -0.1]]],
            [[[0.1, 0.0, -0.1, 0.0]]],
            [[[-0.1, 0.0, 0.1, 0.0]]],
        ]
    ).reshape(3, 1, 4)
    eval_x = torch.tensor([[[10.0, 10.1, 10.0, 9.9]]])
    train_regime = _causal_patch_regime_descriptor(train_x)
    eval_regime = _causal_patch_regime_descriptor(eval_x)
    values = torch.ones(1, 1, 1)

    result = _walk_forward_patch_reliability_metrics(
        train_time_n=torch.tensor([0, 1, 2]),
        train_gain_ncq=torch.ones(3, 1, 1),
        eval_time_n=torch.tensor([4]),
        eval_base_mse_ncq=values,
        eval_candidate_mse_ncq=torch.zeros_like(values),
        eval_base_mae_ncq=values,
        eval_candidate_mae_ncq=torch.zeros_like(values),
        active_channel_mask_c=torch.tensor([True]),
        train_regime_ncf=train_regime,
        eval_regime_ncf=eval_regime,
        max_abs_regime_z=3.0,
        label_delay=1,
        lookback_windows=3,
        min_history_windows=1,
    )

    assert result["regime_support_rate"] == 0.0
    assert result["adoption_rate"] == 0.0
    assert result["selected_mse"] == pytest.approx(1.0)


def test_walk_forward_least_squares_scale_repairs_unit_correction_overshoot() -> None:
    train_count = 4
    train_cross = torch.ones(train_count, 1, 1)
    train_delta_sq = 4.0 * torch.ones_like(train_cross)
    eval_base_residual = -0.5 * torch.ones(1, 1, 1, 1)
    eval_delta = 2.0 * torch.ones_like(eval_base_residual)
    values = torch.tensor([[[0.25]]])

    result = _walk_forward_patch_reliability_metrics(
        train_time_n=torch.arange(train_count),
        train_gain_ncq=-2.0 * torch.ones(train_count, 1, 1),
        eval_time_n=torch.tensor([4]),
        eval_base_mse_ncq=values,
        eval_candidate_mse_ncq=torch.tensor([[[2.25]]]),
        eval_base_mae_ncq=torch.tensor([[[0.5]]]),
        eval_candidate_mae_ncq=torch.tensor([[[1.5]]]),
        active_channel_mask_c=torch.tensor([True]),
        train_cross_ncq=train_cross,
        train_delta_sq_ncq=train_delta_sq,
        eval_cross_ncq=torch.ones(1, 1, 1),
        eval_delta_sq_ncq=4.0 * torch.ones(1, 1, 1),
        eval_base_residual_ncqr=eval_base_residual,
        eval_candidate_delta_ncqr=eval_delta,
        scale_mode="least_squares",
        max_scale=1.0,
        label_delay=1,
        lookback_windows=4,
        min_history_windows=1,
    )

    assert result["mean_scale"] == pytest.approx(0.25)
    assert result["selected_mse"] == pytest.approx(0.0)
    assert result["selected_mae"] == pytest.approx(0.0)


def test_walk_forward_temporal_scale_consensus_rejects_unstable_history() -> None:
    train_cross = torch.tensor([1.0, 1.0, -1.0, 1.0]).view(4, 1, 1)
    train_delta_sq = torch.ones_like(train_cross)
    values = torch.ones(1, 1, 1)

    result = _walk_forward_patch_reliability_metrics(
        train_time_n=torch.arange(4),
        train_gain_ncq=train_cross,
        eval_time_n=torch.tensor([5]),
        eval_base_mse_ncq=values,
        eval_candidate_mse_ncq=torch.zeros_like(values),
        eval_base_mae_ncq=values,
        eval_candidate_mae_ncq=torch.zeros_like(values),
        active_channel_mask_c=torch.tensor([True]),
        train_cross_ncq=train_cross,
        train_delta_sq_ncq=train_delta_sq,
        eval_cross_ncq=torch.ones_like(values),
        eval_delta_sq_ncq=torch.ones_like(values),
        scale_mode="least_squares",
        max_scale=1.0,
        scale_consensus_blocks=4,
        label_delay=1,
        lookback_windows=4,
        min_history_windows=4,
    )

    assert result["mean_scale"] == 0.0
    assert result["selected_mse"] == pytest.approx(1.0)


def test_causal_patch_scale_features_include_forecast_position() -> None:
    features = _causal_patch_scale_features(
        torch.arange(8, dtype=torch.float32).reshape(1, 1, 8),
        torch.zeros(1, 1, 4),
        torch.ones(1, 1, 2, 2),
    )

    assert features.shape == (1, 1, 2, 21)
    assert not torch.equal(features[:, :, 0, -3:], features[:, :, 1, -3:])


def test_walk_forward_feature_ridge_uses_current_patch_signal() -> None:
    train_cross = torch.tensor([-1.0, -1.0, 1.0, 1.0]).view(4, 1, 1)
    train_feature = train_cross.unsqueeze(-1)
    values = torch.ones(1, 1, 1)

    result = _walk_forward_patch_reliability_metrics(
        train_time_n=torch.arange(4),
        train_gain_ncq=train_cross,
        eval_time_n=torch.tensor([4]),
        eval_base_mse_ncq=values,
        eval_candidate_mse_ncq=torch.zeros_like(values),
        eval_base_mae_ncq=values,
        eval_candidate_mae_ncq=torch.zeros_like(values),
        active_channel_mask_c=torch.tensor([True]),
        train_cross_ncq=train_cross,
        train_delta_sq_ncq=torch.ones_like(train_cross),
        eval_cross_ncq=torch.ones_like(values),
        eval_delta_sq_ncq=torch.ones_like(values),
        train_scale_feature_ncqf=train_feature,
        eval_scale_feature_ncqf=torch.ones(1, 1, 1, 1),
        scale_mode="feature_ridge",
        max_scale=1.0,
        feature_ridge=0.01,
        feature_update_blocks=1,
        label_delay=1,
        lookback_windows=4,
        min_history_windows=4,
    )

    assert result["mean_scale"] > 0.8
    assert result["selected_mse"] < 0.05


def test_walk_forward_patch_end_delay_uses_only_matured_patch_labels() -> None:
    train_count = 6
    train_shape = (train_count, 1, 2)
    eval_shape = (1, 1, 2)
    train_cross = torch.ones(train_shape)
    eval_values = torch.ones(eval_shape)

    result = _walk_forward_patch_reliability_metrics(
        train_time_n=torch.arange(train_count),
        train_gain_ncq=train_cross,
        eval_time_n=torch.tensor([6]),
        eval_base_mse_ncq=eval_values,
        eval_candidate_mse_ncq=torch.zeros_like(eval_values),
        eval_base_mae_ncq=eval_values,
        eval_candidate_mae_ncq=torch.zeros_like(eval_values),
        active_channel_mask_c=torch.tensor([True]),
        train_cross_ncq=train_cross,
        train_delta_sq_ncq=torch.ones(train_shape),
        eval_cross_ncq=eval_values,
        eval_delta_sq_ncq=eval_values,
        scale_mode="least_squares",
        patch_label_delay_q=torch.tensor([1, 4]),
        label_delay=4,
        lookback_windows=6,
        min_history_windows=4,
    )

    patch_rows = result["per_channel_patch"][0]
    assert patch_rows[0]["mean_scale"] == pytest.approx(1.0)
    assert patch_rows[1]["mean_scale"] == pytest.approx(0.0)
    assert result["patch_label_delay"] == [1, 4]


def test_walk_forward_history_stride_uses_only_same_phase_origins() -> None:
    train_time = torch.arange(8)
    train_cross = torch.where(
        train_time.remainder(2).view(-1, 1, 1) == 0,
        torch.ones(8, 1, 1),
        -torch.ones(8, 1, 1),
    )
    eval_values = torch.ones(1, 1, 1)

    result = _walk_forward_patch_reliability_metrics(
        train_time_n=train_time,
        train_gain_ncq=torch.zeros_like(train_cross),
        eval_time_n=torch.tensor([8]),
        eval_base_mse_ncq=eval_values,
        eval_candidate_mse_ncq=torch.zeros_like(eval_values),
        eval_base_mae_ncq=eval_values,
        eval_candidate_mae_ncq=torch.zeros_like(eval_values),
        active_channel_mask_c=torch.tensor([True]),
        train_cross_ncq=train_cross,
        train_delta_sq_ncq=torch.ones_like(train_cross),
        eval_cross_ncq=eval_values,
        eval_delta_sq_ncq=eval_values,
        scale_mode="least_squares",
        label_delay=1,
        lookback_windows=8,
        min_history_windows=4,
        history_stride=2,
    )

    assert result["history_stride"] == 2
    assert result["history_count_min"] == 4
    assert result["history_count_max"] == 4
    assert result["mean_scale"] == pytest.approx(1.0)
    assert result["selected_mse"] == pytest.approx(0.0)


def test_walk_forward_can_condition_reliability_on_selected_expert() -> None:
    result = _walk_forward_patch_reliability_metrics(
        train_time_n=torch.arange(4),
        train_gain_ncq=torch.tensor([1.0, -1.0, 1.0, -1.0]).view(4, 1, 1),
        train_penalty_ncq=torch.tensor([0, 1, 0, 1]).view(4, 1, 1),
        eval_time_n=torch.tensor([4, 5]),
        eval_base_mse_ncq=torch.ones(2, 1, 1),
        eval_candidate_mse_ncq=torch.tensor([0.0, 2.0]).view(2, 1, 1),
        eval_base_mae_ncq=torch.ones(2, 1, 1),
        eval_candidate_mae_ncq=torch.tensor([0.0, 2.0]).view(2, 1, 1),
        eval_penalty_ncq=torch.tensor([0, 1]).view(2, 1, 1),
        active_channel_mask_c=torch.tensor([True]),
        label_delay=1,
        lookback_windows=4,
        min_history_windows=2,
        history_stride=1,
        min_mean_gain=0.0,
    )

    assert result["policy"].endswith("selected_expert")
    assert result["adoption_rate"] == pytest.approx(0.5)
    assert result["adoption_precision"] == pytest.approx(1.0)
    assert result["mse_gain_pct"] == pytest.approx(50.0)


def test_walk_forward_expert_rerank_falls_back_in_input_gate_order() -> None:
    train_gain = torch.tensor(
        [
            [[[-1.0, 1.0]]],
            [[[-1.0, 1.0]]],
        ]
    )
    result = _walk_forward_expert_reliability_rerank_metrics(
        train_time_n=torch.tensor([0, 1]),
        train_gain_ncqp=train_gain,
        eval_time_n=torch.tensor([3]),
        eval_base_mse_ncq=torch.ones(1, 1, 1),
        eval_candidate_mse_ncqp=torch.tensor([[[[2.0, 0.0]]]]),
        eval_base_mae_ncq=torch.ones(1, 1, 1),
        eval_candidate_mae_ncqp=torch.tensor([[[[2.0, 0.0]]]]),
        eval_score_ncqp=torch.tensor([[[[0.9, 0.5]]]]),
        active_channel_mask_c=torch.tensor([True]),
        label_delay=1,
        lookback_windows=4,
        min_history_windows=2,
    )

    assert result["policy"].endswith("input_gate_order")
    assert result["fallback_rate_given_route"] == pytest.approx(1.0)
    assert result["selected_mse"] == pytest.approx(0.0)
    assert result["selected_dual_precision"] == pytest.approx(1.0)


def test_walk_forward_expert_rerank_waits_for_matured_counterfactuals() -> None:
    result = _walk_forward_expert_reliability_rerank_metrics(
        train_time_n=torch.tensor([0, 1]),
        train_gain_ncqp=torch.ones(2, 1, 1, 2),
        eval_time_n=torch.tensor([2, 4]),
        eval_base_mse_ncq=torch.ones(2, 1, 1),
        eval_candidate_mse_ncqp=torch.zeros(2, 1, 1, 2),
        eval_base_mae_ncq=torch.ones(2, 1, 1),
        eval_candidate_mae_ncqp=torch.zeros(2, 1, 1, 2),
        eval_score_ncqp=torch.ones(2, 1, 1, 2),
        active_channel_mask_c=torch.tensor([True]),
        label_delay=2,
        lookback_windows=8,
        min_history_windows=2,
    )

    assert result["temporal_blocks"] == []
    assert result["route_rate"] == pytest.approx(0.5)
    assert result["mse_gain_pct"] == pytest.approx(50.0)


def test_causal_expert_feedback_ridge_learns_input_plus_matured_state() -> None:
    train_count = 8
    base = torch.ones(train_count, 1, 1)
    candidate = torch.tensor([0.0, 2.0]).view(1, 1, 1, 2).expand(
        train_count,
        -1,
        -1,
        -1,
    )
    score = torch.tensor([0.2, 0.9]).view(1, 1, 1, 2).expand_as(candidate)
    result = _causal_expert_feedback_ridge_metrics(
        train_time_n=torch.arange(train_count),
        train_base_mse_ncq=base,
        train_candidate_mse_ncqp=candidate,
        train_base_mae_ncq=base,
        train_candidate_mae_ncqp=candidate,
        train_score_ncqp=score,
        eval_time_n=torch.tensor([9]),
        eval_base_mse_ncq=torch.ones(1, 1, 1),
        eval_candidate_mse_ncqp=torch.tensor([[[[0.0, 2.0]]]]),
        eval_base_mae_ncq=torch.ones(1, 1, 1),
        eval_candidate_mae_ncqp=torch.tensor([[[[0.0, 2.0]]]]),
        eval_score_ncqp=torch.tensor([[[[0.2, 0.9]]]]),
        active_channel_mask_c=torch.tensor([True]),
        label_delay=1,
        lookback_windows=8,
        min_history_windows=2,
        ridge=0.1,
    )

    assert result["policy"].startswith("ridge_dual_utility")
    assert result["route_rate"] == pytest.approx(1.0)
    assert result["selected_mse"] == pytest.approx(0.0)
    assert result["selected_dual_precision"] == pytest.approx(1.0)


def test_causal_expert_feedback_ridge_rejects_zero_delay() -> None:
    base = torch.ones(2, 1, 1)
    candidate = torch.ones(2, 1, 1, 1)
    with pytest.raises(ValueError, match="label_delay"):
        _causal_expert_feedback_ridge_metrics(
            train_time_n=torch.tensor([0, 1]),
            train_base_mse_ncq=base,
            train_candidate_mse_ncqp=candidate,
            train_base_mae_ncq=base,
            train_candidate_mae_ncqp=candidate,
            train_score_ncqp=candidate,
            eval_time_n=torch.tensor([2]),
            eval_base_mse_ncq=torch.ones(1, 1, 1),
            eval_candidate_mse_ncqp=torch.ones(1, 1, 1, 1),
            eval_base_mae_ncq=torch.ones(1, 1, 1),
            eval_candidate_mae_ncqp=torch.ones(1, 1, 1, 1),
            eval_score_ncqp=torch.ones(1, 1, 1, 1),
            active_channel_mask_c=torch.tensor([True]),
            label_delay=0,
            lookback_windows=4,
            min_history_windows=1,
        )


def test_cluster_gate_can_route_noop_as_competing_class() -> None:
    gate = ClusterwiseMoEGate(
        num_clusters=1,
        feat_dim=3,
        num_penalties=2,
        hidden_dim=4,
        topk=1,
        allow_skip=True,
        skip_competes=True,
    )
    with torch.no_grad():
        gate.W1[0].zero_()
        gate.b1[0].zero_()
        gate.W2[0].zero_()
        gate.W_skip[0].zero_()

        gate.b2[0].copy_(torch.tensor([0.0, 0.0]))
        gate.b_skip[0].fill_(8.0)
        mask, probs, skip, skip_prob = gate(torch.zeros(2, 1, 3), straight_through=False)
        assert torch.allclose(mask, torch.zeros_like(mask))
        assert torch.allclose(skip, torch.ones_like(skip))
        assert torch.all(skip_prob > 0.99)
        assert torch.allclose(skip_prob + probs.sum(dim=-1), torch.ones(2, 1), atol=1.0e-6)

        gate.b2[0].copy_(torch.tensor([8.0, -8.0]))
        gate.b_skip[0].fill_(-8.0)
        mask, _, skip, skip_prob = gate(torch.zeros(2, 1, 3), straight_through=False)
        assert torch.allclose(mask[..., 0], torch.ones(2, 1))
        assert torch.allclose(mask[..., 1], torch.zeros(2, 1))
        assert torch.allclose(skip, torch.zeros_like(skip))
        assert torch.all(skip_prob < 1.0e-6)


def test_cluster_gate_argmax_noop_keeps_skip_from_topk_overriding_best_penalty() -> None:
    gate = ClusterwiseMoEGate(
        num_clusters=1,
        feat_dim=3,
        num_penalties=2,
        hidden_dim=4,
        topk=2,
        allow_skip=True,
        skip_competes=True,
        skip_argmax_noop=True,
    )
    with torch.no_grad():
        gate.W1[0].zero_()
        gate.b1[0].zero_()
        gate.W2[0].zero_()
        gate.W_skip[0].zero_()
        gate.b2[0].copy_(torch.tensor([6.0, 0.0]))
        gate.b_skip[0].fill_(4.0)

    mask, _, skip, _ = gate(torch.zeros(2, 1, 3), straight_through=False)

    assert torch.allclose(skip, torch.zeros_like(skip))
    assert torch.allclose(mask[..., 0], torch.ones(2, 1))
    assert torch.allclose(mask[..., 1], torch.ones(2, 1))


def test_cluster_gate_default_skip_competes_topk_behavior_is_unchanged() -> None:
    gate = ClusterwiseMoEGate(
        num_clusters=1,
        feat_dim=3,
        num_penalties=2,
        hidden_dim=4,
        topk=2,
        allow_skip=True,
        skip_competes=True,
    )
    with torch.no_grad():
        gate.W1[0].zero_()
        gate.b1[0].zero_()
        gate.W2[0].zero_()
        gate.W_skip[0].zero_()
        gate.b2[0].copy_(torch.tensor([6.0, 0.0]))
        gate.b_skip[0].fill_(4.0)

    mask, _, skip, _ = gate(torch.zeros(2, 1, 3), straight_through=False)

    assert torch.allclose(skip, torch.ones_like(skip))
    assert torch.allclose(mask, torch.zeros_like(mask))


def test_shared_cluster_gate_uses_one_parameter_set_for_all_clusters() -> None:
    torch.manual_seed(101)
    gate = ClusterwiseMoEGate(
        num_clusters=3,
        feat_dim=4,
        num_penalties=2,
        hidden_dim=5,
        topk=1,
        shared_across_clusters=True,
    )

    feat_one_cluster = torch.randn(2, 1, 4)
    feat = feat_one_cluster.expand(-1, 3, -1).contiguous()
    _, probs, _, _ = gate(feat, straight_through=False)

    assert len(gate.W1) == 1
    assert len(gate.get_cluster_params(0)) > 0
    assert gate.get_cluster_params(1) == []
    assert torch.allclose(probs[:, 0], probs[:, 1])
    assert torch.allclose(probs[:, 0], probs[:, 2])

    for param in gate.get_cluster_params(0):
        param.grad = torch.ones_like(param)
    gate.mask_cluster_grads(torch.tensor([True, False, False]))
    assert all(param.grad is not None and param.grad.abs().sum().item() > 0 for param in gate.get_cluster_params(0))
    gate.mask_cluster_grads(torch.tensor([True, True, True]))
    assert all(param.grad is not None and param.grad.abs().sum().item() == 0 for param in gate.get_cluster_params(0))


def test_adaptive_penalty_selector_is_cluster_specific() -> None:
    torch.manual_seed(11)
    model = ClusterwisePredResidualMoE(
        num_clusters=2,
        num_penalties=2,
        input_len=6,
        pred_len=3,
        hidden_dim=4,
        init_alpha=4.0,
        alpha_scale=1.0,
        intervention_enable=False,
        penalty_selector_enable=True,
        fusion_gate_enable=False,
    )
    with torch.no_grad():
        model.W_selector[0].zero_()
        model.W_selector[1].zero_()
        model.b_selector[0].copy_(torch.tensor([6.0, -6.0]))
        model.b_selector[1].copy_(torch.tensor([-6.0, 6.0]))

    x = torch.randn(2, 4, 6)
    y_base = torch.randn(2, 4, 3)
    cluster_id = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    route = torch.ones(2, 2, 2)
    out = model(x, y_base, cluster_id, route)

    selector = out["selector_bcp"]
    assert selector[0, 0, 0] > 0.99
    assert selector[0, 0, 1] < 0.01
    assert selector[0, 2, 0] < 0.01
    assert selector[0, 2, 1] > 0.99


def test_shared_pred_residual_moe_has_single_expert_pool_across_clusters() -> None:
    torch.manual_seed(103)
    model = ClusterwisePredResidualMoE(
        num_clusters=2,
        num_penalties=2,
        input_len=4,
        pred_len=2,
        hidden_dim=3,
        init_alpha=10.0,
        alpha_scale=1.0,
        intervention_enable=False,
        penalty_selector_enable=False,
        fusion_gate_enable=False,
        shared_across_clusters=True,
    )
    with torch.no_grad():
        for w in model.W1:
            w.zero_()
        for w in model.W2:
            w.zero_()
        model.b2[0].fill_(1.0)
        model.b2[1].fill_(2.0)

    x = torch.randn(1, 4, 4)
    y_base = torch.zeros(1, 4, 2)
    cluster_id = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    route = torch.ones(1, 2, 2)
    out = model(x, y_base, cluster_id, route)

    assert len(model.W1) == 2
    assert len(model.get_cluster_params(0)) > 0
    assert model.get_cluster_params(1) == []
    assert torch.allclose(out["residuals"][0, 0], out["residuals"][0, 2])
    assert torch.allclose(out["alpha_cp"][0], out["alpha_cp"][2])


def test_patch_router_requires_shared_experts() -> None:
    with pytest.raises(ValueError, match="patch_router.*shared_across_clusters"):
        ClusterwisePredResidualMoE(
            num_clusters=2,
            num_penalties=2,
            input_len=4,
            pred_len=4,
            hidden_dim=3,
            shared_across_clusters=False,
            patch_router_cfg={"enable": True, "patch_len": 2},
        )


def test_patch_router_short_history_requires_explicit_projection() -> None:
    with pytest.raises(ValueError, match="short_history_mode=cycle"):
        ClusterwisePredResidualMoE(
            num_clusters=1,
            num_penalties=2,
            input_len=4,
            pred_len=8,
            hidden_dim=3,
            shared_across_clusters=True,
            patch_router_cfg={"enable": True, "patch_len": 2},
        )


def test_patch_router_cycles_causal_input_patches_for_long_forecasts() -> None:
    model = ClusterwisePredResidualMoE(
        num_clusters=1,
        num_penalties=2,
        input_len=4,
        pred_len=8,
        hidden_dim=3,
        intervention_enable=False,
        shared_across_clusters=True,
        patch_router_cfg={
            "enable": True,
            "patch_len": 2,
            "short_history_mode": "cycle",
            "hidden_dim": 3,
            "allow_skip": True,
            "use_base_forecast": True,
            "hierarchical_recall": {
                "enable": True,
                "expert_conditional_risk": {
                    "enable": True,
                    "candidate_aware": True,
                    "candidate_compatibility": True,
                    "decoupled_encoder": True,
                },
            },
        },
    )
    assert model.patch_router is not None
    router = model.patch_router
    x = torch.tensor([[[-2.0, -1.0, 1.0, 2.0]]])
    expected = torch.tensor(
        [[[[ -2.0, -1.0], [1.0, 2.0], [-2.0, -1.0], [1.0, 2.0]]]]
    )
    assert router.history_patch_projection == "cycle"
    assert torch.equal(router._history_patches(x), expected)

    base = torch.zeros(1, 1, 8)
    candidate_delta = torch.randn(1, 1, 2, 8, requires_grad=True)
    routed = router(
        x,
        y_base_bch=base,
        candidate_delta_bcpH=candidate_delta,
        straight_through=False,
    )
    assert routed["patch_penalty_benefit_probs_bcqp"].shape == (1, 1, 4, 2)
    assert routed["patch_penalty_risk_benefit_probs_bcqp"].shape == (1, 1, 4, 2)
    routed["patch_penalty_benefit_probs_bcqp"].sum().backward()
    assert router.W_proposal_candidate is not None
    assert router.W_proposal_candidate.grad is not None
    assert float(router.W_proposal_candidate.grad.abs().sum().item()) > 0.0
    assert candidate_delta.grad is None


def test_input_patch_router_replaces_cluster_route_per_forecast_patch() -> None:
    model = ClusterwisePredResidualMoE(
        num_clusters=2,
        num_penalties=2,
        input_len=4,
        pred_len=4,
        hidden_dim=3,
        init_alpha=10.0,
        alpha_scale=1.0,
        intervention_enable=False,
        shared_across_clusters=True,
        patch_router_cfg={
            "enable": True,
            "patch_len": 2,
            "hidden_dim": 1,
            "topk": 1,
            "allow_skip": False,
            "noise_std": 0.0,
        },
    )
    model.eval()
    assert model.patch_router is not None
    with torch.no_grad():
        for w in model.W1:
            w.zero_()
        for w in model.W2:
            w.zero_()
        model.b2[0].fill_(1.0)
        model.b2[1].fill_(2.0)

        model.patch_router.W1.zero_()
        model.patch_router.b1.zero_()
        model.patch_router.W2.zero_()
        model.patch_router.b2.zero_()
        model.patch_router.W1[model.patch_router.level_feature_index, 0] = 1.0
        model.patch_router.W2[0, 0] = 10.0
        model.patch_router.W2[0, 1] = -10.0

    x = torch.tensor([[[-2.0, -2.0, 2.0, 2.0], [-2.0, -2.0, 2.0, 2.0]]])
    y_base = torch.zeros(1, 2, 4)
    cluster_id = torch.tensor([0, 1], dtype=torch.long)
    cluster_route_a = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    cluster_route_b = 1.0 - cluster_route_a

    out_a = model(x, y_base, cluster_id, cluster_route_a)
    out_b = model(x, y_base, cluster_id, cluster_route_b)
    patch_route = out_a["patch_route_bcph"]

    assert patch_route.shape == (1, 2, 2, 4)
    assert torch.equal(patch_route[..., 0], patch_route[..., 1])
    assert torch.equal(patch_route[..., 2], patch_route[..., 3])
    assert not torch.equal(patch_route[..., 0], patch_route[..., 2])
    assert torch.allclose(out_a["y_final"], out_b["y_final"])
    assert torch.allclose(out_a["patch_route_bcph"], out_b["patch_route_bcph"])


def test_input_patch_router_parameters_belong_to_shared_owner_and_receive_gradients() -> None:
    torch.manual_seed(107)
    model = ClusterwisePredResidualMoE(
        num_clusters=3,
        num_penalties=2,
        input_len=4,
        pred_len=4,
        hidden_dim=3,
        init_alpha=5.0,
        alpha_scale=1.0,
        intervention_enable=False,
        shared_across_clusters=True,
        patch_router_cfg={"enable": True, "patch_len": 2, "hidden_dim": 4, "topk": 1},
    )
    assert model.patch_router is not None
    with torch.no_grad():
        for w in model.W2:
            w.zero_()
        model.b2[0].fill_(0.5)
        model.b2[1].fill_(1.5)

    owner_ids = {id(param) for param in model.get_cluster_params(0)}
    assert all(id(param) in owner_ids for param in model.patch_router.parameters())
    assert model.get_cluster_params(1) == []
    saved_state = model.get_cluster_state(0)
    saved_router_w1 = model.patch_router.W1.detach().clone()
    with torch.no_grad():
        model.patch_router.W1.zero_()
    model.load_cluster_state(0, saved_state)
    assert torch.allclose(model.patch_router.W1, saved_router_w1)

    x = torch.randn(2, 3, 4)
    y_base = torch.zeros(2, 3, 4)
    cluster_id = torch.tensor([0, 1, 2], dtype=torch.long)
    cluster_route = torch.ones(2, 3, 2)
    loss = model(x, y_base, cluster_id, cluster_route)["y_final"].square().mean()
    loss.backward()

    grad_total = sum(
        float(param.grad.abs().sum().item())
        for param in model.patch_router.parameters()
        if param.grad is not None
    )
    assert grad_total > 0.0


def test_hierarchical_patch_router_decouples_adoption_from_penalty_ranking() -> None:
    model = ClusterwisePredResidualMoE(
        num_clusters=2,
        num_penalties=2,
        input_len=4,
        pred_len=4,
        hidden_dim=3,
        intervention_enable=False,
        shared_across_clusters=True,
        patch_router_cfg={
            "enable": True,
            "patch_len": 2,
            "hidden_dim": 3,
            "topk": 1,
            "allow_skip": True,
            "noise_std": 0.0,
            "hierarchical_recall": {"enable": True, "adopt_threshold": 0.5},
        },
    )
    model.eval()
    assert model.patch_router is not None
    router = model.patch_router
    assert router.W_adopt is not None and router.b_adopt is not None
    with torch.no_grad():
        router.W1.zero_()
        router.b1.zero_()
        router.W2.zero_()
        router.b2.copy_(torch.tensor([2.0, -2.0]))
        router.W_adopt.zero_()
        router.b_adopt.fill_(-8.0)

    x = torch.randn(1, 2, 4)
    base = torch.zeros(1, 2, 4)
    cluster_id = torch.tensor([0, 1])
    cluster_route = torch.ones(1, 2, 2)
    skipped = model(x, base, cluster_id, cluster_route)
    assert torch.equal(skipped["patch_skip_bcq"], torch.ones(1, 2, 2))

    with torch.no_grad():
        router.b_adopt.fill_(8.0)
    adopted = model(x, base, cluster_id, cluster_route)
    assert torch.equal(adopted["patch_skip_bcq"], torch.zeros(1, 2, 2))
    assert torch.equal(adopted["patch_route_bcph"][:, :, 0], torch.ones(1, 2, 4))
    assert torch.equal(adopted["patch_route_bcph"][:, :, 1], torch.zeros(1, 2, 4))
    assert torch.allclose(
        adopted["patch_probs_bcqp"].sum(dim=-1) + adopted["patch_skip_prob_bcq"],
        torch.ones(1, 2, 2),
    )


def test_hierarchical_utility_verifier_vetoes_nonpositive_candidates() -> None:
    model = ClusterwisePredResidualMoE(
        num_clusters=1,
        num_penalties=2,
        input_len=4,
        pred_len=4,
        hidden_dim=3,
        intervention_enable=False,
        shared_across_clusters=True,
        patch_router_cfg={
            "enable": True,
            "patch_len": 2,
            "hidden_dim": 2,
            "topk": 1,
            "allow_skip": True,
            "hierarchical_recall": {
                "enable": True,
                "utility_verifier": {"enable": True, "temperature": 0.25},
            },
        },
    )
    model.eval()
    assert model.patch_router is not None
    router = model.patch_router
    assert router.W_adopt is not None and router.b_adopt is not None
    assert router.W_benefit is not None and router.b_benefit is not None
    with torch.no_grad():
        router.W1.zero_()
        router.b1.zero_()
        router.W_adopt.zero_()
        router.b_adopt.fill_(8.0)
        router.W_benefit.zero_()
        router.b_benefit.fill_(2.0)
        router.W2.zero_()
        router.b2.fill_(-2.0)

    x = torch.randn(1, 1, 4)
    base = torch.zeros(1, 1, 4)
    route = torch.ones(1, 1, 2)
    rejected = model(x, base, torch.tensor([0]), route)
    assert torch.equal(rejected["patch_skip_bcq"], torch.ones(1, 1, 2))
    assert torch.all(rejected["patch_proposal_adopt_prob_bcq"] > 0.99)

    with torch.no_grad():
        router.b2[0] = 2.0
    accepted = model(x, base, torch.tensor([0]), route)
    assert torch.equal(accepted["patch_skip_bcq"], torch.zeros(1, 1, 2))


def test_expert_conditional_risk_gate_verifies_the_selected_penalty() -> None:
    model = ClusterwisePredResidualMoE(
        num_clusters=1,
        num_penalties=2,
        input_len=4,
        pred_len=4,
        hidden_dim=3,
        intervention_enable=False,
        shared_across_clusters=True,
        patch_router_cfg={
            "enable": True,
            "patch_len": 2,
            "hidden_dim": 2,
            "topk": 1,
            "allow_skip": True,
            "hierarchical_recall": {
                "enable": True,
                "expert_conditional_risk": {
                    "enable": True,
                    "candidate_aware": False,
                    "decoupled_encoder": False,
                },
            },
        },
    )
    model.eval()
    assert model.patch_router is not None
    router = model.patch_router
    assert router.W_risk_sign is not None and router.b_risk_sign is not None
    assert router.W_risk_gain is not None and router.b_risk_gain is not None
    assert router.W_risk_cost is not None and router.b_risk_cost is not None
    assert router.W_adopt is not None and router.b_adopt is not None
    assert router.W_benefit is not None and router.b_benefit is not None
    with torch.no_grad():
        router.W1.zero_()
        router.b1.zero_()
        router.W_adopt.zero_()
        router.b_adopt.fill_(-8.0)
        router.W_benefit.zero_()
        router.b_benefit.fill_(8.0)
        router.W_risk_sign.zero_()
        router.b_risk_sign.copy_(torch.tensor([4.0, -4.0]))
        router.W_risk_gain.zero_()
        router.b_risk_gain.fill_(-2.0)
        router.W_risk_cost.zero_()
        router.b_risk_cost.fill_(-2.0)

    x = torch.randn(1, 1, 4)
    base = torch.zeros(1, 1, 4)
    route = torch.ones(1, 1, 2)
    accepted = model(x, base, torch.tensor([0]), route)
    assert torch.equal(accepted["patch_skip_bcq"], torch.zeros(1, 1, 2))
    assert torch.equal(accepted["patch_route_bcph"][:, :, 0], torch.ones(1, 1, 4))

    with torch.no_grad():
        router.b_risk_sign.fill_(-4.0)
    rejected = model(x, base, torch.tensor([0]), route)
    assert torch.equal(rejected["patch_skip_bcq"], torch.ones(1, 1, 2))


def test_temporal_domain_risk_heads_train_by_domain_and_average_at_eval() -> None:
    model = ClusterwisePredResidualMoE(
        num_clusters=1,
        num_penalties=1,
        input_len=4,
        pred_len=4,
        hidden_dim=3,
        intervention_enable=False,
        shared_across_clusters=True,
        num_channels=1,
        patch_router_cfg={
            "enable": True,
            "patch_len": 2,
            "hidden_dim": 2,
            "topk": 1,
            "allow_skip": True,
            "fixed_penalty_index_by_channel": [0],
            "hierarchical_recall": {
                "enable": True,
                "expert_conditional_risk": {
                    "enable": True,
                    "candidate_aware": False,
                    "decoupled_encoder": False,
                    "adoption_source": "benefit_probability",
                    "adopt_threshold": 0.5,
                    "temporal_domain_ensemble": {
                        "enable": True,
                        "num_domains": 2,
                        "train_window_count": 4,
                        "combine": "mean",
                    },
                },
            },
        },
    )
    assert model.patch_router is not None
    router = model.patch_router
    assert router.W_risk_sign_domain_delta is not None
    assert router.b_risk_sign_domain_delta is not None
    with torch.no_grad():
        router.W1.zero_()
        router.b1.zero_()
        router.W_risk_sign.zero_()
        router.b_risk_sign.zero_()
        router.W_risk_sign_domain_delta.zero_()
        router.b_risk_sign_domain_delta.copy_(torch.tensor([[4.0], [-4.0]]))

    x = torch.randn(1, 1, 4)
    base = torch.zeros(1, 1, 4)
    route = torch.ones(1, 1, 1)
    model.train()
    domain_zero = model(
        x,
        base,
        torch.tensor([0]),
        route,
        query_start_abs_b=torch.tensor([0]),
    )
    domain_one = model(
        x,
        base,
        torch.tensor([0]),
        route,
        query_start_abs_b=torch.tensor([3]),
    )
    assert domain_zero["patch_selected_risk_benefit_prob_bcq"].min() > 0.9
    assert domain_one["patch_selected_risk_benefit_prob_bcq"].max() < 0.1
    assert torch.equal(domain_zero["patch_skip_bcq"], torch.zeros(1, 1, 2))
    assert torch.equal(domain_one["patch_skip_bcq"], torch.ones(1, 1, 2))

    model.eval()
    averaged = model(x, base, torch.tensor([0]), route)
    assert torch.allclose(
        averaged["patch_selected_risk_benefit_prob_bcq"],
        torch.full((1, 1, 2), 0.5),
    )
    assert averaged["patch_selected_risk_domain_std_bcq"].min() > 0.45


def test_candidate_aware_risk_gate_receives_detached_expert_corrections() -> None:
    model = ClusterwisePredResidualMoE(
        num_clusters=1,
        num_penalties=2,
        input_len=4,
        pred_len=4,
        hidden_dim=3,
        intervention_enable=False,
        shared_across_clusters=True,
        patch_router_cfg={
            "enable": True,
            "patch_len": 2,
            "hidden_dim": 3,
            "allow_skip": True,
            "hierarchical_recall": {
                "enable": True,
                "expert_conditional_risk": {
                    "enable": True,
                    "candidate_aware": True,
                    "candidate_compatibility": True,
                    "decoupled_encoder": True,
                },
            },
        },
    )
    x = torch.randn(2, 1, 4)
    base = torch.zeros(2, 1, 4)
    out = model(x, base, torch.tensor([0]), torch.ones(2, 1, 2))
    assert out["patch_penalty_risk_benefit_probs_bcqp"].shape == (2, 1, 2, 2)
    assert out["patch_penalty_risk_positive_magnitude_bcqp"].shape == (2, 1, 2, 2)
    assert out["patch_penalty_risk_negative_magnitude_bcqp"].shape == (2, 1, 2, 2)
    assert torch.equal(
        out["patch_penalty_proposal_mask_bcqp"].sum(dim=-1),
        torch.full((2, 1, 2), 2),
    )
    router = model.patch_router
    assert router.W_proposal_candidate is not None
    candidate_delta = torch.randn(2, 1, 2, 4, requires_grad=True)
    router_out = router(
        x,
        y_base_bch=base,
        candidate_delta_bcpH=candidate_delta,
        straight_through=False,
    )
    router_out["patch_penalty_benefit_probs_bcqp"].sum().backward()
    assert router.W_proposal_candidate.grad is not None
    assert float(router.W_proposal_candidate.grad.abs().sum().item()) > 0.0
    assert candidate_delta.grad is None


def test_lower_quantile_risk_head_vetoes_positive_mean_utility() -> None:
    model = ClusterwisePredResidualMoE(
        num_clusters=1,
        num_penalties=2,
        input_len=4,
        pred_len=4,
        hidden_dim=3,
        intervention_enable=False,
        shared_across_clusters=True,
        patch_router_cfg={
            "enable": True,
            "patch_len": 2,
            "hidden_dim": 2,
            "topk": 1,
            "allow_skip": True,
            "hierarchical_recall": {
                "enable": True,
                "expert_conditional_risk": {
                    "enable": True,
                    "candidate_aware": False,
                    "decoupled_encoder": False,
                    "proposal_topk": 2,
                    "lower_quantile": {"enable": True, "quantile": 0.2},
                },
            },
        },
    )
    model.eval()
    assert model.patch_router is not None
    router = model.patch_router
    assert router.W_risk_sign is not None and router.b_risk_sign is not None
    assert router.W_risk_gain is not None and router.b_risk_gain is not None
    assert router.W_risk_cost is not None and router.b_risk_cost is not None
    assert router.W_risk_lower_quantile is not None
    assert router.b_risk_lower_quantile is not None
    with torch.no_grad():
        router.W1.zero_()
        router.b1.zero_()
        router.W_risk_sign.zero_()
        router.b_risk_sign.copy_(torch.tensor([4.0, -4.0]))
        router.W_risk_gain.zero_()
        router.b_risk_gain.fill_(-2.0)
        router.W_risk_cost.zero_()
        router.b_risk_cost.fill_(-2.0)
        router.W_risk_lower_quantile.zero_()
        router.b_risk_lower_quantile.fill_(-1.0)

    x = torch.randn(1, 1, 4)
    base = torch.zeros(1, 1, 4)
    route = torch.ones(1, 1, 2)
    rejected = model(x, base, torch.tensor([0]), route)
    assert torch.equal(rejected["patch_skip_bcq"], torch.ones(1, 1, 2))

    with torch.no_grad():
        router.b_risk_lower_quantile[0] = 1.0
    accepted = model(x, base, torch.tensor([0]), route)
    assert torch.equal(accepted["patch_skip_bcq"], torch.zeros(1, 1, 2))


def test_selected_benefit_probability_can_drive_final_adoption() -> None:
    model = ClusterwisePredResidualMoE(
        num_clusters=1,
        num_penalties=2,
        input_len=4,
        pred_len=4,
        hidden_dim=3,
        intervention_enable=False,
        shared_across_clusters=True,
        patch_router_cfg={
            "enable": True,
            "patch_len": 2,
            "hidden_dim": 2,
            "topk": 1,
            "allow_skip": True,
            "hierarchical_recall": {
                "enable": True,
                "expert_conditional_risk": {
                    "enable": True,
                    "candidate_aware": False,
                    "decoupled_encoder": False,
                    "proposal_topk": 2,
                    "adoption_source": "benefit_probability",
                    "adopt_threshold": 0.5,
                    "lower_quantile": {"enable": True, "quantile": 0.2},
                },
            },
        },
    )
    model.eval()
    assert model.patch_router is not None
    router = model.patch_router
    assert router.W_risk_sign is not None and router.b_risk_sign is not None
    assert router.W_risk_gain is not None and router.b_risk_gain is not None
    assert router.W_risk_cost is not None and router.b_risk_cost is not None
    assert router.W_risk_lower_quantile is not None
    assert router.b_risk_lower_quantile is not None
    with torch.no_grad():
        router.W1.zero_()
        router.b1.zero_()
        router.W_risk_sign.zero_()
        router.b_risk_sign.copy_(torch.tensor([4.0, -4.0]))
        router.W_risk_gain.zero_()
        router.b_risk_gain.fill_(-2.0)
        router.W_risk_cost.zero_()
        router.b_risk_cost.fill_(-2.0)
        router.W_risk_lower_quantile.zero_()
        router.b_risk_lower_quantile.fill_(-1.0)

    x = torch.randn(1, 1, 4)
    base = torch.zeros(1, 1, 4)
    route = torch.ones(1, 1, 2)
    accepted = model(x, base, torch.tensor([0]), route)
    assert torch.equal(accepted["patch_selected_penalty_index_bcq"], torch.zeros(1, 1, 2, dtype=torch.long))
    assert torch.equal(accepted["patch_skip_bcq"], torch.zeros(1, 1, 2))
    assert torch.all(accepted["patch_selected_risk_benefit_prob_bcq"] > 0.5)

    with torch.no_grad():
        router.b_risk_sign.fill_(-4.0)
    rejected = model(x, base, torch.tensor([0]), route)
    assert torch.equal(rejected["patch_skip_bcq"], torch.ones(1, 1, 2))
    assert torch.all(rejected["patch_selected_risk_benefit_prob_bcq"] < 0.5)


def test_selected_penalty_uses_its_own_risk_threshold() -> None:
    model = ClusterwisePredResidualMoE(
        num_clusters=1,
        num_penalties=2,
        input_len=4,
        pred_len=4,
        hidden_dim=3,
        intervention_enable=False,
        shared_across_clusters=True,
        patch_router_cfg={
            "enable": True,
            "patch_len": 2,
            "hidden_dim": 2,
            "topk": 1,
            "allow_skip": True,
            "hierarchical_recall": {
                "enable": True,
                "expert_conditional_risk": {
                    "enable": True,
                    "candidate_aware": False,
                    "decoupled_encoder": False,
                    "proposal_topk": 2,
                    "adoption_source": "benefit_probability",
                    "adopt_threshold": 0.5,
                    "temporal_calibration": {"enable": True, "per_penalty": True},
                },
            },
        },
    )
    model.eval()
    assert model.patch_router is not None
    router = model.patch_router
    assert router.W1 is not None and router.b1 is not None
    assert router.W_risk_sign is not None and router.b_risk_sign is not None
    assert router.W_risk_gain is not None and router.b_risk_gain is not None
    assert router.W_risk_cost is not None and router.b_risk_cost is not None
    router.set_expert_risk_adopt_threshold_by_penalty([0.99, 0.1])
    with torch.no_grad():
        router.W1.zero_()
        router.b1.zero_()
        router.W_risk_sign.zero_()
        router.W_risk_gain.zero_()
        router.b_risk_gain.fill_(-2.0)
        router.W_risk_cost.zero_()
        router.b_risk_cost.fill_(-2.0)
        router.b_risk_sign.copy_(torch.tensor([4.0, -4.0]))

    x = torch.randn(1, 1, 4)
    base = torch.zeros(1, 1, 4)
    route = torch.ones(1, 1, 2)
    rejected = model(x, base, torch.tensor([0]), route)
    assert torch.equal(
        rejected["patch_selected_penalty_index_bcq"],
        torch.zeros(1, 1, 2, dtype=torch.long),
    )
    assert torch.equal(rejected["patch_skip_bcq"], torch.ones(1, 1, 2))

    with torch.no_grad():
        router.b_risk_sign.copy_(torch.tensor([-4.0, 4.0]))
    accepted = model(x, base, torch.tensor([0]), route)
    assert torch.equal(
        accepted["patch_selected_penalty_index_bcq"],
        torch.ones(1, 1, 2, dtype=torch.long),
    )
    assert torch.equal(accepted["patch_skip_bcq"], torch.zeros(1, 1, 2))


def test_patch_router_can_fix_candidate_identity_and_keep_binary_adoption() -> None:
    model = ClusterwisePredResidualMoE(
        num_clusters=1,
        num_penalties=2,
        input_len=4,
        pred_len=4,
        hidden_dim=3,
        intervention_enable=False,
        shared_across_clusters=True,
        num_channels=2,
        patch_router_cfg={
            "enable": True,
            "patch_len": 2,
            "hidden_dim": 2,
            "topk": 1,
            "allow_skip": True,
            "fixed_penalty_index_by_channel": [1, -1],
            "candidate_scale_by_channel": [2.0, 0.0],
            "hierarchical_recall": {
                "enable": True,
                "expert_conditional_risk": {
                    "enable": True,
                    "candidate_aware": False,
                    "decoupled_encoder": False,
                    "proposal_topk": 2,
                    "adoption_source": "benefit_probability",
                    "adopt_threshold": 0.5,
                },
            },
        },
    )
    model.eval()
    assert model.patch_router is not None
    router = model.patch_router
    assert router.W1 is not None and router.b1 is not None
    assert router.W_risk_sign is not None and router.b_risk_sign is not None
    assert router.W_risk_gain is not None and router.b_risk_gain is not None
    assert router.W_risk_cost is not None and router.b_risk_cost is not None
    with torch.no_grad():
        router.W1.zero_()
        router.b1.zero_()
        router.W_risk_sign.zero_()
        router.b_risk_sign.fill_(4.0)
        router.W_risk_gain.zero_()
        router.b_risk_gain.fill_(-2.0)
        router.W_risk_cost.zero_()
        router.b_risk_cost.fill_(-2.0)

    result = model(
        torch.randn(1, 2, 4),
        torch.zeros(1, 2, 4),
        torch.tensor([0, 0]),
        torch.ones(1, 1, 2),
    )
    assert torch.equal(
        result["patch_selected_penalty_index_bcq"][:, 0],
        torch.ones(1, 2, dtype=torch.long),
    )
    assert torch.equal(result["patch_skip_bcq"][:, 0], torch.zeros(1, 2))
    assert torch.equal(result["patch_skip_bcq"][:, 1], torch.ones(1, 2))
    assert torch.equal(
        result["patch_fixed_penalty_active_bcq"],
        torch.tensor([[[True, True], [False, False]]]),
    )
    assert result["patch_candidate_scale_c"].tolist() == pytest.approx([2.0, 0.0])


def test_hierarchical_patch_loss_excludes_inactive_fixed_channels() -> None:
    base = torch.zeros(1, 2, 4)
    y = torch.tensor(
        [[[1.0, 1.0, -1.0, -1.0], [2.0, 2.0, 2.0, 2.0]]]
    )
    candidates = torch.tensor(
        [
            [
                [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, -1.0]],
                [[3.0, 3.0, 3.0, 3.0], [-3.0, -3.0, -3.0, -3.0]],
            ]
        ]
    )
    proposal_adopt = torch.tensor(
        [[[0.8, 0.8], [0.99, 0.99]]],
        requires_grad=True,
    )
    conditional = torch.tensor(
        [
            [
                [[0.9, 0.1], [0.1, 0.9]],
                [[0.99, 0.01], [0.99, 0.01]],
            ]
        ]
    )
    benefit = conditional.detach().clone().requires_grad_(True)
    utility = torch.zeros(1, 2, 2, 2, requires_grad=True)
    risk = torch.tensor(
        [
            [
                [[0.9, 0.1], [0.1, 0.9]],
                [[0.99, 0.99], [0.99, 0.99]],
            ]
        ],
        requires_grad=True,
    )
    risk_positive = torch.full_like(risk, 0.5, requires_grad=True)
    risk_negative = torch.full_like(risk, 0.5, requires_grad=True)
    proposal_logits = torch.tensor(
        [
            [
                [[3.0, -3.0], [-3.0, 3.0]],
                [[5.0, -5.0], [5.0, -5.0]],
            ]
        ],
        requires_grad=True,
    )
    final_adopt = torch.tensor(
        [[[0.9, 0.9], [0.99, 0.99]]],
        requires_grad=True,
    )
    active_mask = torch.tensor([[[True, True], [False, False]]])
    loss_weights = {
        "adoption_bce_weight": 1.0,
        "proposal_bce_weight": 1.0,
        "proposal_gain_listwise_weight": 1.0,
        "proposal_rescue_ce_weight": 0.0,
        "ranking_ce_weight": 1.0,
        "utility_regression_weight": 1.0,
        "adoption_recall_weight": 1.0,
        "false_adopt_weight": 1.0,
        "penalty_recall_weight": 1.0,
        "false_penalty_weight": 1.0,
        "risk_calibration_weight": 1.0,
        "risk_sign_bce_weight": 1.0,
        "risk_magnitude_weight": 1.0,
        "selected_utility_policy_weight": 1.0,
        "selected_adoption_bce_weight": 1.0,
        "selected_adoption_recall_weight": 1.0,
        "selected_false_adopt_weight": 1.0,
    }
    masked = _patch_router_hierarchical_recall_loss_terms(
        base_bch=base,
        candidate_bcpH=candidates,
        y_bch=y,
        patch_adopt_prob_bcq=proposal_adopt,
        patch_penalty_conditional_probs_bcqp=conditional,
        patch_penalty_benefit_probs_bcqp=benefit,
        patch_penalty_utility_scores_bcqp=utility,
        patch_penalty_risk_benefit_probs_bcqp=risk,
        patch_penalty_risk_positive_magnitude_bcqp=risk_positive,
        patch_penalty_risk_negative_magnitude_bcqp=risk_negative,
        patch_penalty_proposal_logits_bcqp=proposal_logits,
        patch_final_adopt_prob_bcq=final_adopt,
        patch_active_mask_bcq=active_mask,
        cluster_id_c=torch.tensor([0, 0]),
        K=1,
        **loss_weights,
    )
    reference = _patch_router_hierarchical_recall_loss_terms(
        base_bch=base[:, :1],
        candidate_bcpH=candidates[:, :1],
        y_bch=y[:, :1],
        patch_adopt_prob_bcq=proposal_adopt[:, :1].detach(),
        patch_penalty_conditional_probs_bcqp=conditional[:, :1],
        patch_penalty_benefit_probs_bcqp=benefit[:, :1].detach(),
        patch_penalty_utility_scores_bcqp=utility[:, :1].detach(),
        patch_penalty_risk_benefit_probs_bcqp=risk[:, :1].detach(),
        patch_penalty_risk_positive_magnitude_bcqp=risk_positive[:, :1].detach(),
        patch_penalty_risk_negative_magnitude_bcqp=risk_negative[:, :1].detach(),
        patch_penalty_proposal_logits_bcqp=proposal_logits[:, :1].detach(),
        patch_final_adopt_prob_bcq=final_adopt[:, :1].detach(),
        cluster_id_c=torch.tensor([0]),
        K=1,
        **loss_weights,
    )

    for key in masked:
        assert torch.allclose(masked[key], reference[key], atol=1.0e-6)

    masked["total_bk"].sum().backward()
    for value in (
        proposal_adopt,
        benefit,
        utility,
        risk,
        risk_positive,
        risk_negative,
        proposal_logits,
        final_adopt,
    ):
        assert value.grad is not None
        assert bool((value.grad[:, 0].abs() > 0.0).any().item())
        assert torch.equal(value.grad[:, 1], torch.zeros_like(value.grad[:, 1]))


def test_candidate_predictions_apply_patch_candidate_channel_scale() -> None:
    base = torch.zeros(1, 2, 4)
    pred_out = {
        "residuals": torch.ones(1, 2, 2, 4),
        "alpha_cp": torch.ones(2, 2),
        "intervention_bcp": torch.ones(1, 2, 2),
        "selector_bcp": torch.ones(1, 2, 2),
        "confidence_active_bcp": torch.ones(1, 2, 2),
        "patch_candidate_scale_c": torch.tensor([2.0, 0.5]),
    }
    candidates = _pred_residual_candidate_predictions(
        base,
        pred_out,
        include_patch_route=False,
    )
    assert candidates is not None
    assert torch.allclose(candidates[:, 0], torch.full((1, 2, 4), 2.0))
    assert torch.allclose(candidates[:, 1], torch.full((1, 2, 4), 0.5))


def test_independent_utility_veto_can_reject_a_recalled_candidate() -> None:
    model = ClusterwisePredResidualMoE(
        num_clusters=1,
        num_penalties=2,
        input_len=4,
        pred_len=4,
        hidden_dim=3,
        intervention_enable=False,
        shared_across_clusters=True,
        patch_router_cfg={
            "enable": True,
            "patch_len": 2,
            "hidden_dim": 2,
            "topk": 1,
            "allow_skip": True,
            "hierarchical_recall": {
                "enable": True,
                "expert_conditional_risk": {
                    "enable": True,
                    "candidate_aware": False,
                    "decoupled_encoder": False,
                    "proposal_topk": 2,
                    "adoption_source": "utility_veto",
                    "adopt_threshold": 0.5,
                    "utility_veto": {"enable": True, "detach_features": True},
                },
            },
        },
    )
    model.eval()
    assert model.patch_router is not None
    router = model.patch_router
    assert router.W_risk_sign is not None and router.b_risk_sign is not None
    assert router.W_risk_gain is not None and router.b_risk_gain is not None
    assert router.W_risk_cost is not None and router.b_risk_cost is not None
    assert router.W_risk_utility_veto is not None
    assert router.b_risk_utility_veto is not None
    with torch.no_grad():
        router.W1.zero_()
        router.b1.zero_()
        router.W_risk_sign.zero_()
        router.b_risk_sign.copy_(torch.tensor([4.0, -4.0]))
        router.W_risk_gain.zero_()
        router.b_risk_gain.fill_(-2.0)
        router.W_risk_cost.zero_()
        router.b_risk_cost.fill_(-2.0)
        router.W_risk_utility_veto.zero_()
        router.b_risk_utility_veto.fill_(-4.0)

    x = torch.randn(1, 1, 4)
    base = torch.zeros(1, 1, 4)
    route = torch.ones(1, 1, 2)
    rejected = model(x, base, torch.tensor([0]), route)
    assert torch.equal(
        rejected["patch_selected_penalty_index_bcq"],
        torch.zeros(1, 1, 2, dtype=torch.long),
    )
    assert torch.all(rejected["patch_selected_risk_benefit_prob_bcq"] > 0.5)
    assert torch.all(rejected["patch_selected_utility_veto_prob_bcq"] < 0.5)
    assert torch.equal(rejected["patch_skip_bcq"], torch.ones(1, 1, 2))

    with torch.no_grad():
        router.b_risk_utility_veto[0] = 4.0
    accepted = model(x, base, torch.tensor([0]), route)
    assert torch.equal(accepted["patch_skip_bcq"], torch.zeros(1, 1, 2))


def test_utility_veto_adoption_source_requires_the_veto_head() -> None:
    with pytest.raises(ValueError, match="utility_veto.enable"):
        ClusterwisePredResidualMoE(
            num_clusters=1,
            num_penalties=2,
            input_len=4,
            pred_len=4,
            hidden_dim=3,
            intervention_enable=False,
            shared_across_clusters=True,
            patch_router_cfg={
                "enable": True,
                "patch_len": 2,
                "hidden_dim": 2,
                "allow_skip": True,
                "hierarchical_recall": {
                    "enable": True,
                    "expert_conditional_risk": {
                        "enable": True,
                        "candidate_aware": False,
                        "adoption_source": "utility_veto",
                    },
                },
            },
        )


def test_selected_utility_policy_gradient_reaches_only_detached_veto_head() -> None:
    model = ClusterwisePredResidualMoE(
        num_clusters=1,
        num_penalties=2,
        input_len=4,
        pred_len=4,
        hidden_dim=3,
        intervention_enable=False,
        shared_across_clusters=True,
        patch_router_cfg={
            "enable": True,
            "patch_len": 2,
            "hidden_dim": 4,
            "topk": 1,
            "allow_skip": True,
            "hierarchical_recall": {
                "enable": True,
                "expert_conditional_risk": {
                    "enable": True,
                    "candidate_aware": False,
                    "proposal_topk": 2,
                    "adoption_source": "utility_veto",
                    "utility_veto": {"enable": True, "detach_features": True},
                },
            },
        },
    )
    model.train()
    assert model.patch_router is not None
    router = model.patch_router
    assert router.W_risk_utility_veto is not None
    assert router.W_risk_sign is not None
    x = torch.randn(2, 1, 4)
    base = torch.zeros(2, 1, 4)
    out = model(x, base, torch.tensor([0]), torch.ones(2, 1, 2))
    y = torch.tensor(
        [
            [[1.0, 1.0, -1.0, -1.0]],
            [[-1.0, -1.0, 1.0, 1.0]],
        ]
    )
    candidates = torch.tensor(
        [
            [[[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, -1.0, -1.0]]],
            [[[-1.0, -1.0, -1.0, -1.0], [0.0, 0.0, 1.0, 1.0]]],
        ]
    )
    terms = _patch_router_hierarchical_recall_loss_terms(
        base_bch=base,
        candidate_bcpH=candidates,
        y_bch=y,
        patch_adopt_prob_bcq=out["patch_proposal_adopt_prob_bcq"],
        patch_penalty_conditional_probs_bcqp=out[
            "patch_penalty_conditional_probs_bcqp"
        ],
        patch_penalty_benefit_probs_bcqp=out["patch_penalty_benefit_probs_bcqp"],
        patch_penalty_utility_scores_bcqp=out["patch_penalty_utility_scores_bcqp"],
        patch_final_adopt_prob_bcq=out["patch_adopt_prob_bcq"],
        cluster_id_c=torch.tensor([0]),
        K=1,
        adoption_bce_weight=0.0,
        proposal_bce_weight=0.0,
        ranking_ce_weight=0.0,
        utility_regression_weight=0.0,
        selected_utility_policy_weight=1.0,
        adoption_recall_weight=0.0,
        false_adopt_weight=0.0,
        penalty_recall_weight=0.0,
        false_penalty_weight=0.0,
    )
    terms["total_bk"].mean().backward()

    assert router.W_risk_utility_veto.grad is not None
    assert float(router.W_risk_utility_veto.grad.abs().sum().item()) > 0.0
    assert router.W_risk_sign.grad is not None
    assert float(router.W_risk_sign.grad.abs().sum().item()) == 0.0


def test_proposal_rescue_head_supplies_a_distinct_second_candidate() -> None:
    model = ClusterwisePredResidualMoE(
        num_clusters=1,
        num_penalties=3,
        input_len=4,
        pred_len=4,
        hidden_dim=3,
        intervention_enable=False,
        shared_across_clusters=True,
        patch_router_cfg={
            "enable": True,
            "patch_len": 2,
            "hidden_dim": 2,
            "topk": 1,
            "allow_skip": True,
            "hierarchical_recall": {
                "enable": True,
                "expert_conditional_risk": {
                    "enable": True,
                    "candidate_aware": False,
                    "decoupled_encoder": False,
                    "proposal_topk": 2,
                    "proposal_rescue": True,
                },
            },
        },
    )
    model.eval()
    assert model.patch_router is not None
    router = model.patch_router
    assert router.W_benefit is not None and router.b_benefit is not None
    assert router.W_proposal_rescue is not None and router.b_proposal_rescue is not None
    with torch.no_grad():
        router.W1.zero_()
        router.b1.zero_()
        router.W_benefit.zero_()
        router.b_benefit.copy_(torch.tensor([4.0, 0.0, -4.0]))
        router.W_proposal_rescue.zero_()
        router.b_proposal_rescue.copy_(torch.tensor([0.0, 4.0, -4.0]))

    out = model(
        torch.randn(1, 1, 4),
        torch.zeros(1, 1, 4),
        torch.tensor([0]),
        torch.ones(1, 1, 3),
    )
    expected = torch.tensor([[[[True, True, False], [True, True, False]]]])
    assert torch.equal(out["patch_penalty_proposal_mask_bcqp"], expected)


def test_pairwise_rank_head_selects_within_the_proposal_shortlist() -> None:
    model = ClusterwisePredResidualMoE(
        num_clusters=1,
        num_penalties=2,
        input_len=4,
        pred_len=4,
        hidden_dim=3,
        intervention_enable=False,
        shared_across_clusters=True,
        patch_router_cfg={
            "enable": True,
            "patch_len": 2,
            "hidden_dim": 2,
            "topk": 1,
            "allow_skip": True,
            "hierarchical_recall": {
                "enable": True,
                "expert_conditional_risk": {
                    "enable": True,
                    "candidate_aware": False,
                    "decoupled_encoder": False,
                    "proposal_topk": 2,
                    "pairwise_rank": {"enable": True, "detach_features": True},
                },
            },
        },
    )
    model.eval()
    assert model.patch_router is not None
    router = model.patch_router
    assert router.W_pairwise_rank is not None and router.b_pairwise_rank is not None
    with torch.no_grad():
        router.W_pairwise_rank.zero_()
        router.b_pairwise_rank.copy_(torch.tensor([-1.0, 1.0]))
    out = model(
        torch.randn(1, 1, 4),
        torch.zeros(1, 1, 4),
        torch.tensor([0]),
        torch.ones(1, 1, 2),
    )
    assert torch.equal(
        out["patch_selected_penalty_index_bcq"],
        torch.ones(1, 1, 2, dtype=torch.long),
    )
    frozen = _freeze_module_params_except_prefixes(
        model,
        ("patch_router.W_pairwise_rank", "patch_router.b_pairwise_rank"),
    )
    assert frozen > 0
    trainable_names = {
        name for name, param in model.named_parameters() if param.requires_grad
    }
    assert trainable_names == {
        "patch_router.W_pairwise_rank",
        "patch_router.b_pairwise_rank",
    }


def test_patch_router_expected_mse_supervision_rewards_patch_specific_candidates() -> None:
    y = torch.tensor([[[1.0, 1.0, -1.0, -1.0]]])
    base = torch.zeros_like(y)
    candidates = torch.tensor(
        [[[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, -1.0]]]]
    )
    correct_probs = torch.tensor([[[[0.9, 0.0], [0.0, 0.9]]]], requires_grad=True)
    swapped_probs = torch.tensor([[[[0.0, 0.9], [0.9, 0.0]]]])
    skip_prob = torch.full((1, 1, 2), 0.1)

    correct = _patch_router_expected_mse_loss_bk(
        base_bch=base,
        candidate_bcpH=candidates,
        y_bch=y,
        patch_probs_bcqp=correct_probs,
        patch_skip_prob_bcq=skip_prob,
        cluster_id_c=torch.tensor([0]),
        K=1,
    )
    swapped = _patch_router_expected_mse_loss_bk(
        base_bch=base,
        candidate_bcpH=candidates,
        y_bch=y,
        patch_probs_bcqp=swapped_probs,
        patch_skip_prob_bcq=skip_prob,
        cluster_id_c=torch.tensor([0]),
        K=1,
    )

    assert correct.shape == (1, 1)
    assert correct.item() < swapped.item()
    correct.mean().backward()
    assert correct_probs.grad is not None
    assert correct_probs.grad[0, 0, 0, 0] < correct_probs.grad[0, 0, 0, 1]
    assert correct_probs.grad[0, 0, 1, 1] < correct_probs.grad[0, 0, 1, 0]


def test_patch_router_expected_mse_supervision_can_include_signed_mae_utility() -> None:
    y = torch.zeros(1, 1, 2)
    base = torch.ones_like(y)
    candidates = torch.tensor(
        [[[[1.0, 0.0], [2.0**-0.5, 2.0**-0.5]]]]
    )
    choose_sparse = torch.tensor([[[[1.0, 0.0]]]])
    choose_dense = torch.tensor([[[[0.0, 1.0]]]])
    no_skip = torch.zeros(1, 1, 1)

    mse_sparse = _patch_router_expected_mse_loss_bk(
        base_bch=base,
        candidate_bcpH=candidates,
        y_bch=y,
        patch_probs_bcqp=choose_sparse,
        patch_skip_prob_bcq=no_skip,
        cluster_id_c=torch.tensor([0]),
        K=1,
    )
    mse_dense = _patch_router_expected_mse_loss_bk(
        base_bch=base,
        candidate_bcpH=candidates,
        y_bch=y,
        patch_probs_bcqp=choose_dense,
        patch_skip_prob_bcq=no_skip,
        cluster_id_c=torch.tensor([0]),
        K=1,
    )
    dual_sparse = _patch_router_expected_mse_loss_bk(
        base_bch=base,
        candidate_bcpH=candidates,
        y_bch=y,
        patch_probs_bcqp=choose_sparse,
        patch_skip_prob_bcq=no_skip,
        cluster_id_c=torch.tensor([0]),
        K=1,
        mae_weight=1.0,
    )
    dual_dense = _patch_router_expected_mse_loss_bk(
        base_bch=base,
        candidate_bcpH=candidates,
        y_bch=y,
        patch_probs_bcqp=choose_dense,
        patch_skip_prob_bcq=no_skip,
        cluster_id_c=torch.tensor([0]),
        K=1,
        mae_weight=1.0,
    )

    assert mse_sparse.item() == pytest.approx(mse_dense.item(), abs=1.0e-6)
    assert dual_sparse.item() < dual_dense.item()


def test_dual_signed_utility_router_starts_as_exact_skip_and_uses_zero_boundary() -> None:
    model = ClusterwisePredResidualMoE(
        num_clusters=1,
        num_penalties=2,
        input_len=4,
        pred_len=4,
        hidden_dim=3,
        intervention_enable=False,
        shared_across_clusters=True,
        patch_router_cfg={
            "enable": True,
            "patch_len": 2,
            "hidden_dim": 3,
            "topk": 1,
            "allow_skip": True,
            "use_base_forecast": True,
            "hierarchical_recall": {
                "enable": True,
                "expert_conditional_risk": {
                    "enable": True,
                    "candidate_aware": True,
                    "adoption_source": "expected_utility",
                    "adopt_threshold": 0.0,
                    "dual_signed_utility": {"enable": True},
                },
            },
        },
    )
    model.eval()
    assert model.patch_router is not None
    router = model.patch_router
    x = torch.randn(1, 1, 4)
    base = torch.zeros(1, 1, 4)
    candidates = torch.zeros(1, 1, 2, 4)

    initial = router(
        x,
        y_base_bch=base,
        candidate_delta_bcpH=candidates,
        straight_through=False,
    )
    assert torch.equal(initial["patch_skip_bcq"], torch.ones(1, 1, 2))
    assert torch.equal(initial["patch_route_bcph"], torch.zeros(1, 1, 2, 4))
    assert torch.equal(
        initial["patch_penalty_mse_utility_scores_bcqp"],
        torch.zeros(1, 1, 2, 2),
    )
    assert torch.equal(
        initial["patch_penalty_mae_utility_scores_bcqp"],
        torch.zeros(1, 1, 2, 2),
    )

    assert router.b_risk_mse_utility is not None
    assert router.b_risk_mae_utility is not None
    with torch.no_grad():
        router.b_risk_mse_utility.copy_(torch.tensor([2.0, 3.0]))
        router.b_risk_mae_utility.copy_(torch.tensor([2.0, -1.0]))
    accepted = router(
        x,
        y_base_bch=base,
        candidate_delta_bcpH=candidates,
        straight_through=False,
    )
    assert torch.equal(accepted["patch_skip_bcq"], torch.zeros(1, 1, 2))
    assert torch.equal(
        accepted["patch_selected_penalty_index_bcq"],
        torch.zeros(1, 1, 2, dtype=torch.long),
    )

    with torch.no_grad():
        router.b_risk_mse_utility.fill_(1.0)
        router.b_risk_mae_utility.zero_()
    zero_boundary = router(
        x,
        y_base_bch=base,
        candidate_delta_bcpH=candidates,
        straight_through=False,
    )
    assert torch.equal(zero_boundary["patch_skip_bcq"], torch.ones(1, 1, 2))

    with torch.no_grad():
        router.b_risk_mse_utility.fill_(-1.0)
        router.b_risk_mae_utility.fill_(1.0)
    rejected = router(
        x,
        y_base_bch=base,
        candidate_delta_bcpH=candidates,
        straight_through=False,
    )
    assert torch.equal(rejected["patch_skip_bcq"], torch.ones(1, 1, 2))


def test_patch_router_soft_inference_uses_input_conditioned_probability_mass() -> None:
    router = ChannelPatchPenaltyRouter(
        input_len=4,
        pred_len=4,
        num_penalties=2,
        cfg={
            "patch_len": 2,
            "hidden_dim": 3,
            "allow_skip": True,
            "skip_init_bias": 0.0,
            "inference_route_mode": "soft",
        },
    )
    router.eval()
    out = router(torch.randn(1, 1, 4), straight_through=False)

    probs = out["patch_probs_bcqp"]
    skip_prob = out["patch_skip_prob_bcq"]
    route = out["patch_route_bcph"].reshape(1, 1, 2, 2, 2)[..., 0]

    assert torch.allclose(route, probs.permute(0, 1, 3, 2))
    assert torch.allclose(out["patch_skip_bcq"], skip_prob)
    assert torch.allclose(probs.sum(dim=-1) + skip_prob, torch.ones_like(skip_prob))


def test_patch_router_mixture_loss_matches_forecast_and_trains_probabilities() -> None:
    base = torch.zeros(1, 1, 4)
    candidates = torch.tensor([[[[2.0, 2.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0]]]])
    y = torch.ones(1, 1, 4)
    probs = torch.full((1, 1, 2, 2), 0.25, requires_grad=True)

    loss = _patch_router_mixture_mse_loss_bk(
        base_bch=base,
        candidate_bcpH=candidates,
        y_bch=y,
        patch_probs_bcqp=probs,
        cluster_id_c=torch.tensor([0]),
        K=1,
    )
    loss.sum().backward()

    assert loss.item() == pytest.approx(0.0625)
    assert probs.grad is not None
    assert torch.count_nonzero(probs.grad).item() == probs.numel()


def test_patch_router_rejects_unknown_inference_route_mode() -> None:
    with pytest.raises(ValueError, match="inference_route_mode"):
        ChannelPatchPenaltyRouter(
            input_len=4,
            pred_len=4,
            num_penalties=2,
            cfg={"patch_len": 2, "inference_route_mode": "rank"},
        )


def test_dual_signed_utility_loss_learns_mse_and_mae_directions_separately() -> None:
    y = torch.zeros(1, 1, 2)
    base = torch.tensor([[[2.0, 0.0]]])
    candidates = torch.tensor([[[[1.2, 1.2]]]])
    adopt = torch.full((1, 1, 1), 0.5)
    conditional = torch.ones(1, 1, 1, 1)
    benefit = torch.full((1, 1, 1, 1), 0.5)
    good_mse = torch.full((1, 1, 1, 1), 0.25, requires_grad=True)
    good_mae = torch.full((1, 1, 1, 1), -0.2, requires_grad=True)
    bad_mse = -good_mse
    bad_mae = -good_mae

    good = _patch_router_hierarchical_recall_loss_terms(
        base_bch=base,
        candidate_bcpH=candidates,
        y_bch=y,
        patch_adopt_prob_bcq=adopt,
        patch_penalty_conditional_probs_bcqp=conditional,
        patch_penalty_benefit_probs_bcqp=benefit,
        patch_penalty_utility_scores_bcqp=torch.minimum(good_mse, good_mae),
        patch_penalty_mse_utility_scores_bcqp=good_mse,
        patch_penalty_mae_utility_scores_bcqp=good_mae,
        cluster_id_c=torch.tensor([0]),
        K=1,
    )
    bad = _patch_router_hierarchical_recall_loss_terms(
        base_bch=base,
        candidate_bcpH=candidates,
        y_bch=y,
        patch_adopt_prob_bcq=adopt,
        patch_penalty_conditional_probs_bcqp=conditional,
        patch_penalty_benefit_probs_bcqp=benefit,
        patch_penalty_utility_scores_bcqp=torch.minimum(bad_mse, bad_mae),
        patch_penalty_mse_utility_scores_bcqp=bad_mse,
        patch_penalty_mae_utility_scores_bcqp=bad_mae,
        cluster_id_c=torch.tensor([0]),
        K=1,
    )

    assert good["mse_utility_regression_bk"].item() < bad[
        "mse_utility_regression_bk"
    ].item()
    assert good["mae_utility_regression_bk"].item() < bad[
        "mae_utility_regression_bk"
    ].item()
    assert good["utility_regression_bk"].item() < bad[
        "utility_regression_bk"
    ].item()
    good["utility_regression_bk"].sum().backward()
    assert good_mse.grad is not None
    assert good_mae.grad is not None
    assert float(good_mse.grad.abs().sum().item()) > 0.0
    assert float(good_mae.grad.abs().sum().item()) > 0.0


def test_dual_signed_utility_configuration_preserves_default_output_contract() -> None:
    common = {
        "enable": True,
        "patch_len": 2,
        "hidden_dim": 3,
        "allow_skip": True,
        "hierarchical_recall": {
            "enable": True,
            "expert_conditional_risk": {
                "enable": True,
                "candidate_aware": True,
            },
        },
    }
    model = ClusterwisePredResidualMoE(
        num_clusters=1,
        num_penalties=2,
        input_len=4,
        pred_len=4,
        hidden_dim=3,
        intervention_enable=False,
        shared_across_clusters=True,
        patch_router_cfg=common,
    )
    assert model.patch_router is not None
    out = model.patch_router(
        torch.randn(1, 1, 4),
        y_base_bch=torch.zeros(1, 1, 4),
        candidate_delta_bcpH=torch.zeros(1, 1, 2, 4),
        straight_through=False,
    )
    assert "patch_penalty_mse_utility_scores_bcqp" not in out
    assert "patch_penalty_mae_utility_scores_bcqp" not in out

    invalid = dict(common)
    invalid["hierarchical_recall"] = {
        "enable": True,
        "expert_conditional_risk": {
            "enable": True,
            "candidate_aware": True,
            "adoption_source": "expected_utility",
            "adopt_threshold": 0.1,
            "dual_signed_utility": {"enable": True},
        },
    }
    with pytest.raises(ValueError, match="adopt_threshold must be 0"):
        ClusterwisePredResidualMoE(
            num_clusters=1,
            num_penalties=2,
            input_len=4,
            pred_len=4,
            hidden_dim=3,
            intervention_enable=False,
            shared_across_clusters=True,
            patch_router_cfg=invalid,
        )


def test_patch_router_oracle_stats_use_patch_level_skip_and_penalty_labels() -> None:
    y = torch.tensor([[[1.0, 1.0, -1.0, -1.0]]])
    base = torch.zeros_like(y)
    candidates = torch.tensor(
        [[[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, -1.0]]]]
    )
    patch_route = torch.tensor([[[[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]]])
    stats = _patch_router_oracle_batch_stats(
        base_bch=base,
        candidate_bcpH=candidates,
        y_bch=y,
        patch_route_bcph=patch_route,
        patch_skip_bcq=torch.zeros(1, 1, 2),
        patch_penalty_benefit_probs_bcqp=torch.tensor(
            [[[[0.9, 0.1], [0.9, 0.9]]]]
        ),
        patch_penalty_proposal_mask_bcqp=torch.tensor(
            [[[[True, False], [True, True]]]]
        ),
        patch_selected_penalty_index_bcq=torch.zeros(1, 1, 2, dtype=torch.long),
    )

    assert stats["count"].item() == 2
    assert torch.equal(stats["oracle_class_count"], torch.tensor([0.0, 1.0, 1.0]))
    assert torch.equal(stats["selected_class_count"], torch.tensor([0.0, 2.0, 0.0]))
    assert torch.equal(
        stats["confusion_matrix"],
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
    )
    assert stats["correct_count"].item() == 1
    assert stats["oracle_penalty_count"].item() == 2
    assert stats["selected_penalty_count"].item() == 2
    assert stats["adoption_true_positive_count"].item() == 2
    assert stats["selected_beneficial_count"].item() == 1
    assert stats["selected_harmful_count"].item() == 1
    assert stats["dual_oracle_penalty_count"].item() == 2
    assert stats["selected_dual_beneficial_count"].item() == 1
    assert stats["selected_dual_harmful_count"].item() == 1
    assert stats["selected_positive_gain_sum"].item() == pytest.approx(1.0)
    assert stats["selected_negative_cost_sum"].item() == pytest.approx(0.0)
    assert torch.equal(stats["selected_beneficial_count_by_penalty"], torch.tensor([1.0, 0.0]))
    assert torch.equal(stats["selected_count_by_penalty"], torch.tensor([2.0, 0.0]))
    assert torch.equal(stats["selected_gain_sum_by_penalty"], torch.tensor([1.0, 0.0]))
    assert stats["oracle_penalty_hit_count"].item() == 1
    assert torch.equal(stats["beneficial_penalty_count"], torch.tensor([1.0, 1.0]))
    assert torch.equal(stats["proposed_penalty_count"], torch.tensor([2.0, 1.0]))
    assert torch.equal(stats["proposal_true_positive_count"], torch.tensor([1.0, 1.0]))
    assert stats["proposal_oracle_hit_count"].item() == 2
    assert stats["shortlist_pairwise_count"].item() == 1
    assert stats["shortlist_pairwise_correct_count"].item() == 0
    assert torch.equal(stats["proposal_oracle_hit_count_by_penalty"], torch.tensor([1.0, 1.0]))
    assert stats["beneficial_cardinality_sum"].item() == 2
    assert torch.equal(
        stats["beneficial_cardinality_histogram"],
        torch.tensor([0.0, 2.0, 0.0]),
    )
    assert stats["base_error_sum"].item() == pytest.approx(2.0)
    assert stats["oracle_error_sum"].item() == pytest.approx(0.0)
    assert stats["selected_error_sum"].item() == pytest.approx(1.0)
    assert stats["base_mae_sum"].item() == pytest.approx(2.0)
    assert stats["oracle_mae_sum"].item() == pytest.approx(0.0)
    assert stats["selected_mae_sum"].item() == pytest.approx(1.0)


def test_hierarchical_patch_recall_loss_rewards_recall_and_rejects_false_adoption() -> None:
    y = torch.tensor([[[1.0, 1.0, -1.0, -1.0]]])
    base = torch.zeros_like(y)
    candidates = torch.tensor(
        [[[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, -1.0]]]]
    )
    conditional = torch.tensor([[[[0.9, 0.1], [0.1, 0.9]]]])
    good = _patch_router_hierarchical_recall_loss_terms(
        base_bch=base,
        candidate_bcpH=candidates,
        y_bch=y,
        patch_adopt_prob_bcq=torch.full((1, 1, 2), 0.9),
        patch_penalty_conditional_probs_bcqp=conditional,
        patch_penalty_benefit_probs_bcqp=conditional,
        patch_penalty_utility_scores_bcqp=torch.tensor([[[[0.8, -0.2], [-0.2, 0.8]]]]),
        cluster_id_c=torch.tensor([0]),
        K=1,
    )
    missed = _patch_router_hierarchical_recall_loss_terms(
        base_bch=base,
        candidate_bcpH=candidates,
        y_bch=y,
        patch_adopt_prob_bcq=torch.full((1, 1, 2), 0.1),
        patch_penalty_conditional_probs_bcqp=conditional,
        patch_penalty_benefit_probs_bcqp=torch.full((1, 1, 2, 2), 0.1),
        patch_penalty_utility_scores_bcqp=torch.zeros(1, 1, 2, 2),
        cluster_id_c=torch.tensor([0]),
        K=1,
    )
    assert good["total_bk"].item() < missed["total_bk"].item()

    utility_common = {
        "base_bch": base,
        "candidate_bcpH": candidates,
        "y_bch": y,
        "patch_adopt_prob_bcq": torch.full((1, 1, 2), 0.9),
        "patch_penalty_conditional_probs_bcqp": conditional,
        "patch_penalty_benefit_probs_bcqp": conditional,
        "cluster_id_c": torch.tensor([0]),
        "K": 1,
        "adoption_bce_weight": 0.0,
        "proposal_bce_weight": 0.0,
        "ranking_ce_weight": 0.0,
        "utility_regression_weight": 1.0,
        "adoption_recall_weight": 0.0,
        "false_adopt_weight": 0.0,
        "penalty_recall_weight": 0.0,
        "false_penalty_weight": 0.0,
    }
    utility_good = _patch_router_hierarchical_recall_loss_terms(
        **utility_common,
        patch_penalty_utility_scores_bcqp=torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
    )
    utility_bad = _patch_router_hierarchical_recall_loss_terms(
        **utility_common,
        patch_penalty_utility_scores_bcqp=torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]]),
    )
    assert utility_good["total_bk"].item() < utility_bad["total_bk"].item()

    proposal_common = {
        **utility_common,
        "utility_regression_weight": 0.0,
        "proposal_gain_listwise_weight": 1.0,
        "patch_penalty_utility_scores_bcqp": torch.zeros(1, 1, 2, 2),
    }
    proposal_common.pop("patch_penalty_benefit_probs_bcqp")
    proposal_good = _patch_router_hierarchical_recall_loss_terms(
        **proposal_common,
        patch_penalty_benefit_probs_bcqp=conditional,
    )
    proposal_bad = _patch_router_hierarchical_recall_loss_terms(
        **proposal_common,
        patch_penalty_benefit_probs_bcqp=1.0 - conditional,
    )
    assert proposal_good["total_bk"].item() < proposal_bad["total_bk"].item()
    extreme_proposal_logits = torch.tensor(
        [[[[1000.0, -1000.0], [-1000.0, 1000.0]]]],
        requires_grad=True,
    )
    proposal_extreme = _patch_router_hierarchical_recall_loss_terms(
        **proposal_common,
        patch_penalty_benefit_probs_bcqp=conditional,
        patch_penalty_proposal_logits_bcqp=extreme_proposal_logits,
    )
    assert torch.isfinite(proposal_extreme["total_bk"]).all()
    proposal_extreme["total_bk"].mean().backward()
    assert extreme_proposal_logits.grad is not None
    assert torch.isfinite(extreme_proposal_logits.grad).all()

    risk_common = {
        **utility_common,
        "patch_penalty_utility_scores_bcqp": torch.zeros(1, 1, 2, 2),
        "utility_regression_weight": 0.0,
        "risk_calibration_weight": 1.0,
        "risk_magnitude_weight": 1.0,
    }
    risk_good = _patch_router_hierarchical_recall_loss_terms(
        **risk_common,
        patch_penalty_risk_benefit_probs_bcqp=conditional,
        patch_penalty_risk_positive_magnitude_bcqp=torch.tensor(
            [[[[0.95, 0.0], [0.0, 0.95]]]]
        ),
        patch_penalty_risk_negative_magnitude_bcqp=torch.zeros(1, 1, 2, 2),
    )
    risk_bad = _patch_router_hierarchical_recall_loss_terms(
        **risk_common,
        patch_penalty_risk_benefit_probs_bcqp=1.0 - conditional,
        patch_penalty_risk_positive_magnitude_bcqp=torch.tensor(
            [[[[0.0, 0.95], [0.95, 0.0]]]]
        ),
        patch_penalty_risk_negative_magnitude_bcqp=torch.full((1, 1, 2, 2), 0.5),
    )
    assert risk_good["total_bk"].item() < risk_bad["total_bk"].item()

    quantile_common = {
        **utility_common,
        "utility_regression_weight": 0.0,
        "risk_lower_quantile_weight": 1.0,
        "risk_lower_quantile": 0.2,
        "patch_penalty_utility_scores_bcqp": torch.zeros(1, 1, 2, 2),
    }
    quantile_good = _patch_router_hierarchical_recall_loss_terms(
        **quantile_common,
        patch_penalty_risk_lower_quantile_scores_bcqp=torch.tensor(
            [[[[0.95, 0.0], [0.0, 0.95]]]]
        ),
    )
    quantile_bad = _patch_router_hierarchical_recall_loss_terms(
        **quantile_common,
        patch_penalty_risk_lower_quantile_scores_bcqp=torch.tensor(
            [[[[0.0, 0.95], [0.95, 0.0]]]]
        ),
    )
    assert quantile_good["total_bk"].item() < quantile_bad["total_bk"].item()

    no_benefit_candidates = base.unsqueeze(2).expand(-1, -1, 2, -1)
    safe_skip = _patch_router_hierarchical_recall_loss_terms(
        base_bch=base,
        candidate_bcpH=no_benefit_candidates,
        y_bch=y,
        patch_adopt_prob_bcq=torch.full((1, 1, 2), 0.1),
        patch_penalty_conditional_probs_bcqp=torch.full((1, 1, 2, 2), 0.5),
        patch_penalty_benefit_probs_bcqp=torch.full((1, 1, 2, 2), 0.1),
        patch_penalty_utility_scores_bcqp=torch.zeros(1, 1, 2, 2),
        cluster_id_c=torch.tensor([0]),
        K=1,
    )
    false_adopt = _patch_router_hierarchical_recall_loss_terms(
        base_bch=base,
        candidate_bcpH=no_benefit_candidates,
        y_bch=y,
        patch_adopt_prob_bcq=torch.full((1, 1, 2), 0.9),
        patch_penalty_conditional_probs_bcqp=torch.full((1, 1, 2, 2), 0.5),
        patch_penalty_benefit_probs_bcqp=torch.full((1, 1, 2, 2), 0.9),
        patch_penalty_utility_scores_bcqp=torch.zeros(1, 1, 2, 2),
        cluster_id_c=torch.tensor([0]),
        K=1,
    )
    assert safe_skip["total_bk"].item() < false_adopt["total_bk"].item()


def test_hierarchical_patch_loss_is_finite_for_saturated_float32_probabilities() -> None:
    y = torch.tensor([[[1.0, 1.0, -1.0, -1.0]]])
    base = torch.zeros_like(y)
    candidates = torch.tensor(
        [[[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, -1.0]]]]
    )
    adopt = torch.ones(1, 1, 2, requires_grad=True)
    conditional = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0]]]],
        requires_grad=True,
    )
    benefit = conditional.detach().clone().requires_grad_(True)
    risk = conditional.detach().clone().requires_grad_(True)
    final_adopt = torch.ones(1, 1, 2, requires_grad=True)
    utility = torch.zeros(1, 1, 2, 2, requires_grad=True)
    terms = _patch_router_hierarchical_recall_loss_terms(
        base_bch=base,
        candidate_bcpH=candidates,
        y_bch=y,
        patch_adopt_prob_bcq=adopt,
        patch_penalty_conditional_probs_bcqp=conditional,
        patch_penalty_benefit_probs_bcqp=benefit,
        patch_penalty_utility_scores_bcqp=utility,
        patch_penalty_risk_benefit_probs_bcqp=risk,
        patch_penalty_risk_positive_magnitude_bcqp=torch.ones_like(risk),
        patch_penalty_risk_negative_magnitude_bcqp=torch.zeros_like(risk),
        patch_final_adopt_prob_bcq=final_adopt,
        cluster_id_c=torch.tensor([0]),
        K=1,
        proposal_gain_listwise_weight=1.0,
        risk_calibration_weight=1.0,
        risk_sign_bce_weight=1.0,
        risk_magnitude_weight=1.0,
        selected_utility_policy_weight=1.0,
        selected_adoption_bce_weight=1.0,
        selected_adoption_recall_weight=1.0,
        selected_false_adopt_weight=1.0,
    )
    assert all(torch.isfinite(value).all() for value in terms.values())
    terms["total_bk"].mean().backward()
    for probability in (adopt, conditional, benefit, risk, final_adopt, utility):
        assert probability.grad is not None
        assert torch.isfinite(probability.grad).all()


def test_selected_adoption_supervision_targets_the_inference_candidate() -> None:
    y = torch.tensor([[[1.0, 1.0, -1.0, -1.0]]])
    base = torch.zeros_like(y)
    candidates = torch.tensor(
        [[[[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, -1.0, -1.0]]]]
    )
    conditional = torch.tensor([[[[0.9, 0.1], [0.9, 0.1]]]])
    common = {
        "base_bch": base,
        "candidate_bcpH": candidates,
        "y_bch": y,
        "patch_adopt_prob_bcq": torch.full((1, 1, 2), 0.5),
        "patch_penalty_conditional_probs_bcqp": conditional,
        "patch_penalty_benefit_probs_bcqp": conditional,
        "patch_penalty_utility_scores_bcqp": torch.zeros(1, 1, 2, 2),
        "cluster_id_c": torch.tensor([0]),
        "K": 1,
        "adoption_bce_weight": 0.0,
        "proposal_bce_weight": 0.0,
        "ranking_ce_weight": 0.0,
        "utility_regression_weight": 0.0,
        "adoption_recall_weight": 0.0,
        "false_adopt_weight": 0.0,
        "penalty_recall_weight": 0.0,
        "false_penalty_weight": 0.0,
        "selected_adoption_bce_weight": 1.0,
        "selected_adoption_recall_weight": 1.0,
        "selected_false_adopt_weight": 1.0,
    }
    good_prob = torch.tensor([[[0.9, 0.1]]], requires_grad=True)
    bad_prob = torch.tensor([[[0.1, 0.9]]])
    good = _patch_router_hierarchical_recall_loss_terms(
        **common,
        patch_final_adopt_prob_bcq=good_prob,
    )
    bad = _patch_router_hierarchical_recall_loss_terms(
        **common,
        patch_final_adopt_prob_bcq=bad_prob,
    )

    assert good["total_bk"].item() < bad["total_bk"].item()
    good["total_bk"].mean().backward()
    assert good_prob.grad is not None
    assert good_prob.grad[0, 0, 0] < 0.0
    assert good_prob.grad[0, 0, 1] > 0.0


def test_balanced_risk_sign_bce_rewards_candidate_benefit_classification() -> None:
    y = torch.tensor([[[1.0, 1.0, -1.0, -1.0]]])
    base = torch.zeros_like(y)
    candidates = torch.tensor(
        [[[[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, -1.0, -1.0]]]]
    )
    common = {
        "base_bch": base,
        "candidate_bcpH": candidates,
        "y_bch": y,
        "patch_adopt_prob_bcq": torch.full((1, 1, 2), 0.5),
        "patch_penalty_conditional_probs_bcqp": torch.full((1, 1, 2, 2), 0.5),
        "patch_penalty_benefit_probs_bcqp": torch.full((1, 1, 2, 2), 0.5),
        "patch_penalty_utility_scores_bcqp": torch.zeros(1, 1, 2, 2),
        "patch_penalty_risk_positive_magnitude_bcqp": torch.zeros(1, 1, 2, 2),
        "patch_penalty_risk_negative_magnitude_bcqp": torch.zeros(1, 1, 2, 2),
        "cluster_id_c": torch.tensor([0]),
        "K": 1,
        "adoption_bce_weight": 0.0,
        "proposal_bce_weight": 0.0,
        "ranking_ce_weight": 0.0,
        "utility_regression_weight": 0.0,
        "risk_calibration_weight": 0.0,
        "risk_sign_bce_weight": 1.0,
        "risk_magnitude_weight": 0.0,
        "adoption_recall_weight": 0.0,
        "false_adopt_weight": 0.0,
        "penalty_recall_weight": 0.0,
        "false_penalty_weight": 0.0,
    }
    good = _patch_router_hierarchical_recall_loss_terms(
        **common,
        patch_penalty_risk_benefit_probs_bcqp=torch.tensor(
            [[[[0.9, 0.1], [0.1, 0.9]]]]
        ),
    )
    bad = _patch_router_hierarchical_recall_loss_terms(
        **common,
        patch_penalty_risk_benefit_probs_bcqp=torch.tensor(
            [[[[0.1, 0.9], [0.9, 0.1]]]]
        ),
    )

    assert good["total_bk"].item() < bad["total_bk"].item()


def test_proposal_rescue_loss_targets_primary_misses() -> None:
    y = torch.ones(1, 1, 2)
    base = torch.zeros_like(y)
    candidates = torch.tensor([[[[1.0, 1.0], [0.5, 0.5], [0.0, 0.0]]]])
    common = {
        "base_bch": base,
        "candidate_bcpH": candidates,
        "y_bch": y,
        "patch_adopt_prob_bcq": torch.full((1, 1, 1), 0.5),
        "patch_penalty_conditional_probs_bcqp": torch.full((1, 1, 1, 3), 1.0 / 3.0),
        "patch_penalty_benefit_probs_bcqp": torch.full((1, 1, 1, 3), 0.5),
        "patch_penalty_utility_scores_bcqp": torch.zeros(1, 1, 1, 3),
        "patch_penalty_proposal_logits_bcqp": torch.tensor([[[[0.0, 5.0, 0.0]]]]),
        "cluster_id_c": torch.tensor([0]),
        "K": 1,
        "adoption_bce_weight": 0.0,
        "proposal_bce_weight": 0.0,
        "proposal_gain_listwise_weight": 0.0,
        "proposal_rescue_ce_weight": 1.0,
        "ranking_ce_weight": 0.0,
        "utility_regression_weight": 0.0,
        "adoption_recall_weight": 0.0,
        "false_adopt_weight": 0.0,
        "penalty_recall_weight": 0.0,
        "false_penalty_weight": 0.0,
    }
    good = _patch_router_hierarchical_recall_loss_terms(
        **common,
        patch_penalty_proposal_rescue_logits_bcqp=torch.tensor([[[[5.0, 0.0, -5.0]]]]),
    )
    bad = _patch_router_hierarchical_recall_loss_terms(
        **common,
        patch_penalty_proposal_rescue_logits_bcqp=torch.tensor([[[[-5.0, 0.0, 5.0]]]]),
    )
    assert good["total_bk"].item() < bad["total_bk"].item()


def test_selected_utility_policy_loss_balances_recall_and_harm() -> None:
    y = torch.tensor([[[1.0, 1.0, -1.0, -1.0]]])
    base = torch.zeros_like(y)
    candidates = torch.tensor([[[[1.0, 1.0, 1.0, 1.0]]]])
    common = {
        "base_bch": base,
        "candidate_bcpH": candidates,
        "y_bch": y,
        "patch_adopt_prob_bcq": torch.full((1, 1, 2), 0.5),
        "patch_penalty_conditional_probs_bcqp": torch.ones(1, 1, 2, 1),
        "patch_penalty_benefit_probs_bcqp": torch.full((1, 1, 2, 1), 0.5),
        "patch_penalty_utility_scores_bcqp": torch.zeros(1, 1, 2, 1),
        "cluster_id_c": torch.tensor([0]),
        "K": 1,
        "adoption_bce_weight": 0.0,
        "proposal_bce_weight": 0.0,
        "ranking_ce_weight": 0.0,
        "utility_regression_weight": 0.0,
        "adoption_recall_weight": 0.0,
        "false_adopt_weight": 0.0,
        "penalty_recall_weight": 0.0,
        "false_penalty_weight": 0.0,
        "selected_utility_policy_weight": 1.0,
    }
    good = _patch_router_hierarchical_recall_loss_terms(
        **common,
        patch_final_adopt_prob_bcq=torch.tensor([[[0.9, 0.1]]]),
    )
    bad = _patch_router_hierarchical_recall_loss_terms(
        **common,
        patch_final_adopt_prob_bcq=torch.tensor([[[0.1, 0.9]]]),
    )
    assert good["total_bk"].item() < bad["total_bk"].item()


def test_pairwise_rank_loss_uses_gain_difference_inside_shortlist() -> None:
    y = torch.ones(1, 1, 2)
    base = torch.zeros_like(y)
    candidates = torch.tensor([[[[1.0, 1.0], [0.5, 0.5]]]])
    common = {
        "base_bch": base,
        "candidate_bcpH": candidates,
        "y_bch": y,
        "patch_adopt_prob_bcq": torch.full((1, 1, 1), 0.5),
        "patch_penalty_conditional_probs_bcqp": torch.full((1, 1, 1, 2), 0.5),
        "patch_penalty_benefit_probs_bcqp": torch.full((1, 1, 1, 2), 0.5),
        "patch_penalty_utility_scores_bcqp": torch.zeros(1, 1, 1, 2),
        "patch_penalty_proposal_mask_bcqp": torch.ones(1, 1, 1, 2, dtype=torch.bool),
        "cluster_id_c": torch.tensor([0]),
        "K": 1,
        "adoption_bce_weight": 0.0,
        "proposal_bce_weight": 0.0,
        "ranking_ce_weight": 0.0,
        "utility_regression_weight": 0.0,
        "adoption_recall_weight": 0.0,
        "false_adopt_weight": 0.0,
        "penalty_recall_weight": 0.0,
        "false_penalty_weight": 0.0,
        "pairwise_rank_weight": 1.0,
    }
    good = _patch_router_hierarchical_recall_loss_terms(
        **common,
        patch_penalty_pairwise_rank_scores_bcqp=torch.tensor([[[[1.0, -1.0]]]]),
    )
    bad = _patch_router_hierarchical_recall_loss_terms(
        **common,
        patch_penalty_pairwise_rank_scores_bcqp=torch.tensor([[[[-1.0, 1.0]]]]),
    )
    assert good["total_bk"].item() < bad["total_bk"].item()


def test_patch_router_oracle_ce_trains_skip_and_patch_penalty_classes() -> None:
    y = torch.tensor([[[1.0, 1.0, -1.0, -1.0]]])
    base = torch.zeros_like(y)
    candidates = torch.tensor(
        [[[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, -1.0]]]]
    )
    correct_penalty_probs = torch.tensor(
        [[[[0.8, 0.1], [0.1, 0.8]]]],
        requires_grad=True,
    )
    swapped_penalty_probs = torch.tensor([[[[0.1, 0.8], [0.8, 0.1]]]])
    skip_prob = torch.full((1, 1, 2), 0.1)

    correct = _patch_router_oracle_ce_loss_bk(
        base_bch=base,
        candidate_bcpH=candidates,
        y_bch=y,
        patch_probs_bcqp=correct_penalty_probs,
        patch_skip_prob_bcq=skip_prob,
        cluster_id_c=torch.tensor([0]),
        K=1,
    )
    swapped = _patch_router_oracle_ce_loss_bk(
        base_bch=base,
        candidate_bcpH=candidates,
        y_bch=y,
        patch_probs_bcqp=swapped_penalty_probs,
        patch_skip_prob_bcq=skip_prob,
        cluster_id_c=torch.tensor([0]),
        K=1,
    )
    assert correct.item() < swapped.item()
    correct.mean().backward()
    assert correct_penalty_probs.grad is not None

    skip = _patch_router_oracle_ce_loss_bk(
        base_bch=base,
        candidate_bcpH=base.unsqueeze(2).expand(-1, -1, 2, -1),
        y_bch=y,
        patch_probs_bcqp=torch.full((1, 1, 2, 2), 0.05),
        patch_skip_prob_bcq=torch.full((1, 1, 2), 0.9),
        cluster_id_c=torch.tensor([0]),
        K=1,
    )
    assert skip.item() == pytest.approx(-torch.log(torch.tensor(0.9)).item())


def test_patch_router_second_stage_can_freeze_experts_but_keep_router_trainable() -> None:
    model = ClusterwisePredResidualMoE(
        num_clusters=2,
        num_penalties=2,
        input_len=4,
        pred_len=4,
        hidden_dim=3,
        shared_across_clusters=True,
        patch_router_cfg={"enable": True, "patch_len": 2, "hidden_dim": 4},
    )
    frozen = _freeze_module_params_except_prefixes(model, ("patch_router.",))

    assert frozen > 0
    assert all(param.requires_grad for param in model.patch_router.parameters())
    assert all(
        (name.startswith("patch_router.")) or (not param.requires_grad)
        for name, param in model.named_parameters()
    )


def test_patch_router_can_add_target_free_base_forecast_mismatch_features() -> None:
    model = ClusterwisePredResidualMoE(
        num_clusters=1,
        num_penalties=2,
        input_len=4,
        pred_len=4,
        hidden_dim=3,
        shared_across_clusters=True,
        patch_router_cfg={
            "enable": True,
            "patch_len": 2,
            "hidden_dim": 4,
            "use_base_forecast": True,
        },
    )
    router = model.patch_router
    assert router is not None
    x = torch.tensor([[[0.0, 1.0, 0.0, 1.0]]])
    features_a = router._features(x, torch.zeros(1, 1, 4))
    features_b = router._features(x, torch.ones(1, 1, 4) * 2.0)

    assert router.feature_source == "input_base"
    assert features_a.shape[-1] == (2 * router.patch_len) + 9
    assert not torch.allclose(features_a, features_b)


def test_patch_router_can_condition_every_decision_on_full_causal_history() -> None:
    model = ClusterwisePredResidualMoE(
        num_clusters=1,
        num_penalties=2,
        input_len=4,
        pred_len=4,
        hidden_dim=3,
        shared_across_clusters=True,
        patch_router_cfg={
            "enable": True,
            "patch_len": 2,
            "hidden_dim": 4,
            "use_base_forecast": True,
            "use_full_history_features": True,
        },
    )
    router = model.patch_router
    assert router is not None
    x_a = torch.tensor([[[0.0, 1.0, 0.0, 1.0]]])
    x_b = x_a.clone()
    x_b[..., 0] = 3.0
    base = torch.zeros(1, 1, 4)

    features_a = router._features(x_a, base)
    features_b = router._features(x_b, base)

    assert router.feature_source == "input_base_full_history"
    assert features_a.shape[-1] == (2 * router.patch_len) + 9 + router.L
    # Changing an early causal observation is visible even to the final
    # forecast-patch decision through the repeated full-history shape.
    assert not torch.allclose(features_a[:, :, -1], features_b[:, :, -1])


def test_shared_patch_router_can_learn_stable_channel_identity() -> None:
    model = ClusterwisePredResidualMoE(
        num_clusters=1,
        num_penalties=2,
        input_len=4,
        pred_len=4,
        hidden_dim=3,
        num_channels=2,
        shared_across_clusters=True,
        patch_router_cfg={
            "enable": True,
            "patch_len": 2,
            "hidden_dim": 4,
            "use_base_forecast": True,
            "use_channel_identity_features": True,
        },
    )
    router = model.patch_router
    assert router is not None
    x = torch.tensor(
        [[[0.0, 1.0, 0.0, 1.0]], [[0.0, 1.0, 0.0, 1.0]]]
    ).permute(1, 0, 2)
    base = torch.zeros(1, 2, 4)

    features = router._features(x, base)

    assert router.feature_source == "input_base_channel_id"
    assert features.shape[-1] == (2 * router.patch_len) + 9 + 2
    identity_start = router.patch_len + 6
    identity_end = identity_start + 2
    assert torch.equal(
        features[0, 0, :, identity_start:identity_end],
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
    )
    assert torch.equal(
        features[0, 1, :, identity_start:identity_end],
        torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
    )


def test_analytic_residual_gate_scores_candidate_from_predicted_base_error() -> None:
    model = ClusterwisePredResidualMoE(
        num_clusters=1,
        num_penalties=2,
        input_len=4,
        pred_len=4,
        hidden_dim=3,
        num_channels=1,
        shared_across_clusters=True,
        patch_router_cfg={
            "enable": True,
            "patch_len": 2,
            "hidden_dim": 4,
            "topk": 1,
            "allow_skip": True,
            "hierarchical_recall": {
                "enable": True,
                "expert_conditional_risk": {
                    "enable": True,
                    "candidate_aware": False,
                    "proposal_topk": 2,
                    "adoption_source": "expected_utility",
                    "dual_signed_utility": {
                        "enable": True,
                        "analytic_residual": {"enable": True},
                    },
                },
            },
        },
    )
    router = model.patch_router
    assert router is not None
    x = torch.tensor([[[0.0, 1.0, 0.0, 1.0]]])
    base = torch.zeros(1, 1, 4)
    candidate_delta = torch.stack(
        [torch.ones(1, 1, 4), -torch.ones(1, 1, 4)],
        dim=2,
    )

    initial = router(
        x,
        y_base_bch=base,
        candidate_delta_bcpH=candidate_delta,
        straight_through=False,
    )
    assert bool(initial["patch_skip_bcq"].bool().all())

    assert router.b_predicted_residual is not None
    with torch.no_grad():
        router.b_predicted_residual.fill_(2.0)
    scored = router(
        x,
        y_base_bch=base,
        candidate_delta_bcpH=candidate_delta,
        straight_through=False,
    )

    mse_scores = scored["patch_penalty_mse_utility_scores_bcqp"]
    mae_scores = scored["patch_penalty_mae_utility_scores_bcqp"]
    assert bool((mse_scores[..., 0] > 0.0).all())
    assert bool((mae_scores[..., 0] > 0.0).all())
    assert bool((mse_scores[..., 1] < 0.0).all())
    assert bool((mae_scores[..., 1] < 0.0).all())
    assert torch.equal(
        scored["patch_selected_penalty_index_bcq"],
        torch.zeros(1, 1, 2, dtype=torch.long),
    )
    assert not bool(scored["patch_skip_bcq"].bool().any())


def test_patch_router_time_phase_features_track_forecast_patch_position() -> None:
    model = ClusterwisePredResidualMoE(
        num_clusters=1,
        num_penalties=2,
        input_len=4,
        pred_len=4,
        hidden_dim=3,
        num_channels=1,
        shared_across_clusters=True,
        patch_router_cfg={
            "enable": True,
            "patch_len": 2,
            "hidden_dim": 4,
            "time_phase_periods": [4, 8],
        },
    )
    router = model.patch_router
    assert router is not None
    x = torch.tensor([[[0.0, 1.0, 0.0, 1.0]]])

    features_a = router._features(
        x,
        query_start_abs_b=torch.tensor([0]),
    )
    features_b = router._features(
        x,
        query_start_abs_b=torch.tensor([1]),
    )

    assert router.feature_source == "input_only_time_phase"
    assert features_a.shape[-1] == router.patch_len + 6 + 4
    assert not torch.allclose(features_a[..., -4:], features_b[..., -4:])
    assert not torch.allclose(features_a[:, :, 0, -4:], features_a[:, :, 1, -4:])


def test_patch_router_lagged_delta_features_preserve_same_phase_change() -> None:
    model = ClusterwisePredResidualMoE(
        num_clusters=1,
        num_penalties=2,
        input_len=4,
        pred_len=4,
        hidden_dim=3,
        num_channels=1,
        shared_across_clusters=True,
        patch_router_cfg={
            "enable": True,
            "patch_len": 2,
            "hidden_dim": 4,
            "regime_context": {"enable": True, "lengths": [8]},
            "lagged_delta_periods": [4],
        },
    )
    router = model.patch_router
    assert router is not None
    x = torch.tensor([[[3.0, 5.0, 7.0, 9.0]]])
    context = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 3.0, 5.0, 7.0, 9.0]]])

    lagged = router._lagged_delta_features(x, context)
    features = router._features(x, regime_context_bcl=context)

    assert lagged is not None
    scale = x.std(dim=-1, unbiased=False, keepdim=True)
    expected_delta = (x - context[..., :4]) / scale
    assert torch.allclose(lagged[:, :, 0, :2], expected_delta[..., :2])
    assert torch.allclose(lagged[:, :, 1, :2], expected_delta[..., 2:])
    assert router.feature_source == "input_only_lagged_delta"
    assert features.shape[-1] == (router.patch_len + 6) * 2 + 6


def test_patch_router_lagged_delta_features_require_sufficient_causal_context() -> None:
    with pytest.raises(ValueError, match="input_len plus the largest"):
        ClusterwisePredResidualMoE(
            num_clusters=1,
            num_penalties=2,
            input_len=4,
            pred_len=4,
            hidden_dim=3,
            num_channels=1,
            shared_across_clusters=True,
            patch_router_cfg={
                "enable": True,
                "patch_len": 2,
                "hidden_dim": 4,
                "regime_context": {"enable": True, "lengths": [4]},
                "lagged_delta_periods": [4],
            },
        )


def test_dual_utility_can_activate_multiple_structural_adapters_independently() -> None:
    model = ClusterwisePredResidualMoE(
        num_clusters=1,
        num_penalties=2,
        input_len=4,
        pred_len=4,
        hidden_dim=3,
        num_channels=1,
        shared_across_clusters=True,
        patch_router_cfg={
            "enable": True,
            "patch_len": 2,
            "hidden_dim": 4,
            "allow_skip": True,
            "hierarchical_recall": {
                "enable": True,
                "expert_conditional_risk": {
                    "enable": True,
                    "candidate_aware": False,
                    "proposal_topk": 2,
                    "adoption_source": "expected_utility",
                    "dual_signed_utility": {
                        "enable": True,
                        "independent_activation": {"enable": True},
                    },
                },
            },
        },
    )
    router = model.patch_router
    assert router is not None
    assert router.b_risk_mse_utility is not None
    assert router.b_risk_mae_utility is not None
    x = torch.tensor([[[0.0, 1.0, 0.0, 1.0]]])
    base = torch.zeros(1, 1, 4)
    candidate_delta = torch.zeros(1, 1, 2, 4)

    initial = router(
        x,
        y_base_bch=base,
        candidate_delta_bcpH=candidate_delta,
        straight_through=False,
    )
    assert bool(initial["patch_skip_bcq"].bool().all())

    with torch.no_grad():
        router.b_risk_mse_utility.copy_(torch.tensor([0.5, 0.25]))
        router.b_risk_mae_utility.copy_(torch.tensor([0.4, 0.2]))
    activated = router(
        x,
        y_base_bch=base,
        candidate_delta_bcpH=candidate_delta,
        straight_through=False,
    )

    assert torch.equal(
        activated["patch_route_bcph"],
        torch.ones(1, 1, 2, 4),
    )
    assert not bool(activated["patch_skip_bcq"].bool().any())


def test_patch_application_scale_changes_correction_without_changing_router_candidate() -> None:
    torch.manual_seed(91)
    base_kwargs = dict(
        num_clusters=1,
        num_penalties=2,
        input_len=4,
        pred_len=4,
        hidden_dim=3,
        num_channels=1,
        shared_across_clusters=True,
        intervention_enable=False,
        patch_router_cfg={
            "enable": True,
            "patch_len": 2,
            "hidden_dim": 4,
            "allow_skip": False,
        },
    )
    reference = ClusterwisePredResidualMoE(**base_kwargs)
    scaled_kwargs = dict(base_kwargs)
    scaled_kwargs["patch_router_cfg"] = {
        **base_kwargs["patch_router_cfg"],
        "application_scale_by_penalty": [0.0, 2.0],
    }
    scaled = ClusterwisePredResidualMoE(**scaled_kwargs)
    scaled.load_state_dict(reference.state_dict(), strict=True)
    x = torch.randn(2, 1, 4)
    base = torch.randn(2, 1, 4)
    cluster_id = torch.zeros(1, dtype=torch.long)
    outer_mask = torch.ones(2, 1, 2)

    out_reference = reference(x, base, cluster_id, outer_mask)
    out_scaled = scaled(x, base, cluster_id, outer_mask)

    assert torch.equal(
        out_reference["patch_penalty_utility_scores_bcqp"],
        out_scaled["patch_penalty_utility_scores_bcqp"],
    )
    assert torch.equal(
        out_reference["patch_route_bcph"],
        out_scaled["patch_route_bcph"],
    )
    route = out_reference["patch_route_bcph"]
    residuals = out_reference["residuals"]
    alpha = out_reference["alpha_cp"].view(1, 1, 2, 1)
    expected = base + (
        route
        * residuals
        * alpha
        * torch.tensor([0.0, 2.0]).view(1, 1, 2, 1)
    ).sum(dim=2)
    assert torch.allclose(out_scaled["y_final"], expected, atol=1.0e-6)


def test_eval_path_candidates_include_patch_application_scale() -> None:
    base = torch.zeros(1, 1, 4)
    residuals = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0]]]])
    pred_out = {
        "candidate_base_bch": base,
        "residuals": residuals,
        "alpha_cp": torch.ones(1, 2),
        "intervention_bcp": torch.ones(1, 1, 2),
        "selector_bcp": torch.ones(1, 1, 2),
        "patch_application_scale_p": torch.tensor([0.5, 2.0]),
    }

    eval_base, candidates = _pred_residual_candidates_on_eval_path(
        base,
        pred_out,
        include_patch_route=False,
    )

    assert torch.equal(eval_base, base)
    assert candidates is not None
    assert torch.equal(candidates[:, :, 0, :], 0.5 * residuals[:, :, 0, :])
    assert torch.equal(candidates[:, :, 1, :], 2.0 * residuals[:, :, 1, :])


def test_patch_router_regime_features_use_long_causal_context() -> None:
    model = ClusterwisePredResidualMoE(
        num_clusters=1,
        num_penalties=2,
        input_len=4,
        pred_len=4,
        hidden_dim=3,
        shared_across_clusters=True,
        patch_router_cfg={
            "enable": True,
            "patch_len": 2,
            "hidden_dim": 4,
            "use_base_forecast": True,
            "regime_context": {"enable": True, "lengths": [4, 8]},
        },
    )
    router = model.patch_router
    assert router is not None
    x = torch.tensor([[[4.0, 5.0, 6.0, 7.0]]])
    base = torch.zeros(1, 1, 4)
    context_a = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
    context_b = context_a.clone()
    context_b[..., :4] += 10.0

    features_a = router._features(x, base, regime_context_bcl=context_a)
    features_b = router._features(x, base, regime_context_bcl=context_b)

    assert router.regime_context_lengths == [4, 8]
    assert router.regime_feature_dim == 12
    assert features_a.shape[-1] == router.feature_dim
    assert not torch.allclose(features_a, features_b)


def test_pred_residual_gathers_only_history_before_forecast_origin() -> None:
    model = ClusterwisePredResidualMoE(
        num_clusters=1,
        num_penalties=2,
        input_len=4,
        pred_len=4,
        hidden_dim=3,
        shared_across_clusters=True,
        patch_router_cfg={
            "enable": True,
            "patch_len": 2,
            "hidden_dim": 4,
            "regime_context": {"enable": True, "lengths": [8]},
        },
    )
    history = torch.arange(20, dtype=torch.float32).reshape(20, 1)
    model.set_patch_router_observed_history(history)
    x = history[4:8].transpose(0, 1).unsqueeze(0)

    context = model._patch_router_regime_context(x, torch.tensor([4]))

    assert context is not None
    assert torch.equal(context, torch.arange(8, dtype=torch.float32).reshape(1, 1, 8))
    assert float(context.max().item()) == 7.0


def test_fusion_gate_prevents_plain_branch_sum() -> None:
    torch.manual_seed(13)
    model = ClusterwisePredResidualMoE(
        num_clusters=1,
        num_penalties=2,
        input_len=6,
        pred_len=3,
        hidden_dim=4,
        init_alpha=4.0,
        alpha_scale=1.0,
        intervention_enable=False,
        penalty_selector_enable=False,
        fusion_gate_enable=True,
        fusion_init=-8.0,
    )
    with torch.no_grad():
        for b2 in model.b2:
            b2.fill_(1.0)
        model.W_fusion[0].zero_()

    x = torch.randn(2, 3, 6)
    y_base = torch.zeros(2, 3, 3)
    cluster_id = torch.zeros(3, dtype=torch.long)
    route = torch.ones(2, 1, 2)
    out = model(x, y_base, cluster_id, route)

    plain_sum = out["branches"].sum(dim=2)
    fused_delta = out["y_final"] - y_base
    assert out["fusion_bc"].amax().item() < 0.01
    assert torch.max(torch.abs(fused_delta)).item() < torch.max(torch.abs(plain_sum)).item() * 0.01


def test_channel_penalty_mask_blocks_disallowed_channel_penalties() -> None:
    torch.manual_seed(17)
    model = ClusterwisePredResidualMoE(
        num_clusters=1,
        num_penalties=2,
        input_len=6,
        pred_len=3,
        hidden_dim=4,
        init_alpha=4.0,
        alpha_scale=1.0,
        intervention_enable=False,
        penalty_selector_enable=False,
        fusion_gate_enable=False,
    )
    model.set_channel_penalty_allowed_mask(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    x = torch.randn(2, 2, 6)
    y_base = torch.zeros(2, 2, 3)
    cluster_id = torch.zeros(2, dtype=torch.long)
    route = torch.ones(2, 1, 2)
    out = model(x, y_base, cluster_id, route)

    assert out["route_bcp"][:, 0, 1].abs().max().item() == 0.0
    assert out["route_bcp"][:, 1, 0].abs().max().item() == 0.0
    assert out["effective_route_bcp"][:, 0, 1].abs().max().item() == 0.0
    assert out["effective_route_bcp"][:, 1, 0].abs().max().item() == 0.0


def test_empty_channel_penalty_mask_is_bitwise_identical_to_unset_mask() -> None:
    torch.manual_seed(18)
    model = ClusterwisePredResidualMoE(
        num_clusters=2,
        num_penalties=3,
        input_len=6,
        pred_len=3,
        hidden_dim=4,
        init_alpha=1.0,
        alpha_scale=1.0,
        intervention_enable=False,
        penalty_selector_enable=False,
        fusion_gate_enable=False,
    )

    x = torch.randn(2, 3, 6)
    y_base = torch.randn(2, 3, 3)
    cluster_id = torch.tensor([0, 1, 0], dtype=torch.long)
    route = torch.rand(2, 2, 3)

    expected = model(x, y_base, cluster_id, route)

    for empty_mask in (torch.empty(0), torch.empty(0, 3), torch.empty(3, 0)):
        model.set_channel_penalty_allowed_mask(empty_mask)
        actual = model(x, y_base, cluster_id, route)
        for key, expected_value in expected.items():
            assert torch.equal(actual[key], expected_value), key

    model.set_channel_penalty_allowed_mask(None)
    actual = model(x, y_base, cluster_id, route)
    for key, expected_value in expected.items():
        assert torch.equal(actual[key], expected_value), key


def test_channel_expert_adapter_overrides_only_marked_channels() -> None:
    torch.manual_seed(19)
    model = ClusterwisePredResidualMoE(
        num_clusters=1,
        num_penalties=1,
        input_len=4,
        pred_len=2,
        hidden_dim=3,
        init_alpha=10.0,
        alpha_scale=1.0,
        intervention_enable=False,
        penalty_selector_enable=False,
        fusion_gate_enable=False,
        num_channels=2,
        channel_expert_mask_c=torch.tensor([False, True]),
        channel_expert_cluster_id_c=torch.tensor([0, 0]),
    )
    with torch.no_grad():
        for w in model.W1:
            w.zero_()
        for w in model.W2:
            w.zero_()
        for b in model.b2:
            b.fill_(1.0)
        for w in model.channel_W1:
            w.zero_()
        for w in model.channel_W2:
            w.zero_()
        for b in model.channel_b2:
            b.fill_(3.0)

    x = torch.randn(1, 2, 4)
    y_base = torch.zeros(1, 2, 2)
    cluster_id = torch.zeros(2, dtype=torch.long)
    route = torch.ones(1, 1, 1)
    out = model(x, y_base, cluster_id, route)

    assert torch.allclose(out["residuals"][0, 0, 0], torch.ones(2), atol=1e-5)
    assert torch.allclose(out["residuals"][0, 1, 0], torch.full((2,), 3.0), atol=1e-5)


def test_channel_expert_delta_refines_shared_adapter() -> None:
    torch.manual_seed(23)
    model = ClusterwisePredResidualMoE(
        num_clusters=1,
        num_penalties=1,
        input_len=4,
        pred_len=2,
        hidden_dim=3,
        init_alpha=10.0,
        alpha_scale=1.0,
        intervention_enable=False,
        penalty_selector_enable=False,
        fusion_gate_enable=False,
        num_channels=2,
        channel_expert_mask_c=torch.tensor([False, True]),
        channel_expert_cluster_id_c=torch.tensor([0, 0]),
        channel_expert_mode="delta",
    )
    with torch.no_grad():
        for w in model.W1:
            w.zero_()
        for w in model.W2:
            w.zero_()
        for b in model.b2:
            b.fill_(1.0)
        for w in model.channel_W1:
            w.zero_()
        for w in model.channel_W2:
            w.zero_()
        for b in model.channel_b2:
            b.fill_(3.0)

    x = torch.randn(1, 2, 4)
    y_base = torch.zeros(1, 2, 2)
    cluster_id = torch.zeros(2, dtype=torch.long)
    route = torch.ones(1, 1, 1)
    out = model(x, y_base, cluster_id, route)

    assert torch.allclose(out["residuals"][0, 0, 0], torch.ones(2), atol=1e-5)
    assert torch.allclose(out["residuals"][0, 1, 0], torch.full((2,), 4.0), atol=1e-5)


if __name__ == "__main__":
    test_adaptive_penalty_selector_is_cluster_specific()
    test_fusion_gate_prevents_plain_branch_sum()
    test_channel_penalty_mask_blocks_disallowed_channel_penalties()
    test_channel_expert_adapter_overrides_only_marked_channels()
    test_channel_expert_delta_refines_shared_adapter()
