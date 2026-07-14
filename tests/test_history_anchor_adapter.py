import ast
import inspect
import textwrap

import torch
import pytest

import src.train as train_module
from src.train import (
    PredResidualCandidateSelector,
    StaticPredResidualCandidateSelector,
    _candidate_selector_adoption_decision,
    _candidate_selector_candidate_scale,
    _candidate_selector_choose_temporal_margin_row,
    _candidate_selector_feature_gain_diagnostics,
    _candidate_selector_feature_standardization_stats,
    _candidate_selector_feature_names,
    _candidate_selector_features,
    _candidate_selector_patch_views,
    _candidate_selector_expected_error_loss,
    _candidate_selector_rate_alignment_loss,
    _candidate_selector_select_confirm_indices,
    _candidate_selector_targets,
    _concat_pred_residual_selector_tensors,
    _collect_pred_residual_selector_tensors,
    _copy_learnable_output_anchor_active_masks,
    _fit_static_candidate_channel_selector_from_tensors,
    _cluster_penalty_mask_to_channel_mask,
    _load_finetune_pred_residual_state,
    _make_cluster_optimizer_param_groups,
    _mix_selected_channel_metrics,
    _pred_residual_selector_metrics_from_tensors,
    _patchify_pred_residual_selector_tensors,
    _pred_residual_candidates_on_eval_path,
    _candidate_selector_temporal_block_adoption_guard,
    _pred_residual_selector_temporal_block_metrics,
    _select_learnable_output_anchor_channel_mask,
    _select_learnable_output_anchor_channel_horizon_mask,
    _normalize_history_anchor_cfg,
    _summarize_learnable_output_anchor_refiner,
    _validate_strict_history_anchor_scope,
    apply_default_moe_output_anchor_cfg,
    apply_moe_output_anchor_experts,
    apply_moe_history_anchor_expert,
    apply_history_anchor_adapter,
    apply_train_stat_anchor_expert,
    apply_train_stat_input_centering,
    apply_train_residual_anchor_expert,
    build_train_stat_anchor_from_config,
    build_train_phase_delta_anchor_table,
    build_train_phase_anchor_table,
    build_train_phase_residual_anchor_table,
    build_train_residual_anchor_table_from_loader,
    default_moe_output_anchor_cfg,
    select_channel_anchor_scales,
    select_channel_horizon_anchor_scales,
    select_train_stat_anchor_scales_from_loader,
    select_train_residual_anchor_scales_from_loader,
)
from src.models.learnable_anchor import ClusterwiseLearnableOutputAnchor
from src.models.residual_moe import ClusterwisePredResidualMoE


class _ZeroBackbone(torch.nn.Module):
    def __init__(self, pred_len: int) -> None:
        super().__init__()
        self.pred_len = int(pred_len)

    def forward(self, x: torch.Tensor, cluster_id_c: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], x.shape[1], self.pred_len, device=x.device, dtype=x.dtype)


class _MeanBackbone(torch.nn.Module):
    def __init__(self, pred_len: int) -> None:
        super().__init__()
        self.pred_len = int(pred_len)

    def forward(self, x: torch.Tensor, cluster_id_c: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=-1, keepdim=True).expand(-1, -1, self.pred_len)


def test_finetune_pred_residual_state_load_restores_checkpoint_weights() -> None:
    torch.manual_seed(12)
    source = ClusterwisePredResidualMoE(
        num_clusters=2,
        num_penalties=2,
        input_len=4,
        pred_len=2,
        hidden_dim=3,
        init_alpha=2.0,
        alpha_scale=1.0,
        intervention_enable=False,
        penalty_selector_enable=False,
        fusion_gate_enable=False,
    )
    target = ClusterwisePredResidualMoE(
        num_clusters=2,
        num_penalties=2,
        input_len=4,
        pred_len=2,
        hidden_dim=3,
        init_alpha=-5.0,
        alpha_scale=1.0,
        intervention_enable=False,
        penalty_selector_enable=False,
        fusion_gate_enable=False,
    )
    with torch.no_grad():
        for i, param in enumerate(source.parameters()):
            param.fill_(0.01 * float(i + 1))
        for param in target.parameters():
            param.zero_()

    loaded = _load_finetune_pred_residual_state(
        pred_residual=target,
        checkpoint={"pred_residual_state": source.state_dict()},
        source_penalty_names=["a", "b"],
        target_penalty_names=["a", "b"],
        strict=True,
    )

    assert loaded is True
    x = torch.randn(2, 2, 4)
    y_base = torch.randn(2, 2, 2)
    cluster_id = torch.tensor([0, 1], dtype=torch.long)
    route = torch.ones(2, 2, 2)
    source_out = source(x, y_base, cluster_id, route)
    target_out = target(x, y_base, cluster_id, route)
    for key, source_value in source_out.items():
        assert torch.equal(target_out[key], source_value), key


def test_non_strict_pred_residual_load_keeps_new_router_head_and_loads_experts() -> None:
    common = {
        "num_clusters": 1,
        "num_penalties": 2,
        "input_len": 8,
        "pred_len": 8,
        "hidden_dim": 3,
        "num_channels": 2,
        "penalty_names": ["a", "b"],
        "shared_across_clusters": True,
        "intervention_enable": False,
    }
    source = ClusterwisePredResidualMoE(
        **common,
        patch_router_cfg={
            "enable": True,
            "patch_len": 4,
            "hidden_dim": 4,
            "allow_skip": True,
        },
    )
    target = ClusterwisePredResidualMoE(
        **common,
        patch_router_cfg={
            "enable": True,
            "patch_len": 4,
            "hidden_dim": 4,
            "allow_skip": True,
            "hierarchical_recall": {"enable": True},
        },
    )
    with torch.no_grad():
        source.b2[0].fill_(0.25)
        target.b2[0].zero_()
    target_router_head_before = target.patch_router.W2.detach().clone()

    loaded = _load_finetune_pred_residual_state(
        pred_residual=target,
        checkpoint={"pred_residual_state": source.state_dict()},
        source_penalty_names=["a", "b"],
        target_penalty_names=["a", "b"],
        strict=False,
    )

    assert loaded is True
    assert torch.equal(target.b2[0], source.b2[0])
    assert torch.equal(target.patch_router.W2, target_router_head_before)


def test_cluster_penalty_mask_broadcasts_to_channel_penalty_mask() -> None:
    allowed_kp = torch.tensor(
        [
            [True, False, True],
            [False, True, False],
        ]
    )
    cluster_id_c = torch.tensor([1, 0, 1, 0], dtype=torch.long)

    allowed_cp = _cluster_penalty_mask_to_channel_mask(allowed_kp, cluster_id_c)

    assert torch.equal(
        allowed_cp,
        torch.tensor(
            [
                [False, True, False],
                [True, False, True],
                [False, True, False],
                [True, False, True],
            ]
        ),
    )


class _NoopPredResidual(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        y_base: torch.Tensor,
        cluster_id_c: torch.Tensor,
        mask_bkp: torch.Tensor,
        skip_bk: torch.Tensor | None = None,
        **_: object,
    ):
        b, c, h = y_base.shape
        p = mask_bkp.shape[-1]
        return {
            "y_final": y_base,
            "residuals": torch.zeros(b, c, p, h, device=y_base.device, dtype=y_base.dtype),
            "intervention_bcp": torch.ones(b, c, p, device=y_base.device, dtype=y_base.dtype),
            "selector_bcp": torch.ones(b, c, p, device=y_base.device, dtype=y_base.dtype),
            "alpha_cp": torch.ones(c, p, device=y_base.device, dtype=y_base.dtype),
        }


def test_history_anchor_adapter_reads_only_observed_values_before_forecast_start() -> None:
    observed = torch.arange(12, dtype=torch.float32).view(12, 1)
    pred = torch.zeros(1, 1, 3)
    out = apply_history_anchor_adapter(
        pred,
        base_pred_bch=pred,
        observed_history_tc=observed,
        query_start_abs_b=torch.tensor([4]),
        input_len=4,
        cfg={"enable": True, "lags": [4], "alpha": 1.0, "blend_target": "prediction"},
    )

    assert torch.allclose(out, torch.tensor([[[4.0, 5.0, 6.0]]]))


def test_history_anchor_adapter_averages_valid_lags_and_ignores_future_indices() -> None:
    observed = torch.arange(20, dtype=torch.float32).view(20, 1)
    pred = torch.zeros(1, 1, 2)
    out = apply_history_anchor_adapter(
        pred,
        base_pred_bch=pred,
        observed_history_tc=observed,
        query_start_abs_b=torch.tensor([3]),
        input_len=4,
        cfg={
            "enable": True,
            "lags": [1, 4, 8],
            "alpha": 1.0,
            "blend_target": "prediction",
            "history_scope": "all_observed",
        },
    )

    assert torch.allclose(out, torch.tensor([[[4.5, 2.0]]]))


def test_history_anchor_adapter_defaults_to_current_input_window() -> None:
    observed = torch.arange(20, dtype=torch.float32).view(20, 1)
    pred = torch.zeros(1, 1, 2)
    out = apply_history_anchor_adapter(
        pred,
        base_pred_bch=pred,
        observed_history_tc=observed,
        query_start_abs_b=torch.tensor([3]),
        input_len=4,
        cfg={"enable": True, "lags": [1, 4, 8], "alpha": 1.0, "blend_target": "prediction"},
    )

    assert torch.allclose(out, torch.tensor([[[4.5, 4.0]]]))


def test_history_anchor_adapter_can_restrict_to_current_input_window() -> None:
    observed = torch.arange(20, dtype=torch.float32).view(20, 1)
    pred = torch.zeros(1, 1, 2)
    out = apply_history_anchor_adapter(
        pred,
        base_pred_bch=pred,
        observed_history_tc=observed,
        query_start_abs_b=torch.tensor([3]),
        input_len=4,
        cfg={
            "enable": True,
            "lags": [1, 4, 8],
            "alpha": 1.0,
            "blend_target": "prediction",
            "history_scope": "input_window",
        },
    )

    assert torch.allclose(out, torch.tensor([[[4.5, 4.0]]]))


def test_training_history_anchor_defaults_to_current_input_window() -> None:
    cfg = _normalize_history_anchor_cfg(
        {"enable": True, "lags": [96, 192], "alpha": 0.2, "blend_target": "prediction"}
    )

    assert cfg["history_scope"] == "input_window"


def test_training_history_anchor_preserves_explicit_all_observed_scope() -> None:
    cfg = _normalize_history_anchor_cfg(
        {
            "enable": True,
            "lags": [96, 192],
            "alpha": 0.2,
            "blend_target": "prediction",
            "history_scope": "all_observed",
        }
    )

    assert cfg["history_scope"] == "all_observed"


def test_strict_history_anchor_rejects_all_observed_scope() -> None:
    cfg = {
        "enable": True,
        "lags": [96],
        "alpha": 0.2,
        "blend_target": "prediction",
        "history_scope": "all_observed",
    }

    with pytest.raises(ValueError, match="model.history_anchor.history_scope must be 'input_window'"):
        _validate_strict_history_anchor_scope(cfg, source="model.history_anchor")


def test_pred_residual_selector_tensor_collection_uses_history_anchored_base() -> None:
    x = torch.ones(1, 1, 4)
    y = torch.zeros(1, 1, 2)
    idx = torch.tensor([1])
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y, idx), batch_size=1)
    observed = torch.arange(20, dtype=torch.float32).view(20, 1)

    tensors = _collect_pred_residual_selector_tensors(
        model=_ZeroBackbone(pred_len=2),
        pred_residual=_NoopPredResidual(),
        loader=loader,
        cluster_id_c=torch.tensor([0]),
        K=1,
        moe_cfg={"enable": True},
        device=torch.device("cpu"),
        penalty_count=1,
        history_anchor_cfg={"enable": True, "lags": [4], "alpha": 1.0, "blend_target": "prediction"},
        observed_history_tc=observed,
        input_len=4,
        eval_start=3,
    )

    assert tensors is not None
    assert torch.allclose(tensors["base"], torch.tensor([[[4.0, 5.0]]]))


def test_train_residual_anchor_table_uses_model_train_stat_adapter() -> None:
    x = torch.tensor([[[2.0, 4.0]]])
    y = torch.tensor([[[10.0, 20.0]]])
    idx = torch.tensor([0])
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y, idx), batch_size=1)
    profile = torch.tensor([[2.0], [4.0], [10.0], [20.0]])
    adapter_cfg = {
        "enable": True,
        "period": 4,
        "mode": "phase_mean",
        "alpha": 1.0,
        "blend_target": "prediction",
        "combine_mode": "anchor_plus_prediction",
        "input_center": True,
    }

    table, counts, windows = build_train_residual_anchor_table_from_loader(
        model=_MeanBackbone(pred_len=2),
        loader=loader,
        cluster_id_c=torch.tensor([0]),
        device=torch.device("cpu"),
        history_anchor_cfg={},
        observed_history_tc=None,
        input_len=2,
        eval_start=0,
        period=4,
        model_train_stat_adapter_pc=profile,
        model_train_stat_adapter_cfg=adapter_cfg,
    )

    assert windows == 1
    assert counts.tolist() == [0, 0, 1, 0]
    assert torch.allclose(table[2], torch.zeros(2, 1))


def test_train_residual_anchor_scale_selection_uses_model_train_stat_adapter() -> None:
    x = torch.tensor([[[2.0, 4.0]]])
    y = torch.tensor([[[10.0, 20.0]]])
    idx = torch.tensor([0])
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y, idx), batch_size=1)
    profile = torch.tensor([[2.0], [4.0], [10.0], [20.0]])
    adapter_cfg = {
        "enable": True,
        "period": 4,
        "mode": "phase_mean",
        "alpha": 1.0,
        "blend_target": "prediction",
        "combine_mode": "anchor_plus_prediction",
        "input_center": True,
    }
    residual_table = torch.zeros(4, 2, 1)
    residual_table[2, :, 0] = torch.tensor([1.0, 1.0])

    scales, _, count = select_train_residual_anchor_scales_from_loader(
        model=_MeanBackbone(pred_len=2),
        loader=loader,
        cluster_id_c=torch.tensor([0]),
        device=torch.device("cpu"),
        history_anchor_cfg={},
        observed_history_tc=None,
        input_len=2,
        eval_start=0,
        residual_anchor_phc=residual_table,
        train_residual_anchor_cfg={"enable": True, "period": 4, "alpha": 0.0},
        metric="mse",
        max_scale=1.0,
        steps=3,
        model_train_stat_adapter_pc=profile,
        model_train_stat_adapter_cfg=adapter_cfg,
    )

    assert count == 1
    assert torch.allclose(scales, torch.zeros(1))


def test_train_stat_anchor_scale_selection_uses_input_center_for_anchor_plus_prediction() -> None:
    x = torch.tensor([[[3.0, 5.0]]])
    y = torch.tensor([[[2.0, 2.0]]])
    idx = torch.tensor([0])
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y, idx), batch_size=1)
    profile = torch.tensor([[2.0], [4.0], [1.0], [1.0]])
    adapter_cfg = {
        "enable": True,
        "period": 4,
        "mode": "phase_mean",
        "alpha": 1.0,
        "blend_target": "prediction",
        "combine_mode": "anchor_plus_prediction",
        "input_center": True,
    }

    scales, _, count = select_train_stat_anchor_scales_from_loader(
        model=_MeanBackbone(pred_len=2),
        loader=loader,
        cluster_id_c=torch.tensor([0]),
        device=torch.device("cpu"),
        history_anchor_cfg={},
        observed_history_tc=None,
        input_len=2,
        eval_start=0,
        stat_anchor_pc=profile,
        train_stat_anchor_cfg=adapter_cfg,
        metric="mse",
        max_scale=2.0,
        steps=3,
    )

    assert count == 1
    assert torch.allclose(scales, torch.ones(1))


def test_main_passes_history_anchor_context_to_gate_penalty_hit_metrics() -> None:
    source = textwrap.dedent(inspect.getsource(train_module.main))
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "evaluate_gate_penalty_hit_metrics"
    ]
    assert calls
    required_keywords = {"history_anchor_cfg", "observed_history_tc", "input_len", "eval_start"}
    for call in calls:
        assert required_keywords.issubset({kw.arg for kw in call.keywords if kw.arg is not None})


def test_eval_loop_defers_train_residual_anchor_until_table_exists() -> None:
    source = inspect.getsource(train_module.apply_moe_output_anchor_experts)

    assert "train_residual_anchor_phc is not None" in source


def test_candidate_selector_targets_require_positive_gain_over_skip() -> None:
    base = torch.zeros(1, 2, 2)
    y = torch.zeros(1, 2, 2)
    cand = torch.stack(
        [
            torch.tensor([[[1.0, 1.0], [0.0, 0.0]]]),
            torch.tensor([[[0.0, 0.0], [2.0, 2.0]]]),
        ],
        dim=2,
    )

    target = _candidate_selector_targets(
        base_bch=base,
        cand_bcpH=cand,
        y_bch=y,
        min_abs_improvement=0.01,
        min_rel_improvement=0.0,
    )

    assert torch.equal(target, torch.zeros(1, 2, dtype=torch.long))


def test_candidate_selector_targets_respect_allowed_penalties() -> None:
    base = torch.ones(1, 2, 2)
    y = torch.zeros(1, 2, 2)
    cand = torch.stack(
        [
            torch.ones(1, 2, 2),
            torch.zeros(1, 2, 2),
        ],
        dim=2,
    )
    allowed_mask_cp = torch.tensor([[True, False], [False, True]])

    target = _candidate_selector_targets(
        base_bch=base,
        cand_bcpH=cand,
        y_bch=y,
        allowed_mask_cp=allowed_mask_cp,
    )

    assert torch.equal(target, torch.tensor([[0, 2]]))


def test_candidate_selector_can_hard_select_penalty_candidate() -> None:
    selector = PredResidualCandidateSelector(feat_dim=13, num_channels=1, num_penalties=2, hidden_dim=2)
    with torch.no_grad():
        selector.skip_bias.fill_(0.0)
        selector.penalty_bias[:] = torch.tensor([0.0, 3.0])
        selector.net[-1].weight.zero_()
        selector.net[-1].bias.zero_()
        selector.skip_net[-1].weight.zero_()
        selector.skip_net[-1].bias.zero_()

    x = torch.zeros(1, 1, 2)
    base = torch.zeros(1, 1, 2)
    cand = torch.tensor([[[[1.0, 1.0], [2.0, 2.0]]]])
    selected, selected_class = selector.select_prediction(x, base, cand)

    assert torch.equal(selected_class, torch.tensor([[2]]))
    assert torch.allclose(selected, torch.tensor([[[2.0, 2.0]]]))


def test_candidate_selector_patch_views_align_history_and_forecast_patches() -> None:
    x = torch.arange(8, dtype=torch.float32).view(1, 1, 8)
    base = (10.0 + torch.arange(4, dtype=torch.float32)).view(1, 1, 4)
    cand = torch.stack([base + 1.0, base + 2.0], dim=2)

    x_patch, base_patch, cand_patch, patch_count = _candidate_selector_patch_views(x, base, cand, 2)

    assert patch_count == 2
    assert torch.equal(x_patch[:, 0], torch.tensor([[4.0, 5.0], [6.0, 7.0]]))
    assert torch.equal(base_patch[:, 0], torch.tensor([[10.0, 11.0], [12.0, 13.0]]))
    assert cand_patch.shape == (2, 1, 2, 2)


def test_candidate_selector_can_choose_skip_and_penalty_by_forecast_patch() -> None:
    selector = PredResidualCandidateSelector(
        feat_dim=13,
        num_channels=1,
        num_penalties=1,
        hidden_dim=1,
        use_time_features=True,
        time_feature_periods=[4],
        time_feature_offset=0,
        patch_len=2,
    )
    with torch.no_grad():
        for module in list(selector.net) + list(selector.skip_net):
            if isinstance(module, torch.nn.Linear):
                module.weight.zero_()
                module.bias.zero_()
        selector.net[0].weight[0, 13] = 1.0
        selector.net[-1].weight[0, 0] = 10.0
        selector.skip_bias.zero_()
        selector.penalty_bias.zero_()
        selector.penalty_channel_bias.zero_()

    x = torch.zeros(1, 1, 4)
    base = torch.zeros(1, 1, 4)
    cand = torch.ones(1, 1, 1, 4)
    selected, selected_class = selector.select_prediction(
        x,
        base,
        cand,
        query_start_abs_b=torch.tensor([1]),
    )

    assert torch.equal(selected_class, torch.tensor([[[1, 0]]]))
    assert torch.equal(selected, torch.tensor([[[1.0, 1.0, 0.0, 0.0]]]))


def test_patchify_selector_tensors_rebuilds_patch_features_and_query_times() -> None:
    x = torch.arange(8, dtype=torch.float32).view(1, 1, 8)
    base = torch.zeros(1, 1, 4)
    cand = torch.ones(1, 1, 1, 4)
    tensors = {
        "x": x,
        "base": base,
        "cand": cand,
        "y": torch.zeros_like(base),
        "confidence": torch.ones(1, 1, 1),
        "query_start_abs": torch.tensor([100]),
        "skip_feat": torch.empty(0),
        "cand_feat": torch.empty(0),
    }

    patchified = _patchify_pred_residual_selector_tensors(
        tensors,
        patch_len=2,
        feature_mode="base",
    )

    assert patchified["base"].shape == (2, 1, 2)
    assert patchified["cand"].shape == (2, 1, 1, 2)
    assert patchified["skip_feat"].shape == (2, 1, 13)
    assert torch.equal(patchified["query_start_abs"], torch.tensor([100, 102]))
    assert int(patchified["patch_count"].item()) == 2


def test_candidate_bank_can_ignore_online_patch_route() -> None:
    base = torch.zeros(1, 1, 4)
    residuals = torch.ones(1, 1, 1, 4)
    pred_out = {
        "residuals": residuals,
        "alpha_cp": torch.ones(1, 1),
        "intervention_bcp": torch.ones(1, 1, 1),
        "selector_bcp": torch.ones(1, 1, 1),
        "patch_route_bcph": torch.zeros(1, 1, 1, 4),
    }

    _, routed = _pred_residual_candidates_on_eval_path(base, pred_out, include_patch_route=True)
    _, raw = _pred_residual_candidates_on_eval_path(base, pred_out, include_patch_route=False)

    assert routed is not None and torch.equal(routed, base.unsqueeze(2))
    assert raw is not None and torch.equal(raw, torch.ones_like(raw))


def test_candidate_bank_eval_path_applies_output_anchor_to_base_and_candidates() -> None:
    base = torch.zeros(1, 1, 4)
    pred_out = {
        "residuals": torch.ones(1, 1, 1, 4),
        "alpha_cp": torch.ones(1, 1),
        "intervention_bcp": torch.ones(1, 1, 1),
        "selector_bcp": torch.ones(1, 1, 1),
    }
    anchor_cfg = {
        "train_stat_anchor_expert": {
            "enable": True,
            "mode": "phase_mean",
            "alpha_by_channel": [0.5],
            "blend_target": "prediction",
        }
    }

    anchored_base, anchored_candidates = _pred_residual_candidates_on_eval_path(
        base,
        pred_out,
        apply_output_anchors=True,
        x_bcl=torch.zeros(1, 1, 4),
        query_start_abs_b=torch.tensor([0]),
        input_len=0,
        moe_cfg=anchor_cfg,
        moe_enable=True,
        train_stat_anchor_pc=torch.full((2, 1), 2.0),
    )

    assert torch.equal(anchored_base, torch.ones_like(base))
    assert anchored_candidates is not None
    assert torch.equal(anchored_candidates, torch.full_like(anchored_candidates, 1.5))


def test_candidate_selector_hard_selection_masks_disallowed_penalties() -> None:
    selector = PredResidualCandidateSelector(feat_dim=13, num_channels=1, num_penalties=2, hidden_dim=2)
    selector.set_allowed_penalty_mask(torch.tensor([[True, False]]))
    with torch.no_grad():
        selector.skip_bias.fill_(0.0)
        selector.penalty_bias[:] = torch.tensor([1.0, 5.0])
        selector.net[-1].weight.zero_()
        selector.net[-1].bias.zero_()
        selector.skip_net[-1].weight.zero_()
        selector.skip_net[-1].bias.zero_()

    x = torch.zeros(1, 1, 2)
    base = torch.zeros(1, 1, 2)
    cand = torch.tensor([[[[1.0, 1.0], [9.0, 9.0]]]])
    selected, selected_class = selector.select_prediction(x, base, cand)

    assert torch.equal(selected_class, torch.tensor([[1]]))
    assert torch.allclose(selected, torch.tensor([[[1.0, 1.0]]]))


def test_candidate_selector_margin_can_force_skip_when_penalty_edge_is_small() -> None:
    selector = PredResidualCandidateSelector(feat_dim=13, num_channels=1, num_penalties=1, hidden_dim=2)
    with torch.no_grad():
        selector.skip_bias.fill_(0.0)
        selector.penalty_bias[:] = torch.tensor([1.0])
        selector.net[-1].weight.zero_()
        selector.net[-1].bias.zero_()
        selector.skip_net[-1].weight.zero_()
        selector.skip_net[-1].bias.zero_()
    selector.decision_margin = 2.0

    x = torch.zeros(1, 1, 2)
    base = torch.zeros(1, 1, 2)
    cand = torch.tensor([[[[2.0, 2.0]]]])
    selected, selected_class = selector.select_prediction(x, base, cand)

    assert torch.equal(selected_class, torch.tensor([[0]]))
    assert torch.allclose(selected, base)


def test_candidate_selector_can_append_penalty_identity_features() -> None:
    selector = PredResidualCandidateSelector(
        feat_dim=13,
        num_channels=1,
        num_penalties=2,
        hidden_dim=2,
        use_penalty_identity=True,
    )

    x = torch.zeros(1, 1, 2)
    base = torch.zeros(1, 1, 2)
    cand = torch.tensor([[[[1.0, 1.0], [2.0, 2.0]]]])
    logits = selector.logits(x, base, cand)

    assert selector.F == 15
    assert logits.shape == (1, 1, 3)


def test_candidate_selector_channel_identity_can_change_channel_logits() -> None:
    selector = PredResidualCandidateSelector(
        feat_dim=1,
        num_channels=2,
        num_penalties=1,
        hidden_dim=1,
        use_channel_identity=True,
    )
    with torch.no_grad():
        for module in list(selector.net) + list(selector.skip_net):
            if isinstance(module, torch.nn.Linear):
                module.weight.zero_()
                module.bias.zero_()
        selector.net[0].weight[0, 1] = 4.0
        selector.net[-1].weight[0, 0] = 1.0
        selector.skip_bias.zero_()
        selector.penalty_bias.zero_()
        selector.penalty_channel_bias.zero_()

    skip_feat = torch.zeros(1, 2, 1)
    cand_feat = torch.zeros(1, 2, 1, 1)
    logits = selector.logits_from_features(skip_feat, cand_feat)

    assert selector.F == 3
    assert logits.shape == (1, 2, 2)
    assert logits[0, 0, 1].item() > logits[0, 1, 1].item()


def test_candidate_selector_time_features_can_change_logits_by_phase() -> None:
    selector = PredResidualCandidateSelector(
        feat_dim=1,
        num_channels=1,
        num_penalties=1,
        hidden_dim=1,
        use_time_features=True,
        time_feature_periods=[4],
    )
    with torch.no_grad():
        for module in list(selector.net) + list(selector.skip_net):
            if isinstance(module, torch.nn.Linear):
                module.weight.zero_()
                module.bias.zero_()
        selector.net[0].weight[0, 1] = 4.0
        selector.net[-1].weight[0, 0] = 1.0
        selector.skip_bias.zero_()
        selector.penalty_bias.zero_()
        selector.penalty_channel_bias.zero_()

    skip_feat = torch.zeros(2, 1, 1)
    cand_feat = torch.zeros(2, 1, 1, 1)
    logits = selector.logits_from_features(
        skip_feat,
        cand_feat,
        query_start_abs_b=torch.tensor([0, 1]),
    )

    assert selector.F == 3
    assert logits[1, 0, 1].item() > logits[0, 0, 1].item()


def test_candidate_selector_expected_error_loss_prefers_better_candidate_logits() -> None:
    base = torch.zeros(1, 1, 2)
    cand = torch.ones(1, 1, 1, 2)
    y = torch.ones(1, 1, 2)
    skip_logits = torch.tensor([[[5.0, 0.0]]])
    candidate_logits = torch.tensor([[[0.0, 5.0]]])

    skip_loss = _candidate_selector_expected_error_loss(
        logits_bcq=skip_logits,
        base_bch=base,
        cand_bcpH=cand,
        y_bch=y,
    )
    candidate_loss = _candidate_selector_expected_error_loss(
        logits_bcq=candidate_logits,
        base_bch=base,
        cand_bcpH=cand,
        y_bch=y,
    )

    assert candidate_loss.item() < skip_loss.item()


def test_concat_pred_residual_selector_tensors_preserves_time_order() -> None:
    first = {
        "skip_feat": torch.zeros(2, 1, 1),
        "cand_feat": torch.zeros(2, 1, 1, 1),
        "base": torch.zeros(2, 1, 1),
        "cand": torch.zeros(2, 1, 1, 1),
        "confidence": torch.zeros(2, 1, 1),
        "y": torch.zeros(2, 1, 1),
    }
    second = {k: v + 1.0 for k, v in first.items()}

    joined = _concat_pred_residual_selector_tensors([first, None, second])

    assert joined is not None
    assert joined["base"].shape[0] == 4
    assert torch.equal(joined["base"][:, 0, 0], torch.tensor([0.0, 0.0, 1.0, 1.0]))


def test_candidate_selector_rate_alignment_penalizes_collapsed_class_prior() -> None:
    target_rate = torch.tensor([0.25, 0.25, 0.50])
    collapsed = torch.tensor([[[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]]])
    balanced = torch.log(target_rate).view(1, 1, 3).expand_as(collapsed)

    collapsed_loss = _candidate_selector_rate_alignment_loss(
        logits_bcq=collapsed,
        target_rate_q=target_rate,
    )
    balanced_loss = _candidate_selector_rate_alignment_loss(
        logits_bcq=balanced,
        target_rate_q=target_rate,
    )

    assert balanced_loss.item() < collapsed_loss.item()


def test_candidate_selector_metrics_oracle_respects_allowed_penalties() -> None:
    selector = PredResidualCandidateSelector(feat_dim=13, num_channels=2, num_penalties=2, hidden_dim=2)
    allowed_mask_cp = torch.tensor([[True, False], [False, True]])
    selector.set_allowed_penalty_mask(allowed_mask_cp)

    base = torch.ones(1, 2, 2)
    y = torch.zeros(1, 2, 2)
    cand = torch.stack(
        [
            torch.ones(1, 2, 2),
            torch.zeros(1, 2, 2),
        ],
        dim=2,
    )
    tensors = {
        "skip_feat": torch.zeros(1, 2, 13),
        "cand_feat": torch.zeros(1, 2, 2, 13),
        "base": base,
        "cand": cand,
        "y": y,
    }

    metrics = _pred_residual_selector_metrics_from_tensors(
        tensors=tensors,
        selector=selector,
        device=torch.device("cpu"),
        batch_size=1,
        min_abs_improvement=0.0,
        min_rel_improvement=0.0,
        allowed_mask_cp=allowed_mask_cp,
        penalty_names=["allowed0", "allowed1"],
    )

    assert metrics is not None
    assert metrics["oracle_gain_pct_vs_base"] == pytest.approx(50.0)
    assert metrics["target_class_rate"]["skip"] == pytest.approx(0.5)
    assert metrics["target_class_rate"]["allowed1"] == pytest.approx(0.5)
    assert metrics["oracle_class_rate"]["skip"] == pytest.approx(0.5)
    assert metrics["oracle_class_rate"]["allowed1"] == pytest.approx(0.5)


def test_candidate_selector_temporal_block_metrics_expose_target_rate_shift() -> None:
    selector = PredResidualCandidateSelector(feat_dim=1, num_channels=1, num_penalties=1, hidden_dim=2)
    selector.skip_bias.data.fill_(10.0)
    selector.penalty_bias.data.fill_(0.0)

    tensors = {
        "skip_feat": torch.zeros(4, 1, 1),
        "cand_feat": torch.zeros(4, 1, 1, 1),
        "base": torch.tensor([[[0.0]], [[0.0]], [[1.0]], [[1.0]]]),
        "cand": torch.tensor([[[[1.0]]], [[[1.0]]], [[[0.0]]], [[[0.0]]]]),
        "y": torch.zeros(4, 1, 1),
    }

    blocks = _pred_residual_selector_temporal_block_metrics(
        tensors=tensors,
        selector=selector,
        device=torch.device("cpu"),
        batch_size=2,
        num_blocks=2,
        min_abs_improvement=0.0,
        min_rel_improvement=0.0,
        penalty_names=["candidate"],
    )

    assert len(blocks) == 2
    assert blocks[0]["target_class_rate"]["skip"] == pytest.approx(1.0)
    assert blocks[0]["target_class_rate"]["candidate"] == pytest.approx(0.0)
    assert blocks[1]["target_class_rate"]["skip"] == pytest.approx(0.0)
    assert blocks[1]["target_class_rate"]["candidate"] == pytest.approx(1.0)
    assert blocks[1]["selected_class_rate"]["skip"] == pytest.approx(1.0)
    assert blocks[1]["target_gain_pct_vs_base"] > 0.0


def test_candidate_selector_adoption_rejects_mse_regression_even_if_mae_improves() -> None:
    decision = _candidate_selector_adoption_decision(
        current_mse=0.2136176,
        current_mae=0.3190942,
        selector_mse=0.2165933,
        selector_mae=0.3173998,
        min_abs_improvement=0.0,
        min_rel_improvement=0.0,
        max_rel_mae_regression=0.0,
    )

    assert decision["adopt"] is False
    assert decision["mse_improvement"] < 0.0


def test_candidate_selector_adoption_accepts_mse_win_without_mae_regression() -> None:
    decision = _candidate_selector_adoption_decision(
        current_mse=0.2136176,
        current_mae=0.3190942,
        selector_mse=0.209,
        selector_mae=0.318,
        min_abs_improvement=0.0,
        min_rel_improvement=0.001,
        max_rel_mae_regression=0.0,
    )

    assert decision["adopt"] is True
    assert decision["mse_improvement"] > decision["required_mse_improvement"]


def test_candidate_selector_temporal_block_adoption_guard_rejects_unstable_gain() -> None:
    blocks = [
        {"selected_gain_pct_vs_base": 1.2},
        {"selected_gain_pct_vs_base": -0.1},
        {"selected_gain_pct_vs_base": 0.4},
    ]

    rejected = _candidate_selector_temporal_block_adoption_guard(
        blocks=blocks,
        min_gain_pct=0.0,
    )
    accepted = _candidate_selector_temporal_block_adoption_guard(
        blocks=blocks,
        min_gain_pct=0.0,
        min_positive_blocks=2,
    )

    assert rejected["passed"] is False
    assert rejected["positive_block_count"] == 2
    assert rejected["required_positive_blocks"] == 3
    assert accepted["passed"] is True


def test_candidate_selector_temporal_margin_prefers_stable_feasible_row() -> None:
    rows = [
        {
            "margin": 0.5,
            "selected_mse": 0.9,
            "min_block_gain_pct": -0.1,
            "positive_block_count": 2,
        },
        {
            "margin": 1.0,
            "selected_mse": 0.95,
            "min_block_gain_pct": 0.01,
            "positive_block_count": 3,
        },
        {
            "margin": 1.5,
            "selected_mse": 0.97,
            "min_block_gain_pct": 0.02,
            "positive_block_count": 3,
        },
    ]

    selected = _candidate_selector_choose_temporal_margin_row(rows, required_positive_blocks=3)

    assert selected is rows[1]


def test_mix_selected_channel_metrics_falls_back_to_base_for_skipped_channels() -> None:
    base_mse = torch.tensor([1.0, 2.0, 3.0])
    base_mae = torch.tensor([0.5, 0.6, 0.7])
    residual_mse = torch.tensor([0.8, 1.5, 5.0])
    residual_mae = torch.tensor([0.4, 0.9, 0.1])
    use_residual = torch.tensor([True, False, True])

    mixed_mse, mixed_mae = _mix_selected_channel_metrics(
        base_mse_c=base_mse,
        base_mae_c=base_mae,
        residual_mse_c=residual_mse,
        residual_mae_c=residual_mae,
        use_residual_c=use_residual,
    )

    assert mixed_mse.tolist() == pytest.approx([0.8, 2.0, 5.0])
    assert mixed_mae.tolist() == pytest.approx([0.4, 0.6, 0.1])


def test_candidate_selector_candidate_scale_can_use_unscaled_candidates() -> None:
    scale = torch.tensor([1.0, 0.0])

    chosen, mode = _candidate_selector_candidate_scale(
        pred_residual_scale_c=scale,
        selector_cfg={"use_channel_scale_for_candidates": False},
    )

    assert chosen is None
    assert mode == "unscaled"


def test_candidate_selector_candidate_scale_defaults_to_channel_scale() -> None:
    scale = torch.tensor([1.0, 0.0])

    chosen, mode = _candidate_selector_candidate_scale(
        pred_residual_scale_c=scale,
        selector_cfg={},
    )

    assert chosen is scale
    assert mode == "channel_scale"


def test_candidate_selector_feature_gain_diagnostic_finds_signal_and_masks_disallowed() -> None:
    base = torch.ones(1, 2, 2)
    y = torch.zeros(1, 2, 2)
    cand = torch.stack(
        [
            torch.tensor([[[0.0, 0.0], [0.5, 0.5]]]),
            torch.full((1, 2, 2), 3.0),
        ],
        dim=2,
    )
    cand_feat = torch.zeros(1, 2, 2, 2)
    cand_feat[..., 0] = torch.tensor([[[1.0, 999.0], [0.75, 999.0]]])
    cand_feat[..., 1] = torch.tensor([[[0.0, 5.0], [0.0, 5.0]]])
    tensors = {
        "skip_feat": torch.zeros(1, 2, 2),
        "cand_feat": cand_feat,
        "base": base,
        "cand": cand,
        "y": y,
    }

    diag = _candidate_selector_feature_gain_diagnostics(
        tensors=tensors,
        feature_names=["signal", "noise"],
        penalty_names=["allowed", "blocked"],
        allowed_mask_cp=torch.tensor([[True, False], [True, False]]),
        topk=2,
    )

    assert diag is not None
    assert diag["samples"] == 2
    assert diag["positive_rate"] == pytest.approx(1.0)
    assert diag["top_abs_gain_corr"][0]["feature"] == "signal"
    assert diag["top_abs_gain_corr"][0]["corr"] == pytest.approx(1.0)
    assert diag["by_penalty"]["allowed"]["samples"] == 2
    assert diag["by_penalty"]["blocked"]["samples"] == 0


def test_candidate_selector_robust_standardization_resists_outlier_and_clip_applies() -> None:
    selector = PredResidualCandidateSelector(feat_dim=1, num_channels=1, num_penalties=1, hidden_dim=2)
    skip_feat = torch.tensor([[[0.0]], [[0.0]], [[0.0]], [[0.0]], [[1000.0]]])
    cand_feat = torch.tensor([[[[0.0]]], [[[1.0]]], [[[1.0]]], [[[0.0]]], [[[1000.0]]]])
    train_idx = torch.arange(5)
    mean_std_mean, mean_std_std, mean_std_summary = _candidate_selector_feature_standardization_stats(
        skip_feat=skip_feat,
        cand_feat=cand_feat,
        selector=selector,
        train_idx=train_idx,
        mode="mean_std",
    )
    robust_mean, robust_std, robust_summary = _candidate_selector_feature_standardization_stats(
        skip_feat=skip_feat,
        cand_feat=cand_feat,
        selector=selector,
        train_idx=train_idx,
        mode="robust",
    )

    assert mean_std_summary["mode"] == "mean_std"
    assert robust_summary["mode"] == "robust"
    assert robust_mean.item() < mean_std_mean.item()
    assert robust_std.item() < mean_std_std.item()

    selector.set_feature_standardization(robust_mean, robust_std)
    selector.set_feature_standardize_clip(2.0)
    standardized = selector._standardize_feat(torch.tensor([[[1000.0]]]))

    assert standardized.max().item() == pytest.approx(2.0)


def test_candidate_selector_history_proxy_features_measure_candidate_distance_to_history() -> None:
    x = torch.tensor([[[0.0, 1.0, 2.0, 3.0]]])
    base = torch.tensor([[[2.0, 3.0]]])
    cand = torch.tensor([[[4.0, 6.0]]])

    names = _candidate_selector_feature_names("history_proxy")
    feat = _candidate_selector_features(x, base, cand, feature_mode="history_proxy")
    proxy_mse_delta_idx = names.index("proxy_mse_delta")
    proxy_mae_delta_idx = names.index("proxy_mae_delta")

    assert feat.shape[-1] == len(names)
    assert feat[0, 0, proxy_mse_delta_idx].item() > 0.0
    assert feat[0, 0, proxy_mae_delta_idx].item() > 0.0


def test_candidate_selector_shape_proxy_features_are_target_free_shape_descriptors() -> None:
    x = torch.tensor([[[0.0, 1.0, 2.0, 3.0]]])
    base = torch.tensor([[[3.0, 4.0]]])
    cand = torch.tensor([[[3.0, 5.0]]])

    names = _candidate_selector_feature_names("shape_proxy")
    feat = _candidate_selector_features(x, base, cand, feature_mode="shape_proxy")

    assert feat.shape[-1] == len(names)
    assert len(names) > len(_candidate_selector_feature_names("history_proxy"))
    assert "hybrid_base_slope_delta" in names
    assert "hybrid_hist_corr" in names
    assert "hybrid_diff_rms" in names
    assert torch.isfinite(feat).all()
    assert feat[0, 0, names.index("hybrid_base_slope_delta")].item() > 0.0


def test_static_candidate_channel_selector_uses_only_allowed_improving_candidates() -> None:
    base = torch.ones(2, 2, 2)
    y = torch.zeros(2, 2, 2)
    cand = torch.stack(
        [
            torch.zeros(2, 2, 2),
            torch.full((2, 2, 2), 0.5),
        ],
        dim=2,
    )
    tensors = {
        "skip_feat": torch.zeros(2, 2, 1),
        "cand_feat": torch.zeros(2, 2, 2, 1),
        "base": base,
        "cand": cand,
        "y": y,
    }

    selector, summary = _fit_static_candidate_channel_selector_from_tensors(
        tensors=tensors,
        allowed_mask_cp=torch.tensor([[False, True], [True, False]]),
        penalty_names=["blocked_best", "allowed_second"],
        channel_names=["ch0", "ch1"],
        min_abs_improvement=0.01,
    )
    selected, selected_class = selector.select_prediction(torch.zeros(2, 2, 2), base, cand)

    assert isinstance(selector, StaticPredResidualCandidateSelector)
    assert summary["selected_class"] == [2, 1]
    assert summary["selected_penalty_by_channel"] == ["allowed_second", "blocked_best"]
    assert torch.equal(selected_class, torch.tensor([[2, 1], [2, 1]]))
    assert torch.allclose(selected[:, 0], torch.full((2, 2), 0.5))
    assert torch.allclose(selected[:, 1], torch.zeros(2, 2))


def test_static_candidate_channel_selector_chunked_metrics_match_single_chunk() -> None:
    y = torch.zeros(5, 2, 3)
    base = torch.arange(30, dtype=torch.float32).reshape(5, 2, 3) / 10.0
    cand = torch.stack((base * 0.5, base + 0.25), dim=2)
    tensors = {
        "skip_feat": torch.zeros(5, 2, 1),
        "cand_feat": torch.zeros(5, 2, 2, 1),
        "base": base,
        "cand": cand,
        "y": y,
    }

    selector_full, summary_full = _fit_static_candidate_channel_selector_from_tensors(
        tensors=tensors,
        metric_max_elements=1_000_000,
    )
    selector_chunked, summary_chunked = _fit_static_candidate_channel_selector_from_tensors(
        tensors=tensors,
        metric_max_elements=1,
    )

    assert torch.equal(selector_full.selected_class_c, selector_chunked.selected_class_c)
    for key in (
        "select_base_mse_per_channel",
        "select_best_candidate_mse_per_channel",
        "eval_base_mse_per_channel",
        "eval_selected_mse_per_channel",
    ):
        assert summary_chunked[key] == pytest.approx(summary_full[key])


def test_static_candidate_channel_selector_keeps_base_without_required_gain() -> None:
    base = torch.zeros(1, 1, 2)
    y = torch.zeros(1, 1, 2)
    cand = torch.ones(1, 1, 1, 2)
    tensors = {
        "skip_feat": torch.zeros(1, 1, 1),
        "cand_feat": torch.zeros(1, 1, 1, 1),
        "base": base,
        "cand": cand,
        "y": y,
    }

    selector, summary = _fit_static_candidate_channel_selector_from_tensors(
        tensors=tensors,
        allowed_mask_cp=torch.ones(1, 1, dtype=torch.bool),
        penalty_names=["bad"],
        channel_names=["ch0"],
        min_abs_improvement=0.01,
    )

    selected, selected_class = selector.select_prediction(torch.zeros(1, 1, 2), base, cand)

    assert summary["selected_class"] == [0]
    assert summary["num_candidate_channels"] == 0
    assert torch.equal(selected_class, torch.zeros(1, 1, dtype=torch.long))
    assert torch.allclose(selected, base)


def test_static_candidate_channel_selector_mae_guard_skips_mae_regressing_best_mse_candidate() -> None:
    y = torch.zeros(1, 1, 4)
    base = torch.tensor([[[2.0, 0.0, 0.0, 0.0]]])
    cand = torch.stack(
        [
            torch.tensor([[[1.0, 1.0, 1.0, 0.0]]]),
            torch.tensor([[[1.8, 0.0, 0.0, 0.0]]]),
        ],
        dim=2,
    )
    tensors = {
        "skip_feat": torch.zeros(1, 1, 1),
        "cand_feat": torch.zeros(1, 1, 2, 1),
        "base": base,
        "cand": cand,
        "y": y,
    }

    selector, summary = _fit_static_candidate_channel_selector_from_tensors(
        tensors=tensors,
        allowed_mask_cp=torch.ones(1, 2, dtype=torch.bool),
        penalty_names=["best_mse_mae_regresses", "stable_mae"],
        channel_names=["ch0"],
        min_abs_improvement=0.0,
        min_abs_mae_improvement=0.0,
    )

    selected, selected_class = selector.select_prediction(torch.zeros(1, 1, 4), base, cand)

    assert summary["selected_class"] == [2]
    assert summary["selected_penalty_by_channel"] == ["stable_mae"]
    assert summary["mae_guard_enabled"] is True
    assert torch.equal(selected_class, torch.tensor([[2]]))
    assert torch.allclose(selected, cand[:, :, 1])


def test_static_candidate_channel_selector_default_allows_best_mse_candidate_with_mae_regression() -> None:
    y = torch.zeros(1, 1, 4)
    base = torch.tensor([[[2.0, 0.0, 0.0, 0.0]]])
    cand = torch.stack(
        [
            torch.tensor([[[1.0, 1.0, 1.0, 0.0]]]),
            torch.tensor([[[1.8, 0.0, 0.0, 0.0]]]),
        ],
        dim=2,
    )
    tensors = {
        "skip_feat": torch.zeros(1, 1, 1),
        "cand_feat": torch.zeros(1, 1, 2, 1),
        "base": base,
        "cand": cand,
        "y": y,
    }

    selector, summary = _fit_static_candidate_channel_selector_from_tensors(
        tensors=tensors,
        allowed_mask_cp=torch.ones(1, 2, dtype=torch.bool),
        penalty_names=["best_mse_mae_regresses", "stable_mae"],
        channel_names=["ch0"],
        min_abs_improvement=0.0,
    )

    selected, selected_class = selector.select_prediction(torch.zeros(1, 1, 4), base, cand)

    assert summary["selected_class"] == [1]
    assert summary["selected_penalty_by_channel"] == ["best_mse_mae_regresses"]
    assert summary["mae_guard_enabled"] is False
    assert summary["min_abs_mae_improvement"] is None
    assert torch.equal(selected_class, torch.tensor([[1]]))
    assert torch.allclose(selected, cand[:, :, 0])


def test_static_candidate_channel_selector_can_select_by_mae_with_mse_guard() -> None:
    y = torch.zeros(1, 1, 4)
    base = torch.tensor([[[2.0, 0.0, 0.0, 0.0]]])
    cand = torch.stack(
        [
            torch.tensor([[[1.0, 1.0, 1.0, 0.0]]]),
            torch.tensor([[[1.8, 0.0, 0.0, 0.0]]]),
        ],
        dim=2,
    )
    tensors = {
        "skip_feat": torch.zeros(1, 1, 1),
        "cand_feat": torch.zeros(1, 1, 2, 1),
        "base": base,
        "cand": cand,
        "y": y,
    }

    selector, summary = _fit_static_candidate_channel_selector_from_tensors(
        tensors=tensors,
        allowed_mask_cp=torch.ones(1, 2, dtype=torch.bool),
        penalty_names=["best_mse_mae_regresses", "best_mae"],
        channel_names=["ch0"],
        selection_metric="mae",
        min_abs_improvement=0.0,
    )

    selected, selected_class = selector.select_prediction(torch.zeros(1, 1, 4), base, cand)

    assert summary["selection_metric"] == "mae"
    assert summary["selected_class"] == [2]
    assert summary["selected_penalty_by_channel"] == ["best_mae"]
    assert torch.equal(selected_class, torch.tensor([[2]]))
    assert torch.allclose(selected, cand[:, :, 1])


def test_static_candidate_channel_selector_confirm_guard_rejects_select_only_gain() -> None:
    y = torch.zeros(2, 1, 1)
    base = torch.ones(2, 1, 1)
    cand = torch.tensor([[[[0.0]]], [[[2.0]]]])
    tensors = {
        "skip_feat": torch.zeros(2, 1, 1),
        "cand_feat": torch.zeros(2, 1, 1, 1),
        "base": base,
        "cand": cand,
        "y": y,
    }

    selector, summary = _fit_static_candidate_channel_selector_from_tensors(
        tensors=tensors,
        allowed_mask_cp=torch.ones(1, 1, dtype=torch.bool),
        penalty_names=["select_only"],
        channel_names=["ch0"],
        select_indices=torch.tensor([0]),
        eval_indices=torch.tensor([1]),
        min_abs_improvement=0.0,
        confirm_min_abs_improvement=0.0,
        confirm_min_abs_mae_improvement=0.0,
    )

    selected, selected_class = selector.select_prediction(torch.zeros(2, 1, 1), base, cand)

    assert summary["selected_class"] == [0]
    assert summary["confirm_guard_enabled"] is True
    assert summary["confirm_mae_guard_enabled"] is True
    assert summary["confirm_mse_gain_per_channel"] == [-3.0]
    assert torch.equal(selected_class, torch.zeros(2, 1, dtype=torch.long))
    assert torch.allclose(selected, base)


def test_static_candidate_channel_selector_segment_guard_rejects_unstable_gain() -> None:
    y = torch.zeros(4, 1, 1)
    base = torch.ones(4, 1, 1)
    cand = torch.tensor([[[[0.0]]], [[[0.0]]], [[[0.0]]], [[[2.0]]]])
    tensors = {
        "skip_feat": torch.zeros(4, 1, 1),
        "cand_feat": torch.zeros(4, 1, 1, 1),
        "base": base,
        "cand": cand,
        "y": y,
    }

    selector, summary = _fit_static_candidate_channel_selector_from_tensors(
        tensors=tensors,
        allowed_mask_cp=torch.ones(1, 1, dtype=torch.bool),
        penalty_names=["unstable"],
        channel_names=["ch0"],
        min_abs_improvement=0.0,
        segment_count=4,
        segment_min_positive=4,
        segment_min_abs_improvement=0.0,
        segment_min_abs_mae_improvement=0.0,
    )

    selected, selected_class = selector.select_prediction(torch.zeros(4, 1, 1), base, cand)

    assert summary["selected_class"] == [0]
    assert summary["segment_guard_enabled"] is True
    assert summary["segment_positive_count_per_channel"] == [3]
    assert torch.equal(selected_class, torch.zeros(4, 1, dtype=torch.long))
    assert torch.allclose(selected, base)


def test_candidate_selector_select_confirm_indices_split_tail_confirmation() -> None:
    select_indices, confirm_indices = _candidate_selector_select_confirm_indices(6, 0.5)

    assert torch.equal(select_indices, torch.tensor([0, 1, 2]))
    assert torch.equal(confirm_indices, torch.tensor([3, 4, 5]))

    select_indices, confirm_indices = _candidate_selector_select_confirm_indices(6, 0.0)

    assert select_indices is None
    assert confirm_indices is None


def test_default_moe_output_anchor_cfg_uses_pems_main_table_defaults() -> None:
    cfg = default_moe_output_anchor_cfg("data/PEMS08.csv", 96)

    assert cfg["history_anchor_expert"] == {"enable": False}
    assert cfg["train_stat_anchor_expert"]["period"] == 288
    assert cfg["train_stat_anchor_expert"]["scale_selection"] == {
        "enable": True,
        "metric": "mse",
        "max_scale": 0.2,
        "steps": 9,
    }
    assert cfg["train_residual_anchor_expert"]["period"] == 288
    assert cfg["train_residual_anchor_expert"]["scale_selection"] == {
        "enable": True,
        "metric": "mse",
        "max_scale": 1.2,
        "steps": 49,
        "horizon_segments": 4,
    }


def test_default_moe_output_anchor_cfg_uses_ettm2_h96_best_defaults() -> None:
    cfg = default_moe_output_anchor_cfg("ETTm2_H96", 96)

    assert cfg["history_anchor_expert"] == {"enable": False}
    assert cfg["train_stat_anchor_expert"]["scale_selection"] == {
        "enable": True,
        "metric": "mae",
        "max_scale": 0.18,
        "steps": 8,
    }
    assert cfg["train_residual_anchor_expert"]["scale_selection"] == {
        "enable": True,
        "metric": "mae",
        "max_scale": 1.2,
        "steps": 49,
        "horizon_segments": 7,
    }


@pytest.mark.parametrize(
    ("horizon", "period", "metric", "stat_max", "stat_steps", "resid_max", "resid_steps", "segments"),
    [
        (96, 144, "mse", 0.4, 13, 0.8, 25, 8),
        (192, 144, "mae", 0.5, 13, 1.0, 25, 8),
        (336, 96, "mae", 0.2, 9, 1.2, 49, 7),
        (720, 96, "mae", 0.2, 9, 1.2, 49, 7),
    ],
)
def test_default_moe_output_anchor_cfg_uses_weather_best_anchor_defaults(
    horizon: int,
    period: int,
    metric: str,
    stat_max: float,
    stat_steps: int,
    resid_max: float,
    resid_steps: int,
    segments: int,
) -> None:
    cfg = default_moe_output_anchor_cfg("data/weather.csv", horizon)

    assert cfg["history_anchor_expert"] == {"enable": False}
    assert cfg["train_stat_anchor_expert"]["period"] == period
    assert cfg["train_stat_anchor_expert"]["scale_selection"] == {
        "enable": True,
        "metric": metric,
        "max_scale": stat_max,
        "steps": stat_steps,
    }
    assert cfg["train_residual_anchor_expert"]["period"] == period
    assert cfg["train_residual_anchor_expert"]["scale_selection"] == {
        "enable": True,
        "metric": metric,
        "max_scale": resid_max,
        "steps": resid_steps,
        "horizon_segments": segments,
    }


def test_default_moe_output_anchor_cfg_normalizes_weather_horizon_suffix() -> None:
    assert default_moe_output_anchor_cfg("weather_H192", 192)["train_stat_anchor_expert"]["period"] == 144


def test_default_moe_output_anchor_cfg_preserves_disabled_residual_cell() -> None:
    cfg = default_moe_output_anchor_cfg("ETTh2", 720)

    assert cfg["train_stat_anchor_expert"]["scale_selection"]["max_scale"] == 0.4
    assert cfg["train_residual_anchor_expert"] == {"enable": False}


def test_apply_default_moe_output_anchor_cfg_merges_explicit_experiment_override() -> None:
    cfg = apply_default_moe_output_anchor_cfg(
        {"train_residual_anchor_expert": {"enable": False}},
        dataset_name="ETTm1",
        pred_len=96,
    )

    assert cfg["train_stat_anchor_expert"]["enable"] is True
    assert cfg["train_residual_anchor_expert"]["enable"] is False
    assert cfg["train_residual_anchor_expert"]["scale_selection"]["max_scale"] == 1.6


def test_moe_history_anchor_expert_is_disabled_when_moe_side_config_is_off() -> None:
    observed = torch.arange(12, dtype=torch.float32).view(12, 1)
    pred = torch.zeros(1, 1, 2)
    out = apply_moe_history_anchor_expert(
        pred,
        base_pred_bch=pred,
        observed_history_tc=observed,
        query_start_abs_b=torch.tensor([4]),
        input_len=4,
        cfg={"enable": False, "lags": [4], "alpha": 1.0},
    )

    assert torch.allclose(out, pred)


def test_moe_output_anchor_experts_apply_when_penalty_moe_is_disabled() -> None:
    pred = torch.zeros(1, 1, 2)
    residual_anchor_phc = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])

    out = apply_moe_output_anchor_experts(
        pred,
        base_pred_bch=pred,
        x_bcl=torch.zeros(1, 1, 2),
        query_start_abs_b=torch.tensor([0]),
        input_len=0,
        moe_cfg={
            "train_residual_anchor_expert": {
                "enable": True,
                "period": 2,
                "alpha": 1.0,
                "blend_target": "prediction",
            }
        },
        moe_enable=False,
        train_residual_anchor_phc=residual_anchor_phc,
    )

    assert torch.allclose(out, torch.tensor([[[1.0, 2.0]]]))


def test_learnable_output_anchor_zero_init_preserves_static_anchor() -> None:
    pred = torch.full((1, 2, 2), 2.0)
    stat_anchor_pc = torch.tensor([[10.0, 20.0]], dtype=torch.float32)
    residual_anchor_phc = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
    cluster_id_c = torch.tensor([0, 1], dtype=torch.long)
    module = ClusterwiseLearnableOutputAnchor(
        num_clusters=2,
        num_channels=2,
        pred_len=2,
        cfg={"enable": True, "learn_bias": True},
    )

    static_out = apply_moe_output_anchor_experts(
        pred,
        base_pred_bch=pred,
        x_bcl=torch.zeros(1, 2, 2),
        query_start_abs_b=torch.tensor([0]),
        input_len=0,
        moe_cfg={
            "train_stat_anchor_expert": {"enable": True, "alpha": 0.5},
            "train_residual_anchor_expert": {"enable": True, "alpha": 0.25},
        },
        moe_enable=False,
        train_stat_anchor_pc=stat_anchor_pc,
        train_residual_anchor_phc=residual_anchor_phc,
    )
    learned_out = apply_moe_output_anchor_experts(
        pred,
        base_pred_bch=pred,
        x_bcl=torch.zeros(1, 2, 2),
        query_start_abs_b=torch.tensor([0]),
        input_len=0,
        moe_cfg={
            "learnable_output_anchor": {"enable": True, "learn_bias": True},
            "train_stat_anchor_expert": {"enable": True, "alpha": 0.5},
            "train_residual_anchor_expert": {"enable": True, "alpha": 0.25},
        },
        moe_enable=False,
        train_stat_anchor_pc=stat_anchor_pc,
        train_residual_anchor_phc=residual_anchor_phc,
        learnable_output_anchor=module,
        cluster_id_c=cluster_id_c,
    )

    assert torch.allclose(learned_out, static_out)


def test_learnable_output_anchor_accepts_boolean_config_shorthand() -> None:
    pred = torch.ones(1, 2, 2)
    cluster_id_c = torch.tensor([0, 1], dtype=torch.long)
    module = ClusterwiseLearnableOutputAnchor(
        num_clusters=2,
        num_channels=2,
        pred_len=2,
        cfg={"enable": True},
    )

    out = apply_moe_output_anchor_experts(
        pred,
        base_pred_bch=pred,
        x_bcl=torch.zeros(1, 2, 2),
        query_start_abs_b=torch.tensor([0]),
        input_len=0,
        moe_cfg={"learnable_output_anchor": True},
        moe_enable=False,
        learnable_output_anchor=module,
        cluster_id_c=cluster_id_c,
    )

    assert torch.allclose(out, pred)


def test_learnable_output_anchor_learns_cluster_channel_horizon_adjustments() -> None:
    pred = torch.zeros(1, 2, 2, requires_grad=True)
    stat_anchor_pc = torch.tensor([[10.0, 20.0]], dtype=torch.float32)
    cluster_id_c = torch.tensor([0, 1], dtype=torch.long)
    module = ClusterwiseLearnableOutputAnchor(
        num_clusters=2,
        num_channels=2,
        pred_len=2,
        cfg={
            "enable": True,
            "max_scale_delta": 1.0,
            "learn_bias": True,
            "max_bias": 1.0,
            "scale_parameterization": "channel_horizon",
            "bias_parameterization": "channel_horizon",
        },
    )
    module.stat_scale_delta_raw[0].data[0, 0] = 0.5
    module.bias_raw[1].data[1, 1] = 0.25

    out = apply_moe_output_anchor_experts(
        pred,
        base_pred_bch=pred,
        x_bcl=torch.zeros(1, 2, 2),
        query_start_abs_b=torch.tensor([0]),
        input_len=0,
        moe_cfg={
            "learnable_output_anchor": {
                "enable": True,
                "max_scale_delta": 1.0,
                "learn_bias": True,
                "max_bias": 1.0,
            },
            "train_stat_anchor_expert": {"enable": True, "alpha": 0.5},
        },
        moe_enable=False,
        train_stat_anchor_pc=stat_anchor_pc,
        learnable_output_anchor=module,
        cluster_id_c=cluster_id_c,
    )
    loss = out.sum()
    loss.backward()

    expected = torch.tensor([[[5.0, 5.0], [10.0, 10.0]]], dtype=torch.float32)
    expected[0, 0, 0] += torch.tanh(torch.tensor(0.5)).item() * 5.0
    expected[0, 1, 1] += torch.tanh(torch.tensor(0.25)).item()
    assert torch.allclose(out.detach(), expected, atol=1.0e-6)
    assert module.stat_scale_delta_raw[0].grad is not None
    assert module.stat_scale_delta_raw[0].grad[0, 0].abs().item() > 0.0
    assert module.bias_raw[1].grad is not None
    assert module.bias_raw[1].grad[1, 1].abs().item() > 0.0


def test_learnable_output_anchor_defaults_to_channel_shared_scale() -> None:
    pred = torch.zeros(1, 2, 3)
    stat_anchor_pc = torch.tensor([[10.0, 20.0]], dtype=torch.float32)
    cluster_id_c = torch.tensor([0, 1], dtype=torch.long)
    module = ClusterwiseLearnableOutputAnchor(
        num_clusters=2,
        num_channels=2,
        pred_len=3,
        cfg={"enable": True, "max_scale_delta": 1.0},
    )
    module.stat_scale_delta_raw[0].data[0, 0] = 0.5

    out = apply_moe_output_anchor_experts(
        pred,
        base_pred_bch=pred,
        x_bcl=torch.zeros(1, 2, 2),
        query_start_abs_b=torch.tensor([0]),
        input_len=0,
        moe_cfg={
            "learnable_output_anchor": {"enable": True, "max_scale_delta": 1.0},
            "train_stat_anchor_expert": {"enable": True, "alpha": 0.5},
        },
        moe_enable=False,
        train_stat_anchor_pc=stat_anchor_pc,
        learnable_output_anchor=module,
        cluster_id_c=cluster_id_c,
    )

    assert module.scale_parameterization == "channel"
    assert tuple(module.stat_scale_delta_raw[0].shape) == (2, 1)
    assert out[0, 0, 0].item() == pytest.approx(out[0, 0, 1].item())
    assert out[0, 0, 1].item() == pytest.approx(out[0, 0, 2].item())


def test_learnable_output_anchor_temporal_basis_zero_init_preserves_static_anchor() -> None:
    pred = torch.full((1, 2, 4), 2.0)
    stat_anchor_pc = torch.tensor([[10.0, 20.0]], dtype=torch.float32)
    cluster_id_c = torch.tensor([0, 1], dtype=torch.long)
    module = ClusterwiseLearnableOutputAnchor(
        num_clusters=2,
        num_channels=2,
        pred_len=4,
        cfg={"enable": True, "scale_temporal_basis_rank": 2},
    )

    static_out = apply_moe_output_anchor_experts(
        pred,
        base_pred_bch=pred,
        x_bcl=torch.zeros(1, 2, 2),
        query_start_abs_b=torch.tensor([0]),
        input_len=0,
        moe_cfg={"train_stat_anchor_expert": {"enable": True, "alpha": 0.5}},
        moe_enable=False,
        train_stat_anchor_pc=stat_anchor_pc,
    )
    learned_out = apply_moe_output_anchor_experts(
        pred,
        base_pred_bch=pred,
        x_bcl=torch.zeros(1, 2, 2),
        query_start_abs_b=torch.tensor([0]),
        input_len=0,
        moe_cfg={
            "learnable_output_anchor": {"enable": True, "scale_temporal_basis_rank": 2},
            "train_stat_anchor_expert": {"enable": True, "alpha": 0.5},
        },
        moe_enable=False,
        train_stat_anchor_pc=stat_anchor_pc,
        learnable_output_anchor=module,
        cluster_id_c=cluster_id_c,
    )

    assert module.scale_temporal_basis_rank == 2
    assert tuple(module.scale_temporal_basis_rh.shape) == (2, 4)
    assert torch.allclose(learned_out, static_out)


def test_learnable_output_anchor_temporal_basis_learns_horizon_shape() -> None:
    pred = torch.zeros(1, 2, 4, requires_grad=True)
    stat_anchor_pc = torch.tensor([[10.0, 20.0]], dtype=torch.float32)
    cluster_id_c = torch.tensor([0, 1], dtype=torch.long)
    module = ClusterwiseLearnableOutputAnchor(
        num_clusters=2,
        num_channels=2,
        pred_len=4,
        cfg={"enable": True, "max_scale_delta": 1.0, "scale_temporal_basis_rank": 2},
    )
    module.stat_scale_temporal_coef_raw[0].data[0, 0] = 0.5

    out = apply_moe_output_anchor_experts(
        pred,
        base_pred_bch=pred,
        x_bcl=torch.zeros(1, 2, 2),
        query_start_abs_b=torch.tensor([0]),
        input_len=0,
        moe_cfg={
            "learnable_output_anchor": {
                "enable": True,
                "max_scale_delta": 1.0,
                "scale_temporal_basis_rank": 2,
            },
            "train_stat_anchor_expert": {"enable": True, "alpha": 0.5},
        },
        moe_enable=False,
        train_stat_anchor_pc=stat_anchor_pc,
        learnable_output_anchor=module,
        cluster_id_c=cluster_id_c,
    )
    loss = out[:, 0, 0].sum()
    loss.backward()

    static_first_channel = torch.full((4,), 5.0)
    temporal_raw_h = 0.5 * module.scale_temporal_basis_rh[0]
    expected_first_channel = static_first_channel + torch.tanh(temporal_raw_h) * static_first_channel
    assert torch.allclose(out.detach()[0, 0], expected_first_channel, atol=1.0e-6)
    assert not torch.allclose(out.detach()[0, 0, :1], out.detach()[0, 0, 1:2])
    assert torch.allclose(out.detach()[0, 1], torch.full((4,), 10.0))
    assert module.stat_scale_temporal_coef_raw[0].grad is not None
    assert module.stat_scale_temporal_coef_raw[0].grad[0, 0].abs().item() > 0.0


def test_learnable_output_anchor_history_trend_learns_sample_conditioned_correction() -> None:
    pred = torch.zeros(2, 1, 4, requires_grad=True)
    x_bcl = torch.tensor(
        [
            [[0.0, 1.0, 2.0, 3.0]],
            [[3.0, 3.0, 3.0, 3.0]],
        ]
    )
    cluster_id_c = torch.tensor([0], dtype=torch.long)
    module = ClusterwiseLearnableOutputAnchor(
        num_clusters=1,
        num_channels=1,
        pred_len=4,
        cfg={
            "enable": True,
            "learn_history_trend": True,
            "max_history_trend_delta": 1.0,
            "history_trend_window": 4,
            "history_trend_projection": "linear",
        },
    )
    module.history_trend_delta_raw[0].data[0, 0] = 1.0

    out = apply_moe_output_anchor_experts(
        pred,
        base_pred_bch=pred,
        x_bcl=x_bcl,
        query_start_abs_b=torch.tensor([0, 0]),
        input_len=4,
        moe_cfg={
            "learnable_output_anchor": {
                "enable": True,
                "learn_history_trend": True,
                "max_history_trend_delta": 1.0,
                "history_trend_window": 4,
                "history_trend_projection": "linear",
            },
        },
        moe_enable=False,
        learnable_output_anchor=module,
        cluster_id_c=cluster_id_c,
    )
    loss = out.sum()
    loss.backward()

    expected_trend = torch.tensor(1.5)
    expected_h = torch.tanh(torch.tensor(1.0)) * expected_trend * torch.linspace(0.25, 1.0, 4)
    assert torch.allclose(out.detach()[0, 0], expected_h, atol=1.0e-6)
    assert torch.allclose(out.detach()[1, 0], torch.zeros(4))
    assert module.history_trend_delta_raw[0].grad is not None
    assert module.history_trend_delta_raw[0].grad[0, 0].abs().item() > 0.0


def test_learnable_output_anchor_history_trend_supports_recent_level_feature() -> None:
    pred = torch.zeros(2, 1, 3)
    x_bcl = torch.tensor(
        [
            [[1.0, 2.0, 4.0, 8.0]],
            [[0.0, 0.0, 0.0, 0.0]],
        ]
    )
    cluster_id_c = torch.tensor([0], dtype=torch.long)
    module = ClusterwiseLearnableOutputAnchor(
        num_clusters=1,
        num_channels=1,
        pred_len=3,
        cfg={
            "enable": True,
            "learn_history_trend": True,
            "max_history_trend_delta": 1.0,
            "history_trend_window": 2,
            "history_trend_feature": "recent_level",
            "history_trend_projection": "constant",
        },
    )
    module.history_trend_delta_raw[0].data[0, 0] = 1.0

    out = apply_moe_output_anchor_experts(
        pred,
        base_pred_bch=pred,
        x_bcl=x_bcl,
        query_start_abs_b=torch.tensor([0, 0]),
        input_len=4,
        moe_cfg={"learnable_output_anchor": {"enable": True}},
        moe_enable=False,
        learnable_output_anchor=module,
        cluster_id_c=cluster_id_c,
    )

    expected_level = torch.tensor(6.0)
    expected = torch.full((3,), torch.tanh(torch.tensor(1.0)) * expected_level)
    assert torch.allclose(out.detach()[0, 0], expected, atol=1.0e-6)
    assert torch.allclose(out.detach()[1, 0], torch.zeros(3))


def test_learnable_output_anchor_history_trend_supports_mean_abs_diff_feature() -> None:
    pred = torch.zeros(2, 1, 3)
    x_bcl = torch.tensor(
        [
            [[0.0, 1.0, 3.0, 6.0]],
            [[5.0, 5.0, 5.0, 5.0]],
        ]
    )
    cluster_id_c = torch.tensor([0], dtype=torch.long)
    module = ClusterwiseLearnableOutputAnchor(
        num_clusters=1,
        num_channels=1,
        pred_len=3,
        cfg={
            "enable": True,
            "learn_history_trend": True,
            "max_history_trend_delta": 1.0,
            "history_trend_window": 4,
            "history_trend_feature": "mean_abs_diff",
            "history_trend_projection": "linear",
        },
    )
    module.history_trend_delta_raw[0].data[0, 0] = 1.0

    out = apply_moe_output_anchor_experts(
        pred,
        base_pred_bch=pred,
        x_bcl=x_bcl,
        query_start_abs_b=torch.tensor([0, 0]),
        input_len=4,
        moe_cfg={"learnable_output_anchor": {"enable": True}},
        moe_enable=False,
        learnable_output_anchor=module,
        cluster_id_c=cluster_id_c,
    )

    expected_volatility = torch.tensor(2.0)
    expected = torch.tanh(torch.tensor(1.0)) * expected_volatility * torch.linspace(1.0 / 3.0, 1.0, 3)
    assert torch.allclose(out.detach()[0, 0], expected, atol=1.0e-6)
    assert torch.allclose(out.detach()[1, 0], torch.zeros(3))


def test_learnable_output_anchor_history_trend_supports_recent_slope_feature() -> None:
    pred = torch.zeros(2, 1, 3)
    x_bcl = torch.tensor(
        [
            [[0.0, 2.0, 2.0, 6.0]],
            [[5.0, 5.0, 5.0, 5.0]],
        ]
    )
    cluster_id_c = torch.tensor([0], dtype=torch.long)
    module = ClusterwiseLearnableOutputAnchor(
        num_clusters=1,
        num_channels=1,
        pred_len=3,
        cfg={
            "enable": True,
            "learn_history_trend": True,
            "max_history_trend_delta": 1.0,
            "history_trend_window": 4,
            "history_trend_feature": "recent_slope",
            "history_trend_projection": "linear",
        },
    )
    module.history_trend_delta_raw[0].data[0, 0] = 1.0

    out = apply_moe_output_anchor_experts(
        pred,
        base_pred_bch=pred,
        x_bcl=x_bcl,
        query_start_abs_b=torch.tensor([0, 0]),
        input_len=4,
        moe_cfg={"learnable_output_anchor": {"enable": True}},
        moe_enable=False,
        learnable_output_anchor=module,
        cluster_id_c=cluster_id_c,
    )

    expected_fitted_drift = torch.tensor(5.4)
    expected = torch.tanh(torch.tensor(1.0)) * expected_fitted_drift * torch.linspace(1.0 / 3.0, 1.0, 3)
    assert torch.allclose(out.detach()[0, 0], expected, atol=1.0e-6)
    assert torch.allclose(out.detach()[1, 0], torch.zeros(3))


def test_learnable_output_anchor_cluster_state_includes_temporal_basis_coefficients() -> None:
    module = ClusterwiseLearnableOutputAnchor(
        num_clusters=2,
        num_channels=2,
        pred_len=4,
        cfg={"enable": True, "scale_temporal_basis_rank": 2},
    )
    module.stat_scale_temporal_coef_raw[1].data[1, 0] = 0.75
    module.residual_scale_temporal_coef_raw[1].data[1, 1] = -0.25
    state = module.get_cluster_state(1)

    restored = ClusterwiseLearnableOutputAnchor(
        num_clusters=2,
        num_channels=2,
        pred_len=4,
        cfg={"enable": True, "scale_temporal_basis_rank": 2},
    )
    restored.load_cluster_state(1, state)

    assert restored.stat_scale_temporal_coef_raw[1][1, 0].item() == pytest.approx(0.75)
    assert restored.residual_scale_temporal_coef_raw[1][1, 1].item() == pytest.approx(-0.25)


def test_learnable_output_anchor_channel_mask_falls_back_to_static_channel() -> None:
    pred = torch.zeros(1, 2, 2)
    stat_anchor_pc = torch.tensor([[10.0, 20.0]], dtype=torch.float32)
    cluster_id_c = torch.tensor([0, 1], dtype=torch.long)
    module = ClusterwiseLearnableOutputAnchor(
        num_clusters=2,
        num_channels=2,
        pred_len=2,
        cfg={"enable": True, "max_scale_delta": 1.0},
    )
    module.stat_scale_delta_raw[0].data[0, 0] = 0.5
    module.stat_scale_delta_raw[1].data[1, 0] = 0.5
    module.set_active_channel_mask(torch.tensor([1.0, 0.0]))

    out = apply_moe_output_anchor_experts(
        pred,
        base_pred_bch=pred,
        x_bcl=torch.zeros(1, 2, 2),
        query_start_abs_b=torch.tensor([0]),
        input_len=0,
        moe_cfg={
            "learnable_output_anchor": {"enable": True, "max_scale_delta": 1.0},
            "train_stat_anchor_expert": {"enable": True, "alpha": 0.5},
        },
        moe_enable=False,
        train_stat_anchor_pc=stat_anchor_pc,
        learnable_output_anchor=module,
        cluster_id_c=cluster_id_c,
    )
    static = apply_moe_output_anchor_experts(
        pred,
        base_pred_bch=pred,
        x_bcl=torch.zeros(1, 2, 2),
        query_start_abs_b=torch.tensor([0]),
        input_len=0,
        moe_cfg={"train_stat_anchor_expert": {"enable": True, "alpha": 0.5}},
        moe_enable=False,
        train_stat_anchor_pc=stat_anchor_pc,
    )

    assert out[0, 0, 0].item() > static[0, 0, 0].item()
    assert torch.allclose(out[:, 1, :], static[:, 1, :])


def test_learnable_output_anchor_channel_horizon_mask_falls_back_to_static_steps() -> None:
    pred = torch.zeros(1, 2, 3)
    stat_delta = torch.ones_like(pred)
    cluster_id_c = torch.tensor([0, 0], dtype=torch.long)
    module = ClusterwiseLearnableOutputAnchor(
        num_clusters=1,
        num_channels=2,
        pred_len=3,
        cfg={"enable": True, "max_scale_delta": 1.0},
    )
    module.stat_scale_delta_raw[0].data[:, 0] = 0.5
    module.set_active_channel_horizon_mask(
        torch.tensor(
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
    )

    out = module(pred, cluster_id_c=cluster_id_c, stat_delta_bch=stat_delta)

    assert out[0, 0, 0].item() > 0.0
    assert out[0, 0, 1].item() == pytest.approx(0.0)
    assert out[0, 0, 2].item() > 0.0
    assert out[0, 1, 0].item() == pytest.approx(0.0)
    assert out[0, 1, 1].item() > 0.0
    assert out[0, 1, 2].item() == pytest.approx(0.0)


def test_learnable_output_anchor_channel_horizon_mask_loads_old_state_dict() -> None:
    module = ClusterwiseLearnableOutputAnchor(
        num_clusters=1,
        num_channels=2,
        pred_len=3,
        cfg={"enable": True},
    )
    old_state = dict(module.state_dict())
    old_state.pop("active_channel_horizon_mask_ch")

    restored = ClusterwiseLearnableOutputAnchor(
        num_clusters=1,
        num_channels=2,
        pred_len=3,
        cfg={"enable": True},
    )
    loaded = restored.load_state_dict(old_state, strict=False)

    assert loaded.missing_keys == ["active_channel_horizon_mask_ch"]
    assert torch.allclose(restored.active_channel_horizon_mask_ch, torch.ones(2, 3))


def test_learnable_output_anchor_uses_dedicated_optimizer_group() -> None:
    gate_param = torch.nn.Parameter(torch.tensor([1.0]))
    anchor_param = torch.nn.Parameter(torch.tensor([2.0]))

    groups = _make_cluster_optimizer_param_groups(
        base_params=[],
        gate_params=[gate_param],
        pred_residual_params=[],
        dynamic_lambda_params=[],
        learnable_lambda_params=[],
        learnable_anchor_params=[anchor_param],
        base_weight_decay=0.01,
        moe_weight_decay=0.02,
        pred_residual_weight_decay=None,
        learnable_anchor_weight_decay=0.03,
        learnable_anchor_lr=1.0e-4,
    )

    assert len(groups) == 2
    assert any(anchor_param is param for param in groups[1]["params"])
    assert all(anchor_param is not param for param in groups[0]["params"])
    assert groups[1]["weight_decay"] == pytest.approx(0.03)
    assert groups[1]["lr"] == pytest.approx(1.0e-4)


def test_learnable_output_anchor_refiner_summary_adopts_clear_val_gain() -> None:
    summary = _summarize_learnable_output_anchor_refiner(
        static_mse=1.0,
        static_mae=0.50,
        refined_mse=0.98,
        refined_mae=0.49,
        cfg={"enable": True, "adoption": {"selection_metric": "mse", "min_rel_improvement": 0.01}},
        skip_test=True,
        num_channels=3,
    )

    assert summary["adopted"] is True
    assert summary["final_eval_uses_learnable"] is True
    assert summary["adopted_channel_count"] == 3
    assert summary["metric_gain"] == pytest.approx(0.02)
    assert summary["required_gain"] == pytest.approx(0.01)
    assert summary["test_read"] is False


def test_learnable_output_anchor_refiner_summary_reports_channel_adoption_mask() -> None:
    summary = _summarize_learnable_output_anchor_refiner(
        static_mse=1.0,
        static_mae=0.50,
        refined_mse=0.98,
        refined_mae=0.49,
        cfg={"enable": True, "adoption": {"selection_metric": "mse", "adoption_scope": "channel"}},
        skip_test=True,
        num_channels=3,
        adopted_channel_mask=[True, False, True],
    )

    assert summary["adopted"] is True
    assert summary["adopted_channel_count"] == 2
    assert summary["adopted_channel_mask"] == [True, False, True]


def test_learnable_output_anchor_refiner_summary_reports_channel_horizon_adoption() -> None:
    summary = _summarize_learnable_output_anchor_refiner(
        static_mse=1.0,
        static_mae=0.50,
        refined_mse=0.86,
        refined_mae=0.48,
        cfg={"enable": True, "adoption": {"selection_metric": "mse", "adoption_scope": "channel_horizon"}},
        skip_test=True,
        num_channels=2,
        adopted_channel_horizon_mask=[
            [True, True, False, False],
            [False, False, True, True],
        ],
    )

    assert summary["adopted"] is True
    assert summary["adopted_channel_count"] == 2
    assert summary["adopted_channel_mask"] == [True, True]
    assert summary["adopted_channel_horizon_count"] == 4
    assert summary["adopted_channel_horizon_total"] == 8
    assert summary["adopted_horizon_count_per_channel"] == [2, 2]


def test_learnable_output_anchor_refiner_summary_preserves_unmasked_metrics() -> None:
    summary = _summarize_learnable_output_anchor_refiner(
        static_mse=1.0,
        static_mae=0.50,
        refined_mse=1.0,
        refined_mae=0.50,
        unmasked_refined_mse=0.98,
        unmasked_refined_mae=0.49,
        cfg={"enable": True, "adoption": {"selection_metric": "mse", "adoption_scope": "channel"}},
        skip_test=True,
        num_channels=2,
        adopted_channel_mask=[False, False],
    )

    assert summary["adopted"] is False
    assert summary["val_refined_mse"] == pytest.approx(1.0)
    assert summary["val_refined_mse_unmasked"] == pytest.approx(0.98)
    assert summary["val_refined_mae_unmasked"] == pytest.approx(0.49)


def test_learnable_output_anchor_refiner_summary_rejects_mae_regression() -> None:
    summary = _summarize_learnable_output_anchor_refiner(
        static_mse=1.0,
        static_mae=0.50,
        refined_mse=0.95,
        refined_mae=0.51,
        cfg={"enable": True, "adoption": {"selection_metric": "mse", "max_rel_mae_regression": 0.0}},
        skip_test=True,
        num_channels=2,
    )

    assert summary["adopted"] is False
    assert summary["final_eval_uses_learnable"] is False
    assert summary["adopted_channel_count"] == 0
    assert summary["adopted_channel_mask"] == [False, False]
    assert summary["fallback_reason"] == "val_refiner_did_not_clear_static_anchor_guard"


def test_learnable_output_anchor_refiner_summary_rejects_insufficient_mae_margin() -> None:
    summary = _summarize_learnable_output_anchor_refiner(
        static_mse=1.0,
        static_mae=0.50,
        refined_mse=0.95,
        refined_mae=0.4995,
        cfg={
            "enable": True,
            "adoption": {
                "selection_metric": "mse",
                "aggregate_min_abs_mae_improvement": 0.001,
            },
        },
        skip_test=True,
        num_channels=2,
    )

    assert summary["adopted"] is False
    assert summary["final_eval_uses_learnable"] is False
    assert summary["mae_gain"] == pytest.approx(0.0005)
    assert summary["required_mae_gain"] == pytest.approx(0.001)


def test_learnable_output_anchor_refiner_summary_preserves_default_mae_tolerance() -> None:
    summary = _summarize_learnable_output_anchor_refiner(
        static_mse=1.0,
        static_mae=0.50,
        refined_mse=0.95,
        refined_mae=0.51,
        cfg={
            "enable": True,
            "adoption": {
                "selection_metric": "mse",
                "aggregate_max_abs_mae_regression": 0.02,
            },
        },
        skip_test=True,
        num_channels=2,
    )

    assert summary["adopted"] is True
    assert summary["mae_gain"] == pytest.approx(-0.01)
    assert summary["required_mae_gain"] == pytest.approx(-0.02)


def test_learnable_output_anchor_refiner_summary_keeps_segment_mae_guard_strict() -> None:
    summary = _summarize_learnable_output_anchor_refiner(
        static_mse=1.0,
        static_mae=0.50,
        refined_mse=0.95,
        refined_mae=0.49,
        cfg={
            "enable": True,
            "adoption": {
                "selection_metric": "mse",
                "aggregate_max_abs_mae_regression": 0.02,
                "max_abs_mae_regression": 0.0,
                "min_positive_segments": 2,
            },
        },
        skip_test=True,
        num_channels=2,
        segment_metrics=[
            {"static_mse": 1.0, "static_mae": 0.50, "refined_mse": 0.95, "refined_mae": 0.49},
            {"static_mse": 1.0, "static_mae": 0.50, "refined_mse": 0.95, "refined_mae": 0.51},
        ],
    )

    assert summary["adopted"] is False
    assert summary["segment_guard"]["mae_regressed_segment_count"] == 1
    assert summary["segment_guard"]["passed"] is False


def test_learnable_output_anchor_refiner_summary_rejects_segment_degradation() -> None:
    summary = _summarize_learnable_output_anchor_refiner(
        static_mse=1.0,
        static_mae=0.50,
        refined_mse=0.98,
        refined_mae=0.49,
        cfg={"enable": True, "adoption": {"selection_metric": "mse", "max_segment_abs_degradation": 0.0}},
        skip_test=True,
        num_channels=2,
        segment_metrics=[
            {"static_mse": 1.0, "static_mae": 0.50, "refined_mse": 0.95, "refined_mae": 0.48},
            {"static_mse": 1.0, "static_mae": 0.50, "refined_mse": 1.01, "refined_mae": 0.50},
        ],
    )

    assert summary["adopted"] is False
    assert summary["final_eval_uses_learnable"] is False
    assert summary["segment_guard"]["applied"] is True
    assert summary["segment_guard"]["degraded_segment_count"] == 1
    assert summary["segment_guard"]["passed"] is False


def test_learnable_output_anchor_hybrid_mask_adds_safe_margin_channels() -> None:
    mask, diagnostics = _select_learnable_output_anchor_channel_mask(
        static_mse_c=torch.tensor([1.00, 1.00, 1.00]),
        refined_mse_c=torch.tensor([0.90, 0.94, 1.02]),
        static_mae_c=torch.tensor([0.50, 0.50, 0.50]),
        refined_mae_c=torch.tensor([0.49, 0.505, 0.49]),
        segment_channel_metrics=[
            {
                "static_mse_c": torch.tensor([1.00, 1.00, 1.00]),
                "refined_mse_c": torch.tensor([0.90, 0.94, 1.02]),
                "static_mae_c": torch.tensor([0.50, 0.50, 0.50]),
                "refined_mae_c": torch.tensor([0.49, 0.505, 0.49]),
            },
            {
                "static_mse_c": torch.tensor([1.00, 1.00, 1.00]),
                "refined_mse_c": torch.tensor([0.91, 0.95, 1.01]),
                "static_mae_c": torch.tensor([0.50, 0.50, 0.50]),
                "refined_mae_c": torch.tensor([0.49, 0.505, 0.49]),
            },
        ],
        adoption_cfg={
            "adoption_scope": "hybrid",
            "selection_metric": "mse",
            "max_abs_mae_regression": 0.0,
            "aggregate_max_abs_mae_regression": 0.002,
            "aggregate_min_abs_improvement": 0.05,
            "eval_segments": 2,
            "min_positive_segments": 2,
        },
    )

    assert mask.tolist() == [True, True, False]
    assert diagnostics["strict_channel_count"] == 1
    assert diagnostics["added_channel_count"] == 1
    assert diagnostics["aggregate"]["metric_gain"] == pytest.approx(0.053333, abs=1.0e-6)
    assert diagnostics["aggregate"]["passed"] is True


def test_learnable_output_anchor_frozen_periodic_source_copies_adoption_masks() -> None:
    cfg = {
        "enable": True,
        "scale_parameterization": "channel",
        "bias_parameterization": "channel",
    }
    source = ClusterwiseLearnableOutputAnchor(
        num_clusters=2, num_channels=3, pred_len=4, cfg=cfg
    )
    target = ClusterwiseLearnableOutputAnchor(
        num_clusters=2, num_channels=3, pred_len=4, cfg=cfg
    )
    source.set_active_channel_mask(torch.tensor([1.0, 0.0, 1.0]))
    source.set_active_channel_horizon_mask(
        torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
            ]
        )
    )

    summary = _copy_learnable_output_anchor_active_masks(target, source)

    assert target.active_channel_mask_c.tolist() == [1.0, 0.0, 1.0]
    assert target.active_channel_horizon_mask_ch.tolist() == source.active_channel_horizon_mask_ch.tolist()
    assert summary["active_channel_mask"] == [1.0, 0.0, 1.0]


def test_learnable_output_anchor_frozen_periodic_source_rejects_mask_shape_mismatch() -> None:
    source = ClusterwiseLearnableOutputAnchor(
        num_clusters=1, num_channels=3, pred_len=4, cfg={"enable": True}
    )
    target = ClusterwiseLearnableOutputAnchor(
        num_clusters=1, num_channels=2, pred_len=4, cfg={"enable": True}
    )

    with pytest.raises(ValueError, match="channel-mask shape mismatch"):
        _copy_learnable_output_anchor_active_masks(target, source)


def test_learnable_output_anchor_channel_horizon_mask_selects_stable_blocks() -> None:
    mask, diagnostics = _select_learnable_output_anchor_channel_horizon_mask(
        static_mse_ch=torch.ones(2, 4),
        refined_mse_ch=torch.tensor(
            [
                [0.80, 0.80, 1.10, 1.10],
                [1.05, 1.05, 0.70, 0.70],
            ]
        ),
        static_mae_ch=torch.full((2, 4), 0.50),
        refined_mae_ch=torch.tensor(
            [
                [0.49, 0.49, 0.52, 0.52],
                [0.51, 0.51, 0.48, 0.48],
            ]
        ),
        adoption_cfg={
            "adoption_scope": "channel_horizon",
            "selection_metric": "mse",
            "horizon_segments": 2,
            "min_abs_improvement": 0.05,
            "max_abs_mae_regression": 0.0,
            "aggregate_min_abs_improvement": 0.10,
            "aggregate_max_abs_mae_regression": 0.0,
        },
    )

    assert mask.tolist() == [[True, True, False, False], [False, False, True, True]]
    assert diagnostics["horizon_segments"] == 2
    assert diagnostics["adopted_channel_count"] == 2
    assert diagnostics["adopted_channel_horizon_count"] == 4
    assert diagnostics["aggregate"]["metric_gain"] == pytest.approx(0.125)
    assert diagnostics["aggregate"]["mae_improvement"] == pytest.approx(0.0075)
    assert diagnostics["aggregate"]["passed"] is True


def test_learnable_output_anchor_channel_horizon_mask_can_use_aggregate_segment_guard_only() -> None:
    segment_rows = [
        {
            "static_mse_ch": torch.ones(1, 2),
            "refined_mse_ch": torch.tensor([[1.01, 0.70]]),
            "static_mae_ch": torch.full((1, 2), 0.50),
            "refined_mae_ch": torch.tensor([[0.49, 0.49]]),
        },
        {
            "static_mse_ch": torch.ones(1, 2),
            "refined_mse_ch": torch.tensor([[0.70, 0.70]]),
            "static_mae_ch": torch.full((1, 2), 0.50),
            "refined_mae_ch": torch.tensor([[0.49, 0.49]]),
        },
    ]
    guarded_mask, guarded_diagnostics = _select_learnable_output_anchor_channel_horizon_mask(
        static_mse_ch=torch.ones(1, 2),
        refined_mse_ch=torch.tensor([[0.80, 0.80]]),
        static_mae_ch=torch.full((1, 2), 0.50),
        refined_mae_ch=torch.full((1, 2), 0.49),
        segment_channel_horizon_metrics=segment_rows,
        adoption_cfg={
            "adoption_scope": "channel_horizon",
            "horizon_segments": 2,
            "min_positive_segments": 2,
            "max_segment_abs_degradation": 0.0,
            "max_abs_mae_regression": 0.0,
        },
    )
    relaxed_mask, relaxed_diagnostics = _select_learnable_output_anchor_channel_horizon_mask(
        static_mse_ch=torch.ones(1, 2),
        refined_mse_ch=torch.tensor([[0.80, 0.80]]),
        static_mae_ch=torch.full((1, 2), 0.50),
        refined_mae_ch=torch.full((1, 2), 0.49),
        segment_channel_horizon_metrics=segment_rows,
        adoption_cfg={
            "adoption_scope": "channel_horizon",
            "horizon_segments": 2,
            "candidate_segment_guard": False,
            "min_positive_segments": 2,
            "max_segment_abs_degradation": 0.0,
            "max_abs_mae_regression": 0.0,
        },
    )

    assert guarded_mask.tolist() == [[False, True]]
    assert guarded_diagnostics["candidate_segment_guard"] is True
    assert relaxed_mask.tolist() == [[True, True]]
    assert relaxed_diagnostics["candidate_segment_guard"] is False
    assert relaxed_diagnostics["aggregate"]["segment_guard_passed"] is True


def test_learnable_output_anchor_hybrid_mask_reports_insufficient_mae_margin() -> None:
    mask, diagnostics = _select_learnable_output_anchor_channel_mask(
        static_mse_c=torch.tensor([1.00, 1.00]),
        refined_mse_c=torch.tensor([0.90, 0.90]),
        static_mae_c=torch.tensor([0.50, 0.50]),
        refined_mae_c=torch.tensor([0.4998, 0.4998]),
        segment_channel_metrics=[
            {
                "static_mse_c": torch.tensor([1.00, 1.00]),
                "refined_mse_c": torch.tensor([0.90, 0.90]),
                "static_mae_c": torch.tensor([0.50, 0.50]),
                "refined_mae_c": torch.tensor([0.4998, 0.4998]),
            },
            {
                "static_mse_c": torch.tensor([1.00, 1.00]),
                "refined_mse_c": torch.tensor([0.91, 0.91]),
                "static_mae_c": torch.tensor([0.50, 0.50]),
                "refined_mae_c": torch.tensor([0.4998, 0.4998]),
            },
        ],
        adoption_cfg={
            "adoption_scope": "hybrid",
            "selection_metric": "mse",
            "aggregate_min_abs_improvement": 0.05,
            "aggregate_min_abs_mae_improvement": 0.001,
            "max_abs_mae_regression": 0.0,
            "aggregate_max_abs_mae_regression": 0.0,
            "min_positive_segments": 2,
        },
    )

    assert mask.tolist() == [True, True]
    assert diagnostics["aggregate"]["metric_gain"] == pytest.approx(0.1)
    assert diagnostics["aggregate"]["mae_improvement"] == pytest.approx(0.0002, abs=1.0e-6)
    assert diagnostics["aggregate"]["required_mae_improvement"] == pytest.approx(0.001)
    assert diagnostics["aggregate"]["passed"] is False


def test_learnable_output_anchor_hybrid_mask_preserves_default_mae_tolerance() -> None:
    mask, diagnostics = _select_learnable_output_anchor_channel_mask(
        static_mse_c=torch.tensor([1.00, 1.00]),
        refined_mse_c=torch.tensor([0.90, 0.90]),
        static_mae_c=torch.tensor([0.50, 0.50]),
        refined_mae_c=torch.tensor([0.505, 0.505]),
        segment_channel_metrics=[],
        adoption_cfg={
            "adoption_scope": "hybrid",
            "selection_metric": "mse",
            "aggregate_min_abs_improvement": 0.05,
            "max_abs_mae_regression": 0.01,
            "aggregate_max_abs_mae_regression": 0.01,
        },
    )

    assert mask.tolist() == [True, True]
    assert diagnostics["aggregate"]["mae_improvement"] == pytest.approx(-0.005, abs=1.0e-6)
    assert diagnostics["aggregate"]["required_mae_improvement"] == pytest.approx(-0.01)
    assert diagnostics["aggregate"]["passed"] is True


def test_learnable_output_anchor_hybrid_mask_keeps_segment_mae_guard_strict() -> None:
    mask, diagnostics = _select_learnable_output_anchor_channel_mask(
        static_mse_c=torch.tensor([1.00, 1.00]),
        refined_mse_c=torch.tensor([0.90, 0.90]),
        static_mae_c=torch.tensor([0.50, 0.50]),
        refined_mae_c=torch.tensor([0.49, 0.49]),
        segment_channel_metrics=[
            {
                "static_mse_c": torch.tensor([1.00, 1.00]),
                "refined_mse_c": torch.tensor([0.90, 0.90]),
                "static_mae_c": torch.tensor([0.50, 0.50]),
                "refined_mae_c": torch.tensor([0.49, 0.49]),
            },
            {
                "static_mse_c": torch.tensor([1.00, 1.00]),
                "refined_mse_c": torch.tensor([0.90, 0.90]),
                "static_mae_c": torch.tensor([0.50, 0.50]),
                "refined_mae_c": torch.tensor([0.51, 0.51]),
            },
        ],
        adoption_cfg={
            "adoption_scope": "hybrid",
            "selection_metric": "mse",
            "aggregate_min_abs_improvement": 0.05,
            "max_abs_mae_regression": 0.0,
            "aggregate_max_abs_mae_regression": 0.02,
            "min_positive_segments": 2,
        },
    )

    assert mask.tolist() == [False, False]
    assert diagnostics["strict_channel_count"] == 0
    assert diagnostics["added_channel_count"] == 0
    assert diagnostics["aggregate"]["passed"] is False


def test_moe_history_anchor_expert_applies_anchor_when_enabled() -> None:
    observed = torch.arange(12, dtype=torch.float32).view(12, 1)
    pred = torch.zeros(1, 1, 2)
    out = apply_moe_history_anchor_expert(
        pred,
        base_pred_bch=pred,
        observed_history_tc=observed,
        query_start_abs_b=torch.tensor([4]),
        input_len=4,
        cfg={"enable": True, "lags": [4], "alpha": 1.0, "blend_target": "prediction"},
    )

    assert torch.allclose(out, torch.tensor([[[4.0, 5.0]]]))


def test_moe_history_anchor_expert_accepts_channel_alpha() -> None:
    observed = torch.stack(
        [
            torch.arange(12, dtype=torch.float32),
            torch.arange(100, 112, dtype=torch.float32),
        ],
        dim=1,
    )
    pred = torch.zeros(1, 2, 2)
    out = apply_moe_history_anchor_expert(
        pred,
        base_pred_bch=pred,
        observed_history_tc=observed,
        query_start_abs_b=torch.tensor([4]),
        input_len=4,
        cfg={
            "enable": True,
            "lags": [4],
            "alpha": 0.0,
            "alpha_by_channel": [1.0, 0.5],
            "blend_target": "prediction",
        },
    )

    assert torch.allclose(out, torch.tensor([[[4.0, 5.0], [52.0, 52.5]]]))


def test_train_stat_anchor_expert_uses_train_phase_statistics_only() -> None:
    train = torch.arange(6, dtype=torch.float32).view(6, 1)
    future = torch.full((6, 1), 1000.0)
    data = torch.cat([train, future], dim=0)
    table, counts = build_train_phase_anchor_table(data, train_end=6, period=3)
    pred = torch.zeros(1, 1, 3)

    out = apply_train_stat_anchor_expert(
        pred,
        base_pred_bch=pred,
        query_start_abs_b=torch.tensor([6]),
        input_len=2,
        stat_anchor_pc=table,
        cfg={"enable": True, "alpha": 1.0, "blend_target": "prediction"},
    )

    assert torch.equal(counts, torch.tensor([2, 2, 2]))
    assert torch.allclose(table.squeeze(1), torch.tensor([1.5, 2.5, 3.5]))
    assert torch.allclose(out, torch.tensor([[[3.5, 1.5, 2.5]]]))


def test_build_train_stat_anchor_from_config_uses_model_adapter_prefix() -> None:
    train = torch.arange(6, dtype=torch.float32).view(6, 1)
    future = torch.full((6, 1), 1000.0)
    data = torch.cat([train, future], dim=0)

    table, counts, summary = build_train_stat_anchor_from_config(
        data,
        train_end=6,
        input_len=2,
        pred_len=3,
        cfg={"enable": True, "period": 3, "mode": "phase_mean", "alpha": 0.25},
        prefix="model.train_stat_adapter",
    )

    assert torch.equal(counts, torch.tensor([2, 2, 2]))
    assert torch.allclose(table.squeeze(1), torch.tensor([1.5, 2.5, 3.5]))
    assert summary == {
        "enable": True,
        "period": 3,
        "mode": "phase_mean",
        "reference": "last",
        "source_split": "train",
        "train_end": 6,
        "min_count": 2,
        "max_count": 2,
        "alpha": 0.25,
        "blend_target": "prediction",
    }


def test_train_stat_anchor_expert_accepts_channel_alpha() -> None:
    table = torch.tensor([[10.0, 20.0]])
    pred = torch.zeros(1, 2, 1)

    out = apply_train_stat_anchor_expert(
        pred,
        base_pred_bch=pred,
        query_start_abs_b=torch.tensor([0]),
        input_len=1,
        stat_anchor_pc=table,
        cfg={
            "enable": True,
            "alpha": 0.0,
            "alpha_by_channel": [1.0, 0.25],
            "blend_target": "prediction",
        },
    )

    assert torch.allclose(out, torch.tensor([[[10.0], [5.0]]]))


def test_train_stat_anchor_expert_accepts_channel_horizon_alpha() -> None:
    table = torch.tensor([[10.0]])
    pred = torch.zeros(1, 1, 4)

    out = apply_train_stat_anchor_expert(
        pred,
        base_pred_bch=pred,
        query_start_abs_b=torch.tensor([0]),
        input_len=1,
        stat_anchor_pc=table,
        cfg={
            "enable": True,
            "alpha": 0.0,
            "alpha_by_channel_horizon": [[1.0, 0.25]],
            "alpha_horizon_segments": 2,
            "blend_target": "prediction",
        },
    )

    assert torch.allclose(out, torch.tensor([[[10.0, 10.0, 2.5, 2.5]]]))


def test_train_stat_anchor_expert_can_treat_prediction_as_anchor_residual() -> None:
    table = torch.tensor([[10.0]])
    pred = torch.full((1, 1, 2), 2.0)

    out = apply_train_stat_anchor_expert(
        pred,
        base_pred_bch=pred,
        query_start_abs_b=torch.tensor([0]),
        input_len=1,
        stat_anchor_pc=table,
        cfg={
            "enable": True,
            "alpha": 0.5,
            "blend_target": "prediction",
            "combine_mode": "anchor_plus_prediction",
        },
    )

    assert torch.allclose(out, torch.tensor([[[11.0, 11.0]]]))


def test_train_stat_input_centering_subtracts_phase_profile_from_input() -> None:
    table = torch.tensor([[10.0], [20.0], [30.0]])
    x = torch.tensor([[[11.0, 22.0, 33.0]]])

    out = apply_train_stat_input_centering(
        x,
        query_start_abs_b=torch.tensor([0]),
        stat_anchor_pc=table,
        cfg={"enable": True, "input_center": True, "mode": "phase_mean"},
    )

    assert torch.allclose(out, torch.tensor([[[1.0, 2.0, 3.0]]]))


def test_select_channel_anchor_scales_uses_validation_error_per_channel() -> None:
    base = torch.zeros(1, 2, 2)
    anchor = torch.tensor([[[2.0, 2.0], [4.0, 4.0]]])
    target = torch.tensor([[[2.0, 2.0], [0.0, 0.0]]])

    scales, scores = select_channel_anchor_scales(
        base,
        anchor,
        target,
        metric="mse",
        max_scale=1.0,
        steps=3,
    )

    assert torch.allclose(scales, torch.tensor([1.0, 0.0]))
    assert torch.allclose(scores, torch.tensor([0.0, 0.0]))


def test_select_channel_anchor_scales_chunking_matches_full_selection() -> None:
    base = torch.zeros(2, 7, 3)
    channel_scale = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0, 0.25, 0.5]).view(1, 7, 1)
    anchor = torch.ones_like(base) * 4.0
    target = anchor * channel_scale

    full_scales, full_scores = select_channel_anchor_scales(
        base,
        anchor,
        target,
        metric="mse",
        max_scale=1.0,
        steps=5,
    )
    chunked_scales, chunked_scores = select_channel_anchor_scales(
        base,
        anchor,
        target,
        metric="mse",
        max_scale=1.0,
        steps=5,
        channel_chunk_size=3,
    )

    assert torch.allclose(chunked_scales, full_scales)
    assert torch.allclose(chunked_scores, full_scores)


def test_select_channel_anchor_scales_sample_chunking_matches_full_selection() -> None:
    base = torch.zeros(3, 4, 5)
    anchor = torch.ones_like(base) * 2.0
    target = anchor * torch.tensor([0.0, 0.5, 1.0, 0.25]).view(1, 4, 1)

    full_scales, full_scores = select_channel_anchor_scales(
        base,
        anchor,
        target,
        metric="mse",
        max_scale=1.0,
        steps=5,
    )
    chunked_scales, chunked_scores = select_channel_anchor_scales(
        base,
        anchor,
        target,
        metric="mse",
        max_scale=1.0,
        steps=5,
        sample_chunk_size=1,
    )

    assert torch.allclose(chunked_scales, full_scales)
    assert torch.allclose(chunked_scores, full_scores)


def test_select_channel_horizon_anchor_scales_can_vary_by_segment() -> None:
    base = torch.zeros(1, 1, 4)
    anchor = torch.full((1, 1, 4), 2.0)
    target = torch.tensor([[[2.0, 2.0, 0.0, 0.0]]])

    scales, scores = select_channel_horizon_anchor_scales(
        base,
        anchor,
        target,
        metric="mse",
        max_scale=1.0,
        steps=3,
        segments=2,
    )

    assert torch.allclose(scales, torch.tensor([[1.0, 0.0]]))
    assert torch.allclose(scores, torch.tensor([[0.0, 0.0]]))


def test_select_channel_horizon_anchor_scales_chunking_matches_full_selection() -> None:
    base = torch.zeros(2, 7, 4)
    anchor = torch.ones_like(base) * 2.0
    target = torch.zeros_like(base)
    target[:, 0, :2] = 2.0
    target[:, 1, 2:] = 1.0
    target[:, 2, :] = 1.0

    full_scales, full_scores = select_channel_horizon_anchor_scales(
        base,
        anchor,
        target,
        metric="mse",
        max_scale=1.0,
        steps=3,
        segments=2,
    )
    chunked_scales, chunked_scores = select_channel_horizon_anchor_scales(
        base,
        anchor,
        target,
        metric="mse",
        max_scale=1.0,
        steps=3,
        segments=2,
        channel_chunk_size=3,
    )

    assert torch.allclose(chunked_scales, full_scales)
    assert torch.allclose(chunked_scores, full_scores)


def test_train_stat_anchor_expert_can_use_train_delta_template_with_input_reference() -> None:
    train = torch.arange(10, dtype=torch.float32).view(10, 1)
    future = torch.full((6, 1), 1000.0)
    data = torch.cat([train, future], dim=0)
    table, counts = build_train_phase_delta_anchor_table(
        data,
        train_end=10,
        input_len=2,
        pred_len=2,
        period=2,
        reference="last",
    )
    pred = torch.zeros(1, 1, 2)
    x = torch.tensor([[[20.0, 30.0]]])

    out = apply_train_stat_anchor_expert(
        pred,
        base_pred_bch=pred,
        x_bcl=x,
        query_start_abs_b=torch.tensor([10]),
        input_len=2,
        stat_anchor_pc=table,
        cfg={"enable": True, "alpha": 1.0, "blend_target": "prediction", "mode": "phase_delta", "reference": "last"},
    )

    assert torch.equal(counts, torch.tensor([4, 3]))
    assert torch.allclose(table[:, :, 0], torch.tensor([[1.0, 2.0], [1.0, 2.0]]))
    assert torch.allclose(out, torch.tensor([[[31.0, 32.0]]]))


def test_train_residual_anchor_table_uses_train_residuals_by_phase() -> None:
    base = torch.tensor(
        [
            [[1.0, 2.0]],
            [[10.0, 20.0]],
            [[100.0, 200.0]],
        ]
    )
    target = torch.tensor(
        [
            [[2.0, 4.0]],
            [[13.0, 24.0]],
            [[105.0, 206.0]],
        ]
    )

    table, counts = build_train_phase_residual_anchor_table(
        base,
        target,
        query_start_abs_n=torch.tensor([0, 1, 2]),
        input_len=2,
        period=2,
    )

    assert counts.tolist() == [2, 1]
    assert torch.allclose(table[0], torch.tensor([[3.0], [4.0]]))
    assert torch.allclose(table[1], torch.tensor([[3.0], [4.0]]))


def test_train_residual_anchor_expert_adds_phase_residual_template() -> None:
    pred = torch.zeros(2, 1, 2)
    table = torch.tensor(
        [
            [[1.0], [2.0]],
            [[10.0], [20.0]],
        ]
    )

    out = apply_train_residual_anchor_expert(
        pred,
        base_pred_bch=pred,
        query_start_abs_b=torch.tensor([0, 1]),
        input_len=2,
        residual_anchor_phc=table,
        cfg={"enable": True, "alpha": 0.5},
    )

    assert torch.allclose(out, torch.tensor([[[0.5, 1.0]], [[5.0, 10.0]]]))


def test_train_residual_anchor_table_loader_can_stack_on_train_stat_anchor() -> None:
    x = torch.zeros(1, 1, 4)
    y = torch.tensor([[[13.0, 25.0]]])
    idx = torch.tensor([0])
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y, idx), batch_size=1)
    stat_anchor = torch.zeros(4, 1)
    stat_anchor[0, 0] = 10.0
    stat_anchor[1, 0] = 20.0

    table, counts, n_windows = build_train_residual_anchor_table_from_loader(
        model=_ZeroBackbone(pred_len=2),
        loader=loader,
        cluster_id_c=torch.tensor([0]),
        device=torch.device("cpu"),
        history_anchor_cfg={},
        observed_history_tc=None,
        input_len=4,
        eval_start=0,
        period=4,
        train_stat_anchor_pc=stat_anchor,
        train_stat_anchor_cfg={
            "enable": True,
            "period": 4,
            "alpha": 1.0,
            "mode": "phase_mean",
            "blend_target": "prediction",
        },
    )

    assert n_windows == 1
    assert counts[0].item() == 1
    assert torch.allclose(table[0, :, 0], torch.tensor([3.0, 5.0]))


def test_train_residual_anchor_table_loader_streams_batches_without_full_cat(monkeypatch: pytest.MonkeyPatch) -> None:
    x = torch.zeros(2, 1, 4)
    y = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
    idx = torch.tensor([0, 1])
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y, idx), batch_size=1)
    original_cat = train_module.torch.cat

    def guarded_cat(tensors, *args, **kwargs):
        tensor_list = list(tensors)
        if tensor_list and isinstance(tensor_list[0], torch.Tensor) and tensor_list[0].ndim == 3:
            raise AssertionError("build_train_residual_anchor_table_from_loader should stream residual sums")
        return original_cat(tensor_list, *args, **kwargs)

    monkeypatch.setattr(train_module.torch, "cat", guarded_cat)

    table, counts, n_windows = build_train_residual_anchor_table_from_loader(
        model=_ZeroBackbone(pred_len=2),
        loader=loader,
        cluster_id_c=torch.tensor([0]),
        device=torch.device("cpu"),
        history_anchor_cfg={},
        observed_history_tc=None,
        input_len=4,
        eval_start=0,
        period=2,
    )

    assert n_windows == 2
    assert torch.equal(counts, torch.tensor([1, 1]))
    assert torch.allclose(table[0, :, 0], torch.tensor([1.0, 2.0]))
    assert torch.allclose(table[1, :, 0], torch.tensor([3.0, 4.0]))
