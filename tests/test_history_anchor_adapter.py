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
    _candidate_selector_feature_gain_diagnostics,
    _candidate_selector_feature_standardization_stats,
    _candidate_selector_feature_names,
    _candidate_selector_features,
    _candidate_selector_targets,
    _collect_pred_residual_gate_tensors,
    _fit_static_candidate_channel_selector_from_tensors,
    _cluster_penalty_mask_to_channel_mask,
    _load_finetune_pred_residual_state,
    _pred_residual_selector_metrics_from_tensors,
    _normalize_history_anchor_cfg,
    _validate_strict_history_anchor_scope,
    apply_default_moe_output_anchor_cfg,
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


class _OnePenaltyGate(torch.nn.Module):
    def forward(
        self,
        feat_bkf: torch.Tensor,
        straight_through: bool = False,
        penalty_context_bkp: torch.Tensor | None = None,
        penalty_context_mode: str = "learned",
        penalty_context_weight: float = 0.0,
        penalty_context_detach: bool = True,
        penalty_context_score: str = "high_violation",
    ):
        b, k, _ = feat_bkf.shape
        mask = torch.ones(b, k, 1, device=feat_bkf.device, dtype=feat_bkf.dtype)
        probs = torch.ones_like(mask)
        return mask, probs, None, {}


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


def test_pred_residual_gate_tensor_collection_uses_history_anchored_base() -> None:
    x = torch.ones(1, 1, 4)
    y = torch.zeros(1, 1, 2)
    idx = torch.tensor([1])
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y, idx), batch_size=1)
    observed = torch.arange(20, dtype=torch.float32).view(20, 1)

    tensors = _collect_pred_residual_gate_tensors(
        model=_ZeroBackbone(pred_len=2),
        gate=_OnePenaltyGate(),
        pred_residual=_NoopPredResidual(),
        loader=loader,
        cluster_id_c=torch.tensor([0]),
        K=1,
        moe_cfg={"enable": True},
        device=torch.device("cpu"),
        penalty_names=["zero"],
        penalty_fns={"zero": lambda yhat, yref: torch.zeros_like(yhat[..., 0])},
        penalty_scale=None,
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
