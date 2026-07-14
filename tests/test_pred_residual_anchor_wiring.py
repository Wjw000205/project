import torch

from src.train import (
    _build_gate_routing_features,
    _cluster_utility_threshold_stats,
    _cluster_route_label_feature_diagnostics,
    _cluster_route_label_phase_diagnostics,
    _cluster_top1_confidence_gain_diagnostics,
    _gate_feature_names_for_mode,
    _build_penalty_route_learnability_class_features,
    _collect_penalty_route_learnability_tensors,
    _explainability_train_subsplit_ranges,
    _fit_penalty_route_learnability_head_from_tensors,
    _mse_utility_gate_supervision_loss,
    _normalize_confidence_gate_source_split,
    _normalize_pred_residual_selection_policy,
    _pred_residual_candidate_supervision_loss,
    _pred_residual_intervention_supervision_loss,
    _penalty_route_learnability_metrics_from_scores,
    _route_binary_adoption_loss_from_probs,
    _route_positive_recall_loss_from_probs,
    _route_precision_constrained_recall_loss_from_probs,
    _route_rate_alignment_loss_from_probs,
    _select_pred_residual_confidence_thresholds_from_tensors,
    eval_loop,
    evaluate_gate_penalty_hit_metrics,
    evaluate_penalty_explainability,
)
from src.models.residual_moe import ClusterwisePredResidualMoE


def test_pred_residual_selection_policy_accepts_guarded_alias() -> None:
    assert (
        _normalize_pred_residual_selection_policy("val_mse_candidate_channel_guarded")
        == "val_mse_candidate_channel"
    )
    assert _normalize_pred_residual_selection_policy("off") == "none"


class _ZeroBackbone(torch.nn.Module):
    def forward(self, x_bcl: torch.Tensor, cluster_id_c: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x_bcl.shape[0], x_bcl.shape[1], 2, device=x_bcl.device, dtype=x_bcl.dtype)


class _AllRouteGate(torch.nn.Module):
    def forward(self, feat_bkf: torch.Tensor, **kwargs):
        b, k, _ = feat_bkf.shape
        mask = torch.ones(b, k, 1, device=feat_bkf.device, dtype=feat_bkf.dtype)
        probs = mask.clone()
        skip = torch.zeros(b, k, device=feat_bkf.device, dtype=feat_bkf.dtype)
        return mask, probs, skip, skip


class _ConstantResidual(torch.nn.Module):
    def eval(self):
        return self

    def forward(
        self,
        x_bcl: torch.Tensor,
        y_base_bch: torch.Tensor,
        cluster_id_c: torch.Tensor,
        mask_bkp: torch.Tensor,
        skip_bk=None,
        **kwargs,
    ):
        residuals = torch.full(
            (x_bcl.shape[0], x_bcl.shape[1], 1, y_base_bch.shape[-1]),
            2.0,
            device=x_bcl.device,
            dtype=x_bcl.dtype,
        )
        route = mask_bkp[:, cluster_id_c, :]
        branches = route.unsqueeze(-1) * residuals
        return {
            "y_final": y_base_bch + branches.sum(dim=2),
            "residuals": residuals,
            "branches": branches,
            "route_bcp": route,
            "intervention_bcp": torch.ones_like(route),
            "effective_route_bcp": route,
            "alpha_cp": torch.ones(x_bcl.shape[1], 1, device=x_bcl.device, dtype=x_bcl.dtype),
        }


class _TwoClusterGate(torch.nn.Module):
    def eval(self):
        return self

    def forward(self, feat_bkf: torch.Tensor, **kwargs):
        b, k, _ = feat_bkf.shape
        mask = torch.zeros(b, k, 2, device=feat_bkf.device, dtype=feat_bkf.dtype)
        mask[:, 0, 0] = 1.0
        mask[:, 1, 1] = 1.0
        skip = torch.zeros(b, k, device=feat_bkf.device, dtype=feat_bkf.dtype)
        skip[:, 1] = 1.0
        return mask, mask.clone(), skip, skip.clone()


class _FirstPenaltyGate(torch.nn.Module):
    def eval(self):
        return self

    def forward(self, feat_bkf: torch.Tensor, **kwargs):
        b, k, _ = feat_bkf.shape
        mask = torch.zeros(b, k, 2, device=feat_bkf.device, dtype=feat_bkf.dtype)
        mask[:, :, 0] = 1.0
        skip = torch.zeros(b, k, device=feat_bkf.device, dtype=feat_bkf.dtype)
        return mask, mask.clone(), skip, skip.clone()


class _AllPenaltyGate(torch.nn.Module):
    def eval(self):
        return self

    def forward(self, feat_bkf: torch.Tensor, **kwargs):
        b, k, _ = feat_bkf.shape
        mask = torch.ones(b, k, 2, device=feat_bkf.device, dtype=feat_bkf.dtype)
        skip = torch.zeros(b, k, device=feat_bkf.device, dtype=feat_bkf.dtype)
        return mask, mask.clone(), skip, skip.clone()


class _TwoPenaltyResidual(torch.nn.Module):
    def eval(self):
        return self

    def forward(
        self,
        x_bcl: torch.Tensor,
        y_base_bch: torch.Tensor,
        cluster_id_c: torch.Tensor,
        mask_bkp: torch.Tensor,
        skip_bk=None,
        **kwargs,
    ):
        residuals = torch.zeros(
            x_bcl.shape[0],
            x_bcl.shape[1],
            2,
            y_base_bch.shape[-1],
            device=x_bcl.device,
            dtype=x_bcl.dtype,
        )
        residuals[:, :, 0, :] = 1.0
        residuals[:, :, 1, :] = -1.0
        route = mask_bkp[:, cluster_id_c, :]
        if skip_bk is not None:
            route = route * (1.0 - skip_bk[:, cluster_id_c].unsqueeze(-1))
        branches = route.unsqueeze(-1) * residuals
        return {
            "y_final": y_base_bch + branches.sum(dim=2),
            "residuals": residuals,
            "branches": branches,
            "route_bcp": route,
            "intervention_bcp": torch.ones_like(route),
            "effective_route_bcp": route,
            "alpha_cp": torch.ones(x_bcl.shape[1], 2, device=x_bcl.device, dtype=x_bcl.dtype),
        }


class _PatchRouterResidual(torch.nn.Module):
    """Patch route skips the first patch while the outer gate never skips."""

    def eval(self):
        return self

    def forward(
        self,
        x_bcl: torch.Tensor,
        y_base_bch: torch.Tensor,
        cluster_id_c: torch.Tensor,
        mask_bkp: torch.Tensor,
        skip_bk=None,
        **kwargs,
    ):
        batch, channels, horizon = y_base_bch.shape
        assert horizon == 2
        residuals = torch.zeros(
            batch,
            channels,
            2,
            horizon,
            device=x_bcl.device,
            dtype=x_bcl.dtype,
        )
        residuals[:, :, 0, :] = 1.0
        residuals[:, :, 1, :] = -1.0
        patch_route = torch.zeros_like(residuals)
        patch_route[:, :, 0, 1] = 1.0
        branches = patch_route * residuals
        route = patch_route.mean(dim=-1)
        patch_skip = torch.tensor(
            [[[1.0, 0.0]]],
            device=x_bcl.device,
            dtype=x_bcl.dtype,
        ).expand(batch, channels, 2)
        selected_penalty = torch.zeros(
            batch,
            channels,
            2,
            device=x_bcl.device,
            dtype=torch.long,
        )
        return {
            "y_final": y_base_bch + branches.sum(dim=2),
            "residuals": residuals,
            "branches": branches,
            "route_bcp": route,
            "intervention_bcp": torch.ones_like(route),
            "selector_bcp": torch.ones_like(route),
            "effective_route_bcp": route,
            "alpha_cp": torch.ones(
                channels,
                2,
                device=x_bcl.device,
                dtype=x_bcl.dtype,
            ),
            "patch_route_bcph": patch_route,
            "patch_skip_bcq": patch_skip,
            "patch_selected_penalty_index_bcq": selected_penalty,
        }


class _SelectorAnchorPathResidual(torch.nn.Module):
    def eval(self):
        return self

    def forward(
        self,
        x_bcl: torch.Tensor,
        y_base_bch: torch.Tensor,
        cluster_id_c: torch.Tensor,
        mask_bkp: torch.Tensor,
        skip_bk=None,
        **kwargs,
    ):
        residuals = torch.zeros(
            x_bcl.shape[0],
            x_bcl.shape[1],
            2,
            y_base_bch.shape[-1],
            device=x_bcl.device,
            dtype=x_bcl.dtype,
        )
        residuals[:, :, 1, :] = 1.0
        route = mask_bkp[:, cluster_id_c, :]
        return {
            "y_final": y_base_bch,
            "residuals": residuals,
            "branches": torch.zeros_like(residuals),
            "route_bcp": route,
            "intervention_bcp": torch.ones_like(route),
            "effective_route_bcp": route,
            "alpha_cp": torch.ones(x_bcl.shape[1], 2, device=x_bcl.device, dtype=x_bcl.dtype),
        }


class _AnchorAwareSelector(torch.nn.Module):
    def eval(self):
        return self

    def select_prediction(
        self,
        x_bcl: torch.Tensor,
        base_bch: torch.Tensor,
        cand_bcpH: torch.Tensor,
    ):
        choose_first = base_bch[..., :1] > 5.0
        selected = torch.where(choose_first, cand_bcpH[:, :, 0, :], cand_bcpH[:, :, 1, :])
        selected_class = torch.where(
            choose_first.squeeze(-1),
            torch.ones_like(base_bch[..., 0], dtype=torch.long),
            torch.full_like(base_bch[..., 0], 2, dtype=torch.long),
        )
        return selected, selected_class


def test_explainability_train_subsplit_ranges_keep_chronological_holdout() -> None:
    ranges = _explainability_train_subsplit_ranges(num_windows=10, holdout_fraction=0.3)

    assert ranges == {
        "train_fit": (0, 7),
        "train_holdout": (7, 10),
    }


def test_explainability_train_subsplit_ranges_keep_nonempty_sides_for_small_train() -> None:
    ranges = _explainability_train_subsplit_ranges(num_windows=2, holdout_fraction=0.9)

    assert ranges == {
        "train_fit": (0, 1),
        "train_holdout": (1, 2),
    }


def test_cluster_utility_threshold_stats_match_gate_utility_validity() -> None:
    gain_bcp = torch.tensor(
        [
            [[1.0, -1.0], [3.0, -1.0]],
            [[-1.0, 4.0], [-1.0, 2.0]],
        ]
    )
    cluster_id_c = torch.tensor([0, 0], dtype=torch.long)

    stats = _cluster_utility_threshold_stats(
        gain_bcp=gain_bcp,
        cluster_id_c=cluster_id_c,
        K=1,
        allowed_mask_kp=torch.tensor([[1.0, 1.0]]),
        thresholds=[0.0, 2.5],
    )

    assert torch.equal(stats["valid_count_kt"], torch.tensor([[2.0, 1.0]], dtype=torch.float64))
    assert torch.equal(stats["total_count_k"], torch.tensor([2.0], dtype=torch.float64))
    assert torch.allclose(stats["best_gain_sum_k"], torch.tensor([5.0], dtype=torch.float64))

    blocked = _cluster_utility_threshold_stats(
        gain_bcp=gain_bcp,
        cluster_id_c=cluster_id_c,
        K=1,
        allowed_mask_kp=torch.tensor([[0.0, 1.0]]),
        thresholds=[0.0],
    )

    assert torch.equal(blocked["valid_count_kt"], torch.tensor([[1.0]], dtype=torch.float64))
    assert torch.allclose(blocked["best_gain_sum_k"], torch.tensor([2.0], dtype=torch.float64))


def test_eval_selector_uses_output_anchor_candidate_path_before_selecting() -> None:
    x = torch.zeros(1, 1, 1)
    y = torch.full((1, 1, 2), 10.0)
    idx = torch.zeros(1, dtype=torch.long)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y, idx),
        batch_size=1,
        shuffle=False,
    )

    _, mse_k, mae_k, mse_c, mae_c, *_ = eval_loop(
        model=_ZeroBackbone(),
        gate=_AllPenaltyGate(),
        lambda_kp=torch.zeros(1, 2),
        penalty_names=["jump", "delta"],
        penalty_fns={
            "jump": lambda pred, target: torch.zeros(pred.shape[:2], dtype=pred.dtype),
            "delta": lambda pred, target: torch.zeros(pred.shape[:2], dtype=pred.dtype),
        },
        loader=loader,
        cluster_id_c=torch.tensor([0], dtype=torch.long),
        K=1,
        moe_cfg={
            "enable": True,
            "detach_penalty_grad": True,
            "train_stat_anchor_expert": {
                "enable": True,
                "mode": "phase_mean",
                "alpha": 1.0,
                "combine_mode": "anchor_plus_prediction",
            },
        },
        device=torch.device("cpu"),
        select_ranks=None,
        channel_count=1,
        pred_residual=_SelectorAnchorPathResidual(),
        pred_residual_selector=_AnchorAwareSelector(),
        input_len=1,
        train_stat_anchor_pc=torch.full((1, 1), 10.0),
    )

    assert mse_k.item() == 0.0
    assert mae_k.item() == 0.0
    assert mse_c.item() == 0.0
    assert mae_c.item() == 0.0


def test_cluster_route_label_feature_diagnostics_reports_stump_lift() -> None:
    feat_bkf = torch.tensor(
        [
            [[0.0]],
            [[0.1]],
            [[1.0]],
            [[1.2]],
        ],
        dtype=torch.float32,
    )
    route_label_bk = torch.tensor([[0], [0], [1], [1]], dtype=torch.long)

    diag = _cluster_route_label_feature_diagnostics(
        feat_bkf=feat_bkf,
        route_label_bk=route_label_bk,
        penalty_names=["delta"],
        feature_names=["route_feature"],
    )

    cluster = diag["per_cluster"][0]
    assert cluster["label_counts"] == {"skip": 2, "delta": 2}
    assert cluster["majority_acc"] == 0.5
    assert cluster["best_stump"]["feature"] == "route_feature"
    assert cluster["best_stump"]["accuracy"] == 1.0
    assert cluster["best_stump"]["lift_vs_majority"] == 0.5


def test_penalty_route_learnability_class_features_include_skip_and_penalty_rows() -> None:
    gate_feat = torch.tensor([[[1.0, 2.0]]])
    skip_feat = torch.tensor([[[0.5, 0.25]]])
    cand_feat = torch.tensor([[[[3.0, 4.0], [5.0, 6.0]]]])
    probs = torch.tensor([[[0.7, 0.3]]])
    route = torch.tensor([[[1.0, 0.0]]])
    intervention = torch.tensor([[[0.8, 0.2]]])
    selector = torch.tensor([[[0.9, 0.1]]])
    alpha = torch.tensor([[[0.4, 0.6]]])
    skip_prob = torch.tensor([[0.15]])

    features, names = _build_penalty_route_learnability_class_features(
        gate_feat_bkf=gate_feat,
        skip_feat_bkf=skip_feat,
        cand_feat_bkpf=cand_feat,
        gate_prob_bkp=probs,
        route_bkp=route,
        intervention_bkp=intervention,
        selector_bkp=selector,
        alpha_bkp=alpha,
        skip_prob_bk=skip_prob,
        cluster_count=1,
        penalty_names=["a", "b"],
    )

    assert features.shape[0:3] == (1, 1, 3)
    assert len(names) == features.shape[-1]
    class_skip_idx = names.index("class_skip")
    class_a_idx = names.index("class_a")
    route_weight_idx = names.index("route_weight")
    gate_prob_idx = names.index("gate_prob")
    skip_prob_idx = names.index("skip_prob")
    assert features[0, 0, 0, class_skip_idx].item() == 1.0
    assert features[0, 0, 1, class_a_idx].item() == 1.0
    assert features[0, 0, 0, route_weight_idx].item() == 0.0
    assert features[0, 0, 1, route_weight_idx].item() == 1.0
    assert abs(features[0, 0, 2, gate_prob_idx].item() - 0.3) < 1.0e-6
    assert abs(features[0, 0, 0, skip_prob_idx].item() - 0.15) < 1.0e-6


def test_penalty_route_learnability_metrics_compare_head_current_and_majority() -> None:
    labels = torch.tensor([0, 1, 1, 2], dtype=torch.long)
    current = torch.tensor([1, 1, 2, 2], dtype=torch.long)
    scores = torch.tensor(
        [
            [3.0, 1.0, 0.0],
            [0.0, 3.0, 1.0],
            [0.0, 2.0, 1.0],
            [0.0, 1.0, 3.0],
        ]
    )

    metrics = _penalty_route_learnability_metrics_from_scores(
        scores=scores,
        labels=labels,
        current_pred=current,
        label_names=["skip", "a", "b"],
    )

    assert metrics["accuracy_all"] == 1.0
    assert metrics["current_accuracy_all"] == 0.5
    assert metrics["majority_accuracy_all"] == 0.5
    assert metrics["accuracy_on_positive_oracle"] == 1.0
    assert abs(metrics["current_accuracy_on_positive_oracle"] - (2.0 / 3.0)) < 1.0e-6
    assert metrics["prediction_counts"] == {"skip": 1, "a": 2, "b": 1}


def test_penalty_route_learnability_head_learns_separable_oracle_labels() -> None:
    torch.manual_seed(123)
    n = 48
    labels = torch.arange(n, dtype=torch.long) % 3
    features = torch.zeros(n, 3, 5)
    features[:, :, 0] = torch.eye(3)[labels]
    features[:, :, 1:] = 0.01 * torch.randn(n, 3, 4)
    current = torch.zeros(n, dtype=torch.long)

    summary, _ = _fit_penalty_route_learnability_head_from_tensors(
        train_tensors={
            "features": features,
            "labels": labels,
            "current_pred": current,
        },
        eval_tensors_by_split={
            "val": {
                "features": features.clone(),
                "labels": labels.clone(),
                "current_pred": current.clone(),
            }
        },
        label_names=["skip", "a", "b"],
        feature_names=[f"f{i}" for i in range(5)],
        cfg={
            "epochs": 80,
            "batch_size": 16,
            "lr": 0.05,
            "hidden_dim": 0,
            "weight_decay": 0.0,
            "seed": 123,
        },
        device=torch.device("cpu"),
    )

    assert summary["splits"]["val"]["accuracy_all"] > 0.95
    assert summary["splits"]["val"]["current_accuracy_all"] < 0.5


def test_penalty_route_learnability_head_records_eval_early_stop_selection() -> None:
    torch.manual_seed(321)
    n = 36
    labels = torch.arange(n, dtype=torch.long) % 3
    features = torch.zeros(n, 3, 4)
    features[:, :, 0] = torch.eye(3)[labels]
    features[:, :, 1:] = 0.02 * torch.randn(n, 3, 3)
    current = torch.zeros(n, dtype=torch.long)

    summary, artifact = _fit_penalty_route_learnability_head_from_tensors(
        train_tensors={
            "features": features,
            "labels": labels,
            "current_pred": current,
        },
        eval_tensors_by_split={
            "val": {
                "features": features.clone(),
                "labels": labels.clone(),
                "current_pred": current.clone(),
            }
        },
        label_names=["skip", "a", "b"],
        feature_names=[f"f{i}" for i in range(4)],
        cfg={
            "epochs": 20,
            "batch_size": 12,
            "lr": 0.05,
            "hidden_dim": 0,
            "weight_decay": 0.0,
            "early_stop_split": "val",
            "selection_metric": "accuracy",
            "early_stop_patience": 3,
            "seed": 321,
        },
        device=torch.device("cpu"),
    )

    assert summary["selection"]["split"] == "val"
    assert summary["selection"]["metric"] == "accuracy"
    assert 1 <= summary["selection"]["best_epoch"] <= 20
    assert artifact["selection"]["split"] == "val"


def test_penalty_route_learnability_balanced_class_weight_is_reported_and_clipped() -> None:
    labels = torch.tensor([0] * 9 + [1] * 2 + [2], dtype=torch.long)
    n = int(labels.numel())
    features = torch.randn(n, 3, 4)
    current = torch.zeros(n, dtype=torch.long)

    summary, artifact = _fit_penalty_route_learnability_head_from_tensors(
        train_tensors={
            "features": features,
            "labels": labels,
            "current_pred": current,
        },
        eval_tensors_by_split={},
        label_names=["skip", "a", "b"],
        feature_names=[f"f{i}" for i in range(4)],
        cfg={
            "epochs": 2,
            "batch_size": 6,
            "lr": 0.01,
            "hidden_dim": 0,
            "class_weight": "balanced",
            "class_weight_min": 0.5,
            "class_weight_max": 2.0,
            "seed": 12,
        },
        device=torch.device("cpu"),
    )

    weights = summary["config"]["class_weight_values"]
    assert len(weights) == 3
    assert min(weights) >= 0.5
    assert max(weights) <= 2.0
    assert artifact["class_weight"].shape == (3,)


def test_penalty_route_learnability_flat_head_can_use_cross_candidate_context() -> None:
    torch.manual_seed(11)
    n = 64
    signal = torch.cat([torch.ones(n // 2), -torch.ones(n // 2)])
    labels = torch.where(signal > 0, torch.ones(n, dtype=torch.long), torch.full((n,), 2, dtype=torch.long))
    features = torch.zeros(n, 3, 3)
    features[:, 0, 0] = signal
    features[:, 1:, 1:] = 0.01 * torch.randn(n, 2, 2)
    current = torch.zeros(n, dtype=torch.long)

    summary, _ = _fit_penalty_route_learnability_head_from_tensors(
        train_tensors={
            "features": features,
            "labels": labels,
            "current_pred": current,
        },
        eval_tensors_by_split={
            "val": {
                "features": features.clone(),
                "labels": labels.clone(),
                "current_pred": current.clone(),
            }
        },
        label_names=["skip", "a", "b"],
        feature_names=[f"f{i}" for i in range(3)],
        cfg={
            "head_mode": "flat",
            "epochs": 80,
            "batch_size": 16,
            "lr": 0.05,
            "hidden_dim": 0,
            "weight_decay": 0.0,
            "early_stop_split": "val",
            "selection_metric": "accuracy",
            "seed": 11,
        },
        device=torch.device("cpu"),
    )

    assert summary["config"]["head_mode"] == "flat"
    assert summary["splits"]["val"]["accuracy_all"] > 0.95


def test_penalty_route_learnability_flat_head_epoch0_prior_can_tie_majority() -> None:
    labels = torch.tensor([1] * 10 + [2] * 4 + [0] * 2, dtype=torch.long)
    features = torch.zeros(int(labels.numel()), 3, 4)
    current = torch.zeros_like(labels)

    summary, _ = _fit_penalty_route_learnability_head_from_tensors(
        train_tensors={
            "features": features,
            "labels": labels,
            "current_pred": current,
        },
        eval_tensors_by_split={
            "val": {
                "features": features.clone(),
                "labels": labels.clone(),
                "current_pred": current.clone(),
            }
        },
        label_names=["skip", "a", "b"],
        feature_names=[f"f{i}" for i in range(4)],
        cfg={
            "head_mode": "flat",
            "epochs": 5,
            "batch_size": 8,
            "lr": 0.01,
            "hidden_dim": 0,
            "weight_decay": 0.0,
            "early_stop_split": "val",
            "selection_metric": "accuracy",
            "init_bias": "train_prior",
            "include_initial_eval": True,
            "early_stop_patience": 1,
            "seed": 22,
        },
        device=torch.device("cpu"),
    )

    val_metrics = summary["splits"]["val"]
    assert summary["selection_history"][0]["epoch"] == 0
    assert val_metrics["accuracy_all"] == val_metrics["majority_accuracy_all"]
    assert val_metrics["lift_vs_majority"] == 0.0


def test_collect_penalty_route_learnability_tensors_exports_cluster_oracle_labels() -> None:
    x = torch.zeros(2, 1, 3)
    y = torch.zeros(2, 1, 2)
    y[0, 0, :] = 1.0
    idx = torch.arange(2, dtype=torch.long)
    cluster_id_c = torch.tensor([0], dtype=torch.long)

    tensors = _collect_penalty_route_learnability_tensors(
        model=_ZeroBackbone(),
        gate=_FirstPenaltyGate(),
        pred_residual=_TwoPenaltyResidual(),
        loader=[(x, y, idx)],
        cluster_id_c=cluster_id_c,
        K=1,
        moe_cfg={"enable": True, "allow_skip": True},
        device=torch.device("cpu"),
        penalty_names=["positive", "negative"],
        penalty_fns={
            "positive": lambda yhat, ytrue: torch.zeros(yhat.shape[:2]),
            "negative": lambda yhat, ytrue: torch.zeros(yhat.shape[:2]),
        },
        penalty_scale=None,
        select_ranks=None,
        gate_soft_weight=0.0,
        split_name="val",
        feature_mode="base",
    )

    assert tensors is not None
    assert torch.equal(tensors["labels"], torch.tensor([1, 0]))
    assert torch.equal(tensors["current_pred"], torch.tensor([1, 1]))
    assert tensors["features"].shape[0:2] == (2, 3)
    assert tensors["label_names"] == ["skip", "positive", "negative"]


def test_cluster_route_label_phase_diagnostics_reports_phase_lift() -> None:
    query_start_abs_b = torch.arange(8, dtype=torch.long)
    route_label_bk = torch.tensor(
        [
            [1],
            [1],
            [0],
            [0],
            [1],
            [1],
            [0],
            [0],
        ],
        dtype=torch.long,
    )

    diag = _cluster_route_label_phase_diagnostics(
        query_start_abs_b=query_start_abs_b,
        route_label_bk=route_label_bk,
        penalty_names=["delta"],
        periods=[4],
        num_bins=4,
    )

    period_payload = diag["per_period"][0]
    cluster = period_payload["per_cluster"][0]
    assert period_payload["period"] == 4
    assert period_payload["num_bins"] == 4
    assert cluster["global_majority_acc"] == 0.5
    assert cluster["phase_majority_acc"] == 1.0
    assert cluster["lift_vs_global"] == 0.5
    assert [item["majority_label"] for item in cluster["bins"]] == ["delta", "delta", "skip", "skip"]


def test_cluster_top1_confidence_gain_diagnostics_bins_selected_gain() -> None:
    top1_conf_bc = torch.tensor([[0.2, 0.8], [0.3, 0.9]], dtype=torch.float32)
    top1_gain_bc = torch.tensor([[-1.0, 2.0], [-3.0, 4.0]], dtype=torch.float32)
    top1_p_bc = torch.zeros(2, 2, dtype=torch.long)
    top1_active_bc = torch.ones(2, 2, dtype=torch.bool)
    skip_bc = torch.zeros(2, 2, dtype=torch.bool)
    cluster_id_c = torch.zeros(2, dtype=torch.long)

    diag = _cluster_top1_confidence_gain_diagnostics(
        top1_conf_bc=top1_conf_bc,
        top1_gain_bc=top1_gain_bc,
        top1_p_bc=top1_p_bc,
        top1_active_bc=top1_active_bc,
        skip_bc=skip_bc,
        cluster_id_c=cluster_id_c,
        K=1,
        penalty_names=["delta"],
        bins=[0.0, 0.5, 1.0],
    )

    cluster = diag["per_cluster"][0]
    low, high = cluster["all"]["bins"]
    assert low["samples"] == 2
    assert low["mean_gain_mse"] == -2.0
    assert low["positive_rate"] == 0.0
    assert high["samples"] == 2
    assert high["mean_gain_mse"] == 3.0
    assert high["positive_rate"] == 1.0


def test_history_base_gate_features_append_forecast_shape_descriptors() -> None:
    x = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 2.0, 2.0, 2.0],
            ]
        ],
        dtype=torch.float32,
    )
    y_base = torch.tensor(
        [
            [
                [4.0, 5.0, 6.0],
                [1.0, 1.0, 1.0],
            ]
        ],
        dtype=torch.float32,
    )
    cluster_id_c = torch.tensor([0, 0], dtype=torch.long)

    history = _build_gate_routing_features(x, None, cluster_id_c, K=1, mode="history")
    history_base = _build_gate_routing_features(x, y_base, cluster_id_c, K=1, mode="history_base")
    names = _gate_feature_names_for_mode("history_base")

    assert history.shape == (1, 1, len(_gate_feature_names_for_mode("history")))
    assert history_base.shape == (1, 1, len(names))
    assert history_base.shape[-1] > history.shape[-1]
    assert "base_mean_shift_over_hist_std" in names
    assert "base_std_over_hist_std" in names


def test_utility_gate_supervision_uses_eval_path_candidates_when_provided() -> None:
    cluster_id_c = torch.zeros(1, dtype=torch.long)
    y = torch.ones(1, 1, 2)
    y_base_raw = torch.full((1, 1, 2), 100.0)
    y_base_eval = torch.zeros(1, 1, 2)
    cand_eval = torch.stack(
        [
            torch.full((1, 1, 2), 10.0),
            torch.ones(1, 1, 2),
        ],
        dim=2,
    )

    loss_prefers_bad = _mse_utility_gate_supervision_loss(
        probs_bkp=torch.tensor([[[0.8, 0.2]]]),
        allowed_mask_kp=torch.ones(1, 2),
        y_base_bch=y_base_raw,
        pred_out={},
        y_bch=y,
        cluster_id_c=cluster_id_c,
        K=1,
        y_base_eval_bch=y_base_eval,
        cand_eval_bcpH=cand_eval,
    )
    loss_prefers_good = _mse_utility_gate_supervision_loss(
        probs_bkp=torch.tensor([[[0.2, 0.8]]]),
        allowed_mask_kp=torch.ones(1, 2),
        y_base_bch=y_base_raw,
        pred_out={},
        y_bch=y,
        cluster_id_c=cluster_id_c,
        K=1,
        y_base_eval_bch=y_base_eval,
        cand_eval_bcpH=cand_eval,
    )

    assert loss_prefers_bad is not None
    assert loss_prefers_good is not None
    assert loss_prefers_good.mean() < loss_prefers_bad.mean()


def test_skip_aware_utility_gate_supervision_prefers_noop_when_all_candidates_hurt() -> None:
    cluster_id_c = torch.zeros(1, dtype=torch.long)
    y = torch.zeros(1, 1, 2)
    y_base = torch.zeros(1, 1, 2)
    cand_eval = torch.stack(
        [
            torch.ones(1, 1, 2),
            torch.full((1, 1, 2), 2.0),
        ],
        dim=2,
    )

    loss_low_skip = _mse_utility_gate_supervision_loss(
        probs_bkp=torch.tensor([[[0.5, 0.5]]]),
        skip_prob_bk=torch.tensor([[0.1]]),
        allowed_mask_kp=torch.ones(1, 2),
        y_base_bch=y_base,
        pred_out={},
        y_bch=y,
        cluster_id_c=cluster_id_c,
        K=1,
        y_base_eval_bch=y_base,
        cand_eval_bcpH=cand_eval,
        include_skip=True,
    )
    loss_high_skip = _mse_utility_gate_supervision_loss(
        probs_bkp=torch.tensor([[[0.5, 0.5]]]),
        skip_prob_bk=torch.tensor([[0.9]]),
        allowed_mask_kp=torch.ones(1, 2),
        y_base_bch=y_base,
        pred_out={},
        y_bch=y,
        cluster_id_c=cluster_id_c,
        K=1,
        y_base_eval_bch=y_base,
        cand_eval_bcpH=cand_eval,
        include_skip=True,
    )

    assert loss_low_skip is not None
    assert loss_high_skip is not None
    assert loss_high_skip.mean() < loss_low_skip.mean()


def test_skip_aware_utility_gate_supervision_uses_cluster_mean_gain_for_skip_target() -> None:
    cluster_id_c = torch.zeros(2, dtype=torch.long)
    y = torch.zeros(1, 2, 2)
    y_base = torch.ones(1, 2, 2)
    cand_eval = torch.zeros(1, 2, 1, 2)
    cand_eval[:, 0, 0, :] = 0.0
    cand_eval[:, 1, 0, :] = 3.0

    loss_low_skip = _mse_utility_gate_supervision_loss(
        probs_bkp=torch.tensor([[[1.0]]]),
        skip_prob_bk=torch.tensor([[0.1]]),
        allowed_mask_kp=torch.ones(1, 1),
        y_base_bch=y_base,
        pred_out={},
        y_bch=y,
        cluster_id_c=cluster_id_c,
        K=1,
        y_base_eval_bch=y_base,
        cand_eval_bcpH=cand_eval,
        include_skip=True,
    )
    loss_high_skip = _mse_utility_gate_supervision_loss(
        probs_bkp=torch.tensor([[[1.0]]]),
        skip_prob_bk=torch.tensor([[0.9]]),
        allowed_mask_kp=torch.ones(1, 1),
        y_base_bch=y_base,
        pred_out={},
        y_bch=y,
        cluster_id_c=cluster_id_c,
        K=1,
        y_base_eval_bch=y_base,
        cand_eval_bcpH=cand_eval,
        include_skip=True,
    )

    assert loss_low_skip is not None
    assert loss_high_skip is not None
    assert loss_high_skip.mean() < loss_low_skip.mean()


def test_skip_aware_utility_gate_supervision_prefers_good_penalty_over_skip() -> None:
    cluster_id_c = torch.zeros(1, dtype=torch.long)
    y = torch.ones(1, 1, 2)
    y_base = torch.zeros(1, 1, 2)
    cand_eval = torch.stack(
        [
            torch.zeros(1, 1, 2),
            torch.ones(1, 1, 2),
        ],
        dim=2,
    )

    loss_prefers_skip = _mse_utility_gate_supervision_loss(
        probs_bkp=torch.tensor([[[0.9, 0.1]]]),
        skip_prob_bk=torch.tensor([[0.8]]),
        allowed_mask_kp=torch.ones(1, 2),
        y_base_bch=y_base,
        pred_out={},
        y_bch=y,
        cluster_id_c=cluster_id_c,
        K=1,
        y_base_eval_bch=y_base,
        cand_eval_bcpH=cand_eval,
        include_skip=True,
    )
    loss_prefers_penalty = _mse_utility_gate_supervision_loss(
        probs_bkp=torch.tensor([[[0.1, 0.9]]]),
        skip_prob_bk=torch.tensor([[0.1]]),
        allowed_mask_kp=torch.ones(1, 2),
        y_base_bch=y_base,
        pred_out={},
        y_bch=y,
        cluster_id_c=cluster_id_c,
        K=1,
        y_base_eval_bch=y_base,
        cand_eval_bcpH=cand_eval,
        include_skip=True,
    )

    assert loss_prefers_skip is not None
    assert loss_prefers_penalty is not None
    assert loss_prefers_penalty.mean() < loss_prefers_skip.mean()


def test_skip_aware_utility_gate_supervision_accepts_joint_penalty_mass() -> None:
    cluster_id_c = torch.zeros(1, dtype=torch.long)
    y = torch.ones(1, 1, 2)
    y_base = torch.zeros(1, 1, 2)
    cand_eval = torch.ones(1, 1, 1, 2)

    loss_bk = _mse_utility_gate_supervision_loss(
        probs_bkp=torch.tensor([[[0.8]]]),
        skip_prob_bk=torch.tensor([[0.2]]),
        allowed_mask_kp=torch.ones(1, 1),
        y_base_bch=y_base,
        pred_out={},
        y_bch=y,
        cluster_id_c=cluster_id_c,
        K=1,
        y_base_eval_bch=y_base,
        cand_eval_bcpH=cand_eval,
        include_skip=True,
        probs_include_skip_mass=True,
    )

    assert loss_bk is not None
    assert torch.allclose(loss_bk, torch.full((1, 1), -torch.log(torch.tensor(0.8))))


def test_hard_oracle_utility_gate_supervision_targets_best_route() -> None:
    cluster_id_c = torch.zeros(1, dtype=torch.long)
    y = torch.zeros(1, 1, 2)
    y_base = torch.full((1, 1, 2), 2.0)
    cand_eval = torch.stack(
        [
            torch.ones(1, 1, 2),
            torch.zeros(1, 1, 2),
        ],
        dim=2,
    )

    loss_prefers_weaker = _mse_utility_gate_supervision_loss(
        probs_bkp=torch.tensor([[[0.8, 0.1]]]),
        skip_prob_bk=torch.tensor([[0.1]]),
        allowed_mask_kp=torch.ones(1, 2),
        y_base_bch=y_base,
        pred_out={},
        y_bch=y,
        cluster_id_c=cluster_id_c,
        K=1,
        y_base_eval_bch=y_base,
        cand_eval_bcpH=cand_eval,
        include_skip=True,
        probs_include_skip_mass=True,
        target_mode="hard_oracle",
    )
    loss_prefers_best = _mse_utility_gate_supervision_loss(
        probs_bkp=torch.tensor([[[0.1, 0.8]]]),
        skip_prob_bk=torch.tensor([[0.1]]),
        allowed_mask_kp=torch.ones(1, 2),
        y_base_bch=y_base,
        pred_out={},
        y_bch=y,
        cluster_id_c=cluster_id_c,
        K=1,
        y_base_eval_bch=y_base,
        cand_eval_bcpH=cand_eval,
        include_skip=True,
        probs_include_skip_mass=True,
        target_mode="hard_oracle",
    )

    assert loss_prefers_weaker is not None
    assert loss_prefers_best is not None
    assert loss_prefers_best.mean() < loss_prefers_weaker.mean()


def test_utility_gate_supervision_can_return_skip_target_diagnostics() -> None:
    cluster_id_c = torch.zeros(1, dtype=torch.long)
    y = torch.zeros(1, 1, 2)
    y_base = torch.zeros(1, 1, 2)
    cand_eval = torch.ones(1, 1, 1, 2)

    result = _mse_utility_gate_supervision_loss(
        probs_bkp=torch.tensor([[[1.0]]]),
        skip_prob_bk=torch.tensor([[0.25]]),
        allowed_mask_kp=torch.ones(1, 1),
        y_base_bch=y_base,
        pred_out={},
        y_bch=y,
        cluster_id_c=cluster_id_c,
        K=1,
        y_base_eval_bch=y_base,
        cand_eval_bcpH=cand_eval,
        include_skip=True,
        return_diagnostics=True,
    )

    assert result is not None
    loss_bk, diag = result
    assert loss_bk is not None
    assert torch.allclose(diag["valid_bk"], torch.zeros(1, 1))
    assert torch.allclose(diag["target_skip_bk"], torch.ones(1, 1))
    assert torch.allclose(diag["skip_prob_bk"], torch.full((1, 1), 0.25))
    assert torch.allclose(diag["best_gain_bk"], torch.full((1, 1), -1.0))


def test_skip_aware_utility_gate_supervision_keeps_skip_prob_gradient() -> None:
    cluster_id_c = torch.zeros(1, dtype=torch.long)
    y = torch.zeros(1, 1, 2)
    y_base = torch.zeros(1, 1, 2)
    cand_eval = torch.ones(1, 1, 1, 2)
    skip_prob = torch.tensor([[0.25]], requires_grad=True)

    loss_bk = _mse_utility_gate_supervision_loss(
        probs_bkp=torch.tensor([[[1.0]]]),
        skip_prob_bk=skip_prob,
        allowed_mask_kp=torch.ones(1, 1),
        y_base_bch=y_base,
        pred_out={},
        y_bch=y,
        cluster_id_c=cluster_id_c,
        K=1,
        y_base_eval_bch=y_base,
        cand_eval_bcpH=cand_eval,
        include_skip=True,
    )

    assert loss_bk is not None
    loss_bk.mean().backward()
    assert skip_prob.grad is not None
    assert skip_prob.grad.item() < 0.0


def test_binary_adoption_loss_treats_skip_label_as_all_zero_penalties() -> None:
    labels = torch.zeros(1, 1, dtype=torch.long)
    skip_prob = torch.tensor([[0.8]])
    low_penalty = torch.tensor([[[0.1, 0.1]]])
    high_penalty = torch.tensor([[[0.45, 0.45]]])

    low_loss = _route_binary_adoption_loss_from_probs(
        probs_bkp=low_penalty,
        labels_bk=labels,
        skip_prob_bk=skip_prob,
        probs_include_skip_mass=True,
    )
    high_loss = _route_binary_adoption_loss_from_probs(
        probs_bkp=high_penalty,
        labels_bk=labels,
        skip_prob_bk=torch.tensor([[0.1]]),
        probs_include_skip_mass=True,
    )

    assert low_loss is not None
    assert high_loss is not None
    assert low_loss.mean() < high_loss.mean()


def test_binary_adoption_loss_rewards_only_the_labeled_penalty() -> None:
    labels = torch.ones(1, 1, dtype=torch.long)
    skip_prob = torch.tensor([[0.1]])
    correct = torch.tensor([[[0.8, 0.1]]])
    wrong = torch.tensor([[[0.1, 0.8]]])

    correct_loss = _route_binary_adoption_loss_from_probs(
        probs_bkp=correct,
        labels_bk=labels,
        skip_prob_bk=skip_prob,
        probs_include_skip_mass=True,
    )
    wrong_loss = _route_binary_adoption_loss_from_probs(
        probs_bkp=wrong,
        labels_bk=labels,
        skip_prob_bk=skip_prob,
        probs_include_skip_mass=True,
    )

    assert correct_loss is not None
    assert wrong_loss is not None
    assert correct_loss.mean() < wrong_loss.mean()


def test_binary_adoption_loss_respects_allowed_penalty_mask() -> None:
    labels = torch.zeros(1, 1, dtype=torch.long)
    probs = torch.tensor([[[0.1, 0.8]]])
    skip_prob = torch.tensor([[0.1]])

    loss_allowed = _route_binary_adoption_loss_from_probs(
        probs_bkp=probs,
        labels_bk=labels,
        skip_prob_bk=skip_prob,
        probs_include_skip_mass=True,
        allowed_mask_kp=torch.tensor([[1.0, 0.0]]),
    )
    loss_unmasked = _route_binary_adoption_loss_from_probs(
        probs_bkp=probs,
        labels_bk=labels,
        skip_prob_bk=skip_prob,
        probs_include_skip_mass=True,
    )

    assert loss_allowed is not None
    assert loss_unmasked is not None
    assert loss_allowed.mean() < loss_unmasked.mean()


def test_route_rate_alignment_loss_is_zero_when_batch_rates_match_labels() -> None:
    probs = torch.tensor([[[0.5]], [[0.5]]])
    skip_prob = torch.tensor([[0.5], [0.5]])
    labels = torch.tensor([[0], [1]], dtype=torch.long)

    loss = _route_rate_alignment_loss_from_probs(
        probs_bkp=probs,
        skip_prob_bk=skip_prob,
        labels_bk=labels,
        probs_include_skip_mass=True,
    )

    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_route_rate_alignment_loss_penalizes_all_skip_when_labels_need_penalty() -> None:
    probs = torch.zeros(2, 1, 1)
    skip_prob = torch.ones(2, 1)
    labels = torch.tensor([[0], [1]], dtype=torch.long)

    loss = _route_rate_alignment_loss_from_probs(
        probs_bkp=probs,
        skip_prob_bk=skip_prob,
        labels_bk=labels,
        probs_include_skip_mass=True,
    )

    assert torch.all(loss > 0.0)


def test_route_rate_alignment_loss_uses_active_mask_for_rate_targets() -> None:
    probs = torch.zeros(2, 1, 1)
    skip_prob = torch.ones(2, 1)
    labels = torch.tensor([[0], [1]], dtype=torch.long)
    active = torch.tensor([[1], [0]], dtype=torch.bool)

    loss = _route_rate_alignment_loss_from_probs(
        probs_bkp=probs,
        skip_prob_bk=skip_prob,
        labels_bk=labels,
        active_mask_bk=active,
        probs_include_skip_mass=True,
    )

    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_positive_recall_loss_ignores_skip_labels() -> None:
    probs = torch.tensor([[[0.2, 0.8]]])
    skip_prob = torch.tensor([[0.1]])
    labels = torch.zeros(1, 1, dtype=torch.long)

    loss = _route_positive_recall_loss_from_probs(
        probs_bkp=probs,
        skip_prob_bk=skip_prob,
        labels_bk=labels,
        probs_include_skip_mass=True,
    )

    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_positive_recall_loss_rewards_the_labeled_penalty() -> None:
    labels = torch.ones(1, 1, dtype=torch.long)
    skip_prob = torch.tensor([[0.1]])
    correct = torch.tensor([[[0.8, 0.1]]])
    wrong = torch.tensor([[[0.1, 0.8]]])

    correct_loss = _route_positive_recall_loss_from_probs(
        probs_bkp=correct,
        skip_prob_bk=skip_prob,
        labels_bk=labels,
        probs_include_skip_mass=True,
    )
    wrong_loss = _route_positive_recall_loss_from_probs(
        probs_bkp=wrong,
        skip_prob_bk=skip_prob,
        labels_bk=labels,
        probs_include_skip_mass=True,
    )

    assert correct_loss.mean() < wrong_loss.mean()


def test_positive_recall_loss_respects_active_mask() -> None:
    probs = torch.zeros(2, 1, 1)
    skip_prob = torch.ones(2, 1)
    labels = torch.ones(2, 1, dtype=torch.long)
    active = torch.tensor([[1], [0]], dtype=torch.bool)

    loss = _route_positive_recall_loss_from_probs(
        probs_bkp=probs,
        skip_prob_bk=skip_prob,
        labels_bk=labels,
        active_mask_bk=active,
        probs_include_skip_mass=True,
    )

    assert loss[0, 0].item() > 1.0
    assert loss[1, 0].item() == 0.0


def test_positive_recall_margin_loss_stops_after_target_probability_floor() -> None:
    labels = torch.ones(2, 1, dtype=torch.long)
    skip_prob = torch.tensor([[0.4], [0.4]])
    probs = torch.tensor([[[0.7]], [[0.2]]])

    loss = _route_positive_recall_loss_from_probs(
        probs_bkp=probs,
        skip_prob_bk=skip_prob,
        labels_bk=labels,
        probs_include_skip_mass=True,
        mode="margin",
        target_probability=0.5,
    )

    assert loss[0, 0].item() == 0.0
    assert loss[1, 0].item() > 0.0


def test_precision_constrained_recall_penalizes_false_adopt_on_skip_label() -> None:
    labels = torch.zeros(2, 1, dtype=torch.long)
    skip_prob = torch.tensor([[0.2], [0.8]])
    probs = torch.tensor([[[0.8]], [[0.2]]])

    loss = _route_precision_constrained_recall_loss_from_probs(
        probs_bkp=probs,
        skip_prob_bk=skip_prob,
        labels_bk=labels,
        probs_include_skip_mass=True,
        false_adopt_max_probability=0.5,
    )

    assert loss[0, 0].item() > 0.0
    assert loss[1, 0].item() == 0.0


def test_precision_constrained_recall_still_rewards_positive_penalty_recall() -> None:
    labels = torch.ones(1, 1, dtype=torch.long)
    skip_prob = torch.tensor([[0.1]])
    correct = torch.tensor([[[0.8, 0.1]]])
    wrong = torch.tensor([[[0.1, 0.8]]])

    correct_loss = _route_precision_constrained_recall_loss_from_probs(
        probs_bkp=correct,
        skip_prob_bk=skip_prob,
        labels_bk=labels,
        probs_include_skip_mass=True,
        false_adopt_max_probability=0.5,
    )
    wrong_loss = _route_precision_constrained_recall_loss_from_probs(
        probs_bkp=wrong,
        skip_prob_bk=skip_prob,
        labels_bk=labels,
        probs_include_skip_mass=True,
        false_adopt_max_probability=0.5,
    )

    assert correct_loss.mean() < wrong_loss.mean()


def test_candidate_supervision_respects_allowed_penalty_mask() -> None:
    cluster_id_c = torch.zeros(1, dtype=torch.long)
    y_base = torch.zeros(1, 1, 2)
    y = torch.ones(1, 1, 2)
    pred_out = {
        "residuals": torch.tensor([[[[1.0, 1.0], [5.0, 5.0]]]]),
        "alpha_cp": torch.ones(1, 2),
        "intervention_bcp": torch.ones(1, 1, 2),
    }

    good_only = _pred_residual_candidate_supervision_loss(
        y_base_bch=y_base,
        pred_out=pred_out,
        y_bch=y,
        cluster_id_c=cluster_id_c,
        K=1,
        allowed_mask_kp=torch.tensor([[1.0, 0.0]]),
    )
    bad_only = _pred_residual_candidate_supervision_loss(
        y_base_bch=y_base,
        pred_out=pred_out,
        y_bch=y,
        cluster_id_c=cluster_id_c,
        K=1,
        allowed_mask_kp=torch.tensor([[0.0, 1.0]]),
    )

    assert good_only is not None
    assert bad_only is not None
    assert torch.allclose(good_only, torch.zeros(1, 1))
    assert torch.allclose(bad_only, torch.full((1, 1), 16.0))


def test_candidate_supervision_can_use_own_penalty_instead_of_mse() -> None:
    cluster_id_c = torch.zeros(1, dtype=torch.long)
    y_base = torch.zeros(1, 1, 2)
    y = torch.ones(1, 1, 2)
    pred_out = {
        "residuals": torch.tensor([[[[0.0, 0.0], [5.0, 5.0]]]]),
        "alpha_cp": torch.ones(1, 2),
        "intervention_bcp": torch.ones(1, 1, 2),
    }

    loss = _pred_residual_candidate_supervision_loss(
        y_base_bch=y_base,
        pred_out=pred_out,
        y_bch=y,
        cluster_id_c=cluster_id_c,
        K=1,
        penalty_names=["zero", "mean"],
        penalty_fns={
            "zero": lambda yhat, ytrue: torch.zeros(yhat.shape[:2], dtype=yhat.dtype),
            "mean": lambda yhat, ytrue: yhat.mean(dim=-1),
        },
        loss_kind="own_penalty",
        only_allowed=False,
    )

    assert loss is not None
    assert torch.allclose(loss, torch.full((1, 1), 2.5))


def test_candidate_supervision_can_add_forecast_mse_to_own_penalty() -> None:
    cluster_id_c = torch.zeros(1, dtype=torch.long)
    y_base = torch.zeros(1, 1, 2)
    y = torch.ones(1, 1, 2)
    residuals = torch.tensor(
        [[[[0.0, 0.0], [5.0, 5.0]]]],
        requires_grad=True,
    )
    pred_out = {
        "residuals": residuals,
        "alpha_cp": torch.ones(1, 2),
        "intervention_bcp": torch.ones(1, 1, 2),
    }

    loss = _pred_residual_candidate_supervision_loss(
        y_base_bch=y_base,
        pred_out=pred_out,
        y_bch=y,
        cluster_id_c=cluster_id_c,
        K=1,
        penalty_names=["zero", "mean"],
        penalty_fns={
            "zero": lambda yhat, ytrue: torch.zeros(
                yhat.shape[:2],
                dtype=yhat.dtype,
                device=yhat.device,
            ),
            "mean": lambda yhat, ytrue: yhat.mean(dim=-1),
        },
        loss_kind="own_penalty_mse",
        forecast_mse_weight=2.0,
        only_allowed=False,
        include_intervention=False,
        include_selector=False,
        include_patch_route=False,
    )

    # Own-penalty mean is (0 + 5) / 2 = 2.5. Candidate MSE mean is
    # (1 + 16) / 2 = 8.5, so weight 2 makes the composite equal 19.5.
    assert loss is not None
    assert torch.allclose(loss, torch.full((1, 1), 19.5))
    loss.mean().backward()
    assert residuals.grad is not None
    assert torch.all(residuals.grad.abs().sum(dim=-1) > 0.0)


def test_candidate_supervision_gain_hinge_penalizes_only_worse_than_base() -> None:
    cluster_id_c = torch.zeros(1, dtype=torch.long)
    y_base = torch.zeros(1, 1, 2)
    y = torch.ones(1, 1, 2)
    pred_out = {
        "residuals": torch.tensor([[[[1.0, 1.0], [3.0, 3.0]]]]),
        "alpha_cp": torch.ones(1, 2),
        "intervention_bcp": torch.ones(1, 1, 2),
    }

    good_only = _pred_residual_candidate_supervision_loss(
        y_base_bch=y_base,
        pred_out=pred_out,
        y_bch=y,
        cluster_id_c=cluster_id_c,
        K=1,
        allowed_mask_kp=torch.tensor([[1.0, 0.0]]),
        loss_kind="gain_hinge_mse",
    )
    bad_only = _pred_residual_candidate_supervision_loss(
        y_base_bch=y_base,
        pred_out=pred_out,
        y_bch=y,
        cluster_id_c=cluster_id_c,
        K=1,
        allowed_mask_kp=torch.tensor([[0.0, 1.0]]),
        loss_kind="gain_hinge_mse",
    )

    assert good_only is not None
    assert bad_only is not None
    assert torch.allclose(good_only, torch.zeros(1, 1))
    assert torch.allclose(bad_only, torch.full((1, 1), 3.0))


def test_candidate_supervision_gain_hinge_can_require_margin_over_base() -> None:
    cluster_id_c = torch.zeros(1, dtype=torch.long)
    y_base = torch.zeros(1, 1, 2)
    y = torch.ones(1, 1, 2)
    pred_out = {
        "residuals": torch.tensor([[[[0.8, 0.8]]]]),
        "alpha_cp": torch.ones(1, 1),
        "intervention_bcp": torch.ones(1, 1, 1),
    }

    loss = _pred_residual_candidate_supervision_loss(
        y_base_bch=y_base,
        pred_out=pred_out,
        y_bch=y,
        cluster_id_c=cluster_id_c,
        K=1,
        allowed_mask_kp=torch.ones(1, 1),
        loss_kind="gain_hinge_mse",
        min_abs_improvement=1.0,
    )

    assert loss is not None
    assert torch.allclose(loss, torch.full((1, 1), 0.04), atol=1.0e-6)


def test_candidate_supervision_can_ignore_intervention_for_adapter_training() -> None:
    cluster_id_c = torch.zeros(1, dtype=torch.long)
    y_base = torch.zeros(1, 1, 2)
    y = torch.ones(1, 1, 2)
    pred_out = {
        "residuals": torch.tensor([[[[1.0, 1.0]]]]),
        "alpha_cp": torch.ones(1, 1),
        "intervention_bcp": torch.zeros(1, 1, 1),
    }

    loss = _pred_residual_candidate_supervision_loss(
        y_base_bch=y_base,
        pred_out=pred_out,
        y_bch=y,
        cluster_id_c=cluster_id_c,
        K=1,
        allowed_mask_kp=torch.ones(1, 1),
        loss_kind="mse",
        include_intervention=False,
    )

    assert loss is not None
    assert torch.allclose(loss, torch.zeros(1, 1))


def test_intervention_supervision_uses_unmasked_candidate_gain() -> None:
    cluster_id_c = torch.zeros(2, dtype=torch.long)
    y_base = torch.zeros(1, 2, 2)
    y = torch.ones(1, 2, 2)
    intervention = torch.tensor([[[0.25], [0.75]]], requires_grad=True)
    pred_out = {
        "residuals": torch.tensor([[[[1.0, 1.0]], [[3.0, 3.0]]]]),
        "alpha_cp": torch.ones(2, 1),
        "intervention_bcp": intervention,
    }

    loss = _pred_residual_intervention_supervision_loss(
        y_base_bch=y_base,
        pred_out=pred_out,
        y_bch=y,
        cluster_id_c=cluster_id_c,
        K=1,
        allowed_mask_kp=torch.ones(1, 1),
        min_gain=0.0,
    )

    assert loss is not None
    loss.mean().backward()
    assert intervention.grad is not None
    assert intervention.grad[0, 0, 0].item() < 0.0
    assert intervention.grad[0, 1, 0].item() > 0.0


def test_pred_residual_confidence_gate_suppresses_low_confidence_penalty() -> None:
    pred_residual = ClusterwisePredResidualMoE(
        num_clusters=1,
        num_penalties=2,
        input_len=1,
        pred_len=1,
        hidden_dim=1,
        init_alpha=20.0,
        alpha_scale=1.0,
        use_y_base_input=False,
        intervention_enable=True,
    )
    with torch.no_grad():
        for b in pred_residual.b2:
            b.fill_(1.0)
        pred_residual.b_gate[0].fill_(2.0)
        pred_residual.b_gate[1].fill_(-2.0)

    x = torch.zeros(1, 1, 1)
    y_base = torch.zeros(1, 1, 1)
    cluster_id = torch.zeros(1, dtype=torch.long)
    route = torch.ones(1, 1, 2)

    pred_residual.set_confidence_gate(
        penalty_threshold_kp=torch.tensor([[0.95, 0.95]]),
        enable=True,
    )
    skipped = pred_residual(x, y_base, cluster_id, route)
    assert torch.allclose(skipped["y_final"], y_base, atol=1.0e-6)
    assert torch.equal(skipped["confidence_active_bcp"], torch.zeros(1, 1, 2))

    pred_residual.set_confidence_gate(
        penalty_threshold_kp=torch.tensor([[0.80, 0.95]]),
        enable=True,
    )
    accepted = pred_residual(x, y_base, cluster_id, route)
    assert accepted["confidence_active_bcp"][0, 0, 0].item() == 1.0
    assert accepted["confidence_active_bcp"][0, 0, 1].item() == 0.0
    assert accepted["y_final"][0, 0, 0].item() > 0.5


def test_confidence_gate_source_split_rejects_test_y_base() -> None:
    assert _normalize_confidence_gate_source_split("train") == "train"
    assert _normalize_confidence_gate_source_split("train_holdout") == "train_holdout"

    try:
        _normalize_confidence_gate_source_split("test")
    except ValueError as exc:
        assert "train or train_holdout" in str(exc)
    else:
        raise AssertionError("test source split must be rejected for confidence-gate source selection")


def test_confidence_threshold_selection_uses_gain_labels_and_allowed_mask() -> None:
    base = torch.zeros(4, 1, 1)
    y = torch.ones(4, 1, 1)
    cand = torch.zeros(4, 1, 2, 1)
    cand[:2, :, 0, :] = 1.0
    cand[2:, :, 0, :] = -1.0
    cand[:, :, 1, :] = 1.0
    confidence = torch.tensor(
        [
            [[0.90, 0.99]],
            [[0.85, 0.99]],
            [[0.20, 0.99]],
            [[0.10, 0.99]],
        ],
        dtype=torch.float32,
    )

    thresholds, summary = _select_pred_residual_confidence_thresholds_from_tensors(
        tensors={
            "base": base,
            "cand": cand,
            "y": y,
            "confidence": confidence,
        },
        cluster_id_c=torch.zeros(1, dtype=torch.long),
        K=1,
        allowed_mask_kp=torch.tensor([[1.0, 0.0]]),
        penalty_names=["good", "blocked"],
        selection_metric="mse",
    )

    assert thresholds.shape == (1, 2)
    assert 0.20 < thresholds[0, 0].item() <= 0.851
    assert thresholds[0, 1].item() > 1.0
    assert summary["per_cluster_penalty"]["0"]["good"]["selected_gain_pct_vs_base"] > 0.0
    assert summary["per_cluster_penalty"]["0"]["blocked"]["allowed"] is False


def test_confidence_threshold_selection_can_require_precision_guard() -> None:
    base = torch.zeros(6, 1, 1)
    y = torch.ones(6, 1, 1)
    cand = torch.zeros(6, 1, 1, 1)
    cand[[0, 3], :, 0, :] = 1.0
    cand[[1, 2, 4, 5], :, 0, :] = 1.0 + (1.05 ** 0.5)
    confidence = torch.tensor(
        [
            [[0.95]],
            [[0.90]],
            [[0.80]],
            [[0.75]],
            [[0.70]],
            [[0.25]],
        ],
        dtype=torch.float32,
    )

    thresholds, summary = _select_pred_residual_confidence_thresholds_from_tensors(
        tensors={
            "base": base,
            "cand": cand,
            "y": y,
            "confidence": confidence,
        },
        cluster_id_c=torch.zeros(1, dtype=torch.long),
        K=1,
        allowed_mask_kp=torch.ones(1, 1),
        penalty_names=["guarded"],
        selection_metric="precision_guarded_mse",
        min_precision=0.75,
        max_pred_positive_rate=0.50,
    )

    stats = summary["per_cluster_penalty"]["0"]["guarded"]
    assert thresholds[0, 0].item() >= 0.95 - 1.0e-6
    assert stats["precision"] >= 0.75
    assert stats["pred_positive_rate"] <= 0.50


def test_confidence_threshold_selection_skips_when_no_threshold_meets_precision_guard() -> None:
    base = torch.zeros(4, 1, 1)
    y = torch.ones(4, 1, 1)
    cand = torch.full((4, 1, 1, 1), 1.0 + (1.05 ** 0.5))
    cand[0, :, 0, :] = 1.0
    confidence = torch.tensor([[[0.95]], [[0.90]], [[0.85]], [[0.80]]], dtype=torch.float32)

    thresholds, summary = _select_pred_residual_confidence_thresholds_from_tensors(
        tensors={
            "base": base,
            "cand": cand,
            "y": y,
            "confidence": confidence,
        },
        cluster_id_c=torch.zeros(1, dtype=torch.long),
        K=1,
        allowed_mask_kp=torch.ones(1, 1),
        penalty_names=["guarded"],
        selection_metric="precision_guarded_mse",
        min_precision=1.01,
    )

    stats = summary["per_cluster_penalty"]["0"]["guarded"]
    assert thresholds[0, 0].item() > 0.95
    assert stats["reason"] == "no_threshold_meets_confidence_guard"
    assert stats["skip_selected"] is True


def test_penalty_explainability_reports_oracle_top1_and_skip_by_cluster() -> None:
    x = torch.zeros(2, 2, 3)
    y = torch.ones(2, 2, 2)
    idx = torch.arange(2, dtype=torch.long)
    cluster_id_c = torch.tensor([0, 1], dtype=torch.long)

    payload = evaluate_penalty_explainability(
        model=_ZeroBackbone(),
        gate=_TwoClusterGate(),
        pred_residual=_TwoPenaltyResidual(),
        loader=[(x, y, idx)],
        cluster_id_c=cluster_id_c,
        K=2,
        moe_cfg={"enable": True, "allow_skip": True},
        device=torch.device("cpu"),
        penalty_names=["good", "bad"],
        penalty_fns={"good": lambda yhat, ytrue: torch.zeros(yhat.shape[:2]), "bad": lambda yhat, ytrue: torch.zeros(yhat.shape[:2])},
        penalty_scale=None,
        select_ranks=None,
        gate_soft_weight=0.0,
        split_name="val",
    )

    assert payload is not None
    cluster0 = payload["per_cluster"][0]
    cluster1 = payload["per_cluster"][1]
    assert cluster0["oracle_gain_pct_vs_base"] == 100.0
    assert cluster0["skip_rate"] == 0.0
    assert cluster1["oracle_gain_pct_vs_base"] == 100.0
    assert cluster1["skip_rate"] == 1.0
    good_row = next(row for row in payload["rows"] if row["cluster_id"] == 0 and row["penalty"] == "good")
    bad_row = next(row for row in payload["rows"] if row["cluster_id"] == 1 and row["penalty"] == "bad")
    assert good_row["top1_selected_count"] == 2
    assert good_row["top1_selected_positive_rate"] == 1.0
    assert bad_row["top1_selected_count"] == 0
    assert bad_row["skipped_on_oracle_positive_count"] == 2


def test_penalty_explainability_uses_the_deployed_patch_skip_route() -> None:
    x = torch.zeros(1, 1, 3)
    y = torch.ones(1, 1, 2)
    idx = torch.zeros(1, dtype=torch.long)

    payload = evaluate_penalty_explainability(
        model=_ZeroBackbone(),
        gate=_FirstPenaltyGate(),
        pred_residual=_PatchRouterResidual(),
        loader=[(x, y, idx)],
        cluster_id_c=torch.zeros(1, dtype=torch.long),
        K=1,
        moe_cfg={
            "enable": True,
            "allow_skip": True,
            "explainability": {
                "adapter_specialization": {
                    "enable": True,
                    "scale_sweep": [0.0, 0.5, 1.0],
                },
                "periodic_action_space": {"enable": True},
            },
        },
        device=torch.device("cpu"),
        penalty_names=["good", "bad"],
        penalty_fns={
            "good": lambda yhat, ytrue: (yhat - ytrue).square().mean(dim=-1),
            "bad": lambda yhat, ytrue: (yhat - ytrue).square().mean(dim=-1),
        },
        penalty_scale=None,
        select_ranks=None,
        gate_soft_weight=0.0,
        split_name="val",
    )

    assert payload is not None
    # The legacy outer gate never skips, which is precisely the old diagnostic
    # bug this integration test distinguishes from the deployed patch route.
    assert payload["per_cluster"][0]["skip_rate"] == 0.0
    assert payload["routing_diagnostic_sources"]["legacy_fields"] == {
        "routing_granularity": "sample_channel_full_horizon",
        "skip_source": "outer_cluster_gate_skip_bk",
        "selected_penalty_source": "outer_cluster_gate_mask_and_probs",
        "replaced_by_patch_router": True,
    }
    patch = payload["patch_router_route_diagnostics"]
    assert patch["routing_granularity"] == "sample_channel_patch"
    assert patch["skip_source"] == "patch_skip_bcq"
    assert patch["aggregate"]["decision_count"] == 2
    assert patch["aggregate"]["actual_skip_count"] == 1
    assert patch["aggregate"]["actual_skip_rate"] == 0.5
    assert patch["aggregate"]["route_skip_mismatch_count"] == 0
    assert patch["aggregate"]["route_skip_mismatch_rate"] == 0.0
    assert patch["aggregate"]["missed_dual_action_count"] == 1
    assert patch["per_cluster"][0]["actual_skip_rate"] == 0.5
    assert patch["per_cluster"][0]["per_penalty"][0]["selected_count"] == 1
    gated_good = patch["aggregate"]["per_penalty"][0]
    assert gated_good["named_penalty_available_rate"] == 1.0
    assert gated_good["selected_named_penalty_positive_precision"] == 1.0
    assert gated_good["selected_named_penalty_reduction_pct"] == 100.0
    assert gated_good["selected_mean_mse_gain"] == 1.0
    assert gated_good["selected_mean_mae_gain"] == 1.0
    assert gated_good["selected_region_proof_pass"] is True
    assert (
        patch["aggregate"]["conditional_specialization_proof"]
        ["all_activated_experts_pass"]
        is True
    )
    gated_scale = payload["gated_adapter_scale_sweep"]
    assert gated_scale["all_activated_experts_have_joint_safe_nonzero_scale"] is True
    assert gated_scale["rows"][0]["expert"] == "good"
    assert gated_scale["rows"][0]["best_joint_scale"] == 1.0
    action_oracle = payload["periodic_action_space_oracle"]
    assert action_oracle["action_space"] == [
        "backbone",
        "periodic",
        "periodic+good",
        "periodic+bad",
    ]
    assert action_oracle["periodic_selectable_dual_safe_oracle"]["mse"] == 0.0
    assert (
        action_oracle["periodic_selectable_dual_safe_oracle"]
        ["periodic_plus_other_action_rate"]
        == 1.0
    )


def test_penalty_explainability_reports_cluster_route_oracle_with_skip() -> None:
    x = torch.zeros(1, 2, 3)
    y = torch.zeros(1, 2, 2)
    y[:, 0, :] = 1.0
    y[:, 1, :] = -1.0
    idx = torch.arange(1, dtype=torch.long)
    cluster_id_c = torch.tensor([0, 0], dtype=torch.long)

    payload = evaluate_penalty_explainability(
        model=_ZeroBackbone(),
        gate=_FirstPenaltyGate(),
        pred_residual=_TwoPenaltyResidual(),
        loader=[(x, y, idx)],
        cluster_id_c=cluster_id_c,
        K=1,
        moe_cfg={"enable": True, "allow_skip": True},
        device=torch.device("cpu"),
        penalty_names=["positive", "negative"],
        penalty_fns={
            "positive": lambda yhat, ytrue: torch.zeros(yhat.shape[:2]),
            "negative": lambda yhat, ytrue: torch.zeros(yhat.shape[:2]),
        },
        penalty_scale=None,
        select_ranks=None,
        gate_soft_weight=0.0,
        split_name="val",
    )

    assert payload is not None
    cluster = payload["per_cluster"][0]
    assert cluster["oracle_gain_pct_vs_base"] == 100.0
    assert cluster["cluster_route_oracle_gain_pct_vs_base"] == 0.0
    assert cluster["cluster_route_oracle_skip_rate"] == 1.0
    assert payload["cluster_route_oracle_gain_pct_vs_base"] == 0.0


def test_penalty_explainability_cluster_route_oracle_respects_allowed_mask() -> None:
    x = torch.zeros(1, 1, 3)
    y = torch.ones(1, 1, 2)
    idx = torch.arange(1, dtype=torch.long)
    cluster_id_c = torch.tensor([0], dtype=torch.long)

    payload = evaluate_penalty_explainability(
        model=_ZeroBackbone(),
        gate=_FirstPenaltyGate(),
        pred_residual=_TwoPenaltyResidual(),
        loader=[(x, y, idx)],
        cluster_id_c=cluster_id_c,
        K=1,
        moe_cfg={"enable": True, "allow_skip": True},
        device=torch.device("cpu"),
        penalty_names=["mse_proxy", "shape_axis"],
        penalty_fns={
            "mse_proxy": lambda yhat, ytrue: torch.zeros(yhat.shape[:2]),
            "shape_axis": lambda yhat, ytrue: torch.zeros(yhat.shape[:2]),
        },
        penalty_scale=None,
        select_ranks=None,
        gate_soft_weight=0.0,
        split_name="val",
        allowed_mask_kp=torch.tensor([[0.0, 1.0]]),
    )

    assert payload is not None
    cluster = payload["per_cluster"][0]
    assert cluster["cluster_route_oracle_gain_pct_vs_base"] == 0.0
    assert cluster["cluster_route_oracle_skip_rate"] == 1.0
    route_diag = payload["route_label_feature_diagnostics"]["per_cluster"][0]
    assert route_diag["label_counts"] == {"skip": 1, "mse_proxy": 0, "shape_axis": 0}


def test_gate_penalty_hit_oracle_respects_allowed_mask() -> None:
    x = torch.zeros(1, 1, 3)
    y = torch.ones(1, 1, 2)
    idx = torch.arange(1, dtype=torch.long)
    cluster_id_c = torch.tensor([0], dtype=torch.long)

    payload = evaluate_gate_penalty_hit_metrics(
        model=_ZeroBackbone(),
        gate=_FirstPenaltyGate(),
        pred_residual=_TwoPenaltyResidual(),
        loader=[(x, y, idx)],
        cluster_id_c=cluster_id_c,
        K=1,
        moe_cfg={"enable": True, "allow_skip": True},
        device=torch.device("cpu"),
        penalty_names=["mse_proxy", "shape_axis"],
        penalty_fns={
            "mse_proxy": lambda yhat, ytrue: torch.zeros(yhat.shape[:2]),
            "shape_axis": lambda yhat, ytrue: torch.zeros(yhat.shape[:2]),
        },
        penalty_scale=None,
        select_ranks=None,
        gate_soft_weight=0.0,
        allowed_mask_kp=torch.tensor([[0.0, 1.0]]),
    )

    assert payload is not None
    assert payload["oracle_count"] == {"mse_proxy": 0, "shape_axis": 1}
    assert payload["oracle_gain_pct_vs_base"] < 0.0
