from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.penalties import build_penalty_bank
from src.models.residual_moe import ClusterwisePredResidualMoE
from src.training.selectors import _pred_residual_candidate_supervision_loss
from src.training.evaluation import (
    apply_position_daily_residual_ridge,
    evaluate_adapter_gradient_isolation,
    evaluate_penalty_explainability,
    fit_position_daily_residual_ridge_from_prediction_parts,
    position_daily_feature_batch,
)
from src.training.anchors import (
    apply_moe_output_anchor_experts,
    build_moe_output_anchor_fixed_expert_delta,
)


PENALTY_NAMES = ["level", "delta", "d2_match", "diff_amp"]


def _model(**kwargs) -> ClusterwisePredResidualMoE:
    cfg = {
        "num_clusters": 1,
        "num_penalties": len(PENALTY_NAMES),
        "input_len": 8,
        "pred_len": 8,
        "hidden_dim": 4,
        "use_y_base_input": True,
        "intervention_enable": False,
        "penalty_names": PENALTY_NAMES,
        "named_output_projection_enable": True,
        "named_output_projection_fixed_alpha": True,
    }
    cfg.update(kwargs)
    return ClusterwisePredResidualMoE(**cfg)


def test_named_projection_enforces_each_adapter_output_space() -> None:
    model = _model(residual_clip=0.0)
    torch.manual_seed(7)
    raw = torch.randn(2, 3, len(PENALTY_NAMES), 8)
    base = torch.randn(2, 3, 8)

    projected = model._project_named_residuals(raw, base)

    level = projected[:, :, 0, :]
    assert torch.allclose(level.diff(dim=-1), torch.zeros_like(level.diff(dim=-1)), atol=1.0e-6)

    delta = projected[:, :, 1, :]
    assert torch.allclose(delta.mean(dim=-1), torch.zeros_like(delta[..., 0]), atol=1.0e-6)

    d2 = projected[:, :, 2, :]
    trend = torch.linspace(-1.0, 1.0, 8)
    assert torch.allclose(d2.mean(dim=-1), torch.zeros_like(d2[..., 0]), atol=1.0e-6)
    assert torch.allclose((d2 * trend).sum(dim=-1), torch.zeros_like(d2[..., 0]), atol=1.0e-5)

    diff_amp = projected[:, :, 3, :]
    carrier = base - base.mean(dim=-1, keepdim=True)
    coef = (diff_amp * carrier).sum(dim=-1) / carrier.pow(2).sum(dim=-1).clamp_min(1.0e-12)
    reconstructed = coef.unsqueeze(-1) * carrier
    assert torch.allclose(diff_amp, reconstructed, atol=1.0e-6)
    candidate = base + diff_amp
    assert torch.allclose(candidate.mean(dim=-1), base.mean(dim=-1), atol=1.0e-6)
    assert bool((coef.abs() <= 0.5 + 1.0e-6).all())


def test_named_projection_can_reduce_delta_and_d2_to_bounded_carrier_scales() -> None:
    model = _model(
        residual_clip=0.0,
        named_output_projection_carrier_names=["delta", "d2_match"],
    )
    torch.manual_seed(17)
    raw = torch.randn(2, 3, len(PENALTY_NAMES), 8)
    base = torch.randn(2, 3, 8)

    projected = model._project_named_residuals(raw, base)
    carriers = {
        1: base - base.mean(dim=-1, keepdim=True),
        2: model._remove_affine_component(base),
    }
    for index, carrier in carriers.items():
        residual = projected[:, :, index, :]
        denom = carrier.pow(2).sum(dim=-1).clamp_min(1.0e-12)
        coef = (residual * carrier).sum(dim=-1) / denom
        reconstructed = coef.unsqueeze(-1) * carrier
        assert torch.allclose(residual, reconstructed, atol=1.0e-6)
        assert bool((coef.abs() <= 0.5 + 1.0e-6).all())


def test_named_projection_patch_len_is_independent_from_router_patch_len() -> None:
    model = _model(
        pred_len=8,
        input_len=8,
        residual_clip=0.0,
        shared_across_clusters=True,
        named_output_projection_patch_len=4,
        patch_router_cfg={
            "enable": True,
            "patch_len": 2,
            "hidden_dim": 4,
            "allow_skip": True,
        },
    )
    raw = torch.zeros(1, 1, len(PENALTY_NAMES), 8)
    raw[:, :, 0, :4] = 1.0
    raw[:, :, 0, 4:] = 3.0
    base = torch.randn(1, 1, 8)

    projected = model._project_named_residuals(raw, base)

    assert torch.equal(projected[:, :, 0, :4], torch.ones(1, 1, 4))
    assert torch.equal(projected[:, :, 0, 4:], torch.full((1, 1, 4), 3.0))
    assert model.patch_router is not None
    assert model.patch_router.patch_len == 2
    assert model.named_output_projection_patch_len == 4


def test_direct_attribute_supervision_trains_all_adapters_despite_hard_route() -> None:
    torch.manual_seed(11)
    model = _model(residual_clip=0.0)
    x = torch.randn(4, 2, 8)
    base = torch.randn(4, 2, 8)
    target = base + torch.linspace(-1.0, 1.5, 8).view(1, 1, 8)
    target = target + 0.3 * torch.sin(torch.arange(8, dtype=target.dtype)).view(1, 1, 8)
    cluster_id = torch.zeros(2, dtype=torch.long)
    hard_mask = torch.zeros(4, 1, len(PENALTY_NAMES))
    hard_mask[:, :, 0] = 1.0

    pred_out = model(x, base, cluster_id, hard_mask)
    loss_bk = _pred_residual_candidate_supervision_loss(
        y_base_bch=base,
        pred_out=pred_out,
        y_bch=target,
        cluster_id_c=cluster_id,
        K=1,
        penalty_names=PENALTY_NAMES,
        penalty_fns=build_penalty_bank(PENALTY_NAMES, jump_thr=1.0),
        only_allowed=False,
        loss_kind="direct_attribute",
        include_intervention=False,
        include_selector=False,
        include_patch_route=False,
    )
    assert loss_bk is not None
    loss_bk.mean().backward()

    grad_norms = [float(model.b2[p].grad.abs().sum().item()) for p in range(len(PENALTY_NAMES))]
    assert all(value > 0.0 for value in grad_norms), grad_norms


def test_periodic_anchor_is_reserved_exact_expert_and_survives_skip_clip_and_alpha() -> None:
    model = _model(
        residual_clip=0.01,
        init_alpha=-20.0,
        alpha_scale=0.01,
        periodic_anchor_expert_enable=True,
        periodic_anchor_expert_scale=1.0,
    )
    x = torch.randn(2, 1, 8)
    base = torch.randn(2, 1, 8)
    fixed_delta = torch.linspace(-0.4, 0.7, 8).view(1, 1, 8).expand_as(base)
    cluster_id = torch.zeros(1, dtype=torch.long)
    no_pkr_route = torch.zeros(2, 1, len(PENALTY_NAMES))
    skip = torch.ones(2, 1)

    out = model(
        x,
        base,
        cluster_id,
        no_pkr_route,
        skip_bk=skip,
        fixed_expert_delta_bch=fixed_delta,
    )

    assert torch.equal(out["candidate_base_bch"], base + fixed_delta)
    assert torch.equal(out["y_final"], base + fixed_delta)
    assert torch.equal(out["periodic_expert_branch_bch"], fixed_delta)
    assert torch.equal(out["periodic_expert_route_bc"], torch.ones_like(base[..., 0]))


def test_new_features_are_bit_exact_noops_when_disabled() -> None:
    torch.manual_seed(23)
    legacy = ClusterwisePredResidualMoE(
        num_clusters=1,
        num_penalties=1,
        input_len=4,
        pred_len=4,
        hidden_dim=3,
        penalty_names=["level"],
        intervention_enable=False,
    )
    explicit_off = ClusterwisePredResidualMoE(
        num_clusters=1,
        num_penalties=1,
        input_len=4,
        pred_len=4,
        hidden_dim=3,
        penalty_names=["level"],
        intervention_enable=False,
        named_output_projection_enable=False,
        periodic_anchor_expert_enable=False,
    )
    explicit_off.load_state_dict(legacy.state_dict(), strict=True)
    x = torch.randn(3, 2, 4)
    base = torch.randn(3, 2, 4)
    cluster_id = torch.zeros(2, dtype=torch.long)
    mask = torch.ones(3, 1, 1)

    legacy_out = legacy(x, base, cluster_id, mask)
    explicit_out = explicit_off(x, base, cluster_id, mask)

    assert torch.equal(legacy_out["y_final"], explicit_out["y_final"])
    assert torch.equal(legacy_out["residuals"], explicit_out["residuals"])
    assert torch.equal(legacy_out["branches"], explicit_out["branches"])


def test_corrector_channel_identity_distinguishes_identical_channel_inputs() -> None:
    model = _model(
        num_channels=3,
        use_channel_identity_features=True,
        named_output_projection_enable=False,
    )
    x = torch.zeros(2, 3, 8)
    base = torch.zeros(2, 3, 8)

    features = model._input_features(x, base)

    assert features.shape[-1] == model.input_dim
    expected_identity = torch.eye(3).unsqueeze(0).expand(2, -1, -1)
    assert torch.equal(features[..., -3:], expected_identity)
    assert not torch.equal(features[:, 0], features[:, 1])


def test_corrector_channel_identity_requires_channel_count() -> None:
    try:
        _model(
            num_channels=0,
            use_channel_identity_features=True,
        )
    except ValueError as exc:
        assert "num_channels" in str(exc)
    else:
        raise AssertionError("expected channel identity validation to fail")


def test_position_daily_residual_ridge_fits_and_applies_prediction_parts() -> None:
    idx = torch.arange(32, dtype=torch.long)
    features = position_daily_feature_batch(idx, input_len=8, harmonics=2, period=8)
    true_coef = torch.zeros(2, features.shape[1], 3, dtype=torch.float64)
    true_coef[:, 0, :] = torch.tensor([[0.2, -0.1, 0.3], [-0.4, 0.2, 0.1]])
    true_coef[:, 1, :] = 0.15
    pred = torch.zeros(32, 2, 3)
    target = torch.einsum("bf,cfh->bch", features, true_coef).float()

    fitted, summary = fit_position_daily_residual_ridge_from_prediction_parts(
        idx_parts=[idx],
        y_true_parts=[target],
        y_pred_parts=[pred],
        input_len=8,
        cfg={"daily_harmonics": 2, "daily_period": 8, "ridge": 0.0},
    )
    corrected = apply_position_daily_residual_ridge(
        pred,
        query_start_abs_b=idx,
        input_len=8,
        coef_cfh=fitted,
        cfg={"daily_harmonics": 2, "daily_period": 8},
    )

    assert summary["fit_windows"] == 32
    assert torch.allclose(corrected, target, atol=1.0e-5)


def test_position_daily_residual_expert_runs_inside_pred_residual_forward() -> None:
    model = _model(
        num_channels=2,
        position_daily_residual_expert_enable=True,
        position_daily_residual_period=8,
        position_daily_residual_harmonics=2,
    )
    coef = torch.zeros(2, 5, 8)
    coef[:, 0, :] = torch.tensor([[0.25] * 8, [-0.5] * 8])
    coef[:, 1, :] = 0.1
    model.set_position_daily_residual_expert(coef, period=8, harmonics=2)

    base = torch.randn(3, 2, 8)
    query_start = torch.tensor([0, 1, 5], dtype=torch.long)
    out = model(
        torch.randn(3, 2, 8),
        base,
        torch.zeros(2, dtype=torch.long),
        torch.zeros(3, 1, len(PENALTY_NAMES)),
        query_start_abs_b=query_start,
    )
    expected = apply_position_daily_residual_ridge(
        base,
        query_start_abs_b=query_start,
        input_len=8,
        coef_cfh=coef,
        cfg={"daily_harmonics": 2, "daily_period": 8},
    )

    assert torch.equal(out["candidate_base_bch"], base)
    assert torch.allclose(out["y_final"], expected, atol=1.0e-6)
    assert torch.allclose(
        out["position_daily_residual_expert_branch_bch"], expected - base,
        atol=1.0e-6,
    )


def test_anchor_ridge_gate_jointly_scales_both_internal_fixed_branches() -> None:
    model = _model(
        num_channels=2,
        periodic_anchor_expert_enable=True,
        position_daily_residual_expert_enable=True,
        position_daily_residual_period=8,
        position_daily_residual_harmonics=2,
        anchor_ridge_gate_cfg={"enable": True, "hidden_dim": 4},
    )
    coef = torch.zeros(2, 5, 8)
    coef[:, 0, :] = 0.4
    model.set_position_daily_residual_expert(coef, period=8, harmonics=2)
    assert model.anchor_ridge_gate is not None
    with torch.no_grad():
        for parameter in model.anchor_ridge_gate.parameters():
            parameter.zero_()
        model.anchor_ridge_gate[2].bias.copy_(torch.tensor([10.0, -10.0]))
    model.set_anchor_ridge_gate_normalization(
        torch.zeros(model.input_dim),
        torch.ones(model.input_dim),
        fitted=True,
    )
    model.eval()

    base = torch.randn(3, 2, 8)
    anchor = torch.full_like(base, 0.2)
    out = model(
        torch.randn(3, 2, 8),
        base,
        torch.zeros(2, dtype=torch.long),
        torch.zeros(3, 1, len(PENALTY_NAMES)),
        query_start_abs_b=torch.tensor([0, 1, 5]),
        fixed_expert_delta_bch=anchor,
    )

    assert torch.allclose(
        out["anchor_ridge_gate_weights_bc2"],
        torch.tensor([1.0, 0.0]).view(1, 1, 2).expand(3, 2, 2),
    )
    assert torch.allclose(out["y_final"], base + anchor, atol=1.0e-6)
    assert torch.equal(out["candidate_base_bch"], base + anchor)


def test_output_anchor_moves_inside_moe_without_posthoc_double_application() -> None:
    base = torch.zeros(1, 1, 2)
    x = torch.zeros(1, 1, 2)
    query_start = torch.zeros(1, dtype=torch.long)
    stat_table = torch.tensor([[1.0], [2.0]])
    cfg = {
        "enable": True,
        "pred_side_residual": {
            "enable": True,
            "periodic_anchor_expert": True,
        },
        "train_stat_anchor_expert": {
            "enable": True,
            "mode": "phase_mean",
            "alpha": 1.0,
        },
    }

    fixed_delta = build_moe_output_anchor_fixed_expert_delta(
        base,
        x_bcl=x,
        query_start_abs_b=query_start,
        input_len=2,
        moe_cfg=cfg,
        moe_enable=True,
        train_stat_anchor_pc=stat_table,
    )
    assert fixed_delta is not None
    assert torch.equal(fixed_delta, torch.tensor([[[1.0, 2.0]]]))

    posthoc = apply_moe_output_anchor_experts(
        base + fixed_delta,
        base_pred_bch=base,
        x_bcl=x,
        query_start_abs_b=query_start,
        input_len=2,
        moe_cfg=cfg,
        moe_enable=True,
        train_stat_anchor_pc=stat_table,
    )
    assert torch.equal(posthoc, base + fixed_delta)


def test_periodic_guard_does_not_disable_posthoc_anchor_without_residual_moe() -> None:
    base = torch.zeros(1, 1, 2)
    x = torch.zeros(1, 1, 2)
    query_start = torch.zeros(1, dtype=torch.long)
    stat_table = torch.tensor([[1.0], [2.0]])
    cfg = {
        "enable": True,
        "pred_side_residual": {
            "enable": False,
            "periodic_anchor_expert": True,
        },
        "train_stat_anchor_expert": {
            "enable": True,
            "mode": "phase_mean",
            "alpha": 1.0,
        },
    }

    out = apply_moe_output_anchor_experts(
        base,
        base_pred_bch=base,
        x_bcl=x,
        query_start_abs_b=query_start,
        input_len=2,
        moe_cfg=cfg,
        moe_enable=True,
        train_stat_anchor_pc=stat_table,
    )
    assert torch.equal(out, torch.tensor([[[1.0, 2.0]]]))


class _FixedProfileBackbone(torch.nn.Module):
    def __init__(self, profile_h: torch.Tensor):
        super().__init__()
        self.register_buffer("profile_h", profile_h)

    def forward(self, x_bcl: torch.Tensor, cluster_id_c: torch.Tensor) -> torch.Tensor:
        return self.profile_h.view(1, 1, -1).expand(
            int(x_bcl.shape[0]),
            int(x_bcl.shape[1]),
            -1,
        )


class _AllFourGate(torch.nn.Module):
    def forward(self, feat_bkf: torch.Tensor, **kwargs):
        batch, clusters, _ = feat_bkf.shape
        mask = torch.ones(batch, clusters, 4, dtype=feat_bkf.dtype, device=feat_bkf.device)
        skip = torch.zeros(batch, clusters, dtype=feat_bkf.dtype, device=feat_bkf.device)
        return mask, mask / 4.0, skip, skip


class _PerfectNamedCandidates(torch.nn.Module):
    def forward(
        self,
        x_bcl: torch.Tensor,
        y_base_bch: torch.Tensor,
        cluster_id_c: torch.Tensor,
        mask_bkp: torch.Tensor,
        **kwargs,
    ):
        target_h = torch.tensor(
            [1.0, 2.0, 0.0, 3.0],
            device=x_bcl.device,
            dtype=x_bcl.dtype,
        )
        target = target_h.view(1, 1, -1).expand_as(y_base_bch)
        level = target.mean(dim=-1, keepdim=True).expand_as(target)
        delta = target - target.mean(dim=-1, keepdim=True)
        residuals = torch.stack([level, delta, target, target], dim=2)
        route = mask_bkp[:, cluster_id_c, :]
        branches = route.unsqueeze(-1) * residuals
        return {
            "y_final": y_base_bch + branches.sum(dim=2),
            "residuals": residuals,
            "branches": branches,
            "route_bcp": route,
            "effective_route_bcp": route,
            "intervention_bcp": torch.ones_like(route),
            "selector_bcp": torch.ones_like(route),
            "alpha_cp": torch.ones(
                int(x_bcl.shape[1]),
                4,
                dtype=x_bcl.dtype,
                device=x_bcl.device,
            ),
            "candidate_base_bch": y_base_bch,
        }


def test_gradient_isolation_diagnostic_is_diagonal_and_state_preserving() -> None:
    torch.manual_seed(101)
    model = _model(residual_clip=0.0)
    profile = torch.tensor([-0.8, -0.2, 0.6, 1.0, 0.4, -0.5, -0.9, 0.2])
    backbone = _FixedProfileBackbone(profile)
    x = torch.randn(3, 2, 8)
    base = profile.view(1, 1, -1)
    target = 1.4 * base + 0.35 * torch.sin(torch.arange(8)).view(1, 1, -1)
    target = target.expand(3, 2, -1).clone()
    idx = torch.arange(3, dtype=torch.long)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    payload = evaluate_adapter_gradient_isolation(
        model=backbone,
        pred_residual=model,
        loader=[(x, target, idx)],
        cluster_id_c=torch.zeros(2, dtype=torch.long),
        K=1,
        moe_cfg={
            "enable": True,
            "explainability": {
                "adapter_specialization": {"enable": True},
            },
        },
        device=torch.device("cpu"),
        penalty_names=PENALTY_NAMES,
        split_name="train_holdout",
        max_batches=1,
    )

    assert payload is not None
    assert payload["passed"] is True
    assert payload["diagonal_min"] > 0.0
    assert payload["off_diagonal_max"] == 0.0
    assert payload["max_parameter_change"] == 0.0
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


def test_explainability_reports_named_specialization_matrix() -> None:
    x = torch.zeros(2, 1, 4)
    target_h = torch.tensor([1.0, 2.0, 0.0, 3.0])
    y = target_h.view(1, 1, -1).expand(2, 1, -1).clone()
    idx = torch.arange(2, dtype=torch.long)
    zero_backbone = _FixedProfileBackbone(torch.zeros(4))
    penalty_fns = build_penalty_bank(PENALTY_NAMES, jump_thr=1.0)

    payload = evaluate_penalty_explainability(
        model=zero_backbone,
        gate=_AllFourGate(),
        pred_residual=_PerfectNamedCandidates(),
        loader=[(x, y, idx)],
        cluster_id_c=torch.zeros(1, dtype=torch.long),
        K=1,
        moe_cfg={
            "enable": True,
            "explainability": {
                "adapter_specialization": {"enable": True},
            },
        },
        device=torch.device("cpu"),
        penalty_names=PENALTY_NAMES,
        penalty_fns=penalty_fns,
        penalty_scale=None,
        select_ranks=None,
        gate_soft_weight=0.0,
        split_name="val",
    )

    assert payload is not None
    semantic = payload["named_attribute_specialization"]
    assert semantic["expert_names"] == PENALTY_NAMES
    assert semantic["metric_names"] == PENALTY_NAMES
    assert semantic["all_diagonal_improved"] is True
    assert len(semantic["relative_improvement_pct_expert_by_metric"]) == 4
    assert all(row["improved"] for row in semantic["diagonal"])
    exact = payload["named_penalty_specialization"]
    assert exact["all_diagonal_improved"] is True
    assert all(
        row["positive_oracle_relative_improvement_pct_overall"] > 0.0
        for row in exact["diagonal"]
    )
