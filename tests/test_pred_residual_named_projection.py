from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.penalties import build_penalty_bank
from src.models.residual_moe import ClusterwisePredResidualMoE
from src.training.selectors import _pred_residual_candidate_supervision_loss
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
