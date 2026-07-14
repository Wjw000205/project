import copy

import torch

from src.models.residual_moe import ClusterwisePredResidualMoE


def _patch_router_cfg(*, compositional: bool) -> dict:
    return {
        "enable": True,
        "patch_len": 2,
        "hidden_dim": 4,
        "topk": 1,
        "allow_skip": True,
        "use_base_forecast": True,
        "compositional_periodic_gate": {"enable": compositional},
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
    }


def _model(*, compositional: bool, penalties: int = 2) -> ClusterwisePredResidualMoE:
    return ClusterwisePredResidualMoE(
        num_clusters=1,
        num_penalties=penalties,
        input_len=4,
        pred_len=4,
        hidden_dim=4,
        intervention_enable=False,
        shared_across_clusters=True,
        patch_router_cfg=_patch_router_cfg(compositional=compositional),
        penalty_names=[f"free_{p}" for p in range(penalties)],
        named_output_projection_enable=True,
        named_output_projection_fixed_alpha=True,
        named_output_projection_scale_by_name={
            f"free_{p}": 1.0 for p in range(penalties)
        },
        periodic_anchor_expert_enable=True,
    )


def _forward(model: ClusterwisePredResidualMoE) -> dict:
    x = torch.tensor([[[0.0, 1.0, -1.0, 0.5]]])
    base = torch.zeros(1, 1, 4)
    return model(
        x,
        base,
        torch.zeros(1, dtype=torch.long),
        torch.ones(1, 1, model.P),
        fixed_expert_delta_bch=torch.full_like(base, 2.0),
    )


def test_compositional_periodic_gate_default_off_is_bit_exact() -> None:
    cfg_without_feature = _patch_router_cfg(compositional=False)
    del cfg_without_feature["compositional_periodic_gate"]
    cfg_explicit_off = copy.deepcopy(cfg_without_feature)
    cfg_explicit_off["compositional_periodic_gate"] = {"enable": False}

    torch.manual_seed(17)
    omitted = ClusterwisePredResidualMoE(
        num_clusters=1,
        num_penalties=2,
        input_len=4,
        pred_len=4,
        hidden_dim=4,
        intervention_enable=False,
        shared_across_clusters=True,
        patch_router_cfg=cfg_without_feature,
        periodic_anchor_expert_enable=True,
    ).eval()
    torch.manual_seed(17)
    explicit_off = ClusterwisePredResidualMoE(
        num_clusters=1,
        num_penalties=2,
        input_len=4,
        pred_len=4,
        hidden_dim=4,
        intervention_enable=False,
        shared_across_clusters=True,
        patch_router_cfg=cfg_explicit_off,
        periodic_anchor_expert_enable=True,
    ).eval()

    omitted_state = omitted.state_dict()
    explicit_state = explicit_off.state_dict()
    assert omitted_state.keys() == explicit_state.keys()
    for name in omitted_state:
        assert torch.equal(omitted_state[name], explicit_state[name]), name

    omitted_out = _forward(omitted)
    explicit_out = _forward(explicit_off)
    assert omitted_out.keys() == explicit_out.keys()
    for name in omitted_out:
        assert torch.equal(omitted_out[name], explicit_out[name]), name
    assert "patch_action_index_bcq" not in omitted_out
    assert "patch_periodic_route_bcq" not in omitted_out


def test_zero_initialized_periodic_heads_preserve_the_legacy_adapter_action() -> None:
    torch.manual_seed(23)
    legacy = _model(compositional=False, penalties=4).eval()
    torch.manual_seed(23)
    compositional = _model(compositional=True, penalties=4).eval()
    compositional.load_state_dict(legacy.state_dict(), strict=False)

    assert legacy.patch_router is not None
    assert compositional.patch_router is not None
    with torch.no_grad():
        legacy.patch_router.b_risk_mse_utility.copy_(
            torch.tensor([2.0, -2.0, -2.0, -2.0])
        )
        legacy.patch_router.b_risk_mae_utility.copy_(
            torch.tensor([2.0, -2.0, -2.0, -2.0])
        )
        compositional.patch_router.b_risk_mse_utility.copy_(
            legacy.patch_router.b_risk_mse_utility
        )
        compositional.patch_router.b_risk_mae_utility.copy_(
            legacy.patch_router.b_risk_mae_utility
        )

    legacy_out = _forward(legacy)
    compositional_out = _forward(compositional)
    for name in (
        "y_final",
        "patch_route_bcph",
        "patch_skip_bcq",
        "patch_selected_penalty_index_bcq",
        "patch_penalty_utility_scores_bcqp",
    ):
        assert torch.equal(legacy_out[name], compositional_out[name]), name
    assert torch.equal(
        compositional_out["patch_action_index_bcq"],
        torch.full((1, 1, 2), 2, dtype=torch.long),
    )
    assert torch.equal(
        compositional_out["patch_periodic_route_bcq"],
        torch.ones(1, 1, 2),
    )
    assert compositional_out["patch_route_bcph"].shape == (1, 1, 4, 4)
    assert compositional_out["patch_action_scores_bcqa"].shape == (1, 1, 2, 6)
    assert not bool(compositional_out["patch_periodic_only_bcq"].any().item())
    assert not bool(compositional_out["patch_backbone_route_bcq"].any().item())


def test_compositional_periodic_gate_selects_backbone_periodic_or_combined_output() -> None:
    model = _model(compositional=True, penalties=1).eval()
    assert model.patch_router is not None
    router = model.patch_router
    with torch.no_grad():
        model.W1[0].zero_()
        model.b1[0].zero_()
        model.W2[0].zero_()
        model.b2[0].fill_(3.0)
        router.W_periodic_mse_utility.zero_()
        router.W_periodic_mae_utility.zero_()
        router.W_risk_mse_utility.zero_()
        router.W_risk_mae_utility.zero_()

        router.b_periodic_mse_utility.fill_(-2.0)
        router.b_periodic_mae_utility.fill_(-2.0)
        router.b_risk_mse_utility.zero_()
        router.b_risk_mae_utility.zero_()
    backbone = _forward(model)
    assert torch.equal(
        backbone["patch_action_index_bcq"],
        torch.zeros(1, 1, 2, dtype=torch.long),
    )
    assert torch.equal(backbone["y_final"], torch.zeros(1, 1, 4))

    with torch.no_grad():
        router.b_periodic_mse_utility.fill_(2.0)
        router.b_periodic_mae_utility.fill_(2.0)
        router.b_risk_mse_utility.fill_(-2.0)
        router.b_risk_mae_utility.fill_(-2.0)
    periodic = _forward(model)
    assert torch.equal(
        periodic["patch_action_index_bcq"],
        torch.ones(1, 1, 2, dtype=torch.long),
    )
    assert bool(periodic["patch_periodic_only_bcq"].all().item())
    assert torch.equal(periodic["y_final"], torch.full((1, 1, 4), 2.0))

    with torch.no_grad():
        router.b_periodic_mse_utility.fill_(0.5)
        router.b_periodic_mae_utility.fill_(0.5)
        router.b_risk_mse_utility.fill_(2.0)
        router.b_risk_mae_utility.fill_(2.0)
    combined = _forward(model)
    assert torch.equal(
        combined["patch_action_index_bcq"],
        torch.full((1, 1, 2), 2, dtype=torch.long),
    )
    assert torch.equal(combined["patch_route_bcph"], torch.ones(1, 1, 1, 4))
    assert torch.equal(combined["y_final"], torch.full((1, 1, 4), 5.0))
    assert combined["patch_action_scores_bcqa"].shape == (1, 1, 2, 3)


def test_periodic_signed_utility_heads_are_trainable() -> None:
    model = _model(compositional=True, penalties=1).train()
    assert model.patch_router is not None
    router = model.patch_router
    owner_param_ids = {id(parameter) for parameter in model.get_cluster_params(0)}
    saved_state = model.get_cluster_state(0)
    out = _forward(model)
    loss = (
        out["patch_periodic_mse_utility_scores_bcq"].mean()
        + out["patch_periodic_mae_utility_scores_bcq"].mean()
    )
    loss.backward()

    for parameter in (
        router.W_periodic_mse_utility,
        router.b_periodic_mse_utility,
        router.W_periodic_mae_utility,
        router.b_periodic_mae_utility,
    ):
        assert parameter is not None
        assert id(parameter) in owner_param_ids
        assert parameter.grad is not None
        assert float(parameter.grad.abs().sum().item()) > 0.0
    assert "patch_router.W_periodic_mse_utility" in saved_state
    assert "patch_router.b_periodic_mse_utility" in saved_state
    assert "patch_router.W_periodic_mae_utility" in saved_state
    assert "patch_router.b_periodic_mae_utility" in saved_state


def test_compositional_soft_route_mixes_backbone_periodic_and_adapter_actions() -> None:
    cfg = _patch_router_cfg(compositional=True)
    cfg["inference_route_mode"] = "soft"
    model = ClusterwisePredResidualMoE(
        num_clusters=1,
        num_penalties=1,
        input_len=4,
        pred_len=4,
        hidden_dim=4,
        intervention_enable=False,
        shared_across_clusters=True,
        patch_router_cfg=cfg,
        penalty_names=["free_0"],
        named_output_projection_enable=True,
        named_output_projection_fixed_alpha=True,
        named_output_projection_scale_by_name={"free_0": 1.0},
        periodic_anchor_expert_enable=True,
    ).eval()
    assert model.patch_router is not None
    with torch.no_grad():
        model.W1[0].zero_()
        model.b1[0].zero_()
        model.W2[0].zero_()
        model.b2[0].fill_(3.0)
        model.patch_router.W_periodic_mse_utility.zero_()
        model.patch_router.W_periodic_mae_utility.zero_()
        model.patch_router.W_risk_mse_utility.zero_()
        model.patch_router.W_risk_mae_utility.zero_()
        model.patch_router.b_periodic_mse_utility.zero_()
        model.patch_router.b_periodic_mae_utility.zero_()
        model.patch_router.b_risk_mse_utility.zero_()
        model.patch_router.b_risk_mae_utility.zero_()

    out = _forward(model)

    assert torch.allclose(
        out["patch_action_probs_bcqa"],
        torch.full((1, 1, 2, 3), 1.0 / 3.0),
    )
    assert torch.allclose(
        out["patch_periodic_route_bcq"],
        torch.full((1, 1, 2), 2.0 / 3.0),
    )
    assert torch.allclose(
        out["patch_route_bcph"],
        torch.full((1, 1, 1, 4), 1.0 / 3.0),
    )
    assert torch.allclose(out["y_final"], torch.full((1, 1, 4), 7.0 / 3.0))


def test_compositional_training_soft_route_backpropagates_forecast_loss() -> None:
    cfg = _patch_router_cfg(compositional=True)
    cfg["training_route_mode"] = "soft"
    model = ClusterwisePredResidualMoE(
        num_clusters=1,
        num_penalties=1,
        input_len=4,
        pred_len=4,
        hidden_dim=4,
        intervention_enable=False,
        shared_across_clusters=True,
        patch_router_cfg=cfg,
        periodic_anchor_expert_enable=True,
    ).train()
    assert model.patch_router is not None
    with torch.no_grad():
        model.b2[0].fill_(1.0)

    out = _forward(model)
    out["y_final"].square().mean().backward()

    assert model.patch_router.W_periodic_mse_utility.grad is not None
    assert model.patch_router.W_risk_mse_utility.grad is not None
    assert float(model.patch_router.W_periodic_mse_utility.grad.abs().sum()) > 0.0
    assert float(model.patch_router.W_risk_mse_utility.grad.abs().sum()) > 0.0
