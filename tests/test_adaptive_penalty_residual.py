from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.residual_moe import ClusterwisePredResidualMoE


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
