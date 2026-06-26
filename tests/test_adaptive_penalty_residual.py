from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.moe_gate import ClusterwiseMoEGate
from src.models.residual_moe import ClusterwisePredResidualMoE


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
