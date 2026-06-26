import torch

from src.models.moe_gate import ClusterwiseMoEGate
from src.train import (
    _gate_cluster_params,
    _make_cluster_optimizer_param_groups,
    _mask_gate_grads_after_epoch,
)


def test_pred_residual_params_default_to_base_weight_decay_group() -> None:
    base_param = torch.nn.Parameter(torch.ones(1))
    gate_param = torch.nn.Parameter(torch.ones(1) * 2.0)
    residual_param = torch.nn.Parameter(torch.ones(1) * 3.0)

    groups = _make_cluster_optimizer_param_groups(
        base_params=[base_param],
        gate_params=[gate_param],
        pred_residual_params=[residual_param],
        dynamic_lambda_params=[],
        learnable_lambda_params=[],
        base_weight_decay=1.0e-3,
        moe_weight_decay=None,
        pred_residual_weight_decay=None,
    )

    assert len(groups) == 2
    assert groups[0]["params"] == [base_param, gate_param]
    assert groups[0]["weight_decay"] == 1.0e-3
    assert groups[1]["params"] == [residual_param]
    assert groups[1]["weight_decay"] == 1.0e-3


def test_pred_residual_weight_decay_can_be_overridden() -> None:
    residual_param = torch.nn.Parameter(torch.ones(1))

    groups = _make_cluster_optimizer_param_groups(
        base_params=[],
        gate_params=[],
        pred_residual_params=[residual_param],
        dynamic_lambda_params=[],
        learnable_lambda_params=[],
        base_weight_decay=1.0e-3,
        moe_weight_decay=None,
        pred_residual_weight_decay=2.0e-4,
    )

    assert groups == [{"params": [residual_param], "weight_decay": 2.0e-4}]


def test_moe_weight_decay_can_be_overridden_separately_from_backbone() -> None:
    base_param = torch.nn.Parameter(torch.ones(1))
    gate_param = torch.nn.Parameter(torch.ones(1) * 2.0)
    dynamic_param = torch.nn.Parameter(torch.ones(1) * 3.0)

    groups = _make_cluster_optimizer_param_groups(
        base_params=[base_param],
        gate_params=[gate_param],
        pred_residual_params=[],
        dynamic_lambda_params=[dynamic_param],
        learnable_lambda_params=[],
        base_weight_decay=1.0e-3,
        moe_weight_decay=0.0,
        pred_residual_weight_decay=None,
    )

    assert len(groups) == 2
    assert groups[0]["params"] == [base_param]
    assert groups[0]["weight_decay"] == 1.0e-3
    assert groups[1]["params"] == [gate_param, dynamic_param]
    assert groups[1]["weight_decay"] == 0.0


def test_pred_residual_defaults_to_moe_weight_decay_when_moe_is_overridden() -> None:
    gate_param = torch.nn.Parameter(torch.ones(1) * 2.0)
    residual_param = torch.nn.Parameter(torch.ones(1) * 3.0)

    groups = _make_cluster_optimizer_param_groups(
        base_params=[],
        gate_params=[gate_param],
        pred_residual_params=[residual_param],
        dynamic_lambda_params=[],
        learnable_lambda_params=[],
        base_weight_decay=1.0e-3,
        moe_weight_decay=0.0,
        pred_residual_weight_decay=None,
    )

    assert len(groups) == 2
    assert groups[0]["params"] == [gate_param]
    assert groups[0]["weight_decay"] == 0.0
    assert groups[1]["params"] == [residual_param]
    assert groups[1]["weight_decay"] == 0.0


def test_gate_cluster_params_include_skip_head_when_enabled() -> None:
    gate = ClusterwiseMoEGate(
        num_clusters=2,
        feat_dim=3,
        num_penalties=2,
        hidden_dim=4,
        allow_skip=True,
    )

    params = _gate_cluster_params(gate, 1)

    assert any(param is gate.W1[1] for param in params)
    assert any(param is gate.b1[1] for param in params)
    assert any(param is gate.W2[1] for param in params)
    assert any(param is gate.b2[1] for param in params)
    assert any(param is gate.W_skip[1] for param in params)
    assert any(param is gate.b_skip[1] for param in params)


def test_mask_gate_grads_after_epoch_freezes_gate_only_after_threshold() -> None:
    gate = ClusterwiseMoEGate(
        num_clusters=2,
        feat_dim=3,
        num_penalties=2,
        hidden_dim=4,
        allow_skip=True,
    )
    residual_param = torch.nn.Parameter(torch.ones(1))

    gate_params = _gate_cluster_params(gate, 0)
    for param in gate_params:
        param.grad = torch.ones_like(param)
    residual_param.grad = torch.ones_like(residual_param)

    assert _mask_gate_grads_after_epoch(
        gate=gate,
        epoch=2,
        freeze_after_epoch=2,
        stopped=torch.tensor([False, False]),
    ) is False
    assert all(param.grad is not None and torch.count_nonzero(param.grad).item() > 0 for param in gate_params)
    assert residual_param.grad is not None and torch.count_nonzero(residual_param.grad).item() > 0

    assert _mask_gate_grads_after_epoch(
        gate=gate,
        epoch=3,
        freeze_after_epoch=2,
        stopped=torch.tensor([False, False]),
    ) is True
    assert all(param.grad is not None and torch.count_nonzero(param.grad).item() == 0 for param in gate_params)
    assert residual_param.grad is not None and torch.count_nonzero(residual_param.grad).item() > 0
