import torch

from src.models.moe_gate import ClusterwiseMoEGate
from src.train import (
    _gate_cluster_params,
    _make_cluster_optimizer_param_groups,
    _mask_gate_grads_after_epoch,
    _optimizer_slot_active,
    _resolve_overfit_diagnostic_range,
    _training_cluster_weight,
)


def test_overfit_diagnostic_range_is_disabled_by_default() -> None:
    assert _resolve_overfit_diagnostic_range(100, {}) is None
    assert _resolve_overfit_diagnostic_range(100, {"enable": False}) is None


def test_overfit_diagnostic_range_supports_fixed_positions() -> None:
    assert _resolve_overfit_diagnostic_range(
        100,
        {"enable": True, "num_windows": 20, "position": "head"},
    ) == (0, 20)
    assert _resolve_overfit_diagnostic_range(
        100,
        {"enable": True, "num_windows": 20, "position": "center"},
    ) == (40, 60)
    assert _resolve_overfit_diagnostic_range(
        100,
        {"enable": True, "num_windows": 20, "position": "tail"},
    ) == (80, 100)
    assert _resolve_overfit_diagnostic_range(
        100,
        {"enable": True, "num_windows": 20, "start_idx": 7},
    ) == (7, 27)


def test_overfit_diagnostic_range_rejects_out_of_bounds_requests() -> None:
    for cfg in (
        {"enable": True, "num_windows": 0},
        {"enable": True, "num_windows": 101},
        {"enable": True, "num_windows": 20, "start_idx": 90},
        {"enable": True, "num_windows": 20, "position": "random"},
    ):
        try:
            _resolve_overfit_diagnostic_range(100, cfg)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {cfg}")


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


def test_backbone_lr_override_separates_base_group_from_moe_params() -> None:
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
        base_lr=3.0e-5,
    )

    assert len(groups) == 3
    assert groups[0]["params"] == [base_param]
    assert groups[0]["weight_decay"] == 1.0e-3
    assert groups[0]["lr"] == 3.0e-5
    assert groups[1]["params"] == [gate_param]
    assert "lr" not in groups[1]
    assert groups[2]["params"] == [residual_param]


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


def test_shared_gate_params_are_owned_by_cluster_zero_only() -> None:
    gate = ClusterwiseMoEGate(
        num_clusters=3,
        feat_dim=3,
        num_penalties=2,
        hidden_dim=4,
        allow_skip=True,
        shared_across_clusters=True,
    )

    owner_params = _gate_cluster_params(gate, 0)

    assert any(param is gate.W1[0] for param in owner_params)
    assert any(param is gate.b_skip[0] for param in owner_params)
    assert _gate_cluster_params(gate, 1) == []
    assert _gate_cluster_params(gate, 2) == []


def test_shared_moe_optimizer_owner_stays_active_while_any_cluster_is_active() -> None:
    stopped = torch.tensor([True, False, True])

    assert _optimizer_slot_active(stopped, 0, shared_moe=True) is True
    assert _optimizer_slot_active(stopped, 1, shared_moe=True) is True
    assert _optimizer_slot_active(stopped, 2, shared_moe=True) is False
    assert _optimizer_slot_active(stopped, 0, shared_moe=False) is False
    assert _optimizer_slot_active(stopped, 1, shared_moe=False) is True
    assert _optimizer_slot_active(torch.tensor([True, True, True]), 0, shared_moe=True) is False


def test_shared_moe_training_weight_drops_stopped_cluster_losses() -> None:
    weight = torch.tensor([0.2, 0.3, 0.5])
    stopped = torch.tensor([True, False, False])

    shared_weight = _training_cluster_weight(weight, stopped, shared_moe=True)
    default_weight = _training_cluster_weight(weight, stopped, shared_moe=False)

    assert torch.allclose(shared_weight, torch.tensor([0.0, 0.3, 0.5]))
    assert torch.equal(default_weight, weight)


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
