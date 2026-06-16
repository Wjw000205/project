import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.cluster_predictor import build_cluster_predictor


def test_mlp_default_does_not_add_last_value_when_weights_are_zero():
    model = build_cluster_predictor(
        num_clusters=2,
        input_len=4,
        pred_len=3,
        model_cfg={"predictor": "mlp", "hidden_dim": 5, "dropout": 0.0},
    )
    for param in model.parameters():
        param.data.zero_()

    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [10.0, 12.0, 14.0, 16.0]]])
    cluster_id = torch.tensor([0, 1])

    y = model(x, cluster_id)

    assert torch.allclose(y, torch.zeros_like(y))


def test_mlp_residual_anchor_adds_last_value_when_weights_are_zero():
    model = build_cluster_predictor(
        num_clusters=2,
        input_len=4,
        pred_len=3,
        model_cfg={
            "predictor": "mlp",
            "hidden_dim": 5,
            "dropout": 0.0,
            "mlp_residual_anchor": True,
        },
    )
    for param in model.parameters():
        param.data.zero_()

    x = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [10.0, 12.0, 14.0, 16.0]],
            [[-1.0, 0.0, 1.0, 2.0], [5.0, 3.0, 1.0, -1.0]],
        ]
    )
    cluster_id = torch.tensor([0, 1])

    y = model(x, cluster_id)

    expected = x[..., -1:].expand(-1, -1, 3)
    assert torch.allclose(y, expected)


def test_mlp_residual_anchor_with_revin_keeps_shape_metadata():
    cluster_id = torch.tensor([0, 1])
    model = build_cluster_predictor(
        num_clusters=2,
        input_len=4,
        pred_len=3,
        model_cfg={
            "predictor": "mlp",
            "hidden_dim": 5,
            "dropout": 0.0,
            "revin": True,
            "mlp_residual_anchor": True,
            "temporal_basis_adapter": {
                "enable": True,
                "rank": 2,
                "scale": 0.15,
                "init": "zero_delta",
            },
        },
        num_channels=2,
        cluster_id_c=cluster_id,
    )

    x = torch.randn(2, 2, 4)
    y = model(x, cluster_id)

    assert y.shape == (2, 2, 3)
    assert model.L == 4
    assert model.H == 3


def test_seasonal_blend_adapter_can_be_disabled_with_zero_max_mix():
    cluster_id = torch.tensor([0, 1])
    model = build_cluster_predictor(
        num_clusters=2,
        input_len=4,
        pred_len=3,
        model_cfg={
            "predictor": "mlp",
            "hidden_dim": 5,
            "dropout": 0.0,
            "seasonal_blend_adapter": {
                "enable": True,
                "period": 4,
                "num_periods": 1,
                "max_mix": 0.0,
                "init_mix": 0.0,
            },
        },
        num_channels=2,
        cluster_id_c=cluster_id,
    )
    for param in model.parameters():
        param.data.zero_()

    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [10.0, 12.0, 14.0, 16.0]]])
    y = model(x, cluster_id)

    assert torch.allclose(y, torch.zeros_like(y))


def test_seasonal_blend_adapter_mixes_same_phase_history():
    cluster_id = torch.tensor([0, 1])
    model = build_cluster_predictor(
        num_clusters=2,
        input_len=4,
        pred_len=3,
        model_cfg={
            "predictor": "mlp",
            "hidden_dim": 5,
            "dropout": 0.0,
            "seasonal_blend_adapter": {
                "enable": True,
                "period": 4,
                "num_periods": 1,
                "max_mix": 0.5,
                "init_mix": 0.0,
            },
        },
        num_channels=2,
        cluster_id_c=cluster_id,
    )
    for param in model.base.parameters():
        param.data.zero_()
    for logit in model.mix_logit:
        logit.data.fill_(20.0)

    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [10.0, 12.0, 14.0, 16.0]]])
    y = model(x, cluster_id)

    expected = 0.5 * x[..., :3]
    assert torch.allclose(y, expected, atol=1.0e-6)


def test_seasonal_blend_adapter_loads_legacy_base_cluster_state():
    cluster_id = torch.tensor([0, 0, 1])
    base = build_cluster_predictor(
        num_clusters=2,
        input_len=4,
        pred_len=3,
        model_cfg={"predictor": "mlp", "hidden_dim": 5, "dropout": 0.0},
    )
    adapted = build_cluster_predictor(
        num_clusters=2,
        input_len=4,
        pred_len=3,
        model_cfg={
            "predictor": "mlp",
            "hidden_dim": 5,
            "dropout": 0.0,
            "seasonal_blend_adapter": {
                "enable": True,
                "period": 4,
                "num_periods": 1,
                "max_mix": 0.0,
                "init_mix": 0.0,
            },
        },
        num_channels=3,
        cluster_id_c=cluster_id,
    )
    for param in base.parameters():
        param.data.uniform_(-0.1, 0.1)
    for param in adapted.parameters():
        param.data.zero_()
    for k in range(2):
        adapted.load_cluster_state(k, base.get_cluster_state(k))

    x = torch.randn(2, 3, 4)

    assert torch.allclose(adapted(x, cluster_id), base(x, cluster_id))


def test_selected_channel_adapter_only_updates_configured_channels():
    cluster_id = torch.tensor([0, 0, 1])
    model = build_cluster_predictor(
        num_clusters=2,
        input_len=4,
        pred_len=3,
        model_cfg={
            "predictor": "channel_head_mlp",
            "hidden_dim": 5,
            "dropout": 0.0,
            "channel_head_residual": False,
            "selected_channel_adapter": {
                "enable": True,
                "channel_indices": [1],
                "rank": 1,
                "scale": 1.0,
                "init": "zero_delta",
            },
        },
        num_channels=3,
        cluster_id_c=cluster_id,
    )
    for param in model.parameters():
        param.data.zero_()
    model.down[0].data.fill_(1.0)
    model.up[0].data.fill_(1.0)

    x = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0], [5.0, 5.0, 5.0, 5.0]],
        ]
    )
    y = model(x, cluster_id)

    assert torch.allclose(y[:, 0, :], torch.zeros_like(y[:, 0, :]))
    assert torch.allclose(y[:, 2, :], torch.zeros_like(y[:, 2, :]))
    assert torch.max(torch.abs(y[:, 1, :])).item() > 1.0e-6


def test_channel_adapter_loads_legacy_base_cluster_state():
    cluster_id = torch.tensor([0, 0, 1])
    base = build_cluster_predictor(
        num_clusters=2,
        input_len=4,
        pred_len=3,
        model_cfg={"predictor": "mlp", "hidden_dim": 5, "dropout": 0.0},
    )
    adapted = build_cluster_predictor(
        num_clusters=2,
        input_len=4,
        pred_len=3,
        model_cfg={
            "predictor": "mlp",
            "hidden_dim": 5,
            "dropout": 0.0,
            "channel_adapter": {
                "enable": True,
                "rank": 2,
                "scale": 0.5,
                "init": "zero_delta",
            },
        },
        num_channels=3,
        cluster_id_c=cluster_id,
    )
    for param in base.parameters():
        param.data.uniform_(-0.1, 0.1)
    for param in adapted.parameters():
        param.data.zero_()
    adapted.reset_parameters(init="zero_delta")
    for k in range(2):
        adapted.load_cluster_state(k, base.get_cluster_state(k))

    x = torch.randn(2, 3, 4)

    assert torch.allclose(adapted(x, cluster_id), base(x, cluster_id))


def test_channel_adapter_freeze_base_excludes_base_params():
    cluster_id = torch.tensor([0, 0, 1])
    model = build_cluster_predictor(
        num_clusters=2,
        input_len=4,
        pred_len=3,
        model_cfg={
            "predictor": "mlp",
            "hidden_dim": 5,
            "dropout": 0.0,
            "channel_adapter": {
                "enable": True,
                "rank": 2,
                "scale": 0.5,
                "init": "zero_delta",
                "freeze_base": True,
            },
        },
        num_channels=3,
        cluster_id_c=cluster_id,
    )

    assert all(not p.requires_grad for p in model.base.parameters())
    assert any(p.requires_grad for p in model.get_cluster_params(0))


def test_horizon_bias_adapter_zero_init_preserves_base_output():
    cluster_id = torch.tensor([0, 0, 1])
    model = build_cluster_predictor(
        num_clusters=2,
        input_len=4,
        pred_len=3,
        model_cfg={
            "predictor": "channel_head_mlp",
            "hidden_dim": 5,
            "dropout": 0.0,
            "channel_head_residual": False,
            "horizon_bias_adapter": {
                "enable": True,
                "init_bias": 0.0,
            },
        },
        num_channels=3,
        cluster_id_c=cluster_id,
    )
    for param in model.base.parameters():
        param.data.fill_(0.25)

    x = torch.randn(2, 3, 4)
    y = model(x, cluster_id)
    expected = model.base(x, cluster_id)

    assert torch.allclose(y, expected)


def test_horizon_bias_adapter_updates_only_matching_cluster_channels():
    cluster_id = torch.tensor([0, 0, 1])
    model = build_cluster_predictor(
        num_clusters=2,
        input_len=4,
        pred_len=3,
        model_cfg={
            "predictor": "channel_head_mlp",
            "hidden_dim": 5,
            "dropout": 0.0,
            "channel_head_residual": False,
            "horizon_bias_adapter": {
                "enable": True,
                "init_bias": 0.0,
            },
        },
        num_channels=3,
        cluster_id_c=cluster_id,
    )
    for param in model.parameters():
        param.data.zero_()
    model.horizon_bias[0].data[1] = torch.tensor([0.1, 0.2, 0.3])
    model.horizon_bias[1].data[0] = torch.tensor([-0.5, -0.6, -0.7])

    x = torch.randn(2, 3, 4)
    y = model(x, cluster_id)

    assert torch.allclose(y[:, 0, :], torch.zeros_like(y[:, 0, :]))
    assert torch.allclose(y[:, 1, :], torch.tensor([0.1, 0.2, 0.3]).expand_as(y[:, 1, :]))
    assert torch.allclose(y[:, 2, :], torch.tensor([-0.5, -0.6, -0.7]).expand_as(y[:, 2, :]))


def test_horizon_bias_adapter_loads_legacy_base_cluster_state():
    cluster_id = torch.tensor([0, 0, 1])
    base = build_cluster_predictor(
        num_clusters=2,
        input_len=4,
        pred_len=3,
        model_cfg={
            "predictor": "channel_head_mlp",
            "hidden_dim": 5,
            "dropout": 0.0,
            "channel_head_residual": False,
            "channel_adapter": {
                "enable": True,
                "rank": 1,
                "scale": 0.5,
                "init": "zero_delta",
            },
        },
        num_channels=3,
        cluster_id_c=cluster_id,
    )
    adapted = build_cluster_predictor(
        num_clusters=2,
        input_len=4,
        pred_len=3,
        model_cfg={
            "predictor": "channel_head_mlp",
            "hidden_dim": 5,
            "dropout": 0.0,
            "channel_head_residual": False,
            "channel_adapter": {
                "enable": True,
                "rank": 1,
                "scale": 0.5,
                "init": "zero_delta",
            },
            "horizon_bias_adapter": {
                "enable": True,
                "init_bias": 0.0,
            },
        },
        num_channels=3,
        cluster_id_c=cluster_id,
    )
    for param in base.parameters():
        param.data.uniform_(-0.1, 0.1)
    for param in adapted.parameters():
        param.data.zero_()
    for k in range(2):
        adapted.load_cluster_state(k, base.get_cluster_state(k))

    x = torch.randn(2, 3, 4)

    assert torch.allclose(adapted(x, cluster_id), base(x, cluster_id))
