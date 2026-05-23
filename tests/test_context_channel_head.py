from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.cluster_predictor import build_cluster_predictor


def test_context_channel_head_uses_cluster_context() -> None:
    torch.manual_seed(7)
    cluster_id_c = torch.tensor([0, 0, 1], dtype=torch.long)
    model = build_cluster_predictor(
        num_clusters=2,
        input_len=8,
        pred_len=4,
        model_cfg={
            "predictor": "context_channel_head_mlp",
            "hidden_dim": 6,
            "dropout": 0.0,
        },
        num_channels=3,
        cluster_id_c=cluster_id_c,
    )
    model.eval()

    x = torch.randn(2, 3, 8)
    y = model(x, cluster_id_c)
    assert y.shape == (2, 3, 4)

    x_changed = x.clone()
    x_changed[:, 1, :] += torch.linspace(-1.0, 1.0, steps=8)
    y_changed = model(x_changed, cluster_id_c)

    # Channel 0's own input is unchanged. Its forecast should still react because
    # channel 1 belongs to the same cluster and changes the cluster context.
    assert torch.max(torch.abs(y_changed[:, 0, :] - y[:, 0, :])).item() > 1.0e-6


def test_lstm_revin_anchors_predictions_to_input_level() -> None:
    torch.manual_seed(11)
    cluster_id_c = torch.tensor([0, 1, 1], dtype=torch.long)
    model = build_cluster_predictor(
        num_clusters=2,
        input_len=8,
        pred_len=5,
        model_cfg={
            "predictor": "lstm_revin",
            "hidden_dim": 7,
            "dropout": 0.0,
            "lstm_num_layers": 1,
        },
        num_channels=3,
        cluster_id_c=cluster_id_c,
    )
    model.eval()

    x = torch.randn(4, 3, 8)
    y = model(x, cluster_id_c)
    y_shifted = model(x + 10.0, cluster_id_c)

    assert y.shape == (4, 3, 5)
    assert torch.allclose(y_shifted - y, torch.full_like(y, 10.0), atol=1.0e-4, rtol=1.0e-4)
    state = model.get_cluster_state(0)
    model.load_cluster_state(0, state)


def test_channel_lstm_mixer_builds_and_anchors_to_input_level() -> None:
    torch.manual_seed(13)
    cluster_id_c = torch.tensor([0, 0, 1], dtype=torch.long)
    model = build_cluster_predictor(
        num_clusters=2,
        input_len=8,
        pred_len=5,
        model_cfg={
            "predictor": "channel_lstm_mixer",
            "hidden_dim": 7,
            "dropout": 0.0,
            "lstm_num_layers": 1,
            "backbone_mix_init": -2.0,
        },
        num_channels=3,
        cluster_id_c=cluster_id_c,
    )
    model.eval()

    x = torch.randn(4, 3, 8)
    y = model(x, cluster_id_c)
    y_shifted = model(x + 10.0, cluster_id_c)

    assert y.shape == (4, 3, 5)
    assert torch.allclose(y_shifted - y, torch.full_like(y, 10.0), atol=1.0e-4, rtol=1.0e-4)
    state = model.get_cluster_state(1)
    model.load_cluster_state(1, state)


def test_channel_lstm_mixer_hard_routes_configured_channels() -> None:
    torch.manual_seed(17)
    cluster_id_c = torch.tensor([0, 0, 1], dtype=torch.long)
    model = build_cluster_predictor(
        num_clusters=2,
        input_len=8,
        pred_len=5,
        model_cfg={
            "predictor": "channel_lstm_mixer",
            "hidden_dim": 7,
            "dropout": 0.0,
            "lstm_num_layers": 1,
            "backbone_hard_route": True,
            "backbone_lstm_channel_indices": [1],
        },
        num_channels=3,
        cluster_id_c=cluster_id_c,
    )
    model.eval()

    x = torch.randn(4, 3, 8)
    y = model(x, cluster_id_c)
    y_channel = model.channel_head(x, cluster_id_c)
    y_lstm = model.lstm_revin(x, cluster_id_c)

    assert torch.allclose(y[:, 0, :], y_channel[:, 0, :])
    assert torch.allclose(y[:, 1, :], y_lstm[:, 1, :])
    assert torch.allclose(y[:, 2, :], y_channel[:, 2, :])


def test_seasonal_anchor_wraps_full_forecast_without_double_counting_level() -> None:
    torch.manual_seed(19)
    cluster_id_c = torch.tensor([0, 0, 1], dtype=torch.long)
    model = build_cluster_predictor(
        num_clusters=2,
        input_len=8,
        pred_len=5,
        model_cfg={
            "predictor": "channel_head_mlp",
            "hidden_dim": 7,
            "dropout": 0.0,
            "seasonal_anchor": True,
            "seasonal_anchor_period": 4,
        },
        num_channels=3,
        cluster_id_c=cluster_id_c,
    )
    model.eval()

    x = torch.randn(2, 3, 8)
    y = model(x, cluster_id_c)
    y_shifted = model(x + 10.0, cluster_id_c)

    assert y.shape == (2, 3, 5)
    assert torch.allclose(y_shifted - y, torch.full_like(y, 10.0), atol=1.0e-4, rtol=1.0e-4)


def test_predictor_input_len_uses_only_recent_tail_for_base_head() -> None:
    torch.manual_seed(23)
    cluster_id_c = torch.tensor([0, 0, 1], dtype=torch.long)
    model = build_cluster_predictor(
        num_clusters=2,
        input_len=8,
        pred_len=5,
        model_cfg={
            "predictor": "channel_head_mlp",
            "hidden_dim": 7,
            "dropout": 0.0,
            "predictor_input_len": 4,
        },
        num_channels=3,
        cluster_id_c=cluster_id_c,
    )
    model.eval()

    x = torch.randn(2, 3, 8)
    y = model(x, cluster_id_c)

    x_old_changed = x.clone()
    x_old_changed[..., :4] += 100.0
    y_old_changed = model(x_old_changed, cluster_id_c)

    x_recent_changed = x.clone()
    x_recent_changed[..., 4:] += 0.5
    y_recent_changed = model(x_recent_changed, cluster_id_c)

    assert y.shape == (2, 3, 5)
    assert torch.allclose(y_old_changed, y, atol=1.0e-6, rtol=1.0e-6)
    assert torch.max(torch.abs(y_recent_changed - y)).item() > 1.0e-6


def test_long_context_channel_head_uses_old_history_summary() -> None:
    torch.manual_seed(29)
    cluster_id_c = torch.tensor([0, 0, 1], dtype=torch.long)
    model = build_cluster_predictor(
        num_clusters=2,
        input_len=8,
        pred_len=5,
        model_cfg={
            "predictor": "long_context_channel_head_mlp",
            "hidden_dim": 7,
            "dropout": 0.0,
            "predictor_input_len": 4,
        },
        num_channels=3,
        cluster_id_c=cluster_id_c,
    )
    model.eval()

    x = torch.randn(2, 3, 8)
    y = model(x, cluster_id_c)

    x_old_changed = x.clone()
    x_old_changed[..., :4] += torch.linspace(-2.0, 2.0, steps=4)
    y_old_changed = model(x_old_changed, cluster_id_c)

    x_recent_changed = x.clone()
    x_recent_changed[..., 4:] += 0.5
    y_recent_changed = model(x_recent_changed, cluster_id_c)

    assert y.shape == (2, 3, 5)
    assert torch.max(torch.abs(y_old_changed - y)).item() > 1.0e-6
    assert torch.max(torch.abs(y_recent_changed - y)).item() > 1.0e-6


def test_long_context_channel_head_context_features_are_bounded_for_flat_segments() -> None:
    torch.manual_seed(31)
    cluster_id_c = torch.tensor([0, 1], dtype=torch.long)
    model = build_cluster_predictor(
        num_clusters=2,
        input_len=192,
        pred_len=12,
        model_cfg={
            "predictor": "long_context_channel_head_mlp",
            "hidden_dim": 7,
            "dropout": 0.0,
            "predictor_input_len": 96,
        },
        num_channels=2,
        cluster_id_c=cluster_id_c,
    )

    x = torch.zeros(2, 2, 192)
    x[..., -96:] = torch.linspace(-1.0, 1.0, steps=96)
    feat = model._context_features(x)
    context = feat[..., -model.context_dim :]

    assert torch.isfinite(feat).all()
    assert context.abs().max().item() <= 8.0


def test_long_context_channel_head_can_include_previous_cycle_profile() -> None:
    torch.manual_seed(37)
    cluster_id_c = torch.tensor([0, 1], dtype=torch.long)
    model = build_cluster_predictor(
        num_clusters=2,
        input_len=12,
        pred_len=5,
        model_cfg={
            "predictor": "long_context_channel_head_mlp",
            "hidden_dim": 7,
            "dropout": 0.0,
            "predictor_input_len": 4,
            "long_context_include_seasonal_profile": True,
        },
        num_channels=2,
        cluster_id_c=cluster_id_c,
    )
    model.eval()

    x = torch.randn(2, 2, 12)
    feat = model._context_features(x)
    y = model(x, cluster_id_c)

    assert feat.shape[-1] == 4 + 4 + model.context_dim
    x_prev_changed = x.clone()
    x_prev_changed[..., -8:-4] += 1.0
    y_prev_changed = model(x_prev_changed, cluster_id_c)
    assert y.shape == (2, 2, 5)
    assert torch.max(torch.abs(y_prev_changed - y)).item() > 1.0e-6


def test_long_context_anchor_channel_head_uses_profile_with_shift_invariance() -> None:
    torch.manual_seed(41)
    cluster_id_c = torch.tensor([0, 0, 1], dtype=torch.long)
    model = build_cluster_predictor(
        num_clusters=2,
        input_len=12,
        pred_len=7,
        model_cfg={
            "predictor": "long_context_anchor_channel_head_mlp",
            "hidden_dim": 7,
            "dropout": 0.0,
            "predictor_input_len": 4,
            "long_context_include_seasonal_profile": True,
            "anchor_chunk_len": 4,
            "anchor_detail_scale": 0.25,
        },
        num_channels=3,
        cluster_id_c=cluster_id_c,
    )
    model.eval()

    x = torch.randn(2, 3, 12)
    y = model(x, cluster_id_c)
    y_shifted = model(x + 10.0, cluster_id_c)

    x_prev_changed = x.clone()
    x_prev_changed[..., -8:-4] += 1.0
    y_prev_changed = model(x_prev_changed, cluster_id_c)

    assert y.shape == (2, 3, 7)
    assert torch.allclose(y_shifted - y, torch.full_like(y, 10.0), atol=1.0e-4, rtol=1.0e-4)
    assert torch.max(torch.abs(y_prev_changed - y)).item() > 1.0e-6


def test_seasonality_gated_channel_head_can_fall_back_to_either_branch() -> None:
    torch.manual_seed(43)
    cluster_id_c = torch.tensor([0, 0, 1], dtype=torch.long)
    model = build_cluster_predictor(
        num_clusters=2,
        input_len=12,
        pred_len=7,
        model_cfg={
            "predictor": "seasonality_gated_channel_head_mlp",
            "hidden_dim": 7,
            "dropout": 0.0,
            "predictor_input_len": 4,
            "long_context_include_seasonal_profile": True,
            "anchor_chunk_len": 4,
            "anchor_detail_scale": 0.25,
            "seasonal_mix_init": -20.0,
            "seasonal_gate_strength": 0.0,
        },
        num_channels=3,
        cluster_id_c=cluster_id_c,
    )
    model.eval()

    x = torch.randn(2, 3, 12)
    y_full = model.full_head(x, cluster_id_c)
    y = model(x, cluster_id_c)
    assert torch.allclose(y, y_full, atol=1.0e-5, rtol=1.0e-5)

    model.mix_logit_c.data.fill_(20.0)
    y_seasonal = model.seasonal_head(x, cluster_id_c)
    y = model(x, cluster_id_c)
    assert torch.allclose(y, y_seasonal, atol=1.0e-5, rtol=1.0e-5)

    y_shifted = model(x + 10.0, cluster_id_c)
    assert torch.allclose(y_shifted - y, torch.full_like(y, 10.0), atol=1.0e-4, rtol=1.0e-4)
    state = model.get_cluster_state(0)
    model.load_cluster_state(0, state)


def test_seasonality_gated_channel_head_saves_state_after_cuda_move() -> None:
    if not torch.cuda.is_available():
        return
    torch.manual_seed(47)
    cluster_id_c = torch.tensor([0, 0, 1], dtype=torch.long, device="cuda")
    model = build_cluster_predictor(
        num_clusters=2,
        input_len=12,
        pred_len=7,
        model_cfg={
            "predictor": "seasonality_gated_channel_head_mlp",
            "hidden_dim": 7,
            "dropout": 0.0,
            "predictor_input_len": 4,
            "anchor_chunk_len": 4,
        },
        num_channels=3,
        cluster_id_c=cluster_id_c,
    ).cuda()

    state = model.get_cluster_state(0)
    model.load_cluster_state(0, state)


if __name__ == "__main__":
    test_context_channel_head_uses_cluster_context()
    test_lstm_revin_anchors_predictions_to_input_level()
    test_channel_lstm_mixer_builds_and_anchors_to_input_level()
    test_channel_lstm_mixer_hard_routes_configured_channels()
    test_seasonal_anchor_wraps_full_forecast_without_double_counting_level()
    test_predictor_input_len_uses_only_recent_tail_for_base_head()
    test_long_context_channel_head_uses_old_history_summary()
    test_long_context_channel_head_context_features_are_bounded_for_flat_segments()
    test_long_context_channel_head_can_include_previous_cycle_profile()
    test_long_context_anchor_channel_head_uses_profile_with_shift_invariance()
    test_seasonality_gated_channel_head_can_fall_back_to_either_branch()
    test_seasonality_gated_channel_head_saves_state_after_cuda_move()
