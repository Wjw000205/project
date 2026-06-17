import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.cluster_mlp import ClusterwiseMLP  # noqa: E402
from src.models.cluster_predictor import ClusterwiseContextChannelHeadMLP  # noqa: E402
from src.models.gi_moe import ClusterMLPBaseWithFeatures  # noqa: E402


def _film_cfg():
    return {
        "enable": True,
        "dim": 4,
        "mode": "film",
        "film_scale": 0.1,
        "init_std": 0.02,
        "film_init": "zero",
    }


def test_cluster_embedding_off_matches_baseline_forward_bitwise():
    x = torch.randn(2, 3, 5)
    cluster_id = torch.tensor([0, 1, 0])

    torch.manual_seed(123)
    baseline = ClusterwiseMLP(2, 5, 4, 6, 0.0)
    torch.manual_seed(123)
    disabled = ClusterwiseMLP(
        2,
        5,
        4,
        6,
        0.0,
        cluster_embedding_cfg={"enable": False},
    )

    assert torch.equal(disabled(x, cluster_id), baseline(x, cluster_id))
    assert not hasattr(disabled, "cluster_embedding")


def test_cluster_embedding_zero_film_init_matches_baseline_forward():
    x = torch.randn(2, 3, 5)
    cluster_id = torch.tensor([0, 1, 0])

    torch.manual_seed(321)
    baseline = ClusterwiseMLP(2, 5, 4, 6, 0.0)
    torch.manual_seed(321)
    film = ClusterwiseMLP(2, 5, 4, 6, 0.0, cluster_embedding_cfg=_film_cfg())

    assert torch.equal(film(x, cluster_id), baseline(x, cluster_id))


def test_cluster_embedding_params_are_cluster_scoped_masked_and_stateful():
    model = ClusterwiseMLP(2, 5, 4, 6, 0.0, cluster_embedding_cfg=_film_cfg())
    cluster_id = torch.tensor([0, 1, 0])
    x = torch.randn(2, 3, 5)

    params_k1 = model.get_cluster_params(1)
    assert any(param is model.cluster_embedding[1] for param in params_k1)
    assert any(param is model.film_weight[1] for param in params_k1)
    assert any(param is model.film_bias[1] for param in params_k1)

    model(x, cluster_id).sum().backward()
    assert model.film_bias[1].grad is not None
    model.mask_cluster_grads(torch.tensor([False, True]))
    for param in [model.cluster_embedding[1], model.film_weight[1], model.film_bias[1]]:
        if param.grad is not None:
            assert torch.count_nonzero(param.grad).item() == 0

    state = model.get_cluster_state(1)
    assert {"cluster_embedding", "film_weight", "film_bias"}.issubset(state.keys())
    saved_embedding = state["cluster_embedding"].clone()
    saved_weight = state["film_weight"].clone()
    saved_bias = state["film_bias"].clone()
    model.cluster_embedding[1].data.add_(1.0)
    model.film_weight[1].data.add_(1.0)
    model.film_bias[1].data.add_(1.0)

    model.load_cluster_state(1, state)

    assert torch.equal(model.cluster_embedding[1].detach().cpu(), saved_embedding)
    assert torch.equal(model.film_weight[1].detach().cpu(), saved_weight)
    assert torch.equal(model.film_bias[1].detach().cpu(), saved_bias)


def test_gi_cluster_mlp_base_delegates_encode_decode_to_inner_mlp():
    model = ClusterMLPBaseWithFeatures(
        num_clusters=2,
        input_len=5,
        pred_len=4,
        hidden_dim=6,
        dropout=0.0,
        cluster_embedding_cfg=_film_cfg(),
    )
    cluster_id = torch.tensor([0, 1, 0])
    x = torch.randn(2, 3, 5)

    h = model.encode(x, cluster_id)
    y = model.decode(h, cluster_id)

    last = x[..., -1:]
    expected_h = model.inner.encode(x - last, cluster_id)
    expected_y = model.inner.decode(expected_h, cluster_id) + last

    assert torch.equal(h, expected_h)
    assert torch.equal(y, expected_y)


def test_context_channel_head_cluster_embedding_off_matches_baseline_bitwise():
    x = torch.randn(2, 3, 5)
    cluster_id = torch.tensor([0, 1, 0])

    torch.manual_seed(111)
    baseline = ClusterwiseContextChannelHeadMLP(
        num_clusters=2,
        input_len=5,
        pred_len=4,
        hidden_dim=6,
        num_channels=3,
        cluster_id_c=cluster_id,
        dropout=0.0,
    )
    torch.manual_seed(111)
    disabled = ClusterwiseContextChannelHeadMLP(
        num_clusters=2,
        input_len=5,
        pred_len=4,
        hidden_dim=6,
        num_channels=3,
        cluster_id_c=cluster_id,
        dropout=0.0,
        cluster_embedding_cfg={"enable": False},
    )

    assert torch.equal(disabled(x, cluster_id), baseline(x, cluster_id))
    assert not hasattr(disabled, "cluster_embedding")


def test_context_channel_head_zero_film_init_matches_baseline_forward():
    x = torch.randn(2, 3, 5)
    cluster_id = torch.tensor([0, 1, 0])

    torch.manual_seed(222)
    baseline = ClusterwiseContextChannelHeadMLP(
        num_clusters=2,
        input_len=5,
        pred_len=4,
        hidden_dim=6,
        num_channels=3,
        cluster_id_c=cluster_id,
        dropout=0.0,
    )
    torch.manual_seed(222)
    film = ClusterwiseContextChannelHeadMLP(
        num_clusters=2,
        input_len=5,
        pred_len=4,
        hidden_dim=6,
        num_channels=3,
        cluster_id_c=cluster_id,
        dropout=0.0,
        cluster_embedding_cfg=_film_cfg(),
    )

    assert torch.equal(film(x, cluster_id), baseline(x, cluster_id))


def test_context_channel_head_cluster_embedding_params_are_scoped_masked_and_stateful():
    cluster_id = torch.tensor([0, 1, 0])
    model = ClusterwiseContextChannelHeadMLP(
        num_clusters=2,
        input_len=5,
        pred_len=4,
        hidden_dim=6,
        num_channels=3,
        cluster_id_c=cluster_id,
        dropout=0.0,
        cluster_embedding_cfg=_film_cfg(),
    )
    x = torch.randn(2, 3, 5)

    params_k1 = model.get_cluster_params(1)
    assert any(param is model.cluster_embedding[1] for param in params_k1)
    assert any(param is model.film_weight[1] for param in params_k1)
    assert any(param is model.film_bias[1] for param in params_k1)

    model(x, cluster_id).sum().backward()
    assert model.film_bias[1].grad is not None
    model.mask_cluster_grads(torch.tensor([False, True]))
    for param in [model.cluster_embedding[1], model.film_weight[1], model.film_bias[1]]:
        if param.grad is not None:
            assert torch.count_nonzero(param.grad).item() == 0

    state = model.get_cluster_state(1)
    assert {"cluster_embedding", "film_weight", "film_bias"}.issubset(state.keys())
    saved_embedding = state["cluster_embedding"].clone()
    saved_weight = state["film_weight"].clone()
    saved_bias = state["film_bias"].clone()
    model.cluster_embedding[1].data.add_(1.0)
    model.film_weight[1].data.add_(1.0)
    model.film_bias[1].data.add_(1.0)

    model.load_cluster_state(1, state)

    assert torch.equal(model.cluster_embedding[1].detach().cpu(), saved_embedding)
    assert torch.equal(model.film_weight[1].detach().cpu(), saved_weight)
    assert torch.equal(model.film_bias[1].detach().cpu(), saved_bias)


def test_context_channel_head_zero_out_residual_blocks_match_baseline_forward():
    x = torch.randn(2, 3, 5)
    cluster_id = torch.tensor([0, 1, 0])

    torch.manual_seed(333)
    baseline = ClusterwiseContextChannelHeadMLP(
        num_clusters=2,
        input_len=5,
        pred_len=4,
        hidden_dim=6,
        num_channels=3,
        cluster_id_c=cluster_id,
        dropout=0.0,
    )
    torch.manual_seed(333)
    deep = ClusterwiseContextChannelHeadMLP(
        num_clusters=2,
        input_len=5,
        pred_len=4,
        hidden_dim=6,
        num_channels=3,
        cluster_id_c=cluster_id,
        dropout=0.0,
        residual_blocks=2,
        residual_block_scale=0.5,
        residual_block_init="zero_out",
    )

    assert torch.equal(deep(x, cluster_id), baseline(x, cluster_id))


def test_context_channel_head_residual_blocks_are_cluster_scoped_and_stateful():
    cluster_id = torch.tensor([0, 1, 0])
    model = ClusterwiseContextChannelHeadMLP(
        num_clusters=2,
        input_len=5,
        pred_len=4,
        hidden_dim=6,
        num_channels=3,
        cluster_id_c=cluster_id,
        dropout=0.0,
        residual_blocks=2,
        residual_block_scale=0.5,
        residual_block_init="zero_out",
    )
    x = torch.randn(2, 3, 5)

    params_k1 = model.get_cluster_params(1)
    offset = model._context_block_offset(1, 1)
    assert any(param is model.context_block_w1[offset] for param in params_k1)
    assert any(param is model.context_block_w2[offset] for param in params_k1)

    model(x, cluster_id).sum().backward()
    model.mask_cluster_grads(torch.tensor([False, True]))
    for param in [
        model.context_block_w1[offset],
        model.context_block_b1[offset],
        model.context_block_w2[offset],
        model.context_block_b2[offset],
    ]:
        if param.grad is not None:
            assert torch.count_nonzero(param.grad).item() == 0

    state = model.get_cluster_state(1)
    assert "context_residual_blocks" in state
    assert len(state["context_residual_blocks"]) == 2
    saved_w2 = state["context_residual_blocks"][1]["W2"].clone()
    model.context_block_w2[offset].data.add_(1.0)
    model.load_cluster_state(1, state)
    assert torch.equal(model.context_block_w2[offset].detach().cpu(), saved_w2)
