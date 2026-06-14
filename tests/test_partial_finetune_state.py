import torch
from torch import nn

from src.train import _partial_load_matching_state_dict


def test_partial_load_matching_state_dict_skips_mismatched_output_head() -> None:
    target = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 5))
    before_head_weight = target[1].weight.detach().clone()
    before_head_bias = target[1].bias.detach().clone()

    source_state = {
        "0.weight": torch.full_like(target[0].weight, 1.5),
        "0.bias": torch.full_like(target[0].bias, -0.5),
        "1.weight": torch.ones((4, 2)),
        "1.bias": torch.ones((4,)),
        "extra.weight": torch.ones((1,)),
    }

    summary = _partial_load_matching_state_dict(target, source_state)

    assert torch.allclose(target[0].weight, torch.full_like(target[0].weight, 1.5))
    assert torch.allclose(target[0].bias, torch.full_like(target[0].bias, -0.5))
    assert torch.allclose(target[1].weight, before_head_weight)
    assert torch.allclose(target[1].bias, before_head_bias)
    assert summary["loaded_count"] == 2
    assert summary["skipped_shape_count"] == 2
    assert summary["skipped_missing_count"] == 1
