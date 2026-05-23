from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.moe_gate import ClusterwiseMoEGate


def test_penalty_allowed_mask_blocks_disallowed_penalties() -> None:
    torch.manual_seed(17)
    gate = ClusterwiseMoEGate(num_clusters=2, feat_dim=5, num_penalties=3, hidden_dim=4, topk=2)
    gate.set_penalty_allowed_mask(
        torch.tensor(
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
    )
    feat = torch.randn(3, 2, 5)
    mask, probs, _, _ = gate(feat, straight_through=False)

    assert torch.all(probs[:, 0, 1] == 0)
    assert torch.all(mask[:, 0, 1] == 0)
    assert torch.all(probs[:, 1, 0] == 0)
    assert torch.all(probs[:, 1, 2] == 0)
    assert torch.all(mask[:, 1, 0] == 0)
    assert torch.all(mask[:, 1, 2] == 0)
    assert torch.all(mask[:, 1, 1] == 1)


if __name__ == "__main__":
    test_penalty_allowed_mask_blocks_disallowed_penalties()
