import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train import (  # noqa: E402
    _apply_mae_objective_weight,
    _build_mae_per_cluster_diagnostics_from_targets,
    _mae_objective_weight_is_nonzero,
    _scale_mae_objective_weight,
)


def test_scalar_mae_objective_weight_keeps_existing_expression():
    mae_bk = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    weighted = _apply_mae_objective_weight(mae_bk, 0.4)

    assert torch.equal(weighted, 0.4 * mae_bk)
    assert _mae_objective_weight_is_nonzero(0.4)
    assert not _mae_objective_weight_is_nonzero(0.0)


def test_vector_mae_objective_weight_applies_per_cluster_before_reduce_and_has_no_grad():
    mae_bk = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    weight_k = torch.tensor([0.4, 0.5, 0.6])

    weighted = _apply_mae_objective_weight(mae_bk, weight_k)
    weighted.sum().backward()

    assert torch.equal(weighted, mae_bk.detach() * weight_k.view(1, -1))
    assert weight_k.requires_grad is False
    assert weight_k.grad is None
    assert _mae_objective_weight_is_nonzero(weight_k)


def test_per_cluster_mae_diagnostics_build_fixed_multiplier_from_train_targets():
    # Cluster 0 has zero gap, cluster 1 has a large mean/median gap.
    targets = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 0.0], [0.0, 100.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 100.0]],
        ]
    )
    cluster_id = torch.tensor([0, 1, 1])

    result = _build_mae_per_cluster_diagnostics_from_targets(
        targets_bch=targets,
        cluster_id_c=cluster_id,
        K=2,
        base_weight=0.4,
        cfg={
            "enable": True,
            "diagnostic": "mean_median_gap",
            "normalize": "std",
            "pivot": "median",
            "max_multiplier": 1.25,
            "min_multiplier": 1.0,
        },
    )

    multiplier = result["multiplier_k"]
    effective = result["effective_weight_k"]
    assert multiplier.shape == (2,)
    assert multiplier.requires_grad is False
    assert torch.equal(multiplier, torch.tensor([1.0, 1.25]))
    assert torch.equal(effective, torch.tensor([0.4, 0.5]))
    assert result["rows"][0]["channels"] == 1
    assert result["rows"][1]["channels"] == 2


def test_mae_weight_warmup_scalar_composes_with_fixed_cluster_multiplier():
    multiplier = torch.tensor([1.0, 1.25])

    scaled = _scale_mae_objective_weight(0.2, multiplier)

    assert torch.equal(scaled, torch.tensor([0.2, 0.25]))
    assert scaled.requires_grad is False
