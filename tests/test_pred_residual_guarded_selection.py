from __future__ import annotations

import torch

from src.train import _pred_residual_channel_keep_mask


def test_pred_residual_mse_gate_keeps_only_mse_improvements() -> None:
    base_mse = torch.tensor([1.0, 1.0, 1.0])
    cand_mse = torch.tensor([0.9, 1.0, 1.1])
    base_mae = torch.tensor([1.0, 1.0, 1.0])
    cand_mae = torch.tensor([1.2, 0.8, 0.8])

    keep = _pred_residual_channel_keep_mask(
        "val_mse_gate_guarded",
        base_mse,
        cand_mse,
        base_mae,
        cand_mae,
        min_abs_improvement=0.0,
        min_rel_improvement=0.0,
    )

    assert keep.tolist() == [True, False, False]


def test_pred_residual_mae_guarded_respects_mse_regression_budget() -> None:
    base_mse = torch.tensor([1.0, 1.0, 1.0, 1.0])
    cand_mse = torch.tensor([1.003, 1.006, 0.99, 1.0])
    base_mae = torch.tensor([1.0, 1.0, 1.0, 1.0])
    cand_mae = torch.tensor([0.95, 0.95, 0.95, 1.0])

    keep = _pred_residual_channel_keep_mask(
        "val_mae_gate_guarded",
        base_mse,
        cand_mse,
        base_mae,
        cand_mae,
        min_abs_improvement=0.0,
        min_rel_improvement=0.0,
        max_rel_mse_regression=0.005,
    )

    assert keep.tolist() == [True, False, True, False]
