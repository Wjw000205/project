from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.penalties import build_penalty_compute, supported_penalty_names
from src.models.residual_moe import ClusterwisePredResidualMoE


def _zero_learned_residuals(model: ClusterwisePredResidualMoE) -> None:
    with torch.no_grad():
        for w in model.W1:
            w.zero_()
        for b in model.b1:
            b.zero_()
        for w in model.W2:
            w.zero_()
        for b in model.b2:
            b.zero_()
        for alpha in model.log_alpha:
            alpha.fill_(20.0)


def test_seasonal_align_adapter_adds_previous_cycle_residual() -> None:
    model = ClusterwisePredResidualMoE(
        num_clusters=1,
        num_penalties=1,
        input_len=8,
        pred_len=4,
        hidden_dim=2,
        init_alpha=20.0,
        alpha_scale=1.0,
        use_y_base_input=False,
        intervention_enable=False,
        penalty_names=["seasonal_align"],
        seasonal_anchor_names=["seasonal_align"],
        seasonal_anchor_period=4,
        seasonal_anchor_num_periods=1,
        seasonal_anchor_scale=1.0,
    )
    _zero_learned_residuals(model)

    x = torch.tensor([[[10.0, 11.0, 12.0, 13.0, 1.0, 2.0, 3.0, 4.0]]])
    y_base = torch.zeros(1, 1, 4)
    cluster_id = torch.tensor([0], dtype=torch.long)
    mask = torch.ones(1, 1, 1)

    out = model(x, y_base, cluster_id, mask)

    expected = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    assert torch.allclose(out["residuals"][:, :, 0, :], expected, atol=1.0e-5, rtol=1.0e-5)
    assert torch.allclose(out["y_final"], expected, atol=1.0e-4, rtol=1.0e-4)


def test_nonseasonal_adapter_does_not_get_previous_cycle_residual() -> None:
    model = ClusterwisePredResidualMoE(
        num_clusters=1,
        num_penalties=1,
        input_len=8,
        pred_len=4,
        hidden_dim=2,
        init_alpha=20.0,
        alpha_scale=1.0,
        use_y_base_input=False,
        intervention_enable=False,
        penalty_names=["level"],
        seasonal_anchor_names=["seasonal_align"],
        seasonal_anchor_period=4,
        seasonal_anchor_num_periods=1,
        seasonal_anchor_scale=1.0,
    )
    _zero_learned_residuals(model)

    x = torch.tensor([[[10.0, 11.0, 12.0, 13.0, 1.0, 2.0, 3.0, 4.0]]])
    y_base = torch.zeros(1, 1, 4)
    cluster_id = torch.tensor([0], dtype=torch.long)
    mask = torch.ones(1, 1, 1)

    out = model(x, y_base, cluster_id, mask)

    assert torch.allclose(out["residuals"], torch.zeros_like(out["residuals"]))
    assert torch.allclose(out["y_final"], y_base)


def test_seasonal_anchor_averages_available_previous_cycles() -> None:
    model = ClusterwisePredResidualMoE(
        num_clusters=1,
        num_penalties=1,
        input_len=12,
        pred_len=4,
        hidden_dim=2,
        use_y_base_input=False,
        penalty_names=["seasonal_align"],
        seasonal_anchor_names=["seasonal_align"],
        seasonal_anchor_period=4,
        seasonal_anchor_num_periods=2,
    )

    x = torch.tensor([[[10.0, 20.0, 30.0, 40.0, 2.0, 4.0, 6.0, 8.0, 4.0, 8.0, 12.0, 16.0]]])
    anchor = model._seasonal_anchor_forecast(x)

    expected = torch.tensor([[[3.0, 6.0, 9.0, 12.0]]])
    assert torch.allclose(anchor, expected)


def test_seasonal_align_penalty_is_registered_and_computable() -> None:
    assert "seasonal_align" in supported_penalty_names()
    compute = build_penalty_compute(["seasonal_align"], jump_thr=2.0)
    y = torch.tensor([[[1.0, 2.0, 1.0, 2.0], [0.0, 1.0, 2.0, 3.0]]])
    yhat = y.clone()
    pen = compute(yhat, y)

    assert pen.shape == (1, 2, 1)
    assert torch.allclose(pen, torch.zeros_like(pen))


if __name__ == "__main__":
    test_seasonal_align_adapter_adds_previous_cycle_residual()
    test_nonseasonal_adapter_does_not_get_previous_cycle_residual()
    test_seasonal_anchor_averages_available_previous_cycles()
    test_seasonal_align_penalty_is_registered_and_computable()
