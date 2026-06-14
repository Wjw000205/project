from __future__ import annotations

import inspect

import pandas as pd
import torch

import src.train as train_module
from src.train import (
    _calendar_features_from_datetime,
    _fit_calendar_residual_from_prediction_parts,
    apply_calendar_residual_correction,
    calendar_feature_batch,
    fit_calendar_residual_correction,
)


class _ZeroBackbone(torch.nn.Module):
    def __init__(self, pred_len: int) -> None:
        super().__init__()
        self.pred_len = int(pred_len)

    def forward(self, x: torch.Tensor, cluster_id_c: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], x.shape[1], self.pred_len, device=x.device, dtype=x.dtype)


def test_calendar_features_repeat_daily_time_of_day_phase() -> None:
    dates = pd.to_datetime(["2026-01-01 06:00:00", "2026-01-02 06:00:00"])

    features, names = _calendar_features_from_datetime(
        dates,
        {"include_bias": True, "time_of_day": True, "tod_harmonics": 1, "day_of_week": False},
    )

    assert names == ["bias", "tod_sin_1", "tod_cos_1"]
    assert torch.allclose(features[:, 0], torch.ones(2))
    assert torch.allclose(features[0, 1:], features[1, 1:], atol=1.0e-6)
    assert torch.allclose(features[0, 1:], torch.tensor([1.0, 0.0]), atol=1.0e-6)


def test_calendar_feature_batch_starts_at_forecast_timestamp() -> None:
    feature_tf = torch.arange(12, dtype=torch.float32).view(6, 2)

    batch = calendar_feature_batch(
        feature_tf,
        query_start_abs_b=torch.tensor([1]),
        input_len=2,
        pred_len=3,
    )

    assert torch.equal(batch, feature_tf[3:6].view(1, 3, 2))


def test_fit_calendar_residual_correction_uses_training_residuals() -> None:
    x = torch.zeros(3, 2, 2)
    y = torch.tensor([1.5, -0.5], dtype=torch.float32).view(1, 2, 1).expand(3, 2, 2).clone()
    idx = torch.arange(3)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y, idx), batch_size=2)
    calendar_feature_tf = torch.ones(8, 1)

    coef_cf, summary = fit_calendar_residual_correction(
        model=_ZeroBackbone(pred_len=2),
        loader=loader,
        cluster_id_c=torch.tensor([0, 0]),
        device=torch.device("cpu"),
        calendar_feature_tf=calendar_feature_tf,
        input_len=2,
        eval_start=0,
        cfg={"ridge": 0.0, "shrink": 1.0},
    )

    assert coef_cf is not None
    assert summary["source_split"] == "train"
    assert summary["fit_windows"] == 3
    assert torch.allclose(coef_cf, torch.tensor([[1.5], [-0.5]]), atol=1.0e-6)

    corrected = apply_calendar_residual_correction(
        torch.zeros(1, 2, 2),
        calendar_feature_tf,
        coef_cf,
        query_start_abs_b=torch.tensor([1]),
        input_len=2,
    )
    assert torch.allclose(corrected, torch.tensor([[[1.5, 1.5], [-0.5, -0.5]]]), atol=1.0e-6)


def test_fit_calendar_residual_from_prediction_parts_uses_final_predictions() -> None:
    y_true = torch.tensor(
        [
            [[3.0, 5.0], [1.0, 3.0]],
            [[4.0, 6.0], [2.0, 4.0]],
        ],
        dtype=torch.float32,
    )
    y_pred = y_true - torch.tensor([2.0, -1.0], dtype=torch.float32).view(1, 2, 1)
    calendar_feature_tf = torch.ones(8, 1)

    coef_cf, summary = _fit_calendar_residual_from_prediction_parts(
        idx_parts=[torch.tensor([0, 1])],
        y_true_parts=[y_true],
        y_pred_parts=[y_pred],
        calendar_feature_tf=calendar_feature_tf,
        input_len=2,
        cfg={"ridge": 0.0, "shrink": 1.0},
    )

    assert coef_cf is not None
    assert summary["fit_windows"] == 2
    assert torch.allclose(coef_cf, torch.tensor([[2.0], [-1.0]]), atol=1.0e-6)


def test_main_fits_and_records_calendar_residual_summary() -> None:
    source = inspect.getsource(train_module.main)

    assert "fit_calendar_residual_correction(" in source
    assert "fit_calendar_residual_correction_from_eval_path(" in source
    assert '"calendar_residual": calendar_residual_summary' in source
