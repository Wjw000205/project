import pytest
import torch

from src.utils.knn_shape import KNNShapeConfig, NearestNeighbors, ShapeKNNHybrid


@pytest.mark.skipif(NearestNeighbors is None, reason="scikit-learn is required for fixed KNN")
def test_fixed_knn_can_use_forecast_phase_features_to_break_shape_ties():
    cfg = KNNShapeConfig.from_dict(
        {
            "enable": True,
            "mode": "fixed",
            "scope": "same_channel",
            "k": 1,
            "alpha": 1.0,
            "bank_stride": 1,
            "feature_mode": "hist",
            "template_mode": "future",
            "anchor_mode": "last",
            "time_feature_mode": "forecast_phase",
            "time_periods": [8],
            "time_feature_weight": 10.0,
        }
    )

    x_bank = torch.zeros(2, 1, 4)
    y_bank = torch.tensor([[[1.0, 1.0]], [[-1.0, -1.0]]])
    cluster_id_c = torch.tensor([0])
    hybrid = ShapeKNNHybrid.fit(
        x_bank_ncl=x_bank,
        y_bank_nch=y_bank,
        cluster_id_c=cluster_id_c,
        cfg=cfg,
        start_offsets_n=torch.tensor([0, 4]),
    )

    x_query = torch.zeros(2, 1, 4)
    base_query = torch.zeros(2, 1, 2)
    pred = hybrid.hybridize_batch(
        x_query,
        base_query,
        cluster_id_c,
        query_start_abs_b=torch.tensor([0, 4]),
    )

    assert torch.allclose(pred[0, 0], torch.tensor([1.0, 1.0]), atol=1.0e-5)
    assert torch.allclose(pred[1, 0], torch.tensor([-1.0, -1.0]), atol=1.0e-5)


@pytest.mark.skipif(NearestNeighbors is None, reason="scikit-learn is required for fixed KNN")
def test_history_anchor_can_correct_from_observed_values_before_forecast_start():
    cfg = KNNShapeConfig.from_dict(
        {
            "enable": True,
            "mode": "fixed",
            "scope": "same_channel",
            "k": 1,
            "alpha": 0.0,
            "bank_stride": 1,
            "feature_mode": "hist",
            "template_mode": "future",
            "history_anchor": {
                "enable": True,
                "lags": [2],
                "alpha": 1.0,
                "blend_target": "prediction",
            },
        }
    )

    x_bank = torch.zeros(1, 1, 2)
    y_bank = torch.zeros(1, 1, 2)
    observed = torch.arange(10, dtype=torch.float32).view(10, 1)
    cluster_id_c = torch.tensor([0])
    hybrid = ShapeKNNHybrid.fit(
        x_bank_ncl=x_bank,
        y_bank_nch=y_bank,
        cluster_id_c=cluster_id_c,
        cfg=cfg,
        start_offsets_n=torch.tensor([0]),
        observed_history_tc=observed,
    )

    pred = hybrid.hybridize_batch(
        torch.zeros(1, 1, 2),
        torch.zeros(1, 1, 2),
        cluster_id_c,
        query_start_abs_b=torch.tensor([4]),
    )

    assert torch.allclose(pred, torch.tensor([[[4.0, 5.0]]]), atol=1.0e-5)
