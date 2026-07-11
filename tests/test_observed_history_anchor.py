import torch

from src.training.anchors import apply_history_anchor_adapter


def test_observed_history_anchor_uses_only_values_before_forecast_start() -> None:
    data = torch.arange(20, dtype=torch.float32).view(20, 1)
    starts = torch.tensor([10])

    base = torch.zeros(1, 1, 3)
    anchor = apply_history_anchor_adapter(
        base,
        base_pred_bch=base,
        observed_history_tc=data,
        query_start_abs_b=starts,
        input_len=4,
        cfg={
            "enable": True,
            "lags": [1, 4, 8],
            "alpha": 1.0,
            "blend_target": "prediction",
            "history_scope": "all_observed",
        },
    )

    expected = torch.tensor([[[29.0 / 3.0, 9.0, 10.0]]])
    assert torch.allclose(anchor, expected)
