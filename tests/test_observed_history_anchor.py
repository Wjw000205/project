import torch

from scripts.probe_observed_history_anchor import observed_history_anchor


def test_observed_history_anchor_uses_only_values_before_forecast_start() -> None:
    data = torch.arange(20, dtype=torch.float32).view(20, 1)
    starts = torch.tensor([10])

    anchor = observed_history_anchor(
        data_tc=data,
        starts_n=starts,
        input_len=4,
        pred_len=3,
        lags=(1, 4, 8),
    )

    expected = torch.tensor([[[29.0 / 3.0, 9.0, 10.0]]])
    assert torch.allclose(anchor, expected)
