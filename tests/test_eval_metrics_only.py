from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import src.training.core as core_module
import src.training.evaluation as evaluation_module
from src.train import eval_loop


class _LastValueModel(nn.Module):
    def __init__(self, horizon: int):
        super().__init__()
        self.horizon = int(horizon)

    def forward(self, x_bcl: torch.Tensor, cluster_id_c: torch.Tensor) -> torch.Tensor:
        return x_bcl[..., -1:].expand(-1, -1, self.horizon)


def _run_eval(*, collect_samples: bool, base_metric_collector=None):
    x = torch.tensor(
        [
            [[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]],
            [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]],
            [[2.0, 3.0, 4.0], [4.0, 3.0, 2.0]],
        ]
    )
    y = torch.tensor(
        [
            [[2.0, 2.5], [0.0, -0.5]],
            [[3.0, 3.5], [1.0, 0.5]],
            [[4.0, 4.5], [2.0, 1.5]],
        ]
    )
    loader = DataLoader(
        TensorDataset(x, y, torch.arange(x.shape[0], dtype=torch.long)),
        batch_size=2,
        shuffle=False,
    )
    return eval_loop(
        model=_LastValueModel(horizon=2),
        gate=nn.Identity(),
        lambda_kp=torch.zeros(2, 0),
        penalty_names=[],
        penalty_fns={},
        loader=loader,
        cluster_id_c=torch.tensor([0, 1], dtype=torch.long),
        K=2,
        moe_cfg={"enable": False, "detach_penalty_grad": True},
        device=torch.device("cpu"),
        channel_count=2,
        collect_samples=collect_samples,
        base_metric_collector=base_metric_collector,
    )


def test_metrics_only_eval_preserves_aggregates_without_collecting_samples() -> None:
    full = _run_eval(collect_samples=True)
    metrics_only = _run_eval(collect_samples=False)

    for full_value, metrics_value in zip(full[:5], metrics_only[:5]):
        assert torch.equal(full_value, metrics_value)
    assert full[5] == metrics_only[5] == {}
    assert set(full[6]) == set(full[7]) == {0, 1}
    assert metrics_only[6] == {}
    assert metrics_only[7] == {}


def test_eval_reuses_gate_features_when_dynamic_lambda_is_disabled(monkeypatch) -> None:
    calls = 0
    extract_gate_features = core_module.extract_gate_features

    def counted_extract(x_bcl: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return extract_gate_features(x_bcl)

    def unexpected_series(*args, **kwargs):
        raise AssertionError("cluster series must not be built without dynamic lambda")

    monkeypatch.setattr(core_module, "extract_gate_features", counted_extract)
    monkeypatch.setattr(evaluation_module, "extract_gate_features", counted_extract)
    monkeypatch.setattr(evaluation_module, "scatter_mean_bcl_to_bkl", unexpected_series)

    _run_eval(collect_samples=False)

    assert calls == 2


def test_eval_collects_pre_moe_base_metrics_without_an_extra_loader_pass() -> None:
    collector = {}

    result = _run_eval(
        collect_samples=False,
        base_metric_collector=collector,
    )

    assert torch.equal(collector["avg_mse_k"], result[1])
    assert torch.equal(collector["avg_mae_k"], result[2])
    assert torch.equal(collector["mse_c"], result[3])
    assert torch.equal(collector["mae_c"], result[4])
    assert collector["num_prediction_elements_per_channel"] == 6
