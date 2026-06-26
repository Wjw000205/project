import torch

from scripts.next11d_binary_adoption_refit import (
    _binary_examples_for_penalty,
    _classify_binary_adoption_summary,
    _cluster_route_predictions_to_forecast,
    _forecast_metrics_from_route_predictions,
    _route_from_binary_scores,
)


def test_binary_examples_use_skip_as_negative_and_exclude_other_penalties() -> None:
    features = torch.arange(5 * 3 * 2, dtype=torch.float32).reshape(5, 3, 2)
    labels = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)

    x, y, indices = _binary_examples_for_penalty(features, labels, penalty_class=1)

    assert indices.tolist() == [0, 1, 3, 4]
    assert y.tolist() == [0.0, 1.0, 0.0, 1.0]
    assert torch.equal(x, features[indices, 1, :])


def test_route_from_binary_scores_falls_back_to_skip_when_no_penalty_passes() -> None:
    scores = torch.tensor(
        [
            [0.20, 0.10],
            [0.80, 0.70],
            [0.40, 0.95],
        ],
        dtype=torch.float32,
    )
    thresholds = torch.tensor([0.50, 0.90], dtype=torch.float32)

    pred = _route_from_binary_scores(scores, thresholds)

    assert pred.tolist() == [0, 1, 2]


def test_route_from_binary_scores_picks_largest_margin_when_multiple_pass() -> None:
    scores = torch.tensor([[0.82, 0.91]], dtype=torch.float32)
    thresholds = torch.tensor([0.70, 0.85], dtype=torch.float32)

    pred = _route_from_binary_scores(scores, thresholds)

    assert pred.tolist() == [1]


def test_binary_adoption_verdict_flags_zero_skip_adoption_first() -> None:
    summary = {
        "splits": {
            "train": {
                "accuracy_all": 0.82,
                "majority_accuracy_all": 0.70,
                "oracle_skip_rate": 0.35,
                "head_skip_rate": 0.0,
            },
            "train_holdout": {"accuracy_all": 0.75, "majority_accuracy_all": 0.65},
            "val": {"accuracy_all": 0.72, "majority_accuracy_all": 0.68},
        }
    }

    verdict = _classify_binary_adoption_summary(summary, min_train_accuracy=0.70)

    assert verdict["failure_layer"] == "skip/no-op behavior"
    assert verdict["decision"] == "binary_adoption_skip_not_adopted"


def test_cluster_route_predictions_to_forecast_uses_skip_base_and_cluster_penalty() -> None:
    base = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]],
        ]
    )
    cand = torch.stack([base + 10.0, base + 20.0], dim=2)
    cluster_id = torch.tensor([0, 1, 0], dtype=torch.long)
    route = torch.tensor([[0, 2], [1, 0]], dtype=torch.long)

    selected = _cluster_route_predictions_to_forecast(
        base_bch=base,
        cand_bcpH=cand,
        cluster_id_c=cluster_id,
        route_pred_bk=route,
    )

    expected = torch.stack(
        [
            torch.stack([base[0, 0], cand[0, 1, 1], base[0, 2]], dim=0),
            torch.stack([cand[1, 0, 0], base[1, 1], cand[1, 2, 0]], dim=0),
        ],
        dim=0,
    )
    assert torch.equal(selected, expected)


def test_forecast_metrics_from_route_predictions_reports_gain_vs_base() -> None:
    y = torch.zeros(1, 2, 2)
    base = torch.ones(1, 2, 2)
    cand = torch.stack([torch.zeros_like(base), torch.full_like(base, 3.0)], dim=2)
    cluster_id = torch.tensor([0, 1], dtype=torch.long)
    route = torch.tensor([[1, 0]], dtype=torch.long)

    metrics = _forecast_metrics_from_route_predictions(
        base_bch=base,
        cand_bcpH=cand,
        y_bch=y,
        cluster_id_c=cluster_id,
        route_pred_bk=route,
    )

    assert metrics["base_mse"] == 1.0
    assert metrics["selected_mse"] == 0.5
    assert metrics["selected_mae"] == 0.5
    assert metrics["selected_gain_pct_vs_base"] == 50.0
