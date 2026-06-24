import torch

from scripts.next11d_split_stable_adoption_guard import (
    _apply_cluster_penalty_guard,
    _apply_cluster_penalty_score_guard,
    _select_score_threshold_guard,
    _stable_cluster_penalty_mask,
    _summarize_cluster_penalty_stats,
)


def test_stable_cluster_penalty_mask_requires_fit_and_holdout_precision() -> None:
    label_names = ["skip", "trend", "direction"]
    train_fit = {
        "labels": torch.tensor([1, 2, 1, 0, 0, 2, 1, 0], dtype=torch.long),
        "current_pred": torch.tensor([1, 2, 1, 0, 1, 2, 1, 0], dtype=torch.long),
    }
    train_holdout = {
        "labels": torch.tensor([1, 2, 1, 0, 0, 2, 0, 0], dtype=torch.long),
        "current_pred": torch.tensor([1, 2, 1, 0, 1, 2, 1, 0], dtype=torch.long),
    }

    fit_stats = _summarize_cluster_penalty_stats(
        labels=train_fit["labels"],
        pred=train_fit["current_pred"],
        cluster_count=2,
        label_names=label_names,
    )
    holdout_stats = _summarize_cluster_penalty_stats(
        labels=train_holdout["labels"],
        pred=train_holdout["current_pred"],
        cluster_count=2,
        label_names=label_names,
    )

    mask, rows = _stable_cluster_penalty_mask(
        fit_stats=fit_stats,
        holdout_stats=holdout_stats,
        label_names=label_names,
        min_support=2,
        min_exact_precision=0.50,
        min_exact_recall=0.01,
        max_precision_gap=0.25,
    )

    assert mask.tolist() == [[True, False], [False, True]]
    by_key = {(int(row["cluster"]), row["penalty"]): row for row in rows}
    assert by_key[(0, "trend")]["allowed"] is True
    assert by_key[(1, "direction")]["allowed"] is True
    assert by_key[(1, "trend")]["allowed"] is False


def test_apply_cluster_penalty_guard_routes_disallowed_predictions_to_skip() -> None:
    pred = torch.tensor([1, 2, 1, 2, 1, 2], dtype=torch.long)
    allowed_kp = torch.tensor([[True, False], [False, True]], dtype=torch.bool)

    guarded = _apply_cluster_penalty_guard(pred, cluster_count=2, allowed_kp=allowed_kp)

    assert guarded.tolist() == [1, 2, 1, 2, 1, 2]

    guarded_none = _apply_cluster_penalty_guard(
        pred,
        cluster_count=2,
        allowed_kp=torch.zeros(2, 2, dtype=torch.bool),
    )

    assert guarded_none.tolist() == [0, 0, 0, 0, 0, 0]


def test_apply_cluster_penalty_guard_keeps_existing_skip() -> None:
    pred = torch.tensor([0, 2, 1, 0], dtype=torch.long)
    allowed_kp = torch.tensor([[True, False], [False, True]], dtype=torch.bool)

    guarded = _apply_cluster_penalty_guard(pred, cluster_count=2, allowed_kp=allowed_kp)

    assert guarded.tolist() == [0, 2, 1, 0]


def test_score_threshold_guard_selects_lowest_train_threshold_that_is_holdout_stable() -> None:
    label_names = ["skip", "trend"]
    feature_names = ["gate_prob"]
    train_fit = {
        "labels": torch.tensor([0, 0, 1, 1, 1], dtype=torch.long),
        "current_pred": torch.tensor([1, 1, 1, 1, 1], dtype=torch.long),
        "features": torch.tensor(
            [
                [[0.0], [0.10]],
                [[0.0], [0.20]],
                [[0.0], [0.70]],
                [[0.0], [0.80]],
                [[0.0], [0.90]],
            ],
            dtype=torch.float32,
        ),
    }
    train_holdout = {
        "labels": torch.tensor([0, 1, 1, 1], dtype=torch.long),
        "current_pred": torch.tensor([1, 1, 1, 1], dtype=torch.long),
        "features": torch.tensor(
            [
                [[0.0], [0.30]],
                [[0.0], [0.75]],
                [[0.0], [0.85]],
                [[0.0], [0.95]],
            ],
            dtype=torch.float32,
        ),
    }

    allowed, thresholds, rows = _select_score_threshold_guard(
        train_fit=train_fit,
        train_holdout=train_holdout,
        cluster_count=1,
        label_names=label_names,
        feature_names=feature_names,
        score_feature="gate_prob",
        quantiles=[0.0, 0.5],
        min_support=2,
        min_exact_precision=0.75,
        min_exact_recall=0.01,
        max_precision_gap=0.25,
    )

    assert allowed.tolist() == [[True]]
    assert thresholds.tolist() == [[0.699999988079071]]
    assert rows[0]["allowed"] is True
    assert rows[0]["selected_quantile"] == 0.5


def test_apply_cluster_penalty_score_guard_requires_allowed_score_threshold() -> None:
    pred = torch.tensor([1, 1, 1], dtype=torch.long)
    features = torch.tensor(
        [
            [[0.0], [0.40]],
            [[0.0], [0.70]],
            [[0.0], [0.90]],
        ],
        dtype=torch.float32,
    )
    allowed_kp = torch.tensor([[True]], dtype=torch.bool)
    thresholds = torch.tensor([[0.70]], dtype=torch.float32)

    guarded = _apply_cluster_penalty_score_guard(
        pred,
        features=features,
        cluster_count=1,
        allowed_kp=allowed_kp,
        score_threshold_kp=thresholds,
        score_feature_idx=0,
    )

    assert guarded.tolist() == [0, 1, 1]
