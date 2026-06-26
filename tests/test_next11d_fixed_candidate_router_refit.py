from types import SimpleNamespace

from scripts.next11d_fixed_candidate_router_refit import (
    _classify_fixed_candidate_refit,
    _route_label_cfg_from_args,
    _route_head_cfg_from_args,
)


def test_fixed_candidate_refit_classifies_low_train_accuracy_as_feature_insufficiency() -> None:
    summary = {
        "splits": {
            "train": {"accuracy_all": 0.64, "majority_accuracy_all": 0.35},
            "train_holdout": {"accuracy_all": 0.57, "majority_accuracy_all": 0.42},
            "val": {"accuracy_all": 0.54, "majority_accuracy_all": 0.41},
        }
    }

    verdict = _classify_fixed_candidate_refit(summary, min_train_accuracy=0.70)

    assert verdict["train_accuracy_sanity_pass"] is False
    assert verdict["failure_layer"] == "gate feature insufficiency"


def test_fixed_candidate_refit_classifies_holdout_drop_as_train_val_shift() -> None:
    summary = {
        "splits": {
            "train": {"accuracy_all": 0.82, "majority_accuracy_all": 0.40},
            "train_holdout": {"accuracy_all": 0.41, "majority_accuracy_all": 0.45},
            "val": {"accuracy_all": 0.44, "majority_accuracy_all": 0.43},
        }
    }

    verdict = _classify_fixed_candidate_refit(summary, min_train_accuracy=0.70)

    assert verdict["train_accuracy_sanity_pass"] is True
    assert verdict["holdout_lift_positive"] is False
    assert verdict["failure_layer"] == "train-val utility shift"


def test_route_head_cfg_omits_zero_class_weight_max() -> None:
    cfg = _route_head_cfg_from_args(
        SimpleNamespace(
            epochs=10,
            head_batch_size=128,
            lr=0.003,
            weight_decay=0.0001,
            hidden_dim=0,
            dropout=0.0,
            head_mode="classwise",
            class_weight="balanced",
            class_weight_max=0.0,
            selection_split="train",
            selection_metric="accuracy",
            patience=5,
            min_delta=0.0,
            init_bias="none",
            seed=2026,
        )
    )

    assert cfg["class_weight"] == "balanced"
    assert cfg["selection_split"] == "train"
    assert "class_weight_max" not in cfg


def test_route_label_cfg_carries_action_floor() -> None:
    cfg = _route_label_cfg_from_args(
        SimpleNamespace(
            min_abs_improvement=0.0,
            min_rel_improvement=0.0,
            min_candidate_delta_rms=0.001,
        )
    )

    assert cfg == {
        "min_abs_improvement": 0.0,
        "min_rel_improvement": 0.0,
        "min_candidate_delta_rms": 0.001,
    }
