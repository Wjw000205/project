import inspect
import json

import pytest
import torch

from scripts import shape_bucket_mask_eval as mask_eval


def _bucket_stats_payload() -> dict:
    return {
        "accepted": [
            {
                "q": 4,
                "cluster_id": 1,
                "feature": "history_d2_rms",
                "feature_index": 0,
                "bucket": 2,
                "penalty": "jump",
                "penalty_index": 0,
                "base_mse_proxy_flag": False,
                "splits": {
                    "train_fit": {"support_count": 200, "mean_gain": 0.02, "positive_rate": 0.70},
                    "train_holdout": {"support_count": 160, "mean_gain": 0.03, "positive_rate": 0.65},
                },
                "passes_thresholds": [
                    {"n_min": 128, "margin": 0.001, "positive_rate_holdout": 0.60}
                ],
            },
            {
                "q": 4,
                "cluster_id": 0,
                "feature": "history_d2_rms",
                "feature_index": 0,
                "bucket": 1,
                "penalty": "delta",
                "penalty_index": 3,
                "base_mse_proxy_flag": False,
                "splits": {
                    "train_fit": {"support_count": 180, "mean_gain": 0.01, "positive_rate": 0.66},
                    "train_holdout": {"support_count": 140, "mean_gain": 0.02, "positive_rate": 0.62},
                },
                "passes_thresholds": [
                    {"n_min": 128, "margin": 0.001, "positive_rate_holdout": 0.60}
                ],
            },
            {
                "q": 4,
                "cluster_id": 1,
                "feature": "std_ratio_base_to_history",
                "feature_index": 1,
                "bucket": 0,
                "penalty": "jump",
                "penalty_index": 0,
                "base_mse_proxy_flag": False,
                "splits": {
                    "train_fit": {"support_count": 200, "mean_gain": 0.20, "positive_rate": 0.70},
                    "train_holdout": {"support_count": 20, "mean_gain": 0.20, "positive_rate": 0.95},
                },
                "passes_thresholds": [],
            },
        ]
    }


def test_mask_fit_signature_has_no_val_or_test_label_inputs() -> None:
    sig = inspect.signature(mask_eval.fit_shape_bucket_mask)
    assert "val_labels" not in sig.parameters
    assert "test_labels" not in sig.parameters
    assert "y_true" not in sig.parameters


def test_train_only_shape_mask_is_serialized_and_loaded(tmp_path) -> None:
    edges = {
        "q4": {
            "q": 4,
            "feature_names": ["history_d2_rms", "std_ratio_base_to_history"],
            "edges": [[[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]], [[0.4, 0.5, 0.6], [4.0, 5.0, 6.0]]],
        }
    }
    allowed = torch.tensor([[True, False, False, True], [True, True, False, True]])
    spec = mask_eval.fit_shape_bucket_mask(
        bucket_stats=_bucket_stats_payload(),
        bucket_edges=edges,
        penalty_names=["jump", "amp_under", "level", "delta"],
        allowed_mask_kp=allowed,
        n_min=128,
        margin=0.001,
        positive_rate_holdout=0.60,
    )

    assert spec["q"] == 4
    assert spec["feature"] == "history_d2_rms"
    assert spec["feature_index"] == 0
    assert spec["source_splits"] == ["train_fit", "train_holdout"]
    assert spec["allow_kbp"][0][1] == [False, False, False, True]
    assert spec["allow_kbp"][1][2] == [True, False, False, False]

    path = tmp_path / "shape_mask.json"
    mask_eval.save_shape_mask(spec, path)
    loaded = mask_eval.load_shape_mask(path)
    assert loaded == json.loads(path.read_text())
    assert loaded["allow_kbp"] == spec["allow_kbp"]


def test_mask_fit_resolves_feature_index_from_bucket_edges_when_stats_omit_it() -> None:
    payload = _bucket_stats_payload()
    for row in payload["accepted"]:
        row.pop("feature_index", None)
        row.pop("penalty_index", None)
    edges = {
        "q4": {
            "q": 4,
            "feature_names": ["history_d2_rms", "std_ratio_base_to_history"],
            "edges": [[[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]], [[0.4, 0.5, 0.6], [4.0, 5.0, 6.0]]],
        }
    }
    allowed = torch.tensor([[True, False, False, True], [True, True, False, True]])

    spec = mask_eval.fit_shape_bucket_mask(
        bucket_stats=payload,
        bucket_edges=edges,
        penalty_names=["jump", "amp_under", "level", "delta"],
        allowed_mask_kp=allowed,
        n_min=128,
        margin=0.001,
        positive_rate_holdout=0.60,
    )

    assert spec["feature"] == "history_d2_rms"
    assert spec["feature_index"] == 0
    assert spec["allow_kbp"][0][1] == [False, False, False, True]


def test_shape_route_mask_intersects_cluster_prior_before_shape_mask() -> None:
    probs = torch.tensor([[[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]])
    bucket_ids = torch.tensor([[0, 0]])
    spec = {
        "q": 2,
        "feature": "shape",
        "feature_index": 0,
        "penalty_names": ["jump", "amp_under", "delta"],
        "cluster_allowed_mask_kp": [[True, False, True], [True, True, True]],
        "allow_kbp": [
            [[False, True, False], [True, False, False]],
            [[False, True, False], [False, False, True]],
        ],
    }

    route, stats = mask_eval.route_mask_from_shape_buckets(
        probs_bkp=probs,
        bucket_ids_bk=bucket_ids,
        mask_spec=spec,
        select_ranks=[1],
    )

    expected = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    torch.testing.assert_close(route, expected)
    assert stats["no_op_rate"] == 0.5
    assert stats["shape_allowed_rate"] == pytest.approx(1.0 / 6.0)


def test_empty_shape_mask_falls_back_to_no_op_base() -> None:
    probs = torch.tensor([[[0.7, 0.2, 0.1]]])
    bucket_ids = torch.tensor([[1]])
    spec = {
        "q": 2,
        "feature": "shape",
        "feature_index": 0,
        "penalty_names": ["jump", "amp_under", "delta"],
        "cluster_allowed_mask_kp": [[True, True, True]],
        "allow_kbp": [
            [[True, False, False], [False, False, False]],
        ],
    }

    route, stats = mask_eval.route_mask_from_shape_buckets(
        probs_bkp=probs,
        bucket_ids_bk=bucket_ids,
        mask_spec=spec,
        select_ranks=[1, 2],
    )

    assert torch.count_nonzero(route).item() == 0
    assert stats["no_op_rate"] == 1.0
