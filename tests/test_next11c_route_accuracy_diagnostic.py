import pytest
import torch

from scripts.next11c_route_accuracy_diagnostic import (
    _build_arg_parser,
    _normalize_requested_splits,
    _route_label_thresholds_from_config,
    route_accuracy_summary,
    topk_set_overlap_summary,
)
from src.train import (
    _cluster_route_oracle_labels_and_gain_from_candidates,
    _cluster_route_oracle_labels_from_candidates,
    _pred_residual_training_skip_arg,
    _route_ce_active_mask_from_gain,
    _route_ce_class_weight_from_labels,
    _route_accuracy_summary_from_labels,
    _route_ce_loss_from_probs,
    _stage2_route_audit_thresholds,
)


def test_route_accuracy_summary_treats_skip_as_real_class() -> None:
    labels = torch.tensor([0, 1, 1, 2, 0, 2])
    current = torch.tensor([1, 1, 2, 2, 0, 0])

    summary = route_accuracy_summary(
        labels=labels,
        current_pred=current,
        label_names=["skip", "trend", "direction"],
    )

    assert summary["samples"] == 6
    assert summary["current_accuracy_all"] == pytest.approx(3 / 6)
    assert summary["oracle_skip_rate"] == pytest.approx(2 / 6)
    assert summary["actual_skip_rate"] == pytest.approx(2 / 6)
    assert summary["skip_recall"] == pytest.approx(1 / 2)
    assert summary["skip_false_positive_rate_on_oracle_penalty"] == pytest.approx(1 / 4)
    assert summary["penalty_accuracy_on_oracle_penalty"] == pytest.approx(2 / 4)
    assert summary["oracle_penalty_routed_to_wrong_penalty_rate"] == pytest.approx(1 / 4)
    assert summary["oracle_penalty_routed_to_skip_rate"] == pytest.approx(1 / 4)
    assert summary["confusion_matrix_counts"] == [
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1],
    ]
    assert summary["per_class"]["skip"]["recall"] == pytest.approx(1 / 2)
    assert summary["per_class"]["trend"]["precision"] == pytest.approx(1 / 2)


def test_normalize_requested_splits_preserves_order_and_aliases_train() -> None:
    assert _normalize_requested_splits(["val", "train", "train_holdout", "val"]) == [
        "val",
        "train_fit",
        "train_holdout",
    ]


def test_topk_set_overlap_flag_is_default_off_and_preserves_legacy_defaults() -> None:
    args = _build_arg_parser().parse_args([])

    assert args.topk_set_overlap is False
    assert args.cells == ["ETTm2_H96", "ETTh1_H96"]
    assert args.variants == ["d_moe_only_no_anchors", "c_full"]


def test_topk_set_overlap_summary_reports_cluster_channel_and_majority_baseline() -> None:
    gains = torch.tensor(
        [
            [[0.2, 0.1, -0.1], [-0.2, 0.3, 0.0], [-0.1, -0.2, 0.4]],
            [[-0.1, -0.2, 0.5], [0.0, -0.1, -0.2], [0.2, 0.3, 0.4]],
        ],
        dtype=torch.float32,
    )
    applied = torch.tensor(
        [
            [[1, 0, 1], [1, 0, 1], [0, 1, 1]],
            [[1, 0, 1], [1, 0, 1], [0, 1, 1]],
        ],
        dtype=torch.bool,
    )

    summary = topk_set_overlap_summary(
        gain_bcp=gains,
        applied_bcp=applied,
        cluster_id_c=torch.tensor([0, 0, 1]),
        penalty_names=["a", "b", "c"],
        channel_names=["c0", "c1", "c2"],
        tau=0.0,
    )

    assert summary["overall"]["precision"] == pytest.approx(5 / 12)
    assert summary["overall"]["recall"] == pytest.approx(5 / 8)
    assert summary["majority_overall"]["precision"] == pytest.approx(6 / 12)
    assert summary["majority_overall"]["recall"] == pytest.approx(6 / 8)
    assert summary["per_cluster"][0]["precision"] == pytest.approx(2 / 8)
    assert summary["per_cluster"][0]["recall"] == pytest.approx(2 / 4)
    assert summary["per_channel"][0]["precision"] == pytest.approx(2 / 4)
    assert summary["per_channel"][0]["recall"] == pytest.approx(2 / 3)


def test_route_label_thresholds_manual_source_uses_cli_values() -> None:
    thresholds = _route_label_thresholds_from_config(
        {},
        source="manual",
        min_abs_improvement=0.004,
        min_rel_improvement=0.02,
        min_candidate_delta_rms=0.005,
    )

    assert thresholds == {
        "min_abs_improvement": 0.004,
        "min_rel_improvement": 0.02,
        "min_candidate_delta_rms": 0.005,
        "source": "manual",
    }


def test_route_label_thresholds_stage2_source_matches_training_audit_precedence() -> None:
    cfg = {
        "moe": {
            "enable": True,
            "pred_side_residual": {"enable": True},
            "route_ce_supervision": {
                "enable": True,
                "weight": 1.0,
                "min_abs_improvement": 0.0,
                "min_candidate_delta_rms": 0.001,
            },
            "binary_adoption_supervision": {
                "enable": True,
                "weight": 1.0,
                "min_abs_improvement": 0.001,
                "min_candidate_delta_rms": 0.002,
            },
        },
        "diagnostics": {
            "stage2_route_audit": {
                "min_abs_improvement": 0.003,
                "min_rel_improvement": 0.01,
                "min_candidate_delta_rms": 0.004,
            }
        },
    }

    thresholds = _route_label_thresholds_from_config(
        cfg,
        source="stage2",
        min_abs_improvement=0.0,
        min_rel_improvement=0.0,
        min_candidate_delta_rms=0.0,
    )

    assert thresholds == {
        "min_abs_improvement": 0.003,
        "min_rel_improvement": 0.01,
        "min_candidate_delta_rms": 0.004,
        "source": "diagnostics.stage2_route_audit",
    }


def test_route_accuracy_summary_can_apply_action_floor_to_gain_labels() -> None:
    labels = torch.tensor([1, 1, 2, 0])
    current = torch.tensor([1, 0, 2, 1])
    gain = torch.tensor([0.0020, 0.0002, 0.0030, 0.0])

    summary = route_accuracy_summary(
        labels=labels,
        current_pred=current,
        label_names=["skip", "trend", "direction"],
        oracle_gain_mse=gain,
        min_abs_improvement=0.001,
    )

    assert summary["label_counts"] == {"skip": 2, "trend": 1, "direction": 1}
    assert summary["oracle_skip_rate"] == pytest.approx(2 / 4)
    assert summary["penalty_accuracy_on_oracle_penalty"] == pytest.approx(2 / 2)
    assert summary["oracle_penalty_routed_to_skip_rate"] == pytest.approx(0.0)


def test_train_route_accuracy_summary_matches_skip_inclusive_counts() -> None:
    labels = torch.tensor([0, 1, 1, 2, 0, 2])
    current = torch.tensor([1, 1, 2, 2, 0, 0])
    features = torch.zeros(6, 3, 4)
    features[:, 0, 2] = torch.tensor([0.1, 0.2, 0.3, 0.6, 0.4, 0.8])

    summary = _route_accuracy_summary_from_labels(
        labels=labels,
        current_pred=current,
        label_names=["skip", "trend", "direction"],
        features=features,
        feature_names=["a", "b", "skip_prob", "c"],
    )

    assert summary["current_accuracy_all"] == pytest.approx(3 / 6)
    assert summary["majority_accuracy_all"] == pytest.approx(2 / 6)
    assert summary["oracle_skip_rate"] == pytest.approx(2 / 6)
    assert summary["actual_skip_rate"] == pytest.approx(2 / 6)
    assert summary["skip_recall"] == pytest.approx(1 / 2)
    assert summary["penalty_accuracy_on_oracle_penalty"] == pytest.approx(2 / 4)
    assert summary["penalty_adoption_recall_on_oracle_penalty"] == pytest.approx(3 / 4)
    assert summary["penalty_adoption_precision"] == pytest.approx(3 / 4)
    assert summary["penalty_exact_precision"] == pytest.approx(2 / 4)
    assert summary["missed_positive_adoption_rate"] == pytest.approx(1 / 4)
    assert summary["penalty_adoption_rate_gap_vs_oracle"] == pytest.approx(0.0)
    assert summary["confusion_matrix_counts"] == [
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1],
    ]
    assert summary["skip_probability"]["mean"] == pytest.approx(0.4)
    assert summary["skip_probability"]["gt_0_5_rate"] == pytest.approx(2 / 6)


def test_cluster_route_oracle_labels_use_skip_zero_and_allowed_mask() -> None:
    y = torch.zeros(2, 3, 1)
    base = torch.ones(2, 3, 1)
    cand = base.unsqueeze(2).repeat(1, 1, 2, 1)
    # Batch 0: cluster 0 penalty 0 helps, cluster 1 has no allowed improvement.
    cand[0, 0:2, 0, 0] = 0.0
    cand[0, 0:2, 1, 0] = 2.0
    cand[0, 2, 1, 0] = 1.5
    # Batch 1: cluster 0 only the disallowed penalty helps; cluster 1 penalty 1 helps.
    cand[1, 0:2, 0, 0] = 1.2
    cand[1, 0:2, 1, 0] = 0.0
    cand[1, 2, 1, 0] = 0.0
    cluster_id = torch.tensor([0, 0, 1])
    allowed = torch.tensor([[1, 0], [0, 1]], dtype=torch.bool)

    labels = _cluster_route_oracle_labels_from_candidates(
        base_bch=base,
        cand_bcpH=cand,
        y_bch=y,
        cluster_id_c=cluster_id,
        K=2,
        allowed_mask_kp=allowed,
    )

    assert labels.tolist() == [[1, 0], [0, 2]]


def test_stage2_route_audit_thresholds_prefer_explicit_config() -> None:
    thresholds = _stage2_route_audit_thresholds(
        stage2_route_audit_cfg={
            "min_abs_improvement": 0.003,
            "min_rel_improvement": 0.02,
            "min_candidate_delta_rms": 0.004,
        },
        route_ce_min_abs_improvement=0.0,
        route_ce_min_rel_improvement=0.0,
        route_ce_min_candidate_delta_rms=0.001,
        binary_adoption_weight=1.0,
        binary_adoption_min_abs_improvement=0.001,
        binary_adoption_min_rel_improvement=0.0,
        binary_adoption_min_candidate_delta_rms=0.001,
    )

    assert thresholds == {
        "min_abs_improvement": 0.003,
        "min_rel_improvement": 0.02,
        "min_candidate_delta_rms": 0.004,
        "source": "diagnostics.stage2_route_audit",
    }


def test_stage2_route_audit_thresholds_follow_binary_adoption_when_active() -> None:
    thresholds = _stage2_route_audit_thresholds(
        stage2_route_audit_cfg={},
        route_ce_min_abs_improvement=0.0,
        route_ce_min_rel_improvement=0.0,
        route_ce_min_candidate_delta_rms=0.0,
        binary_adoption_weight=1.0,
        binary_adoption_min_abs_improvement=0.001,
        binary_adoption_min_rel_improvement=0.0,
        binary_adoption_min_candidate_delta_rms=0.002,
    )

    assert thresholds == {
        "min_abs_improvement": 0.001,
        "min_rel_improvement": 0.0,
        "min_candidate_delta_rms": 0.002,
        "source": "moe.binary_adoption_supervision",
    }


def test_cluster_route_oracle_labels_and_gain_share_allowed_mask_logic() -> None:
    y = torch.zeros(2, 3, 1)
    base = torch.ones(2, 3, 1)
    cand = base.unsqueeze(2).repeat(1, 1, 2, 1)
    cand[0, 0:2, 0, 0] = 0.0
    cand[0, 0:2, 1, 0] = 2.0
    cand[0, 2, 1, 0] = 1.5
    cand[1, 0:2, 0, 0] = 1.2
    cand[1, 0:2, 1, 0] = 0.0
    cand[1, 2, 1, 0] = 0.0
    cluster_id = torch.tensor([0, 0, 1])
    allowed = torch.tensor([[1, 0], [0, 1]], dtype=torch.bool)

    labels, gain = _cluster_route_oracle_labels_and_gain_from_candidates(
        base_bch=base,
        cand_bcpH=cand,
        y_bch=y,
        cluster_id_c=cluster_id,
        K=2,
        allowed_mask_kp=allowed,
    )

    assert labels.tolist() == [[1, 0], [0, 2]]
    torch.testing.assert_close(gain, torch.tensor([[1.0, -1.25], [-0.44, 1.0]]))


def test_cluster_route_oracle_can_ignore_near_noop_candidate_when_enabled() -> None:
    y = torch.zeros(2, 1, 1)
    base = torch.ones(2, 1, 1)
    cand = base.unsqueeze(2).repeat(1, 1, 2, 1)
    cand[0, 0, 0, 0] = 2.0
    cand[0, 0, 1, 0] = 0.999
    cand[1, 0, 0, 0] = 0.999
    cand[1, 0, 1, 0] = 0.998
    cluster_id = torch.tensor([0])

    default_labels, default_gain = _cluster_route_oracle_labels_and_gain_from_candidates(
        base_bch=base,
        cand_bcpH=cand,
        y_bch=y,
        cluster_id_c=cluster_id,
        K=1,
    )
    floor_labels, floor_gain = _cluster_route_oracle_labels_and_gain_from_candidates(
        base_bch=base,
        cand_bcpH=cand,
        y_bch=y,
        cluster_id_c=cluster_id,
        K=1,
        min_candidate_delta_rms=0.01,
    )

    assert default_labels.tolist() == [[2], [2]]
    assert floor_labels.tolist() == [[0], [0]]
    torch.testing.assert_close(default_gain, torch.tensor([[0.00199902], [0.00399601]]), rtol=1.0e-4, atol=1.0e-6)
    torch.testing.assert_close(floor_gain, torch.tensor([[-3.0], [0.0]]), rtol=1.0e-4, atol=1.0e-6)


def test_route_ce_active_mask_ignores_only_near_zero_gain_when_enabled() -> None:
    gain = torch.tensor([[-0.0020, -0.0005, 0.0, 0.0002, 0.0030]])

    assert _route_ce_active_mask_from_gain(gain, ignore_abs_gain_below=0.0).tolist() == [
        [True, True, True, True, True]
    ]
    assert _route_ce_active_mask_from_gain(gain, ignore_abs_gain_below=0.001).tolist() == [
        [True, False, False, False, True]
    ]


def test_pred_residual_training_skip_arg_is_default_on_and_can_be_decoupled() -> None:
    skip = torch.tensor([[0.0, 1.0]])

    assert _pred_residual_training_skip_arg(
        skip_bk=skip,
        allow_skip=True,
        ignore_skip_during_training=False,
    ) is skip
    assert _pred_residual_training_skip_arg(
        skip_bk=skip,
        allow_skip=False,
        ignore_skip_during_training=False,
    ) is None
    assert _pred_residual_training_skip_arg(
        skip_bk=skip,
        allow_skip=True,
        ignore_skip_during_training=True,
    ) is None


def test_route_ce_loss_from_probs_trains_skip_and_penalty_classes() -> None:
    logits = torch.tensor(
        [
            [[0.2, 0.1, -0.3], [-0.4, 0.3, 0.0]],
            [[0.0, -0.1, 0.2], [0.1, 0.0, -0.2]],
        ],
        requires_grad=True,
    )
    route_probs = torch.softmax(logits, dim=-1)
    labels = torch.tensor([[0, 2], [1, 0]])

    loss_bk = _route_ce_loss_from_probs(
        probs_bkp=route_probs[..., 1:],
        skip_prob_bk=route_probs[..., 0],
        labels_bk=labels,
        probs_include_skip_mass=True,
    )
    loss = loss_bk.mean()
    loss.backward()

    assert loss_bk.shape == labels.shape
    for b in range(labels.shape[0]):
        for k in range(labels.shape[1]):
            target = int(labels[b, k].item())
            assert logits.grad[b, k, target].item() < 0.0


def test_route_ce_balanced_class_weight_uses_train_batch_labels_only() -> None:
    labels = torch.tensor([[0, 1, 1], [1, 2, 1]])

    weights = _route_ce_class_weight_from_labels(
        labels_bk=labels,
        num_classes=3,
        mode="balanced",
    )

    assert weights.tolist() == pytest.approx([2.0, 0.5, 2.0])
    assert _route_ce_class_weight_from_labels(
        labels_bk=labels,
        num_classes=3,
        mode="none",
    ) is None


def test_route_ce_balanced_class_weight_respects_active_mask() -> None:
    labels = torch.tensor([[0, 0, 1, 1, 2]])
    active = torch.tensor([[1, 0, 1, 0, 1]], dtype=torch.bool)

    weights = _route_ce_class_weight_from_labels(
        labels_bk=labels,
        num_classes=3,
        mode="balanced",
        active_mask_bk=active,
    )

    assert weights.tolist() == pytest.approx([1.0, 1.0, 1.0])
