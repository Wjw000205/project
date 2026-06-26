import pytest
import torch

from scripts.next11d_channel_action_space_diagnostic import (
    _channel_conflict_metrics,
    _channel_label_route_metrics,
    _channel_oracle_labels_from_gain,
    _cluster_projection_from_channel_labels,
    _normalize_requested_splits,
)


def test_channel_oracle_labels_respect_margin_and_allowed_mask() -> None:
    gain = torch.tensor(
        [
            [[0.20, 0.30], [0.01, 0.02]],
            [[-0.10, 0.30], [0.00, 0.20]],
        ],
        dtype=torch.float32,
    )
    allowed = torch.tensor(
        [
            [True, False],
            [True, True],
        ]
    )

    labels = _channel_oracle_labels_from_gain(gain, allowed_cp=allowed, margin=0.05)

    assert labels.tolist() == [
        [1, 0],
        [0, 2],
    ]


def test_cluster_projection_exposes_recall_precision_tradeoff_for_channel_labels() -> None:
    labels = torch.tensor(
        [
            [0, 1, 1],
            [0, 0, 2],
        ],
        dtype=torch.long,
    )
    cluster_id = torch.tensor([0, 0, 0], dtype=torch.long)
    label_names = ["skip", "trend", "direction"]

    majority_route = _cluster_projection_from_channel_labels(
        labels_bc=labels,
        cluster_id_c=cluster_id,
        cluster_count=1,
        label_count=len(label_names),
        mode="majority",
    )
    positive_first_route = _cluster_projection_from_channel_labels(
        labels_bc=labels,
        cluster_id_c=cluster_id,
        cluster_count=1,
        label_count=len(label_names),
        mode="positive_first",
    )

    assert majority_route.tolist() == [[1], [0]]
    assert positive_first_route.tolist() == [[1], [2]]

    majority_metrics = _channel_label_route_metrics(
        labels_bc=labels,
        route_bk=majority_route,
        cluster_id_c=cluster_id,
        label_names=label_names,
    )
    positive_first_metrics = _channel_label_route_metrics(
        labels_bc=labels,
        route_bk=positive_first_route,
        cluster_id_c=cluster_id,
        label_names=label_names,
    )
    conflict = _channel_conflict_metrics(
        labels_bc=labels,
        cluster_id_c=cluster_id,
        cluster_count=1,
        label_names=label_names,
    )

    assert majority_metrics["accuracy_all"] == pytest.approx(4.0 / 6.0)
    assert majority_metrics["positive_recall_any"] == pytest.approx(2.0 / 3.0)
    assert majority_metrics["positive_precision_any"] == pytest.approx(2.0 / 3.0)
    assert positive_first_metrics["accuracy_all"] == pytest.approx(3.0 / 6.0)
    assert positive_first_metrics["positive_recall_any"] == pytest.approx(1.0)
    assert positive_first_metrics["positive_precision_any"] == pytest.approx(0.5)
    assert positive_first_metrics["oracle_skip_routed_to_penalty_rate"] == pytest.approx(1.0)
    assert conflict["mixed_skip_positive_rate"] == pytest.approx(1.0)
    assert conflict["multi_positive_penalty_rate"] == pytest.approx(0.0)


def test_channel_conflict_ceiling_counts_channels_for_uneven_clusters() -> None:
    labels = torch.tensor([[0, 1, 2]], dtype=torch.long)
    cluster_id = torch.tensor([0, 0, 1], dtype=torch.long)

    conflict = _channel_conflict_metrics(
        labels_bc=labels,
        cluster_id_c=cluster_id,
        cluster_count=2,
        label_names=["skip", "trend", "direction"],
    )

    assert conflict["best_single_cluster_label_channel_accuracy_ceiling"] == pytest.approx(2.0 / 3.0)


def test_normalize_requested_splits_refuses_test() -> None:
    with pytest.raises(ValueError, match="refuses to read test"):
        _normalize_requested_splits(["train_fit", "test"])
