from __future__ import annotations

import itertools

import pytest
import torch

from src.utils.cluster_memory import balance_cluster_assignment_by_source_counts


def _score(corr_ck: torch.Tensor, route: list[int]) -> float:
    idx = torch.arange(corr_ck.shape[0])
    return float(corr_ck[idx, torch.tensor(route)].sum().item())


def test_balance_cluster_assignment_repairs_collapsed_argmax_to_source_counts() -> None:
    corr_ck = torch.tensor(
        [
            [0.7576, 0.8681],
            [0.7421, 0.8595],
            [0.7609, 0.8839],
            [0.7366, 0.8454],
            [0.5656, 0.7145],
            [0.7022, 0.8178],
            [0.8124, 0.9649],
        ],
        dtype=torch.float32,
    )
    source_cluster_id_c = torch.tensor([0, 0, 1, 0, 0, 1, 1])

    repaired = balance_cluster_assignment_by_source_counts(corr_ck, source_cluster_id_c)

    assert torch.argmax(corr_ck, dim=1).tolist() == [1, 1, 1, 1, 1, 1, 1]
    assert torch.bincount(repaired, minlength=2).tolist() == [4, 3]

    best_score = max(
        _score(corr_ck, list(route))
        for route in itertools.product([0, 1], repeat=7)
        if list(route).count(0) == 4 and list(route).count(1) == 3
    )
    assert _score(corr_ck, repaired.tolist()) == pytest.approx(best_score)


def test_balance_cluster_assignment_keeps_argmax_when_counts_already_match() -> None:
    corr_ck = torch.tensor(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.6, 0.4],
            [0.2, 0.8],
            [0.1, 0.9],
            [0.3, 0.7],
        ],
        dtype=torch.float32,
    )
    source_cluster_id_c = torch.tensor([0, 0, 0, 0, 1, 1, 1])

    repaired = balance_cluster_assignment_by_source_counts(corr_ck, source_cluster_id_c)

    assert repaired.tolist() == torch.argmax(corr_ck, dim=1).tolist()
