from __future__ import annotations

import numpy as np

from scripts.profile_base_residual_penalty_pool import remap_cluster_labels


def test_remap_cluster_labels_keeps_ordered_dense_labels() -> None:
    remapped, mapping = remap_cluster_labels(np.array([2, 2, 5, 2, 9]))

    assert remapped.tolist() == [0, 0, 1, 0, 2]
    assert mapping == {2: 0, 5: 1, 9: 2}
