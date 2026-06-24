from __future__ import annotations

import numpy as np

from scripts.compute_train_residual_penalty_portrait import finalize_pearson_corr, select_pool_top3
from scripts.run_anchorless_moe_diagnostic import diagnostic_pools


def test_select_pool_top3_ranks_by_global_mean_ratio() -> None:
    penalty_names = ["level", "delta", "direction", "corr"]
    portrait_raw = np.array(
        [
            [2.0, 6.0, 1.0, 0.5],
            [1.0, 2.0, 9.0, 0.2],
        ],
        dtype=np.float64,
    )
    penalty_global_mean = np.array([1.0, 3.0, 1.0, 0.1], dtype=np.float64)

    selected = select_pool_top3(portrait_raw, penalty_global_mean, penalty_names)

    assert selected == {
        "0": ["corr", "level", "delta"],
        "1": ["direction", "corr", "level"],
    }


def test_select_pool_top3_excludes_high_mse_corr_penalties() -> None:
    penalty_names = ["level", "delta", "direction", "corr"]
    portrait_raw = np.array(
        [
            [8.0, 7.0, 6.0, 5.0],
            [8.0, 7.0, 6.0, 5.0],
        ],
        dtype=np.float64,
    )
    penalty_global_mean = np.ones(4, dtype=np.float64)
    penalty_mse_corr_by_cluster = np.array(
        [
            [0.91, 0.10, -0.30, 0.81],
            [0.20, 0.85, -0.95, 0.10],
        ],
        dtype=np.float64,
    )

    selected = select_pool_top3(
        portrait_raw,
        penalty_global_mean,
        penalty_names,
        penalty_mse_corr=penalty_mse_corr_by_cluster,
        max_abs_mse_corr=0.80,
    )

    assert selected == {
        "0": ["delta", "direction"],
        "1": ["level", "corr"],
    }


def test_finalize_pearson_corr_handles_positive_negative_and_constant_columns() -> None:
    x = np.array(
        [
            [1.0, 3.0, 5.0],
            [2.0, 2.0, 5.0],
            [3.0, 1.0, 5.0],
        ],
        dtype=np.float64,
    )
    y = np.array([2.0, 4.0, 6.0], dtype=np.float64)

    corr = finalize_pearson_corr(
        sum_x=x.sum(axis=0),
        sum_y=np.full(3, y.sum(), dtype=np.float64),
        sum_xx=(x * x).sum(axis=0),
        sum_yy=np.full(3, (y * y).sum(), dtype=np.float64),
        sum_xy=(x * y.reshape(-1, 1)).sum(axis=0),
        count=np.full(3, len(y), dtype=np.float64),
    )

    np.testing.assert_allclose(corr, np.array([1.0, -1.0, 0.0]), atol=1.0e-12)


def test_diagnostic_pools_excludes_global_high_mse_corr_from_global3() -> None:
    payload = {
        "portrait_raw": [[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]],
        "penalty_global_mean": [1.0] * 10,
        "selected_pool_top3": {"0": ["amp_under", "delta", "diff_amp"]},
        "penalty_mse_corr": [0.95, 0.10, 0.20, 0.30, 0.10, 0.20, 0.30, 0.10, 0.20, 0.30],
        "mse_corr_exclusion": {"max_abs_corr": 0.80},
    }

    pools = diagnostic_pools(payload)

    assert pools["diag_global3"] == ["amp_under", "delta", "diff_amp"]
