from __future__ import annotations

import numpy as np


def select_prediction_sample_indices(
    total: int,
    sample_count: int,
    *,
    strategy: str = "first",
    seed: int = 0,
) -> list[int]:
    total = max(int(total), 0)
    count = max(int(sample_count), 0)
    if total == 0 or count == 0:
        return []
    count = min(total, count)
    mode = str(strategy or "first").lower()
    if mode in {"first", "consecutive"}:
        return list(range(count))
    if mode in {"stratified_random", "random_distributed", "distributed_random"}:
        rng = np.random.default_rng(int(seed))
        selected = []
        for i in range(count):
            lo = int(np.floor(i * total / count))
            hi = int(np.floor((i + 1) * total / count)) - 1
            hi = max(lo, hi)
            selected.append(int(rng.integers(lo, hi + 1)))
        return selected
    if mode == "random":
        rng = np.random.default_rng(int(seed))
        return sorted(int(v) for v in rng.choice(total, size=count, replace=False).tolist())
    raise ValueError(f"unknown prediction sample strategy: {strategy}")
