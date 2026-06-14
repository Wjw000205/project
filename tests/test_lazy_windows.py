from __future__ import annotations

import torch

from src.data import windows


def test_lazy_strict_window_dataset_matches_materialized_windows() -> None:
    data = torch.arange(16 * 3, dtype=torch.float32).reshape(16, 3)
    input_len = 4
    pred_len = 2
    start = 3
    end = 14

    x_eager, y_eager = windows.make_strict_windows(data, input_len, pred_len, start, end)
    make_lazy = getattr(windows, "make_lazy_strict_window_dataset", None)

    assert make_lazy is not None
    lazy = make_lazy(data, input_len, pred_len, start, end)

    assert len(lazy) == len(x_eager)
    assert torch.equal(lazy.start_offsets, torch.arange(start, start + len(x_eager)))
    for idx in (0, 2, len(lazy) - 1):
        x_lazy, y_lazy, rel_idx = lazy[idx]
        assert rel_idx == idx
        assert torch.equal(x_lazy, x_eager[idx])
        assert torch.equal(y_lazy, y_eager[idx])


def test_lazy_label_range_window_dataset_matches_materialized_windows() -> None:
    data = torch.arange(20 * 2, dtype=torch.float32).reshape(20, 2)
    input_len = 5
    pred_len = 3
    label_start = 12
    label_end = 18

    x_eager, y_eager, eager_first_start = windows.make_label_range_windows(
        data,
        input_len,
        pred_len,
        label_start,
        label_end,
    )
    make_lazy = getattr(windows, "make_lazy_label_range_window_dataset", None)

    assert make_lazy is not None
    lazy, lazy_first_start = make_lazy(data, input_len, pred_len, label_start, label_end)

    assert lazy_first_start == eager_first_start
    assert len(lazy) == len(x_eager)
    assert torch.equal(lazy.start_offsets, torch.arange(eager_first_start, eager_first_start + len(x_eager)))
    for idx in (0, len(lazy) - 1):
        x_lazy, y_lazy, rel_idx = lazy[idx]
        assert rel_idx == idx
        assert torch.equal(x_lazy, x_eager[idx])
        assert torch.equal(y_lazy, y_eager[idx])
