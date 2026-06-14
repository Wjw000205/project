from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import (  # noqa: E402
    _accumulate_detached_sum_,
    _contiguous_segment_ranges,
    _freeze_module_params,
    _lr_warmup_scale,
    _make_torch_generator,
    _normalize_loss_terms,
    _should_update_swa,
    _top_positive_improvement_mask,
    _validation_holdout_split_counts,
)


def test_accumulate_detached_sum_does_not_retain_batch_graph() -> None:
    accumulator = torch.zeros(2)
    values = torch.tensor([[1.0, 2.0], [3.0, 5.0]], requires_grad=True)

    returned = _accumulate_detached_sum_(accumulator, values)

    assert returned is accumulator
    assert torch.equal(accumulator, torch.tensor([4.0, 7.0]))
    assert accumulator.grad_fn is None

    values.sum().backward()
    assert values.grad is not None
    assert accumulator.grad is None


def test_loss_term_normalization_equalizes_batch_std_and_keeps_gradients() -> None:
    mse = torch.tensor([[1.0, 3.0], [5.0, 7.0]], requires_grad=True)
    penalty = torch.tensor([[10.0, 14.0], [18.0, 22.0]], requires_grad=True)

    normalized, scales = _normalize_loss_terms(
        {"mse": mse, "penalty": penalty},
        {"enable": True, "mode": "std", "eps": 1.0e-6},
    )

    mse_std = normalized["mse"].detach().std(unbiased=False)
    penalty_std = normalized["penalty"].detach().std(unbiased=False)
    assert torch.allclose(mse_std, penalty_std, atol=1.0e-6)
    assert torch.allclose(scales["mse"], torch.tensor(mse.detach().std(unbiased=False).item()))
    assert torch.allclose(scales["penalty"], torch.tensor(penalty.detach().std(unbiased=False).item()))

    (normalized["mse"].sum() + normalized["penalty"].sum()).backward()
    assert mse.grad is not None
    assert penalty.grad is not None


def test_loss_term_normalization_can_target_selected_terms() -> None:
    mse = torch.tensor([[1.0, 2.0]])
    penalty = torch.tensor([[4.0, 8.0]])

    normalized, scales = _normalize_loss_terms(
        {"mse": mse, "penalty": penalty},
        {"enable": True, "terms": ["penalty"], "mode": "mean_abs", "eps": 1.0e-6},
    )

    assert torch.equal(normalized["mse"], mse)
    assert torch.allclose(normalized["penalty"], torch.tensor([[2.0 / 3.0, 4.0 / 3.0]]))
    assert scales["mse"] is None
    assert torch.allclose(scales["penalty"], torch.tensor(6.0))


def test_seeded_torch_generator_replays_shuffle_order() -> None:
    generator_a = _make_torch_generator(2026)
    generator_b = _make_torch_generator(2026)

    assert generator_a is not None
    assert generator_b is not None
    assert torch.equal(
        torch.randperm(20, generator=generator_a),
        torch.randperm(20, generator=generator_b),
    )
    assert _make_torch_generator(None) is None


def test_lr_warmup_scale_reaches_one_on_final_warmup_epoch() -> None:
    assert _lr_warmup_scale(epoch=1, warmup_epochs=5, start_factor=0.2) == 0.2
    assert _lr_warmup_scale(epoch=3, warmup_epochs=5, start_factor=0.2) == 0.6
    assert _lr_warmup_scale(epoch=5, warmup_epochs=5, start_factor=0.2) == 1.0
    assert _lr_warmup_scale(epoch=6, warmup_epochs=5, start_factor=0.2) == 1.0


def test_swa_update_schedule_starts_at_configured_epoch() -> None:
    assert not _should_update_swa(epoch=4, start_epoch=5, update_every=2)
    assert _should_update_swa(epoch=5, start_epoch=5, update_every=2)
    assert not _should_update_swa(epoch=6, start_epoch=5, update_every=2)
    assert _should_update_swa(epoch=7, start_epoch=5, update_every=2)


def test_freeze_module_params_disables_gradients_and_counts_params() -> None:
    module = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))

    frozen = _freeze_module_params(module)

    assert frozen == sum(param.numel() for param in module.parameters())
    assert all(not param.requires_grad for param in module.parameters())


def test_validation_holdout_split_counts_keep_nonempty_splits() -> None:
    assert _validation_holdout_split_counts(total=1000, holdout_fraction=0.4, min_holdout=128) == (600, 400)
    assert _validation_holdout_split_counts(total=200, holdout_fraction=0.5, min_holdout=128) == (72, 128)
    assert _validation_holdout_split_counts(total=64, holdout_fraction=0.5, min_holdout=128) == (64, 0)
    assert _validation_holdout_split_counts(total=1000, holdout_fraction=0.0, min_holdout=128) == (1000, 0)


def test_contiguous_segment_ranges_cover_total_without_overlap() -> None:
    assert _contiguous_segment_ranges(total=0, segment_count=3) == []
    assert _contiguous_segment_ranges(total=3, segment_count=8) == [(0, 1), (1, 2), (2, 3)]
    assert _contiguous_segment_ranges(total=10, segment_count=3) == [(0, 4), (4, 7), (7, 10)]


def test_top_positive_improvement_mask_keeps_only_best_positive_channels() -> None:
    improvement = torch.tensor([0.05, -0.1, 0.02, 0.07, 0.0])

    mask = _top_positive_improvement_mask(improvement, max_channels=2)

    assert torch.equal(mask, torch.tensor([True, False, False, True, False]))
    assert torch.equal(_top_positive_improvement_mask(improvement, max_channels=0), improvement > 0)
