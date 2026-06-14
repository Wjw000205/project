from __future__ import annotations

from src.utils.diagnostic_sampling import select_prediction_sample_indices


def test_stratified_random_samples_cover_full_range_without_consecutive_prefix() -> None:
    indices = select_prediction_sample_indices(
        total=100,
        sample_count=10,
        strategy="stratified_random",
        seed=7,
    )

    assert len(indices) == 10
    assert indices != list(range(10))
    assert indices == sorted(indices)
    assert all(i * 10 <= value < (i + 1) * 10 for i, value in enumerate(indices))


def test_first_sampling_keeps_legacy_consecutive_behavior() -> None:
    assert select_prediction_sample_indices(total=100, sample_count=5, strategy="first", seed=7) == [0, 1, 2, 3, 4]


def test_sampling_caps_at_available_windows() -> None:
    assert select_prediction_sample_indices(total=3, sample_count=10, strategy="stratified_random", seed=7) == [0, 1, 2]
