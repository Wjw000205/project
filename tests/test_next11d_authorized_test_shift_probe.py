import pytest

from scripts.next11d_authorized_test_shift_probe import (
    _classify_val_test_shift,
    _normalize_requested_splits,
)


def test_normalize_requested_splits_requires_explicit_test_authorization() -> None:
    with pytest.raises(ValueError, match="requires --allow-test-read"):
        _normalize_requested_splits(["val", "test"], allow_test_read=False)

    assert _normalize_requested_splits(["val", "test"], allow_test_read=True) == ["val", "test"]


def test_classify_val_test_shift_flags_persistent_candidate_instability() -> None:
    verdict = _classify_val_test_shift(
        {
            "val": {
                "channel_oracle_gain_pct_vs_base": 2.0,
                "stable_temporal_channel_candidates": 0,
                "stable_temporal_cluster_candidates": 0,
            },
            "test": {
                "channel_oracle_gain_pct_vs_base": 1.8,
                "stable_temporal_channel_candidates": 0,
                "stable_temporal_cluster_candidates": 0,
            },
        },
        min_oracle_gain_pct=0.25,
    )

    assert verdict["failure_layer"] == "adapter candidate quality"
    assert verdict["decision"] == "candidate_temporal_instability_persists_val_and_test"


def test_classify_val_test_shift_flags_test_mismatch_when_oracle_collapses() -> None:
    verdict = _classify_val_test_shift(
        {
            "val": {
                "channel_oracle_gain_pct_vs_base": 2.0,
                "stable_temporal_channel_candidates": 1,
                "stable_temporal_cluster_candidates": 0,
            },
            "test": {
                "channel_oracle_gain_pct_vs_base": -0.2,
                "stable_temporal_channel_candidates": 0,
                "stable_temporal_cluster_candidates": 0,
            },
        },
        min_oracle_gain_pct=0.25,
    )

    assert verdict["failure_layer"] == "train-val utility shift"
    assert verdict["decision"] == "val_test_candidate_oracle_or_stability_mismatch"
