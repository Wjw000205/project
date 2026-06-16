from __future__ import annotations

from scripts.build_backbone_gap_penalty_profile import select_penalties_for_cluster


def test_well_learned_cluster_gets_light_corr_only() -> None:
    row = {
        "cluster_id": 0,
        "channels": 2,
        "mse": 0.05,
        "mae": 0.15,
        "bias_mean_y_minus_pred": 0.01,
        "pred_to_y_std_ratio": 1.02,
        "corr_pred_y": 0.95,
        "horizon_bias_abs_mean": 0.01,
        "spike_miss_mean": 0.08,
        "spike_abs_error": 0.10,
    }

    selected, gap_tags = select_penalties_for_cluster(row)

    assert selected == ["corr"]
    assert gap_tags == ["well_learned"]


def test_compressed_event_cluster_uses_amplitude_event_penalties_without_corr() -> None:
    row = {
        "cluster_id": 3,
        "channels": 2,
        "mse": 2.5,
        "mae": 0.37,
        "bias_mean_y_minus_pred": 0.25,
        "pred_to_y_std_ratio": 0.35,
        "corr_pred_y": 0.24,
        "horizon_bias_abs_mean": 0.25,
        "spike_miss_mean": 4.5,
        "spike_abs_error": 4.6,
    }

    selected, gap_tags = select_penalties_for_cluster(row)

    assert "amp_under" in selected
    assert "range" in selected
    assert "level" in selected
    assert "seasonal_align" in selected
    assert "corr" not in selected
    assert "delta" not in selected
    assert "direction" not in selected
    assert "variance_compression" in gap_tags
    assert "event_miss" in gap_tags


def test_low_correlation_without_variance_collapse_gets_dynamic_penalties() -> None:
    row = {
        "cluster_id": 2,
        "channels": 5,
        "mse": 0.35,
        "mae": 0.33,
        "bias_mean_y_minus_pred": 0.01,
        "pred_to_y_std_ratio": 0.78,
        "corr_pred_y": 0.54,
        "horizon_bias_abs_mean": 0.02,
        "spike_miss_mean": 0.78,
        "spike_abs_error": 0.80,
    }

    selected, gap_tags = select_penalties_for_cluster(row)

    assert "delta" in selected
    assert "direction" in selected
    assert "seasonal_align" in selected
    assert "amp_under" in selected
    assert "corr" not in selected
    assert "dynamic_mismatch" in gap_tags
