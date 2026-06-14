from __future__ import annotations

import random

from scripts.run_input96_frozen_moe_bayes_search import (
    FrozenMoeCandidate,
    candidate_from_json,
    candidate_to_json,
    candidate_to_patch,
    choose_next_candidate,
    objective_from_summary,
    seed_candidates,
    short_variant_name,
)


def test_objective_prefers_scaled_validation_mse() -> None:
    summary = {
        "val": {"avg_mse": 0.25, "avg_mae": 0.4},
        "moe_residual_selection": {"val_scaled_avg_mse": 0.22, "val_scaled_avg_mae": 0.37},
    }

    objective, source = objective_from_summary(summary)

    assert objective == 0.22
    assert source == "moe_residual_selection.val_scaled_avg_mse"


def test_objective_falls_back_to_raw_validation_mse() -> None:
    summary = {"val": {"avg_mse": 0.25, "avg_mae": 0.4}}

    objective, source = objective_from_summary(summary)

    assert objective == 0.25
    assert source == "val.avg_mse"


def test_objective_can_penalize_overactive_residual_selection() -> None:
    summary = {
        "val": {"avg_mse": 0.25},
        "moe_residual_selection": {
            "val_scaled_avg_mse": 0.22,
            "mean_scale": 0.4,
            "num_residual_channels": 5,
        },
    }

    objective, source = objective_from_summary(
        summary,
        max_residual_channels=3,
        max_mean_scale=0.25,
        channel_penalty=0.01,
        mean_scale_penalty=0.1,
    )

    assert objective == 0.22 + 2 * 0.01 + 0.15 * 0.1
    assert source == "moe_residual_selection.val_scaled_avg_mse+complexity_guard"


def test_objective_can_penalize_raw_residual_degradation() -> None:
    summary = {
        "val": {"avg_mse": 0.25},
        "moe_residual_selection": {
            "val_scaled_avg_mse": 0.22,
            "val_pred_base_avg_mse": 0.21,
            "val_residual_avg_mse": 0.24,
        },
    }

    objective, source = objective_from_summary(summary, residual_degradation_penalty=0.1)

    assert objective == 0.22 + (0.24 - 0.21) * 0.1
    assert source == "moe_residual_selection.val_scaled_avg_mse+residual_degradation_guard"


def test_candidate_patch_sets_scale_residual_moe_controls() -> None:
    cand = FrozenMoeCandidate(
        penalty_pool="current",
        lambda_scale=0.005,
        lambda_profile="flat",
        alpha_scale=1.5,
        residual_clip=4.0,
        selection_scale_max=1.25,
        selection_scale_steps=26,
        selection_min_rel_improvement=0.0005,
        corrector_hidden=64,
        init_alpha=-2.5,
        norm_weight=1.0e-5,
        lr=5.0e-4,
        weight_decay=1.0e-5,
        topk=2,
        router_mode="penalty_context",
        router_context_weight=1.0,
        feature_mode="legacy",
        gate_temperature=1.0,
        gate_noise_std=0.1,
        skip_cost=0.15,
        selection_policy="val_mse_scale",
    )

    patch = candidate_to_patch(cand)
    residual = patch["moe"]["pred_side_residual"]

    assert patch["penalties"]["enabled"] == ["jump", "amp_under", "level", "delta"]
    assert patch["moe"]["enable"] is True
    assert patch["moe"]["dynamic_lambda"]["enable"] is False
    assert patch["moe"]["lambda_init"] == {
        "jump": 0.005,
        "amp_under": 0.005,
        "level": 0.005,
        "delta": 0.005,
    }
    assert patch["moe"]["topk"] == 2
    assert patch["moe"]["select_ranks"] == [1, 2]
    assert patch["moe"]["router_mode"] == "penalty_context"
    assert patch["moe"]["router_penalty_context_weight"] == 1.0
    assert residual["selection_policy"] == "val_mse_scale"
    assert residual["alpha_scale"] == 1.5
    assert residual["residual_clip"] == 4.0
    assert residual["selection_scale_max"] == 1.25
    assert residual["selection_scale_steps"] == 26
    assert residual["corrector_hidden"] == 64
    assert residual["init_alpha"] == -2.5
    assert patch["train"]["lr"] == 5.0e-4
    assert patch["train"]["weight_decay"] == 1.0e-5


def test_candidate_patch_can_use_holdout_scale_selection() -> None:
    cand = FrozenMoeCandidate(
        penalty_pool="current",
        lambda_scale=0.005,
        lambda_profile="flat",
        alpha_scale=1.5,
        residual_clip=4.0,
        selection_scale_max=1.25,
        selection_scale_steps=26,
        selection_min_rel_improvement=0.0005,
        corrector_hidden=64,
        init_alpha=-2.5,
        norm_weight=1.0e-5,
        lr=5.0e-4,
        weight_decay=1.0e-5,
        topk=1,
        router_mode="learned",
        router_context_weight=0.0,
        feature_mode="legacy",
        gate_temperature=1.0,
        gate_noise_std=0.1,
        skip_cost=0.15,
        selection_policy="val_mse_scale_holdout",
        selection_holdout_fraction=0.4,
        selection_holdout_min_windows=256,
        selection_max_residual_channels=3,
        selection_eval_segments=3,
        selection_min_positive_segments=2,
        selection_max_segment_rel_degradation=0.001,
        selection_max_segment_abs_degradation=1.0e-4,
    )

    residual = candidate_to_patch(cand)["moe"]["pred_side_residual"]

    assert residual["selection_policy"] == "val_mse_scale_holdout"
    assert residual["selection_holdout_fraction"] == 0.4
    assert residual["selection_holdout_min_windows"] == 256
    assert residual["selection_max_residual_channels"] == 3
    assert residual["selection_eval_segments"] == 3
    assert residual["selection_min_positive_segments"] == 2
    assert residual["selection_max_segment_rel_degradation"] == 0.001
    assert residual["selection_max_segment_abs_degradation"] == 1.0e-4


def test_candidate_json_roundtrip_keeps_key() -> None:
    cand = seed_candidates()[0]

    restored = candidate_from_json(candidate_to_json(cand))

    assert restored == cand
    assert restored.key() == cand.key()


def test_choose_next_candidate_avoids_tried_candidates() -> None:
    seed = seed_candidates()[0]
    rng = random.Random(2026)

    cand = choose_next_candidate(
        rng=rng,
        observed=[],
        tried={seed.key()},
        pool_size=12,
    )

    assert cand.key() != seed.key()


def test_short_variant_name_keeps_long_candidates_path_safe() -> None:
    cand = FrozenMoeCandidate(
        penalty_pool="jump_amp_level",
        lambda_scale=0.02989,
        lambda_profile="delta_heavy",
        alpha_scale=2.004,
        residual_clip=6.0,
        selection_scale_max=0.6338,
        selection_scale_steps=41,
        selection_min_rel_improvement=0.0005,
        corrector_hidden=64,
        init_alpha=-2.356,
        norm_weight=1.0e-5,
        lr=1.059e-4,
        weight_decay=2.641e-5,
        topk=1,
        router_mode="penalty_context",
        router_context_weight=1.0,
        feature_mode="legacy",
        gate_temperature=1.0,
        gate_noise_std=0.1,
        skip_cost=0.15,
        selection_policy="val_mse_scale_holdout",
        selection_holdout_fraction=0.4,
        selection_holdout_min_windows=256,
    )

    name = short_variant_name(7, cand)

    assert name.startswith("trial_007_")
    assert len(name) <= 70
