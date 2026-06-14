from __future__ import annotations

import pytest

from scripts.run_input96_ettm2_frozen_moe_recipe import select_variants, variants


def test_ettm2_recipe_includes_activation_guard_variants() -> None:
    by_name = {name: patch for name, patch in variants()}

    patch = by_name["zero_safe_act_mse_min1e3"]
    residual = patch["moe"]["pred_side_residual"]
    gate = residual["gate_calibrator"]

    assert patch["penalties"]["enabled"] == ["trend", "direction"]
    assert patch["moe"]["lambda_init"] == {"trend": 0.0, "direction": 0.0}
    assert residual["feature_mode"] == "safe_augmented"
    assert residual["selection_policy"] == "val_mse_gate_guarded"
    assert gate["activation_head_enable"] is True
    assert gate["apply_activation_threshold"] is True
    assert gate["activation_threshold"] == "auto"
    assert gate["activation_threshold_selection_metric"] == "mse"
    assert gate["activation_threshold_scope"] == "channel"
    assert gate["activation_label_min_improvement"] == 0.001
    assert gate["activation_pos_weight_scope"] == "channel"


def test_ettm2_recipe_select_variants_keeps_order() -> None:
    selected = select_variants(
        variants(),
        ["zero_safe_act_balacc_bce02", "zero_lambda_channel_delta_no_extra_safe_aug"],
    )

    assert [name for name, _ in selected] == [
        "zero_safe_act_balacc_bce02",
        "zero_lambda_channel_delta_no_extra_safe_aug",
    ]


def test_ettm2_recipe_includes_safe_aug_strength_variants() -> None:
    by_name = {name: patch for name, patch in variants()}

    strong = by_name["zero_safe_alpha2_clip6_gate_ms25"]["moe"]["pred_side_residual"]
    conservative = by_name["zero_safe_alpha1_clip2_gate_ms15"]["moe"]["pred_side_residual"]

    assert strong["feature_mode"] == "safe_augmented"
    assert strong["alpha_scale"] == 2.0
    assert strong["residual_clip"] == 6.0
    assert strong["gate_calibrator"]["max_scale"] == 2.5
    assert conservative["alpha_scale"] == 1.0
    assert conservative["residual_clip"] == 2.0
    assert conservative["gate_calibrator"]["max_scale"] == 1.5


def test_ettm2_recipe_select_variants_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown variants"):
        select_variants(variants(), ["missing_variant"])
