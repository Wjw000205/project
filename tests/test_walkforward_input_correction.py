import torch

from scripts.diagnose_etth1_walkforward_input_correction import (
    apply_weighted_ridge_residual,
    apply_correction_patch_mask,
    build_input_correction_features,
    build_correction_gate_features,
    correction_gate_targets,
    fit_weighted_ridge_residual,
    online_refit_corrections,
    stabilize_correction_horizon_mean,
    walk_forward_diagnostic,
)
from scripts.diagnose_etth1_fixed_expert_patch_moe import (
    _route_with_default_margin,
    apply_route_participation_guard,
    apply_fixed_expert_routes,
    build_fixed_expert_gate_features,
    fixed_expert_gate_targets,
)


def test_input_correction_features_are_causal_and_finite() -> None:
    x = torch.arange(2 * 3 * 96, dtype=torch.float32).reshape(2, 3, 96)
    base = torch.arange(2 * 3 * 96, dtype=torch.float32).reshape(2, 3, 96) * 0.1

    features, names = build_input_correction_features(x, base)

    assert features.shape[:2] == (2, 3)
    assert features.shape[-1] == len(names)
    assert "hist_mean_96" in names
    assert "base_patch_mean_3" in names
    assert torch.isfinite(features).all()


def test_weighted_ridge_recovers_linear_residual_mapping() -> None:
    torch.manual_seed(4)
    features = torch.randn(64, 2, 3)
    coef = torch.tensor(
        [
            [[1.0, -0.5], [0.2, 0.3], [-0.4, 0.1]],
            [[-0.2, 0.8], [0.5, -0.1], [0.3, 0.4]],
        ]
    )
    residual = torch.einsum("ncf,cfh->nch", features, coef) + 0.25
    state = fit_weighted_ridge_residual(features, residual, ridge=1.0e-6, half_life=0.0)
    pred = apply_weighted_ridge_residual(
        features,
        state,
        shrink=1.0,
        max_abs_scale=0.0,
        x_ncl=torch.ones(64, 2, 4),
    )

    assert torch.allclose(pred, residual, atol=1.0e-4)


def test_walk_forward_uses_only_prior_blocks_and_reports_gain() -> None:
    torch.manual_seed(9)
    n = 60
    x = torch.randn(n, 1, 96)
    base = torch.zeros(n, 1, 96)
    signal = x[..., -1:].expand(-1, -1, 96) * 0.2
    y = base + signal

    summary = walk_forward_diagnostic(
        {"x": x, "base": base, "y": y},
        blocks=6,
        warmup_blocks=2,
        ridge=1.0e-3,
        half_life=0.0,
        shrink=1.0,
        max_abs_scale=0.0,
    )

    assert summary["required_blocks"] == 4
    assert [row["fit_windows"] for row in summary["blocks"]] == [20, 30, 40, 50]
    assert summary["mse_gain_pct"] > 99.0

    purged = walk_forward_diagnostic(
        {"x": x, "base": base, "y": y},
        blocks=6,
        warmup_blocks=2,
        ridge=1.0e-3,
        half_life=0.0,
        shrink=1.0,
        max_abs_scale=0.0,
        label_delay=4,
    )
    assert [row["fit_end_window"] for row in purged["blocks"]] == [17, 27, 37, 47]


def test_walk_forward_channel_mask_leaves_inactive_channel_at_base() -> None:
    torch.manual_seed(10)
    n = 60
    x = torch.randn(n, 2, 96)
    base = torch.zeros(n, 2, 96)
    y = base + x[..., -1:].expand(-1, -1, 96) * 0.2

    summary = walk_forward_diagnostic(
        {"x": x, "base": base, "y": y},
        blocks=6,
        warmup_blocks=2,
        ridge=1.0e-3,
        half_life=0.0,
        shrink=1.0,
        max_abs_scale=0.0,
        active_channels=[0],
    )

    assert summary["config"]["active_channels"] == [0]
    assert summary["per_channel"][1]["corrected_mse"] == summary["per_channel"][1]["base_mse"]


def test_correction_mean_stabilization_uses_reference_not_eval_batch_mean() -> None:
    correction = torch.tensor([[[3.0, 5.0]], [[7.0, 9.0]]])
    reference = torch.tensor([[[1.0, 3.0]], [[3.0, 5.0]]])

    stabilized = stabilize_correction_horizon_mean(
        correction,
        reference,
        x_ncl=torch.ones(2, 1, 4),
        max_abs_scale=0.0,
    )

    assert torch.allclose(stabilized.mean(dim=-1), torch.full((2, 1), 3.0))
    assert torch.allclose(stabilized.diff(dim=-1), correction.diff(dim=-1))


def test_correction_gate_features_and_targets_are_patch_level() -> None:
    x = torch.zeros(1, 1, 4)
    base = torch.ones(1, 1, 4)
    correction = torch.tensor([[[ -1.0, -1.0, 1.0, 1.0 ]]])
    target = torch.zeros(1, 1, 4)

    features, patch_count = build_correction_gate_features(x, base, correction, patch_len=2)
    labels = correction_gate_targets(base, correction, target, patch_len=2)

    assert patch_count == 2
    assert features.shape[:3] == (1, 1, 2)
    assert torch.isfinite(features).all()
    assert torch.equal(labels, torch.tensor([[[1.0, 0.0]]]))


def test_apply_correction_patch_mask_reconstructs_horizon() -> None:
    correction = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
    active = torch.tensor([[[True, False, True, False]]])

    gated = apply_correction_patch_mask(correction, active, patch_len=2)

    assert torch.equal(gated, torch.tensor([[[0.0, 1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 0.0]]]))


def test_domain_aligned_features_remove_eval_affine_shift() -> None:
    features = torch.tensor([[[1.0]], [[2.0]], [[4.0]]])
    shifted = 10.0 + 3.0 * features
    state = {
        "feature_mean_cf": torch.zeros(1, 1),
        "feature_std_cf": torch.ones(1, 1),
        "coef_cfh": torch.tensor([[[0.0], [1.0]]]),
    }
    x = torch.ones(3, 1, 2)

    original = apply_weighted_ridge_residual(
        features,
        state,
        shrink=1.0,
        max_abs_scale=0.0,
        x_ncl=x,
        domain_align_features=True,
    )
    aligned = apply_weighted_ridge_residual(
        shifted,
        state,
        shrink=1.0,
        max_abs_scale=0.0,
        x_ncl=x,
        domain_align_features=True,
    )

    assert torch.allclose(original, aligned, atol=1.0e-6)


def test_domain_alignment_can_be_limited_to_selected_channels() -> None:
    features = torch.tensor([[[1.0], [1.0]], [[2.0], [2.0]], [[4.0], [4.0]]])
    shifted = 10.0 + 3.0 * features
    state = {
        "feature_mean_cf": torch.zeros(2, 1),
        "feature_std_cf": torch.ones(2, 1),
        "coef_cfh": torch.tensor([[[0.0], [1.0]], [[0.0], [1.0]]]),
    }
    x = torch.ones(3, 2, 2)

    partial = apply_weighted_ridge_residual(
        shifted,
        state,
        shrink=1.0,
        max_abs_scale=0.0,
        x_ncl=x,
        domain_align_channels=[0],
    )

    assert abs(float(partial[:, 0].mean().item())) < 1.0e-6
    assert float(partial[:, 1].mean().item()) > 10.0


def test_causal_domain_alignment_is_invariant_to_future_samples() -> None:
    features = torch.tensor([[[1.0]], [[2.0]], [[100.0]]])
    state = {
        "feature_mean_cf": torch.zeros(1, 1),
        "feature_std_cf": torch.ones(1, 1),
        "coef_cfh": torch.tensor([[[0.0], [1.0]]]),
    }
    x = torch.ones(3, 1, 2)

    prefix = apply_weighted_ridge_residual(
        features[:2],
        state,
        shrink=1.0,
        max_abs_scale=0.0,
        x_ncl=x[:2],
        domain_align_channels=[0],
        domain_align_causal_prior_count=2,
    )
    full = apply_weighted_ridge_residual(
        features,
        state,
        shrink=1.0,
        max_abs_scale=0.0,
        x_ncl=x,
        domain_align_channels=[0],
        domain_align_causal_prior_count=2,
    )

    assert torch.allclose(prefix, full[:2], atol=1.0e-6)

    prefix_ema = apply_weighted_ridge_residual(
        features[:2],
        state,
        shrink=1.0,
        max_abs_scale=0.0,
        x_ncl=x[:2],
        domain_align_channels=[0],
        domain_align_causal_half_life=2.0,
    )
    full_ema = apply_weighted_ridge_residual(
        features,
        state,
        shrink=1.0,
        max_abs_scale=0.0,
        x_ncl=x,
        domain_align_channels=[0],
        domain_align_causal_half_life=2.0,
    )

    assert torch.allclose(prefix_ema, full_ema[:2], atol=1.0e-6)


def test_causal_warmup_alignment_falls_back_then_freezes_prefix_stats() -> None:
    features = torch.tensor([[[1.0]], [[2.0]], [[4.0]], [[100.0]]])
    state = {
        "feature_mean_cf": torch.zeros(1, 1),
        "feature_std_cf": torch.ones(1, 1),
        "coef_cfh": torch.tensor([[[0.0], [1.0]]]),
    }
    x = torch.ones(4, 1, 2)

    first_three = apply_weighted_ridge_residual(
        features[:3],
        state,
        shrink=1.0,
        max_abs_scale=0.0,
        x_ncl=x[:3],
        domain_align_channels=[0],
        domain_align_causal_warmup_windows=2,
    )
    full = apply_weighted_ridge_residual(
        features,
        state,
        shrink=1.0,
        max_abs_scale=0.0,
        x_ncl=x,
        domain_align_channels=[0],
        domain_align_causal_warmup_windows=2,
    )

    assert torch.equal(full[:2], torch.zeros_like(full[:2]))
    assert torch.allclose(first_three, full[:3], atol=1.0e-6)


def test_online_refit_only_uses_fully_observed_delayed_labels() -> None:
    features = torch.arange(10, dtype=torch.float32).view(10, 1, 1)
    x = torch.ones(10, 1, 2)
    base = torch.zeros(10, 1, 2)
    target = features.expand(-1, -1, 2)

    correction, updates = online_refit_corrections(
        features_ncf=features,
        x_ncl=x,
        base_nch=base,
        target_nch=target,
        eval_start=5,
        eval_end=10,
        update_interval=2,
        label_delay=3,
        ridge=1.0,
        half_life=0.0,
        shrink=1.0,
        max_abs_scale=0.0,
        fit_loss="ridge",
        huber_delta=0.1,
        huber_iterations=1,
    )

    assert correction.shape == (5, 1, 2)
    assert [row["fit_end"] for row in updates] == [3, 5, 7]
    assert all(row["latest_label_window"] + 3 <= row["prediction_start"] for row in updates)


def test_fixed_expert_gate_features_are_patch_level_and_finite() -> None:
    torch.manual_seed(12)
    x = torch.randn(8, 2, 8)
    base = torch.randn(8, 2, 8)
    corrections = torch.randn(8, 2, 2, 8) * 0.1

    features = build_fixed_expert_gate_features(x, base, corrections, patch_len=2)
    within_only = build_fixed_expert_gate_features(
        x,
        base,
        corrections,
        patch_len=2,
        include_domain_descriptor=False,
    )

    assert features.shape[:3] == (8, 2, 4)
    assert within_only.shape[:-1] == features.shape[:-1]
    assert within_only.shape[-1] < features.shape[-1]
    assert torch.isfinite(features).all()


def test_fixed_expert_targets_prefer_meaningful_improvement_over_default() -> None:
    base = torch.ones(1, 1, 4)
    target = torch.zeros_like(base)
    raw = torch.tensor([[[ -1.0, -1.0, 0.0, 0.0 ]]])
    aligned = torch.tensor([[[ -0.5, -0.5, -0.5, -0.5 ]]])
    corrections = torch.stack([raw, aligned], dim=2)

    routes, _, _, _ = fixed_expert_gate_targets(
        base,
        corrections,
        target,
        patch_len=2,
        min_gain=0.01,
    )
    pred = apply_fixed_expert_routes(base, corrections, routes, patch_len=2)

    assert torch.equal(routes, torch.tensor([[[1, 2]]]))
    assert torch.equal(pred, torch.tensor([[[0.0, 0.0, 0.5, 0.5]]]))


def test_fixed_expert_gate_margin_falls_back_to_aligned_expert() -> None:
    logits = torch.tensor([[[[0.1, 1.0, 0.9], [2.0, 0.0, 0.5]]]])
    active = torch.tensor([True])

    permissive = _route_with_default_margin(logits, margin=0.0, active_mask_c=active)
    guarded = _route_with_default_margin(logits, margin=0.2, active_mask_c=active)

    assert torch.equal(permissive, torch.tensor([[[1, 0]]]))
    assert torch.equal(guarded, torch.tensor([[[2, 0]]]))

    two_channel_logits = torch.tensor([[[[3.0, 0.0, 1.0]], [[0.0, 3.0, 1.0]]]])
    routed = torch.tensor([False, True])
    correction_active = torch.tensor([True, True])
    subset_route = _route_with_default_margin(
        two_channel_logits,
        margin=0.0,
        active_mask_c=routed,
        default_active_mask_c=correction_active,
    )
    assert torch.equal(subset_route, torch.tensor([[[2], [1]]]))


def test_fixed_expert_participation_guard_abstains_on_weak_batch_support() -> None:
    route = torch.tensor([[[2, 2, 0, 2]], [[2, 2, 2, 2]]])
    guarded, summary = apply_route_participation_guard(
        route,
        routed_mask_c=torch.tensor([True]),
        min_nondefault_rate=0.2,
    )

    assert summary["observed_nondefault_rate"] == 0.125
    assert summary["abstained"] is True
    assert torch.equal(guarded, torch.full_like(route, 2))
