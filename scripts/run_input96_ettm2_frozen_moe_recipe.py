from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_input96_h96_targeted_tuning import Candidate, load_yaml, resolve, run_candidate  # noqa: E402
from scripts.run_input96_moe_positive_search import apply_moe_training_controls  # noqa: E402


FIELDS = [
    "variant",
    "status",
    "test_mse",
    "test_mae",
    "val_mse",
    "val_mae",
    "baseline_mse",
    "baseline_mae",
    "mse_gain_pct",
    "mae_gain_pct",
    "residual_num_channels",
    "residual_mean_scale",
    "config_path",
    "out_dir",
    "returncode",
    "error",
]


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def best_gate(max_scale: float = 2.0, init_scale: float = 0.5, scale_reg: float = 0.0001) -> dict[str, Any]:
    return {
        "loss": "mse",
        "selection_metric": "mse",
        "epochs": 30,
        "train_fraction": 0.7,
        "hidden_dim": 32,
        "batch_size": 256,
        "max_scale": float(max_scale),
        "init_scale": float(init_scale),
        "scale_reg": float(scale_reg),
        "scale_mode": "sigmoid",
        "standardize_features": True,
    }


def activation_gate(
    *,
    max_scale: float = 2.0,
    init_scale: float = 0.5,
    scale_reg: float = 0.0001,
    threshold_metric: str = "mse",
    threshold_scope: str = "channel",
    label_min_improvement: float = 0.0,
    bce_weight: float = 0.2,
    inactive_scale_weight: float = 0.05,
    rate_balance_weight: float = 0.0,
    activation_train_soft_gating: bool = False,
) -> dict[str, Any]:
    gate = best_gate(max_scale=max_scale, init_scale=init_scale, scale_reg=scale_reg)
    gate.update(
        {
            "activation_head_enable": True,
            "apply_activation_threshold": True,
            "activation_threshold": "auto",
            "activation_threshold_selection_metric": str(threshold_metric),
            "activation_threshold_scope": str(threshold_scope),
            "activation_label_min_improvement": float(label_min_improvement),
            "activation_bce_weight": float(bce_weight),
            "activation_inactive_scale_weight": float(inactive_scale_weight),
            "activation_pos_weight": "auto",
            "activation_pos_weight_scope": "channel",
            "activation_pos_weight_max": 3.0,
            "activation_rate_balance_weight": float(rate_balance_weight),
            "activation_rate_balance_scope": "cluster",
            "activation_train_soft_gating": bool(activation_train_soft_gating),
        }
    )
    return gate


def moe_patch(
    *,
    feature_mode: str,
    lambda_scale: float,
    dynamic_lambda: bool,
    gate_max_scale: float = 2.0,
    gate_init_scale: float = 0.5,
    gate_scale_reg: float = 0.0001,
    gate_calibrator: dict[str, Any] | None = None,
    alpha_scale: float = 1.5,
    residual_clip: float = 4.0,
) -> dict[str, Any]:
    lambdas = {"trend": float(lambda_scale), "direction": float(lambda_scale) * 2.0}
    return {
        "penalties": {"enabled": ["trend", "direction"]},
        "moe": {
            "enable": True,
            "topk": 1,
            "select_ranks": [1],
            "dynamic_lambda": {
                "enable": bool(dynamic_lambda),
                "mode": "multiscale",
                "hidden_dim": 32,
                "segment_bins": [4, 8],
                "max_factor": 1.5,
                "mix": 0.6,
                "dropout": 0.0,
                "reg_weight": 0.0001,
            },
            "lambda_init": dict(lambdas),
            "lambda_min": {name: 0.0 for name in lambdas},
            "lambda_schedule": {name: "none" for name in lambdas},
            "gate_temperature": 1.0,
            "gate_noise_std": 0.2,
            "skip_cost": 0.15,
            "pred_side_residual": {
                "enable": True,
                "feature_mode": str(feature_mode),
                "residual_clip": float(residual_clip),
                "corrector_hidden": 64,
                "init_alpha": -2.5,
                "alpha_scale": float(alpha_scale),
                "specialization_weight": 0.0,
                "norm_weight": 0.0,
                "intervention_weight": 0.0,
                "use_y_base_input": True,
                "intervention_enable": False,
                "detach_routed_penalty_pred": False,
                "selection_policy": "val_mse_gate_guarded",
                "selection_min_abs_improvement": 0.0,
                "selection_min_rel_improvement": 0.0005,
                "gate_calibrator": (
                    dict(gate_calibrator)
                    if gate_calibrator is not None
                    else best_gate(gate_max_scale, gate_init_scale, gate_scale_reg)
                ),
                "channel_expert_adapters": {"enable": True, "mode": "all", "mode_type": "delta"},
            },
        },
    }


def variants() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("fixed_current_channel_delta_no_extra", moe_patch(feature_mode="legacy", lambda_scale=0.07875, dynamic_lambda=False)),
        (
            "fixed_current_channel_delta_no_extra_safe_aug",
            moe_patch(feature_mode="safe_augmented", lambda_scale=0.07875, dynamic_lambda=False),
        ),
        ("zero_lambda_channel_delta_no_extra", moe_patch(feature_mode="legacy", lambda_scale=0.0, dynamic_lambda=False)),
        (
            "zero_lambda_channel_delta_no_extra_safe_aug",
            moe_patch(feature_mode="safe_augmented", lambda_scale=0.0, dynamic_lambda=False),
        ),
        (
            "fixed_current_channel_delta_no_extra_gate_ms15",
            moe_patch(
                feature_mode="legacy",
                lambda_scale=0.07875,
                dynamic_lambda=False,
                gate_max_scale=1.5,
                gate_init_scale=0.3,
                gate_scale_reg=0.0005,
            ),
        ),
        (
            "dynamic_current_channel_delta_no_extra",
            moe_patch(feature_mode="legacy", lambda_scale=0.07875, dynamic_lambda=True),
        ),
        (
            "zero_safe_act_mse_bce02",
            moe_patch(
                feature_mode="safe_augmented",
                lambda_scale=0.0,
                dynamic_lambda=False,
                gate_calibrator=activation_gate(threshold_metric="mse", bce_weight=0.2, inactive_scale_weight=0.05),
            ),
        ),
        (
            "zero_safe_act_mse_min1e3",
            moe_patch(
                feature_mode="safe_augmented",
                lambda_scale=0.0,
                dynamic_lambda=False,
                gate_calibrator=activation_gate(
                    threshold_metric="mse",
                    label_min_improvement=0.001,
                    bce_weight=0.2,
                    inactive_scale_weight=0.05,
                ),
            ),
        ),
        (
            "zero_safe_act_balacc_bce02",
            moe_patch(
                feature_mode="safe_augmented",
                lambda_scale=0.0,
                dynamic_lambda=False,
                gate_calibrator=activation_gate(
                    threshold_metric="balanced_accuracy",
                    bce_weight=0.2,
                    inactive_scale_weight=0.05,
                ),
            ),
        ),
        (
            "zero_safe_act_mse_bce05_inact20",
            moe_patch(
                feature_mode="safe_augmented",
                lambda_scale=0.0,
                dynamic_lambda=False,
                gate_calibrator=activation_gate(
                    threshold_metric="mse",
                    bce_weight=0.5,
                    inactive_scale_weight=0.2,
                    rate_balance_weight=0.02,
                ),
            ),
        ),
        (
            "zero_safe_act_mse_soft",
            moe_patch(
                feature_mode="safe_augmented",
                lambda_scale=0.0,
                dynamic_lambda=False,
                gate_calibrator=activation_gate(
                    threshold_metric="mse",
                    bce_weight=0.2,
                    inactive_scale_weight=0.05,
                    activation_train_soft_gating=True,
                ),
            ),
        ),
        (
            "zero_safe_gate_ms15",
            moe_patch(
                feature_mode="safe_augmented",
                lambda_scale=0.0,
                dynamic_lambda=False,
                gate_max_scale=1.5,
                gate_init_scale=0.3,
                gate_scale_reg=0.0005,
            ),
        ),
        (
            "zero_safe_gate_ms25",
            moe_patch(
                feature_mode="safe_augmented",
                lambda_scale=0.0,
                dynamic_lambda=False,
                gate_max_scale=2.5,
                gate_init_scale=0.5,
                gate_scale_reg=0.0001,
            ),
        ),
        (
            "zero_safe_gate_ms3_init08",
            moe_patch(
                feature_mode="safe_augmented",
                lambda_scale=0.0,
                dynamic_lambda=False,
                gate_max_scale=3.0,
                gate_init_scale=0.8,
                gate_scale_reg=0.00005,
            ),
        ),
        (
            "zero_safe_alpha2_clip6_gate_ms25",
            moe_patch(
                feature_mode="safe_augmented",
                lambda_scale=0.0,
                dynamic_lambda=False,
                gate_max_scale=2.5,
                gate_init_scale=0.5,
                gate_scale_reg=0.0001,
                alpha_scale=2.0,
                residual_clip=6.0,
            ),
        ),
        (
            "zero_safe_alpha1_clip2_gate_ms15",
            moe_patch(
                feature_mode="safe_augmented",
                lambda_scale=0.0,
                dynamic_lambda=False,
                gate_max_scale=1.5,
                gate_init_scale=0.3,
                gate_scale_reg=0.0005,
                alpha_scale=1.0,
                residual_clip=2.0,
            ),
        ),
    ]


def select_variants(
    items: list[tuple[str, dict[str, Any]]],
    requested: list[str] | None,
) -> list[tuple[str, dict[str, Any]]]:
    if not requested:
        return items
    by_name = {name: patch for name, patch in items}
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise ValueError(f"Unknown variants: {missing}. Supported: {sorted(by_name)}")
    return [(name, by_name[name]) for name in requested]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run frozen ETTm2 input-96 MoE recipes on a strong channel-head backbone.")
    ap.add_argument(
        "--base-config",
        default="outputs/fresh_input_len96_20260610_ettm2_backbone_lowdrop/configs/ETTm2/H96/common_backbone_h96/channel_h256_do0_wd1e3_mae06.yaml",
    )
    ap.add_argument(
        "--warm-start-checkpoint",
        default="outputs/fresh_input_len96_20260610_ettm2_backbone_lowdrop/runs/ETTm2/H96/common_backbone_h96/channel_h256_do0_wd1e3_mae06/best_checkpoint.pt",
    )
    ap.add_argument("--out-root", default="outputs/fresh_input_len96_20260610_ettm2_frozen_moe_recipe")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--variants", nargs="*", help="Optional exact variant names to run, in order.")
    ap.add_argument("--baseline-mse", type=float, default=0.1765178143978119)
    ap.add_argument("--baseline-mae", type=float, default=0.2583235800266266)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base_cfg = load_yaml(resolve(args.base_config))
    apply_moe_training_controls(
        base_cfg,
        warm_start_checkpoint=str(args.warm_start_checkpoint),
        freeze_backbone=True,
        lr=0.0005,
        weight_decay=1.0e-5,
    )

    out_root = resolve(args.out_root)
    selected = select_variants(variants(), args.variants)
    if int(args.limit) > 0:
        selected = selected[: int(args.limit)]
    result_path = out_root / "ettm2_frozen_moe_results.csv"
    rows = read_rows(result_path)
    done = {str(row.get("variant")) for row in rows if row.get("status") == "ok"}

    for name, patch in selected:
        if name in done:
            print(f"[skip] {name}", flush=True)
            continue
        row, _ = run_candidate(
            dataset="ETTm2",
            pred_len=96,
            base_cfg=base_cfg,
            cand=Candidate("ettm2_frozen_moe", name, json.loads(json.dumps(patch))),
            out_root=out_root,
            device=str(args.device),
            epochs=int(args.epochs),
            skip_test=False,
            dry_run=bool(args.dry_run),
        )
        test_mse = float(row["test_mse"]) if row.get("test_mse") not in (None, "") else float("nan")
        test_mae = float(row["test_mae"]) if row.get("test_mae") not in (None, "") else float("nan")
        row = {
            "variant": name,
            "status": row.get("status"),
            "test_mse": row.get("test_mse"),
            "test_mae": row.get("test_mae"),
            "val_mse": row.get("val_mse"),
            "val_mae": row.get("val_mae"),
            "baseline_mse": float(args.baseline_mse),
            "baseline_mae": float(args.baseline_mae),
            "mse_gain_pct": 100.0 * (float(args.baseline_mse) - test_mse) / max(abs(float(args.baseline_mse)), 1.0e-12),
            "mae_gain_pct": 100.0 * (float(args.baseline_mae) - test_mae) / max(abs(float(args.baseline_mae)), 1.0e-12),
            "residual_num_channels": row.get("residual_num_channels"),
            "residual_mean_scale": row.get("residual_mean_scale"),
            "config_path": row.get("config_path"),
            "out_dir": row.get("out_dir"),
            "returncode": row.get("returncode"),
            "error": row.get("error"),
        }
        rows.append(row)
        write_rows(result_path, rows)
        print(
            f"{name} status={row['status']} test={row['test_mse']} "
            f"gain={row['mse_gain_pct']:.3f}%",
            flush=True,
        )


if __name__ == "__main__":
    main()
