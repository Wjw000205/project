from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path
from typing import Any

from run_input96_h96_targeted_tuning import (
    Candidate,
    DATASET_CONFIGS,
    FIELDS,
    deep_update,
    load_yaml,
    resolve,
    run_candidate,
    set_moe_off,
    value,
)


def train_patch(*, weight_decay: float = 1.0e-4, mae_weight: float = 0.6, lr: float = 1.0e-3) -> dict[str, Any]:
    return {
        "train": {
            "lr": float(lr),
            "selection_metric": "val_mse",
            "weight_decay": float(weight_decay),
            "mae_objective": {"enable": True, "kind": "l1", "weight": float(mae_weight), "warmup_epochs": 5},
            "lr_scheduler": {"name": "plateau", "factor": 0.5, "patience": 3, "min_lr": 1.0e-6},
        },
        "early_stop": {"patience": 10, "min_delta": 1.0e-6},
    }


def make_patch(model: dict[str, Any], **train_kwargs: Any) -> dict[str, Any]:
    patch = train_patch(**train_kwargs)
    deep_update(patch, {"model": model})
    return patch


VARIANTS: dict[str, dict[str, Any]] = {
    "mlp_h320_do02_wd1e4_mae06": make_patch(
        {"predictor": "mlp", "hidden_dim": 320, "dropout": 0.2},
    ),
    "mlp_revin_h256_do02_wd1e4_mae06": make_patch(
        {"predictor": "mlp", "hidden_dim": 256, "dropout": 0.2, "revin": True},
    ),
    "mlp_anchor_h256_do02_wd1e4_mae06": make_patch(
        {"predictor": "mlp", "hidden_dim": 256, "dropout": 0.2, "mlp_residual_anchor": True},
    ),
    "mlp_anchor_h320_do02_wd1e4_mae06": make_patch(
        {"predictor": "mlp", "hidden_dim": 320, "dropout": 0.2, "mlp_residual_anchor": True},
    ),
    "mlp_anchor_basis_h256_r16_wd1e4_mae06": make_patch(
        {
            "predictor": "mlp",
            "hidden_dim": 256,
            "dropout": 0.2,
            "mlp_residual_anchor": True,
            "temporal_basis_adapter": {"enable": True, "rank": 16, "scale": 0.15, "init": "zero_delta"},
        },
    ),
    "mlp_anchor_basis_h256_r8_s010_wd1e4_mae06_lr5e4": make_patch(
        {
            "predictor": "mlp",
            "hidden_dim": 256,
            "dropout": 0.2,
            "mlp_residual_anchor": True,
            "temporal_basis_adapter": {"enable": True, "rank": 8, "scale": 0.10, "init": "zero_delta"},
        },
        lr=5.0e-4,
    ),
    "mlp_anchor_basis_h256_r16_s010_wd5e5_mae06_lr5e4": make_patch(
        {
            "predictor": "mlp",
            "hidden_dim": 256,
            "dropout": 0.2,
            "mlp_residual_anchor": True,
            "temporal_basis_adapter": {"enable": True, "rank": 16, "scale": 0.10, "init": "zero_delta"},
        },
        lr=5.0e-4,
        weight_decay=5.0e-5,
    ),
    "mlp_anchor_basis_h320_r16_s010_wd1e4_mae06_lr5e4": make_patch(
        {
            "predictor": "mlp",
            "hidden_dim": 320,
            "dropout": 0.2,
            "mlp_residual_anchor": True,
            "temporal_basis_adapter": {"enable": True, "rank": 16, "scale": 0.10, "init": "zero_delta"},
        },
        lr=5.0e-4,
    ),
    "mlp_anchor_basis_h256_r32_s005_wd1e4_mae06_lr5e4": make_patch(
        {
            "predictor": "mlp",
            "hidden_dim": 256,
            "dropout": 0.2,
            "mlp_residual_anchor": True,
            "temporal_basis_adapter": {"enable": True, "rank": 32, "scale": 0.05, "init": "zero_delta"},
        },
        lr=5.0e-4,
    ),
    "mlp_seasonal_resid_h256_do02_wd1e4_mae06": make_patch(
        {
            "predictor": "mlp",
            "hidden_dim": 256,
            "dropout": 0.2,
            "seasonal_residual": True,
            "seasonal_period": 96,
            "seasonal_num_periods": 1,
        },
    ),
    "mlp_seasonal_anchor_h256_do02_wd1e4_mae06": make_patch(
        {
            "predictor": "mlp",
            "hidden_dim": 256,
            "dropout": 0.2,
            "seasonal_anchor": True,
            "seasonal_anchor_period": 96,
            "seasonal_anchor_num_periods": 1,
            "seasonal_anchor_delta_scale": 1.0,
        },
    ),
    "mlp_seasonal_anchor_basis_h256_r16_wd1e4_mae06_lr5e4": make_patch(
        {
            "predictor": "mlp",
            "hidden_dim": 256,
            "dropout": 0.2,
            "mlp_residual_anchor": True,
            "seasonal_anchor": True,
            "seasonal_anchor_period": 96,
            "seasonal_anchor_num_periods": 1,
            "seasonal_anchor_delta_scale": 1.0,
            "temporal_basis_adapter": {"enable": True, "rank": 16, "scale": 0.10, "init": "zero_delta"},
        },
        lr=5.0e-4,
    ),
    "mlp_anchor_seasonal_s025_basis_h256_r16_wd1e4_mae06": make_patch(
        {
            "predictor": "mlp",
            "hidden_dim": 256,
            "dropout": 0.2,
            "mlp_residual_anchor": True,
            "seasonal_anchor": True,
            "seasonal_anchor_period": 96,
            "seasonal_anchor_num_periods": 1,
            "seasonal_anchor_delta_scale": 0.25,
            "temporal_basis_adapter": {"enable": True, "rank": 16, "scale": 0.15, "init": "zero_delta"},
        },
    ),
    "mlp_anchor_seasonal_s050_basis_h256_r16_wd1e4_mae06": make_patch(
        {
            "predictor": "mlp",
            "hidden_dim": 256,
            "dropout": 0.2,
            "mlp_residual_anchor": True,
            "seasonal_anchor": True,
            "seasonal_anchor_period": 96,
            "seasonal_anchor_num_periods": 1,
            "seasonal_anchor_delta_scale": 0.50,
            "temporal_basis_adapter": {"enable": True, "rank": 16, "scale": 0.15, "init": "zero_delta"},
        },
    ),
    "mlp_anchor_seasonal_s075_basis_h256_r16_wd1e4_mae06": make_patch(
        {
            "predictor": "mlp",
            "hidden_dim": 256,
            "dropout": 0.2,
            "mlp_residual_anchor": True,
            "seasonal_anchor": True,
            "seasonal_anchor_period": 96,
            "seasonal_anchor_num_periods": 1,
            "seasonal_anchor_delta_scale": 0.75,
            "temporal_basis_adapter": {"enable": True, "rank": 16, "scale": 0.15, "init": "zero_delta"},
        },
    ),
    "mlp_anchor_basis_seasblend_m025_h256_r16_wd1e4_mae06": make_patch(
        {
            "predictor": "mlp",
            "hidden_dim": 256,
            "dropout": 0.2,
            "mlp_residual_anchor": True,
            "temporal_basis_adapter": {"enable": True, "rank": 16, "scale": 0.15, "init": "zero_delta"},
            "seasonal_blend_adapter": {
                "enable": True,
                "period": 96,
                "num_periods": 1,
                "max_mix": 0.25,
                "init_mix": 0.02,
            },
        },
    ),
    "mlp_anchor_basis_seasblend_m010_h256_r16_wd1e4_mae06": make_patch(
        {
            "predictor": "mlp",
            "hidden_dim": 256,
            "dropout": 0.2,
            "mlp_residual_anchor": True,
            "temporal_basis_adapter": {"enable": True, "rank": 16, "scale": 0.15, "init": "zero_delta"},
            "seasonal_blend_adapter": {
                "enable": True,
                "period": 96,
                "num_periods": 1,
                "max_mix": 0.10,
                "init_mix": 0.005,
            },
        },
    ),
    "mlp_anchor_basis_seasblend_m050_h256_r16_wd1e4_mae06": make_patch(
        {
            "predictor": "mlp",
            "hidden_dim": 256,
            "dropout": 0.2,
            "mlp_residual_anchor": True,
            "temporal_basis_adapter": {"enable": True, "rank": 16, "scale": 0.15, "init": "zero_delta"},
            "seasonal_blend_adapter": {
                "enable": True,
                "period": 96,
                "num_periods": 1,
                "max_mix": 0.50,
                "init_mix": 0.02,
            },
        },
    ),
    "nlinear_wd1e4_mae06": make_patch(
        {"predictor": "nlinear", "hidden_dim": 64, "dropout": 0.0},
    ),
    "dlinear_k25_wd1e4_mae06": make_patch(
        {"predictor": "dlinear", "hidden_dim": 64, "dropout": 0.0, "dlinear_kernel_size": 25},
    ),
    "dlinear_revin_k25_wd1e4_mae06": make_patch(
        {"predictor": "dlinear", "hidden_dim": 64, "dropout": 0.0, "dlinear_kernel_size": 25, "revin": True},
    ),
    "gru_h64_l1_do0_wd1e4_mae06": make_patch(
        {"predictor": "gru", "hidden_dim": 64, "dropout": 0.0, "gru_num_layers": 1, "revin": True},
    ),
    "gru_h128_l1_do0_wd1e4_mae06": make_patch(
        {"predictor": "gru", "hidden_dim": 128, "dropout": 0.0, "gru_num_layers": 1, "revin": True},
    ),
    "gru_h128_l1_do01_wd1e4_mae06_lr5e4": make_patch(
        {"predictor": "gru", "hidden_dim": 128, "dropout": 0.1, "gru_num_layers": 1, "revin": True},
        lr=5.0e-4,
    ),
    "lstm_h64_l1_do0_wd1e4_mae06": make_patch(
        {"predictor": "lstm", "hidden_dim": 64, "dropout": 0.0, "lstm_num_layers": 1, "revin": True},
    ),
    "lstm_h128_l1_do0_wd1e4_mae06": make_patch(
        {"predictor": "lstm", "hidden_dim": 128, "dropout": 0.0, "lstm_num_layers": 1, "revin": True},
    ),
    "lstm_h128_l1_do01_wd1e4_mae06_lr5e4": make_patch(
        {"predictor": "lstm", "hidden_dim": 128, "dropout": 0.1, "lstm_num_layers": 1, "revin": True},
        lr=5.0e-4,
    ),
    "tcn_h16_l1_k3_do0_wd1e4_mae06_lr5e4": make_patch(
        {
            "predictor": "tcn",
            "hidden_dim": 16,
            "dropout": 0.0,
            "tcn_levels": 1,
            "tcn_kernel_size": 3,
            "tcn_dilation_base": 2,
            "revin": True,
        },
        lr=5.0e-4,
    ),
    "tcn_h32_l1_k5_do01_wd1e4_mae06_lr5e4": make_patch(
        {
            "predictor": "tcn",
            "hidden_dim": 32,
            "dropout": 0.1,
            "tcn_levels": 1,
            "tcn_kernel_size": 5,
            "tcn_dilation_base": 2,
            "revin": True,
        },
        lr=5.0e-4,
    ),
    "tcn_h32_l2_k3_do0_wd1e4_mae06": make_patch(
        {
            "predictor": "tcn",
            "hidden_dim": 32,
            "dropout": 0.0,
            "tcn_levels": 2,
            "tcn_kernel_size": 3,
            "tcn_dilation_base": 2,
            "revin": True,
        },
    ),
    "tcn_h64_l2_k3_do0_wd1e4_mae06": make_patch(
        {
            "predictor": "tcn",
            "hidden_dim": 64,
            "dropout": 0.0,
            "tcn_levels": 2,
            "tcn_kernel_size": 3,
            "tcn_dilation_base": 2,
            "revin": True,
        },
    ),
}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser(description="Search lightweight general input-96 H96 backbones.")
    parser.add_argument("--out-root", default="outputs/fresh_input_len96_light_backbone_search")
    parser.add_argument("--datasets", nargs="+", default=["ETTh1", "ETTh2", "ETTm1", "ETTm2"], choices=list(DATASET_CONFIGS))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    missing = [name for name in args.variants if name not in VARIANTS]
    if missing:
        raise SystemExit(f"Unknown variants: {missing}. Supported: {sorted(VARIANTS)}")

    out_root = resolve(args.out_root)
    rows_path = out_root / "light_backbone_rows.csv"
    rows: list[dict[str, Any]] = []

    for variant in args.variants:
        print(f"=== light_backbone {variant} ===", flush=True)
        for dataset in args.datasets:
            base_cfg = load_yaml(resolve(DATASET_CONFIGS[dataset]))
            set_moe_off(base_cfg)
            patch = copy.deepcopy(VARIANTS[variant])
            deep_update(patch, {"memory": {"save_checkpoint": bool(args.save_checkpoint)}})
            cand = Candidate("light_backbone", variant, patch)
            row, _ = run_candidate(
                dataset=dataset,
                pred_len=96,
                base_cfg=base_cfg,
                cand=cand,
                out_root=out_root,
                device=args.device,
                epochs=args.epochs,
                skip_test=bool(args.skip_test),
                dry_run=bool(args.dry_run),
            )
            rows.append(row)
            write_rows(rows_path, rows)
            print(
                f"[{variant} {dataset}] {row.get('status')} val={row.get('val_mse')} "
                f"test={row.get('test_mse')} mae={row.get('test_mae')} sec={row.get('total_sec')}",
                flush=True,
            )

    ok = [row for row in rows if row.get("status") == "ok" and row.get("val_mse") != ""]
    if ok:
        best_by_dataset: dict[str, dict[str, Any]] = {}
        for row in ok:
            dataset = str(row["dataset"])
            if dataset not in best_by_dataset or (value(row, "val_mse"), value(row, "val_mae")) < (
                value(best_by_dataset[dataset], "val_mse"),
                value(best_by_dataset[dataset], "val_mae"),
            ):
                best_by_dataset[dataset] = row
        for dataset, row in sorted(best_by_dataset.items()):
            print(
                f"BEST_VAL {dataset} {row.get('variant')} val={row.get('val_mse')} "
                f"test={row.get('test_mse')}",
                flush=True,
            )
    print(f"ROWS {rows_path}", flush=True)


if __name__ == "__main__":
    main()
