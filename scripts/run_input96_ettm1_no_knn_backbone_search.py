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


def train_patch(
    *,
    lr: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
    mae_weight: float = 0.6,
    batch_size: int = 64,
    patience: int = 10,
) -> dict[str, Any]:
    return {
        "train": {
            "batch_size": int(batch_size),
            "lr": float(lr),
            "selection_metric": "val_mse",
            "weight_decay": float(weight_decay),
            "mae_objective": {
                "enable": float(mae_weight) > 0.0,
                "kind": "l1",
                "weight": float(mae_weight),
                "warmup_epochs": 5,
            },
            "lr_scheduler": {"name": "plateau", "factor": 0.5, "patience": 3, "min_lr": 1.0e-6},
        },
        "early_stop": {"patience": int(patience), "min_delta": 1.0e-6},
    }


def patch(model: dict[str, Any], **train_kwargs: Any) -> dict[str, Any]:
    out = train_patch(**train_kwargs)
    deep_update(out, {"model": model})
    return out


def mlp_anchor_basis(
    *,
    hidden_dim: int = 256,
    dropout: float = 0.2,
    basis_rank: int = 16,
    basis_scale: float = 0.15,
    seasonal_mix: float = 0.1,
    seasonal_init: float = 0.005,
) -> dict[str, Any]:
    return {
        "predictor": "mlp",
        "hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
        "mlp_residual_anchor": True,
        "temporal_basis_adapter": {
            "enable": True,
            "rank": int(basis_rank),
            "scale": float(basis_scale),
            "init": "zero_delta",
        },
        "seasonal_blend_adapter": {
            "enable": True,
            "period": 96,
            "num_periods": 1,
            "max_mix": float(seasonal_mix),
            "init_mix": float(seasonal_init),
        },
    }


def patchtst_model(
    *,
    d_model: int,
    patch_len: int,
    patch_stride: int,
    num_layers: int,
    num_heads: int,
    ff_dim: int,
    dropout: float,
    basis_rank: int = 0,
    seasonal_mix: float = 0.0,
) -> dict[str, Any]:
    model: dict[str, Any] = {
        "predictor": "patchtst",
        "hidden_dim": int(d_model),
        "dropout": float(dropout),
        "patch_d_model": int(d_model),
        "patch_len": int(patch_len),
        "patch_stride": int(patch_stride),
        "patch_num_layers": int(num_layers),
        "patch_num_heads": int(num_heads),
        "patch_ff_dim": int(ff_dim),
    }
    if int(basis_rank) > 0:
        model["temporal_basis_adapter"] = {
            "enable": True,
            "rank": int(basis_rank),
            "scale": 0.08,
            "init": "zero_delta",
        }
    if float(seasonal_mix) > 0.0:
        model["seasonal_blend_adapter"] = {
            "enable": True,
            "period": 96,
            "num_periods": 1,
            "max_mix": float(seasonal_mix),
            "init_mix": min(float(seasonal_mix) * 0.1, 0.01),
        }
    return model


VARIANTS: dict[str, dict[str, Any]] = {
    "mlp_ab_m010_h256_r16_s015_do02_wd1e4_mae06": patch(
        mlp_anchor_basis(hidden_dim=256, dropout=0.2, basis_rank=16, basis_scale=0.15, seasonal_mix=0.10),
        weight_decay=1.0e-4,
        mae_weight=0.6,
    ),
    "mlp_ab_m015_h256_r16_s015_do02_wd1e4_mae05": patch(
        mlp_anchor_basis(hidden_dim=256, dropout=0.2, basis_rank=16, basis_scale=0.15, seasonal_mix=0.15),
        weight_decay=1.0e-4,
        mae_weight=0.5,
    ),
    "mlp_ab_m010_h320_r24_s010_do015_wd5e5_mae05_lr7e4": patch(
        mlp_anchor_basis(hidden_dim=320, dropout=0.15, basis_rank=24, basis_scale=0.10, seasonal_mix=0.10),
        lr=7.0e-4,
        weight_decay=5.0e-5,
        mae_weight=0.5,
    ),
    "mlp_ab_m006_h256_r16_s020_do02_wd1e4_mae06": patch(
        mlp_anchor_basis(hidden_dim=256, dropout=0.2, basis_rank=16, basis_scale=0.20, seasonal_mix=0.06, seasonal_init=0.003),
        weight_decay=1.0e-4,
        mae_weight=0.6,
    ),
    "patchtst_d96_p8s4_l2_do005_wd5e5_mae04": patch(
        patchtst_model(
            d_model=96,
            patch_len=8,
            patch_stride=4,
            num_layers=2,
            num_heads=4,
            ff_dim=192,
            dropout=0.05,
        ),
        lr=8.0e-4,
        weight_decay=5.0e-5,
        mae_weight=0.4,
    ),
    "patchtst_d128_p16s8_l2_do01_wd1e4_mae04": patch(
        patchtst_model(
            d_model=128,
            patch_len=16,
            patch_stride=8,
            num_layers=2,
            num_heads=4,
            ff_dim=256,
            dropout=0.1,
        ),
        weight_decay=1.0e-4,
        mae_weight=0.4,
    ),
    "patchtst_d128_p8s4_l1_do005_wd5e5_mae04_seas005": patch(
        patchtst_model(
            d_model=128,
            patch_len=8,
            patch_stride=4,
            num_layers=1,
            num_heads=4,
            ff_dim=256,
            dropout=0.05,
            seasonal_mix=0.05,
        ),
        lr=8.0e-4,
        weight_decay=5.0e-5,
        mae_weight=0.4,
    ),
    "patchtst_d128_p16s8_l2_do01_wd5e5_mae04_basis8_seas005": patch(
        patchtst_model(
            d_model=128,
            patch_len=16,
            patch_stride=8,
            num_layers=2,
            num_heads=4,
            ff_dim=256,
            dropout=0.1,
            basis_rank=8,
            seasonal_mix=0.05,
        ),
        lr=8.0e-4,
        weight_decay=5.0e-5,
        mae_weight=0.4,
    ),
    "channel_dlinear_k25_seas005_wd5e5_mae04": patch(
        {
            "predictor": "channel_dlinear",
            "hidden_dim": 64,
            "dropout": 0.0,
            "dlinear_kernel_size": 25,
            "seasonal_blend_adapter": {
                "enable": True,
                "period": 96,
                "num_periods": 1,
                "max_mix": 0.05,
                "init_mix": 0.005,
            },
        },
        lr=8.0e-4,
        weight_decay=5.0e-5,
        mae_weight=0.4,
    ),
    "tcn_h48_l2_k5_revin_seas005_wd5e5_mae04": patch(
        {
            "predictor": "tcn",
            "hidden_dim": 48,
            "dropout": 0.05,
            "tcn_levels": 2,
            "tcn_kernel_size": 5,
            "tcn_dilation_base": 2,
            "revin": True,
            "seasonal_blend_adapter": {
                "enable": True,
                "period": 96,
                "num_periods": 1,
                "max_mix": 0.05,
                "init_mix": 0.005,
            },
        },
        lr=8.0e-4,
        weight_decay=5.0e-5,
        mae_weight=0.4,
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
    parser = argparse.ArgumentParser(description="Narrow ETTm1 input-96 backbone search with KNN disabled.")
    parser.add_argument("--out-root", default="outputs/fresh_input_len96_20260610_ettm1_no_knn_backbone_search")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    missing = [name for name in args.variants if name not in VARIANTS]
    if missing:
        raise SystemExit(f"Unknown variants: {missing}. Supported: {sorted(VARIANTS)}")

    out_root = resolve(args.out_root)
    base_cfg = load_yaml(resolve(DATASET_CONFIGS["ETTm1"]))
    set_moe_off(base_cfg)
    rows_path = out_root / "backbone_rows.csv"
    rows: list[dict[str, Any]] = []

    for name in args.variants:
        run_patch = copy.deepcopy(VARIANTS[name])
        deep_update(run_patch, {"memory": {"save_checkpoint": bool(args.save_checkpoint)}})
        cand = Candidate("no_knn_backbone", name, run_patch)
        print(f"=== ETTm1 H{args.horizon} no_knn_backbone {name} ===", flush=True)
        row, _ = run_candidate(
            dataset="ETTm1",
            pred_len=int(args.horizon),
            base_cfg=copy.deepcopy(base_cfg),
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
            f"[{name}] {row.get('status')} val={row.get('val_mse')} "
            f"test={row.get('test_mse')} mae={row.get('test_mae')} sec={row.get('total_sec')}",
            flush=True,
        )

    ok = [row for row in rows if row.get("status") == "ok" and row.get("val_mse") != ""]
    if ok:
        best_val = min(ok, key=lambda row: (value(row, "val_mse"), value(row, "val_mae")))
        best_test = min(ok, key=lambda row: (value(row, "test_mse"), value(row, "test_mae")))
        print(f"BEST_VAL {best_val.get('variant')} val={best_val.get('val_mse')} test={best_val.get('test_mse')}", flush=True)
        print(f"BEST_TEST {best_test.get('variant')} val={best_test.get('val_mse')} test={best_test.get('test_mse')}", flush=True)
    print(f"ROWS {rows_path}", flush=True)


if __name__ == "__main__":
    main()
