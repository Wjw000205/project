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


def mlp_patch(
    *,
    hidden_dim: int,
    dropout: float,
    weight_decay: float,
    mae_weight: float,
    lr: float = 1.0e-3,
    batch_size: int = 64,
) -> dict[str, Any]:
    return {
        "model": {
            "predictor": "mlp",
            "hidden_dim": int(hidden_dim),
            "dropout": float(dropout),
        },
        "train": {
            "batch_size": int(batch_size),
            "lr": float(lr),
            "selection_metric": "val_mse",
            "weight_decay": float(weight_decay),
            "mae_objective": {"enable": True, "kind": "l1", "weight": float(mae_weight), "warmup_epochs": 5},
            "lr_scheduler": {"name": "plateau", "factor": 0.5, "patience": 3, "min_lr": 1.0e-6},
        },
        "early_stop": {"patience": 10, "min_delta": 1.0e-6},
    }


VARIANTS: dict[str, dict[str, Any]] = {
    "mlp_h256_do02_wd1e4_mae06": mlp_patch(hidden_dim=256, dropout=0.2, weight_decay=1.0e-4, mae_weight=0.6),
    "mlp_h256_do01_wd1e4_mae06": mlp_patch(hidden_dim=256, dropout=0.1, weight_decay=1.0e-4, mae_weight=0.6),
    "mlp_h256_do03_wd1e4_mae06": mlp_patch(hidden_dim=256, dropout=0.3, weight_decay=1.0e-4, mae_weight=0.6),
    "mlp_h256_do02_wd5e5_mae06": mlp_patch(hidden_dim=256, dropout=0.2, weight_decay=5.0e-5, mae_weight=0.6),
    "mlp_h256_do02_wd5e4_mae06": mlp_patch(hidden_dim=256, dropout=0.2, weight_decay=5.0e-4, mae_weight=0.6),
    "mlp_h256_do02_wd1e4_mae04": mlp_patch(hidden_dim=256, dropout=0.2, weight_decay=1.0e-4, mae_weight=0.4),
    "mlp_h256_do02_wd1e4_mae08": mlp_patch(hidden_dim=256, dropout=0.2, weight_decay=1.0e-4, mae_weight=0.8),
    "mlp_h192_do02_wd1e4_mae06": mlp_patch(hidden_dim=192, dropout=0.2, weight_decay=1.0e-4, mae_weight=0.6),
    "mlp_h320_do02_wd1e4_mae06": mlp_patch(hidden_dim=320, dropout=0.2, weight_decay=1.0e-4, mae_weight=0.6),
    "mlp_h384_do02_wd1e4_mae06": mlp_patch(hidden_dim=384, dropout=0.2, weight_decay=1.0e-4, mae_weight=0.6),
    "mlp_h256_do02_wd1e4_mae06_lr5e4": mlp_patch(
        hidden_dim=256, dropout=0.2, weight_decay=1.0e-4, mae_weight=0.6, lr=5.0e-4
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
    parser = argparse.ArgumentParser(description="Search ETTm1 input-96 MLP-only backbone settings.")
    parser.add_argument("--out-root", default="outputs/fresh_input_len96_20260610_ettm1_mlp_search")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--epochs", type=int, default=80)
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
    rows_path = out_root / "mlp_rows.csv"
    rows: list[dict[str, Any]] = []

    for name in args.variants:
        patch = copy.deepcopy(VARIANTS[name])
        deep_update(patch, {"memory": {"save_checkpoint": bool(args.save_checkpoint)}})
        cand = Candidate("mlp_backbone", name, patch)
        print(f"=== ETTm1 H96 mlp_backbone {name} ===", flush=True)
        row, _ = run_candidate(
            dataset="ETTm1",
            pred_len=96,
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
