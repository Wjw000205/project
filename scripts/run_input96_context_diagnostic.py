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


def make_patch(model: dict[str, Any], *, input_len: int = 336, **train_kwargs: Any) -> dict[str, Any]:
    patch = train_patch(**train_kwargs)
    deep_update(
        patch,
        {
            "window": {"input_len": int(input_len), "past_context": True},
            "model": model,
        },
    )
    return patch


VARIANTS: dict[str, dict[str, Any]] = {
    "in336_tail96_anchor_basis": make_patch(
        {
            "predictor": "mlp",
            "predictor_input_len": 96,
            "hidden_dim": 256,
            "dropout": 0.2,
            "mlp_residual_anchor": True,
            "temporal_basis_adapter": {"enable": True, "rank": 16, "scale": 0.15, "init": "zero_delta"},
        },
    ),
    "in336_tail96_seasblend_m010_np3": make_patch(
        {
            "predictor": "mlp",
            "predictor_input_len": 96,
            "hidden_dim": 256,
            "dropout": 0.2,
            "mlp_residual_anchor": True,
            "temporal_basis_adapter": {"enable": True, "rank": 16, "scale": 0.15, "init": "zero_delta"},
            "seasonal_blend_adapter": {
                "enable": True,
                "period": 96,
                "num_periods": 3,
                "max_mix": 0.10,
                "init_mix": 0.005,
            },
        },
    ),
    "in336_tail96_seasblend_m025_np3": make_patch(
        {
            "predictor": "mlp",
            "predictor_input_len": 96,
            "hidden_dim": 256,
            "dropout": 0.2,
            "mlp_residual_anchor": True,
            "temporal_basis_adapter": {"enable": True, "rank": 16, "scale": 0.15, "init": "zero_delta"},
            "seasonal_blend_adapter": {
                "enable": True,
                "period": 96,
                "num_periods": 3,
                "max_mix": 0.25,
                "init_mix": 0.02,
            },
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
    parser = argparse.ArgumentParser(description="Diagnose whether input-336 context helps a tail-96 MLP backbone.")
    parser.add_argument("--out-root", default="outputs/fresh_input_len96_20260610_context_diagnostic")
    parser.add_argument("--dataset", default="ETTm1", choices=list(DATASET_CONFIGS))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    missing = [name for name in args.variants if name not in VARIANTS]
    if missing:
        raise SystemExit(f"Unknown variants: {missing}. Supported: {sorted(VARIANTS)}")

    out_root = resolve(args.out_root)
    base_cfg = load_yaml(resolve(DATASET_CONFIGS[args.dataset]))
    set_moe_off(base_cfg)
    rows_path = out_root / "context_diagnostic_rows.csv"
    rows: list[dict[str, Any]] = []

    for name in args.variants:
        print(f"=== {args.dataset} H96 context_diag {name} ===", flush=True)
        row, _ = run_candidate(
            dataset=args.dataset,
            pred_len=96,
            base_cfg=copy.deepcopy(base_cfg),
            cand=Candidate("context_diag", name, copy.deepcopy(VARIANTS[name])),
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
        print(f"BEST_VAL {best_val.get('variant')} val={best_val.get('val_mse')} test={best_val.get('test_mse')}", flush=True)
    print(f"ROWS {rows_path}", flush=True)


if __name__ == "__main__":
    main()
