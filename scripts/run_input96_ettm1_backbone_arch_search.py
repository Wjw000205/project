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


BASE_PATCH: dict[str, Any] = {
    "train": {
        "selection_metric": "val_mse",
        "lr_scheduler": {"name": "plateau", "factor": 0.5, "patience": 3, "min_lr": 1.0e-6},
    }
}


VARIANTS: dict[str, dict[str, Any]] = {
    "attn_h192_do01_wd5e4_mae04": {
        "model": {"predictor": "attn_mlp", "hidden_dim": 192, "dropout": 0.1, "attn_dim": 64},
        "train": {"weight_decay": 5.0e-4, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
    },
    "attn_h384_do01_wd1e3_mae04": {
        "model": {"predictor": "attn_mlp", "hidden_dim": 384, "dropout": 0.1, "attn_dim": 96},
        "train": {"weight_decay": 1.0e-3, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
    },
    "lcch_h192_do01_wd1e3_mae04": {
        "model": {
            "predictor": "long_context_channel_head_mlp",
            "hidden_dim": 192,
            "dropout": 0.1,
            "predictor_input_len": 96,
            "long_context_channel_head_residual": True,
            "long_context_include_seasonal_profile": False,
            "long_context_output_mode": "direct",
        },
        "train": {"weight_decay": 1.0e-3, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
    },
    "lcanch_h192_do01_wd1e3_mae04": {
        "model": {
            "predictor": "long_context_anchor_channel_head_mlp",
            "hidden_dim": 192,
            "dropout": 0.1,
            "predictor_input_len": 96,
            "long_context_channel_head_residual": True,
            "long_context_include_seasonal_profile": False,
            "anchor_detail_scale": 0.35,
        },
        "train": {"weight_decay": 1.0e-3, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
    },
    "seasgate_h128_do01_wd1e3_mae04": {
        "model": {
            "predictor": "seasonality_gated_channel_head_mlp",
            "hidden_dim": 128,
            "dropout": 0.1,
            "predictor_input_len": 96,
            "seasonal_hybrid_residual": True,
            "long_context_include_seasonal_profile": False,
            "seasonal_mix_init": -2.0,
            "seasonal_gate_strength": 0.0,
            "anchor_detail_scale": 0.25,
        },
        "train": {"weight_decay": 1.0e-3, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
    },
    "channel_dlinear_k25_wd1e4_mae04": {
        "model": {"predictor": "channel_dlinear", "hidden_dim": 64, "dropout": 0.0, "dlinear_kernel_size": 25},
        "train": {"weight_decay": 1.0e-4, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
    },
    "patchtst_h128_p16s8_l2_do01_wd1e4_mae04": {
        "model": {
            "predictor": "patchtst",
            "hidden_dim": 128,
            "dropout": 0.1,
            "patch_d_model": 128,
            "patch_len": 16,
            "patch_stride": 8,
            "patch_num_layers": 2,
            "patch_num_heads": 4,
            "patch_ff_dim": 256,
        },
        "train": {"weight_decay": 1.0e-4, "mae_objective": {"enable": True, "kind": "l1", "weight": 0.4}},
    },
}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def make_candidate(name: str, *, save_checkpoint: bool) -> Candidate:
    patch = copy.deepcopy(BASE_PATCH)
    deep_update(patch, copy.deepcopy(VARIANTS[name]))
    deep_update(patch, {"memory": {"save_checkpoint": bool(save_checkpoint)}})
    return Candidate("backbone_arch", name, patch)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ETTm1 input-96 backbone architecture candidates.")
    parser.add_argument("--out-root", default="outputs/fresh_input_len96_20260610_ettm1_backbone_arch_search")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
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
    rows: list[dict[str, Any]] = []
    rows_path = out_root / "arch_rows.csv"

    for name in args.variants:
        cand = make_candidate(name, save_checkpoint=bool(args.save_checkpoint))
        print(f"=== ETTm1 H96 backbone_arch {name} ===", flush=True)
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
            f"test={row.get('test_mse')} sec={row.get('total_sec')}",
            flush=True,
        )

    ok = [row for row in rows if row.get("status") == "ok" and row.get("val_mse") != ""]
    if ok:
        best = min(ok, key=lambda row: (value(row, "val_mse"), value(row, "test_mse")))
        print(f"BEST_VAL {best.get('variant')} val={best.get('val_mse')} test={best.get('test_mse')}", flush=True)
    print(f"ROWS {rows_path}", flush=True)


if __name__ == "__main__":
    main()
