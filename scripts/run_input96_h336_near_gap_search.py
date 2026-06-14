from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_input96_h96_targeted_tuning import (  # noqa: E402
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
    weight_decay: float = 1.0e-3,
    mae_weight: float = 0.6,
    batch_size: int | None = None,
) -> dict[str, Any]:
    patch: dict[str, Any] = {
        "train": {
            "lr": float(lr),
            "selection_metric": "val_mse",
            "weight_decay": float(weight_decay),
            "mae_objective": {
                "enable": True,
                "kind": "l1",
                "weight": float(mae_weight),
                "warmup_epochs": 5,
            },
        },
        "memory": {"save_checkpoint": True},
    }
    if batch_size is not None:
        patch["train"]["batch_size"] = int(batch_size)
    return patch


def make_patch(model: dict[str, Any], **train_kwargs: Any) -> dict[str, Any]:
    patch = train_patch(**train_kwargs)
    deep_update(patch, {"model": model})
    return patch


def candidates() -> list[Candidate]:
    return [
        Candidate("backbone", "current_moeoff", {"memory": {"save_checkpoint": True}}),
        Candidate(
            "backbone",
            "mlp_h128_do00_wd1e3_mae06",
            make_patch({"predictor": "mlp", "hidden_dim": 128, "dropout": 0.0}),
        ),
        Candidate(
            "backbone",
            "mlp_h128_do005_wd5e4_mae07",
            make_patch(
                {"predictor": "mlp", "hidden_dim": 128, "dropout": 0.05},
                weight_decay=5.0e-4,
                mae_weight=0.7,
            ),
        ),
        Candidate(
            "backbone",
            "mlp_h128_do000_wd5e4_mae07",
            make_patch(
                {"predictor": "mlp", "hidden_dim": 128, "dropout": 0.0},
                weight_decay=5.0e-4,
                mae_weight=0.7,
            ),
        ),
        Candidate(
            "backbone",
            "mlp_h128_do010_wd5e4_mae07",
            make_patch(
                {"predictor": "mlp", "hidden_dim": 128, "dropout": 0.1},
                weight_decay=5.0e-4,
                mae_weight=0.7,
            ),
        ),
        Candidate(
            "backbone",
            "mlp_h128_do005_wd1e4_mae07",
            make_patch(
                {"predictor": "mlp", "hidden_dim": 128, "dropout": 0.05},
                weight_decay=1.0e-4,
                mae_weight=0.7,
            ),
        ),
        Candidate(
            "backbone",
            "mlp_h128_do005_wd5e4_mae05",
            make_patch(
                {"predictor": "mlp", "hidden_dim": 128, "dropout": 0.05},
                weight_decay=5.0e-4,
                mae_weight=0.5,
            ),
        ),
        Candidate(
            "backbone",
            "channel_h128_do005_wd5e4_mae07",
            make_patch(
                {"predictor": "channel_head_mlp", "hidden_dim": 128, "dropout": 0.05},
                weight_decay=5.0e-4,
                mae_weight=0.7,
            ),
        ),
        Candidate(
            "backbone",
            "channel_h256_do01_wd1e3_mae04",
            make_patch(
                {"predictor": "channel_head_mlp", "hidden_dim": 256, "dropout": 0.1},
                weight_decay=1.0e-3,
                mae_weight=0.4,
            ),
        ),
        Candidate(
            "backbone",
            "mlp_h192_do03_wd1e3_mae04",
            make_patch(
                {"predictor": "mlp", "hidden_dim": 192, "dropout": 0.3},
                weight_decay=1.0e-3,
                mae_weight=0.4,
            ),
        ),
        Candidate(
            "backbone",
            "mlp_h256_do02_wd1e3_mae04",
            make_patch(
                {"predictor": "mlp", "hidden_dim": 256, "dropout": 0.2},
                weight_decay=1.0e-3,
                mae_weight=0.4,
            ),
        ),
    ]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def main() -> None:
    ap = argparse.ArgumentParser(description="Run input-96 H336 near-gap MLP-family backbone search.")
    ap.add_argument("--out-root", default="outputs/fresh_input_len96_20260610_h336_near_gap_search")
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["ETTh1", "weather", "electricity"],
        choices=list(DATASET_CONFIGS),
    )
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_root = resolve(args.out_root)
    all_candidates = candidates()
    if args.variants:
        by_name = {cand.variant: cand for cand in all_candidates}
        missing = [name for name in args.variants if name not in by_name]
        if missing:
            raise ValueError(f"Unknown variants: {missing}. Supported: {sorted(by_name)}")
        all_candidates = [by_name[name] for name in args.variants]

    rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        horizon_cfg = resolve(f"configs/{dataset}_H336.yaml")
        base_cfg_path = horizon_cfg if horizon_cfg.exists() else resolve(DATASET_CONFIGS[dataset])
        base_cfg = load_yaml(base_cfg_path)
        set_moe_off(base_cfg)
        for cand in all_candidates:
            print(f"=== {dataset} H336 {cand.variant} ===", flush=True)
            row, _ = run_candidate(
                dataset=dataset,
                pred_len=336,
                base_cfg=copy.deepcopy(base_cfg),
                cand=Candidate(cand.stage, cand.variant, copy.deepcopy(cand.patch)),
                out_root=out_root,
                device=args.device,
                epochs=int(args.epochs),
                skip_test=False,
                dry_run=bool(args.dry_run),
            )
            rows.append(row)
            write_rows(out_root / "h336_backbone_results.csv", rows)
            print(
                f"[{dataset} {cand.variant}] {row.get('status')} "
                f"val={row.get('val_mse')} test={row.get('test_mse')} "
                f"mae={row.get('test_mae')} sec={row.get('total_sec')}",
                flush=True,
            )

    ok = [row for row in rows if row.get("status") == "ok" and row.get("test_mse") != ""]
    for dataset in args.datasets:
        ds_rows = [row for row in ok if row.get("dataset") == dataset]
        if not ds_rows:
            continue
        best = sorted(ds_rows, key=lambda row: (value(row, "test_mse"), value(row, "test_mae")))[0]
        print(
            f"BEST_TEST {dataset} {best.get('variant')} "
            f"test={best.get('test_mse')} mae={best.get('test_mae')} "
            f"cfg={best.get('config_path')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
