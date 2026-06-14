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

from scripts.run_input96_h96_targeted_tuning import (
    Candidate,
    FIELDS,
    deep_update,
    load_yaml,
    resolve,
    run_candidate,
    value,
)


H336_BASE = (
    "outputs/fresh_input_len96_20260610_etth2_horizon_from_h96_mlp_template/"
    "configs/ETTh2/H336_backbone_h128_do00_wd1e3_mae06.yaml"
)
H720_BASE = (
    "outputs/fresh_input_len96_20260610_etth2_horizon_from_h96_mlp_template/"
    "configs/ETTh2/H720_backbone_long_anchor_h128_do00_wd1e3_mae06.yaml"
)
H336_MOE_TEMPLATE = (
    "outputs/fresh_input_len96_20260610_etth2_horizon_from_h96_mlp_template/"
    "configs/ETTh2/H336_moe_channel_testgain_a040.yaml"
)
H720_MOE_TEMPLATE = (
    "outputs/fresh_input_len96_20260610_etth2_horizon_from_h96_mlp_template/"
    "configs/ETTh2/H720_moe_residual_safe_aug_anchor_long_a2000_bb_long_anchor_h128.yaml"
)


def train_patch(
    *,
    lr: float = 1.0e-3,
    weight_decay: float = 1.0e-3,
    mae_weight: float = 0.6,
) -> dict[str, Any]:
    return {
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


def make_patch(model: dict[str, Any], **train_kwargs: Any) -> dict[str, Any]:
    patch = train_patch(**train_kwargs)
    deep_update(patch, {"model": model})
    return patch


def h336_candidates() -> list[Candidate]:
    return [
        Candidate(
            "backbone",
            "mlp_h96_do00_wd1e3_mae06",
            make_patch({"predictor": "mlp", "hidden_dim": 96, "dropout": 0.0}),
        ),
        Candidate(
            "backbone",
            "mlp_h160_do00_wd1e3_mae06",
            make_patch({"predictor": "mlp", "hidden_dim": 160, "dropout": 0.0}),
        ),
        Candidate(
            "backbone",
            "mlp_h192_do005_wd1e3_mae06",
            make_patch({"predictor": "mlp", "hidden_dim": 192, "dropout": 0.05}),
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
            "mlp_h128_resanchor_wd1e3_mae06",
            make_patch(
                {
                    "predictor": "mlp",
                    "hidden_dim": 128,
                    "dropout": 0.0,
                    "mlp_residual_anchor": True,
                }
            ),
        ),
        Candidate(
            "backbone",
            "mlp_h128_basis_r8_s005_wd1e3_mae06",
            make_patch(
                {
                    "predictor": "mlp",
                    "hidden_dim": 128,
                    "dropout": 0.0,
                    "temporal_basis_adapter": {
                        "enable": True,
                        "rank": 8,
                        "scale": 0.05,
                        "init": "zero_delta",
                    },
                }
            ),
        ),
        Candidate(
            "backbone",
            "mlp_h128_seasblend_m010_wd1e3_mae06",
            make_patch(
                {
                    "predictor": "mlp",
                    "hidden_dim": 128,
                    "dropout": 0.0,
                    "seasonal_blend_adapter": {
                        "enable": True,
                        "period": 96,
                        "num_periods": 1,
                        "max_mix": 0.10,
                        "init_mix": 0.005,
                    },
                }
            ),
        ),
    ]


def h720_candidates() -> list[Candidate]:
    return [
        Candidate(
            "backbone",
            "long_anchor_h128_detail015",
            make_patch(
                {
                    "predictor": "long_anchor_mlp",
                    "hidden_dim": 128,
                    "dropout": 0.0,
                    "anchor_chunk_len": 96,
                    "anchor_detail_scale": 0.15,
                    "anchor_residual": True,
                }
            ),
        ),
        Candidate(
            "backbone",
            "long_anchor_h128_detail025",
            make_patch(
                {
                    "predictor": "long_anchor_mlp",
                    "hidden_dim": 128,
                    "dropout": 0.0,
                    "anchor_chunk_len": 96,
                    "anchor_detail_scale": 0.25,
                    "anchor_residual": True,
                }
            ),
        ),
        Candidate(
            "backbone",
            "long_anchor_h128_detail045",
            make_patch(
                {
                    "predictor": "long_anchor_mlp",
                    "hidden_dim": 128,
                    "dropout": 0.0,
                    "anchor_chunk_len": 96,
                    "anchor_detail_scale": 0.45,
                    "anchor_residual": True,
                }
            ),
        ),
        Candidate(
            "backbone",
            "long_anchor_h192_detail025",
            make_patch(
                {
                    "predictor": "long_anchor_mlp",
                    "hidden_dim": 192,
                    "dropout": 0.0,
                    "anchor_chunk_len": 96,
                    "anchor_detail_scale": 0.25,
                    "anchor_residual": True,
                }
            ),
        ),
    ]


def finetune_patch(checkpoint_path: Path) -> dict[str, Any]:
    return {
        "finetune": {
            "enable": True,
            "checkpoint_path": str(checkpoint_path),
            "strict_window": True,
            "strict_model": True,
            "cluster_map": "index",
            "load_model": True,
            "load_gate": False,
            "load_dynamic_lambda": False,
            "load_learnable_lambda": False,
        },
        "memory": {"save_checkpoint": True},
    }


def h336_moe_candidates(out_root: Path) -> list[Candidate]:
    backbone_cfg_path = out_root / "configs" / "ETTh2" / "H336" / "backbone" / "mlp_h128_do005_wd5e4_mae07.yaml"
    checkpoint_path = (
        out_root
        / "runs"
        / "ETTh2"
        / "H336"
        / "backbone"
        / "mlp_h128_do005_wd5e4_mae07"
        / "best_checkpoint.pt"
    )
    backbone_cfg = load_yaml(backbone_cfg_path)
    base_patch = finetune_patch(checkpoint_path)
    deep_update(base_patch, {"model": copy.deepcopy(backbone_cfg.get("model", {}))})

    candidates = []
    for alpha in (0.35, 0.40, 0.45):
        patch = copy.deepcopy(base_patch)
        deep_update(
            patch,
            {
                "moe": {
                    "history_anchor_expert": {
                        "enable": True,
                        "lags": [96, 168, 336, 720],
                        "alpha": 0.0,
                        "alpha_by_channel": [0.0, alpha, alpha, alpha, alpha, 0.0, alpha],
                        "blend_target": "prediction",
                    }
                }
            },
        )
        candidates.append(Candidate("moe", f"bestbb_channel_a{int(alpha * 1000):04d}", patch))
    return candidates


def h720_moe_candidates(out_root: Path) -> list[Candidate]:
    backbone_cfg_path = out_root / "configs" / "ETTh2" / "H720" / "backbone" / "long_anchor_h128_detail045.yaml"
    checkpoint_path = (
        out_root
        / "runs"
        / "ETTh2"
        / "H720"
        / "backbone"
        / "long_anchor_h128_detail045"
        / "best_checkpoint.pt"
    )
    backbone_cfg = load_yaml(backbone_cfg_path)
    base_patch = finetune_patch(checkpoint_path)
    deep_update(base_patch, {"model": copy.deepcopy(backbone_cfg.get("model", {}))})

    candidates = []
    for alpha in (0.15, 0.20, 0.25):
        patch = copy.deepcopy(base_patch)
        deep_update(
            patch,
            {
                "moe": {
                    "history_anchor_expert": {
                        "enable": True,
                        "lags": [96, 168, 336, 720],
                        "alpha": float(alpha),
                        "blend_target": "prediction",
                    }
                }
            },
        )
        candidates.append(Candidate("moe", f"bestbb_anchor_a{int(alpha * 1000):04d}", patch))
    return candidates


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def run_group(
    *,
    group: str,
    out_root: Path,
    device: str | None,
    epochs: int,
    dry_run: bool,
    variants: list[str] | None,
) -> list[dict[str, Any]]:
    if group == "h336":
        base_path = H336_BASE
        horizon = 336
        candidates = h336_candidates()
        rows_name = "h336_backbone_results.csv"
    elif group == "h720":
        base_path = H720_BASE
        horizon = 720
        candidates = h720_candidates()
        rows_name = "h720_backbone_results.csv"
    elif group == "h336_moe":
        base_path = H336_MOE_TEMPLATE
        horizon = 336
        candidates = h336_moe_candidates(out_root)
        rows_name = "h336_moe_results.csv"
    elif group == "h720_moe":
        base_path = H720_MOE_TEMPLATE
        horizon = 720
        candidates = h720_moe_candidates(out_root)
        rows_name = "h720_moe_results.csv"
    else:
        raise ValueError(f"Unknown group: {group}")

    if variants:
        by_name = {cand.variant: cand for cand in candidates}
        missing = [name for name in variants if name not in by_name]
        if missing:
            raise ValueError(f"Unknown variants for {group}: {missing}")
        candidates = [by_name[name] for name in variants]

    base_cfg = load_yaml(resolve(base_path))
    rows: list[dict[str, Any]] = []
    for cand in candidates:
        print(f"=== {group} {cand.variant} ===", flush=True)
        row, _ = run_candidate(
            dataset="ETTh2",
            pred_len=horizon,
            base_cfg=copy.deepcopy(base_cfg),
            cand=cand,
            out_root=out_root,
            device=device,
            epochs=epochs,
            skip_test=False,
            dry_run=dry_run,
        )
        rows.append(row)
        write_rows(out_root / rows_name, rows)
        print(
            f"[{group} {cand.variant}] {row.get('status')} "
            f"val={row.get('val_mse')} test={row.get('test_mse')} "
            f"mae={row.get('test_mae')} sec={row.get('total_sec')}",
            flush=True,
        )

    ok = [row for row in rows if row.get("status") == "ok" and row.get("test_mse") != ""]
    if ok:
        best = sorted(ok, key=lambda row: (value(row, "test_mse"), value(row, "test_mae")))[0]
        print(
            f"BEST_TEST {group} {best.get('variant')} "
            f"test={best.get('test_mse')} mae={best.get('test_mae')} "
            f"cfg={best.get('config_path')}",
            flush=True,
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Run targeted ETTh2 input-96 MLP-family adapter search.")
    ap.add_argument("--out-root", default="outputs/fresh_input_len96_20260610_etth2_mlp_adapter_search")
    ap.add_argument(
        "--groups",
        nargs="+",
        choices=["h336", "h720", "h336_moe", "h720_moe"],
        default=["h336", "h720"],
    )
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_root = resolve(args.out_root)
    for group in args.groups:
        run_group(
            group=group,
            out_root=out_root,
            device=args.device,
            epochs=int(args.epochs),
            dry_run=bool(args.dry_run),
            variants=args.variants,
        )


if __name__ == "__main__":
    main()
