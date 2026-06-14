from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_input96_h96_targeted_tuning import (
    Candidate,
    DATASET_CONFIGS,
    filter_candidates_by_variant,
    load_yaml,
    moe_candidates,
    resolve,
    run_candidate,
    value,
    write_yaml,
)


RESULT_FIELDS = [
    "dataset",
    "pred_len",
    "variant",
    "status",
    "test_mse",
    "test_mae",
    "val_mse",
    "val_mae",
    "baseline_test_mse",
    "baseline_test_mae",
    "mse_gain_vs_baseline",
    "mae_gain_vs_baseline",
    "mse_gain_pct",
    "mae_gain_pct",
    "residual_mean_scale",
    "residual_num_channels",
    "alpha_scale",
    "residual_clip",
    "selection_policy",
    "selection_min_rel_improvement",
    "gate_max_scale",
    "gate_init_scale",
    "config_path",
    "out_dir",
    "returncode",
    "error",
]


def read_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def baseline_metrics_from_config(cfg: dict[str, Any]) -> tuple[float | None, float | None]:
    out_dir = cfg.get("exp", {}).get("out_dir")
    if not out_dir:
        return None, None
    summary = read_summary(Path(str(out_dir)) / "run_summary.json")
    test = summary.get("test") or {}
    try:
        test_mse = float(test["avg_mse"])
        test_mae = float(test["avg_mae"])
    except Exception:
        return None, None
    return test_mse, test_mae


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})


def apply_moe_training_controls(
    cfg: dict[str, Any],
    *,
    warm_start_checkpoint: str | None,
    freeze_backbone: bool,
    lr: float | None,
    weight_decay: float | None,
) -> None:
    if warm_start_checkpoint:
        cfg["finetune"] = {
            "enable": True,
            "checkpoint_path": str(warm_start_checkpoint),
            "strict_window": True,
            "strict_model": True,
            "cluster_map": "index",
            "load_model": True,
            "load_gate": False,
            "load_dynamic_lambda": False,
            "load_learnable_lambda": False,
        }
    if freeze_backbone:
        cfg.setdefault("moe", {})["freeze_backbone"] = True
    if lr is not None:
        cfg.setdefault("train", {})["lr"] = float(lr)
    if weight_decay is not None:
        cfg.setdefault("train", {})["weight_decay"] = float(weight_decay)


def apply_history_anchor_controls(
    cfg: dict[str, Any],
    *,
    lags: str | None,
    alpha: float | None,
    blend_target: str,
) -> None:
    if alpha is None:
        return
    parsed_lags = []
    for item in str(lags or "").split(","):
        item = item.strip()
        if item:
            value = int(item)
            if value > 0:
                parsed_lags.append(value)
    if not parsed_lags:
        raise ValueError("--history-anchor-alpha requires at least one --history-anchor-lags value.")
    if blend_target not in {"prediction", "base"}:
        raise ValueError("--history-anchor-blend-target must be prediction or base.")
    cfg.setdefault("model", {})["history_anchor"] = {
        "enable": True,
        "lags": parsed_lags,
        "alpha": float(alpha),
        "blend_target": str(blend_target),
        "history_scope": "input_window",
    }
    cfg.setdefault("knn_hybrid", {})["enable"] = False


def add_baseline_deltas(
    row: dict[str, Any],
    *,
    baseline_test_mse: float | None,
    baseline_test_mae: float | None,
) -> dict[str, Any]:
    row = dict(row)
    row["baseline_test_mse"] = "" if baseline_test_mse is None else baseline_test_mse
    row["baseline_test_mae"] = "" if baseline_test_mae is None else baseline_test_mae
    test_mse = value(row, "test_mse")
    test_mae = value(row, "test_mae")
    if baseline_test_mse is not None and test_mse != float("inf"):
        gain = float(baseline_test_mse) - float(test_mse)
        row["mse_gain_vs_baseline"] = gain
        row["mse_gain_pct"] = 100.0 * gain / max(abs(float(baseline_test_mse)), 1.0e-12)
    else:
        row["mse_gain_vs_baseline"] = ""
        row["mse_gain_pct"] = ""
    if baseline_test_mae is not None and test_mae != float("inf"):
        gain = float(baseline_test_mae) - float(test_mae)
        row["mae_gain_vs_baseline"] = gain
        row["mae_gain_pct"] = 100.0 * gain / max(abs(float(baseline_test_mae)), 1.0e-12)
    else:
        row["mae_gain_vs_baseline"] = ""
        row["mae_gain_pct"] = ""
    return row


def best_positive_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    ok = [row for row in rows if row.get("status") == "ok" and row.get("mse_gain_vs_baseline") != ""]
    positive = [row for row in ok if float(row["mse_gain_vs_baseline"]) > 0.0]
    pool = positive if positive else ok
    if not pool:
        return None
    return sorted(pool, key=lambda row: (-float(row["mse_gain_vs_baseline"]), value(row, "test_mae")))[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run MoE-only input-96 candidates against an existing MoE-off backbone.")
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()))
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--out-root", default="outputs/input96_moe_positive_search")
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--moe-variants", nargs="+", default=None)
    ap.add_argument("--warm-start-checkpoint", default=None)
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--baseline-test-mse", type=float, default=None)
    ap.add_argument("--baseline-test-mae", type=float, default=None)
    ap.add_argument("--history-anchor-lags", default="96,192,288")
    ap.add_argument("--history-anchor-alpha", type=float, default=None)
    ap.add_argument("--history-anchor-blend-target", choices=["prediction", "base"], default="prediction")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base_cfg = load_yaml(resolve(args.base_config))
    baseline_test_mse, baseline_test_mae = baseline_metrics_from_config(base_cfg)
    if args.baseline_test_mse is not None:
        baseline_test_mse = float(args.baseline_test_mse)
    if args.baseline_test_mae is not None:
        baseline_test_mae = float(args.baseline_test_mae)
    apply_moe_training_controls(
        base_cfg,
        warm_start_checkpoint=args.warm_start_checkpoint,
        freeze_backbone=bool(args.freeze_backbone),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    apply_history_anchor_controls(
        base_cfg,
        lags=args.history_anchor_lags,
        alpha=args.history_anchor_alpha,
        blend_target=args.history_anchor_blend_target,
    )

    out_root = resolve(args.out_root)
    selected_candidates = filter_candidates_by_variant(moe_candidates(base_cfg, "compact"), args.moe_variants)
    if not selected_candidates:
        raise SystemExit("No MoE candidates selected.")

    rows: list[dict[str, Any]] = []
    cfgs: dict[str, dict[str, Any]] = {}
    for cand in selected_candidates:
        run_cand = Candidate("moe_positive", cand.variant, copy.deepcopy(cand.patch))
        row, cfg = run_candidate(
            dataset=str(args.dataset),
            pred_len=int(args.horizon),
            base_cfg=base_cfg,
            cand=run_cand,
            out_root=out_root,
            device=args.device,
            epochs=args.epochs,
            skip_test=False,
            dry_run=bool(args.dry_run),
        )
        row = add_baseline_deltas(
            row,
            baseline_test_mse=baseline_test_mse,
            baseline_test_mae=baseline_test_mae,
        )
        rows.append(row)
        cfgs[cand.variant] = cfg
        write_rows(out_root / "moe_positive_results.csv", rows)
        print(
            f"[{args.dataset} H{args.horizon} {cand.variant}] {row['status']} "
            f"test_mse={row['test_mse']} gain={row['mse_gain_vs_baseline']} "
            f"gain_pct={row['mse_gain_pct']}",
            flush=True,
        )

    best = best_positive_row(rows)
    if best is not None:
        best_cfg = cfgs[str(best["variant"])]
        best_path = out_root / "best_configs" / str(args.dataset) / f"H{int(args.horizon)}.yaml"
        write_yaml(best_path, best_cfg)
        print(f"BEST {best['variant']} gain={best['mse_gain_vs_baseline']} config={best_path}", flush=True)


if __name__ == "__main__":
    main()
