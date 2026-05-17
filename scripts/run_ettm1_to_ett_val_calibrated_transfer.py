from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_ettm1_to_ettm2_input_route_selection import load_context, read_yaml  # noqa: E402
from scripts.run_ettm1_to_ettm2_val_calibrated_transfer import (  # noqa: E402
    collect_predictions,
    fit_apply,
    metrics,
    route_from_assignment,
    route_from_summary,
)


FIELDS = [
    "target",
    "horizon",
    "residual_mode",
    "route_name",
    "calibration",
    "shrink",
    "val_mse",
    "val_mae",
    "test_mse",
    "test_mae",
    "target_self_mse",
    "target_self_mae",
    "pct_vs_target_self",
    "old_policy",
    "old_test_mse",
    "old_test_mae",
    "old_pct_vs_target_self",
    "delta_mse_vs_old",
    "delta_pct_points_vs_old",
    "route",
]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})


def pair_dir(target: str, horizon: int) -> Path:
    if int(horizon) == 96:
        return ROOT / "outputs" / "aligned_h96_transfer_matrix" / f"ETTm1_to_{target}"
    return ROOT / "outputs" / "ettm1_other_ett_horizon_transfer" / f"ETTm1_to_{target}" / f"pred_{horizon}"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def target_self_metrics(target: str, horizon: int) -> tuple[float, float]:
    path = ROOT / "outputs" / "ett_horizon_sweep" / "runs" / target / f"pred_{horizon}" / "run_summary.json"
    payload = load_json(path)
    return float(payload["test"]["avg_mse"]), float(payload["test"]["avg_mae"])


def old_selected_metrics(target: str, horizon: int) -> tuple[str, float, float]:
    base = pair_dir(target, horizon)
    direct = load_json(base / "direct_transfer" / "transfer_summary.json")
    direct_mse = float(direct["avg_mse"])
    summary_path = base / "val_loss_selection" / "summary.json"
    if summary_path.exists():
        selected = load_json(summary_path)
        val_mse = float(selected["selected_test_mse"])
        val_mae = float(selected["selected_test_mae"])
    else:
        selected_summary = base / "val_loss_selection" / "selected_test_transfer" / "transfer_summary.json"
        selected = load_json(selected_summary)
        val_mse = float(selected["avg_mse"])
        val_mae = float(selected["avg_mae"])
    if val_mse > direct_mse:
        return "direct_train_only", direct_mse, float(direct["avg_mae"])
    return "val_route", val_mse, val_mae


def route_candidates(target: str, horizon: int) -> dict[str, tuple[int, ...]]:
    base = pair_dir(target, horizon)
    static_route = route_from_assignment(base / "direct_transfer" / "cluster_assignment.csv")
    summary_path = base / "val_loss_selection" / "summary.json"
    if summary_path.exists():
        selected_route = route_from_summary(summary_path)
    else:
        selected_route = route_from_assignment(
            base / "val_loss_selection" / "selected_test_transfer" / "cluster_assignment.csv"
        )
    k_count = max(max(static_route), max(selected_route)) + 1
    k_count = max(k_count, 3)
    out = {
        "static_corr_train": static_route,
        "val_route": selected_route,
    }
    for k in range(k_count):
        out[f"all_{k}"] = tuple([k] * len(static_route))
    return out


def make_summary_plots(out_root: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return
    best = pd.read_csv(out_root / "best_by_val.csv")
    targets = ["ETTh1", "ETTh2", "ETTm2"]
    horizons = [96, 192, 336, 720]

    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=180)
    x = np.arange(len(horizons))
    width = 0.24
    colors = {"ETTh1": "#2f6f9f", "ETTh2": "#d05a3f", "ETTm2": "#4e8b4a"}
    for i, target in enumerate(targets):
        sub = best[best["target"] == target].set_index("horizon").reindex(horizons)
        vals = sub["pct_vs_target_self"].to_numpy(dtype=float)
        xs = x + (i - 1) * width
        ax.bar(xs, vals, width=width, label=target, color=colors.get(target))
        for xi, v in zip(xs, vals):
            if np.isfinite(v):
                ax.text(xi, v + (1.2 if v >= 0 else -1.8), f"{v:+.1f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
    ax.axhline(10, color="#555", ls="--", lw=1, label="10% target")
    ax.axhline(0, color="#333", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"H{h}" for h in horizons])
    ax.set_ylabel("val-calibrated transfer MSE vs target self (%)")
    ax.set_title("ETTm1 -> ETT targets, selected by validation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=4, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_root / "best_by_val_pct.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=180)
    for target in targets:
        sub = best[best["target"] == target].sort_values("horizon")
        ax.plot(sub["horizon"], sub["test_mse"], marker="o", label=target, color=colors.get(target))
    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel("Val-calibrated transfer test MSE")
    ax.set_title("ETTm1 -> ETT targets, test MSE")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_root / "best_by_val_mse.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "ettm1_to_ett_val_calibrated_transfer")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--targets", nargs="+", default=["ETTh1", "ETTh2", "ETTm2"])
    ap.add_argument("--horizons", type=int, nargs="+", default=[96, 192, 336, 720])
    args = ap.parse_args()

    methods = [
        "none",
        "bias_channel",
        "bias_channel_horizon",
        "affine_channel",
        "affine_channel_horizon",
    ]
    shrinks = [0.25, 0.5, 0.75, 1.0]
    rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    args.out_root.mkdir(parents=True, exist_ok=True)

    for target in args.targets:
        for horizon in args.horizons:
            base = pair_dir(target, horizon)
            cfg = read_yaml(base / "base_config.yaml")
            cfg.setdefault("exp", {})["device"] = args.device
            target_self_mse, target_self_mae = target_self_metrics(target, horizon)
            old_policy, old_mse, old_mae = old_selected_metrics(target, horizon)
            old_pct = (old_mse / target_self_mse - 1.0) * 100.0
            ctx = load_context(cfg, torch.device(args.device))
            candidate_rows: list[dict[str, Any]] = []
            for route_name, route in route_candidates(target, horizon).items():
                print(f"{target} H{horizon} {route_name}", flush=True)
                pred_val, true_val = collect_predictions(ctx, route, split="val", batch_size=args.batch_size)
                pred_test, true_test = collect_predictions(ctx, route, split="test", batch_size=args.batch_size)
                for method in methods:
                    method_shrinks = [1.0] if method == "none" else shrinks
                    for shrink in method_shrinks:
                        cal_val, cal_test = fit_apply(pred_val, true_val, pred_test, method=method, shrink=shrink)
                        val_mse, val_mae = metrics(cal_val, true_val)
                        test_mse, test_mae = metrics(cal_test, true_test)
                        pct = (test_mse / target_self_mse - 1.0) * 100.0
                        row = {
                            "target": target,
                            "horizon": horizon,
                            "residual_mode": "full_moe_residual",
                            "route_name": route_name,
                            "calibration": method,
                            "shrink": shrink,
                            "val_mse": val_mse,
                            "val_mae": val_mae,
                            "test_mse": test_mse,
                            "test_mae": test_mae,
                            "target_self_mse": target_self_mse,
                            "target_self_mae": target_self_mae,
                            "pct_vs_target_self": pct,
                            "old_policy": old_policy,
                            "old_test_mse": old_mse,
                            "old_test_mae": old_mae,
                            "old_pct_vs_target_self": old_pct,
                            "delta_mse_vs_old": test_mse - old_mse,
                            "delta_pct_points_vs_old": pct - old_pct,
                            "route": json.dumps(list(route)),
                        }
                        rows.append(row)
                        candidate_rows.append(row)
                write_rows(args.out_root / "all_results.csv", rows)
            best = min(candidate_rows, key=lambda r: (float(r["val_mse"]), float(r["val_mae"])))
            best_rows.append(best)
            write_rows(args.out_root / "best_by_val.csv", best_rows)

    write_rows(args.out_root / "all_results.csv", rows)
    write_rows(args.out_root / "best_by_val.csv", best_rows)
    with (args.out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"selection": "lowest validation MSE per target/horizon", "best_by_val": best_rows}, f, ensure_ascii=False, indent=2)
    make_summary_plots(args.out_root)
    print(args.out_root / "best_by_val.csv")
    print(pd.DataFrame(best_rows)[[
        "target",
        "horizon",
        "old_test_mse",
        "test_mse",
        "target_self_mse",
        "old_pct_vs_target_self",
        "pct_vs_target_self",
        "route_name",
        "calibration",
        "shrink",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
