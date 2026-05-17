from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_ettm1_to_ett_val_calibrated_transfer import pair_dir  # noqa: E402
from scripts.run_ettm1_to_ettm2_input_route_selection import load_context, read_yaml  # noqa: E402
from scripts.run_ettm1_to_ettm2_val_calibrated_transfer import collect_predictions, metrics  # noqa: E402


def fit_affine_channel_horizon(
    pred_val: torch.Tensor,
    true_val: torch.Tensor,
    pred_test: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    eps = 1.0e-6
    x = pred_val
    y = true_val
    mx = x.mean(dim=0, keepdim=True)
    my = y.mean(dim=0, keepdim=True)
    vx = (x - mx).pow(2).mean(dim=0, keepdim=True).clamp_min(eps)
    cov = ((x - mx) * (y - my)).mean(dim=0, keepdim=True)
    a = (cov / vx).clamp(0.5, 1.5)
    b = my - a * mx
    return a, b, a * pred_val + b, a * pred_test + b


def corr_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.detach().cpu().reshape(-1).double()
    y = b.detach().cpu().reshape(-1).double()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom) <= 1.0e-12:
        return float("nan")
    return float((x @ y / denom).item())


def load_best_rows(path: Path, targets: list[str], horizons: list[int]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["target"].isin(targets) & df["horizon"].astype(int).isin(horizons)].copy()
    df["horizon"] = df["horizon"].astype(int)
    return df.sort_values(["target", "horizon"]).reset_index(drop=True)


def plot_outputs(out_root: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=180)
    x = np.arange(len(df))
    width = 0.24
    labels = [f"{r.target}-H{int(r.horizon)}" for r in df.itertuples()]
    ax.bar(x - width, df["route_no_cal_test_mse"], width=width, label="selected route, no calibration", color="#8b8b8b")
    ax.bar(x, df["cal_test_mse"], width=width, label="val affine calibrated", color="#3f7f5f")
    ax.bar(x + width, df["target_self_mse"], width=width, label="target self", color="#4c78a8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Test MSE")
    ax.set_title("Calibration decomposes transfer gains")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_root / "calibration_decomposition_mse.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=180)
    for r in rows:
        val = np.asarray(r["residual_mean_val_flat"], dtype=float)
        test = np.asarray(r["residual_mean_test_flat"], dtype=float)
        ax.scatter(val, test, s=16, alpha=0.55, label=f"{r['target']}-H{r['horizon']} r={r['residual_mean_val_test_corr']:.2f}")
    lim = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]), abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]))
    ax.plot([-lim, lim], [-lim, lim], color="#333", lw=1, ls="--")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Validation residual mean per channel-horizon")
    ax.set_ylabel("Test residual mean per channel-horizon")
    ax.set_title("Val-test bias consistency before calibration")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_root / "val_test_residual_bias_scatter.png")
    plt.close(fig)

    for r in rows:
        h = int(r["horizon"])
        target = str(r["target"])
        before = np.asarray(r["test_bias_by_horizon_before"], dtype=float)
        after = np.asarray(r["test_bias_by_horizon_after"], dtype=float)
        fig, ax = plt.subplots(figsize=(8.4, 3.8), dpi=180)
        xs = np.arange(h)
        ax.plot(xs, before, label="before", color="#c86d4a", lw=1.3)
        ax.plot(xs, after, label="after", color="#3f7f5f", lw=1.3)
        ax.set_xlabel("Horizon step")
        ax.set_ylabel("Mean abs test bias across channels")
        ax.set_title(f"{target} H{h}: horizon-wise bias reduction")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_root / f"{target}_H{h}_test_bias_by_horizon.png")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", type=Path, default=ROOT / "outputs" / "ettm1_to_ett_val_calibrated_transfer" / "best_by_val.csv")
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "ett_val_calibration_diagnostics")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--targets", nargs="+", default=["ETTm2", "ETTh2"])
    ap.add_argument("--horizons", type=int, nargs="+", default=[192, 336, 720])
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    best = load_best_rows(args.best, args.targets, args.horizons)
    rows: list[dict[str, Any]] = []

    for item in best.to_dict("records"):
        target = str(item["target"])
        horizon = int(item["horizon"])
        route = tuple(int(v) for v in json.loads(str(item["route"])))
        cfg = read_yaml(pair_dir(target, horizon) / "base_config.yaml")
        cfg.setdefault("exp", {})["device"] = args.device
        ctx = load_context(cfg, torch.device(args.device))
        print(f"diagnose {target} H{horizon}", flush=True)
        pred_val, true_val = collect_predictions(ctx, route, split="val", batch_size=args.batch_size)
        pred_test, true_test = collect_predictions(ctx, route, split="test", batch_size=args.batch_size)
        a, b, cal_val, cal_test = fit_affine_channel_horizon(pred_val, true_val, pred_test)

        route_val_mse, route_val_mae = metrics(pred_val, true_val)
        route_test_mse, route_test_mae = metrics(pred_test, true_test)
        cal_val_mse, cal_val_mae = metrics(cal_val, true_val)
        cal_test_mse, cal_test_mae = metrics(cal_test, true_test)

        residual_mean_val = (true_val - pred_val).mean(dim=0)
        residual_mean_test = (true_test - pred_test).mean(dim=0)
        residual_after_test = (true_test - cal_test).mean(dim=0)

        row = {
            "target": target,
            "horizon": horizon,
            "route_name": item["route_name"],
            "calibration": item["calibration"],
            "route_no_cal_val_mse": route_val_mse,
            "route_no_cal_test_mse": route_test_mse,
            "cal_val_mse": cal_val_mse,
            "cal_test_mse": cal_test_mse,
            "target_self_mse": float(item["target_self_mse"]),
            "old_selected_test_mse": float(item["old_test_mse"]),
            "route_no_cal_test_mae": route_test_mae,
            "cal_test_mae": cal_test_mae,
            "residual_mean_val_test_corr": corr_flat(residual_mean_val, residual_mean_test),
            "test_bias_abs_mean_before": float(residual_mean_test.abs().mean().item()),
            "test_bias_abs_mean_after": float(residual_after_test.abs().mean().item()),
            "bias_abs_reduction_pct": float((1.0 - residual_after_test.abs().mean().item() / max(residual_mean_test.abs().mean().item(), 1.0e-12)) * 100.0),
            "slope_mean": float(a.mean().item()),
            "slope_std": float(a.std().item()),
            "slope_min": float(a.min().item()),
            "slope_max": float(a.max().item()),
            "intercept_abs_mean": float(b.abs().mean().item()),
            "intercept_std": float(b.std().item()),
            "residual_mean_val_flat": residual_mean_val.reshape(-1).detach().cpu().tolist(),
            "residual_mean_test_flat": residual_mean_test.reshape(-1).detach().cpu().tolist(),
            "test_bias_by_horizon_before": residual_mean_test.abs().mean(dim=0).detach().cpu().tolist(),
            "test_bias_by_horizon_after": residual_after_test.abs().mean(dim=0).detach().cpu().tolist(),
        }
        rows.append(row)

    full_path = args.out_root / "diagnostics_full.json"
    with full_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    compact_rows = [
        {k: v for k, v in row.items() if not isinstance(v, list)}
        for row in rows
    ]
    compact = pd.DataFrame(compact_rows)
    compact.to_csv(args.out_root / "diagnostics.csv", index=False)
    plot_outputs(args.out_root, rows)
    print(compact.to_string(index=False))
    print(args.out_root / "diagnostics.csv")


if __name__ == "__main__":
    main()
