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

from scripts.run_ettm1_to_ett_val_calibrated_transfer import pair_dir  # noqa: E402
from scripts.run_ettm1_to_ettm2_input_route_selection import load_context, read_yaml  # noqa: E402
from scripts.run_ettm1_to_ettm2_val_calibrated_transfer import collect_predictions, metrics  # noqa: E402
from scripts.run_target_self_val_calibration_control import (  # noqa: E402
    collect_self_predictions,
    load_self_context,
)


FIELDS = [
    "target",
    "horizon",
    "model_kind",
    "route_name",
    "calibration_form",
    "parameter_shape",
    "num_parameters",
    "val_mse",
    "val_mae",
    "test_mse",
    "test_mae",
    "target_self_raw_mse",
    "target_self_raw_mae",
    "pct_vs_target_self_raw",
]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})


def fit_form(
    pred_val: torch.Tensor,
    true_val: torch.Tensor,
    pred_test: torch.Tensor,
    *,
    form: str,
) -> tuple[torch.Tensor, torch.Tensor, str, int]:
    eps = 1.0e-6
    _, channels, horizon = pred_val.shape
    if form == "none":
        return pred_val, pred_test, "none", 0

    if form == "global_affine":
        reduce_dims = (0, 1, 2)
        shape = "a,b scalar"
        num_params = 2
    elif form == "channel_affine":
        reduce_dims = (0, 2)
        shape = f"a,b per channel [C={channels}]"
        num_params = 2 * channels
    elif form == "channel_step_affine":
        reduce_dims = (0,)
        shape = f"a,b per channel-step [C={channels}, H={horizon}]"
        num_params = 2 * channels * horizon
    else:
        raise ValueError(f"unknown calibration form: {form}")

    x = pred_val
    y = true_val
    mx = x.mean(dim=reduce_dims, keepdim=True)
    my = y.mean(dim=reduce_dims, keepdim=True)
    vx = (x - mx).pow(2).mean(dim=reduce_dims, keepdim=True).clamp_min(eps)
    cov = ((x - mx) * (y - my)).mean(dim=reduce_dims, keepdim=True)
    a = (cov / vx).clamp(0.5, 1.5)
    b = my - a * mx
    return a * pred_val + b, a * pred_test + b, shape, int(num_params)


def eval_forms(
    *,
    target: str,
    horizon: int,
    model_kind: str,
    route_name: str,
    pred_val: torch.Tensor,
    true_val: torch.Tensor,
    pred_test: torch.Tensor,
    true_test: torch.Tensor,
    target_self_raw_mse: float,
    target_self_raw_mae: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for form in ["none", "global_affine", "channel_affine", "channel_step_affine"]:
        cal_val, cal_test, shape, num_params = fit_form(pred_val, true_val, pred_test, form=form)
        val_mse, val_mae = metrics(cal_val, true_val)
        test_mse, test_mae = metrics(cal_test, true_test)
        rows.append(
            {
                "target": target,
                "horizon": int(horizon),
                "model_kind": model_kind,
                "route_name": route_name,
                "calibration_form": form,
                "parameter_shape": shape,
                "num_parameters": int(num_params),
                "val_mse": val_mse,
                "val_mae": val_mae,
                "test_mse": test_mse,
                "test_mae": test_mae,
                "target_self_raw_mse": target_self_raw_mse,
                "target_self_raw_mae": target_self_raw_mae,
                "pct_vs_target_self_raw": (test_mse / max(target_self_raw_mse, 1.0e-12) - 1.0) * 100.0,
            }
        )
    return rows


def make_plots(out_root: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    df = pd.DataFrame(rows)
    transfer = df[df["model_kind"] == "transfer"].copy()
    targets = ["ETTh1", "ETTh2", "ETTm2"]
    forms = ["none", "global_affine", "channel_affine", "channel_step_affine"]
    colors = {
        "none": "#8A8A8A",
        "global_affine": "#4C78A8",
        "channel_affine": "#72A24D",
        "channel_step_affine": "#D65F45",
    }

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.2), dpi=220, sharey=True)
    for ax, target in zip(axes, targets):
        sub = transfer[transfer["target"] == target]
        for form in forms:
            ss = sub[sub["calibration_form"] == form].sort_values("horizon")
            ax.plot(
                ss["horizon"],
                ss["pct_vs_target_self_raw"],
                marker="o",
                lw=1.8,
                label=form,
                color=colors[form],
            )
        ax.axhline(0, color="#222222", lw=1.0)
        ax.axhline(10, color="#B04A3A", lw=1.0, ls=(0, (4, 3)))
        ax.set_title(target, fontweight="bold")
        ax.set_xticks([96, 192, 336, 720])
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Transfer test MSE vs target self raw (%)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Calibration capacity ablation for ETTm1 transfer", fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(out_root / "transfer_calibration_capacity_pct.png", bbox_inches="tight")
    plt.close(fig)

    pivot = transfer.pivot_table(
        index=["target", "horizon"],
        columns="calibration_form",
        values="test_mse",
        aggfunc="first",
    ).reset_index()
    fig, ax = plt.subplots(figsize=(10.8, 4.8), dpi=220)
    cases = [f"{r.target}\nH{int(r.horizon)}" for r in pivot.itertuples()]
    x = np.arange(len(pivot))
    width = 0.18
    for i, form in enumerate(forms):
        ax.bar(x + (i - 1.5) * width, pivot[form], width=width, label=form, color=colors[form])
    ax.set_xticks(x)
    ax.set_xticklabels(cases, fontsize=8)
    ax.set_ylabel("Transfer test MSE")
    ax.set_title("Calibration form changes transfer test MSE")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_root / "transfer_calibration_capacity_mse.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transfer-best", type=Path, default=ROOT / "outputs" / "ettm1_to_ett_val_calibrated_transfer" / "best_by_val.csv")
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "calibration_capacity_ablation")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--targets", nargs="+", default=["ETTh1", "ETTh2", "ETTm2"])
    ap.add_argument("--horizons", type=int, nargs="+", default=[96, 192, 336, 720])
    args = ap.parse_args()

    transfer_best = pd.read_csv(args.transfer_best)
    transfer_best["horizon"] = transfer_best["horizon"].astype(int)
    transfer_best = transfer_best.set_index(["target", "horizon"])
    args.out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    rows: list[dict[str, Any]] = []
    for target in args.targets:
        for horizon in args.horizons:
            print(f"ablation {target} H{horizon}", flush=True)
            self_ctx = load_self_context(target, int(horizon), device)
            self_pred_val, self_true_val = collect_self_predictions(self_ctx, split="val", batch_size=args.batch_size)
            self_pred_test, self_true_test = collect_self_predictions(self_ctx, split="test", batch_size=args.batch_size)
            self_raw_mse, self_raw_mae = metrics(self_pred_test, self_true_test)
            rows.extend(
                eval_forms(
                    target=target,
                    horizon=int(horizon),
                    model_kind="target_self",
                    route_name="self_clusters",
                    pred_val=self_pred_val,
                    true_val=self_true_val,
                    pred_test=self_pred_test,
                    true_test=self_true_test,
                    target_self_raw_mse=self_raw_mse,
                    target_self_raw_mae=self_raw_mae,
                )
            )

            best = transfer_best.loc[(target, int(horizon))]
            route = tuple(int(v) for v in json.loads(str(best["route"])))
            cfg = read_yaml(pair_dir(target, int(horizon)) / "base_config.yaml")
            cfg.setdefault("exp", {})["device"] = args.device
            transfer_ctx = load_context(cfg, device)
            transfer_pred_val, transfer_true_val = collect_predictions(
                transfer_ctx,
                route,
                split="val",
                batch_size=args.batch_size,
            )
            transfer_pred_test, transfer_true_test = collect_predictions(
                transfer_ctx,
                route,
                split="test",
                batch_size=args.batch_size,
            )
            rows.extend(
                eval_forms(
                    target=target,
                    horizon=int(horizon),
                    model_kind="transfer",
                    route_name=str(best["route_name"]),
                    pred_val=transfer_pred_val,
                    true_val=transfer_true_val,
                    pred_test=transfer_pred_test,
                    true_test=transfer_true_test,
                    target_self_raw_mse=self_raw_mse,
                    target_self_raw_mae=self_raw_mae,
                )
            )
            write_rows(args.out_root / "calibration_capacity_ablation.csv", rows)

    write_rows(args.out_root / "calibration_capacity_ablation.csv", rows)
    make_plots(args.out_root, rows)
    summary = {
        "forms": {
            "none": "y' = yhat",
            "global_affine": "y' = a*yhat + b, with scalar a,b fitted on validation over all windows/channels/steps",
            "channel_affine": "y'[c,h] = a[c]*yhat[c,h] + b[c], with a,b fitted per channel",
            "channel_step_affine": "y'[c,h] = a[c,h]*yhat[c,h] + b[c,h], fitted per channel and prediction step",
        },
        "selection": "route fixed to each previously selected transfer route; calibration form is ablated with shrink=1.0",
    }
    with (args.out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    out = pd.DataFrame(rows)
    print(out[out["model_kind"] == "transfer"][[
        "target",
        "horizon",
        "calibration_form",
        "num_parameters",
        "test_mse",
        "pct_vs_target_self_raw",
    ]].to_string(index=False))
    print(args.out_root / "calibration_capacity_ablation.csv")


if __name__ == "__main__":
    main()
