from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_ettm1_to_ettm2_input_route_selection import load_context, read_yaml  # noqa: E402
from src.transfer import _predict_with_optional_residual  # noqa: E402


FIELDS = [
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
    "route",
]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})


def route_from_assignment(path: Path) -> tuple[int, ...]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return tuple(int(row["cluster_id"]) for row in csv.DictReader(f))


def route_from_summary(path: Path) -> tuple[int, ...]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return tuple(int(v) for v in payload["selected_route"])


def target_self_metrics(horizon: int) -> tuple[float, float]:
    path = ROOT / "outputs" / "ett_horizon_sweep" / "runs" / "ETTm2" / f"pred_{horizon}" / "run_summary.json"
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return float(payload["test"]["avg_mse"]), float(payload["test"]["avg_mae"])


def collect_predictions(
    ctx: dict[str, Any],
    route: tuple[int, ...],
    *,
    split: str,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    L = int(ctx["input_len"])
    H = int(ctx["pred_len"])
    data_tc = ctx["data_tc"]
    eval_label_start, eval_end = (ctx["t_train"], ctx["t_val"]) if split == "val" else (ctx["t_val"], ctx["T"])
    eval_start = max(0, int(eval_label_start) - L) if bool(ctx.get("past_context", False)) else int(eval_label_start)
    eval_seg = data_tc[eval_start:eval_end]
    n_windows = int(eval_seg.shape[0] - L - H + 1)
    if n_windows <= 0:
        raise ValueError(f"No {split} windows available.")

    cluster_id_c = torch.tensor(route, device=data_tc.device, dtype=torch.long)
    preds: list[torch.Tensor] = []
    trues: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, n_windows, batch_size):
            end = min(start + batch_size, n_windows)
            xs = []
            ys = []
            for i in range(start, end):
                win = eval_seg[i : i + L + H]
                xs.append(win[:L].T)
                ys.append(win[L:].T)
            x = torch.stack(xs, dim=0)
            y = torch.stack(ys, dim=0)
            _, yhat = _predict_with_optional_residual(
                model=ctx["model"],
                gate=ctx["gate"],
                pred_residual=ctx["pred_residual"],
                x=x,
                cluster_id_c=cluster_id_c,
                meta=ctx["meta"],
                residual_scale_c=ctx["residual_scale_c"],
            )
            preds.append(yhat.detach().cpu())
            trues.append(y.detach().cpu())
    return torch.cat(preds, dim=0), torch.cat(trues, dim=0)


def metrics(pred: torch.Tensor, true: torch.Tensor) -> tuple[float, float]:
    return float((pred - true).pow(2).mean().item()), float((pred - true).abs().mean().item())


def fit_apply(
    pred_val: torch.Tensor,
    true_val: torch.Tensor,
    pred_test: torch.Tensor,
    *,
    method: str,
    shrink: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    eps = 1.0e-6
    if method == "none":
        cal_val = pred_val
        cal_test = pred_test
    elif method == "bias_channel":
        bias = (true_val - pred_val).mean(dim=(0, 2), keepdim=True)
        cal_val = pred_val + bias
        cal_test = pred_test + bias
    elif method == "bias_channel_horizon":
        bias = (true_val - pred_val).mean(dim=0, keepdim=True)
        cal_val = pred_val + bias
        cal_test = pred_test + bias
    elif method == "affine_channel":
        x = pred_val
        y = true_val
        mx = x.mean(dim=(0, 2), keepdim=True)
        my = y.mean(dim=(0, 2), keepdim=True)
        vx = (x - mx).pow(2).mean(dim=(0, 2), keepdim=True).clamp_min(eps)
        cov = ((x - mx) * (y - my)).mean(dim=(0, 2), keepdim=True)
        a = (cov / vx).clamp(0.5, 1.5)
        b = my - a * mx
        cal_val = a * pred_val + b
        cal_test = a * pred_test + b
    elif method == "affine_channel_horizon":
        x = pred_val
        y = true_val
        mx = x.mean(dim=0, keepdim=True)
        my = y.mean(dim=0, keepdim=True)
        vx = (x - mx).pow(2).mean(dim=0, keepdim=True).clamp_min(eps)
        cov = ((x - mx) * (y - my)).mean(dim=0, keepdim=True)
        a = (cov / vx).clamp(0.5, 1.5)
        b = my - a * mx
        cal_val = a * pred_val + b
        cal_test = a * pred_test + b
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    shrink = float(shrink)
    out_val = pred_val + shrink * (cal_val - pred_val)
    out_test = pred_test + shrink * (cal_test - pred_test)
    return out_val, out_test


def route_candidates(horizon: int) -> dict[str, tuple[int, ...]]:
    base = ROOT / "outputs" / "ettm1_other_ett_horizon_transfer" / "ETTm1_to_ETTm2" / f"pred_{horizon}"
    static_route = route_from_assignment(base / "direct_transfer" / "cluster_assignment.csv")
    selected_route = route_from_summary(base / "val_loss_selection" / "summary.json")
    k_count = 3
    out = {
        "static_corr_train": static_route,
        "val_route": selected_route,
    }
    for k in range(k_count):
        out[f"all_{k}"] = tuple([k] * len(static_route))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "ettm1_to_ettm2_val_calibrated_transfer")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--horizons", type=int, nargs="+", default=[192, 336])
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    methods = [
        "none",
        "bias_channel",
        "bias_channel_horizon",
        "affine_channel",
        "affine_channel_horizon",
    ]
    shrinks = [0.25, 0.5, 0.75, 1.0]
    rows: list[dict[str, Any]] = []

    for horizon in args.horizons:
        target_self_mse, target_self_mae = target_self_metrics(horizon)
        cfg_path = ROOT / "outputs" / "ettm1_other_ett_horizon_transfer" / "ETTm1_to_ETTm2" / f"pred_{horizon}" / "base_config.yaml"
        base_cfg = read_yaml(cfg_path)
        for residual_mode in ["full_moe_residual", "base_only"]:
            cfg = json.loads(json.dumps(base_cfg))
            cfg.setdefault("exp", {})["device"] = args.device
            if residual_mode == "base_only":
                cfg.setdefault("transfer", {})["use_pred_residual"] = False
            ctx = load_context(cfg, torch.device(args.device))
            for route_name, route in route_candidates(horizon).items():
                print(f"H{horizon} {residual_mode} {route_name}", flush=True)
                pred_val, true_val = collect_predictions(ctx, route, split="val", batch_size=args.batch_size)
                pred_test, true_test = collect_predictions(ctx, route, split="test", batch_size=args.batch_size)
                for method in methods:
                    method_shrinks = [1.0] if method == "none" else shrinks
                    for shrink in method_shrinks:
                        cal_val, cal_test = fit_apply(pred_val, true_val, pred_test, method=method, shrink=shrink)
                        val_mse, val_mae = metrics(cal_val, true_val)
                        test_mse, test_mae = metrics(cal_test, true_test)
                        row = {
                            "horizon": horizon,
                            "residual_mode": residual_mode,
                            "route_name": route_name,
                            "calibration": method,
                            "shrink": shrink,
                            "val_mse": val_mse,
                            "val_mae": val_mae,
                            "test_mse": test_mse,
                            "test_mae": test_mae,
                            "target_self_mse": target_self_mse,
                            "target_self_mae": target_self_mae,
                            "pct_vs_target_self": (test_mse / target_self_mse - 1.0) * 100.0,
                            "route": json.dumps(list(route)),
                        }
                        rows.append(row)
                write_rows(args.out_root / "all_results.csv", rows)

    ok = sorted(rows, key=lambda r: (int(r["horizon"]), float(r["val_mse"]), float(r["val_mae"])))
    best_by_h: list[dict[str, Any]] = []
    for horizon in args.horizons:
        winner = next(row for row in ok if int(row["horizon"]) == horizon)
        best_by_h.append(winner)
    write_rows(args.out_root / "best_by_val.csv", best_by_h)
    write_rows(args.out_root / "all_results.csv", rows)
    with (args.out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"selection": "lowest validation MSE per horizon", "best_by_val": best_by_h}, f, ensure_ascii=False, indent=2)
    print(json.dumps(best_by_h, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
