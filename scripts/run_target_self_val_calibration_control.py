from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_ettm1_to_ettm2_val_calibrated_transfer import fit_apply, metrics  # noqa: E402
from src.models.cluster_predictor import build_cluster_predictor  # noqa: E402
from src.transfer import (  # noqa: E402
    _build_moe_modules,
    _df_to_tensor,
    _load_residual_scales,
    _predict_with_optional_residual,
)
from src.utils.cluster_memory import load_cluster_checkpoint  # noqa: E402


FIELDS = [
    "target",
    "horizon",
    "self_raw_mse",
    "self_raw_mae",
    "self_val_cal_mse",
    "self_val_cal_mae",
    "self_calibration",
    "self_shrink",
    "self_val_mse",
    "self_val_mae",
    "transfer_val_cal_mse",
    "transfer_val_cal_mae",
    "transfer_route",
    "transfer_calibration",
    "transfer_shrink",
    "transfer_vs_self_raw_pct",
    "transfer_vs_self_cal_pct",
    "self_cal_gain_pct",
    "checkpoint_path",
]


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})


def candidate_run_dirs(target: str, horizon: int) -> list[Path]:
    return [
        ROOT / "outputs" / "ett_global_h96_param_base" / "runs" / target / f"pred_{horizon}",
        ROOT / "outputs" / "ett_horizon_sweep" / "runs" / target / f"pred_{horizon}",
    ]


def find_self_run(target: str, horizon: int) -> Path:
    for run_dir in candidate_run_dirs(target, horizon):
        if (run_dir / "best_checkpoint.pt").exists() and (run_dir / "run_summary.json").exists():
            return run_dir
    tried = "\n".join(str(p) for p in candidate_run_dirs(target, horizon))
    raise FileNotFoundError(f"Missing self-train checkpoint for {target} H{horizon}. Tried:\n{tried}")


def load_self_context(target: str, horizon: int, device: torch.device) -> dict[str, Any]:
    run_dir = find_self_run(target, horizon)
    summary = read_json(run_dir / "run_summary.json")
    cfg_path = Path(str(summary["config_path"]))
    cfg = read_yaml(cfg_path)

    ckpt = load_cluster_checkpoint(str(run_dir / "best_checkpoint.pt"), device=device)
    meta = ckpt["meta"]
    input_len = int(meta["input_len"])
    pred_len = int(meta["pred_len"])
    model = build_cluster_predictor(
        num_clusters=int(meta["K"]),
        input_len=input_len,
        pred_len=pred_len,
        model_cfg=meta["model_cfg"],
        num_channels=meta.get("num_channels"),
        cluster_id_c=meta.get("cluster_id_c"),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    gate, pred_residual, _ = _build_moe_modules(ckpt, meta, device)

    raw_df = pd.read_csv(cfg["data"]["csv_path"])
    max_rows = int(cfg.get("data", {}).get("max_rows", 0) or 0)
    if max_rows > 0:
        raw_df = raw_df.iloc[:max_rows].copy()
    date_col = raw_df.columns[int(cfg["data"].get("date_col", 0))]
    data_tc, channel_names = _df_to_tensor(raw_df, date_col)
    data_tc = data_tc.to(device)
    t_total = int(data_tc.shape[0])
    t_train = int(t_total * float(cfg["data"]["train_ratio"]))
    t_val = int(t_total * (float(cfg["data"]["train_ratio"]) + float(cfg["data"]["val_ratio"])))

    norm_cfg = cfg.get("normalize", {}) or {}
    if bool(norm_cfg.get("global_zscore", True)):
        fit_seg = data_tc[:t_train] if bool(norm_cfg.get("train_only", True)) else data_tc
        mean_c = fit_seg.mean(dim=0, keepdim=True)
        std_c = fit_seg.std(dim=0, keepdim=True).clamp_min(1.0e-6)
        data_tc = (data_tc - mean_c) / std_c

    cluster_id_c = meta.get("cluster_id_c")
    if cluster_id_c is None:
        raise ValueError(f"{run_dir / 'best_checkpoint.pt'} has no meta.cluster_id_c")
    cluster_id_c = torch.as_tensor(cluster_id_c, device=device, dtype=torch.long)
    residual_scale_c = _load_residual_scales(summary, list(channel_names), device) if pred_residual is not None else None
    return {
        "target": target,
        "horizon": horizon,
        "run_dir": run_dir,
        "summary": summary,
        "model": model,
        "gate": gate,
        "pred_residual": pred_residual,
        "meta": meta,
        "data_tc": data_tc,
        "cluster_id_c": cluster_id_c,
        "input_len": input_len,
        "pred_len": pred_len,
        "t_train": t_train,
        "t_val": t_val,
        "t_total": t_total,
        "past_context": bool(cfg.get("window", {}).get("past_context", False)),
        "residual_scale_c": residual_scale_c,
    }


def collect_self_predictions(ctx: dict[str, Any], *, split: str, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    input_len = int(ctx["input_len"])
    pred_len = int(ctx["pred_len"])
    data_tc = ctx["data_tc"]
    label_start, label_end = (ctx["t_train"], ctx["t_val"]) if split == "val" else (ctx["t_val"], ctx["t_total"])
    eval_start = max(0, int(label_start) - input_len) if bool(ctx.get("past_context", False)) else int(label_start)
    eval_seg = data_tc[eval_start:label_end]
    n_windows = int(eval_seg.shape[0] - input_len - pred_len + 1)
    if n_windows <= 0:
        raise ValueError(f"No {split} windows for {ctx['target']} H{ctx['horizon']}")

    preds: list[torch.Tensor] = []
    trues: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, n_windows, batch_size):
            end = min(start + batch_size, n_windows)
            xs = []
            ys = []
            for i in range(start, end):
                win = eval_seg[i : i + input_len + pred_len]
                xs.append(win[:input_len].T)
                ys.append(win[input_len:].T)
            x = torch.stack(xs, dim=0)
            y = torch.stack(ys, dim=0)
            _, yhat = _predict_with_optional_residual(
                model=ctx["model"],
                gate=ctx["gate"],
                pred_residual=ctx["pred_residual"],
                x=x,
                cluster_id_c=ctx["cluster_id_c"],
                meta=ctx["meta"],
                residual_scale_c=ctx["residual_scale_c"],
            )
            preds.append(yhat.detach().cpu())
            trues.append(y.detach().cpu())
    return torch.cat(preds, dim=0), torch.cat(trues, dim=0)


def best_self_calibration(
    pred_val: torch.Tensor,
    true_val: torch.Tensor,
    pred_test: torch.Tensor,
    true_test: torch.Tensor,
) -> dict[str, Any]:
    methods = ["none", "bias_channel", "bias_channel_horizon", "affine_channel", "affine_channel_horizon"]
    shrinks = [0.25, 0.5, 0.75, 1.0]
    rows: list[dict[str, Any]] = []
    for method in methods:
        method_shrinks = [1.0] if method == "none" else shrinks
        for shrink in method_shrinks:
            cal_val, cal_test = fit_apply(pred_val, true_val, pred_test, method=method, shrink=shrink)
            val_mse, val_mae = metrics(cal_val, true_val)
            test_mse, test_mae = metrics(cal_test, true_test)
            rows.append(
                {
                    "calibration": method,
                    "shrink": shrink,
                    "val_mse": val_mse,
                    "val_mae": val_mae,
                    "test_mse": test_mse,
                    "test_mae": test_mae,
                }
            )
    return min(rows, key=lambda r: (float(r["val_mse"]), float(r["val_mae"])))


def make_plots(out_root: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    df = pd.DataFrame(rows).sort_values(["target", "horizon"])
    cases = [f"{r.target}\nH{int(r.horizon)}" for r in df.itertuples()]
    x = np.arange(len(df))
    width = 0.24

    fig, ax = plt.subplots(figsize=(12.2, 4.8), dpi=220)
    ax.bar(x - width, df["self_raw_mse"], width=width, label="self raw", color="#4C78A8")
    ax.bar(x, df["self_val_cal_mse"], width=width, label="self + val calibration", color="#72A24D")
    ax.bar(x + width, df["transfer_val_cal_mse"], width=width, label="transfer + val calibration", color="#D65F45")
    ax.set_xticks(x)
    ax.set_xticklabels(cases, fontsize=8)
    ax.set_ylabel("Test MSE")
    ax.set_title("Fair control: target self-train also gets validation calibration")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_root / "self_vs_transfer_val_calibrated_mse.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12.2, 4.2), dpi=220)
    vals = df["transfer_vs_self_cal_pct"].to_numpy(float)
    colors = ["#2F7D52" if v < 0 else "#B04A3A" for v in vals]
    ax.bar(x, vals, color=colors, width=0.62)
    ax.axhline(0, color="#222222", lw=1.0)
    ax.axhline(10, color="#B04A3A", lw=1.0, ls=(0, (4, 3)))
    ax.set_xticks(x)
    ax.set_xticklabels(cases, fontsize=8)
    ax.set_ylabel("Transfer vs calibrated self (%)")
    ax.set_title("Does transfer still beat self after the same validation calibration?")
    ax.grid(axis="y", alpha=0.25)
    for xi, v in zip(x, vals):
        ax.text(xi, v + (0.8 if v >= 0 else -0.8), f"{v:+.1f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_root / "transfer_vs_calibrated_self_pct.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transfer-best", type=Path, default=ROOT / "outputs" / "ettm1_to_ett_val_calibrated_transfer" / "best_by_val.csv")
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "target_self_val_calibration_control")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--targets", nargs="+", default=["ETTh1", "ETTh2", "ETTm2"])
    ap.add_argument("--horizons", type=int, nargs="+", default=[96, 192, 336, 720])
    args = ap.parse_args()

    transfer = pd.read_csv(args.transfer_best)
    transfer["horizon"] = transfer["horizon"].astype(int)
    transfer = transfer.set_index(["target", "horizon"])
    args.out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    device = torch.device(args.device)
    for target in args.targets:
        for horizon in args.horizons:
            print(f"self-control {target} H{horizon}", flush=True)
            ctx = load_self_context(target, int(horizon), device)
            pred_val, true_val = collect_self_predictions(ctx, split="val", batch_size=args.batch_size)
            pred_test, true_test = collect_self_predictions(ctx, split="test", batch_size=args.batch_size)
            self_raw_mse, self_raw_mae = metrics(pred_test, true_test)
            best = best_self_calibration(pred_val, true_val, pred_test, true_test)
            tr = transfer.loc[(target, int(horizon))]
            transfer_mse = float(tr["test_mse"])
            transfer_mae = float(tr["test_mae"])
            row = {
                "target": target,
                "horizon": int(horizon),
                "self_raw_mse": self_raw_mse,
                "self_raw_mae": self_raw_mae,
                "self_val_cal_mse": float(best["test_mse"]),
                "self_val_cal_mae": float(best["test_mae"]),
                "self_calibration": best["calibration"],
                "self_shrink": best["shrink"],
                "self_val_mse": float(best["val_mse"]),
                "self_val_mae": float(best["val_mae"]),
                "transfer_val_cal_mse": transfer_mse,
                "transfer_val_cal_mae": transfer_mae,
                "transfer_route": tr["route_name"],
                "transfer_calibration": tr["calibration"],
                "transfer_shrink": tr["shrink"],
                "transfer_vs_self_raw_pct": (transfer_mse / max(self_raw_mse, 1.0e-12) - 1.0) * 100.0,
                "transfer_vs_self_cal_pct": (transfer_mse / max(float(best["test_mse"]), 1.0e-12) - 1.0) * 100.0,
                "self_cal_gain_pct": (float(best["test_mse"]) / max(self_raw_mse, 1.0e-12) - 1.0) * 100.0,
                "checkpoint_path": str(ctx["run_dir"] / "best_checkpoint.pt"),
            }
            rows.append(row)
            write_rows(args.out_root / "self_vs_transfer_control.csv", rows)

    write_rows(args.out_root / "self_vs_transfer_control.csv", rows)
    make_plots(args.out_root, rows)
    with (args.out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"selection": "target self calibration selected by validation MSE", "rows": rows}, f, ensure_ascii=False, indent=2)
    print(pd.DataFrame(rows)[[
        "target",
        "horizon",
        "self_raw_mse",
        "self_val_cal_mse",
        "transfer_val_cal_mse",
        "transfer_vs_self_raw_pct",
        "transfer_vs_self_cal_pct",
        "self_calibration",
        "self_shrink",
    ]].to_string(index=False))
    print(args.out_root / "self_vs_transfer_control.csv")


if __name__ == "__main__":
    main()
