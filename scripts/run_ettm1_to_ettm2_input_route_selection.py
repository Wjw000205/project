from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.cluster_predictor import build_cluster_predictor
from src.transfer import (
    _build_moe_modules,
    _df_to_tensor,
    _infer_step_minutes,
    _load_residual_scales,
    _load_source_summary,
    _predict_with_optional_residual,
    _resample_df,
)
from src.utils.cluster_memory import load_cluster_checkpoint, load_cluster_memory
from src.utils.metrics import accumulate_channel_errors, mse_mae_from_sums


FIELDS = [
    "status",
    "name",
    "eval_split",
    "mse",
    "mae",
    "align",
    "input_len",
    "score_len",
    "max_lag",
    "route_counts",
    "route_margin_mean",
    "route_margin_min",
    "out_dir",
    "error",
]


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})


def zscore_last(x: torch.Tensor, dim: int) -> torch.Tensor:
    x0 = x - x.mean(dim=dim, keepdim=True)
    return x0 / x0.std(dim=dim, keepdim=True).clamp_min(1.0e-6)


def input_route_scores(
    x_bcl: torch.Tensor,
    prototypes_kt: torch.Tensor,
    *,
    align: str,
    score_len: int,
    max_lag: int,
) -> torch.Tensor:
    score_len = min(int(score_len), int(x_bcl.shape[-1]), int(prototypes_kt.shape[-1]))
    max_lag = max(0, min(int(max_lag), score_len - 2))
    x = x_bcl[..., -score_len:]
    proto = prototypes_kt[:, -score_len:] if align == "tail" else prototypes_kt[:, :score_len]
    best = None
    for tau in range(-max_lag, max_lag + 1):
        if tau >= 0:
            xs = x[..., tau:]
            ps = proto[:, : score_len - tau]
        else:
            xs = x[..., : score_len + tau]
            ps = proto[:, -tau:]
        n = int(xs.shape[-1])
        if n <= 1:
            continue
        xz = zscore_last(xs, dim=-1)
        pz = zscore_last(ps, dim=-1)
        corr = torch.einsum("bcl,kl->bck", xz, pz) / max(n - 1, 1)
        corr = corr.clamp(-1.0, 1.0)
        best = corr if best is None else torch.maximum(best, corr)
    if best is None:
        best = x_bcl.new_zeros((x_bcl.shape[0], x_bcl.shape[1], prototypes_kt.shape[0]))
    return best


def make_variants() -> list[dict[str, Any]]:
    variants = []
    for align in ["head", "tail"]:
        for score_len in [96, 192, 336]:
            for max_lag in [0, 24, 48, 96]:
                if max_lag >= score_len:
                    continue
                variants.append(
                    {
                        "name": f"input_{align}_l{score_len}_lag{max_lag}",
                        "align": align,
                        "score_len": score_len,
                        "max_lag": max_lag,
                    }
                )
    return variants


def load_context(cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    source_cfg = cfg["source"]
    memory = load_cluster_memory(str(source_cfg["memory_path"]), device=device)
    ckpt = load_cluster_checkpoint(str(source_cfg["checkpoint_path"]), device=device)
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
    gate, pred_residual, penalty_names = _build_moe_modules(ckpt, meta, device)
    if not bool(cfg.get("transfer", {}).get("use_pred_residual", True)):
        pred_residual = None

    raw_df = pd.read_csv(cfg["data"]["csv_path"])
    max_rows = int(cfg.get("data", {}).get("max_rows", 0) or 0)
    if max_rows > 0:
        raw_df = raw_df.iloc[:max_rows].copy()
    date_col = raw_df.columns[int(cfg["data"].get("date_col", 0))]
    resample_cfg = cfg.get("transfer", {}).get("resample", {})
    if bool(resample_cfg.get("enable", False)):
        target_step_min = resample_cfg.get("target_step_minutes", None)
        if target_step_min is None:
            target_step_min = cfg.get("source", {}).get("step_minutes", None)
        if target_step_min is None:
            target_step_min = _infer_step_minutes(raw_df, date_col)
        target_step_min = int(target_step_min) if target_step_min is not None else 0
        method = resample_cfg.get("method", None)
        if method is None:
            cur_step = _infer_step_minutes(raw_df, date_col)
            method = "mean" if cur_step > 0 and target_step_min > cur_step else "linear"
        raw_df = _resample_df(raw_df, date_col, target_step_min, str(method).lower())
    data_tc, channel_names = _df_to_tensor(raw_df, date_col)
    data_tc = data_tc.to(device)
    T, C = data_tc.shape
    tr = float(cfg["data"]["train_ratio"])
    vr = float(cfg["data"]["val_ratio"])
    t_train = int(T * tr)
    t_val = int(T * (tr + vr))
    norm_cfg = cfg.get("normalize", {}) or {}
    if bool(norm_cfg.get("global_zscore", True)):
        if bool(norm_cfg.get("train_only", True)):
            fit_seg = data_tc[:t_train]
        else:
            fit_seg = data_tc
        mean_c = fit_seg.mean(dim=0, keepdim=True)
        std_c = fit_seg.std(dim=0, keepdim=True).clamp_min(1.0e-6)
        data_tc = (data_tc - mean_c) / std_c
    source_summary = _load_source_summary(source_cfg, str(source_cfg["checkpoint_path"]))
    residual_scale_c = (
        _load_residual_scales(source_summary, channel_names, device=device)
        if pred_residual is not None
        else None
    )
    return {
        "model": model,
        "gate": gate,
        "pred_residual": pred_residual,
        "penalty_names": penalty_names,
        "meta": meta,
        "prototypes_kt": memory["prototypes_kt"].to(device),
        "data_tc": data_tc,
        "channel_names": channel_names,
        "input_len": input_len,
        "pred_len": pred_len,
        "K": int(meta["K"]),
        "C": C,
        "T": T,
        "t_train": t_train,
        "t_val": t_val,
        "past_context": bool(cfg.get("window", {}).get("past_context", False)),
        "data_max_rows": max_rows,
        "residual_scale_c": residual_scale_c,
    }


def predict_dynamic(
    ctx: dict[str, Any],
    x: torch.Tensor,
    route_bc: torch.Tensor,
) -> torch.Tensor:
    K = int(ctx["K"])
    C = int(ctx["C"])
    preds = []
    for k in range(K):
        cid_c = torch.full((C,), k, device=x.device, dtype=torch.long)
        _, yhat_k = _predict_with_optional_residual(
            model=ctx["model"],
            gate=ctx["gate"],
            pred_residual=ctx["pred_residual"],
            x=x,
            cluster_id_c=cid_c,
            meta=ctx["meta"],
            residual_scale_c=ctx["residual_scale_c"],
        )
        preds.append(yhat_k)
    y_bckh = torch.stack(preds, dim=2)
    w_bck = F.one_hot(route_bc.to(torch.long), num_classes=K).to(dtype=x.dtype)
    return (y_bckh * w_bck.unsqueeze(-1)).sum(dim=2)


def evaluate(ctx: dict[str, Any], variant: dict[str, Any], eval_split: str, out_dir: Path, batch_size: int) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    L = int(ctx["input_len"])
    H = int(ctx["pred_len"])
    C = int(ctx["C"])
    K = int(ctx["K"])
    data_tc = ctx["data_tc"]
    eval_start, eval_end = (ctx["t_train"], ctx["t_val"]) if eval_split == "val" else (ctx["t_val"], ctx["T"])
    eval_seg = data_tc[eval_start:eval_end]
    n_windows = int(eval_seg.shape[0] - L - H + 1)
    if n_windows <= 0:
        raise ValueError(f"No {eval_split} windows available.")

    se_c = torch.zeros(C, device=data_tc.device)
    ae_c = torch.zeros(C, device=data_tc.device)
    route_count = torch.zeros(K, device=data_tc.device)
    margins: list[torch.Tensor] = []
    denom = 0
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
            scores = input_route_scores(
                x,
                ctx["prototypes_kt"],
                align=str(variant["align"]),
                score_len=int(variant["score_len"]),
                max_lag=int(variant["max_lag"]),
            )
            top2 = torch.topk(scores, k=min(2, K), dim=-1).values
            if top2.shape[-1] == 2:
                margins.append((top2[..., 0] - top2[..., 1]).reshape(-1).detach().cpu())
            route_bc = scores.argmax(dim=-1)
            route_count += torch.bincount(route_bc.reshape(-1), minlength=K).to(route_count)
            yhat = predict_dynamic(ctx, x, route_bc)
            accumulate_channel_errors(se_c, ae_c, yhat, y)
            denom += int(x.shape[0] * H)
    mse_c, mae_c = mse_mae_from_sums(se_c, ae_c, denom)
    metrics = pd.DataFrame(
        {
            "channel": ctx["channel_names"],
            "MAE": mae_c.detach().cpu().numpy(),
            "MSE": mse_c.detach().cpu().numpy(),
        }
    )
    metrics.to_csv(out_dir / f"{eval_split}_metrics.csv", index=False)
    margin_cat = torch.cat(margins) if margins else torch.empty(0)
    summary = {
        "avg_mse": float(metrics["MSE"].mean()),
        "avg_mae": float(metrics["MAE"].mean()),
        "eval_split": eval_split,
        "eval_start_index": int(eval_start),
        "eval_end_index": int(eval_end),
        "num_eval_windows": int(n_windows),
        "route_mode": "input_dynamic",
        "route_uses_future_y": False,
        "normalize_train_only": True,
        "route_counts": {str(k): int(v) for k, v in enumerate(route_count.detach().cpu().tolist())},
        "route_margin_mean": float(margin_cat.mean().item()) if margin_cat.numel() else None,
        "route_margin_min": float(margin_cat.min().item()) if margin_cat.numel() else None,
        "variant": dict(variant),
    }
    with (out_dir / "transfer_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def row_from_summary(variant: dict[str, Any], eval_split: str, summary: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    return {
        "status": "ok",
        "name": variant["name"],
        "eval_split": eval_split,
        "mse": summary["avg_mse"],
        "mae": summary["avg_mae"],
        "align": variant["align"],
        "input_len": 336,
        "score_len": variant["score_len"],
        "max_lag": variant["max_lag"],
        "route_counts": json.dumps(summary["route_counts"], ensure_ascii=False),
        "route_margin_mean": summary.get("route_margin_mean"),
        "route_margin_min": summary.get("route_margin_min"),
        "out_dir": str(out_dir),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "ETTm1ToETTm2.yaml")
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "ettm1_to_ettm2_input_route_selection")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    if args.device is not None:
        cfg.setdefault("exp", {})["device"] = args.device
    device = torch.device(str(cfg.get("exp", {}).get("device", "cuda:0")))
    ctx = load_context(cfg, device)

    rows: list[dict[str, Any]] = []
    for variant in make_variants():
        out_dir = args.out_root / "val_runs" / variant["name"]
        try:
            summary = evaluate(ctx, variant, "val", out_dir, args.batch_size)
            row = row_from_summary(variant, "val", summary, out_dir)
            print(f"[val ok] {variant['name']} mse={summary['avg_mse']:.6f} mae={summary['avg_mae']:.6f}")
        except Exception as exc:
            row = {**{k: "" for k in FIELDS}, "status": "failed", "name": variant["name"], "eval_split": "val", "error": str(exc)}
            print(f"[val failed] {variant['name']}: {exc}")
        rows.append(row)
        write_rows(args.out_root / "val_results.csv", rows)

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    ok_rows.sort(key=lambda r: (float(r["mse"]), float(r["mae"])))
    write_rows(args.out_root / "val_results_ranked.csv", ok_rows)
    if not ok_rows:
        raise RuntimeError("No input-route val runs completed.")
    winner_name = ok_rows[0]["name"]
    winner = next(v for v in make_variants() if v["name"] == winner_name)
    test_out = args.out_root / "test_winner" / winner_name
    test_summary = evaluate(ctx, winner, "test", test_out, args.batch_size)
    test_row = row_from_summary(winner, "test", test_summary, test_out)
    write_rows(args.out_root / "selected_test.csv", [test_row])
    final = {
        "selection_metric": "val.avg_mse",
        "selected_name": winner_name,
        "selected_val_mse": ok_rows[0]["mse"],
        "selected_val_mae": ok_rows[0]["mae"],
        "selected_test_mse": test_summary["avg_mse"],
        "selected_test_mae": test_summary["avg_mae"],
        "selected_out_dir": str(test_out),
        "route_uses_future_y": False,
    }
    with (args.out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
