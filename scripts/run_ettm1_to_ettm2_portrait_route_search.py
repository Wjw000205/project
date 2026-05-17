from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_ettm1_to_ettm2_input_route_selection import load_context, read_yaml  # noqa: E402
from scripts.run_ettm1_to_ettm2_input_fallback_selection import static_route_from_train  # noqa: E402
from scripts.run_ettm1_to_ettm2_val_loss_route_selection import evaluate_route, make_fixed_cfg, write_yaml  # noqa: E402
from src.utils.cluster_memory import assign_channels_by_cycle_template  # noqa: E402


HORIZONS = [192, 336]
CHANNELS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

RESULT_FIELDS = [
    "horizon",
    "candidate",
    "route",
    "corr_weight",
    "stat_weight",
    "dyn_weight",
    "cycle_weight",
    "name_weight",
    "val_mse",
    "val_mae",
    "test_mse",
    "test_mae",
    "target_self_mse",
    "target_self_mae",
    "source_test_mse",
    "source_test_mae",
]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in RESULT_FIELDS})


def _zscore_1d(x: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    x = x.to(torch.float32)
    return (x - x.mean()) / x.std().clamp_min(eps)


def _safe_quantiles(x: torch.Tensor) -> torch.Tensor:
    qs = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=x.device)
    return torch.quantile(x.to(torch.float32), qs)


def _autocorr(x: torch.Tensor, lag: int) -> torch.Tensor:
    if int(x.numel()) <= lag + 2:
        return x.new_tensor(0.0)
    a = x[:-lag].to(torch.float32)
    b = x[lag:].to(torch.float32)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.pow(2).mean().sqrt() * b.pow(2).mean().sqrt()).clamp_min(1.0e-6)
    return ((a * b).mean() / denom).clamp(-1.0, 1.0)


def _cycle_template(x: torch.Tensor, period: int, bins: int) -> torch.Tensor:
    n = int(x.numel())
    if n == 0:
        return torch.zeros(bins, device=x.device)
    t = torch.arange(n, device=x.device)
    idx = torch.floor((torch.remainder(t, int(period)).to(torch.float32) / float(period)) * bins).to(torch.long)
    idx = idx.clamp(0, bins - 1)
    sums = torch.zeros(bins, device=x.device, dtype=torch.float32)
    cnt = torch.zeros(bins, device=x.device, dtype=torch.float32)
    xf = x.to(torch.float32)
    sums.scatter_add_(0, idx, xf)
    cnt.scatter_add_(0, idx, torch.ones_like(xf))
    tmpl = sums / cnt.clamp_min(1.0)
    tmpl = torch.where(cnt > 0, tmpl, xf.mean().expand_as(tmpl))
    return _zscore_1d(tmpl)


def _slope(x: torch.Tensor) -> torch.Tensor:
    n = int(x.numel())
    if n <= 2:
        return x.new_tensor(0.0)
    y = x.to(torch.float32)
    t = torch.linspace(-1.0, 1.0, n, device=x.device)
    y0 = y - y.mean()
    t0 = t - t.mean()
    return (t0 * y0).mean() / t0.pow(2).mean().clamp_min(1.0e-6)


def portrait_one(x: torch.Tensor) -> dict[str, torch.Tensor]:
    x = x.to(torch.float32)
    dx = x[1:] - x[:-1] if x.numel() > 1 else x.new_zeros(1)
    d2 = dx[1:] - dx[:-1] if dx.numel() > 1 else x.new_zeros(1)

    stat = torch.cat(
        [
            torch.stack(
                [
                    x.mean(),
                    x.std(),
                    x.min(),
                    x.max(),
                    x.max() - x.min(),
                    _slope(x),
                ]
            ),
            _safe_quantiles(x),
        ]
    )
    dyn = torch.cat(
        [
            torch.stack(
                [
                    dx.mean(),
                    dx.std(),
                    dx.abs().mean(),
                    torch.quantile(dx.abs(), 0.95),
                    d2.mean(),
                    d2.std(),
                    d2.abs().mean(),
                    torch.quantile(d2.abs(), 0.95),
                ]
            ),
            torch.stack([_autocorr(x, lag) for lag in [4, 16, 48, 96, 192, 336, 672]]),
        ]
    )
    cycle = torch.cat(
        [
            _cycle_template(x, period=96, bins=24),
            _cycle_template(x, period=672, bins=24),
        ]
    )
    return {"stat": stat, "dyn": dyn, "cycle": cycle}


def portraits(series_n_t: torch.Tensor) -> dict[str, torch.Tensor]:
    parts: dict[str, list[torch.Tensor]] = {"stat": [], "dyn": [], "cycle": []}
    for i in range(series_n_t.shape[0]):
        cur = portrait_one(series_n_t[i])
        for key in parts:
            parts[key].append(cur[key])
    return {key: torch.stack(vals, dim=0) for key, vals in parts.items()}


def standardized_pair_distance(target_c_d: torch.Tensor, source_k_d: torch.Tensor) -> torch.Tensor:
    all_x = torch.cat([target_c_d, source_k_d], dim=0)
    mean = all_x.mean(dim=0, keepdim=True)
    std = all_x.std(dim=0, keepdim=True).clamp_min(1.0e-6)
    t = (target_c_d - mean) / std
    s = (source_k_d - mean) / std
    return torch.cdist(t, s, p=2) / math.sqrt(max(1, t.shape[1]))


def name_distance(ctx: dict[str, Any]) -> torch.Tensor:
    source_cluster = ctx["meta"].get("cluster_id_c")
    if source_cluster is None:
        return torch.zeros((int(ctx["C"]), int(ctx["K"])), device=ctx["data_tc"].device)
    source_cluster = source_cluster.to(device=ctx["data_tc"].device, dtype=torch.long)
    channels = list(ctx["channel_names"])
    out = torch.ones((int(ctx["C"]), int(ctx["K"])), device=ctx["data_tc"].device)
    source_channels = CHANNELS[: int(source_cluster.numel())]
    for c, name in enumerate(channels):
        if name in source_channels:
            src_idx = source_channels.index(name)
            out[c, int(source_cluster[src_idx].item())] = 0.0
    return out


def load_source_train_ct(horizon: int, device: torch.device) -> torch.Tensor:
    cfg_path = ROOT / "outputs" / "ett_global_h96_param_base" / "configs" / f"ETTm1_pred_{horizon}.yaml"
    cfg = read_yaml(cfg_path)
    df = pd.read_csv(cfg["data"]["csv_path"])
    max_rows = int(cfg.get("data", {}).get("max_rows", 0) or 0)
    if max_rows > 0:
        df = df.iloc[:max_rows].copy()
    date_col = df.columns[int(cfg.get("data", {}).get("date_col", 0))]
    value_cols = [c for c in df.columns if c != date_col]
    data_tc = torch.tensor(df[value_cols].to_numpy(dtype="float32"), device=device)
    t_train = int(data_tc.shape[0] * float(cfg["data"]["train_ratio"]))
    norm_cfg = cfg.get("normalize", {}) or {}
    if bool(norm_cfg.get("global_zscore", True)):
        fit = data_tc[:t_train] if bool(norm_cfg.get("train_only", True)) else data_tc
        mean = fit.mean(dim=0, keepdim=True)
        std = fit.std(dim=0, keepdim=True).clamp_min(1.0e-6)
        data_tc = (data_tc - mean) / std
    return data_tc[:t_train].T.contiguous()


def channel_to_cluster_distance(
    target_c_d: torch.Tensor,
    source_ch_d: torch.Tensor,
    source_cluster_c: torch.Tensor,
    *,
    num_clusters: int,
    reduce: str,
) -> torch.Tensor:
    dist_c_s = standardized_pair_distance(target_c_d, source_ch_d)
    out = torch.zeros((target_c_d.shape[0], num_clusters), device=target_c_d.device)
    for k in range(num_clusters):
        vals = dist_c_s[:, source_cluster_c == k]
        if vals.numel() == 0:
            out[:, k] = dist_c_s.max() + 1.0
        elif reduce == "min":
            out[:, k] = vals.min(dim=1).values
        else:
            out[:, k] = vals.mean(dim=1)
    return out


def cycle_corr(ctx: dict[str, Any], cfg: dict[str, Any]) -> torch.Tensor:
    transfer_cfg = cfg.get("transfer", {}) or {}
    data_tc = ctx["data_tc"]
    route_data = data_tc[: int(ctx["t_train"])]
    step_min = 15.0
    period_min = transfer_cfg.get("period_min", None)
    period_max = transfer_cfg.get("period_max", None)
    if transfer_cfg.get("period_min_hours", None) is not None:
        period_min = int(round(float(transfer_cfg["period_min_hours"]) * 60.0 / step_min))
    if transfer_cfg.get("period_max_hours", None) is not None:
        period_max = int(round(float(transfer_cfg["period_max_hours"]) * 60.0 / step_min))
    _, corr_ck, _ = assign_channels_by_cycle_template(
        route_data,
        ctx["prototypes_kt"],
        phase_bins=int(transfer_cfg.get("phase_bins", 64)),
        period_min=period_min,
        period_max=period_max,
        align=str(transfer_cfg.get("corr_align", "head")),
        phase_max_shift=transfer_cfg.get("phase_max_shift", None),
    )
    return corr_ck


def candidate_routes(ctx: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    data_tc = ctx["data_tc"]
    t_train = int(ctx["t_train"])
    target_train_ct = data_tc[:t_train].T.contiguous()
    source_kt = ctx["prototypes_kt"].contiguous()

    target_p = portraits(target_train_ct)
    source_p = portraits(source_kt)
    source_train_ct = load_source_train_ct(int(ctx["pred_len"]), data_tc.device)
    source_ch_p = portraits(source_train_ct)
    source_cluster_c = ctx["meta"]["cluster_id_c"].to(device=data_tc.device, dtype=torch.long)
    dist_stat = standardized_pair_distance(target_p["stat"], source_p["stat"])
    dist_dyn = standardized_pair_distance(target_p["dyn"], source_p["dyn"])
    dist_cycle = standardized_pair_distance(target_p["cycle"], source_p["cycle"])
    dist_ch_mean_stat = channel_to_cluster_distance(
        target_p["stat"], source_ch_p["stat"], source_cluster_c, num_clusters=int(ctx["K"]), reduce="mean"
    )
    dist_ch_mean_dyn = channel_to_cluster_distance(
        target_p["dyn"], source_ch_p["dyn"], source_cluster_c, num_clusters=int(ctx["K"]), reduce="mean"
    )
    dist_ch_mean_cycle = channel_to_cluster_distance(
        target_p["cycle"], source_ch_p["cycle"], source_cluster_c, num_clusters=int(ctx["K"]), reduce="mean"
    )
    dist_ch_min_stat = channel_to_cluster_distance(
        target_p["stat"], source_ch_p["stat"], source_cluster_c, num_clusters=int(ctx["K"]), reduce="min"
    )
    dist_ch_min_dyn = channel_to_cluster_distance(
        target_p["dyn"], source_ch_p["dyn"], source_cluster_c, num_clusters=int(ctx["K"]), reduce="min"
    )
    dist_ch_min_cycle = channel_to_cluster_distance(
        target_p["cycle"], source_ch_p["cycle"], source_cluster_c, num_clusters=int(ctx["K"]), reduce="min"
    )
    dist_name = name_distance(ctx)
    corr = cycle_corr(ctx, cfg)

    grids = [
        ("corr_only", 1.0, 0.0, 0.0, 0.0, 0.0),
        ("portrait_stat", 0.0, 1.0, 0.0, 0.0, 0.0),
        ("portrait_dyn", 0.0, 0.0, 1.0, 0.0, 0.0),
        ("portrait_cycle", 0.0, 0.0, 0.0, 1.0, 0.0),
        ("portrait_all", 0.0, 0.5, 0.5, 0.5, 0.0),
        ("hybrid_light", 1.0, 0.1, 0.1, 0.1, 0.0),
        ("hybrid_med", 1.0, 0.25, 0.25, 0.25, 0.0),
        ("hybrid_strong", 1.0, 0.5, 0.5, 0.5, 0.0),
        ("hybrid_cycle", 1.0, 0.0, 0.1, 0.5, 0.0),
        ("hybrid_dyn", 1.0, 0.1, 0.5, 0.1, 0.0),
        ("same_name", 0.0, 0.0, 0.0, 0.0, 1.0),
        ("same_name_light", 1.0, 0.1, 0.1, 0.1, 0.25),
        ("same_name_med", 1.0, 0.2, 0.2, 0.2, 0.5),
    ]
    out: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    corr_route = tuple(int(v) for v in static_route_from_train(ctx, cfg).detach().cpu().tolist())
    for item in [{"candidate": "static_corr_train", "route": corr_route, "weights": (1.0, 0.0, 0.0, 0.0, 0.0)}]:
        seen.add(tuple(item["route"]))
        out.append(item)
    for name, cw, sw, dw, cyw, nw in grids:
        score = (
            float(cw) * corr
            - float(sw) * dist_stat
            - float(dw) * dist_dyn
            - float(cyw) * dist_cycle
            - float(nw) * dist_name
        )
        route = tuple(int(v) for v in torch.argmax(score, dim=1).detach().cpu().tolist())
        if route in seen:
            continue
        seen.add(route)
        out.append({"candidate": name, "route": route, "weights": (cw, sw, dw, cyw, nw)})
    channel_grids = [
        ("channel_mean_all", 0.0, 0.4, 0.4, 0.4, 0.0, "mean"),
        ("channel_min_all", 0.0, 0.4, 0.4, 0.4, 0.0, "min"),
        ("hybrid_channel_mean_light", 1.0, 0.1, 0.1, 0.1, 0.0, "mean"),
        ("hybrid_channel_mean_med", 1.0, 0.25, 0.25, 0.25, 0.0, "mean"),
        ("hybrid_channel_min_light", 1.0, 0.1, 0.1, 0.1, 0.0, "min"),
        ("hybrid_channel_min_med", 1.0, 0.25, 0.25, 0.25, 0.0, "min"),
        ("hybrid_channel_name", 1.0, 0.1, 0.1, 0.1, 0.25, "min"),
    ]
    for name, cw, sw, dw, cyw, nw, reduce in channel_grids:
        if reduce == "min":
            ds, dd, dc = dist_ch_min_stat, dist_ch_min_dyn, dist_ch_min_cycle
        else:
            ds, dd, dc = dist_ch_mean_stat, dist_ch_mean_dyn, dist_ch_mean_cycle
        score = (
            float(cw) * corr
            - float(sw) * ds
            - float(dw) * dd
            - float(cyw) * dc
            - float(nw) * dist_name
        )
        route = tuple(int(v) for v in torch.argmax(score, dim=1).detach().cpu().tolist())
        if route in seen:
            continue
        seen.add(route)
        out.append({"candidate": name, "route": route, "weights": (cw, sw, dw, cyw, nw)})
    return out


def source_target_metrics(horizon: int) -> tuple[dict[str, Any], dict[str, Any]]:
    source_summary = ROOT / "outputs" / "ett_global_h96_param_base" / "runs" / "ETTm1" / f"pred_{horizon}" / "run_summary.json"
    target_summary = ROOT / "outputs" / "ett_horizon_sweep" / "runs" / "ETTm2" / f"pred_{horizon}" / "run_summary.json"
    with source_summary.open("r", encoding="utf-8") as f:
        src = json.load(f)
    with target_summary.open("r", encoding="utf-8") as f:
        tgt = json.load(f)
    return src, tgt


def run_transfer(py: str, cfg_path: Path) -> None:
    proc = subprocess.run(
        [py, "-u", "-m", "src.transfer", "--config", str(cfg_path)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout[-3000:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "ettm1_to_ettm2_portrait_route_search")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--python", type=Path, default=Path(sys.executable))
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        base_cfg_path = ROOT / "outputs" / "ettm1_other_ett_horizon_transfer" / "ETTm1_to_ETTm2" / f"pred_{horizon}" / "base_config.yaml"
        cfg = read_yaml(base_cfg_path)
        ctx = load_context(cfg, device)
        src_summary, tgt_summary = source_target_metrics(horizon)

        for cand in candidate_routes(ctx, cfg):
            route = tuple(cand["route"])
            val = evaluate_route(ctx, route, split="val", batch_size=args.batch_size)
            test = evaluate_route(ctx, route, split="test", batch_size=args.batch_size)
            cw, sw, dw, cyw, nw = cand["weights"]
            rows.append(
                {
                    "horizon": horizon,
                    "candidate": cand["candidate"],
                    "route": json.dumps(route),
                    "corr_weight": cw,
                    "stat_weight": sw,
                    "dyn_weight": dw,
                    "cycle_weight": cyw,
                    "name_weight": nw,
                    "val_mse": val["avg_mse"],
                    "val_mae": val["avg_mae"],
                    "test_mse": test["avg_mse"],
                    "test_mae": test["avg_mae"],
                    "target_self_mse": tgt_summary["test"]["avg_mse"],
                    "target_self_mae": tgt_summary["test"]["avg_mae"],
                    "source_test_mse": src_summary["test"]["avg_mse"],
                    "source_test_mae": src_summary["test"]["avg_mae"],
                }
            )
            write_rows(args.out_root / "portrait_route_results.csv", rows)

        cur = [r for r in rows if int(r["horizon"]) == horizon]
        selected = sorted(cur, key=lambda r: (float(r["val_mse"]), float(r["val_mae"])))[0]
        route = tuple(json.loads(selected["route"]))
        final_dir = args.out_root / f"H{horizon}" / "selected_portrait_transfer"
        final_cfg = make_fixed_cfg(cfg, route, final_dir, "test")
        final_cfg_path = args.out_root / f"H{horizon}" / "selected_portrait_config.yaml"
        write_yaml(final_cfg_path, final_cfg)
        run_transfer(str(args.python), final_cfg_path)
        with (final_dir / "transfer_summary.json").open("r", encoding="utf-8") as f:
            final_summary = json.load(f)
        summaries.append(
            {
                "horizon": horizon,
                "selected_candidate": selected["candidate"],
                "selected_route": selected["route"],
                "selected_val_mse": selected["val_mse"],
                "selected_val_mae": selected["val_mae"],
                "selected_test_mse": final_summary["avg_mse"],
                "selected_test_mae": final_summary["avg_mae"],
                "target_self_mse": tgt_summary["test"]["avg_mse"],
                "target_self_mae": tgt_summary["test"]["avg_mae"],
                "source_test_mse": src_summary["test"]["avg_mse"],
                "source_test_mae": src_summary["test"]["avg_mae"],
                "config_path": str(final_cfg_path),
                "out_dir": str(final_dir),
            }
        )
        with (args.out_root / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)

    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(args.out_root / "portrait_route_results.csv")
    print(args.out_root / "summary.json")


if __name__ == "__main__":
    main()
