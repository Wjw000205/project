import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.cluster_predictor import build_cluster_predictor
from src.utils.cluster_memory import (
    assign_channels_by_corr,
    compute_cluster_prototypes,
    load_cluster_checkpoint,
    load_cluster_memory,
    save_cluster_memory,
)


SOURCE_CONFIG = ROOT / "outputs" / "ettm1_val_refinement_base" / "configs" / "ETTm1_pred_96.yaml"
SOURCE_CHECKPOINT = (
    ROOT
    / "outputs"
    / "ettm1_val_refinement_base"
    / "runs"
    / "ETTm1"
    / "pred_96"
    / "best_checkpoint.pt"
)
HEAD_SELECTION_POLICY = "channel_cluster_pearson_argmax"

TARGETS = [
    {"target": "ETTh2", "csv_path": "data/ETTh2.csv", "phase_bins": 64, "batch_size": 64},
    {"target": "ETTh1", "csv_path": "data/ETTh1.csv", "phase_bins": 64, "batch_size": 64},
    {"target": "ETTm2", "csv_path": "data/ETTm2.csv", "phase_bins": 64, "batch_size": 64},
    {"target": "weather", "csv_path": "data/weather.csv", "phase_bins": 64, "batch_size": 64},
    {"target": "traffic", "csv_path": "data/traffic.csv", "phase_bins": 48, "batch_size": 16},
]


def load_yaml(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_step_minutes(dates: pd.Series) -> float:
    dt = pd.to_datetime(dates)
    diffs = dt.diff().dropna()
    if diffs.empty:
        return 0.0
    return float(diffs.mode().iloc[0].total_seconds() / 60.0)


def read_series(path: Path) -> Tuple[pd.DataFrame, str, List[str], float]:
    df = pd.read_csv(path)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    value_cols = [c for c in df.columns if c != date_col]
    df = (
        df.groupby(date_col, as_index=False)[value_cols]
        .mean()
        .sort_values(date_col)
        .reset_index(drop=True)
    )
    df[value_cols] = df[value_cols].ffill().bfill()
    step_minutes = infer_step_minutes(df[date_col])
    return df, date_col, value_cols, step_minutes


def normalize_train_only(values: np.ndarray, train_end: int) -> np.ndarray:
    train = values[:train_end]
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.maximum(std, 1.0e-6)
    return ((values - mean) / std).astype(np.float32, copy=False)


def interpolate_values(times: np.ndarray, values: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    out = np.empty((query_times.shape[0], values.shape[1]), dtype=np.float32)
    for c in range(values.shape[1]):
        out[:, c] = np.interp(query_times, times, values[:, c]).astype(np.float32, copy=False)
    return out


def resample_prototypes(prototypes: torch.Tensor, source_step: float, target_step: float) -> torch.Tensor:
    if abs(float(source_step) - float(target_step)) < 1.0e-6:
        return prototypes.detach().cpu()
    proto = prototypes.detach().cpu().numpy().astype(np.float32, copy=False)
    k_count, length = proto.shape
    source_t = np.arange(length, dtype=np.float64) * float(source_step)
    target_t = np.arange(0.0, source_t[-1] + 1.0e-6, float(target_step), dtype=np.float64)
    out = np.empty((k_count, target_t.shape[0]), dtype=np.float32)
    for k in range(k_count):
        out[k] = np.interp(target_t, source_t, proto[k]).astype(np.float32, copy=False)
    return torch.tensor(out, dtype=torch.float32)


def ensure_source_memory(out_root: Path) -> Path:
    memory_path = out_root / "source_period" / "cluster_memory.pt"
    if memory_path.exists():
        return memory_path
    if not SOURCE_CONFIG.exists():
        raise FileNotFoundError(f"Missing source config: {SOURCE_CONFIG}")
    if not SOURCE_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing source checkpoint: {SOURCE_CHECKPOINT}")

    cfg = load_yaml(SOURCE_CONFIG)
    data_cfg = cfg["data"]
    df, date_col, value_cols, _ = read_series(ROOT / str(data_cfg["csv_path"]))
    max_rows = int(data_cfg.get("max_rows", 0) or 0)
    if max_rows > 0:
        df = df.iloc[:max_rows].copy()
    values = df[value_cols].to_numpy(dtype=np.float32)
    train_end = int(values.shape[0] * float(data_cfg["train_ratio"]))
    if bool((cfg.get("normalize") or {}).get("global_zscore", False)):
        values = normalize_train_only(values, train_end)

    ckpt = torch.load(SOURCE_CHECKPOINT, map_location="cpu")
    meta = ckpt.get("meta", {})
    cluster_id_c = meta["cluster_id_c"].detach().cpu().to(torch.long)
    prototypes_kt = compute_cluster_prototypes(torch.tensor(values[:train_end]), cluster_id_c)
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    save_cluster_memory(
        str(memory_path),
        prototypes_kt,
        cluster_id_c,
        value_cols,
        meta={
            "kind": "train_segment_prototype_synthesized",
            "source": "ETTm1",
            "source_step_minutes": 15,
            "input_len": int(meta["input_len"]),
            "pred_len": int(meta["pred_len"]),
            "memory_len": int(train_end),
        },
    )
    return memory_path


def ensure_checkpoint(out_root: Path) -> Path:
    patched = out_root / "source_period" / "best_checkpoint_transfer.pt"
    patched.parent.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(SOURCE_CHECKPOINT, map_location="cpu")
    meta = dict(ckpt.get("meta", {}) or {})
    moe_cfg = dict(meta.get("moe_cfg", {}) or {})
    if "hidden_dim" not in moe_cfg and "gate_hidden_dim" in moe_cfg:
        moe_cfg["hidden_dim"] = moe_cfg["gate_hidden_dim"]
    meta["moe_cfg"] = moe_cfg
    ckpt["meta"] = meta
    torch.save(ckpt, patched)
    return patched


def load_source_model(checkpoint_path: Path, device: torch.device):
    ckpt = load_cluster_checkpoint(str(checkpoint_path), device=device)
    meta = ckpt["meta"]
    model = build_cluster_predictor(
        num_clusters=int(meta["K"]),
        input_len=int(meta["input_len"]),
        pred_len=int(meta["pred_len"]),
        model_cfg=meta["model_cfg"],
        num_channels=meta.get("num_channels", None),
        cluster_id_c=meta.get("cluster_id_c", None),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, meta


def forecast_source_steps(
    model,
    x_bcl: torch.Tensor,
    cluster_id_c: torch.Tensor,
    input_len: int,
    needed_source_steps: int,
) -> torch.Tensor:
    chunks = []
    ctx = x_bcl
    made = 0
    while made < needed_source_steps:
        y = model(ctx, cluster_id_c)
        chunks.append(y)
        made += int(y.shape[-1])
        ctx = torch.cat([ctx, y], dim=-1)[..., -input_len:]
    return torch.cat(chunks, dim=-1)[..., :needed_source_steps]


def run_target(
    target: Dict[str, object],
    out_root: Path,
    model,
    source_meta: Dict[str, object],
    memory_path: Path,
    device: torch.device,
    source_step_minutes: float,
) -> Dict[str, object]:
    name = str(target["target"])
    run_dir = out_root / "period_runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)

    df, date_col, channel_names, target_step = read_series(ROOT / str(target["csv_path"]))
    prepared_path = out_root / "period_prepared" / f"{name}.csv"
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(prepared_path, index=False)

    values = df[channel_names].to_numpy(dtype=np.float32)
    total_t, channels = values.shape
    train_end = int(total_t * 0.7)
    test_start = int(total_t * 0.8)
    values = normalize_train_only(values, train_end)

    dates = pd.to_datetime(df[date_col])
    times_min = (dates - dates.iloc[0]).dt.total_seconds().to_numpy(dtype=np.float64) / 60.0

    memory = load_cluster_memory(str(memory_path), device=torch.device("cpu"))
    proto_target = resample_prototypes(
        memory["prototypes_kt"],
        source_step=float(source_step_minutes),
        target_step=float(target_step),
    )
    target_tensor = torch.tensor(values, dtype=torch.float32)
    cluster_id_c, corr_ck = assign_channels_by_corr(
        target_tensor,
        proto_target,
        align="head",
        max_lag=0,
    )
    cluster_id_c = cluster_id_c.to(device)
    corr_np = corr_ck.detach().cpu().numpy()
    corr_max = corr_ck.max(dim=1).values.detach().cpu().numpy()

    input_len = int(source_meta["input_len"])
    source_pred_len = int(source_meta["pred_len"])
    target_pred_len = 96
    context_minutes = input_len * float(source_step_minutes)
    context_native_steps = int(math.ceil(context_minutes / float(target_step)))
    first_origin = max(test_start + context_native_steps, 1)
    last_origin_exclusive = total_t - target_pred_len + 1
    origins = np.arange(first_origin, last_origin_exclusive, dtype=np.int64)
    if origins.size == 0:
        raise ValueError(f"No period-aware test windows for {name}.")

    max_label_offset = (target_pred_len - 1) * float(target_step)
    needed_source_steps = int(math.floor(max_label_offset / float(source_step_minutes))) + 2
    needed_source_steps = max(source_pred_len, needed_source_steps)
    target_offsets = np.arange(target_pred_len, dtype=np.float64) * float(target_step)
    pos = target_offsets / float(source_step_minutes)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, needed_source_steps - 1)
    w = torch.tensor((pos - lo).astype(np.float32), device=device).view(1, 1, -1)
    lo_t = torch.tensor(lo, dtype=torch.long, device=device)
    hi_t = torch.tensor(hi, dtype=torch.long, device=device)

    batch_size = int(target["batch_size"])
    se_c = torch.zeros(channels, dtype=torch.float64, device=device)
    ae_c = torch.zeros(channels, dtype=torch.float64, device=device)
    denom = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, origins.size, batch_size):
            batch_origins = origins[start : start + batch_size]
            x_batch = []
            y_batch = []
            for origin in batch_origins:
                origin_time = float(times_min[origin])
                ctx_times = origin_time - float(source_step_minutes) * np.arange(input_len, 0, -1)
                ctx = interpolate_values(times_min, values, ctx_times).T
                y = values[origin : origin + target_pred_len].T
                x_batch.append(ctx)
                y_batch.append(y)
            x = torch.tensor(np.stack(x_batch, axis=0), dtype=torch.float32, device=device)
            y_true = torch.tensor(np.stack(y_batch, axis=0), dtype=torch.float32, device=device)
            y_src = forecast_source_steps(model, x, cluster_id_c, input_len, needed_source_steps)
            y_lo = y_src.index_select(dim=2, index=lo_t)
            y_hi = y_src.index_select(dim=2, index=hi_t)
            y_pred = y_lo * (1.0 - w) + y_hi * w
            err = y_pred - y_true
            se_c += (err.double() ** 2).sum(dim=(0, 2))
            ae_c += err.abs().double().sum(dim=(0, 2))
            denom += int(y_true.shape[0] * y_true.shape[2])

    mse_c = (se_c / max(denom, 1)).detach().cpu().numpy()
    mae_c = (ae_c / max(denom, 1)).detach().cpu().numpy()
    metrics_payload = {
        "channel": channel_names,
        "MAE": mae_c,
        "MSE": mse_c,
        "selected_head": cluster_id_c.detach().cpu().numpy(),
        "corr_max": corr_max,
    }
    for k in range(corr_np.shape[1]):
        metrics_payload[f"corr_head_{k}"] = corr_np[:, k]
    metrics_df = pd.DataFrame(metrics_payload)
    metrics_df.to_csv(run_dir / "test_metrics.csv", index=False)

    assign_payload = {
        "channel": channel_names,
        "selected_head": cluster_id_c.detach().cpu().numpy(),
        "corr_max": corr_max,
    }
    for k in range(corr_np.shape[1]):
        assign_payload[f"corr_head_{k}"] = corr_np[:, k]
    assign_df = pd.DataFrame(assign_payload)
    assign_df.to_csv(run_dir / "cluster_assignment.csv", index=False)
    np.save(run_dir / "channel_cluster_corr.npy", corr_np)

    # Backward-compatible alias for downstream tooling that expects cluster_id.
    metrics_df_compat = pd.DataFrame(
        {
            "channel": channel_names,
            "MAE": mae_c,
            "MSE": mse_c,
            "cluster_id": cluster_id_c.detach().cpu().numpy(),
            "corr_max": corr_max,
        }
    )
    metrics_df_compat.to_csv(run_dir / "test_metrics_compat.csv", index=False)

    summary = {
        "avg_mae": float(np.mean(mae_c)),
        "avg_mse": float(np.mean(mse_c)),
        "elapsed_sec": float(time.perf_counter() - t0),
        "num_test_windows": int(origins.size),
        "num_channels": int(channels),
        "source_step_minutes": float(source_step_minutes),
        "target_step_minutes": float(target_step),
        "input_context_hours": float(context_minutes / 60.0),
        "target_pred_span_hours": float((target_pred_len * target_step) / 60.0),
        "needed_source_steps": int(needed_source_steps),
        "prepared_csv": str(prepared_path),
        "head_selection_policy": HEAD_SELECTION_POLICY,
        "corr_mode": "pearson",
        "corr_align": "head",
        "corr_max_lag": 0,
    }
    with (run_dir / "transfer_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    counts = pd.Series(cluster_id_c.detach().cpu().numpy()).value_counts().sort_index()
    return {
        "status": "ok",
        "source": "ETTm1",
        "target": name,
        "pred_len": target_pred_len,
        "input_len": input_len,
        "source_step_minutes": float(source_step_minutes),
        "target_step_minutes": float(target_step),
        "input_context_hours": float(context_minutes / 60.0),
        "target_pred_span_hours": float((target_pred_len * target_step) / 60.0),
        "target_csv": str(target["csv_path"]),
        "prepared_csv": str(prepared_path),
        "out_dir": str(run_dir),
        "avg_mse": summary["avg_mse"],
        "avg_mae": summary["avg_mae"],
        "num_channels": int(channels),
        "num_test_windows": int(origins.size),
        "mean_corr_max": float(np.mean(corr_max)),
        "min_corr_max": float(np.min(corr_max)),
        "cluster_sizes": ";".join(f"{int(k)}:{int(v)}" for k, v in counts.items()),
        "needed_source_steps": int(needed_source_steps),
        "head_selection_policy": HEAD_SELECTION_POLICY,
        "error": "",
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "status",
        "source",
        "target",
        "pred_len",
        "input_len",
        "source_step_minutes",
        "target_step_minutes",
        "input_context_hours",
        "target_pred_span_hours",
        "target_csv",
        "prepared_csv",
        "out_dir",
        "avg_mse",
        "avg_mae",
        "num_channels",
        "num_test_windows",
        "mean_corr_max",
        "min_corr_max",
        "cluster_sizes",
        "needed_source_steps",
        "head_selection_policy",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default=str(ROOT / "outputs" / "ettm1_h96_transfer"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--targets", nargs="*", default=[t["target"] for t in TARGETS])
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve()
    selected = {x.lower() for x in args.targets}
    targets = [t for t in TARGETS if t["target"].lower() in selected]
    if not targets:
        raise ValueError(f"No targets selected: {args.targets}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    memory_path = ensure_source_memory(out_root)
    checkpoint_path = ensure_checkpoint(out_root)
    model, meta = load_source_model(checkpoint_path, device=device)
    rows: List[Dict[str, object]] = []
    for target in targets:
        print(f"[period-transfer] ETTm1 -> {target['target']}")
        try:
            row = run_target(
                target,
                out_root,
                model,
                meta,
                memory_path,
                device,
                source_step_minutes=15.0,
            )
            print(
                f"[ok] {target['target']} mse={row['avg_mse']:.6f} "
                f"mae={row['avg_mae']:.6f} target_step={row['target_step_minutes']:.0f}min"
            )
        except Exception as exc:
            row = {
                "status": "failed",
                "source": "ETTm1",
                "target": target["target"],
                "pred_len": 96,
                "input_len": int(meta.get("input_len", 336)),
                "source_step_minutes": 15.0,
                "target_step_minutes": "",
                "input_context_hours": "",
                "target_pred_span_hours": "",
                "target_csv": target["csv_path"],
                "prepared_csv": "",
                "out_dir": str(out_root / "period_runs" / target["target"]),
                "avg_mse": "",
                "avg_mae": "",
                "num_channels": "",
                "num_test_windows": "",
                "mean_corr_max": "",
                "min_corr_max": "",
                "cluster_sizes": "",
                "needed_source_steps": "",
                "error": str(exc),
            }
            print(f"[failed] {target['target']}: {exc}")
        rows.append(row)
        write_csv(out_root / "transfer.csv", rows)
    write_csv(out_root / "transfer.csv", rows)
    print(f"Saved: {out_root / 'transfer.csv'}")


if __name__ == "__main__":
    main()
