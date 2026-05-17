import argparse
import csv
import json
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
from src.models.moe_gate import ClusterwiseMoEGate
from src.models.residual_moe import ClusterwisePredResidualMoE
from src.models.moe_gate import scatter_mean_bcf_to_bkf
from src.train import extract_gate_features, _select_rank_mask
from src.utils.cluster_memory import (
    assign_channels_by_corr,
    assign_channels_by_cycle_template,
    compute_cluster_prototypes,
    load_cluster_checkpoint,
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
SOURCE_SUMMARY = (
    ROOT
    / "outputs"
    / "ettm1_val_refinement_base"
    / "runs"
    / "ETTm1"
    / "pred_96"
    / "run_summary.json"
)


TARGETS: List[Dict[str, object]] = [
    {
        "target": "ETTh2",
        "csv_path": "data/ETTh2.csv",
        "resample_enable": True,
        "target_step_minutes": 15,
        "resample_method": "linear",
        "normalize_train_only": True,
        "route_mode": "cycle_template",
        "batch_size": 64,
    },
    {
        "target": "ETTh1",
        "csv_path": "data/ETTh1.csv",
        "resample_enable": True,
        "target_step_minutes": 15,
        "resample_method": "linear",
        "normalize_train_only": True,
        "route_mode": "cycle_template",
        "batch_size": 64,
    },
    {
        "target": "ETTm2",
        "csv_path": "data/ETTm2.csv",
        "resample_enable": False,
        "target_step_minutes": 15,
        "resample_method": "linear",
        "normalize_train_only": True,
        "route_mode": "cycle_template",
        "batch_size": 64,
    },
    {
        "target": "weather",
        "csv_path": "data/weather.csv",
        "resample_enable": True,
        "target_step_minutes": 15,
        "resample_method": "linear",
        "normalize_train_only": True,
        "route_mode": "cycle_template",
        "batch_size": 64,
    },
    {
        "target": "traffic",
        "csv_path": "data/traffic.csv",
        "resample_enable": True,
        "target_step_minutes": 15,
        "resample_method": "linear",
        "normalize_train_only": True,
        "route_mode": "cycle_template",
        "batch_size": 16,
    },
]


def load_yaml(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_step_minutes(df: pd.DataFrame, date_col: str) -> float:
    dt = pd.to_datetime(df[date_col])
    diffs = dt.diff().dropna()
    if diffs.empty:
        return 0.0
    return float(diffs.mode().iloc[0].total_seconds() / 60.0)


def dedupe_dates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    value_cols = [c for c in out.columns if c != date_col]
    out = (
        out.groupby(date_col, as_index=False)[value_cols]
        .mean()
        .sort_values(date_col)
        .reset_index(drop=True)
    )
    out[value_cols] = out[value_cols].ffill().bfill()
    return out


def resample_df(df: pd.DataFrame, date_col: str, target_step_min: int, method: str) -> pd.DataFrame:
    if target_step_min <= 0:
        return df
    tmp = dedupe_dates(df, date_col).set_index(date_col)
    rule = f"{int(target_step_min)}min"
    method = str(method).lower()
    if method in {"mean", "avg"}:
        out = tmp.resample(rule).mean().interpolate("time").ffill().bfill()
    elif method in {"last", "ffill"}:
        out = tmp.resample(rule).last().ffill().bfill()
    else:
        out = tmp.resample(rule).interpolate("time").ffill().bfill()
    return out.reset_index()


def read_value_frame(path: Path) -> Tuple[pd.DataFrame, str, List[str]]:
    df = pd.read_csv(path)
    date_col = df.columns[0]
    df = dedupe_dates(df, date_col)
    value_cols = [c for c in df.columns if c != date_col]
    return df, date_col, value_cols


def zscore(values: np.ndarray, train_end: int, train_only: bool) -> np.ndarray:
    ref = values[:train_end] if train_only else values
    mean = ref.mean(axis=0, keepdims=True)
    std = np.maximum(ref.std(axis=0, keepdims=True), 1.0e-6)
    return ((values - mean) / std).astype(np.float32, copy=False)


def ensure_source_memory(out_root: Path) -> Path:
    memory_path = out_root / "source" / "cluster_memory.pt"
    if memory_path.exists():
        return memory_path
    if not SOURCE_CONFIG.exists():
        raise FileNotFoundError(f"Missing source config: {SOURCE_CONFIG}")
    if not SOURCE_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing source checkpoint: {SOURCE_CHECKPOINT}")

    cfg = load_yaml(SOURCE_CONFIG)
    df, date_col, value_cols = read_value_frame(ROOT / str(cfg["data"]["csv_path"]))
    max_rows = int(cfg["data"].get("max_rows", 0) or 0)
    if max_rows > 0:
        df = df.iloc[:max_rows].copy()
    values = df[value_cols].to_numpy(dtype=np.float32)
    train_end = int(values.shape[0] * float(cfg["data"]["train_ratio"]))
    if bool((cfg.get("normalize") or {}).get("global_zscore", False)):
        values = zscore(
            values,
            train_end=train_end,
            train_only=bool((cfg.get("normalize") or {}).get("train_only", False)),
        )
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
            "source_config": str(SOURCE_CONFIG),
            "source_checkpoint": str(SOURCE_CHECKPOINT),
            "input_len": int(meta["input_len"]),
            "pred_len": int(meta["pred_len"]),
            "memory_len": int(train_end),
        },
    )
    return memory_path


def load_source_model(device: torch.device):
    ckpt = load_cluster_checkpoint(str(SOURCE_CHECKPOINT), device=device)
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
    return model, meta, ckpt


def load_residual_scales(channel_names: List[str], device: torch.device) -> torch.Tensor:
    if not SOURCE_SUMMARY.exists():
        return torch.ones(len(channel_names), dtype=torch.float32, device=device)
    with SOURCE_SUMMARY.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    selection = summary.get("moe_residual_selection", {}) or {}
    source_channels = list(selection.get("gate_calibrator", {}).get("channel_names") or selection.get("residual_channels") or [])
    scale_values = list(selection.get("scale_values") or [])
    mean_scale = float(selection.get("mean_scale", 1.0) or 1.0)
    scale_by_name = {
        str(name): float(scale)
        for name, scale in zip(source_channels, scale_values)
    }
    scales = [scale_by_name.get(str(name), mean_scale) for name in channel_names]
    return torch.tensor(scales, dtype=torch.float32, device=device)


def build_moe_modules(ckpt: Dict[str, object], meta: Dict[str, object], device: torch.device):
    penalty_names = list(meta.get("penalty_names", []) or [])
    moe_cfg = dict(meta.get("moe_cfg", {}) or {})
    if not bool(moe_cfg.get("enable", True)) or len(penalty_names) == 0:
        return None, None, penalty_names

    k_count = int(meta["K"])
    gate_feat_dim = int(meta.get("gate_feat_dim", 10))
    gate_state = ckpt.get("gate_state", None)
    if gate_state is None:
        return None, None, penalty_names

    allow_skip = any(str(name).startswith("W_skip.") for name in gate_state.keys())
    gate = ClusterwiseMoEGate(
        num_clusters=k_count,
        feat_dim=gate_feat_dim,
        num_penalties=len(penalty_names),
        hidden_dim=int(moe_cfg.get("gate_hidden_dim", moe_cfg.get("hidden_dim", 64))),
        topk=int(moe_cfg.get("topk", 2)),
        allow_skip=allow_skip,
        skip_init_bias=float(moe_cfg.get("skip_init_bias", -2.0)),
    ).to(device)
    gate.temperature = float(moe_cfg.get("gate_temperature", 1.0))
    gate.noise_std = float(moe_cfg.get("gate_noise_std", 0.0))
    gate.logit_clip = float(moe_cfg.get("gate_logit_clip", 0.0))
    gate.prob_floor = float(moe_cfg.get("gate_prob_floor", 0.0))
    gate.load_state_dict(gate_state, strict=True)
    gate.eval()

    pred_state = ckpt.get("pred_residual_state", None)
    pred_cfg = dict((moe_cfg.get("pred_side_residual", {}) or {}))
    pred_residual = None
    if pred_state is not None and bool(pred_cfg.get("enable", False)):
        pred_residual = ClusterwisePredResidualMoE(
            num_clusters=k_count,
            num_penalties=len(penalty_names),
            input_len=int(meta["input_len"]),
            pred_len=int(meta["pred_len"]),
            hidden_dim=int(pred_cfg.get("corrector_hidden", 32)),
            init_alpha=float(pred_cfg.get("init_alpha", -3.0)),
            alpha_scale=float(pred_cfg.get("alpha_scale", 0.5)),
            use_y_base_input=bool(pred_cfg.get("use_y_base_input", True)),
            feature_mode=str(pred_cfg.get("feature_mode", "legacy")),
            residual_clip=float(pred_cfg.get("residual_clip", 0.0)),
            intervention_enable=bool(pred_cfg.get("intervention_enable", False)),
            intervention_init=float(pred_cfg.get("intervention_init", -2.0)),
        ).to(device)
        pred_residual.load_state_dict(pred_state, strict=True)
        pred_residual.eval()
    return gate, pred_residual, penalty_names


def predict_with_optional_residual(
    *,
    model,
    gate,
    pred_residual,
    x: torch.Tensor,
    cluster_id_c: torch.Tensor,
    meta: Dict[str, object],
    residual_scale_c: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    yhat_base = model(x, cluster_id_c)
    if gate is None or pred_residual is None:
        return yhat_base, yhat_base

    moe_cfg = dict(meta.get("moe_cfg", {}) or {})
    k_count = int(meta["K"])
    feat_bcf = extract_gate_features(x)
    feat_bkf = scatter_mean_bcf_to_bkf(feat_bcf, cluster_id_c, k_count)
    straight_through = False
    mask_bkp, probs_bkp, skip_bk, _ = gate(feat_bkf, straight_through=straight_through)
    raw_ranks = moe_cfg.get("select_ranks", None)
    if raw_ranks is not None:
        select_ranks = [int(v) for v in raw_ranks]
        mask_bkp = _select_rank_mask(probs_bkp, select_ranks, straight_through=False)
    gate_soft_weight = float(moe_cfg.get("gate_soft_weight", 0.0))
    if gate_soft_weight > 0.0:
        target_mass = mask_bkp.detach().sum(dim=-1, keepdim=True).clamp_min(1.0)
        probs_sel = probs_bkp * target_mass
        mask_bkp = (1.0 - gate_soft_weight) * mask_bkp + gate_soft_weight * probs_sel
    allow_skip = bool(moe_cfg.get("allow_skip", False))
    pred_out = pred_residual(
        x,
        yhat_base,
        cluster_id_c,
        mask_bkp,
        skip_bk=skip_bk if allow_skip else None,
    )
    yhat = pred_out["y_final"]
    if residual_scale_c is not None:
        scale = residual_scale_c.to(device=yhat.device, dtype=yhat.dtype).view(1, -1, 1)
        yhat = yhat_base + scale * (yhat - yhat_base)
    return yhat_base, yhat


def prepare_target(target: Dict[str, object], out_root: Path) -> Tuple[pd.DataFrame, str, List[str], Dict[str, object]]:
    raw_path = ROOT / str(target["csv_path"])
    raw = pd.read_csv(raw_path)
    date_col = raw.columns[0]
    original_rows = int(raw.shape[0])
    original_step = infer_step_minutes(raw, date_col)
    if bool(target["resample_enable"]):
        df = resample_df(
            raw,
            date_col,
            target_step_min=int(target["target_step_minutes"]),
            method=str(target["resample_method"]),
        )
    else:
        df = dedupe_dates(raw, date_col)
    value_cols = [c for c in df.columns if c != date_col]
    df[value_cols] = df[value_cols].ffill().bfill()
    prepared_path = out_root / "prepared" / f"{target['target']}.csv"
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(prepared_path, index=False)
    info = {
        "original_rows": original_rows,
        "prepared_rows": int(df.shape[0]),
        "original_step_minutes": original_step,
        "prepared_step_minutes": infer_step_minutes(df, date_col),
        "prepared_csv": str(prepared_path),
    }
    return df, date_col, value_cols, info


def evaluate_target(
    target: Dict[str, object],
    out_root: Path,
    model,
    gate,
    pred_residual,
    source_meta: Dict[str, object],
    prototypes_kt: torch.Tensor,
    device: torch.device,
) -> Dict[str, object]:
    name = str(target["target"])
    run_dir = out_root / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    df, date_col, value_cols, prep_info = prepare_target(target, out_root)
    values = df[value_cols].to_numpy(dtype=np.float32)
    total_t, channels = values.shape
    train_end = int(total_t * 0.7)
    test_start = int(total_t * 0.8)
    values = zscore(
        values,
        train_end=train_end,
        train_only=bool(target["normalize_train_only"]),
    )
    data_cpu = torch.tensor(values, dtype=torch.float32)
    route_data_cpu = data_cpu[:train_end].contiguous()
    route_mode = str(target.get("route_mode", "pearson")).lower()
    best_tau = None
    if route_mode in {"cycle", "cycle_template", "phase"}:
        step_min = float(prep_info["prepared_step_minutes"] or target.get("target_step_minutes", 15) or 15)
        period_min = int(round(12.0 * 60.0 / step_min))
        period_max = int(round(168.0 * 60.0 / step_min))
        cluster_id_c, corr_ck, best_tau = assign_channels_by_cycle_template(
            route_data_cpu,
            prototypes_kt.detach().cpu(),
            phase_bins=int(target.get("phase_bins", 64)),
            period_min=period_min,
            period_max=period_max,
            align="head",
            phase_max_shift=target.get("phase_max_shift", None),
        )
        head_policy = "channel_cluster_cycle_template_argmax"
    else:
        cluster_id_c, corr_ck = assign_channels_by_corr(
            route_data_cpu,
            prototypes_kt.detach().cpu(),
            align="head",
            max_lag=int(target.get("corr_max_lag", 0)),
        )
        head_policy = "channel_cluster_pearson_argmax"
    corr_np = corr_ck.detach().cpu().numpy()
    corr_max = corr_np.max(axis=1)
    cluster_id_device = cluster_id_c.to(device=device, dtype=torch.long)
    residual_scale_c = load_residual_scales(value_cols, device=device)

    input_len = int(source_meta["input_len"])
    pred_len = int(source_meta["pred_len"])
    if pred_len != 96:
        raise ValueError(f"Expected source pred_len=96, got {pred_len}.")
    seg = data_cpu[test_start:]
    num_windows = int(seg.shape[0] - input_len - pred_len + 1)
    if num_windows <= 0:
        raise ValueError(f"No H96 test windows for {name}.")

    batch_size = int(target["batch_size"])
    se_c = torch.zeros(channels, dtype=torch.float64, device=device)
    ae_c = torch.zeros(channels, dtype=torch.float64, device=device)
    base_se_c = torch.zeros(channels, dtype=torch.float64, device=device)
    base_ae_c = torch.zeros(channels, dtype=torch.float64, device=device)
    denom = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, num_windows, batch_size):
            end = min(start + batch_size, num_windows)
            xs = []
            ys = []
            for i in range(start, end):
                win = seg[i : i + input_len + pred_len]
                xs.append(win[:input_len].T)
                ys.append(win[input_len:].T)
            x = torch.stack(xs, dim=0).to(device=device, non_blocking=True)
            y = torch.stack(ys, dim=0).to(device=device, non_blocking=True)
            yhat_base, yhat = predict_with_optional_residual(
                model=model,
                gate=gate,
                pred_residual=pred_residual,
                x=x,
                cluster_id_c=cluster_id_device,
                meta=source_meta,
                residual_scale_c=residual_scale_c,
            )
            base_err = yhat_base - y
            base_se_c += (base_err.double() ** 2).sum(dim=(0, 2))
            base_ae_c += base_err.abs().double().sum(dim=(0, 2))
            err = yhat - y
            se_c += (err.double() ** 2).sum(dim=(0, 2))
            ae_c += err.abs().double().sum(dim=(0, 2))
            denom += int(y.shape[0] * y.shape[2])

    mse_c = (se_c / max(denom, 1)).detach().cpu().numpy()
    mae_c = (ae_c / max(denom, 1)).detach().cpu().numpy()
    base_mse_c = (base_se_c / max(denom, 1)).detach().cpu().numpy()
    base_mae_c = (base_ae_c / max(denom, 1)).detach().cpu().numpy()
    selected_head = cluster_id_c.detach().cpu().numpy()
    metrics = {
        "channel": value_cols,
        "MAE": mae_c,
        "MSE": mse_c,
        "MAE_base": base_mae_c,
        "MSE_base": base_mse_c,
        "selected_head": selected_head,
        "cluster_id": selected_head,
        "corr_max": corr_max,
    }
    for k in range(corr_np.shape[1]):
        metrics[f"corr_head_{k}"] = corr_np[:, k]
    pd.DataFrame(metrics).to_csv(run_dir / "test_metrics.csv", index=False)
    pd.DataFrame(
        {
            "channel": value_cols,
            "selected_head": selected_head,
            "corr_max": corr_max,
            **{f"corr_head_{k}": corr_np[:, k] for k in range(corr_np.shape[1])},
        }
    ).to_csv(run_dir / "cluster_assignment.csv", index=False)
    np.save(run_dir / "channel_cluster_corr.npy", corr_np)

    counts = pd.Series(selected_head).value_counts().sort_index()
    summary = {
        "avg_mae": float(np.mean(mae_c)),
        "avg_mse": float(np.mean(mse_c)),
        "base_avg_mae": float(np.mean(base_mae_c)),
        "base_avg_mse": float(np.mean(base_mse_c)),
        "elapsed_sec": float(time.perf_counter() - t0),
        "num_test_windows": num_windows,
        "num_channels": channels,
        "pred_len": pred_len,
        "input_len": input_len,
        "head_selection_policy": head_policy,
        "route_mode": route_mode,
        "route_fit_scope": "train",
        "predictor_variant": "full_moe_residual" if pred_residual is not None else "base",
        "protocol": "legacy_direct_target_resample_then_h96",
        "normalize_train_only": bool(target["normalize_train_only"]),
        "residual_scale_mean": float(residual_scale_c.detach().mean().item()) if residual_scale_c is not None else 0.0,
        **prep_info,
    }
    with (run_dir / "transfer_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return {
        "status": "ok",
        "source": "ETTm1",
        "target": name,
        "pred_len": pred_len,
        "input_len": input_len,
        "avg_mse": summary["avg_mse"],
        "avg_mae": summary["avg_mae"],
        "base_avg_mse": summary["base_avg_mse"],
        "base_avg_mae": summary["base_avg_mae"],
        "moe_gain_mse": summary["base_avg_mse"] - summary["avg_mse"],
        "moe_gain_mae": summary["base_avg_mae"] - summary["avg_mae"],
        "num_channels": channels,
        "num_test_windows": num_windows,
        "head_selection_policy": summary["head_selection_policy"],
        "route_mode": summary["route_mode"],
        "route_fit_scope": summary["route_fit_scope"],
        "predictor_variant": summary["predictor_variant"],
        "protocol": summary["protocol"],
        "normalize_train_only": summary["normalize_train_only"],
        "original_step_minutes": prep_info["original_step_minutes"],
        "prepared_step_minutes": prep_info["prepared_step_minutes"],
        "original_rows": prep_info["original_rows"],
        "prepared_rows": prep_info["prepared_rows"],
        "target_csv": str(target["csv_path"]),
        "prepared_csv": prep_info["prepared_csv"],
        "out_dir": str(run_dir),
        "mean_corr_max": float(np.mean(corr_max)),
        "min_corr_max": float(np.min(corr_max)),
        "cluster_sizes": ";".join(f"{int(k)}:{int(v)}" for k, v in counts.items()),
        "residual_scale_mean": summary["residual_scale_mean"],
        "error": "",
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields = [
        "status",
        "source",
        "target",
        "pred_len",
        "input_len",
        "avg_mse",
        "avg_mae",
        "base_avg_mse",
        "base_avg_mae",
        "moe_gain_mse",
        "moe_gain_mae",
        "num_channels",
        "num_test_windows",
        "head_selection_policy",
        "route_mode",
        "route_fit_scope",
        "predictor_variant",
        "protocol",
        "normalize_train_only",
        "original_step_minutes",
        "prepared_step_minutes",
        "original_rows",
        "prepared_rows",
        "target_csv",
        "prepared_csv",
        "out_dir",
        "mean_corr_max",
        "min_corr_max",
        "cluster_sizes",
        "residual_scale_mean",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_result_only_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields = [
        "status",
        "source",
        "target",
        "pred_len",
        "input_len",
        "test_mse",
        "test_mae",
        "num_channels",
        "num_test_windows",
        "head_selection_policy",
        "route_mode",
        "route_fit_scope",
        "predictor_variant",
        "protocol",
        "normalize_train_only",
        "original_step_minutes",
        "prepared_step_minutes",
        "target_csv",
        "prepared_csv",
        "out_dir",
        "error",
    ]
    normalized = []
    for row in rows:
        normalized.append(
            {
                "status": row.get("status", ""),
                "source": row.get("source", ""),
                "target": row.get("target", ""),
                "pred_len": row.get("pred_len", ""),
                "input_len": row.get("input_len", ""),
                "test_mse": row.get("avg_mse", ""),
                "test_mae": row.get("avg_mae", ""),
                "num_channels": row.get("num_channels", ""),
                "num_test_windows": row.get("num_test_windows", ""),
                "head_selection_policy": row.get("head_selection_policy", ""),
                "route_mode": row.get("route_mode", ""),
                "route_fit_scope": row.get("route_fit_scope", ""),
                "predictor_variant": row.get("predictor_variant", ""),
                "protocol": row.get("protocol", ""),
                "normalize_train_only": row.get("normalize_train_only", ""),
                "original_step_minutes": row.get("original_step_minutes", ""),
                "prepared_step_minutes": row.get("prepared_step_minutes", ""),
                "target_csv": row.get("target_csv", ""),
                "prepared_csv": row.get("prepared_csv", ""),
                "out_dir": row.get("out_dir", ""),
                "error": row.get("error", ""),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(normalized)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default=str(ROOT / "outputs" / "ettm1_h96_legacy_transfer"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--route-mode", choices=["pearson", "cycle_template"], default=None)
    parser.add_argument("--normalize-train-only", choices=["default", "true", "false"], default="default")
    parser.add_argument("--result-only", action="store_true")
    parser.add_argument("--targets", nargs="*", default=[str(t["target"]) for t in TARGETS])
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve()
    selected = {t.lower() for t in args.targets}
    targets = [dict(t) for t in TARGETS if str(t["target"]).lower() in selected]
    if not targets:
        raise ValueError(f"No targets selected: {args.targets}")
    for t in targets:
        if args.route_mode is not None:
            t["route_mode"] = args.route_mode
        if args.normalize_train_only != "default":
            t["normalize_train_only"] = args.normalize_train_only == "true"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    memory_path = ensure_source_memory(out_root)
    memory = torch.load(memory_path, map_location="cpu")
    prototypes_kt = memory["prototypes_kt"].detach().cpu()
    model, source_meta, ckpt = load_source_model(device)
    gate, pred_residual, _ = build_moe_modules(ckpt, source_meta, device)

    rows: List[Dict[str, object]] = []
    for target in targets:
        print(f"[legacy-transfer] ETTm1 -> {target['target']}")
        try:
            row = evaluate_target(target, out_root, model, gate, pred_residual, source_meta, prototypes_kt, device)
            print(
                f"[ok] {target['target']} mse={row['avg_mse']:.6f} "
                f"mae={row['avg_mae']:.6f} base_mse={row['base_avg_mse']:.6f} "
                f"gain={row['moe_gain_mse']:.6f} windows={row['num_test_windows']}"
            )
        except Exception as exc:
            row = {
                "status": "failed",
                "source": "ETTm1",
                "target": str(target["target"]),
                "pred_len": 96,
                "input_len": 336,
                "avg_mse": "",
                "avg_mae": "",
                "base_avg_mse": "",
                "base_avg_mae": "",
                "moe_gain_mse": "",
                "moe_gain_mae": "",
                "num_channels": "",
                "num_test_windows": "",
                "head_selection_policy": "channel_cluster_pearson_argmax",
                "route_mode": str(target.get("route_mode", "pearson")),
                "route_fit_scope": "train",
                "predictor_variant": "full_moe_residual",
                "protocol": "legacy_direct_target_resample_then_h96",
                "normalize_train_only": bool(target["normalize_train_only"]),
                "original_step_minutes": "",
                "prepared_step_minutes": "",
                "original_rows": "",
                "prepared_rows": "",
                "target_csv": str(target["csv_path"]),
                "prepared_csv": "",
                "out_dir": str(out_root / "runs" / str(target["target"])),
                "mean_corr_max": "",
                "min_corr_max": "",
                "cluster_sizes": "",
                "residual_scale_mean": "",
                "error": str(exc),
            }
            print(f"[failed] {target['target']}: {exc}")
        rows.append(row)
        if args.result_only:
            write_result_only_csv(out_root / "transfer.csv", rows)
        else:
            write_csv(out_root / "transfer.csv", rows)
    if args.result_only:
        write_result_only_csv(out_root / "transfer.csv", rows)
    else:
        write_csv(out_root / "transfer.csv", rows)
    print(f"Saved: {out_root / 'transfer.csv'}")


if __name__ == "__main__":
    main()
