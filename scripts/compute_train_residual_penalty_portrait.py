from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.reader import read_csv_time_series
from src.data.windows import WindowTensorDataset, global_zscore, make_lazy_strict_window_dataset, make_strict_windows
from src.models.cluster_predictor import build_cluster_predictor
from src.models.moe_gate import scatter_mean_bcf_to_bkf
from src.models.penalties import build_penalty_compute
from src.utils.cluster_memory import load_cluster_checkpoint


PENALTY_NAMES = [
    "level",
    "amp_under",
    "delta",
    "diff_amp",
    "d2_match",
    "direction",
    "trend",
    "corr",
    "range",
    "seasonal_align",
]

DEFAULT_MAX_ABS_MSE_CORR = 0.80


@dataclass(frozen=True)
class CellSpec:
    name: str
    config_path: str
    checkpoint_path: str


DEFAULT_CELLS = [
    CellSpec(
        name="PEMS04_H96",
        config_path="outputs/pems_depth_rollout/configs/PEMS04_H96_hid192_b2.yaml",
        checkpoint_path="outputs/pems_depth_rollout/runs/PEMS04_H96_hid192_b2/best_checkpoint.pt",
    ),
    CellSpec(
        name="PEMS03_H96",
        config_path="outputs/pems_depth_rollout/configs/PEMS03_H96_hid192_b2.yaml",
        checkpoint_path="outputs/pems_depth_rollout/runs/PEMS03_H96_hid192_b2/best_checkpoint.pt",
    ),
    CellSpec(
        name="PEMS07_H96",
        config_path="outputs/pems_depth_rollout/configs/PEMS07_H96_hid192_b2.yaml",
        checkpoint_path="outputs/pems_depth_rollout/runs/PEMS07_H96_hid192_b2/best_checkpoint.pt",
    ),
    CellSpec(
        name="PEMS08_H96",
        config_path="outputs/pems08_h96_backbone_capacity/configs/hid192_b2.yaml",
        checkpoint_path="outputs/pems08_h96_backbone_capacity/runs/hid192_b2/best_checkpoint.pt",
    ),
    CellSpec(
        name="ETTm1_H96",
        config_path="outputs/input96_mse_gate_cluster_moe_retrain_20260616/configs/ETTm1/H96/mse_gate_w005_softprior.yaml",
        checkpoint_path=(
            "outputs/fresh_input_len96_20260610_ettm1_backbone_arch_search/runs/ETTm1/H96/"
            "backbone_arch/patchtst_h128_p16s8_l2_do01_wd1e4_mae04/best_checkpoint.pt"
        ),
    ),
    CellSpec(
        name="Weather_H96",
        config_path="outputs/input96_mse_gate_cluster_moe_retrain_20260616/configs/Weather/H96/mse_gate_w005_softprior.yaml",
        checkpoint_path=(
            "outputs/fresh_input_len96_20260614_weather_h96_mae_arch_refine2/"
            "runs/r4_s005_mse03_mae20_valmae/best_checkpoint.pt"
        ),
    ),
]


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_train_series(cfg: dict[str, Any]) -> tuple[torch.Tensor, list[str], int]:
    data_cfg = cfg["data"]
    data_tc, channel_names = read_csv_time_series(
        str(resolve(str(data_cfg["csv_path"]))),
        date_col=int(data_cfg.get("date_col", 0)),
    )
    max_rows = int(data_cfg.get("max_rows", 0) or 0)
    if max_rows > 0:
        data_tc = data_tc[:max_rows]

    T = int(data_tc.shape[0])
    train_ratio = float(data_cfg["train_ratio"])
    val_ratio = float(data_cfg.get("val_ratio", 0.0))
    test_ratio = float(data_cfg.get("test_ratio", 1.0 - train_ratio - val_ratio))
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1.0e-5:
        raise ValueError("data train/val/test ratios must sum to 1.")
    t_train = int(T * train_ratio)

    norm_cfg = cfg.get("normalize", {}) or {}
    if bool(norm_cfg.get("global_zscore", False)):
        if bool(norm_cfg.get("train_only", False)):
            train_seg = data_tc[:t_train]
            mean_c = train_seg.mean(dim=0, keepdim=True)
            std_c = train_seg.std(dim=0, keepdim=True).clamp_min(1.0e-6)
            data_tc = (data_tc - mean_c) / std_c
        else:
            data_tc, _, _ = global_zscore(data_tc)
    return data_tc.detach().cpu(), list(channel_names), t_train


def build_train_loader(
    cfg: dict[str, Any],
    data_tc: torch.Tensor,
    t_train: int,
    *,
    batch_size: int,
    materialize_windows: bool,
    pin_memory: bool,
) -> DataLoader:
    L = int(cfg["window"]["input_len"])
    H = int(cfg["window"]["pred_len"])
    if materialize_windows:
        xtr, ytr = make_strict_windows(data_tc, L, H, 0, int(t_train))
        dataset = WindowTensorDataset(xtr, ytr)
    else:
        dataset = make_lazy_strict_window_dataset(data_tc, L, H, 0, int(t_train))
    if len(dataset) <= 0:
        raise ValueError("No train windows are available for the requested config.")
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=bool(pin_memory),
    )


def _tensor_long(value: Any, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=torch.long)
    return torch.tensor(value, device=device, dtype=torch.long)


def load_backbone_model(
    cfg: dict[str, Any],
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.Tensor, dict[str, Any]]:
    ckpt = load_cluster_checkpoint(str(checkpoint_path), device=device)
    meta = ckpt.get("meta", {}) or {}
    if not meta:
        raise ValueError(f"Checkpoint has no meta: {checkpoint_path}")

    cluster_id_c = _tensor_long(meta["cluster_id_c"], device)
    input_len = int(meta["input_len"])
    pred_len = int(meta["pred_len"])
    if input_len != int(cfg["window"]["input_len"]) or pred_len != int(cfg["window"]["pred_len"]):
        raise ValueError(
            f"Checkpoint window {input_len}/{pred_len} does not match config "
            f"{cfg['window']['input_len']}/{cfg['window']['pred_len']}."
        )

    model = build_cluster_predictor(
        num_clusters=int(meta["K"]),
        input_len=input_len,
        pred_len=pred_len,
        model_cfg=dict(meta.get("model_cfg", cfg.get("model", {}))),
        num_channels=int(meta.get("num_channels", int(cluster_id_c.numel()))),
        cluster_id_c=cluster_id_c,
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, cluster_id_c, meta


def finalize_pearson_corr(
    *,
    sum_x: np.ndarray,
    sum_y: np.ndarray,
    sum_xx: np.ndarray,
    sum_yy: np.ndarray,
    sum_xy: np.ndarray,
    count: np.ndarray,
    eps: float = 1.0e-12,
) -> np.ndarray:
    sx = np.asarray(sum_x, dtype=np.float64)
    sy = np.asarray(sum_y, dtype=np.float64)
    sxx = np.asarray(sum_xx, dtype=np.float64)
    syy = np.asarray(sum_yy, dtype=np.float64)
    sxy = np.asarray(sum_xy, dtype=np.float64)
    n = np.asarray(count, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        cov = sxy - (sx * sy / np.maximum(n, 1.0))
        var_x = sxx - (sx * sx / np.maximum(n, 1.0))
        var_y = syy - (sy * sy / np.maximum(n, 1.0))
        denom = np.sqrt(np.maximum(var_x, 0.0) * np.maximum(var_y, 0.0))
        corr = np.divide(cov, denom, out=np.zeros_like(cov, dtype=np.float64), where=(n > 1.0) & (denom > eps))
    return np.clip(np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)


def _mse_corr_matrix_for_selection(
    penalty_mse_corr: np.ndarray | None,
    *,
    K: int,
    P: int,
) -> np.ndarray | None:
    if penalty_mse_corr is None:
        return None
    corr = np.asarray(penalty_mse_corr, dtype=np.float64)
    if corr.ndim == 1:
        if corr.shape[0] != P:
            raise ValueError("penalty_mse_corr must have length P when passed as a vector.")
        return np.broadcast_to(corr.reshape(1, P), (K, P))
    if corr.ndim == 2:
        if corr.shape != (K, P):
            raise ValueError("penalty_mse_corr matrix must have shape [K,P].")
        return corr
    raise ValueError("penalty_mse_corr must be either [P] or [K,P].")


def select_pool_top3(
    portrait_raw: np.ndarray,
    penalty_global_mean: np.ndarray,
    penalty_names: list[str],
    *,
    penalty_mse_corr: np.ndarray | None = None,
    max_abs_mse_corr: float | None = None,
) -> dict[str, list[str]]:
    portrait = np.asarray(portrait_raw, dtype=np.float64)
    global_mean = np.asarray(penalty_global_mean, dtype=np.float64)
    if portrait.ndim != 2:
        raise ValueError("portrait_raw must be a [K,P] matrix.")
    if global_mean.ndim != 1 or global_mean.shape[0] != portrait.shape[1]:
        raise ValueError("penalty_global_mean must be a [P] vector matching portrait_raw.")
    if len(penalty_names) != portrait.shape[1]:
        raise ValueError("penalty_names length must match portrait_raw columns.")

    denom = np.maximum(global_mean, 1.0e-12)
    scores = portrait / denom.reshape(1, -1)
    corr_matrix = (
        _mse_corr_matrix_for_selection(penalty_mse_corr, K=scores.shape[0], P=scores.shape[1])
        if max_abs_mse_corr is not None
        else None
    )
    corr_threshold = float(max_abs_mse_corr) if max_abs_mse_corr is not None else None
    selected: dict[str, list[str]] = {}
    for k in range(scores.shape[0]):
        order = sorted(range(scores.shape[1]), key=lambda idx: (-float(scores[k, idx]), idx))
        if corr_matrix is not None and corr_threshold is not None:
            order = [
                idx
                for idx in order
                if not (
                    np.isfinite(corr_matrix[k, idx])
                    and abs(float(corr_matrix[k, idx])) > corr_threshold
                )
            ]
        selected[str(k)] = [str(penalty_names[idx]) for idx in order[: min(3, len(order))]]
    return selected


def excluded_by_mse_corr(
    penalty_names: list[str],
    penalty_mse_corr: np.ndarray,
    *,
    max_abs_mse_corr: float,
) -> dict[str, list[str]]:
    corr = np.asarray(penalty_mse_corr, dtype=np.float64)
    if corr.ndim != 2:
        raise ValueError("penalty_mse_corr must be [K,P] for per-cluster exclusions.")
    if corr.shape[1] != len(penalty_names):
        raise ValueError("penalty_names length must match penalty_mse_corr columns.")
    threshold = float(max_abs_mse_corr)
    out: dict[str, list[str]] = {}
    for k in range(corr.shape[0]):
        out[str(k)] = [
            str(penalty_names[p])
            for p in range(corr.shape[1])
            if np.isfinite(corr[k, p]) and abs(float(corr[k, p])) > threshold
        ]
    return out


@torch.no_grad()
def compute_portrait(
    model: torch.nn.Module,
    loader: DataLoader,
    cluster_id_c: torch.Tensor,
    *,
    K: int,
    penalty_names: list[str],
    jump_threshold: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    penalty_compute = build_penalty_compute(penalty_names, jump_thr=float(jump_threshold))
    P = len(penalty_names)
    sum_kp = torch.zeros(int(K), P, device=device, dtype=torch.float64)
    sum_p = torch.zeros(P, device=device, dtype=torch.float64)
    sum_pen_global = torch.zeros(P, device=device, dtype=torch.float64)
    sum_pen2_global = torch.zeros(P, device=device, dtype=torch.float64)
    sum_pen_mse_global = torch.zeros(P, device=device, dtype=torch.float64)
    sum_mse_global = torch.zeros(P, device=device, dtype=torch.float64)
    sum_mse2_global = torch.zeros(P, device=device, dtype=torch.float64)
    count_corr_global = torch.zeros(P, device=device, dtype=torch.float64)
    sum_pen_cluster = torch.zeros(int(K), P, device=device, dtype=torch.float64)
    sum_pen2_cluster = torch.zeros(int(K), P, device=device, dtype=torch.float64)
    sum_pen_mse_cluster = torch.zeros(int(K), P, device=device, dtype=torch.float64)
    sum_mse_cluster = torch.zeros(int(K), P, device=device, dtype=torch.float64)
    sum_mse2_cluster = torch.zeros(int(K), P, device=device, dtype=torch.float64)
    count_corr_cluster = torch.zeros(int(K), P, device=device, dtype=torch.float64)
    count_windows = 0
    count_global = 0

    for x, y, _ in loader:
        x = x.to(device=device, non_blocking=True)
        y = y.to(device=device, non_blocking=True)
        y_base = model(x, cluster_id_c)
        pen_bcp = penalty_compute(y_base, y).to(dtype=torch.float64)
        mse_bc = (y_base - y).pow(2).mean(dim=-1).to(dtype=torch.float64)
        pen_bkp = scatter_mean_bcf_to_bkf(pen_bcp, cluster_id_c, int(K))
        sum_kp += pen_bkp.sum(dim=0)
        sum_p += pen_bcp.sum(dim=(0, 1))
        sum_pen_global += pen_bcp.sum(dim=(0, 1))
        sum_pen2_global += pen_bcp.pow(2).sum(dim=(0, 1))
        sum_pen_mse_global += (pen_bcp * mse_bc.unsqueeze(-1)).sum(dim=(0, 1))
        mse_sum = mse_bc.sum()
        mse2_sum = mse_bc.pow(2).sum()
        event_count = int(mse_bc.numel())
        sum_mse_global += mse_sum
        sum_mse2_global += mse2_sum
        count_corr_global += float(event_count)

        for k in range(int(K)):
            channel_mask = cluster_id_c == int(k)
            cluster_events = int(channel_mask.sum().item()) * int(mse_bc.shape[0])
            if cluster_events <= 0:
                continue
            pen_bcp_k = pen_bcp[:, channel_mask, :]
            mse_bc_k = mse_bc[:, channel_mask]
            sum_pen_cluster[k] += pen_bcp_k.sum(dim=(0, 1))
            sum_pen2_cluster[k] += pen_bcp_k.pow(2).sum(dim=(0, 1))
            sum_pen_mse_cluster[k] += (pen_bcp_k * mse_bc_k.unsqueeze(-1)).sum(dim=(0, 1))
            sum_mse_cluster[k] += mse_bc_k.sum()
            sum_mse2_cluster[k] += mse_bc_k.pow(2).sum()
            count_corr_cluster[k] += float(cluster_events)

        count_windows += int(pen_bkp.shape[0])
        count_global += int(pen_bcp.shape[0] * pen_bcp.shape[1])

    if count_windows <= 0 or count_global <= 0:
        raise ValueError("No train windows were processed.")
    portrait_raw = (sum_kp / float(count_windows)).detach().cpu().numpy()
    penalty_global_mean = (sum_p / float(count_global)).detach().cpu().numpy()
    penalty_mse_corr = finalize_pearson_corr(
        sum_x=sum_pen_global.detach().cpu().numpy(),
        sum_y=sum_mse_global.detach().cpu().numpy(),
        sum_xx=sum_pen2_global.detach().cpu().numpy(),
        sum_yy=sum_mse2_global.detach().cpu().numpy(),
        sum_xy=sum_pen_mse_global.detach().cpu().numpy(),
        count=count_corr_global.detach().cpu().numpy(),
    )
    penalty_mse_corr_by_cluster = finalize_pearson_corr(
        sum_x=sum_pen_cluster.detach().cpu().numpy(),
        sum_y=sum_mse_cluster.detach().cpu().numpy(),
        sum_xx=sum_pen2_cluster.detach().cpu().numpy(),
        sum_yy=sum_mse2_cluster.detach().cpu().numpy(),
        sum_xy=sum_pen_mse_cluster.detach().cpu().numpy(),
        count=count_corr_cluster.detach().cpu().numpy(),
    )
    return portrait_raw, penalty_global_mean, penalty_mse_corr, penalty_mse_corr_by_cluster


def build_cell_payload(
    *,
    portrait_raw: np.ndarray,
    penalty_global_mean: np.ndarray,
    penalty_mse_corr: np.ndarray,
    penalty_mse_corr_by_cluster: np.ndarray,
    max_abs_mse_corr: float,
    cluster_id_c: torch.Tensor,
    K: int,
    channel_names: list[str],
    penalty_names: list[str],
) -> dict[str, Any]:
    cluster_id = [int(v) for v in cluster_id_c.detach().cpu().to(dtype=torch.long).tolist()]
    cluster_sizes = torch.bincount(
        cluster_id_c.detach().cpu().to(dtype=torch.long),
        minlength=int(K),
    ).tolist()
    payload: dict[str, Any] = {
        "n_clusters": int(K),
        "cluster_sizes": [int(v) for v in cluster_sizes],
        "cluster_id": cluster_id,
    }
    if len(channel_names) == len(cluster_id):
        payload["channel_names"] = [str(name) for name in channel_names]
    payload.update(
        {
            "penalty_global_mean": [float(v) for v in penalty_global_mean.tolist()],
            "penalty_mse_corr": [float(v) for v in penalty_mse_corr.tolist()],
            "penalty_mse_corr_by_cluster": [
                [float(v) for v in row] for row in penalty_mse_corr_by_cluster.tolist()
            ],
            "mse_corr_exclusion": {
                "max_abs_corr": float(max_abs_mse_corr),
                "excluded_by_cluster": excluded_by_mse_corr(
                    penalty_names,
                    penalty_mse_corr_by_cluster,
                    max_abs_mse_corr=float(max_abs_mse_corr),
                ),
            },
            "portrait_raw": [[float(v) for v in row] for row in portrait_raw.tolist()],
            "selected_pool_top3": select_pool_top3(
                portrait_raw,
                penalty_global_mean,
                penalty_names,
                penalty_mse_corr=penalty_mse_corr_by_cluster,
                max_abs_mse_corr=float(max_abs_mse_corr),
            ),
        }
    )
    return payload


def run_cell(
    spec: CellSpec,
    *,
    penalty_names: list[str],
    batch_size: int | None,
    device: torch.device,
    materialize_windows: bool,
    max_abs_mse_corr: float = DEFAULT_MAX_ABS_MSE_CORR,
) -> dict[str, Any]:
    cfg_path = resolve(spec.config_path)
    checkpoint_path = resolve(spec.checkpoint_path)
    cfg = read_yaml(cfg_path)
    data_tc, channel_names, t_train = load_train_series(cfg)
    bs = int(batch_size or cfg.get("train", {}).get("batch_size", 64))
    pin_memory = device.type == "cuda"
    loader = build_train_loader(
        cfg,
        data_tc,
        t_train,
        batch_size=bs,
        materialize_windows=materialize_windows,
        pin_memory=pin_memory,
    )
    model, cluster_id_c, meta = load_backbone_model(cfg, checkpoint_path, device=device)
    K = int(meta["K"])
    portrait_raw, penalty_global_mean, penalty_mse_corr, penalty_mse_corr_by_cluster = compute_portrait(
        model,
        loader,
        cluster_id_c,
        K=K,
        penalty_names=penalty_names,
        jump_threshold=float(cfg.get("penalties", {}).get("jump_threshold", 0.6)),
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return build_cell_payload(
        portrait_raw=portrait_raw,
        penalty_global_mean=penalty_global_mean,
        penalty_mse_corr=penalty_mse_corr,
        penalty_mse_corr_by_cluster=penalty_mse_corr_by_cluster,
        max_abs_mse_corr=float(max_abs_mse_corr),
        cluster_id_c=cluster_id_c,
        K=K,
        channel_names=channel_names,
        penalty_names=penalty_names,
    )


def parse_cells(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute frozen-backbone train residual penalty portraits by cluster."
    )
    parser.add_argument("--out-json", default="outputs/penalty_diagnostic/penalty_portrait.json")
    parser.add_argument("--cells", default=",".join(spec.name for spec in DEFAULT_CELLS))
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max-abs-mse-corr",
        type=float,
        default=DEFAULT_MAX_ABS_MSE_CORR,
        help=(
            "Exclude a penalty from selected_pool_top3 for a cluster when its train-only "
            "abs(corr(penalty(y_base,y), mse(y_base,y))) exceeds this threshold."
        ),
    )
    parser.add_argument(
        "--materialize-windows",
        action="store_true",
        help="Materialize train windows instead of using the equivalent lazy train dataset.",
    )
    args = parser.parse_args()

    os.environ.setdefault("PYTHONUTF8", "1")
    available = {spec.name: spec for spec in DEFAULT_CELLS}
    requested = parse_cells(args.cells)
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(f"Unknown cells: {missing}. Available: {sorted(available)}")

    use_cuda = torch.cuda.is_available() and str(args.device).startswith("cuda")
    device = torch.device(str(args.device) if use_cuda else "cpu")
    result: dict[str, Any] = {
        "meta": {
            "input_len": 96,
            "split": "train",
            "target": "y_base vs y (frozen backbone residual error by shape axis)",
            "mse_corr_exclusion": {
                "enabled": True,
                "max_abs_corr": float(args.max_abs_mse_corr),
                "scope": "per_cluster_train_window_channel",
            },
        },
        "penalty_names": list(PENALTY_NAMES),
        "cells": {},
    }

    for cell_name in requested:
        print(f"[portrait] cell={cell_name} split=train device={device}")
        result["cells"][cell_name] = run_cell(
            available[cell_name],
            penalty_names=list(PENALTY_NAMES),
            batch_size=int(args.batch_size) if int(args.batch_size) > 0 else None,
            device=device,
            materialize_windows=bool(args.materialize_windows),
            max_abs_mse_corr=float(args.max_abs_mse_corr),
        )

    out_path = resolve(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"penalty_portrait_json={out_path}")


if __name__ == "__main__":
    main()
