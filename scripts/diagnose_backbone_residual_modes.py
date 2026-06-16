from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.windows import WindowTensorDataset, make_label_range_windows, make_strict_windows
from src.models.cluster_predictor import build_cluster_predictor
from src.utils.cluster_memory import load_cluster_checkpoint


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_data(cfg: dict[str, Any]) -> tuple[list[str], torch.Tensor, int, int]:
    csv_path = resolve(str(cfg["data"]["csv_path"]))
    date_col = int(cfg["data"].get("date_col", 0))
    raw_df = pd.read_csv(csv_path)
    max_rows = int(cfg.get("data", {}).get("max_rows", 0) or 0)
    if max_rows > 0:
        raw_df = raw_df.iloc[:max_rows].copy()
    value_cols = [col for i, col in enumerate(raw_df.columns) if i != date_col]
    raw_tc = torch.tensor(raw_df[value_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
    T = int(raw_tc.shape[0])
    t_train = int(T * float(cfg["data"].get("train_ratio", 0.7)))
    t_val = int(T * float(float(cfg["data"].get("train_ratio", 0.7)) + float(cfg["data"].get("val_ratio", 0.1))))
    norm_tc = raw_tc.clone()
    if bool((cfg.get("normalize", {}) or {}).get("global_zscore", False)):
        train_tc = norm_tc[:t_train]
        mean_c = train_tc.mean(dim=0)
        std_c = train_tc.std(dim=0).clamp_min(1.0e-6)
        norm_tc = (norm_tc - mean_c.view(1, -1)) / std_c.view(1, -1)
    return value_cols, norm_tc, t_train, t_val


def split_windows(cfg: dict[str, Any], norm_tc: torch.Tensor, *, split: str, t_train: int, t_val: int):
    L = int(cfg["window"]["input_len"])
    H = int(cfg["window"]["pred_len"])
    if split == "train":
        return make_strict_windows(norm_tc, L, H, 0, t_train)
    if split == "val":
        if bool((cfg.get("window", {}) or {}).get("past_context", False)):
            x, y, _ = make_label_range_windows(norm_tc, L, H, t_train, t_val)
            return x, y
        return make_strict_windows(norm_tc, L, H, t_train, t_val)
    raise ValueError("Only train/val are allowed; test is intentionally unsupported.")


def load_model(cfg: dict[str, Any], checkpoint: Path, device: torch.device):
    ckpt = load_cluster_checkpoint(str(checkpoint), device=device)
    meta = ckpt.get("meta", {}) or {}
    cluster_id_c = meta["cluster_id_c"].to(device=device, dtype=torch.long)
    model = build_cluster_predictor(
        num_clusters=int(meta["K"]),
        input_len=int(meta["input_len"]),
        pred_len=int(meta["pred_len"]),
        model_cfg=meta.get("model_cfg", cfg.get("model", {})),
        num_channels=int(meta.get("num_channels", int(cluster_id_c.numel()))),
        cluster_id_c=cluster_id_c,
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, cluster_id_c.detach().cpu().numpy().astype(int), meta


def corr_np(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.reshape(-1)
    bb = b.reshape(-1)
    if aa.size < 2 or float(np.std(aa)) < 1.0e-12 or float(np.std(bb)) < 1.0e-12:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def slope_np(v: np.ndarray) -> float:
    if v.size < 2:
        return 0.0
    x = np.arange(v.size, dtype=np.float64)
    x = x - x.mean()
    y = v.astype(np.float64) - float(v.mean())
    denom = float(np.dot(x, x))
    return 0.0 if denom <= 0.0 else float(np.dot(x, y) / denom)


@torch.no_grad()
def collect_predictions(model, cluster_id_c: np.ndarray, x: torch.Tensor, y: torch.Tensor, *, batch_size: int, device: torch.device):
    loader = DataLoader(WindowTensorDataset(x, y), batch_size=batch_size, shuffle=False, num_workers=0)
    cid = torch.tensor(cluster_id_c, dtype=torch.long, device=device)
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    for xb, yb, _ in loader:
        xb = xb.to(device)
        pred = model(xb, cid).detach().cpu().numpy()
        preds.append(pred)
        ys.append(yb.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(ys, axis=0)


def diagnose_channel(pred_nch: np.ndarray, y_nch: np.ndarray, *, spike_q: float) -> dict[str, float]:
    residual = y_nch - pred_nch
    y_flat = y_nch.reshape(-1)
    pred_flat = pred_nch.reshape(-1)
    res_flat = residual.reshape(-1)
    y_std = float(np.std(y_flat))
    pred_std = float(np.std(pred_flat))
    q_hi = float(np.quantile(y_flat, spike_q))
    spike_mask = y_nch >= q_hi
    spike_count = int(spike_mask.sum())
    if spike_count > 0:
        spike_miss = float((y_nch[spike_mask] - pred_nch[spike_mask]).mean())
        spike_abs = float(np.abs(y_nch[spike_mask] - pred_nch[spike_mask]).mean())
    else:
        spike_miss = 0.0
        spike_abs = 0.0
    return {
        "mse": float(np.mean(res_flat ** 2)),
        "mae": float(np.mean(np.abs(res_flat))),
        "bias_mean_y_minus_pred": float(np.mean(res_flat)),
        "bias_median_y_minus_pred": float(np.median(res_flat)),
        "pred_to_y_std_ratio": pred_std / max(y_std, 1.0e-12),
        "corr_pred_y": corr_np(pred_flat, y_flat),
        "residual_std": float(np.std(res_flat)),
        "trend_slope_residual": slope_np(residual.mean(axis=0)),
        "horizon_bias_abs_mean": float(np.abs(residual.mean(axis=0)).mean()),
        "spike_threshold": q_hi,
        "spike_miss_mean": spike_miss,
        "spike_abs_error": spike_abs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose what the backbone misses from train/val residuals.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--spike-quantile", type=float, default=0.95)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    cfg_path = resolve(args.config)
    cfg = read_yaml(cfg_path)
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    channel_names, norm_tc, t_train, t_val = load_data(cfg)
    x, y = split_windows(cfg, norm_tc, split=str(args.split), t_train=t_train, t_val=t_val)
    model, cluster_id_c, meta = load_model(cfg, resolve(args.checkpoint), device)
    pred, y_np = collect_predictions(model, cluster_id_c, x, y, batch_size=int(args.batch_size), device=device)

    channel_rows = []
    for c, name in enumerate(channel_names):
        row = {
            "channel_idx": c,
            "channel": name,
            "cluster_id": int(cluster_id_c[c]),
        }
        row.update(diagnose_channel(pred[:, c, :], y_np[:, c, :], spike_q=float(args.spike_quantile)))
        channel_rows.append(row)
    channel_df = pd.DataFrame(channel_rows)

    cluster_rows = []
    for k in sorted(channel_df["cluster_id"].unique()):
        sub = channel_df[channel_df["cluster_id"] == k]
        row = {"cluster_id": int(k), "channels": int(len(sub))}
        for col in [
            "mse",
            "mae",
            "bias_mean_y_minus_pred",
            "bias_median_y_minus_pred",
            "pred_to_y_std_ratio",
            "corr_pred_y",
            "residual_std",
            "trend_slope_residual",
            "horizon_bias_abs_mean",
            "spike_miss_mean",
            "spike_abs_error",
        ]:
            row[col] = float(sub[col].mean())
        row["channels_list"] = ";".join(sub["channel"].astype(str).tolist())
        cluster_rows.append(row)
    cluster_df = pd.DataFrame(cluster_rows)

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = str(args.tag or f"{cfg_path.stem}_{args.split}_backbone_residual")
    channel_path = out_dir / f"{tag}_channel_diagnostic.csv"
    cluster_path = out_dir / f"{tag}_cluster_diagnostic.csv"
    json_path = out_dir / f"{tag}_diagnostic_meta.json"
    channel_df.to_csv(channel_path, index=False, encoding="utf-8-sig")
    cluster_df.to_csv(cluster_path, index=False, encoding="utf-8-sig")
    json_path.write_text(
        json.dumps(
            {
                "test_used": False,
                "split": str(args.split),
                "config": str(cfg_path),
                "checkpoint": str(resolve(args.checkpoint)),
                "input_len": int(meta["input_len"]),
                "pred_len": int(meta["pred_len"]),
                "cluster_source": "checkpoint",
                "spike_quantile": float(args.spike_quantile),
                "channel_diagnostic_csv": str(channel_path),
                "cluster_diagnostic_csv": str(cluster_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    printable_cols = [
        "cluster_id",
        "channels",
        "mse",
        "mae",
        "bias_mean_y_minus_pred",
        "pred_to_y_std_ratio",
        "corr_pred_y",
        "horizon_bias_abs_mean",
        "spike_miss_mean",
        "spike_abs_error",
    ]
    print(cluster_df[printable_cols].to_string(index=False))
    print(f"channel_diagnostic_csv={channel_path}")
    print(f"cluster_diagnostic_csv={cluster_path}")


if __name__ == "__main__":
    main()
