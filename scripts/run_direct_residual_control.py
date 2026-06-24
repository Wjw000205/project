from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compute_train_residual_penalty_portrait import load_backbone_model, load_train_series  # noqa: E402
from src.data.windows import make_lazy_label_range_window_dataset, make_lazy_strict_window_dataset  # noqa: E402


CELL_SPECS: dict[str, dict[str, str]] = {
    "ETTh1_H96": {
        "config_path": "outputs/input96_mse_gate_cluster_moe_retrain_20260616/configs/ETTh1/H96/mse_gate_w002_softprior.yaml",
        "checkpoint_path": "outputs/fresh_input_len96_20260610_etth1_ettm1_backbone_probe/runs/ETTh1/H96/common_backbone_h96/mlp_h128_do0_wd1e4_mae04/best_checkpoint.pt",
    },
    "ETTm2_H96": {
        "config_path": "outputs/input96_mse_gate_cluster_moe_retrain_20260616/configs/ETTm2/H96/mse_gate_w002_top2.yaml",
        "checkpoint_path": "outputs/fresh_input_len96_20260610_ettm2_backbone_lowdrop/runs/ETTm2/H96/common_backbone_h96/channel_h256_do0_wd1e3_mae06/best_checkpoint.pt",
    },
}


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class DirectClusterResidual(nn.Module):
    def __init__(self, num_clusters: int, input_len: int, pred_len: int, hidden_dim: int = 32):
        super().__init__()
        self.K = int(num_clusters)
        self.L = int(input_len)
        self.H = int(pred_len)
        self.D = int(input_len + pred_len)
        self.hidden_dim = int(hidden_dim)
        self.W1 = nn.ParameterList([nn.Parameter(torch.empty(self.D, self.hidden_dim)) for _ in range(self.K)])
        self.b1 = nn.ParameterList([nn.Parameter(torch.zeros(self.hidden_dim)) for _ in range(self.K)])
        self.W2 = nn.ParameterList([nn.Parameter(torch.empty(self.hidden_dim, self.H)) for _ in range(self.K)])
        self.b2 = nn.ParameterList([nn.Parameter(torch.zeros(self.H)) for _ in range(self.K)])
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for w in self.W1:
            nn.init.xavier_uniform_(w)
        for w in self.W2:
            nn.init.zeros_(w)

    def forward(self, x_bcl: torch.Tensor, y_base_bch: torch.Tensor, cluster_id_c: torch.Tensor) -> torch.Tensor:
        last = x_bcl[..., -1:]
        feat = torch.cat([x_bcl - last, y_base_bch - last], dim=-1)
        cluster_id_c = cluster_id_c.to(device=x_bcl.device, dtype=torch.long)
        W1 = torch.stack(list(self.W1), dim=0).index_select(0, cluster_id_c)
        b1 = torch.stack(list(self.b1), dim=0).index_select(0, cluster_id_c)
        W2 = torch.stack(list(self.W2), dim=0).index_select(0, cluster_id_c)
        b2 = torch.stack(list(self.b2), dim=0).index_select(0, cluster_id_c)
        h = F.gelu(torch.einsum("bcd,cdm->bcm", feat, W1) + b1.unsqueeze(0))
        return torch.einsum("bcm,cmh->bch", h, W2) + b2.unsqueeze(0)


def split_points(cfg: dict[str, Any], T: int) -> tuple[int, int]:
    data_cfg = cfg["data"]
    train_ratio = float(data_cfg["train_ratio"])
    val_ratio = float(data_cfg.get("val_ratio", 0.0))
    t_train = int(T * train_ratio)
    t_val = int(T * (train_ratio + val_ratio))
    return t_train, t_val


def make_loader(cfg: dict[str, Any], data_tc: torch.Tensor, split: str, batch_size: int) -> DataLoader:
    L = int(cfg["window"]["input_len"])
    H = int(cfg["window"]["pred_len"])
    T = int(data_tc.shape[0])
    t_train, t_val = split_points(cfg, T)
    past_context = bool(cfg.get("window", {}).get("past_context", False))
    if split == "train":
        dataset = make_lazy_strict_window_dataset(data_tc, L, H, 0, t_train)
    elif split == "val":
        if past_context:
            dataset, _ = make_lazy_label_range_window_dataset(data_tc, L, H, t_train, t_val)
        else:
            dataset = make_lazy_strict_window_dataset(data_tc, L, H, t_train, t_val)
    elif split == "test":
        if past_context:
            dataset, _ = make_lazy_label_range_window_dataset(data_tc, L, H, t_val, T)
        else:
            dataset = make_lazy_strict_window_dataset(data_tc, L, H, t_val, T)
    else:
        raise ValueError(f"Unsupported split: {split}")
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=(split == "train"), num_workers=0, pin_memory=True)


@torch.no_grad()
def evaluate(
    backbone: nn.Module,
    residual: DirectClusterResidual,
    loader: DataLoader,
    cluster_id_c: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    base_sq = 0.0
    pred_sq = 0.0
    base_abs = 0.0
    pred_abs = 0.0
    numel = 0
    residual_sq = 0.0
    base_pred_sq = 0.0
    for x, y, _ in loader:
        x = x.to(device=device, non_blocking=True)
        y = y.to(device=device, non_blocking=True)
        y_base = backbone(x, cluster_id_c)
        r = residual(x, y_base, cluster_id_c)
        y_hat = y_base + r
        base_err = y_base - y
        pred_err = y_hat - y
        base_sq += float(base_err.pow(2).sum().item())
        pred_sq += float(pred_err.pow(2).sum().item())
        base_abs += float(base_err.abs().sum().item())
        pred_abs += float(pred_err.abs().sum().item())
        residual_sq += float(r.pow(2).sum().item())
        base_pred_sq += float(y_base.pow(2).sum().item())
        numel += int(y.numel())
    base_mse = base_sq / max(numel, 1)
    pred_mse = pred_sq / max(numel, 1)
    base_mae = base_abs / max(numel, 1)
    pred_mae = pred_abs / max(numel, 1)
    return {
        "base_mse": base_mse,
        "residual_mse": pred_mse,
        "base_mae": base_mae,
        "residual_mae": pred_mae,
        "mse_reduction_pct": (base_mse - pred_mse) / base_mse * 100.0 if base_mse else 0.0,
        "mae_reduction_pct": (base_mae - pred_mae) / base_mae * 100.0 if base_mae else 0.0,
        "residual_base_rms_ratio": (residual_sq / max(base_pred_sq, 1.0)) ** 0.5,
    }


def run_cell(cell: str, spec: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    cfg = read_yaml(resolve(spec["config_path"]))
    data_tc, channel_names, _ = load_train_series(cfg)
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    backbone, cluster_id_c, meta = load_backbone_model(cfg, resolve(spec["checkpoint_path"]), device=device)
    residual = DirectClusterResidual(
        num_clusters=int(meta["K"]),
        input_len=int(meta["input_len"]),
        pred_len=int(meta["pred_len"]),
        hidden_dim=int(args.hidden_dim),
    ).to(device)
    train_loader = make_loader(cfg, data_tc, "train", int(args.batch_size or cfg.get("train", {}).get("batch_size", 64)))
    val_loader = make_loader(cfg, data_tc, "val", int(args.batch_size or cfg.get("train", {}).get("batch_size", 64)))
    test_loader = make_loader(cfg, data_tc, "test", int(args.batch_size or cfg.get("train", {}).get("batch_size", 64)))
    opt = torch.optim.AdamW(residual.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best_state = copy.deepcopy(residual.state_dict())
    best_val_mse = float("inf")
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        residual.train()
        for x, y, _ in train_loader:
            x = x.to(device=device, non_blocking=True)
            y = y.to(device=device, non_blocking=True)
            with torch.no_grad():
                y_base = backbone(x, cluster_id_c)
            y_hat = y_base + residual(x, y_base, cluster_id_c)
            loss = F.mse_loss(y_hat, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(residual.parameters(), float(args.grad_clip))
            opt.step()
        residual.eval()
        train_metrics = evaluate(backbone, residual, train_loader, cluster_id_c, device)
        val_metrics = evaluate(backbone, residual, val_loader, cluster_id_c, device)
        if val_metrics["residual_mse"] < best_val_mse:
            best_val_mse = float(val_metrics["residual_mse"])
            best_state = copy.deepcopy(residual.state_dict())
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"{cell} epoch={epoch} "
            f"train_red={train_metrics['mse_reduction_pct']:.3f}% "
            f"val_red={val_metrics['mse_reduction_pct']:.3f}%",
            flush=True,
        )
    residual.load_state_dict(best_state)
    payload = {
        "config_path": str(resolve(spec["config_path"])),
        "checkpoint_path": str(resolve(spec["checkpoint_path"])),
        "channel_names": channel_names,
        "cluster_id": [int(v) for v in cluster_id_c.detach().cpu().tolist()],
        "history": history,
        "best_by_val": {
            "train": evaluate(backbone, residual, train_loader, cluster_id_c, device),
            "val": evaluate(backbone, residual, val_loader, cluster_id_c, device),
            "test": evaluate(backbone, residual, test_loader, cluster_id_c, device),
        },
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct residual MLP control for frozen ETT backbones.")
    parser.add_argument("--cells", default="ETTh1_H96,ETTm2_H96")
    parser.add_argument("--out-json", default="outputs/anchorless_moe_diagnostic/direct_residual_control.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    args = parser.parse_args()

    selected = [item.strip() for item in str(args.cells).split(",") if item.strip()]
    result: dict[str, Any] = {
        "meta": {
            "purpose": "Train a direct residual MLP on frozen backbone outputs to test residual predictability without penalty routing.",
            "epochs": int(args.epochs),
            "hidden_dim": int(args.hidden_dim),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
        },
        "cells": {},
    }
    for cell in selected:
        if cell not in CELL_SPECS:
            raise ValueError(f"Unknown cell {cell}; available={sorted(CELL_SPECS)}")
        print(f"=== direct residual control {cell} ===", flush=True)
        result["cells"][cell] = run_cell(cell, CELL_SPECS[cell], args)
    out_path = resolve(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"direct_residual_control_json={out_path}")


if __name__ == "__main__":
    main()
