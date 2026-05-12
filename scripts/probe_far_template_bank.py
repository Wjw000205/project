from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.reader import read_csv_time_series
from src.data.windows import global_zscore, make_label_range_windows, make_strict_windows, WindowTensorDataset
from src.models.cluster_predictor import build_cluster_predictor
from src.utils.cluster_memory import load_cluster_checkpoint, scatter_mean_bcl_to_bkl
from src.utils.clustering import cluster_channels_by_corr
from src.utils.far_template_bank import (
    ClusterFarTemplateBank,
    build_far_features,
    lowpass_coeff,
    make_dct_basis,
    make_far_mask,
    masked_template_prediction,
)
from src.utils.knn_shape import KNNShapeConfig, ShapeKNNHybrid
from src.utils.pearson import pearson_corr_matrix
from src.utils.seed import set_seed


def _parse_int_list(text: str) -> List[int]:
    return [int(v.strip()) for v in str(text).split(",") if v.strip()]


def _parse_float_list(text: str) -> List[float]:
    return [float(v.strip()) for v in str(text).split(",") if v.strip()]


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_context(cfg: dict, device: torch.device):
    data_tc, channel_names = read_csv_time_series(cfg["data"]["csv_path"], date_col=int(cfg["data"]["date_col"]))
    max_rows = int(cfg.get("data", {}).get("max_rows", 0) or 0)
    if max_rows > 0:
        data_tc = data_tc[:max_rows]
    data_tc = data_tc.to(device)

    t_total, c_count = data_tc.shape
    tr = float(cfg["data"]["train_ratio"])
    vr = float(cfg["data"]["val_ratio"])
    t_train = int(t_total * tr)
    t_val = int(t_total * (tr + vr))

    norm_cfg = cfg.get("normalize", {})
    if bool(norm_cfg.get("global_zscore", False)):
        if bool(norm_cfg.get("train_only", False)):
            train_seg = data_tc[:t_train]
            mean_c = train_seg.mean(dim=0, keepdim=True)
            std_c = train_seg.std(dim=0, keepdim=True).clamp_min(1.0e-6)
            data_tc = (data_tc - mean_c) / std_c
        else:
            data_tc, _, _ = global_zscore(data_tc)

    cluster_cfg = cfg["cluster"]
    corr_cc = pearson_corr_matrix(data_tc)
    rs = cluster_cfg.get("random_state", 0)
    cluster_id_c, clusters = cluster_channels_by_corr(
        corr_cc=corr_cc,
        data_tc=data_tc,
        n_clusters=cluster_cfg.get("n_clusters", None),
        distance_threshold=cluster_cfg.get("distance_threshold", None),
        linkage=cluster_cfg.get("linkage", "average"),
        method=cluster_cfg.get("method", "agglomerative"),
        kmeans_n_init=int(cluster_cfg.get("kmeans_n_init", 10)),
        kmeans_max_iter=int(cluster_cfg.get("kmeans_max_iter", 300)),
        spectral_affinity=cluster_cfg.get("spectral_affinity", "corr"),
        rbf_gamma=float(cluster_cfg.get("rbf_gamma", 1.0)),
        dbscan_eps=cluster_cfg.get("dbscan_eps", None),
        dbscan_min_samples=int(cluster_cfg.get("dbscan_min_samples", 5)),
        random_state=None if rs is None else int(rs),
        min_cluster_size=int(cluster_cfg["min_cluster_size"]),
        merge_small_clusters=bool(cluster_cfg["merge_small_clusters"]),
        no_merge_if_channels_lt=int(cluster_cfg["no_merge_if_channels_lt"]),
    )
    cluster_id_c = cluster_id_c.to(device)

    input_len = int(cfg["window"]["input_len"])
    pred_len = int(cfg["window"]["pred_len"])
    data_cpu = data_tc.detach().cpu()
    xtr, ytr = make_strict_windows(data_cpu, input_len, pred_len, 0, t_train)
    past_context = bool(cfg.get("window", {}).get("past_context", False))
    if past_context:
        xva, yva, val_eval_start = make_label_range_windows(data_cpu, input_len, pred_len, t_train, t_val)
        xte, yte, test_eval_start = make_label_range_windows(data_cpu, input_len, pred_len, t_val, t_total)
    else:
        xva, yva = make_strict_windows(data_cpu, input_len, pred_len, t_train, t_val)
        xte, yte = make_strict_windows(data_cpu, input_len, pred_len, t_val, t_total)
        val_eval_start = t_train
        test_eval_start = t_val

    x_pre, y_pre = make_strict_windows(data_cpu, input_len, pred_len, 0, t_val)
    return {
        "data_tc": data_tc,
        "data_cpu": data_cpu,
        "channel_names": channel_names,
        "cluster_id_c": cluster_id_c,
        "clusters": clusters,
        "xtr": xtr,
        "ytr": ytr,
        "xva": xva,
        "yva": yva,
        "xte": xte,
        "yte": yte,
        "xpre": x_pre,
        "ypre": y_pre,
        "val_eval_start": val_eval_start,
        "test_eval_start": test_eval_start,
        "t_train": t_train,
        "t_val": t_val,
        "T": t_total,
        "C": c_count,
        "L": input_len,
        "H": pred_len,
    }


@torch.no_grad()
def build_model(cfg: dict, ckpt_path: str, cluster_id_c: torch.Tensor, num_channels: int, device: torch.device):
    ckpt = load_cluster_checkpoint(ckpt_path, device=device)
    meta = ckpt.get("meta", {})
    model_cfg = dict(meta.get("model_cfg", cfg["model"]))
    model = build_cluster_predictor(
        num_clusters=int(meta.get("K", int(cluster_id_c.max().item() + 1))),
        input_len=int(meta.get("input_len", cfg["window"]["input_len"])),
        pred_len=int(meta.get("pred_len", cfg["window"]["pred_len"])),
        model_cfg=model_cfg,
        num_channels=num_channels,
        cluster_id_c=cluster_id_c.detach().cpu(),
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def collect_bank_tensors(
    model,
    x_ncl: torch.Tensor,
    y_nch: torch.Tensor,
    model_cluster_id_c: torch.Tensor,
    template_id_c: torch.Tensor,
    template_count: int,
    basis_rh: torch.Tensor,
    batch_size: int,
    device: torch.device,
    feature_cfg: dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(WindowTensorDataset(x_ncl, y_nch), batch_size=batch_size, shuffle=False)
    feat_parts = []
    coeff_parts = []
    basis = basis_rh.to(device)
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        base = model(x, model_cluster_id_c)
        x_k = scatter_mean_bcl_to_bkl(x, template_id_c, template_count)
        y_k = scatter_mean_bcl_to_bkl(y, template_id_c, template_count)
        base_k = scatter_mean_bcl_to_bkl(base, template_id_c, template_count)
        bsz = int(x.shape[0])
        feat_flat = build_far_features(
            x_k.reshape(bsz * template_count, -1),
            base_k.reshape(bsz * template_count, -1),
            shape_bins=int(feature_cfg["shape_bins"]),
            diff_bins=int(feature_cfg["diff_bins"]),
            pred_shape_bins=int(feature_cfg["pred_shape_bins"]),
            pred_diff_bins=int(feature_cfg["pred_diff_bins"]),
        )
        feat_parts.append(feat_flat.reshape(bsz, template_count, -1).detach().cpu())
        coeff_parts.append(lowpass_coeff(y_k - base_k, basis).detach().cpu())
    return torch.cat(feat_parts, dim=0), torch.cat(coeff_parts, dim=0)


@torch.no_grad()
def collect_eval_context(
    model,
    x_ncl: torch.Tensor,
    y_nch: torch.Tensor,
    model_cluster_id_c: torch.Tensor,
    template_id_c: torch.Tensor,
    template_count: int,
    batch_size: int,
    device: torch.device,
    feature_cfg: dict,
) -> Dict[str, torch.Tensor]:
    loader = DataLoader(WindowTensorDataset(x_ncl, y_nch), batch_size=batch_size, shuffle=False)
    feat_parts = []
    base_parts = []
    y_parts = []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        base = model(x, model_cluster_id_c)
        x_k = scatter_mean_bcl_to_bkl(x, template_id_c, template_count)
        base_k = scatter_mean_bcl_to_bkl(base, template_id_c, template_count)
        bsz = int(x.shape[0])
        feat_flat = build_far_features(
            x_k.reshape(bsz * template_count, -1),
            base_k.reshape(bsz * template_count, -1),
            shape_bins=int(feature_cfg["shape_bins"]),
            diff_bins=int(feature_cfg["diff_bins"]),
            pred_shape_bins=int(feature_cfg["pred_shape_bins"]),
            pred_diff_bins=int(feature_cfg["pred_diff_bins"]),
        )
        feat_parts.append(feat_flat.reshape(bsz, template_count, -1).detach().cpu())
        base_parts.append(base.detach().cpu())
        y_parts.append(y.detach().cpu())
    return {
        "features": torch.cat(feat_parts, dim=0),
        "base": torch.cat(base_parts, dim=0),
        "y": torch.cat(y_parts, dim=0),
    }


def metrics(pred: torch.Tensor, y: torch.Tensor) -> Dict[str, object]:
    err = pred - y
    mae_c = err.abs().mean(dim=(0, 2))
    mse_c = err.pow(2).mean(dim=(0, 2))
    return {
        "avg_mae": float(mae_c.mean().item()),
        "avg_mse": float(mse_c.mean().item()),
        "mae_c": [float(v) for v in mae_c.tolist()],
        "mse_c": [float(v) for v in mse_c.tolist()],
    }


def predict_variant(
    bank: ClusterFarTemplateBank,
    context: Dict[str, torch.Tensor],
    cluster_id_c: torch.Tensor,
    far_mask_h: torch.Tensor,
    k: int,
    alpha: float,
    temperature: float,
    weight_mode: str,
) -> Dict[str, object]:
    template_bkh = bank.query_templates(
        context["features"],
        k=int(k),
        temperature=float(temperature),
        weight_mode=str(weight_mode),
    )
    pred = masked_template_prediction(
        context["base"],
        template_bkh,
        cluster_id_c.detach().cpu(),
        far_mask_h,
        alpha=float(alpha),
    )
    return pred


def evaluate_variant(
    bank: ClusterFarTemplateBank,
    context: Dict[str, torch.Tensor],
    cluster_id_c: torch.Tensor,
    far_mask_h: torch.Tensor,
    k: int,
    alpha: float,
    temperature: float,
    weight_mode: str,
) -> Dict[str, object]:
    pred = predict_variant(
        bank,
        context,
        cluster_id_c,
        far_mask_h,
        k=k,
        alpha=alpha,
        temperature=temperature,
        weight_mode=weight_mode,
    )
    out = metrics(pred, context["y"])
    out["k"] = int(k)
    out["alpha"] = float(alpha)
    out["temperature"] = float(temperature)
    out["weight_mode"] = str(weight_mode)
    return out


def apply_channel_guard(
    base_pred: torch.Tensor,
    template_pred: torch.Tensor,
    use_template_c: np.ndarray,
) -> torch.Tensor:
    use = torch.as_tensor(use_template_c, dtype=torch.bool, device=base_pred.device).view(1, -1, 1)
    return torch.where(use, template_pred, base_pred)


def build_channel_guard(
    base_stats: Dict[str, object],
    template_stats: Dict[str, object],
    min_rel_mae_improvement: float,
    max_rel_mse_regression: float,
) -> Dict[str, object]:
    eps = 1.0e-12
    base_mae = np.asarray(base_stats["mae_c"], dtype=np.float64)
    tpl_mae = np.asarray(template_stats["mae_c"], dtype=np.float64)
    base_mse = np.asarray(base_stats["mse_c"], dtype=np.float64)
    tpl_mse = np.asarray(template_stats["mse_c"], dtype=np.float64)
    rel_mae_improvement = (base_mae - tpl_mae) / np.maximum(base_mae, eps)
    rel_mse_regression = (tpl_mse - base_mse) / np.maximum(base_mse, eps)
    use_template = (
        rel_mae_improvement > float(min_rel_mae_improvement)
    ) & (
        rel_mse_regression <= float(max_rel_mse_regression)
    )
    return {
        "use_template_c": [bool(v) for v in use_template.tolist()],
        "rel_mae_improvement_c": [float(v) for v in rel_mae_improvement.tolist()],
        "rel_mse_regression_c": [float(v) for v in rel_mse_regression.tolist()],
        "min_rel_mae_improvement": float(min_rel_mae_improvement),
        "max_rel_mse_regression": float(max_rel_mse_regression),
    }


@torch.no_grad()
def predict_knn_from_base(
    x_ncl: torch.Tensor,
    base_nch: torch.Tensor,
    hybrid: ShapeKNNHybrid,
    cluster_id_c: torch.Tensor,
    batch_size: int,
    eval_start: int,
) -> torch.Tensor:
    pred_parts = []
    cluster_cpu = cluster_id_c.detach().cpu()
    for b0 in range(0, int(x_ncl.shape[0]), int(batch_size)):
        b1 = min(b0 + int(batch_size), int(x_ncl.shape[0]))
        starts = torch.arange(int(eval_start) + b0, int(eval_start) + b1, dtype=torch.long)
        pred = hybrid.hybridize_batch(
            x_ncl[b0:b1],
            base_nch[b0:b1],
            cluster_cpu,
            query_start_abs_b=starts,
        )
        pred_parts.append(pred.detach().cpu())
    return torch.cat(pred_parts, dim=0)


@torch.no_grad()
def evaluate_knn_from_base(
    x_ncl: torch.Tensor,
    y_nch: torch.Tensor,
    base_nch: torch.Tensor,
    hybrid: ShapeKNNHybrid,
    cluster_id_c: torch.Tensor,
    batch_size: int,
    eval_start: int,
) -> Dict[str, object]:
    pred = predict_knn_from_base(
        x_ncl=x_ncl,
        base_nch=base_nch,
        hybrid=hybrid,
        cluster_id_c=cluster_id_c,
        batch_size=batch_size,
        eval_start=eval_start,
    )
    return metrics(pred, y_nch.detach().cpu())


def apply_channel_scale(base_nch: torch.Tensor, hybrid_nch: torch.Tensor, scale_c: torch.Tensor) -> torch.Tensor:
    scale = scale_c.to(device=base_nch.device, dtype=base_nch.dtype).view(1, -1, 1)
    return base_nch + scale * (hybrid_nch - base_nch)


def optimize_channel_scale(
    base_nch: torch.Tensor,
    hybrid_nch: torch.Tensor,
    y_nch: torch.Tensor,
    max_scale: float,
    steps: int,
) -> torch.Tensor:
    base = base_nch.detach().cpu()
    delta = (hybrid_nch - base_nch).detach().cpu()
    y = y_nch.detach().cpu()
    scales = torch.linspace(0.0, float(max_scale), steps=max(2, int(steps)))
    chosen = []
    for c in range(int(base.shape[1])):
        best_scale = scales[0]
        best_mae = float("inf")
        base_ch = base[:, c, :]
        delta_ch = delta[:, c, :]
        y_ch = y[:, c, :]
        for scale in scales:
            mae = (base_ch + scale * delta_ch - y_ch).abs().mean().item()
            if mae < best_mae:
                best_mae = float(mae)
                best_scale = scale
        chosen.append(float(best_scale.item()))
    return torch.tensor(chosen, dtype=base.dtype)


def knn_gate_features(
    x_bcl: torch.Tensor,
    base_bch: torch.Tensor,
    hybrid_bch: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    hist_mean = x_bcl.mean(dim=-1)
    hist_std = x_bcl.std(dim=-1).clamp_min(eps)
    hist_last = x_bcl[..., -1]
    hist_range = (x_bcl.max(dim=-1).values - x_bcl.min(dim=-1).values) / hist_std
    t_l = torch.linspace(-1.0, 1.0, steps=x_bcl.shape[-1], device=x_bcl.device, dtype=x_bcl.dtype).view(1, 1, -1)
    hist_center = x_bcl - hist_mean.unsqueeze(-1)
    hist_slope = (hist_center * t_l).mean(dim=-1) / t_l.pow(2).mean().clamp_min(eps)
    delta_bch = hybrid_bch - base_bch
    delta_abs_mean = delta_bch.abs().mean(dim=-1) / hist_std
    delta_abs_max = delta_bch.abs().amax(dim=-1) / hist_std
    delta_std = delta_bch.std(dim=-1) / hist_std
    delta_bias = delta_bch.mean(dim=-1) / hist_std
    base_std = base_bch.std(dim=-1) / hist_std
    hybrid_std = hybrid_bch.std(dim=-1) / hist_std
    base_shift = (base_bch.mean(dim=-1) - hist_last) / hist_std
    hybrid_shift = (hybrid_bch.mean(dim=-1) - hist_last) / hist_std
    return torch.stack(
        [
            hist_mean,
            hist_std.log(),
            hist_last,
            hist_range,
            hist_slope,
            delta_abs_mean,
            delta_abs_max,
            delta_std,
            delta_bias,
            base_std,
            hybrid_std,
            base_shift,
            hybrid_shift,
        ],
        dim=-1,
    )


class ResidualGate(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        num_channels: int,
        hidden_dim: int,
        dropout: float,
        max_scale: float,
        init_scale: float,
    ):
        super().__init__()
        self.max_scale = max(float(max_scale), 1.0e-6)
        init_scale = max(1.0e-6, min(float(init_scale), self.max_scale - 1.0e-6))
        init_prob = init_scale / self.max_scale
        init_bias = math.log(init_prob / max(1.0 - init_prob, 1.0e-6))
        self.net = nn.Sequential(
            nn.Linear(int(feat_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity(),
            nn.Linear(int(hidden_dim), 1),
        )
        self.channel_bias = nn.Parameter(torch.zeros(int(num_channels)))
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.constant_(last.bias, float(init_bias))

    def forward(self, feat_bcf: torch.Tensor) -> torch.Tensor:
        logits = self.net(feat_bcf).squeeze(-1) + self.channel_bias.view(1, -1)
        return self.max_scale * torch.sigmoid(logits)


def apply_residual_gate(
    gate: ResidualGate,
    x_ncl: torch.Tensor,
    base_nch: torch.Tensor,
    hybrid_nch: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    out_parts = []
    gate.eval()
    with torch.no_grad():
        for b0 in range(0, int(x_ncl.shape[0]), int(batch_size)):
            b1 = min(b0 + int(batch_size), int(x_ncl.shape[0]))
            x = x_ncl[b0:b1].to(device)
            base = base_nch[b0:b1].to(device)
            hybrid = hybrid_nch[b0:b1].to(device)
            feat = knn_gate_features(x, base, hybrid)
            scale = gate(feat).unsqueeze(-1)
            out_parts.append((base + scale * (hybrid - base)).detach().cpu())
    return torch.cat(out_parts, dim=0)


def train_residual_gate(
    x_ncl: torch.Tensor,
    base_nch: torch.Tensor,
    hybrid_nch: torch.Tensor,
    y_nch: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[ResidualGate, Dict[str, object]]:
    feat = knn_gate_features(x_ncl, base_nch, hybrid_nch).detach().cpu()
    base = base_nch.detach().cpu()
    delta = (hybrid_nch - base_nch).detach().cpu()
    y = y_nch.detach().cpu()
    n = int(feat.shape[0])
    c = int(feat.shape[1])
    split = int(max(1, min(n - 1, round(n * float(args.gate_train_fraction))))) if n > 1 else n
    train_idx = torch.arange(0, split, dtype=torch.long)
    hold_idx = torch.arange(split, n, dtype=torch.long)
    if hold_idx.numel() == 0:
        hold_idx = train_idx
    gate = ResidualGate(
        feat_dim=int(feat.shape[-1]),
        num_channels=c,
        hidden_dim=int(args.gate_hidden_dim),
        dropout=float(args.gate_dropout),
        max_scale=float(args.gate_max_scale),
        init_scale=float(args.gate_init_scale),
    ).to(device)
    opt = torch.optim.AdamW(gate.parameters(), lr=float(args.gate_lr), weight_decay=float(args.gate_weight_decay))
    batch_size = max(1, int(args.gate_batch_size))
    scale_reg = float(args.gate_scale_reg)
    init_scale = float(args.gate_init_scale)
    best_state = None
    best_hold = float("inf")
    best_epoch = 0

    def eval_idx(idx: torch.Tensor) -> Tuple[float, float]:
        pred = []
        target = []
        gate.eval()
        with torch.no_grad():
            for b0 in range(0, int(idx.numel()), batch_size):
                batch_idx = idx[b0:b0 + batch_size]
                feat_b = feat.index_select(0, batch_idx).to(device)
                base_b = base.index_select(0, batch_idx).to(device)
                delta_b = delta.index_select(0, batch_idx).to(device)
                y_b = y.index_select(0, batch_idx).to(device)
                scale_b = gate(feat_b)
                pred.append((base_b + scale_b.unsqueeze(-1) * delta_b).detach().cpu())
                target.append(y_b.detach().cpu())
        stat = metrics(torch.cat(pred, dim=0), torch.cat(target, dim=0))
        return float(stat["avg_mae"]), float(stat["avg_mse"])

    for ep in range(1, int(args.gate_epochs) + 1):
        gate.train()
        perm = train_idx[torch.randperm(int(train_idx.numel()))]
        for b0 in range(0, int(perm.numel()), batch_size):
            batch_idx = perm[b0:b0 + batch_size]
            feat_b = feat.index_select(0, batch_idx).to(device)
            base_b = base.index_select(0, batch_idx).to(device)
            delta_b = delta.index_select(0, batch_idx).to(device)
            y_b = y.index_select(0, batch_idx).to(device)
            scale_b = gate(feat_b)
            pred_b = base_b + scale_b.unsqueeze(-1) * delta_b
            loss = (pred_b - y_b).abs().mean()
            if scale_reg > 0.0:
                loss = loss + scale_reg * (scale_b - init_scale).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), float(args.gate_grad_clip))
            opt.step()
        hold_mae, _ = eval_idx(hold_idx)
        if hold_mae < best_hold:
            best_hold = hold_mae
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in gate.state_dict().items()}

    if best_state is not None:
        gate.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    train_mae, train_mse = eval_idx(train_idx)
    hold_mae, hold_mse = eval_idx(hold_idx)
    return gate, {
        "best_epoch": int(best_epoch),
        "train_windows": int(train_idx.numel()),
        "holdout_windows": int(hold_idx.numel()),
        "train_mae": float(train_mae),
        "train_mse": float(train_mse),
        "holdout_mae": float(hold_mae),
        "holdout_mse": float(hold_mse),
    }


def make_knn_cfg(
    cfg: dict,
    args: argparse.Namespace,
    pred_len: int,
    k_override: int | None = None,
    alpha_override: float | None = None,
) -> KNNShapeConfig:
    knn_dict = dict(cfg.get("knn_hybrid", {}) or {})
    knn_dict.pop("sweep", None)
    knn_dict.pop("path", None)
    knn_dict["enable"] = True
    knn_dict["mode"] = "rolling"
    knn_dict["bank_split"] = "history"
    knn_dict["scope"] = str(args.knn_scope)
    knn_dict["feature_mode"] = str(args.knn_feature_mode)
    knn_dict["template_mode"] = "residual"
    knn_k = int(args.knn_k) if k_override is None else int(k_override)
    knn_alpha = float(args.knn_alpha) if alpha_override is None else float(alpha_override)
    if knn_k > 0:
        knn_dict["k"] = int(knn_k)
    if knn_alpha >= 0.0:
        knn_dict["alpha"] = float(knn_alpha)
    if str(args.knn_adaptive_alpha):
        knn_dict["adaptive_alpha"] = str(args.knn_adaptive_alpha)
    if int(args.knn_bank_stride) > 0:
        knn_dict["bank_stride"] = int(args.knn_bank_stride)
    return KNNShapeConfig.from_dict(knn_dict).resolved_for_horizon(int(pred_len))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="outputs/ettm1_720_dlinear_knn_sweep2.yaml")
    ap.add_argument("--checkpoint", default="outputs/ettm1_720_dlinear_knn_sweep2/best_checkpoint.pt")
    ap.add_argument("--out-dir", default="outputs/ettm1_720_far_template_probe")
    ap.add_argument("--k-list", default="8,16,32,64")
    ap.add_argument("--alpha-list", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--rank", type=int, default=24)
    ap.add_argument("--tau", type=float, default=240.0)
    ap.add_argument("--softness", type=float, default=80.0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--weight-mode", default="inverse", choices=["inverse", "softmax"])
    ap.add_argument("--template-scope", default="cluster", choices=["cluster", "channel"])
    ap.add_argument("--far-k", type=int, default=0)
    ap.add_argument("--far-alpha", type=float, default=-1.0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--shape-bins", type=int, default=24)
    ap.add_argument("--diff-bins", type=int, default=12)
    ap.add_argument("--pred-shape-bins", type=int, default=16)
    ap.add_argument("--pred-diff-bins", type=int, default=8)
    ap.add_argument("--channel-guard", action="store_true")
    ap.add_argument("--guard-min-rel-mae-improvement", type=float, default=0.0)
    ap.add_argument("--guard-max-rel-mse-regression", type=float, default=0.03)
    ap.add_argument("--knn-combine", action="store_true")
    ap.add_argument("--knn-k", type=int, default=96)
    ap.add_argument("--knn-alpha", type=float, default=0.8)
    ap.add_argument("--knn-k-list", default="")
    ap.add_argument("--knn-alpha-list", default="")
    ap.add_argument("--knn-scope", default="same_channel", choices=["same_channel", "same_cluster"])
    ap.add_argument("--knn-feature-mode", default="joint", choices=["hist", "joint"])
    ap.add_argument("--knn-adaptive-alpha", default="none")
    ap.add_argument("--knn-bank-stride", type=int, default=2)
    ap.add_argument("--knn-channel-scale", action="store_true")
    ap.add_argument("--knn-scale-max", type=float, default=1.0)
    ap.add_argument("--knn-scale-steps", type=int, default=21)
    ap.add_argument("--knn-nn-gate", action="store_true")
    ap.add_argument("--gate-hidden-dim", type=int, default=16)
    ap.add_argument("--gate-dropout", type=float, default=0.1)
    ap.add_argument("--gate-max-scale", type=float, default=1.0)
    ap.add_argument("--gate-init-scale", type=float, default=0.95)
    ap.add_argument("--gate-train-fraction", type=float, default=0.7)
    ap.add_argument("--gate-epochs", type=int, default=40)
    ap.add_argument("--gate-batch-size", type=int, default=128)
    ap.add_argument("--gate-lr", type=float, default=8.0e-4)
    ap.add_argument("--gate-weight-decay", type=float, default=5.0e-4)
    ap.add_argument("--gate-scale-reg", type=float, default=1.0e-3)
    ap.add_argument("--gate-grad-clip", type=float, default=1.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _load_yaml(args.config)
    set_seed(int(cfg["exp"].get("seed", 2026)), deterministic=bool(cfg["exp"].get("deterministic", False)))
    device = torch.device(cfg["exp"]["device"] if torch.cuda.is_available() else "cpu")
    ctx = prepare_context(cfg, device=device)
    cluster_id_c = ctx["cluster_id_c"]
    k_count = int(cluster_id_c.max().item() + 1)
    template_scope = str(args.template_scope).lower()
    if template_scope == "channel":
        template_id_c = torch.arange(int(ctx["C"]), device=device, dtype=torch.long)
        template_count = int(ctx["C"])
    else:
        template_id_c = cluster_id_c
        template_count = int(k_count)
    model = build_model(cfg, args.checkpoint, cluster_id_c, int(ctx["C"]), device=device)

    feature_cfg = {
        "shape_bins": int(args.shape_bins),
        "diff_bins": int(args.diff_bins),
        "pred_shape_bins": int(args.pred_shape_bins),
        "pred_diff_bins": int(args.pred_diff_bins),
    }
    basis_rh = make_dct_basis(rank=int(args.rank), pred_len=int(ctx["H"]), device=device)
    far_mask_h = make_far_mask(int(ctx["H"]), tau=float(args.tau), softness=float(args.softness))

    print("Building train far-template bank...")
    train_feat, train_coeff = collect_bank_tensors(
        model,
        ctx["xtr"],
        ctx["ytr"],
        cluster_id_c,
        template_id_c,
        template_count,
        basis_rh,
        batch_size=int(args.batch_size),
        device=device,
        feature_cfg=feature_cfg,
    )
    train_bank = ClusterFarTemplateBank.fit(train_feat, train_coeff).with_basis(basis_rh.cpu())

    print("Building pre-test far-template bank...")
    pre_feat, pre_coeff = collect_bank_tensors(
        model,
        ctx["xpre"],
        ctx["ypre"],
        cluster_id_c,
        template_id_c,
        template_count,
        basis_rh,
        batch_size=int(args.batch_size),
        device=device,
        feature_cfg=feature_cfg,
    )
    pre_bank = ClusterFarTemplateBank.fit(pre_feat, pre_coeff).with_basis(basis_rh.cpu())

    print("Collecting validation/test contexts...")
    val_ctx = collect_eval_context(
        model,
        ctx["xva"],
        ctx["yva"],
        cluster_id_c,
        template_id_c,
        template_count,
        batch_size=int(args.batch_size),
        device=device,
        feature_cfg=feature_cfg,
    )
    test_ctx = collect_eval_context(
        model,
        ctx["xte"],
        ctx["yte"],
        cluster_id_c,
        template_id_c,
        template_count,
        batch_size=int(args.batch_size),
        device=device,
        feature_cfg=feature_cfg,
    )

    base_val = metrics(val_ctx["base"], val_ctx["y"])
    base_test = metrics(test_ctx["base"], test_ctx["y"])
    print(f"Base val MAE={base_val['avg_mae']:.6f}, MSE={base_val['avg_mse']:.6f}")
    print(f"Base test MAE={base_test['avg_mae']:.6f}, MSE={base_test['avg_mse']:.6f}")

    rows = []
    best_row = None
    for kk in _parse_int_list(args.k_list):
        for alpha in _parse_float_list(args.alpha_list):
            val = evaluate_variant(
                train_bank,
                val_ctx,
                template_id_c,
                far_mask_h,
                k=kk,
                alpha=alpha,
                temperature=float(args.temperature),
                weight_mode=str(args.weight_mode),
            )
            row = {
                "k": int(kk),
                "alpha": float(alpha),
                "val_mae": val["avg_mae"],
                "val_mse": val["avg_mse"],
            }
            rows.append(row)
            if best_row is None or float(row["val_mae"]) < float(best_row["val_mae"]):
                best_row = dict(row)
            print(
                f"val k={kk} alpha={alpha:.3f}: "
                f"MAE={val['avg_mae']:.6f}, MSE={val['avg_mse']:.6f}"
            )

    assert best_row is not None
    far_selection_source = "val_mae"
    if int(args.far_k) > 0 and float(args.far_alpha) >= 0.0:
        override_val = evaluate_variant(
            train_bank,
            val_ctx,
            template_id_c,
            far_mask_h,
            k=int(args.far_k),
            alpha=float(args.far_alpha),
            temperature=float(args.temperature),
            weight_mode=str(args.weight_mode),
        )
        best_row = {
            "k": int(args.far_k),
            "alpha": float(args.far_alpha),
            "val_mae": float(override_val["avg_mae"]),
            "val_mse": float(override_val["avg_mse"]),
        }
        far_selection_source = "override"
        print(
            "Override far-template selection: "
            f"k={best_row['k']} alpha={best_row['alpha']:.3f} | "
            f"val MAE={best_row['val_mae']:.6f}, MSE={best_row['val_mse']:.6f}"
        )
    val_pred = predict_variant(
        train_bank,
        val_ctx,
        template_id_c,
        far_mask_h,
        k=int(best_row["k"]),
        alpha=float(best_row["alpha"]),
        temperature=float(args.temperature),
        weight_mode=str(args.weight_mode),
    )
    val_selected = metrics(val_pred, val_ctx["y"])
    test_pred = predict_variant(
        pre_bank,
        test_ctx,
        template_id_c,
        far_mask_h,
        k=int(best_row["k"]),
        alpha=float(best_row["alpha"]),
        temperature=float(args.temperature),
        weight_mode=str(args.weight_mode),
    )
    test = metrics(test_pred, test_ctx["y"])
    print(
        "Selected by val: "
        f"k={best_row['k']} alpha={best_row['alpha']:.3f} | "
        f"test MAE={test['avg_mae']:.6f}, MSE={test['avg_mse']:.6f}"
    )

    guarded = None
    if bool(args.channel_guard):
        guard = build_channel_guard(
            base_stats=base_val,
            template_stats=val_selected,
            min_rel_mae_improvement=float(args.guard_min_rel_mae_improvement),
            max_rel_mse_regression=float(args.guard_max_rel_mse_regression),
        )
        use_template = np.asarray(guard["use_template_c"], dtype=bool)
        val_guard_pred = apply_channel_guard(val_ctx["base"], val_pred, use_template)
        test_guard_pred = apply_channel_guard(test_ctx["base"], test_pred, use_template)
        val_guard = metrics(val_guard_pred, val_ctx["y"])
        test_guard = metrics(test_guard_pred, test_ctx["y"])
        enabled_names = [
            str(name) for name, use in zip(ctx["channel_names"], use_template.tolist()) if bool(use)
        ]
        guarded = {
            **guard,
            "enabled_channel_names": enabled_names,
            "val_mae": float(val_guard["avg_mae"]),
            "val_mse": float(val_guard["avg_mse"]),
            "test_mae": float(test_guard["avg_mae"]),
            "test_mse": float(test_guard["avg_mse"]),
            "test_mae_c": test_guard["mae_c"],
            "test_mse_c": test_guard["mse_c"],
        }
        print(
            "Channel-guarded: "
            f"channels={enabled_names} | "
            f"test MAE={test_guard['avg_mae']:.6f}, MSE={test_guard['avg_mse']:.6f}"
        )

    knn_combined = None
    if bool(args.knn_combine):
        first_knn_cfg = make_knn_cfg(cfg, args, pred_len=int(ctx["H"]))
        print(
            "Building KNN on far-template base: "
            f"adaptive={first_knn_cfg.adaptive_alpha}, stride={first_knn_cfg.bank_stride}"
        )
        data_cpu = ctx["data_cpu"]
        x_all, y_all = make_strict_windows(data_cpu, int(ctx["L"]), int(ctx["H"]), 0, int(ctx["T"]))

        print("Preparing far-template base for validation KNN bank...")
        val_bank_ctx = collect_eval_context(
            model,
            ctx["xpre"],
            ctx["ypre"],
            cluster_id_c,
            template_id_c,
            template_count,
            batch_size=int(args.batch_size),
            device=device,
            feature_cfg=feature_cfg,
        )
        val_bank_base = predict_variant(
            train_bank,
            val_bank_ctx,
            template_id_c,
            far_mask_h,
            k=int(best_row["k"]),
            alpha=float(best_row["alpha"]),
            temperature=float(args.temperature),
            weight_mode=str(args.weight_mode),
        )
        starts_pre = torch.arange(0, int(ctx["xpre"].shape[0]), dtype=torch.long)

        print("Preparing far-template base for test KNN bank...")
        test_bank_ctx = collect_eval_context(
            model,
            x_all,
            y_all,
            cluster_id_c,
            template_id_c,
            template_count,
            batch_size=int(args.batch_size),
            device=device,
            feature_cfg=feature_cfg,
        )
        test_bank_base = predict_variant(
            pre_bank,
            test_bank_ctx,
            template_id_c,
            far_mask_h,
            k=int(best_row["k"]),
            alpha=float(best_row["alpha"]),
            temperature=float(args.temperature),
            weight_mode=str(args.weight_mode),
        )
        starts_all = torch.arange(0, int(x_all.shape[0]), dtype=torch.long)
        knn_k_values = _parse_int_list(args.knn_k_list) if str(args.knn_k_list).strip() else [int(args.knn_k)]
        knn_alpha_values = (
            _parse_float_list(args.knn_alpha_list)
            if str(args.knn_alpha_list).strip()
            else [float(args.knn_alpha)]
        )
        knn_rows = []
        best_knn = None
        for knn_k in knn_k_values:
            for knn_alpha in knn_alpha_values:
                cand_knn_cfg = make_knn_cfg(
                    cfg,
                    args,
                    pred_len=int(ctx["H"]),
                    k_override=int(knn_k),
                    alpha_override=float(knn_alpha),
                )
                knn_val = ShapeKNNHybrid.fit(
                    x_bank_ncl=ctx["xpre"],
                    y_bank_nch=ctx["ypre"],
                    cluster_id_c=cluster_id_c.detach().cpu(),
                    cfg=cand_knn_cfg,
                    start_offsets_n=starts_pre,
                    base_bank_pred_nch=val_bank_base,
                )
                knn_test = ShapeKNNHybrid.fit(
                    x_bank_ncl=x_all,
                    y_bank_nch=y_all,
                    cluster_id_c=cluster_id_c.detach().cpu(),
                    cfg=cand_knn_cfg,
                    start_offsets_n=starts_all,
                    base_bank_pred_nch=test_bank_base,
                )

                knn_val.reset_confidence_stats()
                val_knn_pred = predict_knn_from_base(
                    ctx["xva"],
                    val_pred,
                    knn_val,
                    cluster_id_c,
                    batch_size=int(args.batch_size),
                    eval_start=int(ctx["val_eval_start"]),
                )
                val_knn = metrics(val_knn_pred, ctx["yva"])
                val_knn_conf = knn_val.get_confidence_stats()
                knn_test.reset_confidence_stats()
                test_knn_pred = predict_knn_from_base(
                    ctx["xte"],
                    test_pred,
                    knn_test,
                    cluster_id_c,
                    batch_size=int(args.batch_size),
                    eval_start=int(ctx["test_eval_start"]),
                )
                test_knn = metrics(test_knn_pred, ctx["yte"])
                test_knn_conf = knn_test.get_confidence_stats()
                scaled_payload = None
                gated_payload = None
                selection_val_mae = float(val_knn["avg_mae"])
                selection_val_mse = float(val_knn["avg_mse"])
                if bool(args.knn_channel_scale):
                    scale_c = optimize_channel_scale(
                        val_pred,
                        val_knn_pred,
                        ctx["yva"],
                        max_scale=float(args.knn_scale_max),
                        steps=int(args.knn_scale_steps),
                    )
                    val_scaled_pred = apply_channel_scale(val_pred, val_knn_pred, scale_c)
                    test_scaled_pred = apply_channel_scale(test_pred, test_knn_pred, scale_c)
                    val_scaled = metrics(val_scaled_pred, ctx["yva"])
                    test_scaled = metrics(test_scaled_pred, ctx["yte"])
                    scaled_payload = {
                        "scale_c": [float(v) for v in scale_c.tolist()],
                        "val_mae": float(val_scaled["avg_mae"]),
                        "val_mse": float(val_scaled["avg_mse"]),
                        "test_mae": float(test_scaled["avg_mae"]),
                        "test_mse": float(test_scaled["avg_mse"]),
                        "test_mae_c": test_scaled["mae_c"],
                        "test_mse_c": test_scaled["mse_c"],
                    }
                    selection_val_mae = float(val_scaled["avg_mae"])
                    selection_val_mse = float(val_scaled["avg_mse"])
                if bool(args.knn_nn_gate):
                    gate_model, gate_summary = train_residual_gate(
                        ctx["xva"],
                        val_pred,
                        val_knn_pred,
                        ctx["yva"],
                        args=args,
                        device=device,
                    )
                    val_gate_pred = apply_residual_gate(
                        gate_model,
                        ctx["xva"],
                        val_pred,
                        val_knn_pred,
                        batch_size=int(args.batch_size),
                        device=device,
                    )
                    test_gate_pred = apply_residual_gate(
                        gate_model,
                        ctx["xte"],
                        test_pred,
                        test_knn_pred,
                        batch_size=int(args.batch_size),
                        device=device,
                    )
                    val_gate = metrics(val_gate_pred, ctx["yva"])
                    test_gate = metrics(test_gate_pred, ctx["yte"])
                    gated_payload = {
                        **gate_summary,
                        "val_mae": float(val_gate["avg_mae"]),
                        "val_mse": float(val_gate["avg_mse"]),
                        "test_mae": float(test_gate["avg_mae"]),
                        "test_mse": float(test_gate["avg_mse"]),
                        "test_mae_c": test_gate["mae_c"],
                        "test_mse_c": test_gate["mse_c"],
                    }
                    selection_val_mae = float(gate_summary["holdout_mae"])
                    selection_val_mse = float(gate_summary["holdout_mse"])
                row = {
                    "k": int(knn_k),
                    "alpha": float(knn_alpha),
                    "val_mae": float(val_knn["avg_mae"]),
                    "val_mse": float(val_knn["avg_mse"]),
                    "test_mae": float(test_knn["avg_mae"]),
                    "test_mse": float(test_knn["avg_mse"]),
                    "config": knn_test.describe(),
                    "test_mae_c": test_knn["mae_c"],
                    "test_mse_c": test_knn["mse_c"],
                    "val_confidence": val_knn_conf,
                    "test_confidence": test_knn_conf,
                    "channel_scaled": scaled_payload,
                    "nn_gated": gated_payload,
                    "selection_val_mae": selection_val_mae,
                    "selection_val_mse": selection_val_mse,
                }
                knn_rows.append(row)
                if best_knn is None or row["selection_val_mae"] < best_knn["selection_val_mae"]:
                    best_knn = row
                msg = (
                    "Far-template + KNN "
                    f"k={knn_k} alpha={knn_alpha:.3f}: "
                    f"val MAE={val_knn['avg_mae']:.6f}, MSE={val_knn['avg_mse']:.6f} | "
                    f"test MAE={test_knn['avg_mae']:.6f}, MSE={test_knn['avg_mse']:.6f}"
                )
                if scaled_payload is not None:
                    msg += (
                        " | scaled "
                        f"val MAE={scaled_payload['val_mae']:.6f}, "
                        f"test MAE={scaled_payload['test_mae']:.6f}"
                    )
                if gated_payload is not None:
                    msg += (
                        " | gated "
                        f"hold MAE={gated_payload['holdout_mae']:.6f}, "
                        f"test MAE={gated_payload['test_mae']:.6f}"
                    )
                print(msg)
        assert best_knn is not None
        knn_combined = {
            "selected_by": "val_mae",
            "selected": best_knn,
            "sweep": knn_rows,
        }

    csv_path = out_dir / "far_template_sweep.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "alpha", "val_mae", "val_mse"])
        writer.writeheader()
        writer.writerows(rows)

    knn_csv_path = None
    if knn_combined is not None:
        knn_csv_path = out_dir / "far_template_knn_sweep.csv"
        with open(knn_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "k",
                "alpha",
                "val_mae",
                "val_mse",
                "test_mae",
                "test_mse",
                "scaled_val_mae",
                "scaled_val_mse",
                "scaled_test_mae",
                "scaled_test_mse",
                "gated_holdout_mae",
                "gated_val_mae",
                "gated_val_mse",
                "gated_test_mae",
                "gated_test_mse",
            ])
            writer.writeheader()
            for row in knn_combined["sweep"]:
                scaled = row.get("channel_scaled") or {}
                gated = row.get("nn_gated") or {}
                writer.writerow({
                    "k": row["k"],
                    "alpha": row["alpha"],
                    "val_mae": row["val_mae"],
                    "val_mse": row["val_mse"],
                    "test_mae": row["test_mae"],
                    "test_mse": row["test_mse"],
                    "scaled_val_mae": scaled.get("val_mae", ""),
                    "scaled_val_mse": scaled.get("val_mse", ""),
                    "scaled_test_mae": scaled.get("test_mae", ""),
                    "scaled_test_mse": scaled.get("test_mse", ""),
                    "gated_holdout_mae": gated.get("holdout_mae", ""),
                    "gated_val_mae": gated.get("val_mae", ""),
                    "gated_val_mse": gated.get("val_mse", ""),
                    "gated_test_mae": gated.get("test_mae", ""),
                    "gated_test_mse": gated.get("test_mse", ""),
                })

    summary = {
        "config_path": str(args.config),
        "checkpoint": str(args.checkpoint),
        "out_dir": str(out_dir),
        "channel_names": list(ctx["channel_names"]),
        "clusters": {
            str(k): [ctx["channel_names"][i] for i in members]
            for k, members in ctx["clusters"].items()
        },
        "bank": {
            "rank": int(args.rank),
            "tau": float(args.tau),
            "softness": float(args.softness),
            "weight_mode": str(args.weight_mode),
            "temperature": float(args.temperature),
            "train_templates": int(train_feat.shape[0]),
            "pretest_templates": int(pre_feat.shape[0]),
            "template_scope": template_scope,
            "num_template_groups": int(template_count),
            "num_clusters": int(k_count),
            **feature_cfg,
        },
        "base_val": base_val,
        "base_test": base_test,
        "selected": {
            "k": int(best_row["k"]),
            "alpha": float(best_row["alpha"]),
            "source": far_selection_source,
            "val_mae": float(best_row["val_mae"]),
            "val_mse": float(best_row["val_mse"]),
            "test_mae": float(test["avg_mae"]),
            "test_mse": float(test["avg_mse"]),
            "test_mae_c": test["mae_c"],
            "test_mse_c": test["mse_c"],
        },
        "channel_guarded": guarded,
        "knn_combined": knn_combined,
        "sweep": rows,
    }
    summary_path = out_dir / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved sweep to: {csv_path}")
    if knn_csv_path is not None:
        print(f"Saved KNN sweep to: {knn_csv_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
