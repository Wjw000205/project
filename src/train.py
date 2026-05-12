from __future__ import annotations

import os
import json
import argparse
import time
import math
import sys
import builtins
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
try:
    from torch.func import functional_call as _torch_functional_call

    def _module_call(module: nn.Module, params: Optional[Dict[str, torch.Tensor]], *args, **kwargs):
        if params is None:
            return module(*args, **kwargs)
        return _torch_functional_call(module, params, args=args, kwargs=kwargs)
except Exception:
    from torch.nn.utils.stateless import functional_call as _torch_stateless_functional_call

    def _module_call(module: nn.Module, params: Optional[Dict[str, torch.Tensor]], *args, **kwargs):
        if params is None:
            return module(*args, **kwargs)
        return _torch_stateless_functional_call(module, params, args, kwargs)
try:
    import psutil  # type: ignore
except Exception:
    psutil = None

from .utils.yaml_io import load_yaml
from .utils.seed import set_seed
from .data.reader import read_csv_time_series
from .data.windows import global_zscore, make_label_range_windows, make_strict_windows, WindowTensorDataset
from .utils.pearson import pearson_corr_matrix
from .utils.clustering import cluster_channels_by_corr
from .models.cluster_predictor import build_cluster_predictor
from .models.dynamic_lambda import ClusterwiseDynamicLambda
from .models.learnable_lambda import ClusterwiseLearnableLambda
from .models.moe_gate import ClusterwiseMoEGate, scatter_mean_bc_to_bk, scatter_mean_bcf_to_bkf
from .models.penalties import build_penalty_bank, normalize_penalties
from .models.residual_moe import ClusterwisePredResidualMoE
from .utils.metrics import accumulate_channel_errors, mse_mae_from_sums
from .utils.plotting import save_channel_plots, save_cluster_metric_curves
from .utils.cluster_portrait import save_cluster_portraits
from .utils.cluster_memory import (
    OnlineClusterMemory,
    compute_cluster_prototypes,
    scatter_mean_bcl_to_bkl,
    save_cluster_memory,
    save_cluster_checkpoint,
    load_cluster_memory,
    load_cluster_checkpoint,
)
from .utils.knn_shape import KNNShapeConfig, ShapeKNNHybrid, predict_bank_outputs, save_shape_knn_bank
from .utils.console_progress import PurpleProgressBar

def _get_rss_mb() -> float:
    if psutil is None:
        return -1.0
    try:
        proc = psutil.Process(os.getpid())
        return float(proc.memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return -1.0

def _dir_size_mb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                continue
    return float(total) / (1024.0 * 1024.0)

def print_clusters(clusters: Dict[int, List[int]], channel_names: List[str]):
    for k in sorted(clusters.keys()):
        chs = [channel_names[i] for i in clusters[k]]
        print(f"Cluster {k}: [" + ", ".join(chs) + "]")

GATE_FEATURE_NAMES = [
    "mean",
    "std",
    "last_centered",
    "trend_slope",
    "range_over_std",
    "mad1",
    "mad2",
    "jump_rate",
    "acf1",
    "curvature_ratio",
]


def get_gate_feature_dim() -> int:
    return len(GATE_FEATURE_NAMES)


def reduce_cluster_metric(x: torch.Tensor, weight_k: torch.Tensor) -> torch.Tensor:
    """
    Weighted reduction over cluster dimension.
    x: [..., K]
    weight_k: [K], sum to 1
    returns: [...] with cluster dimension reduced
    """
    w = weight_k.to(device=x.device, dtype=x.dtype)
    view_shape = [1] * x.dim()
    view_shape[-1] = w.shape[0]
    return (x * w.view(*view_shape)).sum(dim=-1)


def extract_gate_features(x_bcl: torch.Tensor) -> torch.Tensor:
    """
    Extract lightweight but shape-aware per-channel descriptors for gate routing.
    Returns x_bcf: [B, C, F]
    """
    B, C, L = x_bcl.shape
    mean = x_bcl.mean(dim=-1)
    std = x_bcl.std(dim=-1)
    eps = 1.0e-6

    last_centered = x_bcl[..., -1] - mean
    x_min = x_bcl.min(dim=-1).values
    x_max = x_bcl.max(dim=-1).values
    range_over_std = (x_max - x_min) / std.clamp_min(eps)

    t = torch.linspace(-1.0, 1.0, steps=L, device=x_bcl.device, dtype=x_bcl.dtype).view(1, 1, L)
    slope = ((x_bcl - mean.unsqueeze(-1)) * t).mean(dim=-1) / t.pow(2).mean(dim=-1).clamp_min(eps)

    if L >= 2:
        d1 = x_bcl[..., 1:] - x_bcl[..., :-1]
        mad1 = d1.abs().mean(dim=-1)
        jump_thr = 1.5 * std.unsqueeze(-1).clamp_min(eps)
        jump_rate = (d1.abs() > jump_thr).to(x_bcl.dtype).mean(dim=-1)

        # ACF lag-1: 用全局均值去中心化，分子=E[(x_t-μ)(x_{t-1}-μ)]，分母=Var(x)
        x_centered = x_bcl - mean.unsqueeze(-1)
        acf1_num = (x_centered[..., 1:] * x_centered[..., :-1]).mean(dim=-1)
        acf1_den = x_centered.pow(2).mean(dim=-1)
        acf1 = acf1_num / acf1_den.clamp_min(eps)
    else:
        d1 = None
        mad1 = torch.zeros_like(mean)
        jump_rate = torch.zeros_like(mean)
        acf1 = torch.zeros_like(mean)

    if L >= 3:
        d2 = x_bcl[..., 2:] - 2 * x_bcl[..., 1:-1] + x_bcl[..., :-2]
        mad2 = d2.abs().mean(dim=-1)
        curvature_ratio = d2.pow(2).mean(dim=-1) / d1.pow(2).mean(dim=-1).clamp_min(eps)
    else:
        mad2 = torch.zeros_like(mean)
        curvature_ratio = torch.zeros_like(mean)

    feat = torch.stack(
        [
            mean,
            std,
            last_centered,
            slope,
            range_over_std,
            mad1,
            mad2,
            jump_rate,
            acf1,
            curvature_ratio,
        ],
        dim=-1,
    )  # [B,C,F]
    return feat


def build_gate_prior_from_penalty_portrait(
    penalty_kp: Optional[torch.Tensor],
    penalty_scale: Optional[torch.Tensor],
    temperature: float = 1.0,
    smoothing: float = 0.0,
    use_normalized_penalty: bool = True,
) -> Optional[torch.Tensor]:
    if penalty_kp is None or penalty_kp.numel() == 0:
        return None
    prior = penalty_kp.detach().clone().to(dtype=torch.float32)
    if use_normalized_penalty and penalty_scale is not None and penalty_scale.numel() == prior.shape[-1]:
        scale = penalty_scale.detach().to(device=prior.device, dtype=prior.dtype).view(1, -1).clamp_min(1.0e-6)
        prior = prior / scale
    prior = prior.clamp_min(1.0e-8)
    temp = max(float(temperature), 1.0e-6)
    prior = torch.softmax(prior.log() / temp, dim=-1)
    smooth = float(max(0.0, min(smoothing, 0.99)))
    if smooth > 0.0:
        prior = (1.0 - smooth) * prior + smooth * (1.0 / float(prior.shape[-1]))
    prior = prior / prior.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    return prior

@torch.no_grad()
def compute_cluster_penalty_portrait(
    loader: DataLoader,
    penalty_names: List[str],
    penalty_fns: Dict[str, callable],
    cluster_id_c: torch.Tensor,
    K: int,
    pred_len: int,
    device: torch.device,
) -> torch.Tensor:
    if len(penalty_names) == 0 or len(loader) == 0:
        return torch.zeros((K, len(penalty_names)), device=device)
    sum_kp = torch.zeros(K, len(penalty_names), device=device)
    cnt = 0
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        # Use a last-value baseline to estimate penalty magnitudes per cluster.
        last = x[..., -1:]
        yhat = last.expand(-1, -1, pred_len)
        pen_bcp = []
        for name in penalty_names:
            pen_bc = penalty_fns[name](yhat, y)
            pen_bcp.append(pen_bc)
        pen_bcp = torch.stack(pen_bcp, dim=-1)
        pen_bkp = scatter_mean_bcf_to_bkf(pen_bcp, cluster_id_c, K)
        sum_kp += pen_bkp.sum(dim=0)
        cnt += pen_bkp.shape[0]
    if cnt == 0:
        return torch.zeros((K, len(penalty_names)), device=device)
    return sum_kp / float(cnt)


def _rowwise_corr(a_kt: torch.Tensor, b_mt: torch.Tensor, align: str = "head", eps: float = 1.0e-6) -> torch.Tensor:
    use_t = min(int(a_kt.shape[1]), int(b_mt.shape[1]))
    if use_t <= 1:
        return torch.zeros((a_kt.shape[0], b_mt.shape[0]), device=a_kt.device, dtype=a_kt.dtype)
    align = str(align).lower()
    if align == "tail":
        a = a_kt[:, -use_t:]
        b = b_mt[:, -use_t:]
    else:
        a = a_kt[:, :use_t]
        b = b_mt[:, :use_t]
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    a = a / a.std(dim=1, keepdim=True).clamp_min(eps)
    b = b / b.std(dim=1, keepdim=True).clamp_min(eps)
    return (a @ b.t()) / max(use_t - 1, 1)


def _expand_penalty_setting_for_names(raw_value, penalty_names: List[str], default_value, caster):
    if isinstance(raw_value, dict):
        base_default = raw_value.get("default", default_value)
        return [caster(raw_value.get(name, base_default)) for name in penalty_names]
    if isinstance(raw_value, (list, tuple)):
        if len(raw_value) != len(penalty_names):
            raise ValueError(f"Expected {len(penalty_names)} values, got {len(raw_value)}")
        return [caster(v) for v in raw_value]
    value = default_value if raw_value is None else raw_value
    return [caster(value) for _ in penalty_names]

def _select_rank_mask(probs_bkp: torch.Tensor, select_ranks: List[int], straight_through: bool = True) -> torch.Tensor:
    """
    select_ranks: list of 1-based ranks, e.g., [1,3,4].
    Returns mask [B,K,P] with straight-through option.
    """
    B, K, P = probs_bkp.shape
    ranks = []
    if select_ranks is not None:
        for r in select_ranks:
            try:
                r = int(r)
            except Exception:
                continue
            if 1 <= r <= P:
                ranks.append(r - 1)
    if len(ranks) == 0:
        k = min(P, 2)
        top_idx = probs_bkp.topk(k=k, dim=-1).indices
    else:
        order = torch.argsort(probs_bkp, dim=-1, descending=True)
        top_idx = torch.stack([order[..., r] for r in ranks], dim=-1)
    hard = torch.zeros_like(probs_bkp)
    hard.scatter_(-1, top_idx, 1.0)
    if straight_through:
        return hard - probs_bkp.detach() + probs_bkp
    return hard


def _compute_lambda_bkp(
    base_lambda_kp: torch.Tensor,
    feat_bkf: torch.Tensor,
    series_bkl: Optional[torch.Tensor] = None,
    dynamic_lambda: ClusterwiseDynamicLambda = None,
    dynamic_lambda_params: Optional[Dict[str, torch.Tensor]] = None,
    lambda_min_kp: torch.Tensor = None,
) -> torch.Tensor:
    lam = base_lambda_kp.unsqueeze(0).expand(feat_bkf.shape[0], -1, -1)
    if dynamic_lambda is not None:
        lam = lam * _module_call(dynamic_lambda, dynamic_lambda_params, feat_bkf, series_bkl=series_bkl)
    if lambda_min_kp is not None:
        lam = torch.maximum(lam, lambda_min_kp.unsqueeze(0))
    return lam


def _routed_penalty_loss(
    mask_bkp: torch.Tensor,
    lam_bkp: torch.Tensor,
    pen_bkp: torch.Tensor,
    gate_route_on_penalty_only: bool = True,
) -> torch.Tensor:
    if not gate_route_on_penalty_only:
        return (mask_bkp * lam_bkp * pen_bkp).sum(dim=-1)
    # Forward value = mask * lam * pen（数值上不变），但梯度拆分：
    # gate 参数只从 surrogate 项（mask * pen，无 lambda 缩放）获得梯度，
    # 使 gate 的路由决策基于惩罚量级本身，而非 lambda 放大后的值。
    weighted_penalty_bk = (mask_bkp.detach() * lam_bkp * pen_bkp).sum(dim=-1)
    gate_surrogate_bk = (mask_bkp * pen_bkp.detach()).sum(dim=-1)
    return weighted_penalty_bk + gate_surrogate_bk - gate_surrogate_bk.detach()


def _apply_skip_to_penalty_loss(
    penalty_loss_bk: torch.Tensor,
    skip_bk: Optional[torch.Tensor] = None,
    skip_cost: float = 0.0,
) -> torch.Tensor:
    if skip_bk is None:
        return penalty_loss_bk
    return (1.0 - skip_bk) * penalty_loss_bk + float(skip_cost) * skip_bk


def _pred_residual_loss_terms(
    pred_out: Optional[Dict[str, torch.Tensor]],
    y_base: torch.Tensor,
    y_final: torch.Tensor,
    y: torch.Tensor,
    penalty_names: List[str],
    penalty_fns: Dict[str, callable],
    cluster_id_c: torch.Tensor,
    K: int,
    penalty_scale: Optional[torch.Tensor] = None,
    specialization_weight: float = 0.0,
    norm_weight: float = 0.0,
    intervention_weight: float = 0.0,
) -> Dict[str, torch.Tensor]:
    zero_bk = y_final.new_zeros((y_final.shape[0], K))
    out = {
        "total_bk": zero_bk,
        "specialization_bk": zero_bk,
        "norm_bk": zero_bk,
        "intervention_bk": zero_bk,
    }
    if pred_out is None or (specialization_weight == 0.0 and norm_weight == 0.0 and intervention_weight == 0.0):
        return out
    residuals = pred_out.get("residuals")
    route_bcp = pred_out.get("route_bcp")
    effective_route_bcp = pred_out.get("effective_route_bcp", route_bcp)
    intervention_bcp = pred_out.get("intervention_bcp")
    alpha_cp = pred_out.get("alpha_cp")
    if residuals is None or route_bcp is None or effective_route_bcp is None or alpha_cp is None or residuals.numel() == 0:
        return out

    spec_bk = zero_bk
    if specialization_weight != 0.0 and len(penalty_names) > 0:
        y_final_sg = y_final.detach()
        for p, name in enumerate(penalty_names):
            visible_scale = (effective_route_bcp[..., p] * alpha_cp[:, p].unsqueeze(0)).detach()
            visible = visible_scale.unsqueeze(-1) * residuals[:, :, p, :]
            y_view = y_final_sg + visible - visible.detach()
            pen_bc = penalty_fns[name](y_view, y)
            if penalty_scale is not None and penalty_scale.numel() > p:
                scale_p = penalty_scale[p].to(device=pen_bc.device, dtype=pen_bc.dtype).clamp_min(1.0e-6)
                pen_bc = pen_bc / scale_p
            # Do not let unselected branches affect the scalar objective.
            pen_bc = pen_bc * effective_route_bcp[..., p].detach()
            spec_bk = spec_bk + scatter_mean_bc_to_bk(pen_bc, cluster_id_c, K)

    norm_bk = zero_bk
    if norm_weight != 0.0:
        branch_mse_bc = (y_final - y_base.detach()).pow(2).mean(dim=-1)
        norm_bk = scatter_mean_bc_to_bk(branch_mse_bc, cluster_id_c, K)

    intervention_bk = zero_bk
    if intervention_weight != 0.0 and intervention_bcp is not None:
        intervention_bc = (route_bcp.detach() * intervention_bcp).sum(dim=-1)
        intervention_bk = scatter_mean_bc_to_bk(intervention_bc, cluster_id_c, K)

    total_bk = (
        (float(specialization_weight) * spec_bk)
        + (float(norm_weight) * norm_bk)
        + (float(intervention_weight) * intervention_bk)
    )
    out["total_bk"] = total_bk
    out["specialization_bk"] = spec_bk
    out["norm_bk"] = norm_bk
    out["intervention_bk"] = intervention_bk
    return out


def _mae_objective_bc_from_abs(
    abs_err_bch: torch.Tensor,
    kind: str = "l1",
    beta: float = 1.0,
) -> torch.Tensor:
    kind = str(kind).lower()
    if kind in {"l1", "mae"}:
        return abs_err_bch.mean(dim=-1)
    if kind in {"smooth_l1", "huber"}:
        beta_t = max(float(beta), 1.0e-8)
        loss_bch = torch.where(
            abs_err_bch < beta_t,
            0.5 * abs_err_bch.pow(2) / beta_t,
            abs_err_bch - 0.5 * beta_t,
        )
        return loss_bch.mean(dim=-1)
    raise ValueError(f"Unsupported train.mae_objective.kind='{kind}'. Expected l1 or smooth_l1.")


def _gate_regularization(
    probs_bkp: torch.Tensor,
    gate_entropy_weight: float = 0.0,
    gate_balance_weight: float = 0.0,
    gate_entropy_target_frac: float = 0.0,
    gate_balance_target_kp: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if probs_bkp.numel() == 0 or (gate_entropy_weight == 0.0 and gate_balance_weight == 0.0):
        return probs_bkp.new_zeros(())
    p = probs_bkp.clamp_min(1.0e-8)
    reg = p.new_zeros(())
    if gate_entropy_weight != 0.0:
        ent = -(p * p.log()).sum(dim=-1).mean()
        if gate_entropy_target_frac > 0.0:
            target = gate_entropy_target_frac * float(np.log(float(p.shape[-1])))
            reg = reg + gate_entropy_weight * torch.relu(torch.tensor(target, device=ent.device, dtype=ent.dtype) - ent)
        else:
            reg = reg + (-gate_entropy_weight * ent)
    if gate_balance_weight != 0.0:
        avg = p.mean(dim=0)
        if gate_balance_target_kp is None:
            target = torch.full_like(avg, 1.0 / float(p.shape[-1]))
        else:
            target = gate_balance_target_kp.to(device=avg.device, dtype=avg.dtype)
        reg = reg + gate_balance_weight * (avg - target).pow(2).mean()
    return reg


def _apply_router_penalty_context(
    probs_bkp: torch.Tensor,
    pen_bkp: torch.Tensor,
    router_mode: str = "learned",
    penalty_context_weight: float = 0.0,
    detach_penalty_context: bool = True,
) -> torch.Tensor:
    mode = str(router_mode).lower()
    if probs_bkp is None or probs_bkp.numel() == 0:
        return probs_bkp
    if pen_bkp is None or pen_bkp.numel() == 0 or mode == "learned":
        return probs_bkp
    pen = pen_bkp.detach() if detach_penalty_context else pen_bkp
    pen_logits = pen.clamp_min(1.0e-8).log()
    if mode == "penalty_only":
        weight = float(penalty_context_weight) if penalty_context_weight > 0.0 else 1.0
        return torch.softmax(weight * pen_logits, dim=-1)
    if mode == "penalty_context":
        if penalty_context_weight <= 0.0:
            return probs_bkp
        route_logits = probs_bkp.clamp_min(1.0e-8).log() + (float(penalty_context_weight) * pen_logits)
        return torch.softmax(route_logits, dim=-1)
    raise ValueError(f"Unsupported moe.router_mode='{router_mode}'. Expected learned, penalty_context, or penalty_only.")


def _history_proxy_forecast(x_bcl: torch.Tensor, pred_len: int) -> torch.Tensor:
    """
    Build a target-free horizon proxy from observed history only.

    Router penalty context must be available at inference time.  For penalties
    that compare prediction to a reference series, use the latest observed
    history segment instead of the future label.
    """
    pred_len = int(pred_len)
    if pred_len <= 0:
        raise ValueError("pred_len must be positive")
    L = int(x_bcl.shape[-1])
    if L <= 0:
        raise ValueError("history length must be positive")
    if L >= pred_len:
        return x_bcl[..., -pred_len:]
    pad = x_bcl[..., -1:].expand(*x_bcl.shape[:-1], pred_len - L)
    return torch.cat([x_bcl, pad], dim=-1)


def _router_penalty_context_from_history(
    *,
    x_bcl: torch.Tensor,
    yhat_base_bch: torch.Tensor,
    penalty_names: List[str],
    penalty_fns: Dict[str, callable],
    penalty_scale: Optional[torch.Tensor],
    cluster_id_c: torch.Tensor,
    K: int,
) -> torch.Tensor:
    if len(penalty_names) == 0:
        return torch.zeros((x_bcl.shape[0], K, 0), device=x_bcl.device, dtype=x_bcl.dtype)
    y_ref = _history_proxy_forecast(x_bcl, int(yhat_base_bch.shape[-1]))
    y_ref = y_ref.to(device=yhat_base_bch.device, dtype=yhat_base_bch.dtype)
    route_pen_bcp = torch.stack(
        [penalty_fns[name](yhat_base_bch, y_ref) for name in penalty_names],
        dim=-1,
    )
    route_pen_bcp = normalize_penalties(route_pen_bcp, scale=penalty_scale)
    return scatter_mean_bcf_to_bkf(route_pen_bcp, cluster_id_c, K)


def _named_param_dict(module: Optional[nn.Module], detach: bool = False) -> Optional[Dict[str, torch.Tensor]]:
    if module is None:
        return None
    params = {}
    for name, param in module.named_parameters():
        params[name] = param.detach() if detach else param
    return params

@torch.no_grad()
def eval_loop(
    model: nn.Module,
    gate: ClusterwiseMoEGate,
    lambda_kp: torch.Tensor,
    penalty_names: List[str],
    penalty_fns: Dict[str, callable],
    loader: DataLoader,
    cluster_id_c: torch.Tensor,
    K: int,
    moe_cfg: dict,
    device: torch.device,
    select_ranks: List[int] = None,
    collect_plot: bool = False,
    plot_idx: torch.Tensor = None,
    channel_count: int = None,
    mse_weight: float = 1.0,
    gate_entropy_weight: float = 0.0,
    gate_balance_weight: float = 0.0,
    gate_soft_weight: float = 0.0,
    gate_entropy_target_frac: float = 0.0,
    penalty_scale: torch.Tensor = None,
    dynamic_lambda: ClusterwiseDynamicLambda = None,
    lambda_min_kp: torch.Tensor = None,
    mae_objective_weight: float = 0.0,
    mae_objective_kind: str = "l1",
    mae_objective_beta: float = 1.0,
    pred_residual: Optional[ClusterwisePredResidualMoE] = None,
    pred_residual_gate: Optional[nn.Module] = None,
    pred_residual_scale_c: Optional[torch.Tensor] = None,
    residual_correction_ch: Optional[torch.Tensor] = None,
    knn_hybrid: ShapeKNNHybrid = None,
    knn_fusion_scale_ch: Optional[torch.Tensor] = None,
    knn_fusion_gate: Optional[nn.Module] = None,
    eval_start: int = 0,
):
    model.eval()
    gate.eval()
    if dynamic_lambda is not None:
        dynamic_lambda.eval()
    if pred_residual is not None:
        pred_residual.eval()

    moe_enable = bool(moe_cfg.get("enable", True))
    allow_skip = bool(moe_cfg.get("allow_skip", False)) and moe_enable
    router_mode = str(moe_cfg.get("router_mode", "learned")).lower()
    router_penalty_context_weight = float(moe_cfg.get("router_penalty_context_weight", 0.0))
    router_detach_penalty_context = bool(moe_cfg.get("router_detach_penalty_context", True))

    P = len(penalty_names)
    total_loss_sum = torch.zeros(K, device=device)
    total_cnt = torch.zeros(K, device=device)
    mse_loss_sum = torch.zeros(K, device=device)
    mae_loss_sum = torch.zeros(K, device=device)

    # per-channel metrics
    se_c = torch.zeros(channel_count, device=device)
    ae_c = torch.zeros(channel_count, device=device)
    denom = 0

    plot_cache = {}  # idx -> (x[C,L], y[C,H], yhat[C,H])
    best_sample = {}   # c -> (x[L], y[H], yhat[H], mse)
    worst_sample = {}  # c -> (x[L], y[H], yhat[H], mse)
    best_mse = torch.full((channel_count,), float("inf"), device=device)
    worst_mse = torch.full((channel_count,), -float("inf"), device=device)

    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        idx = idx.to(device=device, dtype=torch.long, non_blocking=True)

        yhat_base = model(x, cluster_id_c)

        # Route on the base prediction. This keeps the router's penalty
        # context independent from the residual expert it is selecting.
        feat_bcf = extract_gate_features(x)            # [B,C,F]
        feat_bkf = scatter_mean_bcf_to_bkf(feat_bcf, cluster_id_c, K)  # [B,K,F]
        series_bkl = scatter_mean_bcl_to_bkl(x, cluster_id_c, K)  # [B,K,L]
        route_pen_bkp = _router_penalty_context_from_history(
            x_bcl=x,
            yhat_base_bch=yhat_base,
            penalty_names=penalty_names,
            penalty_fns=penalty_fns,
            penalty_scale=penalty_scale,
            cluster_id_c=cluster_id_c,
            K=K,
        )

        if moe_enable and P > 0:
            straight_through = (not moe_cfg["detach_penalty_grad"])
            mask_bkp, probs_bkp, skip_bk, _ = gate(feat_bkf, straight_through=straight_through)
            probs_bkp = _apply_router_penalty_context(
                probs_bkp,
                route_pen_bkp,
                router_mode=router_mode,
                penalty_context_weight=router_penalty_context_weight,
                detach_penalty_context=router_detach_penalty_context,
            )
            rank_mask = None
            if select_ranks is not None:
                mask_bkp = _select_rank_mask(probs_bkp, select_ranks, straight_through=straight_through)
                rank_mask = _select_rank_mask(probs_bkp, select_ranks, straight_through=False)
            if gate_soft_weight > 0.0:
                probs_sel = probs_bkp
                if rank_mask is not None:
                    probs_sel = probs_sel * rank_mask
                    probs_sel = probs_sel / probs_sel.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                target_mass = mask_bkp.detach().sum(dim=-1, keepdim=True).clamp_min(1.0)
                probs_sel = probs_sel * target_mass
                mask_bkp = (1.0 - gate_soft_weight) * mask_bkp + gate_soft_weight * probs_sel
        else:
            mask_bkp = torch.zeros_like(route_pen_bkp)
            probs_bkp = None
            skip_bk = None

        if pred_residual is not None and moe_enable and P > 0:
            pred_out = pred_residual(
                x,
                yhat_base,
                cluster_id_c,
                mask_bkp,
                skip_bk=skip_bk if allow_skip else None,
            )
            yhat = pred_out["y_final"]
            if pred_residual_gate is not None:
                pred_residual_gate.eval()
                gate_feat = _knn_gate_features(x, yhat_base, yhat)
                scale = pred_residual_gate(gate_feat).to(device=yhat.device, dtype=yhat.dtype).unsqueeze(-1)
                if pred_residual_scale_c is not None:
                    channel_scale = pred_residual_scale_c.to(device=yhat.device, dtype=yhat.dtype).view(1, -1, 1)
                    scale = scale * channel_scale
                yhat = yhat_base + scale * (yhat - yhat_base)
            elif pred_residual_scale_c is not None:
                scale = pred_residual_scale_c.to(device=yhat.device, dtype=yhat.dtype).view(1, -1, 1)
                yhat = yhat_base + scale * (yhat - yhat_base)
        else:
            yhat = yhat_base

        if knn_hybrid is not None:
            yhat_pre_knn = yhat
            query_start_abs_b = eval_start + idx
            yhat = knn_hybrid.hybridize_batch(x, yhat, cluster_id_c, query_start_abs_b=query_start_abs_b)
            if knn_fusion_scale_ch is not None:
                scale = knn_fusion_scale_ch.to(device=yhat.device, dtype=yhat.dtype).view(1, -1, 1)
                yhat = yhat_pre_knn + scale * (yhat - yhat_pre_knn)
            if knn_fusion_gate is not None:
                knn_fusion_gate.eval()
                gate_feat = _knn_gate_features(x, yhat_pre_knn, yhat)
                scale = knn_fusion_gate(gate_feat).to(device=yhat.device, dtype=yhat.dtype).unsqueeze(-1)
                yhat = yhat_pre_knn + scale * (yhat - yhat_pre_knn)
        if residual_correction_ch is not None:
            corr_ch = residual_correction_ch.to(device=yhat.device, dtype=yhat.dtype)
            yhat = yhat + corr_ch.unsqueeze(0)

        # base error metrics per channel
        err_bch = yhat - y
        abs_err_bch = err_bch.abs()
        mse_bc = err_bch.pow(2).mean(dim=-1)  # [B,C]
        mae_bc = abs_err_bch.mean(dim=-1)  # [B,C]
        mse_bk = scatter_mean_bc_to_bk(mse_bc, cluster_id_c, K)  # [B,K]
        mae_bk = scatter_mean_bc_to_bk(mae_bc, cluster_id_c, K)  # [B,K]
        mse_loss_sum += mse_bk.sum(dim=0)
        mae_loss_sum += mae_bk.sum(dim=0)

        # penalties on the final prediction.
        if P > 0:
            pen_bcp = []
            for name in penalty_names:
                pen_bc = penalty_fns[name](yhat, y)  # [B,C]
                pen_bcp.append(pen_bc)
            pen_bcp = torch.stack(pen_bcp, dim=-1)  # [B,C,P]
            pen_bcp = normalize_penalties(pen_bcp, scale=penalty_scale)
            pen_bkp = scatter_mean_bcf_to_bkf(pen_bcp, cluster_id_c, K)  # [B,K,P]
        else:
            pen_bkp = route_pen_bkp
        # loss per cluster
        lam = _compute_lambda_bkp(
            base_lambda_kp=lambda_kp,
            feat_bkf=feat_bkf,
            series_bkl=series_bkl,
            dynamic_lambda=dynamic_lambda,
            lambda_min_kp=lambda_min_kp,
        )
        # Validation/test objective excludes gate regularizers; they are training-only priors.
        penalty_loss_bk = (mask_bkp * lam * pen_bkp).sum(dim=-1)
        penalty_loss_bk = _apply_skip_to_penalty_loss(
            penalty_loss_bk,
            skip_bk=skip_bk if allow_skip else None,
            skip_cost=0.0,
        )
        if mae_objective_weight != 0.0:
            mae_objective_bc = _mae_objective_bc_from_abs(
                abs_err_bch,
                kind=mae_objective_kind,
                beta=mae_objective_beta,
            )
            mae_objective_bk = scatter_mean_bc_to_bk(mae_objective_bc, cluster_id_c, K)
        else:
            mae_objective_bk = torch.zeros_like(mse_bk)
        loss_bk = (mse_weight * mse_bk) + (float(mae_objective_weight) * mae_objective_bk) + penalty_loss_bk  # [B,K]
        total_loss_sum += loss_bk.sum(dim=0)
        total_cnt += torch.tensor([x.shape[0]], device=device).expand_as(total_cnt)

        # per-channel metrics
        accumulate_channel_errors(se_c, ae_c, yhat, y)
        denom += int(x.shape[0] * y.shape[2])

        # Track best/worst sample per channel by window MSE.
        win_mse_bc = (yhat - y).pow(2).mean(dim=-1)  # [B,C]

        for b in range(x.shape[0]):
            cur = win_mse_bc[b]  # [C]
            better = cur < best_mse
            worse = cur > worst_mse
            if better.any():
                idxs = better.nonzero(as_tuple=False).view(-1).tolist()
                for c in idxs:
                    best_mse[c] = cur[c]
                    best_sample[c] = (x[b, c].detach().cpu(), y[b, c].detach().cpu(), yhat[b, c].detach().cpu(), float(cur[c].item()))
            if worse.any():
                idxs = worse.nonzero(as_tuple=False).view(-1).tolist()
                for c in idxs:
                    worst_mse[c] = cur[c]
                    worst_sample[c] = (x[b, c].detach().cpu(), y[b, c].detach().cpu(), yhat[b, c].detach().cpu(), float(cur[c].item()))

        # Cache selected windows for plotting.
        if collect_plot and plot_idx is not None:
            # idx: [B]
            idx = idx.to(device)
            hit = torch.isin(idx, plot_idx)
            if hit.any():
                hit_pos = hit.nonzero(as_tuple=False).view(-1)
                for p in hit_pos.tolist():
                    gidx = int(idx[p].item())
                    if gidx not in plot_cache:
                        plot_cache[gidx] = (x[p].detach().cpu(), y[p].detach().cpu(), yhat[p].detach().cpu())

    avg_loss_k = total_loss_sum / total_cnt.clamp_min(1.0)
    avg_mse_k = mse_loss_sum / total_cnt.clamp_min(1.0)
    avg_mae_k = mae_loss_sum / total_cnt.clamp_min(1.0)
    mse_c, mae_c = mse_mae_from_sums(se_c, ae_c, denom)
    return avg_loss_k, avg_mse_k, avg_mae_k, mse_c.detach().cpu(), mae_c.detach().cpu(), plot_cache, best_sample, worst_sample


@torch.no_grad()
def estimate_residual_correction(
    model: nn.Module,
    loader: DataLoader,
    cluster_id_c: torch.Tensor,
    device: torch.device,
    method: str = "median",
    shrink: float = 1.0,
    max_abs: float = 0.0,
    knn_hybrid: ShapeKNNHybrid = None,
    eval_start: int = 0,
) -> Optional[torch.Tensor]:
    if len(loader) == 0:
        return None
    method = str(method).lower()
    if method not in {"median", "mean"}:
        raise ValueError(f"Unsupported calibration.method='{method}'. Expected median or mean.")
    model.eval()
    residuals = []
    sum_ch = None
    count = 0
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        idx = idx.to(device=device, dtype=torch.long, non_blocking=True)
        yhat = model(x, cluster_id_c)
        if knn_hybrid is not None:
            query_start_abs_b = eval_start + idx
            yhat = knn_hybrid.hybridize_batch(x, yhat, cluster_id_c, query_start_abs_b=query_start_abs_b)
        residual = y - yhat
        if method == "mean":
            batch_sum = residual.sum(dim=0)
            sum_ch = batch_sum if sum_ch is None else (sum_ch + batch_sum)
            count += int(residual.shape[0])
        else:
            residuals.append(residual.detach().cpu())
    if method == "mean":
        if sum_ch is None or count == 0:
            return None
        corr_ch = sum_ch / float(count)
    else:
        if len(residuals) == 0:
            return None
        corr_ch = torch.cat(residuals, dim=0).median(dim=0).values.to(device=device)
    corr_ch = corr_ch * float(shrink)
    if max_abs and float(max_abs) > 0.0:
        corr_ch = corr_ch.clamp(min=-float(max_abs), max=float(max_abs))
    return corr_ch.detach().cpu()


@torch.no_grad()
def estimate_knn_fusion_scale(
    model: nn.Module,
    loader: DataLoader,
    cluster_id_c: torch.Tensor,
    device: torch.device,
    knn_hybrid: ShapeKNNHybrid,
    eval_start: int = 0,
    metric: str = "mae",
    min_scale: float = 0.0,
    max_scale: float = 1.5,
    grid_steps: int = 31,
) -> Tuple[Optional[torch.Tensor], Dict[str, object]]:
    if len(loader) == 0 or knn_hybrid is None:
        return None, {"enable": False, "reason": "empty_loader_or_knn_disabled"}
    metric = str(metric).lower()
    if metric not in {"mae", "mse"}:
        raise ValueError("knn_hybrid.fusion_calibrator.metric must be mae or mse.")
    min_scale = float(min_scale)
    max_scale = float(max_scale)
    if max_scale < min_scale:
        min_scale, max_scale = max_scale, min_scale
    grid_steps = max(2, int(grid_steps))

    model.eval()
    C = int(cluster_id_c.numel())
    if metric == "mse":
        num_c = torch.zeros(C, device=device)
        den_c = torch.zeros(C, device=device)
    else:
        grid_g = torch.linspace(min_scale, max_scale, steps=grid_steps, device=device)
        ae_gc = torch.zeros((grid_steps, C), device=device)
    count = 0

    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        idx = idx.to(device=device, dtype=torch.long, non_blocking=True)
        base = model(x, cluster_id_c)
        query_start_abs_b = eval_start + idx
        hybrid = knn_hybrid.hybridize_batch(x, base, cluster_id_c, query_start_abs_b=query_start_abs_b)
        delta = hybrid - base
        target_delta = y - base
        if metric == "mse":
            num_c += (delta * target_delta).sum(dim=(0, 2))
            den_c += delta.pow(2).sum(dim=(0, 2))
        else:
            pred_gbch = base.unsqueeze(0) + grid_g.view(-1, 1, 1, 1) * delta.unsqueeze(0)
            ae_gc += (pred_gbch - y.unsqueeze(0)).abs().sum(dim=(1, 3))
        count += int(x.shape[0] * x.shape[-2] * y.shape[-1])

    if metric == "mse":
        scale_c = num_c / den_c.clamp_min(1.0e-12)
        scale_c = scale_c.clamp(min=min_scale, max=max_scale)
    else:
        best_idx_c = ae_gc.argmin(dim=0)
        scale_c = grid_g.index_select(0, best_idx_c)

    summary = {
        "enable": True,
        "metric": metric,
        "min_scale": min_scale,
        "max_scale": max_scale,
        "grid_steps": int(grid_steps),
        "scale": [float(v) for v in scale_c.detach().cpu().tolist()],
        "mean_scale": float(scale_c.mean().item()),
        "min_fitted_scale": float(scale_c.min().item()),
        "max_fitted_scale": float(scale_c.max().item()),
    }
    return scale_c.detach().cpu(), summary


def _knn_gate_features(
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
    base_std = base_bch.std(dim=-1) / hist_std
    hybrid_std = hybrid_bch.std(dim=-1) / hist_std
    base_shift = (base_bch.mean(dim=-1) - hist_last) / hist_std
    hybrid_shift = (hybrid_bch.mean(dim=-1) - hist_last) / hist_std
    delta_bias = delta_bch.mean(dim=-1) / hist_std

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


class KNNResidualGate(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        num_channels: int,
        hidden_dim: int = 32,
        dropout: float = 0.0,
        max_scale: float = 1.5,
        init_scale: float = 1.0,
        scale_mode: str = "sigmoid",
    ):
        super().__init__()
        self.F = int(feat_dim)
        self.C = int(num_channels)
        self.max_scale = max(float(max_scale), 1.0e-6)
        self.scale_mode = str(scale_mode).lower()
        if self.scale_mode not in {"sigmoid", "signed_tanh"}:
            raise ValueError("Residual gate scale_mode must be 'sigmoid' or 'signed_tanh'.")
        self.register_buffer("feature_mean", torch.zeros(1, 1, self.F), persistent=True)
        self.register_buffer("feature_std", torch.ones(1, 1, self.F), persistent=True)
        self.register_buffer("feature_standardize_enabled", torch.tensor(0.0), persistent=True)
        if self.scale_mode == "signed_tanh":
            init_scale = max(-self.max_scale + 1.0e-6, min(float(init_scale), self.max_scale - 1.0e-6))
            init_ratio = init_scale / self.max_scale
            init_bias = 0.5 * math.log((1.0 + init_ratio) / max(1.0 - init_ratio, 1.0e-6))
        else:
            init_scale = max(1.0e-6, min(float(init_scale), self.max_scale - 1.0e-6))
            init_prob = init_scale / self.max_scale
            init_bias = math.log(init_prob / max(1.0 - init_prob, 1.0e-6))
        self.net = nn.Sequential(
            nn.Linear(self.F, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity(),
            nn.Linear(int(hidden_dim), 1),
        )
        self.channel_bias = nn.Parameter(torch.zeros(self.C))
        self.reset_parameters(init_bias)

    def reset_parameters(self, init_bias: float):
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.constant_(last.bias, float(init_bias))

    def set_feature_standardization(self, mean_f: torch.Tensor, std_f: torch.Tensor) -> None:
        mean = mean_f.detach().view(1, 1, self.F).to(device=self.feature_mean.device, dtype=self.feature_mean.dtype)
        std = std_f.detach().view(1, 1, self.F).to(device=self.feature_std.device, dtype=self.feature_std.dtype)
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std.clamp_min(1.0e-6))
        self.feature_standardize_enabled.fill_(1.0)

    def clear_feature_standardization(self) -> None:
        self.feature_mean.zero_()
        self.feature_std.fill_(1.0)
        self.feature_standardize_enabled.zero_()

    def forward(self, feat_bcf: torch.Tensor) -> torch.Tensor:
        if bool(self.feature_standardize_enabled.item() > 0.5):
            feat_bcf = (feat_bcf - self.feature_mean.to(device=feat_bcf.device, dtype=feat_bcf.dtype)) / self.feature_std.to(
                device=feat_bcf.device,
                dtype=feat_bcf.dtype,
            )
        logits_bc = self.net(feat_bcf).squeeze(-1) + self.channel_bias.view(1, -1)
        if self.scale_mode == "signed_tanh":
            return self.max_scale * torch.tanh(logits_bc)
        return self.max_scale * torch.sigmoid(logits_bc)


@torch.no_grad()
def _collect_knn_gate_tensors(
    model: nn.Module,
    loader: DataLoader,
    cluster_id_c: torch.Tensor,
    device: torch.device,
    knn_hybrid: ShapeKNNHybrid,
    eval_start: int,
) -> Optional[Dict[str, torch.Tensor]]:
    if len(loader) == 0 or knn_hybrid is None:
        return None
    model.eval()
    feat_parts = []
    base_parts = []
    delta_parts = []
    y_parts = []
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        idx = idx.to(device=device, dtype=torch.long, non_blocking=True)
        base = model(x, cluster_id_c)
        query_start_abs_b = eval_start + idx
        hybrid = knn_hybrid.hybridize_batch(x, base, cluster_id_c, query_start_abs_b=query_start_abs_b)
        feat = _knn_gate_features(x, base, hybrid)
        feat_parts.append(feat.detach().cpu())
        base_parts.append(base.detach().cpu())
        delta_parts.append((hybrid - base).detach().cpu())
        y_parts.append(y.detach().cpu())
    if len(feat_parts) == 0:
        return None
    return {
        "feat": torch.cat(feat_parts, dim=0),
        "base": torch.cat(base_parts, dim=0),
        "delta": torch.cat(delta_parts, dim=0),
        "y": torch.cat(y_parts, dim=0),
    }


def train_knn_residual_gate(
    model: nn.Module,
    loader: DataLoader,
    cluster_id_c: torch.Tensor,
    device: torch.device,
    knn_hybrid: ShapeKNNHybrid,
    eval_start: int,
    channel_names: List[str],
    cfg: Dict[str, object],
) -> Tuple[Optional[KNNResidualGate], Dict[str, object]]:
    tensors = _collect_knn_gate_tensors(
        model=model,
        loader=loader,
        cluster_id_c=cluster_id_c,
        device=device,
        knn_hybrid=knn_hybrid,
        eval_start=eval_start,
    )
    if tensors is None:
        return None, {"enable": False, "reason": "empty_loader_or_knn_disabled"}

    feat = tensors["feat"]
    base = tensors["base"]
    delta = tensors["delta"]
    y = tensors["y"]
    n = int(feat.shape[0])
    c = int(feat.shape[1])
    train_fraction = float(cfg.get("train_fraction", 0.7))
    split = int(max(1, min(n - 1, round(n * train_fraction)))) if n > 1 else n
    train_idx = torch.arange(0, split, dtype=torch.long)
    hold_idx = torch.arange(split, n, dtype=torch.long)
    if hold_idx.numel() == 0:
        hold_idx = train_idx

    gate = KNNResidualGate(
        feat_dim=int(feat.shape[-1]),
        num_channels=c,
        hidden_dim=int(cfg.get("hidden_dim", 32)),
        dropout=float(cfg.get("dropout", 0.0)),
        max_scale=float(cfg.get("max_scale", 1.5)),
        init_scale=float(cfg.get("init_scale", 1.0)),
        scale_mode=str(cfg.get("scale_mode", "sigmoid")),
    ).to(device)
    lr = float(cfg.get("lr", 1.0e-3))
    weight_decay = float(cfg.get("weight_decay", 1.0e-4))
    batch_size = max(1, int(cfg.get("batch_size", 128)))
    epochs = max(1, int(cfg.get("epochs", 30)))
    scale_reg = float(cfg.get("scale_reg", 0.0))
    init_scale = float(cfg.get("init_scale", 1.0))
    loss_kind = str(cfg.get("loss", "mae")).lower()
    if loss_kind not in {"mae", "smooth_l1"}:
        raise ValueError("knn_hybrid.gate_calibrator.loss must be mae or smooth_l1.")
    beta = float(cfg.get("beta", 0.5))
    opt = torch.optim.AdamW(gate.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_hold = float("inf")
    best_epoch = 0

    def _loss_for(scale_bc: torch.Tensor, base_bch: torch.Tensor, delta_bch: torch.Tensor, y_bch: torch.Tensor):
        pred = base_bch + scale_bc.unsqueeze(-1) * delta_bch
        if loss_kind == "smooth_l1":
            loss = torch.nn.functional.smooth_l1_loss(pred, y_bch, beta=beta)
        else:
            loss = (pred - y_bch).abs().mean()
        if scale_reg > 0.0:
            loss = loss + scale_reg * (scale_bc - init_scale).pow(2).mean()
        return loss

    def _eval_idx(idx: torch.Tensor) -> Tuple[float, float, torch.Tensor]:
        gate.eval()
        ae = 0.0
        se = 0.0
        denom = 0
        scale_sum = torch.zeros(c, device=device)
        count = 0
        with torch.no_grad():
            for b0 in range(0, int(idx.numel()), batch_size):
                batch_idx = idx[b0:b0 + batch_size]
                feat_b = feat.index_select(0, batch_idx).to(device)
                base_b = base.index_select(0, batch_idx).to(device)
                delta_b = delta.index_select(0, batch_idx).to(device)
                y_b = y.index_select(0, batch_idx).to(device)
                scale_b = gate(feat_b)
                pred_b = base_b + scale_b.unsqueeze(-1) * delta_b
                err = pred_b - y_b
                ae += float(err.abs().sum().item())
                se += float(err.pow(2).sum().item())
                denom += int(err.numel())
                scale_sum += scale_b.sum(dim=0)
                count += int(scale_b.shape[0])
        return ae / max(denom, 1), se / max(denom, 1), (scale_sum / max(count, 1)).detach().cpu()

    for ep in range(1, epochs + 1):
        gate.train()
        perm = train_idx[torch.randperm(train_idx.numel())]
        for b0 in range(0, int(perm.numel()), batch_size):
            batch_idx = perm[b0:b0 + batch_size]
            feat_b = feat.index_select(0, batch_idx).to(device)
            base_b = base.index_select(0, batch_idx).to(device)
            delta_b = delta.index_select(0, batch_idx).to(device)
            y_b = y.index_select(0, batch_idx).to(device)
            scale_b = gate(feat_b)
            loss = _loss_for(scale_b, base_b, delta_b, y_b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), float(cfg.get("grad_clip", 1.0)))
            opt.step()
        hold_mae, _, _ = _eval_idx(hold_idx)
        if hold_mae < best_hold:
            best_hold = hold_mae
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in gate.state_dict().items()}

    if best_state is not None:
        gate.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    train_mae, train_mse, train_scale_c = _eval_idx(train_idx)
    hold_mae, hold_mse, hold_scale_c = _eval_idx(hold_idx)
    summary = {
        "enable": True,
        "train_windows": int(train_idx.numel()),
        "holdout_windows": int(hold_idx.numel()),
        "best_epoch": int(best_epoch),
        "train_mae": float(train_mae),
        "train_mse": float(train_mse),
        "holdout_mae": float(hold_mae),
        "holdout_mse": float(hold_mse),
        "channel_names": list(channel_names),
        "train_mean_scale": [float(v) for v in train_scale_c.tolist()],
        "holdout_mean_scale": [float(v) for v in hold_scale_c.tolist()],
    }
    gate.eval()
    return gate, summary


@torch.no_grad()
def _collect_pred_residual_gate_tensors(
    model: nn.Module,
    gate: ClusterwiseMoEGate,
    pred_residual: Optional[ClusterwisePredResidualMoE],
    loader: DataLoader,
    cluster_id_c: torch.Tensor,
    K: int,
    moe_cfg: dict,
    device: torch.device,
    penalty_names: List[str],
    penalty_fns: Dict[str, callable],
    penalty_scale: Optional[torch.Tensor],
    select_ranks: Optional[List[int]] = None,
    gate_soft_weight: float = 0.0,
) -> Optional[Dict[str, torch.Tensor]]:
    if len(loader) == 0 or pred_residual is None or len(penalty_names) == 0:
        return None
    moe_enable = bool(moe_cfg.get("enable", True))
    if not moe_enable:
        return None

    model.eval()
    gate.eval()
    pred_residual.eval()
    allow_skip = bool(moe_cfg.get("allow_skip", False)) and moe_enable
    router_mode = str(moe_cfg.get("router_mode", "learned")).lower()
    router_penalty_context_weight = float(moe_cfg.get("router_penalty_context_weight", 0.0))
    router_detach_penalty_context = bool(moe_cfg.get("router_detach_penalty_context", True))

    feat_parts = []
    base_parts = []
    delta_parts = []
    y_parts = []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        y_base = model(x, cluster_id_c)
        feat_bcf = extract_gate_features(x)
        feat_bkf = scatter_mean_bcf_to_bkf(feat_bcf, cluster_id_c, K)
        route_pen_bkp = _router_penalty_context_from_history(
            x_bcl=x,
            yhat_base_bch=y_base,
            penalty_names=penalty_names,
            penalty_fns=penalty_fns,
            penalty_scale=penalty_scale,
            cluster_id_c=cluster_id_c,
            K=K,
        )
        mask_bkp, probs_bkp, skip_bk, _ = gate(feat_bkf, straight_through=False)
        probs_bkp = _apply_router_penalty_context(
            probs_bkp,
            route_pen_bkp,
            router_mode=router_mode,
            penalty_context_weight=router_penalty_context_weight,
            detach_penalty_context=router_detach_penalty_context,
        )
        rank_mask = None
        if select_ranks is not None:
            mask_bkp = _select_rank_mask(probs_bkp, select_ranks, straight_through=False)
            rank_mask = mask_bkp
        if gate_soft_weight > 0.0:
            probs_sel = probs_bkp
            if rank_mask is not None:
                probs_sel = probs_sel * rank_mask
                probs_sel = probs_sel / probs_sel.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            target_mass = mask_bkp.detach().sum(dim=-1, keepdim=True).clamp_min(1.0)
            probs_sel = probs_sel * target_mass
            mask_bkp = (1.0 - gate_soft_weight) * mask_bkp + gate_soft_weight * probs_sel

        pred_out = pred_residual(
            x,
            y_base,
            cluster_id_c,
            mask_bkp,
            skip_bk=skip_bk if allow_skip else None,
        )
        y_residual = pred_out["y_final"]
        feat = _knn_gate_features(x, y_base, y_residual)
        feat_parts.append(feat.detach().cpu())
        base_parts.append(y_base.detach().cpu())
        delta_parts.append((y_residual - y_base).detach().cpu())
        y_parts.append(y.detach().cpu())

    if len(feat_parts) == 0:
        return None
    return {
        "feat": torch.cat(feat_parts, dim=0),
        "base": torch.cat(base_parts, dim=0),
        "delta": torch.cat(delta_parts, dim=0),
        "y": torch.cat(y_parts, dim=0),
    }


def train_pred_residual_gate(
    model: nn.Module,
    gate: ClusterwiseMoEGate,
    pred_residual: Optional[ClusterwisePredResidualMoE],
    loader: DataLoader,
    cluster_id_c: torch.Tensor,
    K: int,
    moe_cfg: dict,
    device: torch.device,
    penalty_names: List[str],
    penalty_fns: Dict[str, callable],
    penalty_scale: Optional[torch.Tensor],
    select_ranks: Optional[List[int]],
    gate_soft_weight: float,
    channel_names: List[str],
    cfg: Dict[str, object],
) -> Tuple[Optional[KNNResidualGate], Dict[str, object]]:
    tensors = _collect_pred_residual_gate_tensors(
        model=model,
        gate=gate,
        pred_residual=pred_residual,
        loader=loader,
        cluster_id_c=cluster_id_c,
        K=K,
        moe_cfg=moe_cfg,
        device=device,
        penalty_names=penalty_names,
        penalty_fns=penalty_fns,
        penalty_scale=penalty_scale,
        select_ranks=select_ranks,
        gate_soft_weight=gate_soft_weight,
    )
    if tensors is None:
        return None, {"enable": False, "reason": "empty_loader_or_residual_disabled"}

    feat = tensors["feat"]
    base = tensors["base"]
    delta = tensors["delta"]
    y = tensors["y"]
    n = int(feat.shape[0])
    c = int(feat.shape[1])
    train_fraction = float(cfg.get("train_fraction", 0.7))
    split = int(max(1, min(n - 1, round(n * train_fraction)))) if n > 1 else n
    train_idx = torch.arange(0, split, dtype=torch.long)
    hold_idx = torch.arange(split, n, dtype=torch.long)
    if hold_idx.numel() == 0:
        hold_idx = train_idx
    standardize_features = bool(cfg.get("standardize_features", False))

    residual_gate = KNNResidualGate(
        feat_dim=int(feat.shape[-1]),
        num_channels=c,
        hidden_dim=int(cfg.get("hidden_dim", 32)),
        dropout=float(cfg.get("dropout", 0.0)),
        max_scale=float(cfg.get("max_scale", 1.0)),
        init_scale=float(cfg.get("init_scale", 0.8)),
        scale_mode=str(cfg.get("scale_mode", "sigmoid")),
    ).to(device)
    lr = float(cfg.get("lr", 1.0e-3))
    weight_decay = float(cfg.get("weight_decay", 1.0e-4))
    batch_size = max(1, int(cfg.get("batch_size", 256)))
    epochs = max(1, int(cfg.get("epochs", 30)))
    scale_reg = float(cfg.get("scale_reg", 1.0e-4))
    init_scale = float(cfg.get("init_scale", 0.8))
    loss_kind = str(cfg.get("loss", "mse")).lower()
    if loss_kind not in {"mse", "mae", "smooth_l1"}:
        raise ValueError("moe.pred_side_residual.gate_calibrator.loss must be mse, mae, or smooth_l1.")
    selection_metric = str(cfg.get("selection_metric", loss_kind)).lower()
    if selection_metric not in {"mse", "mae"}:
        raise ValueError("moe.pred_side_residual.gate_calibrator.selection_metric must be mse or mae.")
    beta = float(cfg.get("beta", 0.5))
    opt = torch.optim.AdamW(residual_gate.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_hold = float("inf")
    best_epoch = 0
    feature_std_summary: Dict[str, object] = {
        "standardize_features": bool(standardize_features),
        "fit_windows": int(train_idx.numel()) if standardize_features else 0,
        "min_std": None,
        "max_std": None,
    }
    if standardize_features:
        feat_train = feat.index_select(0, train_idx)
        feat_flat = feat_train.reshape(-1, int(feat.shape[-1]))
        feat_mean = feat_flat.mean(dim=0)
        feat_std = feat_flat.std(dim=0, unbiased=False).clamp_min(1.0e-6)
        residual_gate.set_feature_standardization(feat_mean.to(device), feat_std.to(device))
        feature_std_summary.update(
            {
                "min_std": float(feat_std.min().item()),
                "max_std": float(feat_std.max().item()),
            }
        )

    def _loss_for(scale_bc: torch.Tensor, base_bch: torch.Tensor, delta_bch: torch.Tensor, y_bch: torch.Tensor):
        pred = base_bch + scale_bc.unsqueeze(-1) * delta_bch
        if loss_kind == "mse":
            loss = (pred - y_bch).pow(2).mean()
        elif loss_kind == "smooth_l1":
            loss = torch.nn.functional.smooth_l1_loss(pred, y_bch, beta=beta)
        else:
            loss = (pred - y_bch).abs().mean()
        if scale_reg > 0.0:
            loss = loss + scale_reg * (scale_bc - init_scale).pow(2).mean()
        return loss

    def _eval_idx(idx: torch.Tensor) -> Tuple[float, float, torch.Tensor]:
        residual_gate.eval()
        ae = 0.0
        se = 0.0
        denom = 0
        scale_sum = torch.zeros(c, device=device)
        count = 0
        with torch.no_grad():
            for b0 in range(0, int(idx.numel()), batch_size):
                batch_idx = idx[b0:b0 + batch_size]
                feat_b = feat.index_select(0, batch_idx).to(device)
                base_b = base.index_select(0, batch_idx).to(device)
                delta_b = delta.index_select(0, batch_idx).to(device)
                y_b = y.index_select(0, batch_idx).to(device)
                scale_b = residual_gate(feat_b)
                pred_b = base_b + scale_b.unsqueeze(-1) * delta_b
                err = pred_b - y_b
                ae += float(err.abs().sum().item())
                se += float(err.pow(2).sum().item())
                denom += int(err.numel())
                scale_sum += scale_b.sum(dim=0)
                count += int(scale_b.shape[0])
        return ae / max(denom, 1), se / max(denom, 1), (scale_sum / max(count, 1)).detach().cpu()

    for ep in range(1, epochs + 1):
        residual_gate.train()
        perm = train_idx[torch.randperm(train_idx.numel())]
        for b0 in range(0, int(perm.numel()), batch_size):
            batch_idx = perm[b0:b0 + batch_size]
            feat_b = feat.index_select(0, batch_idx).to(device)
            base_b = base.index_select(0, batch_idx).to(device)
            delta_b = delta.index_select(0, batch_idx).to(device)
            y_b = y.index_select(0, batch_idx).to(device)
            scale_b = residual_gate(feat_b)
            loss = _loss_for(scale_b, base_b, delta_b, y_b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(residual_gate.parameters(), float(cfg.get("grad_clip", 1.0)))
            opt.step()
        hold_mae, hold_mse, _ = _eval_idx(hold_idx)
        hold_metric = hold_mse if selection_metric == "mse" else hold_mae
        if hold_metric < best_hold:
            best_hold = hold_metric
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in residual_gate.state_dict().items()}

    if best_state is not None:
        residual_gate.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    train_mae, train_mse, train_scale_c = _eval_idx(train_idx)
    hold_mae, hold_mse, hold_scale_c = _eval_idx(hold_idx)
    summary = {
        "enable": True,
        "loss": loss_kind,
        "selection_metric": selection_metric,
        "scale_mode": str(residual_gate.scale_mode),
        "train_windows": int(train_idx.numel()),
        "holdout_windows": int(hold_idx.numel()),
        "best_epoch": int(best_epoch),
        "train_mae": float(train_mae),
        "train_mse": float(train_mse),
        "holdout_mae": float(hold_mae),
        "holdout_mse": float(hold_mse),
        "channel_names": list(channel_names),
        "train_mean_scale": [float(v) for v in train_scale_c.tolist()],
        "holdout_mean_scale": [float(v) for v in hold_scale_c.tolist()],
        "feature_standardization": feature_std_summary,
    }
    residual_gate.eval()
    return residual_gate, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    final_print = builtins.print
    cfg = load_yaml(args.config)
    if bool(cfg.get("console", {}).get("quiet", True)) and sys.stdout.isatty():
        builtins.print = lambda *args, **kwargs: None

    out_dir = cfg["exp"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    t_all0 = time.perf_counter()
    set_seed(
        int(cfg["exp"]["seed"]),
        deterministic=bool(cfg.get("exp", {}).get("deterministic", False)),
    )
    device = torch.device(cfg["exp"]["device"] if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # 1) 璇绘暟鎹?& 璁板綍閫氶亾鍚嶏紙璺宠繃 date 鍒楋紱header 涓嶈繘鍏ユ暟鎹級
    data_cfg = cfg["data"]
    data_tc, channel_names = read_csv_time_series(data_cfg["csv_path"], date_col=int(data_cfg["date_col"]))
    max_rows = int(data_cfg.get("max_rows", 0) or 0)
    if max_rows > 0:
        data_tc = data_tc[:max_rows]
    data_tc = data_tc.to(device)

    T, C = data_tc.shape
    print(f"Loaded data: T={T}, C={C}")

    tr = float(cfg["data"]["train_ratio"])
    vr = float(cfg["data"]["val_ratio"])
    te = float(cfg["data"]["test_ratio"])
    assert abs(tr + vr + te - 1.0) < 1e-6

    t_train = int(T * tr)
    t_val = int(T * (tr + vr))

    # 2) Normalize the time series.
    norm_cfg = cfg["normalize"]
    if norm_cfg["global_zscore"]:
        if norm_cfg.get("train_only", False):
            train_seg = data_tc[:t_train]
            mean_c = train_seg.mean(dim=0, keepdim=True)
            std_c = train_seg.std(dim=0, keepdim=True).clamp_min(1e-6)
            data_tc = (data_tc - mean_c) / std_c
            mean_c = mean_c.squeeze(0)
            std_c = std_c.squeeze(0)
        else:
            data_tc, mean_c, std_c = global_zscore(data_tc)

    # 3) corr matrix (skip when using random grouping)
    cl = cfg["cluster"]
    method_norm = str(cl.get("method", "agglomerative")).lower()
    cluster_fit_tc = data_tc[:t_train] if bool(cl.get("train_only", True)) else data_tc
    if bool(cl.get("train_only", True)):
        print("Cluster fit uses train split only.")
    if method_norm in {"random", "rand"}:
        C = int(data_tc.shape[1])
        corr_cc = torch.eye(C, device=data_tc.device, dtype=data_tc.dtype)
        if cfg["corr"]["compute"]:
            print("Skip corr matrix compute: cluster.method=random")
    else:
        corr_cc = pearson_corr_matrix(cluster_fit_tc)
        if cfg["corr"]["compute"]:
            save_path = cfg["corr"]["save_path"]
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, corr_cc.detach().cpu().numpy())
            print(f"Saved corr matrix to {save_path} (shape {corr_cc.shape})")

    # 4) 鑱氱被 + 灏忕皣鍚堝苟绛栫暐
    rs = cl.get("random_state", 0)
    cluster_id_c, clusters = cluster_channels_by_corr(
        corr_cc=corr_cc,
        data_tc=cluster_fit_tc,
        n_clusters=cl.get("n_clusters", None),
        distance_threshold=cl.get("distance_threshold", None),
        linkage=cl.get("linkage", "average"),
        method=cl.get("method", "agglomerative"),
        kmeans_n_init=int(cl.get("kmeans_n_init", 10)),
        kmeans_max_iter=int(cl.get("kmeans_max_iter", 300)),
        spectral_affinity=cl.get("spectral_affinity", "corr"),
        rbf_gamma=float(cl.get("rbf_gamma", 1.0)),
        dbscan_eps=cl.get("dbscan_eps", None),
        dbscan_min_samples=int(cl.get("dbscan_min_samples", 5)),
        random_state=None if rs is None else int(rs),
        min_cluster_size=int(cl["min_cluster_size"]),
        merge_small_clusters=bool(cl["merge_small_clusters"]),
        no_merge_if_channels_lt=int(cl["no_merge_if_channels_lt"]),
    )
    K = int(cluster_id_c.max().item() + 1)
    print(f"Clusters: K={K}")
    print_clusters(clusters, channel_names)
    cluster_sizes = torch.bincount(cluster_id_c, minlength=K).tolist()
    cluster_weight_k = torch.tensor(cluster_sizes, device=device, dtype=torch.float32)
    cluster_weight_k = cluster_weight_k / cluster_weight_k.sum().clamp_min(1.0)
    print("Cluster sizes: " + ", ".join(f"{k}:{n}" for k, n in enumerate(cluster_sizes)))

    # cluster memory config
    memory_cfg = cfg.get("memory", {})
    memory_enable = bool(memory_cfg.get("enable", False))
    memory_path = str(memory_cfg.get("path", os.path.join(out_dir, "cluster_memory.pt")))

    L = int(cfg["window"]["input_len"])
    H = int(cfg["window"]["pred_len"])
    eval_cfg = cfg.get("eval", {}) or {}
    skip_test = bool(eval_cfg.get("skip_test", True))

    # Keep materialized windows on CPU.  Electricity-style datasets with many
    # channels and long horizons can expand to tens of GB; batches are moved to
    # CUDA by the train/eval loops.
    data_window_tc = data_tc.detach().cpu()
    past_context = bool(cfg.get("window", {}).get("past_context", False))
    xtr, ytr = make_strict_windows(data_window_tc, L, H, 0, t_train)
    train_start_offsets = torch.arange(0, len(xtr), dtype=torch.long)
    val_eval_start = t_train
    test_eval_start = t_val
    if past_context:
        xva, yva, val_eval_start = make_label_range_windows(data_window_tc, L, H, t_train, t_val)
        if skip_test:
            xte = torch.empty(0, C, L, dtype=data_window_tc.dtype)
            yte = torch.empty(0, C, H, dtype=data_window_tc.dtype)
        else:
            xte, yte, test_eval_start = make_label_range_windows(data_window_tc, L, H, t_val, T)
    else:
        xva, yva = make_strict_windows(data_window_tc, L, H, t_train, t_val)
        if skip_test:
            xte = torch.empty(0, C, L, dtype=data_window_tc.dtype)
            yte = torch.empty(0, C, H, dtype=data_window_tc.dtype)
        else:
            xte, yte = make_strict_windows(data_window_tc, L, H, t_val, T)

    print(
        f"Windows: train={len(xtr)}, val={len(xva)}, test={len(xte)}, "
        f"past_context={past_context}"
    )

    dtr = WindowTensorDataset(xtr, ytr)
    dva = WindowTensorDataset(xva, yva)
    dte = WindowTensorDataset(xte, yte)

    knn_cfg = KNNShapeConfig.from_dict(cfg.get("knn_hybrid", {})).resolved_for_horizon(H)
    knn_hybrid_val = None
    knn_hybrid_test = None
    knn_bank_path = None
    knn_test_bank_windows = None
    knn_requires_base_bank = bool(knn_cfg.enable and knn_cfg.needs_base_bank_prediction())
    x_hist_val = y_hist_val = None
    starts_hist_val = None
    x_all = y_all = starts_all = None
    if knn_cfg.enable:
        if len(xtr) == 0:
            raise ValueError("knn_hybrid requires non-empty training windows.")
        if knn_requires_base_bank and knn_cfg.use_for_model_selection:
            raise ValueError(
                "knn_hybrid.use_for_model_selection=true is not supported when feature_mode=joint "
                "or template_mode=residual because the bank depends on model predictions. "
                "Set use_for_model_selection=false."
            )
        knn_bank_path = str(cfg.get("knn_hybrid", {}).get("path", os.path.join(out_dir, "knn_shape_bank.pt")))

    cluster_memory_bank = None
    if memory_enable:
        cluster_memory_bank = OnlineClusterMemory(
            num_clusters=K,
            memory_len=t_train,
            device=device,
            dtype=data_tc.dtype,
        )

    bs = int(cfg["train"]["batch_size"])
    pin_mem = (device.type == "cuda") and (xtr.device.type == "cpu")
    dl_tr = DataLoader(dtr, batch_size=bs, shuffle=True, num_workers=0, pin_memory=pin_mem)
    dl_va = DataLoader(dva, batch_size=bs, shuffle=False, num_workers=0, pin_memory=pin_mem)
    dl_te = DataLoader(dte, batch_size=bs, shuffle=False, num_workers=0, pin_memory=pin_mem)
    # penalties
    penalty_names = list(cfg["penalties"]["enabled"])
    penalty_fns = build_penalty_bank(penalty_names, jump_thr=float(cfg["penalties"]["jump_threshold"]))
    P = len(penalty_names)
    mse_weight = float(cfg["train"].get("mse_weight", 1.0))
    mae_objective_cfg = cfg["train"].get("mae_objective", {}) or {}
    mae_objective_enable = bool(mae_objective_cfg.get("enable", False))
    mae_objective_kind = str(mae_objective_cfg.get("kind", "l1")).lower()
    if mae_objective_kind not in {"l1", "mae", "smooth_l1", "huber"}:
        raise ValueError(
            f"Unsupported train.mae_objective.kind='{mae_objective_kind}'. Expected l1 or smooth_l1."
        )
    mae_objective_weight_final = float(mae_objective_cfg.get("weight", 0.0)) if mae_objective_enable else 0.0
    mae_objective_warmup_epochs = int(mae_objective_cfg.get("warmup_epochs", 0)) if mae_objective_enable else 0
    mae_objective_beta = float(mae_objective_cfg.get("beta", 1.0))
    if mae_objective_beta <= 0.0:
        raise ValueError("train.mae_objective.beta must be positive.")

    def mae_objective_weight_at(epoch_idx: int) -> float:
        if (not mae_objective_enable) or mae_objective_weight_final == 0.0:
            return 0.0
        if mae_objective_warmup_epochs <= 0:
            return mae_objective_weight_final
        scale = min(1.0, max(0.0, float(epoch_idx) / float(mae_objective_warmup_epochs)))
        return mae_objective_weight_final * scale

    calibration_cfg = cfg.get("calibration", {}) or {}
    calibration_enable = bool(calibration_cfg.get("enable", False))
    calibration_method = str(calibration_cfg.get("method", "median")).lower()
    calibration_shrink = float(calibration_cfg.get("shrink", 1.0))
    calibration_max_abs = float(calibration_cfg.get("max_abs", 0.0))
    if calibration_method not in {"median", "mean"}:
        raise ValueError(f"Unsupported calibration.method='{calibration_method}'. Expected median or mean.")
    if calibration_shrink < 0.0:
        raise ValueError("calibration.shrink must be non-negative.")

    selection_metric = str(cfg["train"].get("selection_metric", "val_loss")).lower()
    if selection_metric not in {"val_loss", "val_mse", "val_mae", "train_loss", "train_mse", "train_mae"}:
        raise ValueError(
            f"Unsupported train.selection_metric='{selection_metric}'. "
            "Expected one of: val_loss, val_mse, val_mae, train_loss, train_mse, train_mae."
        )
    penalty_warmup_epochs = int(cfg["train"].get("penalty_warmup_epochs", 0))
    penalty_scale_floor = float(cfg["train"].get("penalty_scale_floor", 1.0e-3))

    def compute_penalty_scale(loader: DataLoader, pred_len: int) -> torch.Tensor:
        if len(loader) == 0:
            return torch.full((P,), penalty_scale_floor, device=device)
        sum_all = torch.zeros(P, device=device)
        sum_pos = torch.zeros(P, device=device)
        cnt_all = 0
        cnt_pos = torch.zeros(P, device=device)
        for x, y, _ in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            last = x[..., -1:]
            yhat = last.expand(-1, -1, pred_len)
            pen_bcp = []
            for name in penalty_names:
                pen_bc = penalty_fns[name](yhat, y)
                pen_bcp.append(pen_bc)
            pen_bcp = torch.stack(pen_bcp, dim=-1)
            pen_flat = pen_bcp.reshape(-1, P)
            sum_all += pen_flat.sum(dim=0)
            cnt_all += int(pen_flat.shape[0])
            pos = pen_flat > 0
            sum_pos += (pen_flat * pos).sum(dim=0)
            cnt_pos += pos.sum(dim=0)
        if cnt_all == 0:
            return torch.full((P,), penalty_scale_floor, device=device)
        mean_all = sum_all / float(cnt_all)
        mean_pos = sum_pos / cnt_pos.clamp_min(1.0)
        scale = torch.where(cnt_pos > 0, mean_pos, mean_all)
        return scale.clamp_min(penalty_scale_floor)

    penalty_scale = compute_penalty_scale(dl_tr, H)

    # cluster portraits (prototype + penalty metrics)
    portrait_cfg = cfg.get("portrait", {})
    gate_prior_cfg = cfg.get("moe", {}).get("gate_prior", {})
    need_penalty_portrait = bool(portrait_cfg.get("enable", False)) or bool(gate_prior_cfg.get("enable", False))
    penalty_portrait_kp = None
    if need_penalty_portrait and len(penalty_names) > 0:
        # Portrait generation is diagnostic; keep it from advancing the shuffled
        # training loader RNG before model/gate initialization.
        portrait_generator = torch.Generator()
        portrait_generator.manual_seed(int(cfg["exp"]["seed"]))
        if len(dtr) > 0:
            portrait_loader = DataLoader(
                dtr, batch_size=bs, shuffle=False, num_workers=0,
                pin_memory=pin_mem, generator=portrait_generator
            )
        elif len(dva) > 0:
            portrait_loader = DataLoader(
                dva, batch_size=bs, shuffle=False, num_workers=0,
                pin_memory=pin_mem, generator=portrait_generator
            )
        else:
            portrait_loader = DataLoader(
                dte, batch_size=bs, shuffle=False, num_workers=0,
                pin_memory=pin_mem, generator=portrait_generator
            )
        penalty_portrait_kp = compute_cluster_penalty_portrait(
            portrait_loader, penalty_names, penalty_fns, cluster_id_c, K, H, device
        )
    if bool(portrait_cfg.get("enable", False)):
        portrait_dir = portrait_cfg.get("out_dir", os.path.join(out_dir, "cluster_portraits"))
        portrait_dpi = int(portrait_cfg.get("dpi", 140))
        max_points = int(portrait_cfg.get("max_points", 2000))
        jump_thr = float(portrait_cfg.get("jump_threshold", cfg.get("penalties", {}).get("jump_threshold", 2.0)))
        if penalty_portrait_kp is not None:
            metric_names = penalty_names
            metric_values = penalty_portrait_kp
        else:
            metric_names = None
            metric_values = None
        paths = save_cluster_portraits(
            out_dir=portrait_dir,
            data_tc=data_tc,
            cluster_id_c=cluster_id_c,
            jump_thr=jump_thr,
            dpi=portrait_dpi,
            max_points=max_points,
            metric_names=metric_names,
            metric_values_km=metric_values,
        )
        print(f"Saved cluster portraits to: {paths['dir']}")
        print(f"Portrait metrics: {paths['metrics_csv']}")

    # 6) Build the clusterwise predictor.
    model_cfg = cfg["model"]
    model = build_cluster_predictor(
        num_clusters=K,
        input_len=L,
        pred_len=H,
        model_cfg=model_cfg,
        num_channels=C,
        cluster_id_c=cluster_id_c,
    ).to(device)

    # 7) Configure MoE routing and lambda weighting.
    moe_cfg = cfg["moe"]
    moe_enable = bool(moe_cfg.get("enable", True))
    gate_entropy_weight = float(moe_cfg.get("gate_entropy_weight", 0.0))
    gate_balance_weight = float(moe_cfg.get("gate_balance_weight", 0.0))
    gate_soft_weight = float(moe_cfg.get("gate_soft_weight", 0.0))
    gate_entropy_target_frac = float(moe_cfg.get("gate_entropy_target_frac", 0.0))
    gate_route_on_penalty_only = bool(moe_cfg.get("gate_route_on_penalty_only", False))
    router_mode = str(moe_cfg.get("router_mode", "learned")).lower()
    router_penalty_context_weight = float(moe_cfg.get("router_penalty_context_weight", 0.0))
    router_detach_penalty_context = bool(moe_cfg.get("router_detach_penalty_context", True))
    pred_residual_cfg = moe_cfg.get("pred_side_residual", {}) or {}
    pred_residual_enable = bool(pred_residual_cfg.get("enable", False)) and moe_enable and P > 0
    pred_residual_specialization_weight = (
        float(pred_residual_cfg.get("specialization_weight", 0.1)) if pred_residual_enable else 0.0
    )
    pred_residual_norm_weight = float(pred_residual_cfg.get("norm_weight", 1.0e-4)) if pred_residual_enable else 0.0
    pred_residual_intervention_weight = (
        float(pred_residual_cfg.get("intervention_weight", 1.0e-3)) if pred_residual_enable else 0.0
    )
    pred_residual_detach_routed_penalty_pred = (
        bool(pred_residual_cfg.get("detach_routed_penalty_pred", False)) if pred_residual_enable else False
    )
    allow_skip = bool(moe_cfg.get("allow_skip", False)) and moe_enable and P > 0
    skip_cost = float(moe_cfg.get("skip_cost", 0.0)) if allow_skip else 0.0
    skip_init_bias = float(moe_cfg.get("skip_init_bias", -2.0))
    raw_ranks = moe_cfg.get("select_ranks", None)
    if raw_ranks is None:
        select_ranks = [1, 2]
    else:
        select_ranks = [int(x) for x in raw_ranks]
    gate_feat_dim = get_gate_feature_dim()
    gate = ClusterwiseMoEGate(
        num_clusters=K,
        feat_dim=gate_feat_dim,
        num_penalties=P,
        hidden_dim=int(moe_cfg.get("gate_hidden_dim", 64)),
        topk=int(moe_cfg["topk"]),
        allow_skip=allow_skip,
        skip_init_bias=skip_init_bias,
    ).to(device)
    gate.temperature = float(moe_cfg.get("gate_temperature", 1.0))
    gate.noise_std = float(moe_cfg.get("gate_noise_std", 0.0))
    gate.logit_clip = float(moe_cfg.get("gate_logit_clip", 0.0))
    gate.prob_floor = float(moe_cfg.get("gate_prob_floor", 0.0))
    gate_init_bias_cfg = moe_cfg.get("gate_init_bias", {}) or {}
    if P > 0 and bool(gate_init_bias_cfg.get("enable", False)):
        raw_bias = gate_init_bias_cfg.get("values", {}) or {}
        default_bias = float(raw_bias.get("default", 0.0)) if isinstance(raw_bias, dict) else 0.0
        bias_p = torch.tensor(
            [
                float(raw_bias.get(name, default_bias)) if isinstance(raw_bias, dict) else default_bias
                for name in penalty_names
            ],
            device=device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            for k in range(K):
                gate.b2[k].add_(bias_p)
        print(f"Gate init bias applied: {dict(zip(penalty_names, [float(v) for v in bias_p.detach().cpu().tolist()]))}")
    pred_residual = None
    if pred_residual_enable:
        pred_residual = ClusterwisePredResidualMoE(
            num_clusters=K,
            num_penalties=P,
            input_len=L,
            pred_len=H,
            hidden_dim=int(pred_residual_cfg.get("corrector_hidden", 32)),
            init_alpha=float(pred_residual_cfg.get("init_alpha", -3.0)),
            alpha_scale=float(pred_residual_cfg.get("alpha_scale", 0.5)),
            use_y_base_input=bool(pred_residual_cfg.get("use_y_base_input", True)),
            feature_mode=str(pred_residual_cfg.get("feature_mode", "legacy")),
            residual_clip=float(pred_residual_cfg.get("residual_clip", 0.0)),
            intervention_enable=bool(pred_residual_cfg.get("intervention_enable", False)),
            intervention_init=float(pred_residual_cfg.get("intervention_init", -2.0)),
        ).to(device)
        print(
            "Prediction residual MoE enabled: "
            f"hidden={pred_residual.hidden_dim}, feature_mode={pred_residual.feature_mode}, "
            f"alpha_scale={pred_residual.alpha_scale:.3f}, "
            f"residual_clip={pred_residual.residual_clip:.3f}, "
            f"specialization_weight={pred_residual_specialization_weight:.6f}, "
            f"norm_weight={pred_residual_norm_weight:.6f}, "
            f"intervention_weight={pred_residual_intervention_weight:.6f}, "
            f"detach_routed_penalty_pred={pred_residual_detach_routed_penalty_pred}"
        )
    gate_balance_target_kp = None
    gate_prior_enable = bool(gate_prior_cfg.get("enable", False)) and penalty_portrait_kp is not None and P > 0
    if gate_prior_enable:
        gate_prior_prob_kp = build_gate_prior_from_penalty_portrait(
            penalty_kp=penalty_portrait_kp,
            penalty_scale=penalty_scale,
            temperature=float(gate_prior_cfg.get("temperature", 1.0)),
            smoothing=float(gate_prior_cfg.get("smoothing", 0.0)),
            use_normalized_penalty=bool(gate_prior_cfg.get("use_normalized_penalty", True)),
        )
        gate.set_penalty_prior(
            gate_prior_prob_kp,
            strength=float(gate_prior_cfg.get("logit_strength", 1.0)),
        )
        if bool(gate_prior_cfg.get("use_as_balance_target", True)):
            gate_balance_target_kp = gate_prior_prob_kp
        print(f"Gate prior enabled: strength={gate.penalty_prior_strength:.3f}, prior={gate_prior_prob_kp.detach().cpu().tolist()}")

    epochs = int(cfg["train"]["epochs"])

    lambda_init_p = _expand_penalty_setting_for_names(moe_cfg.get("lambda_init", 1.0), penalty_names, 1.0, float)
    lambda_min_p = _expand_penalty_setting_for_names(moe_cfg.get("lambda_min", 0.0), penalty_names, 0.0, float)
    lambda_schedule_p = _expand_penalty_setting_for_names(
        moe_cfg.get("lambda_schedule", "cosine"),
        penalty_names,
        "cosine",
        lambda v: str(v).lower(),
    )
    lambda_min_kp = torch.tensor(lambda_min_p, device=device, dtype=torch.float32).view(1, P).expand(K, P)
    lambda_init_kp = torch.tensor(lambda_init_p, device=device, dtype=torch.float32).view(1, P).expand(K, P)

    learnable_lambda_cfg = moe_cfg.get("learnable_lambda", {})
    learnable_lambda_enable = (
        bool(learnable_lambda_cfg.get("enable", False))
        and moe_enable
        and P > 0
        and (not bool(moe_cfg.get("freeze_lambda", False)))
    )
    learnable_lambda_reg_weight = float(learnable_lambda_cfg.get("reg_weight", 0.0))
    learnable_lambda_share_floor = float(learnable_lambda_cfg.get("share_floor", 0.0))
    bilevel_cfg = learnable_lambda_cfg.get("bilevel", {})
    learnable_lambda = None
    if learnable_lambda_enable:
        learnable_lambda = ClusterwiseLearnableLambda(
            init_lambda_kp=lambda_init_kp,
            lambda_min_kp=lambda_min_kp,
            share_floor=learnable_lambda_share_floor,
        ).to(device)

    dyn_cfg = moe_cfg.get("dynamic_lambda", {})
    dynamic_lambda_enable = bool(dyn_cfg.get("enable", False)) and moe_enable and P > 0
    dynamic_lambda_reg_weight = float(dyn_cfg.get("reg_weight", 0.0))
    dynamic_lambda = None
    if dynamic_lambda_enable:
        dynamic_lambda = ClusterwiseDynamicLambda(
            num_clusters=K,
            feat_dim=gate_feat_dim,
            num_penalties=P,
            hidden_dim=int(dyn_cfg.get("hidden_dim", 32)),
            max_factor=float(dyn_cfg.get("max_factor", 2.0)),
            dropout=float(dyn_cfg.get("dropout", 0.0)),
            mode=str(dyn_cfg.get("mode", "multiscale")),
            mix=float(dyn_cfg.get("mix", 0.6)),
            tau_min=float(dyn_cfg.get("tau_min", 1.0)),
            tau_max=float(dyn_cfg.get("tau_max", 6.0)),
            series_downsample_len=int(dyn_cfg.get("series_downsample_len", 32)),
            segment_bins=dyn_cfg.get("segment_bins", (4, 8)),
        ).to(device)

    lambda_modules_present = (learnable_lambda is not None) or (dynamic_lambda is not None)
    bilevel_requested = bool(bilevel_cfg.get("enable", True)) if lambda_modules_present else False
    # Use a liquid-transformer-style unrolled update:
    # predictor/gate take an inner train-objective step, then lambda is updated by val_mse.
    bilevel_enable = lambda_modules_present and bilevel_requested and (len(dva) > 0)
    bilevel_optimize_gate = bool(bilevel_cfg.get("optimize_gate", False)) and bilevel_enable
    if lambda_modules_present and bilevel_requested and len(dva) == 0:
        raise ValueError("Lambda bilevel update requires a validation split because lambda must be updated from val_mse.")
    bilevel_outer_lr = float(bilevel_cfg.get("outer_lr", cfg["train"]["lr"]))
    bilevel_inner_lr = float(bilevel_cfg.get("inner_lr", cfg["train"]["lr"]))
    bilevel_outer_metric = str(bilevel_cfg.get("val_metric", "mse")).lower()
    if bilevel_enable and bilevel_outer_metric not in {"val_mse", "mse"}:
        print("Lambda outer optimization now uses val_mse only; learnable_lambda.bilevel.val_metric is ignored.")
    bilevel_steps_per_epoch = max(1, int(bilevel_cfg.get("steps_per_epoch", 1)))

    def lambda_value_at(epoch_idx: int, penalty_idx: int) -> float:
        lambda_max = lambda_init_p[penalty_idx]
        lambda_min = lambda_min_p[penalty_idx]
        lambda_schedule = lambda_schedule_p[penalty_idx]
        if lambda_schedule in {"cosine", "cosineannealing"}:
            if epochs <= 1:
                return lambda_max
            t = (epoch_idx - 1) / max(epochs - 1, 1)
            return lambda_min + 0.5 * (lambda_max - lambda_min) * (1.0 + math.cos(math.pi * t))
        return lambda_max

    def scheduled_lambda_kp_at(epoch_idx: int) -> torch.Tensor:
        lam_p = torch.tensor(
            [lambda_value_at(epoch_idx, p) for p in range(P)],
            device=device,
            dtype=torch.float32,
        )
        return lam_p.view(1, P).expand(K, P)

    def lambda_kp_at(epoch_idx: int, detach: bool = True) -> torch.Tensor:
        if learnable_lambda is not None:
            lam = learnable_lambda()
        else:
            lam = scheduled_lambda_kp_at(epoch_idx)
        return lam.detach() if detach else lam

    def lambda_kp_from_epochs(epoch_k: torch.Tensor) -> torch.Tensor:
        if learnable_lambda is not None:
            return learnable_lambda().detach()
        rows = [
            torch.tensor(
                [lambda_value_at(int(e), p) for p in range(P)],
                device=device,
                dtype=torch.float32,
            )
            for e in epoch_k.detach().cpu().tolist()
        ]
        if len(rows) == 0:
            return torch.zeros((0, P), device=device)
        return torch.stack(rows, dim=0)

    finetune_summary = None

    def apply_finetune_warm_start():
        nonlocal finetune_summary
        ft_cfg = cfg.get("finetune", {})
        if not bool(ft_cfg.get("enable", False)):
            return

        ckpt_path = str(ft_cfg.get("checkpoint_path", ""))
        if len(ckpt_path) == 0:
            raise ValueError("finetune.enable=true requires finetune.checkpoint_path.")
        ckpt = load_cluster_checkpoint(ckpt_path, device=device)
        meta = ckpt.get("meta", {})
        if len(meta) == 0:
            raise ValueError(f"Fine-tune checkpoint meta is missing: {ckpt_path}")

        src_k_count = int(meta.get("K", 0))
        src_input_len = int(meta.get("input_len", -1))
        src_pred_len = int(meta.get("pred_len", -1))
        if bool(ft_cfg.get("strict_window", True)) and (src_input_len != L or src_pred_len != H):
            raise ValueError(
                "Fine-tune checkpoint window mismatch: "
                f"source input_len/pred_len={src_input_len}/{src_pred_len}, target={L}/{H}. "
                "Train or choose a source checkpoint with the same horizon."
            )
        if src_k_count <= 0:
            raise ValueError(f"Invalid source cluster count in fine-tune checkpoint: {src_k_count}")

        src_model_cfg = dict(meta.get("model_cfg", {}))
        if bool(ft_cfg.get("strict_model", True)) and src_model_cfg != dict(model_cfg):
            raise ValueError("Fine-tune source model_cfg differs from target model_cfg.")
        src_cluster_id_c = meta.get("cluster_id_c", None)
        src_num_channels = meta.get("num_channels", None)
        if bool(dict(src_model_cfg.get("channel_adapter", {}) or {}).get("enable", False)):
            if src_cluster_id_c is None or src_num_channels is None:
                raise ValueError("Fine-tune source checkpoint with channel_adapter requires cluster_id_c and num_channels in meta.")
        source_model = build_cluster_predictor(
            num_clusters=src_k_count,
            input_len=src_input_len,
            pred_len=src_pred_len,
            model_cfg=src_model_cfg,
            num_channels=None if src_num_channels is None else int(src_num_channels),
            cluster_id_c=src_cluster_id_c,
        ).to(device)
        source_model.load_state_dict(ckpt["model_state"], strict=True)
        source_model.eval()

        map_mode = str(ft_cfg.get("cluster_map", "index")).lower()
        if map_mode in {"index", "same"}:
            target_to_source_k = torch.arange(K, device=device, dtype=torch.long) % src_k_count
            corr_map = None
        else:
            memory_path = str(ft_cfg.get("memory_path", ""))
            if len(memory_path) == 0:
                raise ValueError("finetune.cluster_map requires finetune.memory_path unless cluster_map=index.")
            source_memory = load_cluster_memory(memory_path, device=device)
            source_proto_kt = source_memory["prototypes_kt"].to(device)
            target_proto_kt = compute_cluster_prototypes(data_tc[:t_train], cluster_id_c)
            corr_map = _rowwise_corr(
                target_proto_kt,
                source_proto_kt,
                align=str(ft_cfg.get("corr_align", "head")),
            )
            target_to_source_k = torch.argmax(corr_map, dim=1).to(torch.long)

        if bool(ft_cfg.get("load_model", True)):
            for k in range(K):
                src_k = int(target_to_source_k[k].item())
                model.load_cluster_state(k, source_model.get_cluster_state(src_k))

        source_penalty_names = list(meta.get("penalty_names", []))
        same_penalties = source_penalty_names == penalty_names
        if bool(ft_cfg.get("load_gate", True)) and "gate_state" in ckpt:
            if not same_penalties:
                raise ValueError(
                    "Fine-tune gate loading requires identical penalty_names: "
                    f"source={source_penalty_names}, target={penalty_names}"
                )
            src_moe_cfg = dict(meta.get("moe_cfg", {}))
            source_gate_state = ckpt["gate_state"]
            source_gate_allow_skip = any(str(name).startswith("W_skip.") for name in source_gate_state.keys())
            source_gate = ClusterwiseMoEGate(
                num_clusters=src_k_count,
                feat_dim=int(meta.get("gate_feat_dim", gate_feat_dim)),
                num_penalties=len(source_penalty_names),
                hidden_dim=int(src_moe_cfg.get("gate_hidden_dim", src_moe_cfg.get("hidden_dim", 64))),
                topk=int(src_moe_cfg.get("topk", 1)),
                allow_skip=source_gate_allow_skip,
                skip_init_bias=float(src_moe_cfg.get("skip_init_bias", -2.0)),
            ).to(device)
            source_gate.load_state_dict(source_gate_state, strict=True)
            source_gate.eval()
            for k in range(K):
                src_k = int(target_to_source_k[k].item())
                gate.load_cluster_state(k, source_gate.get_cluster_state(src_k))

        if bool(ft_cfg.get("load_dynamic_lambda", True)) and dynamic_lambda is not None and "dynamic_lambda_state" in ckpt:
            if not same_penalties:
                raise ValueError(
                    "Fine-tune dynamic_lambda loading requires identical penalty_names: "
                    f"source={source_penalty_names}, target={penalty_names}"
                )
            src_moe_cfg = dict(meta.get("moe_cfg", {}))
            src_dyn_cfg = src_moe_cfg.get("dynamic_lambda", {})
            source_dynamic_lambda = ClusterwiseDynamicLambda(
                num_clusters=src_k_count,
                feat_dim=int(meta.get("gate_feat_dim", gate_feat_dim)),
                num_penalties=len(source_penalty_names),
                hidden_dim=int(src_dyn_cfg.get("hidden_dim", 32)),
                max_factor=float(src_dyn_cfg.get("max_factor", 2.0)),
                dropout=float(src_dyn_cfg.get("dropout", 0.0)),
                mode=str(src_dyn_cfg.get("mode", "multiscale")),
                mix=float(src_dyn_cfg.get("mix", 0.6)),
                tau_min=float(src_dyn_cfg.get("tau_min", 1.0)),
                tau_max=float(src_dyn_cfg.get("tau_max", 6.0)),
                series_downsample_len=int(src_dyn_cfg.get("series_downsample_len", 32)),
                segment_bins=src_dyn_cfg.get("segment_bins", (4, 8)),
            ).to(device)
            source_dynamic_lambda.load_state_dict(ckpt["dynamic_lambda_state"], strict=True)
            source_dynamic_lambda.eval()
            for k in range(K):
                src_k = int(target_to_source_k[k].item())
                dynamic_lambda.load_cluster_state(k, source_dynamic_lambda.get_cluster_state(src_k))

        if bool(ft_cfg.get("load_learnable_lambda", True)) and learnable_lambda is not None and "learnable_lambda_state" in ckpt:
            if not same_penalties:
                raise ValueError(
                    "Fine-tune learnable_lambda loading requires identical penalty_names: "
                    f"source={source_penalty_names}, target={penalty_names}"
                )
            init = torch.ones((src_k_count, len(source_penalty_names)), device=device, dtype=torch.float32)
            mins = torch.zeros_like(init)
            source_learnable_lambda = ClusterwiseLearnableLambda(init, mins).to(device)
            source_learnable_lambda.load_state_dict(ckpt["learnable_lambda_state"], strict=False)
            for k in range(K):
                src_k = int(target_to_source_k[k].item())
                learnable_lambda.load_cluster_state(k, source_learnable_lambda.get_cluster_state(src_k))

        finetune_summary = {
            "checkpoint_path": ckpt_path,
            "memory_path": str(ft_cfg.get("memory_path", "")),
            "cluster_map": map_mode,
            "target_to_source_cluster": [int(v) for v in target_to_source_k.detach().cpu().tolist()],
            "cluster_corr": None if corr_map is None else corr_map.detach().cpu().tolist(),
        }
        print(f"Fine-tune warm start loaded from: {ckpt_path}")
        print(f"Fine-tune target->source cluster map: {finetune_summary['target_to_source_cluster']}")

    apply_finetune_warm_start()

    cluster_params = []
    for k in range(K):
        params_k = [
            *model.get_cluster_params(k),
        ]
        if not (bilevel_enable and bilevel_optimize_gate):
            params_k.extend([gate.W1[k], gate.b1[k], gate.W2[k], gate.b2[k]])
        if pred_residual is not None:
            params_k.extend(pred_residual.get_cluster_params(k))
        if dynamic_lambda is not None and (not bilevel_enable):
            params_k.extend(dynamic_lambda.get_cluster_params(k))
        if learnable_lambda is not None and (not bilevel_enable):
            params_k.append(learnable_lambda.raw[k])
        cluster_params.append(params_k)

    optimizers = [
        torch.optim.Adam(
            params_k,
            lr=float(cfg["train"]["lr"]),
            weight_decay=float(cfg["train"]["weight_decay"]),
        )
        for params_k in cluster_params
    ]
    sched_cfg = cfg["train"].get("lr_scheduler", {"name": "none"})
    sched_name = str(sched_cfg.get("name", "none")).lower()
    if sched_name in {"plateau", "reduce", "reduce_on_plateau"}:
        schedulers = [
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt_k,
                mode="min",
                factor=float(sched_cfg.get("factor", 0.5)),
                patience=int(sched_cfg.get("patience", 3)),
                min_lr=float(sched_cfg.get("min_lr", 1.0e-6)),
            )
            for opt_k in optimizers
        ]
    elif sched_name in {"cosine", "cosineannealing"}:
        schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(
                opt_k,
                T_max=int(sched_cfg.get("t_max", 50)),
                eta_min=float(sched_cfg.get("min_lr", 1.0e-6)),
            )
            for opt_k in optimizers
        ]
    elif sched_name in {"step", "steplr"}:
        schedulers = [
            torch.optim.lr_scheduler.StepLR(
                opt_k,
                step_size=int(sched_cfg.get("step_size", 10)),
                gamma=float(sched_cfg.get("gamma", 0.5)),
            )
            for opt_k in optimizers
        ]
    else:
        schedulers = None

    lambda_optimizer = None
    if bilevel_enable:
        lambda_params = []
        if bilevel_optimize_gate:
            lambda_params.extend(list(gate.parameters()))
        if dynamic_lambda is not None:
            lambda_params.extend(list(dynamic_lambda.parameters()))
        if learnable_lambda is not None:
            lambda_params.extend(list(learnable_lambda.parameters()))
        if len(lambda_params) > 0:
            lambda_optimizer = torch.optim.Adam(
                lambda_params,
                lr=bilevel_outer_lr,
                weight_decay=0.0,
            )

    def _fit_knn_hybrid_bank(
        x_bank_ncl: torch.Tensor,
        y_bank_nch: torch.Tensor,
        start_offsets_n: torch.Tensor,
    ) -> ShapeKNNHybrid:
        base_bank_pred = None
        if knn_requires_base_bank:
            base_bank_pred = predict_bank_outputs(
                model=model,
                x_bank_ncl=x_bank_ncl,
                cluster_id_c=cluster_id_c,
                batch_size=max(bs, 64),
                device=device,
            )
        return ShapeKNNHybrid.fit(
            x_bank_ncl=x_bank_ncl,
            y_bank_nch=y_bank_nch,
            cluster_id_c=cluster_id_c,
            cfg=knn_cfg,
            start_offsets_n=start_offsets_n,
            base_bank_pred_nch=base_bank_pred,
        )

    def _make_knn_bank_windows(start: int, end: int, label: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_bank, y_bank = make_strict_windows(data_window_tc, L, H, start, end)
        if len(x_bank) == 0:
            raise ValueError(f"KNN hybrid {label} bank is empty. Increase available history or change knn_hybrid.bank_split.")
        starts = torch.arange(int(start), int(start) + len(x_bank), dtype=torch.long)
        return x_bank, y_bank, starts

    def _refresh_knn_hybrids() -> None:
        nonlocal knn_hybrid_val, knn_hybrid_test, knn_test_bank_windows
        nonlocal x_hist_val, y_hist_val, starts_hist_val, x_all, y_all, starts_all
        if not knn_cfg.enable:
            return

        x_bank_val = xtr
        y_bank_val = ytr
        starts_bank_val = train_start_offsets
        knn_val_bank_split = "train"
        if knn_cfg.mode == "rolling" and knn_cfg.bank_split in {"pre_test", "history"}:
            if x_hist_val is None or y_hist_val is None:
                x_hist_val, y_hist_val, starts_hist_val = _make_knn_bank_windows(0, t_val, "pre_test")
            if x_hist_val is None or y_hist_val is None or len(x_hist_val) == 0:
                raise ValueError("knn_hybrid bank_split=pre_test requires non-empty windows before test.")
            x_bank_val, y_bank_val = x_hist_val, y_hist_val
            starts_bank_val = starts_hist_val
            knn_val_bank_split = "pre_test"

        knn_hybrid_val = _fit_knn_hybrid_bank(x_bank_val, y_bank_val, starts_bank_val)
        knn_hybrid_test = knn_hybrid_val
        knn_test_bank_windows = int(len(x_bank_val))
        knn_val_info = knn_hybrid_val.describe()
        print(
            "KNN hybrid (val): "
            f"mode={knn_val_info['mode']}, "
            f"scope={knn_val_info['scope']}, "
            f"bank_split={knn_val_bank_split}, "
            f"k={knn_val_info['k']}, "
            f"alpha={knn_val_info['alpha']:.3f}, "
            f"feature_mode={knn_val_info['feature_mode']}, "
            f"template_mode={knn_val_info['template_mode']}, "
            f"bank_sizes={knn_val_info['bank_sizes']}"
        )

        if knn_cfg.bank_split in {"pre_test", "history"}:
            if x_hist_val is None or y_hist_val is None:
                x_hist_val, y_hist_val, starts_hist_val = _make_knn_bank_windows(0, t_val, "pre_test")
            if x_hist_val is None or y_hist_val is None or len(x_hist_val) == 0:
                raise ValueError("knn_hybrid bank_split=pre_test requires non-empty windows before test.")
            x_bank_test, y_bank_test = x_hist_val, y_hist_val
            starts_bank_test = starts_hist_val
            if knn_cfg.mode == "rolling" and knn_cfg.bank_split == "history":
                if x_all is None or y_all is None:
                    x_all, y_all, starts_all = _make_knn_bank_windows(0, T, "full-history")
                if x_all is None or y_all is None or len(x_all) == 0:
                    raise ValueError("knn_hybrid mode=rolling bank_split=history requires non-empty full-history windows.")
                x_bank_test, y_bank_test = x_all, y_all
                starts_bank_test = starts_all
            knn_hybrid_test = _fit_knn_hybrid_bank(x_bank_test, y_bank_test, starts_bank_test)
            knn_test_bank_windows = int(len(x_bank_test))
            if knn_hybrid_test is not knn_hybrid_val:
                knn_test_info = knn_hybrid_test.describe()
                print(
                    "KNN hybrid (test): "
                    f"mode={knn_test_info['mode']}, "
                    f"scope={knn_test_info['scope']}, "
                    f"bank_split={knn_cfg.bank_split}, "
                    f"k={knn_test_info['k']}, "
                    f"alpha={knn_test_info['alpha']:.3f}, "
                    f"feature_mode={knn_test_info['feature_mode']}, "
                    f"template_mode={knn_test_info['template_mode']}, "
                    f"bank_sizes={knn_test_info['bank_sizes']}"
                )

    def _build_knn_hybrids_for(eval_knn_cfg: KNNShapeConfig) -> Tuple[ShapeKNNHybrid, ShapeKNNHybrid, int]:
        def _fit_bank_for_cfg(
            x_bank_ncl: torch.Tensor,
            y_bank_nch: torch.Tensor,
            start_offsets_n: torch.Tensor,
        ) -> ShapeKNNHybrid:
            base_bank_pred = None
            if eval_knn_cfg.needs_base_bank_prediction():
                base_bank_pred = predict_bank_outputs(
                    model=model,
                    x_bank_ncl=x_bank_ncl,
                    cluster_id_c=cluster_id_c,
                    batch_size=max(bs, 64),
                    device=device,
                )
            return ShapeKNNHybrid.fit(
                x_bank_ncl=x_bank_ncl,
                y_bank_nch=y_bank_nch,
                cluster_id_c=cluster_id_c,
                cfg=eval_knn_cfg,
                start_offsets_n=start_offsets_n,
                base_bank_pred_nch=base_bank_pred,
            )

        x_bank_val = xtr
        y_bank_val = ytr
        starts_bank_val = train_start_offsets
        knn_val_bank_split = "train"
        if eval_knn_cfg.mode == "rolling" and eval_knn_cfg.bank_split in {"pre_test", "history"}:
            x_bank_val, y_bank_val, starts_bank_val = _make_knn_bank_windows(0, t_val, "pre_test")
            knn_val_bank_split = "pre_test"

        hybrid_val = _fit_bank_for_cfg(x_bank_val, y_bank_val, starts_bank_val)
        hybrid_test = hybrid_val
        bank_windows = int(len(x_bank_val))
        val_info = hybrid_val.describe()
        print(
            "KNN sweep (val): "
            f"mode={val_info['mode']}, "
            f"scope={val_info['scope']}, "
            f"bank_split={knn_val_bank_split}, "
            f"k={val_info['k']}, "
            f"alpha={val_info['alpha']:.3f}, "
            f"adaptive_alpha={val_info['adaptive_alpha']}, "
            f"distance_weight={val_info['distance_weight']}"
        )

        if eval_knn_cfg.bank_split in {"pre_test", "history"}:
            x_bank_test, y_bank_test, starts_bank_test = _make_knn_bank_windows(0, t_val, "pre_test")
            if eval_knn_cfg.mode == "rolling" and eval_knn_cfg.bank_split == "history":
                x_bank_test, y_bank_test, starts_bank_test = _make_knn_bank_windows(0, T, "full-history")
            hybrid_test = _fit_bank_for_cfg(x_bank_test, y_bank_test, starts_bank_test)
            bank_windows = int(len(x_bank_test))
            test_info = hybrid_test.describe()
            print(
                "KNN sweep (test): "
                f"mode={test_info['mode']}, "
                f"scope={test_info['scope']}, "
                f"bank_split={test_info['bank_split']}, "
                f"k={test_info['k']}, "
                f"alpha={test_info['alpha']:.3f}, "
                f"adaptive_alpha={test_info['adaptive_alpha']}, "
                f"distance_weight={test_info['distance_weight']}"
            )
        return hybrid_val, hybrid_test, bank_windows

    if knn_cfg.enable and (not knn_requires_base_bank):
        _refresh_knn_hybrids()
    elif knn_cfg.enable:
        print(
            "KNN hybrid bank construction is deferred until the best checkpoint is loaded "
            "because feature_mode=joint or template_mode=residual depends on model predictions."
        )

    monitor_metric = selection_metric
    if len(dva) == 0 and monitor_metric.startswith("val_"):
        monitor_metric = "train_" + monitor_metric[4:]
        print(f"Validation split is empty; fallback train.selection_metric -> {monitor_metric}")

    def _select_monitor_k(
        train_loss_k: torch.Tensor,
        train_mse_k: torch.Tensor,
        train_mae_k: torch.Tensor,
        val_loss_k: torch.Tensor,
        val_mse_k: torch.Tensor,
        val_mae_k: torch.Tensor,
    ) -> torch.Tensor:
        if monitor_metric == "val_loss":
            return val_loss_k
        if monitor_metric == "val_mse":
            return val_mse_k
        if monitor_metric == "val_mae":
            return val_mae_k
        if monitor_metric == "train_loss":
            return train_loss_k
        if monitor_metric == "train_mse":
            return train_mse_k
        return train_mae_k

    early_stop_start_epoch = max(1, penalty_warmup_epochs + 1)
    selection_start_epoch = int(cfg["train"].get("model_selection_start_epoch", early_stop_start_epoch))
    selection_start_epoch = max(1, min(selection_start_epoch, epochs))
    if early_stop_start_epoch > 1:
        print(f"Early stop counting starts at epoch {early_stop_start_epoch} after penalty warmup.")
    if selection_start_epoch > 1:
        print(f"Checkpoint selection starts at epoch {selection_start_epoch}.")

    # early stop
    es = cfg["early_stop"]
    patience = int(es["patience"])
    min_delta = float(es["min_delta"])
    best_monitor = torch.full((K,), float("inf"), device=device)
    bad_cnt = torch.zeros((K,), dtype=torch.long, device=device)
    stopped = torch.zeros((K,), dtype=torch.bool, device=device)

    best_state = [
        {"model": None, "gate": None, "pred_residual": None, "dynamic_lambda": None, "learnable_lambda": None}
        for _ in range(K)
    ]
    best_epoch = torch.ones((K,), dtype=torch.long, device=device)
    train_mse_hist = []
    val_mse_hist = []
    epoch_times = []

    def save_best(k: int, epoch_idx: int):
        best_state[k]["model"] = model.get_cluster_state(k)
        best_state[k]["gate"] = gate.get_cluster_state(k)
        if pred_residual is not None:
            best_state[k]["pred_residual"] = pred_residual.get_cluster_state(k)
        if dynamic_lambda is not None:
            best_state[k]["dynamic_lambda"] = dynamic_lambda.get_cluster_state(k)
        if learnable_lambda is not None:
            best_state[k]["learnable_lambda"] = learnable_lambda.get_cluster_state(k)
        best_epoch[k] = epoch_idx

    def load_best_all():
        for k in range(K):
            if best_state[k]["model"] is not None:
                model.load_cluster_state(k, best_state[k]["model"])
                gate.load_cluster_state(k, best_state[k]["gate"])
                if pred_residual is not None and best_state[k]["pred_residual"] is not None:
                    pred_residual.load_cluster_state(k, best_state[k]["pred_residual"])
                if dynamic_lambda is not None and best_state[k]["dynamic_lambda"] is not None:
                    dynamic_lambda.load_cluster_state(k, best_state[k]["dynamic_lambda"])
                if learnable_lambda is not None and best_state[k]["learnable_lambda"] is not None:
                    learnable_lambda.load_cluster_state(k, best_state[k]["learnable_lambda"])

    @torch.no_grad()
    def average_lambda_kp(loader: DataLoader, base_lambda_kp: torch.Tensor) -> torch.Tensor:
        if len(loader) == 0:
            return base_lambda_kp
        if dynamic_lambda is None:
            return base_lambda_kp
        sum_lam = torch.zeros((K, P), device=device)
        cnt = 0
        model.eval()
        dynamic_lambda.eval()
        for x, _, _ in loader:
            x = x.to(device, non_blocking=True)
            feat_bcf = extract_gate_features(x)
            feat_bkf = scatter_mean_bcf_to_bkf(feat_bcf, cluster_id_c, K)
            series_bkl = scatter_mean_bcl_to_bkl(x, cluster_id_c, K)
            lam_bkp = _compute_lambda_bkp(
                base_lambda_kp=base_lambda_kp,
                feat_bkf=feat_bkf,
                series_bkl=series_bkl,
                dynamic_lambda=dynamic_lambda,
                lambda_min_kp=lambda_min_kp,
            )
            sum_lam += lam_bkp.sum(dim=0)
            cnt += lam_bkp.shape[0]
        if cnt == 0:
            return base_lambda_kp
        return sum_lam / float(cnt)

    @torch.no_grad()
    def collect_lambda_stats(loader: DataLoader, base_lambda_kp: torch.Tensor) -> Optional[Dict[str, torch.Tensor]]:
        if len(loader) == 0 or dynamic_lambda is None:
            return None
        sum_lam = torch.zeros((K, P), device=device)
        sum_sq_lam = torch.zeros((K, P), device=device)
        min_lam = torch.full((K, P), float("inf"), device=device)
        max_lam = torch.full((K, P), float("-inf"), device=device)
        cnt = 0
        model.eval()
        dynamic_lambda.eval()
        for x, _, _ in loader:
            x = x.to(device, non_blocking=True)
            feat_bcf = extract_gate_features(x)
            feat_bkf = scatter_mean_bcf_to_bkf(feat_bcf, cluster_id_c, K)
            series_bkl = scatter_mean_bcl_to_bkl(x, cluster_id_c, K)
            lam_bkp = _compute_lambda_bkp(
                base_lambda_kp=base_lambda_kp,
                feat_bkf=feat_bkf,
                series_bkl=series_bkl,
                dynamic_lambda=dynamic_lambda,
                lambda_min_kp=lambda_min_kp,
            )
            sum_lam += lam_bkp.sum(dim=0)
            sum_sq_lam += lam_bkp.pow(2).sum(dim=0)
            min_lam = torch.minimum(min_lam, lam_bkp.amin(dim=0))
            max_lam = torch.maximum(max_lam, lam_bkp.amax(dim=0))
            cnt += lam_bkp.shape[0]
        if cnt == 0:
            return None
        mean_lam = sum_lam / float(cnt)
        std_lam = (sum_sq_lam / float(cnt) - mean_lam.pow(2)).clamp_min(0.0).sqrt()
        return {
            "mean": mean_lam,
            "std": std_lam,
            "min": min_lam,
            "max": max_lam,
        }

    @torch.no_grad()
    def print_dynamic_lambda_summary(
        title: str,
        lambda_stats: Optional[Dict[str, torch.Tensor]],
        csv_path: str = None,
    ):
        if lambda_stats is None:
            return
        print(f"\nDynamic lambda summary ({title}):")
        rows = []
        mean_lam = lambda_stats["mean"].detach()
        std_lam = lambda_stats["std"].detach()
        min_lam = lambda_stats["min"].detach()
        max_lam = lambda_stats["max"].detach()
        for k in range(K):
            parts = []
            for p, name in enumerate(penalty_names):
                parts.append(
                    f"{name}(mean={float(mean_lam[k, p].item()):.6f}, "
                    f"std={float(std_lam[k, p].item()):.6f}, "
                    f"min={float(min_lam[k, p].item()):.6f}, "
                    f"max={float(max_lam[k, p].item()):.6f})"
                )
                rows.append({
                    "cluster_id": k,
                    "penalty": name,
                    "lambda_mean": float(mean_lam[k, p].item()),
                    "lambda_std": float(std_lam[k, p].item()),
                    "lambda_min": float(min_lam[k, p].item()),
                    "lambda_max": float(max_lam[k, p].item()),
                })
            print(f"  Cluster {k}: " + ", ".join(parts))
        if csv_path is not None:
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            print(f"Saved dynamic lambda statistics to: {csv_path}")

    @torch.no_grad()
    def print_cluster_penalty_summary(loader: DataLoader, title: str, lam_kp: torch.Tensor, csv_path: str = None):
        if (not moe_enable) or P == 0:
            print("\nPenalty summary: MoE disabled or no penalties.")
            return None
        if len(loader) == 0:
            print("\nPenalty summary: empty loader, skipped.")
            return None
        model.eval()
        gate.eval()

        sum_probs = torch.zeros(K, P, device=device)
        sum_skip_prob = torch.zeros(K, device=device)
        sum_skip_active = torch.zeros(K, device=device)
        cnt_k = torch.zeros(K, device=device)
        for x, y, _ in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yhat = model(x, cluster_id_c)
            pen_bkp = _router_penalty_context_from_history(
                x_bcl=x,
                yhat_base_bch=yhat,
                penalty_names=penalty_names,
                penalty_fns=penalty_fns,
                penalty_scale=penalty_scale,
                cluster_id_c=cluster_id_c,
                K=K,
            )
            feat_bcf = extract_gate_features(x)  # [B,C,F]
            feat_bkf = scatter_mean_bcf_to_bkf(feat_bcf, cluster_id_c, K)  # [B,K,F]
            _, probs_bkp, skip_bk, skip_prob_bk = gate(feat_bkf, straight_through=False)  # [B,K,P]
            probs_bkp = _apply_router_penalty_context(
                probs_bkp,
                pen_bkp,
                router_mode=router_mode,
                penalty_context_weight=router_penalty_context_weight,
                detach_penalty_context=router_detach_penalty_context,
            )
            sum_probs += probs_bkp.sum(dim=0)
            if allow_skip:
                sum_skip_prob += skip_prob_bk.sum(dim=0)
                sum_skip_active += skip_bk.sum(dim=0)
            cnt_k += probs_bkp.shape[0]

        avg_probs = sum_probs / cnt_k.clamp_min(1.0).view(K, 1)
        avg_skip_prob = sum_skip_prob / cnt_k.clamp_min(1.0)
        avg_skip_active = sum_skip_active / cnt_k.clamp_min(1.0)
        lam = lam_kp.detach()  # [K,P]
        print(f"\nPenalty summary ({title}):")
        rows = []
        for k in range(K):
            order = torch.argsort(avg_probs[k], descending=True)
            parts = []
            penalty_rank = 0
            if allow_skip:
                parts.append(
                    f"skip(active={float(avg_skip_active[k].item()):.3f}, p={float(avg_skip_prob[k].item()):.3f}, cost={skip_cost:.3f})"
                )
                rows.append({
                    "cluster_id": k,
                    "penalty": "skip",
                    "avg_prob": float(avg_skip_prob[k].item()),
                    "avg_lambda": 0.0,
                    "rank": 0,
                    "avg_skip_active": float(avg_skip_active[k].item()),
                    "skip_cost": skip_cost,
                })
            for idx in order.tolist():
                p = int(idx)
                penalty_rank += 1
                parts.append(
                    f"{penalty_names[p]}(lambda={float(lam[k, p].item()):.3f}, p={float(avg_probs[k, p].item()):.3f})"
                )
                rows.append({
                    "cluster_id": k,
                    "penalty": penalty_names[p],
                    "avg_prob": float(avg_probs[k, p].item()),
                    "avg_lambda": float(lam[k, p].item()),
                    "rank": penalty_rank,
                    "avg_skip_active": float(avg_skip_active[k].item()) if allow_skip else 0.0,
                    "skip_cost": skip_cost if allow_skip else 0.0,
                })
            print(f"  Cluster {k}: " + ", ".join(parts))
        if csv_path is not None:
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            print(f"Saved cluster penalty probabilities to: {csv_path}")
        return avg_probs.detach()

    @torch.no_grad()
    def collect_pred_residual_summary(loader: DataLoader) -> Dict[str, object]:
        cfg_summary = {
            "enabled": bool(pred_residual is not None),
            "specialization_weight": float(pred_residual_specialization_weight),
            "norm_weight": float(pred_residual_norm_weight),
            "intervention_weight": float(pred_residual_intervention_weight),
            "detach_routed_penalty_pred": bool(pred_residual_detach_routed_penalty_pred),
        }
        if pred_residual is None or P == 0 or len(loader) == 0:
            return cfg_summary
        cfg_summary["feature_mode"] = str(getattr(pred_residual, "feature_mode", "legacy"))
        cfg_summary["input_dim"] = int(getattr(pred_residual, "input_dim", 0))

        model.eval()
        gate.eval()
        pred_residual.eval()
        alpha_kp = pred_residual.alpha_values().detach()
        branch_sq_sum = 0.0
        branch_numel = 0
        delta_sq_sum = 0.0
        base_sq_sum = 0.0
        spec_sum_k = torch.zeros(K, device=device)
        norm_sum_k = torch.zeros(K, device=device)
        intervention_sum_k = torch.zeros(K, device=device)
        selected_intervention_sum_p = torch.zeros(P, device=device)
        route_sum_p = torch.zeros(P, device=device)
        effective_route_sum_p = torch.zeros(P, device=device)
        route_numel = 0
        cnt = 0

        for x, y, _ in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_base = model(x, cluster_id_c)
            route_pen_bkp = _router_penalty_context_from_history(
                x_bcl=x,
                yhat_base_bch=y_base,
                penalty_names=penalty_names,
                penalty_fns=penalty_fns,
                penalty_scale=penalty_scale,
                cluster_id_c=cluster_id_c,
                K=K,
            )
            feat_bcf = extract_gate_features(x)
            feat_bkf = scatter_mean_bcf_to_bkf(feat_bcf, cluster_id_c, K)
            mask_bkp, probs_bkp, skip_bk, _ = gate(feat_bkf, straight_through=False)
            probs_bkp = _apply_router_penalty_context(
                probs_bkp,
                route_pen_bkp,
                router_mode=router_mode,
                penalty_context_weight=router_penalty_context_weight,
                detach_penalty_context=router_detach_penalty_context,
            )
            if select_ranks is not None:
                mask_bkp = _select_rank_mask(probs_bkp, select_ranks, straight_through=False)
            pred_out = pred_residual(
                x,
                y_base,
                cluster_id_c,
                mask_bkp,
                skip_bk=skip_bk if allow_skip else None,
            )
            y_final = pred_out["y_final"]
            terms = _pred_residual_loss_terms(
                pred_out=pred_out,
                y_base=y_base,
                y_final=y_final,
                y=y,
                penalty_names=penalty_names,
                penalty_fns=penalty_fns,
                cluster_id_c=cluster_id_c,
                K=K,
                penalty_scale=penalty_scale,
                specialization_weight=1.0,
                norm_weight=1.0,
                intervention_weight=1.0,
            )
            spec_sum_k += terms["specialization_bk"].sum(dim=0)
            norm_sum_k += terms["norm_bk"].sum(dim=0)
            intervention_sum_k += terms["intervention_bk"].sum(dim=0)
            cnt += int(x.shape[0])
            branches = pred_out["branches"]
            branch_sq_sum += float(branches.pow(2).sum().item())
            branch_numel += int(branches.numel())
            delta = y_final - y_base
            delta_sq_sum += float(delta.pow(2).sum().item())
            base_sq_sum += float(y_base.pow(2).sum().item())
            route_bcp = pred_out["route_bcp"].detach()
            intervention_bcp = pred_out.get("intervention_bcp", torch.ones_like(route_bcp)).detach()
            effective_route_bcp = pred_out.get("effective_route_bcp", route_bcp * intervention_bcp).detach()
            route_sum_p += route_bcp.sum(dim=(0, 1))
            selected_intervention_sum_p += (route_bcp * intervention_bcp).sum(dim=(0, 1))
            effective_route_sum_p += effective_route_bcp.sum(dim=(0, 1))
            route_numel += int(route_bcp.shape[0] * route_bcp.shape[1])

        spec_k = spec_sum_k / max(cnt, 1)
        norm_k = norm_sum_k / max(cnt, 1)
        intervention_k = intervention_sum_k / max(cnt, 1)
        route_denom_p = route_sum_p.clamp_min(1.0e-8)
        selected_intervention_p = selected_intervention_sum_p / route_denom_p
        effective_route_p = effective_route_sum_p / max(route_numel, 1)
        cfg_summary.update(
            {
                "alpha_mean": float(alpha_kp.mean().item()),
                "alpha_by_penalty": {
                    penalty_names[p]: float(alpha_kp[:, p].mean().item()) for p in range(P)
                },
                "intervention_mean_selected": float(
                    (selected_intervention_sum_p.sum() / route_sum_p.sum().clamp_min(1.0e-8)).item()
                ),
                "intervention_by_penalty": {
                    penalty_names[p]: float(selected_intervention_p[p].item()) for p in range(P)
                },
                "effective_route_by_penalty": {
                    penalty_names[p]: float(effective_route_p[p].item()) for p in range(P)
                },
                "branch_rms": float((branch_sq_sum / max(branch_numel, 1)) ** 0.5),
                "residual_base_rms_ratio": float((delta_sq_sum / max(base_sq_sum, 1.0e-12)) ** 0.5),
                "specialization_loss": float(reduce_cluster_metric(spec_k, cluster_weight_k).item()),
                "norm_loss": float(reduce_cluster_metric(norm_k, cluster_weight_k).item()),
                "intervention_loss": float(reduce_cluster_metric(intervention_k, cluster_weight_k).item()),
            }
        )
        return cfg_summary

    def compute_batch_terms(
        x: torch.Tensor,
        y: torch.Tensor,
        base_lambda_kp: torch.Tensor,
        model_params: Optional[Dict[str, torch.Tensor]] = None,
        gate_params: Optional[Dict[str, torch.Tensor]] = None,
        pred_residual_params: Optional[Dict[str, torch.Tensor]] = None,
        dynamic_lambda_params: Optional[Dict[str, torch.Tensor]] = None,
        straight_through: bool = True,
        mae_objective_weight: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        yhat_base = _module_call(model, model_params, x, cluster_id_c)
        feat_bcf = extract_gate_features(x)
        feat_bkf = scatter_mean_bcf_to_bkf(feat_bcf, cluster_id_c, K)
        series_bkl = scatter_mean_bcl_to_bkl(x, cluster_id_c, K)
        probs_bkp = None
        skip_bk = None
        skip_prob_bk = None
        pred_out = None

        route_pen_bkp = _router_penalty_context_from_history(
            x_bcl=x,
            yhat_base_bch=yhat_base,
            penalty_names=penalty_names,
            penalty_fns=penalty_fns,
            penalty_scale=penalty_scale,
            cluster_id_c=cluster_id_c,
            K=K,
        )

        if moe_enable and P > 0:
            mask_bkp, probs_bkp, skip_bk, skip_prob_bk = _module_call(
                gate,
                gate_params,
                feat_bkf,
                straight_through=straight_through,
            )
            probs_bkp = _apply_router_penalty_context(
                probs_bkp,
                route_pen_bkp,
                router_mode=router_mode,
                penalty_context_weight=router_penalty_context_weight,
                detach_penalty_context=router_detach_penalty_context,
            )
            rank_mask = None
            if select_ranks is not None:
                mask_bkp = _select_rank_mask(probs_bkp, select_ranks, straight_through=straight_through)
                rank_mask = _select_rank_mask(probs_bkp, select_ranks, straight_through=False)
            if gate_soft_weight > 0.0:
                probs_sel = probs_bkp
                if rank_mask is not None:
                    probs_sel = probs_sel * rank_mask
                    probs_sel = probs_sel / probs_sel.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                target_mass = mask_bkp.detach().sum(dim=-1, keepdim=True).clamp_min(1.0)
                probs_sel = probs_sel * target_mass
                mask_bkp = (1.0 - gate_soft_weight) * mask_bkp + gate_soft_weight * probs_sel
        else:
            mask_bkp = torch.zeros_like(route_pen_bkp)

        if pred_residual is not None and moe_enable and P > 0:
            pred_out = _module_call(
                pred_residual,
                pred_residual_params,
                x,
                yhat_base,
                cluster_id_c,
                mask_bkp,
                skip_bk=skip_bk if allow_skip else None,
            )
            yhat = pred_out["y_final"]
        else:
            yhat = yhat_base

        err_bch = yhat - y
        abs_err_bch = err_bch.abs()
        mse_bc = err_bch.pow(2).mean(dim=-1)
        mae_bc = abs_err_bch.mean(dim=-1)
        mse_bk = scatter_mean_bc_to_bk(mse_bc, cluster_id_c, K)
        mae_bk = scatter_mean_bc_to_bk(mae_bc, cluster_id_c, K)
        if mae_objective_weight != 0.0:
            mae_objective_bc = _mae_objective_bc_from_abs(
                abs_err_bch,
                kind=mae_objective_kind,
                beta=mae_objective_beta,
            )
            mae_objective_bk = scatter_mean_bc_to_bk(mae_objective_bc, cluster_id_c, K)
        else:
            mae_objective_bk = torch.zeros_like(mse_bk)

        if P > 0:
            if pred_out is not None:
                yhat_for_penalty = yhat_base + (yhat - yhat_base).detach()
                if pred_residual_detach_routed_penalty_pred:
                    yhat_for_penalty = yhat_for_penalty.detach()
            else:
                yhat_for_penalty = yhat
            pen_bcp = []
            for name in penalty_names:
                pen_bcp.append(penalty_fns[name](yhat_for_penalty, y))
            pen_bcp = torch.stack(pen_bcp, dim=-1)
            pen_bcp = normalize_penalties(pen_bcp, scale=penalty_scale)
            pen_bkp = scatter_mean_bcf_to_bkf(pen_bcp, cluster_id_c, K)
        else:
            pen_bkp = route_pen_bkp

        if P > 0:
            lam_bkp = _compute_lambda_bkp(
                base_lambda_kp=base_lambda_kp,
                feat_bkf=feat_bkf,
                series_bkl=series_bkl,
                dynamic_lambda=dynamic_lambda,
                dynamic_lambda_params=dynamic_lambda_params,
                lambda_min_kp=lambda_min_kp,
            )
            penalty_loss_bk = _routed_penalty_loss(
                mask_bkp=mask_bkp,
                lam_bkp=lam_bkp,
                pen_bkp=pen_bkp,
                gate_route_on_penalty_only=gate_route_on_penalty_only,
            )
            penalty_loss_bk = _apply_skip_to_penalty_loss(
                penalty_loss_bk,
                skip_bk=skip_bk if allow_skip else None,
                skip_cost=skip_cost,
            )
        else:
            lam_bkp = pen_bkp
            penalty_loss_bk = torch.zeros_like(mse_bk)

        objective_loss_bk = (mse_weight * mse_bk) + (float(mae_objective_weight) * mae_objective_bk) + penalty_loss_bk
        pred_loss_terms = _pred_residual_loss_terms(
            pred_out=pred_out,
            y_base=yhat_base,
            y_final=yhat,
            y=y,
            penalty_names=penalty_names,
            penalty_fns=penalty_fns,
            cluster_id_c=cluster_id_c,
            K=K,
            penalty_scale=penalty_scale,
            specialization_weight=pred_residual_specialization_weight,
            norm_weight=pred_residual_norm_weight,
            intervention_weight=pred_residual_intervention_weight,
        )
        objective_loss_bk = objective_loss_bk + pred_loss_terms["total_bk"]
        return {
            "mse_bk": mse_bk,
            "mae_bk": mae_bk,
            "mae_objective_bk": mae_objective_bk,
            "objective_loss_bk": objective_loss_bk,
            "pen_bkp": pen_bkp,
            "mask_bkp": mask_bkp,
            "probs_bkp": probs_bkp,
            "lam_bkp": lam_bkp,
            "skip_bk": skip_bk,
            "skip_prob_bk": skip_prob_bk,
        }

    outer_train_state = [None]
    outer_val_state = [None]

    def next_outer_batch(loader: DataLoader, iterator_state):
        iterator = iterator_state[0]
        if iterator is None:
            iterator = iter(loader)
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        iterator_state[0] = iterator
        return batch

    inner_named = []
    inner_modules = [("model", model)]
    if not (bilevel_enable and bilevel_optimize_gate):
        inner_modules.append(("gate", gate))
    if pred_residual is not None:
        inner_modules.append(("pred_residual", pred_residual))
    for prefix, module in inner_modules:
        for name, param in module.named_parameters():
            inner_named.append((prefix, name, param))

    def bilevel_outer_step(epoch_idx: int, warmup_scale: float) -> Optional[float]:
        if (not bilevel_enable) or lambda_optimizer is None or stopped.all():
            return None

        train_batch = next_outer_batch(dl_tr, outer_train_state)
        val_batch = next_outer_batch(dl_va, outer_val_state)
        x_tr, y_tr, _ = train_batch
        x_va, y_va, _ = val_batch
        x_tr = x_tr.to(device, non_blocking=True)
        y_tr = y_tr.to(device, non_blocking=True)
        x_va = x_va.to(device, non_blocking=True)
        y_va = y_va.to(device, non_blocking=True)

        base_lambda_kp = lambda_kp_at(epoch_idx, detach=False) * warmup_scale
        train_terms = compute_batch_terms(
            x_tr, y_tr,
            base_lambda_kp=base_lambda_kp,
            straight_through=(not bool(moe_cfg["detach_penalty_grad"])),
            mae_objective_weight=mae_objective_weight_at(epoch_idx),
        )
        inner_loss_bk = train_terms["objective_loss_bk"]
        if moe_enable and P > 0 and (gate_entropy_weight != 0.0 or gate_balance_weight != 0.0):
            inner_loss_bk = inner_loss_bk + _gate_regularization(
                train_terms["probs_bkp"],
                gate_entropy_weight=gate_entropy_weight,
                gate_balance_weight=gate_balance_weight,
                gate_entropy_target_frac=gate_entropy_target_frac,
                gate_balance_target_kp=gate_balance_target_kp,
            )
        inner_loss = reduce_cluster_metric(inner_loss_bk, cluster_weight_k).mean()

        inner_params = [param for _, _, param in inner_named]
        inner_grads = torch.autograd.grad(
            inner_loss,
            inner_params,
            create_graph=True,
            allow_unused=True,
        )
        fast_model_params = {}
        fast_gate_params = {}
        fast_pred_residual_params = {}
        for (prefix, name, param), grad in zip(inner_named, inner_grads):
            fast_param = param if grad is None else (param - bilevel_inner_lr * grad)
            if prefix == "model":
                fast_model_params[name] = fast_param
            elif prefix == "gate":
                fast_gate_params[name] = fast_param
            else:
                fast_pred_residual_params[name] = fast_param

        model_was_training = model.training
        gate_was_training = gate.training
        pred_residual_was_training = pred_residual.training if pred_residual is not None else False
        dyn_was_training = dynamic_lambda.training if dynamic_lambda is not None else False
        model.eval()
        gate.eval()
        if pred_residual is not None:
            pred_residual.eval()
        if dynamic_lambda is not None:
            dynamic_lambda.eval()
        val_terms = compute_batch_terms(
            x_va, y_va,
            base_lambda_kp=base_lambda_kp,
            model_params=fast_model_params,
            gate_params=fast_gate_params if len(fast_gate_params) > 0 else None,
            pred_residual_params=fast_pred_residual_params if len(fast_pred_residual_params) > 0 else None,
            straight_through=False,
            mae_objective_weight=mae_objective_weight_at(epoch_idx),
        )
        if model_was_training:
            model.train()
        if gate_was_training:
            gate.train()
        if pred_residual is not None and pred_residual_was_training:
            pred_residual.train()
        if dynamic_lambda is not None and dyn_was_training:
            dynamic_lambda.train()
        outer_metric_bk = val_terms["mse_bk"]
        outer_loss = reduce_cluster_metric(outer_metric_bk, cluster_weight_k).mean()
        if learnable_lambda is not None and learnable_lambda_reg_weight > 0.0:
            outer_loss = outer_loss + learnable_lambda_reg_weight * reduce_cluster_metric(
                learnable_lambda.regularization(), cluster_weight_k
            )
        if dynamic_lambda is not None and dynamic_lambda_reg_weight > 0.0 and P > 0:
            base_lam = base_lambda_kp.unsqueeze(0).expand(x_va.shape[0], K, P).clamp_min(1.0e-8)
            scale_bkp = val_terms["lam_bkp"] / base_lam
            outer_loss = outer_loss + dynamic_lambda_reg_weight * scale_bkp.log().pow(2).mean()

        lambda_optimizer.zero_grad(set_to_none=True)
        outer_loss.backward()
        if grad_clip > 0:
            lambda_params = []
            if dynamic_lambda is not None:
                lambda_params.extend(list(dynamic_lambda.parameters()))
            if learnable_lambda is not None:
                lambda_params.extend(list(learnable_lambda.parameters()))
            if len(lambda_params) > 0:
                torch.nn.utils.clip_grad_norm_(lambda_params, grad_clip)
        if dynamic_lambda is not None:
            dynamic_lambda.mask_cluster_grads(stopped)
        if learnable_lambda is not None:
            learnable_lambda.mask_cluster_grads(stopped)
        lambda_optimizer.step()
        return float(outer_loss.item())
    # keep console output minimal during training

    # training
    grad_clip = float(cfg["train"]["grad_clip"])
    steps_per_epoch = max(len(dl_tr), 1)
    train_label = f"Train {os.path.splitext(os.path.basename(cfg['data']['csv_path']))[0]} H={H}"
    train_progress = PurpleProgressBar(
        total=max(int(epochs) * steps_per_epoch, 1),
        label=train_label,
        unit="batch",
    )
    early_stopped = False

    for ep in range(1, epochs + 1):
        t_ep0 = time.perf_counter()
        if penalty_warmup_epochs > 0:
            warmup_scale = min(1.0, float(ep) / float(penalty_warmup_epochs))
        else:
            warmup_scale = 1.0
        model.train()
        gate.train()
        if pred_residual is not None:
            pred_residual.train()
        if dynamic_lambda is not None:
            dynamic_lambda.train()
        running = 0.0
        n_batches = 0
        act_sum = torch.zeros(P, device=device)
        active_cnt = 0
        k_active_sum = 0.0
        train_loss_sum_k = torch.zeros(K, device=device)
        train_mse_sum_k = torch.zeros(K, device=device)
        train_mae_sum_k = torch.zeros(K, device=device)
        train_cnt = 0
        act_sum_kp = torch.zeros(K, P, device=device)
        active_cnt_k = torch.zeros(K, device=device)
        rank_counts = None
        rank_total = 0
        if moe_enable and P > 0:
            rank_counts = torch.zeros(P, P, device=device)

        for x, y, idx in dl_tr:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            idx = idx.to(torch.long)
            if cluster_memory_bank is not None:
                train_window = torch.cat([x, y], dim=-1)
                cluster_memory_bank.update(train_window, idx, cluster_id_c)

            yhat_base = model(x, cluster_id_c)
            feat_bcf = extract_gate_features(x)  # [B,C,F]
            feat_bkf = scatter_mean_bcf_to_bkf(feat_bcf, cluster_id_c, K)  # [B,K,F]
            series_bkl = scatter_mean_bcl_to_bkl(x, cluster_id_c, K)  # [B,K,L]
            skip_bk = None
            pred_out = None
            route_pen_bkp = _router_penalty_context_from_history(
                x_bcl=x,
                yhat_base_bch=yhat_base,
                penalty_names=penalty_names,
                penalty_fns=penalty_fns,
                penalty_scale=penalty_scale,
                cluster_id_c=cluster_id_c,
                K=K,
            )

            if moe_enable and P > 0:
                straight_through = (not bool(moe_cfg["detach_penalty_grad"])) and (not (bilevel_enable and bilevel_optimize_gate))
                mask_bkp, probs_bkp, skip_bk, skip_prob_bk = gate(feat_bkf, straight_through=straight_through)
                probs_bkp = _apply_router_penalty_context(
                    probs_bkp,
                    route_pen_bkp,
                    router_mode=router_mode,
                    penalty_context_weight=router_penalty_context_weight,
                    detach_penalty_context=router_detach_penalty_context,
                )
                rank_mask = None
                if select_ranks is not None:
                    mask_bkp = _select_rank_mask(probs_bkp, select_ranks, straight_through=straight_through)
                    rank_mask = _select_rank_mask(probs_bkp, select_ranks, straight_through=False)
                if gate_soft_weight > 0.0:
                    probs_sel = probs_bkp
                    if rank_mask is not None:
                        probs_sel = probs_sel * rank_mask
                        probs_sel = probs_sel / probs_sel.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    target_mass = mask_bkp.detach().sum(dim=-1, keepdim=True).clamp_min(1.0)
                    probs_sel = probs_sel * target_mass
                    mask_bkp = (1.0 - gate_soft_weight) * mask_bkp + gate_soft_weight * probs_sel
                with torch.no_grad():
                    act_sum += mask_bkp.sum(dim=(0, 1))
                    active_cnt += int(mask_bkp.shape[0] * mask_bkp.shape[1])
                    k_active_sum += float(mask_bkp.sum().item())
                    act_sum_kp += mask_bkp.sum(dim=0)
                    active_cnt_k += mask_bkp.shape[0]
                if rank_counts is not None:
                    with torch.no_grad():
                        order = torch.argsort(probs_bkp.detach(), dim=-1, descending=True)
                        for r in range(P):
                            pen_idx = order[..., r].reshape(-1)
                            cnt = torch.bincount(pen_idx, minlength=P)
                            rank_counts[:, r] += cnt
                        rank_total += int(order.shape[0] * order.shape[1])
            else:
                mask_bkp = torch.zeros_like(route_pen_bkp)

            if pred_residual is not None and moe_enable and P > 0:
                pred_out = pred_residual(
                    x,
                    yhat_base,
                    cluster_id_c,
                    mask_bkp,
                    skip_bk=skip_bk if allow_skip else None,
                )
                yhat = pred_out["y_final"]
            else:
                yhat = yhat_base

            err_bch = yhat - y
            abs_err_bch = err_bch.abs()
            mse_bc = err_bch.pow(2).mean(dim=-1)  # [B,C]
            mae_bc = abs_err_bch.mean(dim=-1)  # [B,C]
            mse_bk = scatter_mean_bc_to_bk(mse_bc, cluster_id_c, K)  # [B,K]
            mae_bk = scatter_mean_bc_to_bk(mae_bc, cluster_id_c, K)  # [B,K]
            mae_objective_weight_ep = mae_objective_weight_at(ep)
            if mae_objective_weight_ep != 0.0:
                mae_objective_bc = _mae_objective_bc_from_abs(
                    abs_err_bch,
                    kind=mae_objective_kind,
                    beta=mae_objective_beta,
                )
                mae_objective_bk = scatter_mean_bc_to_bk(mae_objective_bc, cluster_id_c, K)
            else:
                mae_objective_bk = torch.zeros_like(mse_bk)

            if P > 0:
                if pred_out is not None:
                    yhat_for_penalty = yhat_base + (yhat - yhat_base).detach()
                    if pred_residual_detach_routed_penalty_pred:
                        yhat_for_penalty = yhat_for_penalty.detach()
                else:
                    yhat_for_penalty = yhat
                pen_bcp = []
                for name in penalty_names:
                    pen_bc = penalty_fns[name](yhat_for_penalty, y)  # [B,C]
                    pen_bcp.append(pen_bc)
                pen_bcp = torch.stack(pen_bcp, dim=-1)  # [B,C,P]
                pen_bcp = normalize_penalties(pen_bcp, scale=penalty_scale)
                pen_bkp = scatter_mean_bcf_to_bkf(pen_bcp, cluster_id_c, K)  # [B,K,P]
            else:
                pen_bkp = route_pen_bkp

            if P > 0:
                base_lambda_kp = lambda_kp_at(ep, detach=bilevel_enable) * warmup_scale
                dynamic_lambda_params = _named_param_dict(dynamic_lambda, detach=True) if (bilevel_enable and dynamic_lambda is not None) else None
                lam = _compute_lambda_bkp(
                    base_lambda_kp=base_lambda_kp,
                    feat_bkf=feat_bkf,
                    series_bkl=series_bkl,
                    dynamic_lambda=dynamic_lambda,
                    dynamic_lambda_params=dynamic_lambda_params,
                    lambda_min_kp=lambda_min_kp,
                )
                penalty_loss_bk = _routed_penalty_loss(
                    mask_bkp=mask_bkp,
                    lam_bkp=lam,
                    pen_bkp=pen_bkp,
                    gate_route_on_penalty_only=gate_route_on_penalty_only,
                )
                penalty_loss_bk = _apply_skip_to_penalty_loss(
                    penalty_loss_bk,
                    skip_bk=skip_bk if allow_skip else None,
                    skip_cost=skip_cost,
                )
                objective_loss_bk = (
                    (mse_weight * mse_bk)
                    + (float(mae_objective_weight_ep) * mae_objective_bk)
                    + penalty_loss_bk
                )  # [B,K]
                pred_loss_terms = _pred_residual_loss_terms(
                    pred_out=pred_out,
                    y_base=yhat_base,
                    y_final=yhat,
                    y=y,
                    penalty_names=penalty_names,
                    penalty_fns=penalty_fns,
                    cluster_id_c=cluster_id_c,
                    K=K,
                    penalty_scale=penalty_scale,
                    specialization_weight=pred_residual_specialization_weight,
                    norm_weight=pred_residual_norm_weight,
                    intervention_weight=pred_residual_intervention_weight,
                )
                loss_bk = objective_loss_bk + pred_loss_terms["total_bk"]
                if (not bilevel_enable) and learnable_lambda is not None and learnable_lambda_reg_weight > 0.0:
                    loss_bk = loss_bk + learnable_lambda_reg_weight * learnable_lambda.regularization().unsqueeze(0)
                if (not bilevel_enable) and dynamic_lambda is not None and dynamic_lambda_reg_weight > 0.0:
                    base_lam = base_lambda_kp.unsqueeze(0).expand(x.shape[0], K, P).clamp_min(1.0e-8)
                    scale_bkp = lam / base_lam
                    loss_bk = loss_bk + dynamic_lambda_reg_weight * scale_bkp.log().pow(2).mean(dim=-1)
                if moe_enable and (gate_entropy_weight != 0.0 or gate_balance_weight != 0.0):
                    loss_bk = loss_bk + _gate_regularization(
                        probs_bkp,
                        gate_entropy_weight=gate_entropy_weight,
                        gate_balance_weight=gate_balance_weight,
                        gate_entropy_target_frac=gate_entropy_target_frac,
                        gate_balance_target_kp=gate_balance_target_kp,
                    )
            else:
                objective_loss_bk = (mse_weight * mse_bk) + (float(mae_objective_weight_ep) * mae_objective_bk)
                loss_bk = objective_loss_bk
            train_loss_sum_k += objective_loss_bk.sum(dim=0)
            train_mse_sum_k += mse_bk.sum(dim=0)
            train_mae_sum_k += mae_bk.sum(dim=0)
            train_cnt += int(loss_bk.shape[0])
            loss = reduce_cluster_metric(loss_bk, cluster_weight_k).mean()

            for opt_k in optimizers:
                opt_k.zero_grad(set_to_none=True)
            loss.backward()

            if grad_clip > 0:
                for k, params_k in enumerate(cluster_params):
                    if stopped[k]:
                        continue
                    torch.nn.utils.clip_grad_norm_(params_k, grad_clip)

            model.mask_cluster_grads(stopped)
            if moe_enable:
                gate.mask_cluster_grads(stopped)
            if pred_residual is not None:
                pred_residual.mask_cluster_grads(stopped)
            if dynamic_lambda is not None:
                dynamic_lambda.mask_cluster_grads(stopped)
            if learnable_lambda is not None:
                learnable_lambda.mask_cluster_grads(stopped)
            for k, opt_k in enumerate(optimizers):
                if stopped[k]:
                    continue
                opt_k.step()

            running += float(loss.item())
            n_batches += 1
            if train_progress.enabled:
                step_now = (ep - 1) * steps_per_epoch + min(n_batches, steps_per_epoch)
                train_progress.update(
                    step_now,
                    suffix=(
                        f"epoch={ep}/{epochs} batch={n_batches}/{steps_per_epoch} "
                        f"loss={running / max(n_batches, 1):.6f}"
                    ),
                )

        outer_loss_epoch = None
        if bilevel_enable:
            outer_vals = []
            for _ in range(bilevel_steps_per_epoch):
                outer_val = bilevel_outer_step(ep, warmup_scale)
                if outer_val is not None:
                    outer_vals.append(outer_val)
            if len(outer_vals) > 0:
                outer_loss_epoch = float(sum(outer_vals) / len(outer_vals))

        if train_progress.enabled:
            train_progress.update(
                ep * steps_per_epoch,
                suffix=f"epoch={ep}/{epochs} loss={running / max(n_batches, 1):.6f} validating",
                force=True,
            )
        val_loss_k, val_mse_k, val_mae_k, _, _, _, _, _ = eval_loop(
            model, gate, lambda_kp_at(ep, detach=True),
            penalty_names, penalty_fns,
            dl_va, cluster_id_c, K, moe_cfg, device,
            select_ranks=select_ranks,
            collect_plot=False, channel_count=C,
            mse_weight=mse_weight,
            gate_entropy_weight=gate_entropy_weight,
            gate_balance_weight=gate_balance_weight,
            gate_soft_weight=gate_soft_weight,
            gate_entropy_target_frac=gate_entropy_target_frac,
            penalty_scale=penalty_scale,
            dynamic_lambda=dynamic_lambda,
            lambda_min_kp=lambda_min_kp,
            mae_objective_weight=mae_objective_weight_at(ep),
            mae_objective_kind=mae_objective_kind,
            mae_objective_beta=mae_objective_beta,
            pred_residual=pred_residual,
            knn_hybrid=knn_hybrid_val if knn_cfg.use_for_model_selection else None,
            eval_start=val_eval_start,
        )
        train_loss_k = train_loss_sum_k / max(train_cnt, 1)
        train_mse_k = train_mse_sum_k / max(train_cnt, 1)
        train_mae_k = train_mae_sum_k / max(train_cnt, 1)
        train_mse_hist.append(train_mse_k.detach().cpu())
        val_mse_hist.append(val_mse_k.detach().cpu())

        monitor_k = _select_monitor_k(train_loss_k, train_mse_k, train_mae_k, val_loss_k, val_mse_k, val_mae_k)
        selection_active = ep >= selection_start_epoch
        improved = (best_monitor - monitor_k) > min_delta if selection_active else torch.zeros_like(stopped)
        for k in range(K):
            if stopped[k]:
                continue
            if improved[k]:
                best_monitor[k] = monitor_k[k]
                bad_cnt[k] = 0
                save_best(k, ep)
            else:
                if ep < early_stop_start_epoch or not selection_active:
                    continue
                bad_cnt[k] += 1
                if bad_cnt[k] >= patience:
                    stopped[k] = True

        if schedulers is not None:
            for k, sched in enumerate(schedulers):
                if stopped[k]:
                    continue
                if sched_name in {"plateau", "reduce", "reduce_on_plateau"}:
                    sched.step(float(monitor_k[k].item()))
                else:
                    sched.step()
        train_loss_agg = float(reduce_cluster_metric(train_loss_k, cluster_weight_k).item())
        val_loss_agg = float(reduce_cluster_metric(val_loss_k, cluster_weight_k).item())
        progress_suffix = (
            f"epoch={ep}/{epochs} loss={train_loss_agg:.6f} val_loss={val_loss_agg:.6f}"
        )
        if outer_loss_epoch is not None:
            progress_suffix += f" lambda_loss={outer_loss_epoch:.6f}"
        if train_progress.enabled:
            train_progress.update(ep * steps_per_epoch, suffix=progress_suffix, force=True)
        else:
            msg = (
                f"[Epoch {ep:03d}] loss={train_loss_agg:.6f} | "
                f"val_loss={val_loss_agg:.6f}"
            )
            if outer_loss_epoch is not None:
                msg += f" | lambda_loss={outer_loss_epoch:.6f}"
            print(msg)

        epoch_times.append(time.perf_counter() - t_ep0)
        if stopped.all():
            early_stopped = True
            if not train_progress.enabled:
                print("All clusters early-stopped.")
            break

    train_progress.finish(
        current=min(len(epoch_times) * steps_per_epoch, train_progress.total),
        suffix="early stopped" if early_stopped else "done",
    )

    plot_cfg = cfg.get("plot", {}) or {}
    if bool(plot_cfg.get("save_loss_curves", False)):
        loss_dir = os.path.join(out_dir, "loss_curves")
        save_cluster_metric_curves(
            out_dir=loss_dir,
            train_metric_hist=train_mse_hist,
            val_metric_hist=val_mse_hist,
            metric_name="mse",
            dpi=int(plot_cfg.get("dpi", 140)),
        )
        print(f"Saved MSE curves to: {loss_dir}")

    load_best_all()
    if knn_cfg.enable and knn_requires_base_bank:
        _refresh_knn_hybrids()

    if memory_enable:
        if cluster_memory_bank is not None and cluster_memory_bank.total_updates > 0:
            prototypes_kt = cluster_memory_bank.finalize()
            memory_meta = {
                "kind": "online_train_memory",
                "source_split": "train",
                "memory_len": int(t_train),
                "input_len": L,
                "pred_len": H,
                "num_window_updates": int(cluster_memory_bank.total_updates),
            }
        else:
            prototypes_kt = compute_cluster_prototypes(data_tc[:t_train], cluster_id_c)
            memory_meta = {
                "kind": "train_segment_prototype_fallback",
                "source_split": "train",
                "memory_len": int(t_train),
                "input_len": L,
                "pred_len": H,
                "num_window_updates": 0,
            }
        save_cluster_memory(memory_path, prototypes_kt, cluster_id_c, channel_names, meta=memory_meta)
        print(f"Saved cluster memory to: {memory_path}")

    if bool(memory_cfg.get("save_checkpoint", False)):
        ckpt_path = str(memory_cfg.get("checkpoint_path", os.path.join(out_dir, "best_checkpoint.pt")))
        meta = {
            "K": K,
            "input_len": L,
            "pred_len": H,
            "num_channels": C,
            "cluster_id_c": cluster_id_c.detach().cpu(),
            "model_cfg": dict(model_cfg),
            "moe_cfg": dict(moe_cfg),
            "gate_feat_dim": gate_feat_dim,
            "gate_feature_names": list(GATE_FEATURE_NAMES),
            "penalty_names": list(penalty_names),
            "best_epoch": best_epoch.detach().cpu(),
        }
        pred_residual_state = None if pred_residual is None else pred_residual.state_dict()
        dynamic_lambda_state = None if dynamic_lambda is None else dynamic_lambda.state_dict()
        learnable_lambda_state = None if learnable_lambda is None else learnable_lambda.state_dict()
        save_cluster_checkpoint(
            ckpt_path,
            model.state_dict(),
            gate.state_dict(),
            meta,
            pred_residual_state=pred_residual_state,
            dynamic_lambda_state=dynamic_lambda_state,
            learnable_lambda_state=learnable_lambda_state,
        )
        print(f"Saved best checkpoint to: {ckpt_path}")
    if knn_hybrid_test is not None and knn_bank_path is not None:
        knn_meta = {
            "kind": "shape_knn_bank",
            "source_split": str(knn_cfg.bank_split),
            "input_len": L,
            "pred_len": H,
            "num_train_windows": int(len(xtr)),
            "num_bank_windows": int(knn_test_bank_windows),
            "rolling_no_future": bool(knn_cfg.mode == "rolling"),
            "past_context": bool(past_context),
        }
        save_shape_knn_bank(
            knn_bank_path,
            knn_hybrid_test,
            cluster_id_c=cluster_id_c,
            channel_names=channel_names,
            meta=knn_meta,
        )
        print(f"Saved KNN hybrid bank to: {knn_bank_path}")

    # print per-cluster penalty selection after training
    summary_loader = dl_va if len(dva) > 0 else dl_tr
    summary_name = "val" if len(dva) > 0 else "train"
    lam_kp_best = lambda_kp_from_epochs(best_epoch)
    lam_kp_summary = average_lambda_kp(summary_loader, lam_kp_best)
    lambda_stats = collect_lambda_stats(summary_loader, lam_kp_best)
    summary_csv_path = os.path.join(out_dir, "cluster_penalty_probs.csv")
    avg_probs_summary = print_cluster_penalty_summary(summary_loader, title=summary_name, lam_kp=lam_kp_summary, csv_path=summary_csv_path)
    lambda_stats_csv_path = os.path.join(out_dir, "cluster_lambda_stats.csv")
    print_dynamic_lambda_summary(summary_name, lambda_stats, csv_path=lambda_stats_csv_path)
    moe_residual_summary = collect_pred_residual_summary(summary_loader)
    if bool(portrait_cfg.get("enable", False)) and (avg_probs_summary is not None) and len(penalty_names) > 0:
        portrait_dir = portrait_cfg.get("out_dir", os.path.join(out_dir, "cluster_portraits"))
        portrait_dpi = int(portrait_cfg.get("dpi", 140))
        max_points = int(portrait_cfg.get("max_points", 2000))
        jump_thr = float(portrait_cfg.get("jump_threshold", cfg.get("penalties", {}).get("jump_threshold", 2.0)))
        paths = save_cluster_portraits(
            out_dir=portrait_dir,
            data_tc=data_tc,
            cluster_id_c=cluster_id_c,
            jump_thr=jump_thr,
            dpi=portrait_dpi,
            max_points=max_points,
            metric_names=penalty_names,
            metric_values_km=avg_probs_summary,
            portrait_title="expert selection portrait (p)",
            metric_scale_mode="raw_0_1",
        )
        print(f"Updated cluster portraits with expert selection radar: {paths['dir']}")
    plot_cfg = cfg.get("plot", {}) or {}
    plot_enable = bool(plot_cfg.get("enable", False))
    random_n = int(plot_cfg.get("random_n", 0))
    plot_idx = None
    if plot_enable and len(dte) > 0 and random_n > 0:
        rng = np.random.default_rng(int(cfg["exp"]["seed"]))
        idxs = rng.choice(len(dte), size=min(random_n, len(dte)), replace=False)
        plot_idx = torch.tensor(sorted([int(i) for i in idxs]), device=device, dtype=torch.long)

    val_summary = None
    val_summary_hybrid = None
    val_hybrid_confidence = None
    val_mse_c_base = None
    val_mae_c_base = None
    val_mse_c_hybrid = None
    val_mae_c_hybrid = None
    knn_fusion_scale_ch = None
    knn_fusion_summary = None
    knn_gate_model = None
    knn_gate_summary = None
    residual_correction_base_ch = None
    residual_correction_hybrid_ch = None
    pred_residual_channel_scale_c = None
    pred_residual_gate_model = None
    pred_residual_gate_summary = None
    pred_residual_selection_summary = None
    calibration_summary = {
        "enable": bool(calibration_enable),
        "method": str(calibration_method),
        "shrink": float(calibration_shrink),
        "max_abs": float(calibration_max_abs),
        "base_mean_abs": None,
        "hybrid_mean_abs": None,
    }
    mae_eval_weight = mae_objective_weight_final if mae_objective_enable else 0.0
    if skip_test:
        print("eval.skip_test=true: test split windows, evaluation, and metrics are disabled.")
    if len(dva) > 0:
        val_loader_summary = DataLoader(dva, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)
        if calibration_enable:
            residual_correction_base_ch = estimate_residual_correction(
                model,
                val_loader_summary,
                cluster_id_c,
                device,
                method=calibration_method,
                shrink=calibration_shrink,
                max_abs=calibration_max_abs,
                knn_hybrid=None,
                eval_start=val_eval_start,
            )
            if residual_correction_base_ch is not None:
                calibration_summary["base_mean_abs"] = float(residual_correction_base_ch.abs().mean().item())
            if knn_hybrid_val is not None:
                knn_hybrid_val.reset_confidence_stats()
                residual_correction_hybrid_ch = estimate_residual_correction(
                    model,
                    val_loader_summary,
                    cluster_id_c,
                    device,
                    method=calibration_method,
                    shrink=calibration_shrink,
                    max_abs=calibration_max_abs,
                    knn_hybrid=knn_hybrid_val,
                    eval_start=val_eval_start,
                )
                if residual_correction_hybrid_ch is not None:
                    calibration_summary["hybrid_mean_abs"] = float(residual_correction_hybrid_ch.abs().mean().item())
                knn_hybrid_val.reset_confidence_stats()
        val_loss_best_k, val_mse_best_k, val_mae_best_k, val_mse_c_base, val_mae_c_base, _, _, _ = eval_loop(
            model, gate, lam_kp_best,
            penalty_names, penalty_fns,
            val_loader_summary, cluster_id_c, K, moe_cfg, device,
            select_ranks=select_ranks,
            collect_plot=False, channel_count=C,
            mse_weight=mse_weight,
            gate_entropy_weight=gate_entropy_weight,
            gate_balance_weight=gate_balance_weight,
            gate_soft_weight=gate_soft_weight,
            gate_entropy_target_frac=gate_entropy_target_frac,
            penalty_scale=penalty_scale,
            dynamic_lambda=dynamic_lambda,
            lambda_min_kp=lambda_min_kp,
            mae_objective_weight=mae_eval_weight,
            mae_objective_kind=mae_objective_kind,
            mae_objective_beta=mae_objective_beta,
            pred_residual=pred_residual,
            residual_correction_ch=residual_correction_base_ch,
            knn_hybrid=None,
            eval_start=val_eval_start,
        )
        val_summary = {
            "avg_loss": float(reduce_cluster_metric(val_loss_best_k, cluster_weight_k).item()),
            "avg_mse": float(reduce_cluster_metric(val_mse_best_k, cluster_weight_k).item()),
            "avg_mae": float(reduce_cluster_metric(val_mae_best_k, cluster_weight_k).item()),
            "per_cluster_loss": [float(v) for v in val_loss_best_k.detach().cpu().tolist()],
            "per_cluster_mse": [float(v) for v in val_mse_best_k.detach().cpu().tolist()],
            "per_cluster_mae": [float(v) for v in val_mae_best_k.detach().cpu().tolist()],
        }
        residual_selection_policy = str(pred_residual_cfg.get("selection_policy", "none")).lower()
        if residual_selection_policy in {"false", "off", "disable", "disabled"}:
            residual_selection_policy = "none"
        if residual_selection_policy not in {"none", "val_mse_channel", "val_mse_scale", "val_mse_gate", "val_mse_gate_guarded"}:
            raise ValueError(
                "Unsupported moe.pred_side_residual.selection_policy="
                f"'{residual_selection_policy}'. Expected none, val_mse_channel, val_mse_scale, "
                "val_mse_gate, or val_mse_gate_guarded."
            )
        if pred_residual is not None and residual_selection_policy in {"val_mse_channel", "val_mse_scale", "val_mse_gate", "val_mse_gate_guarded"}:
            zero_residual_scale_c = torch.zeros(C, device=device, dtype=torch.float32)
            residual_scale_mean_value = 0.0
            (
                val_loss_pred_base_k,
                val_mse_pred_base_k,
                val_mae_pred_base_k,
                val_mse_c_pred_base,
                val_mae_c_pred_base,
                _,
                _,
                _,
            ) = eval_loop(
                model, gate, lam_kp_best,
                penalty_names, penalty_fns,
                val_loader_summary, cluster_id_c, K, moe_cfg, device,
                select_ranks=select_ranks,
                collect_plot=False, channel_count=C,
                mse_weight=mse_weight,
                gate_entropy_weight=gate_entropy_weight,
                gate_balance_weight=gate_balance_weight,
                gate_soft_weight=gate_soft_weight,
                gate_entropy_target_frac=gate_entropy_target_frac,
                penalty_scale=penalty_scale,
                dynamic_lambda=dynamic_lambda,
                lambda_min_kp=lambda_min_kp,
                mae_objective_weight=mae_eval_weight,
                mae_objective_kind=mae_objective_kind,
                mae_objective_beta=mae_objective_beta,
                pred_residual=pred_residual,
                pred_residual_scale_c=zero_residual_scale_c,
                residual_correction_ch=residual_correction_base_ch,
                knn_hybrid=None,
                eval_start=val_eval_start,
            )
            val_scaled_mse_c = val_mse_c_base
            val_scaled_mae_c = val_mae_c_base
            if residual_selection_policy == "val_mse_scale":
                scale_min = float(pred_residual_cfg.get("selection_scale_min", 0.0))
                scale_max = float(pred_residual_cfg.get("selection_scale_max", 1.0))
                scale_steps = int(pred_residual_cfg.get("selection_scale_steps", 21))
                if scale_steps < 2:
                    raise ValueError("moe.pred_side_residual.selection_scale_steps must be >= 2")
                scale_grid = torch.linspace(scale_min, scale_max, scale_steps, device=device, dtype=torch.float32)
                best_mse_c = torch.full((C,), float("inf"), dtype=val_mse_c_pred_base.dtype)
                best_mae_c = torch.full((C,), float("inf"), dtype=val_mae_c_pred_base.dtype)
                best_scale_c = torch.zeros((C,), dtype=torch.float32)
                for scale_value in scale_grid.tolist():
                    scale_c = torch.full((C,), float(scale_value), device=device, dtype=torch.float32)
                    (
                        _,
                        _,
                        _,
                        cand_mse_c,
                        cand_mae_c,
                        _,
                        _,
                        _,
                    ) = eval_loop(
                        model, gate, lam_kp_best,
                        penalty_names, penalty_fns,
                        val_loader_summary, cluster_id_c, K, moe_cfg, device,
                        select_ranks=select_ranks,
                        collect_plot=False, channel_count=C,
                        mse_weight=mse_weight,
                        gate_entropy_weight=gate_entropy_weight,
                        gate_balance_weight=gate_balance_weight,
                        gate_soft_weight=gate_soft_weight,
                        gate_entropy_target_frac=gate_entropy_target_frac,
                        penalty_scale=penalty_scale,
                        dynamic_lambda=dynamic_lambda,
                        lambda_min_kp=lambda_min_kp,
                        mae_objective_weight=mae_eval_weight,
                        mae_objective_kind=mae_objective_kind,
                        mae_objective_beta=mae_objective_beta,
                        pred_residual=pred_residual,
                        pred_residual_scale_c=scale_c,
                        residual_correction_ch=residual_correction_base_ch,
                        knn_hybrid=None,
                        eval_start=val_eval_start,
                    )
                    better = cand_mse_c < best_mse_c
                    best_mse_c = torch.where(better, cand_mse_c, best_mse_c)
                    best_mae_c = torch.where(better, cand_mae_c, best_mae_c)
                    best_scale_c = torch.where(
                        better,
                        torch.full_like(best_scale_c, float(scale_value)),
                        best_scale_c,
                    )
                pred_residual_channel_scale_c = best_scale_c.to(device=device, dtype=torch.float32)
                val_scaled_mse_c = best_mse_c
                val_scaled_mae_c = best_mae_c
                use_residual_c = pred_residual_channel_scale_c.detach().cpu() > 1.0e-8
                scale_values = [float(v) for v in pred_residual_channel_scale_c.detach().cpu().tolist()]
                residual_scale_mean_value = float(pred_residual_channel_scale_c.mean().item())
            elif residual_selection_policy in {"val_mse_gate", "val_mse_gate_guarded"}:
                gate_calib_cfg = pred_residual_cfg.get("gate_calibrator", {}) or {}
                pred_residual_gate_model, pred_residual_gate_summary = train_pred_residual_gate(
                    model=model,
                    gate=gate,
                    pred_residual=pred_residual,
                    loader=val_loader_summary,
                    cluster_id_c=cluster_id_c,
                    K=K,
                    moe_cfg=moe_cfg,
                    device=device,
                    penalty_names=penalty_names,
                    penalty_fns=penalty_fns,
                    penalty_scale=penalty_scale,
                    select_ranks=select_ranks,
                    gate_soft_weight=gate_soft_weight,
                    channel_names=channel_names,
                    cfg=gate_calib_cfg,
                )
                if pred_residual_gate_model is None:
                    pred_residual_channel_scale_c = zero_residual_scale_c
                    val_scaled_mse_c = val_mse_c_pred_base
                    val_scaled_mae_c = val_mae_c_pred_base
                    use_residual_c = torch.zeros(C, dtype=torch.bool)
                    pred_residual_channel_scale_c = zero_residual_scale_c
                    scale_values = [0.0 for _ in range(C)]
                    residual_scale_mean_value = 0.0
                else:
                    (
                        _,
                        _,
                        _,
                        val_gate_mse_c,
                        val_gate_mae_c,
                        _,
                        _,
                        _,
                    ) = eval_loop(
                        model, gate, lam_kp_best,
                        penalty_names, penalty_fns,
                        val_loader_summary, cluster_id_c, K, moe_cfg, device,
                        select_ranks=select_ranks,
                        collect_plot=False, channel_count=C,
                        mse_weight=mse_weight,
                        gate_entropy_weight=gate_entropy_weight,
                        gate_balance_weight=gate_balance_weight,
                        gate_soft_weight=gate_soft_weight,
                        gate_entropy_target_frac=gate_entropy_target_frac,
                        penalty_scale=penalty_scale,
                        dynamic_lambda=dynamic_lambda,
                        lambda_min_kp=lambda_min_kp,
                        mae_objective_weight=mae_eval_weight,
                        mae_objective_kind=mae_objective_kind,
                        mae_objective_beta=mae_objective_beta,
                        pred_residual=pred_residual,
                        pred_residual_gate=pred_residual_gate_model,
                        residual_correction_ch=residual_correction_base_ch,
                        knn_hybrid=None,
                        eval_start=val_eval_start,
                    )
                    val_scaled_mse_c = val_gate_mse_c
                    val_scaled_mae_c = val_gate_mae_c
                    hold_scales = pred_residual_gate_summary.get("holdout_mean_scale", [])
                    min_abs = float(pred_residual_cfg.get("selection_min_abs_improvement", 0.0))
                    min_rel = float(pred_residual_cfg.get("selection_min_rel_improvement", 0.0))
                    required = torch.maximum(
                        torch.full_like(val_mse_c_pred_base, min_abs),
                        min_rel * val_mse_c_pred_base.abs().clamp_min(1.0e-12),
                    )
                    use_residual_c = (val_mse_c_pred_base - val_scaled_mse_c) > required
                    if residual_selection_policy == "val_mse_gate_guarded":
                        pred_residual_channel_scale_c = use_residual_c.to(device=device, dtype=torch.float32)
                        (
                            _,
                            _,
                            _,
                            val_guarded_mse_c,
                            val_guarded_mae_c,
                            _,
                            _,
                            _,
                        ) = eval_loop(
                            model, gate, lam_kp_best,
                            penalty_names, penalty_fns,
                            val_loader_summary, cluster_id_c, K, moe_cfg, device,
                            select_ranks=select_ranks,
                            collect_plot=False, channel_count=C,
                            mse_weight=mse_weight,
                            gate_entropy_weight=gate_entropy_weight,
                            gate_balance_weight=gate_balance_weight,
                            gate_soft_weight=gate_soft_weight,
                            gate_entropy_target_frac=gate_entropy_target_frac,
                            penalty_scale=penalty_scale,
                            dynamic_lambda=dynamic_lambda,
                            lambda_min_kp=lambda_min_kp,
                            mae_objective_weight=mae_eval_weight,
                            mae_objective_kind=mae_objective_kind,
                            mae_objective_beta=mae_objective_beta,
                            pred_residual=pred_residual,
                            pred_residual_gate=pred_residual_gate_model,
                            pred_residual_scale_c=pred_residual_channel_scale_c,
                            residual_correction_ch=residual_correction_base_ch,
                            knn_hybrid=None,
                            eval_start=val_eval_start,
                        )
                        val_scaled_mse_c = val_guarded_mse_c
                        val_scaled_mae_c = val_guarded_mae_c
                        scale_values = [float(v) for v in pred_residual_channel_scale_c.detach().cpu().tolist()]
                        residual_scale_mean_value = float(pred_residual_channel_scale_c.mean().item())
                    else:
                        pred_residual_channel_scale_c = None
                        scale_values = [float(v) for v in hold_scales] if len(hold_scales) == C else []
                        residual_scale_mean_value = (
                            float(sum(scale_values) / max(len(scale_values), 1)) if len(scale_values) > 0 else 1.0
                        )
            else:
                min_abs = float(pred_residual_cfg.get("selection_min_abs_improvement", 0.0))
                min_rel = float(pred_residual_cfg.get("selection_min_rel_improvement", 0.0))
                required = torch.maximum(
                    torch.full_like(val_mse_c_pred_base, min_abs),
                    min_rel * val_mse_c_pred_base.abs().clamp_min(1.0e-12),
                )
                use_residual_c = (val_mse_c_pred_base - val_mse_c_base) > required
                pred_residual_channel_scale_c = use_residual_c.to(device=device, dtype=torch.float32)
                scale_values = [float(v) for v in pred_residual_channel_scale_c.detach().cpu().tolist()]
                residual_scale_mean_value = float(pred_residual_channel_scale_c.mean().item())
            pred_residual_selection_summary = {
                "policy": residual_selection_policy,
                "min_abs_improvement": float(pred_residual_cfg.get("selection_min_abs_improvement", 0.0)),
                "min_rel_improvement": float(pred_residual_cfg.get("selection_min_rel_improvement", 0.0)),
                "scale_values": scale_values,
                "mean_scale": float(residual_scale_mean_value),
                "num_residual_channels": int(use_residual_c.sum().item()),
                "residual_channels": [
                    channel_names[i] for i, use_residual in enumerate(use_residual_c.tolist()) if bool(use_residual)
                ],
                "base_channels": [
                    channel_names[i] for i, use_residual in enumerate(use_residual_c.tolist()) if not bool(use_residual)
                ],
                "val_pred_base_avg_mse": float(reduce_cluster_metric(val_mse_pred_base_k, cluster_weight_k).item()),
                "val_pred_base_avg_mae": float(reduce_cluster_metric(val_mae_pred_base_k, cluster_weight_k).item()),
                "val_residual_avg_mse": float(reduce_cluster_metric(val_mse_best_k, cluster_weight_k).item()),
                "val_residual_avg_mae": float(reduce_cluster_metric(val_mae_best_k, cluster_weight_k).item()),
                "val_scaled_avg_mse": float(val_scaled_mse_c.mean().item()),
                "val_scaled_avg_mae": float(val_scaled_mae_c.mean().item()),
                "val_pred_base_mse_per_channel": [float(v) for v in val_mse_c_pred_base.detach().cpu().tolist()],
                "val_residual_mse_per_channel": [float(v) for v in val_mse_c_base.detach().cpu().tolist()],
                "val_scaled_mse_per_channel": [float(v) for v in val_scaled_mse_c.detach().cpu().tolist()],
                "val_pred_base_mae_per_channel": [float(v) for v in val_mae_c_pred_base.detach().cpu().tolist()],
                "val_residual_mae_per_channel": [float(v) for v in val_mae_c_base.detach().cpu().tolist()],
                "val_scaled_mae_per_channel": [float(v) for v in val_scaled_mae_c.detach().cpu().tolist()],
                "gate_calibrator": pred_residual_gate_summary,
            }
            print(
                "Prediction residual selection: "
                f"policy={residual_selection_policy}, "
                f"residual_channels={pred_residual_selection_summary['num_residual_channels']}/{C}, "
                f"val_base_MSE={pred_residual_selection_summary['val_pred_base_avg_mse']:.6f}, "
                f"val_residual_MSE={pred_residual_selection_summary['val_residual_avg_mse']:.6f}, "
                f"val_scaled_MSE={pred_residual_selection_summary['val_scaled_avg_mse']:.6f}, "
                f"mean_scale={pred_residual_selection_summary['mean_scale']:.3f}"
            )
        if knn_hybrid_val is not None:
            knn_hybrid_val.reset_confidence_stats()
            val_loss_hybrid_k, val_mse_hybrid_k, val_mae_hybrid_k, val_mse_c_hybrid, val_mae_c_hybrid, _, _, _ = eval_loop(
                model, gate, lam_kp_best,
                penalty_names, penalty_fns,
                val_loader_summary, cluster_id_c, K, moe_cfg, device,
                select_ranks=select_ranks,
                collect_plot=False, channel_count=C,
                mse_weight=mse_weight,
                gate_entropy_weight=gate_entropy_weight,
                gate_balance_weight=gate_balance_weight,
                gate_soft_weight=gate_soft_weight,
                gate_entropy_target_frac=gate_entropy_target_frac,
                penalty_scale=penalty_scale,
                dynamic_lambda=dynamic_lambda,
                lambda_min_kp=lambda_min_kp,
                mae_objective_weight=mae_eval_weight,
                mae_objective_kind=mae_objective_kind,
                mae_objective_beta=mae_objective_beta,
                pred_residual=pred_residual,
                residual_correction_ch=residual_correction_hybrid_ch,
                knn_hybrid=knn_hybrid_val,
                eval_start=val_eval_start,
            )
            val_summary_hybrid = {
                "avg_loss": float(reduce_cluster_metric(val_loss_hybrid_k, cluster_weight_k).item()),
                "avg_mse": float(reduce_cluster_metric(val_mse_hybrid_k, cluster_weight_k).item()),
                "avg_mae": float(reduce_cluster_metric(val_mae_hybrid_k, cluster_weight_k).item()),
                "per_cluster_loss": [float(v) for v in val_loss_hybrid_k.detach().cpu().tolist()],
                "per_cluster_mse": [float(v) for v in val_mse_hybrid_k.detach().cpu().tolist()],
                "per_cluster_mae": [float(v) for v in val_mae_hybrid_k.detach().cpu().tolist()],
            }
            val_hybrid_confidence = knn_hybrid_val.get_confidence_stats()

    lam_kp_test = lam_kp_best
    test_loss_k = test_mse_k = test_mae_k = None
    mse_c = mae_c = None
    plot_cache = {}
    best_sample = {}
    worst_sample = {}
    if not skip_test:
        test_loss_k, test_mse_k, test_mae_k, mse_c, mae_c, plot_cache, best_sample, worst_sample = eval_loop(
            model, gate, lam_kp_test,
            penalty_names, penalty_fns,
            dl_te, cluster_id_c, K, moe_cfg, device,
            select_ranks=select_ranks,
            collect_plot=plot_enable, plot_idx=plot_idx, channel_count=C,
            mse_weight=mse_weight,
            gate_entropy_weight=gate_entropy_weight,
            gate_balance_weight=gate_balance_weight,
            gate_soft_weight=gate_soft_weight,
            gate_entropy_target_frac=gate_entropy_target_frac,
            penalty_scale=penalty_scale,
            dynamic_lambda=dynamic_lambda,
            lambda_min_kp=lambda_min_kp,
            mae_objective_weight=mae_eval_weight,
            mae_objective_kind=mae_objective_kind,
            mae_objective_beta=mae_objective_beta,
            pred_residual=pred_residual,
            pred_residual_gate=pred_residual_gate_model,
            pred_residual_scale_c=pred_residual_channel_scale_c,
            residual_correction_ch=residual_correction_base_ch,
            knn_hybrid=None,
            eval_start=test_eval_start,
        )
    test_loss_hybrid_k = None
    test_mse_hybrid_k = None
    test_mae_hybrid_k = None
    mse_c_hybrid = None
    mae_c_hybrid = None
    test_hybrid_confidence = None
    if knn_hybrid_test is not None and not skip_test:
        knn_hybrid_test.reset_confidence_stats()
        test_loss_hybrid_k, test_mse_hybrid_k, test_mae_hybrid_k, mse_c_hybrid, mae_c_hybrid, _, _, _ = eval_loop(
            model, gate, lam_kp_test,
            penalty_names, penalty_fns,
            dl_te, cluster_id_c, K, moe_cfg, device,
            select_ranks=select_ranks,
            collect_plot=False, plot_idx=None, channel_count=C,
            mse_weight=mse_weight,
            gate_entropy_weight=gate_entropy_weight,
            gate_balance_weight=gate_balance_weight,
            gate_soft_weight=gate_soft_weight,
            gate_entropy_target_frac=gate_entropy_target_frac,
            penalty_scale=penalty_scale,
            dynamic_lambda=dynamic_lambda,
            lambda_min_kp=lambda_min_kp,
            mae_objective_weight=mae_eval_weight,
            mae_objective_kind=mae_objective_kind,
            mae_objective_beta=mae_objective_beta,
            pred_residual=pred_residual,
            residual_correction_ch=residual_correction_hybrid_ch,
            knn_hybrid=knn_hybrid_test,
            eval_start=test_eval_start,
        )
        test_hybrid_confidence = knn_hybrid_test.get_confidence_stats()

    knn_sweep_results = []
    knn_sweep_cfgs = cfg.get("knn_hybrid", {}).get("sweep", []) or []
    if (not skip_test) and knn_cfg.enable and len(knn_sweep_cfgs) > 0 and len(dva) > 0 and val_summary is not None:
        base_knn_dict = dict(cfg.get("knn_hybrid", {}))
        base_knn_dict.pop("sweep", None)
        base_knn_dict.pop("path", None)
        val_loader_sweep = DataLoader(dva, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)
        best_sweep_val_mae = float("inf") if val_summary_hybrid is None else float(val_summary_hybrid["avg_mae"])
        best_sweep_payload = None
        sweep_guard = float(cfg.get("knn_hybrid", {}).get("selection_max_rel_mse_regression", 0.03))
        base_val_mse_for_guard = float(val_summary["avg_mse"])
        allowed_sweep_val_mse = base_val_mse_for_guard * (1.0 + max(0.0, sweep_guard))
        for sweep_idx, sweep_override in enumerate(knn_sweep_cfgs):
            if sweep_override is None:
                sweep_override = {}
            sweep_override = dict(sweep_override)
            sweep_name = str(sweep_override.pop("name", f"sweep_{sweep_idx:02d}"))
            cand_knn_dict = dict(base_knn_dict)
            cand_knn_dict.update(sweep_override)
            cand_knn_cfg = KNNShapeConfig.from_dict(cand_knn_dict).resolved_for_horizon(H)
            cand_val_hybrid, cand_test_hybrid, cand_bank_windows = _build_knn_hybrids_for(cand_knn_cfg)

            cand_val_hybrid.reset_confidence_stats()
            cand_val_loss_k, cand_val_mse_k, cand_val_mae_k, cand_val_mse_c, cand_val_mae_c, _, _, _ = eval_loop(
                model, gate, lam_kp_best,
                penalty_names, penalty_fns,
                val_loader_sweep, cluster_id_c, K, moe_cfg, device,
                select_ranks=select_ranks,
                collect_plot=False, channel_count=C,
                mse_weight=mse_weight,
                gate_entropy_weight=gate_entropy_weight,
                gate_balance_weight=gate_balance_weight,
                gate_soft_weight=gate_soft_weight,
                gate_entropy_target_frac=gate_entropy_target_frac,
                penalty_scale=penalty_scale,
                dynamic_lambda=dynamic_lambda,
                lambda_min_kp=lambda_min_kp,
                mae_objective_weight=mae_eval_weight,
                mae_objective_kind=mae_objective_kind,
                mae_objective_beta=mae_objective_beta,
                pred_residual=pred_residual,
                residual_correction_ch=residual_correction_hybrid_ch,
                knn_hybrid=cand_val_hybrid,
                eval_start=val_eval_start,
            )
            cand_val_confidence = cand_val_hybrid.get_confidence_stats()

            cand_test_hybrid.reset_confidence_stats()
            cand_test_loss_k, cand_test_mse_k, cand_test_mae_k, cand_mse_c, cand_mae_c, _, _, _ = eval_loop(
                model, gate, lam_kp_test,
                penalty_names, penalty_fns,
                dl_te, cluster_id_c, K, moe_cfg, device,
                select_ranks=select_ranks,
                collect_plot=False, plot_idx=None, channel_count=C,
                mse_weight=mse_weight,
                gate_entropy_weight=gate_entropy_weight,
                gate_balance_weight=gate_balance_weight,
                gate_soft_weight=gate_soft_weight,
                gate_entropy_target_frac=gate_entropy_target_frac,
                penalty_scale=penalty_scale,
                dynamic_lambda=dynamic_lambda,
                lambda_min_kp=lambda_min_kp,
                mae_objective_weight=mae_eval_weight,
                mae_objective_kind=mae_objective_kind,
                mae_objective_beta=mae_objective_beta,
                pred_residual=pred_residual,
                residual_correction_ch=residual_correction_hybrid_ch,
                knn_hybrid=cand_test_hybrid,
                eval_start=test_eval_start,
            )
            cand_test_confidence = cand_test_hybrid.get_confidence_stats()
            cand_val_summary = {
                "avg_loss": float(reduce_cluster_metric(cand_val_loss_k, cluster_weight_k).item()),
                "avg_mse": float(reduce_cluster_metric(cand_val_mse_k, cluster_weight_k).item()),
                "avg_mae": float(reduce_cluster_metric(cand_val_mae_k, cluster_weight_k).item()),
                "per_cluster_loss": [float(v) for v in cand_val_loss_k.detach().cpu().tolist()],
                "per_cluster_mse": [float(v) for v in cand_val_mse_k.detach().cpu().tolist()],
                "per_cluster_mae": [float(v) for v in cand_val_mae_k.detach().cpu().tolist()],
            }
            cand_test_summary = {
                "avg_mae": float(cand_mae_c.mean().item()),
                "avg_mse": float(reduce_cluster_metric(cand_test_mse_k, cluster_weight_k).item()),
                "per_cluster_loss": [float(v) for v in cand_test_loss_k.detach().cpu().tolist()],
                "per_cluster_mse": [float(v) for v in cand_test_mse_k.detach().cpu().tolist()],
                "per_cluster_mae": [float(v) for v in cand_test_mae_k.detach().cpu().tolist()],
            }
            cand_result = {
                "name": sweep_name,
                "config": cand_test_hybrid.describe(),
                "val": cand_val_summary,
                "test": cand_test_summary,
                "val_hybrid_confidence": cand_val_confidence,
                "test_hybrid_confidence": cand_test_confidence,
                "selected_by_val_mae_guard": bool(
                    cand_val_summary["avg_mae"] < best_sweep_val_mae
                    and cand_val_summary["avg_mse"] <= allowed_sweep_val_mse
                ),
            }
            knn_sweep_results.append(cand_result)
            print(
                f"KNN sweep {sweep_name}: "
                f"val_MAE={cand_val_summary['avg_mae']:.6f}, "
                f"val_MSE={cand_val_summary['avg_mse']:.6f}, "
                f"test_MAE={cand_test_summary['avg_mae']:.6f}, "
                f"test_MSE={cand_test_summary['avg_mse']:.6f}"
            )

            if cand_result["selected_by_val_mae_guard"]:
                best_sweep_val_mae = float(cand_val_summary["avg_mae"])
                best_sweep_payload = {
                    "val_hybrid": cand_val_summary,
                    "test_loss_hybrid_k": cand_test_loss_k,
                    "test_mse_hybrid_k": cand_test_mse_k,
                    "test_mae_hybrid_k": cand_test_mae_k,
                    "mse_c_hybrid": cand_mse_c,
                    "mae_c_hybrid": cand_mae_c,
                    "val_mse_c_hybrid": cand_val_mse_c,
                    "val_mae_c_hybrid": cand_val_mae_c,
                    "val_hybrid_confidence": cand_val_confidence,
                    "test_hybrid_confidence": cand_test_confidence,
                    "knn_hybrid_val": cand_val_hybrid,
                    "knn_hybrid_test": cand_test_hybrid,
                    "knn_test_bank_windows": cand_bank_windows,
                }

        if best_sweep_payload is not None:
            val_summary_hybrid = best_sweep_payload["val_hybrid"]
            test_loss_hybrid_k = best_sweep_payload["test_loss_hybrid_k"]
            test_mse_hybrid_k = best_sweep_payload["test_mse_hybrid_k"]
            test_mae_hybrid_k = best_sweep_payload["test_mae_hybrid_k"]
            mse_c_hybrid = best_sweep_payload["mse_c_hybrid"]
            mae_c_hybrid = best_sweep_payload["mae_c_hybrid"]
            val_mse_c_hybrid = best_sweep_payload["val_mse_c_hybrid"]
            val_mae_c_hybrid = best_sweep_payload["val_mae_c_hybrid"]
            val_hybrid_confidence = best_sweep_payload["val_hybrid_confidence"]
            test_hybrid_confidence = best_sweep_payload["test_hybrid_confidence"]
            knn_hybrid_val = best_sweep_payload["knn_hybrid_val"]
            knn_hybrid_test = best_sweep_payload["knn_hybrid_test"]
            knn_test_bank_windows = int(best_sweep_payload["knn_test_bank_windows"])

    fusion_cfg = cfg.get("knn_hybrid", {}).get("fusion_calibrator", {}) or {}
    if bool(dict(fusion_cfg).get("enable", False)) and knn_hybrid_val is not None and len(dva) > 0:
        val_loader_fusion = DataLoader(dva, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)
        knn_hybrid_val.reset_confidence_stats()
        knn_fusion_scale_ch, knn_fusion_summary = estimate_knn_fusion_scale(
            model=model,
            loader=val_loader_fusion,
            cluster_id_c=cluster_id_c,
            device=device,
            knn_hybrid=knn_hybrid_val,
            eval_start=val_eval_start,
            metric=str(dict(fusion_cfg).get("metric", "mae")),
            min_scale=float(dict(fusion_cfg).get("min_scale", 0.0)),
            max_scale=float(dict(fusion_cfg).get("max_scale", 1.5)),
            grid_steps=int(dict(fusion_cfg).get("grid_steps", 31)),
        )
        if knn_fusion_scale_ch is not None:
            knn_fusion_summary["channel_names"] = list(channel_names)
            print(
                "KNN fusion calibrator: "
                f"metric={knn_fusion_summary['metric']}, "
                f"mean_scale={knn_fusion_summary['mean_scale']:.3f}, "
                f"range=[{knn_fusion_summary['min_fitted_scale']:.3f}, {knn_fusion_summary['max_fitted_scale']:.3f}]"
            )

            knn_hybrid_val.reset_confidence_stats()
            val_loss_hybrid_k, val_mse_hybrid_k, val_mae_hybrid_k, val_mse_c_hybrid, val_mae_c_hybrid, _, _, _ = eval_loop(
                model, gate, lam_kp_best,
                penalty_names, penalty_fns,
                val_loader_fusion, cluster_id_c, K, moe_cfg, device,
                select_ranks=select_ranks,
                collect_plot=False, channel_count=C,
                mse_weight=mse_weight,
                gate_entropy_weight=gate_entropy_weight,
                gate_balance_weight=gate_balance_weight,
                gate_soft_weight=gate_soft_weight,
                gate_entropy_target_frac=gate_entropy_target_frac,
                penalty_scale=penalty_scale,
                dynamic_lambda=dynamic_lambda,
                lambda_min_kp=lambda_min_kp,
                mae_objective_weight=mae_eval_weight,
                mae_objective_kind=mae_objective_kind,
                mae_objective_beta=mae_objective_beta,
                pred_residual=pred_residual,
                residual_correction_ch=residual_correction_hybrid_ch,
                knn_hybrid=knn_hybrid_val,
                knn_fusion_scale_ch=knn_fusion_scale_ch,
                eval_start=val_eval_start,
            )
            val_summary_hybrid = {
                "avg_loss": float(reduce_cluster_metric(val_loss_hybrid_k, cluster_weight_k).item()),
                "avg_mse": float(reduce_cluster_metric(val_mse_hybrid_k, cluster_weight_k).item()),
                "avg_mae": float(reduce_cluster_metric(val_mae_hybrid_k, cluster_weight_k).item()),
                "per_cluster_loss": [float(v) for v in val_loss_hybrid_k.detach().cpu().tolist()],
                "per_cluster_mse": [float(v) for v in val_mse_hybrid_k.detach().cpu().tolist()],
                "per_cluster_mae": [float(v) for v in val_mae_hybrid_k.detach().cpu().tolist()],
            }
            val_hybrid_confidence = knn_hybrid_val.get_confidence_stats()

            if knn_hybrid_test is not None and not skip_test:
                knn_hybrid_test.reset_confidence_stats()
                test_loss_hybrid_k, test_mse_hybrid_k, test_mae_hybrid_k, mse_c_hybrid, mae_c_hybrid, _, _, _ = eval_loop(
                    model, gate, lam_kp_test,
                    penalty_names, penalty_fns,
                    dl_te, cluster_id_c, K, moe_cfg, device,
                    select_ranks=select_ranks,
                    collect_plot=False, plot_idx=None, channel_count=C,
                    mse_weight=mse_weight,
                    gate_entropy_weight=gate_entropy_weight,
                    gate_balance_weight=gate_balance_weight,
                    gate_soft_weight=gate_soft_weight,
                    gate_entropy_target_frac=gate_entropy_target_frac,
                    penalty_scale=penalty_scale,
                    dynamic_lambda=dynamic_lambda,
                    lambda_min_kp=lambda_min_kp,
                    mae_objective_weight=mae_eval_weight,
                    mae_objective_kind=mae_objective_kind,
                    mae_objective_beta=mae_objective_beta,
                    pred_residual=pred_residual,
                    residual_correction_ch=residual_correction_hybrid_ch,
                    knn_hybrid=knn_hybrid_test,
                    knn_fusion_scale_ch=knn_fusion_scale_ch,
                    eval_start=test_eval_start,
                )
                test_hybrid_confidence = knn_hybrid_test.get_confidence_stats()

    gate_calib_cfg = cfg.get("knn_hybrid", {}).get("gate_calibrator", {}) or {}
    if bool(dict(gate_calib_cfg).get("enable", False)) and knn_hybrid_val is not None and len(dva) > 0:
        val_loader_gate = DataLoader(dva, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)
        knn_hybrid_val.reset_confidence_stats()
        knn_gate_model, knn_gate_summary = train_knn_residual_gate(
            model=model,
            loader=val_loader_gate,
            cluster_id_c=cluster_id_c,
            device=device,
            knn_hybrid=knn_hybrid_val,
            eval_start=val_eval_start,
            channel_names=channel_names,
            cfg=dict(gate_calib_cfg),
        )
        if knn_gate_model is not None:
            print(
                "KNN residual gate: "
                f"holdout_MAE={knn_gate_summary['holdout_mae']:.6f}, "
                f"holdout_MSE={knn_gate_summary['holdout_mse']:.6f}, "
                f"best_epoch={knn_gate_summary['best_epoch']}"
            )

            knn_hybrid_val.reset_confidence_stats()
            val_loss_hybrid_k, val_mse_hybrid_k, val_mae_hybrid_k, val_mse_c_hybrid, val_mae_c_hybrid, _, _, _ = eval_loop(
                model, gate, lam_kp_best,
                penalty_names, penalty_fns,
                val_loader_gate, cluster_id_c, K, moe_cfg, device,
                select_ranks=select_ranks,
                collect_plot=False, channel_count=C,
                mse_weight=mse_weight,
                gate_entropy_weight=gate_entropy_weight,
                gate_balance_weight=gate_balance_weight,
                gate_soft_weight=gate_soft_weight,
                gate_entropy_target_frac=gate_entropy_target_frac,
                penalty_scale=penalty_scale,
                dynamic_lambda=dynamic_lambda,
                lambda_min_kp=lambda_min_kp,
                mae_objective_weight=mae_eval_weight,
                mae_objective_kind=mae_objective_kind,
                mae_objective_beta=mae_objective_beta,
                pred_residual=pred_residual,
                residual_correction_ch=residual_correction_hybrid_ch,
                knn_hybrid=knn_hybrid_val,
                knn_fusion_gate=knn_gate_model,
                eval_start=val_eval_start,
            )
            val_summary_hybrid = {
                "avg_loss": float(reduce_cluster_metric(val_loss_hybrid_k, cluster_weight_k).item()),
                "avg_mse": float(reduce_cluster_metric(val_mse_hybrid_k, cluster_weight_k).item()),
                "avg_mae": float(reduce_cluster_metric(val_mae_hybrid_k, cluster_weight_k).item()),
                "per_cluster_loss": [float(v) for v in val_loss_hybrid_k.detach().cpu().tolist()],
                "per_cluster_mse": [float(v) for v in val_mse_hybrid_k.detach().cpu().tolist()],
                "per_cluster_mae": [float(v) for v in val_mae_hybrid_k.detach().cpu().tolist()],
            }
            val_hybrid_confidence = knn_hybrid_val.get_confidence_stats()

            if knn_hybrid_test is not None and not skip_test:
                knn_hybrid_test.reset_confidence_stats()
                test_loss_hybrid_k, test_mse_hybrid_k, test_mae_hybrid_k, mse_c_hybrid, mae_c_hybrid, _, _, _ = eval_loop(
                    model, gate, lam_kp_test,
                    penalty_names, penalty_fns,
                    dl_te, cluster_id_c, K, moe_cfg, device,
                    select_ranks=select_ranks,
                    collect_plot=False, plot_idx=None, channel_count=C,
                    mse_weight=mse_weight,
                    gate_entropy_weight=gate_entropy_weight,
                    gate_balance_weight=gate_balance_weight,
                    gate_soft_weight=gate_soft_weight,
                    gate_entropy_target_frac=gate_entropy_target_frac,
                    penalty_scale=penalty_scale,
                    dynamic_lambda=dynamic_lambda,
                    lambda_min_kp=lambda_min_kp,
                    mae_objective_weight=mae_eval_weight,
                    mae_objective_kind=mae_objective_kind,
                    mae_objective_beta=mae_objective_beta,
                    pred_residual=pred_residual,
                    residual_correction_ch=residual_correction_hybrid_ch,
                    knn_hybrid=knn_hybrid_test,
                    knn_fusion_gate=knn_gate_model,
                    eval_start=test_eval_start,
                )
                test_hybrid_confidence = knn_hybrid_test.get_confidence_stats()

    df = None
    avg_mae = None
    avg_mse = None
    avg_mae_hybrid = None
    avg_mse_hybrid = None
    if not skip_test:
        df = pd.DataFrame({
            "channel": channel_names,
            "MAE": mae_c.numpy(),
            "MSE": mse_c.numpy(),
            "cluster_id": cluster_id_c.detach().cpu().numpy()
        })
        if mse_c_hybrid is not None and mae_c_hybrid is not None:
            df["MAE_hybrid"] = mae_c_hybrid.numpy()
            df["MSE_hybrid"] = mse_c_hybrid.numpy()
        avg_mae = float(df["MAE"].mean())
        avg_mse = float(reduce_cluster_metric(test_mse_k, cluster_weight_k).item())
        if mse_c_hybrid is not None and mae_c_hybrid is not None and test_mse_hybrid_k is not None:
            avg_mae_hybrid = float(df["MAE_hybrid"].mean())
            avg_mse_hybrid = float(reduce_cluster_metric(test_mse_hybrid_k, cluster_weight_k).item())

    channel_selection_summary = None
    channel_selected_avg_mae = None
    channel_selected_avg_mse = None
    channel_use_hybrid = None
    channel_selection_policy = str(cfg.get("knn_hybrid", {}).get("channel_selection_policy", "none")).lower()
    if channel_selection_policy in {"false", "off", "disable", "disabled"}:
        channel_selection_policy = "none"
    if channel_selection_policy not in {"none", "val_mae_guarded"}:
        raise ValueError(
            "Unsupported knn_hybrid.channel_selection_policy="
            f"'{channel_selection_policy}'. Expected none or val_mae_guarded."
        )
    if (
        channel_selection_policy == "val_mae_guarded"
        and val_mae_c_base is not None
        and val_mae_c_hybrid is not None
        and val_mse_c_base is not None
        and not skip_test
        and val_mse_c_hybrid is not None
        and mae_c_hybrid is not None
        and mse_c_hybrid is not None
    ):
        channel_min_rel = float(cfg.get("knn_hybrid", {}).get("channel_selection_min_rel_improvement", 0.0))
        channel_max_mse_reg = float(
            cfg.get(
                "knn_hybrid",
                {},
            ).get(
                "channel_selection_max_rel_mse_regression",
                cfg.get("knn_hybrid", {}).get("selection_max_rel_mse_regression", 0.03),
            )
        )
        required_mae_drop = channel_min_rel * val_mae_c_base.abs().clamp_min(1.0e-12)
        mae_ok = (val_mae_c_base - val_mae_c_hybrid) > required_mae_drop
        mse_ok = (val_mse_c_hybrid - val_mse_c_base) <= (
            channel_max_mse_reg * val_mse_c_base.abs().clamp_min(1.0e-12)
        )
        channel_use_hybrid = (mae_ok & mse_ok).detach().cpu()
        selected_mae_c = torch.where(channel_use_hybrid, mae_c_hybrid.cpu(), mae_c.cpu())
        selected_mse_c = torch.where(channel_use_hybrid, mse_c_hybrid.cpu(), mse_c.cpu())
        df["use_hybrid_by_val"] = channel_use_hybrid.numpy().astype(bool)
        df["MAE_selected_channel"] = selected_mae_c.numpy()
        df["MSE_selected_channel"] = selected_mse_c.numpy()
        channel_selected_avg_mae = float(selected_mae_c.mean().item())
        channel_selected_avg_mse = float(selected_mse_c.mean().item())
        channel_selection_summary = {
            "policy": channel_selection_policy,
            "min_rel_mae_improvement": float(channel_min_rel),
            "max_rel_mse_regression": float(channel_max_mse_reg),
            "num_hybrid_channels": int(channel_use_hybrid.sum().item()),
            "hybrid_channels": [
                channel_names[i] for i, use_hybrid in enumerate(channel_use_hybrid.tolist()) if bool(use_hybrid)
            ],
            "base_channels": [
                channel_names[i] for i, use_hybrid in enumerate(channel_use_hybrid.tolist()) if not bool(use_hybrid)
            ],
            "avg_mae": channel_selected_avg_mae,
            "avg_mse": channel_selected_avg_mse,
            "val_base_mae_per_channel": [float(v) for v in val_mae_c_base.detach().cpu().tolist()],
            "val_hybrid_mae_per_channel": [float(v) for v in val_mae_c_hybrid.detach().cpu().tolist()],
            "val_base_mse_per_channel": [float(v) for v in val_mse_c_base.detach().cpu().tolist()],
            "val_hybrid_mse_per_channel": [float(v) for v in val_mse_c_hybrid.detach().cpu().tolist()],
        }

    selected_variant = "base"
    selected_avg_mae = avg_mae
    selected_avg_mse = avg_mse
    selected_base_val_mse = None if val_summary is None else val_summary.get("avg_mse")
    selected_hybrid_val_mse = None if val_summary_hybrid is None else val_summary_hybrid.get("avg_mse")
    selected_base_val_mae = None if val_summary is None else val_summary.get("avg_mae")
    selected_hybrid_val_mae = None if val_summary_hybrid is None else val_summary_hybrid.get("avg_mae")
    knn_selection_policy = str(cfg.get("knn_hybrid", {}).get("selection_policy", "hybrid")).lower()
    if knn_selection_policy == "val_mse":
        knn_selection_policy = "val_mse_margin"
    if knn_selection_policy not in {"hybrid", "val_mse_margin", "val_mae_guarded", "base"}:
        raise ValueError(
            "Unsupported knn_hybrid.selection_policy="
            f"'{knn_selection_policy}'. Expected hybrid, val_mse_margin, val_mae_guarded, val_mse, or base."
        )
    knn_selection_min_abs_improvement = float(cfg.get("knn_hybrid", {}).get("selection_min_abs_improvement", 0.0))
    knn_selection_min_rel_improvement = float(cfg.get("knn_hybrid", {}).get("selection_min_rel_improvement", 0.0))
    knn_selection_max_rel_mse_regression = float(cfg.get("knn_hybrid", {}).get("selection_max_rel_mse_regression", 0.03))
    selected_val_mse_improvement = None
    selected_val_mse_improvement_pct = None
    required_val_mse_improvement = None
    selected_val_mae_improvement = None
    selected_val_mae_improvement_pct = None
    required_val_mae_improvement = None
    selected_val_mse_regression = None
    selected_val_mse_regression_pct = None
    allowed_val_mse_regression = None
    if selected_base_val_mse is not None and selected_hybrid_val_mse is not None:
        selected_val_mse_improvement = float(selected_base_val_mse) - float(selected_hybrid_val_mse)
        selected_val_mse_improvement_pct = (
            100.0 * selected_val_mse_improvement / max(abs(float(selected_base_val_mse)), 1.0e-12)
        )
        required_val_mse_improvement = max(
            knn_selection_min_abs_improvement,
            knn_selection_min_rel_improvement * max(abs(float(selected_base_val_mse)), 1.0e-12),
        )
        selected_val_mse_regression = float(selected_hybrid_val_mse) - float(selected_base_val_mse)
        selected_val_mse_regression_pct = (
            100.0 * selected_val_mse_regression / max(abs(float(selected_base_val_mse)), 1.0e-12)
        )
        allowed_val_mse_regression = (
            knn_selection_max_rel_mse_regression * max(abs(float(selected_base_val_mse)), 1.0e-12)
        )
    if selected_base_val_mae is not None and selected_hybrid_val_mae is not None:
        selected_val_mae_improvement = float(selected_base_val_mae) - float(selected_hybrid_val_mae)
        selected_val_mae_improvement_pct = (
            100.0 * selected_val_mae_improvement / max(abs(float(selected_base_val_mae)), 1.0e-12)
        )
        required_val_mae_improvement = max(
            knn_selection_min_abs_improvement,
            knn_selection_min_rel_improvement * max(abs(float(selected_base_val_mae)), 1.0e-12),
        )
    selected_criterion = knn_selection_policy
    selected_selection_policy = knn_selection_policy
    moe_residual_variant = "none"
    if knn_selection_policy == "hybrid":
        if avg_mae_hybrid is not None and avg_mse_hybrid is not None:
            selected_variant = "hybrid"
            selected_avg_mae = avg_mae_hybrid
            selected_avg_mse = avg_mse_hybrid
        else:
            selected_criterion = "hybrid_fallback_base"
    elif (
        knn_selection_policy == "val_mse_margin"
        and selected_base_val_mse is not None
        and selected_hybrid_val_mse is not None
        and avg_mae_hybrid is not None
        and avg_mse_hybrid is not None
        and selected_val_mse_improvement is not None
        and required_val_mse_improvement is not None
        and selected_val_mse_improvement > required_val_mse_improvement
    ):
        selected_variant = "hybrid"
        selected_avg_mae = avg_mae_hybrid
        selected_avg_mse = avg_mse_hybrid
    elif (
        knn_selection_policy == "val_mae_guarded"
        and selected_base_val_mae is not None
        and selected_hybrid_val_mae is not None
        and selected_base_val_mse is not None
        and selected_hybrid_val_mse is not None
        and avg_mae_hybrid is not None
        and avg_mse_hybrid is not None
        and selected_val_mae_improvement is not None
        and required_val_mae_improvement is not None
        and selected_val_mse_regression is not None
        and allowed_val_mse_regression is not None
        and selected_val_mae_improvement > required_val_mae_improvement
        and selected_val_mse_regression <= allowed_val_mse_regression
    ):
        selected_variant = "hybrid"
        selected_avg_mae = avg_mae_hybrid
        selected_avg_mse = avg_mse_hybrid

    if pred_residual_selection_summary is not None:
        moe_residual_variant = (
            "moe_residual_gate"
            if str(pred_residual_selection_summary.get("policy", "")) == "val_mse_gate"
            else "moe_residual_channel"
        )

    if channel_selection_summary is not None:
        selected_variant = "channel_hybrid" if int(channel_selection_summary["num_hybrid_channels"]) > 0 else "base"
        selected_avg_mae = float(channel_selected_avg_mae)
        selected_avg_mse = float(channel_selected_avg_mse)
        selected_criterion = channel_selection_policy
        selected_selection_policy = channel_selection_policy

    if skip_test:
        val_mse_print = None if val_summary is None else val_summary.get("avg_mse")
        val_mae_print = None if val_summary is None else val_summary.get("avg_mae")
        if pred_residual_selection_summary is not None:
            val_mse_print = pred_residual_selection_summary.get("val_scaled_avg_mse", val_mse_print)
            val_mae_print = pred_residual_selection_summary.get("val_scaled_avg_mae", val_mae_print)
        if val_mse_print is not None and val_mae_print is not None:
            print(f"\nValidation-only: avg_MAE={val_mae_print:.6f}, avg_MSE={val_mse_print:.6f}")
            final_print(
                "FINAL_VALIDATION "
                f"selected={selected_variant} "
                f"moe_residual={moe_residual_variant} "
                f"val_MAE={val_mae_print:.6f} "
                f"val_MSE={val_mse_print:.6f} "
                "test_MAE=skipped test_MSE=skipped",
                flush=True,
            )
        else:
            print("\nValidation-only: validation metrics unavailable")
            final_print(
                "FINAL_VALIDATION "
                f"selected={selected_variant} "
                f"moe_residual={moe_residual_variant} "
                "val_MAE=nan val_MSE=nan test_MAE=skipped test_MSE=skipped",
                flush=True,
            )
    elif avg_mae_hybrid is not None and avg_mse_hybrid is not None:
        print(f"\nOverall(base): test_MAE={avg_mae:.6f}, test_MSE={avg_mse:.6f}")
        print(f"Overall(hybrid): test_MAE={avg_mae_hybrid:.6f}, test_MSE={avg_mse_hybrid:.6f}")
        if channel_selection_summary is not None:
            print(
                f"Overall(channel_hybrid): test_MAE={channel_selected_avg_mae:.6f}, "
                f"test_MSE={channel_selected_avg_mse:.6f}, "
                f"hybrid_channels={channel_selection_summary['num_hybrid_channels']}/{C}"
            )
        print(
            f"Overall(selected={selected_variant}, moe_residual={moe_residual_variant}): "
            f"test_MAE={selected_avg_mae:.6f}, test_MSE={selected_avg_mse:.6f}"
        )
        final_print(
            "FINAL_TEST "
            f"selected={selected_variant} "
            f"moe_residual={moe_residual_variant} "
            f"test_MAE={selected_avg_mae:.6f} "
            f"test_MSE={selected_avg_mse:.6f}",
            flush=True,
        )
    else:
        print(
            f"\nOverall(selected={selected_variant}, moe_residual={moe_residual_variant}): "
            f"test_MAE={selected_avg_mae:.6f}, test_MSE={selected_avg_mse:.6f}"
        )
        final_print(
            "FINAL_TEST "
            f"selected={selected_variant} "
            f"moe_residual={moe_residual_variant} "
            f"test_MAE={selected_avg_mae:.6f} "
            f"test_MSE={selected_avg_mse:.6f}",
            flush=True,
        )

    if not skip_test and df is not None:
        df.to_csv(os.path.join(out_dir, "test_metrics.csv"), index=False)
        np.save(os.path.join(out_dir, "test_loss_per_cluster.npy"), test_loss_k.detach().cpu().numpy())

    if (not skip_test) and plot_enable and (plot_idx is not None):
        plot_dir = os.path.join(out_dir, "plots")
        save_channel_plots(
            out_dir=plot_dir,
            channel_names=channel_names,
            plot_cache=plot_cache,
            best_sample=best_sample,
            worst_sample=worst_sample,
            input_len=L,
            pred_len=H,
            dpi=int(plot_cfg["dpi"])
        )
        print(f"Saved plots to: {plot_dir}")

    total_time = time.perf_counter() - t_all0
    avg_epoch_time = sum(epoch_times) / max(len(epoch_times), 1)
    cpu_rss_mb = _get_rss_mb()
    gpu_alloc_mb = -1.0
    gpu_reserved_mb = -1.0
    if device.type == "cuda":
        gpu_alloc_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
        gpu_reserved_mb = float(torch.cuda.max_memory_reserved()) / (1024.0 * 1024.0)
    out_dir_mb = _dir_size_mb(out_dir)

    summary = {
        "config_path": args.config,
        "out_dir": out_dir,
        "penalty_names": list(penalty_names),
        "best_epoch": [int(v) for v in best_epoch.detach().cpu().tolist()],
        "windowing": {
            "past_context": bool(past_context),
            "train_start": 0,
            "val_eval_start": int(val_eval_start),
            "test_eval_start": int(test_eval_start),
            "val_label_start": int(t_train),
            "test_label_start": int(t_val),
            "num_train_windows": int(len(dtr)),
            "num_val_windows": int(len(dva)),
            "num_test_windows": int(len(dte)),
            "normalize_train_only": bool(norm_cfg.get("train_only", False)),
            "data_max_rows": int(max_rows),
        },
        "mae_objective": {
            "enable": bool(mae_objective_enable),
            "kind": str(mae_objective_kind),
            "weight": float(mae_objective_weight_final),
            "warmup_epochs": int(mae_objective_warmup_epochs),
            "beta": float(mae_objective_beta),
        },
        "eval": {
            "skip_test": bool(skip_test),
        },
        "calibration": calibration_summary,
        "moe_residual": moe_residual_summary,
        "moe_residual_selection": pred_residual_selection_summary,
        "moe_residual_gate_calibrator": pred_residual_gate_summary,
        "val": val_summary,
        "test": None if skip_test else {
            "avg_mae": avg_mae,
            "avg_mse": avg_mse,
            "per_cluster_loss": [float(v) for v in test_loss_k.detach().cpu().tolist()],
            "per_cluster_mse": [float(v) for v in test_mse_k.detach().cpu().tolist()],
            "per_cluster_mae": [float(v) for v in test_mae_k.detach().cpu().tolist()],
        },
        "selected": {
            "variant": selected_variant,
            "moe_residual_variant": moe_residual_variant,
            "criterion": selected_criterion,
            "selection_policy": selected_selection_policy,
            "avg_mae": selected_avg_mae,
            "avg_mse": selected_avg_mse,
            "base_val_mse": selected_base_val_mse,
            "hybrid_val_mse": selected_hybrid_val_mse,
            "base_val_mae": selected_base_val_mae,
            "hybrid_val_mae": selected_hybrid_val_mae,
            "val_mse_improvement": selected_val_mse_improvement,
            "val_mse_improvement_pct": selected_val_mse_improvement_pct,
            "required_val_mse_improvement": required_val_mse_improvement,
            "val_mae_improvement": selected_val_mae_improvement,
            "val_mae_improvement_pct": selected_val_mae_improvement_pct,
            "required_val_mae_improvement": required_val_mae_improvement,
            "val_mse_regression": selected_val_mse_regression,
            "val_mse_regression_pct": selected_val_mse_regression_pct,
            "allowed_val_mse_regression": allowed_val_mse_regression,
            "selection_min_abs_improvement": knn_selection_min_abs_improvement,
            "selection_min_rel_improvement": knn_selection_min_rel_improvement,
            "selection_max_rel_mse_regression": knn_selection_max_rel_mse_regression,
        },
        "channel_selection": channel_selection_summary,
        "knn_fusion_calibrator": knn_fusion_summary,
        "knn_gate_calibrator": knn_gate_summary,
        "timing": {
            "total_sec": float(total_time),
            "avg_epoch_sec": float(avg_epoch_time),
        },
        "resources": {
            "cpu_rss_mb": float(cpu_rss_mb),
            "gpu_alloc_mb": float(gpu_alloc_mb),
            "gpu_reserved_mb": float(gpu_reserved_mb),
            "out_dir_size_mb": float(out_dir_mb),
        },
    }
    if knn_hybrid_test is not None and not skip_test:
        summary["knn_hybrid"] = knn_hybrid_test.describe()
    if len(knn_sweep_results) > 0:
        summary["knn_sweep"] = knn_sweep_results
    if finetune_summary is not None:
        summary["finetune"] = finetune_summary
    if val_summary_hybrid is not None:
        summary["val_hybrid"] = val_summary_hybrid
    if val_hybrid_confidence is not None:
        summary["val_hybrid_confidence"] = val_hybrid_confidence
    if (
        avg_mae_hybrid is not None
        and avg_mse_hybrid is not None
        and test_loss_hybrid_k is not None
        and test_mse_hybrid_k is not None
        and test_mae_hybrid_k is not None
    ):
        summary["test_hybrid"] = {
            "avg_mae": avg_mae_hybrid,
            "avg_mse": avg_mse_hybrid,
            "per_cluster_loss": [float(v) for v in test_loss_hybrid_k.detach().cpu().tolist()],
            "per_cluster_mse": [float(v) for v in test_mse_hybrid_k.detach().cpu().tolist()],
            "per_cluster_mae": [float(v) for v in test_mae_hybrid_k.detach().cpu().tolist()],
        }
    if test_hybrid_confidence is not None:
        summary["test_hybrid_confidence"] = test_hybrid_confidence
    summary_path = os.path.join(out_dir, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved run summary to: {summary_path}")

    print("\nTime/Space Summary:")
    print(f"- total_time_s: {total_time:.3f}")
    print(f"- avg_epoch_time_s: {avg_epoch_time:.3f}")
    if cpu_rss_mb >= 0:
        print(f"- cpu_rss_mb: {cpu_rss_mb:.2f}")
    if device.type == "cuda":
        print(f"- gpu_max_alloc_mb: {gpu_alloc_mb:.2f}")
        print(f"- gpu_max_reserved_mb: {gpu_reserved_mb:.2f}")
    print(f"- out_dir_size_mb: {out_dir_mb:.2f}")


if __name__ == "__main__":
    main()
