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
from torch.utils.data import DataLoader, Subset
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
from .data.windows import (
    WindowTensorDataset,
    global_zscore,
    make_label_range_windows,
    make_lazy_label_range_window_dataset,
    make_lazy_strict_window_dataset,
    make_strict_windows,
)
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
from .utils.diagnostic_sampling import select_prediction_sample_indices


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


def _shape_tuple(tensor: torch.Tensor) -> Tuple[int, ...]:
    return tuple(int(v) for v in tensor.shape)


@torch.no_grad()
def _partial_load_matching_state_dict(
    module: nn.Module,
    source_state: Dict[str, torch.Tensor],
    *,
    max_report: int = 40,
) -> Dict[str, object]:
    """Copy only source tensors whose names and shapes match the target module."""
    target_state = module.state_dict()
    loaded: List[str] = []
    skipped_shape: List[Dict[str, object]] = []
    skipped_missing: List[str] = []
    skipped_non_tensor: List[str] = []

    for name, src_tensor in source_state.items():
        if not torch.is_tensor(src_tensor):
            skipped_non_tensor.append(str(name))
            continue
        if name not in target_state:
            skipped_missing.append(str(name))
            continue
        dst_tensor = target_state[name]
        if _shape_tuple(dst_tensor) != _shape_tuple(src_tensor):
            skipped_shape.append(
                {
                    "name": str(name),
                    "source_shape": list(_shape_tuple(src_tensor)),
                    "target_shape": list(_shape_tuple(dst_tensor)),
                }
            )
            continue
        dst_tensor.copy_(src_tensor.to(device=dst_tensor.device, dtype=dst_tensor.dtype))
        loaded.append(str(name))

    return {
        "loaded_count": len(loaded),
        "skipped_shape_count": len(skipped_shape),
        "skipped_missing_count": len(skipped_missing),
        "skipped_non_tensor_count": len(skipped_non_tensor),
        "loaded": loaded[:max_report],
        "skipped_shape": skipped_shape[:max_report],
        "skipped_missing": skipped_missing[:max_report],
        "skipped_non_tensor": skipped_non_tensor[:max_report],
    }

def print_clusters(clusters: Dict[int, List[int]], channel_names: List[str]):
    for k in sorted(clusters.keys()):
        chs = [channel_names[i] for i in clusters[k]]
        print(f"Cluster {k}: [" + ", ".join(chs) + "]")


def _make_torch_generator(seed: Optional[int]) -> Optional[torch.Generator]:
    if seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _freeze_module_params(module: nn.Module) -> int:
    frozen = 0
    for param in module.parameters():
        param.requires_grad_(False)
        frozen += int(param.numel())
    return frozen


def _validation_holdout_split_counts(total: int, holdout_fraction: float, min_holdout: int) -> Tuple[int, int]:
    total = int(total)
    if total <= 0:
        return 0, 0
    holdout_fraction = float(holdout_fraction)
    min_holdout = max(0, int(min_holdout))
    if holdout_fraction <= 0.0:
        return total, 0
    holdout_n = max(min_holdout, int(round(total * holdout_fraction)))
    if holdout_n <= 0 or holdout_n >= total:
        return total, 0
    select_n = total - holdout_n
    if select_n <= 0:
        return total, 0
    return select_n, holdout_n


def _contiguous_segment_ranges(total: int, segment_count: int) -> List[Tuple[int, int]]:
    total = int(total)
    if total <= 0:
        return []
    segment_count = max(1, min(int(segment_count), total))
    base = total // segment_count
    extra = total % segment_count
    ranges: List[Tuple[int, int]] = []
    start = 0
    for idx in range(segment_count):
        length = base + (1 if idx < extra else 0)
        end = start + length
        if end > start:
            ranges.append((start, end))
        start = end
    return ranges


def _top_positive_improvement_mask(improvement_c: torch.Tensor, max_channels: int) -> torch.Tensor:
    improvement_c = improvement_c.detach().cpu()
    positive_c = improvement_c > 0
    max_channels = int(max_channels)
    if max_channels <= 0 or int(positive_c.sum().item()) <= max_channels:
        return positive_c
    scores = torch.where(positive_c, improvement_c, torch.full_like(improvement_c, float("-inf")))
    keep_idx = torch.topk(scores, k=max_channels).indices
    keep_c = torch.zeros_like(positive_c, dtype=torch.bool)
    keep_c[keep_idx] = True
    return keep_c


def _loss_normalization_enabled(cfg: object) -> bool:
    if isinstance(cfg, bool):
        return bool(cfg)
    if not isinstance(cfg, dict):
        return False
    return bool(cfg.get("enable", False))


def _loss_normalization_term_set(cfg: object, names: List[str]) -> set:
    if not isinstance(cfg, dict):
        return set(names)
    raw_terms = cfg.get("terms", "all")
    if raw_terms is None or str(raw_terms).lower() == "all":
        return set(names)
    if isinstance(raw_terms, str):
        return {part.strip() for part in raw_terms.split(",") if part.strip()}
    return {str(part) for part in raw_terms}


def _normalize_loss_terms(
    terms: Dict[str, torch.Tensor],
    cfg: object,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Optional[torch.Tensor]]]:
    if not _loss_normalization_enabled(cfg):
        return dict(terms), {name: None for name in terms}

    cfg_dict = cfg if isinstance(cfg, dict) else {}
    mode = str(cfg_dict.get("mode", "std")).lower()
    eps = float(cfg_dict.get("eps", 1.0e-6))
    min_scale = float(cfg_dict.get("min_scale", eps))
    max_scale = float(cfg_dict.get("max_scale", 0.0))
    include = _loss_normalization_term_set(cfg, list(terms.keys()))

    normalized: Dict[str, torch.Tensor] = {}
    scales: Dict[str, Optional[torch.Tensor]] = {}
    for name, value in terms.items():
        if name not in include:
            normalized[name] = value
            scales[name] = None
            continue
        ref = value.detach().to(dtype=torch.float32)
        if ref.numel() == 0:
            scale = torch.tensor(min_scale, device=value.device, dtype=torch.float32)
        elif mode in {"std", "batch_std"}:
            scale = ref.std(unbiased=False)
        elif mode in {"mean_abs", "abs_mean", "mean"}:
            scale = ref.abs().mean()
        elif mode == "rms":
            scale = ref.pow(2).mean().sqrt()
        else:
            raise ValueError(
                f"Unsupported train.loss_normalization.mode='{mode}'. "
                "Expected std, mean_abs, or rms."
            )
        scale = scale.clamp_min(max(eps, min_scale))
        if max_scale > 0.0:
            scale = scale.clamp_max(max_scale)
        normalized[name] = value / scale.to(device=value.device, dtype=value.dtype)
        scales[name] = scale.detach().cpu()
    return normalized, scales


def _accumulate_detached_sum_(
    accumulator: torch.Tensor,
    values: torch.Tensor,
    *,
    dim: int = 0,
) -> torch.Tensor:
    accumulator.add_(values.detach().sum(dim=dim).to(device=accumulator.device, dtype=accumulator.dtype))
    return accumulator


def _lr_warmup_scale(epoch: int, warmup_epochs: int, start_factor: float = 0.1) -> float:
    warmup_epochs = int(warmup_epochs)
    if warmup_epochs <= 0 or int(epoch) > warmup_epochs:
        return 1.0
    if warmup_epochs == 1:
        return 1.0
    start = min(max(float(start_factor), 0.0), 1.0)
    progress = max(0.0, min(1.0, float(int(epoch) - 1) / float(warmup_epochs - 1)))
    return round(start + (1.0 - start) * progress, 12)


def _set_optimizer_lr_scale(optimizers: List[torch.optim.Optimizer], scale: float) -> None:
    scale = float(scale)
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", float(group["lr"]))
            group["lr"] = float(group["initial_lr"]) * scale


def _should_update_swa(epoch: int, start_epoch: int, update_every: int) -> bool:
    update_every = max(1, int(update_every))
    epoch = int(epoch)
    start_epoch = int(start_epoch)
    return epoch >= start_epoch and ((epoch - start_epoch) % update_every == 0)


def compute_channel_shape_features(data_tc: torch.Tensor, acf_lags: Optional[List[int]] = None) -> torch.Tensor:
    """
    Train-only channel descriptors for feature-aware clustering.
    Returns [C,F], one row per channel. These descriptors use only the
    clustering fit segment and never inspect validation/test targets.
    """
    if data_tc.ndim != 2:
        raise ValueError(f"Expected data_tc [T,C], got {tuple(data_tc.shape)}")
    T, C = data_tc.shape
    x = data_tc.to(dtype=torch.float32)
    device = x.device
    eps = 1.0e-6
    mean_c = x.mean(dim=0)
    std_c = x.std(dim=0).clamp_min(eps)
    centered = x - mean_c.view(1, C)
    t = torch.linspace(-1.0, 1.0, steps=max(T, 1), device=device, dtype=x.dtype).view(T, 1)
    trend_c = (centered * t).mean(dim=0) / t.pow(2).mean().clamp_min(eps)
    range_c = (x.max(dim=0).values - x.min(dim=0).values) / std_c
    q95_c = torch.quantile(x, 0.95, dim=0)
    q05_c = torch.quantile(x, 0.05, dim=0)
    iqr90_c = (q95_c - q05_c) / std_c
    if T > 1:
        d1 = x[1:] - x[:-1]
        d1_abs_c = d1.abs().mean(dim=0) / std_c
        d1_std_c = d1.std(dim=0) / std_c
        jump_thr = d1.abs().mean(dim=0, keepdim=True) + 2.0 * d1.std(dim=0, keepdim=True).clamp_min(eps)
        jump_rate_c = (d1.abs() > jump_thr).to(dtype=x.dtype).mean(dim=0)
    else:
        d1_abs_c = torch.zeros(C, device=device, dtype=x.dtype)
        d1_std_c = torch.zeros_like(d1_abs_c)
        jump_rate_c = torch.zeros_like(d1_abs_c)
    if T > 2:
        d2 = x[2:] - (2.0 * x[1:-1]) + x[:-2]
        d2_abs_c = d2.abs().mean(dim=0) / std_c
        curvature_c = d2.abs().mean(dim=0) / (d1_abs_c * std_c + eps)
    else:
        d2_abs_c = torch.zeros(C, device=device, dtype=x.dtype)
        curvature_c = torch.zeros_like(d2_abs_c)
    feats = [
        mean_c,
        std_c,
        trend_c,
        range_c,
        iqr90_c,
        d1_abs_c,
        d1_std_c,
        d2_abs_c,
        curvature_c,
        jump_rate_c,
    ]
    lags = acf_lags if acf_lags is not None else [1, 24, 96]
    for lag in lags:
        lag = int(lag)
        if lag <= 0 or T <= lag + 1:
            feats.append(torch.zeros(C, device=device, dtype=x.dtype))
            continue
        a = centered[:-lag]
        b = centered[lag:]
        cov = (a * b).mean(dim=0)
        var = centered.pow(2).mean(dim=0).clamp_min(eps)
        feats.append((cov / var).clamp(-1.0, 1.0))
    return torch.stack(feats, dim=1)


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


def build_topk_penalty_mask(prior_kp: Optional[torch.Tensor], topk: int) -> Optional[torch.Tensor]:
    if prior_kp is None or prior_kp.numel() == 0:
        return None
    K, P = prior_kp.shape
    if P <= 0:
        return torch.zeros_like(prior_kp)
    k = max(1, min(int(topk), P))
    idx = prior_kp.topk(k=k, dim=-1).indices
    mask = torch.zeros_like(prior_kp, dtype=torch.float32)
    mask.scatter_(-1, idx, 1.0)
    return mask


def build_named_penalty_mask(
    allowed_by_cluster,
    penalty_names: List[str],
    K: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if allowed_by_cluster is None:
        return None
    name_to_idx = {str(name): i for i, name in enumerate(penalty_names)}
    mask = torch.zeros((int(K), len(penalty_names)), device=device, dtype=torch.float32)
    if isinstance(allowed_by_cluster, dict):
        iterator = allowed_by_cluster.items()
    else:
        iterator = enumerate(allowed_by_cluster)
    for raw_k, raw_names in iterator:
        k = int(raw_k)
        if k < 0 or k >= int(K):
            raise ValueError(f"cluster_penalty_prior.allowed_by_cluster has invalid cluster id {k}.")
        if isinstance(raw_names, str):
            names = [raw_names]
        else:
            names = list(raw_names or [])
        for raw_name in names:
            name = str(raw_name)
            if name not in name_to_idx:
                raise ValueError(
                    "cluster_penalty_prior.allowed_by_cluster contains unknown penalty "
                    f"{name!r}; available={penalty_names}"
                )
            mask[k, name_to_idx[name]] = 1.0
    empty = mask.sum(dim=-1, keepdim=True) <= 0.0
    if bool(empty.any().item()):
        mask = torch.where(empty, torch.ones_like(mask), mask)
    return mask


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


@torch.no_grad()
def compute_channel_penalty_portrait(
    loader: DataLoader,
    penalty_names: List[str],
    penalty_fns: Dict[str, callable],
    num_channels: int,
    pred_len: int,
    device: torch.device,
) -> torch.Tensor:
    if len(penalty_names) == 0 or len(loader) == 0:
        return torch.zeros((num_channels, len(penalty_names)), device=device)
    sum_cp = torch.zeros(num_channels, len(penalty_names), device=device)
    cnt = 0
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
        sum_cp += pen_bcp.sum(dim=0)
        cnt += int(pen_bcp.shape[0])
    if cnt == 0:
        return torch.zeros((num_channels, len(penalty_names)), device=device)
    return sum_cp / float(cnt)


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
    hard = hard * (probs_bkp > 0).to(dtype=hard.dtype)
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


def _mae_objective_weight_is_nonzero(weight) -> bool:
    if torch.is_tensor(weight):
        if weight.numel() == 0:
            return False
        return bool(weight.detach().abs().max().item() != 0.0)
    return bool(float(weight) != 0.0)


def _apply_mae_objective_weight(mae_objective_bk: torch.Tensor, weight) -> torch.Tensor:
    if torch.is_tensor(weight):
        weight_k = weight.detach().to(device=mae_objective_bk.device, dtype=mae_objective_bk.dtype)
        if weight_k.dim() != 1 or int(weight_k.numel()) != int(mae_objective_bk.shape[1]):
            raise ValueError(
                "Per-cluster mae_objective weight must be a [K] tensor matching loss_bk clusters."
            )
        return mae_objective_bk * weight_k.view(1, -1)
    return float(weight) * mae_objective_bk


def _scale_mae_objective_weight(base_weight: float, multiplier_k: Optional[torch.Tensor]):
    if multiplier_k is None:
        return float(base_weight)
    return multiplier_k.detach().to(dtype=torch.float32) * float(base_weight)


def _cluster_target_stats_from_targets(
    targets_bch: torch.Tensor,
    cluster_id_c: torch.Tensor,
    K: int,
) -> List[Dict[str, float]]:
    targets_bch = targets_bch.detach().cpu().to(dtype=torch.float32)
    cluster_id_c = cluster_id_c.detach().cpu().to(dtype=torch.long)
    rows: List[Dict[str, float]] = []
    for k in range(int(K)):
        idx = (cluster_id_c == k).nonzero(as_tuple=False).view(-1)
        if int(idx.numel()) == 0:
            rows.append(
                {
                    "cluster_id": int(k),
                    "channels": 0,
                    "mean": 0.0,
                    "median": 0.0,
                    "std": 0.0,
                    "gap": 0.0,
                }
            )
            continue
        values = targets_bch.index_select(1, idx).reshape(-1)
        mean = values.mean()
        median = values.median()
        std = values.std(unbiased=False).clamp_min(1.0e-6)
        gap = (mean - median).abs() / std
        rows.append(
            {
                "cluster_id": int(k),
                "channels": int(idx.numel()),
                "mean": float(mean.item()),
                "median": float(median.item()),
                "std": float(std.item()),
                "gap": float(gap.item()),
            }
        )
    return rows


def _gap_multiplier_from_rows(
    rows: List[Dict[str, float]],
    cfg: Dict[str, object],
) -> torch.Tensor:
    gaps = torch.tensor([float(row["gap"]) for row in rows], dtype=torch.float32)
    min_multiplier = float(cfg.get("min_multiplier", 1.0))
    max_multiplier = float(cfg.get("max_multiplier", 1.25))
    if max_multiplier < min_multiplier:
        raise ValueError("train.mae_objective.per_cluster.max_multiplier must be >= min_multiplier.")
    pivot_cfg = cfg.get("pivot", "median")
    if isinstance(pivot_cfg, str):
        pivot_mode = pivot_cfg.lower()
        if pivot_mode == "median":
            pivot = gaps.median()
        elif pivot_mode in {"mean", "avg"}:
            pivot = gaps.mean()
        elif pivot_mode in {"zero", "none"}:
            pivot = torch.tensor(0.0, dtype=gaps.dtype)
        else:
            raise ValueError("train.mae_objective.per_cluster.pivot must be median, mean, zero, or a number.")
    else:
        pivot = torch.tensor(float(pivot_cfg), dtype=gaps.dtype)
    excess = (gaps - pivot).clamp_min(0.0)
    denom = excess.max().clamp_min(1.0e-12)
    normalized = torch.where(denom > 0.0, excess / denom, torch.zeros_like(excess))
    multiplier = 1.0 + (max_multiplier - 1.0) * normalized
    multiplier = multiplier.clamp(min=min_multiplier, max=max_multiplier)
    multiplier = torch.where(excess > 0.0, multiplier, torch.ones_like(multiplier).clamp(min=min_multiplier))
    return multiplier.detach()


def _build_mae_per_cluster_diagnostics_from_targets(
    targets_bch: torch.Tensor,
    cluster_id_c: torch.Tensor,
    K: int,
    base_weight: float,
    cfg: Dict[str, object],
) -> Dict[str, object]:
    if str(cfg.get("diagnostic", "mean_median_gap")).lower() != "mean_median_gap":
        raise ValueError("train.mae_objective.per_cluster.diagnostic currently supports mean_median_gap only.")
    if str(cfg.get("source", "train_targets")).lower() != "train_targets":
        raise ValueError("train.mae_objective.per_cluster.source currently supports train_targets only.")
    if str(cfg.get("normalize", "std")).lower() != "std":
        raise ValueError("train.mae_objective.per_cluster.normalize currently supports std only.")
    rows = _cluster_target_stats_from_targets(targets_bch, cluster_id_c, K)
    multiplier_k = _gap_multiplier_from_rows(rows, cfg)
    effective_weight_k = multiplier_k * float(base_weight)
    for k, row in enumerate(rows):
        row["multiplier"] = float(multiplier_k[k].item())
        row["base_weight"] = float(base_weight)
        row["effective_weight"] = float(effective_weight_k[k].item())
    return {
        "rows": rows,
        "multiplier_k": multiplier_k.detach(),
        "effective_weight_k": effective_weight_k.detach(),
    }


def _collect_train_targets_bch(loader: DataLoader, max_windows: int = 0) -> torch.Tensor:
    parts = []
    seen = 0
    limit = int(max_windows or 0)
    for _, y, _ in loader:
        if limit > 0:
            remaining = max(0, limit - seen)
            if remaining <= 0:
                break
            y = y[:remaining]
        parts.append(y.detach().cpu())
        seen += int(y.shape[0])
        if limit > 0 and seen >= limit:
            break
    if len(parts) == 0:
        raise ValueError("Cannot build per-cluster MAE diagnostics from an empty train loader.")
    return torch.cat(parts, dim=0)


def _save_mae_per_cluster_diagnostics_csv(rows: List[Dict[str, float]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _cosine_similarity_matrix(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().cpu().to(dtype=torch.float32)
    denom = x.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
    x_norm = x / denom
    return x_norm @ x_norm.t()


def _write_matrix_csv(matrix: torch.Tensor, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    matrix = matrix.detach().cpu()
    columns = ["cluster_id"] + [f"cluster_{i}" for i in range(int(matrix.shape[1]))]
    rows = []
    for i in range(int(matrix.shape[0])):
        rows.append([i] + [float(v) for v in matrix[i].tolist()])
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def _save_cluster_embedding_artifacts(model: nn.Module, out_dir: str) -> Dict[str, object]:
    for module in model.modules():
        diag_fn = getattr(module, "cluster_embedding_diagnostics", None)
        if diag_fn is None:
            continue
        diag = diag_fn()
        if diag is None:
            continue
        embedding = diag["embedding"].detach().cpu().to(dtype=torch.float32)
        gamma = diag["gamma"].detach().cpu().to(dtype=torch.float32)
        beta = diag["beta"].detach().cpu().to(dtype=torch.float32)

        embedding_path = os.path.join(out_dir, "cluster_embedding.csv")
        embedding_sim_path = os.path.join(out_dir, "cluster_embedding_similarity.csv")
        film_norm_path = os.path.join(out_dir, "cluster_film_norms.csv")
        gamma_sim_path = os.path.join(out_dir, "cluster_film_gamma_similarity.csv")
        beta_sim_path = os.path.join(out_dir, "cluster_film_beta_similarity.csv")

        emb_rows = []
        for k in range(int(embedding.shape[0])):
            row = {"cluster_id": int(k)}
            row.update({f"e{i}": float(v) for i, v in enumerate(embedding[k].tolist())})
            emb_rows.append(row)
        pd.DataFrame(emb_rows).to_csv(embedding_path, index=False)
        _write_matrix_csv(_cosine_similarity_matrix(embedding), embedding_sim_path)

        norm_rows = []
        for k in range(int(gamma.shape[0])):
            norm_rows.append(
                {
                    "cluster_id": int(k),
                    "gamma_norm": float(gamma[k].norm().item()),
                    "beta_norm": float(beta[k].norm().item()),
                    "gamma_mean_abs": float(gamma[k].abs().mean().item()),
                    "beta_mean_abs": float(beta[k].abs().mean().item()),
                }
            )
        pd.DataFrame(norm_rows).to_csv(film_norm_path, index=False)
        _write_matrix_csv(_cosine_similarity_matrix(gamma), gamma_sim_path)
        _write_matrix_csv(_cosine_similarity_matrix(beta), beta_sim_path)
        return {
            "enable": True,
            "cluster_embedding": embedding_path,
            "cluster_embedding_similarity": embedding_sim_path,
            "cluster_film_norms": film_norm_path,
            "cluster_film_gamma_similarity": gamma_sim_path,
            "cluster_film_beta_similarity": beta_sim_path,
        }
    return {"enable": False}


def _pred_residual_channel_keep_mask(
    policy: str,
    base_mse_c: torch.Tensor,
    cand_mse_c: torch.Tensor,
    base_mae_c: torch.Tensor,
    cand_mae_c: torch.Tensor,
    *,
    min_abs_improvement: float = 0.0,
    min_rel_improvement: float = 0.0,
    max_abs_mse_regression: float = 0.0,
    max_rel_mse_regression: float = 0.0,
) -> torch.Tensor:
    policy = str(policy).lower()
    if policy in {"val_mse_channel", "val_mse_scale", "val_mse_scale_holdout", "val_mse_gate", "val_mse_gate_guarded"}:
        required = torch.maximum(
            torch.full_like(base_mse_c, float(min_abs_improvement)),
            float(min_rel_improvement) * base_mse_c.abs().clamp_min(1.0e-12),
        )
        return (base_mse_c - cand_mse_c) > required
    if policy == "val_mae_gate_guarded":
        required_mae = torch.maximum(
            torch.full_like(base_mae_c, float(min_abs_improvement)),
            float(min_rel_improvement) * base_mae_c.abs().clamp_min(1.0e-12),
        )
        allowed_mse_regression = torch.maximum(
            torch.full_like(base_mse_c, max(0.0, float(max_abs_mse_regression))),
            max(0.0, float(max_rel_mse_regression)) * base_mse_c.abs().clamp_min(1.0e-12),
        )
        improves_mae = (base_mae_c - cand_mae_c) > required_mae
        respects_mse = (cand_mse_c - base_mse_c) <= allowed_mse_regression
        return improves_mae & respects_mse
    raise ValueError(f"Unsupported residual selection policy for channel keep mask: {policy}")


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


def _parse_positive_ints(raw) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, str):
        values = []
        for part in raw.split(","):
            part = part.strip()
            if part:
                value = int(part)
                if value > 0:
                    values.append(value)
        return values
    if isinstance(raw, (int, float)):
        value = int(raw)
        return [value] if value > 0 else []
    values = []
    for item in raw:
        value = int(item)
        if value > 0:
            values.append(value)
    return values


def history_anchor_enabled(cfg: Optional[dict]) -> bool:
    cfg = cfg or {}
    alpha_by_channel = cfg.get("alpha_by_channel", None)
    if alpha_by_channel is not None:
        alpha_enabled = any(float(v) > 0.0 for v in alpha_by_channel)
    else:
        alpha_enabled = float(cfg.get("alpha", 0.0) or 0.0) > 0.0
    return (
        bool(cfg.get("enable", False))
        and len(_parse_positive_ints(cfg.get("lags", ()))) > 0
        and alpha_enabled
    )


def _normalize_history_anchor_cfg(cfg: Optional[dict]) -> dict:
    out = dict(cfg or {})
    if history_anchor_enabled(out) and "history_scope" not in out:
        out["history_scope"] = "input_window"
    return out


def _validate_strict_history_anchor_scope(cfg: Optional[dict], *, source: str) -> None:
    cfg = cfg or {}
    if not history_anchor_enabled(cfg):
        return
    if bool(cfg.get("allow_all_observed", False)):
        return
    history_scope = str(cfg.get("history_scope", "input_window")).lower()
    if history_scope != "input_window":
        raise ValueError(
            f"{source}.history_scope must be 'input_window' for strict input-window training; "
            f"got {history_scope!r}. Set {source}.allow_all_observed=true only for oracle diagnostics."
        )


def _history_anchor_values(
    observed_history_tc: torch.Tensor,
    query_start_abs_b: torch.Tensor,
    *,
    input_len: int,
    pred_len: int,
    channel_count: int,
    lags: List[int],
    device: torch.device,
    dtype: torch.dtype,
    history_scope: str = "input_window",
) -> Tuple[torch.Tensor, torch.Tensor]:
    observed = observed_history_tc.detach().to(device=device, dtype=dtype)
    if observed.ndim != 2:
        raise ValueError("history_anchor observed history must have shape [time, channel].")
    if int(observed.shape[1]) != int(channel_count):
        raise ValueError("history_anchor observed history channel count must match predictions.")
    history_scope = str(history_scope).lower()
    if history_scope not in {"all_observed", "input_window"}:
        raise ValueError("history_anchor.history_scope must be 'all_observed' or 'input_window'.")

    starts = query_start_abs_b.detach().to(device=device, dtype=torch.long).reshape(1, -1, 1)
    steps = torch.arange(int(pred_len), device=device, dtype=torch.long).view(1, 1, -1)
    lag_t = torch.as_tensor(lags, device=device, dtype=torch.long).view(-1, 1, 1)
    forecast_start = starts + int(input_len)
    idx_lbh = forecast_start + steps - lag_t
    valid_lbh = (
        (idx_lbh >= 0)
        & (idx_lbh < forecast_start)
        & (idx_lbh < int(observed.shape[0]))
    )
    if history_scope == "input_window":
        valid_lbh = valid_lbh & (idx_lbh >= starts)
    idx_lbh = idx_lbh.clamp(min=0, max=max(int(observed.shape[0]) - 1, 0))
    values_lbhc = observed.index_select(0, idx_lbh.reshape(-1)).view(
        int(lag_t.shape[0]),
        int(query_start_abs_b.numel()),
        int(pred_len),
        int(channel_count),
    )
    values_bchl = values_lbhc.permute(1, 3, 2, 0)
    valid_bh1l = valid_lbh.permute(1, 2, 0).unsqueeze(2).to(dtype=dtype)
    valid_bchl = valid_bh1l.permute(0, 2, 1, 3)
    count_b1h = valid_bh1l.sum(dim=-1).permute(0, 2, 1).clamp_min(1.0)
    anchor_bch = (values_bchl * valid_bchl).sum(dim=-1) / count_b1h
    mask_b1h = valid_bh1l.sum(dim=-1).permute(0, 2, 1) > 0
    return anchor_bch, mask_b1h


def apply_history_anchor_adapter(
    pred_bch: torch.Tensor,
    *,
    base_pred_bch: torch.Tensor,
    observed_history_tc: Optional[torch.Tensor],
    query_start_abs_b: torch.Tensor,
    input_len: int,
    cfg: Optional[dict],
) -> torch.Tensor:
    cfg = cfg or {}
    if not history_anchor_enabled(cfg):
        return pred_bch
    if observed_history_tc is None:
        raise ValueError("model.history_anchor requires observed history.")
    lags = _parse_positive_ints(cfg.get("lags", ()))
    blend_target = str(cfg.get("blend_target", "prediction")).lower()
    if blend_target not in {"prediction", "base"}:
        raise ValueError("model.history_anchor.blend_target must be 'prediction' or 'base'.")

    anchor_bch, mask_b1h = _history_anchor_values(
        observed_history_tc,
        query_start_abs_b,
        input_len=int(input_len),
        pred_len=int(pred_bch.shape[-1]),
        channel_count=int(pred_bch.shape[1]),
        lags=lags,
        device=pred_bch.device,
        dtype=pred_bch.dtype,
        history_scope=str(cfg.get("history_scope", "input_window")),
    )
    alpha_by_channel = cfg.get("alpha_by_channel", None)
    if alpha_by_channel is not None:
        alpha_values = [float(v) for v in alpha_by_channel]
        if len(alpha_values) != int(pred_bch.shape[1]):
            raise ValueError(
                "model.history_anchor.alpha_by_channel length must match the channel count "
                f"({len(alpha_values)} != {int(pred_bch.shape[1])})."
            )
        alpha = torch.as_tensor(alpha_values, device=pred_bch.device, dtype=pred_bch.dtype).view(1, -1, 1)
    else:
        alpha = float(cfg.get("alpha", 0.0) or 0.0)
    if blend_target == "prediction":
        blended = pred_bch + alpha * (anchor_bch - pred_bch)
    else:
        blended = pred_bch + alpha * (anchor_bch - base_pred_bch)
    return torch.where(mask_b1h.to(device=pred_bch.device), blended, pred_bch)


def apply_moe_history_anchor_expert(
    pred_bch: torch.Tensor,
    *,
    base_pred_bch: torch.Tensor,
    observed_history_tc: Optional[torch.Tensor],
    query_start_abs_b: torch.Tensor,
    input_len: int,
    cfg: Optional[dict],
) -> torch.Tensor:
    cfg = cfg or {}
    if not bool(cfg.get("enable", False)):
        return pred_bch
    expert_cfg = dict(cfg)
    expert_cfg["enable"] = True
    expert_cfg = _normalize_history_anchor_cfg(expert_cfg)
    return apply_history_anchor_adapter(
        pred_bch,
        base_pred_bch=base_pred_bch,
        observed_history_tc=observed_history_tc,
        query_start_abs_b=query_start_abs_b,
        input_len=int(input_len),
        cfg=expert_cfg,
    )


def build_train_phase_anchor_table(
    data_tc: torch.Tensor,
    *,
    train_end: int,
    period: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if int(period) <= 0:
        raise ValueError("train_stat_anchor_expert.period must be positive.")
    train_end = max(0, min(int(train_end), int(data_tc.shape[0])))
    if train_end <= 0:
        raise ValueError("train_stat_anchor_expert requires non-empty train data.")
    train = data_tc[:train_end].detach()
    period = int(period)
    table = torch.zeros(period, int(train.shape[1]), dtype=train.dtype, device=train.device)
    counts = torch.zeros(period, dtype=torch.long, device=train.device)
    phases = torch.arange(train_end, device=train.device, dtype=torch.long) % period
    table.index_add_(0, phases, train)
    counts.index_add_(0, phases, torch.ones(train_end, dtype=torch.long, device=train.device))
    global_mean = train.mean(dim=0)
    nonempty = counts > 0
    if bool(nonempty.any()):
        table[nonempty] = table[nonempty] / counts[nonempty].to(dtype=train.dtype).unsqueeze(-1)
    if bool((~nonempty).any()):
        table[~nonempty] = global_mean
    return table, counts


def build_train_phase_delta_anchor_table(
    data_tc: torch.Tensor,
    *,
    train_end: int,
    input_len: int,
    pred_len: int,
    period: int,
    reference: str = "last",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if int(period) <= 0:
        raise ValueError("train_stat_anchor_expert.period must be positive.")
    reference = str(reference).lower()
    if reference not in {"last", "repeat"}:
        raise ValueError("train_stat_anchor_expert.reference must be 'last' or 'repeat'.")
    train_end = max(0, min(int(train_end), int(data_tc.shape[0])))
    input_len = int(input_len)
    pred_len = int(pred_len)
    period = int(period)
    n_windows = train_end - input_len - pred_len + 1
    if n_windows <= 0:
        raise ValueError("train_stat_anchor_expert phase_delta requires at least one full train window.")
    data = data_tc.detach()
    table = torch.zeros(period, pred_len, int(data.shape[1]), dtype=data.dtype, device=data.device)
    counts = torch.zeros(period, dtype=torch.long, device=data.device)
    global_sum = torch.zeros(pred_len, int(data.shape[1]), dtype=data.dtype, device=data.device)
    for start in range(n_windows):
        forecast_start = start + input_len
        phase = int(forecast_start % period)
        target_hc = data[forecast_start : forecast_start + pred_len]
        if reference == "last":
            ref_hc = data[forecast_start - 1].view(1, -1).expand(pred_len, -1)
        else:
            pos_h = torch.arange(pred_len, device=data.device, dtype=torch.long) % input_len
            ref_hc = data[start : start + input_len].index_select(0, pos_h)
        delta_hc = target_hc - ref_hc
        table[phase] += delta_hc
        counts[phase] += 1
        global_sum += delta_hc
    global_delta = global_sum / float(n_windows)
    nonempty = counts > 0
    if bool(nonempty.any()):
        table[nonempty] = table[nonempty] / counts[nonempty].to(dtype=data.dtype).view(-1, 1, 1)
    if bool((~nonempty).any()):
        table[~nonempty] = global_delta
    return table, counts


def build_train_phase_residual_anchor_table(
    base_pred_nch: torch.Tensor,
    target_nch: torch.Tensor,
    *,
    query_start_abs_n: torch.Tensor,
    input_len: int,
    period: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if int(period) <= 0:
        raise ValueError("train_residual_anchor_expert.period must be positive.")
    if base_pred_nch.shape != target_nch.shape:
        raise ValueError("base_pred and target must have the same [window, channel, horizon] shape.")
    if base_pred_nch.ndim != 3:
        raise ValueError("base_pred and target must have shape [window, channel, horizon].")
    starts = query_start_abs_n.detach().to(dtype=torch.long).reshape(-1)
    if int(starts.numel()) != int(base_pred_nch.shape[0]):
        raise ValueError("query_start_abs_n length must match the number of windows.")
    period = int(period)
    residual_nhc = (target_nch.detach() - base_pred_nch.detach()).permute(0, 2, 1).contiguous()
    horizon = int(residual_nhc.shape[1])
    channel_count = int(residual_nhc.shape[2])
    table = torch.zeros(period, horizon, channel_count, dtype=residual_nhc.dtype, device=residual_nhc.device)
    counts = torch.zeros(period, dtype=torch.long, device=residual_nhc.device)
    phases = (starts.to(device=residual_nhc.device) + int(input_len)) % period
    table.index_add_(0, phases, residual_nhc)
    counts.index_add_(0, phases, torch.ones_like(phases, dtype=torch.long, device=residual_nhc.device))
    global_mean = residual_nhc.mean(dim=0)
    nonempty = counts > 0
    if bool(nonempty.any()):
        table[nonempty] = table[nonempty] / counts[nonempty].to(dtype=residual_nhc.dtype).view(-1, 1, 1)
    if bool((~nonempty).any()):
        table[~nonempty] = global_mean
    return table, counts


def build_train_stat_anchor_from_config(
    data_tc: torch.Tensor,
    *,
    train_end: int,
    input_len: int,
    pred_len: int,
    cfg: Optional[dict],
    prefix: str = "moe.train_stat_anchor_expert",
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, object]]:
    cfg = cfg or {}
    summary: Dict[str, object] = {"enable": bool(cfg.get("enable", False))}
    if not bool(cfg.get("enable", False)):
        return None, None, summary

    period = int(cfg.get("period", 96))
    mode = str(cfg.get("mode", "phase_mean")).lower()
    reference = str(cfg.get("reference", "last")).lower()
    if mode == "phase_delta":
        table, counts = build_train_phase_delta_anchor_table(
            data_tc,
            train_end=int(train_end),
            input_len=int(input_len),
            pred_len=int(pred_len),
            period=period,
            reference=reference,
        )
    elif mode == "phase_mean":
        table, counts = build_train_phase_anchor_table(
            data_tc,
            train_end=int(train_end),
            period=period,
        )
    else:
        raise ValueError(f"{prefix}.mode must be 'phase_mean' or 'phase_delta'.")

    summary.update(
        {
            "period": int(period),
            "mode": str(mode),
            "reference": str(reference),
            "source_split": "train",
            "train_end": int(train_end),
            "min_count": int(counts.min().item()),
            "max_count": int(counts.max().item()),
            "alpha": float(cfg.get("alpha", 0.0) or 0.0),
            "blend_target": str(cfg.get("blend_target", "prediction")),
        }
    )
    return table, counts, summary


def _anchor_alpha_from_cfg(
    cfg: dict,
    *,
    channel_count: int,
    horizon: int,
    device: torch.device,
    dtype: torch.dtype,
    prefix: str,
) -> Tuple[float | torch.Tensor, bool]:
    alpha_by_channel_horizon = cfg.get("alpha_by_channel_horizon", None)
    alpha_by_channel = cfg.get("alpha_by_channel", None)
    if alpha_by_channel_horizon is not None:
        rows = [[float(v) for v in row] for row in alpha_by_channel_horizon]
        if len(rows) != int(channel_count):
            raise ValueError(
                f"{prefix}.alpha_by_channel_horizon row count must match the channel count "
                f"({len(rows)} != {int(channel_count)})."
            )
        segments = int(cfg.get("alpha_horizon_segments", len(rows[0]) if rows else 0))
        if segments <= 0:
            raise ValueError(f"{prefix}.alpha_horizon_segments must be positive.")
        if any(len(row) != segments for row in rows):
            raise ValueError("Each alpha_by_channel_horizon row must match alpha_horizon_segments.")
        scale_cs = torch.as_tensor(rows, device=device, dtype=dtype)
        seg_idx_h = torch.div(
            torch.arange(int(horizon), device=device, dtype=torch.long) * segments,
            max(int(horizon), 1),
            rounding_mode="floor",
        ).clamp_max(segments - 1)
        alpha: float | torch.Tensor = scale_cs.index_select(1, seg_idx_h).view(1, int(channel_count), int(horizon))
        return alpha, any(v > 0.0 for row in rows for v in row)
    if alpha_by_channel is not None:
        alpha_values = [float(v) for v in alpha_by_channel]
        if len(alpha_values) != int(channel_count):
            raise ValueError(
                f"{prefix}.alpha_by_channel length must match the channel count "
                f"({len(alpha_values)} != {int(channel_count)})."
            )
        alpha = torch.as_tensor(alpha_values, device=device, dtype=dtype).view(1, -1, 1)
        return alpha, any(v > 0.0 for v in alpha_values)
    alpha = float(cfg.get("alpha", 0.0) or 0.0)
    return alpha, alpha > 0.0


def apply_train_residual_anchor_expert(
    pred_bch: torch.Tensor,
    *,
    base_pred_bch: torch.Tensor,
    query_start_abs_b: torch.Tensor,
    input_len: int,
    residual_anchor_phc: Optional[torch.Tensor],
    cfg: Optional[dict],
) -> torch.Tensor:
    cfg = cfg or {}
    if not bool(cfg.get("enable", False)):
        return pred_bch
    if residual_anchor_phc is None:
        raise ValueError("moe.train_residual_anchor_expert requires a train residual anchor table.")
    alpha, alpha_active = _anchor_alpha_from_cfg(
        cfg,
        channel_count=int(pred_bch.shape[1]),
        horizon=int(pred_bch.shape[-1]),
        device=pred_bch.device,
        dtype=pred_bch.dtype,
        prefix="moe.train_residual_anchor_expert",
    )
    if not alpha_active:
        return pred_bch
    table = residual_anchor_phc.detach().to(device=pred_bch.device, dtype=pred_bch.dtype)
    if table.ndim != 3 or int(table.shape[1]) != int(pred_bch.shape[-1]) or int(table.shape[2]) != int(pred_bch.shape[1]):
        raise ValueError("train_residual_anchor_expert table must have shape [period, horizon, channel].")
    blend_target = str(cfg.get("blend_target", "prediction")).lower()
    if blend_target not in {"prediction", "base"}:
        raise ValueError("moe.train_residual_anchor_expert.blend_target must be 'prediction' or 'base'.")
    phases_b = (query_start_abs_b.detach().to(device=pred_bch.device, dtype=torch.long) + int(input_len)) % int(table.shape[0])
    residual_bhc = table.index_select(0, phases_b).view(
        int(pred_bch.shape[0]),
        int(pred_bch.shape[-1]),
        int(pred_bch.shape[1]),
    )
    residual_bch = residual_bhc.permute(0, 2, 1).contiguous()
    if blend_target == "prediction":
        return pred_bch + alpha * residual_bch
    return pred_bch + alpha * (base_pred_bch + residual_bch - base_pred_bch)


def apply_train_stat_input_centering(
    x_bcl: torch.Tensor,
    *,
    query_start_abs_b: torch.Tensor,
    stat_anchor_pc: Optional[torch.Tensor],
    cfg: Optional[dict],
) -> torch.Tensor:
    cfg = cfg or {}
    if not (bool(cfg.get("enable", False)) and bool(cfg.get("input_center", False))):
        return x_bcl
    if stat_anchor_pc is None:
        raise ValueError("model.train_stat_adapter input_center requires a train phase anchor table.")
    mode = str(cfg.get("mode", "phase_mean")).lower()
    if mode != "phase_mean":
        raise ValueError("model.train_stat_adapter input_center currently requires mode='phase_mean'.")
    table = stat_anchor_pc.detach().to(device=x_bcl.device, dtype=x_bcl.dtype)
    if table.ndim != 2 or int(table.shape[1]) != int(x_bcl.shape[1]):
        raise ValueError("model.train_stat_adapter phase_mean table must have shape [period, channel].")
    starts = query_start_abs_b.detach().to(device=x_bcl.device, dtype=torch.long).view(-1, 1)
    steps = torch.arange(int(x_bcl.shape[-1]), device=x_bcl.device, dtype=torch.long).view(1, -1)
    phases_bl = (starts + steps) % int(table.shape[0])
    anchor_blc = table.index_select(0, phases_bl.reshape(-1)).view(
        int(x_bcl.shape[0]),
        int(x_bcl.shape[-1]),
        int(x_bcl.shape[1]),
    )
    scale = float(cfg.get("input_center_scale", 1.0) or 0.0)
    return x_bcl - scale * anchor_blc.permute(0, 2, 1).contiguous()


def apply_train_stat_anchor_expert(
    pred_bch: torch.Tensor,
    *,
    base_pred_bch: torch.Tensor,
    x_bcl: Optional[torch.Tensor] = None,
    query_start_abs_b: torch.Tensor,
    input_len: int,
    stat_anchor_pc: Optional[torch.Tensor],
    cfg: Optional[dict],
) -> torch.Tensor:
    cfg = cfg or {}
    if not bool(cfg.get("enable", False)):
        return pred_bch
    if stat_anchor_pc is None:
        raise ValueError("moe.train_stat_anchor_expert requires a train phase anchor table.")
    alpha_by_channel_horizon = cfg.get("alpha_by_channel_horizon", None)
    alpha_by_channel = cfg.get("alpha_by_channel", None)
    if alpha_by_channel_horizon is not None:
        rows = [[float(v) for v in row] for row in alpha_by_channel_horizon]
        if len(rows) != int(pred_bch.shape[1]):
            raise ValueError(
                "moe.train_stat_anchor_expert.alpha_by_channel_horizon row count must match the channel count "
                f"({len(rows)} != {int(pred_bch.shape[1])})."
            )
        segments = int(cfg.get("alpha_horizon_segments", len(rows[0]) if rows else 0))
        if segments <= 0:
            raise ValueError("moe.train_stat_anchor_expert.alpha_horizon_segments must be positive.")
        if any(len(row) != segments for row in rows):
            raise ValueError("Each alpha_by_channel_horizon row must match alpha_horizon_segments.")
        scale_cs = torch.as_tensor(rows, device=pred_bch.device, dtype=pred_bch.dtype)
        horizon = int(pred_bch.shape[-1])
        seg_idx_h = torch.div(
            torch.arange(horizon, device=pred_bch.device, dtype=torch.long) * segments,
            max(horizon, 1),
            rounding_mode="floor",
        ).clamp_max(segments - 1)
        alpha: float | torch.Tensor = scale_cs.index_select(1, seg_idx_h).view(1, int(pred_bch.shape[1]), horizon)
        alpha_active = any(v > 0.0 for row in rows for v in row)
    elif alpha_by_channel is not None:
        alpha_values = [float(v) for v in alpha_by_channel]
        if len(alpha_values) != int(pred_bch.shape[1]):
            raise ValueError(
                "moe.train_stat_anchor_expert.alpha_by_channel length must match the channel count "
                f"({len(alpha_values)} != {int(pred_bch.shape[1])})."
            )
        alpha: float | torch.Tensor = torch.as_tensor(
            alpha_values,
            device=pred_bch.device,
            dtype=pred_bch.dtype,
        ).view(1, -1, 1)
        alpha_active = any(v > 0.0 for v in alpha_values)
    else:
        alpha = float(cfg.get("alpha", 0.0) or 0.0)
        alpha_active = alpha > 0.0
    if not alpha_active:
        return pred_bch
    blend_target = str(cfg.get("blend_target", "prediction")).lower()
    if blend_target not in {"prediction", "base"}:
        raise ValueError("moe.train_stat_anchor_expert.blend_target must be 'prediction' or 'base'.")
    combine_mode = str(cfg.get("combine_mode", "blend")).lower()
    if combine_mode not in {"blend", "anchor_plus_prediction"}:
        raise ValueError("moe.train_stat_anchor_expert.combine_mode must be 'blend' or 'anchor_plus_prediction'.")
    table = stat_anchor_pc.detach().to(device=pred_bch.device, dtype=pred_bch.dtype)
    mode = str(cfg.get("mode", "phase_mean")).lower()
    if mode not in {"phase_mean", "phase_delta"}:
        raise ValueError("moe.train_stat_anchor_expert.mode must be 'phase_mean' or 'phase_delta'.")
    period = int(table.shape[0])
    starts = query_start_abs_b.detach().to(device=pred_bch.device, dtype=torch.long)
    steps = torch.arange(int(pred_bch.shape[-1]), device=pred_bch.device, dtype=torch.long).view(1, -1)
    if mode == "phase_delta":
        if table.ndim != 3 or int(table.shape[1]) != int(pred_bch.shape[-1]) or int(table.shape[2]) != int(pred_bch.shape[1]):
            raise ValueError("train_stat_anchor_expert phase_delta table must have shape [period, horizon, channel].")
        if x_bcl is None:
            raise ValueError("moe.train_stat_anchor_expert phase_delta requires x_bcl.")
        reference = str(cfg.get("reference", "last")).lower()
        if reference == "last":
            ref_bch = x_bcl[..., -1:].to(device=pred_bch.device, dtype=pred_bch.dtype).expand_as(pred_bch)
        elif reference == "repeat":
            pos_h = torch.arange(int(pred_bch.shape[-1]), device=pred_bch.device, dtype=torch.long) % int(x_bcl.shape[-1])
            ref_bch = x_bcl.to(device=pred_bch.device, dtype=pred_bch.dtype).index_select(-1, pos_h)
        else:
            raise ValueError("moe.train_stat_anchor_expert.reference must be 'last' or 'repeat'.")
        phases_b = (starts + int(input_len)) % period
        delta_bhc = table.index_select(0, phases_b).view(
            int(pred_bch.shape[0]),
            int(pred_bch.shape[-1]),
            int(pred_bch.shape[1]),
        )
        anchor_bch = ref_bch + delta_bhc.permute(0, 2, 1).contiguous()
    else:
        if table.ndim != 2 or int(table.shape[1]) != int(pred_bch.shape[1]):
            raise ValueError("train_stat_anchor_expert phase_mean table must have shape [period, channel].")
        phases_bh = (starts.view(-1, 1) + int(input_len) + steps) % period
        anchor_bhc = table.index_select(0, phases_bh.reshape(-1)).view(
            int(pred_bch.shape[0]),
            int(pred_bch.shape[-1]),
            int(pred_bch.shape[1]),
        )
        anchor_bch = anchor_bhc.permute(0, 2, 1).contiguous()
    if combine_mode == "anchor_plus_prediction":
        return anchor_bch + alpha * pred_bch
    if blend_target == "prediction":
        return pred_bch + alpha * (anchor_bch - pred_bch)
    return pred_bch + alpha * (anchor_bch - base_pred_bch)


def select_channel_anchor_scales(
    base_bch: torch.Tensor,
    anchor_bch: torch.Tensor,
    target_bch: torch.Tensor,
    *,
    metric: str = "mse",
    max_scale: float = 1.0,
    steps: int = 21,
    channel_chunk_size: int = 8,
    sample_chunk_size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if base_bch.shape != anchor_bch.shape or base_bch.shape != target_bch.shape:
        raise ValueError("base, anchor, and target tensors must have the same [batch, channel, horizon] shape.")
    metric = str(metric).lower()
    if metric not in {"mse", "mae"}:
        raise ValueError("train_stat_anchor_expert.scale_selection.metric must be mse or mae.")
    steps = max(2, int(steps))
    scale_grid = torch.linspace(
        0.0,
        float(max_scale),
        steps,
        device=base_bch.device,
        dtype=base_bch.dtype,
    )
    channel_count = int(base_bch.shape[1])
    chunk_size = max(1, min(channel_count, int(channel_chunk_size))) if channel_count > 0 else 1
    sample_chunk_size = max(1, min(int(base_bch.shape[0]), int(sample_chunk_size))) if int(base_bch.shape[0]) > 0 else 1
    scales_c = torch.empty(channel_count, device=base_bch.device, dtype=base_bch.dtype)
    scores_c = torch.empty_like(scales_c)
    for c0 in range(0, channel_count, chunk_size):
        c1 = min(channel_count, c0 + chunk_size)
        score_sum_sc = torch.zeros(
            int(scale_grid.numel()),
            c1 - c0,
            device=base_bch.device,
            dtype=base_bch.dtype,
        )
        total_count = 0
        for b0 in range(0, int(base_bch.shape[0]), sample_chunk_size):
            b1 = min(int(base_bch.shape[0]), b0 + sample_chunk_size)
            base_chunk = base_bch[b0:b1, c0:c1, :]
            anchor_chunk = anchor_bch[b0:b1, c0:c1, :]
            target_chunk = target_bch[b0:b1, c0:c1, :]
            cand_sbch = base_chunk.unsqueeze(0) + scale_grid.view(-1, 1, 1, 1) * (
                anchor_chunk - base_chunk
            ).unsqueeze(0)
            err_sbch = cand_sbch - target_chunk.unsqueeze(0)
            if metric == "mae":
                score_sum_sc += err_sbch.abs().sum(dim=(1, 3))
            else:
                score_sum_sc += err_sbch.pow(2).sum(dim=(1, 3))
            total_count += int(b1 - b0) * int(base_bch.shape[-1])
        score_sc = score_sum_sc / max(float(total_count), 1.0)
        best_idx_c = score_sc.argmin(dim=0)
        scales_c[c0:c1] = scale_grid.index_select(0, best_idx_c)
        scores_c[c0:c1] = score_sc.gather(0, best_idx_c.view(1, -1)).squeeze(0)
    return scales_c.detach(), scores_c.detach()


def select_channel_horizon_anchor_scales(
    base_bch: torch.Tensor,
    anchor_bch: torch.Tensor,
    target_bch: torch.Tensor,
    *,
    metric: str = "mse",
    max_scale: float = 1.0,
    steps: int = 21,
    segments: int = 4,
    channel_chunk_size: int = 8,
    sample_chunk_size: int = 256,
    scale_chunk_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if base_bch.shape != anchor_bch.shape or base_bch.shape != target_bch.shape:
        raise ValueError("base, anchor, and target tensors must have the same [batch, channel, horizon] shape.")
    metric = str(metric).lower()
    if metric not in {"mse", "mae"}:
        raise ValueError("train_stat_anchor_expert.scale_selection.metric must be mse or mae.")
    steps = max(2, int(steps))
    segments = max(1, int(segments))
    horizon = int(base_bch.shape[-1])
    scale_grid = torch.linspace(
        0.0,
        float(max_scale),
        steps,
        device=base_bch.device,
        dtype=base_bch.dtype,
    )
    scales_cs = torch.zeros(int(base_bch.shape[1]), segments, device=base_bch.device, dtype=base_bch.dtype)
    scores_cs = torch.zeros_like(scales_cs)
    channel_count = int(base_bch.shape[1])
    chunk_size = max(1, min(channel_count, int(channel_chunk_size))) if channel_count > 0 else 1
    sample_chunk = max(1, int(sample_chunk_size))
    scale_chunk = max(1, int(scale_chunk_size))
    for segment in range(segments):
        start = (segment * horizon) // segments
        end = ((segment + 1) * horizon) // segments
        if end <= start:
            end = min(horizon, start + 1)
        for c0 in range(0, channel_count, chunk_size):
            c1 = min(channel_count, c0 + chunk_size)
            width = int(c1 - c0)
            best_score_c = torch.full((width,), float("inf"), device=base_bch.device, dtype=base_bch.dtype)
            best_scale_c = torch.zeros((width,), device=base_bch.device, dtype=base_bch.dtype)
            for s0 in range(0, int(scale_grid.numel()), scale_chunk):
                s1 = min(int(scale_grid.numel()), s0 + scale_chunk)
                local_grid = scale_grid[s0:s1]
                score_sum_sc = torch.zeros((int(local_grid.numel()), width), device=base_bch.device, dtype=base_bch.dtype)
                total_count = 0
                for b0 in range(0, int(base_bch.shape[0]), sample_chunk):
                    b1 = min(int(base_bch.shape[0]), b0 + sample_chunk)
                    base_seg = base_bch[b0:b1, c0:c1, start:end]
                    anchor_seg = anchor_bch[b0:b1, c0:c1, start:end]
                    target_seg = target_bch[b0:b1, c0:c1, start:end]
                    cand_sbch = base_seg.unsqueeze(0) + local_grid.view(-1, 1, 1, 1) * (
                        anchor_seg - base_seg
                    ).unsqueeze(0)
                    err_sbch = cand_sbch - target_seg.unsqueeze(0)
                    if metric == "mae":
                        score_sum_sc += err_sbch.abs().sum(dim=(1, 3))
                    else:
                        score_sum_sc += err_sbch.pow(2).sum(dim=(1, 3))
                    total_count += int(b1 - b0) * int(end - start)
                score_sc = score_sum_sc / max(float(total_count), 1.0)
                local_best_idx_c = score_sc.argmin(dim=0)
                local_score_c = score_sc.gather(0, local_best_idx_c.view(1, -1)).squeeze(0)
                local_scale_c = local_grid.index_select(0, local_best_idx_c)
                update_c = local_score_c < best_score_c
                best_score_c = torch.where(update_c, local_score_c, best_score_c)
                best_scale_c = torch.where(update_c, local_scale_c, best_scale_c)
            scales_cs[c0:c1, segment] = best_scale_c
            scores_cs[c0:c1, segment] = best_score_c
    return scales_cs.detach(), scores_cs.detach()


@torch.no_grad()
def select_train_stat_anchor_scales_from_loader(
    *,
    model: nn.Module,
    loader: DataLoader,
    cluster_id_c: torch.Tensor,
    device: torch.device,
    history_anchor_cfg: Optional[dict],
    observed_history_tc: Optional[torch.Tensor],
    input_len: int,
    eval_start: int,
    stat_anchor_pc: torch.Tensor,
    train_stat_anchor_cfg: dict,
    metric: str,
    max_scale: float,
    steps: int,
    horizon_segments: int = 1,
    model_train_stat_adapter_pc: Optional[torch.Tensor] = None,
    model_train_stat_adapter_cfg: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    if len(loader) == 0:
        raise ValueError("train_stat_anchor_expert.scale_selection requires a non-empty validation loader.")
    model.eval()
    base_parts: List[torch.Tensor] = []
    anchor_parts: List[torch.Tensor] = []
    target_parts: List[torch.Tensor] = []
    unit_cfg = dict(train_stat_anchor_cfg)
    unit_cfg["enable"] = True
    unit_cfg["alpha"] = 1.0
    unit_cfg.pop("alpha_by_channel", None)
    unit_cfg.pop("alpha_by_channel_horizon", None)
    unit_cfg.pop("alpha_horizon_segments", None)
    combine_mode = str(unit_cfg.get("combine_mode", "blend")).lower()
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        idx = idx.to(device=device, dtype=torch.long, non_blocking=True)
        query_start_abs_b = int(eval_start) + idx
        if bool((model_train_stat_adapter_cfg or {}).get("enable", False)) and bool(
            (model_train_stat_adapter_cfg or {}).get("input_center", False)
        ):
            x_model = apply_train_stat_input_centering(
                x,
                query_start_abs_b=query_start_abs_b,
                stat_anchor_pc=model_train_stat_adapter_pc,
                cfg=model_train_stat_adapter_cfg,
            )
        else:
            x_model = apply_train_stat_input_centering(
                x,
                query_start_abs_b=query_start_abs_b,
                stat_anchor_pc=stat_anchor_pc,
                cfg=train_stat_anchor_cfg,
            )
        y_base_raw = model(x_model, cluster_id_c)
        y_base = apply_history_anchor_adapter(
            y_base_raw,
            base_pred_bch=y_base_raw,
            observed_history_tc=observed_history_tc,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len),
            cfg=history_anchor_cfg,
        )
        y_base = apply_train_stat_anchor_expert(
            y_base,
            base_pred_bch=y_base,
            x_bcl=x,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len),
            stat_anchor_pc=model_train_stat_adapter_pc,
            cfg=model_train_stat_adapter_cfg,
        )
        y_anchor = apply_train_stat_anchor_expert(
            y_base,
            base_pred_bch=y_base,
            x_bcl=x,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len),
            stat_anchor_pc=stat_anchor_pc,
            cfg=unit_cfg,
        )
        if combine_mode == "anchor_plus_prediction":
            anchor_only = y_anchor - y_base
            base_parts.append(anchor_only.detach().cpu())
            anchor_parts.append(y_anchor.detach().cpu())
        else:
            base_parts.append(y_base.detach().cpu())
            anchor_parts.append(y_anchor.detach().cpu())
        target_parts.append(y.detach().cpu())
    base_bch = torch.cat(base_parts, dim=0)
    anchor_bch = torch.cat(anchor_parts, dim=0)
    target_bch = torch.cat(target_parts, dim=0)
    if int(horizon_segments) > 1:
        scales, scores = select_channel_horizon_anchor_scales(
            base_bch,
            anchor_bch,
            target_bch,
            metric=metric,
            max_scale=float(max_scale),
            steps=int(steps),
            segments=int(horizon_segments),
        )
    else:
        scales, scores = select_channel_anchor_scales(
            base_bch,
            anchor_bch,
            target_bch,
            metric=metric,
            max_scale=float(max_scale),
            steps=int(steps),
        )
    return scales, scores, int(base_bch.shape[0])


@torch.no_grad()
def build_train_residual_anchor_table_from_loader(
    *,
    model: nn.Module,
    loader: DataLoader,
    cluster_id_c: torch.Tensor,
    device: torch.device,
    history_anchor_cfg: Optional[dict],
    observed_history_tc: Optional[torch.Tensor],
    input_len: int,
    eval_start: int,
    period: int,
    model_train_stat_adapter_pc: Optional[torch.Tensor] = None,
    model_train_stat_adapter_cfg: Optional[dict] = None,
    train_stat_anchor_pc: Optional[torch.Tensor] = None,
    train_stat_anchor_cfg: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    if len(loader) == 0:
        raise ValueError("train_residual_anchor_expert requires a non-empty train loader.")
    if int(period) <= 0:
        raise ValueError("train_residual_anchor_expert.period must be positive.")
    model.eval()
    period = int(period)
    table_sum_phc: Optional[torch.Tensor] = None
    global_sum_hc: Optional[torch.Tensor] = None
    counts_p: Optional[torch.Tensor] = None
    n_windows = 0
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        idx = idx.to(device=device, dtype=torch.long, non_blocking=True)
        query_start_abs_b = int(eval_start) + idx
        x_model = apply_train_stat_input_centering(
            x,
            query_start_abs_b=query_start_abs_b,
            stat_anchor_pc=model_train_stat_adapter_pc,
            cfg=model_train_stat_adapter_cfg,
        )
        y_base_raw = model(x_model, cluster_id_c)
        y_base = apply_history_anchor_adapter(
            y_base_raw,
            base_pred_bch=y_base_raw,
            observed_history_tc=observed_history_tc,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len),
            cfg=history_anchor_cfg,
        )
        y_base = apply_train_stat_anchor_expert(
            y_base,
            base_pred_bch=y_base,
            x_bcl=x,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len),
            stat_anchor_pc=model_train_stat_adapter_pc,
            cfg=model_train_stat_adapter_cfg,
        )
        y_base = apply_train_stat_anchor_expert(
            y_base,
            base_pred_bch=y_base,
            x_bcl=x,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len),
            stat_anchor_pc=train_stat_anchor_pc,
            cfg=train_stat_anchor_cfg,
        )
        residual_bhc = (y.detach() - y_base.detach()).permute(0, 2, 1).contiguous().cpu()
        if table_sum_phc is None:
            horizon = int(residual_bhc.shape[1])
            channel_count = int(residual_bhc.shape[2])
            table_sum_phc = torch.zeros(period, horizon, channel_count, dtype=residual_bhc.dtype)
            global_sum_hc = torch.zeros(horizon, channel_count, dtype=residual_bhc.dtype)
            counts_p = torch.zeros(period, dtype=torch.long)
        phases_b = (query_start_abs_b.detach().cpu().to(dtype=torch.long) + int(input_len)) % period
        table_sum_phc.index_add_(0, phases_b, residual_bhc)
        counts_p.index_add_(0, phases_b, torch.ones_like(phases_b, dtype=torch.long))
        global_sum_hc += residual_bhc.sum(dim=0)
        n_windows += int(residual_bhc.shape[0])
    if table_sum_phc is None or global_sum_hc is None or counts_p is None or n_windows <= 0:
        raise ValueError("train_residual_anchor_expert requires at least one train window.")
    table_phc = table_sum_phc
    global_mean_hc = global_sum_hc / float(n_windows)
    nonempty = counts_p > 0
    if bool(nonempty.any()):
        table_phc[nonempty] = table_phc[nonempty] / counts_p[nonempty].to(dtype=table_phc.dtype).view(-1, 1, 1)
    if bool((~nonempty).any()):
        table_phc[~nonempty] = global_mean_hc
    return table_phc, counts_p, int(n_windows)


@torch.no_grad()
def select_train_residual_anchor_scales_from_loader(
    *,
    model: nn.Module,
    loader: DataLoader,
    cluster_id_c: torch.Tensor,
    device: torch.device,
    history_anchor_cfg: Optional[dict],
    observed_history_tc: Optional[torch.Tensor],
    input_len: int,
    eval_start: int,
    residual_anchor_phc: torch.Tensor,
    train_residual_anchor_cfg: dict,
    metric: str,
    max_scale: float,
    steps: int,
    horizon_segments: int = 1,
    model_train_stat_adapter_pc: Optional[torch.Tensor] = None,
    model_train_stat_adapter_cfg: Optional[dict] = None,
    train_stat_anchor_pc: Optional[torch.Tensor] = None,
    train_stat_anchor_cfg: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    if len(loader) == 0:
        raise ValueError("train_residual_anchor_expert.scale_selection requires a non-empty validation loader.")
    model.eval()
    unit_cfg = dict(train_residual_anchor_cfg)
    unit_cfg["enable"] = True
    unit_cfg["alpha"] = 1.0
    unit_cfg.pop("alpha_by_channel", None)
    unit_cfg.pop("alpha_by_channel_horizon", None)
    metric = str(metric).lower()
    if metric not in {"mse", "mae"}:
        raise ValueError("train_residual_anchor_expert.scale_selection.metric must be mse or mae.")
    steps = max(2, int(steps))
    segments = max(1, int(horizon_segments))
    scale_grid: Optional[torch.Tensor] = None
    score_sum_scs: Optional[torch.Tensor] = None
    total_count_s: Optional[torch.Tensor] = None
    n_windows = 0
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        idx = idx.to(device=device, dtype=torch.long, non_blocking=True)
        query_start_abs_b = int(eval_start) + idx
        x_model = apply_train_stat_input_centering(
            x,
            query_start_abs_b=query_start_abs_b,
            stat_anchor_pc=model_train_stat_adapter_pc,
            cfg=model_train_stat_adapter_cfg,
        )
        y_base_raw = model(x_model, cluster_id_c)
        y_base = apply_history_anchor_adapter(
            y_base_raw,
            base_pred_bch=y_base_raw,
            observed_history_tc=observed_history_tc,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len),
            cfg=history_anchor_cfg,
        )
        y_base = apply_train_stat_anchor_expert(
            y_base,
            base_pred_bch=y_base,
            x_bcl=x,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len),
            stat_anchor_pc=model_train_stat_adapter_pc,
            cfg=model_train_stat_adapter_cfg,
        )
        y_base = apply_train_stat_anchor_expert(
            y_base,
            base_pred_bch=y_base,
            x_bcl=x,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len),
            stat_anchor_pc=train_stat_anchor_pc,
            cfg=train_stat_anchor_cfg,
        )
        y_anchor = apply_train_residual_anchor_expert(
            y_base,
            base_pred_bch=y_base,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len),
            residual_anchor_phc=residual_anchor_phc,
            cfg=unit_cfg,
        )
        base_cpu = y_base.detach().cpu()
        anchor_cpu = y_anchor.detach().cpu()
        target_cpu = y.detach().cpu()
        batch_size = int(base_cpu.shape[0])
        horizon = int(base_cpu.shape[-1])
        channel_count = int(base_cpu.shape[1])
        if score_sum_scs is None:
            scale_grid = torch.linspace(0.0, float(max_scale), steps, dtype=base_cpu.dtype)
            score_sum_scs = torch.zeros(steps, channel_count, segments, dtype=base_cpu.dtype)
            total_count_s = torch.zeros(segments, dtype=base_cpu.dtype)
        assert scale_grid is not None and score_sum_scs is not None and total_count_s is not None
        if int(score_sum_scs.shape[1]) != channel_count:
            raise ValueError(
                "train_residual_anchor_expert.scale_selection saw inconsistent channel counts: "
                f"{int(score_sum_scs.shape[1])} vs {channel_count}"
            )
        diff_cpu = anchor_cpu - base_cpu
        for segment in range(segments):
            start = (segment * horizon) // segments
            end = ((segment + 1) * horizon) // segments
            if end <= start:
                end = min(horizon, start + 1)
            base_seg = base_cpu[:, :, start:end]
            diff_seg = diff_cpu[:, :, start:end]
            target_seg = target_cpu[:, :, start:end]
            for s0 in range(0, steps, 32):
                s1 = min(steps, s0 + 32)
                local_grid = scale_grid[s0:s1]
                pred_sbch = base_seg.unsqueeze(0) + local_grid.view(-1, 1, 1, 1) * diff_seg.unsqueeze(0)
                err_sbch = pred_sbch - target_seg.unsqueeze(0)
                if metric == "mae":
                    score_sum_scs[s0:s1, :, segment] += err_sbch.abs().sum(dim=(1, 3))
                else:
                    score_sum_scs[s0:s1, :, segment] += err_sbch.pow(2).sum(dim=(1, 3))
            total_count_s[segment] += float(batch_size * int(end - start))
        n_windows += batch_size
    if score_sum_scs is None or total_count_s is None or scale_grid is None or n_windows <= 0:
        raise ValueError("train_residual_anchor_expert.scale_selection requires at least one validation window.")
    score_scs = score_sum_scs / total_count_s.view(1, 1, segments).clamp_min(1.0)
    best_idx_cs = score_scs.argmin(dim=0)
    scales_cs = scale_grid.index_select(0, best_idx_cs.reshape(-1)).reshape(best_idx_cs.shape)
    scores_cs = score_scs.gather(0, best_idx_cs.unsqueeze(0)).squeeze(0)
    if segments <= 1:
        return scales_cs[:, 0].detach(), scores_cs[:, 0].detach(), int(n_windows)
    return scales_cs.detach(), scores_cs.detach(), int(n_windows)


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
    mae_objective_weight=0.0,
    mae_objective_kind: str = "l1",
    mae_objective_beta: float = 1.0,
    pred_residual: Optional[ClusterwisePredResidualMoE] = None,
    pred_residual_gate: Optional[nn.Module] = None,
    pred_residual_selector: Optional[nn.Module] = None,
    pred_residual_scale_c: Optional[torch.Tensor] = None,
    residual_correction_ch: Optional[torch.Tensor] = None,
    knn_hybrid: ShapeKNNHybrid = None,
    knn_fusion_scale_ch: Optional[torch.Tensor] = None,
    knn_fusion_gate: Optional[nn.Module] = None,
    eval_start: int = 0,
    history_anchor_cfg: Optional[dict] = None,
    observed_history_tc: Optional[torch.Tensor] = None,
    input_len: int = 0,
    model_train_stat_adapter_pc: Optional[torch.Tensor] = None,
    model_train_stat_adapter_cfg: Optional[dict] = None,
    train_stat_anchor_pc: Optional[torch.Tensor] = None,
    train_residual_anchor_phc: Optional[torch.Tensor] = None,
    calendar_feature_tf: Optional[torch.Tensor] = None,
    calendar_residual_coef_cf: Optional[torch.Tensor] = None,
    diagnostic_collector: Optional[Dict[str, object]] = None,
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
    router_penalty_context_score = str(moe_cfg.get("router_penalty_context_score", "high_violation")).lower()
    moe_history_anchor_expert_cfg = moe_cfg.get("history_anchor_expert", {}) or {}
    train_stat_anchor_expert_cfg = moe_cfg.get("train_stat_anchor_expert", {}) or {}
    train_residual_anchor_expert_cfg = moe_cfg.get("train_residual_anchor_expert", {}) or {}

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

        query_start_abs_b = eval_start + idx
        x_model = apply_train_stat_input_centering(
            x,
            query_start_abs_b=query_start_abs_b,
            stat_anchor_pc=model_train_stat_adapter_pc,
            cfg=model_train_stat_adapter_cfg,
        )
        yhat_base_raw = model(x_model, cluster_id_c)
        yhat_base = apply_history_anchor_adapter(
            yhat_base_raw,
            base_pred_bch=yhat_base_raw,
            observed_history_tc=observed_history_tc,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len or x.shape[-1]),
            cfg=history_anchor_cfg,
        )
        yhat_base = apply_train_stat_anchor_expert(
            yhat_base,
            base_pred_bch=yhat_base,
            x_bcl=x,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len or x.shape[-1]),
            stat_anchor_pc=model_train_stat_adapter_pc,
            cfg=model_train_stat_adapter_cfg,
        )

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
            mask_bkp, probs_bkp, skip_bk, _ = gate(
                feat_bkf,
                straight_through=straight_through,
                penalty_context_bkp=route_pen_bkp,
                penalty_context_mode=router_mode,
                penalty_context_weight=router_penalty_context_weight,
                penalty_context_detach=router_detach_penalty_context,
                penalty_context_score=router_penalty_context_score,
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

        yhat_residual_raw = yhat_base
        residual_gate_scale = None
        if pred_residual is not None and moe_enable and P > 0:
            pred_out = pred_residual(
                x,
                yhat_base,
                cluster_id_c,
                mask_bkp,
                skip_bk=skip_bk if allow_skip else None,
            )
            yhat_residual_raw = pred_out["y_final"]
            yhat = yhat_residual_raw
            if pred_residual_selector is not None:
                pred_residual_selector.eval()
                cand_bcpH = _pred_residual_candidate_predictions(
                    yhat_base,
                    pred_out,
                    pred_residual_scale_c=pred_residual_scale_c,
                )
                if cand_bcpH is not None:
                    yhat, _ = pred_residual_selector.select_prediction(x, yhat_base, cand_bcpH)
                    yhat_residual_raw = yhat
            elif pred_residual_gate is not None:
                pred_residual_gate.eval()
                gate_feat = _knn_gate_features(x, yhat_base, yhat)
                scale = pred_residual_gate(gate_feat).to(device=yhat.device, dtype=yhat.dtype).unsqueeze(-1)
                if bool(getattr(pred_residual_gate, "apply_activation_threshold", False)):
                    threshold_raw = getattr(
                        pred_residual_gate,
                        "activation_threshold_c",
                        getattr(pred_residual_gate, "activation_threshold", 0.0),
                    )
                    threshold = torch.as_tensor(threshold_raw, device=scale.device, dtype=scale.dtype)
                    if threshold.numel() == 1:
                        threshold = threshold.reshape(1, 1, 1)
                    elif threshold.numel() == scale.shape[1]:
                        threshold = threshold.reshape(1, scale.shape[1], 1)
                    else:
                        raise ValueError(
                            f"Residual gate activation threshold must be scalar or length {scale.shape[1]}, "
                            f"got {int(threshold.numel())}."
                        )
                    if bool(getattr(pred_residual_gate, "activation_head_enable", False)):
                        active_score = pred_residual_gate.activation_prob(gate_feat).to(
                            device=scale.device,
                            dtype=scale.dtype,
                        ).unsqueeze(-1)
                        active = active_score > threshold
                    else:
                        by_abs = bool(getattr(pred_residual_gate, "activation_by_abs_scale", False))
                        active = scale.abs() > threshold if by_abs else scale > threshold
                    scale = scale * active.to(dtype=scale.dtype)
                if pred_residual_scale_c is not None:
                    channel_scale = pred_residual_scale_c.to(device=yhat.device, dtype=yhat.dtype).view(1, -1, 1)
                    scale = scale * channel_scale
                residual_gate_scale = scale
                yhat = yhat_base + scale * (yhat - yhat_base)
            elif pred_residual_scale_c is not None:
                scale = pred_residual_scale_c.to(device=yhat.device, dtype=yhat.dtype).view(1, -1, 1)
                residual_gate_scale = scale.expand(yhat.shape[0], -1, -1)
                yhat = yhat_base + scale * (yhat - yhat_base)
        else:
            yhat = yhat_base

        if moe_enable and bool(moe_history_anchor_expert_cfg.get("enable", False)):
            yhat = apply_moe_history_anchor_expert(
                yhat,
                base_pred_bch=yhat_base,
                observed_history_tc=observed_history_tc,
                query_start_abs_b=query_start_abs_b,
                input_len=int(input_len or x.shape[-1]),
                cfg=moe_history_anchor_expert_cfg,
            )
        if moe_enable and bool(train_stat_anchor_expert_cfg.get("enable", False)):
            yhat = apply_train_stat_anchor_expert(
                yhat,
                base_pred_bch=yhat_base,
                x_bcl=x,
                query_start_abs_b=query_start_abs_b,
                input_len=int(input_len or x.shape[-1]),
                stat_anchor_pc=train_stat_anchor_pc,
                cfg=train_stat_anchor_expert_cfg,
            )
        if (
            moe_enable
            and bool(train_residual_anchor_expert_cfg.get("enable", False))
            and train_residual_anchor_phc is not None
        ):
            yhat = apply_train_residual_anchor_expert(
                yhat,
                base_pred_bch=yhat_base,
                query_start_abs_b=query_start_abs_b,
                input_len=int(input_len or x.shape[-1]),
                residual_anchor_phc=train_residual_anchor_phc,
                cfg=train_residual_anchor_expert_cfg,
            )

        if knn_hybrid is not None:
            yhat_pre_knn = yhat
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
        yhat = apply_calendar_residual_correction(
            yhat,
            calendar_feature_tf=calendar_feature_tf,
            calendar_residual_coef_cf=calendar_residual_coef_cf,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len or x.shape[-1]),
        )

        if diagnostic_collector is not None:
            limit = int(diagnostic_collector.get("limit", 0))
            count = int(diagnostic_collector.get("count", 0))
            selected_idx = diagnostic_collector.get("indices")
            if selected_idx is not None:
                selected_idx = selected_idx.to(device=idx.device, dtype=torch.long)
                take_pos = torch.isin(idx, selected_idx).nonzero(as_tuple=False).view(-1)
                remaining = max(0, limit - count)
                if int(take_pos.numel()) > remaining:
                    take_pos = take_pos[:remaining]
            else:
                take = max(0, min(int(x.shape[0]), limit - count))
                take_pos = torch.arange(take, device=x.device, dtype=torch.long)
            if int(take_pos.numel()) > 0:
                parts = diagnostic_collector.setdefault("parts", {})
                parts.setdefault("idx", []).append((eval_start + idx.index_select(0, take_pos)).detach().cpu())
                parts.setdefault("x", []).append(x.index_select(0, take_pos).detach().cpu())
                parts.setdefault("y_true", []).append(y.index_select(0, take_pos).detach().cpu())
                parts.setdefault("y_base", []).append(yhat_base.index_select(0, take_pos).detach().cpu())
                parts.setdefault("y_residual_raw", []).append(yhat_residual_raw.index_select(0, take_pos).detach().cpu())
                parts.setdefault("y_final", []).append(yhat.index_select(0, take_pos).detach().cpu())
                if probs_bkp is not None:
                    parts.setdefault("gate_probs", []).append(probs_bkp.index_select(0, take_pos).detach().cpu())
                if mask_bkp is not None:
                    parts.setdefault("gate_mask", []).append(mask_bkp.index_select(0, take_pos).detach().cpu())
                if skip_bk is not None:
                    parts.setdefault("skip_prob", []).append(skip_bk.index_select(0, take_pos).detach().cpu())
                if residual_gate_scale is not None:
                    parts.setdefault("residual_gate_scale", []).append(
                        residual_gate_scale.index_select(0, take_pos).detach().cpu()
                    )
                diagnostic_collector["count"] = count + int(take_pos.numel())

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
        if _mae_objective_weight_is_nonzero(mae_objective_weight):
            mae_objective_bc = _mae_objective_bc_from_abs(
                abs_err_bch,
                kind=mae_objective_kind,
                beta=mae_objective_beta,
            )
            mae_objective_bk = scatter_mean_bc_to_bk(mae_objective_bc, cluster_id_c, K)
        else:
            mae_objective_bk = torch.zeros_like(mse_bk)
        loss_bk = (mse_weight * mse_bk) + _apply_mae_objective_weight(mae_objective_bk, mae_objective_weight) + penalty_loss_bk  # [B,K]
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
def estimate_per_channel_guarded_shrink_correction(
    model: nn.Module,
    loader: DataLoader,
    cluster_id_c: torch.Tensor,
    device: torch.device,
    grid: List[float],
    max_rel_mse_regression: float = 0.01,
    max_abs: float = 0.0,
    eval_start: int = 0,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Per-channel median-offset calibration with a per-channel MSE-regression cap.

    Builds the unit (shrink=1) median residual offset per (channel, horizon) on
    `loader`, then for each CHANNEL picks the largest shrink from `grid` whose
    val MSE regression on that channel stays within `max_rel_mse_regression`
    (relative to the uncalibrated MSE), while minimizing that channel's MAE.

    The offset only shifts predictions by a constant per (c, h), so for residual
    r = y - yhat the calibrated residual is r - s * unit[c, h]; per-channel MAE/MSE
    over a shrink grid are evaluated analytically from the collected residuals —
    no extra forward passes. Returns (corr_ch [C, H], chosen_shrink_c [C]).
    """
    if len(loader) == 0:
        return None
    model.eval()
    residuals = []
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        residuals.append((y - model(x, cluster_id_c)).detach().cpu())
    if len(residuals) == 0:
        return None
    r_nch = torch.cat(residuals, dim=0)                       # [N, C, H]
    unit_ch = r_nch.median(dim=0).values                      # [C, H]
    grid_t = torch.as_tensor(sorted(set(float(g) for g in grid)), dtype=r_nch.dtype)
    if float(grid_t.min()) > 0.0:                             # always allow s=0 (no change)
        grid_t = torch.cat([torch.zeros(1, dtype=r_nch.dtype), grid_t])
    C = r_nch.shape[1]
    # residual under shrink s: r - s*unit  -> per (s, channel) MAE/MSE over n,h
    # shapes: r_nch [N,C,H], unit_ch [C,H], grid_t [S]
    shifted = r_nch.unsqueeze(0) - grid_t.view(-1, 1, 1, 1) * unit_ch.unsqueeze(0).unsqueeze(0)  # [S,N,C,H]
    mse_sc = shifted.pow(2).mean(dim=(1, 3))                  # [S, C]
    mae_sc = shifted.abs().mean(dim=(1, 3))                   # [S, C]
    base_mse_c = mse_sc[0]                                    # s=0 row (uncalibrated)
    cap_c = (1.0 + float(max_rel_mse_regression)) * base_mse_c
    allowed = mse_sc <= cap_c.unsqueeze(0)                    # [S, C]
    big = mae_sc.max() + 1.0
    mae_masked = torch.where(allowed, mae_sc, mae_sc + big)   # forbid disallowed shrinks
    best_s_idx = mae_masked.argmin(dim=0)                     # [C]
    chosen_shrink_c = grid_t[best_s_idx]                      # [C]
    corr_ch = chosen_shrink_c.view(C, 1) * unit_ch           # [C, H]
    if max_abs and float(max_abs) > 0.0:
        corr_ch = corr_ch.clamp(min=-float(max_abs), max=float(max_abs))
    return corr_ch.detach().cpu(), chosen_shrink_c.detach().cpu()


def _calendar_features_from_datetime(
    dates: pd.Series | pd.DatetimeIndex,
    cfg: dict,
) -> Tuple[torch.Tensor, List[str]]:
    dt = pd.to_datetime(dates, errors="coerce")
    if isinstance(dt, pd.Series):
        dt = pd.DatetimeIndex(dt)
    if dt.isna().any():
        raise ValueError("calendar_residual requires parseable timestamps in data.date_col.")

    features: List[np.ndarray] = []
    names: List[str] = []
    if bool(cfg.get("include_bias", True)):
        features.append(np.ones(len(dt), dtype=np.float32))
        names.append("bias")

    harmonics = max(1, int(cfg.get("tod_harmonics", cfg.get("harmonics", 2))))
    if bool(cfg.get("time_of_day", True)):
        seconds = (
            dt.hour.to_numpy(dtype=np.float32) * 3600.0
            + dt.minute.to_numpy(dtype=np.float32) * 60.0
            + dt.second.to_numpy(dtype=np.float32)
        )
        phase = seconds / float(24 * 3600)
        for h in range(1, harmonics + 1):
            angle = (2.0 * math.pi * float(h)) * phase
            features.append(np.sin(angle).astype(np.float32))
            names.append(f"tod_sin_{h}")
            features.append(np.cos(angle).astype(np.float32))
            names.append(f"tod_cos_{h}")

    if bool(cfg.get("day_of_week", True)):
        phase = dt.dayofweek.to_numpy(dtype=np.float32) / 7.0
        angle = 2.0 * math.pi * phase
        features.append(np.sin(angle).astype(np.float32))
        names.append("dow_sin")
        features.append(np.cos(angle).astype(np.float32))
        names.append("dow_cos")

    if bool(cfg.get("month_of_year", False)):
        phase = (dt.month.to_numpy(dtype=np.float32) - 1.0) / 12.0
        angle = 2.0 * math.pi * phase
        features.append(np.sin(angle).astype(np.float32))
        names.append("month_sin")
        features.append(np.cos(angle).astype(np.float32))
        names.append("month_cos")

    if len(features) == 0:
        raise ValueError("calendar_residual must enable at least one feature.")
    return torch.from_numpy(np.stack(features, axis=1).astype(np.float32)), names


def build_calendar_feature_tensor(csv_path: str, date_col: int, max_rows: int, cfg: dict) -> Tuple[torch.Tensor, List[str]]:
    df = pd.read_csv(csv_path, usecols=[int(date_col)])
    if int(max_rows or 0) > 0:
        df = df.iloc[: int(max_rows)]
    return _calendar_features_from_datetime(df.iloc[:, 0], cfg)


def calendar_feature_batch(
    calendar_feature_tf: torch.Tensor,
    query_start_abs_b: torch.Tensor,
    input_len: int,
    pred_len: int,
) -> torch.Tensor:
    device = query_start_abs_b.device
    steps_h = torch.arange(int(pred_len), device=device, dtype=torch.long)
    idx_bh = query_start_abs_b.to(device=device, dtype=torch.long).view(-1, 1) + int(input_len) + steps_h.view(1, -1)
    if int(idx_bh.min().item()) < 0 or int(idx_bh.max().item()) >= int(calendar_feature_tf.shape[0]):
        raise ValueError("calendar_residual forecast index is outside timestamp range.")
    flat = idx_bh.reshape(-1)
    feat = calendar_feature_tf.to(device=device).index_select(0, flat)
    return feat.view(idx_bh.shape[0], idx_bh.shape[1], int(calendar_feature_tf.shape[1]))


def apply_calendar_residual_correction(
    yhat_bch: torch.Tensor,
    calendar_feature_tf: Optional[torch.Tensor],
    calendar_residual_coef_cf: Optional[torch.Tensor],
    query_start_abs_b: torch.Tensor,
    input_len: int,
) -> torch.Tensor:
    if calendar_feature_tf is None or calendar_residual_coef_cf is None:
        return yhat_bch
    feat_bhf = calendar_feature_batch(
        calendar_feature_tf=calendar_feature_tf,
        query_start_abs_b=query_start_abs_b,
        input_len=int(input_len),
        pred_len=int(yhat_bch.shape[-1]),
    ).to(device=yhat_bch.device, dtype=yhat_bch.dtype)
    coef_cf = calendar_residual_coef_cf.to(device=yhat_bch.device, dtype=yhat_bch.dtype)
    return yhat_bch + torch.einsum("bhf,cf->bch", feat_bhf, coef_cf)


def _solve_calendar_residual_coefficients(
    xtx: torch.Tensor,
    xty: Optional[torch.Tensor],
    n_rows: int,
    n_windows: int,
    cfg: dict,
) -> Tuple[Optional[torch.Tensor], Dict[str, object]]:
    if xty is None or n_rows <= 0:
        return None, {"enable": False, "reason": "no_fit_rows"}
    feat_dim = int(xtx.shape[0])
    ridge = max(0.0, float(cfg.get("ridge", 1.0e-3)))
    shrink = float(cfg.get("shrink", 1.0))
    max_abs = float(cfg.get("max_abs", 0.0))
    eye = torch.eye(feat_dim, dtype=torch.float64, device=xtx.device)
    if not bool(cfg.get("regularize_bias", False)) and feat_dim > 0:
        eye[0, 0] = 0.0
    coef_fc = torch.linalg.solve(xtx + ridge * eye, xty)
    coef_cf = coef_fc.transpose(0, 1).to(dtype=torch.float32) * float(shrink)
    if max_abs > 0.0:
        coef_cf = coef_cf.clamp(min=-max_abs, max=max_abs)
    return coef_cf.detach(), {
        "enable": True,
        "source_split": str(cfg.get("source_split", "train")),
        "fit_windows": int(n_windows),
        "fit_rows": int(n_rows),
        "feature_dim": int(feat_dim),
        "ridge": float(ridge),
        "shrink": float(shrink),
        "max_abs": float(max_abs),
        "coef_mean_abs": float(coef_cf.abs().mean().item()),
        "coef_max_abs": float(coef_cf.abs().max().item()),
    }


def _fit_calendar_residual_from_prediction_parts(
    idx_parts: List[torch.Tensor],
    y_true_parts: List[torch.Tensor],
    y_pred_parts: List[torch.Tensor],
    calendar_feature_tf: torch.Tensor,
    input_len: int,
    cfg: dict,
) -> Tuple[Optional[torch.Tensor], Dict[str, object]]:
    if len(idx_parts) == 0:
        return None, {"enable": False, "reason": "empty_prediction_parts"}
    device = calendar_feature_tf.device
    feat_dim = int(calendar_feature_tf.shape[1])
    xtx = torch.zeros((feat_dim, feat_dim), dtype=torch.float64, device=device)
    xty = None
    n_rows = 0
    n_windows = 0
    max_windows = int(cfg.get("max_windows", 0) or 0)
    for idx_part, y_true_part, y_pred_part in zip(idx_parts, y_true_parts, y_pred_parts):
        if max_windows > 0 and n_windows >= max_windows:
            break
        idx = idx_part.to(device=device, dtype=torch.long)
        y_true = y_true_part.to(device=device)
        y_pred = y_pred_part.to(device=device, dtype=y_true.dtype)
        if max_windows > 0 and n_windows + int(idx.shape[0]) > max_windows:
            keep = max(0, max_windows - n_windows)
            idx = idx[:keep]
            y_true = y_true[:keep]
            y_pred = y_pred[:keep]
        feat_bhf = calendar_feature_batch(
            calendar_feature_tf=calendar_feature_tf,
            query_start_abs_b=idx,
            input_len=int(input_len),
            pred_len=int(y_true.shape[-1]),
        ).to(device=device, dtype=torch.float64)
        feat_nf = feat_bhf.reshape(-1, feat_dim)
        residual_nc = (y_true - y_pred).permute(0, 2, 1).reshape(-1, y_true.shape[1]).to(dtype=torch.float64)
        xtx += feat_nf.transpose(0, 1).matmul(feat_nf)
        batch_xty = feat_nf.transpose(0, 1).matmul(residual_nc)
        xty = batch_xty if xty is None else xty + batch_xty
        n_rows += int(feat_nf.shape[0])
        n_windows += int(idx.shape[0])
    coef_cf, summary = _solve_calendar_residual_coefficients(
        xtx=xtx,
        xty=xty,
        n_rows=n_rows,
        n_windows=n_windows,
        cfg=cfg,
    )
    summary["fit_target"] = "prediction_parts"
    return coef_cf, summary


@torch.no_grad()
def fit_calendar_residual_correction(
    model: nn.Module,
    loader: DataLoader,
    cluster_id_c: torch.Tensor,
    device: torch.device,
    calendar_feature_tf: torch.Tensor,
    input_len: int,
    eval_start: int,
    cfg: dict,
    history_anchor_cfg: Optional[dict] = None,
    observed_history_tc: Optional[torch.Tensor] = None,
    model_train_stat_adapter_pc: Optional[torch.Tensor] = None,
    model_train_stat_adapter_cfg: Optional[dict] = None,
) -> Tuple[Optional[torch.Tensor], Dict[str, object]]:
    if len(loader) == 0:
        return None, {"enable": False, "reason": "empty_loader"}
    max_windows = int(cfg.get("max_windows", 0) or 0)

    model.eval()
    feat_dim = int(calendar_feature_tf.shape[1])
    xtx = torch.zeros((feat_dim, feat_dim), dtype=torch.float64, device=device)
    xty = None
    n_rows = 0
    n_windows = 0

    for x, y, idx in loader:
        if max_windows > 0 and n_windows >= max_windows:
            break
        if max_windows > 0 and n_windows + int(x.shape[0]) > max_windows:
            keep = max(0, max_windows - n_windows)
            x = x[:keep]
            y = y[:keep]
            idx = idx[:keep]
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        idx = idx.to(device=device, dtype=torch.long, non_blocking=True)
        query_start_abs_b = int(eval_start) + idx
        x_model = apply_train_stat_input_centering(
            x,
            query_start_abs_b=query_start_abs_b,
            stat_anchor_pc=model_train_stat_adapter_pc,
            cfg=model_train_stat_adapter_cfg,
        )
        yhat = model(x_model, cluster_id_c)
        yhat = apply_history_anchor_adapter(
            yhat,
            base_pred_bch=yhat,
            observed_history_tc=observed_history_tc,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len),
            cfg=history_anchor_cfg,
        )
        feat_bhf = calendar_feature_batch(
            calendar_feature_tf=calendar_feature_tf,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len),
            pred_len=int(y.shape[-1]),
        ).to(device=device, dtype=torch.float64)
        feat_nf = feat_bhf.reshape(-1, feat_dim)
        residual_nc = (y - yhat).permute(0, 2, 1).reshape(-1, y.shape[1]).to(dtype=torch.float64)
        xtx += feat_nf.transpose(0, 1).matmul(feat_nf)
        batch_xty = feat_nf.transpose(0, 1).matmul(residual_nc)
        xty = batch_xty if xty is None else xty + batch_xty
        n_rows += int(feat_nf.shape[0])
        n_windows += int(x.shape[0])

    coef_cf, summary = _solve_calendar_residual_coefficients(
        xtx=xtx,
        xty=xty,
        n_rows=n_rows,
        n_windows=n_windows,
        cfg=cfg,
    )
    summary["fit_target"] = "base_path"
    return coef_cf, summary


@torch.no_grad()
def fit_calendar_residual_correction_from_eval_path(
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
    calendar_feature_tf: torch.Tensor,
    input_len: int,
    cfg: dict,
    channel_count: int,
    select_ranks: Optional[List[int]] = None,
    mse_weight: float = 1.0,
    gate_entropy_weight: float = 0.0,
    gate_balance_weight: float = 0.0,
    gate_soft_weight: float = 0.0,
    gate_entropy_target_frac: float = 0.0,
    penalty_scale: Optional[torch.Tensor] = None,
    dynamic_lambda: Optional[ClusterwiseDynamicLambda] = None,
    lambda_min_kp: Optional[torch.Tensor] = None,
    mae_objective_weight=0.0,
    mae_objective_kind: str = "l1",
    mae_objective_beta: float = 1.0,
    pred_residual: Optional[ClusterwisePredResidualMoE] = None,
    pred_residual_gate: Optional[nn.Module] = None,
    pred_residual_selector: Optional[nn.Module] = None,
    pred_residual_scale_c: Optional[torch.Tensor] = None,
    residual_correction_ch: Optional[torch.Tensor] = None,
    knn_hybrid: ShapeKNNHybrid = None,
    knn_fusion_scale_ch: Optional[torch.Tensor] = None,
    knn_fusion_gate: Optional[nn.Module] = None,
    eval_start: int = 0,
    history_anchor_cfg: Optional[dict] = None,
    observed_history_tc: Optional[torch.Tensor] = None,
    model_train_stat_adapter_pc: Optional[torch.Tensor] = None,
    model_train_stat_adapter_cfg: Optional[dict] = None,
    train_stat_anchor_pc: Optional[torch.Tensor] = None,
    train_residual_anchor_phc: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], Dict[str, object]]:
    if len(loader) == 0:
        return None, {"enable": False, "reason": "empty_loader"}
    max_windows = int(cfg.get("max_windows", 0) or 0)
    try:
        loader_windows = int(len(loader.dataset))
    except Exception:
        loader_windows = int(len(loader))
    collect_limit = loader_windows if max_windows <= 0 else min(loader_windows, max_windows)
    diagnostic_collector: Dict[str, object] = {"limit": int(collect_limit), "count": 0, "parts": {}}
    eval_loop(
        model=model,
        gate=gate,
        lambda_kp=lambda_kp,
        penalty_names=penalty_names,
        penalty_fns=penalty_fns,
        loader=loader,
        cluster_id_c=cluster_id_c,
        K=K,
        moe_cfg=moe_cfg,
        device=device,
        select_ranks=select_ranks,
        collect_plot=False,
        channel_count=channel_count,
        mse_weight=mse_weight,
        gate_entropy_weight=gate_entropy_weight,
        gate_balance_weight=gate_balance_weight,
        gate_soft_weight=gate_soft_weight,
        gate_entropy_target_frac=gate_entropy_target_frac,
        penalty_scale=penalty_scale,
        dynamic_lambda=dynamic_lambda,
        lambda_min_kp=lambda_min_kp,
        mae_objective_weight=mae_objective_weight,
        mae_objective_kind=mae_objective_kind,
        mae_objective_beta=mae_objective_beta,
        pred_residual=pred_residual,
        pred_residual_gate=pred_residual_gate,
        pred_residual_selector=pred_residual_selector,
        pred_residual_scale_c=pred_residual_scale_c,
        residual_correction_ch=residual_correction_ch,
        knn_hybrid=knn_hybrid,
        knn_fusion_scale_ch=knn_fusion_scale_ch,
        knn_fusion_gate=knn_fusion_gate,
        eval_start=eval_start,
        history_anchor_cfg=history_anchor_cfg,
        observed_history_tc=observed_history_tc,
        input_len=input_len,
        model_train_stat_adapter_pc=model_train_stat_adapter_pc,
        model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
        train_stat_anchor_pc=train_stat_anchor_pc,
        train_residual_anchor_phc=train_residual_anchor_phc,
        calendar_feature_tf=None,
        calendar_residual_coef_cf=None,
        diagnostic_collector=diagnostic_collector,
    )
    parts = diagnostic_collector.get("parts", {}) or {}
    coef_cf, summary = _fit_calendar_residual_from_prediction_parts(
        idx_parts=list(parts.get("idx", [])),
        y_true_parts=list(parts.get("y_true", [])),
        y_pred_parts=list(parts.get("y_final", [])),
        calendar_feature_tf=calendar_feature_tf,
        input_len=input_len,
        cfg=cfg,
    )
    summary["fit_target"] = "final_eval_path"
    summary["fit_eval_start"] = int(eval_start)
    return coef_cf, summary


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


def _pred_residual_candidate_predictions(
    y_base_bch: torch.Tensor,
    pred_out: Dict[str, torch.Tensor],
    pred_residual_scale_c: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    residuals = pred_out.get("residuals")
    alpha_cp = pred_out.get("alpha_cp")
    intervention_bcp = pred_out.get("intervention_bcp")
    selector_bcp = pred_out.get("selector_bcp")
    if residuals is None or alpha_cp is None or intervention_bcp is None:
        return None
    if residuals.numel() == 0:
        return None
    if selector_bcp is None:
        selector_bcp = torch.ones_like(intervention_bcp)
    scale_bcp = intervention_bcp * selector_bcp * alpha_cp.unsqueeze(0)
    if pred_residual_scale_c is not None:
        channel_scale = pred_residual_scale_c.to(device=y_base_bch.device, dtype=y_base_bch.dtype).view(1, -1, 1)
        scale_bcp = scale_bcp * channel_scale
    return y_base_bch.unsqueeze(2) + scale_bcp.unsqueeze(-1) * residuals


def _candidate_selector_targets(
    *,
    base_bch: torch.Tensor,
    cand_bcpH: torch.Tensor,
    y_bch: torch.Tensor,
    min_abs_improvement: float = 0.0,
    min_rel_improvement: float = 0.0,
) -> torch.Tensor:
    base_err_bc = (base_bch - y_bch).pow(2).mean(dim=-1)
    cand_err_bcp = (cand_bcpH - y_bch.unsqueeze(2)).pow(2).mean(dim=-1)
    best_err_bc, best_p_bc = cand_err_bcp.min(dim=-1)
    gain_bc = base_err_bc - best_err_bc
    required_bc = torch.maximum(
        torch.full_like(base_err_bc, max(0.0, float(min_abs_improvement))),
        max(0.0, float(min_rel_improvement)) * base_err_bc.abs().clamp_min(1.0e-12),
    )
    return torch.where(gain_bc > required_bc, best_p_bc.to(dtype=torch.long) + 1, torch.zeros_like(best_p_bc))


def _mse_utility_gate_supervision_loss(
    *,
    probs_bkp: Optional[torch.Tensor],
    allowed_mask_kp: Optional[torch.Tensor],
    y_base_bch: torch.Tensor,
    pred_out: Optional[Dict[str, torch.Tensor]],
    y_bch: torch.Tensor,
    cluster_id_c: torch.Tensor,
    K: int,
    temperature: float = 1.0,
    min_gain: float = 0.0,
    target_power: float = 1.0,
    eps: float = 1.0e-8,
) -> Optional[torch.Tensor]:
    if probs_bkp is None or pred_out is None or probs_bkp.numel() == 0:
        return None
    cand_bcpH = _pred_residual_candidate_predictions(y_base_bch, pred_out)
    if cand_bcpH is None or cand_bcpH.numel() == 0:
        return None
    with torch.no_grad():
        base_err_bc = (y_base_bch - y_bch).pow(2).mean(dim=-1)
        cand_err_bcp = (cand_bcpH - y_bch.unsqueeze(2)).pow(2).mean(dim=-1)
        gain_bcp = base_err_bc.unsqueeze(-1) - cand_err_bcp
        utility_bcp = (gain_bcp - float(min_gain)).clamp_min(0.0)
        if allowed_mask_kp is not None and allowed_mask_kp.numel() > 0:
            allowed = allowed_mask_kp.to(device=probs_bkp.device, dtype=torch.bool)
            allowed_bkp = allowed.unsqueeze(0).expand_as(probs_bkp).to(dtype=utility_bcp.dtype)
        else:
            allowed_bkp = (probs_bkp.detach() > 0.0).to(dtype=utility_bcp.dtype)
        utility_bkp = scatter_mean_bcf_to_bkf(utility_bcp, cluster_id_c, K) * allowed_bkp
        if float(target_power) != 1.0:
            utility_bkp = utility_bkp.clamp_min(0.0).pow(float(target_power))
        valid_bk = utility_bkp.sum(dim=-1) > 0.0
        if not bool(valid_bk.any().item()):
            return None
        temp = max(float(temperature), 1.0e-6)
        target_bkp = utility_bkp.clamp_min(eps).log() / temp
        target_bkp = target_bkp.masked_fill(~valid_bk.unsqueeze(-1), 0.0)
        target_bkp = torch.softmax(target_bkp, dim=-1) * valid_bk.unsqueeze(-1).to(dtype=utility_bkp.dtype)
        target_bkp = target_bkp * allowed_bkp
        target_bkp = target_bkp / target_bkp.sum(dim=-1, keepdim=True).clamp_min(eps)
    log_probs = probs_bkp.clamp_min(eps).log()
    loss_bk = -(target_bkp * log_probs).sum(dim=-1)
    return torch.where(valid_bk, loss_bk, torch.zeros_like(loss_bk))


class PredResidualCandidateSelector(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        num_channels: int,
        num_penalties: int,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        init_skip_bias: float = 0.0,
        init_penalty_bias: float = 0.0,
        use_penalty_identity: bool = False,
    ):
        super().__init__()
        self.base_F = int(feat_dim)
        self.C = int(num_channels)
        self.P = int(num_penalties)
        self.use_penalty_identity = bool(use_penalty_identity)
        self.F = self.base_F + (self.P if self.use_penalty_identity else 0)
        hidden = int(hidden_dim)
        self.register_buffer("feature_mean", torch.zeros(1, 1, self.F), persistent=True)
        self.register_buffer("feature_std", torch.ones(1, 1, self.F), persistent=True)
        self.register_buffer("feature_standardize_enabled", torch.tensor(0.0), persistent=True)
        self.net = nn.Sequential(
            nn.Linear(self.F, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity(),
            nn.Linear(hidden, 1),
        )
        self.skip_net = nn.Sequential(
            nn.Linear(self.F, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity(),
            nn.Linear(hidden, 1),
        )
        self.skip_bias = nn.Parameter(torch.full((self.C,), float(init_skip_bias)))
        self.penalty_bias = nn.Parameter(torch.full((self.P,), float(init_penalty_bias)))
        self.penalty_channel_bias = nn.Parameter(torch.zeros(self.C, self.P))
        self.decision_margin = 0.0
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in list(self.net) + list(self.skip_net):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for seq in (self.net, self.skip_net):
            last = seq[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def set_feature_standardization(self, mean_f: torch.Tensor, std_f: torch.Tensor) -> None:
        mean = mean_f.detach().view(1, 1, self.F).to(device=self.feature_mean.device, dtype=self.feature_mean.dtype)
        std = std_f.detach().view(1, 1, self.F).to(device=self.feature_std.device, dtype=self.feature_std.dtype)
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std.clamp_min(1.0e-6))
        self.feature_standardize_enabled.fill_(1.0)

    def _standardize_feat(self, feat: torch.Tensor) -> torch.Tensor:
        if bool(self.feature_standardize_enabled.item() > 0.5):
            feat = (feat - self.feature_mean.to(device=feat.device, dtype=feat.dtype)) / self.feature_std.to(
                device=feat.device,
                dtype=feat.dtype,
            )
        return feat

    def _append_penalty_identity(
        self,
        skip_feat_bcf: torch.Tensor,
        cand_feat_bcpf: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.use_penalty_identity:
            return skip_feat_bcf, cand_feat_bcpf
        if int(skip_feat_bcf.shape[-1]) != self.base_F or int(cand_feat_bcpf.shape[-1]) != self.base_F:
            raise ValueError(
                f"Candidate selector expected base feature dim {self.base_F}, "
                f"got skip={int(skip_feat_bcf.shape[-1])}, cand={int(cand_feat_bcpf.shape[-1])}."
            )
        P = int(cand_feat_bcpf.shape[2])
        if P != self.P:
            raise ValueError(f"Candidate selector expected {self.P} penalties, got {P}.")
        skip_eye = torch.zeros(
            *skip_feat_bcf.shape[:-1],
            self.P,
            device=skip_feat_bcf.device,
            dtype=skip_feat_bcf.dtype,
        )
        eye = torch.eye(self.P, device=cand_feat_bcpf.device, dtype=cand_feat_bcpf.dtype).view(1, 1, self.P, self.P)
        eye = eye.expand(*cand_feat_bcpf.shape[:2], self.P, self.P)
        return torch.cat([skip_feat_bcf, skip_eye], dim=-1), torch.cat([cand_feat_bcpf, eye], dim=-1)

    def logits_from_features(
        self,
        skip_feat_bcf: torch.Tensor,
        cand_feat_bcpf: torch.Tensor,
        apply_decision_margin: bool = False,
    ) -> torch.Tensor:
        skip_feat_bcf, cand_feat_bcpf = self._append_penalty_identity(skip_feat_bcf, cand_feat_bcpf)
        skip_feat_bcf = self._standardize_feat(skip_feat_bcf)
        cand_feat_bcpf = self._standardize_feat(cand_feat_bcpf)
        skip_score_bc = self.skip_net(skip_feat_bcf).squeeze(-1) + self.skip_bias.view(1, -1)
        cand_score_bcp = self.net(cand_feat_bcpf).squeeze(-1)
        cand_score_bcp = cand_score_bcp + self.penalty_bias.view(1, 1, -1)
        cand_score_bcp = cand_score_bcp + self.penalty_channel_bias.view(1, self.C, self.P)
        if apply_decision_margin:
            margin = torch.as_tensor(
                float(getattr(self, "decision_margin", 0.0)),
                device=cand_score_bcp.device,
                dtype=cand_score_bcp.dtype,
            )
            cand_score_bcp = cand_score_bcp - margin
        return torch.cat([skip_score_bc.unsqueeze(-1), cand_score_bcp], dim=-1)

    def logits(self, x_bcl: torch.Tensor, base_bch: torch.Tensor, cand_bcpH: torch.Tensor) -> torch.Tensor:
        skip_feat = _knn_gate_features(x_bcl, base_bch, base_bch)
        cand_feat = torch.stack(
            [_knn_gate_features(x_bcl, base_bch, cand_bcpH[:, :, p, :]) for p in range(int(cand_bcpH.shape[2]))],
            dim=2,
        )
        return self.logits_from_features(skip_feat, cand_feat)

    def select_from_features(
        self,
        skip_feat_bcf: torch.Tensor,
        cand_feat_bcpf: torch.Tensor,
        base_bch: torch.Tensor,
        cand_bcpH: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits_bcq = self.logits_from_features(skip_feat_bcf, cand_feat_bcpf, apply_decision_margin=True)
        selected_class_bc = logits_bcq.argmax(dim=-1)
        all_pred_bcqh = torch.cat([base_bch.unsqueeze(2), cand_bcpH], dim=2)
        gather_idx = selected_class_bc.view(*selected_class_bc.shape, 1, 1).expand(-1, -1, 1, int(base_bch.shape[-1]))
        selected_bch = all_pred_bcqh.gather(2, gather_idx).squeeze(2)
        return selected_bch, selected_class_bc

    def select_prediction(
        self,
        x_bcl: torch.Tensor,
        base_bch: torch.Tensor,
        cand_bcpH: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        skip_feat = _knn_gate_features(x_bcl, base_bch, base_bch)
        cand_feat = torch.stack(
            [_knn_gate_features(x_bcl, base_bch, cand_bcpH[:, :, p, :]) for p in range(int(cand_bcpH.shape[2]))],
            dim=2,
        )
        return self.select_from_features(skip_feat, cand_feat, base_bch, cand_bcpH)


def _activation_feature_mask_for_mode(mode: str, feat_dim: int) -> torch.Tensor:
    mode_l = str(mode).lower()
    mask = torch.zeros(int(feat_dim), dtype=torch.float32)
    if mode_l in {"full", "all"}:
        mask.fill_(1.0)
    elif mode_l in {"input", "history", "history_only", "input_only"}:
        mask[: min(5, int(feat_dim))] = 1.0
    elif mode_l in {"input_base", "history_base", "input_and_base"}:
        keep = [0, 1, 2, 3, 4, 9, 11]
        for idx in keep:
            if idx < int(feat_dim):
                mask[idx] = 1.0
    elif mode_l in {"no_delta", "no_residual_delta"}:
        keep = [0, 1, 2, 3, 4, 9, 10, 11, 12]
        for idx in keep:
            if idx < int(feat_dim):
                mask[idx] = 1.0
    else:
        raise ValueError(
            "activation_feature_mode must be full, input_only, input_base, or no_delta "
            f"(got {mode!r})."
        )
    if float(mask.sum().item()) <= 0.0:
        mask.fill_(1.0)
    return mask


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
        activation_head: bool = False,
        activation_init_prob: float = 0.5,
        activation_cluster_ids: Optional[torch.Tensor] = None,
        activation_cluster_bias: bool = False,
    ):
        super().__init__()
        self.F = int(feat_dim)
        self.C = int(num_channels)
        self.max_scale = max(float(max_scale), 1.0e-6)
        self.scale_mode = str(scale_mode).lower()
        self.activation_head_enable = bool(activation_head)
        self.activation_cluster_bias_enable = bool(activation_cluster_bias)
        if self.scale_mode not in {"sigmoid", "signed_tanh"}:
            raise ValueError("Residual gate scale_mode must be 'sigmoid' or 'signed_tanh'.")
        self.register_buffer("feature_mean", torch.zeros(1, 1, self.F), persistent=True)
        self.register_buffer("feature_std", torch.ones(1, 1, self.F), persistent=True)
        self.register_buffer("feature_standardize_enabled", torch.tensor(0.0), persistent=True)
        self.register_buffer("activation_feature_mask", torch.ones(1, 1, self.F), persistent=True)
        if activation_cluster_ids is None:
            cluster_id = torch.zeros(self.C, dtype=torch.long)
        else:
            cluster_id = activation_cluster_ids.detach().cpu().to(dtype=torch.long).reshape(-1)
            if int(cluster_id.numel()) != self.C:
                raise ValueError(f"activation_cluster_ids must have {self.C} entries, got {int(cluster_id.numel())}.")
            cluster_id = cluster_id.clamp_min(0)
        self.register_buffer("activation_cluster_id_c", cluster_id, persistent=True)
        activation_num_clusters = int(cluster_id.max().item()) + 1 if self.C > 0 else 1
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
        if self.activation_head_enable:
            activation_init_prob = min(max(float(activation_init_prob), 1.0e-6), 1.0 - 1.0e-6)
            activation_init_bias = math.log(activation_init_prob / max(1.0 - activation_init_prob, 1.0e-6))
            self.activation_net = nn.Sequential(
                nn.Linear(self.F, int(hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity(),
                nn.Linear(int(hidden_dim), 1),
            )
            self.activation_channel_bias = nn.Parameter(torch.zeros(self.C))
            if self.activation_cluster_bias_enable:
                self.activation_cluster_bias = nn.Parameter(torch.zeros(activation_num_clusters))
            else:
                self.activation_cluster_bias = None
        else:
            activation_init_bias = 0.0
            self.activation_net = None
            self.activation_channel_bias = None
            self.activation_cluster_bias = None
        self.reset_parameters(init_bias)
        if self.activation_head_enable:
            self.reset_activation_parameters(activation_init_bias)

    def reset_parameters(self, init_bias: float):
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.constant_(last.bias, float(init_bias))

    def reset_activation_parameters(self, init_bias: float):
        if self.activation_net is None:
            return
        for module in self.activation_net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        last = self.activation_net[-1]
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

    def set_activation_feature_mask(self, mask_f: torch.Tensor) -> None:
        mask = mask_f.detach().view(1, 1, self.F).to(
            device=self.activation_feature_mask.device,
            dtype=self.activation_feature_mask.dtype,
        )
        self.activation_feature_mask.copy_(mask.clamp(0.0, 1.0))

    def _standardize_feat(self, feat_bcf: torch.Tensor) -> torch.Tensor:
        if bool(self.feature_standardize_enabled.item() > 0.5):
            feat_bcf = (feat_bcf - self.feature_mean.to(device=feat_bcf.device, dtype=feat_bcf.dtype)) / self.feature_std.to(
                device=feat_bcf.device,
                dtype=feat_bcf.dtype,
            )
        return feat_bcf

    def forward(self, feat_bcf: torch.Tensor) -> torch.Tensor:
        feat_bcf = self._standardize_feat(feat_bcf)
        logits_bc = self.net(feat_bcf).squeeze(-1) + self.channel_bias.view(1, -1)
        if self.scale_mode == "signed_tanh":
            return self.max_scale * torch.tanh(logits_bc)
        return self.max_scale * torch.sigmoid(logits_bc)

    def activation_prob(self, feat_bcf: torch.Tensor) -> torch.Tensor:
        if self.activation_net is None or self.activation_channel_bias is None:
            scale = self.forward(feat_bcf)
            if self.scale_mode == "signed_tanh":
                return ((scale / self.max_scale) + 1.0).mul(0.5).clamp(1.0e-6, 1.0 - 1.0e-6)
            return (scale / self.max_scale).clamp(1.0e-6, 1.0 - 1.0e-6)
        feat_bcf = self._standardize_feat(feat_bcf)
        feat_bcf = feat_bcf * self.activation_feature_mask.to(device=feat_bcf.device, dtype=feat_bcf.dtype)
        logits_bc = self.activation_net(feat_bcf).squeeze(-1) + self.activation_channel_bias.view(1, -1)
        if self.activation_cluster_bias is not None:
            cluster_bias_c = self.activation_cluster_bias.index_select(
                0,
                self.activation_cluster_id_c.to(device=self.activation_cluster_bias.device),
            )
            logits_bc = logits_bc + cluster_bias_c.to(device=logits_bc.device, dtype=logits_bc.dtype).view(1, -1)
        return torch.sigmoid(logits_bc).clamp(1.0e-6, 1.0 - 1.0e-6)


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
            loss = _loss_for(scale_b, base_b, delta_b, y_b, feat_bcf=feat_b)
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
    history_anchor_cfg: Optional[dict] = None,
    observed_history_tc: Optional[torch.Tensor] = None,
    input_len: int = 0,
    eval_start: int = 0,
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
    router_penalty_context_score = str(moe_cfg.get("router_penalty_context_score", "high_violation")).lower()

    feat_parts = []
    base_parts = []
    delta_parts = []
    y_parts = []
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        idx = idx.to(device=device, dtype=torch.long)
        y_base_raw = model(x, cluster_id_c)
        y_base = apply_history_anchor_adapter(
            y_base_raw,
            base_pred_bch=y_base_raw,
            observed_history_tc=observed_history_tc,
            query_start_abs_b=int(eval_start) + idx,
            input_len=int(input_len),
            cfg=history_anchor_cfg,
        )
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
        mask_bkp, probs_bkp, skip_bk, _ = gate(
            feat_bkf,
            straight_through=False,
            penalty_context_bkp=route_pen_bkp,
            penalty_context_mode=router_mode,
            penalty_context_weight=router_penalty_context_weight,
            penalty_context_detach=router_detach_penalty_context,
            penalty_context_score=router_penalty_context_score,
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


@torch.no_grad()
def _pred_residual_gate_activation_metrics_from_tensors(
    tensors: Optional[Dict[str, torch.Tensor]],
    residual_gate: Optional[KNNResidualGate],
    device: torch.device,
    batch_size: int,
    activation_threshold: float,
    label_min_improvement: float = 0.0,
    activation_by_abs_scale: bool = False,
    apply_activation_threshold: bool = False,
    channel_scale_c: Optional[torch.Tensor] = None,
    indices: Optional[torch.Tensor] = None,
    channel_names: Optional[List[str]] = None,
    cluster_id_c: Optional[torch.Tensor] = None,
) -> Optional[Dict[str, object]]:
    if tensors is None or residual_gate is None:
        return None
    feat = tensors["feat"]
    base = tensors["base"]
    delta = tensors["delta"]
    y = tensors["y"]
    n = int(feat.shape[0])
    if n == 0:
        return None
    if indices is None:
        indices = torch.arange(0, n, dtype=torch.long)
    else:
        indices = indices.detach().cpu().to(dtype=torch.long)
    if indices.numel() == 0:
        return None

    residual_gate.eval()
    c = int(feat.shape[1])
    threshold_t = torch.as_tensor(activation_threshold, dtype=torch.float32)
    if threshold_t.numel() == 1:
        threshold_summary = float(max(0.0, float(threshold_t.reshape(-1)[0].item())))
        threshold_bc_cpu = torch.full((1, c), threshold_summary, dtype=torch.float32)
    elif threshold_t.numel() == c:
        threshold_bc_cpu = threshold_t.reshape(1, c).clamp_min(0.0)
        threshold_summary = [float(v) for v in threshold_bc_cpu.reshape(-1).tolist()]
    else:
        raise ValueError(f"activation_threshold must be scalar or length {c}, got {int(threshold_t.numel())}")
    min_improvement = max(0.0, float(label_min_improvement))
    batch_size = max(1, int(batch_size))

    tp = fp = tn = fn = 0
    tp_c = torch.zeros(c, dtype=torch.long)
    fp_c = torch.zeros(c, dtype=torch.long)
    tn_c = torch.zeros(c, dtype=torch.long)
    fn_c = torch.zeros(c, dtype=torch.long)
    base_se = raw_se = scaled_se = 0.0
    denom = 0
    scale_sum = 0.0
    scale_abs_sum = 0.0
    scale_count = 0
    channel_scale = None
    if channel_scale_c is not None:
        channel_scale = channel_scale_c.detach().to(device=device, dtype=torch.float32).view(1, -1)

    for b0 in range(0, int(indices.numel()), batch_size):
        batch_idx = indices[b0:b0 + batch_size]
        feat_b = feat.index_select(0, batch_idx).to(device)
        base_b = base.index_select(0, batch_idx).to(device)
        delta_b = delta.index_select(0, batch_idx).to(device)
        y_b = y.index_select(0, batch_idx).to(device)

        scale_b = residual_gate(feat_b)
        if channel_scale is not None:
            scale_b = scale_b * channel_scale.to(device=scale_b.device, dtype=scale_b.dtype)
        threshold_bc = threshold_bc_cpu.to(device=scale_b.device, dtype=scale_b.dtype)
        if bool(getattr(residual_gate, "activation_head_enable", False)):
            active_score_b = residual_gate.activation_prob(feat_b).to(device=scale_b.device, dtype=scale_b.dtype)
        elif activation_by_abs_scale:
            active_score_b = scale_b.abs()
        else:
            active_score_b = scale_b

        raw_pred_b = base_b + delta_b
        base_err_bc = (base_b - y_b).pow(2).mean(dim=-1)
        raw_err_bc = (raw_pred_b - y_b).pow(2).mean(dim=-1)
        active_label = (base_err_bc - raw_err_bc) > min_improvement
        active_pred = active_score_b > threshold_bc
        scale_for_pred = scale_b * active_pred.to(dtype=scale_b.dtype) if apply_activation_threshold else scale_b
        scaled_pred_b = base_b + scale_for_pred.unsqueeze(-1) * delta_b

        tp += int((active_pred & active_label).sum().item())
        fp += int((active_pred & (~active_label)).sum().item())
        tn += int(((~active_pred) & (~active_label)).sum().item())
        fn += int(((~active_pred) & active_label).sum().item())
        tp_c += (active_pred & active_label).detach().cpu().sum(dim=0).to(dtype=torch.long)
        fp_c += (active_pred & (~active_label)).detach().cpu().sum(dim=0).to(dtype=torch.long)
        tn_c += ((~active_pred) & (~active_label)).detach().cpu().sum(dim=0).to(dtype=torch.long)
        fn_c += ((~active_pred) & active_label).detach().cpu().sum(dim=0).to(dtype=torch.long)
        base_se += float((base_b - y_b).pow(2).sum().item())
        raw_se += float((raw_pred_b - y_b).pow(2).sum().item())
        scaled_se += float((scaled_pred_b - y_b).pow(2).sum().item())
        denom += int(y_b.numel())
        scale_sum += float(scale_b.sum().item())
        scale_abs_sum += float(scale_b.abs().sum().item())
        scale_count += int(scale_b.numel())

    total = tp + fp + tn + fn
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    base_mse = base_se / max(denom, 1)
    raw_mse = raw_se / max(denom, 1)
    scaled_mse = scaled_se / max(denom, 1)
    names = list(channel_names) if channel_names is not None and len(channel_names) == c else [f"ch_{i}" for i in range(c)]
    per_channel = []
    for ci in range(c):
        c_tp = int(tp_c[ci].item())
        c_fp = int(fp_c[ci].item())
        c_tn = int(tn_c[ci].item())
        c_fn = int(fn_c[ci].item())
        c_total = c_tp + c_fp + c_tn + c_fn
        c_precision = c_tp / max(c_tp + c_fp, 1)
        c_recall = c_tp / max(c_tp + c_fn, 1)
        c_specificity = c_tn / max(c_tn + c_fp, 1)
        c_f1 = 2.0 * c_precision * c_recall / max(c_precision + c_recall, 1.0e-12)
        per_channel.append(
            {
                "channel": names[ci],
                "cluster_id": (
                    int(cluster_id_c.detach().cpu()[ci].item())
                    if cluster_id_c is not None and int(cluster_id_c.numel()) == c
                    else None
                ),
                "samples": int(c_total),
                "start_hit_rate": float(c_recall),
                "stop_hit_rate": float(c_specificity),
                "precision": float(c_precision),
                "f1": float(c_f1),
                "accuracy": float((c_tp + c_tn) / max(c_total, 1)),
                "target_positive_rate": float((c_tp + c_fn) / max(c_total, 1)),
                "pred_positive_rate": float((c_tp + c_fp) / max(c_total, 1)),
                "false_start_rate": float(c_fp / max(c_fp + c_tn, 1)),
                "miss_start_rate": float(c_fn / max(c_fn + c_tp, 1)),
                "true_positive": c_tp,
                "false_positive": c_fp,
                "true_negative": c_tn,
                "false_negative": c_fn,
            }
        )
    per_cluster = []
    if cluster_id_c is not None and int(cluster_id_c.numel()) == c:
        cid = cluster_id_c.detach().cpu().to(dtype=torch.long)
        for k in torch.unique(cid).tolist():
            mask = cid == int(k)
            k_tp = int(tp_c[mask].sum().item())
            k_fp = int(fp_c[mask].sum().item())
            k_tn = int(tn_c[mask].sum().item())
            k_fn = int(fn_c[mask].sum().item())
            k_total = k_tp + k_fp + k_tn + k_fn
            k_precision = k_tp / max(k_tp + k_fp, 1)
            k_recall = k_tp / max(k_tp + k_fn, 1)
            k_specificity = k_tn / max(k_tn + k_fp, 1)
            k_f1 = 2.0 * k_precision * k_recall / max(k_precision + k_recall, 1.0e-12)
            per_cluster.append(
                {
                    "cluster_id": int(k),
                    "samples": int(k_total),
                    "start_hit_rate": float(k_recall),
                    "stop_hit_rate": float(k_specificity),
                    "precision": float(k_precision),
                    "f1": float(k_f1),
                    "accuracy": float((k_tp + k_tn) / max(k_total, 1)),
                    "target_positive_rate": float((k_tp + k_fn) / max(k_total, 1)),
                    "pred_positive_rate": float((k_tp + k_fp) / max(k_total, 1)),
                    "false_start_rate": float(k_fp / max(k_fp + k_tn, 1)),
                    "miss_start_rate": float(k_fn / max(k_fn + k_tp, 1)),
                    "true_positive": k_tp,
                    "false_positive": k_fp,
                    "true_negative": k_tn,
                    "false_negative": k_fn,
                }
            )
    return {
        "enable": True,
        "threshold": threshold_summary,
        "label_min_improvement": float(min_improvement),
        "activation_by_abs_scale": bool(activation_by_abs_scale),
        "apply_activation_threshold": bool(apply_activation_threshold),
        "samples": int(total),
        "accuracy": float((tp + tn) / max(total, 1)),
        "hit_rate": float((tp + tn) / max(total, 1)),
        "start_hit_rate": float(recall),
        "stop_hit_rate": float(specificity),
        "false_start_rate": float(fp / max(fp + tn, 1)),
        "miss_start_rate": float(fn / max(fn + tp, 1)),
        "balanced_accuracy": float(0.5 * (recall + specificity)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
        "target_positive_rate": float((tp + fn) / max(total, 1)),
        "pred_positive_rate": float((tp + fp) / max(total, 1)),
        "mean_scale": float(scale_sum / max(scale_count, 1)),
        "mean_abs_scale": float(scale_abs_sum / max(scale_count, 1)),
        "base_mse": float(base_mse),
        "raw_residual_mse": float(raw_mse),
        "scaled_mse": float(scaled_mse),
        "raw_residual_gain_pct_vs_base": float(100.0 * (base_mse - raw_mse) / max(abs(base_mse), 1.0e-12)),
        "scaled_gain_pct_vs_base": float(100.0 * (base_mse - scaled_mse) / max(abs(base_mse), 1.0e-12)),
        "per_channel": per_channel,
        "per_cluster": per_cluster,
    }


@torch.no_grad()
def _select_pred_residual_activation_threshold(
    tensors: Optional[Dict[str, torch.Tensor]],
    residual_gate: Optional[KNNResidualGate],
    device: torch.device,
    batch_size: int,
    label_min_improvement: float,
    activation_by_abs_scale: bool,
    indices: Optional[torch.Tensor],
    max_candidates: int = 101,
    selection_metric: str = "accuracy",
    scope: str = "global",
    channel_names: Optional[List[str]] = None,
    cluster_id_c: Optional[torch.Tensor] = None,
) -> Tuple[object, Dict[str, object]]:
    if tensors is None or residual_gate is None:
        return 0.0, {"mode": "auto", "reason": "missing_tensors_or_gate"}
    feat = tensors["feat"]
    base = tensors["base"]
    delta = tensors["delta"]
    y = tensors["y"]
    n = int(feat.shape[0])
    if n == 0:
        return 0.0, {"mode": "auto", "reason": "empty_tensors"}
    if indices is None:
        indices = torch.arange(0, n, dtype=torch.long)
    else:
        indices = indices.detach().cpu().to(dtype=torch.long)
    if indices.numel() == 0:
        return 0.0, {"mode": "auto", "reason": "empty_indices"}

    residual_gate.eval()
    batch_size = max(1, int(batch_size))
    min_improvement = max(0.0, float(label_min_improvement))
    score_parts = []
    label_parts = []
    base_err_parts = []
    scaled_err_parts = []
    for b0 in range(0, int(indices.numel()), batch_size):
        batch_idx = indices[b0:b0 + batch_size]
        feat_b = feat.index_select(0, batch_idx).to(device)
        base_b = base.index_select(0, batch_idx).to(device)
        delta_b = delta.index_select(0, batch_idx).to(device)
        y_b = y.index_select(0, batch_idx).to(device)
        scale_b = residual_gate(feat_b)
        if bool(getattr(residual_gate, "activation_head_enable", False)):
            score_b = residual_gate.activation_prob(feat_b)
        else:
            score_b = scale_b.abs() if activation_by_abs_scale else scale_b
        base_err_bc = (base_b - y_b).pow(2).mean(dim=-1)
        raw_err_bc = (base_b + delta_b - y_b).pow(2).mean(dim=-1)
        scaled_err_bc = (base_b + scale_b.unsqueeze(-1) * delta_b - y_b).pow(2).mean(dim=-1)
        label_b = (base_err_bc - raw_err_bc) > min_improvement
        score_parts.append(score_b.detach().cpu())
        label_parts.append(label_b.detach().cpu())
        base_err_parts.append(base_err_bc.detach().cpu())
        scaled_err_parts.append(scaled_err_bc.detach().cpu())

    scores = torch.cat(score_parts, dim=0)
    labels = torch.cat(label_parts, dim=0)
    base_err = torch.cat(base_err_parts, dim=0)
    scaled_err = torch.cat(scaled_err_parts, dim=0)
    if scores.numel() == 0:
        return 0.0, {"mode": "auto", "reason": "empty_scores"}

    metric_mode = str(selection_metric).lower()
    if metric_mode not in {"accuracy", "balanced_accuracy", "f1", "mse"}:
        raise ValueError("activation_threshold_selection_metric must be accuracy, balanced_accuracy, f1, or mse.")

    def _best_for_vectors(
        score_v: torch.Tensor,
        label_v: torch.Tensor,
        base_err_v: torch.Tensor,
        scaled_err_v: torch.Tensor,
    ) -> Tuple[float, Dict[str, object]]:
        score_v = score_v.reshape(-1)
        label_v = label_v.reshape(-1).to(dtype=torch.bool)
        base_err_v = base_err_v.reshape(-1)
        scaled_err_v = scaled_err_v.reshape(-1)
        if score_v.numel() == 0:
            return 0.0, {"reason": "empty_scores"}
        if int(label_v.sum().item()) == 0:
            threshold = float(score_v.max().item()) + 1.0e-6
            return threshold, {"reason": "no_positive_labels", "threshold": threshold}
        uniq = torch.unique(score_v)
        if uniq.numel() > max_candidates:
            candidates_v = torch.unique(torch.quantile(score_v, torch.linspace(0.0, 1.0, steps=max_candidates)))
        else:
            candidates_v = uniq
        candidates_v = torch.unique(
            torch.cat(
                [
                    torch.tensor([0.0], dtype=score_v.dtype),
                    candidates_v,
                    torch.tensor([float(score_v.max().item()) + 1.0e-6], dtype=score_v.dtype),
                ],
                dim=0,
            )
        )
        best_v = {
            "threshold": 0.0,
            "accuracy": -1.0,
            "balanced_accuracy": -1.0,
            "f1": -1.0,
            "mse": float("inf"),
            "precision": 0.0,
            "recall": 0.0,
            "specificity": 0.0,
            "pred_positive_rate": 0.0,
            "target_positive_rate": float(label_v.float().mean().item()),
            "num_candidates": int(candidates_v.numel()),
        }
        for cand in candidates_v.tolist():
            pred = score_v > float(cand)
            tp = int((pred & label_v).sum().item())
            fp = int((pred & (~label_v)).sum().item())
            tn = int(((~pred) & (~label_v)).sum().item())
            fn = int(((~pred) & label_v).sum().item())
            total = tp + fp + tn + fn
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            specificity = tn / max(tn + fp, 1)
            balanced_accuracy = 0.5 * (recall + specificity)
            f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
            acc = (tp + tn) / max(total, 1)
            mse = float(torch.where(pred, scaled_err_v, base_err_v).mean().item())
            if metric_mode == "mse":
                better = (mse < best_v["mse"]) or (
                    abs(mse - best_v["mse"]) <= 1.0e-12 and balanced_accuracy > best_v["balanced_accuracy"]
                )
            elif metric_mode == "f1":
                better = (f1 > best_v["f1"]) or (
                    abs(f1 - best_v["f1"]) <= 1.0e-12 and balanced_accuracy > best_v["balanced_accuracy"]
                )
            elif metric_mode == "balanced_accuracy":
                better = (balanced_accuracy > best_v["balanced_accuracy"]) or (
                    abs(balanced_accuracy - best_v["balanced_accuracy"]) <= 1.0e-12 and f1 > best_v["f1"]
                )
            else:
                better = (acc > best_v["accuracy"]) or (abs(acc - best_v["accuracy"]) <= 1.0e-12 and f1 > best_v["f1"])
            if better:
                best_v.update(
                    {
                        "threshold": float(cand),
                        "accuracy": float(acc),
                        "balanced_accuracy": float(balanced_accuracy),
                        "f1": float(f1),
                        "mse": float(mse),
                        "precision": float(precision),
                        "recall": float(recall),
                        "specificity": float(specificity),
                        "pred_positive_rate": float((tp + fp) / max(total, 1)),
                    }
                )
        return float(best_v["threshold"]), best_v

    scope_mode = str(scope).lower()
    if scope_mode in {"per_channel", "channel", "channels"}:
        c = int(scores.shape[1])
        names = list(channel_names) if channel_names is not None and len(channel_names) == c else [f"ch_{i}" for i in range(c)]
        thresholds = []
        per_channel = []
        for ci in range(c):
            thr, info = _best_for_vectors(scores[:, ci], labels[:, ci], base_err[:, ci], scaled_err[:, ci])
            thresholds.append(float(thr))
            item = dict(info)
            item["channel"] = names[ci]
            item["cluster_id"] = (
                int(cluster_id_c.detach().cpu()[ci].item())
                if cluster_id_c is not None and int(cluster_id_c.numel()) == c
                else None
            )
            per_channel.append(item)
        return thresholds, {
            "mode": "auto",
            "scope": "channel",
            "selection_metric": metric_mode,
            "threshold": thresholds,
            "per_channel": per_channel,
        }
    if scope_mode in {"per_cluster", "cluster", "clusters"}:
        c = int(scores.shape[1])
        if cluster_id_c is None or int(cluster_id_c.numel()) != c:
            raise ValueError("activation_threshold_scope=cluster requires cluster_id_c with one id per channel.")
        cid = cluster_id_c.detach().cpu().to(dtype=torch.long)
        thresholds = torch.zeros(c, dtype=torch.float32)
        per_cluster = []
        for k in torch.unique(cid).tolist():
            mask = cid == int(k)
            thr, info = _best_for_vectors(scores[:, mask], labels[:, mask], base_err[:, mask], scaled_err[:, mask])
            thresholds[mask] = float(thr)
            item = dict(info)
            item["cluster_id"] = int(k)
            per_cluster.append(item)
        return [float(v) for v in thresholds.tolist()], {
            "mode": "auto",
            "scope": "cluster",
            "selection_metric": metric_mode,
            "threshold": [float(v) for v in thresholds.tolist()],
            "per_cluster": per_cluster,
        }

    thr, info = _best_for_vectors(scores, labels, base_err, scaled_err)
    info.update({"mode": "auto", "scope": "global", "selection_metric": metric_mode})
    return float(thr), info


@torch.no_grad()
def evaluate_gate_penalty_hit_metrics(
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
    label_min_improvement: float = 0.0,
    history_anchor_cfg: Optional[dict] = None,
    observed_history_tc: Optional[torch.Tensor] = None,
    input_len: int = 0,
    eval_start: int = 0,
) -> Optional[Dict[str, object]]:
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
    router_penalty_context_score = str(moe_cfg.get("router_penalty_context_score", "high_violation")).lower()

    P = len(penalty_names)
    total = positive_total = hit_total = positive_hit_total = 0
    selected_positive = 0
    base_se = oracle_se = selected_se = 0.0
    denom = 0
    oracle_count_p = torch.zeros(P, dtype=torch.long)
    selected_count_p = torch.zeros(P, dtype=torch.long)
    positive_oracle_count_p = torch.zeros(P, dtype=torch.long)
    min_improvement = max(0.0, float(label_min_improvement))

    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        idx = idx.to(device=device, dtype=torch.long)
        y_base_raw = model(x, cluster_id_c)
        y_base = apply_history_anchor_adapter(
            y_base_raw,
            base_pred_bch=y_base_raw,
            observed_history_tc=observed_history_tc,
            query_start_abs_b=int(eval_start) + idx,
            input_len=int(input_len),
            cfg=history_anchor_cfg,
        )
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
        mask_bkp, probs_bkp, skip_bk, _ = gate(
            feat_bkf,
            straight_through=False,
            penalty_context_bkp=route_pen_bkp,
            penalty_context_mode=router_mode,
            penalty_context_weight=router_penalty_context_weight,
            penalty_context_detach=router_detach_penalty_context,
            penalty_context_score=router_penalty_context_score,
        )
        rank_mask = None
        if select_ranks is not None:
            mask_bkp = _select_rank_mask(probs_bkp, select_ranks, straight_through=False)
            rank_mask = mask_bkp
        if gate_soft_weight > 0.0:
            probs_sel = probs_bkp
            if rank_mask is not None:
                probs_sel = probs_sel * rank_mask
                probs_sel = probs_sel / probs_sel.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
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
        residuals = pred_out.get("residuals")
        intervention_bcp = pred_out.get("intervention_bcp")
        selector_bcp = pred_out.get("selector_bcp")
        alpha_cp = pred_out.get("alpha_cp")
        if residuals is None or intervention_bcp is None or alpha_cp is None or residuals.numel() == 0:
            continue
        if selector_bcp is None:
            selector_bcp = torch.ones_like(intervention_bcp)

        scale_bcp = intervention_bcp * selector_bcp * alpha_cp.unsqueeze(0)
        cand_bcpH = y_base.unsqueeze(2) + scale_bcp.unsqueeze(-1) * residuals
        err_bcp = (cand_bcpH - y.unsqueeze(2)).pow(2).mean(dim=-1)
        base_err_bc = (y_base - y).pow(2).mean(dim=-1)
        oracle_err_bc, oracle_p_bc = err_bcp.min(dim=-1)
        route_bcp = mask_bkp[:, cluster_id_c, :]
        probs_bcp = probs_bkp[:, cluster_id_c, :]
        selected_score_bcp = route_bcp * probs_bcp
        selected_p_bc = selected_score_bcp.argmax(dim=-1)
        selected_err_bc = err_bcp.gather(-1, selected_p_bc.unsqueeze(-1)).squeeze(-1)

        positive_bc = (base_err_bc - oracle_err_bc) > min_improvement
        hit_bc = selected_p_bc == oracle_p_bc
        selected_positive_bc = (base_err_bc - selected_err_bc) > min_improvement

        total += int(hit_bc.numel())
        positive_total += int(positive_bc.sum().item())
        hit_total += int(hit_bc.sum().item())
        positive_hit_total += int((hit_bc & positive_bc).sum().item())
        selected_positive += int(selected_positive_bc.sum().item())
        base_se += float((y_base - y).pow(2).sum().item())
        oracle_se += float(oracle_err_bc.sum().item() * y.shape[-1])
        selected_se += float(selected_err_bc.sum().item() * y.shape[-1])
        denom += int(y.numel())
        oracle_count_p += torch.bincount(oracle_p_bc.reshape(-1).detach().cpu(), minlength=P)[:P]
        selected_count_p += torch.bincount(selected_p_bc.reshape(-1).detach().cpu(), minlength=P)[:P]
        positive_oracle_count_p += torch.bincount(
            oracle_p_bc[positive_bc].reshape(-1).detach().cpu(),
            minlength=P,
        )[:P]

    if total <= 0:
        return None
    base_mse = base_se / max(denom, 1)
    oracle_mse = oracle_se / max(denom, 1)
    selected_mse = selected_se / max(denom, 1)
    return {
        "enable": True,
        "samples": int(total),
        "label_min_improvement": float(min_improvement),
        "top1_hit_rate_all": float(hit_total / max(total, 1)),
        "top1_hit_rate_on_positive_oracle": float(positive_hit_total / max(positive_total, 1)),
        "oracle_positive_rate": float(positive_total / max(total, 1)),
        "selected_positive_rate": float(selected_positive / max(total, 1)),
        "base_mse": float(base_mse),
        "oracle_mse": float(oracle_mse),
        "selected_top1_mse": float(selected_mse),
        "oracle_gain_pct_vs_base": float(100.0 * (base_mse - oracle_mse) / max(abs(base_mse), 1.0e-12)),
        "selected_top1_gain_pct_vs_base": float(
            100.0 * (base_mse - selected_mse) / max(abs(base_mse), 1.0e-12)
        ),
        "oracle_count": {name: int(oracle_count_p[i].item()) for i, name in enumerate(penalty_names)},
        "positive_oracle_count": {
            name: int(positive_oracle_count_p[i].item()) for i, name in enumerate(penalty_names)
        },
        "selected_count": {name: int(selected_count_p[i].item()) for i, name in enumerate(penalty_names)},
    }


def _pearson_list(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt((x * x).sum()) * np.sqrt((y * y).sum()))
    if denom <= 1.0e-12:
        return None
    return float((x * y).sum() / denom)


@torch.no_grad()
def evaluate_penalty_explainability(
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
    split_name: str,
    penalty_portrait_kp: Optional[torch.Tensor] = None,
    prior_prob_kp: Optional[torch.Tensor] = None,
    allowed_mask_kp: Optional[torch.Tensor] = None,
    max_batches: int = 0,
) -> Optional[Dict[str, object]]:
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
    router_penalty_context_score = str(moe_cfg.get("router_penalty_context_score", "high_violation")).lower()
    moe_history_anchor_expert_cfg = moe_cfg.get("history_anchor_expert", {}) or {}

    P = len(penalty_names)
    cid_c = cluster_id_c.detach().to(device=device, dtype=torch.long)
    cid_cpu = cluster_id_c.detach().cpu().to(dtype=torch.long)
    cluster_channel_count = torch.bincount(cid_cpu, minlength=K).clamp_min(1).to(dtype=torch.float32)

    total_bc_k = torch.zeros(K, dtype=torch.float64)
    base_err_sum_k = torch.zeros(K, dtype=torch.float64)
    final_err_sum_k = torch.zeros(K, dtype=torch.float64)
    fusion_sum_k = torch.zeros(K, dtype=torch.float64)
    selected_count_kp = torch.zeros(K, P, dtype=torch.float64)
    oracle_count_kp = torch.zeros(K, P, dtype=torch.float64)
    positive_oracle_count_kp = torch.zeros(K, P, dtype=torch.float64)
    selected_positive_count_kp = torch.zeros(K, P, dtype=torch.float64)
    gate_prob_sum_kp = torch.zeros(K, P, dtype=torch.float64)
    route_weight_sum_kp = torch.zeros(K, P, dtype=torch.float64)
    selector_sum_kp = torch.zeros(K, P, dtype=torch.float64)
    intervention_sum_kp = torch.zeros(K, P, dtype=torch.float64)
    single_gain_sum_kp = torch.zeros(K, P, dtype=torch.float64)
    selected_gain_sum_kp = torch.zeros(K, P, dtype=torch.float64)
    selected_gain_count_kp = torch.zeros(K, P, dtype=torch.float64)

    total_decisions = 0
    total_selected = 0
    total_oracle_positive = 0
    total_selected_positive = 0
    batch_count = 0

    for x, y, _ in loader:
        batch_count += 1
        if max_batches > 0 and batch_count > int(max_batches):
            break
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
        mask_bkp, probs_bkp, skip_bk, _ = gate(
            feat_bkf,
            straight_through=False,
            penalty_context_bkp=route_pen_bkp,
            penalty_context_mode=router_mode,
            penalty_context_weight=router_penalty_context_weight,
            penalty_context_detach=router_detach_penalty_context,
            penalty_context_score=router_penalty_context_score,
        )
        rank_mask = None
        if select_ranks is not None:
            mask_bkp = _select_rank_mask(probs_bkp, select_ranks, straight_through=False)
            rank_mask = mask_bkp
        if gate_soft_weight > 0.0:
            probs_sel = probs_bkp
            if rank_mask is not None:
                probs_sel = probs_sel * rank_mask
                probs_sel = probs_sel / probs_sel.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
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
        residuals = pred_out.get("residuals")
        alpha_cp = pred_out.get("alpha_cp")
        if residuals is None or alpha_cp is None or residuals.numel() == 0:
            continue

        route_bcp = pred_out.get("route_bcp", mask_bkp[:, cid_c, :])
        effective_route_bcp = pred_out.get("effective_route_bcp", route_bcp)
        selector_bcp = pred_out.get("selector_bcp", torch.ones_like(route_bcp))
        intervention_bcp = pred_out.get("intervention_bcp", torch.ones_like(route_bcp))
        fusion_bc = pred_out.get("fusion_bc", torch.ones_like(route_bcp[..., 0]))
        y_final = pred_out["y_final"]

        alpha_bcp = alpha_cp.unsqueeze(0)
        single_scale_bcp = alpha_bcp * selector_bcp * intervention_bcp
        cand_bcpH = y_base.unsqueeze(2) + single_scale_bcp.unsqueeze(-1) * residuals
        base_err_bc = (y_base - y).pow(2).mean(dim=-1)
        final_err_bc = (y_final - y).pow(2).mean(dim=-1)
        cand_err_bcp = (cand_bcpH - y.unsqueeze(2)).pow(2).mean(dim=-1)
        gain_bcp = base_err_bc.unsqueeze(-1) - cand_err_bcp
        oracle_gain_bc, oracle_p_bc = gain_bcp.max(dim=-1)
        selected_bool_bcp = route_bcp > 0
        selected_positive_bcp = selected_bool_bcp & (gain_bcp > 0)
        positive_oracle_bc = oracle_gain_bc > 0

        total_decisions += int(base_err_bc.numel())
        total_selected += int(selected_bool_bcp.sum().item())
        total_oracle_positive += int(positive_oracle_bc.sum().item())
        total_selected_positive += int(selected_positive_bcp.sum().item())

        for k in range(K):
            ch_mask = cid_c == int(k)
            if not bool(ch_mask.any().item()):
                continue
            base_k = base_err_bc[:, ch_mask]
            final_k = final_err_bc[:, ch_mask]
            total_k = int(base_k.numel())
            total_bc_k[k] += total_k
            base_err_sum_k[k] += float(base_k.sum().item())
            final_err_sum_k[k] += float(final_k.sum().item())
            fusion_sum_k[k] += float(fusion_bc[:, ch_mask].sum().item())

            selected_count_kp[k] += selected_bool_bcp[:, ch_mask, :].sum(dim=(0, 1)).detach().cpu().to(dtype=torch.float64)
            selected_positive_count_kp[k] += selected_positive_bcp[:, ch_mask, :].sum(dim=(0, 1)).detach().cpu().to(dtype=torch.float64)
            gate_prob_sum_kp[k] += probs_bkp[:, k, :].sum(dim=0).detach().cpu().to(dtype=torch.float64) * int(ch_mask.sum().item())
            route_weight_sum_kp[k] += effective_route_bcp[:, ch_mask, :].sum(dim=(0, 1)).detach().cpu().to(dtype=torch.float64)
            selector_sum_kp[k] += selector_bcp[:, ch_mask, :].sum(dim=(0, 1)).detach().cpu().to(dtype=torch.float64)
            intervention_sum_kp[k] += intervention_bcp[:, ch_mask, :].sum(dim=(0, 1)).detach().cpu().to(dtype=torch.float64)
            single_gain_sum_kp[k] += gain_bcp[:, ch_mask, :].sum(dim=(0, 1)).detach().cpu().to(dtype=torch.float64)
            selected_gain_sum_kp[k] += (gain_bcp[:, ch_mask, :] * selected_bool_bcp[:, ch_mask, :].to(gain_bcp.dtype)).sum(dim=(0, 1)).detach().cpu().to(dtype=torch.float64)
            selected_gain_count_kp[k] += selected_bool_bcp[:, ch_mask, :].sum(dim=(0, 1)).detach().cpu().to(dtype=torch.float64)

            oracle_k = oracle_p_bc[:, ch_mask].detach().cpu().reshape(-1)
            oracle_pos_k = oracle_p_bc[:, ch_mask][positive_oracle_bc[:, ch_mask]].detach().cpu().reshape(-1)
            oracle_count_kp[k] += torch.bincount(oracle_k, minlength=P)[:P].to(dtype=torch.float64)
            if oracle_pos_k.numel() > 0:
                positive_oracle_count_kp[k] += torch.bincount(oracle_pos_k, minlength=P)[:P].to(dtype=torch.float64)

    if total_decisions <= 0:
        return None

    if penalty_portrait_kp is not None:
        portrait = penalty_portrait_kp.detach().cpu().to(dtype=torch.float64)
    else:
        portrait = torch.full((K, P), float("nan"), dtype=torch.float64)
    if prior_prob_kp is not None:
        prior = prior_prob_kp.detach().cpu().to(dtype=torch.float64)
    else:
        prior = torch.full((K, P), float("nan"), dtype=torch.float64)
    if allowed_mask_kp is not None:
        allowed = allowed_mask_kp.detach().cpu().to(dtype=torch.float64)
    else:
        allowed = torch.ones(K, P, dtype=torch.float64)

    penalty_corr = {}
    denom_kp = (total_bc_k.view(K, 1)).clamp_min(1.0)
    mean_gain_kp = single_gain_sum_kp / denom_kp
    for p, name in enumerate(penalty_names):
        xs = [float(portrait[k, p].item()) for k in range(K) if math.isfinite(float(portrait[k, p].item()))]
        ys = [float(mean_gain_kp[k, p].item()) for k in range(K) if math.isfinite(float(portrait[k, p].item()))]
        penalty_corr[name] = _pearson_list(xs, ys)

    rows = []
    per_cluster = []
    for k in range(K):
        cluster_total = float(total_bc_k[k].item())
        base_mse_k = float(base_err_sum_k[k].item() / max(cluster_total, 1.0))
        final_mse_k = float(final_err_sum_k[k].item() / max(cluster_total, 1.0))
        cluster_gain = float(100.0 * (base_mse_k - final_mse_k) / max(abs(base_mse_k), 1.0e-12))
        cluster_rows = []
        for p, name in enumerate(penalty_names):
            selected_count = float(selected_count_kp[k, p].item())
            selected_gain_count = float(selected_gain_count_kp[k, p].item())
            mean_gain = float(mean_gain_kp[k, p].item())
            selected_mean_gain = float(selected_gain_sum_kp[k, p].item() / max(selected_gain_count, 1.0))
            prior_value = float(prior[k, p].item()) if math.isfinite(float(prior[k, p].item())) else None
            portrait_value = float(portrait[k, p].item()) if math.isfinite(float(portrait[k, p].item())) else None
            allowed_value = bool(allowed[k, p].item() > 0.5)
            if allowed_value and mean_gain > 0.0:
                reason = "train_prior_allowed_and_positive_split_gain"
            elif allowed_value:
                reason = "train_prior_allowed_but_nonpositive_split_gain"
            elif mean_gain > 0.0:
                reason = "blocked_by_train_prior_but_positive_split_gain"
            else:
                reason = "low_prior_or_nonpositive_split_gain"
            row = {
                "split": split_name,
                "cluster_id": int(k),
                "cluster_channels": int(cluster_channel_count[k].item()),
                "penalty": name,
                "train_diagnostic_score": portrait_value,
                "train_prior_prob": prior_value,
                "allowed_by_train_prior": allowed_value,
                "prior_actual_gain_corr_for_penalty": penalty_corr.get(name),
                "selected_count": int(selected_count),
                "selected_rate": float(selected_count / max(cluster_total, 1.0)),
                "oracle_count": int(oracle_count_kp[k, p].item()),
                "oracle_rate": float(oracle_count_kp[k, p].item() / max(cluster_total, 1.0)),
                "positive_oracle_count": int(positive_oracle_count_kp[k, p].item()),
                "positive_oracle_rate": float(positive_oracle_count_kp[k, p].item() / max(cluster_total, 1.0)),
                "selected_positive_count": int(selected_positive_count_kp[k, p].item()),
                "selected_positive_rate": float(selected_positive_count_kp[k, p].item() / max(selected_count, 1.0)),
                "mean_gate_prob": float(gate_prob_sum_kp[k, p].item() / max(cluster_total, 1.0)),
                "mean_effective_route_weight": float(route_weight_sum_kp[k, p].item() / max(cluster_total, 1.0)),
                "mean_selector": float(selector_sum_kp[k, p].item() / max(cluster_total, 1.0)),
                "mean_intervention": float(intervention_sum_kp[k, p].item() / max(cluster_total, 1.0)),
                "mean_single_penalty_gain_mse": mean_gain,
                "selected_mean_gain_mse": selected_mean_gain,
                "cluster_base_mse": base_mse_k,
                "cluster_final_mse": final_mse_k,
                "cluster_final_gain_pct": cluster_gain,
                "mean_fusion_gate": float(fusion_sum_k[k].item() / max(cluster_total, 1.0)),
                "reason": reason,
            }
            rows.append(row)
            cluster_rows.append(row)
        cluster_rows_sorted = sorted(
            cluster_rows,
            key=lambda item: (
                bool(item["allowed_by_train_prior"]),
                float(item["mean_single_penalty_gain_mse"]),
                float(item["selected_rate"]),
            ),
            reverse=True,
        )
        per_cluster.append(
            {
                "cluster_id": int(k),
                "channels": int(cluster_channel_count[k].item()),
                "base_mse": base_mse_k,
                "final_mse": final_mse_k,
                "final_gain_pct": cluster_gain,
                "top_penalties": [
                    {
                        "penalty": item["penalty"],
                        "allowed_by_train_prior": item["allowed_by_train_prior"],
                        "train_prior_prob": item["train_prior_prob"],
                        "mean_single_penalty_gain_mse": item["mean_single_penalty_gain_mse"],
                        "selected_rate": item["selected_rate"],
                        "reason": item["reason"],
                    }
                    for item in cluster_rows_sorted[: min(3, len(cluster_rows_sorted))]
                ],
            }
        )

    base_mse = float(base_err_sum_k.sum().item() / max(total_decisions, 1))
    final_mse = float(final_err_sum_k.sum().item() / max(total_decisions, 1))
    return {
        "split": split_name,
        "samples": int(total_decisions),
        "selected_penalty_events": int(total_selected),
        "oracle_positive_events": int(total_oracle_positive),
        "selected_positive_events": int(total_selected_positive),
        "base_mse": base_mse,
        "final_mse": final_mse,
        "final_gain_pct_vs_base": float(100.0 * (base_mse - final_mse) / max(abs(base_mse), 1.0e-12)),
        "prior_actual_gain_corr": penalty_corr,
        "per_cluster": per_cluster,
        "rows": rows,
    }


def save_penalty_explainability_artifacts(
    out_dir: str,
    explainability: Dict[str, object],
) -> Dict[str, str]:
    paths = {}
    json_path = os.path.join(out_dir, "penalty_explainability.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(explainability, f, ensure_ascii=False, indent=2)
    paths["json"] = json_path
    rows = []
    for split_payload in explainability.get("splits", {}).values():
        if isinstance(split_payload, dict):
            rows.extend(split_payload.get("rows", []))
    if len(rows) > 0:
        csv_path = os.path.join(out_dir, "penalty_explainability.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
        paths["csv"] = csv_path
    return paths


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
    history_anchor_cfg: Optional[dict] = None,
    observed_history_tc: Optional[torch.Tensor] = None,
    input_len: int = 0,
    eval_start: int = 0,
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
        history_anchor_cfg=history_anchor_cfg,
        observed_history_tc=observed_history_tc,
        input_len=input_len,
        eval_start=eval_start,
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
    activation_head_enable = bool(cfg.get("activation_head_enable", False))
    activation_feature_mode = str(cfg.get("activation_feature_mode", "full"))
    activation_train_soft_gating = bool(cfg.get("activation_train_soft_gating", False))
    activation_cluster_bias_enable = bool(cfg.get("activation_cluster_bias_enable", False))

    residual_gate = KNNResidualGate(
        feat_dim=int(feat.shape[-1]),
        num_channels=c,
        hidden_dim=int(cfg.get("hidden_dim", 32)),
        dropout=float(cfg.get("dropout", 0.0)),
        max_scale=float(cfg.get("max_scale", 1.0)),
        init_scale=float(cfg.get("init_scale", 0.8)),
        scale_mode=str(cfg.get("scale_mode", "sigmoid")),
        activation_head=activation_head_enable,
        activation_init_prob=float(cfg.get("activation_init_prob", 0.5)),
        activation_cluster_ids=cluster_id_c.detach().cpu() if activation_cluster_bias_enable else None,
        activation_cluster_bias=activation_cluster_bias_enable,
    ).to(device)
    activation_feature_mask = _activation_feature_mask_for_mode(activation_feature_mode, int(feat.shape[-1]))
    residual_gate.set_activation_feature_mask(activation_feature_mask.to(device))
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
    activation_threshold_raw = cfg.get("activation_threshold", 0.1)
    activation_threshold_auto = str(activation_threshold_raw).lower() == "auto"
    activation_threshold = 0.1 if activation_threshold_auto else float(activation_threshold_raw)
    activation_label_min_improvement = float(cfg.get("activation_label_min_improvement", 0.0))
    activation_by_abs_scale = bool(
        cfg.get("activation_by_abs_scale", str(cfg.get("scale_mode", "sigmoid")).lower() == "signed_tanh")
    )
    apply_activation_threshold = bool(cfg.get("apply_activation_threshold", False))
    activation_bce_weight = float(cfg.get("activation_bce_weight", 0.0))
    activation_inactive_scale_weight = float(cfg.get("activation_inactive_scale_weight", 0.0))
    activation_pos_weight_cfg = cfg.get("activation_pos_weight", "auto")
    activation_pos_weight_scope = str(cfg.get("activation_pos_weight_scope", "global")).lower()
    if activation_pos_weight_scope not in {"global", "cluster", "channel"}:
        raise ValueError("activation_pos_weight_scope must be global, cluster, or channel.")
    activation_pos_weight_min = float(cfg.get("activation_pos_weight_min", 1.0e-6))
    activation_pos_weight_max = float(cfg.get("activation_pos_weight_max", 5.0))
    activation_rate_balance_weight = float(cfg.get("activation_rate_balance_weight", 0.0))
    activation_rate_balance_scope = str(cfg.get("activation_rate_balance_scope", "global")).lower()
    if activation_rate_balance_scope not in {"global", "cluster"}:
        raise ValueError("activation_rate_balance_scope must be global or cluster.")
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

    max_gate_scale = max(float(residual_gate.max_scale), 1.0e-6)

    def _active_labels(base_bch: torch.Tensor, delta_bch: torch.Tensor, y_bch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            base_err_bc = (base_bch - y_bch).pow(2).mean(dim=-1)
            raw_err_bc = (base_bch + delta_bch - y_bch).pow(2).mean(dim=-1)
            return (base_err_bc - raw_err_bc) > activation_label_min_improvement

    def _scale_activation_prob(scale_bc: torch.Tensor) -> torch.Tensor:
        if activation_by_abs_scale:
            prob = scale_bc.abs() / max_gate_scale
        elif str(residual_gate.scale_mode) == "signed_tanh":
            prob = (scale_bc / max_gate_scale + 1.0) * 0.5
        else:
            prob = scale_bc / max_gate_scale
        return prob.clamp(1.0e-6, 1.0 - 1.0e-6)

    activation_pos_weight = 1.0
    activation_pos_weight_bc: Optional[torch.Tensor] = None
    activation_pos_weight_summary: object = 1.0
    if activation_bce_weight > 0.0:
        if str(activation_pos_weight_cfg).lower() == "auto":
            pos_c = torch.zeros(c, dtype=torch.float64)
            neg_c = torch.zeros(c, dtype=torch.float64)
            for b0 in range(0, int(train_idx.numel()), batch_size):
                batch_idx = train_idx[b0:b0 + batch_size]
                labels_b = _active_labels(
                    base.index_select(0, batch_idx),
                    delta.index_select(0, batch_idx),
                    y.index_select(0, batch_idx),
                ).detach().cpu()
                pos_c += labels_b.sum(dim=0).to(dtype=torch.float64)
                neg_c += (~labels_b).sum(dim=0).to(dtype=torch.float64)
            if activation_pos_weight_scope == "channel":
                weight_c = (neg_c / pos_c.clamp_min(1.0)).clamp(
                    max(activation_pos_weight_min, 1.0e-6),
                    max(activation_pos_weight_max, 1.0e-6),
                )
                activation_pos_weight_bc = weight_c.to(device=device, dtype=torch.float32).view(1, -1)
                activation_pos_weight_summary = [float(v) for v in weight_c.tolist()]
                activation_pos_weight = float(weight_c.mean().item())
            elif activation_pos_weight_scope == "cluster":
                cid = cluster_id_c.detach().cpu().to(dtype=torch.long)
                weight_c = torch.ones(c, dtype=torch.float64)
                for k in torch.unique(cid).tolist():
                    mask = cid == int(k)
                    pos_k = pos_c[mask].sum()
                    neg_k = neg_c[mask].sum()
                    weight_k = (neg_k / pos_k.clamp_min(1.0)).clamp(
                        max(activation_pos_weight_min, 1.0e-6),
                        max(activation_pos_weight_max, 1.0e-6),
                    )
                    weight_c[mask] = weight_k
                activation_pos_weight_bc = weight_c.to(device=device, dtype=torch.float32).view(1, -1)
                activation_pos_weight_summary = [float(v) for v in weight_c.tolist()]
                activation_pos_weight = float(weight_c.mean().item())
            else:
                pos = float(pos_c.sum().item())
                neg = float(neg_c.sum().item())
                activation_pos_weight = min(
                    max(neg / max(pos, 1.0), max(activation_pos_weight_min, 1.0e-6)),
                    max(activation_pos_weight_max, 1.0e-6),
                )
                activation_pos_weight_summary = float(activation_pos_weight)
        else:
            activation_pos_weight = float(activation_pos_weight_cfg)
            activation_pos_weight_summary = float(activation_pos_weight)

    def _loss_for(
        scale_bc: torch.Tensor,
        base_bch: torch.Tensor,
        delta_bch: torch.Tensor,
        y_bch: torch.Tensor,
        feat_bcf: Optional[torch.Tensor] = None,
    ):
        scale_for_loss = scale_bc
        activation_prob_bc = None
        if activation_head_enable and feat_bcf is not None:
            activation_prob_bc = residual_gate.activation_prob(feat_bcf).to(dtype=scale_bc.dtype)
            if activation_train_soft_gating:
                scale_for_loss = scale_for_loss * activation_prob_bc
        pred = base_bch + scale_for_loss.unsqueeze(-1) * delta_bch
        if loss_kind == "mse":
            loss = (pred - y_bch).pow(2).mean()
        elif loss_kind == "smooth_l1":
            loss = torch.nn.functional.smooth_l1_loss(pred, y_bch, beta=beta)
        else:
            loss = (pred - y_bch).abs().mean()
        if activation_bce_weight > 0.0:
            labels_bc = _active_labels(base_bch, delta_bch, y_bch).to(dtype=scale_bc.dtype)
            prob_bc = (
                activation_prob_bc
                if activation_prob_bc is not None
                else _scale_activation_prob(scale_bc)
            )
            pos_weight = (
                activation_pos_weight_bc.to(device=scale_bc.device, dtype=scale_bc.dtype)
                if activation_pos_weight_bc is not None
                else torch.as_tensor(float(activation_pos_weight), device=scale_bc.device, dtype=scale_bc.dtype)
            )
            bce_bc = -(
                pos_weight * labels_bc * prob_bc.log()
                + (1.0 - labels_bc) * (1.0 - prob_bc).log()
            )
            loss = loss + activation_bce_weight * bce_bc.mean()
            if activation_rate_balance_weight > 0.0:
                if activation_rate_balance_scope == "cluster":
                    cid = cluster_id_c.to(device=prob_bc.device, dtype=torch.long)
                    rate_loss = torch.zeros((), device=prob_bc.device, dtype=prob_bc.dtype)
                    num_rates = 0
                    for k in torch.unique(cid).tolist():
                        mask = cid == int(k)
                        if bool(mask.any().item()):
                            prob_rate = prob_bc[:, mask].mean()
                            label_rate = labels_bc[:, mask].mean()
                            rate_loss = rate_loss + (prob_rate - label_rate).pow(2)
                            num_rates += 1
                    loss = loss + activation_rate_balance_weight * rate_loss / max(num_rates, 1)
                else:
                    loss = loss + activation_rate_balance_weight * (prob_bc.mean() - labels_bc.mean()).pow(2)
            if activation_inactive_scale_weight > 0.0:
                inactive_bc = labels_bc < 0.5
                if bool(inactive_bc.any().item()):
                    loss = loss + activation_inactive_scale_weight * prob_bc[inactive_bc].pow(2).mean()
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
                scale_for_pred_b = scale_b
                if activation_train_soft_gating and activation_head_enable:
                    scale_for_pred_b = scale_for_pred_b * residual_gate.activation_prob(feat_b).to(dtype=scale_b.dtype)
                pred_b = base_b + scale_for_pred_b.unsqueeze(-1) * delta_b
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
            loss = _loss_for(scale_b, base_b, delta_b, y_b, feat_bcf=feat_b)
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
    activation_threshold_selection = {"mode": "fixed", "threshold": float(activation_threshold)}
    if activation_threshold_auto:
        activation_threshold, activation_threshold_selection = _select_pred_residual_activation_threshold(
            tensors=tensors,
            residual_gate=residual_gate,
            device=device,
            batch_size=batch_size,
            label_min_improvement=activation_label_min_improvement,
            activation_by_abs_scale=activation_by_abs_scale,
            indices=hold_idx,
            max_candidates=int(cfg.get("activation_threshold_candidates", 101)),
            selection_metric=str(cfg.get("activation_threshold_selection_metric", "accuracy")),
            scope=str(cfg.get("activation_threshold_scope", "global")),
            channel_names=channel_names,
            cluster_id_c=cluster_id_c,
        )
    activation_threshold_tensor = torch.as_tensor(activation_threshold, dtype=torch.float32)
    if activation_threshold_tensor.numel() == 1:
        residual_gate.activation_threshold = float(activation_threshold_tensor.reshape(-1)[0].item())
    elif activation_threshold_tensor.numel() == c:
        residual_gate.activation_threshold = [float(v) for v in activation_threshold_tensor.reshape(-1).tolist()]
        residual_gate.activation_threshold_c = activation_threshold_tensor.reshape(-1).detach().cpu()
    else:
        raise ValueError(f"activation_threshold must be scalar or length {c}, got {int(activation_threshold_tensor.numel())}")
    residual_gate.apply_activation_threshold = bool(apply_activation_threshold)
    residual_gate.activation_by_abs_scale = bool(activation_by_abs_scale)
    train_activation = _pred_residual_gate_activation_metrics_from_tensors(
        tensors=tensors,
        residual_gate=residual_gate,
        device=device,
        batch_size=batch_size,
        activation_threshold=activation_threshold,
        label_min_improvement=activation_label_min_improvement,
        activation_by_abs_scale=activation_by_abs_scale,
        apply_activation_threshold=apply_activation_threshold,
        indices=train_idx,
        channel_names=channel_names,
        cluster_id_c=cluster_id_c,
    )
    hold_activation = _pred_residual_gate_activation_metrics_from_tensors(
        tensors=tensors,
        residual_gate=residual_gate,
        device=device,
        batch_size=batch_size,
        activation_threshold=activation_threshold,
        label_min_improvement=activation_label_min_improvement,
        activation_by_abs_scale=activation_by_abs_scale,
        apply_activation_threshold=apply_activation_threshold,
        indices=hold_idx,
        channel_names=channel_names,
        cluster_id_c=cluster_id_c,
    )
    summary = {
        "enable": True,
        "loss": loss_kind,
        "selection_metric": selection_metric,
        "scale_mode": str(residual_gate.scale_mode),
        "activation_threshold": (
            [float(v) for v in torch.as_tensor(activation_threshold).reshape(-1).tolist()]
            if torch.as_tensor(activation_threshold).numel() > 1
            else float(torch.as_tensor(activation_threshold).reshape(-1)[0].item())
        ),
        "activation_threshold_selection": activation_threshold_selection,
        "activation_label_min_improvement": float(activation_label_min_improvement),
        "activation_by_abs_scale": bool(activation_by_abs_scale),
        "apply_activation_threshold": bool(apply_activation_threshold),
        "activation_head_enable": bool(activation_head_enable),
        "activation_feature_mode": str(activation_feature_mode),
        "activation_feature_mask": [float(v) for v in activation_feature_mask.reshape(-1).tolist()],
        "activation_train_soft_gating": bool(activation_train_soft_gating),
        "activation_cluster_bias_enable": bool(activation_cluster_bias_enable),
        "activation_bce_weight": float(activation_bce_weight),
        "activation_inactive_scale_weight": float(activation_inactive_scale_weight),
        "activation_pos_weight": activation_pos_weight_summary,
        "activation_pos_weight_scope": str(activation_pos_weight_scope),
        "activation_pos_weight_min": float(activation_pos_weight_min),
        "activation_pos_weight_max": float(activation_pos_weight_max),
        "activation_rate_balance_weight": float(activation_rate_balance_weight),
        "activation_rate_balance_scope": str(activation_rate_balance_scope),
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
        "train_activation": train_activation,
        "holdout_activation": hold_activation,
    }
    residual_gate.eval()
    return residual_gate, summary


@torch.no_grad()
def _collect_pred_residual_selector_tensors(
    model: nn.Module,
    pred_residual: Optional[ClusterwisePredResidualMoE],
    loader: DataLoader,
    cluster_id_c: torch.Tensor,
    K: int,
    moe_cfg: dict,
    device: torch.device,
    penalty_count: int,
    pred_residual_scale_c: Optional[torch.Tensor] = None,
    history_anchor_cfg: Optional[dict] = None,
    observed_history_tc: Optional[torch.Tensor] = None,
    input_len: int = 0,
    eval_start: int = 0,
) -> Optional[Dict[str, torch.Tensor]]:
    if len(loader) == 0 or pred_residual is None or int(penalty_count) <= 0:
        return None
    if not bool(moe_cfg.get("enable", True)):
        return None

    model.eval()
    pred_residual.eval()
    skip_feat_parts = []
    cand_feat_parts = []
    base_parts = []
    cand_parts = []
    y_parts = []
    P = int(penalty_count)
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        idx = idx.to(device=device, dtype=torch.long)
        y_base_raw = model(x, cluster_id_c)
        y_base = apply_history_anchor_adapter(
            y_base_raw,
            base_pred_bch=y_base_raw,
            observed_history_tc=observed_history_tc,
            query_start_abs_b=int(eval_start) + idx,
            input_len=int(input_len),
            cfg=history_anchor_cfg,
        )
        mask_all_bkp = torch.ones(x.shape[0], int(K), P, device=device, dtype=y_base.dtype)
        pred_out = pred_residual(
            x,
            y_base,
            cluster_id_c,
            mask_all_bkp,
            skip_bk=None,
        )
        cand_bcpH = _pred_residual_candidate_predictions(
            y_base,
            pred_out,
            pred_residual_scale_c=pred_residual_scale_c,
        )
        if cand_bcpH is None:
            continue
        skip_feat = _knn_gate_features(x, y_base, y_base)
        cand_feat = torch.stack(
            [_knn_gate_features(x, y_base, cand_bcpH[:, :, p, :]) for p in range(P)],
            dim=2,
        )
        skip_feat_parts.append(skip_feat.detach().cpu())
        cand_feat_parts.append(cand_feat.detach().cpu())
        base_parts.append(y_base.detach().cpu())
        cand_parts.append(cand_bcpH.detach().cpu())
        y_parts.append(y.detach().cpu())

    if len(skip_feat_parts) == 0:
        return None
    return {
        "skip_feat": torch.cat(skip_feat_parts, dim=0),
        "cand_feat": torch.cat(cand_feat_parts, dim=0),
        "base": torch.cat(base_parts, dim=0),
        "cand": torch.cat(cand_parts, dim=0),
        "y": torch.cat(y_parts, dim=0),
    }


@torch.no_grad()
def _pred_residual_selector_metrics_from_tensors(
    tensors: Optional[Dict[str, torch.Tensor]],
    selector: Optional[PredResidualCandidateSelector],
    device: torch.device,
    batch_size: int,
    min_abs_improvement: float,
    min_rel_improvement: float,
    indices: Optional[torch.Tensor] = None,
    penalty_names: Optional[List[str]] = None,
) -> Optional[Dict[str, object]]:
    if tensors is None or selector is None:
        return None
    skip_feat = tensors["skip_feat"]
    cand_feat = tensors["cand_feat"]
    base = tensors["base"]
    cand = tensors["cand"]
    y = tensors["y"]
    n = int(base.shape[0])
    if n == 0:
        return None
    if indices is None:
        indices = torch.arange(0, n, dtype=torch.long)
    else:
        indices = indices.detach().cpu().to(dtype=torch.long)
    if int(indices.numel()) == 0:
        return None

    selector.eval()
    P = int(cand.shape[2])
    q = P + 1
    batch_size = max(1, int(batch_size))
    base_se = selected_se = target_se = oracle_se = 0.0
    denom = 0
    correct = 0
    total_bc = 0
    selected_count_q = torch.zeros(q, dtype=torch.long)
    target_count_q = torch.zeros(q, dtype=torch.long)
    oracle_count_q = torch.zeros(q, dtype=torch.long)
    for b0 in range(0, int(indices.numel()), batch_size):
        batch_idx = indices[b0:b0 + batch_size]
        skip_feat_b = skip_feat.index_select(0, batch_idx).to(device)
        cand_feat_b = cand_feat.index_select(0, batch_idx).to(device)
        base_b = base.index_select(0, batch_idx).to(device)
        cand_b = cand.index_select(0, batch_idx).to(device)
        y_b = y.index_select(0, batch_idx).to(device)
        selected_b, selected_class_b = selector.select_from_features(skip_feat_b, cand_feat_b, base_b, cand_b)
        target_b = _candidate_selector_targets(
            base_bch=base_b,
            cand_bcpH=cand_b,
            y_bch=y_b,
            min_abs_improvement=min_abs_improvement,
            min_rel_improvement=min_rel_improvement,
        )
        all_pred_bq = torch.cat([base_b.unsqueeze(2), cand_b], dim=2)
        target_pred_b = all_pred_bq.gather(
            2,
            target_b.view(*target_b.shape, 1, 1).expand(-1, -1, 1, int(base_b.shape[-1])),
        ).squeeze(2)
        base_err_bc = (base_b - y_b).pow(2).mean(dim=-1)
        cand_err_bcp = (cand_b - y_b.unsqueeze(2)).pow(2).mean(dim=-1)
        oracle_err_bc, oracle_p_bc = cand_err_bcp.min(dim=-1)
        selected_se += float((selected_b - y_b).pow(2).sum().item())
        target_se += float((target_pred_b - y_b).pow(2).sum().item())
        oracle_se += float(oracle_err_bc.sum().item() * y_b.shape[-1])
        base_se += float((base_b - y_b).pow(2).sum().item())
        denom += int(y_b.numel())
        total_bc += int(target_b.numel())
        correct += int((selected_class_b == target_b).sum().item())
        selected_count_q += torch.bincount(selected_class_b.detach().cpu().reshape(-1), minlength=q)[:q]
        target_count_q += torch.bincount(target_b.detach().cpu().reshape(-1), minlength=q)[:q]
        oracle_count_q += torch.bincount((oracle_p_bc.detach().cpu().reshape(-1) + 1), minlength=q)[:q]

    if denom <= 0:
        return None
    base_mse = base_se / max(denom, 1)
    selected_mse = selected_se / max(denom, 1)
    target_mse = target_se / max(denom, 1)
    oracle_mse = oracle_se / max(denom, 1)
    names = ["skip"] + [str(name) for name in (penalty_names or [f"p{p}" for p in range(P)])]
    return {
        "samples": int(total_bc),
        "accuracy": float(correct / max(total_bc, 1)),
        "base_mse": float(base_mse),
        "selected_mse": float(selected_mse),
        "target_mse": float(target_mse),
        "oracle_mse": float(oracle_mse),
        "selected_gain_pct_vs_base": float(100.0 * (base_mse - selected_mse) / max(abs(base_mse), 1.0e-12)),
        "target_gain_pct_vs_base": float(100.0 * (base_mse - target_mse) / max(abs(base_mse), 1.0e-12)),
        "oracle_gain_pct_vs_base": float(100.0 * (base_mse - oracle_mse) / max(abs(base_mse), 1.0e-12)),
        "selected_class_rate": {
            names[i]: float(selected_count_q[i].item() / max(total_bc, 1)) for i in range(q)
        },
        "target_class_rate": {
            names[i]: float(target_count_q[i].item() / max(total_bc, 1)) for i in range(q)
        },
        "oracle_class_rate": {
            names[i]: float(oracle_count_q[i].item() / max(total_bc, 1)) for i in range(q)
        },
    }


def train_pred_residual_candidate_selector(
    model: nn.Module,
    pred_residual: Optional[ClusterwisePredResidualMoE],
    loader: DataLoader,
    cluster_id_c: torch.Tensor,
    K: int,
    moe_cfg: dict,
    device: torch.device,
    penalty_names: List[str],
    channel_names: List[str],
    cfg: Dict[str, object],
    pred_residual_scale_c: Optional[torch.Tensor] = None,
    history_anchor_cfg: Optional[dict] = None,
    observed_history_tc: Optional[torch.Tensor] = None,
    input_len: int = 0,
    eval_start: int = 0,
) -> Tuple[Optional[PredResidualCandidateSelector], Dict[str, object]]:
    tensors = _collect_pred_residual_selector_tensors(
        model=model,
        pred_residual=pred_residual,
        loader=loader,
        cluster_id_c=cluster_id_c,
        K=K,
        moe_cfg=moe_cfg,
        device=device,
        penalty_count=len(penalty_names),
        pred_residual_scale_c=pred_residual_scale_c,
        history_anchor_cfg=history_anchor_cfg,
        observed_history_tc=observed_history_tc,
        input_len=input_len,
        eval_start=eval_start,
    )
    if tensors is None:
        return None, {"enable": False, "reason": "empty_loader_or_residual_disabled"}

    skip_feat = tensors["skip_feat"]
    cand_feat = tensors["cand_feat"]
    base = tensors["base"]
    cand = tensors["cand"]
    y = tensors["y"]
    n = int(base.shape[0])
    c = int(base.shape[1])
    p = int(cand.shape[2])
    train_fraction = float(cfg.get("train_fraction", 0.75))
    split = int(max(1, min(n - 1, round(n * train_fraction)))) if n > 1 else n
    train_idx = torch.arange(0, split, dtype=torch.long)
    hold_idx = torch.arange(split, n, dtype=torch.long)
    if hold_idx.numel() == 0:
        hold_idx = train_idx

    selector = PredResidualCandidateSelector(
        feat_dim=int(skip_feat.shape[-1]),
        num_channels=c,
        num_penalties=p,
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        dropout=float(cfg.get("dropout", 0.0)),
        init_skip_bias=float(cfg.get("init_skip_bias", 0.0)),
        init_penalty_bias=float(cfg.get("init_penalty_bias", 0.0)),
        use_penalty_identity=bool(cfg.get("use_penalty_identity", False)),
    ).to(device)
    standardize_features = bool(cfg.get("standardize_features", True))
    feature_std_summary: Dict[str, object] = {
        "standardize_features": bool(standardize_features),
        "fit_windows": int(train_idx.numel()) if standardize_features else 0,
        "min_std": None,
        "max_std": None,
    }
    if standardize_features:
        skip_train_raw = skip_feat.index_select(0, train_idx)
        cand_train_raw = cand_feat.index_select(0, train_idx)
        skip_train_aug, cand_train_aug = selector._append_penalty_identity(skip_train_raw, cand_train_raw)
        skip_train = skip_train_aug.reshape(-1, int(skip_train_aug.shape[-1]))
        cand_train = cand_train_aug.reshape(-1, int(cand_train_aug.shape[-1]))
        feat_train = torch.cat([skip_train, cand_train], dim=0)
        feat_mean = feat_train.mean(dim=0)
        feat_std = feat_train.std(dim=0, unbiased=False).clamp_min(1.0e-6)
        selector.set_feature_standardization(feat_mean.to(device), feat_std.to(device))
        feature_std_summary.update({"min_std": float(feat_std.min().item()), "max_std": float(feat_std.max().item())})

    min_abs_improvement = float(cfg.get("label_min_abs_improvement", cfg.get("min_abs_improvement", 0.0)))
    min_rel_improvement = float(cfg.get("label_min_rel_improvement", cfg.get("min_rel_improvement", 0.0)))
    lr = float(cfg.get("lr", 1.0e-3))
    weight_decay = float(cfg.get("weight_decay", 1.0e-4))
    batch_size = max(1, int(cfg.get("batch_size", 256)))
    epochs = max(1, int(cfg.get("epochs", 40)))
    grad_clip = float(cfg.get("grad_clip", 1.0))
    positive_sample_weight = float(cfg.get("positive_sample_weight", 1.0))
    negative_sample_weight = float(cfg.get("negative_sample_weight", 1.0))
    class_weight_cfg = cfg.get("class_weight", "auto")
    class_weight = None
    class_weight_summary: object = None
    train_target = _candidate_selector_targets(
        base_bch=base.index_select(0, train_idx),
        cand_bcpH=cand.index_select(0, train_idx),
        y_bch=y.index_select(0, train_idx),
        min_abs_improvement=min_abs_improvement,
        min_rel_improvement=min_rel_improvement,
    ).reshape(-1)
    if isinstance(class_weight_cfg, str) and class_weight_cfg.lower() == "auto":
        counts = torch.bincount(train_target, minlength=p + 1).to(dtype=torch.float32)
        total = counts.sum().clamp_min(1.0)
        weight = total / (float(p + 1) * counts.clamp_min(1.0))
        weight = weight.clamp(
            min=float(cfg.get("class_weight_min", 0.25)),
            max=float(cfg.get("class_weight_max", 8.0)),
        )
        class_weight = weight.to(device)
        class_weight_summary = [float(v) for v in weight.tolist()]
    elif isinstance(class_weight_cfg, (list, tuple)):
        weight = torch.as_tensor(class_weight_cfg, dtype=torch.float32)
        if int(weight.numel()) != p + 1:
            raise ValueError(f"candidate_selector.class_weight must have {p + 1} entries.")
        class_weight = weight.to(device)
        class_weight_summary = [float(v) for v in weight.tolist()]
    elif isinstance(class_weight_cfg, str) and class_weight_cfg.lower() in {"none", "false", "off", "0"}:
        class_weight = None
        class_weight_summary = None
    elif bool(class_weight_cfg):
        scalar = float(class_weight_cfg)
        class_weight = torch.ones(p + 1, dtype=torch.float32, device=device)
        class_weight[1:] = scalar
        class_weight_summary = [float(v) for v in class_weight.detach().cpu().tolist()]

    opt = torch.optim.AdamW(selector.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_hold = float("inf")
    best_epoch = 0

    def _loss_for(batch_idx: torch.Tensor) -> torch.Tensor:
        skip_feat_b = skip_feat.index_select(0, batch_idx).to(device)
        cand_feat_b = cand_feat.index_select(0, batch_idx).to(device)
        base_b = base.index_select(0, batch_idx).to(device)
        cand_b = cand.index_select(0, batch_idx).to(device)
        y_b = y.index_select(0, batch_idx).to(device)
        target_b = _candidate_selector_targets(
            base_bch=base_b,
            cand_bcpH=cand_b,
            y_bch=y_b,
            min_abs_improvement=min_abs_improvement,
            min_rel_improvement=min_rel_improvement,
        )
        logits_bcq = selector.logits_from_features(skip_feat_b, cand_feat_b)
        loss_bc = torch.nn.functional.cross_entropy(
            logits_bcq.reshape(-1, p + 1),
            target_b.reshape(-1),
            weight=class_weight,
            reduction="none",
        ).view_as(target_b).to(dtype=logits_bcq.dtype)
        if positive_sample_weight != 1.0 or negative_sample_weight != 1.0:
            sample_weight = torch.where(
                target_b > 0,
                torch.full_like(loss_bc, positive_sample_weight),
                torch.full_like(loss_bc, negative_sample_weight),
            )
            loss_bc = loss_bc * sample_weight
        return loss_bc.mean()

    for ep in range(1, epochs + 1):
        selector.train()
        perm = train_idx[torch.randperm(train_idx.numel())]
        for b0 in range(0, int(perm.numel()), batch_size):
            batch_idx = perm[b0:b0 + batch_size]
            loss = _loss_for(batch_idx)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(selector.parameters(), grad_clip)
            opt.step()
        hold_metrics = _pred_residual_selector_metrics_from_tensors(
            tensors=tensors,
            selector=selector,
            device=device,
            batch_size=batch_size,
            min_abs_improvement=min_abs_improvement,
            min_rel_improvement=min_rel_improvement,
            indices=hold_idx,
            penalty_names=penalty_names,
        )
        hold_mse = float((hold_metrics or {}).get("selected_mse", float("inf")))
        if hold_mse < best_hold:
            best_hold = hold_mse
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in selector.state_dict().items()}

    if best_state is not None:
        selector.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    margin_raw = cfg.get("decision_margin", "auto")
    margin_selection: Dict[str, object] = {"mode": "fixed", "margin": 0.0}
    if isinstance(margin_raw, str) and margin_raw.lower() == "auto":
        max_margin = max(0.0, float(cfg.get("decision_margin_max", 6.0)))
        num_margin = max(2, int(cfg.get("decision_margin_candidates", 61)))
        best_margin = 0.0
        best_margin_mse = float("inf")
        best_margin_gain = float("-inf")
        margins = torch.linspace(0.0, max_margin, steps=num_margin).tolist()
        for margin in margins:
            selector.decision_margin = float(margin)
            metrics = _pred_residual_selector_metrics_from_tensors(
                tensors=tensors,
                selector=selector,
                device=device,
                batch_size=batch_size,
                min_abs_improvement=min_abs_improvement,
                min_rel_improvement=min_rel_improvement,
                indices=hold_idx,
                penalty_names=penalty_names,
            )
            if metrics is None:
                continue
            mse = float(metrics.get("selected_mse", float("inf")))
            gain = float(metrics.get("selected_gain_pct_vs_base", float("-inf")))
            if mse < best_margin_mse:
                best_margin_mse = mse
                best_margin_gain = gain
                best_margin = float(margin)
        selector.decision_margin = float(best_margin)
        margin_selection = {
            "mode": "auto",
            "margin": float(best_margin),
            "holdout_selected_mse": float(best_margin_mse),
            "holdout_gain_pct_vs_base": float(best_margin_gain),
            "max_margin": float(max_margin),
            "candidates": int(num_margin),
        }
    else:
        selector.decision_margin = float(margin_raw)
        margin_selection = {"mode": "fixed", "margin": float(selector.decision_margin)}
    train_metrics = _pred_residual_selector_metrics_from_tensors(
        tensors=tensors,
        selector=selector,
        device=device,
        batch_size=batch_size,
        min_abs_improvement=min_abs_improvement,
        min_rel_improvement=min_rel_improvement,
        indices=train_idx,
        penalty_names=penalty_names,
    )
    hold_metrics = _pred_residual_selector_metrics_from_tensors(
        tensors=tensors,
        selector=selector,
        device=device,
        batch_size=batch_size,
        min_abs_improvement=min_abs_improvement,
        min_rel_improvement=min_rel_improvement,
        indices=hold_idx,
        penalty_names=penalty_names,
    )
    summary = {
        "enable": True,
        "train_windows": int(train_idx.numel()),
        "holdout_windows": int(hold_idx.numel()),
        "best_epoch": int(best_epoch),
        "label_min_abs_improvement": float(min_abs_improvement),
        "label_min_rel_improvement": float(min_rel_improvement),
        "positive_sample_weight": float(positive_sample_weight),
        "negative_sample_weight": float(negative_sample_weight),
        "class_weight": class_weight_summary,
        "use_penalty_identity": bool(selector.use_penalty_identity),
        "decision_margin": float(selector.decision_margin),
        "decision_margin_selection": margin_selection,
        "feature_standardization": feature_std_summary,
        "channel_names": list(channel_names),
        "penalty_names": list(penalty_names),
        "train": train_metrics,
        "holdout": hold_metrics,
    }
    selector.eval()
    return selector, summary


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
    feature_aware_cfg = cl.get("feature_aware", {}) or {}
    cluster_extra_features_cf = None
    if bool(feature_aware_cfg.get("enable", False)):
        raw_lags = feature_aware_cfg.get("acf_lags", [1, 24, 96])
        if raw_lags is None:
            acf_lags = []
        elif isinstance(raw_lags, (list, tuple)):
            acf_lags = [int(v) for v in raw_lags]
        else:
            acf_lags = [int(raw_lags)]
        cluster_extra_features_cf = compute_channel_shape_features(cluster_fit_tc, acf_lags=acf_lags)
        print(
            "Feature-aware clustering enabled: "
            f"feature_weight={float(feature_aware_cfg.get('weight', 0.0)):.3f}, "
            f"features={int(cluster_extra_features_cf.shape[1])}, acf_lags={acf_lags}"
        )

    # 4) 鑱氱被 + 灏忕皣鍚堝苟绛栫暐
    fixed_cluster_id = cl.get("fixed_cluster_id", None)
    if fixed_cluster_id is not None:
        cluster_id_c = torch.tensor(fixed_cluster_id, dtype=torch.long, device=device)
        if int(cluster_id_c.numel()) != C:
            raise ValueError(
                f"cluster.fixed_cluster_id must contain one id per channel: "
                f"got {int(cluster_id_c.numel())}, expected {C}."
            )
        if int(cluster_id_c.min().item()) < 0:
            raise ValueError("cluster.fixed_cluster_id must be non-negative.")
        # Preserve ids so transfer/fine-tune can map target channels directly
        # onto the corresponding source cluster heads.
        clusters = {
            int(k): (cluster_id_c == int(k)).nonzero(as_tuple=False).view(-1).detach().cpu().tolist()
            for k in range(int(cluster_id_c.max().item()) + 1)
        }
        print("Using fixed channel cluster assignment from cluster.fixed_cluster_id.")
    else:
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
            singleton_merge_strategy=str(cl.get("singleton_merge_strategy", "pool")),
            singleton_merge_distance_threshold=cl.get("singleton_merge_distance_threshold", None),
            singleton_merge_min_size=int(cl.get("singleton_merge_min_size", 2)),
            no_merge_if_channels_lt=int(cl["no_merge_if_channels_lt"]),
            extra_features_cf=cluster_extra_features_cf,
            feature_weight=float(feature_aware_cfg.get("weight", 0.0)) if cluster_extra_features_cf is not None else 0.0,
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
    window_cfg = cfg.get("window", {}) or {}
    past_context = bool(window_cfg.get("past_context", False))
    lazy_windows = bool(window_cfg.get("lazy", False))
    knn_cfg = KNNShapeConfig.from_dict(cfg.get("knn_hybrid", {})).resolved_for_horizon(H)
    if lazy_windows and knn_cfg.enable:
        raise ValueError("window.lazy=true is incompatible with knn_hybrid.enable=true.")
    history_anchor_cfg = cfg.get("model", {}).get("history_anchor", cfg.get("history_anchor", {})) or {}
    history_anchor_cfg = _normalize_history_anchor_cfg(history_anchor_cfg)
    _validate_strict_history_anchor_scope(history_anchor_cfg, source="model.history_anchor")
    history_anchor_active = history_anchor_enabled(history_anchor_cfg)
    if history_anchor_active:
        print(
            "History anchor adapter enabled: "
            f"lags={_parse_positive_ints(history_anchor_cfg.get('lags', ()))}, "
            f"alpha={float(history_anchor_cfg.get('alpha', 0.0)):.3f}, "
            f"blend_target={str(history_anchor_cfg.get('blend_target', 'prediction')).lower()}, "
            f"history_scope={str(history_anchor_cfg.get('history_scope', 'input_window')).lower()}"
        )
    calendar_residual_cfg = cfg.get("calendar_residual", {}) or {}
    calendar_feature_tf = None
    calendar_feature_names: List[str] = []
    calendar_residual_coef_cf = None
    calendar_residual_summary: Dict[str, object] = {
        "enable": bool(calendar_residual_cfg.get("enable", False)),
    }
    if bool(calendar_residual_cfg.get("enable", False)):
        calendar_feature_tf, calendar_feature_names = build_calendar_feature_tensor(
            data_cfg["csv_path"],
            date_col=int(data_cfg["date_col"]),
            max_rows=max_rows,
            cfg=calendar_residual_cfg,
        )
        calendar_feature_tf = calendar_feature_tf.to(device=device)
        calendar_residual_summary.update(
            {
                "feature_names": list(calendar_feature_names),
                "feature_dim": int(calendar_feature_tf.shape[1]),
                "fit_source": str(calendar_residual_cfg.get("source_split", "train")),
                "train_only": True,
            }
        )
        print(
            "Calendar residual adapter enabled: "
            f"features={calendar_feature_names}, source=train"
        )

    val_eval_start = t_train
    test_eval_start = t_val

    if lazy_windows:
        xtr = ytr = xva = yva = xte = yte = None
        dtr = make_lazy_strict_window_dataset(data_window_tc, L, H, 0, t_train)
        train_start_offsets = dtr.start_offsets.clone()
        if past_context:
            dva, val_eval_start = make_lazy_label_range_window_dataset(data_window_tc, L, H, t_train, t_val)
            if skip_test:
                dte = make_lazy_strict_window_dataset(data_window_tc, L, H, 0, 0)
            else:
                dte, test_eval_start = make_lazy_label_range_window_dataset(data_window_tc, L, H, t_val, T)
        else:
            dva = make_lazy_strict_window_dataset(data_window_tc, L, H, t_train, t_val)
            if skip_test:
                dte = make_lazy_strict_window_dataset(data_window_tc, L, H, 0, 0)
            else:
                dte = make_lazy_strict_window_dataset(data_window_tc, L, H, t_val, T)
    else:
        xtr, ytr = make_strict_windows(data_window_tc, L, H, 0, t_train)
        train_start_offsets = torch.arange(0, len(xtr), dtype=torch.long)
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
        dtr = WindowTensorDataset(xtr, ytr)
        dva = WindowTensorDataset(xva, yva)
        dte = WindowTensorDataset(xte, yte)

    print(
        f"Windows: train={len(dtr)}, val={len(dva)}, test={len(dte)}, "
        f"past_context={past_context}, lazy={lazy_windows}"
    )

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
    pin_mem = (device.type == "cuda") and (data_window_tc.device.type == "cpu")
    shuffle_seed = cfg["train"].get("shuffle_seed", None)
    if shuffle_seed is None and bool(cfg["train"].get("fixed_shuffle_seed", False)):
        shuffle_seed = int(cfg["exp"]["seed"])
    train_generator = _make_torch_generator(None if shuffle_seed is None else int(shuffle_seed))
    dl_tr = DataLoader(
        dtr,
        batch_size=bs,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_mem,
        generator=train_generator,
    )
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
    mae_objective_per_cluster_cfg = mae_objective_cfg.get("per_cluster", {}) or {}
    mae_objective_per_cluster_enable = (
        bool(mae_objective_enable)
        and bool(mae_objective_per_cluster_cfg.get("enable", False))
        and mae_objective_weight_final != 0.0
    )
    mae_objective_multiplier_k: Optional[torch.Tensor] = None
    mae_objective_per_cluster_summary: Dict[str, object] = {
        "enable": bool(mae_objective_per_cluster_enable),
    }

    def mae_objective_weight_at(epoch_idx: int):
        if (not mae_objective_enable) or mae_objective_weight_final == 0.0:
            return 0.0
        if mae_objective_warmup_epochs <= 0:
            base_weight = mae_objective_weight_final
        else:
            scale = min(1.0, max(0.0, float(epoch_idx) / float(mae_objective_warmup_epochs)))
            base_weight = mae_objective_weight_final * scale
        return _scale_mae_objective_weight(base_weight, mae_objective_multiplier_k)

    if mae_objective_per_cluster_enable:
        diagnostic_loader = DataLoader(
            dtr,
            batch_size=bs,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        target_bch = _collect_train_targets_bch(
            diagnostic_loader,
            max_windows=int(mae_objective_per_cluster_cfg.get("max_windows", 0) or 0),
        )
        per_cluster_diag = _build_mae_per_cluster_diagnostics_from_targets(
            targets_bch=target_bch,
            cluster_id_c=cluster_id_c,
            K=K,
            base_weight=mae_objective_weight_final,
            cfg=mae_objective_per_cluster_cfg,
        )
        mae_objective_multiplier_k = per_cluster_diag["multiplier_k"].to(device=device, dtype=torch.float32).detach()
        artifact_name = str(mae_objective_per_cluster_cfg.get("artifact", "cluster_mae_weights.csv"))
        artifact_path = artifact_name if os.path.isabs(artifact_name) else os.path.join(out_dir, artifact_name)
        _save_mae_per_cluster_diagnostics_csv(per_cluster_diag["rows"], artifact_path)
        mae_objective_per_cluster_summary = {
            "enable": True,
            "diagnostic": str(mae_objective_per_cluster_cfg.get("diagnostic", "mean_median_gap")),
            "source": str(mae_objective_per_cluster_cfg.get("source", "train_targets")),
            "normalize": str(mae_objective_per_cluster_cfg.get("normalize", "std")),
            "pivot": mae_objective_per_cluster_cfg.get("pivot", "median"),
            "min_multiplier": float(mae_objective_per_cluster_cfg.get("min_multiplier", 1.0)),
            "max_multiplier": float(mae_objective_per_cluster_cfg.get("max_multiplier", 1.25)),
            "artifact": artifact_path,
            "multiplier": [float(v) for v in mae_objective_multiplier_k.detach().cpu().tolist()],
            "effective_weight": [
                float(v) for v in per_cluster_diag["effective_weight_k"].detach().cpu().tolist()
            ],
        }
        print(f"Saved per-cluster MAE objective weights to: {artifact_path}")

    calibration_cfg = cfg.get("calibration", {}) or {}
    calibration_enable = bool(calibration_cfg.get("enable", False))
    calibration_method = str(calibration_cfg.get("method", "median")).lower()
    calibration_shrink = float(calibration_cfg.get("shrink", 1.0))
    calibration_max_abs = float(calibration_cfg.get("max_abs", 0.0))
    # Optional post-hoc shrink sweep. Calibration is an eval-only operation, so a
    # single trained model can be evaluated at many shrink values without
    # retraining. The unit (shrink=1) correction is estimated once on val and
    # then linearly rescaled per shrink — identical results to separate runs at
    # ~1/N the cost and with zero cross-run nondeterminism.
    calibration_shrink_sweep = [float(s) for s in (calibration_cfg.get("shrink_sweep", []) or [])]
    if any(s < 0.0 for s in calibration_shrink_sweep):
        raise ValueError("calibration.shrink_sweep values must be non-negative.")
    # Optional per-channel guarded shrink: each channel picks its own shrink under
    # a per-channel val MSE-regression cap (richer than one global shrink, still
    # MSE-safe by construction). Evaluated post-hoc on the single trained model.
    per_channel_shrink_cfg = calibration_cfg.get("per_channel_shrink", {}) or {}
    per_channel_shrink_enable = bool(per_channel_shrink_cfg.get("enable", False))
    per_channel_shrink_grid = [float(s) for s in (per_channel_shrink_cfg.get("grid", [0.0, 0.3, 0.5, 0.7, 0.85, 1.0]) or [])]
    per_channel_shrink_max_rel_mse = float(per_channel_shrink_cfg.get("max_rel_mse_regression", 0.01))
    if any(s < 0.0 for s in per_channel_shrink_grid):
        raise ValueError("calibration.per_channel_shrink.grid values must be non-negative.")
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
    loss_normalization_cfg = cfg["train"].get("loss_normalization", {}) or {}
    if isinstance(loss_normalization_cfg, bool):
        loss_normalization_cfg = {"enable": bool(loss_normalization_cfg)}
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

    _validate_strict_history_anchor_scope(
        cfg.get("moe", {}).get("history_anchor_expert", {}) or {},
        source="moe.history_anchor_expert",
    )
    model_train_stat_adapter_cfg = cfg.get("model", {}).get("train_stat_adapter", {}) or {}
    (
        model_train_stat_adapter_pc,
        model_train_stat_adapter_counts,
        model_train_stat_adapter_summary,
    ) = build_train_stat_anchor_from_config(
        data_window_tc,
        train_end=t_train,
        input_len=L,
        pred_len=H,
        cfg=model_train_stat_adapter_cfg,
        prefix="model.train_stat_adapter",
    )
    if bool(model_train_stat_adapter_cfg.get("enable", False)):
        print(
            "Model train-stat adapter enabled: "
            f"mode={model_train_stat_adapter_summary.get('mode')}, "
            f"period={model_train_stat_adapter_summary.get('period')}, "
            f"alpha={float(model_train_stat_adapter_cfg.get('alpha', 0.0) or 0.0):.3f}, "
            f"source=train[0:{t_train}]"
        )

    train_stat_anchor_cfg = cfg.get("moe", {}).get("train_stat_anchor_expert", {}) or {}
    train_stat_anchor_pc, train_stat_anchor_counts, train_stat_anchor_summary = build_train_stat_anchor_from_config(
        data_window_tc,
        train_end=t_train,
        input_len=L,
        pred_len=H,
        cfg=train_stat_anchor_cfg,
        prefix="moe.train_stat_anchor_expert",
    )
    train_residual_anchor_cfg = cfg.get("moe", {}).get("train_residual_anchor_expert", {}) or {}
    train_residual_anchor_phc = None
    train_residual_anchor_summary: Dict[str, object] = {
        "enable": bool(train_residual_anchor_cfg.get("enable", False)),
    }
    if bool(train_stat_anchor_cfg.get("enable", False)):
        print(
            "Train-stat anchor expert enabled: "
            f"mode={train_stat_anchor_summary.get('mode')}, period={train_stat_anchor_summary.get('period')}, "
            f"alpha={float(train_stat_anchor_cfg.get('alpha', 0.0) or 0.0):.3f}, "
            f"source=train[0:{t_train}]"
        )

    def eval_loop_with_history(*args, **kwargs):
        kwargs.setdefault("history_anchor_cfg", history_anchor_cfg)
        kwargs.setdefault("observed_history_tc", data_window_tc)
        kwargs.setdefault("input_len", L)
        kwargs.setdefault("model_train_stat_adapter_pc", model_train_stat_adapter_pc)
        kwargs.setdefault("model_train_stat_adapter_cfg", model_train_stat_adapter_cfg)
        kwargs.setdefault("train_stat_anchor_pc", train_stat_anchor_pc)
        kwargs.setdefault("train_residual_anchor_phc", train_residual_anchor_phc)
        kwargs.setdefault("calendar_feature_tf", calendar_feature_tf)
        kwargs.setdefault("calendar_residual_coef_cf", calendar_residual_coef_cf)
        return eval_loop(*args, **kwargs)

    # cluster portraits (prototype + penalty metrics)
    portrait_cfg = cfg.get("portrait", {})
    gate_prior_cfg = cfg.get("moe", {}).get("gate_prior", {})
    cluster_penalty_prior_cfg = cfg.get("moe", {}).get("cluster_penalty_prior", {}) or {}
    channel_penalty_prior_cfg = cfg.get("moe", {}).get("channel_penalty_prior", {}) or {}
    need_penalty_portrait = (
        bool(portrait_cfg.get("enable", False))
        or bool(gate_prior_cfg.get("enable", False))
        or bool(cluster_penalty_prior_cfg.get("enable", False))
        or bool(channel_penalty_prior_cfg.get("enable", False))
    )
    penalty_portrait_kp = None
    channel_penalty_portrait_cp = None
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
        if bool(channel_penalty_prior_cfg.get("enable", False)):
            channel_penalty_portrait_cp = compute_channel_penalty_portrait(
                portrait_loader, penalty_names, penalty_fns, C, H, device
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
    router_penalty_context_score = str(moe_cfg.get("router_penalty_context_score", "high_violation")).lower()
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
    skip_supervision_weight = float(moe_cfg.get("skip_supervision_weight", 0.0)) if allow_skip else 0.0
    skip_supervision_margin = float(moe_cfg.get("skip_supervision_margin", 0.0))
    mse_utility_gate_cfg = moe_cfg.get("mse_utility_gate_supervision", {}) or {}
    mse_utility_gate_enable = (
        bool(mse_utility_gate_cfg.get("enable", False))
        and moe_enable
        and pred_residual_enable
        and P > 0
    )
    mse_utility_gate_weight = float(mse_utility_gate_cfg.get("weight", 0.0)) if mse_utility_gate_enable else 0.0
    mse_utility_gate_temperature = float(mse_utility_gate_cfg.get("temperature", 1.0))
    mse_utility_gate_min_gain = float(mse_utility_gate_cfg.get("min_gain", 0.0))
    mse_utility_gate_target_power = float(mse_utility_gate_cfg.get("target_power", 1.0))
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
    channel_expert_mask_c = None
    channel_expert_cfg = pred_residual_cfg.get("channel_expert_adapters", {}) or {}
    if pred_residual_enable and bool(channel_expert_cfg.get("enable", False)):
        raw_cluster_id_c, _ = cluster_channels_by_corr(
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
            min_cluster_size=1,
            merge_small_clusters=False,
            no_merge_if_channels_lt=int(cl["no_merge_if_channels_lt"]),
            extra_features_cf=cluster_extra_features_cf,
            feature_weight=float(feature_aware_cfg.get("weight", 0.0)) if cluster_extra_features_cf is not None else 0.0,
        )
        raw_sizes = torch.bincount(raw_cluster_id_c, minlength=int(raw_cluster_id_c.max().item() + 1))
        final_sizes = torch.bincount(cluster_id_c, minlength=K)
        mode = str(channel_expert_cfg.get("mode", "merged_singletons")).lower()
        if mode in {"all", "all_channels"}:
            channel_expert_mask_c = torch.ones(C, dtype=torch.bool, device=device)
        else:
            channel_expert_mask_c = (
                (raw_sizes[raw_cluster_id_c].to(device=device) == 1)
                & (final_sizes[cluster_id_c].to(device=device) > 1)
            )
        print(
            "Channel expert adapters enabled: "
            f"mode={mode}, channels={int(channel_expert_mask_c.sum().item())}/{C}, "
            f"mask={[bool(v) for v in channel_expert_mask_c.detach().cpu().tolist()]}"
        )
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
            penalty_selector_enable=bool(pred_residual_cfg.get("penalty_selector_enable", False)),
            selector_temperature=float(pred_residual_cfg.get("selector_temperature", 1.0)),
            selector_use_cluster_context=bool(pred_residual_cfg.get("selector_use_cluster_context", True)),
            fusion_gate_enable=bool(pred_residual_cfg.get("fusion_gate_enable", False)),
            fusion_init=float(pred_residual_cfg.get("fusion_init", 0.0)),
            fusion_use_cluster_context=bool(pred_residual_cfg.get("fusion_use_cluster_context", True)),
            num_channels=C,
            channel_expert_mask_c=channel_expert_mask_c,
            channel_expert_cluster_id_c=cluster_id_c,
            channel_expert_mode=str((pred_residual_cfg.get("channel_expert_adapters", {}) or {}).get("mode_type", "override")),
            penalty_names=penalty_names,
            seasonal_anchor_names=list(pred_residual_cfg.get("seasonal_anchor_names", [])),
            seasonal_anchor_period=int(pred_residual_cfg.get("seasonal_anchor_period", 96)),
            seasonal_anchor_num_periods=int(pred_residual_cfg.get("seasonal_anchor_num_periods", 1)),
            seasonal_anchor_scale=float(pred_residual_cfg.get("seasonal_anchor_scale", 1.0)),
        ).to(device)
        print(
            "Prediction residual MoE enabled: "
            f"hidden={pred_residual.hidden_dim}, feature_mode={pred_residual.feature_mode}, "
            f"alpha_scale={pred_residual.alpha_scale:.3f}, "
            f"residual_clip={pred_residual.residual_clip:.3f}, "
            f"seasonal_anchor_names={list(pred_residual_cfg.get('seasonal_anchor_names', []))}, "
            f"seasonal_anchor_period={int(pred_residual_cfg.get('seasonal_anchor_period', 96))}, "
            f"seasonal_anchor_scale={float(pred_residual_cfg.get('seasonal_anchor_scale', 1.0)):.3f}, "
            f"specialization_weight={pred_residual_specialization_weight:.6f}, "
            f"norm_weight={pred_residual_norm_weight:.6f}, "
            f"intervention_weight={pred_residual_intervention_weight:.6f}, "
            f"detach_routed_penalty_pred={pred_residual_detach_routed_penalty_pred}, "
            f"penalty_selector={pred_residual.penalty_selector_enable}, "
            f"fusion_gate={pred_residual.fusion_gate_enable}"
        )
    gate_balance_target_kp = None
    gate_prior_prob_kp = None
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
    cluster_penalty_prior_enable = (
        bool(cluster_penalty_prior_cfg.get("enable", False))
        and penalty_portrait_kp is not None
        and P > 0
    )
    cluster_penalty_prior_prob_kp = None
    cluster_penalty_allowed_mask_kp = None
    if cluster_penalty_prior_enable:
        cluster_penalty_prior_prob_kp = build_gate_prior_from_penalty_portrait(
            penalty_kp=penalty_portrait_kp,
            penalty_scale=penalty_scale,
            temperature=float(cluster_penalty_prior_cfg.get("temperature", 1.0)),
            smoothing=float(cluster_penalty_prior_cfg.get("smoothing", 0.0)),
            use_normalized_penalty=bool(cluster_penalty_prior_cfg.get("use_normalized_penalty", True)),
        )
        logit_strength = float(cluster_penalty_prior_cfg.get("logit_strength", 0.0))
        if logit_strength > 0.0:
            gate.set_penalty_prior(cluster_penalty_prior_prob_kp, strength=logit_strength)
        topk = int(cluster_penalty_prior_cfg.get("topk", 0))
        manual_allowed = build_named_penalty_mask(
            cluster_penalty_prior_cfg.get("allowed_by_cluster", None),
            penalty_names,
            K,
            device,
        )
        if manual_allowed is not None:
            cluster_penalty_allowed_mask_kp = manual_allowed
            gate.set_penalty_allowed_mask(cluster_penalty_allowed_mask_kp)
        elif topk > 0 and bool(cluster_penalty_prior_cfg.get("hard_topk", True)):
            cluster_penalty_allowed_mask_kp = build_topk_penalty_mask(cluster_penalty_prior_prob_kp, topk=topk)
            gate.set_penalty_allowed_mask(cluster_penalty_allowed_mask_kp)
        always_include = cluster_penalty_prior_cfg.get("always_include", []) or []
        if isinstance(always_include, str):
            always_include = [always_include]
        if len(always_include) > 0:
            if cluster_penalty_allowed_mask_kp is None:
                cluster_penalty_allowed_mask_kp = torch.zeros((K, P), device=device, dtype=torch.float32)
            name_to_idx = {str(name): i for i, name in enumerate(penalty_names)}
            for raw_name in always_include:
                name = str(raw_name)
                if name not in name_to_idx:
                    raise ValueError(
                        "cluster_penalty_prior.always_include contains unknown penalty "
                        f"{name!r}; available={penalty_names}"
                    )
                cluster_penalty_allowed_mask_kp[:, name_to_idx[name]] = 1.0
            empty = cluster_penalty_allowed_mask_kp.sum(dim=-1, keepdim=True) <= 0.0
            if bool(empty.any().item()):
                cluster_penalty_allowed_mask_kp = torch.where(
                    empty,
                    torch.ones_like(cluster_penalty_allowed_mask_kp),
                    cluster_penalty_allowed_mask_kp,
                )
            gate.set_penalty_allowed_mask(cluster_penalty_allowed_mask_kp)
        if bool(cluster_penalty_prior_cfg.get("use_as_balance_target", False)):
            gate_balance_target_kp = cluster_penalty_prior_prob_kp
        prior_list = (
            cluster_penalty_prior_prob_kp.detach().cpu().tolist()
            if cluster_penalty_prior_prob_kp is not None
            else None
        )
        mask_list = (
            cluster_penalty_allowed_mask_kp.detach().cpu().tolist()
            if cluster_penalty_allowed_mask_kp is not None
            else None
        )
        print(
            "Cluster penalty prior enabled: "
            f"topk={topk}, hard_topk={bool(cluster_penalty_prior_cfg.get('hard_topk', True))}, "
            f"logit_strength={logit_strength:.3f}, prior={prior_list}, allowed_mask={mask_list}"
        )
    if (
        bool(channel_penalty_prior_cfg.get("enable", False))
        and pred_residual is not None
        and channel_penalty_portrait_cp is not None
        and P > 0
    ):
        channel_penalty_prior_prob_cp = build_gate_prior_from_penalty_portrait(
            penalty_kp=channel_penalty_portrait_cp,
            penalty_scale=penalty_scale,
            temperature=float(channel_penalty_prior_cfg.get("temperature", 1.0)),
            smoothing=float(channel_penalty_prior_cfg.get("smoothing", 0.0)),
            use_normalized_penalty=bool(channel_penalty_prior_cfg.get("use_normalized_penalty", True)),
        )
        topk = int(channel_penalty_prior_cfg.get("topk", 0))
        if topk > 0 and bool(channel_penalty_prior_cfg.get("hard_topk", True)):
            channel_penalty_allowed_mask_cp = build_topk_penalty_mask(channel_penalty_prior_prob_cp, topk=topk)
            pred_residual.set_channel_penalty_allowed_mask(channel_penalty_allowed_mask_cp)
        else:
            channel_penalty_allowed_mask_cp = None
        print(
            "Channel penalty prior enabled: "
            f"topk={topk}, hard_topk={bool(channel_penalty_prior_cfg.get('hard_topk', True))}, "
            f"allowed_mask={channel_penalty_allowed_mask_cp.detach().cpu().tolist() if channel_penalty_allowed_mask_cp is not None else None}"
        )

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
        partial_model_state = bool(ft_cfg.get("partial_model_state", ft_cfg.get("partial_model", False)))
        if bool(ft_cfg.get("strict_window", True)) and (src_input_len != L or src_pred_len != H):
            raise ValueError(
                "Fine-tune checkpoint window mismatch: "
                f"source input_len/pred_len={src_input_len}/{src_pred_len}, target={L}/{H}. "
                "Train or choose a source checkpoint with the same horizon."
            )
        if src_k_count <= 0:
            raise ValueError(f"Invalid source cluster count in fine-tune checkpoint: {src_k_count}")

        src_model_cfg = dict(meta.get("model_cfg", {}))
        src_compare_model_cfg = dict(src_model_cfg)
        tgt_compare_model_cfg = dict(model_cfg)
        src_compare_model_cfg.pop("history_anchor", None)
        tgt_compare_model_cfg.pop("history_anchor", None)
        if bool(ft_cfg.get("strict_model", True)) and src_compare_model_cfg != tgt_compare_model_cfg:
            raise ValueError("Fine-tune source model_cfg differs from target model_cfg.")
        src_cluster_id_c = meta.get("cluster_id_c", None)
        src_num_channels = meta.get("num_channels", None)
        if bool(dict(src_model_cfg.get("channel_adapter", {}) or {}).get("enable", False)):
            if src_cluster_id_c is None or src_num_channels is None:
                raise ValueError("Fine-tune source checkpoint with channel_adapter requires cluster_id_c and num_channels in meta.")
        source_model = None
        if bool(ft_cfg.get("load_model", True)) and not partial_model_state:
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

        def load_finetune_model_cluster_state(k: int, src_k: int) -> None:
            try:
                model.load_cluster_state(k, source_model.get_cluster_state(src_k))
                return
            except ValueError as exc:
                if "channel_head_mlp cluster" not in str(exc):
                    raise
                required = ("W1", "b1", "W2", "b2", "_cluster_channel_idx")
                if not all(hasattr(model, name) for name in required) or not all(hasattr(source_model, name) for name in required):
                    raise
                device = model.W1[k].device
                model.W1[k].data.copy_(source_model.W1[src_k].to(device))
                model.b1[k].data.copy_(source_model.b1[src_k].to(device))
                target_idx = model._cluster_channel_idx(k)
                for i in target_idx:
                    c = int(i.item())
                    if c >= len(source_model.W2):
                        raise ValueError(f"Fine-tune channel-head transfer missing source channel {c}.") from exc
                    model.W2[c].data.copy_(source_model.W2[c].to(device))
                    model.b2[c].data.copy_(source_model.b2[c].to(device))
                print(
                    "Fine-tune channel_head_mlp warm start used source cluster shared layer "
                    f"{src_k}->target {k} and channel-index output heads."
                )

        partial_model_summary = None
        if bool(ft_cfg.get("load_model", True)):
            if partial_model_state:
                if "model_state" not in ckpt:
                    raise ValueError(f"Fine-tune checkpoint is missing model_state: {ckpt_path}")
                partial_model_summary = _partial_load_matching_state_dict(model, ckpt["model_state"])
                if int(partial_model_summary["loaded_count"]) <= 0 and not bool(ft_cfg.get("allow_empty_partial_model", False)):
                    raise ValueError(
                        "Fine-tune partial_model_state loaded zero tensors. "
                        "Check that source and target predictors share parameter names."
                    )
                print(
                    "Fine-tune partial model warm start: "
                    f"loaded={partial_model_summary['loaded_count']}, "
                    f"skipped_shape={partial_model_summary['skipped_shape_count']}, "
                    f"skipped_missing={partial_model_summary['skipped_missing_count']}"
                )
            else:
                assert source_model is not None
                for k in range(K):
                    src_k = int(target_to_source_k[k].item())
                    load_finetune_model_cluster_state(k, src_k)

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
            "partial_model_state": partial_model_state,
            "partial_model_load": partial_model_summary,
        }
        print(f"Fine-tune warm start loaded from: {ckpt_path}")
        print(f"Fine-tune target->source cluster map: {finetune_summary['target_to_source_cluster']}")

    apply_finetune_warm_start()

    freeze_backbone = bool(moe_cfg.get("freeze_backbone", cfg.get("train", {}).get("freeze_backbone", False)))
    frozen_backbone_params = 0
    if freeze_backbone:
        frozen_backbone_params = _freeze_module_params(model)
        print(f"Backbone frozen for MoE training: params={frozen_backbone_params}")

    cluster_params = []
    for k in range(K):
        params_k = []
        if not freeze_backbone:
            params_k.extend(model.get_cluster_params(k))
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
    for opt_k in optimizers:
        for group in opt_k.param_groups:
            group.setdefault("initial_lr", float(group["lr"]))
    sched_cfg = cfg["train"].get("lr_scheduler", {"name": "none"})
    lr_warmup_epochs = int(sched_cfg.get("warmup_epochs", cfg["train"].get("lr_warmup_epochs", 0)))
    lr_warmup_start_factor = float(
        sched_cfg.get("warmup_start_factor", cfg["train"].get("lr_warmup_start_factor", 0.1))
    )
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

    swa_cfg = cfg["train"].get("swa", {}) or {}
    if isinstance(swa_cfg, bool):
        swa_cfg = {"enable": bool(swa_cfg)}
    swa_enable = bool(swa_cfg.get("enable", False))
    swa_start_epoch = int(
        swa_cfg.get(
            "start_epoch",
            max(1, int(math.ceil(float(epochs) * float(swa_cfg.get("start_fraction", 0.75))))),
        )
    )
    swa_update_every = max(1, int(swa_cfg.get("update_every", 1)))
    swa_selection_metric = str(swa_cfg.get("selection_metric", "val_mse")).lower()
    if swa_selection_metric not in {"val_loss", "val_mse", "val_mae"}:
        raise ValueError("train.swa.selection_metric must be val_loss, val_mse, or val_mae.")
    swa_min_delta = float(swa_cfg.get("min_delta", 0.0))
    swa_averagers = {}
    swa_updates = 0
    swa_summary = {
        "enable": bool(swa_enable),
        "selected": False,
        "updates": 0,
        "start_epoch": int(swa_start_epoch),
        "update_every": int(swa_update_every),
        "selection_metric": str(swa_selection_metric),
    }
    if swa_enable:
        swa_averagers["model"] = torch.optim.swa_utils.AveragedModel(model)
        swa_averagers["gate"] = torch.optim.swa_utils.AveragedModel(gate)
        if pred_residual is not None:
            swa_averagers["pred_residual"] = torch.optim.swa_utils.AveragedModel(pred_residual)
        if dynamic_lambda is not None:
            swa_averagers["dynamic_lambda"] = torch.optim.swa_utils.AveragedModel(dynamic_lambda)
        if learnable_lambda is not None:
            swa_averagers["learnable_lambda"] = torch.optim.swa_utils.AveragedModel(learnable_lambda)

    def update_swa_averagers(epoch_idx: int) -> None:
        nonlocal swa_updates
        if not swa_enable or not _should_update_swa(epoch_idx, swa_start_epoch, swa_update_every):
            return
        swa_averagers["model"].update_parameters(model)
        swa_averagers["gate"].update_parameters(gate)
        if pred_residual is not None and "pred_residual" in swa_averagers:
            swa_averagers["pred_residual"].update_parameters(pred_residual)
        if dynamic_lambda is not None and "dynamic_lambda" in swa_averagers:
            swa_averagers["dynamic_lambda"].update_parameters(dynamic_lambda)
        if learnable_lambda is not None and "learnable_lambda" in swa_averagers:
            swa_averagers["learnable_lambda"].update_parameters(learnable_lambda)
        swa_updates += 1

    def load_swa_averagers() -> None:
        model.load_state_dict(swa_averagers["model"].module.state_dict())
        gate.load_state_dict(swa_averagers["gate"].module.state_dict())
        if pred_residual is not None and "pred_residual" in swa_averagers:
            pred_residual.load_state_dict(swa_averagers["pred_residual"].module.state_dict())
        if dynamic_lambda is not None and "dynamic_lambda" in swa_averagers:
            dynamic_lambda.load_state_dict(swa_averagers["dynamic_lambda"].module.state_dict())
        if learnable_lambda is not None and "learnable_lambda" in swa_averagers:
            learnable_lambda.load_state_dict(swa_averagers["learnable_lambda"].module.state_dict())

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
            observed_history_tc=data_window_tc,
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
                observed_history_tc=data_window_tc,
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

    def _aggregate_val_metric(
        val_loss_k: torch.Tensor,
        val_mse_k: torch.Tensor,
        val_mae_k: torch.Tensor,
        metric: str,
    ) -> float:
        metric = str(metric).lower()
        if metric == "val_loss":
            value_k = val_loss_k
        elif metric == "val_mae":
            value_k = val_mae_k
        elif metric == "val_mse":
            value_k = val_mse_k
        else:
            raise ValueError("SWA selection metric must be val_loss, val_mse, or val_mae.")
        return float(reduce_cluster_metric(value_k, cluster_weight_k).item())

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
            _, probs_bkp, skip_bk, skip_prob_bk = gate(
                feat_bkf,
                straight_through=False,
                penalty_context_bkp=pen_bkp,
                penalty_context_mode=router_mode,
                penalty_context_weight=router_penalty_context_weight,
                penalty_context_detach=router_detach_penalty_context,
                penalty_context_score=router_penalty_context_score,
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
            mask_bkp, probs_bkp, skip_bk, _ = gate(
                feat_bkf,
                straight_through=False,
                penalty_context_bkp=route_pen_bkp,
                penalty_context_mode=router_mode,
                penalty_context_weight=router_penalty_context_weight,
                penalty_context_detach=router_detach_penalty_context,
                penalty_context_score=router_penalty_context_score,
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
        idx: torch.Tensor,
        base_lambda_kp: torch.Tensor,
        model_params: Optional[Dict[str, torch.Tensor]] = None,
        gate_params: Optional[Dict[str, torch.Tensor]] = None,
        pred_residual_params: Optional[Dict[str, torch.Tensor]] = None,
        dynamic_lambda_params: Optional[Dict[str, torch.Tensor]] = None,
        straight_through: bool = True,
        mae_objective_weight=0.0,
    ) -> Dict[str, torch.Tensor]:
        x_model = apply_train_stat_input_centering(
            x,
            query_start_abs_b=idx,
            stat_anchor_pc=model_train_stat_adapter_pc,
            cfg=model_train_stat_adapter_cfg,
        )
        yhat_base_raw = _module_call(model, model_params, x_model, cluster_id_c)
        yhat_base = apply_history_anchor_adapter(
            yhat_base_raw,
            base_pred_bch=yhat_base_raw,
            observed_history_tc=data_window_tc,
            query_start_abs_b=idx,
            input_len=L,
            cfg=history_anchor_cfg,
        )
        yhat_base = apply_train_stat_anchor_expert(
            yhat_base,
            base_pred_bch=yhat_base,
            x_bcl=x,
            query_start_abs_b=idx,
            input_len=L,
            stat_anchor_pc=model_train_stat_adapter_pc,
            cfg=model_train_stat_adapter_cfg,
        )
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
                penalty_context_bkp=route_pen_bkp,
                penalty_context_mode=router_mode,
                penalty_context_weight=router_penalty_context_weight,
                penalty_context_detach=router_detach_penalty_context,
                penalty_context_score=router_penalty_context_score,
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
        if _mae_objective_weight_is_nonzero(mae_objective_weight):
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
        loss_terms_bk, _ = _normalize_loss_terms(
            {
                "mse": mse_bk,
                "mae_objective": mae_objective_bk,
                "penalty": penalty_loss_bk,
                "pred_residual": pred_loss_terms["total_bk"],
            },
            loss_normalization_cfg,
        )
        objective_loss_bk = (
            (mse_weight * loss_terms_bk["mse"])
            + _apply_mae_objective_weight(loss_terms_bk["mae_objective"], mae_objective_weight)
            + loss_terms_bk["penalty"]
            + loss_terms_bk["pred_residual"]
        )
        if mse_utility_gate_weight > 0.0:
            mse_gate_loss_bk = _mse_utility_gate_supervision_loss(
                probs_bkp=probs_bkp,
                allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                y_base_bch=yhat_base,
                pred_out=pred_out,
                y_bch=y,
                cluster_id_c=cluster_id_c,
                K=K,
                temperature=mse_utility_gate_temperature,
                min_gain=mse_utility_gate_min_gain,
                target_power=mse_utility_gate_target_power,
            )
            if mse_gate_loss_bk is not None:
                objective_loss_bk = objective_loss_bk + mse_utility_gate_weight * mse_gate_loss_bk
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
    inner_modules = []
    if not freeze_backbone:
        inner_modules.append(("model", model))
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
        x_tr, y_tr, idx_tr = train_batch
        x_va, y_va, idx_va = val_batch
        x_tr = x_tr.to(device, non_blocking=True)
        y_tr = y_tr.to(device, non_blocking=True)
        idx_tr = idx_tr.to(device=device, dtype=torch.long, non_blocking=True)
        x_va = x_va.to(device, non_blocking=True)
        y_va = y_va.to(device, non_blocking=True)
        idx_va = idx_va.to(device=device, dtype=torch.long, non_blocking=True)

        base_lambda_kp = lambda_kp_at(epoch_idx, detach=False) * warmup_scale
        train_terms = compute_batch_terms(
            x_tr, y_tr, idx_tr,
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
            x_va, y_va, idx_va,
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
        if lr_warmup_epochs > 0 and ep <= lr_warmup_epochs:
            _set_optimizer_lr_scale(
                optimizers,
                _lr_warmup_scale(ep, lr_warmup_epochs, lr_warmup_start_factor),
            )
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

            x_model = apply_train_stat_input_centering(
                x,
                query_start_abs_b=idx,
                stat_anchor_pc=model_train_stat_adapter_pc,
                cfg=model_train_stat_adapter_cfg,
            )
            yhat_base_raw = model(x_model, cluster_id_c)
            yhat_base = apply_history_anchor_adapter(
                yhat_base_raw,
                base_pred_bch=yhat_base_raw,
                observed_history_tc=data_window_tc,
                query_start_abs_b=idx,
                input_len=L,
                cfg=history_anchor_cfg,
            )
            yhat_base = apply_train_stat_anchor_expert(
                yhat_base,
                base_pred_bch=yhat_base,
                x_bcl=x,
                query_start_abs_b=idx,
                input_len=L,
                stat_anchor_pc=model_train_stat_adapter_pc,
                cfg=model_train_stat_adapter_cfg,
            )
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
                mask_bkp, probs_bkp, skip_bk, skip_prob_bk = gate(
                    feat_bkf,
                    straight_through=straight_through,
                    penalty_context_bkp=route_pen_bkp,
                    penalty_context_mode=router_mode,
                    penalty_context_weight=router_penalty_context_weight,
                    penalty_context_detach=router_detach_penalty_context,
                    penalty_context_score=router_penalty_context_score,
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
            if _mae_objective_weight_is_nonzero(mae_objective_weight_ep):
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
                raw_objective_loss_bk = (
                    (mse_weight * mse_bk)
                    + _apply_mae_objective_weight(mae_objective_bk, mae_objective_weight_ep)
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
                loss_terms_bk, _ = _normalize_loss_terms(
                    {
                        "mse": mse_bk,
                        "mae_objective": mae_objective_bk,
                        "penalty": penalty_loss_bk,
                        "pred_residual": pred_loss_terms["total_bk"],
                    },
                    loss_normalization_cfg,
                )
                objective_loss_bk = (
                    (mse_weight * loss_terms_bk["mse"])
                    + _apply_mae_objective_weight(loss_terms_bk["mae_objective"], mae_objective_weight_ep)
                    + loss_terms_bk["penalty"]
                )
                loss_bk = objective_loss_bk + loss_terms_bk["pred_residual"]
                if (
                    allow_skip
                    and skip_supervision_weight > 0.0
                    and pred_residual is not None
                    and skip_prob_bk is not None
                ):
                    with torch.no_grad():
                        pred_no_skip = pred_residual(
                            x,
                            yhat_base,
                            cluster_id_c,
                            mask_bkp.detach(),
                            skip_bk=None,
                        )
                        yhat_no_skip = pred_no_skip["y_final"]
                        base_mse_bc_for_skip = (yhat_base - y).pow(2).mean(dim=-1)
                        no_skip_mse_bc = (yhat_no_skip - y).pow(2).mean(dim=-1)
                        base_mse_bk_for_skip = scatter_mean_bc_to_bk(base_mse_bc_for_skip, cluster_id_c, K)
                        no_skip_mse_bk = scatter_mean_bc_to_bk(no_skip_mse_bc, cluster_id_c, K)
                        skip_label_bk = (
                            base_mse_bk_for_skip + float(skip_supervision_margin) < no_skip_mse_bk
                        ).to(dtype=skip_prob_bk.dtype)
                    skip_prob_clamped = skip_prob_bk.clamp(1.0e-6, 1.0 - 1.0e-6)
                    skip_bce_bk = -(
                        skip_label_bk * skip_prob_clamped.log()
                        + (1.0 - skip_label_bk) * (1.0 - skip_prob_clamped).log()
                    )
                    loss_bk = loss_bk + skip_supervision_weight * skip_bce_bk
                if mse_utility_gate_weight > 0.0:
                    mse_gate_loss_bk = _mse_utility_gate_supervision_loss(
                        probs_bkp=probs_bkp,
                        allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                        y_base_bch=yhat_base,
                        pred_out=pred_out,
                        y_bch=y,
                        cluster_id_c=cluster_id_c,
                        K=K,
                        temperature=mse_utility_gate_temperature,
                        min_gain=mse_utility_gate_min_gain,
                        target_power=mse_utility_gate_target_power,
                    )
                    if mse_gate_loss_bk is not None:
                        loss_bk = loss_bk + mse_utility_gate_weight * mse_gate_loss_bk
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
                raw_objective_loss_bk = (
                    (mse_weight * mse_bk)
                    + _apply_mae_objective_weight(mae_objective_bk, mae_objective_weight_ep)
                )
                loss_terms_bk, _ = _normalize_loss_terms(
                    {
                        "mse": mse_bk,
                        "mae_objective": mae_objective_bk,
                        "penalty": torch.zeros_like(mse_bk),
                        "pred_residual": torch.zeros_like(mse_bk),
                    },
                    loss_normalization_cfg,
                )
                objective_loss_bk = (
                    (mse_weight * loss_terms_bk["mse"])
                    + _apply_mae_objective_weight(loss_terms_bk["mae_objective"], mae_objective_weight_ep)
                )
                loss_bk = objective_loss_bk
            _accumulate_detached_sum_(train_loss_sum_k, raw_objective_loss_bk)
            _accumulate_detached_sum_(train_mse_sum_k, mse_bk)
            _accumulate_detached_sum_(train_mae_sum_k, mae_bk)
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
        val_loss_k, val_mse_k, val_mae_k, _, _, _, _, _ = eval_loop_with_history(
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
        update_swa_averagers(ep)

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

        if schedulers is not None and ep > lr_warmup_epochs:
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
    swa_summary["updates"] = int(swa_updates)
    if swa_enable and swa_updates <= 0:
        swa_summary["reason"] = "no_swa_updates"
    if swa_enable and swa_updates > 0:
        swa_mae_eval_weight = _scale_mae_objective_weight(
            mae_objective_weight_final if mae_objective_enable else 0.0,
            mae_objective_multiplier_k,
        )
        lam_kp_for_swa_eval = lambda_kp_from_epochs(best_epoch)
        best_val_loss_k, best_val_mse_k, best_val_mae_k, _, _, _, _, _ = eval_loop_with_history(
            model, gate, lam_kp_for_swa_eval,
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
            mae_objective_weight=swa_mae_eval_weight,
            mae_objective_kind=mae_objective_kind,
            mae_objective_beta=mae_objective_beta,
            pred_residual=pred_residual,
            knn_hybrid=knn_hybrid_val if knn_cfg.use_for_model_selection else None,
            eval_start=val_eval_start,
        )
        best_swa_metric = _aggregate_val_metric(
            best_val_loss_k,
            best_val_mse_k,
            best_val_mae_k,
            swa_selection_metric,
        )
        load_swa_averagers()
        swa_val_loss_k, swa_val_mse_k, swa_val_mae_k, _, _, _, _, _ = eval_loop_with_history(
            model, gate, lam_kp_for_swa_eval,
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
            mae_objective_weight=swa_mae_eval_weight,
            mae_objective_kind=mae_objective_kind,
            mae_objective_beta=mae_objective_beta,
            pred_residual=pred_residual,
            knn_hybrid=knn_hybrid_val if knn_cfg.use_for_model_selection else None,
            eval_start=val_eval_start,
        )
        swa_metric = _aggregate_val_metric(
            swa_val_loss_k,
            swa_val_mse_k,
            swa_val_mae_k,
            swa_selection_metric,
        )
        use_swa = (best_swa_metric - swa_metric) > swa_min_delta
        if not use_swa:
            load_best_all()
        swa_summary.update(
            {
                "selected": bool(use_swa),
                "best_metric": float(best_swa_metric),
                "swa_metric": float(swa_metric),
                "min_delta": float(swa_min_delta),
            }
        )
        print(
            "SWA selection: "
            f"updates={swa_updates}, metric={swa_selection_metric}, "
            f"best={best_swa_metric:.6f}, swa={swa_metric:.6f}, "
            f"selected={bool(use_swa)}"
        )
    if knn_cfg.enable and knn_requires_base_bank:
        _refresh_knn_hybrids()

    def _run_train_stat_anchor_scale_selection() -> bool:
        train_stat_scale_selection_cfg = train_stat_anchor_cfg.get("scale_selection", {}) or {}
        if not (
            bool(train_stat_anchor_cfg.get("enable", False))
            and bool(train_stat_scale_selection_cfg.get("enable", False))
            and train_stat_anchor_pc is not None
        ):
            return False
        horizon_segments = int(train_stat_scale_selection_cfg.get("horizon_segments", 1))
        scales_c, scores_c, selection_count = select_train_stat_anchor_scales_from_loader(
            model=model,
            loader=dl_va,
            cluster_id_c=cluster_id_c,
            device=device,
            history_anchor_cfg=history_anchor_cfg,
            observed_history_tc=data_window_tc,
            input_len=L,
            eval_start=val_eval_start,
            stat_anchor_pc=train_stat_anchor_pc,
            train_stat_anchor_cfg=train_stat_anchor_cfg,
            metric=str(train_stat_scale_selection_cfg.get("metric", "mse")),
            max_scale=float(train_stat_scale_selection_cfg.get("max_scale", 0.3)),
            steps=int(train_stat_scale_selection_cfg.get("steps", 13)),
            horizon_segments=horizon_segments,
            model_train_stat_adapter_pc=model_train_stat_adapter_pc,
            model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
        )
        if int(scales_c.ndim) == 2:
            alpha_values = [[float(v) for v in row] for row in scales_c.tolist()]
            train_stat_anchor_cfg["alpha_by_channel_horizon"] = alpha_values
            train_stat_anchor_cfg["alpha_horizon_segments"] = int(horizon_segments)
            alpha_key = "alpha_by_channel_horizon"
        else:
            alpha_values = [float(v) for v in scales_c.tolist()]
            train_stat_anchor_cfg["alpha_by_channel"] = alpha_values
            alpha_key = "alpha_by_channel"
        train_stat_anchor_summary["scale_selection"] = {
            "enable": True,
            "source_split": "val",
            "metric": str(train_stat_scale_selection_cfg.get("metric", "mse")),
            "max_scale": float(train_stat_scale_selection_cfg.get("max_scale", 0.3)),
            "steps": int(train_stat_scale_selection_cfg.get("steps", 13)),
            "horizon_segments": int(horizon_segments),
            "num_windows": int(selection_count),
            alpha_key: alpha_values,
            "score": [[float(v) for v in row] for row in scores_c.tolist()]
            if int(scores_c.ndim) == 2
            else [float(v) for v in scores_c.tolist()],
            "mean_alpha": float(scales_c.mean().item()) if int(scales_c.numel()) > 0 else 0.0,
        }
        print(
            "Train-stat anchor scale selection: "
            f"metric={train_stat_anchor_summary['scale_selection']['metric']}, "
            f"mean_alpha={train_stat_anchor_summary['scale_selection']['mean_alpha']:.4f}, "
            f"horizon_segments={int(horizon_segments)}, channels={int(scales_c.shape[0])}"
        )
        return True

    def _run_model_train_stat_adapter_scale_selection() -> bool:
        model_train_stat_scale_selection_cfg = model_train_stat_adapter_cfg.get("scale_selection", {}) or {}
        if not (
            bool(model_train_stat_adapter_cfg.get("enable", False))
            and bool(model_train_stat_scale_selection_cfg.get("enable", False))
            and model_train_stat_adapter_pc is not None
        ):
            return False
        horizon_segments = int(model_train_stat_scale_selection_cfg.get("horizon_segments", 1))
        scales_c, scores_c, selection_count = select_train_stat_anchor_scales_from_loader(
            model=model,
            loader=dl_va,
            cluster_id_c=cluster_id_c,
            device=device,
            history_anchor_cfg=history_anchor_cfg,
            observed_history_tc=data_window_tc,
            input_len=L,
            eval_start=val_eval_start,
            stat_anchor_pc=model_train_stat_adapter_pc,
            train_stat_anchor_cfg=model_train_stat_adapter_cfg,
            metric=str(model_train_stat_scale_selection_cfg.get("metric", "mse")),
            max_scale=float(model_train_stat_scale_selection_cfg.get("max_scale", 0.3)),
            steps=int(model_train_stat_scale_selection_cfg.get("steps", 13)),
            horizon_segments=horizon_segments,
        )
        if int(scales_c.ndim) == 2:
            alpha_values = [[float(v) for v in row] for row in scales_c.tolist()]
            model_train_stat_adapter_cfg["alpha_by_channel_horizon"] = alpha_values
            model_train_stat_adapter_cfg["alpha_horizon_segments"] = int(horizon_segments)
            alpha_key = "alpha_by_channel_horizon"
        else:
            alpha_values = [float(v) for v in scales_c.tolist()]
            model_train_stat_adapter_cfg["alpha_by_channel"] = alpha_values
            alpha_key = "alpha_by_channel"
        model_train_stat_adapter_summary["scale_selection"] = {
            "enable": True,
            "source_split": "val",
            "metric": str(model_train_stat_scale_selection_cfg.get("metric", "mse")),
            "max_scale": float(model_train_stat_scale_selection_cfg.get("max_scale", 0.3)),
            "steps": int(model_train_stat_scale_selection_cfg.get("steps", 13)),
            "horizon_segments": int(horizon_segments),
            "num_windows": int(selection_count),
            alpha_key: alpha_values,
            "score": [[float(v) for v in row] for row in scores_c.tolist()]
            if int(scores_c.ndim) == 2
            else [float(v) for v in scores_c.tolist()],
            "mean_alpha": float(scales_c.mean().item()) if int(scales_c.numel()) > 0 else 0.0,
        }
        print(
            "Model train-stat adapter scale selection: "
            f"metric={model_train_stat_adapter_summary['scale_selection']['metric']}, "
            f"mean_alpha={model_train_stat_adapter_summary['scale_selection']['mean_alpha']:.4f}, "
            f"horizon_segments={int(horizon_segments)}, channels={int(scales_c.shape[0])}"
        )
        return True

    _run_model_train_stat_adapter_scale_selection()
    train_stat_scale_selection_done = _run_train_stat_anchor_scale_selection()

    train_residual_scale_selection_cfg = train_residual_anchor_cfg.get("scale_selection", {}) or {}
    if bool(train_residual_anchor_cfg.get("enable", False)):
        train_residual_anchor_period = int(train_residual_anchor_cfg.get("period", 96))
        train_residual_anchor_phc, train_residual_anchor_counts, residual_train_count = (
            build_train_residual_anchor_table_from_loader(
                model=model,
                loader=dl_tr,
                cluster_id_c=cluster_id_c,
                device=device,
                history_anchor_cfg=history_anchor_cfg,
                observed_history_tc=data_window_tc,
                input_len=L,
                eval_start=0,
                period=train_residual_anchor_period,
                model_train_stat_adapter_pc=model_train_stat_adapter_pc,
                model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
                train_stat_anchor_pc=train_stat_anchor_pc,
                train_stat_anchor_cfg=train_stat_anchor_cfg,
            )
        )
        train_residual_anchor_summary.update(
            {
                "period": int(train_residual_anchor_period),
                "source_split": "train",
                "train_windows": int(residual_train_count),
                "min_count": int(train_residual_anchor_counts.min().item()),
                "max_count": int(train_residual_anchor_counts.max().item()),
                "alpha": float(train_residual_anchor_cfg.get("alpha", 0.0) or 0.0),
                "blend_target": str(train_residual_anchor_cfg.get("blend_target", "prediction")),
            }
        )
        print(
            "Train residual anchor expert enabled: "
            f"period={train_residual_anchor_period}, "
            f"alpha={float(train_residual_anchor_cfg.get('alpha', 0.0) or 0.0):.3f}, "
            f"train_windows={int(residual_train_count)}"
        )
    if (
        bool(train_residual_anchor_cfg.get("enable", False))
        and bool(train_residual_scale_selection_cfg.get("enable", False))
        and train_residual_anchor_phc is not None
    ):
        horizon_segments = int(train_residual_scale_selection_cfg.get("horizon_segments", 1))
        scales_c, scores_c, selection_count = select_train_residual_anchor_scales_from_loader(
            model=model,
            loader=dl_va,
            cluster_id_c=cluster_id_c,
            device=device,
            history_anchor_cfg=history_anchor_cfg,
            observed_history_tc=data_window_tc,
            input_len=L,
            eval_start=val_eval_start,
            residual_anchor_phc=train_residual_anchor_phc,
            train_residual_anchor_cfg=train_residual_anchor_cfg,
            metric=str(train_residual_scale_selection_cfg.get("metric", "mse")),
            max_scale=float(train_residual_scale_selection_cfg.get("max_scale", 0.5)),
            steps=int(train_residual_scale_selection_cfg.get("steps", 21)),
            horizon_segments=horizon_segments,
            model_train_stat_adapter_pc=model_train_stat_adapter_pc,
            model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
            train_stat_anchor_pc=train_stat_anchor_pc,
            train_stat_anchor_cfg=train_stat_anchor_cfg,
        )
        if int(scales_c.ndim) == 2:
            alpha_values = [[float(v) for v in row] for row in scales_c.tolist()]
            train_residual_anchor_cfg["alpha_by_channel_horizon"] = alpha_values
            train_residual_anchor_cfg["alpha_horizon_segments"] = int(horizon_segments)
            alpha_key = "alpha_by_channel_horizon"
        else:
            alpha_values = [float(v) for v in scales_c.tolist()]
            train_residual_anchor_cfg["alpha_by_channel"] = alpha_values
            alpha_key = "alpha_by_channel"
        train_residual_anchor_summary["scale_selection"] = {
            "enable": True,
            "source_split": "val",
            "metric": str(train_residual_scale_selection_cfg.get("metric", "mse")),
            "max_scale": float(train_residual_scale_selection_cfg.get("max_scale", 0.5)),
            "steps": int(train_residual_scale_selection_cfg.get("steps", 21)),
            "horizon_segments": int(horizon_segments),
            "num_windows": int(selection_count),
            alpha_key: alpha_values,
            "score_by_channel": scores_c.detach().cpu().tolist(),
            "mean_alpha": float(scales_c.mean().item()),
        }
        print(
            "Train residual anchor scale selection: "
            f"metric={train_residual_anchor_summary['scale_selection']['metric']}, "
            f"mean_alpha={train_residual_anchor_summary['scale_selection']['mean_alpha']:.4f}, "
            f"windows={int(selection_count)}"
        )

    train_stat_scale_selection_cfg = train_stat_anchor_cfg.get("scale_selection", {}) or {}
    if (
        not train_stat_scale_selection_done
        and bool(train_stat_anchor_cfg.get("enable", False))
        and bool(train_stat_scale_selection_cfg.get("enable", False))
        and train_stat_anchor_pc is not None
    ):
        horizon_segments = int(train_stat_scale_selection_cfg.get("horizon_segments", 1))
        scales_c, scores_c, selection_count = select_train_stat_anchor_scales_from_loader(
            model=model,
            loader=dl_va,
            cluster_id_c=cluster_id_c,
            device=device,
            history_anchor_cfg=history_anchor_cfg,
            observed_history_tc=data_window_tc,
            input_len=L,
            eval_start=val_eval_start,
            stat_anchor_pc=train_stat_anchor_pc,
            train_stat_anchor_cfg=train_stat_anchor_cfg,
            metric=str(train_stat_scale_selection_cfg.get("metric", "mse")),
            max_scale=float(train_stat_scale_selection_cfg.get("max_scale", 0.3)),
            steps=int(train_stat_scale_selection_cfg.get("steps", 13)),
            horizon_segments=horizon_segments,
            model_train_stat_adapter_pc=model_train_stat_adapter_pc,
            model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
        )
        if int(scales_c.ndim) == 2:
            alpha_values = [[float(v) for v in row] for row in scales_c.tolist()]
            train_stat_anchor_cfg["alpha_by_channel_horizon"] = alpha_values
            train_stat_anchor_cfg["alpha_horizon_segments"] = int(horizon_segments)
            alpha_key = "alpha_by_channel_horizon"
        else:
            alpha_values = [float(v) for v in scales_c.tolist()]
            train_stat_anchor_cfg["alpha_by_channel"] = alpha_values
            alpha_key = "alpha_by_channel"
        train_stat_anchor_summary["scale_selection"] = {
            "enable": True,
            "source_split": "val",
            "metric": str(train_stat_scale_selection_cfg.get("metric", "mse")),
            "max_scale": float(train_stat_scale_selection_cfg.get("max_scale", 0.3)),
            "steps": int(train_stat_scale_selection_cfg.get("steps", 13)),
            "horizon_segments": int(horizon_segments),
            "num_windows": int(selection_count),
            alpha_key: alpha_values,
            "score": [[float(v) for v in row] for row in scores_c.tolist()]
            if int(scores_c.ndim) == 2
            else [float(v) for v in scores_c.tolist()],
            "mean_alpha": float(scales_c.mean().item()) if int(scales_c.numel()) > 0 else 0.0,
        }
        print(
            "Train-stat anchor scale selection: "
            f"metric={train_stat_anchor_summary['scale_selection']['metric']}, "
            f"mean_alpha={train_stat_anchor_summary['scale_selection']['mean_alpha']:.4f}, "
            f"horizon_segments={int(horizon_segments)}, channels={int(scales_c.shape[0])}"
        )

    if bool(calendar_residual_cfg.get("enable", False)):
        source_split = str(calendar_residual_cfg.get("source_split", "train")).lower()
        if source_split not in {"train", "training"}:
            raise ValueError("calendar_residual.source_split must be 'train' for strict input96 experiments.")
        calendar_fit_target = str(calendar_residual_cfg.get("fit_target", "base_path")).lower()
        if calendar_fit_target in {"base", "base_path", "backbone"}:
            calendar_fit_loader = DataLoader(
                dtr,
                batch_size=int(cfg["train"]["batch_size"]),
                shuffle=False,
                num_workers=0,
                pin_memory=pin_mem,
            )
            calendar_residual_coef_cf, calendar_fit_summary = fit_calendar_residual_correction(
                model=model,
                loader=calendar_fit_loader,
                cluster_id_c=cluster_id_c,
                device=device,
                calendar_feature_tf=calendar_feature_tf,
                input_len=L,
                eval_start=0,
                cfg=calendar_residual_cfg,
                history_anchor_cfg=history_anchor_cfg,
                observed_history_tc=data_window_tc,
                model_train_stat_adapter_pc=model_train_stat_adapter_pc,
                model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
            )
            calendar_residual_summary.update(calendar_fit_summary)
            calendar_residual_summary["feature_names"] = list(calendar_feature_names)
            calendar_residual_summary["train_only"] = True
            if calendar_residual_coef_cf is not None:
                print(
                    "Calendar residual fitted: "
                    f"target=base_path, features={len(calendar_feature_names)}, "
                    f"fit_windows={calendar_residual_summary.get('fit_windows')}, "
                    f"coef_mean_abs={float(calendar_residual_summary.get('coef_mean_abs', 0.0)):.6f}"
                )
        elif calendar_fit_target in {"final", "final_eval", "final_eval_path", "eval_path"}:
            calendar_residual_summary["fit_target"] = "final_eval_path"
            calendar_residual_summary["pending_final_eval_path_fit"] = True
        else:
            raise ValueError(
                "calendar_residual.fit_target must be base_path or final_eval_path "
                f"(got {calendar_fit_target!r})."
            )

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
    pred_residual_selector_model = None
    pred_residual_selector_summary = None
    pred_residual_selection_summary = None
    moe_gate_penalty_hit_summary = None
    penalty_explainability_summary = None
    calibration_summary = {
        "enable": bool(calibration_enable),
        "method": str(calibration_method),
        "shrink": float(calibration_shrink),
        "max_abs": float(calibration_max_abs),
        "base_mean_abs": None,
        "hybrid_mean_abs": None,
    }
    mae_eval_weight = _scale_mae_objective_weight(
        mae_objective_weight_final if mae_objective_enable else 0.0,
        mae_objective_multiplier_k,
    )
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
        val_loss_best_k, val_mse_best_k, val_mae_best_k, val_mse_c_base, val_mae_c_base, _, _, _ = eval_loop_with_history(
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
            "per_channel_mse": [float(v) for v in val_mse_c_base.detach().cpu().tolist()],
            "per_channel_mae": [float(v) for v in val_mae_c_base.detach().cpu().tolist()],
        }
        residual_selection_policy = str(pred_residual_cfg.get("selection_policy", "none")).lower()
        if residual_selection_policy in {"false", "off", "disable", "disabled"}:
            residual_selection_policy = "none"
        if residual_selection_policy not in {
            "none",
            "val_mse_channel",
            "val_mse_scale",
            "val_mse_scale_holdout",
            "val_mse_gate",
            "val_mse_gate_guarded",
            "val_mae_gate_guarded",
        }:
            raise ValueError(
                "Unsupported moe.pred_side_residual.selection_policy="
                f"'{residual_selection_policy}'. Expected none, val_mse_channel, val_mse_scale, "
                "val_mse_scale_holdout, val_mse_gate, val_mse_gate_guarded, or val_mae_gate_guarded."
            )
        if pred_residual is not None and residual_selection_policy in {
            "val_mse_channel",
            "val_mse_scale",
            "val_mse_scale_holdout",
            "val_mse_gate",
            "val_mse_gate_guarded",
            "val_mae_gate_guarded",
        }:
            zero_residual_scale_c = torch.zeros(C, device=device, dtype=torch.float32)
            residual_scale_mean_value = 0.0
            selection_max_residual_channels = int(pred_residual_cfg.get("selection_max_residual_channels", 0))
            selection_eval_segments = int(pred_residual_cfg.get("selection_eval_segments", 1))
            selection_min_positive_segments = int(pred_residual_cfg.get("selection_min_positive_segments", 0))
            selection_max_segment_rel_degradation = float(
                pred_residual_cfg.get("selection_max_segment_rel_degradation", 0.0)
            )
            selection_max_segment_abs_degradation = float(
                pred_residual_cfg.get("selection_max_segment_abs_degradation", 0.0)
            )
            selection_segment_improvement_mse_sc = None
            selection_segment_keep_c = None
            selection_eval_split = "val"
            selection_select_windows = len(dva)
            selection_eval_windows = len(dva)
            selection_eval_base_mse_c = None
            selection_eval_base_mae_c = None
            val_scaled_full_mse_c = None
            val_scaled_full_mae_c = None
            (
                val_loss_pred_base_k,
                val_mse_pred_base_k,
                val_mae_pred_base_k,
                val_mse_c_pred_base,
                val_mae_c_pred_base,
                _,
                _,
                _,
            ) = eval_loop_with_history(
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
            if residual_selection_policy in {"val_mse_scale", "val_mse_scale_holdout"}:
                scale_min = float(pred_residual_cfg.get("selection_scale_min", 0.0))
                scale_max = float(pred_residual_cfg.get("selection_scale_max", 1.0))
                scale_steps = int(pred_residual_cfg.get("selection_scale_steps", 21))
                if scale_steps < 2:
                    raise ValueError("moe.pred_side_residual.selection_scale_steps must be >= 2")
                scale_select_loader = val_loader_summary
                scale_eval_loader = val_loader_summary
                scale_eval_start = val_eval_start
                scale_eval_base_mse_c = val_mse_c_pred_base
                scale_eval_base_mae_c = val_mae_c_pred_base
                if residual_selection_policy == "val_mse_scale_holdout":
                    holdout_fraction = float(pred_residual_cfg.get("selection_holdout_fraction", 0.4))
                    holdout_min_windows = int(pred_residual_cfg.get("selection_holdout_min_windows", 256))
                    select_n, holdout_n = _validation_holdout_split_counts(
                        len(dva),
                        holdout_fraction=holdout_fraction,
                        min_holdout=holdout_min_windows,
                    )
                    if holdout_n > 0:
                        scale_select_loader = DataLoader(
                            Subset(dva, range(0, select_n)),
                            batch_size=int(cfg["train"]["batch_size"]),
                            shuffle=False,
                            num_workers=0,
                            pin_memory=pin_mem,
                        )
                        scale_eval_loader = DataLoader(
                            Subset(dva, range(select_n, select_n + holdout_n)),
                            batch_size=int(cfg["train"]["batch_size"]),
                            shuffle=False,
                            num_workers=0,
                            pin_memory=pin_mem,
                        )
                        scale_eval_start = val_eval_start + select_n
                        selection_eval_split = "val_holdout"
                        selection_select_windows = select_n
                        selection_eval_windows = holdout_n
                        (
                            _,
                            _,
                            _,
                            scale_eval_base_mse_c,
                            scale_eval_base_mae_c,
                            _,
                            _,
                            _,
                        ) = eval_loop_with_history(
                            model, gate, lam_kp_best,
                            penalty_names, penalty_fns,
                            scale_eval_loader, cluster_id_c, K, moe_cfg, device,
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
                            eval_start=scale_eval_start,
                        )
                selection_eval_base_mse_c = scale_eval_base_mse_c
                selection_eval_base_mae_c = scale_eval_base_mae_c
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
                    ) = eval_loop_with_history(
                        model, gate, lam_kp_best,
                        penalty_names, penalty_fns,
                        scale_select_loader, cluster_id_c, K, moe_cfg, device,
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
                if residual_selection_policy == "val_mse_scale_holdout" and selection_eval_split == "val_holdout":
                    (
                        _,
                        _,
                        _,
                        holdout_scaled_mse_c,
                        holdout_scaled_mae_c,
                        _,
                        _,
                        _,
                    ) = eval_loop_with_history(
                        model, gate, lam_kp_best,
                        penalty_names, penalty_fns,
                        scale_eval_loader, cluster_id_c, K, moe_cfg, device,
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
                        pred_residual_scale_c=pred_residual_channel_scale_c,
                        residual_correction_ch=residual_correction_base_ch,
                        knn_hybrid=None,
                        eval_start=scale_eval_start,
                    )
                    min_abs = float(pred_residual_cfg.get("selection_min_abs_improvement", 0.0))
                    min_rel = float(pred_residual_cfg.get("selection_min_rel_improvement", 0.0))
                    required = torch.maximum(
                        torch.full_like(scale_eval_base_mse_c, min_abs),
                        min_rel * scale_eval_base_mse_c.abs().clamp_min(1.0e-12),
                    )
                    use_residual_device_c = (scale_eval_base_mse_c - holdout_scaled_mse_c) > required
                    segment_ranges = _contiguous_segment_ranges(selection_eval_windows, selection_eval_segments)
                    if len(segment_ranges) > 1:
                        segment_base_parts = []
                        segment_scaled_parts = []
                        for segment_start, segment_end in segment_ranges:
                            segment_loader = DataLoader(
                                Subset(dva, range(select_n + segment_start, select_n + segment_end)),
                                batch_size=int(cfg["train"]["batch_size"]),
                                shuffle=False,
                                num_workers=0,
                                pin_memory=pin_mem,
                            )
                            segment_eval_start = val_eval_start + select_n + segment_start
                            (
                                _,
                                _,
                                _,
                                segment_base_mse_c,
                                _,
                                _,
                                _,
                                _,
                            ) = eval_loop_with_history(
                                model, gate, lam_kp_best,
                                penalty_names, penalty_fns,
                                segment_loader, cluster_id_c, K, moe_cfg, device,
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
                                eval_start=segment_eval_start,
                            )
                            (
                                _,
                                _,
                                _,
                                segment_scaled_mse_c,
                                _,
                                _,
                                _,
                                _,
                            ) = eval_loop_with_history(
                                model, gate, lam_kp_best,
                                penalty_names, penalty_fns,
                                segment_loader, cluster_id_c, K, moe_cfg, device,
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
                                pred_residual_scale_c=pred_residual_channel_scale_c,
                                residual_correction_ch=residual_correction_base_ch,
                                knn_hybrid=None,
                                eval_start=segment_eval_start,
                            )
                            segment_base_parts.append(segment_base_mse_c.detach().cpu())
                            segment_scaled_parts.append(segment_scaled_mse_c.detach().cpu())
                        segment_base_sc = torch.stack(segment_base_parts, dim=0)
                        segment_scaled_sc = torch.stack(segment_scaled_parts, dim=0)
                        selection_segment_improvement_mse_sc = segment_base_sc - segment_scaled_sc
                        segment_required_sc = torch.maximum(
                            torch.full_like(segment_base_sc, min_abs),
                            min_rel * segment_base_sc.abs().clamp_min(1.0e-12),
                        )
                        segment_keep_c = torch.ones(C, dtype=torch.bool)
                        if selection_min_positive_segments > 0:
                            positive_counts_c = (selection_segment_improvement_mse_sc > segment_required_sc).sum(dim=0)
                            segment_keep_c &= positive_counts_c >= int(selection_min_positive_segments)
                        allowed_degradation_sc = torch.maximum(
                            torch.full_like(segment_base_sc, max(0.0, selection_max_segment_abs_degradation)),
                            max(0.0, selection_max_segment_rel_degradation)
                            * segment_base_sc.abs().clamp_min(1.0e-12),
                        )
                        segment_keep_c &= (selection_segment_improvement_mse_sc >= -allowed_degradation_sc).all(dim=0)
                        selection_segment_keep_c = segment_keep_c
                        use_residual_device_c &= segment_keep_c.to(device=use_residual_device_c.device)
                    pred_residual_channel_scale_c = torch.where(
                        use_residual_device_c.to(device=device),
                        pred_residual_channel_scale_c,
                        zero_residual_scale_c,
                    )
                    val_scaled_mse_c = torch.where(use_residual_device_c, holdout_scaled_mse_c, scale_eval_base_mse_c)
                    val_scaled_mae_c = torch.where(use_residual_device_c, holdout_scaled_mae_c, scale_eval_base_mae_c)
                    (
                        _,
                        _,
                        _,
                        val_scaled_full_mse_c,
                        val_scaled_full_mae_c,
                        _,
                        _,
                        _,
                    ) = eval_loop_with_history(
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
                        pred_residual_scale_c=pred_residual_channel_scale_c,
                        residual_correction_ch=residual_correction_base_ch,
                        knn_hybrid=None,
                        eval_start=val_eval_start,
                    )
                if selection_max_residual_channels > 0:
                    improvement_c = selection_eval_base_mse_c.detach().cpu() - val_scaled_mse_c.detach().cpu()
                    limit_c = _top_positive_improvement_mask(improvement_c, selection_max_residual_channels)
                    active_c = pred_residual_channel_scale_c.detach().cpu() > 1.0e-8
                    keep_c = active_c & limit_c
                    keep_scale_c = keep_c.to(device=pred_residual_channel_scale_c.device)
                    keep_mse_c = keep_c.to(device=val_scaled_mse_c.device)
                    keep_mae_c = keep_c.to(device=val_scaled_mae_c.device)
                    pred_residual_channel_scale_c = torch.where(
                        keep_scale_c,
                        pred_residual_channel_scale_c,
                        zero_residual_scale_c,
                    )
                    val_scaled_mse_c = torch.where(
                        keep_mse_c,
                        val_scaled_mse_c,
                        selection_eval_base_mse_c.to(
                            device=val_scaled_mse_c.device,
                            dtype=val_scaled_mse_c.dtype,
                        ),
                    )
                    val_scaled_mae_c = torch.where(
                        keep_mae_c,
                        val_scaled_mae_c,
                        selection_eval_base_mae_c.to(
                            device=val_scaled_mae_c.device,
                            dtype=val_scaled_mae_c.dtype,
                        ),
                    )
                    if val_scaled_full_mse_c is not None and val_scaled_full_mae_c is not None:
                        keep_full_mse_c = keep_c.to(device=val_scaled_full_mse_c.device)
                        keep_full_mae_c = keep_c.to(device=val_scaled_full_mae_c.device)
                        val_scaled_full_mse_c = torch.where(
                            keep_full_mse_c,
                            val_scaled_full_mse_c,
                            val_mse_c_pred_base.to(
                                device=val_scaled_full_mse_c.device,
                                dtype=val_scaled_full_mse_c.dtype,
                            ),
                        )
                        val_scaled_full_mae_c = torch.where(
                            keep_full_mae_c,
                            val_scaled_full_mae_c,
                            val_mae_c_pred_base.to(
                                device=val_scaled_full_mae_c.device,
                                dtype=val_scaled_full_mae_c.dtype,
                            ),
                        )
                use_residual_c = pred_residual_channel_scale_c.detach().cpu() > 1.0e-8
                scale_values = [float(v) for v in pred_residual_channel_scale_c.detach().cpu().tolist()]
                residual_scale_mean_value = float(pred_residual_channel_scale_c.mean().item())
            elif residual_selection_policy in {"val_mse_gate", "val_mse_gate_guarded", "val_mae_gate_guarded"}:
                gate_calib_cfg = pred_residual_cfg.get("gate_calibrator", {}) or {}
                gate_calib_source_split = str(gate_calib_cfg.get("source_split", "val")).lower()
                if gate_calib_source_split in {"train", "training"}:
                    gate_calib_loader = DataLoader(
                        dtr,
                        batch_size=int(cfg["train"]["batch_size"]),
                        shuffle=False,
                        num_workers=0,
                        pin_memory=pin_mem,
                    )
                    gate_calib_source_split = "train"
                    gate_calib_eval_start = 0
                elif gate_calib_source_split in {"val", "validation"}:
                    gate_calib_loader = val_loader_summary
                    gate_calib_source_split = "val"
                    gate_calib_eval_start = val_eval_start
                else:
                    raise ValueError(
                        "moe.pred_side_residual.gate_calibrator.source_split must be train or val "
                        f"(got {gate_calib_source_split!r})."
                    )
                pred_residual_gate_model, pred_residual_gate_summary = train_pred_residual_gate(
                    model=model,
                    gate=gate,
                    pred_residual=pred_residual,
                    loader=gate_calib_loader,
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
                    history_anchor_cfg=history_anchor_cfg,
                    observed_history_tc=data_window_tc,
                    input_len=L,
                    eval_start=gate_calib_eval_start,
                )
                if pred_residual_gate_summary is not None:
                    pred_residual_gate_summary["source_split"] = gate_calib_source_split
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
                    ) = eval_loop_with_history(
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
                    val_gate_tensors = _collect_pred_residual_gate_tensors(
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
                        history_anchor_cfg=history_anchor_cfg,
                        observed_history_tc=data_window_tc,
                        input_len=L,
                        eval_start=val_eval_start,
                    )
                    pred_residual_gate_summary["val_activation"] = _pred_residual_gate_activation_metrics_from_tensors(
                        tensors=val_gate_tensors,
                        residual_gate=pred_residual_gate_model,
                        device=device,
                        batch_size=int(gate_calib_cfg.get("batch_size", 256)),
                        activation_threshold=getattr(
                            pred_residual_gate_model,
                            "activation_threshold_c",
                            getattr(
                                pred_residual_gate_model,
                                "activation_threshold",
                                pred_residual_gate_summary.get("activation_threshold", 0.1),
                            ),
                        ),
                        label_min_improvement=float(gate_calib_cfg.get("activation_label_min_improvement", 0.0)),
                        activation_by_abs_scale=bool(
                            gate_calib_cfg.get(
                                "activation_by_abs_scale",
                                str(gate_calib_cfg.get("scale_mode", "sigmoid")).lower() == "signed_tanh",
                            )
                        ),
                        apply_activation_threshold=bool(
                            getattr(
                                pred_residual_gate_model,
                                "apply_activation_threshold",
                                gate_calib_cfg.get("apply_activation_threshold", False),
                            )
                        ),
                        channel_names=channel_names,
                        cluster_id_c=cluster_id_c,
                    )
                    val_scaled_mse_c = val_gate_mse_c
                    val_scaled_mae_c = val_gate_mae_c
                    hold_scales = pred_residual_gate_summary.get("holdout_mean_scale", [])
                    min_abs = float(pred_residual_cfg.get("selection_min_abs_improvement", 0.0))
                    min_rel = float(pred_residual_cfg.get("selection_min_rel_improvement", 0.0))
                    max_abs_mse_regression = float(pred_residual_cfg.get("selection_max_abs_mse_regression", 0.0))
                    max_rel_mse_regression = float(pred_residual_cfg.get("selection_max_rel_mse_regression", 0.0))
                    use_residual_c = _pred_residual_channel_keep_mask(
                        residual_selection_policy,
                        val_mse_c_pred_base,
                        val_scaled_mse_c,
                        val_mae_c_pred_base,
                        val_scaled_mae_c,
                        min_abs_improvement=min_abs,
                        min_rel_improvement=min_rel,
                        max_abs_mse_regression=max_abs_mse_regression,
                        max_rel_mse_regression=max_rel_mse_regression,
                    )
                    if residual_selection_policy in {"val_mse_gate_guarded", "val_mae_gate_guarded"}:
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
                        ) = eval_loop_with_history(
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
                "eval_split": selection_eval_split,
                "selection_windows": int(selection_select_windows),
                "eval_windows": int(selection_eval_windows),
                "max_residual_channels": int(selection_max_residual_channels),
                "eval_segments": int(selection_eval_segments),
                "min_positive_segments": int(selection_min_positive_segments),
                "max_segment_rel_degradation": float(selection_max_segment_rel_degradation),
                "max_segment_abs_degradation": float(selection_max_segment_abs_degradation),
                "min_abs_improvement": float(pred_residual_cfg.get("selection_min_abs_improvement", 0.0)),
                "min_rel_improvement": float(pred_residual_cfg.get("selection_min_rel_improvement", 0.0)),
                "max_abs_mse_regression": float(pred_residual_cfg.get("selection_max_abs_mse_regression", 0.0)),
                "max_rel_mse_regression": float(pred_residual_cfg.get("selection_max_rel_mse_regression", 0.0)),
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
            if val_scaled_full_mse_c is not None and val_scaled_full_mae_c is not None:
                pred_residual_selection_summary.update(
                    {
                        "val_scaled_full_avg_mse": float(val_scaled_full_mse_c.mean().item()),
                        "val_scaled_full_avg_mae": float(val_scaled_full_mae_c.mean().item()),
                        "val_scaled_full_mse_per_channel": [
                            float(v) for v in val_scaled_full_mse_c.detach().cpu().tolist()
                        ],
                        "val_scaled_full_mae_per_channel": [
                            float(v) for v in val_scaled_full_mae_c.detach().cpu().tolist()
                        ],
                    }
                )
            if selection_segment_improvement_mse_sc is not None:
                pred_residual_selection_summary.update(
                    {
                        "segment_improvement_mse_per_channel": [
                            [float(v) for v in row]
                            for row in selection_segment_improvement_mse_sc.detach().cpu().tolist()
                        ],
                        "segment_keep_channels": [
                            bool(v) for v in selection_segment_keep_c.detach().cpu().tolist()
                        ]
                        if selection_segment_keep_c is not None
                        else [],
                    }
                )
            if selection_eval_base_mse_c is not None and selection_eval_base_mae_c is not None:
                pred_residual_selection_summary.update(
                    {
                        "eval_pred_base_avg_mse": float(selection_eval_base_mse_c.mean().item()),
                        "eval_pred_base_avg_mae": float(selection_eval_base_mae_c.mean().item()),
                        "eval_pred_base_mse_per_channel": [
                            float(v) for v in selection_eval_base_mse_c.detach().cpu().tolist()
                        ],
                        "eval_pred_base_mae_per_channel": [
                            float(v) for v in selection_eval_base_mae_c.detach().cpu().tolist()
                        ],
                    }
                )
            print(
                "Prediction residual selection: "
                f"policy={residual_selection_policy}, "
                f"eval_split={selection_eval_split}, "
                f"residual_channels={pred_residual_selection_summary['num_residual_channels']}/{C}, "
                f"val_base_MSE={pred_residual_selection_summary['val_pred_base_avg_mse']:.6f}, "
                f"val_residual_MSE={pred_residual_selection_summary['val_residual_avg_mse']:.6f}, "
                f"val_scaled_MSE={pred_residual_selection_summary['val_scaled_avg_mse']:.6f}, "
                f"mean_scale={pred_residual_selection_summary['mean_scale']:.3f}"
            )
        selector_cfg = pred_residual_cfg.get("candidate_selector", {}) or {}
        if pred_residual is not None and moe_enable and P > 0 and bool(selector_cfg.get("enable", False)):
            selector_source_split = str(selector_cfg.get("source_split", "val")).lower()
            if selector_source_split in {"train", "training"}:
                selector_loader = DataLoader(
                    dtr,
                    batch_size=int(cfg["train"]["batch_size"]),
                    shuffle=False,
                    num_workers=0,
                    pin_memory=pin_mem,
                )
                selector_source_split = "train"
                selector_eval_start = 0
            elif selector_source_split in {"val", "validation"}:
                selector_loader = val_loader_summary
                selector_source_split = "val"
                selector_eval_start = val_eval_start
            else:
                raise ValueError(
                    "moe.pred_side_residual.candidate_selector.source_split must be train or val "
                    f"(got {selector_source_split!r})."
                )
            pred_residual_selector_model, pred_residual_selector_summary = train_pred_residual_candidate_selector(
                model=model,
                pred_residual=pred_residual,
                loader=selector_loader,
                cluster_id_c=cluster_id_c,
                K=K,
                moe_cfg=moe_cfg,
                device=device,
                penalty_names=penalty_names,
                channel_names=channel_names,
                cfg=selector_cfg,
                pred_residual_scale_c=pred_residual_channel_scale_c,
                history_anchor_cfg=history_anchor_cfg,
                observed_history_tc=data_window_tc,
                input_len=L,
                eval_start=selector_eval_start,
            )
            if pred_residual_selector_summary is not None:
                pred_residual_selector_summary["source_split"] = selector_source_split
            if pred_residual_selector_model is not None:
                (
                    val_selector_loss_k,
                    val_selector_mse_k,
                    val_selector_mae_k,
                    val_selector_mse_c,
                    val_selector_mae_c,
                    _,
                    _,
                    _,
                ) = eval_loop_with_history(
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
                    pred_residual_selector=pred_residual_selector_model,
                    pred_residual_scale_c=pred_residual_channel_scale_c,
                    residual_correction_ch=residual_correction_base_ch,
                    knn_hybrid=None,
                    eval_start=val_eval_start,
                )
                val_mse_c_base = val_selector_mse_c
                val_mae_c_base = val_selector_mae_c
                val_summary = {
                    "avg_loss": float(reduce_cluster_metric(val_selector_loss_k, cluster_weight_k).item()),
                    "avg_mse": float(reduce_cluster_metric(val_selector_mse_k, cluster_weight_k).item()),
                    "avg_mae": float(reduce_cluster_metric(val_selector_mae_k, cluster_weight_k).item()),
                    "per_cluster_loss": [float(v) for v in val_selector_loss_k.detach().cpu().tolist()],
                    "per_cluster_mse": [float(v) for v in val_selector_mse_k.detach().cpu().tolist()],
                    "per_cluster_mae": [float(v) for v in val_selector_mae_k.detach().cpu().tolist()],
                }
                if pred_residual_selection_summary is None:
                    pred_residual_selection_summary = {"policy": residual_selection_policy}
                pred_residual_selection_summary["candidate_selector"] = pred_residual_selector_summary
                pred_residual_selection_summary["val_selector_avg_mse"] = float(val_selector_mse_c.mean().item())
                pred_residual_selection_summary["val_selector_avg_mae"] = float(val_selector_mae_c.mean().item())
                print(
                    "Prediction residual candidate selector: "
                    f"source={selector_source_split}, "
                    f"val_MSE={val_summary['avg_mse']:.6f}, "
                    f"holdout_gain={((pred_residual_selector_summary or {}).get('holdout') or {}).get('selected_gain_pct_vs_base')}"
                )
        gate_penalty_hit_cfg = moe_cfg.get("gate_penalty_hit", {}) or {}
        gate_penalty_hit_enable = bool(gate_penalty_hit_cfg.get("enable", True))
        if gate_penalty_hit_enable and pred_residual is not None and moe_enable and P > 0:
            val_penalty_hit = evaluate_gate_penalty_hit_metrics(
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
                label_min_improvement=float(
                    pred_residual_cfg.get(
                        "gate_hit_label_min_improvement",
                        pred_residual_cfg.get("selection_min_abs_improvement", 0.0),
                    )
                ),
                history_anchor_cfg=history_anchor_cfg,
                observed_history_tc=data_window_tc,
                input_len=L,
                eval_start=val_eval_start,
            )
            moe_gate_penalty_hit_summary = {"val": val_penalty_hit, "test": None}
            if val_penalty_hit is not None:
                print(
                    "Gate penalty hit(val): "
                    f"top1={val_penalty_hit['top1_hit_rate_all']:.3f}, "
                    f"positive_top1={val_penalty_hit['top1_hit_rate_on_positive_oracle']:.3f}, "
                    f"selected_gain={val_penalty_hit['selected_top1_gain_pct_vs_base']:.3f}%"
                )
        if knn_hybrid_val is not None:
            knn_hybrid_val.reset_confidence_stats()
            val_loss_hybrid_k, val_mse_hybrid_k, val_mae_hybrid_k, val_mse_c_hybrid, val_mae_c_hybrid, _, _, _ = eval_loop_with_history(
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

    if (
        bool(calendar_residual_cfg.get("enable", False))
        and str(calendar_residual_cfg.get("fit_target", "base_path")).lower()
        in {"final", "final_eval", "final_eval_path", "eval_path"}
    ):
        calendar_fit_loader = DataLoader(
            dtr,
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=False,
            num_workers=0,
            pin_memory=pin_mem,
        )
        calendar_residual_coef_cf, calendar_fit_summary = fit_calendar_residual_correction_from_eval_path(
            model=model,
            gate=gate,
            lambda_kp=lam_kp_best,
            penalty_names=penalty_names,
            penalty_fns=penalty_fns,
            loader=calendar_fit_loader,
            cluster_id_c=cluster_id_c,
            K=K,
            moe_cfg=moe_cfg,
            device=device,
            calendar_feature_tf=calendar_feature_tf,
            input_len=L,
            cfg=calendar_residual_cfg,
            channel_count=C,
            select_ranks=select_ranks,
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
            pred_residual_selector=pred_residual_selector_model,
            pred_residual_scale_c=pred_residual_channel_scale_c,
            residual_correction_ch=residual_correction_base_ch,
            knn_hybrid=None,
            eval_start=0,
            history_anchor_cfg=history_anchor_cfg,
            observed_history_tc=data_window_tc,
            model_train_stat_adapter_pc=model_train_stat_adapter_pc,
            model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
            train_stat_anchor_pc=train_stat_anchor_pc,
            train_residual_anchor_phc=train_residual_anchor_phc,
        )
        calendar_residual_summary.update(calendar_fit_summary)
        calendar_residual_summary["feature_names"] = list(calendar_feature_names)
        calendar_residual_summary["train_only"] = True
        calendar_residual_summary.pop("pending_final_eval_path_fit", None)
        if calendar_residual_coef_cf is not None:
            print(
                "Calendar residual fitted: "
                f"target=final_eval_path, features={len(calendar_feature_names)}, "
                f"fit_windows={calendar_residual_summary.get('fit_windows')}, "
                f"coef_mean_abs={float(calendar_residual_summary.get('coef_mean_abs', 0.0)):.6f}"
            )

    lam_kp_test = lam_kp_best
    test_loss_k = test_mse_k = test_mae_k = None
    mse_c = mae_c = None
    plot_cache = {}
    best_sample = {}
    worst_sample = {}
    diagnostics_cfg = cfg.get("diagnostics", {}) or {}
    prediction_diag = bool(diagnostics_cfg.get("save_prediction_intermediates", False))
    prediction_diag_collector = None
    if prediction_diag:
        prediction_sample_count = int(diagnostics_cfg.get("prediction_sample_count", 32))
        prediction_sample_strategy = str(diagnostics_cfg.get("prediction_sample_strategy", "first"))
        prediction_sample_seed = int(diagnostics_cfg.get("prediction_sample_seed", 0))
        prediction_sample_indices = select_prediction_sample_indices(
            total=len(dte),
            sample_count=prediction_sample_count,
            strategy=prediction_sample_strategy,
            seed=prediction_sample_seed,
        )
        prediction_diag_collector = {
            "limit": len(prediction_sample_indices),
            "count": 0,
            "parts": {},
            "indices": torch.as_tensor(prediction_sample_indices, dtype=torch.long),
            "strategy": prediction_sample_strategy,
            "seed": prediction_sample_seed,
            "relative_indices": prediction_sample_indices,
        }
    if not skip_test:
        test_loss_k, test_mse_k, test_mae_k, mse_c, mae_c, plot_cache, best_sample, worst_sample = eval_loop_with_history(
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
            pred_residual_selector=pred_residual_selector_model,
            pred_residual_scale_c=pred_residual_channel_scale_c,
            residual_correction_ch=residual_correction_base_ch,
            knn_hybrid=None,
            eval_start=test_eval_start,
            diagnostic_collector=prediction_diag_collector,
        )
        if pred_residual_gate_model is not None and pred_residual_gate_summary is not None:
            test_gate_tensors = _collect_pred_residual_gate_tensors(
                model=model,
                gate=gate,
                pred_residual=pred_residual,
                loader=dl_te,
                cluster_id_c=cluster_id_c,
                K=K,
                moe_cfg=moe_cfg,
                device=device,
                penalty_names=penalty_names,
                penalty_fns=penalty_fns,
                penalty_scale=penalty_scale,
                select_ranks=select_ranks,
                gate_soft_weight=gate_soft_weight,
                history_anchor_cfg=history_anchor_cfg,
                observed_history_tc=data_window_tc,
                input_len=L,
                eval_start=test_eval_start,
            )
            gate_calib_cfg_for_report = pred_residual_cfg.get("gate_calibrator", {}) or {}
            pred_residual_gate_summary["test_activation"] = _pred_residual_gate_activation_metrics_from_tensors(
                tensors=test_gate_tensors,
                residual_gate=pred_residual_gate_model,
                device=device,
                batch_size=int(gate_calib_cfg_for_report.get("batch_size", 256)),
                activation_threshold=getattr(
                    pred_residual_gate_model,
                    "activation_threshold_c",
                    getattr(
                        pred_residual_gate_model,
                        "activation_threshold",
                        pred_residual_gate_summary.get("activation_threshold", 0.1),
                    ),
                ),
                label_min_improvement=float(gate_calib_cfg_for_report.get("activation_label_min_improvement", 0.0)),
                activation_by_abs_scale=bool(
                    gate_calib_cfg_for_report.get(
                        "activation_by_abs_scale",
                        str(gate_calib_cfg_for_report.get("scale_mode", "sigmoid")).lower() == "signed_tanh",
                    )
                ),
                apply_activation_threshold=bool(
                    getattr(
                        pred_residual_gate_model,
                        "apply_activation_threshold",
                        gate_calib_cfg_for_report.get("apply_activation_threshold", False),
                    )
                ),
                channel_scale_c=pred_residual_channel_scale_c,
                channel_names=channel_names,
                cluster_id_c=cluster_id_c,
            )
        gate_penalty_hit_cfg = moe_cfg.get("gate_penalty_hit", {}) or {}
        gate_penalty_hit_enable = bool(gate_penalty_hit_cfg.get("enable", True))
        if gate_penalty_hit_enable and pred_residual is not None and moe_enable and P > 0:
            test_penalty_hit = evaluate_gate_penalty_hit_metrics(
                model=model,
                gate=gate,
                pred_residual=pred_residual,
                loader=dl_te,
                cluster_id_c=cluster_id_c,
                K=K,
                moe_cfg=moe_cfg,
                device=device,
                penalty_names=penalty_names,
                penalty_fns=penalty_fns,
                penalty_scale=penalty_scale,
                select_ranks=select_ranks,
                gate_soft_weight=gate_soft_weight,
                label_min_improvement=float(
                    pred_residual_cfg.get(
                        "gate_hit_label_min_improvement",
                        pred_residual_cfg.get("selection_min_abs_improvement", 0.0),
                    )
                ),
                history_anchor_cfg=history_anchor_cfg,
                observed_history_tc=data_window_tc,
                input_len=L,
                eval_start=test_eval_start,
            )
            if moe_gate_penalty_hit_summary is None:
                moe_gate_penalty_hit_summary = {"val": None, "test": test_penalty_hit}
            else:
                moe_gate_penalty_hit_summary["test"] = test_penalty_hit
            if test_penalty_hit is not None:
                print(
                    "Gate penalty hit(test): "
                    f"top1={test_penalty_hit['top1_hit_rate_all']:.3f}, "
                    f"positive_top1={test_penalty_hit['top1_hit_rate_on_positive_oracle']:.3f}, "
                    f"selected_gain={test_penalty_hit['selected_top1_gain_pct_vs_base']:.3f}%"
                )
    explain_cfg = moe_cfg.get("explainability", {}) or {}
    explain_enable = bool(explain_cfg.get("enable", False))
    if explain_enable and pred_residual is not None and moe_enable and P > 0:
        max_batches = int(explain_cfg.get("max_batches", 0))
        requested_splits = [str(x).lower() for x in explain_cfg.get("splits", ["train", "val", "test"])]
        split_loaders: Dict[str, DataLoader] = {}
        if "train" in requested_splits and len(dtr) > 0:
            split_loaders["train"] = DataLoader(
                dtr,
                batch_size=int(cfg["train"]["batch_size"]),
                shuffle=False,
                num_workers=0,
                pin_memory=pin_mem,
            )
        if "val" in requested_splits and len(dva) > 0:
            split_loaders["val"] = DataLoader(
                dva,
                batch_size=int(cfg["train"]["batch_size"]),
                shuffle=False,
                num_workers=0,
                pin_memory=pin_mem,
            )
        if "test" in requested_splits and (not skip_test) and len(dte) > 0:
            split_loaders["test"] = dl_te

        prior_for_explain = cluster_penalty_prior_prob_kp if cluster_penalty_prior_prob_kp is not None else gate_prior_prob_kp
        allowed_for_explain = cluster_penalty_allowed_mask_kp
        split_payloads = {}
        for split_name, split_loader in split_loaders.items():
            payload = evaluate_penalty_explainability(
                model=model,
                gate=gate,
                pred_residual=pred_residual,
                loader=split_loader,
                cluster_id_c=cluster_id_c,
                K=K,
                moe_cfg=moe_cfg,
                device=device,
                penalty_names=penalty_names,
                penalty_fns=penalty_fns,
                penalty_scale=penalty_scale,
                select_ranks=select_ranks,
                gate_soft_weight=gate_soft_weight,
                split_name=split_name,
                penalty_portrait_kp=penalty_portrait_kp,
                prior_prob_kp=prior_for_explain,
                allowed_mask_kp=allowed_for_explain,
                max_batches=max_batches,
            )
            if payload is not None:
                split_payloads[split_name] = payload
                print(
                    f"Penalty explainability({split_name}): "
                    f"gain={payload['final_gain_pct_vs_base']:.3f}%, "
                    f"selected_events={payload['selected_penalty_events']}, "
                    f"oracle_positive_events={payload['oracle_positive_events']}"
                )
        penalty_explainability_summary = {
            "enable": True,
            "max_batches": int(max_batches),
            "splits": split_payloads,
            "train_only_prior": {
                "source": "train_split_penalty_portrait" if penalty_portrait_kp is not None else None,
                "penalty_names": list(penalty_names),
                "diagnostic_score": (
                    penalty_portrait_kp.detach().cpu().tolist()
                    if penalty_portrait_kp is not None
                    else None
                ),
                "prior_prob": (
                    prior_for_explain.detach().cpu().tolist()
                    if prior_for_explain is not None
                    else None
                ),
                "allowed_mask": (
                    allowed_for_explain.detach().cpu().tolist()
                    if allowed_for_explain is not None
                    else None
                ),
            },
        }
        penalty_explainability_summary["artifact_paths"] = save_penalty_explainability_artifacts(
            out_dir,
            penalty_explainability_summary,
        )
    test_loss_hybrid_k = None
    test_mse_hybrid_k = None
    test_mae_hybrid_k = None
    mse_c_hybrid = None
    mae_c_hybrid = None
    test_hybrid_confidence = None
    if knn_hybrid_test is not None and not skip_test:
        knn_hybrid_test.reset_confidence_stats()
        test_loss_hybrid_k, test_mse_hybrid_k, test_mae_hybrid_k, mse_c_hybrid, mae_c_hybrid, _, _, _ = eval_loop_with_history(
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
            cand_val_loss_k, cand_val_mse_k, cand_val_mae_k, cand_val_mse_c, cand_val_mae_c, _, _, _ = eval_loop_with_history(
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
            cand_test_loss_k, cand_test_mse_k, cand_test_mae_k, cand_mse_c, cand_mae_c, _, _, _ = eval_loop_with_history(
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
            val_loss_hybrid_k, val_mse_hybrid_k, val_mae_hybrid_k, val_mse_c_hybrid, val_mae_c_hybrid, _, _, _ = eval_loop_with_history(
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
                test_loss_hybrid_k, test_mse_hybrid_k, test_mae_hybrid_k, mse_c_hybrid, mae_c_hybrid, _, _, _ = eval_loop_with_history(
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
            val_loss_hybrid_k, val_mse_hybrid_k, val_mae_hybrid_k, val_mse_c_hybrid, val_mae_c_hybrid, _, _, _ = eval_loop_with_history(
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
                test_loss_hybrid_k, test_mse_hybrid_k, test_mae_hybrid_k, mse_c_hybrid, mae_c_hybrid, _, _, _ = eval_loop_with_history(
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

    # --- Post-hoc calibration shrink sweep (single model, no retraining) ---
    # Calibration is eval-only, so all shrink values share the same trained model
    # and the same unit (shrink=1) median residual. We estimate that unit once on
    # val, then linearly rescale per shrink and re-run val/test evaluation. This
    # mirrors the per-split metric conventions of the reported val/test blocks
    # (val: cluster-reduced; test: per-channel-mean MAE) so numbers are directly
    # comparable, at ~1/N the cost of N separate runs.
    if (
        calibration_enable
        and calibration_method == "median"
        and not skip_test
        and len(calibration_shrink_sweep) > 0
        and len(dva) > 0
    ):
        sweep_val_loader = DataLoader(dva, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)
        unit_corr_ch = estimate_residual_correction(
            model, sweep_val_loader, cluster_id_c, device,
            method=calibration_method, shrink=1.0, max_abs=0.0,
            knn_hybrid=None, eval_start=val_eval_start,
        )
        if unit_corr_ch is not None:
            common_sweep_kwargs = dict(
                select_ranks=select_ranks, collect_plot=False, channel_count=C,
                mse_weight=mse_weight, gate_entropy_weight=gate_entropy_weight,
                gate_balance_weight=gate_balance_weight, gate_soft_weight=gate_soft_weight,
                gate_entropy_target_frac=gate_entropy_target_frac, penalty_scale=penalty_scale,
                dynamic_lambda=dynamic_lambda, lambda_min_kp=lambda_min_kp,
                mae_objective_weight=mae_eval_weight, mae_objective_kind=mae_objective_kind,
                mae_objective_beta=mae_objective_beta, pred_residual=pred_residual,
                knn_hybrid=None,
            )
            sweep_rows = []
            for s in calibration_shrink_sweep:
                corr_s = unit_corr_ch * float(s)
                if calibration_max_abs and float(calibration_max_abs) > 0.0:
                    corr_s = corr_s.clamp(min=-float(calibration_max_abs), max=float(calibration_max_abs))
                # val mirrors the reported val eval (no pred_residual gate/selector/scale)
                _, v_mse_k, v_mae_k, _, _, _, _, _ = eval_loop_with_history(
                    model, gate, lam_kp_best, penalty_names, penalty_fns,
                    sweep_val_loader, cluster_id_c, K, moe_cfg, device,
                    residual_correction_ch=corr_s, eval_start=val_eval_start,
                    **common_sweep_kwargs,
                )
                # test mirrors the reported test eval (includes gate/selector/scale)
                _, t_mse_k, _, _, t_mae_c, _, _, _ = eval_loop_with_history(
                    model, gate, lam_kp_test, penalty_names, penalty_fns,
                    dl_te, cluster_id_c, K, moe_cfg, device,
                    pred_residual_gate=pred_residual_gate_model,
                    pred_residual_selector=pred_residual_selector_model,
                    pred_residual_scale_c=pred_residual_channel_scale_c,
                    residual_correction_ch=corr_s, eval_start=test_eval_start,
                    **common_sweep_kwargs,
                )
                sweep_rows.append({
                    "shrink": float(s),
                    "val_mse": float(reduce_cluster_metric(v_mse_k, cluster_weight_k).item()),
                    "val_mae": float(reduce_cluster_metric(v_mae_k, cluster_weight_k).item()),
                    "test_mse": float(reduce_cluster_metric(t_mse_k, cluster_weight_k).item()),
                    "test_mae": float(t_mae_c.mean().item()),
                })
            calibration_summary["shrink_sweep"] = sweep_rows
            calibration_summary["shrink_sweep_unit_mean_abs"] = float(unit_corr_ch.abs().mean().item())
            # Leakage-free pick: lowest val_mae, MSE-primary tiebreak on val_mse.
            best_sweep = min(sweep_rows, key=lambda r: (r["val_mae"], r["val_mse"]))
            calibration_summary["shrink_sweep_best_by_val_mae"] = float(best_sweep["shrink"])
            print("\nCalibration shrink sweep (single model, post-hoc):")
            print(f"  {'shrink':>7} {'val_mse':>10} {'val_mae':>10} {'test_mse':>10} {'test_mae':>10}")
            for r in sweep_rows:
                print(
                    f"  {r['shrink']:>7.3f} {r['val_mse']:>10.6f} {r['val_mae']:>10.6f} "
                    f"{r['test_mse']:>10.6f} {r['test_mae']:>10.6f}"
                )
            print(f"  -> best shrink by val_mae (val_mse tiebreak): {best_sweep['shrink']:.3f}")

            # --- Per-channel guarded shrink (richer, still MSE-capped) ---
            if per_channel_shrink_enable:
                pc = estimate_per_channel_guarded_shrink_correction(
                    model, sweep_val_loader, cluster_id_c, device,
                    grid=per_channel_shrink_grid,
                    max_rel_mse_regression=per_channel_shrink_max_rel_mse,
                    max_abs=calibration_max_abs,
                    eval_start=val_eval_start,
                )
                if pc is not None:
                    corr_pc_ch, chosen_shrink_c = pc
                    _, vpc_mse_k, vpc_mae_k, _, _, _, _, _ = eval_loop_with_history(
                        model, gate, lam_kp_best, penalty_names, penalty_fns,
                        sweep_val_loader, cluster_id_c, K, moe_cfg, device,
                        residual_correction_ch=corr_pc_ch, eval_start=val_eval_start,
                        **common_sweep_kwargs,
                    )
                    _, tpc_mse_k, _, _, tpc_mae_c, _, _, _ = eval_loop_with_history(
                        model, gate, lam_kp_test, penalty_names, penalty_fns,
                        dl_te, cluster_id_c, K, moe_cfg, device,
                        pred_residual_gate=pred_residual_gate_model,
                        pred_residual_selector=pred_residual_selector_model,
                        pred_residual_scale_c=pred_residual_channel_scale_c,
                        residual_correction_ch=corr_pc_ch, eval_start=test_eval_start,
                        **common_sweep_kwargs,
                    )
                    pc_summary = {
                        "val_mse": float(reduce_cluster_metric(vpc_mse_k, cluster_weight_k).item()),
                        "val_mae": float(reduce_cluster_metric(vpc_mae_k, cluster_weight_k).item()),
                        "test_mse": float(reduce_cluster_metric(tpc_mse_k, cluster_weight_k).item()),
                        "test_mae": float(tpc_mae_c.mean().item()),
                        "grid": [float(g) for g in per_channel_shrink_grid],
                        "max_rel_mse_regression": float(per_channel_shrink_max_rel_mse),
                        "chosen_shrink_per_channel": [float(s) for s in chosen_shrink_c.tolist()],
                        "mean_shrink": float(chosen_shrink_c.mean().item()),
                    }
                    calibration_summary["per_channel_shrink"] = pc_summary
                    print("Per-channel guarded shrink (cap on val MSE regression "
                          f"<= {per_channel_shrink_max_rel_mse:.3%}):")
                    print(f"  mean_shrink={pc_summary['mean_shrink']:.3f}  "
                          f"val_mse={pc_summary['val_mse']:.6f}  val_mae={pc_summary['val_mae']:.6f}  "
                          f"test_mse={pc_summary['test_mse']:.6f}  test_mae={pc_summary['test_mae']:.6f}")

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
        if prediction_diag_collector is not None:
            diag_parts = prediction_diag_collector.get("parts", {}) or {}
            arrays = {
                key: torch.cat(value, dim=0).numpy()
                for key, value in diag_parts.items()
                if isinstance(value, list) and len(value) > 0
            }
            arrays["cluster_id"] = cluster_id_c.detach().cpu().numpy()
            np.savez_compressed(os.path.join(out_dir, "prediction_intermediates.npz"), **arrays)
            diag_meta = {
                "sample_count": int(prediction_diag_collector.get("count", 0)),
                "channel_names": list(channel_names),
                "penalty_names": list(penalty_names),
                "sample_strategy": str(prediction_diag_collector.get("strategy", "first")),
                "sample_seed": int(prediction_diag_collector.get("seed", 0)),
                "relative_indices": [int(v) for v in prediction_diag_collector.get("relative_indices", [])],
                "note": "Rows are test windows; y_residual_raw is before residual gate scaling, y_final is after selected residual/calibration before optional KNN.",
            }
            with open(os.path.join(out_dir, "prediction_intermediates_meta.json"), "w", encoding="utf-8") as f:
                json.dump(diag_meta, f, ensure_ascii=False, indent=2)

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
    cluster_embedding_summary = _save_cluster_embedding_artifacts(model, out_dir)

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
            "per_cluster": mae_objective_per_cluster_summary,
        },
        "cluster_embedding": cluster_embedding_summary,
        "training_stability": {
            "shuffle_seed": None if shuffle_seed is None else int(shuffle_seed),
            "freeze_backbone": bool(freeze_backbone),
            "frozen_backbone_params": int(frozen_backbone_params),
            "loss_normalization": dict(loss_normalization_cfg),
            "lr_warmup_epochs": int(lr_warmup_epochs),
            "lr_warmup_start_factor": float(lr_warmup_start_factor),
            "swa": dict(swa_summary),
        },
        "eval": {
            "skip_test": bool(skip_test),
        },
        "calibration": calibration_summary,
        "calendar_residual": calendar_residual_summary,
        "moe_residual": moe_residual_summary,
        "moe_residual_selection": pred_residual_selection_summary,
        "moe_residual_gate_calibrator": pred_residual_gate_summary,
        "moe_residual_candidate_selector": pred_residual_selector_summary,
        "model_train_stat_adapter": model_train_stat_adapter_summary,
        "train_stat_anchor_expert": train_stat_anchor_summary,
        "train_residual_anchor_expert": train_residual_anchor_summary,
        "moe_gate_penalty_hit": moe_gate_penalty_hit_summary,
        "penalty_explainability": penalty_explainability_summary,
        "moe_router": {
            "mode": str(router_mode),
            "penalty_context_weight": float(router_penalty_context_weight),
            "penalty_context_score": str(router_penalty_context_score),
            "detach_penalty_context": bool(router_detach_penalty_context),
            "context_applied_inside_gate_logits": True,
            "allow_skip": bool(allow_skip),
            "skip_cost": float(skip_cost),
            "skip_supervision_weight": float(skip_supervision_weight),
            "skip_supervision_margin": float(skip_supervision_margin),
            "mse_utility_gate_supervision": {
                "enable": bool(mse_utility_gate_enable),
                "weight": float(mse_utility_gate_weight),
                "temperature": float(mse_utility_gate_temperature),
                "min_gain": float(mse_utility_gate_min_gain),
                "target_power": float(mse_utility_gate_target_power),
            },
        },
        "val": val_summary,
        "test": None if skip_test else {
            "avg_mae": avg_mae,
            "avg_mse": avg_mse,
            "per_cluster_loss": [float(v) for v in test_loss_k.detach().cpu().tolist()],
            "per_cluster_mse": [float(v) for v in test_mse_k.detach().cpu().tolist()],
            "per_cluster_mae": [float(v) for v in test_mae_k.detach().cpu().tolist()],
            "per_channel_mse": [float(v) for v in mse_c.detach().cpu().tolist()],
            "per_channel_mae": [float(v) for v in mae_c.detach().cpu().tolist()],
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
