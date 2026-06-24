from __future__ import annotations

import os
import json
import argparse
import time
import math
import sys
import builtins
from typing import Dict, Iterable, List, Tuple, Optional
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


def _normalize_confidence_gate_source_split(source_split: object) -> str:
    value = str(source_split or "train_holdout").strip().lower()
    if value in {"train", "training"}:
        return "train"
    if value in {"train_holdout", "train-holdout", "holdout"}:
        return "train_holdout"
    raise ValueError(
        "moe.pred_side_residual.confidence_gate.source_split must be train or train_holdout "
    )


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


def _make_cluster_optimizer_param_groups(
    *,
    base_params: List[nn.Parameter],
    gate_params: List[nn.Parameter],
    pred_residual_params: List[nn.Parameter],
    dynamic_lambda_params: List[nn.Parameter],
    learnable_lambda_params: List[nn.Parameter],
    base_weight_decay: float,
    moe_weight_decay: Optional[float],
    pred_residual_weight_decay: Optional[float],
) -> List[Dict[str, object]]:
    moe_params: List[nn.Parameter] = []
    moe_params.extend(gate_params)
    moe_params.extend(dynamic_lambda_params)
    moe_params.extend(learnable_lambda_params)

    groups: List[Dict[str, object]] = []
    if moe_weight_decay is None:
        common_params: List[nn.Parameter] = []
        common_params.extend(base_params)
        common_params.extend(moe_params)
        if len(common_params) > 0:
            groups.append({
                "params": common_params,
                "weight_decay": float(base_weight_decay),
            })
    else:
        if len(base_params) > 0:
            groups.append({
                "params": base_params,
                "weight_decay": float(base_weight_decay),
            })
    if moe_weight_decay is not None and len(moe_params) > 0:
        groups.append({
            "params": moe_params,
            "weight_decay": float(moe_weight_decay),
        })
    if len(pred_residual_params) > 0:
        residual_wd = pred_residual_weight_decay
        if residual_wd is None:
            residual_wd = moe_weight_decay if moe_weight_decay is not None else base_weight_decay
        groups.append({
            "params": pred_residual_params,
            "weight_decay": float(residual_wd),
        })
    return groups


def _load_finetune_pred_residual_state(
    *,
    pred_residual: Optional[ClusterwisePredResidualMoE],
    checkpoint: Dict[str, object],
    source_penalty_names: List[str],
    target_penalty_names: List[str],
    strict: bool = True,
) -> bool:
    if pred_residual is None or "pred_residual_state" not in checkpoint:
        return False
    if list(source_penalty_names) != list(target_penalty_names):
        raise ValueError(
            "Fine-tune pred_residual loading requires identical penalty_names: "
            f"source={list(source_penalty_names)}, target={list(target_penalty_names)}"
        )
    pred_residual.load_state_dict(checkpoint["pred_residual_state"], strict=bool(strict))
    return True


def _cluster_penalty_mask_to_channel_mask(
    allowed_mask_kp: Optional[torch.Tensor],
    cluster_id_c: torch.Tensor,
) -> Optional[torch.Tensor]:
    if allowed_mask_kp is None or int(allowed_mask_kp.numel()) == 0:
        return None
    if int(allowed_mask_kp.ndim) != 2:
        raise ValueError(f"cluster penalty mask must have shape [K,P], got {tuple(allowed_mask_kp.shape)}")
    allowed_kp = allowed_mask_kp.detach().to(dtype=torch.bool)
    cid_c = cluster_id_c.detach().to(device=allowed_kp.device, dtype=torch.long).view(-1)
    if int(cid_c.numel()) <= 0:
        return torch.empty(0, int(allowed_kp.shape[1]), device=allowed_kp.device, dtype=torch.bool)
    if int(cid_c.min().item()) < 0 or int(cid_c.max().item()) >= int(allowed_kp.shape[0]):
        raise ValueError(
            "cluster_id_c contains ids outside the cluster penalty mask range "
            f"[0,{int(allowed_kp.shape[0]) - 1}]"
        )
    return allowed_kp.index_select(0, cid_c).detach().clone()


def _gate_cluster_params(gate: ClusterwiseMoEGate, k: int) -> List[nn.Parameter]:
    params: List[nn.Parameter] = [gate.W1[k], gate.b1[k], gate.W2[k], gate.b2[k]]
    if gate.W_skip is not None and gate.b_skip is not None:
        params.extend([gate.W_skip[k], gate.b_skip[k]])
    return params


def _mask_gate_grads_after_epoch(
    *,
    gate: ClusterwiseMoEGate,
    epoch: int,
    freeze_after_epoch: int,
    stopped: torch.Tensor,
) -> bool:
    if int(freeze_after_epoch) <= 0 or int(epoch) <= int(freeze_after_epoch):
        return False
    freeze_mask = torch.ones(
        gate.K,
        dtype=torch.bool,
        device=stopped.device if isinstance(stopped, torch.Tensor) else None,
    )
    gate.mask_cluster_grads(freeze_mask)
    return True


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


BASE_FORECAST_GATE_FEATURE_NAMES = [
    "base_mean_shift_over_hist_std",
    "base_first_shift_over_hist_std",
    "base_last_shift_over_hist_std",
    "base_std_over_hist_std",
    "base_range_over_hist_std",
    "base_slope_over_hist_std",
    "base_mad1_over_hist_std",
    "base_mad2_over_hist_std",
    "base_last_minus_first_over_hist_std",
]


def _normalize_gate_feature_mode(mode: str) -> str:
    value = str(mode or "history").lower()
    if value in {"history", "input", "legacy"}:
        return "history"
    if value in {"history_base", "history+base", "input_base", "safe_augmented"}:
        return "history_base"
    raise ValueError("moe.gate_feature_mode must be 'history' or 'history_base'.")


def _gate_feature_names_for_mode(mode: str = "history") -> List[str]:
    mode = _normalize_gate_feature_mode(mode)
    if mode == "history":
        return list(GATE_FEATURE_NAMES)
    return list(GATE_FEATURE_NAMES) + list(BASE_FORECAST_GATE_FEATURE_NAMES)


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


def _weighted_cluster_sum_mean(
    sum_k: torch.Tensor,
    count: int,
    cluster_weight_k: torch.Tensor,
) -> float:
    count = max(int(count), 1)
    mean_k = sum_k.detach().to(device=cluster_weight_k.device, dtype=cluster_weight_k.dtype) / float(count)
    return float(reduce_cluster_metric(mean_k, cluster_weight_k).detach().cpu().item())


def _stage2_loss_epoch_summary(
    *,
    epoch: int,
    count: int,
    cluster_weight_k: torch.Tensor,
    total_loss_sum_k: torch.Tensor,
    forecast_loss_sum_k: torch.Tensor,
    penalty_loss_sum_k: torch.Tensor,
    pred_residual_aux_loss_sum_k: torch.Tensor,
    candidate_supervision_loss_sum_k: torch.Tensor,
    gate_utility_loss_sum_k: torch.Tensor,
    skip_noop_loss_sum_k: torch.Tensor,
    intervention_supervision_loss_sum_k: torch.Tensor,
    other_aux_loss_sum_k: torch.Tensor,
    train_mse_sum_k: torch.Tensor,
    train_mae_sum_k: torch.Tensor,
) -> Dict[str, float]:
    return {
        "epoch": int(epoch),
        "total_train_loss": _weighted_cluster_sum_mean(total_loss_sum_k, count, cluster_weight_k),
        "forecast_loss_only": _weighted_cluster_sum_mean(forecast_loss_sum_k, count, cluster_weight_k),
        "aux_penalty_loss": _weighted_cluster_sum_mean(penalty_loss_sum_k, count, cluster_weight_k),
        "pred_residual_aux_loss": _weighted_cluster_sum_mean(pred_residual_aux_loss_sum_k, count, cluster_weight_k),
        "candidate_supervision_loss": _weighted_cluster_sum_mean(candidate_supervision_loss_sum_k, count, cluster_weight_k),
        "gate_utility_loss": _weighted_cluster_sum_mean(gate_utility_loss_sum_k, count, cluster_weight_k),
        "skip_noop_loss": _weighted_cluster_sum_mean(skip_noop_loss_sum_k, count, cluster_weight_k),
        "intervention_supervision_loss": _weighted_cluster_sum_mean(intervention_supervision_loss_sum_k, count, cluster_weight_k),
        "other_aux_loss": _weighted_cluster_sum_mean(other_aux_loss_sum_k, count, cluster_weight_k),
        "train_mse": _weighted_cluster_sum_mean(train_mse_sum_k, count, cluster_weight_k),
        "train_mae": _weighted_cluster_sum_mean(train_mae_sum_k, count, cluster_weight_k),
    }


def _stage2_route_epoch_summary(
    *,
    penalty_names: List[str],
    cluster_weight_k: torch.Tensor,
    route_count_k: torch.Tensor,
    route_prob_sum_kp: torch.Tensor,
    route_actual_sum_kp: torch.Tensor,
    route_entropy_sum_k: torch.Tensor,
    skip_prob_sum_k: torch.Tensor,
    skip_active_sum_k: torch.Tensor,
) -> Dict[str, object]:
    if route_prob_sum_kp.numel() == 0:
        return {
            "route_entropy": 0.0,
            "prob_distribution": {},
            "actual_route_distribution": {},
            "skip_prob": 0.0,
            "skip_noop_rate": 0.0,
            "per_cluster": [],
        }
    denom_k = route_count_k.detach().to(device=cluster_weight_k.device, dtype=cluster_weight_k.dtype).clamp_min(1.0)
    prob_kp = route_prob_sum_kp.detach().to(device=cluster_weight_k.device, dtype=cluster_weight_k.dtype) / denom_k.view(-1, 1)
    actual_kp = route_actual_sum_kp.detach().to(device=cluster_weight_k.device, dtype=cluster_weight_k.dtype) / denom_k.view(-1, 1)
    entropy_k = route_entropy_sum_k.detach().to(device=cluster_weight_k.device, dtype=cluster_weight_k.dtype) / denom_k
    skip_prob_k = skip_prob_sum_k.detach().to(device=cluster_weight_k.device, dtype=cluster_weight_k.dtype) / denom_k
    skip_active_k = skip_active_sum_k.detach().to(device=cluster_weight_k.device, dtype=cluster_weight_k.dtype) / denom_k
    route_entropy = float(reduce_cluster_metric(entropy_k, cluster_weight_k).detach().cpu().item())
    skip_prob = float(reduce_cluster_metric(skip_prob_k, cluster_weight_k).detach().cpu().item())
    skip_rate = float(reduce_cluster_metric(skip_active_k, cluster_weight_k).detach().cpu().item())
    prob_p = reduce_cluster_metric(prob_kp.transpose(0, 1), cluster_weight_k).detach().cpu()
    actual_p = reduce_cluster_metric(actual_kp.transpose(0, 1), cluster_weight_k).detach().cpu()
    per_cluster = []
    for k in range(int(prob_kp.shape[0])):
        per_cluster.append(
            {
                "cluster_id": int(k),
                "route_entropy": float(entropy_k[k].detach().cpu().item()),
                "skip_prob": float(skip_prob_k[k].detach().cpu().item()),
                "skip_noop_rate": float(skip_active_k[k].detach().cpu().item()),
                "prob_distribution": {
                    penalty_names[p]: float(prob_kp[k, p].detach().cpu().item())
                    for p in range(min(len(penalty_names), int(prob_kp.shape[1])))
                },
                "actual_distribution": {
                    penalty_names[p]: float(actual_kp[k, p].detach().cpu().item())
                    for p in range(min(len(penalty_names), int(actual_kp.shape[1])))
                },
            }
        )
    return {
        "route_entropy": route_entropy,
        "prob_distribution": {
            penalty_names[p]: float(prob_p[p].item())
            for p in range(min(len(penalty_names), int(prob_p.numel())))
        },
        "actual_route_distribution": {
            penalty_names[p]: float(actual_p[p].item())
            for p in range(min(len(penalty_names), int(actual_p.numel())))
        },
        "skip_prob": skip_prob,
        "skip_noop_rate": skip_rate,
        "per_cluster": per_cluster,
    }


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if float(den) > 0.0 else 0.0


def _route_accuracy_summary_from_labels(
    *,
    labels: torch.Tensor,
    current_pred: torch.Tensor,
    label_names: List[str],
    features: Optional[torch.Tensor] = None,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, object]:
    label_count = int(len(label_names))
    if label_count <= 0:
        raise ValueError("label_names must be non-empty.")
    labels_cpu = labels.detach().cpu().to(dtype=torch.long).view(-1)
    current_cpu = current_pred.detach().cpu().to(dtype=torch.long).view(-1)
    if int(labels_cpu.numel()) != int(current_cpu.numel()):
        raise ValueError("labels and current_pred must have the same length.")
    valid = (labels_cpu >= 0) & (labels_cpu < label_count)
    if int(valid.sum().item()) <= 0:
        return {
            "samples": 0,
            "current_accuracy_all": 0.0,
            "majority_accuracy_all": 0.0,
            "lift_vs_majority": 0.0,
            "oracle_skip_rate": 0.0,
            "actual_skip_rate": 0.0,
            "skip_recall": 0.0,
            "skip_precision": 0.0,
            "penalty_accuracy_on_oracle_penalty": 0.0,
            "penalty_adoption_recall_on_oracle_penalty": 0.0,
            "penalty_adoption_precision": 0.0,
            "penalty_exact_precision": 0.0,
            "oracle_penalty_rate": 0.0,
            "penalty_adoption_rate": 0.0,
            "penalty_adoption_rate_gap_vs_oracle": 0.0,
            "missed_positive_adoption_rate": 0.0,
            "confusion_matrix_counts": [[0 for _ in range(label_count)] for _ in range(label_count)],
            "label_rates": {str(name): 0.0 for name in label_names},
            "current_prediction_rates": {str(name): 0.0 for name in label_names},
        }
    labels_v = labels_cpu[valid]
    current_v = current_cpu[valid].clamp(0, label_count - 1)
    samples = int(labels_v.numel())
    label_counts = torch.bincount(labels_v, minlength=label_count)[:label_count]
    current_counts = torch.bincount(current_v, minlength=label_count)[:label_count]
    confusion = torch.zeros(label_count, label_count, dtype=torch.long)
    for label, pred in zip(labels_v.tolist(), current_v.tolist()):
        confusion[int(label), int(pred)] += 1
    correct = current_v == labels_v
    accuracy = float(correct.to(dtype=torch.float32).mean().item())
    majority = _safe_ratio(float(label_counts.max().item()), float(samples))
    oracle_skip = labels_v == 0
    actual_skip = current_v == 0
    oracle_penalty = labels_v > 0
    actual_penalty = current_v > 0
    oracle_skip_count = int(oracle_skip.sum().item())
    actual_skip_count = int(actual_skip.sum().item())
    oracle_penalty_count = int(oracle_penalty.sum().item())
    actual_penalty_count = int(actual_penalty.sum().item())
    skip_true_positive = int((oracle_skip & actual_skip).sum().item())
    correct_penalty = int((oracle_penalty & (current_v == labels_v)).sum().item())
    adopted_penalty = int((oracle_penalty & actual_penalty).sum().item())
    wrong_penalty = int((oracle_penalty & actual_penalty & (current_v != labels_v)).sum().item())
    oracle_penalty_to_skip = int((oracle_penalty & actual_skip).sum().item())
    oracle_penalty_rate = _safe_ratio(oracle_penalty_count, samples)
    penalty_adoption_rate = _safe_ratio(actual_penalty_count, samples)
    summary: Dict[str, object] = {
        "samples": samples,
        "current_accuracy_all": accuracy,
        "majority_accuracy_all": majority,
        "lift_vs_majority": float(accuracy - majority),
        "oracle_skip_count": oracle_skip_count,
        "oracle_skip_rate": _safe_ratio(oracle_skip_count, samples),
        "actual_skip_count": actual_skip_count,
        "actual_skip_rate": _safe_ratio(actual_skip_count, samples),
        "skip_recall": _safe_ratio(skip_true_positive, oracle_skip_count),
        "skip_precision": _safe_ratio(skip_true_positive, actual_skip_count),
        "oracle_penalty_samples": oracle_penalty_count,
        "penalty_accuracy_on_oracle_penalty": _safe_ratio(correct_penalty, oracle_penalty_count),
        "penalty_adoption_recall_on_oracle_penalty": _safe_ratio(adopted_penalty, oracle_penalty_count),
        "penalty_adoption_precision": _safe_ratio(adopted_penalty, actual_penalty_count),
        "penalty_exact_precision": _safe_ratio(correct_penalty, actual_penalty_count),
        "oracle_penalty_rate": oracle_penalty_rate,
        "penalty_adoption_count": actual_penalty_count,
        "penalty_adoption_rate": penalty_adoption_rate,
        "penalty_adoption_rate_gap_vs_oracle": float(penalty_adoption_rate - oracle_penalty_rate),
        "missed_positive_adoption_rate": _safe_ratio(oracle_penalty_to_skip, oracle_penalty_count),
        "oracle_penalty_routed_to_skip_rate": _safe_ratio(oracle_penalty_to_skip, oracle_penalty_count),
        "oracle_penalty_routed_to_wrong_penalty_rate": _safe_ratio(wrong_penalty, oracle_penalty_count),
        "oracle_skip_routed_to_penalty_rate": _safe_ratio(int((oracle_skip & actual_penalty).sum().item()), oracle_skip_count),
        "label_counts": {str(name): int(label_counts[i].item()) for i, name in enumerate(label_names)},
        "label_rates": {
            str(name): _safe_ratio(int(label_counts[i].item()), samples)
            for i, name in enumerate(label_names)
        },
        "current_prediction_counts": {
            str(name): int(current_counts[i].item())
            for i, name in enumerate(label_names)
        },
        "current_prediction_rates": {
            str(name): _safe_ratio(int(current_counts[i].item()), samples)
            for i, name in enumerate(label_names)
        },
        "confusion_matrix_counts": confusion.tolist(),
        "confusion_matrix_rows": list(label_names),
        "confusion_matrix_cols": list(label_names),
    }
    if features is not None and feature_names is not None and "skip_prob" in feature_names:
        idx = int(feature_names.index("skip_prob"))
        feat_cpu = features.detach().cpu().to(dtype=torch.float32)
        if feat_cpu.dim() == 3 and int(feat_cpu.shape[0]) == int(valid.numel()) and int(feat_cpu.shape[-1]) > idx:
            skip_prob = feat_cpu[:, 0, idx].reshape(-1)[valid]
            finite = torch.isfinite(skip_prob)
            if int(finite.sum().item()) > 0:
                skip_prob = skip_prob[finite]
                summary["skip_probability"] = {
                    "available": True,
                    "mean": float(skip_prob.mean().item()),
                    "p50": float(skip_prob.quantile(0.50).item()),
                    "p95": float(skip_prob.quantile(0.95).item()),
                    "max": float(skip_prob.max().item()),
                    "gt_0_5_rate": float((skip_prob > 0.5).to(dtype=torch.float32).mean().item()),
                }
    return summary


def _route_distribution_entropy_from_rates(rates: Dict[str, float]) -> float:
    total = float(sum(max(0.0, float(v)) for v in rates.values()))
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for value in rates.values():
        p = max(0.0, float(value)) / total
        if p > 0.0:
            entropy -= p * math.log(p)
    return float(entropy)


def _route_audit_summary_from_tensors(
    *,
    tensors: Dict[str, object],
    explainability: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    label_names = list(tensors.get("label_names", []))
    summary = _route_accuracy_summary_from_labels(
        labels=tensors["labels"],  # type: ignore[arg-type,index]
        current_pred=tensors["current_pred"],  # type: ignore[arg-type,index]
        label_names=label_names,
        features=tensors.get("features"),  # type: ignore[arg-type]
        feature_names=list(tensors.get("feature_names", [])),
    )
    summary["oracle_route_distribution"] = dict(summary.get("label_rates", {}))
    summary["current_route_distribution"] = dict(summary.get("current_prediction_rates", {}))
    summary["oracle_route_entropy"] = _route_distribution_entropy_from_rates(
        summary.get("label_rates", {})  # type: ignore[arg-type]
    )
    summary["current_route_entropy"] = _route_distribution_entropy_from_rates(
        summary.get("current_prediction_rates", {})  # type: ignore[arg-type]
    )
    if explainability is not None:
        summary["candidate_oracle_gain_pct_vs_base"] = explainability.get("oracle_gain_pct_vs_base")
        summary["cluster_penalty_oracle_gain_pct_vs_base"] = explainability.get(
            "cluster_penalty_oracle_gain_pct_vs_base"
        )
        summary["cluster_route_oracle_gain_pct_vs_base"] = explainability.get(
            "cluster_route_oracle_gain_pct_vs_base"
        )
        summary["current_raw_routed_gain_pct_vs_base"] = explainability.get("final_gain_pct_vs_base")
        summary["base_mse"] = explainability.get("base_mse")
        summary["raw_routed_mse"] = explainability.get("final_mse")
    return summary


def _stage2_route_audit_thresholds(
    *,
    stage2_route_audit_cfg: Dict[str, object],
    route_ce_min_abs_improvement: float,
    route_ce_min_rel_improvement: float,
    route_ce_min_candidate_delta_rms: float,
    binary_adoption_weight: float,
    binary_adoption_min_abs_improvement: float,
    binary_adoption_min_rel_improvement: float,
    binary_adoption_min_candidate_delta_rms: float,
) -> Dict[str, object]:
    cfg = stage2_route_audit_cfg if isinstance(stage2_route_audit_cfg, dict) else {}
    explicit_keys = {
        "min_abs_improvement",
        "min_rel_improvement",
        "min_candidate_delta_rms",
        "candidate_action_floor",
    }
    if any(key in cfg for key in explicit_keys):
        default_abs = (
            binary_adoption_min_abs_improvement
            if float(binary_adoption_weight) > 0.0
            else route_ce_min_abs_improvement
        )
        default_rel = (
            binary_adoption_min_rel_improvement
            if float(binary_adoption_weight) > 0.0
            else route_ce_min_rel_improvement
        )
        default_delta = (
            binary_adoption_min_candidate_delta_rms
            if float(binary_adoption_weight) > 0.0
            else route_ce_min_candidate_delta_rms
        )
        return {
            "min_abs_improvement": float(cfg.get("min_abs_improvement", default_abs)),
            "min_rel_improvement": float(cfg.get("min_rel_improvement", default_rel)),
            "min_candidate_delta_rms": float(
                cfg.get(
                    "min_candidate_delta_rms",
                    cfg.get("candidate_action_floor", default_delta),
                )
            ),
            "source": "diagnostics.stage2_route_audit",
        }
    if float(binary_adoption_weight) > 0.0:
        return {
            "min_abs_improvement": float(binary_adoption_min_abs_improvement),
            "min_rel_improvement": float(binary_adoption_min_rel_improvement),
            "min_candidate_delta_rms": float(binary_adoption_min_candidate_delta_rms),
            "source": "moe.binary_adoption_supervision",
        }
    return {
        "min_abs_improvement": float(route_ce_min_abs_improvement),
        "min_rel_improvement": float(route_ce_min_rel_improvement),
        "min_candidate_delta_rms": float(route_ce_min_candidate_delta_rms),
        "source": "moe.route_ce_supervision",
    }


@torch.no_grad()
def _cluster_route_oracle_labels_and_gain_from_candidates(
    *,
    base_bch: torch.Tensor,
    cand_bcpH: Optional[torch.Tensor],
    y_bch: torch.Tensor,
    cluster_id_c: torch.Tensor,
    K: int,
    allowed_mask_kp: Optional[torch.Tensor] = None,
    min_abs_improvement: float = 0.0,
    min_rel_improvement: float = 0.0,
    min_candidate_delta_rms: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return route labels and raw best-penalty gain with 0=no-op and 1..P=penalty."""
    B = int(base_bch.shape[0])
    if cand_bcpH is None or int(cand_bcpH.numel()) == 0:
        labels = torch.zeros((B, int(K)), device=base_bch.device, dtype=torch.long)
        gain = torch.zeros((B, int(K)), device=base_bch.device, dtype=base_bch.dtype)
        return labels, gain
    P = int(cand_bcpH.shape[2])
    if P <= 0:
        labels = torch.zeros((B, int(K)), device=base_bch.device, dtype=torch.long)
        gain = torch.zeros((B, int(K)), device=base_bch.device, dtype=base_bch.dtype)
        return labels, gain
    base_err_bc = (base_bch - y_bch).pow(2).mean(dim=-1)
    cand_err_bcp = (cand_bcpH - y_bch.unsqueeze(2)).pow(2).mean(dim=-1)
    base_err_bk = scatter_mean_bc_to_bk(base_err_bc, cluster_id_c, int(K))
    cand_err_bkp = scatter_mean_bcf_to_bkf(cand_err_bcp, cluster_id_c, int(K))
    min_delta = max(0.0, float(min_candidate_delta_rms))
    cand_delta_bkp = None
    if min_delta > 0.0:
        cand_delta_bcp = (cand_bcpH - base_bch.unsqueeze(2)).pow(2).mean(dim=-1).sqrt()
        cand_delta_bkp = scatter_mean_bcf_to_bkf(cand_delta_bcp, cluster_id_c, int(K))
    if allowed_mask_kp is not None and int(allowed_mask_kp.numel()) > 0:
        allowed = allowed_mask_kp.to(device=cand_err_bkp.device, dtype=torch.bool)
        if tuple(allowed.shape) != (int(K), P):
            raise ValueError(
                "cluster route oracle allowed mask must have shape [K,P], "
                f"got {tuple(allowed.shape)} vs {(int(K), P)}."
            )
    else:
        allowed = torch.ones((int(K), P), device=cand_err_bkp.device, dtype=torch.bool)
    masked_err_bkp = cand_err_bkp.masked_fill(~allowed.unsqueeze(0), float("inf"))
    if cand_delta_bkp is not None:
        masked_err_bkp = masked_err_bkp.masked_fill(cand_delta_bkp < min_delta, float("inf"))
    best_err_bk, best_idx_bk = masked_err_bkp.min(dim=-1)
    has_candidate_bk = torch.isfinite(best_err_bk)
    gain_bk = torch.where(has_candidate_bk, base_err_bk - best_err_bk, torch.zeros_like(base_err_bk))
    required_bk = torch.maximum(
        torch.full_like(base_err_bk, max(0.0, float(min_abs_improvement))),
        max(0.0, float(min_rel_improvement)) * base_err_bk.abs().clamp_min(1.0e-12),
    )
    use_penalty_bk = has_candidate_bk & (gain_bk > required_bk)
    labels_bk = torch.where(
        use_penalty_bk,
        best_idx_bk.to(dtype=torch.long) + 1,
        torch.zeros_like(best_idx_bk, dtype=torch.long),
    )
    return labels_bk, gain_bk


@torch.no_grad()
def _cluster_route_oracle_labels_from_candidates(
    *,
    base_bch: torch.Tensor,
    cand_bcpH: Optional[torch.Tensor],
    y_bch: torch.Tensor,
    cluster_id_c: torch.Tensor,
    K: int,
    allowed_mask_kp: Optional[torch.Tensor] = None,
    min_abs_improvement: float = 0.0,
    min_rel_improvement: float = 0.0,
    min_candidate_delta_rms: float = 0.0,
) -> torch.Tensor:
    """Return cluster route labels with 0=no-op and 1..P=penalty index + 1."""
    labels_bk, _ = _cluster_route_oracle_labels_and_gain_from_candidates(
        base_bch=base_bch,
        cand_bcpH=cand_bcpH,
        y_bch=y_bch,
        cluster_id_c=cluster_id_c,
        K=K,
        allowed_mask_kp=allowed_mask_kp,
        min_abs_improvement=min_abs_improvement,
        min_rel_improvement=min_rel_improvement,
        min_candidate_delta_rms=min_candidate_delta_rms,
    )
    return labels_bk


@torch.no_grad()
def _route_ce_active_mask_from_gain(
    best_penalty_gain_bk: torch.Tensor,
    *,
    ignore_abs_gain_below: float = 0.0,
) -> torch.Tensor:
    if best_penalty_gain_bk.dim() != 2:
        raise ValueError("best_penalty_gain_bk must have shape [B,K].")
    if float(ignore_abs_gain_below) <= 0.0:
        return torch.ones_like(best_penalty_gain_bk, dtype=torch.bool)
    gain = best_penalty_gain_bk.detach()
    return torch.isfinite(gain) & (gain.abs() > float(ignore_abs_gain_below))


def _pred_residual_training_skip_arg(
    *,
    skip_bk: Optional[torch.Tensor],
    allow_skip: bool,
    ignore_skip_during_training: bool = False,
) -> Optional[torch.Tensor]:
    if not bool(allow_skip):
        return None
    if bool(ignore_skip_during_training):
        return None
    return skip_bk


def _route_probs_with_skip_class(
    *,
    probs_bkp: torch.Tensor,
    skip_prob_bk: Optional[torch.Tensor] = None,
    probs_include_skip_mass: bool = False,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    if probs_bkp.dim() != 3:
        raise ValueError(f"probs_bkp must have shape [B,K,P], got {tuple(probs_bkp.shape)}.")
    if skip_prob_bk is None:
        skip_mass_bk = torch.zeros(probs_bkp.shape[:2], device=probs_bkp.device, dtype=probs_bkp.dtype)
        penalty_mass_bkp = probs_bkp
    else:
        if tuple(skip_prob_bk.shape) != tuple(probs_bkp.shape[:2]):
            raise ValueError(
                "skip_prob_bk must have shape [B,K], "
                f"got {tuple(skip_prob_bk.shape)} vs {tuple(probs_bkp.shape[:2])}."
            )
        skip_mass_bk = skip_prob_bk
        if bool(probs_include_skip_mass):
            penalty_mass_bkp = probs_bkp
        else:
            penalty_mass_bkp = (1.0 - skip_prob_bk).unsqueeze(-1) * probs_bkp
    route_probs_bkq = torch.cat([skip_mass_bk.unsqueeze(-1), penalty_mass_bkp], dim=-1)
    route_probs_bkq = torch.nan_to_num(route_probs_bkq, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    return route_probs_bkq / route_probs_bkq.sum(dim=-1, keepdim=True).clamp_min(float(eps))


def _route_ce_loss_from_probs(
    *,
    probs_bkp: torch.Tensor,
    labels_bk: torch.Tensor,
    skip_prob_bk: Optional[torch.Tensor] = None,
    probs_include_skip_mass: bool = False,
    class_weight_q: Optional[torch.Tensor] = None,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    route_probs_bkq = _route_probs_with_skip_class(
        probs_bkp=probs_bkp,
        skip_prob_bk=skip_prob_bk,
        probs_include_skip_mass=probs_include_skip_mass,
        eps=eps,
    )
    labels = labels_bk.to(device=route_probs_bkq.device, dtype=torch.long)
    if tuple(labels.shape) != tuple(route_probs_bkq.shape[:2]):
        raise ValueError(
            "route CE labels must have shape [B,K], "
            f"got {tuple(labels.shape)} vs {tuple(route_probs_bkq.shape[:2])}."
        )
    if int(labels.min().item()) < 0 or int(labels.max().item()) >= int(route_probs_bkq.shape[-1]):
        raise ValueError(
            "route CE labels must be in [0,P], "
            f"got min={int(labels.min().item())}, max={int(labels.max().item())}, "
            f"P={int(route_probs_bkq.shape[-1]) - 1}."
        )
    target_prob_bk = route_probs_bkq.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    loss_bk = -target_prob_bk.clamp_min(float(eps)).log()
    if class_weight_q is not None and int(class_weight_q.numel()) > 0:
        weight_q = class_weight_q.to(device=loss_bk.device, dtype=loss_bk.dtype).view(-1)
        if int(weight_q.numel()) != int(route_probs_bkq.shape[-1]):
            raise ValueError(
                "route CE class_weight_q must have P+1 entries, "
                f"got {int(weight_q.numel())} vs {int(route_probs_bkq.shape[-1])}."
            )
        loss_bk = loss_bk * weight_q.index_select(0, labels.reshape(-1)).reshape_as(labels).to(dtype=loss_bk.dtype)
    return loss_bk


def _route_binary_adoption_loss_from_probs(
    *,
    probs_bkp: torch.Tensor,
    labels_bk: torch.Tensor,
    skip_prob_bk: Optional[torch.Tensor] = None,
    probs_include_skip_mass: bool = False,
    allowed_mask_kp: Optional[torch.Tensor] = None,
    active_mask_bk: Optional[torch.Tensor] = None,
    positive_weight: float = 1.0,
    negative_weight: float = 1.0,
    eps: float = 1.0e-8,
) -> Optional[torch.Tensor]:
    if probs_bkp.dim() != 3:
        raise ValueError(f"probs_bkp must have shape [B,K,P], got {tuple(probs_bkp.shape)}.")
    B, K, P = [int(v) for v in probs_bkp.shape]
    if P <= 0:
        return None
    labels = labels_bk.to(device=probs_bkp.device, dtype=torch.long)
    if tuple(labels.shape) != (B, K):
        raise ValueError(
            "binary adoption labels must have shape [B,K], "
            f"got {tuple(labels.shape)} vs {(B, K)}."
        )
    if int(labels.min().item()) < 0 or int(labels.max().item()) > P:
        raise ValueError(
            "binary adoption labels must be in [0,P], "
            f"got min={int(labels.min().item())}, max={int(labels.max().item())}, P={P}."
        )

    route_probs_bkq = _route_probs_with_skip_class(
        probs_bkp=probs_bkp,
        skip_prob_bk=skip_prob_bk,
        probs_include_skip_mass=probs_include_skip_mass,
        eps=eps,
    )
    penalty_mass_bkp = route_probs_bkq[..., 1:].clamp(float(eps), 1.0 - float(eps))
    target_bkp = torch.zeros_like(penalty_mass_bkp)
    positive_bk = labels > 0
    if bool(positive_bk.any().item()):
        target_bkp.scatter_(
            dim=-1,
            index=(labels.clamp_min(1) - 1).unsqueeze(-1),
            src=positive_bk.to(dtype=target_bkp.dtype).unsqueeze(-1),
        )

    weight_bkp = torch.ones_like(penalty_mass_bkp)
    if allowed_mask_kp is not None and int(allowed_mask_kp.numel()) > 0:
        allowed = allowed_mask_kp.to(device=probs_bkp.device, dtype=torch.bool)
        if tuple(allowed.shape) != (K, P):
            raise ValueError(
                "binary adoption allowed mask must have shape [K,P], "
                f"got {tuple(allowed.shape)} vs {(K, P)}."
            )
        empty = allowed.sum(dim=-1, keepdim=True) <= 0
        allowed = torch.where(empty, torch.ones_like(allowed), allowed)
        weight_bkp = allowed.unsqueeze(0).to(dtype=penalty_mass_bkp.dtype).expand(B, K, P)
        target_bkp = target_bkp * weight_bkp

    pos_w = max(0.0, float(positive_weight))
    neg_w = max(0.0, float(negative_weight))
    bce_bkp = -(
        (pos_w * target_bkp * penalty_mass_bkp.log())
        + (neg_w * (1.0 - target_bkp) * (1.0 - penalty_mass_bkp).clamp_min(float(eps)).log())
    )
    denom_bk = weight_bkp.sum(dim=-1).clamp_min(1.0)
    loss_bk = (bce_bkp * weight_bkp).sum(dim=-1) / denom_bk
    if active_mask_bk is not None:
        active = active_mask_bk.to(device=loss_bk.device, dtype=torch.bool)
        if tuple(active.shape) != (B, K):
            raise ValueError(
                "binary adoption active_mask_bk must have shape [B,K], "
                f"got {tuple(active.shape)} vs {(B, K)}."
            )
        loss_bk = loss_bk * active.to(dtype=loss_bk.dtype)
    return loss_bk


def _route_rate_alignment_loss_from_probs(
    *,
    probs_bkp: torch.Tensor,
    labels_bk: torch.Tensor,
    skip_prob_bk: Optional[torch.Tensor] = None,
    probs_include_skip_mass: bool = False,
    allowed_mask_kp: Optional[torch.Tensor] = None,
    active_mask_bk: Optional[torch.Tensor] = None,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Match batch/cluster predicted route rates to oracle route-label rates.

    This is intentionally a low-capacity aggregate loss. It does not introduce new
    labels beyond the existing 0=no-op, 1..P=penalty oracle labels; it only
    discourages the gate from drifting to an adoption rate that disagrees with
    the train batch's own label distribution.
    """
    if probs_bkp.dim() != 3:
        raise ValueError(f"probs_bkp must have shape [B,K,P], got {tuple(probs_bkp.shape)}.")
    B, K, P = [int(v) for v in probs_bkp.shape]
    labels = labels_bk.to(device=probs_bkp.device, dtype=torch.long)
    if tuple(labels.shape) != (B, K):
        raise ValueError(
            "route-rate labels must have shape [B,K], "
            f"got {tuple(labels.shape)} vs {(B, K)}."
        )
    if int(labels.min().item()) < 0 or int(labels.max().item()) > P:
        raise ValueError(
            "route-rate labels must be in [0,P], "
            f"got min={int(labels.min().item())}, max={int(labels.max().item())}, P={P}."
        )

    route_probs_bkq = _route_probs_with_skip_class(
        probs_bkp=probs_bkp,
        skip_prob_bk=skip_prob_bk,
        probs_include_skip_mass=probs_include_skip_mass,
        eps=eps,
    )
    Q = int(route_probs_bkq.shape[-1])
    target_bkq = torch.zeros_like(route_probs_bkq)
    target_bkq.scatter_(dim=-1, index=labels.unsqueeze(-1), src=torch.ones_like(labels, dtype=route_probs_bkq.dtype).unsqueeze(-1))

    active_bk = torch.ones((B, K), device=probs_bkp.device, dtype=route_probs_bkq.dtype)
    if active_mask_bk is not None:
        active = active_mask_bk.to(device=probs_bkp.device, dtype=torch.bool)
        if tuple(active.shape) != (B, K):
            raise ValueError(
                "route-rate active_mask_bk must have shape [B,K], "
                f"got {tuple(active.shape)} vs {(B, K)}."
            )
        active_bk = active.to(dtype=route_probs_bkq.dtype)

    class_mask_kq = torch.ones((K, Q), device=probs_bkp.device, dtype=route_probs_bkq.dtype)
    if allowed_mask_kp is not None and int(allowed_mask_kp.numel()) > 0:
        allowed = allowed_mask_kp.to(device=probs_bkp.device, dtype=torch.bool)
        if tuple(allowed.shape) != (K, P):
            raise ValueError(
                "route-rate allowed mask must have shape [K,P], "
                f"got {tuple(allowed.shape)} vs {(K, P)}."
            )
        empty = allowed.sum(dim=-1, keepdim=True) <= 0
        allowed = torch.where(empty, torch.ones_like(allowed), allowed)
        class_mask_kq = torch.cat(
            [
                torch.ones((K, 1), device=probs_bkp.device, dtype=torch.bool),
                allowed,
            ],
            dim=-1,
        ).to(dtype=route_probs_bkq.dtype)

    denom_k = active_bk.sum(dim=0).clamp_min(1.0)
    pred_rate_kq = (route_probs_bkq * active_bk.unsqueeze(-1)).sum(dim=0) / denom_k.unsqueeze(-1)
    target_rate_kq = (target_bkq * active_bk.unsqueeze(-1)).sum(dim=0) / denom_k.unsqueeze(-1)
    diff_kq = (pred_rate_kq - target_rate_kq).pow(2) * class_mask_kq
    loss_k = diff_kq.sum(dim=-1) / class_mask_kq.sum(dim=-1).clamp_min(1.0)
    has_active_k = (active_bk.sum(dim=0) > 0).to(dtype=loss_k.dtype)
    loss_k = loss_k * has_active_k
    return loss_k.unsqueeze(0).expand(B, K)


def _route_positive_recall_loss_from_probs(
    *,
    probs_bkp: torch.Tensor,
    labels_bk: torch.Tensor,
    skip_prob_bk: Optional[torch.Tensor] = None,
    probs_include_skip_mass: bool = False,
    active_mask_bk: Optional[torch.Tensor] = None,
    mode: str = "ce",
    target_probability: float = 1.0,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Positive-only route CE that trains recall of the oracle penalty class."""
    if probs_bkp.dim() != 3:
        raise ValueError(f"probs_bkp must have shape [B,K,P], got {tuple(probs_bkp.shape)}.")
    B, K, P = [int(v) for v in probs_bkp.shape]
    labels = labels_bk.to(device=probs_bkp.device, dtype=torch.long)
    if tuple(labels.shape) != (B, K):
        raise ValueError(
            "positive-recall labels must have shape [B,K], "
            f"got {tuple(labels.shape)} vs {(B, K)}."
        )
    if int(labels.min().item()) < 0 or int(labels.max().item()) > P:
        raise ValueError(
            "positive-recall labels must be in [0,P], "
            f"got min={int(labels.min().item())}, max={int(labels.max().item())}, P={P}."
        )
    route_probs_bkq = _route_probs_with_skip_class(
        probs_bkp=probs_bkp,
        skip_prob_bk=skip_prob_bk,
        probs_include_skip_mass=probs_include_skip_mass,
        eps=eps,
    )
    target_prob_bk = route_probs_bkq.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    positive_bk = (labels > 0).to(dtype=target_prob_bk.dtype)
    if active_mask_bk is not None:
        active = active_mask_bk.to(device=probs_bkp.device, dtype=torch.bool)
        if tuple(active.shape) != (B, K):
            raise ValueError(
                "positive-recall active_mask_bk must have shape [B,K], "
                f"got {tuple(active.shape)} vs {(B, K)}."
            )
        positive_bk = positive_bk * active.to(dtype=target_prob_bk.dtype)
    mode_l = str(mode or "ce").lower()
    if mode_l in {"margin", "hinge", "floor"}:
        target = min(1.0, max(0.0, float(target_probability)))
        return (target - target_prob_bk).clamp_min(0.0).pow(2) * positive_bk
    if mode_l not in {"ce", "cross_entropy", "nll"}:
        raise ValueError(f"Unsupported positive recall loss mode: {mode}")
    return -target_prob_bk.clamp_min(float(eps)).log() * positive_bk


def _route_precision_constrained_recall_loss_from_probs(
    *,
    probs_bkp: torch.Tensor,
    labels_bk: torch.Tensor,
    skip_prob_bk: Optional[torch.Tensor] = None,
    probs_include_skip_mass: bool = False,
    active_mask_bk: Optional[torch.Tensor] = None,
    recall_mode: str = "ce",
    recall_target_probability: float = 1.0,
    false_adopt_max_probability: float = 0.5,
    false_adopt_weight: float = 1.0,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Recall oracle-positive penalties while capping false adoption on no-op labels."""
    if probs_bkp.dim() != 3:
        raise ValueError(f"probs_bkp must have shape [B,K,P], got {tuple(probs_bkp.shape)}.")
    B, K, P = [int(v) for v in probs_bkp.shape]
    labels = labels_bk.to(device=probs_bkp.device, dtype=torch.long)
    if tuple(labels.shape) != (B, K):
        raise ValueError(
            "precision-recall labels must have shape [B,K], "
            f"got {tuple(labels.shape)} vs {(B, K)}."
        )
    if int(labels.min().item()) < 0 or int(labels.max().item()) > P:
        raise ValueError(
            "precision-recall labels must be in [0,P], "
            f"got min={int(labels.min().item())}, max={int(labels.max().item())}, P={P}."
        )
    recall_loss_bk = _route_positive_recall_loss_from_probs(
        probs_bkp=probs_bkp,
        skip_prob_bk=skip_prob_bk,
        labels_bk=labels,
        probs_include_skip_mass=probs_include_skip_mass,
        active_mask_bk=active_mask_bk,
        mode=recall_mode,
        target_probability=recall_target_probability,
        eps=eps,
    )
    route_probs_bkq = _route_probs_with_skip_class(
        probs_bkp=probs_bkp,
        skip_prob_bk=skip_prob_bk,
        probs_include_skip_mass=probs_include_skip_mass,
        eps=eps,
    )
    penalty_mass_bk = route_probs_bkq[..., 1:].sum(dim=-1)
    skip_label_bk = (labels == 0).to(dtype=penalty_mass_bk.dtype)
    max_prob = min(1.0, max(0.0, float(false_adopt_max_probability)))
    skip_guard_bk = (penalty_mass_bk - max_prob).clamp_min(0.0).pow(2) * skip_label_bk
    if active_mask_bk is not None:
        active = active_mask_bk.to(device=probs_bkp.device, dtype=torch.bool)
        if tuple(active.shape) != (B, K):
            raise ValueError(
                "precision-recall active_mask_bk must have shape [B,K], "
                f"got {tuple(active.shape)} vs {(B, K)}."
            )
        skip_guard_bk = skip_guard_bk * active.to(dtype=skip_guard_bk.dtype)
    return recall_loss_bk + max(0.0, float(false_adopt_weight)) * skip_guard_bk


@torch.no_grad()
def _route_ce_class_weight_from_labels(
    *,
    labels_bk: torch.Tensor,
    num_classes: int,
    mode: str = "none",
    max_weight: float = 0.0,
    active_mask_bk: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    mode_l = str(mode or "none").lower()
    if mode_l in {"", "none", "off", "false", "0"}:
        return None
    if mode_l not in {"balanced", "inverse_frequency", "inverse_freq"}:
        raise ValueError(f"Unsupported route CE class_weight mode {mode!r}.")
    labels = labels_bk.detach().to(dtype=torch.long).reshape(-1)
    valid = (labels >= 0) & (labels < int(num_classes))
    if active_mask_bk is not None:
        active = active_mask_bk.detach().to(device=labels_bk.device, dtype=torch.bool).reshape(-1)
        if int(active.numel()) != int(labels.numel()):
            raise ValueError("route CE active_mask_bk must have the same number of elements as labels_bk.")
        valid = valid & active.to(device=valid.device)
    if int(valid.sum().item()) <= 0:
        return None
    counts = torch.bincount(labels[valid], minlength=int(num_classes)).to(dtype=torch.float32)
    weights = torch.zeros(int(num_classes), device=labels_bk.device, dtype=torch.float32)
    present = counts > 0.0
    if not bool(present.any().item()):
        return None
    weights[present] = float(valid.sum().item()) / (
        float(int(present.sum().item())) * counts[present].clamp_min(1.0)
    )
    if float(max_weight) > 0.0:
        weights = weights.clamp_max(float(max_weight))
    return weights.to(device=labels_bk.device)


def _parameter_grad_l2_norm(params: Iterable[nn.Parameter]) -> float:
    total = 0.0
    for param in params:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        total += float(grad.pow(2).sum().item())
    return float(total ** 0.5)


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


def _extract_base_forecast_gate_features(
    x_bcl: torch.Tensor,
    y_base_bch: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    hist_std = x_bcl.std(dim=-1).clamp_min(eps)
    hist_last = x_bcl[..., -1]
    base_mean = y_base_bch.mean(dim=-1)
    base_first = y_base_bch[..., 0]
    base_last = y_base_bch[..., -1]
    base_std = y_base_bch.std(dim=-1) if y_base_bch.shape[-1] > 1 else torch.zeros_like(base_mean)
    base_range = y_base_bch.max(dim=-1).values - y_base_bch.min(dim=-1).values
    t = torch.linspace(-1.0, 1.0, steps=y_base_bch.shape[-1], device=y_base_bch.device, dtype=y_base_bch.dtype).view(1, 1, -1)
    base_center = y_base_bch - base_mean.unsqueeze(-1)
    base_slope = (base_center * t).mean(dim=-1) / t.pow(2).mean(dim=-1).clamp_min(eps)
    if y_base_bch.shape[-1] >= 2:
        d1 = y_base_bch[..., 1:] - y_base_bch[..., :-1]
        base_mad1 = d1.abs().mean(dim=-1)
    else:
        d1 = None
        base_mad1 = torch.zeros_like(base_mean)
    if y_base_bch.shape[-1] >= 3:
        d2 = y_base_bch[..., 2:] - 2 * y_base_bch[..., 1:-1] + y_base_bch[..., :-2]
        base_mad2 = d2.abs().mean(dim=-1)
    else:
        base_mad2 = torch.zeros_like(base_mean)
    return torch.stack(
        [
            (base_mean - hist_last) / hist_std,
            (base_first - hist_last) / hist_std,
            (base_last - hist_last) / hist_std,
            base_std / hist_std,
            base_range / hist_std,
            base_slope / hist_std,
            base_mad1 / hist_std,
            base_mad2 / hist_std,
            (base_last - base_first) / hist_std,
        ],
        dim=-1,
    )


def _build_gate_routing_features(
    x_bcl: torch.Tensor,
    y_base_bch: Optional[torch.Tensor],
    cluster_id_c: torch.Tensor,
    K: int,
    mode: str = "history",
) -> torch.Tensor:
    mode = _normalize_gate_feature_mode(mode)
    feat_bcf = extract_gate_features(x_bcl)
    if mode == "history_base":
        if y_base_bch is None:
            raise ValueError("history_base gate features require y_base_bch.")
        base_feat_bcf = _extract_base_forecast_gate_features(x_bcl, y_base_bch)
        feat_bcf = torch.cat([feat_bcf, base_feat_bcf], dim=-1)
    return scatter_mean_bcf_to_bkf(feat_bcf, cluster_id_c, K)


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
    allow_empty_clusters: bool = False,
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
    if not bool(allow_empty_clusters):
        empty = mask.sum(dim=-1, keepdim=True) <= 0.0
        if bool(empty.any().item()):
            mask = torch.where(empty, torch.ones_like(mask), mask)
    return mask


def normalize_cluster_penalty_prior_apply_stage(apply_stage: object) -> str:
    value = str(apply_stage or "train_and_eval").strip().lower().replace("-", "_")
    if value in {"train", "train_eval", "train_and_eval", "train+eval", "always"}:
        return "train_and_eval"
    if value in {"eval", "eval_only", "late", "late_eval", "final", "final_eval"}:
        return "eval_only"
    raise ValueError(
        "cluster_penalty_prior.apply_stage must be 'train_and_eval' or 'eval_only' "
        f"(got {value!r})."
    )


def split_cluster_penalty_prior_allowed_mask_by_stage(
    allowed_mask_kp: Optional[torch.Tensor],
    apply_stage: object,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], str]:
    stage = normalize_cluster_penalty_prior_apply_stage(apply_stage)
    if allowed_mask_kp is None or int(allowed_mask_kp.numel()) == 0:
        return None, None, stage
    if stage == "eval_only":
        return None, allowed_mask_kp, stage
    return allowed_mask_kp, None, stage


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
    if policy in {"val_mse_channel", "val_mse_scale", "val_mse_scale_holdout"}:
        required = torch.maximum(
            torch.full_like(base_mse_c, float(min_abs_improvement)),
            float(min_rel_improvement) * base_mse_c.abs().clamp_min(1.0e-12),
        )
        return (base_mse_c - cand_mse_c) > required
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


_MOE_OUTPUT_ANCHOR_KEYS = (
    "history_anchor_expert",
    "train_stat_anchor_expert",
    "train_residual_anchor_expert",
)


def _clone_anchor_cfg(value):
    if isinstance(value, dict):
        return {key: _clone_anchor_cfg(sub_value) for key, sub_value in value.items()}
    if isinstance(value, list):
        return [_clone_anchor_cfg(sub_value) for sub_value in value]
    return value


def _merge_anchor_cfg(default: dict, override: Optional[dict]) -> dict:
    out = _clone_anchor_cfg(default)
    if not isinstance(override, dict):
        return out
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_anchor_cfg(out[key], value)
        else:
            out[key] = _clone_anchor_cfg(value)
    return out


def _stat_anchor_default(
    *,
    period: int,
    metric: str = "mse",
    max_scale: float = 0.2,
    steps: int = 9,
    horizon_segments: Optional[int] = None,
) -> dict:
    scale_selection = {
        "enable": True,
        "metric": str(metric),
        "max_scale": float(max_scale),
        "steps": int(steps),
    }
    if horizon_segments is not None:
        scale_selection["horizon_segments"] = int(horizon_segments)
    return {
        "enable": True,
        "period": int(period),
        "alpha": 0.0,
        "mode": "phase_mean",
        "reference": "last",
        "blend_target": "prediction",
        "scale_selection": scale_selection,
    }


def _residual_anchor_default(
    *,
    period: int,
    metric: str = "mse",
    max_scale: float = 1.2,
    steps: int = 49,
    horizon_segments: Optional[int] = None,
) -> dict:
    scale_selection = {
        "enable": True,
        "metric": str(metric),
        "max_scale": float(max_scale),
        "steps": int(steps),
    }
    if horizon_segments is not None:
        scale_selection["horizon_segments"] = int(horizon_segments)
    return {
        "enable": True,
        "period": int(period),
        "alpha": 0.0,
        "blend_target": "prediction",
        "scale_selection": scale_selection,
    }


def _moe_output_anchor_default(
    *,
    stat: Optional[dict],
    residual: Optional[dict],
    history: Optional[dict] = None,
) -> dict:
    return {
        "history_anchor_expert": _clone_anchor_cfg(history or {"enable": False}),
        "train_stat_anchor_expert": _clone_anchor_cfg(stat or {"enable": False}),
        "train_residual_anchor_expert": _clone_anchor_cfg(residual or {"enable": False}),
    }


_MAIN_TABLE_MOE_OUTPUT_ANCHOR_DEFAULTS = {
    ("etth1", 96): _moe_output_anchor_default(
        stat=_stat_anchor_default(period=24),
        residual=_residual_anchor_default(period=24, horizon_segments=12),
    ),
    ("etth1", 192): _moe_output_anchor_default(
        stat=_stat_anchor_default(period=24, metric="mae", horizon_segments=12),
        residual=_residual_anchor_default(period=24, metric="mae", horizon_segments=12),
    ),
    ("etth1", 336): _moe_output_anchor_default(
        stat=_stat_anchor_default(period=96),
        residual=_residual_anchor_default(period=96, max_scale=0.8, steps=33, horizon_segments=4),
    ),
    ("etth1", 720): _moe_output_anchor_default(
        stat=_stat_anchor_default(period=96, max_scale=0.15, steps=7),
        residual=_residual_anchor_default(period=96, max_scale=0.6, steps=25, horizon_segments=7),
    ),
    ("etth2", 96): _moe_output_anchor_default(
        stat=_stat_anchor_default(period=96),
        residual=_residual_anchor_default(period=96, horizon_segments=7),
    ),
    ("etth2", 192): _moe_output_anchor_default(
        stat=_stat_anchor_default(period=96, metric="mae"),
        residual=_residual_anchor_default(period=96, metric="mae", horizon_segments=7),
    ),
    ("etth2", 336): _moe_output_anchor_default(
        stat=_stat_anchor_default(period=96, metric="mae"),
        residual=_residual_anchor_default(period=96, metric="mae", max_scale=2.6, steps=105, horizon_segments=7),
    ),
    ("etth2", 720): _moe_output_anchor_default(
        stat=_stat_anchor_default(period=96, max_scale=0.4, steps=17),
        residual=None,
    ),
    ("ettm1", 96): _moe_output_anchor_default(
        stat=_stat_anchor_default(period=96),
        residual=_residual_anchor_default(period=96, max_scale=1.6, steps=65, horizon_segments=7),
    ),
    ("ettm1", 192): _moe_output_anchor_default(
        stat=_stat_anchor_default(period=96),
        residual=_residual_anchor_default(period=96, max_scale=2.65, steps=107, horizon_segments=7),
    ),
    ("ettm1", 336): _moe_output_anchor_default(
        stat=_stat_anchor_default(period=96),
        residual=_residual_anchor_default(period=96, max_scale=2.4, steps=97, horizon_segments=7),
    ),
    ("ettm1", 720): _moe_output_anchor_default(
        history={
            "enable": True,
            "lags": [96, 192, 288],
            "alpha": 0.2,
            "blend_target": "prediction",
            "history_scope": "input_window",
        },
        stat=_stat_anchor_default(period=96),
        residual=_residual_anchor_default(period=96, horizon_segments=7),
    ),
    ("ettm2", 96): _moe_output_anchor_default(
        stat=_stat_anchor_default(period=96, metric="mae", max_scale=0.18, steps=8),
        residual=_residual_anchor_default(period=96, metric="mae", horizon_segments=7),
    ),
    ("ettm2", 192): _moe_output_anchor_default(
        stat=_stat_anchor_default(period=96),
        residual=_residual_anchor_default(period=96, horizon_segments=4),
    ),
    ("ettm2", 336): _moe_output_anchor_default(
        stat=_stat_anchor_default(period=96, horizon_segments=12),
        residual=_residual_anchor_default(period=96, horizon_segments=12),
    ),
    ("ettm2", 720): _moe_output_anchor_default(
        stat=_stat_anchor_default(period=96, metric="mae", max_scale=0.18, steps=8),
        residual=_residual_anchor_default(period=96, metric="mae", horizon_segments=7),
    ),
}


def _normalize_anchor_default_dataset_name(dataset_name: object) -> str:
    raw = str(dataset_name or "").strip()
    if not raw:
        return ""
    stem = os.path.splitext(os.path.basename(raw))[0]
    key = stem.lower()
    if "_h" in key:
        head, tail = key.rsplit("_h", 1)
        if tail.isdigit():
            return head
    return key


def default_moe_output_anchor_cfg(dataset_name: object, pred_len: int) -> dict:
    dataset_key = _normalize_anchor_default_dataset_name(dataset_name)
    horizon = int(pred_len)
    if dataset_key.startswith("pems"):
        return _moe_output_anchor_default(
            stat=_stat_anchor_default(period=288),
            residual=_residual_anchor_default(period=288, horizon_segments=4),
        )
    defaults = _MAIN_TABLE_MOE_OUTPUT_ANCHOR_DEFAULTS.get((dataset_key, horizon), {})
    return _clone_anchor_cfg(defaults)


def apply_default_moe_output_anchor_cfg(moe_cfg: Optional[dict], *, dataset_name: object, pred_len: int) -> dict:
    out = dict(moe_cfg or {})
    defaults = default_moe_output_anchor_cfg(dataset_name, pred_len)
    for key in _MOE_OUTPUT_ANCHOR_KEYS:
        if key not in out and key in defaults:
            out[key] = _clone_anchor_cfg(defaults[key])
        elif key in out and key in defaults and isinstance(out.get(key), dict):
            out[key] = _merge_anchor_cfg(defaults[key], out[key])
    return out


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


def apply_moe_output_anchor_experts(
    pred_bch: torch.Tensor,
    *,
    base_pred_bch: torch.Tensor,
    x_bcl: torch.Tensor,
    query_start_abs_b: torch.Tensor,
    input_len: int,
    moe_cfg: Optional[dict],
    moe_enable: bool,
    observed_history_tc: Optional[torch.Tensor] = None,
    train_stat_anchor_pc: Optional[torch.Tensor] = None,
    train_residual_anchor_phc: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if not bool(moe_enable):
        return pred_bch
    moe_cfg = moe_cfg or {}
    out = pred_bch
    history_cfg = moe_cfg.get("history_anchor_expert", {}) or {}
    stat_cfg = moe_cfg.get("train_stat_anchor_expert", {}) or {}
    residual_cfg = moe_cfg.get("train_residual_anchor_expert", {}) or {}
    if bool(history_cfg.get("enable", False)):
        out = apply_moe_history_anchor_expert(
            out,
            base_pred_bch=base_pred_bch,
            observed_history_tc=observed_history_tc,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len),
            cfg=history_cfg,
        )
    if bool(stat_cfg.get("enable", False)):
        out = apply_train_stat_anchor_expert(
            out,
            base_pred_bch=base_pred_bch,
            x_bcl=x_bcl,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len),
            stat_anchor_pc=train_stat_anchor_pc,
            cfg=stat_cfg,
        )
    if bool(residual_cfg.get("enable", False)) and train_residual_anchor_phc is not None:
        out = apply_train_residual_anchor_expert(
            out,
            base_pred_bch=base_pred_bch,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len),
            residual_anchor_phc=train_residual_anchor_phc,
            cfg=residual_cfg,
        )
    return out


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
    gate_feature_mode: str = "history",
    penalty_scale: torch.Tensor = None,
    dynamic_lambda: ClusterwiseDynamicLambda = None,
    lambda_min_kp: torch.Tensor = None,
    mae_objective_weight=0.0,
    mae_objective_kind: str = "l1",
    mae_objective_beta: float = 1.0,
    pred_residual: Optional[ClusterwisePredResidualMoE] = None,
    pred_residual_selector: Optional[nn.Module] = None,
    pred_residual_scale_c: Optional[torch.Tensor] = None,
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
    gate_feature_mode = _normalize_gate_feature_mode(gate_feature_mode)
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
        feat_bkf = scatter_mean_bcf_to_bkf(feat_bcf, cluster_id_c, K)  # [B,K,F] for dynamic lambda
        gate_feat_bkf = _build_gate_routing_features(x, yhat_base, cluster_id_c, K, mode=gate_feature_mode)
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
                gate_feat_bkf,
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
        output_anchors_applied = False
        if pred_residual is not None and moe_enable and P > 0:
            pred_out = pred_residual(
                x,
                yhat_base,
                cluster_id_c,
                mask_bkp,
                skip_bk=skip_bk if allow_skip else None,
                query_start_abs_b=query_start_abs_b,
            )
            yhat_residual_raw = pred_out["y_final"]
            yhat = yhat_residual_raw
            if pred_residual_selector is not None:
                pred_residual_selector.eval()
                selector_base_bch, cand_bcpH = _pred_residual_candidates_on_eval_path(
                    yhat_base,
                    pred_out,
                    pred_residual_scale_c=pred_residual_scale_c,
                    apply_output_anchors=True,
                    x_bcl=x,
                    query_start_abs_b=query_start_abs_b,
                    input_len=int(input_len or x.shape[-1]),
                    moe_cfg=moe_cfg,
                    moe_enable=moe_enable,
                    observed_history_tc=observed_history_tc,
                    train_stat_anchor_pc=train_stat_anchor_pc,
                    train_residual_anchor_phc=train_residual_anchor_phc,
                )
                if cand_bcpH is not None:
                    yhat, _ = pred_residual_selector.select_prediction(x, selector_base_bch, cand_bcpH)
                    yhat_residual_raw = yhat
                    output_anchors_applied = True
            elif pred_residual_scale_c is not None:
                scale = pred_residual_scale_c.to(device=yhat.device, dtype=yhat.dtype).view(1, -1, 1)
                residual_gate_scale = scale.expand(yhat.shape[0], -1, -1)
                yhat = yhat_base + scale * (yhat - yhat_base)
        else:
            yhat = yhat_base

        if not output_anchors_applied:
            yhat = apply_moe_output_anchor_experts(
                yhat,
                base_pred_bch=yhat_base,
                x_bcl=x,
                query_start_abs_b=query_start_abs_b,
                input_len=int(input_len or x.shape[-1]),
                moe_cfg=moe_cfg,
                moe_enable=moe_enable,
                observed_history_tc=observed_history_tc,
                train_stat_anchor_pc=train_stat_anchor_pc,
                train_residual_anchor_phc=train_residual_anchor_phc,
            )

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


@torch.no_grad()


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
    pred_residual_selector: Optional[nn.Module] = None,
    pred_residual_scale_c: Optional[torch.Tensor] = None,
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
        pred_residual_selector=pred_residual_selector,
        pred_residual_scale_c=pred_residual_scale_c,
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


CANDIDATE_DELTA_FEATURE_NAMES = [
    "hist_mean",
    "log_hist_std",
    "hist_last",
    "hist_range",
    "hist_slope",
    "delta_abs_mean",
    "delta_abs_max",
    "delta_std",
    "delta_bias",
    "base_std",
    "hybrid_std",
    "base_shift",
    "hybrid_shift",
]


HISTORY_PROXY_SELECTOR_FEATURE_NAMES = [
    "proxy_base_mse",
    "proxy_hybrid_mse",
    "proxy_mse_delta",
    "proxy_base_mae",
    "proxy_hybrid_mae",
    "proxy_mae_delta",
    "proxy_hybrid_bias",
]


SHAPE_PROXY_SELECTOR_FEATURE_NAMES = [
    "hist_diff_rms",
    "hist_d2_rms",
    "base_slope",
    "hybrid_slope",
    "base_hist_slope_gap",
    "hybrid_hist_slope_gap",
    "hybrid_base_slope_delta",
    "base_diff_rms",
    "hybrid_diff_rms",
    "hybrid_base_diff_rms_delta",
    "base_d2_rms",
    "hybrid_d2_rms",
    "hybrid_base_d2_rms_delta",
    "base_hist_corr",
    "hybrid_hist_corr",
    "hybrid_base_corr_delta",
    "base_hist_std_ratio",
    "hybrid_hist_std_ratio",
    "hybrid_base_std_ratio_delta",
]


def _candidate_selector_feature_names(feature_mode: str = "base") -> List[str]:
    mode = str(feature_mode or "base").lower()
    if mode in {"base", "default"}:
        return list(CANDIDATE_DELTA_FEATURE_NAMES)
    if mode in {"history_proxy", "proxy"}:
        return list(CANDIDATE_DELTA_FEATURE_NAMES) + list(HISTORY_PROXY_SELECTOR_FEATURE_NAMES)
    if mode in {"shape_proxy", "shape", "rich_shape"}:
        return list(CANDIDATE_DELTA_FEATURE_NAMES) + list(HISTORY_PROXY_SELECTOR_FEATURE_NAMES) + list(SHAPE_PROXY_SELECTOR_FEATURE_NAMES)
    raise ValueError("candidate_selector.feature_mode must be base, history_proxy, or shape_proxy.")


def _history_proxy_for_candidate_selector(x_bcl: torch.Tensor, pred_len: int) -> torch.Tensor:
    H = int(pred_len)
    if H <= 0:
        raise ValueError("candidate selector history proxy requires pred_len > 0.")
    L = int(x_bcl.shape[-1])
    if L >= H:
        return x_bcl[..., -H:]
    return x_bcl[..., -1:].expand(*x_bcl.shape[:-1], H)


def _sequence_slope_bch(values_bch: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    if values_bch.shape[-1] <= 1:
        return torch.zeros_like(values_bch[..., 0])
    t = torch.linspace(-1.0, 1.0, steps=values_bch.shape[-1], device=values_bch.device, dtype=values_bch.dtype)
    t = t.view(*([1] * (values_bch.dim() - 1)), -1)
    centered = values_bch - values_bch.mean(dim=-1, keepdim=True)
    return (centered * t).mean(dim=-1) / t.pow(2).mean().clamp_min(eps)


def _sequence_diff_rms_bch(values_bch: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    if values_bch.shape[-1] <= 1:
        return torch.zeros_like(values_bch[..., 0])
    return values_bch.diff(dim=-1).pow(2).mean(dim=-1).clamp_min(eps).sqrt()


def _sequence_d2_rms_bch(values_bch: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    if values_bch.shape[-1] <= 2:
        return torch.zeros_like(values_bch[..., 0])
    return values_bch.diff(dim=-1).diff(dim=-1).pow(2).mean(dim=-1).clamp_min(eps).sqrt()


def _sequence_corr_bch(a_bch: torch.Tensor, b_bch: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    if a_bch.shape[-1] <= 1:
        return torch.zeros_like(a_bch[..., 0])
    a_center = a_bch - a_bch.mean(dim=-1, keepdim=True)
    b_center = b_bch - b_bch.mean(dim=-1, keepdim=True)
    denom = a_center.pow(2).mean(dim=-1).sqrt() * b_center.pow(2).mean(dim=-1).sqrt()
    return (a_center * b_center).mean(dim=-1) / denom.clamp_min(eps)


def _candidate_selector_features(
    x_bcl: torch.Tensor,
    base_bch: torch.Tensor,
    hybrid_bch: torch.Tensor,
    feature_mode: str = "base",
) -> torch.Tensor:
    base_feat = _candidate_delta_features(x_bcl, base_bch, hybrid_bch)
    mode = str(feature_mode or "base").lower()
    if mode in {"base", "default"}:
        return base_feat
    if mode not in {"history_proxy", "proxy", "shape_proxy", "shape", "rich_shape"}:
        raise ValueError("candidate_selector.feature_mode must be base, history_proxy, or shape_proxy.")
    eps = 1.0e-6
    hist_std = x_bcl.std(dim=-1).clamp_min(eps)
    hist_proxy = _history_proxy_for_candidate_selector(x_bcl, int(base_bch.shape[-1])).to(
        device=base_bch.device,
        dtype=base_bch.dtype,
    )
    scale2 = hist_std.pow(2).clamp_min(eps).to(device=base_bch.device, dtype=base_bch.dtype)
    scale1 = hist_std.to(device=base_bch.device, dtype=base_bch.dtype)
    base_proxy_mse = (base_bch - hist_proxy).pow(2).mean(dim=-1) / scale2
    hybrid_proxy_mse = (hybrid_bch - hist_proxy).pow(2).mean(dim=-1) / scale2
    base_proxy_mae = (base_bch - hist_proxy).abs().mean(dim=-1) / scale1
    hybrid_proxy_mae = (hybrid_bch - hist_proxy).abs().mean(dim=-1) / scale1
    proxy_extra = torch.stack(
        [
            base_proxy_mse,
            hybrid_proxy_mse,
            hybrid_proxy_mse - base_proxy_mse,
            base_proxy_mae,
            hybrid_proxy_mae,
            hybrid_proxy_mae - base_proxy_mae,
            (hybrid_bch - hist_proxy).mean(dim=-1) / scale1,
        ],
        dim=-1,
    )
    if mode in {"history_proxy", "proxy"}:
        return torch.cat([base_feat, proxy_extra], dim=-1)

    hist_proxy = hist_proxy.to(device=base_bch.device, dtype=base_bch.dtype)
    hist_slope = _sequence_slope_bch(hist_proxy, eps=eps)
    base_slope = _sequence_slope_bch(base_bch, eps=eps)
    hybrid_slope = _sequence_slope_bch(hybrid_bch, eps=eps)
    hist_diff_rms = _sequence_diff_rms_bch(hist_proxy, eps=eps).to(dtype=base_bch.dtype) / scale1
    base_diff_rms = _sequence_diff_rms_bch(base_bch, eps=eps) / scale1
    hybrid_diff_rms = _sequence_diff_rms_bch(hybrid_bch, eps=eps) / scale1
    hist_d2_rms = _sequence_d2_rms_bch(hist_proxy, eps=eps).to(dtype=base_bch.dtype) / scale1
    base_d2_rms = _sequence_d2_rms_bch(base_bch, eps=eps) / scale1
    hybrid_d2_rms = _sequence_d2_rms_bch(hybrid_bch, eps=eps) / scale1
    base_corr = _sequence_corr_bch(base_bch, hist_proxy, eps=eps)
    hybrid_corr = _sequence_corr_bch(hybrid_bch, hist_proxy, eps=eps)
    hist_proxy_std = hist_proxy.std(dim=-1).clamp_min(eps)
    base_std_ratio = base_bch.std(dim=-1) / hist_proxy_std
    hybrid_std_ratio = hybrid_bch.std(dim=-1) / hist_proxy_std
    shape_extra = torch.stack(
        [
            hist_diff_rms,
            hist_d2_rms,
            base_slope,
            hybrid_slope,
            (base_slope - hist_slope) / scale1,
            (hybrid_slope - hist_slope) / scale1,
            (hybrid_slope - base_slope) / scale1,
            base_diff_rms,
            hybrid_diff_rms,
            hybrid_diff_rms - base_diff_rms,
            base_d2_rms,
            hybrid_d2_rms,
            hybrid_d2_rms - base_d2_rms,
            base_corr,
            hybrid_corr,
            hybrid_corr - base_corr,
            base_std_ratio,
            hybrid_std_ratio,
            hybrid_std_ratio - base_std_ratio,
        ],
        dim=-1,
    )
    return torch.cat([base_feat, proxy_extra, shape_extra], dim=-1)


def _candidate_delta_features(
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
    include_intervention: bool = True,
    include_selector: bool = True,
) -> Optional[torch.Tensor]:
    residuals = pred_out.get("residuals")
    alpha_cp = pred_out.get("alpha_cp")
    intervention_bcp = pred_out.get("intervention_bcp")
    selector_bcp = pred_out.get("selector_bcp")
    confidence_active_bcp = pred_out.get("confidence_active_bcp")
    if residuals is None or alpha_cp is None or intervention_bcp is None:
        return None
    if residuals.numel() == 0:
        return None
    if selector_bcp is None:
        selector_bcp = torch.ones_like(intervention_bcp)
    if confidence_active_bcp is None:
        confidence_active_bcp = torch.ones_like(intervention_bcp)
    if not bool(include_intervention):
        intervention_bcp = torch.ones_like(intervention_bcp)
        confidence_active_bcp = torch.ones_like(confidence_active_bcp)
    if not bool(include_selector):
        selector_bcp = torch.ones_like(selector_bcp)
    scale_bcp = intervention_bcp * selector_bcp * confidence_active_bcp * alpha_cp.unsqueeze(0)
    if pred_residual_scale_c is not None:
        channel_scale = pred_residual_scale_c.to(device=y_base_bch.device, dtype=y_base_bch.dtype).view(1, -1, 1)
        scale_bcp = scale_bcp * channel_scale
    return y_base_bch.unsqueeze(2) + scale_bcp.unsqueeze(-1) * residuals


def _pred_residual_candidates_on_eval_path(
    y_base_bch: torch.Tensor,
    pred_out: Dict[str, torch.Tensor],
    *,
    pred_residual_scale_c: Optional[torch.Tensor] = None,
    apply_output_anchors: bool = False,
    x_bcl: Optional[torch.Tensor] = None,
    query_start_abs_b: Optional[torch.Tensor] = None,
    input_len: int = 0,
    moe_cfg: Optional[dict] = None,
    moe_enable: bool = True,
    observed_history_tc: Optional[torch.Tensor] = None,
    train_stat_anchor_pc: Optional[torch.Tensor] = None,
    train_residual_anchor_phc: Optional[torch.Tensor] = None,
    include_intervention: bool = True,
    include_selector: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    cand_bcpH = _pred_residual_candidate_predictions(
        y_base_bch,
        pred_out,
        pred_residual_scale_c=pred_residual_scale_c,
        include_intervention=include_intervention,
        include_selector=include_selector,
    )
    if cand_bcpH is None or not apply_output_anchors:
        return y_base_bch, cand_bcpH
    if x_bcl is None or query_start_abs_b is None:
        raise ValueError("Output-anchor candidate evaluation requires x_bcl and query_start_abs_b.")
    y_base_final = apply_moe_output_anchor_experts(
        y_base_bch,
        base_pred_bch=y_base_bch,
        x_bcl=x_bcl,
        query_start_abs_b=query_start_abs_b,
        input_len=int(input_len),
        moe_cfg=moe_cfg,
        moe_enable=moe_enable,
        observed_history_tc=observed_history_tc,
        train_stat_anchor_pc=train_stat_anchor_pc,
        train_residual_anchor_phc=train_residual_anchor_phc,
    )
    cand_final_parts = [
        apply_moe_output_anchor_experts(
            cand_bcpH[:, :, p, :],
            base_pred_bch=y_base_bch,
            x_bcl=x_bcl,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len),
            moe_cfg=moe_cfg,
            moe_enable=moe_enable,
            observed_history_tc=observed_history_tc,
            train_stat_anchor_pc=train_stat_anchor_pc,
            train_residual_anchor_phc=train_residual_anchor_phc,
        )
        for p in range(int(cand_bcpH.shape[2]))
    ]
    return y_base_final, torch.stack(cand_final_parts, dim=2)


def _candidate_selector_targets(
    *,
    base_bch: torch.Tensor,
    cand_bcpH: torch.Tensor,
    y_bch: torch.Tensor,
    min_abs_improvement: float = 0.0,
    min_rel_improvement: float = 0.0,
    allowed_mask_cp: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    base_err_bc = (base_bch - y_bch).pow(2).mean(dim=-1)
    cand_err_bcp = (cand_bcpH - y_bch.unsqueeze(2)).pow(2).mean(dim=-1)
    allowed = _selector_allowed_mask_cp(
        allowed_mask_cp,
        C=int(cand_err_bcp.shape[1]),
        P=int(cand_err_bcp.shape[2]),
        device=cand_err_bcp.device,
        context="candidate selector target",
    )
    if allowed is not None:
        cand_err_bcp = cand_err_bcp.masked_fill(~allowed.unsqueeze(0), float("inf"))
    best_err_bc, best_p_bc = cand_err_bcp.min(dim=-1)
    gain_bc = base_err_bc - best_err_bc
    gain_bc = torch.where(torch.isfinite(gain_bc), gain_bc, torch.full_like(gain_bc, float("-inf")))
    required_bc = torch.maximum(
        torch.full_like(base_err_bc, max(0.0, float(min_abs_improvement))),
        max(0.0, float(min_rel_improvement)) * base_err_bc.abs().clamp_min(1.0e-12),
    )
    return torch.where(gain_bc > required_bc, best_p_bc.to(dtype=torch.long) + 1, torch.zeros_like(best_p_bc))


def _selector_allowed_mask_cp(
    allowed_mask_cp: Optional[torch.Tensor],
    *,
    C: int,
    P: int,
    device: torch.device,
    context: str,
) -> Optional[torch.Tensor]:
    if allowed_mask_cp is None or int(allowed_mask_cp.numel()) == 0:
        return None
    allowed = allowed_mask_cp.to(device=device, dtype=torch.bool)
    if tuple(allowed.shape) != (int(C), int(P)):
        raise ValueError(
            f"{context} allowed mask must have shape [C,P], "
            f"got {tuple(allowed.shape)} vs {(int(C), int(P))}."
        )
    return allowed


def _candidate_selector_adoption_decision(
    *,
    current_mse: float,
    current_mae: float,
    selector_mse: float,
    selector_mae: float,
    min_abs_improvement: float = 0.0,
    min_rel_improvement: float = 0.0,
    max_rel_mae_regression: float = 0.0,
) -> Dict[str, object]:
    current_mse_f = float(current_mse)
    current_mae_f = float(current_mae)
    selector_mse_f = float(selector_mse)
    selector_mae_f = float(selector_mae)
    required = max(
        max(0.0, float(min_abs_improvement)),
        max(0.0, float(min_rel_improvement)) * max(abs(current_mse_f), 1.0e-12),
    )
    mse_improvement = current_mse_f - selector_mse_f
    mae_regression = selector_mae_f - current_mae_f
    allowed_mae_regression = max(0.0, float(max_rel_mae_regression)) * max(abs(current_mae_f), 1.0e-12)
    finite = all(math.isfinite(v) for v in (current_mse_f, current_mae_f, selector_mse_f, selector_mae_f))
    adopt = bool(finite and mse_improvement > required and mae_regression <= allowed_mae_regression)
    return {
        "adopt": adopt,
        "current_mse": current_mse_f,
        "current_mae": current_mae_f,
        "selector_mse": selector_mse_f,
        "selector_mae": selector_mae_f,
        "mse_improvement": float(mse_improvement),
        "mse_improvement_pct": float(100.0 * mse_improvement / max(abs(current_mse_f), 1.0e-12)),
        "required_mse_improvement": float(required),
        "min_abs_improvement": float(max(0.0, float(min_abs_improvement))),
        "min_rel_improvement": float(max(0.0, float(min_rel_improvement))),
        "mae_regression": float(mae_regression),
        "mae_regression_pct": float(100.0 * mae_regression / max(abs(current_mae_f), 1.0e-12)),
        "allowed_mae_regression": float(allowed_mae_regression),
        "max_rel_mae_regression": float(max(0.0, float(max_rel_mae_regression))),
        "finite": bool(finite),
    }


def _candidate_selector_candidate_scale(
    *,
    pred_residual_scale_c: Optional[torch.Tensor],
    selector_cfg: Dict[str, object],
) -> Tuple[Optional[torch.Tensor], str]:
    use_channel_scale = bool(selector_cfg.get("use_channel_scale_for_candidates", True))
    if use_channel_scale:
        return pred_residual_scale_c, "channel_scale"
    return None, "unscaled"


def _cluster_utility_threshold_stats(
    *,
    gain_bcp: torch.Tensor,
    cluster_id_c: torch.Tensor,
    K: int,
    allowed_mask_kp: Optional[torch.Tensor],
    thresholds: List[float],
) -> Dict[str, torch.Tensor]:
    if gain_bcp.numel() == 0:
        return {
            "valid_count_kt": torch.zeros(int(K), len(thresholds), dtype=torch.float64),
            "total_count_k": torch.zeros(int(K), dtype=torch.float64),
            "best_gain_sum_k": torch.zeros(int(K), dtype=torch.float64),
            "best_gain_positive_count_k": torch.zeros(int(K), dtype=torch.float64),
        }
    gain_bkp = scatter_mean_bcf_to_bkf(gain_bcp, cluster_id_c, int(K)).to(dtype=torch.float64)
    B, _, P = gain_bkp.shape
    if allowed_mask_kp is not None and allowed_mask_kp.numel() > 0:
        allowed = allowed_mask_kp.to(device=gain_bkp.device, dtype=torch.bool)
        if tuple(allowed.shape) != (int(K), int(P)):
            raise ValueError(
                "utility threshold allowed mask must have shape [K,P], "
                f"got {tuple(allowed.shape)} vs {(int(K), int(P))}."
            )
    else:
        allowed = torch.ones((int(K), int(P)), device=gain_bkp.device, dtype=torch.bool)
    allowed_bkp = allowed.unsqueeze(0).expand_as(gain_bkp)
    masked_gain_bkp = gain_bkp.masked_fill(~allowed_bkp, -float("inf"))
    best_gain_bk = masked_gain_bkp.max(dim=-1).values
    best_gain_bk = torch.nan_to_num(best_gain_bk, nan=0.0, posinf=0.0, neginf=0.0)
    valid_counts = []
    allowed_float = allowed_bkp.to(dtype=gain_bkp.dtype)
    for threshold in thresholds:
        utility_bkp = (gain_bkp - float(threshold)).clamp_min(0.0) * allowed_float
        valid_counts.append((utility_bkp.sum(dim=-1) > 0.0).sum(dim=0).to(dtype=torch.float64))
    if valid_counts:
        valid_count_kt = torch.stack(valid_counts, dim=-1)
    else:
        valid_count_kt = torch.zeros(int(K), 0, device=gain_bkp.device, dtype=torch.float64)
    return {
        "valid_count_kt": valid_count_kt.detach().cpu(),
        "total_count_k": torch.full((int(K),), int(B), dtype=torch.float64),
        "best_gain_sum_k": best_gain_bk.sum(dim=0).detach().cpu().to(dtype=torch.float64),
        "best_gain_positive_count_k": (best_gain_bk > 0.0).sum(dim=0).detach().cpu().to(dtype=torch.float64),
    }


def _mse_utility_gate_supervision_loss(
    *,
    probs_bkp: Optional[torch.Tensor],
    skip_prob_bk: Optional[torch.Tensor] = None,
    allowed_mask_kp: Optional[torch.Tensor] = None,
    y_base_bch: torch.Tensor,
    pred_out: Optional[Dict[str, torch.Tensor]],
    y_bch: torch.Tensor,
    cluster_id_c: torch.Tensor,
    K: int,
    y_base_eval_bch: Optional[torch.Tensor] = None,
    cand_eval_bcpH: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    min_gain: float = 0.0,
    target_power: float = 1.0,
    include_skip: bool = False,
    probs_include_skip_mass: bool = False,
    target_mode: str = "soft_utility",
    return_diagnostics: bool = False,
    eps: float = 1.0e-8,
) -> Optional[torch.Tensor]:
    if probs_bkp is None or pred_out is None or probs_bkp.numel() == 0:
        return (None, None) if return_diagnostics else None
    target_mode_l = str(target_mode or "soft_utility").lower()
    hard_target = target_mode_l in {"hard", "hard_oracle", "argmax", "one_hot"}
    if target_mode_l not in {"soft", "soft_utility", "utility", "hard", "hard_oracle", "argmax", "one_hot"}:
        raise ValueError(
            "mse utility gate supervision target_mode must be one of "
            "soft_utility or hard_oracle aliases, "
            f"got {target_mode!r}."
        )
    base_bch = y_base_eval_bch if y_base_eval_bch is not None else y_base_bch
    cand_bcpH = cand_eval_bcpH if cand_eval_bcpH is not None else _pred_residual_candidate_predictions(y_base_bch, pred_out)
    if cand_bcpH is None or cand_bcpH.numel() == 0:
        return (None, None) if return_diagnostics else None
    with torch.no_grad():
        base_err_bc = (base_bch - y_bch).pow(2).mean(dim=-1)
        cand_err_bcp = (cand_bcpH - y_bch.unsqueeze(2)).pow(2).mean(dim=-1)
        gain_bcp = base_err_bc.unsqueeze(-1) - cand_err_bcp
        gain_bkp = scatter_mean_bcf_to_bkf(gain_bcp, cluster_id_c, K)
        if allowed_mask_kp is not None and allowed_mask_kp.numel() > 0:
            allowed = allowed_mask_kp.to(device=probs_bkp.device, dtype=torch.bool)
            allowed_bkp = allowed.unsqueeze(0).expand_as(probs_bkp).to(dtype=gain_bkp.dtype)
        else:
            allowed_bkp = (probs_bkp.detach() > 0.0).to(dtype=gain_bkp.dtype)
        utility_bkp = (gain_bkp - float(min_gain)).clamp_min(0.0) * allowed_bkp
        if float(target_power) != 1.0:
            utility_bkp = utility_bkp.clamp_min(0.0).pow(float(target_power))
        valid_bk = utility_bkp.sum(dim=-1) > 0.0
        masked_gain_bkp = gain_bkp.masked_fill(allowed_bkp <= 0.0, float("-inf"))
        best_gain_bk, best_idx_bk = masked_gain_bkp.max(dim=-1)
        best_gain_bk = torch.where(torch.isfinite(best_gain_bk), best_gain_bk, torch.zeros_like(best_gain_bk))
        target_skip_bk = (~valid_bk).to(dtype=utility_bkp.dtype) if bool(include_skip) else torch.zeros_like(utility_bkp[..., 0])
        diagnostics = {
            "valid_bk": valid_bk.to(dtype=utility_bkp.dtype).detach(),
            "target_skip_bk": target_skip_bk.detach(),
            "best_gain_bk": best_gain_bk.detach(),
            "utility_sum_bk": utility_bkp.sum(dim=-1).detach(),
            "probs_include_skip_mass": torch.full_like(valid_bk, float(bool(probs_include_skip_mass)), dtype=utility_bkp.dtype).detach(),
        }
        if skip_prob_bk is not None:
            diagnostics["skip_prob_bk"] = skip_prob_bk.detach()
        if bool(include_skip):
            if skip_prob_bk is None:
                return (None, diagnostics) if return_diagnostics else None
            if tuple(skip_prob_bk.shape) != tuple(probs_bkp.shape[:2]):
                raise ValueError(
                    "skip-aware mse utility gate supervision requires skip_prob_bk shape [B,K], "
                    f"got {tuple(skip_prob_bk.shape)} vs {tuple(probs_bkp.shape[:2])}."
                )
            target_bkp = torch.zeros_like(utility_bkp)
            if bool(valid_bk.any().item()):
                if hard_target:
                    target_bkp.scatter_(-1, best_idx_bk.unsqueeze(-1), 1.0)
                    target_bkp = target_bkp * valid_bk.unsqueeze(-1).to(dtype=utility_bkp.dtype)
                    target_bkp = target_bkp * allowed_bkp
                else:
                    temp = max(float(temperature), 1.0e-6)
                    pos_logits = utility_bkp.clamp_min(eps).log() / temp
                    pos_logits = pos_logits.masked_fill(~valid_bk.unsqueeze(-1), 0.0)
                    target_bkp = torch.softmax(pos_logits, dim=-1) * valid_bk.unsqueeze(-1).to(dtype=utility_bkp.dtype)
                    target_bkp = target_bkp * allowed_bkp
                    target_bkp = target_bkp / target_bkp.sum(dim=-1, keepdim=True).clamp_min(eps)
        else:
            if not bool(valid_bk.any().item()):
                return (None, diagnostics) if return_diagnostics else None
            if hard_target:
                target_bkp = torch.zeros_like(utility_bkp)
                target_bkp.scatter_(-1, best_idx_bk.unsqueeze(-1), 1.0)
                target_bkp = target_bkp * valid_bk.unsqueeze(-1).to(dtype=utility_bkp.dtype)
                target_bkp = target_bkp * allowed_bkp
            else:
                temp = max(float(temperature), 1.0e-6)
                target_bkp = utility_bkp.clamp_min(eps).log() / temp
                target_bkp = target_bkp.masked_fill(~valid_bk.unsqueeze(-1), 0.0)
                target_bkp = torch.softmax(target_bkp, dim=-1) * valid_bk.unsqueeze(-1).to(dtype=utility_bkp.dtype)
                target_bkp = target_bkp * allowed_bkp
                target_bkp = target_bkp / target_bkp.sum(dim=-1, keepdim=True).clamp_min(eps)
    if bool(include_skip):
        if bool(probs_include_skip_mass):
            penalty_mass_bkp = probs_bkp.clamp_min(eps)
        else:
            penalty_mass_bkp = (1.0 - skip_prob_bk).clamp_min(eps).unsqueeze(-1) * probs_bkp.clamp_min(eps)
        loss_penalty_bk = -(target_bkp * penalty_mass_bkp.log()).sum(dim=-1)
        loss_skip_bk = -(target_skip_bk * skip_prob_bk.clamp_min(eps).log())
        loss_bk = loss_penalty_bk + loss_skip_bk
        return (loss_bk, diagnostics) if return_diagnostics else loss_bk
    log_probs = probs_bkp.clamp_min(eps).log()
    loss_bk = -(target_bkp * log_probs).sum(dim=-1)
    loss_bk = torch.where(valid_bk, loss_bk, torch.zeros_like(loss_bk))
    return (loss_bk, diagnostics) if return_diagnostics else loss_bk


def _pred_residual_candidate_supervision_loss(
    *,
    y_base_bch: torch.Tensor,
    pred_out: Optional[Dict[str, torch.Tensor]],
    y_bch: torch.Tensor,
    cluster_id_c: torch.Tensor,
    K: int,
    penalty_names: Optional[List[str]] = None,
    penalty_fns: Optional[Dict[str, callable]] = None,
    penalty_scale: Optional[torch.Tensor] = None,
    allowed_mask_kp: Optional[torch.Tensor] = None,
    only_allowed: bool = True,
    loss_kind: str = "mse",
    min_abs_improvement: float = 0.0,
    min_rel_improvement: float = 0.0,
    include_intervention: bool = True,
    include_selector: bool = True,
    apply_output_anchors: bool = False,
    x_bcl: Optional[torch.Tensor] = None,
    query_start_abs_b: Optional[torch.Tensor] = None,
    input_len: int = 0,
    moe_cfg: Optional[dict] = None,
    moe_enable: bool = True,
    observed_history_tc: Optional[torch.Tensor] = None,
    train_stat_anchor_pc: Optional[torch.Tensor] = None,
    train_residual_anchor_phc: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    if pred_out is None:
        return None
    base_eval_bch, cand_bcpH = _pred_residual_candidates_on_eval_path(
        y_base_bch,
        pred_out,
        apply_output_anchors=apply_output_anchors,
        x_bcl=x_bcl,
        query_start_abs_b=query_start_abs_b,
        input_len=int(input_len),
        moe_cfg=moe_cfg,
        moe_enable=moe_enable,
        observed_history_tc=observed_history_tc,
        train_stat_anchor_pc=train_stat_anchor_pc,
        train_residual_anchor_phc=train_residual_anchor_phc,
        include_intervention=include_intervention,
        include_selector=include_selector,
    )
    if cand_bcpH is None or cand_bcpH.numel() == 0:
        return None
    loss_mode = str(loss_kind).lower()
    if loss_mode in {"own_penalty", "penalty", "attribute", "shape"}:
        if penalty_names is None or penalty_fns is None or len(penalty_names) != int(cand_bcpH.shape[2]):
            raise ValueError("own_penalty candidate supervision requires one penalty name/function per branch.")
        per_penalty = []
        for p, name in enumerate(penalty_names):
            pen_bc = penalty_fns[name](cand_bcpH[:, :, p, :], y_bch)
            if penalty_scale is not None and penalty_scale.numel() > p:
                scale_p = penalty_scale[p].to(device=pen_bc.device, dtype=pen_bc.dtype).clamp_min(1.0e-6)
                pen_bc = pen_bc / scale_p
            per_penalty.append(pen_bc)
        err_bcp = torch.stack(per_penalty, dim=-1)
    elif loss_mode in {"mse", "l2"}:
        err_bcpH = cand_bcpH - y_bch.unsqueeze(2)
        err_bcp = err_bcpH.pow(2).mean(dim=-1)
    elif loss_mode in {"mae", "l1"}:
        err_bcpH = cand_bcpH - y_bch.unsqueeze(2)
        err_bcp = err_bcpH.abs().mean(dim=-1)
    elif loss_mode in {"gain_hinge", "gain_hinge_mse", "mse_gain_hinge", "residual_gain_hinge"}:
        cand_err_bcp = (cand_bcpH - y_bch.unsqueeze(2)).pow(2).mean(dim=-1)
        base_err_bc = (base_eval_bch - y_bch).pow(2).mean(dim=-1)
        required_bc = torch.maximum(
            torch.full_like(base_err_bc, max(0.0, float(min_abs_improvement))),
            max(0.0, float(min_rel_improvement)) * base_err_bc.detach().abs().clamp_min(1.0e-12),
        )
        err_bcp = (cand_err_bcp - base_err_bc.detach().unsqueeze(-1) + required_bc.unsqueeze(-1)).clamp_min(0.0)
    elif loss_mode in {"gain_hinge_mae", "mae_gain_hinge", "l1_gain_hinge"}:
        cand_err_bcp = (cand_bcpH - y_bch.unsqueeze(2)).abs().mean(dim=-1)
        base_err_bc = (base_eval_bch - y_bch).abs().mean(dim=-1)
        required_bc = torch.maximum(
            torch.full_like(base_err_bc, max(0.0, float(min_abs_improvement))),
            max(0.0, float(min_rel_improvement)) * base_err_bc.detach().abs().clamp_min(1.0e-12),
        )
        err_bcp = (cand_err_bcp - base_err_bc.detach().unsqueeze(-1) + required_bc.unsqueeze(-1)).clamp_min(0.0)
    else:
        raise ValueError(
            "moe.pred_side_residual.candidate_supervision.loss must be mse, mae, own_penalty, "
            "gain_hinge_mse, or gain_hinge_mae "
            f"(got {loss_kind!r})."
        )
    err_bkp = scatter_mean_bcf_to_bkf(err_bcp, cluster_id_c, K)
    if bool(only_allowed) and allowed_mask_kp is not None and allowed_mask_kp.numel() > 0:
        allowed = allowed_mask_kp.to(device=err_bkp.device, dtype=err_bkp.dtype)
        if allowed.shape != err_bkp.shape[1:]:
            raise ValueError(
                "candidate_supervision allowed mask must have shape [K,P], "
                f"got {tuple(allowed.shape)} vs {tuple(err_bkp.shape[1:])}."
            )
        empty = allowed.sum(dim=-1, keepdim=True) <= 0.0
        allowed = torch.where(empty, torch.ones_like(allowed), allowed)
        return (err_bkp * allowed.unsqueeze(0)).sum(dim=-1) / allowed.sum(dim=-1).clamp_min(1.0).view(1, -1)
    return err_bkp.mean(dim=-1)


def _pred_residual_intervention_supervision_loss(
    *,
    y_base_bch: torch.Tensor,
    pred_out: Optional[Dict[str, torch.Tensor]],
    y_bch: torch.Tensor,
    cluster_id_c: torch.Tensor,
    K: int,
    allowed_mask_kp: Optional[torch.Tensor] = None,
    only_allowed: bool = True,
    min_gain: float = 0.0,
    pos_weight: float = 1.0,
    apply_output_anchors: bool = False,
    x_bcl: Optional[torch.Tensor] = None,
    query_start_abs_b: Optional[torch.Tensor] = None,
    input_len: int = 0,
    moe_cfg: Optional[dict] = None,
    moe_enable: bool = True,
    observed_history_tc: Optional[torch.Tensor] = None,
    train_stat_anchor_pc: Optional[torch.Tensor] = None,
    train_residual_anchor_phc: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    if pred_out is None:
        return None
    intervention_bcp = pred_out.get("intervention_bcp")
    if intervention_bcp is None or intervention_bcp.numel() == 0:
        return None
    base_eval_bch, cand_bcpH = _pred_residual_candidates_on_eval_path(
        y_base_bch,
        pred_out,
        apply_output_anchors=apply_output_anchors,
        x_bcl=x_bcl,
        query_start_abs_b=query_start_abs_b,
        input_len=int(input_len),
        moe_cfg=moe_cfg,
        moe_enable=moe_enable,
        observed_history_tc=observed_history_tc,
        train_stat_anchor_pc=train_stat_anchor_pc,
        train_residual_anchor_phc=train_residual_anchor_phc,
        include_intervention=False,
        include_selector=False,
    )
    if cand_bcpH is None or cand_bcpH.numel() == 0:
        return None
    base_err_bc = (base_eval_bch.detach() - y_bch.detach()).pow(2).mean(dim=-1)
    cand_err_bcp = (cand_bcpH.detach() - y_bch.detach().unsqueeze(2)).pow(2).mean(dim=-1)
    target_bcp = ((base_err_bc.unsqueeze(-1) - cand_err_bcp) > float(min_gain)).to(dtype=intervention_bcp.dtype)
    prob_bcp = intervention_bcp.clamp(1.0e-6, 1.0 - 1.0e-6)
    loss_bcp = -(
        float(max(pos_weight, 0.0)) * target_bcp * prob_bcp.log()
        + (1.0 - target_bcp) * (1.0 - prob_bcp).log()
    )
    if bool(only_allowed) and allowed_mask_kp is not None and allowed_mask_kp.numel() > 0:
        allowed_cp = allowed_mask_kp.to(device=loss_bcp.device, dtype=torch.bool).index_select(
            0,
            cluster_id_c.to(device=loss_bcp.device, dtype=torch.long),
        )
        if allowed_cp.shape != loss_bcp.shape[1:]:
            raise ValueError(
                "intervention supervision allowed mask must broadcast to [C,P], "
                f"got {tuple(allowed_cp.shape)} vs {tuple(loss_bcp.shape[1:])}."
            )
        loss_bcp = loss_bcp * allowed_cp.to(dtype=loss_bcp.dtype).unsqueeze(0)
        denom_bc = allowed_cp.to(dtype=loss_bcp.dtype).sum(dim=-1).clamp_min(1.0).view(1, -1)
        loss_bc = loss_bcp.sum(dim=-1) / denom_bc
    else:
        loss_bc = loss_bcp.mean(dim=-1)
    return scatter_mean_bc_to_bk(loss_bc, cluster_id_c, K)


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
        feature_mode: str = "base",
    ):
        super().__init__()
        self.base_F = int(feat_dim)
        self.C = int(num_channels)
        self.P = int(num_penalties)
        self.use_penalty_identity = bool(use_penalty_identity)
        self.feature_mode = str(feature_mode or "base").lower()
        self.F = self.base_F + (self.P if self.use_penalty_identity else 0)
        hidden = int(hidden_dim)
        self.register_buffer("feature_mean", torch.zeros(1, 1, self.F), persistent=True)
        self.register_buffer("feature_std", torch.ones(1, 1, self.F), persistent=True)
        self.register_buffer("feature_standardize_enabled", torch.tensor(0.0), persistent=True)
        self.register_buffer("feature_standardize_clip", torch.tensor(0.0), persistent=True)
        self.register_buffer("allowed_penalty_mask_cp", torch.empty(0, 0, dtype=torch.bool), persistent=False)
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

    def set_feature_standardize_clip(self, clip_value: float) -> None:
        self.feature_standardize_clip.fill_(max(0.0, float(clip_value)))

    def set_allowed_penalty_mask(self, allowed_mask_cp: Optional[torch.Tensor]) -> None:
        if allowed_mask_cp is None or int(allowed_mask_cp.numel()) == 0:
            self.allowed_penalty_mask_cp = torch.empty(0, 0, device=self.skip_bias.device, dtype=torch.bool)
            return
        allowed = _selector_allowed_mask_cp(
            allowed_mask_cp,
            C=self.C,
            P=self.P,
            device=self.skip_bias.device,
            context="candidate selector",
        )
        self.allowed_penalty_mask_cp = allowed.detach().clone()

    def _resolved_allowed_penalty_mask(self, allowed_mask_cp: Optional[torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
        mask = allowed_mask_cp
        if mask is None or int(mask.numel()) == 0:
            mask = self.allowed_penalty_mask_cp
        return _selector_allowed_mask_cp(mask, C=self.C, P=self.P, device=device, context="candidate selector")

    def _mask_disallowed_logits(
        self,
        logits_bcq: torch.Tensor,
        allowed_mask_cp: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        allowed = self._resolved_allowed_penalty_mask(allowed_mask_cp, logits_bcq.device)
        if allowed is None:
            return logits_bcq
        penalty_logits = logits_bcq[..., 1:].masked_fill(~allowed.unsqueeze(0), torch.finfo(logits_bcq.dtype).min)
        return torch.cat([logits_bcq[..., :1], penalty_logits], dim=-1)

    def _standardize_feat(self, feat: torch.Tensor) -> torch.Tensor:
        if bool(self.feature_standardize_enabled.item() > 0.5):
            feat = (feat - self.feature_mean.to(device=feat.device, dtype=feat.dtype)) / self.feature_std.to(
                device=feat.device,
                dtype=feat.dtype,
            )
            clip_value = float(self.feature_standardize_clip.item())
            if clip_value > 0.0:
                feat = feat.clamp(min=-clip_value, max=clip_value)
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
        allowed_mask_cp: Optional[torch.Tensor] = None,
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
        logits_bcq = torch.cat([skip_score_bc.unsqueeze(-1), cand_score_bcp], dim=-1)
        return self._mask_disallowed_logits(logits_bcq, allowed_mask_cp=allowed_mask_cp)

    def logits(
        self,
        x_bcl: torch.Tensor,
        base_bch: torch.Tensor,
        cand_bcpH: torch.Tensor,
        allowed_mask_cp: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        skip_feat = _candidate_selector_features(x_bcl, base_bch, base_bch, feature_mode=self.feature_mode)
        cand_feat = torch.stack(
            [
                _candidate_selector_features(x_bcl, base_bch, cand_bcpH[:, :, p, :], feature_mode=self.feature_mode)
                for p in range(int(cand_bcpH.shape[2]))
            ],
            dim=2,
        )
        return self.logits_from_features(skip_feat, cand_feat, allowed_mask_cp=allowed_mask_cp)

    def select_from_features(
        self,
        skip_feat_bcf: torch.Tensor,
        cand_feat_bcpf: torch.Tensor,
        base_bch: torch.Tensor,
        cand_bcpH: torch.Tensor,
        allowed_mask_cp: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits_bcq = self.logits_from_features(
            skip_feat_bcf,
            cand_feat_bcpf,
            apply_decision_margin=True,
            allowed_mask_cp=allowed_mask_cp,
        )
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
        allowed_mask_cp: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        skip_feat = _candidate_selector_features(x_bcl, base_bch, base_bch, feature_mode=self.feature_mode)
        cand_feat = torch.stack(
            [
                _candidate_selector_features(x_bcl, base_bch, cand_bcpH[:, :, p, :], feature_mode=self.feature_mode)
                for p in range(int(cand_bcpH.shape[2]))
            ],
            dim=2,
        )
        return self.select_from_features(skip_feat, cand_feat, base_bch, cand_bcpH, allowed_mask_cp=allowed_mask_cp)


class StaticPredResidualCandidateSelector(nn.Module):
    def __init__(self, selected_class_c: torch.Tensor):
        super().__init__()
        selected = selected_class_c.detach().to(dtype=torch.long).view(-1)
        self.register_buffer("selected_class_c", selected, persistent=True)

    def select_prediction(
        self,
        x_bcl: torch.Tensor,
        base_bch: torch.Tensor,
        cand_bcpH: torch.Tensor,
        allowed_mask_cp: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H = base_bch.shape
        if int(self.selected_class_c.numel()) != int(C):
            raise ValueError(
                f"Static candidate selector expected {int(self.selected_class_c.numel())} channels, got {int(C)}."
            )
        selected_class_c = self.selected_class_c.to(device=base_bch.device)
        if allowed_mask_cp is not None and int(allowed_mask_cp.numel()) > 0:
            allowed = _selector_allowed_mask_cp(
                allowed_mask_cp,
                C=int(C),
                P=int(cand_bcpH.shape[2]),
                device=base_bch.device,
                context="static candidate selector",
            )
            penalty_idx_c = (selected_class_c - 1).clamp_min(0)
            candidate_selected_c = selected_class_c > 0
            allowed_selected_c = allowed.gather(1, penalty_idx_c.view(-1, 1)).squeeze(1)
            selected_class_c = torch.where(
                candidate_selected_c & allowed_selected_c,
                selected_class_c,
                torch.zeros_like(selected_class_c),
            )
        selected_class_bc = selected_class_c.view(1, C).expand(B, C)
        all_pred_bcqh = torch.cat([base_bch.unsqueeze(2), cand_bcpH], dim=2)
        gather_idx = selected_class_bc.view(B, C, 1, 1).expand(B, C, 1, H)
        selected_bch = all_pred_bcqh.gather(2, gather_idx).squeeze(2)
        return selected_bch, selected_class_bc


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




@torch.no_grad()




@torch.no_grad()


@torch.no_grad()


@torch.no_grad()


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
    model_train_stat_adapter_pc: Optional[torch.Tensor] = None,
    model_train_stat_adapter_cfg: Optional[dict] = None,
    train_stat_anchor_pc: Optional[torch.Tensor] = None,
    train_residual_anchor_phc: Optional[torch.Tensor] = None,
    allowed_mask_kp: Optional[torch.Tensor] = None,
    gate_feature_mode: str = "history",
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
    cid_c = cluster_id_c.detach().to(device=device, dtype=torch.long)
    allowed_mask_device = None
    if allowed_mask_kp is not None:
        allowed_mask_device = allowed_mask_kp.detach().to(device=device, dtype=torch.bool)
        if tuple(allowed_mask_device.shape) != (int(K), int(P)):
            raise ValueError("allowed_mask_kp must have shape [K,P] for gate penalty hit diagnostics.")
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
        feat_bkf = _build_gate_routing_features(x, y_base, cluster_id_c, K, mode=gate_feature_mode)
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
            query_start_abs_b=query_start_abs_b,
        )
        residuals = pred_out.get("residuals")
        intervention_bcp = pred_out.get("intervention_bcp")
        selector_bcp = pred_out.get("selector_bcp")
        alpha_cp = pred_out.get("alpha_cp")
        if residuals is None or intervention_bcp is None or alpha_cp is None or residuals.numel() == 0:
            continue
        if selector_bcp is None:
            selector_bcp = torch.ones_like(intervention_bcp)

        y_base_final, cand_bcpH = _pred_residual_candidates_on_eval_path(
            y_base,
            pred_out,
            apply_output_anchors=True,
            x_bcl=x,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len),
            moe_cfg=moe_cfg,
            moe_enable=moe_enable,
            observed_history_tc=observed_history_tc,
            train_stat_anchor_pc=train_stat_anchor_pc,
            train_residual_anchor_phc=train_residual_anchor_phc,
        )
        if cand_bcpH is None:
            continue
        err_bcp = (cand_bcpH - y.unsqueeze(2)).pow(2).mean(dim=-1)
        base_err_bc = (y_base_final - y).pow(2).mean(dim=-1)
        if allowed_mask_device is not None:
            allowed_bcp = allowed_mask_device.index_select(0, cid_c).unsqueeze(0)
            err_for_oracle_bcp = err_bcp.masked_fill(~allowed_bcp, float("inf"))
        else:
            err_for_oracle_bcp = err_bcp
        oracle_penalty_err_bc, oracle_p_bc = err_for_oracle_bcp.min(dim=-1)
        oracle_err_bc = torch.where(torch.isfinite(oracle_penalty_err_bc), oracle_penalty_err_bc, base_err_bc)
        route_bcp = mask_bkp[:, cid_c, :]
        probs_bcp = probs_bkp[:, cid_c, :]
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
        base_se += float((y_base_final - y).pow(2).sum().item())
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


def _explainability_train_subsplit_ranges(
    num_windows: int,
    holdout_fraction: float = 0.30,
) -> Dict[str, Tuple[int, int]]:
    n = int(num_windows)
    if n <= 0:
        return {}
    if n == 1:
        return {"train_fit": (0, 1)}
    frac = max(0.0, min(float(holdout_fraction), 0.95))
    holdout = int(math.ceil(float(n) * frac))
    holdout = max(1, min(holdout, n - 1))
    cut = n - holdout
    return {
        "train_fit": (0, cut),
        "train_holdout": (cut, n),
    }


def _cluster_route_label_feature_diagnostics(
    feat_bkf: torch.Tensor,
    route_label_bk: torch.Tensor,
    penalty_names: List[str],
    feature_names: Optional[List[str]] = None,
) -> Dict[str, object]:
    label_names = ["skip"] + [str(name) for name in penalty_names]
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(int(feat_bkf.shape[-1]))]
    else:
        feature_names = [str(name) for name in feature_names]

    def _label_name(label: int) -> Optional[str]:
        if 0 <= int(label) < len(label_names):
            return label_names[int(label)]
        return None

    def _majority_label(labels_n: torch.Tensor, label_count: int) -> Tuple[int, float]:
        if labels_n.numel() <= 0:
            return -1, 0.0
        counts = torch.bincount(labels_n.to(dtype=torch.long), minlength=label_count)[:label_count]
        label = int(counts.argmax().item())
        acc = float(counts[label].item() / max(int(labels_n.numel()), 1))
        return label, acc

    feat = feat_bkf.detach().cpu().to(dtype=torch.float32)
    labels = route_label_bk.detach().cpu().to(dtype=torch.long)
    if feat.dim() != 3 or labels.dim() != 2:
        raise ValueError("feat_bkf must be [B,K,F] and route_label_bk must be [B,K].")
    if int(feat.shape[0]) != int(labels.shape[0]) or int(feat.shape[1]) != int(labels.shape[1]):
        raise ValueError("feat_bkf and route_label_bk must share [B,K].")

    B, K, F = int(feat.shape[0]), int(feat.shape[1]), int(feat.shape[2])
    per_cluster = []
    label_count = len(label_names)
    for k in range(K):
        valid = (labels[:, k] >= 0) & (labels[:, k] < label_count)
        labels_k = labels[valid, k]
        feat_kf = feat[valid, k, :]
        samples = int(labels_k.numel())
        counts = torch.bincount(labels_k, minlength=label_count)[:label_count] if samples > 0 else torch.zeros(label_count, dtype=torch.long)
        rates = counts.to(dtype=torch.float64) / max(samples, 1)
        majority_label, majority_acc = _majority_label(labels_k, label_count)
        entropy_bits = 0.0
        for rate in rates.tolist():
            if float(rate) > 0.0:
                entropy_bits -= float(rate) * math.log(float(rate), 2)

        best_stump = None
        if samples >= 2 and F > 0 and int((counts > 0).sum().item()) >= 2:
            best_acc = -1.0
            best_payload = None
            for f in range(F):
                values = feat_kf[:, f]
                finite = torch.isfinite(values)
                if int(finite.sum().item()) < 2:
                    continue
                values_f = values[finite]
                labels_f = labels_k[finite]
                unique = torch.unique(values_f).sort().values
                if int(unique.numel()) < 2:
                    continue
                if int(unique.numel()) <= 16:
                    thresholds = (unique[:-1] + unique[1:]) * 0.5
                else:
                    quantiles = torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90], dtype=values_f.dtype)
                    thresholds = torch.unique(torch.quantile(values_f, quantiles)).sort().values
                for threshold in thresholds:
                    left = values_f <= threshold
                    right = ~left
                    if int(left.sum().item()) <= 0 or int(right.sum().item()) <= 0:
                        continue
                    left_label, _ = _majority_label(labels_f[left], label_count)
                    right_label, _ = _majority_label(labels_f[right], label_count)
                    pred = torch.where(
                        left,
                        torch.full_like(labels_f, int(left_label)),
                        torch.full_like(labels_f, int(right_label)),
                    )
                    acc = float((pred == labels_f).to(dtype=torch.float32).mean().item())
                    if acc > best_acc:
                        best_acc = acc
                        best_payload = {
                            "feature": feature_names[f] if f < len(feature_names) else f"feature_{f}",
                            "feature_index": int(f),
                            "threshold": float(threshold.item()),
                            "op": "<=",
                            "left_label": _label_name(left_label),
                            "right_label": _label_name(right_label),
                            "accuracy": acc,
                            "lift_vs_majority": float(acc - majority_acc),
                        }
            best_stump = best_payload

        label_counts = {label_names[i]: int(counts[i].item()) for i in range(label_count)}
        label_rates = {label_names[i]: float(rates[i].item()) for i in range(label_count)}
        per_cluster.append(
            {
                "cluster_id": int(k),
                "samples": samples,
                "label_counts": label_counts,
                "label_rates": label_rates,
                "majority_label": _label_name(majority_label),
                "majority_acc": float(majority_acc),
                "entropy_bits": float(entropy_bits),
                "best_stump": best_stump,
            }
        )

    return {
        "samples": int(B * K),
        "label_names": label_names,
        "feature_names": feature_names,
        "per_cluster": per_cluster,
    }


def _cluster_route_label_phase_diagnostics(
    query_start_abs_b: torch.Tensor,
    route_label_bk: torch.Tensor,
    penalty_names: List[str],
    periods: List[int],
    num_bins: int = 8,
    phase_offset: int = 0,
) -> Dict[str, object]:
    label_names = ["skip"] + [str(name) for name in penalty_names]
    label_count = len(label_names)

    def _label_name(label: int) -> Optional[str]:
        if 0 <= int(label) < label_count:
            return label_names[int(label)]
        return None

    def _counts_payload(labels_n: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, int], Dict[str, float], Optional[str], float]:
        if labels_n.numel() <= 0:
            counts = torch.zeros(label_count, dtype=torch.long)
            return counts, {name: 0 for name in label_names}, {name: 0.0 for name in label_names}, None, 0.0
        counts = torch.bincount(labels_n.to(dtype=torch.long), minlength=label_count)[:label_count]
        total = int(labels_n.numel())
        majority = int(counts.argmax().item())
        label_counts = {label_names[i]: int(counts[i].item()) for i in range(label_count)}
        label_rates = {label_names[i]: float(counts[i].item() / max(total, 1)) for i in range(label_count)}
        majority_acc = float(counts[majority].item() / max(total, 1))
        return counts, label_counts, label_rates, _label_name(majority), majority_acc

    starts = query_start_abs_b.detach().cpu().to(dtype=torch.long).reshape(-1)
    labels = route_label_bk.detach().cpu().to(dtype=torch.long)
    if labels.dim() != 2:
        raise ValueError("route_label_bk must have shape [B,K].")
    if int(labels.shape[0]) != int(starts.numel()):
        raise ValueError("query_start_abs_b length must match route_label_bk batch dimension.")

    clean_periods = []
    for period in periods:
        period_i = int(period)
        if period_i > 0 and period_i not in clean_periods:
            clean_periods.append(period_i)
    bins_requested = int(num_bins)
    if bins_requested <= 0:
        bins_requested = 8

    K = int(labels.shape[1])
    per_period = []
    for period in clean_periods:
        bins = max(1, min(int(bins_requested), int(period)))
        phase_b = (starts + int(phase_offset)).remainder(int(period))
        bin_b = torch.div(phase_b * bins, int(period), rounding_mode="floor").clamp(min=0, max=bins - 1)
        period_clusters = []
        for k in range(K):
            valid = (labels[:, k] >= 0) & (labels[:, k] < label_count)
            labels_k = labels[valid, k]
            bin_k = bin_b[valid]
            samples = int(labels_k.numel())
            _, label_counts, label_rates, global_majority_label, global_majority_acc = _counts_payload(labels_k)
            bin_payloads = []
            phase_correct = 0
            for b in range(bins):
                bin_mask = bin_k == int(b)
                labels_bin = labels_k[bin_mask]
                bin_samples = int(labels_bin.numel())
                counts, bin_counts, bin_rates, majority_label, majority_acc = _counts_payload(labels_bin)
                if bin_samples > 0:
                    phase_correct += int(counts.max().item())
                start_phase = int(math.floor(float(b) * float(period) / float(bins)))
                end_phase = int(math.floor(float(b + 1) * float(period) / float(bins)))
                bin_payloads.append(
                    {
                        "bin": int(b),
                        "phase_start": start_phase,
                        "phase_end": end_phase,
                        "samples": bin_samples,
                        "label_counts": bin_counts,
                        "label_rates": bin_rates,
                        "majority_label": majority_label,
                        "majority_acc": float(majority_acc),
                    }
                )
            phase_majority_acc = float(phase_correct / max(samples, 1))
            period_clusters.append(
                {
                    "cluster_id": int(k),
                    "samples": samples,
                    "label_counts": label_counts,
                    "label_rates": label_rates,
                    "global_majority_label": global_majority_label,
                    "global_majority_acc": float(global_majority_acc),
                    "phase_majority_acc": phase_majority_acc,
                    "lift_vs_global": float(phase_majority_acc - global_majority_acc),
                    "bins": bin_payloads,
                }
            )
        per_period.append(
            {
                "period": int(period),
                "num_bins": int(bins),
                "per_cluster": period_clusters,
            }
        )

    return {
        "samples": int(labels.numel()),
        "window_count": int(labels.shape[0]),
        "label_names": label_names,
        "periods": clean_periods,
        "num_bins_requested": int(bins_requested),
        "phase_offset": int(phase_offset),
        "per_period": per_period,
    }


def _cluster_top1_confidence_gain_diagnostics(
    *,
    top1_conf_bc: torch.Tensor,
    top1_gain_bc: torch.Tensor,
    top1_p_bc: torch.Tensor,
    top1_active_bc: torch.Tensor,
    skip_bc: torch.Tensor,
    cluster_id_c: torch.Tensor,
    K: int,
    penalty_names: List[str],
    bins: List[float],
) -> Dict[str, object]:
    conf = top1_conf_bc.detach().cpu().to(dtype=torch.float32)
    gain = top1_gain_bc.detach().cpu().to(dtype=torch.float32)
    top1_p = top1_p_bc.detach().cpu().to(dtype=torch.long)
    active = top1_active_bc.detach().cpu().to(dtype=torch.bool)
    skipped = skip_bc.detach().cpu().to(dtype=torch.bool)
    cid = cluster_id_c.detach().cpu().to(dtype=torch.long)
    if conf.shape != gain.shape or conf.shape != top1_p.shape or conf.shape != active.shape or conf.shape != skipped.shape:
        raise ValueError("top1 confidence diagnostics inputs must share shape [B,C].")
    if conf.dim() != 2:
        raise ValueError("top1 confidence diagnostics inputs must have shape [B,C].")
    if int(cid.numel()) != int(conf.shape[1]):
        raise ValueError("cluster_id_c length must match confidence channel dimension.")

    clean_bins = sorted({float(v) for v in bins if math.isfinite(float(v))})
    if len(clean_bins) < 2:
        clean_bins = [0.0, 0.5, 1.0]
    if clean_bins[0] > 0.0:
        clean_bins = [0.0] + clean_bins
    if clean_bins[-1] < 1.0:
        clean_bins = clean_bins + [1.0]
    label_names = [str(name) for name in penalty_names]

    def _bin_payload(mask_bc: torch.Tensor, lower: float, upper: float, is_last: bool) -> Dict[str, object]:
        if is_last:
            bin_mask = mask_bc & (conf >= float(lower)) & (conf <= float(upper))
        else:
            bin_mask = mask_bc & (conf >= float(lower)) & (conf < float(upper))
        samples = int(bin_mask.sum().item())
        active_mask = bin_mask & active
        skipped_mask = bin_mask & skipped
        active_count = int(active_mask.sum().item())
        skipped_count = int(skipped_mask.sum().item())
        if samples > 0:
            mean_gain = float(gain[bin_mask].mean().item())
            positive_rate = float((gain[bin_mask] > 0.0).to(dtype=torch.float32).mean().item())
        else:
            mean_gain = 0.0
            positive_rate = 0.0
        if active_count > 0:
            active_mean_gain = float(gain[active_mask].mean().item())
            active_positive_rate = float((gain[active_mask] > 0.0).to(dtype=torch.float32).mean().item())
            harmful_not_skipped_rate = float((gain[active_mask] <= 0.0).to(dtype=torch.float32).mean().item())
        else:
            active_mean_gain = 0.0
            active_positive_rate = 0.0
            harmful_not_skipped_rate = 0.0
        if skipped_count > 0:
            skipped_positive_rate = float((gain[skipped_mask] > 0.0).to(dtype=torch.float32).mean().item())
        else:
            skipped_positive_rate = 0.0
        return {
            "low": float(lower),
            "high": float(upper),
            "samples": samples,
            "active_count": active_count,
            "skipped_count": skipped_count,
            "mean_gain_mse": mean_gain,
            "positive_rate": positive_rate,
            "active_mean_gain_mse": active_mean_gain,
            "active_positive_rate": active_positive_rate,
            "harmful_not_skipped_rate": harmful_not_skipped_rate,
            "skipped_positive_rate": skipped_positive_rate,
        }

    def _group_payload(mask_bc: torch.Tensor) -> Dict[str, object]:
        payload_bins = []
        for i in range(len(clean_bins) - 1):
            payload_bins.append(
                _bin_payload(
                    mask_bc,
                    lower=clean_bins[i],
                    upper=clean_bins[i + 1],
                    is_last=i == len(clean_bins) - 2,
                )
            )
        samples = int(mask_bc.sum().item())
        if samples > 0:
            mean_gain = float(gain[mask_bc].mean().item())
            positive_rate = float((gain[mask_bc] > 0.0).to(dtype=torch.float32).mean().item())
        else:
            mean_gain = 0.0
            positive_rate = 0.0
        return {
            "samples": samples,
            "mean_gain_mse": mean_gain,
            "positive_rate": positive_rate,
            "bins": payload_bins,
        }

    per_cluster = []
    for k in range(int(K)):
        cluster_mask = (cid == int(k)).view(1, -1).expand_as(conf)
        per_penalty = []
        for p, name in enumerate(label_names):
            per_penalty.append(
                {
                    "penalty": name,
                    **_group_payload(cluster_mask & (top1_p == int(p))),
                }
            )
        per_cluster.append(
            {
                "cluster_id": int(k),
                "all": _group_payload(cluster_mask),
                "per_penalty": per_penalty,
            }
        )

    return {
        "bins": [float(v) for v in clean_bins],
        "penalty_names": label_names,
        "per_cluster": per_cluster,
    }


def _build_penalty_route_learnability_class_features(
    *,
    gate_feat_bkf: torch.Tensor,
    skip_feat_bkf: torch.Tensor,
    cand_feat_bkpf: torch.Tensor,
    gate_prob_bkp: torch.Tensor,
    route_bkp: torch.Tensor,
    intervention_bkp: torch.Tensor,
    selector_bkp: torch.Tensor,
    alpha_bkp: torch.Tensor,
    skip_prob_bk: Optional[torch.Tensor],
    cluster_count: int,
    penalty_names: List[str],
) -> Tuple[torch.Tensor, List[str]]:
    if gate_feat_bkf.dim() != 3:
        raise ValueError("gate_feat_bkf must have shape [B,K,F].")
    if skip_feat_bkf.dim() != 3:
        raise ValueError("skip_feat_bkf must have shape [B,K,F].")
    if cand_feat_bkpf.dim() != 4:
        raise ValueError("cand_feat_bkpf must have shape [B,K,P,F].")
    B, K, gate_dim = [int(v) for v in gate_feat_bkf.shape]
    if tuple(skip_feat_bkf.shape[:2]) != (B, K):
        raise ValueError("skip_feat_bkf must share [B,K] with gate_feat_bkf.")
    if tuple(cand_feat_bkpf.shape[:3]) != (B, K, len(penalty_names)):
        raise ValueError("cand_feat_bkpf must share [B,K,P] with penalty_names.")
    P = int(cand_feat_bkpf.shape[2])
    Q = P + 1
    stat_shape = (B, K, P)
    for name, value in [
        ("gate_prob_bkp", gate_prob_bkp),
        ("route_bkp", route_bkp),
        ("intervention_bkp", intervention_bkp),
        ("selector_bkp", selector_bkp),
    ]:
        if tuple(value.shape) != stat_shape:
            raise ValueError(f"{name} must have shape [B,K,P].")
    if alpha_bkp.dim() == 2:
        if tuple(alpha_bkp.shape) != (K, P):
            raise ValueError("alpha_bkp must have shape [K,P] or [B,K,P].")
        alpha_bkp = alpha_bkp.unsqueeze(0).expand(B, K, P)
    elif tuple(alpha_bkp.shape) != stat_shape:
        raise ValueError("alpha_bkp must have shape [K,P] or [B,K,P].")
    if skip_prob_bk is None:
        skip_prob_bk = torch.zeros(B, K, device=gate_feat_bkf.device, dtype=gate_feat_bkf.dtype)
    if tuple(skip_prob_bk.shape) != (B, K):
        raise ValueError("skip_prob_bk must have shape [B,K].")

    dtype = gate_feat_bkf.dtype
    device = gate_feat_bkf.device
    zero_bk1 = torch.zeros(B, K, 1, device=device, dtype=dtype)
    class_gate_prob = torch.cat([zero_bk1, gate_prob_bkp.to(device=device, dtype=dtype)], dim=-1)
    class_route = torch.cat([zero_bk1, route_bkp.to(device=device, dtype=dtype)], dim=-1)
    class_intervention = torch.cat([zero_bk1, intervention_bkp.to(device=device, dtype=dtype)], dim=-1)
    class_selector = torch.cat([zero_bk1, selector_bkp.to(device=device, dtype=dtype)], dim=-1)
    class_alpha = torch.cat([zero_bk1, alpha_bkp.to(device=device, dtype=dtype)], dim=-1)
    class_skip_prob = skip_prob_bk.to(device=device, dtype=dtype).unsqueeze(-1).expand(B, K, Q)
    stat_features = torch.stack(
        [
            class_gate_prob,
            class_route,
            class_intervention,
            class_selector,
            class_alpha,
            class_skip_prob,
        ],
        dim=-1,
    )

    candidate_features = torch.cat(
        [
            skip_feat_bkf.to(device=device, dtype=dtype).unsqueeze(2),
            cand_feat_bkpf.to(device=device, dtype=dtype),
        ],
        dim=2,
    )
    gate_features = gate_feat_bkf.to(device=device, dtype=dtype).unsqueeze(2).expand(B, K, Q, gate_dim)
    class_eye = torch.eye(Q, device=device, dtype=dtype).view(1, 1, Q, Q).expand(B, K, Q, Q)
    cluster_total = int(cluster_count) if int(cluster_count) > 0 else K
    if cluster_total < K:
        raise ValueError("cluster_count cannot be smaller than gate_feat_bkf.shape[1].")
    cluster_eye = torch.eye(cluster_total, device=device, dtype=dtype)[:K].view(1, K, 1, cluster_total)
    cluster_features = cluster_eye.expand(B, K, Q, cluster_total)
    features = torch.cat(
        [
            gate_features,
            candidate_features,
            stat_features,
            class_eye,
            cluster_features,
        ],
        dim=-1,
    ).contiguous()

    feature_names = (
        [f"gate_feature_{i}" for i in range(gate_dim)]
        + [f"candidate_feature_{i}" for i in range(int(candidate_features.shape[-1]))]
        + ["gate_prob", "route_weight", "intervention", "selector", "alpha", "skip_prob"]
        + ["class_skip"]
        + [f"class_{name}" for name in penalty_names]
        + [f"cluster_{k}" for k in range(cluster_total)]
    )
    return features, feature_names


def _scatter_mean_bcpf_to_bkpf(values_bcpf: torch.Tensor, cluster_id_c: torch.Tensor, K: int) -> torch.Tensor:
    if values_bcpf.dim() != 4:
        raise ValueError("values_bcpf must have shape [B,C,P,F].")
    B, C, P, F_dim = [int(v) for v in values_bcpf.shape]
    flat_bcf = values_bcpf.reshape(B, C, P * F_dim)
    pooled = scatter_mean_bcf_to_bkf(flat_bcf, cluster_id_c, int(K))
    return pooled.reshape(B, int(K), P, F_dim)


@torch.no_grad()
def _collect_penalty_route_learnability_tensors(
    *,
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
    feature_mode: str = "base",
    allowed_mask_kp: Optional[torch.Tensor] = None,
    min_abs_improvement: float = 0.0,
    min_rel_improvement: float = 0.0,
    min_candidate_delta_rms: float = 0.0,
    max_batches: int = 0,
    history_anchor_cfg: Optional[dict] = None,
    observed_history_tc: Optional[torch.Tensor] = None,
    input_len: int = 0,
    eval_start: int = 0,
    model_train_stat_adapter_pc: Optional[torch.Tensor] = None,
    model_train_stat_adapter_cfg: Optional[dict] = None,
    train_stat_anchor_pc: Optional[torch.Tensor] = None,
    train_residual_anchor_phc: Optional[torch.Tensor] = None,
    gate_feature_mode: str = "history",
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
    gate_feature_mode = _normalize_gate_feature_mode(gate_feature_mode)
    P = len(penalty_names)
    cid_c = cluster_id_c.detach().to(device=device, dtype=torch.long)
    allowed_mask_device = None
    if allowed_mask_kp is not None:
        allowed_mask_device = allowed_mask_kp.detach().to(device=device, dtype=torch.bool)
        if tuple(allowed_mask_device.shape) != (int(K), int(P)):
            raise ValueError("allowed_mask_kp must have shape [K,P] for route learnability diagnostics.")

    feature_chunks: List[torch.Tensor] = []
    label_chunks: List[torch.Tensor] = []
    current_chunks: List[torch.Tensor] = []
    query_start_chunks: List[torch.Tensor] = []
    gain_chunks: List[torch.Tensor] = []
    feature_names: Optional[List[str]] = None
    batch_count = 0

    for x, y, idx in loader:
        batch_count += 1
        if max_batches > 0 and batch_count > int(max_batches):
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        idx = idx.to(device=device, dtype=torch.long)
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
        feat_bkf = _build_gate_routing_features(x, y_base, cluster_id_c, K, mode=gate_feature_mode)
        route_pen_bkp = _router_penalty_context_from_history(
            x_bcl=x,
            yhat_base_bch=y_base,
            penalty_names=penalty_names,
            penalty_fns=penalty_fns,
            penalty_scale=penalty_scale,
            cluster_id_c=cluster_id_c,
            K=K,
        )
        mask_bkp, probs_bkp, skip_bk, skip_prob_bk = gate(
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
            query_start_abs_b=query_start_abs_b,
        )
        y_base_final, cand_bcpH = _pred_residual_candidates_on_eval_path(
            y_base,
            pred_out,
            apply_output_anchors=True,
            x_bcl=x,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len),
            moe_cfg=moe_cfg,
            moe_enable=moe_enable,
            observed_history_tc=observed_history_tc,
            train_stat_anchor_pc=train_stat_anchor_pc,
            train_residual_anchor_phc=train_residual_anchor_phc,
        )
        if cand_bcpH is None:
            continue
        labels_bk, raw_gain_bk = _cluster_route_oracle_labels_and_gain_from_candidates(
            base_bch=y_base_final,
            cand_bcpH=cand_bcpH,
            y_bch=y,
            cluster_id_c=cid_c,
            K=int(K),
            allowed_mask_kp=allowed_mask_device,
            min_abs_improvement=float(min_abs_improvement),
            min_rel_improvement=float(min_rel_improvement),
            min_candidate_delta_rms=float(min_candidate_delta_rms),
        )
        gain_bk = raw_gain_bk.clamp_min(0.0)

        route_bcp = pred_out.get("route_bcp", mask_bkp[:, cid_c, :])
        intervention_bcp = pred_out.get("intervention_bcp", torch.ones_like(route_bcp))
        selector_bcp = pred_out.get("selector_bcp", torch.ones_like(route_bcp))
        alpha_cp = pred_out.get("alpha_cp")
        if alpha_cp is None:
            alpha_cp = torch.ones(int(cid_c.numel()), P, device=device, dtype=x.dtype)
        skip_feat_bcf = _candidate_selector_features(x, y_base_final, y_base_final, feature_mode=feature_mode)
        cand_feat_parts = [
            _candidate_selector_features(x, y_base_final, cand_bcpH[:, :, p, :], feature_mode=feature_mode)
            for p in range(P)
        ]
        cand_feat_bcpf = torch.stack(cand_feat_parts, dim=2)
        skip_feat_bkf = scatter_mean_bcf_to_bkf(skip_feat_bcf, cluster_id_c, int(K))
        cand_feat_bkpf = _scatter_mean_bcpf_to_bkpf(cand_feat_bcpf, cluster_id_c, int(K))
        route_bkp = scatter_mean_bcf_to_bkf(route_bcp, cluster_id_c, int(K))
        intervention_bkp = scatter_mean_bcf_to_bkf(intervention_bcp, cluster_id_c, int(K))
        selector_bkp = scatter_mean_bcf_to_bkf(selector_bcp, cluster_id_c, int(K))
        alpha_bkp = scatter_mean_bcf_to_bkf(alpha_cp.unsqueeze(0).expand(x.shape[0], -1, -1), cluster_id_c, int(K))
        class_features_bkqf, names = _build_penalty_route_learnability_class_features(
            gate_feat_bkf=feat_bkf,
            skip_feat_bkf=skip_feat_bkf,
            cand_feat_bkpf=cand_feat_bkpf,
            gate_prob_bkp=probs_bkp,
            route_bkp=route_bkp,
            intervention_bkp=intervention_bkp,
            selector_bkp=selector_bkp,
            alpha_bkp=alpha_bkp,
            skip_prob_bk=skip_prob_bk if allow_skip else None,
            cluster_count=int(K),
            penalty_names=penalty_names,
        )
        feature_names = names
        current_penalty_bk = (mask_bkp * probs_bkp).argmax(dim=-1).to(dtype=torch.long) + 1
        if allow_skip:
            current_penalty_bk = torch.where(
                skip_bk > 0.5,
                torch.zeros_like(current_penalty_bk),
                current_penalty_bk,
            )
        feature_chunks.append(class_features_bkqf.detach().cpu().reshape(-1, P + 1, class_features_bkqf.shape[-1]))
        label_chunks.append(labels_bk.detach().cpu().reshape(-1))
        current_chunks.append(current_penalty_bk.detach().cpu().reshape(-1))
        query_start_chunks.append(query_start_abs_b.detach().cpu().view(-1, 1).expand(-1, int(K)).reshape(-1))
        gain_chunks.append(gain_bk.detach().cpu().reshape(-1))

    if not feature_chunks:
        return None
    return {
        "split": str(split_name),
        "features": torch.cat(feature_chunks, dim=0),
        "labels": torch.cat(label_chunks, dim=0),
        "current_pred": torch.cat(current_chunks, dim=0),
        "query_start_abs": torch.cat(query_start_chunks, dim=0),
        "oracle_gain_mse": torch.cat(gain_chunks, dim=0),
        "label_names": ["skip"] + [str(name) for name in penalty_names],
        "feature_names": list(feature_names or []),
    }


def _penalty_route_learnability_metrics_from_scores(
    *,
    scores: torch.Tensor,
    labels: torch.Tensor,
    current_pred: torch.Tensor,
    label_names: List[str],
) -> Dict[str, object]:
    if scores.dim() != 2:
        raise ValueError("scores must have shape [N,num_classes].")
    label_count = int(scores.shape[1])
    if len(label_names) != label_count:
        raise ValueError("label_names length must match scores.shape[1].")
    labels = labels.detach().cpu().to(dtype=torch.long).view(-1)
    current_pred = current_pred.detach().cpu().to(dtype=torch.long).view(-1)
    if int(labels.numel()) != int(scores.shape[0]) or int(current_pred.numel()) != int(scores.shape[0]):
        raise ValueError("scores, labels, and current_pred must share N.")
    valid = (labels >= 0) & (labels < label_count)
    if int(valid.sum().item()) <= 0:
        return {
            "samples": 0,
            "accuracy_all": 0.0,
            "current_accuracy_all": 0.0,
            "majority_accuracy_all": 0.0,
            "accuracy_on_positive_oracle": 0.0,
            "current_accuracy_on_positive_oracle": 0.0,
            "prediction_counts": {name: 0 for name in label_names},
            "current_prediction_counts": {name: 0 for name in label_names},
            "label_counts": {name: 0 for name in label_names},
        }
    labels_v = labels[valid]
    current_v = current_pred[valid].clamp(0, label_count - 1)
    pred_v = scores.detach().cpu()[valid].argmax(dim=-1).to(dtype=torch.long)
    label_counts_t = torch.bincount(labels_v, minlength=label_count)[:label_count]
    pred_counts_t = torch.bincount(pred_v, minlength=label_count)[:label_count]
    current_counts_t = torch.bincount(current_v, minlength=label_count)[:label_count]
    samples = int(labels_v.numel())
    positive = labels_v > 0
    positive_count = int(positive.sum().item())
    majority_acc = float(label_counts_t.max().item() / max(samples, 1))
    accuracy = float((pred_v == labels_v).to(dtype=torch.float32).mean().item())
    current_accuracy = float((current_v == labels_v).to(dtype=torch.float32).mean().item())
    if positive_count > 0:
        positive_accuracy = float((pred_v[positive] == labels_v[positive]).to(dtype=torch.float32).mean().item())
        current_positive_accuracy = float(
            (current_v[positive] == labels_v[positive]).to(dtype=torch.float32).mean().item()
        )
    else:
        positive_accuracy = 0.0
        current_positive_accuracy = 0.0
    return {
        "samples": samples,
        "positive_oracle_samples": positive_count,
        "accuracy_all": accuracy,
        "current_accuracy_all": current_accuracy,
        "majority_accuracy_all": majority_acc,
        "lift_vs_current": float(accuracy - current_accuracy),
        "lift_vs_majority": float(accuracy - majority_acc),
        "accuracy_on_positive_oracle": positive_accuracy,
        "current_accuracy_on_positive_oracle": current_positive_accuracy,
        "prediction_counts": {name: int(pred_counts_t[i].item()) for i, name in enumerate(label_names)},
        "current_prediction_counts": {name: int(current_counts_t[i].item()) for i, name in enumerate(label_names)},
        "label_counts": {name: int(label_counts_t[i].item()) for i, name in enumerate(label_names)},
        "label_rates": {
            name: float(label_counts_t[i].item() / max(samples, 1)) for i, name in enumerate(label_names)
        },
    }


class _PenaltyRouteLearnabilityHead(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        hidden_dim: int = 0,
        dropout: float = 0.0,
        head_mode: str = "classwise",
    ):
        super().__init__()
        self.head_mode = str(head_mode or "classwise").lower()
        if self.head_mode in {"flat", "multiclass"}:
            in_dim = int(feat_dim) * int(num_classes)
            if int(hidden_dim) > 0:
                self.net = nn.Sequential(
                    nn.Linear(in_dim, int(hidden_dim)),
                    nn.ReLU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(int(hidden_dim), int(num_classes)),
                )
            else:
                self.net = nn.Linear(in_dim, int(num_classes))
            return
        if self.head_mode not in {"classwise", "candidate", "shared"}:
            raise ValueError("route learnability head_mode must be classwise or flat.")
        if int(hidden_dim) > 0:
            self.net = nn.Sequential(
                nn.Linear(int(feat_dim), int(hidden_dim)),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden_dim), 1),
            )
        else:
            self.net = nn.Linear(int(feat_dim), 1)

    def forward(self, features_nqf: torch.Tensor) -> torch.Tensor:
        if features_nqf.dim() != 3:
            raise ValueError("features_nqf must have shape [N,num_classes,F].")
        if self.head_mode in {"flat", "multiclass"}:
            return self.net(features_nqf.flatten(start_dim=1))
        return self.net(features_nqf).squeeze(-1)


def _fit_penalty_route_learnability_head_from_tensors(
    *,
    train_tensors: Dict[str, torch.Tensor],
    eval_tensors_by_split: Dict[str, Dict[str, torch.Tensor]],
    label_names: List[str],
    feature_names: List[str],
    cfg: Dict[str, object],
    device: torch.device,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    train_features = train_tensors["features"].detach().to(dtype=torch.float32)
    train_labels = train_tensors["labels"].detach().to(dtype=torch.long).view(-1)
    train_current = train_tensors["current_pred"].detach().to(dtype=torch.long).view(-1)
    if train_features.dim() != 3:
        raise ValueError("train features must have shape [N,num_classes,F].")
    if int(train_features.shape[0]) != int(train_labels.numel()):
        raise ValueError("train labels length must match train features N.")
    label_count = int(train_features.shape[1])
    if len(label_names) != label_count:
        raise ValueError("label_names length must match train feature class count.")
    valid = (train_labels >= 0) & (train_labels < label_count)
    if int(valid.sum().item()) <= 0:
        raise ValueError("penalty route learnability probe has no valid train labels.")
    train_features = train_features[valid]
    train_labels = train_labels[valid]
    train_current = train_current[valid]
    flat = train_features.reshape(-1, int(train_features.shape[-1]))
    feat_mean = flat.mean(dim=0)
    feat_std = flat.std(dim=0).clamp_min(1.0e-6)

    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    model = _PenaltyRouteLearnabilityHead(
        feat_dim=int(train_features.shape[-1]),
        num_classes=label_count,
        hidden_dim=int(cfg.get("hidden_dim", 32)),
        dropout=float(cfg.get("dropout", 0.0)),
        head_mode=str(cfg.get("head_mode", "classwise")),
    ).to(device)
    init_bias_mode = str(cfg.get("init_bias", "none")).lower()
    if init_bias_mode in {"train_prior", "label_prior", "prior"}:
        counts = torch.bincount(train_labels, minlength=label_count).to(dtype=torch.float32)
        counts = counts + float(cfg.get("prior_smoothing", 1.0e-6))
        prior_logits = torch.log(counts / counts.sum().clamp_min(1.0e-12)).to(device=device)
        with torch.no_grad():
            final_layer = model.net if isinstance(model.net, nn.Linear) else model.net[-1]
            if isinstance(final_layer, nn.Linear) and int(final_layer.out_features) == label_count:
                if bool(cfg.get("zero_init_output", True)):
                    final_layer.weight.zero_()
                final_layer.bias.copy_(prior_logits)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 1.0e-3)),
        weight_decay=float(cfg.get("weight_decay", 1.0e-4)),
    )
    batch_size = max(1, int(cfg.get("batch_size", 256)))
    epochs = max(1, int(cfg.get("epochs", 60)))
    class_weight_mode = str(cfg.get("class_weight", "none")).lower()
    class_weight = None
    class_weight_summary = None
    if class_weight_mode in {"balanced", "auto"}:
        counts = torch.bincount(train_labels, minlength=label_count).to(dtype=torch.float32).clamp_min(1.0)
        weight = float(train_labels.numel()) / (float(label_count) * counts)
        if "class_weight_min" in cfg or "class_weight_max" in cfg:
            weight = weight.clamp(
                min=float(cfg.get("class_weight_min", 0.0)),
                max=float(cfg.get("class_weight_max", float("inf"))),
            )
        class_weight = weight.to(device=device)
        class_weight_summary = [float(v) for v in weight.detach().cpu().tolist()]
    elif isinstance(cfg.get("class_weight"), (list, tuple)):
        weight = torch.as_tensor(cfg.get("class_weight"), dtype=torch.float32)
        if int(weight.numel()) != label_count:
            raise ValueError(f"route learnability class_weight must have {label_count} entries.")
        class_weight = weight.to(device=device)
        class_weight_summary = [float(v) for v in weight.detach().cpu().tolist()]

    def _standardize(features: torch.Tensor) -> torch.Tensor:
        return (features.to(device=device, dtype=torch.float32) - feat_mean.to(device=device).view(1, 1, -1)) / feat_std.to(
            device=device
        ).view(1, 1, -1)

    train_features_device = train_features.to(device=device)
    train_labels_device = train_labels.to(device=device)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_loss = float("inf")
    best_train_loss = float("inf")
    best_epoch = 0
    selection_split_requested = str(cfg.get("early_stop_split", cfg.get("selection_split", "train"))).lower()
    if selection_split_requested in {"", "none", "off", "false", "0"}:
        selection_split_requested = "train"
    selection_split = selection_split_requested
    selection_fallback_reason = None
    if selection_split != "train" and selection_split not in eval_tensors_by_split:
        selection_fallback_reason = f"selection split {selection_split!r} not available; fell back to train loss"
        selection_split = "train"
    selection_metric = str(cfg.get("selection_metric", cfg.get("early_stop_metric", "loss"))).lower()
    metric_alias = {
        "ce": "loss",
        "cross_entropy": "loss",
        "val_loss": "loss",
        "acc": "accuracy",
        "accuracy_all": "accuracy",
        "val_accuracy": "accuracy",
        "lift": "lift_vs_majority",
        "majority_lift": "lift_vs_majority",
    }
    selection_metric = metric_alias.get(selection_metric, selection_metric)
    if selection_metric not in {"loss", "accuracy", "lift_vs_majority"}:
        raise ValueError("route learnability selection_metric must be loss, accuracy, or lift_vs_majority.")
    minimize_selection = selection_metric == "loss"
    best_selection_value = float("inf") if minimize_selection else -float("inf")
    best_selection_loss = float("inf")
    best_selection_metrics: Dict[str, object] = {}
    early_stop_patience = int(cfg.get("early_stop_patience", cfg.get("patience", 0)))
    early_stop_min_delta = float(cfg.get("early_stop_min_delta", cfg.get("min_delta", 0.0)))
    include_initial_eval = bool(cfg.get("include_initial_eval", False))
    epochs_without_improvement = 0
    stopped_epoch = 0
    selection_history: List[Dict[str, object]] = []

    def _filtered_eval_tensors(tensors: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = tensors["features"].detach().to(dtype=torch.float32)
        labels = tensors["labels"].detach().to(dtype=torch.long).view(-1)
        current = tensors["current_pred"].detach().to(dtype=torch.long).view(-1)
        valid_eval = (labels >= 0) & (labels < label_count)
        return features[valid_eval], labels[valid_eval], current[valid_eval]

    def _loss_and_metrics_for_features(
        features: torch.Tensor,
        labels: torch.Tensor,
        current: torch.Tensor,
    ) -> Tuple[float, Dict[str, object]]:
        if int(labels.numel()) <= 0:
            metrics = _penalty_route_learnability_metrics_from_scores(
                scores=torch.zeros(0, label_count),
                labels=labels,
                current_pred=current,
                label_names=label_names,
            )
            return float("inf"), metrics
        with torch.no_grad():
            scores_device = model(_standardize(features))
            loss = nn.functional.cross_entropy(scores_device, labels.to(device=device), weight=None)
            scores = scores_device.detach().cpu()
        metrics = _penalty_route_learnability_metrics_from_scores(
            scores=scores,
            labels=labels,
            current_pred=current,
            label_names=label_names,
        )
        return float(loss.detach().item()), metrics

    train_selection_tensors = {
        "features": train_features.detach().cpu(),
        "labels": train_labels.detach().cpu(),
        "current_pred": train_current.detach().cpu(),
    }

    def _selection_value_for_epoch(epoch_train_loss: float) -> Tuple[float, float, Dict[str, object]]:
        if selection_split == "train":
            if selection_metric == "loss":
                return float(epoch_train_loss), float(epoch_train_loss), {}
            features, labels, current = _filtered_eval_tensors(train_selection_tensors)
        else:
            features, labels, current = _filtered_eval_tensors(eval_tensors_by_split[selection_split])
        eval_loss, metrics = _loss_and_metrics_for_features(features, labels, current)
        if selection_metric == "loss":
            value = eval_loss
        elif selection_metric == "accuracy":
            value = float(metrics.get("accuracy_all", 0.0))
        else:
            value = float(metrics.get("lift_vs_majority", 0.0))
        return float(value), float(eval_loss), metrics

    def _selection_improved(value: float) -> bool:
        if minimize_selection:
            return value < (best_selection_value - early_stop_min_delta)
        return value > (best_selection_value + early_stop_min_delta)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if include_initial_eval:
        model.eval()
        selection_value, selection_loss, selection_metrics = _selection_value_for_epoch(float("inf"))
        selection_history.append(
            {
                "epoch": 0,
                "train_loss": None,
                "selection_split": selection_split,
                "selection_metric": selection_metric,
                "selection_value": float(selection_value),
                "selection_loss": float(selection_loss),
            }
        )
        if _selection_improved(selection_value):
            best_epoch = 0
            best_selection_value = float(selection_value)
            best_selection_loss = float(selection_loss)
            best_selection_metrics = dict(selection_metrics)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for epoch in range(1, epochs + 1):
        order = torch.randperm(int(train_features.shape[0]), generator=generator)
        model.train()
        total_loss = 0.0
        total_seen = 0
        for start in range(0, int(order.numel()), batch_size):
            idx = order[start : start + batch_size].to(device=device)
            features = _standardize(train_features_device.index_select(0, idx))
            target = train_labels_device.index_select(0, idx)
            logits = model(features)
            loss = nn.functional.cross_entropy(logits, target, weight=class_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_n = int(target.numel())
            total_loss += float(loss.detach().item()) * batch_n
            total_seen += batch_n
        epoch_loss = total_loss / max(total_seen, 1)
        best_train_loss = min(best_train_loss, float(epoch_loss))
        model.eval()
        selection_value, selection_loss, selection_metrics = _selection_value_for_epoch(epoch_loss)
        selection_history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(epoch_loss),
                "selection_split": selection_split,
                "selection_metric": selection_metric,
                "selection_value": float(selection_value),
                "selection_loss": float(selection_loss),
            }
        )
        if _selection_improved(selection_value):
            best_loss = float(epoch_loss)
            best_epoch = int(epoch)
            best_selection_value = float(selection_value)
            best_selection_loss = float(selection_loss)
            best_selection_metrics = dict(selection_metrics)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
                stopped_epoch = int(epoch)
                break
    if best_epoch <= 0:
        best_epoch = 1
    model.load_state_dict(best_state, strict=True)
    model.eval()

    def _metrics_for(tensors: Dict[str, torch.Tensor]) -> Dict[str, object]:
        features, labels, current = _filtered_eval_tensors(tensors)
        if int(labels.numel()) <= 0:
            return _penalty_route_learnability_metrics_from_scores(
                scores=torch.zeros(0, label_count),
                labels=labels,
                current_pred=current,
                label_names=label_names,
            )
        with torch.no_grad():
            scores = model(_standardize(features)).detach().cpu()
        return _penalty_route_learnability_metrics_from_scores(
            scores=scores,
            labels=labels,
            current_pred=current,
            label_names=label_names,
        )

    split_metrics = {
        "train": _metrics_for(
            {
                "features": train_features.detach().cpu(),
                "labels": train_labels.detach().cpu(),
                "current_pred": train_current.detach().cpu(),
            }
        )
    }
    for split_name, tensors in eval_tensors_by_split.items():
        split_metrics[str(split_name)] = _metrics_for(tensors)

    summary = {
        "enable": True,
        "samples_train": int(train_labels.numel()),
        "label_names": list(label_names),
        "feature_names": list(feature_names),
        "config": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": float(cfg.get("lr", 1.0e-3)),
            "head_mode": str(cfg.get("head_mode", "classwise")).lower(),
            "hidden_dim": int(cfg.get("hidden_dim", 32)),
            "dropout": float(cfg.get("dropout", 0.0)),
            "weight_decay": float(cfg.get("weight_decay", 1.0e-4)),
            "class_weight": class_weight_mode,
            "class_weight_values": class_weight_summary,
            "class_weight_min": None if "class_weight_min" not in cfg else float(cfg.get("class_weight_min", 0.0)),
            "class_weight_max": None if "class_weight_max" not in cfg else float(cfg.get("class_weight_max", 0.0)),
            "selection_split": selection_split_requested,
            "selection_metric": selection_metric,
            "early_stop_patience": int(early_stop_patience),
            "early_stop_min_delta": float(early_stop_min_delta),
            "init_bias": init_bias_mode,
            "include_initial_eval": bool(include_initial_eval),
            "seed": int(seed),
        },
        "best_train_loss": float(best_train_loss),
        "selection": {
            "split": selection_split,
            "requested_split": selection_split_requested,
            "fallback_reason": selection_fallback_reason,
            "metric": selection_metric,
            "best_epoch": int(best_epoch),
            "best_value": float(best_selection_value),
            "best_loss": float(best_selection_loss),
            "best_train_epoch_loss": float(best_loss),
            "minimize": bool(minimize_selection),
            "patience": int(early_stop_patience),
            "min_delta": float(early_stop_min_delta),
            "stopped_epoch": int(stopped_epoch),
            "best_metrics": best_selection_metrics,
        },
        "selection_history": selection_history,
        "splits": split_metrics,
    }
    artifact = {
        "state_dict": best_state,
        "feature_mean": feat_mean.detach().cpu(),
        "feature_std": feat_std.detach().cpu(),
        "class_weight": None if class_weight is None else class_weight.detach().cpu(),
        "label_names": list(label_names),
        "feature_names": list(feature_names),
        "config": dict(summary["config"]),
        "selection": dict(summary["selection"]),
    }
    return summary, artifact


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
    history_anchor_cfg: Optional[dict] = None,
    observed_history_tc: Optional[torch.Tensor] = None,
    input_len: int = 0,
    eval_start: int = 0,
    model_train_stat_adapter_pc: Optional[torch.Tensor] = None,
    model_train_stat_adapter_cfg: Optional[dict] = None,
    train_stat_anchor_pc: Optional[torch.Tensor] = None,
    train_residual_anchor_phc: Optional[torch.Tensor] = None,
    gate_feature_mode: str = "history",
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
    gate_feature_mode = _normalize_gate_feature_mode(gate_feature_mode)
    explain_cfg = moe_cfg.get("explainability", {}) or {}
    utility_thresholds = [float(v) for v in (explain_cfg.get("utility_thresholds", []) or [])]
    route_label_phase_periods = [
        int(v) for v in (explain_cfg.get("route_label_phase_periods", []) or []) if int(v) > 0
    ]
    route_label_phase_bins = int(explain_cfg.get("route_label_phase_bins", 8))
    top1_confidence_bins = [float(v) for v in (explain_cfg.get("top1_confidence_bins", []) or [])]

    P = len(penalty_names)
    cid_c = cluster_id_c.detach().to(device=device, dtype=torch.long)
    cid_cpu = cluster_id_c.detach().cpu().to(dtype=torch.long)
    cluster_channel_count = torch.bincount(cid_cpu, minlength=K).clamp_min(1).to(dtype=torch.float32)
    allowed_mask_device = None
    if allowed_mask_kp is not None:
        allowed_mask_device = allowed_mask_kp.detach().to(device=device, dtype=torch.bool)
        if tuple(allowed_mask_device.shape) != (int(K), int(P)):
            raise ValueError("allowed_mask_kp must have shape [K,P] for explainability diagnostics.")

    total_bc_k = torch.zeros(K, dtype=torch.float64)
    base_err_sum_k = torch.zeros(K, dtype=torch.float64)
    final_err_sum_k = torch.zeros(K, dtype=torch.float64)
    oracle_err_sum_k = torch.zeros(K, dtype=torch.float64)
    cluster_penalty_oracle_err_sum_k = torch.zeros(K, dtype=torch.float64)
    cluster_route_oracle_err_sum_k = torch.zeros(K, dtype=torch.float64)
    cluster_route_oracle_skip_count_k = torch.zeros(K, dtype=torch.float64)
    cluster_route_oracle_decision_count_k = torch.zeros(K, dtype=torch.float64)
    fusion_sum_k = torch.zeros(K, dtype=torch.float64)
    skip_count_k = torch.zeros(K, dtype=torch.float64)
    skip_on_oracle_positive_count_k = torch.zeros(K, dtype=torch.float64)
    selected_count_kp = torch.zeros(K, P, dtype=torch.float64)
    top1_intended_count_kp = torch.zeros(K, P, dtype=torch.float64)
    top1_selected_count_kp = torch.zeros(K, P, dtype=torch.float64)
    top1_selected_positive_count_kp = torch.zeros(K, P, dtype=torch.float64)
    top1_selected_gain_sum_kp = torch.zeros(K, P, dtype=torch.float64)
    skipped_top1_count_kp = torch.zeros(K, P, dtype=torch.float64)
    skipped_on_oracle_positive_count_kp = torch.zeros(K, P, dtype=torch.float64)
    harmful_top1_not_skipped_count_kp = torch.zeros(K, P, dtype=torch.float64)
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
    utility_valid_count_kt = torch.zeros(K, len(utility_thresholds), dtype=torch.float64)
    utility_total_count_k = torch.zeros(K, dtype=torch.float64)
    utility_best_gain_sum_k = torch.zeros(K, dtype=torch.float64)
    utility_best_gain_positive_count_k = torch.zeros(K, dtype=torch.float64)

    total_decisions = 0
    total_selected = 0
    total_oracle_positive = 0
    total_selected_positive = 0
    batch_count = 0
    route_label_feature_chunks: List[torch.Tensor] = []
    route_label_chunks: List[torch.Tensor] = []
    route_label_query_start_chunks: List[torch.Tensor] = []
    top1_conf_chunks: List[torch.Tensor] = []
    top1_gain_chunks: List[torch.Tensor] = []
    top1_penalty_chunks: List[torch.Tensor] = []
    top1_active_chunks: List[torch.Tensor] = []
    top1_skip_chunks: List[torch.Tensor] = []

    for x, y, idx in loader:
        batch_count += 1
        if max_batches > 0 and batch_count > int(max_batches):
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        idx = idx.to(device=device, dtype=torch.long)
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
        feat_bkf = _build_gate_routing_features(x, y_base, cluster_id_c, K, mode=gate_feature_mode)
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
            query_start_abs_b=query_start_abs_b,
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
        y_final_raw = pred_out["y_final"]
        y_final = apply_moe_output_anchor_experts(
            y_final_raw,
            base_pred_bch=y_base,
            x_bcl=x,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len),
            moe_cfg=moe_cfg,
            moe_enable=moe_enable,
            observed_history_tc=observed_history_tc,
            train_stat_anchor_pc=train_stat_anchor_pc,
            train_residual_anchor_phc=train_residual_anchor_phc,
        )

        y_base_final, cand_bcpH = _pred_residual_candidates_on_eval_path(
            y_base,
            pred_out,
            apply_output_anchors=True,
            x_bcl=x,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len),
            moe_cfg=moe_cfg,
            moe_enable=moe_enable,
            observed_history_tc=observed_history_tc,
            train_stat_anchor_pc=train_stat_anchor_pc,
            train_residual_anchor_phc=train_residual_anchor_phc,
        )
        if cand_bcpH is None:
            continue
        base_err_bc = (y_base_final - y).pow(2).mean(dim=-1)
        final_err_bc = (y_final - y).pow(2).mean(dim=-1)
        cand_err_bcp = (cand_bcpH - y.unsqueeze(2)).pow(2).mean(dim=-1)
        gain_bcp = base_err_bc.unsqueeze(-1) - cand_err_bcp
        if utility_thresholds:
            utility_stats = _cluster_utility_threshold_stats(
                gain_bcp=gain_bcp.detach(),
                cluster_id_c=cluster_id_c,
                K=K,
                allowed_mask_kp=allowed_mask_kp,
                thresholds=utility_thresholds,
            )
            utility_valid_count_kt += utility_stats["valid_count_kt"]
            utility_total_count_k += utility_stats["total_count_k"]
            utility_best_gain_sum_k += utility_stats["best_gain_sum_k"]
            utility_best_gain_positive_count_k += utility_stats["best_gain_positive_count_k"]
        if allowed_mask_device is not None:
            allowed_bcp = allowed_mask_device.index_select(0, cid_c).unsqueeze(0)
            cand_err_for_oracle_bcp = cand_err_bcp.masked_fill(~allowed_bcp, float("inf"))
        else:
            cand_err_for_oracle_bcp = cand_err_bcp
        oracle_penalty_err_bc, oracle_p_bc = cand_err_for_oracle_bcp.min(dim=-1)
        oracle_err_bc = torch.where(torch.isfinite(oracle_penalty_err_bc), oracle_penalty_err_bc, base_err_bc)
        oracle_gain_bc = base_err_bc - oracle_err_bc
        selected_bool_bcp = route_bcp > 0
        selected_positive_bcp = selected_bool_bcp & (gain_bcp > 0)
        positive_oracle_bc = oracle_gain_bc > 0
        raw_route_bcp = mask_bkp[:, cid_c, :]
        raw_score_bcp = raw_route_bcp * probs_bkp[:, cid_c, :]
        top1_p_bc = raw_score_bcp.argmax(dim=-1)
        top1_conf_bc = probs_bkp[:, cid_c, :].gather(-1, top1_p_bc.unsqueeze(-1)).squeeze(-1)
        top1_weight_bc = raw_route_bcp.gather(-1, top1_p_bc.unsqueeze(-1)).squeeze(-1)
        skip_bc = (skip_bk[:, cid_c] > 0.5) if allow_skip else torch.zeros_like(base_err_bc, dtype=torch.bool)
        top1_active_bc = (top1_weight_bc > 0) & (~skip_bc)
        top1_gain_bc = gain_bcp.gather(-1, top1_p_bc.unsqueeze(-1)).squeeze(-1)
        top1_positive_bc = top1_active_bc & (top1_gain_bc > 0)
        top1_harmful_bc = top1_active_bc & (top1_gain_bc <= 0)

        total_decisions += int(base_err_bc.numel())
        total_selected += int(selected_bool_bcp.sum().item())
        total_oracle_positive += int(positive_oracle_bc.sum().item())
        total_selected_positive += int(selected_positive_bcp.sum().item())

        route_label_bk = torch.full((base_err_bc.shape[0], K), -1, device=device, dtype=torch.long)
        for k in range(K):
            ch_mask = cid_c == int(k)
            if not bool(ch_mask.any().item()):
                continue
            base_k = base_err_bc[:, ch_mask]
            final_k = final_err_bc[:, ch_mask]
            cand_err_k = cand_err_bcp[:, ch_mask, :]
            total_k = int(base_k.numel())
            channels_k = int(ch_mask.sum().item())
            total_bc_k[k] += total_k
            base_err_sum_k[k] += float(base_k.sum().item())
            final_err_sum_k[k] += float(final_k.sum().item())
            oracle_err_sum_k[k] += float(oracle_err_bc[:, ch_mask].sum().item())
            cluster_base_err_b = base_k.mean(dim=1)
            cluster_penalty_err_bp = cand_err_k.mean(dim=1)
            if allowed_mask_device is not None:
                allowed_p = allowed_mask_device[int(k)].view(1, P)
                cluster_penalty_err_bp = cluster_penalty_err_bp.masked_fill(~allowed_p, float("inf"))
            best_cluster_penalty_err_b, best_cluster_penalty_p_b = cluster_penalty_err_bp.min(dim=-1)
            has_allowed_penalty_b = torch.isfinite(best_cluster_penalty_err_b)
            best_cluster_penalty_err_for_sum_b = torch.where(
                has_allowed_penalty_b,
                best_cluster_penalty_err_b,
                cluster_base_err_b,
            )
            best_cluster_route_err_b = torch.minimum(cluster_base_err_b, best_cluster_penalty_err_for_sum_b)
            cluster_route_label_b = torch.where(
                cluster_base_err_b <= best_cluster_penalty_err_for_sum_b,
                torch.zeros_like(best_cluster_penalty_p_b),
                best_cluster_penalty_p_b + 1,
            )
            cluster_route_label_b = torch.where(
                has_allowed_penalty_b,
                cluster_route_label_b,
                torch.zeros_like(cluster_route_label_b),
            )
            route_label_bk[:, int(k)] = cluster_route_label_b
            cluster_penalty_oracle_err_sum_k[k] += float(best_cluster_penalty_err_for_sum_b.sum().item() * channels_k)
            cluster_route_oracle_err_sum_k[k] += float(best_cluster_route_err_b.sum().item() * channels_k)
            cluster_route_oracle_skip_count_k[k] += float((cluster_route_label_b == 0).sum().item())
            cluster_route_oracle_decision_count_k[k] += int(cluster_base_err_b.numel())
            fusion_sum_k[k] += float(fusion_bc[:, ch_mask].sum().item())
            skip_count_k[k] += float(skip_bc[:, ch_mask].sum().item())
            skip_on_oracle_positive_count_k[k] += float((skip_bc[:, ch_mask] & positive_oracle_bc[:, ch_mask]).sum().item())

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
            top1_k = top1_p_bc[:, ch_mask].detach().cpu().reshape(-1)
            top1_active_k = top1_p_bc[:, ch_mask][top1_active_bc[:, ch_mask]].detach().cpu().reshape(-1)
            top1_positive_k = top1_p_bc[:, ch_mask][top1_positive_bc[:, ch_mask]].detach().cpu().reshape(-1)
            top1_harmful_k = top1_p_bc[:, ch_mask][top1_harmful_bc[:, ch_mask]].detach().cpu().reshape(-1)
            skipped_top1_k = top1_p_bc[:, ch_mask][skip_bc[:, ch_mask]].detach().cpu().reshape(-1)
            skipped_positive_k = top1_p_bc[:, ch_mask][skip_bc[:, ch_mask] & positive_oracle_bc[:, ch_mask]].detach().cpu().reshape(-1)
            oracle_count_kp[k] += torch.bincount(oracle_k, minlength=P)[:P].to(dtype=torch.float64)
            if oracle_pos_k.numel() > 0:
                positive_oracle_count_kp[k] += torch.bincount(oracle_pos_k, minlength=P)[:P].to(dtype=torch.float64)
            top1_intended_count_kp[k] += torch.bincount(top1_k, minlength=P)[:P].to(dtype=torch.float64)
            if top1_active_k.numel() > 0:
                top1_selected_count_kp[k] += torch.bincount(top1_active_k, minlength=P)[:P].to(dtype=torch.float64)
            if top1_positive_k.numel() > 0:
                top1_selected_positive_count_kp[k] += torch.bincount(top1_positive_k, minlength=P)[:P].to(dtype=torch.float64)
            if top1_harmful_k.numel() > 0:
                harmful_top1_not_skipped_count_kp[k] += torch.bincount(top1_harmful_k, minlength=P)[:P].to(dtype=torch.float64)
            if skipped_top1_k.numel() > 0:
                skipped_top1_count_kp[k] += torch.bincount(skipped_top1_k, minlength=P)[:P].to(dtype=torch.float64)
            if skipped_positive_k.numel() > 0:
                skipped_on_oracle_positive_count_kp[k] += torch.bincount(skipped_positive_k, minlength=P)[:P].to(dtype=torch.float64)
            active_gain_k = top1_gain_bc[:, ch_mask] * top1_active_bc[:, ch_mask].to(dtype=top1_gain_bc.dtype)
            for p in range(P):
                top1_selected_gain_sum_kp[k, p] += float(
                    active_gain_k[top1_p_bc[:, ch_mask] == int(p)].sum().item()
                )
        route_label_feature_chunks.append(feat_bkf.detach().cpu())
        route_label_chunks.append(route_label_bk.detach().cpu())
        route_label_query_start_chunks.append(query_start_abs_b.detach().cpu())
        if top1_confidence_bins:
            top1_conf_chunks.append(top1_conf_bc.detach().cpu())
            top1_gain_chunks.append(top1_gain_bc.detach().cpu())
            top1_penalty_chunks.append(top1_p_bc.detach().cpu())
            top1_active_chunks.append(top1_active_bc.detach().cpu())
            top1_skip_chunks.append(skip_bc.detach().cpu())

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
        oracle_mse_k = float(oracle_err_sum_k[k].item() / max(cluster_total, 1.0))
        cluster_penalty_oracle_mse_k = float(cluster_penalty_oracle_err_sum_k[k].item() / max(cluster_total, 1.0))
        cluster_route_oracle_mse_k = float(cluster_route_oracle_err_sum_k[k].item() / max(cluster_total, 1.0))
        cluster_gain = float(100.0 * (base_mse_k - final_mse_k) / max(abs(base_mse_k), 1.0e-12))
        oracle_gain_pct = float(100.0 * (base_mse_k - oracle_mse_k) / max(abs(base_mse_k), 1.0e-12))
        cluster_penalty_oracle_gain_pct = float(
            100.0 * (base_mse_k - cluster_penalty_oracle_mse_k) / max(abs(base_mse_k), 1.0e-12)
        )
        cluster_route_oracle_gain_pct = float(
            100.0 * (base_mse_k - cluster_route_oracle_mse_k) / max(abs(base_mse_k), 1.0e-12)
        )
        cluster_route_oracle_skip_rate = float(
            cluster_route_oracle_skip_count_k[k].item() / max(cluster_route_oracle_decision_count_k[k].item(), 1.0)
        )
        skip_rate = float(skip_count_k[k].item() / max(cluster_total, 1.0))
        skip_on_positive_rate = float(skip_on_oracle_positive_count_k[k].item() / max(total_bc_k[k].item(), 1.0))
        cluster_rows = []
        for p, name in enumerate(penalty_names):
            selected_count = float(selected_count_kp[k, p].item())
            selected_gain_count = float(selected_gain_count_kp[k, p].item())
            top1_selected_count = float(top1_selected_count_kp[k, p].item())
            mean_gain = float(mean_gain_kp[k, p].item())
            selected_mean_gain = float(selected_gain_sum_kp[k, p].item() / max(selected_gain_count, 1.0))
            top1_selected_mean_gain = float(top1_selected_gain_sum_kp[k, p].item() / max(top1_selected_count, 1.0))
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
                "top1_intended_count": int(top1_intended_count_kp[k, p].item()),
                "top1_intended_rate": float(top1_intended_count_kp[k, p].item() / max(cluster_total, 1.0)),
                "top1_selected_count": int(top1_selected_count),
                "top1_selected_rate": float(top1_selected_count / max(cluster_total, 1.0)),
                "top1_selected_positive_count": int(top1_selected_positive_count_kp[k, p].item()),
                "top1_selected_positive_rate": float(
                    top1_selected_positive_count_kp[k, p].item() / max(top1_selected_count, 1.0)
                ),
                "top1_selected_mean_gain_mse": top1_selected_mean_gain,
                "harmful_top1_not_skipped_count": int(harmful_top1_not_skipped_count_kp[k, p].item()),
                "harmful_top1_not_skipped_rate": float(
                    harmful_top1_not_skipped_count_kp[k, p].item() / max(top1_selected_count, 1.0)
                ),
                "skipped_top1_count": int(skipped_top1_count_kp[k, p].item()),
                "skipped_top1_rate": float(skipped_top1_count_kp[k, p].item() / max(cluster_total, 1.0)),
                "skipped_on_oracle_positive_count": int(skipped_on_oracle_positive_count_kp[k, p].item()),
                "skipped_on_oracle_positive_rate": float(
                    skipped_on_oracle_positive_count_kp[k, p].item() / max(cluster_total, 1.0)
                ),
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
                "cluster_oracle_mse": oracle_mse_k,
                "cluster_penalty_oracle_mse": cluster_penalty_oracle_mse_k,
                "cluster_penalty_oracle_gain_pct_vs_base": cluster_penalty_oracle_gain_pct,
                "cluster_route_oracle_mse": cluster_route_oracle_mse_k,
                "cluster_route_oracle_gain_pct_vs_base": cluster_route_oracle_gain_pct,
                "cluster_route_oracle_skip_rate": cluster_route_oracle_skip_rate,
                "cluster_final_gain_pct": cluster_gain,
                "cluster_oracle_gain_pct_vs_base": oracle_gain_pct,
                "cluster_skip_rate": skip_rate,
                "cluster_skip_on_oracle_positive_rate": skip_on_positive_rate,
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
                "oracle_mse": oracle_mse_k,
                "cluster_penalty_oracle_mse": cluster_penalty_oracle_mse_k,
                "cluster_penalty_oracle_gain_pct_vs_base": cluster_penalty_oracle_gain_pct,
                "cluster_route_oracle_mse": cluster_route_oracle_mse_k,
                "cluster_route_oracle_gain_pct_vs_base": cluster_route_oracle_gain_pct,
                "cluster_route_oracle_skip_rate": cluster_route_oracle_skip_rate,
                "final_gain_pct": cluster_gain,
                "oracle_gain_pct_vs_base": oracle_gain_pct,
                "skip_count": int(skip_count_k[k].item()),
                "skip_rate": skip_rate,
                "skip_on_oracle_positive_count": int(skip_on_oracle_positive_count_k[k].item()),
                "skip_on_oracle_positive_rate": skip_on_positive_rate,
                "top_penalties": [
                    {
                        "penalty": item["penalty"],
                        "allowed_by_train_prior": item["allowed_by_train_prior"],
                        "train_prior_prob": item["train_prior_prob"],
                        "mean_single_penalty_gain_mse": item["mean_single_penalty_gain_mse"],
                        "selected_rate": item["selected_rate"],
                        "top1_selected_positive_rate": item["top1_selected_positive_rate"],
                        "harmful_top1_not_skipped_rate": item["harmful_top1_not_skipped_rate"],
                        "skipped_top1_rate": item["skipped_top1_rate"],
                        "reason": item["reason"],
                    }
                    for item in cluster_rows_sorted[: min(3, len(cluster_rows_sorted))]
                ],
            }
        )

    base_mse = float(base_err_sum_k.sum().item() / max(total_decisions, 1))
    final_mse = float(final_err_sum_k.sum().item() / max(total_decisions, 1))
    oracle_mse = float(oracle_err_sum_k.sum().item() / max(total_decisions, 1))
    cluster_penalty_oracle_mse = float(cluster_penalty_oracle_err_sum_k.sum().item() / max(total_decisions, 1))
    cluster_route_oracle_mse = float(cluster_route_oracle_err_sum_k.sum().item() / max(total_decisions, 1))
    cluster_route_oracle_decisions = float(cluster_route_oracle_decision_count_k.sum().item())
    cluster_route_oracle_skip_rate = float(
        cluster_route_oracle_skip_count_k.sum().item() / max(cluster_route_oracle_decisions, 1.0)
    )
    utility_threshold_summary = []
    if utility_thresholds:
        for k in range(K):
            total_k = float(utility_total_count_k[k].item())
            utility_threshold_summary.append(
                {
                    "cluster_id": int(k),
                    "windows": int(total_k),
                    "mean_best_allowed_gain_mse": float(utility_best_gain_sum_k[k].item() / max(total_k, 1.0)),
                    "positive_best_allowed_gain_rate": float(
                        utility_best_gain_positive_count_k[k].item() / max(total_k, 1.0)
                    ),
                    "thresholds": [
                        {
                            "min_gain": float(threshold),
                            "valid_count": int(utility_valid_count_kt[k, t].item()),
                            "valid_rate": float(utility_valid_count_kt[k, t].item() / max(total_k, 1.0)),
                        }
                        for t, threshold in enumerate(utility_thresholds)
                    ],
                }
            )
    route_label_feature_diagnostics = None
    if route_label_feature_chunks and route_label_chunks:
        route_label_feature_diagnostics = _cluster_route_label_feature_diagnostics(
            feat_bkf=torch.cat(route_label_feature_chunks, dim=0),
            route_label_bk=torch.cat(route_label_chunks, dim=0),
            penalty_names=penalty_names,
            feature_names=_gate_feature_names_for_mode(gate_feature_mode),
        )
    route_label_phase_diagnostics = None
    if route_label_phase_periods and route_label_chunks and route_label_query_start_chunks:
        route_label_phase_diagnostics = _cluster_route_label_phase_diagnostics(
            query_start_abs_b=torch.cat(route_label_query_start_chunks, dim=0),
            route_label_bk=torch.cat(route_label_chunks, dim=0),
            penalty_names=penalty_names,
            periods=route_label_phase_periods,
            num_bins=route_label_phase_bins,
            phase_offset=int(input_len),
        )
    top1_confidence_gain_diagnostics = None
    if top1_confidence_bins and top1_conf_chunks:
        top1_confidence_gain_diagnostics = _cluster_top1_confidence_gain_diagnostics(
            top1_conf_bc=torch.cat(top1_conf_chunks, dim=0),
            top1_gain_bc=torch.cat(top1_gain_chunks, dim=0),
            top1_p_bc=torch.cat(top1_penalty_chunks, dim=0),
            top1_active_bc=torch.cat(top1_active_chunks, dim=0),
            skip_bc=torch.cat(top1_skip_chunks, dim=0),
            cluster_id_c=cluster_id_c.detach().cpu(),
            K=K,
            penalty_names=penalty_names,
            bins=top1_confidence_bins,
        )
    return {
        "split": split_name,
        "samples": int(total_decisions),
        "selected_penalty_events": int(total_selected),
        "oracle_positive_events": int(total_oracle_positive),
        "selected_positive_events": int(total_selected_positive),
        "base_mse": base_mse,
        "final_mse": final_mse,
        "final_gain_pct_vs_base": float(100.0 * (base_mse - final_mse) / max(abs(base_mse), 1.0e-12)),
        "oracle_mse": oracle_mse,
        "oracle_gain_pct_vs_base": float(100.0 * (base_mse - oracle_mse) / max(abs(base_mse), 1.0e-12)),
        "cluster_penalty_oracle_mse": cluster_penalty_oracle_mse,
        "cluster_penalty_oracle_gain_pct_vs_base": float(
            100.0 * (base_mse - cluster_penalty_oracle_mse) / max(abs(base_mse), 1.0e-12)
        ),
        "cluster_route_oracle_mse": cluster_route_oracle_mse,
        "cluster_route_oracle_gain_pct_vs_base": float(
            100.0 * (base_mse - cluster_route_oracle_mse) / max(abs(base_mse), 1.0e-12)
        ),
        "cluster_route_oracle_skip_rate": cluster_route_oracle_skip_rate,
        "prior_actual_gain_corr": penalty_corr,
        "utility_target_thresholds": utility_threshold_summary,
        "route_label_feature_diagnostics": route_label_feature_diagnostics,
        "route_label_phase_diagnostics": route_label_phase_diagnostics,
        "top1_confidence_gain_diagnostics": top1_confidence_gain_diagnostics,
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
    model_train_stat_adapter_pc: Optional[torch.Tensor] = None,
    model_train_stat_adapter_cfg: Optional[dict] = None,
    train_stat_anchor_pc: Optional[torch.Tensor] = None,
    train_residual_anchor_phc: Optional[torch.Tensor] = None,
    candidate_feature_mode: str = "base",
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
    confidence_parts = []
    y_parts = []
    P = int(penalty_count)
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        idx = idx.to(device=device, dtype=torch.long)
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
        mask_all_bkp = torch.ones(x.shape[0], int(K), P, device=device, dtype=y_base.dtype)
        pred_out = pred_residual(
            x,
            y_base,
            cluster_id_c,
            mask_all_bkp,
            skip_bk=None,
            query_start_abs_b=query_start_abs_b,
        )
        y_base_final, cand_bcpH = _pred_residual_candidates_on_eval_path(
            y_base,
            pred_out,
            pred_residual_scale_c=pred_residual_scale_c,
            apply_output_anchors=True,
            x_bcl=x,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len),
            moe_cfg=moe_cfg,
            moe_enable=True,
            observed_history_tc=observed_history_tc,
            train_stat_anchor_pc=train_stat_anchor_pc,
            train_residual_anchor_phc=train_residual_anchor_phc,
        )
        if cand_bcpH is None:
            continue
        skip_feat = _candidate_selector_features(x, y_base_final, y_base_final, feature_mode=candidate_feature_mode)
        cand_feat = torch.stack(
            [
                _candidate_selector_features(
                    x,
                    y_base_final,
                    cand_bcpH[:, :, p, :],
                    feature_mode=candidate_feature_mode,
                )
                for p in range(P)
            ],
            dim=2,
        )
        skip_feat_parts.append(skip_feat.detach().cpu())
        cand_feat_parts.append(cand_feat.detach().cpu())
        base_parts.append(y_base_final.detach().cpu())
        cand_parts.append(cand_bcpH.detach().cpu())
        confidence_parts.append(
            pred_out.get("intervention_bcp", torch.ones_like(cand_bcpH[..., 0])).detach().cpu()
        )
        y_parts.append(y.detach().cpu())

    if len(skip_feat_parts) == 0:
        return None
    return {
        "skip_feat": torch.cat(skip_feat_parts, dim=0),
        "cand_feat": torch.cat(cand_feat_parts, dim=0),
        "base": torch.cat(base_parts, dim=0),
        "cand": torch.cat(cand_parts, dim=0),
        "confidence": torch.cat(confidence_parts, dim=0),
        "y": torch.cat(y_parts, dim=0),
    }


@torch.no_grad()
def _select_pred_residual_confidence_thresholds_from_tensors(
    *,
    tensors: Optional[Dict[str, torch.Tensor]],
    cluster_id_c: torch.Tensor,
    K: int,
    allowed_mask_kp: Optional[torch.Tensor] = None,
    penalty_names: Optional[List[str]] = None,
    min_abs_improvement: float = 0.0,
    min_rel_improvement: float = 0.0,
    max_candidates: int = 101,
    selection_metric: str = "mse",
    min_precision: float = 0.0,
    max_pred_positive_rate: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    if tensors is None:
        raise ValueError("confidence threshold selection requires collected train tensors.")
    base = tensors.get("base")
    cand = tensors.get("cand")
    y = tensors.get("y")
    confidence = tensors.get("confidence")
    if base is None or cand is None or y is None or confidence is None:
        raise ValueError("confidence threshold selection requires base, cand, y, and confidence tensors.")
    if cand.ndim != 4:
        raise ValueError(f"candidate tensor must have shape [B,C,P,H], got {tuple(cand.shape)}")
    B, C, P, _ = cand.shape
    if base.shape[:2] != (B, C) or y.shape[:2] != (B, C) or confidence.shape != (B, C, P):
        raise ValueError(
            "confidence threshold tensors have inconsistent shapes: "
            f"base={tuple(base.shape)}, cand={tuple(cand.shape)}, "
            f"y={tuple(y.shape)}, confidence={tuple(confidence.shape)}"
        )
    cluster_id = cluster_id_c.detach().cpu().to(dtype=torch.long).view(-1)
    if int(cluster_id.numel()) != int(C):
        raise ValueError(f"cluster_id_c must have one entry per channel, got {int(cluster_id.numel())} vs {int(C)}")
    K = int(K)
    if K <= 0:
        raise ValueError("K must be positive for confidence threshold selection.")
    metric_mode = str(selection_metric).lower()
    if metric_mode not in {"mse", "precision_guarded_mse"}:
        raise ValueError(
            "confidence threshold selection currently supports selection_metric='mse' "
            "or 'precision_guarded_mse'."
        )

    if allowed_mask_kp is None:
        allowed = torch.ones(K, P, dtype=torch.bool)
    else:
        if allowed_mask_kp.shape != (K, P):
            raise ValueError(f"allowed_mask_kp must have shape [{K},{P}], got {tuple(allowed_mask_kp.shape)}")
        allowed = allowed_mask_kp.detach().cpu().to(dtype=torch.float32) > 0.0

    names = list(penalty_names or [f"p{p}" for p in range(P)])
    if len(names) != P:
        names = [f"p{p}" for p in range(P)]

    min_abs = max(0.0, float(min_abs_improvement))
    min_rel = max(0.0, float(min_rel_improvement))
    min_precision = max(0.0, float(min_precision))
    max_rate = None
    if max_pred_positive_rate is not None:
        max_rate = min(1.0, max(0.0, float(max_pred_positive_rate)))
    max_candidates = max(2, int(max_candidates))
    base_err_bc = (base - y).pow(2).mean(dim=-1).detach().cpu()
    cand_err_bcp = (cand - y.unsqueeze(2)).pow(2).mean(dim=-1).detach().cpu()
    confidence_bcp = confidence.detach().cpu().to(dtype=torch.float32)
    thresholds = torch.full((K, P), 1.000001, dtype=torch.float32)
    per_cluster_penalty: Dict[str, Dict[str, object]] = {}

    for k in range(K):
        channel_mask_c = cluster_id == int(k)
        per_penalty: Dict[str, object] = {}
        for p in range(P):
            name = str(names[p])
            if not bool(allowed[k, p].item()):
                per_penalty[name] = {
                    "allowed": False,
                    "threshold": float(thresholds[k, p].item()),
                    "samples": 0,
                    "reason": "disallowed",
                }
                continue
            if not bool(channel_mask_c.any().item()):
                per_penalty[name] = {
                    "allowed": True,
                    "threshold": float(thresholds[k, p].item()),
                    "samples": 0,
                    "reason": "empty_cluster",
                }
                continue

            score_v = confidence_bcp[:, channel_mask_c, p].reshape(-1)
            base_err_v = base_err_bc[:, channel_mask_c].reshape(-1)
            cand_err_v = cand_err_bcp[:, channel_mask_c, p].reshape(-1)
            finite = torch.isfinite(score_v) & torch.isfinite(base_err_v) & torch.isfinite(cand_err_v)
            score_v = score_v[finite]
            base_err_v = base_err_v[finite]
            cand_err_v = cand_err_v[finite]
            if int(score_v.numel()) == 0:
                per_penalty[name] = {
                    "allowed": True,
                    "threshold": float(thresholds[k, p].item()),
                    "samples": 0,
                    "reason": "empty_scores",
                }
                continue

            gain_v = base_err_v - cand_err_v
            required_v = torch.maximum(
                torch.full_like(gain_v, min_abs),
                min_rel * base_err_v.abs().clamp_min(1.0e-12),
            )
            positive_v = gain_v > required_v
            if int(positive_v.sum().item()) == 0:
                threshold = float(score_v.max().item()) + 1.0e-6
                thresholds[k, p] = float(threshold)
                per_penalty[name] = {
                    "allowed": True,
                    "threshold": threshold,
                    "samples": int(score_v.numel()),
                    "reason": "no_positive_gain_labels",
                    "positive_rate": 0.0,
                    "base_mse": float(base_err_v.mean().item()),
                    "selected_mse": float(base_err_v.mean().item()),
                    "selected_gain_pct_vs_base": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "pred_positive_rate": 0.0,
                }
                continue

            uniq = torch.unique(score_v)
            if int(uniq.numel()) > max_candidates:
                quantiles = torch.linspace(0.0, 1.0, steps=max_candidates, dtype=score_v.dtype)
                candidates_v = torch.unique(torch.quantile(score_v, quantiles))
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
            base_mse = float(base_err_v.mean().item())
            skip_threshold = float(score_v.max().item()) + 1.0e-6
            best = {
                "threshold": skip_threshold,
                "mse": base_mse,
                "precision": 0.0,
                "recall": 0.0,
                "pred_positive_rate": 0.0,
                "reason": "no_improving_threshold",
                "skip_selected": True,
            }
            feasible_count = 0
            for candidate_threshold in candidates_v.tolist():
                pred_active = score_v >= float(candidate_threshold)
                selected_err = torch.where(pred_active, cand_err_v, base_err_v)
                selected_mse = float(selected_err.mean().item())
                tp = int((pred_active & positive_v).sum().item())
                fp = int((pred_active & (~positive_v)).sum().item())
                fn = int(((~pred_active) & positive_v).sum().item())
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                pred_rate = float(pred_active.to(dtype=torch.float32).mean().item())
                if metric_mode == "precision_guarded_mse":
                    if precision + 1.0e-12 < min_precision:
                        continue
                    if max_rate is not None and pred_rate > float(max_rate) + 1.0e-12:
                        continue
                feasible_count += 1
                better = selected_mse < float(best["mse"]) - 1.0e-12
                if not better and abs(selected_mse - float(best["mse"])) <= 1.0e-12:
                    better = precision > float(best["precision"]) + 1.0e-12
                if not better and abs(selected_mse - float(best["mse"])) <= 1.0e-12 and abs(precision - float(best["precision"])) <= 1.0e-12:
                    better = float(candidate_threshold) > float(best["threshold"])
                if better:
                    best = {
                        "threshold": float(candidate_threshold),
                        "mse": selected_mse,
                        "precision": float(precision),
                        "recall": float(recall),
                        "pred_positive_rate": pred_rate,
                        "reason": "selected",
                        "skip_selected": False,
                    }
            if metric_mode == "precision_guarded_mse" and feasible_count == 0:
                best["reason"] = "no_threshold_meets_confidence_guard"
            thresholds[k, p] = float(best["threshold"])
            per_penalty[name] = {
                "allowed": True,
                "threshold": float(best["threshold"]),
                "samples": int(score_v.numel()),
                "feasible_threshold_count": int(feasible_count),
                "reason": str(best["reason"]),
                "skip_selected": bool(best["skip_selected"]),
                "positive_rate": float(positive_v.to(dtype=torch.float32).mean().item()),
                "base_mse": base_mse,
                "selected_mse": float(best["mse"]),
                "selected_gain_pct_vs_base": float(
                    100.0 * (base_mse - float(best["mse"])) / max(abs(base_mse), 1.0e-12)
                ),
                "precision": float(best["precision"]),
                "recall": float(best["recall"]),
                "pred_positive_rate": float(best["pred_positive_rate"]),
            }
        per_cluster_penalty[str(k)] = per_penalty

    summary = {
        "enable": True,
        "source_requirement": "train_only",
        "selection_metric": metric_mode,
        "min_abs_improvement": float(min_abs),
        "min_rel_improvement": float(min_rel),
        "min_precision": float(min_precision),
        "max_pred_positive_rate": None if max_rate is None else float(max_rate),
        "threshold_kp": [[float(v) for v in row] for row in thresholds.tolist()],
        "penalty_names": names,
        "per_cluster_penalty": per_cluster_penalty,
    }
    return thresholds, summary


@torch.no_grad()
def _candidate_selector_feature_gain_diagnostics(
    *,
    tensors: Optional[Dict[str, torch.Tensor]],
    feature_names: Optional[List[str]] = None,
    penalty_names: Optional[List[str]] = None,
    allowed_mask_cp: Optional[torch.Tensor] = None,
    indices: Optional[torch.Tensor] = None,
    cluster_id_c: Optional[torch.Tensor] = None,
    K: Optional[int] = None,
    min_abs_improvement: float = 0.0,
    min_rel_improvement: float = 0.0,
    topk: int = 5,
) -> Optional[Dict[str, object]]:
    if tensors is None:
        return None
    cand_feat = tensors.get("cand_feat")
    base = tensors.get("base")
    cand = tensors.get("cand")
    y = tensors.get("y")
    if cand_feat is None or base is None or cand is None or y is None:
        return None
    n = int(base.shape[0])
    if n <= 0:
        return None
    if indices is None:
        indices = torch.arange(0, n, dtype=torch.long)
    else:
        indices = indices.detach().cpu().to(dtype=torch.long)
    if int(indices.numel()) == 0:
        return None

    cand_feat = cand_feat.index_select(0, indices)
    base = base.index_select(0, indices)
    cand = cand.index_select(0, indices)
    y = y.index_select(0, indices)
    B, C, P, F = cand_feat.shape
    names = list(feature_names or [f"f{i}" for i in range(F)])
    if len(names) != F:
        names = [f"f{i}" for i in range(F)]
    penalties = list(penalty_names or [f"p{i}" for i in range(P)])
    if len(penalties) != P:
        penalties = [f"p{i}" for i in range(P)]

    allowed = _selector_allowed_mask_cp(
        allowed_mask_cp,
        C=int(C),
        P=int(P),
        device=cand_feat.device,
        context="candidate selector feature diagnostics",
    )
    valid_bcp = torch.ones((B, C, P), dtype=torch.bool, device=cand_feat.device)
    if allowed is not None:
        valid_bcp = valid_bcp & allowed.unsqueeze(0)

    base_err_bc = (base - y).pow(2).mean(dim=-1)
    cand_err_bcp = (cand - y.unsqueeze(2)).pow(2).mean(dim=-1)
    gain_bcp = base_err_bc.unsqueeze(-1) - cand_err_bcp
    required_bcp = torch.maximum(
        torch.full_like(gain_bcp, max(0.0, float(min_abs_improvement))),
        max(0.0, float(min_rel_improvement)) * base_err_bc.abs().unsqueeze(-1).clamp_min(1.0e-12),
    )
    positive_bcp = gain_bcp > required_bcp

    finite_bcpf = torch.isfinite(cand_feat).all(dim=-1)
    finite_bcp = finite_bcpf & torch.isfinite(gain_bcp)
    valid_bcp = valid_bcp & finite_bcp

    def _corr_table(feat_nf: torch.Tensor, target_n: torch.Tensor, limit: int) -> List[Dict[str, object]]:
        if int(feat_nf.shape[0]) <= 1:
            return [
                {"feature": names[i], "corr": 0.0, "abs_corr": 0.0, "std": 0.0}
                for i in range(min(int(limit), int(feat_nf.shape[1])))
            ]
        x = feat_nf.to(dtype=torch.float64)
        target = target_n.to(dtype=torch.float64)
        y_center = target - target.mean()
        y_ss = y_center.pow(2).sum()
        x_center = x - x.mean(dim=0, keepdim=True)
        x_ss = x_center.pow(2).sum(dim=0)
        denom = (x_ss * y_ss).clamp_min(0.0).sqrt()
        corr_f = torch.where(denom > 1.0e-12, (x_center * y_center.unsqueeze(-1)).sum(dim=0) / denom, torch.zeros_like(denom))
        std_f = x.std(dim=0, unbiased=False)
        order = sorted(range(int(x.shape[1])), key=lambda i: (-abs(float(corr_f[i].item())), i))
        rows = []
        for i in order[: max(0, int(limit))]:
            corr = float(corr_f[i].item())
            rows.append(
                {
                    "feature": names[i],
                    "corr": corr,
                    "abs_corr": abs(corr),
                    "std": float(std_f[i].item()),
                }
            )
        return rows

    def _summarize(mask_bcp: torch.Tensor) -> Dict[str, object]:
        flat_mask = mask_bcp.reshape(-1)
        samples = int(flat_mask.sum().item())
        if samples <= 0:
            return {
                "samples": 0,
                "positive_rate": 0.0,
                "mean_gain": 0.0,
                "top_abs_gain_corr": [],
                "top_abs_positive_corr": [],
            }
        feat_nf = cand_feat.reshape(-1, F)[flat_mask]
        gain_n = gain_bcp.reshape(-1)[flat_mask]
        positive_n = positive_bcp.reshape(-1)[flat_mask].to(dtype=torch.float32)
        return {
            "samples": samples,
            "positive_rate": float(positive_n.mean().item()),
            "mean_gain": float(gain_n.mean().item()),
            "top_abs_gain_corr": _corr_table(feat_nf, gain_n, int(topk)),
            "top_abs_positive_corr": _corr_table(feat_nf, positive_n, int(topk)),
        }

    summary = _summarize(valid_bcp)
    summary["feature_names"] = names
    summary["penalty_names"] = penalties
    summary["min_abs_improvement"] = float(max(0.0, float(min_abs_improvement)))
    summary["min_rel_improvement"] = float(max(0.0, float(min_rel_improvement)))
    summary["allowed_candidates"] = int(valid_bcp.sum().item())
    by_penalty = {}
    for p_idx, name in enumerate(penalties):
        penalty_mask = torch.zeros_like(valid_bcp)
        penalty_mask[:, :, p_idx] = valid_bcp[:, :, p_idx]
        by_penalty[str(name)] = _summarize(penalty_mask)
    summary["by_penalty"] = by_penalty

    if cluster_id_c is not None:
        cluster_idx = cluster_id_c.detach().cpu().to(dtype=torch.long)
        if int(cluster_idx.numel()) == int(C):
            k_count = int(K) if K is not None else int(cluster_idx.max().item() + 1)
            by_cluster = {}
            for k in range(k_count):
                channel_mask_c = (cluster_idx == int(k)).to(device=valid_bcp.device)
                cluster_mask = valid_bcp & channel_mask_c.view(1, C, 1)
                by_cluster[str(k)] = _summarize(cluster_mask)
            summary["by_cluster"] = by_cluster
    return summary


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
    allowed_mask_cp: Optional[torch.Tensor] = None,
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
    allowed_mask_device = _selector_allowed_mask_cp(
        allowed_mask_cp,
        C=int(base.shape[1]),
        P=P,
        device=device,
        context="candidate selector metrics",
    )
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
        selected_b, selected_class_b = selector.select_from_features(
            skip_feat_b,
            cand_feat_b,
            base_b,
            cand_b,
            allowed_mask_cp=allowed_mask_device,
        )
        target_b = _candidate_selector_targets(
            base_bch=base_b,
            cand_bcpH=cand_b,
            y_bch=y_b,
            min_abs_improvement=min_abs_improvement,
            min_rel_improvement=min_rel_improvement,
            allowed_mask_cp=allowed_mask_device,
        )
        all_pred_bq = torch.cat([base_b.unsqueeze(2), cand_b], dim=2)
        target_pred_b = all_pred_bq.gather(
            2,
            target_b.view(*target_b.shape, 1, 1).expand(-1, -1, 1, int(base_b.shape[-1])),
        ).squeeze(2)
        base_err_bc = (base_b - y_b).pow(2).mean(dim=-1)
        cand_err_bcp = (cand_b - y_b.unsqueeze(2)).pow(2).mean(dim=-1)
        if allowed_mask_device is not None:
            cand_err_bcp = cand_err_bcp.masked_fill(~allowed_mask_device.unsqueeze(0), float("inf"))
        best_cand_err_bc, oracle_p_bc = cand_err_bcp.min(dim=-1)
        oracle_use_bc = torch.isfinite(best_cand_err_bc) & (best_cand_err_bc < base_err_bc)
        oracle_err_bc = torch.where(oracle_use_bc, best_cand_err_bc, base_err_bc)
        oracle_class_bc = torch.where(
            oracle_use_bc,
            oracle_p_bc.to(dtype=torch.long) + 1,
            torch.zeros_like(oracle_p_bc, dtype=torch.long),
        )
        selected_se += float((selected_b - y_b).pow(2).sum().item())
        target_se += float((target_pred_b - y_b).pow(2).sum().item())
        oracle_se += float(oracle_err_bc.sum().item() * y_b.shape[-1])
        base_se += float((base_b - y_b).pow(2).sum().item())
        denom += int(y_b.numel())
        total_bc += int(target_b.numel())
        correct += int((selected_class_b == target_b).sum().item())
        selected_count_q += torch.bincount(selected_class_b.detach().cpu().reshape(-1), minlength=q)[:q]
        target_count_q += torch.bincount(target_b.detach().cpu().reshape(-1), minlength=q)[:q]
        oracle_count_q += torch.bincount(oracle_class_bc.detach().cpu().reshape(-1), minlength=q)[:q]

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


def _candidate_selector_feature_standardization_stats(
    *,
    skip_feat: torch.Tensor,
    cand_feat: torch.Tensor,
    selector: PredResidualCandidateSelector,
    train_idx: torch.Tensor,
    mode: str = "mean_std",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    skip_train_raw = skip_feat.index_select(0, train_idx)
    cand_train_raw = cand_feat.index_select(0, train_idx)
    skip_train_aug, cand_train_aug = selector._append_penalty_identity(skip_train_raw, cand_train_raw)
    skip_train = skip_train_aug.reshape(-1, int(skip_train_aug.shape[-1]))
    cand_train = cand_train_aug.reshape(-1, int(cand_train_aug.shape[-1]))
    feat_train = torch.cat([skip_train, cand_train], dim=0).to(dtype=torch.float32)
    mode_norm = str(mode or "mean_std").lower()
    if mode_norm in {"robust", "median_iqr", "iqr"}:
        feat_mean = feat_train.median(dim=0).values
        q = torch.quantile(feat_train, torch.tensor([0.25, 0.75], device=feat_train.device), dim=0)
        iqr_scale = (q[1] - q[0]) / 1.349
        std_fallback = feat_train.std(dim=0, unbiased=False)
        feat_std = torch.where(iqr_scale > 1.0e-6, iqr_scale, std_fallback).clamp_min(1.0e-6)
        mode_summary = "robust"
    elif mode_norm in {"mean_std", "standard", "std"}:
        feat_mean = feat_train.mean(dim=0)
        feat_std = feat_train.std(dim=0, unbiased=False).clamp_min(1.0e-6)
        mode_summary = "mean_std"
    else:
        raise ValueError("candidate_selector.standardization_mode must be mean_std or robust.")
    summary = {
        "mode": mode_summary,
        "min_std": float(feat_std.min().item()),
        "max_std": float(feat_std.max().item()),
    }
    return feat_mean, feat_std, summary


@torch.no_grad()
def _fit_static_candidate_channel_selector_from_tensors(
    *,
    tensors: Dict[str, torch.Tensor],
    allowed_mask_cp: Optional[torch.Tensor] = None,
    penalty_names: Optional[List[str]] = None,
    channel_names: Optional[List[str]] = None,
    select_indices: Optional[torch.Tensor] = None,
    eval_indices: Optional[torch.Tensor] = None,
    min_abs_improvement: float = 0.0,
    min_rel_improvement: float = 0.0,
) -> Tuple[StaticPredResidualCandidateSelector, Dict[str, object]]:
    base = tensors["base"]
    cand = tensors["cand"]
    y = tensors["y"]
    n, C, _ = base.shape
    P = int(cand.shape[2])
    if select_indices is None:
        select_indices = torch.arange(0, n, dtype=torch.long)
    else:
        select_indices = select_indices.detach().cpu().to(dtype=torch.long)
    if eval_indices is None:
        eval_indices = select_indices
    else:
        eval_indices = eval_indices.detach().cpu().to(dtype=torch.long)
    allowed = _selector_allowed_mask_cp(
        allowed_mask_cp,
        C=int(C),
        P=P,
        device=base.device,
        context="static candidate channel selector",
    )

    def _mse_mae_for_indices(indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        base_i = base.index_select(0, indices)
        cand_i = cand.index_select(0, indices)
        y_i = y.index_select(0, indices)
        base_mse_c = (base_i - y_i).pow(2).mean(dim=(0, 2))
        base_mae_c = (base_i - y_i).abs().mean(dim=(0, 2))
        cand_mse_cp = (cand_i - y_i.unsqueeze(2)).pow(2).mean(dim=(0, 3))
        cand_mae_cp = (cand_i - y_i.unsqueeze(2)).abs().mean(dim=(0, 3))
        if allowed is not None:
            cand_mse_cp = cand_mse_cp.masked_fill(~allowed, float("inf"))
            cand_mae_cp = cand_mae_cp.masked_fill(~allowed, float("inf"))
        return base_mse_c, base_mae_c, cand_mse_cp, cand_mae_cp

    select_base_mse_c, select_base_mae_c, select_cand_mse_cp, select_cand_mae_cp = _mse_mae_for_indices(select_indices)
    best_cand_mse_c, best_p_c = select_cand_mse_cp.min(dim=-1)
    best_cand_mae_c = select_cand_mae_cp.gather(1, best_p_c.view(-1, 1)).squeeze(1)
    required_c = torch.maximum(
        torch.full_like(select_base_mse_c, max(0.0, float(min_abs_improvement))),
        max(0.0, float(min_rel_improvement)) * select_base_mse_c.abs().clamp_min(1.0e-12),
    )
    use_candidate_c = torch.isfinite(best_cand_mse_c) & ((select_base_mse_c - best_cand_mse_c) > required_c)
    selected_class_c = torch.where(
        use_candidate_c,
        best_p_c.to(dtype=torch.long) + 1,
        torch.zeros_like(best_p_c, dtype=torch.long),
    )
    selector = StaticPredResidualCandidateSelector(selected_class_c)

    eval_base_mse_c, eval_base_mae_c, eval_cand_mse_cp, eval_cand_mae_cp = _mse_mae_for_indices(eval_indices)
    safe_p_c = (selected_class_c - 1).clamp_min(0)
    chosen_cand_mse_c = eval_cand_mse_cp.gather(1, safe_p_c.view(-1, 1)).squeeze(1)
    chosen_cand_mae_c = eval_cand_mae_cp.gather(1, safe_p_c.view(-1, 1)).squeeze(1)
    selected_mse_c = torch.where(selected_class_c > 0, chosen_cand_mse_c, eval_base_mse_c)
    selected_mae_c = torch.where(selected_class_c > 0, chosen_cand_mae_c, eval_base_mae_c)

    penalties = list(penalty_names or [f"p{i}" for i in range(P)])
    if len(penalties) != P:
        penalties = [f"p{i}" for i in range(P)]
    channels = list(channel_names or [f"ch_{i}" for i in range(C)])
    if len(channels) != C:
        channels = [f"ch_{i}" for i in range(C)]
    selected_classes = [int(v) for v in selected_class_c.detach().cpu().tolist()]
    selected_names = [penalties[cls - 1] if cls > 0 else "skip" for cls in selected_classes]
    base_avg_mse = float(eval_base_mse_c.mean().item())
    base_avg_mae = float(eval_base_mae_c.mean().item())
    selected_avg_mse = float(selected_mse_c.mean().item())
    selected_avg_mae = float(selected_mae_c.mean().item())
    summary = {
        "mode": "static_candidate_channel",
        "selection_windows": int(select_indices.numel()),
        "eval_windows": int(eval_indices.numel()),
        "min_abs_improvement": float(max(0.0, float(min_abs_improvement))),
        "min_rel_improvement": float(max(0.0, float(min_rel_improvement))),
        "selected_class": selected_classes,
        "selected_penalty_by_channel": selected_names,
        "selected_channels": [channels[i] for i, cls in enumerate(selected_classes) if cls > 0],
        "num_candidate_channels": int((selected_class_c > 0).sum().item()),
        "select_base_mse_per_channel": [float(v) for v in select_base_mse_c.detach().cpu().tolist()],
        "select_best_candidate_mse_per_channel": [float(v) for v in best_cand_mse_c.detach().cpu().tolist()],
        "select_best_candidate_mae_per_channel": [float(v) for v in best_cand_mae_c.detach().cpu().tolist()],
        "eval_base_avg_mse": base_avg_mse,
        "eval_base_avg_mae": base_avg_mae,
        "eval_selected_avg_mse": selected_avg_mse,
        "eval_selected_avg_mae": selected_avg_mae,
        "eval_gain_pct_vs_base": float(100.0 * (base_avg_mse - selected_avg_mse) / max(abs(base_avg_mse), 1.0e-12)),
        "eval_mae_gain_pct_vs_base": float(100.0 * (base_avg_mae - selected_avg_mae) / max(abs(base_avg_mae), 1.0e-12)),
        "eval_base_mse_per_channel": [float(v) for v in eval_base_mse_c.detach().cpu().tolist()],
        "eval_selected_mse_per_channel": [float(v) for v in selected_mse_c.detach().cpu().tolist()],
        "eval_base_mae_per_channel": [float(v) for v in eval_base_mae_c.detach().cpu().tolist()],
        "eval_selected_mae_per_channel": [float(v) for v in selected_mae_c.detach().cpu().tolist()],
    }
    return selector, summary


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
    model_train_stat_adapter_pc: Optional[torch.Tensor] = None,
    model_train_stat_adapter_cfg: Optional[dict] = None,
    train_stat_anchor_pc: Optional[torch.Tensor] = None,
    train_residual_anchor_phc: Optional[torch.Tensor] = None,
    allowed_mask_kp: Optional[torch.Tensor] = None,
    gate_feature_mode: str = "history",
) -> Tuple[Optional[PredResidualCandidateSelector], Dict[str, object]]:
    candidate_feature_mode = str(cfg.get("feature_mode", "base")).lower()
    candidate_feature_names = _candidate_selector_feature_names(candidate_feature_mode)
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
        model_train_stat_adapter_pc=model_train_stat_adapter_pc,
        model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
        train_stat_anchor_pc=train_stat_anchor_pc,
        train_residual_anchor_phc=train_residual_anchor_phc,
        candidate_feature_mode=candidate_feature_mode,
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
    allowed_mask_cp = None
    if allowed_mask_kp is not None and int(allowed_mask_kp.numel()) > 0:
        allowed_kp = allowed_mask_kp.detach().to(dtype=torch.bool)
        if tuple(allowed_kp.shape) != (int(K), p):
            raise ValueError(
                "candidate_selector allowed_mask_kp must have shape [K,P], "
                f"got {tuple(allowed_kp.shape)} vs {(int(K), p)}."
            )
        cluster_idx = cluster_id_c.detach().cpu().to(dtype=torch.long)
        allowed_mask_cp = allowed_kp.detach().cpu().index_select(0, cluster_idx)
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
        feature_mode=candidate_feature_mode,
    ).to(device)
    selector.set_allowed_penalty_mask(allowed_mask_cp)
    standardize_features = bool(cfg.get("standardize_features", True))
    feature_std_summary: Dict[str, object] = {
        "standardize_features": bool(standardize_features),
        "fit_windows": int(train_idx.numel()) if standardize_features else 0,
        "mode": None,
        "min_std": None,
        "max_std": None,
        "clip": 0.0,
    }
    if standardize_features:
        feat_mean, feat_std, std_stats = _candidate_selector_feature_standardization_stats(
            skip_feat=skip_feat,
            cand_feat=cand_feat,
            selector=selector,
            train_idx=train_idx,
            mode=str(cfg.get("standardization_mode", "mean_std")),
        )
        selector.set_feature_standardization(feat_mean.to(device), feat_std.to(device))
        feature_clip = float(cfg.get("standardize_clip", cfg.get("feature_clip", 0.0)))
        selector.set_feature_standardize_clip(feature_clip)
        feature_std_summary.update(std_stats)
        feature_std_summary["clip"] = float(max(0.0, feature_clip))

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
        allowed_mask_cp=allowed_mask_cp,
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
            allowed_mask_cp=allowed_mask_cp,
        )
        logits_bcq = selector.logits_from_features(skip_feat_b, cand_feat_b, allowed_mask_cp=allowed_mask_cp)
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
            allowed_mask_cp=allowed_mask_cp,
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
                allowed_mask_cp=allowed_mask_cp,
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
        allowed_mask_cp=allowed_mask_cp,
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
        allowed_mask_cp=allowed_mask_cp,
    )
    feature_gain_diagnostics = {
        "train": _candidate_selector_feature_gain_diagnostics(
            tensors=tensors,
            feature_names=candidate_feature_names,
            penalty_names=penalty_names,
            allowed_mask_cp=allowed_mask_cp,
            indices=train_idx,
            cluster_id_c=cluster_id_c,
            K=K,
            min_abs_improvement=min_abs_improvement,
            min_rel_improvement=min_rel_improvement,
            topk=int(cfg.get("feature_gain_diagnostic_topk", 5)),
        ),
        "holdout": _candidate_selector_feature_gain_diagnostics(
            tensors=tensors,
            feature_names=candidate_feature_names,
            penalty_names=penalty_names,
            allowed_mask_cp=allowed_mask_cp,
            indices=hold_idx,
            cluster_id_c=cluster_id_c,
            K=K,
            min_abs_improvement=min_abs_improvement,
            min_rel_improvement=min_rel_improvement,
            topk=int(cfg.get("feature_gain_diagnostic_topk", 5)),
        ),
    }
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
        "feature_mode": candidate_feature_mode,
        "decision_margin": float(selector.decision_margin),
        "decision_margin_selection": margin_selection,
        "feature_standardization": feature_std_summary,
        "channel_names": list(channel_names),
        "penalty_names": list(penalty_names),
        "allowed_mask_cp": allowed_mask_cp.to(dtype=torch.long).tolist() if allowed_mask_cp is not None else None,
        "gate_feature_mode": _normalize_gate_feature_mode(gate_feature_mode),
        "feature_gain_diagnostics": feature_gain_diagnostics,
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
    cfg["moe"] = apply_default_moe_output_anchor_cfg(
        cfg.get("moe", {}) or {},
        dataset_name=cfg.get("data", {}).get("csv_path", ""),
        pred_len=H,
    )
    eval_cfg = cfg.get("eval", {}) or {}
    skip_test = bool(eval_cfg.get("skip_test", True))
    diagnostics_cfg = cfg.get("diagnostics", {}) or {}
    stage2_loss_audit_cfg = diagnostics_cfg.get("stage2_loss_audit", {}) or {}
    stage2_loss_audit_enable = bool(stage2_loss_audit_cfg.get("enable", False))
    if stage2_loss_audit_enable:
        print("Stage-2 loss audit diagnostics enabled.")
    stage2_route_audit_cfg = diagnostics_cfg.get("stage2_route_audit", {}) or {}
    if not isinstance(stage2_route_audit_cfg, dict):
        stage2_route_audit_cfg = {"enable": bool(stage2_route_audit_cfg)}
    stage2_route_audit_enable = bool(stage2_route_audit_cfg.get("enable", False))
    if stage2_route_audit_enable:
        print("Stage-2 route audit diagnostics enabled.")

    # Keep materialized windows on CPU.  Electricity-style datasets with many
    # channels and long horizons can expand to tens of GB; batches are moved to
    # CUDA by the train/eval loops.
    data_window_tc = data_tc.detach().cpu()
    window_cfg = cfg.get("window", {}) or {}
    past_context = bool(window_cfg.get("past_context", False))
    lazy_windows = bool(window_cfg.get("lazy", False))
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
    stage2_route_audit_loaders: Dict[str, DataLoader] = {}
    stage2_route_audit_eval_starts: Dict[str, int] = {}
    stage2_route_audit_train_subsplits: Dict[str, Tuple[int, int]] = {}
    if stage2_route_audit_enable:
        requested_route_splits = [
            str(name).lower()
            for name in (stage2_route_audit_cfg.get("splits", ["train_fit", "train_holdout", "val"]) or [])
        ]
        if "test" in requested_route_splits and not bool(stage2_route_audit_cfg.get("allow_test", False)):
            raise ValueError("diagnostics.stage2_route_audit refuses to read test unless allow_test=true.")
        if len(dtr) > 0 and "train" in requested_route_splits:
            stage2_route_audit_loaders["train"] = DataLoader(
                dtr,
                batch_size=bs,
                shuffle=False,
                num_workers=0,
                pin_memory=pin_mem,
            )
            stage2_route_audit_eval_starts["train"] = 0
        train_subsplit_names = {"train_fit", "train_holdout"}
        if len(dtr) > 0 and any(name in requested_route_splits for name in train_subsplit_names):
            stage2_route_audit_train_subsplits = _explainability_train_subsplit_ranges(
                num_windows=len(dtr),
                holdout_fraction=float(stage2_route_audit_cfg.get("train_holdout_fraction", 0.30)),
            )
            for split_name in ("train_fit", "train_holdout"):
                if split_name not in requested_route_splits:
                    continue
                if split_name not in stage2_route_audit_train_subsplits:
                    continue
                start_i, end_i = stage2_route_audit_train_subsplits[split_name]
                if int(end_i) <= int(start_i):
                    continue
                stage2_route_audit_loaders[split_name] = DataLoader(
                    Subset(dtr, range(int(start_i), int(end_i))),
                    batch_size=bs,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=pin_mem,
                )
                stage2_route_audit_eval_starts[split_name] = 0
        if len(dva) > 0 and "val" in requested_route_splits:
            stage2_route_audit_loaders["val"] = dl_va
            stage2_route_audit_eval_starts["val"] = int(val_eval_start)
        if len(dte) > 0 and "test" in requested_route_splits and bool(stage2_route_audit_cfg.get("allow_test", False)):
            stage2_route_audit_loaders["test"] = dl_te
            stage2_route_audit_eval_starts["test"] = int(test_eval_start)
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
        kwargs.setdefault("gate_feature_mode", gate_feature_mode)
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
    gate_feature_mode = _normalize_gate_feature_mode(moe_cfg.get("gate_feature_mode", "history"))
    router_mode = str(moe_cfg.get("router_mode", "learned")).lower()
    router_penalty_context_weight = float(moe_cfg.get("router_penalty_context_weight", 0.0))
    router_detach_penalty_context = bool(moe_cfg.get("router_detach_penalty_context", True))
    router_penalty_context_score = str(moe_cfg.get("router_penalty_context_score", "high_violation")).lower()
    pred_residual_cfg = moe_cfg.get("pred_side_residual", {}) or {}
    pred_residual_enable = bool(pred_residual_cfg.get("enable", False)) and moe_enable and P > 0
    phase_residual_candidate_cfg = pred_residual_cfg.get("phase_residual_candidate", {}) or {}
    if not isinstance(phase_residual_candidate_cfg, dict):
        phase_residual_candidate_cfg = {"enable": bool(phase_residual_candidate_cfg)}
    phase_residual_candidate_enable = bool(phase_residual_candidate_cfg.get("enable", False)) and pred_residual_enable
    raw_phase_residual_candidate_names = phase_residual_candidate_cfg.get(
        "names",
        phase_residual_candidate_cfg.get("penalty_names", []),
    )
    if isinstance(raw_phase_residual_candidate_names, str):
        phase_residual_candidate_names = [raw_phase_residual_candidate_names]
    else:
        phase_residual_candidate_names = [str(v) for v in (raw_phase_residual_candidate_names or [])]
    if phase_residual_candidate_enable and len(phase_residual_candidate_names) == 0:
        raise ValueError("moe.pred_side_residual.phase_residual_candidate.names must be non-empty when enabled.")
    phase_residual_candidate_period = int(phase_residual_candidate_cfg.get("period", 96))
    if phase_residual_candidate_enable and phase_residual_candidate_period <= 0:
        raise ValueError("moe.pred_side_residual.phase_residual_candidate.period must be positive.")
    phase_residual_candidate_scale = float(phase_residual_candidate_cfg.get("scale", 1.0))
    phase_residual_candidate_summary: Dict[str, object] = {
        "enable": bool(phase_residual_candidate_enable),
        "names": list(phase_residual_candidate_names),
        "period": int(phase_residual_candidate_period),
        "scale": float(phase_residual_candidate_scale),
        "source_split": "train" if phase_residual_candidate_enable else None,
    }
    pred_residual_ignore_skip_during_training = bool(
        pred_residual_cfg.get(
            "ignore_skip_during_training",
            pred_residual_cfg.get("train_ignore_skip", False),
        )
    ) and pred_residual_enable
    pred_residual_specialization_weight = (
        float(pred_residual_cfg.get("specialization_weight", 0.1)) if pred_residual_enable else 0.0
    )
    pred_residual_norm_weight = float(pred_residual_cfg.get("norm_weight", 1.0e-4)) if pred_residual_enable else 0.0
    pred_residual_intervention_weight = (
        float(pred_residual_cfg.get("intervention_weight", 1.0e-3)) if pred_residual_enable else 0.0
    )
    pred_residual_candidate_supervision_cfg = (
        pred_residual_cfg.get(
            "adapter_attribute_supervision",
            pred_residual_cfg.get("candidate_supervision", {}),
        )
        or {}
    )
    if not isinstance(pred_residual_candidate_supervision_cfg, dict):
        pred_residual_candidate_supervision_cfg = {"weight": float(pred_residual_candidate_supervision_cfg)}
    pred_residual_candidate_supervision_weight = (
        float(
            pred_residual_candidate_supervision_cfg.get(
                "weight",
                pred_residual_cfg.get("candidate_supervision_weight", 0.0),
            )
        )
        if pred_residual_enable
        else 0.0
    )
    pred_residual_candidate_supervision_loss = str(
        pred_residual_candidate_supervision_cfg.get("loss", "mse")
    ).lower()
    pred_residual_candidate_supervision_min_abs = float(
        pred_residual_candidate_supervision_cfg.get("min_abs_improvement", 0.0)
    )
    pred_residual_candidate_supervision_min_rel = float(
        pred_residual_candidate_supervision_cfg.get("min_rel_improvement", 0.0)
    )
    pred_residual_candidate_supervision_only_allowed = bool(
        pred_residual_candidate_supervision_cfg.get("only_allowed", True)
    )
    pred_residual_candidate_supervision_include_intervention = bool(
        pred_residual_candidate_supervision_cfg.get("include_intervention", False)
    )
    pred_residual_candidate_supervision_include_selector = bool(
        pred_residual_candidate_supervision_cfg.get("include_selector", False)
    )
    pred_residual_intervention_supervision_cfg = pred_residual_cfg.get("intervention_supervision", {}) or {}
    if not isinstance(pred_residual_intervention_supervision_cfg, dict):
        pred_residual_intervention_supervision_cfg = {"weight": float(pred_residual_intervention_supervision_cfg)}
    pred_residual_intervention_supervision_weight = (
        float(pred_residual_intervention_supervision_cfg.get("weight", 0.0))
        if pred_residual_enable
        else 0.0
    )
    pred_residual_intervention_supervision_min_gain = float(
        pred_residual_intervention_supervision_cfg.get("min_gain", 0.0)
    )
    pred_residual_intervention_supervision_pos_weight = float(
        pred_residual_intervention_supervision_cfg.get("pos_weight", 1.0)
    )
    pred_residual_intervention_supervision_only_allowed = bool(
        pred_residual_intervention_supervision_cfg.get("only_allowed", True)
    )
    pred_residual_confidence_gate_cfg = pred_residual_cfg.get("confidence_gate", {}) or {}
    if not isinstance(pred_residual_confidence_gate_cfg, dict):
        pred_residual_confidence_gate_cfg = {"enable": bool(pred_residual_confidence_gate_cfg)}
    pred_residual_confidence_gate_enable = (
        bool(pred_residual_confidence_gate_cfg.get("enable", False))
        and pred_residual_enable
        and P > 0
    )
    pred_residual_confidence_gate_source_split = "train_holdout"
    if pred_residual_confidence_gate_enable:
        pred_residual_confidence_gate_source_split = _normalize_confidence_gate_source_split(
            pred_residual_confidence_gate_cfg.get("source_split", "train_holdout")
        )
    pred_residual_confidence_gate_threshold = pred_residual_confidence_gate_cfg.get("threshold", "auto")
    pred_residual_confidence_gate_min_abs = float(
        pred_residual_confidence_gate_cfg.get(
            "min_abs_improvement",
            pred_residual_intervention_supervision_cfg.get("min_gain", 0.0),
        )
    )
    pred_residual_confidence_gate_min_rel = float(
        pred_residual_confidence_gate_cfg.get("min_rel_improvement", 0.0)
    )
    pred_residual_confidence_gate_holdout_fraction = float(
        pred_residual_confidence_gate_cfg.get("train_holdout_fraction", 0.30)
    )
    pred_residual_confidence_gate_max_candidates = int(
        pred_residual_confidence_gate_cfg.get("threshold_candidates", 101)
    )
    pred_residual_confidence_gate_selection_metric = str(
        pred_residual_confidence_gate_cfg.get("selection_metric", "mse")
    ).lower()
    pred_residual_confidence_gate_min_precision = float(
        pred_residual_confidence_gate_cfg.get("min_precision", 0.0)
    )
    pred_residual_confidence_gate_max_pred_rate_raw = pred_residual_confidence_gate_cfg.get(
        "max_pred_positive_rate",
        None,
    )
    pred_residual_confidence_gate_max_pred_rate = (
        None
        if pred_residual_confidence_gate_max_pred_rate_raw is None
        else float(pred_residual_confidence_gate_max_pred_rate_raw)
    )
    pred_residual_detach_routed_penalty_pred = (
        bool(pred_residual_cfg.get("detach_routed_penalty_pred", False)) if pred_residual_enable else False
    )
    pred_residual_freeze_gate_after_epoch = (
        int(pred_residual_cfg.get("freeze_gate_after_epoch", 0)) if pred_residual_enable else 0
    )
    pred_residual_weight_decay = None
    if pred_residual_enable:
        raw_pred_residual_wd = pred_residual_cfg.get("weight_decay", None)
        if raw_pred_residual_wd is None:
            raw_pred_residual_wd = pred_residual_cfg.get("optimizer_weight_decay", None)
        if raw_pred_residual_wd is not None:
            pred_residual_weight_decay = float(raw_pred_residual_wd)
    allow_skip = bool(moe_cfg.get("allow_skip", False)) and moe_enable and P > 0
    skip_cost = float(moe_cfg.get("skip_cost", 0.0)) if allow_skip else 0.0
    skip_init_bias = float(moe_cfg.get("skip_init_bias", -2.0))
    skip_competes = bool(
        moe_cfg.get("skip_competes_with_penalties", moe_cfg.get("noop_compete_enable", False))
    ) and allow_skip
    skip_argmax_noop = bool(moe_cfg.get("skip_argmax_noop", False)) and skip_competes
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
    mse_utility_gate_target_mode = str(mse_utility_gate_cfg.get("target_mode", "soft_utility"))
    mse_utility_gate_include_skip = bool(
        mse_utility_gate_cfg.get("include_skip", mse_utility_gate_cfg.get("skip_aware", False))
    ) and allow_skip
    route_ce_cfg = moe_cfg.get("route_ce_supervision", {}) or {}
    if not isinstance(route_ce_cfg, dict):
        route_ce_cfg = {"enable": bool(route_ce_cfg)}
    route_ce_enable = (
        bool(route_ce_cfg.get("enable", False))
        and moe_enable
        and pred_residual_enable
        and P > 0
    )
    route_ce_weight = float(route_ce_cfg.get("weight", 0.0)) if route_ce_enable else 0.0
    route_ce_min_abs_improvement = float(route_ce_cfg.get("min_abs_improvement", 0.0))
    route_ce_min_rel_improvement = float(route_ce_cfg.get("min_rel_improvement", 0.0))
    route_ce_min_candidate_delta_rms = float(
        route_ce_cfg.get(
            "min_candidate_delta_rms",
            route_ce_cfg.get("candidate_action_floor", 0.0),
        )
    )
    route_ce_ignore_abs_gain_below = float(
        route_ce_cfg.get(
            "ignore_abs_gain_below",
            route_ce_cfg.get("ignore_near_zero_abs_gain", 0.0),
        )
    )
    route_ce_class_weight_mode = str(route_ce_cfg.get("class_weight", "none"))
    route_ce_max_class_weight = float(route_ce_cfg.get("max_class_weight", 0.0))
    route_ce_require_skip = bool(route_ce_cfg.get("require_skip", True))
    route_ce_require_skip_competes = bool(route_ce_cfg.get("require_skip_competes", True))
    route_ce_require_skip_argmax_noop = bool(route_ce_cfg.get("require_skip_argmax_noop", True))
    if route_ce_weight > 0.0:
        if route_ce_require_skip and not allow_skip:
            raise ValueError("moe.route_ce_supervision requires moe.allow_skip=true.")
        if route_ce_require_skip_competes and not skip_competes:
            raise ValueError("moe.route_ce_supervision requires moe.skip_competes_with_penalties=true.")
        if route_ce_require_skip_argmax_noop and not skip_argmax_noop:
            raise ValueError("moe.route_ce_supervision requires moe.skip_argmax_noop=true.")
    binary_adoption_cfg = moe_cfg.get("binary_adoption_supervision", {}) or {}
    if not isinstance(binary_adoption_cfg, dict):
        binary_adoption_cfg = {"enable": bool(binary_adoption_cfg)}
    binary_adoption_enable = (
        bool(binary_adoption_cfg.get("enable", False))
        and moe_enable
        and pred_residual_enable
        and P > 0
    )
    binary_adoption_weight = (
        float(binary_adoption_cfg.get("weight", 0.0)) if binary_adoption_enable else 0.0
    )
    binary_adoption_min_abs_improvement = float(
        binary_adoption_cfg.get("min_abs_improvement", route_ce_min_abs_improvement)
    )
    binary_adoption_min_rel_improvement = float(
        binary_adoption_cfg.get("min_rel_improvement", route_ce_min_rel_improvement)
    )
    binary_adoption_min_candidate_delta_rms = float(
        binary_adoption_cfg.get("min_candidate_delta_rms", route_ce_min_candidate_delta_rms)
    )
    binary_adoption_ignore_abs_gain_below = float(
        binary_adoption_cfg.get("ignore_abs_gain_below", route_ce_ignore_abs_gain_below)
    )
    binary_adoption_positive_weight = float(binary_adoption_cfg.get("positive_weight", 1.0))
    binary_adoption_negative_weight = float(binary_adoption_cfg.get("negative_weight", 1.0))
    binary_adoption_require_skip = bool(binary_adoption_cfg.get("require_skip", True))
    binary_adoption_require_skip_competes = bool(binary_adoption_cfg.get("require_skip_competes", True))
    binary_adoption_require_skip_argmax_noop = bool(
        binary_adoption_cfg.get("require_skip_argmax_noop", True)
    )
    if binary_adoption_weight > 0.0:
        if binary_adoption_require_skip and not allow_skip:
            raise ValueError("moe.binary_adoption_supervision requires moe.allow_skip=true.")
        if binary_adoption_require_skip_competes and not skip_competes:
            raise ValueError(
                "moe.binary_adoption_supervision requires moe.skip_competes_with_penalties=true."
            )
        if binary_adoption_require_skip_argmax_noop and not skip_argmax_noop:
            raise ValueError("moe.binary_adoption_supervision requires moe.skip_argmax_noop=true.")
    route_rate_alignment_cfg = moe_cfg.get("route_rate_alignment_supervision", {}) or {}
    if not isinstance(route_rate_alignment_cfg, dict):
        route_rate_alignment_cfg = {"enable": bool(route_rate_alignment_cfg)}
    route_rate_alignment_enable = (
        bool(route_rate_alignment_cfg.get("enable", False))
        and moe_enable
        and pred_residual_enable
        and P > 0
    )
    route_rate_alignment_weight = (
        float(route_rate_alignment_cfg.get("weight", 0.0)) if route_rate_alignment_enable else 0.0
    )
    route_rate_alignment_min_abs_improvement = float(
        route_rate_alignment_cfg.get("min_abs_improvement", binary_adoption_min_abs_improvement)
    )
    route_rate_alignment_min_rel_improvement = float(
        route_rate_alignment_cfg.get("min_rel_improvement", binary_adoption_min_rel_improvement)
    )
    route_rate_alignment_min_candidate_delta_rms = float(
        route_rate_alignment_cfg.get(
            "min_candidate_delta_rms",
            binary_adoption_min_candidate_delta_rms,
        )
    )
    route_rate_alignment_ignore_abs_gain_below = float(
        route_rate_alignment_cfg.get("ignore_abs_gain_below", binary_adoption_ignore_abs_gain_below)
    )
    route_rate_alignment_require_skip = bool(route_rate_alignment_cfg.get("require_skip", True))
    route_rate_alignment_require_skip_competes = bool(
        route_rate_alignment_cfg.get("require_skip_competes", True)
    )
    route_rate_alignment_require_skip_argmax_noop = bool(
        route_rate_alignment_cfg.get("require_skip_argmax_noop", True)
    )
    if route_rate_alignment_weight > 0.0:
        if route_rate_alignment_require_skip and not allow_skip:
            raise ValueError("moe.route_rate_alignment_supervision requires moe.allow_skip=true.")
        if route_rate_alignment_require_skip_competes and not skip_competes:
            raise ValueError(
                "moe.route_rate_alignment_supervision requires "
                "moe.skip_competes_with_penalties=true."
            )
        if route_rate_alignment_require_skip_argmax_noop and not skip_argmax_noop:
            raise ValueError(
                "moe.route_rate_alignment_supervision requires moe.skip_argmax_noop=true."
            )
    route_positive_recall_cfg = moe_cfg.get("route_positive_recall_supervision", {}) or {}
    if not isinstance(route_positive_recall_cfg, dict):
        route_positive_recall_cfg = {"enable": bool(route_positive_recall_cfg)}
    route_positive_recall_enable = (
        bool(route_positive_recall_cfg.get("enable", False))
        and moe_enable
        and pred_residual_enable
        and P > 0
    )
    route_positive_recall_weight = (
        float(route_positive_recall_cfg.get("weight", 0.0)) if route_positive_recall_enable else 0.0
    )
    route_positive_recall_min_abs_improvement = float(
        route_positive_recall_cfg.get("min_abs_improvement", binary_adoption_min_abs_improvement)
    )
    route_positive_recall_min_rel_improvement = float(
        route_positive_recall_cfg.get("min_rel_improvement", binary_adoption_min_rel_improvement)
    )
    route_positive_recall_min_candidate_delta_rms = float(
        route_positive_recall_cfg.get(
            "min_candidate_delta_rms",
            binary_adoption_min_candidate_delta_rms,
        )
    )
    route_positive_recall_ignore_abs_gain_below = float(
        route_positive_recall_cfg.get("ignore_abs_gain_below", binary_adoption_ignore_abs_gain_below)
    )
    route_positive_recall_mode = str(route_positive_recall_cfg.get("mode", "ce"))
    route_positive_recall_target_probability = float(
        route_positive_recall_cfg.get("target_probability", 1.0)
    )
    route_positive_recall_require_skip = bool(route_positive_recall_cfg.get("require_skip", True))
    route_positive_recall_require_skip_competes = bool(
        route_positive_recall_cfg.get("require_skip_competes", True)
    )
    route_positive_recall_require_skip_argmax_noop = bool(
        route_positive_recall_cfg.get("require_skip_argmax_noop", True)
    )
    if route_positive_recall_weight > 0.0:
        if route_positive_recall_require_skip and not allow_skip:
            raise ValueError("moe.route_positive_recall_supervision requires moe.allow_skip=true.")
        if route_positive_recall_require_skip_competes and not skip_competes:
            raise ValueError(
                "moe.route_positive_recall_supervision requires "
                "moe.skip_competes_with_penalties=true."
            )
        if route_positive_recall_require_skip_argmax_noop and not skip_argmax_noop:
            raise ValueError(
                "moe.route_positive_recall_supervision requires moe.skip_argmax_noop=true."
            )
    route_precision_recall_cfg = moe_cfg.get("route_precision_recall_supervision", {}) or {}
    if not isinstance(route_precision_recall_cfg, dict):
        route_precision_recall_cfg = {"enable": bool(route_precision_recall_cfg)}
    route_precision_recall_enable = (
        bool(route_precision_recall_cfg.get("enable", False))
        and moe_enable
        and pred_residual_enable
        and P > 0
    )
    route_precision_recall_weight = (
        float(route_precision_recall_cfg.get("weight", 0.0)) if route_precision_recall_enable else 0.0
    )
    route_precision_recall_min_abs_improvement = float(
        route_precision_recall_cfg.get("min_abs_improvement", binary_adoption_min_abs_improvement)
    )
    route_precision_recall_min_rel_improvement = float(
        route_precision_recall_cfg.get("min_rel_improvement", binary_adoption_min_rel_improvement)
    )
    route_precision_recall_min_candidate_delta_rms = float(
        route_precision_recall_cfg.get(
            "min_candidate_delta_rms",
            binary_adoption_min_candidate_delta_rms,
        )
    )
    route_precision_recall_ignore_abs_gain_below = float(
        route_precision_recall_cfg.get("ignore_abs_gain_below", binary_adoption_ignore_abs_gain_below)
    )
    route_precision_recall_mode = str(route_precision_recall_cfg.get("recall_mode", "ce"))
    route_precision_recall_target_probability = float(
        route_precision_recall_cfg.get("recall_target_probability", 1.0)
    )
    route_precision_recall_false_adopt_max_probability = float(
        route_precision_recall_cfg.get("false_adopt_max_probability", 0.5)
    )
    route_precision_recall_false_adopt_weight = float(
        route_precision_recall_cfg.get("false_adopt_weight", 1.0)
    )
    route_precision_recall_require_skip = bool(route_precision_recall_cfg.get("require_skip", True))
    route_precision_recall_require_skip_competes = bool(
        route_precision_recall_cfg.get("require_skip_competes", True)
    )
    route_precision_recall_require_skip_argmax_noop = bool(
        route_precision_recall_cfg.get("require_skip_argmax_noop", True)
    )
    if route_precision_recall_weight > 0.0:
        if route_precision_recall_require_skip and not allow_skip:
            raise ValueError("moe.route_precision_recall_supervision requires moe.allow_skip=true.")
        if route_precision_recall_require_skip_competes and not skip_competes:
            raise ValueError(
                "moe.route_precision_recall_supervision requires "
                "moe.skip_competes_with_penalties=true."
            )
        if route_precision_recall_require_skip_argmax_noop and not skip_argmax_noop:
            raise ValueError(
                "moe.route_precision_recall_supervision requires moe.skip_argmax_noop=true."
            )
    raw_ranks = moe_cfg.get("select_ranks", None)
    if raw_ranks is None:
        select_ranks = [1, 2]
    else:
        select_ranks = [int(x) for x in raw_ranks]
    gate_feat_dim = len(_gate_feature_names_for_mode(gate_feature_mode))
    gate = ClusterwiseMoEGate(
        num_clusters=K,
        feat_dim=gate_feat_dim,
        num_penalties=P,
        hidden_dim=int(moe_cfg.get("gate_hidden_dim", 64)),
        topk=int(moe_cfg["topk"]),
        allow_skip=allow_skip,
        skip_init_bias=skip_init_bias,
        skip_competes=skip_competes,
        skip_argmax_noop=skip_argmax_noop,
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
            phase_residual_candidate_names=phase_residual_candidate_names,
            phase_residual_candidate_scale=phase_residual_candidate_scale,
        ).to(device)
        print(
            "Prediction residual MoE enabled: "
            f"hidden={pred_residual.hidden_dim}, feature_mode={pred_residual.feature_mode}, "
            f"alpha_scale={pred_residual.alpha_scale:.3f}, "
            f"residual_clip={pred_residual.residual_clip:.3f}, "
            f"seasonal_anchor_names={list(pred_residual_cfg.get('seasonal_anchor_names', []))}, "
            f"seasonal_anchor_period={int(pred_residual_cfg.get('seasonal_anchor_period', 96))}, "
            f"seasonal_anchor_scale={float(pred_residual_cfg.get('seasonal_anchor_scale', 1.0)):.3f}, "
            f"phase_residual_candidate={phase_residual_candidate_names}, "
            f"phase_residual_period={int(phase_residual_candidate_period)}, "
            f"phase_residual_scale={float(phase_residual_candidate_scale):.3f}, "
            f"specialization_weight={pred_residual_specialization_weight:.6f}, "
            f"norm_weight={pred_residual_norm_weight:.6f}, "
            f"intervention_weight={pred_residual_intervention_weight:.6f}, "
            f"candidate_supervision_weight={pred_residual_candidate_supervision_weight:.6f}, "
            f"candidate_supervision_loss={pred_residual_candidate_supervision_loss}, "
            f"ignore_skip_during_training={pred_residual_ignore_skip_during_training}, "
            f"intervention_supervision_weight={pred_residual_intervention_supervision_weight:.6f}, "
            f"route_ce_weight={route_ce_weight:.6f}, "
            f"route_ce_min_candidate_delta_rms={route_ce_min_candidate_delta_rms:.6g}, "
            f"binary_adoption_weight={binary_adoption_weight:.6f}, "
            f"binary_adoption_min_candidate_delta_rms={binary_adoption_min_candidate_delta_rms:.6g}, "
            f"confidence_gate={bool(pred_residual_confidence_gate_enable)}, "
            f"freeze_gate_after_epoch={int(pred_residual_freeze_gate_after_epoch)}, "
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
    cluster_penalty_prior_configured_mask_kp = None
    cluster_penalty_late_allowed_mask_kp = None
    cluster_penalty_prior_apply_stage = "train_and_eval"
    cluster_penalty_prior_late_applied = False
    if cluster_penalty_prior_enable:
        cluster_penalty_prior_apply_stage = normalize_cluster_penalty_prior_apply_stage(
            cluster_penalty_prior_cfg.get("apply_stage", "train_and_eval")
        )
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
            allow_empty_clusters=bool(cluster_penalty_prior_cfg.get("allow_empty_clusters", False)),
        )
        if manual_allowed is not None:
            cluster_penalty_prior_configured_mask_kp = manual_allowed
        elif topk > 0 and bool(cluster_penalty_prior_cfg.get("hard_topk", True)):
            cluster_penalty_prior_configured_mask_kp = build_topk_penalty_mask(
                cluster_penalty_prior_prob_kp,
                topk=topk,
            )
        always_include = cluster_penalty_prior_cfg.get("always_include", []) or []
        if isinstance(always_include, str):
            always_include = [always_include]
        if len(always_include) > 0:
            if cluster_penalty_prior_configured_mask_kp is None:
                cluster_penalty_prior_configured_mask_kp = torch.zeros((K, P), device=device, dtype=torch.float32)
            name_to_idx = {str(name): i for i, name in enumerate(penalty_names)}
            for raw_name in always_include:
                name = str(raw_name)
                if name not in name_to_idx:
                    raise ValueError(
                        "cluster_penalty_prior.always_include contains unknown penalty "
                        f"{name!r}; available={penalty_names}"
                    )
                cluster_penalty_prior_configured_mask_kp[:, name_to_idx[name]] = 1.0
            empty = cluster_penalty_prior_configured_mask_kp.sum(dim=-1, keepdim=True) <= 0.0
            if bool(empty.any().item()):
                cluster_penalty_prior_configured_mask_kp = torch.where(
                    empty,
                    torch.ones_like(cluster_penalty_prior_configured_mask_kp),
                    cluster_penalty_prior_configured_mask_kp,
                )
        (
            cluster_penalty_allowed_mask_kp,
            cluster_penalty_late_allowed_mask_kp,
            cluster_penalty_prior_apply_stage,
        ) = split_cluster_penalty_prior_allowed_mask_by_stage(
            cluster_penalty_prior_configured_mask_kp,
            cluster_penalty_prior_apply_stage,
        )
        gate.set_penalty_allowed_mask(cluster_penalty_allowed_mask_kp)
        if bool(cluster_penalty_prior_cfg.get("use_as_balance_target", False)):
            gate_balance_target_kp = cluster_penalty_prior_prob_kp
        pred_residual_allowed_mask_cp = None
        if (
            pred_residual is not None
            and cluster_penalty_allowed_mask_kp is not None
            and bool(cluster_penalty_prior_cfg.get("apply_to_pred_residual", False))
        ):
            pred_residual_allowed_mask_cp = _cluster_penalty_mask_to_channel_mask(
                cluster_penalty_allowed_mask_kp,
                cluster_id_c,
            )
            pred_residual.set_allowed_penalty_mask(pred_residual_allowed_mask_cp)
        prior_list = (
            cluster_penalty_prior_prob_kp.detach().cpu().tolist()
            if cluster_penalty_prior_prob_kp is not None
            else None
        )
        configured_mask_list = (
            cluster_penalty_prior_configured_mask_kp.detach().cpu().tolist()
            if cluster_penalty_prior_configured_mask_kp is not None
            else None
        )
        active_mask_list = (
            cluster_penalty_allowed_mask_kp.detach().cpu().tolist()
            if cluster_penalty_allowed_mask_kp is not None
            else None
        )
        late_mask_list = (
            cluster_penalty_late_allowed_mask_kp.detach().cpu().tolist()
            if cluster_penalty_late_allowed_mask_kp is not None
            else None
        )
        print(
            "Cluster penalty prior enabled: "
            f"topk={topk}, hard_topk={bool(cluster_penalty_prior_cfg.get('hard_topk', True))}, "
            f"logit_strength={logit_strength:.3f}, apply_stage={cluster_penalty_prior_apply_stage}, "
            f"prior={prior_list}, configured_allowed_mask={configured_mask_list}, "
            f"active_allowed_mask={active_mask_list}, late_allowed_mask={late_mask_list}, "
            f"apply_to_pred_residual={bool(cluster_penalty_prior_cfg.get('apply_to_pred_residual', False))}, "
            f"pred_residual_channel_mask={pred_residual_allowed_mask_cp.detach().cpu().tolist() if pred_residual_allowed_mask_cp is not None else None}"
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
        loaded_pred_residual_state = False
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

        if bool(ft_cfg.get("load_pred_residual", False)) and pred_residual is not None:
            loaded_pred_residual_state = _load_finetune_pred_residual_state(
                pred_residual=pred_residual,
                checkpoint=ckpt,
                source_penalty_names=source_penalty_names,
                target_penalty_names=penalty_names,
                strict=bool(ft_cfg.get("strict_pred_residual", True)),
            )
            if not loaded_pred_residual_state:
                raise ValueError(f"Fine-tune checkpoint is missing pred_residual_state: {ckpt_path}")

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
            "loaded_pred_residual": bool(loaded_pred_residual_state),
        }
        print(f"Fine-tune warm start loaded from: {ckpt_path}")
        print(f"Fine-tune target->source cluster map: {finetune_summary['target_to_source_cluster']}")

    apply_finetune_warm_start()

    freeze_backbone = bool(moe_cfg.get("freeze_backbone", cfg.get("train", {}).get("freeze_backbone", False)))
    frozen_backbone_params = 0
    if freeze_backbone:
        frozen_backbone_params = _freeze_module_params(model)
        print(f"Backbone frozen for MoE training: params={frozen_backbone_params}")
    pred_residual_train_with_eval_anchors = (
        pred_residual is not None
        and bool(pred_residual_cfg.get("train_with_eval_anchors", bool(freeze_backbone)))
    )
    if pred_residual_train_with_eval_anchors:
        print(
            "Prediction residual training uses the same MoE anchor post-processing modules as eval "
            "(train-side anchor scales are selected on train only)."
        )
    raw_moe_weight_decay = moe_cfg.get("weight_decay", None)
    if raw_moe_weight_decay is None:
        raw_moe_weight_decay = moe_cfg.get("optimizer_weight_decay", None)
    if raw_moe_weight_decay is not None:
        moe_weight_decay = float(raw_moe_weight_decay)
    else:
        moe_weight_decay = None

    cluster_params = []
    cluster_param_groups = []
    stage2_trainable_param_counts = []
    for k in range(K):
        base_params_k = []
        gate_params_k = []
        pred_residual_params_k = []
        dynamic_lambda_params_k = []
        learnable_lambda_params_k = []
        if not freeze_backbone:
            base_params_k.extend(model.get_cluster_params(k))
        if not (bilevel_enable and bilevel_optimize_gate):
            gate_params_k.extend(_gate_cluster_params(gate, k))
        if pred_residual is not None:
            pred_residual_params_k.extend(pred_residual.get_cluster_params(k))
        if dynamic_lambda is not None and (not bilevel_enable):
            dynamic_lambda_params_k.extend(dynamic_lambda.get_cluster_params(k))
        if learnable_lambda is not None and (not bilevel_enable):
            learnable_lambda_params_k.append(learnable_lambda.raw[k])
        if stage2_loss_audit_enable:
            stage2_trainable_param_counts.append(
                {
                    "cluster_id": int(k),
                    "backbone": int(sum(param.numel() for param in base_params_k)),
                    "gate": int(sum(param.numel() for param in gate_params_k)),
                    "pred_residual": int(sum(param.numel() for param in pred_residual_params_k)),
                    "dynamic_lambda": int(sum(param.numel() for param in dynamic_lambda_params_k)),
                    "learnable_lambda": int(sum(param.numel() for param in learnable_lambda_params_k)),
                }
            )
        param_groups_k = _make_cluster_optimizer_param_groups(
            base_params=base_params_k,
            gate_params=gate_params_k,
            pred_residual_params=pred_residual_params_k,
            dynamic_lambda_params=dynamic_lambda_params_k,
            learnable_lambda_params=learnable_lambda_params_k,
            base_weight_decay=float(cfg["train"]["weight_decay"]),
            moe_weight_decay=moe_weight_decay,
            pred_residual_weight_decay=pred_residual_weight_decay,
        )
        params_k = [param for group in param_groups_k for param in group["params"]]
        cluster_params.append(params_k)
        cluster_param_groups.append(param_groups_k)
    stage2_trainable_parameter_groups = None
    if stage2_loss_audit_enable:
        totals = {
            "backbone": int(sum(row["backbone"] for row in stage2_trainable_param_counts)),
            "gate": int(sum(row["gate"] for row in stage2_trainable_param_counts)),
            "pred_residual": int(sum(row["pred_residual"] for row in stage2_trainable_param_counts)),
            "dynamic_lambda": int(sum(row["dynamic_lambda"] for row in stage2_trainable_param_counts)),
            "learnable_lambda": int(sum(row["learnable_lambda"] for row in stage2_trainable_param_counts)),
        }
        stage2_trainable_parameter_groups = {
            "total": totals,
            "per_cluster": stage2_trainable_param_counts,
        }

    optimizers = [
        torch.optim.Adam(
            param_groups_k,
            lr=float(cfg["train"]["lr"]),
        )
        for param_groups_k in cluster_param_groups
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
            feat_bkf = _build_gate_routing_features(x, yhat, cluster_id_c, K, mode=gate_feature_mode)
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
    def collect_pred_residual_summary(loader: DataLoader, eval_start: int = 0) -> Dict[str, object]:
        cfg_summary = {
            "enabled": bool(pred_residual is not None),
            "specialization_weight": float(pred_residual_specialization_weight),
            "norm_weight": float(pred_residual_norm_weight),
            "intervention_weight": float(pred_residual_intervention_weight),
            "candidate_supervision_weight": float(pred_residual_candidate_supervision_weight),
            "candidate_supervision_loss": str(pred_residual_candidate_supervision_loss),
            "candidate_supervision_min_abs_improvement": float(pred_residual_candidate_supervision_min_abs),
            "candidate_supervision_min_rel_improvement": float(pred_residual_candidate_supervision_min_rel),
            "candidate_supervision_only_allowed": bool(pred_residual_candidate_supervision_only_allowed),
            "candidate_supervision_include_intervention": bool(pred_residual_candidate_supervision_include_intervention),
            "candidate_supervision_include_selector": bool(pred_residual_candidate_supervision_include_selector),
            "ignore_skip_during_training": bool(pred_residual_ignore_skip_during_training),
            "intervention_supervision_weight": float(pred_residual_intervention_supervision_weight),
            "intervention_supervision_min_gain": float(pred_residual_intervention_supervision_min_gain),
            "intervention_supervision_pos_weight": float(pred_residual_intervention_supervision_pos_weight),
            "intervention_supervision_only_allowed": bool(pred_residual_intervention_supervision_only_allowed),
            "route_ce_supervision_weight": float(route_ce_weight),
            "route_ce_supervision_min_abs_improvement": float(route_ce_min_abs_improvement),
            "route_ce_supervision_min_rel_improvement": float(route_ce_min_rel_improvement),
            "route_ce_supervision_min_candidate_delta_rms": float(route_ce_min_candidate_delta_rms),
            "route_ce_supervision_ignore_abs_gain_below": float(route_ce_ignore_abs_gain_below),
            "route_ce_supervision_class_weight": str(route_ce_class_weight_mode),
            "route_ce_supervision_max_class_weight": float(route_ce_max_class_weight),
            "binary_adoption_supervision_weight": float(binary_adoption_weight),
            "binary_adoption_supervision_min_abs_improvement": float(binary_adoption_min_abs_improvement),
            "binary_adoption_supervision_min_rel_improvement": float(binary_adoption_min_rel_improvement),
            "binary_adoption_supervision_min_candidate_delta_rms": float(binary_adoption_min_candidate_delta_rms),
            "binary_adoption_supervision_ignore_abs_gain_below": float(binary_adoption_ignore_abs_gain_below),
            "binary_adoption_supervision_positive_weight": float(binary_adoption_positive_weight),
            "binary_adoption_supervision_negative_weight": float(binary_adoption_negative_weight),
            "route_rate_alignment_supervision_weight": float(route_rate_alignment_weight),
            "route_rate_alignment_supervision_min_abs_improvement": float(route_rate_alignment_min_abs_improvement),
            "route_rate_alignment_supervision_min_rel_improvement": float(route_rate_alignment_min_rel_improvement),
            "route_rate_alignment_supervision_min_candidate_delta_rms": float(route_rate_alignment_min_candidate_delta_rms),
            "route_rate_alignment_supervision_ignore_abs_gain_below": float(route_rate_alignment_ignore_abs_gain_below),
            "route_positive_recall_supervision_weight": float(route_positive_recall_weight),
            "route_positive_recall_supervision_min_abs_improvement": float(route_positive_recall_min_abs_improvement),
            "route_positive_recall_supervision_min_rel_improvement": float(route_positive_recall_min_rel_improvement),
            "route_positive_recall_supervision_min_candidate_delta_rms": float(route_positive_recall_min_candidate_delta_rms),
            "route_positive_recall_supervision_ignore_abs_gain_below": float(route_positive_recall_ignore_abs_gain_below),
            "route_positive_recall_supervision_mode": str(route_positive_recall_mode),
            "route_positive_recall_supervision_target_probability": float(route_positive_recall_target_probability),
            "route_precision_recall_supervision_weight": float(route_precision_recall_weight),
            "route_precision_recall_supervision_min_abs_improvement": float(route_precision_recall_min_abs_improvement),
            "route_precision_recall_supervision_min_rel_improvement": float(route_precision_recall_min_rel_improvement),
            "route_precision_recall_supervision_min_candidate_delta_rms": float(route_precision_recall_min_candidate_delta_rms),
            "route_precision_recall_supervision_ignore_abs_gain_below": float(route_precision_recall_ignore_abs_gain_below),
            "route_precision_recall_supervision_recall_mode": str(route_precision_recall_mode),
            "route_precision_recall_supervision_recall_target_probability": float(route_precision_recall_target_probability),
            "route_precision_recall_supervision_false_adopt_max_probability": float(route_precision_recall_false_adopt_max_probability),
            "route_precision_recall_supervision_false_adopt_weight": float(route_precision_recall_false_adopt_weight),
            "confidence_gate_enable": bool(pred_residual_confidence_gate_enable),
            "confidence_gate_source_split": str(pred_residual_confidence_gate_source_split),
            "confidence_gate_threshold": str(pred_residual_confidence_gate_threshold),
            "confidence_gate_min_abs_improvement": float(pred_residual_confidence_gate_min_abs),
            "confidence_gate_min_rel_improvement": float(pred_residual_confidence_gate_min_rel),
            "confidence_gate_min_precision": float(pred_residual_confidence_gate_min_precision),
            "confidence_gate_max_pred_positive_rate": (
                None
                if pred_residual_confidence_gate_max_pred_rate is None
                else float(pred_residual_confidence_gate_max_pred_rate)
            ),
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
        delta_numel = 0
        base_sq_sum = 0.0
        spec_sum_k = torch.zeros(K, device=device)
        norm_sum_k = torch.zeros(K, device=device)
        intervention_sum_k = torch.zeros(K, device=device)
        selected_intervention_sum_p = torch.zeros(P, device=device)
        route_sum_p = torch.zeros(P, device=device)
        effective_route_sum_p = torch.zeros(P, device=device)
        route_numel = 0
        cnt = 0

        for x, y, idx in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            idx = idx.to(device=device, dtype=torch.long, non_blocking=True)
            query_start_abs_b = int(eval_start) + idx
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
            feat_bkf = _build_gate_routing_features(x, y_base, cluster_id_c, K, mode=gate_feature_mode)
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
                query_start_abs_b=query_start_abs_b,
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
            delta_numel += int(delta.numel())
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
        if stage2_loss_audit_enable:
            cfg_summary["residual_delta_rms"] = float((delta_sq_sum / max(delta_numel, 1)) ** 0.5)
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
        lambda_feat_bkf = scatter_mean_bcf_to_bkf(feat_bcf, cluster_id_c, K)
        gate_feat_bkf = _build_gate_routing_features(x, yhat_base, cluster_id_c, K, mode=gate_feature_mode)
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
                gate_feat_bkf,
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
                skip_bk=_pred_residual_training_skip_arg(
                    skip_bk=skip_bk,
                    allow_skip=allow_skip,
                    ignore_skip_during_training=pred_residual_ignore_skip_during_training,
                ),
            )
            yhat_residual_raw = pred_out["y_final"]
            yhat = yhat_residual_raw
        else:
            yhat_residual_raw = yhat_base
            yhat = yhat_base
        if pred_residual_train_with_eval_anchors:
            yhat = apply_moe_output_anchor_experts(
                yhat,
                base_pred_bch=yhat_base,
                x_bcl=x,
                query_start_abs_b=idx,
                input_len=L,
                moe_cfg=moe_cfg,
                moe_enable=moe_enable,
                observed_history_tc=data_window_tc,
                train_stat_anchor_pc=train_stat_anchor_pc,
                train_residual_anchor_phc=train_residual_anchor_phc,
            )

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
                feat_bkf=lambda_feat_bkf,
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
            y_final=yhat_residual_raw,
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
        candidate_supervision_loss_bk = None
        if pred_residual_candidate_supervision_weight > 0.0:
            candidate_supervision_loss_bk = _pred_residual_candidate_supervision_loss(
                y_base_bch=yhat_base,
                pred_out=pred_out,
                y_bch=y,
                cluster_id_c=cluster_id_c,
                K=K,
                penalty_names=penalty_names,
                penalty_fns=penalty_fns,
                penalty_scale=penalty_scale,
                allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                only_allowed=pred_residual_candidate_supervision_only_allowed,
                loss_kind=pred_residual_candidate_supervision_loss,
                min_abs_improvement=pred_residual_candidate_supervision_min_abs,
                min_rel_improvement=pred_residual_candidate_supervision_min_rel,
                include_intervention=pred_residual_candidate_supervision_include_intervention,
                include_selector=pred_residual_candidate_supervision_include_selector,
                apply_output_anchors=pred_residual_train_with_eval_anchors,
                x_bcl=x,
                query_start_abs_b=idx,
                input_len=L,
                moe_cfg=moe_cfg,
                moe_enable=moe_enable,
                observed_history_tc=data_window_tc,
                train_stat_anchor_pc=train_stat_anchor_pc,
                train_residual_anchor_phc=train_residual_anchor_phc,
            )
        intervention_supervision_loss_bk = None
        if pred_residual_intervention_supervision_weight > 0.0:
            intervention_supervision_loss_bk = _pred_residual_intervention_supervision_loss(
                y_base_bch=yhat_base,
                pred_out=pred_out,
                y_bch=y,
                cluster_id_c=cluster_id_c,
                K=K,
                allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                only_allowed=pred_residual_intervention_supervision_only_allowed,
                min_gain=pred_residual_intervention_supervision_min_gain,
                pos_weight=pred_residual_intervention_supervision_pos_weight,
                apply_output_anchors=pred_residual_train_with_eval_anchors,
                x_bcl=x,
                query_start_abs_b=idx,
                input_len=L,
                moe_cfg=moe_cfg,
                moe_enable=moe_enable,
                observed_history_tc=data_window_tc,
                train_stat_anchor_pc=train_stat_anchor_pc,
                train_residual_anchor_phc=train_residual_anchor_phc,
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
        if candidate_supervision_loss_bk is not None:
            objective_loss_bk = (
                objective_loss_bk
                + pred_residual_candidate_supervision_weight * candidate_supervision_loss_bk
            )
        if intervention_supervision_loss_bk is not None:
            objective_loss_bk = (
                objective_loss_bk
                + pred_residual_intervention_supervision_weight * intervention_supervision_loss_bk
            )
        utility_base_bch = None
        utility_cand_bcpH = None
        if (
            route_ce_weight > 0.0
            or binary_adoption_weight > 0.0
            or route_rate_alignment_weight > 0.0
            or route_positive_recall_weight > 0.0
            or route_precision_recall_weight > 0.0
            or mse_utility_gate_weight > 0.0
        ):
            utility_base_bch, utility_cand_bcpH = _pred_residual_candidates_on_eval_path(
                yhat_base,
                pred_out,
                apply_output_anchors=pred_residual_train_with_eval_anchors,
                x_bcl=x,
                query_start_abs_b=idx,
                input_len=L,
                moe_cfg=moe_cfg,
                moe_enable=moe_enable,
                observed_history_tc=data_window_tc,
                train_stat_anchor_pc=train_stat_anchor_pc,
                train_residual_anchor_phc=train_residual_anchor_phc,
            )
        if route_ce_weight > 0.0 and utility_cand_bcpH is not None:
            route_labels_bk, route_gain_bk = _cluster_route_oracle_labels_and_gain_from_candidates(
                base_bch=utility_base_bch,
                cand_bcpH=utility_cand_bcpH,
                y_bch=y,
                cluster_id_c=cluster_id_c,
                K=K,
                allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                min_abs_improvement=route_ce_min_abs_improvement,
                min_rel_improvement=route_ce_min_rel_improvement,
                min_candidate_delta_rms=route_ce_min_candidate_delta_rms,
            )
            route_ce_active_mask_bk = None
            if route_ce_ignore_abs_gain_below > 0.0:
                route_ce_active_mask_bk = _route_ce_active_mask_from_gain(
                    route_gain_bk,
                    ignore_abs_gain_below=route_ce_ignore_abs_gain_below,
                )
            route_ce_loss_bk = _route_ce_loss_from_probs(
                probs_bkp=probs_bkp,
                skip_prob_bk=skip_prob_bk if allow_skip else None,
                labels_bk=route_labels_bk,
                probs_include_skip_mass=bool(skip_competes),
                class_weight_q=_route_ce_class_weight_from_labels(
                    labels_bk=route_labels_bk,
                    num_classes=P + 1,
                    mode=route_ce_class_weight_mode,
                    max_weight=route_ce_max_class_weight,
                    active_mask_bk=route_ce_active_mask_bk,
                ),
            )
            if route_ce_active_mask_bk is not None:
                route_ce_loss_bk = route_ce_loss_bk * route_ce_active_mask_bk.to(dtype=route_ce_loss_bk.dtype)
            objective_loss_bk = objective_loss_bk + route_ce_weight * route_ce_loss_bk
        if binary_adoption_weight > 0.0 and utility_cand_bcpH is not None:
            binary_labels_bk, binary_gain_bk = _cluster_route_oracle_labels_and_gain_from_candidates(
                base_bch=utility_base_bch,
                cand_bcpH=utility_cand_bcpH,
                y_bch=y,
                cluster_id_c=cluster_id_c,
                K=K,
                allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                min_abs_improvement=binary_adoption_min_abs_improvement,
                min_rel_improvement=binary_adoption_min_rel_improvement,
                min_candidate_delta_rms=binary_adoption_min_candidate_delta_rms,
            )
            binary_active_mask_bk = None
            if binary_adoption_ignore_abs_gain_below > 0.0:
                binary_active_mask_bk = _route_ce_active_mask_from_gain(
                    binary_gain_bk,
                    ignore_abs_gain_below=binary_adoption_ignore_abs_gain_below,
                )
            binary_loss_bk = _route_binary_adoption_loss_from_probs(
                probs_bkp=probs_bkp,
                skip_prob_bk=skip_prob_bk if allow_skip else None,
                labels_bk=binary_labels_bk,
                probs_include_skip_mass=bool(skip_competes),
                allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                active_mask_bk=binary_active_mask_bk,
                positive_weight=binary_adoption_positive_weight,
                negative_weight=binary_adoption_negative_weight,
            )
            if binary_loss_bk is not None:
                objective_loss_bk = objective_loss_bk + binary_adoption_weight * binary_loss_bk
        if route_rate_alignment_weight > 0.0 and utility_cand_bcpH is not None:
            rate_labels_bk, rate_gain_bk = _cluster_route_oracle_labels_and_gain_from_candidates(
                base_bch=utility_base_bch,
                cand_bcpH=utility_cand_bcpH,
                y_bch=y,
                cluster_id_c=cluster_id_c,
                K=K,
                allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                min_abs_improvement=route_rate_alignment_min_abs_improvement,
                min_rel_improvement=route_rate_alignment_min_rel_improvement,
                min_candidate_delta_rms=route_rate_alignment_min_candidate_delta_rms,
            )
            rate_active_mask_bk = None
            if route_rate_alignment_ignore_abs_gain_below > 0.0:
                rate_active_mask_bk = _route_ce_active_mask_from_gain(
                    rate_gain_bk,
                    ignore_abs_gain_below=route_rate_alignment_ignore_abs_gain_below,
                )
            rate_loss_bk = _route_rate_alignment_loss_from_probs(
                probs_bkp=probs_bkp,
                skip_prob_bk=skip_prob_bk if allow_skip else None,
                labels_bk=rate_labels_bk,
                probs_include_skip_mass=bool(skip_competes),
                allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                active_mask_bk=rate_active_mask_bk,
            )
            objective_loss_bk = objective_loss_bk + route_rate_alignment_weight * rate_loss_bk
        if route_positive_recall_weight > 0.0 and utility_cand_bcpH is not None:
            recall_labels_bk, recall_gain_bk = _cluster_route_oracle_labels_and_gain_from_candidates(
                base_bch=utility_base_bch,
                cand_bcpH=utility_cand_bcpH,
                y_bch=y,
                cluster_id_c=cluster_id_c,
                K=K,
                allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                min_abs_improvement=route_positive_recall_min_abs_improvement,
                min_rel_improvement=route_positive_recall_min_rel_improvement,
                min_candidate_delta_rms=route_positive_recall_min_candidate_delta_rms,
            )
            recall_active_mask_bk = None
            if route_positive_recall_ignore_abs_gain_below > 0.0:
                recall_active_mask_bk = _route_ce_active_mask_from_gain(
                    recall_gain_bk,
                    ignore_abs_gain_below=route_positive_recall_ignore_abs_gain_below,
                )
            recall_loss_bk = _route_positive_recall_loss_from_probs(
                probs_bkp=probs_bkp,
                skip_prob_bk=skip_prob_bk if allow_skip else None,
                labels_bk=recall_labels_bk,
                probs_include_skip_mass=bool(skip_competes),
                active_mask_bk=recall_active_mask_bk,
                mode=route_positive_recall_mode,
                target_probability=route_positive_recall_target_probability,
            )
            objective_loss_bk = objective_loss_bk + route_positive_recall_weight * recall_loss_bk
        if route_precision_recall_weight > 0.0 and utility_cand_bcpH is not None:
            precision_labels_bk, precision_gain_bk = _cluster_route_oracle_labels_and_gain_from_candidates(
                base_bch=utility_base_bch,
                cand_bcpH=utility_cand_bcpH,
                y_bch=y,
                cluster_id_c=cluster_id_c,
                K=K,
                allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                min_abs_improvement=route_precision_recall_min_abs_improvement,
                min_rel_improvement=route_precision_recall_min_rel_improvement,
                min_candidate_delta_rms=route_precision_recall_min_candidate_delta_rms,
            )
            precision_active_mask_bk = None
            if route_precision_recall_ignore_abs_gain_below > 0.0:
                precision_active_mask_bk = _route_ce_active_mask_from_gain(
                    precision_gain_bk,
                    ignore_abs_gain_below=route_precision_recall_ignore_abs_gain_below,
                )
            precision_loss_bk = _route_precision_constrained_recall_loss_from_probs(
                probs_bkp=probs_bkp,
                skip_prob_bk=skip_prob_bk if allow_skip else None,
                labels_bk=precision_labels_bk,
                probs_include_skip_mass=bool(skip_competes),
                active_mask_bk=precision_active_mask_bk,
                recall_mode=route_precision_recall_mode,
                recall_target_probability=route_precision_recall_target_probability,
                false_adopt_max_probability=route_precision_recall_false_adopt_max_probability,
                false_adopt_weight=route_precision_recall_false_adopt_weight,
            )
            objective_loss_bk = objective_loss_bk + route_precision_recall_weight * precision_loss_bk
        if mse_utility_gate_weight > 0.0:
            mse_gate_loss_bk = _mse_utility_gate_supervision_loss(
                probs_bkp=probs_bkp,
                skip_prob_bk=skip_prob_bk if allow_skip else None,
                allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                y_base_bch=yhat_base,
                pred_out=pred_out,
                y_bch=y,
                cluster_id_c=cluster_id_c,
                K=K,
                y_base_eval_bch=utility_base_bch,
                cand_eval_bcpH=utility_cand_bcpH,
                temperature=mse_utility_gate_temperature,
                min_gain=mse_utility_gate_min_gain,
                target_power=mse_utility_gate_target_power,
                include_skip=mse_utility_gate_include_skip,
                probs_include_skip_mass=bool(skip_competes),
                target_mode=mse_utility_gate_target_mode,
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
            "candidate_supervision_loss_bk": candidate_supervision_loss_bk,
            "intervention_supervision_loss_bk": intervention_supervision_loss_bk,
        }

    def _store_anchor_scale_selection(
        anchor_cfg: dict,
        anchor_summary: Dict[str, object],
        scale_selection_cfg: dict,
        scales_c: torch.Tensor,
        scores_c: torch.Tensor,
        selection_count: int,
        *,
        source_split: str,
        score_key: str,
        default_metric: str,
        default_max_scale: float,
        default_steps: int,
        horizon_segments: int,
    ) -> None:
        if int(scales_c.ndim) == 2:
            alpha_values = [[float(v) for v in row] for row in scales_c.tolist()]
            anchor_cfg["alpha_by_channel_horizon"] = alpha_values
            anchor_cfg["alpha_horizon_segments"] = int(horizon_segments)
            alpha_key = "alpha_by_channel_horizon"
        else:
            alpha_values = [float(v) for v in scales_c.tolist()]
            anchor_cfg["alpha_by_channel"] = alpha_values
            alpha_key = "alpha_by_channel"
        score_payload = (
            [[float(v) for v in row] for row in scores_c.tolist()]
            if int(scores_c.ndim) == 2
            else [float(v) for v in scores_c.tolist()]
        )
        anchor_summary["scale_selection"] = {
            "enable": True,
            "source_split": source_split,
            "metric": str(scale_selection_cfg.get("metric", default_metric)),
            "max_scale": float(scale_selection_cfg.get("max_scale", default_max_scale)),
            "steps": int(scale_selection_cfg.get("steps", default_steps)),
            "horizon_segments": int(horizon_segments),
            "num_windows": int(selection_count),
            alpha_key: alpha_values,
            score_key: score_payload,
            "mean_alpha": float(scales_c.mean().item()) if int(scales_c.numel()) > 0 else 0.0,
        }

    def _prepare_train_anchors_for_pred_residual_training() -> None:
        nonlocal train_residual_anchor_phc
        if not pred_residual_train_with_eval_anchors:
            return
        if len(dtr) <= 0:
            return
        if bool(train_stat_anchor_cfg.get("enable", False)):
            stat_scale_cfg = train_stat_anchor_cfg.get("scale_selection", {}) or {}
            if bool(stat_scale_cfg.get("enable", False)) and train_stat_anchor_pc is not None:
                horizon_segments = int(stat_scale_cfg.get("horizon_segments", 1))
                scales_c, scores_c, selection_count = select_train_stat_anchor_scales_from_loader(
                    model=model,
                    loader=dl_tr,
                    cluster_id_c=cluster_id_c,
                    device=device,
                    history_anchor_cfg=history_anchor_cfg,
                    observed_history_tc=data_window_tc,
                    input_len=L,
                    eval_start=0,
                    stat_anchor_pc=train_stat_anchor_pc,
                    train_stat_anchor_cfg=train_stat_anchor_cfg,
                    metric=str(stat_scale_cfg.get("metric", "mse")),
                    max_scale=float(stat_scale_cfg.get("max_scale", 0.3)),
                    steps=int(stat_scale_cfg.get("steps", 13)),
                    horizon_segments=horizon_segments,
                    model_train_stat_adapter_pc=model_train_stat_adapter_pc,
                    model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
                )
                _store_anchor_scale_selection(
                    train_stat_anchor_cfg,
                    train_stat_anchor_summary,
                    stat_scale_cfg,
                    scales_c,
                    scores_c,
                    selection_count,
                    source_split="train_pretrain_for_pred_residual",
                    score_key="score",
                    default_metric="mse",
                    default_max_scale=0.3,
                    default_steps=13,
                    horizon_segments=horizon_segments,
                )
                print(
                    "Preselected train-stat anchor scales for pred residual training: "
                    "source=train, "
                    f"mean_alpha={train_stat_anchor_summary['scale_selection']['mean_alpha']:.4f}"
                )
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
                    "source_split": "train_pretrain_for_pred_residual",
                    "train_windows": int(residual_train_count),
                    "min_count": int(train_residual_anchor_counts.min().item()),
                    "max_count": int(train_residual_anchor_counts.max().item()),
                    "alpha": float(train_residual_anchor_cfg.get("alpha", 0.0) or 0.0),
                    "blend_target": str(train_residual_anchor_cfg.get("blend_target", "prediction")),
                }
            )
            residual_scale_cfg = train_residual_anchor_cfg.get("scale_selection", {}) or {}
            if bool(residual_scale_cfg.get("enable", False)):
                horizon_segments = int(residual_scale_cfg.get("horizon_segments", 1))
                scales_c, scores_c, selection_count = select_train_residual_anchor_scales_from_loader(
                    model=model,
                    loader=dl_tr,
                    cluster_id_c=cluster_id_c,
                    device=device,
                    history_anchor_cfg=history_anchor_cfg,
                    observed_history_tc=data_window_tc,
                    input_len=L,
                    eval_start=0,
                    residual_anchor_phc=train_residual_anchor_phc,
                    train_residual_anchor_cfg=train_residual_anchor_cfg,
                    metric=str(residual_scale_cfg.get("metric", "mse")),
                    max_scale=float(residual_scale_cfg.get("max_scale", 0.5)),
                    steps=int(residual_scale_cfg.get("steps", 21)),
                    horizon_segments=horizon_segments,
                    model_train_stat_adapter_pc=model_train_stat_adapter_pc,
                    model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
                    train_stat_anchor_pc=train_stat_anchor_pc,
                    train_stat_anchor_cfg=train_stat_anchor_cfg,
                )
                _store_anchor_scale_selection(
                    train_residual_anchor_cfg,
                    train_residual_anchor_summary,
                    residual_scale_cfg,
                    scales_c,
                    scores_c,
                    selection_count,
                    source_split="train_pretrain_for_pred_residual",
                    score_key="score_by_channel",
                    default_metric="mse",
                    default_max_scale=0.5,
                    default_steps=21,
                    horizon_segments=horizon_segments,
                )
                print(
                    "Preselected train-residual anchor scales for pred residual training: "
                    "source=train, "
                    f"mean_alpha={train_residual_anchor_summary['scale_selection']['mean_alpha']:.4f}"
                )

    def _prepare_phase_residual_candidate_for_pred_residual() -> None:
        if not phase_residual_candidate_enable or pred_residual is None:
            return
        table_phc, counts_p, train_windows = build_train_residual_anchor_table_from_loader(
            model=model,
            loader=dl_tr,
            cluster_id_c=cluster_id_c,
            device=device,
            history_anchor_cfg=history_anchor_cfg,
            observed_history_tc=data_window_tc,
            input_len=L,
            eval_start=0,
            period=int(phase_residual_candidate_period),
            model_train_stat_adapter_pc=model_train_stat_adapter_pc,
            model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
            train_stat_anchor_pc=train_stat_anchor_pc,
            train_stat_anchor_cfg=train_stat_anchor_cfg,
        )
        pred_residual.set_phase_residual_candidate_table(table_phc)
        phase_residual_candidate_summary.update(
            {
                "enable": True,
                "source_split": "train",
                "train_windows": int(train_windows),
                "min_count": int(counts_p.min().item()),
                "max_count": int(counts_p.max().item()),
                "table_shape": [int(v) for v in table_phc.shape],
                "train_only": True,
                "output_anchor_enabled": False,
            }
        )
        print(
            "Prediction residual phase candidate table built: "
            f"names={phase_residual_candidate_names}, period={int(phase_residual_candidate_period)}, "
            f"train_windows={int(train_windows)}, "
            f"min_count={int(counts_p.min().item())}, max_count={int(counts_p.max().item())}"
        )

    _prepare_train_anchors_for_pred_residual_training()
    _prepare_phase_residual_candidate_for_pred_residual()

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
    mse_gate_train_diag_history: List[Dict[str, object]] = []
    stage2_loss_audit_history: List[Dict[str, object]] = []
    stage2_route_audit_history: List[Dict[str, object]] = []
    stage2_route_audit_frequency = max(1, int(stage2_route_audit_cfg.get("frequency_epochs", 1)))

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
        if stage2_loss_audit_enable:
            stage2_total_loss_sum_k = torch.zeros(K, device=device)
            stage2_forecast_loss_sum_k = torch.zeros(K, device=device)
            stage2_penalty_loss_sum_k = torch.zeros(K, device=device)
            stage2_pred_residual_aux_loss_sum_k = torch.zeros(K, device=device)
            stage2_candidate_supervision_loss_sum_k = torch.zeros(K, device=device)
            stage2_gate_utility_loss_sum_k = torch.zeros(K, device=device)
            stage2_skip_noop_loss_sum_k = torch.zeros(K, device=device)
            stage2_intervention_supervision_loss_sum_k = torch.zeros(K, device=device)
            stage2_other_aux_loss_sum_k = torch.zeros(K, device=device)
            stage2_route_prob_sum_kp = torch.zeros(K, P, device=device)
            stage2_route_actual_sum_kp = torch.zeros(K, P, device=device)
            stage2_route_entropy_sum_k = torch.zeros(K, device=device)
            stage2_route_count_k = torch.zeros(K, device=device)
            stage2_skip_prob_sum_k = torch.zeros(K, device=device)
            stage2_skip_active_sum_k = torch.zeros(K, device=device)
            stage2_grad_norm_sum = {
                "backbone": 0.0,
                "gate": 0.0,
                "pred_residual": 0.0,
                "dynamic_lambda": 0.0,
                "learnable_lambda": 0.0,
            }
            stage2_grad_norm_batches = 0
        mse_gate_loss_sum_k = torch.zeros(K, device=device)
        mse_gate_valid_sum_k = torch.zeros(K, device=device)
        mse_gate_skip_target_sum_k = torch.zeros(K, device=device)
        mse_gate_skip_prob_sum_k = torch.zeros(K, device=device)
        mse_gate_best_gain_sum_k = torch.zeros(K, device=device)
        mse_gate_diag_count_k = torch.zeros(K, device=device)
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
            feat_bkf = scatter_mean_bcf_to_bkf(feat_bcf, cluster_id_c, K)  # [B,K,F] for dynamic lambda
            gate_feat_bkf = _build_gate_routing_features(x, yhat_base, cluster_id_c, K, mode=gate_feature_mode)
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
                    gate_feat_bkf,
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
                if stage2_loss_audit_enable:
                    with torch.no_grad():
                        probs_det = probs_bkp.detach()
                        probs_safe = probs_det.clamp_min(1.0e-8)
                        stage2_route_prob_sum_kp += probs_det.sum(dim=0)
                        stage2_route_actual_sum_kp += mask_bkp.detach().sum(dim=0)
                        stage2_route_entropy_sum_k += (-(probs_safe * probs_safe.log()).sum(dim=-1)).sum(dim=0)
                        stage2_route_count_k += probs_det.shape[0]
                        if allow_skip and skip_prob_bk is not None and skip_bk is not None:
                            stage2_skip_prob_sum_k += skip_prob_bk.detach().sum(dim=0)
                            stage2_skip_active_sum_k += skip_bk.detach().sum(dim=0)
            else:
                mask_bkp = torch.zeros_like(route_pen_bkp)

            if pred_residual is not None and moe_enable and P > 0:
                pred_out = pred_residual(
                    x,
                    yhat_base,
                    cluster_id_c,
                    mask_bkp,
                    skip_bk=_pred_residual_training_skip_arg(
                        skip_bk=skip_bk,
                        allow_skip=allow_skip,
                        ignore_skip_during_training=pred_residual_ignore_skip_during_training,
                    ),
                    query_start_abs_b=idx,
                )
                yhat_residual_raw = pred_out["y_final"]
                yhat = yhat_residual_raw
            else:
                yhat_residual_raw = yhat_base
                yhat = yhat_base
            if pred_residual_train_with_eval_anchors:
                yhat = apply_moe_output_anchor_experts(
                    yhat,
                    base_pred_bch=yhat_base,
                    x_bcl=x,
                    query_start_abs_b=idx,
                    input_len=L,
                    moe_cfg=moe_cfg,
                    moe_enable=moe_enable,
                    observed_history_tc=data_window_tc,
                    train_stat_anchor_pc=train_stat_anchor_pc,
                    train_residual_anchor_phc=train_residual_anchor_phc,
                )

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

            mse_gate_loss_bk = None
            mse_gate_diag = None
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
                    y_final=yhat_residual_raw,
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
                candidate_supervision_loss_bk = None
                if pred_residual_candidate_supervision_weight > 0.0:
                    candidate_supervision_loss_bk = _pred_residual_candidate_supervision_loss(
                        y_base_bch=yhat_base,
                        pred_out=pred_out,
                        y_bch=y,
                        cluster_id_c=cluster_id_c,
                        K=K,
                        penalty_names=penalty_names,
                        penalty_fns=penalty_fns,
                        penalty_scale=penalty_scale,
                        allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                        only_allowed=pred_residual_candidate_supervision_only_allowed,
                        loss_kind=pred_residual_candidate_supervision_loss,
                        min_abs_improvement=pred_residual_candidate_supervision_min_abs,
                        min_rel_improvement=pred_residual_candidate_supervision_min_rel,
                        include_intervention=pred_residual_candidate_supervision_include_intervention,
                        include_selector=pred_residual_candidate_supervision_include_selector,
                        apply_output_anchors=pred_residual_train_with_eval_anchors,
                        x_bcl=x,
                        query_start_abs_b=idx,
                        input_len=L,
                        moe_cfg=moe_cfg,
                        moe_enable=moe_enable,
                        observed_history_tc=data_window_tc,
                        train_stat_anchor_pc=train_stat_anchor_pc,
                        train_residual_anchor_phc=train_residual_anchor_phc,
                    )
                intervention_supervision_loss_bk = None
                if pred_residual_intervention_supervision_weight > 0.0:
                    intervention_supervision_loss_bk = _pred_residual_intervention_supervision_loss(
                        y_base_bch=yhat_base,
                        pred_out=pred_out,
                        y_bch=y,
                        cluster_id_c=cluster_id_c,
                        K=K,
                        allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                        only_allowed=pred_residual_intervention_supervision_only_allowed,
                        min_gain=pred_residual_intervention_supervision_min_gain,
                        pos_weight=pred_residual_intervention_supervision_pos_weight,
                        apply_output_anchors=pred_residual_train_with_eval_anchors,
                        x_bcl=x,
                        query_start_abs_b=idx,
                        input_len=L,
                        moe_cfg=moe_cfg,
                        moe_enable=moe_enable,
                        observed_history_tc=data_window_tc,
                        train_stat_anchor_pc=train_stat_anchor_pc,
                        train_residual_anchor_phc=train_residual_anchor_phc,
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
                forecast_loss_component_bk = (
                    (mse_weight * loss_terms_bk["mse"])
                    + _apply_mae_objective_weight(loss_terms_bk["mae_objective"], mae_objective_weight_ep)
                )
                penalty_loss_component_bk = loss_terms_bk["penalty"]
                pred_residual_aux_component_bk = loss_terms_bk["pred_residual"]
                candidate_supervision_component_bk = torch.zeros_like(mse_bk)
                intervention_supervision_component_bk = torch.zeros_like(mse_bk)
                skip_noop_component_bk = torch.zeros_like(mse_bk)
                gate_utility_component_bk = torch.zeros_like(mse_bk)
                objective_loss_bk = (
                    forecast_loss_component_bk
                    + penalty_loss_component_bk
                )
                loss_bk = objective_loss_bk + pred_residual_aux_component_bk
                if candidate_supervision_loss_bk is not None:
                    candidate_supervision_component_bk = (
                        pred_residual_candidate_supervision_weight * candidate_supervision_loss_bk
                    )
                    loss_bk = loss_bk + candidate_supervision_component_bk
                if intervention_supervision_loss_bk is not None:
                    intervention_supervision_component_bk = (
                        pred_residual_intervention_supervision_weight * intervention_supervision_loss_bk
                    )
                    loss_bk = loss_bk + intervention_supervision_component_bk
                utility_base_bch = None
                utility_cand_bcpH = None
                if (
                    route_ce_weight > 0.0
                    or binary_adoption_weight > 0.0
                    or route_rate_alignment_weight > 0.0
                    or route_positive_recall_weight > 0.0
                    or route_precision_recall_weight > 0.0
                    or mse_utility_gate_weight > 0.0
                ):
                    utility_base_bch, utility_cand_bcpH = _pred_residual_candidates_on_eval_path(
                        yhat_base,
                        pred_out,
                        apply_output_anchors=pred_residual_train_with_eval_anchors,
                        x_bcl=x,
                        query_start_abs_b=idx,
                        input_len=L,
                        moe_cfg=moe_cfg,
                        moe_enable=moe_enable,
                        observed_history_tc=data_window_tc,
                        train_stat_anchor_pc=train_stat_anchor_pc,
                        train_residual_anchor_phc=train_residual_anchor_phc,
                    )
                if route_ce_weight > 0.0 and utility_cand_bcpH is not None:
                    route_labels_bk, route_gain_bk = _cluster_route_oracle_labels_and_gain_from_candidates(
                        base_bch=utility_base_bch,
                        cand_bcpH=utility_cand_bcpH,
                        y_bch=y,
                        cluster_id_c=cluster_id_c,
                        K=K,
                        allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                        min_abs_improvement=route_ce_min_abs_improvement,
                        min_rel_improvement=route_ce_min_rel_improvement,
                        min_candidate_delta_rms=route_ce_min_candidate_delta_rms,
                    )
                    route_ce_active_mask_bk = None
                    if route_ce_ignore_abs_gain_below > 0.0:
                        route_ce_active_mask_bk = _route_ce_active_mask_from_gain(
                            route_gain_bk,
                            ignore_abs_gain_below=route_ce_ignore_abs_gain_below,
                        )
                    route_ce_loss_bk = _route_ce_loss_from_probs(
                        probs_bkp=probs_bkp,
                        skip_prob_bk=skip_prob_bk if allow_skip else None,
                        labels_bk=route_labels_bk,
                        probs_include_skip_mass=bool(skip_competes),
                        class_weight_q=_route_ce_class_weight_from_labels(
                            labels_bk=route_labels_bk,
                            num_classes=P + 1,
                            mode=route_ce_class_weight_mode,
                            max_weight=route_ce_max_class_weight,
                            active_mask_bk=route_ce_active_mask_bk,
                        ),
                    )
                    if route_ce_active_mask_bk is not None:
                        route_ce_loss_bk = route_ce_loss_bk * route_ce_active_mask_bk.to(dtype=route_ce_loss_bk.dtype)
                    gate_utility_component_bk = gate_utility_component_bk + route_ce_weight * route_ce_loss_bk
                    loss_bk = loss_bk + route_ce_weight * route_ce_loss_bk
                if binary_adoption_weight > 0.0 and utility_cand_bcpH is not None:
                    binary_labels_bk, binary_gain_bk = _cluster_route_oracle_labels_and_gain_from_candidates(
                        base_bch=utility_base_bch,
                        cand_bcpH=utility_cand_bcpH,
                        y_bch=y,
                        cluster_id_c=cluster_id_c,
                        K=K,
                        allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                        min_abs_improvement=binary_adoption_min_abs_improvement,
                        min_rel_improvement=binary_adoption_min_rel_improvement,
                        min_candidate_delta_rms=binary_adoption_min_candidate_delta_rms,
                    )
                    binary_active_mask_bk = None
                    if binary_adoption_ignore_abs_gain_below > 0.0:
                        binary_active_mask_bk = _route_ce_active_mask_from_gain(
                            binary_gain_bk,
                            ignore_abs_gain_below=binary_adoption_ignore_abs_gain_below,
                        )
                    binary_loss_bk = _route_binary_adoption_loss_from_probs(
                        probs_bkp=probs_bkp,
                        skip_prob_bk=skip_prob_bk if allow_skip else None,
                        labels_bk=binary_labels_bk,
                        probs_include_skip_mass=bool(skip_competes),
                        allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                        active_mask_bk=binary_active_mask_bk,
                        positive_weight=binary_adoption_positive_weight,
                        negative_weight=binary_adoption_negative_weight,
                    )
                    if binary_loss_bk is not None:
                        binary_component_bk = binary_adoption_weight * binary_loss_bk
                        gate_utility_component_bk = gate_utility_component_bk + binary_component_bk
                        loss_bk = loss_bk + binary_component_bk
                if route_rate_alignment_weight > 0.0 and utility_cand_bcpH is not None:
                    rate_labels_bk, rate_gain_bk = _cluster_route_oracle_labels_and_gain_from_candidates(
                        base_bch=utility_base_bch,
                        cand_bcpH=utility_cand_bcpH,
                        y_bch=y,
                        cluster_id_c=cluster_id_c,
                        K=K,
                        allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                        min_abs_improvement=route_rate_alignment_min_abs_improvement,
                        min_rel_improvement=route_rate_alignment_min_rel_improvement,
                        min_candidate_delta_rms=route_rate_alignment_min_candidate_delta_rms,
                    )
                    rate_active_mask_bk = None
                    if route_rate_alignment_ignore_abs_gain_below > 0.0:
                        rate_active_mask_bk = _route_ce_active_mask_from_gain(
                            rate_gain_bk,
                            ignore_abs_gain_below=route_rate_alignment_ignore_abs_gain_below,
                        )
                    rate_loss_bk = _route_rate_alignment_loss_from_probs(
                        probs_bkp=probs_bkp,
                        skip_prob_bk=skip_prob_bk if allow_skip else None,
                        labels_bk=rate_labels_bk,
                        probs_include_skip_mass=bool(skip_competes),
                        allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                        active_mask_bk=rate_active_mask_bk,
                    )
                    rate_component_bk = route_rate_alignment_weight * rate_loss_bk
                    gate_utility_component_bk = gate_utility_component_bk + rate_component_bk
                    loss_bk = loss_bk + rate_component_bk
                if route_positive_recall_weight > 0.0 and utility_cand_bcpH is not None:
                    recall_labels_bk, recall_gain_bk = _cluster_route_oracle_labels_and_gain_from_candidates(
                        base_bch=utility_base_bch,
                        cand_bcpH=utility_cand_bcpH,
                        y_bch=y,
                        cluster_id_c=cluster_id_c,
                        K=K,
                        allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                        min_abs_improvement=route_positive_recall_min_abs_improvement,
                        min_rel_improvement=route_positive_recall_min_rel_improvement,
                        min_candidate_delta_rms=route_positive_recall_min_candidate_delta_rms,
                    )
                    recall_active_mask_bk = None
                    if route_positive_recall_ignore_abs_gain_below > 0.0:
                        recall_active_mask_bk = _route_ce_active_mask_from_gain(
                            recall_gain_bk,
                            ignore_abs_gain_below=route_positive_recall_ignore_abs_gain_below,
                        )
                    recall_loss_bk = _route_positive_recall_loss_from_probs(
                        probs_bkp=probs_bkp,
                        skip_prob_bk=skip_prob_bk if allow_skip else None,
                        labels_bk=recall_labels_bk,
                        probs_include_skip_mass=bool(skip_competes),
                        active_mask_bk=recall_active_mask_bk,
                        mode=route_positive_recall_mode,
                        target_probability=route_positive_recall_target_probability,
                    )
                    recall_component_bk = route_positive_recall_weight * recall_loss_bk
                    gate_utility_component_bk = gate_utility_component_bk + recall_component_bk
                    loss_bk = loss_bk + recall_component_bk
                if route_precision_recall_weight > 0.0 and utility_cand_bcpH is not None:
                    precision_labels_bk, precision_gain_bk = _cluster_route_oracle_labels_and_gain_from_candidates(
                        base_bch=utility_base_bch,
                        cand_bcpH=utility_cand_bcpH,
                        y_bch=y,
                        cluster_id_c=cluster_id_c,
                        K=K,
                        allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                        min_abs_improvement=route_precision_recall_min_abs_improvement,
                        min_rel_improvement=route_precision_recall_min_rel_improvement,
                        min_candidate_delta_rms=route_precision_recall_min_candidate_delta_rms,
                    )
                    precision_active_mask_bk = None
                    if route_precision_recall_ignore_abs_gain_below > 0.0:
                        precision_active_mask_bk = _route_ce_active_mask_from_gain(
                            precision_gain_bk,
                            ignore_abs_gain_below=route_precision_recall_ignore_abs_gain_below,
                        )
                    precision_loss_bk = _route_precision_constrained_recall_loss_from_probs(
                        probs_bkp=probs_bkp,
                        skip_prob_bk=skip_prob_bk if allow_skip else None,
                        labels_bk=precision_labels_bk,
                        probs_include_skip_mass=bool(skip_competes),
                        active_mask_bk=precision_active_mask_bk,
                        recall_mode=route_precision_recall_mode,
                        recall_target_probability=route_precision_recall_target_probability,
                        false_adopt_max_probability=route_precision_recall_false_adopt_max_probability,
                        false_adopt_weight=route_precision_recall_false_adopt_weight,
                    )
                    precision_component_bk = route_precision_recall_weight * precision_loss_bk
                    gate_utility_component_bk = gate_utility_component_bk + precision_component_bk
                    loss_bk = loss_bk + precision_component_bk
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
                            query_start_abs_b=idx,
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
                    skip_noop_component_bk = skip_supervision_weight * skip_bce_bk
                    loss_bk = loss_bk + skip_noop_component_bk
                if mse_utility_gate_weight > 0.0:
                    mse_gate_result = _mse_utility_gate_supervision_loss(
                        probs_bkp=probs_bkp,
                        skip_prob_bk=skip_prob_bk if allow_skip else None,
                        allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                        y_base_bch=yhat_base,
                        pred_out=pred_out,
                        y_bch=y,
                        cluster_id_c=cluster_id_c,
                        K=K,
                        y_base_eval_bch=utility_base_bch,
                        cand_eval_bcpH=utility_cand_bcpH,
                        temperature=mse_utility_gate_temperature,
                        min_gain=mse_utility_gate_min_gain,
                        target_power=mse_utility_gate_target_power,
                        include_skip=mse_utility_gate_include_skip,
                        probs_include_skip_mass=bool(skip_competes),
                        target_mode=mse_utility_gate_target_mode,
                        return_diagnostics=True,
                    )
                    mse_gate_loss_bk, mse_gate_diag = (
                        mse_gate_result if mse_gate_result is not None else (None, None)
                    )
                    if mse_gate_loss_bk is not None:
                        mse_gate_component_bk = mse_utility_gate_weight * mse_gate_loss_bk
                        gate_utility_component_bk = gate_utility_component_bk + mse_gate_component_bk
                        loss_bk = loss_bk + mse_gate_component_bk
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
                known_component_bk = (
                    forecast_loss_component_bk
                    + penalty_loss_component_bk
                    + pred_residual_aux_component_bk
                    + candidate_supervision_component_bk
                    + intervention_supervision_component_bk
                    + skip_noop_component_bk
                    + gate_utility_component_bk
                )
                other_aux_component_bk = loss_bk - known_component_bk
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
                forecast_loss_component_bk = objective_loss_bk
                penalty_loss_component_bk = torch.zeros_like(mse_bk)
                pred_residual_aux_component_bk = torch.zeros_like(mse_bk)
                candidate_supervision_component_bk = torch.zeros_like(mse_bk)
                intervention_supervision_component_bk = torch.zeros_like(mse_bk)
                skip_noop_component_bk = torch.zeros_like(mse_bk)
                gate_utility_component_bk = torch.zeros_like(mse_bk)
                other_aux_component_bk = loss_bk - forecast_loss_component_bk
            _accumulate_detached_sum_(train_loss_sum_k, raw_objective_loss_bk)
            _accumulate_detached_sum_(train_mse_sum_k, mse_bk)
            _accumulate_detached_sum_(train_mae_sum_k, mae_bk)
            if stage2_loss_audit_enable:
                _accumulate_detached_sum_(stage2_total_loss_sum_k, loss_bk)
                _accumulate_detached_sum_(stage2_forecast_loss_sum_k, forecast_loss_component_bk)
                _accumulate_detached_sum_(stage2_penalty_loss_sum_k, penalty_loss_component_bk)
                _accumulate_detached_sum_(stage2_pred_residual_aux_loss_sum_k, pred_residual_aux_component_bk)
                _accumulate_detached_sum_(stage2_candidate_supervision_loss_sum_k, candidate_supervision_component_bk)
                _accumulate_detached_sum_(stage2_gate_utility_loss_sum_k, gate_utility_component_bk)
                _accumulate_detached_sum_(stage2_skip_noop_loss_sum_k, skip_noop_component_bk)
                _accumulate_detached_sum_(stage2_intervention_supervision_loss_sum_k, intervention_supervision_component_bk)
                _accumulate_detached_sum_(stage2_other_aux_loss_sum_k, other_aux_component_bk)
            if mse_gate_diag is not None:
                count_bk = torch.ones_like(mse_gate_diag["valid_bk"])
                _accumulate_detached_sum_(mse_gate_diag_count_k, count_bk)
                _accumulate_detached_sum_(mse_gate_valid_sum_k, mse_gate_diag["valid_bk"])
                _accumulate_detached_sum_(mse_gate_skip_target_sum_k, mse_gate_diag["target_skip_bk"])
                _accumulate_detached_sum_(mse_gate_best_gain_sum_k, mse_gate_diag["best_gain_bk"])
                if "skip_prob_bk" in mse_gate_diag:
                    _accumulate_detached_sum_(mse_gate_skip_prob_sum_k, mse_gate_diag["skip_prob_bk"])
            if mse_gate_loss_bk is not None:
                _accumulate_detached_sum_(mse_gate_loss_sum_k, mse_gate_loss_bk)
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
                _mask_gate_grads_after_epoch(
                    gate=gate,
                    epoch=ep,
                    freeze_after_epoch=pred_residual_freeze_gate_after_epoch,
                    stopped=stopped,
                )
            if pred_residual is not None:
                pred_residual.mask_cluster_grads(stopped)
            if dynamic_lambda is not None:
                dynamic_lambda.mask_cluster_grads(stopped)
            if learnable_lambda is not None:
                learnable_lambda.mask_cluster_grads(stopped)
            if stage2_loss_audit_enable:
                stage2_grad_norm_sum["backbone"] += _parameter_grad_l2_norm(model.parameters())
                stage2_grad_norm_sum["gate"] += _parameter_grad_l2_norm(gate.parameters())
                if pred_residual is not None:
                    stage2_grad_norm_sum["pred_residual"] += _parameter_grad_l2_norm(pred_residual.parameters())
                if dynamic_lambda is not None:
                    stage2_grad_norm_sum["dynamic_lambda"] += _parameter_grad_l2_norm(dynamic_lambda.parameters())
                if learnable_lambda is not None:
                    stage2_grad_norm_sum["learnable_lambda"] += _parameter_grad_l2_norm(learnable_lambda.parameters())
                stage2_grad_norm_batches += 1
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
            eval_start=val_eval_start,
        )
        train_loss_k = train_loss_sum_k / max(train_cnt, 1)
        train_mse_k = train_mse_sum_k / max(train_cnt, 1)
        train_mae_k = train_mae_sum_k / max(train_cnt, 1)
        mse_gate_diag_den_k = mse_gate_diag_count_k.clamp_min(1.0)
        if bool((mse_gate_diag_count_k > 0.0).any().item()):
            mse_gate_train_diag_history.append(
                {
                    "epoch": int(ep),
                    "weight": float(mse_utility_gate_weight),
                    "min_gain": float(mse_utility_gate_min_gain),
                    "target_mode": str(mse_utility_gate_target_mode),
                    "include_skip": bool(mse_utility_gate_include_skip),
                    "per_cluster": [
                        {
                            "cluster_id": int(k),
                            "samples": float(mse_gate_diag_count_k[k].detach().cpu().item()),
                            "mean_loss": float((mse_gate_loss_sum_k[k] / mse_gate_diag_den_k[k]).detach().cpu().item()),
                            "valid_rate": float((mse_gate_valid_sum_k[k] / mse_gate_diag_den_k[k]).detach().cpu().item()),
                            "skip_target_rate": float((mse_gate_skip_target_sum_k[k] / mse_gate_diag_den_k[k]).detach().cpu().item()),
                            "mean_skip_prob": float((mse_gate_skip_prob_sum_k[k] / mse_gate_diag_den_k[k]).detach().cpu().item()),
                            "mean_best_allowed_gain": float((mse_gate_best_gain_sum_k[k] / mse_gate_diag_den_k[k]).detach().cpu().item()),
                        }
                        for k in range(K)
                    ],
                }
            )
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
        if stage2_loss_audit_enable:
            epoch_loss_summary = _stage2_loss_epoch_summary(
                epoch=ep,
                count=train_cnt,
                cluster_weight_k=cluster_weight_k,
                total_loss_sum_k=stage2_total_loss_sum_k,
                forecast_loss_sum_k=stage2_forecast_loss_sum_k,
                penalty_loss_sum_k=stage2_penalty_loss_sum_k,
                pred_residual_aux_loss_sum_k=stage2_pred_residual_aux_loss_sum_k,
                candidate_supervision_loss_sum_k=stage2_candidate_supervision_loss_sum_k,
                gate_utility_loss_sum_k=stage2_gate_utility_loss_sum_k,
                skip_noop_loss_sum_k=stage2_skip_noop_loss_sum_k,
                intervention_supervision_loss_sum_k=stage2_intervention_supervision_loss_sum_k,
                other_aux_loss_sum_k=stage2_other_aux_loss_sum_k,
                train_mse_sum_k=train_mse_sum_k,
                train_mae_sum_k=train_mae_sum_k,
            )
            route_summary = _stage2_route_epoch_summary(
                penalty_names=penalty_names,
                cluster_weight_k=cluster_weight_k,
                route_count_k=stage2_route_count_k,
                route_prob_sum_kp=stage2_route_prob_sum_kp,
                route_actual_sum_kp=stage2_route_actual_sum_kp,
                route_entropy_sum_k=stage2_route_entropy_sum_k,
                skip_prob_sum_k=stage2_skip_prob_sum_k,
                skip_active_sum_k=stage2_skip_active_sum_k,
            )
            grad_den = max(int(stage2_grad_norm_batches), 1)
            epoch_loss_summary["route"] = route_summary
            epoch_loss_summary["gradient_l2_mean"] = {
                name: float(value / grad_den) for name, value in stage2_grad_norm_sum.items()
            }
            epoch_loss_summary["val_loss"] = val_loss_agg
            epoch_loss_summary["val_mse"] = float(reduce_cluster_metric(val_mse_k, cluster_weight_k).item())
            epoch_loss_summary["val_mae"] = float(reduce_cluster_metric(val_mae_k, cluster_weight_k).item())
            stage2_loss_audit_history.append(epoch_loss_summary)
        if (
            stage2_route_audit_enable
            and pred_residual is not None
            and moe_enable
            and P > 0
            and (int(ep) % int(stage2_route_audit_frequency) == 0)
        ):
            route_audit_max_batches = int(stage2_route_audit_cfg.get("max_batches", 0))
            route_audit_feature_mode = str(stage2_route_audit_cfg.get("feature_mode", "base"))
            route_audit_thresholds = _stage2_route_audit_thresholds(
                stage2_route_audit_cfg=stage2_route_audit_cfg,
                route_ce_min_abs_improvement=route_ce_min_abs_improvement,
                route_ce_min_rel_improvement=route_ce_min_rel_improvement,
                route_ce_min_candidate_delta_rms=route_ce_min_candidate_delta_rms,
                binary_adoption_weight=binary_adoption_weight,
                binary_adoption_min_abs_improvement=binary_adoption_min_abs_improvement,
                binary_adoption_min_rel_improvement=binary_adoption_min_rel_improvement,
                binary_adoption_min_candidate_delta_rms=binary_adoption_min_candidate_delta_rms,
            )
            prior_for_route_audit = (
                cluster_penalty_prior_prob_kp
                if cluster_penalty_prior_prob_kp is not None
                else gate_prior_prob_kp
            )
            epoch_route_audit: Dict[str, object] = {
                "epoch": int(ep),
                "max_batches": int(route_audit_max_batches),
                "splits": {},
                "val_eval_mse": float(reduce_cluster_metric(val_mse_k, cluster_weight_k).item()),
                "val_eval_mae": float(reduce_cluster_metric(val_mae_k, cluster_weight_k).item()),
                "selected_scaled_eval_mse": None,
                "selected_scaled_eval_mae": None,
                "selected_scaled_note": (
                    "Per-epoch selected/scaled channel-selection metrics are not computed in this hook; "
                    "final selected/scaled metrics are reported after residual selection."
                ),
                "label_thresholds": route_audit_thresholds,
            }
            for split_name, split_loader in stage2_route_audit_loaders.items():
                route_tensors = _collect_penalty_route_learnability_tensors(
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
                    feature_mode=route_audit_feature_mode,
                    allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                    min_abs_improvement=float(route_audit_thresholds["min_abs_improvement"]),
                    min_rel_improvement=float(route_audit_thresholds["min_rel_improvement"]),
                    min_candidate_delta_rms=float(route_audit_thresholds["min_candidate_delta_rms"]),
                    max_batches=route_audit_max_batches,
                    history_anchor_cfg=history_anchor_cfg,
                    observed_history_tc=data_window_tc,
                    input_len=L,
                    eval_start=int(stage2_route_audit_eval_starts.get(split_name, 0)),
                    model_train_stat_adapter_pc=model_train_stat_adapter_pc,
                    model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
                    train_stat_anchor_pc=train_stat_anchor_pc,
                    train_residual_anchor_phc=train_residual_anchor_phc,
                    gate_feature_mode=gate_feature_mode,
                )
                if route_tensors is None:
                    continue
                explain_payload = evaluate_penalty_explainability(
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
                    prior_prob_kp=prior_for_route_audit,
                    allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                    max_batches=route_audit_max_batches,
                    history_anchor_cfg=history_anchor_cfg,
                    observed_history_tc=data_window_tc,
                    input_len=L,
                    eval_start=int(stage2_route_audit_eval_starts.get(split_name, 0)),
                    model_train_stat_adapter_pc=model_train_stat_adapter_pc,
                    model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
                    train_stat_anchor_pc=train_stat_anchor_pc,
                    train_residual_anchor_phc=train_residual_anchor_phc,
                    gate_feature_mode=gate_feature_mode,
                )
                split_summary = _route_audit_summary_from_tensors(
                    tensors=route_tensors,
                    explainability=explain_payload,
                )
                cast_splits = epoch_route_audit["splits"]
                assert isinstance(cast_splits, dict)
                cast_splits[split_name] = split_summary
            stage2_route_audit_history.append(epoch_route_audit)
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

    pred_residual_confidence_summary = None
    if pred_residual_confidence_gate_enable and pred_residual is not None and P > 0:
        confidence_source_split = str(pred_residual_confidence_gate_source_split)
        confidence_source_range = (0, len(dtr))
        if confidence_source_split == "train_holdout":
            ranges = _explainability_train_subsplit_ranges(
                num_windows=len(dtr),
                holdout_fraction=pred_residual_confidence_gate_holdout_fraction,
            )
            if "train_holdout" in ranges:
                confidence_source_range = ranges["train_holdout"]
            else:
                confidence_source_split = "train"
        if len(dtr) <= 0:
            pred_residual_confidence_summary = {
                "enable": False,
                "reason": "empty_train_split",
                "source_requirement": "train_only",
            }
        else:
            start_i, end_i = confidence_source_range
            start_i = max(0, int(start_i))
            end_i = min(len(dtr), int(end_i))
            if end_i <= start_i:
                start_i, end_i = 0, len(dtr)
                confidence_source_split = "train"
            if confidence_source_split == "train" and start_i == 0 and end_i == len(dtr):
                confidence_loader = DataLoader(
                    dtr,
                    batch_size=int(cfg["train"]["batch_size"]),
                    shuffle=False,
                    num_workers=0,
                    pin_memory=pin_mem,
                )
            else:
                confidence_loader = DataLoader(
                    Subset(dtr, range(start_i, end_i)),
                    batch_size=int(cfg["train"]["batch_size"]),
                    shuffle=False,
                    num_workers=0,
                    pin_memory=pin_mem,
                )

            threshold_raw = pred_residual_confidence_gate_threshold
            threshold_is_auto = str(threshold_raw).strip().lower() == "auto"
            if threshold_is_auto:
                confidence_tensors = _collect_pred_residual_selector_tensors(
                    model=model,
                    pred_residual=pred_residual,
                    loader=confidence_loader,
                    cluster_id_c=cluster_id_c,
                    K=K,
                    moe_cfg=moe_cfg,
                    device=device,
                    penalty_count=P,
                    pred_residual_scale_c=None,
                    history_anchor_cfg=history_anchor_cfg,
                    observed_history_tc=data_window_tc,
                    input_len=L,
                    eval_start=0,
                    model_train_stat_adapter_pc=model_train_stat_adapter_pc,
                    model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
                    train_stat_anchor_pc=train_stat_anchor_pc,
                    train_residual_anchor_phc=train_residual_anchor_phc,
                    candidate_feature_mode="base",
                )
                threshold_kp, pred_residual_confidence_summary = (
                    _select_pred_residual_confidence_thresholds_from_tensors(
                        tensors=confidence_tensors,
                        cluster_id_c=cluster_id_c,
                        K=K,
                        allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                        penalty_names=penalty_names,
                        min_abs_improvement=pred_residual_confidence_gate_min_abs,
                        min_rel_improvement=pred_residual_confidence_gate_min_rel,
                        max_candidates=pred_residual_confidence_gate_max_candidates,
                        selection_metric=pred_residual_confidence_gate_selection_metric,
                        min_precision=pred_residual_confidence_gate_min_precision,
                        max_pred_positive_rate=pred_residual_confidence_gate_max_pred_rate,
                    )
                )
                pred_residual_confidence_summary["threshold_mode"] = "auto"
            else:
                threshold_tensor = torch.as_tensor(threshold_raw, dtype=torch.float32)
                if int(threshold_tensor.numel()) == 1:
                    threshold_kp = torch.full((K, P), float(threshold_tensor.reshape(-1)[0].item()), dtype=torch.float32)
                elif tuple(threshold_tensor.shape) == (K, P):
                    threshold_kp = threshold_tensor.reshape(K, P).to(dtype=torch.float32)
                else:
                    raise ValueError(
                        "moe.pred_side_residual.confidence_gate.threshold must be 'auto', "
                        f"a scalar, or shape [{K},{P}], got {tuple(threshold_tensor.shape)}."
                    )
                pred_residual_confidence_summary = {
                    "enable": True,
                    "source_requirement": "train_only",
                    "threshold_mode": "fixed",
                    "threshold_kp": [[float(v) for v in row] for row in threshold_kp.tolist()],
                    "penalty_names": list(penalty_names),
                    "selection_metric": str(pred_residual_confidence_gate_selection_metric),
                    "min_abs_improvement": float(pred_residual_confidence_gate_min_abs),
                    "min_rel_improvement": float(pred_residual_confidence_gate_min_rel),
                    "min_precision": float(pred_residual_confidence_gate_min_precision),
                    "max_pred_positive_rate": (
                        None
                        if pred_residual_confidence_gate_max_pred_rate is None
                        else float(pred_residual_confidence_gate_max_pred_rate)
                    ),
                }
            skip_threshold_raw = pred_residual_confidence_gate_cfg.get("skip_threshold", None)
            skip_threshold_k = None
            if skip_threshold_raw is not None:
                skip_threshold_tensor = torch.as_tensor(skip_threshold_raw, dtype=torch.float32)
                if int(skip_threshold_tensor.numel()) == 1:
                    skip_threshold_k = torch.full((K,), float(skip_threshold_tensor.reshape(-1)[0].item()), dtype=torch.float32)
                elif int(skip_threshold_tensor.numel()) == K:
                    skip_threshold_k = skip_threshold_tensor.reshape(K).to(dtype=torch.float32)
                else:
                    raise ValueError(
                        "moe.pred_side_residual.confidence_gate.skip_threshold must be scalar "
                        f"or length {K}, got {int(skip_threshold_tensor.numel())}."
                    )
                pred_residual_confidence_summary["skip_threshold_k"] = [
                    float(v) for v in skip_threshold_k.tolist()
                ]
            pred_residual.set_confidence_gate(
                penalty_threshold_kp=threshold_kp.to(device=device),
                skip_threshold_k=None if skip_threshold_k is None else skip_threshold_k.to(device=device),
                enable=True,
            )
            pred_residual_confidence_summary.update(
                {
                    "enable": True,
                    "source_split": str(confidence_source_split),
                    "source_range": [int(start_i), int(end_i)],
                    "source_windows": int(end_i - start_i),
                    "test_y_base_used": False,
                }
            )
            print(
                "Prediction residual confidence gate trained: "
                f"source={confidence_source_split}[{start_i}:{end_i}], "
                f"threshold_mode={pred_residual_confidence_summary.get('threshold_mode')}"
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
            "gate_feature_mode": str(gate_feature_mode),
            "gate_feature_names": _gate_feature_names_for_mode(gate_feature_mode),
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
    if cluster_penalty_late_allowed_mask_kp is not None:
        cluster_penalty_allowed_mask_kp = cluster_penalty_late_allowed_mask_kp
        gate.set_penalty_allowed_mask(cluster_penalty_allowed_mask_kp)
        late_pred_residual_allowed_mask_cp = None
        if pred_residual is not None and bool(cluster_penalty_prior_cfg.get("apply_to_pred_residual", False)):
            late_pred_residual_allowed_mask_cp = _cluster_penalty_mask_to_channel_mask(
                cluster_penalty_allowed_mask_kp,
                cluster_id_c,
            )
            pred_residual.set_allowed_penalty_mask(late_pred_residual_allowed_mask_cp)
        cluster_penalty_prior_late_applied = True
        print(
            "Cluster penalty prior late-eval mask activated: "
            f"allowed_mask={cluster_penalty_allowed_mask_kp.detach().cpu().tolist()}, "
            f"pred_residual_channel_mask={late_pred_residual_allowed_mask_cp.detach().cpu().tolist() if late_pred_residual_allowed_mask_cp is not None else None}"
        )

    # print per-cluster penalty selection after training
    summary_loader = dl_va if len(dva) > 0 else dl_tr
    summary_name = "val" if len(dva) > 0 else "train"
    summary_eval_start = val_eval_start if len(dva) > 0 else 0
    lam_kp_best = lambda_kp_from_epochs(best_epoch)
    lam_kp_summary = average_lambda_kp(summary_loader, lam_kp_best)
    lambda_stats = collect_lambda_stats(summary_loader, lam_kp_best)
    summary_csv_path = os.path.join(out_dir, "cluster_penalty_probs.csv")
    avg_probs_summary = print_cluster_penalty_summary(summary_loader, title=summary_name, lam_kp=lam_kp_summary, csv_path=summary_csv_path)
    lambda_stats_csv_path = os.path.join(out_dir, "cluster_lambda_stats.csv")
    print_dynamic_lambda_summary(summary_name, lambda_stats, csv_path=lambda_stats_csv_path)
    moe_residual_summary = collect_pred_residual_summary(summary_loader, eval_start=summary_eval_start)
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
    val_mse_c_base = None
    val_mae_c_base = None
    pred_residual_channel_scale_c = None
    pred_residual_selector_model = None
    pred_residual_selector_summary = None
    pred_residual_selection_summary = None
    moe_gate_penalty_hit_summary = None
    penalty_explainability_summary = None
    penalty_route_learnability_summary = None
    mae_eval_weight = _scale_mae_objective_weight(
        mae_objective_weight_final if mae_objective_enable else 0.0,
        mae_objective_multiplier_k,
    )
    if skip_test:
        print("eval.skip_test=true: test split windows, evaluation, and metrics are disabled.")
    if len(dva) > 0:
        val_loader_summary = DataLoader(dva, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)
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
            "val_mse_candidate_channel",
        }:
            raise ValueError(
                "Unsupported moe.pred_side_residual.selection_policy="
                f"'{residual_selection_policy}'. Expected none, val_mse_channel, val_mse_scale, "
                "val_mse_scale_holdout, or val_mse_candidate_channel."
            )
        if pred_residual is not None and residual_selection_policy in {
            "val_mse_channel",
            "val_mse_scale",
            "val_mse_scale_holdout",
            "val_mse_candidate_channel",
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
                eval_start=val_eval_start,
            )
            val_scaled_mse_c = val_mse_c_base
            val_scaled_mae_c = val_mae_c_base
            candidate_channel_selector_summary = None
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
            elif residual_selection_policy == "val_mse_candidate_channel":
                min_abs = float(pred_residual_cfg.get("selection_min_abs_improvement", 0.0))
                min_rel = float(pred_residual_cfg.get("selection_min_rel_improvement", 0.0))
                allowed_mask_cp = None
                if cluster_penalty_allowed_mask_kp is not None and int(cluster_penalty_allowed_mask_kp.numel()) > 0:
                    allowed_kp = cluster_penalty_allowed_mask_kp.detach().cpu().to(dtype=torch.bool)
                    cluster_idx = cluster_id_c.detach().cpu().to(dtype=torch.long)
                    allowed_mask_cp = allowed_kp.index_select(0, cluster_idx)
                candidate_tensors = _collect_pred_residual_selector_tensors(
                    model=model,
                    pred_residual=pred_residual,
                    loader=val_loader_summary,
                    cluster_id_c=cluster_id_c,
                    K=K,
                    moe_cfg=moe_cfg,
                    device=device,
                    penalty_count=len(penalty_names),
                    pred_residual_scale_c=None,
                    history_anchor_cfg=history_anchor_cfg,
                    observed_history_tc=data_window_tc,
                    input_len=L,
                    eval_start=val_eval_start,
                    model_train_stat_adapter_pc=model_train_stat_adapter_pc,
                    model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
                    train_stat_anchor_pc=train_stat_anchor_pc,
                    train_residual_anchor_phc=train_residual_anchor_phc,
                    candidate_feature_mode="base",
                )
                if candidate_tensors is None:
                    use_residual_c = torch.zeros(C, dtype=torch.bool)
                    scale_values = [0.0 for _ in range(C)]
                    residual_scale_mean_value = 0.0
                else:
                    static_selector, candidate_channel_selector_summary = _fit_static_candidate_channel_selector_from_tensors(
                        tensors=candidate_tensors,
                        allowed_mask_cp=allowed_mask_cp,
                        penalty_names=penalty_names,
                        channel_names=channel_names,
                        min_abs_improvement=min_abs,
                        min_rel_improvement=min_rel,
                    )
                    pred_residual_selector_model = static_selector.to(device)
                    pred_residual_channel_scale_c = None
                    (
                        _,
                        _,
                        _,
                        val_static_mse_c,
                        val_static_mae_c,
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
                        pred_residual_scale_c=None,
                                        eval_start=val_eval_start,
                    )
                    val_scaled_mse_c = val_static_mse_c
                    val_scaled_mae_c = val_static_mae_c
                    selected_class_c = torch.tensor(
                        candidate_channel_selector_summary.get("selected_class", []),
                        dtype=torch.long,
                    )
                    use_residual_c = selected_class_c > 0
                    scale_values = [float(v) for v in selected_class_c.tolist()]
                    residual_scale_mean_value = float(use_residual_c.to(dtype=torch.float32).mean().item())
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
            if candidate_channel_selector_summary is not None:
                pred_residual_selection_summary["candidate_channel_selector"] = candidate_channel_selector_summary
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
            selector_candidate_scale_c, selector_candidate_scale_mode = _candidate_selector_candidate_scale(
                pred_residual_scale_c=pred_residual_channel_scale_c,
                selector_cfg=selector_cfg,
            )
            candidate_selector_model, pred_residual_selector_summary = train_pred_residual_candidate_selector(
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
                pred_residual_scale_c=selector_candidate_scale_c,
                history_anchor_cfg=history_anchor_cfg,
                observed_history_tc=data_window_tc,
                input_len=L,
                eval_start=selector_eval_start,
                model_train_stat_adapter_pc=model_train_stat_adapter_pc,
                model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
                train_stat_anchor_pc=train_stat_anchor_pc,
                train_residual_anchor_phc=train_residual_anchor_phc,
                allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                gate_feature_mode=gate_feature_mode,
            )
            if pred_residual_selector_summary is not None:
                pred_residual_selector_summary["source_split"] = selector_source_split
                pred_residual_selector_summary["candidate_scale_mode"] = selector_candidate_scale_mode
            if candidate_selector_model is not None:
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
                    pred_residual_selector=candidate_selector_model,
                    pred_residual_scale_c=selector_candidate_scale_c,
                            eval_start=val_eval_start,
                )
                selector_val_summary = {
                    "avg_loss": float(reduce_cluster_metric(val_selector_loss_k, cluster_weight_k).item()),
                    "avg_mse": float(reduce_cluster_metric(val_selector_mse_k, cluster_weight_k).item()),
                    "avg_mae": float(reduce_cluster_metric(val_selector_mae_k, cluster_weight_k).item()),
                    "per_cluster_loss": [float(v) for v in val_selector_loss_k.detach().cpu().tolist()],
                    "per_cluster_mse": [float(v) for v in val_selector_mse_k.detach().cpu().tolist()],
                    "per_cluster_mae": [float(v) for v in val_selector_mae_k.detach().cpu().tolist()],
                    "per_channel_mse": [float(v) for v in val_selector_mse_c.detach().cpu().tolist()],
                    "per_channel_mae": [float(v) for v in val_selector_mae_c.detach().cpu().tolist()],
                }
                if pred_residual_selection_summary is None:
                    pred_residual_selection_summary = {
                        "policy": "candidate_selector",
                        "num_residual_channels": int(C),
                    }
                current_selector_ref_mse = float(
                    pred_residual_selection_summary.get(
                        "val_scaled_avg_mse",
                        (val_summary or {}).get("avg_mse", selector_val_summary["avg_mse"]),
                    )
                )
                current_selector_ref_mae = float(
                    pred_residual_selection_summary.get(
                        "val_scaled_avg_mae",
                        (val_summary or {}).get("avg_mae", selector_val_summary["avg_mae"]),
                    )
                )
                selector_adoption = _candidate_selector_adoption_decision(
                    current_mse=current_selector_ref_mse,
                    current_mae=current_selector_ref_mae,
                    selector_mse=float(selector_val_summary["avg_mse"]),
                    selector_mae=float(selector_val_summary["avg_mae"]),
                    min_abs_improvement=float(
                        selector_cfg.get(
                            "adopt_min_abs_improvement",
                            pred_residual_cfg.get("selection_min_abs_improvement", 0.0),
                        )
                    ),
                    min_rel_improvement=float(
                        selector_cfg.get(
                            "adopt_min_rel_improvement",
                            pred_residual_cfg.get("selection_min_rel_improvement", 0.0),
                        )
                    ),
                    max_rel_mae_regression=float(selector_cfg.get("adopt_max_rel_mae_regression", 0.0)),
                )
                pred_residual_selector_summary["adoption"] = selector_adoption
                pred_residual_selection_summary["candidate_selector"] = pred_residual_selector_summary
                pred_residual_selection_summary["val_selector_avg_mse"] = float(selector_val_summary["avg_mse"])
                pred_residual_selection_summary["val_selector_avg_mae"] = float(selector_val_summary["avg_mae"])
                pred_residual_selection_summary["candidate_selector_adopted"] = bool(selector_adoption["adopt"])
                if bool(selector_adoption["adopt"]):
                    pred_residual_selector_model = candidate_selector_model
                    pred_residual_channel_scale_c = selector_candidate_scale_c
                    val_mse_c_base = val_selector_mse_c
                    val_mae_c_base = val_selector_mae_c
                    val_summary = selector_val_summary
                    pred_residual_selection_summary["selected_residual_evaluator"] = "candidate_selector"
                    pred_residual_selection_summary["val_scaled_avg_mse"] = float(selector_val_summary["avg_mse"])
                    pred_residual_selection_summary["val_scaled_avg_mae"] = float(selector_val_summary["avg_mae"])
                    pred_residual_selection_summary["val_scaled_mse_per_channel"] = [
                        float(v) for v in val_selector_mse_c.detach().cpu().tolist()
                    ]
                    pred_residual_selection_summary["val_scaled_mae_per_channel"] = [
                        float(v) for v in val_selector_mae_c.detach().cpu().tolist()
                    ]
                else:
                    pred_residual_selector_model = None
                    pred_residual_selection_summary.setdefault("selected_residual_evaluator", "channel_scale")
                print(
                    "Prediction residual candidate selector: "
                    f"source={selector_source_split}, "
                    f"val_MSE={selector_val_summary['avg_mse']:.6f}, "
                    f"adopted={bool(selector_adoption['adopt'])}, "
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
                model_train_stat_adapter_pc=model_train_stat_adapter_pc,
                model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
                train_stat_anchor_pc=train_stat_anchor_pc,
                train_residual_anchor_phc=train_residual_anchor_phc,
                allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                gate_feature_mode=gate_feature_mode,
            )
            moe_gate_penalty_hit_summary = {"val": val_penalty_hit, "test": None}
            if val_penalty_hit is not None:
                print(
                    "Gate penalty hit(val): "
                    f"top1={val_penalty_hit['top1_hit_rate_all']:.3f}, "
                    f"positive_top1={val_penalty_hit['top1_hit_rate_on_positive_oracle']:.3f}, "
                    f"selected_gain={val_penalty_hit['selected_top1_gain_pct_vs_base']:.3f}%"
                )

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
            pred_residual_selector=pred_residual_selector_model,
            pred_residual_scale_c=pred_residual_channel_scale_c,
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
            pred_residual_selector=pred_residual_selector_model,
            pred_residual_scale_c=pred_residual_channel_scale_c,
            eval_start=test_eval_start,
            diagnostic_collector=prediction_diag_collector,
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
                model_train_stat_adapter_pc=model_train_stat_adapter_pc,
                model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
                train_stat_anchor_pc=train_stat_anchor_pc,
                train_residual_anchor_phc=train_residual_anchor_phc,
                allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                gate_feature_mode=gate_feature_mode,
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
        split_eval_starts: Dict[str, int] = {}
        train_subsplit_ranges: Dict[str, Tuple[int, int]] = {}
        if "train" in requested_splits and len(dtr) > 0:
            split_loaders["train"] = DataLoader(
                dtr,
                batch_size=int(cfg["train"]["batch_size"]),
                shuffle=False,
                num_workers=0,
                pin_memory=pin_mem,
            )
            split_eval_starts["train"] = 0
        train_subsplit_names = {"train_fit", "train_holdout"}
        if any(name in requested_splits for name in train_subsplit_names) and len(dtr) > 0:
            holdout_fraction = float(
                explain_cfg.get(
                    "train_holdout_fraction",
                    explain_cfg.get("holdout_fraction", 0.30),
                )
            )
            train_subsplit_ranges = _explainability_train_subsplit_ranges(
                num_windows=len(dtr),
                holdout_fraction=holdout_fraction,
            )
            for split_name in ("train_fit", "train_holdout"):
                if split_name not in requested_splits or split_name not in train_subsplit_ranges:
                    continue
                start_i, end_i = train_subsplit_ranges[split_name]
                if int(end_i) <= int(start_i):
                    continue
                split_loaders[split_name] = DataLoader(
                    Subset(dtr, range(int(start_i), int(end_i))),
                    batch_size=int(cfg["train"]["batch_size"]),
                    shuffle=False,
                    num_workers=0,
                    pin_memory=pin_mem,
                )
                split_eval_starts[split_name] = 0
        if "val" in requested_splits and len(dva) > 0:
            split_loaders["val"] = DataLoader(
                dva,
                batch_size=int(cfg["train"]["batch_size"]),
                shuffle=False,
                num_workers=0,
                pin_memory=pin_mem,
            )
            split_eval_starts["val"] = int(val_eval_start)
        if "test" in requested_splits and (not skip_test) and len(dte) > 0:
            split_loaders["test"] = dl_te
            split_eval_starts["test"] = int(test_eval_start)

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
                history_anchor_cfg=history_anchor_cfg,
                observed_history_tc=data_window_tc,
                input_len=L,
                eval_start=int(split_eval_starts.get(split_name, 0)),
                model_train_stat_adapter_pc=model_train_stat_adapter_pc,
                model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
                train_stat_anchor_pc=train_stat_anchor_pc,
                train_residual_anchor_phc=train_residual_anchor_phc,
                gate_feature_mode=gate_feature_mode,
            )
            if payload is not None:
                split_payloads[split_name] = payload
                print(
                    f"Penalty explainability({split_name}): "
                    f"gain={payload['final_gain_pct_vs_base']:.3f}%, "
                    f"selected_events={payload['selected_penalty_events']}, "
                    f"oracle_positive_events={payload['oracle_positive_events']}"
                )
        route_probe_cfg = explain_cfg.get("route_learnability_probe", {}) or {}
        if not isinstance(route_probe_cfg, dict):
            route_probe_cfg = {"enable": bool(route_probe_cfg)}
        if bool(route_probe_cfg.get("enable", False)):
            train_split_name = str(route_probe_cfg.get("train_split", "train_fit")).lower()
            if train_split_name not in split_loaders and "train" in split_loaders:
                train_split_name = "train"
            eval_split_names = [
                str(name).lower()
                for name in (route_probe_cfg.get("eval_splits", ["train_holdout", "val"]) or [])
            ]
            allow_test_probe = bool(route_probe_cfg.get("allow_test", False))
            probe_split_names = []
            for name in [train_split_name] + eval_split_names:
                if name == "test" and not allow_test_probe:
                    continue
                if name in split_loaders and name not in probe_split_names:
                    probe_split_names.append(name)
            route_tensors_by_split: Dict[str, Dict[str, object]] = {}
            route_feature_mode = str(route_probe_cfg.get("feature_mode", "base"))
            route_max_batches = int(route_probe_cfg.get("max_batches", max_batches))
            for split_name in probe_split_names:
                tensors = _collect_penalty_route_learnability_tensors(
                    model=model,
                    gate=gate,
                    pred_residual=pred_residual,
                    loader=split_loaders[split_name],
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
                    feature_mode=route_feature_mode,
                    allowed_mask_kp=allowed_for_explain,
                    max_batches=route_max_batches,
                    history_anchor_cfg=history_anchor_cfg,
                    observed_history_tc=data_window_tc,
                    input_len=L,
                    eval_start=int(split_eval_starts.get(split_name, 0)),
                    model_train_stat_adapter_pc=model_train_stat_adapter_pc,
                    model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
                    train_stat_anchor_pc=train_stat_anchor_pc,
                    train_residual_anchor_phc=train_residual_anchor_phc,
                    gate_feature_mode=gate_feature_mode,
                )
                if tensors is not None:
                    route_tensors_by_split[split_name] = tensors
            artifact_paths: Dict[str, object] = {}
            if train_split_name in route_tensors_by_split:
                train_route_tensors = route_tensors_by_split[train_split_name]
                eval_route_tensors = {
                    name: tensors
                    for name, tensors in route_tensors_by_split.items()
                    if name != train_split_name
                }
                head_cfg = route_probe_cfg.get("head", route_probe_cfg) or {}
                if not isinstance(head_cfg, dict):
                    head_cfg = {}
                penalty_route_learnability_summary, route_head_artifact = _fit_penalty_route_learnability_head_from_tensors(
                    train_tensors=train_route_tensors,  # type: ignore[arg-type]
                    eval_tensors_by_split=eval_route_tensors,  # type: ignore[arg-type]
                    label_names=list(train_route_tensors["label_names"]),  # type: ignore[index]
                    feature_names=list(train_route_tensors["feature_names"]),  # type: ignore[index]
                    cfg=head_cfg,
                    device=device,
                )
                penalty_route_learnability_summary["train_split"] = train_split_name
                penalty_route_learnability_summary["eval_splits"] = list(eval_route_tensors.keys())
                penalty_route_learnability_summary["feature_mode"] = route_feature_mode
                penalty_route_learnability_summary["max_batches"] = int(route_max_batches)
                head_path = os.path.join(out_dir, "penalty_route_learnability_head.pt")
                torch.save(route_head_artifact, head_path)
                artifact_paths["head"] = head_path
                label_names = list(train_route_tensors["label_names"])  # type: ignore[index]
                for split_name, tensors in route_tensors_by_split.items():
                    tensor_path = os.path.join(out_dir, f"penalty_route_learnability_{split_name}.pt")
                    torch.save(tensors, tensor_path)
                    artifact_paths[f"{split_name}_tensors"] = tensor_path
                    labels_cpu = tensors["labels"].detach().cpu().to(dtype=torch.long)  # type: ignore[index]
                    current_cpu = tensors["current_pred"].detach().cpu().to(dtype=torch.long)  # type: ignore[index]
                    query_cpu = tensors["query_start_abs"].detach().cpu().to(dtype=torch.long)  # type: ignore[index]
                    gain_cpu = tensors["oracle_gain_mse"].detach().cpu().to(dtype=torch.float32)  # type: ignore[index]
                    label_df = pd.DataFrame(
                        {
                            "split": split_name,
                            "row": list(range(int(labels_cpu.numel()))),
                            "query_start_abs": [int(v) for v in query_cpu.tolist()],
                            "oracle_class": [int(v) for v in labels_cpu.tolist()],
                            "oracle_label": [
                                label_names[int(v)] if 0 <= int(v) < len(label_names) else ""
                                for v in labels_cpu.tolist()
                            ],
                            "current_class": [int(v) for v in current_cpu.tolist()],
                            "current_label": [
                                label_names[int(v)] if 0 <= int(v) < len(label_names) else ""
                                for v in current_cpu.tolist()
                            ],
                            "oracle_gain_mse": [float(v) for v in gain_cpu.tolist()],
                        }
                    )
                    csv_path = os.path.join(out_dir, f"penalty_route_oracle_labels_{split_name}.csv")
                    label_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                    artifact_paths[f"{split_name}_labels_csv"] = csv_path
                summary_path = os.path.join(out_dir, "penalty_route_learnability.json")
                artifact_paths["summary"] = summary_path
                penalty_route_learnability_summary["artifact_paths"] = artifact_paths
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(penalty_route_learnability_summary, f, ensure_ascii=False, indent=2)
                val_metrics = (penalty_route_learnability_summary.get("splits", {}) or {}).get("val")
                if isinstance(val_metrics, dict):
                    print(
                        "Penalty route learnability(val): "
                        f"head_acc={float(val_metrics.get('accuracy_all', 0.0)):.3f}, "
                        f"current_acc={float(val_metrics.get('current_accuracy_all', 0.0)):.3f}, "
                        f"majority_acc={float(val_metrics.get('majority_accuracy_all', 0.0)):.3f}"
                    )
            else:
                penalty_route_learnability_summary = {
                    "enable": True,
                    "skipped": True,
                    "reason": f"train_split {train_split_name!r} was not available",
                    "available_splits": list(split_loaders.keys()),
                }
        penalty_explainability_summary = {
            "enable": True,
            "max_batches": int(max_batches),
            "train_subsplits": {
                name: {"start": int(start_i), "end": int(end_i)}
                for name, (start_i, end_i) in train_subsplit_ranges.items()
            },
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
    df = None
    avg_mae = None
    avg_mse = None
    selected_variant = "base"
    selected_criterion = "base"
    selected_selection_policy = "base"
    selected_avg_mae = None
    selected_avg_mse = None
    if not skip_test:
        df = pd.DataFrame({
            "channel": channel_names,
            "MAE": mae_c.numpy(),
            "MSE": mse_c.numpy(),
            "cluster_id": cluster_id_c.detach().cpu().numpy(),
        })
        avg_mae = float(df["MAE"].mean())
        avg_mse = float(reduce_cluster_metric(test_mse_k, cluster_weight_k).item())
        selected_avg_mae = avg_mae
        selected_avg_mse = avg_mse

    moe_residual_variant = "none"
    if pred_residual_selection_summary is not None:
        moe_residual_variant = "moe_residual_channel"
        if int(pred_residual_selection_summary.get("num_residual_channels", 0) or 0) > 0:
            selected_variant = moe_residual_variant
            selected_criterion = str(pred_residual_selection_summary.get("policy", selected_criterion))
            selected_selection_policy = str(pred_residual_selection_summary.get("policy", selected_selection_policy))

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
    stage2_loss_diagnostics_summary = None
    if stage2_loss_audit_enable:
        residual_selection = pred_residual_selection_summary or {}
        moe_residual_diag = moe_residual_summary or {}
        latest_route = (
            stage2_loss_audit_history[-1].get("route", {})
            if len(stage2_loss_audit_history) > 0
            else {}
        )
        val_base_mse = residual_selection.get("val_pred_base_avg_mse", (val_summary or {}).get("avg_mse"))
        val_base_mae = residual_selection.get("val_pred_base_avg_mae", (val_summary or {}).get("avg_mae"))
        val_raw_moe_mse = residual_selection.get("val_residual_avg_mse", (val_summary or {}).get("avg_mse"))
        val_raw_moe_mae = residual_selection.get("val_residual_avg_mae", (val_summary or {}).get("avg_mae"))
        val_scaled_mse = residual_selection.get("val_scaled_avg_mse", (val_summary or {}).get("avg_mse"))
        val_scaled_mae = residual_selection.get("val_scaled_avg_mae", (val_summary or {}).get("avg_mae"))
        stage2_loss_diagnostics_summary = {
            "enabled": True,
            "losses_are_stage2_only": True,
            "do_not_compare_to_stage1_training_loss": True,
            "trainable_parameter_groups": stage2_trainable_parameter_groups,
            "epochs": stage2_loss_audit_history,
            "final_eval": {
                "val_base_mse": val_base_mse,
                "val_base_mae": val_base_mae,
                "val_raw_moe_mse": val_raw_moe_mse,
                "val_raw_moe_mae": val_raw_moe_mae,
                "val_scaled_or_selected_moe_mse": val_scaled_mse,
                "val_scaled_or_selected_moe_mae": val_scaled_mae,
                "residual_delta_rms": moe_residual_diag.get("residual_delta_rms"),
                "residual_base_rms_ratio": moe_residual_diag.get("residual_base_rms_ratio"),
                "route_entropy": latest_route.get("route_entropy"),
                "actual_route_distribution": moe_residual_diag.get(
                    "effective_route_by_penalty",
                    latest_route.get("actual_route_distribution"),
                ),
                "skip_noop_rate": latest_route.get("skip_noop_rate"),
                "skip_prob": latest_route.get("skip_prob"),
            },
        }
    stage2_route_audit_summary = None
    if stage2_route_audit_enable:
        residual_selection = pred_residual_selection_summary or {}
        final_scaled_mse = residual_selection.get("val_scaled_avg_mse", (val_summary or {}).get("avg_mse"))
        final_scaled_mae = residual_selection.get("val_scaled_avg_mae", (val_summary or {}).get("avg_mae"))
        stage2_route_audit_summary = {
            "enabled": True,
            "splits": list(stage2_route_audit_loaders.keys()),
            "train_subsplits": {
                name: {"start": int(start_i), "end": int(end_i)}
                for name, (start_i, end_i) in stage2_route_audit_train_subsplits.items()
            },
            "max_batches": int(stage2_route_audit_cfg.get("max_batches", 0)),
            "frequency_epochs": int(stage2_route_audit_frequency),
            "skip_noop_is_class_zero": True,
            "test_read": False,
            "final_selected_scaled_eval": {
                "val_scaled_or_selected_moe_mse": final_scaled_mse,
                "val_scaled_or_selected_moe_mae": final_scaled_mae,
                "source": "final_moe_residual_selection",
            },
            "epochs": stage2_route_audit_history,
        }
        route_audit_path = os.path.join(out_dir, "stage2_route_audit.json")
        with open(route_audit_path, "w", encoding="utf-8") as f:
            json.dump(stage2_route_audit_summary, f, ensure_ascii=False, indent=2)
        stage2_route_audit_summary["artifact_path"] = route_audit_path

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
        "calendar_residual": calendar_residual_summary,
        "moe_residual": moe_residual_summary,
        "moe_residual_phase_candidate": phase_residual_candidate_summary,
        "moe_residual_confidence_gate": pred_residual_confidence_summary,
        "moe_residual_selection": pred_residual_selection_summary,
        "moe_residual_candidate_selector": pred_residual_selector_summary,
        "cluster_penalty_prior": {
            "enable": bool(cluster_penalty_prior_enable),
            "apply_stage": str(cluster_penalty_prior_apply_stage),
            "late_eval_applied": bool(cluster_penalty_prior_late_applied),
            "apply_to_pred_residual": bool(cluster_penalty_prior_cfg.get("apply_to_pred_residual", False)),
            "prior": (
                cluster_penalty_prior_prob_kp.detach().cpu().tolist()
                if cluster_penalty_prior_prob_kp is not None
                else None
            ),
            "configured_allowed_mask": (
                cluster_penalty_prior_configured_mask_kp.detach().cpu().tolist()
                if cluster_penalty_prior_configured_mask_kp is not None
                else None
            ),
            "active_allowed_mask": (
                cluster_penalty_allowed_mask_kp.detach().cpu().tolist()
                if cluster_penalty_allowed_mask_kp is not None
                else None
            ),
            "late_allowed_mask": (
                cluster_penalty_late_allowed_mask_kp.detach().cpu().tolist()
                if cluster_penalty_late_allowed_mask_kp is not None
                else None
            ),
        },
        "model_train_stat_adapter": model_train_stat_adapter_summary,
        "train_stat_anchor_expert": train_stat_anchor_summary,
        "train_residual_anchor_expert": train_residual_anchor_summary,
        "moe_gate_penalty_hit": moe_gate_penalty_hit_summary,
        "penalty_explainability": penalty_explainability_summary,
        "penalty_route_learnability": penalty_route_learnability_summary,
        "moe_router": {
            "mode": str(router_mode),
            "penalty_context_weight": float(router_penalty_context_weight),
            "penalty_context_score": str(router_penalty_context_score),
            "detach_penalty_context": bool(router_detach_penalty_context),
            "context_applied_inside_gate_logits": True,
            "allow_skip": bool(allow_skip),
            "skip_competes_with_penalties": bool(skip_competes),
            "skip_argmax_noop": bool(skip_argmax_noop),
            "skip_cost": float(skip_cost),
            "skip_supervision_weight": float(skip_supervision_weight),
            "skip_supervision_margin": float(skip_supervision_margin),
            "freeze_gate_after_epoch": int(pred_residual_freeze_gate_after_epoch),
            "route_ce_supervision": {
                "enable": bool(route_ce_enable),
                "weight": float(route_ce_weight),
                "min_abs_improvement": float(route_ce_min_abs_improvement),
                "min_rel_improvement": float(route_ce_min_rel_improvement),
                "min_candidate_delta_rms": float(route_ce_min_candidate_delta_rms),
                "ignore_abs_gain_below": float(route_ce_ignore_abs_gain_below),
                "class_weight": str(route_ce_class_weight_mode),
                "max_class_weight": float(route_ce_max_class_weight),
                "require_skip": bool(route_ce_require_skip),
                "require_skip_competes": bool(route_ce_require_skip_competes),
                "require_skip_argmax_noop": bool(route_ce_require_skip_argmax_noop),
                "probs_include_skip_mass": bool(skip_competes),
            },
            "binary_adoption_supervision": {
                "enable": bool(binary_adoption_enable),
                "weight": float(binary_adoption_weight),
                "min_abs_improvement": float(binary_adoption_min_abs_improvement),
                "min_rel_improvement": float(binary_adoption_min_rel_improvement),
                "min_candidate_delta_rms": float(binary_adoption_min_candidate_delta_rms),
                "ignore_abs_gain_below": float(binary_adoption_ignore_abs_gain_below),
                "positive_weight": float(binary_adoption_positive_weight),
                "negative_weight": float(binary_adoption_negative_weight),
                "require_skip": bool(binary_adoption_require_skip),
                "require_skip_competes": bool(binary_adoption_require_skip_competes),
                "require_skip_argmax_noop": bool(binary_adoption_require_skip_argmax_noop),
                "probs_include_skip_mass": bool(skip_competes),
            },
            "route_rate_alignment_supervision": {
                "enable": bool(route_rate_alignment_enable),
                "weight": float(route_rate_alignment_weight),
                "min_abs_improvement": float(route_rate_alignment_min_abs_improvement),
                "min_rel_improvement": float(route_rate_alignment_min_rel_improvement),
                "min_candidate_delta_rms": float(route_rate_alignment_min_candidate_delta_rms),
                "ignore_abs_gain_below": float(route_rate_alignment_ignore_abs_gain_below),
                "require_skip": bool(route_rate_alignment_require_skip),
                "require_skip_competes": bool(route_rate_alignment_require_skip_competes),
                "require_skip_argmax_noop": bool(route_rate_alignment_require_skip_argmax_noop),
                "probs_include_skip_mass": bool(skip_competes),
            },
            "route_positive_recall_supervision": {
                "enable": bool(route_positive_recall_enable),
                "weight": float(route_positive_recall_weight),
                "min_abs_improvement": float(route_positive_recall_min_abs_improvement),
                "min_rel_improvement": float(route_positive_recall_min_rel_improvement),
                "min_candidate_delta_rms": float(route_positive_recall_min_candidate_delta_rms),
                "ignore_abs_gain_below": float(route_positive_recall_ignore_abs_gain_below),
                "mode": str(route_positive_recall_mode),
                "target_probability": float(route_positive_recall_target_probability),
                "require_skip": bool(route_positive_recall_require_skip),
                "require_skip_competes": bool(route_positive_recall_require_skip_competes),
                "require_skip_argmax_noop": bool(route_positive_recall_require_skip_argmax_noop),
                "probs_include_skip_mass": bool(skip_competes),
            },
            "route_precision_recall_supervision": {
                "enable": bool(route_precision_recall_enable),
                "weight": float(route_precision_recall_weight),
                "min_abs_improvement": float(route_precision_recall_min_abs_improvement),
                "min_rel_improvement": float(route_precision_recall_min_rel_improvement),
                "min_candidate_delta_rms": float(route_precision_recall_min_candidate_delta_rms),
                "ignore_abs_gain_below": float(route_precision_recall_ignore_abs_gain_below),
                "recall_mode": str(route_precision_recall_mode),
                "recall_target_probability": float(route_precision_recall_target_probability),
                "false_adopt_max_probability": float(route_precision_recall_false_adopt_max_probability),
                "false_adopt_weight": float(route_precision_recall_false_adopt_weight),
                "require_skip": bool(route_precision_recall_require_skip),
                "require_skip_competes": bool(route_precision_recall_require_skip_competes),
                "require_skip_argmax_noop": bool(route_precision_recall_require_skip_argmax_noop),
                "probs_include_skip_mass": bool(skip_competes),
            },
            "mse_utility_gate_supervision": {
                "enable": bool(mse_utility_gate_enable),
                "weight": float(mse_utility_gate_weight),
                "temperature": float(mse_utility_gate_temperature),
                "min_gain": float(mse_utility_gate_min_gain),
                "target_power": float(mse_utility_gate_target_power),
                "target_mode": str(mse_utility_gate_target_mode),
                "include_skip": bool(mse_utility_gate_include_skip),
                "probs_include_skip_mass": bool(skip_competes),
                "train_diagnostics": list(mse_gate_train_diag_history),
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
            "base_val_mse": None if val_summary is None else val_summary.get("avg_mse"),
            "base_val_mae": None if val_summary is None else val_summary.get("avg_mae"),
        },
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
    if stage2_loss_diagnostics_summary is not None:
        summary["stage2_loss_diagnostics"] = stage2_loss_diagnostics_summary
    if stage2_route_audit_summary is not None:
        summary["stage2_route_audit"] = stage2_route_audit_summary
    if finetune_summary is not None:
        summary["finetune"] = finetune_summary
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
