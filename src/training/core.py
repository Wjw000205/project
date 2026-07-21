"""Core PKR-MoE training, routing, loss, and optimization helpers."""
from __future__ import annotations

import math
import os
import hashlib
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

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

from ..models.dynamic_lambda import ClusterwiseDynamicLambda
from ..models.moe_gate import ClusterwiseMoEGate, scatter_mean_bc_to_bk, scatter_mean_bcf_to_bkf
from ..models.penalties import normalize_penalties
from ..models.residual_moe import ClusterwisePredResidualMoE


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


def _resolve_overfit_diagnostic_range(
    num_windows: int,
    cfg: Optional[dict],
    *,
    config_name: str = "train.overfit_diagnostic",
) -> Optional[Tuple[int, int]]:
    """Resolve the fixed contiguous train-window range used by a gate overfit audit."""
    cfg = cfg or {}
    if not bool(cfg.get("enable", False)):
        return None
    total = int(num_windows)
    if total <= 0:
        raise ValueError(f"{config_name} requires a non-empty training split.")
    count = int(cfg.get("num_windows", 256))
    if count <= 0 or count > total:
        raise ValueError(
            f"{config_name}.num_windows must be in "
            f"[1, {total}], got {count}."
        )
    position = str(cfg.get("position", "head")).strip().lower()
    if "start_idx" in cfg:
        start = int(cfg["start_idx"])
    elif position == "head":
        start = 0
    elif position == "center":
        start = (total - count) // 2
    elif position == "tail":
        start = total - count
    else:
        raise ValueError(
            f"{config_name}.position must be head, center, or tail "
            f"(got {position!r})."
        )
    end = start + count
    if start < 0 or end > total:
        raise ValueError(
            f"{config_name} range is outside the training split: "
            f"start={start}, end={end}, total={total}."
        )
    return int(start), int(end)


def _freeze_module_params(module: nn.Module) -> int:
    frozen = 0
    for param in module.parameters():
        param.requires_grad_(False)
        frozen += int(param.numel())
    return frozen


def _set_module_train_mode(
    module: nn.Module,
    *,
    training: bool,
    keep_frozen_eval: bool = False,
) -> bool:
    """Set train/eval state without re-enabling stochastic frozen modules."""
    effective_training = bool(training) and not bool(keep_frozen_eval)
    module.train(effective_training)
    return bool(module.training)


def _freeze_module_params_except_prefixes(
    module: nn.Module,
    trainable_prefixes: Tuple[str, ...],
) -> int:
    prefixes = tuple(str(prefix) for prefix in trainable_prefixes)
    frozen = 0
    for name, param in module.named_parameters():
        keep_trainable = any(name.startswith(prefix) for prefix in prefixes)
        param.requires_grad_(keep_trainable)
        if not keep_trainable:
            param.grad = None
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


def _normalize_pred_residual_selection_policy(selection_policy: object) -> str:
    value = str(selection_policy or "none").strip().lower()
    if value in {"false", "off", "disable", "disabled"}:
        return "none"
    if value == "val_mse_candidate_channel_guarded":
        return "val_mse_candidate_channel"
    return value


def _normalize_pred_residual_candidate_selection_metric(selection_metric: object) -> str:
    value = str(selection_metric or "mse").strip().lower()
    if value == "val_mse":
        return "mse"
    if value == "val_mae":
        return "mae"
    if value not in {"mse", "mae"}:
        raise ValueError("moe.pred_side_residual.selection_metric must be mse, mae, val_mse, or val_mae.")
    return value


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


def _normalize_learnable_output_anchor_cfg(cfg: object) -> Dict[str, object]:
    if isinstance(cfg, dict):
        return dict(cfg)
    if cfg is None:
        return {}
    return {"enable": bool(cfg)}


def _clone_module_state_dict(module: Optional[nn.Module]) -> Optional[Dict[str, torch.Tensor]]:
    if module is None:
        return None
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


@torch.no_grad()
def _copy_learnable_output_anchor_active_masks(
    target: nn.Module,
    source: nn.Module,
) -> Dict[str, List[float]]:
    """Copy the deployment adoption masks of a frozen learnable anchor source."""
    target_channel = getattr(target, "active_channel_mask_c", None)
    source_channel = getattr(source, "active_channel_mask_c", None)
    target_horizon = getattr(target, "active_channel_horizon_mask_ch", None)
    source_horizon = getattr(source, "active_channel_horizon_mask_ch", None)
    if not all(
        isinstance(value, torch.Tensor)
        for value in (target_channel, source_channel, target_horizon, source_horizon)
    ):
        raise ValueError("learnable output anchor source/target must expose persistent active masks.")
    if tuple(target_channel.shape) != tuple(source_channel.shape):
        raise ValueError(
            "learnable output anchor channel-mask shape mismatch: "
            f"source={tuple(source_channel.shape)}, target={tuple(target_channel.shape)}."
        )
    if tuple(target_horizon.shape) != tuple(source_horizon.shape):
        raise ValueError(
            "learnable output anchor channel-horizon-mask shape mismatch: "
            f"source={tuple(source_horizon.shape)}, target={tuple(target_horizon.shape)}."
        )
    target_channel.copy_(source_channel.to(device=target_channel.device, dtype=target_channel.dtype))
    target_horizon.copy_(source_horizon.to(device=target_horizon.device, dtype=target_horizon.dtype))
    return {
        "active_channel_mask": [float(v) for v in target_channel.detach().cpu().tolist()],
        "active_channel_horizon_mask_flat": [
            float(v) for v in target_horizon.detach().cpu().reshape(-1).tolist()
        ],
    }


def _normalize_learnable_output_anchor_adoption_scope(scope: object) -> str:
    value = str(scope if scope is not None else "global").lower().replace("-", "_")
    aliases = {
        "global": "global",
        "all": "global",
        "full": "global",
        "channel": "channel",
        "channels": "channel",
        "per_channel": "channel",
        "hybrid": "hybrid",
        "channel_greedy": "hybrid",
        "greedy_channel": "hybrid",
        "channel_hybrid": "hybrid",
        "channel_horizon": "channel_horizon",
        "channel_horizon_block": "channel_horizon",
        "channel_horizon_blocks": "channel_horizon",
        "horizon_block": "channel_horizon",
        "horizon_blocks": "channel_horizon",
        "block": "channel_horizon",
        "blocks": "channel_horizon",
    }
    if value not in aliases:
        raise ValueError(
            "moe.learnable_output_anchor.adoption.adoption_scope must be one of "
            "global, channel, hybrid, channel_horizon, or channel_horizon_block."
        )
    return aliases[value]


def _summarize_learnable_output_anchor_refiner(
    *,
    static_mse: float,
    static_mae: float,
    refined_mse: float,
    refined_mae: float,
    unmasked_refined_mse: Optional[float] = None,
    unmasked_refined_mae: Optional[float] = None,
    cfg: Optional[dict],
    skip_test: bool,
    num_channels: int,
    segment_metrics: Optional[List[Dict[str, float]]] = None,
    adopted_channel_mask: Optional[List[bool]] = None,
    adopted_channel_horizon_mask: Optional[List[List[bool]]] = None,
) -> Dict[str, object]:
    cfg = _normalize_learnable_output_anchor_cfg(cfg)
    adoption_cfg = cfg.get("adoption", {}) or {}
    if not isinstance(adoption_cfg, dict):
        adoption_cfg = {"adopt_on_val": bool(adoption_cfg)}

    selection_metric = str(adoption_cfg.get("selection_metric", "mse")).lower()
    if selection_metric == "val_mse":
        selection_metric = "mse"
    elif selection_metric == "val_mae":
        selection_metric = "mae"
    if selection_metric not in {"mse", "mae"}:
        raise ValueError(
            "moe.learnable_output_anchor.adoption.selection_metric must be mse, mae, val_mse, or val_mae."
        )

    min_abs_improvement = float(adoption_cfg.get("min_abs_improvement", 0.0))
    min_rel_improvement = float(adoption_cfg.get("min_rel_improvement", 0.0))
    aggregate_min_abs_improvement = float(
        adoption_cfg.get("aggregate_min_abs_improvement", min_abs_improvement)
    )
    aggregate_min_rel_improvement = float(
        adoption_cfg.get("aggregate_min_rel_improvement", min_rel_improvement)
    )
    aggregate_min_abs_mae_improvement = float(
        adoption_cfg.get("aggregate_min_abs_mae_improvement", 0.0)
    )
    aggregate_min_rel_mae_improvement = float(
        adoption_cfg.get("aggregate_min_rel_mae_improvement", 0.0)
    )
    aggregate_mae_improvement_guard_enabled = (
        "aggregate_min_abs_mae_improvement" in adoption_cfg
        or "aggregate_min_rel_mae_improvement" in adoption_cfg
    )
    max_abs_mae_regression = float(adoption_cfg.get("max_abs_mae_regression", 0.0))
    max_rel_mae_regression = float(adoption_cfg.get("max_rel_mae_regression", 0.0))
    aggregate_max_abs_mae_regression = float(
        adoption_cfg.get("aggregate_max_abs_mae_regression", max_abs_mae_regression)
    )
    aggregate_max_rel_mae_regression = float(
        adoption_cfg.get("aggregate_max_rel_mae_regression", max_rel_mae_regression)
    )
    max_segment_abs_degradation = float(adoption_cfg.get("max_segment_abs_degradation", 0.0))
    max_segment_rel_degradation = float(adoption_cfg.get("max_segment_rel_degradation", 0.0))

    static_metric = float(static_mse if selection_metric == "mse" else static_mae)
    refined_metric = float(refined_mse if selection_metric == "mse" else refined_mae)
    metric_gain = static_metric - refined_metric
    required_gain = max(
        aggregate_min_abs_improvement,
        aggregate_min_rel_improvement * abs(static_metric),
    )
    mae_gain = float(static_mae) - float(refined_mae)
    mae_regression = -mae_gain
    max_mae_regression = max(
        aggregate_max_abs_mae_regression,
        aggregate_max_rel_mae_regression * abs(float(static_mae)),
    )
    if aggregate_mae_improvement_guard_enabled:
        required_mae_gain = max(
            aggregate_min_abs_mae_improvement,
            aggregate_min_rel_mae_improvement * abs(float(static_mae)),
        )
    else:
        required_mae_gain = -max_mae_regression
    adopted = bool(
        metric_gain > required_gain
        and mae_regression <= max_mae_regression
        and mae_gain >= required_mae_gain
    )
    channel_count = max(0, int(num_channels))
    if adopted_channel_mask is not None and len(adopted_channel_mask) != channel_count:
        raise ValueError("learnable output anchor adopted_channel_mask length must match num_channels.")
    horizon_mask = None
    horizon_count = 0
    if adopted_channel_horizon_mask is not None:
        if len(adopted_channel_horizon_mask) != channel_count:
            raise ValueError(
                "learnable output anchor adopted_channel_horizon_mask row count must match num_channels."
            )
        horizon_mask = [[bool(v) for v in row] for row in adopted_channel_horizon_mask]
        horizon_lengths = {len(row) for row in horizon_mask}
        if len(horizon_lengths) != 1:
            raise ValueError("learnable output anchor adopted_channel_horizon_mask rows must share length.")
        horizon_count = int(next(iter(horizon_lengths), 0))
        horizon_channel_mask = [any(row) for row in horizon_mask]
        if adopted_channel_mask is not None and [bool(v) for v in adopted_channel_mask] != horizon_channel_mask:
            raise ValueError(
                "learnable output anchor adopted_channel_mask must match the channel-horizon mask projection."
            )
        adopted_channel_mask = horizon_channel_mask

    def _rel_pct(delta: float, denom: float) -> Optional[float]:
        denom = abs(float(denom))
        if denom <= 0.0:
            return None
        return 100.0 * float(delta) / denom

    mse_gain = float(static_mse) - float(refined_mse)
    if unmasked_refined_mse is None:
        unmasked_refined_mse = float(refined_mse)
    if unmasked_refined_mae is None:
        unmasked_refined_mae = float(refined_mae)
    segment_rows: List[Dict[str, object]] = []
    for idx, row in enumerate(segment_metrics or []):
        segment_static_mse = float(row["static_mse"])
        segment_static_mae = float(row["static_mae"])
        segment_refined_mse = float(row["refined_mse"])
        segment_refined_mae = float(row["refined_mae"])
        segment_static_metric = segment_static_mse if selection_metric == "mse" else segment_static_mae
        segment_refined_metric = segment_refined_mse if selection_metric == "mse" else segment_refined_mae
        segment_gain = segment_static_metric - segment_refined_metric
        segment_required = max(min_abs_improvement, min_rel_improvement * abs(segment_static_metric))
        segment_allowed_degradation = max(
            max_segment_abs_degradation,
            max_segment_rel_degradation * abs(segment_static_metric),
        )
        segment_mae_regression = segment_refined_mae - segment_static_mae
        segment_max_mae_regression = max(
            max_abs_mae_regression,
            max_rel_mae_regression * abs(segment_static_mae),
        )
        segment_rows.append(
            {
                "index": int(idx),
                "start": int(row.get("start", 0)),
                "end": int(row.get("end", 0)),
                "static_mse": segment_static_mse,
                "static_mae": segment_static_mae,
                "refined_mse": segment_refined_mse,
                "refined_mae": segment_refined_mae,
                "metric_gain": float(segment_gain),
                "required_gain": float(segment_required),
                "allowed_degradation": float(segment_allowed_degradation),
                "mae_regression": float(segment_mae_regression),
                "max_mae_regression": float(segment_max_mae_regression),
                "positive": bool(segment_gain > segment_required),
                "degraded": bool(segment_gain < -segment_allowed_degradation),
                "mae_regressed": bool(segment_mae_regression > segment_max_mae_regression),
            }
        )
    segment_count = len(segment_rows)
    min_positive_segments = int(adoption_cfg.get("min_positive_segments", 0))
    if min_positive_segments < 0:
        min_positive_segments = 0
    positive_segment_count = sum(1 for row in segment_rows if bool(row["positive"]))
    degraded_segment_count = sum(1 for row in segment_rows if bool(row["degraded"]))
    mae_regressed_segment_count = sum(1 for row in segment_rows if bool(row["mae_regressed"]))
    segment_guard_applied = segment_count > 1
    segment_guard_passed = (
        (not segment_guard_applied)
        or (
            positive_segment_count >= min_positive_segments
            and degraded_segment_count == 0
            and mae_regressed_segment_count == 0
        )
    )
    channel_mask = None if adopted_channel_mask is None else [bool(v) for v in adopted_channel_mask]
    if channel_mask is not None:
        adopted = bool(adopted and segment_guard_passed and any(channel_mask))
    else:
        adopted = bool(adopted and segment_guard_passed)
    final_channel_mask = channel_mask
    final_horizon_mask = horizon_mask
    if channel_mask is not None and not adopted:
        final_channel_mask = [False for _ in range(channel_count)]
    if horizon_mask is not None and not adopted:
        final_horizon_mask = [[False for _ in range(horizon_count)] for _ in range(channel_count)]
    horizon_count_per_channel = (
        [int(sum(row)) for row in final_horizon_mask]
        if final_horizon_mask is not None
        else []
    )
    adopted_channel_horizon_count = int(sum(horizon_count_per_channel))
    adopted_channel_horizon_total = int(channel_count * horizon_count) if final_horizon_mask is not None else 0
    if final_horizon_mask is not None:
        adopted_mask_kind = "channel_horizon"
    elif final_channel_mask is not None:
        adopted_mask_kind = "channel"
    else:
        adopted_mask_kind = "global"
    return {
        "enable": bool(cfg.get("enable", False)),
        "compare_on_val": True,
        "eval_skip_test": bool(skip_test),
        "test_read": False,
        "adopt_on_val": bool(adoption_cfg.get("adopt_on_val", True)),
        "adoption_scope": str(adoption_cfg.get("adoption_scope", "global")),
        "selection_metric": selection_metric,
        "min_abs_improvement": min_abs_improvement,
        "min_rel_improvement": min_rel_improvement,
        "aggregate_min_abs_improvement": aggregate_min_abs_improvement,
        "aggregate_min_rel_improvement": aggregate_min_rel_improvement,
        "aggregate_min_abs_mae_improvement": aggregate_min_abs_mae_improvement,
        "aggregate_min_rel_mae_improvement": aggregate_min_rel_mae_improvement,
        "aggregate_mae_improvement_guard_enabled": bool(aggregate_mae_improvement_guard_enabled),
        "max_abs_mae_regression": max_abs_mae_regression,
        "max_rel_mae_regression": max_rel_mae_regression,
        "aggregate_max_abs_mae_regression": aggregate_max_abs_mae_regression,
        "aggregate_max_rel_mae_regression": aggregate_max_rel_mae_regression,
        "max_segment_abs_degradation": max_segment_abs_degradation,
        "max_segment_rel_degradation": max_segment_rel_degradation,
        "val_static_mse": float(static_mse),
        "val_static_mae": float(static_mae),
        "val_refined_mse": float(refined_mse),
        "val_refined_mae": float(refined_mae),
        "val_refined_mse_unmasked": float(unmasked_refined_mse),
        "val_refined_mae_unmasked": float(unmasked_refined_mae),
        "mse_gain": mse_gain,
        "mae_gain": mae_gain,
        "mse_gain_rel_pct": _rel_pct(mse_gain, static_mse),
        "mae_gain_rel_pct": _rel_pct(mae_gain, static_mae),
        "metric_gain": float(metric_gain),
        "metric_gain_rel_pct": _rel_pct(metric_gain, static_metric),
        "required_gain": float(required_gain),
        "required_mae_gain": float(required_mae_gain),
        "mae_regression": float(mae_regression),
        "max_mae_regression": float(max_mae_regression),
        "adopted": adopted,
        "adopted_channel_count": (
            int(sum(final_channel_mask)) if final_channel_mask is not None else (channel_count if adopted else 0)
        ),
        "adopted_channel_mask": final_channel_mask if final_channel_mask is not None else [adopted for _ in range(channel_count)],
        "adopted_mask_kind": adopted_mask_kind,
        "adopted_channel_horizon_count": adopted_channel_horizon_count,
        "adopted_channel_horizon_total": adopted_channel_horizon_total,
        "adopted_horizon_count_per_channel": horizon_count_per_channel,
        "fallback_reason": None if adopted else "val_refiner_did_not_clear_static_anchor_guard",
        "final_eval_uses_learnable": adopted or not bool(adoption_cfg.get("adopt_on_val", True)),
        "segment_guard": {
            "applied": bool(segment_guard_applied),
            "segment_count": int(segment_count),
            "min_positive_segments": int(min_positive_segments),
            "positive_segment_count": int(positive_segment_count),
            "degraded_segment_count": int(degraded_segment_count),
            "mae_regressed_segment_count": int(mae_regressed_segment_count),
            "passed": bool(segment_guard_passed),
            "segments": segment_rows,
        },
    }


def _select_learnable_output_anchor_channel_mask(
    *,
    static_mse_c: torch.Tensor,
    refined_mse_c: torch.Tensor,
    static_mae_c: torch.Tensor,
    refined_mae_c: torch.Tensor,
    segment_channel_metrics: Optional[List[Dict[str, torch.Tensor]]],
    adoption_cfg: Dict[str, object],
) -> Tuple[torch.Tensor, Dict[str, object]]:
    adoption_cfg = adoption_cfg or {}
    selection_metric = str(adoption_cfg.get("selection_metric", "mse")).lower()
    if selection_metric == "val_mse":
        selection_metric = "mse"
    elif selection_metric == "val_mae":
        selection_metric = "mae"
    if selection_metric not in {"mse", "mae"}:
        raise ValueError(
            "moe.learnable_output_anchor.adoption.selection_metric must be mse, mae, val_mse, or val_mae."
        )
    scope = str(adoption_cfg.get("adoption_scope", "channel")).lower().replace("-", "_")
    is_hybrid = scope in {"hybrid", "channel_greedy", "greedy_channel", "channel_hybrid"}

    static_mse_c = static_mse_c.detach().cpu().float()
    refined_mse_c = refined_mse_c.detach().cpu().float()
    static_mae_c = static_mae_c.detach().cpu().float()
    refined_mae_c = refined_mae_c.detach().cpu().float()
    if not (
        static_mse_c.shape == refined_mse_c.shape == static_mae_c.shape == refined_mae_c.shape
    ):
        raise ValueError("learnable output anchor per-channel metric tensors must share shape.")

    static_metric_c = static_mse_c if selection_metric == "mse" else static_mae_c
    refined_metric_c = refined_mse_c if selection_metric == "mse" else refined_mae_c
    min_abs = float(adoption_cfg.get("min_abs_improvement", 0.0))
    min_rel = float(adoption_cfg.get("min_rel_improvement", 0.0))
    max_abs_mae_regression = float(adoption_cfg.get("max_abs_mae_regression", 0.0))
    max_rel_mae_regression = float(adoption_cfg.get("max_rel_mae_regression", 0.0))
    max_segment_abs_degradation = float(adoption_cfg.get("max_segment_abs_degradation", 0.0))
    max_segment_rel_degradation = float(adoption_cfg.get("max_segment_rel_degradation", 0.0))
    min_positive_segments = max(0, int(adoption_cfg.get("min_positive_segments", 0)))
    aggregate_min_abs = float(adoption_cfg.get("aggregate_min_abs_improvement", min_abs))
    aggregate_min_rel = float(adoption_cfg.get("aggregate_min_rel_improvement", min_rel))
    aggregate_min_abs_mae_improvement = float(
        adoption_cfg.get("aggregate_min_abs_mae_improvement", 0.0)
    )
    aggregate_min_rel_mae_improvement = float(
        adoption_cfg.get("aggregate_min_rel_mae_improvement", 0.0)
    )
    aggregate_mae_improvement_guard_enabled = (
        "aggregate_min_abs_mae_improvement" in adoption_cfg
        or "aggregate_min_rel_mae_improvement" in adoption_cfg
    )
    aggregate_max_abs_mae_regression = float(
        adoption_cfg.get("aggregate_max_abs_mae_regression", max_abs_mae_regression)
    )
    aggregate_max_rel_mae_regression = float(
        adoption_cfg.get("aggregate_max_rel_mae_regression", max_rel_mae_regression)
    )

    metric_gain_c = static_metric_c - refined_metric_c
    required_c = torch.maximum(
        torch.full_like(static_metric_c, min_abs),
        min_rel * static_metric_c.abs().clamp_min(1.0e-12),
    )
    mae_regression_c = refined_mae_c - static_mae_c
    max_mae_regression_c = torch.maximum(
        torch.full_like(static_mae_c, max_abs_mae_regression),
        max_rel_mae_regression * static_mae_c.abs().clamp_min(1.0e-12),
    )
    strict_keep_c = (metric_gain_c > required_c) & (mae_regression_c <= max_mae_regression_c)

    segment_rows: List[Dict[str, torch.Tensor]] = []
    for row in segment_channel_metrics or []:
        segment_rows.append(
            {
                "static_mse_c": row["static_mse_c"].detach().cpu().float(),
                "refined_mse_c": row["refined_mse_c"].detach().cpu().float(),
                "static_mae_c": row["static_mae_c"].detach().cpu().float(),
                "refined_mae_c": row["refined_mae_c"].detach().cpu().float(),
            }
        )
    if len(segment_rows) > 1:
        segment_keep_rows = []
        for row in segment_rows:
            seg_static_metric_c = (
                row["static_mse_c"] if selection_metric == "mse" else row["static_mae_c"]
            )
            seg_refined_metric_c = (
                row["refined_mse_c"] if selection_metric == "mse" else row["refined_mae_c"]
            )
            seg_gain_c = seg_static_metric_c - seg_refined_metric_c
            seg_required_c = torch.maximum(
                torch.full_like(seg_static_metric_c, min_abs),
                min_rel * seg_static_metric_c.abs().clamp_min(1.0e-12),
            )
            seg_allowed_degradation_c = torch.maximum(
                torch.full_like(seg_static_metric_c, max_segment_abs_degradation),
                max_segment_rel_degradation * seg_static_metric_c.abs().clamp_min(1.0e-12),
            )
            seg_mae_regression_c = row["refined_mae_c"] - row["static_mae_c"]
            seg_max_mae_regression_c = torch.maximum(
                torch.full_like(row["static_mae_c"], max_abs_mae_regression),
                max_rel_mae_regression * row["static_mae_c"].abs().clamp_min(1.0e-12),
            )
            segment_keep_rows.append(
                {
                    "positive": seg_gain_c > seg_required_c,
                    "not_degraded": seg_gain_c >= -seg_allowed_degradation_c,
                    "mae_ok": seg_mae_regression_c <= seg_max_mae_regression_c,
                }
            )
        positive_sc = torch.stack([row["positive"] for row in segment_keep_rows], dim=0)
        not_degraded_sc = torch.stack([row["not_degraded"] for row in segment_keep_rows], dim=0)
        mae_ok_sc = torch.stack([row["mae_ok"] for row in segment_keep_rows], dim=0)
        if min_positive_segments > 0:
            strict_keep_c &= positive_sc.sum(dim=0) >= min_positive_segments
        strict_keep_c &= not_degraded_sc.all(dim=0)
        strict_keep_c &= mae_ok_sc.all(dim=0)

    def _aggregate(mask_c: torch.Tensor) -> Dict[str, object]:
        mask_c = mask_c.bool()
        mixed_mse_c = torch.where(mask_c, refined_mse_c, static_mse_c)
        mixed_mae_c = torch.where(mask_c, refined_mae_c, static_mae_c)
        static_mse = float(static_mse_c.mean().item())
        static_mae = float(static_mae_c.mean().item())
        refined_mse = float(mixed_mse_c.mean().item())
        refined_mae = float(mixed_mae_c.mean().item())
        static_metric = static_mse if selection_metric == "mse" else static_mae
        refined_metric = refined_mse if selection_metric == "mse" else refined_mae
        metric_gain = float(static_metric - refined_metric)
        required_gain = max(aggregate_min_abs, aggregate_min_rel * abs(static_metric))
        mae_regression = float(refined_mae - static_mae)
        mae_improvement = -mae_regression
        max_mae_regression = max(
            aggregate_max_abs_mae_regression,
            aggregate_max_rel_mae_regression * abs(static_mae),
        )
        if aggregate_mae_improvement_guard_enabled:
            required_mae_improvement = max(
                aggregate_min_abs_mae_improvement,
                aggregate_min_rel_mae_improvement * abs(static_mae),
            )
        else:
            required_mae_improvement = -max_mae_regression
        segments: List[Dict[str, object]] = []
        for idx, row in enumerate(segment_rows):
            mixed_seg_mse_c = torch.where(mask_c, row["refined_mse_c"], row["static_mse_c"])
            mixed_seg_mae_c = torch.where(mask_c, row["refined_mae_c"], row["static_mae_c"])
            segment_static_mse = float(row["static_mse_c"].mean().item())
            segment_static_mae = float(row["static_mae_c"].mean().item())
            segment_refined_mse = float(mixed_seg_mse_c.mean().item())
            segment_refined_mae = float(mixed_seg_mae_c.mean().item())
            segment_static_metric = segment_static_mse if selection_metric == "mse" else segment_static_mae
            segment_refined_metric = segment_refined_mse if selection_metric == "mse" else segment_refined_mae
            segment_gain = float(segment_static_metric - segment_refined_metric)
            segment_required = max(min_abs, min_rel * abs(segment_static_metric))
            segment_allowed_degradation = max(
                max_segment_abs_degradation,
                max_segment_rel_degradation * abs(segment_static_metric),
            )
            segment_mae_regression = float(segment_refined_mae - segment_static_mae)
            segment_max_mae_regression = max(
                max_abs_mae_regression,
                max_rel_mae_regression * abs(segment_static_mae),
            )
            segments.append(
                {
                    "index": int(idx),
                    "metric_gain": segment_gain,
                    "required_gain": float(segment_required),
                    "allowed_degradation": float(segment_allowed_degradation),
                    "mae_regression": segment_mae_regression,
                    "max_mae_regression": float(segment_max_mae_regression),
                    "positive": bool(segment_gain > segment_required),
                    "degraded": bool(segment_gain < -segment_allowed_degradation),
                    "mae_regressed": bool(segment_mae_regression > segment_max_mae_regression),
                }
            )
        positive_segment_count = sum(1 for row in segments if bool(row["positive"]))
        degraded_segment_count = sum(1 for row in segments if bool(row["degraded"]))
        mae_regressed_segment_count = sum(1 for row in segments if bool(row["mae_regressed"]))
        segment_guard_applied = len(segments) > 1
        segment_guard_passed = (
            (not segment_guard_applied)
            or (
                positive_segment_count >= min_positive_segments
                and degraded_segment_count == 0
                and mae_regressed_segment_count == 0
            )
        )
        passed = bool(
            metric_gain > required_gain
            and mae_regression <= max_mae_regression
            and mae_improvement >= required_mae_improvement
            and segment_guard_passed
            and bool(mask_c.any().item())
        )
        return {
            "static_mse": static_mse,
            "static_mae": static_mae,
            "refined_mse": refined_mse,
            "refined_mae": refined_mae,
            "metric_gain": metric_gain,
            "required_gain": float(required_gain),
            "mae_improvement": mae_improvement,
            "required_mae_improvement": float(required_mae_improvement),
            "mae_improvement_guard_enabled": bool(aggregate_mae_improvement_guard_enabled),
            "mae_regression": mae_regression,
            "max_mae_regression": float(max_mae_regression),
            "positive_segment_count": int(positive_segment_count),
            "degraded_segment_count": int(degraded_segment_count),
            "mae_regressed_segment_count": int(mae_regressed_segment_count),
            "segment_guard_passed": bool(segment_guard_passed),
            "passed": passed,
            "segments": segments,
        }

    keep_c = strict_keep_c.clone()
    added_channels: List[int] = []
    if is_hybrid:
        current = _aggregate(keep_c)
        candidate_c = (metric_gain_c > required_c) & (~keep_c)
        order = torch.argsort(metric_gain_c, descending=True).tolist()
        for idx in order:
            idx = int(idx)
            if not bool(candidate_c[idx].item()):
                continue
            proposal_c = keep_c.clone()
            proposal_c[idx] = True
            proposal = _aggregate(proposal_c)
            if (
                proposal["mae_regression"] <= proposal["max_mae_regression"]
                and proposal["degraded_segment_count"] == 0
                and proposal["mae_regressed_segment_count"] == 0
                and proposal["metric_gain"] > current["metric_gain"] + 1.0e-12
            ):
                keep_c = proposal_c
                current = proposal
                added_channels.append(idx)

    aggregate = _aggregate(keep_c)
    diagnostics = {
        "scope": scope,
        "selection_metric": selection_metric,
        "strict_channel_count": int(strict_keep_c.sum().item()),
        "adopted_channel_count": int(keep_c.sum().item()),
        "added_channel_count": int(len(added_channels)),
        "added_channels": added_channels,
        "aggregate": aggregate,
    }
    return keep_c, diagnostics


def _select_learnable_output_anchor_channel_horizon_mask(
    *,
    static_mse_ch: torch.Tensor,
    refined_mse_ch: torch.Tensor,
    static_mae_ch: torch.Tensor,
    refined_mae_ch: torch.Tensor,
    adoption_cfg: Dict[str, object],
    segment_channel_horizon_metrics: Optional[List[Dict[str, torch.Tensor]]] = None,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    adoption_cfg = adoption_cfg or {}
    selection_metric = str(adoption_cfg.get("selection_metric", "mse")).lower()
    if selection_metric == "val_mse":
        selection_metric = "mse"
    elif selection_metric == "val_mae":
        selection_metric = "mae"
    if selection_metric not in {"mse", "mae"}:
        raise ValueError(
            "moe.learnable_output_anchor.adoption.selection_metric must be mse, mae, val_mse, or val_mae."
        )
    scope = _normalize_learnable_output_anchor_adoption_scope(
        adoption_cfg.get("adoption_scope", "channel_horizon")
    )
    if scope != "channel_horizon":
        raise ValueError("channel-horizon adoption requires adoption_scope=channel_horizon or channel_horizon_block.")

    static_mse_ch = static_mse_ch.detach().cpu().float()
    refined_mse_ch = refined_mse_ch.detach().cpu().float()
    static_mae_ch = static_mae_ch.detach().cpu().float()
    refined_mae_ch = refined_mae_ch.detach().cpu().float()
    if static_mse_ch.ndim != 2:
        raise ValueError("learnable output anchor channel-horizon metrics must have shape [channel, horizon].")
    if not (
        static_mse_ch.shape == refined_mse_ch.shape == static_mae_ch.shape == refined_mae_ch.shape
    ):
        raise ValueError("learnable output anchor channel-horizon metric tensors must share shape.")
    channel_count, horizon = int(static_mse_ch.shape[0]), int(static_mse_ch.shape[1])
    horizon_segments = int(
        adoption_cfg.get(
            "horizon_segments",
            adoption_cfg.get("horizon_blocks", adoption_cfg.get("horizon_block_count", 4)),
        )
    )
    horizon_segments = max(1, min(horizon_segments, horizon))
    block_ranges = _contiguous_segment_ranges(horizon, horizon_segments)

    min_abs = float(adoption_cfg.get("min_abs_improvement", 0.0))
    min_rel = float(adoption_cfg.get("min_rel_improvement", 0.0))
    max_abs_mae_regression = float(adoption_cfg.get("max_abs_mae_regression", 0.0))
    max_rel_mae_regression = float(adoption_cfg.get("max_rel_mae_regression", 0.0))
    aggregate_min_abs = float(adoption_cfg.get("aggregate_min_abs_improvement", min_abs))
    aggregate_min_rel = float(adoption_cfg.get("aggregate_min_rel_improvement", min_rel))
    aggregate_min_abs_mae_improvement = float(
        adoption_cfg.get("aggregate_min_abs_mae_improvement", 0.0)
    )
    aggregate_min_rel_mae_improvement = float(
        adoption_cfg.get("aggregate_min_rel_mae_improvement", 0.0)
    )
    aggregate_mae_improvement_guard_enabled = (
        "aggregate_min_abs_mae_improvement" in adoption_cfg
        or "aggregate_min_rel_mae_improvement" in adoption_cfg
    )
    aggregate_max_abs_mae_regression = float(
        adoption_cfg.get("aggregate_max_abs_mae_regression", max_abs_mae_regression)
    )
    aggregate_max_rel_mae_regression = float(
        adoption_cfg.get("aggregate_max_rel_mae_regression", max_rel_mae_regression)
    )
    max_segment_abs_degradation = float(adoption_cfg.get("max_segment_abs_degradation", 0.0))
    max_segment_rel_degradation = float(adoption_cfg.get("max_segment_rel_degradation", 0.0))
    min_positive_segments = max(0, int(adoption_cfg.get("min_positive_segments", 0)))
    candidate_segment_guard = bool(
        adoption_cfg.get(
            "candidate_segment_guard",
            adoption_cfg.get("block_candidate_segment_guard", True),
        )
    )

    segment_rows: List[Dict[str, torch.Tensor]] = []
    for row in segment_channel_horizon_metrics or []:
        segment_rows.append(
            {
                "static_mse_ch": row["static_mse_ch"].detach().cpu().float(),
                "refined_mse_ch": row["refined_mse_ch"].detach().cpu().float(),
                "static_mae_ch": row["static_mae_ch"].detach().cpu().float(),
                "refined_mae_ch": row["refined_mae_ch"].detach().cpu().float(),
            }
        )

    keep_ch = torch.zeros_like(static_mse_ch, dtype=torch.bool)
    block_diagnostics: List[Dict[str, object]] = []
    for block_idx, (start, end) in enumerate(block_ranges):
        static_block_mse_c = static_mse_ch[:, start:end].mean(dim=1)
        refined_block_mse_c = refined_mse_ch[:, start:end].mean(dim=1)
        static_block_mae_c = static_mae_ch[:, start:end].mean(dim=1)
        refined_block_mae_c = refined_mae_ch[:, start:end].mean(dim=1)
        static_metric_c = static_block_mse_c if selection_metric == "mse" else static_block_mae_c
        refined_metric_c = refined_block_mse_c if selection_metric == "mse" else refined_block_mae_c
        gain_c = static_metric_c - refined_metric_c
        required_c = torch.maximum(
            torch.full_like(static_metric_c, min_abs),
            min_rel * static_metric_c.abs().clamp_min(1.0e-12),
        )
        mae_regression_c = refined_block_mae_c - static_block_mae_c
        max_mae_regression_c = torch.maximum(
            torch.full_like(static_block_mae_c, max_abs_mae_regression),
            max_rel_mae_regression * static_block_mae_c.abs().clamp_min(1.0e-12),
        )
        keep_c = (gain_c > required_c) & (mae_regression_c <= max_mae_regression_c)

        if candidate_segment_guard and len(segment_rows) > 1:
            positive_count_c = torch.zeros(channel_count, dtype=torch.long)
            not_degraded_c = torch.ones(channel_count, dtype=torch.bool)
            mae_ok_c = torch.ones(channel_count, dtype=torch.bool)
            for row in segment_rows:
                seg_static_mse_c = row["static_mse_ch"][:, start:end].mean(dim=1)
                seg_refined_mse_c = row["refined_mse_ch"][:, start:end].mean(dim=1)
                seg_static_mae_c = row["static_mae_ch"][:, start:end].mean(dim=1)
                seg_refined_mae_c = row["refined_mae_ch"][:, start:end].mean(dim=1)
                seg_static_metric_c = seg_static_mse_c if selection_metric == "mse" else seg_static_mae_c
                seg_refined_metric_c = seg_refined_mse_c if selection_metric == "mse" else seg_refined_mae_c
                seg_gain_c = seg_static_metric_c - seg_refined_metric_c
                seg_required_c = torch.maximum(
                    torch.full_like(seg_static_metric_c, min_abs),
                    min_rel * seg_static_metric_c.abs().clamp_min(1.0e-12),
                )
                seg_allowed_degradation_c = torch.maximum(
                    torch.full_like(seg_static_metric_c, max_segment_abs_degradation),
                    max_segment_rel_degradation * seg_static_metric_c.abs().clamp_min(1.0e-12),
                )
                seg_mae_regression_c = seg_refined_mae_c - seg_static_mae_c
                seg_max_mae_regression_c = torch.maximum(
                    torch.full_like(seg_static_mae_c, max_abs_mae_regression),
                    max_rel_mae_regression * seg_static_mae_c.abs().clamp_min(1.0e-12),
                )
                positive_count_c += (seg_gain_c > seg_required_c).to(dtype=torch.long)
                not_degraded_c &= seg_gain_c >= -seg_allowed_degradation_c
                mae_ok_c &= seg_mae_regression_c <= seg_max_mae_regression_c
            if min_positive_segments > 0:
                keep_c &= positive_count_c >= min_positive_segments
            keep_c &= not_degraded_c & mae_ok_c

        keep_ch[:, start:end] = keep_c.view(channel_count, 1)
        block_diagnostics.append(
            {
                "index": int(block_idx),
                "start": int(start),
                "end": int(end),
                "kept_channel_count": int(keep_c.sum().item()),
                "mean_metric_gain": float(gain_c.mean().item()),
            }
        )

    def _aggregate(mask_ch: torch.Tensor) -> Dict[str, object]:
        mask_ch = mask_ch.bool()
        mixed_mse_ch = torch.where(mask_ch, refined_mse_ch, static_mse_ch)
        mixed_mae_ch = torch.where(mask_ch, refined_mae_ch, static_mae_ch)
        static_mse = float(static_mse_ch.mean().item())
        static_mae = float(static_mae_ch.mean().item())
        refined_mse = float(mixed_mse_ch.mean().item())
        refined_mae = float(mixed_mae_ch.mean().item())
        static_metric = static_mse if selection_metric == "mse" else static_mae
        refined_metric = refined_mse if selection_metric == "mse" else refined_mae
        metric_gain = float(static_metric - refined_metric)
        required_gain = max(aggregate_min_abs, aggregate_min_rel * abs(static_metric))
        mae_regression = float(refined_mae - static_mae)
        mae_improvement = -mae_regression
        max_mae_regression = max(
            aggregate_max_abs_mae_regression,
            aggregate_max_rel_mae_regression * abs(static_mae),
        )
        if aggregate_mae_improvement_guard_enabled:
            required_mae_improvement = max(
                aggregate_min_abs_mae_improvement,
                aggregate_min_rel_mae_improvement * abs(static_mae),
            )
        else:
            required_mae_improvement = -max_mae_regression

        segments: List[Dict[str, object]] = []
        for idx, row in enumerate(segment_rows):
            seg_mixed_mse_ch = torch.where(mask_ch, row["refined_mse_ch"], row["static_mse_ch"])
            seg_mixed_mae_ch = torch.where(mask_ch, row["refined_mae_ch"], row["static_mae_ch"])
            seg_static_mse = float(row["static_mse_ch"].mean().item())
            seg_static_mae = float(row["static_mae_ch"].mean().item())
            seg_refined_mse = float(seg_mixed_mse_ch.mean().item())
            seg_refined_mae = float(seg_mixed_mae_ch.mean().item())
            seg_static_metric = seg_static_mse if selection_metric == "mse" else seg_static_mae
            seg_refined_metric = seg_refined_mse if selection_metric == "mse" else seg_refined_mae
            seg_gain = float(seg_static_metric - seg_refined_metric)
            seg_required = max(min_abs, min_rel * abs(seg_static_metric))
            seg_allowed_degradation = max(
                max_segment_abs_degradation,
                max_segment_rel_degradation * abs(seg_static_metric),
            )
            seg_mae_regression = float(seg_refined_mae - seg_static_mae)
            seg_max_mae_regression = max(
                max_abs_mae_regression,
                max_rel_mae_regression * abs(seg_static_mae),
            )
            segments.append(
                {
                    "index": int(idx),
                    "metric_gain": seg_gain,
                    "required_gain": float(seg_required),
                    "allowed_degradation": float(seg_allowed_degradation),
                    "mae_regression": seg_mae_regression,
                    "max_mae_regression": float(seg_max_mae_regression),
                    "positive": bool(seg_gain > seg_required),
                    "degraded": bool(seg_gain < -seg_allowed_degradation),
                    "mae_regressed": bool(seg_mae_regression > seg_max_mae_regression),
                }
            )
        positive_segment_count = sum(1 for row in segments if bool(row["positive"]))
        degraded_segment_count = sum(1 for row in segments if bool(row["degraded"]))
        mae_regressed_segment_count = sum(1 for row in segments if bool(row["mae_regressed"]))
        segment_guard_applied = len(segments) > 1
        segment_guard_passed = (
            (not segment_guard_applied)
            or (
                positive_segment_count >= min_positive_segments
                and degraded_segment_count == 0
                and mae_regressed_segment_count == 0
            )
        )
        passed = bool(
            metric_gain > required_gain
            and mae_regression <= max_mae_regression
            and mae_improvement >= required_mae_improvement
            and segment_guard_passed
            and bool(mask_ch.any().item())
        )
        return {
            "static_mse": static_mse,
            "static_mae": static_mae,
            "refined_mse": refined_mse,
            "refined_mae": refined_mae,
            "metric_gain": metric_gain,
            "required_gain": float(required_gain),
            "mae_improvement": mae_improvement,
            "required_mae_improvement": float(required_mae_improvement),
            "mae_improvement_guard_enabled": bool(aggregate_mae_improvement_guard_enabled),
            "mae_regression": mae_regression,
            "max_mae_regression": float(max_mae_regression),
            "positive_segment_count": int(positive_segment_count),
            "degraded_segment_count": int(degraded_segment_count),
            "mae_regressed_segment_count": int(mae_regressed_segment_count),
            "segment_guard_passed": bool(segment_guard_passed),
            "passed": passed,
            "segments": segments,
        }

    aggregate = _aggregate(keep_ch)
    diagnostics = {
        "scope": "channel_horizon",
        "selection_metric": selection_metric,
        "horizon_segments": int(horizon_segments),
        "candidate_segment_guard": bool(candidate_segment_guard),
        "block_ranges": [[int(start), int(end)] for start, end in block_ranges],
        "blocks": block_diagnostics,
        "adopted_channel_count": int(keep_ch.any(dim=1).sum().item()),
        "adopted_channel_horizon_count": int(keep_ch.sum().item()),
        "adopted_channel_horizon_total": int(channel_count * horizon),
        "adopted_horizon_count_per_channel": [int(v) for v in keep_ch.sum(dim=1).tolist()],
        "aggregate": aggregate,
    }
    return keep_ch, diagnostics


def _finalize_channel_horizon_metric_collector(
    collector: Dict[str, object],
) -> Tuple[torch.Tensor, torch.Tensor]:
    se_ch = collector.get("se_ch")
    ae_ch = collector.get("ae_ch")
    count = int(collector.get("count", 0) or 0)
    if not isinstance(se_ch, torch.Tensor) or not isinstance(ae_ch, torch.Tensor) or count <= 0:
        raise ValueError("channel-horizon metric collector did not receive any prediction windows.")
    denom = float(max(count, 1))
    return (se_ch.detach().cpu().float() / denom), (ae_ch.detach().cpu().float() / denom)


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


def _set_optimizer_lr_scale(optimizers: List[Optional[torch.optim.Optimizer]], scale: float) -> None:
    scale = float(scale)
    for optimizer in optimizers:
        if optimizer is None:
            continue
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", float(group["lr"]))
            group["lr"] = float(group["initial_lr"]) * scale


def _optimizer_slot_active(stopped: torch.Tensor, k: int, shared_moe: bool) -> bool:
    stopped_bool = stopped.detach().to(dtype=torch.bool).view(-1)
    if int(k) < 0 or int(k) >= int(stopped_bool.numel()):
        return False
    if bool(shared_moe) and int(k) == 0:
        return not bool(stopped_bool.all().item())
    return not bool(stopped_bool[int(k)].item())


def _training_cluster_weight(
    cluster_weight_k: torch.Tensor,
    stopped: torch.Tensor,
    shared_moe: bool,
) -> torch.Tensor:
    if not bool(shared_moe):
        return cluster_weight_k
    active = (~stopped.detach().to(device=cluster_weight_k.device, dtype=torch.bool)).to(dtype=cluster_weight_k.dtype)
    if float(active.sum().item()) <= 0.0:
        return cluster_weight_k
    return cluster_weight_k * active


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
    learnable_anchor_params: Optional[List[nn.Parameter]] = None,
    base_weight_decay: float = 0.0,
    moe_weight_decay: Optional[float] = None,
    pred_residual_weight_decay: Optional[float] = None,
    learnable_anchor_weight_decay: Optional[float] = None,
    learnable_anchor_lr: Optional[float] = None,
    base_lr: Optional[float] = None,
) -> List[Dict[str, object]]:
    learnable_anchor_params = list(learnable_anchor_params or [])
    moe_params: List[nn.Parameter] = []
    moe_params.extend(gate_params)
    moe_params.extend(dynamic_lambda_params)
    moe_params.extend(learnable_lambda_params)

    groups: List[Dict[str, object]] = []
    if base_lr is not None and len(base_params) > 0:
        groups.append({
            "params": base_params,
            "weight_decay": float(base_weight_decay),
            "lr": float(base_lr),
        })
        if moe_weight_decay is None and len(moe_params) > 0:
            groups.append({
                "params": moe_params,
                "weight_decay": float(base_weight_decay),
            })
    elif moe_weight_decay is None:
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
    if len(learnable_anchor_params) > 0:
        anchor_wd = learnable_anchor_weight_decay
        if anchor_wd is None:
            anchor_wd = moe_weight_decay if moe_weight_decay is not None else base_weight_decay
        group: Dict[str, object] = {
            "params": learnable_anchor_params,
            "weight_decay": float(anchor_wd),
        }
        if learnable_anchor_lr is not None:
            group["lr"] = float(learnable_anchor_lr)
        groups.append(group)
    return groups


def _validate_semantic_bank_training_provenance(
    provenance: Optional[Dict[str, object]],
) -> bool:
    """Validate the immutable Stage-1 producer recipe independently of inference shape."""
    if provenance is None:
        raise ValueError(
            "Semantic bank checkpoint is missing pred_residual_training_provenance."
        )
    common: Dict[str, object] = {
        "stage": "semantic_bank_stage1",
        "candidate_supervision_weight": 1.0,
        "independent_optimization": True,
        "independent_optimizer": "sgd",
        "need_quantile": 0.75,
        "need_patch_len": 12,
        "only_allowed": False,
        "include_intervention": False,
        "include_selector": False,
        "include_patch_route": False,
        "penalty_scale_source": "frozen_backbone_patch",
        "penalty_scale_floor": 1.0e-3,
        "penalty_scale_rule": "positive_patch_mean_floored",
        "threshold_source": "train_frozen_pure_backbone_patch_q75",
        "threshold_interpolation": "linear",
        "need_comparison": ">=",
        "optimizer": {
            "name": "sgd",
            "momentum": 0.0,
            "dampening": 0.0,
            "nesterov": False,
        },
        "scheduler": "none",
        "checkpoint_selection": "semantic_per_expert",
    }
    recipes: Dict[int, Dict[str, object]] = {
        1: {
            "version": 1,
            "candidate_loss": "acceptance_guarded_own_penalty",
            "high_mse_relative_tolerance": 0.0,
            "low_mse_relative_tolerance": 1.0e-3,
            "low_high_rms_ratio_max": 0.25,
            "forecast_mse_weight": 1.0,
            "noop_weight": 1.0,
            "constraint_weight": 1.0,
            "constraint_eps": 1.0e-8,
        },
        2: {
            "version": 2,
            "candidate_loss": "need_weighted_own_penalty_mse",
            "high_mse_relative_tolerance": 0.0,
            "low_mse_relative_tolerance": 1.0e-3,
            "low_high_rms_ratio_max": 0.25,
            "forecast_mse_weight": 1.0,
            "noop_weight": 1.0,
        },
        3: {
            "version": 3,
            "candidate_loss": "high_need_own_penalty",
            "forecast_mse_weight": 0.0,
            "noop_weight": 0.0,
            "active_penalty": provenance.get("active_penalty"),
            "validation_blocks": 3,
            "min_high_need_improved_fraction": 0.60,
            "min_matching_gain_by_name": {
                "level": 0.10,
                "trend": 0.10,
                "d2_match": 0.10,
                "diff_amp": 0.70,
            },
            "acceptance_mode": "semantic_only_high_need_patch",
        },
        4: {
            "version": 4,
            "candidate_loss": "high_need_own_penalty",
            "forecast_mse_weight": 0.0,
            "noop_weight": 0.0,
            "active_penalty": provenance.get("active_penalty"),
            "validation_blocks": 3,
            "min_high_need_improved_fraction": 0.60,
            "min_matching_gain_by_name": {
                "level": 0.10,
                "trend": 0.10,
                "d2_match": 0.10,
                "diff_amp": 0.70,
            },
            "acceptance_mode": "semantic_only_high_need_patch",
            "raw_gradient_accumulation": {
                "enabled": True,
                "microbatches": 16,
                "missing_gradient": "zero",
                "reduction": "mean_by_actual_count",
                "clip": "once_after_mean",
                "weight_decay": "once_per_optimizer_step",
                "tail": "actual_count",
            },
        },
        5: {
            "version": 5,
            "candidate_loss": "level_residual_gate",
            "forecast_mse_weight": 0.0,
            "noop_weight": 0.0,
            "active_penalty": "level",
            "validation_blocks": 3,
            "min_high_need_improved_fraction": 0.60,
            "min_matching_gain_by_name": {
                "level": 0.10,
                "trend": 0.10,
                "d2_match": 0.10,
                "diff_amp": 0.70,
            },
            "acceptance_mode": "semantic_only_high_need_patch",
            "level_loss_weights": {
                "amplitude": 1.0,
                "need_bce": 1.0,
                "executed": 1.0,
            },
            "raw_gradient_accumulation": {
                "enabled": True,
                "microbatches": 16,
                "missing_gradient": "zero",
                "reduction": "mean_by_actual_count",
                "clip": "once_after_mean",
                "weight_decay": "once_per_optimizer_step",
                "tail": "actual_count",
            },
        },
        6: {
            "version": 6,
            "candidate_loss": "level_residual_separate_gate",
            "forecast_mse_weight": 0.0,
            "noop_weight": 0.0,
            "active_penalty": "level",
            "validation_blocks": 3,
            "min_high_need_improved_fraction": 0.60,
            "min_matching_gain_by_name": {
                "level": 0.10,
                "trend": 0.10,
                "d2_match": 0.10,
                "diff_amp": 0.70,
            },
            "acceptance_mode": "semantic_only_high_need_patch_with_local_gate",
            "level_loss_weights": {
                "amplitude": 1.0,
                "need_balanced_bce": 1.0,
                "executed": 0.0,
            },
            "level_need_positive_weight": 3.0,
            "level_optimizer_groups": ["amplitude", "need_gate"],
            "gradient_clip": "independent_per_level_group",
            "raw_gradient_accumulation": {
                "enabled": True,
                "microbatches": 16,
                "missing_gradient": "zero",
                "reduction": "mean_by_actual_count",
                "clip": "once_after_mean_per_level_group",
                "weight_decay": "once_per_optimizer_step_per_level_group",
                "tail": "actual_count",
            },
        },
        7: {
            "version": 7,
            "independent_optimizer": "mixed_level_adam_amplitude_sgd_need_gate",
            "candidate_loss": "level_residual_separate_gate",
            "forecast_mse_weight": 0.0,
            "noop_weight": 0.0,
            "active_penalty": "level",
            "validation_blocks": 3,
            "min_high_need_improved_fraction": 0.60,
            "min_matching_gain_by_name": {
                "level": 0.10,
                "trend": 0.10,
                "d2_match": 0.10,
                "diff_amp": 0.70,
            },
            "acceptance_mode": "semantic_only_high_need_patch_with_local_gate",
            "level_loss_weights": {
                "amplitude": 1.0,
                "need_balanced_bce": 1.0,
                "executed": 0.0,
            },
            "level_need_positive_weight": 3.0,
            "level_optimizer_groups": ["amplitude", "need_gate"],
            "gradient_clip": "independent_per_level_group",
            "optimizer": {
                "name": "disjoint_level_adam_amplitude_sgd_need_gate",
                "amplitude": {
                    "name": "adam",
                    "lr": 1.0e-3,
                    "weight_decay": 0.0,
                    "betas": [0.9, 0.999],
                    "eps": 1.0e-8,
                    "amsgrad": False,
                },
                "need_gate": {
                    "name": "sgd",
                    "lr": 3.0e-4,
                    "weight_decay": 1.0e-3,
                    "momentum": 0.0,
                    "dampening": 0.0,
                    "nesterov": False,
                },
            },
            "raw_gradient_accumulation": {
                "enabled": True,
                "microbatches": 16,
                "missing_gradient": "zero",
                "reduction": "mean_by_actual_count",
                "clip": "once_after_mean_per_level_group",
                "weight_decay": "once_per_optimizer_step_per_level_group",
                "tail": "actual_count",
            },
        },
    }
    recipes[8] = {
        **recipes[7],
        "version": 8,
        "candidate_loss": "level_residual_high_need_separate_gate",
        "level_amplitude_population": "detached_q75_high_need_only",
        "level_acceptance_candidate": "raw_amplitude",
        "level_need_gate_acceptance_role": "diagnostic_only",
    }
    version = provenance.get("version")
    if version not in recipes:
        raise ValueError(
            "Semantic bank training provenance mismatch: "
            f"unsupported version={version!r}."
        )
    if int(version) in {3, 4, 5, 6, 7, 8} and provenance.get("active_penalty") not in {
        "level", "trend", "d2_match", "diff_amp"
    }:
        raise ValueError(
            "Semantic staged provenance requires a valid active_penalty."
        )
    if int(version) in {4, 5, 6, 7, 8} and provenance.get("active_penalty") != "level":
        raise ValueError(
            "Semantic raw-gradient accumulation provenance is currently level-only."
        )
    expected = {**common, **recipes[int(version)]}
    missing = [field for field in expected if field not in provenance]
    unexpected = [field for field in provenance if field not in expected]
    mismatches = {
        field: {"actual": provenance.get(field), "expected": value}
        for field, value in expected.items()
        if field in provenance and provenance.get(field) != value
    }
    if missing or unexpected or mismatches:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        if mismatches:
            details.append(f"mismatches={mismatches}")
        raise ValueError(
            "Semantic bank training provenance mismatch: " + "; ".join(details)
        )
    return True


_SEMANTIC_BODY_PREFIXES = (
    "W1.",
    "b1.",
    "W2.",
    "b2.",
    "semantic_level_controller.",
)


def _semantic_body_state_sha256(state: Dict[str, torch.Tensor]) -> str:
    """Hash the exact ordered semantic-body tensor schema and bytes."""
    digest = hashlib.sha256()
    keys = sorted(
        str(name) for name in state if str(name).startswith(_SEMANTIC_BODY_PREFIXES)
    )
    if not keys:
        raise ValueError("semantic body state contains no W1/b1/W2/b2 tensors")
    for name in keys:
        value = torch.as_tensor(state[name]).detach().cpu().contiguous()
        header = (
            f"name={name}\n"
            f"dtype={value.dtype}\n"
            f"shape={','.join(str(int(dim)) for dim in value.shape)}\n"
            f"numel={int(value.numel())}\n"
        ).encode("ascii")
        digest.update(header)
        digest.update(value.numpy().tobytes(order="C"))
        digest.update(b"\n")
    return digest.hexdigest()


def _semantic_penalty_body_state_sha256(state: Dict[str, torch.Tensor]) -> str:
    """Hash one complete named expert body, including its ordered ownership schema."""
    expected = {"penalty_index", "W1", "b1", "W2", "b2"}
    actual = set(state.keys())
    controller_required = {
        "penalty_index",
        "controller_state_version",
        "controller_patch_len",
    }
    if actual == expected:
        ordered_names = ("penalty_index", "W1", "b1", "W2", "b2")
    elif controller_required.issubset(actual) and all(
        name in controller_required or name.startswith("controller.") for name in actual
    ) and any(name.startswith("controller.") for name in actual):
        ordered_names = (
            "penalty_index",
            "controller_state_version",
            "controller_patch_len",
            *sorted(name for name in actual if name.startswith("controller.")),
        )
    else:
        raise ValueError(
            "semantic penalty body hash requires an exact legacy or LEVEL-controller schema: "
            f"keys={sorted(actual)}"
        )
    digest = hashlib.sha256()
    for name in ordered_names:
        value = torch.as_tensor(state[name]).detach().cpu().contiguous()
        digest.update(
            (
                f"name={name}\n"
                f"dtype={value.dtype}\n"
                f"shape={','.join(str(int(dim)) for dim in value.shape)}\n"
                f"numel={int(value.numel())}\n"
            ).encode("ascii")
        )
        digest.update(value.numpy().tobytes(order="C"))
        digest.update(b"\n")
    return digest.hexdigest()


def _semantic_penalty_body_state_from_state_dict(
    state: Dict[str, torch.Tensor],
    *,
    penalty_index: int,
    num_penalties: int,
    encoder_param_clusters: int,
    output_param_clusters: int,
    body_type: str = "legacy_shared_v1",
    controller_patch_len: int = 12,
    controller_state_version: int = 1,
) -> Dict[str, torch.Tensor]:
    p = int(penalty_index)
    P = int(num_penalties)
    if p < 0 or p >= P:
        raise IndexError(f"penalty_index must be in [0,{P}), got {p}")
    if str(body_type) == "level_local_controller_v1":
        prefix = "semantic_level_controller."
        controller_keys = sorted(name for name in state if str(name).startswith(prefix))
        if not controller_keys:
            raise ValueError("semantic checkpoint is missing LEVEL controller tensors.")
        body = {
            "penalty_index": torch.tensor(p, dtype=torch.long),
            "controller_state_version": torch.tensor(
                int(controller_state_version), dtype=torch.long
            ),
            "controller_patch_len": torch.tensor(int(controller_patch_len), dtype=torch.long),
        }
        body.update(
            {
                f"controller.{name[len(prefix):]}": torch.as_tensor(state[name]).detach().cpu().clone()
                for name in controller_keys
            }
        )
        return body

    def stacked(family: str, count: int) -> torch.Tensor:
        keys = [f"{family}.{owner_k * P + p}" for owner_k in range(int(count))]
        missing = [name for name in keys if name not in state]
        if missing:
            raise ValueError(
                f"semantic checkpoint is missing {family} keys for penalty {p}: {missing}"
            )
        return torch.stack(
            [torch.as_tensor(state[name]).detach().cpu().clone() for name in keys],
            dim=0,
        )

    return {
        "penalty_index": torch.tensor(p, dtype=torch.long),
        "W1": stacked("W1", encoder_param_clusters),
        "b1": stacked("b1", encoder_param_clusters),
        "W2": stacked("W2", output_param_clusters),
        "b2": stacked("b2", output_param_clusters),
    }


def _validate_semantic_release_metadata(
    release: object,
    *,
    penalty_names: List[str],
    source_state: Dict[str, torch.Tensor],
    source_contract: Dict[str, object],
) -> List[str]:
    if not isinstance(release, dict) or not bool(release.get("release_ready", False)):
        raise ValueError("Per-cluster frozen consumer requires release_ready=true metadata.")
    rows = release.get("per_penalty")
    if not isinstance(rows, list) or len(rows) != len(penalty_names):
        raise ValueError(
            "Semantic release metadata requires one ordered row per penalty."
        )
    stored_hashes = release.get("body_sha256_by_penalty")
    if not isinstance(stored_hashes, list) or len(stored_hashes) != len(penalty_names):
        raise ValueError("Semantic release metadata is missing ordered per-penalty body hashes.")
    encoder_count = int(source_contract.get("semantic_encoder_param_clusters", 0))
    output_count = int(source_contract.get("semantic_output_param_clusters", 0))
    body_type_by_penalty = source_contract.get("body_type_by_penalty", {}) or {}
    controller_contract = source_contract.get("semantic_level_controller", {}) or {}
    computed_hashes: List[str] = []
    for p, name in enumerate(penalty_names):
        row = rows[p]
        if not isinstance(row, dict) or row.get("penalty") != name:
            raise ValueError(
                f"Semantic release penalty order mismatch at {p}: expected {name!r}."
            )
        if not bool((row.get("acceptance") or {}).get("pass", False)):
            raise ValueError(f"Semantic release penalty {name} is not independently accepted.")
        body_state = _semantic_penalty_body_state_from_state_dict(
            source_state,
            penalty_index=p,
            num_penalties=len(penalty_names),
            encoder_param_clusters=encoder_count,
            output_param_clusters=output_count,
            body_type=str(body_type_by_penalty.get(name, "legacy_shared_v1")),
            controller_patch_len=int(controller_contract.get("patch_len", 12)),
            controller_state_version=int(controller_contract.get("state_version", 1)),
        )
        body_hash = _semantic_penalty_body_state_sha256(body_state)
        if row.get("body_sha256") != body_hash or stored_hashes[p] != body_hash:
            raise ValueError(
                f"Semantic release body hash mismatch for {name}: "
                f"row={row.get('body_sha256')}, ordered={stored_hashes[p]}, actual={body_hash}"
            )
        computed_hashes.append(body_hash)
    return computed_hashes


def _validate_semantic_partial_metadata(
    release: object,
    *,
    penalty_names: List[str],
    next_active_penalty: str,
    source_state: Dict[str, torch.Tensor],
    source_contract: Dict[str, object],
) -> List[Dict[str, object]]:
    """Validate an ordered accepted prefix for the next staged Stage-1 run."""
    if next_active_penalty not in penalty_names:
        raise ValueError(f"Unknown staged active penalty {next_active_penalty!r}.")
    next_index = penalty_names.index(next_active_penalty)
    expected_accepted = penalty_names[:next_index]
    if not isinstance(release, dict):
        raise ValueError("Semantic staged checkpoint is missing selection metadata.")
    if bool(release.get("release_ready", False)):
        raise ValueError("A partial Stage-1 resume must not claim release_ready=true.")
    if not bool(release.get("stage_complete", False)):
        raise ValueError("Semantic staged checkpoint requires stage_complete=true.")
    if list(release.get("accepted_penalties", [])) != expected_accepted:
        raise ValueError(
            "Semantic staged checkpoint accepted prefix mismatch: "
            f"expected={expected_accepted}, actual={release.get('accepted_penalties')}."
        )
    rows = release.get("per_penalty")
    hashes = release.get("body_sha256_by_penalty")
    if not isinstance(rows, list) or len(rows) != len(penalty_names):
        raise ValueError("Semantic staged checkpoint requires one ordered row per penalty.")
    if not isinstance(hashes, list) or len(hashes) != len(penalty_names):
        raise ValueError("Semantic staged checkpoint requires ordered body hashes.")
    encoder_count = int(source_contract.get("semantic_encoder_param_clusters", 0))
    output_count = int(source_contract.get("semantic_output_param_clusters", 0))
    body_type_by_penalty = source_contract.get("body_type_by_penalty", {}) or {}
    controller_contract = source_contract.get("semantic_level_controller", {}) or {}
    accepted_rows: List[Dict[str, object]] = []
    for p, name in enumerate(penalty_names):
        row = rows[p]
        if not isinstance(row, dict) or row.get("penalty") != name:
            raise ValueError(f"Semantic staged penalty order mismatch at index {p}.")
        should_be_accepted = p < next_index
        if bool(row.get("accepted", False)) != should_be_accepted:
            raise ValueError(
                f"Semantic staged accepted flag mismatch for {name}."
            )
        if not should_be_accepted:
            if hashes[p] is not None or row.get("body_sha256") is not None:
                raise ValueError(
                    f"Unaccepted staged penalty {name} must not carry an accepted-body hash."
                )
            continue
        if not bool((row.get("acceptance") or {}).get("pass", False)):
            raise ValueError(f"Accepted staged penalty {name} has no PASS evidence.")
        body_provenance = row.get("training_provenance")
        _validate_semantic_bank_training_provenance(body_provenance)
        if not isinstance(body_provenance, dict) or body_provenance.get(
            "active_penalty"
        ) != name:
            raise ValueError(
                f"Accepted staged penalty {name} has mismatched body provenance."
            )
        body_state = _semantic_penalty_body_state_from_state_dict(
            source_state,
            penalty_index=p,
            num_penalties=len(penalty_names),
            encoder_param_clusters=encoder_count,
            output_param_clusters=output_count,
            body_type=str(body_type_by_penalty.get(name, "legacy_shared_v1")),
            controller_patch_len=int(controller_contract.get("patch_len", 12)),
            controller_state_version=int(controller_contract.get("state_version", 1)),
        )
        body_hash = _semantic_penalty_body_state_sha256(body_state)
        if hashes[p] != body_hash or row.get("body_sha256") != body_hash:
            raise ValueError(f"Semantic staged body hash mismatch for {name}.")
        accepted_rows.append(
            {
                "penalty": name,
                "epoch": row.get("epoch"),
                "matching_penalty_relative_gain": float(
                    row.get("matching_penalty_relative_gain")
                ),
                "state": body_state,
                "body_sha256": body_hash,
                "acceptance": dict(row["acceptance"]),
                "training_provenance": dict(body_provenance),
                "optimizer_state_identity": str(
                    row.get("optimizer_state_identity", "")
                ),
            }
        )
    return accepted_rows


def _configure_semantic_bank_body_only(
    pred_residual: ClusterwisePredResidualMoE,
    *,
    active_penalty_index: Optional[int] = None,
) -> Dict[str, List[str]]:
    body_prefixes = _SEMANTIC_BODY_PREFIXES
    active_ids = None
    if active_penalty_index is not None:
        active_ids = {
            id(param)
            for param in pred_residual.get_penalty_body_params(
                int(active_penalty_index)
            )
        }
    trainable: List[str] = []
    frozen: List[str] = []
    for name, param in pred_residual.named_parameters():
        is_body = name.startswith(body_prefixes)
        is_active_body = bool(
            id(param) in active_ids
            if active_ids is not None
            else is_body
        )
        param.requires_grad_(is_active_body)
        if not is_active_body:
            param.grad = None
            frozen.append(name)
        else:
            trainable.append(name)
    expected_ids = {
        id(param)
        for p in (
            range(pred_residual.P)
            if active_penalty_index is None
            else [int(active_penalty_index)]
        )
        for param in pred_residual.get_penalty_body_params(p)
    }
    actual_ids = {
        id(param) for param in pred_residual.parameters() if param.requires_grad
    }
    if actual_ids != expected_ids:
        raise RuntimeError("semantic Stage-1 body-only trainability does not match ownership")
    return {"trainable": trainable, "frozen": frozen}


def _validate_semantic_frozen_consumer_paths(paths: Dict[str, bool]) -> None:
    forbidden = sorted(name for name, enabled in paths.items() if bool(enabled))
    if forbidden:
        raise ValueError(
            "Per-cluster frozen consumer permits only the exact bank plus a new patch_router; "
            "forbidden paths: " + ", ".join(forbidden)
        )


def _validate_semantic_frozen_consumer_lifecycle_flags(
    *,
    freeze_adapter_bank: bool,
    freeze_backbone: bool,
    patch_router_enable: bool,
    finetune_enable: bool,
    load_pred_residual: bool,
    load_gate: bool,
    require_training_provenance: bool,
) -> None:
    required = {
        "freeze_adapter_bank": freeze_adapter_bank,
        "freeze_backbone": freeze_backbone,
        "patch_router_enable": patch_router_enable,
        "finetune_enable": finetune_enable,
        "load_pred_residual": load_pred_residual,
        "require_training_provenance": require_training_provenance,
    }
    missing = sorted(name for name, enabled in required.items() if not bool(enabled))
    if missing or bool(load_gate):
        raise ValueError(
            "Per-cluster frozen consumer lifecycle mismatch: "
            f"missing_required={missing}, load_gate={bool(load_gate)}"
        )


def _freeze_semantic_frozen_consumer(
    pred_residual: ClusterwisePredResidualMoE,
) -> Dict[str, object]:
    frozen_count = _freeze_module_params_except_prefixes(
        pred_residual,
        ("patch_router.",),
    )
    body_ids = {
        id(param)
        for p in range(pred_residual.P)
        for param in pred_residual.get_penalty_body_params(p)
    }
    named = list(pred_residual.named_parameters())
    body_trainable = [name for name, param in named if id(param) in body_ids and param.requires_grad]
    nonrouter_trainable = [
        name for name, param in named
        if not name.startswith("patch_router.") and param.requires_grad
    ]
    router_frozen = [
        name for name, param in named
        if name.startswith("patch_router.") and not param.requires_grad
    ]
    if body_trainable or nonrouter_trainable or router_frozen:
        raise RuntimeError(
            "Per-cluster frozen consumer trainability mismatch: "
            f"body_trainable={body_trainable}, nonrouter_trainable={nonrouter_trainable}, "
            f"router_frozen={router_frozen}"
        )
    return {
        "frozen_params": int(frozen_count),
        "frozen_names": [
            name for name, param in named if not param.requires_grad
        ],
        "trainable_router_names": [
            name for name, param in named if param.requires_grad
        ],
    }


def _assert_semantic_patch_router_only_trainable(
    *,
    model: nn.Module,
    gate: nn.Module,
    pred_residual: ClusterwisePredResidualMoE,
    dynamic_lambda: Optional[nn.Module] = None,
    learnable_lambda: Optional[nn.Module] = None,
    learnable_output_anchor: Optional[nn.Module] = None,
) -> List[str]:
    unexpected: List[str] = []
    for prefix, module in (
        ("model", model),
        ("gate", gate),
        ("dynamic_lambda", dynamic_lambda),
        ("learnable_lambda", learnable_lambda),
        ("learnable_output_anchor", learnable_output_anchor),
    ):
        if module is None:
            continue
        unexpected.extend(
            f"{prefix}.{name}" for name, param in module.named_parameters() if param.requires_grad
        )
    router_names: List[str] = []
    for name, param in pred_residual.named_parameters():
        if not param.requires_grad:
            continue
        if not name.startswith("patch_router."):
            unexpected.append(f"pred_residual.{name}")
        else:
            router_names.append(f"pred_residual.{name}")
    if unexpected or not router_names:
        raise RuntimeError(
            "Per-cluster frozen consumer must have only target-new patch_router trainable: "
            f"unexpected={unexpected}, router_names={router_names}"
        )
    return router_names


def _canonical_cluster_map_sha256(K: int, cluster_id_c: Iterable[int]) -> str:
    cluster_count = int(K)
    values = [int(value) for value in cluster_id_c]
    if cluster_count <= 0 or not values:
        raise ValueError("canonical cluster map requires positive K and a nonempty channel map")
    if any(value < 0 or value >= cluster_count for value in values):
        raise ValueError(
            f"canonical cluster map values must be in [0,{cluster_count}), got {values}"
        )
    payload = (
        f"K={cluster_count}\ncluster_id_c="
        + ",".join(str(value) for value in values)
        + "\n"
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _normalized_pred_residual_contract(
    source_contract: Dict[str, object],
    target_contract: Dict[str, object],
) -> Dict[str, object]:
    """Normalize only a legacy semantic contract to the shared representation."""
    normalized = dict(source_contract)
    if "semantic_output_head_scope" in normalized:
        return normalized
    if target_contract.get("semantic_output_head_scope", "shared") != "shared":
        raise ValueError(
            "Legacy pred_residual contracts can normalize only to shared output heads."
        )
    normalized.update({
        "version": target_contract.get("version"),
        "semantic_output_head_scope": "shared",
        "semantic_encoder_param_clusters": target_contract.get(
            "semantic_encoder_param_clusters", 1
        ),
        "semantic_output_param_clusters": target_contract.get(
            "semantic_output_param_clusters", 1
        ),
        "semantic_parameter_ownership": "legacy_shared_v1",
    })
    return normalized


def _validate_exact_semantic_body_state(
    source_state: Dict[str, torch.Tensor],
    target_state: Dict[str, torch.Tensor],
) -> None:
    source_keys = {
        str(name) for name in source_state if str(name).startswith(_SEMANTIC_BODY_PREFIXES)
    }
    target_keys = {
        str(name) for name in target_state if str(name).startswith(_SEMANTIC_BODY_PREFIXES)
    }
    if source_keys != target_keys:
        raise ValueError(
            "Semantic body key mismatch: "
            f"missing={sorted(target_keys - source_keys)}, "
            f"unexpected={sorted(source_keys - target_keys)}"
        )
    for name in sorted(target_keys):
        source = torch.as_tensor(source_state[name])
        target = torch.as_tensor(target_state[name])
        if tuple(source.shape) != tuple(target.shape) or source.dtype != target.dtype:
            raise ValueError(
                f"Semantic body tensor mismatch for {name}: "
                f"source={tuple(source.shape)}/{source.dtype}, "
                f"target={tuple(target.shape)}/{target.dtype}"
            )


def _new_semantic_bank_best_candidates(
    penalty_names: List[str],
) -> List[Dict[str, object]]:
    return [
        {
            "penalty": str(name),
            "epoch": None,
            "matching_penalty_relative_gain": float("-inf"),
            "state": None,
            "body_sha256": None,
            "acceptance": None,
        }
        for name in penalty_names
    ]


def _update_semantic_bank_best_candidate(
    best_by_penalty: List[Dict[str, object]],
    *,
    penalty_index: int,
    penalty_name: str,
    epoch: int,
    acceptance: Dict[str, object],
    body_state: Dict[str, torch.Tensor],
) -> bool:
    """Retain one body only when that same expert independently passes."""
    p = int(penalty_index)
    if p < 0 or p >= len(best_by_penalty):
        raise IndexError(f"penalty_index {p} is outside candidate bank")
    current = best_by_penalty[p]
    if str(current.get("penalty")) != str(penalty_name):
        raise ValueError(
            "semantic bank candidate penalty mismatch: "
            f"slot={current.get('penalty')!r}, submitted={penalty_name!r}"
        )
    passed = bool(acceptance.get("pass", False))
    gain = float(acceptance.get("matching_penalty_relative_gain", float("-inf")))
    selected = bool(
        passed
        and math.isfinite(gain)
        and gain > float(current.get("matching_penalty_relative_gain", float("-inf")))
    )
    if selected:
        best_by_penalty[p] = {
            "penalty": str(penalty_name),
            "epoch": int(epoch),
            "matching_penalty_relative_gain": gain,
            "state": {
                key: value.detach().cpu().clone()
                for key, value in body_state.items()
            },
            "body_sha256": _semantic_penalty_body_state_sha256(body_state),
            "acceptance": dict(acceptance),
        }
    return selected


def _semantic_bank_release_ready(
    best_by_penalty: List[Dict[str, object]],
    penalty_names: List[str],
) -> bool:
    if len(best_by_penalty) != len(penalty_names) or not penalty_names:
        return False
    for p, (item, expected_name) in enumerate(zip(best_by_penalty, penalty_names)):
        if item.get("penalty") != expected_name:
            return False
        state = item.get("state")
        if not isinstance(state, dict):
            return False
        if not bool((item.get("acceptance") or {}).get("pass", False)):
            return False
        if item.get("body_sha256") != _semantic_penalty_body_state_sha256(state):
            return False
        if int(torch.as_tensor(state.get("penalty_index", -1)).item()) != p:
            return False
    return True


def _semantic_bank_checkpoint_save_allowed(
    *,
    save_requested: bool,
    semantic_bank_stage1: bool,
    release_ready: bool,
) -> bool:
    return bool(save_requested) and (
        not bool(semantic_bank_stage1) or bool(release_ready)
    )


def _load_finetune_pred_residual_state(
    *,
    pred_residual: Optional[ClusterwisePredResidualMoE],
    checkpoint: Dict[str, object],
    source_penalty_names: List[str],
    target_penalty_names: List[str],
    strict: bool = True,
    source_contract: Optional[Dict[str, object]] = None,
    target_contract: Optional[Dict[str, object]] = None,
    source_training_provenance: Optional[Dict[str, object]] = None,
    require_training_provenance: bool = False,
) -> bool:
    if pred_residual is None or "pred_residual_state" not in checkpoint:
        return False
    if list(source_penalty_names) != list(target_penalty_names):
        raise ValueError(
            "Fine-tune pred_residual loading requires identical penalty_names: "
            f"source={list(source_penalty_names)}, target={list(target_penalty_names)}"
        )
    if target_contract is not None:
        if source_contract is None:
            raise ValueError(
                "Fine-tune pred_residual loading requires checkpoint pred_residual_contract metadata."
            )
        target_has_output_scope = "semantic_output_head_scope" in target_contract
        normalized_source_contract = (
            _normalized_pred_residual_contract(source_contract, target_contract)
            if target_has_output_scope
            else dict(source_contract)
        )
        contract_fields = (
            "version",
            "penalty_names",
            "projection_mode",
            "projection_patch_len",
            "base_type",
            "fixed_alpha",
            "scale_by_name",
            "diff_amp_max",
        )
        if target_has_output_scope:
            contract_fields = (
                *contract_fields,
                "semantic_output_head_scope",
                "semantic_encoder_param_clusters",
                "semantic_output_param_clusters",
                "semantic_parameter_ownership",
            )
        if "semantic_level_controller" in target_contract:
            contract_fields = (
                *contract_fields,
                "semantic_level_controller",
                "body_type_by_penalty",
            )
        if target_contract.get("semantic_output_head_scope") == "per_cluster":
            contract_fields = (*contract_fields, "cluster_count", "cluster_id_c", "cluster_map_sha256")
            for label, contract in (
                ("source", normalized_source_contract),
                ("target", target_contract),
            ):
                canonical_hash = _canonical_cluster_map_sha256(
                    int(contract.get("cluster_count", 0)),
                    contract.get("cluster_id_c", []),
                )
                if contract.get("cluster_map_sha256") != canonical_hash:
                    raise ValueError(
                        f"{label} pred_residual contract has a noncanonical cluster_map_sha256."
                    )
        mismatches = {
            field: {
                "source": normalized_source_contract.get(field),
                "target": target_contract.get(field),
            }
            for field in contract_fields
            if normalized_source_contract.get(field) != target_contract.get(field)
        }
        if mismatches:
            raise ValueError(
                "Fine-tune pred_residual contract mismatch: "
                + ", ".join(
                    f"{field}=source:{values['source']!r}/target:{values['target']!r}"
                    for field, values in mismatches.items()
                )
            )
    if bool(require_training_provenance) or source_training_provenance is not None:
        _validate_semantic_bank_training_provenance(source_training_provenance)
    source_state = checkpoint["pred_residual_state"]
    target_state = pred_residual.state_dict()
    per_cluster_semantic_body = bool(
        target_contract is not None
        and target_contract.get("semantic_output_head_scope") == "per_cluster"
    )
    if per_cluster_semantic_body:
        source_router_keys = sorted(
            name for name in source_state if str(name).startswith("patch_router.")
        )
        if source_router_keys:
            raise ValueError(
                "Per-cluster frozen consumer requires a target-new patch_router; "
                f"source checkpoint contains router keys: {source_router_keys}"
            )
        _validate_exact_semantic_body_state(source_state, target_state)
        expected_body_hash = source_contract.get("semantic_body_sha256") if source_contract else None
        if not isinstance(expected_body_hash, str) or len(expected_body_hash) != 64:
            raise ValueError(
                "Per-cluster semantic bank checkpoint is missing semantic_body_sha256."
            )
        actual_body_hash = _semantic_body_state_sha256(source_state)
        if actual_body_hash != expected_body_hash:
            raise ValueError(
                "Per-cluster semantic body hash mismatch: "
                f"metadata={expected_body_hash}, actual={actual_body_hash}"
            )
    if bool(strict):
        if per_cluster_semantic_body:
            source_keys = set(source_state.keys())
            target_keys = set(target_state.keys())
            if source_keys != target_keys:
                raise ValueError(
                    "Strict per-cluster consumer requires an exact full state key set: "
                    f"missing={sorted(target_keys - source_keys)}, "
                    f"unexpected={sorted(source_keys - target_keys)}"
                )
            for name in sorted(target_keys):
                source_value = torch.as_tensor(source_state[name])
                target_value = torch.as_tensor(target_state[name])
                if (
                    tuple(source_value.shape) != tuple(target_value.shape)
                    or source_value.dtype != target_value.dtype
                ):
                    raise ValueError(
                        f"Strict per-cluster state mismatch for {name}: "
                        f"source={tuple(source_value.shape)}/{source_value.dtype}, "
                        f"target={tuple(target_value.shape)}/{target_value.dtype}"
                    )
        pred_residual.load_state_dict(source_state, strict=True)
        return True
    compatible_state = {}
    incompatible_keys = []
    for name, value in source_state.items():
        target_value = target_state.get(name)
        if target_value is None or tuple(target_value.shape) != tuple(value.shape):
            incompatible_keys.append(str(name))
            continue
        compatible_state[name] = value
    if per_cluster_semantic_body:
        unexpected_source = sorted(set(source_state.keys()) - set(target_state.keys()))
        missing_target = sorted(set(target_state.keys()) - set(source_state.keys()))
        forbidden_unexpected = [
            name for name in unexpected_source if not str(name).startswith("patch_router.")
        ]
        forbidden_missing = [
            name for name in missing_target if not str(name).startswith("patch_router.")
        ]
        if forbidden_unexpected or forbidden_missing or incompatible_keys:
            raise ValueError(
                "Non-strict per-cluster consumer permits only explicit patch_router nonbody keys: "
                f"missing={forbidden_missing}, unexpected={forbidden_unexpected}, "
                f"incompatible={incompatible_keys}"
            )
    if not compatible_state:
        return False
    pred_residual.load_state_dict(compatible_state, strict=False)
    if incompatible_keys:
        print(
            "Non-strict pred_residual warm start skipped incompatible keys: "
            + ", ".join(incompatible_keys)
        )
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
    return list(gate.get_cluster_params(k))


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


def _semantic_bank_finalize_raw_gradient_accumulation(
    optimizer: torch.optim.Optimizer,
    params: Iterable[nn.Parameter],
    *,
    accumulation_count: int,
    grad_clip: float,
) -> Dict[str, float | int]:
    """Mean accumulated raw grads, clip once, apply WD/SGD once, then clear grads."""
    params_list = list(params)
    count = int(accumulation_count)
    if count <= 0:
        raise ValueError("semantic raw-gradient accumulation_count must be positive.")
    optimizer_params = [
        param for group in optimizer.param_groups for param in group["params"]
    ]
    if [id(param) for param in optimizer_params] != [id(param) for param in params_list]:
        raise ValueError(
            "semantic raw-gradient optimizer ownership must exactly match active params."
        )
    present = [param for param in params_list if param.grad is not None]
    if not present:
        raise RuntimeError("semantic raw-gradient accumulation produced no gradients.")
    for param in present:
        param.grad.div_(float(count))
    raw_norm = _parameter_grad_l2_norm(params_list)
    if float(grad_clip) > 0.0:
        torch.nn.utils.clip_grad_norm_(params_list, float(grad_clip))
    clipped_norm = _parameter_grad_l2_norm(params_list)
    snapshots = [param.detach().clone() for param in params_list]
    optimizer.step()
    update_sq = 0.0
    for before_param, after_param in zip(snapshots, params_list):
        update_sq += float(
            (after_param.detach() - before_param).double().square().sum().item()
        )
    optimizer.zero_grad(set_to_none=True)
    return {
        "accumulation_count": count,
        "raw_gradient_l2": float(raw_norm),
        "clipped_gradient_l2": float(clipped_norm),
        "parameter_update_l2": math.sqrt(update_sq),
    }


def _semantic_bank_should_flush_raw_gradient_accumulation(
    *,
    pending_count: int,
    target_microbatches: int,
    completed_batches: int,
    total_batches: int,
) -> bool:
    """Return the production flush decision for a full group or epoch tail."""
    pending = int(pending_count)
    target = int(target_microbatches)
    completed = int(completed_batches)
    total = int(total_batches)
    if target <= 0 or total <= 0:
        raise ValueError("semantic accumulation target and total_batches must be positive.")
    if pending <= 0 or pending > target:
        raise ValueError("semantic accumulation pending_count must be in [1,target].")
    if completed <= 0 or completed > total:
        raise ValueError("semantic accumulation completed_batches must be in [1,total].")
    return bool(pending == target or completed == total)


def _semantic_bank_temporal_block_indices(
    relative_window_index: torch.Tensor,
    *,
    split_window_count: int,
    block_count: int,
) -> torch.Tensor:
    """Assign split-relative windows to fixed contiguous semantic-audit blocks."""
    windows = int(split_window_count)
    blocks = int(block_count)
    if windows <= 0:
        raise ValueError("semantic audit split_window_count must be positive.")
    if blocks <= 0:
        raise ValueError("semantic audit block_count must be positive.")
    if relative_window_index.dtype != torch.long:
        raise ValueError("semantic audit relative_window_index must use torch.long.")
    if bool((relative_window_index < 0).any().item()):
        raise ValueError("semantic audit relative_window_index must be nonnegative.")
    return torch.div(
        relative_window_index * blocks,
        windows,
        rounding_mode="floor",
    ).clamp_max(blocks - 1)


def _make_semantic_level_disjoint_optimizers(
    amplitude_params: Iterable[nn.Parameter],
    need_gate_params: Iterable[nn.Parameter],
    *,
    amplitude_optimizer_name: str,
    amplitude_lr: float,
    amplitude_weight_decay: float,
    need_gate_lr: float,
    need_gate_weight_decay: float,
) -> Dict[str, torch.optim.Optimizer]:
    """Build independently owned LEVEL amplitude and need-gate optimizers.

    The need gate deliberately retains the legacy zero-momentum SGD rule.  The
    amplitude rule is explicit so an optimizer repair cannot silently affect the
    gate or any other named expert.
    """

    amplitude = list(amplitude_params)
    need_gate = list(need_gate_params)
    if not amplitude or not need_gate:
        raise ValueError("LEVEL disjoint optimizer groups must be non-empty.")
    amplitude_ids = [id(parameter) for parameter in amplitude]
    need_gate_ids = [id(parameter) for parameter in need_gate]
    if len(amplitude_ids) != len(set(amplitude_ids)) or len(need_gate_ids) != len(
        set(need_gate_ids)
    ):
        raise ValueError("LEVEL disjoint optimizer groups contain duplicate parameters.")
    if set(amplitude_ids).intersection(need_gate_ids):
        raise ValueError("LEVEL amplitude and need-gate optimizer groups overlap.")

    name = str(amplitude_optimizer_name).strip().lower()
    lr = float(amplitude_lr)
    weight_decay = float(amplitude_weight_decay)
    gate_lr = float(need_gate_lr)
    gate_weight_decay = float(need_gate_weight_decay)
    if lr <= 0.0 or gate_lr <= 0.0:
        raise ValueError("LEVEL optimizer learning rates must be positive.")
    if weight_decay < 0.0 or gate_weight_decay < 0.0:
        raise ValueError("LEVEL optimizer weight decay must be non-negative.")

    if name == "adam":
        amplitude_optimizer: torch.optim.Optimizer = torch.optim.Adam(
            [{"params": amplitude, "weight_decay": weight_decay}],
            lr=lr,
            betas=(0.9, 0.999),
            eps=1.0e-8,
            amsgrad=False,
        )
    elif name == "sgd":
        amplitude_optimizer = torch.optim.SGD(
            [{"params": amplitude, "weight_decay": weight_decay}],
            lr=lr,
            momentum=0.0,
            dampening=0.0,
            nesterov=False,
        )
    else:
        raise ValueError(
            "LEVEL amplitude optimizer must be one of {'sgd','adam'}, "
            f"got {amplitude_optimizer_name!r}."
        )

    need_gate_optimizer = torch.optim.SGD(
        [{"params": need_gate, "weight_decay": gate_weight_decay}],
        lr=gate_lr,
        momentum=0.0,
        dampening=0.0,
        nesterov=False,
    )
    for optimizer in (amplitude_optimizer, need_gate_optimizer):
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", float(group["lr"]))
    return {
        "amplitude": amplitude_optimizer,
        "need_gate": need_gate_optimizer,
    }


def _semantic_level_optimizer_identity_fields(
    *,
    amplitude_optimizer_name: str,
    amplitude_lr: float,
    amplitude_weight_decay: float,
    need_gate_lr: float,
    need_gate_weight_decay: float,
) -> List[str]:
    """Return versioned identity fields without changing the legacy SGD hash.

    Version-6 checkpoints predate an amplitude-specific override, so inherited
    SGD contributes no new fields. Version 7 binds every behavior-affecting
    constant of the reviewed Adam-amplitude/SGD-gate rule.
    """

    name = str(amplitude_optimizer_name).strip().lower()
    if name == "sgd":
        if float(amplitude_lr) != float(need_gate_lr):
            raise ValueError(
                "Legacy LEVEL SGD identity requires amplitude lr to equal "
                "the inherited need-gate/train lr."
            )
        if float(amplitude_weight_decay) != float(need_gate_weight_decay):
            raise ValueError(
                "Legacy LEVEL SGD identity requires amplitude weight decay to "
                "equal the inherited need-gate/residual weight decay."
            )
        return []
    if name != "adam":
        raise ValueError(
            "LEVEL optimizer identity supports only inherited sgd or reviewed adam."
        )
    return [
        "amplitude_optimizer=adam",
        "amplitude_lr=" + repr(float(amplitude_lr)),
        "amplitude_weight_decay=" + repr(float(amplitude_weight_decay)),
        "amplitude_betas=(0.9,0.999)",
        "amplitude_eps=1e-08",
        "amplitude_amsgrad=false",
        "need_gate_optimizer=sgd",
        "need_gate_lr=" + repr(float(need_gate_lr)),
        "need_gate_weight_decay=" + repr(float(need_gate_weight_decay)),
        "need_gate_momentum=0.0",
        "need_gate_dampening=0.0",
        "need_gate_nesterov=false",
    ]


def _semantic_bank_finalize_gradient_step(
    optimizer: torch.optim.Optimizer,
    params: Iterable[nn.Parameter],
    *,
    grad_clip: float,
    raw_gradient_accumulation: bool,
    accumulation_count: int,
) -> Dict[str, float | int]:
    """Production dispatch for legacy per-batch or raw-mean semantic SGD."""
    params_list = list(params)
    if raw_gradient_accumulation:
        return _semantic_bank_finalize_raw_gradient_accumulation(
            optimizer,
            params_list,
            accumulation_count=int(accumulation_count),
            grad_clip=float(grad_clip),
        )
    if int(accumulation_count) != 1:
        raise ValueError("legacy semantic gradient step requires accumulation_count=1.")
    snapshots = [param.detach().clone() for param in params_list]
    raw_norm = _parameter_grad_l2_norm(params_list)
    if float(grad_clip) > 0.0:
        torch.nn.utils.clip_grad_norm_(params_list, float(grad_clip))
    clipped_norm = _parameter_grad_l2_norm(params_list)
    optimizer.step()
    update_sq = 0.0
    for before_param, after_param in zip(snapshots, params_list):
        update_sq += float(
            (after_param.detach() - before_param).double().square().sum().item()
        )
    return {
        "accumulation_count": 1,
        "raw_gradient_l2": float(raw_norm),
        "clipped_gradient_l2": float(clipped_norm),
        "parameter_update_l2": math.sqrt(update_sq),
    }


def _semantic_bank_finalize_disjoint_gradient_step(
    optimizers_by_group: Dict[str, torch.optim.Optimizer],
    params_by_group: Dict[str, Iterable[nn.Parameter]],
    *,
    grad_clip: float,
    raw_gradient_accumulation: bool,
    accumulation_count: int,
) -> Dict[str, object]:
    """Finalize disjoint LEVEL adapter/gate groups without cross-group clipping."""

    if set(optimizers_by_group) != set(params_by_group) or len(params_by_group) < 2:
        raise ValueError(
            "semantic disjoint gradient step requires matching optimizer/parameter groups."
        )
    materialized = {
        name: list(params_by_group[name]) for name in sorted(params_by_group)
    }
    seen: Dict[int, str] = {}
    for name, params in materialized.items():
        if not params:
            raise ValueError(f"semantic disjoint parameter group {name!r} is empty.")
        for parameter in params:
            previous = seen.get(id(parameter))
            if previous is not None:
                raise ValueError(
                    "semantic disjoint parameter groups overlap: "
                    f"{previous!r} and {name!r}."
                )
            seen[id(parameter)] = name
        optimizer_params = [
            parameter
            for optimizer_group in optimizers_by_group[name].param_groups
            for parameter in optimizer_group["params"]
        ]
        optimizer_ids = [id(parameter) for parameter in optimizer_params]
        parameter_ids = [id(parameter) for parameter in params]
        if len(optimizer_ids) != len(set(optimizer_ids)):
            raise ValueError(
                f"semantic disjoint optimizer group {name!r} contains duplicate parameters."
            )
        if set(optimizer_ids) != set(parameter_ids):
            raise ValueError(
                f"semantic disjoint optimizer group {name!r} does not exactly match "
                "its declared parameter group."
            )

    group_metrics: Dict[str, Dict[str, float | int]] = {}
    for name, params in materialized.items():
        group_metrics[name] = _semantic_bank_finalize_gradient_step(
            optimizers_by_group[name],
            params,
            grad_clip=float(grad_clip),
            raw_gradient_accumulation=bool(raw_gradient_accumulation),
            accumulation_count=int(accumulation_count),
        )
    return {
        "accumulation_count": int(accumulation_count),
        "raw_gradient_l2": math.sqrt(
            sum(float(row["raw_gradient_l2"]) ** 2 for row in group_metrics.values())
        ),
        "clipped_gradient_l2": math.sqrt(
            sum(float(row["clipped_gradient_l2"]) ** 2 for row in group_metrics.values())
        ),
        "parameter_update_l2": math.sqrt(
            sum(float(row["parameter_update_l2"]) ** 2 for row in group_metrics.values())
        ),
        "groups": group_metrics,
    }


def extract_gate_features(x_bcl: torch.Tensor) -> torch.Tensor:
    """
    Extract lightweight but shape-aware per-channel descriptors for gate routing.
    Returns x_bcf: [B, C, F]
    """
    B, C, L = x_bcl.shape
    mean = x_bcl.mean(dim=-1)
    std = x_bcl.std(dim=-1) if L > 1 else torch.zeros_like(mean)
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
    effective_route_bcph = pred_out.get("effective_route_bcph")
    intervention_bcp = pred_out.get("intervention_bcp")
    alpha_cp = pred_out.get("alpha_cp")
    if residuals is None or route_bcp is None or effective_route_bcp is None or alpha_cp is None or residuals.numel() == 0:
        return out

    spec_bk = zero_bk
    if specialization_weight != 0.0 and len(penalty_names) > 0:
        y_final_sg = y_final.detach()
        for p, name in enumerate(penalty_names):
            if effective_route_bcph is not None:
                visible_scale = (
                    effective_route_bcph[:, :, p, :] * alpha_cp[:, p].view(1, -1, 1)
                ).detach()
                visible = visible_scale * residuals[:, :, p, :]
                route_activity_bc = effective_route_bcph[:, :, p, :].mean(dim=-1).detach()
            else:
                visible_scale = (effective_route_bcp[..., p] * alpha_cp[:, p].unsqueeze(0)).detach()
                visible = visible_scale.unsqueeze(-1) * residuals[:, :, p, :]
                route_activity_bc = effective_route_bcp[..., p].detach()
            y_view = y_final_sg + visible - visible.detach()
            pen_bc = penalty_fns[name](y_view, y)
            if penalty_scale is not None and penalty_scale.numel() > p:
                scale_p = penalty_scale[p].to(device=pen_bc.device, dtype=pen_bc.dtype).clamp_min(1.0e-6)
                pen_bc = pen_bc / scale_p
            # Do not let unselected branches affect the scalar objective.
            pen_bc = pen_bc * route_activity_bc
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
    router_mode: Optional[str] = None,
) -> torch.Tensor:
    context_shape = (x_bcl.shape[0], K, len(penalty_names))
    if len(penalty_names) == 0 or str(router_mode or "").lower() == "learned":
        return torch.zeros(context_shape, device=x_bcl.device, dtype=x_bcl.dtype)
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


__all__ = [
    '_get_rss_mb',
    '_dir_size_mb',
    '_shape_tuple',
    '_partial_load_matching_state_dict',
    'print_clusters',
    '_make_torch_generator',
    '_resolve_overfit_diagnostic_range',
    '_freeze_module_params',
    '_set_module_train_mode',
    '_freeze_module_params_except_prefixes',
    '_validation_holdout_split_counts',
    '_normalize_confidence_gate_source_split',
    '_normalize_pred_residual_selection_policy',
    '_normalize_pred_residual_candidate_selection_metric',
    '_contiguous_segment_ranges',
    '_top_positive_improvement_mask',
    '_normalize_learnable_output_anchor_cfg',
    '_clone_module_state_dict',
    '_copy_learnable_output_anchor_active_masks',
    '_normalize_learnable_output_anchor_adoption_scope',
    '_summarize_learnable_output_anchor_refiner',
    '_select_learnable_output_anchor_channel_mask',
    '_select_learnable_output_anchor_channel_horizon_mask',
    '_finalize_channel_horizon_metric_collector',
    '_loss_normalization_enabled',
    '_loss_normalization_term_set',
    '_normalize_loss_terms',
    '_accumulate_detached_sum_',
    '_lr_warmup_scale',
    '_set_optimizer_lr_scale',
    '_optimizer_slot_active',
    '_training_cluster_weight',
    '_should_update_swa',
    '_make_cluster_optimizer_param_groups',
    '_validate_semantic_bank_training_provenance',
    '_semantic_body_state_sha256',
    '_semantic_penalty_body_state_sha256',
    '_semantic_penalty_body_state_from_state_dict',
    '_validate_semantic_release_metadata',
    '_configure_semantic_bank_body_only',
    '_validate_semantic_frozen_consumer_paths',
    '_validate_semantic_frozen_consumer_lifecycle_flags',
    '_freeze_semantic_frozen_consumer',
    '_assert_semantic_patch_router_only_trainable',
    '_canonical_cluster_map_sha256',
    '_new_semantic_bank_best_candidates',
    '_update_semantic_bank_best_candidate',
    '_semantic_bank_release_ready',
    '_validate_semantic_partial_metadata',
    '_semantic_bank_checkpoint_save_allowed',
    '_load_finetune_pred_residual_state',
    '_make_semantic_level_disjoint_optimizers',
    '_semantic_level_optimizer_identity_fields',
    '_cluster_penalty_mask_to_channel_mask',
    '_gate_cluster_params',
    '_mask_gate_grads_after_epoch',
    'compute_channel_shape_features',
    'GATE_FEATURE_NAMES',
    'BASE_FORECAST_GATE_FEATURE_NAMES',
    '_normalize_gate_feature_mode',
    '_gate_feature_names_for_mode',
    'get_gate_feature_dim',
    'reduce_cluster_metric',
    '_weighted_cluster_sum_mean',
    '_stage2_loss_epoch_summary',
    '_stage2_route_epoch_summary',
    '_safe_ratio',
    '_route_accuracy_summary_from_labels',
    '_route_distribution_entropy_from_rates',
    '_route_audit_summary_from_tensors',
    '_stage2_route_audit_thresholds',
    '_cluster_route_oracle_labels_and_gain_from_candidates',
    '_cluster_route_oracle_labels_from_candidates',
    '_route_ce_active_mask_from_gain',
    '_pred_residual_training_skip_arg',
    '_route_probs_with_skip_class',
    '_route_ce_loss_from_probs',
    '_route_binary_adoption_loss_from_probs',
    '_route_rate_alignment_loss_from_probs',
    '_route_positive_recall_loss_from_probs',
    '_route_precision_constrained_recall_loss_from_probs',
    '_route_ce_class_weight_from_labels',
    '_parameter_grad_l2_norm',
    '_semantic_bank_finalize_raw_gradient_accumulation',
    '_semantic_bank_should_flush_raw_gradient_accumulation',
    '_semantic_bank_temporal_block_indices',
    '_semantic_bank_finalize_gradient_step',
    '_semantic_bank_finalize_disjoint_gradient_step',
    'extract_gate_features',
    '_extract_base_forecast_gate_features',
    '_build_gate_routing_features',
    'build_gate_prior_from_penalty_portrait',
    'build_topk_penalty_mask',
    'build_named_penalty_mask',
    'normalize_cluster_penalty_prior_apply_stage',
    'split_cluster_penalty_prior_allowed_mask_by_stage',
    'compute_cluster_penalty_portrait',
    'compute_channel_penalty_portrait',
    '_rowwise_corr',
    '_expand_penalty_setting_for_names',
    '_select_rank_mask',
    '_compute_lambda_bkp',
    '_routed_penalty_loss',
    '_apply_skip_to_penalty_loss',
    '_pred_residual_loss_terms',
    '_mae_objective_bc_from_abs',
    '_mae_objective_weight_is_nonzero',
    '_apply_mae_objective_weight',
    '_scale_mae_objective_weight',
    '_cluster_target_stats_from_targets',
    '_gap_multiplier_from_rows',
    '_build_mae_per_cluster_diagnostics_from_targets',
    '_collect_train_targets_bch',
    '_save_mae_per_cluster_diagnostics_csv',
    '_cosine_similarity_matrix',
    '_write_matrix_csv',
    '_save_cluster_embedding_artifacts',
    '_gate_regularization',
    '_apply_router_penalty_context',
    '_history_proxy_forecast',
    '_router_penalty_context_from_history',
    '_named_param_dict',
    '_parse_positive_ints',
]
