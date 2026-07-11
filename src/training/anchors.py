"""History-derived and train-stat output anchor helpers."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..models.learnable_anchor import ClusterwiseLearnableOutputAnchor
from .core import _normalize_learnable_output_anchor_cfg, _parse_positive_ints


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
    ("weather", 96): _moe_output_anchor_default(
        stat=_stat_anchor_default(period=144, metric="mse", max_scale=0.4, steps=13),
        residual=_residual_anchor_default(period=144, metric="mse", max_scale=0.8, steps=25, horizon_segments=8),
    ),
    ("weather", 192): _moe_output_anchor_default(
        stat=_stat_anchor_default(period=144, metric="mae", max_scale=0.5, steps=13),
        residual=_residual_anchor_default(period=144, metric="mae", max_scale=1.0, steps=25, horizon_segments=8),
    ),
    ("weather", 336): _moe_output_anchor_default(
        stat=_stat_anchor_default(period=96, metric="mae", max_scale=0.2, steps=9),
        residual=_residual_anchor_default(period=96, metric="mae", max_scale=1.2, steps=49, horizon_segments=7),
    ),
    ("weather", 720): _moe_output_anchor_default(
        stat=_stat_anchor_default(period=96, metric="mae", max_scale=0.2, steps=9),
        residual=_residual_anchor_default(period=96, metric="mae", max_scale=1.2, steps=49, horizon_segments=7),
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
    learnable_output_anchor: Optional[ClusterwiseLearnableOutputAnchor] = None,
    cluster_id_c: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    moe_cfg = moe_cfg or {}
    out = pred_bch
    history_cfg = moe_cfg.get("history_anchor_expert", {}) or {}
    stat_cfg = moe_cfg.get("train_stat_anchor_expert", {}) or {}
    residual_cfg = moe_cfg.get("train_residual_anchor_expert", {}) or {}
    learnable_cfg = _normalize_learnable_output_anchor_cfg(moe_cfg.get("learnable_output_anchor", {}))
    stat_delta_bch = None
    residual_delta_bch = None
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
        before = out
        out = apply_train_stat_anchor_expert(
            out,
            base_pred_bch=base_pred_bch,
            x_bcl=x_bcl,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len),
            stat_anchor_pc=train_stat_anchor_pc,
            cfg=stat_cfg,
        )
        stat_delta_bch = out - before
    if bool(residual_cfg.get("enable", False)) and train_residual_anchor_phc is not None:
        before = out
        out = apply_train_residual_anchor_expert(
            out,
            base_pred_bch=base_pred_bch,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len),
            residual_anchor_phc=train_residual_anchor_phc,
            cfg=residual_cfg,
        )
        residual_delta_bch = out - before
    if learnable_output_anchor is not None and bool(learnable_cfg.get("enable", False)):
        if cluster_id_c is None:
            raise ValueError("moe.learnable_output_anchor requires cluster_id_c.")
        out = learnable_output_anchor(
            out,
            cluster_id_c=cluster_id_c,
            x_bcl=x_bcl,
            stat_delta_bch=stat_delta_bch,
            residual_delta_bch=residual_delta_bch,
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


__all__ = [
    'history_anchor_enabled',
    '_normalize_history_anchor_cfg',
    '_validate_strict_history_anchor_scope',
    '_MOE_OUTPUT_ANCHOR_KEYS',
    '_clone_anchor_cfg',
    '_merge_anchor_cfg',
    '_stat_anchor_default',
    '_residual_anchor_default',
    '_moe_output_anchor_default',
    '_MAIN_TABLE_MOE_OUTPUT_ANCHOR_DEFAULTS',
    '_normalize_anchor_default_dataset_name',
    'default_moe_output_anchor_cfg',
    'apply_default_moe_output_anchor_cfg',
    '_history_anchor_values',
    'apply_history_anchor_adapter',
    'apply_moe_history_anchor_expert',
    'build_train_phase_anchor_table',
    'build_train_phase_delta_anchor_table',
    'build_train_phase_residual_anchor_table',
    'build_train_stat_anchor_from_config',
    '_anchor_alpha_from_cfg',
    'apply_train_residual_anchor_expert',
    'apply_moe_output_anchor_experts',
    'apply_train_stat_input_centering',
    'apply_train_stat_anchor_expert',
    'select_channel_anchor_scales',
    'select_channel_horizon_anchor_scales',
    'select_train_stat_anchor_scales_from_loader',
    'build_train_residual_anchor_table_from_loader',
    'select_train_residual_anchor_scales_from_loader',
]
