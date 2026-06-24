from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.reader import read_csv_time_series
from src.data.windows import (
    WindowTensorDataset,
    global_zscore,
    make_label_range_windows,
    make_lazy_label_range_window_dataset,
    make_lazy_strict_window_dataset,
    make_strict_windows,
)
from src.models.cluster_predictor import build_cluster_predictor
from src.models.moe_gate import ClusterwiseMoEGate, scatter_mean_bc_to_bk, scatter_mean_bcf_to_bkf
from src.models.penalties import build_penalty_bank
from src.models.residual_moe import ClusterwisePredResidualMoE
from src.train import (
    _build_gate_routing_features,
    _explainability_train_subsplit_ranges,
    _normalize_gate_feature_mode,
    _normalize_history_anchor_cfg,
    _pred_residual_candidates_on_eval_path,
    _router_penalty_context_from_history,
    _select_rank_mask,
    _validate_strict_history_anchor_scope,
    apply_history_anchor_adapter,
    apply_train_stat_anchor_expert,
    apply_train_stat_input_centering,
    build_train_residual_anchor_table_from_loader,
    build_train_stat_anchor_from_config,
)
from src.utils.seed import set_seed
from src.utils.yaml_io import load_yaml


DEFAULT_ALLOWED_BY_CLUSTER = {
    0: ["jump", "delta"],
    1: ["amp_under", "delta", "jump"],
}
DEFAULT_BUCKET_Q = (4, 6)
DEFAULT_N_MIN = (64, 128)
DEFAULT_MARGINS = (0.0, 0.0005, 0.001)
DEFAULT_POSITIVE_RATE = (0.52, 0.55, 0.60)
BASE_MSE_PROXY_ABS_CORR = 0.50


def build_allowed_mask(
    *,
    penalty_names: List[str],
    K: int,
    allowed_by_cluster: Dict[int, Iterable[str]],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    name_to_idx = {str(name): i for i, name in enumerate(penalty_names)}
    mask = torch.zeros((int(K), len(penalty_names)), dtype=torch.bool, device=device)
    for raw_k, raw_names in allowed_by_cluster.items():
        k = int(raw_k)
        if k < 0 or k >= int(K):
            raise ValueError(f"allowed_by_cluster has invalid cluster id {k}.")
        for raw_name in raw_names:
            name = str(raw_name)
            if name not in name_to_idx:
                raise ValueError(f"unknown penalty {name!r}; available={penalty_names}")
            mask[k, name_to_idx[name]] = True
    return mask


def _safe_std(x: torch.Tensor) -> torch.Tensor:
    return x.std(dim=-1, unbiased=False).clamp_min(1.0e-6)


def _range(x: torch.Tensor) -> torch.Tensor:
    return x.amax(dim=-1) - x.amin(dim=-1)


def _slope(x: torch.Tensor) -> torch.Tensor:
    n = int(x.shape[-1])
    if n <= 1:
        return torch.zeros_like(x[..., 0])
    return (x[..., -1] - x[..., 0]) / float(n - 1)


def _diff_rms(x: torch.Tensor) -> torch.Tensor:
    if int(x.shape[-1]) <= 1:
        return torch.zeros_like(x[..., 0])
    d = x.diff(dim=-1)
    return d.pow(2).mean(dim=-1).clamp_min(0.0).sqrt()


def _d2_rms(x: torch.Tensor) -> torch.Tensor:
    if int(x.shape[-1]) <= 2:
        return torch.zeros_like(x[..., 0])
    d2 = x.diff(dim=-1).diff(dim=-1)
    return d2.pow(2).mean(dim=-1).clamp_min(0.0).sqrt()


def _turning_rate(x: torch.Tensor) -> torch.Tensor:
    if int(x.shape[-1]) <= 2:
        return torch.zeros_like(x[..., 0])
    d = x.diff(dim=-1)
    turns = (d[..., 1:] * d[..., :-1]) < 0
    return turns.to(dtype=x.dtype).mean(dim=-1)


def _jump_stats(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if int(x.shape[-1]) <= 1:
        z = torch.zeros_like(x[..., 0])
        return z, z
    d = x.diff(dim=-1).abs()
    scale = _safe_std(x).unsqueeze(-1)
    norm_d = d / scale
    return norm_d.amax(dim=-1), (norm_d > 2.5).to(dtype=x.dtype).mean(dim=-1)


def _corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n = min(int(a.shape[-1]), int(b.shape[-1]))
    if n <= 1:
        return torch.zeros_like(a[..., 0])
    aa = a[..., -n:]
    bb = b[..., :n]
    aa = aa - aa.mean(dim=-1, keepdim=True)
    bb = bb - bb.mean(dim=-1, keepdim=True)
    denom = aa.pow(2).sum(dim=-1).sqrt() * bb.pow(2).sum(dim=-1).sqrt()
    return (aa * bb).sum(dim=-1) / denom.clamp_min(1.0e-6)


def _low_high_ratio(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    n = int(x.shape[-1])
    if n <= 3:
        z = torch.zeros_like(x[..., 0])
        return z, torch.full_like(z, float(n))
    centered = x - x.mean(dim=-1, keepdim=True)
    power = torch.fft.rfft(centered.to(dtype=torch.float32), dim=-1).abs().pow(2)
    bins = int(power.shape[-1])
    low_end = max(2, min(bins, n // 8 + 1))
    high_start = max(low_end, min(bins - 1, n // 4))
    low = power[..., 1:low_end].sum(dim=-1)
    high = power[..., high_start:].sum(dim=-1)
    ratio = low / high.clamp_min(1.0e-6)
    nonzero = power[..., 1:]
    idx = nonzero.argmax(dim=-1).to(dtype=torch.float32) + 1.0
    period = float(n) / idx.clamp_min(1.0)
    return ratio.to(dtype=x.dtype), period.to(dtype=x.dtype)


def compute_shape_features(
    x_bcl: torch.Tensor,
    y_base_bch: torch.Tensor,
    cluster_id_c: torch.Tensor,
    K: int,
) -> Tuple[torch.Tensor, List[str]]:
    """Target-free shape features from history and base prediction only."""
    if x_bcl.dim() != 3:
        raise ValueError("x_bcl must have shape [B,C,L].")
    if y_base_bch.dim() != 3:
        raise ValueError("y_base_bch must have shape [B,C,H].")
    if tuple(x_bcl.shape[:2]) != tuple(y_base_bch.shape[:2]):
        raise ValueError("x_bcl and y_base_bch must share [B,C].")
    H = int(y_base_bch.shape[-1])
    hist = x_bcl[..., -min(int(x_bcl.shape[-1]), H):]
    base = y_base_bch
    hist_std = _safe_std(hist)
    base_std = _safe_std(base)
    hist_range = _range(hist)
    base_range = _range(base)
    hist_slope = _slope(hist)
    base_slope = _slope(base)
    hist_diff = _diff_rms(hist)
    base_diff = _diff_rms(base)
    hist_d2 = _d2_rms(hist)
    base_d2 = _d2_rms(base)
    hist_turn = _turning_rate(hist)
    base_turn = _turning_rate(base)
    hist_jump_mag, hist_jump_density = _jump_stats(hist)
    base_jump_mag, base_jump_density = _jump_stats(base)
    hist_lh, hist_period = _low_high_ratio(hist)
    base_lh, base_period = _low_high_ratio(base)
    corr = _corr(hist, base)
    names = [
        "history_slope",
        "y_base_slope",
        "slope_mismatch_abs",
        "slope_mismatch_signed",
        "history_diff_rms",
        "y_base_diff_rms",
        "diff_rms_ratio_base_to_history",
        "history_d2_rms",
        "y_base_d2_rms",
        "d2_rms_ratio_base_to_history",
        "history_turning_rate",
        "y_base_turning_rate",
        "turning_rate_mismatch_abs",
        "history_jump_magnitude",
        "y_base_jump_magnitude",
        "history_jump_density",
        "y_base_jump_density",
        "range_ratio_base_to_history",
        "std_ratio_base_to_history",
        "forecast_history_shape_corr",
        "history_low_high_energy_ratio",
        "y_base_low_high_energy_ratio",
        "low_high_energy_ratio_mismatch_abs",
        "dominant_period_mismatch",
        "base_mean_minus_last_history_over_history_std",
    ]
    feat_bcf = torch.stack(
        [
            hist_slope,
            base_slope,
            (base_slope - hist_slope).abs(),
            base_slope - hist_slope,
            hist_diff,
            base_diff,
            base_diff / hist_diff.clamp_min(1.0e-6),
            hist_d2,
            base_d2,
            base_d2 / hist_d2.clamp_min(1.0e-6),
            hist_turn,
            base_turn,
            (base_turn - hist_turn).abs(),
            hist_jump_mag,
            base_jump_mag,
            hist_jump_density,
            base_jump_density,
            base_range / hist_range.clamp_min(1.0e-6),
            base_std / hist_std,
            corr,
            hist_lh,
            base_lh,
            (base_lh - hist_lh).abs(),
            (base_period - hist_period).abs() / float(max(H, 1)),
            (base.mean(dim=-1) - hist[..., -1]) / hist_std,
        ],
        dim=-1,
    )
    feat_bcf = torch.nan_to_num(feat_bcf, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
    return scatter_mean_bcf_to_bkf(feat_bcf, cluster_id_c.to(device=feat_bcf.device), int(K)), names


def fit_quantile_bucket_edges(features_bkf: torch.Tensor, feature_names: List[str], q: int) -> Dict[str, object]:
    if features_bkf.dim() != 3:
        raise ValueError("features_bkf must have shape [B,K,F].")
    q = int(q)
    if q < 2:
        raise ValueError("q must be >= 2.")
    _, K, F = [int(v) for v in features_bkf.shape]
    qs = torch.linspace(0.0, 1.0, q + 1, dtype=features_bkf.dtype, device=features_bkf.device)[1:-1]
    edges: List[List[List[float]]] = []
    for k in range(K):
        cluster_edges = []
        for f in range(F):
            vals = features_bkf[:, k, f]
            vals = vals[torch.isfinite(vals)]
            if vals.numel() == 0:
                cur = [0.0 for _ in range(q - 1)]
            else:
                cur = [float(v) for v in torch.quantile(vals, qs).detach().cpu().tolist()]
            cluster_edges.append(cur)
        edges.append(cluster_edges)
    return {"q": q, "feature_names": list(feature_names), "edges": edges}


def apply_quantile_bucket_edges(features_bkf: torch.Tensor, edges: Dict[str, object]) -> torch.Tensor:
    if features_bkf.dim() != 3:
        raise ValueError("features_bkf must have shape [B,K,F].")
    q = int(edges["q"])
    raw_edges = edges["edges"]
    B, K, F = [int(v) for v in features_bkf.shape]
    out = torch.zeros((B, K, F), dtype=torch.long, device=features_bkf.device)
    for k in range(K):
        for f in range(F):
            e = torch.as_tensor(raw_edges[k][f], device=features_bkf.device, dtype=features_bkf.dtype)
            vals = features_bkf[:, k, f]
            bucket = torch.bucketize(vals.contiguous(), e, right=False).clamp(0, q - 1)
            out[:, k, f] = bucket.to(dtype=torch.long)
    return out


def compute_bucket_gain_stats(
    *,
    bucket_ids: torch.Tensor,
    gains_bkp: torch.Tensor,
    allowed_mask_kp: torch.Tensor,
    feature_names: List[str],
    penalty_names: List[str],
    split_name: str,
    q: int,
) -> List[Dict[str, object]]:
    if bucket_ids.dim() != 3:
        raise ValueError("bucket_ids must have shape [B,K,F].")
    if gains_bkp.dim() != 3:
        raise ValueError("gains_bkp must have shape [B,K,P].")
    B, K, F = [int(v) for v in bucket_ids.shape]
    if tuple(gains_bkp.shape[:2]) != (B, K):
        raise ValueError("bucket_ids and gains_bkp must share [B,K].")
    P = int(gains_bkp.shape[-1])
    allowed = allowed_mask_kp.to(device=gains_bkp.device, dtype=torch.bool)
    if tuple(allowed.shape) != (K, P):
        raise ValueError("allowed_mask_kp must have shape [K,P].")
    rows: List[Dict[str, object]] = []
    for k in range(K):
        for f, feature_name in enumerate(feature_names):
            ids = bucket_ids[:, k, f]
            for b in range(int(q)):
                in_bucket = ids == int(b)
                support = int(in_bucket.sum().item())
                if support <= 0:
                    continue
                for p, penalty in enumerate(penalty_names):
                    if not bool(allowed[k, p].item()):
                        continue
                    vals = gains_bkp[:, k, p][in_bucket]
                    vals = vals[torch.isfinite(vals)]
                    if int(vals.numel()) <= 0:
                        continue
                    rows.append(
                        {
                            "split": str(split_name),
                            "q": int(q),
                            "cluster_id": int(k),
                            "feature": str(feature_name),
                            "feature_index": int(f),
                            "bucket": int(b),
                            "penalty": str(penalty),
                            "penalty_index": int(p),
                            "support_count": int(vals.numel()),
                            "mean_gain": float(vals.mean().item()),
                            "median_gain": float(vals.median().item()),
                            "positive_rate": float((vals > 0.0).to(dtype=torch.float32).mean().item()),
                        }
                    )
    return rows


def _pearson(x: torch.Tensor, y: torch.Tensor) -> Optional[float]:
    x = x.detach().cpu().to(dtype=torch.float64).view(-1)
    y = y.detach().cpu().to(dtype=torch.float64).view(-1)
    mask = torch.isfinite(x) & torch.isfinite(y)
    if int(mask.sum().item()) < 3:
        return None
    x = x[mask] - x[mask].mean()
    y = y[mask] - y[mask].mean()
    denom = x.pow(2).sum().sqrt() * y.pow(2).sum().sqrt()
    if float(denom.item()) <= 1.0e-12:
        return None
    return float((x * y).sum().item() / denom.item())


def _feature_summary(features_bkf: torch.Tensor, feature_names: List[str], split: str) -> Dict[str, object]:
    out = {"split": split, "feature_names": list(feature_names), "per_cluster": []}
    for k in range(int(features_bkf.shape[1])):
        rows = []
        for f, name in enumerate(feature_names):
            vals = features_bkf[:, k, f].detach().cpu().to(dtype=torch.float32)
            vals = vals[torch.isfinite(vals)]
            if int(vals.numel()) <= 0:
                continue
            qs = torch.quantile(vals, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
            rows.append(
                {
                    "feature": name,
                    "mean": float(vals.mean().item()),
                    "std": float(vals.std(unbiased=False).item()),
                    "p05": float(qs[0].item()),
                    "p25": float(qs[1].item()),
                    "p50": float(qs[2].item()),
                    "p75": float(qs[3].item()),
                    "p95": float(qs[4].item()),
                }
            )
        out["per_cluster"].append({"cluster_id": int(k), "features": rows})
    return out


def _correlations_with_gain(
    samples: Dict[str, torch.Tensor],
    feature_names: List[str],
    penalty_names: List[str],
    allowed_mask_kp: torch.Tensor,
    split: str,
) -> List[Dict[str, object]]:
    features = samples["features"]
    gains = samples["gains"]
    rows = []
    for k in range(int(features.shape[1])):
        for p, penalty in enumerate(penalty_names):
            if not bool(allowed_mask_kp[k, p].item()):
                continue
            for f, feature in enumerate(feature_names):
                rows.append(
                    {
                        "split": split,
                        "cluster_id": int(k),
                        "penalty": penalty,
                        "feature": feature,
                        "corr": _pearson(features[:, k, f], gains[:, k, p]),
                    }
                )
    return rows


def _correlations_with_base_mse(
    samples: Dict[str, torch.Tensor],
    feature_names: List[str],
    split: str,
) -> List[Dict[str, object]]:
    features = samples["features"]
    base_mse = samples["base_mse"]
    rows = []
    for k in range(int(features.shape[1])):
        for f, feature in enumerate(feature_names):
            rows.append(
                {
                    "split": split,
                    "cluster_id": int(k),
                    "feature": feature,
                    "corr": _pearson(features[:, k, f], base_mse[:, k]),
                }
            )
    return rows


def _bucket_key(row: Dict[str, object]) -> Tuple[int, int, str, int, str]:
    return (
        int(row["q"]),
        int(row["cluster_id"]),
        str(row["feature"]),
        int(row["bucket"]),
        str(row["penalty"]),
    )


def _merge_bucket_stats(
    rows_by_split: Dict[str, List[Dict[str, object]]],
    base_mse_corr_rows: List[Dict[str, object]],
) -> Dict[str, object]:
    by_key: Dict[Tuple[int, int, str, int, str], Dict[str, object]] = {}
    for split, rows in rows_by_split.items():
        for row in rows:
            key = _bucket_key(row)
            merged = by_key.setdefault(
                key,
                {
                    "q": key[0],
                    "cluster_id": key[1],
                    "feature": key[2],
                    "bucket": key[3],
                    "penalty": key[4],
                    "splits": {},
                },
            )
            merged["splits"][split] = {
                "support_count": int(row["support_count"]),
                "mean_gain": float(row["mean_gain"]),
                "positive_rate": float(row["positive_rate"]),
            }
    proxy_corr: Dict[Tuple[int, str], float] = {}
    for row in base_mse_corr_rows:
        corr = row.get("corr")
        if corr is None:
            continue
        key = (int(row["cluster_id"]), str(row["feature"]))
        proxy_corr[key] = max(abs(float(corr)), proxy_corr.get(key, 0.0))
    accepted = []
    merged_rows = []
    for row in by_key.values():
        splits = row["splits"]
        fit = splits.get("train_fit")
        holdout = splits.get("train_holdout")
        val = splits.get("val")
        proxy_abs_corr = proxy_corr.get((int(row["cluster_id"]), str(row["feature"])), 0.0)
        row["base_mse_proxy_abs_corr_train_max"] = float(proxy_abs_corr)
        row["base_mse_proxy_flag"] = bool(proxy_abs_corr >= BASE_MSE_PROXY_ABS_CORR)
        if fit is not None and holdout is not None:
            row["fit_holdout_sign_agreement"] = bool(
                math.copysign(1.0, float(fit["mean_gain"])) == math.copysign(1.0, float(holdout["mean_gain"]))
            )
        else:
            row["fit_holdout_sign_agreement"] = None
        if holdout is not None and val is not None:
            row["holdout_val_sign_agreement_diagnostic"] = bool(
                math.copysign(1.0, float(holdout["mean_gain"])) == math.copysign(1.0, float(val["mean_gain"]))
            )
        else:
            row["holdout_val_sign_agreement_diagnostic"] = None
        passes = []
        if fit is not None and holdout is not None and not bool(row["base_mse_proxy_flag"]):
            for n_min in DEFAULT_N_MIN:
                for margin in DEFAULT_MARGINS:
                    for pos_rate in DEFAULT_POSITIVE_RATE:
                        ok = (
                            int(fit["support_count"]) >= int(n_min)
                            and int(holdout["support_count"]) >= int(n_min)
                            and float(fit["mean_gain"]) > float(margin)
                            and float(holdout["mean_gain"]) > float(margin)
                            and float(holdout["positive_rate"]) >= float(pos_rate)
                        )
                        if ok:
                            passes.append(
                                {
                                    "n_min": int(n_min),
                                    "margin": float(margin),
                                    "positive_rate_holdout": float(pos_rate),
                                }
                            )
        row["passes_thresholds"] = passes
        if passes:
            accepted.append(row)
        merged_rows.append(row)
    accepted.sort(
        key=lambda r: (
            min(
                float(r["splits"]["train_fit"]["mean_gain"]),
                float(r["splits"]["train_holdout"]["mean_gain"]),
            ),
            float(r["splits"]["train_holdout"]["positive_rate"]),
            int(r["splits"]["train_holdout"]["support_count"]),
        ),
        reverse=True,
    )
    return {
        "rows": merged_rows,
        "accepted": accepted,
        "accepted_count": len(accepted),
        "threshold_grid": {
            "n_min": list(DEFAULT_N_MIN),
            "margin": list(DEFAULT_MARGINS),
            "positive_rate_holdout": list(DEFAULT_POSITIVE_RATE),
            "base_mse_proxy_abs_corr_max": BASE_MSE_PROXY_ABS_CORR,
        },
    }


def _make_loaders(cfg: Dict[str, object], data_tc: torch.Tensor, batch_size: int):
    data_cfg = cfg["data"]
    max_rows = int(data_cfg.get("max_rows", 0) or 0)
    if max_rows > 0:
        data_tc = data_tc[:max_rows]
    T = int(data_tc.shape[0])
    t_train = int(T * float(data_cfg["train_ratio"]))
    t_val = int(T * (float(data_cfg["train_ratio"]) + float(data_cfg["val_ratio"])))
    norm_cfg = cfg["normalize"]
    if bool(norm_cfg["global_zscore"]):
        if bool(norm_cfg.get("train_only", False)):
            train_seg = data_tc[:t_train]
            mean_c = train_seg.mean(dim=0, keepdim=True)
            std_c = train_seg.std(dim=0, keepdim=True).clamp_min(1.0e-6)
            data_tc = (data_tc - mean_c) / std_c
        else:
            data_tc, _, _ = global_zscore(data_tc)
    L = int(cfg["window"]["input_len"])
    H = int(cfg["window"]["pred_len"])
    past_context = bool((cfg.get("window", {}) or {}).get("past_context", False))
    lazy_windows = bool((cfg.get("window", {}) or {}).get("lazy", False))
    skip_test = bool((cfg.get("eval", {}) or {}).get("skip_test", True))
    data_window_tc = data_tc.detach().cpu()
    if lazy_windows:
        dtr = make_lazy_strict_window_dataset(data_window_tc, L, H, 0, t_train)
        if past_context:
            dva, val_eval_start = make_lazy_label_range_window_dataset(data_window_tc, L, H, t_train, t_val)
            dte = make_lazy_strict_window_dataset(data_window_tc, L, H, 0, 0) if skip_test else make_lazy_label_range_window_dataset(data_window_tc, L, H, t_val, T)[0]
        else:
            dva = make_lazy_strict_window_dataset(data_window_tc, L, H, t_train, t_val)
            val_eval_start = t_train
            dte = make_lazy_strict_window_dataset(data_window_tc, L, H, 0, 0) if skip_test else make_lazy_strict_window_dataset(data_window_tc, L, H, t_val, T)
    else:
        xtr, ytr = make_strict_windows(data_window_tc, L, H, 0, t_train)
        if past_context:
            xva, yva, val_eval_start = make_label_range_windows(data_window_tc, L, H, t_train, t_val)
        else:
            xva, yva = make_strict_windows(data_window_tc, L, H, t_train, t_val)
            val_eval_start = t_train
        dtr = WindowTensorDataset(xtr, ytr)
        dva = WindowTensorDataset(xva, yva)
        dte = WindowTensorDataset(
            torch.empty(0, data_tc.shape[1], L, dtype=data_window_tc.dtype),
            torch.empty(0, data_tc.shape[1], H, dtype=data_window_tc.dtype),
        )
    ranges = _explainability_train_subsplit_ranges(
        num_windows=len(dtr),
        holdout_fraction=float(((cfg.get("moe", {}) or {}).get("explainability", {}) or {}).get("train_holdout_fraction", 0.30)),
    )
    loaders = {
        "train_fit": DataLoader(Subset(dtr, range(*ranges["train_fit"])), batch_size=batch_size, shuffle=False),
        "train_holdout": DataLoader(Subset(dtr, range(*ranges["train_holdout"])), batch_size=batch_size, shuffle=False),
        "val": DataLoader(dva, batch_size=batch_size, shuffle=False),
    }
    eval_starts = {"train_fit": 0, "train_holdout": 0, "val": int(val_eval_start)}
    train_loader = DataLoader(dtr, batch_size=batch_size, shuffle=False)
    return data_window_tc, loaders, eval_starts, train_loader, {"T": T, "t_train": t_train, "t_val": t_val, "L": L, "H": H, "skip_test": skip_test}


def _build_modules(cfg: Dict[str, object], checkpoint: Dict[str, object], device: torch.device):
    meta = checkpoint["meta"]
    model_cfg = dict(meta.get("model_cfg", cfg.get("model", {}) or {}))
    moe_cfg = dict(meta.get("moe_cfg", cfg.get("moe", {}) or {}))
    penalty_names = list(meta.get("penalty_names", cfg["penalties"]["enabled"]))
    cluster_id_c = meta["cluster_id_c"].to(device=device, dtype=torch.long)
    K = int(meta["K"])
    C = int(meta["num_channels"])
    L = int(meta["input_len"])
    H = int(meta["pred_len"])
    model = build_cluster_predictor(
        num_clusters=K,
        input_len=L,
        pred_len=H,
        model_cfg=model_cfg,
        num_channels=C,
        cluster_id_c=cluster_id_c,
    ).to(device)
    model.load_state_dict({k: v.to(device) for k, v in checkpoint["model_state"].items()}, strict=True)
    allow_skip = bool(moe_cfg.get("allow_skip", False)) and bool(moe_cfg.get("enable", True))
    gate = ClusterwiseMoEGate(
        num_clusters=K,
        feat_dim=int(meta.get("gate_feat_dim", 10)),
        num_penalties=len(penalty_names),
        hidden_dim=int(moe_cfg.get("gate_hidden_dim", 64)),
        topk=int(moe_cfg.get("topk", 1)),
        allow_skip=allow_skip,
        skip_init_bias=float(moe_cfg.get("skip_init_bias", -2.0)),
        skip_competes=bool(moe_cfg.get("skip_competes_with_penalties", False)),
        skip_argmax_noop=bool(moe_cfg.get("skip_argmax_noop", False)),
    ).to(device)
    gate.temperature = float(moe_cfg.get("gate_temperature", 1.0))
    gate.noise_std = float(moe_cfg.get("gate_noise_std", 0.0))
    gate.logit_clip = float(moe_cfg.get("gate_logit_clip", 0.0))
    gate.prob_floor = float(moe_cfg.get("gate_prob_floor", 0.0))
    gate.load_state_dict({k: v.to(device) for k, v in checkpoint["gate_state"].items()}, strict=True)
    pred_cfg = moe_cfg.get("pred_side_residual", {}) or {}
    ch_cfg = pred_cfg.get("channel_expert_adapters", {}) or {}
    if bool(ch_cfg.get("enable", False)) and str(ch_cfg.get("mode", "")).lower() in {"all", "all_channels"}:
        channel_expert_mask_c = torch.ones(C, dtype=torch.bool, device=device)
    else:
        channel_expert_mask_c = None
    pred_residual = ClusterwisePredResidualMoE(
        num_clusters=K,
        num_penalties=len(penalty_names),
        input_len=L,
        pred_len=H,
        hidden_dim=int(pred_cfg.get("corrector_hidden", 32)),
        init_alpha=float(pred_cfg.get("init_alpha", -3.0)),
        alpha_scale=float(pred_cfg.get("alpha_scale", 0.5)),
        use_y_base_input=bool(pred_cfg.get("use_y_base_input", True)),
        feature_mode=str(pred_cfg.get("feature_mode", "legacy")),
        residual_clip=float(pred_cfg.get("residual_clip", 0.0)),
        intervention_enable=bool(pred_cfg.get("intervention_enable", False)),
        intervention_init=float(pred_cfg.get("intervention_init", -2.0)),
        penalty_selector_enable=bool(pred_cfg.get("penalty_selector_enable", False)),
        selector_temperature=float(pred_cfg.get("selector_temperature", 1.0)),
        selector_use_cluster_context=bool(pred_cfg.get("selector_use_cluster_context", True)),
        fusion_gate_enable=bool(pred_cfg.get("fusion_gate_enable", False)),
        fusion_init=float(pred_cfg.get("fusion_init", 0.0)),
        fusion_use_cluster_context=bool(pred_cfg.get("fusion_use_cluster_context", True)),
        num_channels=C,
        channel_expert_mask_c=channel_expert_mask_c,
        channel_expert_cluster_id_c=cluster_id_c,
        channel_expert_mode=str(ch_cfg.get("mode_type", "override")),
        penalty_names=penalty_names,
        seasonal_anchor_names=list(pred_cfg.get("seasonal_anchor_names", [])),
        seasonal_anchor_period=int(pred_cfg.get("seasonal_anchor_period", 96)),
        seasonal_anchor_num_periods=int(pred_cfg.get("seasonal_anchor_num_periods", 1)),
        seasonal_anchor_scale=float(pred_cfg.get("seasonal_anchor_scale", 1.0)),
    ).to(device)
    pred_residual.load_state_dict({k: v.to(device) for k, v in checkpoint["pred_residual_state"].items()}, strict=True)
    model.eval()
    gate.eval()
    pred_residual.eval()
    return model, gate, pred_residual, cluster_id_c, K, moe_cfg, penalty_names


@torch.no_grad()
def _compute_penalty_scale(loader: DataLoader, penalty_names: List[str], penalty_fns: Dict[str, object], H: int, device: torch.device) -> torch.Tensor:
    if len(loader) == 0:
        return torch.ones(len(penalty_names), device=device)
    P = len(penalty_names)
    sum_pos = torch.zeros(P, device=device)
    cnt_pos = torch.zeros(P, device=device)
    sum_all = torch.zeros(P, device=device)
    cnt_all = 0
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        yhat = x[..., -1:].expand(-1, -1, H)
        pen = torch.stack([penalty_fns[name](yhat, y) for name in penalty_names], dim=-1)
        flat = pen.reshape(-1, P)
        sum_all += flat.sum(dim=0)
        cnt_all += int(flat.shape[0])
        pos = flat > 0
        sum_pos += (flat * pos).sum(dim=0)
        cnt_pos += pos.sum(dim=0)
    mean_all = sum_all / max(cnt_all, 1)
    mean_pos = sum_pos / cnt_pos.clamp_min(1.0)
    return torch.where(cnt_pos > 0, mean_pos, mean_all).clamp_min(1.0e-3)


@torch.no_grad()
def _collect_shape_samples(
    *,
    split_name: str,
    loader: DataLoader,
    eval_start: int,
    model: nn.Module,
    gate: ClusterwiseMoEGate,
    pred_residual: ClusterwisePredResidualMoE,
    cluster_id_c: torch.Tensor,
    K: int,
    moe_cfg: Dict[str, object],
    device: torch.device,
    penalty_names: List[str],
    penalty_fns: Dict[str, object],
    penalty_scale: torch.Tensor,
    allowed_mask_kp: torch.Tensor,
    history_anchor_cfg: Dict[str, object],
    observed_history_tc: torch.Tensor,
    input_len: int,
    model_train_stat_adapter_pc: Optional[torch.Tensor],
    model_train_stat_adapter_cfg: Dict[str, object],
    train_stat_anchor_pc: Optional[torch.Tensor],
    train_residual_anchor_phc: Optional[torch.Tensor],
    gate_feature_mode: str,
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    allow_skip = bool(moe_cfg.get("allow_skip", False)) and bool(moe_cfg.get("enable", True))
    router_mode = str(moe_cfg.get("router_mode", "learned")).lower()
    router_weight = float(moe_cfg.get("router_penalty_context_weight", 0.0))
    router_detach = bool(moe_cfg.get("router_detach_penalty_context", True))
    router_score = str(moe_cfg.get("router_penalty_context_score", "high_violation")).lower()
    select_ranks = moe_cfg.get("select_ranks", None)
    if select_ranks is not None:
        select_ranks = [int(v) for v in select_ranks]
    gate_soft_weight = float(moe_cfg.get("gate_soft_weight", 0.0))
    feats, gains, base_mses, starts = [], [], [], []
    feature_names: Optional[List[str]] = None
    for x, y, idx in loader:
        x = x.to(device)
        y = y.to(device)
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
            penalty_context_weight=router_weight,
            penalty_context_detach=router_detach,
            penalty_context_score=router_score,
        )
        if select_ranks is not None:
            rank_mask = _select_rank_mask(probs_bkp, select_ranks, straight_through=False)
            mask_bkp = rank_mask
            if gate_soft_weight > 0.0:
                probs_sel = probs_bkp * rank_mask
                probs_sel = probs_sel / probs_sel.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
                target_mass = mask_bkp.detach().sum(dim=-1, keepdim=True).clamp_min(1.0)
                mask_bkp = (1.0 - gate_soft_weight) * mask_bkp + gate_soft_weight * probs_sel * target_mass
        pred_out = pred_residual(x, y_base, cluster_id_c, mask_bkp, skip_bk=skip_bk if allow_skip else None)
        y_base_final, cand_bcpH = _pred_residual_candidates_on_eval_path(
            y_base,
            pred_out,
            apply_output_anchors=True,
            x_bcl=x,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len),
            moe_cfg=moe_cfg,
            moe_enable=bool(moe_cfg.get("enable", True)),
            observed_history_tc=observed_history_tc,
            train_stat_anchor_pc=train_stat_anchor_pc,
            train_residual_anchor_phc=train_residual_anchor_phc,
        )
        if cand_bcpH is None:
            continue
        shape_bkf, names = compute_shape_features(x, y_base_final, cluster_id_c, K)
        feature_names = names
        base_err_bc = (y_base_final - y).pow(2).mean(dim=-1)
        cand_err_bcp = (cand_bcpH - y.unsqueeze(2)).pow(2).mean(dim=-1)
        gain_bcp = base_err_bc.unsqueeze(-1) - cand_err_bcp
        gain_bkp = scatter_mean_bcf_to_bkf(gain_bcp, cluster_id_c, K)
        base_mse_bk = scatter_mean_bc_to_bk(base_err_bc, cluster_id_c, K)
        gain_bkp = gain_bkp.masked_fill(~allowed_mask_kp.to(device=device).unsqueeze(0), float("nan"))
        feats.append(shape_bkf.detach().cpu())
        gains.append(gain_bkp.detach().cpu())
        base_mses.append(base_mse_bk.detach().cpu())
        starts.append(query_start_abs_b.detach().cpu())
    if not feats or feature_names is None:
        raise RuntimeError(f"no shape-prior samples collected for split {split_name}")
    return {
        "features": torch.cat(feats, dim=0),
        "gains": torch.cat(gains, dim=0),
        "base_mse": torch.cat(base_mses, dim=0),
        "query_start_abs": torch.cat(starts, dim=0),
    }, feature_names


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, object]:
    cfg = load_yaml(args.config)
    if bool(args.skip_test):
        cfg.setdefault("eval", {})["skip_test"] = True
    if not bool((cfg.get("eval", {}) or {}).get("skip_test", True)):
        raise ValueError("shape prior diagnostic refuses to run unless eval.skip_test is true.")
    requested = [str(s).lower() for s in args.splits]
    if "test" in requested:
        raise ValueError("shape prior diagnostic does not read test.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(cfg["exp"]["seed"]), deterministic=bool((cfg.get("exp", {}) or {}).get("deterministic", False)))
    device = torch.device(str(args.device or cfg["exp"].get("device", "cpu")) if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model, gate, pred_residual, cluster_id_c, K, moe_cfg, penalty_names = _build_modules(cfg, checkpoint, device)
    data_tc, _ = read_csv_time_series(cfg["data"]["csv_path"], date_col=int(cfg["data"]["date_col"]))
    data_window_tc, loaders, eval_starts, train_loader, window_meta = _make_loaders(
        cfg,
        data_tc,
        batch_size=int(cfg["train"]["batch_size"]),
    )
    observed_history_tc = data_window_tc
    penalty_fns = build_penalty_bank(penalty_names, jump_thr=float(cfg["penalties"].get("jump_threshold", 0.6)))
    penalty_scale = _compute_penalty_scale(train_loader, penalty_names, penalty_fns, int(window_meta["H"]), device)
    model_cfg = dict(checkpoint["meta"].get("model_cfg", cfg.get("model", {}) or {}))
    history_anchor_cfg = _normalize_history_anchor_cfg(model_cfg.get("history_anchor", cfg.get("history_anchor", {}) or {}))
    _validate_strict_history_anchor_scope(history_anchor_cfg, source="shape_prior.history_anchor")
    model_train_stat_adapter_cfg = model_cfg.get("train_stat_adapter", {}) or {}
    model_train_stat_adapter_pc, _, _ = build_train_stat_anchor_from_config(
        data_window_tc,
        train_end=int(window_meta["t_train"]),
        input_len=int(window_meta["L"]),
        pred_len=int(window_meta["H"]),
        cfg=model_train_stat_adapter_cfg,
        prefix="shape_prior.model.train_stat_adapter",
    )
    train_stat_anchor_cfg = moe_cfg.get("train_stat_anchor_expert", {}) or {}
    train_stat_anchor_pc, _, _ = build_train_stat_anchor_from_config(
        data_window_tc,
        train_end=int(window_meta["t_train"]),
        input_len=int(window_meta["L"]),
        pred_len=int(window_meta["H"]),
        cfg=train_stat_anchor_cfg,
        prefix="shape_prior.moe.train_stat_anchor_expert",
    )
    train_residual_anchor_phc = None
    train_residual_anchor_cfg = moe_cfg.get("train_residual_anchor_expert", {}) or {}
    if bool(train_residual_anchor_cfg.get("enable", False)):
        train_residual_anchor_phc, _, _ = build_train_residual_anchor_table_from_loader(
            model=model,
            loader=train_loader,
            cluster_id_c=cluster_id_c,
            device=device,
            history_anchor_cfg=history_anchor_cfg,
            observed_history_tc=observed_history_tc,
            input_len=int(window_meta["L"]),
            eval_start=0,
            period=int(train_residual_anchor_cfg.get("period", 96)),
            model_train_stat_adapter_pc=model_train_stat_adapter_pc,
            model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
            train_stat_anchor_pc=train_stat_anchor_pc,
            train_stat_anchor_cfg=train_stat_anchor_cfg,
        )
    allowed_mask = build_allowed_mask(
        penalty_names=penalty_names,
        K=K,
        allowed_by_cluster=DEFAULT_ALLOWED_BY_CLUSTER,
        device=torch.device("cpu"),
    )
    gate_feature_mode = _normalize_gate_feature_mode(str(checkpoint["meta"].get("gate_feature_mode", "history")))
    samples_by_split: Dict[str, Dict[str, torch.Tensor]] = {}
    feature_names: Optional[List[str]] = None
    for split in requested:
        samples, names = _collect_shape_samples(
            split_name=split,
            loader=loaders[split],
            eval_start=int(eval_starts[split]),
            model=model,
            gate=gate,
            pred_residual=pred_residual,
            cluster_id_c=cluster_id_c,
            K=K,
            moe_cfg=moe_cfg,
            device=device,
            penalty_names=penalty_names,
            penalty_fns=penalty_fns,
            penalty_scale=penalty_scale,
            allowed_mask_kp=allowed_mask.to(device=device),
            history_anchor_cfg=history_anchor_cfg,
            observed_history_tc=observed_history_tc,
            input_len=int(window_meta["L"]),
            model_train_stat_adapter_pc=model_train_stat_adapter_pc,
            model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
            train_stat_anchor_pc=train_stat_anchor_pc,
            train_residual_anchor_phc=train_residual_anchor_phc,
            gate_feature_mode=gate_feature_mode,
        )
        samples_by_split[split] = samples
        feature_names = names
        torch.save(
            {
                **samples,
                "feature_names": names,
                "penalty_names": penalty_names,
                "allowed_mask_kp": allowed_mask,
                "split": split,
            },
            out_dir / f"shape_prior_samples_{split}.pt",
        )
    assert feature_names is not None
    feature_summary = {
        "feature_names": feature_names,
        "splits": {
            split: _feature_summary(samples["features"], feature_names, split)
            for split, samples in samples_by_split.items()
        },
        "feature_source": "target_free_history_and_anchored_base_prediction",
        "uses_y_true_for_features": False,
    }
    gain_corr = []
    for split, samples in samples_by_split.items():
        gain_corr.extend(_correlations_with_gain(samples, feature_names, penalty_names, allowed_mask, split))
    base_corr = []
    for split in ("train_fit", "train_holdout"):
        if split in samples_by_split:
            base_corr.extend(_correlations_with_base_mse(samples_by_split[split], feature_names, split))
    bucket_rows_by_split: Dict[str, List[Dict[str, object]]] = {}
    bucket_edge_payload: Dict[str, object] = {}
    for q in DEFAULT_BUCKET_Q:
        edges = fit_quantile_bucket_edges(samples_by_split["train_fit"]["features"], feature_names, q=int(q))
        bucket_edge_payload[f"q{q}"] = edges
        for split, samples in samples_by_split.items():
            ids = apply_quantile_bucket_edges(samples["features"], edges)
            bucket_rows_by_split.setdefault(split, []).extend(
                compute_bucket_gain_stats(
                    bucket_ids=ids,
                    gains_bkp=samples["gains"],
                    allowed_mask_kp=allowed_mask,
                    feature_names=feature_names,
                    penalty_names=penalty_names,
                    split_name=split,
                    q=int(q),
                )
            )
    merged_buckets = _merge_bucket_stats(bucket_rows_by_split, base_corr)
    summary = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "out_dir": str(out_dir),
        "splits": requested,
        "skip_test": True,
        "feature_names": feature_names,
        "penalty_names_checkpoint": penalty_names,
        "filtered_branch_penalties": ["jump", "amp_under", "delta"],
        "allowed_by_cluster": {str(k): v for k, v in DEFAULT_ALLOWED_BY_CLUSTER.items()},
        "excluded_high_mse_correlated": ["level", "seasonal_align"],
        "allowed_mask_kp": allowed_mask.to(dtype=torch.long).tolist(),
        "samples": {split: int(samples["features"].shape[0]) for split, samples in samples_by_split.items()},
        "accepted_bucket_count": int(merged_buckets["accepted_count"]),
        "top_accepted_buckets": merged_buckets["accepted"][:25],
        "refuted": bool(int(merged_buckets["accepted_count"]) == 0),
        "refutation_reason": None
        if int(merged_buckets["accepted_count"]) > 0
        else "No target-free shape bucket passed train_fit/train_holdout support, gain, holdout positive-rate, and base-MSE proxy filters.",
    }
    _write_json(out_dir / "shape_feature_summary.json", feature_summary)
    _write_json(out_dir / "candidate_gain_correlations.json", {"rows": gain_corr})
    _write_json(out_dir / "base_mse_correlations_train_only.json", {"rows": base_corr})
    _write_json(out_dir / "shape_bucket_edges.json", bucket_edge_payload)
    _write_json(out_dir / "shape_bucket_gain_stats.json", merged_buckets)
    _write_json(out_dir / "shape_prior_step1_summary.json", summary)
    md_lines = [
        "# Shape Prior Step 1 Summary",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Splits: `{requested}`",
        f"- Test read: `no`",
        f"- Accepted bucket count: `{summary['accepted_bucket_count']}`",
        f"- Refuted: `{summary['refuted']}`",
    ]
    for row in summary["top_accepted_buckets"][:10]:
        fit = row["splits"]["train_fit"]
        holdout = row["splits"]["train_holdout"]
        val = row["splits"].get("val", {})
        md_lines.append(
            "- "
            f"q{row['q']} cluster{row['cluster_id']} `{row['feature']}` bucket {row['bucket']} "
            f"penalty `{row['penalty']}`: fit mean {fit['mean_gain']:.6f}, "
            f"holdout mean {holdout['mean_gain']:.6f}, holdout pos {holdout['positive_rate']:.3f}, "
            f"val mean {float(val.get('mean_gain', float('nan'))):.6f}"
        )
    (out_dir / "shape_prior_step1_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train_fit", "train_holdout", "val"])
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--device", default="")
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps({"out_dir": summary["out_dir"], "accepted_bucket_count": summary["accepted_bucket_count"], "refuted": summary["refuted"]}, indent=2))


if __name__ == "__main__":
    main()
