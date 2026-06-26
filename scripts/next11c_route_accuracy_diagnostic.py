from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.shape_prior_diagnostic import _build_modules, _compute_penalty_scale
from src.data.reader import read_csv_time_series
from src.data.windows import (
    WindowTensorDataset,
    global_zscore,
    make_label_range_windows,
    make_lazy_label_range_window_dataset,
    make_lazy_strict_window_dataset,
    make_strict_windows,
)
from src.models.penalties import build_penalty_bank
from src.train import (
    _build_gate_routing_features,
    _collect_penalty_route_learnability_tensors,
    _explainability_train_subsplit_ranges,
    _normalize_gate_feature_mode,
    _normalize_history_anchor_cfg,
    _pred_residual_candidates_on_eval_path,
    _router_penalty_context_from_history,
    _select_rank_mask,
    _stage2_route_audit_thresholds,
    _validate_strict_history_anchor_scope,
    apply_history_anchor_adapter,
    apply_train_stat_anchor_expert,
    apply_train_stat_input_centering,
    build_gate_prior_from_penalty_portrait,
    build_named_penalty_mask,
    build_topk_penalty_mask,
    build_train_residual_anchor_table_from_loader,
    build_train_stat_anchor_from_config,
    compute_cluster_penalty_portrait,
)
from src.utils.seed import set_seed
from src.utils.yaml_io import load_yaml


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if float(den) > 0.0 else 0.0


def _normalize_requested_splits(raw_splits: Optional[Iterable[str]]) -> List[str]:
    allowed = {"train_fit", "train_holdout", "val", "test"}
    if raw_splits is None:
        return ["train_fit", "train_holdout", "val", "test"]
    splits: List[str] = []
    for raw in raw_splits:
        split = str(raw).strip().lower()
        if split == "train":
            split = "train_fit"
        if split not in allowed:
            raise ValueError(f"unsupported split {raw!r}; expected train_fit, train_holdout, val, or test.")
        if split not in splits:
            splits.append(split)
    return splits or ["train_fit", "train_holdout", "val", "test"]


def _route_label_thresholds_from_config(
    cfg: Dict[str, object],
    *,
    source: str,
    min_abs_improvement: float,
    min_rel_improvement: float,
    min_candidate_delta_rms: float,
) -> Dict[str, object]:
    source_l = str(source or "manual").strip().lower()
    if source_l in {"", "manual", "cli"}:
        return {
            "min_abs_improvement": float(min_abs_improvement),
            "min_rel_improvement": float(min_rel_improvement),
            "min_candidate_delta_rms": float(min_candidate_delta_rms),
            "source": "manual",
        }
    if source_l not in {"stage2", "stage2_route_audit", "training"}:
        raise ValueError("threshold source must be 'manual' or 'stage2'.")

    moe_cfg = cfg.get("moe", {}) or {}
    if not isinstance(moe_cfg, dict):
        moe_cfg = {}
    diagnostics_cfg = cfg.get("diagnostics", {}) or {}
    if not isinstance(diagnostics_cfg, dict):
        diagnostics_cfg = {}
    stage2_route_audit_cfg = diagnostics_cfg.get("stage2_route_audit", {}) or {}
    if not isinstance(stage2_route_audit_cfg, dict):
        stage2_route_audit_cfg = {"enable": bool(stage2_route_audit_cfg)}

    moe_enable = bool(moe_cfg.get("enable", True))
    pred_residual_cfg = moe_cfg.get("pred_side_residual", {}) or {}
    if not isinstance(pred_residual_cfg, dict):
        pred_residual_cfg = {"enable": bool(pred_residual_cfg)}
    pred_residual_enable = bool(pred_residual_cfg.get("enable", False))

    route_ce_cfg = moe_cfg.get("route_ce_supervision", {}) or {}
    if not isinstance(route_ce_cfg, dict):
        route_ce_cfg = {"enable": bool(route_ce_cfg)}
    route_ce_enable = bool(route_ce_cfg.get("enable", False)) and moe_enable and pred_residual_enable
    route_ce_weight = float(route_ce_cfg.get("weight", 0.0)) if route_ce_enable else 0.0
    route_ce_min_abs = float(route_ce_cfg.get("min_abs_improvement", 0.0))
    route_ce_min_rel = float(route_ce_cfg.get("min_rel_improvement", 0.0))
    route_ce_min_delta = float(
        route_ce_cfg.get(
            "min_candidate_delta_rms",
            route_ce_cfg.get("candidate_action_floor", 0.0),
        )
    )

    binary_cfg = moe_cfg.get("binary_adoption_supervision", {}) or {}
    if not isinstance(binary_cfg, dict):
        binary_cfg = {"enable": bool(binary_cfg)}
    binary_enable = bool(binary_cfg.get("enable", False)) and moe_enable and pred_residual_enable
    binary_weight = float(binary_cfg.get("weight", 0.0)) if binary_enable else 0.0
    binary_min_abs = float(binary_cfg.get("min_abs_improvement", route_ce_min_abs))
    binary_min_rel = float(binary_cfg.get("min_rel_improvement", route_ce_min_rel))
    binary_min_delta = float(binary_cfg.get("min_candidate_delta_rms", route_ce_min_delta))

    return _stage2_route_audit_thresholds(
        stage2_route_audit_cfg=stage2_route_audit_cfg,
        route_ce_min_abs_improvement=route_ce_min_abs,
        route_ce_min_rel_improvement=route_ce_min_rel,
        route_ce_min_candidate_delta_rms=route_ce_min_delta,
        binary_adoption_weight=binary_weight,
        binary_adoption_min_abs_improvement=binary_min_abs,
        binary_adoption_min_rel_improvement=binary_min_rel,
        binary_adoption_min_candidate_delta_rms=binary_min_delta,
    )


def route_accuracy_summary(
    *,
    labels: torch.Tensor,
    current_pred: torch.Tensor,
    label_names: List[str],
    oracle_gain_mse: Optional[torch.Tensor] = None,
    min_abs_improvement: float = 0.0,
) -> Dict[str, object]:
    """Summarize current route accuracy against oracle labels, with skip as class 0."""
    label_count = int(len(label_names))
    if label_count <= 0:
        raise ValueError("label_names must be non-empty.")
    labels = labels.detach().cpu().to(dtype=torch.long).view(-1)
    current = current_pred.detach().cpu().to(dtype=torch.long).view(-1)
    if int(labels.numel()) != int(current.numel()):
        raise ValueError("labels and current_pred must have the same length.")
    gain_all = None
    if oracle_gain_mse is not None:
        gain_all = oracle_gain_mse.detach().cpu().to(dtype=torch.float32).view(-1)
        if int(gain_all.numel()) != int(labels.numel()):
            raise ValueError("oracle_gain_mse must have the same number of elements as labels.")
        if float(min_abs_improvement) > 0.0:
            keep_penalty = (labels > 0) & (gain_all > float(min_abs_improvement))
            labels = torch.where(keep_penalty, labels, torch.zeros_like(labels))
    valid = (labels >= 0) & (labels < label_count)
    if int(valid.sum().item()) <= 0:
        return {
            "samples": 0,
            "current_accuracy_all": 0.0,
            "majority_accuracy_all": 0.0,
            "oracle_skip_rate": 0.0,
            "actual_skip_rate": 0.0,
            "skip_recall": 0.0,
            "skip_precision": 0.0,
            "skip_false_positive_rate_on_oracle_penalty": 0.0,
            "penalty_accuracy_on_oracle_penalty": 0.0,
            "oracle_penalty_routed_to_skip_rate": 0.0,
            "oracle_penalty_routed_to_wrong_penalty_rate": 0.0,
            "confusion_matrix_counts": [[0 for _ in range(label_count)] for _ in range(label_count)],
            "per_class": {name: {"label_count": 0, "current_count": 0, "recall": 0.0, "precision": 0.0} for name in label_names},
        }
    labels = labels[valid]
    current = current[valid].clamp(0, label_count - 1)
    samples = int(labels.numel())
    label_counts = torch.bincount(labels, minlength=label_count)[:label_count]
    current_counts = torch.bincount(current, minlength=label_count)[:label_count]
    confusion = torch.zeros(label_count, label_count, dtype=torch.long)
    for y, pred in zip(labels.tolist(), current.tolist()):
        confusion[int(y), int(pred)] += 1

    correct = current == labels
    oracle_skip = labels == 0
    actual_skip = current == 0
    oracle_penalty = labels > 0
    actual_penalty = current > 0
    oracle_penalty_count = int(oracle_penalty.sum().item())
    oracle_skip_count = int(oracle_skip.sum().item())
    actual_skip_count = int(actual_skip.sum().item())
    correct_skip = int((oracle_skip & actual_skip).sum().item())
    wrong_penalty_class = oracle_penalty & actual_penalty & (current != labels)
    correct_penalty = oracle_penalty & (current == labels)
    skipped_oracle_penalty = oracle_penalty & actual_skip
    majority_acc = _safe_div(float(label_counts.max().item()), float(samples))

    per_class: Dict[str, Dict[str, object]] = {}
    for idx, name in enumerate(label_names):
        true_count = int(label_counts[idx].item())
        pred_count = int(current_counts[idx].item())
        true_positive = int(confusion[idx, idx].item())
        per_class[str(name)] = {
            "label_count": true_count,
            "label_rate": _safe_div(true_count, samples),
            "current_count": pred_count,
            "current_rate": _safe_div(pred_count, samples),
            "recall": _safe_div(true_positive, true_count),
            "precision": _safe_div(true_positive, pred_count),
        }

    out: Dict[str, object] = {
        "samples": samples,
        "current_accuracy_all": float(correct.to(dtype=torch.float32).mean().item()),
        "majority_accuracy_all": majority_acc,
        "lift_vs_majority": float(correct.to(dtype=torch.float32).mean().item() - majority_acc),
        "oracle_skip_count": oracle_skip_count,
        "oracle_skip_rate": _safe_div(oracle_skip_count, samples),
        "actual_skip_count": actual_skip_count,
        "actual_skip_rate": _safe_div(actual_skip_count, samples),
        "skip_recall": _safe_div(correct_skip, oracle_skip_count),
        "skip_precision": _safe_div(correct_skip, actual_skip_count),
        "skip_false_positive_count_on_oracle_penalty": int(skipped_oracle_penalty.sum().item()),
        "skip_false_positive_rate_on_oracle_penalty": _safe_div(
            int(skipped_oracle_penalty.sum().item()),
            oracle_penalty_count,
        ),
        "oracle_penalty_samples": oracle_penalty_count,
        "penalty_accuracy_on_oracle_penalty": _safe_div(int(correct_penalty.sum().item()), oracle_penalty_count),
        "oracle_penalty_routed_to_skip_rate": _safe_div(int(skipped_oracle_penalty.sum().item()), oracle_penalty_count),
        "oracle_penalty_routed_to_wrong_penalty_rate": _safe_div(
            int(wrong_penalty_class.sum().item()),
            oracle_penalty_count,
        ),
        "oracle_skip_routed_to_penalty_rate": _safe_div(int((oracle_skip & actual_penalty).sum().item()), oracle_skip_count),
        "label_counts": {str(name): int(label_counts[i].item()) for i, name in enumerate(label_names)},
        "label_rates": {str(name): _safe_div(int(label_counts[i].item()), samples) for i, name in enumerate(label_names)},
        "current_prediction_counts": {str(name): int(current_counts[i].item()) for i, name in enumerate(label_names)},
        "current_prediction_rates": {
            str(name): _safe_div(int(current_counts[i].item()), samples) for i, name in enumerate(label_names)
        },
        "confusion_matrix_counts": confusion.tolist(),
        "confusion_matrix_rows": list(label_names),
        "confusion_matrix_cols": list(label_names),
        "per_class": per_class,
    }
    if oracle_gain_mse is not None:
        gain = gain_all[valid] if gain_all is not None else oracle_gain_mse.detach().cpu().to(dtype=torch.float32).view(-1)[valid]
        finite = torch.isfinite(gain)
        if int(finite.sum().item()) > 0:
            gain_f = gain[finite]
            out["oracle_gain_mse_mean"] = float(gain_f.mean().item())
            out["oracle_gain_mse_positive_rate"] = float((gain_f > 0.0).to(dtype=torch.float32).mean().item())
            if int((oracle_penalty & finite).sum().item()) > 0:
                out["oracle_gain_mse_mean_on_oracle_penalty"] = float(gain[oracle_penalty & finite].mean().item())
            if int((oracle_skip & finite).sum().item()) > 0:
                out["oracle_gain_mse_mean_on_oracle_skip"] = float(gain[oracle_skip & finite].mean().item())
    return out


def _set_overlap_counts(oracle_positive: torch.Tensor, applied: torch.Tensor) -> Dict[str, object]:
    oracle = oracle_positive.detach().cpu().to(dtype=torch.bool)
    applied_bool = applied.detach().cpu().to(dtype=torch.bool)
    if oracle.shape != applied_bool.shape:
        raise ValueError(f"oracle_positive and applied must share shape, got {oracle.shape} vs {applied_bool.shape}.")
    intersection = oracle & applied_bool
    oracle_count = int(oracle.sum().item())
    applied_count = int(applied_bool.sum().item())
    intersection_count = int(intersection.sum().item())
    decisions = int(oracle.reshape(-1, oracle.shape[-1]).shape[0]) if oracle.dim() > 0 else 0
    return {
        "decisions": decisions,
        "intersection_count": intersection_count,
        "applied_count": applied_count,
        "oracle_positive_count": oracle_count,
        "precision": _safe_div(intersection_count, applied_count),
        "recall": _safe_div(intersection_count, oracle_count),
        "mean_applied_set_size": _safe_div(applied_count, decisions),
        "mean_oracle_positive_set_size": _safe_div(oracle_count, decisions),
    }


def _majority_set_for_group(oracle_np: torch.Tensor, applied_np: torch.Tensor) -> torch.Tensor:
    oracle = oracle_np.detach().cpu().to(dtype=torch.bool)
    applied = applied_np.detach().cpu().to(dtype=torch.bool)
    if oracle.dim() != 2 or applied.dim() != 2 or oracle.shape != applied.shape:
        raise ValueError("majority-set inputs must share shape [N,P].")
    P = int(oracle.shape[-1])
    if P <= 0 or int(oracle.shape[0]) <= 0:
        return torch.zeros(P, dtype=torch.bool)
    applied_sizes = applied.sum(dim=-1).to(dtype=torch.float32)
    k = int(round(float(applied_sizes.mean().item())))
    k = max(0, min(k, P))
    if k <= 0:
        return torch.zeros(P, dtype=torch.bool)
    rates = oracle.to(dtype=torch.float32).mean(dim=0)
    order = sorted(range(P), key=lambda p: (-float(rates[p].item()), int(p)))
    out = torch.zeros(P, dtype=torch.bool)
    out[order[:k]] = True
    return out


def _majority_counts_by_channel(oracle_positive_bcp: torch.Tensor, applied_bcp: torch.Tensor) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    oracle = oracle_positive_bcp.detach().cpu().to(dtype=torch.bool)
    applied = applied_bcp.detach().cpu().to(dtype=torch.bool)
    if oracle.dim() != 3 or applied.shape != oracle.shape:
        raise ValueError("channel majority inputs must share shape [B,C,P].")
    B, C, P = [int(v) for v in oracle.shape]
    majority = torch.zeros_like(applied)
    rows: List[Dict[str, object]] = []
    for c in range(C):
        majority_set = _majority_set_for_group(oracle[:, c, :].reshape(B, P), applied[:, c, :].reshape(B, P))
        majority[:, c, :] = majority_set.view(1, P).expand(B, P)
        row = _set_overlap_counts(oracle[:, c, :], majority[:, c, :])
        row["channel_index"] = int(c)
        row["majority_set_indices"] = [int(i) for i in torch.nonzero(majority_set, as_tuple=False).view(-1).tolist()]
        rows.append(row)
    return _set_overlap_counts(oracle, majority), rows


def topk_set_overlap_summary(
    *,
    gain_bcp: torch.Tensor,
    applied_bcp: torch.Tensor,
    cluster_id_c: torch.Tensor,
    penalty_names: List[str],
    channel_names: Optional[List[str]] = None,
    tau: float = 0.0,
) -> Dict[str, object]:
    """Summarize top-k applied-set overlap against oracle-positive penalty sets."""
    gains = gain_bcp.detach().cpu().to(dtype=torch.float32)
    applied = applied_bcp.detach().cpu().to(dtype=torch.bool)
    if gains.dim() != 3:
        raise ValueError("gain_bcp must have shape [B,C,P].")
    if applied.shape != gains.shape:
        raise ValueError(f"applied_bcp must match gain_bcp, got {applied.shape} vs {gains.shape}.")
    B, C, P = [int(v) for v in gains.shape]
    if len(penalty_names) != P:
        raise ValueError("penalty_names length must match gain_bcp P.")
    cid = cluster_id_c.detach().cpu().to(dtype=torch.long).view(-1)
    if int(cid.numel()) != C:
        raise ValueError("cluster_id_c length must match gain_bcp C.")
    names_c = list(channel_names or [f"channel_{i}" for i in range(C)])
    if len(names_c) != C:
        raise ValueError("channel_names length must match gain_bcp C.")

    oracle = torch.isfinite(gains) & (gains > float(tau))
    overall = _set_overlap_counts(oracle, applied)
    majority_overall, majority_per_channel = _majority_counts_by_channel(oracle, applied)

    per_cluster: List[Dict[str, object]] = []
    majority_per_cluster: List[Dict[str, object]] = []
    K = int(cid.max().item()) + 1 if C > 0 else 0
    for k in range(K):
        ch_mask = cid == int(k)
        if not bool(ch_mask.any().item()):
            continue
        row = _set_overlap_counts(oracle[:, ch_mask, :], applied[:, ch_mask, :])
        row["cluster_id"] = int(k)
        row["channels"] = [names_c[i] for i in torch.nonzero(ch_mask, as_tuple=False).view(-1).tolist()]
        per_cluster.append(row)

        oracle_k = oracle[:, ch_mask, :].reshape(-1, P)
        applied_k = applied[:, ch_mask, :].reshape(-1, P)
        majority_set = _majority_set_for_group(oracle_k, applied_k)
        majority_k = majority_set.view(1, P).expand_as(applied_k)
        maj_row = _set_overlap_counts(oracle_k, majority_k)
        maj_row["cluster_id"] = int(k)
        maj_row["majority_set"] = [penalty_names[i] for i in torch.nonzero(majority_set, as_tuple=False).view(-1).tolist()]
        majority_per_cluster.append(maj_row)

    per_channel: List[Dict[str, object]] = []
    for c in range(C):
        row = _set_overlap_counts(oracle[:, c, :], applied[:, c, :])
        row["channel_index"] = int(c)
        row["channel"] = names_c[c]
        row["cluster_id"] = int(cid[c].item())
        per_channel.append(row)
        majority_per_channel[c]["channel"] = names_c[c]
        majority_per_channel[c]["cluster_id"] = int(cid[c].item())
        majority_per_channel[c]["majority_set"] = [
            penalty_names[i] for i in majority_per_channel[c].pop("majority_set_indices")
        ]

    return {
        "tau": float(tau),
        "penalty_names": list(penalty_names),
        "overall": overall,
        "majority_overall": majority_overall,
        "per_cluster": per_cluster,
        "majority_per_cluster": majority_per_cluster,
        "per_channel": per_channel,
        "majority_per_channel": majority_per_channel,
    }


def skip_probability_summary(
    *,
    tensors: Dict[str, torch.Tensor],
) -> Dict[str, object]:
    names = list(tensors.get("feature_names", []) or [])
    if "skip_prob" not in names or "features" not in tensors:
        return {"available": False}
    idx = int(names.index("skip_prob"))
    features = tensors["features"].detach().cpu().to(dtype=torch.float32)
    if features.dim() != 3 or int(features.shape[0]) <= 0:
        return {"available": False}
    skip_prob = features[:, 0, idx].contiguous()
    labels = tensors["labels"].detach().cpu().to(dtype=torch.long).view(-1)
    valid = torch.isfinite(skip_prob) & (labels >= 0)
    if int(valid.sum().item()) <= 0:
        return {"available": False}
    skip_prob = skip_prob[valid]
    labels = labels[valid]
    oracle_skip = labels == 0
    oracle_penalty = labels > 0

    def _mean_for(mask: torch.Tensor) -> Optional[float]:
        if int(mask.sum().item()) <= 0:
            return None
        return float(skip_prob[mask].mean().item())

    return {
        "available": True,
        "mean": float(skip_prob.mean().item()),
        "p50": float(skip_prob.quantile(0.50).item()),
        "p95": float(skip_prob.quantile(0.95).item()),
        "max": float(skip_prob.max().item()),
        "gt_0_5_rate": float((skip_prob > 0.5).to(dtype=torch.float32).mean().item()),
        "oracle_skip_mean": _mean_for(oracle_skip),
        "oracle_penalty_mean": _mean_for(oracle_penalty),
    }


def _read_data_for_cfg(cfg: Dict[str, object]) -> Tuple[torch.Tensor, Dict[str, object]]:
    data_cfg = cfg["data"]
    data_tc, channel_names = read_csv_time_series(str(data_cfg["csv_path"]), date_col=int(data_cfg.get("date_col", 0)))
    max_rows = int(data_cfg.get("max_rows", 0) or 0)
    if max_rows > 0:
        data_tc = data_tc[:max_rows]
    T = int(data_tc.shape[0])
    t_train = int(T * float(data_cfg["train_ratio"]))
    t_val = int(T * (float(data_cfg["train_ratio"]) + float(data_cfg["val_ratio"])))
    norm_cfg = cfg.get("normalize", {}) or {}
    if bool(norm_cfg.get("global_zscore", False)):
        if bool(norm_cfg.get("train_only", False)):
            train_seg = data_tc[:t_train]
            mean_c = train_seg.mean(dim=0, keepdim=True)
            std_c = train_seg.std(dim=0, keepdim=True).clamp_min(1.0e-6)
            data_tc = (data_tc - mean_c) / std_c
        else:
            data_tc, _, _ = global_zscore(data_tc)
    return data_tc.detach().cpu(), {"T": T, "t_train": t_train, "t_val": t_val, "channel_names": list(channel_names)}


def _make_datasets(cfg: Dict[str, object], data_tc: torch.Tensor) -> Tuple[Dict[str, object], Dict[str, int]]:
    meta: Dict[str, int] = {}
    T = int(data_tc.shape[0])
    data_cfg = cfg["data"]
    t_train = int(T * float(data_cfg["train_ratio"]))
    t_val = int(T * (float(data_cfg["train_ratio"]) + float(data_cfg["val_ratio"])))
    L = int(cfg["window"]["input_len"])
    H = int(cfg["window"]["pred_len"])
    past_context = bool((cfg.get("window", {}) or {}).get("past_context", False))
    lazy_windows = bool((cfg.get("window", {}) or {}).get("lazy", False))
    if lazy_windows:
        dtr = make_lazy_strict_window_dataset(data_tc, L, H, 0, t_train)
        if past_context:
            dva, val_eval_start = make_lazy_label_range_window_dataset(data_tc, L, H, t_train, t_val)
            dte, test_eval_start = make_lazy_label_range_window_dataset(data_tc, L, H, t_val, T)
        else:
            dva = make_lazy_strict_window_dataset(data_tc, L, H, t_train, t_val)
            dte = make_lazy_strict_window_dataset(data_tc, L, H, t_val, T)
            val_eval_start = t_train
            test_eval_start = t_val
    else:
        xtr, ytr = make_strict_windows(data_tc, L, H, 0, t_train)
        dtr = WindowTensorDataset(xtr, ytr)
        if past_context:
            xva, yva, val_eval_start = make_label_range_windows(data_tc, L, H, t_train, t_val)
            xte, yte, test_eval_start = make_label_range_windows(data_tc, L, H, t_val, T)
        else:
            xva, yva = make_strict_windows(data_tc, L, H, t_train, t_val)
            xte, yte = make_strict_windows(data_tc, L, H, t_val, T)
            val_eval_start = t_train
            test_eval_start = t_val
        dva = WindowTensorDataset(xva, yva)
        dte = WindowTensorDataset(xte, yte)
    meta.update(
        {
            "T": int(T),
            "t_train": int(t_train),
            "t_val": int(t_val),
            "L": int(L),
            "H": int(H),
            "past_context": bool(past_context),
            "lazy_windows": bool(lazy_windows),
            "val_eval_start": int(val_eval_start),
            "test_eval_start": int(test_eval_start),
            "train_windows": int(len(dtr)),
            "val_windows": int(len(dva)),
            "test_windows": int(len(dte)),
        }
    )
    return {"train": dtr, "val": dva, "test": dte}, meta


def _make_loaders(
    cfg: Dict[str, object],
    data_tc: torch.Tensor,
    batch_size: int,
) -> Tuple[Dict[str, DataLoader], Dict[str, int], DataLoader, Dict[str, int]]:
    datasets, meta = _make_datasets(cfg, data_tc)
    train_dataset = datasets["train"]
    ranges = _explainability_train_subsplit_ranges(
        num_windows=len(train_dataset),
        holdout_fraction=float(((cfg.get("moe", {}) or {}).get("explainability", {}) or {}).get("train_holdout_fraction", 0.30)),
    )
    loaders = {
        "train_fit": DataLoader(Subset(train_dataset, range(*ranges["train_fit"])), batch_size=batch_size, shuffle=False),
        "train_holdout": DataLoader(
            Subset(train_dataset, range(*ranges["train_holdout"])),
            batch_size=batch_size,
            shuffle=False,
        ),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False),
    }
    eval_starts = {
        "train_fit": 0,
        "train_holdout": 0,
        "val": int(meta["val_eval_start"]),
        "test": int(meta["test_eval_start"]),
    }
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    return loaders, eval_starts, train_loader, meta


def _restore_cluster_penalty_prior(
    *,
    gate,
    cfg: Dict[str, object],
    moe_cfg: Dict[str, object],
    train_loader: DataLoader,
    penalty_names: List[str],
    penalty_fns: Dict[str, object],
    penalty_scale: torch.Tensor,
    cluster_id_c: torch.Tensor,
    K: int,
    H: int,
    device: torch.device,
) -> Dict[str, object]:
    prior_cfg = moe_cfg.get("cluster_penalty_prior", {}) or {}
    if not bool(prior_cfg.get("enable", False)) or len(penalty_names) == 0:
        return {
            "enable": False,
            "prior_prob": None,
            "allowed_mask": None,
            "logit_strength": 0.0,
            "source": "disabled",
        }
    portrait = compute_cluster_penalty_portrait(
        train_loader,
        penalty_names,
        penalty_fns,
        cluster_id_c.to(device=device),
        int(K),
        int(H),
        device,
    )
    prior = build_gate_prior_from_penalty_portrait(
        penalty_kp=portrait,
        penalty_scale=penalty_scale,
        temperature=float(prior_cfg.get("temperature", 1.0)),
        smoothing=float(prior_cfg.get("smoothing", 0.0)),
        use_normalized_penalty=bool(prior_cfg.get("use_normalized_penalty", True)),
    )
    logit_strength = float(prior_cfg.get("logit_strength", 0.0))
    if prior is not None and logit_strength > 0.0:
        gate.set_penalty_prior(prior, strength=logit_strength)
    allowed_mask = build_named_penalty_mask(
        prior_cfg.get("allowed_by_cluster", None),
        penalty_names,
        int(K),
        device,
        allow_empty_clusters=bool(prior_cfg.get("allow_empty_clusters", False)),
    )
    topk = int(prior_cfg.get("topk", 0))
    if allowed_mask is None and prior is not None and topk > 0 and bool(prior_cfg.get("hard_topk", True)):
        allowed_mask = build_topk_penalty_mask(prior, topk=topk)
    always_include = prior_cfg.get("always_include", []) or []
    if isinstance(always_include, str):
        always_include = [always_include]
    if len(always_include) > 0:
        if allowed_mask is None:
            allowed_mask = torch.zeros((int(K), len(penalty_names)), device=device, dtype=torch.float32)
        name_to_idx = {str(name): i for i, name in enumerate(penalty_names)}
        for raw_name in always_include:
            name = str(raw_name)
            if name not in name_to_idx:
                raise ValueError(f"unknown cluster_penalty_prior.always_include penalty {name!r}")
            allowed_mask[:, name_to_idx[name]] = 1.0
        empty = allowed_mask.sum(dim=-1, keepdim=True) <= 0.0
        if bool(empty.any().item()):
            allowed_mask = torch.where(empty, torch.ones_like(allowed_mask), allowed_mask)
    if allowed_mask is not None:
        gate.set_penalty_allowed_mask(allowed_mask)
    return {
        "enable": True,
        "source": "train_only_recomputed_from_config",
        "topk": int(topk),
        "hard_topk": bool(prior_cfg.get("hard_topk", True)),
        "logit_strength": float(logit_strength),
        "portrait": None if portrait is None else portrait.detach().cpu().tolist(),
        "prior_prob": None if prior is None else prior.detach().cpu().tolist(),
        "allowed_mask": None if allowed_mask is None else allowed_mask.detach().cpu().to(dtype=torch.bool).tolist(),
    }


def _build_anchor_artifacts(
    *,
    cfg: Dict[str, object],
    checkpoint: Dict[str, object],
    model,
    cluster_id_c: torch.Tensor,
    data_tc: torch.Tensor,
    train_loader: DataLoader,
    window_meta: Dict[str, int],
    device: torch.device,
) -> Dict[str, object]:
    model_cfg = dict(checkpoint["meta"].get("model_cfg", cfg.get("model", {}) or {}))
    moe_cfg = dict(checkpoint["meta"].get("moe_cfg", cfg.get("moe", {}) or {}))
    history_anchor_cfg = _normalize_history_anchor_cfg(model_cfg.get("history_anchor", cfg.get("history_anchor", {}) or {}))
    _validate_strict_history_anchor_scope(history_anchor_cfg, source="next11c_route_accuracy.history_anchor")
    model_train_stat_adapter_cfg = model_cfg.get("train_stat_adapter", {}) or {}
    model_train_stat_adapter_pc, _, model_train_stat_summary = build_train_stat_anchor_from_config(
        data_tc,
        train_end=int(window_meta["t_train"]),
        input_len=int(window_meta["L"]),
        pred_len=int(window_meta["H"]),
        cfg=model_train_stat_adapter_cfg,
        prefix="next11c_route_accuracy.model.train_stat_adapter",
    )
    train_stat_anchor_cfg = moe_cfg.get("train_stat_anchor_expert", {}) or {}
    train_stat_anchor_pc, _, train_stat_summary = build_train_stat_anchor_from_config(
        data_tc,
        train_end=int(window_meta["t_train"]),
        input_len=int(window_meta["L"]),
        pred_len=int(window_meta["H"]),
        cfg=train_stat_anchor_cfg,
        prefix="next11c_route_accuracy.moe.train_stat_anchor_expert",
    )
    train_residual_anchor_phc = None
    train_residual_summary: Dict[str, object] = {"enable": bool((moe_cfg.get("train_residual_anchor_expert", {}) or {}).get("enable", False))}
    train_residual_anchor_cfg = moe_cfg.get("train_residual_anchor_expert", {}) or {}
    if bool(train_residual_anchor_cfg.get("enable", False)):
        train_residual_anchor_phc, counts, n_windows = build_train_residual_anchor_table_from_loader(
            model=model,
            loader=train_loader,
            cluster_id_c=cluster_id_c,
            device=device,
            history_anchor_cfg=history_anchor_cfg,
            observed_history_tc=data_tc,
            input_len=int(window_meta["L"]),
            eval_start=0,
            period=int(train_residual_anchor_cfg.get("period", 96)),
            model_train_stat_adapter_pc=model_train_stat_adapter_pc,
            model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
            train_stat_anchor_pc=train_stat_anchor_pc,
            train_stat_anchor_cfg=train_stat_anchor_cfg,
        )
        train_residual_summary.update(
            {
                "period": int(train_residual_anchor_cfg.get("period", 96)),
                "windows": int(n_windows),
                "nonempty_phases": int((counts > 0).sum().item()),
                "alpha_by_channel_horizon_present": "alpha_by_channel_horizon" in train_residual_anchor_cfg,
            }
        )
    return {
        "history_anchor_cfg": history_anchor_cfg,
        "model_train_stat_adapter_cfg": model_train_stat_adapter_cfg,
        "model_train_stat_adapter_pc": model_train_stat_adapter_pc,
        "train_stat_anchor_cfg": train_stat_anchor_cfg,
        "train_stat_anchor_pc": train_stat_anchor_pc,
        "train_residual_anchor_phc": train_residual_anchor_phc,
        "summary": {
            "model_train_stat_adapter": model_train_stat_summary,
            "train_stat_anchor": train_stat_summary,
            "train_residual_anchor": train_residual_summary,
        },
    }


def _cluster_ids_for_tensors(labels: torch.Tensor, K: int) -> torch.Tensor:
    n = int(labels.numel())
    if int(K) <= 0 or n % int(K) != 0:
        return torch.full((n,), -1, dtype=torch.long)
    return torch.arange(int(K), dtype=torch.long).repeat(n // int(K))


def _summaries_by_cluster(
    tensors: Dict[str, torch.Tensor],
    label_names: List[str],
    K: int,
    *,
    min_abs_improvement: float = 0.0,
) -> List[Dict[str, object]]:
    labels = tensors["labels"].detach().cpu().to(dtype=torch.long).view(-1)
    current = tensors["current_pred"].detach().cpu().to(dtype=torch.long).view(-1)
    gains = tensors.get("oracle_gain_mse")
    cluster_ids = _cluster_ids_for_tensors(labels, int(K))
    rows = []
    for k in range(int(K)):
        mask = cluster_ids == int(k)
        if int(mask.sum().item()) <= 0:
            continue
        row = route_accuracy_summary(
            labels=labels[mask],
            current_pred=current[mask],
            oracle_gain_mse=None if gains is None else gains.detach().cpu().view(-1)[mask],
            label_names=label_names,
            min_abs_improvement=float(min_abs_improvement),
        )
        row["cluster_id"] = int(k)
        rows.append(row)
    return rows


def _concat_tensors(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key in ("labels", "current_pred", "query_start_abs", "oracle_gain_mse"):
        if key in a and key in b:
            out[key] = torch.cat([a[key].detach().cpu(), b[key].detach().cpu()], dim=0)
    return out


def _write_split_rows_csv(
    path: Path,
    *,
    split: str,
    tensors: Dict[str, torch.Tensor],
    label_names: List[str],
    K: int,
) -> None:
    labels = tensors["labels"].detach().cpu().to(dtype=torch.long).view(-1)
    current = tensors["current_pred"].detach().cpu().to(dtype=torch.long).view(-1)
    starts = tensors.get("query_start_abs", torch.arange(int(labels.numel()))).detach().cpu().to(dtype=torch.long).view(-1)
    gains = tensors.get("oracle_gain_mse", torch.zeros_like(labels, dtype=torch.float32)).detach().cpu().to(dtype=torch.float32).view(-1)
    cluster_ids = _cluster_ids_for_tensors(labels, int(K))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "row",
                "query_start_abs",
                "cluster_id",
                "oracle_class",
                "oracle_label",
                "current_class",
                "current_label",
                "oracle_gain_mse",
                "correct",
            ],
        )
        writer.writeheader()
        for i in range(int(labels.numel())):
            y = int(labels[i].item())
            pred = int(current[i].item())
            writer.writerow(
                {
                    "split": split,
                    "row": i,
                    "query_start_abs": int(starts[i].item()),
                    "cluster_id": int(cluster_ids[i].item()),
                    "oracle_class": y,
                    "oracle_label": label_names[y] if 0 <= y < len(label_names) else "invalid",
                    "current_class": pred,
                    "current_label": label_names[pred] if 0 <= pred < len(label_names) else "invalid",
                    "oracle_gain_mse": float(gains[i].item()),
                    "correct": bool(y == pred),
                }
            )


def _flatten_summary_rows(cell: str, variant: str, split: str, summary: Dict[str, object]) -> Dict[str, object]:
    skip_prob = summary.get("skip_probability", {}) or {}
    row = {
        "cell": cell,
        "variant": variant,
        "split": split,
        "samples": summary.get("samples", 0),
        "current_accuracy_all": summary.get("current_accuracy_all", 0.0),
        "majority_accuracy_all": summary.get("majority_accuracy_all", 0.0),
        "lift_vs_majority": summary.get("lift_vs_majority", 0.0),
        "oracle_skip_rate": summary.get("oracle_skip_rate", 0.0),
        "actual_skip_rate": summary.get("actual_skip_rate", 0.0),
        "skip_recall": summary.get("skip_recall", 0.0),
        "skip_precision": summary.get("skip_precision", 0.0),
        "skip_false_positive_rate_on_oracle_penalty": summary.get("skip_false_positive_rate_on_oracle_penalty", 0.0),
        "penalty_accuracy_on_oracle_penalty": summary.get("penalty_accuracy_on_oracle_penalty", 0.0),
        "oracle_penalty_routed_to_skip_rate": summary.get("oracle_penalty_routed_to_skip_rate", 0.0),
        "oracle_penalty_routed_to_wrong_penalty_rate": summary.get("oracle_penalty_routed_to_wrong_penalty_rate", 0.0),
        "oracle_gain_mse_mean": summary.get("oracle_gain_mse_mean", None),
        "skip_prob_mean": skip_prob.get("mean", None),
        "skip_prob_p95": skip_prob.get("p95", None),
        "skip_prob_max": skip_prob.get("max", None),
        "skip_prob_gt_0_5_rate": skip_prob.get("gt_0_5_rate", None),
        "skip_prob_oracle_skip_mean": skip_prob.get("oracle_skip_mean", None),
        "skip_prob_oracle_penalty_mean": skip_prob.get("oracle_penalty_mean", None),
    }
    return row


def _read_run_summary(run_dir: Path) -> Dict[str, object]:
    path = run_dir / "run_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _topk_pool_composition(allowed_mask: object, penalty_names: List[str], K: int) -> List[Dict[str, object]]:
    if allowed_mask is None:
        return [
            {"cluster_id": int(k), "pool": list(penalty_names), "pool_size": int(len(penalty_names))}
            for k in range(int(K))
        ]
    allowed = torch.as_tensor(allowed_mask, dtype=torch.bool)
    if allowed.dim() != 2 or int(allowed.shape[0]) != int(K) or int(allowed.shape[1]) != len(penalty_names):
        raise ValueError("allowed_mask must have shape [K,P] for pool composition.")
    rows = []
    for k in range(int(K)):
        pool = [penalty_names[p] for p in range(len(penalty_names)) if bool(allowed[k, p].item())]
        rows.append({"cluster_id": int(k), "pool": pool, "pool_size": int(len(pool))})
    return rows


@torch.no_grad()
def _collect_topk_set_overlap_for_split(
    *,
    model,
    gate,
    pred_residual,
    loader: DataLoader,
    cluster_id_c: torch.Tensor,
    K: int,
    moe_cfg: Dict[str, object],
    device: torch.device,
    penalty_names: List[str],
    penalty_fns: Dict[str, object],
    penalty_scale: torch.Tensor,
    select_ranks: Optional[List[int]],
    gate_soft_weight: float,
    split_name: str,
    tau_grid: List[float],
    channel_names: List[str],
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
    cid_c = cluster_id_c.detach().to(device=device, dtype=torch.long)
    C = int(cid_c.numel())

    gain_parts: List[torch.Tensor] = []
    applied_parts: List[torch.Tensor] = []
    base_se_c = torch.zeros(C, dtype=torch.float64)
    base_ae_c = torch.zeros(C, dtype=torch.float64)
    oracle_se_c = torch.zeros(C, dtype=torch.float64)
    oracle_ae_c = torch.zeros(C, dtype=torch.float64)
    denom_c = torch.zeros(C, dtype=torch.float64)
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
        feat_bkf = _build_gate_routing_features(x, y_base, cluster_id_c, int(K), mode=gate_feature_mode)
        route_pen_bkp = _router_penalty_context_from_history(
            x_bcl=x,
            yhat_base_bch=y_base,
            penalty_names=penalty_names,
            penalty_fns=penalty_fns,
            penalty_scale=penalty_scale,
            cluster_id_c=cluster_id_c,
            K=int(K),
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
        base_mse_bc = (y_base_final - y).pow(2).mean(dim=-1)
        cand_mse_bcp = (cand_bcpH - y.unsqueeze(2)).pow(2).mean(dim=-1)
        gain_bcp = base_mse_bc.unsqueeze(-1) - cand_mse_bcp
        applied_bcp = mask_bkp[:, cid_c, :] > 0.0
        gain_parts.append(gain_bcp.detach().cpu())
        applied_parts.append(applied_bcp.detach().cpu())

        best_cand_mse_bc, best_p_bc = cand_mse_bcp.min(dim=-1)
        best_gain_bc = base_mse_bc - best_cand_mse_bc
        best_pred_bch = cand_bcpH.gather(
            dim=2,
            index=best_p_bc.view(best_p_bc.shape[0], best_p_bc.shape[1], 1, 1).expand(-1, -1, 1, int(cand_bcpH.shape[-1])),
        ).squeeze(2)
        oracle_pred_bch = torch.where((best_gain_bc > 0.0).unsqueeze(-1), best_pred_bch, y_base_final)
        base_se_c += (y_base_final - y).pow(2).sum(dim=(0, 2)).detach().cpu().to(dtype=torch.float64)
        base_ae_c += (y_base_final - y).abs().sum(dim=(0, 2)).detach().cpu().to(dtype=torch.float64)
        oracle_se_c += (oracle_pred_bch - y).pow(2).sum(dim=(0, 2)).detach().cpu().to(dtype=torch.float64)
        oracle_ae_c += (oracle_pred_bch - y).abs().sum(dim=(0, 2)).detach().cpu().to(dtype=torch.float64)
        denom_c += torch.full((C,), int(y.shape[0]) * int(y.shape[-1]), dtype=torch.float64)

    if not gain_parts:
        return None
    gain_all = torch.cat(gain_parts, dim=0)
    applied_all = torch.cat(applied_parts, dim=0)
    tau_summaries = [
        topk_set_overlap_summary(
            gain_bcp=gain_all,
            applied_bcp=applied_all,
            cluster_id_c=cluster_id_c.detach().cpu(),
            penalty_names=penalty_names,
            channel_names=channel_names,
            tau=float(tau),
        )
        for tau in tau_grid
    ]
    denom_safe_c = denom_c.clamp_min(1.0)
    base_mse_c = base_se_c / denom_safe_c
    base_mae_c = base_ae_c / denom_safe_c
    oracle_mse_c = oracle_se_c / denom_safe_c
    oracle_mae_c = oracle_ae_c / denom_safe_c
    total_denom = float(denom_c.sum().item())
    return {
        "split": str(split_name),
        "samples": int(gain_all.shape[0] * gain_all.shape[1]),
        "tau_grid": [float(tau) for tau in tau_grid],
        "tau_summaries": tau_summaries,
        "channel_oracle": {
            "base_mse": _safe_div(float(base_se_c.sum().item()), total_denom),
            "base_mae": _safe_div(float(base_ae_c.sum().item()), total_denom),
            "oracle_mse": _safe_div(float(oracle_se_c.sum().item()), total_denom),
            "oracle_mae": _safe_div(float(oracle_ae_c.sum().item()), total_denom),
            "oracle_gain_pct_vs_base": 100.0
            * _safe_div(float(base_se_c.sum().item() - oracle_se_c.sum().item()), float(base_se_c.sum().item())),
            "oracle_mae_gain_pct_vs_base": 100.0
            * _safe_div(float(base_ae_c.sum().item() - oracle_ae_c.sum().item()), float(base_ae_c.sum().item())),
            "base_mse_per_channel": [float(v) for v in base_mse_c.tolist()],
            "base_mae_per_channel": [float(v) for v in base_mae_c.tolist()],
            "oracle_mse_per_channel": [float(v) for v in oracle_mse_c.tolist()],
            "oracle_mae_per_channel": [float(v) for v in oracle_mae_c.tolist()],
            "channel_names": list(channel_names),
        },
    }


def _run_one(
    *,
    cell: str,
    variant: str,
    config_path: Path,
    checkpoint_path: Path,
    run_dir: Path,
    out_dir: Path,
    device_arg: Optional[str],
    max_batches: int,
    requested_splits: Optional[Iterable[str]] = None,
    threshold_source: str = "manual",
    min_abs_improvement: float = 0.0,
    min_rel_improvement: float = 0.0,
    min_candidate_delta_rms: float = 0.0,
    topk_set_overlap: bool = False,
    topk_tau_grid: Optional[List[float]] = None,
) -> Dict[str, object]:
    cfg = load_yaml(str(config_path))
    route_label_thresholds = _route_label_thresholds_from_config(
        cfg,
        source=str(threshold_source),
        min_abs_improvement=float(min_abs_improvement),
        min_rel_improvement=float(min_rel_improvement),
        min_candidate_delta_rms=float(min_candidate_delta_rms),
    )
    route_min_abs = float(route_label_thresholds["min_abs_improvement"])
    route_min_rel = float(route_label_thresholds["min_rel_improvement"])
    route_min_delta = float(route_label_thresholds["min_candidate_delta_rms"])
    set_seed(int(cfg["exp"]["seed"]), deterministic=bool((cfg.get("exp", {}) or {}).get("deterministic", False)))
    requested_device = str(device_arg or cfg.get("exp", {}).get("device", "cpu"))
    device = torch.device(requested_device if torch.cuda.is_available() and requested_device != "cpu" else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model, gate, pred_residual, cluster_id_c, K, moe_cfg, penalty_names = _build_modules(cfg, checkpoint, device)
    data_tc, data_info = _read_data_for_cfg(cfg)
    channel_names = list(data_info.get("channel_names", []) or [])
    batch_size = int(cfg.get("train", {}).get("batch_size", 64))
    loaders, eval_starts, train_loader, window_meta = _make_loaders(cfg, data_tc, batch_size=batch_size)
    penalty_fns = build_penalty_bank(penalty_names, jump_thr=float(cfg.get("penalties", {}).get("jump_threshold", 0.6)))
    penalty_scale = _compute_penalty_scale(train_loader, penalty_names, penalty_fns, int(window_meta["H"]), device)
    anchor = _build_anchor_artifacts(
        cfg=cfg,
        checkpoint=checkpoint,
        model=model,
        cluster_id_c=cluster_id_c,
        data_tc=data_tc,
        train_loader=train_loader,
        window_meta=window_meta,
        device=device,
    )
    prior_summary = _restore_cluster_penalty_prior(
        gate=gate,
        cfg=cfg,
        moe_cfg=moe_cfg,
        train_loader=train_loader,
        penalty_names=penalty_names,
        penalty_fns=penalty_fns,
        penalty_scale=penalty_scale,
        cluster_id_c=cluster_id_c,
        K=int(K),
        H=int(window_meta["H"]),
        device=device,
    )
    raw_ranks = moe_cfg.get("select_ranks", None)
    select_ranks = None if raw_ranks is None else [int(v) for v in raw_ranks]
    if raw_ranks is None:
        select_ranks = [1, 2]
    gate_soft_weight = float(moe_cfg.get("gate_soft_weight", 0.0))
    gate_feature_mode = _normalize_gate_feature_mode(str(checkpoint["meta"].get("gate_feature_mode", moe_cfg.get("gate_feature_mode", "history"))))
    allowed_mask = prior_summary.get("allowed_mask")
    allowed_mask_kp = None
    if allowed_mask is not None:
        allowed_mask_kp = torch.as_tensor(allowed_mask, device=device, dtype=torch.bool)

    case_dir = out_dir / cell / variant
    case_dir.mkdir(parents=True, exist_ok=True)
    split_tensors: Dict[str, Dict[str, torch.Tensor]] = {}
    split_summaries: Dict[str, object] = {}
    per_cluster: Dict[str, object] = {}
    topk_diagnostics: Dict[str, object] = {}
    splits_to_run = _normalize_requested_splits(requested_splits)
    for split in splits_to_run:
        if bool(topk_set_overlap):
            diag = _collect_topk_set_overlap_for_split(
                model=model,
                gate=gate,
                pred_residual=pred_residual,
                loader=loaders[split],
                cluster_id_c=cluster_id_c,
                K=int(K),
                moe_cfg=moe_cfg,
                device=device,
                penalty_names=penalty_names,
                penalty_fns=penalty_fns,
                penalty_scale=penalty_scale,
                select_ranks=select_ranks,
                gate_soft_weight=gate_soft_weight,
                split_name=split,
                tau_grid=[float(v) for v in (topk_tau_grid or [0.0])],
                channel_names=channel_names,
                max_batches=int(max_batches),
                history_anchor_cfg=anchor["history_anchor_cfg"],
                observed_history_tc=data_tc,
                input_len=int(window_meta["L"]),
                eval_start=int(eval_starts[split]),
                model_train_stat_adapter_pc=anchor["model_train_stat_adapter_pc"],
                model_train_stat_adapter_cfg=anchor["model_train_stat_adapter_cfg"],
                train_stat_anchor_pc=anchor["train_stat_anchor_pc"],
                train_residual_anchor_phc=anchor["train_residual_anchor_phc"],
                gate_feature_mode=gate_feature_mode,
            )
            if diag is not None:
                topk_diagnostics[split] = diag
            continue
        tensors = _collect_penalty_route_learnability_tensors(
            model=model,
            gate=gate,
            pred_residual=pred_residual,
            loader=loaders[split],
            cluster_id_c=cluster_id_c,
            K=int(K),
            moe_cfg=moe_cfg,
            device=device,
            penalty_names=penalty_names,
            penalty_fns=penalty_fns,
            penalty_scale=penalty_scale,
            select_ranks=select_ranks,
            gate_soft_weight=gate_soft_weight,
            split_name=split,
            feature_mode=str(((moe_cfg.get("explainability", {}) or {}).get("route_learnability", {}) or {}).get("feature_mode", "base")),
            allowed_mask_kp=allowed_mask_kp,
            min_abs_improvement=route_min_abs,
            min_rel_improvement=route_min_rel,
            min_candidate_delta_rms=route_min_delta,
            max_batches=int(max_batches),
            history_anchor_cfg=anchor["history_anchor_cfg"],
            observed_history_tc=data_tc,
            input_len=int(window_meta["L"]),
            eval_start=int(eval_starts[split]),
            model_train_stat_adapter_pc=anchor["model_train_stat_adapter_pc"],
            model_train_stat_adapter_cfg=anchor["model_train_stat_adapter_cfg"],
            train_stat_anchor_pc=anchor["train_stat_anchor_pc"],
            train_residual_anchor_phc=anchor["train_residual_anchor_phc"],
            gate_feature_mode=gate_feature_mode,
        )
        if tensors is None:
            continue
        split_tensors[split] = tensors
        labels = tensors["labels"]
        label_names = list(tensors.get("label_names", ["skip"] + penalty_names))
        split_summary = route_accuracy_summary(
            labels=labels,
            current_pred=tensors["current_pred"],
            oracle_gain_mse=tensors.get("oracle_gain_mse"),
            label_names=label_names,
            min_abs_improvement=route_min_abs,
        )
        split_summary["skip_probability"] = skip_probability_summary(tensors=tensors)
        split_summaries[split] = split_summary
        per_cluster[split] = _summaries_by_cluster(
            tensors,
            label_names,
            int(K),
            min_abs_improvement=route_min_abs,
        )
        torch.save(
            {
                "split": split,
                "cell": cell,
                "variant": variant,
                "label_names": label_names,
                "penalty_names": penalty_names,
                "cluster_count": int(K),
                "tensors": tensors,
            },
            case_dir / f"route_tensors_{split}.pt",
        )
        _write_split_rows_csv(
            case_dir / f"route_rows_{split}.csv",
            split=split,
            tensors=tensors,
            label_names=label_names,
            K=int(K),
        )

    if (not bool(topk_set_overlap)) and "train_fit" in split_tensors and "train_holdout" in split_tensors:
        train_tensors = _concat_tensors(split_tensors["train_fit"], split_tensors["train_holdout"])
        label_names = list(split_tensors["train_fit"].get("label_names", ["skip"] + penalty_names))
        split_summaries["train"] = route_accuracy_summary(
            labels=train_tensors["labels"],
            current_pred=train_tensors["current_pred"],
            oracle_gain_mse=train_tensors.get("oracle_gain_mse"),
            label_names=label_names,
            min_abs_improvement=route_min_abs,
        )
        fit_skip = (split_summaries.get("train_fit", {}) or {}).get("skip_probability", {}) or {}
        holdout_skip = (split_summaries.get("train_holdout", {}) or {}).get("skip_probability", {}) or {}
        if fit_skip.get("available") and holdout_skip.get("available"):
            fit_n = int((split_summaries.get("train_fit", {}) or {}).get("samples", 0))
            holdout_n = int((split_summaries.get("train_holdout", {}) or {}).get("samples", 0))
            total_n = max(fit_n + holdout_n, 1)
            split_summaries["train"]["skip_probability"] = {
                "available": True,
                "mean": (float(fit_skip["mean"]) * fit_n + float(holdout_skip["mean"]) * holdout_n) / total_n,
                "p50": None,
                "p95": None,
                "max": max(float(fit_skip["max"]), float(holdout_skip["max"])),
                "gt_0_5_rate": (
                    float(fit_skip["gt_0_5_rate"]) * fit_n + float(holdout_skip["gt_0_5_rate"]) * holdout_n
                )
                / total_n,
                "oracle_skip_mean": None,
                "oracle_penalty_mean": None,
                "note": "train aggregate mean/rate are sample-weighted from train_fit and train_holdout; quantiles are not recomputed.",
            }
        per_cluster["train"] = _summaries_by_cluster(
            train_tensors,
            label_names,
            int(K),
            min_abs_improvement=route_min_abs,
        )

    run_summary = _read_run_summary(run_dir)
    payload = {
        "cell": cell,
        "variant": variant,
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "run_dir": str(run_dir),
        "device": str(device),
        "window_meta": window_meta,
        "eval_metrics_from_existing_run_summary": run_summary,
        "route_context": {
            "label_names": list(["skip"] + penalty_names),
            "penalty_names": list(penalty_names),
            "cluster_count": int(K),
            "requested_splits": list(splits_to_run),
            "allow_skip": bool(moe_cfg.get("allow_skip", False)),
            "skip_competes_with_penalties": bool(moe_cfg.get("skip_competes_with_penalties", moe_cfg.get("noop_compete_enable", False))),
            "skip_supervision_weight": float(moe_cfg.get("skip_supervision_weight", 0.0) or 0.0),
            "select_ranks": select_ranks,
            "gate_soft_weight": float(gate_soft_weight),
            "gate_feature_mode": gate_feature_mode,
            "allowed_mask_from_prior": prior_summary.get("allowed_mask"),
            "route_label_thresholds": {
                "min_abs_improvement": route_min_abs,
                "min_rel_improvement": route_min_rel,
                "min_candidate_delta_rms": route_min_delta,
                "source": str(route_label_thresholds.get("source", threshold_source)),
            },
            "prior_restored": prior_summary,
            "topk_pool_composition": _topk_pool_composition(prior_summary.get("allowed_mask"), penalty_names, int(K)),
            "anchors": anchor["summary"],
            "note": "current route uses skip only when the configured skip head returns skip_bk > 0.5; skip is not in top-k competition unless skip_competes_with_penalties is true.",
        },
        "split_summaries": split_summaries,
        "per_cluster": per_cluster,
        "topk_set_overlap": {
            "enable": bool(topk_set_overlap),
            "note": "Oracle-positive set is candidate MSE gain > tau per sample/channel/penalty; applied set is the gate top-k penalty pool before any per-sample skip zeroing.",
            "diagnostics": topk_diagnostics,
        },
    }
    (case_dir / "route_accuracy_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _write_summary_csv(path: Path, payloads: Iterable[Dict[str, object]]) -> None:
    rows = []
    for payload in payloads:
        cell = str(payload["cell"])
        variant = str(payload["variant"])
        for split, summary in (payload.get("split_summaries", {}) or {}).items():
            rows.append(_flatten_summary_rows(cell, variant, str(split), summary))
    fieldnames = [
        "cell",
        "variant",
        "split",
        "samples",
        "current_accuracy_all",
        "majority_accuracy_all",
        "lift_vs_majority",
        "oracle_skip_rate",
        "actual_skip_rate",
        "skip_recall",
        "skip_precision",
        "skip_false_positive_rate_on_oracle_penalty",
        "penalty_accuracy_on_oracle_penalty",
        "oracle_penalty_routed_to_skip_rate",
        "oracle_penalty_routed_to_wrong_penalty_rate",
        "oracle_gain_mse_mean",
        "skip_prob_mean",
        "skip_prob_p95",
        "skip_prob_max",
        "skip_prob_gt_0_5_rate",
        "skip_prob_oracle_skip_mean",
        "skip_prob_oracle_penalty_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pct(v: object) -> str:
    try:
        return f"{100.0 * float(v):.2f}%"
    except Exception:
        return "n/a"


def _write_markdown(path: Path, payloads: List[Dict[str, object]]) -> None:
    lines = [
        "# NEXT-11c route accuracy diagnostic",
        "",
        "Diagnostic only: test labels are used here to explain the already frozen test-once runs, not to select or tune a new config.",
        "Oracle label class 0 is skip/no-op; classes 1..P are the configured penalties.",
        "",
        "## Key split summary",
        "",
        "| cell | variant | split | acc | majority | oracle skip | actual skip | skip prob mean/p95 | skip recall | penalty acc on oracle penalty | wrong penalty | oracle penalty -> skip |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    split_order = ["train", "train_fit", "train_holdout", "val", "test"]
    for payload in payloads:
        for split in split_order:
            summary = (payload.get("split_summaries", {}) or {}).get(split)
            if not summary:
                continue
            skip_prob = summary.get("skip_probability", {}) or {}
            if skip_prob.get("available") and skip_prob.get("p95") is not None:
                skip_prob_text = f"{_pct(skip_prob.get('mean'))}/{_pct(skip_prob.get('p95'))}"
            elif skip_prob.get("available"):
                skip_prob_text = _pct(skip_prob.get("mean"))
            else:
                skip_prob_text = "n/a"
            lines.append(
                "| {cell} | {variant} | {split} | {acc} | {maj} | {osk} | {ask} | {skipprob} | {sr} | {pacc} | {wrong} | {pskip} |".format(
                    cell=payload["cell"],
                    variant=payload["variant"],
                    split=split,
                    acc=_pct(summary.get("current_accuracy_all")),
                    maj=_pct(summary.get("majority_accuracy_all")),
                    osk=_pct(summary.get("oracle_skip_rate")),
                    ask=_pct(summary.get("actual_skip_rate")),
                    skipprob=skip_prob_text,
                    sr=_pct(summary.get("skip_recall")),
                    pacc=_pct(summary.get("penalty_accuracy_on_oracle_penalty")),
                    wrong=_pct(summary.get("oracle_penalty_routed_to_wrong_penalty_rate")),
                    pskip=_pct(summary.get("oracle_penalty_routed_to_skip_rate")),
                )
            )
    lines.extend(["", "## Route context", ""])
    for payload in payloads:
        ctx = payload.get("route_context", {}) or {}
        lines.append(
            f"- {payload['cell']} {payload['variant']}: penalties={ctx.get('penalty_names')}, "
            f"allow_skip={ctx.get('allow_skip')}, skip_competes_with_penalties={ctx.get('skip_competes_with_penalties')}, "
            f"skip_supervision_weight={ctx.get('skip_supervision_weight')}, allowed_mask={ctx.get('allowed_mask_from_prior')}"
        )
    lines.extend(["", "## Artifacts", ""])
    lines.append(f"- CSV summary: `{path.parent / 'route_accuracy_summary.csv'}`")
    lines.append("- Per-case JSON: `<out>/<cell>/<variant>/route_accuracy_summary.json`")
    lines.append("- Per-split rows: `<out>/<cell>/<variant>/route_rows_<split>.csv`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_float(value: object, digits: int = 6) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "n/a"


def _pct_text(value: object) -> str:
    try:
        return f"{100.0 * float(value):.2f}%"
    except Exception:
        return "n/a"


def _real_path_metrics(run_summary: Dict[str, object], channel_oracle: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    residual = run_summary.get("moe_residual_selection", {}) or {}
    return {
        "base": {
            "mse": residual.get("val_pred_base_avg_mse"),
            "mae": residual.get("val_pred_base_avg_mae"),
        },
        "raw_residual": {
            "mse": residual.get("val_residual_avg_mse", (run_summary.get("val", {}) or {}).get("avg_mse")),
            "mae": residual.get("val_residual_avg_mae", (run_summary.get("val", {}) or {}).get("avg_mae")),
        },
        "selected_scaled": {
            "mse": residual.get("val_scaled_avg_mse"),
            "mae": residual.get("val_scaled_avg_mae"),
        },
        "channel_oracle": {
            "mse": channel_oracle.get("oracle_mse"),
            "mae": channel_oracle.get("oracle_mae"),
        },
    }


def _write_topk_markdown(path: Path, payloads: List[Dict[str, object]]) -> None:
    lines = [
        "# NEXT-11 ETTh2-H96 Top-k Set-Overlap Report",
        "",
        "Diagnostic-only, caliber-aligned path: oracle-positive set is per-sample/channel penalty gain > tau; applied set is the gate top-k penalty pool before per-sample skip zeroing.",
        "",
        "## Real Adoption Path",
        "",
        "| cell | variant | path | val MSE | val MAE |",
        "|---|---|---:|---:|---:|",
    ]
    for payload in payloads:
        val_diag = (((payload.get("topk_set_overlap", {}) or {}).get("diagnostics", {}) or {}).get("val", {}) or {})
        metrics = _real_path_metrics(
            payload.get("eval_metrics_from_existing_run_summary", {}) or {},
            val_diag.get("channel_oracle", {}) or {},
        )
        for name in ["base", "raw_residual", "selected_scaled", "channel_oracle"]:
            row = metrics[name]
            lines.append(
                f"| {payload['cell']} | {payload['variant']} | {name} | {_fmt_float(row.get('mse'))} | {_fmt_float(row.get('mae'))} |"
            )

    lines.extend(
        [
            "",
            "## Tau Sweep",
            "",
            "| cell | variant | split | tau | applied precision | applied recall | majority precision | majority recall | oracle set size |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for payload in payloads:
        diagnostics = ((payload.get("topk_set_overlap", {}) or {}).get("diagnostics", {}) or {})
        for split, split_diag in diagnostics.items():
            for summary in split_diag.get("tau_summaries", []) or []:
                overall = summary.get("overall", {}) or {}
                majority = summary.get("majority_overall", {}) or {}
                lines.append(
                    f"| {payload['cell']} | {payload['variant']} | {split} | {_fmt_float(summary.get('tau'), 6)} | "
                    f"{_pct_text(overall.get('precision'))} | {_pct_text(overall.get('recall'))} | "
                    f"{_pct_text(majority.get('precision'))} | {_pct_text(majority.get('recall'))} | "
                    f"{_fmt_float(overall.get('mean_oracle_positive_set_size'), 3)} |"
                )

    lines.extend(
        [
            "",
            "## Val Per-cluster",
            "",
            "| cell | variant | tau | cluster | pool | precision | recall | majority set | majority precision | majority recall |",
            "|---|---|---:|---:|---|---:|---:|---|---:|---:|",
        ]
    )
    for payload in payloads:
        pool_rows = {
            int(row["cluster_id"]): row.get("pool", [])
            for row in (((payload.get("route_context", {}) or {}).get("topk_pool_composition", []) or []))
        }
        val_diag = (((payload.get("topk_set_overlap", {}) or {}).get("diagnostics", {}) or {}).get("val", {}) or {})
        summaries = val_diag.get("tau_summaries", []) or []
        if not summaries:
            continue
        summary = summaries[0]
        majority_by_cluster = {
            int(row["cluster_id"]): row for row in (summary.get("majority_per_cluster", []) or [])
        }
        for row in summary.get("per_cluster", []) or []:
            cid = int(row["cluster_id"])
            maj = majority_by_cluster.get(cid, {})
            lines.append(
                f"| {payload['cell']} | {payload['variant']} | {_fmt_float(summary.get('tau'), 6)} | {cid} | "
                f"{', '.join(pool_rows.get(cid, []))} | {_pct_text(row.get('precision'))} | {_pct_text(row.get('recall'))} | "
                f"{', '.join(maj.get('majority_set', []) or [])} | {_pct_text(maj.get('precision'))} | {_pct_text(maj.get('recall'))} |"
            )

    lines.extend(
        [
            "",
            "## Val Per-channel",
            "",
            "| cell | variant | tau | channel | cluster | precision | recall | majority set | majority precision | majority recall |",
            "|---|---|---:|---|---:|---:|---:|---|---:|---:|",
        ]
    )
    for payload in payloads:
        val_diag = (((payload.get("topk_set_overlap", {}) or {}).get("diagnostics", {}) or {}).get("val", {}) or {})
        summaries = val_diag.get("tau_summaries", []) or []
        if not summaries:
            continue
        summary = summaries[0]
        majority_by_channel = {
            int(row["channel_index"]): row for row in (summary.get("majority_per_channel", []) or [])
        }
        for row in summary.get("per_channel", []) or []:
            cidx = int(row["channel_index"])
            maj = majority_by_channel.get(cidx, {})
            lines.append(
                f"| {payload['cell']} | {payload['variant']} | {_fmt_float(summary.get('tau'), 6)} | {row.get('channel')} | "
                f"{row.get('cluster_id')} | {_pct_text(row.get('precision'))} | {_pct_text(row.get('recall'))} | "
                f"{', '.join(maj.get('majority_set', []) or [])} | {_pct_text(maj.get('precision'))} | {_pct_text(maj.get('recall'))} |"
            )

    lines.extend(["", "## Artifacts", ""])
    lines.append(f"- JSON summary: `{path.parent / 'topk_set_overlap_summary.json'}`")
    lines.append(f"- Markdown report: `{path}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze NEXT-11c train/val/test route accuracy with skip as class 0.")
    parser.add_argument(
        "--runs-root",
        default="outputs/next11c_fair_stage2_audit/fair_test_once",
        help="Root containing configs/<cell>/<variant>.yaml and runs/<cell>/<variant>/best_checkpoint.pt.",
    )
    parser.add_argument("--cells", nargs="+", default=["ETTm2_H96", "ETTh1_H96"])
    parser.add_argument("--variants", nargs="+", default=["d_moe_only_no_anchors", "c_full"])
    parser.add_argument("--out-dir", default="outputs/next11c_fair_stage2_audit/route_accuracy_diagnostic")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--threshold-source", choices=["manual", "stage2"], default="manual")
    parser.add_argument("--min-abs-improvement", type=float, default=0.0)
    parser.add_argument("--min-rel-improvement", type=float, default=0.0)
    parser.add_argument("--min-candidate-delta-rms", type=float, default=0.0)
    parser.add_argument(
        "--topk-set-overlap",
        action="store_true",
        default=False,
        help="Default-off NEXT-11 caliber-aligned top-k set overlap diagnostic.",
    )
    parser.add_argument(
        "--topk-set-overlap-tau-grid",
        nargs="+",
        type=float,
        default=[0.0, 1.0e-5, 1.0e-4, 5.0e-4, 1.0e-3],
        help="Small val tau grid for oracle-positive set diagnostics; used only with --topk-set-overlap.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payloads: List[Dict[str, object]] = []
    for cell in args.cells:
        for variant in args.variants:
            config_path = runs_root / "configs" / cell / f"{variant}.yaml"
            checkpoint_path = runs_root / "runs" / cell / variant / "best_checkpoint.pt"
            run_dir = runs_root / "runs" / cell / variant
            if not config_path.exists():
                raise FileNotFoundError(config_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(checkpoint_path)
            print(f"[route-accuracy] {cell} {variant}")
            payloads.append(
                _run_one(
                    cell=str(cell),
                    variant=str(variant),
                    config_path=config_path,
                    checkpoint_path=checkpoint_path,
                    run_dir=run_dir,
                    out_dir=out_dir,
                    device_arg=args.device,
                    max_batches=int(args.max_batches),
                    requested_splits=args.splits,
                    threshold_source=str(args.threshold_source),
                    min_abs_improvement=float(args.min_abs_improvement),
                    min_rel_improvement=float(args.min_rel_improvement),
                    min_candidate_delta_rms=float(args.min_candidate_delta_rms),
                    topk_set_overlap=bool(args.topk_set_overlap),
                    topk_tau_grid=[float(v) for v in args.topk_set_overlap_tau_grid],
                )
            )
    if bool(args.topk_set_overlap):
        summary_path = out_dir / "topk_set_overlap_summary.json"
        summary_path.write_text(json.dumps({"runs_root": str(runs_root), "payloads": payloads}, indent=2), encoding="utf-8")
        _write_topk_markdown(out_dir / "topk_set_overlap_report.md", payloads)
        print(f"[topk-set-overlap] wrote {summary_path}")
        print(f"[topk-set-overlap] wrote {out_dir / 'topk_set_overlap_report.md'}")
    else:
        summary_path = out_dir / "route_accuracy_summary.json"
        summary_path.write_text(json.dumps({"runs_root": str(runs_root), "payloads": payloads}, indent=2), encoding="utf-8")
        _write_summary_csv(out_dir / "route_accuracy_summary.csv", payloads)
        _write_markdown(out_dir / "route_accuracy_report.md", payloads)
        print(f"[route-accuracy] wrote {summary_path}")
        print(f"[route-accuracy] wrote {out_dir / 'route_accuracy_report.md'}")


if __name__ == "__main__":
    main()
