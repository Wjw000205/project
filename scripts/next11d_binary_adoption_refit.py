from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _to_jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _safe_rate(num: int, denom: int) -> float:
    return float(num) / max(int(denom), 1)


def _binary_examples_for_penalty(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    penalty_class: int,
    ignore_other_positive: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if features.dim() != 3:
        raise ValueError("features must have shape [N,num_classes,F].")
    labels = labels.detach().cpu().to(dtype=torch.long).view(-1)
    if int(features.shape[0]) != int(labels.numel()):
        raise ValueError("features and labels must share N.")
    penalty_class = int(penalty_class)
    if penalty_class <= 0 or penalty_class >= int(features.shape[1]):
        raise ValueError("penalty_class must be in [1, num_classes).")
    if bool(ignore_other_positive):
        mask = (labels == 0) | (labels == penalty_class)
    else:
        mask = labels >= 0
    indices = torch.nonzero(mask, as_tuple=False).view(-1)
    x = features.detach().cpu().to(dtype=torch.float32).index_select(0, indices)[:, penalty_class, :]
    y = (labels.index_select(0, indices) == penalty_class).to(dtype=torch.float32)
    return x, y, indices


def _route_from_binary_scores(scores: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    scores = torch.as_tensor(scores, dtype=torch.float32)
    thresholds = torch.as_tensor(thresholds, dtype=torch.float32).view(-1)
    if scores.dim() != 2:
        raise ValueError("scores must have shape [N,P].")
    if int(scores.shape[1]) != int(thresholds.numel()):
        raise ValueError("thresholds must have one value per penalty score.")
    if int(scores.shape[1]) == 0:
        return torch.zeros(int(scores.shape[0]), dtype=torch.long)
    margins = scores - thresholds.view(1, -1)
    best_margin, best_idx = margins.max(dim=1)
    pred = best_idx.to(dtype=torch.long) + 1
    return torch.where(best_margin >= 0.0, pred, torch.zeros_like(pred))


def _cluster_route_predictions_to_forecast(
    *,
    base_bch: torch.Tensor,
    cand_bcpH: torch.Tensor,
    cluster_id_c: torch.Tensor,
    route_pred_bk: torch.Tensor,
) -> torch.Tensor:
    if base_bch.dim() != 3:
        raise ValueError("base_bch must have shape [B,C,H].")
    if cand_bcpH.dim() != 4:
        raise ValueError("cand_bcpH must have shape [B,C,P,H].")
    B, C, H = [int(v) for v in base_bch.shape]
    if tuple(cand_bcpH.shape[:2]) != (B, C) or int(cand_bcpH.shape[-1]) != H:
        raise ValueError("cand_bcpH must share [B,C,H] with base_bch.")
    P = int(cand_bcpH.shape[2])
    cluster_id = cluster_id_c.detach().cpu().to(dtype=torch.long).view(-1)
    if int(cluster_id.numel()) != C:
        raise ValueError("cluster_id_c must have one entry per channel.")
    route = route_pred_bk.detach().cpu().to(dtype=torch.long)
    if route.dim() == 1:
        K = int(route.numel()) // max(B, 1)
        if int(route.numel()) != B * K:
            raise ValueError("flat route_pred_bk length must be B*K.")
        route = route.reshape(B, K)
    if route.dim() != 2 or int(route.shape[0]) != B:
        raise ValueError("route_pred_bk must have shape [B,K] or flat [B*K].")
    K = int(route.shape[1])
    if int(cluster_id.max().item()) >= K or int(cluster_id.min().item()) < 0:
        raise ValueError("cluster_id_c contains a cluster outside route_pred_bk.")
    route_bc = route.index_select(1, cluster_id).clamp(min=0, max=P)
    selected = base_bch.detach().cpu().clone()
    for p in range(1, P + 1):
        mask = route_bc == p
        if bool(mask.any().item()):
            selected = torch.where(mask.unsqueeze(-1), cand_bcpH.detach().cpu()[:, :, p - 1, :], selected)
    return selected


def _forecast_metrics_from_route_predictions(
    *,
    base_bch: torch.Tensor,
    cand_bcpH: torch.Tensor,
    y_bch: torch.Tensor,
    cluster_id_c: torch.Tensor,
    route_pred_bk: torch.Tensor,
) -> Dict[str, object]:
    if tuple(base_bch.shape) != tuple(y_bch.shape):
        raise ValueError("base_bch and y_bch must share shape [B,C,H].")
    selected = _cluster_route_predictions_to_forecast(
        base_bch=base_bch,
        cand_bcpH=cand_bcpH,
        cluster_id_c=cluster_id_c,
        route_pred_bk=route_pred_bk,
    )
    base = base_bch.detach().cpu().to(dtype=torch.float32)
    y = y_bch.detach().cpu().to(dtype=torch.float32)
    selected = selected.to(dtype=torch.float32)
    base_err = base - y
    selected_err = selected - y
    base_mse = float(base_err.pow(2).mean().item())
    base_mae = float(base_err.abs().mean().item())
    selected_mse = float(selected_err.pow(2).mean().item())
    selected_mae = float(selected_err.abs().mean().item())
    route = route_pred_bk.detach().cpu().to(dtype=torch.long)
    if route.dim() == 1:
        B = int(base.shape[0])
        route = route.reshape(B, int(route.numel()) // max(B, 1))
    skip_rate = float((route == 0).to(dtype=torch.float32).mean().item()) if route.numel() > 0 else 0.0
    return {
        "samples": int(base.shape[0] * base.shape[1]),
        "base_mse": base_mse,
        "base_mae": base_mae,
        "selected_mse": selected_mse,
        "selected_mae": selected_mae,
        "selected_gain_pct_vs_base": float(100.0 * (base_mse - selected_mse) / max(abs(base_mse), 1.0e-12)),
        "selected_mae_gain_pct_vs_base": float(100.0 * (base_mae - selected_mae) / max(abs(base_mae), 1.0e-12)),
        "skip_rate_cluster": skip_rate,
    }


def _label_count_dict(values: torch.Tensor, label_names: List[str]) -> Dict[str, int]:
    clean = values.detach().cpu().to(dtype=torch.long).view(-1).clamp(0, len(label_names) - 1)
    counts = torch.bincount(clean, minlength=len(label_names))[: len(label_names)]
    return {name: int(counts[i].item()) for i, name in enumerate(label_names)}


def _route_metrics_from_predictions(
    *,
    pred: torch.Tensor,
    labels: torch.Tensor,
    current_pred: Optional[torch.Tensor],
    label_names: List[str],
) -> Dict[str, object]:
    label_count = len(label_names)
    labels = labels.detach().cpu().to(dtype=torch.long).view(-1)
    pred = pred.detach().cpu().to(dtype=torch.long).view(-1)
    if int(labels.numel()) != int(pred.numel()):
        raise ValueError("pred and labels must share N.")
    valid = (labels >= 0) & (labels < label_count)
    labels_v = labels[valid]
    pred_v = pred[valid].clamp(0, label_count - 1)
    if current_pred is None:
        current_v = torch.zeros_like(pred_v)
    else:
        current = current_pred.detach().cpu().to(dtype=torch.long).view(-1)
        if int(current.numel()) != int(labels.numel()):
            raise ValueError("current_pred must share N with labels.")
        current_v = current[valid].clamp(0, label_count - 1)
    samples = int(labels_v.numel())
    if samples <= 0:
        return {
            "samples": 0,
            "accuracy_all": 0.0,
            "current_accuracy_all": 0.0,
            "majority_accuracy_all": 0.0,
            "balanced_accuracy": 0.0,
            "positive_recall_any": 0.0,
            "positive_precision_any": 0.0,
            "oracle_skip_rate": 0.0,
            "head_skip_rate": 0.0,
            "current_skip_rate": 0.0,
            "prediction_counts": {name: 0 for name in label_names},
            "current_prediction_counts": {name: 0 for name in label_names},
            "label_counts": {name: 0 for name in label_names},
            "per_penalty": {},
        }

    label_counts = torch.bincount(labels_v, minlength=label_count)[:label_count]
    majority_count = int(label_counts.max().item())
    accuracy = float((pred_v == labels_v).to(dtype=torch.float32).mean().item())
    current_accuracy = float((current_v == labels_v).to(dtype=torch.float32).mean().item())
    recalls = []
    for label_idx in range(label_count):
        denom = int((labels_v == label_idx).sum().item())
        if denom > 0:
            recalls.append(float(((pred_v == label_idx) & (labels_v == label_idx)).sum().item() / denom))
    oracle_positive = labels_v > 0
    pred_positive = pred_v > 0
    positive_count = int(oracle_positive.sum().item())
    pred_positive_count = int(pred_positive.sum().item())
    true_positive_any = int((oracle_positive & pred_positive).sum().item())
    oracle_skip = labels_v == 0
    pred_skip = pred_v == 0
    current_skip = current_v == 0
    per_penalty: Dict[str, object] = {}
    for label_idx, name in enumerate(label_names[1:], start=1):
        oracle_p = labels_v == label_idx
        pred_p = pred_v == label_idx
        tp = int((oracle_p & pred_p).sum().item())
        pred_count = int(pred_p.sum().item())
        oracle_count = int(oracle_p.sum().item())
        per_penalty[name] = {
            "oracle_count": oracle_count,
            "pred_count": pred_count,
            "precision": _safe_rate(tp, pred_count),
            "recall": _safe_rate(tp, oracle_count),
        }
    return {
        "samples": samples,
        "positive_oracle_samples": positive_count,
        "accuracy_all": accuracy,
        "current_accuracy_all": current_accuracy,
        "majority_accuracy_all": _safe_rate(majority_count, samples),
        "lift_vs_majority": float(accuracy - _safe_rate(majority_count, samples)),
        "lift_vs_current": float(accuracy - current_accuracy),
        "balanced_accuracy": float(sum(recalls) / max(len(recalls), 1)),
        "positive_recall_any": _safe_rate(true_positive_any, positive_count),
        "positive_precision_any": _safe_rate(true_positive_any, pred_positive_count),
        "oracle_skip_rate": float(oracle_skip.to(dtype=torch.float32).mean().item()),
        "head_skip_rate": float(pred_skip.to(dtype=torch.float32).mean().item()),
        "current_skip_rate": float(current_skip.to(dtype=torch.float32).mean().item()),
        "skip_recall": _safe_rate(int((oracle_skip & pred_skip).sum().item()), int(oracle_skip.sum().item())),
        "skip_precision": _safe_rate(int((oracle_skip & pred_skip).sum().item()), int(pred_skip.sum().item())),
        "oracle_skip_routed_to_penalty_rate": _safe_rate(
            int((oracle_skip & pred_positive).sum().item()),
            int(oracle_skip.sum().item()),
        ),
        "oracle_penalty_routed_to_skip_rate": _safe_rate(
            int((oracle_positive & pred_skip).sum().item()),
            positive_count,
        ),
        "prediction_counts": _label_count_dict(pred_v, label_names),
        "current_prediction_counts": _label_count_dict(current_v, label_names),
        "label_counts": _label_count_dict(labels_v, label_names),
        "label_rates": {
            name: float(label_counts[i].item() / max(samples, 1)) for i, name in enumerate(label_names)
        },
        "per_penalty": per_penalty,
    }


def _split_metric(summary: Dict[str, object], split: str, key: str, default: float = 0.0) -> float:
    split_payload = (summary.get("splits", {}) or {}).get(split, {})  # type: ignore[union-attr]
    if not isinstance(split_payload, dict):
        return float(default)
    try:
        return float(split_payload.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _classify_binary_adoption_summary(
    summary: Dict[str, object],
    *,
    min_train_accuracy: float = 0.70,
    min_oracle_skip_rate: float = 0.20,
    min_head_skip_rate: float = 1.0e-4,
    min_apply_rate: float = 0.01,
) -> Dict[str, object]:
    train_acc = _split_metric(summary, "train", "accuracy_all")
    train_maj = _split_metric(summary, "train", "majority_accuracy_all")
    holdout_acc = _split_metric(summary, "train_holdout", "accuracy_all")
    holdout_maj = _split_metric(summary, "train_holdout", "majority_accuracy_all")
    val_acc = _split_metric(summary, "val", "accuracy_all")
    val_maj = _split_metric(summary, "val", "majority_accuracy_all")
    train_oracle_skip = _split_metric(summary, "train", "oracle_skip_rate")
    train_head_skip = _split_metric(summary, "train", "head_skip_rate")
    train_positive = max(0.0, 1.0 - train_oracle_skip)
    train_apply = max(0.0, 1.0 - train_head_skip)
    train_pass = bool(train_acc >= float(min_train_accuracy))
    holdout_lift = float(holdout_acc - holdout_maj)
    val_lift = float(val_acc - val_maj)
    if train_oracle_skip >= float(min_oracle_skip_rate) and train_head_skip <= float(min_head_skip_rate):
        failure_layer = "skip/no-op behavior"
        decision = "binary_adoption_skip_not_adopted"
    elif train_positive >= 0.05 and train_apply <= float(min_apply_rate):
        failure_layer = "skip/no-op behavior"
        decision = "binary_adoption_skip_all_degenerate"
    elif not train_pass:
        failure_layer = "selection/adoption policy"
        decision = "binary_adoption_failed_train_sanity"
    elif holdout_lift <= 0.0 or val_lift <= 0.0:
        failure_layer = "train-val utility shift"
        decision = "binary_adoption_overfits_train_or_holdout"
    else:
        failure_layer = "selection/adoption policy"
        decision = "binary_adoption_passes_route_sanity"
    return {
        "min_train_accuracy": float(min_train_accuracy),
        "train_accuracy": float(train_acc),
        "train_majority_accuracy": float(train_maj),
        "train_lift_vs_majority": float(train_acc - train_maj),
        "train_accuracy_sanity_pass": train_pass,
        "train_oracle_skip_rate": float(train_oracle_skip),
        "train_head_skip_rate": float(train_head_skip),
        "train_apply_rate": float(train_apply),
        "holdout_accuracy": float(holdout_acc),
        "holdout_majority_accuracy": float(holdout_maj),
        "holdout_lift_vs_majority": holdout_lift,
        "holdout_lift_positive": bool(holdout_lift > 0.0),
        "val_accuracy": float(val_acc),
        "val_majority_accuracy": float(val_maj),
        "val_lift_vs_majority": val_lift,
        "val_lift_positive": bool(val_lift > 0.0),
        "failure_layer": failure_layer,
        "decision": decision,
    }


class _BinaryAdoptionHead(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int = 0, dropout: float = 0.0):
        super().__init__()
        if int(hidden_dim) > 0:
            self.net = nn.Sequential(
                nn.Linear(int(feat_dim), int(hidden_dim)),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden_dim), 1),
            )
        else:
            self.net = nn.Linear(int(feat_dim), 1)

    def forward(self, features_nf: torch.Tensor) -> torch.Tensor:
        if features_nf.dim() != 2:
            raise ValueError("features_nf must have shape [N,F].")
        return self.net(features_nf).squeeze(-1)


def _fit_one_binary_head(
    *,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    holdout_x: torch.Tensor,
    holdout_y: torch.Tensor,
    cfg: Dict[str, object],
    device: torch.device,
    seed: int,
) -> Tuple[Optional[Dict[str, torch.Tensor]], Dict[str, object]]:
    pos = int((train_y > 0.5).sum().item())
    neg = int((train_y <= 0.5).sum().item())
    if pos <= 0 or neg <= 0:
        return None, {
            "enabled": False,
            "train_positive": pos,
            "train_negative": neg,
            "reason": "missing_positive_or_negative_examples",
        }
    feat_mean = train_x.mean(dim=0)
    feat_std = train_x.std(dim=0).clamp_min(1.0e-6)

    def _standardize(x: torch.Tensor) -> torch.Tensor:
        return (x.to(device=device, dtype=torch.float32) - feat_mean.to(device=device)) / feat_std.to(device=device)

    torch.manual_seed(int(seed))
    model = _BinaryAdoptionHead(
        feat_dim=int(train_x.shape[-1]),
        hidden_dim=int(cfg.get("hidden_dim", 0)),
        dropout=float(cfg.get("dropout", 0.0)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 3.0e-3)),
        weight_decay=float(cfg.get("weight_decay", 1.0e-4)),
    )
    pos_weight = None
    class_weight = str(cfg.get("class_weight", "balanced")).lower()
    if class_weight in {"balanced", "auto"}:
        weight = float(neg) / max(float(pos), 1.0)
        if float(cfg.get("class_weight_max", 0.0)) > 0.0:
            weight = min(weight, float(cfg.get("class_weight_max", 0.0)))
        pos_weight = torch.tensor(weight, device=device, dtype=torch.float32)
    batch_size = max(1, int(cfg.get("batch_size", 512)))
    epochs = max(1, int(cfg.get("epochs", 200)))
    patience = int(cfg.get("patience", 30))
    min_delta = float(cfg.get("min_delta", 1.0e-4))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    train_x_d = train_x.to(device=device, dtype=torch.float32)
    train_y_d = train_y.to(device=device, dtype=torch.float32)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    best_holdout_loss = float("inf")
    best_train_loss = float("inf")
    stopped_epoch = 0
    no_improve = 0
    history: List[Dict[str, float]] = []

    def _eval_loss(x: torch.Tensor, y: torch.Tensor) -> float:
        if int(y.numel()) <= 0:
            return float("inf")
        with torch.no_grad():
            logits = model(_standardize(x))
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y.to(device=device, dtype=torch.float32))
        return float(loss.detach().item())

    for epoch in range(1, epochs + 1):
        order = torch.randperm(int(train_x.shape[0]), generator=generator)
        model.train()
        total_loss = 0.0
        total_seen = 0
        for start in range(0, int(order.numel()), batch_size):
            idx = order[start : start + batch_size].to(device=device)
            logits = model(_standardize(train_x_d.index_select(0, idx)))
            target = train_y_d.index_select(0, idx)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_n = int(target.numel())
            total_loss += float(loss.detach().item()) * batch_n
            total_seen += batch_n
        train_loss = total_loss / max(total_seen, 1)
        holdout_loss = _eval_loss(holdout_x, holdout_y) if int(holdout_y.numel()) > 0 else train_loss
        history.append({"epoch": float(epoch), "train_loss": float(train_loss), "holdout_loss": float(holdout_loss)})
        best_train_loss = min(best_train_loss, float(train_loss))
        if holdout_loss < best_holdout_loss - min_delta:
            best_holdout_loss = float(holdout_loss)
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                stopped_epoch = int(epoch)
                break
    model.load_state_dict(best_state, strict=True)
    artifact = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "feature_mean": feat_mean.detach().cpu(),
        "feature_std": feat_std.detach().cpu(),
    }
    summary = {
        "enabled": True,
        "train_positive": pos,
        "train_negative": neg,
        "pos_weight": None if pos_weight is None else float(pos_weight.detach().cpu().item()),
        "best_epoch": int(best_epoch),
        "best_train_loss": float(best_train_loss),
        "best_holdout_loss": float(best_holdout_loss),
        "stopped_epoch": int(stopped_epoch),
        "history": history,
    }
    return artifact, summary


def _score_with_head(
    *,
    artifact: Optional[Dict[str, torch.Tensor]],
    features_nqf: torch.Tensor,
    penalty_class: int,
    hidden_dim: int,
    dropout: float,
    device: torch.device,
) -> torch.Tensor:
    x = features_nqf.detach().cpu().to(dtype=torch.float32)[:, int(penalty_class), :]
    if artifact is None:
        return torch.zeros(int(x.shape[0]), dtype=torch.float32)
    model = _BinaryAdoptionHead(int(x.shape[-1]), hidden_dim=int(hidden_dim), dropout=float(dropout)).to(device)
    model.load_state_dict(artifact["state_dict"], strict=True)
    model.eval()
    mean = artifact["feature_mean"].to(device=device)
    std = artifact["feature_std"].to(device=device).clamp_min(1.0e-6)
    scores = []
    batch_size = 4096
    with torch.no_grad():
        for start in range(0, int(x.shape[0]), batch_size):
            xb = x[start : start + batch_size].to(device=device, dtype=torch.float32)
            logits = model((xb - mean) / std)
            scores.append(torch.sigmoid(logits).detach().cpu())
    return torch.cat(scores, dim=0) if scores else torch.zeros(0, dtype=torch.float32)


def _candidate_threshold_values(scores: torch.Tensor, max_values: int) -> List[float]:
    values = scores.detach().cpu().to(dtype=torch.float32)
    finite = values[torch.isfinite(values)]
    if int(finite.numel()) <= 0:
        return [1.0]
    quantiles = torch.linspace(0.02, 0.98, steps=max(3, int(max_values) - 1))
    qs = torch.quantile(finite, quantiles).tolist()
    candidates = [0.5] + [float(v) for v in qs]
    clean = sorted({min(1.0, max(0.0, float(v))) for v in candidates if math.isfinite(float(v))})
    if len(clean) > int(max_values):
        keep = torch.linspace(0, len(clean) - 1, steps=int(max_values)).round().to(dtype=torch.long).tolist()
        clean = [clean[int(i)] for i in keep]
    return clean or [0.5]


def _metric_value(metrics: Dict[str, object], metric: str) -> float:
    metric = str(metric).lower()
    if metric in {"accuracy", "acc", "accuracy_all"}:
        return float(metrics.get("accuracy_all", 0.0))
    if metric in {"balanced_accuracy", "balanced_acc", "bal_acc"}:
        return float(metrics.get("balanced_accuracy", 0.0))
    if metric in {"lift", "lift_vs_majority", "majority_lift"}:
        return float(metrics.get("lift_vs_majority", 0.0))
    if metric in {"positive_recall", "positive_recall_any"}:
        return float(metrics.get("positive_recall_any", 0.0))
    raise ValueError("selection_metric must be accuracy, balanced_accuracy, lift_vs_majority, or positive_recall.")


def _threshold_combinations(grids: List[List[float]], max_combinations: int) -> Iterable[Tuple[float, ...]]:
    total = 1
    for grid in grids:
        total *= max(1, len(grid))
    if total <= int(max_combinations):
        return itertools.product(*grids)
    # Keep deterministic coverage if more penalties are added later.
    trimmed = []
    per_dim = max(2, int(round(float(max_combinations) ** (1.0 / max(len(grids), 1)))))
    for grid in grids:
        if len(grid) <= per_dim:
            trimmed.append(grid)
        else:
            idxs = torch.linspace(0, len(grid) - 1, steps=per_dim).round().to(dtype=torch.long).tolist()
            trimmed.append([grid[int(i)] for i in idxs])
    return itertools.product(*trimmed)


def _select_thresholds(
    *,
    scores_np: torch.Tensor,
    labels: torch.Tensor,
    current_pred: torch.Tensor,
    label_names: List[str],
    selection_metric: str,
    max_thresholds_per_penalty: int,
    max_threshold_combinations: int,
    require_nondegenerate: bool,
    min_head_skip_rate: float,
    min_apply_rate: float,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    P = int(scores_np.shape[1])
    grids = [_candidate_threshold_values(scores_np[:, p], max_thresholds_per_penalty) for p in range(P)]
    best_thresholds: Optional[torch.Tensor] = None
    best_metrics: Optional[Dict[str, object]] = None
    best_value = -float("inf")
    best_degenerate = True
    candidates_seen = 0
    valid_candidates_seen = 0
    for combo in _threshold_combinations(grids, max_threshold_combinations):
        thresholds = torch.tensor(combo, dtype=torch.float32)
        pred = _route_from_binary_scores(scores_np, thresholds)
        metrics = _route_metrics_from_predictions(
            pred=pred,
            labels=labels,
            current_pred=current_pred,
            label_names=label_names,
        )
        candidates_seen += 1
        head_skip = float(metrics.get("head_skip_rate", 0.0))
        apply_rate = max(0.0, 1.0 - head_skip)
        degenerate = bool(head_skip <= float(min_head_skip_rate) or apply_rate <= float(min_apply_rate))
        if bool(require_nondegenerate) and degenerate:
            candidate_allowed = False
        else:
            candidate_allowed = True
        value = _metric_value(metrics, selection_metric)
        tie_break = float(metrics.get("accuracy_all", 0.0))
        if candidate_allowed:
            valid_candidates_seen += 1
        should_take = False
        if best_thresholds is None:
            should_take = True
        elif candidate_allowed and best_degenerate:
            should_take = True
        elif candidate_allowed == (not best_degenerate):
            if value > best_value + 1.0e-12:
                should_take = True
            elif abs(value - best_value) <= 1.0e-12 and best_metrics is not None:
                should_take = tie_break > float(best_metrics.get("accuracy_all", 0.0))
        elif not bool(require_nondegenerate) and value > best_value + 1.0e-12:
            should_take = True
        if should_take:
            best_thresholds = thresholds
            best_metrics = metrics
            best_value = float(value)
            best_degenerate = degenerate
    if best_thresholds is None or best_metrics is None:
        best_thresholds = torch.full((P,), 0.5, dtype=torch.float32)
        best_metrics = _route_metrics_from_predictions(
            pred=_route_from_binary_scores(scores_np, best_thresholds),
            labels=labels,
            current_pred=current_pred,
            label_names=label_names,
        )
        best_value = _metric_value(best_metrics, selection_metric)
        best_degenerate = True
    return best_thresholds, {
        "selection_metric": str(selection_metric),
        "threshold_grids": grids,
        "candidates_seen": int(candidates_seen),
        "valid_candidates_seen": int(valid_candidates_seen),
        "best_value": float(best_value),
        "best_degenerate": bool(best_degenerate),
        "best_metrics": best_metrics,
    }


def _load_split_tensors(tensors_dir: Path) -> Dict[str, Dict[str, torch.Tensor]]:
    paths = {
        "train": tensors_dir / "fixed_candidate_route_tensors_train_fit.pt",
        "train_holdout": tensors_dir / "fixed_candidate_route_tensors_train_holdout.pt",
        "val": tensors_dir / "fixed_candidate_route_tensors_val.pt",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required no-test route tensors: {missing}")
    return {split: torch.load(path, map_location="cpu") for split, path in paths.items()}


def _split_row(split: str, metrics: Dict[str, object]) -> Dict[str, object]:
    return {
        "split": split,
        "samples": int(metrics.get("samples", 0) or 0),
        "head_accuracy": float(metrics.get("accuracy_all", 0.0) or 0.0),
        "current_accuracy": float(metrics.get("current_accuracy_all", 0.0) or 0.0),
        "majority_accuracy": float(metrics.get("majority_accuracy_all", 0.0) or 0.0),
        "balanced_accuracy": float(metrics.get("balanced_accuracy", 0.0) or 0.0),
        "head_lift_vs_majority": float(metrics.get("lift_vs_majority", 0.0) or 0.0),
        "oracle_skip_rate": float(metrics.get("oracle_skip_rate", 0.0) or 0.0),
        "head_skip_rate": float(metrics.get("head_skip_rate", 0.0) or 0.0),
        "current_skip_rate": float(metrics.get("current_skip_rate", 0.0) or 0.0),
        "positive_recall_any": float(metrics.get("positive_recall_any", 0.0) or 0.0),
        "positive_precision_any": float(metrics.get("positive_precision_any", 0.0) or 0.0),
        "label_counts": metrics.get("label_counts", {}),
        "head_prediction_counts": metrics.get("prediction_counts", {}),
        "current_prediction_counts": metrics.get("current_prediction_counts", {}),
    }


def _markdown_report(payload: Dict[str, object]) -> str:
    lines = [
        "# NEXT-11d Binary Adoption Refit",
        "",
        f"- tensors_dir: `{payload['tensors_dir']}`",
        f"- no_test_read: `{payload['no_test_read']}`",
        f"- failure_layer: `{payload['verdict']['failure_layer']}`",
        f"- decision: `{payload['verdict']['decision']}`",
        "",
        "## Split Metrics",
        "",
        "| split | samples | head acc | current acc | majority | balanced | oracle skip | head skip | current skip | pos precision | pos recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("split_rows", []):
        lines.append(
            "| {split} | {samples} | {head:.4f} | {current:.4f} | {majority:.4f} | {balanced:.4f} | {oracle_skip:.4f} | {head_skip:.4f} | {current_skip:.4f} | {pos_precision:.4f} | {pos_recall:.4f} |".format(
                split=row["split"],
                samples=int(row["samples"]),
                head=float(row["head_accuracy"]),
                current=float(row["current_accuracy"]),
                majority=float(row["majority_accuracy"]),
                balanced=float(row["balanced_accuracy"]),
                oracle_skip=float(row["oracle_skip_rate"]),
                head_skip=float(row["head_skip_rate"]),
                current_skip=float(row["current_skip_rate"]),
                pos_precision=float(row["positive_precision_any"]),
                pos_recall=float(row["positive_recall_any"]),
            )
        )
    lines.extend(
        [
            "",
            "## Thresholds",
            "",
            f"- selection_split: `{payload['selection']['split']}`",
            f"- selection_metric: `{payload['selection']['selection_metric']}`",
            f"- thresholds: `{payload['selection']['thresholds']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run_refit(args: argparse.Namespace) -> Dict[str, object]:
    split_tensors = _load_split_tensors(Path(args.tensors_dir))
    label_names = list(split_tensors["train"]["label_names"])  # type: ignore[index]
    feature_names = list(split_tensors["train"]["feature_names"])  # type: ignore[index]
    if len(label_names) < 2:
        raise ValueError("binary adoption refit requires skip plus at least one penalty label.")
    label_count = len(label_names)
    device = torch.device(str(args.device) if str(args.device) != "cuda" or torch.cuda.is_available() else "cpu")
    cfg = {
        "epochs": int(args.epochs),
        "batch_size": int(args.head_batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "hidden_dim": 0 if str(args.head_mode).lower() == "linear" else int(args.hidden_dim),
        "dropout": float(args.dropout),
        "class_weight": str(args.class_weight),
        "class_weight_max": float(args.class_weight_max),
        "patience": int(args.patience),
        "min_delta": float(args.min_delta),
    }
    artifacts: Dict[int, Optional[Dict[str, torch.Tensor]]] = {}
    head_summaries: Dict[str, object] = {}
    train_features = split_tensors["train"]["features"]
    train_labels = split_tensors["train"]["labels"]
    holdout_features = split_tensors["train_holdout"]["features"]
    holdout_labels = split_tensors["train_holdout"]["labels"]
    for penalty_class in range(1, label_count):
        train_x, train_y, _ = _binary_examples_for_penalty(
            train_features,
            train_labels,
            penalty_class=penalty_class,
            ignore_other_positive=True,
        )
        holdout_x, holdout_y, _ = _binary_examples_for_penalty(
            holdout_features,
            holdout_labels,
            penalty_class=penalty_class,
            ignore_other_positive=True,
        )
        artifact, head_summary = _fit_one_binary_head(
            train_x=train_x,
            train_y=train_y,
            holdout_x=holdout_x,
            holdout_y=holdout_y,
            cfg=cfg,
            device=device,
            seed=int(args.seed) + penalty_class,
        )
        artifacts[penalty_class] = artifact
        head_summaries[label_names[penalty_class]] = head_summary

    split_scores: Dict[str, torch.Tensor] = {}
    for split, tensors in split_tensors.items():
        scores = []
        for penalty_class in range(1, label_count):
            scores.append(
                _score_with_head(
                    artifact=artifacts.get(penalty_class),
                    features_nqf=tensors["features"],
                    penalty_class=penalty_class,
                    hidden_dim=int(cfg["hidden_dim"]),
                    dropout=float(cfg["dropout"]),
                    device=device,
                )
            )
        split_scores[split] = torch.stack(scores, dim=1) if scores else torch.zeros(0, 0)

    selection_split = str(args.selection_split)
    if selection_split not in split_scores:
        raise ValueError("selection_split must be train or train_holdout.")
    thresholds, selection_summary = _select_thresholds(
        scores_np=split_scores[selection_split],
        labels=split_tensors[selection_split]["labels"],
        current_pred=split_tensors[selection_split]["current_pred"],
        label_names=label_names,
        selection_metric=str(args.selection_metric),
        max_thresholds_per_penalty=int(args.max_thresholds_per_penalty),
        max_threshold_combinations=int(args.max_threshold_combinations),
        require_nondegenerate=not bool(args.allow_degenerate_selection),
        min_head_skip_rate=float(args.min_head_skip_rate),
        min_apply_rate=float(args.min_apply_rate),
    )
    split_metrics = {}
    split_predictions = {}
    for split, tensors in split_tensors.items():
        pred = _route_from_binary_scores(split_scores[split], thresholds)
        split_predictions[split] = pred
        split_metrics[split] = _route_metrics_from_predictions(
            pred=pred,
            labels=tensors["labels"],
            current_pred=tensors["current_pred"],
            label_names=label_names,
        )
    summary = {
        "enable": True,
        "no_test_read": True,
        "label_names": label_names,
        "feature_names": feature_names,
        "head_config": {
            **cfg,
            "head_mode": str(args.head_mode),
            "selection_split": selection_split,
            "selection_metric": str(args.selection_metric),
            "seed": int(args.seed),
            "ignore_other_positive": True,
        },
        "head_summaries": head_summaries,
        "selection": {
            "split": selection_split,
            "thresholds": [float(v) for v in thresholds.tolist()],
            **selection_summary,
        },
        "splits": split_metrics,
    }
    verdict = _classify_binary_adoption_summary(summary, min_train_accuracy=float(args.min_train_accuracy))
    split_rows = [_split_row(split, metrics) for split, metrics in split_metrics.items()]
    payload = {
        "tensors_dir": str(args.tensors_dir),
        "out_dir": str(args.out_dir),
        "device": str(device),
        "no_test_read": True,
        "summary": summary,
        "split_rows": split_rows,
        "selection": summary["selection"],
        "verdict": verdict,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "heads": artifacts,
            "thresholds": thresholds.detach().cpu(),
            "label_names": label_names,
            "feature_names": feature_names,
            "config": summary["head_config"],
            "selection": summary["selection"],
        },
        out_dir / "binary_adoption_heads.pt",
    )
    torch.save(split_predictions, out_dir / "binary_adoption_predictions.pt")
    (out_dir / "binary_adoption_refit.json").write_text(
        json.dumps(_to_jsonable(payload), indent=2),
        encoding="utf-8",
    )
    (out_dir / "binary_adoption_refit.md").write_text(_markdown_report(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="NEXT-11d offline binary apply-vs-skip adoption refit.")
    parser.add_argument("--tensors-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--head-mode", type=str, default="linear", choices=["linear", "mlp"])
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min-delta", type=float, default=1.0e-4)
    parser.add_argument("--head-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--class-weight", type=str, default="balanced", choices=["none", "balanced", "auto"])
    parser.add_argument("--class-weight-max", type=float, default=0.0)
    parser.add_argument("--selection-split", type=str, default="train_holdout", choices=["train", "train_holdout"])
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="balanced_accuracy",
        choices=["accuracy", "balanced_accuracy", "lift_vs_majority", "positive_recall"],
    )
    parser.add_argument("--max-thresholds-per-penalty", type=int, default=17)
    parser.add_argument("--max-threshold-combinations", type=int, default=5000)
    parser.add_argument("--allow-degenerate-selection", action="store_true")
    parser.add_argument("--min-head-skip-rate", type=float, default=1.0e-4)
    parser.add_argument("--min-apply-rate", type=float, default=0.01)
    parser.add_argument("--min-train-accuracy", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    payload = run_refit(args)
    verdict = payload["verdict"]
    print(
        "train_acc={:.4f} holdout_acc={:.4f} val_acc={:.4f} train_skip={:.4f} failure_layer={} decision={}".format(
            float(verdict["train_accuracy"]),
            float(verdict["holdout_accuracy"]),
            float(verdict["val_accuracy"]),
            float(verdict["train_head_skip_rate"]),
            str(verdict["failure_layer"]),
            str(verdict["decision"]),
        )
    )


if __name__ == "__main__":
    main()
