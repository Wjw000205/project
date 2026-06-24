from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.next11d_binary_adoption_refit import _route_metrics_from_predictions, _to_jsonable


def _safe_rate(num: int, denom: int) -> float:
    return float(num) / max(int(denom), 1)


def _load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _route_tensor_path(tensors_dir: Path, split: str) -> Path:
    suffix = "train_fit" if split == "train" else split
    return tensors_dir / f"fixed_candidate_route_tensors_{suffix}.pt"


def _load_split_tensors(tensors_dir: Path) -> Dict[str, Dict[str, torch.Tensor]]:
    paths = {
        "train": _route_tensor_path(tensors_dir, "train"),
        "train_holdout": _route_tensor_path(tensors_dir, "train_holdout"),
        "val": _route_tensor_path(tensors_dir, "val"),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required no-test route tensors: {missing}")
    return {split: torch.load(path, map_location="cpu") for split, path in paths.items()}


def _infer_cluster_count(
    *,
    explicit_cluster_count: int,
    refit_summary: Dict[str, object],
    tensors: Dict[str, Dict[str, torch.Tensor]],
) -> int:
    if int(explicit_cluster_count) > 0:
        return int(explicit_cluster_count)
    route_context = refit_summary.get("route_context", {}) if isinstance(refit_summary, dict) else {}
    if isinstance(route_context, dict):
        try:
            cluster_count = int(route_context.get("cluster_count", 0))
            if cluster_count > 0:
                return cluster_count
        except (TypeError, ValueError):
            pass
    train = tensors["train"]
    query = train.get("query_start_abs")
    if isinstance(query, torch.Tensor) and int(query.numel()) > 0:
        first = query.detach().cpu().view(-1)[0]
        same_first = int((query.detach().cpu().view(-1) == first).sum().item())
        if same_first > 0 and int(train["labels"].numel()) % same_first == 0:
            return same_first
    raise ValueError("Could not infer cluster_count; pass --cluster-count or provide fixed_candidate_router_refit.json.")


def _cluster_index_for_flat_predictions(numel: int, cluster_count: int) -> torch.Tensor:
    if int(cluster_count) <= 0:
        raise ValueError("cluster_count must be positive.")
    if int(numel) % int(cluster_count) != 0:
        raise ValueError("flat prediction length must be divisible by cluster_count.")
    return torch.arange(int(numel), dtype=torch.long) % int(cluster_count)


def _summarize_cluster_penalty_stats(
    *,
    labels: torch.Tensor,
    pred: torch.Tensor,
    cluster_count: int,
    label_names: List[str],
) -> Dict[Tuple[int, int], Dict[str, object]]:
    labels_v = labels.detach().cpu().to(dtype=torch.long).view(-1)
    pred_v = pred.detach().cpu().to(dtype=torch.long).view(-1)
    if int(labels_v.numel()) != int(pred_v.numel()):
        raise ValueError("labels and pred must share N.")
    label_count = int(len(label_names))
    if label_count < 2:
        raise ValueError("label_names must contain skip plus at least one penalty.")
    clusters = _cluster_index_for_flat_predictions(int(labels_v.numel()), int(cluster_count))
    stats: Dict[Tuple[int, int], Dict[str, object]] = {}
    for cluster in range(int(cluster_count)):
        in_cluster = clusters == int(cluster)
        cluster_samples = int(in_cluster.sum().item())
        for penalty_class in range(1, label_count):
            predicted = in_cluster & (pred_v == int(penalty_class))
            oracle = in_cluster & (labels_v == int(penalty_class))
            oracle_positive = in_cluster & (labels_v > 0)
            support_pred = int(predicted.sum().item())
            oracle_count = int(oracle.sum().item())
            exact_tp = int((predicted & oracle).sum().item())
            positive_tp = int((predicted & oracle_positive).sum().item())
            false_adopt = int((predicted & (labels_v == 0)).sum().item())
            wrong_penalty = int((predicted & (labels_v > 0) & (labels_v != int(penalty_class))).sum().item())
            stats[(cluster, penalty_class)] = {
                "cluster": int(cluster),
                "penalty_class": int(penalty_class),
                "penalty": str(label_names[penalty_class]),
                "cluster_samples": cluster_samples,
                "pred_support": support_pred,
                "oracle_count": oracle_count,
                "exact_true_positive": exact_tp,
                "positive_true_positive": positive_tp,
                "false_adopt_skip": false_adopt,
                "wrong_penalty_positive": wrong_penalty,
                "exact_precision": _safe_rate(exact_tp, support_pred),
                "positive_precision": _safe_rate(positive_tp, support_pred),
                "exact_recall": _safe_rate(exact_tp, oracle_count),
                "pred_rate": _safe_rate(support_pred, cluster_samples),
            }
    return stats


def _stable_cluster_penalty_mask(
    *,
    fit_stats: Dict[Tuple[int, int], Dict[str, object]],
    holdout_stats: Dict[Tuple[int, int], Dict[str, object]],
    label_names: List[str],
    min_support: int,
    min_exact_precision: float,
    min_exact_recall: float,
    max_precision_gap: float,
) -> Tuple[torch.Tensor, List[Dict[str, object]]]:
    keys = sorted(set(fit_stats.keys()) | set(holdout_stats.keys()))
    cluster_count = max((int(k[0]) for k in keys), default=-1) + 1
    penalty_count = max(0, int(len(label_names)) - 1)
    mask = torch.zeros(cluster_count, penalty_count, dtype=torch.bool)
    rows: List[Dict[str, object]] = []
    for cluster, penalty_class in keys:
        fit = fit_stats.get((cluster, penalty_class), {})
        holdout = holdout_stats.get((cluster, penalty_class), {})
        fit_support = int(fit.get("pred_support", 0) or 0)
        holdout_support = int(holdout.get("pred_support", 0) or 0)
        fit_precision = float(fit.get("exact_precision", 0.0) or 0.0)
        holdout_precision = float(holdout.get("exact_precision", 0.0) or 0.0)
        fit_recall = float(fit.get("exact_recall", 0.0) or 0.0)
        holdout_recall = float(holdout.get("exact_recall", 0.0) or 0.0)
        precision_gap = abs(fit_precision - holdout_precision)
        allowed = bool(
            fit_support >= int(min_support)
            and holdout_support >= int(min_support)
            and fit_precision >= float(min_exact_precision)
            and holdout_precision >= float(min_exact_precision)
            and fit_recall >= float(min_exact_recall)
            and holdout_recall >= float(min_exact_recall)
            and precision_gap <= float(max_precision_gap)
        )
        if allowed and penalty_class > 0:
            mask[int(cluster), int(penalty_class) - 1] = True
        rows.append(
            {
                "cluster": int(cluster),
                "penalty_class": int(penalty_class),
                "penalty": str(label_names[penalty_class]) if penalty_class < len(label_names) else str(penalty_class),
                "allowed": allowed,
                "fit_pred_support": fit_support,
                "holdout_pred_support": holdout_support,
                "fit_exact_precision": fit_precision,
                "holdout_exact_precision": holdout_precision,
                "precision_gap": float(precision_gap),
                "fit_exact_recall": fit_recall,
                "holdout_exact_recall": holdout_recall,
                "fit_positive_precision": float(fit.get("positive_precision", 0.0) or 0.0),
                "holdout_positive_precision": float(holdout.get("positive_precision", 0.0) or 0.0),
            }
        )
    return mask, rows


def _apply_cluster_penalty_guard(
    pred: torch.Tensor,
    *,
    cluster_count: int,
    allowed_kp: torch.Tensor,
) -> torch.Tensor:
    pred_v = pred.detach().cpu().to(dtype=torch.long).view(-1)
    allowed = allowed_kp.detach().cpu().to(dtype=torch.bool)
    if allowed.dim() != 2:
        raise ValueError("allowed_kp must have shape [K,P].")
    if int(allowed.shape[0]) != int(cluster_count):
        raise ValueError("allowed_kp cluster dimension must equal cluster_count.")
    clusters = _cluster_index_for_flat_predictions(int(pred_v.numel()), int(cluster_count))
    guarded = torch.zeros_like(pred_v)
    penalty_count = int(allowed.shape[1])
    positive = (pred_v > 0) & (pred_v <= penalty_count)
    if bool(positive.any().item()):
        pred_penalty_idx = pred_v[positive] - 1
        cluster_idx = clusters[positive]
        keep = allowed[cluster_idx, pred_penalty_idx]
        positive_indices = torch.nonzero(positive, as_tuple=False).view(-1)
        kept_indices = positive_indices[keep]
        guarded[kept_indices] = pred_v[kept_indices]
    return guarded


def _score_feature_index(feature_names: List[str], score_feature: str) -> int:
    try:
        return int(feature_names.index(str(score_feature)))
    except ValueError as exc:
        raise ValueError(f"score feature {score_feature!r} not found in route tensor feature_names.") from exc


def _score_threshold_stats(
    *,
    tensors: Dict[str, torch.Tensor],
    cluster: int,
    penalty_class: int,
    cluster_count: int,
    score_feature_idx: int,
    threshold: float,
) -> Dict[str, object]:
    labels_v = tensors["labels"].detach().cpu().to(dtype=torch.long).view(-1)
    pred_v = tensors["current_pred"].detach().cpu().to(dtype=torch.long).view(-1)
    features = tensors["features"].detach().cpu().to(dtype=torch.float32)
    if features.dim() != 3 or int(features.shape[0]) != int(labels_v.numel()):
        raise ValueError("features must have shape [N,num_classes,F] and share labels N.")
    clusters = _cluster_index_for_flat_predictions(int(labels_v.numel()), int(cluster_count))
    in_cluster = clusters == int(cluster)
    score_v = features[:, int(penalty_class), int(score_feature_idx)]
    predicted = in_cluster & (pred_v == int(penalty_class)) & torch.isfinite(score_v) & (score_v >= float(threshold))
    oracle = in_cluster & (labels_v == int(penalty_class))
    oracle_positive = in_cluster & (labels_v > 0)
    support_pred = int(predicted.sum().item())
    oracle_count = int(oracle.sum().item())
    exact_tp = int((predicted & oracle).sum().item())
    positive_tp = int((predicted & oracle_positive).sum().item())
    return {
        "pred_support": support_pred,
        "oracle_count": oracle_count,
        "exact_true_positive": exact_tp,
        "positive_true_positive": positive_tp,
        "exact_precision": _safe_rate(exact_tp, support_pred),
        "positive_precision": _safe_rate(positive_tp, support_pred),
        "exact_recall": _safe_rate(exact_tp, oracle_count),
    }


def _score_threshold_candidates(
    *,
    train_fit: Dict[str, torch.Tensor],
    cluster: int,
    penalty_class: int,
    cluster_count: int,
    score_feature_idx: int,
    quantiles: List[float],
) -> List[Tuple[float, float]]:
    labels_v = train_fit["labels"].detach().cpu().to(dtype=torch.long).view(-1)
    pred_v = train_fit["current_pred"].detach().cpu().to(dtype=torch.long).view(-1)
    features = train_fit["features"].detach().cpu().to(dtype=torch.float32)
    clusters = _cluster_index_for_flat_predictions(int(labels_v.numel()), int(cluster_count))
    score_v = features[:, int(penalty_class), int(score_feature_idx)]
    mask = (clusters == int(cluster)) & (pred_v == int(penalty_class)) & torch.isfinite(score_v)
    values = score_v[mask]
    if int(values.numel()) <= 0:
        return []
    candidates: List[Tuple[float, float]] = []
    for quantile in quantiles:
        q = min(1.0, max(0.0, float(quantile)))
        threshold = float(torch.quantile(values, q).item())
        pair = (q, threshold)
        if pair not in candidates:
            candidates.append(pair)
    return sorted(candidates, key=lambda item: item[1])


def _select_score_threshold_guard(
    *,
    train_fit: Dict[str, torch.Tensor],
    train_holdout: Dict[str, torch.Tensor],
    cluster_count: int,
    label_names: List[str],
    feature_names: List[str],
    score_feature: str,
    quantiles: List[float],
    min_support: int,
    min_exact_precision: float,
    min_exact_recall: float,
    max_precision_gap: float,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, object]]]:
    score_idx = _score_feature_index(feature_names, score_feature)
    penalty_count = max(0, int(len(label_names)) - 1)
    allowed = torch.zeros(int(cluster_count), penalty_count, dtype=torch.bool)
    thresholds = torch.full((int(cluster_count), penalty_count), float("inf"), dtype=torch.float32)
    rows: List[Dict[str, object]] = []
    for cluster in range(int(cluster_count)):
        for penalty_class in range(1, int(len(label_names))):
            selected: Optional[Dict[str, object]] = None
            candidates = _score_threshold_candidates(
                train_fit=train_fit,
                cluster=cluster,
                penalty_class=penalty_class,
                cluster_count=int(cluster_count),
                score_feature_idx=score_idx,
                quantiles=quantiles,
            )
            for quantile, threshold in candidates:
                fit = _score_threshold_stats(
                    tensors=train_fit,
                    cluster=cluster,
                    penalty_class=penalty_class,
                    cluster_count=int(cluster_count),
                    score_feature_idx=score_idx,
                    threshold=threshold,
                )
                holdout = _score_threshold_stats(
                    tensors=train_holdout,
                    cluster=cluster,
                    penalty_class=penalty_class,
                    cluster_count=int(cluster_count),
                    score_feature_idx=score_idx,
                    threshold=threshold,
                )
                fit_precision = float(fit.get("exact_precision", 0.0) or 0.0)
                holdout_precision = float(holdout.get("exact_precision", 0.0) or 0.0)
                precision_gap = abs(fit_precision - holdout_precision)
                pass_guard = bool(
                    int(fit.get("pred_support", 0) or 0) >= int(min_support)
                    and int(holdout.get("pred_support", 0) or 0) >= int(min_support)
                    and fit_precision >= float(min_exact_precision)
                    and holdout_precision >= float(min_exact_precision)
                    and float(fit.get("exact_recall", 0.0) or 0.0) >= float(min_exact_recall)
                    and float(holdout.get("exact_recall", 0.0) or 0.0) >= float(min_exact_recall)
                    and precision_gap <= float(max_precision_gap)
                )
                if pass_guard:
                    selected = {
                        "cluster": int(cluster),
                        "penalty_class": int(penalty_class),
                        "penalty": str(label_names[penalty_class]),
                        "allowed": True,
                        "selected_quantile": float(quantile),
                        "score_threshold": float(threshold),
                        "fit_pred_support": int(fit.get("pred_support", 0) or 0),
                        "holdout_pred_support": int(holdout.get("pred_support", 0) or 0),
                        "fit_exact_precision": fit_precision,
                        "holdout_exact_precision": holdout_precision,
                        "precision_gap": float(precision_gap),
                        "fit_exact_recall": float(fit.get("exact_recall", 0.0) or 0.0),
                        "holdout_exact_recall": float(holdout.get("exact_recall", 0.0) or 0.0),
                        "fit_positive_precision": float(fit.get("positive_precision", 0.0) or 0.0),
                        "holdout_positive_precision": float(holdout.get("positive_precision", 0.0) or 0.0),
                    }
                    break
            if selected is None:
                selected = {
                    "cluster": int(cluster),
                    "penalty_class": int(penalty_class),
                    "penalty": str(label_names[penalty_class]),
                    "allowed": False,
                    "selected_quantile": None,
                    "score_threshold": None,
                    "fit_pred_support": 0,
                    "holdout_pred_support": 0,
                    "fit_exact_precision": 0.0,
                    "holdout_exact_precision": 0.0,
                    "precision_gap": 0.0,
                    "fit_exact_recall": 0.0,
                    "holdout_exact_recall": 0.0,
                    "fit_positive_precision": 0.0,
                    "holdout_positive_precision": 0.0,
                }
            else:
                allowed[int(cluster), int(penalty_class) - 1] = True
                thresholds[int(cluster), int(penalty_class) - 1] = float(selected["score_threshold"])
            rows.append(selected)
    return allowed, thresholds, rows


def _apply_cluster_penalty_score_guard(
    pred: torch.Tensor,
    *,
    features: torch.Tensor,
    cluster_count: int,
    allowed_kp: torch.Tensor,
    score_threshold_kp: torch.Tensor,
    score_feature_idx: int,
) -> torch.Tensor:
    pred_v = pred.detach().cpu().to(dtype=torch.long).view(-1)
    feature_v = features.detach().cpu().to(dtype=torch.float32)
    allowed = allowed_kp.detach().cpu().to(dtype=torch.bool)
    thresholds = score_threshold_kp.detach().cpu().to(dtype=torch.float32)
    if feature_v.dim() != 3 or int(feature_v.shape[0]) != int(pred_v.numel()):
        raise ValueError("features must have shape [N,num_classes,F] and share pred N.")
    if tuple(thresholds.shape) != tuple(allowed.shape):
        raise ValueError("score_threshold_kp must share allowed_kp shape.")
    clusters = _cluster_index_for_flat_predictions(int(pred_v.numel()), int(cluster_count))
    guarded = torch.zeros_like(pred_v)
    penalty_count = int(allowed.shape[1])
    positive = (pred_v > 0) & (pred_v <= penalty_count)
    if bool(positive.any().item()):
        positive_indices = torch.nonzero(positive, as_tuple=False).view(-1)
        pred_penalty_idx = pred_v[positive_indices] - 1
        cluster_idx = clusters[positive_indices]
        score = feature_v[positive_indices, pred_v[positive_indices], int(score_feature_idx)]
        keep = allowed[cluster_idx, pred_penalty_idx] & torch.isfinite(score) & (
            score >= thresholds[cluster_idx, pred_penalty_idx]
        )
        kept_indices = positive_indices[keep]
        guarded[kept_indices] = pred_v[kept_indices]
    return guarded


def _split_row(split: str, metrics: Dict[str, object]) -> Dict[str, object]:
    return {
        "split": str(split),
        "samples": int(metrics.get("samples", 0) or 0),
        "guard_accuracy": float(metrics.get("accuracy_all", 0.0) or 0.0),
        "current_accuracy": float(metrics.get("current_accuracy_all", 0.0) or 0.0),
        "majority_accuracy": float(metrics.get("majority_accuracy_all", 0.0) or 0.0),
        "guard_lift_vs_majority": float(metrics.get("lift_vs_majority", 0.0) or 0.0),
        "oracle_skip_rate": float(metrics.get("oracle_skip_rate", 0.0) or 0.0),
        "guard_skip_rate": float(metrics.get("head_skip_rate", 0.0) or 0.0),
        "current_skip_rate": float(metrics.get("current_skip_rate", 0.0) or 0.0),
        "positive_precision_any": float(metrics.get("positive_precision_any", 0.0) or 0.0),
        "positive_recall_any": float(metrics.get("positive_recall_any", 0.0) or 0.0),
        "oracle_skip_routed_to_penalty_rate": float(metrics.get("oracle_skip_routed_to_penalty_rate", 0.0) or 0.0),
        "oracle_penalty_routed_to_skip_rate": float(metrics.get("oracle_penalty_routed_to_skip_rate", 0.0) or 0.0),
    }


def _classify_guard_result(split_metrics: Dict[str, Dict[str, object]], allowed_count: int) -> Dict[str, object]:
    val = split_metrics.get("val", {})
    train = split_metrics.get("train", {})
    holdout = split_metrics.get("train_holdout", {})
    if int(allowed_count) <= 0:
        failure_layer = "selection/adoption policy"
        decision = "stable_guard_empty_mask_over_skips"
    elif float(train.get("accuracy_all", 0.0)) < float(train.get("majority_accuracy_all", 0.0)):
        failure_layer = "selection/adoption policy"
        decision = "stable_guard_hurts_train_labels"
    elif float(holdout.get("accuracy_all", 0.0)) < float(holdout.get("majority_accuracy_all", 0.0)):
        failure_layer = "train-val utility shift"
        decision = "stable_guard_not_holdout_stable"
    elif float(val.get("accuracy_all", 0.0)) < float(val.get("majority_accuracy_all", 0.0)):
        failure_layer = "train-val utility shift"
        decision = "stable_guard_train_stable_val_shift"
    else:
        failure_layer = "selection/adoption policy"
        decision = "stable_guard_route_sanity_pass"
    return {
        "allowed_cluster_penalty_count": int(allowed_count),
        "failure_layer": failure_layer,
        "decision": decision,
    }


def _parse_quantiles(raw: str) -> List[float]:
    values: List[float] = []
    for piece in str(raw).split(","):
        piece = piece.strip()
        if not piece:
            continue
        values.append(min(1.0, max(0.0, float(piece))))
    return values or [0.5, 0.75, 0.90, 0.95, 0.98]


def _markdown_report(payload: Dict[str, object]) -> str:
    lines = [
        "# NEXT-11d Split-Stable Adoption Guard",
        "",
        f"- tensors_dir: `{payload['tensors_dir']}`",
        f"- no_test_read: `{payload['no_test_read']}`",
        f"- failure_layer: `{payload['verdict']['failure_layer']}`",
        f"- decision: `{payload['verdict']['decision']}`",
        "",
        "## Guard Criteria",
        "",
        f"- min_support: `{payload['guard_config']['min_support']}`",
        f"- min_exact_precision: `{payload['guard_config']['min_exact_precision']}`",
        f"- min_exact_recall: `{payload['guard_config']['min_exact_recall']}`",
        f"- max_precision_gap: `{payload['guard_config']['max_precision_gap']}`",
        f"- score_feature: `{payload['guard_config'].get('score_feature')}`",
        "",
        "## Split Metrics",
        "",
        "| split | samples | guard acc | current acc | majority | guard skip | current skip | pos precision | pos recall | skip->penalty | penalty->skip |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("split_rows", []):
        lines.append(
            "| {split} | {samples} | {guard_acc:.4f} | {current_acc:.4f} | {majority:.4f} | {guard_skip:.4f} | {current_skip:.4f} | {precision:.4f} | {recall:.4f} | {skip_to_penalty:.4f} | {penalty_to_skip:.4f} |".format(
                split=row["split"],
                samples=int(row["samples"]),
                guard_acc=float(row["guard_accuracy"]),
                current_acc=float(row["current_accuracy"]),
                majority=float(row["majority_accuracy"]),
                guard_skip=float(row["guard_skip_rate"]),
                current_skip=float(row["current_skip_rate"]),
                precision=float(row["positive_precision_any"]),
                recall=float(row["positive_recall_any"]),
                skip_to_penalty=float(row["oracle_skip_routed_to_penalty_rate"]),
                penalty_to_skip=float(row["oracle_penalty_routed_to_skip_rate"]),
            )
        )
    lines.extend(
        [
            "",
            "## Allowed Cluster Penalties",
            "",
            "| cluster | penalty | allowed | fit support | holdout support | fit precision | holdout precision | fit recall | holdout recall |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload.get("stability_rows", []):
        if not bool(row.get("allowed", False)):
            continue
        lines.append(
            "| {cluster} | {penalty} | {allowed} | {fit_support} | {holdout_support} | {fit_precision:.4f} | {holdout_precision:.4f} | {fit_recall:.4f} | {holdout_recall:.4f} |".format(
                cluster=int(row["cluster"]),
                penalty=row["penalty"],
                allowed=str(bool(row["allowed"])),
                fit_support=int(row["fit_pred_support"]),
                holdout_support=int(row["holdout_pred_support"]),
                fit_precision=float(row["fit_exact_precision"]),
                holdout_precision=float(row["holdout_exact_precision"]),
                fit_recall=float(row["fit_exact_recall"]),
                holdout_recall=float(row["holdout_exact_recall"]),
            )
        )
    return "\n".join(lines) + "\n"


def run_guard(args: argparse.Namespace) -> Dict[str, object]:
    tensors_dir = Path(args.tensors_dir)
    split_tensors = _load_split_tensors(tensors_dir)
    refit_summary = _load_json(tensors_dir / "fixed_candidate_router_refit.json")
    cluster_count = _infer_cluster_count(
        explicit_cluster_count=int(args.cluster_count),
        refit_summary=refit_summary,
        tensors=split_tensors,
    )
    label_names = list(split_tensors["train"]["label_names"])  # type: ignore[index]
    fit_stats = _summarize_cluster_penalty_stats(
        labels=split_tensors["train"]["labels"],
        pred=split_tensors["train"]["current_pred"],
        cluster_count=int(cluster_count),
        label_names=label_names,
    )
    holdout_stats = _summarize_cluster_penalty_stats(
        labels=split_tensors["train_holdout"]["labels"],
        pred=split_tensors["train_holdout"]["current_pred"],
        cluster_count=int(cluster_count),
        label_names=label_names,
    )
    allowed_kp, stability_rows = _stable_cluster_penalty_mask(
        fit_stats=fit_stats,
        holdout_stats=holdout_stats,
        label_names=label_names,
        min_support=int(args.min_support),
        min_exact_precision=float(args.min_exact_precision),
        min_exact_recall=float(args.min_exact_recall),
        max_precision_gap=float(args.max_precision_gap),
    )
    score_threshold_kp: Optional[torch.Tensor] = None
    score_feature = str(args.score_feature or "").strip()
    score_feature_idx: Optional[int] = None
    score_quantiles = _parse_quantiles(str(args.score_quantiles))
    if score_feature:
        feature_names = list(split_tensors["train"]["feature_names"])  # type: ignore[index]
        allowed_kp, score_threshold_kp, stability_rows = _select_score_threshold_guard(
            train_fit=split_tensors["train"],
            train_holdout=split_tensors["train_holdout"],
            cluster_count=int(cluster_count),
            label_names=label_names,
            feature_names=feature_names,
            score_feature=score_feature,
            quantiles=score_quantiles,
            min_support=int(args.min_support),
            min_exact_precision=float(args.min_exact_precision),
            min_exact_recall=float(args.min_exact_recall),
            max_precision_gap=float(args.max_precision_gap),
        )
        score_feature_idx = _score_feature_index(feature_names, score_feature)
    predictions: Dict[str, torch.Tensor] = {}
    split_metrics: Dict[str, Dict[str, object]] = {}
    for split, tensors in split_tensors.items():
        if score_feature and score_threshold_kp is not None and score_feature_idx is not None:
            pred = _apply_cluster_penalty_score_guard(
                tensors["current_pred"],
                features=tensors["features"],
                cluster_count=int(cluster_count),
                allowed_kp=allowed_kp,
                score_threshold_kp=score_threshold_kp,
                score_feature_idx=int(score_feature_idx),
            )
        else:
            pred = _apply_cluster_penalty_guard(
                tensors["current_pred"],
                cluster_count=int(cluster_count),
                allowed_kp=allowed_kp,
            )
        predictions[split] = pred
        split_metrics[split] = _route_metrics_from_predictions(
            pred=pred,
            labels=tensors["labels"],
            current_pred=tensors["current_pred"],
            label_names=label_names,
        )
    allowed_count = int(allowed_kp.sum().item())
    split_rows = [_split_row(split, metrics) for split, metrics in split_metrics.items()]
    payload = {
        "tensors_dir": str(tensors_dir),
        "out_dir": str(args.out_dir),
        "no_test_read": True,
        "cluster_count": int(cluster_count),
        "label_names": label_names,
        "guard_config": {
            "min_support": int(args.min_support),
            "min_exact_precision": float(args.min_exact_precision),
            "min_exact_recall": float(args.min_exact_recall),
            "max_precision_gap": float(args.max_precision_gap),
            "score_feature": score_feature or None,
            "score_quantiles": score_quantiles if score_feature else None,
        },
        "allowed_kp": allowed_kp,
        "score_threshold_kp": score_threshold_kp,
        "stability_rows": stability_rows,
        "splits": split_metrics,
        "split_rows": split_rows,
        "verdict": _classify_guard_result(split_metrics, allowed_count),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(predictions, out_dir / "binary_adoption_predictions.pt")
    torch.save({"allowed_kp": allowed_kp, "label_names": label_names, "stability_rows": stability_rows}, out_dir / "split_stable_guard.pt")
    refit_payload = {
        "tensors_dir": str(tensors_dir),
        "route_feature_mode": refit_summary.get("route_feature_mode", "shape_proxy") if isinstance(refit_summary, dict) else "shape_proxy",
        "no_test_read": True,
        "guard_source": "current_pred",
        "guard_mode": "score_threshold" if score_feature else "class_level",
    }
    (out_dir / "binary_adoption_refit.json").write_text(json.dumps(refit_payload, indent=2), encoding="utf-8")
    (out_dir / "split_stable_adoption_guard.json").write_text(
        json.dumps(_to_jsonable(payload), indent=2),
        encoding="utf-8",
    )
    (out_dir / "split_stable_adoption_guard.md").write_text(_markdown_report(_to_jsonable(payload)), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="NEXT-11d offline split-stable adoption guard.")
    parser.add_argument("--tensors-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cluster-count", type=int, default=0)
    parser.add_argument("--min-support", type=int, default=16)
    parser.add_argument("--min-exact-precision", type=float, default=0.50)
    parser.add_argument("--min-exact-recall", type=float, default=0.01)
    parser.add_argument("--max-precision-gap", type=float, default=0.25)
    parser.add_argument("--score-feature", type=str, default="")
    parser.add_argument("--score-quantiles", type=str, default="0.50,0.75,0.90,0.95,0.98")
    args = parser.parse_args()
    payload = run_guard(args)
    verdict = payload["verdict"]
    val = payload["splits"]["val"]  # type: ignore[index]
    print(
        "allowed={} val_acc={:.4f} val_majority={:.4f} val_skip={:.4f} failure_layer={} decision={} no_test_read=True".format(
            int(verdict["allowed_cluster_penalty_count"]),
            float(val["accuracy_all"]),
            float(val["majority_accuracy_all"]),
            float(val["head_skip_rate"]),
            str(verdict["failure_layer"]),
            str(verdict["decision"]),
        )
    )


if __name__ == "__main__":
    main()
