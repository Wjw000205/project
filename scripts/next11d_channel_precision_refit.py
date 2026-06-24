from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.next11c_route_accuracy_diagnostic import _build_anchor_artifacts, _restore_cluster_penalty_prior
from scripts.next11d_binary_adoption_refit import (
    _binary_examples_for_penalty,
    _candidate_threshold_values,
    _fit_one_binary_head,
    _route_from_binary_scores,
    _route_metrics_from_predictions,
    _score_with_head,
    _threshold_combinations,
    _to_jsonable,
)
from scripts.next11d_candidate_utility_stability import _candidate_gain_by_channel_penalty
from scripts.next11d_channel_action_space_diagnostic import (
    _allowed_mask_by_channel,
    _channel_oracle_labels_from_gain,
    _forecast_metrics_from_channel_labels,
    _normalize_requested_splits,
)
from scripts.next11d_fixed_candidate_router_refit import _read_data_for_cfg
from scripts.shape_prior_diagnostic import _build_modules, _compute_penalty_scale, _make_loaders
from src.models.penalties import build_penalty_bank
from src.train import _collect_pred_residual_selector_tensors
from src.utils.seed import set_seed
from src.utils.yaml_io import load_yaml


def _channel_feature_table(*, skip_feat_bcf: torch.Tensor, cand_feat_bcpf: torch.Tensor) -> torch.Tensor:
    if skip_feat_bcf.dim() != 3:
        raise ValueError("skip_feat_bcf must have shape [B,C,F].")
    if cand_feat_bcpf.dim() != 4:
        raise ValueError("cand_feat_bcpf must have shape [B,C,P,F].")
    B, C, F = [int(v) for v in skip_feat_bcf.shape]
    if tuple(cand_feat_bcpf.shape[:2]) != (B, C) or int(cand_feat_bcpf.shape[-1]) != F:
        raise ValueError("cand_feat_bcpf must share [B,C,F] with skip_feat_bcf.")
    skip = skip_feat_bcf.detach().cpu().to(dtype=torch.float32).reshape(B * C, 1, F)
    cand = cand_feat_bcpf.detach().cpu().to(dtype=torch.float32).reshape(B * C, int(cand_feat_bcpf.shape[2]), F)
    return torch.cat([skip, cand], dim=1)


def _select_precision_thresholds(
    *,
    scores_np: torch.Tensor,
    labels: torch.Tensor,
    label_names: List[str],
    precision_floor: float,
    min_recall: float,
    max_thresholds_per_penalty: int,
    max_threshold_combinations: int,
    min_apply_rate: float,
    gain_np: Optional[torch.Tensor] = None,
    selection_objective: str = "recall",
) -> Tuple[torch.Tensor, Dict[str, object]]:
    scores = scores_np.detach().cpu().to(dtype=torch.float32)
    labels_flat = labels.detach().cpu().to(dtype=torch.long).view(-1)
    if scores.dim() != 2:
        raise ValueError("scores_np must have shape [N,P].")
    if int(scores.shape[0]) != int(labels_flat.numel()):
        raise ValueError("scores_np and labels must share N.")
    P = int(scores.shape[1])
    gain = None
    if gain_np is not None:
        gain = gain_np.detach().cpu().to(dtype=torch.float32)
        if gain.dim() != 2 or tuple(gain.shape) != tuple(scores.shape):
            raise ValueError("gain_np must have shape [N,P] matching scores_np.")
    objective = str(selection_objective).lower()
    if objective not in {"recall", "utility_gain"}:
        raise ValueError("selection_objective must be 'recall' or 'utility_gain'.")
    grids = [_candidate_threshold_values(scores[:, p], int(max_thresholds_per_penalty)) for p in range(P)]
    best_thresholds: Optional[torch.Tensor] = None
    best_metrics: Optional[Dict[str, object]] = None
    best_key: Optional[Tuple[float, float, float, float]] = None
    fallback_thresholds: Optional[torch.Tensor] = None
    fallback_metrics: Optional[Dict[str, object]] = None
    fallback_key: Optional[Tuple[float, float, float, float]] = None
    best_gain_mean = -float("inf")
    candidates_seen = 0
    valid_candidates_seen = 0
    for combo in _threshold_combinations(grids, int(max_threshold_combinations)):
        thresholds = torch.tensor(combo, dtype=torch.float32)
        pred = _route_from_binary_scores(scores, thresholds)
        metrics = _route_metrics_from_predictions(
            pred=pred,
            labels=labels_flat,
            current_pred=None,
            label_names=label_names,
        )
        candidates_seen += 1
        precision = float(metrics.get("positive_precision_any", 0.0) or 0.0)
        recall = float(metrics.get("positive_recall_any", 0.0) or 0.0)
        accuracy = float(metrics.get("accuracy_all", 0.0) or 0.0)
        apply_rate = max(0.0, 1.0 - float(metrics.get("head_skip_rate", 0.0) or 0.0))
        gain_mean = 0.0
        if gain is not None:
            selected_gain = torch.zeros(int(pred.numel()), dtype=torch.float32)
            for penalty_class in range(1, P + 1):
                mask = pred == penalty_class
                if bool(mask.any().item()):
                    selected_gain[mask] = gain[:, penalty_class - 1][mask]
            gain_mean = float(selected_gain.mean().item()) if int(selected_gain.numel()) > 0 else 0.0
        if objective == "utility_gain":
            key = (gain_mean, recall, precision, accuracy)
            fallback_rank = (gain_mean, precision, recall, accuracy)
        else:
            key = (recall, precision, accuracy, apply_rate)
            fallback_rank = (precision, recall, accuracy, apply_rate)
        if fallback_key is None or fallback_rank > fallback_key:
            fallback_key = fallback_rank
            fallback_thresholds = thresholds
            fallback_metrics = metrics
        if precision + 1.0e-12 < float(precision_floor):
            continue
        if recall + 1.0e-12 < float(min_recall):
            continue
        if apply_rate + 1.0e-12 < float(min_apply_rate):
            continue
        valid_candidates_seen += 1
        if best_key is None or key > best_key:
            best_key = key
            best_thresholds = thresholds
            best_metrics = metrics
            best_gain_mean = float(gain_mean)
    used_fallback = False
    if best_thresholds is None or best_metrics is None:
        used_fallback = True
        best_thresholds = fallback_thresholds if fallback_thresholds is not None else torch.full((P,), 1.0)
        best_metrics = fallback_metrics if fallback_metrics is not None else _route_metrics_from_predictions(
            pred=_route_from_binary_scores(scores, best_thresholds),
            labels=labels_flat,
            current_pred=None,
            label_names=label_names,
        )
        if gain is not None:
            pred = _route_from_binary_scores(scores, best_thresholds)
            selected_gain = torch.zeros(int(pred.numel()), dtype=torch.float32)
            for penalty_class in range(1, P + 1):
                mask = pred == penalty_class
                if bool(mask.any().item()):
                    selected_gain[mask] = gain[:, penalty_class - 1][mask]
            best_gain_mean = float(selected_gain.mean().item()) if int(selected_gain.numel()) > 0 else 0.0
    return best_thresholds, {
        "selection_objective": objective,
        "precision_floor": float(precision_floor),
        "min_recall": float(min_recall),
        "min_apply_rate": float(min_apply_rate),
        "threshold_grids": grids,
        "candidates_seen": int(candidates_seen),
        "valid_candidates_seen": int(valid_candidates_seen),
        "used_fallback": bool(used_fallback),
        "best_gain_mean": float(best_gain_mean),
        "best_metrics": best_metrics,
    }


def _score_all_penalties(
    *,
    features_nqf: torch.Tensor,
    artifacts: Dict[int, Optional[Dict[str, torch.Tensor]]],
    hidden_dim: int,
    dropout: float,
    device: torch.device,
) -> torch.Tensor:
    P = int(features_nqf.shape[1]) - 1
    scores = []
    for penalty_class in range(1, P + 1):
        scores.append(
            _score_with_head(
                artifact=artifacts.get(penalty_class),
                features_nqf=features_nqf,
                penalty_class=penalty_class,
                hidden_dim=int(hidden_dim),
                dropout=float(dropout),
                device=device,
            )
        )
    return torch.stack(scores, dim=1) if scores else torch.zeros(int(features_nqf.shape[0]), 0)


def _prediction_forecast_metrics(
    *,
    tensors: Dict[str, torch.Tensor],
    pred_flat: torch.Tensor,
) -> Dict[str, object]:
    B, C = [int(v) for v in tensors["base"].shape[:2]]
    labels_bc = pred_flat.detach().cpu().to(dtype=torch.long).view(B, C)
    return _forecast_metrics_from_channel_labels(
        base_bch=tensors["base"],
        cand_bcpH=tensors["cand"],
        y_bch=tensors["y"],
        labels_bc=labels_bc,
    )


def _label_summary(labels: torch.Tensor, label_names: List[str]) -> Dict[str, object]:
    flat = labels.detach().cpu().to(dtype=torch.long).view(-1).clamp(0, len(label_names) - 1)
    samples = int(flat.numel())
    counts = torch.bincount(flat, minlength=len(label_names))[: len(label_names)]
    return {
        "samples": samples,
        "label_counts": {name: int(counts[i].item()) for i, name in enumerate(label_names)},
        "label_rates": {name: float(counts[i].item() / max(samples, 1)) for i, name in enumerate(label_names)},
        "positive_rate": float((flat > 0).to(dtype=torch.float32).mean().item()) if samples else 0.0,
    }


def _classify(payload: Dict[str, object]) -> Dict[str, object]:
    splits = payload.get("splits", {}) or {}
    holdout = splits.get("train_holdout", {}) if isinstance(splits, dict) else {}
    val = splits.get("val", {}) if isinstance(splits, dict) else {}
    h_metrics = (holdout.get("route_metrics", {}) or {}) if isinstance(holdout, dict) else {}
    v_metrics = (val.get("route_metrics", {}) or {}) if isinstance(val, dict) else {}
    h_precision = float(h_metrics.get("positive_precision_any", 0.0) or 0.0)
    h_recall = float(h_metrics.get("positive_recall_any", 0.0) or 0.0)
    v_precision = float(v_metrics.get("positive_precision_any", 0.0) or 0.0)
    v_recall = float(v_metrics.get("positive_recall_any", 0.0) or 0.0)
    v_forecast = (val.get("forecast_metrics", {}) or {}) if isinstance(val, dict) else {}
    val_gain = float(v_forecast.get("selected_gain_pct_vs_base", 0.0) or 0.0)
    selection = payload.get("selection", {}) or {}
    precision_floor = float(selection.get("precision_floor", 0.0) or 0.0)
    used_fallback = bool(selection.get("used_fallback", False))
    if used_fallback or h_precision + 1.0e-12 < precision_floor:
        failure_layer = "selection/adoption policy"
        decision = "precision_floor_not_reached_on_train_holdout"
    elif v_precision + 0.05 < h_precision:
        failure_layer = "train-val utility shift"
        decision = "precision_guard_train_holdout_to_val_shift"
    elif val_gain <= 0.0:
        failure_layer = "selection/adoption policy"
        decision = "precision_guard_no_forecast_gain"
    else:
        failure_layer = "selection/adoption policy"
        decision = "precision_guard_viable_offline"
    return {
        "failure_layer": failure_layer,
        "decision": decision,
        "train_holdout_precision": h_precision,
        "train_holdout_recall": h_recall,
        "val_precision": v_precision,
        "val_recall": v_recall,
        "val_forecast_gain_pct": val_gain,
    }


def _markdown_report(payload: Dict[str, object]) -> str:
    lines = [
        "# NEXT-11d Channel Precision Refit",
        "",
        f"- config: `{payload['config_path']}`",
        f"- checkpoint: `{payload['checkpoint_path']}`",
        f"- no_test_read: `{payload['no_test_read']}`",
        f"- failure_layer: `{payload['verdict']['failure_layer']}`",
        f"- decision: `{payload['verdict']['decision']}`",
        "",
        "## Split Metrics",
        "",
        "| split | acc | majority | precision | recall | skip | forecast gain | mae gain | label positive |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, row in (payload.get("splits", {}) or {}).items():
        metrics = row["route_metrics"]
        forecast = row["forecast_metrics"]
        label_summary = row["label_summary"]
        lines.append(
            "| {split} | {acc:.4f} | {maj:.4f} | {precision:.4f} | {recall:.4f} | {skip:.4f} | {gain:.3f}% | {mae_gain:.3f}% | {positive:.4f} |".format(
                split=split,
                acc=float(metrics["accuracy_all"]),
                maj=float(metrics["majority_accuracy_all"]),
                precision=float(metrics["positive_precision_any"]),
                recall=float(metrics["positive_recall_any"]),
                skip=float(metrics["head_skip_rate"]),
                gain=float(forecast["selected_gain_pct_vs_base"]),
                mae_gain=float(forecast["selected_mae_gain_pct_vs_base"]),
                positive=float(label_summary["positive_rate"]),
            )
        )
    lines.extend(
        [
            "",
            "## Selection",
            "",
            f"- selection_objective: `{payload['selection'].get('selection_objective', 'recall')}`",
            f"- precision_floor: `{payload['selection']['precision_floor']}`",
            f"- min_recall: `{payload['selection']['min_recall']}`",
            f"- thresholds: `{payload['selection']['thresholds']}`",
            f"- used_fallback: `{payload['selection']['used_fallback']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> Dict[str, object]:
    cfg = load_yaml(str(args.config))
    cfg.setdefault("eval", {})["skip_test"] = True
    splits = _normalize_requested_splits(args.splits)
    if "train_fit" not in splits or "train_holdout" not in splits:
        raise ValueError("channel precision refit requires train_fit and train_holdout.")
    set_seed(int(cfg["exp"]["seed"]), deterministic=bool((cfg.get("exp", {}) or {}).get("deterministic", False)))
    requested_device = str(args.device or cfg.get("exp", {}).get("device", "cpu"))
    device = torch.device(requested_device if torch.cuda.is_available() and requested_device != "cpu" else "cpu")
    checkpoint = torch.load(str(args.checkpoint), map_location=device)
    model, gate, pred_residual, cluster_id_c, K, moe_cfg, penalty_names = _build_modules(cfg, checkpoint, device)
    data_tc = _read_data_for_cfg(cfg)
    batch_size = int(cfg.get("train", {}).get("batch_size", 64))
    data_window_tc, loaders, eval_starts, train_loader, window_meta = _make_loaders(cfg, data_tc, batch_size=batch_size)
    penalty_fns = build_penalty_bank(penalty_names, jump_thr=float(cfg.get("penalties", {}).get("jump_threshold", 0.6)))
    penalty_scale = _compute_penalty_scale(train_loader, penalty_names, penalty_fns, int(window_meta["H"]), device)
    anchor = _build_anchor_artifacts(
        cfg=cfg,
        checkpoint=checkpoint,
        model=model,
        cluster_id_c=cluster_id_c,
        data_tc=data_window_tc,
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
    allowed_mask_kp = None
    if prior_summary.get("allowed_mask") is not None:
        allowed_mask_kp = torch.as_tensor(prior_summary["allowed_mask"], dtype=torch.bool)
    label_names = ["skip"] + [str(name) for name in penalty_names]
    tensors_by_split: Dict[str, Dict[str, torch.Tensor]] = {}
    features_by_split: Dict[str, torch.Tensor] = {}
    labels_by_split: Dict[str, torch.Tensor] = {}
    gain_by_split: Dict[str, torch.Tensor] = {}
    for split in splits:
        tensors = _collect_pred_residual_selector_tensors(
            model=model,
            pred_residual=pred_residual,
            loader=loaders[split],
            cluster_id_c=cluster_id_c,
            K=int(K),
            moe_cfg=moe_cfg,
            device=device,
            penalty_count=len(penalty_names),
            history_anchor_cfg=anchor["history_anchor_cfg"],
            observed_history_tc=data_window_tc,
            input_len=int(window_meta["L"]),
            eval_start=int(eval_starts[split]),
            model_train_stat_adapter_pc=anchor["model_train_stat_adapter_pc"],
            model_train_stat_adapter_cfg=anchor["model_train_stat_adapter_cfg"],
            train_stat_anchor_pc=anchor["train_stat_anchor_pc"],
            train_residual_anchor_phc=anchor["train_residual_anchor_phc"],
            candidate_feature_mode=str(args.candidate_feature_mode),
        )
        if tensors is None:
            raise RuntimeError(f"Could not collect candidate tensors for split {split}.")
        gain_bcp = _candidate_gain_by_channel_penalty(
            base_bch=tensors["base"],
            cand_bcpH=tensors["cand"],
            y_bch=tensors["y"],
        )
        allowed_cp = _allowed_mask_by_channel(
            allowed_mask_kp=allowed_mask_kp,
            cluster_id_c=cluster_id_c.detach().cpu(),
            channel_count=int(gain_bcp.shape[1]),
            penalty_count=int(gain_bcp.shape[2]),
        )
        labels_bc = _channel_oracle_labels_from_gain(
            gain_bcp,
            allowed_cp=allowed_cp,
            margin=float(args.margin),
        )
        tensors_by_split[split] = tensors
        labels_by_split[split] = labels_bc.reshape(-1)
        gain_by_split[split] = gain_bcp.reshape(-1, int(gain_bcp.shape[-1]))
        features_by_split[split] = _channel_feature_table(
            skip_feat_bcf=tensors["skip_feat"],
            cand_feat_bcpf=tensors["cand_feat"],
        )

    head_cfg = {
        "epochs": int(args.epochs),
        "batch_size": int(args.head_batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "class_weight": str(args.class_weight),
        "class_weight_max": float(args.class_weight_max),
        "patience": int(args.patience),
        "min_delta": float(args.min_delta),
    }
    artifacts: Dict[int, Optional[Dict[str, torch.Tensor]]] = {}
    head_summaries: Dict[str, object] = {}
    for penalty_class, penalty_name in enumerate(penalty_names, start=1):
        train_x, train_y, _ = _binary_examples_for_penalty(
            features_by_split["train_fit"],
            labels_by_split["train_fit"],
            penalty_class=penalty_class,
            ignore_other_positive=True,
        )
        holdout_x, holdout_y, _ = _binary_examples_for_penalty(
            features_by_split["train_holdout"],
            labels_by_split["train_holdout"],
            penalty_class=penalty_class,
            ignore_other_positive=True,
        )
        artifact, summary = _fit_one_binary_head(
            train_x=train_x,
            train_y=train_y,
            holdout_x=holdout_x,
            holdout_y=holdout_y,
            cfg=head_cfg,
            device=device,
            seed=int(args.seed) + penalty_class,
        )
        artifacts[penalty_class] = artifact
        head_summaries[str(penalty_name)] = summary

    scores_by_split = {
        split: _score_all_penalties(
            features_nqf=features,
            artifacts=artifacts,
            hidden_dim=int(args.hidden_dim),
            dropout=float(args.dropout),
            device=device,
        )
        for split, features in features_by_split.items()
    }
    thresholds, selection_summary = _select_precision_thresholds(
        scores_np=scores_by_split["train_holdout"],
        labels=labels_by_split["train_holdout"],
        label_names=label_names,
        precision_floor=float(args.precision_floor),
        min_recall=float(args.min_recall),
        max_thresholds_per_penalty=int(args.max_thresholds_per_penalty),
        max_threshold_combinations=int(args.max_threshold_combinations),
        min_apply_rate=float(args.min_apply_rate),
        gain_np=gain_by_split["train_holdout"],
        selection_objective=str(args.selection_objective),
    )
    split_payloads: Dict[str, object] = {}
    predictions: Dict[str, torch.Tensor] = {}
    for split in splits:
        pred = _route_from_binary_scores(scores_by_split[split], thresholds)
        predictions[split] = pred.detach().cpu()
        split_payloads[split] = {
            "label_summary": _label_summary(labels_by_split[split], label_names),
            "route_metrics": _route_metrics_from_predictions(
                pred=pred,
                labels=labels_by_split[split],
                current_pred=None,
                label_names=label_names,
            ),
            "forecast_metrics": _prediction_forecast_metrics(
                tensors=tensors_by_split[split],
                pred_flat=pred,
            ),
        }
    payload = {
        "config_path": str(args.config),
        "checkpoint_path": str(args.checkpoint),
        "out_dir": str(args.out_dir),
        "device": str(device),
        "no_test_read": True,
        "splits_collected": splits,
        "criteria": {
            "margin": float(args.margin),
            "precision_floor": float(args.precision_floor),
            "min_recall": float(args.min_recall),
        },
        "penalty_names": list(penalty_names),
        "label_names": label_names,
        "head_config": head_cfg,
        "head_summaries": head_summaries,
        "selection": {
            "split": "train_holdout",
            "thresholds": [float(v) for v in thresholds.tolist()],
            **selection_summary,
        },
        "splits": split_payloads,
        "prior_summary": prior_summary,
    }
    payload["verdict"] = _classify(payload)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "artifacts": artifacts,
            "thresholds": thresholds.detach().cpu(),
            "head_config": head_cfg,
            "label_names": label_names,
            "penalty_names": list(penalty_names),
            "predictions": predictions,
        },
        out_dir / "channel_precision_refit.pt",
    )
    (out_dir / "channel_precision_refit.json").write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")
    (out_dir / "channel_precision_refit.md").write_text(_markdown_report(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="NEXT-11d channel-level precision refit diagnostic.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--splits", nargs="*", default=["train_fit", "train_holdout", "val"])
    parser.add_argument("--candidate-feature-mode", type=str, default="shape_proxy")
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--precision-floor", type=float, default=0.80)
    parser.add_argument("--min-recall", type=float, default=0.20)
    parser.add_argument("--min-apply-rate", type=float, default=0.01)
    parser.add_argument("--selection-objective", type=str, default="recall", choices=["recall", "utility_gain"])
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--head-batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--class-weight", type=str, default="balanced")
    parser.add_argument("--class-weight-max", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1.0e-4)
    parser.add_argument("--max-thresholds-per-penalty", type=int, default=21)
    parser.add_argument("--max-threshold-combinations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1104)
    args = parser.parse_args()
    payload = run(args)
    verdict = payload["verdict"]
    val = payload["splits"].get("val", {})
    metrics = val.get("route_metrics", {}) if isinstance(val, dict) else {}
    print(
        "failure_layer={} decision={} holdout_precision={:.4f} val_precision={:.4f} val_recall={:.4f} val_gain={:.3f} no_test_read=True".format(
            str(verdict["failure_layer"]),
            str(verdict["decision"]),
            float(verdict["train_holdout_precision"]),
            float(metrics.get("positive_precision_any", 0.0) or 0.0),
            float(metrics.get("positive_recall_any", 0.0) or 0.0),
            float(verdict["val_forecast_gain_pct"]),
        )
    )


if __name__ == "__main__":
    main()
