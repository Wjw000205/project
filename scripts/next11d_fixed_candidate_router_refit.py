from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.next11c_route_accuracy_diagnostic import _build_anchor_artifacts, _restore_cluster_penalty_prior
from scripts.shape_prior_diagnostic import _build_modules, _compute_penalty_scale, _make_loaders
from src.data.reader import read_csv_time_series
from src.models.penalties import build_penalty_bank
from src.train import (
    _collect_penalty_route_learnability_tensors,
    _fit_penalty_route_learnability_head_from_tensors,
    _normalize_gate_feature_mode,
)
from src.utils.seed import set_seed
from src.utils.yaml_io import load_yaml


def _read_data_for_cfg(cfg: Dict[str, object]) -> torch.Tensor:
    data_cfg = cfg["data"]
    data_tc, _ = read_csv_time_series(str(data_cfg["csv_path"]), date_col=int(data_cfg.get("date_col", 0)))
    return data_tc.detach().cpu()


def _split_metric(summary: Dict[str, object], split: str, key: str, default: float = 0.0) -> float:
    splits = summary.get("splits", {}) or {}
    split_payload = splits.get(split, {}) if isinstance(splits, dict) else {}
    if not isinstance(split_payload, dict):
        return float(default)
    try:
        return float(split_payload.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _classify_fixed_candidate_refit(
    summary: Dict[str, object],
    *,
    min_train_accuracy: float = 0.70,
) -> Dict[str, object]:
    train_acc = _split_metric(summary, "train", "accuracy_all")
    train_maj = _split_metric(summary, "train", "majority_accuracy_all")
    holdout_acc = _split_metric(summary, "train_holdout", "accuracy_all")
    holdout_maj = _split_metric(summary, "train_holdout", "majority_accuracy_all")
    val_acc = _split_metric(summary, "val", "accuracy_all")
    val_maj = _split_metric(summary, "val", "majority_accuracy_all")
    train_pass = bool(train_acc >= float(min_train_accuracy))
    holdout_lift = float(holdout_acc - holdout_maj)
    val_lift = float(val_acc - val_maj)
    if not train_pass:
        failure_layer = "gate feature insufficiency"
        decision = "fixed_candidate_router_refit_failed_train_sanity"
    elif holdout_lift <= 0.0 or val_lift <= 0.0:
        failure_layer = "train-val utility shift"
        decision = "fixed_candidate_router_refit_overfits_train"
    else:
        failure_layer = "selection/adoption policy"
        decision = "fixed_candidate_router_refit_passes_route_sanity"
    return {
        "min_train_accuracy": float(min_train_accuracy),
        "train_accuracy": float(train_acc),
        "train_majority_accuracy": float(train_maj),
        "train_lift_vs_majority": float(train_acc - train_maj),
        "train_accuracy_sanity_pass": train_pass,
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


def _count_rate(counts: Dict[str, object], label: str, samples: int) -> float:
    try:
        return float(counts.get(label, 0)) / max(int(samples), 1)
    except (AttributeError, TypeError, ValueError):
        return 0.0


def _split_row(split: str, metrics: Dict[str, object]) -> Dict[str, object]:
    samples = int(metrics.get("samples", 0) or 0)
    label_counts = metrics.get("label_counts", {}) or {}
    pred_counts = metrics.get("prediction_counts", {}) or {}
    current_counts = metrics.get("current_prediction_counts", {}) or {}
    return {
        "split": split,
        "samples": samples,
        "head_accuracy": float(metrics.get("accuracy_all", 0.0) or 0.0),
        "current_accuracy": float(metrics.get("current_accuracy_all", 0.0) or 0.0),
        "majority_accuracy": float(metrics.get("majority_accuracy_all", 0.0) or 0.0),
        "head_lift_vs_majority": float(metrics.get("lift_vs_majority", 0.0) or 0.0),
        "head_lift_vs_current": float(metrics.get("lift_vs_current", 0.0) or 0.0),
        "oracle_skip_rate": _count_rate(label_counts, "skip", samples),
        "head_skip_rate": _count_rate(pred_counts, "skip", samples),
        "current_skip_rate": _count_rate(current_counts, "skip", samples),
        "label_counts": label_counts,
        "head_prediction_counts": pred_counts,
        "current_prediction_counts": current_counts,
    }


def _route_head_cfg_from_args(args: argparse.Namespace) -> Dict[str, object]:
    cfg: Dict[str, object] = {
        "epochs": int(args.epochs),
        "batch_size": int(args.head_batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "head_mode": str(args.head_mode),
        "class_weight": str(args.class_weight),
        "selection_split": str(args.selection_split),
        "selection_metric": str(args.selection_metric),
        "early_stop_patience": int(args.patience),
        "early_stop_min_delta": float(args.min_delta),
        "init_bias": str(args.init_bias),
        "include_initial_eval": True,
        "seed": int(args.seed),
    }
    if float(args.class_weight_max) > 0.0:
        cfg["class_weight_max"] = float(args.class_weight_max)
    return cfg


def _route_label_cfg_from_args(args: argparse.Namespace) -> Dict[str, float]:
    return {
        "min_abs_improvement": float(args.min_abs_improvement),
        "min_rel_improvement": float(args.min_rel_improvement),
        "min_candidate_delta_rms": float(args.min_candidate_delta_rms),
    }


def _markdown_report(payload: Dict[str, object]) -> str:
    lines = [
        "# NEXT-11d Fixed-Candidate Router Refit",
        "",
        f"- config: `{payload['config_path']}`",
        f"- checkpoint: `{payload['checkpoint_path']}`",
        f"- no_test_read: `{payload['no_test_read']}`",
        f"- failure_layer: `{payload['verdict']['failure_layer']}`",
        f"- decision: `{payload['verdict']['decision']}`",
        "",
        "## Split Metrics",
        "",
        "| split | samples | head acc | current acc | majority | head lift | oracle skip | head skip | current skip |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("split_rows", []):
        lines.append(
            "| {split} | {samples} | {head:.4f} | {current:.4f} | {majority:.4f} | {lift:.4f} | {oracle_skip:.4f} | {head_skip:.4f} | {current_skip:.4f} |".format(
                split=row["split"],
                samples=int(row["samples"]),
                head=float(row["head_accuracy"]),
                current=float(row["current_accuracy"]),
                majority=float(row["majority_accuracy"]),
                lift=float(row["head_lift_vs_majority"]),
                oracle_skip=float(row["oracle_skip_rate"]),
                head_skip=float(row["head_skip_rate"]),
                current_skip=float(row["current_skip_rate"]),
            )
        )
    return "\n".join(lines) + "\n"


def run_refit(args: argparse.Namespace) -> Dict[str, object]:
    cfg = load_yaml(str(args.config))
    cfg.setdefault("eval", {})
    cfg["eval"]["skip_test"] = True
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
        allowed_mask_kp = torch.as_tensor(prior_summary["allowed_mask"], device=device, dtype=torch.bool)
    raw_ranks = moe_cfg.get("select_ranks", None)
    select_ranks = [1, 2] if raw_ranks is None else [int(v) for v in raw_ranks]
    gate_soft_weight = float(moe_cfg.get("gate_soft_weight", 0.0))
    gate_feature_mode = _normalize_gate_feature_mode(
        str(checkpoint["meta"].get("gate_feature_mode", moe_cfg.get("gate_feature_mode", "history")))
    )
    route_feature_mode = str(args.route_feature_mode)
    route_label_cfg = _route_label_cfg_from_args(args)
    max_batches = int(args.max_batches)
    split_tensors: Dict[str, Dict[str, torch.Tensor]] = {}
    for split in ("train_fit", "train_holdout", "val"):
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
            feature_mode=route_feature_mode,
            allowed_mask_kp=allowed_mask_kp,
            min_abs_improvement=float(route_label_cfg["min_abs_improvement"]),
            min_rel_improvement=float(route_label_cfg["min_rel_improvement"]),
            min_candidate_delta_rms=float(route_label_cfg["min_candidate_delta_rms"]),
            max_batches=max_batches,
            history_anchor_cfg=anchor["history_anchor_cfg"],
            observed_history_tc=data_window_tc,
            input_len=int(window_meta["L"]),
            eval_start=int(eval_starts[split]),
            model_train_stat_adapter_pc=anchor["model_train_stat_adapter_pc"],
            model_train_stat_adapter_cfg=anchor["model_train_stat_adapter_cfg"],
            train_stat_anchor_pc=anchor["train_stat_anchor_pc"],
            train_residual_anchor_phc=anchor["train_residual_anchor_phc"],
            gate_feature_mode=gate_feature_mode,
        )
        if tensors is None:
            raise RuntimeError(f"Could not collect route tensors for split {split}.")
        split_tensors[split] = tensors
    label_names = list(split_tensors["train_fit"]["label_names"])  # type: ignore[index]
    feature_names = list(split_tensors["train_fit"]["feature_names"])  # type: ignore[index]
    head_cfg = _route_head_cfg_from_args(args)
    summary, artifact = _fit_penalty_route_learnability_head_from_tensors(
        train_tensors=split_tensors["train_fit"],
        eval_tensors_by_split={
            "train_holdout": split_tensors["train_holdout"],
            "val": split_tensors["val"],
        },
        label_names=label_names,
        feature_names=feature_names,
        cfg=head_cfg,
        device=device,
    )
    verdict = _classify_fixed_candidate_refit(summary, min_train_accuracy=float(args.min_train_accuracy))
    split_rows = [
        _split_row(split, metrics)
        for split, metrics in (summary.get("splits", {}) or {}).items()
        if isinstance(metrics, dict)
    ]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, out_dir / "fixed_candidate_router_head.pt")
    for split, tensors in split_tensors.items():
        torch.save(tensors, out_dir / f"fixed_candidate_route_tensors_{split}.pt")
    payload = {
        "config_path": str(args.config),
        "checkpoint_path": str(args.checkpoint),
        "out_dir": str(out_dir),
        "device": str(device),
        "no_test_read": True,
        "splits_collected": ["train_fit", "train_holdout", "val"],
        "max_batches": int(max_batches),
        "route_feature_mode": route_feature_mode,
        "route_label_config": route_label_cfg,
        "head_config": head_cfg,
        "route_context": {
            "penalty_names": penalty_names,
            "label_names": label_names,
            "feature_names": feature_names,
            "cluster_count": int(K),
            "allow_skip": bool(moe_cfg.get("allow_skip", False)),
            "skip_competes_with_penalties": bool(moe_cfg.get("skip_competes_with_penalties", False)),
            "allowed_mask": prior_summary.get("allowed_mask"),
            "prior_restored": prior_summary,
            "gate_feature_mode": gate_feature_mode,
        },
        "summary": summary,
        "split_rows": split_rows,
        "verdict": verdict,
    }
    (out_dir / "fixed_candidate_router_refit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "fixed_candidate_router_refit.md").write_text(_markdown_report(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="NEXT-11d fixed-candidate router refit diagnostic.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--route-feature-mode", type=str, default="base")
    parser.add_argument("--min-abs-improvement", type=float, default=0.0)
    parser.add_argument("--min-rel-improvement", type=float, default=0.0)
    parser.add_argument("--min-candidate-delta-rms", type=float, default=0.0)
    parser.add_argument("--head-mode", type=str, default="classwise")
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min-delta", type=float, default=1.0e-4)
    parser.add_argument("--head-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--class-weight", type=str, default="balanced")
    parser.add_argument("--class-weight-max", type=float, default=0.0)
    parser.add_argument("--selection-metric", type=str, default="accuracy")
    parser.add_argument("--selection-split", type=str, default="train_holdout", choices=["train", "train_holdout"])
    parser.add_argument("--init-bias", type=str, default="none")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-train-accuracy", type=float, default=0.70)
    args = parser.parse_args()
    payload = run_refit(args)
    verdict = payload["verdict"]
    print(
        "train_acc={:.4f} holdout_acc={:.4f} val_acc={:.4f} failure_layer={} decision={}".format(
            float(verdict["train_accuracy"]),
            float(verdict["holdout_accuracy"]),
            float(verdict["val_accuracy"]),
            str(verdict["failure_layer"]),
            str(verdict["decision"]),
        )
    )


if __name__ == "__main__":
    main()
