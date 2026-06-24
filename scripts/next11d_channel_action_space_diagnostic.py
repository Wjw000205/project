from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.next11c_route_accuracy_diagnostic import _build_anchor_artifacts, _restore_cluster_penalty_prior
from scripts.next11d_binary_adoption_forecast_eval import _oracle_channel_metrics
from scripts.next11d_binary_adoption_refit import _forecast_metrics_from_route_predictions, _to_jsonable
from scripts.next11d_candidate_utility_stability import _candidate_gain_by_channel_penalty
from scripts.next11d_fixed_candidate_router_refit import _read_data_for_cfg
from scripts.shape_prior_diagnostic import _build_modules, _compute_penalty_scale, _make_loaders
from src.models.penalties import build_penalty_bank
from src.train import _collect_pred_residual_selector_tensors
from src.utils.seed import set_seed
from src.utils.yaml_io import load_yaml


def _normalize_requested_splits(raw_splits: Iterable[str]) -> List[str]:
    allowed = {"train_fit", "train_holdout", "val"}
    splits: List[str] = []
    for raw in raw_splits:
        split = str(raw).strip().lower()
        if split == "train":
            split = "train_fit"
        if split == "test":
            raise ValueError("channel action-space diagnostic refuses to read test.")
        if split not in allowed:
            raise ValueError(f"unsupported split {raw!r}; expected train_fit, train_holdout, or val.")
        if split not in splits:
            splits.append(split)
    return splits or ["train_fit", "train_holdout", "val"]


def _channel_oracle_labels_from_gain(
    gain_bcp: torch.Tensor,
    *,
    allowed_cp: Optional[torch.Tensor] = None,
    margin: float = 0.0,
) -> torch.Tensor:
    if gain_bcp.dim() != 3:
        raise ValueError("gain_bcp must have shape [B,C,P].")
    B, C, P = [int(v) for v in gain_bcp.shape]
    gain = gain_bcp.detach().cpu().to(dtype=torch.float32)
    if allowed_cp is not None:
        allowed = allowed_cp.detach().cpu().to(dtype=torch.bool)
        if tuple(allowed.shape) != (C, P):
            raise ValueError("allowed_cp must have shape [C,P].")
        gain = gain.masked_fill(~allowed.view(1, C, P), float("-inf"))
    best_gain, best_idx = gain.max(dim=-1)
    labels = best_idx.to(dtype=torch.long) + 1
    return torch.where(best_gain > float(margin), labels, torch.zeros_like(labels))


def _cluster_projection_from_channel_labels(
    *,
    labels_bc: torch.Tensor,
    cluster_id_c: torch.Tensor,
    cluster_count: int,
    label_count: int,
    mode: str = "majority",
) -> torch.Tensor:
    if labels_bc.dim() != 2:
        raise ValueError("labels_bc must have shape [B,C].")
    B, C = [int(v) for v in labels_bc.shape]
    cluster_id = cluster_id_c.detach().cpu().to(dtype=torch.long).view(-1)
    if int(cluster_id.numel()) != C:
        raise ValueError("cluster_id_c must have one id per channel.")
    if int(label_count) <= 0:
        raise ValueError("label_count must be positive.")
    mode = str(mode)
    if mode not in {"majority", "positive_first"}:
        raise ValueError("mode must be 'majority' or 'positive_first'.")
    labels = labels_bc.detach().cpu().to(dtype=torch.long).clamp(0, int(label_count) - 1)
    route = torch.zeros(B, int(cluster_count), dtype=torch.long)
    for cluster in range(int(cluster_count)):
        channels = torch.nonzero(cluster_id == cluster, as_tuple=False).view(-1)
        if int(channels.numel()) <= 0:
            continue
        vals_bm = labels.index_select(1, channels)
        for b in range(B):
            vals = vals_bm[b]
            if mode == "positive_first":
                pos = vals[vals > 0]
                if int(pos.numel()) > 0:
                    counts = torch.bincount(pos, minlength=int(label_count))[: int(label_count)]
                    route[b, cluster] = int(torch.argmax(counts).item())
                    continue
            counts = torch.bincount(vals, minlength=int(label_count))[: int(label_count)]
            route[b, cluster] = int(torch.argmax(counts).item())
    return route


def _channel_labels_from_cluster_route(
    *,
    route_bk: torch.Tensor,
    cluster_id_c: torch.Tensor,
    channel_count: int,
    label_count: int,
) -> torch.Tensor:
    route = route_bk.detach().cpu().to(dtype=torch.long)
    if route.dim() != 2:
        raise ValueError("route_bk must have shape [B,K].")
    B, K = [int(v) for v in route.shape]
    cluster_id = cluster_id_c.detach().cpu().to(dtype=torch.long).view(-1)
    if int(cluster_id.numel()) != int(channel_count):
        raise ValueError("cluster_id_c must have one id per channel.")
    if int(cluster_id.min().item()) < 0 or int(cluster_id.max().item()) >= K:
        raise ValueError("cluster_id_c contains cluster outside route_bk.")
    return route.index_select(1, cluster_id).clamp(0, int(label_count) - 1).reshape(B, int(channel_count))


def _label_count_dict(values: torch.Tensor, label_names: List[str]) -> Dict[str, int]:
    clean = values.detach().cpu().to(dtype=torch.long).view(-1).clamp(0, len(label_names) - 1)
    counts = torch.bincount(clean, minlength=len(label_names))[: len(label_names)]
    return {str(name): int(counts[i].item()) for i, name in enumerate(label_names)}


def _safe_rate(num: int, denom: int) -> float:
    return float(num) / max(int(denom), 1)


def _channel_label_route_metrics(
    *,
    labels_bc: torch.Tensor,
    route_bk: torch.Tensor,
    cluster_id_c: torch.Tensor,
    label_names: List[str],
) -> Dict[str, object]:
    if labels_bc.dim() != 2:
        raise ValueError("labels_bc must have shape [B,C].")
    label_count = int(len(label_names))
    labels = labels_bc.detach().cpu().to(dtype=torch.long).clamp(0, label_count - 1)
    B, C = [int(v) for v in labels.shape]
    pred = _channel_labels_from_cluster_route(
        route_bk=route_bk,
        cluster_id_c=cluster_id_c,
        channel_count=C,
        label_count=label_count,
    )
    labels_flat = labels.reshape(-1)
    pred_flat = pred.reshape(-1)
    samples = int(labels_flat.numel())
    label_counts = torch.bincount(labels_flat, minlength=label_count)[:label_count]
    majority = int(label_counts.max().item()) if samples > 0 else 0
    oracle_positive = labels_flat > 0
    pred_positive = pred_flat > 0
    oracle_skip = labels_flat == 0
    pred_skip = pred_flat == 0
    tp_any = int((oracle_positive & pred_positive).sum().item())
    pred_pos = int(pred_positive.sum().item())
    oracle_pos = int(oracle_positive.sum().item())
    per_penalty: Dict[str, object] = {}
    for label_idx, name in enumerate(label_names[1:], start=1):
        oracle_p = labels_flat == label_idx
        pred_p = pred_flat == label_idx
        tp = int((oracle_p & pred_p).sum().item())
        per_penalty[str(name)] = {
            "oracle_count": int(oracle_p.sum().item()),
            "pred_count": int(pred_p.sum().item()),
            "recall": _safe_rate(tp, int(oracle_p.sum().item())),
            "precision": _safe_rate(tp, int(pred_p.sum().item())),
        }
    return {
        "samples": samples,
        "accuracy_all": _safe_rate(int((pred_flat == labels_flat).sum().item()), samples),
        "majority_accuracy_all": _safe_rate(majority, samples),
        "positive_recall_any": _safe_rate(tp_any, oracle_pos),
        "positive_precision_any": _safe_rate(tp_any, pred_pos),
        "oracle_skip_rate": _safe_rate(int(oracle_skip.sum().item()), samples),
        "route_skip_rate": _safe_rate(int(pred_skip.sum().item()), samples),
        "oracle_skip_routed_to_penalty_rate": _safe_rate(int((oracle_skip & pred_positive).sum().item()), int(oracle_skip.sum().item())),
        "oracle_penalty_routed_to_skip_rate": _safe_rate(int((oracle_positive & pred_skip).sum().item()), oracle_pos),
        "label_counts": _label_count_dict(labels_flat, label_names),
        "prediction_counts": _label_count_dict(pred_flat, label_names),
        "per_penalty": per_penalty,
    }


def _channel_conflict_metrics(
    *,
    labels_bc: torch.Tensor,
    cluster_id_c: torch.Tensor,
    cluster_count: int,
    label_names: List[str],
) -> Dict[str, object]:
    if labels_bc.dim() != 2:
        raise ValueError("labels_bc must have shape [B,C].")
    labels = labels_bc.detach().cpu().to(dtype=torch.long).clamp(0, len(label_names) - 1)
    B, C = [int(v) for v in labels.shape]
    cluster_id = cluster_id_c.detach().cpu().to(dtype=torch.long).view(-1)
    if int(cluster_id.numel()) != C:
        raise ValueError("cluster_id_c must have one id per channel.")
    total = 0
    mixed_skip_positive = 0
    multi_positive_penalty = 0
    nonuniform = 0
    best_count_sum = 0
    channel_decisions = 0
    for cluster in range(int(cluster_count)):
        channels = torch.nonzero(cluster_id == cluster, as_tuple=False).view(-1)
        if int(channels.numel()) <= 0:
            continue
        vals_bm = labels.index_select(1, channels)
        for b in range(B):
            vals = vals_bm[b]
            total += 1
            unique = torch.unique(vals)
            if int(unique.numel()) > 1:
                nonuniform += 1
            positive = vals[vals > 0]
            if int(positive.numel()) > 0 and int(positive.numel()) < int(vals.numel()):
                mixed_skip_positive += 1
            if int(torch.unique(positive).numel()) > 1:
                multi_positive_penalty += 1
            counts = torch.bincount(vals, minlength=len(label_names))[: len(label_names)]
            best_count_sum += int(counts.max().item())
            channel_decisions += int(vals.numel())
    return {
        "cluster_samples": int(total),
        "nonuniform_label_rate": _safe_rate(nonuniform, total),
        "mixed_skip_positive_rate": _safe_rate(mixed_skip_positive, total),
        "multi_positive_penalty_rate": _safe_rate(multi_positive_penalty, total),
        "best_single_cluster_label_channel_accuracy_ceiling": _safe_rate(best_count_sum, channel_decisions),
    }


def _forecast_metrics_from_channel_labels(
    *,
    base_bch: torch.Tensor,
    cand_bcpH: torch.Tensor,
    y_bch: torch.Tensor,
    labels_bc: torch.Tensor,
) -> Dict[str, object]:
    base = base_bch.detach().cpu().to(dtype=torch.float32)
    cand = cand_bcpH.detach().cpu().to(dtype=torch.float32)
    y = y_bch.detach().cpu().to(dtype=torch.float32)
    labels = labels_bc.detach().cpu().to(dtype=torch.long)
    if base.dim() != 3 or cand.dim() != 4 or y.dim() != 3:
        raise ValueError("base/y must be [B,C,H] and cand must be [B,C,P,H].")
    B, C, H = [int(v) for v in base.shape]
    if tuple(y.shape) != (B, C, H) or tuple(cand.shape[:2]) != (B, C) or int(cand.shape[-1]) != H:
        raise ValueError("base, cand, and y shapes do not align.")
    selected = base.clone()
    for p in range(1, int(cand.shape[2]) + 1):
        mask = labels == p
        if bool(mask.any().item()):
            selected = torch.where(mask.unsqueeze(-1), cand[:, :, p - 1, :], selected)
    base_mse = float((base - y).pow(2).mean().item())
    base_mae = float((base - y).abs().mean().item())
    selected_mse = float((selected - y).pow(2).mean().item())
    selected_mae = float((selected - y).abs().mean().item())
    return {
        "base_mse": base_mse,
        "base_mae": base_mae,
        "selected_mse": selected_mse,
        "selected_mae": selected_mae,
        "selected_gain_pct_vs_base": float(100.0 * (base_mse - selected_mse) / max(abs(base_mse), 1.0e-12)),
        "selected_mae_gain_pct_vs_base": float(100.0 * (base_mae - selected_mae) / max(abs(base_mae), 1.0e-12)),
        "candidate_use_rate_channel": float((labels > 0).to(dtype=torch.float32).mean().item()) if labels.numel() else 0.0,
    }


def _markdown_report(payload: Dict[str, object]) -> str:
    lines = [
        "# NEXT-11d Channel Action-Space Diagnostic",
        "",
        f"- config: `{payload['config_path']}`",
        f"- checkpoint: `{payload['checkpoint_path']}`",
        f"- no_test_read: `{payload['no_test_read']}`",
        f"- failure_layer: `{payload['verdict']['failure_layer']}`",
        f"- decision: `{payload['verdict']['decision']}`",
        "",
        "## Split Summary",
        "",
        "| split | channel positive | oracle skip | mixed skip/penalty | multi-penalty conflict | majority acc/recall/precision | positive-first acc/recall/precision | channel oracle gain | majority forecast gain | positive-first forecast gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, row in (payload.get("splits", {}) or {}).items():
        if not isinstance(row, dict):
            continue
        majority = row["projection_metrics"]["majority"]
        positive_first = row["projection_metrics"]["positive_first"]
        conflict = row["conflict_metrics"]
        channel_oracle = row["forecast_metrics"]["channel_label_oracle"]
        majority_forecast = row["forecast_metrics"]["cluster_majority_projection"]
        positive_forecast = row["forecast_metrics"]["cluster_positive_first_projection"]
        lines.append(
            "| {split} | {positive:.4f} | {skip:.4f} | {mix:.4f} | {multi:.4f} | {maj_acc:.4f}/{maj_rec:.4f}/{maj_prec:.4f} | {pos_acc:.4f}/{pos_rec:.4f}/{pos_prec:.4f} | {chan_gain:.3f}% | {maj_gain:.3f}% | {pos_gain:.3f}% |".format(
                split=split,
                positive=float(row["label_summary"]["positive_rate"]),
                skip=float(row["label_summary"]["skip_rate"]),
                mix=float(conflict["mixed_skip_positive_rate"]),
                multi=float(conflict["multi_positive_penalty_rate"]),
                maj_acc=float(majority["accuracy_all"]),
                maj_rec=float(majority["positive_recall_any"]),
                maj_prec=float(majority["positive_precision_any"]),
                pos_acc=float(positive_first["accuracy_all"]),
                pos_rec=float(positive_first["positive_recall_any"]),
                pos_prec=float(positive_first["positive_precision_any"]),
                chan_gain=float(channel_oracle["selected_gain_pct_vs_base"]),
                maj_gain=float(majority_forecast["selected_gain_pct_vs_base"]),
                pos_gain=float(positive_forecast["selected_gain_pct_vs_base"]),
            )
        )
    lines.extend(["", "## Verdict", "", str(payload["verdict"])])
    return "\n".join(lines) + "\n"


def _label_summary(labels_bc: torch.Tensor, label_names: List[str]) -> Dict[str, object]:
    labels = labels_bc.detach().cpu().to(dtype=torch.long).view(-1).clamp(0, len(label_names) - 1)
    samples = int(labels.numel())
    counts = torch.bincount(labels, minlength=len(label_names))[: len(label_names)]
    return {
        "samples": samples,
        "label_counts": {str(name): int(counts[i].item()) for i, name in enumerate(label_names)},
        "label_rates": {str(name): _safe_rate(int(counts[i].item()), samples) for i, name in enumerate(label_names)},
        "skip_rate": _safe_rate(int(counts[0].item()), samples) if len(label_names) > 0 else 0.0,
        "positive_rate": _safe_rate(int((labels > 0).sum().item()), samples),
        "majority_accuracy_all": _safe_rate(int(counts.max().item()) if samples > 0 else 0, samples),
    }


def _allowed_mask_by_channel(
    *,
    allowed_mask_kp: Optional[torch.Tensor],
    cluster_id_c: torch.Tensor,
    channel_count: int,
    penalty_count: int,
) -> Optional[torch.Tensor]:
    if allowed_mask_kp is None:
        return None
    allowed = allowed_mask_kp.detach().cpu().to(dtype=torch.bool)
    if allowed.dim() != 2 or int(allowed.shape[1]) != int(penalty_count):
        raise ValueError("allowed_mask_kp must have shape [K,P].")
    cluster_id = cluster_id_c.detach().cpu().to(dtype=torch.long).view(-1)
    if int(cluster_id.numel()) != int(channel_count):
        raise ValueError("cluster_id_c must have one id per channel.")
    return allowed.index_select(0, cluster_id)


def _classify(payload_splits: Dict[str, object]) -> Dict[str, object]:
    train_holdout = payload_splits.get("train_holdout", {}) if isinstance(payload_splits, dict) else {}
    val = payload_splits.get("val", {}) if isinstance(payload_splits, dict) else {}
    th_conflict = float(((train_holdout or {}).get("conflict_metrics", {}) or {}).get("mixed_skip_positive_rate", 0.0))
    th_majority = ((train_holdout or {}).get("projection_metrics", {}) or {}).get("majority", {}) or {}
    th_positive = ((train_holdout or {}).get("projection_metrics", {}) or {}).get("positive_first", {}) or {}
    val_forecasts = ((val or {}).get("forecast_metrics", {}) or {})
    channel_gain = float((val_forecasts.get("channel_label_oracle", {}) or {}).get("selected_gain_pct_vs_base", 0.0))
    majority_gain = float((val_forecasts.get("cluster_majority_projection", {}) or {}).get("selected_gain_pct_vs_base", 0.0))
    positive_gain = float((val_forecasts.get("cluster_positive_first_projection", {}) or {}).get("selected_gain_pct_vs_base", 0.0))
    if th_conflict >= 0.50:
        failure_layer = "routing target mismatch"
        decision = "cluster_action_space_forces_recall_precision_tradeoff"
    elif channel_gain > max(majority_gain, positive_gain) + 0.25:
        failure_layer = "routing target mismatch"
        decision = "channel_level_action_space_has_oracle_headroom"
    elif float(th_majority.get("positive_recall_any", 0.0)) < 0.50 and float(th_positive.get("positive_precision_any", 0.0)) < 0.50:
        failure_layer = "selection/adoption policy"
        decision = "channel_label_recall_precision_tradeoff_unresolved"
    else:
        failure_layer = "adapter candidate quality"
        decision = "channel_action_space_not_clearly_supported"
    return {
        "failure_layer": failure_layer,
        "decision": decision,
        "train_holdout_mixed_skip_positive_rate": th_conflict,
        "train_holdout_majority_positive_recall": float(th_majority.get("positive_recall_any", 0.0)),
        "train_holdout_positive_first_precision": float(th_positive.get("positive_precision_any", 0.0)),
        "val_channel_oracle_gain_pct": channel_gain,
        "val_cluster_majority_projection_gain_pct": majority_gain,
        "val_cluster_positive_first_projection_gain_pct": positive_gain,
    }


def run(args: argparse.Namespace) -> Dict[str, object]:
    cfg = load_yaml(str(args.config))
    cfg.setdefault("eval", {})["skip_test"] = True
    splits = _normalize_requested_splits(args.splits)
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
    split_payloads: Dict[str, object] = {}
    label_names = ["skip"] + [str(name) for name in penalty_names]
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
        majority_route = _cluster_projection_from_channel_labels(
            labels_bc=labels_bc,
            cluster_id_c=cluster_id_c.detach().cpu(),
            cluster_count=int(K),
            label_count=len(label_names),
            mode="majority",
        )
        positive_first_route = _cluster_projection_from_channel_labels(
            labels_bc=labels_bc,
            cluster_id_c=cluster_id_c.detach().cpu(),
            cluster_count=int(K),
            label_count=len(label_names),
            mode="positive_first",
        )
        split_payloads[split] = {
            "label_summary": _label_summary(labels_bc, label_names),
            "conflict_metrics": _channel_conflict_metrics(
                labels_bc=labels_bc,
                cluster_id_c=cluster_id_c.detach().cpu(),
                cluster_count=int(K),
                label_names=label_names,
            ),
            "projection_metrics": {
                "majority": _channel_label_route_metrics(
                    labels_bc=labels_bc,
                    route_bk=majority_route,
                    cluster_id_c=cluster_id_c.detach().cpu(),
                    label_names=label_names,
                ),
                "positive_first": _channel_label_route_metrics(
                    labels_bc=labels_bc,
                    route_bk=positive_first_route,
                    cluster_id_c=cluster_id_c.detach().cpu(),
                    label_names=label_names,
                ),
            },
            "forecast_metrics": {
                "channel_label_oracle": _forecast_metrics_from_channel_labels(
                    base_bch=tensors["base"],
                    cand_bcpH=tensors["cand"],
                    y_bch=tensors["y"],
                    labels_bc=labels_bc,
                ),
                "channel_best_oracle_margin0": _oracle_channel_metrics(tensors),
                "cluster_majority_projection": _forecast_metrics_from_route_predictions(
                    base_bch=tensors["base"],
                    cand_bcpH=tensors["cand"],
                    y_bch=tensors["y"],
                    cluster_id_c=cluster_id_c.detach().cpu(),
                    route_pred_bk=majority_route,
                ),
                "cluster_positive_first_projection": _forecast_metrics_from_route_predictions(
                    base_bch=tensors["base"],
                    cand_bcpH=tensors["cand"],
                    y_bch=tensors["y"],
                    cluster_id_c=cluster_id_c.detach().cpu(),
                    route_pred_bk=positive_first_route,
                ),
            },
        }
    payload = {
        "config_path": str(args.config),
        "checkpoint_path": str(args.checkpoint),
        "out_dir": str(args.out_dir),
        "device": str(device),
        "no_test_read": True,
        "splits_collected": splits,
        "criteria": {"margin": float(args.margin)},
        "penalty_names": list(penalty_names),
        "label_names": label_names,
        "cluster_count": int(K),
        "prior_summary": prior_summary,
        "splits": split_payloads,
    }
    payload["verdict"] = _classify(split_payloads)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "channel_action_space_diagnostic.json").write_text(
        json.dumps(_to_jsonable(payload), indent=2),
        encoding="utf-8",
    )
    (out_dir / "channel_action_space_diagnostic.md").write_text(_markdown_report(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="NEXT-11d channel action-space diagnostic.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--splits", nargs="*", default=["train_fit", "train_holdout", "val"])
    parser.add_argument("--candidate-feature-mode", type=str, default="shape_proxy")
    parser.add_argument("--margin", type=float, default=0.0)
    args = parser.parse_args()
    payload = run(args)
    verdict = payload["verdict"]
    print(
        "failure_layer={} decision={} val_channel_gain={:.3f} val_majority_projection_gain={:.3f} no_test_read=True".format(
            str(verdict["failure_layer"]),
            str(verdict["decision"]),
            float(verdict["val_channel_oracle_gain_pct"]),
            float(verdict["val_cluster_majority_projection_gain_pct"]),
        )
    )


if __name__ == "__main__":
    main()
