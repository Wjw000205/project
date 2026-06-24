from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.next11c_route_accuracy_diagnostic import _build_anchor_artifacts, _restore_cluster_penalty_prior
from scripts.shape_prior_diagnostic import _build_modules, _compute_penalty_scale, _make_loaders
from scripts.next11d_fixed_candidate_router_refit import _read_data_for_cfg
from src.models.penalties import build_penalty_bank
from src.train import (
    _build_gate_routing_features,
    _normalize_gate_feature_mode,
    _pred_residual_candidates_on_eval_path,
    _router_penalty_context_from_history,
    _select_rank_mask,
    apply_history_anchor_adapter,
    apply_train_stat_anchor_expert,
    apply_train_stat_input_centering,
)
from src.utils.seed import set_seed
from src.utils.yaml_io import load_yaml


def _reject_test_splits(splits: Iterable[str]) -> None:
    for split in splits:
        if str(split).strip().lower() == "test":
            raise ValueError("No test read is allowed for NEXT-11d skip-zero diagnostics.")


def _cluster_best_penalty_gain_and_labels(
    *,
    base_err_bc: torch.Tensor,
    cand_err_bcp: torch.Tensor,
    cluster_id_c: torch.Tensor,
    K: int,
    allowed_mask_kp: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if base_err_bc.dim() != 2:
        raise ValueError("base_err_bc must have shape [B,C].")
    if cand_err_bcp.dim() != 3:
        raise ValueError("cand_err_bcp must have shape [B,C,P].")
    if tuple(cand_err_bcp.shape[:2]) != tuple(base_err_bc.shape):
        raise ValueError("cand_err_bcp and base_err_bc must share [B,C].")
    B, _, P = [int(v) for v in cand_err_bcp.shape]
    cid_c = cluster_id_c.detach().to(device=base_err_bc.device, dtype=torch.long).view(-1)
    if int(cid_c.numel()) != int(base_err_bc.shape[1]):
        raise ValueError("cluster_id_c length must match channel count.")
    allowed = None
    if allowed_mask_kp is not None:
        allowed = allowed_mask_kp.detach().to(device=base_err_bc.device, dtype=torch.bool)
        if tuple(allowed.shape) != (int(K), int(P)):
            raise ValueError("allowed_mask_kp must have shape [K,P].")

    gains = torch.full((B, int(K)), float("nan"), device=base_err_bc.device, dtype=base_err_bc.dtype)
    labels = torch.zeros(B, int(K), device=base_err_bc.device, dtype=torch.long)
    for k in range(int(K)):
        ch_mask = cid_c == int(k)
        if not bool(ch_mask.any().item()):
            continue
        cluster_base = base_err_bc[:, ch_mask].mean(dim=1)
        cluster_penalty = cand_err_bcp[:, ch_mask, :].mean(dim=1)
        if allowed is not None:
            cluster_penalty = cluster_penalty.masked_fill(~allowed[int(k)].view(1, P), float("inf"))
        best_err, best_p = cluster_penalty.min(dim=-1)
        finite = torch.isfinite(best_err)
        gain = cluster_base - best_err
        gains[:, int(k)] = torch.where(finite, gain, torch.full_like(gain, float("nan")))
        penalty_label = best_p.to(dtype=torch.long) + 1
        labels[:, int(k)] = torch.where(finite & (gain > 0.0), penalty_label, torch.zeros_like(penalty_label))
    return gains, labels


def _cluster_penalty_gain_and_delta(
    *,
    base_err_bc: torch.Tensor,
    cand_err_bcp: torch.Tensor,
    delta_rms_bcp: torch.Tensor,
    cluster_id_c: torch.Tensor,
    K: int,
    allowed_mask_kp: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if base_err_bc.dim() != 2:
        raise ValueError("base_err_bc must have shape [B,C].")
    if cand_err_bcp.dim() != 3 or delta_rms_bcp.dim() != 3:
        raise ValueError("cand_err_bcp and delta_rms_bcp must have shape [B,C,P].")
    if tuple(cand_err_bcp.shape) != tuple(delta_rms_bcp.shape):
        raise ValueError("cand_err_bcp and delta_rms_bcp must share shape [B,C,P].")
    if tuple(cand_err_bcp.shape[:2]) != tuple(base_err_bc.shape):
        raise ValueError("candidate tensors and base_err_bc must share [B,C].")
    B, _, P = [int(v) for v in cand_err_bcp.shape]
    cid_c = cluster_id_c.detach().to(device=base_err_bc.device, dtype=torch.long).view(-1)
    if int(cid_c.numel()) != int(base_err_bc.shape[1]):
        raise ValueError("cluster_id_c length must match channel count.")
    allowed = None
    if allowed_mask_kp is not None:
        allowed = allowed_mask_kp.detach().to(device=base_err_bc.device, dtype=torch.bool)
        if tuple(allowed.shape) != (int(K), int(P)):
            raise ValueError("allowed_mask_kp must have shape [K,P].")
    gains = torch.full((B, int(K), P), float("nan"), device=base_err_bc.device, dtype=base_err_bc.dtype)
    deltas = torch.full((B, int(K), P), float("nan"), device=base_err_bc.device, dtype=delta_rms_bcp.dtype)
    for k in range(int(K)):
        ch_mask = cid_c == int(k)
        if not bool(ch_mask.any().item()):
            continue
        cluster_base = base_err_bc[:, ch_mask].mean(dim=1)
        cluster_penalty = cand_err_bcp[:, ch_mask, :].mean(dim=1)
        cluster_delta = delta_rms_bcp[:, ch_mask, :].mean(dim=1)
        if allowed is not None:
            allowed_k = allowed[int(k)].view(1, P)
            cluster_penalty = cluster_penalty.masked_fill(~allowed_k, float("nan"))
            cluster_delta = cluster_delta.masked_fill(~allowed_k, float("nan"))
        gains[:, int(k), :] = cluster_base.unsqueeze(-1) - cluster_penalty
        deltas[:, int(k), :] = cluster_delta
    return gains, deltas


def _mean_or_none(values: torch.Tensor) -> Optional[float]:
    if int(values.numel()) <= 0:
        return None
    return float(values.to(dtype=torch.float32).mean().item())


def _quantiles(values: torch.Tensor, probs: Sequence[float]) -> Dict[str, Optional[float]]:
    if int(values.numel()) <= 0:
        return {f"p{int(p * 100):02d}": None for p in probs}
    v = values.to(dtype=torch.float32)
    return {f"p{int(p * 100):02d}": float(v.quantile(float(p)).item()) for p in probs}


def _per_penalty_support_summary(
    *,
    penalty_gain_bkp: torch.Tensor,
    penalty_delta_rms_bkp: torch.Tensor,
    penalty_names: Sequence[str],
    strong_threshold: float = 1.0e-3,
    near_zero_threshold: float = 1.0e-3,
) -> Dict[str, Dict[str, object]]:
    if penalty_gain_bkp.dim() != 3 or penalty_delta_rms_bkp.dim() != 3:
        raise ValueError("penalty_gain_bkp and penalty_delta_rms_bkp must have shape [B,K,P].")
    if tuple(penalty_gain_bkp.shape) != tuple(penalty_delta_rms_bkp.shape):
        raise ValueError("penalty gain and delta tensors must share shape.")
    P = int(penalty_gain_bkp.shape[-1])
    if len(penalty_names) != P:
        raise ValueError("penalty_names length must match tensor P.")
    out: Dict[str, Dict[str, object]] = {}
    gain = penalty_gain_bkp.detach().cpu().to(dtype=torch.float32)
    delta = penalty_delta_rms_bkp.detach().cpu().to(dtype=torch.float32)
    strong = float(strong_threshold)
    near = float(near_zero_threshold)
    for p, name in enumerate(penalty_names):
        gain_p = gain[..., p].reshape(-1)
        delta_p = delta[..., p].reshape(-1)
        valid = torch.isfinite(gain_p) & torch.isfinite(delta_p)
        gain_v = gain_p[valid]
        delta_v = delta_p[valid]
        samples = int(gain_v.numel())
        if samples <= 0:
            out[str(name)] = {
                "samples": 0,
                "gain_mean": None,
                "positive_rate": None,
                "strong_positive_rate": None,
                "strong_negative_rate": None,
                "near_zero_rate": None,
                "delta_rms_mean": None,
                "delta_rms_p95": None,
            }
            continue
        out[str(name)] = {
            "samples": samples,
            "gain_mean": float(gain_v.mean().item()),
            "gain_quantiles": _quantiles(gain_v, [0.05, 0.50, 0.95]),
            "positive_rate": float((gain_v > 0.0).to(dtype=torch.float32).mean().item()),
            "strong_positive_rate": float((gain_v > strong).to(dtype=torch.float32).mean().item()),
            "strong_negative_rate": float((gain_v < -strong).to(dtype=torch.float32).mean().item()),
            "near_zero_rate": float((gain_v.abs() <= near).to(dtype=torch.float32).mean().item()),
            "delta_rms_mean": float(delta_v.mean().item()),
            "delta_rms_p95": float(delta_v.quantile(0.95).item()),
        }
    return out


def _best_penalty_gain_and_labels_from_penalty_tensors(
    *,
    penalty_gain_bkp: torch.Tensor,
    penalty_delta_rms_bkp: torch.Tensor,
    min_delta_rms: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if penalty_gain_bkp.dim() != 3 or penalty_delta_rms_bkp.dim() != 3:
        raise ValueError("penalty_gain_bkp and penalty_delta_rms_bkp must have shape [B,K,P].")
    if tuple(penalty_gain_bkp.shape) != tuple(penalty_delta_rms_bkp.shape):
        raise ValueError("penalty gain and delta tensors must share shape.")
    min_delta = float(min_delta_rms)
    valid = torch.isfinite(penalty_gain_bkp) & torch.isfinite(penalty_delta_rms_bkp)
    if min_delta > 0.0:
        valid = valid & (penalty_delta_rms_bkp >= min_delta)
    gain_for_select = penalty_gain_bkp.masked_fill(~valid, float("-inf"))
    best_gain, best_p = gain_for_select.max(dim=-1)
    has_actionable = torch.isfinite(best_gain)
    best_gain = torch.where(has_actionable, best_gain, torch.zeros_like(best_gain))
    penalty_label = best_p.to(dtype=torch.long) + 1
    labels = torch.where(has_actionable & (best_gain > 0.0), penalty_label, torch.zeros_like(penalty_label))
    return best_gain, labels, has_actionable


def _skip_margin_summary(
    *,
    best_penalty_gain_bk: torch.Tensor,
    labels_bk: torch.Tensor,
    current_pred_bk: Optional[torch.Tensor] = None,
    near_zero_thresholds: Sequence[float] = (1.0e-4, 5.0e-4, 1.0e-3, 5.0e-3),
) -> Dict[str, object]:
    gain = best_penalty_gain_bk.detach().cpu().to(dtype=torch.float32).reshape(-1)
    labels = labels_bk.detach().cpu().to(dtype=torch.long).reshape(-1)
    valid = torch.isfinite(gain) & (labels >= 0)
    gain = gain[valid]
    labels = labels[valid]
    samples = int(labels.numel())
    if samples <= 0:
        return {
            "samples": 0,
            "oracle_skip_rate": 0.0,
            "actual_skip_rate": None,
            "near_zero_abs_gain_rates": {f"{float(thr):g}": 0.0 for thr in near_zero_thresholds},
        }
    oracle_skip = labels == 0
    oracle_penalty = labels > 0
    out: Dict[str, object] = {
        "samples": samples,
        "best_penalty_gain_mean": float(gain.mean().item()),
        "best_penalty_gain_quantiles": _quantiles(gain, [0.05, 0.25, 0.50, 0.75, 0.95]),
        "oracle_skip_rate": float(oracle_skip.to(dtype=torch.float32).mean().item()),
        "oracle_penalty_rate": float(oracle_penalty.to(dtype=torch.float32).mean().item()),
        "oracle_skip_gain_mean": _mean_or_none(gain[oracle_skip]),
        "oracle_skip_gain_quantiles": _quantiles(gain[oracle_skip], [0.05, 0.50, 0.95]),
        "oracle_penalty_gain_mean": _mean_or_none(gain[oracle_penalty]),
        "oracle_penalty_gain_quantiles": _quantiles(gain[oracle_penalty], [0.05, 0.50, 0.95]),
        "near_zero_abs_gain_rates": {
            f"{float(thr):g}": float((gain.abs() <= float(thr)).to(dtype=torch.float32).mean().item())
            for thr in near_zero_thresholds
        },
        "near_zero_oracle_skip_rates": {
            f"{float(thr):g}": float(((gain.abs() <= float(thr)) & oracle_skip).to(dtype=torch.float32).mean().item())
            for thr in near_zero_thresholds
        },
        "near_zero_oracle_penalty_rates": {
            f"{float(thr):g}": float(((gain.abs() <= float(thr)) & oracle_penalty).to(dtype=torch.float32).mean().item())
            for thr in near_zero_thresholds
        },
    }
    if current_pred_bk is not None:
        current = current_pred_bk.detach().cpu().to(dtype=torch.long).reshape(-1)[valid]
        actual_skip = current == 0
        correct_skip = oracle_skip & actual_skip
        oracle_skip_count = int(oracle_skip.sum().item())
        actual_skip_count = int(actual_skip.sum().item())
        out.update(
            {
                "actual_skip_rate": float(actual_skip.to(dtype=torch.float32).mean().item()),
                "skip_recall": float(correct_skip.sum().item() / max(oracle_skip_count, 1)),
                "skip_precision": float(correct_skip.sum().item() / max(actual_skip_count, 1)),
                "oracle_skip_routed_to_penalty_rate": float((oracle_skip & ~actual_skip).sum().item() / max(oracle_skip_count, 1)),
                "oracle_penalty_routed_to_skip_rate": float((oracle_penalty & actual_skip).sum().item() / max(int(oracle_penalty.sum().item()), 1)),
            }
        )
    else:
        out["actual_skip_rate"] = None
    return out


@torch.no_grad()
def _collect_split_margin(
    *,
    model: torch.nn.Module,
    gate: torch.nn.Module,
    pred_residual: Optional[torch.nn.Module],
    loader,
    cluster_id_c: torch.Tensor,
    K: int,
    moe_cfg: Dict[str, object],
    device: torch.device,
    penalty_names: List[str],
    penalty_fns: Dict[str, object],
    penalty_scale: Optional[torch.Tensor],
    select_ranks: Optional[List[int]],
    gate_soft_weight: float,
    allowed_mask_kp: Optional[torch.Tensor],
    max_batches: int,
    history_anchor_cfg: Optional[dict],
    observed_history_tc: Optional[torch.Tensor],
    input_len: int,
    eval_start: int,
    model_train_stat_adapter_pc: Optional[torch.Tensor],
    model_train_stat_adapter_cfg: Optional[dict],
    train_stat_anchor_pc: Optional[torch.Tensor],
    train_residual_anchor_phc: Optional[torch.Tensor],
    gate_feature_mode: str,
) -> Dict[str, torch.Tensor]:
    if pred_residual is None:
        raise RuntimeError("skip-zero diagnostic requires pred_residual candidates.")
    model.eval()
    gate.eval()
    pred_residual.eval()
    allow_skip = bool(moe_cfg.get("allow_skip", False))
    router_mode = str(moe_cfg.get("router_mode", "learned")).lower()
    router_penalty_context_weight = float(moe_cfg.get("router_penalty_context_weight", 0.0))
    router_detach_penalty_context = bool(moe_cfg.get("router_detach_penalty_context", True))
    router_penalty_context_score = str(moe_cfg.get("router_penalty_context_score", "high_violation")).lower()
    cid_c = cluster_id_c.detach().to(device=device, dtype=torch.long)
    allowed = None if allowed_mask_kp is None else allowed_mask_kp.detach().to(device=device, dtype=torch.bool)
    gate_feature_mode = _normalize_gate_feature_mode(gate_feature_mode)

    gain_chunks: List[torch.Tensor] = []
    penalty_gain_chunks: List[torch.Tensor] = []
    penalty_delta_chunks: List[torch.Tensor] = []
    label_chunks: List[torch.Tensor] = []
    current_chunks: List[torch.Tensor] = []
    skip_prob_chunks: List[torch.Tensor] = []
    batch_count = 0
    for x, y, idx in loader:
        batch_count += 1
        if int(max_batches) > 0 and batch_count > int(max_batches):
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
        pred_out = pred_residual(x, y_base, cluster_id_c, mask_bkp, skip_bk=skip_bk if allow_skip else None)
        y_base_final, cand_bcpH = _pred_residual_candidates_on_eval_path(
            y_base,
            pred_out,
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
        base_err_bc = (y_base_final - y).pow(2).mean(dim=-1)
        cand_err_bcp = (cand_bcpH - y.unsqueeze(2)).pow(2).mean(dim=-1)
        delta_rms_bcp = (cand_bcpH - y_base_final.unsqueeze(2)).pow(2).mean(dim=-1).sqrt()
        gain_bk, labels_bk = _cluster_best_penalty_gain_and_labels(
            base_err_bc=base_err_bc,
            cand_err_bcp=cand_err_bcp,
            cluster_id_c=cid_c,
            K=int(K),
            allowed_mask_kp=allowed,
        )
        penalty_gain_bkp, penalty_delta_bkp = _cluster_penalty_gain_and_delta(
            base_err_bc=base_err_bc,
            cand_err_bcp=cand_err_bcp,
            delta_rms_bcp=delta_rms_bcp,
            cluster_id_c=cid_c,
            K=int(K),
            allowed_mask_kp=allowed,
        )
        current_bk = (mask_bkp * probs_bkp).argmax(dim=-1).to(dtype=torch.long) + 1
        if allow_skip:
            current_bk = torch.where(skip_bk > 0.5, torch.zeros_like(current_bk), current_bk)
        gain_chunks.append(gain_bk.detach().cpu())
        penalty_gain_chunks.append(penalty_gain_bkp.detach().cpu())
        penalty_delta_chunks.append(penalty_delta_bkp.detach().cpu())
        label_chunks.append(labels_bk.detach().cpu())
        current_chunks.append(current_bk.detach().cpu())
        if allow_skip:
            skip_prob_chunks.append(skip_prob_bk.detach().cpu())
    if not gain_chunks:
        raise RuntimeError("skip-zero diagnostic collected no batches.")
    result = {
        "best_penalty_gain": torch.cat(gain_chunks, dim=0),
        "penalty_gain": torch.cat(penalty_gain_chunks, dim=0),
        "penalty_delta_rms": torch.cat(penalty_delta_chunks, dim=0),
        "labels": torch.cat(label_chunks, dim=0),
        "current_pred": torch.cat(current_chunks, dim=0),
    }
    if skip_prob_chunks:
        result["skip_prob"] = torch.cat(skip_prob_chunks, dim=0)
    return result


def _skip_probability_summary(skip_prob_bk: Optional[torch.Tensor], labels_bk: torch.Tensor) -> Dict[str, object]:
    if skip_prob_bk is None:
        return {"available": False}
    skip_prob = skip_prob_bk.detach().cpu().to(dtype=torch.float32).reshape(-1)
    labels = labels_bk.detach().cpu().to(dtype=torch.long).reshape(-1)
    valid = torch.isfinite(skip_prob) & (labels >= 0)
    skip_prob = skip_prob[valid]
    labels = labels[valid]
    if int(skip_prob.numel()) <= 0:
        return {"available": False}
    oracle_skip = labels == 0
    oracle_penalty = labels > 0
    return {
        "available": True,
        "mean": float(skip_prob.mean().item()),
        "p95": float(skip_prob.quantile(0.95).item()),
        "max": float(skip_prob.max().item()),
        "gt_0_5_rate": float((skip_prob > 0.5).to(dtype=torch.float32).mean().item()),
        "oracle_skip_mean": _mean_or_none(skip_prob[oracle_skip]),
        "oracle_penalty_mean": _mean_or_none(skip_prob[oracle_penalty]),
    }


def _markdown_report(payload: Dict[str, object]) -> str:
    lines = [
        "# NEXT-11d Skip-Zero Margin Diagnostic",
        "",
        f"- config: `{payload['config_path']}`",
        f"- checkpoint: `{payload['checkpoint_path']}`",
        f"- no_test_read: `{payload['no_test_read']}`",
        f"- failure_layer: `{payload['failure_layer']}`",
        "",
        "| split | samples | oracle skip | actual skip | skip recall | skip precision | |gain|<=1e-3 | skip gain mean | penalty gain mean | skip_prob mean/p95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, summary in (payload.get("splits", {}) or {}).items():
        probs = summary.get("skip_probability", {}) or {}
        near = summary.get("near_zero_abs_gain_rates", {}) or {}
        lines.append(
            "| {split} | {samples} | {os:.4f} | {asr:.4f} | {rec:.4f} | {prec:.4f} | {nz:.4f} | {sg} | {pg} | {sp}/{sp95} |".format(
                split=split,
                samples=int(summary.get("samples", 0) or 0),
                os=float(summary.get("oracle_skip_rate", 0.0) or 0.0),
                asr=float(summary.get("actual_skip_rate", 0.0) or 0.0),
                rec=float(summary.get("skip_recall", 0.0) or 0.0),
                prec=float(summary.get("skip_precision", 0.0) or 0.0),
                nz=float(near.get("0.001", 0.0) or 0.0),
                sg="n/a" if summary.get("oracle_skip_gain_mean") is None else f"{float(summary['oracle_skip_gain_mean']):.6g}",
                pg="n/a" if summary.get("oracle_penalty_gain_mean") is None else f"{float(summary['oracle_penalty_gain_mean']):.6g}",
                sp="n/a" if not probs.get("available") else f"{float(probs.get('mean', 0.0)):.4f}",
                sp95="n/a" if not probs.get("available") else f"{float(probs.get('p95', 0.0)):.4f}",
            )
        )
    lines.extend(
        [
            "",
            "## Per-Penalty Candidate Support",
            "",
            "| split | penalty | samples | gain mean | positive | strong positive | strong negative | near zero | delta RMS mean/p95 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split, summary in (payload.get("splits", {}) or {}).items():
        per_penalty = summary.get("per_penalty_support", {}) if isinstance(summary, dict) else {}
        if not isinstance(per_penalty, dict):
            continue
        for penalty, stats in per_penalty.items():
            if not isinstance(stats, dict):
                continue
            samples = int(stats.get("samples", 0) or 0)
            lines.append(
                "| {split} | {penalty} | {samples} | {gain} | {pos} | {spos} | {sneg} | {near} | {delta}/{delta95} |".format(
                    split=split,
                    penalty=penalty,
                    samples=samples,
                    gain="n/a" if stats.get("gain_mean") is None else f"{float(stats['gain_mean']):.6g}",
                    pos="n/a" if stats.get("positive_rate") is None else f"{float(stats['positive_rate']):.4f}",
                    spos="n/a" if stats.get("strong_positive_rate") is None else f"{float(stats['strong_positive_rate']):.4f}",
                    sneg="n/a" if stats.get("strong_negative_rate") is None else f"{float(stats['strong_negative_rate']):.4f}",
                    near="n/a" if stats.get("near_zero_rate") is None else f"{float(stats['near_zero_rate']):.4f}",
                    delta="n/a" if stats.get("delta_rms_mean") is None else f"{float(stats['delta_rms_mean']):.6g}",
                    delta95="n/a" if stats.get("delta_rms_p95") is None else f"{float(stats['delta_rms_p95']):.6g}",
                )
            )
    lines.extend(
        [
            "",
            "## Action-Floor Oracle Counterfactual",
            "",
            "| split | min delta RMS | oracle skip | oracle penalty | actionable candidate | empty actionable | |gain|<=1e-3 | skip gain mean | penalty gain mean |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split, summary in (payload.get("splits", {}) or {}).items():
        floors = summary.get("action_floor_oracle", {}) if isinstance(summary, dict) else {}
        if not isinstance(floors, dict):
            continue
        for floor, stats in floors.items():
            if not isinstance(stats, dict):
                continue
            near = stats.get("near_zero_abs_gain_rates", {}) or {}
            lines.append(
                "| {split} | {floor} | {os:.4f} | {op:.4f} | {ar:.4f} | {er:.4f} | {nz:.4f} | {sg} | {pg} |".format(
                    split=split,
                    floor=floor,
                    os=float(stats.get("oracle_skip_rate", 0.0) or 0.0),
                    op=float(stats.get("oracle_penalty_rate", 0.0) or 0.0),
                    ar=float(stats.get("actionable_candidate_rate", 0.0) or 0.0),
                    er=float(stats.get("empty_actionable_rate", 0.0) or 0.0),
                    nz=float(near.get("0.001", 0.0) or 0.0),
                    sg="n/a" if stats.get("oracle_skip_gain_mean") is None else f"{float(stats['oracle_skip_gain_mean']):.6g}",
                    pg="n/a" if stats.get("oracle_penalty_gain_mean") is None else f"{float(stats['oracle_penalty_gain_mean']):.6g}",
                )
            )
    return "\n".join(lines) + "\n"


def run_diagnostic(args: argparse.Namespace) -> Dict[str, object]:
    splits = [str(v) for v in args.splits]
    _reject_test_splits(splits)
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
    out_splits: Dict[str, object] = {}
    for split in splits:
        if split not in loaders:
            raise ValueError(f"Unknown split {split!r}; available splits: {sorted(loaders.keys())}.")
        tensors = _collect_split_margin(
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
            allowed_mask_kp=allowed_mask_kp,
            max_batches=int(args.max_batches),
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
        summary = _skip_margin_summary(
            best_penalty_gain_bk=tensors["best_penalty_gain"],
            labels_bk=tensors["labels"],
            current_pred_bk=tensors["current_pred"],
            near_zero_thresholds=args.near_zero_threshold,
        )
        summary["skip_probability"] = _skip_probability_summary(
            tensors.get("skip_prob"),
            tensors["labels"],
        )
        summary["per_penalty_support"] = _per_penalty_support_summary(
            penalty_gain_bkp=tensors["penalty_gain"],
            penalty_delta_rms_bkp=tensors["penalty_delta_rms"],
            penalty_names=penalty_names,
            strong_threshold=float(args.strong_gain_threshold),
            near_zero_threshold=float(args.per_penalty_near_zero_threshold),
        )
        action_floor: Dict[str, object] = {}
        for floor in args.candidate_action_floor or []:
            floor_gain, floor_labels, has_actionable = _best_penalty_gain_and_labels_from_penalty_tensors(
                penalty_gain_bkp=tensors["penalty_gain"],
                penalty_delta_rms_bkp=tensors["penalty_delta_rms"],
                min_delta_rms=float(floor),
            )
            floor_summary = _skip_margin_summary(
                best_penalty_gain_bk=floor_gain,
                labels_bk=floor_labels,
                current_pred_bk=tensors["current_pred"],
                near_zero_thresholds=args.near_zero_threshold,
            )
            action_rate = has_actionable.detach().cpu().to(dtype=torch.float32)
            floor_summary["actionable_candidate_rate"] = float(action_rate.mean().item())
            floor_summary["empty_actionable_rate"] = float((~has_actionable).detach().cpu().to(dtype=torch.float32).mean().item())
            action_floor[f"{float(floor):g}"] = floor_summary
        if action_floor:
            summary["action_floor_oracle"] = action_floor
        out_splits[str(split)] = summary
    train = out_splits.get("train_fit", {})
    holdout = out_splits.get("train_holdout", {})
    val = out_splits.get("val", {})
    if (
        isinstance(train, dict)
        and isinstance(holdout, dict)
        and float(train.get("actual_skip_rate", 0.0) or 0.0) < 0.01
        and float(holdout.get("actual_skip_rate", 0.0) or 0.0) < 0.01
    ):
        failure_layer = "skip/no-op behavior"
    elif isinstance(val, dict) and float(val.get("actual_skip_rate", 0.0) or 0.0) < 0.01:
        failure_layer = "train-val utility shift"
    else:
        failure_layer = "routing target mismatch"
    payload: Dict[str, object] = {
        "config_path": str(args.config),
        "checkpoint_path": str(args.checkpoint),
        "out_dir": str(args.out_dir),
        "no_test_read": True,
        "splits": out_splits,
        "penalty_names": penalty_names,
        "allowed_mask": prior_summary.get("allowed_mask"),
        "gate_feature_mode": gate_feature_mode,
        "failure_layer": failure_layer,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "skip_zero_margin_diagnostic.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "skip_zero_margin_diagnostic.md").write_text(_markdown_report(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="NEXT-11d skip/no-op zero-rate margin diagnostic.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--splits", nargs="+", default=["train_fit", "train_holdout", "val"])
    parser.add_argument("--near-zero-threshold", nargs="+", type=float, default=[1.0e-4, 5.0e-4, 1.0e-3, 5.0e-3])
    parser.add_argument("--strong-gain-threshold", type=float, default=1.0e-3)
    parser.add_argument("--per-penalty-near-zero-threshold", type=float, default=1.0e-3)
    parser.add_argument("--candidate-action-floor", nargs="*", type=float, default=[])
    args = parser.parse_args()
    payload = run_diagnostic(args)
    for split, summary in payload["splits"].items():
        print(
            "{split}: oracle_skip={oracle:.4f} actual_skip={actual:.4f} near_1e-3={near:.4f}".format(
                split=split,
                oracle=float(summary.get("oracle_skip_rate", 0.0) or 0.0),
                actual=float(summary.get("actual_skip_rate", 0.0) or 0.0),
                near=float((summary.get("near_zero_abs_gain_rates", {}) or {}).get("0.001", 0.0) or 0.0),
            )
        )


if __name__ == "__main__":
    main()
