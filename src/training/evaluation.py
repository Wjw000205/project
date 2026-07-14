"""Evaluation loops, calendar correction, and routing diagnostics."""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..models.dynamic_lambda import ClusterwiseDynamicLambda
from ..models.learnable_anchor import ClusterwiseLearnableOutputAnchor
from ..models.moe_gate import ClusterwiseMoEGate, scatter_mean_bc_to_bk, scatter_mean_bcf_to_bkf
from ..models.penalties import normalize_penalties
from ..models.residual_moe import ClusterwisePredResidualMoE
from ..utils.cluster_memory import scatter_mean_bcl_to_bkl
from ..utils.metrics import accumulate_channel_errors, mse_mae_from_sums
from .anchors import (
    apply_history_anchor_adapter,
    apply_moe_output_anchor_experts,
    apply_train_stat_anchor_expert,
    apply_train_stat_input_centering,
    build_moe_output_anchor_fixed_expert_delta,
)
from .core import (
    _apply_mae_objective_weight,
    _apply_skip_to_penalty_loss,
    _build_gate_routing_features,
    _cluster_route_oracle_labels_and_gain_from_candidates,
    _compute_lambda_bkp,
    _gate_feature_names_for_mode,
    _mae_objective_bc_from_abs,
    _mae_objective_weight_is_nonzero,
    _normalize_gate_feature_mode,
    _router_penalty_context_from_history,
    _select_rank_mask,
    extract_gate_features,
)
from .selectors import (
    _candidate_selector_features,
    _cluster_utility_threshold_stats,
    _named_forecast_attribute_error,
    _pred_residual_candidates_on_eval_path,
)


def _fixed_expert_candidate_base(
    y_base_bch: torch.Tensor,
    pred_residual: Optional[ClusterwisePredResidualMoE],
    fixed_expert_delta_bch: Optional[torch.Tensor],
) -> torch.Tensor:
    """Return the exact base seen by PKR adapters when a fixed expert is active."""
    if (
        pred_residual is None
        or fixed_expert_delta_bch is None
        or not bool(getattr(pred_residual, "periodic_anchor_expert_enable", False))
    ):
        return y_base_bch
    return y_base_bch + float(
        getattr(pred_residual, "periodic_anchor_expert_scale", 1.0)
    ) * fixed_expert_delta_bch.to(device=y_base_bch.device, dtype=y_base_bch.dtype)


_DIRECT_ATTRIBUTE_PENALTY_NAMES = {
    "level",
    "delta",
    "d2_match",
    "diff_amp",
    "amp",
    "amp_under",
}


def _pred_residual_expert_parameter_groups(
    pred_residual: ClusterwisePredResidualMoE,
) -> Tuple[List[List[nn.Parameter]], List[List[nn.Parameter]]]:
    """Return independent adapter-body and auxiliary parameter groups by penalty.

    The body group is the correction-producing MLP.  The auxiliary group contains
    alpha/intervention parameters, which are expected to be unused by the current
    fixed-alpha, intervention-free adapter-training contract.
    """
    body_groups: List[List[nn.Parameter]] = [[] for _ in range(pred_residual.P)]
    auxiliary_groups: List[List[nn.Parameter]] = [[] for _ in range(pred_residual.P)]
    for k in range(pred_residual.param_K):
        for p in range(pred_residual.P):
            idx = pred_residual._idx(k, p)
            body_groups[p].extend(
                [
                    pred_residual.W1[idx],
                    pred_residual.b1[idx],
                    pred_residual.W2[idx],
                    pred_residual.b2[idx],
                ]
            )
            auxiliary_groups[p].extend(
                [
                    pred_residual.log_alpha[idx],
                    pred_residual.W_gate[idx],
                    pred_residual.b_gate[idx],
                ]
            )
    if pred_residual.channel_expert_enable:
        for c in range(pred_residual.C_channel):
            if not bool(pred_residual.channel_expert_mask_c[c].item()):
                continue
            for p in range(pred_residual.P):
                idx = pred_residual._ch_idx(c, p)
                body_groups[p].extend(
                    [
                        pred_residual.channel_W1[idx],
                        pred_residual.channel_b1[idx],
                        pred_residual.channel_W2[idx],
                        pred_residual.channel_b2[idx],
                    ]
                )
                auxiliary_groups[p].extend(
                    [
                        pred_residual.channel_log_alpha[idx],
                        pred_residual.channel_W_gate[idx],
                        pred_residual.channel_b_gate[idx],
                    ]
                )
    return body_groups, auxiliary_groups


def _append_unique_parameters(
    destination: List[nn.Parameter],
    parameters: List[nn.Parameter],
) -> None:
    seen = {id(param) for param in destination}
    for param in parameters:
        if id(param) not in seen:
            destination.append(param)
            seen.add(id(param))


def evaluate_adapter_gradient_isolation(
    model: nn.Module,
    pred_residual: Optional[ClusterwisePredResidualMoE],
    loader: DataLoader,
    cluster_id_c: torch.Tensor,
    K: int,
    moe_cfg: dict,
    device: torch.device,
    penalty_names: List[str],
    split_name: str,
    max_batches: int = 4,
    off_diagonal_tolerance: float = 1.0e-12,
    min_diagonal_norm: float = 1.0e-12,
    history_anchor_cfg: Optional[dict] = None,
    observed_history_tc: Optional[torch.Tensor] = None,
    input_len: int = 0,
    eval_start: int = 0,
    model_train_stat_adapter_pc: Optional[torch.Tensor] = None,
    model_train_stat_adapter_cfg: Optional[dict] = None,
    train_stat_anchor_pc: Optional[torch.Tensor] = None,
    train_residual_anchor_phc: Optional[torch.Tensor] = None,
    learnable_output_anchor: Optional[ClusterwiseLearnableOutputAnchor] = None,
) -> Optional[Dict[str, object]]:
    """Measure ``named loss -> adapter`` gradient provenance on the real eval path.

    Backbone and periodic-source predictions are deliberately detached, matching
    the frozen-bank training contract.  Each row differentiates the named direct
    attribute objective of candidate ``p`` against every physical adapter body.
    """
    if pred_residual is None or len(loader) == 0 or len(penalty_names) == 0:
        return None
    if int(pred_residual.P) != len(penalty_names):
        raise ValueError("gradient-isolation penalty names must match the adapter bank")
    supported_indices = [
        p for p, name in enumerate(penalty_names)
        if str(name) in _DIRECT_ATTRIBUTE_PENALTY_NAMES
    ]
    if not supported_indices:
        return None

    body_groups, auxiliary_groups = _pred_residual_expert_parameter_groups(pred_residual)
    audit_params: List[nn.Parameter] = []
    for group in body_groups + auxiliary_groups:
        _append_unique_parameters(audit_params, group)
    original_requires_grad = {id(param): bool(param.requires_grad) for param in audit_params}
    original_model_training = bool(model.training)
    original_pred_training = bool(pred_residual.training)
    state_before = {id(param): param.detach().clone() for param in audit_params}

    row_count = len(supported_indices)
    body_grad_sq = torch.zeros(row_count, len(penalty_names), dtype=torch.float64)
    auxiliary_grad_sq = torch.zeros(row_count, len(penalty_names), dtype=torch.float64)
    loss_sum = torch.zeros(row_count, dtype=torch.float64)
    batches = 0
    sample_channels = 0
    moe_enable = bool(moe_cfg.get("enable", True))
    cid_c = cluster_id_c.detach().to(device=device, dtype=torch.long)
    flat_params: List[nn.Parameter] = []
    for group in body_groups + auxiliary_groups:
        _append_unique_parameters(flat_params, group)

    try:
        for param in audit_params:
            param.requires_grad_(True)
        model.eval()
        pred_residual.eval()
        for x, y, idx in loader:
            if max_batches > 0 and batches >= int(max_batches):
                break
            batches += 1
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            idx = idx.to(device=device, dtype=torch.long)
            query_start_abs_b = int(eval_start) + idx
            with torch.no_grad():
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
                fixed_expert_delta_bch = build_moe_output_anchor_fixed_expert_delta(
                    y_base,
                    x_bcl=x,
                    query_start_abs_b=query_start_abs_b,
                    input_len=int(input_len),
                    moe_cfg=moe_cfg,
                    moe_enable=moe_enable,
                    observed_history_tc=observed_history_tc,
                    train_stat_anchor_pc=train_stat_anchor_pc,
                    train_residual_anchor_phc=train_residual_anchor_phc,
                    learnable_output_anchor=learnable_output_anchor,
                    cluster_id_c=cluster_id_c,
                )
            route_bkp = torch.ones(
                int(x.shape[0]),
                int(K),
                len(penalty_names),
                device=device,
                dtype=x.dtype,
            )
            pred_out = pred_residual(
                x.detach(),
                y_base.detach(),
                cluster_id_c,
                route_bkp,
                skip_bk=None,
                query_start_abs_b=query_start_abs_b,
                fixed_expert_delta_bch=(
                    None
                    if fixed_expert_delta_bch is None
                    else fixed_expert_delta_bch.detach()
                ),
            )
            _, cand_bcpH = _pred_residual_candidates_on_eval_path(
                y_base.detach(),
                pred_out,
                apply_output_anchors=True,
                x_bcl=x.detach(),
                query_start_abs_b=query_start_abs_b,
                input_len=int(input_len),
                moe_cfg=moe_cfg,
                moe_enable=moe_enable,
                observed_history_tc=observed_history_tc,
                train_stat_anchor_pc=train_stat_anchor_pc,
                train_residual_anchor_phc=train_residual_anchor_phc,
                learnable_output_anchor=learnable_output_anchor,
                cluster_id_c=cluster_id_c,
                include_intervention=False,
                include_selector=False,
                include_patch_route=False,
            )
            if cand_bcpH is None:
                continue
            sample_channels += int(y.shape[0] * y.shape[1])
            for row, penalty_idx in enumerate(supported_indices):
                loss = _named_forecast_attribute_error(
                    cand_bcpH[:, :, penalty_idx, :],
                    y,
                    str(penalty_names[penalty_idx]),
                ).mean()
                loss_sum[row] += float(loss.detach().item())
                grads = torch.autograd.grad(
                    loss,
                    flat_params,
                    retain_graph=row + 1 < row_count,
                    allow_unused=True,
                )
                grad_by_id = {id(param): grad for param, grad in zip(flat_params, grads)}
                for expert_idx in range(len(penalty_names)):
                    for param in body_groups[expert_idx]:
                        grad = grad_by_id.get(id(param))
                        if grad is not None:
                            body_grad_sq[row, expert_idx] += float(
                                grad.detach().double().square().sum().item()
                            )
                    for param in auxiliary_groups[expert_idx]:
                        grad = grad_by_id.get(id(param))
                        if grad is not None:
                            auxiliary_grad_sq[row, expert_idx] += float(
                                grad.detach().double().square().sum().item()
                            )
    finally:
        pred_residual.zero_grad(set_to_none=True)
        for param in audit_params:
            param.requires_grad_(original_requires_grad[id(param)])
        model.train(original_model_training)
        pred_residual.train(original_pred_training)

    body_grad_norm = body_grad_sq.sqrt()
    auxiliary_grad_norm = auxiliary_grad_sq.sqrt()
    off_diagonal_values: List[float] = []
    auxiliary_off_diagonal_values: List[float] = []
    diagonal_values: List[float] = []
    for row, penalty_idx in enumerate(supported_indices):
        diagonal_values.append(float(body_grad_norm[row, penalty_idx].item()))
        off_diagonal_values.extend(
            float(body_grad_norm[row, expert_idx].item())
            for expert_idx in range(len(penalty_names))
            if expert_idx != penalty_idx
        )
        auxiliary_off_diagonal_values.extend(
            float(auxiliary_grad_norm[row, expert_idx].item())
            for expert_idx in range(len(penalty_names))
            if expert_idx != penalty_idx
        )
    off_diagonal_max = max(off_diagonal_values, default=0.0)
    auxiliary_off_diagonal_max = max(
        auxiliary_off_diagonal_values,
        default=0.0,
    )
    diagonal_min = min(diagonal_values, default=0.0)
    auxiliary_max = float(auxiliary_grad_norm.max().item()) if auxiliary_grad_norm.numel() else 0.0
    inactive_auxiliary_expected = bool(
        pred_residual.named_output_projection_enable
        and pred_residual.named_output_projection_fixed_alpha
    )
    max_parameter_change = max(
        (
            float((param.detach() - state_before[id(param)]).abs().max().item())
            for param in audit_params
            if param.numel() > 0
        ),
        default=0.0,
    )
    passed = bool(
        batches > 0
        and diagonal_min > float(min_diagonal_norm)
        and off_diagonal_max <= float(off_diagonal_tolerance)
        and auxiliary_off_diagonal_max <= float(off_diagonal_tolerance)
        and (
            (not inactive_auxiliary_expected)
            or auxiliary_max <= float(off_diagonal_tolerance)
        )
        and max_parameter_change == 0.0
    )
    return {
        "split": str(split_name),
        "test_split_used": str(split_name).lower() == "test",
        "batches": int(batches),
        "sample_channels": int(sample_channels),
        "loss_names": [str(penalty_names[p]) for p in supported_indices],
        "expert_names": [str(name) for name in penalty_names],
        "objective": "direct future-attribute error of independently reachable candidate_e",
        "base_contract": "frozen backbone + fixed periodic expert; both detached",
        "adapter_contract": "intervention, selector, patch route disabled; output projection and eval anchors retained",
        "body_parameter_names": ["W1", "b1", "W2", "b2"],
        "body_gradient_l2": body_grad_norm.tolist(),
        "auxiliary_parameter_names": ["log_alpha", "W_gate", "b_gate"],
        "auxiliary_gradient_l2": auxiliary_grad_norm.tolist(),
        "mean_named_loss": (loss_sum / max(batches, 1)).tolist(),
        "diagonal_min": float(diagonal_min),
        "off_diagonal_max": float(off_diagonal_max),
        "auxiliary_max": float(auxiliary_max),
        "auxiliary_off_diagonal_max": float(auxiliary_off_diagonal_max),
        "inactive_auxiliary_expected": bool(inactive_auxiliary_expected),
        "off_diagonal_tolerance": float(off_diagonal_tolerance),
        "min_diagonal_norm": float(min_diagonal_norm),
        "max_parameter_change": float(max_parameter_change),
        "passed": passed,
    }


@torch.no_grad()
def eval_loop(
    model: nn.Module,
    gate: ClusterwiseMoEGate,
    lambda_kp: torch.Tensor,
    penalty_names: List[str],
    penalty_fns: Dict[str, callable],
    loader: DataLoader,
    cluster_id_c: torch.Tensor,
    K: int,
    moe_cfg: dict,
    device: torch.device,
    select_ranks: List[int] = None,
    collect_plot: bool = False,
    plot_idx: torch.Tensor = None,
    channel_count: int = None,
    mse_weight: float = 1.0,
    gate_entropy_weight: float = 0.0,
    gate_balance_weight: float = 0.0,
    gate_soft_weight: float = 0.0,
    gate_entropy_target_frac: float = 0.0,
    gate_feature_mode: str = "history",
    penalty_scale: torch.Tensor = None,
    dynamic_lambda: ClusterwiseDynamicLambda = None,
    lambda_min_kp: torch.Tensor = None,
    mae_objective_weight=0.0,
    mae_objective_kind: str = "l1",
    mae_objective_beta: float = 1.0,
    pred_residual: Optional[ClusterwisePredResidualMoE] = None,
    pred_residual_selector: Optional[nn.Module] = None,
    pred_residual_scale_c: Optional[torch.Tensor] = None,
    eval_start: int = 0,
    history_anchor_cfg: Optional[dict] = None,
    observed_history_tc: Optional[torch.Tensor] = None,
    input_len: int = 0,
    model_train_stat_adapter_pc: Optional[torch.Tensor] = None,
    model_train_stat_adapter_cfg: Optional[dict] = None,
    train_stat_anchor_pc: Optional[torch.Tensor] = None,
    train_residual_anchor_phc: Optional[torch.Tensor] = None,
    learnable_output_anchor: Optional[ClusterwiseLearnableOutputAnchor] = None,
    calendar_feature_tf: Optional[torch.Tensor] = None,
    calendar_residual_coef_cf: Optional[torch.Tensor] = None,
    position_daily_residual_coef_cfh: Optional[torch.Tensor] = None,
    position_daily_residual_cfg: Optional[dict] = None,
    diagnostic_collector: Optional[Dict[str, object]] = None,
    channel_horizon_metric_collector: Optional[Dict[str, object]] = None,
    collect_samples: bool = True,
    base_metric_collector: Optional[Dict[str, object]] = None,
):
    model.eval()
    gate.eval()
    if dynamic_lambda is not None:
        dynamic_lambda.eval()
    if pred_residual is not None:
        pred_residual.eval()
    if learnable_output_anchor is not None:
        learnable_output_anchor.eval()

    moe_enable = bool(moe_cfg.get("enable", True))
    allow_skip = bool(moe_cfg.get("allow_skip", False)) and moe_enable
    router_mode = str(moe_cfg.get("router_mode", "learned")).lower()
    router_penalty_context_weight = float(moe_cfg.get("router_penalty_context_weight", 0.0))
    router_detach_penalty_context = bool(moe_cfg.get("router_detach_penalty_context", True))
    router_penalty_context_score = str(moe_cfg.get("router_penalty_context_score", "high_violation")).lower()
    gate_feature_mode = _normalize_gate_feature_mode(gate_feature_mode)
    moe_history_anchor_expert_cfg = moe_cfg.get("history_anchor_expert", {}) or {}
    train_stat_anchor_expert_cfg = moe_cfg.get("train_stat_anchor_expert", {}) or {}
    train_residual_anchor_expert_cfg = moe_cfg.get("train_residual_anchor_expert", {}) or {}

    P = len(penalty_names)
    total_loss_sum = torch.zeros(K, device=device)
    total_cnt = torch.zeros(K, device=device)
    mse_loss_sum = torch.zeros(K, device=device)
    mae_loss_sum = torch.zeros(K, device=device)

    # per-channel metrics
    se_c = torch.zeros(channel_count, device=device)
    ae_c = torch.zeros(channel_count, device=device)
    denom = 0
    if base_metric_collector is not None:
        base_mse_loss_sum = torch.zeros(K, device=device)
        base_mae_loss_sum = torch.zeros(K, device=device)
        base_total_cnt = torch.zeros(K, device=device)
        base_se_c = torch.zeros(channel_count, device=device)
        base_ae_c = torch.zeros(channel_count, device=device)
        base_denom = 0

    plot_cache = {}  # idx -> (x[C,L], y[C,H], yhat[C,H])
    best_sample = {}   # c -> (x[L], y[H], yhat[H], mse)
    worst_sample = {}  # c -> (x[L], y[H], yhat[H], mse)
    best_mse = torch.full((channel_count,), float("inf"), device=device) if collect_samples else None
    worst_mse = torch.full((channel_count,), -float("inf"), device=device) if collect_samples else None

    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        idx = idx.to(device=device, dtype=torch.long, non_blocking=True)

        query_start_abs_b = eval_start + idx
        x_model = apply_train_stat_input_centering(
            x,
            query_start_abs_b=query_start_abs_b,
            stat_anchor_pc=model_train_stat_adapter_pc,
            cfg=model_train_stat_adapter_cfg,
        )
        yhat_base_raw = model(x_model, cluster_id_c)
        yhat_base = apply_history_anchor_adapter(
            yhat_base_raw,
            base_pred_bch=yhat_base_raw,
            observed_history_tc=observed_history_tc,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len or x.shape[-1]),
            cfg=history_anchor_cfg,
        )
        yhat_base = apply_train_stat_anchor_expert(
            yhat_base,
            base_pred_bch=yhat_base,
            x_bcl=x,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len or x.shape[-1]),
            stat_anchor_pc=model_train_stat_adapter_pc,
            cfg=model_train_stat_adapter_cfg,
        )

        if base_metric_collector is not None:
            base_err_bch = yhat_base - y
            base_mse_bc = base_err_bch.pow(2).mean(dim=-1)
            base_mae_bc = base_err_bch.abs().mean(dim=-1)
            base_mse_loss_sum += scatter_mean_bc_to_bk(
                base_mse_bc, cluster_id_c, K
            ).sum(dim=0)
            base_mae_loss_sum += scatter_mean_bc_to_bk(
                base_mae_bc, cluster_id_c, K
            ).sum(dim=0)
            base_total_cnt += torch.tensor(
                [x.shape[0]], device=device
            ).expand_as(base_total_cnt)
            accumulate_channel_errors(base_se_c, base_ae_c, yhat_base, y)
            base_denom += int(x.shape[0] * y.shape[2])

        fixed_expert_delta_bch = None
        candidate_base_bch = yhat_base
        if pred_residual is not None and moe_enable and P > 0:
            fixed_expert_delta_bch = build_moe_output_anchor_fixed_expert_delta(
                yhat_base,
                x_bcl=x,
                query_start_abs_b=query_start_abs_b,
                input_len=int(input_len or x.shape[-1]),
                moe_cfg=moe_cfg,
                moe_enable=moe_enable,
                observed_history_tc=observed_history_tc,
                train_stat_anchor_pc=train_stat_anchor_pc,
                train_residual_anchor_phc=train_residual_anchor_phc,
                learnable_output_anchor=learnable_output_anchor,
                cluster_id_c=cluster_id_c,
            )
            candidate_base_bch = _fixed_expert_candidate_base(
                yhat_base,
                pred_residual,
                fixed_expert_delta_bch,
            )

        # Route on the fixed-expert base. This keeps the router independent
        # from the PKR adapter it is selecting while matching deployment.
        gate_feat_bkf = _build_gate_routing_features(
            x,
            candidate_base_bch,
            cluster_id_c,
            K,
            mode=gate_feature_mode,
        )
        if dynamic_lambda is None:
            feat_bkf = gate_feat_bkf
            series_bkl = None
        else:
            feat_bkf = gate_feat_bkf
            if gate_feature_mode != "history":
                feat_bcf = extract_gate_features(x)
                feat_bkf = scatter_mean_bcf_to_bkf(feat_bcf, cluster_id_c, K)
            series_bkl = scatter_mean_bcl_to_bkl(x, cluster_id_c, K)  # [B,K,L]
        route_pen_bkp = _router_penalty_context_from_history(
            x_bcl=x,
            yhat_base_bch=candidate_base_bch,
            penalty_names=penalty_names,
            penalty_fns=penalty_fns,
            penalty_scale=penalty_scale,
            cluster_id_c=cluster_id_c,
            K=K,
            router_mode=router_mode,
        )

        if moe_enable and P > 0:
            straight_through = (not moe_cfg["detach_penalty_grad"])
            mask_bkp, probs_bkp, skip_bk, _ = gate(
                gate_feat_bkf,
                straight_through=straight_through,
                penalty_context_bkp=route_pen_bkp,
                penalty_context_mode=router_mode,
                penalty_context_weight=router_penalty_context_weight,
                penalty_context_detach=router_detach_penalty_context,
                penalty_context_score=router_penalty_context_score,
            )
            rank_mask = None
            if select_ranks is not None:
                mask_bkp = _select_rank_mask(probs_bkp, select_ranks, straight_through=straight_through)
                rank_mask = _select_rank_mask(probs_bkp, select_ranks, straight_through=False)
            if gate_soft_weight > 0.0:
                probs_sel = probs_bkp
                if rank_mask is not None:
                    probs_sel = probs_sel * rank_mask
                    probs_sel = probs_sel / probs_sel.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                target_mass = mask_bkp.detach().sum(dim=-1, keepdim=True).clamp_min(1.0)
                probs_sel = probs_sel * target_mass
                mask_bkp = (1.0 - gate_soft_weight) * mask_bkp + gate_soft_weight * probs_sel
        else:
            mask_bkp = torch.zeros_like(route_pen_bkp)
            probs_bkp = None
            skip_bk = None

        yhat_residual_raw = yhat_base
        residual_gate_scale = None
        output_anchors_applied = False
        pred_out = None
        if pred_residual is not None and moe_enable and P > 0:
            pred_out = pred_residual(
                x,
                yhat_base,
                cluster_id_c,
                mask_bkp,
                skip_bk=skip_bk if allow_skip else None,
                query_start_abs_b=query_start_abs_b,
                fixed_expert_delta_bch=fixed_expert_delta_bch,
            )
            yhat_residual_raw = pred_out["y_final"]
            yhat = yhat_residual_raw
            if pred_residual_selector is not None:
                pred_residual_selector.eval()
                selector_base_bch, cand_bcpH = _pred_residual_candidates_on_eval_path(
                    yhat_base,
                    pred_out,
                    pred_residual_scale_c=pred_residual_scale_c,
                    apply_output_anchors=True,
                    x_bcl=x,
                    query_start_abs_b=query_start_abs_b,
                    input_len=int(input_len or x.shape[-1]),
                    moe_cfg=moe_cfg,
                    moe_enable=moe_enable,
                    observed_history_tc=observed_history_tc,
                    train_stat_anchor_pc=train_stat_anchor_pc,
                    train_residual_anchor_phc=train_residual_anchor_phc,
                    learnable_output_anchor=learnable_output_anchor,
                    cluster_id_c=cluster_id_c,
                    include_patch_route=False,
                )
                if cand_bcpH is not None:
                    selector_kwargs = {}
                    if bool(getattr(pred_residual_selector, "use_time_features", False)):
                        selector_kwargs["query_start_abs_b"] = query_start_abs_b
                    yhat, _ = pred_residual_selector.select_prediction(
                        x,
                        selector_base_bch,
                        cand_bcpH,
                        **selector_kwargs,
                    )
                    yhat_residual_raw = yhat
                    output_anchors_applied = True
            elif pred_residual_scale_c is not None:
                scale = pred_residual_scale_c.to(device=yhat.device, dtype=yhat.dtype).view(1, -1, 1)
                residual_gate_scale = scale.expand(yhat.shape[0], -1, -1)
                residual_base_bch = pred_out.get("candidate_base_bch", candidate_base_bch)
                yhat = residual_base_bch + scale * (yhat - residual_base_bch)
        else:
            yhat = yhat_base

        if not output_anchors_applied:
            yhat = apply_moe_output_anchor_experts(
                yhat,
                base_pred_bch=yhat_base,
                x_bcl=x,
                query_start_abs_b=query_start_abs_b,
                input_len=int(input_len or x.shape[-1]),
                moe_cfg=moe_cfg,
                moe_enable=moe_enable,
                observed_history_tc=observed_history_tc,
                train_stat_anchor_pc=train_stat_anchor_pc,
                train_residual_anchor_phc=train_residual_anchor_phc,
                learnable_output_anchor=learnable_output_anchor,
                cluster_id_c=cluster_id_c,
            )

        yhat = apply_calendar_residual_correction(
            yhat,
            calendar_feature_tf=calendar_feature_tf,
            calendar_residual_coef_cf=calendar_residual_coef_cf,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len or x.shape[-1]),
        )
        yhat = apply_position_daily_residual_ridge(
            yhat,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len or x.shape[-1]),
            coef_cfh=position_daily_residual_coef_cfh,
            cfg=position_daily_residual_cfg,
        )

        if diagnostic_collector is not None:
            limit = int(diagnostic_collector.get("limit", 0))
            count = int(diagnostic_collector.get("count", 0))
            selected_idx = diagnostic_collector.get("indices")
            if selected_idx is not None:
                selected_idx = selected_idx.to(device=idx.device, dtype=torch.long)
                take_pos = torch.isin(idx, selected_idx).nonzero(as_tuple=False).view(-1)
                remaining = max(0, limit - count)
                if int(take_pos.numel()) > remaining:
                    take_pos = take_pos[:remaining]
            else:
                take = max(0, min(int(x.shape[0]), limit - count))
                take_pos = torch.arange(take, device=x.device, dtype=torch.long)
            if int(take_pos.numel()) > 0:
                parts = diagnostic_collector.setdefault("parts", {})
                parts.setdefault("idx", []).append((eval_start + idx.index_select(0, take_pos)).detach().cpu())
                parts.setdefault("x", []).append(x.index_select(0, take_pos).detach().cpu())
                parts.setdefault("y_true", []).append(y.index_select(0, take_pos).detach().cpu())
                parts.setdefault("y_base", []).append(yhat_base.index_select(0, take_pos).detach().cpu())
                parts.setdefault("y_residual_raw", []).append(yhat_residual_raw.index_select(0, take_pos).detach().cpu())
                parts.setdefault("y_final", []).append(yhat.index_select(0, take_pos).detach().cpu())
                if pred_out is not None:
                    fixed_features = pred_out.get("anchor_ridge_gate_features_bcd")
                    anchor_branch = pred_out.get("periodic_expert_branch_bch")
                    pred_y_final = pred_out.get("y_final")
                    if fixed_features is not None:
                        parts.setdefault("anchor_ridge_gate_features", []).append(
                            fixed_features.index_select(0, take_pos).detach().cpu()
                        )
                    if anchor_branch is not None:
                        parts.setdefault("periodic_expert_branch", []).append(
                            anchor_branch.index_select(0, take_pos).detach().cpu()
                        )
                    if pred_y_final is not None:
                        parts.setdefault("pred_residual_y_final", []).append(
                            pred_y_final.index_select(0, take_pos).detach().cpu()
                        )
                if probs_bkp is not None:
                    parts.setdefault("gate_probs", []).append(probs_bkp.index_select(0, take_pos).detach().cpu())
                if mask_bkp is not None:
                    parts.setdefault("gate_mask", []).append(mask_bkp.index_select(0, take_pos).detach().cpu())
                if skip_bk is not None:
                    parts.setdefault("skip_prob", []).append(skip_bk.index_select(0, take_pos).detach().cpu())
                if residual_gate_scale is not None:
                    parts.setdefault("residual_gate_scale", []).append(
                        residual_gate_scale.index_select(0, take_pos).detach().cpu()
                    )
                diagnostic_collector["count"] = count + int(take_pos.numel())

        # base error metrics per channel
        err_bch = yhat - y
        abs_err_bch = err_bch.abs()
        mse_bc = err_bch.pow(2).mean(dim=-1)  # [B,C]
        mae_bc = abs_err_bch.mean(dim=-1)  # [B,C]
        mse_bk = scatter_mean_bc_to_bk(mse_bc, cluster_id_c, K)  # [B,K]
        mae_bk = scatter_mean_bc_to_bk(mae_bc, cluster_id_c, K)  # [B,K]
        mse_loss_sum += mse_bk.sum(dim=0)
        mae_loss_sum += mae_bk.sum(dim=0)

        # penalties on the final prediction.
        if P > 0:
            pen_bcp = []
            for name in penalty_names:
                pen_bc = penalty_fns[name](yhat, y)  # [B,C]
                pen_bcp.append(pen_bc)
            pen_bcp = torch.stack(pen_bcp, dim=-1)  # [B,C,P]
            pen_bcp = normalize_penalties(pen_bcp, scale=penalty_scale)
            pen_bkp = scatter_mean_bcf_to_bkf(pen_bcp, cluster_id_c, K)  # [B,K,P]
        else:
            pen_bkp = route_pen_bkp
        # loss per cluster
        lam = _compute_lambda_bkp(
            base_lambda_kp=lambda_kp,
            feat_bkf=feat_bkf,
            series_bkl=series_bkl,
            dynamic_lambda=dynamic_lambda,
            lambda_min_kp=lambda_min_kp,
        )
        # Validation/test objective excludes gate regularizers; they are training-only priors.
        penalty_loss_bk = (mask_bkp * lam * pen_bkp).sum(dim=-1)
        penalty_loss_bk = _apply_skip_to_penalty_loss(
            penalty_loss_bk,
            skip_bk=skip_bk if allow_skip else None,
            skip_cost=0.0,
        )
        if _mae_objective_weight_is_nonzero(mae_objective_weight):
            mae_objective_bc = _mae_objective_bc_from_abs(
                abs_err_bch,
                kind=mae_objective_kind,
                beta=mae_objective_beta,
            )
            mae_objective_bk = scatter_mean_bc_to_bk(mae_objective_bc, cluster_id_c, K)
        else:
            mae_objective_bk = torch.zeros_like(mse_bk)
        loss_bk = (mse_weight * mse_bk) + _apply_mae_objective_weight(mae_objective_bk, mae_objective_weight) + penalty_loss_bk  # [B,K]
        total_loss_sum += loss_bk.sum(dim=0)
        total_cnt += torch.tensor([x.shape[0]], device=device).expand_as(total_cnt)

        # per-channel metrics
        accumulate_channel_errors(se_c, ae_c, yhat, y)
        denom += int(x.shape[0] * y.shape[2])
        if channel_horizon_metric_collector is not None:
            se_ch = channel_horizon_metric_collector.get("se_ch")
            ae_ch = channel_horizon_metric_collector.get("ae_ch")
            if se_ch is None or ae_ch is None:
                se_ch = torch.zeros(channel_count, y.shape[-1], device=device, dtype=err_bch.dtype)
                ae_ch = torch.zeros(channel_count, y.shape[-1], device=device, dtype=err_bch.dtype)
                channel_horizon_metric_collector["se_ch"] = se_ch
                channel_horizon_metric_collector["ae_ch"] = ae_ch
            se_ch += err_bch.pow(2).sum(dim=0)
            ae_ch += abs_err_bch.sum(dim=0)
            channel_horizon_metric_collector["count"] = int(
                channel_horizon_metric_collector.get("count", 0)
            ) + int(x.shape[0])

        if collect_samples:
            for b in range(x.shape[0]):
                cur = mse_bc[b]  # [C]
                better = cur < best_mse
                worse = cur > worst_mse
                if better.any():
                    idxs = better.nonzero(as_tuple=False).view(-1).tolist()
                    for c in idxs:
                        best_mse[c] = cur[c]
                        best_sample[c] = (x[b, c].detach().cpu(), y[b, c].detach().cpu(), yhat[b, c].detach().cpu(), float(cur[c].item()))
                if worse.any():
                    idxs = worse.nonzero(as_tuple=False).view(-1).tolist()
                    for c in idxs:
                        worst_mse[c] = cur[c]
                        worst_sample[c] = (x[b, c].detach().cpu(), y[b, c].detach().cpu(), yhat[b, c].detach().cpu(), float(cur[c].item()))

        # Cache selected windows for plotting.
        if collect_plot and plot_idx is not None:
            # idx: [B]
            idx = idx.to(device)
            hit = torch.isin(idx, plot_idx)
            if hit.any():
                hit_pos = hit.nonzero(as_tuple=False).view(-1)
                for p in hit_pos.tolist():
                    gidx = int(idx[p].item())
                    if gidx not in plot_cache:
                        plot_cache[gidx] = (x[p].detach().cpu(), y[p].detach().cpu(), yhat[p].detach().cpu())

    avg_loss_k = total_loss_sum / total_cnt.clamp_min(1.0)
    avg_mse_k = mse_loss_sum / total_cnt.clamp_min(1.0)
    avg_mae_k = mae_loss_sum / total_cnt.clamp_min(1.0)
    mse_c, mae_c = mse_mae_from_sums(se_c, ae_c, denom)
    if base_metric_collector is not None:
        base_mse_c, base_mae_c = mse_mae_from_sums(
            base_se_c, base_ae_c, base_denom
        )
        base_metric_collector.update(
            {
                "avg_mse_k": (
                    base_mse_loss_sum / base_total_cnt.clamp_min(1.0)
                ).detach().cpu(),
                "avg_mae_k": (
                    base_mae_loss_sum / base_total_cnt.clamp_min(1.0)
                ).detach().cpu(),
                "mse_c": base_mse_c.detach().cpu(),
                "mae_c": base_mae_c.detach().cpu(),
                "num_prediction_elements_per_channel": int(base_denom),
            }
        )
    return avg_loss_k, avg_mse_k, avg_mae_k, mse_c.detach().cpu(), mae_c.detach().cpu(), plot_cache, best_sample, worst_sample


@torch.no_grad()


@torch.no_grad()


def _calendar_features_from_datetime(
    dates: pd.Series | pd.DatetimeIndex,
    cfg: dict,
) -> Tuple[torch.Tensor, List[str]]:
    dt = pd.to_datetime(dates, errors="coerce")
    if isinstance(dt, pd.Series):
        dt = pd.DatetimeIndex(dt)
    if dt.isna().any():
        raise ValueError("calendar_residual requires parseable timestamps in data.date_col.")

    features: List[np.ndarray] = []
    names: List[str] = []
    if bool(cfg.get("include_bias", True)):
        features.append(np.ones(len(dt), dtype=np.float32))
        names.append("bias")

    harmonics = max(1, int(cfg.get("tod_harmonics", cfg.get("harmonics", 2))))
    if bool(cfg.get("time_of_day", True)):
        seconds = (
            dt.hour.to_numpy(dtype=np.float32) * 3600.0
            + dt.minute.to_numpy(dtype=np.float32) * 60.0
            + dt.second.to_numpy(dtype=np.float32)
        )
        phase = seconds / float(24 * 3600)
        for h in range(1, harmonics + 1):
            angle = (2.0 * math.pi * float(h)) * phase
            features.append(np.sin(angle).astype(np.float32))
            names.append(f"tod_sin_{h}")
            features.append(np.cos(angle).astype(np.float32))
            names.append(f"tod_cos_{h}")

    if bool(cfg.get("day_of_week", True)):
        phase = dt.dayofweek.to_numpy(dtype=np.float32) / 7.0
        angle = 2.0 * math.pi * phase
        features.append(np.sin(angle).astype(np.float32))
        names.append("dow_sin")
        features.append(np.cos(angle).astype(np.float32))
        names.append("dow_cos")

    if bool(cfg.get("month_of_year", False)):
        phase = (dt.month.to_numpy(dtype=np.float32) - 1.0) / 12.0
        angle = 2.0 * math.pi * phase
        features.append(np.sin(angle).astype(np.float32))
        names.append("month_sin")
        features.append(np.cos(angle).astype(np.float32))
        names.append("month_cos")

    if len(features) == 0:
        raise ValueError("calendar_residual must enable at least one feature.")
    return torch.from_numpy(np.stack(features, axis=1).astype(np.float32)), names


def build_calendar_feature_tensor(csv_path: str, date_col: int, max_rows: int, cfg: dict) -> Tuple[torch.Tensor, List[str]]:
    df = pd.read_csv(csv_path, usecols=[int(date_col)])
    if int(max_rows or 0) > 0:
        df = df.iloc[: int(max_rows)]
    return _calendar_features_from_datetime(df.iloc[:, 0], cfg)


def calendar_feature_batch(
    calendar_feature_tf: torch.Tensor,
    query_start_abs_b: torch.Tensor,
    input_len: int,
    pred_len: int,
) -> torch.Tensor:
    device = query_start_abs_b.device
    steps_h = torch.arange(int(pred_len), device=device, dtype=torch.long)
    idx_bh = query_start_abs_b.to(device=device, dtype=torch.long).view(-1, 1) + int(input_len) + steps_h.view(1, -1)
    if int(idx_bh.min().item()) < 0 or int(idx_bh.max().item()) >= int(calendar_feature_tf.shape[0]):
        raise ValueError("calendar_residual forecast index is outside timestamp range.")
    flat = idx_bh.reshape(-1)
    feat = calendar_feature_tf.to(device=device).index_select(0, flat)
    return feat.view(idx_bh.shape[0], idx_bh.shape[1], int(calendar_feature_tf.shape[1]))


def apply_calendar_residual_correction(
    yhat_bch: torch.Tensor,
    calendar_feature_tf: Optional[torch.Tensor],
    calendar_residual_coef_cf: Optional[torch.Tensor],
    query_start_abs_b: torch.Tensor,
    input_len: int,
) -> torch.Tensor:
    if calendar_feature_tf is None or calendar_residual_coef_cf is None:
        return yhat_bch
    feat_bhf = calendar_feature_batch(
        calendar_feature_tf=calendar_feature_tf,
        query_start_abs_b=query_start_abs_b,
        input_len=int(input_len),
        pred_len=int(yhat_bch.shape[-1]),
    ).to(device=yhat_bch.device, dtype=yhat_bch.dtype)
    coef_cf = calendar_residual_coef_cf.to(device=yhat_bch.device, dtype=yhat_bch.dtype)
    return yhat_bch + torch.einsum("bhf,cf->bch", feat_bhf, coef_cf)


def position_daily_feature_batch(
    query_start_abs_b: torch.Tensor,
    input_len: int,
    harmonics: int = 4,
    period: int = 96,
) -> torch.Tensor:
    origin = query_start_abs_b.to(dtype=torch.float64) + float(input_len)
    angle = 2.0 * math.pi * origin / float(max(1, int(period)))
    features = [torch.ones_like(angle)]
    for harmonic in range(1, max(1, int(harmonics)) + 1):
        features.extend(
            [torch.sin(float(harmonic) * angle), torch.cos(float(harmonic) * angle)]
        )
    return torch.stack(features, dim=-1)


def apply_position_daily_residual_ridge(
    yhat_bch: torch.Tensor,
    query_start_abs_b: torch.Tensor,
    input_len: int,
    coef_cfh: Optional[torch.Tensor],
    cfg: Optional[dict] = None,
) -> torch.Tensor:
    if coef_cfh is None:
        return yhat_bch
    cfg = cfg or {}
    features_bf = position_daily_feature_batch(
        query_start_abs_b,
        input_len=int(input_len),
        harmonics=int(cfg.get("daily_harmonics", 4)),
        period=int(cfg.get("daily_period", 96)),
    ).to(device=yhat_bch.device, dtype=yhat_bch.dtype)
    coef = coef_cfh.to(device=yhat_bch.device, dtype=yhat_bch.dtype)
    return yhat_bch + torch.einsum("bf,cfh->bch", features_bf, coef)


def fit_position_daily_residual_ridge_from_prediction_parts(
    idx_parts: List[torch.Tensor],
    y_true_parts: List[torch.Tensor],
    y_pred_parts: List[torch.Tensor],
    input_len: int,
    cfg: dict,
) -> Tuple[Optional[torch.Tensor], Dict[str, object]]:
    if not idx_parts:
        return None, {"enable": False, "reason": "empty_prediction_parts"}
    idx = torch.cat(idx_parts, dim=0).double()
    y_true = torch.cat(y_true_parts, dim=0).double()
    y_pred = torch.cat(y_pred_parts, dim=0).double()
    features = position_daily_feature_batch(
        idx,
        input_len=int(input_len),
        harmonics=int(cfg.get("daily_harmonics", 4)),
        period=int(cfg.get("daily_period", 96)),
    ).double()
    residual = y_true - y_pred
    sample_count = int(features.shape[0])
    xtx = features.transpose(0, 1).matmul(features)
    xtr_fch = torch.einsum("bf,bch->fch", features, residual)
    feature_scale = torch.sqrt(torch.diag(xtx) / float(sample_count)).clamp_min(1.0e-6)
    xtx_scaled = xtx / feature_scale[:, None] / feature_scale[None, :]
    xtr_scaled = xtr_fch / feature_scale[:, None, None]
    ridge = max(0.0, float(cfg.get("ridge", 0.3)))
    regularizer = ridge * float(sample_count) * torch.eye(
        int(features.shape[1]), dtype=torch.float64
    )
    if not bool(cfg.get("regularize_bias", False)):
        regularizer[0, 0] = 0.0
    coef_fch_scaled = torch.linalg.solve(
        xtx_scaled + regularizer,
        xtr_scaled.reshape(int(features.shape[1]), -1),
    ).reshape_as(xtr_scaled)
    coef_cfh = (
        coef_fch_scaled / feature_scale[:, None, None]
    ).permute(1, 0, 2).contiguous().float()
    return coef_cfh, {
        "enable": True,
        "source_split": str(cfg.get("source_split", "val")),
        "fit_windows": sample_count,
        "daily_period": int(cfg.get("daily_period", 96)),
        "daily_harmonics": int(cfg.get("daily_harmonics", 4)),
        "feature_dim": int(features.shape[1]),
        "ridge": ridge,
        "regularize_bias": bool(cfg.get("regularize_bias", False)),
        "coef_mean_abs": float(coef_cfh.abs().mean().item()),
        "coef_max_abs": float(coef_cfh.abs().max().item()),
    }


def fit_anchor_ridge_gate_from_prediction_parts(
    pred_residual: ClusterwisePredResidualMoE,
    parts: Dict[str, object],
    position_daily_coef_cfh: torch.Tensor,
    input_len: int,
    position_cfg: dict,
    gate_cfg: dict,
) -> Dict[str, object]:
    required = (
        "idx",
        "y_true",
        "pred_residual_y_final",
        "periodic_expert_branch",
        "anchor_ridge_gate_features",
    )
    if any(not list(parts.get(name, []) or []) for name in required):
        return {
            "enable": False,
            "adopted": False,
            "reason": "missing_prediction_parts",
        }
    if not pred_residual.anchor_ridge_gate_enable or pred_residual.anchor_ridge_gate is None:
        return {
            "enable": False,
            "adopted": False,
            "reason": "gate_not_enabled",
        }

    idx = torch.cat(list(parts["idx"]), dim=0).long()
    y_true = torch.cat(list(parts["y_true"]), dim=0).float()
    pred_y = torch.cat(list(parts["pred_residual_y_final"]), dim=0).float()
    anchor = torch.cat(list(parts["periodic_expert_branch"]), dim=0).float()
    features = torch.cat(list(parts["anchor_ridge_gate_features"]), dim=0).float()
    if not (
        int(idx.shape[0])
        == int(y_true.shape[0])
        == int(pred_y.shape[0])
        == int(anchor.shape[0])
        == int(features.shape[0])
    ):
        raise ValueError("anchor/ridge gate prediction parts have inconsistent lengths.")
    ridge = apply_position_daily_residual_ridge(
        torch.zeros_like(pred_y),
        query_start_abs_b=idx,
        input_len=int(input_len),
        coef_cfh=position_daily_coef_cfh,
        cfg=position_cfg,
    )
    # The collection pass runs before ridge coefficients are loaded and an
    # unfitted gate is an exact identity. Removing the anchor therefore leaves
    # the routed penalty path plus the frozen backbone.
    base_without_fixed = pred_y - anchor

    windows = int(features.shape[0])
    channels = int(features.shape[1])
    holdout_fraction = min(
        0.8,
        max(0.05, float(gate_cfg.get("holdout_fraction", 0.3))),
    )
    train_windows = max(1, min(windows - 1, int(round(windows * (1.0 - holdout_fraction)))))
    if windows < 2:
        return {
            "enable": True,
            "adopted": False,
            "reason": "insufficient_windows",
            "fit_windows": windows,
        }

    feature_train = features[:train_windows].reshape(-1, int(features.shape[-1]))
    feature_mean = feature_train.mean(dim=0)
    feature_std = feature_train.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    pred_residual.set_anchor_ridge_gate_normalization(
        feature_mean,
        feature_std,
        fitted=True,
    )

    device = next(pred_residual.anchor_ridge_gate.parameters()).device
    train_feature = features[:train_windows].reshape(-1, int(features.shape[-1])).to(device)
    train_base = base_without_fixed[:train_windows].reshape(-1, int(pred_y.shape[-1])).to(device)
    train_anchor = anchor[:train_windows].reshape_as(train_base).to(device)
    train_ridge = ridge[:train_windows].reshape_as(train_base).to(device)
    train_target = y_true[:train_windows].reshape_as(train_base).to(device)
    hold_feature = features[train_windows:].reshape(-1, int(features.shape[-1])).to(device)
    hold_base = base_without_fixed[train_windows:].reshape(-1, int(pred_y.shape[-1])).to(device)
    hold_anchor = anchor[train_windows:].reshape_as(hold_base).to(device)
    hold_ridge = ridge[train_windows:].reshape_as(hold_base).to(device)
    hold_target = y_true[train_windows:].reshape_as(hold_base).to(device)

    def prediction(
        feature_nd: torch.Tensor,
        base_nh: torch.Tensor,
        anchor_nh: torch.Tensor,
        ridge_nh: torch.Tensor,
        *,
        hard: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        weight_n2 = pred_residual.anchor_ridge_gate_weights_from_features(
            feature_nd.unsqueeze(1),
            hard=hard,
        ).squeeze(1)
        yhat_nh = (
            base_nh
            + weight_n2[:, :1] * anchor_nh
            + weight_n2[:, 1:] * ridge_nh
        )
        return yhat_nh, weight_n2

    with torch.no_grad():
        identity_hold_pred = hold_base + hold_anchor + hold_ridge
        identity_hold_mse = float(
            (identity_hold_pred - hold_target).square().mean().item()
        )

    gate_parameters = list(pred_residual.anchor_ridge_gate.parameters())
    original_requires_grad = [bool(parameter.requires_grad) for parameter in gate_parameters]
    for parameter in gate_parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        gate_parameters,
        lr=float(gate_cfg.get("lr", 1.0e-3)),
        weight_decay=max(0.0, float(gate_cfg.get("weight_decay", 1.0e-4))),
    )
    epochs = max(1, int(gate_cfg.get("epochs", 200)))
    batch_size = max(1, int(gate_cfg.get("batch_size", 4096)))
    identity_weight = max(0.0, float(gate_cfg.get("identity_weight", 1.0e-4)))
    patience = max(1, int(gate_cfg.get("patience", 30)))
    best_soft_hold_mse = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(gate_cfg.get("seed", 2026)))
    sample_count = int(train_feature.shape[0])
    pred_residual.anchor_ridge_gate.train()
    for epoch in range(1, epochs + 1):
        order = torch.randperm(sample_count, generator=generator)
        for start in range(0, sample_count, batch_size):
            batch = order[start : start + batch_size].to(device=device)
            yhat, weight = prediction(
                train_feature.index_select(0, batch),
                train_base.index_select(0, batch),
                train_anchor.index_select(0, batch),
                train_ridge.index_select(0, batch),
            )
            loss = (yhat - train_target.index_select(0, batch)).square().mean()
            if identity_weight > 0.0:
                loss = loss + identity_weight * (weight - 1.0).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        pred_residual.anchor_ridge_gate.eval()
        with torch.no_grad():
            hold_pred, _ = prediction(
                hold_feature,
                hold_base,
                hold_anchor,
                hold_ridge,
            )
            hold_mse = float((hold_pred - hold_target).square().mean().item())
        if hold_mse + 1.0e-12 < best_soft_hold_mse:
            best_soft_hold_mse = hold_mse
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in pred_residual.anchor_ridge_gate.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
        pred_residual.anchor_ridge_gate.train()

    pred_residual.anchor_ridge_gate.eval()
    if best_state is not None:
        pred_residual.anchor_ridge_gate.load_state_dict(best_state)
    with torch.no_grad():
        hold_probability = pred_residual.anchor_ridge_gate_weights_from_features(
            hold_feature.unsqueeze(1),
            hard=False,
        ).squeeze(1)
    raw_thresholds = gate_cfg.get(
        "threshold_grid",
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    ) or []
    threshold_grid = sorted(
        {
            min(1.0, max(0.0, float(value)))
            for value in raw_thresholds
        }
    )
    if not threshold_grid:
        threshold_grid = [0.5]
    best_hard_hold_mse = float("inf")
    best_thresholds = (0.5, 0.5)
    with torch.no_grad():
        for anchor_threshold in threshold_grid:
            anchor_on = (
                hold_probability[:, :1] >= float(anchor_threshold)
            ).to(dtype=hold_base.dtype)
            for ridge_threshold in threshold_grid:
                ridge_on = (
                    hold_probability[:, 1:] >= float(ridge_threshold)
                ).to(dtype=hold_base.dtype)
                hard_pred = (
                    hold_base
                    + anchor_on * hold_anchor
                    + ridge_on * hold_ridge
                )
                hard_mse = float(
                    (hard_pred - hold_target).square().mean().item()
                )
                if hard_mse + 1.0e-12 < best_hard_hold_mse:
                    best_hard_hold_mse = hard_mse
                    best_thresholds = (
                        float(anchor_threshold),
                        float(ridge_threshold),
                    )
    pred_residual.set_anchor_ridge_gate_thresholds(*best_thresholds)
    min_improvement = max(
        0.0,
        float(gate_cfg.get("min_holdout_mse_improvement", 0.0)),
    )
    adopted = best_hard_hold_mse < identity_hold_mse - min_improvement
    pred_residual.anchor_ridge_gate_fitted.fill_(bool(adopted))
    with torch.no_grad():
        all_feature = features.reshape(-1, int(features.shape[-1])).to(device)
        if adopted:
            all_weight = pred_residual.anchor_ridge_gate_weights_from_features(
                all_feature.unsqueeze(1)
            ).squeeze(1)
        else:
            all_weight = torch.ones(
                int(all_feature.shape[0]), 2, device=device
            )
    for parameter, requires_grad in zip(gate_parameters, original_requires_grad):
        parameter.requires_grad_(requires_grad)
    return {
        "enable": True,
        "adopted": bool(adopted),
        "source_split": "val",
        "fit_windows": windows,
        "train_window_range": [0, int(train_windows)],
        "holdout_window_range": [int(train_windows), windows],
        "test_labels_used_for_fit": False,
        "best_epoch": int(best_epoch),
        "identity_holdout_mse": float(identity_hold_mse),
        "best_soft_holdout_mse": float(best_soft_hold_mse),
        "best_gated_holdout_mse": float(best_hard_hold_mse),
        "holdout_mse_improvement_pct": float(
            100.0 * (identity_hold_mse - best_hard_hold_mse) / max(identity_hold_mse, 1.0e-12)
        ),
        "anchor_threshold": float(best_thresholds[0]),
        "ridge_threshold": float(best_thresholds[1]),
        "anchor_weight_mean": float(all_weight[:, 0].mean().item()),
        "anchor_weight_min": float(all_weight[:, 0].min().item()),
        "anchor_weight_max": float(all_weight[:, 0].max().item()),
        "ridge_weight_mean": float(all_weight[:, 1].mean().item()),
        "ridge_weight_min": float(all_weight[:, 1].min().item()),
        "ridge_weight_max": float(all_weight[:, 1].max().item()),
        "anchor_activation_rate": float(all_weight[:, 0].mean().item()),
        "ridge_activation_rate": float(all_weight[:, 1].mean().item()),
        "independent_binary_gates": True,
        "feature_dim": int(features.shape[-1]),
        "hidden_dim": int(pred_residual.anchor_ridge_gate_hidden_dim),
    }


def _solve_calendar_residual_coefficients(
    xtx: torch.Tensor,
    xty: Optional[torch.Tensor],
    n_rows: int,
    n_windows: int,
    cfg: dict,
) -> Tuple[Optional[torch.Tensor], Dict[str, object]]:
    if xty is None or n_rows <= 0:
        return None, {"enable": False, "reason": "no_fit_rows"}
    feat_dim = int(xtx.shape[0])
    ridge = max(0.0, float(cfg.get("ridge", 1.0e-3)))
    shrink = float(cfg.get("shrink", 1.0))
    max_abs = float(cfg.get("max_abs", 0.0))
    eye = torch.eye(feat_dim, dtype=torch.float64, device=xtx.device)
    if not bool(cfg.get("regularize_bias", False)) and feat_dim > 0:
        eye[0, 0] = 0.0
    coef_fc = torch.linalg.solve(xtx + ridge * eye, xty)
    coef_cf = coef_fc.transpose(0, 1).to(dtype=torch.float32) * float(shrink)
    if max_abs > 0.0:
        coef_cf = coef_cf.clamp(min=-max_abs, max=max_abs)
    return coef_cf.detach(), {
        "enable": True,
        "source_split": str(cfg.get("source_split", "train")),
        "fit_windows": int(n_windows),
        "fit_rows": int(n_rows),
        "feature_dim": int(feat_dim),
        "ridge": float(ridge),
        "shrink": float(shrink),
        "max_abs": float(max_abs),
        "coef_mean_abs": float(coef_cf.abs().mean().item()),
        "coef_max_abs": float(coef_cf.abs().max().item()),
    }


def _fit_calendar_residual_from_prediction_parts(
    idx_parts: List[torch.Tensor],
    y_true_parts: List[torch.Tensor],
    y_pred_parts: List[torch.Tensor],
    calendar_feature_tf: torch.Tensor,
    input_len: int,
    cfg: dict,
) -> Tuple[Optional[torch.Tensor], Dict[str, object]]:
    if len(idx_parts) == 0:
        return None, {"enable": False, "reason": "empty_prediction_parts"}
    device = calendar_feature_tf.device
    feat_dim = int(calendar_feature_tf.shape[1])
    xtx = torch.zeros((feat_dim, feat_dim), dtype=torch.float64, device=device)
    xty = None
    n_rows = 0
    n_windows = 0
    max_windows = int(cfg.get("max_windows", 0) or 0)
    for idx_part, y_true_part, y_pred_part in zip(idx_parts, y_true_parts, y_pred_parts):
        if max_windows > 0 and n_windows >= max_windows:
            break
        idx = idx_part.to(device=device, dtype=torch.long)
        y_true = y_true_part.to(device=device)
        y_pred = y_pred_part.to(device=device, dtype=y_true.dtype)
        if max_windows > 0 and n_windows + int(idx.shape[0]) > max_windows:
            keep = max(0, max_windows - n_windows)
            idx = idx[:keep]
            y_true = y_true[:keep]
            y_pred = y_pred[:keep]
        feat_bhf = calendar_feature_batch(
            calendar_feature_tf=calendar_feature_tf,
            query_start_abs_b=idx,
            input_len=int(input_len),
            pred_len=int(y_true.shape[-1]),
        ).to(device=device, dtype=torch.float64)
        feat_nf = feat_bhf.reshape(-1, feat_dim)
        residual_nc = (y_true - y_pred).permute(0, 2, 1).reshape(-1, y_true.shape[1]).to(dtype=torch.float64)
        xtx += feat_nf.transpose(0, 1).matmul(feat_nf)
        batch_xty = feat_nf.transpose(0, 1).matmul(residual_nc)
        xty = batch_xty if xty is None else xty + batch_xty
        n_rows += int(feat_nf.shape[0])
        n_windows += int(idx.shape[0])
    coef_cf, summary = _solve_calendar_residual_coefficients(
        xtx=xtx,
        xty=xty,
        n_rows=n_rows,
        n_windows=n_windows,
        cfg=cfg,
    )
    summary["fit_target"] = "prediction_parts"
    return coef_cf, summary


@torch.no_grad()
def fit_calendar_residual_correction(
    model: nn.Module,
    loader: DataLoader,
    cluster_id_c: torch.Tensor,
    device: torch.device,
    calendar_feature_tf: torch.Tensor,
    input_len: int,
    eval_start: int,
    cfg: dict,
    history_anchor_cfg: Optional[dict] = None,
    observed_history_tc: Optional[torch.Tensor] = None,
    model_train_stat_adapter_pc: Optional[torch.Tensor] = None,
    model_train_stat_adapter_cfg: Optional[dict] = None,
) -> Tuple[Optional[torch.Tensor], Dict[str, object]]:
    if len(loader) == 0:
        return None, {"enable": False, "reason": "empty_loader"}
    max_windows = int(cfg.get("max_windows", 0) or 0)

    model.eval()
    feat_dim = int(calendar_feature_tf.shape[1])
    xtx = torch.zeros((feat_dim, feat_dim), dtype=torch.float64, device=device)
    xty = None
    n_rows = 0
    n_windows = 0

    for x, y, idx in loader:
        if max_windows > 0 and n_windows >= max_windows:
            break
        if max_windows > 0 and n_windows + int(x.shape[0]) > max_windows:
            keep = max(0, max_windows - n_windows)
            x = x[:keep]
            y = y[:keep]
            idx = idx[:keep]
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
        yhat = model(x_model, cluster_id_c)
        yhat = apply_history_anchor_adapter(
            yhat,
            base_pred_bch=yhat,
            observed_history_tc=observed_history_tc,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len),
            cfg=history_anchor_cfg,
        )
        feat_bhf = calendar_feature_batch(
            calendar_feature_tf=calendar_feature_tf,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len),
            pred_len=int(y.shape[-1]),
        ).to(device=device, dtype=torch.float64)
        feat_nf = feat_bhf.reshape(-1, feat_dim)
        residual_nc = (y - yhat).permute(0, 2, 1).reshape(-1, y.shape[1]).to(dtype=torch.float64)
        xtx += feat_nf.transpose(0, 1).matmul(feat_nf)
        batch_xty = feat_nf.transpose(0, 1).matmul(residual_nc)
        xty = batch_xty if xty is None else xty + batch_xty
        n_rows += int(feat_nf.shape[0])
        n_windows += int(x.shape[0])

    coef_cf, summary = _solve_calendar_residual_coefficients(
        xtx=xtx,
        xty=xty,
        n_rows=n_rows,
        n_windows=n_windows,
        cfg=cfg,
    )
    summary["fit_target"] = "base_path"
    return coef_cf, summary


@torch.no_grad()
def fit_calendar_residual_correction_from_eval_path(
    model: nn.Module,
    gate: ClusterwiseMoEGate,
    lambda_kp: torch.Tensor,
    penalty_names: List[str],
    penalty_fns: Dict[str, callable],
    loader: DataLoader,
    cluster_id_c: torch.Tensor,
    K: int,
    moe_cfg: dict,
    device: torch.device,
    calendar_feature_tf: torch.Tensor,
    input_len: int,
    cfg: dict,
    channel_count: int,
    select_ranks: Optional[List[int]] = None,
    mse_weight: float = 1.0,
    gate_entropy_weight: float = 0.0,
    gate_balance_weight: float = 0.0,
    gate_soft_weight: float = 0.0,
    gate_entropy_target_frac: float = 0.0,
    penalty_scale: Optional[torch.Tensor] = None,
    dynamic_lambda: Optional[ClusterwiseDynamicLambda] = None,
    lambda_min_kp: Optional[torch.Tensor] = None,
    mae_objective_weight=0.0,
    mae_objective_kind: str = "l1",
    mae_objective_beta: float = 1.0,
    pred_residual: Optional[ClusterwisePredResidualMoE] = None,
    pred_residual_selector: Optional[nn.Module] = None,
    pred_residual_scale_c: Optional[torch.Tensor] = None,
    eval_start: int = 0,
    history_anchor_cfg: Optional[dict] = None,
    observed_history_tc: Optional[torch.Tensor] = None,
    model_train_stat_adapter_pc: Optional[torch.Tensor] = None,
    model_train_stat_adapter_cfg: Optional[dict] = None,
    train_stat_anchor_pc: Optional[torch.Tensor] = None,
    train_residual_anchor_phc: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], Dict[str, object]]:
    if len(loader) == 0:
        return None, {"enable": False, "reason": "empty_loader"}
    max_windows = int(cfg.get("max_windows", 0) or 0)
    try:
        loader_windows = int(len(loader.dataset))
    except Exception:
        loader_windows = int(len(loader))
    collect_limit = loader_windows if max_windows <= 0 else min(loader_windows, max_windows)
    diagnostic_collector: Dict[str, object] = {"limit": int(collect_limit), "count": 0, "parts": {}}
    eval_loop(
        model=model,
        gate=gate,
        lambda_kp=lambda_kp,
        penalty_names=penalty_names,
        penalty_fns=penalty_fns,
        loader=loader,
        cluster_id_c=cluster_id_c,
        K=K,
        moe_cfg=moe_cfg,
        device=device,
        select_ranks=select_ranks,
        collect_plot=False,
        channel_count=channel_count,
        mse_weight=mse_weight,
        gate_entropy_weight=gate_entropy_weight,
        gate_balance_weight=gate_balance_weight,
        gate_soft_weight=gate_soft_weight,
        gate_entropy_target_frac=gate_entropy_target_frac,
        penalty_scale=penalty_scale,
        dynamic_lambda=dynamic_lambda,
        lambda_min_kp=lambda_min_kp,
        mae_objective_weight=mae_objective_weight,
        mae_objective_kind=mae_objective_kind,
        mae_objective_beta=mae_objective_beta,
        pred_residual=pred_residual,
        pred_residual_selector=pred_residual_selector,
        pred_residual_scale_c=pred_residual_scale_c,
        eval_start=eval_start,
        history_anchor_cfg=history_anchor_cfg,
        observed_history_tc=observed_history_tc,
        input_len=input_len,
        model_train_stat_adapter_pc=model_train_stat_adapter_pc,
        model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
        train_stat_anchor_pc=train_stat_anchor_pc,
        train_residual_anchor_phc=train_residual_anchor_phc,
        calendar_feature_tf=None,
        calendar_residual_coef_cf=None,
        diagnostic_collector=diagnostic_collector,
    )
    parts = diagnostic_collector.get("parts", {}) or {}
    coef_cf, summary = _fit_calendar_residual_from_prediction_parts(
        idx_parts=list(parts.get("idx", [])),
        y_true_parts=list(parts.get("y_true", [])),
        y_pred_parts=list(parts.get("y_final", [])),
        calendar_feature_tf=calendar_feature_tf,
        input_len=input_len,
        cfg=cfg,
    )
    summary["fit_target"] = "final_eval_path"
    summary["fit_eval_start"] = int(eval_start)
    return coef_cf, summary


@torch.no_grad()
def evaluate_gate_penalty_hit_metrics(
    model: nn.Module,
    gate: ClusterwiseMoEGate,
    pred_residual: Optional[ClusterwisePredResidualMoE],
    loader: DataLoader,
    cluster_id_c: torch.Tensor,
    K: int,
    moe_cfg: dict,
    device: torch.device,
    penalty_names: List[str],
    penalty_fns: Dict[str, callable],
    penalty_scale: Optional[torch.Tensor],
    select_ranks: Optional[List[int]],
    gate_soft_weight: float,
    label_min_improvement: float = 0.0,
    history_anchor_cfg: Optional[dict] = None,
    observed_history_tc: Optional[torch.Tensor] = None,
    input_len: int = 0,
    eval_start: int = 0,
    model_train_stat_adapter_pc: Optional[torch.Tensor] = None,
    model_train_stat_adapter_cfg: Optional[dict] = None,
    train_stat_anchor_pc: Optional[torch.Tensor] = None,
    train_residual_anchor_phc: Optional[torch.Tensor] = None,
    learnable_output_anchor: Optional[ClusterwiseLearnableOutputAnchor] = None,
    allowed_mask_kp: Optional[torch.Tensor] = None,
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

    P = len(penalty_names)
    cid_c = cluster_id_c.detach().to(device=device, dtype=torch.long)
    allowed_mask_device = None
    if allowed_mask_kp is not None:
        allowed_mask_device = allowed_mask_kp.detach().to(device=device, dtype=torch.bool)
        if tuple(allowed_mask_device.shape) != (int(K), int(P)):
            raise ValueError("allowed_mask_kp must have shape [K,P] for gate penalty hit diagnostics.")
    total = positive_total = hit_total = positive_hit_total = 0
    selected_positive = 0
    base_se = oracle_se = selected_se = 0.0
    denom = 0
    oracle_count_p = torch.zeros(P, dtype=torch.long)
    selected_count_p = torch.zeros(P, dtype=torch.long)
    positive_oracle_count_p = torch.zeros(P, dtype=torch.long)
    min_improvement = max(0.0, float(label_min_improvement))

    for x, y, idx in loader:
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
        fixed_expert_delta_bch = build_moe_output_anchor_fixed_expert_delta(
            y_base,
            x_bcl=x,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len),
            moe_cfg=moe_cfg,
            moe_enable=moe_enable,
            observed_history_tc=observed_history_tc,
            train_stat_anchor_pc=train_stat_anchor_pc,
            train_residual_anchor_phc=train_residual_anchor_phc,
            learnable_output_anchor=learnable_output_anchor,
            cluster_id_c=cluster_id_c,
        )
        candidate_base_bch = _fixed_expert_candidate_base(
            y_base,
            pred_residual,
            fixed_expert_delta_bch,
        )
        feat_bkf = _build_gate_routing_features(
            x,
            candidate_base_bch,
            cluster_id_c,
            K,
            mode=gate_feature_mode,
        )
        route_pen_bkp = _router_penalty_context_from_history(
            x_bcl=x,
            yhat_base_bch=candidate_base_bch,
            penalty_names=penalty_names,
            penalty_fns=penalty_fns,
            penalty_scale=penalty_scale,
            cluster_id_c=cluster_id_c,
            K=K,
            router_mode=router_mode,
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
            query_start_abs_b=query_start_abs_b,
            fixed_expert_delta_bch=fixed_expert_delta_bch,
        )
        residuals = pred_out.get("residuals")
        intervention_bcp = pred_out.get("intervention_bcp")
        selector_bcp = pred_out.get("selector_bcp")
        alpha_cp = pred_out.get("alpha_cp")
        if residuals is None or intervention_bcp is None or alpha_cp is None or residuals.numel() == 0:
            continue
        if selector_bcp is None:
            selector_bcp = torch.ones_like(intervention_bcp)

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
            learnable_output_anchor=learnable_output_anchor,
            cluster_id_c=cluster_id_c,
        )
        if cand_bcpH is None:
            continue
        err_bcp = (cand_bcpH - y.unsqueeze(2)).pow(2).mean(dim=-1)
        base_err_bc = (y_base_final - y).pow(2).mean(dim=-1)
        if allowed_mask_device is not None:
            allowed_bcp = allowed_mask_device.index_select(0, cid_c).unsqueeze(0)
            err_for_oracle_bcp = err_bcp.masked_fill(~allowed_bcp, float("inf"))
        else:
            err_for_oracle_bcp = err_bcp
        oracle_penalty_err_bc, oracle_p_bc = err_for_oracle_bcp.min(dim=-1)
        oracle_err_bc = torch.where(torch.isfinite(oracle_penalty_err_bc), oracle_penalty_err_bc, base_err_bc)
        route_bcp = mask_bkp[:, cid_c, :]
        probs_bcp = probs_bkp[:, cid_c, :]
        selected_score_bcp = route_bcp * probs_bcp
        selected_p_bc = selected_score_bcp.argmax(dim=-1)
        selected_err_bc = err_bcp.gather(-1, selected_p_bc.unsqueeze(-1)).squeeze(-1)

        positive_bc = (base_err_bc - oracle_err_bc) > min_improvement
        hit_bc = selected_p_bc == oracle_p_bc
        selected_positive_bc = (base_err_bc - selected_err_bc) > min_improvement

        total += int(hit_bc.numel())
        positive_total += int(positive_bc.sum().item())
        hit_total += int(hit_bc.sum().item())
        positive_hit_total += int((hit_bc & positive_bc).sum().item())
        selected_positive += int(selected_positive_bc.sum().item())
        base_se += float((y_base_final - y).pow(2).sum().item())
        oracle_se += float(oracle_err_bc.sum().item() * y.shape[-1])
        selected_se += float(selected_err_bc.sum().item() * y.shape[-1])
        denom += int(y.numel())
        oracle_count_p += torch.bincount(oracle_p_bc.reshape(-1).detach().cpu(), minlength=P)[:P]
        selected_count_p += torch.bincount(selected_p_bc.reshape(-1).detach().cpu(), minlength=P)[:P]
        positive_oracle_count_p += torch.bincount(
            oracle_p_bc[positive_bc].reshape(-1).detach().cpu(),
            minlength=P,
        )[:P]

    if total <= 0:
        return None
    base_mse = base_se / max(denom, 1)
    oracle_mse = oracle_se / max(denom, 1)
    selected_mse = selected_se / max(denom, 1)
    return {
        "enable": True,
        "samples": int(total),
        "label_min_improvement": float(min_improvement),
        "top1_hit_rate_all": float(hit_total / max(total, 1)),
        "top1_hit_rate_on_positive_oracle": float(positive_hit_total / max(positive_total, 1)),
        "oracle_positive_rate": float(positive_total / max(total, 1)),
        "selected_positive_rate": float(selected_positive / max(total, 1)),
        "base_mse": float(base_mse),
        "oracle_mse": float(oracle_mse),
        "selected_top1_mse": float(selected_mse),
        "oracle_gain_pct_vs_base": float(100.0 * (base_mse - oracle_mse) / max(abs(base_mse), 1.0e-12)),
        "selected_top1_gain_pct_vs_base": float(
            100.0 * (base_mse - selected_mse) / max(abs(base_mse), 1.0e-12)
        ),
        "oracle_count": {name: int(oracle_count_p[i].item()) for i, name in enumerate(penalty_names)},
        "positive_oracle_count": {
            name: int(positive_oracle_count_p[i].item()) for i, name in enumerate(penalty_names)
        },
        "selected_count": {name: int(selected_count_p[i].item()) for i, name in enumerate(penalty_names)},
    }


def _pearson_list(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt((x * x).sum()) * np.sqrt((y * y).sum()))
    if denom <= 1.0e-12:
        return None
    return float((x * y).sum() / denom)


def _explainability_train_subsplit_ranges(
    num_windows: int,
    holdout_fraction: float = 0.30,
) -> Dict[str, Tuple[int, int]]:
    n = int(num_windows)
    if n <= 0:
        return {}
    if n == 1:
        return {"train_fit": (0, 1)}
    frac = max(0.0, min(float(holdout_fraction), 0.95))
    holdout = int(math.ceil(float(n) * frac))
    holdout = max(1, min(holdout, n - 1))
    cut = n - holdout
    return {
        "train_fit": (0, cut),
        "train_holdout": (cut, n),
    }


def _cluster_route_label_feature_diagnostics(
    feat_bkf: torch.Tensor,
    route_label_bk: torch.Tensor,
    penalty_names: List[str],
    feature_names: Optional[List[str]] = None,
) -> Dict[str, object]:
    label_names = ["skip"] + [str(name) for name in penalty_names]
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(int(feat_bkf.shape[-1]))]
    else:
        feature_names = [str(name) for name in feature_names]

    def _label_name(label: int) -> Optional[str]:
        if 0 <= int(label) < len(label_names):
            return label_names[int(label)]
        return None

    def _majority_label(labels_n: torch.Tensor, label_count: int) -> Tuple[int, float]:
        if labels_n.numel() <= 0:
            return -1, 0.0
        counts = torch.bincount(labels_n.to(dtype=torch.long), minlength=label_count)[:label_count]
        label = int(counts.argmax().item())
        acc = float(counts[label].item() / max(int(labels_n.numel()), 1))
        return label, acc

    feat = feat_bkf.detach().cpu().to(dtype=torch.float32)
    labels = route_label_bk.detach().cpu().to(dtype=torch.long)
    if feat.dim() != 3 or labels.dim() != 2:
        raise ValueError("feat_bkf must be [B,K,F] and route_label_bk must be [B,K].")
    if int(feat.shape[0]) != int(labels.shape[0]) or int(feat.shape[1]) != int(labels.shape[1]):
        raise ValueError("feat_bkf and route_label_bk must share [B,K].")

    B, K, F = int(feat.shape[0]), int(feat.shape[1]), int(feat.shape[2])
    per_cluster = []
    label_count = len(label_names)
    for k in range(K):
        valid = (labels[:, k] >= 0) & (labels[:, k] < label_count)
        labels_k = labels[valid, k]
        feat_kf = feat[valid, k, :]
        samples = int(labels_k.numel())
        counts = torch.bincount(labels_k, minlength=label_count)[:label_count] if samples > 0 else torch.zeros(label_count, dtype=torch.long)
        rates = counts.to(dtype=torch.float64) / max(samples, 1)
        majority_label, majority_acc = _majority_label(labels_k, label_count)
        entropy_bits = 0.0
        for rate in rates.tolist():
            if float(rate) > 0.0:
                entropy_bits -= float(rate) * math.log(float(rate), 2)

        best_stump = None
        if samples >= 2 and F > 0 and int((counts > 0).sum().item()) >= 2:
            best_acc = -1.0
            best_payload = None
            for f in range(F):
                values = feat_kf[:, f]
                finite = torch.isfinite(values)
                if int(finite.sum().item()) < 2:
                    continue
                values_f = values[finite]
                labels_f = labels_k[finite]
                unique = torch.unique(values_f).sort().values
                if int(unique.numel()) < 2:
                    continue
                if int(unique.numel()) <= 16:
                    thresholds = (unique[:-1] + unique[1:]) * 0.5
                else:
                    quantiles = torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90], dtype=values_f.dtype)
                    thresholds = torch.unique(torch.quantile(values_f, quantiles)).sort().values
                for threshold in thresholds:
                    left = values_f <= threshold
                    right = ~left
                    if int(left.sum().item()) <= 0 or int(right.sum().item()) <= 0:
                        continue
                    left_label, _ = _majority_label(labels_f[left], label_count)
                    right_label, _ = _majority_label(labels_f[right], label_count)
                    pred = torch.where(
                        left,
                        torch.full_like(labels_f, int(left_label)),
                        torch.full_like(labels_f, int(right_label)),
                    )
                    acc = float((pred == labels_f).to(dtype=torch.float32).mean().item())
                    if acc > best_acc:
                        best_acc = acc
                        best_payload = {
                            "feature": feature_names[f] if f < len(feature_names) else f"feature_{f}",
                            "feature_index": int(f),
                            "threshold": float(threshold.item()),
                            "op": "<=",
                            "left_label": _label_name(left_label),
                            "right_label": _label_name(right_label),
                            "accuracy": acc,
                            "lift_vs_majority": float(acc - majority_acc),
                        }
            best_stump = best_payload

        label_counts = {label_names[i]: int(counts[i].item()) for i in range(label_count)}
        label_rates = {label_names[i]: float(rates[i].item()) for i in range(label_count)}
        per_cluster.append(
            {
                "cluster_id": int(k),
                "samples": samples,
                "label_counts": label_counts,
                "label_rates": label_rates,
                "majority_label": _label_name(majority_label),
                "majority_acc": float(majority_acc),
                "entropy_bits": float(entropy_bits),
                "best_stump": best_stump,
            }
        )

    return {
        "samples": int(B * K),
        "label_names": label_names,
        "feature_names": feature_names,
        "per_cluster": per_cluster,
    }


def _cluster_route_label_phase_diagnostics(
    query_start_abs_b: torch.Tensor,
    route_label_bk: torch.Tensor,
    penalty_names: List[str],
    periods: List[int],
    num_bins: int = 8,
    phase_offset: int = 0,
) -> Dict[str, object]:
    label_names = ["skip"] + [str(name) for name in penalty_names]
    label_count = len(label_names)

    def _label_name(label: int) -> Optional[str]:
        if 0 <= int(label) < label_count:
            return label_names[int(label)]
        return None

    def _counts_payload(labels_n: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, int], Dict[str, float], Optional[str], float]:
        if labels_n.numel() <= 0:
            counts = torch.zeros(label_count, dtype=torch.long)
            return counts, {name: 0 for name in label_names}, {name: 0.0 for name in label_names}, None, 0.0
        counts = torch.bincount(labels_n.to(dtype=torch.long), minlength=label_count)[:label_count]
        total = int(labels_n.numel())
        majority = int(counts.argmax().item())
        label_counts = {label_names[i]: int(counts[i].item()) for i in range(label_count)}
        label_rates = {label_names[i]: float(counts[i].item() / max(total, 1)) for i in range(label_count)}
        majority_acc = float(counts[majority].item() / max(total, 1))
        return counts, label_counts, label_rates, _label_name(majority), majority_acc

    starts = query_start_abs_b.detach().cpu().to(dtype=torch.long).reshape(-1)
    labels = route_label_bk.detach().cpu().to(dtype=torch.long)
    if labels.dim() != 2:
        raise ValueError("route_label_bk must have shape [B,K].")
    if int(labels.shape[0]) != int(starts.numel()):
        raise ValueError("query_start_abs_b length must match route_label_bk batch dimension.")

    clean_periods = []
    for period in periods:
        period_i = int(period)
        if period_i > 0 and period_i not in clean_periods:
            clean_periods.append(period_i)
    bins_requested = int(num_bins)
    if bins_requested <= 0:
        bins_requested = 8

    K = int(labels.shape[1])
    per_period = []
    for period in clean_periods:
        bins = max(1, min(int(bins_requested), int(period)))
        phase_b = (starts + int(phase_offset)).remainder(int(period))
        bin_b = torch.div(phase_b * bins, int(period), rounding_mode="floor").clamp(min=0, max=bins - 1)
        period_clusters = []
        for k in range(K):
            valid = (labels[:, k] >= 0) & (labels[:, k] < label_count)
            labels_k = labels[valid, k]
            bin_k = bin_b[valid]
            samples = int(labels_k.numel())
            _, label_counts, label_rates, global_majority_label, global_majority_acc = _counts_payload(labels_k)
            bin_payloads = []
            phase_correct = 0
            for b in range(bins):
                bin_mask = bin_k == int(b)
                labels_bin = labels_k[bin_mask]
                bin_samples = int(labels_bin.numel())
                counts, bin_counts, bin_rates, majority_label, majority_acc = _counts_payload(labels_bin)
                if bin_samples > 0:
                    phase_correct += int(counts.max().item())
                start_phase = int(math.floor(float(b) * float(period) / float(bins)))
                end_phase = int(math.floor(float(b + 1) * float(period) / float(bins)))
                bin_payloads.append(
                    {
                        "bin": int(b),
                        "phase_start": start_phase,
                        "phase_end": end_phase,
                        "samples": bin_samples,
                        "label_counts": bin_counts,
                        "label_rates": bin_rates,
                        "majority_label": majority_label,
                        "majority_acc": float(majority_acc),
                    }
                )
            phase_majority_acc = float(phase_correct / max(samples, 1))
            period_clusters.append(
                {
                    "cluster_id": int(k),
                    "samples": samples,
                    "label_counts": label_counts,
                    "label_rates": label_rates,
                    "global_majority_label": global_majority_label,
                    "global_majority_acc": float(global_majority_acc),
                    "phase_majority_acc": phase_majority_acc,
                    "lift_vs_global": float(phase_majority_acc - global_majority_acc),
                    "bins": bin_payloads,
                }
            )
        per_period.append(
            {
                "period": int(period),
                "num_bins": int(bins),
                "per_cluster": period_clusters,
            }
        )

    return {
        "samples": int(labels.numel()),
        "window_count": int(labels.shape[0]),
        "label_names": label_names,
        "periods": clean_periods,
        "num_bins_requested": int(bins_requested),
        "phase_offset": int(phase_offset),
        "per_period": per_period,
    }


def _cluster_top1_confidence_gain_diagnostics(
    *,
    top1_conf_bc: torch.Tensor,
    top1_gain_bc: torch.Tensor,
    top1_p_bc: torch.Tensor,
    top1_active_bc: torch.Tensor,
    skip_bc: torch.Tensor,
    cluster_id_c: torch.Tensor,
    K: int,
    penalty_names: List[str],
    bins: List[float],
) -> Dict[str, object]:
    conf = top1_conf_bc.detach().cpu().to(dtype=torch.float32)
    gain = top1_gain_bc.detach().cpu().to(dtype=torch.float32)
    top1_p = top1_p_bc.detach().cpu().to(dtype=torch.long)
    active = top1_active_bc.detach().cpu().to(dtype=torch.bool)
    skipped = skip_bc.detach().cpu().to(dtype=torch.bool)
    cid = cluster_id_c.detach().cpu().to(dtype=torch.long)
    if conf.shape != gain.shape or conf.shape != top1_p.shape or conf.shape != active.shape or conf.shape != skipped.shape:
        raise ValueError("top1 confidence diagnostics inputs must share shape [B,C].")
    if conf.dim() != 2:
        raise ValueError("top1 confidence diagnostics inputs must have shape [B,C].")
    if int(cid.numel()) != int(conf.shape[1]):
        raise ValueError("cluster_id_c length must match confidence channel dimension.")

    clean_bins = sorted({float(v) for v in bins if math.isfinite(float(v))})
    if len(clean_bins) < 2:
        clean_bins = [0.0, 0.5, 1.0]
    if clean_bins[0] > 0.0:
        clean_bins = [0.0] + clean_bins
    if clean_bins[-1] < 1.0:
        clean_bins = clean_bins + [1.0]
    label_names = [str(name) for name in penalty_names]

    def _bin_payload(mask_bc: torch.Tensor, lower: float, upper: float, is_last: bool) -> Dict[str, object]:
        if is_last:
            bin_mask = mask_bc & (conf >= float(lower)) & (conf <= float(upper))
        else:
            bin_mask = mask_bc & (conf >= float(lower)) & (conf < float(upper))
        samples = int(bin_mask.sum().item())
        active_mask = bin_mask & active
        skipped_mask = bin_mask & skipped
        active_count = int(active_mask.sum().item())
        skipped_count = int(skipped_mask.sum().item())
        if samples > 0:
            mean_gain = float(gain[bin_mask].mean().item())
            positive_rate = float((gain[bin_mask] > 0.0).to(dtype=torch.float32).mean().item())
        else:
            mean_gain = 0.0
            positive_rate = 0.0
        if active_count > 0:
            active_mean_gain = float(gain[active_mask].mean().item())
            active_positive_rate = float((gain[active_mask] > 0.0).to(dtype=torch.float32).mean().item())
            harmful_not_skipped_rate = float((gain[active_mask] <= 0.0).to(dtype=torch.float32).mean().item())
        else:
            active_mean_gain = 0.0
            active_positive_rate = 0.0
            harmful_not_skipped_rate = 0.0
        if skipped_count > 0:
            skipped_positive_rate = float((gain[skipped_mask] > 0.0).to(dtype=torch.float32).mean().item())
        else:
            skipped_positive_rate = 0.0
        return {
            "low": float(lower),
            "high": float(upper),
            "samples": samples,
            "active_count": active_count,
            "skipped_count": skipped_count,
            "mean_gain_mse": mean_gain,
            "positive_rate": positive_rate,
            "active_mean_gain_mse": active_mean_gain,
            "active_positive_rate": active_positive_rate,
            "harmful_not_skipped_rate": harmful_not_skipped_rate,
            "skipped_positive_rate": skipped_positive_rate,
        }

    def _group_payload(mask_bc: torch.Tensor) -> Dict[str, object]:
        payload_bins = []
        for i in range(len(clean_bins) - 1):
            payload_bins.append(
                _bin_payload(
                    mask_bc,
                    lower=clean_bins[i],
                    upper=clean_bins[i + 1],
                    is_last=i == len(clean_bins) - 2,
                )
            )
        samples = int(mask_bc.sum().item())
        if samples > 0:
            mean_gain = float(gain[mask_bc].mean().item())
            positive_rate = float((gain[mask_bc] > 0.0).to(dtype=torch.float32).mean().item())
        else:
            mean_gain = 0.0
            positive_rate = 0.0
        return {
            "samples": samples,
            "mean_gain_mse": mean_gain,
            "positive_rate": positive_rate,
            "bins": payload_bins,
        }

    per_cluster = []
    for k in range(int(K)):
        cluster_mask = (cid == int(k)).view(1, -1).expand_as(conf)
        per_penalty = []
        for p, name in enumerate(label_names):
            per_penalty.append(
                {
                    "penalty": name,
                    **_group_payload(cluster_mask & (top1_p == int(p))),
                }
            )
        per_cluster.append(
            {
                "cluster_id": int(k),
                "all": _group_payload(cluster_mask),
                "per_penalty": per_penalty,
            }
        )

    return {
        "bins": [float(v) for v in clean_bins],
        "penalty_names": label_names,
        "per_cluster": per_cluster,
    }


def _build_penalty_route_learnability_class_features(
    *,
    gate_feat_bkf: torch.Tensor,
    skip_feat_bkf: torch.Tensor,
    cand_feat_bkpf: torch.Tensor,
    gate_prob_bkp: torch.Tensor,
    route_bkp: torch.Tensor,
    intervention_bkp: torch.Tensor,
    selector_bkp: torch.Tensor,
    alpha_bkp: torch.Tensor,
    skip_prob_bk: Optional[torch.Tensor],
    cluster_count: int,
    penalty_names: List[str],
) -> Tuple[torch.Tensor, List[str]]:
    if gate_feat_bkf.dim() != 3:
        raise ValueError("gate_feat_bkf must have shape [B,K,F].")
    if skip_feat_bkf.dim() != 3:
        raise ValueError("skip_feat_bkf must have shape [B,K,F].")
    if cand_feat_bkpf.dim() != 4:
        raise ValueError("cand_feat_bkpf must have shape [B,K,P,F].")
    B, K, gate_dim = [int(v) for v in gate_feat_bkf.shape]
    if tuple(skip_feat_bkf.shape[:2]) != (B, K):
        raise ValueError("skip_feat_bkf must share [B,K] with gate_feat_bkf.")
    if tuple(cand_feat_bkpf.shape[:3]) != (B, K, len(penalty_names)):
        raise ValueError("cand_feat_bkpf must share [B,K,P] with penalty_names.")
    P = int(cand_feat_bkpf.shape[2])
    Q = P + 1
    stat_shape = (B, K, P)
    for name, value in [
        ("gate_prob_bkp", gate_prob_bkp),
        ("route_bkp", route_bkp),
        ("intervention_bkp", intervention_bkp),
        ("selector_bkp", selector_bkp),
    ]:
        if tuple(value.shape) != stat_shape:
            raise ValueError(f"{name} must have shape [B,K,P].")
    if alpha_bkp.dim() == 2:
        if tuple(alpha_bkp.shape) != (K, P):
            raise ValueError("alpha_bkp must have shape [K,P] or [B,K,P].")
        alpha_bkp = alpha_bkp.unsqueeze(0).expand(B, K, P)
    elif tuple(alpha_bkp.shape) != stat_shape:
        raise ValueError("alpha_bkp must have shape [K,P] or [B,K,P].")
    if skip_prob_bk is None:
        skip_prob_bk = torch.zeros(B, K, device=gate_feat_bkf.device, dtype=gate_feat_bkf.dtype)
    if tuple(skip_prob_bk.shape) != (B, K):
        raise ValueError("skip_prob_bk must have shape [B,K].")

    dtype = gate_feat_bkf.dtype
    device = gate_feat_bkf.device
    zero_bk1 = torch.zeros(B, K, 1, device=device, dtype=dtype)
    class_gate_prob = torch.cat([zero_bk1, gate_prob_bkp.to(device=device, dtype=dtype)], dim=-1)
    class_route = torch.cat([zero_bk1, route_bkp.to(device=device, dtype=dtype)], dim=-1)
    class_intervention = torch.cat([zero_bk1, intervention_bkp.to(device=device, dtype=dtype)], dim=-1)
    class_selector = torch.cat([zero_bk1, selector_bkp.to(device=device, dtype=dtype)], dim=-1)
    class_alpha = torch.cat([zero_bk1, alpha_bkp.to(device=device, dtype=dtype)], dim=-1)
    class_skip_prob = skip_prob_bk.to(device=device, dtype=dtype).unsqueeze(-1).expand(B, K, Q)
    stat_features = torch.stack(
        [
            class_gate_prob,
            class_route,
            class_intervention,
            class_selector,
            class_alpha,
            class_skip_prob,
        ],
        dim=-1,
    )

    candidate_features = torch.cat(
        [
            skip_feat_bkf.to(device=device, dtype=dtype).unsqueeze(2),
            cand_feat_bkpf.to(device=device, dtype=dtype),
        ],
        dim=2,
    )
    gate_features = gate_feat_bkf.to(device=device, dtype=dtype).unsqueeze(2).expand(B, K, Q, gate_dim)
    class_eye = torch.eye(Q, device=device, dtype=dtype).view(1, 1, Q, Q).expand(B, K, Q, Q)
    cluster_total = int(cluster_count) if int(cluster_count) > 0 else K
    if cluster_total < K:
        raise ValueError("cluster_count cannot be smaller than gate_feat_bkf.shape[1].")
    cluster_eye = torch.eye(cluster_total, device=device, dtype=dtype)[:K].view(1, K, 1, cluster_total)
    cluster_features = cluster_eye.expand(B, K, Q, cluster_total)
    features = torch.cat(
        [
            gate_features,
            candidate_features,
            stat_features,
            class_eye,
            cluster_features,
        ],
        dim=-1,
    ).contiguous()

    feature_names = (
        [f"gate_feature_{i}" for i in range(gate_dim)]
        + [f"candidate_feature_{i}" for i in range(int(candidate_features.shape[-1]))]
        + ["gate_prob", "route_weight", "intervention", "selector", "alpha", "skip_prob"]
        + ["class_skip"]
        + [f"class_{name}" for name in penalty_names]
        + [f"cluster_{k}" for k in range(cluster_total)]
    )
    return features, feature_names


def _scatter_mean_bcpf_to_bkpf(values_bcpf: torch.Tensor, cluster_id_c: torch.Tensor, K: int) -> torch.Tensor:
    if values_bcpf.dim() != 4:
        raise ValueError("values_bcpf must have shape [B,C,P,F].")
    B, C, P, F_dim = [int(v) for v in values_bcpf.shape]
    flat_bcf = values_bcpf.reshape(B, C, P * F_dim)
    pooled = scatter_mean_bcf_to_bkf(flat_bcf, cluster_id_c, int(K))
    return pooled.reshape(B, int(K), P, F_dim)


@torch.no_grad()
def _collect_penalty_route_learnability_tensors(
    *,
    model: nn.Module,
    gate: ClusterwiseMoEGate,
    pred_residual: Optional[ClusterwisePredResidualMoE],
    loader: DataLoader,
    cluster_id_c: torch.Tensor,
    K: int,
    moe_cfg: dict,
    device: torch.device,
    penalty_names: List[str],
    penalty_fns: Dict[str, callable],
    penalty_scale: Optional[torch.Tensor],
    select_ranks: Optional[List[int]],
    gate_soft_weight: float,
    split_name: str,
    feature_mode: str = "base",
    allowed_mask_kp: Optional[torch.Tensor] = None,
    min_abs_improvement: float = 0.0,
    min_rel_improvement: float = 0.0,
    min_candidate_delta_rms: float = 0.0,
    max_batches: int = 0,
    history_anchor_cfg: Optional[dict] = None,
    observed_history_tc: Optional[torch.Tensor] = None,
    input_len: int = 0,
    eval_start: int = 0,
    model_train_stat_adapter_pc: Optional[torch.Tensor] = None,
    model_train_stat_adapter_cfg: Optional[dict] = None,
    train_stat_anchor_pc: Optional[torch.Tensor] = None,
    train_residual_anchor_phc: Optional[torch.Tensor] = None,
    learnable_output_anchor: Optional[ClusterwiseLearnableOutputAnchor] = None,
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
    P = len(penalty_names)
    cid_c = cluster_id_c.detach().to(device=device, dtype=torch.long)
    allowed_mask_device = None
    if allowed_mask_kp is not None:
        allowed_mask_device = allowed_mask_kp.detach().to(device=device, dtype=torch.bool)
        if tuple(allowed_mask_device.shape) != (int(K), int(P)):
            raise ValueError("allowed_mask_kp must have shape [K,P] for route learnability diagnostics.")

    feature_chunks: List[torch.Tensor] = []
    label_chunks: List[torch.Tensor] = []
    current_chunks: List[torch.Tensor] = []
    query_start_chunks: List[torch.Tensor] = []
    gain_chunks: List[torch.Tensor] = []
    feature_names: Optional[List[str]] = None
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
        fixed_expert_delta_bch = build_moe_output_anchor_fixed_expert_delta(
            y_base,
            x_bcl=x,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len),
            moe_cfg=moe_cfg,
            moe_enable=moe_enable,
            observed_history_tc=observed_history_tc,
            train_stat_anchor_pc=train_stat_anchor_pc,
            train_residual_anchor_phc=train_residual_anchor_phc,
            learnable_output_anchor=learnable_output_anchor,
            cluster_id_c=cluster_id_c,
        )
        candidate_base_bch = _fixed_expert_candidate_base(
            y_base,
            pred_residual,
            fixed_expert_delta_bch,
        )
        feat_bkf = _build_gate_routing_features(
            x,
            candidate_base_bch,
            cluster_id_c,
            K,
            mode=gate_feature_mode,
        )
        route_pen_bkp = _router_penalty_context_from_history(
            x_bcl=x,
            yhat_base_bch=candidate_base_bch,
            penalty_names=penalty_names,
            penalty_fns=penalty_fns,
            penalty_scale=penalty_scale,
            cluster_id_c=cluster_id_c,
            K=K,
            router_mode=router_mode,
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

        pred_out = pred_residual(
            x,
            y_base,
            cluster_id_c,
            mask_bkp,
            skip_bk=skip_bk if allow_skip else None,
            query_start_abs_b=query_start_abs_b,
            fixed_expert_delta_bch=fixed_expert_delta_bch,
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
            learnable_output_anchor=learnable_output_anchor,
            cluster_id_c=cluster_id_c,
        )
        if cand_bcpH is None:
            continue
        labels_bk, raw_gain_bk = _cluster_route_oracle_labels_and_gain_from_candidates(
            base_bch=y_base_final,
            cand_bcpH=cand_bcpH,
            y_bch=y,
            cluster_id_c=cid_c,
            K=int(K),
            allowed_mask_kp=allowed_mask_device,
            min_abs_improvement=float(min_abs_improvement),
            min_rel_improvement=float(min_rel_improvement),
            min_candidate_delta_rms=float(min_candidate_delta_rms),
        )
        gain_bk = raw_gain_bk.clamp_min(0.0)

        route_bcp = pred_out.get("route_bcp", mask_bkp[:, cid_c, :])
        intervention_bcp = pred_out.get("intervention_bcp", torch.ones_like(route_bcp))
        selector_bcp = pred_out.get("selector_bcp", torch.ones_like(route_bcp))
        alpha_cp = pred_out.get("alpha_cp")
        if alpha_cp is None:
            alpha_cp = torch.ones(int(cid_c.numel()), P, device=device, dtype=x.dtype)
        skip_feat_bcf = _candidate_selector_features(x, y_base_final, y_base_final, feature_mode=feature_mode)
        cand_feat_parts = [
            _candidate_selector_features(x, y_base_final, cand_bcpH[:, :, p, :], feature_mode=feature_mode)
            for p in range(P)
        ]
        cand_feat_bcpf = torch.stack(cand_feat_parts, dim=2)
        skip_feat_bkf = scatter_mean_bcf_to_bkf(skip_feat_bcf, cluster_id_c, int(K))
        cand_feat_bkpf = _scatter_mean_bcpf_to_bkpf(cand_feat_bcpf, cluster_id_c, int(K))
        route_bkp = scatter_mean_bcf_to_bkf(route_bcp, cluster_id_c, int(K))
        intervention_bkp = scatter_mean_bcf_to_bkf(intervention_bcp, cluster_id_c, int(K))
        selector_bkp = scatter_mean_bcf_to_bkf(selector_bcp, cluster_id_c, int(K))
        alpha_bkp = scatter_mean_bcf_to_bkf(alpha_cp.unsqueeze(0).expand(x.shape[0], -1, -1), cluster_id_c, int(K))
        class_features_bkqf, names = _build_penalty_route_learnability_class_features(
            gate_feat_bkf=feat_bkf,
            skip_feat_bkf=skip_feat_bkf,
            cand_feat_bkpf=cand_feat_bkpf,
            gate_prob_bkp=probs_bkp,
            route_bkp=route_bkp,
            intervention_bkp=intervention_bkp,
            selector_bkp=selector_bkp,
            alpha_bkp=alpha_bkp,
            skip_prob_bk=skip_prob_bk if allow_skip else None,
            cluster_count=int(K),
            penalty_names=penalty_names,
        )
        feature_names = names
        current_penalty_bk = (mask_bkp * probs_bkp).argmax(dim=-1).to(dtype=torch.long) + 1
        if allow_skip:
            current_penalty_bk = torch.where(
                skip_bk > 0.5,
                torch.zeros_like(current_penalty_bk),
                current_penalty_bk,
            )
        feature_chunks.append(class_features_bkqf.detach().cpu().reshape(-1, P + 1, class_features_bkqf.shape[-1]))
        label_chunks.append(labels_bk.detach().cpu().reshape(-1))
        current_chunks.append(current_penalty_bk.detach().cpu().reshape(-1))
        query_start_chunks.append(query_start_abs_b.detach().cpu().view(-1, 1).expand(-1, int(K)).reshape(-1))
        gain_chunks.append(gain_bk.detach().cpu().reshape(-1))

    if not feature_chunks:
        return None
    return {
        "split": str(split_name),
        "features": torch.cat(feature_chunks, dim=0),
        "labels": torch.cat(label_chunks, dim=0),
        "current_pred": torch.cat(current_chunks, dim=0),
        "query_start_abs": torch.cat(query_start_chunks, dim=0),
        "oracle_gain_mse": torch.cat(gain_chunks, dim=0),
        "label_names": ["skip"] + [str(name) for name in penalty_names],
        "feature_names": list(feature_names or []),
    }


def _penalty_route_learnability_metrics_from_scores(
    *,
    scores: torch.Tensor,
    labels: torch.Tensor,
    current_pred: torch.Tensor,
    label_names: List[str],
) -> Dict[str, object]:
    if scores.dim() != 2:
        raise ValueError("scores must have shape [N,num_classes].")
    label_count = int(scores.shape[1])
    if len(label_names) != label_count:
        raise ValueError("label_names length must match scores.shape[1].")
    labels = labels.detach().cpu().to(dtype=torch.long).view(-1)
    current_pred = current_pred.detach().cpu().to(dtype=torch.long).view(-1)
    if int(labels.numel()) != int(scores.shape[0]) or int(current_pred.numel()) != int(scores.shape[0]):
        raise ValueError("scores, labels, and current_pred must share N.")
    valid = (labels >= 0) & (labels < label_count)
    if int(valid.sum().item()) <= 0:
        return {
            "samples": 0,
            "accuracy_all": 0.0,
            "current_accuracy_all": 0.0,
            "majority_accuracy_all": 0.0,
            "accuracy_on_positive_oracle": 0.0,
            "current_accuracy_on_positive_oracle": 0.0,
            "prediction_counts": {name: 0 for name in label_names},
            "current_prediction_counts": {name: 0 for name in label_names},
            "label_counts": {name: 0 for name in label_names},
        }
    labels_v = labels[valid]
    current_v = current_pred[valid].clamp(0, label_count - 1)
    pred_v = scores.detach().cpu()[valid].argmax(dim=-1).to(dtype=torch.long)
    label_counts_t = torch.bincount(labels_v, minlength=label_count)[:label_count]
    pred_counts_t = torch.bincount(pred_v, minlength=label_count)[:label_count]
    current_counts_t = torch.bincount(current_v, minlength=label_count)[:label_count]
    samples = int(labels_v.numel())
    positive = labels_v > 0
    positive_count = int(positive.sum().item())
    majority_acc = float(label_counts_t.max().item() / max(samples, 1))
    accuracy = float((pred_v == labels_v).to(dtype=torch.float32).mean().item())
    current_accuracy = float((current_v == labels_v).to(dtype=torch.float32).mean().item())
    if positive_count > 0:
        positive_accuracy = float((pred_v[positive] == labels_v[positive]).to(dtype=torch.float32).mean().item())
        current_positive_accuracy = float(
            (current_v[positive] == labels_v[positive]).to(dtype=torch.float32).mean().item()
        )
    else:
        positive_accuracy = 0.0
        current_positive_accuracy = 0.0
    return {
        "samples": samples,
        "positive_oracle_samples": positive_count,
        "accuracy_all": accuracy,
        "current_accuracy_all": current_accuracy,
        "majority_accuracy_all": majority_acc,
        "lift_vs_current": float(accuracy - current_accuracy),
        "lift_vs_majority": float(accuracy - majority_acc),
        "accuracy_on_positive_oracle": positive_accuracy,
        "current_accuracy_on_positive_oracle": current_positive_accuracy,
        "prediction_counts": {name: int(pred_counts_t[i].item()) for i, name in enumerate(label_names)},
        "current_prediction_counts": {name: int(current_counts_t[i].item()) for i, name in enumerate(label_names)},
        "label_counts": {name: int(label_counts_t[i].item()) for i, name in enumerate(label_names)},
        "label_rates": {
            name: float(label_counts_t[i].item() / max(samples, 1)) for i, name in enumerate(label_names)
        },
    }


class _PenaltyRouteLearnabilityHead(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        hidden_dim: int = 0,
        dropout: float = 0.0,
        head_mode: str = "classwise",
    ):
        super().__init__()
        self.head_mode = str(head_mode or "classwise").lower()
        if self.head_mode in {"flat", "multiclass"}:
            in_dim = int(feat_dim) * int(num_classes)
            if int(hidden_dim) > 0:
                self.net = nn.Sequential(
                    nn.Linear(in_dim, int(hidden_dim)),
                    nn.ReLU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(int(hidden_dim), int(num_classes)),
                )
            else:
                self.net = nn.Linear(in_dim, int(num_classes))
            return
        if self.head_mode not in {"classwise", "candidate", "shared"}:
            raise ValueError("route learnability head_mode must be classwise or flat.")
        if int(hidden_dim) > 0:
            self.net = nn.Sequential(
                nn.Linear(int(feat_dim), int(hidden_dim)),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden_dim), 1),
            )
        else:
            self.net = nn.Linear(int(feat_dim), 1)

    def forward(self, features_nqf: torch.Tensor) -> torch.Tensor:
        if features_nqf.dim() != 3:
            raise ValueError("features_nqf must have shape [N,num_classes,F].")
        if self.head_mode in {"flat", "multiclass"}:
            return self.net(features_nqf.flatten(start_dim=1))
        return self.net(features_nqf).squeeze(-1)


def _fit_penalty_route_learnability_head_from_tensors(
    *,
    train_tensors: Dict[str, torch.Tensor],
    eval_tensors_by_split: Dict[str, Dict[str, torch.Tensor]],
    label_names: List[str],
    feature_names: List[str],
    cfg: Dict[str, object],
    device: torch.device,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    train_features = train_tensors["features"].detach().to(dtype=torch.float32)
    train_labels = train_tensors["labels"].detach().to(dtype=torch.long).view(-1)
    train_current = train_tensors["current_pred"].detach().to(dtype=torch.long).view(-1)
    if train_features.dim() != 3:
        raise ValueError("train features must have shape [N,num_classes,F].")
    if int(train_features.shape[0]) != int(train_labels.numel()):
        raise ValueError("train labels length must match train features N.")
    label_count = int(train_features.shape[1])
    if len(label_names) != label_count:
        raise ValueError("label_names length must match train feature class count.")
    valid = (train_labels >= 0) & (train_labels < label_count)
    if int(valid.sum().item()) <= 0:
        raise ValueError("penalty route learnability probe has no valid train labels.")
    train_features = train_features[valid]
    train_labels = train_labels[valid]
    train_current = train_current[valid]
    flat = train_features.reshape(-1, int(train_features.shape[-1]))
    feat_mean = flat.mean(dim=0)
    feat_std = flat.std(dim=0).clamp_min(1.0e-6)

    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    model = _PenaltyRouteLearnabilityHead(
        feat_dim=int(train_features.shape[-1]),
        num_classes=label_count,
        hidden_dim=int(cfg.get("hidden_dim", 32)),
        dropout=float(cfg.get("dropout", 0.0)),
        head_mode=str(cfg.get("head_mode", "classwise")),
    ).to(device)
    init_bias_mode = str(cfg.get("init_bias", "none")).lower()
    if init_bias_mode in {"train_prior", "label_prior", "prior"}:
        counts = torch.bincount(train_labels, minlength=label_count).to(dtype=torch.float32)
        counts = counts + float(cfg.get("prior_smoothing", 1.0e-6))
        prior_logits = torch.log(counts / counts.sum().clamp_min(1.0e-12)).to(device=device)
        with torch.no_grad():
            final_layer = model.net if isinstance(model.net, nn.Linear) else model.net[-1]
            if isinstance(final_layer, nn.Linear) and int(final_layer.out_features) == label_count:
                if bool(cfg.get("zero_init_output", True)):
                    final_layer.weight.zero_()
                final_layer.bias.copy_(prior_logits)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 1.0e-3)),
        weight_decay=float(cfg.get("weight_decay", 1.0e-4)),
    )
    batch_size = max(1, int(cfg.get("batch_size", 256)))
    epochs = max(1, int(cfg.get("epochs", 60)))
    class_weight_mode = str(cfg.get("class_weight", "none")).lower()
    class_weight = None
    class_weight_summary = None
    if class_weight_mode in {"balanced", "auto"}:
        counts = torch.bincount(train_labels, minlength=label_count).to(dtype=torch.float32).clamp_min(1.0)
        weight = float(train_labels.numel()) / (float(label_count) * counts)
        if "class_weight_min" in cfg or "class_weight_max" in cfg:
            weight = weight.clamp(
                min=float(cfg.get("class_weight_min", 0.0)),
                max=float(cfg.get("class_weight_max", float("inf"))),
            )
        class_weight = weight.to(device=device)
        class_weight_summary = [float(v) for v in weight.detach().cpu().tolist()]
    elif isinstance(cfg.get("class_weight"), (list, tuple)):
        weight = torch.as_tensor(cfg.get("class_weight"), dtype=torch.float32)
        if int(weight.numel()) != label_count:
            raise ValueError(f"route learnability class_weight must have {label_count} entries.")
        class_weight = weight.to(device=device)
        class_weight_summary = [float(v) for v in weight.detach().cpu().tolist()]

    def _standardize(features: torch.Tensor) -> torch.Tensor:
        return (features.to(device=device, dtype=torch.float32) - feat_mean.to(device=device).view(1, 1, -1)) / feat_std.to(
            device=device
        ).view(1, 1, -1)

    train_features_device = train_features.to(device=device)
    train_labels_device = train_labels.to(device=device)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_loss = float("inf")
    best_train_loss = float("inf")
    best_epoch = 0
    selection_split_requested = str(cfg.get("early_stop_split", cfg.get("selection_split", "train"))).lower()
    if selection_split_requested in {"", "none", "off", "false", "0"}:
        selection_split_requested = "train"
    selection_split = selection_split_requested
    selection_fallback_reason = None
    if selection_split != "train" and selection_split not in eval_tensors_by_split:
        selection_fallback_reason = f"selection split {selection_split!r} not available; fell back to train loss"
        selection_split = "train"
    selection_metric = str(cfg.get("selection_metric", cfg.get("early_stop_metric", "loss"))).lower()
    metric_alias = {
        "ce": "loss",
        "cross_entropy": "loss",
        "val_loss": "loss",
        "acc": "accuracy",
        "accuracy_all": "accuracy",
        "val_accuracy": "accuracy",
        "lift": "lift_vs_majority",
        "majority_lift": "lift_vs_majority",
    }
    selection_metric = metric_alias.get(selection_metric, selection_metric)
    if selection_metric not in {"loss", "accuracy", "lift_vs_majority"}:
        raise ValueError("route learnability selection_metric must be loss, accuracy, or lift_vs_majority.")
    minimize_selection = selection_metric == "loss"
    best_selection_value = float("inf") if minimize_selection else -float("inf")
    best_selection_loss = float("inf")
    best_selection_metrics: Dict[str, object] = {}
    early_stop_patience = int(cfg.get("early_stop_patience", cfg.get("patience", 0)))
    early_stop_min_delta = float(cfg.get("early_stop_min_delta", cfg.get("min_delta", 0.0)))
    include_initial_eval = bool(cfg.get("include_initial_eval", False))
    epochs_without_improvement = 0
    stopped_epoch = 0
    selection_history: List[Dict[str, object]] = []

    def _filtered_eval_tensors(tensors: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = tensors["features"].detach().to(dtype=torch.float32)
        labels = tensors["labels"].detach().to(dtype=torch.long).view(-1)
        current = tensors["current_pred"].detach().to(dtype=torch.long).view(-1)
        valid_eval = (labels >= 0) & (labels < label_count)
        return features[valid_eval], labels[valid_eval], current[valid_eval]

    def _loss_and_metrics_for_features(
        features: torch.Tensor,
        labels: torch.Tensor,
        current: torch.Tensor,
    ) -> Tuple[float, Dict[str, object]]:
        if int(labels.numel()) <= 0:
            metrics = _penalty_route_learnability_metrics_from_scores(
                scores=torch.zeros(0, label_count),
                labels=labels,
                current_pred=current,
                label_names=label_names,
            )
            return float("inf"), metrics
        with torch.no_grad():
            scores_device = model(_standardize(features))
            loss = nn.functional.cross_entropy(scores_device, labels.to(device=device), weight=None)
            scores = scores_device.detach().cpu()
        metrics = _penalty_route_learnability_metrics_from_scores(
            scores=scores,
            labels=labels,
            current_pred=current,
            label_names=label_names,
        )
        return float(loss.detach().item()), metrics

    train_selection_tensors = {
        "features": train_features.detach().cpu(),
        "labels": train_labels.detach().cpu(),
        "current_pred": train_current.detach().cpu(),
    }

    def _selection_value_for_epoch(epoch_train_loss: float) -> Tuple[float, float, Dict[str, object]]:
        if selection_split == "train":
            if selection_metric == "loss":
                return float(epoch_train_loss), float(epoch_train_loss), {}
            features, labels, current = _filtered_eval_tensors(train_selection_tensors)
        else:
            features, labels, current = _filtered_eval_tensors(eval_tensors_by_split[selection_split])
        eval_loss, metrics = _loss_and_metrics_for_features(features, labels, current)
        if selection_metric == "loss":
            value = eval_loss
        elif selection_metric == "accuracy":
            value = float(metrics.get("accuracy_all", 0.0))
        else:
            value = float(metrics.get("lift_vs_majority", 0.0))
        return float(value), float(eval_loss), metrics

    def _selection_improved(value: float) -> bool:
        if minimize_selection:
            return value < (best_selection_value - early_stop_min_delta)
        return value > (best_selection_value + early_stop_min_delta)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if include_initial_eval:
        model.eval()
        selection_value, selection_loss, selection_metrics = _selection_value_for_epoch(float("inf"))
        selection_history.append(
            {
                "epoch": 0,
                "train_loss": None,
                "selection_split": selection_split,
                "selection_metric": selection_metric,
                "selection_value": float(selection_value),
                "selection_loss": float(selection_loss),
            }
        )
        if _selection_improved(selection_value):
            best_epoch = 0
            best_selection_value = float(selection_value)
            best_selection_loss = float(selection_loss)
            best_selection_metrics = dict(selection_metrics)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for epoch in range(1, epochs + 1):
        order = torch.randperm(int(train_features.shape[0]), generator=generator)
        model.train()
        total_loss = 0.0
        total_seen = 0
        for start in range(0, int(order.numel()), batch_size):
            idx = order[start : start + batch_size].to(device=device)
            features = _standardize(train_features_device.index_select(0, idx))
            target = train_labels_device.index_select(0, idx)
            logits = model(features)
            loss = nn.functional.cross_entropy(logits, target, weight=class_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_n = int(target.numel())
            total_loss += float(loss.detach().item()) * batch_n
            total_seen += batch_n
        epoch_loss = total_loss / max(total_seen, 1)
        best_train_loss = min(best_train_loss, float(epoch_loss))
        model.eval()
        selection_value, selection_loss, selection_metrics = _selection_value_for_epoch(epoch_loss)
        selection_history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(epoch_loss),
                "selection_split": selection_split,
                "selection_metric": selection_metric,
                "selection_value": float(selection_value),
                "selection_loss": float(selection_loss),
            }
        )
        if _selection_improved(selection_value):
            best_loss = float(epoch_loss)
            best_epoch = int(epoch)
            best_selection_value = float(selection_value)
            best_selection_loss = float(selection_loss)
            best_selection_metrics = dict(selection_metrics)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
                stopped_epoch = int(epoch)
                break
    if best_epoch <= 0:
        best_epoch = 1
    model.load_state_dict(best_state, strict=True)
    model.eval()

    def _metrics_for(tensors: Dict[str, torch.Tensor]) -> Dict[str, object]:
        features, labels, current = _filtered_eval_tensors(tensors)
        if int(labels.numel()) <= 0:
            return _penalty_route_learnability_metrics_from_scores(
                scores=torch.zeros(0, label_count),
                labels=labels,
                current_pred=current,
                label_names=label_names,
            )
        with torch.no_grad():
            scores = model(_standardize(features)).detach().cpu()
        return _penalty_route_learnability_metrics_from_scores(
            scores=scores,
            labels=labels,
            current_pred=current,
            label_names=label_names,
        )

    split_metrics = {
        "train": _metrics_for(
            {
                "features": train_features.detach().cpu(),
                "labels": train_labels.detach().cpu(),
                "current_pred": train_current.detach().cpu(),
            }
        )
    }
    for split_name, tensors in eval_tensors_by_split.items():
        split_metrics[str(split_name)] = _metrics_for(tensors)

    summary = {
        "enable": True,
        "samples_train": int(train_labels.numel()),
        "label_names": list(label_names),
        "feature_names": list(feature_names),
        "config": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": float(cfg.get("lr", 1.0e-3)),
            "head_mode": str(cfg.get("head_mode", "classwise")).lower(),
            "hidden_dim": int(cfg.get("hidden_dim", 32)),
            "dropout": float(cfg.get("dropout", 0.0)),
            "weight_decay": float(cfg.get("weight_decay", 1.0e-4)),
            "class_weight": class_weight_mode,
            "class_weight_values": class_weight_summary,
            "class_weight_min": None if "class_weight_min" not in cfg else float(cfg.get("class_weight_min", 0.0)),
            "class_weight_max": None if "class_weight_max" not in cfg else float(cfg.get("class_weight_max", 0.0)),
            "selection_split": selection_split_requested,
            "selection_metric": selection_metric,
            "early_stop_patience": int(early_stop_patience),
            "early_stop_min_delta": float(early_stop_min_delta),
            "init_bias": init_bias_mode,
            "include_initial_eval": bool(include_initial_eval),
            "seed": int(seed),
        },
        "best_train_loss": float(best_train_loss),
        "selection": {
            "split": selection_split,
            "requested_split": selection_split_requested,
            "fallback_reason": selection_fallback_reason,
            "metric": selection_metric,
            "best_epoch": int(best_epoch),
            "best_value": float(best_selection_value),
            "best_loss": float(best_selection_loss),
            "best_train_epoch_loss": float(best_loss),
            "minimize": bool(minimize_selection),
            "patience": int(early_stop_patience),
            "min_delta": float(early_stop_min_delta),
            "stopped_epoch": int(stopped_epoch),
            "best_metrics": best_selection_metrics,
        },
        "selection_history": selection_history,
        "splits": split_metrics,
    }
    artifact = {
        "state_dict": best_state,
        "feature_mean": feat_mean.detach().cpu(),
        "feature_std": feat_std.detach().cpu(),
        "class_weight": None if class_weight is None else class_weight.detach().cpu(),
        "label_names": list(label_names),
        "feature_names": list(feature_names),
        "config": dict(summary["config"]),
        "selection": dict(summary["selection"]),
    }
    return summary, artifact


def _specialization_summary_from_accumulators(
    *,
    expert_names: List[str],
    metric_names: List[str],
    metric_expert_indices: List[int],
    base_error_sum_j: torch.Tensor,
    candidate_error_sum_pj: torch.Tensor,
    positive_count_pj: torch.Tensor,
    positive_base_error_sum_pj: Optional[torch.Tensor] = None,
    positive_candidate_error_sum_pj: Optional[torch.Tensor] = None,
    sample_channels: int,
    base_definition: str,
    candidate_definition: str,
    aggregation_unit: str,
) -> Optional[Dict[str, object]]:
    if not metric_names or int(sample_channels) <= 0:
        return None
    base_mean_j = base_error_sum_j / float(sample_channels)
    candidate_mean_pj = candidate_error_sum_pj / float(sample_channels)
    gain_pj = base_mean_j.unsqueeze(0) - candidate_mean_pj
    gain_pct_pj = (
        100.0
        * gain_pj
        / base_mean_j.abs().clamp_min(1.0e-12).unsqueeze(0)
    )
    positive_rate_pj = positive_count_pj / float(sample_channels)
    diagonal_rows = []
    for metric_col, expert_idx in enumerate(metric_expert_indices):
        matching_gain = float(gain_pj[expert_idx, metric_col].item())
        better_expert_count = int(
            (gain_pj[:, metric_col] > gain_pj[expert_idx, metric_col]).sum().item()
        )
        diagonal_rows.append(
            {
                "expert": str(expert_names[expert_idx]),
                "metric": str(metric_names[metric_col]),
                "base_error": float(base_mean_j[metric_col].item()),
                "candidate_error": float(candidate_mean_pj[expert_idx, metric_col].item()),
                "mean_signed_improvement": matching_gain,
                "relative_improvement_pct": float(
                    gain_pct_pj[expert_idx, metric_col].item()
                ),
                "positive_sample_channel_rate": float(
                    positive_rate_pj[expert_idx, metric_col].item()
                ),
                "rank_by_mean_improvement": int(better_expert_count + 1),
                "improved": bool(matching_gain > 0.0),
            }
        )
        if (
            positive_base_error_sum_pj is not None
            and positive_candidate_error_sum_pj is not None
        ):
            positive_base = float(
                positive_base_error_sum_pj[expert_idx, metric_col].item()
            )
            positive_candidate = float(
                positive_candidate_error_sum_pj[expert_idx, metric_col].item()
            )
            positive_gain = positive_base - positive_candidate
            base_total = float(base_error_sum_j[metric_col].item())
            diagonal_rows[-1].update(
                {
                    "positive_subset_base_error_sum": positive_base,
                    "positive_subset_candidate_error_sum": positive_candidate,
                    "positive_subset_relative_improvement_pct": float(
                        100.0 * positive_gain / max(abs(positive_base), 1.0e-12)
                    ),
                    "positive_oracle_relative_improvement_pct_overall": float(
                        100.0 * positive_gain / max(abs(base_total), 1.0e-12)
                    ),
                }
            )
    return {
        "base_definition": str(base_definition),
        "candidate_definition": str(candidate_definition),
        "aggregation_unit": str(aggregation_unit),
        "sample_channels": int(sample_channels),
        "expert_names": [str(name) for name in expert_names],
        "metric_names": [str(name) for name in metric_names],
        "base_error_mean_by_metric": base_mean_j.tolist(),
        "candidate_error_mean_expert_by_metric": candidate_mean_pj.tolist(),
        "mean_signed_improvement_expert_by_metric": gain_pj.tolist(),
        "relative_improvement_pct_expert_by_metric": gain_pct_pj.tolist(),
        "positive_sample_channel_rate_expert_by_metric": positive_rate_pj.tolist(),
        "diagonal": diagonal_rows,
        "all_diagonal_improved": bool(
            diagonal_rows and all(bool(row["improved"]) for row in diagonal_rows)
        ),
    }


@torch.no_grad()
def evaluate_penalty_explainability(
    model: nn.Module,
    gate: ClusterwiseMoEGate,
    pred_residual: Optional[ClusterwisePredResidualMoE],
    loader: DataLoader,
    cluster_id_c: torch.Tensor,
    K: int,
    moe_cfg: dict,
    device: torch.device,
    penalty_names: List[str],
    penalty_fns: Dict[str, callable],
    penalty_scale: Optional[torch.Tensor],
    select_ranks: Optional[List[int]],
    gate_soft_weight: float,
    split_name: str,
    penalty_portrait_kp: Optional[torch.Tensor] = None,
    prior_prob_kp: Optional[torch.Tensor] = None,
    allowed_mask_kp: Optional[torch.Tensor] = None,
    max_batches: int = 0,
    history_anchor_cfg: Optional[dict] = None,
    observed_history_tc: Optional[torch.Tensor] = None,
    input_len: int = 0,
    eval_start: int = 0,
    model_train_stat_adapter_pc: Optional[torch.Tensor] = None,
    model_train_stat_adapter_cfg: Optional[dict] = None,
    train_stat_anchor_pc: Optional[torch.Tensor] = None,
    train_residual_anchor_phc: Optional[torch.Tensor] = None,
    learnable_output_anchor: Optional[ClusterwiseLearnableOutputAnchor] = None,
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
    explain_cfg = moe_cfg.get("explainability", {}) or {}
    utility_thresholds = [float(v) for v in (explain_cfg.get("utility_thresholds", []) or [])]
    route_label_phase_periods = [
        int(v) for v in (explain_cfg.get("route_label_phase_periods", []) or []) if int(v) > 0
    ]
    route_label_phase_bins = int(explain_cfg.get("route_label_phase_bins", 8))
    top1_confidence_bins = [float(v) for v in (explain_cfg.get("top1_confidence_bins", []) or [])]
    specialization_cfg = explain_cfg.get("adapter_specialization", {}) or {}
    if not isinstance(specialization_cfg, dict):
        specialization_cfg = {"enable": bool(specialization_cfg)}
    specialization_enable = bool(specialization_cfg.get("enable", False))
    specialization_scale_sweep = sorted(
        {
            float(value)
            for value in (
                specialization_cfg.get("scale_sweep", []) or []
            )
        }
    )
    periodic_action_space_cfg = explain_cfg.get(
        "periodic_action_space",
        {},
    ) or {}
    if not isinstance(periodic_action_space_cfg, dict):
        periodic_action_space_cfg = {
            "enable": bool(periodic_action_space_cfg)
        }
    periodic_action_space_enable = bool(
        periodic_action_space_cfg.get("enable", False)
    )

    P = len(penalty_names)
    cid_c = cluster_id_c.detach().to(device=device, dtype=torch.long)
    cid_cpu = cluster_id_c.detach().cpu().to(dtype=torch.long)
    cluster_channel_count = torch.bincount(cid_cpu, minlength=K).clamp_min(1).to(dtype=torch.float32)
    semantic_metric_indices = [
        p for p, name in enumerate(penalty_names)
        if str(name) in _DIRECT_ATTRIBUTE_PENALTY_NAMES
    ] if specialization_enable else []
    semantic_metric_names = [str(penalty_names[p]) for p in semantic_metric_indices]
    semantic_base_error_sum_j = torch.zeros(
        len(semantic_metric_indices), dtype=torch.float64
    )
    semantic_candidate_error_sum_pj = torch.zeros(
        P, len(semantic_metric_indices), dtype=torch.float64
    )
    semantic_positive_count_pj = torch.zeros(
        P, len(semantic_metric_indices), dtype=torch.float64
    )
    semantic_sample_channels = 0
    penalty_metric_indices = [
        p for p, name in enumerate(penalty_names) if str(name) in penalty_fns
    ] if specialization_enable else []
    penalty_metric_names = [str(penalty_names[p]) for p in penalty_metric_indices]
    penalty_base_error_sum_j = torch.zeros(
        len(penalty_metric_indices), dtype=torch.float64
    )
    penalty_candidate_error_sum_pj = torch.zeros(
        P, len(penalty_metric_indices), dtype=torch.float64
    )
    penalty_positive_count_pj = torch.zeros(
        P, len(penalty_metric_indices), dtype=torch.float64
    )
    penalty_positive_base_error_sum_pj = torch.zeros_like(
        penalty_positive_count_pj
    )
    penalty_positive_candidate_error_sum_pj = torch.zeros_like(
        penalty_positive_count_pj
    )
    penalty_sample_channels = 0
    scale_sweep_error_sum_ps = torch.zeros(
        P,
        len(specialization_scale_sweep),
        dtype=torch.float64,
    )
    scale_sweep_sample_channels = 0
    gated_scale_selected_count_p = torch.zeros(P, dtype=torch.float64)
    gated_scale_named_base_sum_p = torch.zeros(P, dtype=torch.float64)
    gated_scale_mse_base_sum_p = torch.zeros(P, dtype=torch.float64)
    gated_scale_mae_base_sum_p = torch.zeros(P, dtype=torch.float64)
    gated_scale_named_error_sum_ps = torch.zeros(
        P,
        len(specialization_scale_sweep),
        dtype=torch.float64,
    )
    gated_scale_mse_error_sum_ps = torch.zeros_like(
        gated_scale_named_error_sum_ps
    )
    gated_scale_mae_error_sum_ps = torch.zeros_like(
        gated_scale_named_error_sum_ps
    )
    projection_patch_len = 0
    if getattr(pred_residual, "patch_router", None) is not None:
        projection_patch_len = int(pred_residual.patch_router.patch_len)
    patch_penalty_base_error_sum_j = torch.zeros(
        len(penalty_metric_indices), dtype=torch.float64
    )
    patch_penalty_candidate_error_sum_pj = torch.zeros(
        P, len(penalty_metric_indices), dtype=torch.float64
    )
    patch_penalty_positive_count_pj = torch.zeros(
        P, len(penalty_metric_indices), dtype=torch.float64
    )
    patch_penalty_positive_base_error_sum_pj = torch.zeros_like(
        patch_penalty_positive_count_pj
    )
    patch_penalty_positive_candidate_error_sum_pj = torch.zeros_like(
        patch_penalty_positive_count_pj
    )
    patch_penalty_sample_channels = 0
    allowed_mask_device = None
    if allowed_mask_kp is not None:
        allowed_mask_device = allowed_mask_kp.detach().to(device=device, dtype=torch.bool)
        if tuple(allowed_mask_device.shape) != (int(K), int(P)):
            raise ValueError("allowed_mask_kp must have shape [K,P] for explainability diagnostics.")

    total_bc_k = torch.zeros(K, dtype=torch.float64)
    base_err_sum_k = torch.zeros(K, dtype=torch.float64)
    final_err_sum_k = torch.zeros(K, dtype=torch.float64)
    route_scale_cross_sum = 0.0
    route_scale_delta_sq_sum = 0.0
    route_scale_element_count = 0
    route_component_gram_pp = torch.zeros(P, P, dtype=torch.float64)
    route_component_cross_p = torch.zeros(P, dtype=torch.float64)
    route_component_element_count = 0
    route_component_reconstruction_max_abs = 0.0
    route_channel_patch_cross_cs = None
    route_channel_patch_delta_sq_cs = None
    route_channel_patch_element_count = 0
    previous_cycle_prediction_by_start: Dict[int, torch.Tensor] = {}
    previous_cycle_base_sse_c = torch.zeros(int(cid_cpu.numel()), dtype=torch.float64)
    previous_cycle_cross_c = torch.zeros_like(previous_cycle_base_sse_c)
    previous_cycle_delta_sq_c = torch.zeros_like(previous_cycle_base_sse_c)
    previous_cycle_element_count_c = torch.zeros_like(previous_cycle_base_sse_c)
    rolling_position_lookbacks = (96, 672, 1344, 2688)
    rolling_position_error_history: List[torch.Tensor] = []
    rolling_position_sum_by_lookback = {
        lookback: torch.zeros(
            int(cid_cpu.numel()), int(input_len), dtype=torch.float64
        )
        for lookback in rolling_position_lookbacks
    }
    rolling_position_stats = {
        lookback: {
            "base_sse_c": torch.zeros(int(cid_cpu.numel()), dtype=torch.float64),
            "cross_c": torch.zeros(int(cid_cpu.numel()), dtype=torch.float64),
            "delta_sq_c": torch.zeros(int(cid_cpu.numel()), dtype=torch.float64),
            "count_c": torch.zeros(int(cid_cpu.numel()), dtype=torch.float64),
        }
        for lookback in rolling_position_lookbacks
    }
    position_residual_sum_ch = None
    position_residual_sample_count = 0
    position_input_ridge_xtx_cdd = None
    position_input_ridge_xtr_cdh = None
    position_input_ridge_sample_count = 0
    try:
        position_input_ridge_total_windows = int(len(loader.dataset))
    except Exception:
        position_input_ridge_total_windows = 0
    position_input_ridge_temporal_blocks = 4
    position_input_ridge_block_stats = None
    oracle_err_sum_k = torch.zeros(K, dtype=torch.float64)
    cluster_penalty_oracle_err_sum_k = torch.zeros(K, dtype=torch.float64)
    cluster_route_oracle_err_sum_k = torch.zeros(K, dtype=torch.float64)
    cluster_route_oracle_skip_count_k = torch.zeros(K, dtype=torch.float64)
    cluster_route_oracle_decision_count_k = torch.zeros(K, dtype=torch.float64)
    fusion_sum_k = torch.zeros(K, dtype=torch.float64)
    skip_count_k = torch.zeros(K, dtype=torch.float64)
    skip_on_oracle_positive_count_k = torch.zeros(K, dtype=torch.float64)
    selected_count_kp = torch.zeros(K, P, dtype=torch.float64)
    top1_intended_count_kp = torch.zeros(K, P, dtype=torch.float64)
    top1_selected_count_kp = torch.zeros(K, P, dtype=torch.float64)
    top1_selected_positive_count_kp = torch.zeros(K, P, dtype=torch.float64)
    top1_selected_gain_sum_kp = torch.zeros(K, P, dtype=torch.float64)
    skipped_top1_count_kp = torch.zeros(K, P, dtype=torch.float64)
    skipped_on_oracle_positive_count_kp = torch.zeros(K, P, dtype=torch.float64)
    harmful_top1_not_skipped_count_kp = torch.zeros(K, P, dtype=torch.float64)
    oracle_count_kp = torch.zeros(K, P, dtype=torch.float64)
    positive_oracle_count_kp = torch.zeros(K, P, dtype=torch.float64)
    selected_positive_count_kp = torch.zeros(K, P, dtype=torch.float64)
    gate_prob_sum_kp = torch.zeros(K, P, dtype=torch.float64)
    route_weight_sum_kp = torch.zeros(K, P, dtype=torch.float64)
    selector_sum_kp = torch.zeros(K, P, dtype=torch.float64)
    intervention_sum_kp = torch.zeros(K, P, dtype=torch.float64)
    single_gain_sum_kp = torch.zeros(K, P, dtype=torch.float64)
    selected_gain_sum_kp = torch.zeros(K, P, dtype=torch.float64)
    selected_gain_count_kp = torch.zeros(K, P, dtype=torch.float64)
    utility_valid_count_kt = torch.zeros(K, len(utility_thresholds), dtype=torch.float64)
    utility_total_count_k = torch.zeros(K, dtype=torch.float64)
    utility_best_gain_sum_k = torch.zeros(K, dtype=torch.float64)
    utility_best_gain_positive_count_k = torch.zeros(K, dtype=torch.float64)

    # The legacy explainability path below is sample/channel based and reads the
    # outer cluster gate.  A channel-patch router replaces that gate at
    # deployment, so keep a separate, explicitly patch-granular ledger rather
    # than mixing the two routing decisions.
    patch_router_seen = False
    patch_decision_count_k = torch.zeros(K, dtype=torch.float64)
    patch_base_mse_sum_k = torch.zeros(K, dtype=torch.float64)
    patch_final_mse_sum_k = torch.zeros(K, dtype=torch.float64)
    patch_skip_count_k = torch.zeros(K, dtype=torch.float64)
    patch_oracle_dual_skip_count_k = torch.zeros(K, dtype=torch.float64)
    patch_correct_skip_count_k = torch.zeros(K, dtype=torch.float64)
    patch_missed_dual_action_count_k = torch.zeros(K, dtype=torch.float64)
    patch_route_skip_mismatch_count_k = torch.zeros(K, dtype=torch.float64)
    patch_adopted_count_k = torch.zeros(K, dtype=torch.float64)
    patch_selected_dual_safe_count_k = torch.zeros(K, dtype=torch.float64)
    patch_false_adopt_count_k = torch.zeros(K, dtype=torch.float64)
    patch_selected_mse_gain_sum_k = torch.zeros(K, dtype=torch.float64)
    patch_selected_mae_gain_sum_k = torch.zeros(K, dtype=torch.float64)
    patch_false_adopt_mse_cost_sum_k = torch.zeros(K, dtype=torch.float64)
    patch_false_adopt_mae_cost_sum_k = torch.zeros(K, dtype=torch.float64)
    patch_selected_count_kp = torch.zeros(K, P, dtype=torch.float64)
    patch_selected_dual_safe_count_kp = torch.zeros(K, P, dtype=torch.float64)
    patch_selected_mse_gain_sum_kp = torch.zeros(K, P, dtype=torch.float64)
    patch_selected_mae_gain_sum_kp = torch.zeros(K, P, dtype=torch.float64)
    patch_named_penalty_available_count_kp = torch.zeros(
        K, P, dtype=torch.float64
    )
    patch_named_penalty_joint_safe_count_kp = torch.zeros(
        K, P, dtype=torch.float64
    )
    patch_selected_named_penalty_positive_count_kp = torch.zeros(
        K, P, dtype=torch.float64
    )
    patch_selected_named_penalty_joint_safe_count_kp = torch.zeros(
        K, P, dtype=torch.float64
    )
    patch_selected_named_penalty_base_sum_kp = torch.zeros(
        K, P, dtype=torch.float64
    )
    patch_selected_named_penalty_candidate_sum_kp = torch.zeros(
        K, P, dtype=torch.float64
    )
    periodic_action_names = [
        "backbone",
        "periodic",
        *[f"periodic+{name}" for name in penalty_names],
    ]
    periodic_action_decision_count = 0
    periodic_action_backbone_mse_sum = 0.0
    periodic_action_backbone_mae_sum = 0.0
    periodic_action_periodic_mse_sum = 0.0
    periodic_action_periodic_mae_sum = 0.0
    periodic_action_current_mse_sum = 0.0
    periodic_action_current_mae_sum = 0.0
    periodic_action_mandatory_oracle_mse_sum = 0.0
    periodic_action_mandatory_oracle_mae_sum = 0.0
    periodic_action_mandatory_dual_oracle_mse_sum = 0.0
    periodic_action_mandatory_dual_oracle_mae_sum = 0.0
    periodic_action_selectable_oracle_mse_sum = 0.0
    periodic_action_selectable_oracle_mae_sum = 0.0
    periodic_action_selectable_dual_oracle_mse_sum = 0.0
    periodic_action_selectable_dual_oracle_mae_sum = 0.0
    periodic_action_mandatory_oracle_count_a = torch.zeros(
        P + 1, dtype=torch.float64
    )
    periodic_action_mandatory_dual_oracle_count_a = torch.zeros(
        P + 1, dtype=torch.float64
    )
    periodic_action_selectable_oracle_count_a = torch.zeros(
        P + 2, dtype=torch.float64
    )
    periodic_action_selectable_dual_oracle_count_a = torch.zeros(
        P + 2, dtype=torch.float64
    )

    total_decisions = 0
    total_selected = 0
    total_oracle_positive = 0
    total_selected_positive = 0
    batch_count = 0
    route_label_feature_chunks: List[torch.Tensor] = []
    route_label_chunks: List[torch.Tensor] = []
    route_label_query_start_chunks: List[torch.Tensor] = []
    top1_conf_chunks: List[torch.Tensor] = []
    top1_gain_chunks: List[torch.Tensor] = []
    top1_penalty_chunks: List[torch.Tensor] = []
    top1_active_chunks: List[torch.Tensor] = []
    top1_skip_chunks: List[torch.Tensor] = []

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
        fixed_expert_delta_bch = build_moe_output_anchor_fixed_expert_delta(
            y_base,
            x_bcl=x,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len),
            moe_cfg=moe_cfg,
            moe_enable=moe_enable,
            observed_history_tc=observed_history_tc,
            train_stat_anchor_pc=train_stat_anchor_pc,
            train_residual_anchor_phc=train_residual_anchor_phc,
            learnable_output_anchor=learnable_output_anchor,
            cluster_id_c=cluster_id_c,
        )
        candidate_base_bch = _fixed_expert_candidate_base(
            y_base,
            pred_residual,
            fixed_expert_delta_bch,
        )
        feat_bkf = _build_gate_routing_features(
            x,
            candidate_base_bch,
            cluster_id_c,
            K,
            mode=gate_feature_mode,
        )
        route_pen_bkp = _router_penalty_context_from_history(
            x_bcl=x,
            yhat_base_bch=candidate_base_bch,
            penalty_names=penalty_names,
            penalty_fns=penalty_fns,
            penalty_scale=penalty_scale,
            cluster_id_c=cluster_id_c,
            K=K,
            router_mode=router_mode,
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
            query_start_abs_b=query_start_abs_b,
            fixed_expert_delta_bch=fixed_expert_delta_bch,
        )
        residuals = pred_out.get("residuals")
        alpha_cp = pred_out.get("alpha_cp")
        if residuals is None or alpha_cp is None or residuals.numel() == 0:
            continue

        route_bcp = pred_out.get("route_bcp", mask_bkp[:, cid_c, :])
        effective_route_bcp = pred_out.get("effective_route_bcp", route_bcp)
        selector_bcp = pred_out.get("selector_bcp", torch.ones_like(route_bcp))
        intervention_bcp = pred_out.get("intervention_bcp", torch.ones_like(route_bcp))
        fusion_bc = pred_out.get("fusion_bc", torch.ones_like(route_bcp[..., 0]))
        y_final_raw = pred_out["y_final"]
        y_final = apply_moe_output_anchor_experts(
            y_final_raw,
            base_pred_bch=y_base,
            x_bcl=x,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len),
            moe_cfg=moe_cfg,
            moe_enable=moe_enable,
            observed_history_tc=observed_history_tc,
            train_stat_anchor_pc=train_stat_anchor_pc,
            train_residual_anchor_phc=train_residual_anchor_phc,
            learnable_output_anchor=learnable_output_anchor,
            cluster_id_c=cluster_id_c,
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
            learnable_output_anchor=learnable_output_anchor,
            cluster_id_c=cluster_id_c,
            # Explainability compares independently reachable candidates.  The
            # deployed patch route belongs only to y_final; applying it here
            # would turn every unselected candidate patch into a silent no-op.
            include_patch_route=False,
        )
        if (
            int(input_len) == int(y_final.shape[-1])
            and int(x.shape[-1]) >= int(y_final.shape[-1])
        ):
            cycle_lag = int(input_len)
            horizon = int(y_final.shape[-1])
            for sample_i in range(int(y_final.shape[0])):
                query_start_i = int(query_start_abs_b[sample_i].item())
                prior_prediction = previous_cycle_prediction_by_start.get(
                    query_start_i - cycle_lag
                )
                if prior_prediction is not None:
                    observed_previous_target = (
                        x[sample_i, :, -horizon:].detach().cpu().double()
                    )
                    correction_ch = observed_previous_target - prior_prediction
                    deployed_error_ch = (
                        y_final[sample_i].detach().cpu().double()
                        - y[sample_i].detach().cpu().double()
                    )
                    previous_cycle_base_sse_c += deployed_error_ch.square().sum(dim=-1)
                    previous_cycle_cross_c += (
                        deployed_error_ch * correction_ch
                    ).sum(dim=-1)
                    previous_cycle_delta_sq_c += correction_ch.square().sum(dim=-1)
                    previous_cycle_element_count_c += float(horizon)
                    rolling_position_error_history.append(correction_ch)
                    for lookback in rolling_position_lookbacks:
                        rolling_sum = rolling_position_sum_by_lookback[lookback]
                        rolling_sum += correction_ch
                        if len(rolling_position_error_history) > lookback:
                            rolling_sum -= rolling_position_error_history[
                                -lookback - 1
                            ]
                        if len(rolling_position_error_history) >= lookback:
                            rolling_correction = rolling_sum / float(lookback)
                            stats = rolling_position_stats[lookback]
                            stats["base_sse_c"] += deployed_error_ch.square().sum(
                                dim=-1
                            )
                            stats["cross_c"] += (
                                deployed_error_ch * rolling_correction
                            ).sum(dim=-1)
                            stats["delta_sq_c"] += rolling_correction.square().sum(
                                dim=-1
                            )
                            stats["count_c"] += float(horizon)
                previous_cycle_prediction_by_start[query_start_i] = (
                    y_final[sample_i].detach().cpu().double()
                )
        position_residual_batch = (y - y_final).detach().cpu().double().sum(dim=0)
        if position_residual_sum_ch is None:
            position_residual_sum_ch = torch.zeros_like(position_residual_batch)
        position_residual_sum_ch += position_residual_batch
        position_residual_sample_count += int(y_final.shape[0])
        ridge_patch_len = int(projection_patch_len)
        if (
            ridge_patch_len > 0
            and int(input_len) == int(y_final.shape[-1])
            and int(y_final.shape[-1]) % ridge_patch_len == 0
        ):
            ridge_segments = int(y_final.shape[-1]) // ridge_patch_len
            eps = 1.0e-6
            input_last = x[..., -1:]
            input_std = x.std(dim=-1, unbiased=False, keepdim=True).clamp_min(eps)
            input_norm = (x[..., -int(input_len):] - input_last) / input_std
            forecast_norm = (y_final - input_last) / input_std
            input_patch = input_norm.reshape(
                int(x.shape[0]), int(x.shape[1]), ridge_segments, ridge_patch_len
            )
            forecast_patch = forecast_norm.reshape_as(input_patch)
            input_patch_mean = input_patch.mean(dim=-1)
            input_patch_delta = input_patch[..., -1] - input_patch[..., 0]
            forecast_patch_mean = forecast_patch.mean(dim=-1)
            origin_index_b = (
                query_start_abs_b.to(device=x.device, dtype=x.dtype)
                + float(input_len)
            )
            phase_features = []
            for phase_period, phase_harmonics in ((96, 4), (672, 2)):
                base_angle_b = (
                    2.0 * math.pi * origin_index_b / float(phase_period)
                )
                for harmonic in range(1, phase_harmonics + 1):
                    angle_b = float(harmonic) * base_angle_b
                    phase_features.extend(
                        [
                            torch.sin(angle_b).view(-1, 1, 1).expand(
                                -1, int(x.shape[1]), 1
                            ),
                            torch.cos(angle_b).view(-1, 1, 1).expand(
                                -1, int(x.shape[1]), 1
                            ),
                        ]
                    )
            ridge_features_bcd = torch.cat(
                [
                    torch.ones_like(input_patch_mean[..., :1]),
                    input_patch_mean,
                    input_patch_delta,
                    forecast_patch_mean,
                    input_std.log(),
                    *phase_features,
                ],
                dim=-1,
            ).detach().cpu().double()
            ridge_residual_bch = (y - y_final).detach().cpu().double()
            batch_xtx = torch.einsum(
                "bcd,bce->cde", ridge_features_bcd, ridge_features_bcd
            )
            batch_xtr = torch.einsum(
                "bcd,bch->cdh", ridge_features_bcd, ridge_residual_bch
            )
            if position_input_ridge_xtx_cdd is None:
                position_input_ridge_xtx_cdd = torch.zeros_like(batch_xtx)
                position_input_ridge_xtr_cdh = torch.zeros_like(batch_xtr)
                position_input_ridge_block_stats = [
                    {
                        "xtx_cdd": torch.zeros_like(batch_xtx),
                        "xtr_cdh": torch.zeros_like(batch_xtr),
                        "sample_count": 0,
                    }
                    for _ in range(position_input_ridge_temporal_blocks)
                ]
            position_input_ridge_xtx_cdd += batch_xtx
            position_input_ridge_xtr_cdh += batch_xtr
            if (
                position_input_ridge_block_stats is not None
                and position_input_ridge_total_windows > 0
            ):
                batch_start = int(position_input_ridge_sample_count)
                batch_indices = torch.arange(
                    batch_start,
                    batch_start + int(y_final.shape[0]),
                    dtype=torch.long,
                )
                block_indices = torch.clamp(
                    batch_indices
                    * int(position_input_ridge_temporal_blocks)
                    // int(position_input_ridge_total_windows),
                    max=int(position_input_ridge_temporal_blocks) - 1,
                )
                for block_i in range(position_input_ridge_temporal_blocks):
                    block_mask = block_indices == block_i
                    if not bool(block_mask.any().item()):
                        continue
                    block_features = ridge_features_bcd[block_mask]
                    block_residual = ridge_residual_bch[block_mask]
                    block_stats = position_input_ridge_block_stats[block_i]
                    block_stats["xtx_cdd"] += torch.einsum(
                        "bcd,bce->cde", block_features, block_features
                    )
                    block_stats["xtr_cdh"] += torch.einsum(
                        "bcd,bch->cdh", block_features, block_residual
                    )
                    block_stats["sample_count"] += int(block_mask.sum().item())
            position_input_ridge_sample_count += int(y_final.shape[0])
        if cand_bcpH is None:
            continue
        route_scale_base_error = (y_base_final - y).detach().double()
        route_scale_correction = (y_final - y_base_final).detach().double()
        route_scale_cross_sum += float(
            (route_scale_base_error * route_scale_correction).sum().item()
        )
        route_scale_delta_sq_sum += float(
            route_scale_correction.square().sum().item()
        )
        route_scale_element_count += int(route_scale_correction.numel())
        route_patch_len = int(projection_patch_len)
        if (
            route_patch_len > 0
            and int(route_scale_correction.shape[-1]) % route_patch_len == 0
        ):
            route_segments = int(route_scale_correction.shape[-1]) // route_patch_len
            base_error_bcsp = route_scale_base_error.reshape(
                int(route_scale_base_error.shape[0]),
                int(route_scale_base_error.shape[1]),
                route_segments,
                route_patch_len,
            )
            correction_bcsp = route_scale_correction.reshape_as(base_error_bcsp)
            batch_cross_cs = (base_error_bcsp * correction_bcsp).sum(dim=(0, 3)).cpu()
            batch_delta_sq_cs = correction_bcsp.square().sum(dim=(0, 3)).cpu()
            if route_channel_patch_cross_cs is None:
                route_channel_patch_cross_cs = torch.zeros_like(batch_cross_cs)
                route_channel_patch_delta_sq_cs = torch.zeros_like(batch_delta_sq_cs)
            route_channel_patch_cross_cs += batch_cross_cs
            route_channel_patch_delta_sq_cs += batch_delta_sq_cs
            route_channel_patch_element_count += int(route_scale_correction.numel())
        route_components = pred_out.get("branches")
        if (
            route_components is not None
            and route_components.ndim == 4
            and int(route_components.shape[2]) == P
        ):
            component = route_components.detach().double()
            route_component_gram_pp += torch.einsum(
                "bcph,bcqh->pq",
                component,
                component,
            ).cpu()
            route_component_cross_p += torch.einsum(
                "bch,bcph->p",
                route_scale_base_error,
                component,
            ).cpu()
            route_component_element_count += int(route_scale_base_error.numel())
            reconstruction_error = (
                component.sum(dim=2) - route_scale_correction
            ).abs().max()
            route_component_reconstruction_max_abs = max(
                route_component_reconstruction_max_abs,
                float(reconstruction_error.item()),
            )
        if semantic_metric_indices or penalty_metric_indices:
            independent_base_bch, independent_cand_bcpH = (
                _pred_residual_candidates_on_eval_path(
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
                    learnable_output_anchor=learnable_output_anchor,
                    cluster_id_c=cluster_id_c,
                    include_intervention=False,
                    include_selector=False,
                    include_patch_route=False,
                )
            )
            if independent_cand_bcpH is not None and semantic_metric_indices:
                base_attribute_error_bcj = torch.stack(
                    [
                        _named_forecast_attribute_error(
                            independent_base_bch,
                            y,
                            name,
                        )
                        for name in semantic_metric_names
                    ],
                    dim=-1,
                )
                candidate_attribute_error_bcpj = torch.stack(
                    [
                        torch.stack(
                            [
                                _named_forecast_attribute_error(
                                    independent_cand_bcpH[:, :, p, :],
                                    y,
                                    name,
                                )
                                for name in semantic_metric_names
                            ],
                            dim=-1,
                        )
                        for p in range(P)
                    ],
                    dim=2,
                )
                semantic_base_error_sum_j += (
                    base_attribute_error_bcj.sum(dim=(0, 1)).detach().cpu().double()
                )
                semantic_candidate_error_sum_pj += (
                    candidate_attribute_error_bcpj.sum(dim=(0, 1)).detach().cpu().double()
                )
                semantic_positive_count_pj += (
                    (
                        candidate_attribute_error_bcpj
                        < base_attribute_error_bcj.unsqueeze(2)
                    )
                    .sum(dim=(0, 1))
                    .detach()
                    .cpu()
                    .double()
                )
                semantic_sample_channels += int(y.shape[0] * y.shape[1])
            if independent_cand_bcpH is not None and penalty_metric_indices:
                base_penalty_error_bcj = torch.stack(
                    [
                        penalty_fns[name](independent_base_bch, y)
                        for name in penalty_metric_names
                    ],
                    dim=-1,
                )
                candidate_penalty_error_bcpj = torch.stack(
                    [
                        torch.stack(
                            [
                                penalty_fns[name](
                                    independent_cand_bcpH[:, :, p, :],
                                    y,
                                )
                                for name in penalty_metric_names
                            ],
                            dim=-1,
                        )
                        for p in range(P)
                    ],
                    dim=2,
                )
                penalty_base_error_sum_j += (
                    base_penalty_error_bcj.sum(dim=(0, 1)).detach().cpu().double()
                )
                penalty_candidate_error_sum_pj += (
                    candidate_penalty_error_bcpj.sum(dim=(0, 1)).detach().cpu().double()
                )
                penalty_positive_mask_bcpj = (
                    candidate_penalty_error_bcpj
                    < base_penalty_error_bcj.unsqueeze(2)
                )
                penalty_positive_count_pj += (
                    penalty_positive_mask_bcpj
                    .sum(dim=(0, 1))
                    .detach()
                    .cpu()
                    .double()
                )
                penalty_positive_base_error_sum_pj += (
                    (
                        base_penalty_error_bcj.unsqueeze(2)
                        * penalty_positive_mask_bcpj
                    )
                    .sum(dim=(0, 1))
                    .detach()
                    .cpu()
                    .double()
                )
                penalty_positive_candidate_error_sum_pj += (
                    (
                        candidate_penalty_error_bcpj
                        * penalty_positive_mask_bcpj
                    )
                    .sum(dim=(0, 1))
                    .detach()
                    .cpu()
                    .double()
                )
                penalty_sample_channels += int(y.shape[0] * y.shape[1])
                if specialization_scale_sweep:
                    candidate_delta_bcpH = (
                        independent_cand_bcpH - independent_base_bch.unsqueeze(2)
                    )
                    for p, name in enumerate(penalty_names):
                        if str(name) not in penalty_fns:
                            continue
                        for scale_idx, scale in enumerate(
                            specialization_scale_sweep
                        ):
                            scaled_candidate = (
                                independent_base_bch
                                + float(scale) * candidate_delta_bcpH[:, :, p, :]
                            )
                            scale_sweep_error_sum_ps[p, scale_idx] += float(
                                penalty_fns[str(name)](scaled_candidate, y)
                                .sum()
                                .detach()
                                .item()
                            )
                    scale_sweep_sample_channels += int(y.shape[0] * y.shape[1])
                if (
                    projection_patch_len > 0
                    and projection_patch_len < int(y.shape[-1])
                    and int(y.shape[-1]) % projection_patch_len == 0
                ):
                    for start in range(0, int(y.shape[-1]), projection_patch_len):
                        end = start + projection_patch_len
                        base_patch = independent_base_bch[..., start:end]
                        target_patch = y[..., start:end]
                        candidate_patch = independent_cand_bcpH[..., start:end]
                        base_patch_error_bcj = torch.stack(
                            [
                                penalty_fns[name](base_patch, target_patch)
                                for name in penalty_metric_names
                            ],
                            dim=-1,
                        )
                        candidate_patch_error_bcpj = torch.stack(
                            [
                                torch.stack(
                                    [
                                        penalty_fns[name](
                                            candidate_patch[:, :, p, :],
                                            target_patch,
                                        )
                                        for name in penalty_metric_names
                                    ],
                                    dim=-1,
                                )
                                for p in range(P)
                            ],
                            dim=2,
                        )
                        patch_penalty_base_error_sum_j += (
                            base_patch_error_bcj.sum(dim=(0, 1))
                            .detach()
                            .cpu()
                            .double()
                        )
                        patch_penalty_candidate_error_sum_pj += (
                            candidate_patch_error_bcpj.sum(dim=(0, 1))
                            .detach()
                            .cpu()
                            .double()
                        )
                        patch_positive_mask_bcpj = (
                            candidate_patch_error_bcpj
                            < base_patch_error_bcj.unsqueeze(2)
                        )
                        patch_penalty_positive_count_pj += (
                            patch_positive_mask_bcpj
                            .sum(dim=(0, 1))
                            .detach()
                            .cpu()
                            .double()
                        )
                        patch_penalty_positive_base_error_sum_pj += (
                            (
                                base_patch_error_bcj.unsqueeze(2)
                                * patch_positive_mask_bcpj
                            )
                            .sum(dim=(0, 1))
                            .detach()
                            .cpu()
                            .double()
                        )
                        patch_penalty_positive_candidate_error_sum_pj += (
                            (
                                candidate_patch_error_bcpj
                                * patch_positive_mask_bcpj
                            )
                            .sum(dim=(0, 1))
                            .detach()
                            .cpu()
                            .double()
                        )
                        patch_penalty_sample_channels += int(y.shape[0] * y.shape[1])
        base_err_bc = (y_base_final - y).pow(2).mean(dim=-1)
        final_err_bc = (y_final - y).pow(2).mean(dim=-1)
        cand_err_bcp = (cand_bcpH - y.unsqueeze(2)).pow(2).mean(dim=-1)
        gain_bcp = base_err_bc.unsqueeze(-1) - cand_err_bcp
        patch_skip_bcq = pred_out.get("patch_skip_bcq")
        patch_route_bcph = pred_out.get("patch_route_bcph")
        patch_selected_penalty_bcq = pred_out.get(
            "patch_selected_penalty_index_bcq"
        )
        if (
            patch_skip_bcq is not None
            and patch_route_bcph is not None
            and patch_selected_penalty_bcq is not None
        ):
            patch_router_seen = True
            B, C, H = y.shape
            Q = int(patch_skip_bcq.shape[2])
            if tuple(patch_skip_bcq.shape) != (B, C, Q):
                raise ValueError(
                    "patch explainability skip tensor must have shape [B,C,Q]."
                )
            if tuple(patch_route_bcph.shape) != (B, C, P, H):
                raise ValueError(
                    "patch explainability route tensor must have shape [B,C,P,H]."
                )
            if tuple(patch_selected_penalty_bcq.shape) != (B, C, Q):
                raise ValueError(
                    "patch explainability selected penalty must have shape [B,C,Q]."
                )
            if Q <= 0 or H % Q != 0:
                raise ValueError(
                    "patch explainability patch count must divide the horizon."
                )
            patch_len = H // Q
            patch_skip_bool_bcq = patch_skip_bcq > 0.5
            patch_route_bcqp = (
                patch_route_bcph.reshape(B, C, P, Q, patch_len)
                .abs()
                .amax(dim=-1)
                .permute(0, 1, 3, 2)
            )
            patch_no_route_bcq = patch_route_bcqp.amax(dim=-1) <= 0.0
            patch_route_skip_mismatch_bcq = (
                patch_no_route_bcq != patch_skip_bool_bcq
            )
            patch_selected_penalty_bcq = patch_selected_penalty_bcq.to(
                device=device,
                dtype=torch.long,
            )
            if bool(
                (
                    (patch_selected_penalty_bcq < 0)
                    | (patch_selected_penalty_bcq >= P)
                ).any().item()
            ):
                raise ValueError(
                    "patch explainability selected penalty index is out of range."
                )

            patch_base_mse_bcq = (y_base_final - y).square().reshape(
                B, C, Q, patch_len
            ).mean(dim=-1)
            patch_final_mse_bcq = (y_final - y).square().reshape(
                B, C, Q, patch_len
            ).mean(dim=-1)
            patch_candidate_mse_bcqp = (
                (cand_bcpH - y.unsqueeze(2))
                .square()
                .reshape(B, C, P, Q, patch_len)
                .mean(dim=-1)
                .permute(0, 1, 3, 2)
            )
            patch_base_mae_bcq = (y_base_final - y).abs().reshape(
                B, C, Q, patch_len
            ).mean(dim=-1)
            patch_candidate_mae_bcqp = (
                (cand_bcpH - y.unsqueeze(2))
                .abs()
                .reshape(B, C, P, Q, patch_len)
                .mean(dim=-1)
                .permute(0, 1, 3, 2)
            )
            if periodic_action_space_enable:
                patch_backbone_mse_bcq = (
                    (y_base - y)
                    .square()
                    .reshape(B, C, Q, patch_len)
                    .mean(dim=-1)
                )
                patch_backbone_mae_bcq = (
                    (y_base - y)
                    .abs()
                    .reshape(B, C, Q, patch_len)
                    .mean(dim=-1)
                )
                patch_current_mae_bcq = (
                    (y_final - y)
                    .abs()
                    .reshape(B, C, Q, patch_len)
                    .mean(dim=-1)
                )
                action_mse_bcqa = torch.cat(
                    [
                        patch_backbone_mse_bcq.unsqueeze(-1),
                        patch_base_mse_bcq.unsqueeze(-1),
                        patch_candidate_mse_bcqp,
                    ],
                    dim=-1,
                )
                action_mae_bcqa = torch.cat(
                    [
                        patch_backbone_mae_bcq.unsqueeze(-1),
                        patch_base_mae_bcq.unsqueeze(-1),
                        patch_candidate_mae_bcqp,
                    ],
                    dim=-1,
                )
                mandatory_mse_bcqa = action_mse_bcqa[..., 1:]
                mandatory_mae_bcqa = action_mae_bcqa[..., 1:]
                mandatory_oracle_idx_bcq = mandatory_mse_bcqa.argmin(
                    dim=-1
                )
                mandatory_dual_safe_bcqa = (
                    (mandatory_mse_bcqa < patch_base_mse_bcq.unsqueeze(-1))
                    & (
                        mandatory_mae_bcqa
                        < patch_base_mae_bcq.unsqueeze(-1)
                    )
                )
                mandatory_dual_safe_bcqa[..., 0] = True
                mandatory_dual_error_bcqa = mandatory_mse_bcqa.masked_fill(
                    ~mandatory_dual_safe_bcqa,
                    float("inf"),
                )
                mandatory_dual_oracle_idx_bcq = (
                    mandatory_dual_error_bcqa.argmin(dim=-1)
                )
                selectable_oracle_idx_bcq = action_mse_bcqa.argmin(dim=-1)
                selectable_dual_safe_bcqa = (
                    (action_mse_bcqa < patch_backbone_mse_bcq.unsqueeze(-1))
                    & (
                        action_mae_bcqa
                        < patch_backbone_mae_bcq.unsqueeze(-1)
                    )
                )
                selectable_dual_safe_bcqa[..., 0] = True
                selectable_dual_error_bcqa = action_mse_bcqa.masked_fill(
                    ~selectable_dual_safe_bcqa,
                    float("inf"),
                )
                selectable_dual_oracle_idx_bcq = (
                    selectable_dual_error_bcqa.argmin(dim=-1)
                )

                def _gather_action_metric(
                    values_bcqa: torch.Tensor,
                    action_bcq: torch.Tensor,
                ) -> torch.Tensor:
                    return values_bcqa.gather(
                        -1,
                        action_bcq.unsqueeze(-1),
                    ).squeeze(-1)

                mandatory_oracle_mse_bcq = _gather_action_metric(
                    mandatory_mse_bcqa,
                    mandatory_oracle_idx_bcq,
                )
                mandatory_oracle_mae_bcq = _gather_action_metric(
                    mandatory_mae_bcqa,
                    mandatory_oracle_idx_bcq,
                )
                mandatory_dual_oracle_mse_bcq = _gather_action_metric(
                    mandatory_mse_bcqa,
                    mandatory_dual_oracle_idx_bcq,
                )
                mandatory_dual_oracle_mae_bcq = _gather_action_metric(
                    mandatory_mae_bcqa,
                    mandatory_dual_oracle_idx_bcq,
                )
                selectable_oracle_mse_bcq = _gather_action_metric(
                    action_mse_bcqa,
                    selectable_oracle_idx_bcq,
                )
                selectable_oracle_mae_bcq = _gather_action_metric(
                    action_mae_bcqa,
                    selectable_oracle_idx_bcq,
                )
                selectable_dual_oracle_mse_bcq = _gather_action_metric(
                    action_mse_bcqa,
                    selectable_dual_oracle_idx_bcq,
                )
                selectable_dual_oracle_mae_bcq = _gather_action_metric(
                    action_mae_bcqa,
                    selectable_dual_oracle_idx_bcq,
                )
                periodic_action_decision_count += int(
                    patch_backbone_mse_bcq.numel()
                )
                periodic_action_backbone_mse_sum += float(
                    patch_backbone_mse_bcq.sum().item()
                )
                periodic_action_backbone_mae_sum += float(
                    patch_backbone_mae_bcq.sum().item()
                )
                periodic_action_periodic_mse_sum += float(
                    patch_base_mse_bcq.sum().item()
                )
                periodic_action_periodic_mae_sum += float(
                    patch_base_mae_bcq.sum().item()
                )
                periodic_action_current_mse_sum += float(
                    patch_final_mse_bcq.sum().item()
                )
                periodic_action_current_mae_sum += float(
                    patch_current_mae_bcq.sum().item()
                )
                periodic_action_mandatory_oracle_mse_sum += float(
                    mandatory_oracle_mse_bcq.sum().item()
                )
                periodic_action_mandatory_oracle_mae_sum += float(
                    mandatory_oracle_mae_bcq.sum().item()
                )
                periodic_action_mandatory_dual_oracle_mse_sum += float(
                    mandatory_dual_oracle_mse_bcq.sum().item()
                )
                periodic_action_mandatory_dual_oracle_mae_sum += float(
                    mandatory_dual_oracle_mae_bcq.sum().item()
                )
                periodic_action_selectable_oracle_mse_sum += float(
                    selectable_oracle_mse_bcq.sum().item()
                )
                periodic_action_selectable_oracle_mae_sum += float(
                    selectable_oracle_mae_bcq.sum().item()
                )
                periodic_action_selectable_dual_oracle_mse_sum += float(
                    selectable_dual_oracle_mse_bcq.sum().item()
                )
                periodic_action_selectable_dual_oracle_mae_sum += float(
                    selectable_dual_oracle_mae_bcq.sum().item()
                )
                periodic_action_mandatory_oracle_count_a += torch.bincount(
                    mandatory_oracle_idx_bcq.detach().cpu().reshape(-1),
                    minlength=P + 1,
                )[: P + 1].to(dtype=torch.float64)
                periodic_action_mandatory_dual_oracle_count_a += torch.bincount(
                    mandatory_dual_oracle_idx_bcq.detach().cpu().reshape(-1),
                    minlength=P + 1,
                )[: P + 1].to(dtype=torch.float64)
                periodic_action_selectable_oracle_count_a += torch.bincount(
                    selectable_oracle_idx_bcq.detach().cpu().reshape(-1),
                    minlength=P + 2,
                )[: P + 2].to(dtype=torch.float64)
                periodic_action_selectable_dual_oracle_count_a += torch.bincount(
                    selectable_dual_oracle_idx_bcq.detach().cpu().reshape(-1),
                    minlength=P + 2,
                )[: P + 2].to(dtype=torch.float64)
            patch_mse_gain_bcqp = (
                patch_base_mse_bcq.unsqueeze(-1) - patch_candidate_mse_bcqp
            )
            patch_mae_gain_bcqp = (
                patch_base_mae_bcq.unsqueeze(-1) - patch_candidate_mae_bcqp
            )
            patch_dual_safe_bcqp = (
                (patch_mse_gain_bcqp > 0.0)
                & (patch_mae_gain_bcqp > 0.0)
            )
            patch_named_penalty_base_bcqp = torch.zeros_like(
                patch_mse_gain_bcqp
            )
            patch_named_penalty_candidate_bcqp = torch.zeros_like(
                patch_mse_gain_bcqp
            )
            patch_named_penalty_supported_p = []
            for p, penalty_name in enumerate(penalty_names):
                penalty_fn = penalty_fns.get(str(penalty_name))
                supported = penalty_fn is not None
                patch_named_penalty_supported_p.append(supported)
                if not supported:
                    continue
                base_errors = []
                candidate_errors = []
                for q in range(Q):
                    start = q * patch_len
                    end = start + patch_len
                    base_errors.append(
                        penalty_fn(
                            y_base_final[..., start:end],
                            y[..., start:end],
                        )
                    )
                    candidate_errors.append(
                        penalty_fn(
                            cand_bcpH[:, :, p, start:end],
                            y[..., start:end],
                        )
                    )
                patch_named_penalty_base_bcqp[..., p] = torch.stack(
                    base_errors,
                    dim=2,
                )
                patch_named_penalty_candidate_bcqp[..., p] = torch.stack(
                    candidate_errors,
                    dim=2,
                )
            patch_named_penalty_gain_bcqp = (
                patch_named_penalty_base_bcqp
                - patch_named_penalty_candidate_bcqp
            )
            patch_named_penalty_positive_bcqp = (
                patch_named_penalty_gain_bcqp > 0.0
            )
            patch_named_penalty_joint_safe_bcqp = (
                patch_named_penalty_positive_bcqp & patch_dual_safe_bcqp
            )
            patch_oracle_dual_skip_bcq = ~patch_dual_safe_bcqp.any(dim=-1)
            patch_adopted_bcq = ~patch_skip_bool_bcq
            patch_selected_mse_gain_bcq = patch_mse_gain_bcqp.gather(
                -1,
                patch_selected_penalty_bcq.unsqueeze(-1),
            ).squeeze(-1)
            patch_selected_mae_gain_bcq = patch_mae_gain_bcqp.gather(
                -1,
                patch_selected_penalty_bcq.unsqueeze(-1),
            ).squeeze(-1)
            patch_selected_dual_safe_bcq = patch_dual_safe_bcqp.gather(
                -1,
                patch_selected_penalty_bcq.unsqueeze(-1),
            ).squeeze(-1)
            patch_false_adopt_bcq = (
                patch_adopted_bcq & (~patch_selected_dual_safe_bcq)
            )

            if specialization_scale_sweep:
                for p, penalty_name in enumerate(penalty_names):
                    penalty_fn = penalty_fns.get(str(penalty_name))
                    if penalty_fn is None:
                        continue
                    selected_p = (
                        patch_adopted_bcq
                        & (patch_selected_penalty_bcq == int(p))
                    )
                    selected_count = float(selected_p.sum().item())
                    if selected_count <= 0.0:
                        continue
                    gated_scale_selected_count_p[p] += selected_count
                    gated_scale_named_base_sum_p[p] += float(
                        patch_named_penalty_base_bcqp[..., p][selected_p]
                        .sum()
                        .item()
                    )
                    gated_scale_mse_base_sum_p[p] += float(
                        patch_base_mse_bcq[selected_p].sum().item()
                    )
                    gated_scale_mae_base_sum_p[p] += float(
                        patch_base_mae_bcq[selected_p].sum().item()
                    )
                    correction_bch = (
                        cand_bcpH[:, :, p, :] - y_base_final
                    )
                    for scale_idx, scale in enumerate(
                        specialization_scale_sweep
                    ):
                        if abs(float(scale)) <= 1.0e-15:
                            named_error_bcq = (
                                patch_named_penalty_base_bcqp[..., p]
                            )
                            mse_error_bcq = patch_base_mse_bcq
                            mae_error_bcq = patch_base_mae_bcq
                        else:
                            scaled_candidate_bch = (
                                y_base_final
                                + float(scale) * correction_bch
                            )
                            mse_error_bcq = (
                                (scaled_candidate_bch - y)
                                .square()
                                .reshape(B, C, Q, patch_len)
                                .mean(dim=-1)
                            )
                            mae_error_bcq = (
                                (scaled_candidate_bch - y)
                                .abs()
                                .reshape(B, C, Q, patch_len)
                                .mean(dim=-1)
                            )
                            named_error_bcq = torch.stack(
                                [
                                    penalty_fn(
                                        scaled_candidate_bch[
                                            ...,
                                            q * patch_len : (q + 1) * patch_len,
                                        ],
                                        y[
                                            ...,
                                            q * patch_len : (q + 1) * patch_len,
                                        ],
                                    )
                                    for q in range(Q)
                                ],
                                dim=2,
                            )
                        gated_scale_named_error_sum_ps[
                            p, scale_idx
                        ] += float(named_error_bcq[selected_p].sum().item())
                        gated_scale_mse_error_sum_ps[
                            p, scale_idx
                        ] += float(mse_error_bcq[selected_p].sum().item())
                        gated_scale_mae_error_sum_ps[
                            p, scale_idx
                        ] += float(mae_error_bcq[selected_p].sum().item())

            for k in range(K):
                ch_mask = cid_c == int(k)
                if not bool(ch_mask.any().item()):
                    continue
                decision_count = int(B * int(ch_mask.sum().item()) * Q)
                patch_decision_count_k[k] += decision_count
                patch_base_mse_sum_k[k] += float(
                    patch_base_mse_bcq[:, ch_mask].sum().item()
                )
                patch_final_mse_sum_k[k] += float(
                    patch_final_mse_bcq[:, ch_mask].sum().item()
                )
                patch_skip_count_k[k] += float(
                    patch_skip_bool_bcq[:, ch_mask].sum().item()
                )
                patch_oracle_dual_skip_count_k[k] += float(
                    patch_oracle_dual_skip_bcq[:, ch_mask].sum().item()
                )
                patch_correct_skip_count_k[k] += float(
                    (
                        patch_skip_bool_bcq[:, ch_mask]
                        & patch_oracle_dual_skip_bcq[:, ch_mask]
                    ).sum().item()
                )
                patch_missed_dual_action_count_k[k] += float(
                    (
                        patch_skip_bool_bcq[:, ch_mask]
                        & (~patch_oracle_dual_skip_bcq[:, ch_mask])
                    ).sum().item()
                )
                patch_route_skip_mismatch_count_k[k] += float(
                    patch_route_skip_mismatch_bcq[:, ch_mask].sum().item()
                )
                patch_adopted_count_k[k] += float(
                    patch_adopted_bcq[:, ch_mask].sum().item()
                )
                patch_selected_dual_safe_count_k[k] += float(
                    (
                        patch_adopted_bcq[:, ch_mask]
                        & patch_selected_dual_safe_bcq[:, ch_mask]
                    ).sum().item()
                )
                patch_false_adopt_count_k[k] += float(
                    patch_false_adopt_bcq[:, ch_mask].sum().item()
                )
                adopted_weight = patch_adopted_bcq[:, ch_mask].to(
                    dtype=patch_selected_mse_gain_bcq.dtype
                )
                false_adopt_weight = patch_false_adopt_bcq[:, ch_mask].to(
                    dtype=patch_selected_mse_gain_bcq.dtype
                )
                patch_selected_mse_gain_sum_k[k] += float(
                    (
                        patch_selected_mse_gain_bcq[:, ch_mask]
                        * adopted_weight
                    ).sum().item()
                )
                patch_selected_mae_gain_sum_k[k] += float(
                    (
                        patch_selected_mae_gain_bcq[:, ch_mask]
                        * adopted_weight
                    ).sum().item()
                )
                patch_false_adopt_mse_cost_sum_k[k] += float(
                    (
                        (-patch_selected_mse_gain_bcq[:, ch_mask]).clamp_min(0.0)
                        * false_adopt_weight
                    ).sum().item()
                )
                patch_false_adopt_mae_cost_sum_k[k] += float(
                    (
                        (-patch_selected_mae_gain_bcq[:, ch_mask]).clamp_min(0.0)
                        * false_adopt_weight
                    ).sum().item()
                )

                selected_idx = patch_selected_penalty_bcq[:, ch_mask]
                selected_active_idx = selected_idx[
                    patch_adopted_bcq[:, ch_mask]
                ].detach().cpu().reshape(-1)
                selected_safe_idx = selected_idx[
                    patch_adopted_bcq[:, ch_mask]
                    & patch_selected_dual_safe_bcq[:, ch_mask]
                ].detach().cpu().reshape(-1)
                if selected_active_idx.numel() > 0:
                    patch_selected_count_kp[k] += torch.bincount(
                        selected_active_idx,
                        minlength=P,
                    )[:P].to(dtype=torch.float64)
                if selected_safe_idx.numel() > 0:
                    patch_selected_dual_safe_count_kp[k] += torch.bincount(
                        selected_safe_idx,
                        minlength=P,
                    )[:P].to(dtype=torch.float64)
                for p in range(P):
                    selected_p = (
                        patch_adopted_bcq[:, ch_mask]
                        & (selected_idx == int(p))
                    )
                    patch_selected_mse_gain_sum_kp[k, p] += float(
                        patch_selected_mse_gain_bcq[:, ch_mask][selected_p]
                        .sum()
                        .item()
                    )
                    patch_selected_mae_gain_sum_kp[k, p] += float(
                        patch_selected_mae_gain_bcq[:, ch_mask][selected_p]
                        .sum()
                        .item()
                    )
                    if not patch_named_penalty_supported_p[p]:
                        continue
                    named_positive = patch_named_penalty_positive_bcqp[
                        :, ch_mask, :, p
                    ]
                    named_joint_safe = patch_named_penalty_joint_safe_bcqp[
                        :, ch_mask, :, p
                    ]
                    patch_named_penalty_available_count_kp[k, p] += float(
                        named_positive.sum().item()
                    )
                    patch_named_penalty_joint_safe_count_kp[k, p] += float(
                        named_joint_safe.sum().item()
                    )
                    patch_selected_named_penalty_positive_count_kp[
                        k, p
                    ] += float(named_positive[selected_p].sum().item())
                    patch_selected_named_penalty_joint_safe_count_kp[
                        k, p
                    ] += float(named_joint_safe[selected_p].sum().item())
                    patch_selected_named_penalty_base_sum_kp[k, p] += float(
                        patch_named_penalty_base_bcqp[
                            :, ch_mask, :, p
                        ][selected_p]
                        .sum()
                        .item()
                    )
                    patch_selected_named_penalty_candidate_sum_kp[
                        k, p
                    ] += float(
                        patch_named_penalty_candidate_bcqp[
                            :, ch_mask, :, p
                        ][selected_p]
                        .sum()
                        .item()
                    )
        if utility_thresholds:
            utility_stats = _cluster_utility_threshold_stats(
                gain_bcp=gain_bcp.detach(),
                cluster_id_c=cluster_id_c,
                K=K,
                allowed_mask_kp=allowed_mask_kp,
                thresholds=utility_thresholds,
            )
            utility_valid_count_kt += utility_stats["valid_count_kt"]
            utility_total_count_k += utility_stats["total_count_k"]
            utility_best_gain_sum_k += utility_stats["best_gain_sum_k"]
            utility_best_gain_positive_count_k += utility_stats["best_gain_positive_count_k"]
        if allowed_mask_device is not None:
            allowed_bcp = allowed_mask_device.index_select(0, cid_c).unsqueeze(0)
            cand_err_for_oracle_bcp = cand_err_bcp.masked_fill(~allowed_bcp, float("inf"))
        else:
            cand_err_for_oracle_bcp = cand_err_bcp
        oracle_penalty_err_bc, oracle_p_bc = cand_err_for_oracle_bcp.min(dim=-1)
        oracle_err_bc = torch.where(torch.isfinite(oracle_penalty_err_bc), oracle_penalty_err_bc, base_err_bc)
        oracle_gain_bc = base_err_bc - oracle_err_bc
        selected_bool_bcp = route_bcp > 0
        selected_positive_bcp = selected_bool_bcp & (gain_bcp > 0)
        positive_oracle_bc = oracle_gain_bc > 0
        raw_route_bcp = mask_bkp[:, cid_c, :]
        raw_score_bcp = raw_route_bcp * probs_bkp[:, cid_c, :]
        top1_p_bc = raw_score_bcp.argmax(dim=-1)
        top1_conf_bc = probs_bkp[:, cid_c, :].gather(-1, top1_p_bc.unsqueeze(-1)).squeeze(-1)
        top1_weight_bc = raw_route_bcp.gather(-1, top1_p_bc.unsqueeze(-1)).squeeze(-1)
        skip_bc = (skip_bk[:, cid_c] > 0.5) if allow_skip else torch.zeros_like(base_err_bc, dtype=torch.bool)
        top1_active_bc = (top1_weight_bc > 0) & (~skip_bc)
        top1_gain_bc = gain_bcp.gather(-1, top1_p_bc.unsqueeze(-1)).squeeze(-1)
        top1_positive_bc = top1_active_bc & (top1_gain_bc > 0)
        top1_harmful_bc = top1_active_bc & (top1_gain_bc <= 0)

        total_decisions += int(base_err_bc.numel())
        total_selected += int(selected_bool_bcp.sum().item())
        total_oracle_positive += int(positive_oracle_bc.sum().item())
        total_selected_positive += int(selected_positive_bcp.sum().item())

        route_label_bk = torch.full((base_err_bc.shape[0], K), -1, device=device, dtype=torch.long)
        for k in range(K):
            ch_mask = cid_c == int(k)
            if not bool(ch_mask.any().item()):
                continue
            base_k = base_err_bc[:, ch_mask]
            final_k = final_err_bc[:, ch_mask]
            cand_err_k = cand_err_bcp[:, ch_mask, :]
            total_k = int(base_k.numel())
            channels_k = int(ch_mask.sum().item())
            total_bc_k[k] += total_k
            base_err_sum_k[k] += float(base_k.sum().item())
            final_err_sum_k[k] += float(final_k.sum().item())
            oracle_err_sum_k[k] += float(oracle_err_bc[:, ch_mask].sum().item())
            cluster_base_err_b = base_k.mean(dim=1)
            cluster_penalty_err_bp = cand_err_k.mean(dim=1)
            if allowed_mask_device is not None:
                allowed_p = allowed_mask_device[int(k)].view(1, P)
                cluster_penalty_err_bp = cluster_penalty_err_bp.masked_fill(~allowed_p, float("inf"))
            best_cluster_penalty_err_b, best_cluster_penalty_p_b = cluster_penalty_err_bp.min(dim=-1)
            has_allowed_penalty_b = torch.isfinite(best_cluster_penalty_err_b)
            best_cluster_penalty_err_for_sum_b = torch.where(
                has_allowed_penalty_b,
                best_cluster_penalty_err_b,
                cluster_base_err_b,
            )
            best_cluster_route_err_b = torch.minimum(cluster_base_err_b, best_cluster_penalty_err_for_sum_b)
            cluster_route_label_b = torch.where(
                cluster_base_err_b <= best_cluster_penalty_err_for_sum_b,
                torch.zeros_like(best_cluster_penalty_p_b),
                best_cluster_penalty_p_b + 1,
            )
            cluster_route_label_b = torch.where(
                has_allowed_penalty_b,
                cluster_route_label_b,
                torch.zeros_like(cluster_route_label_b),
            )
            route_label_bk[:, int(k)] = cluster_route_label_b
            cluster_penalty_oracle_err_sum_k[k] += float(best_cluster_penalty_err_for_sum_b.sum().item() * channels_k)
            cluster_route_oracle_err_sum_k[k] += float(best_cluster_route_err_b.sum().item() * channels_k)
            cluster_route_oracle_skip_count_k[k] += float((cluster_route_label_b == 0).sum().item())
            cluster_route_oracle_decision_count_k[k] += int(cluster_base_err_b.numel())
            fusion_sum_k[k] += float(fusion_bc[:, ch_mask].sum().item())
            skip_count_k[k] += float(skip_bc[:, ch_mask].sum().item())
            skip_on_oracle_positive_count_k[k] += float((skip_bc[:, ch_mask] & positive_oracle_bc[:, ch_mask]).sum().item())

            selected_count_kp[k] += selected_bool_bcp[:, ch_mask, :].sum(dim=(0, 1)).detach().cpu().to(dtype=torch.float64)
            selected_positive_count_kp[k] += selected_positive_bcp[:, ch_mask, :].sum(dim=(0, 1)).detach().cpu().to(dtype=torch.float64)
            gate_prob_sum_kp[k] += probs_bkp[:, k, :].sum(dim=0).detach().cpu().to(dtype=torch.float64) * int(ch_mask.sum().item())
            route_weight_sum_kp[k] += effective_route_bcp[:, ch_mask, :].sum(dim=(0, 1)).detach().cpu().to(dtype=torch.float64)
            selector_sum_kp[k] += selector_bcp[:, ch_mask, :].sum(dim=(0, 1)).detach().cpu().to(dtype=torch.float64)
            intervention_sum_kp[k] += intervention_bcp[:, ch_mask, :].sum(dim=(0, 1)).detach().cpu().to(dtype=torch.float64)
            single_gain_sum_kp[k] += gain_bcp[:, ch_mask, :].sum(dim=(0, 1)).detach().cpu().to(dtype=torch.float64)
            selected_gain_sum_kp[k] += (gain_bcp[:, ch_mask, :] * selected_bool_bcp[:, ch_mask, :].to(gain_bcp.dtype)).sum(dim=(0, 1)).detach().cpu().to(dtype=torch.float64)
            selected_gain_count_kp[k] += selected_bool_bcp[:, ch_mask, :].sum(dim=(0, 1)).detach().cpu().to(dtype=torch.float64)

            oracle_k = oracle_p_bc[:, ch_mask].detach().cpu().reshape(-1)
            oracle_pos_k = oracle_p_bc[:, ch_mask][positive_oracle_bc[:, ch_mask]].detach().cpu().reshape(-1)
            top1_k = top1_p_bc[:, ch_mask].detach().cpu().reshape(-1)
            top1_active_k = top1_p_bc[:, ch_mask][top1_active_bc[:, ch_mask]].detach().cpu().reshape(-1)
            top1_positive_k = top1_p_bc[:, ch_mask][top1_positive_bc[:, ch_mask]].detach().cpu().reshape(-1)
            top1_harmful_k = top1_p_bc[:, ch_mask][top1_harmful_bc[:, ch_mask]].detach().cpu().reshape(-1)
            skipped_top1_k = top1_p_bc[:, ch_mask][skip_bc[:, ch_mask]].detach().cpu().reshape(-1)
            skipped_positive_k = top1_p_bc[:, ch_mask][skip_bc[:, ch_mask] & positive_oracle_bc[:, ch_mask]].detach().cpu().reshape(-1)
            oracle_count_kp[k] += torch.bincount(oracle_k, minlength=P)[:P].to(dtype=torch.float64)
            if oracle_pos_k.numel() > 0:
                positive_oracle_count_kp[k] += torch.bincount(oracle_pos_k, minlength=P)[:P].to(dtype=torch.float64)
            top1_intended_count_kp[k] += torch.bincount(top1_k, minlength=P)[:P].to(dtype=torch.float64)
            if top1_active_k.numel() > 0:
                top1_selected_count_kp[k] += torch.bincount(top1_active_k, minlength=P)[:P].to(dtype=torch.float64)
            if top1_positive_k.numel() > 0:
                top1_selected_positive_count_kp[k] += torch.bincount(top1_positive_k, minlength=P)[:P].to(dtype=torch.float64)
            if top1_harmful_k.numel() > 0:
                harmful_top1_not_skipped_count_kp[k] += torch.bincount(top1_harmful_k, minlength=P)[:P].to(dtype=torch.float64)
            if skipped_top1_k.numel() > 0:
                skipped_top1_count_kp[k] += torch.bincount(skipped_top1_k, minlength=P)[:P].to(dtype=torch.float64)
            if skipped_positive_k.numel() > 0:
                skipped_on_oracle_positive_count_kp[k] += torch.bincount(skipped_positive_k, minlength=P)[:P].to(dtype=torch.float64)
            active_gain_k = top1_gain_bc[:, ch_mask] * top1_active_bc[:, ch_mask].to(dtype=top1_gain_bc.dtype)
            for p in range(P):
                top1_selected_gain_sum_kp[k, p] += float(
                    active_gain_k[top1_p_bc[:, ch_mask] == int(p)].sum().item()
                )
        route_label_feature_chunks.append(feat_bkf.detach().cpu())
        route_label_chunks.append(route_label_bk.detach().cpu())
        route_label_query_start_chunks.append(query_start_abs_b.detach().cpu())
        if top1_confidence_bins:
            top1_conf_chunks.append(top1_conf_bc.detach().cpu())
            top1_gain_chunks.append(top1_gain_bc.detach().cpu())
            top1_penalty_chunks.append(top1_p_bc.detach().cpu())
            top1_active_chunks.append(top1_active_bc.detach().cpu())
            top1_skip_chunks.append(skip_bc.detach().cpu())

    if total_decisions <= 0:
        return None

    if penalty_portrait_kp is not None:
        portrait = penalty_portrait_kp.detach().cpu().to(dtype=torch.float64)
    else:
        portrait = torch.full((K, P), float("nan"), dtype=torch.float64)
    if prior_prob_kp is not None:
        prior = prior_prob_kp.detach().cpu().to(dtype=torch.float64)
    else:
        prior = torch.full((K, P), float("nan"), dtype=torch.float64)
    if allowed_mask_kp is not None:
        allowed = allowed_mask_kp.detach().cpu().to(dtype=torch.float64)
    else:
        allowed = torch.ones(K, P, dtype=torch.float64)

    penalty_corr = {}
    denom_kp = (total_bc_k.view(K, 1)).clamp_min(1.0)
    mean_gain_kp = single_gain_sum_kp / denom_kp
    for p, name in enumerate(penalty_names):
        xs = [float(portrait[k, p].item()) for k in range(K) if math.isfinite(float(portrait[k, p].item()))]
        ys = [float(mean_gain_kp[k, p].item()) for k in range(K) if math.isfinite(float(portrait[k, p].item()))]
        penalty_corr[name] = _pearson_list(xs, ys)

    rows = []
    per_cluster = []
    for k in range(K):
        cluster_total = float(total_bc_k[k].item())
        base_mse_k = float(base_err_sum_k[k].item() / max(cluster_total, 1.0))
        final_mse_k = float(final_err_sum_k[k].item() / max(cluster_total, 1.0))
        oracle_mse_k = float(oracle_err_sum_k[k].item() / max(cluster_total, 1.0))
        cluster_penalty_oracle_mse_k = float(cluster_penalty_oracle_err_sum_k[k].item() / max(cluster_total, 1.0))
        cluster_route_oracle_mse_k = float(cluster_route_oracle_err_sum_k[k].item() / max(cluster_total, 1.0))
        cluster_gain = float(100.0 * (base_mse_k - final_mse_k) / max(abs(base_mse_k), 1.0e-12))
        oracle_gain_pct = float(100.0 * (base_mse_k - oracle_mse_k) / max(abs(base_mse_k), 1.0e-12))
        cluster_penalty_oracle_gain_pct = float(
            100.0 * (base_mse_k - cluster_penalty_oracle_mse_k) / max(abs(base_mse_k), 1.0e-12)
        )
        cluster_route_oracle_gain_pct = float(
            100.0 * (base_mse_k - cluster_route_oracle_mse_k) / max(abs(base_mse_k), 1.0e-12)
        )
        cluster_route_oracle_skip_rate = float(
            cluster_route_oracle_skip_count_k[k].item() / max(cluster_route_oracle_decision_count_k[k].item(), 1.0)
        )
        skip_rate = float(skip_count_k[k].item() / max(cluster_total, 1.0))
        skip_on_positive_rate = float(skip_on_oracle_positive_count_k[k].item() / max(total_bc_k[k].item(), 1.0))
        cluster_rows = []
        for p, name in enumerate(penalty_names):
            selected_count = float(selected_count_kp[k, p].item())
            selected_gain_count = float(selected_gain_count_kp[k, p].item())
            top1_selected_count = float(top1_selected_count_kp[k, p].item())
            mean_gain = float(mean_gain_kp[k, p].item())
            selected_mean_gain = float(selected_gain_sum_kp[k, p].item() / max(selected_gain_count, 1.0))
            top1_selected_mean_gain = float(top1_selected_gain_sum_kp[k, p].item() / max(top1_selected_count, 1.0))
            prior_value = float(prior[k, p].item()) if math.isfinite(float(prior[k, p].item())) else None
            portrait_value = float(portrait[k, p].item()) if math.isfinite(float(portrait[k, p].item())) else None
            allowed_value = bool(allowed[k, p].item() > 0.5)
            if allowed_value and mean_gain > 0.0:
                reason = "train_prior_allowed_and_positive_split_gain"
            elif allowed_value:
                reason = "train_prior_allowed_but_nonpositive_split_gain"
            elif mean_gain > 0.0:
                reason = "blocked_by_train_prior_but_positive_split_gain"
            else:
                reason = "low_prior_or_nonpositive_split_gain"
            row = {
                "split": split_name,
                "cluster_id": int(k),
                "cluster_channels": int(cluster_channel_count[k].item()),
                "penalty": name,
                "train_diagnostic_score": portrait_value,
                "train_prior_prob": prior_value,
                "allowed_by_train_prior": allowed_value,
                "prior_actual_gain_corr_for_penalty": penalty_corr.get(name),
                "selected_count": int(selected_count),
                "selected_rate": float(selected_count / max(cluster_total, 1.0)),
                "top1_intended_count": int(top1_intended_count_kp[k, p].item()),
                "top1_intended_rate": float(top1_intended_count_kp[k, p].item() / max(cluster_total, 1.0)),
                "top1_selected_count": int(top1_selected_count),
                "top1_selected_rate": float(top1_selected_count / max(cluster_total, 1.0)),
                "top1_selected_positive_count": int(top1_selected_positive_count_kp[k, p].item()),
                "top1_selected_positive_rate": float(
                    top1_selected_positive_count_kp[k, p].item() / max(top1_selected_count, 1.0)
                ),
                "top1_selected_mean_gain_mse": top1_selected_mean_gain,
                "harmful_top1_not_skipped_count": int(harmful_top1_not_skipped_count_kp[k, p].item()),
                "harmful_top1_not_skipped_rate": float(
                    harmful_top1_not_skipped_count_kp[k, p].item() / max(top1_selected_count, 1.0)
                ),
                "skipped_top1_count": int(skipped_top1_count_kp[k, p].item()),
                "skipped_top1_rate": float(skipped_top1_count_kp[k, p].item() / max(cluster_total, 1.0)),
                "skipped_on_oracle_positive_count": int(skipped_on_oracle_positive_count_kp[k, p].item()),
                "skipped_on_oracle_positive_rate": float(
                    skipped_on_oracle_positive_count_kp[k, p].item() / max(cluster_total, 1.0)
                ),
                "oracle_count": int(oracle_count_kp[k, p].item()),
                "oracle_rate": float(oracle_count_kp[k, p].item() / max(cluster_total, 1.0)),
                "positive_oracle_count": int(positive_oracle_count_kp[k, p].item()),
                "positive_oracle_rate": float(positive_oracle_count_kp[k, p].item() / max(cluster_total, 1.0)),
                "selected_positive_count": int(selected_positive_count_kp[k, p].item()),
                "selected_positive_rate": float(selected_positive_count_kp[k, p].item() / max(selected_count, 1.0)),
                "mean_gate_prob": float(gate_prob_sum_kp[k, p].item() / max(cluster_total, 1.0)),
                "mean_effective_route_weight": float(route_weight_sum_kp[k, p].item() / max(cluster_total, 1.0)),
                "mean_selector": float(selector_sum_kp[k, p].item() / max(cluster_total, 1.0)),
                "mean_intervention": float(intervention_sum_kp[k, p].item() / max(cluster_total, 1.0)),
                "mean_single_penalty_gain_mse": mean_gain,
                "selected_mean_gain_mse": selected_mean_gain,
                "cluster_base_mse": base_mse_k,
                "cluster_final_mse": final_mse_k,
                "cluster_oracle_mse": oracle_mse_k,
                "cluster_penalty_oracle_mse": cluster_penalty_oracle_mse_k,
                "cluster_penalty_oracle_gain_pct_vs_base": cluster_penalty_oracle_gain_pct,
                "cluster_route_oracle_mse": cluster_route_oracle_mse_k,
                "cluster_route_oracle_gain_pct_vs_base": cluster_route_oracle_gain_pct,
                "cluster_route_oracle_skip_rate": cluster_route_oracle_skip_rate,
                "cluster_final_gain_pct": cluster_gain,
                "cluster_oracle_gain_pct_vs_base": oracle_gain_pct,
                "cluster_skip_rate": skip_rate,
                "cluster_skip_on_oracle_positive_rate": skip_on_positive_rate,
                "mean_fusion_gate": float(fusion_sum_k[k].item() / max(cluster_total, 1.0)),
                "reason": reason,
            }
            rows.append(row)
            cluster_rows.append(row)
        cluster_rows_sorted = sorted(
            cluster_rows,
            key=lambda item: (
                bool(item["allowed_by_train_prior"]),
                float(item["mean_single_penalty_gain_mse"]),
                float(item["selected_rate"]),
            ),
            reverse=True,
        )
        per_cluster.append(
            {
                "cluster_id": int(k),
                "channels": int(cluster_channel_count[k].item()),
                "base_mse": base_mse_k,
                "final_mse": final_mse_k,
                "oracle_mse": oracle_mse_k,
                "cluster_penalty_oracle_mse": cluster_penalty_oracle_mse_k,
                "cluster_penalty_oracle_gain_pct_vs_base": cluster_penalty_oracle_gain_pct,
                "cluster_route_oracle_mse": cluster_route_oracle_mse_k,
                "cluster_route_oracle_gain_pct_vs_base": cluster_route_oracle_gain_pct,
                "cluster_route_oracle_skip_rate": cluster_route_oracle_skip_rate,
                "final_gain_pct": cluster_gain,
                "oracle_gain_pct_vs_base": oracle_gain_pct,
                "skip_count": int(skip_count_k[k].item()),
                "skip_rate": skip_rate,
                "skip_on_oracle_positive_count": int(skip_on_oracle_positive_count_k[k].item()),
                "skip_on_oracle_positive_rate": skip_on_positive_rate,
                "top_penalties": [
                    {
                        "penalty": item["penalty"],
                        "allowed_by_train_prior": item["allowed_by_train_prior"],
                        "train_prior_prob": item["train_prior_prob"],
                        "mean_single_penalty_gain_mse": item["mean_single_penalty_gain_mse"],
                        "selected_rate": item["selected_rate"],
                        "top1_selected_positive_rate": item["top1_selected_positive_rate"],
                        "harmful_top1_not_skipped_rate": item["harmful_top1_not_skipped_rate"],
                        "skipped_top1_rate": item["skipped_top1_rate"],
                        "reason": item["reason"],
                    }
                    for item in cluster_rows_sorted[: min(3, len(cluster_rows_sorted))]
                ],
            }
        )

    patch_router_route_diagnostics = None
    if patch_router_seen:
        def _patch_value(
            values_k: torch.Tensor,
            cluster_idx: Optional[int],
        ) -> float:
            if cluster_idx is None:
                return float(values_k.sum().item())
            return float(values_k[int(cluster_idx)].item())

        def _patch_penalty_value(
            values_kp: torch.Tensor,
            cluster_idx: Optional[int],
            penalty_idx: int,
        ) -> float:
            if cluster_idx is None:
                return float(values_kp[:, int(penalty_idx)].sum().item())
            return float(
                values_kp[int(cluster_idx), int(penalty_idx)].item()
            )

        def _patch_scope_summary(
            cluster_idx: Optional[int],
        ) -> Dict[str, object]:
            decisions = _patch_value(patch_decision_count_k, cluster_idx)
            base_mse_sum = _patch_value(patch_base_mse_sum_k, cluster_idx)
            final_mse_sum = _patch_value(patch_final_mse_sum_k, cluster_idx)
            actual_skip = _patch_value(patch_skip_count_k, cluster_idx)
            oracle_skip = _patch_value(
                patch_oracle_dual_skip_count_k,
                cluster_idx,
            )
            correct_skip = _patch_value(
                patch_correct_skip_count_k,
                cluster_idx,
            )
            missed_action = _patch_value(
                patch_missed_dual_action_count_k,
                cluster_idx,
            )
            mismatch = _patch_value(
                patch_route_skip_mismatch_count_k,
                cluster_idx,
            )
            adopted = _patch_value(patch_adopted_count_k, cluster_idx)
            dual_safe = _patch_value(
                patch_selected_dual_safe_count_k,
                cluster_idx,
            )
            false_adopt = _patch_value(
                patch_false_adopt_count_k,
                cluster_idx,
            )
            mse_gain_sum = _patch_value(
                patch_selected_mse_gain_sum_k,
                cluster_idx,
            )
            mae_gain_sum = _patch_value(
                patch_selected_mae_gain_sum_k,
                cluster_idx,
            )
            base_mse_scope = base_mse_sum / max(decisions, 1.0)
            final_mse_scope = final_mse_sum / max(decisions, 1.0)
            payload: Dict[str, object] = {
                "cluster_id": (
                    None if cluster_idx is None else int(cluster_idx)
                ),
                "channels": (
                    int(cluster_channel_count.sum().item())
                    if cluster_idx is None
                    else int(cluster_channel_count[int(cluster_idx)].item())
                ),
                "decision_count": int(decisions),
                "base_mse": float(base_mse_scope),
                "final_mse": float(final_mse_scope),
                "final_gain_pct_vs_base": float(
                    100.0
                    * (base_mse_scope - final_mse_scope)
                    / max(abs(base_mse_scope), 1.0e-12)
                ),
                "actual_skip_count": int(actual_skip),
                "actual_skip_rate": float(actual_skip / max(decisions, 1.0)),
                "oracle_dual_skip_count": int(oracle_skip),
                "oracle_dual_skip_rate": float(
                    oracle_skip / max(decisions, 1.0)
                ),
                "correct_skip_count": int(correct_skip),
                "skip_precision": float(
                    correct_skip / max(actual_skip, 1.0)
                ),
                "skip_recall": float(
                    correct_skip / max(oracle_skip, 1.0)
                ),
                "missed_dual_action_count": int(missed_action),
                "missed_dual_action_rate": float(
                    missed_action / max(decisions, 1.0)
                ),
                "route_skip_mismatch_count": int(mismatch),
                "route_skip_mismatch_rate": float(
                    mismatch / max(decisions, 1.0)
                ),
                "adopted_count": int(adopted),
                "adopted_rate": float(adopted / max(decisions, 1.0)),
                "selected_dual_safe_count": int(dual_safe),
                "selected_dual_safe_precision": float(
                    dual_safe / max(adopted, 1.0)
                ),
                "false_adopt_count": int(false_adopt),
                "selected_dual_harmful_rate": float(
                    false_adopt / max(adopted, 1.0)
                ),
                "selected_mse_gain_sum": float(mse_gain_sum),
                "selected_mae_gain_sum": float(mae_gain_sum),
                "selected_mean_mse_gain": float(
                    mse_gain_sum / max(adopted, 1.0)
                ),
                "selected_mean_mae_gain": float(
                    mae_gain_sum / max(adopted, 1.0)
                ),
                "false_adopt_mse_cost_sum": _patch_value(
                    patch_false_adopt_mse_cost_sum_k,
                    cluster_idx,
                ),
                "false_adopt_mae_cost_sum": _patch_value(
                    patch_false_adopt_mae_cost_sum_k,
                    cluster_idx,
                ),
            }
            penalty_rows = []
            for penalty_idx, penalty_name in enumerate(penalty_names):
                selected = _patch_penalty_value(
                    patch_selected_count_kp,
                    cluster_idx,
                    penalty_idx,
                )
                selected_safe = _patch_penalty_value(
                    patch_selected_dual_safe_count_kp,
                    cluster_idx,
                    penalty_idx,
                )
                selected_mse_gain = _patch_penalty_value(
                    patch_selected_mse_gain_sum_kp,
                    cluster_idx,
                    penalty_idx,
                )
                selected_mae_gain = _patch_penalty_value(
                    patch_selected_mae_gain_sum_kp,
                    cluster_idx,
                    penalty_idx,
                )
                named_supported = str(penalty_name) in penalty_fns
                named_available = _patch_penalty_value(
                    patch_named_penalty_available_count_kp,
                    cluster_idx,
                    penalty_idx,
                )
                named_joint_safe = _patch_penalty_value(
                    patch_named_penalty_joint_safe_count_kp,
                    cluster_idx,
                    penalty_idx,
                )
                selected_named_positive = _patch_penalty_value(
                    patch_selected_named_penalty_positive_count_kp,
                    cluster_idx,
                    penalty_idx,
                )
                selected_named_joint_safe = _patch_penalty_value(
                    patch_selected_named_penalty_joint_safe_count_kp,
                    cluster_idx,
                    penalty_idx,
                )
                selected_named_base = _patch_penalty_value(
                    patch_selected_named_penalty_base_sum_kp,
                    cluster_idx,
                    penalty_idx,
                )
                selected_named_candidate = _patch_penalty_value(
                    patch_selected_named_penalty_candidate_sum_kp,
                    cluster_idx,
                    penalty_idx,
                )
                named_gain = selected_named_base - selected_named_candidate
                mean_mse_gain = selected_mse_gain / max(selected, 1.0)
                mean_mae_gain = selected_mae_gain / max(selected, 1.0)
                selected_region_proof_pass = bool(
                    named_supported
                    and selected > 0.0
                    and named_gain > 0.0
                    and mean_mse_gain > 0.0
                    and mean_mae_gain > 0.0
                )
                penalty_rows.append(
                    {
                        "penalty_index": int(penalty_idx),
                        "penalty": str(penalty_name),
                        "selected_count": int(selected),
                        "selected_rate": float(
                            selected / max(decisions, 1.0)
                        ),
                        "selected_dual_safe_count": int(selected_safe),
                        "selected_dual_safe_precision": float(
                            selected_safe / max(selected, 1.0)
                        ),
                        "selected_mean_mse_gain": float(mean_mse_gain),
                        "selected_mean_mae_gain": float(mean_mae_gain),
                        "named_penalty_supported": bool(named_supported),
                        "named_penalty_available_count": int(named_available),
                        "named_penalty_available_rate": float(
                            named_available / max(decisions, 1.0)
                        ),
                        "joint_effective_available_count": int(named_joint_safe),
                        "joint_effective_available_rate": float(
                            named_joint_safe / max(decisions, 1.0)
                        ),
                        "selected_named_penalty_positive_count": int(
                            selected_named_positive
                        ),
                        "selected_named_penalty_positive_precision": float(
                            selected_named_positive / max(selected, 1.0)
                        ),
                        "selected_joint_effective_count": int(
                            selected_named_joint_safe
                        ),
                        "selected_joint_effective_precision": float(
                            selected_named_joint_safe / max(selected, 1.0)
                        ),
                        "selected_named_penalty_base_error": float(
                            selected_named_base / max(selected, 1.0)
                        ),
                        "selected_named_penalty_candidate_error": float(
                            selected_named_candidate / max(selected, 1.0)
                        ),
                        "selected_named_penalty_mean_gain": float(
                            named_gain / max(selected, 1.0)
                        ),
                        "selected_named_penalty_reduction_pct": float(
                            100.0
                            * named_gain
                            / max(abs(selected_named_base), 1.0e-12)
                        ),
                        "selected_region_proof_pass": selected_region_proof_pass,
                    }
                )
            activated_rows = [
                row for row in penalty_rows if int(row["selected_count"]) > 0
            ]
            available_rows = [
                row
                for row in penalty_rows
                if int(row["joint_effective_available_count"]) > 0
            ]
            payload["per_penalty"] = penalty_rows
            payload["conditional_specialization_proof"] = {
                "definition": (
                    "on each gate-activated patch, the selected adapter must "
                    "reduce its matching exact penalty and have positive mean "
                    "MSE and MAE utility versus periodic-only"
                ),
                "activated_expert_count": int(len(activated_rows)),
                "available_expert_count": int(len(available_rows)),
                "all_available_experts_activated": bool(
                    available_rows
                    and all(
                        int(row["selected_count"]) > 0
                        for row in available_rows
                    )
                ),
                "all_activated_experts_pass": bool(
                    activated_rows
                    and all(
                        bool(row["selected_region_proof_pass"])
                        for row in activated_rows
                    )
                ),
            }
            return payload

        patch_router_route_diagnostics = {
            "routing_granularity": "sample_channel_patch",
            "skip_source": "patch_skip_bcq",
            "selected_penalty_source": "patch_selected_penalty_index_bcq",
            "oracle_definition": (
                "skip iff no candidate has both positive MSE and MAE patch utility"
            ),
            "aggregate": _patch_scope_summary(None),
            "per_cluster": [
                _patch_scope_summary(cluster_idx)
                for cluster_idx in range(K)
            ],
        }

    periodic_action_space_oracle = None
    if periodic_action_space_enable and periodic_action_decision_count > 0:
        action_decisions = float(periodic_action_decision_count)

        def _action_mean(total: float) -> float:
            return float(total / max(action_decisions, 1.0))

        def _action_gain_pct(reference: float, candidate: float) -> float:
            return float(
                100.0
                * (float(reference) - float(candidate))
                / max(abs(float(reference)), 1.0e-12)
            )

        backbone_action_mse = _action_mean(
            periodic_action_backbone_mse_sum
        )
        backbone_action_mae = _action_mean(
            periodic_action_backbone_mae_sum
        )
        periodic_action_mse = _action_mean(
            periodic_action_periodic_mse_sum
        )
        periodic_action_mae = _action_mean(
            periodic_action_periodic_mae_sum
        )
        current_action_mse = _action_mean(
            periodic_action_current_mse_sum
        )
        current_action_mae = _action_mean(
            periodic_action_current_mae_sum
        )
        mandatory_oracle_mse = _action_mean(
            periodic_action_mandatory_oracle_mse_sum
        )
        mandatory_oracle_mae = _action_mean(
            periodic_action_mandatory_oracle_mae_sum
        )
        mandatory_dual_oracle_mse = _action_mean(
            periodic_action_mandatory_dual_oracle_mse_sum
        )
        mandatory_dual_oracle_mae = _action_mean(
            periodic_action_mandatory_dual_oracle_mae_sum
        )
        selectable_oracle_mse = _action_mean(
            periodic_action_selectable_oracle_mse_sum
        )
        selectable_oracle_mae = _action_mean(
            periodic_action_selectable_oracle_mae_sum
        )
        selectable_dual_oracle_mse = _action_mean(
            periodic_action_selectable_dual_oracle_mse_sum
        )
        selectable_dual_oracle_mae = _action_mean(
            periodic_action_selectable_dual_oracle_mae_sum
        )

        def _action_rates(
            names: List[str],
            counts: torch.Tensor,
        ) -> List[Dict[str, object]]:
            return [
                {
                    "action": str(name),
                    "count": int(counts[idx].item()),
                    "rate": float(
                        counts[idx].item() / max(action_decisions, 1.0)
                    ),
                }
                for idx, name in enumerate(names)
            ]

        selectable_mse_rates = _action_rates(
            periodic_action_names,
            periodic_action_selectable_oracle_count_a,
        )
        selectable_dual_rates = _action_rates(
            periodic_action_names,
            periodic_action_selectable_dual_oracle_count_a,
        )
        periodic_action_space_oracle = {
            "split": str(split_name),
            "test_selected": str(split_name).lower() == "test",
            "routing_granularity": "sample_channel_patch",
            "decision_count": int(periodic_action_decision_count),
            "action_space": periodic_action_names,
            "backbone_only": {
                "mse": backbone_action_mse,
                "mae": backbone_action_mae,
            },
            "periodic_only": {
                "mse": periodic_action_mse,
                "mae": periodic_action_mae,
                "mse_gain_pct_vs_backbone": _action_gain_pct(
                    backbone_action_mse,
                    periodic_action_mse,
                ),
                "mae_gain_pct_vs_backbone": _action_gain_pct(
                    backbone_action_mae,
                    periodic_action_mae,
                ),
            },
            "current_gate": {
                "mse": current_action_mse,
                "mae": current_action_mae,
                "mse_gain_pct_vs_periodic": _action_gain_pct(
                    periodic_action_mse,
                    current_action_mse,
                ),
                "mae_gain_pct_vs_periodic": _action_gain_pct(
                    periodic_action_mae,
                    current_action_mae,
                ),
                "mse_gain_pct_vs_backbone": _action_gain_pct(
                    backbone_action_mse,
                    current_action_mse,
                ),
                "mae_gain_pct_vs_backbone": _action_gain_pct(
                    backbone_action_mae,
                    current_action_mae,
                ),
            },
            "periodic_mandatory_mse_oracle": {
                "definition": (
                    "choose the lowest-MSE action among periodic-only and "
                    "periodic plus one PKR adapter"
                ),
                "mse": mandatory_oracle_mse,
                "mae_at_mse_choice": mandatory_oracle_mae,
                "mse_gain_pct_vs_periodic": _action_gain_pct(
                    periodic_action_mse,
                    mandatory_oracle_mse,
                ),
                "action_rates": _action_rates(
                    periodic_action_names[1:],
                    periodic_action_mandatory_oracle_count_a,
                ),
            },
            "periodic_mandatory_dual_safe_oracle": {
                "definition": (
                    "choose the lowest-MSE periodic action only when both MSE "
                    "and MAE improve over periodic-only"
                ),
                "mse": mandatory_dual_oracle_mse,
                "mae": mandatory_dual_oracle_mae,
                "mse_gain_pct_vs_periodic": _action_gain_pct(
                    periodic_action_mse,
                    mandatory_dual_oracle_mse,
                ),
                "mae_gain_pct_vs_periodic": _action_gain_pct(
                    periodic_action_mae,
                    mandatory_dual_oracle_mae,
                ),
                "action_rates": _action_rates(
                    periodic_action_names[1:],
                    periodic_action_mandatory_dual_oracle_count_a,
                ),
            },
            "periodic_selectable_mse_oracle": {
                "definition": (
                    "choose the lowest-MSE action among backbone-only, "
                    "periodic-only, and periodic plus one PKR adapter"
                ),
                "mse": selectable_oracle_mse,
                "mae_at_mse_choice": selectable_oracle_mae,
                "mse_gain_pct_vs_backbone": _action_gain_pct(
                    backbone_action_mse,
                    selectable_oracle_mse,
                ),
                "mse_gain_pct_vs_periodic_mandatory_oracle": (
                    _action_gain_pct(
                        mandatory_oracle_mse,
                        selectable_oracle_mse,
                    )
                ),
                "action_rates": selectable_mse_rates,
            },
            "periodic_selectable_dual_safe_oracle": {
                "definition": (
                    "choose the lowest-MSE action only when both MSE and MAE "
                    "improve over backbone-only"
                ),
                "mse": selectable_dual_oracle_mse,
                "mae": selectable_dual_oracle_mae,
                "mse_gain_pct_vs_backbone": _action_gain_pct(
                    backbone_action_mse,
                    selectable_dual_oracle_mse,
                ),
                "mae_gain_pct_vs_backbone": _action_gain_pct(
                    backbone_action_mae,
                    selectable_dual_oracle_mae,
                ),
                "mse_gain_pct_vs_periodic_mandatory_dual_oracle": (
                    _action_gain_pct(
                        mandatory_dual_oracle_mse,
                        selectable_dual_oracle_mse,
                    )
                ),
                "mae_gain_pct_vs_periodic_mandatory_dual_oracle": (
                    _action_gain_pct(
                        mandatory_dual_oracle_mae,
                        selectable_dual_oracle_mae,
                    )
                ),
                "backbone_action_rate": float(
                    selectable_dual_rates[0]["rate"]
                ),
                "periodic_only_action_rate": float(
                    selectable_dual_rates[1]["rate"]
                ),
                "periodic_plus_other_action_rate": float(
                    sum(row["rate"] for row in selectable_dual_rates[2:])
                ),
                "action_rates": selectable_dual_rates,
            },
        }

    base_mse = float(base_err_sum_k.sum().item() / max(total_decisions, 1))
    final_mse = float(final_err_sum_k.sum().item() / max(total_decisions, 1))
    oracle_mse = float(oracle_err_sum_k.sum().item() / max(total_decisions, 1))
    cluster_penalty_oracle_mse = float(cluster_penalty_oracle_err_sum_k.sum().item() / max(total_decisions, 1))
    cluster_route_oracle_mse = float(cluster_route_oracle_err_sum_k.sum().item() / max(total_decisions, 1))
    cluster_route_oracle_decisions = float(cluster_route_oracle_decision_count_k.sum().item())
    cluster_route_oracle_skip_rate = float(
        cluster_route_oracle_skip_count_k.sum().item() / max(cluster_route_oracle_decisions, 1.0)
    )
    utility_threshold_summary = []
    if utility_thresholds:
        for k in range(K):
            total_k = float(utility_total_count_k[k].item())
            utility_threshold_summary.append(
                {
                    "cluster_id": int(k),
                    "windows": int(total_k),
                    "mean_best_allowed_gain_mse": float(utility_best_gain_sum_k[k].item() / max(total_k, 1.0)),
                    "positive_best_allowed_gain_rate": float(
                        utility_best_gain_positive_count_k[k].item() / max(total_k, 1.0)
                    ),
                    "thresholds": [
                        {
                            "min_gain": float(threshold),
                            "valid_count": int(utility_valid_count_kt[k, t].item()),
                            "valid_rate": float(utility_valid_count_kt[k, t].item() / max(total_k, 1.0)),
                        }
                        for t, threshold in enumerate(utility_thresholds)
                    ],
                }
            )
    route_label_feature_diagnostics = None
    if route_label_feature_chunks and route_label_chunks:
        route_label_feature_diagnostics = _cluster_route_label_feature_diagnostics(
            feat_bkf=torch.cat(route_label_feature_chunks, dim=0),
            route_label_bk=torch.cat(route_label_chunks, dim=0),
            penalty_names=penalty_names,
            feature_names=_gate_feature_names_for_mode(gate_feature_mode),
        )
    route_label_phase_diagnostics = None
    if route_label_phase_periods and route_label_chunks and route_label_query_start_chunks:
        route_label_phase_diagnostics = _cluster_route_label_phase_diagnostics(
            query_start_abs_b=torch.cat(route_label_query_start_chunks, dim=0),
            route_label_bk=torch.cat(route_label_chunks, dim=0),
            penalty_names=penalty_names,
            periods=route_label_phase_periods,
            num_bins=route_label_phase_bins,
            phase_offset=int(input_len),
        )
    top1_confidence_gain_diagnostics = None
    if top1_confidence_bins and top1_conf_chunks:
        top1_confidence_gain_diagnostics = _cluster_top1_confidence_gain_diagnostics(
            top1_conf_bc=torch.cat(top1_conf_chunks, dim=0),
            top1_gain_bc=torch.cat(top1_gain_chunks, dim=0),
            top1_p_bc=torch.cat(top1_penalty_chunks, dim=0),
            top1_active_bc=torch.cat(top1_active_chunks, dim=0),
            skip_bc=torch.cat(top1_skip_chunks, dim=0),
            cluster_id_c=cluster_id_c.detach().cpu(),
            K=K,
            penalty_names=penalty_names,
            bins=top1_confidence_bins,
        )
    named_attribute_specialization = None
    if semantic_metric_indices and semantic_sample_channels > 0:
        semantic_base_mean_j = (
            semantic_base_error_sum_j / float(semantic_sample_channels)
        )
        semantic_candidate_mean_pj = (
            semantic_candidate_error_sum_pj / float(semantic_sample_channels)
        )
        semantic_gain_pj = (
            semantic_base_mean_j.unsqueeze(0) - semantic_candidate_mean_pj
        )
        semantic_gain_pct_pj = (
            100.0
            * semantic_gain_pj
            / semantic_base_mean_j.abs().clamp_min(1.0e-12).unsqueeze(0)
        )
        semantic_positive_rate_pj = (
            semantic_positive_count_pj / float(semantic_sample_channels)
        )
        diagonal_rows = []
        for metric_col, expert_idx in enumerate(semantic_metric_indices):
            matching_gain = float(semantic_gain_pj[expert_idx, metric_col].item())
            matching_gain_pct = float(
                semantic_gain_pct_pj[expert_idx, metric_col].item()
            )
            better_expert_count = int(
                (
                    semantic_gain_pj[:, metric_col]
                    > semantic_gain_pj[expert_idx, metric_col]
                ).sum().item()
            )
            diagonal_rows.append(
                {
                    "expert": str(penalty_names[expert_idx]),
                    "metric": str(semantic_metric_names[metric_col]),
                    "base_error": float(semantic_base_mean_j[metric_col].item()),
                    "candidate_error": float(
                        semantic_candidate_mean_pj[expert_idx, metric_col].item()
                    ),
                    "mean_signed_improvement": matching_gain,
                    "relative_improvement_pct": matching_gain_pct,
                    "positive_sample_channel_rate": float(
                        semantic_positive_rate_pj[expert_idx, metric_col].item()
                    ),
                    "rank_by_mean_improvement": int(better_expert_count + 1),
                    "improved": bool(matching_gain > 0.0),
                }
            )
        named_attribute_specialization = {
            "base_definition": "backbone + fixed periodic expert on the exact eval path",
            "candidate_definition": "base + one projected adapter; intervention, selector, and patch route disabled",
            "aggregation_unit": "sample-channel full horizon",
            "sample_channels": int(semantic_sample_channels),
            "expert_names": [str(name) for name in penalty_names],
            "metric_names": list(semantic_metric_names),
            "base_error_mean_by_metric": semantic_base_mean_j.tolist(),
            "candidate_error_mean_expert_by_metric": semantic_candidate_mean_pj.tolist(),
            "mean_signed_improvement_expert_by_metric": semantic_gain_pj.tolist(),
            "relative_improvement_pct_expert_by_metric": semantic_gain_pct_pj.tolist(),
            "positive_sample_channel_rate_expert_by_metric": semantic_positive_rate_pj.tolist(),
            "diagonal": diagonal_rows,
            "all_diagonal_improved": bool(
                diagonal_rows and all(bool(row["improved"]) for row in diagonal_rows)
            ),
        }
    named_penalty_specialization = _specialization_summary_from_accumulators(
        expert_names=[str(name) for name in penalty_names],
        metric_names=penalty_metric_names,
        metric_expert_indices=penalty_metric_indices,
        base_error_sum_j=penalty_base_error_sum_j,
        candidate_error_sum_pj=penalty_candidate_error_sum_pj,
        positive_count_pj=penalty_positive_count_pj,
        positive_base_error_sum_pj=penalty_positive_base_error_sum_pj,
        positive_candidate_error_sum_pj=penalty_positive_candidate_error_sum_pj,
        sample_channels=penalty_sample_channels,
        base_definition="backbone + fixed periodic expert on the exact eval path",
        candidate_definition=(
            "base + one projected adapter; intervention, selector, and patch route disabled"
        ),
        aggregation_unit="sample-channel full horizon; exact configured penalty functions",
    )
    patch_named_penalty_specialization = _specialization_summary_from_accumulators(
        expert_names=[str(name) for name in penalty_names],
        metric_names=penalty_metric_names,
        metric_expert_indices=penalty_metric_indices,
        base_error_sum_j=patch_penalty_base_error_sum_j,
        candidate_error_sum_pj=patch_penalty_candidate_error_sum_pj,
        positive_count_pj=patch_penalty_positive_count_pj,
        positive_base_error_sum_pj=patch_penalty_positive_base_error_sum_pj,
        positive_candidate_error_sum_pj=patch_penalty_positive_candidate_error_sum_pj,
        sample_channels=patch_penalty_sample_channels,
        base_definition="backbone + fixed periodic expert on the exact eval path",
        candidate_definition=(
            "base + one projected adapter; intervention, selector, and patch route disabled"
        ),
        aggregation_unit=(
            f"sample-channel-{projection_patch_len}-step-patch; exact configured penalty functions"
        ),
    )
    adapter_scale_sweep = None
    if specialization_scale_sweep and scale_sweep_sample_channels > 0:
        mean_error_ps = (
            scale_sweep_error_sum_ps / float(scale_sweep_sample_channels)
        )
        scale_rows = []
        for p, name in enumerate(penalty_names):
            if p not in penalty_metric_indices:
                continue
            metric_col = penalty_metric_indices.index(p)
            base_error = float(
                penalty_base_error_sum_j[metric_col].item()
                / float(penalty_sample_channels)
            )
            best_scale_idx = int(mean_error_ps[p].argmin().item())
            best_error = float(mean_error_ps[p, best_scale_idx].item())
            scale_rows.append(
                {
                    "expert": str(name),
                    "metric": str(name),
                    "base_error": base_error,
                    "scales": list(specialization_scale_sweep),
                    "error_by_scale": mean_error_ps[p].tolist(),
                    "improvement_pct_by_scale": [
                        float(
                            100.0
                            * (base_error - float(error))
                            / max(abs(base_error), 1.0e-12)
                        )
                        for error in mean_error_ps[p].tolist()
                    ],
                    "best_scale": float(
                        specialization_scale_sweep[best_scale_idx]
                    ),
                    "best_error": best_error,
                    "best_improvement_pct": float(
                        100.0
                        * (base_error - best_error)
                        / max(abs(base_error), 1.0e-12)
                    ),
                    "positive_nonzero_scale_exists": bool(
                        any(
                            float(scale) > 0.0 and float(error) < base_error
                            for scale, error in zip(
                                specialization_scale_sweep,
                                mean_error_ps[p].tolist(),
                            )
                        )
                    ),
                    "negative_scale_improves": bool(
                        any(
                            float(scale) < 0.0 and float(error) < base_error
                            for scale, error in zip(
                                specialization_scale_sweep,
                                mean_error_ps[p].tolist(),
                            )
                        )
                    ),
                    "nonzero_scale_improves": bool(
                        any(
                            abs(float(scale)) > 0.0 and float(error) < base_error
                            for scale, error in zip(
                                specialization_scale_sweep,
                                mean_error_ps[p].tolist(),
                            )
                        )
                    ),
                    "best_scale_sign": (
                        "positive"
                        if float(specialization_scale_sweep[best_scale_idx]) > 0.0
                        else (
                            "negative"
                            if float(specialization_scale_sweep[best_scale_idx]) < 0.0
                            else "no_op"
                        )
                    ),
                }
            )
        adapter_scale_sweep = {
            "selection_split": str(split_name),
            "test_selected": str(split_name).lower() == "test",
            "candidate_definition": "base + scale * trained_adapter_correction",
            "sample_channels": int(scale_sweep_sample_channels),
            "rows": scale_rows,
            "all_experts_have_positive_nonzero_scale": bool(
                scale_rows
                and all(
                    bool(row["positive_nonzero_scale_exists"])
                    for row in scale_rows
                )
            ),
        }
    gated_adapter_scale_sweep = None
    if specialization_scale_sweep and bool(
        (gated_scale_selected_count_p > 0.0).any().item()
    ):
        gated_scale_rows = []
        for p, name in enumerate(penalty_names):
            selected_count = float(gated_scale_selected_count_p[p].item())
            if selected_count <= 0.0 or str(name) not in penalty_fns:
                continue
            named_base = float(
                gated_scale_named_base_sum_p[p].item() / selected_count
            )
            mse_base = float(
                gated_scale_mse_base_sum_p[p].item() / selected_count
            )
            mae_base = float(
                gated_scale_mae_base_sum_p[p].item() / selected_count
            )
            named_errors = (
                gated_scale_named_error_sum_ps[p] / selected_count
            ).tolist()
            mse_errors = (
                gated_scale_mse_error_sum_ps[p] / selected_count
            ).tolist()
            mae_errors = (
                gated_scale_mae_error_sum_ps[p] / selected_count
            ).tolist()

            def _gain_pct(base_value: float, error_value: float) -> float:
                return float(
                    100.0
                    * (base_value - float(error_value))
                    / max(abs(base_value), 1.0e-12)
                )

            named_gains = [
                _gain_pct(named_base, error) for error in named_errors
            ]
            mse_gains = [_gain_pct(mse_base, error) for error in mse_errors]
            mae_gains = [_gain_pct(mae_base, error) for error in mae_errors]
            joint_safe_indices = [
                idx
                for idx, scale in enumerate(specialization_scale_sweep)
                if abs(float(scale)) > 1.0e-15
                and named_gains[idx] > 0.0
                and mse_gains[idx] > 0.0
                and mae_gains[idx] > 0.0
            ]
            if joint_safe_indices:
                best_idx = max(
                    joint_safe_indices,
                    key=lambda idx: min(
                        named_gains[idx],
                        mse_gains[idx],
                        mae_gains[idx],
                    ),
                )
            else:
                best_idx = min(
                    range(len(specialization_scale_sweep)),
                    key=lambda idx: abs(
                        float(specialization_scale_sweep[idx])
                    ),
                )
            gated_scale_rows.append(
                {
                    "expert": str(name),
                    "selected_count": int(selected_count),
                    "scales": list(specialization_scale_sweep),
                    "named_penalty_improvement_pct_by_scale": named_gains,
                    "mse_improvement_pct_by_scale": mse_gains,
                    "mae_improvement_pct_by_scale": mae_gains,
                    "joint_safe_scales": [
                        float(specialization_scale_sweep[idx])
                        for idx in joint_safe_indices
                    ],
                    "joint_safe_nonzero_scale_exists": bool(
                        joint_safe_indices
                    ),
                    "best_joint_scale": float(
                        specialization_scale_sweep[best_idx]
                    ),
                    "best_joint_named_penalty_improvement_pct": float(
                        named_gains[best_idx]
                    ),
                    "best_joint_mse_improvement_pct": float(
                        mse_gains[best_idx]
                    ),
                    "best_joint_mae_improvement_pct": float(
                        mae_gains[best_idx]
                    ),
                    "best_joint_min_improvement_pct": float(
                        min(
                            named_gains[best_idx],
                            mse_gains[best_idx],
                            mae_gains[best_idx],
                        )
                    ),
                }
            )
        gated_adapter_scale_sweep = {
            "selection_split": str(split_name),
            "test_selected": str(split_name).lower() == "test",
            "route_definition": (
                "hold the deployed gate decisions fixed and evaluate "
                "base + scale * selected_adapter_correction"
            ),
            "rows": gated_scale_rows,
            "all_activated_experts_have_joint_safe_nonzero_scale": bool(
                gated_scale_rows
                and all(
                    bool(row["joint_safe_nonzero_scale_exists"])
                    for row in gated_scale_rows
                )
            ),
        }
    route_scale_calibration = None
    if route_scale_element_count > 0:
        count = float(route_scale_element_count)
        cross_mean = float(route_scale_cross_sum / count)
        correction_sq_mean = float(route_scale_delta_sq_sum / count)
        base_mse_for_scale = float(base_mse)
        if correction_sq_mean > 1.0e-20:
            optimal_scale = max(0.0, -cross_mean / correction_sq_mean)
        else:
            optimal_scale = 0.0
        optimal_mse = float(
            base_mse_for_scale
            + 2.0 * optimal_scale * cross_mean
            + optimal_scale * optimal_scale * correction_sq_mean
        )
        route_scale_calibration = {
            "selection_split": str(split_name),
            "test_selected": str(split_name).lower() == "test",
            "definition": "base + nonnegative_global_scale * deployed_gate_correction",
            "element_count": int(route_scale_element_count),
            "base_mse": base_mse_for_scale,
            "cross_mean": cross_mean,
            "correction_sq_mean": correction_sq_mean,
            "optimal_scale": float(optimal_scale),
            "optimal_mse": optimal_mse,
            "optimal_gain_pct_vs_base": float(
                100.0
                * (base_mse_for_scale - optimal_mse)
                / max(abs(base_mse_for_scale), 1.0e-12)
            ),
        }
    route_component_scale_calibration = None
    if route_component_element_count > 0:
        count = float(route_component_element_count)
        gram = route_component_gram_pp / count
        cross = route_component_cross_p / count
        best_scale = torch.zeros(P, dtype=torch.float64)
        best_delta = 0.0
        for mask_value in range(1, 1 << P):
            active = [p for p in range(P) if mask_value & (1 << p)]
            gram_active = gram[active][:, active]
            cross_active = cross[active]
            ridge = 1.0e-12 * torch.eye(
                len(active),
                dtype=torch.float64,
            )
            try:
                scale_active = torch.linalg.solve(
                    gram_active + ridge,
                    -cross_active,
                )
            except RuntimeError:
                continue
            if bool((scale_active < 0.0).any().item()):
                continue
            candidate_scale = torch.zeros(P, dtype=torch.float64)
            candidate_scale[active] = scale_active
            candidate_delta = float(
                (
                    2.0 * torch.dot(cross, candidate_scale)
                    + torch.dot(candidate_scale, gram @ candidate_scale)
                ).item()
            )
            if candidate_delta < best_delta:
                best_delta = candidate_delta
                best_scale = candidate_scale
        component_optimal_mse = float(base_mse + best_delta)
        route_component_scale_calibration = {
            "selection_split": str(split_name),
            "test_selected": str(split_name).lower() == "test",
            "definition": "base + sum_p nonnegative_scale_p * deployed_gate_component_p",
            "penalty_names": [str(name) for name in penalty_names],
            "element_count": int(route_component_element_count),
            "reconstruction_max_abs": float(
                route_component_reconstruction_max_abs
            ),
            "gram_mean": gram.tolist(),
            "cross_mean": cross.tolist(),
            "optimal_scale": best_scale.tolist(),
            "optimal_mse": component_optimal_mse,
            "optimal_gain_pct_vs_base": float(
                100.0
                * (base_mse - component_optimal_mse)
                / max(abs(base_mse), 1.0e-12)
            ),
        }
    route_channel_patch_scale_calibration = None
    if (
        route_channel_patch_cross_cs is not None
        and route_channel_patch_delta_sq_cs is not None
        and route_channel_patch_element_count > 0
    ):
        valid_cs = route_channel_patch_delta_sq_cs > 1.0e-20
        optimal_scale_cs = torch.zeros_like(route_channel_patch_cross_cs)
        optimal_scale_cs[valid_cs] = (
            -route_channel_patch_cross_cs[valid_cs]
            / route_channel_patch_delta_sq_cs[valid_cs]
        ).clamp_min(0.0)
        optimal_delta_sse = float(
            (
                2.0 * route_channel_patch_cross_cs * optimal_scale_cs
                + route_channel_patch_delta_sq_cs * optimal_scale_cs.square()
            ).sum().item()
        )
        channel_patch_optimal_mse = float(
            base_mse
            + optimal_delta_sse / float(route_channel_patch_element_count)
        )
        route_channel_patch_scale_calibration = {
            "selection_split": str(split_name),
            "test_selected": str(split_name).lower() == "test",
            "definition": (
                "base + nonnegative_scale_channel_patch * "
                "deployed_gate_correction"
            ),
            "patch_len": int(projection_patch_len),
            "channel_count": int(optimal_scale_cs.shape[0]),
            "horizon_segments": int(optimal_scale_cs.shape[1]),
            "element_count": int(route_channel_patch_element_count),
            "optimal_scale_cs": optimal_scale_cs.tolist(),
            "optimal_mse": channel_patch_optimal_mse,
            "optimal_gain_pct_vs_base": float(
                100.0
                * (base_mse - channel_patch_optimal_mse)
                / max(abs(base_mse), 1.0e-12)
            ),
        }
    previous_cycle_error_postprocess = None
    previous_cycle_total_count = float(previous_cycle_element_count_c.sum().item())
    if previous_cycle_total_count > 0.0:
        valid_c = previous_cycle_delta_sq_c > 1.0e-20
        optimal_scale_c = torch.zeros_like(previous_cycle_cross_c)
        optimal_scale_c[valid_c] = (
            -previous_cycle_cross_c[valid_c]
            / previous_cycle_delta_sq_c[valid_c]
        ).clamp_min(0.0)
        base_subset_mse = float(
            previous_cycle_base_sse_c.sum().item() / previous_cycle_total_count
        )
        cross_mean_c = previous_cycle_cross_c / previous_cycle_element_count_c.clamp_min(1.0)
        delta_sq_mean_c = previous_cycle_delta_sq_c / previous_cycle_element_count_c.clamp_min(1.0)
        unit_mse = float(
            (
                previous_cycle_base_sse_c
                + 2.0 * previous_cycle_cross_c
                + previous_cycle_delta_sq_c
            ).sum().item()
            / previous_cycle_total_count
        )
        optimal_delta_sse = (
            2.0 * previous_cycle_cross_c * optimal_scale_c
            + previous_cycle_delta_sq_c * optimal_scale_c.square()
        ).sum()
        optimal_mse = float(
            base_subset_mse
            + optimal_delta_sse.item() / previous_cycle_total_count
        )
        previous_cycle_error_postprocess = {
            "selection_split": str(split_name),
            "test_selected": str(split_name).lower() == "test",
            "definition": (
                "at forecast origin t, add the error of the forecast issued at "
                "t-input_len after its full target has matured"
            ),
            "lag": int(input_len),
            "samples_with_matured_feedback": int(
                previous_cycle_total_count
                / max(1, int(cid_cpu.numel()) * int(input_len))
            ),
            "base_subset_mse": base_subset_mse,
            "unit_scale_mse": unit_mse,
            "unit_scale_gain_pct": float(
                100.0 * (base_subset_mse - unit_mse)
                / max(abs(base_subset_mse), 1.0e-12)
            ),
            "cross_mean_c": cross_mean_c.tolist(),
            "correction_sq_mean_c": delta_sq_mean_c.tolist(),
            "base_mse_c": (
                previous_cycle_base_sse_c
                / previous_cycle_element_count_c.clamp_min(1.0)
            ).tolist(),
            "optimal_nonnegative_scale_c": optimal_scale_c.tolist(),
            "optimal_mse": optimal_mse,
            "optimal_gain_pct": float(
                100.0 * (base_subset_mse - optimal_mse)
                / max(abs(base_subset_mse), 1.0e-12)
            ),
        }
    rolling_position_residual_postprocess = None
    rolling_position_rows = []
    for lookback in rolling_position_lookbacks:
        stats = rolling_position_stats[lookback]
        total_count = float(stats["count_c"].sum().item())
        if total_count <= 0.0:
            continue
        base_mse_rolling = float(stats["base_sse_c"].sum().item() / total_count)
        unit_mse_rolling = float(
            (
                stats["base_sse_c"]
                + 2.0 * stats["cross_c"]
                + stats["delta_sq_c"]
            ).sum().item()
            / total_count
        )
        valid_c = stats["delta_sq_c"] > 1.0e-20
        scale_c = torch.zeros_like(stats["cross_c"])
        scale_c[valid_c] = (
            -stats["cross_c"][valid_c] / stats["delta_sq_c"][valid_c]
        ).clamp_min(0.0)
        optimal_delta_sse = (
            2.0 * stats["cross_c"] * scale_c
            + stats["delta_sq_c"] * scale_c.square()
        ).sum()
        optimal_mse_rolling = float(
            base_mse_rolling + optimal_delta_sse.item() / total_count
        )
        rolling_position_rows.append(
            {
                "lookback_matured_origins": int(lookback),
                "evaluated_elements": int(total_count),
                "base_mse": base_mse_rolling,
                "unit_scale_mse": unit_mse_rolling,
                "unit_scale_gain_pct": float(
                    100.0 * (base_mse_rolling - unit_mse_rolling)
                    / max(abs(base_mse_rolling), 1.0e-12)
                ),
                "cross_mean_c": (
                    stats["cross_c"] / stats["count_c"].clamp_min(1.0)
                ).tolist(),
                "correction_sq_mean_c": (
                    stats["delta_sq_c"] / stats["count_c"].clamp_min(1.0)
                ).tolist(),
                "optimal_nonnegative_scale_c": scale_c.tolist(),
                "optimal_mse": optimal_mse_rolling,
                "optimal_gain_pct": float(
                    100.0 * (base_mse_rolling - optimal_mse_rolling)
                    / max(abs(base_mse_rolling), 1.0e-12)
                ),
            }
        )
    if rolling_position_rows:
        rolling_position_residual_postprocess = {
            "split": str(split_name),
            "definition": (
                "rolling mean of fully matured previous forecast errors, applied "
                "at matching channel and horizon positions"
            ),
            "rows": rolling_position_rows,
        }
    position_residual_calibration = None
    if position_residual_sum_ch is not None and position_residual_sample_count > 0:
        mean_residual_ch = position_residual_sum_ch / float(
            position_residual_sample_count
        )
        own_optimal_mse = float(
            final_mse - mean_residual_ch.square().mean().item()
        )
        position_residual_calibration = {
            "selection_split": str(split_name),
            "test_selected": str(split_name).lower() == "test",
            "definition": (
                "fixed channel-by-forecast-horizon residual table fitted from "
                "the selected split"
            ),
            "sample_count": int(position_residual_sample_count),
            "channel_count": int(mean_residual_ch.shape[0]),
            "horizon": int(mean_residual_ch.shape[1]),
            "mean_residual_ch": mean_residual_ch.tolist(),
            "selected_path_mse": float(final_mse),
            "own_split_optimal_mse": own_optimal_mse,
            "own_split_incremental_gain_pct": float(
                100.0 * (final_mse - own_optimal_mse)
                / max(abs(final_mse), 1.0e-12)
            ),
            "own_split_gain_pct_vs_base": float(
                100.0 * (base_mse - own_optimal_mse)
                / max(abs(base_mse), 1.0e-12)
            ),
        }
    position_input_ridge_statistics = None
    if (
        position_input_ridge_xtx_cdd is not None
        and position_input_ridge_xtr_cdh is not None
        and position_input_ridge_sample_count > 0
    ):
        position_input_ridge_statistics = {
            "split": str(split_name),
            "definition": (
                "per-channel multi-output residual ridge using p12 input means, "
                "input endpoint changes, selected-forecast means, input log-std, "
                "and forecast-origin daily/weekly Fourier position"
            ),
            "sample_count": int(position_input_ridge_sample_count),
            "patch_len": int(projection_patch_len),
            "feature_dim": int(position_input_ridge_xtx_cdd.shape[1]),
            "horizon": int(position_input_ridge_xtr_cdh.shape[2]),
            "selected_path_mse": float(final_mse),
            "xtx_cdd": position_input_ridge_xtx_cdd.tolist(),
            "xtr_cdh": position_input_ridge_xtr_cdh.tolist(),
            "temporal_blocks": [
                {
                    "block": int(block_i),
                    "sample_count": int(block_stats["sample_count"]),
                    "xtx_cdd": block_stats["xtx_cdd"].tolist(),
                    "xtr_cdh": block_stats["xtr_cdh"].tolist(),
                }
                for block_i, block_stats in enumerate(
                    position_input_ridge_block_stats or []
                )
            ],
        }
    return {
        "split": split_name,
        "samples": int(total_decisions),
        "selected_penalty_events": int(total_selected),
        "oracle_positive_events": int(total_oracle_positive),
        "selected_positive_events": int(total_selected_positive),
        "base_mse": base_mse,
        "final_mse": final_mse,
        "final_gain_pct_vs_base": float(100.0 * (base_mse - final_mse) / max(abs(base_mse), 1.0e-12)),
        "oracle_mse": oracle_mse,
        "oracle_gain_pct_vs_base": float(100.0 * (base_mse - oracle_mse) / max(abs(base_mse), 1.0e-12)),
        "cluster_penalty_oracle_mse": cluster_penalty_oracle_mse,
        "cluster_penalty_oracle_gain_pct_vs_base": float(
            100.0 * (base_mse - cluster_penalty_oracle_mse) / max(abs(base_mse), 1.0e-12)
        ),
        "cluster_route_oracle_mse": cluster_route_oracle_mse,
        "cluster_route_oracle_gain_pct_vs_base": float(
            100.0 * (base_mse - cluster_route_oracle_mse) / max(abs(base_mse), 1.0e-12)
        ),
        "cluster_route_oracle_skip_rate": cluster_route_oracle_skip_rate,
        "routing_diagnostic_sources": {
            "legacy_fields": {
                "routing_granularity": "sample_channel_full_horizon",
                "skip_source": "outer_cluster_gate_skip_bk",
                "selected_penalty_source": "outer_cluster_gate_mask_and_probs",
                "replaced_by_patch_router": bool(patch_router_seen),
            },
            "patch_router_fields": (
                "patch_router_route_diagnostics"
                if patch_router_seen
                else None
            ),
        },
        "patch_router_route_diagnostics": patch_router_route_diagnostics,
        "periodic_action_space_oracle": periodic_action_space_oracle,
        "prior_actual_gain_corr": penalty_corr,
        "utility_target_thresholds": utility_threshold_summary,
        "route_label_feature_diagnostics": route_label_feature_diagnostics,
        "route_label_phase_diagnostics": route_label_phase_diagnostics,
        "top1_confidence_gain_diagnostics": top1_confidence_gain_diagnostics,
        "named_attribute_specialization": named_attribute_specialization,
        "named_penalty_specialization": named_penalty_specialization,
        "patch_named_penalty_specialization": patch_named_penalty_specialization,
        "adapter_scale_sweep": adapter_scale_sweep,
        "gated_adapter_scale_sweep": gated_adapter_scale_sweep,
        "route_scale_calibration": route_scale_calibration,
        "route_component_scale_calibration": route_component_scale_calibration,
        "route_channel_patch_scale_calibration": route_channel_patch_scale_calibration,
        "previous_cycle_error_postprocess": previous_cycle_error_postprocess,
        "rolling_position_residual_postprocess": rolling_position_residual_postprocess,
        "position_residual_calibration": position_residual_calibration,
        "position_input_ridge_statistics": position_input_ridge_statistics,
        "per_cluster": per_cluster,
        "rows": rows,
    }


def save_penalty_explainability_artifacts(
    out_dir: str,
    explainability: Dict[str, object],
) -> Dict[str, str]:
    paths = {}
    json_path = os.path.join(out_dir, "penalty_explainability.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(explainability, f, ensure_ascii=False, indent=2)
    paths["json"] = json_path
    rows = []
    for split_payload in explainability.get("splits", {}).values():
        if isinstance(split_payload, dict):
            rows.extend(split_payload.get("rows", []))
    if len(rows) > 0:
        csv_path = os.path.join(out_dir, "penalty_explainability.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
        paths["csv"] = csv_path
    gradient_payload = explainability.get("adapter_gradient_isolation")
    gradient_rows: List[Dict[str, object]] = []
    if isinstance(gradient_payload, dict):
        loss_names = list(gradient_payload.get("loss_names", []))
        expert_names = list(gradient_payload.get("expert_names", []))
        body_matrix = gradient_payload.get("body_gradient_l2", [])
        auxiliary_matrix = gradient_payload.get("auxiliary_gradient_l2", [])
        for row_idx, loss_name in enumerate(loss_names):
            for expert_idx, expert_name in enumerate(expert_names):
                gradient_rows.append(
                    {
                        "split": gradient_payload.get("split"),
                        "loss": loss_name,
                        "expert": expert_name,
                        "is_matching": str(loss_name) == str(expert_name),
                        "body_gradient_l2": float(body_matrix[row_idx][expert_idx]),
                        "auxiliary_gradient_l2": float(
                            auxiliary_matrix[row_idx][expert_idx]
                        ),
                    }
                )
        gradient_csv_path = os.path.join(
            out_dir,
            "adapter_gradient_isolation.csv",
        )
        pd.DataFrame(gradient_rows).to_csv(
            gradient_csv_path,
            index=False,
            encoding="utf-8-sig",
        )
        paths["gradient_isolation_csv"] = gradient_csv_path

    specialization_rows: List[Dict[str, object]] = []
    for split_name, split_payload in explainability.get("splits", {}).items():
        if not isinstance(split_payload, dict):
            continue
        for semantic_kind, semantic_key in (
            ("exact_penalty_full_horizon", "named_penalty_specialization"),
            ("exact_penalty_projection_patch", "patch_named_penalty_specialization"),
            ("direct_attribute_surrogate", "named_attribute_specialization"),
        ):
            semantic = split_payload.get(semantic_key)
            if not isinstance(semantic, dict):
                continue
            expert_names = list(semantic.get("expert_names", []))
            metric_names = list(semantic.get("metric_names", []))
            base_error = list(semantic.get("base_error_mean_by_metric", []))
            candidate_error = semantic.get("candidate_error_mean_expert_by_metric", [])
            signed_gain = semantic.get("mean_signed_improvement_expert_by_metric", [])
            gain_pct = semantic.get("relative_improvement_pct_expert_by_metric", [])
            positive_rate = semantic.get(
                "positive_sample_channel_rate_expert_by_metric",
                [],
            )
            for expert_idx, expert_name in enumerate(expert_names):
                for metric_idx, metric_name in enumerate(metric_names):
                    specialization_rows.append(
                        {
                            "split": split_name,
                            "semantic_kind": semantic_kind,
                            "aggregation_unit": semantic.get("aggregation_unit"),
                            "expert": expert_name,
                            "metric": metric_name,
                            "is_matching": str(expert_name) == str(metric_name),
                            "base_error": float(base_error[metric_idx]),
                            "candidate_error": float(
                                candidate_error[expert_idx][metric_idx]
                            ),
                            "mean_signed_improvement": float(
                                signed_gain[expert_idx][metric_idx]
                            ),
                            "relative_improvement_pct": float(
                                gain_pct[expert_idx][metric_idx]
                            ),
                            "positive_sample_channel_rate": float(
                                positive_rate[expert_idx][metric_idx]
                            ),
                        }
                    )
    if specialization_rows:
        specialization_csv_path = os.path.join(
            out_dir,
            "adapter_named_specialization.csv",
        )
        pd.DataFrame(specialization_rows).to_csv(
            specialization_csv_path,
            index=False,
            encoding="utf-8-sig",
        )
        paths["named_specialization_csv"] = specialization_csv_path

    gated_specialization_rows: List[Dict[str, object]] = []
    for split_name, split_payload in explainability.get("splits", {}).items():
        if not isinstance(split_payload, dict):
            continue
        patch_payload = split_payload.get("patch_router_route_diagnostics")
        if not isinstance(patch_payload, dict):
            continue
        aggregate = patch_payload.get("aggregate")
        if not isinstance(aggregate, dict):
            continue
        for row in aggregate.get("per_penalty", []):
            gated_specialization_rows.append(
                {
                    "split": split_name,
                    **row,
                }
            )
    if gated_specialization_rows:
        gated_specialization_csv_path = os.path.join(
            out_dir,
            "adapter_gated_specialization.csv",
        )
        pd.DataFrame(gated_specialization_rows).to_csv(
            gated_specialization_csv_path,
            index=False,
            encoding="utf-8-sig",
        )
        paths["gated_specialization_csv"] = gated_specialization_csv_path

    scale_sweep_rows: List[Dict[str, object]] = []
    for split_name, split_payload in explainability.get("splits", {}).items():
        if not isinstance(split_payload, dict):
            continue
        scale_payload = split_payload.get("adapter_scale_sweep")
        if not isinstance(scale_payload, dict):
            continue
        for row in scale_payload.get("rows", []):
            for scale, error, gain_pct in zip(
                row.get("scales", []),
                row.get("error_by_scale", []),
                row.get("improvement_pct_by_scale", []),
            ):
                scale_sweep_rows.append(
                    {
                        "split": split_name,
                        "expert": row.get("expert"),
                        "metric": row.get("metric"),
                        "scale": float(scale),
                        "error": float(error),
                        "improvement_pct": float(gain_pct),
                        "is_best": float(scale) == float(row.get("best_scale", -1.0)),
                    }
                )
    if scale_sweep_rows:
        scale_sweep_csv_path = os.path.join(
            out_dir,
            "adapter_scale_sweep.csv",
        )
        pd.DataFrame(scale_sweep_rows).to_csv(
            scale_sweep_csv_path,
            index=False,
            encoding="utf-8-sig",
        )
        paths["adapter_scale_sweep_csv"] = scale_sweep_csv_path

    if gradient_rows or specialization_rows or gated_specialization_rows:
        proof_split_names = [
            str(name).lower()
            for name in explainability.get("splits", {}).keys()
        ]
        semantic_test_used = "test" in proof_split_names
        report_lines = [
            "# PKR adapter proof experiments",
            "",
            "The fixed baseline is `backbone + periodic expert`.",
            (
                "Semantic selection includes **test** and is therefore explicitly test-selected, not an unbiased generalization estimate."
                if semantic_test_used
                else "Test is not used by these proof splits."
            ),
            "",
            "## 1. Gradient isolation",
            "",
        ]
        if isinstance(gradient_payload, dict):
            report_lines.extend(
                [
                    f"- Split: `{gradient_payload.get('split')}`; batches: {gradient_payload.get('batches')}; "
                    f"test used: `{gradient_payload.get('test_split_used')}`.",
                    f"- Minimum matching body gradient: `{float(gradient_payload.get('diagonal_min', 0.0)):.6e}`.",
                    f"- Maximum non-matching body gradient: `{float(gradient_payload.get('off_diagonal_max', 0.0)):.6e}`.",
                    f"- Verdict: **{'PASS' if bool(gradient_payload.get('passed', False)) else 'FAIL'}**.",
                    "",
                ]
            )
            expert_names = list(gradient_payload.get("expert_names", []))
            report_lines.append(
                "| named loss \\ adapter | " + " | ".join(expert_names) + " |"
            )
            report_lines.append(
                "|---|" + "---:|" * len(expert_names)
            )
            for loss_name, values in zip(
                gradient_payload.get("loss_names", []),
                gradient_payload.get("body_gradient_l2", []),
            ):
                report_lines.append(
                    f"| {loss_name} | "
                    + " | ".join(f"{float(value):.6e}" for value in values)
                    + " |"
                )
        else:
            report_lines.extend(["Gradient isolation was not requested.", ""])
        semantic_sections = (
            (
                "## 2. Exact configured penalty specialization (primary)",
                "named_penalty_specialization",
            ),
            (
                "## 3. Exact penalty specialization at the projection-patch granularity",
                "patch_named_penalty_specialization",
            ),
            (
                "## 4. Direct future-attribute surrogate specialization",
                "named_attribute_specialization",
            ),
        )
        for section_title, semantic_key in semantic_sections:
            section_started = False
            for split_name, split_payload in explainability.get("splits", {}).items():
                if not isinstance(split_payload, dict):
                    continue
                semantic = split_payload.get(semantic_key)
                if not isinstance(semantic, dict):
                    continue
                if not section_started:
                    report_lines.extend(
                        [
                            "",
                            section_title,
                            "",
                            "Each cell is the relative error reduction (%) after applying only the row adapter. Positive is better.",
                            "",
                        ]
                    )
                    section_started = True
                expert_names = list(semantic.get("expert_names", []))
                metric_names = list(semantic.get("metric_names", []))
                matrix = semantic.get("relative_improvement_pct_expert_by_metric", [])
                report_lines.extend(
                    [
                        f"### {split_name}",
                        "",
                        f"Aggregation: `{semantic.get('aggregation_unit')}`. Sample units: {semantic.get('sample_channels')}; "
                        f"all matching directions improved: `{semantic.get('all_diagonal_improved')}`.",
                        "",
                        "| expert \\ metric | " + " | ".join(metric_names) + " |",
                        "|---|" + "---:|" * len(metric_names),
                    ]
                )
                for expert_name, values in zip(expert_names, matrix):
                    report_lines.append(
                        f"| {expert_name} | "
                        + " | ".join(f"{float(value):+.4f}" for value in values)
                        + " |"
                    )
                report_lines.extend(["", "Matching-direction detail:", ""])
                report_lines.append(
                    "| expert | error reduction (%) | positive unit rate | rank | improved |"
                )
                report_lines.append("|---|---:|---:|---:|---|")
                for diagonal in semantic.get("diagonal", []):
                    report_lines.append(
                        f"| {diagonal['expert']} | {float(diagonal['relative_improvement_pct']):+.4f} | "
                        f"{100.0 * float(diagonal['positive_sample_channel_rate']):.2f}% | "
                        f"{int(diagonal['rank_by_mean_improvement'])} | {diagonal['improved']} |"
                    )
                report_lines.append("")
        gate_section_started = False
        for split_name, split_payload in explainability.get("splits", {}).items():
            if not isinstance(split_payload, dict):
                continue
            patch_payload = split_payload.get(
                "patch_router_route_diagnostics"
            )
            if not isinstance(patch_payload, dict):
                continue
            aggregate = patch_payload.get("aggregate")
            if not isinstance(aggregate, dict):
                continue
            if not gate_section_started:
                report_lines.extend(
                    [
                        "",
                        "## 5. Gate-conditioned specialization (deployment proof)",
                        "",
                        "Experts are not required to improve when applied globally. This section evaluates only patches actually activated by the deployed gate, against the periodic-only no-op.",
                        "",
                    ]
                )
                gate_section_started = True
            proof = aggregate.get("conditional_specialization_proof", {})
            report_lines.extend(
                [
                    f"### {split_name}",
                    "",
                    f"Activated experts: `{proof.get('activated_expert_count')}`; available experts: `{proof.get('available_expert_count')}`; "
                    f"all activated experts pass: `{proof.get('all_activated_experts_pass')}`.",
                    "",
                    "| expert | activated rate | own-penalty available | joint-effective available | own-penalty reduction on selected | selected MSE gain | selected MAE gain | joint-effective precision | pass |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---|",
                ]
            )
            for row in aggregate.get("per_penalty", []):
                report_lines.append(
                    f"| {row['penalty']} | "
                    f"{100.0 * float(row['selected_rate']):.2f}% | "
                    f"{100.0 * float(row['named_penalty_available_rate']):.2f}% | "
                    f"{100.0 * float(row['joint_effective_available_rate']):.2f}% | "
                    f"{float(row['selected_named_penalty_reduction_pct']):+.4f}% | "
                    f"{float(row['selected_mean_mse_gain']):+.6e} | "
                    f"{float(row['selected_mean_mae_gain']):+.6e} | "
                    f"{100.0 * float(row['selected_joint_effective_precision']):.2f}% | "
                    f"{row['selected_region_proof_pass']} |"
                )
            report_lines.append("")
        scale_section_started = False
        for split_name, split_payload in explainability.get("splits", {}).items():
            if not isinstance(split_payload, dict):
                continue
            scale_payload = split_payload.get("adapter_scale_sweep")
            if not isinstance(scale_payload, dict):
                continue
            if not scale_section_started:
                report_lines.extend(
                    [
                        "",
                        "## 6. Adapter correction scale sweep",
                        "",
                        "The sweep evaluates `base + scale * correction` against each expert's exact configured penalty.",
                        "",
                    ]
                )
                scale_section_started = True
            report_lines.extend(
                [
                    f"### {split_name}",
                    "",
                    f"Test-selected: `{scale_payload.get('test_selected')}`; all experts have a beneficial positive scale: "
                    f"`{scale_payload.get('all_experts_have_positive_nonzero_scale')}`.",
                    "",
                    "| expert | best scale | sign | best penalty reduction (%) | positive improves | negative improves |",
                    "|---|---:|---|---:|---|---|",
                ]
            )
            for row in scale_payload.get("rows", []):
                report_lines.append(
                    f"| {row['expert']} | {float(row['best_scale']):.4f} | "
                    f"{row.get('best_scale_sign', 'unknown')} | "
                    f"{float(row['best_improvement_pct']):+.4f} | "
                    f"{row['positive_nonzero_scale_exists']} | "
                    f"{row.get('negative_scale_improves', False)} |"
                )
            report_lines.append("")
        gated_scale_section_started = False
        for split_name, split_payload in explainability.get("splits", {}).items():
            if not isinstance(split_payload, dict):
                continue
            scale_payload = split_payload.get("gated_adapter_scale_sweep")
            if not isinstance(scale_payload, dict):
                continue
            if not gated_scale_section_started:
                report_lines.extend(
                    [
                        "",
                        "## 7. Fixed-route expert scale search",
                        "",
                        "The deployed gate decisions are held fixed. Each row searches a positive multiplier for only that activated adapter and requires simultaneous improvement in its matching penalty, MSE, and MAE versus periodic-only.",
                        "",
                    ]
                )
                gated_scale_section_started = True
            report_lines.extend(
                [
                    f"### {split_name}",
                    "",
                    f"Test-selected: `{scale_payload.get('test_selected')}`; all activated experts have a jointly safe positive scale: "
                    f"`{scale_payload.get('all_activated_experts_have_joint_safe_nonzero_scale')}`.",
                    "",
                    "| expert | selected patches | best scale | matching-penalty reduction (%) | MSE reduction (%) | MAE reduction (%) |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            for row in scale_payload.get("rows", []):
                report_lines.append(
                    f"| {row['expert']} | {int(row['selected_count'])} | "
                    f"{float(row['best_joint_scale']):.4f} | "
                    f"{float(row['best_joint_named_penalty_improvement_pct']):+.6f} | "
                    f"{float(row['best_joint_mse_improvement_pct']):+.6f} | "
                    f"{float(row['best_joint_mae_improvement_pct']):+.6f} |"
                )
            report_lines.append("")
        report_path = os.path.join(out_dir, "pkr_adapter_proof_experiments.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines).rstrip() + "\n")
        paths["proof_report"] = report_path
    return paths


__all__ = [
    'eval_loop',
    '_calendar_features_from_datetime',
    'build_calendar_feature_tensor',
    'calendar_feature_batch',
    'apply_calendar_residual_correction',
    'position_daily_feature_batch',
    'apply_position_daily_residual_ridge',
    'fit_position_daily_residual_ridge_from_prediction_parts',
    'fit_anchor_ridge_gate_from_prediction_parts',
    '_solve_calendar_residual_coefficients',
    '_fit_calendar_residual_from_prediction_parts',
    'fit_calendar_residual_correction',
    'fit_calendar_residual_correction_from_eval_path',
    'evaluate_gate_penalty_hit_metrics',
    '_pearson_list',
    '_explainability_train_subsplit_ranges',
    '_cluster_route_label_feature_diagnostics',
    '_cluster_route_label_phase_diagnostics',
    '_cluster_top1_confidence_gain_diagnostics',
    '_build_penalty_route_learnability_class_features',
    '_scatter_mean_bcpf_to_bkpf',
    '_collect_penalty_route_learnability_tensors',
    '_penalty_route_learnability_metrics_from_scores',
    '_PenaltyRouteLearnabilityHead',
    '_fit_penalty_route_learnability_head_from_tensors',
    'evaluate_adapter_gradient_isolation',
    'evaluate_penalty_explainability',
    'save_penalty_explainability_artifacts',
]
