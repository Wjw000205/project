"""Prediction-residual candidates, patch routers, and candidate selectors."""
from __future__ import annotations

import math
import time
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..models.learnable_anchor import ClusterwiseLearnableOutputAnchor
from ..models.moe_gate import scatter_mean_bc_to_bk, scatter_mean_bcf_to_bkf
from ..models.residual_moe import ClusterwisePredResidualMoE
from .anchors import (
    apply_history_anchor_adapter,
    apply_moe_output_anchor_experts,
    apply_train_stat_anchor_expert,
    apply_train_stat_input_centering,
    build_moe_output_anchor_fixed_expert_delta,
)
from .core import (
    _contiguous_segment_ranges,
    _normalize_gate_feature_mode,
    _normalize_pred_residual_candidate_selection_metric,
    reduce_cluster_metric,
)


CANDIDATE_DELTA_FEATURE_NAMES = [
    "hist_mean",
    "log_hist_std",
    "hist_last",
    "hist_range",
    "hist_slope",
    "delta_abs_mean",
    "delta_abs_max",
    "delta_std",
    "delta_bias",
    "base_std",
    "hybrid_std",
    "base_shift",
    "hybrid_shift",
]


HISTORY_PROXY_SELECTOR_FEATURE_NAMES = [
    "proxy_base_mse",
    "proxy_hybrid_mse",
    "proxy_mse_delta",
    "proxy_base_mae",
    "proxy_hybrid_mae",
    "proxy_mae_delta",
    "proxy_hybrid_bias",
]


SHAPE_PROXY_SELECTOR_FEATURE_NAMES = [
    "hist_diff_rms",
    "hist_d2_rms",
    "base_slope",
    "hybrid_slope",
    "base_hist_slope_gap",
    "hybrid_hist_slope_gap",
    "hybrid_base_slope_delta",
    "base_diff_rms",
    "hybrid_diff_rms",
    "hybrid_base_diff_rms_delta",
    "base_d2_rms",
    "hybrid_d2_rms",
    "hybrid_base_d2_rms_delta",
    "base_hist_corr",
    "hybrid_hist_corr",
    "hybrid_base_corr_delta",
    "base_hist_std_ratio",
    "hybrid_hist_std_ratio",
    "hybrid_base_std_ratio_delta",
]


def _candidate_selector_feature_names(feature_mode: str = "base") -> List[str]:
    mode = str(feature_mode or "base").lower()
    if mode in {"base", "default"}:
        return list(CANDIDATE_DELTA_FEATURE_NAMES)
    if mode in {"history_proxy", "proxy"}:
        return list(CANDIDATE_DELTA_FEATURE_NAMES) + list(HISTORY_PROXY_SELECTOR_FEATURE_NAMES)
    if mode in {"shape_proxy", "shape", "rich_shape"}:
        return list(CANDIDATE_DELTA_FEATURE_NAMES) + list(HISTORY_PROXY_SELECTOR_FEATURE_NAMES) + list(SHAPE_PROXY_SELECTOR_FEATURE_NAMES)
    raise ValueError("candidate_selector.feature_mode must be base, history_proxy, or shape_proxy.")


def _history_proxy_for_candidate_selector(x_bcl: torch.Tensor, pred_len: int) -> torch.Tensor:
    H = int(pred_len)
    if H <= 0:
        raise ValueError("candidate selector history proxy requires pred_len > 0.")
    L = int(x_bcl.shape[-1])
    if L >= H:
        return x_bcl[..., -H:]
    return x_bcl[..., -1:].expand(*x_bcl.shape[:-1], H)


def _sequence_slope_bch(values_bch: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    if values_bch.shape[-1] <= 1:
        return torch.zeros_like(values_bch[..., 0])
    t = torch.linspace(-1.0, 1.0, steps=values_bch.shape[-1], device=values_bch.device, dtype=values_bch.dtype)
    t = t.view(*([1] * (values_bch.dim() - 1)), -1)
    centered = values_bch - values_bch.mean(dim=-1, keepdim=True)
    return (centered * t).mean(dim=-1) / t.pow(2).mean().clamp_min(eps)


def _sequence_diff_rms_bch(values_bch: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    if values_bch.shape[-1] <= 1:
        return torch.zeros_like(values_bch[..., 0])
    return values_bch.diff(dim=-1).pow(2).mean(dim=-1).clamp_min(eps).sqrt()


def _sequence_d2_rms_bch(values_bch: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    if values_bch.shape[-1] <= 2:
        return torch.zeros_like(values_bch[..., 0])
    return values_bch.diff(dim=-1).diff(dim=-1).pow(2).mean(dim=-1).clamp_min(eps).sqrt()


def _sequence_corr_bch(a_bch: torch.Tensor, b_bch: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    if a_bch.shape[-1] <= 1:
        return torch.zeros_like(a_bch[..., 0])
    a_center = a_bch - a_bch.mean(dim=-1, keepdim=True)
    b_center = b_bch - b_bch.mean(dim=-1, keepdim=True)
    denom = a_center.pow(2).mean(dim=-1).sqrt() * b_center.pow(2).mean(dim=-1).sqrt()
    return (a_center * b_center).mean(dim=-1) / denom.clamp_min(eps)


def _candidate_selector_features(
    x_bcl: torch.Tensor,
    base_bch: torch.Tensor,
    hybrid_bch: torch.Tensor,
    feature_mode: str = "base",
) -> torch.Tensor:
    base_feat = _candidate_delta_features(x_bcl, base_bch, hybrid_bch)
    mode = str(feature_mode or "base").lower()
    if mode in {"base", "default"}:
        return base_feat
    if mode not in {"history_proxy", "proxy", "shape_proxy", "shape", "rich_shape"}:
        raise ValueError("candidate_selector.feature_mode must be base, history_proxy, or shape_proxy.")
    eps = 1.0e-6
    hist_std = x_bcl.std(dim=-1).clamp_min(eps)
    hist_proxy = _history_proxy_for_candidate_selector(x_bcl, int(base_bch.shape[-1])).to(
        device=base_bch.device,
        dtype=base_bch.dtype,
    )
    scale2 = hist_std.pow(2).clamp_min(eps).to(device=base_bch.device, dtype=base_bch.dtype)
    scale1 = hist_std.to(device=base_bch.device, dtype=base_bch.dtype)
    base_proxy_mse = (base_bch - hist_proxy).pow(2).mean(dim=-1) / scale2
    hybrid_proxy_mse = (hybrid_bch - hist_proxy).pow(2).mean(dim=-1) / scale2
    base_proxy_mae = (base_bch - hist_proxy).abs().mean(dim=-1) / scale1
    hybrid_proxy_mae = (hybrid_bch - hist_proxy).abs().mean(dim=-1) / scale1
    proxy_extra = torch.stack(
        [
            base_proxy_mse,
            hybrid_proxy_mse,
            hybrid_proxy_mse - base_proxy_mse,
            base_proxy_mae,
            hybrid_proxy_mae,
            hybrid_proxy_mae - base_proxy_mae,
            (hybrid_bch - hist_proxy).mean(dim=-1) / scale1,
        ],
        dim=-1,
    )
    if mode in {"history_proxy", "proxy"}:
        return torch.cat([base_feat, proxy_extra], dim=-1)

    hist_proxy = hist_proxy.to(device=base_bch.device, dtype=base_bch.dtype)
    hist_slope = _sequence_slope_bch(hist_proxy, eps=eps)
    base_slope = _sequence_slope_bch(base_bch, eps=eps)
    hybrid_slope = _sequence_slope_bch(hybrid_bch, eps=eps)
    hist_diff_rms = _sequence_diff_rms_bch(hist_proxy, eps=eps).to(dtype=base_bch.dtype) / scale1
    base_diff_rms = _sequence_diff_rms_bch(base_bch, eps=eps) / scale1
    hybrid_diff_rms = _sequence_diff_rms_bch(hybrid_bch, eps=eps) / scale1
    hist_d2_rms = _sequence_d2_rms_bch(hist_proxy, eps=eps).to(dtype=base_bch.dtype) / scale1
    base_d2_rms = _sequence_d2_rms_bch(base_bch, eps=eps) / scale1
    hybrid_d2_rms = _sequence_d2_rms_bch(hybrid_bch, eps=eps) / scale1
    base_corr = _sequence_corr_bch(base_bch, hist_proxy, eps=eps)
    hybrid_corr = _sequence_corr_bch(hybrid_bch, hist_proxy, eps=eps)
    hist_proxy_std = hist_proxy.std(dim=-1).clamp_min(eps)
    base_std_ratio = base_bch.std(dim=-1) / hist_proxy_std
    hybrid_std_ratio = hybrid_bch.std(dim=-1) / hist_proxy_std
    shape_extra = torch.stack(
        [
            hist_diff_rms,
            hist_d2_rms,
            base_slope,
            hybrid_slope,
            (base_slope - hist_slope) / scale1,
            (hybrid_slope - hist_slope) / scale1,
            (hybrid_slope - base_slope) / scale1,
            base_diff_rms,
            hybrid_diff_rms,
            hybrid_diff_rms - base_diff_rms,
            base_d2_rms,
            hybrid_d2_rms,
            hybrid_d2_rms - base_d2_rms,
            base_corr,
            hybrid_corr,
            hybrid_corr - base_corr,
            base_std_ratio,
            hybrid_std_ratio,
            hybrid_std_ratio - base_std_ratio,
        ],
        dim=-1,
    )
    return torch.cat([base_feat, proxy_extra, shape_extra], dim=-1)


def _candidate_selector_patch_views(
    x_bcl: torch.Tensor,
    base_bch: torch.Tensor,
    cand_bcpH: torch.Tensor,
    patch_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Flatten forecast patches into independent causal selector examples."""
    patch = int(patch_len)
    if patch <= 0:
        raise ValueError("candidate_selector.patch_len must be positive.")
    if x_bcl.ndim != 3 or base_bch.ndim != 3 or cand_bcpH.ndim != 4:
        raise ValueError("Patch candidate selector expects x/base/candidates with 3/3/4 dimensions.")
    B, C, H = base_bch.shape
    if int(cand_bcpH.shape[0]) != int(B) or int(cand_bcpH.shape[1]) != int(C):
        raise ValueError("Patch candidate selector candidate batch/channel dimensions must match base.")
    if int(cand_bcpH.shape[-1]) != int(H):
        raise ValueError("Patch candidate selector candidate horizon must match base.")
    if int(H) % patch != 0:
        raise ValueError("candidate_selector.patch_len must divide the prediction horizon exactly.")
    if int(x_bcl.shape[-1]) < int(H):
        raise ValueError("Patch candidate selector requires input history at least as long as the horizon.")
    Q = int(H) // patch
    P = int(cand_bcpH.shape[2])
    x_patch = x_bcl[..., -H:].reshape(B, C, Q, patch).permute(0, 2, 1, 3).reshape(B * Q, C, patch)
    base_patch = base_bch.reshape(B, C, Q, patch).permute(0, 2, 1, 3).reshape(B * Q, C, patch)
    cand_patch = (
        cand_bcpH.reshape(B, C, P, Q, patch)
        .permute(0, 3, 1, 2, 4)
        .reshape(B * Q, C, P, patch)
    )
    return x_patch, base_patch, cand_patch, Q


def _candidate_selector_patch_query_starts(
    query_start_abs_b: Optional[torch.Tensor],
    *,
    batch_size: int,
    num_patches: int,
    patch_len: int,
) -> Optional[torch.Tensor]:
    if query_start_abs_b is None:
        return None
    query = query_start_abs_b.reshape(-1)
    if int(query.numel()) != int(batch_size):
        raise ValueError(
            f"Patch candidate selector expected {int(batch_size)} query starts, got {int(query.numel())}."
        )
    offsets = torch.arange(
        int(num_patches),
        device=query.device,
        dtype=query.dtype,
    ) * int(patch_len)
    return (query.view(-1, 1) + offsets.view(1, -1)).reshape(-1)


def _candidate_delta_features(
    x_bcl: torch.Tensor,
    base_bch: torch.Tensor,
    hybrid_bch: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    hist_mean = x_bcl.mean(dim=-1)
    hist_std = x_bcl.std(dim=-1).clamp_min(eps)
    hist_last = x_bcl[..., -1]
    hist_range = (x_bcl.max(dim=-1).values - x_bcl.min(dim=-1).values) / hist_std
    t_l = torch.linspace(-1.0, 1.0, steps=x_bcl.shape[-1], device=x_bcl.device, dtype=x_bcl.dtype).view(1, 1, -1)
    hist_center = x_bcl - hist_mean.unsqueeze(-1)
    hist_slope = (hist_center * t_l).mean(dim=-1) / t_l.pow(2).mean().clamp_min(eps)

    delta_bch = hybrid_bch - base_bch
    delta_abs_mean = delta_bch.abs().mean(dim=-1) / hist_std
    delta_abs_max = delta_bch.abs().amax(dim=-1) / hist_std
    delta_std = delta_bch.std(dim=-1) / hist_std
    base_std = base_bch.std(dim=-1) / hist_std
    hybrid_std = hybrid_bch.std(dim=-1) / hist_std
    base_shift = (base_bch.mean(dim=-1) - hist_last) / hist_std
    hybrid_shift = (hybrid_bch.mean(dim=-1) - hist_last) / hist_std
    delta_bias = delta_bch.mean(dim=-1) / hist_std

    return torch.stack(
        [
            hist_mean,
            hist_std.log(),
            hist_last,
            hist_range,
            hist_slope,
            delta_abs_mean,
            delta_abs_max,
            delta_std,
            delta_bias,
            base_std,
            hybrid_std,
            base_shift,
            hybrid_shift,
        ],
        dim=-1,
    )


def _pred_residual_candidate_predictions(
    y_base_bch: torch.Tensor,
    pred_out: Dict[str, torch.Tensor],
    pred_residual_scale_c: Optional[torch.Tensor] = None,
    include_intervention: bool = True,
    include_selector: bool = True,
    include_patch_route: bool = True,
) -> Optional[torch.Tensor]:
    candidate_base_bch = pred_out.get("candidate_base_bch", y_base_bch)
    residuals = pred_out.get("residuals")
    alpha_cp = pred_out.get("alpha_cp")
    intervention_bcp = pred_out.get("intervention_bcp")
    selector_bcp = pred_out.get("selector_bcp")
    confidence_active_bcp = pred_out.get("confidence_active_bcp")
    patch_route_bcph = pred_out.get("patch_route_bcph")
    patch_candidate_scale_c = pred_out.get("patch_candidate_scale_c")
    if residuals is None or alpha_cp is None or intervention_bcp is None:
        return None
    if residuals.numel() == 0:
        return None
    if selector_bcp is None:
        selector_bcp = torch.ones_like(intervention_bcp)
    if confidence_active_bcp is None:
        confidence_active_bcp = torch.ones_like(intervention_bcp)
    if not bool(include_intervention):
        intervention_bcp = torch.ones_like(intervention_bcp)
        confidence_active_bcp = torch.ones_like(confidence_active_bcp)
    if not bool(include_selector):
        selector_bcp = torch.ones_like(selector_bcp)
    scale_bcp = intervention_bcp * selector_bcp * confidence_active_bcp * alpha_cp.unsqueeze(0)
    if patch_candidate_scale_c is not None and int(patch_candidate_scale_c.numel()) > 0:
        if int(patch_candidate_scale_c.numel()) != int(candidate_base_bch.shape[1]):
            raise ValueError(
                "patch candidate scale length must match the channel count."
            )
        scale_bcp = scale_bcp * patch_candidate_scale_c.to(
            device=candidate_base_bch.device,
            dtype=candidate_base_bch.dtype,
        ).view(1, -1, 1)
    if pred_residual_scale_c is not None:
        channel_scale = pred_residual_scale_c.to(
            device=candidate_base_bch.device,
            dtype=candidate_base_bch.dtype,
        ).view(1, -1, 1)
        scale_bcp = scale_bcp * channel_scale
    if patch_route_bcph is not None and bool(include_patch_route):
        return candidate_base_bch.unsqueeze(2) + scale_bcp.unsqueeze(-1) * patch_route_bcph * residuals
    return candidate_base_bch.unsqueeze(2) + scale_bcp.unsqueeze(-1) * residuals


def _pred_residual_candidates_on_eval_path(
    y_base_bch: torch.Tensor,
    pred_out: Dict[str, torch.Tensor],
    *,
    pred_residual_scale_c: Optional[torch.Tensor] = None,
    apply_output_anchors: bool = False,
    x_bcl: Optional[torch.Tensor] = None,
    query_start_abs_b: Optional[torch.Tensor] = None,
    input_len: int = 0,
    moe_cfg: Optional[dict] = None,
    moe_enable: bool = True,
    observed_history_tc: Optional[torch.Tensor] = None,
    train_stat_anchor_pc: Optional[torch.Tensor] = None,
    train_residual_anchor_phc: Optional[torch.Tensor] = None,
    learnable_output_anchor: Optional[ClusterwiseLearnableOutputAnchor] = None,
    cluster_id_c: Optional[torch.Tensor] = None,
    include_intervention: bool = True,
    include_selector: bool = True,
    include_patch_route: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    candidate_base_bch = pred_out.get("candidate_base_bch", y_base_bch)
    cand_bcpH = _pred_residual_candidate_predictions(
        y_base_bch,
        pred_out,
        pred_residual_scale_c=pred_residual_scale_c,
        include_intervention=include_intervention,
        include_selector=include_selector,
        include_patch_route=include_patch_route,
    )
    patch_application_scale_p = pred_out.get("patch_application_scale_p")
    if (
        cand_bcpH is not None
        and patch_application_scale_p is not None
        and int(patch_application_scale_p.numel()) > 0
    ):
        if int(patch_application_scale_p.numel()) != int(cand_bcpH.shape[2]):
            raise ValueError(
                "patch application scale length must match candidate penalties."
            )
        application_scale = patch_application_scale_p.to(
            device=cand_bcpH.device,
            dtype=cand_bcpH.dtype,
        ).view(1, 1, -1, 1)
        cand_bcpH = candidate_base_bch.unsqueeze(2) + application_scale * (
            cand_bcpH - candidate_base_bch.unsqueeze(2)
        )
    if cand_bcpH is None or not apply_output_anchors:
        return candidate_base_bch, cand_bcpH
    if x_bcl is None or query_start_abs_b is None:
        raise ValueError("Output-anchor candidate evaluation requires x_bcl and query_start_abs_b.")
    y_base_final = apply_moe_output_anchor_experts(
        candidate_base_bch,
        base_pred_bch=y_base_bch,
        x_bcl=x_bcl,
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
    cand_final_parts = [
        apply_moe_output_anchor_experts(
            cand_bcpH[:, :, p, :],
            base_pred_bch=y_base_bch,
            x_bcl=x_bcl,
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
        for p in range(int(cand_bcpH.shape[2]))
    ]
    return y_base_final, torch.stack(cand_final_parts, dim=2)


def _candidate_selector_targets(
    *,
    base_bch: torch.Tensor,
    cand_bcpH: torch.Tensor,
    y_bch: torch.Tensor,
    min_abs_improvement: float = 0.0,
    min_rel_improvement: float = 0.0,
    allowed_mask_cp: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    base_err_bc = (base_bch - y_bch).pow(2).mean(dim=-1)
    cand_err_bcp = (cand_bcpH - y_bch.unsqueeze(2)).pow(2).mean(dim=-1)
    allowed = _selector_allowed_mask_cp(
        allowed_mask_cp,
        C=int(cand_err_bcp.shape[1]),
        P=int(cand_err_bcp.shape[2]),
        device=cand_err_bcp.device,
        context="candidate selector target",
    )
    if allowed is not None:
        cand_err_bcp = cand_err_bcp.masked_fill(~allowed.unsqueeze(0), float("inf"))
    best_err_bc, best_p_bc = cand_err_bcp.min(dim=-1)
    gain_bc = base_err_bc - best_err_bc
    gain_bc = torch.where(torch.isfinite(gain_bc), gain_bc, torch.full_like(gain_bc, float("-inf")))
    required_bc = torch.maximum(
        torch.full_like(base_err_bc, max(0.0, float(min_abs_improvement))),
        max(0.0, float(min_rel_improvement)) * base_err_bc.abs().clamp_min(1.0e-12),
    )
    return torch.where(gain_bc > required_bc, best_p_bc.to(dtype=torch.long) + 1, torch.zeros_like(best_p_bc))


def _candidate_selector_expected_error_loss(
    *,
    logits_bcq: torch.Tensor,
    base_bch: torch.Tensor,
    cand_bcpH: torch.Tensor,
    y_bch: torch.Tensor,
    temperature: float = 1.0,
    loss_kind: str = "mse",
) -> torch.Tensor:
    if int(logits_bcq.shape[-1]) != int(cand_bcpH.shape[2]) + 1:
        raise ValueError(
            "candidate selector expected-error logits must have P+1 classes, "
            f"got logits={tuple(logits_bcq.shape)} and candidates={tuple(cand_bcpH.shape)}."
        )
    kind = str(loss_kind or "mse").lower()
    if kind in {"mse", "l2", "expected_mse"}:
        base_err_bc = (base_bch - y_bch).pow(2).mean(dim=-1)
        cand_err_bcp = (cand_bcpH - y_bch.unsqueeze(2)).pow(2).mean(dim=-1)
    elif kind in {"mae", "l1", "expected_mae"}:
        base_err_bc = (base_bch - y_bch).abs().mean(dim=-1)
        cand_err_bcp = (cand_bcpH - y_bch.unsqueeze(2)).abs().mean(dim=-1)
    else:
        raise ValueError("candidate_selector expected-error loss_kind must be mse or mae.")
    err_bcq = torch.cat([base_err_bc.unsqueeze(-1), cand_err_bcp], dim=-1).detach()
    temp = max(float(temperature), 1.0e-6)
    prob_bcq = torch.softmax(logits_bcq / temp, dim=-1)
    return (prob_bcq * err_bcq).sum(dim=-1).mean()


def _candidate_selector_rate_alignment_loss(
    *,
    logits_bcq: torch.Tensor,
    target_rate_q: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    if logits_bcq.ndim != 3:
        raise ValueError(f"candidate selector logits must be [B,C,Q], got {tuple(logits_bcq.shape)}.")
    target = target_rate_q.to(device=logits_bcq.device, dtype=logits_bcq.dtype).view(-1)
    if int(target.numel()) != int(logits_bcq.shape[-1]):
        raise ValueError(
            f"candidate selector target class rate must have {int(logits_bcq.shape[-1])} entries, "
            f"got {int(target.numel())}."
        )
    target = target.clamp_min(0.0)
    target = target / target.sum().clamp_min(1.0e-12)
    temp = max(float(temperature), 1.0e-6)
    pred_rate_q = torch.softmax(logits_bcq / temp, dim=-1).mean(dim=(0, 1))
    return (pred_rate_q - target).pow(2).mean()


def _selector_allowed_mask_cp(
    allowed_mask_cp: Optional[torch.Tensor],
    *,
    C: int,
    P: int,
    device: torch.device,
    context: str,
) -> Optional[torch.Tensor]:
    if allowed_mask_cp is None or int(allowed_mask_cp.numel()) == 0:
        return None
    allowed = allowed_mask_cp.to(device=device, dtype=torch.bool)
    if tuple(allowed.shape) != (int(C), int(P)):
        raise ValueError(
            f"{context} allowed mask must have shape [C,P], "
            f"got {tuple(allowed.shape)} vs {(int(C), int(P))}."
        )
    return allowed


def _candidate_selector_adoption_decision(
    *,
    current_mse: float,
    current_mae: float,
    selector_mse: float,
    selector_mae: float,
    min_abs_improvement: float = 0.0,
    min_rel_improvement: float = 0.0,
    max_rel_mae_regression: float = 0.0,
) -> Dict[str, object]:
    current_mse_f = float(current_mse)
    current_mae_f = float(current_mae)
    selector_mse_f = float(selector_mse)
    selector_mae_f = float(selector_mae)
    required = max(
        max(0.0, float(min_abs_improvement)),
        max(0.0, float(min_rel_improvement)) * max(abs(current_mse_f), 1.0e-12),
    )
    mse_improvement = current_mse_f - selector_mse_f
    mae_regression = selector_mae_f - current_mae_f
    allowed_mae_regression = max(0.0, float(max_rel_mae_regression)) * max(abs(current_mae_f), 1.0e-12)
    finite = all(math.isfinite(v) for v in (current_mse_f, current_mae_f, selector_mse_f, selector_mae_f))
    adopt = bool(finite and mse_improvement > required and mae_regression <= allowed_mae_regression)
    return {
        "adopt": adopt,
        "current_mse": current_mse_f,
        "current_mae": current_mae_f,
        "selector_mse": selector_mse_f,
        "selector_mae": selector_mae_f,
        "mse_improvement": float(mse_improvement),
        "mse_improvement_pct": float(100.0 * mse_improvement / max(abs(current_mse_f), 1.0e-12)),
        "required_mse_improvement": float(required),
        "min_abs_improvement": float(max(0.0, float(min_abs_improvement))),
        "min_rel_improvement": float(max(0.0, float(min_rel_improvement))),
        "mae_regression": float(mae_regression),
        "mae_regression_pct": float(100.0 * mae_regression / max(abs(current_mae_f), 1.0e-12)),
        "allowed_mae_regression": float(allowed_mae_regression),
        "max_rel_mae_regression": float(max(0.0, float(max_rel_mae_regression))),
        "finite": bool(finite),
    }


def _candidate_selector_temporal_block_adoption_guard(
    *,
    blocks: Optional[List[Dict[str, object]]],
    min_gain_pct: float,
    min_positive_blocks: Optional[int] = None,
) -> Dict[str, object]:
    block_list = list(blocks or [])
    gains = [
        float(block.get("selected_gain_pct_vs_base", float("nan")))
        for block in block_list
        if math.isfinite(float(block.get("selected_gain_pct_vs_base", float("nan"))))
    ]
    block_count = int(len(gains))
    required = block_count if min_positive_blocks is None else int(min_positive_blocks)
    required = max(0, min(required, block_count))
    threshold = float(min_gain_pct)
    positive_count = sum(1 for gain in gains if gain >= threshold)
    passed = bool(block_count > 0 and positive_count >= required)
    return {
        "passed": passed,
        "block_count": block_count,
        "positive_block_count": int(positive_count),
        "required_positive_blocks": int(required),
        "min_gain_pct": threshold,
        "min_observed_gain_pct": float(min(gains)) if gains else None,
        "max_observed_gain_pct": float(max(gains)) if gains else None,
        "selected_gain_pct_by_block": [float(v) for v in gains],
    }


def _candidate_selector_choose_temporal_margin_row(
    rows: List[Dict[str, object]],
    *,
    required_positive_blocks: int,
) -> Optional[Dict[str, object]]:
    if not rows:
        return None
    required = max(0, int(required_positive_blocks))
    feasible = [
        row
        for row in rows
        if int(row.get("positive_block_count", 0)) >= required
    ]
    if feasible:
        return min(
            feasible,
            key=lambda row: (
                float(row.get("selected_mse", float("inf"))),
                -float(row.get("min_block_gain_pct", float("-inf"))),
                float(row.get("margin", 0.0)),
            ),
        )
    return max(
        rows,
        key=lambda row: (
            float(row.get("min_block_gain_pct", float("-inf"))),
            int(row.get("positive_block_count", 0)),
            -float(row.get("selected_mse", float("inf"))),
            -float(row.get("margin", 0.0)),
        ),
    )


def _mix_selected_channel_metrics(
    *,
    base_mse_c: torch.Tensor,
    base_mae_c: torch.Tensor,
    residual_mse_c: torch.Tensor,
    residual_mae_c: torch.Tensor,
    use_residual_c: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    keep = use_residual_c.to(device=base_mse_c.device, dtype=torch.bool)
    residual_mse = residual_mse_c.to(device=base_mse_c.device, dtype=base_mse_c.dtype)
    residual_mae = residual_mae_c.to(device=base_mae_c.device, dtype=base_mae_c.dtype)
    return (
        torch.where(keep, residual_mse, base_mse_c),
        torch.where(keep.to(device=base_mae_c.device), residual_mae, base_mae_c),
    )


def _candidate_selector_candidate_scale(
    *,
    pred_residual_scale_c: Optional[torch.Tensor],
    selector_cfg: Dict[str, object],
) -> Tuple[Optional[torch.Tensor], str]:
    use_channel_scale = bool(selector_cfg.get("use_channel_scale_for_candidates", True))
    if use_channel_scale:
        return pred_residual_scale_c, "channel_scale"
    return None, "unscaled"


def _cluster_utility_threshold_stats(
    *,
    gain_bcp: torch.Tensor,
    cluster_id_c: torch.Tensor,
    K: int,
    allowed_mask_kp: Optional[torch.Tensor],
    thresholds: List[float],
) -> Dict[str, torch.Tensor]:
    if gain_bcp.numel() == 0:
        return {
            "valid_count_kt": torch.zeros(int(K), len(thresholds), dtype=torch.float64),
            "total_count_k": torch.zeros(int(K), dtype=torch.float64),
            "best_gain_sum_k": torch.zeros(int(K), dtype=torch.float64),
            "best_gain_positive_count_k": torch.zeros(int(K), dtype=torch.float64),
        }
    gain_bkp = scatter_mean_bcf_to_bkf(gain_bcp, cluster_id_c, int(K)).to(dtype=torch.float64)
    B, _, P = gain_bkp.shape
    if allowed_mask_kp is not None and allowed_mask_kp.numel() > 0:
        allowed = allowed_mask_kp.to(device=gain_bkp.device, dtype=torch.bool)
        if tuple(allowed.shape) != (int(K), int(P)):
            raise ValueError(
                "utility threshold allowed mask must have shape [K,P], "
                f"got {tuple(allowed.shape)} vs {(int(K), int(P))}."
            )
    else:
        allowed = torch.ones((int(K), int(P)), device=gain_bkp.device, dtype=torch.bool)
    allowed_bkp = allowed.unsqueeze(0).expand_as(gain_bkp)
    masked_gain_bkp = gain_bkp.masked_fill(~allowed_bkp, -float("inf"))
    best_gain_bk = masked_gain_bkp.max(dim=-1).values
    best_gain_bk = torch.nan_to_num(best_gain_bk, nan=0.0, posinf=0.0, neginf=0.0)
    valid_counts = []
    allowed_float = allowed_bkp.to(dtype=gain_bkp.dtype)
    for threshold in thresholds:
        utility_bkp = (gain_bkp - float(threshold)).clamp_min(0.0) * allowed_float
        valid_counts.append((utility_bkp.sum(dim=-1) > 0.0).sum(dim=0).to(dtype=torch.float64))
    if valid_counts:
        valid_count_kt = torch.stack(valid_counts, dim=-1)
    else:
        valid_count_kt = torch.zeros(int(K), 0, device=gain_bkp.device, dtype=torch.float64)
    return {
        "valid_count_kt": valid_count_kt.detach().cpu(),
        "total_count_k": torch.full((int(K),), int(B), dtype=torch.float64),
        "best_gain_sum_k": best_gain_bk.sum(dim=0).detach().cpu().to(dtype=torch.float64),
        "best_gain_positive_count_k": (best_gain_bk > 0.0).sum(dim=0).detach().cpu().to(dtype=torch.float64),
    }


def _mse_utility_gate_supervision_loss(
    *,
    probs_bkp: Optional[torch.Tensor],
    skip_prob_bk: Optional[torch.Tensor] = None,
    allowed_mask_kp: Optional[torch.Tensor] = None,
    y_base_bch: torch.Tensor,
    pred_out: Optional[Dict[str, torch.Tensor]],
    y_bch: torch.Tensor,
    cluster_id_c: torch.Tensor,
    K: int,
    y_base_eval_bch: Optional[torch.Tensor] = None,
    cand_eval_bcpH: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    min_gain: float = 0.0,
    mae_weight: float = 0.0,
    target_power: float = 1.0,
    include_skip: bool = False,
    probs_include_skip_mass: bool = False,
    target_mode: str = "soft_utility",
    return_diagnostics: bool = False,
    eps: float = 1.0e-8,
) -> Optional[torch.Tensor]:
    if probs_bkp is None or pred_out is None or probs_bkp.numel() == 0:
        return (None, None) if return_diagnostics else None
    target_mode_l = str(target_mode or "soft_utility").lower()
    hard_target = target_mode_l in {"hard", "hard_oracle", "argmax", "one_hot"}
    if target_mode_l not in {"soft", "soft_utility", "utility", "hard", "hard_oracle", "argmax", "one_hot"}:
        raise ValueError(
            "mse utility gate supervision target_mode must be one of "
            "soft_utility or hard_oracle aliases, "
            f"got {target_mode!r}."
        )
    base_bch = y_base_eval_bch if y_base_eval_bch is not None else y_base_bch
    cand_bcpH = cand_eval_bcpH if cand_eval_bcpH is not None else _pred_residual_candidate_predictions(y_base_bch, pred_out)
    if cand_bcpH is None or cand_bcpH.numel() == 0:
        return (None, None) if return_diagnostics else None
    with torch.no_grad():
        base_err_bc = (base_bch - y_bch).pow(2).mean(dim=-1)
        cand_err_bcp = (cand_bcpH - y_bch.unsqueeze(2)).pow(2).mean(dim=-1)
        gain_bcp = base_err_bc.unsqueeze(-1) - cand_err_bcp
        if float(mae_weight) != 0.0:
            base_mae_bc = (base_bch - y_bch).abs().mean(dim=-1)
            cand_mae_bcp = (cand_bcpH - y_bch.unsqueeze(2)).abs().mean(dim=-1)
            gain_bcp = gain_bcp + float(mae_weight) * (
                base_mae_bc.unsqueeze(-1) - cand_mae_bcp
            )
        gain_bkp = scatter_mean_bcf_to_bkf(gain_bcp, cluster_id_c, K)
        if allowed_mask_kp is not None and allowed_mask_kp.numel() > 0:
            allowed = allowed_mask_kp.to(device=probs_bkp.device, dtype=torch.bool)
            allowed_bkp = allowed.unsqueeze(0).expand_as(probs_bkp).to(dtype=gain_bkp.dtype)
        else:
            allowed_bkp = (probs_bkp.detach() > 0.0).to(dtype=gain_bkp.dtype)
        utility_bkp = (gain_bkp - float(min_gain)).clamp_min(0.0) * allowed_bkp
        if float(target_power) != 1.0:
            utility_bkp = utility_bkp.clamp_min(0.0).pow(float(target_power))
        valid_bk = utility_bkp.sum(dim=-1) > 0.0
        masked_gain_bkp = gain_bkp.masked_fill(allowed_bkp <= 0.0, float("-inf"))
        best_gain_bk, best_idx_bk = masked_gain_bkp.max(dim=-1)
        best_gain_bk = torch.where(torch.isfinite(best_gain_bk), best_gain_bk, torch.zeros_like(best_gain_bk))
        target_skip_bk = (~valid_bk).to(dtype=utility_bkp.dtype) if bool(include_skip) else torch.zeros_like(utility_bkp[..., 0])
        diagnostics = {
            "valid_bk": valid_bk.to(dtype=utility_bkp.dtype).detach(),
            "target_skip_bk": target_skip_bk.detach(),
            "best_gain_bk": best_gain_bk.detach(),
            "utility_sum_bk": utility_bkp.sum(dim=-1).detach(),
            "probs_include_skip_mass": torch.full_like(valid_bk, float(bool(probs_include_skip_mass)), dtype=utility_bkp.dtype).detach(),
        }
        if skip_prob_bk is not None:
            diagnostics["skip_prob_bk"] = skip_prob_bk.detach()
        if bool(include_skip):
            if skip_prob_bk is None:
                return (None, diagnostics) if return_diagnostics else None
            if tuple(skip_prob_bk.shape) != tuple(probs_bkp.shape[:2]):
                raise ValueError(
                    "skip-aware mse utility gate supervision requires skip_prob_bk shape [B,K], "
                    f"got {tuple(skip_prob_bk.shape)} vs {tuple(probs_bkp.shape[:2])}."
                )
            target_bkp = torch.zeros_like(utility_bkp)
            if bool(valid_bk.any().item()):
                if hard_target:
                    target_bkp.scatter_(-1, best_idx_bk.unsqueeze(-1), 1.0)
                    target_bkp = target_bkp * valid_bk.unsqueeze(-1).to(dtype=utility_bkp.dtype)
                    target_bkp = target_bkp * allowed_bkp
                else:
                    temp = max(float(temperature), 1.0e-6)
                    pos_logits = utility_bkp.clamp_min(eps).log() / temp
                    pos_logits = pos_logits.masked_fill(~valid_bk.unsqueeze(-1), 0.0)
                    target_bkp = torch.softmax(pos_logits, dim=-1) * valid_bk.unsqueeze(-1).to(dtype=utility_bkp.dtype)
                    target_bkp = target_bkp * allowed_bkp
                    target_bkp = target_bkp / target_bkp.sum(dim=-1, keepdim=True).clamp_min(eps)
        else:
            if not bool(valid_bk.any().item()):
                return (None, diagnostics) if return_diagnostics else None
            if hard_target:
                target_bkp = torch.zeros_like(utility_bkp)
                target_bkp.scatter_(-1, best_idx_bk.unsqueeze(-1), 1.0)
                target_bkp = target_bkp * valid_bk.unsqueeze(-1).to(dtype=utility_bkp.dtype)
                target_bkp = target_bkp * allowed_bkp
            else:
                temp = max(float(temperature), 1.0e-6)
                target_bkp = utility_bkp.clamp_min(eps).log() / temp
                target_bkp = target_bkp.masked_fill(~valid_bk.unsqueeze(-1), 0.0)
                target_bkp = torch.softmax(target_bkp, dim=-1) * valid_bk.unsqueeze(-1).to(dtype=utility_bkp.dtype)
                target_bkp = target_bkp * allowed_bkp
                target_bkp = target_bkp / target_bkp.sum(dim=-1, keepdim=True).clamp_min(eps)
    if bool(include_skip):
        if bool(probs_include_skip_mass):
            penalty_mass_bkp = probs_bkp.clamp_min(eps)
        else:
            penalty_mass_bkp = (1.0 - skip_prob_bk).clamp_min(eps).unsqueeze(-1) * probs_bkp.clamp_min(eps)
        loss_penalty_bk = -(target_bkp * penalty_mass_bkp.log()).sum(dim=-1)
        loss_skip_bk = -(target_skip_bk * skip_prob_bk.clamp_min(eps).log())
        loss_bk = loss_penalty_bk + loss_skip_bk
        return (loss_bk, diagnostics) if return_diagnostics else loss_bk
    log_probs = probs_bkp.clamp_min(eps).log()
    loss_bk = -(target_bkp * log_probs).sum(dim=-1)
    loss_bk = torch.where(valid_bk, loss_bk, torch.zeros_like(loss_bk))
    return (loss_bk, diagnostics) if return_diagnostics else loss_bk


def _patch_router_expected_mse_loss_bk(
    *,
    base_bch: torch.Tensor,
    candidate_bcpH: torch.Tensor,
    y_bch: torch.Tensor,
    patch_probs_bcqp: torch.Tensor,
    patch_skip_prob_bcq: torch.Tensor,
    cluster_id_c: torch.Tensor,
    K: int,
    mae_weight: float = 0.0,
) -> torch.Tensor:
    """Train the input-only patch router against detached patch forecast error."""
    if base_bch.ndim != 3 or y_bch.shape != base_bch.shape:
        raise ValueError("patch router utility expects base/y with shape [B,C,H].")
    if candidate_bcpH.ndim != 4 or candidate_bcpH.shape[:2] != base_bch.shape[:2]:
        raise ValueError("patch router utility expects candidates with shape [B,C,P,H].")
    if patch_probs_bcqp.ndim != 4:
        raise ValueError("patch router utility expects probabilities with shape [B,C,Q,P].")
    batch, channels, horizon = base_bch.shape
    patches = int(patch_probs_bcqp.shape[2])
    penalties = int(patch_probs_bcqp.shape[3])
    if patches <= 0 or horizon % patches != 0:
        raise ValueError("patch router utility requires patch count to divide horizon.")
    if tuple(candidate_bcpH.shape) != (batch, channels, penalties, horizon):
        raise ValueError("patch router utility candidate/probability dimensions do not match.")
    if tuple(patch_skip_prob_bcq.shape) != (batch, channels, patches):
        raise ValueError("patch router utility skip probability shape does not match.")

    patch_len = horizon // patches
    base_delta_bch = base_bch - y_bch
    candidate_delta_bcpH = candidate_bcpH - y_bch.unsqueeze(2)
    base_error_bcq = base_delta_bch.square().reshape(
        batch,
        channels,
        patches,
        patch_len,
    ).mean(dim=-1)
    candidate_error_bcqp = candidate_delta_bcpH.square().reshape(
        batch,
        channels,
        penalties,
        patches,
        patch_len,
    ).mean(dim=-1).permute(0, 1, 3, 2)
    mae_weight = max(0.0, float(mae_weight))
    if mae_weight > 0.0:
        base_error_bcq = base_error_bcq + mae_weight * base_delta_bch.abs().reshape(
            batch,
            channels,
            patches,
            patch_len,
        ).mean(dim=-1)
        candidate_error_bcqp = candidate_error_bcqp + mae_weight * candidate_delta_bcpH.abs().reshape(
            batch,
            channels,
            penalties,
            patches,
            patch_len,
        ).mean(dim=-1).permute(0, 1, 3, 2)
    expected_error_bcq = (
        patch_skip_prob_bcq * base_error_bcq.detach()
        + (patch_probs_bcqp * candidate_error_bcqp.detach()).sum(dim=-1)
    )
    return scatter_mean_bc_to_bk(expected_error_bcq.mean(dim=-1), cluster_id_c, int(K))


def _patch_router_mixture_mse_loss_bk(
    *,
    base_bch: torch.Tensor,
    candidate_bcpH: torch.Tensor,
    y_bch: torch.Tensor,
    patch_probs_bcqp: torch.Tensor,
    cluster_id_c: torch.Tensor,
    K: int,
    mae_weight: float = 0.0,
    allowed_penalty_mask_cp: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Train input probabilities on the exact deterministic mixture forecast."""
    if base_bch.ndim != 3 or y_bch.shape != base_bch.shape:
        raise ValueError("patch router mixture expects base/y with shape [B,C,H].")
    if candidate_bcpH.ndim != 4 or candidate_bcpH.shape[:2] != base_bch.shape[:2]:
        raise ValueError("patch router mixture expects candidates with shape [B,C,P,H].")
    if patch_probs_bcqp.ndim != 4:
        raise ValueError("patch router mixture expects probabilities with shape [B,C,Q,P].")
    batch, channels, horizon = base_bch.shape
    patches = int(patch_probs_bcqp.shape[2])
    penalties = int(patch_probs_bcqp.shape[3])
    if patches <= 0 or horizon % patches != 0:
        raise ValueError("patch router mixture requires patch count to divide horizon.")
    if tuple(candidate_bcpH.shape) != (batch, channels, penalties, horizon):
        raise ValueError("patch router mixture candidate/probability dimensions do not match.")

    effective_probs_bcqp = patch_probs_bcqp
    if allowed_penalty_mask_cp is not None and int(allowed_penalty_mask_cp.numel()) > 0:
        if tuple(allowed_penalty_mask_cp.shape) != (channels, penalties):
            raise ValueError(
                "patch router mixture allowed penalty mask must have shape [C,P], "
                f"got {tuple(allowed_penalty_mask_cp.shape)} vs {(channels, penalties)}."
            )
        effective_probs_bcqp = (
            effective_probs_bcqp
            * allowed_penalty_mask_cp.to(
                device=patch_probs_bcqp.device,
                dtype=patch_probs_bcqp.dtype,
            ).view(1, channels, 1, penalties)
        )

    patch_len = horizon // patches
    weight_bcpH = (
        effective_probs_bcqp.permute(0, 1, 3, 2)
        .unsqueeze(-1)
        .expand(-1, -1, -1, -1, patch_len)
        .reshape(batch, channels, penalties, horizon)
    )
    base_fixed = base_bch.detach()
    candidate_delta = candidate_bcpH.detach() - base_fixed.unsqueeze(2)
    mixed_bch = base_fixed + (weight_bcpH * candidate_delta).sum(dim=2)
    error_bch = mixed_bch - y_bch
    loss_bc = error_bch.square().mean(dim=-1)
    mae_weight = max(0.0, float(mae_weight))
    if mae_weight > 0.0:
        loss_bc = loss_bc + mae_weight * error_bch.abs().mean(dim=-1)
    return scatter_mean_bc_to_bk(loss_bc, cluster_id_c, int(K))


def _patch_router_oracle_ce_loss_bk(
    *,
    base_bch: torch.Tensor,
    candidate_bcpH: torch.Tensor,
    y_bch: torch.Tensor,
    patch_probs_bcqp: torch.Tensor,
    patch_skip_prob_bcq: torch.Tensor,
    cluster_id_c: torch.Tensor,
    K: int,
    min_abs_improvement: float = 0.0,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Cross-entropy for the aligned patch-level skip-or-penalty decision."""
    batch, channels, horizon = base_bch.shape
    patches = int(patch_probs_bcqp.shape[2])
    penalties = int(patch_probs_bcqp.shape[3])
    if patches <= 0 or horizon % patches != 0:
        raise ValueError("patch router oracle CE requires patch count to divide horizon.")
    if tuple(candidate_bcpH.shape) != (batch, channels, penalties, horizon):
        raise ValueError("patch router oracle CE candidate/probability dimensions do not match.")
    if tuple(patch_skip_prob_bcq.shape) != (batch, channels, patches):
        raise ValueError("patch router oracle CE skip probability shape does not match.")
    patch_len = horizon // patches
    with torch.no_grad():
        base_error_bcq = (base_bch - y_bch).square().reshape(
            batch,
            channels,
            patches,
            patch_len,
        ).mean(dim=-1)
        candidate_error_bcqp = (candidate_bcpH - y_bch.unsqueeze(2)).square().reshape(
            batch,
            channels,
            penalties,
            patches,
            patch_len,
        ).mean(dim=-1).permute(0, 1, 3, 2)
        best_error_bcq, best_penalty_bcq = candidate_error_bcqp.min(dim=-1)
        use_penalty_bcq = (base_error_bcq - best_error_bcq) > float(min_abs_improvement)
        labels_bcq = torch.where(
            use_penalty_bcq,
            best_penalty_bcq + 1,
            torch.zeros_like(best_penalty_bcq),
        )
    route_probs_bcqk = torch.cat(
        [patch_skip_prob_bcq.unsqueeze(-1), patch_probs_bcqp],
        dim=-1,
    )
    nll_bcq = -route_probs_bcqk.clamp_min(float(eps)).log().gather(
        dim=-1,
        index=labels_bcq.unsqueeze(-1),
    ).squeeze(-1)
    return scatter_mean_bc_to_bk(nll_bcq.mean(dim=-1), cluster_id_c, int(K))


@torch.no_grad()
def _select_recall_constrained_risk_threshold(
    *,
    score_n: torch.Tensor,
    gain_n: torch.Tensor,
    block_n: torch.Tensor,
    min_gain_cost_ratio: float = 1.0,
    min_block_net_gain: float = 0.0,
) -> Dict[str, object]:
    """Maximize positive recall subject to global and temporal utility constraints."""
    score = score_n.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
    gain = gain_n.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
    block = block_n.detach().reshape(-1).to(dtype=torch.long, device="cpu")
    if not (score.numel() == gain.numel() == block.numel()):
        raise ValueError("risk calibration score/gain/block lengths must match.")
    finite = torch.isfinite(score) & torch.isfinite(gain)
    score = score[finite]
    gain = gain[finite]
    block = block[finite]
    if int(score.numel()) == 0:
        return {
            "status": "empty",
            "threshold": 0.0,
            "selected_count": 0,
            "selected_positive_count": 0,
            "positive_recall": 0.0,
            "positive_gain": 0.0,
            "negative_cost": 0.0,
            "gain_cost_ratio": 0.0,
            "net_gain": 0.0,
            "block_net_gain": [],
        }
    if bool((block < 0).any().item()):
        raise ValueError("risk calibration block ids must be nonnegative.")
    min_ratio = max(float(min_gain_cost_ratio), 0.0)
    min_block_gain = float(min_block_net_gain)
    order = torch.argsort(score, descending=True, stable=True)
    sorted_score = score.index_select(0, order)
    sorted_gain = gain.index_select(0, order)
    sorted_block = block.index_select(0, order)
    positive_gain = sorted_gain.clamp_min(0.0).cumsum(dim=0)
    negative_cost = (-sorted_gain).clamp_min(0.0).cumsum(dim=0)
    positive_count = (sorted_gain > 0.0).to(dtype=torch.long).cumsum(dim=0)
    selected_count = torch.arange(1, int(score.numel()) + 1, dtype=torch.long)
    boundary = torch.ones_like(sorted_score, dtype=torch.bool)
    if int(sorted_score.numel()) > 1:
        boundary[:-1] = sorted_score[:-1] > sorted_score[1:]
    block_count = int(sorted_block.max().item()) + 1
    block_net = torch.zeros(
        (int(score.numel()), block_count),
        dtype=torch.float64,
    )
    for block_id in range(block_count):
        block_gain = torch.where(
            sorted_block == block_id,
            sorted_gain,
            torch.zeros_like(sorted_gain),
        )
        block_net[:, block_id] = block_gain.cumsum(dim=0)
    valid = (
        boundary
        & (positive_gain >= min_ratio * negative_cost)
        & (block_net >= min_block_gain).all(dim=1)
    )
    valid_idx = torch.nonzero(valid, as_tuple=False).reshape(-1)
    if int(valid_idx.numel()) == 0:
        max_score = float(sorted_score[0].item())
        threshold = max_score + max(1.0e-6, abs(max_score) * 1.0e-6)
        return {
            "status": "no_feasible_adoption",
            "threshold": threshold,
            "selected_count": 0,
            "selected_positive_count": 0,
            "positive_recall": 0.0,
            "positive_gain": 0.0,
            "negative_cost": 0.0,
            "gain_cost_ratio": 0.0,
            "net_gain": 0.0,
            "block_net_gain": [0.0 for _ in range(block_count)],
        }
    best_positive_count = positive_count.index_select(0, valid_idx).max()
    candidates = valid_idx[
        positive_count.index_select(0, valid_idx) == best_positive_count
    ]
    candidate_net = (
        positive_gain.index_select(0, candidates)
        - negative_cost.index_select(0, candidates)
    )
    best_net = candidate_net.max()
    candidates = candidates[candidate_net == best_net]
    best_idx = int(candidates.min().item())
    included = best_idx + 1
    cutoff = float(sorted_score[best_idx].item())
    if included < int(sorted_score.numel()):
        next_score = float(sorted_score[included].item())
        threshold = 0.5 * (cutoff + next_score)
    else:
        threshold = cutoff - max(1.0e-6, abs(cutoff) * 1.0e-6)
    selected_positive = int(positive_count[best_idx].item())
    total_positive = int((gain > 0.0).sum().item())
    pos_gain = float(positive_gain[best_idx].item())
    neg_cost = float(negative_cost[best_idx].item())
    return {
        "status": "ok",
        "threshold": float(threshold),
        "selected_count": int(selected_count[best_idx].item()),
        "selected_positive_count": selected_positive,
        "positive_recall": float(selected_positive / max(total_positive, 1)),
        "positive_gain": pos_gain,
        "negative_cost": neg_cost,
        "gain_cost_ratio": float(pos_gain / max(neg_cost, 1.0e-12)),
        "net_gain": float(pos_gain - neg_cost),
        "block_net_gain": [float(value) for value in block_net[best_idx].tolist()],
    }


@torch.no_grad()
def _risk_score_threshold_curve_summary(
    *,
    score_n: torch.Tensor,
    gain_n: torch.Tensor,
    fixed_threshold: float,
) -> Dict[str, object]:
    """Summarize score ordering separately from its configured decision cutoff."""
    score = score_n.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
    gain = gain_n.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
    if int(score.numel()) != int(gain.numel()):
        raise ValueError("risk score threshold curve score/gain lengths must match.")
    finite = torch.isfinite(score) & torch.isfinite(gain)
    score = score[finite]
    gain = gain[finite]
    if int(score.numel()) == 0:
        return {
            "status": "empty",
            "fixed_threshold": float(fixed_threshold),
            "total_count": 0,
            "positive_count": 0,
        }

    positive = gain > 0.0
    positive_count = int(positive.sum().item())
    negative_count = int(score.numel()) - positive_count

    def summarize_selection(
        selected: torch.Tensor,
        *,
        threshold: float,
    ) -> Dict[str, object]:
        selected_gain = gain[selected]
        selected_positive = int((selected_gain > 0.0).sum().item())
        selected_count = int(selected.sum().item())
        positive_gain = float(selected_gain.clamp_min(0.0).sum().item())
        negative_cost = float((-selected_gain).clamp_min(0.0).sum().item())
        return {
            "threshold": float(threshold),
            "selected_count": selected_count,
            "adoption_rate": float(selected_count / max(int(score.numel()), 1)),
            "selected_positive_count": selected_positive,
            "positive_recall": float(selected_positive / max(positive_count, 1)),
            "positive_precision": float(selected_positive / max(selected_count, 1)),
            "positive_gain": positive_gain,
            "negative_cost": negative_cost,
            "gain_cost_ratio": float(positive_gain / max(negative_cost, 1.0e-12)),
            "net_gain": float(positive_gain - negative_cost),
        }

    fixed = summarize_selection(
        score > float(fixed_threshold),
        threshold=float(fixed_threshold),
    )
    order = torch.argsort(score, descending=True, stable=True)
    sorted_score = score.index_select(0, order)
    sorted_gain = gain.index_select(0, order)
    sorted_positive = positive.index_select(0, order)
    cumulative_net = sorted_gain.cumsum(dim=0)
    boundary = torch.ones_like(sorted_score, dtype=torch.bool)
    if int(sorted_score.numel()) > 1:
        boundary[:-1] = sorted_score[:-1] > sorted_score[1:]
    boundary_idx = torch.nonzero(boundary, as_tuple=False).reshape(-1)
    boundary_net = cumulative_net.index_select(0, boundary_idx)

    benefit_auroc = None
    benefit_average_precision = None
    if positive_count > 0 and negative_count > 0:
        cumulative_positive = sorted_positive.to(torch.float64).cumsum(dim=0)
        cumulative_negative = (~sorted_positive).to(torch.float64).cumsum(dim=0)
        true_positive_rate = torch.cat(
            (
                torch.zeros(1, dtype=torch.float64),
                cumulative_positive.index_select(0, boundary_idx)
                / float(positive_count),
            )
        )
        false_positive_rate = torch.cat(
            (
                torch.zeros(1, dtype=torch.float64),
                cumulative_negative.index_select(0, boundary_idx)
                / float(negative_count),
            )
        )
        benefit_auroc = float(
            torch.trapz(true_positive_rate, false_positive_rate).item()
        )
        recall = true_positive_rate[1:]
        precision = cumulative_positive.index_select(0, boundary_idx) / (
            boundary_idx.to(torch.float64) + 1.0
        )
        previous_recall = torch.cat(
            (torch.zeros(1, dtype=torch.float64), recall[:-1])
        )
        benefit_average_precision = float(
            ((recall - previous_recall) * precision).sum().item()
        )

    centered_score = score - score.mean()
    centered_gain = gain - gain.mean()
    correlation_denominator = torch.sqrt(
        centered_score.square().sum() * centered_gain.square().sum()
    )
    score_gain_pearson = (
        float(
            (
                (centered_score * centered_gain).sum()
                / correlation_denominator
            ).item()
        )
        if float(correlation_denominator.item()) > 0.0
        else None
    )

    top_prevalence = None
    if positive_count > 0:
        top_positive = sorted_positive[:positive_count]
        top_gain = sorted_gain[:positive_count]
        top_true_positive = int(top_positive.sum().item())
        total_positive_gain = float(gain.clamp_min(0.0).sum().item())
        top_prevalence = {
            "selected_count": positive_count,
            "selected_positive_count": top_true_positive,
            "positive_precision": float(top_true_positive / positive_count),
            "positive_recall": float(top_true_positive / positive_count),
            "positive_gain_capture": float(
                top_gain.clamp_min(0.0).sum().item()
                / max(total_positive_gain, 1.0e-12)
            ),
            "net_gain": float(top_gain.sum().item()),
        }
    best_net = boundary_net.max()
    if float(best_net.item()) <= 0.0:
        max_score = float(sorted_score[0].item())
        max_net = summarize_selection(
            torch.zeros_like(score, dtype=torch.bool),
            threshold=max_score + max(1.0e-6, abs(max_score) * 1.0e-6),
        )
    else:
        best_candidates = boundary_idx[boundary_net == best_net]
        best_idx = int(best_candidates.min().item())
        included = best_idx + 1
        cutoff = float(sorted_score[best_idx].item())
        if included < int(sorted_score.numel()):
            threshold = 0.5 * (cutoff + float(sorted_score[included].item()))
        else:
            threshold = cutoff - max(1.0e-6, abs(cutoff) * 1.0e-6)
        max_net = summarize_selection(score > threshold, threshold=threshold)

    nonnegative_recall = _select_recall_constrained_risk_threshold(
        score_n=score,
        gain_n=gain,
        block_n=torch.zeros_like(gain, dtype=torch.long),
        min_gain_cost_ratio=1.0,
        min_block_net_gain=0.0,
    )
    return {
        "status": "ok",
        "fixed_threshold": float(fixed_threshold),
        "total_count": int(score.numel()),
        "positive_count": positive_count,
        "positive_rate": float(positive.to(torch.float64).mean().item()),
        "positive_score_mean": float(score[positive].mean().item()) if positive_count else None,
        "negative_score_mean": (
            float(score[~positive].mean().item())
            if positive_count < int(score.numel())
            else None
        ),
        "benefit_auroc": benefit_auroc,
        "benefit_average_precision": benefit_average_precision,
        "score_gain_pearson": score_gain_pearson,
        "top_prevalence": top_prevalence,
        "fixed": fixed,
        "max_net_gain": max_net,
        "max_recall_nonnegative": nonnegative_recall,
    }


def _loss_gradient_overlap_summary(
    *,
    reference_loss: torch.Tensor,
    term_losses: Dict[str, torch.Tensor],
    named_parameters: Iterable[Tuple[str, torch.Tensor]],
    parameter_groups: Optional[Dict[str, Tuple[str, ...]]] = None,
) -> Dict[str, object]:
    """Compare loss gradients without mutating ``parameter.grad`` buffers."""
    if reference_loss.ndim != 0:
        raise ValueError("gradient-overlap reference loss must be scalar.")
    parameters = [
        (str(name), parameter)
        for name, parameter in named_parameters
        if parameter.requires_grad
    ]
    if not parameters:
        return {"status": "no_trainable_parameters", "groups": {}}
    for term_name, term_loss in term_losses.items():
        if term_loss.ndim != 0:
            raise ValueError(
                f"gradient-overlap term {term_name} must be scalar."
            )
    if parameter_groups is None:
        parameter_groups = {"all": ("",)}

    parameter_tensors = [parameter for _, parameter in parameters]
    reference_gradients = torch.autograd.grad(
        reference_loss,
        parameter_tensors,
        retain_graph=True,
        allow_unused=True,
    )
    term_gradients = {
        term_name: torch.autograd.grad(
            term_loss,
            parameter_tensors,
            retain_graph=True,
            allow_unused=True,
        )
        for term_name, term_loss in term_losses.items()
    }

    groups: Dict[str, object] = {}
    for group_name, prefixes in parameter_groups.items():
        indices = [
            index
            for index, (name, _) in enumerate(parameters)
            if any(name.startswith(prefix) for prefix in prefixes)
        ]
        reference_sq = sum(
            float(reference_gradients[index].detach().double().square().sum().item())
            for index in indices
            if reference_gradients[index] is not None
        )
        reference_norm = math.sqrt(reference_sq)
        term_rows = {}
        for term_name, gradients in term_gradients.items():
            term_sq = sum(
                float(gradients[index].detach().double().square().sum().item())
                for index in indices
                if gradients[index] is not None
            )
            dot = sum(
                float(
                    (
                        reference_gradients[index].detach().double()
                        * gradients[index].detach().double()
                    ).sum().item()
                )
                for index in indices
                if reference_gradients[index] is not None
                and gradients[index] is not None
            )
            term_norm = math.sqrt(term_sq)
            cosine = (
                float(dot / (reference_norm * term_norm))
                if reference_norm > 0.0 and term_norm > 0.0
                else None
            )
            term_rows[term_name] = {
                "gradient_l2": float(term_norm),
                "dot_with_reference": float(dot),
                "cosine_with_reference": cosine,
            }
        groups[str(group_name)] = {
            "parameter_count": int(
                sum(parameters[index][1].numel() for index in indices)
            ),
            "reference_gradient_l2": float(reference_norm),
            "terms": term_rows,
        }
    return {"status": "ok", "groups": groups}


def _temporal_group_dro_incremental_loss(
    *,
    incremental_loss_bk: torch.Tensor,
    query_index_b: torch.Tensor,
    train_window_count: int,
    cluster_weight_k: torch.Tensor,
    num_domains: int,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Smooth worst-domain loss over chronological incremental-utility groups."""
    if incremental_loss_bk.ndim != 2:
        raise ValueError("temporal group DRO loss must have shape [B,K].")
    batch, clusters = map(int, incremental_loss_bk.shape)
    if tuple(query_index_b.reshape(-1).shape) != (batch,):
        raise ValueError("temporal group DRO query indices must have shape [B].")
    if tuple(cluster_weight_k.reshape(-1).shape) != (clusters,):
        raise ValueError("temporal group DRO cluster weights must have shape [K].")
    if int(train_window_count) <= 0:
        raise ValueError("temporal group DRO train_window_count must be positive.")
    if int(num_domains) <= 1:
        raise ValueError("temporal group DRO requires at least two domains.")
    if float(temperature) < 0.0:
        raise ValueError("temporal group DRO temperature must be nonnegative.")

    query_index = query_index_b.reshape(-1).to(
        device=incremental_loss_bk.device,
        dtype=torch.long,
    )
    domain_id = torch.div(
        query_index * int(num_domains),
        int(train_window_count),
        rounding_mode="floor",
    ).clamp(0, int(num_domains) - 1)
    sample_incremental_loss = reduce_cluster_metric(
        incremental_loss_bk,
        cluster_weight_k.to(
            device=incremental_loss_bk.device,
            dtype=incremental_loss_bk.dtype,
        ),
    )
    present_domain_ids = torch.unique(domain_id, sorted=True)
    domain_losses = torch.stack(
        [
            sample_incremental_loss[domain_id == domain].mean()
            for domain in present_domain_ids
        ]
    )
    if float(temperature) == 0.0:
        worst_domain_loss = domain_losses.max()
    else:
        tau = float(temperature)
        worst_domain_loss = tau * (
            torch.logsumexp(domain_losses / tau, dim=0)
            - math.log(int(domain_losses.numel()))
        )
    return worst_domain_loss, present_domain_ids, domain_losses


@torch.no_grad()
def _select_recall_constrained_risk_threshold_by_penalty(
    *,
    score_n: torch.Tensor,
    gain_n: torch.Tensor,
    block_n: torch.Tensor,
    penalty_n: torch.Tensor,
    penalty_names: List[str],
    min_gain_cost_ratio: float = 1.0,
    min_block_net_gain: float = 0.0,
    no_adoption_threshold: float = 1.0,
) -> Dict[str, object]:
    """Fit independent temporal utility cutoffs for the selected penalty."""
    score = score_n.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
    gain = gain_n.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
    block = block_n.detach().reshape(-1).to(dtype=torch.long, device="cpu")
    penalty = penalty_n.detach().reshape(-1).to(dtype=torch.long, device="cpu")
    if not (score.numel() == gain.numel() == block.numel() == penalty.numel()):
        raise ValueError("per-penalty risk calibration tensor lengths must match.")
    if len(penalty_names) <= 0:
        raise ValueError("per-penalty risk calibration requires penalty names.")
    if int(penalty.numel()) > 0 and bool(
        ((penalty < 0) | (penalty >= len(penalty_names))).any().item()
    ):
        raise ValueError("per-penalty risk calibration received an invalid penalty index.")

    thresholds: List[float] = []
    selection_by_penalty: Dict[str, object] = {}
    for penalty_idx, penalty_name in enumerate(penalty_names):
        penalty_mask = penalty == int(penalty_idx)
        selection = _select_recall_constrained_risk_threshold(
            score_n=score[penalty_mask],
            gain_n=gain[penalty_mask],
            block_n=block[penalty_mask],
            min_gain_cost_ratio=min_gain_cost_ratio,
            min_block_net_gain=min_block_net_gain,
        )
        if selection["status"] == "empty":
            selection = {
                **selection,
                "threshold": float(no_adoption_threshold),
            }
        threshold = float(selection["threshold"])
        thresholds.append(threshold)
        selection_by_penalty[str(penalty_name)] = selection

    finite = torch.isfinite(score) & torch.isfinite(gain)
    threshold_p = torch.as_tensor(thresholds, dtype=torch.float64)
    selected = finite & (score > threshold_p.index_select(0, penalty))
    selected_gain = gain[selected]
    positive_gain = float(selected_gain.clamp_min(0.0).sum().item())
    negative_cost = float((-selected_gain).clamp_min(0.0).sum().item())
    selected_positive_count = int((selected_gain > 0.0).sum().item())
    total_positive_count = int((finite & (gain > 0.0)).sum().item())
    block_count = int(block.max().item()) + 1 if int(block.numel()) > 0 else 0
    block_net_gain = [
        float(gain[selected & (block == block_idx)].sum().item())
        for block_idx in range(block_count)
    ]
    return {
        "status": "ok" if int(score.numel()) > 0 else "empty",
        "threshold_by_penalty": {
            str(name): float(thresholds[idx])
            for idx, name in enumerate(penalty_names)
        },
        "selection_by_penalty": selection_by_penalty,
        "selected_count": int(selected.sum().item()),
        "selected_positive_count": selected_positive_count,
        "positive_recall": float(
            selected_positive_count / max(total_positive_count, 1)
        ),
        "positive_gain": positive_gain,
        "negative_cost": negative_cost,
        "gain_cost_ratio": float(positive_gain / max(negative_cost, 1.0e-12)),
        "net_gain": float(positive_gain - negative_cost),
        "block_net_gain": block_net_gain,
    }


@torch.no_grad()
def _causal_patch_regime_descriptor(x_bcl: torch.Tensor) -> torch.Tensor:
    """Target-free per-channel level/shape descriptor for support checks."""
    if x_bcl.ndim != 3:
        raise ValueError("causal patch regime descriptor expects [B,C,L].")
    eps = 1.0e-6
    mean = x_bcl.mean(dim=-1)
    std = x_bcl.std(dim=-1, unbiased=False).clamp_min(eps)
    d1 = x_bcl.diff(dim=-1)
    mad1 = d1.abs().mean(dim=-1).clamp_min(eps)
    if int(x_bcl.shape[-1]) >= 3:
        mad2 = d1.diff(dim=-1).abs().mean(dim=-1).clamp_min(eps)
    else:
        mad2 = torch.full_like(mad1, eps)
    endpoint = (x_bcl[..., -1] - x_bcl[..., 0]) / std
    last_centered = (x_bcl[..., -1] - mean) / std
    return torch.stack(
        [
            mean,
            std.log(),
            mad1.log(),
            mad2.log(),
            endpoint,
            last_centered,
        ],
        dim=-1,
    )


@torch.no_grad()
def _causal_patch_scale_features(
    x_bcl: torch.Tensor,
    base_bch: torch.Tensor,
    candidate_delta_bcqr: torch.Tensor,
) -> torch.Tensor:
    """Target-free input/base/candidate descriptors for online scale fitting."""
    if x_bcl.ndim != 3 or base_bch.ndim != 3 or candidate_delta_bcqr.ndim != 4:
        raise ValueError(
            "causal patch scale features expect x/base/delta as [B,C,L], [B,C,H], [B,C,Q,R]."
        )
    batch, channels, patches, patch_len = map(int, candidate_delta_bcqr.shape)
    if tuple(x_bcl.shape[:2]) != (batch, channels):
        raise ValueError("causal patch scale input batch/channel shape does not match.")
    if tuple(base_bch.shape) != (batch, channels, patches * patch_len):
        raise ValueError("causal patch scale base shape does not match patch layout.")
    eps = 1.0e-6
    input_mean = x_bcl.mean(dim=-1, keepdim=True)
    input_std = x_bcl.std(dim=-1, unbiased=False, keepdim=True).clamp_min(eps)
    base_patch = base_bch.reshape(batch, channels, patches, patch_len)
    time = torch.linspace(
        -1.0,
        1.0,
        patch_len,
        device=x_bcl.device,
        dtype=x_bcl.dtype,
    ).view(1, 1, 1, patch_len)
    time_energy = time.square().mean().clamp_min(eps)

    def patch_summary(value_bcqr: torch.Tensor, *, center_on_input: bool) -> torch.Tensor:
        mean = value_bcqr.mean(dim=-1, keepdim=True)
        std = value_bcqr.std(dim=-1, unbiased=False, keepdim=True).clamp_min(eps)
        centered = value_bcqr - mean
        slope = (centered * time).mean(dim=-1, keepdim=True) / time_energy
        d1 = value_bcqr.diff(dim=-1)
        mad1 = d1.abs().mean(dim=-1, keepdim=True).clamp_min(eps)
        mad2 = (
            d1.diff(dim=-1).abs().mean(dim=-1, keepdim=True).clamp_min(eps)
            if patch_len >= 3
            else torch.full_like(mad1, eps)
        )
        endpoint = value_bcqr[..., -1:] - value_bcqr[..., :1]
        scale = input_std.unsqueeze(2)
        mean_feature = (
            (mean - input_mean.unsqueeze(2)) / scale
            if center_on_input
            else mean / scale
        )
        return torch.cat(
            [
                mean_feature,
                torch.log(std / scale),
                slope / scale,
                mad1 / scale,
                mad2 / scale,
                endpoint / scale,
            ],
            dim=-1,
        )

    regime = _causal_patch_regime_descriptor(x_bcl).unsqueeze(2).expand(
        -1,
        -1,
        patches,
        -1,
    )
    base_features = patch_summary(base_patch, center_on_input=True)
    delta_features = patch_summary(candidate_delta_bcqr, center_on_input=False)
    progress = torch.linspace(
        -1.0,
        1.0,
        patches,
        device=x_bcl.device,
        dtype=x_bcl.dtype,
    ).view(1, 1, patches, 1).expand(batch, channels, -1, -1)
    position = torch.cat(
        [progress, torch.sin(torch.pi * progress), torch.cos(torch.pi * progress)],
        dim=-1,
    )
    features = torch.cat([regime, base_features, delta_features, position], dim=-1)
    return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).clamp(-8.0, 8.0)


@torch.no_grad()
def _walk_forward_patch_reliability_metrics(
    *,
    train_time_n: torch.Tensor,
    train_gain_ncq: torch.Tensor,
    eval_time_n: torch.Tensor,
    eval_base_mse_ncq: torch.Tensor,
    eval_candidate_mse_ncq: torch.Tensor,
    eval_base_mae_ncq: torch.Tensor,
    eval_candidate_mae_ncq: torch.Tensor,
    active_channel_mask_c: torch.Tensor,
    train_penalty_ncq: Optional[torch.Tensor] = None,
    eval_penalty_ncq: Optional[torch.Tensor] = None,
    train_regime_ncf: Optional[torch.Tensor] = None,
    eval_regime_ncf: Optional[torch.Tensor] = None,
    max_abs_regime_z: Optional[float] = None,
    train_cross_ncq: Optional[torch.Tensor] = None,
    train_delta_sq_ncq: Optional[torch.Tensor] = None,
    eval_cross_ncq: Optional[torch.Tensor] = None,
    eval_delta_sq_ncq: Optional[torch.Tensor] = None,
    eval_base_residual_ncqr: Optional[torch.Tensor] = None,
    eval_candidate_delta_ncqr: Optional[torch.Tensor] = None,
    train_scale_feature_ncqf: Optional[torch.Tensor] = None,
    eval_scale_feature_ncqf: Optional[torch.Tensor] = None,
    scale_mode: str = "binary",
    max_scale: float = 1.0,
    scale_consensus_blocks: int = 1,
    feature_ridge: float = 0.1,
    feature_update_blocks: int = 6,
    patch_label_delay_q: Optional[torch.Tensor] = None,
    label_delay: int,
    lookback_windows: int,
    min_history_windows: int,
    history_stride: int = 1,
    min_mean_gain: float = 0.0,
    temporal_blocks: int = 0,
) -> Dict[str, object]:
    """Causal fixed-candidate adoption from matured rolling patch utility."""
    train_gain = train_gain_ncq.detach().to(dtype=torch.float64, device="cpu")
    base_mse = eval_base_mse_ncq.detach().to(dtype=torch.float64, device="cpu")
    candidate_mse = eval_candidate_mse_ncq.detach().to(
        dtype=torch.float64,
        device="cpu",
    )
    base_mae = eval_base_mae_ncq.detach().to(dtype=torch.float64, device="cpu")
    candidate_mae = eval_candidate_mae_ncq.detach().to(
        dtype=torch.float64,
        device="cpu",
    )
    train_time = train_time_n.detach().reshape(-1).to(dtype=torch.long, device="cpu")
    eval_time = eval_time_n.detach().reshape(-1).to(dtype=torch.long, device="cpu")
    active_c = active_channel_mask_c.detach().reshape(-1).to(
        dtype=torch.bool,
        device="cpu",
    )
    if train_gain.ndim != 3:
        raise ValueError("walk-forward train gain must have shape [N,C,Q].")
    expected_eval_shape = tuple(base_mse.shape)
    if base_mse.ndim != 3 or any(
        tuple(value.shape) != expected_eval_shape
        for value in (candidate_mse, base_mae, candidate_mae)
    ):
        raise ValueError("walk-forward eval error tensors must share shape [N,C,Q].")
    if int(train_gain.shape[1]) != int(base_mse.shape[1]) or int(
        train_gain.shape[2]
    ) != int(base_mse.shape[2]):
        raise ValueError("walk-forward train/eval channel-patch shapes must match.")
    if int(train_time.numel()) != int(train_gain.shape[0]):
        raise ValueError("walk-forward train times must match train windows.")
    if int(eval_time.numel()) != int(base_mse.shape[0]):
        raise ValueError("walk-forward eval times must match eval windows.")
    if int(active_c.numel()) != int(base_mse.shape[1]):
        raise ValueError("walk-forward active mask must match channel count.")
    train_penalty = None
    eval_penalty = None
    if (train_penalty_ncq is None) != (eval_penalty_ncq is None):
        raise ValueError(
            "walk-forward expert conditioning requires both train and eval penalties."
        )
    if train_penalty_ncq is not None and eval_penalty_ncq is not None:
        train_penalty = train_penalty_ncq.detach().to(
            dtype=torch.long,
            device="cpu",
        )
        eval_penalty = eval_penalty_ncq.detach().to(
            dtype=torch.long,
            device="cpu",
        )
        if tuple(train_penalty.shape) != tuple(train_gain.shape):
            raise ValueError(
                "walk-forward train penalty identities must match train gains."
            )
        if tuple(eval_penalty.shape) != tuple(base_mse.shape):
            raise ValueError(
                "walk-forward eval penalty identities must match eval errors."
            )
    scale_policy = str(scale_mode).strip().lower()
    if scale_policy not in {"binary", "least_squares", "feature_ridge"}:
        raise ValueError(
            "walk-forward scale_mode must be binary, least_squares, or feature_ridge."
        )
    if float(max_scale) <= 0.0:
        raise ValueError("walk-forward max_scale must be positive.")
    consensus_blocks = int(scale_consensus_blocks)
    if consensus_blocks <= 0:
        raise ValueError("walk-forward scale_consensus_blocks must be positive.")
    train_cross = None
    train_delta_sq = None
    eval_cross = None
    eval_delta_sq = None
    eval_base_residual = None
    eval_candidate_delta = None
    if scale_policy in {"least_squares", "feature_ridge"}:
        scale_tensors = (
            train_cross_ncq,
            train_delta_sq_ncq,
            eval_cross_ncq,
            eval_delta_sq_ncq,
        )
        if any(value is None for value in scale_tensors):
            raise ValueError(
                "walk-forward least_squares requires cross and delta-square tensors."
            )
        train_cross = train_cross_ncq.detach().to(dtype=torch.float64, device="cpu")
        train_delta_sq = train_delta_sq_ncq.detach().to(dtype=torch.float64, device="cpu")
        eval_cross = eval_cross_ncq.detach().to(dtype=torch.float64, device="cpu")
        eval_delta_sq = eval_delta_sq_ncq.detach().to(dtype=torch.float64, device="cpu")
        if (eval_base_residual_ncqr is None) != (eval_candidate_delta_ncqr is None):
            raise ValueError(
                "walk-forward eval base residual and candidate delta must be provided together."
            )
        if eval_base_residual_ncqr is not None:
            eval_base_residual = eval_base_residual_ncqr.detach().to(
                dtype=torch.float64,
                device="cpu",
            )
            eval_candidate_delta = eval_candidate_delta_ncqr.detach().to(
                dtype=torch.float64,
                device="cpu",
            )
        if any(
            tuple(value.shape) != tuple(train_gain.shape)
            for value in (train_cross, train_delta_sq)
        ):
            raise ValueError("walk-forward train scale statistics must match train gains.")
        if any(
            tuple(value.shape) != tuple(base_mse.shape)
            for value in (eval_cross, eval_delta_sq)
        ):
            raise ValueError("walk-forward eval scale statistics must match eval errors.")
        if eval_base_residual is not None and eval_candidate_delta is not None:
            expected_patch_prefix = tuple(base_mse.shape)
            if (
                eval_base_residual.ndim != 4
                or eval_candidate_delta.ndim != 4
                or tuple(eval_base_residual.shape) != tuple(eval_candidate_delta.shape)
                or tuple(eval_base_residual.shape[:3]) != expected_patch_prefix
            ):
                raise ValueError(
                    "walk-forward eval patch residual tensors must share shape [N,C,Q,R]."
                )
    train_scale_feature = None
    eval_scale_feature = None
    if scale_policy == "feature_ridge":
        if train_scale_feature_ncqf is None or eval_scale_feature_ncqf is None:
            raise ValueError("walk-forward feature_ridge requires train and eval features.")
        train_scale_feature = train_scale_feature_ncqf.detach().to(
            dtype=torch.float64,
            device="cpu",
        )
        eval_scale_feature = eval_scale_feature_ncqf.detach().to(
            dtype=torch.float64,
            device="cpu",
        )
        if train_scale_feature.ndim != 4 or eval_scale_feature.ndim != 4:
            raise ValueError("walk-forward scale features must have shape [N,C,Q,F].")
        if tuple(train_scale_feature.shape[:3]) != tuple(train_gain.shape):
            raise ValueError("walk-forward train scale features do not match gains.")
        if tuple(eval_scale_feature.shape[:3]) != tuple(base_mse.shape):
            raise ValueError("walk-forward eval scale features do not match errors.")
        if int(train_scale_feature.shape[-1]) != int(eval_scale_feature.shape[-1]):
            raise ValueError("walk-forward train/eval scale feature dimensions must match.")
        if float(feature_ridge) < 0.0:
            raise ValueError("walk-forward feature_ridge must be nonnegative.")
        if int(feature_update_blocks) <= 0:
            raise ValueError("walk-forward feature_update_blocks must be positive.")
    regime_enabled = max_abs_regime_z is not None
    train_regime = None
    eval_regime = None
    if regime_enabled:
        if train_regime_ncf is None or eval_regime_ncf is None:
            raise ValueError(
                "walk-forward regime support requires train and eval descriptors."
            )
        train_regime = train_regime_ncf.detach().to(
            dtype=torch.float64,
            device="cpu",
        )
        eval_regime = eval_regime_ncf.detach().to(
            dtype=torch.float64,
            device="cpu",
        )
        if train_regime.ndim != 3 or eval_regime.ndim != 3:
            raise ValueError("walk-forward regime descriptors must have shape [N,C,F].")
        if tuple(train_regime.shape[:2]) != tuple(train_gain.shape[:2]):
            raise ValueError("walk-forward train regime descriptors do not match gains.")
        if tuple(eval_regime.shape[:2]) != tuple(base_mse.shape[:2]):
            raise ValueError("walk-forward eval regime descriptors do not match errors.")
        if int(train_regime.shape[-1]) != int(eval_regime.shape[-1]):
            raise ValueError("walk-forward train/eval regime feature dimensions must match.")
        if float(max_abs_regime_z) <= 0.0:
            raise ValueError("walk-forward max_abs_regime_z must be positive.")
    delay = int(label_delay)
    lookback = int(lookback_windows)
    min_history = int(min_history_windows)
    stride = int(history_stride)
    if delay <= 0:
        raise ValueError("walk-forward label_delay must be positive to prevent leakage.")
    patch_delay = None
    if patch_label_delay_q is not None:
        patch_delay = patch_label_delay_q.detach().reshape(-1).to(
            dtype=torch.long,
            device="cpu",
        )
        if int(patch_delay.numel()) != int(base_mse.shape[2]):
            raise ValueError(
                "walk-forward patch label delays must match forecast patch count."
            )
        if bool((patch_delay <= 0).any().item()):
            raise ValueError("walk-forward patch label delays must be positive.")
        if regime_enabled:
            raise ValueError(
                "walk-forward patch-specific delays do not support regime z filtering."
            )
        if scale_policy == "feature_ridge":
            raise ValueError(
                "walk-forward feature_ridge currently requires a shared label delay."
            )
    if lookback <= 0 or min_history <= 0 or min_history > lookback:
        raise ValueError(
            "walk-forward history requires 0 < min_history_windows <= lookback_windows."
        )
    if stride <= 0:
        raise ValueError("walk-forward history_stride must be positive.")
    if stride > 1:
        if regime_enabled:
            raise ValueError(
                "walk-forward strided history does not support regime z filtering."
            )
        if scale_policy == "feature_ridge":
            raise ValueError(
                "walk-forward feature_ridge currently requires history_stride=1."
            )
        if consensus_blocks != 1:
            raise ValueError(
                "walk-forward scale consensus currently requires history_stride=1."
            )
    if train_penalty is not None and (
        regime_enabled or scale_policy != "binary"
    ):
        raise ValueError(
            "walk-forward expert conditioning currently supports binary scale "
            "without regime filtering."
        )
    if int(eval_time.numel()) == 0:
        return {
            "status": "empty",
            "label_delay": delay,
            "lookback_windows": lookback,
            "min_history_windows": min_history,
            "history_stride": stride,
            "test_read": False,
        }
    if bool((eval_time[1:] < eval_time[:-1]).any().item()):
        raise ValueError("walk-forward eval times must be chronological.")

    eval_gain = base_mse - candidate_mse
    history_time = torch.cat([train_time, eval_time], dim=0)
    history_gain = torch.cat([train_gain, eval_gain], dim=0)
    order = torch.argsort(history_time, stable=True)
    history_time = history_time.index_select(0, order)
    history_gain = history_gain.index_select(0, order)
    history_penalty = (
        torch.cat([train_penalty, eval_penalty], dim=0).index_select(0, order)
        if train_penalty is not None and eval_penalty is not None
        else None
    )
    if patch_delay is None:
        maturity_cutoff = (eval_time - delay).view(-1, 1).expand(
            -1,
            int(base_mse.shape[2]),
        )
    else:
        maturity_cutoff = eval_time.view(-1, 1) - patch_delay.view(1, -1)
    history_start = maturity_cutoff - lookback + 1

    def gather_patch_cumulative(
        cumulative_tcq: torch.Tensor,
        index_nq: torch.Tensor,
    ) -> torch.Tensor:
        cumulative_tqc = cumulative_tcq.permute(0, 2, 1)
        patch_index_nq = torch.arange(
            int(index_nq.shape[1]),
            dtype=torch.long,
        ).view(1, -1).expand_as(index_nq)
        return cumulative_tqc[index_nq, patch_index_nq].permute(0, 2, 1)

    def rolling_patch_history_sum(
        history_value_tcq: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        if stride == 1:
            right_index = torch.searchsorted(
                history_time,
                maturity_cutoff.contiguous(),
                right=True,
            )
            left_index = torch.searchsorted(
                history_time,
                history_start.contiguous(),
                right=False,
            )
            cumulative = torch.cat(
                [
                    torch.zeros_like(history_value_tcq[:1]),
                    history_value_tcq.cumsum(dim=0),
                ],
                dim=0,
            )
            rolling_sum = gather_patch_cumulative(
                cumulative,
                right_index,
            ) - gather_patch_cumulative(cumulative, left_index)
            return rolling_sum, right_index - left_index, left_index, right_index

        rolling_sum = torch.zeros(
            (int(eval_time.numel()), *history_value_tcq.shape[1:]),
            dtype=history_value_tcq.dtype,
            device=history_value_tcq.device,
        )
        rolling_count = torch.zeros(
            (int(eval_time.numel()), int(base_mse.shape[2])),
            dtype=torch.long,
            device=history_value_tcq.device,
        )
        history_phase = torch.remainder(history_time, stride)
        eval_phase = torch.remainder(eval_time, stride)
        for phase_value in torch.unique(eval_phase).tolist():
            phase = int(phase_value)
            eval_index = torch.nonzero(
                eval_phase == phase,
                as_tuple=False,
            ).reshape(-1)
            history_index = torch.nonzero(
                history_phase == phase,
                as_tuple=False,
            ).reshape(-1)
            phase_time = history_time.index_select(0, history_index)
            phase_value_tcq = history_value_tcq.index_select(0, history_index)
            phase_cumulative = torch.cat(
                [
                    torch.zeros_like(phase_value_tcq[:1]),
                    phase_value_tcq.cumsum(dim=0),
                ],
                dim=0,
            )
            phase_cutoff = maturity_cutoff.index_select(0, eval_index)
            phase_start = history_start.index_select(0, eval_index)
            phase_right = torch.searchsorted(
                phase_time,
                phase_cutoff,
                right=True,
            )
            phase_left = torch.searchsorted(
                phase_time,
                phase_start,
                right=False,
            )
            phase_sum = gather_patch_cumulative(
                phase_cumulative,
                phase_right,
            ) - gather_patch_cumulative(phase_cumulative, phase_left)
            rolling_sum.index_copy_(0, eval_index, phase_sum)
            rolling_count.index_copy_(0, eval_index, phase_right - phase_left)
        return rolling_sum, rolling_count, None, None

    if history_penalty is None or eval_penalty is None:
        history_sum, history_count, left, right = rolling_patch_history_sum(
            history_gain
        )
        history_count_ncq = history_count.unsqueeze(1).expand_as(history_sum)
    else:
        history_sum = torch.zeros_like(base_mse)
        history_count_ncq = torch.zeros_like(base_mse)
        left = None
        right = None
        for penalty_idx in torch.unique(eval_penalty).tolist():
            penalty_value = int(penalty_idx)
            history_match = (history_penalty == penalty_value).to(
                dtype=history_gain.dtype
            )
            penalty_sum, _, penalty_left, penalty_right = (
                rolling_patch_history_sum(history_gain * history_match)
            )
            penalty_count, _, _, _ = rolling_patch_history_sum(history_match)
            eval_match = eval_penalty == penalty_value
            history_sum = torch.where(eval_match, penalty_sum, history_sum)
            history_count_ncq = torch.where(
                eval_match,
                penalty_count,
                history_count_ncq,
            )
            if left is None:
                left = penalty_left
                right = penalty_right
        history_count = history_count_ncq
    history_mean = history_sum / history_count_ncq.clamp_min(1).to(
        dtype=history_sum.dtype
    )
    regime_support_nc = torch.ones(
        (int(eval_time.numel()), int(base_mse.shape[1])),
        dtype=torch.bool,
    )
    regime_max_z_nc = torch.zeros_like(regime_support_nc, dtype=torch.float64)
    if regime_enabled:
        assert train_regime is not None and eval_regime is not None
        assert left is not None and right is not None
        regime_right = right[:, 0]
        regime_left = left[:, 0]
        regime_history_count = history_count[:, 0]
        history_regime = torch.cat([train_regime, eval_regime], dim=0).index_select(
            0,
            order,
        )
        cumulative_regime = torch.cat(
            [
                torch.zeros(
                    (1, *history_regime.shape[1:]),
                    dtype=history_regime.dtype,
                ),
                history_regime.cumsum(dim=0),
            ],
            dim=0,
        )
        cumulative_regime_sq = torch.cat(
            [
                torch.zeros(
                    (1, *history_regime.shape[1:]),
                    dtype=history_regime.dtype,
                ),
                history_regime.square().cumsum(dim=0),
            ],
            dim=0,
        )
        regime_count = regime_history_count.clamp_min(1).to(dtype=torch.float64).view(
            -1,
            1,
            1,
        )
        regime_sum = cumulative_regime.index_select(
            0,
            regime_right,
        ) - cumulative_regime.index_select(0, regime_left)
        regime_sq_sum = cumulative_regime_sq.index_select(
            0,
            regime_right,
        ) - cumulative_regime_sq.index_select(0, regime_left)
        regime_mean = regime_sum / regime_count
        regime_var = (regime_sq_sum / regime_count - regime_mean.square()).clamp_min(0.0)
        regime_std = regime_var.sqrt().clamp_min(1.0e-4)
        regime_max_z_nc = ((eval_regime - regime_mean) / regime_std).abs().amax(dim=-1)
        regime_support_nc = regime_max_z_nc <= float(max_abs_regime_z)
    eligible_ncq = (
        (history_count_ncq >= min_history)
        & active_c.view(1, -1, 1)
        & regime_support_nc.unsqueeze(-1)
    )
    if scale_policy == "binary":
        route_ncq = eligible_ncq & (history_mean > float(min_mean_gain))
        scale_ncq = route_ncq.to(dtype=torch.float64)
        selected_mse = torch.where(route_ncq, candidate_mse, base_mse)
        selected_mae = torch.where(route_ncq, candidate_mae, base_mae)
    else:
        assert (
            train_cross is not None
            and train_delta_sq is not None
            and eval_cross is not None
            and eval_delta_sq is not None
        )
        history_cross = torch.cat([train_cross, eval_cross], dim=0).index_select(
            0,
            order,
        )
        history_delta_sq = torch.cat(
            [train_delta_sq, eval_delta_sq],
            dim=0,
        ).index_select(0, order)
        rolling_cross, _, _, _ = rolling_patch_history_sum(history_cross)
        rolling_delta_sq, _, _, _ = rolling_patch_history_sum(history_delta_sq)
        if scale_policy == "feature_ridge":
            assert train_scale_feature is not None and eval_scale_feature is not None
            assert left is not None and right is not None
            history_scale_feature = torch.cat(
                [train_scale_feature, eval_scale_feature],
                dim=0,
            ).index_select(0, order)
            scale_ncq = torch.zeros_like(eval_cross)
            update_blocks = min(
                int(feature_update_blocks),
                int(eval_time.numel()),
            )
            feature_dim = int(history_scale_feature.shape[-1])
            for update_idx in range(update_blocks):
                eval_start_idx = update_idx * int(eval_time.numel()) // update_blocks
                eval_end_idx = (
                    (update_idx + 1) * int(eval_time.numel()) // update_blocks
                )
                if eval_end_idx <= eval_start_idx:
                    continue
                history_left_idx = int(left[eval_start_idx, 0].item())
                history_right_idx = int(right[eval_start_idx, 0].item())
                if history_right_idx <= history_left_idx:
                    continue
                fit_feature = history_scale_feature[
                    history_left_idx:history_right_idx
                ]
                fit_cross = history_cross[history_left_idx:history_right_idx]
                fit_delta_sq = history_delta_sq[
                    history_left_idx:history_right_idx
                ]
                feature_mean = fit_feature.mean(dim=0, keepdim=True)
                feature_std = fit_feature.std(
                    dim=0,
                    unbiased=False,
                    keepdim=True,
                ).clamp_min(1.0e-4)
                fit_standard = ((fit_feature - feature_mean) / feature_std).clamp(
                    -6.0,
                    6.0,
                )
                fit_design = torch.cat(
                    [torch.ones_like(fit_standard[..., :1]), fit_standard],
                    dim=-1,
                )
                sample_count = max(int(fit_design.shape[0]), 1)
                normal = torch.einsum(
                    "ncqf,ncqg,ncq->cqfg",
                    fit_design,
                    fit_design,
                    fit_delta_sq,
                ) / float(sample_count)
                rhs = torch.einsum(
                    "ncqf,ncq->cqf",
                    fit_design,
                    fit_cross,
                ) / float(sample_count)
                ridge_diag = torch.full(
                    (feature_dim + 1,),
                    float(feature_ridge),
                    dtype=normal.dtype,
                )
                ridge_diag[0] = max(float(feature_ridge) * 0.01, 1.0e-8)
                normal = normal + torch.diag(ridge_diag).view(
                    1,
                    1,
                    feature_dim + 1,
                    feature_dim + 1,
                )
                coef = torch.linalg.solve(normal, rhs.unsqueeze(-1)).squeeze(-1)
                eval_feature = eval_scale_feature[eval_start_idx:eval_end_idx]
                eval_standard = ((eval_feature - feature_mean) / feature_std).clamp(
                    -6.0,
                    6.0,
                )
                eval_design = torch.cat(
                    [torch.ones_like(eval_standard[..., :1]), eval_standard],
                    dim=-1,
                )
                scale_ncq[eval_start_idx:eval_end_idx] = torch.einsum(
                    "ncqf,cqf->ncq",
                    eval_design,
                    coef,
                ).clamp(0.0, float(max_scale))
        elif consensus_blocks == 1:
            scale_ncq = (
                rolling_cross / rolling_delta_sq.clamp_min(1.0e-12)
            ).clamp(0.0, float(max_scale))
        else:
            assert left is not None and right is not None
            cumulative_cross = torch.cat(
                [torch.zeros_like(history_cross[:1]), history_cross.cumsum(dim=0)],
                dim=0,
            )
            cumulative_delta_sq = torch.cat(
                [
                    torch.zeros_like(history_delta_sq[:1]),
                    history_delta_sq.cumsum(dim=0),
                ],
                dim=0,
            )
            consensus_scale_parts = []
            history_count_long = history_count.clamp_min(1)
            for consensus_idx in range(consensus_blocks):
                block_left = left + torch.div(
                    history_count_long * consensus_idx,
                    consensus_blocks,
                    rounding_mode="floor",
                )
                block_right = left + torch.div(
                    history_count_long * (consensus_idx + 1),
                    consensus_blocks,
                    rounding_mode="floor",
                )
                block_cross = gather_patch_cumulative(
                    cumulative_cross,
                    block_right,
                ) - gather_patch_cumulative(cumulative_cross, block_left)
                block_delta_sq = gather_patch_cumulative(
                    cumulative_delta_sq,
                    block_right,
                ) - gather_patch_cumulative(cumulative_delta_sq, block_left)
                block_scale = (
                    block_cross / block_delta_sq.clamp_min(1.0e-12)
                ).clamp(0.0, float(max_scale))
                consensus_scale_parts.append(block_scale)
            scale_ncq = torch.stack(consensus_scale_parts, dim=0).amin(dim=0)
        scale_ncq = scale_ncq * eligible_ncq.to(dtype=scale_ncq.dtype)
        route_ncq = scale_ncq > 0.0
        selected_mse = (
            base_mse
            - 2.0 * scale_ncq * eval_cross
            + scale_ncq.square() * eval_delta_sq
        ).clamp_min(0.0)
        selected_mae = (
            (
                eval_base_residual
                + scale_ncq.unsqueeze(-1) * eval_candidate_delta
            ).abs().mean(dim=-1)
            if eval_base_residual is not None and eval_candidate_delta is not None
            else None
        )
    oracle_mse = torch.minimum(base_mse, candidate_mse)
    beneficial = eval_gain > 0.0
    selected_beneficial = route_ncq & beneficial
    selected_count = int(route_ncq.sum().item())
    beneficial_count = int((beneficial & active_c.view(1, -1, 1)).sum().item())
    true_positive_count = int(selected_beneficial.sum().item())

    def mean_metric(value: torch.Tensor) -> float:
        return float(value.mean().item())

    base_mse_mean = mean_metric(base_mse)
    selected_mse_mean = mean_metric(selected_mse)
    base_mae_mean = mean_metric(base_mae)
    selected_mae_mean = mean_metric(selected_mae) if selected_mae is not None else None
    blocks = max(0, min(int(temporal_blocks), int(eval_time.numel())))
    block_rows: List[Dict[str, object]] = []
    if blocks > 1:
        for block_idx in range(blocks):
            start = block_idx * int(eval_time.numel()) // blocks
            end = (block_idx + 1) * int(eval_time.numel()) // blocks
            if end <= start:
                continue
            block_base = mean_metric(base_mse[start:end])
            block_selected = mean_metric(selected_mse[start:end])
            block_rows.append(
                {
                    "block": int(block_idx),
                    "start_window": int(start),
                    "end_window": int(end),
                    "base_mse": block_base,
                    "selected_mse": block_selected,
                    "gain_pct": 100.0
                    * (block_base - block_selected)
                    / max(block_base, 1.0e-12),
                    "adoption_rate": float(route_ncq[start:end].to(torch.float64).mean().item()),
                    "mean_scale": float(scale_ncq[start:end].mean().item()),
                    "regime_support_rate": float(
                        regime_support_nc[start:end].to(torch.float64).mean().item()
                    ),
                    "history_count_min": int(history_count[start:end].min().item()),
                    "history_count_max": int(history_count[start:end].max().item()),
                    "per_channel": [
                        {
                            "channel_index": int(channel),
                            "base_mse": mean_metric(
                                base_mse[start:end, channel]
                            ),
                            "selected_mse": mean_metric(
                                selected_mse[start:end, channel]
                            ),
                            "gain_pct": 100.0
                            * (
                                mean_metric(base_mse[start:end, channel])
                                - mean_metric(selected_mse[start:end, channel])
                            )
                            / max(
                                mean_metric(base_mse[start:end, channel]),
                                1.0e-12,
                            ),
                            "mean_scale": float(
                                scale_ncq[start:end, channel].mean().item()
                            ),
                        }
                        for channel in range(int(base_mse.shape[1]))
                    ],
                }
            )

    return {
        "status": "ok",
        "test_read": False,
        "policy": (
            "matured_blockwise_feature_ridge_scale_channel_patch"
            if scale_policy == "feature_ridge"
            else (
                "matured_rolling_least_squares_scale_channel_patch"
                if scale_policy == "least_squares"
                else (
                    "matured_rolling_mean_gain_channel_patch_selected_expert"
                    if history_penalty is not None
                    else "matured_rolling_mean_gain_channel_patch"
                )
            )
        ),
        "scale_mode": scale_policy,
        "max_scale": float(max_scale),
        "scale_consensus_blocks": int(consensus_blocks),
        "feature_ridge": float(feature_ridge),
        "feature_update_blocks": int(feature_update_blocks),
        "label_delay": delay,
        "patch_label_delay": (
            None if patch_delay is None else [int(value) for value in patch_delay.tolist()]
        ),
        "lookback_windows": lookback,
        "min_history_windows": min_history,
        "history_stride": stride,
        "min_mean_gain": float(min_mean_gain),
        "max_abs_regime_z": (
            None if max_abs_regime_z is None else float(max_abs_regime_z)
        ),
        "base_mse": base_mse_mean,
        "selected_mse": selected_mse_mean,
        "mse_gain_pct": 100.0
        * (base_mse_mean - selected_mse_mean)
        / max(base_mse_mean, 1.0e-12),
        "base_mae": base_mae_mean,
        "selected_mae": selected_mae_mean,
        "mae_gain_pct": (
            100.0
            * (base_mae_mean - selected_mae_mean)
            / max(base_mae_mean, 1.0e-12)
            if selected_mae_mean is not None
            else None
        ),
        "oracle_mse": mean_metric(oracle_mse),
        "adoption_rate": float(route_ncq.to(torch.float64).mean().item()),
        "mean_scale": float(scale_ncq.mean().item()),
        "adoption_recall": float(true_positive_count / max(beneficial_count, 1)),
        "adoption_precision": float(true_positive_count / max(selected_count, 1)),
        "history_count_min": int(history_count.min().item()),
        "history_count_max": int(history_count.max().item()),
        "regime_support_rate": float(
            regime_support_nc.to(torch.float64).mean().item()
        ),
        "regime_max_z_mean": float(regime_max_z_nc.mean().item()),
        "adoption_rate_by_channel": [
            float(route_ncq[:, channel].to(torch.float64).mean().item())
            for channel in range(int(route_ncq.shape[1]))
        ],
        "mean_scale_by_channel": [
            float(scale_ncq[:, channel].mean().item())
            for channel in range(int(scale_ncq.shape[1]))
        ],
        "per_channel": [
            {
                "channel_index": int(channel),
                "base_mse": mean_metric(base_mse[:, channel]),
                "selected_mse": mean_metric(selected_mse[:, channel]),
                "gain_pct": 100.0
                * (
                    mean_metric(base_mse[:, channel])
                    - mean_metric(selected_mse[:, channel])
                )
                / max(mean_metric(base_mse[:, channel]), 1.0e-12),
                "base_mae": mean_metric(base_mae[:, channel]),
                "selected_mae": (
                    mean_metric(selected_mae[:, channel])
                    if selected_mae is not None
                    else None
                ),
                "mean_scale": float(scale_ncq[:, channel].mean().item()),
            }
            for channel in range(int(base_mse.shape[1]))
        ],
        "per_channel_patch": [
            [
                {
                    "channel_index": int(channel),
                    "patch_index": int(patch),
                    "base_mse": mean_metric(base_mse[:, channel, patch]),
                    "selected_mse": mean_metric(selected_mse[:, channel, patch]),
                    "gain_pct": 100.0
                    * (
                        mean_metric(base_mse[:, channel, patch])
                        - mean_metric(selected_mse[:, channel, patch])
                    )
                    / max(mean_metric(base_mse[:, channel, patch]), 1.0e-12),
                    "mean_scale": float(scale_ncq[:, channel, patch].mean().item()),
                }
                for patch in range(int(base_mse.shape[2]))
            ]
            for channel in range(int(base_mse.shape[1]))
        ],
        "regime_support_rate_by_channel": [
            float(regime_support_nc[:, channel].to(torch.float64).mean().item())
            for channel in range(int(regime_support_nc.shape[1]))
        ],
        "temporal_blocks": block_rows,
    }


@torch.no_grad()
def _causal_expert_feedback_ridge_metrics(
    *,
    train_time_n: torch.Tensor,
    train_base_mse_ncq: torch.Tensor,
    train_candidate_mse_ncqp: torch.Tensor,
    train_base_mae_ncq: torch.Tensor,
    train_candidate_mae_ncqp: torch.Tensor,
    train_score_ncqp: torch.Tensor,
    eval_time_n: torch.Tensor,
    eval_base_mse_ncq: torch.Tensor,
    eval_candidate_mse_ncqp: torch.Tensor,
    eval_base_mae_ncq: torch.Tensor,
    eval_candidate_mae_ncqp: torch.Tensor,
    eval_score_ncqp: torch.Tensor,
    active_channel_mask_c: torch.Tensor,
    label_delay: int,
    lookback_windows: int,
    min_history_windows: int,
    history_stride: int = 1,
    ridge: float = 0.1,
    target_clip: float = 2.0,
    temporal_blocks: int = 0,
) -> Dict[str, object]:
    """Fit dual utility from current input score plus matured expert feedback."""
    train_time = train_time_n.detach().reshape(-1).to(dtype=torch.long, device="cpu")
    eval_time = eval_time_n.detach().reshape(-1).to(dtype=torch.long, device="cpu")
    train_base_mse = train_base_mse_ncq.detach().to(dtype=torch.float64, device="cpu")
    train_candidate_mse = train_candidate_mse_ncqp.detach().to(
        dtype=torch.float64,
        device="cpu",
    )
    train_base_mae = train_base_mae_ncq.detach().to(dtype=torch.float64, device="cpu")
    train_candidate_mae = train_candidate_mae_ncqp.detach().to(
        dtype=torch.float64,
        device="cpu",
    )
    train_score = train_score_ncqp.detach().to(dtype=torch.float64, device="cpu")
    eval_base_mse = eval_base_mse_ncq.detach().to(dtype=torch.float64, device="cpu")
    eval_candidate_mse = eval_candidate_mse_ncqp.detach().to(
        dtype=torch.float64,
        device="cpu",
    )
    eval_base_mae = eval_base_mae_ncq.detach().to(dtype=torch.float64, device="cpu")
    eval_candidate_mae = eval_candidate_mae_ncqp.detach().to(
        dtype=torch.float64,
        device="cpu",
    )
    eval_score = eval_score_ncqp.detach().to(dtype=torch.float64, device="cpu")
    active_c = active_channel_mask_c.detach().reshape(-1).to(
        dtype=torch.bool,
        device="cpu",
    )
    train_shape = tuple(train_candidate_mse.shape)
    eval_shape = tuple(eval_candidate_mse.shape)
    if train_candidate_mse.ndim != 4 or eval_candidate_mse.ndim != 4:
        raise ValueError("expert feedback candidates must have shape [N,C,Q,P].")
    if train_shape[1:] != eval_shape[1:]:
        raise ValueError("expert feedback train/eval candidate shapes must match.")
    if any(
        tuple(value.shape) != train_shape
        for value in (train_candidate_mae, train_score)
    ) or any(
        tuple(value.shape) != eval_shape
        for value in (eval_candidate_mae, eval_score)
    ):
        raise ValueError("expert feedback candidate errors and scores must match.")
    if any(
        tuple(value.shape) != train_shape[:3]
        for value in (train_base_mse, train_base_mae)
    ) or any(
        tuple(value.shape) != eval_shape[:3]
        for value in (eval_base_mse, eval_base_mae)
    ):
        raise ValueError("expert feedback base errors must have shape [N,C,Q].")
    if int(train_time.numel()) != train_shape[0] or int(eval_time.numel()) != eval_shape[0]:
        raise ValueError("expert feedback times must match their split windows.")
    if int(active_c.numel()) != train_shape[1]:
        raise ValueError("expert feedback active mask must match channels.")
    delay = int(label_delay)
    lookback = int(lookback_windows)
    min_history = int(min_history_windows)
    stride = int(history_stride)
    if delay <= 0:
        raise ValueError("expert feedback label_delay must be positive.")
    if lookback <= 0 or min_history <= 0 or min_history > lookback:
        raise ValueError(
            "expert feedback requires 0 < min_history_windows <= lookback_windows."
        )
    if stride <= 0:
        raise ValueError("expert feedback history_stride must be positive.")
    if float(ridge) < 0.0 or float(target_clip) <= 0.0:
        raise ValueError("expert feedback ridge must be nonnegative and target_clip positive.")

    train_mse_gain = (
        (train_base_mse.unsqueeze(-1) - train_candidate_mse)
        / train_base_mse.unsqueeze(-1).clamp_min(1.0e-6)
    ).clamp(-float(target_clip), float(target_clip))
    train_mae_gain = (
        (train_base_mae.unsqueeze(-1) - train_candidate_mae)
        / train_base_mae.unsqueeze(-1).clamp_min(1.0e-6)
    ).clamp(-float(target_clip), float(target_clip))
    eval_mse_gain = (
        (eval_base_mse.unsqueeze(-1) - eval_candidate_mse)
        / eval_base_mse.unsqueeze(-1).clamp_min(1.0e-6)
    ).clamp(-float(target_clip), float(target_clip))
    eval_mae_gain = (
        (eval_base_mae.unsqueeze(-1) - eval_candidate_mae)
        / eval_base_mae.unsqueeze(-1).clamp_min(1.0e-6)
    ).clamp(-float(target_clip), float(target_clip))

    def causal_mean(
        history_time_n: torch.Tensor,
        history_value_ncqp: torch.Tensor,
        query_time_n: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        order = torch.argsort(history_time_n, stable=True)
        history_time_sorted = history_time_n.index_select(0, order)
        history_value = history_value_ncqp.index_select(0, order)
        result = torch.zeros(
            (int(query_time_n.numel()), *history_value.shape[1:]),
            dtype=history_value.dtype,
        )
        count = torch.zeros(int(query_time_n.numel()), dtype=torch.long)
        history_phase = torch.remainder(history_time_sorted, stride)
        query_phase = torch.remainder(query_time_n, stride)
        for phase_value in torch.unique(query_phase).tolist():
            phase = int(phase_value)
            query_index = torch.nonzero(query_phase == phase, as_tuple=False).reshape(-1)
            history_index = torch.nonzero(
                history_phase == phase,
                as_tuple=False,
            ).reshape(-1)
            phase_time = history_time_sorted.index_select(0, history_index)
            phase_value_ncqp = history_value.index_select(0, history_index)
            cumulative = torch.cat(
                [
                    torch.zeros_like(phase_value_ncqp[:1]),
                    phase_value_ncqp.cumsum(dim=0),
                ],
                dim=0,
            )
            query_time = query_time_n.index_select(0, query_index)
            cutoff = query_time - delay
            start = cutoff - lookback + 1
            right = torch.searchsorted(phase_time, cutoff, right=True)
            left = torch.searchsorted(phase_time, start, right=False)
            phase_count = right - left
            result.index_copy_(
                0,
                query_index,
                (cumulative.index_select(0, right) - cumulative.index_select(0, left))
                / phase_count.clamp_min(1).to(dtype=torch.float64).view(-1, 1, 1, 1),
            )
            count.index_copy_(0, query_index, phase_count)
        return result, count

    train_history_mse, train_history_count = causal_mean(
        train_time,
        train_mse_gain,
        train_time,
    )
    train_history_mae, _ = causal_mean(
        train_time,
        train_mae_gain,
        train_time,
    )
    combined_time = torch.cat([train_time, eval_time], dim=0)
    eval_history_mse, eval_history_count = causal_mean(
        combined_time,
        torch.cat([train_mse_gain, eval_mse_gain], dim=0),
        eval_time,
    )
    eval_history_mae, _ = causal_mean(
        combined_time,
        torch.cat([train_mae_gain, eval_mae_gain], dim=0),
        eval_time,
    )
    train_eligible = (
        (train_history_count >= min_history).view(-1, 1, 1)
        & active_c.view(1, -1, 1)
    )
    eval_eligible = (
        (eval_history_count >= min_history).view(-1, 1, 1)
        & active_c.view(1, -1, 1)
    )
    penalty_count = int(train_shape[-1])
    predicted_mse = torch.zeros_like(eval_candidate_mse)
    predicted_mae = torch.zeros_like(eval_candidate_mae)
    coefficient_norms: List[float] = []

    def raw_features(
        score_ncqp: torch.Tensor,
        history_mse_ncqp: torch.Tensor,
        history_mae_ncqp: torch.Tensor,
        penalty: int,
    ) -> torch.Tensor:
        score_ncq = score_ncqp[..., penalty]
        mse_ncq = history_mse_ncqp[..., penalty]
        mae_ncq = history_mae_ncqp[..., penalty]
        return torch.stack(
            [
                score_ncq,
                mse_ncq,
                mae_ncq,
                score_ncq * mse_ncq,
                score_ncq * mae_ncq,
                mse_ncq * mae_ncq,
            ],
            dim=-1,
        )

    for penalty in range(penalty_count):
        fit_raw = raw_features(
            train_score,
            train_history_mse,
            train_history_mae,
            penalty,
        )
        eval_raw = raw_features(
            eval_score,
            eval_history_mse,
            eval_history_mae,
            penalty,
        )
        weight = train_eligible.to(dtype=torch.float64)
        sample_count = weight.sum(dim=0).clamp_min(1.0)
        feature_mean = (
            (fit_raw * weight.unsqueeze(-1)).sum(dim=0)
            / sample_count.unsqueeze(-1)
        )
        centered = fit_raw - feature_mean.unsqueeze(0)
        feature_var = (
            (centered.square() * weight.unsqueeze(-1)).sum(dim=0)
            / sample_count.unsqueeze(-1)
        )
        feature_std = feature_var.sqrt().clamp_min(1.0e-4)
        fit_standard = (centered / feature_std.unsqueeze(0)).clamp(-8.0, 8.0)
        eval_standard = (
            (eval_raw - feature_mean.unsqueeze(0)) / feature_std.unsqueeze(0)
        ).clamp(-8.0, 8.0)
        fit_design = torch.cat(
            [torch.ones_like(fit_standard[..., :1]), fit_standard],
            dim=-1,
        )
        eval_design = torch.cat(
            [torch.ones_like(eval_standard[..., :1]), eval_standard],
            dim=-1,
        )
        normal = torch.einsum(
            "ncqf,ncqg,ncq->cqfg",
            fit_design,
            fit_design,
            weight,
        ) / sample_count.unsqueeze(-1).unsqueeze(-1)
        ridge_diag = torch.full(
            (int(fit_design.shape[-1]),),
            float(ridge),
            dtype=torch.float64,
        )
        ridge_diag[0] = max(float(ridge) * 0.01, 1.0e-8)
        normal = normal + torch.diag(ridge_diag).view(
            1,
            1,
            int(fit_design.shape[-1]),
            int(fit_design.shape[-1]),
        )

        def fit_target(target_ncq: torch.Tensor) -> torch.Tensor:
            rhs = torch.einsum(
                "ncqf,ncq,ncq->cqf",
                fit_design,
                target_ncq,
                weight,
            ) / sample_count.unsqueeze(-1)
            return torch.linalg.solve(normal, rhs.unsqueeze(-1)).squeeze(-1)

        coef_mse = fit_target(train_mse_gain[..., penalty])
        coef_mae = fit_target(train_mae_gain[..., penalty])
        predicted_mse[..., penalty] = torch.einsum(
            "ncqf,cqf->ncq",
            eval_design,
            coef_mse,
        )
        predicted_mae[..., penalty] = torch.einsum(
            "ncqf,cqf->ncq",
            eval_design,
            coef_mae,
        )
        coefficient_norms.append(
            float((coef_mse.square().mean() + coef_mae.square().mean()).sqrt().item())
        )
    predicted_dual = torch.minimum(predicted_mse, predicted_mae)
    predicted_dual = predicted_dual.masked_fill(
        ~eval_eligible.unsqueeze(-1),
        -torch.inf,
    )
    selected_score, selected_penalty = predicted_dual.max(dim=-1)
    route = (selected_score > 0.0) & eval_eligible
    selected_index = selected_penalty.unsqueeze(-1)
    chosen_mse = eval_candidate_mse.gather(dim=-1, index=selected_index).squeeze(-1)
    chosen_mae = eval_candidate_mae.gather(dim=-1, index=selected_index).squeeze(-1)
    selected_mse = torch.where(route, chosen_mse, eval_base_mse)
    selected_mae = torch.where(route, chosen_mae, eval_base_mae)
    selected_positive = route & (chosen_mse < eval_base_mse)
    selected_dual_positive = selected_positive & (chosen_mae < eval_base_mae)
    route_count = int(route.sum().item())

    def mean(value: torch.Tensor) -> float:
        return float(value.mean().item())

    def gain_pct(base: torch.Tensor, selected: torch.Tensor) -> float:
        base_mean = mean(base)
        return 100.0 * (base_mean - mean(selected)) / max(base_mean, 1.0e-12)

    blocks = max(0, min(int(temporal_blocks), int(eval_time.numel())))
    block_rows: List[Dict[str, object]] = []
    if blocks > 1:
        for block_idx in range(blocks):
            start_idx = block_idx * int(eval_time.numel()) // blocks
            end_idx = (block_idx + 1) * int(eval_time.numel()) // blocks
            if end_idx <= start_idx:
                continue
            block_rows.append(
                {
                    "block": int(block_idx),
                    "start_window": int(start_idx),
                    "end_window": int(end_idx),
                    "mse_gain_pct": gain_pct(
                        eval_base_mse[start_idx:end_idx],
                        selected_mse[start_idx:end_idx],
                    ),
                    "mae_gain_pct": gain_pct(
                        eval_base_mae[start_idx:end_idx],
                        selected_mae[start_idx:end_idx],
                    ),
                    "route_rate": float(
                        route[start_idx:end_idx].to(torch.float64).mean().item()
                    ),
                }
            )
    return {
        "status": "ok",
        "test_read": False,
        "policy": "ridge_dual_utility_from_input_score_and_matured_expert_feedback",
        "label_delay": delay,
        "lookback_windows": lookback,
        "min_history_windows": min_history,
        "history_stride": stride,
        "ridge": float(ridge),
        "target_clip": float(target_clip),
        "feature_names": [
            "input_gate_score",
            "history_mse_gain",
            "history_mae_gain",
            "score_x_history_mse",
            "score_x_history_mae",
            "history_mse_x_mae",
        ],
        "base_mse": mean(eval_base_mse),
        "selected_mse": mean(selected_mse),
        "mse_gain_pct": gain_pct(eval_base_mse, selected_mse),
        "base_mae": mean(eval_base_mae),
        "selected_mae": mean(selected_mae),
        "mae_gain_pct": gain_pct(eval_base_mae, selected_mae),
        "route_rate": float(route.to(torch.float64).mean().item()),
        "selected_mse_precision": float(
            selected_positive.sum().item() / max(route_count, 1)
        ),
        "selected_dual_precision": float(
            selected_dual_positive.sum().item() / max(route_count, 1)
        ),
        "history_count_min": int(eval_history_count.min().item()),
        "history_count_max": int(eval_history_count.max().item()),
        "coefficient_norm_by_penalty": coefficient_norms,
        "per_channel": [
            {
                "channel_index": int(channel),
                "mse_gain_pct": gain_pct(
                    eval_base_mse[:, channel],
                    selected_mse[:, channel],
                ),
                "mae_gain_pct": gain_pct(
                    eval_base_mae[:, channel],
                    selected_mae[:, channel],
                ),
                "route_rate": float(
                    route[:, channel].to(torch.float64).mean().item()
                ),
            }
            for channel in range(int(eval_base_mse.shape[1]))
        ],
        "temporal_blocks": block_rows,
    }


@torch.no_grad()
def _walk_forward_expert_reliability_rerank_metrics(
    *,
    train_time_n: torch.Tensor,
    train_gain_ncqp: torch.Tensor,
    eval_time_n: torch.Tensor,
    eval_base_mse_ncq: torch.Tensor,
    eval_candidate_mse_ncqp: torch.Tensor,
    eval_base_mae_ncq: torch.Tensor,
    eval_candidate_mae_ncqp: torch.Tensor,
    eval_score_ncqp: torch.Tensor,
    active_channel_mask_c: torch.Tensor,
    label_delay: int,
    lookback_windows: int,
    min_history_windows: int,
    history_stride: int = 1,
    min_mean_gain: float = 0.0,
    require_positive_input_score: bool = True,
    temporal_blocks: int = 0,
) -> Dict[str, object]:
    """Causally mask unreliable experts, then preserve input-gate ordering.

    Every expert's counterfactual error is recorded only after its target has
    matured.  Historical gain never ranks the deployed action: it only marks
    experts as reliable.  The current input-conditioned gate score chooses among
    reliable experts.  A history-ranked result is reported solely as a diagnostic
    causal upper bound.
    """
    train_time = train_time_n.detach().reshape(-1).to(dtype=torch.long, device="cpu")
    eval_time = eval_time_n.detach().reshape(-1).to(dtype=torch.long, device="cpu")
    train_gain = train_gain_ncqp.detach().to(dtype=torch.float64, device="cpu")
    base_mse = eval_base_mse_ncq.detach().to(dtype=torch.float64, device="cpu")
    candidate_mse = eval_candidate_mse_ncqp.detach().to(
        dtype=torch.float64,
        device="cpu",
    )
    base_mae = eval_base_mae_ncq.detach().to(dtype=torch.float64, device="cpu")
    candidate_mae = eval_candidate_mae_ncqp.detach().to(
        dtype=torch.float64,
        device="cpu",
    )
    score = eval_score_ncqp.detach().to(dtype=torch.float64, device="cpu")
    active_c = active_channel_mask_c.detach().reshape(-1).to(
        dtype=torch.bool,
        device="cpu",
    )
    if train_gain.ndim != 4:
        raise ValueError("expert rerank train gain must have shape [N,C,Q,P].")
    eval_shape = tuple(candidate_mse.shape)
    if candidate_mse.ndim != 4 or any(
        tuple(value.shape) != eval_shape for value in (candidate_mae, score)
    ):
        raise ValueError(
            "expert rerank candidate errors and scores must share shape [N,C,Q,P]."
        )
    if tuple(base_mse.shape) != eval_shape[:3] or tuple(base_mae.shape) != eval_shape[:3]:
        raise ValueError("expert rerank base errors must have shape [N,C,Q].")
    if tuple(train_gain.shape[1:]) != eval_shape[1:]:
        raise ValueError("expert rerank train/eval candidate shapes must match.")
    if int(train_time.numel()) != int(train_gain.shape[0]):
        raise ValueError("expert rerank train times must match train gains.")
    if int(eval_time.numel()) != int(candidate_mse.shape[0]):
        raise ValueError("expert rerank eval times must match eval candidates.")
    if int(active_c.numel()) != int(candidate_mse.shape[1]):
        raise ValueError("expert rerank active mask must match channel count.")
    delay = int(label_delay)
    lookback = int(lookback_windows)
    min_history = int(min_history_windows)
    stride = int(history_stride)
    if delay <= 0:
        raise ValueError("expert rerank label_delay must be positive.")
    if lookback <= 0 or min_history <= 0 or min_history > lookback:
        raise ValueError(
            "expert rerank requires 0 < min_history_windows <= lookback_windows."
        )
    if stride <= 0:
        raise ValueError("expert rerank history_stride must be positive.")
    if int(eval_time.numel()) == 0:
        return {
            "status": "empty",
            "test_read": False,
            "label_delay": delay,
        }
    if bool((eval_time[1:] < eval_time[:-1]).any().item()):
        raise ValueError("expert rerank eval times must be chronological.")

    eval_gain = base_mse.unsqueeze(-1) - candidate_mse
    history_time = torch.cat([train_time, eval_time], dim=0)
    history_gain = torch.cat([train_gain, eval_gain], dim=0)
    order = torch.argsort(history_time, stable=True)
    history_time = history_time.index_select(0, order)
    history_gain = history_gain.index_select(0, order)
    history_sum = torch.zeros_like(candidate_mse)
    history_count = torch.zeros(int(eval_time.numel()), dtype=torch.long)
    history_phase = torch.remainder(history_time, stride)
    eval_phase = torch.remainder(eval_time, stride)
    for phase_value in torch.unique(eval_phase).tolist():
        phase = int(phase_value)
        eval_index = torch.nonzero(eval_phase == phase, as_tuple=False).reshape(-1)
        history_index = torch.nonzero(history_phase == phase, as_tuple=False).reshape(-1)
        phase_time = history_time.index_select(0, history_index)
        phase_gain = history_gain.index_select(0, history_index)
        cumulative = torch.cat(
            [torch.zeros_like(phase_gain[:1]), phase_gain.cumsum(dim=0)],
            dim=0,
        )
        phase_eval_time = eval_time.index_select(0, eval_index)
        cutoff = phase_eval_time - delay
        start = cutoff - lookback + 1
        right = torch.searchsorted(phase_time, cutoff, right=True)
        left = torch.searchsorted(phase_time, start, right=False)
        history_sum.index_copy_(
            0,
            eval_index,
            cumulative.index_select(0, right) - cumulative.index_select(0, left),
        )
        history_count.index_copy_(0, eval_index, right - left)
    history_mean = history_sum / history_count.clamp_min(1).to(
        dtype=torch.float64
    ).view(-1, 1, 1, 1)
    finite_score = torch.nan_to_num(
        score,
        nan=-torch.inf,
        posinf=torch.finfo(torch.float64).max,
        neginf=-torch.inf,
    )
    reliable = (
        (history_count >= min_history).view(-1, 1, 1, 1)
        & active_c.view(1, -1, 1, 1)
        & (history_mean > float(min_mean_gain))
    )
    deploy_eligible = reliable
    if bool(require_positive_input_score):
        deploy_eligible = deploy_eligible & (finite_score > 0.0)
    masked_gate_score = finite_score.masked_fill(~deploy_eligible, -torch.inf)
    selected_penalty = masked_gate_score.argmax(dim=-1)
    route = deploy_eligible.any(dim=-1)
    selected_index = selected_penalty.unsqueeze(-1)
    chosen_mse = candidate_mse.gather(dim=-1, index=selected_index).squeeze(-1)
    chosen_mae = candidate_mae.gather(dim=-1, index=selected_index).squeeze(-1)
    selected_mse = torch.where(route, chosen_mse, base_mse)
    selected_mae = torch.where(route, chosen_mae, base_mae)

    original_penalty = finite_score.argmax(dim=-1)
    original_route = (
        finite_score.max(dim=-1).values > 0.0
        if bool(require_positive_input_score)
        else torch.ones_like(route)
    ) & active_c.view(1, -1, 1)
    original_mse = torch.where(
        original_route,
        candidate_mse.gather(
            dim=-1,
            index=original_penalty.unsqueeze(-1),
        ).squeeze(-1),
        base_mse,
    )
    original_mae = torch.where(
        original_route,
        candidate_mae.gather(
            dim=-1,
            index=original_penalty.unsqueeze(-1),
        ).squeeze(-1),
        base_mae,
    )

    history_masked = history_mean.masked_fill(~reliable, -torch.inf)
    history_penalty = history_masked.argmax(dim=-1)
    history_route = reliable.any(dim=-1)
    history_mse = torch.where(
        history_route,
        candidate_mse.gather(
            dim=-1,
            index=history_penalty.unsqueeze(-1),
        ).squeeze(-1),
        base_mse,
    )
    history_mae = torch.where(
        history_route,
        candidate_mae.gather(
            dim=-1,
            index=history_penalty.unsqueeze(-1),
        ).squeeze(-1),
        base_mae,
    )
    oracle_mse = torch.minimum(base_mse, candidate_mse.amin(dim=-1))
    selected_mse_gain = base_mse - chosen_mse
    selected_mae_gain = base_mae - chosen_mae
    selected_positive = route & (selected_mse_gain > 0.0)
    selected_dual_positive = selected_positive & (selected_mae_gain > 0.0)
    route_count = int(route.sum().item())
    positive_count = int(selected_positive.sum().item())
    dual_positive_count = int(selected_dual_positive.sum().item())
    original_positive_prevalence = float(
        (eval_gain > 0.0).to(torch.float64).mean().item()
    )

    def mean(value: torch.Tensor) -> float:
        return float(value.mean().item())

    def gain_pct(base: torch.Tensor, selected: torch.Tensor) -> float:
        base_mean = mean(base)
        return 100.0 * (base_mean - mean(selected)) / max(base_mean, 1.0e-12)

    blocks = max(0, min(int(temporal_blocks), int(eval_time.numel())))
    block_rows: List[Dict[str, object]] = []
    if blocks > 1:
        for block_idx in range(blocks):
            start_idx = block_idx * int(eval_time.numel()) // blocks
            end_idx = (block_idx + 1) * int(eval_time.numel()) // blocks
            if end_idx <= start_idx:
                continue
            block_rows.append(
                {
                    "block": int(block_idx),
                    "start_window": int(start_idx),
                    "end_window": int(end_idx),
                    "mse_gain_pct": gain_pct(
                        base_mse[start_idx:end_idx],
                        selected_mse[start_idx:end_idx],
                    ),
                    "mae_gain_pct": gain_pct(
                        base_mae[start_idx:end_idx],
                        selected_mae[start_idx:end_idx],
                    ),
                    "route_rate": float(
                        route[start_idx:end_idx].to(torch.float64).mean().item()
                    ),
                }
            )
    penalty_count = int(candidate_mse.shape[-1])
    routed_penalties = selected_penalty[route]
    selected_counts = torch.bincount(
        routed_penalties,
        minlength=penalty_count,
    ).to(torch.float64)
    return {
        "status": "ok",
        "test_read": False,
        "policy": "matured_same_expert_reliability_mask_then_input_gate_order",
        "label_delay": delay,
        "lookback_windows": lookback,
        "min_history_windows": min_history,
        "history_stride": stride,
        "min_mean_gain": float(min_mean_gain),
        "require_positive_input_score": bool(require_positive_input_score),
        "base_mse": mean(base_mse),
        "selected_mse": mean(selected_mse),
        "mse_gain_pct": gain_pct(base_mse, selected_mse),
        "base_mae": mean(base_mae),
        "selected_mae": mean(selected_mae),
        "mae_gain_pct": gain_pct(base_mae, selected_mae),
        "original_gate_mse": mean(original_mse),
        "original_gate_mse_gain_pct": gain_pct(base_mse, original_mse),
        "original_gate_mae": mean(original_mae),
        "original_gate_mae_gain_pct": gain_pct(base_mae, original_mae),
        "history_ranked_diagnostic_mse": mean(history_mse),
        "history_ranked_diagnostic_mse_gain_pct": gain_pct(base_mse, history_mse),
        "history_ranked_diagnostic_mae": mean(history_mae),
        "history_ranked_diagnostic_mae_gain_pct": gain_pct(base_mae, history_mae),
        "oracle_mse": mean(oracle_mse),
        "oracle_mse_gain_pct": gain_pct(base_mse, oracle_mse),
        "route_rate": float(route.to(torch.float64).mean().item()),
        "fallback_rate": float(
            (route & (selected_penalty != original_penalty)).to(torch.float64).mean().item()
        ),
        "fallback_rate_given_route": float(
            (route & (selected_penalty != original_penalty)).sum().item()
            / max(route_count, 1)
        ),
        "selected_mse_precision": float(positive_count / max(route_count, 1)),
        "selected_dual_precision": float(dual_positive_count / max(route_count, 1)),
        "all_candidate_positive_prevalence": original_positive_prevalence,
        "history_count_min": int(history_count.min().item()),
        "history_count_max": int(history_count.max().item()),
        "selected_penalty_rate": [
            float(value / max(route_count, 1)) for value in selected_counts.tolist()
        ],
        "per_channel": [
            {
                "channel_index": int(channel),
                "mse_gain_pct": gain_pct(
                    base_mse[:, channel],
                    selected_mse[:, channel],
                ),
                "mae_gain_pct": gain_pct(
                    base_mae[:, channel],
                    selected_mae[:, channel],
                ),
                "route_rate": float(
                    route[:, channel].to(torch.float64).mean().item()
                ),
            }
            for channel in range(int(base_mse.shape[1]))
        ],
        "temporal_blocks": block_rows,
    }


def _patch_router_hierarchical_recall_loss_terms(
    *,
    base_bch: torch.Tensor,
    candidate_bcpH: torch.Tensor,
    y_bch: torch.Tensor,
    patch_adopt_prob_bcq: torch.Tensor,
    patch_penalty_conditional_probs_bcqp: torch.Tensor,
    patch_penalty_benefit_probs_bcqp: torch.Tensor,
    patch_penalty_utility_scores_bcqp: torch.Tensor,
    cluster_id_c: torch.Tensor,
    K: int,
    patch_penalty_mse_utility_scores_bcqp: Optional[torch.Tensor] = None,
    patch_penalty_mae_utility_scores_bcqp: Optional[torch.Tensor] = None,
    patch_penalty_risk_benefit_probs_bcqp: Optional[torch.Tensor] = None,
    patch_penalty_risk_positive_magnitude_bcqp: Optional[torch.Tensor] = None,
    patch_penalty_risk_negative_magnitude_bcqp: Optional[torch.Tensor] = None,
    patch_penalty_proposal_logits_bcqp: Optional[torch.Tensor] = None,
    patch_penalty_proposal_rescue_logits_bcqp: Optional[torch.Tensor] = None,
    patch_penalty_risk_lower_quantile_scores_bcqp: Optional[torch.Tensor] = None,
    patch_final_adopt_prob_bcq: Optional[torch.Tensor] = None,
    patch_penalty_pairwise_rank_scores_bcqp: Optional[torch.Tensor] = None,
    patch_penalty_proposal_mask_bcqp: Optional[torch.Tensor] = None,
    patch_active_mask_bcq: Optional[torch.Tensor] = None,
    min_abs_improvement: float = 0.0,
    adoption_bce_weight: float = 1.0,
    proposal_bce_weight: float = 1.0,
    proposal_gain_listwise_weight: float = 0.0,
    proposal_rescue_ce_weight: float = 0.0,
    ranking_ce_weight: float = 1.0,
    utility_regression_weight: float = 0.0,
    adoption_recall_weight: float = 1.0,
    false_adopt_weight: float = 1.0,
    penalty_recall_weight: float = 1.0,
    false_penalty_weight: float = 1.0,
    risk_calibration_weight: float = 0.0,
    risk_sign_bce_weight: float = 0.0,
    risk_magnitude_weight: float = 0.0,
    risk_lower_quantile_weight: float = 0.0,
    risk_lower_quantile: float = 0.2,
    selected_utility_policy_weight: float = 0.0,
    selected_adoption_bce_weight: float = 0.0,
    selected_adoption_recall_weight: float = 0.0,
    selected_false_adopt_weight: float = 0.0,
    pairwise_rank_weight: float = 0.0,
    target_adopt_probability: float = 0.8,
    false_adopt_max_probability: float = 0.2,
    target_penalty_probability: float = 0.7,
    false_penalty_max_probability: float = 0.3,
    eps: float = 1.0e-8,
) -> Dict[str, torch.Tensor]:
    """Recall-aware hierarchical supervision without changing PKR expert targets."""
    batch, channels, horizon = base_bch.shape
    patches = int(patch_adopt_prob_bcq.shape[2])
    penalties = int(candidate_bcpH.shape[2])
    expected_patch_shape = (batch, channels, patches)
    expected_penalty_shape = (batch, channels, patches, penalties)
    if patches <= 0 or horizon % patches != 0:
        raise ValueError("hierarchical patch recall requires patch count to divide horizon.")
    if tuple(candidate_bcpH.shape) != (batch, channels, penalties, horizon):
        raise ValueError("hierarchical patch recall candidate shape does not match base prediction.")
    if tuple(patch_adopt_prob_bcq.shape) != expected_patch_shape:
        raise ValueError("hierarchical patch recall adopt probability shape does not match.")
    if tuple(patch_penalty_conditional_probs_bcqp.shape) != expected_penalty_shape:
        raise ValueError("hierarchical patch recall conditional probability shape does not match.")
    if tuple(patch_penalty_benefit_probs_bcqp.shape) != expected_penalty_shape:
        raise ValueError("hierarchical patch recall benefit probability shape does not match.")
    if tuple(patch_penalty_utility_scores_bcqp.shape) != expected_penalty_shape:
        raise ValueError("hierarchical patch recall utility score shape does not match.")
    dual_utility_outputs = (
        patch_penalty_mse_utility_scores_bcqp,
        patch_penalty_mae_utility_scores_bcqp,
    )
    if any(value is not None for value in dual_utility_outputs):
        if not all(value is not None for value in dual_utility_outputs):
            raise ValueError(
                "hierarchical dual signed utility requires both MSE and MAE scores."
            )
        if any(
            tuple(value.shape) != expected_penalty_shape
            for value in dual_utility_outputs
            if value is not None
        ):
            raise ValueError(
                "hierarchical dual signed utility score shape does not match."
            )
    risk_outputs = (
        patch_penalty_risk_benefit_probs_bcqp,
        patch_penalty_risk_positive_magnitude_bcqp,
        patch_penalty_risk_negative_magnitude_bcqp,
    )
    if any(value is not None for value in risk_outputs):
        if not all(value is not None for value in risk_outputs):
            raise ValueError("hierarchical patch risk decomposition requires all three outputs.")
        if any(tuple(value.shape) != expected_penalty_shape for value in risk_outputs):
            raise ValueError("hierarchical patch risk decomposition shape does not match.")
    if (
        patch_penalty_proposal_logits_bcqp is not None
        and tuple(patch_penalty_proposal_logits_bcqp.shape) != expected_penalty_shape
    ):
        raise ValueError("hierarchical patch proposal logits shape does not match.")
    if patch_penalty_proposal_rescue_logits_bcqp is not None:
        if patch_penalty_proposal_logits_bcqp is None:
            raise ValueError("hierarchical patch rescue logits require primary proposal logits.")
        if tuple(patch_penalty_proposal_rescue_logits_bcqp.shape) != expected_penalty_shape:
            raise ValueError("hierarchical patch proposal rescue logits shape does not match.")
    if patch_penalty_risk_lower_quantile_scores_bcqp is not None:
        if tuple(patch_penalty_risk_lower_quantile_scores_bcqp.shape) != expected_penalty_shape:
            raise ValueError("hierarchical patch risk lower-quantile shape does not match.")
        if not 0.0 < float(risk_lower_quantile) < 0.5:
            raise ValueError("hierarchical patch risk lower quantile must be in (0,0.5).")
    if (
        patch_final_adopt_prob_bcq is not None
        and tuple(patch_final_adopt_prob_bcq.shape) != expected_patch_shape
    ):
        raise ValueError("hierarchical patch final adopt probability shape does not match.")
    pairwise_outputs = (
        patch_penalty_pairwise_rank_scores_bcqp,
        patch_penalty_proposal_mask_bcqp,
    )
    if any(value is not None for value in pairwise_outputs):
        if not all(value is not None for value in pairwise_outputs):
            raise ValueError("hierarchical patch pairwise rank requires scores and proposal mask.")
        if any(tuple(value.shape) != expected_penalty_shape for value in pairwise_outputs):
            raise ValueError("hierarchical patch pairwise rank shape does not match.")

    if patch_active_mask_bcq is None:
        active_mask_bcq = torch.ones(
            expected_patch_shape,
            dtype=torch.bool,
            device=base_bch.device,
        )
        masked_support = False
    else:
        if tuple(patch_active_mask_bcq.shape) != expected_patch_shape:
            raise ValueError("hierarchical patch active mask shape does not match.")
        active_mask_bcq = patch_active_mask_bcq.to(
            device=base_bch.device,
            dtype=torch.bool,
        )
        masked_support = True
        if not bool(active_mask_bcq.any().item()):
            raise ValueError("hierarchical patch active mask leaves no supervised samples.")
    active_weight_bcq = active_mask_bcq.to(dtype=patch_adopt_prob_bcq.dtype)
    active_penalty_weight_bcqp = active_weight_bcq.unsqueeze(-1)

    patch_len = horizon // patches
    with torch.no_grad():
        base_error_bcq = (base_bch - y_bch).square().reshape(
            batch,
            channels,
            patches,
            patch_len,
        ).mean(dim=-1)
        candidate_error_bcqp = (candidate_bcpH - y_bch.unsqueeze(2)).square().reshape(
            batch,
            channels,
            penalties,
            patches,
            patch_len,
        ).mean(dim=-1).permute(0, 1, 3, 2)
        improvement_bcqp = base_error_bcq.unsqueeze(-1) - candidate_error_bcqp
        beneficial_bcqp = improvement_bcqp > float(min_abs_improvement)
        adopt_target_bcq = beneficial_bcqp.any(dim=-1)
        best_penalty_bcq = candidate_error_bcqp.argmin(dim=-1)
        error_floor_source = (
            base_error_bcq[active_mask_bcq]
            if masked_support
            else base_error_bcq
        )
        error_floor = error_floor_source.median().clamp_min(1.0e-6) * 0.05
        normalized_gain_bcqp = (
            improvement_bcqp / (base_error_bcq.unsqueeze(-1) + error_floor)
        ).clamp(-1.0, 1.0)
        if patch_penalty_mse_utility_scores_bcqp is not None:
            base_mae_bcq = (base_bch - y_bch).abs().reshape(
                batch,
                channels,
                patches,
                patch_len,
            ).mean(dim=-1)
            candidate_mae_bcqp = (
                (candidate_bcpH - y_bch.unsqueeze(2))
                .abs()
                .reshape(
                    batch,
                    channels,
                    penalties,
                    patches,
                    patch_len,
                )
                .mean(dim=-1)
                .permute(0, 1, 3, 2)
            )
            mae_improvement_bcqp = (
                base_mae_bcq.unsqueeze(-1) - candidate_mae_bcqp
            )
            mae_floor_source = (
                base_mae_bcq[active_mask_bcq]
                if masked_support
                else base_mae_bcq
            )
            mae_floor = mae_floor_source.median().clamp_min(1.0e-6) * 0.05
            normalized_mae_gain_bcqp = (
                mae_improvement_bcqp
                / (base_mae_bcq.unsqueeze(-1) + mae_floor)
            ).clamp(-1.0, 1.0)
        else:
            normalized_mae_gain_bcqp = None

    probability_eps = max(
        float(eps),
        float(torch.finfo(patch_adopt_prob_bcq.dtype).eps),
    )
    adopt_prob = patch_adopt_prob_bcq.clamp(
        probability_eps,
        1.0 - probability_eps,
    )
    conditional_prob = patch_penalty_conditional_probs_bcqp.clamp_min(
        probability_eps
    )
    benefit_prob = patch_penalty_benefit_probs_bcqp.clamp(
        probability_eps,
        1.0 - probability_eps,
    )
    utility_scores = patch_penalty_utility_scores_bcqp.clamp(-1.0, 1.0)
    adopt_target = adopt_target_bcq.to(dtype=adopt_prob.dtype)
    beneficial = beneficial_bcqp.to(dtype=benefit_prob.dtype)
    negative_benefit = 1.0 - beneficial

    adoption_bce_bcq = torch.nn.functional.binary_cross_entropy(
        adopt_prob,
        adopt_target,
        reduction="none",
    )

    positive_count_p = (beneficial * active_penalty_weight_bcqp).sum(
        dim=(0, 1, 2)
    )
    present_penalty_p = positive_count_p > 0.0
    positive_total = positive_count_p.sum().clamp_min(1.0)
    present_count = present_penalty_p.sum().clamp_min(1).to(dtype=benefit_prob.dtype)
    positive_macro_weight_p = torch.where(
        present_penalty_p,
        positive_total / (present_count * positive_count_p.clamp_min(1.0)),
        torch.zeros_like(positive_count_p),
    )
    proposal_positive = -beneficial * benefit_prob.log() * positive_macro_weight_p.view(1, 1, 1, -1)
    proposal_negative = -negative_benefit * (1.0 - benefit_prob).log()
    proposal_bce_bcq = proposal_positive.sum(dim=-1) + proposal_negative.mean(dim=-1)
    proposal_gain_target_bcqp = normalized_gain_bcqp.clamp_min(0.0) * beneficial
    proposal_gain_target_bcqp = (
        proposal_gain_target_bcqp
        / proposal_gain_target_bcqp.sum(dim=-1, keepdim=True).clamp_min(float(eps))
    )
    if patch_penalty_proposal_logits_bcqp is not None:
        proposal_log_distribution_bcqp = torch.nn.functional.log_softmax(
            patch_penalty_proposal_logits_bcqp,
            dim=-1,
        )
    else:
        proposal_distribution_bcqp = (
            benefit_prob / benefit_prob.sum(dim=-1, keepdim=True).clamp_min(float(eps))
        )
        proposal_log_distribution_bcqp = proposal_distribution_bcqp.clamp_min(
            float(eps)
        ).log()
    proposal_gain_listwise_bcq = -(
        proposal_gain_target_bcqp
        * proposal_log_distribution_bcqp
    ).sum(dim=-1) * adopt_target
    pairwise_rank_bcq = torch.zeros_like(adoption_bce_bcq)
    if patch_penalty_pairwise_rank_scores_bcqp is not None:
        assert patch_penalty_proposal_mask_bcqp is not None
        pair_index = torch.triu_indices(
            penalties,
            penalties,
            offset=1,
            device=base_bch.device,
        )
        pair_i = pair_index[0]
        pair_j = pair_index[1]
        predicted_pair_delta = (
            patch_penalty_pairwise_rank_scores_bcqp.index_select(-1, pair_i)
            - patch_penalty_pairwise_rank_scores_bcqp.index_select(-1, pair_j)
        )
        target_pair_delta = (
            normalized_gain_bcqp.index_select(-1, pair_i)
            - normalized_gain_bcqp.index_select(-1, pair_j)
        )
        active_pair = (
            patch_penalty_proposal_mask_bcqp.index_select(-1, pair_i).to(dtype=torch.bool)
            & patch_penalty_proposal_mask_bcqp.index_select(-1, pair_j).to(dtype=torch.bool)
            & (target_pair_delta.abs() > float(eps))
        )
        pair_loss = (
            target_pair_delta.abs()
            * torch.nn.functional.softplus(
                -target_pair_delta.sign() * predicted_pair_delta
            )
            * active_pair.to(dtype=predicted_pair_delta.dtype)
        )
        pairwise_rank_bcq = pair_loss.sum(dim=-1) / active_pair.sum(dim=-1).clamp_min(
            1
        ).to(dtype=pair_loss.dtype)

    best_prob_bcq = conditional_prob.gather(
        dim=-1,
        index=best_penalty_bcq.unsqueeze(-1),
    ).squeeze(-1)
    best_count_p = torch.bincount(
        best_penalty_bcq[adopt_target_bcq & active_mask_bcq].reshape(-1),
        minlength=penalties,
    ).to(device=base_bch.device, dtype=benefit_prob.dtype)
    present_best_p = best_count_p > 0.0
    best_total = best_count_p.sum().clamp_min(1.0)
    best_present_count = present_best_p.sum().clamp_min(1).to(dtype=benefit_prob.dtype)
    best_macro_weight_p = torch.where(
        present_best_p,
        best_total / (best_present_count * best_count_p.clamp_min(1.0)),
        torch.zeros_like(best_count_p),
    )
    ranking_ce_bcq = (
        -best_prob_bcq.log()
        * best_macro_weight_p.index_select(0, best_penalty_bcq.reshape(-1)).reshape_as(best_penalty_bcq)
        * adopt_target
    )
    proposal_rescue_ce_bcq = torch.zeros_like(adoption_bce_bcq)
    if patch_penalty_proposal_rescue_logits_bcqp is not None:
        assert patch_penalty_proposal_logits_bcqp is not None
        primary_penalty_bcq = patch_penalty_proposal_logits_bcqp.detach().argmax(dim=-1)
        rescue_target_bcq = (
            adopt_target_bcq
            & active_mask_bcq
            & (primary_penalty_bcq != best_penalty_bcq)
        )
        primary_mask_bcqp = torch.nn.functional.one_hot(
            primary_penalty_bcq,
            num_classes=penalties,
        ).to(dtype=torch.bool)
        rescue_log_probs_bcqp = torch.nn.functional.log_softmax(
            patch_penalty_proposal_rescue_logits_bcqp.masked_fill(
                primary_mask_bcqp,
                -1.0e4,
            ),
            dim=-1,
        )
        rescue_nll_bcq = -rescue_log_probs_bcqp.gather(
            dim=-1,
            index=best_penalty_bcq.unsqueeze(-1),
        ).squeeze(-1)
        rescue_count_p = torch.bincount(
            best_penalty_bcq[rescue_target_bcq].reshape(-1),
            minlength=penalties,
        ).to(device=base_bch.device, dtype=benefit_prob.dtype)
        rescue_present_p = rescue_count_p > 0.0
        rescue_total = rescue_count_p.sum().clamp_min(1.0)
        rescue_present_count = rescue_present_p.sum().clamp_min(1).to(
            dtype=benefit_prob.dtype
        )
        rescue_macro_weight_p = torch.where(
            rescue_present_p,
            rescue_total / (rescue_present_count * rescue_count_p.clamp_min(1.0)),
            torch.zeros_like(rescue_count_p),
        )
        proposal_rescue_ce_bcq = (
            rescue_nll_bcq
            * rescue_macro_weight_p.index_select(
                0,
                best_penalty_bcq.reshape(-1),
            ).reshape_as(best_penalty_bcq)
            * rescue_target_bcq.to(dtype=rescue_nll_bcq.dtype)
        )
    mse_utility_regression_bcq = torch.nn.functional.smooth_l1_loss(
        utility_scores,
        normalized_gain_bcqp.to(dtype=utility_scores.dtype),
        reduction="none",
        beta=0.1,
    ).mean(dim=-1)
    mae_utility_regression_bcq = torch.zeros_like(mse_utility_regression_bcq)
    utility_regression_bcq = mse_utility_regression_bcq
    if patch_penalty_mse_utility_scores_bcqp is not None:
        assert (
            patch_penalty_mae_utility_scores_bcqp is not None
            and normalized_mae_gain_bcqp is not None
        )
        mse_utility_scores = patch_penalty_mse_utility_scores_bcqp.clamp(-1.0, 1.0)
        mae_utility_scores = patch_penalty_mae_utility_scores_bcqp.clamp(-1.0, 1.0)
        mse_utility_regression_bcq = torch.nn.functional.smooth_l1_loss(
            mse_utility_scores,
            normalized_gain_bcqp.to(dtype=mse_utility_scores.dtype),
            reduction="none",
            beta=0.1,
        ).mean(dim=-1)
        mae_utility_regression_bcq = torch.nn.functional.smooth_l1_loss(
            mae_utility_scores,
            normalized_mae_gain_bcqp.to(dtype=mae_utility_scores.dtype),
            reduction="none",
            beta=0.1,
        ).mean(dim=-1)
        utility_regression_bcq = 0.5 * (
            mse_utility_regression_bcq + mae_utility_regression_bcq
        )
    risk_calibration_bcq = torch.zeros_like(adoption_bce_bcq)
    risk_sign_bce_bcq = torch.zeros_like(adoption_bce_bcq)
    risk_magnitude_bcq = torch.zeros_like(adoption_bce_bcq)
    risk_lower_quantile_bcq = torch.zeros_like(adoption_bce_bcq)
    selected_utility_policy_bcq = torch.zeros_like(adoption_bce_bcq)
    selected_adoption_bce_bcq = torch.zeros_like(adoption_bce_bcq)
    selected_adoption_recall_bcq = torch.zeros_like(adoption_bce_bcq)
    selected_false_adopt_bcq = torch.zeros_like(adoption_bce_bcq)
    if patch_penalty_risk_benefit_probs_bcqp is not None:
        assert (
            patch_penalty_risk_positive_magnitude_bcqp is not None
            and patch_penalty_risk_negative_magnitude_bcqp is not None
        )
        risk_prob = patch_penalty_risk_benefit_probs_bcqp.clamp(
            probability_eps,
            1.0 - probability_eps,
        )
        positive_magnitude = patch_penalty_risk_positive_magnitude_bcqp.clamp(0.0, 1.0)
        negative_magnitude = patch_penalty_risk_negative_magnitude_bcqp.clamp(0.0, 1.0)
        risk_calibration_bcq = (risk_prob - beneficial).square().mean(dim=-1)
        if masked_support:
            active_penalty_count = active_penalty_weight_bcqp.sum().clamp_min(1.0)
            active_penalty_count = active_penalty_count * float(penalties)
            risk_positive_rate = (
                (beneficial * active_penalty_weight_bcqp).sum()
                / active_penalty_count
            ).clamp_min(float(eps))
            risk_negative_rate = (
                (negative_benefit * active_penalty_weight_bcqp).sum()
                / active_penalty_count
            ).clamp_min(float(eps))
        else:
            risk_positive_rate = beneficial.mean().clamp_min(float(eps))
            risk_negative_rate = negative_benefit.mean().clamp_min(float(eps))
        risk_sign_bce_bcq = 0.5 * (
            -(beneficial * risk_prob.log()).mean(dim=-1) / risk_positive_rate
            -(negative_benefit * (1.0 - risk_prob).log()).mean(dim=-1)
            / risk_negative_rate
        )
        positive_magnitude_target = normalized_gain_bcqp.clamp_min(0.0)
        negative_magnitude_target = (-normalized_gain_bcqp).clamp_min(0.0)
        positive_magnitude_loss = torch.nn.functional.smooth_l1_loss(
            positive_magnitude,
            positive_magnitude_target.to(dtype=positive_magnitude.dtype),
            reduction="none",
            beta=0.1,
        )
        negative_magnitude_loss = torch.nn.functional.smooth_l1_loss(
            negative_magnitude,
            negative_magnitude_target.to(dtype=negative_magnitude.dtype),
            reduction="none",
            beta=0.1,
        )
        risk_magnitude_bcq = (
            (positive_magnitude_loss * beneficial).sum(dim=-1)
            / beneficial.sum(dim=-1).clamp_min(1.0)
            + (negative_magnitude_loss * negative_benefit).sum(dim=-1)
            / negative_benefit.sum(dim=-1).clamp_min(1.0)
        )
    if patch_penalty_risk_lower_quantile_scores_bcqp is not None:
        lower_quantile_scores = patch_penalty_risk_lower_quantile_scores_bcqp.clamp(
            -1.0,
            1.0,
        )
        quantile_error = normalized_gain_bcqp.to(
            dtype=lower_quantile_scores.dtype
        ) - lower_quantile_scores
        tau = float(risk_lower_quantile)
        risk_lower_quantile_bcq = torch.where(
            quantile_error >= 0.0,
            tau * quantile_error,
            (tau - 1.0) * quantile_error,
        ).mean(dim=-1)
    if patch_final_adopt_prob_bcq is not None:
        final_adopt_prob = patch_final_adopt_prob_bcq.clamp(
            probability_eps,
            1.0 - probability_eps,
        )
        selected_penalty_bcq = conditional_prob.detach().argmax(dim=-1)
        selected_gain_bcq = normalized_gain_bcqp.gather(
            dim=-1,
            index=selected_penalty_bcq.unsqueeze(-1),
        ).squeeze(-1)
        positive_selected_gain = selected_gain_bcq.clamp_min(0.0)
        negative_selected_cost = (-selected_gain_bcq).clamp_min(0.0)
        if masked_support:
            utility_scale = (
                (selected_gain_bcq.abs() * active_weight_bcq).sum()
                / active_weight_bcq.sum().clamp_min(1.0)
            ).detach().clamp_min(1.0e-3)
        else:
            utility_scale = selected_gain_bcq.abs().mean().detach().clamp_min(1.0e-3)
        selected_utility_policy_bcq = (
            -positive_selected_gain * final_adopt_prob.log()
            - negative_selected_cost * (1.0 - final_adopt_prob).log()
        ) / utility_scale
        selected_target = beneficial_bcqp.gather(
            dim=-1,
            index=selected_penalty_bcq.unsqueeze(-1),
        ).squeeze(-1).to(dtype=final_adopt_prob.dtype)
        selected_negative_target = 1.0 - selected_target
        if masked_support:
            active_count = active_weight_bcq.sum().clamp_min(1.0)
            selected_positive_rate = (
                (selected_target * active_weight_bcq).sum() / active_count
            ).clamp_min(float(eps))
            selected_negative_rate = (
                (selected_negative_target * active_weight_bcq).sum() / active_count
            ).clamp_min(float(eps))
        else:
            selected_positive_rate = selected_target.mean().clamp_min(float(eps))
            selected_negative_rate = selected_negative_target.mean().clamp_min(float(eps))
        selected_adoption_bce_bcq = 0.5 * (
            -selected_target * final_adopt_prob.log() / selected_positive_rate
            -selected_negative_target * (1.0 - final_adopt_prob).log()
            / selected_negative_rate
        )
        selected_adoption_recall_bcq = (
            torch.relu(float(target_adopt_probability) - final_adopt_prob).square()
            * selected_target
            / selected_positive_rate
        )
        selected_false_adopt_bcq = (
            torch.relu(final_adopt_prob - float(false_adopt_max_probability)).square()
            * selected_negative_target
            / selected_negative_rate
        )

    if masked_support:
        active_count = active_weight_bcq.sum().clamp_min(1.0)
        positive_rate = (
            (adopt_target * active_weight_bcq).sum() / active_count
        ).clamp_min(float(eps))
        negative_rate = (
            ((1.0 - adopt_target) * active_weight_bcq).sum() / active_count
        ).clamp_min(float(eps))
    else:
        positive_rate = adopt_target.mean().clamp_min(float(eps))
        negative_rate = (1.0 - adopt_target).mean().clamp_min(float(eps))
    adoption_recall_bcq = (
        torch.relu(float(target_adopt_probability) - adopt_prob).square()
        * adopt_target
        / positive_rate
    )
    false_adopt_bcq = (
        torch.relu(adopt_prob - float(false_adopt_max_probability)).square()
        * (1.0 - adopt_target)
        / negative_rate
    )
    penalty_recall_bcq = (
        torch.relu(float(target_penalty_probability) - benefit_prob).square()
        * beneficial
        * positive_macro_weight_p.view(1, 1, 1, -1)
    ).sum(dim=-1)
    if masked_support:
        active_penalty_count = active_penalty_weight_bcqp.sum().clamp_min(1.0)
        active_penalty_count = active_penalty_count * float(penalties)
        negative_penalty_rate = (
            (negative_benefit * active_penalty_weight_bcqp).sum()
            / active_penalty_count
        ).clamp_min(float(eps))
    else:
        negative_penalty_rate = negative_benefit.mean().clamp_min(float(eps))
    false_penalty_bcq = (
        torch.relu(benefit_prob - float(false_penalty_max_probability)).square()
        * negative_benefit
    ).mean(dim=-1) / negative_penalty_rate

    def reduce_bcq(value_bcq: torch.Tensor) -> torch.Tensor:
        if masked_support:
            rows = []
            for cluster_idx in range(int(K)):
                cluster_mask_bcq = active_mask_bcq & (
                    cluster_id_c.to(device=base_bch.device).view(1, -1, 1)
                    == cluster_idx
                )
                cluster_weight_bcq = cluster_mask_bcq.to(dtype=value_bcq.dtype)
                rows.append(
                    (value_bcq * cluster_weight_bcq).sum(dim=(1, 2))
                    / cluster_weight_bcq.sum(dim=(1, 2)).clamp_min(1.0)
                )
            return torch.stack(rows, dim=1)
        return scatter_mean_bc_to_bk(value_bcq.mean(dim=-1), cluster_id_c, int(K))

    terms = {
        "adoption_bce_bk": reduce_bcq(adoption_bce_bcq),
        "proposal_bce_bk": reduce_bcq(proposal_bce_bcq),
        "proposal_gain_listwise_bk": reduce_bcq(proposal_gain_listwise_bcq),
        "proposal_rescue_ce_bk": reduce_bcq(proposal_rescue_ce_bcq),
        "pairwise_rank_bk": reduce_bcq(pairwise_rank_bcq),
        "ranking_ce_bk": reduce_bcq(ranking_ce_bcq),
        "utility_regression_bk": reduce_bcq(utility_regression_bcq),
        "mse_utility_regression_bk": reduce_bcq(mse_utility_regression_bcq),
        "mae_utility_regression_bk": reduce_bcq(mae_utility_regression_bcq),
        "risk_calibration_bk": reduce_bcq(risk_calibration_bcq),
        "risk_sign_bce_bk": reduce_bcq(risk_sign_bce_bcq),
        "risk_magnitude_bk": reduce_bcq(risk_magnitude_bcq),
        "risk_lower_quantile_bk": reduce_bcq(risk_lower_quantile_bcq),
        "selected_utility_policy_bk": reduce_bcq(selected_utility_policy_bcq),
        "selected_adoption_bce_bk": reduce_bcq(selected_adoption_bce_bcq),
        "selected_adoption_recall_bk": reduce_bcq(selected_adoption_recall_bcq),
        "selected_false_adopt_bk": reduce_bcq(selected_false_adopt_bcq),
        "adoption_recall_bk": reduce_bcq(adoption_recall_bcq),
        "false_adopt_bk": reduce_bcq(false_adopt_bcq),
        "penalty_recall_bk": reduce_bcq(penalty_recall_bcq),
        "false_penalty_bk": reduce_bcq(false_penalty_bcq),
    }
    terms["total_bk"] = (
        float(adoption_bce_weight) * terms["adoption_bce_bk"]
        + float(proposal_bce_weight) * terms["proposal_bce_bk"]
        + float(proposal_gain_listwise_weight) * terms["proposal_gain_listwise_bk"]
        + float(proposal_rescue_ce_weight) * terms["proposal_rescue_ce_bk"]
        + float(pairwise_rank_weight) * terms["pairwise_rank_bk"]
        + float(ranking_ce_weight) * terms["ranking_ce_bk"]
        + float(utility_regression_weight) * terms["utility_regression_bk"]
        + float(risk_calibration_weight) * terms["risk_calibration_bk"]
        + float(risk_sign_bce_weight) * terms["risk_sign_bce_bk"]
        + float(risk_magnitude_weight) * terms["risk_magnitude_bk"]
        + float(risk_lower_quantile_weight) * terms["risk_lower_quantile_bk"]
        + float(selected_utility_policy_weight) * terms["selected_utility_policy_bk"]
        + float(selected_adoption_bce_weight) * terms["selected_adoption_bce_bk"]
        + float(selected_adoption_recall_weight) * terms["selected_adoption_recall_bk"]
        + float(selected_false_adopt_weight) * terms["selected_false_adopt_bk"]
        + float(adoption_recall_weight) * terms["adoption_recall_bk"]
        + float(false_adopt_weight) * terms["false_adopt_bk"]
        + float(penalty_recall_weight) * terms["penalty_recall_bk"]
        + float(false_penalty_weight) * terms["false_penalty_bk"]
    )
    return terms


@torch.no_grad()
def _patch_router_oracle_batch_stats(
    *,
    base_bch: torch.Tensor,
    candidate_bcpH: torch.Tensor,
    y_bch: torch.Tensor,
    patch_route_bcph: torch.Tensor,
    patch_skip_bcq: torch.Tensor,
    patch_penalty_benefit_probs_bcqp: Optional[torch.Tensor] = None,
    patch_penalty_risk_benefit_probs_bcqp: Optional[torch.Tensor] = None,
    patch_penalty_proposal_mask_bcqp: Optional[torch.Tensor] = None,
    patch_selected_penalty_index_bcq: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Return additive top-1 patch routing statistics for one batch."""
    batch, channels, horizon = base_bch.shape
    penalties = int(candidate_bcpH.shape[2])
    patches = int(patch_skip_bcq.shape[2])
    if patches <= 0 or horizon % patches != 0:
        raise ValueError("patch router oracle stats require patch count to divide horizon.")
    if tuple(candidate_bcpH.shape) != (batch, channels, penalties, horizon):
        raise ValueError("patch router oracle candidate shape does not match base prediction.")
    if tuple(patch_route_bcph.shape) != (batch, channels, penalties, horizon):
        raise ValueError("patch router oracle route shape does not match candidates.")
    patch_len = horizon // patches
    base_error_bcq = (base_bch - y_bch).square().reshape(
        batch,
        channels,
        patches,
        patch_len,
    ).mean(dim=-1)
    candidate_error_bcqp = (candidate_bcpH - y_bch.unsqueeze(2)).square().reshape(
        batch,
        channels,
        penalties,
        patches,
        patch_len,
    ).mean(dim=-1).permute(0, 1, 3, 2)
    base_mae_bcq = (base_bch - y_bch).abs().reshape(
        batch,
        channels,
        patches,
        patch_len,
    ).mean(dim=-1)
    candidate_mae_bcqp = (candidate_bcpH - y_bch.unsqueeze(2)).abs().reshape(
        batch,
        channels,
        penalties,
        patches,
        patch_len,
    ).mean(dim=-1).permute(0, 1, 3, 2)
    all_error_bcqk = torch.cat([base_error_bcq.unsqueeze(-1), candidate_error_bcqp], dim=-1)
    all_mae_bcqk = torch.cat([base_mae_bcq.unsqueeze(-1), candidate_mae_bcqp], dim=-1)
    beneficial_bcqp = candidate_error_bcqp < base_error_bcq.unsqueeze(-1)
    dual_beneficial_bcqp = beneficial_bcqp & (
        candidate_mae_bcqp < base_mae_bcq.unsqueeze(-1)
    )
    risk_sign_positive_count = torch.tensor(0.0, device=base_bch.device)
    risk_sign_predicted_positive_count = torch.tensor(0.0, device=base_bch.device)
    risk_sign_true_positive_count = torch.tensor(0.0, device=base_bch.device)
    risk_sign_correct_count = torch.tensor(0.0, device=base_bch.device)
    risk_sign_count = torch.tensor(0.0, device=base_bch.device)
    if patch_penalty_risk_benefit_probs_bcqp is not None:
        if tuple(patch_penalty_risk_benefit_probs_bcqp.shape) != tuple(beneficial_bcqp.shape):
            raise ValueError("patch router risk benefit probability shape does not match candidates.")
        risk_sign_predicted = patch_penalty_risk_benefit_probs_bcqp > 0.5
        risk_sign_positive_count = beneficial_bcqp.sum().to(dtype=base_bch.dtype)
        risk_sign_predicted_positive_count = risk_sign_predicted.sum().to(dtype=base_bch.dtype)
        risk_sign_true_positive_count = (
            beneficial_bcqp & risk_sign_predicted
        ).sum().to(dtype=base_bch.dtype)
        risk_sign_correct_count = (
            beneficial_bcqp == risk_sign_predicted
        ).sum().to(dtype=base_bch.dtype)
        risk_sign_count = torch.tensor(
            float(beneficial_bcqp.numel()),
            device=base_bch.device,
        )
    if patch_penalty_proposal_mask_bcqp is not None:
        if tuple(patch_penalty_proposal_mask_bcqp.shape) != tuple(beneficial_bcqp.shape):
            raise ValueError("patch router proposal mask shape does not match candidates.")
        proposed_bcqp = patch_penalty_proposal_mask_bcqp.to(dtype=torch.bool)
    elif patch_penalty_benefit_probs_bcqp is None:
        proposed_bcqp = torch.zeros_like(beneficial_bcqp)
    else:
        if tuple(patch_penalty_benefit_probs_bcqp.shape) != tuple(beneficial_bcqp.shape):
            raise ValueError("patch router benefit probability shape does not match candidates.")
        proposed_bcqp = patch_penalty_benefit_probs_bcqp > 0.5
    oracle_error_bcq, oracle_class_bcq = all_error_bcqk.min(dim=-1)
    # Report both metrics for one realizable oracle route.  Independently
    # minimizing MAE here would splice together two different actions and make
    # the displayed MSE/MAE pair impossible for any selector to attain.
    oracle_mae_bcq = all_mae_bcqk.gather(
        dim=-1,
        index=oracle_class_bcq.unsqueeze(dim=-1),
    ).squeeze(dim=-1)

    route_bcqp = patch_route_bcph.reshape(
        batch,
        channels,
        penalties,
        patches,
        patch_len,
    )[..., 0].permute(0, 1, 3, 2)
    selected_class_bcq = route_bcqp.argmax(dim=-1) + 1
    selected_skip_bcq = (patch_skip_bcq > 0.5) | (route_bcqp.sum(dim=-1) <= 0.0)
    selected_class_bcq = torch.where(
        selected_skip_bcq,
        torch.zeros_like(selected_class_bcq),
        selected_class_bcq,
    )
    oracle_penalty_bcq = oracle_class_bcq > 0
    selected_penalty_bcq = selected_class_bcq > 0
    oracle_penalty_index_bcq = (oracle_class_bcq - 1).clamp_min(0)
    oracle_penalty_selected_bcq = (
        route_bcqp.gather(
            dim=-1,
            index=oracle_penalty_index_bcq.unsqueeze(-1),
        ).squeeze(-1) > 0.0
    ) & oracle_penalty_bcq
    proposal_oracle_hit_bcq = (
        proposed_bcqp.gather(
            dim=-1,
            index=oracle_penalty_index_bcq.unsqueeze(-1),
        ).squeeze(-1)
        & oracle_penalty_bcq
    )
    beneficial_cardinality_bcq = beneficial_bcqp.sum(dim=-1)
    shortlist_pairwise_count = torch.tensor(0.0, device=base_bch.device)
    shortlist_pairwise_correct_count = torch.tensor(0.0, device=base_bch.device)
    if (
        patch_penalty_proposal_mask_bcqp is not None
        and patch_selected_penalty_index_bcq is not None
    ):
        if tuple(patch_selected_penalty_index_bcq.shape) != tuple(base_error_bcq.shape):
            raise ValueError("patch router selected penalty index shape does not match.")
        shortlist_valid_bcq = patch_penalty_proposal_mask_bcqp.sum(dim=-1) >= 2
        shortlist_best_bcq = candidate_error_bcqp.masked_fill(
            ~patch_penalty_proposal_mask_bcqp.to(dtype=torch.bool),
            float("inf"),
        ).argmin(dim=-1)
        shortlist_pairwise_count = shortlist_valid_bcq.sum().to(dtype=base_bch.dtype)
        shortlist_pairwise_correct_count = (
            shortlist_valid_bcq
            & (patch_selected_penalty_index_bcq == shortlist_best_bcq)
        ).sum().to(dtype=base_bch.dtype)
    selected_error_bcq = all_error_bcqk.gather(
        dim=-1,
        index=selected_class_bcq.unsqueeze(-1),
    ).squeeze(-1)
    selected_mae_bcq = all_mae_bcqk.gather(
        dim=-1,
        index=selected_class_bcq.unsqueeze(-1),
    ).squeeze(-1)
    selected_gain_bcq = base_error_bcq - selected_error_bcq
    selected_beneficial_bcq = selected_penalty_bcq & (selected_gain_bcq > 0.0)
    selected_harmful_bcq = selected_penalty_bcq & (selected_gain_bcq <= 0.0)
    selected_penalty_one_hot_bcqp = torch.nn.functional.one_hot(
        (selected_class_bcq - 1).clamp_min(0),
        num_classes=penalties,
    ).to(dtype=torch.bool)
    selected_penalty_one_hot_bcqp &= selected_penalty_bcq.unsqueeze(-1)
    selected_beneficial_by_penalty_bcqp = selected_penalty_one_hot_bcqp & beneficial_bcqp
    selected_dual_beneficial_bcq = (
        selected_penalty_one_hot_bcqp & dual_beneficial_bcqp
    ).any(dim=-1)
    selected_dual_harmful_bcq = selected_penalty_bcq & (~selected_dual_beneficial_bcq)
    dual_oracle_penalty_bcq = dual_beneficial_bcqp.any(dim=-1)
    selected_gain_by_penalty_bcqp = (
        selected_penalty_one_hot_bcqp.to(dtype=base_bch.dtype)
        * (base_error_bcq.unsqueeze(-1) - candidate_error_bcqp)
    )
    class_count = penalties + 1
    confusion_index = oracle_class_bcq.reshape(-1) * class_count + selected_class_bcq.reshape(-1)
    return {
        "count": torch.tensor(float(oracle_class_bcq.numel()), device=base_bch.device),
        "base_error_sum": base_error_bcq.sum(),
        "oracle_error_sum": oracle_error_bcq.sum(),
        "selected_error_sum": selected_error_bcq.sum(),
        "base_mae_sum": base_mae_bcq.sum(),
        "oracle_mae_sum": oracle_mae_bcq.sum(),
        "selected_mae_sum": selected_mae_bcq.sum(),
        "correct_count": (selected_class_bcq == oracle_class_bcq).sum().to(dtype=base_bch.dtype),
        "oracle_penalty_count": oracle_penalty_bcq.sum().to(dtype=base_bch.dtype),
        "selected_penalty_count": selected_penalty_bcq.sum().to(dtype=base_bch.dtype),
        "adoption_true_positive_count": (
            oracle_penalty_bcq & selected_penalty_bcq
        ).sum().to(dtype=base_bch.dtype),
        "selected_beneficial_count": selected_beneficial_bcq.sum().to(dtype=base_bch.dtype),
        "selected_harmful_count": selected_harmful_bcq.sum().to(dtype=base_bch.dtype),
        "dual_oracle_penalty_count": dual_oracle_penalty_bcq.sum().to(dtype=base_bch.dtype),
        "selected_dual_beneficial_count": selected_dual_beneficial_bcq.sum().to(
            dtype=base_bch.dtype
        ),
        "selected_dual_harmful_count": selected_dual_harmful_bcq.sum().to(
            dtype=base_bch.dtype
        ),
        "selected_positive_gain_sum": selected_gain_bcq.clamp_min(0.0).sum(),
        "selected_negative_cost_sum": (-selected_gain_bcq).clamp_min(0.0).sum(),
        "risk_sign_positive_count": risk_sign_positive_count,
        "risk_sign_predicted_positive_count": risk_sign_predicted_positive_count,
        "risk_sign_true_positive_count": risk_sign_true_positive_count,
        "risk_sign_correct_count": risk_sign_correct_count,
        "risk_sign_count": risk_sign_count,
        "selected_beneficial_count_by_penalty": selected_beneficial_by_penalty_bcqp.sum(
            dim=(0, 1, 2)
        ).to(dtype=base_bch.dtype),
        "selected_count_by_penalty": selected_penalty_one_hot_bcqp.sum(
            dim=(0, 1, 2)
        ).to(dtype=base_bch.dtype),
        "selected_gain_sum_by_penalty": selected_gain_by_penalty_bcqp.sum(dim=(0, 1, 2)),
        "oracle_penalty_hit_count": oracle_penalty_selected_bcq.sum().to(dtype=base_bch.dtype),
        "beneficial_penalty_count": beneficial_bcqp.sum(dim=(0, 1, 2)).to(dtype=base_bch.dtype),
        "proposed_penalty_count": proposed_bcqp.sum(dim=(0, 1, 2)).to(dtype=base_bch.dtype),
        "proposal_true_positive_count": (
            beneficial_bcqp & proposed_bcqp
        ).sum(dim=(0, 1, 2)).to(dtype=base_bch.dtype),
        "proposal_oracle_hit_count": proposal_oracle_hit_bcq.sum().to(dtype=base_bch.dtype),
        "shortlist_pairwise_count": shortlist_pairwise_count,
        "shortlist_pairwise_correct_count": shortlist_pairwise_correct_count,
        "proposal_oracle_hit_count_by_penalty": torch.bincount(
            oracle_penalty_index_bcq[proposal_oracle_hit_bcq].reshape(-1),
            minlength=penalties,
        ).to(dtype=base_bch.dtype),
        "beneficial_cardinality_sum": beneficial_cardinality_bcq.sum().to(dtype=base_bch.dtype),
        "beneficial_cardinality_histogram": torch.bincount(
            beneficial_cardinality_bcq.reshape(-1),
            minlength=penalties + 1,
        ).to(dtype=base_bch.dtype),
        "oracle_class_count": torch.bincount(
            oracle_class_bcq.reshape(-1),
            minlength=class_count,
        ).to(dtype=base_bch.dtype),
        "selected_class_count": torch.bincount(
            selected_class_bcq.reshape(-1),
            minlength=class_count,
        ).to(dtype=base_bch.dtype),
        "confusion_matrix": torch.bincount(
            confusion_index,
            minlength=class_count * class_count,
        ).reshape(class_count, class_count).to(dtype=base_bch.dtype),
        "base_error_sum_by_patch": base_error_bcq.sum(dim=(0, 1)),
        "oracle_error_sum_by_patch": oracle_error_bcq.sum(dim=(0, 1)),
        "selected_error_sum_by_patch": selected_error_bcq.sum(dim=(0, 1)),
        "count_by_patch": torch.full(
            (patches,),
            float(batch * channels),
            device=base_bch.device,
            dtype=base_bch.dtype,
        ),
    }


def _remove_forecast_affine_component(values_bch: torch.Tensor) -> torch.Tensor:
    centered = values_bch - values_bch.mean(dim=-1, keepdim=True)
    if int(values_bch.shape[-1]) <= 1:
        return centered
    trend_h = torch.linspace(
        -1.0,
        1.0,
        int(values_bch.shape[-1]),
        device=values_bch.device,
        dtype=values_bch.dtype,
    )
    trend_h = trend_h - trend_h.mean()
    coef_bc = (
        (centered * trend_h.view(1, 1, -1)).sum(dim=-1, keepdim=True)
        / trend_h.pow(2).sum().clamp_min(1.0e-12)
    )
    return centered - coef_bc * trend_h.view(1, 1, -1)


def _patchwise_penalty_bcq(
    prediction_bch: torch.Tensor,
    target_bch: torch.Tensor,
    penalty_fn,
    *,
    patch_len: int,
) -> torch.Tensor:
    """Evaluate one exact penalty independently on aligned horizon patches."""
    if prediction_bch.ndim != 3 or target_bch.shape != prediction_bch.shape:
        raise ValueError("patchwise penalty expects prediction/target with shape [B,C,H].")
    patch_len = int(patch_len)
    horizon = int(prediction_bch.shape[-1])
    if patch_len <= 0 or horizon % patch_len != 0:
        raise ValueError("patchwise penalty requires a positive patch_len dividing H.")
    batch, channels = int(prediction_bch.shape[0]), int(prediction_bch.shape[1])
    patches = horizon // patch_len
    prediction = prediction_bch.reshape(batch, channels, patches, patch_len)
    target = target_bch.reshape(batch, channels, patches, patch_len)
    prediction = prediction.permute(0, 2, 1, 3).reshape(batch * patches, channels, patch_len)
    target = target.permute(0, 2, 1, 3).reshape(batch * patches, channels, patch_len)
    penalty = penalty_fn(prediction, target)
    if tuple(penalty.shape) != (batch * patches, channels):
        raise ValueError(
            "patchwise penalty function must return [B*Q,C], got "
            f"{tuple(penalty.shape)}."
        )
    return penalty.reshape(batch, patches, channels).permute(0, 2, 1).contiguous()


def _level_oracle_patch_diagnostics(
    base_patch: torch.Tensor,
    target_patch: torch.Tensor,
    candidate_patch: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Exact constant-correction oracle and learned LEVEL outcome per patch."""
    if base_patch.shape != target_patch.shape or base_patch.shape != candidate_patch.shape:
        raise ValueError("LEVEL oracle patches must have identical shapes.")
    if base_patch.ndim < 1 or int(base_patch.shape[-1]) <= 0:
        raise ValueError("LEVEL oracle patches must have a nonempty time dimension.")
    base_mean = base_patch.mean(dim=-1)
    target_mean = target_patch.mean(dim=-1)
    candidate_mean = candidate_patch.mean(dim=-1)
    oracle_correction = target_mean - base_mean
    adapter_correction = candidate_mean - base_mean
    oracle_patch = base_patch + oracle_correction.unsqueeze(-1)
    base_penalty = (base_mean - target_mean).square()
    candidate_penalty = (candidate_mean - target_mean).square()
    oracle_penalty = (oracle_patch.mean(dim=-1) - target_mean).square()
    base_mse = (base_patch - target_patch).square().mean(dim=-1)
    candidate_mse = (candidate_patch - target_patch).square().mean(dim=-1)
    oracle_mse = (oracle_patch - target_patch).square().mean(dim=-1)
    return {
        "base_mean": base_mean,
        "target_mean": target_mean,
        "candidate_mean": candidate_mean,
        "oracle_correction": oracle_correction,
        "adapter_correction": adapter_correction,
        "oracle_patch": oracle_patch,
        "base_penalty": base_penalty,
        "candidate_penalty": candidate_penalty,
        "oracle_penalty": oracle_penalty,
        "base_mse": base_mse,
        "candidate_mse": candidate_mse,
        "oracle_mse": oracle_mse,
        "oracle_mse_gain_identity_error": (
            (base_mse - oracle_mse) - base_penalty
        ).abs(),
    }


def _prediction_fit_sufficient_statistics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, object]:
    """Return additive moments sufficient to reproduce zero-R2 and Pearson."""
    if prediction.shape != target.shape:
        raise ValueError("prediction-fit tensors must have identical shapes.")
    prediction_flat = prediction.detach().reshape(-1).to(dtype=torch.float64)
    target_flat = target.detach().reshape(-1).to(dtype=torch.float64)
    if not bool(torch.isfinite(prediction_flat).all()) or not bool(
        torch.isfinite(target_flat).all()
    ):
        raise ValueError("prediction-fit tensors must contain only finite values.")
    error = prediction_flat - target_flat
    return {
        "count": int(target_flat.numel()),
        "prediction_sum": float(prediction_flat.sum().item()),
        "prediction_squared_sum": float(prediction_flat.square().sum().item()),
        "target_sum": float(target_flat.sum().item()),
        "target_squared_sum": float(target_flat.square().sum().item()),
        "prediction_target_cross_sum": float(
            (prediction_flat * target_flat).sum().item()
        ),
        "squared_error_sum": float(error.square().sum().item()),
    }


def _level_stage1_acceptance_candidate_patch(
    *,
    base_patch: torch.Tensor,
    executed_candidate_patch: torch.Tensor,
    raw_amplitude_bcq: torch.Tensor,
    source: str,
) -> torch.Tensor:
    """Choose the LEVEL Stage-1 acceptance candidate without target leakage."""
    if base_patch.shape != executed_candidate_patch.shape:
        raise ValueError("LEVEL acceptance base/executed patches must have identical shapes.")
    if base_patch.ndim != 4 or int(base_patch.shape[-1]) <= 0:
        raise ValueError("LEVEL acceptance patches must have shape [B,C,Q,L].")
    if tuple(raw_amplitude_bcq.shape) != tuple(base_patch.shape[:-1]):
        raise ValueError(
            "LEVEL raw amplitude must have shape [B,C,Q] matching the patches."
        )
    source_mode = str(source).strip().lower()
    if source_mode == "executed":
        return executed_candidate_patch
    if source_mode == "raw_amplitude":
        return base_patch + raw_amplitude_bcq.unsqueeze(-1)
    raise ValueError(
        "LEVEL Stage-1 acceptance candidate source must be executed or raw_amplitude."
    )


def _validate_level_executed_candidate_patch(
    *,
    base_patch: torch.Tensor,
    executed_candidate_patch: torch.Tensor,
    executed_correction_bcq: torch.Tensor,
) -> None:
    """Fail closed unless the observed LEVEL candidate is the hard execution."""
    if base_patch.shape != executed_candidate_patch.shape:
        raise ValueError("LEVEL executed base/candidate patches must have identical shapes.")
    if base_patch.ndim != 4 or int(base_patch.shape[-1]) <= 0:
        raise ValueError("LEVEL executed patches must have shape [B,C,Q,L].")
    if tuple(executed_correction_bcq.shape) != tuple(base_patch.shape[:-1]):
        raise ValueError(
            "LEVEL executed correction must have shape [B,C,Q] matching the patches."
        )
    expected = base_patch + executed_correction_bcq.unsqueeze(-1)
    if not torch.allclose(
        executed_candidate_patch,
        expected,
        atol=1.0e-6,
        rtol=1.0e-6,
    ):
        max_abs_error = float(
            (executed_candidate_patch - expected).abs().max().item()
        )
        raise ValueError(
            "LEVEL hard candidate does not equal base plus the controller's "
            f"executed correction (max_abs_error={max_abs_error:.9g})."
        )


def _semantic_bank_acceptance_metrics(
    *,
    totals_by_penalty: List[Dict[str, float]],
    penalty_names: List[str],
    eps: float = 1.0e-8,
) -> Dict[str, object]:
    """Apply the Stage-1 candidate acceptance contract without approximation."""
    if len(totals_by_penalty) != len(penalty_names):
        raise ValueError("semantic bank acceptance totals must match penalty_names.")
    eps = float(eps)
    if eps <= 0.0:
        raise ValueError("semantic bank acceptance eps must be positive.")
    per_penalty: Dict[str, object] = {}
    all_pass = True
    for name, current in zip(penalty_names, totals_by_penalty):
        high_count = int(current["high_count"])
        low_count = int(current["low_count"])
        high_base_penalty_sum = float(current["high_base_penalty"])
        high_candidate_penalty_sum = float(current["high_candidate_penalty"])
        high_base_mse = float(current["high_base_mse"]) / max(high_count, 1)
        high_candidate_mse = float(current["high_candidate_mse"]) / max(high_count, 1)
        low_base_mse = float(current["low_base_mse"]) / max(low_count, 1)
        low_candidate_mse = float(current["low_candidate_mse"]) / max(low_count, 1)
        high_rms = math.sqrt(
            float(current["high_correction_sq_sum"])
            / max(float(current["high_correction_numel"]), 1.0)
        )
        low_rms = math.sqrt(
            float(current["low_correction_sq_sum"])
            / max(float(current["low_correction_numel"]), 1.0)
        )
        matching_relative_gain = (
            high_base_penalty_sum - high_candidate_penalty_sum
        ) / max(high_base_penalty_sum, eps)
        high_mse_regression = (
            high_candidate_mse - high_base_mse
        ) / max(high_base_mse, eps)
        low_mse_regression = (
            low_candidate_mse - low_base_mse
        ) / max(low_base_mse, eps)
        low_high_ratio = low_rms / max(high_rms, eps)
        checks = {
            "has_high_and_low_units": bool(high_count > 0 and low_count > 0),
            "matching_penalty_relative_gain_ge_1e_4": bool(
                matching_relative_gain >= 1.0e-4
            ),
            "high_need_mse_regression_le_1e_6": bool(
                high_mse_regression <= 1.0e-6
            ),
            "high_correction_rms_gt_1e_8": bool(high_rms > 1.0e-8),
            "low_high_correction_rms_ratio_le_0_25": bool(low_high_ratio <= 0.25),
            "low_need_mse_regression_le_0_001": bool(low_mse_regression <= 0.001),
        }
        passed = all(checks.values())
        all_pass = all_pass and passed
        per_penalty[name] = {
            "high_count": high_count,
            "low_count": low_count,
            "high_base_penalty_sum": high_base_penalty_sum,
            "high_candidate_penalty_sum": high_candidate_penalty_sum,
            "matching_penalty_relative_gain": matching_relative_gain,
            "matching_penalty_gain_pct": 100.0 * matching_relative_gain,
            "high_base_mse": high_base_mse,
            "high_candidate_mse": high_candidate_mse,
            "high_mse_regression_fraction": high_mse_regression,
            "low_base_mse": low_base_mse,
            "low_candidate_mse": low_candidate_mse,
            "low_mse_regression_fraction": low_mse_regression,
            "high_correction_rms": high_rms,
            "low_correction_rms": low_rms,
            "low_high_correction_rms_ratio": low_high_ratio,
            "checks": checks,
            "pass": bool(passed),
        }
    return {"per_penalty": per_penalty, "all_pass": bool(all_pass), "eps": eps}


def _semantic_bank_semantic_only_acceptance_metrics(
    *,
    totals_by_penalty: List[Dict[str, float]],
    temporal_totals_by_penalty: List[List[Dict[str, float]]],
    penalty_names: List[str],
    min_matching_gain_by_name: Dict[str, float],
    eps: float = 1.0e-8,
) -> Dict[str, object]:
    """Stage-1 semantic ability gate; routing safety remains diagnostic only."""
    diagnostics = _semantic_bank_acceptance_metrics(
        totals_by_penalty=totals_by_penalty,
        penalty_names=penalty_names,
        eps=eps,
    )
    if len(temporal_totals_by_penalty) != len(penalty_names):
        raise ValueError("semantic temporal totals must match penalty_names.")
    per_penalty = diagnostics["per_penalty"]
    all_pass = True
    for p, name in enumerate(penalty_names):
        block_rows = temporal_totals_by_penalty[p]
        if len(block_rows) < 3:
            raise ValueError("semantic acceptance requires at least three validation blocks.")
        block_gains: List[float] = []
        block_high_counts: List[int] = []
        for block in block_rows:
            high_count = int(block["high_count"])
            base_sum = float(block["high_base_penalty"])
            candidate_sum = float(block["high_candidate_penalty"])
            block_high_counts.append(high_count)
            block_gains.append((base_sum - candidate_sum) / max(base_sum, float(eps)))
        row = per_penalty[name]
        current = totals_by_penalty[p]
        improved_fraction = float(current.get("high_improved_count", 0.0)) / max(
            int(row["high_count"]), 1
        )
        relative_gain_quantiles = dict(
            current.get("high_relative_gain_quantiles", {})
        )
        min_improved_fraction = float(
            current.get("min_high_need_improved_fraction", 0.60)
        )
        row["high_need_improved_fraction"] = improved_fraction
        row["min_high_need_improved_fraction"] = min_improved_fraction
        row["high_need_relative_gain_quantiles"] = relative_gain_quantiles
        matching_gain = float(row["matching_penalty_relative_gain"])
        min_gain = float(min_matching_gain_by_name[name])
        semantic_checks = {
            "finite_matching_gain": bool(math.isfinite(matching_gain)),
            "matching_penalty_material_gain": bool(matching_gain >= min_gain),
            "high_correction_rms_gt_1e_8": bool(
                float(row["high_correction_rms"]) > 1.0e-8
            ),
            "high_need_improved_fraction_ge_threshold": bool(
                float(row.get("high_need_improved_fraction", 0.0))
                >= min_improved_fraction
            ),
            "high_need_relative_gain_median_gt_0": bool(
                float(row.get("high_need_relative_gain_quantiles", {}).get("0.5", float("-inf")))
                > 0.0
            ),
            "all_temporal_blocks_have_high_units": bool(
                all(count > 0 for count in block_high_counts)
            ),
            "all_temporal_block_matching_gains_gt_0": bool(
                all(math.isfinite(gain) and gain > 0.0 for gain in block_gains)
            ),
        }
        passed = all(semantic_checks.values())
        row["semantic_only_checks"] = semantic_checks
        row["semantic_only_pass"] = bool(passed)
        row["pass"] = bool(passed)
        row["material_matching_gain_threshold"] = min_gain
        row["temporal_block_matching_relative_gain"] = block_gains
        row["temporal_block_matching_gain_pct"] = [100.0 * value for value in block_gains]
        row["temporal_block_high_count"] = block_high_counts
        row["safety_diagnostics_are_non_blocking"] = True
        all_pass = all_pass and passed
    diagnostics["all_pass"] = bool(all_pass)
    diagnostics["acceptance_mode"] = "semantic_only_high_need_materiality"
    return diagnostics


def _named_forecast_attribute_error(
    candidate_bch: torch.Tensor,
    target_bch: torch.Tensor,
    name: str,
) -> torch.Tensor:
    """Direct future-attribute error used to train one named adapter."""
    if name == "level":
        return (candidate_bch.mean(dim=-1) - target_bch.mean(dim=-1)).pow(2)
    if name == "trend":
        if int(candidate_bch.shape[-1]) <= 1 or int(target_bch.shape[-1]) <= 1:
            return torch.zeros_like(candidate_bch[..., 0])
        candidate_trend = candidate_bch[..., -1] - candidate_bch[..., 0]
        target_trend = target_bch[..., -1] - target_bch[..., 0]
        return (candidate_trend - target_trend).pow(2)
    if name == "delta":
        cand_shape = candidate_bch - candidate_bch.mean(dim=-1, keepdim=True)
        target_shape = target_bch - target_bch.mean(dim=-1, keepdim=True)
        return (cand_shape - target_shape).pow(2).mean(dim=-1)
    if name == "d2_match":
        cand_shape = _remove_forecast_affine_component(candidate_bch)
        target_shape = _remove_forecast_affine_component(target_bch)
        return (cand_shape - target_shape).pow(2).mean(dim=-1)
    if name == "diff_amp":
        if int(candidate_bch.shape[-1]) <= 1 or int(target_bch.shape[-1]) <= 1:
            return torch.zeros_like(candidate_bch[..., 0])
        cand_std = candidate_bch.diff(dim=-1).std(dim=-1)
        target_std = target_bch.diff(dim=-1).std(dim=-1)
        return (cand_std - target_std).pow(2)
    if name in {"amp", "amp_under"}:
        cand_std = candidate_bch.std(dim=-1)
        target_std = target_bch.std(dim=-1)
        if name == "amp_under":
            return (target_std - cand_std).clamp_min(0.0).pow(2)
        return (cand_std - target_std).pow(2)
    raise ValueError(f"No direct future-attribute target is defined for penalty {name!r}.")


def _pred_residual_candidate_supervision_loss(
    *,
    y_base_bch: torch.Tensor,
    pred_out: Optional[Dict[str, torch.Tensor]],
    y_bch: torch.Tensor,
    cluster_id_c: torch.Tensor,
    K: int,
    penalty_names: Optional[List[str]] = None,
    penalty_fns: Optional[Dict[str, callable]] = None,
    penalty_scale: Optional[torch.Tensor] = None,
    penalty_need_threshold: Optional[torch.Tensor] = None,
    need_patch_len: int = 0,
    level_need_positive_weight: float = 1.0,
    noop_weight: float = 1.0,
    high_mse_relative_tolerance: float = 0.0,
    low_mse_relative_tolerance: float = 1.0e-3,
    low_high_rms_ratio_max: float = 0.25,
    constraint_weight: float = 1.0,
    constraint_eps: float = 1.0e-8,
    allowed_mask_kp: Optional[torch.Tensor] = None,
    only_allowed: bool = True,
    return_per_penalty: bool = False,
    return_components: bool = False,
    loss_kind: str = "mse",
    forecast_mse_weight: float = 1.0,
    min_abs_improvement: float = 0.0,
    min_rel_improvement: float = 0.0,
    include_intervention: bool = True,
    include_selector: bool = True,
    include_patch_route: bool = True,
    apply_output_anchors: bool = False,
    x_bcl: Optional[torch.Tensor] = None,
    query_start_abs_b: Optional[torch.Tensor] = None,
    input_len: int = 0,
    moe_cfg: Optional[dict] = None,
    moe_enable: bool = True,
    observed_history_tc: Optional[torch.Tensor] = None,
    train_stat_anchor_pc: Optional[torch.Tensor] = None,
    train_residual_anchor_phc: Optional[torch.Tensor] = None,
    learnable_output_anchor: Optional[ClusterwiseLearnableOutputAnchor] = None,
) -> Optional[object]:
    if pred_out is None:
        return None
    base_eval_bch, cand_bcpH = _pred_residual_candidates_on_eval_path(
        y_base_bch,
        pred_out,
        apply_output_anchors=apply_output_anchors,
        x_bcl=x_bcl,
        query_start_abs_b=query_start_abs_b,
        input_len=int(input_len),
        moe_cfg=moe_cfg,
        moe_enable=moe_enable,
        observed_history_tc=observed_history_tc,
        train_stat_anchor_pc=train_stat_anchor_pc,
        train_residual_anchor_phc=train_residual_anchor_phc,
        learnable_output_anchor=learnable_output_anchor,
        cluster_id_c=cluster_id_c,
        include_intervention=include_intervention,
        include_selector=include_selector,
        include_patch_route=include_patch_route,
    )
    if cand_bcpH is None or cand_bcpH.numel() == 0:
        return None
    loss_mode = str(loss_kind).lower()
    level_controller_loss_modes = {
        "level_residual_gate",
        "level_residual_separate_gate",
        "level_residual_high_need_separate_gate",
    }
    if loss_mode in level_controller_loss_modes:
        separate_gate = loss_mode in {
            "level_residual_separate_gate",
            "level_residual_high_need_separate_gate",
        }
        high_need_amplitude = (
            loss_mode == "level_residual_high_need_separate_gate"
        )
        required = {
            "level_amplitude_bcq",
            "level_need_logits_bcq",
        }
        if not separate_gate:
            required.add("level_executed_correction_bcq")
        missing = sorted(required - set(pred_out))
        if missing:
            raise ValueError(
                f"{loss_mode} supervision requires LEVEL controller outputs: "
                + ", ".join(missing)
            )
        if penalty_names is None or penalty_names.count("level") != 1:
            raise ValueError(f"{loss_mode} requires exactly one named level branch.")
        level_p = penalty_names.index("level")
        patch_len = int(need_patch_len)
        if patch_len != 12:
            raise ValueError(f"{loss_mode} requires need_patch_len=12.")
        if int(base_eval_bch.shape[-1]) % patch_len != 0:
            raise ValueError(f"{loss_mode} patch length must divide the horizon.")
        if penalty_scale is None or int(penalty_scale.numel()) != int(cand_bcpH.shape[2]):
            raise ValueError(f"{loss_mode} requires one train-only scale per branch.")
        if (
            penalty_need_threshold is None
            or int(penalty_need_threshold.numel()) != int(cand_bcpH.shape[2])
        ):
            raise ValueError(f"{loss_mode} requires one train-only q75 threshold per branch.")
        if separate_gate and float(level_need_positive_weight) <= 0.0:
            raise ValueError("level_residual_separate_gate requires positive need class weight.")
        batch, channels, horizon = base_eval_bch.shape
        patches = horizon // patch_len
        amplitude_bcq = pred_out["level_amplitude_bcq"]
        logits_bcq = pred_out["level_need_logits_bcq"]
        expected_shape = (batch, channels, patches)
        shape_values = [
            ("level_amplitude_bcq", amplitude_bcq),
            ("level_need_logits_bcq", logits_bcq),
        ]
        if not separate_gate:
            shape_values.append(
                (
                    "level_executed_correction_bcq",
                    pred_out["level_executed_correction_bcq"],
                )
            )
        for name, value in shape_values:
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, got {tuple(value.shape)}."
                )
        scale_level = penalty_scale[level_p].to(
            device=base_eval_bch.device, dtype=base_eval_bch.dtype
        ).clamp_min(1.0e-8)
        threshold_level = penalty_need_threshold[level_p].to(
            device=base_eval_bch.device, dtype=base_eval_bch.dtype
        )
        base_patch = base_eval_bch.reshape(batch, channels, patches, patch_len)
        target_patch = y_bch.reshape(batch, channels, patches, patch_len)
        residual_target_bcq = (target_patch - base_patch).mean(dim=-1).detach()
        need_target_bcq = (
            residual_target_bcq.square() / scale_level >= threshold_level
        ).to(dtype=base_eval_bch.dtype).detach()
        amplitude_error_bcq = (
            amplitude_bcq - residual_target_bcq
        ).square() / scale_level
        if high_need_amplitude:
            need_fraction = need_target_bcq.mean().detach()
            amplitude_loss_bcq = (
                amplitude_error_bcq
                * need_target_bcq
                / need_fraction.clamp_min(1.0e-12)
            )
        else:
            amplitude_loss_bcq = amplitude_error_bcq
        if separate_gate:
            positive_weight = logits_bcq.new_tensor(float(level_need_positive_weight))
            need_loss_bcq = F.binary_cross_entropy_with_logits(
                logits_bcq,
                need_target_bcq,
                pos_weight=positive_weight,
                reduction="none",
            )
            total_bc = amplitude_loss_bcq.mean(dim=-1) + need_loss_bcq.mean(dim=-1)
            component_values = (
                ("level_amplitude_loss", amplitude_loss_bcq),
                ("level_need_balanced_bce", need_loss_bcq),
            )
        else:
            need_loss_bcq = F.binary_cross_entropy_with_logits(
                logits_bcq,
                need_target_bcq,
                reduction="none",
            )
            executed_bcq = pred_out["level_executed_correction_bcq"]
            executed_loss_bcq = (
                executed_bcq - need_target_bcq * residual_target_bcq
            ).square() / scale_level
            total_bc = (
                amplitude_loss_bcq.mean(dim=-1)
                + need_loss_bcq.mean(dim=-1)
                + executed_loss_bcq.mean(dim=-1)
            )
            component_values = (
                ("level_amplitude_loss", amplitude_loss_bcq),
                ("level_need_bce", need_loss_bcq),
                ("level_executed_loss", executed_loss_bcq),
            )
        err_bcp = total_bc.new_zeros((batch, channels, int(cand_bcpH.shape[2])))
        err_bcp[:, :, level_p] = total_bc
        objective_component_bcp = {}
        for key, values_bcq in component_values:
            values_bcp = total_bc.new_zeros(
                (batch, channels, int(cand_bcpH.shape[2]))
            )
            values_bcp[:, :, level_p] = values_bcq.mean(dim=-1)
            objective_component_bcp[key] = values_bcp
    elif loss_mode in {"direct_attribute", "attribute_mse", "future_attribute"}:
        if penalty_names is None or len(penalty_names) != int(cand_bcpH.shape[2]):
            raise ValueError("direct_attribute candidate supervision requires one penalty name per branch.")
        per_penalty = []
        for p, name in enumerate(penalty_names):
            err_bc = _named_forecast_attribute_error(cand_bcpH[:, :, p, :], y_bch, name)
            if penalty_scale is not None and penalty_scale.numel() > p:
                scale_p = penalty_scale[p].to(device=err_bc.device, dtype=err_bc.dtype).clamp_min(1.0e-6)
                err_bc = err_bc / scale_p
            per_penalty.append(err_bc)
        err_bcp = torch.stack(per_penalty, dim=-1)
    elif loss_mode in {
        "own_penalty",
        "penalty",
        "attribute",
        "shape",
        "own_penalty_mse",
        "mse_own_penalty",
        "high_need_own_penalty",
        "need_weighted_own_penalty_mse",
        "acceptance_guarded_own_penalty",
    }:
        if penalty_names is None or penalty_fns is None or len(penalty_names) != int(cand_bcpH.shape[2]):
            raise ValueError("own_penalty candidate supervision requires one penalty name/function per branch.")
        if loss_mode in {
            "high_need_own_penalty",
            "need_weighted_own_penalty_mse",
            "acceptance_guarded_own_penalty",
        }:
            patch_len = int(need_patch_len)
            if patch_len <= 0:
                raise ValueError("need-weighted candidate supervision requires need_patch_len > 0.")
            if penalty_scale is None or int(penalty_scale.numel()) != int(cand_bcpH.shape[2]):
                raise ValueError("need-weighted candidate supervision requires one penalty scale per branch.")
            if penalty_need_threshold is None or int(penalty_need_threshold.numel()) != int(cand_bcpH.shape[2]):
                raise ValueError("need-weighted candidate supervision requires one need threshold per branch.")
            per_penalty = []
            objective_components: Optional[Dict[str, List[torch.Tensor]]]
            if loss_mode == "high_need_own_penalty":
                objective_components = {
                    "high_semantic_contribution": [],
                }
            elif loss_mode == "need_weighted_own_penalty_mse":
                objective_components = {
                    "high_semantic_contribution": [],
                    "high_forecast_mse_contribution": [],
                    "low_noop_contribution": [],
                }
            else:
                objective_components = {
                    "high_semantic": [],
                    "high_forecast_mse": [],
                    "high_mse_violation": [],
                    "low_mse_violation": [],
                    "low_noop_violation": [],
                    "base_guard_objective": [],
                    "constraint_total": [],
                }
            for p, name in enumerate(penalty_names):
                scale_p = penalty_scale[p].to(
                    device=cand_bcpH.device,
                    dtype=cand_bcpH.dtype,
                ).clamp_min(1.0e-8)
                threshold_p = penalty_need_threshold[p].to(
                    device=cand_bcpH.device,
                    dtype=cand_bcpH.dtype,
                )
                base_penalty_bcq = _patchwise_penalty_bcq(
                    base_eval_bch,
                    y_bch,
                    penalty_fns[name],
                    patch_len=patch_len,
                )
                candidate_penalty_bcq = _patchwise_penalty_bcq(
                    cand_bcpH[:, :, p, :],
                    y_bch,
                    penalty_fns[name],
                    patch_len=patch_len,
                )
                batch, channels, horizon = base_eval_bch.shape
                patches = horizon // patch_len
                target_patch = y_bch.reshape(batch, channels, patches, patch_len)
                candidate_patch = cand_bcpH[:, :, p, :].reshape(
                    batch, channels, patches, patch_len
                )
                base_patch = base_eval_bch.reshape(batch, channels, patches, patch_len)
                candidate_mse_bcq = (candidate_patch - target_patch).square().mean(dim=-1)
                correction_mse_bcq = (candidate_patch - base_patch).square().mean(dim=-1)
                need_bcq = (
                    (base_penalty_bcq / scale_p) >= threshold_p
                ).to(dtype=candidate_mse_bcq.dtype).detach()
                if loss_mode == "high_need_own_penalty":
                    high_semantic_bcq = need_bcq * (
                        candidate_penalty_bcq / scale_p
                    )
                    per_penalty.append(high_semantic_bcq.mean(dim=-1))
                    objective_components["high_semantic_contribution"].append(
                        high_semantic_bcq.mean(dim=-1)
                    )
                    continue
                if loss_mode == "need_weighted_own_penalty_mse":
                    high_semantic_bcq = need_bcq * (
                        candidate_penalty_bcq / scale_p
                    )
                    high_forecast_mse_bcq = need_bcq * candidate_mse_bcq
                    low_noop_bcq = (1.0 - need_bcq) * correction_mse_bcq
                    loss_bcq = (
                        high_semantic_bcq
                        + float(forecast_mse_weight) * high_forecast_mse_bcq
                        + float(noop_weight) * low_noop_bcq
                    )
                    per_penalty.append(loss_bcq.mean(dim=-1))
                    objective_components["high_semantic_contribution"].append(
                        high_semantic_bcq.mean(dim=-1)
                    )
                    objective_components[
                        "high_forecast_mse_contribution"
                    ].append(high_forecast_mse_bcq.mean(dim=-1))
                    objective_components["low_noop_contribution"].append(
                        low_noop_bcq.mean(dim=-1)
                    )
                    continue

                eps = float(constraint_eps)
                if eps <= 0.0:
                    raise ValueError("acceptance-guarded constraint_eps must be positive.")
                high_tol = float(high_mse_relative_tolerance)
                low_tol = float(low_mse_relative_tolerance)
                ratio_max = float(low_high_rms_ratio_max)
                constraint = float(constraint_weight)
                if high_tol < 0.0 or low_tol < 0.0:
                    raise ValueError("acceptance-guarded MSE tolerances must be nonnegative.")
                if ratio_max < 0.0 or constraint <= 0.0:
                    raise ValueError(
                        "acceptance-guarded ratio must be nonnegative and constraint_weight positive."
                    )

                low_bcq = 1.0 - need_bcq
                high_count_bc = need_bcq.sum(dim=-1)
                low_count_bc = low_bcq.sum(dim=-1)

                def conditional_mean(values_bcq: torch.Tensor, mask_bcq: torch.Tensor) -> torch.Tensor:
                    count_bc = mask_bcq.sum(dim=-1)
                    mean_bc = (values_bcq * mask_bcq).sum(dim=-1) / count_bc.clamp_min(1.0)
                    return torch.where(count_bc > 0.0, mean_bc, torch.zeros_like(mean_bc))

                base_mse_bcq = (base_patch - target_patch).square().mean(dim=-1).detach()
                high_allowed_bcq = base_mse_bcq * (1.0 + high_tol)
                high_mse_violation_bcq = (
                    (candidate_mse_bcq - high_allowed_bcq).clamp_min(0.0)
                    / base_mse_bcq.clamp_min(eps)
                )

                high_correction_mse_bc = conditional_mean(correction_mse_bcq, need_bcq)
                low_correction_mse_bc = conditional_mean(correction_mse_bcq, low_bcq)
                low_base_mse_bc = conditional_mean(base_mse_bcq, low_bcq).detach()
                low_candidate_mse_bc = conditional_mean(candidate_mse_bcq, low_bcq)
                low_allowed_mse_bc = low_base_mse_bc * (1.0 + low_tol)
                low_mse_violation_bc = (
                    (low_candidate_mse_bc - low_allowed_mse_bc).clamp_min(0.0)
                    / low_base_mse_bc.clamp_min(eps)
                )
                low_noop_excess_bc = (
                    low_correction_mse_bc
                    - (ratio_max * ratio_max) * high_correction_mse_bc.detach()
                )
                low_noop_violation_bc = (
                    low_noop_excess_bc.clamp_min(0.0)
                    / low_base_mse_bc.clamp_min(eps)
                )
                high_semantic_bc = conditional_mean(
                    candidate_penalty_bcq / scale_p,
                    need_bcq,
                )
                base_high_semantic_bc = conditional_mean(
                    base_penalty_bcq.detach() / scale_p,
                    need_bcq,
                )
                high_forecast_mse_bc = conditional_mean(
                    candidate_mse_bcq / base_mse_bcq.clamp_min(eps),
                    need_bcq,
                )
                base_high_forecast_bc = torch.where(
                    high_count_bc > 0.0,
                    torch.ones_like(high_forecast_mse_bc),
                    torch.zeros_like(high_forecast_mse_bc),
                )
                high_mse_violation_bc = conditional_mean(
                    high_mse_violation_bcq,
                    need_bcq,
                )
                constraint_total_bc = (
                    high_mse_violation_bc
                    + low_mse_violation_bc
                    + float(noop_weight) * low_noop_violation_bc
                )
                candidate_objective_bc = (
                    high_semantic_bc
                    + float(forecast_mse_weight) * high_forecast_mse_bc
                )
                base_guard_objective_bc = (
                    base_high_semantic_bc
                    + float(forecast_mse_weight) * base_high_forecast_bc
                ).detach()
                # The final non-tradeable switch is applied after channel-to-cluster
                # reduction.  Until then keep the candidate objective, detached base
                # guard, and constraints as separate tensors so no local semantic
                # improvement can numerically pay for another unit's violation.
                per_penalty.append(candidate_objective_bc)
                objective_components["high_semantic"].append(high_semantic_bc)
                objective_components["high_forecast_mse"].append(
                    high_forecast_mse_bc
                )
                objective_components["high_mse_violation"].append(
                    high_mse_violation_bc
                )
                objective_components["low_mse_violation"].append(
                    low_mse_violation_bc
                )
                objective_components["low_noop_violation"].append(
                    low_noop_violation_bc
                )
                objective_components["base_guard_objective"].append(
                    base_guard_objective_bc
                )
                objective_components["constraint_total"].append(
                    constraint_total_bc
                )
            err_bcp = torch.stack(per_penalty, dim=-1)
            objective_component_bcp = {
                key: torch.stack(values, dim=-1)
                for key, values in objective_components.items()
            }
        else:
            per_penalty = []
            for p, name in enumerate(penalty_names):
                pen_bc = penalty_fns[name](cand_bcpH[:, :, p, :], y_bch)
                if penalty_scale is not None and penalty_scale.numel() > p:
                    scale_p = penalty_scale[p].to(device=pen_bc.device, dtype=pen_bc.dtype).clamp_min(1.0e-6)
                    pen_bc = pen_bc / scale_p
                per_penalty.append(pen_bc)
            err_bcp = torch.stack(per_penalty, dim=-1)
        if loss_mode in {"own_penalty_mse", "mse_own_penalty"}:
            forecast_mse_bcp = (
                cand_bcpH - y_bch.unsqueeze(2)
            ).square().mean(dim=-1)
            err_bcp = err_bcp + float(forecast_mse_weight) * forecast_mse_bcp
    elif loss_mode in {"mse", "l2"}:
        err_bcpH = cand_bcpH - y_bch.unsqueeze(2)
        err_bcp = err_bcpH.pow(2).mean(dim=-1)
    elif loss_mode in {"mae", "l1"}:
        err_bcpH = cand_bcpH - y_bch.unsqueeze(2)
        err_bcp = err_bcpH.abs().mean(dim=-1)
    elif loss_mode in {"gain_hinge", "gain_hinge_mse", "mse_gain_hinge", "residual_gain_hinge"}:
        cand_err_bcp = (cand_bcpH - y_bch.unsqueeze(2)).pow(2).mean(dim=-1)
        base_err_bc = (base_eval_bch - y_bch).pow(2).mean(dim=-1)
        required_bc = torch.maximum(
            torch.full_like(base_err_bc, max(0.0, float(min_abs_improvement))),
            max(0.0, float(min_rel_improvement)) * base_err_bc.detach().abs().clamp_min(1.0e-12),
        )
        err_bcp = (cand_err_bcp - base_err_bc.detach().unsqueeze(-1) + required_bc.unsqueeze(-1)).clamp_min(0.0)
    elif loss_mode in {"gain_hinge_mae", "mae_gain_hinge", "l1_gain_hinge"}:
        cand_err_bcp = (cand_bcpH - y_bch.unsqueeze(2)).abs().mean(dim=-1)
        base_err_bc = (base_eval_bch - y_bch).abs().mean(dim=-1)
        required_bc = torch.maximum(
            torch.full_like(base_err_bc, max(0.0, float(min_abs_improvement))),
            max(0.0, float(min_rel_improvement)) * base_err_bc.detach().abs().clamp_min(1.0e-12),
        )
        err_bcp = (cand_err_bcp - base_err_bc.detach().unsqueeze(-1) + required_bc.unsqueeze(-1)).clamp_min(0.0)
    else:
        raise ValueError(
            "moe.pred_side_residual.candidate_supervision.loss must be mse, mae, own_penalty, "
            "own_penalty_mse, high_need_own_penalty, need_weighted_own_penalty_mse, "
            "acceptance_guarded_own_penalty, "
            "level_residual_gate, level_residual_separate_gate, "
            "level_residual_high_need_separate_gate, "
            "direct_attribute, "
            "gain_hinge_mse, or gain_hinge_mae "
            f"(got {loss_kind!r})."
        )
    err_bkp = scatter_mean_bcf_to_bkf(err_bcp, cluster_id_c, K)
    component_bkp = None
    if loss_mode in {
        "level_residual_gate",
        "level_residual_separate_gate",
        "level_residual_high_need_separate_gate",
        "high_need_own_penalty",
        "need_weighted_own_penalty_mse",
        "acceptance_guarded_own_penalty",
    }:
        component_bkp = {
            key: scatter_mean_bcf_to_bkf(value, cluster_id_c, K)
            for key, value in objective_component_bcp.items()
        }
    if loss_mode == "acceptance_guarded_own_penalty":
        assert component_bkp is not None
        constraint_active_p = (
            component_bkp["constraint_total"]
            .detach()
            .amax(dim=(0, 1))
            > 0.0
        )
        constraint_active_bkp = constraint_active_p.view(1, 1, -1).expand_as(
            err_bkp
        )
        err_bkp = torch.where(
            constraint_active_bkp,
            component_bkp["base_guard_objective"].detach()
            + float(constraint_weight) * component_bkp["constraint_total"],
            err_bkp,
        )
        component_bkp["constraint_active"] = constraint_active_bkp.to(
            dtype=err_bkp.dtype
        )
    if bool(only_allowed) and allowed_mask_kp is not None and allowed_mask_kp.numel() > 0:
        allowed = allowed_mask_kp.to(device=err_bkp.device, dtype=err_bkp.dtype)
        if allowed.shape != err_bkp.shape[1:]:
            raise ValueError(
                "candidate_supervision allowed mask must have shape [K,P], "
                f"got {tuple(allowed.shape)} vs {tuple(err_bkp.shape[1:])}."
            )
        empty = allowed.sum(dim=-1, keepdim=True) <= 0.0
        allowed = torch.where(empty, torch.ones_like(allowed), allowed)
        masked_bkp = err_bkp * allowed.unsqueeze(0)
        if bool(return_per_penalty):
            if bool(return_components):
                assert component_bkp is not None
                return {
                    "loss_bkp": masked_bkp,
                    "components_bkp": {
                        key: value * allowed.unsqueeze(0)
                        for key, value in component_bkp.items()
                    },
                }
            return masked_bkp
        return masked_bkp.sum(dim=-1) / allowed.sum(dim=-1).clamp_min(1.0).view(1, -1)
    if bool(return_per_penalty):
        if bool(return_components):
            assert component_bkp is not None
            return {
                "loss_bkp": err_bkp,
                "components_bkp": component_bkp,
            }
        return err_bkp
    return err_bkp.mean(dim=-1)


def _pred_residual_intervention_supervision_loss(
    *,
    y_base_bch: torch.Tensor,
    pred_out: Optional[Dict[str, torch.Tensor]],
    y_bch: torch.Tensor,
    cluster_id_c: torch.Tensor,
    K: int,
    allowed_mask_kp: Optional[torch.Tensor] = None,
    only_allowed: bool = True,
    min_gain: float = 0.0,
    pos_weight: float = 1.0,
    apply_output_anchors: bool = False,
    x_bcl: Optional[torch.Tensor] = None,
    query_start_abs_b: Optional[torch.Tensor] = None,
    input_len: int = 0,
    moe_cfg: Optional[dict] = None,
    moe_enable: bool = True,
    observed_history_tc: Optional[torch.Tensor] = None,
    train_stat_anchor_pc: Optional[torch.Tensor] = None,
    train_residual_anchor_phc: Optional[torch.Tensor] = None,
    learnable_output_anchor: Optional[ClusterwiseLearnableOutputAnchor] = None,
) -> Optional[torch.Tensor]:
    if pred_out is None:
        return None
    intervention_bcp = pred_out.get("intervention_bcp")
    if intervention_bcp is None or intervention_bcp.numel() == 0:
        return None
    base_eval_bch, cand_bcpH = _pred_residual_candidates_on_eval_path(
        y_base_bch,
        pred_out,
        apply_output_anchors=apply_output_anchors,
        x_bcl=x_bcl,
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
    )
    if cand_bcpH is None or cand_bcpH.numel() == 0:
        return None
    base_err_bc = (base_eval_bch.detach() - y_bch.detach()).pow(2).mean(dim=-1)
    cand_err_bcp = (cand_bcpH.detach() - y_bch.detach().unsqueeze(2)).pow(2).mean(dim=-1)
    target_bcp = ((base_err_bc.unsqueeze(-1) - cand_err_bcp) > float(min_gain)).to(dtype=intervention_bcp.dtype)
    prob_bcp = intervention_bcp.clamp(1.0e-6, 1.0 - 1.0e-6)
    loss_bcp = -(
        float(max(pos_weight, 0.0)) * target_bcp * prob_bcp.log()
        + (1.0 - target_bcp) * (1.0 - prob_bcp).log()
    )
    if bool(only_allowed) and allowed_mask_kp is not None and allowed_mask_kp.numel() > 0:
        allowed_cp = allowed_mask_kp.to(device=loss_bcp.device, dtype=torch.bool).index_select(
            0,
            cluster_id_c.to(device=loss_bcp.device, dtype=torch.long),
        )
        if allowed_cp.shape != loss_bcp.shape[1:]:
            raise ValueError(
                "intervention supervision allowed mask must broadcast to [C,P], "
                f"got {tuple(allowed_cp.shape)} vs {tuple(loss_bcp.shape[1:])}."
            )
        loss_bcp = loss_bcp * allowed_cp.to(dtype=loss_bcp.dtype).unsqueeze(0)
        denom_bc = allowed_cp.to(dtype=loss_bcp.dtype).sum(dim=-1).clamp_min(1.0).view(1, -1)
        loss_bc = loss_bcp.sum(dim=-1) / denom_bc
    else:
        loss_bc = loss_bcp.mean(dim=-1)
    return scatter_mean_bc_to_bk(loss_bc, cluster_id_c, K)


class PredResidualCandidateSelector(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        num_channels: int,
        num_penalties: int,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        init_skip_bias: float = 0.0,
        init_penalty_bias: float = 0.0,
        use_penalty_identity: bool = False,
        use_channel_identity: bool = False,
        use_time_features: bool = False,
        time_feature_periods: Optional[List[int]] = None,
        time_feature_offset: int = 0,
        feature_mode: str = "base",
        patch_len: int = 0,
    ):
        super().__init__()
        self.base_F = int(feat_dim)
        self.C = int(num_channels)
        self.P = int(num_penalties)
        self.use_penalty_identity = bool(use_penalty_identity)
        self.use_channel_identity = bool(use_channel_identity)
        self.use_time_features = bool(use_time_features)
        periods = [int(v) for v in (time_feature_periods or []) if int(v) > 0]
        self.time_feature_periods = periods
        self.time_feature_offset = int(time_feature_offset)
        self.feature_mode = str(feature_mode or "base").lower()
        self.patch_len = max(0, int(patch_len))
        self.F = (
            self.base_F
            + (self.C if self.use_channel_identity else 0)
            + (2 * len(self.time_feature_periods) if self.use_time_features else 0)
            + (self.P if self.use_penalty_identity else 0)
        )
        hidden = int(hidden_dim)
        self.register_buffer("feature_mean", torch.zeros(1, 1, self.F), persistent=True)
        self.register_buffer("feature_std", torch.ones(1, 1, self.F), persistent=True)
        self.register_buffer("feature_standardize_enabled", torch.tensor(0.0), persistent=True)
        self.register_buffer("feature_standardize_clip", torch.tensor(0.0), persistent=True)
        self.register_buffer("allowed_penalty_mask_cp", torch.empty(0, 0, dtype=torch.bool), persistent=False)
        self.net = nn.Sequential(
            nn.Linear(self.F, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity(),
            nn.Linear(hidden, 1),
        )
        self.skip_net = nn.Sequential(
            nn.Linear(self.F, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity(),
            nn.Linear(hidden, 1),
        )
        self.skip_bias = nn.Parameter(torch.full((self.C,), float(init_skip_bias)))
        self.penalty_bias = nn.Parameter(torch.full((self.P,), float(init_penalty_bias)))
        self.penalty_channel_bias = nn.Parameter(torch.zeros(self.C, self.P))
        self.decision_margin = 0.0
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in list(self.net) + list(self.skip_net):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for seq in (self.net, self.skip_net):
            last = seq[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def set_feature_standardization(self, mean_f: torch.Tensor, std_f: torch.Tensor) -> None:
        mean = mean_f.detach().view(1, 1, self.F).to(device=self.feature_mean.device, dtype=self.feature_mean.dtype)
        std = std_f.detach().view(1, 1, self.F).to(device=self.feature_std.device, dtype=self.feature_std.dtype)
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std.clamp_min(1.0e-6))
        self.feature_standardize_enabled.fill_(1.0)

    def set_feature_standardize_clip(self, clip_value: float) -> None:
        self.feature_standardize_clip.fill_(max(0.0, float(clip_value)))

    def set_allowed_penalty_mask(self, allowed_mask_cp: Optional[torch.Tensor]) -> None:
        if allowed_mask_cp is None or int(allowed_mask_cp.numel()) == 0:
            self.allowed_penalty_mask_cp = torch.empty(0, 0, device=self.skip_bias.device, dtype=torch.bool)
            return
        allowed = _selector_allowed_mask_cp(
            allowed_mask_cp,
            C=self.C,
            P=self.P,
            device=self.skip_bias.device,
            context="candidate selector",
        )
        self.allowed_penalty_mask_cp = allowed.detach().clone()

    def _resolved_allowed_penalty_mask(self, allowed_mask_cp: Optional[torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
        mask = allowed_mask_cp
        if mask is None or int(mask.numel()) == 0:
            mask = self.allowed_penalty_mask_cp
        return _selector_allowed_mask_cp(mask, C=self.C, P=self.P, device=device, context="candidate selector")

    def _mask_disallowed_logits(
        self,
        logits_bcq: torch.Tensor,
        allowed_mask_cp: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        allowed = self._resolved_allowed_penalty_mask(allowed_mask_cp, logits_bcq.device)
        if allowed is None:
            return logits_bcq
        penalty_logits = logits_bcq[..., 1:].masked_fill(~allowed.unsqueeze(0), torch.finfo(logits_bcq.dtype).min)
        return torch.cat([logits_bcq[..., :1], penalty_logits], dim=-1)

    def _standardize_feat(self, feat: torch.Tensor) -> torch.Tensor:
        if bool(self.feature_standardize_enabled.item() > 0.5):
            feat = (feat - self.feature_mean.to(device=feat.device, dtype=feat.dtype)) / self.feature_std.to(
                device=feat.device,
                dtype=feat.dtype,
            )
            clip_value = float(self.feature_standardize_clip.item())
            if clip_value > 0.0:
                feat = feat.clamp(min=-clip_value, max=clip_value)
        return feat

    def _append_penalty_identity(
        self,
        skip_feat_bcf: torch.Tensor,
        cand_feat_bcpf: torch.Tensor,
        query_start_abs_b: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if int(skip_feat_bcf.shape[-1]) != self.base_F or int(cand_feat_bcpf.shape[-1]) != self.base_F:
            raise ValueError(
                f"Candidate selector expected base feature dim {self.base_F}, "
                f"got skip={int(skip_feat_bcf.shape[-1])}, cand={int(cand_feat_bcpf.shape[-1])}."
            )
        P = int(cand_feat_bcpf.shape[2])
        if P != self.P:
            raise ValueError(f"Candidate selector expected {self.P} penalties, got {P}.")
        C = int(skip_feat_bcf.shape[1])
        if C != self.C or int(cand_feat_bcpf.shape[1]) != self.C:
            raise ValueError(
                f"Candidate selector expected {self.C} channels, "
                f"got skip={C}, cand={int(cand_feat_bcpf.shape[1])}."
            )
        if self.use_channel_identity:
            channel_eye = torch.eye(self.C, device=skip_feat_bcf.device, dtype=skip_feat_bcf.dtype).view(1, self.C, self.C)
            channel_eye = channel_eye.expand(int(skip_feat_bcf.shape[0]), self.C, self.C)
            skip_feat_bcf = torch.cat([skip_feat_bcf, channel_eye], dim=-1)
            cand_channel_eye = channel_eye.unsqueeze(2).expand(
                int(cand_feat_bcpf.shape[0]),
                self.C,
                self.P,
                self.C,
            )
            cand_feat_bcpf = torch.cat([cand_feat_bcpf, cand_channel_eye], dim=-1)
        if self.use_time_features:
            if query_start_abs_b is None:
                raise ValueError("candidate selector time features require query_start_abs_b.")
            if not self.time_feature_periods:
                raise ValueError("candidate selector time features require at least one positive period.")
            q = query_start_abs_b.to(device=skip_feat_bcf.device, dtype=skip_feat_bcf.dtype).reshape(-1)
            if int(q.numel()) != int(skip_feat_bcf.shape[0]):
                raise ValueError(
                    f"candidate selector expected {int(skip_feat_bcf.shape[0])} query starts, "
                    f"got {int(q.numel())}."
                )
            q = q + float(self.time_feature_offset)
            phase_parts = []
            for period in self.time_feature_periods:
                angle = (2.0 * math.pi / float(period)) * q
                phase_parts.extend([torch.sin(angle), torch.cos(angle)])
            time_feat_bf = torch.stack(phase_parts, dim=-1)
            skip_time = time_feat_bf.view(-1, 1, int(time_feat_bf.shape[-1])).expand(
                int(skip_feat_bcf.shape[0]),
                self.C,
                int(time_feat_bf.shape[-1]),
            )
            cand_time = skip_time.unsqueeze(2).expand(
                int(cand_feat_bcpf.shape[0]),
                self.C,
                self.P,
                int(time_feat_bf.shape[-1]),
            )
            skip_feat_bcf = torch.cat([skip_feat_bcf, skip_time], dim=-1)
            cand_feat_bcpf = torch.cat([cand_feat_bcpf, cand_time], dim=-1)
        if self.use_penalty_identity:
            skip_eye = torch.zeros(
                *skip_feat_bcf.shape[:-1],
                self.P,
                device=skip_feat_bcf.device,
                dtype=skip_feat_bcf.dtype,
            )
            eye = torch.eye(self.P, device=cand_feat_bcpf.device, dtype=cand_feat_bcpf.dtype).view(1, 1, self.P, self.P)
            eye = eye.expand(*cand_feat_bcpf.shape[:2], self.P, self.P)
            skip_feat_bcf = torch.cat([skip_feat_bcf, skip_eye], dim=-1)
            cand_feat_bcpf = torch.cat([cand_feat_bcpf, eye], dim=-1)
        return skip_feat_bcf, cand_feat_bcpf

    def logits_from_features(
        self,
        skip_feat_bcf: torch.Tensor,
        cand_feat_bcpf: torch.Tensor,
        apply_decision_margin: bool = False,
        allowed_mask_cp: Optional[torch.Tensor] = None,
        query_start_abs_b: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        skip_feat_bcf, cand_feat_bcpf = self._append_penalty_identity(
            skip_feat_bcf,
            cand_feat_bcpf,
            query_start_abs_b=query_start_abs_b,
        )
        skip_feat_bcf = self._standardize_feat(skip_feat_bcf)
        cand_feat_bcpf = self._standardize_feat(cand_feat_bcpf)
        skip_score_bc = self.skip_net(skip_feat_bcf).squeeze(-1) + self.skip_bias.view(1, -1)
        cand_score_bcp = self.net(cand_feat_bcpf).squeeze(-1)
        cand_score_bcp = cand_score_bcp + self.penalty_bias.view(1, 1, -1)
        cand_score_bcp = cand_score_bcp + self.penalty_channel_bias.view(1, self.C, self.P)
        if apply_decision_margin:
            margin = torch.as_tensor(
                float(getattr(self, "decision_margin", 0.0)),
                device=cand_score_bcp.device,
                dtype=cand_score_bcp.dtype,
            )
            cand_score_bcp = cand_score_bcp - margin
        logits_bcq = torch.cat([skip_score_bc.unsqueeze(-1), cand_score_bcp], dim=-1)
        return self._mask_disallowed_logits(logits_bcq, allowed_mask_cp=allowed_mask_cp)

    def logits(
        self,
        x_bcl: torch.Tensor,
        base_bch: torch.Tensor,
        cand_bcpH: torch.Tensor,
        allowed_mask_cp: Optional[torch.Tensor] = None,
        query_start_abs_b: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.patch_len > 0:
            B, C, _ = base_bch.shape
            x_patch, base_patch, cand_patch, Q = _candidate_selector_patch_views(
                x_bcl,
                base_bch,
                cand_bcpH,
                self.patch_len,
            )
            query_patch = _candidate_selector_patch_query_starts(
                query_start_abs_b,
                batch_size=B,
                num_patches=Q,
                patch_len=self.patch_len,
            )
            skip_feat = _candidate_selector_features(
                x_patch,
                base_patch,
                base_patch,
                feature_mode=self.feature_mode,
            )
            cand_feat = torch.stack(
                [
                    _candidate_selector_features(
                        x_patch,
                        base_patch,
                        cand_patch[:, :, p, :],
                        feature_mode=self.feature_mode,
                    )
                    for p in range(int(cand_patch.shape[2]))
                ],
                dim=2,
            )
            logits_flat = self.logits_from_features(
                skip_feat,
                cand_feat,
                allowed_mask_cp=allowed_mask_cp,
                query_start_abs_b=query_patch,
            )
            return logits_flat.reshape(B, Q, C, self.P + 1).permute(0, 2, 1, 3).contiguous()
        skip_feat = _candidate_selector_features(x_bcl, base_bch, base_bch, feature_mode=self.feature_mode)
        cand_feat = torch.stack(
            [
                _candidate_selector_features(x_bcl, base_bch, cand_bcpH[:, :, p, :], feature_mode=self.feature_mode)
                for p in range(int(cand_bcpH.shape[2]))
            ],
            dim=2,
        )
        return self.logits_from_features(
            skip_feat,
            cand_feat,
            allowed_mask_cp=allowed_mask_cp,
            query_start_abs_b=query_start_abs_b,
        )

    def select_from_features(
        self,
        skip_feat_bcf: torch.Tensor,
        cand_feat_bcpf: torch.Tensor,
        base_bch: torch.Tensor,
        cand_bcpH: torch.Tensor,
        allowed_mask_cp: Optional[torch.Tensor] = None,
        query_start_abs_b: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits_bcq = self.logits_from_features(
            skip_feat_bcf,
            cand_feat_bcpf,
            apply_decision_margin=True,
            allowed_mask_cp=allowed_mask_cp,
            query_start_abs_b=query_start_abs_b,
        )
        selected_class_bc = logits_bcq.argmax(dim=-1)
        all_pred_bcqh = torch.cat([base_bch.unsqueeze(2), cand_bcpH], dim=2)
        gather_idx = selected_class_bc.view(*selected_class_bc.shape, 1, 1).expand(-1, -1, 1, int(base_bch.shape[-1]))
        selected_bch = all_pred_bcqh.gather(2, gather_idx).squeeze(2)
        return selected_bch, selected_class_bc

    def select_prediction(
        self,
        x_bcl: torch.Tensor,
        base_bch: torch.Tensor,
        cand_bcpH: torch.Tensor,
        allowed_mask_cp: Optional[torch.Tensor] = None,
        query_start_abs_b: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.patch_len > 0:
            B, C, H = base_bch.shape
            x_patch, base_patch, cand_patch, Q = _candidate_selector_patch_views(
                x_bcl,
                base_bch,
                cand_bcpH,
                self.patch_len,
            )
            query_patch = _candidate_selector_patch_query_starts(
                query_start_abs_b,
                batch_size=B,
                num_patches=Q,
                patch_len=self.patch_len,
            )
            skip_feat = _candidate_selector_features(
                x_patch,
                base_patch,
                base_patch,
                feature_mode=self.feature_mode,
            )
            cand_feat = torch.stack(
                [
                    _candidate_selector_features(
                        x_patch,
                        base_patch,
                        cand_patch[:, :, p, :],
                        feature_mode=self.feature_mode,
                    )
                    for p in range(int(cand_patch.shape[2]))
                ],
                dim=2,
            )
            selected_flat, selected_class_flat = self.select_from_features(
                skip_feat,
                cand_feat,
                base_patch,
                cand_patch,
                allowed_mask_cp=allowed_mask_cp,
                query_start_abs_b=query_patch,
            )
            selected = (
                selected_flat.reshape(B, Q, C, self.patch_len)
                .permute(0, 2, 1, 3)
                .reshape(B, C, H)
            )
            selected_class = selected_class_flat.reshape(B, Q, C).permute(0, 2, 1).contiguous()
            return selected, selected_class
        skip_feat = _candidate_selector_features(x_bcl, base_bch, base_bch, feature_mode=self.feature_mode)
        cand_feat = torch.stack(
            [
                _candidate_selector_features(x_bcl, base_bch, cand_bcpH[:, :, p, :], feature_mode=self.feature_mode)
                for p in range(int(cand_bcpH.shape[2]))
            ],
            dim=2,
        )
        return self.select_from_features(
            skip_feat,
            cand_feat,
            base_bch,
            cand_bcpH,
            allowed_mask_cp=allowed_mask_cp,
            query_start_abs_b=query_start_abs_b,
        )


class StaticPredResidualCandidateSelector(nn.Module):
    def __init__(self, selected_class_c: torch.Tensor):
        super().__init__()
        selected = selected_class_c.detach().to(dtype=torch.long).view(-1)
        self.register_buffer("selected_class_c", selected, persistent=True)

    def select_prediction(
        self,
        x_bcl: torch.Tensor,
        base_bch: torch.Tensor,
        cand_bcpH: torch.Tensor,
        allowed_mask_cp: Optional[torch.Tensor] = None,
        query_start_abs_b: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H = base_bch.shape
        if int(self.selected_class_c.numel()) != int(C):
            raise ValueError(
                f"Static candidate selector expected {int(self.selected_class_c.numel())} channels, got {int(C)}."
            )
        selected_class_c = self.selected_class_c.to(device=base_bch.device)
        if allowed_mask_cp is not None and int(allowed_mask_cp.numel()) > 0:
            allowed = _selector_allowed_mask_cp(
                allowed_mask_cp,
                C=int(C),
                P=int(cand_bcpH.shape[2]),
                device=base_bch.device,
                context="static candidate selector",
            )
            penalty_idx_c = (selected_class_c - 1).clamp_min(0)
            candidate_selected_c = selected_class_c > 0
            allowed_selected_c = allowed.gather(1, penalty_idx_c.view(-1, 1)).squeeze(1)
            selected_class_c = torch.where(
                candidate_selected_c & allowed_selected_c,
                selected_class_c,
                torch.zeros_like(selected_class_c),
            )
        selected_class_bc = selected_class_c.view(1, C).expand(B, C)
        all_pred_bcqh = torch.cat([base_bch.unsqueeze(2), cand_bcpH], dim=2)
        gather_idx = selected_class_bc.view(B, C, 1, 1).expand(B, C, 1, H)
        selected_bch = all_pred_bcqh.gather(2, gather_idx).squeeze(2)
        return selected_bch, selected_class_bc


@torch.no_grad()
def _collect_pred_residual_selector_tensors(
    model: nn.Module,
    pred_residual: Optional[ClusterwisePredResidualMoE],
    loader: DataLoader,
    cluster_id_c: torch.Tensor,
    K: int,
    moe_cfg: dict,
    device: torch.device,
    penalty_count: int,
    pred_residual_scale_c: Optional[torch.Tensor] = None,
    history_anchor_cfg: Optional[dict] = None,
    observed_history_tc: Optional[torch.Tensor] = None,
    input_len: int = 0,
    eval_start: int = 0,
    model_train_stat_adapter_pc: Optional[torch.Tensor] = None,
    model_train_stat_adapter_cfg: Optional[dict] = None,
    train_stat_anchor_pc: Optional[torch.Tensor] = None,
    train_residual_anchor_phc: Optional[torch.Tensor] = None,
    learnable_output_anchor: Optional[ClusterwiseLearnableOutputAnchor] = None,
    candidate_feature_mode: str = "base",
) -> Optional[Dict[str, torch.Tensor]]:
    if len(loader) == 0 or pred_residual is None or int(penalty_count) <= 0:
        return None
    if not bool(moe_cfg.get("enable", True)):
        return None

    model.eval()
    pred_residual.eval()
    skip_feat_parts = []
    cand_feat_parts = []
    x_parts = []
    base_parts = []
    cand_parts = []
    confidence_parts = []
    y_parts = []
    query_start_parts = []
    P = int(penalty_count)
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
        mask_all_bkp = torch.ones(x.shape[0], int(K), P, device=device, dtype=y_base.dtype)
        pred_out = pred_residual(
            x,
            y_base,
            cluster_id_c,
            mask_all_bkp,
            skip_bk=None,
            query_start_abs_b=query_start_abs_b,
            fixed_expert_delta_bch=build_moe_output_anchor_fixed_expert_delta(
                y_base,
                x_bcl=x,
                query_start_abs_b=query_start_abs_b,
                input_len=int(input_len),
                moe_cfg=moe_cfg,
                moe_enable=True,
                observed_history_tc=observed_history_tc,
                train_stat_anchor_pc=train_stat_anchor_pc,
                train_residual_anchor_phc=train_residual_anchor_phc,
                learnable_output_anchor=learnable_output_anchor,
                cluster_id_c=cluster_id_c,
            ),
        )
        y_base_final, cand_bcpH = _pred_residual_candidates_on_eval_path(
            y_base,
            pred_out,
            pred_residual_scale_c=pred_residual_scale_c,
            apply_output_anchors=True,
            x_bcl=x,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len),
            moe_cfg=moe_cfg,
            moe_enable=True,
            observed_history_tc=observed_history_tc,
            train_stat_anchor_pc=train_stat_anchor_pc,
            train_residual_anchor_phc=train_residual_anchor_phc,
            learnable_output_anchor=learnable_output_anchor,
            cluster_id_c=cluster_id_c,
            include_patch_route=False,
        )
        if cand_bcpH is None:
            continue
        skip_feat = _candidate_selector_features(x, y_base_final, y_base_final, feature_mode=candidate_feature_mode)
        cand_feat = torch.stack(
            [
                _candidate_selector_features(
                    x,
                    y_base_final,
                    cand_bcpH[:, :, p, :],
                    feature_mode=candidate_feature_mode,
                )
                for p in range(P)
            ],
            dim=2,
        )
        skip_feat_parts.append(skip_feat.detach().cpu())
        cand_feat_parts.append(cand_feat.detach().cpu())
        x_parts.append(x.detach().cpu())
        base_parts.append(y_base_final.detach().cpu())
        cand_parts.append(cand_bcpH.detach().cpu())
        confidence_parts.append(
            pred_out.get("intervention_bcp", torch.ones_like(cand_bcpH[..., 0])).detach().cpu()
        )
        y_parts.append(y.detach().cpu())
        query_start_parts.append(query_start_abs_b.detach().cpu())

    if len(skip_feat_parts) == 0:
        return None
    return {
        "skip_feat": torch.cat(skip_feat_parts, dim=0),
        "cand_feat": torch.cat(cand_feat_parts, dim=0),
        "x": torch.cat(x_parts, dim=0),
        "base": torch.cat(base_parts, dim=0),
        "cand": torch.cat(cand_parts, dim=0),
        "confidence": torch.cat(confidence_parts, dim=0),
        "y": torch.cat(y_parts, dim=0),
        "query_start_abs": torch.cat(query_start_parts, dim=0),
    }


def _concat_pred_residual_selector_tensors(
    tensor_parts: Iterable[Optional[Dict[str, torch.Tensor]]],
) -> Optional[Dict[str, torch.Tensor]]:
    parts = [part for part in tensor_parts if part is not None]
    if not parts:
        return None
    keys = ["skip_feat", "cand_feat", "base", "cand", "confidence", "y"]
    out: Dict[str, torch.Tensor] = {}
    for key in keys:
        tensors = [part[key] for part in parts if key in part]
        if len(tensors) != len(parts):
            raise ValueError(f"Cannot concatenate selector tensors: missing key {key!r}.")
        out[key] = torch.cat(tensors, dim=0)
    if all("x" in part for part in parts):
        out["x"] = torch.cat([part["x"] for part in parts], dim=0)
    if all("query_start_abs" in part for part in parts):
        out["query_start_abs"] = torch.cat([part["query_start_abs"] for part in parts], dim=0)
    return out


def _patchify_pred_residual_selector_tensors(
    tensors: Dict[str, torch.Tensor],
    *,
    patch_len: int,
    feature_mode: str,
) -> Dict[str, torch.Tensor]:
    x = tensors.get("x")
    if x is None:
        raise ValueError("candidate_selector.patch_len requires collected input tensors.")
    base = tensors["base"]
    cand = tensors["cand"]
    y = tensors["y"]
    B, C, H = base.shape
    x_patch, base_patch, cand_patch, Q = _candidate_selector_patch_views(
        x,
        base,
        cand,
        int(patch_len),
    )
    if tuple(y.shape) != tuple(base.shape):
        raise ValueError("Patch candidate selector target shape must match base shape.")
    y_patch = y.reshape(B, C, Q, int(patch_len)).permute(0, 2, 1, 3).reshape(B * Q, C, int(patch_len))
    skip_feat = _candidate_selector_features(
        x_patch,
        base_patch,
        base_patch,
        feature_mode=feature_mode,
    )
    cand_feat = torch.stack(
        [
            _candidate_selector_features(
                x_patch,
                base_patch,
                cand_patch[:, :, p, :],
                feature_mode=feature_mode,
            )
            for p in range(int(cand_patch.shape[2]))
        ],
        dim=2,
    )
    out = {
        "x": x_patch,
        "skip_feat": skip_feat,
        "cand_feat": cand_feat,
        "base": base_patch,
        "cand": cand_patch,
        "y": y_patch,
        "patch_count": torch.tensor(int(Q), dtype=torch.long),
    }
    confidence = tensors.get("confidence")
    if confidence is not None:
        P = int(confidence.shape[-1])
        out["confidence"] = (
            confidence.view(B, 1, C, P)
            .expand(B, Q, C, P)
            .reshape(B * Q, C, P)
        )
    query_start_abs = tensors.get("query_start_abs")
    if query_start_abs is not None:
        query_patch = _candidate_selector_patch_query_starts(
            query_start_abs,
            batch_size=B,
            num_patches=Q,
            patch_len=int(patch_len),
        )
        if query_patch is not None:
            out["query_start_abs"] = query_patch
    return out


@torch.no_grad()
def _select_pred_residual_confidence_thresholds_from_tensors(
    *,
    tensors: Optional[Dict[str, torch.Tensor]],
    cluster_id_c: torch.Tensor,
    K: int,
    allowed_mask_kp: Optional[torch.Tensor] = None,
    penalty_names: Optional[List[str]] = None,
    min_abs_improvement: float = 0.0,
    min_rel_improvement: float = 0.0,
    max_candidates: int = 101,
    selection_metric: str = "mse",
    min_precision: float = 0.0,
    max_pred_positive_rate: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    if tensors is None:
        raise ValueError("confidence threshold selection requires collected train tensors.")
    base = tensors.get("base")
    cand = tensors.get("cand")
    y = tensors.get("y")
    confidence = tensors.get("confidence")
    if base is None or cand is None or y is None or confidence is None:
        raise ValueError("confidence threshold selection requires base, cand, y, and confidence tensors.")
    if cand.ndim != 4:
        raise ValueError(f"candidate tensor must have shape [B,C,P,H], got {tuple(cand.shape)}")
    B, C, P, _ = cand.shape
    if base.shape[:2] != (B, C) or y.shape[:2] != (B, C) or confidence.shape != (B, C, P):
        raise ValueError(
            "confidence threshold tensors have inconsistent shapes: "
            f"base={tuple(base.shape)}, cand={tuple(cand.shape)}, "
            f"y={tuple(y.shape)}, confidence={tuple(confidence.shape)}"
        )
    cluster_id = cluster_id_c.detach().cpu().to(dtype=torch.long).view(-1)
    if int(cluster_id.numel()) != int(C):
        raise ValueError(f"cluster_id_c must have one entry per channel, got {int(cluster_id.numel())} vs {int(C)}")
    K = int(K)
    if K <= 0:
        raise ValueError("K must be positive for confidence threshold selection.")
    metric_mode = str(selection_metric).lower()
    if metric_mode not in {"mse", "precision_guarded_mse"}:
        raise ValueError(
            "confidence threshold selection currently supports selection_metric='mse' "
            "or 'precision_guarded_mse'."
        )

    if allowed_mask_kp is None:
        allowed = torch.ones(K, P, dtype=torch.bool)
    else:
        if allowed_mask_kp.shape != (K, P):
            raise ValueError(f"allowed_mask_kp must have shape [{K},{P}], got {tuple(allowed_mask_kp.shape)}")
        allowed = allowed_mask_kp.detach().cpu().to(dtype=torch.float32) > 0.0

    names = list(penalty_names or [f"p{p}" for p in range(P)])
    if len(names) != P:
        names = [f"p{p}" for p in range(P)]

    min_abs = max(0.0, float(min_abs_improvement))
    min_rel = max(0.0, float(min_rel_improvement))
    min_precision = max(0.0, float(min_precision))
    max_rate = None
    if max_pred_positive_rate is not None:
        max_rate = min(1.0, max(0.0, float(max_pred_positive_rate)))
    max_candidates = max(2, int(max_candidates))
    base_err_bc = (base - y).pow(2).mean(dim=-1).detach().cpu()
    cand_err_bcp = (cand - y.unsqueeze(2)).pow(2).mean(dim=-1).detach().cpu()
    confidence_bcp = confidence.detach().cpu().to(dtype=torch.float32)
    thresholds = torch.full((K, P), 1.000001, dtype=torch.float32)
    per_cluster_penalty: Dict[str, Dict[str, object]] = {}

    for k in range(K):
        channel_mask_c = cluster_id == int(k)
        per_penalty: Dict[str, object] = {}
        for p in range(P):
            name = str(names[p])
            if not bool(allowed[k, p].item()):
                per_penalty[name] = {
                    "allowed": False,
                    "threshold": float(thresholds[k, p].item()),
                    "samples": 0,
                    "reason": "disallowed",
                }
                continue
            if not bool(channel_mask_c.any().item()):
                per_penalty[name] = {
                    "allowed": True,
                    "threshold": float(thresholds[k, p].item()),
                    "samples": 0,
                    "reason": "empty_cluster",
                }
                continue

            score_v = confidence_bcp[:, channel_mask_c, p].reshape(-1)
            base_err_v = base_err_bc[:, channel_mask_c].reshape(-1)
            cand_err_v = cand_err_bcp[:, channel_mask_c, p].reshape(-1)
            finite = torch.isfinite(score_v) & torch.isfinite(base_err_v) & torch.isfinite(cand_err_v)
            score_v = score_v[finite]
            base_err_v = base_err_v[finite]
            cand_err_v = cand_err_v[finite]
            if int(score_v.numel()) == 0:
                per_penalty[name] = {
                    "allowed": True,
                    "threshold": float(thresholds[k, p].item()),
                    "samples": 0,
                    "reason": "empty_scores",
                }
                continue

            gain_v = base_err_v - cand_err_v
            required_v = torch.maximum(
                torch.full_like(gain_v, min_abs),
                min_rel * base_err_v.abs().clamp_min(1.0e-12),
            )
            positive_v = gain_v > required_v
            if int(positive_v.sum().item()) == 0:
                threshold = float(score_v.max().item()) + 1.0e-6
                thresholds[k, p] = float(threshold)
                per_penalty[name] = {
                    "allowed": True,
                    "threshold": threshold,
                    "samples": int(score_v.numel()),
                    "reason": "no_positive_gain_labels",
                    "positive_rate": 0.0,
                    "base_mse": float(base_err_v.mean().item()),
                    "selected_mse": float(base_err_v.mean().item()),
                    "selected_gain_pct_vs_base": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "pred_positive_rate": 0.0,
                }
                continue

            uniq = torch.unique(score_v)
            if int(uniq.numel()) > max_candidates:
                quantiles = torch.linspace(0.0, 1.0, steps=max_candidates, dtype=score_v.dtype)
                candidates_v = torch.unique(torch.quantile(score_v, quantiles))
            else:
                candidates_v = uniq
            candidates_v = torch.unique(
                torch.cat(
                    [
                        torch.tensor([0.0], dtype=score_v.dtype),
                        candidates_v,
                        torch.tensor([float(score_v.max().item()) + 1.0e-6], dtype=score_v.dtype),
                    ],
                    dim=0,
                )
            )
            base_mse = float(base_err_v.mean().item())
            skip_threshold = float(score_v.max().item()) + 1.0e-6
            best = {
                "threshold": skip_threshold,
                "mse": base_mse,
                "precision": 0.0,
                "recall": 0.0,
                "pred_positive_rate": 0.0,
                "reason": "no_improving_threshold",
                "skip_selected": True,
            }
            feasible_count = 0
            for candidate_threshold in candidates_v.tolist():
                pred_active = score_v >= float(candidate_threshold)
                selected_err = torch.where(pred_active, cand_err_v, base_err_v)
                selected_mse = float(selected_err.mean().item())
                tp = int((pred_active & positive_v).sum().item())
                fp = int((pred_active & (~positive_v)).sum().item())
                fn = int(((~pred_active) & positive_v).sum().item())
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                pred_rate = float(pred_active.to(dtype=torch.float32).mean().item())
                if metric_mode == "precision_guarded_mse":
                    if precision + 1.0e-12 < min_precision:
                        continue
                    if max_rate is not None and pred_rate > float(max_rate) + 1.0e-12:
                        continue
                feasible_count += 1
                better = selected_mse < float(best["mse"]) - 1.0e-12
                if not better and abs(selected_mse - float(best["mse"])) <= 1.0e-12:
                    better = precision > float(best["precision"]) + 1.0e-12
                if not better and abs(selected_mse - float(best["mse"])) <= 1.0e-12 and abs(precision - float(best["precision"])) <= 1.0e-12:
                    better = float(candidate_threshold) > float(best["threshold"])
                if better:
                    best = {
                        "threshold": float(candidate_threshold),
                        "mse": selected_mse,
                        "precision": float(precision),
                        "recall": float(recall),
                        "pred_positive_rate": pred_rate,
                        "reason": "selected",
                        "skip_selected": False,
                    }
            if metric_mode == "precision_guarded_mse" and feasible_count == 0:
                best["reason"] = "no_threshold_meets_confidence_guard"
            thresholds[k, p] = float(best["threshold"])
            per_penalty[name] = {
                "allowed": True,
                "threshold": float(best["threshold"]),
                "samples": int(score_v.numel()),
                "feasible_threshold_count": int(feasible_count),
                "reason": str(best["reason"]),
                "skip_selected": bool(best["skip_selected"]),
                "positive_rate": float(positive_v.to(dtype=torch.float32).mean().item()),
                "base_mse": base_mse,
                "selected_mse": float(best["mse"]),
                "selected_gain_pct_vs_base": float(
                    100.0 * (base_mse - float(best["mse"])) / max(abs(base_mse), 1.0e-12)
                ),
                "precision": float(best["precision"]),
                "recall": float(best["recall"]),
                "pred_positive_rate": float(best["pred_positive_rate"]),
            }
        per_cluster_penalty[str(k)] = per_penalty

    summary = {
        "enable": True,
        "source_requirement": "train_only",
        "selection_metric": metric_mode,
        "min_abs_improvement": float(min_abs),
        "min_rel_improvement": float(min_rel),
        "min_precision": float(min_precision),
        "max_pred_positive_rate": None if max_rate is None else float(max_rate),
        "threshold_kp": [[float(v) for v in row] for row in thresholds.tolist()],
        "penalty_names": names,
        "per_cluster_penalty": per_cluster_penalty,
    }
    return thresholds, summary


@torch.no_grad()
def _candidate_selector_feature_gain_diagnostics(
    *,
    tensors: Optional[Dict[str, torch.Tensor]],
    feature_names: Optional[List[str]] = None,
    penalty_names: Optional[List[str]] = None,
    allowed_mask_cp: Optional[torch.Tensor] = None,
    indices: Optional[torch.Tensor] = None,
    cluster_id_c: Optional[torch.Tensor] = None,
    K: Optional[int] = None,
    min_abs_improvement: float = 0.0,
    min_rel_improvement: float = 0.0,
    topk: int = 5,
) -> Optional[Dict[str, object]]:
    if tensors is None:
        return None
    cand_feat = tensors.get("cand_feat")
    base = tensors.get("base")
    cand = tensors.get("cand")
    y = tensors.get("y")
    if cand_feat is None or base is None or cand is None or y is None:
        return None
    n = int(base.shape[0])
    if n <= 0:
        return None
    if indices is None:
        indices = torch.arange(0, n, dtype=torch.long)
    else:
        indices = indices.detach().cpu().to(dtype=torch.long)
    if int(indices.numel()) == 0:
        return None

    cand_feat = cand_feat.index_select(0, indices)
    base = base.index_select(0, indices)
    cand = cand.index_select(0, indices)
    y = y.index_select(0, indices)
    B, C, P, F = cand_feat.shape
    names = list(feature_names or [f"f{i}" for i in range(F)])
    if len(names) != F:
        names = [f"f{i}" for i in range(F)]
    penalties = list(penalty_names or [f"p{i}" for i in range(P)])
    if len(penalties) != P:
        penalties = [f"p{i}" for i in range(P)]

    allowed = _selector_allowed_mask_cp(
        allowed_mask_cp,
        C=int(C),
        P=int(P),
        device=cand_feat.device,
        context="candidate selector feature diagnostics",
    )
    valid_bcp = torch.ones((B, C, P), dtype=torch.bool, device=cand_feat.device)
    if allowed is not None:
        valid_bcp = valid_bcp & allowed.unsqueeze(0)

    base_err_bc = (base - y).pow(2).mean(dim=-1)
    cand_err_bcp = (cand - y.unsqueeze(2)).pow(2).mean(dim=-1)
    gain_bcp = base_err_bc.unsqueeze(-1) - cand_err_bcp
    required_bcp = torch.maximum(
        torch.full_like(gain_bcp, max(0.0, float(min_abs_improvement))),
        max(0.0, float(min_rel_improvement)) * base_err_bc.abs().unsqueeze(-1).clamp_min(1.0e-12),
    )
    positive_bcp = gain_bcp > required_bcp

    finite_bcpf = torch.isfinite(cand_feat).all(dim=-1)
    finite_bcp = finite_bcpf & torch.isfinite(gain_bcp)
    valid_bcp = valid_bcp & finite_bcp

    def _corr_table(feat_nf: torch.Tensor, target_n: torch.Tensor, limit: int) -> List[Dict[str, object]]:
        if int(feat_nf.shape[0]) <= 1:
            return [
                {"feature": names[i], "corr": 0.0, "abs_corr": 0.0, "std": 0.0}
                for i in range(min(int(limit), int(feat_nf.shape[1])))
            ]
        x = feat_nf.to(dtype=torch.float64)
        target = target_n.to(dtype=torch.float64)
        y_center = target - target.mean()
        y_ss = y_center.pow(2).sum()
        x_center = x - x.mean(dim=0, keepdim=True)
        x_ss = x_center.pow(2).sum(dim=0)
        denom = (x_ss * y_ss).clamp_min(0.0).sqrt()
        corr_f = torch.where(denom > 1.0e-12, (x_center * y_center.unsqueeze(-1)).sum(dim=0) / denom, torch.zeros_like(denom))
        std_f = x.std(dim=0, unbiased=False)
        order = sorted(range(int(x.shape[1])), key=lambda i: (-abs(float(corr_f[i].item())), i))
        rows = []
        for i in order[: max(0, int(limit))]:
            corr = float(corr_f[i].item())
            rows.append(
                {
                    "feature": names[i],
                    "corr": corr,
                    "abs_corr": abs(corr),
                    "std": float(std_f[i].item()),
                }
            )
        return rows

    def _summarize(mask_bcp: torch.Tensor) -> Dict[str, object]:
        flat_mask = mask_bcp.reshape(-1)
        samples = int(flat_mask.sum().item())
        if samples <= 0:
            return {
                "samples": 0,
                "positive_rate": 0.0,
                "mean_gain": 0.0,
                "top_abs_gain_corr": [],
                "top_abs_positive_corr": [],
            }
        feat_nf = cand_feat.reshape(-1, F)[flat_mask]
        gain_n = gain_bcp.reshape(-1)[flat_mask]
        positive_n = positive_bcp.reshape(-1)[flat_mask].to(dtype=torch.float32)
        return {
            "samples": samples,
            "positive_rate": float(positive_n.mean().item()),
            "mean_gain": float(gain_n.mean().item()),
            "top_abs_gain_corr": _corr_table(feat_nf, gain_n, int(topk)),
            "top_abs_positive_corr": _corr_table(feat_nf, positive_n, int(topk)),
        }

    summary = _summarize(valid_bcp)
    summary["feature_names"] = names
    summary["penalty_names"] = penalties
    summary["min_abs_improvement"] = float(max(0.0, float(min_abs_improvement)))
    summary["min_rel_improvement"] = float(max(0.0, float(min_rel_improvement)))
    summary["allowed_candidates"] = int(valid_bcp.sum().item())
    by_penalty = {}
    for p_idx, name in enumerate(penalties):
        penalty_mask = torch.zeros_like(valid_bcp)
        penalty_mask[:, :, p_idx] = valid_bcp[:, :, p_idx]
        by_penalty[str(name)] = _summarize(penalty_mask)
    summary["by_penalty"] = by_penalty

    if cluster_id_c is not None:
        cluster_idx = cluster_id_c.detach().cpu().to(dtype=torch.long)
        if int(cluster_idx.numel()) == int(C):
            k_count = int(K) if K is not None else int(cluster_idx.max().item() + 1)
            by_cluster = {}
            for k in range(k_count):
                channel_mask_c = (cluster_idx == int(k)).to(device=valid_bcp.device)
                cluster_mask = valid_bcp & channel_mask_c.view(1, C, 1)
                by_cluster[str(k)] = _summarize(cluster_mask)
            summary["by_cluster"] = by_cluster
    return summary


@torch.no_grad()
def _pred_residual_selector_metrics_from_tensors(
    tensors: Optional[Dict[str, torch.Tensor]],
    selector: Optional[PredResidualCandidateSelector],
    device: torch.device,
    batch_size: int,
    min_abs_improvement: float,
    min_rel_improvement: float,
    indices: Optional[torch.Tensor] = None,
    penalty_names: Optional[List[str]] = None,
    allowed_mask_cp: Optional[torch.Tensor] = None,
) -> Optional[Dict[str, object]]:
    if tensors is None or selector is None:
        return None
    skip_feat = tensors["skip_feat"]
    cand_feat = tensors["cand_feat"]
    base = tensors["base"]
    cand = tensors["cand"]
    y = tensors["y"]
    query_start_abs = tensors.get("query_start_abs")
    query_start_abs = tensors.get("query_start_abs")
    n = int(base.shape[0])
    if n == 0:
        return None
    if indices is None:
        indices = torch.arange(0, n, dtype=torch.long)
    else:
        indices = indices.detach().cpu().to(dtype=torch.long)
    if int(indices.numel()) == 0:
        return None

    selector.eval()
    P = int(cand.shape[2])
    q = P + 1
    allowed_mask_device = _selector_allowed_mask_cp(
        allowed_mask_cp,
        C=int(base.shape[1]),
        P=P,
        device=device,
        context="candidate selector metrics",
    )
    batch_size = max(1, int(batch_size))
    base_se = selected_se = target_se = oracle_se = 0.0
    denom = 0
    correct = 0
    total_bc = 0
    selected_count_q = torch.zeros(q, dtype=torch.long)
    target_count_q = torch.zeros(q, dtype=torch.long)
    oracle_count_q = torch.zeros(q, dtype=torch.long)
    for b0 in range(0, int(indices.numel()), batch_size):
        batch_idx = indices[b0:b0 + batch_size]
        skip_feat_b = skip_feat.index_select(0, batch_idx).to(device)
        cand_feat_b = cand_feat.index_select(0, batch_idx).to(device)
        base_b = base.index_select(0, batch_idx).to(device)
        cand_b = cand.index_select(0, batch_idx).to(device)
        y_b = y.index_select(0, batch_idx).to(device)
        query_b = None
        if query_start_abs is not None:
            query_b = query_start_abs.index_select(0, batch_idx).to(device)
        selected_b, selected_class_b = selector.select_from_features(
            skip_feat_b,
            cand_feat_b,
            base_b,
            cand_b,
            allowed_mask_cp=allowed_mask_device,
            query_start_abs_b=query_b,
        )
        target_b = _candidate_selector_targets(
            base_bch=base_b,
            cand_bcpH=cand_b,
            y_bch=y_b,
            min_abs_improvement=min_abs_improvement,
            min_rel_improvement=min_rel_improvement,
            allowed_mask_cp=allowed_mask_device,
        )
        all_pred_bq = torch.cat([base_b.unsqueeze(2), cand_b], dim=2)
        target_pred_b = all_pred_bq.gather(
            2,
            target_b.view(*target_b.shape, 1, 1).expand(-1, -1, 1, int(base_b.shape[-1])),
        ).squeeze(2)
        base_err_bc = (base_b - y_b).pow(2).mean(dim=-1)
        cand_err_bcp = (cand_b - y_b.unsqueeze(2)).pow(2).mean(dim=-1)
        if allowed_mask_device is not None:
            cand_err_bcp = cand_err_bcp.masked_fill(~allowed_mask_device.unsqueeze(0), float("inf"))
        best_cand_err_bc, oracle_p_bc = cand_err_bcp.min(dim=-1)
        oracle_use_bc = torch.isfinite(best_cand_err_bc) & (best_cand_err_bc < base_err_bc)
        oracle_err_bc = torch.where(oracle_use_bc, best_cand_err_bc, base_err_bc)
        oracle_class_bc = torch.where(
            oracle_use_bc,
            oracle_p_bc.to(dtype=torch.long) + 1,
            torch.zeros_like(oracle_p_bc, dtype=torch.long),
        )
        selected_se += float((selected_b - y_b).pow(2).sum().item())
        target_se += float((target_pred_b - y_b).pow(2).sum().item())
        oracle_se += float(oracle_err_bc.sum().item() * y_b.shape[-1])
        base_se += float((base_b - y_b).pow(2).sum().item())
        denom += int(y_b.numel())
        total_bc += int(target_b.numel())
        correct += int((selected_class_b == target_b).sum().item())
        selected_count_q += torch.bincount(selected_class_b.detach().cpu().reshape(-1), minlength=q)[:q]
        target_count_q += torch.bincount(target_b.detach().cpu().reshape(-1), minlength=q)[:q]
        oracle_count_q += torch.bincount(oracle_class_bc.detach().cpu().reshape(-1), minlength=q)[:q]

    if denom <= 0:
        return None
    base_mse = base_se / max(denom, 1)
    selected_mse = selected_se / max(denom, 1)
    target_mse = target_se / max(denom, 1)
    oracle_mse = oracle_se / max(denom, 1)
    names = ["skip"] + [str(name) for name in (penalty_names or [f"p{p}" for p in range(P)])]
    return {
        "samples": int(total_bc),
        "accuracy": float(correct / max(total_bc, 1)),
        "base_mse": float(base_mse),
        "selected_mse": float(selected_mse),
        "target_mse": float(target_mse),
        "oracle_mse": float(oracle_mse),
        "selected_gain_pct_vs_base": float(100.0 * (base_mse - selected_mse) / max(abs(base_mse), 1.0e-12)),
        "target_gain_pct_vs_base": float(100.0 * (base_mse - target_mse) / max(abs(base_mse), 1.0e-12)),
        "oracle_gain_pct_vs_base": float(100.0 * (base_mse - oracle_mse) / max(abs(base_mse), 1.0e-12)),
        "selected_class_rate": {
            names[i]: float(selected_count_q[i].item() / max(total_bc, 1)) for i in range(q)
        },
        "target_class_rate": {
            names[i]: float(target_count_q[i].item() / max(total_bc, 1)) for i in range(q)
        },
        "oracle_class_rate": {
            names[i]: float(oracle_count_q[i].item() / max(total_bc, 1)) for i in range(q)
        },
    }


@torch.no_grad()
def _pred_residual_selector_temporal_block_metrics(
    *,
    tensors: Optional[Dict[str, torch.Tensor]],
    selector: Optional[PredResidualCandidateSelector],
    device: torch.device,
    batch_size: int,
    num_blocks: int,
    min_abs_improvement: float,
    min_rel_improvement: float,
    indices: Optional[torch.Tensor] = None,
    penalty_names: Optional[List[str]] = None,
    allowed_mask_cp: Optional[torch.Tensor] = None,
) -> List[Dict[str, object]]:
    if tensors is None or selector is None:
        return []
    base = tensors.get("base")
    if base is None:
        return []
    n = int(base.shape[0])
    if n <= 0:
        return []
    if indices is None:
        ordered_indices = torch.arange(0, n, dtype=torch.long)
    else:
        ordered_indices = indices.detach().cpu().to(dtype=torch.long)
        ordered_indices = ordered_indices[(ordered_indices >= 0) & (ordered_indices < n)]
    if int(ordered_indices.numel()) <= 0:
        return []
    total = int(ordered_indices.numel())
    blocks = max(1, min(int(num_blocks), total))
    out: List[Dict[str, object]] = []
    for block_idx in range(blocks):
        start = (total * block_idx) // blocks
        end = (total * (block_idx + 1)) // blocks
        if end <= start:
            continue
        block_indices = ordered_indices[start:end]
        metrics = _pred_residual_selector_metrics_from_tensors(
            tensors=tensors,
            selector=selector,
            device=device,
            batch_size=batch_size,
            min_abs_improvement=min_abs_improvement,
            min_rel_improvement=min_rel_improvement,
            indices=block_indices,
            penalty_names=penalty_names,
            allowed_mask_cp=allowed_mask_cp,
        )
        if metrics is None:
            continue
        payload = dict(metrics)
        payload["block_index"] = int(block_idx)
        payload["start_window"] = int(block_indices[0].item())
        payload["end_window"] = int(block_indices[-1].item() + 1)
        payload["window_count"] = int(block_indices.numel())
        out.append(payload)
    return out


def _candidate_selector_feature_standardization_stats(
    *,
    skip_feat: torch.Tensor,
    cand_feat: torch.Tensor,
    selector: PredResidualCandidateSelector,
    train_idx: torch.Tensor,
    query_start_abs: Optional[torch.Tensor] = None,
    mode: str = "mean_std",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    skip_train_raw = skip_feat.index_select(0, train_idx)
    cand_train_raw = cand_feat.index_select(0, train_idx)
    query_train = None
    if query_start_abs is not None:
        query_train = query_start_abs.index_select(0, train_idx)
    skip_train_aug, cand_train_aug = selector._append_penalty_identity(
        skip_train_raw,
        cand_train_raw,
        query_start_abs_b=query_train,
    )
    skip_train = skip_train_aug.reshape(-1, int(skip_train_aug.shape[-1]))
    cand_train = cand_train_aug.reshape(-1, int(cand_train_aug.shape[-1]))
    feat_train = torch.cat([skip_train, cand_train], dim=0).to(dtype=torch.float32)
    mode_norm = str(mode or "mean_std").lower()
    if mode_norm in {"robust", "median_iqr", "iqr"}:
        feat_mean = feat_train.median(dim=0).values
        q = torch.quantile(feat_train, torch.tensor([0.25, 0.75], device=feat_train.device), dim=0)
        iqr_scale = (q[1] - q[0]) / 1.349
        std_fallback = feat_train.std(dim=0, unbiased=False)
        feat_std = torch.where(iqr_scale > 1.0e-6, iqr_scale, std_fallback).clamp_min(1.0e-6)
        mode_summary = "robust"
    elif mode_norm in {"mean_std", "standard", "std"}:
        feat_mean = feat_train.mean(dim=0)
        feat_std = feat_train.std(dim=0, unbiased=False).clamp_min(1.0e-6)
        mode_summary = "mean_std"
    else:
        raise ValueError("candidate_selector.standardization_mode must be mean_std or robust.")
    summary = {
        "mode": mode_summary,
        "min_std": float(feat_std.min().item()),
        "max_std": float(feat_std.max().item()),
    }
    return feat_mean, feat_std, summary


def _candidate_selector_select_confirm_indices(
    total_windows: int,
    confirm_fraction: float,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    total = int(total_windows)
    fraction = float(confirm_fraction)
    if total <= 1 or fraction <= 0.0:
        return None, None
    if fraction >= 1.0:
        raise ValueError("selection_confirm_fraction must be >= 0.0 and < 1.0.")
    confirm_n = int(math.floor(float(total) * fraction + 0.5))
    confirm_n = min(max(confirm_n, 1), total - 1)
    select_n = total - confirm_n
    return torch.arange(0, select_n, dtype=torch.long), torch.arange(select_n, total, dtype=torch.long)


@torch.no_grad()
def _fit_static_candidate_channel_selector_from_tensors(
    *,
    tensors: Dict[str, torch.Tensor],
    allowed_mask_cp: Optional[torch.Tensor] = None,
    penalty_names: Optional[List[str]] = None,
    channel_names: Optional[List[str]] = None,
    select_indices: Optional[torch.Tensor] = None,
    eval_indices: Optional[torch.Tensor] = None,
    min_abs_improvement: float = 0.0,
    min_rel_improvement: float = 0.0,
    min_abs_mae_improvement: Optional[float] = None,
    selection_metric: str = "mse",
    confirm_min_abs_improvement: Optional[float] = None,
    confirm_min_rel_improvement: float = 0.0,
    confirm_min_abs_mae_improvement: Optional[float] = None,
    segment_count: int = 0,
    segment_min_positive: Optional[int] = None,
    segment_min_abs_improvement: Optional[float] = None,
    segment_min_abs_mae_improvement: Optional[float] = None,
    metric_max_elements: int = 16_000_000,
) -> Tuple[StaticPredResidualCandidateSelector, Dict[str, object]]:
    base = tensors["base"]
    cand = tensors["cand"]
    y = tensors["y"]
    n, C, _ = base.shape
    P = int(cand.shape[2])
    if select_indices is None:
        select_indices = torch.arange(0, n, dtype=torch.long)
    else:
        select_indices = select_indices.detach().cpu().to(dtype=torch.long)
    if eval_indices is None:
        eval_indices = select_indices
    else:
        eval_indices = eval_indices.detach().cpu().to(dtype=torch.long)
    allowed = _selector_allowed_mask_cp(
        allowed_mask_cp,
        C=int(C),
        P=P,
        device=base.device,
        context="static candidate channel selector",
    )
    metric_mode = _normalize_pred_residual_candidate_selection_metric(selection_metric)

    def _mse_mae_for_indices(indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if int(indices.numel()) <= 0:
            raise ValueError("static candidate channel selector metric indices must be non-empty.")
        horizon = int(base.shape[2])
        max_elements = max(1, int(metric_max_elements))
        elements_per_window = max(1, int(C) * max(1, P) * horizon)
        chunk_size = max(1, min(int(indices.numel()), max_elements // elements_per_window))
        sum_dtype = torch.float64
        base_mse_sum_c = torch.zeros(C, device=base.device, dtype=sum_dtype)
        base_mae_sum_c = torch.zeros_like(base_mse_sum_c)
        cand_mse_sum_cp = torch.zeros(C, P, device=base.device, dtype=sum_dtype)
        cand_mae_sum_cp = torch.zeros_like(cand_mse_sum_cp)
        for index_chunk in indices.split(chunk_size):
            index_chunk = index_chunk.to(device=base.device, dtype=torch.long)
            base_i = base.index_select(0, index_chunk)
            cand_i = cand.index_select(0, index_chunk)
            y_i = y.index_select(0, index_chunk)
            base_delta = base_i - y_i
            base_mse_sum_c += base_delta.square().sum(dim=(0, 2), dtype=sum_dtype)
            base_mae_sum_c += base_delta.abs().sum(dim=(0, 2), dtype=sum_dtype)
            cand_delta = cand_i - y_i.unsqueeze(2)
            cand_mse_sum_cp += cand_delta.square().sum(dim=(0, 3), dtype=sum_dtype)
            cand_mae_sum_cp += cand_delta.abs().sum(dim=(0, 3), dtype=sum_dtype)
        denominator = float(int(indices.numel()) * horizon)
        base_mse_c = (base_mse_sum_c / denominator).to(dtype=base.dtype)
        base_mae_c = (base_mae_sum_c / denominator).to(dtype=base.dtype)
        cand_mse_cp = (cand_mse_sum_cp / denominator).to(dtype=cand.dtype)
        cand_mae_cp = (cand_mae_sum_cp / denominator).to(dtype=cand.dtype)
        if allowed is not None:
            cand_mse_cp = cand_mse_cp.masked_fill(~allowed, float("inf"))
            cand_mae_cp = cand_mae_cp.masked_fill(~allowed, float("inf"))
        return base_mse_c, base_mae_c, cand_mse_cp, cand_mae_cp

    select_base_mse_c, select_base_mae_c, select_cand_mse_cp, select_cand_mae_cp = _mse_mae_for_indices(select_indices)
    mae_guard_enabled = min_abs_mae_improvement is not None
    mae_required_c = torch.full_like(
        select_base_mae_c,
        max(0.0, float(min_abs_mae_improvement)) if mae_guard_enabled else 0.0,
    )
    select_base_metric_c = select_base_mae_c if metric_mode == "mae" else select_base_mse_c
    select_cand_metric_cp = select_cand_mae_cp if metric_mode == "mae" else select_cand_mse_cp
    choice_cand_metric_cp = select_cand_metric_cp
    if mae_guard_enabled:
        mae_gain_cp = select_base_mae_c.unsqueeze(1) - select_cand_mae_cp
        choice_cand_metric_cp = choice_cand_metric_cp.masked_fill(
            mae_gain_cp < mae_required_c.unsqueeze(1),
            float("inf"),
        )
    best_cand_metric_c, best_p_c = choice_cand_metric_cp.min(dim=-1)
    best_cand_mse_c = select_cand_mse_cp.gather(1, best_p_c.view(-1, 1)).squeeze(1)
    best_cand_mae_c = select_cand_mae_cp.gather(1, best_p_c.view(-1, 1)).squeeze(1)
    required_c = torch.maximum(
        torch.full_like(select_base_metric_c, max(0.0, float(min_abs_improvement))),
        max(0.0, float(min_rel_improvement)) * select_base_metric_c.abs().clamp_min(1.0e-12),
    )
    use_candidate_c = torch.isfinite(best_cand_metric_c) & ((select_base_metric_c - best_cand_metric_c) > required_c)

    eval_base_mse_c, eval_base_mae_c, eval_cand_mse_cp, eval_cand_mae_cp = _mse_mae_for_indices(eval_indices)
    safe_p_c = best_p_c.clamp_min(0)
    chosen_cand_mse_c = eval_cand_mse_cp.gather(1, safe_p_c.view(-1, 1)).squeeze(1)
    chosen_cand_mae_c = eval_cand_mae_cp.gather(1, safe_p_c.view(-1, 1)).squeeze(1)
    eval_base_metric_c = eval_base_mae_c if metric_mode == "mae" else eval_base_mse_c
    chosen_cand_metric_c = chosen_cand_mae_c if metric_mode == "mae" else chosen_cand_mse_c

    confirm_guard_enabled = confirm_min_abs_improvement is not None
    confirm_required_c = torch.full_like(
        eval_base_metric_c,
        max(0.0, float(confirm_min_abs_improvement)) if confirm_guard_enabled else 0.0,
    )
    if confirm_guard_enabled:
        confirm_required_c = torch.maximum(
            confirm_required_c,
            max(0.0, float(confirm_min_rel_improvement)) * eval_base_metric_c.abs().clamp_min(1.0e-12),
        )
        confirm_metric_gain_c = eval_base_metric_c - chosen_cand_metric_c
        use_candidate_c = use_candidate_c & (confirm_metric_gain_c > confirm_required_c)
    else:
        confirm_metric_gain_c = eval_base_metric_c - chosen_cand_metric_c
    confirm_mse_gain_c = eval_base_mse_c - chosen_cand_mse_c
    confirm_mae_guard_enabled = confirm_min_abs_mae_improvement is not None
    confirm_mae_required_c = torch.full_like(
        eval_base_mae_c,
        max(0.0, float(confirm_min_abs_mae_improvement)) if confirm_mae_guard_enabled else 0.0,
    )
    confirm_mae_gain_c = eval_base_mae_c - chosen_cand_mae_c
    if confirm_mae_guard_enabled:
        use_candidate_c = use_candidate_c & (confirm_mae_gain_c >= confirm_mae_required_c)

    segment_ranges = _contiguous_segment_ranges(int(eval_indices.numel()), int(segment_count))
    segment_min_positive_count = (
        len(segment_ranges)
        if segment_min_positive is None
        else max(0, min(int(segment_min_positive), len(segment_ranges)))
    )
    segment_guard_enabled = bool(len(segment_ranges) > 1 and segment_min_positive_count > 0)
    segment_positive_count_c = torch.zeros_like(eval_base_mse_c, dtype=torch.long)
    segment_metric_gain_sc = torch.empty((0, int(C)), dtype=eval_base_mse_c.dtype, device=eval_base_mse_c.device)
    segment_mae_gain_sc = torch.empty((0, int(C)), dtype=eval_base_mse_c.dtype, device=eval_base_mse_c.device)
    segment_metric_required = (
        max(0.0, float(segment_min_abs_improvement))
        if segment_min_abs_improvement is not None
        else 0.0
    )
    segment_mae_guard_enabled = segment_min_abs_mae_improvement is not None
    segment_mae_required = (
        max(0.0, float(segment_min_abs_mae_improvement))
        if segment_mae_guard_enabled
        else 0.0
    )
    if segment_guard_enabled:
        segment_metric_gains: List[torch.Tensor] = []
        segment_mae_gains: List[torch.Tensor] = []
        segment_positive: List[torch.Tensor] = []
        for start, end in segment_ranges:
            segment_indices = eval_indices[start:end]
            seg_base_mse_c, seg_base_mae_c, seg_cand_mse_cp, seg_cand_mae_cp = _mse_mae_for_indices(segment_indices)
            seg_chosen_mse_c = seg_cand_mse_cp.gather(1, safe_p_c.view(-1, 1)).squeeze(1)
            seg_chosen_mae_c = seg_cand_mae_cp.gather(1, safe_p_c.view(-1, 1)).squeeze(1)
            seg_base_metric_c = seg_base_mae_c if metric_mode == "mae" else seg_base_mse_c
            seg_chosen_metric_c = seg_chosen_mae_c if metric_mode == "mae" else seg_chosen_mse_c
            seg_metric_gain_c = seg_base_metric_c - seg_chosen_metric_c
            seg_mae_gain_c = seg_base_mae_c - seg_chosen_mae_c
            keep_segment_c = seg_metric_gain_c > segment_metric_required
            if segment_mae_guard_enabled:
                keep_segment_c = keep_segment_c & (seg_mae_gain_c >= segment_mae_required)
            segment_metric_gains.append(seg_metric_gain_c)
            segment_mae_gains.append(seg_mae_gain_c)
            segment_positive.append(keep_segment_c)
        segment_metric_gain_sc = torch.stack(segment_metric_gains, dim=0)
        segment_mae_gain_sc = torch.stack(segment_mae_gains, dim=0)
        segment_positive_count_c = torch.stack(segment_positive, dim=0).to(dtype=torch.long).sum(dim=0)
        use_candidate_c = use_candidate_c & (segment_positive_count_c >= segment_min_positive_count)

    selected_class_c = torch.where(
        use_candidate_c,
        best_p_c.to(dtype=torch.long) + 1,
        torch.zeros_like(best_p_c, dtype=torch.long),
    )
    selector = StaticPredResidualCandidateSelector(selected_class_c)
    selected_mse_c = torch.where(selected_class_c > 0, chosen_cand_mse_c, eval_base_mse_c)
    selected_mae_c = torch.where(selected_class_c > 0, chosen_cand_mae_c, eval_base_mae_c)

    penalties = list(penalty_names or [f"p{i}" for i in range(P)])
    if len(penalties) != P:
        penalties = [f"p{i}" for i in range(P)]
    channels = list(channel_names or [f"ch_{i}" for i in range(C)])
    if len(channels) != C:
        channels = [f"ch_{i}" for i in range(C)]
    selected_classes = [int(v) for v in selected_class_c.detach().cpu().tolist()]
    selected_names = [penalties[cls - 1] if cls > 0 else "skip" for cls in selected_classes]
    base_avg_mse = float(eval_base_mse_c.mean().item())
    base_avg_mae = float(eval_base_mae_c.mean().item())
    selected_avg_mse = float(selected_mse_c.mean().item())
    selected_avg_mae = float(selected_mae_c.mean().item())
    summary = {
        "mode": "static_candidate_channel",
        "selection_windows": int(select_indices.numel()),
        "eval_windows": int(eval_indices.numel()),
        "min_abs_improvement": float(max(0.0, float(min_abs_improvement))),
        "min_rel_improvement": float(max(0.0, float(min_rel_improvement))),
        "selection_metric": metric_mode,
        "mae_guard_enabled": bool(mae_guard_enabled),
        "min_abs_mae_improvement": float(mae_required_c.max().item()) if mae_guard_enabled else None,
        "confirm_guard_enabled": bool(confirm_guard_enabled),
        "confirm_min_abs_improvement": float(confirm_required_c.max().item()) if confirm_guard_enabled else None,
        "confirm_min_rel_improvement": float(max(0.0, float(confirm_min_rel_improvement))) if confirm_guard_enabled else 0.0,
        "confirm_mae_guard_enabled": bool(confirm_mae_guard_enabled),
        "confirm_min_abs_mae_improvement": float(confirm_mae_required_c.max().item()) if confirm_mae_guard_enabled else None,
        "segment_guard_enabled": bool(segment_guard_enabled),
        "segment_count": int(len(segment_ranges)) if segment_guard_enabled else 0,
        "segment_min_positive": int(segment_min_positive_count) if segment_guard_enabled else 0,
        "segment_min_abs_improvement": float(segment_metric_required) if segment_guard_enabled else None,
        "segment_mae_guard_enabled": bool(segment_guard_enabled and segment_mae_guard_enabled),
        "segment_min_abs_mae_improvement": float(segment_mae_required) if segment_guard_enabled and segment_mae_guard_enabled else None,
        "segment_positive_count_per_channel": [int(v) for v in segment_positive_count_c.detach().cpu().tolist()],
        "segment_metric_gain_per_channel": [
            [float(v) for v in segment_metric_gain_sc[:, i].detach().cpu().tolist()]
            for i in range(int(C))
        ] if segment_guard_enabled else [],
        "segment_mae_gain_per_channel": [
            [float(v) for v in segment_mae_gain_sc[:, i].detach().cpu().tolist()]
            for i in range(int(C))
        ] if segment_guard_enabled else [],
        "selected_class": selected_classes,
        "selected_penalty_by_channel": selected_names,
        "selected_channels": [channels[i] for i, cls in enumerate(selected_classes) if cls > 0],
        "num_candidate_channels": int((selected_class_c > 0).sum().item()),
        "select_base_mse_per_channel": [float(v) for v in select_base_mse_c.detach().cpu().tolist()],
        "select_best_candidate_mse_per_channel": [float(v) for v in best_cand_mse_c.detach().cpu().tolist()],
        "select_best_candidate_mae_per_channel": [float(v) for v in best_cand_mae_c.detach().cpu().tolist()],
        "select_best_candidate_metric_per_channel": [float(v) for v in best_cand_metric_c.detach().cpu().tolist()],
        "confirm_metric_gain_per_channel": [float(v) for v in confirm_metric_gain_c.detach().cpu().tolist()],
        "confirm_mse_gain_per_channel": [float(v) for v in confirm_mse_gain_c.detach().cpu().tolist()],
        "confirm_mae_gain_per_channel": [float(v) for v in confirm_mae_gain_c.detach().cpu().tolist()],
        "eval_base_avg_mse": base_avg_mse,
        "eval_base_avg_mae": base_avg_mae,
        "eval_selected_avg_mse": selected_avg_mse,
        "eval_selected_avg_mae": selected_avg_mae,
        "eval_gain_pct_vs_base": float(100.0 * (base_avg_mse - selected_avg_mse) / max(abs(base_avg_mse), 1.0e-12)),
        "eval_mae_gain_pct_vs_base": float(100.0 * (base_avg_mae - selected_avg_mae) / max(abs(base_avg_mae), 1.0e-12)),
        "eval_base_mse_per_channel": [float(v) for v in eval_base_mse_c.detach().cpu().tolist()],
        "eval_selected_mse_per_channel": [float(v) for v in selected_mse_c.detach().cpu().tolist()],
        "eval_base_mae_per_channel": [float(v) for v in eval_base_mae_c.detach().cpu().tolist()],
        "eval_selected_mae_per_channel": [float(v) for v in selected_mae_c.detach().cpu().tolist()],
    }
    return selector, summary


def train_pred_residual_candidate_selector(
    model: nn.Module,
    pred_residual: Optional[ClusterwisePredResidualMoE],
    loader: DataLoader,
    cluster_id_c: torch.Tensor,
    K: int,
    moe_cfg: dict,
    device: torch.device,
    penalty_names: List[str],
    channel_names: List[str],
    cfg: Dict[str, object],
    pred_residual_scale_c: Optional[torch.Tensor] = None,
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
    precollected_tensors: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[Optional[PredResidualCandidateSelector], Dict[str, object]]:
    candidate_feature_mode = str(cfg.get("feature_mode", "base")).lower()
    candidate_feature_names = _candidate_selector_feature_names(candidate_feature_mode)
    tensors = precollected_tensors
    if tensors is None:
        tensors = _collect_pred_residual_selector_tensors(
            model=model,
            pred_residual=pred_residual,
            loader=loader,
            cluster_id_c=cluster_id_c,
            K=K,
            moe_cfg=moe_cfg,
            device=device,
            penalty_count=len(penalty_names),
            pred_residual_scale_c=pred_residual_scale_c,
            history_anchor_cfg=history_anchor_cfg,
            observed_history_tc=observed_history_tc,
            input_len=input_len,
            eval_start=eval_start,
            model_train_stat_adapter_pc=model_train_stat_adapter_pc,
            model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
            train_stat_anchor_pc=train_stat_anchor_pc,
            train_residual_anchor_phc=train_residual_anchor_phc,
            learnable_output_anchor=learnable_output_anchor,
            candidate_feature_mode=candidate_feature_mode,
        )
    if tensors is None:
        return None, {"enable": False, "reason": "empty_loader_or_residual_disabled"}

    selector_patch_len = max(0, int(cfg.get("patch_len", 0)))
    if selector_patch_len > 0:
        tensors = _patchify_pred_residual_selector_tensors(
            tensors,
            patch_len=selector_patch_len,
            feature_mode=candidate_feature_mode,
        )

    skip_feat = tensors["skip_feat"]
    cand_feat = tensors["cand_feat"]
    base = tensors["base"]
    cand = tensors["cand"]
    y = tensors["y"]
    query_start_abs = tensors.get("query_start_abs")
    n = int(base.shape[0])
    c = int(base.shape[1])
    p = int(cand.shape[2])
    allowed_mask_cp = None
    if allowed_mask_kp is not None and int(allowed_mask_kp.numel()) > 0:
        allowed_kp = allowed_mask_kp.detach().to(dtype=torch.bool)
        if tuple(allowed_kp.shape) != (int(K), p):
            raise ValueError(
                "candidate_selector allowed_mask_kp must have shape [K,P], "
                f"got {tuple(allowed_kp.shape)} vs {(int(K), p)}."
            )
        cluster_idx = cluster_id_c.detach().cpu().to(dtype=torch.long)
        allowed_mask_cp = allowed_kp.detach().cpu().index_select(0, cluster_idx)
    train_fraction = float(cfg.get("train_fraction", 0.75))
    split = int(max(1, min(n - 1, round(n * train_fraction)))) if n > 1 else n
    train_idx = torch.arange(0, split, dtype=torch.long)
    hold_idx = torch.arange(split, n, dtype=torch.long)
    if hold_idx.numel() == 0:
        hold_idx = train_idx

    selector = PredResidualCandidateSelector(
        feat_dim=int(skip_feat.shape[-1]),
        num_channels=c,
        num_penalties=p,
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        dropout=float(cfg.get("dropout", 0.0)),
        init_skip_bias=float(cfg.get("init_skip_bias", 0.0)),
        init_penalty_bias=float(cfg.get("init_penalty_bias", 0.0)),
        use_penalty_identity=bool(cfg.get("use_penalty_identity", False)),
        use_channel_identity=bool(cfg.get("use_channel_identity", False)),
        use_time_features=bool(cfg.get("use_time_features", False)),
        time_feature_periods=[int(v) for v in cfg.get("time_feature_periods", [24, 168])],
        time_feature_offset=int(cfg.get("time_feature_offset", int(input_len))),
        feature_mode=candidate_feature_mode,
        patch_len=selector_patch_len,
    ).to(device)
    selector.set_allowed_penalty_mask(allowed_mask_cp)
    standardize_features = bool(cfg.get("standardize_features", True))
    feature_std_summary: Dict[str, object] = {
        "standardize_features": bool(standardize_features),
        "fit_windows": int(train_idx.numel()) if standardize_features else 0,
        "mode": None,
        "min_std": None,
        "max_std": None,
        "clip": 0.0,
    }
    if standardize_features:
        feat_mean, feat_std, std_stats = _candidate_selector_feature_standardization_stats(
            skip_feat=skip_feat,
            cand_feat=cand_feat,
            selector=selector,
            train_idx=train_idx,
            query_start_abs=query_start_abs,
            mode=str(cfg.get("standardization_mode", "mean_std")),
        )
        selector.set_feature_standardization(feat_mean.to(device), feat_std.to(device))
        feature_clip = float(cfg.get("standardize_clip", cfg.get("feature_clip", 0.0)))
        selector.set_feature_standardize_clip(feature_clip)
        feature_std_summary.update(std_stats)
        feature_std_summary["clip"] = float(max(0.0, feature_clip))

    min_abs_improvement = float(cfg.get("label_min_abs_improvement", cfg.get("min_abs_improvement", 0.0)))
    min_rel_improvement = float(cfg.get("label_min_rel_improvement", cfg.get("min_rel_improvement", 0.0)))
    lr = float(cfg.get("lr", 1.0e-3))
    weight_decay = float(cfg.get("weight_decay", 1.0e-4))
    batch_size = max(1, int(cfg.get("batch_size", 256)))
    epochs = max(1, int(cfg.get("epochs", 40)))
    grad_clip = float(cfg.get("grad_clip", 1.0))
    selector_loss_kind = str(cfg.get("loss", cfg.get("training_loss", "ce"))).lower()
    expected_error_temperature = float(cfg.get("expected_error_temperature", cfg.get("softmax_temperature", 1.0)))
    expected_error_metric = str(cfg.get("expected_error_metric", "mse")).lower()
    rate_alignment_weight = float(cfg.get("rate_alignment_weight", 0.0))
    rate_alignment_temperature = float(cfg.get("rate_alignment_temperature", expected_error_temperature))
    positive_sample_weight = float(cfg.get("positive_sample_weight", 1.0))
    negative_sample_weight = float(cfg.get("negative_sample_weight", 1.0))
    class_weight_cfg = cfg.get("class_weight", "auto")
    class_weight = None
    class_weight_summary: object = None
    train_target = _candidate_selector_targets(
        base_bch=base.index_select(0, train_idx),
        cand_bcpH=cand.index_select(0, train_idx),
        y_bch=y.index_select(0, train_idx),
        min_abs_improvement=min_abs_improvement,
        min_rel_improvement=min_rel_improvement,
        allowed_mask_cp=allowed_mask_cp,
    ).reshape(-1)
    if isinstance(class_weight_cfg, str) and class_weight_cfg.lower() == "auto":
        counts = torch.bincount(train_target, minlength=p + 1).to(dtype=torch.float32)
        total = counts.sum().clamp_min(1.0)
        weight = total / (float(p + 1) * counts.clamp_min(1.0))
        weight = weight.clamp(
            min=float(cfg.get("class_weight_min", 0.25)),
            max=float(cfg.get("class_weight_max", 8.0)),
        )
        class_weight = weight.to(device)
        class_weight_summary = [float(v) for v in weight.tolist()]
    elif isinstance(class_weight_cfg, (list, tuple)):
        weight = torch.as_tensor(class_weight_cfg, dtype=torch.float32)
        if int(weight.numel()) != p + 1:
            raise ValueError(f"candidate_selector.class_weight must have {p + 1} entries.")
        class_weight = weight.to(device)
        class_weight_summary = [float(v) for v in weight.tolist()]
    elif isinstance(class_weight_cfg, str) and class_weight_cfg.lower() in {"none", "false", "off", "0"}:
        class_weight = None
        class_weight_summary = None
    elif bool(class_weight_cfg):
        scalar = float(class_weight_cfg)
        class_weight = torch.ones(p + 1, dtype=torch.float32, device=device)
        class_weight[1:] = scalar
        class_weight_summary = [float(v) for v in class_weight.detach().cpu().tolist()]
    target_rate_q = torch.bincount(train_target, minlength=p + 1).to(dtype=torch.float32)
    target_rate_q = target_rate_q / target_rate_q.sum().clamp_min(1.0)

    opt = torch.optim.AdamW(selector.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_hold = float("inf")
    best_epoch = 0

    def _loss_for(batch_idx: torch.Tensor) -> torch.Tensor:
        skip_feat_b = skip_feat.index_select(0, batch_idx).to(device)
        cand_feat_b = cand_feat.index_select(0, batch_idx).to(device)
        base_b = base.index_select(0, batch_idx).to(device)
        cand_b = cand.index_select(0, batch_idx).to(device)
        y_b = y.index_select(0, batch_idx).to(device)
        query_b = None
        if query_start_abs is not None:
            query_b = query_start_abs.index_select(0, batch_idx).to(device)
        target_b = _candidate_selector_targets(
            base_bch=base_b,
            cand_bcpH=cand_b,
            y_bch=y_b,
            min_abs_improvement=min_abs_improvement,
            min_rel_improvement=min_rel_improvement,
            allowed_mask_cp=allowed_mask_cp,
        )
        logits_bcq = selector.logits_from_features(
            skip_feat_b,
            cand_feat_b,
            allowed_mask_cp=allowed_mask_cp,
            query_start_abs_b=query_b,
        )
        if selector_loss_kind in {"ce", "cross_entropy", "classification"}:
            loss_bc = torch.nn.functional.cross_entropy(
                logits_bcq.reshape(-1, p + 1),
                target_b.reshape(-1),
                weight=class_weight,
                reduction="none",
            ).view_as(target_b).to(dtype=logits_bcq.dtype)
            if positive_sample_weight != 1.0 or negative_sample_weight != 1.0:
                sample_weight = torch.where(
                    target_b > 0,
                    torch.full_like(loss_bc, positive_sample_weight),
                    torch.full_like(loss_bc, negative_sample_weight),
                )
                loss_bc = loss_bc * sample_weight
            loss = loss_bc.mean()
        if selector_loss_kind in {"expected_mse", "expected_error", "utility", "soft_utility"}:
            loss = _candidate_selector_expected_error_loss(
                logits_bcq=logits_bcq,
                base_bch=base_b,
                cand_bcpH=cand_b,
                y_bch=y_b,
                temperature=expected_error_temperature,
                loss_kind=expected_error_metric,
            )
        elif selector_loss_kind not in {"ce", "cross_entropy", "classification"}:
            raise ValueError(
                "moe.pred_side_residual.candidate_selector.loss must be ce or expected_mse "
                f"(got {selector_loss_kind!r})."
            )
        if rate_alignment_weight > 0.0:
            loss = loss + rate_alignment_weight * _candidate_selector_rate_alignment_loss(
                logits_bcq=logits_bcq,
                target_rate_q=target_rate_q.to(device),
                temperature=rate_alignment_temperature,
            )
        return loss

    for ep in range(1, epochs + 1):
        selector.train()
        perm = train_idx[torch.randperm(train_idx.numel())]
        for b0 in range(0, int(perm.numel()), batch_size):
            batch_idx = perm[b0:b0 + batch_size]
            loss = _loss_for(batch_idx)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(selector.parameters(), grad_clip)
            opt.step()
        hold_metrics = _pred_residual_selector_metrics_from_tensors(
            tensors=tensors,
            selector=selector,
            device=device,
            batch_size=batch_size,
            min_abs_improvement=min_abs_improvement,
            min_rel_improvement=min_rel_improvement,
            indices=hold_idx,
            penalty_names=penalty_names,
            allowed_mask_cp=allowed_mask_cp,
        )
        hold_mse = float((hold_metrics or {}).get("selected_mse", float("inf")))
        if hold_mse < best_hold:
            best_hold = hold_mse
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in selector.state_dict().items()}

    if best_state is not None:
        selector.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    margin_raw = cfg.get("decision_margin", "auto")
    margin_selection: Dict[str, object] = {"mode": "fixed", "margin": 0.0}
    margin_mode = str(margin_raw).lower() if isinstance(margin_raw, str) else "fixed"
    if margin_mode in {"auto", "auto_temporal", "temporal", "temporal_robust"}:
        max_margin = max(0.0, float(cfg.get("decision_margin_max", 6.0)))
        num_margin = max(2, int(cfg.get("decision_margin_candidates", 61)))
        margins = torch.linspace(0.0, max_margin, steps=num_margin).tolist()
        temporal_margin = margin_mode in {"auto_temporal", "temporal", "temporal_robust"}
        temporal_blocks = max(1, int(cfg.get("decision_margin_temporal_blocks", 6)))
        temporal_min_gain = float(cfg.get("decision_margin_min_block_gain_pct", 0.0))
        temporal_required = max(
            0,
            int(cfg.get("decision_margin_min_positive_blocks", temporal_blocks)),
        )
        margin_rows: List[Dict[str, object]] = []
        for margin in margins:
            selector.decision_margin = float(margin)
            metrics = _pred_residual_selector_metrics_from_tensors(
                tensors=tensors,
                selector=selector,
                device=device,
                batch_size=batch_size,
                min_abs_improvement=min_abs_improvement,
                min_rel_improvement=min_rel_improvement,
                indices=hold_idx,
                penalty_names=penalty_names,
                allowed_mask_cp=allowed_mask_cp,
            )
            if metrics is None:
                continue
            mse = float(metrics.get("selected_mse", float("inf")))
            gain = float(metrics.get("selected_gain_pct_vs_base", float("-inf")))
            row: Dict[str, object] = {
                "margin": float(margin),
                "selected_mse": mse,
                "selected_gain_pct_vs_base": gain,
                "selected_skip_rate": float((metrics.get("selected_class_rate") or {}).get("skip", 0.0)),
            }
            if temporal_margin:
                block_metrics = _pred_residual_selector_temporal_block_metrics(
                    tensors=tensors,
                    selector=selector,
                    device=device,
                    batch_size=batch_size,
                    num_blocks=temporal_blocks,
                    min_abs_improvement=min_abs_improvement,
                    min_rel_improvement=min_rel_improvement,
                    indices=hold_idx,
                    penalty_names=penalty_names,
                    allowed_mask_cp=allowed_mask_cp,
                )
                block_gains = [
                    float(block.get("selected_gain_pct_vs_base", float("-inf")))
                    for block in block_metrics
                ]
                row.update(
                    {
                        "block_gain_pct": block_gains,
                        "min_block_gain_pct": min(block_gains) if block_gains else float("-inf"),
                        "positive_block_count": sum(gain_v >= temporal_min_gain for gain_v in block_gains),
                        "block_count": len(block_gains),
                    }
                )
            margin_rows.append(row)
        if temporal_margin:
            best_row = _candidate_selector_choose_temporal_margin_row(
                margin_rows,
                required_positive_blocks=temporal_required,
            )
        else:
            best_row = min(margin_rows, key=lambda row: float(row["selected_mse"])) if margin_rows else None
        best_margin = float((best_row or {}).get("margin", 0.0))
        best_margin_mse = float((best_row or {}).get("selected_mse", float("inf")))
        best_margin_gain = float((best_row or {}).get("selected_gain_pct_vs_base", float("-inf")))
        selector.decision_margin = float(best_margin)
        margin_selection = {
            "mode": "auto_temporal" if temporal_margin else "auto",
            "margin": float(best_margin),
            "holdout_selected_mse": float(best_margin_mse),
            "holdout_gain_pct_vs_base": float(best_margin_gain),
            "max_margin": float(max_margin),
            "candidates": int(num_margin),
        }
        if temporal_margin:
            margin_selection.update(
                {
                    "temporal_blocks": int(temporal_blocks),
                    "min_block_gain_pct": float(temporal_min_gain),
                    "required_positive_blocks": int(temporal_required),
                    "selected_row": best_row,
                    "sweep": margin_rows,
                }
            )
    else:
        selector.decision_margin = float(margin_raw)
        margin_selection = {"mode": "fixed", "margin": float(selector.decision_margin)}
    train_metrics = _pred_residual_selector_metrics_from_tensors(
        tensors=tensors,
        selector=selector,
        device=device,
        batch_size=batch_size,
        min_abs_improvement=min_abs_improvement,
        min_rel_improvement=min_rel_improvement,
        indices=train_idx,
        penalty_names=penalty_names,
        allowed_mask_cp=allowed_mask_cp,
    )
    hold_metrics = _pred_residual_selector_metrics_from_tensors(
        tensors=tensors,
        selector=selector,
        device=device,
        batch_size=batch_size,
        min_abs_improvement=min_abs_improvement,
        min_rel_improvement=min_rel_improvement,
        indices=hold_idx,
        penalty_names=penalty_names,
        allowed_mask_cp=allowed_mask_cp,
    )
    feature_gain_diagnostics = {
        "train": _candidate_selector_feature_gain_diagnostics(
            tensors=tensors,
            feature_names=candidate_feature_names,
            penalty_names=penalty_names,
            allowed_mask_cp=allowed_mask_cp,
            indices=train_idx,
            cluster_id_c=cluster_id_c,
            K=K,
            min_abs_improvement=min_abs_improvement,
            min_rel_improvement=min_rel_improvement,
            topk=int(cfg.get("feature_gain_diagnostic_topk", 5)),
        ),
        "holdout": _candidate_selector_feature_gain_diagnostics(
            tensors=tensors,
            feature_names=candidate_feature_names,
            penalty_names=penalty_names,
            allowed_mask_cp=allowed_mask_cp,
            indices=hold_idx,
            cluster_id_c=cluster_id_c,
            K=K,
            min_abs_improvement=min_abs_improvement,
            min_rel_improvement=min_rel_improvement,
            topk=int(cfg.get("feature_gain_diagnostic_topk", 5)),
        ),
    }
    temporal_block_metrics: Optional[Dict[str, object]] = None
    temporal_block_count = int(
        cfg.get(
            "temporal_block_audit_blocks",
            cfg.get("temporal_audit_blocks", 0),
        )
        or 0
    )
    if temporal_block_count > 1:
        temporal_block_metrics = {
            "num_blocks": int(temporal_block_count),
            "full": _pred_residual_selector_temporal_block_metrics(
                tensors=tensors,
                selector=selector,
                device=device,
                batch_size=batch_size,
                num_blocks=temporal_block_count,
                min_abs_improvement=min_abs_improvement,
                min_rel_improvement=min_rel_improvement,
                penalty_names=penalty_names,
                allowed_mask_cp=allowed_mask_cp,
            ),
            "train": _pred_residual_selector_temporal_block_metrics(
                tensors=tensors,
                selector=selector,
                device=device,
                batch_size=batch_size,
                num_blocks=temporal_block_count,
                min_abs_improvement=min_abs_improvement,
                min_rel_improvement=min_rel_improvement,
                indices=train_idx,
                penalty_names=penalty_names,
                allowed_mask_cp=allowed_mask_cp,
            ),
            "holdout": _pred_residual_selector_temporal_block_metrics(
                tensors=tensors,
                selector=selector,
                device=device,
                batch_size=batch_size,
                num_blocks=temporal_block_count,
                min_abs_improvement=min_abs_improvement,
                min_rel_improvement=min_rel_improvement,
                indices=hold_idx,
                penalty_names=penalty_names,
                allowed_mask_cp=allowed_mask_cp,
            ),
        }
    summary = {
        "enable": True,
        "train_windows": int(train_idx.numel()),
        "holdout_windows": int(hold_idx.numel()),
        "best_epoch": int(best_epoch),
        "label_min_abs_improvement": float(min_abs_improvement),
        "label_min_rel_improvement": float(min_rel_improvement),
        "loss": selector_loss_kind,
        "expected_error_temperature": float(expected_error_temperature),
        "expected_error_metric": expected_error_metric,
        "rate_alignment_weight": float(rate_alignment_weight),
        "rate_alignment_temperature": float(rate_alignment_temperature),
        "target_class_rate_train": [float(v) for v in target_rate_q.detach().cpu().tolist()],
        "positive_sample_weight": float(positive_sample_weight),
        "negative_sample_weight": float(negative_sample_weight),
        "class_weight": class_weight_summary,
        "use_penalty_identity": bool(selector.use_penalty_identity),
        "use_channel_identity": bool(selector.use_channel_identity),
        "use_time_features": bool(selector.use_time_features),
        "time_feature_periods": list(selector.time_feature_periods),
        "time_feature_offset": int(selector.time_feature_offset),
        "effective_feature_dim": int(selector.F),
        "feature_mode": candidate_feature_mode,
        "routing_granularity": "channel_patch" if selector_patch_len > 0 else "channel",
        "patch_len": int(selector_patch_len),
        "num_patches": int(tensors["patch_count"].item()) if selector_patch_len > 0 else 1,
        "decision_margin": float(selector.decision_margin),
        "decision_margin_selection": margin_selection,
        "feature_standardization": feature_std_summary,
        "channel_names": list(channel_names),
        "penalty_names": list(penalty_names),
        "allowed_mask_cp": allowed_mask_cp.to(dtype=torch.long).tolist() if allowed_mask_cp is not None else None,
        "gate_feature_mode": _normalize_gate_feature_mode(gate_feature_mode),
        "feature_gain_diagnostics": feature_gain_diagnostics,
        "temporal_block_metrics": temporal_block_metrics,
        "train": train_metrics,
        "holdout": hold_metrics,
    }
    selector.eval()
    return selector, summary


__all__ = [
    'CANDIDATE_DELTA_FEATURE_NAMES',
    'HISTORY_PROXY_SELECTOR_FEATURE_NAMES',
    'SHAPE_PROXY_SELECTOR_FEATURE_NAMES',
    '_candidate_selector_feature_names',
    '_history_proxy_for_candidate_selector',
    '_sequence_slope_bch',
    '_sequence_diff_rms_bch',
    '_sequence_d2_rms_bch',
    '_sequence_corr_bch',
    '_candidate_selector_features',
    '_candidate_selector_patch_views',
    '_candidate_selector_patch_query_starts',
    '_candidate_delta_features',
    '_pred_residual_candidate_predictions',
    '_pred_residual_candidates_on_eval_path',
    '_candidate_selector_targets',
    '_candidate_selector_expected_error_loss',
    '_candidate_selector_rate_alignment_loss',
    '_selector_allowed_mask_cp',
    '_candidate_selector_adoption_decision',
    '_candidate_selector_temporal_block_adoption_guard',
    '_candidate_selector_choose_temporal_margin_row',
    '_mix_selected_channel_metrics',
    '_candidate_selector_candidate_scale',
    '_cluster_utility_threshold_stats',
    '_mse_utility_gate_supervision_loss',
    '_patch_router_expected_mse_loss_bk',
    '_patch_router_mixture_mse_loss_bk',
    '_patch_router_oracle_ce_loss_bk',
    '_select_recall_constrained_risk_threshold',
    '_risk_score_threshold_curve_summary',
    '_loss_gradient_overlap_summary',
    '_temporal_group_dro_incremental_loss',
    '_select_recall_constrained_risk_threshold_by_penalty',
    '_causal_patch_regime_descriptor',
    '_causal_patch_scale_features',
    '_walk_forward_patch_reliability_metrics',
    '_causal_expert_feedback_ridge_metrics',
    '_walk_forward_expert_reliability_rerank_metrics',
    '_patch_router_hierarchical_recall_loss_terms',
    '_patch_router_oracle_batch_stats',
    '_patchwise_penalty_bcq',
    '_level_oracle_patch_diagnostics',
    '_prediction_fit_sufficient_statistics',
    '_level_stage1_acceptance_candidate_patch',
    '_validate_level_executed_candidate_patch',
    '_semantic_bank_acceptance_metrics',
    '_semantic_bank_semantic_only_acceptance_metrics',
    '_pred_residual_candidate_supervision_loss',
    '_pred_residual_intervention_supervision_loss',
    'PredResidualCandidateSelector',
    'StaticPredResidualCandidateSelector',
    '_collect_pred_residual_selector_tensors',
    '_concat_pred_residual_selector_tensors',
    '_patchify_pred_residual_selector_tensors',
    '_select_pred_residual_confidence_thresholds_from_tensors',
    '_candidate_selector_feature_gain_diagnostics',
    '_pred_residual_selector_metrics_from_tensors',
    '_pred_residual_selector_temporal_block_metrics',
    '_candidate_selector_feature_standardization_stats',
    '_candidate_selector_select_confirm_indices',
    '_fit_static_candidate_channel_selector_from_tensors',
    'train_pred_residual_candidate_selector',
]
