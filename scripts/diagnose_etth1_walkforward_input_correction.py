from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.next11c_route_accuracy_diagnostic import _build_anchor_artifacts
from scripts.next11d_authorized_test_shift_probe import _make_loaders_with_authorized_test
from scripts.next11d_fixed_candidate_router_refit import _read_data_for_cfg
from scripts.shape_prior_diagnostic import _build_modules
from src.models.learnable_anchor import ClusterwiseLearnableOutputAnchor
from src.train import (
    _normalize_learnable_output_anchor_cfg,
    apply_history_anchor_adapter,
    apply_moe_output_anchor_experts,
    apply_train_stat_anchor_expert,
    apply_train_stat_input_centering,
)
from src.utils.seed import set_seed
from src.utils.yaml_io import load_yaml


def _sequence_slope(values: torch.Tensor) -> torch.Tensor:
    if int(values.shape[-1]) <= 1:
        return torch.zeros_like(values[..., 0])
    time = torch.linspace(-1.0, 1.0, int(values.shape[-1]), dtype=values.dtype, device=values.device)
    centered = values - values.mean(dim=-1, keepdim=True)
    return (centered * time).mean(dim=-1) / time.square().mean().clamp_min(1.0e-8)


def build_input_correction_features(
    x_ncl: torch.Tensor,
    base_nch: torch.Tensor,
) -> Tuple[torch.Tensor, List[str]]:
    if x_ncl.ndim != 3 or base_nch.ndim != 3:
        raise ValueError("walk-forward correction expects x/base with shape [N,C,L/H].")
    if tuple(x_ncl.shape[:2]) != tuple(base_nch.shape[:2]):
        raise ValueError("walk-forward correction x/base batch and channel dimensions must match.")
    eps = 1.0e-6
    L = int(x_ncl.shape[-1])
    H = int(base_nch.shape[-1])
    parts: List[torch.Tensor] = []
    names: List[str] = []

    def add(name: str, value: torch.Tensor) -> None:
        parts.append(value)
        names.append(name)

    add("hist_last", x_ncl[..., -1])
    for window in (24, 48, 96):
        width = min(window, L)
        tail = x_ncl[..., -width:]
        add(f"hist_mean_{window}", tail.mean(dim=-1))
        add(f"hist_std_{window}", tail.std(dim=-1, unbiased=False))
        add(f"hist_slope_{window}", _sequence_slope(tail))
        if width >= 2:
            add(f"hist_diff_rms_{window}", tail.diff(dim=-1).square().mean(dim=-1).clamp_min(eps).sqrt())
        if width >= 3:
            add(f"hist_d2_rms_{window}", tail.diff(dim=-1).diff(dim=-1).square().mean(dim=-1).clamp_min(eps).sqrt())

    add("base_mean", base_nch.mean(dim=-1))
    add("base_last", base_nch[..., -1])
    add("base_std", base_nch.std(dim=-1, unbiased=False))
    add("base_slope", _sequence_slope(base_nch))

    patch_count = 4
    if L % patch_count == 0:
        hist_patch = x_ncl.reshape(*x_ncl.shape[:2], patch_count, L // patch_count)
        for patch_idx in range(patch_count):
            add(f"hist_patch_mean_{patch_idx}", hist_patch[..., patch_idx, :].mean(dim=-1))
    if H % patch_count == 0:
        base_patch = base_nch.reshape(*base_nch.shape[:2], patch_count, H // patch_count)
        for patch_idx in range(patch_count):
            add(f"base_patch_mean_{patch_idx}", base_patch[..., patch_idx, :].mean(dim=-1))
    return torch.stack(parts, dim=-1), names


def prepare_input_correction_features(
    x_ncl: torch.Tensor,
    base_nch: torch.Tensor,
    *,
    local_normalize: bool,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    if not bool(local_normalize):
        features, names = build_input_correction_features(x_ncl, base_nch)
        scale = torch.ones(*x_ncl.shape[:2], 1, dtype=x_ncl.dtype, device=x_ncl.device)
        return features, scale, names
    mean = x_ncl.mean(dim=-1, keepdim=True)
    scale = x_ncl.std(dim=-1, unbiased=False, keepdim=True).clamp_min(1.0e-5)
    x_local = (x_ncl - mean) / scale
    base_local = (base_nch - mean) / scale
    features, names = build_input_correction_features(x_local, base_local)
    return features, scale, names


def restore_local_correction_scale(
    correction_nch: torch.Tensor,
    *,
    scale_nc1: torch.Tensor,
    x_ncl: torch.Tensor,
    max_abs_scale: float,
) -> torch.Tensor:
    correction = correction_nch * scale_nc1.to(device=correction_nch.device, dtype=correction_nch.dtype)
    if float(max_abs_scale) > 0.0:
        bound = float(max_abs_scale) * x_ncl.std(dim=-1, unbiased=False).clamp_min(1.0e-5).unsqueeze(-1)
        correction = torch.maximum(torch.minimum(correction, bound), -bound)
    return correction


def online_refit_corrections(
    *,
    features_ncf: torch.Tensor,
    x_ncl: torch.Tensor,
    base_nch: torch.Tensor,
    target_nch: torch.Tensor,
    eval_start: int,
    eval_end: int,
    update_interval: int,
    label_delay: int,
    ridge: float,
    half_life: float,
    shrink: float,
    max_abs_scale: float,
    fit_loss: str,
    huber_delta: float,
    huber_iterations: int,
    max_fit_windows: int = 0,
) -> Tuple[torch.Tensor, List[Dict[str, int]]]:
    start = int(eval_start)
    end = int(eval_end)
    interval = max(1, int(update_interval))
    delay = max(1, int(label_delay))
    if not (0 <= start < end <= int(features_ncf.shape[0])):
        raise ValueError("online refit eval range is invalid.")
    correction_parts: List[torch.Tensor] = []
    updates: List[Dict[str, int]] = []
    for position in range(start, end, interval):
        chunk_end = min(end, position + interval)
        fit_end = max(1, position - delay + 1)
        fit_start = max(0, fit_end - int(max_fit_windows)) if int(max_fit_windows) > 0 else 0
        state = fit_weighted_ridge_residual(
            features_ncf[fit_start:fit_end],
            (target_nch - base_nch)[fit_start:fit_end],
            ridge=ridge,
            half_life=half_life,
            loss=fit_loss,
            huber_delta=huber_delta,
            huber_iterations=huber_iterations,
        )
        correction = apply_weighted_ridge_residual(
            features_ncf[position:chunk_end],
            state,
            shrink=shrink,
            max_abs_scale=max_abs_scale,
            x_ncl=x_ncl[position:chunk_end],
        )
        correction_parts.append(correction)
        updates.append(
            {
                "prediction_start": int(position),
                "prediction_end": int(chunk_end),
                "fit_start": int(fit_start),
                "fit_end": int(fit_end),
                "latest_label_window": int(fit_end - 1),
            }
        )
    return torch.cat(correction_parts, dim=0), updates


def fit_weighted_ridge_residual(
    features_ncf: torch.Tensor,
    residual_nch: torch.Tensor,
    *,
    ridge: float,
    half_life: float,
    loss: str = "ridge",
    huber_delta: float = 0.1,
    huber_iterations: int = 5,
) -> Dict[str, torch.Tensor]:
    if features_ncf.ndim != 3 or residual_nch.ndim != 3:
        raise ValueError("ridge correction expects features/residual with shape [N,C,F/H].")
    if tuple(features_ncf.shape[:2]) != tuple(residual_nch.shape[:2]):
        raise ValueError("ridge correction feature/residual batch and channel dimensions must match.")
    N, C, F = features_ncf.shape
    H = int(residual_nch.shape[-1])
    if int(N) <= 0:
        raise ValueError("ridge correction requires at least one fit sample.")
    features = features_ncf.to(dtype=torch.float64)
    residual = residual_nch.to(dtype=torch.float64)
    mean_cf = features.mean(dim=0)
    std_cf = features.std(dim=0, unbiased=False).clamp_min(1.0e-5)
    z_ncf = (features - mean_cf.unsqueeze(0)) / std_cf.unsqueeze(0)
    design_ncf = torch.cat(
        [torch.ones(N, C, 1, dtype=z_ncf.dtype, device=z_ncf.device), z_ncf],
        dim=-1,
    )
    if float(half_life) > 0.0:
        age = torch.arange(N - 1, -1, -1, dtype=torch.float64, device=features.device)
        weights_n = torch.exp(-math.log(2.0) * age / float(half_life))
    else:
        weights_n = torch.ones(N, dtype=torch.float64, device=features.device)
    weights_n = weights_n / weights_n.mean().clamp_min(1.0e-12)
    coef_cfh = torch.empty(C, F + 1, H, dtype=torch.float64, device=features.device)
    eye = torch.eye(F + 1, dtype=torch.float64, device=features.device)
    eye[0, 0] = 0.0
    for channel in range(C):
        design = design_ncf[:, channel, :]
        target = residual[:, channel, :]
        weighted_design = design * weights_n.sqrt().unsqueeze(-1)
        weighted_target = target * weights_n.sqrt().unsqueeze(-1)
        lhs = weighted_design.transpose(0, 1).matmul(weighted_design) + max(0.0, float(ridge)) * eye
        rhs = weighted_design.transpose(0, 1).matmul(weighted_target)
        coef = torch.linalg.solve(lhs, rhs)
        loss_mode = str(loss or "ridge").lower()
        if loss_mode not in {"ridge", "mse", "huber", "huber_irls"}:
            raise ValueError("walk-forward correction loss must be ridge or huber_irls.")
        if loss_mode in {"huber", "huber_irls"}:
            delta = max(float(huber_delta), 1.0e-6)
            for _ in range(max(1, int(huber_iterations))):
                error_nh = target - design.matmul(coef)
                robust_nh = torch.clamp(delta / error_nh.abs().clamp_min(1.0e-8), max=1.0)
                combined_nh = robust_nh * weights_n.unsqueeze(-1)
                lhs_hdd = torch.einsum("nd,nh,ne->hde", design, combined_nh, design)
                lhs_hdd = lhs_hdd + max(0.0, float(ridge)) * eye.unsqueeze(0)
                rhs_hd = torch.einsum("nd,nh,nh->hd", design, combined_nh, target)
                coef = torch.linalg.solve(lhs_hdd, rhs_hd.unsqueeze(-1)).squeeze(-1).transpose(0, 1)
        coef_cfh[channel] = coef
    return {
        "feature_mean_cf": mean_cf.to(dtype=torch.float32),
        "feature_std_cf": std_cf.to(dtype=torch.float32),
        "coef_cfh": coef_cfh.to(dtype=torch.float32),
    }


def apply_weighted_ridge_residual(
    features_ncf: torch.Tensor,
    state: Dict[str, torch.Tensor],
    *,
    shrink: float,
    max_abs_scale: float,
    x_ncl: torch.Tensor,
    domain_align_features: bool = False,
    domain_align_channels: Optional[List[int]] = None,
    domain_align_causal_prior_count: int = 0,
    domain_align_causal_half_life: float = 0.0,
    domain_align_causal_warmup_windows: int = 0,
) -> torch.Tensor:
    mean = state["feature_mean_cf"].to(device=features_ncf.device, dtype=features_ncf.dtype)
    std = state["feature_std_cf"].to(device=features_ncf.device, dtype=features_ncf.dtype)
    coef = state["coef_cfh"].to(device=features_ncf.device, dtype=features_ncf.dtype)
    z = None
    warmup_windows = max(0, min(int(domain_align_causal_warmup_windows), int(features_ncf.shape[0])))
    if (bool(domain_align_features) or domain_align_channels) and warmup_windows > 0:
        target_mean = features_ncf[:warmup_windows].mean(dim=0)
        target_std = features_ncf[:warmup_windows].std(dim=0, unbiased=False).clamp_min(1.0e-5)
        if bool(domain_align_features):
            mean = target_mean
            std = target_std
        else:
            align_mask = torch.zeros(int(features_ncf.shape[1]), dtype=torch.bool, device=features_ncf.device)
            align_mask[[int(channel) for channel in (domain_align_channels or [])]] = True
            mean = torch.where(align_mask.view(-1, 1), target_mean, mean)
            std = torch.where(align_mask.view(-1, 1), target_std, std)
    elif (bool(domain_align_features) or domain_align_channels) and float(domain_align_causal_half_life) > 0.0:
        alpha = 1.0 - math.exp(-math.log(2.0) / float(domain_align_causal_half_life))
        running_mean = mean
        running_second = std.square() + mean.square()
        mean_parts = []
        std_parts = []
        for sample in features_ncf:
            running_mean = (1.0 - alpha) * running_mean + alpha * sample
            running_second = (1.0 - alpha) * running_second + alpha * sample.square()
            mean_parts.append(running_mean)
            std_parts.append((running_second - running_mean.square()).clamp_min(1.0e-10).sqrt())
        target_mean = torch.stack(mean_parts, dim=0)
        target_std = torch.stack(std_parts, dim=0)
        if bool(domain_align_features):
            use_mean = target_mean
            use_std = target_std
        else:
            align_mask = torch.zeros(int(features_ncf.shape[1]), dtype=torch.bool, device=features_ncf.device)
            align_mask[[int(channel) for channel in (domain_align_channels or [])]] = True
            use_mean = torch.where(align_mask.view(1, -1, 1), target_mean, mean.unsqueeze(0))
            use_std = torch.where(align_mask.view(1, -1, 1), target_std, std.unsqueeze(0))
        z = (features_ncf - use_mean) / use_std.clamp_min(1.0e-5)
    elif (bool(domain_align_features) or domain_align_channels) and int(domain_align_causal_prior_count) > 0:
        prior = float(max(1, int(domain_align_causal_prior_count)))
        train_second = std.square() + mean.square()
        count = prior + torch.arange(
            1,
            int(features_ncf.shape[0]) + 1,
            device=features_ncf.device,
            dtype=features_ncf.dtype,
        ).view(-1, 1, 1)
        running_mean = (prior * mean.unsqueeze(0) + features_ncf.cumsum(dim=0)) / count
        running_second = (prior * train_second.unsqueeze(0) + features_ncf.square().cumsum(dim=0)) / count
        running_std = (running_second - running_mean.square()).clamp_min(1.0e-10).sqrt()
        if bool(domain_align_features):
            use_mean = running_mean
            use_std = running_std
        else:
            align_mask = torch.zeros(int(features_ncf.shape[1]), dtype=torch.bool, device=features_ncf.device)
            align_mask[[int(channel) for channel in (domain_align_channels or [])]] = True
            use_mean = torch.where(align_mask.view(1, -1, 1), running_mean, mean.unsqueeze(0))
            use_std = torch.where(align_mask.view(1, -1, 1), running_std, std.unsqueeze(0))
        z = (features_ncf - use_mean) / use_std.clamp_min(1.0e-5)
    elif bool(domain_align_features) or domain_align_channels:
        eval_mean = features_ncf.mean(dim=0)
        eval_std = features_ncf.std(dim=0, unbiased=False).clamp_min(1.0e-5)
        if bool(domain_align_features):
            mean = eval_mean
            std = eval_std
        else:
            align_mask = torch.zeros(int(features_ncf.shape[1]), dtype=torch.bool, device=features_ncf.device)
            align_mask[[int(channel) for channel in (domain_align_channels or [])]] = True
            mean = torch.where(align_mask.view(-1, 1), eval_mean, mean)
            std = torch.where(align_mask.view(-1, 1), eval_std, std)
    if z is None:
        z = (features_ncf - mean.unsqueeze(0)) / std.unsqueeze(0).clamp_min(1.0e-5)
    design = torch.cat([torch.ones(*z.shape[:2], 1, device=z.device, dtype=z.dtype), z], dim=-1)
    correction = torch.einsum("ncf,cfh->nch", design, coef) * float(shrink)
    if warmup_windows > 0:
        correction[:warmup_windows] = 0.0
    if float(max_abs_scale) > 0.0:
        scale = x_ncl.std(dim=-1, unbiased=False).clamp_min(1.0e-5).unsqueeze(-1)
        bound = float(max_abs_scale) * scale
        correction = torch.maximum(torch.minimum(correction, bound), -bound)
    return correction


def stabilize_correction_horizon_mean(
    correction_nch: torch.Tensor,
    reference_correction_nch: torch.Tensor,
    *,
    x_ncl: torch.Tensor,
    max_abs_scale: float,
) -> torch.Tensor:
    if correction_nch.ndim != 3 or reference_correction_nch.ndim != 3:
        raise ValueError("correction mean stabilization expects [N,C,H] tensors.")
    if int(correction_nch.shape[1]) != int(reference_correction_nch.shape[1]):
        raise ValueError("correction mean stabilization channel dimensions must match.")
    reference_mean_c = reference_correction_nch.mean(dim=(0, 2))
    stabilized = correction_nch - correction_nch.mean(dim=-1, keepdim=True)
    stabilized = stabilized + reference_mean_c.view(1, -1, 1)
    if float(max_abs_scale) > 0.0:
        scale = x_ncl.std(dim=-1, unbiased=False).clamp_min(1.0e-5).unsqueeze(-1)
        bound = float(max_abs_scale) * scale
        stabilized = torch.maximum(torch.minimum(stabilized, bound), -bound)
    return stabilized


def build_correction_gate_features(
    x_ncl: torch.Tensor,
    base_nch: torch.Tensor,
    correction_nch: torch.Tensor,
    *,
    patch_len: int,
) -> Tuple[torch.Tensor, int]:
    if tuple(base_nch.shape) != tuple(correction_nch.shape):
        raise ValueError("correction gate base/correction shapes must match.")
    N, C, H = base_nch.shape
    patch = int(patch_len)
    if patch <= 0 or int(H) % patch != 0 or int(x_ncl.shape[-1]) < int(H):
        raise ValueError("correction gate patch_len must divide the horizon and fit in input history.")
    Q = int(H) // patch
    eps = 1.0e-6
    sample_features, _ = build_input_correction_features(x_ncl, base_nch)
    sample_features = sample_features.unsqueeze(2).expand(N, C, Q, -1)
    hist_patch = x_ncl[..., -H:].reshape(N, C, Q, patch)
    base_patch = base_nch.reshape(N, C, Q, patch)
    corr_patch = correction_nch.reshape(N, C, Q, patch)
    hist_scale = x_ncl.std(dim=-1, unbiased=False).clamp_min(eps).unsqueeze(-1)
    local_parts = [
        hist_patch.mean(dim=-1, keepdim=True),
        hist_patch.std(dim=-1, unbiased=False, keepdim=True),
        _sequence_slope(hist_patch).unsqueeze(-1),
        base_patch.mean(dim=-1, keepdim=True),
        base_patch.std(dim=-1, unbiased=False, keepdim=True),
        _sequence_slope(base_patch).unsqueeze(-1),
        corr_patch.mean(dim=-1, keepdim=True),
        corr_patch.std(dim=-1, unbiased=False, keepdim=True),
        corr_patch.abs().mean(dim=-1, keepdim=True),
        corr_patch.abs().amax(dim=-1, keepdim=True),
        _sequence_slope(corr_patch).unsqueeze(-1),
        (corr_patch.mean(dim=-1) / hist_scale).unsqueeze(-1),
        (corr_patch.abs().mean(dim=-1) / hist_scale).unsqueeze(-1),
    ]
    channel_eye = torch.eye(C, dtype=x_ncl.dtype, device=x_ncl.device).view(1, C, 1, C).expand(N, C, Q, C)
    patch_eye = torch.eye(Q, dtype=x_ncl.dtype, device=x_ncl.device).view(1, 1, Q, Q).expand(N, C, Q, Q)
    features = torch.cat([sample_features, *local_parts, channel_eye, patch_eye], dim=-1)
    return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0), Q


def correction_gate_targets(
    base_nch: torch.Tensor,
    correction_nch: torch.Tensor,
    target_nch: torch.Tensor,
    *,
    patch_len: int,
) -> torch.Tensor:
    N, C, H = base_nch.shape
    patch = int(patch_len)
    Q = int(H) // patch
    base_patch = base_nch.reshape(N, C, Q, patch)
    pred_patch = (base_nch + correction_nch).reshape(N, C, Q, patch)
    target_patch = target_nch.reshape(N, C, Q, patch)
    base_mse = (base_patch - target_patch).square().mean(dim=-1)
    pred_mse = (pred_patch - target_patch).square().mean(dim=-1)
    base_mae = (base_patch - target_patch).abs().mean(dim=-1)
    pred_mae = (pred_patch - target_patch).abs().mean(dim=-1)
    return ((pred_mse < base_mse) & (pred_mae <= base_mae)).to(dtype=torch.float32)


class CorrectionPatchGate(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(feature_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def apply_correction_patch_mask(
    correction_nch: torch.Tensor,
    active_ncq: torch.Tensor,
    *,
    patch_len: int,
) -> torch.Tensor:
    N, C, H = correction_nch.shape
    Q = int(H) // int(patch_len)
    if tuple(active_ncq.shape) != (int(N), int(C), int(Q)):
        raise ValueError("correction patch mask must have shape [N,C,Q].")
    return (
        correction_nch.reshape(N, C, Q, int(patch_len))
        * active_ncq.to(device=correction_nch.device, dtype=correction_nch.dtype).unsqueeze(-1)
    ).reshape(N, C, H)


def fit_correction_patch_gate(
    *,
    x_ncl: torch.Tensor,
    base_nch: torch.Tensor,
    correction_nch: torch.Tensor,
    target_nch: torch.Tensor,
    active_channels: List[int],
    patch_len: int = 24,
    train_fraction: float = 0.75,
    hidden_dim: int = 32,
    epochs: int = 30,
    lr: float = 1.0e-3,
    feature_clip: float = 5.0,
) -> Tuple[CorrectionPatchGate, Dict[str, torch.Tensor], Dict[str, object]]:
    features_ncqf, Q = build_correction_gate_features(
        x_ncl,
        base_nch,
        correction_nch,
        patch_len=int(patch_len),
    )
    labels_ncq = correction_gate_targets(
        base_nch,
        correction_nch,
        target_nch,
        patch_len=int(patch_len),
    )
    N, C, _, F = features_ncqf.shape
    split = max(1, min(N - 1, int(round(N * float(train_fraction))))) if N > 1 else N
    active_c = torch.zeros(C, dtype=torch.bool)
    active_c[[int(channel) for channel in active_channels]] = True
    train_feat = features_ncqf[:split, active_c].reshape(-1, F)
    train_label = labels_ncq[:split, active_c].reshape(-1)
    hold_feat = features_ncqf[split:, active_c].reshape(-1, F)
    mean_f = train_feat.mean(dim=0)
    std_f = train_feat.std(dim=0, unbiased=False).clamp_min(1.0e-5)

    def standardize(feat: torch.Tensor) -> torch.Tensor:
        return ((feat - mean_f) / std_f).clamp(-float(feature_clip), float(feature_clip))

    train_z = standardize(train_feat)
    hold_z = standardize(hold_feat)
    gate = CorrectionPatchGate(F, hidden_dim=int(hidden_dim))
    positives = train_label.sum().clamp_min(1.0)
    negatives = (1.0 - train_label).sum().clamp_min(1.0)
    pos_weight = (negatives / positives).clamp(0.25, 8.0)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=float(lr), weight_decay=1.0e-4)
    best_state = None
    best_hold = float("inf")
    batch_size = 4096
    for _ in range(max(1, int(epochs))):
        gate.train()
        order = torch.randperm(int(train_z.shape[0]))
        for start in range(0, int(order.numel()), batch_size):
            idx = order[start : start + batch_size]
            logits = gate(train_z.index_select(0, idx))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                train_label.index_select(0, idx),
                pos_weight=pos_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        gate.eval()
        with torch.no_grad():
            if int(hold_z.shape[0]) > 0:
                hold_loss = float(
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        gate(hold_z),
                        labels_ncq[split:, active_c].reshape(-1),
                        pos_weight=pos_weight,
                    ).item()
                )
            else:
                hold_loss = 0.0
        if hold_loss < best_hold:
            best_hold = hold_loss
            best_state = {key: value.detach().clone() for key, value in gate.state_dict().items()}
    if best_state is not None:
        gate.load_state_dict(best_state)
    gate.eval()
    state = {"mean_f": mean_f, "std_f": std_f, "feature_clip": torch.tensor(float(feature_clip))}

    threshold_rows = []
    best_threshold = 1.01
    best_mse = float((base_nch[split:] - target_nch[split:]).square().mean().item()) if split < N else float("inf")
    if split < N:
        with torch.no_grad():
            hold_prob_active = torch.sigmoid(gate(hold_z)).reshape(N - split, len(active_channels), Q)
        full_prob = torch.zeros(N - split, C, Q)
        full_prob[:, active_c] = hold_prob_active
        base_hold = base_nch[split:]
        target_hold = target_nch[split:]
        base_mse = float((base_hold - target_hold).square().mean().item())
        base_mae = float((base_hold - target_hold).abs().mean().item())
        thresholds = torch.linspace(0.50, 0.99, 50).tolist() + [1.01]
        for threshold in thresholds:
            active = full_prob >= float(threshold)
            gated = base_hold + apply_correction_patch_mask(
                correction_nch[split:],
                active,
                patch_len=int(patch_len),
            )
            mse = float((gated - target_hold).square().mean().item())
            mae = float((gated - target_hold).abs().mean().item())
            feasible = bool(mse <= base_mse and mae <= base_mae)
            threshold_rows.append(
                {
                    "threshold": float(threshold),
                    "mse": mse,
                    "mae": mae,
                    "mse_gain_pct": 100.0 * (base_mse - mse) / max(base_mse, 1.0e-12),
                    "mae_gain_pct": 100.0 * (base_mae - mae) / max(base_mae, 1.0e-12),
                    "apply_rate": float(active[:, active_c].to(dtype=torch.float32).mean().item()),
                    "feasible": feasible,
                }
            )
            if feasible and mse < best_mse:
                best_mse = mse
                best_threshold = float(threshold)
    summary = {
        "train_windows": int(split),
        "holdout_windows": int(N - split),
        "train_positive_rate": float(train_label.mean().item()),
        "holdout_positive_rate": float(labels_ncq[split:, active_c].mean().item()) if split < N else 0.0,
        "best_holdout_bce": float(best_hold),
        "threshold": float(best_threshold),
        "threshold_rows": threshold_rows,
    }
    return gate, state, summary


@torch.no_grad()
def gate_correction_patches(
    *,
    gate: CorrectionPatchGate,
    state: Dict[str, torch.Tensor],
    x_ncl: torch.Tensor,
    base_nch: torch.Tensor,
    correction_nch: torch.Tensor,
    active_channels: List[int],
    threshold: float,
    patch_len: int = 24,
) -> Tuple[torch.Tensor, float]:
    features, Q = build_correction_gate_features(x_ncl, base_nch, correction_nch, patch_len=int(patch_len))
    mean = state["mean_f"].to(dtype=features.dtype)
    std = state["std_f"].to(dtype=features.dtype)
    clip = float(state["feature_clip"].item())
    z = ((features - mean) / std).clamp(-clip, clip)
    prob = torch.sigmoid(gate(z.reshape(-1, int(z.shape[-1])))).reshape(*z.shape[:-1])
    active_c = torch.zeros(int(base_nch.shape[1]), dtype=torch.bool)
    active_c[[int(channel) for channel in active_channels]] = True
    active = (prob >= float(threshold)) & active_c.view(1, -1, 1)
    gated = apply_correction_patch_mask(correction_nch, active, patch_len=int(patch_len))
    return gated, float(active[:, active_c].to(dtype=torch.float32).mean().item())


@torch.no_grad()
def collect_anchor_predictions(
    *,
    model,
    loader: DataLoader,
    cluster_id_c: torch.Tensor,
    device: torch.device,
    eval_start: int,
    input_len: int,
    moe_cfg: Dict[str, object],
    observed_history_tc: torch.Tensor,
    anchor_artifacts: Dict[str, object],
    learnable_anchor: ClusterwiseLearnableOutputAnchor,
) -> Dict[str, torch.Tensor]:
    model.eval()
    learnable_anchor.eval()
    x_parts: List[torch.Tensor] = []
    y_parts: List[torch.Tensor] = []
    base_parts: List[torch.Tensor] = []
    query_parts: List[torch.Tensor] = []
    history_cfg = anchor_artifacts["history_anchor_cfg"]
    model_stat_cfg = anchor_artifacts["model_train_stat_adapter_cfg"]
    model_stat_pc = anchor_artifacts["model_train_stat_adapter_pc"]
    for x, y, idx in loader:
        x = x.to(device)
        y = y.to(device)
        query = int(eval_start) + idx.to(device=device, dtype=torch.long)
        x_model = apply_train_stat_input_centering(
            x,
            query_start_abs_b=query,
            stat_anchor_pc=model_stat_pc,
            cfg=model_stat_cfg,
        )
        raw = model(x_model, cluster_id_c)
        base = apply_history_anchor_adapter(
            raw,
            base_pred_bch=raw,
            observed_history_tc=observed_history_tc,
            query_start_abs_b=query,
            input_len=int(input_len),
            cfg=history_cfg,
        )
        base = apply_train_stat_anchor_expert(
            base,
            base_pred_bch=base,
            x_bcl=x,
            query_start_abs_b=query,
            input_len=int(input_len),
            stat_anchor_pc=model_stat_pc,
            cfg=model_stat_cfg,
        )
        anchored = apply_moe_output_anchor_experts(
            base,
            base_pred_bch=base,
            x_bcl=x,
            query_start_abs_b=query,
            input_len=int(input_len),
            moe_cfg=moe_cfg,
            moe_enable=True,
            observed_history_tc=observed_history_tc,
            train_stat_anchor_pc=anchor_artifacts["train_stat_anchor_pc"],
            train_residual_anchor_phc=anchor_artifacts["train_residual_anchor_phc"],
            learnable_output_anchor=learnable_anchor,
            cluster_id_c=cluster_id_c,
        )
        x_parts.append(x.cpu())
        y_parts.append(y.cpu())
        base_parts.append(anchored.cpu())
        query_parts.append(query.cpu())
    return {
        "x": torch.cat(x_parts, dim=0),
        "y": torch.cat(y_parts, dim=0),
        "base": torch.cat(base_parts, dim=0),
        "query_start_abs": torch.cat(query_parts, dim=0),
    }


def _metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    return {
        "mse": float((pred - target).square().mean().item()),
        "mae": float((pred - target).abs().mean().item()),
    }


def walk_forward_diagnostic(
    tensors: Dict[str, torch.Tensor],
    *,
    blocks: int,
    warmup_blocks: int,
    ridge: float,
    half_life: float,
    shrink: float,
    max_abs_scale: float,
    active_channels: Optional[List[int]] = None,
    fit_loss: str = "ridge",
    huber_delta: float = 0.1,
    huber_iterations: int = 5,
    max_fit_windows: int = 0,
    stabilize_mean: bool = False,
    use_correction_gate: bool = False,
    domain_align_features: bool = False,
    domain_align_channels: Optional[List[int]] = None,
    domain_align_causal_prior_count: int = 0,
    domain_align_causal_half_life: float = 0.0,
    local_normalize: bool = False,
    domain_align_causal_warmup_windows: int = 0,
    label_delay: int = 0,
    online_refit_interval: int = 0,
    online_label_delay: int = 96,
) -> Dict[str, object]:
    x = tensors["x"]
    y = tensors["y"]
    base = tensors["base"]
    features, residual_scale, feature_names = prepare_input_correction_features(
        x,
        base,
        local_normalize=local_normalize,
    )
    N = int(x.shape[0])
    block_count = max(2, min(int(blocks), N))
    warmup = max(1, min(int(warmup_blocks), block_count - 1))
    rows: List[Dict[str, object]] = []
    active = sorted({int(channel) for channel in (active_channels or range(int(base.shape[1])))})
    if any(channel < 0 or channel >= int(base.shape[1]) for channel in active):
        raise ValueError("walk-forward active channel index is out of range.")
    active_mask = torch.zeros(int(base.shape[1]), dtype=torch.bool)
    active_mask[active] = True
    if int(online_refit_interval) > 0 and (
        bool(local_normalize)
        or bool(stabilize_mean)
        or bool(use_correction_gate)
        or bool(domain_align_features)
        or bool(domain_align_channels)
        or int(domain_align_causal_prior_count) > 0
        or float(domain_align_causal_half_life) > 0.0
        or int(domain_align_causal_warmup_windows) > 0
    ):
        raise ValueError("online refit is a standalone causal correction mode.")
    pred_parts: List[torch.Tensor] = []
    base_parts: List[torch.Tensor] = []
    target_parts: List[torch.Tensor] = []
    for block_idx in range(warmup, block_count):
        start = (N * block_idx) // block_count
        end = (N * (block_idx + 1)) // block_count
        fit_end = max(1, start - int(label_delay) + 1) if int(label_delay) > 0 else start
        fit_start = max(0, fit_end - max(0, int(max_fit_windows))) if int(max_fit_windows) > 0 else 0
        fit_idx = slice(fit_start, fit_end)
        eval_idx = slice(start, end)
        state = fit_weighted_ridge_residual(
            features[fit_idx],
            (y[fit_idx] - base[fit_idx]) / residual_scale[fit_idx]
            if bool(local_normalize)
            else y[fit_idx] - base[fit_idx],
            ridge=ridge,
            half_life=half_life,
            loss=fit_loss,
            huber_delta=huber_delta,
            huber_iterations=huber_iterations,
        )
        correction = apply_weighted_ridge_residual(
            features[eval_idx],
            state,
            shrink=shrink,
            max_abs_scale=0.0 if bool(local_normalize) else max_abs_scale,
            x_ncl=x[eval_idx],
            domain_align_features=domain_align_features,
            domain_align_channels=domain_align_channels,
            domain_align_causal_prior_count=domain_align_causal_prior_count,
            domain_align_causal_half_life=domain_align_causal_half_life,
            domain_align_causal_warmup_windows=domain_align_causal_warmup_windows,
        )
        if bool(local_normalize):
            correction = restore_local_correction_scale(
                correction,
                scale_nc1=residual_scale[eval_idx],
                x_ncl=x[eval_idx],
                max_abs_scale=max_abs_scale,
            )
        online_updates = None
        if int(online_refit_interval) > 0:
            correction, online_updates = online_refit_corrections(
                features_ncf=features,
                x_ncl=x,
                base_nch=base,
                target_nch=y,
                eval_start=start,
                eval_end=end,
                update_interval=int(online_refit_interval),
                label_delay=int(online_label_delay),
                ridge=ridge,
                half_life=half_life,
                shrink=shrink,
                max_abs_scale=max_abs_scale,
                fit_loss=fit_loss,
                huber_delta=huber_delta,
                huber_iterations=huber_iterations,
                max_fit_windows=max_fit_windows,
            )
        if bool(stabilize_mean):
            reference_correction = apply_weighted_ridge_residual(
                features[fit_idx],
                state,
                shrink=shrink,
                max_abs_scale=max_abs_scale,
                x_ncl=x[fit_idx],
            )
            correction = stabilize_correction_horizon_mean(
                correction,
                reference_correction,
                x_ncl=x[eval_idx],
                max_abs_scale=max_abs_scale,
            )
        gate_summary = None
        if bool(use_correction_gate):
            fit_correction = apply_weighted_ridge_residual(
                features[fit_idx],
                state,
                shrink=shrink,
                max_abs_scale=max_abs_scale,
                x_ncl=x[fit_idx],
            )
            gate, gate_state, gate_summary = fit_correction_patch_gate(
                x_ncl=x[fit_idx],
                base_nch=base[fit_idx],
                correction_nch=fit_correction,
                target_nch=y[fit_idx],
                active_channels=active,
            )
            correction, gate_apply_rate = gate_correction_patches(
                gate=gate,
                state=gate_state,
                x_ncl=x[eval_idx],
                base_nch=base[eval_idx],
                correction_nch=correction,
                active_channels=active,
                threshold=float(gate_summary["threshold"]),
            )
            gate_summary = dict(gate_summary)
            gate_summary["eval_apply_rate"] = float(gate_apply_rate)
        correction = correction * active_mask.to(device=correction.device, dtype=correction.dtype).view(1, -1, 1)
        pred = base[eval_idx] + correction
        base_metric = _metrics(base[eval_idx], y[eval_idx])
        pred_metric = _metrics(pred, y[eval_idx])
        per_channel = []
        for channel in range(int(base.shape[1])):
            base_channel = _metrics(base[eval_idx, channel], y[eval_idx, channel])
            pred_channel = _metrics(pred[:, channel], y[eval_idx, channel])
            per_channel.append(
                {
                    "channel_index": int(channel),
                    "base_mse": base_channel["mse"],
                    "corrected_mse": pred_channel["mse"],
                    "mse_gain_pct": 100.0
                    * (base_channel["mse"] - pred_channel["mse"])
                    / max(base_channel["mse"], 1.0e-12),
                    "base_mae": base_channel["mae"],
                    "corrected_mae": pred_channel["mae"],
                    "mae_gain_pct": 100.0
                    * (base_channel["mae"] - pred_channel["mae"])
                    / max(base_channel["mae"], 1.0e-12),
                }
            )
        rows.append(
            {
                "block_index": int(block_idx),
                "fit_windows": int(fit_end),
                "fit_start_window": int(fit_start),
                "fit_end_window": int(fit_end),
                "effective_fit_windows": int(fit_end - fit_start),
                "start_window": int(start),
                "end_window": int(end),
                "base_mse": base_metric["mse"],
                "corrected_mse": pred_metric["mse"],
                "mse_gain_pct": 100.0 * (base_metric["mse"] - pred_metric["mse"]) / max(base_metric["mse"], 1.0e-12),
                "base_mae": base_metric["mae"],
                "corrected_mae": pred_metric["mae"],
                "mae_gain_pct": 100.0 * (base_metric["mae"] - pred_metric["mae"]) / max(base_metric["mae"], 1.0e-12),
                "correction_rms": float(correction.square().mean().sqrt().item()),
                "correction_max_abs": float(correction.abs().max().item()),
                "per_channel": per_channel,
                "correction_gate": gate_summary,
                "online_refit_updates": online_updates,
            }
        )
        pred_parts.append(pred)
        base_parts.append(base[eval_idx])
        target_parts.append(y[eval_idx])
    pred_tail = torch.cat(pred_parts, dim=0)
    base_tail = torch.cat(base_parts, dim=0)
    target_tail = torch.cat(target_parts, dim=0)
    base_metric = _metrics(base_tail, target_tail)
    corrected_metric = _metrics(pred_tail, target_tail)
    positive_mse_blocks = sum(float(row["mse_gain_pct"]) >= 0.0 for row in rows)
    positive_mae_blocks = sum(float(row["mae_gain_pct"]) >= 0.0 for row in rows)
    per_channel_tail = []
    for channel in range(int(base.shape[1])):
        base_channel = _metrics(base_tail[:, channel], target_tail[:, channel])
        pred_channel = _metrics(pred_tail[:, channel], target_tail[:, channel])
        channel_block_rows = [row["per_channel"][channel] for row in rows]
        per_channel_tail.append(
            {
                "channel_index": int(channel),
                "base_mse": base_channel["mse"],
                "corrected_mse": pred_channel["mse"],
                "mse_gain_pct": 100.0
                * (base_channel["mse"] - pred_channel["mse"])
                / max(base_channel["mse"], 1.0e-12),
                "base_mae": base_channel["mae"],
                "corrected_mae": pred_channel["mae"],
                "mae_gain_pct": 100.0
                * (base_channel["mae"] - pred_channel["mae"])
                / max(base_channel["mae"], 1.0e-12),
                "positive_mse_blocks": sum(float(row["mse_gain_pct"]) >= 0.0 for row in channel_block_rows),
                "positive_mae_blocks": sum(float(row["mae_gain_pct"]) >= 0.0 for row in channel_block_rows),
                "mse_gain_pct_by_block": [float(row["mse_gain_pct"]) for row in channel_block_rows],
                "mae_gain_pct_by_block": [float(row["mae_gain_pct"]) for row in channel_block_rows],
            }
        )
    return {
        "config": {
            "blocks": int(block_count),
            "warmup_blocks": int(warmup),
            "ridge": float(ridge),
            "half_life": float(half_life),
            "shrink": float(shrink),
            "max_abs_scale": float(max_abs_scale),
            "active_channels": active,
            "fit_loss": str(fit_loss),
            "huber_delta": float(huber_delta),
            "huber_iterations": int(huber_iterations),
            "max_fit_windows": int(max_fit_windows),
            "stabilize_correction_mean": bool(stabilize_mean),
            "correction_gate": bool(use_correction_gate),
            "domain_align_features": bool(domain_align_features),
            "domain_align_channels": [int(channel) for channel in (domain_align_channels or [])],
            "domain_align_causal_prior_count": int(domain_align_causal_prior_count),
            "domain_align_causal_half_life": float(domain_align_causal_half_life),
            "local_normalize": bool(local_normalize),
            "domain_align_causal_warmup_windows": int(domain_align_causal_warmup_windows),
            "label_delay": int(label_delay),
            "online_refit_interval": int(online_refit_interval),
            "online_label_delay": int(online_label_delay),
        },
        "feature_names": feature_names,
        "full_anchor": _metrics(base, y),
        "walk_forward_base": base_metric,
        "walk_forward_corrected": corrected_metric,
        "mse_gain_pct": 100.0 * (base_metric["mse"] - corrected_metric["mse"]) / max(base_metric["mse"], 1.0e-12),
        "mae_gain_pct": 100.0 * (base_metric["mae"] - corrected_metric["mae"]) / max(base_metric["mae"], 1.0e-12),
        "positive_mse_blocks": int(positive_mse_blocks),
        "positive_mae_blocks": int(positive_mae_blocks),
        "required_blocks": int(len(rows)),
        "per_channel": per_channel_tail,
        "blocks": rows,
    }


def run(args: argparse.Namespace) -> Dict[str, object]:
    cfg = load_yaml(str(args.config))
    cfg.setdefault("eval", {})["skip_test"] = True
    set_seed(int(cfg["exp"]["seed"]), deterministic=bool(cfg["exp"].get("deterministic", False)))
    requested = str(args.device or cfg["exp"].get("device", "cpu"))
    device = torch.device(requested if torch.cuda.is_available() and requested != "cpu" else "cpu")
    checkpoint = torch.load(str(args.checkpoint), map_location=device, weights_only=False)
    model, _, _, cluster_id_c, K, moe_cfg, _ = _build_modules(cfg, checkpoint, device)
    data_tc = _read_data_for_cfg(cfg)
    data_window_tc, loaders, eval_starts, train_loader, window_meta = _make_loaders_with_authorized_test(
        cfg,
        data_tc,
        batch_size=int(cfg["train"].get("batch_size", 64)),
        include_test=False,
    )
    anchor_artifacts = _build_anchor_artifacts(
        cfg=cfg,
        checkpoint=checkpoint,
        model=model,
        cluster_id_c=cluster_id_c,
        data_tc=data_window_tc,
        train_loader=train_loader,
        window_meta=window_meta,
        device=device,
    )
    anchor_cfg = _normalize_learnable_output_anchor_cfg(moe_cfg.get("learnable_output_anchor", {}))
    learnable_anchor = ClusterwiseLearnableOutputAnchor(
        num_clusters=int(K),
        num_channels=int(cluster_id_c.numel()),
        pred_len=int(window_meta["H"]),
        cfg=anchor_cfg,
    ).to(device)
    learnable_anchor.load_state_dict(checkpoint["learnable_output_anchor_state"], strict=True)
    tensors = collect_anchor_predictions(
        model=model,
        loader=loaders["val"],
        cluster_id_c=cluster_id_c,
        device=device,
        eval_start=int(eval_starts["val"]),
        input_len=int(window_meta["L"]),
        moe_cfg=moe_cfg,
        observed_history_tc=data_window_tc,
        anchor_artifacts=anchor_artifacts,
        learnable_anchor=learnable_anchor,
    )
    diagnostic = walk_forward_diagnostic(
        tensors,
        blocks=int(args.blocks),
        warmup_blocks=int(args.warmup_blocks),
        ridge=float(args.ridge),
        half_life=float(args.half_life),
        shrink=float(args.shrink),
        max_abs_scale=float(args.max_abs_scale),
        active_channels=[int(channel) for channel in args.channel_indices],
        fit_loss=str(args.fit_loss),
        huber_delta=float(args.huber_delta),
        huber_iterations=int(args.huber_iterations),
        max_fit_windows=int(args.max_fit_windows),
        stabilize_mean=bool(args.stabilize_correction_mean),
        use_correction_gate=bool(args.correction_gate),
        domain_align_features=bool(args.domain_align_features),
        domain_align_channels=None
        if args.domain_align_channel_indices is None
        else [int(channel) for channel in args.domain_align_channel_indices],
        domain_align_causal_prior_count=int(args.domain_align_causal_prior_count),
        domain_align_causal_half_life=float(args.domain_align_causal_half_life),
        local_normalize=bool(args.local_normalize),
        domain_align_causal_warmup_windows=int(args.domain_align_causal_warmup_windows),
        label_delay=int(args.walkforward_label_delay),
        online_refit_interval=int(args.online_refit_interval),
        online_label_delay=int(args.online_label_delay),
    )
    test_payload = None
    allow_test_read = bool(args.allow_test_read)
    val_gate_passed = bool(
        int(diagnostic["positive_mse_blocks"]) == int(diagnostic["required_blocks"])
        and int(diagnostic["positive_mae_blocks"]) == int(diagnostic["required_blocks"])
        and float(diagnostic["mse_gain_pct"]) > 0.0
        and float(diagnostic["mae_gain_pct"]) >= 0.0
    )
    if allow_test_read and not val_gate_passed:
        raise RuntimeError("Locked walk-forward validation gate failed; test remains unread.")
    if allow_test_read:
        data_window_test_tc, test_loaders, test_eval_starts, _, _ = _make_loaders_with_authorized_test(
            cfg,
            data_tc,
            batch_size=int(cfg["train"].get("batch_size", 64)),
            include_test=True,
        )
        test_tensors = collect_anchor_predictions(
            model=model,
            loader=test_loaders["test"],
            cluster_id_c=cluster_id_c,
            device=device,
            eval_start=int(test_eval_starts["test"]),
            input_len=int(window_meta["L"]),
            moe_cfg=moe_cfg,
            observed_history_tc=data_window_test_tc,
            anchor_artifacts=anchor_artifacts,
            learnable_anchor=learnable_anchor,
        )
        val_features, val_residual_scale, _ = prepare_input_correction_features(
            tensors["x"],
            tensors["base"],
            local_normalize=bool(args.local_normalize),
        )
        test_features, test_residual_scale, _ = prepare_input_correction_features(
            test_tensors["x"],
            test_tensors["base"],
            local_normalize=bool(args.local_normalize),
        )
        final_fit_start = max(0, int(val_features.shape[0]) - max(0, int(args.max_fit_windows))) if int(args.max_fit_windows) > 0 else 0
        final_state = fit_weighted_ridge_residual(
            val_features[final_fit_start:],
            ((tensors["y"] - tensors["base"]) / val_residual_scale)[final_fit_start:]
            if bool(args.local_normalize)
            else (tensors["y"] - tensors["base"])[final_fit_start:],
            ridge=float(args.ridge),
            half_life=float(args.half_life),
            loss=str(args.fit_loss),
            huber_delta=float(args.huber_delta),
            huber_iterations=int(args.huber_iterations),
        )
        test_correction = apply_weighted_ridge_residual(
            test_features,
            final_state,
            shrink=float(args.shrink),
            max_abs_scale=0.0 if bool(args.local_normalize) else float(args.max_abs_scale),
            x_ncl=test_tensors["x"],
            domain_align_features=bool(args.domain_align_features),
            domain_align_channels=None
            if args.domain_align_channel_indices is None
            else [int(channel) for channel in args.domain_align_channel_indices],
            domain_align_causal_prior_count=int(args.domain_align_causal_prior_count),
            domain_align_causal_half_life=float(args.domain_align_causal_half_life),
            domain_align_causal_warmup_windows=int(args.domain_align_causal_warmup_windows),
        )
        online_test_updates = None
        if int(args.online_refit_interval) > 0:
            combined_features = torch.cat([val_features, test_features], dim=0)
            combined_x = torch.cat([tensors["x"], test_tensors["x"]], dim=0)
            combined_base = torch.cat([tensors["base"], test_tensors["base"]], dim=0)
            combined_target = torch.cat([tensors["y"], test_tensors["y"]], dim=0)
            test_correction, online_test_updates = online_refit_corrections(
                features_ncf=combined_features,
                x_ncl=combined_x,
                base_nch=combined_base,
                target_nch=combined_target,
                eval_start=int(val_features.shape[0]),
                eval_end=int(combined_features.shape[0]),
                update_interval=int(args.online_refit_interval),
                label_delay=int(args.online_label_delay),
                ridge=float(args.ridge),
                half_life=float(args.half_life),
                shrink=float(args.shrink),
                max_abs_scale=float(args.max_abs_scale),
                fit_loss=str(args.fit_loss),
                huber_delta=float(args.huber_delta),
                huber_iterations=int(args.huber_iterations),
                max_fit_windows=int(args.max_fit_windows),
            )
        if bool(args.local_normalize):
            test_correction = restore_local_correction_scale(
                test_correction,
                scale_nc1=test_residual_scale,
                x_ncl=test_tensors["x"],
                max_abs_scale=float(args.max_abs_scale),
            )
        val_correction = apply_weighted_ridge_residual(
            val_features,
            final_state,
            shrink=float(args.shrink),
            max_abs_scale=0.0 if bool(args.local_normalize) else float(args.max_abs_scale),
            x_ncl=tensors["x"],
            domain_align_features=bool(args.domain_align_features),
            domain_align_channels=None
            if args.domain_align_channel_indices is None
            else [int(channel) for channel in args.domain_align_channel_indices],
            domain_align_causal_prior_count=int(args.domain_align_causal_prior_count),
            domain_align_causal_half_life=float(args.domain_align_causal_half_life),
            domain_align_causal_warmup_windows=int(args.domain_align_causal_warmup_windows),
        )
        if bool(args.local_normalize):
            val_correction = restore_local_correction_scale(
                val_correction,
                scale_nc1=val_residual_scale,
                x_ncl=tensors["x"],
                max_abs_scale=float(args.max_abs_scale),
            )
        if bool(args.stabilize_correction_mean):
            test_correction = stabilize_correction_horizon_mean(
                test_correction,
                val_correction,
                x_ncl=test_tensors["x"],
                max_abs_scale=float(args.max_abs_scale),
            )
        final_gate_summary = None
        if bool(args.correction_gate):
            gate, gate_state, final_gate_summary = fit_correction_patch_gate(
                x_ncl=tensors["x"][final_fit_start:],
                base_nch=tensors["base"][final_fit_start:],
                correction_nch=val_correction[final_fit_start:],
                target_nch=tensors["y"][final_fit_start:],
                active_channels=[int(channel) for channel in args.channel_indices],
            )
            test_correction, test_gate_apply_rate = gate_correction_patches(
                gate=gate,
                state=gate_state,
                x_ncl=test_tensors["x"],
                base_nch=test_tensors["base"],
                correction_nch=test_correction,
                active_channels=[int(channel) for channel in args.channel_indices],
                threshold=float(final_gate_summary["threshold"]),
            )
            final_gate_summary = dict(final_gate_summary)
            final_gate_summary["test_apply_rate"] = float(test_gate_apply_rate)
        active_mask = torch.zeros(int(test_correction.shape[1]), dtype=test_correction.dtype)
        active_mask[[int(channel) for channel in args.channel_indices]] = 1.0
        val_correction = val_correction * active_mask.view(1, -1, 1)
        test_correction = test_correction * active_mask.view(1, -1, 1)
        test_pred = test_tensors["base"] + test_correction
        test_base_metrics = _metrics(test_tensors["base"], test_tensors["y"])
        test_corrected_metrics = _metrics(test_pred, test_tensors["y"])
        test_per_channel = []
        for channel in range(int(test_pred.shape[1])):
            base_channel = _metrics(test_tensors["base"][:, channel], test_tensors["y"][:, channel])
            pred_channel = _metrics(test_pred[:, channel], test_tensors["y"][:, channel])
            test_per_channel.append(
                {
                    "channel_index": int(channel),
                    "active": bool(active_mask[channel].item() > 0.5),
                    "base_mse": base_channel["mse"],
                    "corrected_mse": pred_channel["mse"],
                    "mse_gain_pct": 100.0
                    * (base_channel["mse"] - pred_channel["mse"])
                    / max(base_channel["mse"], 1.0e-12),
                    "base_mae": base_channel["mae"],
                    "corrected_mae": pred_channel["mae"],
                    "mae_gain_pct": 100.0
                    * (base_channel["mae"] - pred_channel["mae"])
                    / max(base_channel["mae"], 1.0e-12),
                    "val_correction_mean": float(val_correction[:, channel].mean().item()),
                    "test_correction_mean": float(test_correction[:, channel].mean().item()),
                    "val_correction_rms": float(val_correction[:, channel].square().mean().sqrt().item()),
                    "test_correction_rms": float(test_correction[:, channel].square().mean().sqrt().item()),
                }
            )
        test_payload = {
            "fit_source": "validation_only",
            "fit_windows": int(tensors["x"].shape[0]),
            "effective_fit_windows": int(tensors["x"].shape[0] - final_fit_start),
            "test_windows": int(test_tensors["x"].shape[0]),
            "active_channels": [int(channel) for channel in args.channel_indices],
            "base": test_base_metrics,
            "corrected": test_corrected_metrics,
            "mse_gain_pct": 100.0
            * (test_base_metrics["mse"] - test_corrected_metrics["mse"])
            / max(test_base_metrics["mse"], 1.0e-12),
            "mae_gain_pct": 100.0
            * (test_base_metrics["mae"] - test_corrected_metrics["mae"])
            / max(test_base_metrics["mae"], 1.0e-12),
            "correction_rms": float(test_correction.square().mean().sqrt().item()),
            "correction_max_abs": float(test_correction.abs().max().item()),
            "per_channel": test_per_channel,
            "correction_gate": final_gate_summary,
            "online_refit_updates": online_test_updates,
        }
    payload = {
        "config_path": str(Path(args.config).resolve()),
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "test_read": allow_test_read,
        "validation_gate_passed": val_gate_passed,
        "validation_windows": int(tensors["x"].shape[0]),
        "diagnostic": diagnostic,
        "test": test_payload,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "walkforward_input_correction.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--warmup-blocks", type=int, default=2)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--half-life", type=float, default=672.0)
    parser.add_argument("--shrink", type=float, default=0.5)
    parser.add_argument("--max-abs-scale", type=float, default=0.15)
    parser.add_argument("--channel-indices", type=int, nargs="+", default=[0, 2, 6])
    parser.add_argument("--fit-loss", choices=["ridge", "huber_irls"], default="ridge")
    parser.add_argument("--huber-delta", type=float, default=0.1)
    parser.add_argument("--huber-iterations", type=int, default=5)
    parser.add_argument("--max-fit-windows", type=int, default=0)
    parser.add_argument("--stabilize-correction-mean", action="store_true")
    parser.add_argument("--correction-gate", action="store_true")
    parser.add_argument("--domain-align-features", action="store_true")
    parser.add_argument("--domain-align-channel-indices", type=int, nargs="+", default=None)
    parser.add_argument("--domain-align-causal-prior-count", type=int, default=0)
    parser.add_argument("--domain-align-causal-half-life", type=float, default=0.0)
    parser.add_argument("--local-normalize", action="store_true")
    parser.add_argument("--domain-align-causal-warmup-windows", type=int, default=0)
    parser.add_argument("--walkforward-label-delay", type=int, default=0)
    parser.add_argument("--online-refit-interval", type=int, default=0)
    parser.add_argument("--online-label-delay", type=int, default=96)
    parser.add_argument("--allow-test-read", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
