from __future__ import annotations

from typing import Dict, List, Optional

import torch


def _allowed_or_all(gain_bcp: torch.Tensor, allowed_cp: Optional[torch.Tensor]) -> torch.Tensor:
    if gain_bcp.dim() != 3:
        raise ValueError("gain_bcp must have shape [B,C,P].")
    _, C, P = [int(v) for v in gain_bcp.shape]
    if allowed_cp is None:
        return torch.ones(C, P, dtype=torch.bool, device=gain_bcp.device)
    allowed = allowed_cp.to(device=gain_bcp.device, dtype=torch.bool)
    if tuple(allowed.shape) != (C, P):
        raise ValueError(f"allowed_cp must have shape [C,P], got {tuple(allowed.shape)} vs {(C, P)}.")
    return allowed


def _gain_hinge_pressure_metrics(
    *,
    gain_bcp: torch.Tensor,
    allowed_cp: Optional[torch.Tensor],
    margin: float,
) -> Dict[str, float]:
    allowed = _allowed_or_all(gain_bcp, allowed_cp)
    gain = gain_bcp.detach().to(dtype=torch.float32)
    valid = allowed.unsqueeze(0).expand_as(gain)
    positive = (gain > float(margin)) & valid
    allowed_count_bc = valid.sum(dim=-1)
    positive_count_bc = positive.sum(dim=-1)
    sample_valid = allowed_count_bc > 0
    hinge = (float(margin) - gain).clamp_min(0.0).masked_fill(~valid, 0.0)
    skip_target = (positive_count_bc == 0) & sample_valid
    all_positive = (positive_count_bc == allowed_count_bc) & sample_valid
    any_positive = (positive_count_bc > 0) & sample_valid
    single_positive = (positive_count_bc == 1) & sample_valid
    multi_positive = (positive_count_bc > 1) & sample_valid
    sample_count = int(sample_valid.sum().item())
    allowed_branch_count = int(valid.sum().item())
    active_branch_loss = (hinge > 0.0) & valid
    total_loss = float(hinge.sum().item())
    skip_loss = float((hinge * skip_target.unsqueeze(-1).to(dtype=hinge.dtype)).sum().item())
    denom_samples = max(sample_count, 1)
    denom_branches = max(allowed_branch_count, 1)
    return {
        "sample_count": float(sample_count),
        "allowed_branch_count": float(allowed_branch_count),
        "any_positive_rate": float(any_positive.sum().item() / denom_samples),
        "all_allowed_positive_rate": float(all_positive.sum().item() / denom_samples),
        "skip_target_rate": float(skip_target.sum().item() / denom_samples),
        "single_positive_rate": float(single_positive.sum().item() / denom_samples),
        "multi_positive_rate": float(multi_positive.sum().item() / denom_samples),
        "active_branch_loss_rate": float(active_branch_loss.sum().item() / denom_branches),
        "zero_loss_all_branch_sample_rate": float(all_positive.sum().item() / denom_samples),
        "mean_hinge_loss_per_allowed_branch": float(total_loss / denom_branches),
        "loss_share_from_skip_target_samples": float(skip_loss / max(total_loss, 1.0e-12)),
    }


def _split_branch_pressure_rows(
    *,
    gain_bcp: torch.Tensor,
    split: str,
    penalty_names: List[str],
    allowed_cp: Optional[torch.Tensor],
    margin: float,
) -> List[Dict[str, object]]:
    allowed = _allowed_or_all(gain_bcp, allowed_cp)
    gain = gain_bcp.detach().to(dtype=torch.float32)
    _, C, P = [int(v) for v in gain.shape]
    if len(penalty_names) != P:
        raise ValueError("penalty_names length must match gain_bcp penalty dimension.")
    hinge = (float(margin) - gain).clamp_min(0.0)
    total_loss = float(hinge.masked_fill(~allowed.unsqueeze(0), 0.0).sum().item())
    rows: List[Dict[str, object]] = []
    for channel in range(C):
        for penalty_idx, penalty in enumerate(penalty_names):
            if not bool(allowed[channel, penalty_idx].item()):
                continue
            values = gain[:, channel, penalty_idx]
            losses = hinge[:, channel, penalty_idx]
            loss_sum = float(losses.sum().item())
            rows.append(
                {
                    "split": str(split),
                    "channel": int(channel),
                    "penalty_idx": int(penalty_idx),
                    "penalty": str(penalty),
                    "support": int(values.numel()),
                    "mean_gain": float(values.mean().item()) if int(values.numel()) else 0.0,
                    "positive_rate": float((values > float(margin)).to(dtype=torch.float32).mean().item())
                    if int(values.numel())
                    else 0.0,
                    "mean_hinge_loss": float(losses.mean().item()) if int(losses.numel()) else 0.0,
                    "loss_sum": loss_sum,
                    "loss_share": float(loss_sum / max(total_loss, 1.0e-12)),
                }
            )
    return rows
