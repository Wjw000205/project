from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnose_etth1_walkforward_input_correction import (
    _metrics,
    apply_weighted_ridge_residual,
    build_correction_gate_features,
    collect_anchor_predictions,
    fit_weighted_ridge_residual,
    prepare_input_correction_features,
)
from scripts.next11c_route_accuracy_diagnostic import _build_anchor_artifacts
from scripts.next11d_authorized_test_shift_probe import _make_loaders_with_authorized_test
from scripts.next11d_fixed_candidate_router_refit import _read_data_for_cfg
from scripts.shape_prior_diagnostic import _build_modules
from src.models.learnable_anchor import ClusterwiseLearnableOutputAnchor
from src.train import _normalize_learnable_output_anchor_cfg
from src.utils.seed import set_seed
from src.utils.yaml_io import load_yaml


EXPERT_NAMES = ["anchor", "raw_huber", "domain_aligned_huber"]
DEFAULT_EXPERT = 2


def load_learnable_anchor_state_compat(
    learnable_anchor: ClusterwiseLearnableOutputAnchor,
    state: Dict[str, torch.Tensor],
) -> List[str]:
    loaded = learnable_anchor.load_state_dict(state, strict=False)
    allowed_missing = {"active_channel_horizon_mask_ch"}
    unexpected = list(loaded.unexpected_keys)
    unsupported_missing = sorted(set(loaded.missing_keys) - allowed_missing)
    if unexpected or unsupported_missing:
        raise RuntimeError(
            "Unsupported learnable-anchor checkpoint mismatch: "
            f"missing={unsupported_missing}, unexpected={unexpected}."
        )
    return list(loaded.missing_keys)


def _patch_view(values_nch: torch.Tensor, patch_len: int) -> torch.Tensor:
    n, c, h = values_nch.shape
    patch = int(patch_len)
    if patch <= 0 or h % patch != 0:
        raise ValueError("patch_len must divide the forecast horizon.")
    return values_nch.reshape(n, c, h // patch, patch)


def build_fixed_expert_gate_features(
    x_ncl: torch.Tensor,
    base_nch: torch.Tensor,
    corrections_nceh: torch.Tensor,
    *,
    patch_len: int,
    include_domain_descriptor: bool = True,
) -> torch.Tensor:
    """Build within-domain features while retaining block-level shift descriptors."""
    if corrections_nceh.ndim != 4 or int(corrections_nceh.shape[2]) != 2:
        raise ValueError("fixed-expert gate expects two correction experts [N,C,2,H].")
    raw_feat, q = build_correction_gate_features(
        x_ncl,
        base_nch,
        corrections_nceh[:, :, 0, :],
        patch_len=int(patch_len),
    )
    aligned_feat, aligned_q = build_correction_gate_features(
        x_ncl,
        base_nch,
        corrections_nceh[:, :, 1, :],
        patch_len=int(patch_len),
    )
    if q != aligned_q:
        raise ValueError("fixed-expert gate patch counts do not match.")
    identity_dim = int(base_nch.shape[1]) + int(q)
    continuous = torch.cat(
        [
            raw_feat[..., :-identity_dim],
            aligned_feat[..., :-identity_dim],
            aligned_feat[..., :-identity_dim] - raw_feat[..., :-identity_dim],
        ],
        dim=-1,
    )
    domain_mean = continuous.mean(dim=0, keepdim=True)
    domain_std = continuous.std(dim=0, unbiased=False, keepdim=True).clamp_min(1.0e-5)
    within_domain = ((continuous - domain_mean) / domain_std).clamp(-6.0, 6.0)
    identity = raw_feat[..., -identity_dim:]
    domain_descriptor = torch.cat(
        [
            domain_mean.expand_as(continuous),
            torch.log1p(domain_std).expand_as(continuous),
        ],
        dim=-1,
    )
    parts = [within_domain]
    if bool(include_domain_descriptor):
        parts.append(domain_descriptor)
    parts.append(identity)
    return torch.nan_to_num(
        torch.cat(parts, dim=-1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def fixed_expert_patch_errors(
    base_nch: torch.Tensor,
    corrections_nceh: torch.Tensor,
    target_nch: torch.Tensor,
    *,
    patch_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    base_patch = _patch_view(base_nch, patch_len)
    target_patch = _patch_view(target_nch, patch_len)
    candidates = torch.stack(
        [
            base_nch,
            base_nch + corrections_nceh[:, :, 0, :],
            base_nch + corrections_nceh[:, :, 1, :],
        ],
        dim=2,
    )
    n, c, e, h = candidates.shape
    candidate_patch = candidates.reshape(n, c, e, h // int(patch_len), int(patch_len)).permute(0, 1, 3, 2, 4)
    error = candidate_patch - target_patch.unsqueeze(3)
    return error.square().mean(dim=-1), error.abs().mean(dim=-1)


def fixed_expert_gate_targets(
    base_nch: torch.Tensor,
    corrections_nceh: torch.Tensor,
    target_nch: torch.Tensor,
    *,
    patch_len: int,
    min_gain: float,
    mae_tolerance: float = 0.0,
    default_expert: int = DEFAULT_EXPERT,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mse, mae = fixed_expert_patch_errors(
        base_nch,
        corrections_nceh,
        target_nch,
        patch_len=int(patch_len),
    )
    default = int(default_expert)
    default_mse = mse[..., default]
    default_mae = mae[..., default]
    mse_floor = default_mse.detach().median().clamp_min(1.0e-6) * 0.05
    mae_floor = default_mae.detach().median().clamp_min(1.0e-6) * 0.05
    normalized_cost = mse / (default_mse.unsqueeze(-1) + mse_floor)
    normalized_cost = normalized_cost + 0.15 * mae / (default_mae.unsqueeze(-1) + mae_floor)
    # Prefer the stable default on numerical ties.
    normalized_cost[..., default] = normalized_cost[..., default] - 1.0e-7
    best = normalized_cost.argmin(dim=-1)
    best_mse = mse.gather(-1, best.unsqueeze(-1)).squeeze(-1)
    best_mae = mae.gather(-1, best.unsqueeze(-1)).squeeze(-1)
    relative_gain = (default_mse - best_mse) / default_mse.clamp_min(mse_floor)
    feasible = best_mae <= default_mae * (1.0 + max(0.0, float(mae_tolerance)))
    meaningful = relative_gain >= max(0.0, float(min_gain))
    target = torch.where(
        feasible & meaningful,
        best,
        torch.full_like(best, default),
    )
    return target, relative_gain.clamp_min(0.0), mse, mae


def apply_fixed_expert_routes(
    base_nch: torch.Tensor,
    corrections_nceh: torch.Tensor,
    route_ncq: torch.Tensor,
    *,
    patch_len: int,
) -> torch.Tensor:
    n, c, _, h = corrections_nceh.shape
    q = h // int(patch_len)
    zero = torch.zeros_like(corrections_nceh[:, :, :1, :])
    all_correction = torch.cat([zero, corrections_nceh], dim=2)
    patches = all_correction.reshape(n, c, 3, q, int(patch_len)).permute(0, 1, 3, 2, 4)
    gather = route_ncq.to(dtype=torch.long).unsqueeze(-1).unsqueeze(-1).expand(n, c, q, 1, int(patch_len))
    selected = patches.gather(3, gather).squeeze(3).reshape(n, c, h)
    return base_nch + selected


class FixedExpertPatchGate(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        hidden = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(int(feature_dim), hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, len(EXPERT_NAMES)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class SklearnProbabilityGate:
    def __init__(self, classifier) -> None:
        self.classifier = classifier

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        leading_shape = tuple(features.shape[:-1])
        flat = features.reshape(-1, int(features.shape[-1]))
        probabilities = self.classifier.predict_proba(flat.detach().cpu().numpy())
        full = torch.full(
            (int(flat.shape[0]), len(EXPERT_NAMES)),
            1.0e-8,
            dtype=features.dtype,
        )
        for source_idx, class_idx in enumerate(self.classifier.classes_):
            full[:, int(class_idx)] = torch.from_numpy(probabilities[:, source_idx]).to(dtype=features.dtype)
        return full.clamp_min(1.0e-8).log().reshape(*leading_shape, len(EXPERT_NAMES))


def _gate_logits(gate, features: torch.Tensor) -> torch.Tensor:
    return gate(features)


def _class_rates(route: torch.Tensor, active_mask_c: torch.Tensor) -> Dict[str, float]:
    selected = route[:, active_mask_c, :].reshape(-1)
    rates = torch.bincount(selected, minlength=len(EXPERT_NAMES)).to(dtype=torch.float32)
    rates = rates / rates.sum().clamp_min(1.0)
    return {name: float(rates[idx].item()) for idx, name in enumerate(EXPERT_NAMES)}


def _route_with_default_margin(
    logits_ncqk: torch.Tensor,
    *,
    margin: float,
    active_mask_c: torch.Tensor,
    default_expert: int = DEFAULT_EXPERT,
    default_active_mask_c: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    default_active = active_mask_c if default_active_mask_c is None else default_active_mask_c
    route = logits_ncqk.argmax(dim=-1)
    chosen = logits_ncqk.gather(-1, route.unsqueeze(-1)).squeeze(-1)
    default_score = logits_ncqk[..., int(default_expert)]
    use_default = (route != int(default_expert)) & ((chosen - default_score) < float(margin))
    route = torch.where(use_default, torch.full_like(route, int(default_expert)), route)
    route = torch.where(active_mask_c.view(1, -1, 1), route, torch.full_like(route, int(default_expert)))
    route = torch.where(default_active.view(1, -1, 1), route, torch.zeros_like(route))
    return route


def apply_route_participation_guard(
    route_ncq: torch.Tensor,
    *,
    routed_mask_c: torch.Tensor,
    min_nondefault_rate: float,
    default_expert: int = DEFAULT_EXPERT,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    selected = route_ncq[:, routed_mask_c, :]
    nondefault_rate = float((selected != int(default_expert)).to(dtype=torch.float32).mean().item())
    threshold = max(0.0, float(min_nondefault_rate))
    abstain = bool(0.0 < nondefault_rate < threshold)
    if abstain:
        route_ncq = route_ncq.clone()
        route_ncq[:, routed_mask_c, :] = int(default_expert)
    return route_ncq, {
        "min_nondefault_rate": float(threshold),
        "observed_nondefault_rate": float(nondefault_rate),
        "abstained": bool(abstain),
    }


def _flatten_active(records: Sequence[Dict[str, torch.Tensor]], key: str, active_mask_c: torch.Tensor) -> torch.Tensor:
    return torch.cat([record[key][:, active_mask_c].reshape(-1, *record[key].shape[3:]) for record in records], dim=0)


def fixed_expert_class_weights(counts: torch.Tensor, mode: str) -> Optional[torch.Tensor]:
    normalized_mode = str(mode or "balanced").lower()
    if normalized_mode in {"none", "off", "unweighted"}:
        return None
    if normalized_mode != "balanced":
        raise ValueError("fixed-expert gate class_weight_mode must be balanced or none.")
    return (counts.sum() / (len(EXPERT_NAMES) * counts.clamp_min(1.0))).sqrt().clamp(0.5, 3.0)


def _select_margin(
    gate: FixedExpertPatchGate,
    records: Sequence[Dict[str, torch.Tensor]],
    *,
    mean_f: torch.Tensor,
    std_f: torch.Tensor,
    active_mask_c: torch.Tensor,
    default_active_mask_c: torch.Tensor,
    patch_len: int,
) -> Tuple[float, List[Dict[str, float]]]:
    if not records:
        raise ValueError("margin selection requires at least one calibration record.")
    prepared = []
    for record in records:
        features = ((record["features"] - mean_f) / std_f).clamp(-6.0, 6.0)
        with torch.no_grad():
            logits = _gate_logits(gate, features)
        default_route = torch.full(logits.shape[:-1], DEFAULT_EXPERT, dtype=torch.long)
        default_route[:, ~default_active_mask_c, :] = 0
        default_pred = apply_fixed_expert_routes(
            record["base"],
            record["corrections"],
            default_route,
            patch_len=int(patch_len),
        )
        prepared.append((record, logits, default_pred, _metrics(default_pred, record["y"])))
    default_metric = _metrics(
        torch.cat([item[2] for item in prepared], dim=0),
        torch.cat([item[0]["y"] for item in prepared], dim=0),
    )
    margins = torch.linspace(0.0, 4.0, 41).tolist() + [6.0, 100.0]
    rows: List[Dict[str, float]] = []
    best_margin = 100.0
    best_mse = default_metric["mse"]
    for margin in margins:
        pred_parts = []
        target_parts = []
        block_gains = []
        feasible_blocks = 0
        for record, logits, _, block_default_metric in prepared:
            route = _route_with_default_margin(
                logits,
                margin=float(margin),
                active_mask_c=active_mask_c,
                default_active_mask_c=default_active_mask_c,
            )
            pred = apply_fixed_expert_routes(
                record["base"],
                record["corrections"],
                route,
                patch_len=int(patch_len),
            )
            block_metric = _metrics(pred, record["y"])
            mse_gain = 100.0 * (block_default_metric["mse"] - block_metric["mse"]) / max(
                block_default_metric["mse"], 1.0e-12
            )
            mae_gain = 100.0 * (block_default_metric["mae"] - block_metric["mae"]) / max(
                block_default_metric["mae"], 1.0e-12
            )
            feasible_blocks += int(mse_gain >= 0.0 and mae_gain >= 0.0)
            block_gains.append({"mse_gain_pct": float(mse_gain), "mae_gain_pct": float(mae_gain)})
            pred_parts.append(pred)
            target_parts.append(record["y"])
        metric = _metrics(torch.cat(pred_parts, dim=0), torch.cat(target_parts, dim=0))
        feasible = (
            feasible_blocks == len(prepared)
            and metric["mse"] <= default_metric["mse"]
            and metric["mae"] <= default_metric["mae"]
        )
        rows.append(
            {
                "margin": float(margin),
                "mse": metric["mse"],
                "mae": metric["mae"],
                "mse_gain_pct_vs_default": 100.0
                * (default_metric["mse"] - metric["mse"])
                / max(default_metric["mse"], 1.0e-12),
                "mae_gain_pct_vs_default": 100.0
                * (default_metric["mae"] - metric["mae"])
                / max(default_metric["mae"], 1.0e-12),
                "feasible": float(feasible),
                "feasible_blocks": float(feasible_blocks),
                "calibration_blocks": float(len(prepared)),
                "min_block_mse_gain_pct": float(min(row["mse_gain_pct"] for row in block_gains)),
                "min_block_mae_gain_pct": float(min(row["mae_gain_pct"] for row in block_gains)),
            }
        )
        if feasible and metric["mse"] < best_mse:
            best_mse = metric["mse"]
            best_margin = float(margin)
    return best_margin, rows


def fit_fixed_expert_gate(
    train_records: Sequence[Dict[str, torch.Tensor]],
    calibration_records: Sequence[Dict[str, torch.Tensor]],
    *,
    active_mask_c: torch.Tensor,
    default_active_mask_c: torch.Tensor,
    patch_len: int,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    class_weight_mode: str = "balanced",
) -> Tuple[FixedExpertPatchGate, torch.Tensor, torch.Tensor, Dict[str, object]]:
    if not train_records or not calibration_records:
        raise ValueError("gate fitting requires non-empty train and calibration records.")
    torch.manual_seed(int(seed))
    train_features = _flatten_active(train_records, "features", active_mask_c)
    train_targets = _flatten_active(train_records, "targets", active_mask_c).reshape(-1).to(dtype=torch.long)
    train_gain = _flatten_active(train_records, "target_gain", active_mask_c).reshape(-1)
    feature_mean = train_features.mean(dim=0)
    feature_std = train_features.std(dim=0, unbiased=False).clamp_min(1.0e-5)
    train_z = ((train_features - feature_mean) / feature_std).clamp(-6.0, 6.0)
    gate = FixedExpertPatchGate(
        int(train_z.shape[-1]),
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
    )
    counts = torch.bincount(train_targets, minlength=len(EXPERT_NAMES)).to(dtype=torch.float32)
    class_weight = fixed_expert_class_weights(counts, str(class_weight_mode))
    optimizer = torch.optim.AdamW(gate.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    calibration_features = _flatten_active(calibration_records, "features", active_mask_c)
    calibration_features = ((calibration_features - feature_mean) / feature_std).clamp(-6.0, 6.0)
    calibration_targets = _flatten_active(calibration_records, "targets", active_mask_c).reshape(-1).to(dtype=torch.long)
    batch_size = 4096
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_calibration_loss = float("inf")
    best_epoch = 0
    for epoch in range(1, max(1, int(epochs)) + 1):
        gate.train()
        order = torch.randperm(int(train_z.shape[0]))
        for start in range(0, int(order.numel()), batch_size):
            idx = order[start : start + batch_size]
            logits = gate(train_z.index_select(0, idx))
            loss_rows = torch.nn.functional.cross_entropy(
                logits,
                train_targets.index_select(0, idx),
                weight=class_weight,
                reduction="none",
            )
            utility_weight = 1.0 + 2.0 * train_gain.index_select(0, idx).clamp(0.0, 1.0)
            loss = (loss_rows * utility_weight).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
            optimizer.step()
        gate.eval()
        with torch.no_grad():
            calibration_logits = gate(calibration_features)
            calibration_loss = float(
                torch.nn.functional.cross_entropy(
                    calibration_logits,
                    calibration_targets,
                    weight=class_weight,
                ).item()
            )
        if calibration_loss < best_calibration_loss:
            best_calibration_loss = calibration_loss
            best_epoch = int(epoch)
            best_state = {name: value.detach().clone() for name, value in gate.state_dict().items()}
    if best_state is not None:
        gate.load_state_dict(best_state)
    gate.eval()
    margin, margin_rows = _select_margin(
        gate,
        calibration_records,
        mean_f=feature_mean,
        std_f=feature_std,
        active_mask_c=active_mask_c,
        default_active_mask_c=default_active_mask_c,
        patch_len=int(patch_len),
    )
    summary = {
        "train_examples": int(train_targets.numel()),
        "calibration_blocks": int(len(calibration_records)),
        "best_epoch": int(best_epoch),
        "best_calibration_ce": float(best_calibration_loss),
        "class_counts": [int(value) for value in counts.tolist()],
        "class_weight_mode": str(class_weight_mode),
        "class_weights": None if class_weight is None else [float(value) for value in class_weight.tolist()],
        "decision_margin": float(margin),
        "margin_sweep": margin_rows,
    }
    return gate, feature_mean, feature_std, summary


def fit_fixed_expert_tree_gate(
    train_records: Sequence[Dict[str, torch.Tensor]],
    calibration_records: Sequence[Dict[str, torch.Tensor]],
    *,
    active_mask_c: torch.Tensor,
    default_active_mask_c: torch.Tensor,
    patch_len: int,
    seed: int,
) -> Tuple[SklearnProbabilityGate, torch.Tensor, torch.Tensor, Dict[str, object]]:
    from sklearn.ensemble import ExtraTreesClassifier

    if not train_records or not calibration_records:
        raise ValueError("tree gate fitting requires non-empty train and calibration records.")
    train_features = _flatten_active(train_records, "features", active_mask_c)
    train_targets = _flatten_active(train_records, "targets", active_mask_c).reshape(-1).to(dtype=torch.long)
    train_gain = _flatten_active(train_records, "target_gain", active_mask_c).reshape(-1)
    feature_mean = train_features.mean(dim=0)
    feature_std = train_features.std(dim=0, unbiased=False).clamp_min(1.0e-5)
    train_z = ((train_features - feature_mean) / feature_std).clamp(-6.0, 6.0)
    classifier = ExtraTreesClassifier(
        n_estimators=256,
        criterion="log_loss",
        max_depth=18,
        min_samples_leaf=12,
        max_features="sqrt",
        class_weight="balanced",
        random_state=int(seed),
        n_jobs=-1,
    )
    classifier.fit(
        train_z.numpy(),
        train_targets.numpy(),
        sample_weight=(1.0 + 2.0 * train_gain.clamp(0.0, 1.0)).numpy(),
    )
    gate = SklearnProbabilityGate(classifier)
    calibration_features = _flatten_active(calibration_records, "features", active_mask_c)
    calibration_features = ((calibration_features - feature_mean) / feature_std).clamp(-6.0, 6.0)
    calibration_targets = _flatten_active(calibration_records, "targets", active_mask_c).reshape(-1).to(dtype=torch.long)
    calibration_logits = gate(calibration_features)
    calibration_ce = float(torch.nn.functional.cross_entropy(calibration_logits, calibration_targets).item())
    margin, margin_rows = _select_margin(
        gate,
        calibration_records,
        mean_f=feature_mean,
        std_f=feature_std,
        active_mask_c=active_mask_c,
        default_active_mask_c=default_active_mask_c,
        patch_len=int(patch_len),
    )
    counts = torch.bincount(train_targets, minlength=len(EXPERT_NAMES))
    summary = {
        "kind": "extra_trees",
        "train_examples": int(train_targets.numel()),
        "calibration_blocks": int(len(calibration_records)),
        "best_epoch": 0,
        "best_calibration_ce": float(calibration_ce),
        "class_counts": [int(value) for value in counts.tolist()],
        "decision_margin": float(margin),
        "margin_sweep": margin_rows,
        "n_estimators": int(classifier.n_estimators),
        "max_depth": int(classifier.max_depth),
        "min_samples_leaf": int(classifier.min_samples_leaf),
    }
    return gate, feature_mean, feature_std, summary


def build_walk_forward_expert_records(
    tensors: Dict[str, torch.Tensor],
    *,
    blocks: int,
    warmup_blocks: int,
    label_delay: int,
    ridge: float,
    half_life: float,
    shrink: float,
    max_abs_scale: float,
    fit_loss: str,
    huber_delta: float,
    huber_iterations: int,
    active_channels: Sequence[int],
    domain_align_channels: Sequence[int],
    patch_len: int,
    label_min_gain: float,
    include_domain_descriptor: bool = True,
) -> List[Dict[str, torch.Tensor]]:
    x = tensors["x"]
    y = tensors["y"]
    base = tensors["base"]
    features, _, _ = prepare_input_correction_features(x, base, local_normalize=False)
    n = int(x.shape[0])
    c = int(x.shape[1])
    block_count = max(3, min(int(blocks), n))
    warmup = max(1, min(int(warmup_blocks), block_count - 1))
    active_mask = torch.zeros(c, dtype=torch.bool)
    active_mask[[int(channel) for channel in active_channels]] = True
    records: List[Dict[str, torch.Tensor]] = []
    for block_idx in range(warmup, block_count):
        start = (n * block_idx) // block_count
        end = (n * (block_idx + 1)) // block_count
        fit_end = max(1, start - int(label_delay) + 1) if int(label_delay) > 0 else start
        state = fit_weighted_ridge_residual(
            features[:fit_end],
            y[:fit_end] - base[:fit_end],
            ridge=float(ridge),
            half_life=float(half_life),
            loss=str(fit_loss),
            huber_delta=float(huber_delta),
            huber_iterations=int(huber_iterations),
        )
        raw = apply_weighted_ridge_residual(
            features[start:end],
            state,
            shrink=float(shrink),
            max_abs_scale=float(max_abs_scale),
            x_ncl=x[start:end],
        )
        aligned = apply_weighted_ridge_residual(
            features[start:end],
            state,
            shrink=float(shrink),
            max_abs_scale=float(max_abs_scale),
            x_ncl=x[start:end],
            domain_align_channels=[int(channel) for channel in domain_align_channels],
        )
        raw = raw * active_mask.view(1, -1, 1)
        aligned = aligned * active_mask.view(1, -1, 1)
        corrections = torch.stack([raw, aligned], dim=2)
        gate_features = build_fixed_expert_gate_features(
            x[start:end],
            base[start:end],
            corrections,
            patch_len=int(patch_len),
            include_domain_descriptor=bool(include_domain_descriptor),
        )
        targets, target_gain, mse, mae = fixed_expert_gate_targets(
            base[start:end],
            corrections,
            y[start:end],
            patch_len=int(patch_len),
            min_gain=float(label_min_gain),
        )
        targets[:, ~active_mask, :] = 0
        target_gain[:, ~active_mask, :] = 0.0
        records.append(
            {
                "block_index": torch.tensor(int(block_idx)),
                "start": torch.tensor(int(start)),
                "end": torch.tensor(int(end)),
                "fit_end": torch.tensor(int(fit_end)),
                "x": x[start:end],
                "base": base[start:end],
                "y": y[start:end],
                "corrections": corrections,
                "features": gate_features,
                "targets": targets,
                "target_gain": target_gain,
                "mse": mse,
                "mae": mae,
            }
        )
    return records


def _record_route_metrics(
    record: Dict[str, torch.Tensor],
    route: torch.Tensor,
    *,
    routed_mask_c: torch.Tensor,
    default_active_mask_c: torch.Tensor,
    patch_len: int,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    default_route = torch.full_like(route, DEFAULT_EXPERT)
    default_route[:, ~default_active_mask_c, :] = 0
    pred = apply_fixed_expert_routes(
        record["base"],
        record["corrections"],
        route,
        patch_len=int(patch_len),
    )
    default_pred = apply_fixed_expert_routes(
        record["base"],
        record["corrections"],
        default_route,
        patch_len=int(patch_len),
    )
    selected_metric = _metrics(pred, record["y"])
    default_metric = _metrics(default_pred, record["y"])
    per_channel = []
    for channel in range(int(pred.shape[1])):
        selected_channel = _metrics(pred[:, channel], record["y"][:, channel])
        default_channel = _metrics(default_pred[:, channel], record["y"][:, channel])
        per_channel.append(
            {
                "channel_index": int(channel),
                "default_mse": default_channel["mse"],
                "selected_mse": selected_channel["mse"],
                "mse_gain_pct_vs_default": 100.0
                * (default_channel["mse"] - selected_channel["mse"])
                / max(default_channel["mse"], 1.0e-12),
                "default_mae": default_channel["mae"],
                "selected_mae": selected_channel["mae"],
                "mae_gain_pct_vs_default": 100.0
                * (default_channel["mae"] - selected_channel["mae"])
                / max(default_channel["mae"], 1.0e-12),
            }
        )
    return pred, {
        "default_mse": default_metric["mse"],
        "selected_mse": selected_metric["mse"],
        "mse_gain_pct_vs_default": 100.0
        * (default_metric["mse"] - selected_metric["mse"])
        / max(default_metric["mse"], 1.0e-12),
        "default_mae": default_metric["mae"],
        "selected_mae": selected_metric["mae"],
        "mae_gain_pct_vs_default": 100.0
        * (default_metric["mae"] - selected_metric["mae"])
        / max(default_metric["mae"], 1.0e-12),
        "route_rates": _class_rates(route, routed_mask_c),
        "per_channel": per_channel,
    }


def run_moe_validation_diagnostic(
    records: Sequence[Dict[str, torch.Tensor]],
    *,
    active_channels: Sequence[int],
    routed_channels: Optional[Sequence[int]],
    patch_len: int,
    min_prior_records: int,
    calibration_blocks: int,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    gate_kind: str = "mlp",
    min_nondefault_rate: float = 0.0,
    class_weight_mode: str = "balanced",
) -> Dict[str, object]:
    calibration_count = max(1, int(calibration_blocks))
    if int(min_prior_records) <= calibration_count:
        raise ValueError("min_prior_records must exceed calibration_blocks.")
    if len(records) <= int(min_prior_records):
        raise ValueError("not enough walk-forward records to train and evaluate the gate.")
    c = int(records[0]["base"].shape[1])
    active_mask = torch.zeros(c, dtype=torch.bool)
    active_mask[[int(channel) for channel in active_channels]] = True
    routed = list(active_channels if routed_channels is None else routed_channels)
    routed_mask = torch.zeros(c, dtype=torch.bool)
    routed_mask[[int(channel) for channel in routed]] = True
    if bool((routed_mask & ~active_mask).any().item()):
        raise ValueError("routed channels must be a subset of active correction channels.")
    rows: List[Dict[str, object]] = []
    selected_parts: List[torch.Tensor] = []
    default_parts: List[torch.Tensor] = []
    oracle_parts: List[torch.Tensor] = []
    target_parts: List[torch.Tensor] = []
    for record_pos in range(int(min_prior_records), len(records)):
        prior = records[:record_pos]
        calibration = prior[-calibration_count:]
        train_records = prior[:-calibration_count]
        if str(gate_kind).lower() == "extra_trees":
            gate, feature_mean, feature_std, gate_summary = fit_fixed_expert_tree_gate(
                train_records,
                calibration,
                active_mask_c=active_mask,
                default_active_mask_c=active_mask,
                patch_len=int(patch_len),
                seed=int(seed) + record_pos,
            )
        else:
            gate, feature_mean, feature_std, gate_summary = fit_fixed_expert_gate(
                train_records,
                calibration,
                active_mask_c=active_mask,
                default_active_mask_c=active_mask,
                patch_len=int(patch_len),
                hidden_dim=int(hidden_dim),
                dropout=float(dropout),
                epochs=int(epochs),
                lr=float(lr),
                weight_decay=float(weight_decay),
                seed=int(seed) + record_pos,
                class_weight_mode=str(class_weight_mode),
            )
        record = records[record_pos]
        features = ((record["features"] - feature_mean) / feature_std).clamp(-6.0, 6.0)
        with torch.no_grad():
            logits = _gate_logits(gate, features)
        route = _route_with_default_margin(
            logits,
            margin=float(gate_summary["decision_margin"]),
            active_mask_c=routed_mask,
            default_active_mask_c=active_mask,
        )
        route, participation_guard = apply_route_participation_guard(
            route,
            routed_mask_c=routed_mask,
            min_nondefault_rate=float(min_nondefault_rate),
        )
        selected_pred, selected_summary = _record_route_metrics(
            record,
            route,
            routed_mask_c=routed_mask,
            default_active_mask_c=active_mask,
            patch_len=int(patch_len),
        )
        default_route = torch.full_like(route, DEFAULT_EXPERT)
        default_route[:, ~active_mask, :] = 0
        default_pred = apply_fixed_expert_routes(
            record["base"],
            record["corrections"],
            default_route,
            patch_len=int(patch_len),
        )
        best_route = record["mse"].argmin(dim=-1)
        oracle_route = torch.full_like(best_route, DEFAULT_EXPERT)
        oracle_route[:, routed_mask, :] = best_route[:, routed_mask, :]
        oracle_route[:, ~active_mask, :] = 0
        oracle_pred = apply_fixed_expert_routes(
            record["base"],
            record["corrections"],
            oracle_route,
            patch_len=int(patch_len),
        )
        oracle_metric = _metrics(oracle_pred, record["y"])
        default_metric = _metrics(default_pred, record["y"])
        labels = record["targets"][:, routed_mask].reshape(-1)
        route_active = route[:, routed_mask].reshape(-1)
        rows.append(
            {
                "block_index": int(record["block_index"].item()),
                "start_window": int(record["start"].item()),
                "end_window": int(record["end"].item()),
                **selected_summary,
                "oracle_mse": oracle_metric["mse"],
                "oracle_mae": oracle_metric["mae"],
                "oracle_mse_gain_pct_vs_default": 100.0
                * (default_metric["mse"] - oracle_metric["mse"])
                / max(default_metric["mse"], 1.0e-12),
                "oracle_mae_gain_pct_vs_default": 100.0
                * (default_metric["mae"] - oracle_metric["mae"])
                / max(default_metric["mae"], 1.0e-12),
                "target_rates": _class_rates(record["targets"], routed_mask),
                "oracle_rates": _class_rates(oracle_route, routed_mask),
                "route_accuracy": float((route_active == labels).to(dtype=torch.float32).mean().item()),
                "participation_guard": participation_guard,
                "gate": gate_summary,
            }
        )
        selected_parts.append(selected_pred)
        default_parts.append(default_pred)
        oracle_parts.append(oracle_pred)
        target_parts.append(record["y"])
    selected_all = torch.cat(selected_parts, dim=0)
    default_all = torch.cat(default_parts, dim=0)
    oracle_all = torch.cat(oracle_parts, dim=0)
    target_all = torch.cat(target_parts, dim=0)
    selected_metric = _metrics(selected_all, target_all)
    default_metric = _metrics(default_all, target_all)
    oracle_metric = _metrics(oracle_all, target_all)
    selected_gain = 100.0 * (default_metric["mse"] - selected_metric["mse"]) / max(default_metric["mse"], 1.0e-12)
    oracle_gain = 100.0 * (default_metric["mse"] - oracle_metric["mse"]) / max(default_metric["mse"], 1.0e-12)
    positive_mse_blocks = sum(float(row["mse_gain_pct_vs_default"]) >= 0.0 for row in rows)
    positive_mae_blocks = sum(float(row["mae_gain_pct_vs_default"]) >= 0.0 for row in rows)
    supported_rows = [
        row
        for row in rows
        if not bool(row["participation_guard"]["abstained"])
        and float(row["participation_guard"]["observed_nondefault_rate"]) > 0.0
    ]
    supported_capture = [
        100.0 * float(row["mse_gain_pct_vs_default"]) / max(float(row["oracle_mse_gain_pct_vs_default"]), 1.0e-12)
        for row in supported_rows
    ]
    supported_rate_gap = [
        abs(
            (1.0 - float(row["route_rates"][EXPERT_NAMES[DEFAULT_EXPERT]]))
            - (1.0 - float(row["target_rates"][EXPERT_NAMES[DEFAULT_EXPERT]]))
        )
        for row in supported_rows
    ]
    stable_metric_gate = bool(
        positive_mse_blocks == len(rows)
        and positive_mae_blocks == len(rows)
        and selected_metric["mse"] < default_metric["mse"]
        and selected_metric["mae"] <= default_metric["mae"]
    )
    test_gate_passed = bool(
        stable_metric_gate
        and len(supported_rows) >= 2
        and min(supported_capture, default=float("-inf")) >= 25.0
        and max(supported_rate_gap, default=float("inf")) <= 0.25
    )
    return {
        "expert_names": EXPERT_NAMES,
        "default_expert": EXPERT_NAMES[DEFAULT_EXPERT],
        "gate_train_channels": [int(channel) for channel in active_channels],
        "routed_channels": [int(channel) for channel in routed],
        "evaluated_blocks": len(rows),
        "positive_mse_blocks": int(positive_mse_blocks),
        "positive_mae_blocks": int(positive_mae_blocks),
        "default": default_metric,
        "selected": selected_metric,
        "oracle": oracle_metric,
        "mse_gain_pct_vs_default": float(selected_gain),
        "mae_gain_pct_vs_default": 100.0
        * (default_metric["mae"] - selected_metric["mae"])
        / max(default_metric["mae"], 1.0e-12),
        "oracle_mse_gain_pct_vs_default": float(oracle_gain),
        "oracle_mae_gain_pct_vs_default": 100.0
        * (default_metric["mae"] - oracle_metric["mae"])
        / max(default_metric["mae"], 1.0e-12),
        "captured_oracle_mse_headroom_pct": 100.0 * selected_gain / max(oracle_gain, 1.0e-12),
        "validation_gate_passed": stable_metric_gate,
        "supported_nonabstained_blocks": int(len(supported_rows)),
        "min_supported_oracle_capture_pct": float(min(supported_capture, default=0.0)),
        "max_supported_nondefault_rate_gap": float(max(supported_rate_gap, default=1.0)),
        "test_gate_passed": test_gate_passed,
        "blocks": rows,
    }


def _collect_split_tensors(args: argparse.Namespace, *, split: str) -> Dict[str, torch.Tensor]:
    split_name = str(split).lower()
    if split_name not in {"val", "test"}:
        raise ValueError("fixed-expert MoE collection split must be val or test.")
    cfg = load_yaml(str(args.config))
    cfg.setdefault("eval", {})["skip_test"] = True
    set_seed(int(cfg["exp"]["seed"]), deterministic=bool(cfg["exp"].get("deterministic", False)))
    requested = str(args.device or cfg["exp"].get("device", "cpu"))
    device = torch.device(requested if torch.cuda.is_available() and requested != "cpu" else "cpu")
    checkpoint = torch.load(str(args.checkpoint), map_location=device, weights_only=False)
    model, _, _, cluster_id_c, k, moe_cfg, _ = _build_modules(cfg, checkpoint, device)
    data_tc = _read_data_for_cfg(cfg)
    data_window_tc, loaders, eval_starts, train_loader, window_meta = _make_loaders_with_authorized_test(
        cfg,
        data_tc,
        batch_size=int(cfg["train"].get("batch_size", 64)),
        include_test=split_name == "test",
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
        num_clusters=int(k),
        num_channels=int(cluster_id_c.numel()),
        pred_len=int(window_meta["H"]),
        cfg=anchor_cfg,
    ).to(device)
    load_learnable_anchor_state_compat(
        learnable_anchor,
        checkpoint["learnable_output_anchor_state"],
    )
    return collect_anchor_predictions(
        model=model,
        loader=loaders[split_name],
        cluster_id_c=cluster_id_c,
        device=device,
        eval_start=int(eval_starts[split_name]),
        input_len=int(window_meta["L"]),
        moe_cfg=moe_cfg,
        observed_history_tc=data_window_tc,
        anchor_artifacts=anchor_artifacts,
        learnable_anchor=learnable_anchor,
    )


def evaluate_fixed_expert_moe_test(
    val_tensors: Dict[str, torch.Tensor],
    test_tensors: Dict[str, torch.Tensor],
    records: Sequence[Dict[str, torch.Tensor]],
    *,
    active_channels: Sequence[int],
    routed_channels: Sequence[int],
    domain_align_channels: Sequence[int],
    patch_len: int,
    ridge: float,
    half_life: float,
    shrink: float,
    max_abs_scale: float,
    fit_loss: str,
    huber_delta: float,
    huber_iterations: int,
    calibration_blocks: int,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    gate_kind: str,
    include_domain_descriptor: bool,
    min_nondefault_rate: float,
    class_weight_mode: str,
) -> Dict[str, object]:
    val_features, _, _ = prepare_input_correction_features(
        val_tensors["x"],
        val_tensors["base"],
        local_normalize=False,
    )
    test_features, _, _ = prepare_input_correction_features(
        test_tensors["x"],
        test_tensors["base"],
        local_normalize=False,
    )
    correction_state = fit_weighted_ridge_residual(
        val_features,
        val_tensors["y"] - val_tensors["base"],
        ridge=float(ridge),
        half_life=float(half_life),
        loss=str(fit_loss),
        huber_delta=float(huber_delta),
        huber_iterations=int(huber_iterations),
    )
    raw = apply_weighted_ridge_residual(
        test_features,
        correction_state,
        shrink=float(shrink),
        max_abs_scale=float(max_abs_scale),
        x_ncl=test_tensors["x"],
    )
    aligned = apply_weighted_ridge_residual(
        test_features,
        correction_state,
        shrink=float(shrink),
        max_abs_scale=float(max_abs_scale),
        x_ncl=test_tensors["x"],
        domain_align_channels=[int(channel) for channel in domain_align_channels],
    )
    c = int(test_tensors["base"].shape[1])
    active_mask = torch.zeros(c, dtype=torch.bool)
    active_mask[[int(channel) for channel in active_channels]] = True
    routed_mask = torch.zeros(c, dtype=torch.bool)
    routed_mask[[int(channel) for channel in routed_channels]] = True
    raw = raw * active_mask.view(1, -1, 1)
    aligned = aligned * active_mask.view(1, -1, 1)
    corrections = torch.stack([raw, aligned], dim=2)
    gate_features = build_fixed_expert_gate_features(
        test_tensors["x"],
        test_tensors["base"],
        corrections,
        patch_len=int(patch_len),
        include_domain_descriptor=bool(include_domain_descriptor),
    )
    calibration_count = max(1, int(calibration_blocks))
    if len(records) <= calibration_count:
        raise ValueError("not enough OOF records for final gate fitting.")
    gate_train_records = records[:-calibration_count]
    gate_calibration_records = records[-calibration_count:]
    if str(gate_kind).lower() == "extra_trees":
        gate, feature_mean, feature_std, gate_summary = fit_fixed_expert_tree_gate(
            gate_train_records,
            gate_calibration_records,
            active_mask_c=active_mask,
            default_active_mask_c=active_mask,
            patch_len=int(patch_len),
            seed=int(seed) + len(records),
        )
    else:
        gate, feature_mean, feature_std, gate_summary = fit_fixed_expert_gate(
            gate_train_records,
            gate_calibration_records,
            active_mask_c=active_mask,
            default_active_mask_c=active_mask,
            patch_len=int(patch_len),
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
            epochs=int(epochs),
            lr=float(lr),
            weight_decay=float(weight_decay),
            seed=int(seed) + len(records),
            class_weight_mode=str(class_weight_mode),
        )
    standardized = ((gate_features - feature_mean) / feature_std).clamp(-6.0, 6.0)
    with torch.no_grad():
        logits = _gate_logits(gate, standardized)
    route = _route_with_default_margin(
        logits,
        margin=float(gate_summary["decision_margin"]),
        active_mask_c=routed_mask,
        default_active_mask_c=active_mask,
    )
    route, participation_guard = apply_route_participation_guard(
        route,
        routed_mask_c=routed_mask,
        min_nondefault_rate=float(min_nondefault_rate),
    )
    selected_pred, metrics = _record_route_metrics(
        {
            "base": test_tensors["base"],
            "y": test_tensors["y"],
            "corrections": corrections,
        },
        route,
        routed_mask_c=routed_mask,
        default_active_mask_c=active_mask,
        patch_len=int(patch_len),
    )
    return {
        "fit_source": "validation_oof_gate_and_full_validation_experts",
        "fit_windows": int(val_tensors["x"].shape[0]),
        "test_windows": int(test_tensors["x"].shape[0]),
        "default": {"mse": float(metrics["default_mse"]), "mae": float(metrics["default_mae"])},
        "selected": {"mse": float(metrics["selected_mse"]), "mae": float(metrics["selected_mae"])},
        "mse_gain_pct_vs_default": float(metrics["mse_gain_pct_vs_default"]),
        "mae_gain_pct_vs_default": float(metrics["mae_gain_pct_vs_default"]),
        "route_rates": metrics["route_rates"],
        "participation_guard": participation_guard,
        "per_channel": metrics["per_channel"],
        "gate": gate_summary,
        "selected_delta_rms": float((selected_pred - test_tensors["base"]).square().mean().sqrt().item()),
    }


def run(args: argparse.Namespace) -> Dict[str, object]:
    tensors = _collect_split_tensors(args, split="val")
    records = build_walk_forward_expert_records(
        tensors,
        blocks=int(args.blocks),
        warmup_blocks=int(args.warmup_blocks),
        label_delay=int(args.label_delay),
        ridge=float(args.ridge),
        half_life=float(args.half_life),
        shrink=float(args.shrink),
        max_abs_scale=float(args.max_abs_scale),
        fit_loss=str(args.fit_loss),
        huber_delta=float(args.huber_delta),
        huber_iterations=int(args.huber_iterations),
        active_channels=[int(channel) for channel in args.channel_indices],
        domain_align_channels=[int(channel) for channel in args.domain_align_channel_indices],
        patch_len=int(args.patch_len),
        label_min_gain=float(args.label_min_gain),
        include_domain_descriptor=str(args.gate_feature_mode) == "within_plus_domain",
    )
    diagnostic = run_moe_validation_diagnostic(
        records,
        active_channels=[int(channel) for channel in args.channel_indices],
        routed_channels=None
        if args.routed_channel_indices is None
        else [int(channel) for channel in args.routed_channel_indices],
        patch_len=int(args.patch_len),
        min_prior_records=int(args.min_prior_records),
        calibration_blocks=int(args.calibration_blocks),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
        gate_kind=str(args.gate_kind),
        min_nondefault_rate=float(args.min_nondefault_rate),
        class_weight_mode=str(args.class_weight_mode),
    )
    allow_test_read = bool(args.allow_test_read)
    if allow_test_read and not bool(diagnostic["test_gate_passed"]):
        raise RuntimeError("Fixed-expert patch MoE validation gate failed; test remains unread.")
    test_payload = None
    if allow_test_read:
        test_tensors = _collect_split_tensors(args, split="test")
        routed_channels = (
            [int(channel) for channel in args.channel_indices]
            if args.routed_channel_indices is None
            else [int(channel) for channel in args.routed_channel_indices]
        )
        test_payload = evaluate_fixed_expert_moe_test(
            tensors,
            test_tensors,
            records,
            active_channels=[int(channel) for channel in args.channel_indices],
            routed_channels=routed_channels,
            domain_align_channels=[int(channel) for channel in args.domain_align_channel_indices],
            patch_len=int(args.patch_len),
            ridge=float(args.ridge),
            half_life=float(args.half_life),
            shrink=float(args.shrink),
            max_abs_scale=float(args.max_abs_scale),
            fit_loss=str(args.fit_loss),
            huber_delta=float(args.huber_delta),
            huber_iterations=int(args.huber_iterations),
            calibration_blocks=int(args.calibration_blocks),
            hidden_dim=int(args.hidden_dim),
            dropout=float(args.dropout),
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            seed=int(args.seed),
            gate_kind=str(args.gate_kind),
            include_domain_descriptor=str(args.gate_feature_mode) == "within_plus_domain",
            min_nondefault_rate=float(args.min_nondefault_rate),
            class_weight_mode=str(args.class_weight_mode),
        )
    payload = {
        "config_path": str(Path(args.config).resolve()),
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "test_read": allow_test_read,
        "config": {
            "blocks": int(args.blocks),
            "warmup_blocks": int(args.warmup_blocks),
            "label_delay": int(args.label_delay),
            "ridge": float(args.ridge),
            "half_life": float(args.half_life),
            "shrink": float(args.shrink),
            "max_abs_scale": float(args.max_abs_scale),
            "fit_loss": str(args.fit_loss),
            "huber_delta": float(args.huber_delta),
            "huber_iterations": int(args.huber_iterations),
            "channel_indices": [int(channel) for channel in args.channel_indices],
            "domain_align_channel_indices": [int(channel) for channel in args.domain_align_channel_indices],
            "routed_channel_indices": [int(channel) for channel in args.routed_channel_indices]
            if args.routed_channel_indices is not None
            else [int(channel) for channel in args.channel_indices],
            "patch_len": int(args.patch_len),
            "label_min_gain": float(args.label_min_gain),
            "gate_feature_mode": str(args.gate_feature_mode),
            "min_prior_records": int(args.min_prior_records),
            "calibration_blocks": int(args.calibration_blocks),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "gate_kind": str(args.gate_kind),
            "min_nondefault_rate": float(args.min_nondefault_rate),
            "class_weight_mode": str(args.class_weight_mode),
        },
        "diagnostic": diagnostic,
        "test": test_payload,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "fixed_expert_patch_moe.json"
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--warmup-blocks", type=int, default=2)
    parser.add_argument("--label-delay", type=int, default=96)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--half-life", type=float, default=672.0)
    parser.add_argument("--shrink", type=float, default=0.4)
    parser.add_argument("--max-abs-scale", type=float, default=0.15)
    parser.add_argument("--fit-loss", choices=["ridge", "huber_irls"], default="huber_irls")
    parser.add_argument("--huber-delta", type=float, default=0.1)
    parser.add_argument("--huber-iterations", type=int, default=5)
    parser.add_argument("--channel-indices", type=int, nargs="+", default=[0, 2, 3, 6])
    parser.add_argument("--domain-align-channel-indices", type=int, nargs="+", default=[0, 2, 6])
    parser.add_argument("--routed-channel-indices", type=int, nargs="+", default=None)
    parser.add_argument("--patch-len", type=int, default=24)
    parser.add_argument("--label-min-gain", type=float, default=0.01)
    parser.add_argument(
        "--gate-feature-mode",
        choices=["within_plus_domain", "within_domain"],
        default="within_plus_domain",
    )
    parser.add_argument("--min-prior-records", type=int, default=4)
    parser.add_argument("--calibration-blocks", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gate-kind", choices=["mlp", "extra_trees"], default="mlp")
    parser.add_argument("--min-nondefault-rate", type=float, default=0.0)
    parser.add_argument("--class-weight-mode", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--allow-test-read", action="store_true")
    parser.add_argument("--seed", type=int, default=20260710)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
