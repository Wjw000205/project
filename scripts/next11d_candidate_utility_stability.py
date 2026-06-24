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

from scripts.next11c_route_accuracy_diagnostic import _build_anchor_artifacts
from scripts.next11d_binary_adoption_forecast_eval import _oracle_channel_metrics
from scripts.next11d_binary_adoption_refit import _to_jsonable
from scripts.next11d_fixed_candidate_router_refit import _read_data_for_cfg
from scripts.shape_prior_diagnostic import _build_modules, _make_loaders
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
            raise ValueError("candidate utility stability diagnostic refuses to read test.")
        if split not in allowed:
            raise ValueError(f"unsupported split {raw!r}; expected train_fit, train_holdout, or val.")
        if split not in splits:
            splits.append(split)
    return splits or ["train_fit", "train_holdout", "val"]


def _candidate_gain_by_cluster_penalty(
    *,
    base_bch: torch.Tensor,
    cand_bcpH: torch.Tensor,
    y_bch: torch.Tensor,
    cluster_id_c: torch.Tensor,
    cluster_count: int,
) -> torch.Tensor:
    if base_bch.dim() != 3 or y_bch.dim() != 3:
        raise ValueError("base_bch and y_bch must have shape [B,C,H].")
    if cand_bcpH.dim() != 4:
        raise ValueError("cand_bcpH must have shape [B,C,P,H].")
    if tuple(base_bch.shape) != tuple(y_bch.shape):
        raise ValueError("base_bch and y_bch must share shape.")
    B, C, H = [int(v) for v in base_bch.shape]
    if tuple(cand_bcpH.shape[:2]) != (B, C) or int(cand_bcpH.shape[-1]) != H:
        raise ValueError("cand_bcpH must share [B,C,H] with base_bch.")
    cluster_id = cluster_id_c.detach().cpu().to(dtype=torch.long).view(-1)
    if int(cluster_id.numel()) != C:
        raise ValueError("cluster_id_c must have one id per channel.")
    K = int(cluster_count)
    if K <= 0:
        raise ValueError("cluster_count must be positive.")
    P = int(cand_bcpH.shape[2])
    base = base_bch.detach().cpu().to(dtype=torch.float32)
    cand = cand_bcpH.detach().cpu().to(dtype=torch.float32)
    y = y_bch.detach().cpu().to(dtype=torch.float32)
    base_err_bc = (base - y).pow(2).mean(dim=-1)
    cand_err_bcp = (cand - y.unsqueeze(2)).pow(2).mean(dim=-1)
    gain_bcp = base_err_bc.unsqueeze(2) - cand_err_bcp
    out = torch.full((B, K, P), float("nan"), dtype=torch.float32)
    for cluster in range(K):
        mask_c = cluster_id == int(cluster)
        if bool(mask_c.any().item()):
            out[:, cluster, :] = gain_bcp[:, mask_c, :].mean(dim=1)
    return out


def _candidate_gain_by_channel_penalty(
    *,
    base_bch: torch.Tensor,
    cand_bcpH: torch.Tensor,
    y_bch: torch.Tensor,
) -> torch.Tensor:
    if base_bch.dim() != 3 or y_bch.dim() != 3:
        raise ValueError("base_bch and y_bch must have shape [B,C,H].")
    if cand_bcpH.dim() != 4:
        raise ValueError("cand_bcpH must have shape [B,C,P,H].")
    if tuple(base_bch.shape) != tuple(y_bch.shape):
        raise ValueError("base_bch and y_bch must share shape.")
    B, C, H = [int(v) for v in base_bch.shape]
    if tuple(cand_bcpH.shape[:2]) != (B, C) or int(cand_bcpH.shape[-1]) != H:
        raise ValueError("cand_bcpH must share [B,C,H] with base_bch.")
    base = base_bch.detach().cpu().to(dtype=torch.float32)
    cand = cand_bcpH.detach().cpu().to(dtype=torch.float32)
    y = y_bch.detach().cpu().to(dtype=torch.float32)
    base_err_bc = (base - y).pow(2).mean(dim=-1)
    cand_err_bcp = (cand - y.unsqueeze(2)).pow(2).mean(dim=-1)
    return base_err_bc.unsqueeze(2) - cand_err_bcp


def _safe_mean(values: torch.Tensor) -> float:
    finite = values[torch.isfinite(values)]
    if int(finite.numel()) <= 0:
        return 0.0
    return float(finite.mean().item())


def _safe_quantile(values: torch.Tensor, q: float) -> float:
    finite = values[torch.isfinite(values)]
    if int(finite.numel()) <= 0:
        return 0.0
    return float(torch.quantile(finite, float(q)).item())


def _split_candidate_stats(
    *,
    gain_bkp: torch.Tensor,
    split: str,
    penalty_names: List[str],
) -> List[Dict[str, object]]:
    if gain_bkp.dim() != 3:
        raise ValueError("gain_bkp must have shape [B,K,P].")
    _, K, P = [int(v) for v in gain_bkp.shape]
    if int(len(penalty_names)) != P:
        raise ValueError("penalty_names length must match gain_bkp penalty dimension.")
    rows: List[Dict[str, object]] = []
    gain = gain_bkp.detach().cpu().to(dtype=torch.float32)
    for cluster in range(K):
        for penalty_idx, penalty in enumerate(penalty_names):
            values = gain[:, cluster, penalty_idx].reshape(-1)
            finite = values[torch.isfinite(values)]
            support = int(finite.numel())
            positive = finite > 0.0
            rows.append(
                {
                    "split": str(split),
                    "cluster": int(cluster),
                    "penalty_idx": int(penalty_idx),
                    "penalty": str(penalty),
                    "support": support,
                    "mean_gain": _safe_mean(finite),
                    "median_gain": _safe_quantile(finite, 0.50),
                    "p25_gain": _safe_quantile(finite, 0.25),
                    "p75_gain": _safe_quantile(finite, 0.75),
                    "positive_count": int(positive.sum().item()) if support > 0 else 0,
                    "positive_rate": float(positive.to(dtype=torch.float32).mean().item()) if support > 0 else 0.0,
                }
            )
    return rows


def _split_channel_candidate_stats(
    *,
    gain_bcp: torch.Tensor,
    split: str,
    penalty_names: List[str],
) -> List[Dict[str, object]]:
    if gain_bcp.dim() != 3:
        raise ValueError("gain_bcp must have shape [B,C,P].")
    _, C, P = [int(v) for v in gain_bcp.shape]
    if int(len(penalty_names)) != P:
        raise ValueError("penalty_names length must match gain_bcp penalty dimension.")
    rows: List[Dict[str, object]] = []
    gain = gain_bcp.detach().cpu().to(dtype=torch.float32)
    for channel in range(C):
        for penalty_idx, penalty in enumerate(penalty_names):
            values = gain[:, channel, penalty_idx].reshape(-1)
            finite = values[torch.isfinite(values)]
            support = int(finite.numel())
            positive = finite > 0.0
            rows.append(
                {
                    "split": str(split),
                    "channel": int(channel),
                    "penalty_idx": int(penalty_idx),
                    "penalty": str(penalty),
                    "support": support,
                    "mean_gain": _safe_mean(finite),
                    "median_gain": _safe_quantile(finite, 0.50),
                    "p25_gain": _safe_quantile(finite, 0.25),
                    "p75_gain": _safe_quantile(finite, 0.75),
                    "positive_count": int(positive.sum().item()) if support > 0 else 0,
                    "positive_rate": float(positive.to(dtype=torch.float32).mean().item()) if support > 0 else 0.0,
                }
            )
    return rows


def _row_key(row: Dict[str, object]) -> tuple[int, int]:
    return int(row["cluster"]), int(row["penalty_idx"])


def _channel_row_key(row: Dict[str, object]) -> tuple[int, int]:
    return int(row["channel"]), int(row["penalty_idx"])


def _candidate_stability_rows(
    *,
    fit_stats: List[Dict[str, object]],
    holdout_stats: List[Dict[str, object]],
    val_stats: Optional[List[Dict[str, object]]] = None,
    min_support: int,
    margin: float,
    positive_rate_threshold: float,
) -> List[Dict[str, object]]:
    holdout_by_key = {_row_key(row): row for row in holdout_stats}
    val_by_key = {_row_key(row): row for row in (val_stats or [])}
    rows: List[Dict[str, object]] = []
    for fit in fit_stats:
        key = _row_key(fit)
        holdout = holdout_by_key.get(key)
        if holdout is None:
            continue
        val = val_by_key.get(key)
        fit_support = int(fit.get("support", 0) or 0)
        holdout_support = int(holdout.get("support", 0) or 0)
        fit_mean = float(fit.get("mean_gain", 0.0) or 0.0)
        holdout_mean = float(holdout.get("mean_gain", 0.0) or 0.0)
        fit_pos = float(fit.get("positive_rate", 0.0) or 0.0)
        holdout_pos = float(holdout.get("positive_rate", 0.0) or 0.0)
        stable = bool(
            fit_support >= int(min_support)
            and holdout_support >= int(min_support)
            and fit_mean > float(margin)
            and holdout_mean > float(margin)
            and fit_pos >= float(positive_rate_threshold)
            and holdout_pos >= float(positive_rate_threshold)
        )
        row = {
            "cluster": int(fit["cluster"]),
            "penalty_idx": int(fit["penalty_idx"]),
            "penalty": str(fit["penalty"]),
            "stable_train_splits": stable,
            "fit_support": fit_support,
            "holdout_support": holdout_support,
            "fit_mean_gain": fit_mean,
            "holdout_mean_gain": holdout_mean,
            "fit_positive_rate": fit_pos,
            "holdout_positive_rate": holdout_pos,
            "fit_median_gain": float(fit.get("median_gain", 0.0) or 0.0),
            "holdout_median_gain": float(holdout.get("median_gain", 0.0) or 0.0),
        }
        if val is not None:
            val_mean = float(val.get("mean_gain", 0.0) or 0.0)
            row.update(
                {
                    "val_support": int(val.get("support", 0) or 0),
                    "val_mean_gain": val_mean,
                    "val_positive_rate": float(val.get("positive_rate", 0.0) or 0.0),
                    "val_median_gain": float(val.get("median_gain", 0.0) or 0.0),
                    "val_sign_agrees_with_train_stable": bool((not stable) or val_mean > float(margin)),
                }
            )
        rows.append(row)
    return rows


def _candidate_channel_stability_rows(
    *,
    fit_stats: List[Dict[str, object]],
    holdout_stats: List[Dict[str, object]],
    val_stats: Optional[List[Dict[str, object]]] = None,
    min_support: int,
    margin: float,
    positive_rate_threshold: float,
) -> List[Dict[str, object]]:
    holdout_by_key = {_channel_row_key(row): row for row in holdout_stats}
    val_by_key = {_channel_row_key(row): row for row in (val_stats or [])}
    rows: List[Dict[str, object]] = []
    for fit in fit_stats:
        key = _channel_row_key(fit)
        holdout = holdout_by_key.get(key)
        if holdout is None:
            continue
        val = val_by_key.get(key)
        fit_support = int(fit.get("support", 0) or 0)
        holdout_support = int(holdout.get("support", 0) or 0)
        fit_mean = float(fit.get("mean_gain", 0.0) or 0.0)
        holdout_mean = float(holdout.get("mean_gain", 0.0) or 0.0)
        fit_pos = float(fit.get("positive_rate", 0.0) or 0.0)
        holdout_pos = float(holdout.get("positive_rate", 0.0) or 0.0)
        stable = bool(
            fit_support >= int(min_support)
            and holdout_support >= int(min_support)
            and fit_mean > float(margin)
            and holdout_mean > float(margin)
            and fit_pos >= float(positive_rate_threshold)
            and holdout_pos >= float(positive_rate_threshold)
        )
        row = {
            "channel": int(fit["channel"]),
            "penalty_idx": int(fit["penalty_idx"]),
            "penalty": str(fit["penalty"]),
            "stable_train_splits": stable,
            "fit_support": fit_support,
            "holdout_support": holdout_support,
            "fit_mean_gain": fit_mean,
            "holdout_mean_gain": holdout_mean,
            "fit_positive_rate": fit_pos,
            "holdout_positive_rate": holdout_pos,
            "fit_median_gain": float(fit.get("median_gain", 0.0) or 0.0),
            "holdout_median_gain": float(holdout.get("median_gain", 0.0) or 0.0),
        }
        if val is not None:
            val_mean = float(val.get("mean_gain", 0.0) or 0.0)
            row.update(
                {
                    "val_support": int(val.get("support", 0) or 0),
                    "val_mean_gain": val_mean,
                    "val_positive_rate": float(val.get("positive_rate", 0.0) or 0.0),
                    "val_median_gain": float(val.get("median_gain", 0.0) or 0.0),
                    "val_sign_agrees_with_train_stable": bool((not stable) or val_mean > float(margin)),
                }
            )
        rows.append(row)
    return rows


def _base_and_oracle_metrics(tensors: Dict[str, torch.Tensor]) -> Dict[str, object]:
    oracle = _oracle_channel_metrics(tensors)
    return {
        "base_mse": oracle["base_mse"],
        "base_mae": oracle["base_mae"],
        "channel_oracle_mse": oracle["selected_mse"],
        "channel_oracle_mae": oracle["selected_mae"],
        "channel_oracle_gain_pct_vs_base": oracle["selected_gain_pct_vs_base"],
        "channel_oracle_mae_gain_pct_vs_base": oracle["selected_mae_gain_pct_vs_base"],
        "channel_oracle_candidate_use_rate": oracle["candidate_use_rate_channel"],
    }


def _static_channel_guard_metrics(
    *,
    base_bch: torch.Tensor,
    cand_bcpH: torch.Tensor,
    y_bch: torch.Tensor,
    channel_stability_rows: List[Dict[str, object]],
) -> Dict[str, object]:
    if base_bch.dim() != 3 or y_bch.dim() != 3:
        raise ValueError("base_bch and y_bch must have shape [B,C,H].")
    if cand_bcpH.dim() != 4:
        raise ValueError("cand_bcpH must have shape [B,C,P,H].")
    if tuple(base_bch.shape) != tuple(y_bch.shape):
        raise ValueError("base_bch and y_bch must share shape.")
    B, C, H = [int(v) for v in base_bch.shape]
    if tuple(cand_bcpH.shape[:2]) != (B, C) or int(cand_bcpH.shape[-1]) != H:
        raise ValueError("cand_bcpH must share [B,C,H] with base_bch.")
    base = base_bch.detach().cpu().to(dtype=torch.float32)
    cand = cand_bcpH.detach().cpu().to(dtype=torch.float32)
    y = y_bch.detach().cpu().to(dtype=torch.float32)
    selected_by_channel: Dict[int, Dict[str, object]] = {}
    for row in channel_stability_rows:
        if not bool(row.get("stable_train_splits", False)):
            continue
        channel = int(row["channel"])
        penalty_idx = int(row["penalty_idx"])
        if channel < 0 or channel >= C or penalty_idx < 0 or penalty_idx >= int(cand.shape[2]):
            continue
        prev = selected_by_channel.get(channel)
        if prev is None or float(row.get("holdout_mean_gain", 0.0) or 0.0) > float(
            prev.get("holdout_mean_gain", 0.0) or 0.0
        ):
            selected_by_channel[channel] = row
    selected = base.clone()
    for channel, row in selected_by_channel.items():
        selected[:, channel, :] = cand[:, channel, int(row["penalty_idx"]), :]
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
        "candidate_use_rate_channel": float(len(selected_by_channel) / max(C, 1)),
        "selected_channel_count": int(len(selected_by_channel)),
        "selected_channels": {
            str(channel): str(row.get("penalty", row.get("penalty_idx")))
            for channel, row in sorted(selected_by_channel.items())
        },
    }


def _classification(stability_rows: List[Dict[str, object]]) -> Dict[str, object]:
    stable_rows = [row for row in stability_rows if bool(row.get("stable_train_splits", False))]
    val_flips = [
        row
        for row in stable_rows
        if "val_sign_agrees_with_train_stable" in row and not bool(row.get("val_sign_agrees_with_train_stable", True))
    ]
    if not stable_rows:
        return {
            "failure_layer": "adapter candidate quality",
            "decision": "no_train_split_stable_candidate_utility",
            "stable_candidate_count": 0,
            "val_flip_count": 0,
        }
    if val_flips:
        return {
            "failure_layer": "train-val utility shift",
            "decision": "candidate_utility_train_stable_but_val_flips",
            "stable_candidate_count": int(len(stable_rows)),
            "val_flip_count": int(len(val_flips)),
        }
    return {
        "failure_layer": "selection/adoption policy",
        "decision": "candidate_utility_stable_router_is_next_blocker",
        "stable_candidate_count": int(len(stable_rows)),
        "val_flip_count": 0,
    }


def _channel_classification(stability_rows: List[Dict[str, object]]) -> Dict[str, object]:
    stable_rows = [row for row in stability_rows if bool(row.get("stable_train_splits", False))]
    val_flips = [
        row
        for row in stable_rows
        if "val_sign_agrees_with_train_stable" in row and not bool(row.get("val_sign_agrees_with_train_stable", True))
    ]
    if not stable_rows:
        return {
            "failure_layer": "adapter candidate quality",
            "decision": "no_channel_level_train_split_stable_candidate_utility",
            "stable_candidate_count": 0,
            "val_flip_count": 0,
        }
    if val_flips:
        return {
            "failure_layer": "train-val utility shift",
            "decision": "channel_candidate_utility_train_stable_but_val_flips",
            "stable_candidate_count": int(len(stable_rows)),
            "val_flip_count": int(len(val_flips)),
        }
    return {
        "failure_layer": "selection/adoption policy",
        "decision": "channel_candidate_utility_stable_router_or_channel_guard_is_next_blocker",
        "stable_candidate_count": int(len(stable_rows)),
        "val_flip_count": 0,
    }


def _markdown_report(payload: Dict[str, object]) -> str:
    lines = [
        "# NEXT-11d Candidate Utility Stability",
        "",
        f"- config: `{payload['config_path']}`",
        f"- checkpoint: `{payload['checkpoint_path']}`",
        f"- no_test_read: `{payload['no_test_read']}`",
        f"- failure_layer: `{payload['verdict']['failure_layer']}`",
        f"- decision: `{payload['verdict']['decision']}`",
        "",
        "## Split Metrics",
        "",
        "| split | base mse | base mae | channel oracle mse | channel oracle mae | oracle gain | oracle use |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, metrics in payload.get("split_metrics", {}).items():
        lines.append(
            "| {split} | {base_mse:.6f} | {base_mae:.6f} | {oracle_mse:.6f} | {oracle_mae:.6f} | {gain:.3f}% | {use:.4f} |".format(
                split=split,
                base_mse=float(metrics["base_mse"]),
                base_mae=float(metrics["base_mae"]),
                oracle_mse=float(metrics["channel_oracle_mse"]),
                oracle_mae=float(metrics["channel_oracle_mae"]),
                gain=float(metrics["channel_oracle_gain_pct_vs_base"]),
                use=float(metrics["channel_oracle_candidate_use_rate"]),
            )
        )
    if payload.get("static_channel_guard_metrics"):
        lines.extend(
            [
                "",
                "## Static Channel Guard Metrics",
                "",
                "| split | selected mse | selected mae | gain vs base | mae gain vs base | use rate | selected channels |",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for split, metrics in payload.get("static_channel_guard_metrics", {}).items():
            lines.append(
                "| {split} | {mse:.6f} | {mae:.6f} | {gain:.3f}% | {mae_gain:.3f}% | {use:.4f} | `{channels}` |".format(
                    split=split,
                    mse=float(metrics["selected_mse"]),
                    mae=float(metrics["selected_mae"]),
                    gain=float(metrics["selected_gain_pct_vs_base"]),
                    mae_gain=float(metrics["selected_mae_gain_pct_vs_base"]),
                    use=float(metrics["candidate_use_rate_channel"]),
                    channels=metrics["selected_channels"],
                )
            )
    lines.extend(
        [
            "",
            "## Cluster Candidate Stability",
            "",
            "| cluster | penalty | stable | fit mean | holdout mean | val mean | fit pos | holdout pos | val pos |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload.get("stability_rows", []):
        lines.append(
            "| {cluster} | {penalty} | {stable} | {fit_mean:.6g} | {holdout_mean:.6g} | {val_mean:.6g} | {fit_pos:.4f} | {holdout_pos:.4f} | {val_pos:.4f} |".format(
                cluster=int(row["cluster"]),
                penalty=row["penalty"],
                stable=str(bool(row["stable_train_splits"])),
                fit_mean=float(row["fit_mean_gain"]),
                holdout_mean=float(row["holdout_mean_gain"]),
                val_mean=float(row.get("val_mean_gain", 0.0)),
                fit_pos=float(row["fit_positive_rate"]),
                holdout_pos=float(row["holdout_positive_rate"]),
                val_pos=float(row.get("val_positive_rate", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Channel Candidate Stability",
            "",
            "| channel | penalty | stable | fit mean | holdout mean | val mean | fit pos | holdout pos | val pos |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload.get("channel_stability_rows", []):
        if not bool(row.get("stable_train_splits", False)) and not bool(payload.get("include_all_channel_rows", False)):
            continue
        lines.append(
            "| {channel} | {penalty} | {stable} | {fit_mean:.6g} | {holdout_mean:.6g} | {val_mean:.6g} | {fit_pos:.4f} | {holdout_pos:.4f} | {val_pos:.4f} |".format(
                channel=int(row["channel"]),
                penalty=row["penalty"],
                stable=str(bool(row["stable_train_splits"])),
                fit_mean=float(row["fit_mean_gain"]),
                holdout_mean=float(row["holdout_mean_gain"]),
                val_mean=float(row.get("val_mean_gain", 0.0)),
                fit_pos=float(row["fit_positive_rate"]),
                holdout_pos=float(row["holdout_positive_rate"]),
                val_pos=float(row.get("val_positive_rate", 0.0)),
            )
        )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> Dict[str, object]:
    cfg = load_yaml(str(args.config))
    cfg.setdefault("eval", {})["skip_test"] = True
    splits = _normalize_requested_splits(args.splits)
    set_seed(int(cfg["exp"]["seed"]), deterministic=bool((cfg.get("exp", {}) or {}).get("deterministic", False)))
    requested_device = str(args.device or cfg.get("exp", {}).get("device", "cpu"))
    device = torch.device(requested_device if torch.cuda.is_available() and requested_device != "cpu" else "cpu")
    checkpoint = torch.load(str(args.checkpoint), map_location=device)
    model, _, pred_residual, cluster_id_c, K, moe_cfg, penalty_names = _build_modules(cfg, checkpoint, device)
    data_tc = _read_data_for_cfg(cfg)
    batch_size = int(cfg.get("train", {}).get("batch_size", 64))
    data_window_tc, loaders, eval_starts, train_loader, window_meta = _make_loaders(cfg, data_tc, batch_size=batch_size)
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
    split_stats: Dict[str, List[Dict[str, object]]] = {}
    split_channel_stats: Dict[str, List[Dict[str, object]]] = {}
    split_metrics: Dict[str, Dict[str, object]] = {}
    tensors_by_split: Dict[str, Dict[str, torch.Tensor]] = {}
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
        tensors_by_split[split] = tensors
        gain_bkp = _candidate_gain_by_cluster_penalty(
            base_bch=tensors["base"],
            cand_bcpH=tensors["cand"],
            y_bch=tensors["y"],
            cluster_id_c=cluster_id_c.detach().cpu(),
            cluster_count=int(K),
        )
        gain_bcp = _candidate_gain_by_channel_penalty(
            base_bch=tensors["base"],
            cand_bcpH=tensors["cand"],
            y_bch=tensors["y"],
        )
        split_stats[split] = _split_candidate_stats(
            gain_bkp=gain_bkp,
            split=split,
            penalty_names=list(penalty_names),
        )
        split_channel_stats[split] = _split_channel_candidate_stats(
            gain_bcp=gain_bcp,
            split=split,
            penalty_names=list(penalty_names),
        )
        split_metrics[split] = _base_and_oracle_metrics(tensors)
    if "train_fit" not in split_stats or "train_holdout" not in split_stats:
        raise ValueError("candidate stability requires train_fit and train_holdout splits.")
    stability_rows = _candidate_stability_rows(
        fit_stats=split_stats["train_fit"],
        holdout_stats=split_stats["train_holdout"],
        val_stats=split_stats.get("val"),
        min_support=int(args.min_support),
        margin=float(args.margin),
        positive_rate_threshold=float(args.positive_rate_threshold),
    )
    channel_stability_rows = _candidate_channel_stability_rows(
        fit_stats=split_channel_stats["train_fit"],
        holdout_stats=split_channel_stats["train_holdout"],
        val_stats=split_channel_stats.get("val"),
        min_support=int(args.min_support),
        margin=float(args.margin),
        positive_rate_threshold=float(args.positive_rate_threshold),
    )
    static_channel_guard_metrics = {
        split: _static_channel_guard_metrics(
            base_bch=tensors["base"],
            cand_bcpH=tensors["cand"],
            y_bch=tensors["y"],
            channel_stability_rows=channel_stability_rows,
        )
        for split, tensors in tensors_by_split.items()
    }
    payload = {
        "config_path": str(args.config),
        "checkpoint_path": str(args.checkpoint),
        "out_dir": str(args.out_dir),
        "device": str(device),
        "no_test_read": True,
        "splits_collected": splits,
        "penalty_names": list(penalty_names),
        "cluster_count": int(K),
        "criteria": {
            "min_support": int(args.min_support),
            "margin": float(args.margin),
            "positive_rate_threshold": float(args.positive_rate_threshold),
        },
        "split_metrics": split_metrics,
        "split_candidate_stats": split_stats,
        "split_channel_candidate_stats": split_channel_stats,
        "stability_rows": stability_rows,
        "channel_stability_rows": channel_stability_rows,
        "static_channel_guard_metrics": static_channel_guard_metrics,
        "include_all_channel_rows": bool(args.include_all_channel_rows),
        "verdict": _classification(stability_rows),
        "channel_verdict": _channel_classification(channel_stability_rows),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "candidate_utility_stability.json").write_text(
        json.dumps(_to_jsonable(payload), indent=2),
        encoding="utf-8",
    )
    (out_dir / "candidate_utility_stability.md").write_text(_markdown_report(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="NEXT-11d candidate utility stability diagnostic.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--splits", nargs="*", default=["train_fit", "train_holdout", "val"])
    parser.add_argument("--candidate-feature-mode", type=str, default="shape_proxy")
    parser.add_argument("--min-support", type=int, default=64)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--positive-rate-threshold", type=float, default=0.52)
    parser.add_argument("--include-all-channel-rows", action="store_true")
    args = parser.parse_args()
    payload = run(args)
    verdict = payload["verdict"]
    channel_verdict = payload["channel_verdict"]
    print(
        "stable_candidates={} stable_channel_candidates={} failure_layer={} decision={} no_test_read=True".format(
            int(verdict["stable_candidate_count"]),
            int(channel_verdict["stable_candidate_count"]),
            str(verdict["failure_layer"]),
            str(verdict["decision"]),
        )
    )


if __name__ == "__main__":
    main()
