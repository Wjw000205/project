from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.next11c_route_accuracy_diagnostic import _build_anchor_artifacts
from scripts.next11d_binary_adoption_forecast_eval import _oracle_channel_metrics
from scripts.next11d_binary_adoption_refit import _to_jsonable
from scripts.next11d_candidate_utility_stability import (
    _candidate_gain_by_channel_penalty,
    _candidate_gain_by_cluster_penalty,
    _normalize_requested_splits,
)
from scripts.next11d_fixed_candidate_router_refit import _read_data_for_cfg
from scripts.shape_prior_diagnostic import _build_modules, _make_loaders
from src.train import _collect_pred_residual_selector_tensors
from src.utils.seed import set_seed
from src.utils.yaml_io import load_yaml


def _segment_slices(sample_count: int, segments: int) -> List[Tuple[int, int]]:
    if int(segments) <= 0:
        raise ValueError("segments must be positive.")
    n = int(sample_count)
    if n <= 0:
        return []
    active_segments = min(int(segments), n)
    base = n // active_segments
    remainder = n % active_segments
    out: List[Tuple[int, int]] = []
    start = 0
    for segment in range(active_segments):
        width = base + (1 if segment < remainder else 0)
        end = start + width
        out.append((start, end))
        start = end
    return out


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


def _segment_candidate_stats(
    *,
    gain_bep: torch.Tensor,
    split: str,
    entity_name: str,
    penalty_names: List[str],
    segments: int,
) -> List[Dict[str, object]]:
    if entity_name not in {"cluster", "channel"}:
        raise ValueError("entity_name must be 'cluster' or 'channel'.")
    if gain_bep.dim() != 3:
        raise ValueError("gain_bep must have shape [B,E,P].")
    B, E, P = [int(v) for v in gain_bep.shape]
    if int(len(penalty_names)) != P:
        raise ValueError("penalty_names length must match gain_bep penalty dimension.")
    gain = gain_bep.detach().cpu().to(dtype=torch.float32)
    rows: List[Dict[str, object]] = []
    slices = _segment_slices(B, int(segments))
    for segment_idx, (start, end) in enumerate(slices):
        segment_gain = gain[start:end]
        for entity_idx in range(E):
            for penalty_idx, penalty in enumerate(penalty_names):
                values = segment_gain[:, entity_idx, penalty_idx].reshape(-1)
                finite = values[torch.isfinite(values)]
                support = int(finite.numel())
                positive = finite > 0.0
                row = {
                    "split": str(split),
                    "segment": int(segment_idx),
                    "segment_start": int(start),
                    "segment_end": int(end),
                    "segment_support_samples": int(end - start),
                    entity_name: int(entity_idx),
                    "penalty_idx": int(penalty_idx),
                    "penalty": str(penalty),
                    "support": support,
                    "mean_gain": _safe_mean(finite),
                    "median_gain": _safe_quantile(finite, 0.50),
                    "p25_gain": _safe_quantile(finite, 0.25),
                    "p75_gain": _safe_quantile(finite, 0.75),
                    "positive_count": int(positive.sum().item()) if support > 0 else 0,
                    "positive_rate": float(positive.to(dtype=torch.float32).mean().item())
                    if support > 0
                    else 0.0,
                }
                rows.append(row)
    return rows


def _entity_key(row: Dict[str, object], entity_name: str) -> tuple[int, int]:
    return int(row[entity_name]), int(row["penalty_idx"])


def _group_segment_rows(
    rows: Iterable[Dict[str, object]],
    *,
    entity_name: str,
) -> Dict[tuple[int, int], List[Dict[str, object]]]:
    grouped: Dict[tuple[int, int], List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(_entity_key(row, entity_name), []).append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(row["segment"]))
    return grouped


def _summarize_segments(
    rows: List[Dict[str, object]],
    *,
    min_segment_support: int,
    margin: float,
    positive_rate_threshold: float,
) -> Dict[str, object]:
    valid = [row for row in rows if int(row.get("support", 0) or 0) >= int(min_segment_support)]
    positive_rows = [
        row
        for row in valid
        if float(row.get("mean_gain", 0.0) or 0.0) > float(margin)
        and float(row.get("positive_rate", 0.0) or 0.0) >= float(positive_rate_threshold)
    ]
    nonpositive_rows = [row for row in valid if float(row.get("mean_gain", 0.0) or 0.0) <= float(margin)]
    means = torch.tensor([float(row.get("mean_gain", 0.0) or 0.0) for row in valid], dtype=torch.float32)
    positive_rates = torch.tensor(
        [float(row.get("positive_rate", 0.0) or 0.0) for row in valid],
        dtype=torch.float32,
    )
    total_segments = int(len(rows))
    supported_segments = int(len(valid))
    positive_segments = int(len(positive_rows))
    return {
        "total_segments": total_segments,
        "supported_segments": supported_segments,
        "positive_segments": positive_segments,
        "nonpositive_segments": int(len(nonpositive_rows)),
        "all_segments_supported": bool(total_segments > 0 and supported_segments == total_segments),
        "all_segments_positive": bool(total_segments > 0 and positive_segments == total_segments),
        "sign_flip": bool(positive_segments > 0 and int(len(nonpositive_rows)) > 0),
        "mean_gain": _safe_mean(means),
        "min_segment_mean_gain": float(means.min().item()) if int(means.numel()) else 0.0,
        "max_segment_mean_gain": float(means.max().item()) if int(means.numel()) else 0.0,
        "min_positive_rate": float(positive_rates.min().item()) if int(positive_rates.numel()) else 0.0,
        "max_positive_rate": float(positive_rates.max().item()) if int(positive_rates.numel()) else 0.0,
    }


def _temporal_stability_rows(
    *,
    fit_segment_stats: List[Dict[str, object]],
    holdout_segment_stats: List[Dict[str, object]],
    val_segment_stats: Optional[List[Dict[str, object]]],
    entity_name: str,
    min_segment_support: int,
    margin: float,
    positive_rate_threshold: float,
) -> List[Dict[str, object]]:
    if entity_name not in {"cluster", "channel"}:
        raise ValueError("entity_name must be 'cluster' or 'channel'.")
    holdout_by_key = _group_segment_rows(holdout_segment_stats, entity_name=entity_name)
    val_by_key = _group_segment_rows(val_segment_stats or [], entity_name=entity_name)
    rows: List[Dict[str, object]] = []
    for key, fit_rows in _group_segment_rows(fit_segment_stats, entity_name=entity_name).items():
        holdout_rows = holdout_by_key.get(key)
        if holdout_rows is None:
            continue
        val_rows = val_by_key.get(key, [])
        fit_summary = _summarize_segments(
            fit_rows,
            min_segment_support=int(min_segment_support),
            margin=float(margin),
            positive_rate_threshold=float(positive_rate_threshold),
        )
        holdout_summary = _summarize_segments(
            holdout_rows,
            min_segment_support=int(min_segment_support),
            margin=float(margin),
            positive_rate_threshold=float(positive_rate_threshold),
        )
        val_summary = _summarize_segments(
            val_rows,
            min_segment_support=int(min_segment_support),
            margin=float(margin),
            positive_rate_threshold=float(positive_rate_threshold),
        )
        stable_train = bool(
            fit_summary["all_segments_supported"]
            and holdout_summary["all_segments_supported"]
            and fit_summary["all_segments_positive"]
            and holdout_summary["all_segments_positive"]
        )
        has_val = bool(val_rows)
        val_agrees = bool((not stable_train) or (has_val and val_summary["all_segments_positive"]))
        entity_idx, penalty_idx = key
        first = fit_rows[0]
        row = {
            entity_name: int(entity_idx),
            "penalty_idx": int(penalty_idx),
            "penalty": str(first["penalty"]),
            "stable_train_segments": stable_train,
            "val_segment_sign_agrees": val_agrees,
            "fit_segment_count": int(fit_summary["total_segments"]),
            "holdout_segment_count": int(holdout_summary["total_segments"]),
            "val_segment_count": int(val_summary["total_segments"]),
            "fit_supported_segments": int(fit_summary["supported_segments"]),
            "holdout_supported_segments": int(holdout_summary["supported_segments"]),
            "val_supported_segments": int(val_summary["supported_segments"]),
            "fit_positive_segments": int(fit_summary["positive_segments"]),
            "holdout_positive_segments": int(holdout_summary["positive_segments"]),
            "val_positive_segments": int(val_summary["positive_segments"]),
            "fit_mean_gain": float(fit_summary["mean_gain"]),
            "holdout_mean_gain": float(holdout_summary["mean_gain"]),
            "val_mean_gain": float(val_summary["mean_gain"]),
            "fit_min_segment_mean_gain": float(fit_summary["min_segment_mean_gain"]),
            "holdout_min_segment_mean_gain": float(holdout_summary["min_segment_mean_gain"]),
            "val_min_segment_mean_gain": float(val_summary["min_segment_mean_gain"]),
            "fit_max_segment_mean_gain": float(fit_summary["max_segment_mean_gain"]),
            "holdout_max_segment_mean_gain": float(holdout_summary["max_segment_mean_gain"]),
            "val_max_segment_mean_gain": float(val_summary["max_segment_mean_gain"]),
            "fit_min_positive_rate": float(fit_summary["min_positive_rate"]),
            "holdout_min_positive_rate": float(holdout_summary["min_positive_rate"]),
            "val_min_positive_rate": float(val_summary["min_positive_rate"]),
            "fit_sign_flip": bool(fit_summary["sign_flip"]),
            "holdout_sign_flip": bool(holdout_summary["sign_flip"]),
            "val_sign_flip": bool(val_summary["sign_flip"]),
        }
        rows.append(row)
    rows.sort(
        key=lambda row: (
            bool(row.get("stable_train_segments", False)),
            float(row.get("holdout_mean_gain", 0.0) or 0.0),
            float(row.get("fit_mean_gain", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return rows


def _classify_temporal_stability(rows: List[Dict[str, object]]) -> Dict[str, object]:
    stable_rows = [row for row in rows if bool(row.get("stable_train_segments", False))]
    val_flips = [row for row in stable_rows if not bool(row.get("val_segment_sign_agrees", True))]
    fit_flips = [row for row in rows if bool(row.get("fit_sign_flip", False))]
    holdout_flips = [row for row in rows if bool(row.get("holdout_sign_flip", False))]
    if not stable_rows:
        return {
            "failure_layer": "adapter candidate quality",
            "decision": "no_temporally_stable_train_candidate_utility",
            "stable_candidate_count": 0,
            "val_flip_count": 0,
            "fit_sign_flip_count": int(len(fit_flips)),
            "holdout_sign_flip_count": int(len(holdout_flips)),
        }
    if val_flips:
        return {
            "failure_layer": "train-val utility shift",
            "decision": "temporal_candidate_train_stable_but_val_flips",
            "stable_candidate_count": int(len(stable_rows)),
            "val_flip_count": int(len(val_flips)),
            "fit_sign_flip_count": int(len(fit_flips)),
            "holdout_sign_flip_count": int(len(holdout_flips)),
        }
    return {
        "failure_layer": "selection/adoption policy",
        "decision": "temporal_candidate_train_and_val_stable_router_is_next_blocker",
        "stable_candidate_count": int(len(stable_rows)),
        "val_flip_count": 0,
        "fit_sign_flip_count": int(len(fit_flips)),
        "holdout_sign_flip_count": int(len(holdout_flips)),
    }


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


def _markdown_report(payload: Dict[str, object]) -> str:
    lines = [
        "# NEXT-11d Temporal Candidate Utility Stability",
        "",
        f"- config: `{payload['config_path']}`",
        f"- checkpoint: `{payload['checkpoint_path']}`",
        f"- no_test_read: `{payload['no_test_read']}`",
        f"- cluster_failure_layer: `{payload['verdict']['failure_layer']}`",
        f"- cluster_decision: `{payload['verdict']['decision']}`",
        f"- channel_failure_layer: `{payload['channel_verdict']['failure_layer']}`",
        f"- channel_decision: `{payload['channel_verdict']['decision']}`",
        "",
        "## Split Metrics",
        "",
        "| split | base mse | base mae | channel oracle mse | channel oracle mae | oracle mse gain | oracle use |",
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
    lines.extend(
        [
            "",
            "## Cluster Temporal Stability",
            "",
            "| cluster | penalty | stable train | val agrees | fit pos seg | holdout pos seg | val pos seg | fit mean | holdout mean | val mean | fit min | holdout min | val min |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload.get("temporal_stability_rows", []):
        lines.append(
            "| {entity} | {penalty} | {stable} | {val_agrees} | {fit_pos}/{fit_total} | {holdout_pos}/{holdout_total} | {val_pos}/{val_total} | {fit_mean:.6g} | {holdout_mean:.6g} | {val_mean:.6g} | {fit_min:.6g} | {holdout_min:.6g} | {val_min:.6g} |".format(
                entity=int(row["cluster"]),
                penalty=str(row["penalty"]),
                stable=str(bool(row["stable_train_segments"])),
                val_agrees=str(bool(row["val_segment_sign_agrees"])),
                fit_pos=int(row["fit_positive_segments"]),
                fit_total=int(row["fit_segment_count"]),
                holdout_pos=int(row["holdout_positive_segments"]),
                holdout_total=int(row["holdout_segment_count"]),
                val_pos=int(row["val_positive_segments"]),
                val_total=int(row["val_segment_count"]),
                fit_mean=float(row["fit_mean_gain"]),
                holdout_mean=float(row["holdout_mean_gain"]),
                val_mean=float(row["val_mean_gain"]),
                fit_min=float(row["fit_min_segment_mean_gain"]),
                holdout_min=float(row["holdout_min_segment_mean_gain"]),
                val_min=float(row["val_min_segment_mean_gain"]),
            )
        )
    lines.extend(
        [
            "",
            "## Channel Temporal Stability",
            "",
            "| channel | penalty | stable train | val agrees | fit pos seg | holdout pos seg | val pos seg | fit mean | holdout mean | val mean | fit min | holdout min | val min |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    channel_rows = payload.get("channel_temporal_stability_rows", [])
    for row in channel_rows:
        if not bool(row.get("stable_train_segments", False)) and not bool(payload.get("include_all_channel_rows", False)):
            continue
        lines.append(
            "| {entity} | {penalty} | {stable} | {val_agrees} | {fit_pos}/{fit_total} | {holdout_pos}/{holdout_total} | {val_pos}/{val_total} | {fit_mean:.6g} | {holdout_mean:.6g} | {val_mean:.6g} | {fit_min:.6g} | {holdout_min:.6g} | {val_min:.6g} |".format(
                entity=int(row["channel"]),
                penalty=str(row["penalty"]),
                stable=str(bool(row["stable_train_segments"])),
                val_agrees=str(bool(row["val_segment_sign_agrees"])),
                fit_pos=int(row["fit_positive_segments"]),
                fit_total=int(row["fit_segment_count"]),
                holdout_pos=int(row["holdout_positive_segments"]),
                holdout_total=int(row["holdout_segment_count"]),
                val_pos=int(row["val_positive_segments"]),
                val_total=int(row["val_segment_count"]),
                fit_mean=float(row["fit_mean_gain"]),
                holdout_mean=float(row["holdout_mean_gain"]),
                val_mean=float(row["val_mean_gain"]),
                fit_min=float(row["fit_min_segment_mean_gain"]),
                holdout_min=float(row["holdout_min_segment_mean_gain"]),
                val_min=float(row["val_min_segment_mean_gain"]),
            )
        )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> Dict[str, object]:
    cfg = load_yaml(str(args.config))
    cfg.setdefault("eval", {})["skip_test"] = True
    splits = _normalize_requested_splits(args.splits)
    if "train_fit" not in splits or "train_holdout" not in splits:
        raise ValueError("temporal candidate stability requires train_fit and train_holdout splits.")
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
    split_metrics: Dict[str, Dict[str, object]] = {}
    cluster_segment_stats: Dict[str, List[Dict[str, object]]] = {}
    channel_segment_stats: Dict[str, List[Dict[str, object]]] = {}
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
        split_metrics[split] = _base_and_oracle_metrics(tensors)
        cluster_segment_stats[split] = _segment_candidate_stats(
            gain_bep=gain_bkp,
            split=split,
            entity_name="cluster",
            penalty_names=list(penalty_names),
            segments=int(args.segments),
        )
        channel_segment_stats[split] = _segment_candidate_stats(
            gain_bep=gain_bcp,
            split=split,
            entity_name="channel",
            penalty_names=list(penalty_names),
            segments=int(args.segments),
        )
    temporal_rows = _temporal_stability_rows(
        fit_segment_stats=cluster_segment_stats["train_fit"],
        holdout_segment_stats=cluster_segment_stats["train_holdout"],
        val_segment_stats=cluster_segment_stats.get("val"),
        entity_name="cluster",
        min_segment_support=int(args.min_segment_support),
        margin=float(args.margin),
        positive_rate_threshold=float(args.positive_rate_threshold),
    )
    channel_temporal_rows = _temporal_stability_rows(
        fit_segment_stats=channel_segment_stats["train_fit"],
        holdout_segment_stats=channel_segment_stats["train_holdout"],
        val_segment_stats=channel_segment_stats.get("val"),
        entity_name="channel",
        min_segment_support=int(args.min_segment_support),
        margin=float(args.margin),
        positive_rate_threshold=float(args.positive_rate_threshold),
    )
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
            "segments": int(args.segments),
            "min_segment_support": int(args.min_segment_support),
            "margin": float(args.margin),
            "positive_rate_threshold": float(args.positive_rate_threshold),
        },
        "split_metrics": split_metrics,
        "cluster_segment_stats": cluster_segment_stats,
        "channel_segment_stats": channel_segment_stats,
        "temporal_stability_rows": temporal_rows,
        "channel_temporal_stability_rows": channel_temporal_rows,
        "include_all_channel_rows": bool(args.include_all_channel_rows),
        "verdict": _classify_temporal_stability(temporal_rows),
        "channel_verdict": _classify_temporal_stability(channel_temporal_rows),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "temporal_candidate_stability.json").write_text(
        json.dumps(_to_jsonable(payload), indent=2),
        encoding="utf-8",
    )
    (out_dir / "temporal_candidate_stability.md").write_text(_markdown_report(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="NEXT-11d temporal candidate utility stability diagnostic.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--splits", nargs="*", default=["train_fit", "train_holdout", "val"])
    parser.add_argument("--candidate-feature-mode", type=str, default="shape_proxy")
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--min-segment-support", type=int, default=64)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--positive-rate-threshold", type=float, default=0.52)
    parser.add_argument("--include-all-channel-rows", action="store_true")
    args = parser.parse_args()
    payload = run(args)
    verdict = payload["verdict"]
    channel_verdict = payload["channel_verdict"]
    print(
        "stable_temporal_candidates={} stable_temporal_channel_candidates={} failure_layer={} decision={} no_test_read=True".format(
            int(verdict["stable_candidate_count"]),
            int(channel_verdict["stable_candidate_count"]),
            str(verdict["failure_layer"]),
            str(verdict["decision"]),
        )
    )


if __name__ == "__main__":
    main()
