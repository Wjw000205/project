from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.next11c_route_accuracy_diagnostic import _build_anchor_artifacts
from scripts.next11d_binary_adoption_refit import _to_jsonable
from scripts.next11d_candidate_utility_stability import (
    _candidate_gain_by_channel_penalty,
    _candidate_gain_by_cluster_penalty,
)
from scripts.next11d_fixed_candidate_router_refit import _read_data_for_cfg
from scripts.next11d_temporal_candidate_stability import (
    _base_and_oracle_metrics,
    _classify_temporal_stability,
    _segment_candidate_stats,
    _temporal_stability_rows,
)
from scripts.shape_prior_diagnostic import _build_modules
from src.data.windows import (
    WindowTensorDataset,
    global_zscore,
    make_label_range_windows,
    make_lazy_label_range_window_dataset,
    make_lazy_strict_window_dataset,
    make_strict_windows,
)
from src.train import _collect_pred_residual_selector_tensors, _explainability_train_subsplit_ranges
from src.utils.seed import set_seed
from src.utils.yaml_io import load_yaml


def _normalize_requested_splits(raw_splits: Iterable[str], *, allow_test_read: bool) -> List[str]:
    allowed = {"train_fit", "train_holdout", "val", "test"}
    splits: List[str] = []
    for raw in raw_splits:
        split = str(raw).strip().lower()
        if split == "train":
            split = "train_fit"
        if split == "test" and not bool(allow_test_read):
            raise ValueError("test split requires --allow-test-read for this authorized probe.")
        if split not in allowed:
            raise ValueError(f"unsupported split {raw!r}; expected train_fit, train_holdout, val, or test.")
        if split not in splits:
            splits.append(split)
    return splits or ["train_fit", "train_holdout", "val", "test"]


def _classify_val_test_shift(
    split_summaries: Dict[str, Dict[str, object]],
    *,
    min_oracle_gain_pct: float = 0.25,
) -> Dict[str, object]:
    val = split_summaries.get("val", {})
    test = split_summaries.get("test", {})
    val_oracle = float(val.get("channel_oracle_gain_pct_vs_base", 0.0) or 0.0)
    test_oracle = float(test.get("channel_oracle_gain_pct_vs_base", 0.0) or 0.0)
    val_stable_channel = int(val.get("stable_temporal_channel_candidates", 0) or 0)
    test_stable_channel = int(test.get("stable_temporal_channel_candidates", 0) or 0)
    val_stable_cluster = int(val.get("stable_temporal_cluster_candidates", 0) or 0)
    test_stable_cluster = int(test.get("stable_temporal_cluster_candidates", 0) or 0)
    threshold = float(min_oracle_gain_pct)
    if (
        val_oracle >= threshold
        and test_oracle >= threshold
        and val_stable_channel == 0
        and test_stable_channel == 0
        and val_stable_cluster == 0
        and test_stable_cluster == 0
    ):
        return {
            "failure_layer": "adapter candidate quality",
            "decision": "candidate_temporal_instability_persists_val_and_test",
            "val_channel_oracle_gain_pct": val_oracle,
            "test_channel_oracle_gain_pct": test_oracle,
        }
    if (
        (val_oracle >= threshold and test_oracle < threshold)
        or val_stable_channel != test_stable_channel
        or val_stable_cluster != test_stable_cluster
    ):
        return {
            "failure_layer": "train-val utility shift",
            "decision": "val_test_candidate_oracle_or_stability_mismatch",
            "val_channel_oracle_gain_pct": val_oracle,
            "test_channel_oracle_gain_pct": test_oracle,
        }
    if val_oracle < threshold and test_oracle < threshold:
        return {
            "failure_layer": "adapter candidate quality",
            "decision": "candidate_oracle_headroom_absent_on_val_and_test",
            "val_channel_oracle_gain_pct": val_oracle,
            "test_channel_oracle_gain_pct": test_oracle,
        }
    return {
        "failure_layer": "train-val utility shift",
        "decision": "val_test_shift_not_decisive",
        "val_channel_oracle_gain_pct": val_oracle,
        "test_channel_oracle_gain_pct": test_oracle,
    }


def _make_loaders_with_authorized_test(
    cfg: Dict[str, object],
    data_tc: torch.Tensor,
    *,
    batch_size: int,
    include_test: bool,
) -> Tuple[torch.Tensor, Dict[str, DataLoader], Dict[str, int], DataLoader, Dict[str, object]]:
    data_cfg = cfg["data"]
    max_rows = int(data_cfg.get("max_rows", 0) or 0)
    if max_rows > 0:
        data_tc = data_tc[:max_rows]
    T = int(data_tc.shape[0])
    t_train = int(T * float(data_cfg["train_ratio"]))
    t_val = int(T * (float(data_cfg["train_ratio"]) + float(data_cfg["val_ratio"])))
    norm_cfg = cfg["normalize"]
    if bool(norm_cfg["global_zscore"]):
        if bool(norm_cfg.get("train_only", False)):
            train_seg = data_tc[:t_train]
            mean_c = train_seg.mean(dim=0, keepdim=True)
            std_c = train_seg.std(dim=0, keepdim=True).clamp_min(1.0e-6)
            data_tc = (data_tc - mean_c) / std_c
        else:
            data_tc, _, _ = global_zscore(data_tc)
    L = int(cfg["window"]["input_len"])
    H = int(cfg["window"]["pred_len"])
    past_context = bool((cfg.get("window", {}) or {}).get("past_context", False))
    lazy_windows = bool((cfg.get("window", {}) or {}).get("lazy", False))
    data_window_tc = data_tc.detach().cpu()
    if lazy_windows:
        dtr = make_lazy_strict_window_dataset(data_window_tc, L, H, 0, t_train)
        if past_context:
            dva, val_eval_start = make_lazy_label_range_window_dataset(data_window_tc, L, H, t_train, t_val)
            if include_test:
                dte, test_eval_start = make_lazy_label_range_window_dataset(data_window_tc, L, H, t_val, T)
            else:
                dte = make_lazy_strict_window_dataset(data_window_tc, L, H, 0, 0)
                test_eval_start = t_val
        else:
            dva = make_lazy_strict_window_dataset(data_window_tc, L, H, t_train, t_val)
            val_eval_start = t_train
            dte = (
                make_lazy_strict_window_dataset(data_window_tc, L, H, t_val, T)
                if include_test
                else make_lazy_strict_window_dataset(data_window_tc, L, H, 0, 0)
            )
            test_eval_start = t_val
    else:
        xtr, ytr = make_strict_windows(data_window_tc, L, H, 0, t_train)
        if past_context:
            xva, yva, val_eval_start = make_label_range_windows(data_window_tc, L, H, t_train, t_val)
            if include_test:
                xte, yte, test_eval_start = make_label_range_windows(data_window_tc, L, H, t_val, T)
            else:
                xte = torch.empty(0, data_tc.shape[1], L, dtype=data_window_tc.dtype)
                yte = torch.empty(0, data_tc.shape[1], H, dtype=data_window_tc.dtype)
                test_eval_start = t_val
        else:
            xva, yva = make_strict_windows(data_window_tc, L, H, t_train, t_val)
            val_eval_start = t_train
            if include_test:
                xte, yte = make_strict_windows(data_window_tc, L, H, t_val, T)
            else:
                xte = torch.empty(0, data_tc.shape[1], L, dtype=data_window_tc.dtype)
                yte = torch.empty(0, data_tc.shape[1], H, dtype=data_window_tc.dtype)
            test_eval_start = t_val
        dtr = WindowTensorDataset(xtr, ytr)
        dva = WindowTensorDataset(xva, yva)
        dte = WindowTensorDataset(xte, yte)
    ranges = _explainability_train_subsplit_ranges(
        num_windows=len(dtr),
        holdout_fraction=float(((cfg.get("moe", {}) or {}).get("explainability", {}) or {}).get("train_holdout_fraction", 0.30)),
    )
    loaders = {
        "train_fit": DataLoader(Subset(dtr, range(*ranges["train_fit"])), batch_size=batch_size, shuffle=False),
        "train_holdout": DataLoader(Subset(dtr, range(*ranges["train_holdout"])), batch_size=batch_size, shuffle=False),
        "val": DataLoader(dva, batch_size=batch_size, shuffle=False),
        "test": DataLoader(dte, batch_size=batch_size, shuffle=False),
    }
    eval_starts = {
        "train_fit": 0,
        "train_holdout": 0,
        "val": int(val_eval_start),
        "test": int(test_eval_start),
    }
    train_loader = DataLoader(dtr, batch_size=batch_size, shuffle=False)
    return data_window_tc, loaders, eval_starts, train_loader, {
        "T": T,
        "t_train": t_train,
        "t_val": t_val,
        "L": L,
        "H": H,
        "test_read": bool(include_test),
    }


def _stable_count(rows: List[Dict[str, object]]) -> int:
    return int(sum(1 for row in rows if bool(row.get("stable_train_segments", False))))


def _markdown_report(payload: Dict[str, object]) -> str:
    lines = [
        "# NEXT-11d Authorized Test Shift Probe",
        "",
        f"- config: `{payload['config_path']}`",
        f"- checkpoint: `{payload['checkpoint_path']}`",
        f"- test_read: `{payload['test_read']}`",
        f"- failure_layer: `{payload['verdict']['failure_layer']}`",
        f"- decision: `{payload['verdict']['decision']}`",
        "",
        "## Split Summary",
        "",
        "| split | base mse | base mae | channel oracle mse | channel oracle mae | oracle mse gain | stable cluster | stable channel | cluster val/test flips | channel val/test flips |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in payload.get("splits_collected", []):
        metrics = payload.get("split_metrics", {}).get(split, {})
        summary = payload.get("split_summaries", {}).get(split, {})
        lines.append(
            "| {split} | {base_mse:.6f} | {base_mae:.6f} | {oracle_mse:.6f} | {oracle_mae:.6f} | {gain:.3f}% | {sc} | {sch} | {cf} | {chf} |".format(
                split=split,
                base_mse=float(metrics.get("base_mse", 0.0) or 0.0),
                base_mae=float(metrics.get("base_mae", 0.0) or 0.0),
                oracle_mse=float(metrics.get("channel_oracle_mse", 0.0) or 0.0),
                oracle_mae=float(metrics.get("channel_oracle_mae", 0.0) or 0.0),
                gain=float(metrics.get("channel_oracle_gain_pct_vs_base", 0.0) or 0.0),
                sc=int(summary.get("stable_temporal_cluster_candidates", 0) or 0),
                sch=int(summary.get("stable_temporal_channel_candidates", 0) or 0),
                cf=int(summary.get("cluster_eval_flip_count", 0) or 0),
                chf=int(summary.get("channel_eval_flip_count", 0) or 0),
            )
        )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> Dict[str, object]:
    splits = _normalize_requested_splits(args.splits, allow_test_read=bool(args.allow_test_read))
    include_test = "test" in splits
    cfg = load_yaml(str(args.config))
    cfg.setdefault("eval", {})["skip_test"] = not include_test
    set_seed(int(cfg["exp"]["seed"]), deterministic=bool((cfg.get("exp", {}) or {}).get("deterministic", False)))
    requested_device = str(args.device or cfg.get("exp", {}).get("device", "cpu"))
    device = torch.device(requested_device if torch.cuda.is_available() and requested_device != "cpu" else "cpu")
    checkpoint = torch.load(str(args.checkpoint), map_location=device)
    model, _, pred_residual, cluster_id_c, K, moe_cfg, penalty_names = _build_modules(cfg, checkpoint, device)
    data_tc = _read_data_for_cfg(cfg)
    batch_size = int(cfg.get("train", {}).get("batch_size", 64))
    data_window_tc, loaders, eval_starts, train_loader, window_meta = _make_loaders_with_authorized_test(
        cfg,
        data_tc,
        batch_size=batch_size,
        include_test=include_test,
    )
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
    if "train_fit" not in cluster_segment_stats or "train_holdout" not in cluster_segment_stats:
        raise ValueError("authorized test shift probe requires train_fit and train_holdout.")
    split_summaries: Dict[str, Dict[str, object]] = {}
    temporal_rows_by_split: Dict[str, List[Dict[str, object]]] = {}
    channel_temporal_rows_by_split: Dict[str, List[Dict[str, object]]] = {}
    for split in splits:
        if split in {"train_fit", "train_holdout"}:
            continue
        rows = _temporal_stability_rows(
            fit_segment_stats=cluster_segment_stats["train_fit"],
            holdout_segment_stats=cluster_segment_stats["train_holdout"],
            val_segment_stats=cluster_segment_stats[split],
            entity_name="cluster",
            min_segment_support=int(args.min_segment_support),
            margin=float(args.margin),
            positive_rate_threshold=float(args.positive_rate_threshold),
        )
        channel_rows = _temporal_stability_rows(
            fit_segment_stats=channel_segment_stats["train_fit"],
            holdout_segment_stats=channel_segment_stats["train_holdout"],
            val_segment_stats=channel_segment_stats[split],
            entity_name="channel",
            min_segment_support=int(args.min_segment_support),
            margin=float(args.margin),
            positive_rate_threshold=float(args.positive_rate_threshold),
        )
        temporal_rows_by_split[split] = rows
        channel_temporal_rows_by_split[split] = channel_rows
        split_summaries[split] = {
            "channel_oracle_gain_pct_vs_base": float(
                split_metrics[split].get("channel_oracle_gain_pct_vs_base", 0.0) or 0.0
            ),
            "stable_temporal_cluster_candidates": _stable_count(rows),
            "stable_temporal_channel_candidates": _stable_count(channel_rows),
            "cluster_eval_flip_count": int(_classify_temporal_stability(rows).get("val_flip_count", 0) or 0),
            "channel_eval_flip_count": int(_classify_temporal_stability(channel_rows).get("val_flip_count", 0) or 0),
        }
    payload = {
        "config_path": str(args.config),
        "checkpoint_path": str(args.checkpoint),
        "out_dir": str(args.out_dir),
        "device": str(device),
        "test_read": bool(include_test),
        "test_read_authorized": bool(args.allow_test_read),
        "splits_collected": splits,
        "penalty_names": list(penalty_names),
        "criteria": {
            "segments": int(args.segments),
            "min_segment_support": int(args.min_segment_support),
            "margin": float(args.margin),
            "positive_rate_threshold": float(args.positive_rate_threshold),
            "min_oracle_gain_pct": float(args.min_oracle_gain_pct),
        },
        "split_metrics": split_metrics,
        "split_summaries": split_summaries,
        "temporal_rows_by_split": temporal_rows_by_split,
        "channel_temporal_rows_by_split": channel_temporal_rows_by_split,
        "verdict": _classify_val_test_shift(
            split_summaries,
            min_oracle_gain_pct=float(args.min_oracle_gain_pct),
        ),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "authorized_test_shift_probe.json").write_text(
        json.dumps(_to_jsonable(payload), indent=2),
        encoding="utf-8",
    )
    (out_dir / "authorized_test_shift_probe.md").write_text(_markdown_report(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="NEXT-11d authorized val/test candidate shift probe.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--splits", nargs="*", default=["train_fit", "train_holdout", "val", "test"])
    parser.add_argument("--allow-test-read", action="store_true")
    parser.add_argument("--candidate-feature-mode", type=str, default="shape_proxy")
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--min-segment-support", type=int, default=64)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--positive-rate-threshold", type=float, default=0.52)
    parser.add_argument("--min-oracle-gain-pct", type=float, default=0.25)
    args = parser.parse_args()
    payload = run(args)
    verdict = payload["verdict"]
    print(
        "test_read={} failure_layer={} decision={} val_oracle={:.3f} test_oracle={:.3f}".format(
            bool(payload["test_read"]),
            str(verdict["failure_layer"]),
            str(verdict["decision"]),
            float(verdict.get("val_channel_oracle_gain_pct", 0.0) or 0.0),
            float(verdict.get("test_channel_oracle_gain_pct", 0.0) or 0.0),
        )
    )


if __name__ == "__main__":
    main()
