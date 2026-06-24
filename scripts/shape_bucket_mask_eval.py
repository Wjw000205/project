from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import shape_prior_diagnostic as spd
from src.models.moe_gate import scatter_mean_bc_to_bk, scatter_mean_bcf_to_bkf
from src.models.penalties import build_penalty_bank
from src.train import (
    _build_gate_routing_features,
    _normalize_gate_feature_mode,
    _normalize_history_anchor_cfg,
    _pred_residual_candidates_on_eval_path,
    _router_penalty_context_from_history,
    _validate_strict_history_anchor_scope,
    apply_history_anchor_adapter,
    apply_moe_output_anchor_experts,
    apply_train_stat_anchor_expert,
    apply_train_stat_input_centering,
    build_train_residual_anchor_table_from_loader,
    build_train_stat_anchor_from_config,
)
from src.utils.seed import set_seed
from src.utils.yaml_io import load_yaml
from src.data.reader import read_csv_time_series


def _passes(row: Dict[str, object], n_min: int, margin: float, positive_rate_holdout: float) -> bool:
    if bool(row.get("base_mse_proxy_flag", False)):
        return False
    splits = row.get("splits", {}) or {}
    fit = splits.get("train_fit")
    hold = splits.get("train_holdout")
    if not isinstance(fit, dict) or not isinstance(hold, dict):
        return False
    return (
        int(fit.get("support_count", 0)) >= int(n_min)
        and int(hold.get("support_count", 0)) >= int(n_min)
        and float(fit.get("mean_gain", 0.0)) > float(margin)
        and float(hold.get("mean_gain", 0.0)) > float(margin)
        and float(hold.get("positive_rate", 0.0)) >= float(positive_rate_holdout)
    )


def _family_score(rows: List[Dict[str, object]]) -> float:
    score = 0.0
    for row in rows:
        splits = row["splits"]
        fit = splits["train_fit"]
        hold = splits["train_holdout"]
        stable_gain = min(float(fit["mean_gain"]), float(hold["mean_gain"]))
        score += stable_gain * float(hold["support_count"])
    return float(score)


def fit_shape_bucket_mask(
    *,
    bucket_stats: Dict[str, object],
    bucket_edges: Dict[str, object],
    penalty_names: List[str],
    allowed_mask_kp: torch.Tensor,
    n_min: int,
    margin: float,
    positive_rate_holdout: float,
) -> Dict[str, object]:
    """Fit a shape mask from train_fit/train_holdout bucket stats only."""
    rows = [
        row
        for row in (bucket_stats.get("accepted", []) or bucket_stats.get("rows", []) or [])
        if _passes(row, n_min=n_min, margin=margin, positive_rate_holdout=positive_rate_holdout)
    ]
    if not rows:
        raise ValueError("No bucket rows pass the requested train-only mask thresholds.")
    def feature_index_for(q_value: int, feature_name: str) -> int:
        q_key = f"q{int(q_value)}"
        if q_key not in bucket_edges:
            raise ValueError(f"Missing bucket edges for {q_key}.")
        names = [str(name) for name in bucket_edges[q_key]["feature_names"]]
        if str(feature_name) not in names:
            raise ValueError(f"Feature {feature_name!r} is missing from {q_key} bucket edges.")
        return int(names.index(str(feature_name)))

    by_family: Dict[Tuple[int, str, int], List[Dict[str, object]]] = {}
    for row in rows:
        q_value = int(row["q"])
        feature_name = str(row["feature"])
        feature_index = int(row.get("feature_index", feature_index_for(q_value, feature_name)))
        key = (q_value, feature_name, feature_index)
        by_family.setdefault(key, []).append(row)
    best_key, best_rows = max(
        by_family.items(),
        key=lambda item: (_family_score(item[1]), len(item[1]), -item[0][2]),
    )
    q, feature, feature_index = best_key
    penalty_to_idx = {str(name): i for i, name in enumerate(penalty_names)}
    allowed = allowed_mask_kp.detach().cpu().to(dtype=torch.bool)
    K, P = [int(v) for v in allowed.shape]
    allow_kbp = torch.zeros((K, q, P), dtype=torch.bool)
    accepted_rows = []
    for row in best_rows:
        k = int(row["cluster_id"])
        b = int(row["bucket"])
        p = int(row.get("penalty_index", penalty_to_idx[str(row["penalty"])]))
        if 0 <= k < K and 0 <= b < q and 0 <= p < P and bool(allowed[k, p].item()):
            allow_kbp[k, b, p] = True
            accepted_rows.append(row)
    if not accepted_rows:
        raise ValueError("Chosen shape family had no rows after cluster-penalty allowed-mask intersection.")
    q_key = f"q{q}"
    if q_key not in bucket_edges:
        raise ValueError(f"Missing bucket edges for {q_key}.")
    edge_payload = bucket_edges[q_key]
    feature_names = list(edge_payload["feature_names"])
    if feature_names[int(feature_index)] != feature:
        raise ValueError("Bucket edge feature index/name does not match chosen mask family.")
    feature_edges_k = [edge_payload["edges"][k][int(feature_index)] for k in range(K)]
    return {
        "enable": True,
        "q": int(q),
        "feature": str(feature),
        "feature_index": int(feature_index),
        "feature_edges_k": feature_edges_k,
        "penalty_names": list(penalty_names),
        "cluster_allowed_mask_kp": allowed.to(dtype=torch.bool).tolist(),
        "allow_kbp": allow_kbp.tolist(),
        "source_splits": ["train_fit", "train_holdout"],
        "fit_policy": {
            "selection": "best_single_family_train_holdout_score",
            "n_min": int(n_min),
            "margin": float(margin),
            "positive_rate_holdout": float(positive_rate_holdout),
            "score": _family_score(best_rows),
            "accepted_rows": len(accepted_rows),
        },
        "accepted_rows": accepted_rows,
        "uses_val_labels": False,
        "uses_test_labels": False,
    }


def save_shape_mask(mask_spec: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mask_spec, indent=2, sort_keys=True), encoding="utf-8")


def load_shape_mask(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def bucket_ids_for_mask(features_bkf: torch.Tensor, mask_spec: Dict[str, object]) -> torch.Tensor:
    feature_index = int(mask_spec["feature_index"])
    q = int(mask_spec["q"])
    edges = mask_spec["feature_edges_k"]
    B, K, _ = [int(v) for v in features_bkf.shape]
    out = torch.zeros((B, K), dtype=torch.long, device=features_bkf.device)
    vals_bk = features_bkf[:, :, feature_index]
    for k in range(K):
        edge = torch.as_tensor(edges[k], device=features_bkf.device, dtype=features_bkf.dtype)
        out[:, k] = torch.bucketize(vals_bk[:, k].contiguous(), edge, right=False).clamp(0, q - 1)
    return out


def route_mask_from_shape_buckets(
    *,
    probs_bkp: torch.Tensor,
    bucket_ids_bk: torch.Tensor,
    mask_spec: Dict[str, object],
    select_ranks: Optional[List[int]],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if probs_bkp.dim() != 3:
        raise ValueError("probs_bkp must have shape [B,K,P].")
    B, K, P = [int(v) for v in probs_bkp.shape]
    if tuple(bucket_ids_bk.shape) != (B, K):
        raise ValueError("bucket_ids_bk must have shape [B,K].")
    shape_allow = torch.zeros((B, K, P), dtype=torch.bool, device=probs_bkp.device)
    allow_kbp = torch.as_tensor(mask_spec["allow_kbp"], dtype=torch.bool, device=probs_bkp.device)
    cluster_allowed = torch.as_tensor(
        mask_spec["cluster_allowed_mask_kp"],
        dtype=torch.bool,
        device=probs_bkp.device,
    )
    if tuple(allow_kbp.shape) != (K, int(mask_spec["q"]), P):
        raise ValueError("mask_spec allow_kbp shape does not match probs_bkp.")
    if tuple(cluster_allowed.shape) != (K, P):
        raise ValueError("mask_spec cluster_allowed_mask_kp shape does not match probs_bkp.")
    for k in range(K):
        shape_allow[:, k, :] = allow_kbp[k].index_select(0, bucket_ids_bk[:, k])
    allowed = shape_allow & cluster_allowed.unsqueeze(0)
    no_op_bk = allowed.sum(dim=-1) <= 0
    ranks = [1] if select_ranks is None else [int(r) for r in select_ranks]
    route = torch.zeros_like(probs_bkp)
    scores = probs_bkp.masked_fill(~allowed, -float("inf"))
    order = scores.argsort(dim=-1, descending=True)
    allowed_count = allowed.sum(dim=-1)
    for rank in ranks:
        rank_idx = max(int(rank) - 1, 0)
        chosen = order[..., rank_idx]
        valid = allowed_count > rank_idx
        route.scatter_(-1, chosen.unsqueeze(-1), valid.to(dtype=route.dtype).unsqueeze(-1))
    route = route * (~no_op_bk).to(dtype=route.dtype).unsqueeze(-1)
    stats = {
        "no_op_rate": float(no_op_bk.to(dtype=torch.float32).mean().item()),
        "shape_allowed_rate": float(allowed.to(dtype=torch.float32).mean().item()),
        "mean_allowed_penalties": float(allowed_count.to(dtype=torch.float32).mean().item()),
    }
    return route, stats


def _pct_delta(new: float, old: float) -> float:
    return 100.0 * (float(new) - float(old)) / max(abs(float(old)), 1.0e-12)


def _load_step0_refs(out_dir: Path) -> Dict[str, object]:
    candidate = out_dir.parent / "step0_references" / "step0_references.json"
    if candidate.exists():
        return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


@torch.no_grad()
def _evaluate_val_mask(
    *,
    loader,
    eval_start: int,
    model,
    gate,
    pred_residual,
    cluster_id_c: torch.Tensor,
    K: int,
    moe_cfg: Dict[str, object],
    device: torch.device,
    penalty_names: List[str],
    penalty_fns: Dict[str, object],
    penalty_scale: torch.Tensor,
    mask_spec: Dict[str, object],
    history_anchor_cfg: Dict[str, object],
    observed_history_tc: torch.Tensor,
    input_len: int,
    model_train_stat_adapter_pc: Optional[torch.Tensor],
    model_train_stat_adapter_cfg: Dict[str, object],
    train_stat_anchor_pc: Optional[torch.Tensor],
    train_residual_anchor_phc: Optional[torch.Tensor],
    gate_feature_mode: str,
) -> Dict[str, object]:
    router_mode = str(moe_cfg.get("router_mode", "learned")).lower()
    router_weight = float(moe_cfg.get("router_penalty_context_weight", 0.0))
    router_detach = bool(moe_cfg.get("router_detach_penalty_context", True))
    router_score = str(moe_cfg.get("router_penalty_context_score", "high_violation")).lower()
    select_ranks = moe_cfg.get("select_ranks", None)
    if select_ranks is not None:
        select_ranks = [int(v) for v in select_ranks]

    base_sse = final_sse = base_sae = final_sae = 0.0
    count = 0
    route_stats_sum = {"no_op_rate": 0.0, "shape_allowed_rate": 0.0, "mean_allowed_penalties": 0.0}
    route_batches = 0
    no_op_count = 0
    decision_count = 0
    oracle_positive_count = 0
    no_op_on_oracle_positive_count = 0
    selected_count = 0
    selected_positive_count = 0
    selected_gain_sum = 0.0
    for x, y, idx in loader:
        x = x.to(device)
        y = y.to(device)
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
        _, probs_bkp, _, _ = gate(
            feat_bkf,
            straight_through=False,
            penalty_context_bkp=route_pen_bkp,
            penalty_context_mode=router_mode,
            penalty_context_weight=router_weight,
            penalty_context_detach=router_detach,
            penalty_context_score=router_score,
        )
        y_base_for_shape = apply_moe_output_anchor_experts(
            y_base,
            base_pred_bch=y_base,
            x_bcl=x,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len),
            moe_cfg=moe_cfg,
            moe_enable=bool(moe_cfg.get("enable", True)),
            observed_history_tc=observed_history_tc,
            train_stat_anchor_pc=train_stat_anchor_pc,
            train_residual_anchor_phc=train_residual_anchor_phc,
        )
        shape_bkf, _ = spd.compute_shape_features(x, y_base_for_shape, cluster_id_c, K)
        bucket_ids = bucket_ids_for_mask(shape_bkf, mask_spec)
        shape_route_bkp, route_stats = route_mask_from_shape_buckets(
            probs_bkp=probs_bkp,
            bucket_ids_bk=bucket_ids,
            mask_spec=mask_spec,
            select_ranks=select_ranks,
        )
        pred_out = pred_residual(x, y_base, cluster_id_c, shape_route_bkp, skip_bk=None)
        y_final = apply_moe_output_anchor_experts(
            pred_out["y_final"],
            base_pred_bch=y_base,
            x_bcl=x,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len),
            moe_cfg=moe_cfg,
            moe_enable=bool(moe_cfg.get("enable", True)),
            observed_history_tc=observed_history_tc,
            train_stat_anchor_pc=train_stat_anchor_pc,
            train_residual_anchor_phc=train_residual_anchor_phc,
        )
        y_base_final, cand_bcpH = _pred_residual_candidates_on_eval_path(
            y_base,
            pred_out,
            apply_output_anchors=True,
            x_bcl=x,
            query_start_abs_b=query_start_abs_b,
            input_len=int(input_len),
            moe_cfg=moe_cfg,
            moe_enable=bool(moe_cfg.get("enable", True)),
            observed_history_tc=observed_history_tc,
            train_stat_anchor_pc=train_stat_anchor_pc,
            train_residual_anchor_phc=train_residual_anchor_phc,
        )
        base_err = y_base_final - y
        final_err = y_final - y
        base_sse += float(base_err.pow(2).sum().item())
        final_sse += float(final_err.pow(2).sum().item())
        base_sae += float(base_err.abs().sum().item())
        final_sae += float(final_err.abs().sum().item())
        count += int(base_err.numel())
        for key in route_stats_sum:
            route_stats_sum[key] += float(route_stats[key])
        route_batches += 1

        if cand_bcpH is not None:
            base_err_bc = base_err.pow(2).mean(dim=-1)
            cand_err_bcp = (cand_bcpH - y.unsqueeze(2)).pow(2).mean(dim=-1)
            gain_bcp = base_err_bc.unsqueeze(-1) - cand_err_bcp
            gain_bkp = scatter_mean_bcf_to_bkf(gain_bcp, cluster_id_c, K)
            allowed = torch.as_tensor(mask_spec["cluster_allowed_mask_kp"], device=device, dtype=torch.bool)
            best_gain = gain_bkp.masked_fill(~allowed.unsqueeze(0), -float("inf")).max(dim=-1).values
            best_gain = torch.where(torch.isfinite(best_gain), best_gain, torch.zeros_like(best_gain))
            oracle_positive = best_gain > 0.0
            no_op = shape_route_bkp.sum(dim=-1) <= 0.0
            no_op_count += int(no_op.sum().item())
            decision_count += int(no_op.numel())
            oracle_positive_count += int(oracle_positive.sum().item())
            no_op_on_oracle_positive_count += int((no_op & oracle_positive).sum().item())
            selected = shape_route_bkp > 0.0
            selected_count += int(selected.sum().item())
            selected_positive_count += int((selected & (gain_bkp > 0.0)).sum().item())
            selected_gain_sum += float((gain_bkp * selected.to(dtype=gain_bkp.dtype)).sum().item())

    base_mse = base_sse / max(count, 1)
    final_mse = final_sse / max(count, 1)
    base_mae = base_sae / max(count, 1)
    final_mae = final_sae / max(count, 1)
    return {
        "base": {"avg_mse": base_mse, "avg_mae": base_mae},
        "shape_mask": {
            "avg_mse": final_mse,
            "avg_mae": final_mae,
            "delta_pct_vs_base": {
                "avg_mse": _pct_delta(final_mse, base_mse),
                "avg_mae": _pct_delta(final_mae, base_mae),
            },
            "raw_route_gain_pct_vs_base": 100.0 * (base_mse - final_mse) / max(abs(base_mse), 1.0e-12),
        },
        "route_stats": {
            **{k: v / max(route_batches, 1) for k, v in route_stats_sum.items()},
            "no_op_count": int(no_op_count),
            "decision_count": int(decision_count),
            "oracle_positive_count": int(oracle_positive_count),
            "no_op_on_oracle_positive_count": int(no_op_on_oracle_positive_count),
            "no_op_on_oracle_positive_rate_of_oracle_positive": float(
                no_op_on_oracle_positive_count / max(oracle_positive_count, 1)
            ),
            "selected_count": int(selected_count),
            "selected_positive_count": int(selected_positive_count),
            "selected_positive_rate": float(selected_positive_count / max(selected_count, 1)),
            "selected_mean_gain_mse": float(selected_gain_sum / max(selected_count, 1)),
        },
    }


def run(args: argparse.Namespace) -> Dict[str, object]:
    cfg = load_yaml(args.config)
    cfg.setdefault("eval", {})["skip_test"] = True
    requested_skip = bool(args.skip_test)
    if not requested_skip or not bool((cfg.get("eval", {}) or {}).get("skip_test", True)):
        raise ValueError("shape bucket mask eval requires --skip-test and eval.skip_test=true.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(cfg["exp"]["seed"]), deterministic=bool((cfg.get("exp", {}) or {}).get("deterministic", False)))
    device = torch.device(str(args.device or cfg["exp"].get("device", "cpu")) if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model, gate, pred_residual, cluster_id_c, K, moe_cfg, penalty_names = spd._build_modules(cfg, checkpoint, device)
    diag_dir = Path(args.shape_diagnostic_dir)
    bucket_stats = json.loads((diag_dir / "shape_bucket_gain_stats.json").read_text(encoding="utf-8"))
    bucket_edges = json.loads((diag_dir / "shape_bucket_edges.json").read_text(encoding="utf-8"))
    allowed_mask = spd.build_allowed_mask(
        penalty_names=penalty_names,
        K=K,
        allowed_by_cluster=spd.DEFAULT_ALLOWED_BY_CLUSTER,
        device=torch.device("cpu"),
    )
    mask_spec = fit_shape_bucket_mask(
        bucket_stats=bucket_stats,
        bucket_edges=bucket_edges,
        penalty_names=penalty_names,
        allowed_mask_kp=allowed_mask,
        n_min=int(args.n_min),
        margin=float(args.margin),
        positive_rate_holdout=float(args.positive_rate_holdout),
    )
    save_shape_mask(mask_spec, out_dir / "shape_bucket_mask.json")

    data_tc, _ = read_csv_time_series(cfg["data"]["csv_path"], date_col=int(cfg["data"]["date_col"]))
    data_window_tc, loaders, eval_starts, train_loader, window_meta = spd._make_loaders(
        cfg,
        data_tc,
        batch_size=int(cfg["train"]["batch_size"]),
    )
    penalty_fns = build_penalty_bank(penalty_names, jump_thr=float(cfg["penalties"].get("jump_threshold", 0.6)))
    penalty_scale = spd._compute_penalty_scale(train_loader, penalty_names, penalty_fns, int(window_meta["H"]), device)
    model_cfg = dict(checkpoint["meta"].get("model_cfg", cfg.get("model", {}) or {}))
    history_anchor_cfg = _normalize_history_anchor_cfg(model_cfg.get("history_anchor", cfg.get("history_anchor", {}) or {}))
    _validate_strict_history_anchor_scope(history_anchor_cfg, source="shape_bucket_mask.history_anchor")
    model_train_stat_adapter_cfg = model_cfg.get("train_stat_adapter", {}) or {}
    model_train_stat_adapter_pc, _, _ = build_train_stat_anchor_from_config(
        data_window_tc,
        train_end=int(window_meta["t_train"]),
        input_len=int(window_meta["L"]),
        pred_len=int(window_meta["H"]),
        cfg=model_train_stat_adapter_cfg,
        prefix="shape_bucket_mask.model.train_stat_adapter",
    )
    train_stat_anchor_cfg = moe_cfg.get("train_stat_anchor_expert", {}) or {}
    train_stat_anchor_pc, _, _ = build_train_stat_anchor_from_config(
        data_window_tc,
        train_end=int(window_meta["t_train"]),
        input_len=int(window_meta["L"]),
        pred_len=int(window_meta["H"]),
        cfg=train_stat_anchor_cfg,
        prefix="shape_bucket_mask.moe.train_stat_anchor_expert",
    )
    train_residual_anchor_phc = None
    train_residual_anchor_cfg = moe_cfg.get("train_residual_anchor_expert", {}) or {}
    if bool(train_residual_anchor_cfg.get("enable", False)):
        train_residual_anchor_phc, _, _ = build_train_residual_anchor_table_from_loader(
            model=model,
            loader=train_loader,
            cluster_id_c=cluster_id_c,
            device=device,
            history_anchor_cfg=history_anchor_cfg,
            observed_history_tc=data_window_tc,
            input_len=int(window_meta["L"]),
            eval_start=0,
            period=int(train_residual_anchor_cfg.get("period", 96)),
            model_train_stat_adapter_pc=model_train_stat_adapter_pc,
            model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
            train_stat_anchor_pc=train_stat_anchor_pc,
            train_stat_anchor_cfg=train_stat_anchor_cfg,
        )
    gate_feature_mode = _normalize_gate_feature_mode(str(checkpoint["meta"].get("gate_feature_mode", "history")))
    val_result = _evaluate_val_mask(
        loader=loaders["val"],
        eval_start=int(eval_starts["val"]),
        model=model,
        gate=gate,
        pred_residual=pred_residual,
        cluster_id_c=cluster_id_c,
        K=K,
        moe_cfg=moe_cfg,
        device=device,
        penalty_names=penalty_names,
        penalty_fns=penalty_fns,
        penalty_scale=penalty_scale,
        mask_spec=mask_spec,
        history_anchor_cfg=history_anchor_cfg,
        observed_history_tc=data_window_tc,
        input_len=int(window_meta["L"]),
        model_train_stat_adapter_pc=model_train_stat_adapter_pc,
        model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
        train_stat_anchor_pc=train_stat_anchor_pc,
        train_residual_anchor_phc=train_residual_anchor_phc,
        gate_feature_mode=gate_feature_mode,
    )
    refs = _load_step0_refs(out_dir)
    anchored = refs.get("anchored_base_val", {}) if refs else {}
    current_scaled = refs.get("selected_scaled_val", {}) if refs else {}
    summary = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "shape_diagnostic_dir": str(diag_dir),
        "out_dir": str(out_dir),
        "skip_test": True,
        "test_read": False,
        "mask": mask_spec,
        "val": val_result,
        "references": {
            "anchored_base_val": anchored,
            "current_selected_scaled_val": current_scaled,
        },
    }
    if anchored:
        summary["val"]["shape_mask"]["delta_pct_vs_step0_anchored_base"] = {
            "avg_mse": _pct_delta(val_result["shape_mask"]["avg_mse"], float(anchored["avg_mse"])),
            "avg_mae": _pct_delta(val_result["shape_mask"]["avg_mae"], float(anchored["avg_mae"])),
        }
    if current_scaled:
        summary["val"]["shape_mask"]["delta_pct_vs_step0_selected_scaled"] = {
            "avg_mse": _pct_delta(val_result["shape_mask"]["avg_mse"], float(current_scaled["avg_mse"])),
            "avg_mae": _pct_delta(val_result["shape_mask"]["avg_mae"], float(current_scaled["avg_mae"])),
        }
    (out_dir / "shape_bucket_mask_val_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    md = [
        "# Shape Bucket Mask Val Summary",
        "",
        f"- Test read: `no`",
        f"- Mask family: `q{mask_spec['q']} {mask_spec['feature']}`",
        f"- Base val: MSE `{val_result['base']['avg_mse']:.9f}`, MAE `{val_result['base']['avg_mae']:.9f}`",
        f"- Shape mask val: MSE `{val_result['shape_mask']['avg_mse']:.9f}`, MAE `{val_result['shape_mask']['avg_mae']:.9f}`",
        f"- Raw route gain vs base: `{val_result['shape_mask']['raw_route_gain_pct_vs_base']:.3f}%`",
        f"- No-op rate: `{val_result['route_stats']['no_op_rate']:.3f}`",
        f"- No-op on oracle-positive rate: `{val_result['route_stats']['no_op_on_oracle_positive_rate_of_oracle_positive']:.3f}`",
        f"- Selected positive rate: `{val_result['route_stats']['selected_positive_rate']:.3f}`",
    ]
    (out_dir / "shape_bucket_mask_val_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--shape-diagnostic-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-min", type=int, default=128)
    parser.add_argument("--margin", type=float, default=0.001)
    parser.add_argument("--positive-rate-holdout", type=float, default=0.60)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--device", default="")
    args = parser.parse_args()
    summary = run(args)
    val = summary["val"]["shape_mask"]
    print(
        json.dumps(
            {
                "out_dir": summary["out_dir"],
                "mask_family": f"q{summary['mask']['q']} {summary['mask']['feature']}",
                "val_mse": val["avg_mse"],
                "val_mae": val["avg_mae"],
                "raw_route_gain_pct_vs_base": val["raw_route_gain_pct_vs_base"],
                "no_op_rate": summary["val"]["route_stats"]["no_op_rate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
