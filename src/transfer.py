import os
import argparse
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .utils.yaml_io import load_yaml
from .utils.seed import set_seed
from .data.windows import global_zscore, make_strict_windows, WindowTensorDataset
from .utils.metrics import accumulate_channel_errors, mse_mae_from_sums
from .utils.cluster_memory import (
    load_cluster_memory,
    load_cluster_checkpoint,
    assign_channels_by_corr,
    assign_channels_by_cycle_template,
    balance_cluster_assignment_by_source_counts,
    cluster_count_targets_from_source,
)
from .models.cluster_predictor import build_cluster_predictor
from .models.moe_gate import ClusterwiseMoEGate, scatter_mean_bcf_to_bkf
from .models.residual_moe import ClusterwisePredResidualMoE
from .train import extract_gate_features, _select_rank_mask


def _infer_step_minutes(df: pd.DataFrame, date_col: str) -> float:
    dt = pd.to_datetime(df[date_col])
    diffs = dt.diff().dropna()
    if diffs.empty:
        return 0.0
    step = diffs.mode().iloc[0]
    return float(step.total_seconds() / 60.0)


def _df_to_tensor(df: pd.DataFrame, date_col: str) -> tuple[torch.Tensor, list[str]]:
    cols = list(df.columns)
    value_cols = [c for c in cols if c != date_col]
    values = df[value_cols].to_numpy(dtype=np.float32)
    data = torch.tensor(values, dtype=torch.float32)
    return data, value_cols


def _resample_df(
    df: pd.DataFrame,
    date_col: str,
    target_step_min: int,
    method: str,
) -> pd.DataFrame:
    if target_step_min <= 0:
        return df
    rule = f"{int(target_step_min)}min"
    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col])
    value_cols = [c for c in tmp.columns if c != date_col]
    tmp = (
        tmp.groupby(date_col, as_index=False)[value_cols]
        .mean()
        .sort_values(date_col)
        .reset_index(drop=True)
    )
    tmp[value_cols] = tmp[value_cols].ffill().bfill()
    tmp = tmp.set_index(date_col)
    if method in {"mean", "avg"}:
        out = tmp.resample(rule).mean().interpolate("time").ffill().bfill()
    elif method in {"last", "ffill"}:
        out = tmp.resample(rule).last().ffill().bfill()
    else:
        out = tmp.resample(rule).interpolate("time").ffill().bfill()
    out = out.reset_index()
    return out


def _load_source_summary(source_cfg: dict, ckpt_path: str) -> dict:
    candidates = []
    if source_cfg.get("summary_path") is not None:
        candidates.append(Path(str(source_cfg["summary_path"])))
    candidates.append(Path(ckpt_path).with_name("run_summary.json"))
    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    return {}


def _load_residual_scales(
    source_summary: dict,
    channel_names: list[str],
    device: torch.device,
) -> torch.Tensor:
    selection = source_summary.get("moe_residual_selection", {}) or {}
    source_channels = list(selection.get("residual_channels") or [])
    scale_values = list(selection.get("scale_values") or [])
    mean_scale = float(selection.get("mean_scale", 1.0) or 1.0)
    scale_by_name = {
        str(name): float(scale)
        for name, scale in zip(source_channels, scale_values)
    }
    scales = [scale_by_name.get(str(name), mean_scale) for name in channel_names]
    return torch.tensor(scales, dtype=torch.float32, device=device)


def _build_moe_modules(
    ckpt: dict,
    meta: dict,
    device: torch.device,
) -> tuple[ClusterwiseMoEGate | None, ClusterwisePredResidualMoE | None, list[str]]:
    penalty_names = list(meta.get("penalty_names", []) or [])
    moe_cfg = dict(meta.get("moe_cfg", {}) or {})
    if not bool(moe_cfg.get("enable", True)) or len(penalty_names) == 0:
        return None, None, penalty_names

    gate_state = ckpt.get("gate_state", None)
    if gate_state is None:
        return None, None, penalty_names

    k_count = int(meta["K"])
    gate_feat_dim = int(meta.get("gate_feat_dim", 10))
    allow_skip = any(str(name).startswith("W_skip.") for name in gate_state.keys())
    gate = ClusterwiseMoEGate(
        num_clusters=k_count,
        feat_dim=gate_feat_dim,
        num_penalties=len(penalty_names),
        hidden_dim=int(moe_cfg.get("gate_hidden_dim", moe_cfg.get("hidden_dim", 64))),
        topk=int(moe_cfg.get("topk", 2)),
        allow_skip=allow_skip,
        skip_init_bias=float(moe_cfg.get("skip_init_bias", -2.0)),
    ).to(device)
    gate.temperature = float(moe_cfg.get("gate_temperature", 1.0))
    gate.noise_std = float(moe_cfg.get("gate_noise_std", 0.0))
    gate.logit_clip = float(moe_cfg.get("gate_logit_clip", 0.0))
    gate.prob_floor = float(moe_cfg.get("gate_prob_floor", 0.0))
    gate.load_state_dict(gate_state, strict=True)
    gate.eval()

    pred_state = ckpt.get("pred_residual_state", None)
    pred_cfg = dict((moe_cfg.get("pred_side_residual", {}) or {}))
    pred_residual = None
    if pred_state is not None and bool(pred_cfg.get("enable", False)):
        pred_residual = ClusterwisePredResidualMoE(
            num_clusters=k_count,
            num_penalties=len(penalty_names),
            input_len=int(meta["input_len"]),
            pred_len=int(meta["pred_len"]),
            hidden_dim=int(pred_cfg.get("corrector_hidden", 32)),
            init_alpha=float(pred_cfg.get("init_alpha", -3.0)),
            alpha_scale=float(pred_cfg.get("alpha_scale", 0.5)),
            use_y_base_input=bool(pred_cfg.get("use_y_base_input", True)),
            feature_mode=str(pred_cfg.get("feature_mode", "legacy")),
            residual_clip=float(pred_cfg.get("residual_clip", 0.0)),
            intervention_enable=bool(pred_cfg.get("intervention_enable", False)),
            intervention_init=float(pred_cfg.get("intervention_init", -2.0)),
            penalty_selector_enable=bool(pred_cfg.get("penalty_selector_enable", False)),
            selector_temperature=float(pred_cfg.get("selector_temperature", 1.0)),
            selector_use_cluster_context=bool(pred_cfg.get("selector_use_cluster_context", True)),
            fusion_gate_enable=bool(pred_cfg.get("fusion_gate_enable", False)),
            fusion_init=float(pred_cfg.get("fusion_init", 0.0)),
            fusion_use_cluster_context=bool(pred_cfg.get("fusion_use_cluster_context", True)),
            penalty_names=penalty_names,
            seasonal_anchor_names=list(pred_cfg.get("seasonal_anchor_names", [])),
            seasonal_anchor_period=int(pred_cfg.get("seasonal_anchor_period", 96)),
            seasonal_anchor_num_periods=int(pred_cfg.get("seasonal_anchor_num_periods", 1)),
            seasonal_anchor_scale=float(pred_cfg.get("seasonal_anchor_scale", 1.0)),
        ).to(device)
        pred_residual.load_state_dict(pred_state, strict=True)
        pred_residual.eval()
    return gate, pred_residual, penalty_names


def _predict_with_optional_residual(
    *,
    model: torch.nn.Module,
    gate: ClusterwiseMoEGate | None,
    pred_residual: ClusterwisePredResidualMoE | None,
    x: torch.Tensor,
    cluster_id_c: torch.Tensor,
    meta: dict,
    residual_scale_c: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    yhat_base = model(x, cluster_id_c)
    if gate is None or pred_residual is None:
        return yhat_base, yhat_base

    moe_cfg = dict(meta.get("moe_cfg", {}) or {})
    k_count = int(meta["K"])
    feat_bcf = extract_gate_features(x)
    feat_bkf = scatter_mean_bcf_to_bkf(feat_bcf, cluster_id_c, k_count)
    mask_bkp, probs_bkp, skip_bk, _ = gate(feat_bkf, straight_through=False)
    raw_ranks = moe_cfg.get("select_ranks", None)
    if raw_ranks is not None:
        select_ranks = [int(v) for v in raw_ranks]
        mask_bkp = _select_rank_mask(probs_bkp, select_ranks, straight_through=False)
    gate_soft_weight = float(moe_cfg.get("gate_soft_weight", 0.0))
    if gate_soft_weight > 0.0:
        target_mass = mask_bkp.detach().sum(dim=-1, keepdim=True).clamp_min(1.0)
        probs_sel = probs_bkp * target_mass
        mask_bkp = (1.0 - gate_soft_weight) * mask_bkp + gate_soft_weight * probs_sel
    allow_skip = bool(moe_cfg.get("allow_skip", False))
    pred_out = pred_residual(
        x,
        yhat_base,
        cluster_id_c,
        mask_bkp,
        skip_bk=skip_bk if allow_skip else None,
    )
    yhat = pred_out["y_final"]
    if residual_scale_c is not None:
        scale = residual_scale_c.to(device=yhat.device, dtype=yhat.dtype).view(1, -1, 1)
        yhat = yhat_base + scale * (yhat - yhat_base)
    return yhat_base, yhat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out_dir = cfg["exp"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    set_seed(
        int(cfg["exp"]["seed"]),
        deterministic=bool(cfg.get("exp", {}).get("deterministic", False)),
    )
    device = torch.device(cfg["exp"]["device"] if torch.cuda.is_available() else "cpu")

    source_cfg = cfg["source"]
    memory_path = str(source_cfg["memory_path"])
    ckpt_path = str(source_cfg["checkpoint_path"])
    source_summary = _load_source_summary(source_cfg, ckpt_path)

    memory = load_cluster_memory(memory_path, device=device)
    prototypes_kt = memory["prototypes_kt"]
    K = int(prototypes_kt.shape[0])

    ckpt = load_cluster_checkpoint(ckpt_path, device=device)
    meta = ckpt.get("meta", {})
    if len(meta) == 0:
        raise ValueError("Checkpoint meta is missing. Re-train with memory.save_checkpoint enabled.")

    input_len = int(meta["input_len"])
    pred_len = int(meta["pred_len"])
    model_cfg = meta["model_cfg"]

    win_cfg = cfg["window"]
    if int(win_cfg["input_len"]) != input_len or int(win_cfg["pred_len"]) != pred_len:
        raise ValueError("Window config does not match checkpoint input_len/pred_len.")

    model = build_cluster_predictor(
        num_clusters=K,
        input_len=input_len,
        pred_len=pred_len,
        model_cfg=model_cfg,
        num_channels=meta.get("num_channels", None),
        cluster_id_c=meta.get("cluster_id_c", None),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    transfer_cfg = cfg.get("transfer", {})
    gate, pred_residual, penalty_names = _build_moe_modules(ckpt, meta, device)
    if not bool(transfer_cfg.get("use_pred_residual", True)):
        pred_residual = None
    predictor_variant = "full_moe_residual" if pred_residual is not None else "base"

    data_csv = cfg["data"]["csv_path"]
    date_col_idx = int(cfg["data"]["date_col"])
    raw_df = pd.read_csv(data_csv)
    max_rows = int(cfg.get("data", {}).get("max_rows", 0) or 0)
    if max_rows > 0:
        raw_df = raw_df.iloc[:max_rows].copy()
    date_col = raw_df.columns[date_col_idx]

    resample_cfg = transfer_cfg.get("resample", {})
    if bool(resample_cfg.get("enable", False)):
        target_step_min = resample_cfg.get("target_step_minutes", None)
        if target_step_min is None:
            target_step_min = cfg.get("source", {}).get("step_minutes", None)
        if target_step_min is None:
            src_csv = cfg.get("source", {}).get("csv_path", None)
            src_date_col = int(cfg.get("source", {}).get("date_col", 0))
            if src_csv is not None:
                src_df = pd.read_csv(src_csv, nrows=2000)
                target_step_min = _infer_step_minutes(src_df, src_df.columns[src_date_col])
        target_step_min = int(target_step_min) if target_step_min is not None else 0
        method = resample_cfg.get("method", None)
        if method is None:
            cur_step = _infer_step_minutes(raw_df, date_col)
            if cur_step > 0 and target_step_min > cur_step:
                method = "mean"
            else:
                method = "linear"
        method = str(method).lower()
        raw_df = _resample_df(raw_df, date_col, target_step_min, method)

    data_tc, channel_names = _df_to_tensor(raw_df, date_col)
    data_tc = data_tc.to(device)
    residual_scale_c = (
        _load_residual_scales(source_summary, channel_names, device=device)
        if pred_residual is not None
        else None
    )

    T, C = data_tc.shape
    tr = float(cfg["data"]["train_ratio"])
    vr = float(cfg["data"]["val_ratio"])
    te = float(cfg["data"]["test_ratio"])
    assert abs(tr + vr + te - 1.0) < 1e-6

    t_train = int(T * tr)
    t_val = int(T * (tr + vr))

    norm_cfg = cfg.get("normalize", {})
    if bool(norm_cfg.get("global_zscore", False)):
        if norm_cfg.get("train_only", False):
            train_seg = data_tc[:t_train]
            mean_c = train_seg.mean(dim=0, keepdim=True)
            std_c = train_seg.std(dim=0, keepdim=True).clamp_min(1e-6)
            data_tc = (data_tc - mean_c) / std_c
        else:
            data_tc, _, _ = global_zscore(data_tc)

    route_fit_scope = str(transfer_cfg.get("route_fit_scope", "train")).lower()
    if route_fit_scope == "train":
        route_end = t_train
        route_start = 0
    elif route_fit_scope == "val":
        route_start = t_train
        route_end = t_val
    elif route_fit_scope in {"train_val", "train+val", "pre_test"}:
        route_start = 0
        route_end = t_val
        route_fit_scope = "train_val"
    elif route_fit_scope in {"all", "full"}:
        route_start = 0
        route_end = T
        route_fit_scope = "all"
    else:
        raise ValueError(
            "transfer.route_fit_scope must be 'train', 'val', 'train_val', or 'all'."
        )
    route_data_tc = data_tc[route_start:route_end].contiguous()
    if route_data_tc.shape[0] == 0:
        raise ValueError("No samples available for transfer channel matching.")

    align = str(transfer_cfg.get("corr_align", "head")).lower()
    corr_mode = str(transfer_cfg.get("corr_mode", "cycle_template")).lower()
    resample_enabled = bool(resample_cfg.get("enable", False))
    resample_method = str(resample_cfg.get("method", "auto")).lower()
    normalize_train_only = bool(norm_cfg.get("train_only", False))
    fixed_cluster_id_cfg = transfer_cfg.get("fixed_cluster_id", None)
    route_uses_train_only = route_fit_scope == "train" and fixed_cluster_id_cfg is None
    eval_cfg = cfg.get("eval", {})
    eval_split = str(eval_cfg.get("split", "test")).lower()
    if eval_split == "validation":
        eval_split = "val"
    if eval_split not in {"val", "test"}:
        raise ValueError("eval.split must be 'val' or 'test'.")
    eval_uses_test_only = eval_split == "test"
    resample_is_causal = (not resample_enabled) or resample_method in {"last", "ffill"}
    print(
        "Transfer audit: "
        f"input_len={input_len}, pred_len={pred_len}, "
        f"predictor_variant={predictor_variant}"
    )
    print(
        "Leakage audit: "
        f"normalize_train_only={normalize_train_only}, "
        f"route_fit_scope={route_fit_scope}, route_uses_train_only={route_uses_train_only}, "
        f"eval_split={eval_split}, eval_uses_test_only={eval_uses_test_only}, "
        f"resample_enable={resample_enabled}, resample_method={resample_method}, "
        f"resample_causal={resample_is_causal}"
    )
    print(
        "Split audit: "
        f"T={T}, C={C}, train=[0,{t_train}), val=[{t_train},{t_val}), "
        f"test=[{t_val},{T}), route_range=[{route_start},{route_end})"
    )
    if corr_mode in {"cycle", "cycle_template", "phase", "phase_template"}:
        phase_bins = int(transfer_cfg.get("phase_bins", 64))
        period_min = transfer_cfg.get("period_min", None)
        period_max = transfer_cfg.get("period_max", None)
        period_min_h = transfer_cfg.get("period_min_hours", None)
        period_max_h = transfer_cfg.get("period_max_hours", None)
        phase_max_shift = transfer_cfg.get("phase_max_shift", None)
        period_min = int(period_min) if period_min is not None else None
        period_max = int(period_max) if period_max is not None else None
        if period_min_h is not None or period_max_h is not None:
            step_min = _infer_step_minutes(raw_df, date_col)
            if step_min <= 0:
                step_min = 60.0
            if period_min_h is not None:
                period_min = int(round(float(period_min_h) * 60.0 / step_min))
            if period_max_h is not None:
                period_max = int(round(float(period_max_h) * 60.0 / step_min))
        phase_max_shift = int(phase_max_shift) if phase_max_shift is not None else None
        fixed_cluster_id = fixed_cluster_id_cfg
        if fixed_cluster_id is not None:
            cluster_id_c = torch.tensor(fixed_cluster_id, device=device, dtype=torch.long)
            if int(cluster_id_c.numel()) != C:
                raise ValueError(f"transfer.fixed_cluster_id must have {C} entries.")
            corr_ck = torch.zeros((C, K), device=device, dtype=prototypes_kt.dtype)
            best_tau_ck = None
        else:
            cluster_id_c, corr_ck, best_tau_ck = assign_channels_by_cycle_template(
                route_data_tc,
                prototypes_kt,
                phase_bins=phase_bins,
                period_min=period_min,
                period_max=period_max,
                align=align,
                phase_max_shift=phase_max_shift,
            )
    else:
        fixed_cluster_id = fixed_cluster_id_cfg
        if fixed_cluster_id is not None:
            cluster_id_c = torch.tensor(fixed_cluster_id, device=device, dtype=torch.long)
            if int(cluster_id_c.numel()) != C:
                raise ValueError(f"transfer.fixed_cluster_id must have {C} entries.")
            corr_ck = torch.zeros((C, K), device=device, dtype=prototypes_kt.dtype)
            best_tau_ck = None
        else:
            max_lag = int(transfer_cfg.get("corr_max_lag", 0))
            cluster_id_c, corr_ck = assign_channels_by_corr(
                route_data_tc, prototypes_kt, align=align, max_lag=max_lag
            )
            best_tau_ck = None

    cluster_balance_repair_summary = None
    repair_cfg_raw = transfer_cfg.get("cluster_balance_repair", {})
    if isinstance(repair_cfg_raw, bool):
        repair_cfg = {"enable": bool(repair_cfg_raw)}
    else:
        repair_cfg = dict(repair_cfg_raw or {})
    repair_enabled = bool(repair_cfg.get("enable", False))
    if repair_enabled and fixed_cluster_id_cfg is None:
        pre_repair_cluster_id_c = cluster_id_c.clone()
        pre_counts_k = torch.bincount(pre_repair_cluster_id_c, minlength=K)[:K]
        active_clusters = int((pre_counts_k > 0).sum().item())
        dominant_frac = float(pre_counts_k.max().item() / max(int(C), 1))
        min_unique_clusters = int(repair_cfg.get("min_unique_clusters", 0) or 0)
        max_dominant_frac_cfg = repair_cfg.get("max_dominant_frac", None)
        has_trigger = min_unique_clusters > 0 or max_dominant_frac_cfg is not None
        should_repair = not has_trigger
        if min_unique_clusters > 0 and active_clusters < min_unique_clusters:
            should_repair = True
        if max_dominant_frac_cfg is not None and dominant_frac > float(max_dominant_frac_cfg):
            should_repair = True

        source_cluster_id_c = memory.get("cluster_id_c", None)
        if should_repair:
            if source_cluster_id_c is None:
                raise ValueError("transfer.cluster_balance_repair requires source memory cluster_id_c.")
            target_counts_k = cluster_count_targets_from_source(
                source_cluster_id_c,
                num_clusters=K,
                num_target_channels=C,
                device=device,
            )
            repaired_cluster_id_c = balance_cluster_assignment_by_source_counts(
                corr_ck,
                source_cluster_id_c.to(device=device),
                max_exact_states=int(repair_cfg.get("max_exact_states", 200000)),
            )
            post_counts_k = torch.bincount(repaired_cluster_id_c, minlength=K)[:K]
            idx_c = torch.arange(C, device=device)
            pre_score = float(corr_ck[idx_c, pre_repair_cluster_id_c].sum().item())
            post_score = float(corr_ck[idx_c, repaired_cluster_id_c].sum().item())
            changed = not torch.equal(pre_repair_cluster_id_c, repaired_cluster_id_c)
            cluster_id_c = repaired_cluster_id_c
            cluster_balance_repair_summary = {
                "enabled": True,
                "applied": True,
                "changed": bool(changed),
                "trigger_active_clusters": active_clusters,
                "trigger_dominant_frac": dominant_frac,
                "min_unique_clusters": min_unique_clusters,
                "max_dominant_frac": max_dominant_frac_cfg,
                "target_counts": [int(v) for v in target_counts_k.detach().cpu().tolist()],
                "pre_counts": [int(v) for v in pre_counts_k.detach().cpu().tolist()],
                "post_counts": [int(v) for v in post_counts_k.detach().cpu().tolist()],
                "pre_cluster_id": [int(v) for v in pre_repair_cluster_id_c.detach().cpu().tolist()],
                "post_cluster_id": [int(v) for v in repaired_cluster_id_c.detach().cpu().tolist()],
                "pre_corr_score": pre_score,
                "post_corr_score": post_score,
                "corr_score_delta": post_score - pre_score,
            }
            print(
                "Cluster balance repair: "
                f"pre_counts={cluster_balance_repair_summary['pre_counts']}, "
                f"target_counts={cluster_balance_repair_summary['target_counts']}, "
                f"post_counts={cluster_balance_repair_summary['post_counts']}, "
                f"changed={changed}"
            )
        else:
            cluster_balance_repair_summary = {
                "enabled": True,
                "applied": False,
                "trigger_active_clusters": active_clusters,
                "trigger_dominant_frac": dominant_frac,
                "min_unique_clusters": min_unique_clusters,
                "max_dominant_frac": max_dominant_frac_cfg,
                "pre_counts": [int(v) for v in pre_counts_k.detach().cpu().tolist()],
            }

    corr_max_t = corr_ck.max(dim=1).values
    corr_selected_t = corr_ck[torch.arange(C, device=device), cluster_id_c]
    corr_max = corr_max_t.detach().cpu().numpy()
    corr_selected = corr_selected_t.detach().cpu().numpy()

    corr_threshold = transfer_cfg.get("corr_threshold", None)
    fallback_mode = str(transfer_cfg.get("fallback_mode", "hard")).lower()
    fallback_topk = int(transfer_cfg.get("fallback_topk", 2))
    fallback_temp = float(transfer_cfg.get("fallback_temp", 1.0))

    use_soft_assignment = False
    w_ck = None
    low_mask = None
    if corr_threshold is not None and fallback_mode in {"soft", "ensemble", "topk"}:
        threshold = float(corr_threshold)
        low_mask = corr_max_t < threshold
        if low_mask.any():
            use_soft_assignment = True
            k_sel = max(1, min(fallback_topk, K))
            topv, topi = torch.topk(corr_ck, k=k_sel, dim=1)
            temp = max(fallback_temp, 1.0e-6)
            weights = torch.softmax(topv / temp, dim=1)
            w_ck = torch.zeros((C, K), device=device, dtype=corr_ck.dtype)
            w_ck.scatter_(1, cluster_id_c.view(-1, 1), 1.0)
            w_ck_low = torch.zeros((int(low_mask.sum().item()), K), device=device, dtype=corr_ck.dtype)
            w_ck_low.scatter_(1, topi[low_mask], weights[low_mask])
            w_ck[low_mask] = w_ck_low
        else:
            w_ck = None
    if low_mask is not None and use_soft_assignment and pred_residual is not None:
        print("Leakage audit: soft_fallback_disabled_for_full_moe_residual=True")

    L = input_len
    H = pred_len
    if eval_split == "val":
        eval_label_start, eval_end = t_train, t_val
    else:
        eval_label_start, eval_end = t_val, T
    past_context = bool(cfg.get("window", {}).get("past_context", False))
    eval_start = max(0, int(eval_label_start) - L) if past_context else int(eval_label_start)
    eval_seg = data_tc[eval_start:eval_end]
    num_eval_windows = int(eval_seg.shape[0] - L - H + 1)
    if num_eval_windows <= 0:
        raise ValueError(f"No {eval_split} windows available for transfer evaluation.")
    print(
        f"{eval_split.capitalize()} windows: n={num_eval_windows}, "
        f"range=[{eval_start},{eval_end}), label_start={eval_label_start}, "
        f"past_context={past_context}, L={L}, H={H}"
    )

    bs = int(eval_cfg.get("batch_size", 64))

    se_c = torch.zeros(C, device=device)
    ae_c = torch.zeros(C, device=device)
    denom = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        def stream_test_batches():
            for start in range(0, num_eval_windows, bs):
                end = min(start + bs, num_eval_windows)
                xs = []
                ys = []
                for i in range(start, end):
                    win = eval_seg[i : i + L + H]
                    xs.append(win[:L].T)
                    ys.append(win[L:].T)
                idx = torch.arange(start, end, device=device, dtype=torch.long)
                yield torch.stack(xs, dim=0), torch.stack(ys, dim=0), idx

        for x, y, idx in stream_test_batches():
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if use_soft_assignment and w_ck is not None and pred_residual is None:
                yhat = torch.zeros((x.shape[0], C, H), device=device, dtype=x.dtype)
                for k in range(K):
                    cid_k = torch.full((C,), k, device=device, dtype=torch.long)
                    yhat_k = model(x, cid_k)
                    yhat = yhat + yhat_k * w_ck[:, k].view(1, C, 1)
            else:
                _, yhat = _predict_with_optional_residual(
                    model=model,
                    gate=gate,
                    pred_residual=pred_residual,
                    x=x,
                    cluster_id_c=cluster_id_c,
                    meta=meta,
                    residual_scale_c=residual_scale_c,
                )
            accumulate_channel_errors(se_c, ae_c, yhat, y)
            denom += int(x.shape[0] * y.shape[2])

    mse_c, mae_c = mse_mae_from_sums(se_c, ae_c, denom)

    df = pd.DataFrame({
        "channel": channel_names,
        "MAE": mae_c.detach().cpu().numpy(),
        "MSE": mse_c.detach().cpu().numpy(),
        "cluster_id": cluster_id_c.detach().cpu().numpy(),
        "corr_max": corr_max,
        "corr_selected": corr_selected,
    })
    metrics_name = "test_metrics.csv" if eval_split == "test" else "val_metrics.csv"
    df.to_csv(os.path.join(out_dir, metrics_name), index=False)

    assign_payload = {
        "channel": channel_names,
        "cluster_id": cluster_id_c.detach().cpu().numpy(),
        "corr_max": corr_max,
        "corr_selected": corr_selected,
    }
    if best_tau_ck is not None:
        tau_best = best_tau_ck[torch.arange(C, device=device), cluster_id_c].detach().cpu().numpy()
        assign_payload["best_tau"] = tau_best
    if low_mask is not None:
        assign_payload["low_corr"] = low_mask.detach().cpu().numpy()
    assign_df = pd.DataFrame(assign_payload)
    assign_df.to_csv(os.path.join(out_dir, "cluster_assignment.csv"), index=False)

    if bool(transfer_cfg.get("save_corr", True)):
        np.save(os.path.join(out_dir, "cluster_corr.npy"), corr_ck.detach().cpu().numpy())

    avg_mae = float(df["MAE"].mean())
    avg_mse = float(df["MSE"].mean())
    summary = {
        "avg_mae": avg_mae,
        "avg_mse": avg_mse,
        "elapsed_sec": float(time.perf_counter() - t0),
        "num_test_windows": int(num_eval_windows) if eval_split == "test" else 0,
        "num_eval_windows": int(num_eval_windows),
        "num_channels": int(C),
        "eval_split": eval_split,
        "eval_start_index": int(eval_start),
        "eval_label_start_index": int(eval_label_start),
        "eval_end_index": int(eval_end),
        "past_context": bool(past_context),
        "data_max_rows": int(max_rows),
        "route_fit_scope": route_fit_scope,
        "route_fit_start_index": int(route_start),
        "route_fit_end_index": int(route_end),
        "corr_mode": corr_mode,
        "corr_align": align,
        "normalize_train_only": normalize_train_only,
        "route_uses_train_only": route_uses_train_only,
        "eval_uses_test_only": eval_uses_test_only,
        "resample_enable": resample_enabled,
        "resample_method": resample_method,
        "resample_causal": resample_is_causal,
        "predictor_variant": predictor_variant,
        "use_pred_residual": bool(pred_residual is not None),
        "source_checkpoint": ckpt_path,
        "source_memory": memory_path,
        "penalty_names": penalty_names,
        "cluster_id": [int(v) for v in cluster_id_c.detach().cpu().tolist()],
        "corr_max": [float(v) for v in corr_max],
        "corr_selected": [float(v) for v in corr_selected],
        "corr_mean": float(np.mean(corr_selected)) if len(corr_selected) > 0 else 0.0,
        "corr_max_mean": float(np.mean(corr_max)) if len(corr_max) > 0 else 0.0,
    }
    if cluster_balance_repair_summary is not None:
        summary["cluster_balance_repair"] = cluster_balance_repair_summary
    if fixed_cluster_id_cfg is not None:
        summary["fixed_cluster_id"] = [int(v) for v in fixed_cluster_id_cfg]
        summary["route_selection"] = "fixed_cluster_id"
    else:
        print(f"Transfer {eval_split}: avg_MAE={avg_mae:.6f}, avg_MSE={avg_mse:.6f}")

    with open(os.path.join(out_dir, "transfer_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved transfer outputs to: {out_dir}")
    print(f"Elapsed: {summary['elapsed_sec']:.3f}s")


if __name__ == "__main__":
    main()
