"""PKR-MoE training CLI and run orchestration."""
from __future__ import annotations

import os
import json
import argparse
import time
import math
import sys
import builtins
import hashlib
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
import pandas as pd
try:
    from torch.func import functional_call as _torch_functional_call

    def _module_call(module: nn.Module, params: Optional[Dict[str, torch.Tensor]], *args, **kwargs):
        if params is None:
            return module(*args, **kwargs)
        return _torch_functional_call(module, params, args=args, kwargs=kwargs)
except Exception:
    from torch.nn.utils.stateless import functional_call as _torch_stateless_functional_call

    def _module_call(module: nn.Module, params: Optional[Dict[str, torch.Tensor]], *args, **kwargs):
        if params is None:
            return module(*args, **kwargs)
        return _torch_stateless_functional_call(module, params, args, kwargs)
try:
    import psutil  # type: ignore
except Exception:
    psutil = None

from .utils.yaml_io import load_yaml
from .utils.seed import set_seed
from .data.reader import read_csv_time_series
from .data.windows import (
    WindowTensorDataset,
    global_zscore,
    make_label_range_windows,
    make_lazy_label_range_window_dataset,
    make_lazy_strict_window_dataset,
    make_strict_windows,
)
from .utils.pearson import pearson_corr_matrix
from .utils.clustering import cluster_channels_by_corr
from .models.cluster_predictor import build_cluster_predictor
from .models.dynamic_lambda import ClusterwiseDynamicLambda
from .models.learnable_anchor import ClusterwiseLearnableOutputAnchor
from .models.learnable_lambda import ClusterwiseLearnableLambda
from .models.moe_gate import ClusterwiseMoEGate, scatter_mean_bc_to_bk, scatter_mean_bcf_to_bkf
from .models.penalties import build_penalty_bank, normalize_penalties
from .models.residual_moe import ClusterwisePredResidualMoE
from .utils.plotting import save_channel_plots, save_cluster_metric_curves
from .utils.cluster_portrait import save_cluster_portraits
from .utils.cluster_memory import (
    OnlineClusterMemory,
    compute_cluster_prototypes,
    scatter_mean_bcl_to_bkl,
    save_cluster_memory,
    save_cluster_checkpoint,
    load_cluster_memory,
    load_cluster_checkpoint,
)
from .utils.console_progress import PurpleProgressBar
from .utils.diagnostic_sampling import select_prediction_sample_indices



# Preserve the historical ``src.train`` helper API for experiment scripts.
from .train_support import *  # noqa: F401,F403

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    final_print = builtins.print
    cfg = load_yaml(args.config)
    if bool(cfg.get("console", {}).get("quiet", True)) and sys.stdout.isatty():
        builtins.print = lambda *args, **kwargs: None

    out_dir = cfg["exp"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    t_all0 = time.perf_counter()
    set_seed(
        int(cfg["exp"]["seed"]),
        deterministic=bool(cfg.get("exp", {}).get("deterministic", False)),
    )
    device = torch.device(cfg["exp"]["device"] if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # 1) 璇绘暟鎹?& 璁板綍閫氶亾鍚嶏紙璺宠繃 date 鍒楋紱header 涓嶈繘鍏ユ暟鎹級
    data_cfg = cfg["data"]
    data_tc, channel_names = read_csv_time_series(data_cfg["csv_path"], date_col=int(data_cfg["date_col"]))
    max_rows = int(data_cfg.get("max_rows", 0) or 0)
    if max_rows > 0:
        data_tc = data_tc[:max_rows]
    data_tc = data_tc.to(device)

    T, C = data_tc.shape
    print(f"Loaded data: T={T}, C={C}")

    tr = float(cfg["data"]["train_ratio"])
    vr = float(cfg["data"]["val_ratio"])
    te = float(cfg["data"]["test_ratio"])
    assert abs(tr + vr + te - 1.0) < 1e-6

    t_train = int(T * tr)
    t_val = int(T * (tr + vr))

    # 2) Normalize the time series.
    norm_cfg = cfg["normalize"]
    if norm_cfg["global_zscore"]:
        if norm_cfg.get("train_only", False):
            train_seg = data_tc[:t_train]
            mean_c = train_seg.mean(dim=0, keepdim=True)
            std_c = train_seg.std(dim=0, keepdim=True).clamp_min(1e-6)
            data_tc = (data_tc - mean_c) / std_c
            mean_c = mean_c.squeeze(0)
            std_c = std_c.squeeze(0)
        else:
            data_tc, mean_c, std_c = global_zscore(data_tc)

    # 3) corr matrix (skip when using random grouping)
    cl = cfg["cluster"]
    method_norm = str(cl.get("method", "agglomerative")).lower()
    cluster_fit_tc = data_tc[:t_train] if bool(cl.get("train_only", True)) else data_tc
    if bool(cl.get("train_only", True)):
        print("Cluster fit uses train split only.")
    if method_norm in {"random", "rand"}:
        C = int(data_tc.shape[1])
        corr_cc = torch.eye(C, device=data_tc.device, dtype=data_tc.dtype)
        if cfg["corr"]["compute"]:
            print("Skip corr matrix compute: cluster.method=random")
    else:
        corr_cc = pearson_corr_matrix(cluster_fit_tc)
        if cfg["corr"]["compute"]:
            save_path = cfg["corr"]["save_path"]
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, corr_cc.detach().cpu().numpy())
            print(f"Saved corr matrix to {save_path} (shape {corr_cc.shape})")
    feature_aware_cfg = cl.get("feature_aware", {}) or {}
    cluster_extra_features_cf = None
    if bool(feature_aware_cfg.get("enable", False)):
        raw_lags = feature_aware_cfg.get("acf_lags", [1, 24, 96])
        if raw_lags is None:
            acf_lags = []
        elif isinstance(raw_lags, (list, tuple)):
            acf_lags = [int(v) for v in raw_lags]
        else:
            acf_lags = [int(raw_lags)]
        cluster_extra_features_cf = compute_channel_shape_features(cluster_fit_tc, acf_lags=acf_lags)
        print(
            "Feature-aware clustering enabled: "
            f"feature_weight={float(feature_aware_cfg.get('weight', 0.0)):.3f}, "
            f"features={int(cluster_extra_features_cf.shape[1])}, acf_lags={acf_lags}"
        )

    # 4) 鑱氱被 + 灏忕皣鍚堝苟绛栫暐
    fixed_cluster_id = cl.get("fixed_cluster_id", None)
    if fixed_cluster_id is not None:
        cluster_id_c = torch.tensor(fixed_cluster_id, dtype=torch.long, device=device)
        if int(cluster_id_c.numel()) != C:
            raise ValueError(
                f"cluster.fixed_cluster_id must contain one id per channel: "
                f"got {int(cluster_id_c.numel())}, expected {C}."
            )
        if int(cluster_id_c.min().item()) < 0:
            raise ValueError("cluster.fixed_cluster_id must be non-negative.")
        # Preserve ids so transfer/fine-tune can map target channels directly
        # onto the corresponding source cluster heads.
        clusters = {
            int(k): (cluster_id_c == int(k)).nonzero(as_tuple=False).view(-1).detach().cpu().tolist()
            for k in range(int(cluster_id_c.max().item()) + 1)
        }
        print("Using fixed channel cluster assignment from cluster.fixed_cluster_id.")
    else:
        rs = cl.get("random_state", 0)
        cluster_id_c, clusters = cluster_channels_by_corr(
            corr_cc=corr_cc,
            data_tc=cluster_fit_tc,
            n_clusters=cl.get("n_clusters", None),
            distance_threshold=cl.get("distance_threshold", None),
            linkage=cl.get("linkage", "average"),
            method=cl.get("method", "agglomerative"),
            kmeans_n_init=int(cl.get("kmeans_n_init", 10)),
            kmeans_max_iter=int(cl.get("kmeans_max_iter", 300)),
            spectral_affinity=cl.get("spectral_affinity", "corr"),
            rbf_gamma=float(cl.get("rbf_gamma", 1.0)),
            dbscan_eps=cl.get("dbscan_eps", None),
            dbscan_min_samples=int(cl.get("dbscan_min_samples", 5)),
            random_state=None if rs is None else int(rs),
            min_cluster_size=int(cl["min_cluster_size"]),
            merge_small_clusters=bool(cl["merge_small_clusters"]),
            singleton_merge_strategy=str(cl.get("singleton_merge_strategy", "pool")),
            singleton_merge_distance_threshold=cl.get("singleton_merge_distance_threshold", None),
            singleton_merge_min_size=int(cl.get("singleton_merge_min_size", 2)),
            no_merge_if_channels_lt=int(cl["no_merge_if_channels_lt"]),
            extra_features_cf=cluster_extra_features_cf,
            feature_weight=float(feature_aware_cfg.get("weight", 0.0)) if cluster_extra_features_cf is not None else 0.0,
        )
    K = int(cluster_id_c.max().item() + 1)
    print(f"Clusters: K={K}")
    print_clusters(clusters, channel_names)
    cluster_sizes = torch.bincount(cluster_id_c, minlength=K).tolist()
    cluster_weight_k = torch.tensor(cluster_sizes, device=device, dtype=torch.float32)
    cluster_weight_k = cluster_weight_k / cluster_weight_k.sum().clamp_min(1.0)
    print("Cluster sizes: " + ", ".join(f"{k}:{n}" for k, n in enumerate(cluster_sizes)))

    # cluster memory config
    memory_cfg = cfg.get("memory", {})
    memory_enable = bool(memory_cfg.get("enable", False))
    memory_path = str(memory_cfg.get("path", os.path.join(out_dir, "cluster_memory.pt")))

    L = int(cfg["window"]["input_len"])
    H = int(cfg["window"]["pred_len"])
    cfg["moe"] = apply_default_moe_output_anchor_cfg(
        cfg.get("moe", {}) or {},
        dataset_name=cfg.get("data", {}).get("csv_path", ""),
        pred_len=H,
    )
    eval_cfg = cfg.get("eval", {}) or {}
    skip_test = bool(eval_cfg.get("skip_test", True))
    diagnostics_cfg = cfg.get("diagnostics", {}) or {}
    stage2_loss_audit_cfg = diagnostics_cfg.get("stage2_loss_audit", {}) or {}
    stage2_loss_audit_enable = bool(stage2_loss_audit_cfg.get("enable", False))
    stage2_objective_overlap_cfg = stage2_loss_audit_cfg.get(
        "objective_overlap",
        {},
    ) or {}
    if not isinstance(stage2_objective_overlap_cfg, dict):
        stage2_objective_overlap_cfg = {
            "enable": bool(stage2_objective_overlap_cfg)
        }
    stage2_objective_overlap_enable = bool(
        stage2_loss_audit_enable
        and stage2_objective_overlap_cfg.get("enable", False)
    )
    stage2_objective_overlap_max_batches = max(
        1,
        int(stage2_objective_overlap_cfg.get("max_batches", 4)),
    )
    if stage2_loss_audit_enable:
        print("Stage-2 loss audit diagnostics enabled.")
    if stage2_objective_overlap_enable:
        print(
            "Stage-2 gate objective-overlap diagnostics enabled: "
            f"max_batches={stage2_objective_overlap_max_batches}"
        )
    stage2_route_audit_cfg = diagnostics_cfg.get("stage2_route_audit", {}) or {}
    if not isinstance(stage2_route_audit_cfg, dict):
        stage2_route_audit_cfg = {"enable": bool(stage2_route_audit_cfg)}
    stage2_route_audit_enable = bool(stage2_route_audit_cfg.get("enable", False))
    if stage2_route_audit_enable:
        print("Stage-2 route audit diagnostics enabled.")

    # Keep materialized windows on CPU.  Electricity-style datasets with many
    # channels and long horizons can expand to tens of GB; batches are moved to
    # CUDA by the train/eval loops.
    data_window_tc = data_tc.detach().cpu()
    window_cfg = cfg.get("window", {}) or {}
    past_context = bool(window_cfg.get("past_context", False))
    lazy_windows = bool(window_cfg.get("lazy", False))
    history_anchor_cfg = cfg.get("model", {}).get("history_anchor", cfg.get("history_anchor", {})) or {}
    history_anchor_cfg = _normalize_history_anchor_cfg(history_anchor_cfg)
    _validate_strict_history_anchor_scope(history_anchor_cfg, source="model.history_anchor")
    history_anchor_active = history_anchor_enabled(history_anchor_cfg)
    if history_anchor_active:
        print(
            "History anchor adapter enabled: "
            f"lags={_parse_positive_ints(history_anchor_cfg.get('lags', ()))}, "
            f"alpha={float(history_anchor_cfg.get('alpha', 0.0)):.3f}, "
            f"blend_target={str(history_anchor_cfg.get('blend_target', 'prediction')).lower()}, "
            f"history_scope={str(history_anchor_cfg.get('history_scope', 'input_window')).lower()}"
        )
    calendar_residual_cfg = cfg.get("calendar_residual", {}) or {}
    calendar_feature_tf = None
    calendar_feature_names: List[str] = []
    calendar_residual_coef_cf = None
    calendar_residual_summary: Dict[str, object] = {
        "enable": bool(calendar_residual_cfg.get("enable", False)),
    }
    position_daily_residual_cfg = cfg.get("position_daily_residual_ridge", {}) or {}
    if not isinstance(position_daily_residual_cfg, dict):
        position_daily_residual_cfg = {
            "enable": bool(position_daily_residual_cfg)
        }
    position_daily_residual_coef_cfh = None
    position_daily_residual_summary: Dict[str, object] = {
        "enable": bool(position_daily_residual_cfg.get("enable", False)),
    }
    if bool(calendar_residual_cfg.get("enable", False)):
        calendar_feature_tf, calendar_feature_names = build_calendar_feature_tensor(
            data_cfg["csv_path"],
            date_col=int(data_cfg["date_col"]),
            max_rows=max_rows,
            cfg=calendar_residual_cfg,
        )
        calendar_feature_tf = calendar_feature_tf.to(device=device)
        calendar_residual_summary.update(
            {
                "feature_names": list(calendar_feature_names),
                "feature_dim": int(calendar_feature_tf.shape[1]),
                "fit_source": str(calendar_residual_cfg.get("source_split", "train")),
                "train_only": True,
            }
        )
        print(
            "Calendar residual adapter enabled: "
            f"features={calendar_feature_names}, source=train"
        )

    val_eval_start = t_train
    test_eval_start = t_val

    if lazy_windows:
        xtr = ytr = xva = yva = xte = yte = None
        dtr = make_lazy_strict_window_dataset(data_window_tc, L, H, 0, t_train)
        train_start_offsets = dtr.start_offsets.clone()
        if past_context:
            dva, val_eval_start = make_lazy_label_range_window_dataset(data_window_tc, L, H, t_train, t_val)
            if skip_test:
                dte = make_lazy_strict_window_dataset(data_window_tc, L, H, 0, 0)
            else:
                dte, test_eval_start = make_lazy_label_range_window_dataset(data_window_tc, L, H, t_val, T)
        else:
            dva = make_lazy_strict_window_dataset(data_window_tc, L, H, t_train, t_val)
            if skip_test:
                dte = make_lazy_strict_window_dataset(data_window_tc, L, H, 0, 0)
            else:
                dte = make_lazy_strict_window_dataset(data_window_tc, L, H, t_val, T)
    else:
        xtr, ytr = make_strict_windows(data_window_tc, L, H, 0, t_train)
        train_start_offsets = torch.arange(0, len(xtr), dtype=torch.long)
        if past_context:
            xva, yva, val_eval_start = make_label_range_windows(data_window_tc, L, H, t_train, t_val)
            if skip_test:
                xte = torch.empty(0, C, L, dtype=data_window_tc.dtype)
                yte = torch.empty(0, C, H, dtype=data_window_tc.dtype)
            else:
                xte, yte, test_eval_start = make_label_range_windows(data_window_tc, L, H, t_val, T)
        else:
            xva, yva = make_strict_windows(data_window_tc, L, H, t_train, t_val)
            if skip_test:
                xte = torch.empty(0, C, L, dtype=data_window_tc.dtype)
                yte = torch.empty(0, C, H, dtype=data_window_tc.dtype)
            else:
                xte, yte = make_strict_windows(data_window_tc, L, H, t_val, T)
        dtr = WindowTensorDataset(xtr, ytr)
        dva = WindowTensorDataset(xva, yva)
        dte = WindowTensorDataset(xte, yte)

    print(
        f"Windows: train={len(dtr)}, val={len(dva)}, test={len(dte)}, "
        f"past_context={past_context}, lazy={lazy_windows}"
    )

    cluster_memory_bank = None
    if memory_enable:
        cluster_memory_bank = OnlineClusterMemory(
            num_clusters=K,
            memory_len=t_train,
            device=device,
            dtype=data_tc.dtype,
    )

    overfit_diagnostic_cfg = cfg["train"].get("overfit_diagnostic", {}) or {}
    if isinstance(overfit_diagnostic_cfg, bool):
        overfit_diagnostic_cfg = {"enable": bool(overfit_diagnostic_cfg)}
    overfit_diagnostic_range = _resolve_overfit_diagnostic_range(
        len(dtr),
        overfit_diagnostic_cfg,
    )
    optimization_window_cfg = cfg["train"].get("optimization_window", {}) or {}
    if isinstance(optimization_window_cfg, bool):
        optimization_window_cfg = {"enable": bool(optimization_window_cfg)}
    optimization_source = str(
        cfg["train"].get("optimization_source", "train")
    ).strip().lower()
    if optimization_source not in {"train", "val"}:
        raise ValueError("train.optimization_source must be train or val.")
    if optimization_source == "val" and overfit_diagnostic_range is not None:
        raise ValueError(
            "train.optimization_source=val is incompatible with train.overfit_diagnostic."
        )
    optimization_source_dataset = dtr if optimization_source == "train" else dva
    optimization_index_offset = 0 if optimization_source == "train" else int(val_eval_start)
    optimization_window_range = _resolve_overfit_diagnostic_range(
        len(optimization_source_dataset),
        optimization_window_cfg,
        config_name="train.optimization_window",
    )
    if (
        overfit_diagnostic_range is not None
        and optimization_window_range is not None
    ):
        raise ValueError(
            "train.optimization_window and train.overfit_diagnostic cannot both be enabled."
        )
    optimization_dataset = optimization_source_dataset
    if optimization_window_range is not None:
        optimization_start, optimization_end = optimization_window_range
        optimization_dataset = Subset(
            optimization_source_dataset,
            range(optimization_start, optimization_end),
        )
    elif overfit_diagnostic_range is not None:
        overfit_start, overfit_end = overfit_diagnostic_range
        optimization_dataset = Subset(dtr, range(overfit_start, overfit_end))

    bs = int(cfg["train"]["batch_size"])
    pin_mem = (device.type == "cuda") and (data_window_tc.device.type == "cpu")
    shuffle_seed = cfg["train"].get("shuffle_seed", None)
    if shuffle_seed is None and bool(cfg["train"].get("fixed_shuffle_seed", False)):
        shuffle_seed = int(cfg["exp"]["seed"])
    train_generator = _make_torch_generator(None if shuffle_seed is None else int(shuffle_seed))
    dl_tr = DataLoader(
        optimization_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_mem,
        generator=train_generator,
    )
    dl_tr_source = DataLoader(
        dtr,
        batch_size=bs,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_mem,
    )
    dl_overfit_eval = (
        DataLoader(
            optimization_dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_mem,
        )
        if overfit_diagnostic_range is not None
        else None
    )
    dl_va = DataLoader(dva, batch_size=bs, shuffle=False, num_workers=0, pin_memory=pin_mem)
    dl_te = DataLoader(dte, batch_size=bs, shuffle=False, num_workers=0, pin_memory=pin_mem)
    epoch_eval_window_cfg = cfg["train"].get("epoch_eval_window", {}) or {}
    if isinstance(epoch_eval_window_cfg, bool):
        epoch_eval_window_cfg = {"enable": bool(epoch_eval_window_cfg)}
    epoch_eval_window_range = _resolve_overfit_diagnostic_range(
        len(dva),
        epoch_eval_window_cfg,
        config_name="train.epoch_eval_window",
    )
    dl_stage2_epoch_eval = None
    stage2_epoch_eval_start = int(val_eval_start)
    if epoch_eval_window_range is not None:
        epoch_eval_start_idx, epoch_eval_end_idx = epoch_eval_window_range
        dl_stage2_epoch_eval = DataLoader(
            Subset(dva, range(epoch_eval_start_idx, epoch_eval_end_idx)),
            batch_size=bs,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_mem,
        )
        if optimization_source == "val":
            purge_windows = max(
                0,
                int(epoch_eval_window_cfg.get("purge_windows", 0)),
            )
            if optimization_window_range is None:
                raise ValueError(
                    "validation gate optimization with epoch_eval_window requires "
                    "an explicit optimization_window."
                )
            if epoch_eval_start_idx - optimization_end < purge_windows:
                raise ValueError(
                    "validation gate optimization and epoch evaluation overlap or "
                    "violate the configured purge_windows."
                )
    if overfit_diagnostic_range is not None:
        print(
            "Gate overfit diagnostic: "
            f"train_windows=[{overfit_start}:{overfit_end}], "
            f"count={len(optimization_dataset)}, epoch_eval=train_subset"
        )
    elif optimization_window_range is not None:
        print(
            "Stage-2 optimization window: "
            f"source={optimization_source}, "
            f"windows=[{optimization_start}:{optimization_end}], "
            f"count={len(optimization_dataset)}, epoch_eval=validation"
        )
    if epoch_eval_window_range is not None:
        print(
            "Stage-2 epoch evaluation window: "
            f"val_windows=[{epoch_eval_start_idx}:{epoch_eval_end_idx}], "
            f"count={epoch_eval_end_idx - epoch_eval_start_idx}"
        )
    # penalties
    penalty_names = list(cfg["penalties"]["enabled"])
    penalty_fns = build_penalty_bank(penalty_names, jump_thr=float(cfg["penalties"]["jump_threshold"]))
    P = len(penalty_names)
    stage2_route_audit_loaders: Dict[str, DataLoader] = {}
    stage2_route_audit_eval_starts: Dict[str, int] = {}
    stage2_route_audit_train_subsplits: Dict[str, Tuple[int, int]] = {}
    if stage2_route_audit_enable:
        requested_route_splits = [
            str(name).lower()
            for name in (stage2_route_audit_cfg.get("splits", ["train_fit", "train_holdout", "val"]) or [])
        ]
        if "test" in requested_route_splits and not bool(stage2_route_audit_cfg.get("allow_test", False)):
            raise ValueError("diagnostics.stage2_route_audit refuses to read test unless allow_test=true.")
        if len(dtr) > 0 and "train" in requested_route_splits:
            stage2_route_audit_loaders["train"] = DataLoader(
                dtr,
                batch_size=bs,
                shuffle=False,
                num_workers=0,
                pin_memory=pin_mem,
            )
            stage2_route_audit_eval_starts["train"] = 0
        train_subsplit_names = {"train_fit", "train_holdout"}
        if len(dtr) > 0 and any(name in requested_route_splits for name in train_subsplit_names):
            stage2_route_audit_train_subsplits = _explainability_train_subsplit_ranges(
                num_windows=len(dtr),
                holdout_fraction=float(stage2_route_audit_cfg.get("train_holdout_fraction", 0.30)),
            )
            for split_name in ("train_fit", "train_holdout"):
                if split_name not in requested_route_splits:
                    continue
                if split_name not in stage2_route_audit_train_subsplits:
                    continue
                start_i, end_i = stage2_route_audit_train_subsplits[split_name]
                if int(end_i) <= int(start_i):
                    continue
                stage2_route_audit_loaders[split_name] = DataLoader(
                    Subset(dtr, range(int(start_i), int(end_i))),
                    batch_size=bs,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=pin_mem,
                )
                stage2_route_audit_eval_starts[split_name] = 0
        if len(dva) > 0 and "val" in requested_route_splits:
            stage2_route_audit_loaders["val"] = dl_va
            stage2_route_audit_eval_starts["val"] = int(val_eval_start)
        if len(dte) > 0 and "test" in requested_route_splits and bool(stage2_route_audit_cfg.get("allow_test", False)):
            stage2_route_audit_loaders["test"] = dl_te
            stage2_route_audit_eval_starts["test"] = int(test_eval_start)
    mse_weight = float(cfg["train"].get("mse_weight", 1.0))
    mae_objective_cfg = cfg["train"].get("mae_objective", {}) or {}
    mae_objective_enable = bool(mae_objective_cfg.get("enable", False))
    mae_objective_kind = str(mae_objective_cfg.get("kind", "l1")).lower()
    if mae_objective_kind not in {"l1", "mae", "smooth_l1", "huber"}:
        raise ValueError(
            f"Unsupported train.mae_objective.kind='{mae_objective_kind}'. Expected l1 or smooth_l1."
        )
    mae_objective_weight_final = float(mae_objective_cfg.get("weight", 0.0)) if mae_objective_enable else 0.0
    mae_objective_warmup_epochs = int(mae_objective_cfg.get("warmup_epochs", 0)) if mae_objective_enable else 0
    mae_objective_beta = float(mae_objective_cfg.get("beta", 1.0))
    if mae_objective_beta <= 0.0:
        raise ValueError("train.mae_objective.beta must be positive.")
    mae_objective_per_cluster_cfg = mae_objective_cfg.get("per_cluster", {}) or {}
    mae_objective_per_cluster_enable = (
        bool(mae_objective_enable)
        and bool(mae_objective_per_cluster_cfg.get("enable", False))
        and mae_objective_weight_final != 0.0
    )
    mae_objective_multiplier_k: Optional[torch.Tensor] = None
    mae_objective_per_cluster_summary: Dict[str, object] = {
        "enable": bool(mae_objective_per_cluster_enable),
    }

    def mae_objective_weight_at(epoch_idx: int):
        if (not mae_objective_enable) or mae_objective_weight_final == 0.0:
            return 0.0
        if mae_objective_warmup_epochs <= 0:
            base_weight = mae_objective_weight_final
        else:
            scale = min(1.0, max(0.0, float(epoch_idx) / float(mae_objective_warmup_epochs)))
            base_weight = mae_objective_weight_final * scale
        return _scale_mae_objective_weight(base_weight, mae_objective_multiplier_k)

    if mae_objective_per_cluster_enable:
        diagnostic_loader = DataLoader(
            dtr,
            batch_size=bs,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        target_bch = _collect_train_targets_bch(
            diagnostic_loader,
            max_windows=int(mae_objective_per_cluster_cfg.get("max_windows", 0) or 0),
        )
        per_cluster_diag = _build_mae_per_cluster_diagnostics_from_targets(
            targets_bch=target_bch,
            cluster_id_c=cluster_id_c,
            K=K,
            base_weight=mae_objective_weight_final,
            cfg=mae_objective_per_cluster_cfg,
        )
        mae_objective_multiplier_k = per_cluster_diag["multiplier_k"].to(device=device, dtype=torch.float32).detach()
        artifact_name = str(mae_objective_per_cluster_cfg.get("artifact", "cluster_mae_weights.csv"))
        artifact_path = artifact_name if os.path.isabs(artifact_name) else os.path.join(out_dir, artifact_name)
        _save_mae_per_cluster_diagnostics_csv(per_cluster_diag["rows"], artifact_path)
        mae_objective_per_cluster_summary = {
            "enable": True,
            "diagnostic": str(mae_objective_per_cluster_cfg.get("diagnostic", "mean_median_gap")),
            "source": str(mae_objective_per_cluster_cfg.get("source", "train_targets")),
            "normalize": str(mae_objective_per_cluster_cfg.get("normalize", "std")),
            "pivot": mae_objective_per_cluster_cfg.get("pivot", "median"),
            "min_multiplier": float(mae_objective_per_cluster_cfg.get("min_multiplier", 1.0)),
            "max_multiplier": float(mae_objective_per_cluster_cfg.get("max_multiplier", 1.25)),
            "artifact": artifact_path,
            "multiplier": [float(v) for v in mae_objective_multiplier_k.detach().cpu().tolist()],
            "effective_weight": [
                float(v) for v in per_cluster_diag["effective_weight_k"].detach().cpu().tolist()
            ],
        }
        print(f"Saved per-cluster MAE objective weights to: {artifact_path}")

    selection_metric = str(cfg["train"].get("selection_metric", "val_loss")).lower()
    if selection_metric not in {"val_loss", "val_mse", "val_mae", "train_loss", "train_mse", "train_mae"}:
        raise ValueError(
            f"Unsupported train.selection_metric='{selection_metric}'. "
            "Expected one of: val_loss, val_mse, val_mae, train_loss, train_mse, train_mae."
        )
    loss_normalization_cfg = cfg["train"].get("loss_normalization", {}) or {}
    if isinstance(loss_normalization_cfg, bool):
        loss_normalization_cfg = {"enable": bool(loss_normalization_cfg)}
    penalty_warmup_epochs = int(cfg["train"].get("penalty_warmup_epochs", 0))
    penalty_scale_floor = float(cfg["train"].get("penalty_scale_floor", 1.0e-3))
    penalty_scale_source = str(
        cfg["train"].get("penalty_scale_source", "persistence")
    ).lower()
    if penalty_scale_source not in {"persistence", "frozen_backbone_patch"}:
        raise ValueError(
            "train.penalty_scale_source must be persistence or frozen_backbone_patch."
        )

    def compute_penalty_scale(loader: DataLoader, pred_len: int) -> torch.Tensor:
        if len(loader) == 0:
            return torch.full((P,), penalty_scale_floor, device=device)
        sum_all = torch.zeros(P, device=device)
        sum_pos = torch.zeros(P, device=device)
        cnt_all = 0
        cnt_pos = torch.zeros(P, device=device)
        for x, y, _ in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            last = x[..., -1:]
            yhat = last.expand(-1, -1, pred_len)
            pen_bcp = []
            for name in penalty_names:
                pen_bc = penalty_fns[name](yhat, y)
                pen_bcp.append(pen_bc)
            pen_bcp = torch.stack(pen_bcp, dim=-1)
            pen_flat = pen_bcp.reshape(-1, P)
            sum_all += pen_flat.sum(dim=0)
            cnt_all += int(pen_flat.shape[0])
            pos = pen_flat > 0
            sum_pos += (pen_flat * pos).sum(dim=0)
            cnt_pos += pos.sum(dim=0)
        if cnt_all == 0:
            return torch.full((P,), penalty_scale_floor, device=device)
        mean_all = sum_all / float(cnt_all)
        mean_pos = sum_pos / cnt_pos.clamp_min(1.0)
        scale = torch.where(cnt_pos > 0, mean_pos, mean_all)
        return scale.clamp_min(penalty_scale_floor)

    penalty_scale = (
        compute_penalty_scale(dl_tr_source, H)
        if penalty_scale_source == "persistence"
        else torch.full((P,), penalty_scale_floor, device=device)
    )
    penalty_need_threshold: Optional[torch.Tensor] = None
    penalty_need_statistics_summary: Dict[str, object] = {
        "source": str(penalty_scale_source),
        "patch_len": 0,
        "quantile": 0.75,
        "scale": None,
        "threshold": None,
        "coverage": None,
        "source_windows": 0,
        "source_units": 0,
    }

    _validate_strict_history_anchor_scope(
        cfg.get("moe", {}).get("history_anchor_expert", {}) or {},
        source="moe.history_anchor_expert",
    )
    model_train_stat_adapter_cfg = cfg.get("model", {}).get("train_stat_adapter", {}) or {}
    (
        model_train_stat_adapter_pc,
        model_train_stat_adapter_counts,
        model_train_stat_adapter_summary,
    ) = build_train_stat_anchor_from_config(
        data_window_tc,
        train_end=t_train,
        input_len=L,
        pred_len=H,
        cfg=model_train_stat_adapter_cfg,
        prefix="model.train_stat_adapter",
    )
    if bool(model_train_stat_adapter_cfg.get("enable", False)):
        print(
            "Model train-stat adapter enabled: "
            f"mode={model_train_stat_adapter_summary.get('mode')}, "
            f"period={model_train_stat_adapter_summary.get('period')}, "
            f"alpha={float(model_train_stat_adapter_cfg.get('alpha', 0.0) or 0.0):.3f}, "
            f"source=train[0:{t_train}]"
        )

    train_stat_anchor_cfg = cfg.get("moe", {}).get("train_stat_anchor_expert", {}) or {}
    train_stat_anchor_pc, train_stat_anchor_counts, train_stat_anchor_summary = build_train_stat_anchor_from_config(
        data_window_tc,
        train_end=t_train,
        input_len=L,
        pred_len=H,
        cfg=train_stat_anchor_cfg,
        prefix="moe.train_stat_anchor_expert",
    )
    train_residual_anchor_cfg = cfg.get("moe", {}).get("train_residual_anchor_expert", {}) or {}
    train_residual_anchor_phc = None
    train_residual_anchor_summary: Dict[str, object] = {
        "enable": bool(train_residual_anchor_cfg.get("enable", False)),
    }
    if bool(train_stat_anchor_cfg.get("enable", False)):
        print(
            "Train-stat anchor expert enabled: "
            f"mode={train_stat_anchor_summary.get('mode')}, period={train_stat_anchor_summary.get('period')}, "
            f"alpha={float(train_stat_anchor_cfg.get('alpha', 0.0) or 0.0):.3f}, "
            f"source=train[0:{t_train}]"
        )

    def eval_loop_with_history(*args, **kwargs):
        kwargs.setdefault("history_anchor_cfg", history_anchor_cfg)
        kwargs.setdefault("observed_history_tc", data_window_tc)
        kwargs.setdefault("input_len", L)
        kwargs.setdefault("model_train_stat_adapter_pc", model_train_stat_adapter_pc)
        kwargs.setdefault("model_train_stat_adapter_cfg", model_train_stat_adapter_cfg)
        kwargs.setdefault("train_stat_anchor_pc", train_stat_anchor_pc)
        kwargs.setdefault("train_residual_anchor_phc", train_residual_anchor_phc)
        kwargs.setdefault("learnable_output_anchor", learnable_output_anchor)
        kwargs.setdefault("gate_feature_mode", gate_feature_mode)
        kwargs.setdefault("calendar_feature_tf", calendar_feature_tf)
        kwargs.setdefault("calendar_residual_coef_cf", calendar_residual_coef_cf)
        kwargs.setdefault(
            "position_daily_residual_coef_cfh",
            position_daily_residual_coef_cfh,
        )
        kwargs.setdefault(
            "position_daily_residual_cfg",
            position_daily_residual_cfg,
        )
        return eval_loop(*args, **kwargs)

    # cluster portraits (prototype + penalty metrics)
    portrait_cfg = cfg.get("portrait", {})
    gate_prior_cfg = cfg.get("moe", {}).get("gate_prior", {})
    cluster_penalty_prior_cfg = cfg.get("moe", {}).get("cluster_penalty_prior", {}) or {}
    channel_penalty_prior_cfg = cfg.get("moe", {}).get("channel_penalty_prior", {}) or {}
    need_penalty_portrait = (
        bool(portrait_cfg.get("enable", False))
        or bool(gate_prior_cfg.get("enable", False))
        or bool(cluster_penalty_prior_cfg.get("enable", False))
        or bool(channel_penalty_prior_cfg.get("enable", False))
    )
    penalty_portrait_kp = None
    channel_penalty_portrait_cp = None
    if need_penalty_portrait and len(penalty_names) > 0:
        # Portrait generation is diagnostic; keep it from advancing the shuffled
        # training loader RNG before model/gate initialization.
        portrait_generator = torch.Generator()
        portrait_generator.manual_seed(int(cfg["exp"]["seed"]))
        if len(dtr) > 0:
            portrait_loader = DataLoader(
                dtr, batch_size=bs, shuffle=False, num_workers=0,
                pin_memory=pin_mem, generator=portrait_generator
            )
        elif len(dva) > 0:
            portrait_loader = DataLoader(
                dva, batch_size=bs, shuffle=False, num_workers=0,
                pin_memory=pin_mem, generator=portrait_generator
            )
        else:
            portrait_loader = DataLoader(
                dte, batch_size=bs, shuffle=False, num_workers=0,
                pin_memory=pin_mem, generator=portrait_generator
            )
        penalty_portrait_kp = compute_cluster_penalty_portrait(
            portrait_loader, penalty_names, penalty_fns, cluster_id_c, K, H, device
        )
        if bool(channel_penalty_prior_cfg.get("enable", False)):
            channel_penalty_portrait_cp = compute_channel_penalty_portrait(
                portrait_loader, penalty_names, penalty_fns, C, H, device
            )
    if bool(portrait_cfg.get("enable", False)):
        portrait_dir = portrait_cfg.get("out_dir", os.path.join(out_dir, "cluster_portraits"))
        portrait_dpi = int(portrait_cfg.get("dpi", 140))
        max_points = int(portrait_cfg.get("max_points", 2000))
        jump_thr = float(portrait_cfg.get("jump_threshold", cfg.get("penalties", {}).get("jump_threshold", 2.0)))
        if penalty_portrait_kp is not None:
            metric_names = penalty_names
            metric_values = penalty_portrait_kp
        else:
            metric_names = None
            metric_values = None
        paths = save_cluster_portraits(
            out_dir=portrait_dir,
            data_tc=data_tc,
            cluster_id_c=cluster_id_c,
            jump_thr=jump_thr,
            dpi=portrait_dpi,
            max_points=max_points,
            metric_names=metric_names,
            metric_values_km=metric_values,
        )
        print(f"Saved cluster portraits to: {paths['dir']}")
        print(f"Portrait metrics: {paths['metrics_csv']}")

    # 6) Build the clusterwise predictor.
    model_cfg = cfg["model"]
    model = build_cluster_predictor(
        num_clusters=K,
        input_len=L,
        pred_len=H,
        model_cfg=model_cfg,
        num_channels=C,
        cluster_id_c=cluster_id_c,
    ).to(device)

    # 7) Configure MoE routing and lambda weighting.
    moe_cfg = cfg["moe"]
    moe_enable = bool(moe_cfg.get("enable", True))
    shared_moe_across_clusters = bool(
        moe_cfg.get("shared_across_clusters", moe_cfg.get("share_across_clusters", False))
    ) and moe_enable and P > 0
    gate_entropy_weight = float(moe_cfg.get("gate_entropy_weight", 0.0))
    gate_balance_weight = float(moe_cfg.get("gate_balance_weight", 0.0))
    gate_soft_weight = float(moe_cfg.get("gate_soft_weight", 0.0))
    gate_entropy_target_frac = float(moe_cfg.get("gate_entropy_target_frac", 0.0))
    gate_route_on_penalty_only = bool(moe_cfg.get("gate_route_on_penalty_only", False))
    gate_feature_mode = _normalize_gate_feature_mode(moe_cfg.get("gate_feature_mode", "history"))
    router_mode = str(moe_cfg.get("router_mode", "learned")).lower()
    router_penalty_context_weight = float(moe_cfg.get("router_penalty_context_weight", 0.0))
    router_detach_penalty_context = bool(moe_cfg.get("router_detach_penalty_context", True))
    router_penalty_context_score = str(moe_cfg.get("router_penalty_context_score", "high_violation")).lower()
    pred_residual_cfg = moe_cfg.get("pred_side_residual", {}) or {}
    pred_residual_enable = bool(pred_residual_cfg.get("enable", False)) and moe_enable and P > 0
    pred_residual_require_pure_backbone_base = bool(
        pred_residual_cfg.get("require_pure_backbone_base", False)
    ) and pred_residual_enable
    pred_residual_semantic_bank_stage1 = bool(
        pred_residual_cfg.get("semantic_bank_stage1", False)
    ) and pred_residual_enable
    pred_residual_semantic_output_head_scope = str(
        pred_residual_cfg.get("semantic_output_head_scope", "shared")
    ).strip().lower()
    if pred_residual_semantic_output_head_scope not in {"shared", "per_cluster"}:
        raise ValueError(
            "moe.pred_side_residual.semantic_output_head_scope must be shared or per_cluster."
        )
    position_daily_residual_expert_cfg = pred_residual_cfg.get(
        "position_daily_residual_expert", {}
    ) or {}
    if not isinstance(position_daily_residual_expert_cfg, dict):
        position_daily_residual_expert_cfg = {
            "enable": bool(position_daily_residual_expert_cfg)
        }
    position_daily_residual_expert_enable = bool(
        position_daily_residual_expert_cfg.get("enable", False)
    ) and pred_residual_enable
    anchor_ridge_gate_cfg = pred_residual_cfg.get("anchor_ridge_gate", {}) or {}
    if not isinstance(anchor_ridge_gate_cfg, dict):
        anchor_ridge_gate_cfg = {"enable": bool(anchor_ridge_gate_cfg)}
    anchor_ridge_gate_enable = bool(
        anchor_ridge_gate_cfg.get("enable", False)
    ) and pred_residual_enable
    if anchor_ridge_gate_enable and not position_daily_residual_expert_enable:
        raise ValueError(
            "pred_side_residual.anchor_ridge_gate requires "
            "position_daily_residual_expert.enable=true."
        )
    anchor_ridge_gate_summary: Dict[str, object] = {
        "enable": bool(anchor_ridge_gate_enable),
        "inside_pred_residual_forward": bool(anchor_ridge_gate_enable),
    }
    if position_daily_residual_expert_enable and bool(
        position_daily_residual_cfg.get("enable", False)
    ):
        raise ValueError(
            "Enable either pred_side_residual.position_daily_residual_expert "
            "or top-level position_daily_residual_ridge, not both."
        )
    position_daily_residual_expert_summary: Dict[str, object] = {
        "enable": bool(position_daily_residual_expert_enable),
        "inside_pred_residual_forward": bool(position_daily_residual_expert_enable),
        "top_level_postprocess": False,
    }
    named_output_projection_cfg = pred_residual_cfg.get("named_output_projection", {}) or {}
    if not isinstance(named_output_projection_cfg, dict):
        named_output_projection_cfg = {"enable": bool(named_output_projection_cfg)}
    named_output_projection_enable = bool(named_output_projection_cfg.get("enable", False))
    named_output_projection_fixed_alpha = bool(named_output_projection_cfg.get("fixed_alpha", False))
    named_output_projection_scale_by_name = {
        str(name): float(value)
        for name, value in (named_output_projection_cfg.get("scale_by_name", {}) or {}).items()
    }
    named_output_projection_carrier_names = [
        str(name)
        for name in (named_output_projection_cfg.get("carrier_names", []) or [])
    ]
    named_output_projection_patch_len = int(
        named_output_projection_cfg.get("patch_len", 0) or 0
    )
    named_output_projection_mode = str(
        named_output_projection_cfg.get("mode", "legacy_carrier")
    )
    named_output_projection_diff_amp_max = float(
        named_output_projection_cfg.get("diff_amp_max", 1.0)
    )
    periodic_anchor_expert_cfg = pred_residual_cfg.get("periodic_anchor_expert", {}) or {}
    if not isinstance(periodic_anchor_expert_cfg, dict):
        periodic_anchor_expert_cfg = {"enable": bool(periodic_anchor_expert_cfg)}
    periodic_anchor_expert_enable = bool(periodic_anchor_expert_cfg.get("enable", False)) and pred_residual_enable
    periodic_anchor_expert_scale = float(periodic_anchor_expert_cfg.get("scale", 1.0))
    periodic_anchor_expert_freeze_source = bool(periodic_anchor_expert_cfg.get("freeze_source", True))
    periodic_anchor_expert_preserve_loaded_source_mask = bool(
        periodic_anchor_expert_cfg.get("preserve_loaded_source_mask", False)
    )
    if periodic_anchor_expert_preserve_loaded_source_mask and not (
        periodic_anchor_expert_enable and periodic_anchor_expert_freeze_source
    ):
        raise ValueError(
            "periodic_anchor_expert.preserve_loaded_source_mask=true requires "
            "periodic_anchor_expert.enable=true and freeze_source=true."
        )
    pred_residual_freeze_adapter_bank = bool(pred_residual_cfg.get("freeze_adapter_bank", False))
    patch_router_cfg = pred_residual_cfg.get("patch_router", {}) or {}
    if not isinstance(patch_router_cfg, dict):
        patch_router_cfg = {"enable": bool(patch_router_cfg)}
    patch_router_epoch0_noop_cfg = patch_router_cfg.get(
        "epoch0_noop_selection",
        {},
    ) or {}
    if not isinstance(patch_router_epoch0_noop_cfg, dict):
        patch_router_epoch0_noop_cfg = {
            "enable": bool(patch_router_epoch0_noop_cfg)
        }
    patch_router_epoch0_noop_enable = bool(
        patch_router_cfg.get("enable", False)
        and patch_router_epoch0_noop_cfg.get("enable", False)
    )
    patch_router_epoch0_require_dual = bool(
        patch_router_epoch0_noop_cfg.get("require_dual_improvement", True)
    )
    patch_router_expected_mse_weight = (
        max(0.0, float(patch_router_cfg.get("expected_mse_weight", 0.0)))
        if bool(patch_router_cfg.get("enable", False))
        else 0.0
    )
    patch_router_expected_mae_weight = (
        max(0.0, float(patch_router_cfg.get("expected_mae_weight", 0.0)))
        if bool(patch_router_cfg.get("enable", False))
        else 0.0
    )
    patch_router_mixture_mse_weight = (
        max(0.0, float(patch_router_cfg.get("mixture_mse_weight", 0.0)))
        if bool(patch_router_cfg.get("enable", False))
        else 0.0
    )
    patch_router_mixture_mae_weight = (
        max(0.0, float(patch_router_cfg.get("mixture_mae_weight", 0.0)))
        if bool(patch_router_cfg.get("enable", False))
        else 0.0
    )
    patch_router_temporal_group_dro_cfg = patch_router_cfg.get(
        "temporal_group_dro",
        {},
    ) or {}
    if not isinstance(patch_router_temporal_group_dro_cfg, dict):
        patch_router_temporal_group_dro_cfg = {
            "enable": bool(patch_router_temporal_group_dro_cfg)
        }
    patch_router_temporal_group_dro_enable = bool(
        patch_router_cfg.get("enable", False)
        and patch_router_temporal_group_dro_cfg.get("enable", False)
    )
    patch_router_temporal_group_dro_weight = max(
        0.0,
        float(patch_router_temporal_group_dro_cfg.get("weight", 1.0)),
    )
    patch_router_temporal_group_dro_domains = max(
        2,
        int(patch_router_temporal_group_dro_cfg.get("num_domains", 6)),
    )
    patch_router_temporal_group_dro_temperature = max(
        0.0,
        float(patch_router_temporal_group_dro_cfg.get("temperature", 0.01)),
    )
    if (
        patch_router_temporal_group_dro_enable
        and patch_router_expected_mse_weight <= 0.0
    ):
        raise ValueError(
            "patch_router.temporal_group_dro requires expected_mse_weight > 0."
        )
    patch_router_oracle_ce_weight = (
        max(0.0, float(patch_router_cfg.get("oracle_ce_weight", 0.0)))
        if bool(patch_router_cfg.get("enable", False))
        else 0.0
    )
    patch_router_oracle_ce_warmup_epochs = max(
        0,
        int(patch_router_cfg.get("oracle_ce_warmup_epochs", 0)),
    )
    patch_router_freeze_experts_after_warmup = bool(
        patch_router_cfg.get("freeze_experts_after_warmup", False)
    )
    patch_router_supervision_only = bool(
        patch_router_cfg.get("supervision_only", False)
    )
    patch_router_diagnostics_cfg = patch_router_cfg.get(
        "diagnostics",
        pred_residual_cfg.get("diagnostics", {}),
    ) or {}
    if not isinstance(patch_router_diagnostics_cfg, dict):
        patch_router_diagnostics_cfg = {"enable": bool(patch_router_diagnostics_cfg)}
    patch_router_train_oracle_diagnostic = bool(
        patch_router_diagnostics_cfg.get("train_oracle", False)
    )
    patch_router_score_threshold_curve = bool(
        patch_router_diagnostics_cfg.get("score_threshold_curve", False)
    )
    patch_router_score_threshold_max_windows = max(
        0,
        int(patch_router_diagnostics_cfg.get("score_threshold_curve_max_windows", 0)),
    )
    raw_score_threshold_heads = patch_router_diagnostics_cfg.get(
        "score_threshold_curve_heads",
        None,
    )
    if isinstance(raw_score_threshold_heads, str):
        patch_router_score_threshold_heads = {raw_score_threshold_heads}
    elif raw_score_threshold_heads is None:
        patch_router_score_threshold_heads = None
    else:
        patch_router_score_threshold_heads = {
            str(value) for value in raw_score_threshold_heads
        }
    patch_router_train_temporal_blocks = max(
        0,
        int(patch_router_diagnostics_cfg.get("train_temporal_blocks", 0)),
    )
    patch_router_walk_forward_cfg = patch_router_diagnostics_cfg.get(
        "walk_forward_reliability",
        {},
    ) or {}
    if not isinstance(patch_router_walk_forward_cfg, dict):
        patch_router_walk_forward_cfg = {
            "enable": bool(patch_router_walk_forward_cfg)
        }
    patch_router_walk_forward_enable = bool(
        patch_router_walk_forward_cfg.get("enable", False)
    )
    patch_router_walk_forward_test_enable = bool(
        patch_router_walk_forward_cfg.get("test_enable", False)
    )
    patch_router_walk_forward_condition_on_selected_penalty = bool(
        patch_router_walk_forward_cfg.get(
            "condition_on_selected_penalty",
            False,
        )
    )
    patch_router_walk_forward_rerank_all_candidates = bool(
        patch_router_walk_forward_cfg.get(
            "rerank_all_candidates",
            False,
        )
    )
    patch_router_walk_forward_feedback_ridge = bool(
        patch_router_walk_forward_cfg.get("feedback_ridge", False)
    )
    patch_router_walk_forward_feedback_ridge_strength = float(
        patch_router_walk_forward_cfg.get("feedback_ridge_strength", 0.1)
    )
    patch_router_walk_forward_feedback_target_clip = float(
        patch_router_walk_forward_cfg.get("feedback_target_clip", 2.0)
    )
    patch_router_walk_forward_label_delay = int(
        patch_router_walk_forward_cfg.get("label_delay", H)
    )
    patch_router_walk_forward_label_delay_mode = str(
        patch_router_walk_forward_cfg.get("label_delay_mode", "full_horizon")
    ).strip().lower()
    if patch_router_walk_forward_label_delay_mode not in {
        "full_horizon",
        "patch_end",
    }:
        raise ValueError(
            "walk_forward_reliability.label_delay_mode must be full_horizon or patch_end."
        )
    patch_router_walk_forward_lookback = int(
        patch_router_walk_forward_cfg.get("lookback_windows", 2 * H)
    )
    patch_router_walk_forward_min_history = int(
        patch_router_walk_forward_cfg.get("min_history_windows", H)
    )
    patch_router_walk_forward_history_stride = int(
        patch_router_walk_forward_cfg.get("history_stride", 1)
    )
    patch_router_walk_forward_min_mean_gain = float(
        patch_router_walk_forward_cfg.get("min_mean_gain", 0.0)
    )
    patch_router_walk_forward_max_abs_regime_z_raw = (
        patch_router_walk_forward_cfg.get("max_abs_regime_z", None)
    )
    patch_router_walk_forward_max_abs_regime_z = (
        None
        if patch_router_walk_forward_max_abs_regime_z_raw is None
        else float(patch_router_walk_forward_max_abs_regime_z_raw)
    )
    patch_router_walk_forward_scale_mode = str(
        patch_router_walk_forward_cfg.get("scale_mode", "binary")
    ).strip().lower()
    patch_router_walk_forward_max_scale = float(
        patch_router_walk_forward_cfg.get("max_scale", 1.0)
    )
    patch_router_walk_forward_scale_consensus_blocks = int(
        patch_router_walk_forward_cfg.get("scale_consensus_blocks", 1)
    )
    patch_router_walk_forward_feature_ridge = float(
        patch_router_walk_forward_cfg.get("feature_ridge", 0.1)
    )
    patch_router_walk_forward_feature_update_blocks = int(
        patch_router_walk_forward_cfg.get("feature_update_blocks", 6)
    )
    patch_router_walk_forward_temporal_blocks = max(
        0,
        int(patch_router_walk_forward_cfg.get("temporal_blocks", 6)),
    )
    patch_router_walk_forward_train_audit_fraction = float(
        patch_router_walk_forward_cfg.get("train_audit_fraction", 0.4)
    )
    if patch_router_walk_forward_enable and not (
        0.0 < patch_router_walk_forward_train_audit_fraction < 1.0
    ):
        raise ValueError(
            "walk_forward_reliability.train_audit_fraction must be in (0,1)."
        )
    patch_router_validation_temporal_blocks = max(
        0,
        int(patch_router_diagnostics_cfg.get("validation_temporal_blocks", 0)),
    )
    patch_router_frozen_expert_params = 0
    patch_router_expert_freeze_applied = False
    patch_router_oracle_min_abs_improvement = float(
        patch_router_cfg.get("oracle_min_abs_improvement", 0.0)
    )
    patch_router_hierarchical_cfg = patch_router_cfg.get("hierarchical_recall", {}) or {}
    if not isinstance(patch_router_hierarchical_cfg, dict):
        patch_router_hierarchical_cfg = {"enable": bool(patch_router_hierarchical_cfg)}
    patch_router_mask_inactive_fixed_channels = bool(
        patch_router_hierarchical_cfg.get(
            "mask_inactive_fixed_channels",
            False,
        )
    )
    patch_router_hierarchical_enable = bool(
        patch_router_cfg.get("enable", False)
        and patch_router_hierarchical_cfg.get("enable", False)
    )
    patch_router_hierarchical_weight = (
        max(0.0, float(patch_router_hierarchical_cfg.get("supervision_weight", 0.0)))
        if patch_router_hierarchical_enable
        else 0.0
    )
    patch_router_hierarchical_warmup_epochs = max(
        0,
        int(patch_router_hierarchical_cfg.get("warmup_epochs", 0)),
    )
    patch_router_hierarchical_min_abs_improvement = float(
        patch_router_hierarchical_cfg.get(
            "min_abs_improvement",
            patch_router_oracle_min_abs_improvement,
        )
    )
    patch_router_expert_risk_cfg = patch_router_hierarchical_cfg.get(
        "expert_conditional_risk",
        {},
    ) or {}
    if not isinstance(patch_router_expert_risk_cfg, dict):
        patch_router_expert_risk_cfg = {"enable": bool(patch_router_expert_risk_cfg)}
    patch_router_lower_quantile_cfg = patch_router_expert_risk_cfg.get(
        "lower_quantile",
        {},
    ) or {}
    if not isinstance(patch_router_lower_quantile_cfg, dict):
        patch_router_lower_quantile_cfg = {
            "enable": bool(patch_router_lower_quantile_cfg)
        }
    patch_router_pairwise_rank_cfg = patch_router_expert_risk_cfg.get(
        "pairwise_rank",
        {},
    ) or {}
    if not isinstance(patch_router_pairwise_rank_cfg, dict):
        patch_router_pairwise_rank_cfg = {
            "enable": bool(patch_router_pairwise_rank_cfg)
        }
    patch_router_pairwise_freeze_other_parameters = bool(
        patch_router_pairwise_rank_cfg.get("freeze_other_parameters", False)
    )
    patch_router_temporal_calibration_cfg = patch_router_expert_risk_cfg.get(
        "temporal_calibration",
        {},
    ) or {}
    if not isinstance(patch_router_temporal_calibration_cfg, dict):
        patch_router_temporal_calibration_cfg = {
            "enable": bool(patch_router_temporal_calibration_cfg)
        }
    patch_router_temporal_calibration_enable = bool(
        patch_router_hierarchical_enable
        and patch_router_expert_risk_cfg.get("enable", False)
        and patch_router_temporal_calibration_cfg.get("enable", False)
    )
    patch_router_calibration_tail_fraction = float(
        patch_router_temporal_calibration_cfg.get("tail_fraction", 0.2)
    )
    patch_router_calibration_blocks = max(
        1,
        int(patch_router_temporal_calibration_cfg.get("temporal_blocks", 4)),
    )
    patch_router_calibration_purge_windows = max(
        0,
        int(patch_router_temporal_calibration_cfg.get("purge_windows", L + H - 1)),
    )
    patch_router_calibration_min_gain_cost_ratio = max(
        0.0,
        float(patch_router_temporal_calibration_cfg.get("min_gain_cost_ratio", 1.0)),
    )
    patch_router_calibration_min_block_net_gain = float(
        patch_router_temporal_calibration_cfg.get("min_block_net_gain", 0.0)
    )
    patch_router_calibration_per_penalty = bool(
        patch_router_temporal_calibration_cfg.get("per_penalty", False)
    )
    if patch_router_temporal_calibration_enable and not (
        0.0 < patch_router_calibration_tail_fraction < 0.5
    ):
        raise ValueError("patch router calibration tail_fraction must be in (0,0.5).")
    patch_router_calibration_start_idx = int(
        len(dtr) * (1.0 - patch_router_calibration_tail_fraction)
    )
    patch_router_supervision_end_idx = max(
        0,
        patch_router_calibration_start_idx - patch_router_calibration_purge_windows,
    )
    if patch_router_temporal_calibration_enable and not patch_router_supervision_only:
        raise ValueError(
            "patch router temporal calibration requires patch_router.supervision_only=true."
        )
    if (
        patch_router_temporal_calibration_enable
        and patch_router_supervision_end_idx <= 0
    ):
        raise ValueError("patch router temporal calibration leaves no supervision windows.")
    patch_router_temporal_calibration_summary = None
    patch_router_hierarchical_loss_cfg = {
        "adoption_bce_weight": float(patch_router_hierarchical_cfg.get("adoption_bce_weight", 1.0)),
        "proposal_bce_weight": float(patch_router_hierarchical_cfg.get("proposal_bce_weight", 1.0)),
        "proposal_gain_listwise_weight": float(
            patch_router_hierarchical_cfg.get("proposal_gain_listwise_weight", 0.0)
        ),
        "proposal_rescue_ce_weight": float(
            patch_router_hierarchical_cfg.get("proposal_rescue_ce_weight", 0.0)
        ),
        "ranking_ce_weight": float(patch_router_hierarchical_cfg.get("ranking_ce_weight", 1.0)),
        "utility_regression_weight": float(
            patch_router_hierarchical_cfg.get("utility_regression_weight", 0.0)
        ),
        "risk_calibration_weight": float(
            patch_router_hierarchical_cfg.get("risk_calibration_weight", 0.0)
        ),
        "risk_sign_bce_weight": float(
            patch_router_hierarchical_cfg.get("risk_sign_bce_weight", 0.0)
        ),
        "risk_magnitude_weight": float(
            patch_router_hierarchical_cfg.get("risk_magnitude_weight", 0.0)
        ),
        "risk_lower_quantile_weight": float(
            patch_router_lower_quantile_cfg.get("loss_weight", 0.0)
        ),
        "risk_lower_quantile": float(
            patch_router_lower_quantile_cfg.get("quantile", 0.2)
        ),
        "selected_utility_policy_weight": float(
            patch_router_hierarchical_cfg.get("selected_utility_policy_weight", 0.0)
        ),
        "selected_adoption_bce_weight": float(
            patch_router_hierarchical_cfg.get("selected_adoption_bce_weight", 0.0)
        ),
        "selected_adoption_recall_weight": float(
            patch_router_hierarchical_cfg.get(
                "selected_adoption_recall_weight",
                0.0,
            )
        ),
        "selected_false_adopt_weight": float(
            patch_router_hierarchical_cfg.get("selected_false_adopt_weight", 0.0)
        ),
        "pairwise_rank_weight": float(
            patch_router_pairwise_rank_cfg.get("loss_weight", 0.0)
        ),
        "adoption_recall_weight": float(patch_router_hierarchical_cfg.get("adoption_recall_weight", 1.0)),
        "false_adopt_weight": float(patch_router_hierarchical_cfg.get("false_adopt_weight", 1.0)),
        "penalty_recall_weight": float(patch_router_hierarchical_cfg.get("penalty_recall_weight", 1.0)),
        "false_penalty_weight": float(patch_router_hierarchical_cfg.get("false_penalty_weight", 1.0)),
        "target_adopt_probability": float(
            patch_router_hierarchical_cfg.get("target_adopt_probability", 0.8)
        ),
        "false_adopt_max_probability": float(
            patch_router_hierarchical_cfg.get("false_adopt_max_probability", 0.2)
        ),
        "target_penalty_probability": float(
            patch_router_hierarchical_cfg.get("target_penalty_probability", 0.7)
        ),
        "false_penalty_max_probability": float(
            patch_router_hierarchical_cfg.get("false_penalty_max_probability", 0.3)
        ),
    }
    patch_router_expert_warmup_epochs = max(
        int(patch_router_oracle_ce_warmup_epochs),
        int(patch_router_hierarchical_warmup_epochs),
    )
    if shared_moe_across_clusters and bool((pred_residual_cfg.get("channel_expert_adapters", {}) or {}).get("enable", False)):
        raise ValueError("moe.shared_across_clusters does not support pred_side_residual.channel_expert_adapters.")
    phase_residual_candidate_cfg = pred_residual_cfg.get("phase_residual_candidate", {}) or {}
    if not isinstance(phase_residual_candidate_cfg, dict):
        phase_residual_candidate_cfg = {"enable": bool(phase_residual_candidate_cfg)}
    phase_residual_candidate_enable = bool(phase_residual_candidate_cfg.get("enable", False)) and pred_residual_enable
    raw_phase_residual_candidate_names = phase_residual_candidate_cfg.get(
        "names",
        phase_residual_candidate_cfg.get("penalty_names", []),
    )
    if isinstance(raw_phase_residual_candidate_names, str):
        phase_residual_candidate_names = [raw_phase_residual_candidate_names]
    else:
        phase_residual_candidate_names = [str(v) for v in (raw_phase_residual_candidate_names or [])]
    if phase_residual_candidate_enable and len(phase_residual_candidate_names) == 0:
        raise ValueError("moe.pred_side_residual.phase_residual_candidate.names must be non-empty when enabled.")
    phase_residual_candidate_period = int(phase_residual_candidate_cfg.get("period", 96))
    if phase_residual_candidate_enable and phase_residual_candidate_period <= 0:
        raise ValueError("moe.pred_side_residual.phase_residual_candidate.period must be positive.")
    phase_residual_candidate_scale = float(phase_residual_candidate_cfg.get("scale", 1.0))
    phase_residual_candidate_summary: Dict[str, object] = {
        "enable": bool(phase_residual_candidate_enable),
        "names": list(phase_residual_candidate_names),
        "period": int(phase_residual_candidate_period),
        "scale": float(phase_residual_candidate_scale),
        "source_split": "train" if phase_residual_candidate_enable else None,
    }
    pred_residual_ignore_skip_during_training = bool(
        pred_residual_cfg.get(
            "ignore_skip_during_training",
            pred_residual_cfg.get("train_ignore_skip", False),
        )
    ) and pred_residual_enable
    pred_residual_specialization_weight = (
        float(pred_residual_cfg.get("specialization_weight", 0.1)) if pred_residual_enable else 0.0
    )
    pred_residual_norm_weight = float(pred_residual_cfg.get("norm_weight", 1.0e-4)) if pred_residual_enable else 0.0
    pred_residual_intervention_weight = (
        float(pred_residual_cfg.get("intervention_weight", 1.0e-3)) if pred_residual_enable else 0.0
    )
    pred_residual_candidate_supervision_cfg = (
        pred_residual_cfg.get(
            "adapter_attribute_supervision",
            pred_residual_cfg.get("candidate_supervision", {}),
        )
        or {}
    )
    if not isinstance(pred_residual_candidate_supervision_cfg, dict):
        pred_residual_candidate_supervision_cfg = {"weight": float(pred_residual_candidate_supervision_cfg)}
    pred_residual_candidate_supervision_weight = (
        float(
            pred_residual_candidate_supervision_cfg.get(
                "weight",
                pred_residual_cfg.get("candidate_supervision_weight", 0.0),
            )
        )
        if pred_residual_enable
        else 0.0
    )
    pred_residual_candidate_supervision_loss = str(
        pred_residual_candidate_supervision_cfg.get("loss", "mse")
    ).lower()
    pred_residual_semantic_level_controller_cfg = (
        pred_residual_cfg.get("semantic_level_controller", {}) or {}
    )
    if not isinstance(pred_residual_semantic_level_controller_cfg, dict):
        raise ValueError("moe.pred_side_residual.semantic_level_controller must be a mapping.")
    pred_residual_semantic_level_amplitude_optimizer_cfg = (
        pred_residual_semantic_level_controller_cfg.get("amplitude_optimizer", {})
        or {}
    )
    if not isinstance(pred_residual_semantic_level_amplitude_optimizer_cfg, dict):
        raise ValueError(
            "moe.pred_side_residual.semantic_level_controller.amplitude_optimizer "
            "must be a mapping."
        )
    pred_residual_semantic_level_amplitude_optimizer_name = str(
        pred_residual_semantic_level_amplitude_optimizer_cfg.get("name", "sgd")
    ).strip().lower()
    pred_residual_semantic_level_amplitude_lr_raw = (
        pred_residual_semantic_level_amplitude_optimizer_cfg.get("lr", None)
    )
    pred_residual_semantic_level_amplitude_weight_decay_raw = (
        pred_residual_semantic_level_amplitude_optimizer_cfg.get(
            "weight_decay", None
        )
    )
    pred_residual_semantic_level_separate_need_gate = bool(
        pred_residual_semantic_level_controller_cfg.get("separate_need_gate", False)
    )
    pred_residual_semantic_level_need_positive_weight = float(
        pred_residual_candidate_supervision_cfg.get("level_need_positive_weight", 1.0)
    )
    pred_residual_semantic_level_acceptance_candidate_source = str(
        pred_residual_candidate_supervision_cfg.get(
            "level_acceptance_candidate",
            "executed",
        )
    ).strip().lower()
    if pred_residual_semantic_level_acceptance_candidate_source not in {
        "executed",
        "raw_amplitude",
    }:
        raise ValueError(
            "adapter_attribute_supervision.level_acceptance_candidate must be "
            "executed or raw_amplitude."
        )
    pred_residual_candidate_supervision_forecast_mse_weight = float(
        pred_residual_candidate_supervision_cfg.get("forecast_mse_weight", 1.0)
    )
    pred_residual_candidate_supervision_need_patch_len = int(
        pred_residual_candidate_supervision_cfg.get(
            "need_patch_len",
            named_output_projection_patch_len,
        )
        or 0
    )
    pred_residual_candidate_supervision_need_quantile = float(
        pred_residual_candidate_supervision_cfg.get("need_quantile", 0.75)
    )
    pred_residual_candidate_supervision_noop_weight = float(
        pred_residual_candidate_supervision_cfg.get("noop_weight", 1.0)
    )
    pred_residual_candidate_supervision_independent_optimization = bool(
        pred_residual_candidate_supervision_cfg.get(
            "independent_optimization",
            False,
        )
    )
    pred_residual_candidate_supervision_independent_optimizer = str(
        pred_residual_candidate_supervision_cfg.get(
            "independent_optimizer",
            "sgd",
        )
    ).lower()
    pred_residual_semantic_active_penalty = str(
        pred_residual_candidate_supervision_cfg.get("active_penalty", "")
    ).strip()
    pred_residual_semantic_raw_gradient_accumulation_cfg = (
        pred_residual_candidate_supervision_cfg.get(
            "raw_gradient_accumulation", {}
        )
        or {}
    )
    if not isinstance(
        pred_residual_semantic_raw_gradient_accumulation_cfg, dict
    ):
        raise ValueError(
            "adapter_attribute_supervision.raw_gradient_accumulation must be a mapping."
        )
    pred_residual_semantic_raw_gradient_accumulation = bool(
        pred_residual_semantic_raw_gradient_accumulation_cfg.get("enable", False)
    )
    pred_residual_semantic_raw_gradient_microbatches = int(
        pred_residual_semantic_raw_gradient_accumulation_cfg.get(
            "microbatches", 16
        )
    )
    pred_residual_semantic_final_train_audit = bool(
        pred_residual_candidate_supervision_cfg.get(
            "final_train_semantic_audit", False
        )
    )
    pred_residual_semantic_validation_blocks = int(
        pred_residual_candidate_supervision_cfg.get("validation_blocks", 3)
    )
    pred_residual_semantic_min_improved_fraction = float(
        pred_residual_candidate_supervision_cfg.get(
            "min_high_need_improved_fraction", 0.60
        )
    )
    pred_residual_semantic_min_gain_by_name = {
        str(name): float(value)
        for name, value in (
            pred_residual_candidate_supervision_cfg.get(
                "min_matching_gain_by_name",
                {
                    "level": 0.10,
                    "trend": 0.10,
                    "d2_match": 0.10,
                    "diff_amp": 0.70,
                },
            )
            or {}
        ).items()
    }
    pred_residual_candidate_supervision_high_mse_relative_tolerance = float(
        pred_residual_candidate_supervision_cfg.get(
            "high_mse_relative_tolerance",
            0.0,
        )
    )
    pred_residual_candidate_supervision_low_mse_relative_tolerance = float(
        pred_residual_candidate_supervision_cfg.get(
            "low_mse_relative_tolerance",
            1.0e-3,
        )
    )
    pred_residual_candidate_supervision_low_high_rms_ratio_max = float(
        pred_residual_candidate_supervision_cfg.get(
            "low_high_rms_ratio_max",
            0.25,
        )
    )
    pred_residual_candidate_supervision_constraint_weight = float(
        pred_residual_candidate_supervision_cfg.get(
            "constraint_weight",
            1.0,
        )
    )
    pred_residual_candidate_supervision_constraint_eps = float(
        pred_residual_candidate_supervision_cfg.get(
            "constraint_eps",
            1.0e-8,
        )
    )
    if not 0.0 <= pred_residual_candidate_supervision_need_quantile <= 1.0:
        raise ValueError("adapter_attribute_supervision.need_quantile must be in [0,1].")
    if pred_residual_candidate_supervision_noop_weight < 0.0:
        raise ValueError("adapter_attribute_supervision.noop_weight must be nonnegative.")
    if (
        pred_residual_candidate_supervision_high_mse_relative_tolerance < 0.0
        or pred_residual_candidate_supervision_low_mse_relative_tolerance < 0.0
        or pred_residual_candidate_supervision_low_high_rms_ratio_max < 0.0
        or pred_residual_candidate_supervision_constraint_weight <= 0.0
        or pred_residual_candidate_supervision_constraint_eps <= 0.0
    ):
        raise ValueError(
            "adapter_attribute_supervision acceptance constraints require "
            "nonnegative tolerances/ratio and positive weight/eps."
        )
    if pred_residual_semantic_validation_blocks < 3:
        raise ValueError("semantic Stage-1 requires at least three validation blocks.")
    if not 0.0 < pred_residual_semantic_min_improved_fraction <= 1.0:
        raise ValueError(
            "min_high_need_improved_fraction must be in (0,1]."
        )
    penalty_need_statistics_summary.update(
        {
            "patch_len": int(pred_residual_candidate_supervision_need_patch_len),
            "quantile": float(pred_residual_candidate_supervision_need_quantile),
        }
    )
    pred_residual_candidate_supervision_min_abs = float(
        pred_residual_candidate_supervision_cfg.get("min_abs_improvement", 0.0)
    )
    pred_residual_candidate_supervision_min_rel = float(
        pred_residual_candidate_supervision_cfg.get("min_rel_improvement", 0.0)
    )
    pred_residual_candidate_supervision_only_allowed = bool(
        pred_residual_candidate_supervision_cfg.get("only_allowed", True)
    )
    pred_residual_candidate_supervision_include_intervention = bool(
        pred_residual_candidate_supervision_cfg.get("include_intervention", False)
    )
    pred_residual_candidate_supervision_include_selector = bool(
        pred_residual_candidate_supervision_cfg.get("include_selector", False)
    )
    pred_residual_candidate_supervision_include_patch_route = bool(
        pred_residual_candidate_supervision_cfg.get("include_patch_route", True)
    )
    pred_residual_intervention_supervision_cfg = pred_residual_cfg.get("intervention_supervision", {}) or {}
    if not isinstance(pred_residual_intervention_supervision_cfg, dict):
        pred_residual_intervention_supervision_cfg = {"weight": float(pred_residual_intervention_supervision_cfg)}
    pred_residual_intervention_supervision_weight = (
        float(pred_residual_intervention_supervision_cfg.get("weight", 0.0))
        if pred_residual_enable
        else 0.0
    )
    pred_residual_intervention_supervision_min_gain = float(
        pred_residual_intervention_supervision_cfg.get("min_gain", 0.0)
    )
    pred_residual_intervention_supervision_pos_weight = float(
        pred_residual_intervention_supervision_cfg.get("pos_weight", 1.0)
    )
    pred_residual_intervention_supervision_only_allowed = bool(
        pred_residual_intervention_supervision_cfg.get("only_allowed", True)
    )
    pred_residual_confidence_gate_cfg = pred_residual_cfg.get("confidence_gate", {}) or {}
    if not isinstance(pred_residual_confidence_gate_cfg, dict):
        pred_residual_confidence_gate_cfg = {"enable": bool(pred_residual_confidence_gate_cfg)}
    pred_residual_confidence_gate_enable = (
        bool(pred_residual_confidence_gate_cfg.get("enable", False))
        and pred_residual_enable
        and P > 0
    )
    pred_residual_confidence_gate_source_split = "train_holdout"
    if pred_residual_confidence_gate_enable:
        pred_residual_confidence_gate_source_split = _normalize_confidence_gate_source_split(
            pred_residual_confidence_gate_cfg.get("source_split", "train_holdout")
        )
    pred_residual_confidence_gate_threshold = pred_residual_confidence_gate_cfg.get("threshold", "auto")
    pred_residual_confidence_gate_min_abs = float(
        pred_residual_confidence_gate_cfg.get(
            "min_abs_improvement",
            pred_residual_intervention_supervision_cfg.get("min_gain", 0.0),
        )
    )
    pred_residual_confidence_gate_min_rel = float(
        pred_residual_confidence_gate_cfg.get("min_rel_improvement", 0.0)
    )
    pred_residual_confidence_gate_holdout_fraction = float(
        pred_residual_confidence_gate_cfg.get("train_holdout_fraction", 0.30)
    )
    pred_residual_confidence_gate_max_candidates = int(
        pred_residual_confidence_gate_cfg.get("threshold_candidates", 101)
    )
    pred_residual_confidence_gate_selection_metric = str(
        pred_residual_confidence_gate_cfg.get("selection_metric", "mse")
    ).lower()
    pred_residual_confidence_gate_min_precision = float(
        pred_residual_confidence_gate_cfg.get("min_precision", 0.0)
    )
    pred_residual_confidence_gate_max_pred_rate_raw = pred_residual_confidence_gate_cfg.get(
        "max_pred_positive_rate",
        None,
    )
    pred_residual_confidence_gate_max_pred_rate = (
        None
        if pred_residual_confidence_gate_max_pred_rate_raw is None
        else float(pred_residual_confidence_gate_max_pred_rate_raw)
    )
    pred_residual_detach_routed_penalty_pred = (
        bool(pred_residual_cfg.get("detach_routed_penalty_pred", False)) if pred_residual_enable else False
    )
    pred_residual_freeze_gate_after_epoch = (
        int(pred_residual_cfg.get("freeze_gate_after_epoch", 0)) if pred_residual_enable else 0
    )
    pred_residual_weight_decay = None
    if pred_residual_enable:
        raw_pred_residual_wd = pred_residual_cfg.get("weight_decay", None)
        if raw_pred_residual_wd is None:
            raw_pred_residual_wd = pred_residual_cfg.get("optimizer_weight_decay", None)
        if raw_pred_residual_wd is not None:
            pred_residual_weight_decay = float(raw_pred_residual_wd)
    allow_skip = bool(moe_cfg.get("allow_skip", False)) and moe_enable and P > 0
    skip_cost = float(moe_cfg.get("skip_cost", 0.0)) if allow_skip else 0.0
    skip_init_bias = float(moe_cfg.get("skip_init_bias", -2.0))
    skip_competes = bool(
        moe_cfg.get("skip_competes_with_penalties", moe_cfg.get("noop_compete_enable", False))
    ) and allow_skip
    skip_argmax_noop = bool(moe_cfg.get("skip_argmax_noop", False)) and skip_competes
    skip_supervision_weight = float(moe_cfg.get("skip_supervision_weight", 0.0)) if allow_skip else 0.0
    skip_supervision_margin = float(moe_cfg.get("skip_supervision_margin", 0.0))
    mse_utility_gate_cfg = moe_cfg.get("mse_utility_gate_supervision", {}) or {}
    mse_utility_gate_enable = (
        bool(mse_utility_gate_cfg.get("enable", False))
        and moe_enable
        and pred_residual_enable
        and P > 0
    )
    mse_utility_gate_weight = float(mse_utility_gate_cfg.get("weight", 0.0)) if mse_utility_gate_enable else 0.0
    mse_utility_gate_temperature = float(mse_utility_gate_cfg.get("temperature", 1.0))
    mse_utility_gate_min_gain = float(mse_utility_gate_cfg.get("min_gain", 0.0))
    mse_utility_gate_mae_weight = float(mse_utility_gate_cfg.get("mae_weight", 0.0))
    mse_utility_gate_target_power = float(mse_utility_gate_cfg.get("target_power", 1.0))
    mse_utility_gate_target_mode = str(mse_utility_gate_cfg.get("target_mode", "soft_utility"))
    mse_utility_gate_include_skip = bool(
        mse_utility_gate_cfg.get("include_skip", mse_utility_gate_cfg.get("skip_aware", False))
    ) and allow_skip
    route_ce_cfg = moe_cfg.get("route_ce_supervision", {}) or {}
    if not isinstance(route_ce_cfg, dict):
        route_ce_cfg = {"enable": bool(route_ce_cfg)}
    route_ce_enable = (
        bool(route_ce_cfg.get("enable", False))
        and moe_enable
        and pred_residual_enable
        and P > 0
    )
    route_ce_weight = float(route_ce_cfg.get("weight", 0.0)) if route_ce_enable else 0.0
    route_ce_min_abs_improvement = float(route_ce_cfg.get("min_abs_improvement", 0.0))
    route_ce_min_rel_improvement = float(route_ce_cfg.get("min_rel_improvement", 0.0))
    route_ce_min_candidate_delta_rms = float(
        route_ce_cfg.get(
            "min_candidate_delta_rms",
            route_ce_cfg.get("candidate_action_floor", 0.0),
        )
    )
    route_ce_ignore_abs_gain_below = float(
        route_ce_cfg.get(
            "ignore_abs_gain_below",
            route_ce_cfg.get("ignore_near_zero_abs_gain", 0.0),
        )
    )
    route_ce_class_weight_mode = str(route_ce_cfg.get("class_weight", "none"))
    route_ce_max_class_weight = float(route_ce_cfg.get("max_class_weight", 0.0))
    route_ce_require_skip = bool(route_ce_cfg.get("require_skip", True))
    route_ce_require_skip_competes = bool(route_ce_cfg.get("require_skip_competes", True))
    route_ce_require_skip_argmax_noop = bool(route_ce_cfg.get("require_skip_argmax_noop", True))
    if route_ce_weight > 0.0:
        if route_ce_require_skip and not allow_skip:
            raise ValueError("moe.route_ce_supervision requires moe.allow_skip=true.")
        if route_ce_require_skip_competes and not skip_competes:
            raise ValueError("moe.route_ce_supervision requires moe.skip_competes_with_penalties=true.")
        if route_ce_require_skip_argmax_noop and not skip_argmax_noop:
            raise ValueError("moe.route_ce_supervision requires moe.skip_argmax_noop=true.")
    binary_adoption_cfg = moe_cfg.get("binary_adoption_supervision", {}) or {}
    if not isinstance(binary_adoption_cfg, dict):
        binary_adoption_cfg = {"enable": bool(binary_adoption_cfg)}
    binary_adoption_enable = (
        bool(binary_adoption_cfg.get("enable", False))
        and moe_enable
        and pred_residual_enable
        and P > 0
    )
    binary_adoption_weight = (
        float(binary_adoption_cfg.get("weight", 0.0)) if binary_adoption_enable else 0.0
    )
    binary_adoption_min_abs_improvement = float(
        binary_adoption_cfg.get("min_abs_improvement", route_ce_min_abs_improvement)
    )
    binary_adoption_min_rel_improvement = float(
        binary_adoption_cfg.get("min_rel_improvement", route_ce_min_rel_improvement)
    )
    binary_adoption_min_candidate_delta_rms = float(
        binary_adoption_cfg.get("min_candidate_delta_rms", route_ce_min_candidate_delta_rms)
    )
    binary_adoption_ignore_abs_gain_below = float(
        binary_adoption_cfg.get("ignore_abs_gain_below", route_ce_ignore_abs_gain_below)
    )
    binary_adoption_positive_weight = float(binary_adoption_cfg.get("positive_weight", 1.0))
    binary_adoption_negative_weight = float(binary_adoption_cfg.get("negative_weight", 1.0))
    binary_adoption_require_skip = bool(binary_adoption_cfg.get("require_skip", True))
    binary_adoption_require_skip_competes = bool(binary_adoption_cfg.get("require_skip_competes", True))
    binary_adoption_require_skip_argmax_noop = bool(
        binary_adoption_cfg.get("require_skip_argmax_noop", True)
    )
    if binary_adoption_weight > 0.0:
        if binary_adoption_require_skip and not allow_skip:
            raise ValueError("moe.binary_adoption_supervision requires moe.allow_skip=true.")
        if binary_adoption_require_skip_competes and not skip_competes:
            raise ValueError(
                "moe.binary_adoption_supervision requires moe.skip_competes_with_penalties=true."
            )
        if binary_adoption_require_skip_argmax_noop and not skip_argmax_noop:
            raise ValueError("moe.binary_adoption_supervision requires moe.skip_argmax_noop=true.")
    route_rate_alignment_cfg = moe_cfg.get("route_rate_alignment_supervision", {}) or {}
    if not isinstance(route_rate_alignment_cfg, dict):
        route_rate_alignment_cfg = {"enable": bool(route_rate_alignment_cfg)}
    route_rate_alignment_enable = (
        bool(route_rate_alignment_cfg.get("enable", False))
        and moe_enable
        and pred_residual_enable
        and P > 0
    )
    route_rate_alignment_weight = (
        float(route_rate_alignment_cfg.get("weight", 0.0)) if route_rate_alignment_enable else 0.0
    )
    route_rate_alignment_min_abs_improvement = float(
        route_rate_alignment_cfg.get("min_abs_improvement", binary_adoption_min_abs_improvement)
    )
    route_rate_alignment_min_rel_improvement = float(
        route_rate_alignment_cfg.get("min_rel_improvement", binary_adoption_min_rel_improvement)
    )
    route_rate_alignment_min_candidate_delta_rms = float(
        route_rate_alignment_cfg.get(
            "min_candidate_delta_rms",
            binary_adoption_min_candidate_delta_rms,
        )
    )
    route_rate_alignment_ignore_abs_gain_below = float(
        route_rate_alignment_cfg.get("ignore_abs_gain_below", binary_adoption_ignore_abs_gain_below)
    )
    route_rate_alignment_require_skip = bool(route_rate_alignment_cfg.get("require_skip", True))
    route_rate_alignment_require_skip_competes = bool(
        route_rate_alignment_cfg.get("require_skip_competes", True)
    )
    route_rate_alignment_require_skip_argmax_noop = bool(
        route_rate_alignment_cfg.get("require_skip_argmax_noop", True)
    )
    if route_rate_alignment_weight > 0.0:
        if route_rate_alignment_require_skip and not allow_skip:
            raise ValueError("moe.route_rate_alignment_supervision requires moe.allow_skip=true.")
        if route_rate_alignment_require_skip_competes and not skip_competes:
            raise ValueError(
                "moe.route_rate_alignment_supervision requires "
                "moe.skip_competes_with_penalties=true."
            )
        if route_rate_alignment_require_skip_argmax_noop and not skip_argmax_noop:
            raise ValueError(
                "moe.route_rate_alignment_supervision requires moe.skip_argmax_noop=true."
            )
    route_positive_recall_cfg = moe_cfg.get("route_positive_recall_supervision", {}) or {}
    if not isinstance(route_positive_recall_cfg, dict):
        route_positive_recall_cfg = {"enable": bool(route_positive_recall_cfg)}
    route_positive_recall_enable = (
        bool(route_positive_recall_cfg.get("enable", False))
        and moe_enable
        and pred_residual_enable
        and P > 0
    )
    route_positive_recall_weight = (
        float(route_positive_recall_cfg.get("weight", 0.0)) if route_positive_recall_enable else 0.0
    )
    route_positive_recall_min_abs_improvement = float(
        route_positive_recall_cfg.get("min_abs_improvement", binary_adoption_min_abs_improvement)
    )
    route_positive_recall_min_rel_improvement = float(
        route_positive_recall_cfg.get("min_rel_improvement", binary_adoption_min_rel_improvement)
    )
    route_positive_recall_min_candidate_delta_rms = float(
        route_positive_recall_cfg.get(
            "min_candidate_delta_rms",
            binary_adoption_min_candidate_delta_rms,
        )
    )
    route_positive_recall_ignore_abs_gain_below = float(
        route_positive_recall_cfg.get("ignore_abs_gain_below", binary_adoption_ignore_abs_gain_below)
    )
    route_positive_recall_mode = str(route_positive_recall_cfg.get("mode", "ce"))
    route_positive_recall_target_probability = float(
        route_positive_recall_cfg.get("target_probability", 1.0)
    )
    route_positive_recall_require_skip = bool(route_positive_recall_cfg.get("require_skip", True))
    route_positive_recall_require_skip_competes = bool(
        route_positive_recall_cfg.get("require_skip_competes", True)
    )
    route_positive_recall_require_skip_argmax_noop = bool(
        route_positive_recall_cfg.get("require_skip_argmax_noop", True)
    )
    if route_positive_recall_weight > 0.0:
        if route_positive_recall_require_skip and not allow_skip:
            raise ValueError("moe.route_positive_recall_supervision requires moe.allow_skip=true.")
        if route_positive_recall_require_skip_competes and not skip_competes:
            raise ValueError(
                "moe.route_positive_recall_supervision requires "
                "moe.skip_competes_with_penalties=true."
            )
        if route_positive_recall_require_skip_argmax_noop and not skip_argmax_noop:
            raise ValueError(
                "moe.route_positive_recall_supervision requires moe.skip_argmax_noop=true."
            )
    route_precision_recall_cfg = moe_cfg.get("route_precision_recall_supervision", {}) or {}
    if not isinstance(route_precision_recall_cfg, dict):
        route_precision_recall_cfg = {"enable": bool(route_precision_recall_cfg)}
    route_precision_recall_enable = (
        bool(route_precision_recall_cfg.get("enable", False))
        and moe_enable
        and pred_residual_enable
        and P > 0
    )
    route_precision_recall_weight = (
        float(route_precision_recall_cfg.get("weight", 0.0)) if route_precision_recall_enable else 0.0
    )
    route_precision_recall_min_abs_improvement = float(
        route_precision_recall_cfg.get("min_abs_improvement", binary_adoption_min_abs_improvement)
    )
    route_precision_recall_min_rel_improvement = float(
        route_precision_recall_cfg.get("min_rel_improvement", binary_adoption_min_rel_improvement)
    )
    route_precision_recall_min_candidate_delta_rms = float(
        route_precision_recall_cfg.get(
            "min_candidate_delta_rms",
            binary_adoption_min_candidate_delta_rms,
        )
    )
    route_precision_recall_ignore_abs_gain_below = float(
        route_precision_recall_cfg.get("ignore_abs_gain_below", binary_adoption_ignore_abs_gain_below)
    )
    route_precision_recall_mode = str(route_precision_recall_cfg.get("recall_mode", "ce"))
    route_precision_recall_target_probability = float(
        route_precision_recall_cfg.get("recall_target_probability", 1.0)
    )
    route_precision_recall_false_adopt_max_probability = float(
        route_precision_recall_cfg.get("false_adopt_max_probability", 0.5)
    )
    route_precision_recall_false_adopt_weight = float(
        route_precision_recall_cfg.get("false_adopt_weight", 1.0)
    )
    route_precision_recall_require_skip = bool(route_precision_recall_cfg.get("require_skip", True))
    route_precision_recall_require_skip_competes = bool(
        route_precision_recall_cfg.get("require_skip_competes", True)
    )
    route_precision_recall_require_skip_argmax_noop = bool(
        route_precision_recall_cfg.get("require_skip_argmax_noop", True)
    )
    if route_precision_recall_weight > 0.0:
        if route_precision_recall_require_skip and not allow_skip:
            raise ValueError("moe.route_precision_recall_supervision requires moe.allow_skip=true.")
        if route_precision_recall_require_skip_competes and not skip_competes:
            raise ValueError(
                "moe.route_precision_recall_supervision requires "
                "moe.skip_competes_with_penalties=true."
            )
        if route_precision_recall_require_skip_argmax_noop and not skip_argmax_noop:
            raise ValueError(
                "moe.route_precision_recall_supervision requires moe.skip_argmax_noop=true."
            )
    raw_ranks = moe_cfg.get("select_ranks", None)
    if raw_ranks is None:
        select_ranks = [1, 2]
    else:
        select_ranks = [int(x) for x in raw_ranks]
    gate_feat_dim = len(_gate_feature_names_for_mode(gate_feature_mode))
    gate = ClusterwiseMoEGate(
        num_clusters=K,
        feat_dim=gate_feat_dim,
        num_penalties=P,
        hidden_dim=int(moe_cfg.get("gate_hidden_dim", 64)),
        topk=int(moe_cfg["topk"]),
        allow_skip=allow_skip,
        skip_init_bias=skip_init_bias,
        skip_competes=skip_competes,
        skip_argmax_noop=skip_argmax_noop,
        shared_across_clusters=shared_moe_across_clusters,
    ).to(device)
    if shared_moe_across_clusters:
        print("Shared MoE across clusters enabled: gate and prediction residual expert parameters are shared.")
    gate.temperature = float(moe_cfg.get("gate_temperature", 1.0))
    gate.noise_std = float(moe_cfg.get("gate_noise_std", 0.0))
    gate.logit_clip = float(moe_cfg.get("gate_logit_clip", 0.0))
    gate.prob_floor = float(moe_cfg.get("gate_prob_floor", 0.0))
    gate_init_bias_cfg = moe_cfg.get("gate_init_bias", {}) or {}
    if P > 0 and bool(gate_init_bias_cfg.get("enable", False)):
        raw_bias = gate_init_bias_cfg.get("values", {}) or {}
        default_bias = float(raw_bias.get("default", 0.0)) if isinstance(raw_bias, dict) else 0.0
        bias_p = torch.tensor(
            [
                float(raw_bias.get(name, default_bias)) if isinstance(raw_bias, dict) else default_bias
                for name in penalty_names
            ],
            device=device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            for b2 in gate.b2:
                b2.add_(bias_p)
        print(f"Gate init bias applied: {dict(zip(penalty_names, [float(v) for v in bias_p.detach().cpu().tolist()]))}")
    channel_expert_mask_c = None
    channel_expert_cfg = pred_residual_cfg.get("channel_expert_adapters", {}) or {}
    if pred_residual_enable and bool(channel_expert_cfg.get("enable", False)):
        raw_cluster_id_c, _ = cluster_channels_by_corr(
            corr_cc=corr_cc,
            data_tc=cluster_fit_tc,
            n_clusters=cl.get("n_clusters", None),
            distance_threshold=cl.get("distance_threshold", None),
            linkage=cl.get("linkage", "average"),
            method=cl.get("method", "agglomerative"),
            kmeans_n_init=int(cl.get("kmeans_n_init", 10)),
            kmeans_max_iter=int(cl.get("kmeans_max_iter", 300)),
            spectral_affinity=cl.get("spectral_affinity", "corr"),
            rbf_gamma=float(cl.get("rbf_gamma", 1.0)),
            dbscan_eps=cl.get("dbscan_eps", None),
            dbscan_min_samples=int(cl.get("dbscan_min_samples", 5)),
            random_state=None if rs is None else int(rs),
            min_cluster_size=1,
            merge_small_clusters=False,
            no_merge_if_channels_lt=int(cl["no_merge_if_channels_lt"]),
            extra_features_cf=cluster_extra_features_cf,
            feature_weight=float(feature_aware_cfg.get("weight", 0.0)) if cluster_extra_features_cf is not None else 0.0,
        )
        raw_sizes = torch.bincount(raw_cluster_id_c, minlength=int(raw_cluster_id_c.max().item() + 1))
        final_sizes = torch.bincount(cluster_id_c, minlength=K)
        mode = str(channel_expert_cfg.get("mode", "merged_singletons")).lower()
        if mode in {"all", "all_channels"}:
            channel_expert_mask_c = torch.ones(C, dtype=torch.bool, device=device)
        else:
            channel_expert_mask_c = (
                (raw_sizes[raw_cluster_id_c].to(device=device) == 1)
                & (final_sizes[cluster_id_c].to(device=device) > 1)
            )
        print(
            "Channel expert adapters enabled: "
            f"mode={mode}, channels={int(channel_expert_mask_c.sum().item())}/{C}, "
            f"mask={[bool(v) for v in channel_expert_mask_c.detach().cpu().tolist()]}"
        )
    pred_residual = None
    if pred_residual_enable:
        pred_residual = ClusterwisePredResidualMoE(
            num_clusters=K,
            num_penalties=P,
            input_len=L,
            pred_len=H,
            hidden_dim=int(pred_residual_cfg.get("corrector_hidden", 32)),
            init_alpha=float(pred_residual_cfg.get("init_alpha", -3.0)),
            alpha_scale=float(pred_residual_cfg.get("alpha_scale", 0.5)),
            use_y_base_input=bool(pred_residual_cfg.get("use_y_base_input", True)),
            use_channel_identity_features=bool(
                pred_residual_cfg.get("use_channel_identity_features", False)
            ),
            feature_mode=str(pred_residual_cfg.get("feature_mode", "legacy")),
            residual_clip=float(pred_residual_cfg.get("residual_clip", 0.0)),
            intervention_enable=bool(pred_residual_cfg.get("intervention_enable", False)),
            intervention_init=float(pred_residual_cfg.get("intervention_init", -2.0)),
            penalty_selector_enable=bool(pred_residual_cfg.get("penalty_selector_enable", False)),
            selector_temperature=float(pred_residual_cfg.get("selector_temperature", 1.0)),
            selector_use_cluster_context=bool(pred_residual_cfg.get("selector_use_cluster_context", True)),
            fusion_gate_enable=bool(pred_residual_cfg.get("fusion_gate_enable", False)),
            fusion_init=float(pred_residual_cfg.get("fusion_init", 0.0)),
            fusion_use_cluster_context=bool(pred_residual_cfg.get("fusion_use_cluster_context", True)),
            num_channels=C,
            channel_expert_mask_c=channel_expert_mask_c,
            channel_expert_cluster_id_c=cluster_id_c,
            channel_expert_mode=str((pred_residual_cfg.get("channel_expert_adapters", {}) or {}).get("mode_type", "override")),
            penalty_names=penalty_names,
            seasonal_anchor_names=list(pred_residual_cfg.get("seasonal_anchor_names", [])),
            seasonal_anchor_period=int(pred_residual_cfg.get("seasonal_anchor_period", 96)),
            seasonal_anchor_num_periods=int(pred_residual_cfg.get("seasonal_anchor_num_periods", 1)),
            seasonal_anchor_scale=float(pred_residual_cfg.get("seasonal_anchor_scale", 1.0)),
            phase_residual_candidate_names=phase_residual_candidate_names,
            phase_residual_candidate_scale=phase_residual_candidate_scale,
            shared_across_clusters=shared_moe_across_clusters,
            semantic_output_head_scope=pred_residual_semantic_output_head_scope,
            patch_router_cfg=patch_router_cfg,
            named_output_projection_enable=named_output_projection_enable,
            named_output_projection_fixed_alpha=named_output_projection_fixed_alpha,
            named_output_projection_scale_by_name=named_output_projection_scale_by_name,
            named_output_projection_carrier_names=named_output_projection_carrier_names,
            named_output_projection_patch_len=named_output_projection_patch_len,
            named_output_projection_mode=named_output_projection_mode,
            named_output_projection_diff_amp_max=named_output_projection_diff_amp_max,
            periodic_anchor_expert_enable=periodic_anchor_expert_enable,
            periodic_anchor_expert_scale=periodic_anchor_expert_scale,
            position_daily_residual_expert_enable=position_daily_residual_expert_enable,
            position_daily_residual_period=int(
                position_daily_residual_expert_cfg.get("daily_period", 96)
            ),
            position_daily_residual_harmonics=int(
                position_daily_residual_expert_cfg.get("daily_harmonics", 4)
            ),
            anchor_ridge_gate_cfg=anchor_ridge_gate_cfg,
            semantic_level_controller_cfg=pred_residual_semantic_level_controller_cfg,
        ).to(device)
        if (
            pred_residual.patch_router is not None
            and pred_residual.patch_router.regime_context_enable
        ):
            pred_residual.set_patch_router_observed_history(data_window_tc)
        print(
            "Prediction residual MoE enabled: "
            f"hidden={pred_residual.hidden_dim}, feature_mode={pred_residual.feature_mode}, "
            f"channel_identity={bool(pred_residual.use_channel_identity_features)}, "
            f"alpha_scale={pred_residual.alpha_scale:.3f}, "
            f"residual_clip={pred_residual.residual_clip:.3f}, "
            f"seasonal_anchor_names={list(pred_residual_cfg.get('seasonal_anchor_names', []))}, "
            f"seasonal_anchor_period={int(pred_residual_cfg.get('seasonal_anchor_period', 96))}, "
            f"seasonal_anchor_scale={float(pred_residual_cfg.get('seasonal_anchor_scale', 1.0)):.3f}, "
            f"phase_residual_candidate={phase_residual_candidate_names}, "
            f"phase_residual_period={int(phase_residual_candidate_period)}, "
            f"phase_residual_scale={float(phase_residual_candidate_scale):.3f}, "
            f"named_output_projection={bool(named_output_projection_enable)}, "
            f"named_output_projection_fixed_alpha={bool(named_output_projection_fixed_alpha)}, "
            f"named_output_projection_carrier_names={named_output_projection_carrier_names}, "
            f"named_output_projection_patch_len={named_output_projection_patch_len}, "
            f"named_output_projection_mode={named_output_projection_mode}, "
            f"periodic_anchor_expert={bool(periodic_anchor_expert_enable)}, "
            f"periodic_anchor_scale={float(periodic_anchor_expert_scale):.3f}, "
            f"position_daily_residual_expert={bool(position_daily_residual_expert_enable)}, "
            f"anchor_ridge_gate={bool(anchor_ridge_gate_enable)}, "
            f"patch_router={bool(pred_residual.patch_router is not None)}, "
            f"patch_len={int(pred_residual.patch_router.patch_len) if pred_residual.patch_router is not None else 0}, "
            f"history_projection={pred_residual.patch_router.history_patch_projection if pred_residual.patch_router is not None else 'none'}, "
            f"regime_context={pred_residual.patch_router.regime_context_lengths if pred_residual.patch_router is not None else []}, "
            f"specialization_weight={pred_residual_specialization_weight:.6f}, "
            f"norm_weight={pred_residual_norm_weight:.6f}, "
            f"intervention_weight={pred_residual_intervention_weight:.6f}, "
            f"candidate_supervision_weight={pred_residual_candidate_supervision_weight:.6f}, "
            f"candidate_supervision_loss={pred_residual_candidate_supervision_loss}, "
            f"candidate_supervision_include_patch_route={pred_residual_candidate_supervision_include_patch_route}, "
            f"ignore_skip_during_training={pred_residual_ignore_skip_during_training}, "
            f"intervention_supervision_weight={pred_residual_intervention_supervision_weight:.6f}, "
            f"route_ce_weight={route_ce_weight:.6f}, "
            f"route_ce_min_candidate_delta_rms={route_ce_min_candidate_delta_rms:.6g}, "
            f"binary_adoption_weight={binary_adoption_weight:.6f}, "
            f"binary_adoption_min_candidate_delta_rms={binary_adoption_min_candidate_delta_rms:.6g}, "
            f"confidence_gate={bool(pred_residual_confidence_gate_enable)}, "
            f"freeze_gate_after_epoch={int(pred_residual_freeze_gate_after_epoch)}, "
            f"detach_routed_penalty_pred={pred_residual_detach_routed_penalty_pred}, "
            f"penalty_selector={pred_residual.penalty_selector_enable}, "
            f"fusion_gate={pred_residual.fusion_gate_enable}"
        )
    learnable_output_anchor_cfg = _normalize_learnable_output_anchor_cfg(
        moe_cfg.get("learnable_output_anchor", {})
    )
    moe_cfg["learnable_output_anchor"] = learnable_output_anchor_cfg
    learnable_output_anchor_enable = bool(learnable_output_anchor_cfg.get("enable", False))
    learnable_output_anchor_train_mode = str(learnable_output_anchor_cfg.get("train_mode", "joint")).lower()
    if learnable_output_anchor_train_mode in {"anchor-only", "anchor_only", "posthoc", "post_hoc"}:
        learnable_output_anchor_train_mode = "anchor_only"
    if learnable_output_anchor_train_mode not in {"joint", "anchor_only"}:
        raise ValueError("moe.learnable_output_anchor.train_mode must be joint or anchor_only.")
    learnable_output_anchor = None
    learnable_output_anchor_summary: Dict[str, object] = {
        "enable": bool(learnable_output_anchor_enable),
        "source": "static_output_anchor_refiner" if learnable_output_anchor_enable else None,
        "num_clusters": int(K),
        "num_channels": int(C),
        "pred_len": int(H),
        "train_mode": str(learnable_output_anchor_train_mode) if learnable_output_anchor_enable else None,
        "train_with_eval_anchors": bool(learnable_output_anchor_enable),
        "final_eval_enable": bool(learnable_output_anchor_enable),
        "adoption_guard_applied": False,
        "loaded_from_checkpoint": False,
    }
    if learnable_output_anchor_enable:
        learnable_output_anchor = ClusterwiseLearnableOutputAnchor(
            num_clusters=K,
            num_channels=C,
            pred_len=H,
            cfg=learnable_output_anchor_cfg,
        ).to(device)
        learnable_output_anchor_summary.update(
            {
                "max_scale_delta": float(learnable_output_anchor.max_scale_delta),
                "learn_stat_scale": bool(learnable_output_anchor.learn_stat_scale),
                "learn_residual_scale": bool(learnable_output_anchor.learn_residual_scale),
                "learn_bias": bool(learnable_output_anchor.learn_bias),
                "max_bias": float(learnable_output_anchor.max_bias),
                "learn_history_trend": bool(learnable_output_anchor.learn_history_trend),
                "max_history_trend_delta": float(learnable_output_anchor.max_history_trend_delta),
                "history_trend_window": int(learnable_output_anchor.history_trend_window),
                "history_trend_feature": str(learnable_output_anchor.history_trend_feature),
                "history_trend_projection": str(learnable_output_anchor.history_trend_projection),
                "scale_parameterization": str(learnable_output_anchor.scale_parameterization),
                "bias_parameterization": str(learnable_output_anchor.bias_parameterization),
                "history_trend_parameterization": str(learnable_output_anchor.history_trend_parameterization),
                "scale_temporal_basis_rank": int(learnable_output_anchor.scale_temporal_basis_rank),
                "trainable_params": int(
                    sum(param.numel() for param in learnable_output_anchor.parameters() if param.requires_grad)
                ),
                "parameter_shape_per_cluster": {
                    "scale": [int(v) for v in learnable_output_anchor.scale_shape],
                    "bias": [int(v) for v in learnable_output_anchor.bias_shape],
                    "history_trend": [int(v) for v in learnable_output_anchor.history_trend_shape],
                    "history_trend_basis": [
                        int(learnable_output_anchor.history_trend_basis_h.shape[0]),
                    ],
                    "scale_temporal_coef": [
                        int(learnable_output_anchor.scale_shape[0]),
                        int(learnable_output_anchor.scale_temporal_basis_rank),
                    ],
                    "scale_temporal_basis": [
                        int(learnable_output_anchor.scale_temporal_basis_rh.shape[0]),
                        int(learnable_output_anchor.scale_temporal_basis_rh.shape[1]),
                    ],
                },
                "zero_init_static_equivalent": True,
            }
        )
        print(
            "Learnable output anchor enabled: "
            f"max_scale_delta={learnable_output_anchor.max_scale_delta:.3f}, "
            f"learn_stat_scale={bool(learnable_output_anchor.learn_stat_scale)}, "
            f"learn_residual_scale={bool(learnable_output_anchor.learn_residual_scale)}, "
            f"learn_bias={bool(learnable_output_anchor.learn_bias)}, "
            f"max_bias={learnable_output_anchor.max_bias:.3f}, "
            f"learn_history_trend={bool(learnable_output_anchor.learn_history_trend)}, "
            f"max_history_trend_delta={learnable_output_anchor.max_history_trend_delta:.3f}, "
            f"scale_parameterization={learnable_output_anchor.scale_parameterization}, "
            f"scale_temporal_basis_rank={learnable_output_anchor.scale_temporal_basis_rank}, "
            "zero_init_static_equivalent=True"
        )

    if pred_residual_require_pure_backbone_base:
        active_output_modifiers = []
        if history_anchor_active:
            active_output_modifiers.append("model.history_anchor")
        if bool(calendar_residual_cfg.get("enable", False)):
            active_output_modifiers.append("calendar_residual")
        if bool(position_daily_residual_cfg.get("enable", False)):
            active_output_modifiers.append("position_daily_residual_ridge")
        if bool(model_train_stat_adapter_cfg.get("enable", False)):
            active_output_modifiers.append("model.train_stat_adapter")
        if bool(train_stat_anchor_cfg.get("enable", False)):
            active_output_modifiers.append("moe.train_stat_anchor_expert")
        if bool(train_residual_anchor_cfg.get("enable", False)):
            active_output_modifiers.append("moe.train_residual_anchor_expert")
        if learnable_output_anchor_enable:
            active_output_modifiers.append("moe.learnable_output_anchor")
        if periodic_anchor_expert_enable:
            active_output_modifiers.append("pred_side_residual.periodic_anchor_expert")
        if position_daily_residual_expert_enable:
            active_output_modifiers.append("pred_side_residual.position_daily_residual_expert")
        if anchor_ridge_gate_enable:
            active_output_modifiers.append("pred_side_residual.anchor_ridge_gate")
        if list(pred_residual_cfg.get("seasonal_anchor_names", []) or []):
            active_output_modifiers.append("pred_side_residual.seasonal_anchor_names")
        if phase_residual_candidate_enable:
            active_output_modifiers.append("pred_side_residual.phase_residual_candidate")
        if active_output_modifiers:
            raise ValueError(
                "require_pure_backbone_base=true forbids output modifiers: "
                + ", ".join(active_output_modifiers)
            )

    if pred_residual_semantic_bank_stage1:
        expected_penalties = ["level", "trend", "d2_match", "diff_amp"]
        if not pred_residual_require_pure_backbone_base:
            raise ValueError("semantic_bank_stage1 requires require_pure_backbone_base=true.")
        if not shared_moe_across_clusters:
            raise ValueError(
                "semantic_bank_stage1 independent optimization requires shared_across_clusters=true."
            )
        if pred_residual_semantic_output_head_scope == "per_cluster":
            forbidden_semantic_output_features = []
            if bool(pred_residual_cfg.get("intervention_enable", False)):
                forbidden_semantic_output_features.append("intervention_enable")
            if bool(pred_residual_cfg.get("penalty_selector_enable", False)):
                forbidden_semantic_output_features.append("penalty_selector_enable")
            if bool(pred_residual_cfg.get("fusion_gate_enable", False)):
                forbidden_semantic_output_features.append("fusion_gate_enable")
            if bool((pred_residual_cfg.get("channel_expert_adapters", {}) or {}).get("enable", False)):
                forbidden_semantic_output_features.append("channel_expert_adapters")
            if forbidden_semantic_output_features:
                raise ValueError(
                    "semantic per-cluster Stage-1 producer forbids: "
                    + ", ".join(forbidden_semantic_output_features)
                )
        if penalty_names != expected_penalties:
            raise ValueError(
                "semantic_bank_stage1 requires penalty order "
                f"{expected_penalties}, got {penalty_names}."
            )
        if bool(patch_router_cfg.get("enable", False)):
            raise ValueError("semantic_bank_stage1 requires patch_router.enable=false.")
        if not (
            named_output_projection_enable
            and named_output_projection_fixed_alpha
            and named_output_projection_mode == "semantic_residual"
            and named_output_projection_patch_len == 12
        ):
            raise ValueError(
                "semantic_bank_stage1 requires semantic_residual projection, fixed alpha, and patch_len=12."
            )
        effective_scales = [
            float(named_output_projection_scale_by_name.get(name, 1.0))
            for name in penalty_names
        ]
        if any(value != 1.0 for value in effective_scales):
            raise ValueError("semantic_bank_stage1 requires all fixed projection scales to equal 1.0.")
        semantic_level_controller_enable = bool(
            pred_residual_semantic_level_controller_cfg.get("enable", False)
        )
        if semantic_level_controller_enable and pred_residual_semantic_active_penalty == "level":
            if pred_residual_semantic_level_separate_need_gate:
                required_level_loss = (
                    "level_residual_high_need_separate_gate"
                    if pred_residual_semantic_level_acceptance_candidate_source
                    == "raw_amplitude"
                    else "level_residual_separate_gate"
                )
            else:
                required_level_loss = "level_residual_gate"
                if (
                    pred_residual_semantic_level_acceptance_candidate_source
                    != "executed"
                ):
                    raise ValueError(
                        "raw LEVEL acceptance requires separate_need_gate=true."
                    )
            if pred_residual_candidate_supervision_loss != required_level_loss:
                raise ValueError(
                    "semantic_bank_stage1 LEVEL controller mode requires "
                    f"{required_level_loss} supervision."
                )
            if int(pred_residual_semantic_level_controller_cfg.get("patch_len", 12)) != 12:
                raise ValueError("semantic_bank_stage1 LEVEL controller requires patch_len=12.")
            if pred_residual_semantic_level_separate_need_gate:
                if abs(pred_residual_semantic_level_need_positive_weight - 3.0) > 1.0e-12:
                    raise ValueError(
                        "separate LEVEL gate requires level_need_positive_weight=3.0 for q75."
                    )
                if not pred_residual_semantic_raw_gradient_accumulation:
                    raise ValueError(
                        "separate LEVEL gate requires raw-gradient accumulation."
                    )
                if pred_residual_semantic_level_amplitude_optimizer_name not in {
                    "sgd",
                    "adam",
                }:
                    raise ValueError(
                        "separate LEVEL amplitude optimizer must be sgd or adam."
                    )
                if pred_residual_semantic_level_amplitude_optimizer_name == "adam":
                    if (
                        pred_residual_semantic_level_amplitude_lr_raw is None
                        or abs(
                            float(pred_residual_semantic_level_amplitude_lr_raw)
                            - 1.0e-3
                        )
                        > 1.0e-12
                    ):
                        raise ValueError(
                            "reviewed LEVEL amplitude Adam repair requires lr=0.001."
                        )
                    if (
                        pred_residual_semantic_level_amplitude_weight_decay_raw
                        is None
                        or abs(
                            float(
                                pred_residual_semantic_level_amplitude_weight_decay_raw
                            )
                        )
                        > 1.0e-12
                    ):
                        raise ValueError(
                            "reviewed LEVEL amplitude Adam repair requires weight_decay=0."
                        )
            elif pred_residual_semantic_level_amplitude_optimizer_cfg:
                raise ValueError(
                    "LEVEL amplitude optimizer override requires separate_need_gate=true."
                )
        else:
            if pred_residual_semantic_level_amplitude_optimizer_cfg:
                raise ValueError(
                    "LEVEL amplitude optimizer override requires the active disjoint "
                    "LEVEL controller."
                )
            if pred_residual_candidate_supervision_loss in {
                "level_residual_gate",
                "level_residual_separate_gate",
                "level_residual_high_need_separate_gate",
            }:
                raise ValueError(
                    "LEVEL controller supervision requires an enabled controller and active_penalty=level."
                )
            if pred_residual_candidate_supervision_loss != "high_need_own_penalty":
                raise ValueError(
                    "legacy semantic_bank_stage1 staged repair requires high_need_own_penalty supervision."
                )
        if not pred_residual_candidate_supervision_independent_optimization:
            raise ValueError(
                "semantic_bank_stage1 requires independent_optimization=true."
            )
        expected_independent_optimizer = (
            "mixed_level_adam_amplitude_sgd_need_gate"
            if semantic_level_controller_enable
            and pred_residual_semantic_active_penalty == "level"
            and pred_residual_semantic_level_separate_need_gate
            and pred_residual_semantic_level_amplitude_optimizer_name == "adam"
            else "sgd"
        )
        if (
            pred_residual_candidate_supervision_independent_optimizer
            != expected_independent_optimizer
        ):
            raise ValueError(
                "semantic_bank_stage1 independent_optimizer mismatch: expected "
                f"{expected_independent_optimizer!r}."
            )
        if abs(pred_residual_candidate_supervision_weight - 1.0) > 1.0e-12:
            raise ValueError(
                "semantic_bank_stage1 requires adapter_attribute_supervision.weight=1."
            )
        if abs(pred_residual_candidate_supervision_forecast_mse_weight) > 1.0e-12:
            raise ValueError(
                "semantic_bank_stage1 semantic-only repair requires forecast_mse_weight=0."
            )
        if abs(pred_residual_candidate_supervision_noop_weight) > 1.0e-12:
            raise ValueError("semantic_bank_stage1 semantic-only repair requires noop_weight=0.")
        if pred_residual_candidate_supervision_high_mse_relative_tolerance != 0.0:
            raise ValueError(
                "semantic_bank_stage1 requires high_mse_relative_tolerance=0."
            )
        if abs(pred_residual_candidate_supervision_low_mse_relative_tolerance - 1.0e-3) > 1.0e-12:
            raise ValueError(
                "semantic_bank_stage1 requires low_mse_relative_tolerance=0.001."
            )
        if abs(pred_residual_candidate_supervision_low_high_rms_ratio_max - 0.25) > 1.0e-12:
            raise ValueError(
                "semantic_bank_stage1 requires low_high_rms_ratio_max=0.25."
            )
        if pred_residual_candidate_supervision_need_patch_len != 12:
            raise ValueError("semantic_bank_stage1 requires need_patch_len=12.")
        if abs(pred_residual_candidate_supervision_need_quantile - 0.75) > 1.0e-12:
            raise ValueError("semantic_bank_stage1 requires need_quantile=0.75.")
        if pred_residual_semantic_active_penalty not in expected_penalties:
            raise ValueError(
                "semantic_bank_stage1 requires one active_penalty from "
                f"{expected_penalties}, got {pred_residual_semantic_active_penalty!r}."
            )
        if pred_residual_semantic_validation_blocks != 3:
            raise ValueError("semantic_bank_stage1 requires validation_blocks=3.")
        expected_min_gains = {
            "level": 0.10,
            "trend": 0.10,
            "d2_match": 0.10,
            "diff_amp": 0.70,
        }
        if pred_residual_semantic_min_gain_by_name != expected_min_gains:
            raise ValueError(
                "semantic_bank_stage1 materiality thresholds must equal "
                f"{expected_min_gains}."
            )
        if abs(pred_residual_semantic_min_improved_fraction - 0.60) > 1.0e-12:
            raise ValueError(
                "semantic_bank_stage1 requires min_high_need_improved_fraction=0.60."
            )
        if pred_residual_candidate_supervision_only_allowed:
            raise ValueError("semantic_bank_stage1 requires only_allowed=false.")
        if pred_residual_candidate_supervision_include_intervention:
            raise ValueError(
                "semantic_bank_stage1 candidate loss must exclude intervention."
            )
        if pred_residual_candidate_supervision_include_selector:
            raise ValueError(
                "semantic_bank_stage1 candidate loss must exclude selector."
            )
        if pred_residual_candidate_supervision_include_patch_route:
            raise ValueError("semantic_bank_stage1 candidate loss must exclude patch routing.")
        if penalty_scale_source != "frozen_backbone_patch":
            raise ValueError(
                "semantic_bank_stage1 requires train.penalty_scale_source=frozen_backbone_patch."
            )
        if abs(float(cfg["train"].get("penalty_scale_floor", 1.0e-3)) - 1.0e-3) > 1.0e-12:
            raise ValueError("semantic_bank_stage1 requires penalty_scale_floor=0.001.")
        if str(memory_cfg.get("checkpoint_selection", "")).lower() != "semantic_per_expert":
            raise ValueError(
                "semantic_bank_stage1 requires memory.checkpoint_selection=semantic_per_expert."
            )
        if str((cfg["train"].get("lr_scheduler", {}) or {}).get("name", "none")).lower() not in {
            "none",
            "off",
            "disabled",
        }:
            raise ValueError(
                "semantic_bank_stage1 requires a loss-independent scheduler (name=none)."
            )
        if pred_residual_semantic_raw_gradient_accumulation:
            if pred_residual_semantic_active_penalty != "level":
                raise ValueError(
                    "semantic raw-gradient accumulation is currently active-level only."
                )
            if pred_residual_semantic_raw_gradient_microbatches != 16:
                raise ValueError(
                    "semantic raw-gradient accumulation requires microbatches=16."
                )
    else:
        if pred_residual_semantic_raw_gradient_accumulation:
            raise ValueError(
                "semantic raw-gradient accumulation requires semantic_bank_stage1=true."
            )
        if pred_residual_semantic_final_train_audit:
            raise ValueError(
                "final_train_semantic_audit requires semantic_bank_stage1=true."
            )

    semantic_bank_active_penalty_index = (
        penalty_names.index(pred_residual_semantic_active_penalty)
        if pred_residual_semantic_bank_stage1
        else None
    )

    if (
        pred_residual_enable
        and pred_residual_semantic_output_head_scope == "per_cluster"
        and not pred_residual_semantic_bank_stage1
    ):
        finetune_consumer_cfg = cfg.get("finetune", {}) or {}
        _validate_semantic_frozen_consumer_lifecycle_flags(
            freeze_adapter_bank=pred_residual_freeze_adapter_bank,
            freeze_backbone=bool(
                moe_cfg.get("freeze_backbone", cfg.get("train", {}).get("freeze_backbone", False))
            ),
            patch_router_enable=bool(patch_router_cfg.get("enable", False)),
            finetune_enable=bool(finetune_consumer_cfg.get("enable", False)),
            load_pred_residual=bool(finetune_consumer_cfg.get("load_pred_residual", False)),
            load_gate=bool(finetune_consumer_cfg.get("load_gate", True)),
            require_training_provenance=bool(
                finetune_consumer_cfg.get("require_pred_residual_training_provenance", False)
            ),
        )
        if bool((moe_cfg.get("dynamic_lambda", {}) or {}).get("enable", False)) or bool(
            (moe_cfg.get("learnable_lambda", {}) or {}).get("enable", False)
        ):
            raise ValueError("Per-cluster frozen consumer forbids dynamic/learnable lambda parameters.")
        if not pred_residual_require_pure_backbone_base:
            raise ValueError("Per-cluster frozen consumer requires require_pure_backbone_base=true.")
        if penalty_names != ["level", "trend", "d2_match", "diff_amp"]:
            raise ValueError("Per-cluster frozen consumer requires the exact semantic penalty order.")
        if not (
            shared_moe_across_clusters
            and named_output_projection_enable
            and named_output_projection_fixed_alpha
            and named_output_projection_mode == "semantic_residual"
            and named_output_projection_patch_len == 12
        ):
            raise ValueError(
                "Per-cluster frozen consumer requires the exact shared semantic p12 representation."
            )
        consumer_scales = [
            float(named_output_projection_scale_by_name.get(name, 1.0))
            for name in penalty_names
        ]
        if any(value != 1.0 for value in consumer_scales):
            raise ValueError("Per-cluster frozen consumer requires fixed alpha scale 1.0 for every expert.")
        _validate_semantic_frozen_consumer_paths({
            "intervention_enable": bool(pred_residual_cfg.get("intervention_enable", False)),
            "penalty_selector_enable": bool(pred_residual_cfg.get("penalty_selector_enable", False)),
            "fusion_gate_enable": bool(pred_residual_cfg.get("fusion_gate_enable", False)),
            "channel_expert_adapters": bool(
                (pred_residual_cfg.get("channel_expert_adapters", {}) or {}).get("enable", False)
            ),
            "seasonal_anchor_names": bool(list(pred_residual_cfg.get("seasonal_anchor_names", []) or [])),
            "phase_residual_candidate": bool(phase_residual_candidate_enable),
            "periodic_anchor_expert": bool(periodic_anchor_expert_enable),
            "position_daily_residual_expert": bool(position_daily_residual_expert_enable),
            "anchor_ridge_gate": bool(anchor_ridge_gate_enable),
        })

    pred_residual_contract: Optional[Dict[str, object]] = None
    pred_residual_training_provenance: Optional[Dict[str, object]] = None
    if pred_residual_enable:
        pred_residual_contract = {
            # This contract contains inference representation only.  It is what a
            # frozen-bank consumer must match and deliberately excludes the
            # producer's optimizer/objective settings.
            "version": (
                5
                if bool(pred_residual_semantic_level_controller_cfg.get("enable", False))
                and pred_residual_semantic_level_separate_need_gate
                else 4
                if bool(pred_residual_semantic_level_controller_cfg.get("enable", False))
                else (3 if named_output_projection_mode == "semantic_residual" else 1)
            ),
            "penalty_names": list(penalty_names),
            "projection_mode": str(named_output_projection_mode),
            "projection_patch_len": int(named_output_projection_patch_len),
            "base_type": (
                "pure_backbone"
                if pred_residual_require_pure_backbone_base
                else "configured_candidate_base"
            ),
            "fixed_alpha": bool(named_output_projection_fixed_alpha),
            "scale_by_name": {
                name: float(named_output_projection_scale_by_name.get(name, 1.0))
                for name in penalty_names
            },
            "diff_amp_max": float(named_output_projection_diff_amp_max),
            "semantic_output_head_scope": str(pred_residual_semantic_output_head_scope),
            "semantic_encoder_param_clusters": int(pred_residual.param_K),
            "semantic_output_param_clusters": int(pred_residual.output_param_K),
            "semantic_parameter_ownership": (
                "hybrid_level_local_controller_other_legacy_v1"
                if bool(pred_residual_semantic_level_controller_cfg.get("enable", False))
                and not pred_residual_semantic_level_separate_need_gate
                else "hybrid_level_disjoint_local_gate_other_legacy_v2"
                if bool(pred_residual_semantic_level_controller_cfg.get("enable", False))
                else
                "shared_encoder_per_cluster_output_v1"
                if pred_residual_semantic_output_head_scope == "per_cluster"
                else "legacy_shared_v1"
            ),
        }
        if bool(pred_residual_semantic_level_controller_cfg.get("enable", False)):
            pred_residual_contract.update({
                "semantic_level_controller": {
                    "enable": True,
                    "patch_len": int(pred_residual_semantic_level_controller_cfg.get("patch_len", 12)),
                    "hidden_dim": int(
                        pred_residual_semantic_level_controller_cfg.get(
                            "hidden_dim", pred_residual_cfg.get("corrector_hidden", 32)
                        )
                    ),
                    "state_version": 1,
                },
                "body_type_by_penalty": {
                    name: (
                        "level_local_controller_v1"
                        if name == "level"
                        else (
                            "shared_encoder_per_cluster_output_v1"
                            if pred_residual_semantic_output_head_scope == "per_cluster"
                            else "legacy_shared_v1"
                        )
                    )
                    for name in penalty_names
                },
            })
            if pred_residual_semantic_level_separate_need_gate:
                pred_residual_contract["semantic_level_controller"].update({
                    "separate_need_gate": True,
                    "need_hidden_dim": int(
                        pred_residual_semantic_level_controller_cfg.get(
                            "need_hidden_dim",
                            pred_residual_semantic_level_controller_cfg.get(
                                "hidden_dim", pred_residual_cfg.get("corrector_hidden", 32)
                            ),
                        )
                    ),
                })
        if pred_residual_semantic_output_head_scope == "per_cluster":
            cluster_id_values = [
                int(value) for value in cluster_id_c.detach().cpu().tolist()
            ]
            pred_residual_contract.update({
                "cluster_count": int(K),
                "cluster_id_c": cluster_id_values,
                "cluster_map_sha256": _canonical_cluster_map_sha256(
                    K, cluster_id_values
                ),
            })
        if pred_residual_semantic_bank_stage1:
            pred_residual_training_provenance = {
                "version": (
                    8
                    if bool(pred_residual_semantic_level_controller_cfg.get("enable", False))
                    and pred_residual_semantic_active_penalty == "level"
                    and pred_residual_semantic_level_separate_need_gate
                    and pred_residual_semantic_level_amplitude_optimizer_name == "adam"
                    and pred_residual_semantic_level_acceptance_candidate_source
                    == "raw_amplitude"
                    else 7
                    if bool(pred_residual_semantic_level_controller_cfg.get("enable", False))
                    and pred_residual_semantic_active_penalty == "level"
                    and pred_residual_semantic_level_separate_need_gate
                    and pred_residual_semantic_level_amplitude_optimizer_name == "adam"
                    else 6
                    if bool(pred_residual_semantic_level_controller_cfg.get("enable", False))
                    and pred_residual_semantic_active_penalty == "level"
                    and pred_residual_semantic_level_separate_need_gate
                    else 5
                    if bool(pred_residual_semantic_level_controller_cfg.get("enable", False))
                    and pred_residual_semantic_active_penalty == "level"
                    else (4 if pred_residual_semantic_raw_gradient_accumulation else 3)
                ),
                "stage": "semantic_bank_stage1",
                "candidate_loss": str(pred_residual_candidate_supervision_loss),
                "candidate_supervision_weight": float(
                    pred_residual_candidate_supervision_weight
                ),
                "independent_optimization": bool(
                    pred_residual_candidate_supervision_independent_optimization
                ),
                "independent_optimizer": str(
                    pred_residual_candidate_supervision_independent_optimizer
                ),
                "forecast_mse_weight": float(
                    pred_residual_candidate_supervision_forecast_mse_weight
                ),
                "noop_weight": float(pred_residual_candidate_supervision_noop_weight),
                "active_penalty": str(pred_residual_semantic_active_penalty),
                "validation_blocks": int(pred_residual_semantic_validation_blocks),
                "min_high_need_improved_fraction": float(
                    pred_residual_semantic_min_improved_fraction
                ),
                "min_matching_gain_by_name": dict(
                    pred_residual_semantic_min_gain_by_name
                ),
                "acceptance_mode": (
                    "semantic_only_high_need_patch_with_local_gate"
                    if pred_residual_semantic_level_separate_need_gate
                    and pred_residual_semantic_active_penalty == "level"
                    else "semantic_only_high_need_patch"
                ),
                "need_quantile": float(
                    pred_residual_candidate_supervision_need_quantile
                ),
                "need_patch_len": int(
                    pred_residual_candidate_supervision_need_patch_len
                ),
                "only_allowed": bool(
                    pred_residual_candidate_supervision_only_allowed
                ),
                "include_intervention": bool(
                    pred_residual_candidate_supervision_include_intervention
                ),
                "include_selector": bool(
                    pred_residual_candidate_supervision_include_selector
                ),
                "include_patch_route": bool(
                    pred_residual_candidate_supervision_include_patch_route
                ),
                "penalty_scale_source": str(penalty_scale_source),
                "penalty_scale_floor": float(
                    cfg["train"].get("penalty_scale_floor", 1.0e-3)
                ),
                "penalty_scale_rule": "positive_patch_mean_floored",
                "threshold_source": "train_frozen_pure_backbone_patch_q75",
                "threshold_interpolation": "linear",
                "need_comparison": ">=",
                "optimizer": (
                    {
                        "name": "disjoint_level_adam_amplitude_sgd_need_gate",
                        "amplitude": {
                            "name": "adam",
                            "lr": float(
                                pred_residual_semantic_level_amplitude_lr_raw
                            ),
                            "weight_decay": float(
                                pred_residual_semantic_level_amplitude_weight_decay_raw
                            ),
                            "betas": [0.9, 0.999],
                            "eps": 1.0e-8,
                            "amsgrad": False,
                        },
                        "need_gate": {
                            "name": "sgd",
                            "lr": float(cfg["train"]["lr"]),
                            "weight_decay": float(
                                pred_residual_weight_decay
                                if pred_residual_weight_decay is not None
                                else cfg["train"]["weight_decay"]
                            ),
                            "momentum": 0.0,
                            "dampening": 0.0,
                            "nesterov": False,
                        },
                    }
                    if pred_residual_semantic_level_separate_need_gate
                    and pred_residual_semantic_level_amplitude_optimizer_name == "adam"
                    else {
                        "name": "sgd",
                        "momentum": 0.0,
                        "dampening": 0.0,
                        "nesterov": False,
                    }
                ),
                "scheduler": "none",
                "checkpoint_selection": "semantic_per_expert",
            }
            if (
                bool(pred_residual_semantic_level_controller_cfg.get("enable", False))
                and pred_residual_semantic_active_penalty == "level"
            ):
                if pred_residual_semantic_level_separate_need_gate:
                    pred_residual_training_provenance.update({
                        "level_loss_weights": {
                            "amplitude": 1.0,
                            "need_balanced_bce": 1.0,
                            "executed": 0.0,
                        },
                        "level_need_positive_weight": float(
                            pred_residual_semantic_level_need_positive_weight
                        ),
                        "level_optimizer_groups": ["amplitude", "need_gate"],
                        "gradient_clip": "independent_per_level_group",
                    })
                    if (
                        pred_residual_semantic_level_acceptance_candidate_source
                        == "raw_amplitude"
                    ):
                        pred_residual_training_provenance.update(
                            {
                                "level_amplitude_population": (
                                    "detached_q75_high_need_only"
                                ),
                                "level_acceptance_candidate": "raw_amplitude",
                                "level_need_gate_acceptance_role": (
                                    "diagnostic_only"
                                ),
                            }
                        )
                else:
                    pred_residual_training_provenance["level_loss_weights"] = {
                        "amplitude": 1.0,
                        "need_bce": 1.0,
                        "executed": 1.0,
                    }
            if pred_residual_semantic_raw_gradient_accumulation:
                pred_residual_training_provenance["raw_gradient_accumulation"] = {
                    "enabled": True,
                    "microbatches": int(
                        pred_residual_semantic_raw_gradient_microbatches
                    ),
                    "missing_gradient": "zero",
                    "reduction": "mean_by_actual_count",
                    "clip": (
                        "once_after_mean_per_level_group"
                        if pred_residual_semantic_level_separate_need_gate
                        and pred_residual_semantic_active_penalty == "level"
                        else "once_after_mean"
                    ),
                    "weight_decay": (
                        "once_per_optimizer_step_per_level_group"
                        if pred_residual_semantic_level_separate_need_gate
                        and pred_residual_semantic_active_penalty == "level"
                        else "once_per_optimizer_step"
                    ),
                    "tail": "actual_count",
                }
            _validate_semantic_bank_training_provenance(
                pred_residual_training_provenance
            )

    def _assert_pure_candidate_base(
        pred_out: Optional[Dict[str, torch.Tensor]],
        backbone_prediction_bch: torch.Tensor,
        *,
        stage: str,
    ) -> None:
        if not pred_residual_require_pure_backbone_base or pred_out is None:
            return
        candidate_base = pred_out.get("candidate_base_bch")
        if candidate_base is None or not torch.equal(candidate_base, backbone_prediction_bch):
            max_abs = None
            if candidate_base is not None and candidate_base.shape == backbone_prediction_bch.shape:
                max_abs = float((candidate_base - backbone_prediction_bch).abs().max().detach().item())
            raise RuntimeError(
                "pure-backbone candidate base assertion failed: "
                f"stage={stage}, max_abs={max_abs}."
            )

    gate_balance_target_kp = None
    gate_prior_prob_kp = None
    gate_prior_enable = bool(gate_prior_cfg.get("enable", False)) and penalty_portrait_kp is not None and P > 0
    if gate_prior_enable:
        gate_prior_prob_kp = build_gate_prior_from_penalty_portrait(
            penalty_kp=penalty_portrait_kp,
            penalty_scale=penalty_scale,
            temperature=float(gate_prior_cfg.get("temperature", 1.0)),
            smoothing=float(gate_prior_cfg.get("smoothing", 0.0)),
            use_normalized_penalty=bool(gate_prior_cfg.get("use_normalized_penalty", True)),
        )
        gate.set_penalty_prior(
            gate_prior_prob_kp,
            strength=float(gate_prior_cfg.get("logit_strength", 1.0)),
        )
        if bool(gate_prior_cfg.get("use_as_balance_target", True)):
            gate_balance_target_kp = gate_prior_prob_kp
        print(f"Gate prior enabled: strength={gate.penalty_prior_strength:.3f}, prior={gate_prior_prob_kp.detach().cpu().tolist()}")
    cluster_penalty_prior_enable = (
        bool(cluster_penalty_prior_cfg.get("enable", False))
        and penalty_portrait_kp is not None
        and P > 0
    )
    cluster_penalty_prior_prob_kp = None
    cluster_penalty_allowed_mask_kp = None
    cluster_penalty_prior_configured_mask_kp = None
    cluster_penalty_late_allowed_mask_kp = None
    cluster_penalty_prior_apply_stage = "train_and_eval"
    cluster_penalty_prior_late_applied = False
    if cluster_penalty_prior_enable:
        cluster_penalty_prior_apply_stage = normalize_cluster_penalty_prior_apply_stage(
            cluster_penalty_prior_cfg.get("apply_stage", "train_and_eval")
        )
        cluster_penalty_prior_prob_kp = build_gate_prior_from_penalty_portrait(
            penalty_kp=penalty_portrait_kp,
            penalty_scale=penalty_scale,
            temperature=float(cluster_penalty_prior_cfg.get("temperature", 1.0)),
            smoothing=float(cluster_penalty_prior_cfg.get("smoothing", 0.0)),
            use_normalized_penalty=bool(cluster_penalty_prior_cfg.get("use_normalized_penalty", True)),
        )
        logit_strength = float(cluster_penalty_prior_cfg.get("logit_strength", 0.0))
        if logit_strength > 0.0:
            gate.set_penalty_prior(cluster_penalty_prior_prob_kp, strength=logit_strength)
        topk = int(cluster_penalty_prior_cfg.get("topk", 0))
        manual_allowed = build_named_penalty_mask(
            cluster_penalty_prior_cfg.get("allowed_by_cluster", None),
            penalty_names,
            K,
            device,
            allow_empty_clusters=bool(cluster_penalty_prior_cfg.get("allow_empty_clusters", False)),
        )
        if manual_allowed is not None:
            cluster_penalty_prior_configured_mask_kp = manual_allowed
        elif topk > 0 and bool(cluster_penalty_prior_cfg.get("hard_topk", True)):
            cluster_penalty_prior_configured_mask_kp = build_topk_penalty_mask(
                cluster_penalty_prior_prob_kp,
                topk=topk,
            )
        always_include = cluster_penalty_prior_cfg.get("always_include", []) or []
        if isinstance(always_include, str):
            always_include = [always_include]
        if len(always_include) > 0:
            if cluster_penalty_prior_configured_mask_kp is None:
                cluster_penalty_prior_configured_mask_kp = torch.zeros((K, P), device=device, dtype=torch.float32)
            name_to_idx = {str(name): i for i, name in enumerate(penalty_names)}
            for raw_name in always_include:
                name = str(raw_name)
                if name not in name_to_idx:
                    raise ValueError(
                        "cluster_penalty_prior.always_include contains unknown penalty "
                        f"{name!r}; available={penalty_names}"
                    )
                cluster_penalty_prior_configured_mask_kp[:, name_to_idx[name]] = 1.0
            empty = cluster_penalty_prior_configured_mask_kp.sum(dim=-1, keepdim=True) <= 0.0
            if bool(empty.any().item()):
                cluster_penalty_prior_configured_mask_kp = torch.where(
                    empty,
                    torch.ones_like(cluster_penalty_prior_configured_mask_kp),
                    cluster_penalty_prior_configured_mask_kp,
                )
        (
            cluster_penalty_allowed_mask_kp,
            cluster_penalty_late_allowed_mask_kp,
            cluster_penalty_prior_apply_stage,
        ) = split_cluster_penalty_prior_allowed_mask_by_stage(
            cluster_penalty_prior_configured_mask_kp,
            cluster_penalty_prior_apply_stage,
        )
        gate.set_penalty_allowed_mask(cluster_penalty_allowed_mask_kp)
        if bool(cluster_penalty_prior_cfg.get("use_as_balance_target", False)):
            gate_balance_target_kp = cluster_penalty_prior_prob_kp
        pred_residual_allowed_mask_cp = None
        if (
            pred_residual is not None
            and cluster_penalty_allowed_mask_kp is not None
            and bool(cluster_penalty_prior_cfg.get("apply_to_pred_residual", False))
        ):
            pred_residual_allowed_mask_cp = _cluster_penalty_mask_to_channel_mask(
                cluster_penalty_allowed_mask_kp,
                cluster_id_c,
            )
            pred_residual.set_allowed_penalty_mask(pred_residual_allowed_mask_cp)
        prior_list = (
            cluster_penalty_prior_prob_kp.detach().cpu().tolist()
            if cluster_penalty_prior_prob_kp is not None
            else None
        )
        configured_mask_list = (
            cluster_penalty_prior_configured_mask_kp.detach().cpu().tolist()
            if cluster_penalty_prior_configured_mask_kp is not None
            else None
        )
        active_mask_list = (
            cluster_penalty_allowed_mask_kp.detach().cpu().tolist()
            if cluster_penalty_allowed_mask_kp is not None
            else None
        )
        late_mask_list = (
            cluster_penalty_late_allowed_mask_kp.detach().cpu().tolist()
            if cluster_penalty_late_allowed_mask_kp is not None
            else None
        )
        print(
            "Cluster penalty prior enabled: "
            f"topk={topk}, hard_topk={bool(cluster_penalty_prior_cfg.get('hard_topk', True))}, "
            f"logit_strength={logit_strength:.3f}, apply_stage={cluster_penalty_prior_apply_stage}, "
            f"prior={prior_list}, configured_allowed_mask={configured_mask_list}, "
            f"active_allowed_mask={active_mask_list}, late_allowed_mask={late_mask_list}, "
            f"apply_to_pred_residual={bool(cluster_penalty_prior_cfg.get('apply_to_pred_residual', False))}, "
            f"pred_residual_channel_mask={pred_residual_allowed_mask_cp.detach().cpu().tolist() if pred_residual_allowed_mask_cp is not None else None}"
        )
    if (
        bool(channel_penalty_prior_cfg.get("enable", False))
        and pred_residual is not None
        and channel_penalty_portrait_cp is not None
        and P > 0
    ):
        channel_penalty_prior_prob_cp = build_gate_prior_from_penalty_portrait(
            penalty_kp=channel_penalty_portrait_cp,
            penalty_scale=penalty_scale,
            temperature=float(channel_penalty_prior_cfg.get("temperature", 1.0)),
            smoothing=float(channel_penalty_prior_cfg.get("smoothing", 0.0)),
            use_normalized_penalty=bool(channel_penalty_prior_cfg.get("use_normalized_penalty", True)),
        )
        topk = int(channel_penalty_prior_cfg.get("topk", 0))
        if topk > 0 and bool(channel_penalty_prior_cfg.get("hard_topk", True)):
            channel_penalty_allowed_mask_cp = build_topk_penalty_mask(channel_penalty_prior_prob_cp, topk=topk)
            pred_residual.set_channel_penalty_allowed_mask(channel_penalty_allowed_mask_cp)
        else:
            channel_penalty_allowed_mask_cp = None
        print(
            "Channel penalty prior enabled: "
            f"topk={topk}, hard_topk={bool(channel_penalty_prior_cfg.get('hard_topk', True))}, "
            f"allowed_mask={channel_penalty_allowed_mask_cp.detach().cpu().tolist() if channel_penalty_allowed_mask_cp is not None else None}"
        )

    epochs = int(cfg["train"]["epochs"])

    lambda_init_p = _expand_penalty_setting_for_names(moe_cfg.get("lambda_init", 1.0), penalty_names, 1.0, float)
    lambda_min_p = _expand_penalty_setting_for_names(moe_cfg.get("lambda_min", 0.0), penalty_names, 0.0, float)
    lambda_schedule_p = _expand_penalty_setting_for_names(
        moe_cfg.get("lambda_schedule", "cosine"),
        penalty_names,
        "cosine",
        lambda v: str(v).lower(),
    )
    lambda_min_kp = torch.tensor(lambda_min_p, device=device, dtype=torch.float32).view(1, P).expand(K, P)
    lambda_init_kp = torch.tensor(lambda_init_p, device=device, dtype=torch.float32).view(1, P).expand(K, P)

    learnable_lambda_cfg = moe_cfg.get("learnable_lambda", {})
    learnable_lambda_enable = (
        bool(learnable_lambda_cfg.get("enable", False))
        and moe_enable
        and P > 0
        and (not bool(moe_cfg.get("freeze_lambda", False)))
    )
    learnable_lambda_reg_weight = float(learnable_lambda_cfg.get("reg_weight", 0.0))
    learnable_lambda_share_floor = float(learnable_lambda_cfg.get("share_floor", 0.0))
    bilevel_cfg = learnable_lambda_cfg.get("bilevel", {})
    learnable_lambda = None
    if learnable_lambda_enable:
        learnable_lambda = ClusterwiseLearnableLambda(
            init_lambda_kp=lambda_init_kp,
            lambda_min_kp=lambda_min_kp,
            share_floor=learnable_lambda_share_floor,
        ).to(device)

    dyn_cfg = moe_cfg.get("dynamic_lambda", {})
    dynamic_lambda_enable = bool(dyn_cfg.get("enable", False)) and moe_enable and P > 0
    dynamic_lambda_reg_weight = float(dyn_cfg.get("reg_weight", 0.0))
    dynamic_lambda = None
    if dynamic_lambda_enable:
        dynamic_lambda = ClusterwiseDynamicLambda(
            num_clusters=K,
            feat_dim=gate_feat_dim,
            num_penalties=P,
            hidden_dim=int(dyn_cfg.get("hidden_dim", 32)),
            max_factor=float(dyn_cfg.get("max_factor", 2.0)),
            dropout=float(dyn_cfg.get("dropout", 0.0)),
            mode=str(dyn_cfg.get("mode", "multiscale")),
            mix=float(dyn_cfg.get("mix", 0.6)),
            tau_min=float(dyn_cfg.get("tau_min", 1.0)),
            tau_max=float(dyn_cfg.get("tau_max", 6.0)),
            series_downsample_len=int(dyn_cfg.get("series_downsample_len", 32)),
            segment_bins=dyn_cfg.get("segment_bins", (4, 8)),
        ).to(device)

    lambda_modules_present = (learnable_lambda is not None) or (dynamic_lambda is not None)
    bilevel_requested = bool(bilevel_cfg.get("enable", True)) if lambda_modules_present else False
    # Use a liquid-transformer-style unrolled update:
    # predictor/gate take an inner train-objective step, then lambda is updated by val_mse.
    bilevel_enable = lambda_modules_present and bilevel_requested and (len(dva) > 0)
    bilevel_optimize_gate = bool(bilevel_cfg.get("optimize_gate", False)) and bilevel_enable
    if lambda_modules_present and bilevel_requested and len(dva) == 0:
        raise ValueError("Lambda bilevel update requires a validation split because lambda must be updated from val_mse.")
    bilevel_outer_lr = float(bilevel_cfg.get("outer_lr", cfg["train"]["lr"]))
    bilevel_inner_lr = float(bilevel_cfg.get("inner_lr", cfg["train"]["lr"]))
    bilevel_outer_metric = str(bilevel_cfg.get("val_metric", "mse")).lower()
    if bilevel_enable and bilevel_outer_metric not in {"val_mse", "mse"}:
        print("Lambda outer optimization now uses val_mse only; learnable_lambda.bilevel.val_metric is ignored.")
    bilevel_steps_per_epoch = max(1, int(bilevel_cfg.get("steps_per_epoch", 1)))

    def lambda_value_at(epoch_idx: int, penalty_idx: int) -> float:
        lambda_max = lambda_init_p[penalty_idx]
        lambda_min = lambda_min_p[penalty_idx]
        lambda_schedule = lambda_schedule_p[penalty_idx]
        if lambda_schedule in {"cosine", "cosineannealing"}:
            if epochs <= 1:
                return lambda_max
            t = (epoch_idx - 1) / max(epochs - 1, 1)
            return lambda_min + 0.5 * (lambda_max - lambda_min) * (1.0 + math.cos(math.pi * t))
        return lambda_max

    def scheduled_lambda_kp_at(epoch_idx: int) -> torch.Tensor:
        lam_p = torch.tensor(
            [lambda_value_at(epoch_idx, p) for p in range(P)],
            device=device,
            dtype=torch.float32,
        )
        return lam_p.view(1, P).expand(K, P)

    def lambda_kp_at(epoch_idx: int, detach: bool = True) -> torch.Tensor:
        if learnable_lambda is not None:
            lam = learnable_lambda()
        else:
            lam = scheduled_lambda_kp_at(epoch_idx)
        return lam.detach() if detach else lam

    def lambda_kp_from_epochs(epoch_k: torch.Tensor) -> torch.Tensor:
        if learnable_lambda is not None:
            return learnable_lambda().detach()
        rows = [
            torch.tensor(
                [lambda_value_at(max(1, int(e)), p) for p in range(P)],
                device=device,
                dtype=torch.float32,
            )
            for e in epoch_k.detach().cpu().tolist()
        ]
        if len(rows) == 0:
            return torch.zeros((0, P), device=device)
        return torch.stack(rows, dim=0)

    finetune_summary = None
    semantic_bank_loaded_accepted_rows: List[Dict[str, object]] = []

    def apply_finetune_warm_start():
        nonlocal finetune_summary, semantic_bank_loaded_accepted_rows
        ft_cfg = cfg.get("finetune", {})
        if not bool(ft_cfg.get("enable", False)):
            return

        ckpt_path = str(ft_cfg.get("checkpoint_path", ""))
        if len(ckpt_path) == 0:
            raise ValueError("finetune.enable=true requires finetune.checkpoint_path.")
        ckpt = load_cluster_checkpoint(ckpt_path, device=device)
        meta = ckpt.get("meta", {})
        if len(meta) == 0:
            raise ValueError(f"Fine-tune checkpoint meta is missing: {ckpt_path}")

        src_k_count = int(meta.get("K", 0))
        src_input_len = int(meta.get("input_len", -1))
        src_pred_len = int(meta.get("pred_len", -1))
        partial_model_state = bool(ft_cfg.get("partial_model_state", ft_cfg.get("partial_model", False)))
        if bool(ft_cfg.get("strict_window", True)) and (src_input_len != L or src_pred_len != H):
            raise ValueError(
                "Fine-tune checkpoint window mismatch: "
                f"source input_len/pred_len={src_input_len}/{src_pred_len}, target={L}/{H}. "
                "Train or choose a source checkpoint with the same horizon."
            )
        if src_k_count <= 0:
            raise ValueError(f"Invalid source cluster count in fine-tune checkpoint: {src_k_count}")

        src_model_cfg = dict(meta.get("model_cfg", {}))
        src_compare_model_cfg = dict(src_model_cfg)
        tgt_compare_model_cfg = dict(model_cfg)
        src_compare_model_cfg.pop("history_anchor", None)
        tgt_compare_model_cfg.pop("history_anchor", None)
        if bool(ft_cfg.get("strict_model", True)) and src_compare_model_cfg != tgt_compare_model_cfg:
            raise ValueError("Fine-tune source model_cfg differs from target model_cfg.")
        src_cluster_id_c = meta.get("cluster_id_c", None)
        src_num_channels = meta.get("num_channels", None)
        if bool(dict(src_model_cfg.get("channel_adapter", {}) or {}).get("enable", False)):
            if src_cluster_id_c is None or src_num_channels is None:
                raise ValueError("Fine-tune source checkpoint with channel_adapter requires cluster_id_c and num_channels in meta.")
        source_model = None
        if bool(ft_cfg.get("load_model", True)) and not partial_model_state:
            source_model = build_cluster_predictor(
                num_clusters=src_k_count,
                input_len=src_input_len,
                pred_len=src_pred_len,
                model_cfg=src_model_cfg,
                num_channels=None if src_num_channels is None else int(src_num_channels),
                cluster_id_c=src_cluster_id_c,
            ).to(device)
            source_model.load_state_dict(ckpt["model_state"], strict=True)
            source_model.eval()

        map_mode = str(ft_cfg.get("cluster_map", "index")).lower()
        if map_mode in {"index", "same"}:
            target_to_source_k = torch.arange(K, device=device, dtype=torch.long) % src_k_count
            corr_map = None
        else:
            memory_path = str(ft_cfg.get("memory_path", ""))
            if len(memory_path) == 0:
                raise ValueError("finetune.cluster_map requires finetune.memory_path unless cluster_map=index.")
            source_memory = load_cluster_memory(memory_path, device=device)
            source_proto_kt = source_memory["prototypes_kt"].to(device)
            target_proto_kt = compute_cluster_prototypes(data_tc[:t_train], cluster_id_c)
            corr_map = _rowwise_corr(
                target_proto_kt,
                source_proto_kt,
                align=str(ft_cfg.get("corr_align", "head")),
            )
            target_to_source_k = torch.argmax(corr_map, dim=1).to(torch.long)

        def load_finetune_model_cluster_state(k: int, src_k: int) -> None:
            try:
                model.load_cluster_state(k, source_model.get_cluster_state(src_k))
                return
            except ValueError as exc:
                if "channel_head_mlp cluster" not in str(exc):
                    raise
                required = ("W1", "b1", "W2", "b2", "_cluster_channel_idx")
                if not all(hasattr(model, name) for name in required) or not all(hasattr(source_model, name) for name in required):
                    raise
                device = model.W1[k].device
                model.W1[k].data.copy_(source_model.W1[src_k].to(device))
                model.b1[k].data.copy_(source_model.b1[src_k].to(device))
                target_idx = model._cluster_channel_idx(k)
                for i in target_idx:
                    c = int(i.item())
                    if c >= len(source_model.W2):
                        raise ValueError(f"Fine-tune channel-head transfer missing source channel {c}.") from exc
                    model.W2[c].data.copy_(source_model.W2[c].to(device))
                    model.b2[c].data.copy_(source_model.b2[c].to(device))
                print(
                    "Fine-tune channel_head_mlp warm start used source cluster shared layer "
                    f"{src_k}->target {k} and channel-index output heads."
                )

        partial_model_summary = None
        if bool(ft_cfg.get("load_model", True)):
            if partial_model_state:
                if "model_state" not in ckpt:
                    raise ValueError(f"Fine-tune checkpoint is missing model_state: {ckpt_path}")
                partial_model_summary = _partial_load_matching_state_dict(model, ckpt["model_state"])
                if int(partial_model_summary["loaded_count"]) <= 0 and not bool(ft_cfg.get("allow_empty_partial_model", False)):
                    raise ValueError(
                        "Fine-tune partial_model_state loaded zero tensors. "
                        "Check that source and target predictors share parameter names."
                    )
                print(
                    "Fine-tune partial model warm start: "
                    f"loaded={partial_model_summary['loaded_count']}, "
                    f"skipped_shape={partial_model_summary['skipped_shape_count']}, "
                    f"skipped_missing={partial_model_summary['skipped_missing_count']}"
                )
            else:
                assert source_model is not None
                for k in range(K):
                    src_k = int(target_to_source_k[k].item())
                    load_finetune_model_cluster_state(k, src_k)

        source_penalty_names = list(meta.get("penalty_names", []))
        same_penalties = source_penalty_names == penalty_names
        loaded_pred_residual_state = False
        if bool(ft_cfg.get("load_gate", True)) and "gate_state" in ckpt:
            if not same_penalties:
                raise ValueError(
                    "Fine-tune gate loading requires identical penalty_names: "
                    f"source={source_penalty_names}, target={penalty_names}"
                )
            src_moe_cfg = dict(meta.get("moe_cfg", {}))
            source_gate_state = ckpt["gate_state"]
            source_gate_allow_skip = any(str(name).startswith("W_skip.") for name in source_gate_state.keys())
            source_gate = ClusterwiseMoEGate(
                num_clusters=src_k_count,
                feat_dim=int(meta.get("gate_feat_dim", gate_feat_dim)),
                num_penalties=len(source_penalty_names),
                hidden_dim=int(src_moe_cfg.get("gate_hidden_dim", src_moe_cfg.get("hidden_dim", 64))),
                topk=int(src_moe_cfg.get("topk", 1)),
                allow_skip=source_gate_allow_skip,
                skip_init_bias=float(src_moe_cfg.get("skip_init_bias", -2.0)),
                shared_across_clusters=bool(
                    src_moe_cfg.get("shared_across_clusters", src_moe_cfg.get("share_across_clusters", False))
                ),
            ).to(device)
            source_gate.load_state_dict(source_gate_state, strict=True)
            source_gate.eval()
            if shared_moe_across_clusters:
                gate.load_cluster_state(0, source_gate.get_cluster_state(0))
            else:
                for k in range(K):
                    src_k = int(target_to_source_k[k].item())
                    gate.load_cluster_state(k, source_gate.get_cluster_state(src_k))

        if bool(ft_cfg.get("load_pred_residual", False)) and pred_residual is not None:
            if pred_residual_semantic_output_head_scope == "per_cluster":
                release_meta = meta.get("semantic_bank_independent_selection", {}) or {}
                if pred_residual_semantic_bank_stage1:
                    semantic_bank_loaded_accepted_rows = (
                        _validate_semantic_partial_metadata(
                            release_meta,
                            penalty_names=penalty_names,
                            next_active_penalty=pred_residual_semantic_active_penalty,
                            source_state=ckpt["pred_residual_state"],
                            source_contract=meta.get("pred_residual_contract") or {},
                        )
                    )
                else:
                    _validate_semantic_release_metadata(
                        release_meta,
                        penalty_names=penalty_names,
                        source_state=ckpt["pred_residual_state"],
                        source_contract=meta.get("pred_residual_contract") or {},
                    )
            loaded_pred_residual_state = _load_finetune_pred_residual_state(
                pred_residual=pred_residual,
                checkpoint=ckpt,
                source_penalty_names=source_penalty_names,
                target_penalty_names=penalty_names,
                strict=bool(ft_cfg.get("strict_pred_residual", True)),
                source_contract=meta.get("pred_residual_contract"),
                target_contract=pred_residual_contract,
                source_training_provenance=meta.get(
                    "pred_residual_training_provenance"
                ),
                require_training_provenance=bool(
                    ft_cfg.get(
                        "require_pred_residual_training_provenance",
                        False,
                    )
                ),
            )
            if not loaded_pred_residual_state:
                raise ValueError(f"Fine-tune checkpoint is missing pred_residual_state: {ckpt_path}")

        if bool(ft_cfg.get("load_dynamic_lambda", True)) and dynamic_lambda is not None and "dynamic_lambda_state" in ckpt:
            if not same_penalties:
                raise ValueError(
                    "Fine-tune dynamic_lambda loading requires identical penalty_names: "
                    f"source={source_penalty_names}, target={penalty_names}"
                )
            src_moe_cfg = dict(meta.get("moe_cfg", {}))
            src_dyn_cfg = src_moe_cfg.get("dynamic_lambda", {})
            source_dynamic_lambda = ClusterwiseDynamicLambda(
                num_clusters=src_k_count,
                feat_dim=int(meta.get("gate_feat_dim", gate_feat_dim)),
                num_penalties=len(source_penalty_names),
                hidden_dim=int(src_dyn_cfg.get("hidden_dim", 32)),
                max_factor=float(src_dyn_cfg.get("max_factor", 2.0)),
                dropout=float(src_dyn_cfg.get("dropout", 0.0)),
                mode=str(src_dyn_cfg.get("mode", "multiscale")),
                mix=float(src_dyn_cfg.get("mix", 0.6)),
                tau_min=float(src_dyn_cfg.get("tau_min", 1.0)),
                tau_max=float(src_dyn_cfg.get("tau_max", 6.0)),
                series_downsample_len=int(src_dyn_cfg.get("series_downsample_len", 32)),
                segment_bins=src_dyn_cfg.get("segment_bins", (4, 8)),
            ).to(device)
            source_dynamic_lambda.load_state_dict(ckpt["dynamic_lambda_state"], strict=True)
            source_dynamic_lambda.eval()
            for k in range(K):
                src_k = int(target_to_source_k[k].item())
                dynamic_lambda.load_cluster_state(k, source_dynamic_lambda.get_cluster_state(src_k))

        if bool(ft_cfg.get("load_learnable_lambda", True)) and learnable_lambda is not None and "learnable_lambda_state" in ckpt:
            if not same_penalties:
                raise ValueError(
                    "Fine-tune learnable_lambda loading requires identical penalty_names: "
                    f"source={source_penalty_names}, target={penalty_names}"
                )
            init = torch.ones((src_k_count, len(source_penalty_names)), device=device, dtype=torch.float32)
            mins = torch.zeros_like(init)
            source_learnable_lambda = ClusterwiseLearnableLambda(init, mins).to(device)
            source_learnable_lambda.load_state_dict(ckpt["learnable_lambda_state"], strict=False)
            for k in range(K):
                src_k = int(target_to_source_k[k].item())
                learnable_lambda.load_cluster_state(k, source_learnable_lambda.get_cluster_state(src_k))

        if (
            bool(ft_cfg.get("load_learnable_output_anchor", True))
            and learnable_output_anchor is not None
            and "learnable_output_anchor_state" in ckpt
        ):
            source_refiner_summary = meta.get("learnable_output_anchor_refiner", {})
            source_refiner_rejected = (
                isinstance(source_refiner_summary, dict)
                and source_refiner_summary.get("final_eval_uses_learnable") is False
            )
            if source_refiner_rejected and not bool(
                ft_cfg.get("load_rejected_learnable_output_anchor", False)
            ):
                learnable_output_anchor_summary["loaded_from_checkpoint"] = False
                learnable_output_anchor_summary["skipped_checkpoint_reason"] = (
                    "source_refiner_rejected_by_val_guard"
                )
                print("Fine-tune skipped learnable_output_anchor_state: source refiner was rejected by val guard.")
            else:
                src_moe_cfg = dict(meta.get("moe_cfg", {}))
                src_anchor_cfg = _normalize_learnable_output_anchor_cfg(
                    src_moe_cfg.get("learnable_output_anchor", learnable_output_anchor_cfg)
                )
                source_learnable_output_anchor = ClusterwiseLearnableOutputAnchor(
                    num_clusters=src_k_count,
                    num_channels=int(meta.get("num_channels", C)),
                    pred_len=int(meta.get("pred_len", H)),
                    cfg=src_anchor_cfg,
                ).to(device)
                source_learnable_output_anchor.load_state_dict(
                    ckpt["learnable_output_anchor_state"],
                    strict=bool(ft_cfg.get("strict_learnable_output_anchor", False)),
                )
                source_learnable_output_anchor.eval()
                src_num_channels = int(meta.get("num_channels", C))
                src_pred_len = int(meta.get("pred_len", H))
                if src_num_channels != C or src_pred_len != H:
                    if bool(ft_cfg.get("strict_learnable_output_anchor", False)):
                        raise ValueError(
                            "Fine-tune learnable_output_anchor loading requires identical channel/horizon shapes: "
                            f"source=({src_num_channels}, {src_pred_len}), target=({C}, {H})."
                        )
                    learnable_output_anchor_summary["loaded_from_checkpoint"] = False
                    learnable_output_anchor_summary["skipped_checkpoint_reason"] = "shape_mismatch"
                else:
                    for k in range(K):
                        src_k = int(target_to_source_k[k].item())
                        learnable_output_anchor.load_cluster_state(
                            k,
                            source_learnable_output_anchor.get_cluster_state(src_k),
                        )
                    if periodic_anchor_expert_preserve_loaded_source_mask:
                        preserved_masks = _copy_learnable_output_anchor_active_masks(
                            learnable_output_anchor,
                            source_learnable_output_anchor,
                        )
                        learnable_output_anchor_summary[
                            "periodic_source_mask_preserved"
                        ] = True
                        learnable_output_anchor_summary[
                            "preserved_active_channel_mask"
                        ] = preserved_masks["active_channel_mask"]
                    learnable_output_anchor_summary["loaded_from_checkpoint"] = True
                    learnable_output_anchor_summary["loaded_with_cluster_map"] = [
                        int(v) for v in target_to_source_k.detach().cpu().tolist()
                    ]

        finetune_summary = {
            "checkpoint_path": ckpt_path,
            "memory_path": str(ft_cfg.get("memory_path", "")),
            "cluster_map": map_mode,
            "target_to_source_cluster": [int(v) for v in target_to_source_k.detach().cpu().tolist()],
            "cluster_corr": None if corr_map is None else corr_map.detach().cpu().tolist(),
            "partial_model_state": partial_model_state,
            "partial_model_load": partial_model_summary,
            "loaded_pred_residual": bool(loaded_pred_residual_state),
            "loaded_accepted_penalties": [
                str(row["penalty"])
                for row in semantic_bank_loaded_accepted_rows
            ],
        }
        print(f"Fine-tune warm start loaded from: {ckpt_path}")
        print(f"Fine-tune target->source cluster map: {finetune_summary['target_to_source_cluster']}")

    apply_finetune_warm_start()

    periodic_anchor_source_mask_preserved = bool(
        learnable_output_anchor_summary.get("periodic_source_mask_preserved", False)
    )
    if periodic_anchor_expert_preserve_loaded_source_mask and not periodic_anchor_source_mask_preserved:
        raise ValueError(
            "periodic_anchor_expert.preserve_loaded_source_mask=true requires a compatible "
            "loaded learnable_output_anchor_state."
        )

    periodic_anchor_source_frozen_params = 0
    if (
        periodic_anchor_expert_enable
        and periodic_anchor_expert_freeze_source
        and learnable_output_anchor is not None
    ):
        periodic_anchor_source_frozen_params = _freeze_module_params(learnable_output_anchor)
        print(
            "Periodic anchor expert source frozen: "
            f"learnable_params={periodic_anchor_source_frozen_params}"
        )
    frozen_adapter_bank_params = 0
    semantic_frozen_consumer_trainability: Dict[str, object] = {}
    if pred_residual_freeze_adapter_bank and pred_residual is not None:
        if pred_residual_semantic_output_head_scope == "per_cluster":
            if not bool((finetune_summary or {}).get("loaded_pred_residual", False)):
                raise RuntimeError(
                    "Per-cluster frozen consumer must load the complete semantic bank before freezing."
                )
            frozen_consumer = _freeze_semantic_frozen_consumer(pred_residual)
            semantic_frozen_consumer_trainability = dict(frozen_consumer)
            frozen_adapter_bank_params = int(frozen_consumer["frozen_params"])
            frozen_ids = set()
            for p in range(P):
                for param in pred_residual.get_penalty_body_params(p):
                    if id(param) in frozen_ids:
                        raise RuntimeError("Per-cluster semantic body freeze found duplicate ownership.")
                    frozen_ids.add(id(param))
            for p in range(P):
                if any(param.requires_grad for param in pred_residual.get_penalty_body_params(p)):
                    raise RuntimeError("Per-cluster semantic body was not completely frozen.")
        else:
            frozen_adapter_bank_params = _freeze_module_params(pred_residual)
        print(f"Prediction adapter bank frozen for gate training: params={frozen_adapter_bank_params}")

    freeze_backbone = bool(moe_cfg.get("freeze_backbone", cfg.get("train", {}).get("freeze_backbone", False)))
    frozen_backbone_eval_mode = bool(
        cfg.get("train", {}).get("frozen_backbone_eval_mode", False)
    )
    if frozen_backbone_eval_mode and not freeze_backbone:
        raise ValueError("train.frozen_backbone_eval_mode requires moe.freeze_backbone=true.")
    frozen_backbone_params = 0
    if freeze_backbone:
        frozen_backbone_params = _freeze_module_params(model)
        print(f"Backbone frozen for MoE training: params={frozen_backbone_params}")
        if frozen_backbone_eval_mode:
            print(
                "Frozen backbone stays in eval mode during MoE training "
                "so dropout cannot perturb gate utility targets."
            )

    if penalty_scale_source == "frozen_backbone_patch":
        if not (freeze_backbone and pred_residual_require_pure_backbone_base):
            raise ValueError(
                "frozen_backbone_patch penalty statistics require a frozen pure-backbone base."
            )
        patch_len = int(pred_residual_candidate_supervision_need_patch_len)
        if patch_len <= 0 or H % patch_len != 0:
            raise ValueError(
                "frozen_backbone_patch penalty statistics require need_patch_len to divide pred_len."
            )
        was_training = model.training
        model.eval()
        values_by_penalty: List[List[torch.Tensor]] = [[] for _ in range(P)]
        source_windows = 0
        with torch.no_grad():
            for x_stats, y_stats, _ in dl_tr_source:
                x_stats = x_stats.to(device, non_blocking=True)
                y_stats = y_stats.to(device, non_blocking=True)
                y_base_stats = model(x_stats, cluster_id_c)
                source_windows += int(x_stats.shape[0])
                for p, name in enumerate(penalty_names):
                    patch_values = _patchwise_penalty_bcq(
                        y_base_stats,
                        y_stats,
                        penalty_fns[name],
                        patch_len=patch_len,
                    )
                    values_by_penalty[p].append(
                        patch_values.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
                    )
        if was_training and not frozen_backbone_eval_mode:
            model.train()
        scale_values: List[float] = []
        threshold_values: List[float] = []
        coverage_values: List[float] = []
        distribution_by_name: Dict[str, object] = {}
        source_units = 0
        quantile = float(pred_residual_candidate_supervision_need_quantile)
        for p, name in enumerate(penalty_names):
            if not values_by_penalty[p]:
                raise RuntimeError("No train-only units were available for frozen-backbone penalty statistics.")
            values = torch.cat(values_by_penalty[p], dim=0)
            source_units = int(values.numel())
            positive = values[values > 0.0]
            if int(positive.numel()) > 0:
                scale_value = max(float(positive.mean().item()), float(penalty_scale_floor))
            else:
                scale_value = float(penalty_scale_floor)
            normalized = values / scale_value
            threshold_value = float(
                torch.quantile(
                    normalized,
                    quantile,
                    interpolation="linear",
                ).item()
            )
            coverage_value = float((normalized >= threshold_value).to(torch.float64).mean().item())
            scale_values.append(scale_value)
            threshold_values.append(threshold_value)
            coverage_values.append(coverage_value)
            distribution_by_name[name] = {
                "count": int(values.numel()),
                "positive_count": int(positive.numel()),
                "raw_mean": float(values.mean().item()),
                "raw_positive_mean": (
                    float(positive.mean().item()) if int(positive.numel()) > 0 else None
                ),
                "normalized_quantiles": {
                    str(q): float(
                        torch.quantile(normalized, q, interpolation="linear").item()
                    )
                    for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
                },
            }
        penalty_scale = torch.tensor(scale_values, device=device, dtype=torch.float32)
        penalty_need_threshold = torch.tensor(
            threshold_values,
            device=device,
            dtype=torch.float32,
        )
        penalty_need_statistics_summary.update(
            {
                "source": "train_only_frozen_pure_backbone",
                "source_window_range": [0, int(len(dtr))],
                "source_windows": int(source_windows),
                "source_units": int(source_units),
                "patch_len": int(patch_len),
                "scale_rule": "float64 mean of positive exact patch penalties, floored",
                "threshold_rule": "torch.quantile(normalized_defect, q, interpolation=linear)",
                "scale": {
                    name: float(scale_values[p]) for p, name in enumerate(penalty_names)
                },
                "threshold": {
                    name: float(threshold_values[p]) for p, name in enumerate(penalty_names)
                },
                "coverage": {
                    name: float(coverage_values[p]) for p, name in enumerate(penalty_names)
                },
                "distribution": distribution_by_name,
                "test_read": False,
            }
        )
        stats_path = os.path.join(out_dir, "semantic_bank_train_penalty_statistics.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(penalty_need_statistics_summary, f, ensure_ascii=False, indent=2)
        penalty_need_statistics_summary["artifact_path"] = stats_path
        print(
            "Frozen pure-backbone penalty statistics: "
            f"scales={penalty_need_statistics_summary['scale']}, "
            f"thresholds={penalty_need_statistics_summary['threshold']}"
        )

    semantic_bank_frozen_gate_params = 0
    semantic_bank_frozen_nonbody_params = 0
    semantic_bank_trainable_body_names: List[str] = []
    semantic_bank_frozen_nonbody_names: List[str] = []
    if pred_residual_semantic_bank_stage1:
        if pred_residual is None:
            raise RuntimeError("semantic_bank_stage1 requires a prediction residual module.")
        semantic_bank_frozen_gate_params = _freeze_module_params(gate)
        body_only = _configure_semantic_bank_body_only(
            pred_residual,
            active_penalty_index=semantic_bank_active_penalty_index,
        )
        semantic_bank_trainable_body_names = list(body_only["trainable"])
        semantic_bank_frozen_nonbody_names = list(body_only["frozen"])
        param_by_name = dict(pred_residual.named_parameters())
        semantic_bank_frozen_nonbody_params = int(sum(
            param_by_name[name].numel() for name in semantic_bank_frozen_nonbody_names
        ))
        print(
            "Semantic bank body-only training: "
            f"frozen_gate_params={semantic_bank_frozen_gate_params}, "
            f"frozen_nonbody_params={semantic_bank_frozen_nonbody_params}, "
            f"trainable={semantic_bank_trainable_body_names}, "
            f"frozen={semantic_bank_frozen_nonbody_names}"
        )
    patch_router_replaces_cluster_gate = bool(
        pred_residual is not None and getattr(pred_residual, "patch_router", None) is not None
    )
    frozen_cluster_gate_for_patch_router = 0
    if patch_router_replaces_cluster_gate:
        frozen_cluster_gate_for_patch_router = _freeze_module_params(gate)
        print(
            "Input patch router replaces cluster gate for prediction residual routing: "
            f"frozen_gate_params={frozen_cluster_gate_for_patch_router}"
        )
    semantic_consumer_router_trainable_names: List[str] = []
    if (
        pred_residual is not None
        and pred_residual_semantic_output_head_scope == "per_cluster"
        and not pred_residual_semantic_bank_stage1
    ):
        semantic_consumer_router_trainable_names = _assert_semantic_patch_router_only_trainable(
            model=model,
            gate=gate,
            pred_residual=pred_residual,
            dynamic_lambda=dynamic_lambda,
            learnable_lambda=learnable_lambda,
            learnable_output_anchor=learnable_output_anchor,
        )
    patch_router_pairwise_frozen_other_params = 0
    patch_router_pairwise_frozen_reference: Dict[str, torch.Tensor] = {}
    if patch_router_pairwise_freeze_other_parameters:
        if (
            pred_residual is None
            or pred_residual.patch_router is None
            or not pred_residual.patch_router.expert_risk_pairwise_rank_enable
        ):
            raise ValueError(
                "pairwise_rank.freeze_other_parameters requires an enabled pairwise rank head."
            )
        patch_router_pairwise_frozen_other_params = _freeze_module_params_except_prefixes(
            pred_residual,
            (
                "patch_router.W_pairwise_rank",
                "patch_router.b_pairwise_rank",
            ),
        )
        print(
            "Pairwise-only patch gate training: "
            f"frozen_other_pred_residual_params={patch_router_pairwise_frozen_other_params}"
        )
        patch_router_pairwise_frozen_reference = {
            name: param.detach().clone()
            for name, param in pred_residual.named_parameters()
            if not (
                name.startswith("patch_router.W_pairwise_rank")
                or name.startswith("patch_router.b_pairwise_rank")
            )
        }

    def assert_pairwise_frozen_parameters_unchanged(stage: str) -> None:
        if not patch_router_pairwise_frozen_reference or pred_residual is None:
            return
        current = dict(pred_residual.named_parameters())
        for name, expected in patch_router_pairwise_frozen_reference.items():
            actual = current[name].detach()
            if not torch.equal(actual, expected):
                max_abs = float((actual - expected).abs().max().item())
                raise RuntimeError(
                    "pairwise-only frozen parameter changed: "
                    f"stage={stage}, name={name}, max_abs={max_abs:.6g}"
                )
    learnable_output_anchor_anchor_only = bool(
        learnable_output_anchor is not None and learnable_output_anchor_train_mode == "anchor_only"
    )
    if learnable_output_anchor_anchor_only:
        frozen_for_anchor_only = {
            "gate": int(_freeze_module_params(gate)),
            "pred_residual": 0,
            "dynamic_lambda": 0,
            "learnable_lambda": 0,
        }
        if pred_residual is not None:
            frozen_for_anchor_only["pred_residual"] = int(_freeze_module_params(pred_residual))
        if dynamic_lambda is not None:
            frozen_for_anchor_only["dynamic_lambda"] = int(_freeze_module_params(dynamic_lambda))
        if learnable_lambda is not None:
            frozen_for_anchor_only["learnable_lambda"] = int(_freeze_module_params(learnable_lambda))
        learnable_output_anchor_summary["anchor_only_freeze"] = frozen_for_anchor_only
        print(
            "Learnable output anchor train_mode=anchor_only: "
            "gate/pred-residual/lambda modules are frozen; only anchor parameters are optimized."
        )
    pred_residual_train_with_eval_anchors = (
        pred_residual is not None
        and bool(pred_residual_cfg.get("train_with_eval_anchors", bool(freeze_backbone)))
    )
    output_anchor_train_with_eval = bool(pred_residual_train_with_eval_anchors or learnable_output_anchor is not None)
    learnable_output_anchor_summary["train_with_eval_anchors"] = bool(output_anchor_train_with_eval)
    if output_anchor_train_with_eval:
        print(
            "Training uses the same MoE output-anchor post-processing modules as eval "
            "(train-side anchor scales are selected on train only)."
        )
    raw_moe_weight_decay = moe_cfg.get("weight_decay", None)
    if raw_moe_weight_decay is None:
        raw_moe_weight_decay = moe_cfg.get("optimizer_weight_decay", None)
    if raw_moe_weight_decay is not None:
        moe_weight_decay = float(raw_moe_weight_decay)
    else:
        moe_weight_decay = None
    backbone_lr = None
    if (not freeze_backbone) and (not learnable_output_anchor_anchor_only):
        if moe_cfg.get("backbone_lr", None) is not None:
            backbone_lr = float(moe_cfg["backbone_lr"])
        elif moe_cfg.get("backbone_lr_scale", None) is not None:
            backbone_lr = float(cfg["train"]["lr"]) * float(moe_cfg["backbone_lr_scale"])
    learnable_anchor_weight_decay = None
    learnable_anchor_lr = None
    if learnable_output_anchor is not None:
        if learnable_output_anchor_cfg.get("weight_decay", None) is not None:
            learnable_anchor_weight_decay = float(learnable_output_anchor_cfg["weight_decay"])
        if learnable_output_anchor_cfg.get("optimizer_weight_decay", None) is not None:
            learnable_anchor_weight_decay = float(learnable_output_anchor_cfg["optimizer_weight_decay"])
        if learnable_output_anchor_cfg.get("lr", None) is not None:
            learnable_anchor_lr = float(learnable_output_anchor_cfg["lr"])
        elif learnable_output_anchor_cfg.get("lr_scale", None) is not None:
            learnable_anchor_lr = float(cfg["train"]["lr"]) * float(learnable_output_anchor_cfg["lr_scale"])
        learnable_output_anchor_summary["optimizer"] = {
            "lr": None if learnable_anchor_lr is None else float(learnable_anchor_lr),
            "weight_decay": (
                None if learnable_anchor_weight_decay is None else float(learnable_anchor_weight_decay)
            ),
        }

    semantic_bank_params_by_penalty: List[List[nn.Parameter]] = []
    semantic_bank_param_names_by_penalty: List[List[str]] = []
    semantic_level_disjoint_params_by_group: Dict[str, List[nn.Parameter]] = {}
    semantic_level_disjoint_names_by_group: Dict[str, List[str]] = {}
    if pred_residual_semantic_bank_stage1:
        if pred_residual is None:
            raise RuntimeError("semantic bank independent optimization requires pred_residual.")
        name_by_id = {
            id(param): name for name, param in pred_residual.named_parameters()
        }
        seen_semantic_param_ids = set()
        for p, name in enumerate(penalty_names):
            params_p = [
                param
                for param in pred_residual.get_penalty_body_params(p)
                if param.requires_grad
            ]
            is_active_penalty = p == semantic_bank_active_penalty_index
            if is_active_penalty and not params_p:
                raise RuntimeError(f"semantic expert {name} has no trainable body parameters.")
            if not is_active_penalty and params_p:
                raise RuntimeError(
                    f"inactive semantic expert {name} unexpectedly has trainable parameters."
                )
            ids_p = {id(param) for param in params_p}
            overlap = seen_semantic_param_ids.intersection(ids_p)
            if overlap:
                raise RuntimeError(
                    f"semantic expert {name} shares optimizer parameters with another penalty."
                )
            seen_semantic_param_ids.update(ids_p)
            semantic_bank_params_by_penalty.append(params_p)
            semantic_bank_param_names_by_penalty.append(
                [name_by_id[id(param)] for param in params_p]
            )
        all_trainable_ids = {
            id(param) for param in pred_residual.parameters() if param.requires_grad
        }
        if seen_semantic_param_ids != all_trainable_ids:
            raise RuntimeError(
                "semantic bank optimizer ownership does not exactly cover all trainable bodies."
            )
        if pred_residual_semantic_level_separate_need_gate:
            if pred_residual_semantic_active_penalty != "level":
                raise RuntimeError("disjoint LEVEL gate optimization is level-only.")
            amplitude_params = list(pred_residual.get_level_amplitude_params())
            need_gate_params = list(pred_residual.get_level_need_gate_params())
            amplitude_ids = {id(param) for param in amplitude_params}
            need_gate_ids = {id(param) for param in need_gate_params}
            if not amplitude_params or not need_gate_params or amplitude_ids & need_gate_ids:
                raise RuntimeError("LEVEL amplitude and need-gate parameter groups must be nonempty/disjoint.")
            active_ids = {
                id(param)
                for param in semantic_bank_params_by_penalty[
                    int(semantic_bank_active_penalty_index)
                ]
            }
            if amplitude_ids | need_gate_ids != active_ids:
                raise RuntimeError(
                    "LEVEL disjoint optimizer groups must exactly cover the active body."
                )
            semantic_level_disjoint_params_by_group = {
                "amplitude": amplitude_params,
                "need_gate": need_gate_params,
            }
            semantic_level_disjoint_names_by_group = {
                group: [name_by_id[id(param)] for param in params]
                for group, params in semantic_level_disjoint_params_by_group.items()
            }

    cluster_params = []
    cluster_param_groups = []
    stage2_trainable_param_counts = []
    for k in range(K):
        base_params_k = []
        gate_params_k = []
        pred_residual_params_k = []
        dynamic_lambda_params_k = []
        learnable_lambda_params_k = []
        learnable_anchor_params_k = []
        if (not freeze_backbone) and (not learnable_output_anchor_anchor_only):
            base_params_k.extend(model.get_cluster_params(k))
        if (
            (not learnable_output_anchor_anchor_only)
            and (not patch_router_replaces_cluster_gate)
            and not (bilevel_enable and bilevel_optimize_gate)
        ):
            gate_params_k.extend(
                param for param in _gate_cluster_params(gate, k) if param.requires_grad
            )
        if (
            pred_residual is not None
            and (not learnable_output_anchor_anchor_only)
            and not pred_residual_semantic_bank_stage1
        ):
            pred_residual_params_k.extend(
                param for param in pred_residual.get_cluster_params(k) if param.requires_grad
            )
        pred_residual_count_k = int(sum(param.numel() for param in pred_residual_params_k))
        if pred_residual_semantic_bank_stage1 and k == 0:
            pred_residual_count_k = int(
                sum(
                    param.numel()
                    for params_p in semantic_bank_params_by_penalty
                    for param in params_p
                )
            )
        if dynamic_lambda is not None and (not bilevel_enable) and (not learnable_output_anchor_anchor_only):
            dynamic_lambda_params_k.extend(dynamic_lambda.get_cluster_params(k))
        if learnable_lambda is not None and (not bilevel_enable) and (not learnable_output_anchor_anchor_only):
            learnable_lambda_params_k.append(learnable_lambda.raw[k])
        if learnable_output_anchor is not None:
            learnable_anchor_params_k.extend(learnable_output_anchor.get_cluster_params(k))
        stage2_trainable_param_counts.append(
            {
                "cluster_id": int(k),
                "backbone": int(sum(param.numel() for param in base_params_k)),
                "gate": int(sum(param.numel() for param in gate_params_k)),
                "pred_residual": pred_residual_count_k,
                "dynamic_lambda": int(sum(param.numel() for param in dynamic_lambda_params_k)),
                "learnable_lambda": int(sum(param.numel() for param in learnable_lambda_params_k)),
                "learnable_output_anchor": int(sum(param.numel() for param in learnable_anchor_params_k)),
            }
        )
        param_groups_k = _make_cluster_optimizer_param_groups(
            base_params=base_params_k,
            gate_params=gate_params_k,
            pred_residual_params=pred_residual_params_k,
            dynamic_lambda_params=dynamic_lambda_params_k,
            learnable_lambda_params=learnable_lambda_params_k,
            learnable_anchor_params=learnable_anchor_params_k,
            base_weight_decay=float(cfg["train"]["weight_decay"]),
            moe_weight_decay=moe_weight_decay,
            pred_residual_weight_decay=pred_residual_weight_decay,
            learnable_anchor_weight_decay=learnable_anchor_weight_decay,
            learnable_anchor_lr=learnable_anchor_lr,
            base_lr=backbone_lr,
        )
        params_k = [param for group in param_groups_k for param in group["params"]]
        cluster_params.append(params_k)
        cluster_param_groups.append(param_groups_k)
    totals = {
        "backbone": int(sum(row["backbone"] for row in stage2_trainable_param_counts)),
        "gate": int(sum(row["gate"] for row in stage2_trainable_param_counts)),
        "pred_residual": int(sum(row["pred_residual"] for row in stage2_trainable_param_counts)),
        "dynamic_lambda": int(sum(row["dynamic_lambda"] for row in stage2_trainable_param_counts)),
        "learnable_lambda": int(sum(row["learnable_lambda"] for row in stage2_trainable_param_counts)),
        "learnable_output_anchor": int(sum(row["learnable_output_anchor"] for row in stage2_trainable_param_counts)),
    }
    stage2_trainable_parameter_groups = {
        "total": totals,
        "per_cluster": stage2_trainable_param_counts,
        "shared_moe_across_clusters": bool(shared_moe_across_clusters),
    }

    optimizers: List[Optional[torch.optim.Optimizer]] = [
        (
            torch.optim.Adam(
                param_groups_k,
                lr=float(cfg["train"]["lr"]),
            )
            if len(param_groups_k) > 0
            else None
        )
        for param_groups_k in cluster_param_groups
    ]
    semantic_bank_optimizers_p: List[Optional[torch.optim.Optimizer]] = []
    semantic_bank_optimizer_identity_p: List[str] = []
    semantic_level_need_gate_optimizer: Optional[torch.optim.Optimizer] = None
    if pred_residual_semantic_bank_stage1:
        residual_wd = pred_residual_weight_decay
        if residual_wd is None:
            residual_wd = (
                moe_weight_decay
                if moe_weight_decay is not None
                else float(cfg["train"]["weight_decay"])
            )
        semantic_level_amplitude_lr = float(
            pred_residual_semantic_level_amplitude_lr_raw
            if pred_residual_semantic_level_amplitude_lr_raw is not None
            else cfg["train"]["lr"]
        )
        semantic_level_amplitude_weight_decay = float(
            pred_residual_semantic_level_amplitude_weight_decay_raw
            if pred_residual_semantic_level_amplitude_weight_decay_raw is not None
            else residual_wd
        )
        for p, name in enumerate(penalty_names):
            if p != semantic_bank_active_penalty_index:
                semantic_bank_optimizers_p.append(None)
                semantic_bank_optimizer_identity_p.append("frozen_inactive")
                continue
            optimizer_params = (
                semantic_level_disjoint_params_by_group["amplitude"]
                if pred_residual_semantic_level_separate_need_gate
                else semantic_bank_params_by_penalty[p]
            )
            if pred_residual_semantic_level_separate_need_gate:
                level_optimizers = _make_semantic_level_disjoint_optimizers(
                    optimizer_params,
                    semantic_level_disjoint_params_by_group["need_gate"],
                    amplitude_optimizer_name=(
                        pred_residual_semantic_level_amplitude_optimizer_name
                    ),
                    amplitude_lr=semantic_level_amplitude_lr,
                    amplitude_weight_decay=(
                        semantic_level_amplitude_weight_decay
                    ),
                    need_gate_lr=float(cfg["train"]["lr"]),
                    need_gate_weight_decay=float(residual_wd),
                )
                optimizer_p = level_optimizers["amplitude"]
                semantic_level_need_gate_optimizer = level_optimizers["need_gate"]
            else:
                optimizer_p = torch.optim.SGD(
                    [
                        {
                            "params": optimizer_params,
                            "weight_decay": float(residual_wd),
                        }
                    ],
                    lr=float(cfg["train"]["lr"]),
                    momentum=0.0,
                    dampening=0.0,
                    nesterov=False,
                )
                for group in optimizer_p.param_groups:
                    group.setdefault("initial_lr", float(group["lr"]))
            semantic_bank_optimizers_p.append(optimizer_p)
            identity_fields = [name, *semantic_bank_param_names_by_penalty[p]]
            if pred_residual_semantic_level_separate_need_gate:
                identity_fields.extend([
                    "disjoint_groups=amplitude,need_gate",
                    *(
                        "amplitude:" + param_name
                        for param_name in semantic_level_disjoint_names_by_group["amplitude"]
                    ),
                    *(
                        "need_gate:" + param_name
                        for param_name in semantic_level_disjoint_names_by_group["need_gate"]
                    ),
                    "gradient_clip=independent_per_group",
                ])
                identity_fields.extend(
                    _semantic_level_optimizer_identity_fields(
                        amplitude_optimizer_name=(
                            pred_residual_semantic_level_amplitude_optimizer_name
                        ),
                        amplitude_lr=semantic_level_amplitude_lr,
                        amplitude_weight_decay=(
                            semantic_level_amplitude_weight_decay
                        ),
                        need_gate_lr=float(cfg["train"]["lr"]),
                        need_gate_weight_decay=float(residual_wd),
                    )
                )
            if pred_residual_semantic_raw_gradient_accumulation:
                identity_fields.extend(
                    [
                        "raw_gradient_accumulation=mean_then_clip",
                        "microbatches="
                        + str(pred_residual_semantic_raw_gradient_microbatches),
                        "tail=actual_count",
                    ]
                )
            identity_payload = "|".join(identity_fields).encode("utf-8")
            semantic_bank_optimizer_identity_p.append(
                hashlib.sha256(identity_payload).hexdigest()
            )
        if pred_residual_semantic_raw_gradient_accumulation:
            print(
                "Semantic level optimizer: mode="
                + (
                    "disjoint_amplitude_need_gate_mean_then_clip"
                    if pred_residual_semantic_level_separate_need_gate
                    else "raw_gradient_mean_then_clip"
                )
                + ", "
                f"target_microbatches={pred_residual_semantic_raw_gradient_microbatches}, "
                "missing_gradient=zero, tail=actual_count, "
                f"amplitude_optimizer={pred_residual_semantic_level_amplitude_optimizer_name}, "
                f"amplitude_lr={semantic_level_amplitude_lr:.8g}, "
                f"amplitude_weight_decay={semantic_level_amplitude_weight_decay:.8g}, "
                f"need_gate_optimizer=sgd, need_gate_lr={float(cfg['train']['lr']):.8g}, "
                f"need_gate_weight_decay={float(residual_wd):.8g}, "
                "weight_decay=once_per_step"
            )
    for opt_k in optimizers:
        if opt_k is None:
            continue
        for group in opt_k.param_groups:
            group.setdefault("initial_lr", float(group["lr"]))
    sched_cfg = cfg["train"].get("lr_scheduler", {"name": "none"})
    lr_warmup_epochs = int(sched_cfg.get("warmup_epochs", cfg["train"].get("lr_warmup_epochs", 0)))
    lr_warmup_start_factor = float(
        sched_cfg.get("warmup_start_factor", cfg["train"].get("lr_warmup_start_factor", 0.1))
    )
    sched_name = str(sched_cfg.get("name", "none")).lower()
    if pred_residual_semantic_bank_stage1 and sched_name not in {"none", "off", "disabled"}:
        raise ValueError(
            "semantic_bank_stage1 independent optimization requires a deterministic "
            "loss-independent scheduler (name=none)."
        )
    if sched_name in {"plateau", "reduce", "reduce_on_plateau"}:
        schedulers = [
            (
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt_k,
                    mode="min",
                    factor=float(sched_cfg.get("factor", 0.5)),
                    patience=int(sched_cfg.get("patience", 3)),
                    min_lr=float(sched_cfg.get("min_lr", 1.0e-6)),
                )
                if opt_k is not None
                else None
            )
            for opt_k in optimizers
        ]
    elif sched_name in {"cosine", "cosineannealing"}:
        schedulers = [
            (
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt_k,
                    T_max=int(sched_cfg.get("t_max", 50)),
                    eta_min=float(sched_cfg.get("min_lr", 1.0e-6)),
                )
                if opt_k is not None
                else None
            )
            for opt_k in optimizers
        ]
    elif sched_name in {"step", "steplr"}:
        schedulers = [
            (
                torch.optim.lr_scheduler.StepLR(
                    opt_k,
                    step_size=int(sched_cfg.get("step_size", 10)),
                    gamma=float(sched_cfg.get("gamma", 0.5)),
                )
                if opt_k is not None
                else None
            )
            for opt_k in optimizers
        ]
    else:
        schedulers = None

    lambda_optimizer = None
    if bilevel_enable and (not learnable_output_anchor_anchor_only):
        lambda_params = []
        if bilevel_optimize_gate:
            lambda_params.extend(list(gate.parameters()))
        if dynamic_lambda is not None:
            lambda_params.extend(list(dynamic_lambda.parameters()))
        if learnable_lambda is not None:
            lambda_params.extend(list(learnable_lambda.parameters()))
        if len(lambda_params) > 0:
            lambda_optimizer = torch.optim.Adam(
                lambda_params,
                lr=bilevel_outer_lr,
                weight_decay=0.0,
            )

    swa_cfg = cfg["train"].get("swa", {}) or {}
    if isinstance(swa_cfg, bool):
        swa_cfg = {"enable": bool(swa_cfg)}
    swa_enable = bool(swa_cfg.get("enable", False))
    swa_start_epoch = int(
        swa_cfg.get(
            "start_epoch",
            max(1, int(math.ceil(float(epochs) * float(swa_cfg.get("start_fraction", 0.75))))),
        )
    )
    swa_update_every = max(1, int(swa_cfg.get("update_every", 1)))
    swa_selection_metric = str(swa_cfg.get("selection_metric", "val_mse")).lower()
    if swa_selection_metric not in {"val_loss", "val_mse", "val_mae"}:
        raise ValueError("train.swa.selection_metric must be val_loss, val_mse, or val_mae.")
    swa_min_delta = float(swa_cfg.get("min_delta", 0.0))
    swa_averagers = {}
    swa_updates = 0
    swa_summary = {
        "enable": bool(swa_enable),
        "selected": False,
        "updates": 0,
        "start_epoch": int(swa_start_epoch),
        "update_every": int(swa_update_every),
        "selection_metric": str(swa_selection_metric),
    }
    if swa_enable:
        swa_averagers["model"] = torch.optim.swa_utils.AveragedModel(model)
        swa_averagers["gate"] = torch.optim.swa_utils.AveragedModel(gate)
        if pred_residual is not None:
            swa_averagers["pred_residual"] = torch.optim.swa_utils.AveragedModel(pred_residual)
        if dynamic_lambda is not None:
            swa_averagers["dynamic_lambda"] = torch.optim.swa_utils.AveragedModel(dynamic_lambda)
        if learnable_lambda is not None:
            swa_averagers["learnable_lambda"] = torch.optim.swa_utils.AveragedModel(learnable_lambda)
        if learnable_output_anchor is not None:
            swa_averagers["learnable_output_anchor"] = torch.optim.swa_utils.AveragedModel(learnable_output_anchor)

    def update_swa_averagers(epoch_idx: int) -> None:
        nonlocal swa_updates
        if not swa_enable or not _should_update_swa(epoch_idx, swa_start_epoch, swa_update_every):
            return
        swa_averagers["model"].update_parameters(model)
        swa_averagers["gate"].update_parameters(gate)
        if pred_residual is not None and "pred_residual" in swa_averagers:
            swa_averagers["pred_residual"].update_parameters(pred_residual)
        if dynamic_lambda is not None and "dynamic_lambda" in swa_averagers:
            swa_averagers["dynamic_lambda"].update_parameters(dynamic_lambda)
        if learnable_lambda is not None and "learnable_lambda" in swa_averagers:
            swa_averagers["learnable_lambda"].update_parameters(learnable_lambda)
        if learnable_output_anchor is not None and "learnable_output_anchor" in swa_averagers:
            swa_averagers["learnable_output_anchor"].update_parameters(learnable_output_anchor)
        swa_updates += 1

    def load_swa_averagers() -> None:
        model.load_state_dict(swa_averagers["model"].module.state_dict())
        gate.load_state_dict(swa_averagers["gate"].module.state_dict())
        if pred_residual is not None and "pred_residual" in swa_averagers:
            pred_residual.load_state_dict(swa_averagers["pred_residual"].module.state_dict())
        if dynamic_lambda is not None and "dynamic_lambda" in swa_averagers:
            dynamic_lambda.load_state_dict(swa_averagers["dynamic_lambda"].module.state_dict())
        if learnable_lambda is not None and "learnable_lambda" in swa_averagers:
            learnable_lambda.load_state_dict(swa_averagers["learnable_lambda"].module.state_dict())
        if learnable_output_anchor is not None and "learnable_output_anchor" in swa_averagers:
            learnable_output_anchor.load_state_dict(swa_averagers["learnable_output_anchor"].module.state_dict())

    monitor_metric = selection_metric
    if len(dva) == 0 and monitor_metric.startswith("val_"):
        monitor_metric = "train_" + monitor_metric[4:]
        print(f"Validation split is empty; fallback train.selection_metric -> {monitor_metric}")

    def _select_monitor_k(
        train_loss_k: torch.Tensor,
        train_mse_k: torch.Tensor,
        train_mae_k: torch.Tensor,
        val_loss_k: torch.Tensor,
        val_mse_k: torch.Tensor,
        val_mae_k: torch.Tensor,
    ) -> torch.Tensor:
        if monitor_metric == "val_loss":
            return val_loss_k
        if monitor_metric == "val_mse":
            return val_mse_k
        if monitor_metric == "val_mae":
            return val_mae_k
        if monitor_metric == "train_loss":
            return train_loss_k
        if monitor_metric == "train_mse":
            return train_mse_k
        return train_mae_k

    def _aggregate_val_metric(
        val_loss_k: torch.Tensor,
        val_mse_k: torch.Tensor,
        val_mae_k: torch.Tensor,
        metric: str,
    ) -> float:
        metric = str(metric).lower()
        if metric == "val_loss":
            value_k = val_loss_k
        elif metric == "val_mae":
            value_k = val_mae_k
        elif metric == "val_mse":
            value_k = val_mse_k
        else:
            raise ValueError("SWA selection metric must be val_loss, val_mse, or val_mae.")
        return float(reduce_cluster_metric(value_k, cluster_weight_k).item())

    early_stop_start_epoch = max(1, penalty_warmup_epochs + 1)
    selection_start_epoch = int(cfg["train"].get("model_selection_start_epoch", early_stop_start_epoch))
    selection_start_epoch = max(1, min(selection_start_epoch, epochs))
    if early_stop_start_epoch > 1:
        print(f"Early stop counting starts at epoch {early_stop_start_epoch} after penalty warmup.")
    if selection_start_epoch > 1:
        print(f"Checkpoint selection starts at epoch {selection_start_epoch}.")

    # early stop
    es = cfg["early_stop"]
    patience = int(es["patience"])
    min_delta = float(es["min_delta"])
    best_monitor = torch.full((K,), float("inf"), device=device)
    bad_cnt = torch.zeros((K,), dtype=torch.long, device=device)
    stopped = torch.zeros((K,), dtype=torch.bool, device=device)

    best_state = [
        {
            "model": None,
            "gate": None,
            "pred_residual": None,
            "dynamic_lambda": None,
            "learnable_lambda": None,
            "learnable_output_anchor": None,
        }
        for _ in range(K)
    ]
    best_epoch = torch.ones((K,), dtype=torch.long, device=device)
    shared_moe_best_monitor = float("inf")
    shared_moe_best_epoch = 1
    shared_moe_best_state: Dict[str, Optional[Dict[str, torch.Tensor]]] = {
        "gate": None,
        "pred_residual": None,
    }
    patch_router_epoch0_noop_summary: Optional[Dict[str, object]] = None
    patch_router_epoch0_val_mse_k: Optional[torch.Tensor] = None
    patch_router_epoch0_val_mae_k: Optional[torch.Tensor] = None
    train_mse_hist = []
    val_mse_hist = []
    epoch_times = []

    def save_best(k: int, epoch_idx: int):
        best_state[k]["model"] = model.get_cluster_state(k)
        if not shared_moe_across_clusters:
            best_state[k]["gate"] = gate.get_cluster_state(k)
        if pred_residual is not None and not shared_moe_across_clusters:
            best_state[k]["pred_residual"] = pred_residual.get_cluster_state(k)
        if dynamic_lambda is not None:
            best_state[k]["dynamic_lambda"] = dynamic_lambda.get_cluster_state(k)
        if learnable_lambda is not None:
            best_state[k]["learnable_lambda"] = learnable_lambda.get_cluster_state(k)
        if learnable_output_anchor is not None:
            best_state[k]["learnable_output_anchor"] = learnable_output_anchor.get_cluster_state(k)
        best_epoch[k] = epoch_idx

    def load_best_all():
        for k in range(K):
            if best_state[k]["model"] is not None:
                model.load_cluster_state(k, best_state[k]["model"])
                if not shared_moe_across_clusters:
                    gate.load_cluster_state(k, best_state[k]["gate"])
                if pred_residual is not None and (not shared_moe_across_clusters) and best_state[k]["pred_residual"] is not None:
                    pred_residual.load_cluster_state(k, best_state[k]["pred_residual"])
                if dynamic_lambda is not None and best_state[k]["dynamic_lambda"] is not None:
                    dynamic_lambda.load_cluster_state(k, best_state[k]["dynamic_lambda"])
                if learnable_lambda is not None and best_state[k]["learnable_lambda"] is not None:
                    learnable_lambda.load_cluster_state(k, best_state[k]["learnable_lambda"])
                if learnable_output_anchor is not None and best_state[k]["learnable_output_anchor"] is not None:
                    learnable_output_anchor.load_cluster_state(k, best_state[k]["learnable_output_anchor"])
        if shared_moe_across_clusters and shared_moe_best_state["gate"] is not None:
            gate.load_cluster_state(0, shared_moe_best_state["gate"])
            if pred_residual is not None and shared_moe_best_state["pred_residual"] is not None:
                pred_residual.load_cluster_state(0, shared_moe_best_state["pred_residual"])

    @torch.no_grad()
    def average_lambda_kp(loader: DataLoader, base_lambda_kp: torch.Tensor) -> torch.Tensor:
        if len(loader) == 0:
            return base_lambda_kp
        if dynamic_lambda is None:
            return base_lambda_kp
        sum_lam = torch.zeros((K, P), device=device)
        cnt = 0
        model.eval()
        dynamic_lambda.eval()
        for x, _, _ in loader:
            x = x.to(device, non_blocking=True)
            feat_bcf = extract_gate_features(x)
            feat_bkf = scatter_mean_bcf_to_bkf(feat_bcf, cluster_id_c, K)
            series_bkl = scatter_mean_bcl_to_bkl(x, cluster_id_c, K)
            lam_bkp = _compute_lambda_bkp(
                base_lambda_kp=base_lambda_kp,
                feat_bkf=feat_bkf,
                series_bkl=series_bkl,
                dynamic_lambda=dynamic_lambda,
                lambda_min_kp=lambda_min_kp,
            )
            sum_lam += lam_bkp.sum(dim=0)
            cnt += lam_bkp.shape[0]
        if cnt == 0:
            return base_lambda_kp
        return sum_lam / float(cnt)

    @torch.no_grad()
    def collect_lambda_stats(loader: DataLoader, base_lambda_kp: torch.Tensor) -> Optional[Dict[str, torch.Tensor]]:
        if len(loader) == 0 or dynamic_lambda is None:
            return None
        sum_lam = torch.zeros((K, P), device=device)
        sum_sq_lam = torch.zeros((K, P), device=device)
        min_lam = torch.full((K, P), float("inf"), device=device)
        max_lam = torch.full((K, P), float("-inf"), device=device)
        cnt = 0
        model.eval()
        dynamic_lambda.eval()
        for x, _, _ in loader:
            x = x.to(device, non_blocking=True)
            feat_bcf = extract_gate_features(x)
            feat_bkf = scatter_mean_bcf_to_bkf(feat_bcf, cluster_id_c, K)
            series_bkl = scatter_mean_bcl_to_bkl(x, cluster_id_c, K)
            lam_bkp = _compute_lambda_bkp(
                base_lambda_kp=base_lambda_kp,
                feat_bkf=feat_bkf,
                series_bkl=series_bkl,
                dynamic_lambda=dynamic_lambda,
                lambda_min_kp=lambda_min_kp,
            )
            sum_lam += lam_bkp.sum(dim=0)
            sum_sq_lam += lam_bkp.pow(2).sum(dim=0)
            min_lam = torch.minimum(min_lam, lam_bkp.amin(dim=0))
            max_lam = torch.maximum(max_lam, lam_bkp.amax(dim=0))
            cnt += lam_bkp.shape[0]
        if cnt == 0:
            return None
        mean_lam = sum_lam / float(cnt)
        std_lam = (sum_sq_lam / float(cnt) - mean_lam.pow(2)).clamp_min(0.0).sqrt()
        return {
            "mean": mean_lam,
            "std": std_lam,
            "min": min_lam,
            "max": max_lam,
        }

    @torch.no_grad()
    def print_dynamic_lambda_summary(
        title: str,
        lambda_stats: Optional[Dict[str, torch.Tensor]],
        csv_path: str = None,
    ):
        if lambda_stats is None:
            return
        print(f"\nDynamic lambda summary ({title}):")
        rows = []
        mean_lam = lambda_stats["mean"].detach()
        std_lam = lambda_stats["std"].detach()
        min_lam = lambda_stats["min"].detach()
        max_lam = lambda_stats["max"].detach()
        for k in range(K):
            parts = []
            for p, name in enumerate(penalty_names):
                parts.append(
                    f"{name}(mean={float(mean_lam[k, p].item()):.6f}, "
                    f"std={float(std_lam[k, p].item()):.6f}, "
                    f"min={float(min_lam[k, p].item()):.6f}, "
                    f"max={float(max_lam[k, p].item()):.6f})"
                )
                rows.append({
                    "cluster_id": k,
                    "penalty": name,
                    "lambda_mean": float(mean_lam[k, p].item()),
                    "lambda_std": float(std_lam[k, p].item()),
                    "lambda_min": float(min_lam[k, p].item()),
                    "lambda_max": float(max_lam[k, p].item()),
                })
            print(f"  Cluster {k}: " + ", ".join(parts))
        if csv_path is not None:
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            print(f"Saved dynamic lambda statistics to: {csv_path}")

    @torch.no_grad()
    def print_cluster_penalty_summary(loader: DataLoader, title: str, lam_kp: torch.Tensor, csv_path: str = None):
        if (not moe_enable) or P == 0:
            print("\nPenalty summary: MoE disabled or no penalties.")
            return None
        if len(loader) == 0:
            print("\nPenalty summary: empty loader, skipped.")
            return None
        model.eval()
        gate.eval()

        sum_probs = torch.zeros(K, P, device=device)
        sum_skip_prob = torch.zeros(K, device=device)
        sum_skip_active = torch.zeros(K, device=device)
        cnt_k = torch.zeros(K, device=device)
        for x, y, _ in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yhat = model(x, cluster_id_c)
            pen_bkp = _router_penalty_context_from_history(
                x_bcl=x,
                yhat_base_bch=yhat,
                penalty_names=penalty_names,
                penalty_fns=penalty_fns,
                penalty_scale=penalty_scale,
                cluster_id_c=cluster_id_c,
                K=K,
                router_mode=router_mode,
            )
            feat_bkf = _build_gate_routing_features(x, yhat, cluster_id_c, K, mode=gate_feature_mode)
            _, probs_bkp, skip_bk, skip_prob_bk = gate(
                feat_bkf,
                straight_through=False,
                penalty_context_bkp=pen_bkp,
                penalty_context_mode=router_mode,
                penalty_context_weight=router_penalty_context_weight,
                penalty_context_detach=router_detach_penalty_context,
                penalty_context_score=router_penalty_context_score,
            )
            sum_probs += probs_bkp.sum(dim=0)
            if allow_skip:
                sum_skip_prob += skip_prob_bk.sum(dim=0)
                sum_skip_active += skip_bk.sum(dim=0)
            cnt_k += probs_bkp.shape[0]

        avg_probs = sum_probs / cnt_k.clamp_min(1.0).view(K, 1)
        avg_skip_prob = sum_skip_prob / cnt_k.clamp_min(1.0)
        avg_skip_active = sum_skip_active / cnt_k.clamp_min(1.0)
        lam = lam_kp.detach()  # [K,P]
        print(f"\nPenalty summary ({title}):")
        rows = []
        for k in range(K):
            order = torch.argsort(avg_probs[k], descending=True)
            parts = []
            penalty_rank = 0
            if allow_skip:
                parts.append(
                    f"skip(active={float(avg_skip_active[k].item()):.3f}, p={float(avg_skip_prob[k].item()):.3f}, cost={skip_cost:.3f})"
                )
                rows.append({
                    "cluster_id": k,
                    "penalty": "skip",
                    "avg_prob": float(avg_skip_prob[k].item()),
                    "avg_lambda": 0.0,
                    "rank": 0,
                    "avg_skip_active": float(avg_skip_active[k].item()),
                    "skip_cost": skip_cost,
                })
            for idx in order.tolist():
                p = int(idx)
                penalty_rank += 1
                parts.append(
                    f"{penalty_names[p]}(lambda={float(lam[k, p].item()):.3f}, p={float(avg_probs[k, p].item()):.3f})"
                )
                rows.append({
                    "cluster_id": k,
                    "penalty": penalty_names[p],
                    "avg_prob": float(avg_probs[k, p].item()),
                    "avg_lambda": float(lam[k, p].item()),
                    "rank": penalty_rank,
                    "avg_skip_active": float(avg_skip_active[k].item()) if allow_skip else 0.0,
                    "skip_cost": skip_cost if allow_skip else 0.0,
                })
            print(f"  Cluster {k}: " + ", ".join(parts))
        if csv_path is not None:
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            print(f"Saved cluster penalty probabilities to: {csv_path}")
        return avg_probs.detach()

    @torch.no_grad()
    def _eval_path_base_prediction(
        x_bcl: torch.Tensor,
        query_start_abs_b: torch.Tensor,
    ) -> torch.Tensor:
        x_model = apply_train_stat_input_centering(
            x_bcl,
            query_start_abs_b=query_start_abs_b,
            stat_anchor_pc=model_train_stat_adapter_pc,
            cfg=model_train_stat_adapter_cfg,
        )
        y_base_raw = model(x_model, cluster_id_c)
        y_base = apply_history_anchor_adapter(
            y_base_raw,
            base_pred_bch=y_base_raw,
            observed_history_tc=data_window_tc,
            query_start_abs_b=query_start_abs_b,
            input_len=L,
            cfg=history_anchor_cfg,
        )
        return apply_train_stat_anchor_expert(
            y_base,
            base_pred_bch=y_base,
            x_bcl=x_bcl,
            query_start_abs_b=query_start_abs_b,
            input_len=L,
            stat_anchor_pc=model_train_stat_adapter_pc,
            cfg=model_train_stat_adapter_cfg,
        )

    @torch.no_grad()
    def collect_pred_residual_summary(loader: DataLoader, eval_start: int = 0) -> Dict[str, object]:
        cfg_summary = {
            "enabled": bool(pred_residual is not None),
            "specialization_weight": float(pred_residual_specialization_weight),
            "norm_weight": float(pred_residual_norm_weight),
            "intervention_weight": float(pred_residual_intervention_weight),
            "candidate_supervision_weight": float(pred_residual_candidate_supervision_weight),
            "candidate_supervision_loss": str(pred_residual_candidate_supervision_loss),
            "candidate_supervision_forecast_mse_weight": float(
                pred_residual_candidate_supervision_forecast_mse_weight
            ),
            "candidate_supervision_need_patch_len": int(
                pred_residual_candidate_supervision_need_patch_len
            ),
            "candidate_supervision_need_quantile": float(
                pred_residual_candidate_supervision_need_quantile
            ),
            "candidate_supervision_noop_weight": float(
                pred_residual_candidate_supervision_noop_weight
            ),
            "need_weighted_penalty_statistics": dict(penalty_need_statistics_summary),
            "require_pure_backbone_base": bool(pred_residual_require_pure_backbone_base),
            "semantic_bank_stage1": bool(pred_residual_semantic_bank_stage1),
            "semantic_bank_frozen_gate_params": int(semantic_bank_frozen_gate_params),
            "semantic_bank_frozen_nonbody_params": int(semantic_bank_frozen_nonbody_params),
            "semantic_bank_trainable_body_names": list(semantic_bank_trainable_body_names),
            "semantic_bank_frozen_nonbody_names": list(semantic_bank_frozen_nonbody_names),
            "semantic_output_head_scope": str(pred_residual_semantic_output_head_scope),
            "candidate_supervision_min_abs_improvement": float(pred_residual_candidate_supervision_min_abs),
            "candidate_supervision_min_rel_improvement": float(pred_residual_candidate_supervision_min_rel),
            "candidate_supervision_only_allowed": bool(pred_residual_candidate_supervision_only_allowed),
            "candidate_supervision_include_intervention": bool(pred_residual_candidate_supervision_include_intervention),
            "candidate_supervision_include_selector": bool(pred_residual_candidate_supervision_include_selector),
            "candidate_supervision_include_patch_route": bool(pred_residual_candidate_supervision_include_patch_route),
            "ignore_skip_during_training": bool(pred_residual_ignore_skip_during_training),
            "freeze_adapter_bank": bool(pred_residual_freeze_adapter_bank),
            "use_channel_identity_features": bool(
                pred_residual.use_channel_identity_features
                if pred_residual is not None
                else False
            ),
            "frozen_adapter_bank_params": int(frozen_adapter_bank_params),
            "semantic_frozen_consumer_trainability": dict(
                semantic_frozen_consumer_trainability
            ),
            "semantic_consumer_router_trainable_names": list(
                semantic_consumer_router_trainable_names
            ),
            "named_output_projection_enable": bool(named_output_projection_enable),
            "named_output_projection_fixed_alpha": bool(named_output_projection_fixed_alpha),
            "named_output_projection_scale_by_name": dict(named_output_projection_scale_by_name),
            "named_output_projection_carrier_names": list(named_output_projection_carrier_names),
            "named_output_projection_patch_len": int(named_output_projection_patch_len),
            "named_output_projection_mode": str(named_output_projection_mode),
            "named_output_projection_diff_amp_max": float(
                named_output_projection_diff_amp_max
            ),
            "periodic_anchor_expert": {
                "enable": bool(periodic_anchor_expert_enable),
                "role": "reserved_always_on" if periodic_anchor_expert_enable else None,
                "participation": 1.0 if periodic_anchor_expert_enable else 0.0,
                "gate_excluded": bool(periodic_anchor_expert_enable),
                "scale": float(periodic_anchor_expert_scale),
                "source_frozen": bool(periodic_anchor_expert_freeze_source),
                "frozen_params": int(periodic_anchor_source_frozen_params),
                "preserve_loaded_source_mask": bool(
                    periodic_anchor_expert_preserve_loaded_source_mask
                ),
                "loaded_source_mask_preserved": bool(periodic_anchor_source_mask_preserved),
            },
            "intervention_supervision_weight": float(pred_residual_intervention_supervision_weight),
            "intervention_supervision_min_gain": float(pred_residual_intervention_supervision_min_gain),
            "intervention_supervision_pos_weight": float(pred_residual_intervention_supervision_pos_weight),
            "intervention_supervision_only_allowed": bool(pred_residual_intervention_supervision_only_allowed),
            "route_ce_supervision_weight": float(route_ce_weight),
            "route_ce_supervision_min_abs_improvement": float(route_ce_min_abs_improvement),
            "route_ce_supervision_min_rel_improvement": float(route_ce_min_rel_improvement),
            "route_ce_supervision_min_candidate_delta_rms": float(route_ce_min_candidate_delta_rms),
            "route_ce_supervision_ignore_abs_gain_below": float(route_ce_ignore_abs_gain_below),
            "route_ce_supervision_class_weight": str(route_ce_class_weight_mode),
            "route_ce_supervision_max_class_weight": float(route_ce_max_class_weight),
            "binary_adoption_supervision_weight": float(binary_adoption_weight),
            "binary_adoption_supervision_min_abs_improvement": float(binary_adoption_min_abs_improvement),
            "binary_adoption_supervision_min_rel_improvement": float(binary_adoption_min_rel_improvement),
            "binary_adoption_supervision_min_candidate_delta_rms": float(binary_adoption_min_candidate_delta_rms),
            "binary_adoption_supervision_ignore_abs_gain_below": float(binary_adoption_ignore_abs_gain_below),
            "binary_adoption_supervision_positive_weight": float(binary_adoption_positive_weight),
            "binary_adoption_supervision_negative_weight": float(binary_adoption_negative_weight),
            "route_rate_alignment_supervision_weight": float(route_rate_alignment_weight),
            "route_rate_alignment_supervision_min_abs_improvement": float(route_rate_alignment_min_abs_improvement),
            "route_rate_alignment_supervision_min_rel_improvement": float(route_rate_alignment_min_rel_improvement),
            "route_rate_alignment_supervision_min_candidate_delta_rms": float(route_rate_alignment_min_candidate_delta_rms),
            "route_rate_alignment_supervision_ignore_abs_gain_below": float(route_rate_alignment_ignore_abs_gain_below),
            "route_positive_recall_supervision_weight": float(route_positive_recall_weight),
            "route_positive_recall_supervision_min_abs_improvement": float(route_positive_recall_min_abs_improvement),
            "route_positive_recall_supervision_min_rel_improvement": float(route_positive_recall_min_rel_improvement),
            "route_positive_recall_supervision_min_candidate_delta_rms": float(route_positive_recall_min_candidate_delta_rms),
            "route_positive_recall_supervision_ignore_abs_gain_below": float(route_positive_recall_ignore_abs_gain_below),
            "route_positive_recall_supervision_mode": str(route_positive_recall_mode),
            "route_positive_recall_supervision_target_probability": float(route_positive_recall_target_probability),
            "route_precision_recall_supervision_weight": float(route_precision_recall_weight),
            "route_precision_recall_supervision_min_abs_improvement": float(route_precision_recall_min_abs_improvement),
            "route_precision_recall_supervision_min_rel_improvement": float(route_precision_recall_min_rel_improvement),
            "route_precision_recall_supervision_min_candidate_delta_rms": float(route_precision_recall_min_candidate_delta_rms),
            "route_precision_recall_supervision_ignore_abs_gain_below": float(route_precision_recall_ignore_abs_gain_below),
            "route_precision_recall_supervision_recall_mode": str(route_precision_recall_mode),
            "route_precision_recall_supervision_recall_target_probability": float(route_precision_recall_target_probability),
            "route_precision_recall_supervision_false_adopt_max_probability": float(route_precision_recall_false_adopt_max_probability),
            "route_precision_recall_supervision_false_adopt_weight": float(route_precision_recall_false_adopt_weight),
            "confidence_gate_enable": bool(pred_residual_confidence_gate_enable),
            "confidence_gate_source_split": str(pred_residual_confidence_gate_source_split),
            "confidence_gate_threshold": str(pred_residual_confidence_gate_threshold),
            "confidence_gate_min_abs_improvement": float(pred_residual_confidence_gate_min_abs),
            "confidence_gate_min_rel_improvement": float(pred_residual_confidence_gate_min_rel),
            "confidence_gate_min_precision": float(pred_residual_confidence_gate_min_precision),
            "confidence_gate_max_pred_positive_rate": (
                None
                if pred_residual_confidence_gate_max_pred_rate is None
                else float(pred_residual_confidence_gate_max_pred_rate)
            ),
            "detach_routed_penalty_pred": bool(pred_residual_detach_routed_penalty_pred),
        }
        if pred_residual is None or P == 0 or len(loader) == 0:
            return cfg_summary
        cfg_summary["feature_mode"] = str(getattr(pred_residual, "feature_mode", "legacy"))
        cfg_summary["input_dim"] = int(getattr(pred_residual, "input_dim", 0))
        patch_router = getattr(pred_residual, "patch_router", None)
        cfg_summary["routing_granularity"] = "channel_patch" if patch_router is not None else "cluster"
        cfg_summary["patch_router"] = {
            "enable": bool(patch_router is not None),
            "patch_len": int(patch_router.patch_len) if patch_router is not None else 0,
            "num_patches": int(patch_router.num_patches) if patch_router is not None else 0,
            "feature_source": str(patch_router.feature_source) if patch_router is not None else None,
            "use_full_history_features": bool(
                patch_router is not None and patch_router.use_full_history_features
            ),
            "use_channel_identity_features": bool(
                patch_router is not None and patch_router.use_channel_identity_features
            ),
            "time_phase_periods": (
                [int(v) for v in patch_router.time_phase_periods]
                if patch_router is not None
                else []
            ),
            "lagged_delta_periods": (
                [int(v) for v in patch_router.lagged_delta_periods]
                if patch_router is not None
                else []
            ),
            "history_patch_projection": (
                str(patch_router.history_patch_projection)
                if patch_router is not None
                else None
            ),
            "regime_context_enable": bool(
                patch_router is not None and patch_router.regime_context_enable
            ),
            "regime_context_lengths": (
                [int(v) for v in patch_router.regime_context_lengths]
                if patch_router is not None
                else []
            ),
            "fixed_penalty_index_by_channel": (
                [
                    int(v)
                    for v in patch_router.fixed_penalty_index_by_channel_c
                    .detach()
                    .cpu()
                    .tolist()
                ]
                if (
                    patch_router is not None
                    and int(patch_router.fixed_penalty_index_by_channel_c.numel()) > 0
                )
                else None
            ),
            "candidate_scale_by_channel": (
                [
                    float(v)
                    for v in pred_residual.patch_candidate_scale_c
                    .detach()
                    .cpu()
                    .tolist()
                ]
                if (
                    pred_residual is not None
                    and int(pred_residual.patch_candidate_scale_c.numel()) > 0
                )
                else None
            ),
            "hierarchical_gate_enable": bool(
                patch_router is not None and patch_router.hierarchical_recall_enable
            ),
            "utility_verifier_enable": bool(
                patch_router is not None and patch_router.utility_verifier_enable
            ),
            "utility_verifier_temperature": (
                float(patch_router.utility_verifier_temperature) if patch_router is not None else None
            ),
            "expert_conditional_risk_enable": bool(
                patch_router is not None and patch_router.expert_conditional_risk_enable
            ),
            "application_scale_by_penalty": (
                [
                    float(v)
                    for v in pred_residual.patch_application_scale_p.detach()
                    .cpu()
                    .tolist()
                ]
                if int(pred_residual.patch_application_scale_p.numel()) > 0
                else None
            ),
            "expert_risk_dual_signed_utility_enable": bool(
                patch_router is not None
                and patch_router.expert_risk_dual_signed_utility_enable
            ),
            "expert_risk_analytic_residual_enable": bool(
                patch_router is not None
                and patch_router.expert_risk_analytic_residual_enable
            ),
            "expert_risk_analytic_residual_floor": (
                float(patch_router.expert_risk_analytic_residual_floor)
                if patch_router is not None
                else None
            ),
            "expert_risk_independent_activation_enable": bool(
                patch_router is not None
                and patch_router.expert_risk_independent_activation_enable
            ),
            "expert_risk_decoupled_encoder": bool(
                patch_router is not None and patch_router.expert_risk_decoupled_encoder
            ),
            "expert_risk_candidate_aware": bool(
                patch_router is not None and patch_router.expert_risk_candidate_aware
            ),
            "expert_risk_candidate_compatibility": bool(
                patch_router is not None
                and patch_router.expert_risk_candidate_compatibility
            ),
            "expert_risk_temporal_domain_ensemble": {
                "enable": bool(
                    patch_router is not None
                    and patch_router.expert_risk_temporal_domain_enable
                ),
                "num_domains": (
                    int(patch_router.expert_risk_temporal_domain_count)
                    if patch_router is not None
                    else 0
                ),
                "train_window_count": (
                    int(patch_router.expert_risk_temporal_domain_train_windows)
                    if patch_router is not None
                    else 0
                ),
                "combine": (
                    str(patch_router.expert_risk_temporal_domain_combine)
                    if patch_router is not None
                    else None
                ),
            },
            "expert_risk_proposal_candidate_aware": bool(
                patch_router is not None
                and patch_router.expert_risk_proposal_candidate_aware
            ),
            "expert_risk_proposal_topk": (
                int(patch_router.expert_risk_proposal_topk) if patch_router is not None else None
            ),
            "expert_risk_proposal_rescue_enable": bool(
                patch_router is not None
                and patch_router.expert_risk_proposal_rescue_enable
            ),
            "expert_risk_lower_quantile_enable": bool(
                patch_router is not None
                and patch_router.expert_risk_lower_quantile_enable
            ),
            "expert_risk_lower_quantile": (
                float(patch_router.expert_risk_lower_quantile)
                if patch_router is not None
                else None
            ),
            "expert_risk_adoption_source": (
                str(patch_router.expert_risk_adoption_source)
                if patch_router is not None
                else None
            ),
            "expert_risk_utility_veto_enable": bool(
                patch_router is not None
                and patch_router.expert_risk_utility_veto_enable
            ),
            "expert_risk_utility_veto_detach_features": bool(
                patch_router is not None
                and patch_router.expert_risk_utility_veto_detach_features
            ),
            "expert_risk_adopt_threshold": (
                float(patch_router.expert_risk_adopt_threshold.item())
                if (
                    patch_router is not None
                    and patch_router.expert_risk_adopt_threshold is not None
                )
                else None
            ),
            "expert_risk_adopt_threshold_by_penalty": (
                {
                    penalty_names[p]: float(value)
                    for p, value in enumerate(
                        patch_router.expert_risk_adopt_threshold_by_penalty
                        .detach()
                        .cpu()
                        .tolist()
                    )
                }
                if (
                    patch_router is not None
                    and patch_router.expert_risk_adopt_threshold_by_penalty is not None
                )
                else None
            ),
            "expert_risk_pairwise_rank_enable": bool(
                patch_router is not None
                and patch_router.expert_risk_pairwise_rank_enable
            ),
            "expert_risk_pairwise_detach_features": bool(
                patch_router is not None
                and patch_router.expert_risk_pairwise_detach_features
            ),
            "pairwise_frozen_other_params": int(
                patch_router_pairwise_frozen_other_params
            ),
            "expected_mse_weight": float(patch_router_expected_mse_weight),
            "expected_mae_weight": float(patch_router_expected_mae_weight),
            "mixture_mse_weight": float(patch_router_mixture_mse_weight),
            "mixture_mae_weight": float(patch_router_mixture_mae_weight),
            "temporal_group_dro": {
                "enable": bool(patch_router_temporal_group_dro_enable),
                "weight": float(patch_router_temporal_group_dro_weight),
                "num_domains": int(patch_router_temporal_group_dro_domains),
                "temperature": float(
                    patch_router_temporal_group_dro_temperature
                ),
                "target": "expected_patch_mse_minus_frozen_backbone_patch_mse",
            },
            "oracle_ce_weight": float(patch_router_oracle_ce_weight),
            "oracle_ce_warmup_epochs": int(patch_router_oracle_ce_warmup_epochs),
            "freeze_experts_after_warmup": bool(patch_router_freeze_experts_after_warmup),
            "supervision_only": bool(patch_router_supervision_only),
            "epoch0_noop_selection": (
                dict(patch_router_epoch0_noop_summary)
                if patch_router_epoch0_noop_summary is not None
                else {
                    "enable": bool(patch_router_epoch0_noop_enable),
                    "require_dual_improvement": bool(
                        patch_router_epoch0_require_dual
                    ),
                }
            ),
            "train_oracle_diagnostic_enable": bool(patch_router_train_oracle_diagnostic),
            "score_threshold_curve_enable": bool(
                patch_router_score_threshold_curve
            ),
            "score_threshold_curve_max_windows": int(
                patch_router_score_threshold_max_windows
            ),
            "score_threshold_curve_heads": (
                sorted(patch_router_score_threshold_heads)
                if patch_router_score_threshold_heads is not None
                else None
            ),
            "train_temporal_blocks": int(patch_router_train_temporal_blocks),
            "validation_temporal_blocks": int(
                patch_router_validation_temporal_blocks
            ),
            "walk_forward_reliability": {
                "enable": bool(patch_router_walk_forward_enable),
                "test_enable": bool(patch_router_walk_forward_test_enable),
                "condition_on_selected_penalty": bool(
                    patch_router_walk_forward_condition_on_selected_penalty
                ),
                "rerank_all_candidates": bool(
                    patch_router_walk_forward_rerank_all_candidates
                ),
                "feedback_ridge": bool(
                    patch_router_walk_forward_feedback_ridge
                ),
                "feedback_ridge_strength": float(
                    patch_router_walk_forward_feedback_ridge_strength
                ),
                "feedback_target_clip": float(
                    patch_router_walk_forward_feedback_target_clip
                ),
                "label_delay": int(patch_router_walk_forward_label_delay),
                "label_delay_mode": str(
                    patch_router_walk_forward_label_delay_mode
                ),
                "lookback_windows": int(patch_router_walk_forward_lookback),
                "min_history_windows": int(
                    patch_router_walk_forward_min_history
                ),
                "history_stride": int(
                    patch_router_walk_forward_history_stride
                ),
                "min_mean_gain": float(
                    patch_router_walk_forward_min_mean_gain
                ),
                "max_abs_regime_z": patch_router_walk_forward_max_abs_regime_z,
                "scale_mode": str(patch_router_walk_forward_scale_mode),
                "max_scale": float(patch_router_walk_forward_max_scale),
                "scale_consensus_blocks": int(
                    patch_router_walk_forward_scale_consensus_blocks
                ),
                "feature_ridge": float(patch_router_walk_forward_feature_ridge),
                "feature_update_blocks": int(
                    patch_router_walk_forward_feature_update_blocks
                ),
                "temporal_blocks": int(
                    patch_router_walk_forward_temporal_blocks
                ),
                "train_audit_fraction": float(
                    patch_router_walk_forward_train_audit_fraction
                ),
            },
            "temporal_calibration": {
                "enable": bool(patch_router_temporal_calibration_enable),
                "supervision_end_idx": int(patch_router_supervision_end_idx),
                "calibration_start_idx": int(patch_router_calibration_start_idx),
                "calibration_end_idx": int(len(dtr)),
                "purge_windows": int(patch_router_calibration_purge_windows),
                "temporal_blocks": int(patch_router_calibration_blocks),
                "min_gain_cost_ratio": float(
                    patch_router_calibration_min_gain_cost_ratio
                ),
                "min_block_net_gain": float(
                    patch_router_calibration_min_block_net_gain
                ),
                "per_penalty": bool(patch_router_calibration_per_penalty),
                "selection": patch_router_temporal_calibration_summary,
            },
            "frozen_expert_params": int(patch_router_frozen_expert_params),
            "oracle_min_abs_improvement": float(patch_router_oracle_min_abs_improvement),
            "hierarchical_recall": {
                "enable": bool(patch_router_hierarchical_enable),
                "mask_inactive_fixed_channels": bool(
                    patch_router_mask_inactive_fixed_channels
                ),
                "supervision_weight": float(patch_router_hierarchical_weight),
                "warmup_epochs": int(patch_router_hierarchical_warmup_epochs),
                "min_abs_improvement": float(patch_router_hierarchical_min_abs_improvement),
                **{key: float(value) for key, value in patch_router_hierarchical_loss_cfg.items()},
            },
        }

        model.eval()
        gate.eval()
        pred_residual.eval()
        alpha_kp = pred_residual.alpha_values().detach()
        branch_sq_sum = 0.0
        branch_numel = 0
        delta_sq_sum = 0.0
        delta_numel = 0
        base_sq_sum = 0.0
        spec_sum_k = torch.zeros(K, device=device)
        norm_sum_k = torch.zeros(K, device=device)
        intervention_sum_k = torch.zeros(K, device=device)
        selected_intervention_sum_p = torch.zeros(P, device=device)
        route_sum_p = torch.zeros(P, device=device)
        effective_route_sum_p = torch.zeros(P, device=device)
        route_numel = 0
        patch_route_sum_p = torch.zeros(P, device=device)
        patch_route_numel = 0
        patch_skip_sum = 0.0
        patch_skip_numel = 0
        patch_oracle_count = torch.tensor(0.0, device=device)
        patch_oracle_base_error_sum = torch.tensor(0.0, device=device)
        patch_oracle_error_sum = torch.tensor(0.0, device=device)
        patch_selected_error_sum = torch.tensor(0.0, device=device)
        patch_oracle_base_mae_sum = torch.tensor(0.0, device=device)
        patch_oracle_mae_sum = torch.tensor(0.0, device=device)
        patch_selected_mae_sum = torch.tensor(0.0, device=device)
        patch_correct_count = torch.tensor(0.0, device=device)
        patch_oracle_penalty_count = torch.tensor(0.0, device=device)
        patch_selected_penalty_count = torch.tensor(0.0, device=device)
        patch_adoption_true_positive_count = torch.tensor(0.0, device=device)
        patch_selected_beneficial_count = torch.tensor(0.0, device=device)
        patch_selected_harmful_count = torch.tensor(0.0, device=device)
        patch_dual_oracle_penalty_count = torch.tensor(0.0, device=device)
        patch_selected_dual_beneficial_count = torch.tensor(0.0, device=device)
        patch_selected_dual_harmful_count = torch.tensor(0.0, device=device)
        patch_selected_positive_gain_sum = torch.tensor(0.0, device=device)
        patch_selected_negative_cost_sum = torch.tensor(0.0, device=device)
        patch_risk_sign_positive_count = torch.tensor(0.0, device=device)
        patch_risk_sign_predicted_positive_count = torch.tensor(0.0, device=device)
        patch_risk_sign_true_positive_count = torch.tensor(0.0, device=device)
        patch_risk_sign_correct_count = torch.tensor(0.0, device=device)
        patch_risk_sign_count = torch.tensor(0.0, device=device)
        patch_selected_beneficial_count_by_penalty = torch.zeros(P, device=device)
        patch_selected_count_by_penalty = torch.zeros(P, device=device)
        patch_selected_gain_sum_by_penalty = torch.zeros(P, device=device)
        patch_oracle_penalty_hit_count = torch.tensor(0.0, device=device)
        patch_beneficial_penalty_count = torch.zeros(P, device=device)
        patch_proposed_penalty_count = torch.zeros(P, device=device)
        patch_proposal_true_positive_count = torch.zeros(P, device=device)
        patch_proposal_oracle_hit_count = torch.tensor(0.0, device=device)
        patch_shortlist_pairwise_count = torch.tensor(0.0, device=device)
        patch_shortlist_pairwise_correct_count = torch.tensor(0.0, device=device)
        patch_proposal_oracle_hit_count_by_penalty = torch.zeros(P, device=device)
        patch_beneficial_cardinality_sum = torch.tensor(0.0, device=device)
        patch_beneficial_cardinality_histogram = torch.zeros(P + 1, device=device)
        patch_oracle_class_count = torch.zeros(P + 1, device=device)
        patch_selected_class_count = torch.zeros(P + 1, device=device)
        patch_confusion_matrix = torch.zeros(P + 1, P + 1, device=device)
        patch_base_error_sum_by_patch = (
            torch.zeros(patch_router.num_patches, device=device) if patch_router is not None else None
        )
        patch_oracle_error_sum_by_patch = (
            torch.zeros(patch_router.num_patches, device=device) if patch_router is not None else None
        )
        patch_selected_error_sum_by_patch = (
            torch.zeros(patch_router.num_patches, device=device) if patch_router is not None else None
        )
        patch_count_by_patch = (
            torch.zeros(patch_router.num_patches, device=device) if patch_router is not None else None
        )
        cnt = 0

        for x, y, idx in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            idx = idx.to(device=device, dtype=torch.long, non_blocking=True)
            query_start_abs_b = int(eval_start) + idx
            y_base = _eval_path_base_prediction(x, query_start_abs_b)
            fixed_expert_delta_bch = build_moe_output_anchor_fixed_expert_delta(
                y_base,
                x_bcl=x,
                query_start_abs_b=query_start_abs_b,
                input_len=L,
                moe_cfg=moe_cfg,
                moe_enable=moe_enable,
                observed_history_tc=data_window_tc,
                train_stat_anchor_pc=train_stat_anchor_pc,
                train_residual_anchor_phc=train_residual_anchor_phc,
                learnable_output_anchor=learnable_output_anchor,
                cluster_id_c=cluster_id_c,
            )
            routing_base_bch = (
                y_base
                if fixed_expert_delta_bch is None
                else y_base + float(periodic_anchor_expert_scale) * fixed_expert_delta_bch
            )
            route_pen_bkp = _router_penalty_context_from_history(
                x_bcl=x,
                yhat_base_bch=routing_base_bch,
                penalty_names=penalty_names,
                penalty_fns=penalty_fns,
                penalty_scale=penalty_scale,
                cluster_id_c=cluster_id_c,
                K=K,
                router_mode=router_mode,
            )
            feat_bkf = _build_gate_routing_features(
                x, routing_base_bch, cluster_id_c, K, mode=gate_feature_mode
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
            if select_ranks is not None:
                mask_bkp = _select_rank_mask(probs_bkp, select_ranks, straight_through=False)
            pred_out = pred_residual(
                x,
                y_base,
                cluster_id_c,
                mask_bkp,
                skip_bk=skip_bk if allow_skip else None,
                query_start_abs_b=query_start_abs_b,
                fixed_expert_delta_bch=fixed_expert_delta_bch,
            )
            _assert_pure_candidate_base(pred_out, y_base, stage="residual_summary")
            y_final = pred_out["y_final"]
            terms = _pred_residual_loss_terms(
                pred_out=pred_out,
                y_base=y_base,
                y_final=y_final,
                y=y,
                penalty_names=penalty_names,
                penalty_fns=penalty_fns,
                cluster_id_c=cluster_id_c,
                K=K,
                penalty_scale=penalty_scale,
                specialization_weight=1.0,
                norm_weight=1.0,
                intervention_weight=1.0,
            )
            spec_sum_k += terms["specialization_bk"].sum(dim=0)
            norm_sum_k += terms["norm_bk"].sum(dim=0)
            intervention_sum_k += terms["intervention_bk"].sum(dim=0)
            cnt += int(x.shape[0])
            branches = pred_out["branches"]
            branch_sq_sum += float(branches.pow(2).sum().item())
            branch_numel += int(branches.numel())
            delta = y_final - y_base
            delta_sq_sum += float(delta.pow(2).sum().item())
            delta_numel += int(delta.numel())
            base_sq_sum += float(y_base.pow(2).sum().item())
            route_bcp = pred_out["route_bcp"].detach()
            intervention_bcp = pred_out.get("intervention_bcp", torch.ones_like(route_bcp)).detach()
            effective_route_bcp = pred_out.get("effective_route_bcp", route_bcp * intervention_bcp).detach()
            route_sum_p += route_bcp.sum(dim=(0, 1))
            selected_intervention_sum_p += (route_bcp * intervention_bcp).sum(dim=(0, 1))
            effective_route_sum_p += effective_route_bcp.sum(dim=(0, 1))
            route_numel += int(route_bcp.shape[0] * route_bcp.shape[1])
            patch_route_bcph = pred_out.get("patch_route_bcph")
            if patch_route_bcph is not None:
                patch_route_bcph = patch_route_bcph.detach()
                patch_route_sum_p += patch_route_bcph.sum(dim=(0, 1, 3))
                patch_route_numel += int(
                    patch_route_bcph.shape[0] * patch_route_bcph.shape[1] * patch_route_bcph.shape[3]
                )
                patch_skip_bcq = pred_out.get("patch_skip_bcq")
                if patch_skip_bcq is not None:
                    patch_skip_sum += float(patch_skip_bcq.detach().sum().item())
                    patch_skip_numel += int(patch_skip_bcq.numel())
                    oracle_base_bch, candidate_bcpH = _pred_residual_candidates_on_eval_path(
                        y_base,
                        pred_out,
                        apply_output_anchors=output_anchor_train_with_eval,
                        x_bcl=x,
                        query_start_abs_b=query_start_abs_b,
                        input_len=L,
                        moe_cfg=moe_cfg,
                        moe_enable=moe_enable,
                        observed_history_tc=data_window_tc,
                        train_stat_anchor_pc=train_stat_anchor_pc,
                        train_residual_anchor_phc=train_residual_anchor_phc,
                        learnable_output_anchor=learnable_output_anchor,
                        cluster_id_c=cluster_id_c,
                        include_patch_route=False,
                    )
                    if candidate_bcpH is not None:
                        oracle_stats = _patch_router_oracle_batch_stats(
                            base_bch=oracle_base_bch,
                            candidate_bcpH=candidate_bcpH,
                            y_bch=y,
                            patch_route_bcph=patch_route_bcph,
                            patch_skip_bcq=patch_skip_bcq,
                            patch_penalty_benefit_probs_bcqp=pred_out.get(
                                "patch_penalty_benefit_probs_bcqp"
                            ),
                            patch_penalty_risk_benefit_probs_bcqp=pred_out.get(
                                "patch_penalty_risk_benefit_probs_bcqp"
                            ),
                            patch_penalty_proposal_mask_bcqp=pred_out.get(
                                "patch_penalty_proposal_mask_bcqp"
                            ),
                            patch_selected_penalty_index_bcq=pred_out.get(
                                "patch_selected_penalty_index_bcq"
                            ),
                        )
                        patch_oracle_count += oracle_stats["count"]
                        patch_oracle_base_error_sum += oracle_stats["base_error_sum"]
                        patch_oracle_error_sum += oracle_stats["oracle_error_sum"]
                        patch_selected_error_sum += oracle_stats["selected_error_sum"]
                        patch_oracle_base_mae_sum += oracle_stats["base_mae_sum"]
                        patch_oracle_mae_sum += oracle_stats["oracle_mae_sum"]
                        patch_selected_mae_sum += oracle_stats["selected_mae_sum"]
                        patch_correct_count += oracle_stats["correct_count"]
                        patch_oracle_penalty_count += oracle_stats["oracle_penalty_count"]
                        patch_selected_penalty_count += oracle_stats["selected_penalty_count"]
                        patch_adoption_true_positive_count += oracle_stats[
                            "adoption_true_positive_count"
                        ]
                        patch_selected_beneficial_count += oracle_stats[
                            "selected_beneficial_count"
                        ]
                        patch_selected_harmful_count += oracle_stats["selected_harmful_count"]
                        patch_dual_oracle_penalty_count += oracle_stats[
                            "dual_oracle_penalty_count"
                        ]
                        patch_selected_dual_beneficial_count += oracle_stats[
                            "selected_dual_beneficial_count"
                        ]
                        patch_selected_dual_harmful_count += oracle_stats[
                            "selected_dual_harmful_count"
                        ]
                        patch_selected_positive_gain_sum += oracle_stats[
                            "selected_positive_gain_sum"
                        ]
                        patch_selected_negative_cost_sum += oracle_stats[
                            "selected_negative_cost_sum"
                        ]
                        patch_risk_sign_positive_count += oracle_stats[
                            "risk_sign_positive_count"
                        ]
                        patch_risk_sign_predicted_positive_count += oracle_stats[
                            "risk_sign_predicted_positive_count"
                        ]
                        patch_risk_sign_true_positive_count += oracle_stats[
                            "risk_sign_true_positive_count"
                        ]
                        patch_risk_sign_correct_count += oracle_stats[
                            "risk_sign_correct_count"
                        ]
                        patch_risk_sign_count += oracle_stats["risk_sign_count"]
                        patch_selected_beneficial_count_by_penalty += oracle_stats[
                            "selected_beneficial_count_by_penalty"
                        ]
                        patch_selected_count_by_penalty += oracle_stats[
                            "selected_count_by_penalty"
                        ]
                        patch_selected_gain_sum_by_penalty += oracle_stats[
                            "selected_gain_sum_by_penalty"
                        ]
                        patch_oracle_penalty_hit_count += oracle_stats["oracle_penalty_hit_count"]
                        patch_beneficial_penalty_count += oracle_stats["beneficial_penalty_count"]
                        patch_proposed_penalty_count += oracle_stats["proposed_penalty_count"]
                        patch_proposal_true_positive_count += oracle_stats[
                            "proposal_true_positive_count"
                        ]
                        patch_proposal_oracle_hit_count += oracle_stats[
                            "proposal_oracle_hit_count"
                        ]
                        patch_shortlist_pairwise_count += oracle_stats[
                            "shortlist_pairwise_count"
                        ]
                        patch_shortlist_pairwise_correct_count += oracle_stats[
                            "shortlist_pairwise_correct_count"
                        ]
                        patch_proposal_oracle_hit_count_by_penalty += oracle_stats[
                            "proposal_oracle_hit_count_by_penalty"
                        ]
                        patch_beneficial_cardinality_sum += oracle_stats[
                            "beneficial_cardinality_sum"
                        ]
                        patch_beneficial_cardinality_histogram += oracle_stats[
                            "beneficial_cardinality_histogram"
                        ]
                        patch_oracle_class_count += oracle_stats["oracle_class_count"]
                        patch_selected_class_count += oracle_stats["selected_class_count"]
                        patch_confusion_matrix += oracle_stats["confusion_matrix"]
                        patch_base_error_sum_by_patch += oracle_stats["base_error_sum_by_patch"]
                        patch_oracle_error_sum_by_patch += oracle_stats["oracle_error_sum_by_patch"]
                        patch_selected_error_sum_by_patch += oracle_stats["selected_error_sum_by_patch"]
                        patch_count_by_patch += oracle_stats["count_by_patch"]

        spec_k = spec_sum_k / max(cnt, 1)
        norm_k = norm_sum_k / max(cnt, 1)
        intervention_k = intervention_sum_k / max(cnt, 1)
        route_denom_p = route_sum_p.clamp_min(1.0e-8)
        selected_intervention_p = selected_intervention_sum_p / route_denom_p
        effective_route_p = effective_route_sum_p / max(route_numel, 1)
        cfg_summary.update(
            {
                "alpha_mean": float(alpha_kp.mean().item()),
                "alpha_by_penalty": {
                    penalty_names[p]: float(alpha_kp[:, p].mean().item()) for p in range(P)
                },
                "intervention_mean_selected": float(
                    (selected_intervention_sum_p.sum() / route_sum_p.sum().clamp_min(1.0e-8)).item()
                ),
                "intervention_by_penalty": {
                    penalty_names[p]: float(selected_intervention_p[p].item()) for p in range(P)
                },
                "effective_route_by_penalty": {
                    penalty_names[p]: float(effective_route_p[p].item()) for p in range(P)
                },
                "branch_rms": float((branch_sq_sum / max(branch_numel, 1)) ** 0.5),
                "residual_base_rms_ratio": float((delta_sq_sum / max(base_sq_sum, 1.0e-12)) ** 0.5),
                "specialization_loss": float(reduce_cluster_metric(spec_k, cluster_weight_k).item()),
                "norm_loss": float(reduce_cluster_metric(norm_k, cluster_weight_k).item()),
                "intervention_loss": float(reduce_cluster_metric(intervention_k, cluster_weight_k).item()),
            }
        )
        if stage2_loss_audit_enable:
            cfg_summary["residual_delta_rms"] = float((delta_sq_sum / max(delta_numel, 1)) ** 0.5)
        if patch_route_numel > 0:
            patch_rate_p = patch_route_sum_p / float(patch_route_numel)
            cfg_summary["patch_router"].update(
                {
                    "selection_rate_by_penalty": {
                        penalty_names[p]: float(patch_rate_p[p].item()) for p in range(P)
                    },
                    "skip_rate": float(patch_skip_sum / max(patch_skip_numel, 1)),
                }
            )
        if float(patch_oracle_count.item()) > 0.0:
            oracle_count = float(patch_oracle_count.item())
            base_patch_mse = float((patch_oracle_base_error_sum / oracle_count).item())
            oracle_patch_mse = float((patch_oracle_error_sum / oracle_count).item())
            selected_patch_mse = float((patch_selected_error_sum / oracle_count).item())
            base_patch_mae = float((patch_oracle_base_mae_sum / oracle_count).item())
            oracle_patch_mae = float((patch_oracle_mae_sum / oracle_count).item())
            selected_patch_mae = float((patch_selected_mae_sum / oracle_count).item())
            headroom = base_patch_mse - oracle_patch_mse
            class_names = ["skip", *penalty_names]
            true_positive = patch_confusion_matrix.diag()
            oracle_count_by_class = patch_confusion_matrix.sum(dim=1)
            selected_count_by_class = patch_confusion_matrix.sum(dim=0)
            recall_by_class = true_positive / oracle_count_by_class.clamp_min(1.0)
            precision_by_class = true_positive / selected_count_by_class.clamp_min(1.0)
            proposal_recall_p = (
                patch_proposal_true_positive_count / patch_beneficial_penalty_count.clamp_min(1.0)
            )
            proposal_precision_p = (
                patch_proposal_true_positive_count / patch_proposed_penalty_count.clamp_min(1.0)
            )
            proposal_present_p = patch_beneficial_penalty_count > 0.0
            present_class_mask = oracle_count_by_class > 0.0
            present_penalty_mask = present_class_mask.clone()
            present_penalty_mask[0] = False
            cfg_summary["patch_router"]["oracle_diagnostic"] = {
                "path": (
                    "eval_output_anchor"
                    if output_anchor_train_with_eval
                    else "raw_residual_no_output_anchor"
                ),
                "base_patch_mse": base_patch_mse,
                "selected_patch_mse": selected_patch_mse,
                "oracle_patch_mse": oracle_patch_mse,
                "base_patch_mae": base_patch_mae,
                "selected_patch_mae": selected_patch_mae,
                "oracle_patch_mae": oracle_patch_mae,
                "selected_gain_pct": 100.0 * (base_patch_mse - selected_patch_mse) / max(base_patch_mse, 1.0e-12),
                "selected_mae_gain_pct": 100.0 * (base_patch_mae - selected_patch_mae) / max(base_patch_mae, 1.0e-12),
                "oracle_gain_pct": 100.0 * headroom / max(base_patch_mse, 1.0e-12),
                "captured_oracle_headroom_pct": (
                    100.0 * (base_patch_mse - selected_patch_mse) / headroom
                    if headroom > 1.0e-12
                    else 0.0
                ),
                "top1_accuracy": float((patch_correct_count / patch_oracle_count).item()),
                "adoption_recall": float(
                    (patch_adoption_true_positive_count / patch_oracle_penalty_count.clamp_min(1.0)).item()
                ),
                "adoption_precision": float(
                    (patch_adoption_true_positive_count / patch_selected_penalty_count.clamp_min(1.0)).item()
                ),
                "selected_utility_recall": float(
                    (patch_selected_beneficial_count / patch_oracle_penalty_count.clamp_min(1.0)).item()
                ),
                "selected_utility_precision": float(
                    (patch_selected_beneficial_count / patch_selected_penalty_count.clamp_min(1.0)).item()
                ),
                "selected_harmful_rate": float(
                    (patch_selected_harmful_count / patch_selected_penalty_count.clamp_min(1.0)).item()
                ),
                "dual_oracle_action_rate": float(
                    (patch_dual_oracle_penalty_count / patch_oracle_count).item()
                ),
                "selected_dual_utility_recall": float(
                    (
                        patch_selected_dual_beneficial_count
                        / patch_dual_oracle_penalty_count.clamp_min(1.0)
                    ).item()
                ),
                "selected_dual_utility_precision": float(
                    (
                        patch_selected_dual_beneficial_count
                        / patch_selected_penalty_count.clamp_min(1.0)
                    ).item()
                ),
                "selected_dual_harmful_rate": float(
                    (
                        patch_selected_dual_harmful_count
                        / patch_selected_penalty_count.clamp_min(1.0)
                    ).item()
                ),
                "selected_positive_gain_sum": float(patch_selected_positive_gain_sum.item()),
                "selected_negative_cost_sum": float(patch_selected_negative_cost_sum.item()),
                "selected_gain_to_cost_ratio": float(
                    (
                        patch_selected_positive_gain_sum
                        / patch_selected_negative_cost_sum.clamp_min(1.0e-12)
                    ).item()
                ),
                "risk_sign_recall": float(
                    (
                        patch_risk_sign_true_positive_count
                        / patch_risk_sign_positive_count.clamp_min(1.0)
                    ).item()
                ),
                "risk_sign_precision": float(
                    (
                        patch_risk_sign_true_positive_count
                        / patch_risk_sign_predicted_positive_count.clamp_min(1.0)
                    ).item()
                ),
                "risk_sign_accuracy": float(
                    (
                        patch_risk_sign_correct_count
                        / patch_risk_sign_count.clamp_min(1.0)
                    ).item()
                ),
                "risk_sign_predicted_positive_rate": float(
                    (
                        patch_risk_sign_predicted_positive_count
                        / patch_risk_sign_count.clamp_min(1.0)
                    ).item()
                ),
                "selected_utility_precision_by_penalty": {
                    penalty_names[p]: float(
                        (
                            patch_selected_beneficial_count_by_penalty[p]
                            / patch_selected_count_by_penalty[p].clamp_min(1.0)
                        ).item()
                    )
                    for p in range(P)
                },
                "selected_mean_gain_by_penalty": {
                    penalty_names[p]: float(
                        (
                            patch_selected_gain_sum_by_penalty[p]
                            / patch_selected_count_by_penalty[p].clamp_min(1.0)
                        ).item()
                    )
                    for p in range(P)
                },
                "oracle_penalty_recall_at_k": float(
                    (patch_oracle_penalty_hit_count / patch_oracle_penalty_count.clamp_min(1.0)).item()
                ),
                "proposal_macro_recall": float(
                    proposal_recall_p[proposal_present_p].mean().item()
                    if bool(proposal_present_p.any().item())
                    else 0.0
                ),
                "proposal_oracle_best_recall_at_k": float(
                    (
                        patch_proposal_oracle_hit_count
                        / patch_oracle_penalty_count.clamp_min(1.0)
                    ).item()
                ),
                "shortlist_pairwise_accuracy": float(
                    (
                        patch_shortlist_pairwise_correct_count
                        / patch_shortlist_pairwise_count.clamp_min(1.0)
                    ).item()
                ),
                "proposal_oracle_best_recall_by_penalty": {
                    penalty_names[p]: float(
                        (
                            patch_proposal_oracle_hit_count_by_penalty[p]
                            / patch_oracle_class_count[p + 1].clamp_min(1.0)
                        ).item()
                    )
                    for p in range(P)
                },
                "mean_beneficial_penalties_per_patch": float(
                    (patch_beneficial_cardinality_sum / patch_oracle_count).item()
                ),
                "beneficial_penalty_count_distribution": {
                    str(count): float(
                        (patch_beneficial_cardinality_histogram[count] / patch_oracle_count).item()
                    )
                    for count in range(P + 1)
                },
                "proposal_recall_by_penalty": {
                    penalty_names[p]: float(proposal_recall_p[p].item()) for p in range(P)
                },
                "proposal_precision_by_penalty": {
                    penalty_names[p]: float(proposal_precision_p[p].item()) for p in range(P)
                },
                "macro_recall": float(recall_by_class[present_class_mask].mean().item()),
                "macro_penalty_recall": float(
                    recall_by_class[present_penalty_mask].mean().item()
                    if bool(present_penalty_mask.any().item())
                    else 0.0
                ),
                "skip_recall": float(recall_by_class[0].item()),
                "recall_by_class": {
                    class_names[i]: float(recall_by_class[i].item()) for i in range(P + 1)
                },
                "precision_by_class": {
                    class_names[i]: float(precision_by_class[i].item()) for i in range(P + 1)
                },
                "confusion_matrix_oracle_rows_selected_columns": [
                    [int(value) for value in row]
                    for row in patch_confusion_matrix.to(dtype=torch.long).detach().cpu().tolist()
                ],
                "oracle_class_rate": {
                    class_names[i]: float((patch_oracle_class_count[i] / patch_oracle_count).item())
                    for i in range(P + 1)
                },
                "selected_class_rate": {
                    class_names[i]: float((patch_selected_class_count[i] / patch_oracle_count).item())
                    for i in range(P + 1)
                },
                "by_patch": [
                    {
                        "index": int(q),
                        "base_mse": float((patch_base_error_sum_by_patch[q] / patch_count_by_patch[q].clamp_min(1.0)).item()),
                        "selected_mse": float((patch_selected_error_sum_by_patch[q] / patch_count_by_patch[q].clamp_min(1.0)).item()),
                        "oracle_mse": float((patch_oracle_error_sum_by_patch[q] / patch_count_by_patch[q].clamp_min(1.0)).item()),
                    }
                    for q in range(int(patch_router.num_patches))
                ],
            }
        return cfg_summary

    @torch.no_grad()
    def collect_semantic_bank_candidate_audit(
        loader: DataLoader,
        *,
        eval_start: int,
        split_name: str,
        split_window_count: int,
        include_level_oracle_diagnostic: bool = False,
    ) -> Dict[str, object]:
        split = str(split_name).strip().lower()
        if split not in {"train", "val"}:
            raise ValueError("semantic bank candidate audit split must be train or val.")
        window_count = int(split_window_count)
        audit: Dict[str, object] = {
            "enabled": bool(pred_residual_semantic_bank_stage1),
            "split": split,
            "patch_len": int(pred_residual_candidate_supervision_need_patch_len),
            "test_read": False,
        }
        if not pred_residual_semantic_bank_stage1:
            return audit
        if window_count <= 0:
            raise ValueError("semantic bank candidate audit split_window_count must be positive.")
        loader_window_count = int(len(loader.dataset))
        if loader_window_count != window_count:
            raise ValueError(
                "semantic bank candidate audit split_window_count must match "
                f"the loader dataset: {window_count} != {loader_window_count}."
            )
        audit.update(
            {
                "split_window_count": window_count,
                "training_mutation": False,
                "selection_effect": False,
            }
        )
        if pred_residual_semantic_level_acceptance_candidate_source != "executed":
            audit["level_acceptance_candidate"] = str(
                pred_residual_semantic_level_acceptance_candidate_source
            )
        if pred_residual is None or penalty_need_threshold is None or len(loader) == 0:
            audit.update({"all_pass": False, "reason": "missing_model_threshold_or_loader"})
            return audit
        patch_len = int(pred_residual_candidate_supervision_need_patch_len)
        totals = [
            {
                "high_count": 0.0,
                "low_count": 0.0,
                "high_base_penalty": 0.0,
                "high_candidate_penalty": 0.0,
                "high_base_mse": 0.0,
                "high_candidate_mse": 0.0,
                "low_base_mse": 0.0,
                "low_candidate_mse": 0.0,
                "high_correction_sq_sum": 0.0,
                "low_correction_sq_sum": 0.0,
                "high_correction_numel": 0.0,
                "low_correction_numel": 0.0,
                "high_improved_count": 0.0,
                "high_relative_gain_parts": [],
                "min_high_need_improved_fraction": float(
                    pred_residual_semantic_min_improved_fraction
                ),
            }
            for _ in range(P)
        ]
        temporal_totals = [
            [
                {
                    "high_count": 0.0,
                    "high_base_penalty": 0.0,
                    "high_candidate_penalty": 0.0,
                }
                for _ in range(pred_residual_semantic_validation_blocks)
            ]
            for _ in range(P)
        ]
        level_oracle_parts: Optional[Dict[str, object]] = None
        level_penalty_index: Optional[int] = None
        if bool(include_level_oracle_diagnostic):
            if "level" not in penalty_names:
                raise ValueError("LEVEL oracle diagnostic requires a level penalty.")
            level_penalty_index = int(penalty_names.index("level"))
            level_oracle_parts = {
                "base_penalty": [],
                "normalized_base_penalty": [],
                "oracle_correction": [],
                "raw_amplitude": [],
                "raw_signed_ratio": [],
                "raw_absolute_ratio": [],
                "adapter_correction": [],
                "signed_ratio": [],
                "absolute_ratio": [],
                "adapter_semantic_gain": [],
                "oracle_gain_base_mse_ratio": [],
                "sign_agreement_count": 0,
                "raw_sign_agreement_count": 0,
                "ratio_valid_count": 0,
                "need_tp": 0,
                "need_fp": 0,
                "need_tn": 0,
                "need_fn": 0,
                "need_probability": [],
                "count": 0,
                "base_penalty_consistency_error_max": 0.0,
                "oracle_penalty_sum": 0.0,
                "oracle_penalty_max": 0.0,
                "oracle_mse_gain_identity_error_sum": 0.0,
                "oracle_mse_gain_identity_error_max": 0.0,
                "exemplar_base_penalty": float("-inf"),
                "exemplar": None,
                "acceptance_candidate_source": str(
                    pred_residual_semantic_level_acceptance_candidate_source
                ),
            }
        model.eval()
        pred_residual.eval()
        for x_audit, y_audit, idx_audit in loader:
            x_audit = x_audit.to(device, non_blocking=True)
            y_audit = y_audit.to(device, non_blocking=True)
            query_start = int(eval_start) + idx_audit.to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            )
            relative_window_index = idx_audit.to(device=device, dtype=torch.long)
            block_index_b = _semantic_bank_temporal_block_indices(
                relative_window_index,
                split_window_count=window_count,
                block_count=pred_residual_semantic_validation_blocks,
            )
            y_base = _eval_path_base_prediction(x_audit, query_start)
            route = torch.ones(
                int(x_audit.shape[0]),
                K,
                P,
                device=device,
                dtype=y_base.dtype,
            )
            pred_out = pred_residual(
                x_audit,
                y_base,
                cluster_id_c,
                route,
                skip_bk=None,
                query_start_abs_b=query_start,
                fixed_expert_delta_bch=None,
            )
            _assert_pure_candidate_base(pred_out, y_base, stage="semantic_bank_audit")
            candidates = _pred_residual_candidate_predictions(
                y_base,
                pred_out,
                include_intervention=False,
                include_selector=False,
                include_patch_route=False,
            )
            if candidates is None:
                raise RuntimeError("semantic bank audit could not construct candidate predictions.")
            batch, channels, horizon = y_base.shape
            patches = horizon // patch_len
            base_patch = y_base.reshape(batch, channels, patches, patch_len)
            target_patch = y_audit.reshape(batch, channels, patches, patch_len)
            base_mse = (base_patch - target_patch).square().mean(dim=-1)
            for p, name in enumerate(penalty_names):
                candidate = candidates[:, :, p, :]
                candidate_patch = candidate.reshape(batch, channels, patches, patch_len)
                executed_candidate_patch = candidate_patch
                candidate_mse = (candidate_patch - target_patch).square().mean(dim=-1)
                correction_sq = (candidate_patch - base_patch).square()
                base_penalty = _patchwise_penalty_bcq(
                    y_base,
                    y_audit,
                    penalty_fns[name],
                    patch_len=patch_len,
                )
                candidate_penalty = _patchwise_penalty_bcq(
                    candidate,
                    y_audit,
                    penalty_fns[name],
                    patch_len=patch_len,
                )
                if (
                    name == "level"
                    and bool(
                        pred_residual_semantic_level_controller_cfg.get(
                            "enable",
                            False,
                        )
                    )
                ):
                    if "level_amplitude_bcq" not in pred_out:
                        raise RuntimeError(
                            "LEVEL Stage-1 acceptance is missing level_amplitude_bcq."
                        )
                    if "level_executed_correction_bcq" not in pred_out:
                        raise RuntimeError(
                            "LEVEL Stage-1 acceptance is missing "
                            "level_executed_correction_bcq."
                        )
                    raw_acceptance_amplitude = pred_out["level_amplitude_bcq"]
                    executed_acceptance_correction = pred_out[
                        "level_executed_correction_bcq"
                    ]
                    _validate_level_executed_candidate_patch(
                        base_patch=base_patch,
                        executed_candidate_patch=executed_candidate_patch,
                        executed_correction_bcq=executed_acceptance_correction,
                    )
                    acceptance_candidate_patch = _level_stage1_acceptance_candidate_patch(
                        base_patch=base_patch,
                        executed_candidate_patch=executed_candidate_patch,
                        raw_amplitude_bcq=raw_acceptance_amplitude,
                        source=(
                            pred_residual_semantic_level_acceptance_candidate_source
                        ),
                    )
                    if (
                        pred_residual_semantic_level_acceptance_candidate_source
                        == "raw_amplitude"
                    ):
                        candidate_patch = acceptance_candidate_patch
                        candidate = candidate_patch.reshape(batch, channels, horizon)
                        candidate_mse = (
                            candidate_patch - target_patch
                        ).square().mean(dim=-1)
                        correction_sq = (candidate_patch - base_patch).square()
                        candidate_penalty = _patchwise_penalty_bcq(
                            candidate,
                            y_audit,
                            penalty_fns[name],
                            patch_len=patch_len,
                        )
                need = (
                    base_penalty / penalty_scale[p].to(base_penalty)
                    >= penalty_need_threshold[p].to(base_penalty)
                )
                low = ~need
                if (
                    level_oracle_parts is not None
                    and level_penalty_index is not None
                    and p == level_penalty_index
                ):
                    required_controller_outputs = {
                        "level_amplitude_bcq",
                        "level_need_probability_bcq",
                        "level_need_hard_bcq",
                        "level_executed_correction_bcq",
                    }
                    if bool(pred_residual_semantic_level_controller_cfg.get("enable", False)):
                        missing_controller = sorted(
                            required_controller_outputs - set(pred_out)
                        )
                        if missing_controller:
                            raise RuntimeError(
                                "LEVEL audit is missing controller outputs: "
                                + ", ".join(missing_controller)
                            )
                        raw_amplitude_all = pred_out["level_amplitude_bcq"]
                        need_probability_all = pred_out["level_need_probability_bcq"]
                        need_hard_all = pred_out["level_need_hard_bcq"].to(dtype=torch.bool)
                        executed_all = executed_acceptance_correction
                        level_oracle_parts["need_probability"].append(
                            need_probability_all.detach().cpu().reshape(-1)
                        )
                        level_oracle_parts["need_tp"] += int((need_hard_all & need).sum().item())
                        level_oracle_parts["need_fp"] += int((need_hard_all & ~need).sum().item())
                        level_oracle_parts["need_tn"] += int((~need_hard_all & ~need).sum().item())
                        level_oracle_parts["need_fn"] += int((~need_hard_all & need).sum().item())
                    else:
                        raw_amplitude_all = (
                            candidate_patch.mean(dim=-1) - base_patch.mean(dim=-1)
                        )
                        executed_all = raw_amplitude_all
                    level_diag = _level_oracle_patch_diagnostics(
                        base_patch,
                        target_patch,
                        candidate_patch,
                    )
                    high_count_level = int(need.sum().item())
                    if high_count_level > 0:
                        oracle_correction = level_diag["oracle_correction"][need]
                        raw_amplitude = raw_amplitude_all[need]
                        adapter_correction = level_diag["adapter_correction"][need]
                        level_base_penalty = level_diag["base_penalty"][need]
                        level_candidate_penalty = level_diag["candidate_penalty"][need]
                        level_base_mse = level_diag["base_mse"][need]
                        level_oracle_mse = level_diag["oracle_mse"][need]
                        ratio_valid = oracle_correction.abs() > 1.0e-12
                        signed_ratio = adapter_correction[ratio_valid] / oracle_correction[ratio_valid]
                        absolute_ratio = adapter_correction[ratio_valid].abs() / oracle_correction[ratio_valid].abs()
                        raw_signed_ratio = raw_amplitude[ratio_valid] / oracle_correction[ratio_valid]
                        raw_absolute_ratio = raw_amplitude[ratio_valid].abs() / oracle_correction[ratio_valid].abs()
                        semantic_gain = (
                            level_base_penalty - level_candidate_penalty
                        ) / level_base_penalty.clamp_min(1.0e-12)
                        oracle_mse_gain = level_base_mse - level_oracle_mse
                        oracle_gain_base_mse_ratio = (
                            oracle_mse_gain / level_base_mse.clamp_min(1.0e-12)
                        )
                        tensor_parts = {
                            "base_penalty": level_base_penalty,
                            "normalized_base_penalty": (
                                level_base_penalty / penalty_scale[p].to(level_base_penalty)
                            ),
                            "oracle_correction": oracle_correction,
                            "raw_amplitude": raw_amplitude,
                            "raw_signed_ratio": raw_signed_ratio,
                            "raw_absolute_ratio": raw_absolute_ratio,
                            "adapter_correction": adapter_correction,
                            "signed_ratio": signed_ratio,
                            "absolute_ratio": absolute_ratio,
                            "adapter_semantic_gain": semantic_gain,
                            "oracle_gain_base_mse_ratio": oracle_gain_base_mse_ratio,
                        }
                        for key, values in tensor_parts.items():
                            level_oracle_parts[key].append(values.detach().cpu())
                        level_oracle_parts["sign_agreement_count"] += int(
                            ((adapter_correction[ratio_valid] * oracle_correction[ratio_valid]) > 0.0).sum().item()
                        )
                        level_oracle_parts["raw_sign_agreement_count"] += int(
                            ((raw_amplitude[ratio_valid] * oracle_correction[ratio_valid]) > 0.0).sum().item()
                        )
                        level_oracle_parts["ratio_valid_count"] += int(ratio_valid.sum().item())
                        level_oracle_parts["count"] += high_count_level
                        consistency_error = (
                            level_diag["base_penalty"] - base_penalty
                        ).abs()[need]
                        level_oracle_parts["base_penalty_consistency_error_max"] = max(
                            float(level_oracle_parts["base_penalty_consistency_error_max"]),
                            float(consistency_error.max().item()),
                        )
                        oracle_penalty_high = level_diag["oracle_penalty"][need]
                        level_oracle_parts["oracle_penalty_sum"] += float(
                            oracle_penalty_high.sum().item()
                        )
                        level_oracle_parts["oracle_penalty_max"] = max(
                            float(level_oracle_parts["oracle_penalty_max"]),
                            float(oracle_penalty_high.max().item()),
                        )
                        identity_error = level_diag[
                            "oracle_mse_gain_identity_error"
                        ][need]
                        level_oracle_parts[
                            "oracle_mse_gain_identity_error_sum"
                        ] += float(identity_error.sum().item())
                        level_oracle_parts[
                            "oracle_mse_gain_identity_error_max"
                        ] = max(
                            float(
                                level_oracle_parts[
                                    "oracle_mse_gain_identity_error_max"
                                ]
                            ),
                            float(identity_error.max().item()),
                        )

                        masked_base_penalty = base_penalty.masked_fill(
                            ~need,
                            float("-inf"),
                        )
                        batch_max = float(masked_base_penalty.max().item())
                        if batch_max > float(
                            level_oracle_parts["exemplar_base_penalty"]
                        ):
                            flat_index = int(masked_base_penalty.reshape(-1).argmax().item())
                            exemplar_patch = flat_index % patches
                            flat_index //= patches
                            exemplar_channel = flat_index % channels
                            exemplar_batch = flat_index // channels
                            relative_window = int(idx_audit[exemplar_batch].item())
                            oracle_patch = level_diag["oracle_patch"]
                            level_oracle_parts["exemplar_base_penalty"] = batch_max
                            level_oracle_parts["exemplar"] = {
                                "split": split,
                                "relative_window": relative_window,
                                "absolute_query_start": int(eval_start) + relative_window,
                                "channel_index": int(exemplar_channel),
                                "channel_name": str(channel_names[exemplar_channel]),
                                "patch_index": int(exemplar_patch),
                                "horizon_start": int(exemplar_patch * patch_len),
                                "horizon_end_exclusive": int(
                                    (exemplar_patch + 1) * patch_len
                                ),
                                "base_mean": float(
                                    level_diag["base_mean"][
                                        exemplar_batch,
                                        exemplar_channel,
                                        exemplar_patch,
                                    ].item()
                                ),
                                "target_mean": float(
                                    level_diag["target_mean"][
                                        exemplar_batch,
                                        exemplar_channel,
                                        exemplar_patch,
                                    ].item()
                                ),
                                "candidate_mean": float(
                                    level_diag["candidate_mean"][
                                        exemplar_batch,
                                        exemplar_channel,
                                        exemplar_patch,
                                    ].item()
                                ),
                                "adapter_correction": float(
                                    level_diag["adapter_correction"][
                                        exemplar_batch,
                                        exemplar_channel,
                                        exemplar_patch,
                                    ].item()
                                ),
                                "raw_amplitude": float(
                                    raw_amplitude_all[
                                        exemplar_batch,
                                        exemplar_channel,
                                        exemplar_patch,
                                    ].item()
                                ),
                                "need_probability": (
                                    float(need_probability_all[
                                        exemplar_batch,
                                        exemplar_channel,
                                        exemplar_patch,
                                    ].item())
                                    if bool(pred_residual_semantic_level_controller_cfg.get("enable", False))
                                    else None
                                ),
                                "need_hard": (
                                    bool(need_hard_all[
                                        exemplar_batch,
                                        exemplar_channel,
                                        exemplar_patch,
                                    ].item())
                                    if bool(pred_residual_semantic_level_controller_cfg.get("enable", False))
                                    else True
                                ),
                                "executed_correction": float(
                                    executed_all[
                                        exemplar_batch,
                                        exemplar_channel,
                                        exemplar_patch,
                                    ].item()
                                ),
                                "oracle_correction": float(
                                    level_diag["oracle_correction"][
                                        exemplar_batch,
                                        exemplar_channel,
                                        exemplar_patch,
                                    ].item()
                                ),
                                "base_penalty": float(
                                    level_diag["base_penalty"][
                                        exemplar_batch,
                                        exemplar_channel,
                                        exemplar_patch,
                                    ].item()
                                ),
                                "candidate_penalty": float(
                                    level_diag["candidate_penalty"][
                                        exemplar_batch,
                                        exemplar_channel,
                                        exemplar_patch,
                                    ].item()
                                ),
                                "oracle_penalty": float(
                                    level_diag["oracle_penalty"][
                                        exemplar_batch,
                                        exemplar_channel,
                                        exemplar_patch,
                                    ].item()
                                ),
                                "base_mse": float(
                                    level_diag["base_mse"][
                                        exemplar_batch,
                                        exemplar_channel,
                                        exemplar_patch,
                                    ].item()
                                ),
                                "candidate_mse": float(
                                    level_diag["candidate_mse"][
                                        exemplar_batch,
                                        exemplar_channel,
                                        exemplar_patch,
                                    ].item()
                                ),
                                "oracle_mse": float(
                                    level_diag["oracle_mse"][
                                        exemplar_batch,
                                        exemplar_channel,
                                        exemplar_patch,
                                    ].item()
                                ),
                                "base_patch": base_patch[
                                    exemplar_batch,
                                    exemplar_channel,
                                    exemplar_patch,
                                ].detach().cpu().tolist(),
                                "target_patch": target_patch[
                                    exemplar_batch,
                                    exemplar_channel,
                                    exemplar_patch,
                                ].detach().cpu().tolist(),
                                "candidate_patch": candidate_patch[
                                    exemplar_batch,
                                    exemplar_channel,
                                    exemplar_patch,
                                ].detach().cpu().tolist(),
                                "oracle_patch": oracle_patch[
                                    exemplar_batch,
                                    exemplar_channel,
                                    exemplar_patch,
                                ].detach().cpu().tolist(),
                            }
                current = totals[p]
                high_count = int(need.sum().item())
                low_count = int(low.sum().item())
                current["high_count"] += float(high_count)
                current["low_count"] += float(low_count)
                current["high_base_penalty"] += float(base_penalty[need].sum().item())
                current["high_candidate_penalty"] += float(candidate_penalty[need].sum().item())
                current["high_base_mse"] += float(base_mse[need].sum().item())
                current["high_candidate_mse"] += float(candidate_mse[need].sum().item())
                current["low_base_mse"] += float(base_mse[low].sum().item())
                current["low_candidate_mse"] += float(candidate_mse[low].sum().item())
                current["high_correction_sq_sum"] += float(
                    correction_sq[need.unsqueeze(-1).expand_as(correction_sq)].sum().item()
                )
                current["low_correction_sq_sum"] += float(
                    correction_sq[low.unsqueeze(-1).expand_as(correction_sq)].sum().item()
                )
                current["high_correction_numel"] += float(high_count * patch_len)
                current["low_correction_numel"] += float(low_count * patch_len)
                high_relative_gain = (
                    (base_penalty - candidate_penalty)
                    / base_penalty.clamp_min(1.0e-8)
                )
                current["high_improved_count"] += float(
                    ((candidate_penalty < base_penalty) & need).sum().item()
                )
                current["high_relative_gain_parts"].append(
                    high_relative_gain[need].detach().cpu()
                )
                for block_index in range(pred_residual_semantic_validation_blocks):
                    block_mask = need & (
                        block_index_b.view(batch, 1, 1) == block_index
                    )
                    block_row = temporal_totals[p][block_index]
                    block_row["high_count"] += float(block_mask.sum().item())
                    block_row["high_base_penalty"] += float(
                        base_penalty[block_mask].sum().item()
                    )
                    block_row["high_candidate_penalty"] += float(
                        candidate_penalty[block_mask].sum().item()
                    )

        for current in totals:
            parts = current.pop("high_relative_gain_parts")
            values = (
                torch.cat(parts, dim=0).to(dtype=torch.float64)
                if parts
                else torch.empty(0, dtype=torch.float64)
            )
            current["high_relative_gain_quantiles"] = {
                str(q): (
                    float(torch.quantile(values, q).item())
                    if int(values.numel()) > 0
                    else float("nan")
                )
                for q in (0.1, 0.25, 0.5, 0.75, 0.9)
            }
        acceptance = _semantic_bank_semantic_only_acceptance_metrics(
            totals_by_penalty=totals,
            temporal_totals_by_penalty=temporal_totals,
            penalty_names=penalty_names,
            min_matching_gain_by_name=pred_residual_semantic_min_gain_by_name,
            eps=1.0e-8,
        )
        audit.update(
            {
                "scale": {
                    name: float(penalty_scale[p].detach().cpu().item())
                    for p, name in enumerate(penalty_names)
                },
                "threshold": {
                    name: float(penalty_need_threshold[p].detach().cpu().item())
                    for p, name in enumerate(penalty_names)
                },
                **acceptance,
            }
        )
        if level_oracle_parts is not None:
            quantile_levels = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)

            def diagnostic_values(key: str) -> torch.Tensor:
                parts = level_oracle_parts[key]
                return (
                    torch.cat(parts, dim=0).to(dtype=torch.float64)
                    if parts
                    else torch.empty(0, dtype=torch.float64)
                )

            def diagnostic_quantiles(values: torch.Tensor) -> Dict[str, float]:
                return {
                    str(q): (
                        float(torch.quantile(values, q).item())
                        if int(values.numel()) > 0
                        else float("nan")
                    )
                    for q in quantile_levels
                }

            base_penalty_values = diagnostic_values("base_penalty")
            normalized_base_penalty_values = diagnostic_values(
                "normalized_base_penalty"
            )
            oracle_correction_values = diagnostic_values("oracle_correction")
            raw_amplitude_values = diagnostic_values("raw_amplitude")
            adapter_correction_values = diagnostic_values("adapter_correction")

            def prediction_fit(
                prediction: torch.Tensor,
                target: torch.Tensor,
            ) -> Tuple[float, float, Dict[str, object]]:
                if int(target.numel()) == 0 or prediction.shape != target.shape:
                    return float("nan"), float("nan"), {
                        "count": 0,
                        "prediction_sum": 0.0,
                        "prediction_squared_sum": 0.0,
                        "target_sum": 0.0,
                        "target_squared_sum": 0.0,
                        "prediction_target_cross_sum": 0.0,
                        "squared_error_sum": 0.0,
                    }
                sufficient = _prediction_fit_sufficient_statistics(
                    prediction,
                    target,
                )
                count = int(sufficient["count"])
                prediction_centered_ss = max(
                    float(sufficient["prediction_squared_sum"])
                    - float(sufficient["prediction_sum"]) ** 2 / max(count, 1),
                    0.0,
                )
                target_centered_ss = max(
                    float(sufficient["target_squared_sum"])
                    - float(sufficient["target_sum"]) ** 2 / max(count, 1),
                    0.0,
                )
                centered_cross = (
                    float(sufficient["prediction_target_cross_sum"])
                    - float(sufficient["prediction_sum"])
                    * float(sufficient["target_sum"])
                    / max(count, 1)
                )
                correlation_denom = math.sqrt(
                    prediction_centered_ss * target_centered_ss
                )
                pearson = (
                    float(centered_cross / correlation_denom)
                    if correlation_denom > 0.0
                    else float("nan")
                )
                r2_zero = (
                    1.0
                    - float(sufficient["squared_error_sum"])
                    / max(float(sufficient["target_squared_sum"]), 1.0e-12)
                )
                return pearson, r2_zero, sufficient

            raw_oracle_pearson, raw_r2_vs_zero, raw_fit_sufficient = prediction_fit(
                raw_amplitude_values,
                oracle_correction_values,
            )
            adapter_oracle_pearson, adapter_r2_vs_zero, _ = prediction_fit(
                adapter_correction_values,
                oracle_correction_values,
            )
            base_penalty_sum = float(base_penalty_values.sum().item())
            count_level = int(level_oracle_parts["count"])
            need_tp = int(level_oracle_parts["need_tp"])
            need_fp = int(level_oracle_parts["need_fp"])
            need_tn = int(level_oracle_parts["need_tn"])
            need_fn = int(level_oracle_parts["need_fn"])
            need_total = need_tp + need_fp + need_tn + need_fn
            audit["level_oracle_diagnostic"] = {
                "enabled": True,
                "target_usage": "final_audit_only",
                "non_selecting": True,
                "test_read": False,
                **(
                    {
                        "acceptance_candidate_source": str(
                            level_oracle_parts["acceptance_candidate_source"]
                        )
                    }
                    if str(level_oracle_parts["acceptance_candidate_source"])
                    != "executed"
                    else {}
                ),
                "high_need_count": count_level,
                "base_penalty_quantiles": diagnostic_quantiles(
                    base_penalty_values
                ),
                "normalized_base_penalty_quantiles": diagnostic_quantiles(
                    normalized_base_penalty_values
                ),
                "oracle_correction_quantiles": diagnostic_quantiles(
                    oracle_correction_values
                ),
                "raw_amplitude_quantiles": diagnostic_quantiles(
                    raw_amplitude_values
                ),
                "raw_signed_oracle_ratio_quantiles": diagnostic_quantiles(
                    diagnostic_values("raw_signed_ratio")
                ),
                "raw_absolute_oracle_ratio_quantiles": diagnostic_quantiles(
                    diagnostic_values("raw_absolute_ratio")
                ),
                "adapter_correction_quantiles": diagnostic_quantiles(
                    adapter_correction_values
                ),
                "signed_adapter_oracle_ratio_quantiles": diagnostic_quantiles(
                    diagnostic_values("signed_ratio")
                ),
                "absolute_adapter_oracle_ratio_quantiles": diagnostic_quantiles(
                    diagnostic_values("absolute_ratio")
                ),
                "adapter_semantic_gain_quantiles": diagnostic_quantiles(
                    diagnostic_values("adapter_semantic_gain")
                ),
                "oracle_gain_base_mse_ratio_quantiles": diagnostic_quantiles(
                    diagnostic_values("oracle_gain_base_mse_ratio")
                ),
                "adapter_oracle_sign_agreement_fraction": (
                    float(level_oracle_parts["sign_agreement_count"])
                    / max(int(level_oracle_parts["ratio_valid_count"]), 1)
                ),
                "ratio_valid_count": int(level_oracle_parts["ratio_valid_count"]),
                "raw_amplitude_oracle_pearson": raw_oracle_pearson,
                "raw_amplitude_r2_vs_zero_correction": raw_r2_vs_zero,
                "raw_amplitude_fit_sufficient_statistics": raw_fit_sufficient,
                "raw_amplitude_oracle_sign_agreement_fraction": (
                    float(level_oracle_parts["raw_sign_agreement_count"])
                    / max(int(level_oracle_parts["ratio_valid_count"]), 1)
                ),
                "adapter_oracle_pearson": adapter_oracle_pearson,
                "adapter_r2_vs_zero_correction": adapter_r2_vs_zero,
                "need_gate": {
                    "tp": need_tp,
                    "fp": need_fp,
                    "tn": need_tn,
                    "fn": need_fn,
                    "recall": float(need_tp) / max(need_tp + need_fn, 1),
                    "precision": float(need_tp) / max(need_tp + need_fp, 1),
                    "false_positive_rate": float(need_fp) / max(need_fp + need_tn, 1),
                    "predicted_adoption_rate": float(need_tp + need_fp) / max(need_total, 1),
                    "probability_quantiles": diagnostic_quantiles(
                        diagnostic_values("need_probability")
                    ),
                },
                "oracle_matching_penalty_gain": (
                    1.0
                    - float(level_oracle_parts["oracle_penalty_sum"])
                    / max(base_penalty_sum, 1.0e-12)
                ),
                "base_penalty_consistency_error_max": float(
                    level_oracle_parts[
                        "base_penalty_consistency_error_max"
                    ]
                ),
                "oracle_penalty_max": float(
                    level_oracle_parts["oracle_penalty_max"]
                ),
                "oracle_mse_gain_identity_error_mean": (
                    float(
                        level_oracle_parts[
                            "oracle_mse_gain_identity_error_sum"
                        ]
                    )
                    / max(count_level, 1)
                ),
                "oracle_mse_gain_identity_error_max": float(
                    level_oracle_parts[
                        "oracle_mse_gain_identity_error_max"
                    ]
                ),
                "strongest_need_exemplar": level_oracle_parts["exemplar"],
            }
        return audit

    @torch.no_grad()
    def collect_patch_risk_calibration_tensors(
        loader: DataLoader,
        *,
        eval_start: int = 0,
        include_patch_values: bool = False,
        include_all_candidates: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if pred_residual is None or getattr(pred_residual, "patch_router", None) is None:
            raise ValueError("patch risk calibration requires an enabled patch router.")
        model.eval()
        gate.eval()
        pred_residual.eval()
        score_parts = []
        gain_parts = []
        time_parts = []
        penalty_parts = []
        base_mse_parts = []
        candidate_mse_parts = []
        base_mae_parts = []
        candidate_mae_parts = []
        candidate_mse_all_parts = []
        candidate_mae_all_parts = []
        candidate_score_all_parts = []
        regime_parts = []
        cross_parts = []
        delta_sq_parts = []
        base_residual_patch_parts = []
        candidate_delta_patch_parts = []
        scale_feature_parts = []
        head_score_parts: Dict[str, List[torch.Tensor]] = {
            "proposal_adopt_probability": [],
            "proposal_fixed_probability": [],
            "proposal_fixed_logit": [],
            "risk_fixed_probability": [],
            "risk_domain_disagreement": [],
            "utility_fixed_score": [],
            "pairwise_fixed_score": [],
            "lower_quantile_fixed_score": [],
            "utility_veto_fixed_probability": [],
        }
        for x, y, idx in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            idx = idx.to(device=device, dtype=torch.long, non_blocking=True)
            query_start_abs_b = int(eval_start) + idx
            y_base = _eval_path_base_prediction(x, query_start_abs_b)
            fixed_expert_delta_bch = build_moe_output_anchor_fixed_expert_delta(
                y_base,
                x_bcl=x,
                query_start_abs_b=query_start_abs_b,
                input_len=L,
                moe_cfg=moe_cfg,
                moe_enable=moe_enable,
                observed_history_tc=data_window_tc,
                train_stat_anchor_pc=train_stat_anchor_pc,
                train_residual_anchor_phc=train_residual_anchor_phc,
                learnable_output_anchor=learnable_output_anchor,
                cluster_id_c=cluster_id_c,
            )
            routing_base_bch = (
                y_base
                if fixed_expert_delta_bch is None
                else y_base + float(periodic_anchor_expert_scale) * fixed_expert_delta_bch
            )
            route_pen_bkp = _router_penalty_context_from_history(
                x_bcl=x,
                yhat_base_bch=routing_base_bch,
                penalty_names=penalty_names,
                penalty_fns=penalty_fns,
                penalty_scale=penalty_scale,
                cluster_id_c=cluster_id_c,
                K=K,
                router_mode=router_mode,
            )
            feat_bkf = _build_gate_routing_features(
                x,
                routing_base_bch,
                cluster_id_c,
                K,
                mode=gate_feature_mode,
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
            if select_ranks is not None:
                mask_bkp = _select_rank_mask(
                    probs_bkp,
                    select_ranks,
                    straight_through=False,
                )
            pred_out = pred_residual(
                x,
                y_base,
                cluster_id_c,
                mask_bkp,
                skip_bk=skip_bk if allow_skip else None,
                query_start_abs_b=query_start_abs_b,
                fixed_expert_delta_bch=fixed_expert_delta_bch,
            )
            _assert_pure_candidate_base(pred_out, y_base, stage="calibration_summary")
            selected_score_bcq = pred_out.get("patch_selected_risk_score_bcq")
            selected_penalty_bcq = pred_out.get("patch_selected_penalty_index_bcq")
            candidate_score_bcqp = pred_out.get(
                "patch_penalty_utility_scores_bcqp"
            )
            calibration_base_bch, candidate_bcpH = _pred_residual_candidates_on_eval_path(
                y_base,
                pred_out,
                apply_output_anchors=output_anchor_train_with_eval,
                x_bcl=x,
                query_start_abs_b=query_start_abs_b,
                input_len=L,
                moe_cfg=moe_cfg,
                moe_enable=moe_enable,
                observed_history_tc=data_window_tc,
                train_stat_anchor_pc=train_stat_anchor_pc,
                train_residual_anchor_phc=train_residual_anchor_phc,
                learnable_output_anchor=learnable_output_anchor,
                cluster_id_c=cluster_id_c,
                include_patch_route=False,
            )
            if (
                selected_score_bcq is None
                or selected_penalty_bcq is None
                or candidate_bcpH is None
            ):
                raise ValueError("patch risk calibration outputs are incomplete.")

            def gather_selected_head(output_key: str) -> Optional[torch.Tensor]:
                head_bcqp = pred_out.get(output_key)
                if head_bcqp is None:
                    return None
                if tuple(head_bcqp.shape[:-1]) != tuple(selected_penalty_bcq.shape):
                    raise ValueError(
                        f"patch risk head {output_key} shape must match selected penalties."
                    )
                return head_bcqp.gather(
                    dim=-1,
                    index=selected_penalty_bcq.unsqueeze(-1),
                ).squeeze(-1)

            batch_head_scores = {
                "proposal_adopt_probability": pred_out.get(
                    "patch_proposal_adopt_prob_bcq"
                ),
                "proposal_fixed_probability": gather_selected_head(
                    "patch_penalty_benefit_probs_bcqp"
                ),
                "proposal_fixed_logit": gather_selected_head(
                    "patch_penalty_proposal_logits_bcqp"
                ),
                "risk_fixed_probability": gather_selected_head(
                    "patch_penalty_risk_benefit_probs_bcqp"
                ),
                "risk_domain_disagreement": pred_out.get(
                    "patch_selected_risk_domain_std_bcq"
                ),
                "utility_fixed_score": gather_selected_head(
                    "patch_penalty_utility_scores_bcqp"
                ),
                "pairwise_fixed_score": gather_selected_head(
                    "patch_penalty_pairwise_rank_scores_bcqp"
                ),
                "lower_quantile_fixed_score": gather_selected_head(
                    "patch_penalty_risk_lower_quantile_scores_bcqp"
                ),
                "utility_veto_fixed_probability": gather_selected_head(
                    "patch_penalty_risk_utility_veto_probs_bcqp"
                ),
            }
            for head_name, head_score_bcq in batch_head_scores.items():
                if head_score_bcq is None:
                    continue
                if tuple(head_score_bcq.shape) != tuple(selected_penalty_bcq.shape):
                    raise ValueError(
                        f"patch risk head {head_name} shape must match selected penalties."
                    )
                head_score_parts[head_name].append(head_score_bcq.detach().cpu())
            batch, channels, horizon = calibration_base_bch.shape
            patches = int(selected_score_bcq.shape[2])
            if horizon % patches != 0:
                raise ValueError("patch risk calibration patch count must divide horizon.")
            patch_len = horizon // patches
            base_error_bcq = (calibration_base_bch - y).square().reshape(
                batch,
                channels,
                patches,
                patch_len,
            ).mean(dim=-1)
            candidate_error_bcqp = (
                (candidate_bcpH - y.unsqueeze(2))
                .square()
                .reshape(batch, channels, P, patches, patch_len)
                .mean(dim=-1)
                .permute(0, 1, 3, 2)
            )
            selected_error_bcq = candidate_error_bcqp.gather(
                dim=-1,
                index=selected_penalty_bcq.unsqueeze(-1),
            ).squeeze(-1)
            selected_gain_bcq = base_error_bcq - selected_error_bcq
            base_abs_error_bcq = (calibration_base_bch - y).abs().reshape(
                batch,
                channels,
                patches,
                patch_len,
            ).mean(dim=-1)
            candidate_abs_error_bcqp = (
                (candidate_bcpH - y.unsqueeze(2))
                .abs()
                .reshape(batch, channels, P, patches, patch_len)
                .mean(dim=-1)
                .permute(0, 1, 3, 2)
            )
            selected_abs_error_bcq = candidate_abs_error_bcqp.gather(
                dim=-1,
                index=selected_penalty_bcq.unsqueeze(-1),
            ).squeeze(-1)
            candidate_patch_bcqpr = candidate_bcpH.reshape(
                batch,
                channels,
                P,
                patches,
                patch_len,
            ).permute(0, 1, 3, 2, 4)
            selected_candidate_bcqr = candidate_patch_bcqpr.gather(
                dim=3,
                index=selected_penalty_bcq.unsqueeze(-1).unsqueeze(-1).expand(
                    -1,
                    -1,
                    -1,
                    1,
                    patch_len,
                ),
            ).squeeze(3)
            candidate_delta_bcqr = selected_candidate_bcqr - calibration_base_bch.reshape(
                batch,
                channels,
                patches,
                patch_len,
            )
            target_residual_bcqr = (y - calibration_base_bch).reshape(
                batch,
                channels,
                patches,
                patch_len,
            )
            cross_bcq = (candidate_delta_bcqr * target_residual_bcqr).mean(dim=-1)
            delta_sq_bcq = candidate_delta_bcqr.square().mean(dim=-1)
            query_bcq = query_start_abs_b.view(-1, 1, 1).expand(
                -1,
                channels,
                patches,
            )
            score_parts.append(selected_score_bcq.detach().cpu())
            gain_parts.append(selected_gain_bcq.detach().cpu())
            time_parts.append(query_bcq.detach().cpu())
            penalty_parts.append(selected_penalty_bcq.detach().cpu())
            base_mse_parts.append(base_error_bcq.detach().cpu())
            candidate_mse_parts.append(selected_error_bcq.detach().cpu())
            base_mae_parts.append(base_abs_error_bcq.detach().cpu())
            candidate_mae_parts.append(selected_abs_error_bcq.detach().cpu())
            if include_all_candidates:
                if candidate_score_bcqp is None or tuple(
                    candidate_score_bcqp.shape
                ) != tuple(candidate_error_bcqp.shape):
                    raise ValueError(
                        "all-candidate rerank requires utility scores with shape [B,C,Q,P]."
                    )
                candidate_mse_all_parts.append(candidate_error_bcqp.detach().cpu())
                candidate_mae_all_parts.append(
                    candidate_abs_error_bcqp.detach().cpu()
                )
                candidate_score_all_parts.append(candidate_score_bcqp.detach().cpu())
            regime_parts.append(_causal_patch_regime_descriptor(x).detach().cpu())
            cross_parts.append(cross_bcq.detach().cpu())
            delta_sq_parts.append(delta_sq_bcq.detach().cpu())
            scale_feature_parts.append(
                _causal_patch_scale_features(
                    x,
                    calibration_base_bch,
                    candidate_delta_bcqr,
                ).detach().cpu()
            )
            if include_patch_values:
                base_residual_patch_parts.append(
                    (calibration_base_bch - y)
                    .reshape(batch, channels, patches, patch_len)
                    .detach()
                    .cpu()
                )
                candidate_delta_patch_parts.append(
                    candidate_delta_bcqr.detach().cpu()
                )
        if not score_parts:
            empty_result = {
                "score": torch.empty(0),
                "gain": torch.empty(0),
                "time": torch.empty(0, dtype=torch.long),
                "penalty": torch.empty(0, dtype=torch.long),
                "base_mse": torch.empty(0),
                "candidate_mse": torch.empty(0),
                "base_mae": torch.empty(0),
                "candidate_mae": torch.empty(0),
                "candidate_mse_all": torch.empty(0),
                "candidate_mae_all": torch.empty(0),
                "candidate_score_all": torch.empty(0),
                "regime": torch.empty(0),
                "cross": torch.empty(0),
                "delta_sq": torch.empty(0),
                "base_residual_patch": torch.empty(0),
                "candidate_delta_patch": torch.empty(0),
                "scale_feature": torch.empty(0),
            }
            empty_result.update(
                {head_name: torch.empty(0) for head_name in head_score_parts}
            )
            return empty_result
        result = {
            "score": torch.cat(score_parts, dim=0),
            "gain": torch.cat(gain_parts, dim=0),
            "time": torch.cat(time_parts, dim=0),
            "penalty": torch.cat(penalty_parts, dim=0),
            "base_mse": torch.cat(base_mse_parts, dim=0),
            "candidate_mse": torch.cat(candidate_mse_parts, dim=0),
            "base_mae": torch.cat(base_mae_parts, dim=0),
            "candidate_mae": torch.cat(candidate_mae_parts, dim=0),
            "candidate_mse_all": (
                torch.cat(candidate_mse_all_parts, dim=0)
                if candidate_mse_all_parts
                else torch.empty(0)
            ),
            "candidate_mae_all": (
                torch.cat(candidate_mae_all_parts, dim=0)
                if candidate_mae_all_parts
                else torch.empty(0)
            ),
            "candidate_score_all": (
                torch.cat(candidate_score_all_parts, dim=0)
                if candidate_score_all_parts
                else torch.empty(0)
            ),
            "regime": torch.cat(regime_parts, dim=0),
            "cross": torch.cat(cross_parts, dim=0),
            "delta_sq": torch.cat(delta_sq_parts, dim=0),
            "base_residual_patch": (
                torch.cat(base_residual_patch_parts, dim=0)
                if base_residual_patch_parts
                else torch.empty(0)
            ),
            "candidate_delta_patch": (
                torch.cat(candidate_delta_patch_parts, dim=0)
                if candidate_delta_patch_parts
                else torch.empty(0)
            ),
            "scale_feature": torch.cat(scale_feature_parts, dim=0),
        }
        result.update(
            {
                head_name: (
                    torch.cat(parts, dim=0) if parts else torch.empty(0)
                )
                for head_name, parts in head_score_parts.items()
            }
        )
        return result

    def compute_batch_terms(
        x: torch.Tensor,
        y: torch.Tensor,
        idx: torch.Tensor,
        base_lambda_kp: torch.Tensor,
        model_params: Optional[Dict[str, torch.Tensor]] = None,
        gate_params: Optional[Dict[str, torch.Tensor]] = None,
        pred_residual_params: Optional[Dict[str, torch.Tensor]] = None,
        dynamic_lambda_params: Optional[Dict[str, torch.Tensor]] = None,
        straight_through: bool = True,
        mae_objective_weight=0.0,
    ) -> Dict[str, torch.Tensor]:
        x_model = apply_train_stat_input_centering(
            x,
            query_start_abs_b=idx,
            stat_anchor_pc=model_train_stat_adapter_pc,
            cfg=model_train_stat_adapter_cfg,
        )
        yhat_base_raw = _module_call(model, model_params, x_model, cluster_id_c)
        yhat_base = apply_history_anchor_adapter(
            yhat_base_raw,
            base_pred_bch=yhat_base_raw,
            observed_history_tc=data_window_tc,
            query_start_abs_b=idx,
            input_len=L,
            cfg=history_anchor_cfg,
        )
        yhat_base = apply_train_stat_anchor_expert(
            yhat_base,
            base_pred_bch=yhat_base,
            x_bcl=x,
            query_start_abs_b=idx,
            input_len=L,
            stat_anchor_pc=model_train_stat_adapter_pc,
            cfg=model_train_stat_adapter_cfg,
        )
        fixed_expert_delta_bch = build_moe_output_anchor_fixed_expert_delta(
            yhat_base,
            x_bcl=x,
            query_start_abs_b=idx,
            input_len=L,
            moe_cfg=moe_cfg,
            moe_enable=moe_enable,
            observed_history_tc=data_window_tc,
            train_stat_anchor_pc=train_stat_anchor_pc,
            train_residual_anchor_phc=train_residual_anchor_phc,
            learnable_output_anchor=learnable_output_anchor,
            cluster_id_c=cluster_id_c,
        )
        routing_base_bch = (
            yhat_base
            if fixed_expert_delta_bch is None
            else yhat_base + float(periodic_anchor_expert_scale) * fixed_expert_delta_bch
        )
        gate_feat_bkf = _build_gate_routing_features(
            x, routing_base_bch, cluster_id_c, K, mode=gate_feature_mode
        )
        if dynamic_lambda is None:
            lambda_feat_bkf = gate_feat_bkf
            series_bkl = None
        else:
            lambda_feat_bkf = gate_feat_bkf
            if gate_feature_mode != "history":
                feat_bcf = extract_gate_features(x)
                lambda_feat_bkf = scatter_mean_bcf_to_bkf(feat_bcf, cluster_id_c, K)
            series_bkl = scatter_mean_bcl_to_bkl(x, cluster_id_c, K)
        probs_bkp = None
        skip_bk = None
        skip_prob_bk = None
        pred_out = None

        route_pen_bkp = _router_penalty_context_from_history(
            x_bcl=x,
            yhat_base_bch=routing_base_bch,
            penalty_names=penalty_names,
            penalty_fns=penalty_fns,
            penalty_scale=penalty_scale,
            cluster_id_c=cluster_id_c,
            K=K,
            router_mode=router_mode,
        )

        if moe_enable and P > 0:
            mask_bkp, probs_bkp, skip_bk, skip_prob_bk = _module_call(
                gate,
                gate_params,
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

        if pred_residual is not None and moe_enable and P > 0:
            pred_out = _module_call(
                pred_residual,
                pred_residual_params,
                x,
                yhat_base,
                cluster_id_c,
                mask_bkp,
                skip_bk=_pred_residual_training_skip_arg(
                    skip_bk=skip_bk,
                    allow_skip=allow_skip,
                    ignore_skip_during_training=pred_residual_ignore_skip_during_training,
                ),
                query_start_abs_b=idx,
                fixed_expert_delta_bch=fixed_expert_delta_bch,
            )
            _assert_pure_candidate_base(pred_out, yhat_base, stage="compute_batch_terms")
            yhat_residual_raw = pred_out["y_final"]
            yhat = yhat_residual_raw
        else:
            yhat_residual_raw = yhat_base
            yhat = yhat_base
        if output_anchor_train_with_eval:
            yhat = apply_moe_output_anchor_experts(
                yhat,
                base_pred_bch=yhat_base,
                x_bcl=x,
                query_start_abs_b=idx,
                input_len=L,
                moe_cfg=moe_cfg,
                moe_enable=moe_enable,
                observed_history_tc=data_window_tc,
                train_stat_anchor_pc=train_stat_anchor_pc,
                train_residual_anchor_phc=train_residual_anchor_phc,
                learnable_output_anchor=learnable_output_anchor,
                cluster_id_c=cluster_id_c,
            )

        err_bch = yhat - y
        abs_err_bch = err_bch.abs()
        mse_bc = err_bch.pow(2).mean(dim=-1)
        mae_bc = abs_err_bch.mean(dim=-1)
        mse_bk = scatter_mean_bc_to_bk(mse_bc, cluster_id_c, K)
        mae_bk = scatter_mean_bc_to_bk(mae_bc, cluster_id_c, K)
        if _mae_objective_weight_is_nonzero(mae_objective_weight):
            mae_objective_bc = _mae_objective_bc_from_abs(
                abs_err_bch,
                kind=mae_objective_kind,
                beta=mae_objective_beta,
            )
            mae_objective_bk = scatter_mean_bc_to_bk(mae_objective_bc, cluster_id_c, K)
        else:
            mae_objective_bk = torch.zeros_like(mse_bk)

        if P > 0:
            if pred_out is not None:
                yhat_for_penalty = yhat_base + (yhat - yhat_base).detach()
                if pred_residual_detach_routed_penalty_pred:
                    yhat_for_penalty = yhat_for_penalty.detach()
            else:
                yhat_for_penalty = yhat
            pen_bcp = []
            for name in penalty_names:
                pen_bcp.append(penalty_fns[name](yhat_for_penalty, y))
            pen_bcp = torch.stack(pen_bcp, dim=-1)
            pen_bcp = normalize_penalties(pen_bcp, scale=penalty_scale)
            pen_bkp = scatter_mean_bcf_to_bkf(pen_bcp, cluster_id_c, K)
        else:
            pen_bkp = route_pen_bkp

        if P > 0:
            lam_bkp = _compute_lambda_bkp(
                base_lambda_kp=base_lambda_kp,
                feat_bkf=lambda_feat_bkf,
                series_bkl=series_bkl,
                dynamic_lambda=dynamic_lambda,
                dynamic_lambda_params=dynamic_lambda_params,
                lambda_min_kp=lambda_min_kp,
            )
            penalty_loss_bk = _routed_penalty_loss(
                mask_bkp=mask_bkp,
                lam_bkp=lam_bkp,
                pen_bkp=pen_bkp,
                gate_route_on_penalty_only=gate_route_on_penalty_only,
            )
            penalty_loss_bk = _apply_skip_to_penalty_loss(
                penalty_loss_bk,
                skip_bk=skip_bk if allow_skip else None,
                skip_cost=skip_cost,
            )
        else:
            lam_bkp = pen_bkp
            penalty_loss_bk = torch.zeros_like(mse_bk)

        pred_loss_terms = _pred_residual_loss_terms(
            pred_out=pred_out,
            y_base=yhat_base,
            y_final=yhat_residual_raw,
            y=y,
            penalty_names=penalty_names,
            penalty_fns=penalty_fns,
            cluster_id_c=cluster_id_c,
            K=K,
            penalty_scale=penalty_scale,
            specialization_weight=pred_residual_specialization_weight,
            norm_weight=pred_residual_norm_weight,
            intervention_weight=pred_residual_intervention_weight,
        )
        candidate_supervision_loss_bk = None
        if pred_residual_candidate_supervision_weight > 0.0:
            candidate_supervision_loss_bk = _pred_residual_candidate_supervision_loss(
                y_base_bch=yhat_base,
                pred_out=pred_out,
                y_bch=y,
                cluster_id_c=cluster_id_c,
                K=K,
                penalty_names=penalty_names,
                penalty_fns=penalty_fns,
                penalty_scale=penalty_scale,
                penalty_need_threshold=penalty_need_threshold,
                need_patch_len=pred_residual_candidate_supervision_need_patch_len,
                level_need_positive_weight=(
                    pred_residual_semantic_level_need_positive_weight
                ),
                noop_weight=pred_residual_candidate_supervision_noop_weight,
                high_mse_relative_tolerance=(
                    pred_residual_candidate_supervision_high_mse_relative_tolerance
                ),
                low_mse_relative_tolerance=(
                    pred_residual_candidate_supervision_low_mse_relative_tolerance
                ),
                low_high_rms_ratio_max=(
                    pred_residual_candidate_supervision_low_high_rms_ratio_max
                ),
                constraint_weight=(
                    pred_residual_candidate_supervision_constraint_weight
                ),
                constraint_eps=pred_residual_candidate_supervision_constraint_eps,
                allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                only_allowed=pred_residual_candidate_supervision_only_allowed,
                loss_kind=pred_residual_candidate_supervision_loss,
                forecast_mse_weight=(
                    pred_residual_candidate_supervision_forecast_mse_weight
                ),
                min_abs_improvement=pred_residual_candidate_supervision_min_abs,
                min_rel_improvement=pred_residual_candidate_supervision_min_rel,
                include_intervention=pred_residual_candidate_supervision_include_intervention,
                include_selector=pred_residual_candidate_supervision_include_selector,
                include_patch_route=pred_residual_candidate_supervision_include_patch_route,
                apply_output_anchors=output_anchor_train_with_eval,
                x_bcl=x,
                query_start_abs_b=idx,
                input_len=L,
                moe_cfg=moe_cfg,
                moe_enable=moe_enable,
                observed_history_tc=data_window_tc,
                train_stat_anchor_pc=train_stat_anchor_pc,
                train_residual_anchor_phc=train_residual_anchor_phc,
                learnable_output_anchor=learnable_output_anchor,
            )
        intervention_supervision_loss_bk = None
        if pred_residual_intervention_supervision_weight > 0.0:
            intervention_supervision_loss_bk = _pred_residual_intervention_supervision_loss(
                y_base_bch=yhat_base,
                pred_out=pred_out,
                y_bch=y,
                cluster_id_c=cluster_id_c,
                K=K,
                allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                only_allowed=pred_residual_intervention_supervision_only_allowed,
                min_gain=pred_residual_intervention_supervision_min_gain,
                pos_weight=pred_residual_intervention_supervision_pos_weight,
                apply_output_anchors=output_anchor_train_with_eval,
                x_bcl=x,
                query_start_abs_b=idx,
                input_len=L,
                moe_cfg=moe_cfg,
                moe_enable=moe_enable,
                observed_history_tc=data_window_tc,
                train_stat_anchor_pc=train_stat_anchor_pc,
                train_residual_anchor_phc=train_residual_anchor_phc,
                learnable_output_anchor=learnable_output_anchor,
            )
        loss_terms_bk, _ = _normalize_loss_terms(
            {
                "mse": mse_bk,
                "mae_objective": mae_objective_bk,
                "penalty": penalty_loss_bk,
                "pred_residual": pred_loss_terms["total_bk"],
            },
            loss_normalization_cfg,
        )
        objective_loss_bk = (
            (mse_weight * loss_terms_bk["mse"])
            + _apply_mae_objective_weight(loss_terms_bk["mae_objective"], mae_objective_weight)
            + loss_terms_bk["penalty"]
            + loss_terms_bk["pred_residual"]
        )
        if candidate_supervision_loss_bk is not None:
            objective_loss_bk = (
                objective_loss_bk
                + pred_residual_candidate_supervision_weight * candidate_supervision_loss_bk
            )
        if intervention_supervision_loss_bk is not None:
            objective_loss_bk = (
                objective_loss_bk
                + pred_residual_intervention_supervision_weight * intervention_supervision_loss_bk
            )
        utility_base_bch = None
        utility_cand_bcpH = None
        if (
            route_ce_weight > 0.0
            or binary_adoption_weight > 0.0
            or route_rate_alignment_weight > 0.0
            or route_positive_recall_weight > 0.0
            or route_precision_recall_weight > 0.0
            or mse_utility_gate_weight > 0.0
        ):
            utility_base_bch, utility_cand_bcpH = _pred_residual_candidates_on_eval_path(
                yhat_base,
                pred_out,
                apply_output_anchors=output_anchor_train_with_eval,
                x_bcl=x,
                query_start_abs_b=idx,
                input_len=L,
                moe_cfg=moe_cfg,
                moe_enable=moe_enable,
                observed_history_tc=data_window_tc,
                train_stat_anchor_pc=train_stat_anchor_pc,
                train_residual_anchor_phc=train_residual_anchor_phc,
                learnable_output_anchor=learnable_output_anchor,
                cluster_id_c=cluster_id_c,
            )
        if route_ce_weight > 0.0 and utility_cand_bcpH is not None:
            route_labels_bk, route_gain_bk = _cluster_route_oracle_labels_and_gain_from_candidates(
                base_bch=utility_base_bch,
                cand_bcpH=utility_cand_bcpH,
                y_bch=y,
                cluster_id_c=cluster_id_c,
                K=K,
                allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                min_abs_improvement=route_ce_min_abs_improvement,
                min_rel_improvement=route_ce_min_rel_improvement,
                min_candidate_delta_rms=route_ce_min_candidate_delta_rms,
            )
            route_ce_active_mask_bk = None
            if route_ce_ignore_abs_gain_below > 0.0:
                route_ce_active_mask_bk = _route_ce_active_mask_from_gain(
                    route_gain_bk,
                    ignore_abs_gain_below=route_ce_ignore_abs_gain_below,
                )
            route_ce_loss_bk = _route_ce_loss_from_probs(
                probs_bkp=probs_bkp,
                skip_prob_bk=skip_prob_bk if allow_skip else None,
                labels_bk=route_labels_bk,
                probs_include_skip_mass=bool(skip_competes),
                class_weight_q=_route_ce_class_weight_from_labels(
                    labels_bk=route_labels_bk,
                    num_classes=P + 1,
                    mode=route_ce_class_weight_mode,
                    max_weight=route_ce_max_class_weight,
                    active_mask_bk=route_ce_active_mask_bk,
                ),
            )
            if route_ce_active_mask_bk is not None:
                route_ce_loss_bk = route_ce_loss_bk * route_ce_active_mask_bk.to(dtype=route_ce_loss_bk.dtype)
            objective_loss_bk = objective_loss_bk + route_ce_weight * route_ce_loss_bk
        if binary_adoption_weight > 0.0 and utility_cand_bcpH is not None:
            binary_labels_bk, binary_gain_bk = _cluster_route_oracle_labels_and_gain_from_candidates(
                base_bch=utility_base_bch,
                cand_bcpH=utility_cand_bcpH,
                y_bch=y,
                cluster_id_c=cluster_id_c,
                K=K,
                allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                min_abs_improvement=binary_adoption_min_abs_improvement,
                min_rel_improvement=binary_adoption_min_rel_improvement,
                min_candidate_delta_rms=binary_adoption_min_candidate_delta_rms,
            )
            binary_active_mask_bk = None
            if binary_adoption_ignore_abs_gain_below > 0.0:
                binary_active_mask_bk = _route_ce_active_mask_from_gain(
                    binary_gain_bk,
                    ignore_abs_gain_below=binary_adoption_ignore_abs_gain_below,
                )
            binary_loss_bk = _route_binary_adoption_loss_from_probs(
                probs_bkp=probs_bkp,
                skip_prob_bk=skip_prob_bk if allow_skip else None,
                labels_bk=binary_labels_bk,
                probs_include_skip_mass=bool(skip_competes),
                allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                active_mask_bk=binary_active_mask_bk,
                positive_weight=binary_adoption_positive_weight,
                negative_weight=binary_adoption_negative_weight,
            )
            if binary_loss_bk is not None:
                objective_loss_bk = objective_loss_bk + binary_adoption_weight * binary_loss_bk
        if route_rate_alignment_weight > 0.0 and utility_cand_bcpH is not None:
            rate_labels_bk, rate_gain_bk = _cluster_route_oracle_labels_and_gain_from_candidates(
                base_bch=utility_base_bch,
                cand_bcpH=utility_cand_bcpH,
                y_bch=y,
                cluster_id_c=cluster_id_c,
                K=K,
                allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                min_abs_improvement=route_rate_alignment_min_abs_improvement,
                min_rel_improvement=route_rate_alignment_min_rel_improvement,
                min_candidate_delta_rms=route_rate_alignment_min_candidate_delta_rms,
            )
            rate_active_mask_bk = None
            if route_rate_alignment_ignore_abs_gain_below > 0.0:
                rate_active_mask_bk = _route_ce_active_mask_from_gain(
                    rate_gain_bk,
                    ignore_abs_gain_below=route_rate_alignment_ignore_abs_gain_below,
                )
            rate_loss_bk = _route_rate_alignment_loss_from_probs(
                probs_bkp=probs_bkp,
                skip_prob_bk=skip_prob_bk if allow_skip else None,
                labels_bk=rate_labels_bk,
                probs_include_skip_mass=bool(skip_competes),
                allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                active_mask_bk=rate_active_mask_bk,
            )
            objective_loss_bk = objective_loss_bk + route_rate_alignment_weight * rate_loss_bk
        if route_positive_recall_weight > 0.0 and utility_cand_bcpH is not None:
            recall_labels_bk, recall_gain_bk = _cluster_route_oracle_labels_and_gain_from_candidates(
                base_bch=utility_base_bch,
                cand_bcpH=utility_cand_bcpH,
                y_bch=y,
                cluster_id_c=cluster_id_c,
                K=K,
                allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                min_abs_improvement=route_positive_recall_min_abs_improvement,
                min_rel_improvement=route_positive_recall_min_rel_improvement,
                min_candidate_delta_rms=route_positive_recall_min_candidate_delta_rms,
            )
            recall_active_mask_bk = None
            if route_positive_recall_ignore_abs_gain_below > 0.0:
                recall_active_mask_bk = _route_ce_active_mask_from_gain(
                    recall_gain_bk,
                    ignore_abs_gain_below=route_positive_recall_ignore_abs_gain_below,
                )
            recall_loss_bk = _route_positive_recall_loss_from_probs(
                probs_bkp=probs_bkp,
                skip_prob_bk=skip_prob_bk if allow_skip else None,
                labels_bk=recall_labels_bk,
                probs_include_skip_mass=bool(skip_competes),
                active_mask_bk=recall_active_mask_bk,
                mode=route_positive_recall_mode,
                target_probability=route_positive_recall_target_probability,
            )
            objective_loss_bk = objective_loss_bk + route_positive_recall_weight * recall_loss_bk
        if route_precision_recall_weight > 0.0 and utility_cand_bcpH is not None:
            precision_labels_bk, precision_gain_bk = _cluster_route_oracle_labels_and_gain_from_candidates(
                base_bch=utility_base_bch,
                cand_bcpH=utility_cand_bcpH,
                y_bch=y,
                cluster_id_c=cluster_id_c,
                K=K,
                allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                min_abs_improvement=route_precision_recall_min_abs_improvement,
                min_rel_improvement=route_precision_recall_min_rel_improvement,
                min_candidate_delta_rms=route_precision_recall_min_candidate_delta_rms,
            )
            precision_active_mask_bk = None
            if route_precision_recall_ignore_abs_gain_below > 0.0:
                precision_active_mask_bk = _route_ce_active_mask_from_gain(
                    precision_gain_bk,
                    ignore_abs_gain_below=route_precision_recall_ignore_abs_gain_below,
                )
            precision_loss_bk = _route_precision_constrained_recall_loss_from_probs(
                probs_bkp=probs_bkp,
                skip_prob_bk=skip_prob_bk if allow_skip else None,
                labels_bk=precision_labels_bk,
                probs_include_skip_mass=bool(skip_competes),
                active_mask_bk=precision_active_mask_bk,
                recall_mode=route_precision_recall_mode,
                recall_target_probability=route_precision_recall_target_probability,
                false_adopt_max_probability=route_precision_recall_false_adopt_max_probability,
                false_adopt_weight=route_precision_recall_false_adopt_weight,
            )
            objective_loss_bk = objective_loss_bk + route_precision_recall_weight * precision_loss_bk
        if mse_utility_gate_weight > 0.0:
            mse_gate_loss_bk = _mse_utility_gate_supervision_loss(
                probs_bkp=probs_bkp,
                skip_prob_bk=skip_prob_bk if allow_skip else None,
                allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                y_base_bch=yhat_base,
                pred_out=pred_out,
                y_bch=y,
                cluster_id_c=cluster_id_c,
                K=K,
                y_base_eval_bch=utility_base_bch,
                cand_eval_bcpH=utility_cand_bcpH,
                temperature=mse_utility_gate_temperature,
                min_gain=mse_utility_gate_min_gain,
                mae_weight=mse_utility_gate_mae_weight,
                target_power=mse_utility_gate_target_power,
                include_skip=mse_utility_gate_include_skip,
                probs_include_skip_mass=bool(skip_competes),
                target_mode=mse_utility_gate_target_mode,
            )
            if mse_gate_loss_bk is not None:
                objective_loss_bk = objective_loss_bk + mse_utility_gate_weight * mse_gate_loss_bk
        return {
            "mse_bk": mse_bk,
            "mae_bk": mae_bk,
            "mae_objective_bk": mae_objective_bk,
            "objective_loss_bk": objective_loss_bk,
            "pen_bkp": pen_bkp,
            "mask_bkp": mask_bkp,
            "probs_bkp": probs_bkp,
            "lam_bkp": lam_bkp,
            "skip_bk": skip_bk,
            "skip_prob_bk": skip_prob_bk,
            "candidate_supervision_loss_bk": candidate_supervision_loss_bk,
            "intervention_supervision_loss_bk": intervention_supervision_loss_bk,
        }

    def _store_anchor_scale_selection(
        anchor_cfg: dict,
        anchor_summary: Dict[str, object],
        scale_selection_cfg: dict,
        scales_c: torch.Tensor,
        scores_c: torch.Tensor,
        selection_count: int,
        *,
        source_split: str,
        score_key: str,
        default_metric: str,
        default_max_scale: float,
        default_steps: int,
        horizon_segments: int,
    ) -> None:
        if int(scales_c.ndim) == 2:
            alpha_values = [[float(v) for v in row] for row in scales_c.tolist()]
            anchor_cfg["alpha_by_channel_horizon"] = alpha_values
            anchor_cfg["alpha_horizon_segments"] = int(horizon_segments)
            alpha_key = "alpha_by_channel_horizon"
        else:
            alpha_values = [float(v) for v in scales_c.tolist()]
            anchor_cfg["alpha_by_channel"] = alpha_values
            alpha_key = "alpha_by_channel"
        score_payload = (
            [[float(v) for v in row] for row in scores_c.tolist()]
            if int(scores_c.ndim) == 2
            else [float(v) for v in scores_c.tolist()]
        )
        anchor_summary["scale_selection"] = {
            "enable": True,
            "source_split": source_split,
            "metric": str(scale_selection_cfg.get("metric", default_metric)),
            "max_scale": float(scale_selection_cfg.get("max_scale", default_max_scale)),
            "steps": int(scale_selection_cfg.get("steps", default_steps)),
            "horizon_segments": int(horizon_segments),
            "num_windows": int(selection_count),
            alpha_key: alpha_values,
            score_key: score_payload,
            "mean_alpha": float(scales_c.mean().item()) if int(scales_c.numel()) > 0 else 0.0,
        }

    def _prepare_train_anchors_for_pred_residual_training() -> None:
        nonlocal train_residual_anchor_phc
        if not output_anchor_train_with_eval:
            return
        if len(dtr) <= 0:
            return
        if bool(train_stat_anchor_cfg.get("enable", False)):
            stat_scale_cfg = train_stat_anchor_cfg.get("scale_selection", {}) or {}
            if bool(stat_scale_cfg.get("enable", False)) and train_stat_anchor_pc is not None:
                horizon_segments = int(stat_scale_cfg.get("horizon_segments", 1))
                scales_c, scores_c, selection_count = select_train_stat_anchor_scales_from_loader(
                    model=model,
                    loader=dl_tr_source,
                    cluster_id_c=cluster_id_c,
                    device=device,
                    history_anchor_cfg=history_anchor_cfg,
                    observed_history_tc=data_window_tc,
                    input_len=L,
                    eval_start=0,
                    stat_anchor_pc=train_stat_anchor_pc,
                    train_stat_anchor_cfg=train_stat_anchor_cfg,
                    metric=str(stat_scale_cfg.get("metric", "mse")),
                    max_scale=float(stat_scale_cfg.get("max_scale", 0.3)),
                    steps=int(stat_scale_cfg.get("steps", 13)),
                    horizon_segments=horizon_segments,
                    model_train_stat_adapter_pc=model_train_stat_adapter_pc,
                    model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
                )
                _store_anchor_scale_selection(
                    train_stat_anchor_cfg,
                    train_stat_anchor_summary,
                    stat_scale_cfg,
                    scales_c,
                    scores_c,
                    selection_count,
                    source_split="train_pretrain_for_pred_residual",
                    score_key="score",
                    default_metric="mse",
                    default_max_scale=0.3,
                    default_steps=13,
                    horizon_segments=horizon_segments,
                )
                print(
                    "Preselected train-stat anchor scales for pred residual training: "
                    "source=train, "
                    f"mean_alpha={train_stat_anchor_summary['scale_selection']['mean_alpha']:.4f}"
                )
        if bool(train_residual_anchor_cfg.get("enable", False)):
            train_residual_anchor_period = int(train_residual_anchor_cfg.get("period", 96))
            train_residual_anchor_phc, train_residual_anchor_counts, residual_train_count = (
                build_train_residual_anchor_table_from_loader(
                    model=model,
                    loader=dl_tr_source,
                    cluster_id_c=cluster_id_c,
                    device=device,
                    history_anchor_cfg=history_anchor_cfg,
                    observed_history_tc=data_window_tc,
                    input_len=L,
                    eval_start=0,
                    period=train_residual_anchor_period,
                    model_train_stat_adapter_pc=model_train_stat_adapter_pc,
                    model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
                    train_stat_anchor_pc=train_stat_anchor_pc,
                    train_stat_anchor_cfg=train_stat_anchor_cfg,
                )
            )
            train_residual_anchor_summary.update(
                {
                    "period": int(train_residual_anchor_period),
                    "source_split": "train_pretrain_for_pred_residual",
                    "train_windows": int(residual_train_count),
                    "min_count": int(train_residual_anchor_counts.min().item()),
                    "max_count": int(train_residual_anchor_counts.max().item()),
                    "alpha": float(train_residual_anchor_cfg.get("alpha", 0.0) or 0.0),
                    "blend_target": str(train_residual_anchor_cfg.get("blend_target", "prediction")),
                }
            )
            residual_scale_cfg = train_residual_anchor_cfg.get("scale_selection", {}) or {}
            if bool(residual_scale_cfg.get("enable", False)):
                horizon_segments = int(residual_scale_cfg.get("horizon_segments", 1))
                scales_c, scores_c, selection_count = select_train_residual_anchor_scales_from_loader(
                    model=model,
                    loader=dl_tr_source,
                    cluster_id_c=cluster_id_c,
                    device=device,
                    history_anchor_cfg=history_anchor_cfg,
                    observed_history_tc=data_window_tc,
                    input_len=L,
                    eval_start=0,
                    residual_anchor_phc=train_residual_anchor_phc,
                    train_residual_anchor_cfg=train_residual_anchor_cfg,
                    metric=str(residual_scale_cfg.get("metric", "mse")),
                    max_scale=float(residual_scale_cfg.get("max_scale", 0.5)),
                    steps=int(residual_scale_cfg.get("steps", 21)),
                    horizon_segments=horizon_segments,
                    model_train_stat_adapter_pc=model_train_stat_adapter_pc,
                    model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
                    train_stat_anchor_pc=train_stat_anchor_pc,
                    train_stat_anchor_cfg=train_stat_anchor_cfg,
                )
                _store_anchor_scale_selection(
                    train_residual_anchor_cfg,
                    train_residual_anchor_summary,
                    residual_scale_cfg,
                    scales_c,
                    scores_c,
                    selection_count,
                    source_split="train_pretrain_for_pred_residual",
                    score_key="score_by_channel",
                    default_metric="mse",
                    default_max_scale=0.5,
                    default_steps=21,
                    horizon_segments=horizon_segments,
                )
                print(
                    "Preselected train-residual anchor scales for pred residual training: "
                    "source=train, "
                    f"mean_alpha={train_residual_anchor_summary['scale_selection']['mean_alpha']:.4f}"
                )

    def _prepare_phase_residual_candidate_for_pred_residual() -> None:
        if not phase_residual_candidate_enable or pred_residual is None:
            return
        table_phc, counts_p, train_windows = build_train_residual_anchor_table_from_loader(
            model=model,
            loader=dl_tr_source,
            cluster_id_c=cluster_id_c,
            device=device,
            history_anchor_cfg=history_anchor_cfg,
            observed_history_tc=data_window_tc,
            input_len=L,
            eval_start=0,
            period=int(phase_residual_candidate_period),
            model_train_stat_adapter_pc=model_train_stat_adapter_pc,
            model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
            train_stat_anchor_pc=train_stat_anchor_pc,
            train_stat_anchor_cfg=train_stat_anchor_cfg,
        )
        pred_residual.set_phase_residual_candidate_table(table_phc)
        phase_residual_candidate_summary.update(
            {
                "enable": True,
                "source_split": "train",
                "train_windows": int(train_windows),
                "min_count": int(counts_p.min().item()),
                "max_count": int(counts_p.max().item()),
                "table_shape": [int(v) for v in table_phc.shape],
                "train_only": True,
                "output_anchor_enabled": False,
            }
        )
        print(
            "Prediction residual phase candidate table built: "
            f"names={phase_residual_candidate_names}, period={int(phase_residual_candidate_period)}, "
            f"train_windows={int(train_windows)}, "
            f"min_count={int(counts_p.min().item())}, max_count={int(counts_p.max().item())}"
        )

    _prepare_train_anchors_for_pred_residual_training()
    _prepare_phase_residual_candidate_for_pred_residual()

    outer_train_state = [None]
    outer_val_state = [None]

    def next_outer_batch(loader: DataLoader, iterator_state):
        iterator = iterator_state[0]
        if iterator is None:
            iterator = iter(loader)
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        iterator_state[0] = iterator
        return batch

    inner_named = []
    inner_modules = []
    if not freeze_backbone:
        inner_modules.append(("model", model))
    if not (bilevel_enable and bilevel_optimize_gate):
        inner_modules.append(("gate", gate))
    if pred_residual is not None:
        inner_modules.append(("pred_residual", pred_residual))
    for prefix, module in inner_modules:
        for name, param in module.named_parameters():
            inner_named.append((prefix, name, param))

    def bilevel_outer_step(epoch_idx: int, warmup_scale: float) -> Optional[float]:
        if (not bilevel_enable) or lambda_optimizer is None or stopped.all():
            return None

        train_batch = next_outer_batch(dl_tr, outer_train_state)
        val_batch = next_outer_batch(dl_va, outer_val_state)
        x_tr, y_tr, idx_tr = train_batch
        x_va, y_va, idx_va = val_batch
        x_tr = x_tr.to(device, non_blocking=True)
        y_tr = y_tr.to(device, non_blocking=True)
        idx_tr = idx_tr.to(device=device, dtype=torch.long, non_blocking=True)
        x_va = x_va.to(device, non_blocking=True)
        y_va = y_va.to(device, non_blocking=True)
        idx_va = idx_va.to(device=device, dtype=torch.long, non_blocking=True)

        base_lambda_kp = lambda_kp_at(epoch_idx, detach=False) * warmup_scale
        train_terms = compute_batch_terms(
            x_tr, y_tr, idx_tr,
            base_lambda_kp=base_lambda_kp,
            straight_through=(not bool(moe_cfg["detach_penalty_grad"])),
            mae_objective_weight=mae_objective_weight_at(epoch_idx),
        )
        inner_loss_bk = train_terms["objective_loss_bk"]
        if moe_enable and P > 0 and (gate_entropy_weight != 0.0 or gate_balance_weight != 0.0):
            inner_loss_bk = inner_loss_bk + _gate_regularization(
                train_terms["probs_bkp"],
                gate_entropy_weight=gate_entropy_weight,
                gate_balance_weight=gate_balance_weight,
                gate_entropy_target_frac=gate_entropy_target_frac,
                gate_balance_target_kp=gate_balance_target_kp,
            )
        train_weight_k = _training_cluster_weight(
            cluster_weight_k,
            stopped,
            shared_moe=shared_moe_across_clusters,
        )
        inner_loss = reduce_cluster_metric(inner_loss_bk, train_weight_k).mean()

        inner_params = [param for _, _, param in inner_named]
        inner_grads = torch.autograd.grad(
            inner_loss,
            inner_params,
            create_graph=True,
            allow_unused=True,
        )
        fast_model_params = {}
        fast_gate_params = {}
        fast_pred_residual_params = {}
        for (prefix, name, param), grad in zip(inner_named, inner_grads):
            fast_param = param if grad is None else (param - bilevel_inner_lr * grad)
            if prefix == "model":
                fast_model_params[name] = fast_param
            elif prefix == "gate":
                fast_gate_params[name] = fast_param
            else:
                fast_pred_residual_params[name] = fast_param

        model_was_training = model.training
        gate_was_training = gate.training
        pred_residual_was_training = pred_residual.training if pred_residual is not None else False
        dyn_was_training = dynamic_lambda.training if dynamic_lambda is not None else False
        model.eval()
        gate.eval()
        if pred_residual is not None:
            pred_residual.eval()
        if dynamic_lambda is not None:
            dynamic_lambda.eval()
        val_terms = compute_batch_terms(
            x_va, y_va, idx_va,
            base_lambda_kp=base_lambda_kp,
            model_params=fast_model_params,
            gate_params=fast_gate_params if len(fast_gate_params) > 0 else None,
            pred_residual_params=fast_pred_residual_params if len(fast_pred_residual_params) > 0 else None,
            straight_through=False,
            mae_objective_weight=mae_objective_weight_at(epoch_idx),
        )
        if model_was_training:
            model.train()
        if gate_was_training:
            gate.train()
        if pred_residual is not None and pred_residual_was_training:
            pred_residual.train()
        if dynamic_lambda is not None and dyn_was_training:
            dynamic_lambda.train()
        outer_metric_bk = val_terms["mse_bk"]
        outer_loss = reduce_cluster_metric(outer_metric_bk, cluster_weight_k).mean()
        if learnable_lambda is not None and learnable_lambda_reg_weight > 0.0:
            outer_loss = outer_loss + learnable_lambda_reg_weight * reduce_cluster_metric(
                learnable_lambda.regularization(), cluster_weight_k
            )
        if dynamic_lambda is not None and dynamic_lambda_reg_weight > 0.0 and P > 0:
            base_lam = base_lambda_kp.unsqueeze(0).expand(x_va.shape[0], K, P).clamp_min(1.0e-8)
            scale_bkp = val_terms["lam_bkp"] / base_lam
            outer_loss = outer_loss + dynamic_lambda_reg_weight * scale_bkp.log().pow(2).mean()

        lambda_optimizer.zero_grad(set_to_none=True)
        outer_loss.backward()
        if grad_clip > 0:
            lambda_params = []
            if dynamic_lambda is not None:
                lambda_params.extend(list(dynamic_lambda.parameters()))
            if learnable_lambda is not None:
                lambda_params.extend(list(learnable_lambda.parameters()))
            if len(lambda_params) > 0:
                torch.nn.utils.clip_grad_norm_(lambda_params, grad_clip)
        if dynamic_lambda is not None:
            dynamic_lambda.mask_cluster_grads(stopped)
        if learnable_lambda is not None:
            learnable_lambda.mask_cluster_grads(stopped)
        lambda_optimizer.step()
        return float(outer_loss.item())
    # keep console output minimal during training

    # training
    grad_clip = float(cfg["train"]["grad_clip"])
    steps_per_epoch = max(len(dl_tr), 1)
    train_label = f"Train {os.path.splitext(os.path.basename(cfg['data']['csv_path']))[0]} H={H}"
    train_progress = PurpleProgressBar(
        total=max(int(epochs) * steps_per_epoch, 1),
        label=train_label,
        unit="batch",
    )
    early_stopped = False
    mse_gate_train_diag_history: List[Dict[str, object]] = []
    stage2_loss_audit_history: List[Dict[str, object]] = []
    stage2_objective_overlap_batches: List[Dict[str, object]] = []
    stage2_route_audit_history: List[Dict[str, object]] = []
    overfit_diagnostic_history: List[Dict[str, object]] = []
    overfit_diagnostic_metric_epochs: List[int] = []
    semantic_bank_independent_history: List[Dict[str, object]] = []
    semantic_bank_best_by_penalty: List[Dict[str, object]] = (
        _new_semantic_bank_best_candidates(penalty_names)
    )
    for loaded_row in semantic_bank_loaded_accepted_rows:
        loaded_name = str(loaded_row["penalty"])
        loaded_index = penalty_names.index(loaded_name)
        semantic_bank_best_by_penalty[loaded_index] = loaded_row
    if overfit_diagnostic_range is not None:
        configured_metric_epochs = overfit_diagnostic_cfg.get(
            "metric_epochs",
            [1, 5, 10, 20, int(epochs)],
        )
        overfit_diagnostic_metric_epochs = sorted(
            {
                int(epoch_idx)
                for epoch_idx in configured_metric_epochs
                if 1 <= int(epoch_idx) <= int(epochs)
            }
            | {int(epochs)}
        )
    stage2_route_audit_frequency = max(1, int(stage2_route_audit_cfg.get("frequency_epochs", 1)))

    if patch_router_epoch0_noop_enable:
        if len(dva) <= 0:
            raise ValueError(
                "patch_router.epoch0_noop_selection requires a validation split."
            )
        if monitor_metric not in {"val_mse", "val_mae", "val_loss"}:
            raise ValueError(
                "patch_router.epoch0_noop_selection requires a validation selection metric."
            )
        (
            epoch0_val_loss_k,
            epoch0_val_mse_k,
            epoch0_val_mae_k,
            _,
            _,
            _,
            _,
            _,
        ) = eval_loop_with_history(
            model,
            gate,
            lambda_kp_at(1, detach=True),
            penalty_names,
            penalty_fns,
            dl_stage2_epoch_eval if dl_stage2_epoch_eval is not None else dl_va,
            cluster_id_c,
            K,
            moe_cfg,
            device,
            select_ranks=select_ranks,
            collect_plot=False,
            channel_count=C,
            collect_samples=False,
            mse_weight=mse_weight,
            gate_entropy_weight=gate_entropy_weight,
            gate_balance_weight=gate_balance_weight,
            gate_soft_weight=gate_soft_weight,
            gate_entropy_target_frac=gate_entropy_target_frac,
            penalty_scale=penalty_scale,
            dynamic_lambda=dynamic_lambda,
            lambda_min_kp=lambda_min_kp,
            mae_objective_weight=mae_objective_weight_at(1),
            mae_objective_kind=mae_objective_kind,
            mae_objective_beta=mae_objective_beta,
            pred_residual=pred_residual,
            eval_start=stage2_epoch_eval_start,
        )
        patch_router_epoch0_val_mse_k = epoch0_val_mse_k.detach().clone()
        patch_router_epoch0_val_mae_k = epoch0_val_mae_k.detach().clone()
        epoch0_monitor_k = _select_monitor_k(
            epoch0_val_loss_k,
            epoch0_val_mse_k,
            epoch0_val_mae_k,
            epoch0_val_loss_k,
            epoch0_val_mse_k,
            epoch0_val_mae_k,
        )
        best_monitor.copy_(epoch0_monitor_k)
        for k in range(K):
            save_best(k, 0)
        if shared_moe_across_clusters:
            shared_moe_best_monitor = float(
                reduce_cluster_metric(epoch0_monitor_k, cluster_weight_k).item()
            )
            shared_moe_best_epoch = 0
            shared_moe_best_state["gate"] = gate.get_cluster_state(0)
            shared_moe_best_state["pred_residual"] = (
                pred_residual.get_cluster_state(0)
                if pred_residual is not None
                else None
            )
        patch_router_epoch0_noop_summary = {
            "enable": True,
            "require_dual_improvement": bool(
                patch_router_epoch0_require_dual
            ),
            "val_mse": float(
                reduce_cluster_metric(
                    epoch0_val_mse_k,
                    cluster_weight_k,
                ).item()
            ),
            "val_mae": float(
                reduce_cluster_metric(
                    epoch0_val_mae_k,
                    cluster_weight_k,
                ).item()
            ),
        }
        expected_epoch0_mse = patch_router_epoch0_noop_cfg.get(
            "expected_val_mse",
            None,
        )
        expected_epoch0_mae = patch_router_epoch0_noop_cfg.get(
            "expected_val_mae",
            None,
        )
        epoch0_atol = max(
            0.0,
            float(patch_router_epoch0_noop_cfg.get("expected_atol", 5.0e-6)),
        )
        for metric_name, expected_value in (
            ("val_mse", expected_epoch0_mse),
            ("val_mae", expected_epoch0_mae),
        ):
            if expected_value is None:
                continue
            actual_value = float(patch_router_epoch0_noop_summary[metric_name])
            if abs(actual_value - float(expected_value)) > epoch0_atol:
                raise ValueError(
                    "patch-router epoch-0 no-op baseline mismatch: "
                    f"{metric_name}={actual_value:.9f}, "
                    f"expected={float(expected_value):.9f}, "
                    f"atol={epoch0_atol:.3g}."
                )
        patch_router_epoch0_noop_summary.update(
            {
                "expected_val_mse": (
                    None
                    if expected_epoch0_mse is None
                    else float(expected_epoch0_mse)
                ),
                "expected_val_mae": (
                    None
                    if expected_epoch0_mae is None
                    else float(expected_epoch0_mae)
                ),
                "expected_atol": float(epoch0_atol),
                "baseline_match": True,
            }
        )
        print(
            "Patch-router epoch-0 no-op candidate: "
            f"val_MSE={patch_router_epoch0_noop_summary['val_mse']:.6f}, "
            f"val_MAE={patch_router_epoch0_noop_summary['val_mae']:.6f}, "
            f"require_dual={patch_router_epoch0_require_dual}"
        )

    for ep in range(1, epochs + 1):
        t_ep0 = time.perf_counter()
        semantic_loss_sum_p = torch.zeros(P, dtype=torch.float64)
        semantic_grad_raw_sum_p = torch.zeros(P, dtype=torch.float64)
        semantic_grad_clipped_sum_p = torch.zeros(P, dtype=torch.float64)
        semantic_update_sum_p = torch.zeros(P, dtype=torch.float64)
        semantic_component_names = (
            (
                "level_amplitude_loss",
                "level_need_balanced_bce",
            )
            if pred_residual_candidate_supervision_loss
            in {
                "level_residual_separate_gate",
                "level_residual_high_need_separate_gate",
            }
            else
            (
                "level_amplitude_loss",
                "level_need_bce",
                "level_executed_loss",
            )
            if pred_residual_candidate_supervision_loss == "level_residual_gate"
            else
            ("high_semantic_contribution",)
            if pred_residual_candidate_supervision_loss
            == "high_need_own_penalty"
            else
            (
                "high_semantic_contribution",
                "high_forecast_mse_contribution",
                "low_noop_contribution",
            )
            if pred_residual_candidate_supervision_loss
            == "need_weighted_own_penalty_mse"
            else (
                "high_semantic",
                "high_forecast_mse",
                "high_mse_violation",
                "low_mse_violation",
                "low_noop_violation",
                "base_guard_objective",
                "constraint_total",
                "constraint_active",
            )
        )
        semantic_component_sum_by_name_p = {
            key: torch.zeros(P, dtype=torch.float64)
            for key in semantic_component_names
        }
        semantic_step_count = 0
        semantic_batch_count = 0
        semantic_raw_gradient_pending_count = 0
        semantic_raw_gradient_group_counts: List[int] = []
        semantic_level_disjoint_metric_sums = {
            group: {
                "raw_gradient_l2": 0.0,
                "clipped_gradient_l2": 0.0,
                "parameter_update_l2": 0.0,
            }
            for group in semantic_level_disjoint_params_by_group
        }
        if (
            patch_router_freeze_experts_after_warmup
            and not patch_router_expert_freeze_applied
            and ep > patch_router_expert_warmup_epochs
        ):
            if pred_residual is None or getattr(pred_residual, "patch_router", None) is None:
                raise ValueError(
                    "patch_router.freeze_experts_after_warmup requires an enabled patch router."
                )
            patch_router_trainable_prefixes = (
                (
                    "patch_router.W_pairwise_rank",
                    "patch_router.b_pairwise_rank",
                )
                if patch_router_pairwise_freeze_other_parameters
                else ("patch_router.",)
            )
            patch_router_frozen_expert_params = _freeze_module_params_except_prefixes(
                pred_residual,
                patch_router_trainable_prefixes,
            )
            patch_router_expert_freeze_applied = True
            print(
                "Patch-router second stage froze shared residual experts: "
                f"epoch={ep}, params={patch_router_frozen_expert_params}"
            )
        if lr_warmup_epochs > 0 and ep <= lr_warmup_epochs:
            _set_optimizer_lr_scale(
                optimizers,
                _lr_warmup_scale(ep, lr_warmup_epochs, lr_warmup_start_factor),
            )
        if penalty_warmup_epochs > 0:
            warmup_scale = min(1.0, float(ep) / float(penalty_warmup_epochs))
        else:
            warmup_scale = 1.0
        patch_router_oracle_ce_weight_ep = (
            patch_router_oracle_ce_weight
            if ep > patch_router_oracle_ce_warmup_epochs
            else 0.0
        )
        patch_router_hierarchical_weight_ep = (
            patch_router_hierarchical_weight
            if ep > patch_router_hierarchical_warmup_epochs
            else 0.0
        )
        _set_module_train_mode(
            model,
            training=True,
            keep_frozen_eval=bool(freeze_backbone and frozen_backbone_eval_mode),
        )
        gate.train()
        if pred_residual is not None:
            pred_residual.train()
        if dynamic_lambda is not None:
            dynamic_lambda.train()
        running = 0.0
        n_batches = 0
        act_sum = torch.zeros(P, device=device)
        active_cnt = 0
        k_active_sum = 0.0
        train_loss_sum_k = torch.zeros(K, device=device)
        train_mse_sum_k = torch.zeros(K, device=device)
        train_mae_sum_k = torch.zeros(K, device=device)
        train_cnt = 0
        if stage2_loss_audit_enable:
            stage2_total_loss_sum_k = torch.zeros(K, device=device)
            stage2_forecast_loss_sum_k = torch.zeros(K, device=device)
            stage2_penalty_loss_sum_k = torch.zeros(K, device=device)
            stage2_pred_residual_aux_loss_sum_k = torch.zeros(K, device=device)
            stage2_candidate_supervision_loss_sum_k = torch.zeros(K, device=device)
            stage2_gate_utility_loss_sum_k = torch.zeros(K, device=device)
            stage2_skip_noop_loss_sum_k = torch.zeros(K, device=device)
            stage2_intervention_supervision_loss_sum_k = torch.zeros(K, device=device)
            stage2_other_aux_loss_sum_k = torch.zeros(K, device=device)
            stage2_route_prob_sum_kp = torch.zeros(K, P, device=device)
            stage2_route_actual_sum_kp = torch.zeros(K, P, device=device)
            stage2_route_entropy_sum_k = torch.zeros(K, device=device)
            stage2_route_count_k = torch.zeros(K, device=device)
            stage2_skip_prob_sum_k = torch.zeros(K, device=device)
            stage2_skip_active_sum_k = torch.zeros(K, device=device)
            stage2_grad_norm_sum = {
                "backbone": 0.0,
                "gate": 0.0,
                "pred_residual": 0.0,
                "dynamic_lambda": 0.0,
                "learnable_lambda": 0.0,
                "learnable_output_anchor": 0.0,
            }
            stage2_grad_norm_batches = 0
        mse_gate_loss_sum_k = torch.zeros(K, device=device)
        mse_gate_valid_sum_k = torch.zeros(K, device=device)
        mse_gate_skip_target_sum_k = torch.zeros(K, device=device)
        mse_gate_skip_prob_sum_k = torch.zeros(K, device=device)
        mse_gate_best_gain_sum_k = torch.zeros(K, device=device)
        mse_gate_diag_count_k = torch.zeros(K, device=device)
        act_sum_kp = torch.zeros(K, P, device=device)
        active_cnt_k = torch.zeros(K, device=device)
        rank_counts = None
        rank_total = 0
        if moe_enable and P > 0:
            rank_counts = torch.zeros(P, P, device=device)

        for x, y, idx in dl_tr:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            idx = idx.to(torch.long) + int(optimization_index_offset)
            if cluster_memory_bank is not None:
                train_window = torch.cat([x, y], dim=-1)
                cluster_memory_bank.update(train_window, idx, cluster_id_c)

            x_model = apply_train_stat_input_centering(
                x,
                query_start_abs_b=idx,
                stat_anchor_pc=model_train_stat_adapter_pc,
                cfg=model_train_stat_adapter_cfg,
            )
            yhat_base_raw = model(x_model, cluster_id_c)
            yhat_base = apply_history_anchor_adapter(
                yhat_base_raw,
                base_pred_bch=yhat_base_raw,
                observed_history_tc=data_window_tc,
                query_start_abs_b=idx,
                input_len=L,
                cfg=history_anchor_cfg,
            )
            yhat_base = apply_train_stat_anchor_expert(
                yhat_base,
                base_pred_bch=yhat_base,
                x_bcl=x,
                query_start_abs_b=idx,
                input_len=L,
                stat_anchor_pc=model_train_stat_adapter_pc,
                cfg=model_train_stat_adapter_cfg,
            )
            fixed_expert_delta_bch = build_moe_output_anchor_fixed_expert_delta(
                yhat_base,
                x_bcl=x,
                query_start_abs_b=idx,
                input_len=L,
                moe_cfg=moe_cfg,
                moe_enable=moe_enable,
                observed_history_tc=data_window_tc,
                train_stat_anchor_pc=train_stat_anchor_pc,
                train_residual_anchor_phc=train_residual_anchor_phc,
                learnable_output_anchor=learnable_output_anchor,
                cluster_id_c=cluster_id_c,
            )
            routing_base_bch = (
                yhat_base
                if fixed_expert_delta_bch is None
                else yhat_base + float(periodic_anchor_expert_scale) * fixed_expert_delta_bch
            )
            gate_feat_bkf = _build_gate_routing_features(
                x, routing_base_bch, cluster_id_c, K, mode=gate_feature_mode
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
            skip_bk = None
            pred_out = None
            hierarchical_terms = None
            objective_overlap_reference_bk = None
            route_pen_bkp = _router_penalty_context_from_history(
                x_bcl=x,
                yhat_base_bch=routing_base_bch,
                penalty_names=penalty_names,
                penalty_fns=penalty_fns,
                penalty_scale=penalty_scale,
                cluster_id_c=cluster_id_c,
                K=K,
                router_mode=router_mode,
            )

            if moe_enable and P > 0:
                straight_through = (not bool(moe_cfg["detach_penalty_grad"])) and (not (bilevel_enable and bilevel_optimize_gate))
                mask_bkp, probs_bkp, skip_bk, skip_prob_bk = gate(
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
                with torch.no_grad():
                    act_sum += mask_bkp.sum(dim=(0, 1))
                    active_cnt += int(mask_bkp.shape[0] * mask_bkp.shape[1])
                    k_active_sum += float(mask_bkp.sum().item())
                    act_sum_kp += mask_bkp.sum(dim=0)
                    active_cnt_k += mask_bkp.shape[0]
                if rank_counts is not None:
                    with torch.no_grad():
                        order = torch.argsort(probs_bkp.detach(), dim=-1, descending=True)
                        for r in range(P):
                            pen_idx = order[..., r].reshape(-1)
                            cnt = torch.bincount(pen_idx, minlength=P)
                            rank_counts[:, r] += cnt
                        rank_total += int(order.shape[0] * order.shape[1])
                if stage2_loss_audit_enable:
                    with torch.no_grad():
                        probs_det = probs_bkp.detach()
                        probs_safe = probs_det.clamp_min(1.0e-8)
                        stage2_route_prob_sum_kp += probs_det.sum(dim=0)
                        stage2_route_actual_sum_kp += mask_bkp.detach().sum(dim=0)
                        stage2_route_entropy_sum_k += (-(probs_safe * probs_safe.log()).sum(dim=-1)).sum(dim=0)
                        stage2_route_count_k += probs_det.shape[0]
                        if allow_skip and skip_prob_bk is not None and skip_bk is not None:
                            stage2_skip_prob_sum_k += skip_prob_bk.detach().sum(dim=0)
                            stage2_skip_active_sum_k += skip_bk.detach().sum(dim=0)
            else:
                mask_bkp = torch.zeros_like(route_pen_bkp)

            if pred_residual is not None and moe_enable and P > 0:
                pred_out = pred_residual(
                    x,
                    yhat_base,
                    cluster_id_c,
                    mask_bkp,
                    skip_bk=_pred_residual_training_skip_arg(
                        skip_bk=skip_bk,
                        allow_skip=allow_skip,
                        ignore_skip_during_training=pred_residual_ignore_skip_during_training,
                    ),
                    query_start_abs_b=idx,
                    fixed_expert_delta_bch=fixed_expert_delta_bch,
                )
                _assert_pure_candidate_base(pred_out, yhat_base, stage="train_epoch")
                yhat_residual_raw = pred_out["y_final"]
                yhat = yhat_residual_raw
            else:
                yhat_residual_raw = yhat_base
                yhat = yhat_base
            if output_anchor_train_with_eval:
                yhat = apply_moe_output_anchor_experts(
                    yhat,
                    base_pred_bch=yhat_base,
                    x_bcl=x,
                    query_start_abs_b=idx,
                    input_len=L,
                    moe_cfg=moe_cfg,
                    moe_enable=moe_enable,
                    observed_history_tc=data_window_tc,
                    train_stat_anchor_pc=train_stat_anchor_pc,
                    train_residual_anchor_phc=train_residual_anchor_phc,
                    learnable_output_anchor=learnable_output_anchor,
                    cluster_id_c=cluster_id_c,
                )

            err_bch = yhat - y
            abs_err_bch = err_bch.abs()
            mse_bc = err_bch.pow(2).mean(dim=-1)  # [B,C]
            mae_bc = abs_err_bch.mean(dim=-1)  # [B,C]
            mse_bk = scatter_mean_bc_to_bk(mse_bc, cluster_id_c, K)  # [B,K]
            mae_bk = scatter_mean_bc_to_bk(mae_bc, cluster_id_c, K)  # [B,K]
            mae_objective_weight_ep = mae_objective_weight_at(ep)
            if _mae_objective_weight_is_nonzero(mae_objective_weight_ep):
                mae_objective_bc = _mae_objective_bc_from_abs(
                    abs_err_bch,
                    kind=mae_objective_kind,
                    beta=mae_objective_beta,
                )
                mae_objective_bk = scatter_mean_bc_to_bk(mae_objective_bc, cluster_id_c, K)
            else:
                mae_objective_bk = torch.zeros_like(mse_bk)

            mse_gate_loss_bk = None
            mse_gate_diag = None
            if P > 0:
                if pred_out is not None:
                    yhat_for_penalty = yhat_base + (yhat - yhat_base).detach()
                    if pred_residual_detach_routed_penalty_pred:
                        yhat_for_penalty = yhat_for_penalty.detach()
                else:
                    yhat_for_penalty = yhat
                pen_bcp = []
                for name in penalty_names:
                    pen_bc = penalty_fns[name](yhat_for_penalty, y)  # [B,C]
                    pen_bcp.append(pen_bc)
                pen_bcp = torch.stack(pen_bcp, dim=-1)  # [B,C,P]
                pen_bcp = normalize_penalties(pen_bcp, scale=penalty_scale)
                pen_bkp = scatter_mean_bcf_to_bkf(pen_bcp, cluster_id_c, K)  # [B,K,P]
            else:
                pen_bkp = route_pen_bkp

            if P > 0:
                base_lambda_kp = lambda_kp_at(ep, detach=bilevel_enable) * warmup_scale
                dynamic_lambda_params = _named_param_dict(dynamic_lambda, detach=True) if (bilevel_enable and dynamic_lambda is not None) else None
                lam = _compute_lambda_bkp(
                    base_lambda_kp=base_lambda_kp,
                    feat_bkf=feat_bkf,
                    series_bkl=series_bkl,
                    dynamic_lambda=dynamic_lambda,
                    dynamic_lambda_params=dynamic_lambda_params,
                    lambda_min_kp=lambda_min_kp,
                )
                penalty_loss_bk = _routed_penalty_loss(
                    mask_bkp=mask_bkp,
                    lam_bkp=lam,
                    pen_bkp=pen_bkp,
                    gate_route_on_penalty_only=gate_route_on_penalty_only,
                )
                penalty_loss_bk = _apply_skip_to_penalty_loss(
                    penalty_loss_bk,
                    skip_bk=skip_bk if allow_skip else None,
                    skip_cost=skip_cost,
                )
                raw_objective_loss_bk = (
                    (mse_weight * mse_bk)
                    + _apply_mae_objective_weight(mae_objective_bk, mae_objective_weight_ep)
                    + penalty_loss_bk
                )  # [B,K]
                pred_loss_terms = _pred_residual_loss_terms(
                    pred_out=pred_out,
                    y_base=yhat_base,
                    y_final=yhat_residual_raw,
                    y=y,
                    penalty_names=penalty_names,
                    penalty_fns=penalty_fns,
                    cluster_id_c=cluster_id_c,
                    K=K,
                    penalty_scale=penalty_scale,
                    specialization_weight=pred_residual_specialization_weight,
                    norm_weight=pred_residual_norm_weight,
                    intervention_weight=pred_residual_intervention_weight,
                )
                candidate_supervision_loss_bk = None
                candidate_supervision_loss_bkp = None
                candidate_supervision_components_bkp = None
                if pred_residual_candidate_supervision_weight > 0.0:
                    candidate_supervision_raw = _pred_residual_candidate_supervision_loss(
                        y_base_bch=yhat_base,
                        pred_out=pred_out,
                        y_bch=y,
                        cluster_id_c=cluster_id_c,
                        K=K,
                        penalty_names=penalty_names,
                        penalty_fns=penalty_fns,
                        penalty_scale=penalty_scale,
                        penalty_need_threshold=penalty_need_threshold,
                        need_patch_len=pred_residual_candidate_supervision_need_patch_len,
                        level_need_positive_weight=(
                            pred_residual_semantic_level_need_positive_weight
                        ),
                        noop_weight=pred_residual_candidate_supervision_noop_weight,
                        high_mse_relative_tolerance=(
                            pred_residual_candidate_supervision_high_mse_relative_tolerance
                        ),
                        low_mse_relative_tolerance=(
                            pred_residual_candidate_supervision_low_mse_relative_tolerance
                        ),
                        low_high_rms_ratio_max=(
                            pred_residual_candidate_supervision_low_high_rms_ratio_max
                        ),
                        constraint_weight=(
                            pred_residual_candidate_supervision_constraint_weight
                        ),
                        constraint_eps=pred_residual_candidate_supervision_constraint_eps,
                        allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                        only_allowed=pred_residual_candidate_supervision_only_allowed,
                        return_per_penalty=bool(
                            pred_residual_semantic_bank_stage1
                            and pred_residual_candidate_supervision_independent_optimization
                        ),
                        return_components=bool(pred_residual_semantic_bank_stage1),
                        loss_kind=pred_residual_candidate_supervision_loss,
                        forecast_mse_weight=(
                            pred_residual_candidate_supervision_forecast_mse_weight
                        ),
                        min_abs_improvement=pred_residual_candidate_supervision_min_abs,
                        min_rel_improvement=pred_residual_candidate_supervision_min_rel,
                        include_intervention=pred_residual_candidate_supervision_include_intervention,
                        include_selector=pred_residual_candidate_supervision_include_selector,
                        include_patch_route=pred_residual_candidate_supervision_include_patch_route,
                        apply_output_anchors=output_anchor_train_with_eval,
                        x_bcl=x,
                        query_start_abs_b=idx,
                        input_len=L,
                        moe_cfg=moe_cfg,
                        moe_enable=moe_enable,
                        observed_history_tc=data_window_tc,
                        train_stat_anchor_pc=train_stat_anchor_pc,
                        train_residual_anchor_phc=train_residual_anchor_phc,
                        learnable_output_anchor=learnable_output_anchor,
                    )
                    if candidate_supervision_raw is not None:
                        if pred_residual_semantic_bank_stage1:
                            if not isinstance(candidate_supervision_raw, dict):
                                raise RuntimeError(
                                    "semantic bank independent loss must expose loss/components."
                                )
                            candidate_supervision_loss_bkp = candidate_supervision_raw.get(
                                "loss_bkp"
                            )
                            candidate_supervision_components_bkp = candidate_supervision_raw.get(
                                "components_bkp"
                            )
                            if (
                                not isinstance(candidate_supervision_loss_bkp, torch.Tensor)
                                or candidate_supervision_loss_bkp.ndim != 3
                                or not isinstance(candidate_supervision_components_bkp, dict)
                            ):
                                raise RuntimeError(
                                    "semantic bank independent loss must expose [B,K,P] tensors."
                                )
                            candidate_supervision_loss_bk = candidate_supervision_loss_bkp[
                                :, :, int(semantic_bank_active_penalty_index)
                            ]
                        else:
                            if candidate_supervision_raw.ndim != 2:
                                raise RuntimeError(
                                    "legacy candidate supervision must expose [B,K]."
                                )
                            candidate_supervision_loss_bk = candidate_supervision_raw
                intervention_supervision_loss_bk = None
                if pred_residual_intervention_supervision_weight > 0.0:
                    intervention_supervision_loss_bk = _pred_residual_intervention_supervision_loss(
                        y_base_bch=yhat_base,
                        pred_out=pred_out,
                        y_bch=y,
                        cluster_id_c=cluster_id_c,
                        K=K,
                        allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                        only_allowed=pred_residual_intervention_supervision_only_allowed,
                        min_gain=pred_residual_intervention_supervision_min_gain,
                        pos_weight=pred_residual_intervention_supervision_pos_weight,
                        apply_output_anchors=output_anchor_train_with_eval,
                        x_bcl=x,
                        query_start_abs_b=idx,
                        input_len=L,
                        moe_cfg=moe_cfg,
                        moe_enable=moe_enable,
                        observed_history_tc=data_window_tc,
                        train_stat_anchor_pc=train_stat_anchor_pc,
                        train_residual_anchor_phc=train_residual_anchor_phc,
                        learnable_output_anchor=learnable_output_anchor,
                    )
                loss_terms_bk, _ = _normalize_loss_terms(
                    {
                        "mse": mse_bk,
                        "mae_objective": mae_objective_bk,
                        "penalty": penalty_loss_bk,
                        "pred_residual": pred_loss_terms["total_bk"],
                    },
                    loss_normalization_cfg,
                )
                forecast_loss_component_bk = (
                    (mse_weight * loss_terms_bk["mse"])
                    + _apply_mae_objective_weight(loss_terms_bk["mae_objective"], mae_objective_weight_ep)
                )
                penalty_loss_component_bk = loss_terms_bk["penalty"]
                pred_residual_aux_component_bk = loss_terms_bk["pred_residual"]
                if patch_router_supervision_only:
                    forecast_loss_component_bk = forecast_loss_component_bk.detach()
                    penalty_loss_component_bk = penalty_loss_component_bk.detach()
                    pred_residual_aux_component_bk = pred_residual_aux_component_bk.detach()
                candidate_supervision_component_bk = torch.zeros_like(mse_bk)
                intervention_supervision_component_bk = torch.zeros_like(mse_bk)
                skip_noop_component_bk = torch.zeros_like(mse_bk)
                gate_utility_component_bk = torch.zeros_like(mse_bk)
                objective_loss_bk = (
                    forecast_loss_component_bk
                    + penalty_loss_component_bk
                )
                loss_bk = objective_loss_bk + pred_residual_aux_component_bk
                if candidate_supervision_loss_bk is not None:
                    candidate_supervision_component_bk = (
                        pred_residual_candidate_supervision_weight * candidate_supervision_loss_bk
                    )
                    loss_bk = loss_bk + candidate_supervision_component_bk
                if intervention_supervision_loss_bk is not None:
                    intervention_supervision_component_bk = (
                        pred_residual_intervention_supervision_weight * intervention_supervision_loss_bk
                    )
                    loss_bk = loss_bk + intervention_supervision_component_bk
                utility_base_bch = None
                utility_cand_bcpH = None
                if (
                    route_ce_weight > 0.0
                    or binary_adoption_weight > 0.0
                    or route_rate_alignment_weight > 0.0
                    or route_positive_recall_weight > 0.0
                    or route_precision_recall_weight > 0.0
                    or mse_utility_gate_weight > 0.0
                    or patch_router_expected_mse_weight > 0.0
                    or patch_router_mixture_mse_weight > 0.0
                    or patch_router_oracle_ce_weight_ep > 0.0
                    or patch_router_hierarchical_weight_ep > 0.0
                ):
                    utility_base_bch, utility_cand_bcpH = _pred_residual_candidates_on_eval_path(
                        yhat_base,
                        pred_out,
                        apply_output_anchors=output_anchor_train_with_eval,
                        x_bcl=x,
                        query_start_abs_b=idx,
                        input_len=L,
                        moe_cfg=moe_cfg,
                        moe_enable=moe_enable,
                        observed_history_tc=data_window_tc,
                        train_stat_anchor_pc=train_stat_anchor_pc,
                        train_residual_anchor_phc=train_residual_anchor_phc,
                        learnable_output_anchor=learnable_output_anchor,
                        cluster_id_c=cluster_id_c,
                        include_patch_route=not (
                            patch_router_expected_mse_weight > 0.0
                            or patch_router_mixture_mse_weight > 0.0
                            or patch_router_oracle_ce_weight_ep > 0.0
                            or patch_router_hierarchical_weight_ep > 0.0
                        ),
                    )
                if patch_router_expected_mse_weight > 0.0 and utility_cand_bcpH is not None:
                    patch_probs_bcqp = pred_out.get("patch_probs_bcqp")
                    patch_skip_prob_bcq = pred_out.get("patch_skip_prob_bcq")
                    if patch_probs_bcqp is None or patch_skip_prob_bcq is None:
                        raise ValueError(
                            "patch_router.expected_mse_weight requires patch router probabilities."
                        )
                    patch_utility_loss_bk = _patch_router_expected_mse_loss_bk(
                        base_bch=utility_base_bch,
                        candidate_bcpH=utility_cand_bcpH,
                        y_bch=y,
                        patch_probs_bcqp=patch_probs_bcqp,
                        patch_skip_prob_bcq=patch_skip_prob_bcq,
                        cluster_id_c=cluster_id_c,
                        K=K,
                        mae_weight=patch_router_expected_mae_weight,
                    )
                    patch_utility_component_bk = (
                        patch_router_expected_mse_weight * patch_utility_loss_bk
                    )
                    gate_utility_component_bk = gate_utility_component_bk + patch_utility_component_bk
                    loss_bk = loss_bk + patch_utility_component_bk
                    if patch_router_temporal_group_dro_enable:
                        batch_size_now, channel_count_now, horizon_now = (
                            utility_base_bch.shape
                        )
                        patch_count_now = int(patch_skip_prob_bcq.shape[2])
                        if horizon_now % patch_count_now != 0:
                            raise ValueError(
                                "temporal group DRO patch count must divide horizon."
                            )
                        base_error_bcq = (
                            (utility_base_bch - y)
                            .square()
                            .reshape(
                                batch_size_now,
                                channel_count_now,
                                patch_count_now,
                                horizon_now // patch_count_now,
                            )
                            .mean(dim=-1)
                        )
                        base_loss_bk = scatter_mean_bc_to_bk(
                            base_error_bcq.mean(dim=-1),
                            cluster_id_c,
                            K,
                        )
                        group_dro_loss, _, _ = (
                            _temporal_group_dro_incremental_loss(
                                incremental_loss_bk=(
                                    patch_utility_loss_bk - base_loss_bk.detach()
                                ),
                                query_index_b=idx,
                                train_window_count=len(dtr),
                                cluster_weight_k=cluster_weight_k,
                                num_domains=patch_router_temporal_group_dro_domains,
                                temperature=(
                                    patch_router_temporal_group_dro_temperature
                                ),
                            )
                        )
                        group_dro_component_bk = (
                            patch_router_temporal_group_dro_weight
                            * group_dro_loss
                        ).expand_as(patch_utility_loss_bk)
                        gate_utility_component_bk = (
                            gate_utility_component_bk
                            + group_dro_component_bk
                        )
                        loss_bk = loss_bk + group_dro_component_bk
                if patch_router_mixture_mse_weight > 0.0 and utility_cand_bcpH is not None:
                    patch_probs_bcqp = pred_out.get("patch_probs_bcqp")
                    if patch_probs_bcqp is None:
                        raise ValueError(
                            "patch_router.mixture_mse_weight requires patch router probabilities."
                        )
                    patch_mixture_loss_bk = _patch_router_mixture_mse_loss_bk(
                        base_bch=utility_base_bch,
                        candidate_bcpH=utility_cand_bcpH,
                        y_bch=y,
                        patch_probs_bcqp=patch_probs_bcqp,
                        cluster_id_c=cluster_id_c,
                        K=K,
                        mae_weight=patch_router_mixture_mae_weight,
                        allowed_penalty_mask_cp=(
                            pred_residual.channel_penalty_allowed_mask_cp
                            if pred_residual is not None
                            else None
                        ),
                    )
                    patch_mixture_component_bk = (
                        patch_router_mixture_mse_weight * patch_mixture_loss_bk
                    )
                    gate_utility_component_bk = (
                        gate_utility_component_bk + patch_mixture_component_bk
                    )
                    loss_bk = loss_bk + patch_mixture_component_bk
                if patch_router_oracle_ce_weight_ep > 0.0 and utility_cand_bcpH is not None:
                    patch_probs_bcqp = pred_out.get("patch_probs_bcqp")
                    patch_skip_prob_bcq = pred_out.get("patch_skip_prob_bcq")
                    if patch_probs_bcqp is None or patch_skip_prob_bcq is None:
                        raise ValueError(
                            "patch_router.oracle_ce_weight requires patch router probabilities."
                        )
                    patch_ce_loss_bk = _patch_router_oracle_ce_loss_bk(
                        base_bch=utility_base_bch,
                        candidate_bcpH=utility_cand_bcpH,
                        y_bch=y,
                        patch_probs_bcqp=patch_probs_bcqp,
                        patch_skip_prob_bcq=patch_skip_prob_bcq,
                        cluster_id_c=cluster_id_c,
                        K=K,
                        min_abs_improvement=patch_router_oracle_min_abs_improvement,
                    )
                    patch_ce_component_bk = patch_router_oracle_ce_weight_ep * patch_ce_loss_bk
                    gate_utility_component_bk = gate_utility_component_bk + patch_ce_component_bk
                    loss_bk = loss_bk + patch_ce_component_bk
                if patch_router_hierarchical_weight_ep > 0.0 and utility_cand_bcpH is not None:
                    patch_adopt_prob_bcq = pred_out.get("patch_proposal_adopt_prob_bcq")
                    patch_final_adopt_prob_bcq = pred_out.get("patch_adopt_prob_bcq")
                    patch_conditional_probs_bcqp = pred_out.get(
                        "patch_penalty_conditional_probs_bcqp"
                    )
                    patch_benefit_probs_bcqp = pred_out.get("patch_penalty_benefit_probs_bcqp")
                    patch_utility_scores_bcqp = pred_out.get("patch_penalty_utility_scores_bcqp")
                    if (
                        patch_adopt_prob_bcq is None
                        or patch_conditional_probs_bcqp is None
                        or patch_benefit_probs_bcqp is None
                        or patch_utility_scores_bcqp is None
                    ):
                        raise ValueError(
                            "patch_router hierarchical recall supervision requires hierarchical gate outputs."
                        )
                    hierarchical_terms = _patch_router_hierarchical_recall_loss_terms(
                        base_bch=utility_base_bch,
                        candidate_bcpH=utility_cand_bcpH,
                        y_bch=y,
                        patch_adopt_prob_bcq=patch_adopt_prob_bcq,
                        patch_penalty_conditional_probs_bcqp=patch_conditional_probs_bcqp,
                        patch_penalty_benefit_probs_bcqp=patch_benefit_probs_bcqp,
                        patch_penalty_utility_scores_bcqp=patch_utility_scores_bcqp,
                        patch_penalty_mse_utility_scores_bcqp=pred_out.get(
                            "patch_penalty_mse_utility_scores_bcqp"
                        ),
                        patch_penalty_mae_utility_scores_bcqp=pred_out.get(
                            "patch_penalty_mae_utility_scores_bcqp"
                        ),
                        patch_penalty_risk_benefit_probs_bcqp=pred_out.get(
                            "patch_penalty_risk_benefit_probs_bcqp"
                        ),
                        patch_penalty_risk_positive_magnitude_bcqp=pred_out.get(
                            "patch_penalty_risk_positive_magnitude_bcqp"
                        ),
                        patch_penalty_risk_negative_magnitude_bcqp=pred_out.get(
                            "patch_penalty_risk_negative_magnitude_bcqp"
                        ),
                        patch_penalty_proposal_logits_bcqp=pred_out.get(
                            "patch_penalty_proposal_logits_bcqp"
                        ),
                        patch_penalty_proposal_rescue_logits_bcqp=pred_out.get(
                            "patch_penalty_proposal_rescue_logits_bcqp"
                        ),
                        patch_penalty_risk_lower_quantile_scores_bcqp=pred_out.get(
                            "patch_penalty_risk_lower_quantile_scores_bcqp"
                        ),
                        patch_final_adopt_prob_bcq=patch_final_adopt_prob_bcq,
                        patch_penalty_pairwise_rank_scores_bcqp=pred_out.get(
                            "patch_penalty_pairwise_rank_scores_bcqp"
                        ),
                        patch_penalty_proposal_mask_bcqp=pred_out.get(
                            "patch_penalty_proposal_mask_bcqp"
                        ),
                        patch_active_mask_bcq=(
                            pred_out.get("patch_fixed_penalty_active_bcq")
                            if patch_router_mask_inactive_fixed_channels
                            else None
                        ),
                        cluster_id_c=cluster_id_c,
                        K=K,
                        min_abs_improvement=patch_router_hierarchical_min_abs_improvement,
                        **patch_router_hierarchical_loss_cfg,
                    )
                    if (
                        stage2_objective_overlap_enable
                        and len(stage2_objective_overlap_batches)
                        < stage2_objective_overlap_max_batches
                    ):
                        patch_probs_bcqp = pred_out.get("patch_probs_bcqp")
                        patch_skip_prob_bcq = pred_out.get("patch_skip_prob_bcq")
                        if (
                            patch_probs_bcqp is None
                            or patch_skip_prob_bcq is None
                        ):
                            raise ValueError(
                                "objective-overlap diagnostics require soft patch route "
                                "probabilities."
                            )
                        objective_overlap_reference_bk = (
                            _patch_router_expected_mse_loss_bk(
                                base_bch=utility_base_bch,
                                candidate_bcpH=utility_cand_bcpH,
                                y_bch=y,
                                patch_probs_bcqp=patch_probs_bcqp,
                                patch_skip_prob_bcq=patch_skip_prob_bcq,
                                cluster_id_c=cluster_id_c,
                                K=K,
                            )
                        )
                    hierarchical_component_bk = (
                        patch_router_hierarchical_weight_ep * hierarchical_terms["total_bk"]
                    )
                    if patch_router_temporal_calibration_enable:
                        supervision_mask_b = (
                            idx < int(patch_router_supervision_end_idx)
                        ).to(
                            device=hierarchical_component_bk.device,
                            dtype=hierarchical_component_bk.dtype,
                        )
                        hierarchical_component_bk = (
                            hierarchical_component_bk
                            * supervision_mask_b.view(-1, 1)
                        )
                    gate_utility_component_bk = gate_utility_component_bk + hierarchical_component_bk
                    loss_bk = loss_bk + hierarchical_component_bk
                if route_ce_weight > 0.0 and utility_cand_bcpH is not None:
                    route_labels_bk, route_gain_bk = _cluster_route_oracle_labels_and_gain_from_candidates(
                        base_bch=utility_base_bch,
                        cand_bcpH=utility_cand_bcpH,
                        y_bch=y,
                        cluster_id_c=cluster_id_c,
                        K=K,
                        allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                        min_abs_improvement=route_ce_min_abs_improvement,
                        min_rel_improvement=route_ce_min_rel_improvement,
                        min_candidate_delta_rms=route_ce_min_candidate_delta_rms,
                    )
                    route_ce_active_mask_bk = None
                    if route_ce_ignore_abs_gain_below > 0.0:
                        route_ce_active_mask_bk = _route_ce_active_mask_from_gain(
                            route_gain_bk,
                            ignore_abs_gain_below=route_ce_ignore_abs_gain_below,
                        )
                    route_ce_loss_bk = _route_ce_loss_from_probs(
                        probs_bkp=probs_bkp,
                        skip_prob_bk=skip_prob_bk if allow_skip else None,
                        labels_bk=route_labels_bk,
                        probs_include_skip_mass=bool(skip_competes),
                        class_weight_q=_route_ce_class_weight_from_labels(
                            labels_bk=route_labels_bk,
                            num_classes=P + 1,
                            mode=route_ce_class_weight_mode,
                            max_weight=route_ce_max_class_weight,
                            active_mask_bk=route_ce_active_mask_bk,
                        ),
                    )
                    if route_ce_active_mask_bk is not None:
                        route_ce_loss_bk = route_ce_loss_bk * route_ce_active_mask_bk.to(dtype=route_ce_loss_bk.dtype)
                    gate_utility_component_bk = gate_utility_component_bk + route_ce_weight * route_ce_loss_bk
                    loss_bk = loss_bk + route_ce_weight * route_ce_loss_bk
                if binary_adoption_weight > 0.0 and utility_cand_bcpH is not None:
                    binary_labels_bk, binary_gain_bk = _cluster_route_oracle_labels_and_gain_from_candidates(
                        base_bch=utility_base_bch,
                        cand_bcpH=utility_cand_bcpH,
                        y_bch=y,
                        cluster_id_c=cluster_id_c,
                        K=K,
                        allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                        min_abs_improvement=binary_adoption_min_abs_improvement,
                        min_rel_improvement=binary_adoption_min_rel_improvement,
                        min_candidate_delta_rms=binary_adoption_min_candidate_delta_rms,
                    )
                    binary_active_mask_bk = None
                    if binary_adoption_ignore_abs_gain_below > 0.0:
                        binary_active_mask_bk = _route_ce_active_mask_from_gain(
                            binary_gain_bk,
                            ignore_abs_gain_below=binary_adoption_ignore_abs_gain_below,
                        )
                    binary_loss_bk = _route_binary_adoption_loss_from_probs(
                        probs_bkp=probs_bkp,
                        skip_prob_bk=skip_prob_bk if allow_skip else None,
                        labels_bk=binary_labels_bk,
                        probs_include_skip_mass=bool(skip_competes),
                        allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                        active_mask_bk=binary_active_mask_bk,
                        positive_weight=binary_adoption_positive_weight,
                        negative_weight=binary_adoption_negative_weight,
                    )
                    if binary_loss_bk is not None:
                        binary_component_bk = binary_adoption_weight * binary_loss_bk
                        gate_utility_component_bk = gate_utility_component_bk + binary_component_bk
                        loss_bk = loss_bk + binary_component_bk
                if route_rate_alignment_weight > 0.0 and utility_cand_bcpH is not None:
                    rate_labels_bk, rate_gain_bk = _cluster_route_oracle_labels_and_gain_from_candidates(
                        base_bch=utility_base_bch,
                        cand_bcpH=utility_cand_bcpH,
                        y_bch=y,
                        cluster_id_c=cluster_id_c,
                        K=K,
                        allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                        min_abs_improvement=route_rate_alignment_min_abs_improvement,
                        min_rel_improvement=route_rate_alignment_min_rel_improvement,
                        min_candidate_delta_rms=route_rate_alignment_min_candidate_delta_rms,
                    )
                    rate_active_mask_bk = None
                    if route_rate_alignment_ignore_abs_gain_below > 0.0:
                        rate_active_mask_bk = _route_ce_active_mask_from_gain(
                            rate_gain_bk,
                            ignore_abs_gain_below=route_rate_alignment_ignore_abs_gain_below,
                        )
                    rate_loss_bk = _route_rate_alignment_loss_from_probs(
                        probs_bkp=probs_bkp,
                        skip_prob_bk=skip_prob_bk if allow_skip else None,
                        labels_bk=rate_labels_bk,
                        probs_include_skip_mass=bool(skip_competes),
                        allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                        active_mask_bk=rate_active_mask_bk,
                    )
                    rate_component_bk = route_rate_alignment_weight * rate_loss_bk
                    gate_utility_component_bk = gate_utility_component_bk + rate_component_bk
                    loss_bk = loss_bk + rate_component_bk
                if route_positive_recall_weight > 0.0 and utility_cand_bcpH is not None:
                    recall_labels_bk, recall_gain_bk = _cluster_route_oracle_labels_and_gain_from_candidates(
                        base_bch=utility_base_bch,
                        cand_bcpH=utility_cand_bcpH,
                        y_bch=y,
                        cluster_id_c=cluster_id_c,
                        K=K,
                        allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                        min_abs_improvement=route_positive_recall_min_abs_improvement,
                        min_rel_improvement=route_positive_recall_min_rel_improvement,
                        min_candidate_delta_rms=route_positive_recall_min_candidate_delta_rms,
                    )
                    recall_active_mask_bk = None
                    if route_positive_recall_ignore_abs_gain_below > 0.0:
                        recall_active_mask_bk = _route_ce_active_mask_from_gain(
                            recall_gain_bk,
                            ignore_abs_gain_below=route_positive_recall_ignore_abs_gain_below,
                        )
                    recall_loss_bk = _route_positive_recall_loss_from_probs(
                        probs_bkp=probs_bkp,
                        skip_prob_bk=skip_prob_bk if allow_skip else None,
                        labels_bk=recall_labels_bk,
                        probs_include_skip_mass=bool(skip_competes),
                        active_mask_bk=recall_active_mask_bk,
                        mode=route_positive_recall_mode,
                        target_probability=route_positive_recall_target_probability,
                    )
                    recall_component_bk = route_positive_recall_weight * recall_loss_bk
                    gate_utility_component_bk = gate_utility_component_bk + recall_component_bk
                    loss_bk = loss_bk + recall_component_bk
                if route_precision_recall_weight > 0.0 and utility_cand_bcpH is not None:
                    precision_labels_bk, precision_gain_bk = _cluster_route_oracle_labels_and_gain_from_candidates(
                        base_bch=utility_base_bch,
                        cand_bcpH=utility_cand_bcpH,
                        y_bch=y,
                        cluster_id_c=cluster_id_c,
                        K=K,
                        allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                        min_abs_improvement=route_precision_recall_min_abs_improvement,
                        min_rel_improvement=route_precision_recall_min_rel_improvement,
                        min_candidate_delta_rms=route_precision_recall_min_candidate_delta_rms,
                    )
                    precision_active_mask_bk = None
                    if route_precision_recall_ignore_abs_gain_below > 0.0:
                        precision_active_mask_bk = _route_ce_active_mask_from_gain(
                            precision_gain_bk,
                            ignore_abs_gain_below=route_precision_recall_ignore_abs_gain_below,
                        )
                    precision_loss_bk = _route_precision_constrained_recall_loss_from_probs(
                        probs_bkp=probs_bkp,
                        skip_prob_bk=skip_prob_bk if allow_skip else None,
                        labels_bk=precision_labels_bk,
                        probs_include_skip_mass=bool(skip_competes),
                        active_mask_bk=precision_active_mask_bk,
                        recall_mode=route_precision_recall_mode,
                        recall_target_probability=route_precision_recall_target_probability,
                        false_adopt_max_probability=route_precision_recall_false_adopt_max_probability,
                        false_adopt_weight=route_precision_recall_false_adopt_weight,
                    )
                    precision_component_bk = route_precision_recall_weight * precision_loss_bk
                    gate_utility_component_bk = gate_utility_component_bk + precision_component_bk
                    loss_bk = loss_bk + precision_component_bk
                if (
                    allow_skip
                    and skip_supervision_weight > 0.0
                    and pred_residual is not None
                    and skip_prob_bk is not None
                ):
                    with torch.no_grad():
                        pred_no_skip = pred_residual(
                            x,
                            yhat_base,
                            cluster_id_c,
                            mask_bkp.detach(),
                            skip_bk=None,
                            query_start_abs_b=idx,
                            fixed_expert_delta_bch=fixed_expert_delta_bch,
                        )
                        yhat_no_skip = pred_no_skip["y_final"]
                        no_op_base_bch = pred_no_skip.get("candidate_base_bch", yhat_base)
                        base_mse_bc_for_skip = (no_op_base_bch - y).pow(2).mean(dim=-1)
                        no_skip_mse_bc = (yhat_no_skip - y).pow(2).mean(dim=-1)
                        base_mse_bk_for_skip = scatter_mean_bc_to_bk(base_mse_bc_for_skip, cluster_id_c, K)
                        no_skip_mse_bk = scatter_mean_bc_to_bk(no_skip_mse_bc, cluster_id_c, K)
                        skip_label_bk = (
                            base_mse_bk_for_skip + float(skip_supervision_margin) < no_skip_mse_bk
                        ).to(dtype=skip_prob_bk.dtype)
                    skip_prob_clamped = skip_prob_bk.clamp(1.0e-6, 1.0 - 1.0e-6)
                    skip_bce_bk = -(
                        skip_label_bk * skip_prob_clamped.log()
                        + (1.0 - skip_label_bk) * (1.0 - skip_prob_clamped).log()
                    )
                    skip_noop_component_bk = skip_supervision_weight * skip_bce_bk
                    loss_bk = loss_bk + skip_noop_component_bk
                if mse_utility_gate_weight > 0.0:
                    mse_gate_result = _mse_utility_gate_supervision_loss(
                        probs_bkp=probs_bkp,
                        skip_prob_bk=skip_prob_bk if allow_skip else None,
                        allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                        y_base_bch=yhat_base,
                        pred_out=pred_out,
                        y_bch=y,
                        cluster_id_c=cluster_id_c,
                        K=K,
                        y_base_eval_bch=utility_base_bch,
                        cand_eval_bcpH=utility_cand_bcpH,
                        temperature=mse_utility_gate_temperature,
                        min_gain=mse_utility_gate_min_gain,
                        mae_weight=mse_utility_gate_mae_weight,
                        target_power=mse_utility_gate_target_power,
                        include_skip=mse_utility_gate_include_skip,
                        probs_include_skip_mass=bool(skip_competes),
                        target_mode=mse_utility_gate_target_mode,
                        return_diagnostics=True,
                    )
                    mse_gate_loss_bk, mse_gate_diag = (
                        mse_gate_result if mse_gate_result is not None else (None, None)
                    )
                    if mse_gate_loss_bk is not None:
                        mse_gate_component_bk = mse_utility_gate_weight * mse_gate_loss_bk
                        gate_utility_component_bk = gate_utility_component_bk + mse_gate_component_bk
                        loss_bk = loss_bk + mse_gate_component_bk
                if (not bilevel_enable) and learnable_lambda is not None and learnable_lambda_reg_weight > 0.0:
                    loss_bk = loss_bk + learnable_lambda_reg_weight * learnable_lambda.regularization().unsqueeze(0)
                if (not bilevel_enable) and dynamic_lambda is not None and dynamic_lambda_reg_weight > 0.0:
                    base_lam = base_lambda_kp.unsqueeze(0).expand(x.shape[0], K, P).clamp_min(1.0e-8)
                    scale_bkp = lam / base_lam
                    loss_bk = loss_bk + dynamic_lambda_reg_weight * scale_bkp.log().pow(2).mean(dim=-1)
                if moe_enable and (gate_entropy_weight != 0.0 or gate_balance_weight != 0.0):
                    loss_bk = loss_bk + _gate_regularization(
                        probs_bkp,
                        gate_entropy_weight=gate_entropy_weight,
                        gate_balance_weight=gate_balance_weight,
                        gate_entropy_target_frac=gate_entropy_target_frac,
                        gate_balance_target_kp=gate_balance_target_kp,
                    )
                known_component_bk = (
                    forecast_loss_component_bk
                    + penalty_loss_component_bk
                    + pred_residual_aux_component_bk
                    + candidate_supervision_component_bk
                    + intervention_supervision_component_bk
                    + skip_noop_component_bk
                    + gate_utility_component_bk
                )
                other_aux_component_bk = loss_bk - known_component_bk
            else:
                raw_objective_loss_bk = (
                    (mse_weight * mse_bk)
                    + _apply_mae_objective_weight(mae_objective_bk, mae_objective_weight_ep)
                )
                loss_terms_bk, _ = _normalize_loss_terms(
                    {
                        "mse": mse_bk,
                        "mae_objective": mae_objective_bk,
                        "penalty": torch.zeros_like(mse_bk),
                        "pred_residual": torch.zeros_like(mse_bk),
                    },
                    loss_normalization_cfg,
                )
                objective_loss_bk = (
                    (mse_weight * loss_terms_bk["mse"])
                    + _apply_mae_objective_weight(loss_terms_bk["mae_objective"], mae_objective_weight_ep)
                )
                loss_bk = objective_loss_bk
                forecast_loss_component_bk = objective_loss_bk
                penalty_loss_component_bk = torch.zeros_like(mse_bk)
                pred_residual_aux_component_bk = torch.zeros_like(mse_bk)
                candidate_supervision_component_bk = torch.zeros_like(mse_bk)
                intervention_supervision_component_bk = torch.zeros_like(mse_bk)
                skip_noop_component_bk = torch.zeros_like(mse_bk)
                gate_utility_component_bk = torch.zeros_like(mse_bk)
                other_aux_component_bk = loss_bk - forecast_loss_component_bk
            _accumulate_detached_sum_(train_loss_sum_k, raw_objective_loss_bk)
            _accumulate_detached_sum_(train_mse_sum_k, mse_bk)
            _accumulate_detached_sum_(train_mae_sum_k, mae_bk)
            if stage2_loss_audit_enable:
                _accumulate_detached_sum_(stage2_total_loss_sum_k, loss_bk)
                _accumulate_detached_sum_(stage2_forecast_loss_sum_k, forecast_loss_component_bk)
                _accumulate_detached_sum_(stage2_penalty_loss_sum_k, penalty_loss_component_bk)
                _accumulate_detached_sum_(stage2_pred_residual_aux_loss_sum_k, pred_residual_aux_component_bk)
                _accumulate_detached_sum_(stage2_candidate_supervision_loss_sum_k, candidate_supervision_component_bk)
                _accumulate_detached_sum_(stage2_gate_utility_loss_sum_k, gate_utility_component_bk)
                _accumulate_detached_sum_(stage2_skip_noop_loss_sum_k, skip_noop_component_bk)
                _accumulate_detached_sum_(stage2_intervention_supervision_loss_sum_k, intervention_supervision_component_bk)
                _accumulate_detached_sum_(stage2_other_aux_loss_sum_k, other_aux_component_bk)
            if mse_gate_diag is not None:
                count_bk = torch.ones_like(mse_gate_diag["valid_bk"])
                _accumulate_detached_sum_(mse_gate_diag_count_k, count_bk)
                _accumulate_detached_sum_(mse_gate_valid_sum_k, mse_gate_diag["valid_bk"])
                _accumulate_detached_sum_(mse_gate_skip_target_sum_k, mse_gate_diag["target_skip_bk"])
                _accumulate_detached_sum_(mse_gate_best_gain_sum_k, mse_gate_diag["best_gain_bk"])
                if "skip_prob_bk" in mse_gate_diag:
                    _accumulate_detached_sum_(mse_gate_skip_prob_sum_k, mse_gate_diag["skip_prob_bk"])
            if mse_gate_loss_bk is not None:
                _accumulate_detached_sum_(mse_gate_loss_sum_k, mse_gate_loss_bk)
            train_cnt += int(loss_bk.shape[0])
            train_weight_k = _training_cluster_weight(
                cluster_weight_k,
                stopped,
                shared_moe=shared_moe_across_clusters,
            )
            loss = reduce_cluster_metric(loss_bk, train_weight_k).mean()

            if (
                stage2_objective_overlap_enable
                and objective_overlap_reference_bk is not None
                and hierarchical_terms is not None
                and pred_residual is not None
                and getattr(pred_residual, "patch_router", None) is not None
                and len(stage2_objective_overlap_batches)
                < stage2_objective_overlap_max_batches
            ):
                reference_loss = reduce_cluster_metric(
                    objective_overlap_reference_bk,
                    train_weight_k,
                ).mean()
                overlap_term_losses = {
                    "total_active_hierarchical": (
                        float(patch_router_hierarchical_weight_ep)
                        * reduce_cluster_metric(
                            hierarchical_terms["total_bk"],
                            train_weight_k,
                        ).mean()
                    )
                }
                overlap_term_weights = {
                    "total_active_hierarchical": float(
                        patch_router_hierarchical_weight_ep
                    )
                }
                for weight_name, configured_weight in (
                    patch_router_hierarchical_loss_cfg.items()
                ):
                    if not weight_name.endswith("_weight"):
                        continue
                    configured_weight = float(configured_weight)
                    if configured_weight <= 0.0:
                        continue
                    term_name = f"{weight_name[:-7]}_bk"
                    if term_name not in hierarchical_terms:
                        continue
                    report_name = term_name[:-3]
                    effective_weight = (
                        float(patch_router_hierarchical_weight_ep)
                        * configured_weight
                    )
                    overlap_term_losses[report_name] = (
                        effective_weight
                        * reduce_cluster_metric(
                            hierarchical_terms[term_name],
                            train_weight_k,
                        ).mean()
                    )
                    overlap_term_weights[report_name] = float(effective_weight)

                risk_encoder_prefixes = (
                    "W1",
                    "b1",
                    "W_candidate",
                    "b_candidate",
                    "penalty_embedding",
                )
                risk_sign_prefixes = ("W_risk_sign", "b_risk_sign")
                overlap_summary = _loss_gradient_overlap_summary(
                    reference_loss=reference_loss,
                    term_losses=overlap_term_losses,
                    named_parameters=(
                        pred_residual.patch_router.named_parameters()
                    ),
                    parameter_groups={
                        "all_patch_router": ("",),
                        "execution_risk_path": (
                            *risk_encoder_prefixes,
                            *risk_sign_prefixes,
                        ),
                        "execution_risk_encoder": risk_encoder_prefixes,
                        "execution_risk_sign_head": risk_sign_prefixes,
                        "risk_magnitude_heads": (
                            "W_risk_gain",
                            "b_risk_gain",
                            "W_risk_cost",
                            "b_risk_cost",
                        ),
                        "proposal_path": (
                            "W_proposal1",
                            "b_proposal1",
                            "W_proposal_candidate",
                            "b_proposal_candidate",
                            "proposal_penalty_embedding",
                            "W_adopt",
                            "b_adopt",
                            "W_benefit",
                            "b_benefit",
                            "W_proposal_rescue",
                            "b_proposal_rescue",
                        ),
                        "pairwise_head": (
                            "W_pairwise_rank",
                            "b_pairwise_rank",
                        ),
                    },
                )
                overlap_summary.update(
                    {
                        "epoch": int(ep),
                        "batch": int(n_batches),
                        "reference": (
                            "soft expected patch MSE on the exact fixed-candidate "
                            "eval output-anchor path"
                        ),
                        "reference_loss": float(reference_loss.detach().item()),
                        "effective_term_weights": overlap_term_weights,
                        "term_losses": {
                            name: float(value.detach().item())
                            for name, value in overlap_term_losses.items()
                        },
                    }
                )
                stage2_objective_overlap_batches.append(overlap_summary)

            semantic_loss_p: List[torch.Tensor] = []
            if pred_residual_semantic_bank_stage1:
                if candidate_supervision_loss_bkp is None:
                    raise RuntimeError(
                        "semantic bank independent optimization requires per-expert loss exposure."
                    )
                if not isinstance(candidate_supervision_components_bkp, dict):
                    raise RuntimeError(
                        "semantic bank independent optimization requires per-expert components."
                    )
                if (
                    not pred_residual_semantic_raw_gradient_accumulation
                    or semantic_raw_gradient_pending_count == 0
                ):
                    for optimizer_p in semantic_bank_optimizers_p:
                        if optimizer_p is not None:
                            optimizer_p.zero_grad(set_to_none=True)
                    if semantic_level_need_gate_optimizer is not None:
                        semantic_level_need_gate_optimizer.zero_grad(set_to_none=True)
                active_p = int(semantic_bank_active_penalty_index)
                for p in [active_p]:
                    loss_p = (
                        float(pred_residual_candidate_supervision_weight)
                        * reduce_cluster_metric(
                            candidate_supervision_loss_bkp[:, :, p],
                            cluster_weight_k,
                        ).mean()
                    )
                    semantic_loss_p.append(loss_p)
                    for component_name, component_bkp in (
                        candidate_supervision_components_bkp.items()
                    ):
                        component_value = reduce_cluster_metric(
                            component_bkp[:, :, p],
                            cluster_weight_k,
                        ).mean()
                        semantic_component_sum_by_name_p[component_name][p] += float(
                            component_value.detach().item()
                        )
                    loss_p.backward()
                # This mean is reporting-only.  It never participates in backward,
                # clipping, optimizer state, scheduling, or checkpoint selection.
                loss = semantic_loss_p[0].detach()
                for p, params_p in enumerate(semantic_bank_params_by_penalty):
                    if p != active_p:
                        if any(param.grad is not None for param in pred_residual.get_penalty_body_params(p)):
                            raise RuntimeError(
                                f"inactive semantic expert {penalty_names[p]} received a gradient."
                            )
                semantic_loss_sum_p[active_p] += float(
                    semantic_loss_p[0].detach().item()
                )
                semantic_batch_count += 1
                if pred_residual_semantic_raw_gradient_accumulation:
                    if any(param.grad is not None for param in model.parameters()):
                        raise RuntimeError(
                            "semantic raw-gradient accumulation reached the backbone."
                        )
                    if any(param.grad is not None for param in gate.parameters()):
                        raise RuntimeError(
                            "semantic raw-gradient accumulation reached the frozen gate."
                        )
                    semantic_raw_gradient_pending_count += 1
                    if (
                        semantic_raw_gradient_pending_count
                        > pred_residual_semantic_raw_gradient_microbatches
                    ):
                        raise RuntimeError(
                            "semantic raw-gradient accumulation exceeded its target count."
                        )
                    flush_accumulated = (
                        _semantic_bank_should_flush_raw_gradient_accumulation(
                            pending_count=semantic_raw_gradient_pending_count,
                            target_microbatches=(
                                pred_residual_semantic_raw_gradient_microbatches
                            ),
                            completed_batches=n_batches + 1,
                            total_batches=steps_per_epoch,
                        )
                    )
                    if flush_accumulated:
                        optimizer_p = semantic_bank_optimizers_p[active_p]
                        if optimizer_p is None:
                            raise RuntimeError(
                                "active semantic expert is missing its optimizer."
                            )
                        if pred_residual_semantic_level_separate_need_gate:
                            if semantic_level_need_gate_optimizer is None:
                                raise RuntimeError("LEVEL need-gate optimizer is missing.")
                            step_metrics = _semantic_bank_finalize_disjoint_gradient_step(
                                {
                                    "amplitude": optimizer_p,
                                    "need_gate": semantic_level_need_gate_optimizer,
                                },
                                semantic_level_disjoint_params_by_group,
                                raw_gradient_accumulation=True,
                                accumulation_count=semantic_raw_gradient_pending_count,
                                grad_clip=grad_clip,
                            )
                        else:
                            step_metrics = _semantic_bank_finalize_gradient_step(
                                optimizer_p,
                                semantic_bank_params_by_penalty[active_p],
                                raw_gradient_accumulation=True,
                                accumulation_count=(
                                    semantic_raw_gradient_pending_count
                                ),
                                grad_clip=grad_clip,
                            )
                        semantic_grad_raw_sum_p[active_p] += float(
                            step_metrics["raw_gradient_l2"]
                        )
                        semantic_grad_clipped_sum_p[active_p] += float(
                            step_metrics["clipped_gradient_l2"]
                        )
                        semantic_update_sum_p[active_p] += float(
                            step_metrics["parameter_update_l2"]
                        )
                        for group, row in step_metrics.get("groups", {}).items():
                            for metric in semantic_level_disjoint_metric_sums[group]:
                                semantic_level_disjoint_metric_sums[group][metric] += float(
                                    row[metric]
                                )
                        semantic_raw_gradient_group_counts.append(
                            int(step_metrics["accumulation_count"])
                        )
                        semantic_step_count += 1
                        semantic_raw_gradient_pending_count = 0
                else:
                    optimizer_p = semantic_bank_optimizers_p[active_p]
                    if optimizer_p is None:
                        raise RuntimeError(
                            "active semantic expert is missing its optimizer."
                        )
                    if pred_residual_semantic_level_separate_need_gate:
                        if semantic_level_need_gate_optimizer is None:
                            raise RuntimeError("LEVEL need-gate optimizer is missing.")
                        step_metrics = _semantic_bank_finalize_disjoint_gradient_step(
                            {
                                "amplitude": optimizer_p,
                                "need_gate": semantic_level_need_gate_optimizer,
                            },
                            semantic_level_disjoint_params_by_group,
                            grad_clip=grad_clip,
                            raw_gradient_accumulation=False,
                            accumulation_count=1,
                        )
                    else:
                        step_metrics = _semantic_bank_finalize_gradient_step(
                            optimizer_p,
                            semantic_bank_params_by_penalty[active_p],
                            grad_clip=grad_clip,
                            raw_gradient_accumulation=False,
                            accumulation_count=1,
                        )
                    semantic_grad_raw_sum_p[active_p] += float(
                        step_metrics["raw_gradient_l2"]
                    )
                    semantic_grad_clipped_sum_p[active_p] += float(
                        step_metrics["clipped_gradient_l2"]
                    )
                    semantic_update_sum_p[active_p] += float(
                        step_metrics["parameter_update_l2"]
                    )
                    for group, row in step_metrics.get("groups", {}).items():
                        for metric in semantic_level_disjoint_metric_sums[group]:
                            semantic_level_disjoint_metric_sums[group][metric] += float(
                                row[metric]
                            )
                    semantic_step_count += 1
            else:
                for opt_k in optimizers:
                    if opt_k is None:
                        continue
                    opt_k.zero_grad(set_to_none=True)
                loss.backward()

                if grad_clip > 0:
                    for k, params_k in enumerate(cluster_params):
                        if (
                            len(params_k) == 0
                            or not _optimizer_slot_active(stopped, k, shared_moe=shared_moe_across_clusters)
                        ):
                            continue
                        torch.nn.utils.clip_grad_norm_(params_k, grad_clip)

            model.mask_cluster_grads(stopped)
            if moe_enable:
                gate.mask_cluster_grads(stopped)
                _mask_gate_grads_after_epoch(
                    gate=gate,
                    epoch=ep,
                    freeze_after_epoch=pred_residual_freeze_gate_after_epoch,
                    stopped=stopped,
                )
            if pred_residual is not None and not pred_residual_semantic_bank_stage1:
                pred_residual.mask_cluster_grads(stopped)
            if dynamic_lambda is not None:
                dynamic_lambda.mask_cluster_grads(stopped)
            if learnable_lambda is not None:
                learnable_lambda.mask_cluster_grads(stopped)
            if learnable_output_anchor is not None:
                learnable_output_anchor.mask_cluster_grads(stopped)
            if stage2_loss_audit_enable:
                stage2_grad_norm_sum["backbone"] += _parameter_grad_l2_norm(model.parameters())
                stage2_grad_norm_sum["gate"] += _parameter_grad_l2_norm(gate.parameters())
                if pred_residual is not None:
                    stage2_grad_norm_sum["pred_residual"] += _parameter_grad_l2_norm(pred_residual.parameters())
                if dynamic_lambda is not None:
                    stage2_grad_norm_sum["dynamic_lambda"] += _parameter_grad_l2_norm(dynamic_lambda.parameters())
                if learnable_lambda is not None:
                    stage2_grad_norm_sum["learnable_lambda"] += _parameter_grad_l2_norm(learnable_lambda.parameters())
                if learnable_output_anchor is not None:
                    stage2_grad_norm_sum["learnable_output_anchor"] += _parameter_grad_l2_norm(
                        learnable_output_anchor.parameters()
                    )
                stage2_grad_norm_batches += 1
            if not pred_residual_semantic_bank_stage1:
                for k, opt_k in enumerate(optimizers):
                    if opt_k is None:
                        continue
                    if not _optimizer_slot_active(stopped, k, shared_moe=shared_moe_across_clusters):
                        continue
                    opt_k.step()

            running += float(loss.item())
            n_batches += 1
            if train_progress.enabled:
                step_now = (ep - 1) * steps_per_epoch + min(n_batches, steps_per_epoch)
                train_progress.update(
                    step_now,
                    suffix=(
                        f"epoch={ep}/{epochs} batch={n_batches}/{steps_per_epoch} "
                        f"loss={running / max(n_batches, 1):.6f}"
                    ),
                )

        if (
            pred_residual_semantic_raw_gradient_accumulation
            and semantic_raw_gradient_pending_count != 0
        ):
            raise RuntimeError(
                "semantic raw-gradient tail was not flushed at epoch end."
            )
        assert_pairwise_frozen_parameters_unchanged(f"epoch_{ep}_post_optimizer")
        outer_loss_epoch = None
        if bilevel_enable:
            outer_vals = []
            for _ in range(bilevel_steps_per_epoch):
                outer_val = bilevel_outer_step(ep, warmup_scale)
                if outer_val is not None:
                    outer_vals.append(outer_val)
            if len(outer_vals) > 0:
                outer_loss_epoch = float(sum(outer_vals) / len(outer_vals))

        if train_progress.enabled:
            train_progress.update(
                ep * steps_per_epoch,
                suffix=f"epoch={ep}/{epochs} loss={running / max(n_batches, 1):.6f} validating",
                force=True,
            )
        epoch_eval_loader = (
            dl_overfit_eval
            if dl_overfit_eval is not None
            else (
                dl_stage2_epoch_eval
                if dl_stage2_epoch_eval is not None
                else dl_va
            )
        )
        epoch_eval_start = (
            0
            if dl_overfit_eval is not None
            else int(stage2_epoch_eval_start)
        )
        val_loss_k, val_mse_k, val_mae_k, _, _, _, _, _ = eval_loop_with_history(
            model, gate, lambda_kp_at(ep, detach=True),
            penalty_names, penalty_fns,
            epoch_eval_loader, cluster_id_c, K, moe_cfg, device,
            select_ranks=select_ranks,
            collect_plot=False, channel_count=C,
            collect_samples=False,
            mse_weight=mse_weight,
            gate_entropy_weight=gate_entropy_weight,
            gate_balance_weight=gate_balance_weight,
            gate_soft_weight=gate_soft_weight,
            gate_entropy_target_frac=gate_entropy_target_frac,
            penalty_scale=penalty_scale,
            dynamic_lambda=dynamic_lambda,
            lambda_min_kp=lambda_min_kp,
            mae_objective_weight=mae_objective_weight_at(ep),
            mae_objective_kind=mae_objective_kind,
            mae_objective_beta=mae_objective_beta,
            pred_residual=pred_residual,
            eval_start=epoch_eval_start,
        )
        semantic_epoch_audit = None
        if pred_residual_semantic_bank_stage1:
            semantic_epoch_audit = collect_semantic_bank_candidate_audit(
                dl_va,
                eval_start=val_eval_start,
                split_name="val",
                split_window_count=len(dva),
            )
            if pred_residual is None:
                raise RuntimeError("semantic bank audit requires pred_residual.")
            per_penalty_audit = semantic_epoch_audit.get("per_penalty", {})
            epoch_record = {
                "epoch": int(ep),
                "reporting_mean_only": True,
                "active_penalty": pred_residual_semantic_active_penalty,
                "raw_gradient_accumulation": {
                    "enabled": bool(
                        pred_residual_semantic_raw_gradient_accumulation
                    ),
                    "mode": (
                        "disjoint_level_groups_mean_then_independent_clip"
                        if pred_residual_semantic_raw_gradient_accumulation
                        and pred_residual_semantic_level_separate_need_gate
                        else "raw_gradient_mean_then_clip"
                        if pred_residual_semantic_raw_gradient_accumulation
                        else "legacy_per_batch_clip_step"
                    ),
                    "target_microbatches": int(
                        pred_residual_semantic_raw_gradient_microbatches
                    ),
                    "microbatch_count": int(semantic_batch_count),
                    "optimizer_step_count": int(semantic_step_count),
                    "group_counts": list(semantic_raw_gradient_group_counts),
                    "tail_microbatches": (
                        int(semantic_raw_gradient_group_counts[-1])
                        if semantic_raw_gradient_group_counts
                        and semantic_raw_gradient_group_counts[-1]
                        < pred_residual_semantic_raw_gradient_microbatches
                        else 0
                    ),
                },
                "per_penalty": {},
            }
            batch_denom = max(int(semantic_batch_count), 1)
            step_denom = max(int(semantic_step_count), 1)
            for p, name in enumerate(penalty_names):
                row = dict(per_penalty_audit.get(name, {}))
                is_active = p == semantic_bank_active_penalty_index
                runtime_checks = {
                    "active_body_only": bool(is_active),
                    "raw_gradient_l2_gt_0": bool(
                        is_active and float(semantic_grad_raw_sum_p[p].item()) > 0.0
                    ),
                    "clipped_gradient_l2_gt_0": bool(
                        is_active and float(semantic_grad_clipped_sum_p[p].item()) > 0.0
                    ),
                    "parameter_update_l2_gt_0": bool(
                        is_active and float(semantic_update_sum_p[p].item()) > 0.0
                    ),
                }
                row["runtime_gradient_checks"] = runtime_checks
                if is_active:
                    row["pass"] = bool(row.get("pass", False)) and all(
                        runtime_checks.values()
                    )
                    row["semantic_only_pass"] = bool(row["pass"])
                    selected = _update_semantic_bank_best_candidate(
                        semantic_bank_best_by_penalty,
                        penalty_index=p,
                        penalty_name=name,
                        epoch=ep,
                        acceptance=row,
                        body_state=pred_residual.get_penalty_body_state(p),
                    )
                    if selected:
                        semantic_bank_best_by_penalty[p][
                            "training_provenance"
                        ] = dict(pred_residual_training_provenance or {})
                        semantic_bank_best_by_penalty[p][
                            "optimizer_state_identity"
                        ] = semantic_bank_optimizer_identity_p[p]
                else:
                    row["pass"] = False
                    row["semantic_only_pass"] = False
                    row["inactive_not_eligible_for_acceptance"] = True
                    selected = False
                epoch_record["per_penalty"][name] = {
                    "active": bool(is_active),
                    "objective": float(semantic_loss_sum_p[p].item()) / batch_denom,
                    "components": {
                        component_name: float(component_sum_p[p].item()) / batch_denom
                        for component_name, component_sum_p in (
                            semantic_component_sum_by_name_p.items()
                        )
                    },
                    "raw_gradient_l2": float(semantic_grad_raw_sum_p[p].item()) / step_denom,
                    "clipped_gradient_l2": float(semantic_grad_clipped_sum_p[p].item()) / step_denom,
                    "parameter_update_l2": float(semantic_update_sum_p[p].item()) / step_denom,
                    "optimizer_state_identity": semantic_bank_optimizer_identity_p[p],
                    "checkpoint_candidate_selected": selected,
                    "acceptance": row,
                }
                if is_active and semantic_level_disjoint_metric_sums:
                    epoch_record["per_penalty"][name]["optimizer_groups"] = {
                        group: {
                            metric: float(value) / step_denom
                            for metric, value in metrics.items()
                        }
                        for group, metrics in semantic_level_disjoint_metric_sums.items()
                    }
            semantic_bank_independent_history.append(epoch_record)
            if pred_residual_semantic_raw_gradient_accumulation:
                active_row = epoch_record["per_penalty"][
                    pred_residual_semantic_active_penalty
                ]
                print(
                    "Semantic level optimizer audit: "
                    "mode="
                    + str(epoch_record["raw_gradient_accumulation"]["mode"])
                    + ", "
                    f"groups={semantic_raw_gradient_group_counts}, "
                    f"raw_norm={float(active_row['raw_gradient_l2']):.9g}, "
                    f"clipped_norm={float(active_row['clipped_gradient_l2']):.9g}, "
                    f"update={float(active_row['parameter_update_l2']):.9g}, "
                    f"optimizer_groups={active_row.get('optimizer_groups', {})}"
                )
        train_loss_k = train_loss_sum_k / max(train_cnt, 1)
        train_mse_k = train_mse_sum_k / max(train_cnt, 1)
        train_mae_k = train_mae_sum_k / max(train_cnt, 1)
        mse_gate_diag_den_k = mse_gate_diag_count_k.clamp_min(1.0)
        if bool((mse_gate_diag_count_k > 0.0).any().item()):
            mse_gate_train_diag_history.append(
                {
                    "epoch": int(ep),
                    "weight": float(mse_utility_gate_weight),
                    "min_gain": float(mse_utility_gate_min_gain),
                    "mae_weight": float(mse_utility_gate_mae_weight),
                    "target_mode": str(mse_utility_gate_target_mode),
                    "include_skip": bool(mse_utility_gate_include_skip),
                    "per_cluster": [
                        {
                            "cluster_id": int(k),
                            "samples": float(mse_gate_diag_count_k[k].detach().cpu().item()),
                            "mean_loss": float((mse_gate_loss_sum_k[k] / mse_gate_diag_den_k[k]).detach().cpu().item()),
                            "valid_rate": float((mse_gate_valid_sum_k[k] / mse_gate_diag_den_k[k]).detach().cpu().item()),
                            "skip_target_rate": float((mse_gate_skip_target_sum_k[k] / mse_gate_diag_den_k[k]).detach().cpu().item()),
                            "mean_skip_prob": float((mse_gate_skip_prob_sum_k[k] / mse_gate_diag_den_k[k]).detach().cpu().item()),
                            "mean_best_allowed_gain": float((mse_gate_best_gain_sum_k[k] / mse_gate_diag_den_k[k]).detach().cpu().item()),
                        }
                        for k in range(K)
                    ],
                }
            )
        train_mse_hist.append(train_mse_k.detach().cpu())
        val_mse_hist.append(val_mse_k.detach().cpu())
        update_swa_averagers(ep)

        monitor_k = _select_monitor_k(train_loss_k, train_mse_k, train_mae_k, val_loss_k, val_mse_k, val_mae_k)
        selection_active = (
            ep >= selection_start_epoch
            and not pred_residual_semantic_bank_stage1
        )
        if shared_moe_across_clusters and selection_active:
            shared_monitor = float(reduce_cluster_metric(monitor_k, cluster_weight_k).item())
            shared_dual_safe = True
            if (
                patch_router_epoch0_noop_enable
                and patch_router_epoch0_require_dual
            ):
                assert (
                    patch_router_epoch0_val_mse_k is not None
                    and patch_router_epoch0_val_mae_k is not None
                )
                epoch0_mse = float(
                    reduce_cluster_metric(
                        patch_router_epoch0_val_mse_k,
                        cluster_weight_k,
                    ).item()
                )
                epoch0_mae = float(
                    reduce_cluster_metric(
                        patch_router_epoch0_val_mae_k,
                        cluster_weight_k,
                    ).item()
                )
                current_mse = float(
                    reduce_cluster_metric(val_mse_k, cluster_weight_k).item()
                )
                current_mae = float(
                    reduce_cluster_metric(val_mae_k, cluster_weight_k).item()
                )
                shared_dual_safe = bool(
                    (epoch0_mse - current_mse) > min_delta
                    and (epoch0_mae - current_mae) > min_delta
                )
            if (
                shared_dual_safe
                and (shared_moe_best_monitor - shared_monitor) > min_delta
            ):
                shared_moe_best_monitor = shared_monitor
                shared_moe_best_epoch = int(ep)
                shared_moe_best_state["gate"] = gate.get_cluster_state(0)
                shared_moe_best_state["pred_residual"] = (
                    pred_residual.get_cluster_state(0) if pred_residual is not None else None
                )
        improved = (best_monitor - monitor_k) > min_delta if selection_active else torch.zeros_like(stopped)
        if (
            selection_active
            and patch_router_epoch0_noop_enable
            and patch_router_epoch0_require_dual
        ):
            assert (
                patch_router_epoch0_val_mse_k is not None
                and patch_router_epoch0_val_mae_k is not None
            )
            improved = improved & (
                (patch_router_epoch0_val_mse_k - val_mse_k) > min_delta
            ) & (
                (patch_router_epoch0_val_mae_k - val_mae_k) > min_delta
            )
        for k in range(K):
            if stopped[k]:
                continue
            if improved[k]:
                best_monitor[k] = monitor_k[k]
                bad_cnt[k] = 0
                save_best(k, ep)
            else:
                if ep < early_stop_start_epoch or not selection_active:
                    continue
                bad_cnt[k] += 1
                if bad_cnt[k] >= patience:
                    stopped[k] = True

        if schedulers is not None and ep > lr_warmup_epochs:
            shared_monitor_value = float(reduce_cluster_metric(monitor_k, cluster_weight_k).item())
            for k, sched in enumerate(schedulers):
                if sched is None:
                    continue
                if not _optimizer_slot_active(stopped, k, shared_moe=shared_moe_across_clusters):
                    continue
                if sched_name in {"plateau", "reduce", "reduce_on_plateau"}:
                    metric_value = (
                        shared_monitor_value
                        if shared_moe_across_clusters and int(k) == 0
                        else float(monitor_k[k].item())
                    )
                    sched.step(metric_value)
                else:
                    sched.step()
        train_loss_agg = float(reduce_cluster_metric(train_loss_k, cluster_weight_k).item())
        val_loss_agg = float(reduce_cluster_metric(val_loss_k, cluster_weight_k).item())
        if stage2_loss_audit_enable:
            epoch_loss_summary = _stage2_loss_epoch_summary(
                epoch=ep,
                count=train_cnt,
                cluster_weight_k=cluster_weight_k,
                total_loss_sum_k=stage2_total_loss_sum_k,
                forecast_loss_sum_k=stage2_forecast_loss_sum_k,
                penalty_loss_sum_k=stage2_penalty_loss_sum_k,
                pred_residual_aux_loss_sum_k=stage2_pred_residual_aux_loss_sum_k,
                candidate_supervision_loss_sum_k=stage2_candidate_supervision_loss_sum_k,
                gate_utility_loss_sum_k=stage2_gate_utility_loss_sum_k,
                skip_noop_loss_sum_k=stage2_skip_noop_loss_sum_k,
                intervention_supervision_loss_sum_k=stage2_intervention_supervision_loss_sum_k,
                other_aux_loss_sum_k=stage2_other_aux_loss_sum_k,
                train_mse_sum_k=train_mse_sum_k,
                train_mae_sum_k=train_mae_sum_k,
            )
            route_summary = _stage2_route_epoch_summary(
                penalty_names=penalty_names,
                cluster_weight_k=cluster_weight_k,
                route_count_k=stage2_route_count_k,
                route_prob_sum_kp=stage2_route_prob_sum_kp,
                route_actual_sum_kp=stage2_route_actual_sum_kp,
                route_entropy_sum_k=stage2_route_entropy_sum_k,
                skip_prob_sum_k=stage2_skip_prob_sum_k,
                skip_active_sum_k=stage2_skip_active_sum_k,
            )
            grad_den = max(int(stage2_grad_norm_batches), 1)
            epoch_loss_summary["route"] = route_summary
            epoch_loss_summary["gradient_l2_mean"] = {
                name: float(value / grad_den) for name, value in stage2_grad_norm_sum.items()
            }
            epoch_loss_summary["val_loss"] = val_loss_agg
            epoch_loss_summary["val_mse"] = float(reduce_cluster_metric(val_mse_k, cluster_weight_k).item())
            epoch_loss_summary["val_mae"] = float(reduce_cluster_metric(val_mae_k, cluster_weight_k).item())
            stage2_loss_audit_history.append(epoch_loss_summary)
        if (
            stage2_route_audit_enable
            and pred_residual is not None
            and moe_enable
            and P > 0
            and (int(ep) % int(stage2_route_audit_frequency) == 0)
        ):
            route_audit_max_batches = int(stage2_route_audit_cfg.get("max_batches", 0))
            route_audit_feature_mode = str(stage2_route_audit_cfg.get("feature_mode", "base"))
            route_audit_thresholds = _stage2_route_audit_thresholds(
                stage2_route_audit_cfg=stage2_route_audit_cfg,
                route_ce_min_abs_improvement=route_ce_min_abs_improvement,
                route_ce_min_rel_improvement=route_ce_min_rel_improvement,
                route_ce_min_candidate_delta_rms=route_ce_min_candidate_delta_rms,
                binary_adoption_weight=binary_adoption_weight,
                binary_adoption_min_abs_improvement=binary_adoption_min_abs_improvement,
                binary_adoption_min_rel_improvement=binary_adoption_min_rel_improvement,
                binary_adoption_min_candidate_delta_rms=binary_adoption_min_candidate_delta_rms,
            )
            prior_for_route_audit = (
                cluster_penalty_prior_prob_kp
                if cluster_penalty_prior_prob_kp is not None
                else gate_prior_prob_kp
            )
            epoch_route_audit: Dict[str, object] = {
                "epoch": int(ep),
                "max_batches": int(route_audit_max_batches),
                "splits": {},
                "val_eval_mse": float(reduce_cluster_metric(val_mse_k, cluster_weight_k).item()),
                "val_eval_mae": float(reduce_cluster_metric(val_mae_k, cluster_weight_k).item()),
                "selected_scaled_eval_mse": None,
                "selected_scaled_eval_mae": None,
                "selected_scaled_note": (
                    "Per-epoch selected/scaled channel-selection metrics are not computed in this hook; "
                    "final selected/scaled metrics are reported after residual selection."
                ),
                "label_thresholds": route_audit_thresholds,
            }
            for split_name, split_loader in stage2_route_audit_loaders.items():
                route_tensors = _collect_penalty_route_learnability_tensors(
                    model=model,
                    gate=gate,
                    pred_residual=pred_residual,
                    loader=split_loader,
                    cluster_id_c=cluster_id_c,
                    K=K,
                    moe_cfg=moe_cfg,
                    device=device,
                    penalty_names=penalty_names,
                    penalty_fns=penalty_fns,
                    penalty_scale=penalty_scale,
                    select_ranks=select_ranks,
                    gate_soft_weight=gate_soft_weight,
                    split_name=split_name,
                    feature_mode=route_audit_feature_mode,
                    allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                    min_abs_improvement=float(route_audit_thresholds["min_abs_improvement"]),
                    min_rel_improvement=float(route_audit_thresholds["min_rel_improvement"]),
                    min_candidate_delta_rms=float(route_audit_thresholds["min_candidate_delta_rms"]),
                    max_batches=route_audit_max_batches,
                    history_anchor_cfg=history_anchor_cfg,
                    observed_history_tc=data_window_tc,
                    input_len=L,
                    eval_start=int(stage2_route_audit_eval_starts.get(split_name, 0)),
                    model_train_stat_adapter_pc=model_train_stat_adapter_pc,
                    model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
                    train_stat_anchor_pc=train_stat_anchor_pc,
                    train_residual_anchor_phc=train_residual_anchor_phc,
                    learnable_output_anchor=learnable_output_anchor,
                    gate_feature_mode=gate_feature_mode,
                )
                if route_tensors is None:
                    continue
                explain_payload = evaluate_penalty_explainability(
                    model=model,
                    gate=gate,
                    pred_residual=pred_residual,
                    loader=split_loader,
                    cluster_id_c=cluster_id_c,
                    K=K,
                    moe_cfg=moe_cfg,
                    device=device,
                    penalty_names=penalty_names,
                    penalty_fns=penalty_fns,
                    penalty_scale=penalty_scale,
                    select_ranks=select_ranks,
                    gate_soft_weight=gate_soft_weight,
                    split_name=split_name,
                    penalty_portrait_kp=penalty_portrait_kp,
                    prior_prob_kp=prior_for_route_audit,
                    allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                    max_batches=route_audit_max_batches,
                    history_anchor_cfg=history_anchor_cfg,
                    observed_history_tc=data_window_tc,
                    input_len=L,
                    eval_start=int(stage2_route_audit_eval_starts.get(split_name, 0)),
                    model_train_stat_adapter_pc=model_train_stat_adapter_pc,
                    model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
                    train_stat_anchor_pc=train_stat_anchor_pc,
                    train_residual_anchor_phc=train_residual_anchor_phc,
                    learnable_output_anchor=learnable_output_anchor,
                    gate_feature_mode=gate_feature_mode,
                )
                split_summary = _route_audit_summary_from_tensors(
                    tensors=route_tensors,
                    explainability=explain_payload,
                )
                cast_splits = epoch_route_audit["splits"]
                assert isinstance(cast_splits, dict)
                cast_splits[split_name] = split_summary
            stage2_route_audit_history.append(epoch_route_audit)
        progress_suffix = (
            f"epoch={ep}/{epochs} loss={train_loss_agg:.6f} val_loss={val_loss_agg:.6f}"
        )
        if outer_loss_epoch is not None:
            progress_suffix += f" lambda_loss={outer_loss_epoch:.6f}"
        if train_progress.enabled:
            train_progress.update(ep * steps_per_epoch, suffix=progress_suffix, force=True)
        else:
            msg = (
                f"[Epoch {ep:03d}] loss={train_loss_agg:.6f} | "
                f"val_loss={val_loss_agg:.6f}"
            )
            if outer_loss_epoch is not None:
                msg += f" | lambda_loss={outer_loss_epoch:.6f}"
            print(msg)

        if (
            dl_overfit_eval is not None
            and ep in overfit_diagnostic_metric_epochs
            and pred_residual is not None
            and getattr(pred_residual, "patch_router", None) is not None
        ):
            fit_summary = collect_pred_residual_summary(dl_overfit_eval, eval_start=0)
            fit_oracle = (fit_summary.get("patch_router", {}) or {}).get(
                "oracle_diagnostic",
                {},
            ) or {}
            fit_selected_rates = fit_oracle.get("selected_class_rate", {}) or {}
            fit_risk_tensors = collect_patch_risk_calibration_tensors(
                dl_overfit_eval,
                eval_start=0,
            )
            fixed_penalty_c = (
                pred_residual.patch_router.fixed_penalty_index_by_channel_c
            )
            active_channel_c = torch.ones(C, dtype=torch.bool)
            if int(fixed_penalty_c.numel()) == C:
                active_channel_c = fixed_penalty_c.detach().cpu() >= 0
            fit_score = fit_risk_tensors["score"][:, active_channel_c].reshape(-1)
            fit_gain = fit_risk_tensors["gain"][:, active_channel_c].reshape(-1)
            score_ordering = _select_recall_constrained_risk_threshold(
                score_n=fit_score,
                gain_n=fit_gain,
                block_n=torch.zeros_like(fit_gain, dtype=torch.long),
                min_gain_cost_ratio=1.0,
                min_block_net_gain=0.0,
            )
            fit_metrics = {
                "epoch": int(ep),
                "train_loss": float(train_loss_agg),
                "selection_loss": float(val_loss_agg),
                "selected_gain_pct": float(fit_oracle.get("selected_gain_pct", 0.0)),
                "oracle_gain_pct": float(fit_oracle.get("oracle_gain_pct", 0.0)),
                "proposal_oracle_best_recall_at_k": float(
                    fit_oracle.get("proposal_oracle_best_recall_at_k", 0.0)
                ),
                "shortlist_pairwise_accuracy": float(
                    fit_oracle.get("shortlist_pairwise_accuracy", 0.0)
                ),
                "risk_sign_recall": float(fit_oracle.get("risk_sign_recall", 0.0)),
                "risk_sign_precision": float(fit_oracle.get("risk_sign_precision", 0.0)),
                "risk_sign_accuracy": float(fit_oracle.get("risk_sign_accuracy", 0.0)),
                "risk_sign_predicted_positive_rate": float(
                    fit_oracle.get("risk_sign_predicted_positive_rate", 0.0)
                ),
                "selected_utility_recall": float(
                    fit_oracle.get("selected_utility_recall", 0.0)
                ),
                "selected_utility_precision": float(
                    fit_oracle.get("selected_utility_precision", 0.0)
                ),
                "selected_gain_to_cost_ratio": float(
                    fit_oracle.get("selected_gain_to_cost_ratio", 0.0)
                ),
                "skip_rate": float(fit_selected_rates.get("skip", 0.0)),
                "score_ordering_nonnegative": score_ordering,
            }
            overfit_diagnostic_history.append(fit_metrics)
            print(
                "  Gate-overfit fit metrics: "
                f"proposal_r@k={fit_metrics['proposal_oracle_best_recall_at_k']:.4f}, "
                f"pair_acc={fit_metrics['shortlist_pairwise_accuracy']:.4f}, "
                f"risk_recall={fit_metrics['risk_sign_recall']:.4f}, "
                f"utility_recall={fit_metrics['selected_utility_recall']:.4f}, "
                f"utility_precision={fit_metrics['selected_utility_precision']:.4f}, "
                f"ordered_recall={score_ordering['positive_recall']:.4f}, "
                f"gain={fit_metrics['selected_gain_pct']:.4f}%"
            )

        epoch_times.append(time.perf_counter() - t_ep0)
        if stopped.all():
            early_stopped = True
            if not train_progress.enabled:
                print("All clusters early-stopped.")
            break

    train_progress.finish(
        current=min(len(epoch_times) * steps_per_epoch, train_progress.total),
        suffix="early stopped" if early_stopped else "done",
    )

    plot_cfg = cfg.get("plot", {}) or {}
    if bool(plot_cfg.get("save_loss_curves", False)):
        loss_dir = os.path.join(out_dir, "loss_curves")
        save_cluster_metric_curves(
            out_dir=loss_dir,
            train_metric_hist=train_mse_hist,
            val_metric_hist=val_mse_hist,
            metric_name="mse",
            dpi=int(plot_cfg.get("dpi", 140)),
        )
        print(f"Saved MSE curves to: {loss_dir}")

    checkpoint_selection = str(memory_cfg.get("checkpoint_selection", "best")).lower()
    if checkpoint_selection not in {"best", "last", "semantic_per_expert"}:
        raise ValueError(
            "memory.checkpoint_selection must be best, last, or semantic_per_expert."
        )
    semantic_bank_release_ready = False
    semantic_bank_stage_complete = False
    semantic_bank_accepted_penalties: List[str] = []
    if pred_residual_semantic_bank_stage1:
        if checkpoint_selection != "semantic_per_expert":
            raise ValueError(
                "semantic_bank_stage1 requires checkpoint_selection=semantic_per_expert."
            )
        if pred_residual is None:
            raise RuntimeError("semantic bank checkpoint selection requires pred_residual.")
        semantic_bank_release_ready = _semantic_bank_release_ready(
            semantic_bank_best_by_penalty,
            penalty_names,
        )
        semantic_bank_accepted_penalties = [
            str(item["penalty"])
            for item in semantic_bank_best_by_penalty
            if isinstance(item.get("state"), dict)
            and bool((item.get("acceptance") or {}).get("pass", False))
        ]
        active_item = semantic_bank_best_by_penalty[
            int(semantic_bank_active_penalty_index)
        ]
        semantic_bank_stage_complete = bool(
            isinstance(active_item.get("state"), dict)
            and bool((active_item.get("acceptance") or {}).get("pass", False))
        )
        if semantic_bank_release_ready:
            for p, item in enumerate(semantic_bank_best_by_penalty):
                actual_body_hash = _semantic_penalty_body_state_sha256(item["state"])
                if item.get("body_sha256") != actual_body_hash:
                    raise RuntimeError(
                        "Semantic bank selected body hash mismatch before assembly: "
                        f"penalty={penalty_names[p]}, metadata={item.get('body_sha256')}, "
                        f"actual={actual_body_hash}"
                    )
            pred_residual.load_penalty_body_states([
                item["state"] for item in semantic_bank_best_by_penalty
            ])
            for p, item in enumerate(semantic_bank_best_by_penalty):
                assembled_hash = _semantic_penalty_body_state_sha256(
                    pred_residual.get_penalty_body_state(p)
                )
                if assembled_hash != item["body_sha256"]:
                    raise RuntimeError(
                        "Semantic bank assembled body hash mismatch: "
                        f"penalty={penalty_names[p]}, selected={item['body_sha256']}, "
                        f"assembled={assembled_hash}"
                    )
            print(
                "Semantic bank assembled from independently accepted expert checkpoints: "
                + ", ".join(
                    f"{item['penalty']}@epoch{item['epoch']}"
                    for item in semantic_bank_best_by_penalty
                )
            )
        elif semantic_bank_stage_complete:
            pred_residual.load_penalty_body_state(
                int(semantic_bank_active_penalty_index),
                active_item["state"],
            )
            assembled_hash = _semantic_penalty_body_state_sha256(
                pred_residual.get_penalty_body_state(
                    int(semantic_bank_active_penalty_index)
                )
            )
            if assembled_hash != active_item["body_sha256"]:
                raise RuntimeError(
                    "Semantic partial checkpoint active-body hash mismatch: "
                    f"penalty={pred_residual_semantic_active_penalty}."
                )
            print(
                "Semantic staged checkpoint accepted active expert: "
                f"{pred_residual_semantic_active_penalty}@epoch{active_item['epoch']}"
            )
        else:
            print(
                "Semantic bank release blocked: at least one expert has no independently "
                "accepted checkpoint candidate."
            )
    elif checkpoint_selection == "best":
        load_best_all()
    else:
        print("Checkpoint selection uses the final epoch state (adapter-bank training mode).")
    if patch_router_epoch0_noop_summary is not None:
        selected_epoch_payload: object = (
            int(shared_moe_best_epoch)
            if shared_moe_across_clusters
            else [int(v) for v in best_epoch.detach().cpu().tolist()]
        )
        retained_noop = bool(
            int(shared_moe_best_epoch) == 0
            if shared_moe_across_clusters
            else bool((best_epoch == 0).all().item())
        )
        patch_router_epoch0_noop_summary.update(
            {
                "selected_epoch": selected_epoch_payload,
                "retained_noop": retained_noop,
                "selection_reason": (
                    "no_learned_epoch_dual_improved_validation"
                    if retained_noop
                    else "learned_epoch_dual_improved_validation"
                ),
            }
        )
    assert_pairwise_frozen_parameters_unchanged("post_load_best")
    swa_summary["updates"] = int(swa_updates)
    if swa_enable and swa_updates <= 0:
        swa_summary["reason"] = "no_swa_updates"
    if swa_enable and swa_updates > 0:
        swa_mae_eval_weight = _scale_mae_objective_weight(
            mae_objective_weight_final if mae_objective_enable else 0.0,
            mae_objective_multiplier_k,
        )
        lam_kp_for_swa_eval = lambda_kp_from_epochs(best_epoch)
        best_val_loss_k, best_val_mse_k, best_val_mae_k, _, _, _, _, _ = eval_loop_with_history(
            model, gate, lam_kp_for_swa_eval,
            penalty_names, penalty_fns,
            dl_va, cluster_id_c, K, moe_cfg, device,
            select_ranks=select_ranks,
            collect_plot=False, channel_count=C,
            mse_weight=mse_weight,
            gate_entropy_weight=gate_entropy_weight,
            gate_balance_weight=gate_balance_weight,
            gate_soft_weight=gate_soft_weight,
            gate_entropy_target_frac=gate_entropy_target_frac,
            penalty_scale=penalty_scale,
            dynamic_lambda=dynamic_lambda,
            lambda_min_kp=lambda_min_kp,
            mae_objective_weight=swa_mae_eval_weight,
            mae_objective_kind=mae_objective_kind,
            mae_objective_beta=mae_objective_beta,
            pred_residual=pred_residual,
            eval_start=val_eval_start,
        )
        best_swa_metric = _aggregate_val_metric(
            best_val_loss_k,
            best_val_mse_k,
            best_val_mae_k,
            swa_selection_metric,
        )
        load_swa_averagers()
        swa_val_loss_k, swa_val_mse_k, swa_val_mae_k, _, _, _, _, _ = eval_loop_with_history(
            model, gate, lam_kp_for_swa_eval,
            penalty_names, penalty_fns,
            dl_va, cluster_id_c, K, moe_cfg, device,
            select_ranks=select_ranks,
            collect_plot=False, channel_count=C,
            mse_weight=mse_weight,
            gate_entropy_weight=gate_entropy_weight,
            gate_balance_weight=gate_balance_weight,
            gate_soft_weight=gate_soft_weight,
            gate_entropy_target_frac=gate_entropy_target_frac,
            penalty_scale=penalty_scale,
            dynamic_lambda=dynamic_lambda,
            lambda_min_kp=lambda_min_kp,
            mae_objective_weight=swa_mae_eval_weight,
            mae_objective_kind=mae_objective_kind,
            mae_objective_beta=mae_objective_beta,
            pred_residual=pred_residual,
            eval_start=val_eval_start,
        )
        swa_metric = _aggregate_val_metric(
            swa_val_loss_k,
            swa_val_mse_k,
            swa_val_mae_k,
            swa_selection_metric,
        )
        use_swa = (best_swa_metric - swa_metric) > swa_min_delta
        if not use_swa:
            load_best_all()
        swa_summary.update(
            {
                "selected": bool(use_swa),
                "best_metric": float(best_swa_metric),
                "swa_metric": float(swa_metric),
                "min_delta": float(swa_min_delta),
            }
        )
        print(
            "SWA selection: "
            f"updates={swa_updates}, metric={swa_selection_metric}, "
            f"best={best_swa_metric:.6f}, swa={swa_metric:.6f}, "
            f"selected={bool(use_swa)}"
        )
    def _run_train_stat_anchor_scale_selection() -> bool:
        train_stat_scale_selection_cfg = train_stat_anchor_cfg.get("scale_selection", {}) or {}
        if not (
            bool(train_stat_anchor_cfg.get("enable", False))
            and bool(train_stat_scale_selection_cfg.get("enable", False))
            and train_stat_anchor_pc is not None
        ):
            return False
        horizon_segments = int(train_stat_scale_selection_cfg.get("horizon_segments", 1))
        scales_c, scores_c, selection_count = select_train_stat_anchor_scales_from_loader(
            model=model,
            loader=dl_va,
            cluster_id_c=cluster_id_c,
            device=device,
            history_anchor_cfg=history_anchor_cfg,
            observed_history_tc=data_window_tc,
            input_len=L,
            eval_start=val_eval_start,
            stat_anchor_pc=train_stat_anchor_pc,
            train_stat_anchor_cfg=train_stat_anchor_cfg,
            metric=str(train_stat_scale_selection_cfg.get("metric", "mse")),
            max_scale=float(train_stat_scale_selection_cfg.get("max_scale", 0.3)),
            steps=int(train_stat_scale_selection_cfg.get("steps", 13)),
            horizon_segments=horizon_segments,
            model_train_stat_adapter_pc=model_train_stat_adapter_pc,
            model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
        )
        if int(scales_c.ndim) == 2:
            alpha_values = [[float(v) for v in row] for row in scales_c.tolist()]
            train_stat_anchor_cfg["alpha_by_channel_horizon"] = alpha_values
            train_stat_anchor_cfg["alpha_horizon_segments"] = int(horizon_segments)
            alpha_key = "alpha_by_channel_horizon"
        else:
            alpha_values = [float(v) for v in scales_c.tolist()]
            train_stat_anchor_cfg["alpha_by_channel"] = alpha_values
            alpha_key = "alpha_by_channel"
        train_stat_anchor_summary["scale_selection"] = {
            "enable": True,
            "source_split": "val",
            "metric": str(train_stat_scale_selection_cfg.get("metric", "mse")),
            "max_scale": float(train_stat_scale_selection_cfg.get("max_scale", 0.3)),
            "steps": int(train_stat_scale_selection_cfg.get("steps", 13)),
            "horizon_segments": int(horizon_segments),
            "num_windows": int(selection_count),
            alpha_key: alpha_values,
            "score": [[float(v) for v in row] for row in scores_c.tolist()]
            if int(scores_c.ndim) == 2
            else [float(v) for v in scores_c.tolist()],
            "mean_alpha": float(scales_c.mean().item()) if int(scales_c.numel()) > 0 else 0.0,
        }
        print(
            "Train-stat anchor scale selection: "
            f"metric={train_stat_anchor_summary['scale_selection']['metric']}, "
            f"mean_alpha={train_stat_anchor_summary['scale_selection']['mean_alpha']:.4f}, "
            f"horizon_segments={int(horizon_segments)}, channels={int(scales_c.shape[0])}"
        )
        return True

    def _run_model_train_stat_adapter_scale_selection() -> bool:
        model_train_stat_scale_selection_cfg = model_train_stat_adapter_cfg.get("scale_selection", {}) or {}
        if not (
            bool(model_train_stat_adapter_cfg.get("enable", False))
            and bool(model_train_stat_scale_selection_cfg.get("enable", False))
            and model_train_stat_adapter_pc is not None
        ):
            return False
        horizon_segments = int(model_train_stat_scale_selection_cfg.get("horizon_segments", 1))
        scales_c, scores_c, selection_count = select_train_stat_anchor_scales_from_loader(
            model=model,
            loader=dl_va,
            cluster_id_c=cluster_id_c,
            device=device,
            history_anchor_cfg=history_anchor_cfg,
            observed_history_tc=data_window_tc,
            input_len=L,
            eval_start=val_eval_start,
            stat_anchor_pc=model_train_stat_adapter_pc,
            train_stat_anchor_cfg=model_train_stat_adapter_cfg,
            metric=str(model_train_stat_scale_selection_cfg.get("metric", "mse")),
            max_scale=float(model_train_stat_scale_selection_cfg.get("max_scale", 0.3)),
            steps=int(model_train_stat_scale_selection_cfg.get("steps", 13)),
            horizon_segments=horizon_segments,
        )
        if int(scales_c.ndim) == 2:
            alpha_values = [[float(v) for v in row] for row in scales_c.tolist()]
            model_train_stat_adapter_cfg["alpha_by_channel_horizon"] = alpha_values
            model_train_stat_adapter_cfg["alpha_horizon_segments"] = int(horizon_segments)
            alpha_key = "alpha_by_channel_horizon"
        else:
            alpha_values = [float(v) for v in scales_c.tolist()]
            model_train_stat_adapter_cfg["alpha_by_channel"] = alpha_values
            alpha_key = "alpha_by_channel"
        model_train_stat_adapter_summary["scale_selection"] = {
            "enable": True,
            "source_split": "val",
            "metric": str(model_train_stat_scale_selection_cfg.get("metric", "mse")),
            "max_scale": float(model_train_stat_scale_selection_cfg.get("max_scale", 0.3)),
            "steps": int(model_train_stat_scale_selection_cfg.get("steps", 13)),
            "horizon_segments": int(horizon_segments),
            "num_windows": int(selection_count),
            alpha_key: alpha_values,
            "score": [[float(v) for v in row] for row in scores_c.tolist()]
            if int(scores_c.ndim) == 2
            else [float(v) for v in scores_c.tolist()],
            "mean_alpha": float(scales_c.mean().item()) if int(scales_c.numel()) > 0 else 0.0,
        }
        print(
            "Model train-stat adapter scale selection: "
            f"metric={model_train_stat_adapter_summary['scale_selection']['metric']}, "
            f"mean_alpha={model_train_stat_adapter_summary['scale_selection']['mean_alpha']:.4f}, "
            f"horizon_segments={int(horizon_segments)}, channels={int(scales_c.shape[0])}"
        )
        return True

    _run_model_train_stat_adapter_scale_selection()
    train_stat_scale_selection_done = _run_train_stat_anchor_scale_selection()

    train_residual_scale_selection_cfg = train_residual_anchor_cfg.get("scale_selection", {}) or {}
    if bool(train_residual_anchor_cfg.get("enable", False)):
        train_residual_anchor_period = int(train_residual_anchor_cfg.get("period", 96))
        train_residual_anchor_phc, train_residual_anchor_counts, residual_train_count = (
            build_train_residual_anchor_table_from_loader(
                model=model,
                loader=dl_tr_source,
                cluster_id_c=cluster_id_c,
                device=device,
                history_anchor_cfg=history_anchor_cfg,
                observed_history_tc=data_window_tc,
                input_len=L,
                eval_start=0,
                period=train_residual_anchor_period,
                model_train_stat_adapter_pc=model_train_stat_adapter_pc,
                model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
                train_stat_anchor_pc=train_stat_anchor_pc,
                train_stat_anchor_cfg=train_stat_anchor_cfg,
            )
        )
        train_residual_anchor_summary.update(
            {
                "period": int(train_residual_anchor_period),
                "source_split": "train",
                "train_windows": int(residual_train_count),
                "min_count": int(train_residual_anchor_counts.min().item()),
                "max_count": int(train_residual_anchor_counts.max().item()),
                "alpha": float(train_residual_anchor_cfg.get("alpha", 0.0) or 0.0),
                "blend_target": str(train_residual_anchor_cfg.get("blend_target", "prediction")),
            }
        )
        print(
            "Train residual anchor expert enabled: "
            f"period={train_residual_anchor_period}, "
            f"alpha={float(train_residual_anchor_cfg.get('alpha', 0.0) or 0.0):.3f}, "
            f"train_windows={int(residual_train_count)}"
        )
    if (
        bool(train_residual_anchor_cfg.get("enable", False))
        and bool(train_residual_scale_selection_cfg.get("enable", False))
        and train_residual_anchor_phc is not None
    ):
        horizon_segments = int(train_residual_scale_selection_cfg.get("horizon_segments", 1))
        scales_c, scores_c, selection_count = select_train_residual_anchor_scales_from_loader(
            model=model,
            loader=dl_va,
            cluster_id_c=cluster_id_c,
            device=device,
            history_anchor_cfg=history_anchor_cfg,
            observed_history_tc=data_window_tc,
            input_len=L,
            eval_start=val_eval_start,
            residual_anchor_phc=train_residual_anchor_phc,
            train_residual_anchor_cfg=train_residual_anchor_cfg,
            metric=str(train_residual_scale_selection_cfg.get("metric", "mse")),
            max_scale=float(train_residual_scale_selection_cfg.get("max_scale", 0.5)),
            steps=int(train_residual_scale_selection_cfg.get("steps", 21)),
            horizon_segments=horizon_segments,
            model_train_stat_adapter_pc=model_train_stat_adapter_pc,
            model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
            train_stat_anchor_pc=train_stat_anchor_pc,
            train_stat_anchor_cfg=train_stat_anchor_cfg,
        )
        if int(scales_c.ndim) == 2:
            alpha_values = [[float(v) for v in row] for row in scales_c.tolist()]
            train_residual_anchor_cfg["alpha_by_channel_horizon"] = alpha_values
            train_residual_anchor_cfg["alpha_horizon_segments"] = int(horizon_segments)
            alpha_key = "alpha_by_channel_horizon"
        else:
            alpha_values = [float(v) for v in scales_c.tolist()]
            train_residual_anchor_cfg["alpha_by_channel"] = alpha_values
            alpha_key = "alpha_by_channel"
        train_residual_anchor_summary["scale_selection"] = {
            "enable": True,
            "source_split": "val",
            "metric": str(train_residual_scale_selection_cfg.get("metric", "mse")),
            "max_scale": float(train_residual_scale_selection_cfg.get("max_scale", 0.5)),
            "steps": int(train_residual_scale_selection_cfg.get("steps", 21)),
            "horizon_segments": int(horizon_segments),
            "num_windows": int(selection_count),
            alpha_key: alpha_values,
            "score_by_channel": scores_c.detach().cpu().tolist(),
            "mean_alpha": float(scales_c.mean().item()),
        }
        print(
            "Train residual anchor scale selection: "
            f"metric={train_residual_anchor_summary['scale_selection']['metric']}, "
            f"mean_alpha={train_residual_anchor_summary['scale_selection']['mean_alpha']:.4f}, "
            f"windows={int(selection_count)}"
        )

    train_stat_scale_selection_cfg = train_stat_anchor_cfg.get("scale_selection", {}) or {}
    if (
        not train_stat_scale_selection_done
        and bool(train_stat_anchor_cfg.get("enable", False))
        and bool(train_stat_scale_selection_cfg.get("enable", False))
        and train_stat_anchor_pc is not None
    ):
        horizon_segments = int(train_stat_scale_selection_cfg.get("horizon_segments", 1))
        scales_c, scores_c, selection_count = select_train_stat_anchor_scales_from_loader(
            model=model,
            loader=dl_va,
            cluster_id_c=cluster_id_c,
            device=device,
            history_anchor_cfg=history_anchor_cfg,
            observed_history_tc=data_window_tc,
            input_len=L,
            eval_start=val_eval_start,
            stat_anchor_pc=train_stat_anchor_pc,
            train_stat_anchor_cfg=train_stat_anchor_cfg,
            metric=str(train_stat_scale_selection_cfg.get("metric", "mse")),
            max_scale=float(train_stat_scale_selection_cfg.get("max_scale", 0.3)),
            steps=int(train_stat_scale_selection_cfg.get("steps", 13)),
            horizon_segments=horizon_segments,
            model_train_stat_adapter_pc=model_train_stat_adapter_pc,
            model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
        )
        if int(scales_c.ndim) == 2:
            alpha_values = [[float(v) for v in row] for row in scales_c.tolist()]
            train_stat_anchor_cfg["alpha_by_channel_horizon"] = alpha_values
            train_stat_anchor_cfg["alpha_horizon_segments"] = int(horizon_segments)
            alpha_key = "alpha_by_channel_horizon"
        else:
            alpha_values = [float(v) for v in scales_c.tolist()]
            train_stat_anchor_cfg["alpha_by_channel"] = alpha_values
            alpha_key = "alpha_by_channel"
        train_stat_anchor_summary["scale_selection"] = {
            "enable": True,
            "source_split": "val",
            "metric": str(train_stat_scale_selection_cfg.get("metric", "mse")),
            "max_scale": float(train_stat_scale_selection_cfg.get("max_scale", 0.3)),
            "steps": int(train_stat_scale_selection_cfg.get("steps", 13)),
            "horizon_segments": int(horizon_segments),
            "num_windows": int(selection_count),
            alpha_key: alpha_values,
            "score": [[float(v) for v in row] for row in scores_c.tolist()]
            if int(scores_c.ndim) == 2
            else [float(v) for v in scores_c.tolist()],
            "mean_alpha": float(scales_c.mean().item()) if int(scales_c.numel()) > 0 else 0.0,
        }
        print(
            "Train-stat anchor scale selection: "
            f"metric={train_stat_anchor_summary['scale_selection']['metric']}, "
            f"mean_alpha={train_stat_anchor_summary['scale_selection']['mean_alpha']:.4f}, "
            f"horizon_segments={int(horizon_segments)}, channels={int(scales_c.shape[0])}"
        )

    if bool(calendar_residual_cfg.get("enable", False)):
        source_split = str(calendar_residual_cfg.get("source_split", "train")).lower()
        if source_split not in {"train", "training"}:
            raise ValueError("calendar_residual.source_split must be 'train' for strict input96 experiments.")
        calendar_fit_target = str(calendar_residual_cfg.get("fit_target", "base_path")).lower()
        if calendar_fit_target in {"base", "base_path", "backbone"}:
            calendar_fit_loader = DataLoader(
                dtr,
                batch_size=int(cfg["train"]["batch_size"]),
                shuffle=False,
                num_workers=0,
                pin_memory=pin_mem,
            )
            calendar_residual_coef_cf, calendar_fit_summary = fit_calendar_residual_correction(
                model=model,
                loader=calendar_fit_loader,
                cluster_id_c=cluster_id_c,
                device=device,
                calendar_feature_tf=calendar_feature_tf,
                input_len=L,
                eval_start=0,
                cfg=calendar_residual_cfg,
                history_anchor_cfg=history_anchor_cfg,
                observed_history_tc=data_window_tc,
                model_train_stat_adapter_pc=model_train_stat_adapter_pc,
                model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
            )
            calendar_residual_summary.update(calendar_fit_summary)
            calendar_residual_summary["feature_names"] = list(calendar_feature_names)
            calendar_residual_summary["train_only"] = True
            if calendar_residual_coef_cf is not None:
                print(
                    "Calendar residual fitted: "
                    f"target=base_path, features={len(calendar_feature_names)}, "
                    f"fit_windows={calendar_residual_summary.get('fit_windows')}, "
                    f"coef_mean_abs={float(calendar_residual_summary.get('coef_mean_abs', 0.0)):.6f}"
                )
        elif calendar_fit_target in {"final", "final_eval", "final_eval_path", "eval_path"}:
            calendar_residual_summary["fit_target"] = "final_eval_path"
            calendar_residual_summary["pending_final_eval_path_fit"] = True
        else:
            raise ValueError(
                "calendar_residual.fit_target must be base_path or final_eval_path "
                f"(got {calendar_fit_target!r})."
            )

    pred_residual_confidence_summary = None
    if pred_residual_confidence_gate_enable and pred_residual is not None and P > 0:
        confidence_source_split = str(pred_residual_confidence_gate_source_split)
        confidence_source_range = (0, len(dtr))
        if confidence_source_split == "train_holdout":
            ranges = _explainability_train_subsplit_ranges(
                num_windows=len(dtr),
                holdout_fraction=pred_residual_confidence_gate_holdout_fraction,
            )
            if "train_holdout" in ranges:
                confidence_source_range = ranges["train_holdout"]
            else:
                confidence_source_split = "train"
        if len(dtr) <= 0:
            pred_residual_confidence_summary = {
                "enable": False,
                "reason": "empty_train_split",
                "source_requirement": "train_only",
            }
        else:
            start_i, end_i = confidence_source_range
            start_i = max(0, int(start_i))
            end_i = min(len(dtr), int(end_i))
            if end_i <= start_i:
                start_i, end_i = 0, len(dtr)
                confidence_source_split = "train"
            if confidence_source_split == "train" and start_i == 0 and end_i == len(dtr):
                confidence_loader = DataLoader(
                    dtr,
                    batch_size=int(cfg["train"]["batch_size"]),
                    shuffle=False,
                    num_workers=0,
                    pin_memory=pin_mem,
                )
            else:
                confidence_loader = DataLoader(
                    Subset(dtr, range(start_i, end_i)),
                    batch_size=int(cfg["train"]["batch_size"]),
                    shuffle=False,
                    num_workers=0,
                    pin_memory=pin_mem,
                )

            threshold_raw = pred_residual_confidence_gate_threshold
            threshold_is_auto = str(threshold_raw).strip().lower() == "auto"
            if threshold_is_auto:
                confidence_tensors = _collect_pred_residual_selector_tensors(
                    model=model,
                    pred_residual=pred_residual,
                    loader=confidence_loader,
                    cluster_id_c=cluster_id_c,
                    K=K,
                    moe_cfg=moe_cfg,
                    device=device,
                    penalty_count=P,
                    pred_residual_scale_c=None,
                    history_anchor_cfg=history_anchor_cfg,
                    observed_history_tc=data_window_tc,
                    input_len=L,
                    eval_start=0,
                    model_train_stat_adapter_pc=model_train_stat_adapter_pc,
                    model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
                    train_stat_anchor_pc=train_stat_anchor_pc,
                    train_residual_anchor_phc=train_residual_anchor_phc,
                    learnable_output_anchor=learnable_output_anchor,
                    candidate_feature_mode="base",
                )
                threshold_kp, pred_residual_confidence_summary = (
                    _select_pred_residual_confidence_thresholds_from_tensors(
                        tensors=confidence_tensors,
                        cluster_id_c=cluster_id_c,
                        K=K,
                        allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                        penalty_names=penalty_names,
                        min_abs_improvement=pred_residual_confidence_gate_min_abs,
                        min_rel_improvement=pred_residual_confidence_gate_min_rel,
                        max_candidates=pred_residual_confidence_gate_max_candidates,
                        selection_metric=pred_residual_confidence_gate_selection_metric,
                        min_precision=pred_residual_confidence_gate_min_precision,
                        max_pred_positive_rate=pred_residual_confidence_gate_max_pred_rate,
                    )
                )
                pred_residual_confidence_summary["threshold_mode"] = "auto"
            else:
                threshold_tensor = torch.as_tensor(threshold_raw, dtype=torch.float32)
                if int(threshold_tensor.numel()) == 1:
                    threshold_kp = torch.full((K, P), float(threshold_tensor.reshape(-1)[0].item()), dtype=torch.float32)
                elif tuple(threshold_tensor.shape) == (K, P):
                    threshold_kp = threshold_tensor.reshape(K, P).to(dtype=torch.float32)
                else:
                    raise ValueError(
                        "moe.pred_side_residual.confidence_gate.threshold must be 'auto', "
                        f"a scalar, or shape [{K},{P}], got {tuple(threshold_tensor.shape)}."
                    )
                pred_residual_confidence_summary = {
                    "enable": True,
                    "source_requirement": "train_only",
                    "threshold_mode": "fixed",
                    "threshold_kp": [[float(v) for v in row] for row in threshold_kp.tolist()],
                    "penalty_names": list(penalty_names),
                    "selection_metric": str(pred_residual_confidence_gate_selection_metric),
                    "min_abs_improvement": float(pred_residual_confidence_gate_min_abs),
                    "min_rel_improvement": float(pred_residual_confidence_gate_min_rel),
                    "min_precision": float(pred_residual_confidence_gate_min_precision),
                    "max_pred_positive_rate": (
                        None
                        if pred_residual_confidence_gate_max_pred_rate is None
                        else float(pred_residual_confidence_gate_max_pred_rate)
                    ),
                }
            skip_threshold_raw = pred_residual_confidence_gate_cfg.get("skip_threshold", None)
            skip_threshold_k = None
            if skip_threshold_raw is not None:
                skip_threshold_tensor = torch.as_tensor(skip_threshold_raw, dtype=torch.float32)
                if int(skip_threshold_tensor.numel()) == 1:
                    skip_threshold_k = torch.full((K,), float(skip_threshold_tensor.reshape(-1)[0].item()), dtype=torch.float32)
                elif int(skip_threshold_tensor.numel()) == K:
                    skip_threshold_k = skip_threshold_tensor.reshape(K).to(dtype=torch.float32)
                else:
                    raise ValueError(
                        "moe.pred_side_residual.confidence_gate.skip_threshold must be scalar "
                        f"or length {K}, got {int(skip_threshold_tensor.numel())}."
                    )
                pred_residual_confidence_summary["skip_threshold_k"] = [
                    float(v) for v in skip_threshold_k.tolist()
                ]
            pred_residual.set_confidence_gate(
                penalty_threshold_kp=threshold_kp.to(device=device),
                skip_threshold_k=None if skip_threshold_k is None else skip_threshold_k.to(device=device),
                enable=True,
            )
            pred_residual_confidence_summary.update(
                {
                    "enable": True,
                    "source_split": str(confidence_source_split),
                    "source_range": [int(start_i), int(end_i)],
                    "source_windows": int(end_i - start_i),
                    "test_y_base_used": False,
                }
            )
            print(
                "Prediction residual confidence gate trained: "
                f"source={confidence_source_split}[{start_i}:{end_i}], "
                f"threshold_mode={pred_residual_confidence_summary.get('threshold_mode')}"
            )

    if patch_router_temporal_calibration_enable:
        assert pred_residual is not None and pred_residual.patch_router is not None
        calibration_indices = list(
            range(int(patch_router_calibration_start_idx), int(len(dtr)))
        )
        calibration_loader = DataLoader(
            Subset(dtr, calibration_indices),
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=False,
            num_workers=0,
            pin_memory=pin_mem,
        )
        calibration_tensors = collect_patch_risk_calibration_tensors(
            calibration_loader,
            eval_start=0,
        )
        calibration_time = calibration_tensors["time"].to(dtype=torch.long)
        calibration_span = max(
            1,
            int(len(dtr)) - int(patch_router_calibration_start_idx),
        )
        calibration_block = (
            (
                calibration_time - int(patch_router_calibration_start_idx)
            ).clamp_min(0)
            * int(patch_router_calibration_blocks)
            // calibration_span
        ).clamp_max(int(patch_router_calibration_blocks) - 1)
        if patch_router_calibration_per_penalty:
            probability_adoption = (
                pred_residual.patch_router.expert_risk_adoption_source
                in {"benefit_probability", "utility_veto"}
            )
            threshold_selection = (
                _select_recall_constrained_risk_threshold_by_penalty(
                    score_n=calibration_tensors["score"],
                    gain_n=calibration_tensors["gain"],
                    block_n=calibration_block,
                    penalty_n=calibration_tensors["penalty"],
                    penalty_names=penalty_names,
                    min_gain_cost_ratio=patch_router_calibration_min_gain_cost_ratio,
                    min_block_net_gain=patch_router_calibration_min_block_net_gain,
                    no_adoption_threshold=(
                        1.0 if probability_adoption else torch.finfo(torch.float32).max
                    ),
                )
            )
            threshold_by_penalty = threshold_selection["threshold_by_penalty"]
            pred_residual.patch_router.set_expert_risk_adopt_threshold_by_penalty(
                [float(threshold_by_penalty[name]) for name in penalty_names]
            )
        else:
            threshold_selection = _select_recall_constrained_risk_threshold(
                score_n=calibration_tensors["score"],
                gain_n=calibration_tensors["gain"],
                block_n=calibration_block,
                min_gain_cost_ratio=patch_router_calibration_min_gain_cost_ratio,
                min_block_net_gain=patch_router_calibration_min_block_net_gain,
            )
            pred_residual.patch_router.set_expert_risk_adopt_threshold(
                float(threshold_selection["threshold"])
            )
        patch_router_temporal_calibration_summary = {
            "source_split": "train_tail",
            "source_range": [
                int(patch_router_calibration_start_idx),
                int(len(dtr)),
            ],
            "source_windows": int(len(calibration_indices)),
            "test_y_base_used": False,
            **threshold_selection,
        }
        if patch_router_calibration_per_penalty:
            threshold_text = ", ".join(
                f"{name}={float(threshold_selection['threshold_by_penalty'][name]):.6f}"
                for name in penalty_names
            )
        else:
            threshold_text = f"threshold={float(threshold_selection['threshold']):.6f}"
        print(
            "Patch risk threshold calibrated: "
            f"source=train[{patch_router_calibration_start_idx}:{len(dtr)}], "
            f"{threshold_text}, "
            f"recall={float(threshold_selection['positive_recall']):.4f}, "
            f"gain_cost={float(threshold_selection['gain_cost_ratio']):.4f}"
        )

    if memory_enable:
        if cluster_memory_bank is not None and cluster_memory_bank.total_updates > 0:
            prototypes_kt = cluster_memory_bank.finalize()
            memory_meta = {
                "kind": "online_train_memory",
                "source_split": "train",
                "memory_len": int(t_train),
                "input_len": L,
                "pred_len": H,
                "num_window_updates": int(cluster_memory_bank.total_updates),
            }
        else:
            prototypes_kt = compute_cluster_prototypes(data_tc[:t_train], cluster_id_c)
            memory_meta = {
                "kind": "train_segment_prototype_fallback",
                "source_split": "train",
                "memory_len": int(t_train),
                "input_len": L,
                "pred_len": H,
                "num_window_updates": 0,
            }
        save_cluster_memory(memory_path, prototypes_kt, cluster_id_c, channel_names, meta=memory_meta)
        print(f"Saved cluster memory to: {memory_path}")

    best_checkpoint_path = None
    best_checkpoint_meta = None
    best_checkpoint_model_state = None
    best_checkpoint_gate_state = None
    best_checkpoint_pred_residual_state = None
    best_checkpoint_dynamic_lambda_state = None
    best_checkpoint_learnable_lambda_state = None
    best_checkpoint_learnable_output_anchor_state = None
    semantic_bank_checkpoint_ready = bool(
        semantic_bank_release_ready or semantic_bank_stage_complete
    )
    if bool(memory_cfg.get("save_checkpoint", False)) and (
        not pred_residual_semantic_bank_stage1 or semantic_bank_checkpoint_ready
    ):
        ckpt_path = str(memory_cfg.get("checkpoint_path", os.path.join(out_dir, "best_checkpoint.pt")))
        checkpoint_pred_residual_contract = (
            None if pred_residual_contract is None else dict(pred_residual_contract)
        )
        if (
            checkpoint_pred_residual_contract is not None
            and pred_residual is not None
            and pred_residual_semantic_output_head_scope == "per_cluster"
        ):
            checkpoint_pred_residual_contract["semantic_body_sha256"] = (
                _semantic_body_state_sha256(pred_residual.state_dict())
            )
        meta = {
            "K": K,
            "input_len": L,
            "pred_len": H,
            "num_channels": C,
            "cluster_id_c": cluster_id_c.detach().cpu(),
            "model_cfg": dict(model_cfg),
            "moe_cfg": dict(moe_cfg),
            "gate_feat_dim": gate_feat_dim,
            "gate_feature_mode": str(gate_feature_mode),
            "gate_feature_names": _gate_feature_names_for_mode(gate_feature_mode),
            "penalty_names": list(penalty_names),
            "pred_residual_contract": checkpoint_pred_residual_contract,
            "pred_residual_training_provenance": pred_residual_training_provenance,
            "penalty_need_statistics": dict(penalty_need_statistics_summary),
            "best_epoch": best_epoch.detach().cpu(),
            "shared_moe_across_clusters": bool(shared_moe_across_clusters),
            "shared_moe_best_epoch": int(shared_moe_best_epoch),
            "semantic_bank_independent_selection": (
                {
                    "release_ready": bool(semantic_bank_release_ready),
                    "stage_complete": bool(semantic_bank_stage_complete),
                    "active_penalty": str(pred_residual_semantic_active_penalty),
                    "accepted_penalties": list(semantic_bank_accepted_penalties),
                    "body_sha256_by_penalty": [
                        (
                            str(item["body_sha256"])
                            if item.get("body_sha256") is not None
                            else None
                        )
                        for item in semantic_bank_best_by_penalty
                    ],
                    "per_penalty": [
                        {
                            "penalty": str(item["penalty"]),
                            "epoch": item["epoch"],
                            "matching_penalty_relative_gain": (
                                float(item["matching_penalty_relative_gain"])
                                if math.isfinite(float(item["matching_penalty_relative_gain"]))
                                else None
                            ),
                            "accepted": bool(
                                isinstance(item.get("state"), dict)
                                and bool((item.get("acceptance") or {}).get("pass", False))
                            ),
                            "acceptance": item["acceptance"],
                            "optimizer_state_identity": str(
                                item.get(
                                    "optimizer_state_identity",
                                    semantic_bank_optimizer_identity_p[p],
                                )
                            ),
                            "training_provenance": item.get("training_provenance"),
                            "body_sha256": (
                                str(item["body_sha256"])
                                if item.get("body_sha256") is not None
                                else None
                            ),
                        }
                        for p, item in enumerate(semantic_bank_best_by_penalty)
                    ],
                }
                if pred_residual_semantic_bank_stage1
                else None
            ),
        }
        best_checkpoint_path = ckpt_path
        best_checkpoint_meta = meta
        best_checkpoint_model_state = _clone_module_state_dict(model)
        best_checkpoint_gate_state = _clone_module_state_dict(gate)
        best_checkpoint_pred_residual_state = _clone_module_state_dict(pred_residual)
        best_checkpoint_dynamic_lambda_state = _clone_module_state_dict(dynamic_lambda)
        best_checkpoint_learnable_lambda_state = _clone_module_state_dict(learnable_lambda)
        best_checkpoint_learnable_output_anchor_state = _clone_module_state_dict(learnable_output_anchor)
        assert best_checkpoint_model_state is not None
        assert best_checkpoint_gate_state is not None
        save_cluster_checkpoint(
            ckpt_path,
            best_checkpoint_model_state,
            best_checkpoint_gate_state,
            meta,
            pred_residual_state=best_checkpoint_pred_residual_state,
            dynamic_lambda_state=best_checkpoint_dynamic_lambda_state,
            learnable_lambda_state=best_checkpoint_learnable_lambda_state,
            learnable_output_anchor_state=best_checkpoint_learnable_output_anchor_state,
        )
        print(f"Saved best checkpoint to: {ckpt_path}")
    elif bool(memory_cfg.get("save_checkpoint", False)) and pred_residual_semantic_bank_stage1:
        print(
            "Semantic staged checkpoint was not saved because the active expert "
            "did not pass semantic-only acceptance."
        )
    if cluster_penalty_late_allowed_mask_kp is not None:
        cluster_penalty_allowed_mask_kp = cluster_penalty_late_allowed_mask_kp
        gate.set_penalty_allowed_mask(cluster_penalty_allowed_mask_kp)
        late_pred_residual_allowed_mask_cp = None
        if pred_residual is not None and bool(cluster_penalty_prior_cfg.get("apply_to_pred_residual", False)):
            late_pred_residual_allowed_mask_cp = _cluster_penalty_mask_to_channel_mask(
                cluster_penalty_allowed_mask_kp,
                cluster_id_c,
            )
            pred_residual.set_allowed_penalty_mask(late_pred_residual_allowed_mask_cp)
        cluster_penalty_prior_late_applied = True
        print(
            "Cluster penalty prior late-eval mask activated: "
            f"allowed_mask={cluster_penalty_allowed_mask_kp.detach().cpu().tolist()}, "
            f"pred_residual_channel_mask={late_pred_residual_allowed_mask_cp.detach().cpu().tolist() if late_pred_residual_allowed_mask_cp is not None else None}"
        )

    # print per-cluster penalty selection after training
    summary_loader = dl_va if len(dva) > 0 else dl_tr
    summary_name = "val" if len(dva) > 0 else "train"
    summary_eval_start = val_eval_start if len(dva) > 0 else 0
    lam_kp_best = lambda_kp_from_epochs(best_epoch)
    lam_kp_summary = average_lambda_kp(summary_loader, lam_kp_best)
    lambda_stats = collect_lambda_stats(summary_loader, lam_kp_best)
    summary_csv_path = os.path.join(out_dir, "cluster_penalty_probs.csv")
    avg_probs_summary = print_cluster_penalty_summary(summary_loader, title=summary_name, lam_kp=lam_kp_summary, csv_path=summary_csv_path)
    lambda_stats_csv_path = os.path.join(out_dir, "cluster_lambda_stats.csv")
    print_dynamic_lambda_summary(summary_name, lambda_stats, csv_path=lambda_stats_csv_path)
    moe_residual_summary = collect_pred_residual_summary(summary_loader, eval_start=summary_eval_start)
    semantic_bank_candidate_audit = collect_semantic_bank_candidate_audit(
        summary_loader,
        eval_start=summary_eval_start,
        split_name=summary_name,
        split_window_count=(len(dva) if summary_name == "val" else len(dtr)),
        include_level_oracle_diagnostic=bool(
            pred_residual_semantic_final_train_audit
        ),
    )
    if pred_residual_semantic_bank_stage1:
        semantic_audit_path = os.path.join(out_dir, "semantic_bank_candidate_audit.json")
        with open(semantic_audit_path, "w", encoding="utf-8") as f:
            json.dump(semantic_bank_candidate_audit, f, ensure_ascii=False, indent=2)
        semantic_bank_candidate_audit["artifact_path"] = semantic_audit_path
        moe_residual_summary["semantic_bank_candidate_audit"] = semantic_bank_candidate_audit
        semantic_bank_final_train_audit = None
        if pred_residual_semantic_final_train_audit:
            semantic_bank_final_train_audit = collect_semantic_bank_candidate_audit(
                dl_tr_source,
                eval_start=0,
                split_name="train",
                split_window_count=len(dtr),
                include_level_oracle_diagnostic=True,
            )
            train_semantic_audit_path = os.path.join(
                out_dir,
                "semantic_bank_final_train_candidate_audit.json",
            )
            with open(train_semantic_audit_path, "w", encoding="utf-8") as f:
                json.dump(
                    semantic_bank_final_train_audit,
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            semantic_bank_final_train_audit[
                "artifact_path"
            ] = train_semantic_audit_path
            moe_residual_summary[
                "semantic_bank_final_train_candidate_audit"
            ] = semantic_bank_final_train_audit
        moe_residual_summary["semantic_bank_independent_optimization"] = {
            "enabled": True,
            "reporting_mean_only": True,
            "release_ready": bool(semantic_bank_release_ready),
            "stage_complete": bool(semantic_bank_stage_complete),
            "active_penalty": str(pred_residual_semantic_active_penalty),
            "accepted_penalties": list(semantic_bank_accepted_penalties),
            "optimizer": (
                "independent_level_adam_amplitude_sgd_need_gate"
                if pred_residual_semantic_level_separate_need_gate
                and pred_residual_semantic_level_amplitude_optimizer_name == "adam"
                else "independent_sgd_per_penalty_and_disjoint_level_gate"
                if pred_residual_semantic_level_separate_need_gate
                else "independent_sgd_per_penalty"
            ),
            "gradient_clip": (
                "raw_gradient_mean_then_independent_level_group_clip_once"
                if pred_residual_semantic_raw_gradient_accumulation
                and pred_residual_semantic_level_separate_need_gate
                else "raw_gradient_mean_then_branch_clip_once"
                if pred_residual_semantic_raw_gradient_accumulation
                else "branch_only"
            ),
            "raw_gradient_accumulation": (
                dict(
                    (pred_residual_training_provenance or {}).get(
                        "raw_gradient_accumulation", {}
                    )
                )
                if pred_residual_semantic_raw_gradient_accumulation
                else {"enabled": False}
            ),
            "checkpoint_selection": "semantic_per_expert",
            "final_train_semantic_audit": {
                "enabled": bool(pred_residual_semantic_final_train_audit),
                "non_selecting": True,
                "test_read": False,
                "artifact_path": (
                    semantic_bank_final_train_audit.get("artifact_path")
                    if semantic_bank_final_train_audit is not None
                    else None
                ),
            },
            "history": semantic_bank_independent_history,
            "selected": [
                {
                    "penalty": str(item["penalty"]),
                    "epoch": item["epoch"],
                    "matching_penalty_relative_gain": (
                        float(item["matching_penalty_relative_gain"])
                        if math.isfinite(float(item["matching_penalty_relative_gain"]))
                        else None
                    ),
                    "acceptance": item["acceptance"],
                    "optimizer_state_identity": str(
                        item.get(
                            "optimizer_state_identity",
                            semantic_bank_optimizer_identity_p[p],
                        )
                    ),
                    "training_provenance": item.get("training_provenance"),
                    "body_sha256": (
                        str(item["body_sha256"])
                        if item.get("body_sha256") is not None
                        else None
                    ),
                    "accepted": bool(
                        isinstance(item.get("state"), dict)
                        and bool((item.get("acceptance") or {}).get("pass", False))
                    ),
                }
                for p, item in enumerate(semantic_bank_best_by_penalty)
            ],
        }
        print(
            "Semantic bank candidate audit: "
            f"all_pass={bool(semantic_bank_candidate_audit.get('all_pass', False))}, "
            f"artifact={semantic_audit_path}"
        )
    if (
        patch_router_train_oracle_diagnostic
        and pred_residual is not None
        and getattr(pred_residual, "patch_router", None) is not None
        and len(dtr) > 0
    ):
        train_residual_summary = collect_pred_residual_summary(dl_tr, eval_start=0)
        train_patch_summary = train_residual_summary.get("patch_router", {})
        train_oracle_diagnostic = train_patch_summary.get("oracle_diagnostic")
        if train_oracle_diagnostic is not None:
            moe_residual_summary["patch_router"]["train_oracle_diagnostic"] = (
                train_oracle_diagnostic
            )
    def collect_patch_router_temporal_blocks(
        dataset: Dataset,
        *,
        eval_start: int,
        num_blocks: int,
        split_name: str,
    ) -> Dict[str, object]:
        block_metrics: List[Dict[str, object]] = []
        for block_idx in range(num_blocks):
            block_start = block_idx * len(dataset) // num_blocks
            block_end = (block_idx + 1) * len(dataset) // num_blocks
            if block_end <= block_start:
                continue
            block_loader = DataLoader(
                Subset(dataset, range(block_start, block_end)),
                batch_size=bs,
                shuffle=False,
                num_workers=0,
                pin_memory=pin_mem,
            )
            block_summary = collect_pred_residual_summary(
                block_loader,
                eval_start=eval_start,
            )
            block_oracle = (block_summary.get("patch_router", {}) or {}).get(
                "oracle_diagnostic",
                {},
            ) or {}
            block_selected_rate = block_oracle.get("selected_class_rate", {}) or {}
            block_metrics.append(
                {
                    "block": int(block_idx),
                    "start_window": int(block_start),
                    "end_window": int(block_end),
                    "num_windows": int(block_end - block_start),
                    "selected_gain_pct": float(
                        block_oracle.get("selected_gain_pct", 0.0)
                    ),
                    "selected_mae_gain_pct": float(
                        block_oracle.get("selected_mae_gain_pct", 0.0)
                    ),
                    "proposal_oracle_best_recall_at_k": float(
                        block_oracle.get("proposal_oracle_best_recall_at_k", 0.0)
                    ),
                    "shortlist_pairwise_accuracy": float(
                        block_oracle.get("shortlist_pairwise_accuracy", 0.0)
                    ),
                    "risk_sign_recall": float(
                        block_oracle.get("risk_sign_recall", 0.0)
                    ),
                    "risk_sign_precision": float(
                        block_oracle.get("risk_sign_precision", 0.0)
                    ),
                    "selected_utility_recall": float(
                        block_oracle.get("selected_utility_recall", 0.0)
                    ),
                    "selected_utility_precision": float(
                        block_oracle.get("selected_utility_precision", 0.0)
                    ),
                    "selected_dual_utility_precision": float(
                        block_oracle.get(
                            "selected_dual_utility_precision",
                            0.0,
                        )
                    ),
                    "selected_dual_harmful_rate": float(
                        block_oracle.get("selected_dual_harmful_rate", 0.0)
                    ),
                    "selected_gain_to_cost_ratio": float(
                        block_oracle.get("selected_gain_to_cost_ratio", 0.0)
                    ),
                    "skip_rate": float(block_selected_rate.get("skip", 0.0)),
                    "selected_class_rate": dict(block_selected_rate),
                    "selected_utility_precision_by_penalty": dict(
                        block_oracle.get(
                            "selected_utility_precision_by_penalty",
                            {},
                        )
                        or {}
                    ),
                    "selected_mean_gain_by_penalty": dict(
                        block_oracle.get("selected_mean_gain_by_penalty", {}) or {}
                    ),
                    "proposal_oracle_best_recall_by_penalty": dict(
                        block_oracle.get(
                            "proposal_oracle_best_recall_by_penalty",
                            {},
                        )
                        or {}
                    ),
                }
            )
        return {
            "split": str(split_name),
            "num_blocks": int(num_blocks),
            "test_read": False,
            "blocks": block_metrics,
        }
    if (
        pred_residual is not None
        and getattr(pred_residual, "patch_router", None) is not None
    ):
        if stage2_objective_overlap_batches:
            moe_residual_summary["patch_router"][
                "loss_objective_gradient_overlap"
            ] = {
                "test_read": False,
                "max_batches": int(stage2_objective_overlap_max_batches),
                "batches": stage2_objective_overlap_batches,
            }
        if patch_router_train_temporal_blocks > 1 and len(dtr) > 0:
            moe_residual_summary["patch_router"]["train_temporal_block_metrics"] = (
                collect_patch_router_temporal_blocks(
                    dtr,
                    eval_start=0,
                    num_blocks=patch_router_train_temporal_blocks,
                    split_name="train",
                )
            )
        if patch_router_validation_temporal_blocks > 1 and len(dva) > 0:
            moe_residual_summary["patch_router"]["validation_temporal_block_metrics"] = (
                collect_patch_router_temporal_blocks(
                    dva,
                    eval_start=val_eval_start,
                    num_blocks=patch_router_validation_temporal_blocks,
                    split_name="validation",
                )
            )
        if patch_router_score_threshold_curve and len(dtr) > 0 and len(dva) > 0:
            fixed_penalty_c = pred_residual.patch_router.fixed_penalty_index_by_channel_c
            fixed_candidate_mode = int(fixed_penalty_c.numel()) == C
            def score_curve_dataset(dataset: Dataset) -> Dataset:
                max_windows = int(patch_router_score_threshold_max_windows)
                if max_windows <= 0 or len(dataset) <= max_windows:
                    return dataset
                indices = (
                    torch.linspace(0, len(dataset) - 1, steps=max_windows)
                    .round()
                    .to(dtype=torch.long)
                    .unique(sorted=True)
                    .tolist()
                )
                return Subset(dataset, indices)

            score_train_dataset = score_curve_dataset(dtr)
            score_val_dataset = score_curve_dataset(dva)
            chronological_train_loader = DataLoader(
                score_train_dataset,
                batch_size=bs,
                shuffle=False,
                num_workers=0,
                pin_memory=pin_mem,
            )
            train_score_tensors = collect_patch_risk_calibration_tensors(
                chronological_train_loader,
                eval_start=0,
            )
            score_val_loader = DataLoader(
                score_val_dataset,
                batch_size=bs,
                shuffle=False,
                num_workers=0,
                pin_memory=pin_mem,
            )
            val_score_tensors = collect_patch_risk_calibration_tensors(
                score_val_loader,
                eval_start=val_eval_start,
            )
            fixed_penalty_cpu = fixed_penalty_c.detach().cpu()
            active_channel_c = (
                fixed_penalty_cpu >= 0
                if fixed_candidate_mode
                else torch.ones(C, dtype=torch.bool)
            )
            routing_scope = (
                "fixed_channel_candidate"
                if fixed_candidate_mode
                else "dynamic_patch_candidate"
            )
            configured_threshold = float(
                pred_residual.patch_router.expert_risk_adopt_threshold
            )

            def score_curve_for_split(
                tensors: Dict[str, torch.Tensor],
                *,
                score_key: str = "score",
                fixed_threshold: float = configured_threshold,
            ) -> Dict[str, object]:
                aggregate = _risk_score_threshold_curve_summary(
                    score_n=tensors[score_key][:, active_channel_c],
                    gain_n=tensors["gain"][:, active_channel_c],
                    fixed_threshold=fixed_threshold,
                )
                per_channel = []
                for channel_idx in range(C):
                    channel_meta: Dict[str, object] = {
                        "channel_index": int(channel_idx),
                        "channel": str(channel_names[channel_idx]),
                        "routing_scope": routing_scope,
                    }
                    if fixed_candidate_mode:
                        penalty_idx = int(fixed_penalty_cpu[channel_idx].item())
                        if penalty_idx < 0:
                            continue
                        channel_meta.update(
                            {
                                "penalty_index": penalty_idx,
                                "penalty": str(penalty_names[penalty_idx]),
                            }
                        )
                    per_channel.append(
                        {
                            **channel_meta,
                            **_risk_score_threshold_curve_summary(
                                score_n=tensors[score_key][:, channel_idx],
                                gain_n=tensors["gain"][:, channel_idx],
                                fixed_threshold=fixed_threshold,
                            ),
                        }
                    )
                per_penalty = []
                for penalty_idx, penalty_name in enumerate(penalty_names):
                    if fixed_candidate_mode:
                        penalty_channel_c = fixed_penalty_cpu == int(penalty_idx)
                        if not bool(penalty_channel_c.any().item()):
                            continue
                        penalty_score = tensors[score_key][:, penalty_channel_c]
                        penalty_gain = tensors["gain"][:, penalty_channel_c]
                        channel_indices = torch.nonzero(
                            penalty_channel_c,
                            as_tuple=False,
                        ).reshape(-1)
                    else:
                        penalty_entry_mask = tensors["penalty"] == int(penalty_idx)
                        if not bool(penalty_entry_mask.any().item()):
                            continue
                        penalty_score = tensors[score_key][penalty_entry_mask]
                        penalty_gain = tensors["gain"][penalty_entry_mask]
                        channel_indices = torch.nonzero(
                            penalty_entry_mask.any(dim=(0, 2)),
                            as_tuple=False,
                        ).reshape(-1)
                    per_penalty.append(
                        {
                            "penalty_index": int(penalty_idx),
                            "penalty": str(penalty_name),
                            "routing_scope": routing_scope,
                            "channel_indices": [
                                int(value)
                                for value in channel_indices.tolist()
                            ],
                            **_risk_score_threshold_curve_summary(
                                score_n=penalty_score,
                                gain_n=penalty_gain,
                                fixed_threshold=fixed_threshold,
                            ),
                        }
                    )
                return {
                    **aggregate,
                    "per_channel": per_channel,
                    "per_penalty": per_penalty,
                }

            moe_residual_summary["patch_router"]["score_threshold_curve"] = {
                "test_read": False,
                "routing_scope": routing_scope,
                "train_sampled_windows": int(len(score_train_dataset)),
                "validation_sampled_windows": int(len(score_val_dataset)),
                "active_channel_mask": [
                    bool(value) for value in active_channel_c.tolist()
                ],
                "train": score_curve_for_split(train_score_tensors),
                "validation": score_curve_for_split(val_score_tensors),
            }
            head_specs = {
                "executed_risk_score": {
                    "tensor_key": "score",
                    "fixed_threshold": configured_threshold,
                    "native_role": "final executed-candidate adoption decision",
                },
                "proposal_adopt_probability": {
                    "tensor_key": "proposal_adopt_probability",
                    "fixed_threshold": 0.5,
                    "native_role": "whether any penalty candidate should be proposed",
                },
                "proposal_fixed_probability": {
                    "tensor_key": "proposal_fixed_probability",
                    "fixed_threshold": 0.5,
                    "native_role": "proposal score of the executed candidate",
                },
                "proposal_fixed_logit": {
                    "tensor_key": "proposal_fixed_logit",
                    "fixed_threshold": 0.0,
                    "native_role": "proposal logit of the executed candidate",
                },
                "risk_fixed_probability": {
                    "tensor_key": "risk_fixed_probability",
                    "fixed_threshold": 0.5,
                    "native_role": "risk-sign probability of the executed candidate",
                },
                "risk_domain_disagreement": {
                    "tensor_key": "risk_domain_disagreement",
                    "fixed_threshold": 0.0,
                    "native_role": "temporal-domain risk probability disagreement",
                },
                "utility_fixed_score": {
                    "tensor_key": "utility_fixed_score",
                    "fixed_threshold": 0.0,
                    "native_role": "expected utility of the executed candidate",
                },
                "pairwise_fixed_score": {
                    "tensor_key": "pairwise_fixed_score",
                    "fixed_threshold": 0.0,
                    "native_role": "pairwise ranking score of the executed candidate",
                },
                "lower_quantile_fixed_score": {
                    "tensor_key": "lower_quantile_fixed_score",
                    "fixed_threshold": 0.0,
                    "native_role": "lower-quantile utility of the executed candidate",
                },
                "utility_veto_fixed_probability": {
                    "tensor_key": "utility_veto_fixed_probability",
                    "fixed_threshold": 0.5,
                    "native_role": "utility-veto probability of the executed candidate",
                },
            }
            head_target_overlap = {}
            overlap_temporal_blocks = max(
                int(patch_router_train_temporal_blocks),
                int(patch_router_validation_temporal_blocks),
            )

            def chronological_head_overlap(
                tensors: Dict[str, torch.Tensor],
                *,
                score_key: str,
                fixed_threshold: float,
            ) -> List[Dict[str, object]]:
                num_windows = int(tensors[score_key].shape[0])
                if overlap_temporal_blocks <= 1 or num_windows == 0:
                    return []
                rows = []
                for block_idx in range(overlap_temporal_blocks):
                    block_start = block_idx * num_windows // overlap_temporal_blocks
                    block_end = (
                        (block_idx + 1) * num_windows // overlap_temporal_blocks
                    )
                    if block_end <= block_start:
                        continue
                    rows.append(
                        {
                            "block": int(block_idx),
                            "start_window": int(block_start),
                            "end_window": int(block_end),
                            "num_windows": int(block_end - block_start),
                            **_risk_score_threshold_curve_summary(
                                score_n=tensors[score_key][
                                    block_start:block_end,
                                    active_channel_c,
                                ],
                                gain_n=tensors["gain"][
                                    block_start:block_end,
                                    active_channel_c,
                                ],
                                fixed_threshold=fixed_threshold,
                            ),
                        }
                    )
                return rows

            def score_complementarity_summary(
                tensors: Dict[str, torch.Tensor],
                *,
                score_key: str,
            ) -> Dict[str, object]:
                def summarize(
                    score: torch.Tensor,
                    gain: torch.Tensor,
                    base_mse: torch.Tensor,
                    cross: torch.Tensor,
                    delta_sq: torch.Tensor,
                ) -> Dict[str, object]:
                    score = score.detach().reshape(-1).to(torch.float64)
                    gain = gain.detach().reshape(-1).to(torch.float64)
                    base_mse = base_mse.detach().reshape(-1).to(torch.float64)
                    cross = cross.detach().reshape(-1).to(torch.float64)
                    delta_sq = delta_sq.detach().reshape(-1).to(torch.float64)
                    reconstructed_gain = 2.0 * cross - delta_sq
                    alignment = cross / torch.sqrt(
                        (base_mse * delta_sq).clamp_min(1.0e-12)
                    )
                    optimal_scale = cross / delta_sq.clamp_min(1.0e-12)
                    normalized_gain = gain / base_mse.clamp_min(1.0e-12)
                    signals = {
                        "incremental_gain": gain,
                        "backbone_mse": base_mse,
                        "residual_delta_cross": cross,
                        "delta_energy": delta_sq,
                        "residual_delta_cosine": alignment.clamp(-1.0, 1.0),
                        "optimal_delta_scale": optimal_scale.clamp(-5.0, 5.0),
                        "normalized_incremental_gain": normalized_gain.clamp(
                            -5.0,
                            5.0,
                        ),
                    }
                    finite = torch.isfinite(score)
                    for signal in signals.values():
                        finite = finite & torch.isfinite(signal)
                    if not bool(finite.any().item()):
                        return {"status": "empty", "total_count": 0}
                    score = score[finite]
                    signals = {
                        name: signal[finite] for name, signal in signals.items()
                    }
                    gain = signals["incremental_gain"]
                    positive = gain > 0.0
                    positive_count = int(positive.sum().item())

                    def pearson(signal: torch.Tensor) -> Optional[float]:
                        centered_score = score - score.mean()
                        centered_signal = signal - signal.mean()
                        denominator = torch.sqrt(
                            centered_score.square().sum()
                            * centered_signal.square().sum()
                        )
                        if float(denominator.item()) <= 0.0:
                            return None
                        return float(
                            (
                                (centered_score * centered_signal).sum()
                                / denominator
                            ).item()
                        )

                    top_index = None
                    if positive_count > 0:
                        top_index = torch.argsort(
                            score,
                            descending=True,
                            stable=True,
                        )[:positive_count]

                    def signal_means(signal: torch.Tensor) -> Dict[str, object]:
                        all_mean = float(signal.mean().item())
                        beneficial_mean = (
                            float(signal[positive].mean().item())
                            if positive_count > 0
                            else None
                        )
                        top_mean = (
                            float(signal.index_select(0, top_index).mean().item())
                            if top_index is not None
                            else None
                        )
                        return {
                            "all": all_mean,
                            "oracle_beneficial": beneficial_mean,
                            "top_by_gate": top_mean,
                            "top_minus_all": (
                                float(top_mean - all_mean)
                                if top_mean is not None
                                else None
                            ),
                        }

                    reconstructed = reconstructed_gain[finite]
                    return {
                        "status": "ok",
                        "total_count": int(score.numel()),
                        "positive_count": positive_count,
                        "positive_rate": float(
                            positive.to(torch.float64).mean().item()
                        ),
                        "gain_decomposition_max_abs_error": float(
                            (gain - reconstructed).abs().max().item()
                        ),
                        "score_pearson": {
                            name: pearson(signal)
                            for name, signal in signals.items()
                        },
                        "signal_means": {
                            name: signal_means(signal)
                            for name, signal in signals.items()
                        },
                    }

                aggregate = summarize(
                    tensors[score_key][:, active_channel_c],
                    tensors["gain"][:, active_channel_c],
                    tensors["base_mse"][:, active_channel_c],
                    tensors["cross"][:, active_channel_c],
                    tensors["delta_sq"][:, active_channel_c],
                )
                per_channel = []
                for channel_idx in range(C):
                    channel_meta: Dict[str, object] = {
                        "channel_index": int(channel_idx),
                        "channel": str(channel_names[channel_idx]),
                        "routing_scope": routing_scope,
                    }
                    if fixed_candidate_mode:
                        penalty_idx = int(fixed_penalty_cpu[channel_idx].item())
                        if penalty_idx < 0:
                            continue
                        channel_meta.update(
                            {
                                "penalty_index": penalty_idx,
                                "penalty": str(penalty_names[penalty_idx]),
                            }
                        )
                    per_channel.append(
                        {
                            **channel_meta,
                            **summarize(
                                tensors[score_key][:, channel_idx],
                                tensors["gain"][:, channel_idx],
                                tensors["base_mse"][:, channel_idx],
                                tensors["cross"][:, channel_idx],
                                tensors["delta_sq"][:, channel_idx],
                            ),
                        }
                    )
                return {**aggregate, "per_channel": per_channel}

            for head_name, head_spec in head_specs.items():
                if (
                    patch_router_score_threshold_heads is not None
                    and head_name not in patch_router_score_threshold_heads
                ):
                    continue
                tensor_key = str(head_spec["tensor_key"])
                if (
                    tensor_key not in train_score_tensors
                    or tensor_key not in val_score_tensors
                    or int(train_score_tensors[tensor_key].numel()) == 0
                    or int(val_score_tensors[tensor_key].numel()) == 0
                ):
                    continue
                threshold = float(head_spec["fixed_threshold"])
                head_target_overlap[head_name] = {
                    **head_spec,
                    "train": score_curve_for_split(
                        train_score_tensors,
                        score_key=tensor_key,
                        fixed_threshold=threshold,
                    ),
                    "validation": score_curve_for_split(
                        val_score_tensors,
                        score_key=tensor_key,
                        fixed_threshold=threshold,
                    ),
                    "train_chronological_blocks": chronological_head_overlap(
                        train_score_tensors,
                        score_key=tensor_key,
                        fixed_threshold=threshold,
                    ),
                    "validation_chronological_blocks": chronological_head_overlap(
                        val_score_tensors,
                        score_key=tensor_key,
                        fixed_threshold=threshold,
                    ),
                    "train_complementarity": score_complementarity_summary(
                        train_score_tensors,
                        score_key=tensor_key,
                    ),
                    "validation_complementarity": score_complementarity_summary(
                        val_score_tensors,
                        score_key=tensor_key,
                    ),
                }
            moe_residual_summary["patch_router"]["head_target_overlap"] = {
                "test_read": False,
                "routing_scope": routing_scope,
                "chronological_block_count": int(overlap_temporal_blocks),
                "target": (
                    "executed-candidate patch MSE gain > 0 on the exact eval output-anchor path"
                ),
                "heads": head_target_overlap,
            }
        if patch_router_walk_forward_enable and len(dtr) > 0 and len(dva) > 0:
            fixed_penalty_c = pred_residual.patch_router.fixed_penalty_index_by_channel_c
            walk_forward_active_channel_c = (
                fixed_penalty_c >= 0
                if int(fixed_penalty_c.numel()) == C
                else torch.ones(C, dtype=torch.bool)
            )
            chronological_train_loader = DataLoader(
                dtr,
                batch_size=bs,
                shuffle=False,
                num_workers=0,
                pin_memory=pin_mem,
            )
            train_risk_tensors = collect_patch_risk_calibration_tensors(
                chronological_train_loader,
                eval_start=0,
                include_all_candidates=(
                    patch_router_walk_forward_rerank_all_candidates
                    or patch_router_walk_forward_feedback_ridge
                ),
            )
            val_risk_tensors = collect_patch_risk_calibration_tensors(
                dl_va,
                eval_start=val_eval_start,
                include_patch_values=(
                    patch_router_walk_forward_scale_mode
                    in {"least_squares", "feature_ridge"}
                ),
                include_all_candidates=(
                    patch_router_walk_forward_rerank_all_candidates
                    or patch_router_walk_forward_feedback_ridge
                ),
            )
            train_time_n = train_risk_tensors["time"][:, 0, 0]
            val_time_n = val_risk_tensors["time"][:, 0, 0]
            patch_label_delay_q = (
                torch.arange(
                    1,
                    int(pred_residual.patch_router.num_patches) + 1,
                    dtype=torch.long,
                )
                * int(pred_residual.patch_router.patch_len)
                if patch_router_walk_forward_label_delay_mode == "patch_end"
                else None
            )
            train_audit_start = max(
                1,
                min(
                    int(train_time_n.numel()) - 1,
                    round(
                        int(train_time_n.numel())
                        * (1.0 - patch_router_walk_forward_train_audit_fraction)
                    ),
                ),
            )
            moe_residual_summary["patch_router"]["train_walk_forward_audit"] = (
                _walk_forward_patch_reliability_metrics(
                    train_time_n=train_time_n[:train_audit_start],
                    train_gain_ncq=train_risk_tensors["gain"][:train_audit_start],
                    eval_time_n=train_time_n[train_audit_start:],
                    eval_base_mse_ncq=train_risk_tensors["base_mse"][train_audit_start:],
                    eval_candidate_mse_ncq=train_risk_tensors["candidate_mse"][train_audit_start:],
                    eval_base_mae_ncq=train_risk_tensors["base_mae"][train_audit_start:],
                    eval_candidate_mae_ncq=train_risk_tensors["candidate_mae"][train_audit_start:],
                    active_channel_mask_c=walk_forward_active_channel_c,
                    train_penalty_ncq=(
                        train_risk_tensors["penalty"][:train_audit_start]
                        if patch_router_walk_forward_condition_on_selected_penalty
                        else None
                    ),
                    eval_penalty_ncq=(
                        train_risk_tensors["penalty"][train_audit_start:]
                        if patch_router_walk_forward_condition_on_selected_penalty
                        else None
                    ),
                    train_regime_ncf=train_risk_tensors["regime"][:train_audit_start],
                    eval_regime_ncf=train_risk_tensors["regime"][train_audit_start:],
                    max_abs_regime_z=patch_router_walk_forward_max_abs_regime_z,
                    train_cross_ncq=train_risk_tensors["cross"][:train_audit_start],
                    train_delta_sq_ncq=train_risk_tensors["delta_sq"][:train_audit_start],
                    eval_cross_ncq=train_risk_tensors["cross"][train_audit_start:],
                    eval_delta_sq_ncq=train_risk_tensors["delta_sq"][train_audit_start:],
                    train_scale_feature_ncqf=train_risk_tensors["scale_feature"][:train_audit_start],
                    eval_scale_feature_ncqf=train_risk_tensors["scale_feature"][train_audit_start:],
                    scale_mode=patch_router_walk_forward_scale_mode,
                    max_scale=patch_router_walk_forward_max_scale,
                    scale_consensus_blocks=(
                        patch_router_walk_forward_scale_consensus_blocks
                    ),
                    feature_ridge=patch_router_walk_forward_feature_ridge,
                    feature_update_blocks=(
                        patch_router_walk_forward_feature_update_blocks
                    ),
                    patch_label_delay_q=patch_label_delay_q,
                    label_delay=patch_router_walk_forward_label_delay,
                    lookback_windows=patch_router_walk_forward_lookback,
                    min_history_windows=patch_router_walk_forward_min_history,
                    history_stride=patch_router_walk_forward_history_stride,
                    min_mean_gain=patch_router_walk_forward_min_mean_gain,
                    temporal_blocks=patch_router_walk_forward_temporal_blocks,
                )
            )
            moe_residual_summary["patch_router"]["walk_forward_reliability"] = (
                _walk_forward_patch_reliability_metrics(
                    train_time_n=train_time_n,
                    train_gain_ncq=train_risk_tensors["gain"],
                    eval_time_n=val_time_n,
                    eval_base_mse_ncq=val_risk_tensors["base_mse"],
                    eval_candidate_mse_ncq=val_risk_tensors["candidate_mse"],
                    eval_base_mae_ncq=val_risk_tensors["base_mae"],
                    eval_candidate_mae_ncq=val_risk_tensors["candidate_mae"],
                    active_channel_mask_c=walk_forward_active_channel_c,
                    train_penalty_ncq=(
                        train_risk_tensors["penalty"]
                        if patch_router_walk_forward_condition_on_selected_penalty
                        else None
                    ),
                    eval_penalty_ncq=(
                        val_risk_tensors["penalty"]
                        if patch_router_walk_forward_condition_on_selected_penalty
                        else None
                    ),
                    train_regime_ncf=train_risk_tensors["regime"],
                    eval_regime_ncf=val_risk_tensors["regime"],
                    max_abs_regime_z=patch_router_walk_forward_max_abs_regime_z,
                    train_cross_ncq=train_risk_tensors["cross"],
                    train_delta_sq_ncq=train_risk_tensors["delta_sq"],
                    eval_cross_ncq=val_risk_tensors["cross"],
                    eval_delta_sq_ncq=val_risk_tensors["delta_sq"],
                    eval_base_residual_ncqr=val_risk_tensors[
                        "base_residual_patch"
                    ],
                    eval_candidate_delta_ncqr=val_risk_tensors[
                        "candidate_delta_patch"
                    ],
                    train_scale_feature_ncqf=train_risk_tensors["scale_feature"],
                    eval_scale_feature_ncqf=val_risk_tensors["scale_feature"],
                    scale_mode=patch_router_walk_forward_scale_mode,
                    max_scale=patch_router_walk_forward_max_scale,
                    scale_consensus_blocks=(
                        patch_router_walk_forward_scale_consensus_blocks
                    ),
                    feature_ridge=patch_router_walk_forward_feature_ridge,
                    feature_update_blocks=(
                        patch_router_walk_forward_feature_update_blocks
                    ),
                    patch_label_delay_q=patch_label_delay_q,
                    label_delay=patch_router_walk_forward_label_delay,
                    lookback_windows=patch_router_walk_forward_lookback,
                    min_history_windows=patch_router_walk_forward_min_history,
                    history_stride=patch_router_walk_forward_history_stride,
                    min_mean_gain=patch_router_walk_forward_min_mean_gain,
                    temporal_blocks=patch_router_walk_forward_temporal_blocks,
                )
            )
            if patch_router_walk_forward_rerank_all_candidates:
                train_gain_all = (
                    train_risk_tensors["base_mse"].unsqueeze(-1)
                    - train_risk_tensors["candidate_mse_all"]
                )
                moe_residual_summary["patch_router"][
                    "train_expert_reliability_rerank_audit"
                ] = _walk_forward_expert_reliability_rerank_metrics(
                    train_time_n=train_time_n[:train_audit_start],
                    train_gain_ncqp=train_gain_all[:train_audit_start],
                    eval_time_n=train_time_n[train_audit_start:],
                    eval_base_mse_ncq=train_risk_tensors["base_mse"][train_audit_start:],
                    eval_candidate_mse_ncqp=train_risk_tensors[
                        "candidate_mse_all"
                    ][train_audit_start:],
                    eval_base_mae_ncq=train_risk_tensors["base_mae"][train_audit_start:],
                    eval_candidate_mae_ncqp=train_risk_tensors[
                        "candidate_mae_all"
                    ][train_audit_start:],
                    eval_score_ncqp=train_risk_tensors[
                        "candidate_score_all"
                    ][train_audit_start:],
                    active_channel_mask_c=walk_forward_active_channel_c,
                    label_delay=patch_router_walk_forward_label_delay,
                    lookback_windows=patch_router_walk_forward_lookback,
                    min_history_windows=patch_router_walk_forward_min_history,
                    history_stride=patch_router_walk_forward_history_stride,
                    min_mean_gain=patch_router_walk_forward_min_mean_gain,
                    temporal_blocks=patch_router_walk_forward_temporal_blocks,
                )
                moe_residual_summary["patch_router"][
                    "expert_reliability_rerank"
                ] = _walk_forward_expert_reliability_rerank_metrics(
                    train_time_n=train_time_n,
                    train_gain_ncqp=train_gain_all,
                    eval_time_n=val_time_n,
                    eval_base_mse_ncq=val_risk_tensors["base_mse"],
                    eval_candidate_mse_ncqp=val_risk_tensors[
                        "candidate_mse_all"
                    ],
                    eval_base_mae_ncq=val_risk_tensors["base_mae"],
                    eval_candidate_mae_ncqp=val_risk_tensors[
                        "candidate_mae_all"
                    ],
                    eval_score_ncqp=val_risk_tensors["candidate_score_all"],
                    active_channel_mask_c=walk_forward_active_channel_c,
                    label_delay=patch_router_walk_forward_label_delay,
                    lookback_windows=patch_router_walk_forward_lookback,
                    min_history_windows=patch_router_walk_forward_min_history,
                    history_stride=patch_router_walk_forward_history_stride,
                    min_mean_gain=patch_router_walk_forward_min_mean_gain,
                    temporal_blocks=patch_router_walk_forward_temporal_blocks,
                )
            if patch_router_walk_forward_feedback_ridge:
                feedback_common = {
                    "active_channel_mask_c": walk_forward_active_channel_c,
                    "label_delay": patch_router_walk_forward_label_delay,
                    "lookback_windows": patch_router_walk_forward_lookback,
                    "min_history_windows": patch_router_walk_forward_min_history,
                    "history_stride": patch_router_walk_forward_history_stride,
                    "ridge": patch_router_walk_forward_feedback_ridge_strength,
                    "target_clip": patch_router_walk_forward_feedback_target_clip,
                    "temporal_blocks": patch_router_walk_forward_temporal_blocks,
                }
                moe_residual_summary["patch_router"][
                    "train_expert_feedback_ridge_audit"
                ] = _causal_expert_feedback_ridge_metrics(
                    train_time_n=train_time_n[:train_audit_start],
                    train_base_mse_ncq=train_risk_tensors["base_mse"][:train_audit_start],
                    train_candidate_mse_ncqp=train_risk_tensors[
                        "candidate_mse_all"
                    ][:train_audit_start],
                    train_base_mae_ncq=train_risk_tensors["base_mae"][:train_audit_start],
                    train_candidate_mae_ncqp=train_risk_tensors[
                        "candidate_mae_all"
                    ][:train_audit_start],
                    train_score_ncqp=train_risk_tensors[
                        "candidate_score_all"
                    ][:train_audit_start],
                    eval_time_n=train_time_n[train_audit_start:],
                    eval_base_mse_ncq=train_risk_tensors["base_mse"][train_audit_start:],
                    eval_candidate_mse_ncqp=train_risk_tensors[
                        "candidate_mse_all"
                    ][train_audit_start:],
                    eval_base_mae_ncq=train_risk_tensors["base_mae"][train_audit_start:],
                    eval_candidate_mae_ncqp=train_risk_tensors[
                        "candidate_mae_all"
                    ][train_audit_start:],
                    eval_score_ncqp=train_risk_tensors[
                        "candidate_score_all"
                    ][train_audit_start:],
                    **feedback_common,
                )
                moe_residual_summary["patch_router"][
                    "expert_feedback_ridge"
                ] = _causal_expert_feedback_ridge_metrics(
                    train_time_n=train_time_n,
                    train_base_mse_ncq=train_risk_tensors["base_mse"],
                    train_candidate_mse_ncqp=train_risk_tensors[
                        "candidate_mse_all"
                    ],
                    train_base_mae_ncq=train_risk_tensors["base_mae"],
                    train_candidate_mae_ncqp=train_risk_tensors[
                        "candidate_mae_all"
                    ],
                    train_score_ncqp=train_risk_tensors["candidate_score_all"],
                    eval_time_n=val_time_n,
                    eval_base_mse_ncq=val_risk_tensors["base_mse"],
                    eval_candidate_mse_ncqp=val_risk_tensors[
                        "candidate_mse_all"
                    ],
                    eval_base_mae_ncq=val_risk_tensors["base_mae"],
                    eval_candidate_mae_ncqp=val_risk_tensors[
                        "candidate_mae_all"
                    ],
                    eval_score_ncqp=val_risk_tensors["candidate_score_all"],
                    **feedback_common,
                )
            if patch_router_walk_forward_test_enable:
                if len(dte) <= 0:
                    raise ValueError(
                        "walk_forward_reliability.test_enable requires test evaluation."
                    )
                test_risk_tensors = collect_patch_risk_calibration_tensors(
                    dl_te,
                    eval_start=test_eval_start,
                    include_patch_values=(
                        patch_router_walk_forward_scale_mode
                        in {"least_squares", "feature_ridge"}
                    ),
                    include_all_candidates=(
                        patch_router_walk_forward_rerank_all_candidates
                        or patch_router_walk_forward_feedback_ridge
                    ),
                )
                test_time_n = test_risk_tensors["time"][:, 0, 0]

                def train_val_history(key: str) -> torch.Tensor:
                    return torch.cat(
                        [train_risk_tensors[key], val_risk_tensors[key]],
                        dim=0,
                    )

                test_walk_forward = _walk_forward_patch_reliability_metrics(
                    train_time_n=torch.cat([train_time_n, val_time_n], dim=0),
                    train_gain_ncq=train_val_history("gain"),
                    eval_time_n=test_time_n,
                    eval_base_mse_ncq=test_risk_tensors["base_mse"],
                    eval_candidate_mse_ncq=test_risk_tensors["candidate_mse"],
                    eval_base_mae_ncq=test_risk_tensors["base_mae"],
                    eval_candidate_mae_ncq=test_risk_tensors["candidate_mae"],
                    active_channel_mask_c=walk_forward_active_channel_c,
                    train_penalty_ncq=(
                        train_val_history("penalty")
                        if patch_router_walk_forward_condition_on_selected_penalty
                        else None
                    ),
                    eval_penalty_ncq=(
                        test_risk_tensors["penalty"]
                        if patch_router_walk_forward_condition_on_selected_penalty
                        else None
                    ),
                    train_regime_ncf=train_val_history("regime"),
                    eval_regime_ncf=test_risk_tensors["regime"],
                    max_abs_regime_z=patch_router_walk_forward_max_abs_regime_z,
                    train_cross_ncq=train_val_history("cross"),
                    train_delta_sq_ncq=train_val_history("delta_sq"),
                    eval_cross_ncq=test_risk_tensors["cross"],
                    eval_delta_sq_ncq=test_risk_tensors["delta_sq"],
                    eval_base_residual_ncqr=test_risk_tensors[
                        "base_residual_patch"
                    ],
                    eval_candidate_delta_ncqr=test_risk_tensors[
                        "candidate_delta_patch"
                    ],
                    train_scale_feature_ncqf=train_val_history("scale_feature"),
                    eval_scale_feature_ncqf=test_risk_tensors["scale_feature"],
                    scale_mode=patch_router_walk_forward_scale_mode,
                    max_scale=patch_router_walk_forward_max_scale,
                    scale_consensus_blocks=(
                        patch_router_walk_forward_scale_consensus_blocks
                    ),
                    feature_ridge=patch_router_walk_forward_feature_ridge,
                    feature_update_blocks=(
                        patch_router_walk_forward_feature_update_blocks
                    ),
                    patch_label_delay_q=patch_label_delay_q,
                    label_delay=patch_router_walk_forward_label_delay,
                    lookback_windows=patch_router_walk_forward_lookback,
                    min_history_windows=patch_router_walk_forward_min_history,
                    history_stride=patch_router_walk_forward_history_stride,
                    min_mean_gain=patch_router_walk_forward_min_mean_gain,
                    temporal_blocks=patch_router_walk_forward_temporal_blocks,
                )
                test_walk_forward["test_read"] = True
                test_walk_forward["history_splits"] = ["train", "validation"]
                moe_residual_summary["patch_router"][
                    "test_walk_forward_reliability"
                ] = test_walk_forward
                if patch_router_walk_forward_rerank_all_candidates:
                    test_expert_rerank = (
                        _walk_forward_expert_reliability_rerank_metrics(
                            train_time_n=torch.cat(
                                [train_time_n, val_time_n],
                                dim=0,
                            ),
                            train_gain_ncqp=(
                                train_val_history("base_mse").unsqueeze(-1)
                                - train_val_history("candidate_mse_all")
                            ),
                            eval_time_n=test_time_n,
                            eval_base_mse_ncq=test_risk_tensors["base_mse"],
                            eval_candidate_mse_ncqp=test_risk_tensors[
                                "candidate_mse_all"
                            ],
                            eval_base_mae_ncq=test_risk_tensors["base_mae"],
                            eval_candidate_mae_ncqp=test_risk_tensors[
                                "candidate_mae_all"
                            ],
                            eval_score_ncqp=test_risk_tensors[
                                "candidate_score_all"
                            ],
                            active_channel_mask_c=walk_forward_active_channel_c,
                            label_delay=patch_router_walk_forward_label_delay,
                            lookback_windows=patch_router_walk_forward_lookback,
                            min_history_windows=patch_router_walk_forward_min_history,
                            history_stride=patch_router_walk_forward_history_stride,
                            min_mean_gain=patch_router_walk_forward_min_mean_gain,
                            temporal_blocks=patch_router_walk_forward_temporal_blocks,
                        )
                    )
                    test_expert_rerank["test_read"] = True
                    test_expert_rerank["history_splits"] = [
                        "train",
                        "validation",
                    ]
                    moe_residual_summary["patch_router"][
                        "test_expert_reliability_rerank"
                    ] = test_expert_rerank
                if patch_router_walk_forward_feedback_ridge:
                    test_feedback_ridge = _causal_expert_feedback_ridge_metrics(
                        train_time_n=torch.cat(
                            [train_time_n, val_time_n],
                            dim=0,
                        ),
                        train_base_mse_ncq=train_val_history("base_mse"),
                        train_candidate_mse_ncqp=train_val_history(
                            "candidate_mse_all"
                        ),
                        train_base_mae_ncq=train_val_history("base_mae"),
                        train_candidate_mae_ncqp=train_val_history(
                            "candidate_mae_all"
                        ),
                        train_score_ncqp=train_val_history(
                            "candidate_score_all"
                        ),
                        eval_time_n=test_time_n,
                        eval_base_mse_ncq=test_risk_tensors["base_mse"],
                        eval_candidate_mse_ncqp=test_risk_tensors[
                            "candidate_mse_all"
                        ],
                        eval_base_mae_ncq=test_risk_tensors["base_mae"],
                        eval_candidate_mae_ncqp=test_risk_tensors[
                            "candidate_mae_all"
                        ],
                        eval_score_ncqp=test_risk_tensors[
                            "candidate_score_all"
                        ],
                        active_channel_mask_c=walk_forward_active_channel_c,
                        label_delay=patch_router_walk_forward_label_delay,
                        lookback_windows=patch_router_walk_forward_lookback,
                        min_history_windows=patch_router_walk_forward_min_history,
                        history_stride=patch_router_walk_forward_history_stride,
                        ridge=patch_router_walk_forward_feedback_ridge_strength,
                        target_clip=patch_router_walk_forward_feedback_target_clip,
                        temporal_blocks=patch_router_walk_forward_temporal_blocks,
                    )
                    test_feedback_ridge["test_read"] = True
                    test_feedback_ridge["history_splits"] = [
                        "train",
                        "validation",
                    ]
                    moe_residual_summary["patch_router"][
                        "test_expert_feedback_ridge"
                    ] = test_feedback_ridge
    if bool(portrait_cfg.get("enable", False)) and (avg_probs_summary is not None) and len(penalty_names) > 0:
        portrait_dir = portrait_cfg.get("out_dir", os.path.join(out_dir, "cluster_portraits"))
        portrait_dpi = int(portrait_cfg.get("dpi", 140))
        max_points = int(portrait_cfg.get("max_points", 2000))
        jump_thr = float(portrait_cfg.get("jump_threshold", cfg.get("penalties", {}).get("jump_threshold", 2.0)))
        paths = save_cluster_portraits(
            out_dir=portrait_dir,
            data_tc=data_tc,
            cluster_id_c=cluster_id_c,
            jump_thr=jump_thr,
            dpi=portrait_dpi,
            max_points=max_points,
            metric_names=penalty_names,
            metric_values_km=avg_probs_summary,
            portrait_title="expert selection portrait (p)",
            metric_scale_mode="raw_0_1",
        )
        print(f"Updated cluster portraits with expert selection radar: {paths['dir']}")
    plot_cfg = cfg.get("plot", {}) or {}
    plot_enable = bool(plot_cfg.get("enable", False))
    random_n = int(plot_cfg.get("random_n", 0))
    plot_idx = None
    if plot_enable and len(dte) > 0 and random_n > 0:
        rng = np.random.default_rng(int(cfg["exp"]["seed"]))
        idxs = rng.choice(len(dte), size=min(random_n, len(dte)), replace=False)
        plot_idx = torch.tensor(sorted([int(i) for i in idxs]), device=device, dtype=torch.long)

    val_summary = None
    val_mse_c_base = None
    val_mae_c_base = None
    pred_residual_channel_scale_c = None
    pred_residual_selector_model = None
    pred_residual_selector_summary = None
    pred_residual_selection_summary = None
    learnable_output_anchor_refiner_summary = None
    learnable_output_anchor_test_refiner_summary = None
    moe_gate_penalty_hit_summary = None
    penalty_explainability_summary = None
    penalty_route_learnability_summary = None
    mae_eval_weight = _scale_mae_objective_weight(
        mae_objective_weight_final if mae_objective_enable else 0.0,
        mae_objective_multiplier_k,
    )
    if skip_test:
        print("eval.skip_test=true: test split windows, evaluation, and metrics are disabled.")
    if learnable_output_anchor is not None and not periodic_anchor_source_mask_preserved:
        learnable_output_anchor.clear_active_channel_mask()
        learnable_output_anchor.clear_active_channel_horizon_mask()
    if len(dva) > 0:
        val_loader_summary = DataLoader(dva, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)
        val_loss_best_k, val_mse_best_k, val_mae_best_k, val_mse_c_base, val_mae_c_base, _, _, _ = eval_loop_with_history(
            model, gate, lam_kp_best,
            penalty_names, penalty_fns,
            val_loader_summary, cluster_id_c, K, moe_cfg, device,
            select_ranks=select_ranks,
            collect_plot=False, channel_count=C,
            mse_weight=mse_weight,
            gate_entropy_weight=gate_entropy_weight,
            gate_balance_weight=gate_balance_weight,
            gate_soft_weight=gate_soft_weight,
            gate_entropy_target_frac=gate_entropy_target_frac,
            penalty_scale=penalty_scale,
            dynamic_lambda=dynamic_lambda,
            lambda_min_kp=lambda_min_kp,
            mae_objective_weight=mae_eval_weight,
            mae_objective_kind=mae_objective_kind,
            mae_objective_beta=mae_objective_beta,
            pred_residual=pred_residual,
            eval_start=val_eval_start,
        )
        val_summary = {
            "avg_loss": float(reduce_cluster_metric(val_loss_best_k, cluster_weight_k).item()),
            "avg_mse": float(reduce_cluster_metric(val_mse_best_k, cluster_weight_k).item()),
            "avg_mae": float(reduce_cluster_metric(val_mae_best_k, cluster_weight_k).item()),
            "per_cluster_loss": [float(v) for v in val_loss_best_k.detach().cpu().tolist()],
            "per_cluster_mse": [float(v) for v in val_mse_best_k.detach().cpu().tolist()],
            "per_cluster_mae": [float(v) for v in val_mae_best_k.detach().cpu().tolist()],
            "per_channel_mse": [float(v) for v in val_mse_c_base.detach().cpu().tolist()],
            "per_channel_mae": [float(v) for v in val_mae_c_base.detach().cpu().tolist()],
        }
        if learnable_output_anchor is not None:
            (
                val_loss_static_k,
                val_mse_static_k,
                val_mae_static_k,
                val_mse_c_static,
                val_mae_c_static,
                _,
                _,
                _,
            ) = eval_loop_with_history(
                model, gate, lam_kp_best,
                penalty_names, penalty_fns,
                val_loader_summary, cluster_id_c, K, moe_cfg, device,
                select_ranks=select_ranks,
                collect_plot=False, channel_count=C,
                mse_weight=mse_weight,
                gate_entropy_weight=gate_entropy_weight,
                gate_balance_weight=gate_balance_weight,
                gate_soft_weight=gate_soft_weight,
                gate_entropy_target_frac=gate_entropy_target_frac,
                penalty_scale=penalty_scale,
                dynamic_lambda=dynamic_lambda,
                lambda_min_kp=lambda_min_kp,
                mae_objective_weight=mae_eval_weight,
                mae_objective_kind=mae_objective_kind,
                mae_objective_beta=mae_objective_beta,
                pred_residual=pred_residual,
                eval_start=val_eval_start,
                learnable_output_anchor=None,
            )
            static_val_summary = {
                "avg_loss": float(reduce_cluster_metric(val_loss_static_k, cluster_weight_k).item()),
                "avg_mse": float(reduce_cluster_metric(val_mse_static_k, cluster_weight_k).item()),
                "avg_mae": float(reduce_cluster_metric(val_mae_static_k, cluster_weight_k).item()),
                "per_cluster_loss": [float(v) for v in val_loss_static_k.detach().cpu().tolist()],
                "per_cluster_mse": [float(v) for v in val_mse_static_k.detach().cpu().tolist()],
                "per_cluster_mae": [float(v) for v in val_mae_static_k.detach().cpu().tolist()],
                "per_channel_mse": [float(v) for v in val_mse_c_static.detach().cpu().tolist()],
                "per_channel_mae": [float(v) for v in val_mae_c_static.detach().cpu().tolist()],
            }
            learnable_adoption_cfg = learnable_output_anchor_cfg.get("adoption", {}) or {}
            if not isinstance(learnable_adoption_cfg, dict):
                learnable_adoption_cfg = {"adopt_on_val": bool(learnable_adoption_cfg)}
            segment_count_cfg = int(learnable_adoption_cfg.get("eval_segments", 4))
            segment_ranges = _contiguous_segment_ranges(len(dva), segment_count_cfg)

            def _collect_learnable_segment_metrics(
                *,
                collect_channel_horizon: bool = False,
            ) -> Tuple[
                List[Dict[str, float]],
                List[Dict[str, torch.Tensor]],
                List[Dict[str, torch.Tensor]],
            ]:
                metrics: List[Dict[str, float]] = []
                channel_metrics: List[Dict[str, torch.Tensor]] = []
                channel_horizon_metrics: List[Dict[str, torch.Tensor]] = []
                if len(segment_ranges) <= 1:
                    return metrics, channel_metrics, channel_horizon_metrics
                for segment_start, segment_end in segment_ranges:
                    segment_loader = DataLoader(
                        Subset(dva, range(segment_start, segment_end)),
                        batch_size=int(cfg["train"]["batch_size"]),
                        shuffle=False,
                        num_workers=0,
                        pin_memory=pin_mem,
                    )
                    refined_ch_collector: Optional[Dict[str, object]] = {} if collect_channel_horizon else None
                    (
                        _,
                        segment_mse_refined_k,
                        segment_mae_refined_k,
                        segment_mse_refined_c,
                        segment_mae_refined_c,
                        _,
                        _,
                        _,
                    ) = eval_loop_with_history(
                        model, gate, lam_kp_best,
                        penalty_names, penalty_fns,
                        segment_loader, cluster_id_c, K, moe_cfg, device,
                        select_ranks=select_ranks,
                        collect_plot=False, channel_count=C,
                        mse_weight=mse_weight,
                        gate_entropy_weight=gate_entropy_weight,
                        gate_balance_weight=gate_balance_weight,
                        gate_soft_weight=gate_soft_weight,
                        gate_entropy_target_frac=gate_entropy_target_frac,
                        penalty_scale=penalty_scale,
                        dynamic_lambda=dynamic_lambda,
                        lambda_min_kp=lambda_min_kp,
                        mae_objective_weight=mae_eval_weight,
                        mae_objective_kind=mae_objective_kind,
                        mae_objective_beta=mae_objective_beta,
                        pred_residual=pred_residual,
                        eval_start=val_eval_start,
                        learnable_output_anchor=learnable_output_anchor,
                        channel_horizon_metric_collector=refined_ch_collector,
                    )
                    static_ch_collector: Optional[Dict[str, object]] = {} if collect_channel_horizon else None
                    (
                        _,
                        segment_mse_static_k,
                        segment_mae_static_k,
                        segment_mse_static_c,
                        segment_mae_static_c,
                        _,
                        _,
                        _,
                    ) = eval_loop_with_history(
                        model, gate, lam_kp_best,
                        penalty_names, penalty_fns,
                        segment_loader, cluster_id_c, K, moe_cfg, device,
                        select_ranks=select_ranks,
                        collect_plot=False, channel_count=C,
                        mse_weight=mse_weight,
                        gate_entropy_weight=gate_entropy_weight,
                        gate_balance_weight=gate_balance_weight,
                        gate_soft_weight=gate_soft_weight,
                        gate_entropy_target_frac=gate_entropy_target_frac,
                        penalty_scale=penalty_scale,
                        dynamic_lambda=dynamic_lambda,
                        lambda_min_kp=lambda_min_kp,
                        mae_objective_weight=mae_eval_weight,
                        mae_objective_kind=mae_objective_kind,
                        mae_objective_beta=mae_objective_beta,
                        pred_residual=pred_residual,
                        eval_start=val_eval_start,
                        learnable_output_anchor=None,
                        channel_horizon_metric_collector=static_ch_collector,
                    )
                    metrics.append(
                        {
                            "start": float(segment_start),
                            "end": float(segment_end),
                            "static_mse": float(
                                reduce_cluster_metric(segment_mse_static_k, cluster_weight_k).item()
                            ),
                            "static_mae": float(
                                reduce_cluster_metric(segment_mae_static_k, cluster_weight_k).item()
                            ),
                            "refined_mse": float(
                                reduce_cluster_metric(segment_mse_refined_k, cluster_weight_k).item()
                            ),
                            "refined_mae": float(
                                reduce_cluster_metric(segment_mae_refined_k, cluster_weight_k).item()
                            ),
                        }
                    )
                    channel_metrics.append(
                        {
                            "static_mse_c": segment_mse_static_c.detach().cpu(),
                            "static_mae_c": segment_mae_static_c.detach().cpu(),
                            "refined_mse_c": segment_mse_refined_c.detach().cpu(),
                            "refined_mae_c": segment_mae_refined_c.detach().cpu(),
                        }
                    )
                    if collect_channel_horizon and refined_ch_collector is not None and static_ch_collector is not None:
                        segment_mse_refined_ch, segment_mae_refined_ch = _finalize_channel_horizon_metric_collector(
                            refined_ch_collector
                        )
                        segment_mse_static_ch, segment_mae_static_ch = _finalize_channel_horizon_metric_collector(
                            static_ch_collector
                        )
                        channel_horizon_metrics.append(
                            {
                                "static_mse_ch": segment_mse_static_ch,
                                "static_mae_ch": segment_mae_static_ch,
                                "refined_mse_ch": segment_mse_refined_ch,
                                "refined_mae_ch": segment_mae_refined_ch,
                            }
                        )
                return metrics, channel_metrics, channel_horizon_metrics

            segment_metrics, segment_channel_metrics, segment_channel_horizon_metrics = _collect_learnable_segment_metrics()
            unmasked_val_summary = dict(val_summary)
            adopted_channel_mask = None
            adopted_channel_horizon_mask = None
            learnable_channel_adoption_summary = None
            learnable_channel_horizon_adoption_summary = None
            adoption_scope = str(learnable_adoption_cfg.get("adoption_scope", "global")).lower()
            adoption_scope_norm = _normalize_learnable_output_anchor_adoption_scope(adoption_scope)
            apply_selected_learnable_mask = False

            def _collect_learnable_channel_horizon_metrics(
                *,
                use_learnable: bool,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                metric_collector: Dict[str, object] = {}
                eval_loop_with_history(
                    model, gate, lam_kp_best,
                    penalty_names, penalty_fns,
                    val_loader_summary, cluster_id_c, K, moe_cfg, device,
                    select_ranks=select_ranks,
                    collect_plot=False, channel_count=C,
                    mse_weight=mse_weight,
                    gate_entropy_weight=gate_entropy_weight,
                    gate_balance_weight=gate_balance_weight,
                    gate_soft_weight=gate_soft_weight,
                    gate_entropy_target_frac=gate_entropy_target_frac,
                    penalty_scale=penalty_scale,
                    dynamic_lambda=dynamic_lambda,
                    lambda_min_kp=lambda_min_kp,
                    mae_objective_weight=mae_eval_weight,
                    mae_objective_kind=mae_objective_kind,
                    mae_objective_beta=mae_objective_beta,
                    pred_residual=pred_residual,
                    eval_start=val_eval_start,
                    learnable_output_anchor=learnable_output_anchor if use_learnable else None,
                    channel_horizon_metric_collector=metric_collector,
                )
                return _finalize_channel_horizon_metric_collector(metric_collector)

            if periodic_anchor_source_mask_preserved:
                adopted_channel_mask = [
                    bool(v)
                    for v in learnable_output_anchor.active_channel_mask_c.detach().cpu().tolist()
                ]
            elif adoption_scope_norm in {"channel", "hybrid"}:
                keep_c, learnable_channel_adoption_summary = _select_learnable_output_anchor_channel_mask(
                    static_mse_c=val_mse_c_static,
                    refined_mse_c=val_mse_c_base,
                    static_mae_c=val_mae_c_static,
                    refined_mae_c=val_mae_c_base,
                    segment_channel_metrics=segment_channel_metrics,
                    adoption_cfg=learnable_adoption_cfg,
                )
                learnable_output_anchor.set_active_channel_mask(keep_c.to(device=device, dtype=torch.float32))
                learnable_output_anchor.clear_active_channel_horizon_mask()
                adopted_channel_mask = [bool(v) for v in keep_c.tolist()]
                apply_selected_learnable_mask = True
            elif adoption_scope_norm == "channel_horizon":
                learnable_output_anchor.clear_active_channel_mask()
                learnable_output_anchor.clear_active_channel_horizon_mask()
                segment_metrics, segment_channel_metrics, segment_channel_horizon_metrics = (
                    _collect_learnable_segment_metrics(collect_channel_horizon=True)
                )
                static_mse_ch, static_mae_ch = _collect_learnable_channel_horizon_metrics(use_learnable=False)
                refined_mse_ch, refined_mae_ch = _collect_learnable_channel_horizon_metrics(use_learnable=True)
                keep_ch, learnable_channel_horizon_adoption_summary = (
                    _select_learnable_output_anchor_channel_horizon_mask(
                        static_mse_ch=static_mse_ch,
                        refined_mse_ch=refined_mse_ch,
                        static_mae_ch=static_mae_ch,
                        refined_mae_ch=refined_mae_ch,
                        adoption_cfg=learnable_adoption_cfg,
                        segment_channel_horizon_metrics=segment_channel_horizon_metrics,
                    )
                )
                learnable_output_anchor.set_active_channel_horizon_mask(
                    keep_ch.to(device=device, dtype=torch.float32)
                )
                adopted_channel_horizon_mask = [
                    [bool(v) for v in row] for row in keep_ch.tolist()
                ]
                adopted_channel_mask = [any(row) for row in adopted_channel_horizon_mask]
                apply_selected_learnable_mask = True

            if apply_selected_learnable_mask:
                (
                    val_loss_best_k,
                    val_mse_best_k,
                    val_mae_best_k,
                    val_mse_c_base,
                    val_mae_c_base,
                    _,
                    _,
                    _,
                ) = eval_loop_with_history(
                    model, gate, lam_kp_best,
                    penalty_names, penalty_fns,
                    val_loader_summary, cluster_id_c, K, moe_cfg, device,
                    select_ranks=select_ranks,
                    collect_plot=False, channel_count=C,
                    mse_weight=mse_weight,
                    gate_entropy_weight=gate_entropy_weight,
                    gate_balance_weight=gate_balance_weight,
                    gate_soft_weight=gate_soft_weight,
                    gate_entropy_target_frac=gate_entropy_target_frac,
                    penalty_scale=penalty_scale,
                    dynamic_lambda=dynamic_lambda,
                    lambda_min_kp=lambda_min_kp,
                    mae_objective_weight=mae_eval_weight,
                    mae_objective_kind=mae_objective_kind,
                    mae_objective_beta=mae_objective_beta,
                    pred_residual=pred_residual,
                    eval_start=val_eval_start,
                    learnable_output_anchor=learnable_output_anchor,
                )
                val_summary = {
                    "avg_loss": float(reduce_cluster_metric(val_loss_best_k, cluster_weight_k).item()),
                    "avg_mse": float(reduce_cluster_metric(val_mse_best_k, cluster_weight_k).item()),
                    "avg_mae": float(reduce_cluster_metric(val_mae_best_k, cluster_weight_k).item()),
                    "per_cluster_loss": [float(v) for v in val_loss_best_k.detach().cpu().tolist()],
                    "per_cluster_mse": [float(v) for v in val_mse_best_k.detach().cpu().tolist()],
                    "per_cluster_mae": [float(v) for v in val_mae_best_k.detach().cpu().tolist()],
                    "per_channel_mse": [float(v) for v in val_mse_c_base.detach().cpu().tolist()],
                    "per_channel_mae": [float(v) for v in val_mae_c_base.detach().cpu().tolist()],
                }
                segment_metrics, _, _ = _collect_learnable_segment_metrics()
            learnable_summary_cfg = learnable_output_anchor_cfg
            if periodic_anchor_source_mask_preserved:
                learnable_summary_cfg = dict(learnable_output_anchor_cfg)
                preserved_adoption_cfg = dict(learnable_adoption_cfg)
                preserved_adoption_cfg["adopt_on_val"] = False
                preserved_adoption_cfg["adoption_scope"] = "global"
                learnable_summary_cfg["adoption"] = preserved_adoption_cfg
            learnable_output_anchor_refiner_summary = _summarize_learnable_output_anchor_refiner(
                static_mse=float(static_val_summary["avg_mse"]),
                static_mae=float(static_val_summary["avg_mae"]),
                refined_mse=float(val_summary["avg_mse"]),
                refined_mae=float(val_summary["avg_mae"]),
                unmasked_refined_mse=float(unmasked_val_summary["avg_mse"]),
                unmasked_refined_mae=float(unmasked_val_summary["avg_mae"]),
                cfg=learnable_summary_cfg,
                skip_test=skip_test,
                num_channels=C,
                segment_metrics=segment_metrics,
                adopted_channel_mask=adopted_channel_mask,
                adopted_channel_horizon_mask=adopted_channel_horizon_mask,
            )
            if learnable_channel_adoption_summary is not None:
                learnable_output_anchor_refiner_summary["channel_adoption"] = (
                    learnable_channel_adoption_summary
                )
            if learnable_channel_horizon_adoption_summary is not None:
                learnable_output_anchor_refiner_summary["channel_horizon_adoption"] = (
                    learnable_channel_horizon_adoption_summary
                )
            learnable_output_anchor_refiner_summary["periodic_source_mask_preserved"] = bool(
                periodic_anchor_source_mask_preserved
            )
            learnable_output_anchor_summary["adoption_guard_applied"] = not bool(
                periodic_anchor_source_mask_preserved
            )
            learnable_output_anchor_summary["adopted_on_val"] = bool(
                learnable_output_anchor_refiner_summary["adopted"]
            )
            if not bool(learnable_output_anchor_refiner_summary["final_eval_uses_learnable"]):
                learnable_output_anchor_summary["final_eval_enable"] = False
                learnable_output_anchor_summary["final_eval_reason"] = str(
                    learnable_output_anchor_refiner_summary["fallback_reason"]
                )
                val_summary = static_val_summary
                val_mse_c_base = val_mse_c_static
                val_mae_c_base = val_mae_c_static
                learnable_output_anchor.clear_active_channel_mask()
                learnable_output_anchor.clear_active_channel_horizon_mask()
                best_checkpoint_learnable_output_anchor_state = _clone_module_state_dict(learnable_output_anchor)
                learnable_output_anchor = None
                print(
                    "Learnable output anchor rejected by val guard; "
                    "final evaluation falls back to static anchors."
                )
            else:
                learnable_output_anchor_summary["final_eval_enable"] = True
                learnable_output_anchor_summary["final_eval_reason"] = (
                    "frozen_periodic_source_mask"
                    if periodic_anchor_source_mask_preserved
                    else "val_guard_adopted"
                    if bool(learnable_output_anchor_refiner_summary["adopted"])
                    else "adopt_on_val_disabled"
                )
                best_checkpoint_learnable_output_anchor_state = _clone_module_state_dict(learnable_output_anchor)
            if (
                best_checkpoint_path is not None
                and best_checkpoint_meta is not None
                and best_checkpoint_model_state is not None
                and best_checkpoint_gate_state is not None
            ):
                best_checkpoint_meta["learnable_output_anchor_refiner"] = dict(
                    learnable_output_anchor_refiner_summary
                )
                best_checkpoint_meta["learnable_output_anchor_final_eval_enable"] = bool(
                    learnable_output_anchor_refiner_summary["final_eval_uses_learnable"]
                )
                best_checkpoint_meta["learnable_output_anchor_state_status"] = (
                    "trained_refiner_state_adopted"
                    if bool(learnable_output_anchor_refiner_summary["final_eval_uses_learnable"])
                    else "trained_refiner_state_rejected_by_val_guard"
                )
                save_cluster_checkpoint(
                    best_checkpoint_path,
                    best_checkpoint_model_state,
                    best_checkpoint_gate_state,
                    best_checkpoint_meta,
                    pred_residual_state=best_checkpoint_pred_residual_state,
                    dynamic_lambda_state=best_checkpoint_dynamic_lambda_state,
                    learnable_lambda_state=best_checkpoint_learnable_lambda_state,
                    learnable_output_anchor_state=best_checkpoint_learnable_output_anchor_state,
                )
                print(
                    "Updated best checkpoint learnable-output-anchor adoption metadata: "
                    f"{best_checkpoint_path}"
                )
            if not skip_test and len(dte) > 0:
                test_loader_summary = DataLoader(
                    dte,
                    batch_size=int(cfg["train"]["batch_size"]),
                    shuffle=False,
                    num_workers=0,
                    pin_memory=pin_mem,
                )
                (
                    _,
                    test_mse_static_k,
                    test_mae_static_k,
                    test_mse_static_c,
                    test_mae_static_c,
                    _,
                    _,
                    _,
                ) = eval_loop_with_history(
                    model, gate, lam_kp_best,
                    penalty_names, penalty_fns,
                    test_loader_summary, cluster_id_c, K, moe_cfg, device,
                    select_ranks=select_ranks,
                    collect_plot=False, channel_count=C,
                    mse_weight=mse_weight,
                    gate_entropy_weight=gate_entropy_weight,
                    gate_balance_weight=gate_balance_weight,
                    gate_soft_weight=gate_soft_weight,
                    gate_entropy_target_frac=gate_entropy_target_frac,
                    penalty_scale=penalty_scale,
                    dynamic_lambda=dynamic_lambda,
                    lambda_min_kp=lambda_min_kp,
                    mae_objective_weight=mae_eval_weight,
                    mae_objective_kind=mae_objective_kind,
                    mae_objective_beta=mae_objective_beta,
                    pred_residual=pred_residual,
                    eval_start=test_eval_start,
                    learnable_output_anchor=None,
                )
                if learnable_output_anchor is None:
                    test_mse_refined_k = test_mse_static_k
                    test_mae_refined_k = test_mae_static_k
                    test_mse_refined_c = test_mse_static_c
                    test_mae_refined_c = test_mae_static_c
                else:
                    (
                        _,
                        test_mse_refined_k,
                        test_mae_refined_k,
                        test_mse_refined_c,
                        test_mae_refined_c,
                        _,
                        _,
                        _,
                    ) = eval_loop_with_history(
                        model, gate, lam_kp_best,
                        penalty_names, penalty_fns,
                        test_loader_summary, cluster_id_c, K, moe_cfg, device,
                        select_ranks=select_ranks,
                        collect_plot=False, channel_count=C,
                        mse_weight=mse_weight,
                        gate_entropy_weight=gate_entropy_weight,
                        gate_balance_weight=gate_balance_weight,
                        gate_soft_weight=gate_soft_weight,
                        gate_entropy_target_frac=gate_entropy_target_frac,
                        penalty_scale=penalty_scale,
                        dynamic_lambda=dynamic_lambda,
                        lambda_min_kp=lambda_min_kp,
                        mae_objective_weight=mae_eval_weight,
                        mae_objective_kind=mae_objective_kind,
                        mae_objective_beta=mae_objective_beta,
                        pred_residual=pred_residual,
                        eval_start=test_eval_start,
                        learnable_output_anchor=learnable_output_anchor,
                    )
                test_static_mse = float(reduce_cluster_metric(test_mse_static_k, cluster_weight_k).item())
                test_static_mae = float(reduce_cluster_metric(test_mae_static_k, cluster_weight_k).item())
                test_refined_mse = float(reduce_cluster_metric(test_mse_refined_k, cluster_weight_k).item())
                test_refined_mae = float(reduce_cluster_metric(test_mae_refined_k, cluster_weight_k).item())
                test_mse_gain = test_static_mse - test_refined_mse
                test_mae_gain = test_static_mae - test_refined_mae

                def _rel_pct(delta: float, denom: float) -> Optional[float]:
                    denom = abs(float(denom))
                    if denom <= 0.0:
                        return None
                    return 100.0 * float(delta) / denom

                learnable_output_anchor_test_refiner_summary = {
                    "enable": True,
                    "test_read": True,
                    "selection_source": "val_guard_only",
                    "final_eval_uses_learnable": bool(
                        learnable_output_anchor_refiner_summary["final_eval_uses_learnable"]
                    ),
                    "val_adopted": bool(learnable_output_anchor_refiner_summary["adopted"]),
                    "test_static_mse": test_static_mse,
                    "test_static_mae": test_static_mae,
                    "test_refined_mse": test_refined_mse,
                    "test_refined_mae": test_refined_mae,
                    "test_mse_gain": float(test_mse_gain),
                    "test_mae_gain": float(test_mae_gain),
                    "test_mse_gain_rel_pct": _rel_pct(test_mse_gain, test_static_mse),
                    "test_mae_gain_rel_pct": _rel_pct(test_mae_gain, test_static_mae),
                    "per_cluster_static_mse": [float(v) for v in test_mse_static_k.detach().cpu().tolist()],
                    "per_cluster_static_mae": [float(v) for v in test_mae_static_k.detach().cpu().tolist()],
                    "per_cluster_refined_mse": [float(v) for v in test_mse_refined_k.detach().cpu().tolist()],
                    "per_cluster_refined_mae": [float(v) for v in test_mae_refined_k.detach().cpu().tolist()],
                    "per_channel_static_mse": [float(v) for v in test_mse_static_c.detach().cpu().tolist()],
                    "per_channel_static_mae": [float(v) for v in test_mae_static_c.detach().cpu().tolist()],
                    "per_channel_refined_mse": [float(v) for v in test_mse_refined_c.detach().cpu().tolist()],
                    "per_channel_refined_mae": [float(v) for v in test_mae_refined_c.detach().cpu().tolist()],
                }
                print(
                    "Learnable output anchor test check: "
                    f"static={test_static_mse:.6f}/{test_static_mae:.6f}, "
                    f"refined={test_refined_mse:.6f}/{test_refined_mae:.6f}, "
                    f"gain={test_mse_gain:.6f}/{test_mae_gain:.6f}"
                )
        residual_selection_policy = _normalize_pred_residual_selection_policy(
            pred_residual_cfg.get("selection_policy", "none")
        )
        if residual_selection_policy not in {
            "none",
            "val_mse_channel",
            "val_mse_scale",
            "val_mse_scale_holdout",
            "val_mse_candidate_channel",
            "val_mae_candidate_channel",
        }:
            raise ValueError(
                "Unsupported moe.pred_side_residual.selection_policy="
                f"'{residual_selection_policy}'. Expected none, val_mse_channel, val_mse_scale, "
                "val_mse_scale_holdout, val_mse_candidate_channel, val_mae_candidate_channel, or "
                "val_mse_candidate_channel_guarded."
            )
        if pred_residual is not None and residual_selection_policy in {
            "val_mse_channel",
            "val_mse_scale",
            "val_mse_scale_holdout",
            "val_mse_candidate_channel",
            "val_mae_candidate_channel",
        }:
            zero_residual_scale_c = torch.zeros(C, device=device, dtype=torch.float32)
            residual_scale_mean_value = 0.0
            selection_max_residual_channels = int(pred_residual_cfg.get("selection_max_residual_channels", 0))
            selection_eval_segments = int(pred_residual_cfg.get("selection_eval_segments", 1))
            selection_min_positive_segments = int(pred_residual_cfg.get("selection_min_positive_segments", 0))
            selection_max_segment_rel_degradation = float(
                pred_residual_cfg.get("selection_max_segment_rel_degradation", 0.0)
            )
            selection_max_segment_abs_degradation = float(
                pred_residual_cfg.get("selection_max_segment_abs_degradation", 0.0)
            )
            selection_segment_improvement_mse_sc = None
            selection_segment_keep_c = None
            selection_eval_split = "val"
            selection_select_windows = len(dva)
            selection_eval_windows = len(dva)
            selection_eval_base_mse_c = None
            selection_eval_base_mae_c = None
            val_scaled_full_mse_c = None
            val_scaled_full_mae_c = None
            (
                val_loss_pred_base_k,
                val_mse_pred_base_k,
                val_mae_pred_base_k,
                val_mse_c_pred_base,
                val_mae_c_pred_base,
                _,
                _,
                _,
            ) = eval_loop_with_history(
                model, gate, lam_kp_best,
                penalty_names, penalty_fns,
                val_loader_summary, cluster_id_c, K, moe_cfg, device,
                select_ranks=select_ranks,
                collect_plot=False, channel_count=C,
                mse_weight=mse_weight,
                gate_entropy_weight=gate_entropy_weight,
                gate_balance_weight=gate_balance_weight,
                gate_soft_weight=gate_soft_weight,
                gate_entropy_target_frac=gate_entropy_target_frac,
                penalty_scale=penalty_scale,
                dynamic_lambda=dynamic_lambda,
                lambda_min_kp=lambda_min_kp,
                mae_objective_weight=mae_eval_weight,
                mae_objective_kind=mae_objective_kind,
                mae_objective_beta=mae_objective_beta,
                pred_residual=pred_residual,
                pred_residual_scale_c=zero_residual_scale_c,
                eval_start=val_eval_start,
            )
            val_scaled_mse_c = val_mse_c_base
            val_scaled_mae_c = val_mae_c_base
            candidate_channel_selector_summary = None
            if residual_selection_policy in {"val_mse_scale", "val_mse_scale_holdout"}:
                scale_min = float(pred_residual_cfg.get("selection_scale_min", 0.0))
                scale_max = float(pred_residual_cfg.get("selection_scale_max", 1.0))
                scale_steps = int(pred_residual_cfg.get("selection_scale_steps", 21))
                if scale_steps < 2:
                    raise ValueError("moe.pred_side_residual.selection_scale_steps must be >= 2")
                scale_select_loader = val_loader_summary
                scale_eval_loader = val_loader_summary
                scale_eval_start = val_eval_start
                scale_eval_base_mse_c = val_mse_c_pred_base
                scale_eval_base_mae_c = val_mae_c_pred_base
                if residual_selection_policy == "val_mse_scale_holdout":
                    holdout_fraction = float(pred_residual_cfg.get("selection_holdout_fraction", 0.4))
                    holdout_min_windows = int(pred_residual_cfg.get("selection_holdout_min_windows", 256))
                    select_n, holdout_n = _validation_holdout_split_counts(
                        len(dva),
                        holdout_fraction=holdout_fraction,
                        min_holdout=holdout_min_windows,
                    )
                    if holdout_n > 0:
                        scale_select_loader = DataLoader(
                            Subset(dva, range(0, select_n)),
                            batch_size=int(cfg["train"]["batch_size"]),
                            shuffle=False,
                            num_workers=0,
                            pin_memory=pin_mem,
                        )
                        scale_eval_loader = DataLoader(
                            Subset(dva, range(select_n, select_n + holdout_n)),
                            batch_size=int(cfg["train"]["batch_size"]),
                            shuffle=False,
                            num_workers=0,
                            pin_memory=pin_mem,
                        )
                        scale_eval_start = val_eval_start + select_n
                        selection_eval_split = "val_holdout"
                        selection_select_windows = select_n
                        selection_eval_windows = holdout_n
                        (
                            _,
                            _,
                            _,
                            scale_eval_base_mse_c,
                            scale_eval_base_mae_c,
                            _,
                            _,
                            _,
                        ) = eval_loop_with_history(
                            model, gate, lam_kp_best,
                            penalty_names, penalty_fns,
                            scale_eval_loader, cluster_id_c, K, moe_cfg, device,
                            select_ranks=select_ranks,
                            collect_plot=False, channel_count=C,
                            mse_weight=mse_weight,
                            gate_entropy_weight=gate_entropy_weight,
                            gate_balance_weight=gate_balance_weight,
                            gate_soft_weight=gate_soft_weight,
                            gate_entropy_target_frac=gate_entropy_target_frac,
                            penalty_scale=penalty_scale,
                            dynamic_lambda=dynamic_lambda,
                            lambda_min_kp=lambda_min_kp,
                            mae_objective_weight=mae_eval_weight,
                            mae_objective_kind=mae_objective_kind,
                            mae_objective_beta=mae_objective_beta,
                            pred_residual=pred_residual,
                            pred_residual_scale_c=zero_residual_scale_c,
                                                    eval_start=scale_eval_start,
                        )
                selection_eval_base_mse_c = scale_eval_base_mse_c
                selection_eval_base_mae_c = scale_eval_base_mae_c
                scale_grid = torch.linspace(scale_min, scale_max, scale_steps, device=device, dtype=torch.float32)
                best_mse_c = torch.full((C,), float("inf"), dtype=val_mse_c_pred_base.dtype)
                best_mae_c = torch.full((C,), float("inf"), dtype=val_mae_c_pred_base.dtype)
                best_scale_c = torch.zeros((C,), dtype=torch.float32)
                for scale_value in scale_grid.tolist():
                    scale_c = torch.full((C,), float(scale_value), device=device, dtype=torch.float32)
                    (
                        _,
                        _,
                        _,
                        cand_mse_c,
                        cand_mae_c,
                        _,
                        _,
                        _,
                    ) = eval_loop_with_history(
                        model, gate, lam_kp_best,
                        penalty_names, penalty_fns,
                        scale_select_loader, cluster_id_c, K, moe_cfg, device,
                        select_ranks=select_ranks,
                        collect_plot=False, channel_count=C,
                        mse_weight=mse_weight,
                        gate_entropy_weight=gate_entropy_weight,
                        gate_balance_weight=gate_balance_weight,
                        gate_soft_weight=gate_soft_weight,
                        gate_entropy_target_frac=gate_entropy_target_frac,
                        penalty_scale=penalty_scale,
                        dynamic_lambda=dynamic_lambda,
                        lambda_min_kp=lambda_min_kp,
                        mae_objective_weight=mae_eval_weight,
                        mae_objective_kind=mae_objective_kind,
                        mae_objective_beta=mae_objective_beta,
                        pred_residual=pred_residual,
                        pred_residual_scale_c=scale_c,
                                        eval_start=val_eval_start,
                    )
                    better = cand_mse_c < best_mse_c
                    best_mse_c = torch.where(better, cand_mse_c, best_mse_c)
                    best_mae_c = torch.where(better, cand_mae_c, best_mae_c)
                    best_scale_c = torch.where(
                        better,
                        torch.full_like(best_scale_c, float(scale_value)),
                        best_scale_c,
                    )
                pred_residual_channel_scale_c = best_scale_c.to(device=device, dtype=torch.float32)
                val_scaled_mse_c = best_mse_c
                val_scaled_mae_c = best_mae_c
                if residual_selection_policy == "val_mse_scale_holdout" and selection_eval_split == "val_holdout":
                    (
                        _,
                        _,
                        _,
                        holdout_scaled_mse_c,
                        holdout_scaled_mae_c,
                        _,
                        _,
                        _,
                    ) = eval_loop_with_history(
                        model, gate, lam_kp_best,
                        penalty_names, penalty_fns,
                        scale_eval_loader, cluster_id_c, K, moe_cfg, device,
                        select_ranks=select_ranks,
                        collect_plot=False, channel_count=C,
                        mse_weight=mse_weight,
                        gate_entropy_weight=gate_entropy_weight,
                        gate_balance_weight=gate_balance_weight,
                        gate_soft_weight=gate_soft_weight,
                        gate_entropy_target_frac=gate_entropy_target_frac,
                        penalty_scale=penalty_scale,
                        dynamic_lambda=dynamic_lambda,
                        lambda_min_kp=lambda_min_kp,
                        mae_objective_weight=mae_eval_weight,
                        mae_objective_kind=mae_objective_kind,
                        mae_objective_beta=mae_objective_beta,
                        pred_residual=pred_residual,
                        pred_residual_scale_c=pred_residual_channel_scale_c,
                                        eval_start=scale_eval_start,
                    )
                    min_abs = float(pred_residual_cfg.get("selection_min_abs_improvement", 0.0))
                    min_rel = float(pred_residual_cfg.get("selection_min_rel_improvement", 0.0))
                    required = torch.maximum(
                        torch.full_like(scale_eval_base_mse_c, min_abs),
                        min_rel * scale_eval_base_mse_c.abs().clamp_min(1.0e-12),
                    )
                    use_residual_device_c = (scale_eval_base_mse_c - holdout_scaled_mse_c) > required
                    segment_ranges = _contiguous_segment_ranges(selection_eval_windows, selection_eval_segments)
                    if len(segment_ranges) > 1:
                        segment_base_parts = []
                        segment_scaled_parts = []
                        for segment_start, segment_end in segment_ranges:
                            segment_loader = DataLoader(
                                Subset(dva, range(select_n + segment_start, select_n + segment_end)),
                                batch_size=int(cfg["train"]["batch_size"]),
                                shuffle=False,
                                num_workers=0,
                                pin_memory=pin_mem,
                            )
                            segment_eval_start = val_eval_start + select_n + segment_start
                            (
                                _,
                                _,
                                _,
                                segment_base_mse_c,
                                _,
                                _,
                                _,
                                _,
                            ) = eval_loop_with_history(
                                model, gate, lam_kp_best,
                                penalty_names, penalty_fns,
                                segment_loader, cluster_id_c, K, moe_cfg, device,
                                select_ranks=select_ranks,
                                collect_plot=False, channel_count=C,
                                mse_weight=mse_weight,
                                gate_entropy_weight=gate_entropy_weight,
                                gate_balance_weight=gate_balance_weight,
                                gate_soft_weight=gate_soft_weight,
                                gate_entropy_target_frac=gate_entropy_target_frac,
                                penalty_scale=penalty_scale,
                                dynamic_lambda=dynamic_lambda,
                                lambda_min_kp=lambda_min_kp,
                                mae_objective_weight=mae_eval_weight,
                                mae_objective_kind=mae_objective_kind,
                                mae_objective_beta=mae_objective_beta,
                                pred_residual=pred_residual,
                                pred_residual_scale_c=zero_residual_scale_c,
                                                                eval_start=segment_eval_start,
                            )
                            (
                                _,
                                _,
                                _,
                                segment_scaled_mse_c,
                                _,
                                _,
                                _,
                                _,
                            ) = eval_loop_with_history(
                                model, gate, lam_kp_best,
                                penalty_names, penalty_fns,
                                segment_loader, cluster_id_c, K, moe_cfg, device,
                                select_ranks=select_ranks,
                                collect_plot=False, channel_count=C,
                                mse_weight=mse_weight,
                                gate_entropy_weight=gate_entropy_weight,
                                gate_balance_weight=gate_balance_weight,
                                gate_soft_weight=gate_soft_weight,
                                gate_entropy_target_frac=gate_entropy_target_frac,
                                penalty_scale=penalty_scale,
                                dynamic_lambda=dynamic_lambda,
                                lambda_min_kp=lambda_min_kp,
                                mae_objective_weight=mae_eval_weight,
                                mae_objective_kind=mae_objective_kind,
                                mae_objective_beta=mae_objective_beta,
                                pred_residual=pred_residual,
                                pred_residual_scale_c=pred_residual_channel_scale_c,
                                                                eval_start=segment_eval_start,
                            )
                            segment_base_parts.append(segment_base_mse_c.detach().cpu())
                            segment_scaled_parts.append(segment_scaled_mse_c.detach().cpu())
                        segment_base_sc = torch.stack(segment_base_parts, dim=0)
                        segment_scaled_sc = torch.stack(segment_scaled_parts, dim=0)
                        selection_segment_improvement_mse_sc = segment_base_sc - segment_scaled_sc
                        segment_required_sc = torch.maximum(
                            torch.full_like(segment_base_sc, min_abs),
                            min_rel * segment_base_sc.abs().clamp_min(1.0e-12),
                        )
                        segment_keep_c = torch.ones(C, dtype=torch.bool)
                        if selection_min_positive_segments > 0:
                            positive_counts_c = (selection_segment_improvement_mse_sc > segment_required_sc).sum(dim=0)
                            segment_keep_c &= positive_counts_c >= int(selection_min_positive_segments)
                        allowed_degradation_sc = torch.maximum(
                            torch.full_like(segment_base_sc, max(0.0, selection_max_segment_abs_degradation)),
                            max(0.0, selection_max_segment_rel_degradation)
                            * segment_base_sc.abs().clamp_min(1.0e-12),
                        )
                        segment_keep_c &= (selection_segment_improvement_mse_sc >= -allowed_degradation_sc).all(dim=0)
                        selection_segment_keep_c = segment_keep_c
                        use_residual_device_c &= segment_keep_c.to(device=use_residual_device_c.device)
                    pred_residual_channel_scale_c = torch.where(
                        use_residual_device_c.to(device=device),
                        pred_residual_channel_scale_c,
                        zero_residual_scale_c,
                    )
                    val_scaled_mse_c = torch.where(use_residual_device_c, holdout_scaled_mse_c, scale_eval_base_mse_c)
                    val_scaled_mae_c = torch.where(use_residual_device_c, holdout_scaled_mae_c, scale_eval_base_mae_c)
                    (
                        _,
                        _,
                        _,
                        val_scaled_full_mse_c,
                        val_scaled_full_mae_c,
                        _,
                        _,
                        _,
                    ) = eval_loop_with_history(
                        model, gate, lam_kp_best,
                        penalty_names, penalty_fns,
                        val_loader_summary, cluster_id_c, K, moe_cfg, device,
                        select_ranks=select_ranks,
                        collect_plot=False, channel_count=C,
                        mse_weight=mse_weight,
                        gate_entropy_weight=gate_entropy_weight,
                        gate_balance_weight=gate_balance_weight,
                        gate_soft_weight=gate_soft_weight,
                        gate_entropy_target_frac=gate_entropy_target_frac,
                        penalty_scale=penalty_scale,
                        dynamic_lambda=dynamic_lambda,
                        lambda_min_kp=lambda_min_kp,
                        mae_objective_weight=mae_eval_weight,
                        mae_objective_kind=mae_objective_kind,
                        mae_objective_beta=mae_objective_beta,
                        pred_residual=pred_residual,
                        pred_residual_scale_c=pred_residual_channel_scale_c,
                                        eval_start=val_eval_start,
                    )
                if selection_max_residual_channels > 0:
                    improvement_c = selection_eval_base_mse_c.detach().cpu() - val_scaled_mse_c.detach().cpu()
                    limit_c = _top_positive_improvement_mask(improvement_c, selection_max_residual_channels)
                    active_c = pred_residual_channel_scale_c.detach().cpu() > 1.0e-8
                    keep_c = active_c & limit_c
                    keep_scale_c = keep_c.to(device=pred_residual_channel_scale_c.device)
                    keep_mse_c = keep_c.to(device=val_scaled_mse_c.device)
                    keep_mae_c = keep_c.to(device=val_scaled_mae_c.device)
                    pred_residual_channel_scale_c = torch.where(
                        keep_scale_c,
                        pred_residual_channel_scale_c,
                        zero_residual_scale_c,
                    )
                    val_scaled_mse_c = torch.where(
                        keep_mse_c,
                        val_scaled_mse_c,
                        selection_eval_base_mse_c.to(
                            device=val_scaled_mse_c.device,
                            dtype=val_scaled_mse_c.dtype,
                        ),
                    )
                    val_scaled_mae_c = torch.where(
                        keep_mae_c,
                        val_scaled_mae_c,
                        selection_eval_base_mae_c.to(
                            device=val_scaled_mae_c.device,
                            dtype=val_scaled_mae_c.dtype,
                        ),
                    )
                    if val_scaled_full_mse_c is not None and val_scaled_full_mae_c is not None:
                        keep_full_mse_c = keep_c.to(device=val_scaled_full_mse_c.device)
                        keep_full_mae_c = keep_c.to(device=val_scaled_full_mae_c.device)
                        val_scaled_full_mse_c = torch.where(
                            keep_full_mse_c,
                            val_scaled_full_mse_c,
                            val_mse_c_pred_base.to(
                                device=val_scaled_full_mse_c.device,
                                dtype=val_scaled_full_mse_c.dtype,
                            ),
                        )
                        val_scaled_full_mae_c = torch.where(
                            keep_full_mae_c,
                            val_scaled_full_mae_c,
                            val_mae_c_pred_base.to(
                                device=val_scaled_full_mae_c.device,
                                dtype=val_scaled_full_mae_c.dtype,
                            ),
                        )
                use_residual_c = pred_residual_channel_scale_c.detach().cpu() > 1.0e-8
                scale_values = [float(v) for v in pred_residual_channel_scale_c.detach().cpu().tolist()]
                residual_scale_mean_value = float(pred_residual_channel_scale_c.mean().item())
            elif residual_selection_policy in {"val_mse_candidate_channel", "val_mae_candidate_channel"}:
                selector_metric = str(
                    pred_residual_cfg.get(
                        "selection_metric",
                        "mae" if residual_selection_policy == "val_mae_candidate_channel" else "mse",
                    )
                )
                min_abs = float(pred_residual_cfg.get("selection_min_abs_improvement", 0.0))
                min_rel = float(pred_residual_cfg.get("selection_min_rel_improvement", 0.0))
                min_abs_mae_raw = pred_residual_cfg.get("selection_min_abs_mae_improvement", None)
                min_abs_mae = float(min_abs_mae_raw) if min_abs_mae_raw is not None else None
                confirm_fraction = float(pred_residual_cfg.get("selection_confirm_fraction", 0.0))
                confirm_min_abs_raw = pred_residual_cfg.get("selection_confirm_min_abs_improvement", None)
                confirm_min_abs = float(confirm_min_abs_raw) if confirm_min_abs_raw is not None else None
                confirm_min_rel = float(pred_residual_cfg.get("selection_confirm_min_rel_improvement", 0.0))
                confirm_min_abs_mae_raw = pred_residual_cfg.get("selection_confirm_min_abs_mae_improvement", None)
                confirm_min_abs_mae = (
                    float(confirm_min_abs_mae_raw) if confirm_min_abs_mae_raw is not None else None
                )
                segment_count = int(pred_residual_cfg.get("selection_segment_count", 0))
                segment_min_positive_raw = pred_residual_cfg.get("selection_segment_min_positive", None)
                segment_min_positive = (
                    int(segment_min_positive_raw) if segment_min_positive_raw is not None else None
                )
                segment_min_abs_raw = pred_residual_cfg.get("selection_segment_min_abs_improvement", None)
                segment_min_abs = float(segment_min_abs_raw) if segment_min_abs_raw is not None else None
                segment_min_abs_mae_raw = pred_residual_cfg.get("selection_segment_min_abs_mae_improvement", None)
                segment_min_abs_mae = (
                    float(segment_min_abs_mae_raw) if segment_min_abs_mae_raw is not None else None
                )
                allowed_mask_cp = None
                if cluster_penalty_allowed_mask_kp is not None and int(cluster_penalty_allowed_mask_kp.numel()) > 0:
                    allowed_kp = cluster_penalty_allowed_mask_kp.detach().cpu().to(dtype=torch.bool)
                    cluster_idx = cluster_id_c.detach().cpu().to(dtype=torch.long)
                    allowed_mask_cp = allowed_kp.index_select(0, cluster_idx)
                candidate_tensors = _collect_pred_residual_selector_tensors(
                    model=model,
                    pred_residual=pred_residual,
                    loader=val_loader_summary,
                    cluster_id_c=cluster_id_c,
                    K=K,
                    moe_cfg=moe_cfg,
                    device=device,
                    penalty_count=len(penalty_names),
                    pred_residual_scale_c=None,
                    history_anchor_cfg=history_anchor_cfg,
                    observed_history_tc=data_window_tc,
                    input_len=L,
                    eval_start=val_eval_start,
                    model_train_stat_adapter_pc=model_train_stat_adapter_pc,
                    model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
                    train_stat_anchor_pc=train_stat_anchor_pc,
                    train_residual_anchor_phc=train_residual_anchor_phc,
                    learnable_output_anchor=learnable_output_anchor,
                    candidate_feature_mode="base",
                )
                if candidate_tensors is None:
                    use_residual_c = torch.zeros(C, dtype=torch.bool)
                    scale_values = [0.0 for _ in range(C)]
                    residual_scale_mean_value = 0.0
                else:
                    candidate_select_indices, candidate_confirm_indices = _candidate_selector_select_confirm_indices(
                        int(candidate_tensors["base"].shape[0]),
                        confirm_fraction,
                    )
                    if candidate_confirm_indices is not None and confirm_min_abs is None:
                        confirm_min_abs = min_abs
                    static_selector, candidate_channel_selector_summary = _fit_static_candidate_channel_selector_from_tensors(
                        tensors=candidate_tensors,
                        allowed_mask_cp=allowed_mask_cp,
                        penalty_names=penalty_names,
                        channel_names=channel_names,
                        select_indices=candidate_select_indices,
                        eval_indices=candidate_confirm_indices,
                        min_abs_improvement=min_abs,
                        min_rel_improvement=min_rel,
                        min_abs_mae_improvement=min_abs_mae,
                        selection_metric=selector_metric,
                        confirm_min_abs_improvement=confirm_min_abs,
                        confirm_min_rel_improvement=confirm_min_rel,
                        confirm_min_abs_mae_improvement=confirm_min_abs_mae,
                        segment_count=segment_count,
                        segment_min_positive=segment_min_positive,
                        segment_min_abs_improvement=segment_min_abs,
                        segment_min_abs_mae_improvement=segment_min_abs_mae,
                    )
                    pred_residual_selector_model = static_selector.to(device)
                    pred_residual_channel_scale_c = None
                    (
                        _,
                        _,
                        _,
                        val_static_mse_c,
                        val_static_mae_c,
                        _,
                        _,
                        _,
                    ) = eval_loop_with_history(
                        model, gate, lam_kp_best,
                        penalty_names, penalty_fns,
                        val_loader_summary, cluster_id_c, K, moe_cfg, device,
                        select_ranks=select_ranks,
                        collect_plot=False, channel_count=C,
                        mse_weight=mse_weight,
                        gate_entropy_weight=gate_entropy_weight,
                        gate_balance_weight=gate_balance_weight,
                        gate_soft_weight=gate_soft_weight,
                        gate_entropy_target_frac=gate_entropy_target_frac,
                        penalty_scale=penalty_scale,
                        dynamic_lambda=dynamic_lambda,
                        lambda_min_kp=lambda_min_kp,
                        mae_objective_weight=mae_eval_weight,
                        mae_objective_kind=mae_objective_kind,
                        mae_objective_beta=mae_objective_beta,
                        pred_residual=pred_residual,
                        pred_residual_selector=pred_residual_selector_model,
                        pred_residual_scale_c=None,
                                        eval_start=val_eval_start,
                    )
                    val_scaled_mse_c = val_static_mse_c
                    val_scaled_mae_c = val_static_mae_c
                    selected_class_c = torch.tensor(
                        candidate_channel_selector_summary.get("selected_class", []),
                        dtype=torch.long,
                    )
                    use_residual_c = selected_class_c > 0
                    scale_values = [float(v) for v in selected_class_c.tolist()]
                    residual_scale_mean_value = float(use_residual_c.to(dtype=torch.float32).mean().item())
            else:
                min_abs = float(pred_residual_cfg.get("selection_min_abs_improvement", 0.0))
                min_rel = float(pred_residual_cfg.get("selection_min_rel_improvement", 0.0))
                required = torch.maximum(
                    torch.full_like(val_mse_c_pred_base, min_abs),
                    min_rel * val_mse_c_pred_base.abs().clamp_min(1.0e-12),
                )
                use_residual_c = (val_mse_c_pred_base - val_mse_c_base) > required
                val_scaled_mse_c, val_scaled_mae_c = _mix_selected_channel_metrics(
                    base_mse_c=val_mse_c_pred_base,
                    base_mae_c=val_mae_c_pred_base,
                    residual_mse_c=val_mse_c_base,
                    residual_mae_c=val_mae_c_base,
                    use_residual_c=use_residual_c,
                )
                pred_residual_channel_scale_c = use_residual_c.to(device=device, dtype=torch.float32)
                scale_values = [float(v) for v in pred_residual_channel_scale_c.detach().cpu().tolist()]
                residual_scale_mean_value = float(pred_residual_channel_scale_c.mean().item())
            pred_residual_selection_summary = {
                "policy": residual_selection_policy,
                "eval_split": selection_eval_split,
                "selection_windows": int(selection_select_windows),
                "eval_windows": int(selection_eval_windows),
                "max_residual_channels": int(selection_max_residual_channels),
                "eval_segments": int(selection_eval_segments),
                "min_positive_segments": int(selection_min_positive_segments),
                "max_segment_rel_degradation": float(selection_max_segment_rel_degradation),
                "max_segment_abs_degradation": float(selection_max_segment_abs_degradation),
                "min_abs_improvement": float(pred_residual_cfg.get("selection_min_abs_improvement", 0.0)),
                "min_rel_improvement": float(pred_residual_cfg.get("selection_min_rel_improvement", 0.0)),
                "max_abs_mse_regression": float(pred_residual_cfg.get("selection_max_abs_mse_regression", 0.0)),
                "max_rel_mse_regression": float(pred_residual_cfg.get("selection_max_rel_mse_regression", 0.0)),
                "scale_values": scale_values,
                "mean_scale": float(residual_scale_mean_value),
                "num_residual_channels": int(use_residual_c.sum().item()),
                "residual_channels": [
                    channel_names[i] for i, use_residual in enumerate(use_residual_c.tolist()) if bool(use_residual)
                ],
                "base_channels": [
                    channel_names[i] for i, use_residual in enumerate(use_residual_c.tolist()) if not bool(use_residual)
                ],
                "val_pred_base_avg_mse": float(reduce_cluster_metric(val_mse_pred_base_k, cluster_weight_k).item()),
                "val_pred_base_avg_mae": float(reduce_cluster_metric(val_mae_pred_base_k, cluster_weight_k).item()),
                "val_residual_avg_mse": float(reduce_cluster_metric(val_mse_best_k, cluster_weight_k).item()),
                "val_residual_avg_mae": float(reduce_cluster_metric(val_mae_best_k, cluster_weight_k).item()),
                "val_scaled_avg_mse": float(val_scaled_mse_c.mean().item()),
                "val_scaled_avg_mae": float(val_scaled_mae_c.mean().item()),
                "val_pred_base_mse_per_channel": [float(v) for v in val_mse_c_pred_base.detach().cpu().tolist()],
                "val_residual_mse_per_channel": [float(v) for v in val_mse_c_base.detach().cpu().tolist()],
                "val_scaled_mse_per_channel": [float(v) for v in val_scaled_mse_c.detach().cpu().tolist()],
                "val_pred_base_mae_per_channel": [float(v) for v in val_mae_c_pred_base.detach().cpu().tolist()],
                "val_residual_mae_per_channel": [float(v) for v in val_mae_c_base.detach().cpu().tolist()],
                "val_scaled_mae_per_channel": [float(v) for v in val_scaled_mae_c.detach().cpu().tolist()],
            }
            if val_scaled_full_mse_c is not None and val_scaled_full_mae_c is not None:
                pred_residual_selection_summary.update(
                    {
                        "val_scaled_full_avg_mse": float(val_scaled_full_mse_c.mean().item()),
                        "val_scaled_full_avg_mae": float(val_scaled_full_mae_c.mean().item()),
                        "val_scaled_full_mse_per_channel": [
                            float(v) for v in val_scaled_full_mse_c.detach().cpu().tolist()
                        ],
                        "val_scaled_full_mae_per_channel": [
                            float(v) for v in val_scaled_full_mae_c.detach().cpu().tolist()
                        ],
                    }
                )
            if candidate_channel_selector_summary is not None:
                pred_residual_selection_summary["candidate_channel_selector"] = candidate_channel_selector_summary
            if selection_segment_improvement_mse_sc is not None:
                pred_residual_selection_summary.update(
                    {
                        "segment_improvement_mse_per_channel": [
                            [float(v) for v in row]
                            for row in selection_segment_improvement_mse_sc.detach().cpu().tolist()
                        ],
                        "segment_keep_channels": [
                            bool(v) for v in selection_segment_keep_c.detach().cpu().tolist()
                        ]
                        if selection_segment_keep_c is not None
                        else [],
                    }
                )
            if selection_eval_base_mse_c is not None and selection_eval_base_mae_c is not None:
                pred_residual_selection_summary.update(
                    {
                        "eval_pred_base_avg_mse": float(selection_eval_base_mse_c.mean().item()),
                        "eval_pred_base_avg_mae": float(selection_eval_base_mae_c.mean().item()),
                        "eval_pred_base_mse_per_channel": [
                            float(v) for v in selection_eval_base_mse_c.detach().cpu().tolist()
                        ],
                        "eval_pred_base_mae_per_channel": [
                            float(v) for v in selection_eval_base_mae_c.detach().cpu().tolist()
                        ],
                    }
                )
            print(
                "Prediction residual selection: "
                f"policy={residual_selection_policy}, "
                f"eval_split={selection_eval_split}, "
                f"residual_channels={pred_residual_selection_summary['num_residual_channels']}/{C}, "
                f"val_base_MSE={pred_residual_selection_summary['val_pred_base_avg_mse']:.6f}, "
                f"val_residual_MSE={pred_residual_selection_summary['val_residual_avg_mse']:.6f}, "
                f"val_scaled_MSE={pred_residual_selection_summary['val_scaled_avg_mse']:.6f}, "
                f"mean_scale={pred_residual_selection_summary['mean_scale']:.3f}"
            )
        selector_cfg = pred_residual_cfg.get("candidate_selector", {}) or {}
        if pred_residual is not None and moe_enable and P > 0 and bool(selector_cfg.get("enable", False)):
            selector_source_split = str(selector_cfg.get("source_split", "val")).lower()
            selector_precollected_tensors = None
            if selector_source_split in {"train", "training"}:
                selector_loader = DataLoader(
                    dtr,
                    batch_size=int(cfg["train"]["batch_size"]),
                    shuffle=False,
                    num_workers=0,
                    pin_memory=pin_mem,
                )
                selector_source_split = "train"
                selector_eval_start = 0
            elif selector_source_split in {"val", "validation"}:
                selector_loader = val_loader_summary
                selector_source_split = "val"
                selector_eval_start = val_eval_start
            elif selector_source_split in {"train_val", "train+val", "trainval"}:
                selector_loader = val_loader_summary
                selector_source_split = "train_val"
                selector_eval_start = val_eval_start
            else:
                raise ValueError(
                    "moe.pred_side_residual.candidate_selector.source_split must be train, val, or train_val "
                    f"(got {selector_source_split!r})."
                )
            selector_candidate_scale_c, selector_candidate_scale_mode = _candidate_selector_candidate_scale(
                pred_residual_scale_c=pred_residual_channel_scale_c,
                selector_cfg=selector_cfg,
            )
            if selector_source_split == "train_val":
                train_selector_loader = DataLoader(
                    dtr,
                    batch_size=int(cfg["train"]["batch_size"]),
                    shuffle=False,
                    num_workers=0,
                    pin_memory=pin_mem,
                )
                train_selector_tensors = _collect_pred_residual_selector_tensors(
                    model=model,
                    pred_residual=pred_residual,
                    loader=train_selector_loader,
                    cluster_id_c=cluster_id_c,
                    K=K,
                    moe_cfg=moe_cfg,
                    device=device,
                    penalty_count=len(penalty_names),
                    pred_residual_scale_c=selector_candidate_scale_c,
                    history_anchor_cfg=history_anchor_cfg,
                    observed_history_tc=data_window_tc,
                    input_len=L,
                    eval_start=0,
                    model_train_stat_adapter_pc=model_train_stat_adapter_pc,
                    model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
                    train_stat_anchor_pc=train_stat_anchor_pc,
                    train_residual_anchor_phc=train_residual_anchor_phc,
                    learnable_output_anchor=learnable_output_anchor,
                    candidate_feature_mode=str(selector_cfg.get("feature_mode", "base")).lower(),
                )
                val_selector_tensors = _collect_pred_residual_selector_tensors(
                    model=model,
                    pred_residual=pred_residual,
                    loader=val_loader_summary,
                    cluster_id_c=cluster_id_c,
                    K=K,
                    moe_cfg=moe_cfg,
                    device=device,
                    penalty_count=len(penalty_names),
                    pred_residual_scale_c=selector_candidate_scale_c,
                    history_anchor_cfg=history_anchor_cfg,
                    observed_history_tc=data_window_tc,
                    input_len=L,
                    eval_start=val_eval_start,
                    model_train_stat_adapter_pc=model_train_stat_adapter_pc,
                    model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
                    train_stat_anchor_pc=train_stat_anchor_pc,
                    train_residual_anchor_phc=train_residual_anchor_phc,
                    learnable_output_anchor=learnable_output_anchor,
                    candidate_feature_mode=str(selector_cfg.get("feature_mode", "base")).lower(),
                )
                selector_precollected_tensors = _concat_pred_residual_selector_tensors(
                    [train_selector_tensors, val_selector_tensors]
                )
            candidate_selector_model, pred_residual_selector_summary = train_pred_residual_candidate_selector(
                model=model,
                pred_residual=pred_residual,
                loader=selector_loader,
                cluster_id_c=cluster_id_c,
                K=K,
                moe_cfg=moe_cfg,
                device=device,
                penalty_names=penalty_names,
                channel_names=channel_names,
                cfg=selector_cfg,
                pred_residual_scale_c=selector_candidate_scale_c,
                history_anchor_cfg=history_anchor_cfg,
                observed_history_tc=data_window_tc,
                input_len=L,
                eval_start=selector_eval_start,
                model_train_stat_adapter_pc=model_train_stat_adapter_pc,
                model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
                train_stat_anchor_pc=train_stat_anchor_pc,
                train_residual_anchor_phc=train_residual_anchor_phc,
                learnable_output_anchor=learnable_output_anchor,
                allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                gate_feature_mode=gate_feature_mode,
                precollected_tensors=selector_precollected_tensors,
            )
            if pred_residual_selector_summary is not None:
                pred_residual_selector_summary["source_split"] = selector_source_split
                pred_residual_selector_summary["candidate_scale_mode"] = selector_candidate_scale_mode
            if candidate_selector_model is not None:
                (
                    val_selector_loss_k,
                    val_selector_mse_k,
                    val_selector_mae_k,
                    val_selector_mse_c,
                    val_selector_mae_c,
                    _,
                    _,
                    _,
                ) = eval_loop_with_history(
                    model, gate, lam_kp_best,
                    penalty_names, penalty_fns,
                    val_loader_summary, cluster_id_c, K, moe_cfg, device,
                    select_ranks=select_ranks,
                    collect_plot=False, channel_count=C,
                    mse_weight=mse_weight,
                    gate_entropy_weight=gate_entropy_weight,
                    gate_balance_weight=gate_balance_weight,
                    gate_soft_weight=gate_soft_weight,
                    gate_entropy_target_frac=gate_entropy_target_frac,
                    penalty_scale=penalty_scale,
                    dynamic_lambda=dynamic_lambda,
                    lambda_min_kp=lambda_min_kp,
                    mae_objective_weight=mae_eval_weight,
                    mae_objective_kind=mae_objective_kind,
                    mae_objective_beta=mae_objective_beta,
                    pred_residual=pred_residual,
                    pred_residual_selector=candidate_selector_model,
                    pred_residual_scale_c=selector_candidate_scale_c,
                            eval_start=val_eval_start,
                )
                selector_val_summary = {
                    "avg_loss": float(reduce_cluster_metric(val_selector_loss_k, cluster_weight_k).item()),
                    "avg_mse": float(reduce_cluster_metric(val_selector_mse_k, cluster_weight_k).item()),
                    "avg_mae": float(reduce_cluster_metric(val_selector_mae_k, cluster_weight_k).item()),
                    "per_cluster_loss": [float(v) for v in val_selector_loss_k.detach().cpu().tolist()],
                    "per_cluster_mse": [float(v) for v in val_selector_mse_k.detach().cpu().tolist()],
                    "per_cluster_mae": [float(v) for v in val_selector_mae_k.detach().cpu().tolist()],
                    "per_channel_mse": [float(v) for v in val_selector_mse_c.detach().cpu().tolist()],
                    "per_channel_mae": [float(v) for v in val_selector_mae_c.detach().cpu().tolist()],
                }
                if pred_residual_selection_summary is None:
                    pred_residual_selection_summary = {
                        "policy": "candidate_selector",
                        "num_residual_channels": int(C),
                    }
                current_selector_ref_mse = float(
                    pred_residual_selection_summary.get(
                        "val_scaled_avg_mse",
                        (val_summary or {}).get("avg_mse", selector_val_summary["avg_mse"]),
                    )
                )
                current_selector_ref_mae = float(
                    pred_residual_selection_summary.get(
                        "val_scaled_avg_mae",
                        (val_summary or {}).get("avg_mae", selector_val_summary["avg_mae"]),
                    )
                )
                selector_adoption = _candidate_selector_adoption_decision(
                    current_mse=current_selector_ref_mse,
                    current_mae=current_selector_ref_mae,
                    selector_mse=float(selector_val_summary["avg_mse"]),
                    selector_mae=float(selector_val_summary["avg_mae"]),
                    min_abs_improvement=float(
                        selector_cfg.get(
                            "adopt_min_abs_improvement",
                            pred_residual_cfg.get("selection_min_abs_improvement", 0.0),
                        )
                    ),
                    min_rel_improvement=float(
                        selector_cfg.get(
                            "adopt_min_rel_improvement",
                            pred_residual_cfg.get("selection_min_rel_improvement", 0.0),
                        )
                    ),
                    max_rel_mae_regression=float(selector_cfg.get("adopt_max_rel_mae_regression", 0.0)),
                )
                temporal_guard_min_gain_raw = selector_cfg.get("adopt_temporal_block_min_gain_pct", None)
                if temporal_guard_min_gain_raw is not None:
                    temporal_metrics = (pred_residual_selector_summary or {}).get("temporal_block_metrics") or {}
                    temporal_source = str(selector_cfg.get("adopt_temporal_block_source", "holdout")).lower()
                    temporal_blocks = temporal_metrics.get(temporal_source, [])
                    min_positive_raw = selector_cfg.get(
                        "adopt_temporal_block_min_positive_blocks",
                        selector_cfg.get("adopt_temporal_block_min_positive", None),
                    )
                    min_positive = None if min_positive_raw is None else int(min_positive_raw)
                    temporal_guard = _candidate_selector_temporal_block_adoption_guard(
                        blocks=temporal_blocks,
                        min_gain_pct=float(temporal_guard_min_gain_raw),
                        min_positive_blocks=min_positive,
                    )
                    temporal_guard["source"] = temporal_source
                    selector_adoption["temporal_block_guard"] = temporal_guard
                    if not bool(temporal_guard.get("passed", False)):
                        selector_adoption["adopt"] = False
                        selector_adoption["reason"] = "temporal_block_guard_failed"
                pred_residual_selector_summary["adoption"] = selector_adoption
                pred_residual_selection_summary["candidate_selector"] = pred_residual_selector_summary
                pred_residual_selection_summary["val_selector_avg_mse"] = float(selector_val_summary["avg_mse"])
                pred_residual_selection_summary["val_selector_avg_mae"] = float(selector_val_summary["avg_mae"])
                pred_residual_selection_summary["candidate_selector_adopted"] = bool(selector_adoption["adopt"])
                if bool(selector_adoption["adopt"]):
                    pred_residual_selector_model = candidate_selector_model
                    pred_residual_channel_scale_c = selector_candidate_scale_c
                    val_mse_c_base = val_selector_mse_c
                    val_mae_c_base = val_selector_mae_c
                    val_summary = selector_val_summary
                    pred_residual_selection_summary["selected_residual_evaluator"] = "candidate_selector"
                    pred_residual_selection_summary["val_scaled_avg_mse"] = float(selector_val_summary["avg_mse"])
                    pred_residual_selection_summary["val_scaled_avg_mae"] = float(selector_val_summary["avg_mae"])
                    pred_residual_selection_summary["val_scaled_mse_per_channel"] = [
                        float(v) for v in val_selector_mse_c.detach().cpu().tolist()
                    ]
                    pred_residual_selection_summary["val_scaled_mae_per_channel"] = [
                        float(v) for v in val_selector_mae_c.detach().cpu().tolist()
                    ]
                else:
                    pred_residual_selector_model = None
                    pred_residual_selection_summary.setdefault("selected_residual_evaluator", "channel_scale")
                print(
                    "Prediction residual candidate selector: "
                    f"source={selector_source_split}, "
                    f"val_MSE={selector_val_summary['avg_mse']:.6f}, "
                    f"adopted={bool(selector_adoption['adopt'])}, "
                    f"holdout_gain={((pred_residual_selector_summary or {}).get('holdout') or {}).get('selected_gain_pct_vs_base')}"
                )
        gate_penalty_hit_cfg = moe_cfg.get("gate_penalty_hit", {}) or {}
        gate_penalty_hit_enable = bool(gate_penalty_hit_cfg.get("enable", True))
        if gate_penalty_hit_enable and pred_residual is not None and moe_enable and P > 0:
            val_penalty_hit = evaluate_gate_penalty_hit_metrics(
                model=model,
                gate=gate,
                pred_residual=pred_residual,
                loader=val_loader_summary,
                cluster_id_c=cluster_id_c,
                K=K,
                moe_cfg=moe_cfg,
                device=device,
                penalty_names=penalty_names,
                penalty_fns=penalty_fns,
                penalty_scale=penalty_scale,
                select_ranks=select_ranks,
                gate_soft_weight=gate_soft_weight,
                label_min_improvement=float(
                    pred_residual_cfg.get(
                        "gate_hit_label_min_improvement",
                        pred_residual_cfg.get("selection_min_abs_improvement", 0.0),
                    )
                ),
                history_anchor_cfg=history_anchor_cfg,
                observed_history_tc=data_window_tc,
                input_len=L,
                eval_start=val_eval_start,
                model_train_stat_adapter_pc=model_train_stat_adapter_pc,
                model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
                train_stat_anchor_pc=train_stat_anchor_pc,
                train_residual_anchor_phc=train_residual_anchor_phc,
                learnable_output_anchor=learnable_output_anchor,
                allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                gate_feature_mode=gate_feature_mode,
            )
            moe_gate_penalty_hit_summary = {"val": val_penalty_hit, "test": None}
            if val_penalty_hit is not None:
                print(
                    "Gate penalty hit(val): "
                    f"top1={val_penalty_hit['top1_hit_rate_all']:.3f}, "
                    f"positive_top1={val_penalty_hit['top1_hit_rate_on_positive_oracle']:.3f}, "
                    f"selected_gain={val_penalty_hit['selected_top1_gain_pct_vs_base']:.3f}%"
                )

    if (
        bool(calendar_residual_cfg.get("enable", False))
        and str(calendar_residual_cfg.get("fit_target", "base_path")).lower()
        in {"final", "final_eval", "final_eval_path", "eval_path"}
    ):
        calendar_fit_loader = DataLoader(
            dtr,
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=False,
            num_workers=0,
            pin_memory=pin_mem,
        )
        calendar_residual_coef_cf, calendar_fit_summary = fit_calendar_residual_correction_from_eval_path(
            model=model,
            gate=gate,
            lambda_kp=lam_kp_best,
            penalty_names=penalty_names,
            penalty_fns=penalty_fns,
            loader=calendar_fit_loader,
            cluster_id_c=cluster_id_c,
            K=K,
            moe_cfg=moe_cfg,
            device=device,
            calendar_feature_tf=calendar_feature_tf,
            input_len=L,
            cfg=calendar_residual_cfg,
            channel_count=C,
            select_ranks=select_ranks,
            mse_weight=mse_weight,
            gate_entropy_weight=gate_entropy_weight,
            gate_balance_weight=gate_balance_weight,
            gate_soft_weight=gate_soft_weight,
            gate_entropy_target_frac=gate_entropy_target_frac,
            penalty_scale=penalty_scale,
            dynamic_lambda=dynamic_lambda,
            lambda_min_kp=lambda_min_kp,
            mae_objective_weight=mae_eval_weight,
            mae_objective_kind=mae_objective_kind,
            mae_objective_beta=mae_objective_beta,
            pred_residual=pred_residual,
            pred_residual_selector=pred_residual_selector_model,
            pred_residual_scale_c=pred_residual_channel_scale_c,
            eval_start=0,
            history_anchor_cfg=history_anchor_cfg,
            observed_history_tc=data_window_tc,
            model_train_stat_adapter_pc=model_train_stat_adapter_pc,
            model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
            train_stat_anchor_pc=train_stat_anchor_pc,
            train_residual_anchor_phc=train_residual_anchor_phc,
        )
        calendar_residual_summary.update(calendar_fit_summary)
        calendar_residual_summary["feature_names"] = list(calendar_feature_names)
        calendar_residual_summary["train_only"] = True
        calendar_residual_summary.pop("pending_final_eval_path_fit", None)
        if calendar_residual_coef_cf is not None:
            print(
                "Calendar residual fitted: "
                f"target=final_eval_path, features={len(calendar_feature_names)}, "
                f"fit_windows={calendar_residual_summary.get('fit_windows')}, "
                f"coef_mean_abs={float(calendar_residual_summary.get('coef_mean_abs', 0.0)):.6f}"
            )

    position_daily_fit_cfg = (
        position_daily_residual_expert_cfg
        if position_daily_residual_expert_enable
        else position_daily_residual_cfg
    )
    if position_daily_residual_expert_enable or bool(
        position_daily_residual_cfg.get("enable", False)
    ):
        source_split = str(
            position_daily_fit_cfg.get("source_split", "val")
        ).lower()
        if source_split not in {"val", "validation"}:
            raise ValueError(
                "position daily residual fit source_split must be val."
            )
        tail_fraction = float(
            position_daily_fit_cfg.get("tail_fraction", 0.25)
        )
        if not 0.0 < tail_fraction <= 1.0:
            raise ValueError(
                "position daily residual fit tail_fraction must be in (0,1]."
            )
        fit_count = max(1, int(round(len(dva) * tail_fraction)))
        fit_start = max(0, len(dva) - fit_count)
        fit_loader = DataLoader(
            Subset(dva, range(fit_start, len(dva))),
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=False,
            num_workers=0,
            pin_memory=pin_mem,
        )
        fit_collector: Dict[str, object] = {
            "limit": int(fit_count),
            "count": 0,
            "parts": {},
        }
        eval_loop_with_history(
            model, gate, lam_kp_best,
            penalty_names, penalty_fns,
            fit_loader, cluster_id_c, K, moe_cfg, device,
            select_ranks=select_ranks,
            collect_plot=False,
            channel_count=C,
            mse_weight=mse_weight,
            gate_entropy_weight=gate_entropy_weight,
            gate_balance_weight=gate_balance_weight,
            gate_soft_weight=gate_soft_weight,
            gate_entropy_target_frac=gate_entropy_target_frac,
            penalty_scale=penalty_scale,
            dynamic_lambda=dynamic_lambda,
            lambda_min_kp=lambda_min_kp,
            mae_objective_weight=mae_eval_weight,
            mae_objective_kind=mae_objective_kind,
            mae_objective_beta=mae_objective_beta,
            pred_residual=pred_residual,
            pred_residual_selector=pred_residual_selector_model,
            pred_residual_scale_c=pred_residual_channel_scale_c,
            eval_start=val_eval_start,
            diagnostic_collector=fit_collector,
        )
        fit_parts = fit_collector.get("parts", {}) or {}
        fitted_position_daily_coef_cfh, fit_summary = (
            fit_position_daily_residual_ridge_from_prediction_parts(
                idx_parts=list(fit_parts.get("idx", [])),
                y_true_parts=list(fit_parts.get("y_true", [])),
                y_pred_parts=list(fit_parts.get("y_final", [])),
                input_len=L,
                cfg=position_daily_fit_cfg,
            )
        )
        fit_metadata = {
            "source_window_range": [int(fit_start), int(len(dva))],
            "tail_fraction": float(tail_fraction),
            "test_labels_used_for_fit": False,
        }
        if position_daily_residual_expert_enable:
            if pred_residual is None:
                raise RuntimeError(
                    "position_daily_residual_expert requires pred_residual."
                )
            pred_residual.set_position_daily_residual_expert(
                fitted_position_daily_coef_cfh,
                period=int(position_daily_fit_cfg.get("daily_period", 96)),
                harmonics=int(position_daily_fit_cfg.get("daily_harmonics", 4)),
            )
            position_daily_residual_expert_summary.update(fit_summary)
            position_daily_residual_expert_summary.update(fit_metadata)
            position_daily_residual_expert_summary["role"] = (
                "gated_pred_residual_expert"
                if anchor_ridge_gate_enable
                else "always_on_pred_residual_expert"
            )
            if anchor_ridge_gate_enable:
                anchor_ridge_gate_summary.update(
                    fit_anchor_ridge_gate_from_prediction_parts(
                        pred_residual=pred_residual,
                        parts=fit_parts,
                        position_daily_coef_cfh=fitted_position_daily_coef_cfh,
                        input_len=L,
                        position_cfg=position_daily_fit_cfg,
                        gate_cfg=anchor_ridge_gate_cfg,
                    )
                )
                print(
                    "Anchor/ridge input gate fitted: "
                    f"adopted={bool(anchor_ridge_gate_summary.get('adopted', False))}, "
                    f"holdout_gain={float(anchor_ridge_gate_summary.get('holdout_mse_improvement_pct', 0.0)):.4f}%, "
                    f"anchor_mean={float(anchor_ridge_gate_summary.get('anchor_weight_mean', 1.0)):.4f}, "
                    f"ridge_mean={float(anchor_ridge_gate_summary.get('ridge_weight_mean', 1.0)):.4f}"
                )
            active_position_daily_summary = position_daily_residual_expert_summary
        else:
            position_daily_residual_coef_cfh = fitted_position_daily_coef_cfh
            position_daily_residual_summary.update(fit_summary)
            position_daily_residual_summary.update(fit_metadata)
            active_position_daily_summary = position_daily_residual_summary
        print(
            "Position daily residual fitted: "
            f"role={'pred_residual_expert' if position_daily_residual_expert_enable else 'eval_postprocess'}, "
            f"source=val[{fit_start}:{len(dva)}], "
            f"ridge={float(active_position_daily_summary.get('ridge', 0.0)):.6f}, "
            f"coef_mean_abs={float(active_position_daily_summary.get('coef_mean_abs', 0.0)):.6f}"
        )

    lam_kp_test = lam_kp_best
    test_loss_k = test_mse_k = test_mae_k = None
    mse_c = mae_c = None
    plot_cache = {}
    best_sample = {}
    worst_sample = {}
    diagnostics_cfg = cfg.get("diagnostics", {}) or {}
    prediction_diag = bool(diagnostics_cfg.get("save_prediction_intermediates", False))
    prediction_diag_collector = None
    test_base_metric_collector: Optional[Dict[str, object]] = (
        {} if not skip_test else None
    )
    if prediction_diag:
        prediction_sample_count = int(diagnostics_cfg.get("prediction_sample_count", 32))
        prediction_sample_strategy = str(diagnostics_cfg.get("prediction_sample_strategy", "first"))
        prediction_sample_seed = int(diagnostics_cfg.get("prediction_sample_seed", 0))
        prediction_sample_indices = select_prediction_sample_indices(
            total=len(dte),
            sample_count=prediction_sample_count,
            strategy=prediction_sample_strategy,
            seed=prediction_sample_seed,
        )
        prediction_diag_collector = {
            "limit": len(prediction_sample_indices),
            "count": 0,
            "parts": {},
            "indices": torch.as_tensor(prediction_sample_indices, dtype=torch.long),
            "strategy": prediction_sample_strategy,
            "seed": prediction_sample_seed,
            "relative_indices": prediction_sample_indices,
        }
    if not skip_test:
        test_loss_k, test_mse_k, test_mae_k, mse_c, mae_c, plot_cache, best_sample, worst_sample = eval_loop_with_history(
            model, gate, lam_kp_test,
            penalty_names, penalty_fns,
            dl_te, cluster_id_c, K, moe_cfg, device,
            select_ranks=select_ranks,
            collect_plot=plot_enable, plot_idx=plot_idx, channel_count=C,
            mse_weight=mse_weight,
            gate_entropy_weight=gate_entropy_weight,
            gate_balance_weight=gate_balance_weight,
            gate_soft_weight=gate_soft_weight,
            gate_entropy_target_frac=gate_entropy_target_frac,
            penalty_scale=penalty_scale,
            dynamic_lambda=dynamic_lambda,
            lambda_min_kp=lambda_min_kp,
            mae_objective_weight=mae_eval_weight,
            mae_objective_kind=mae_objective_kind,
            mae_objective_beta=mae_objective_beta,
            pred_residual=pred_residual,
            pred_residual_selector=pred_residual_selector_model,
            pred_residual_scale_c=pred_residual_channel_scale_c,
            eval_start=test_eval_start,
            diagnostic_collector=prediction_diag_collector,
            base_metric_collector=test_base_metric_collector,
        )
        gate_penalty_hit_cfg = moe_cfg.get("gate_penalty_hit", {}) or {}
        gate_penalty_hit_enable = bool(gate_penalty_hit_cfg.get("enable", True))
        if gate_penalty_hit_enable and pred_residual is not None and moe_enable and P > 0:
            test_penalty_hit = evaluate_gate_penalty_hit_metrics(
                model=model,
                gate=gate,
                pred_residual=pred_residual,
                loader=dl_te,
                cluster_id_c=cluster_id_c,
                K=K,
                moe_cfg=moe_cfg,
                device=device,
                penalty_names=penalty_names,
                penalty_fns=penalty_fns,
                penalty_scale=penalty_scale,
                select_ranks=select_ranks,
                gate_soft_weight=gate_soft_weight,
                label_min_improvement=float(
                    pred_residual_cfg.get(
                        "gate_hit_label_min_improvement",
                        pred_residual_cfg.get("selection_min_abs_improvement", 0.0),
                    )
                ),
                history_anchor_cfg=history_anchor_cfg,
                observed_history_tc=data_window_tc,
                input_len=L,
                eval_start=test_eval_start,
                model_train_stat_adapter_pc=model_train_stat_adapter_pc,
                model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
                train_stat_anchor_pc=train_stat_anchor_pc,
                train_residual_anchor_phc=train_residual_anchor_phc,
                learnable_output_anchor=learnable_output_anchor,
                allowed_mask_kp=cluster_penalty_allowed_mask_kp,
                gate_feature_mode=gate_feature_mode,
            )
            if moe_gate_penalty_hit_summary is None:
                moe_gate_penalty_hit_summary = {"val": None, "test": test_penalty_hit}
            else:
                moe_gate_penalty_hit_summary["test"] = test_penalty_hit
            if test_penalty_hit is not None:
                print(
                    "Gate penalty hit(test): "
                    f"top1={test_penalty_hit['top1_hit_rate_all']:.3f}, "
                    f"positive_top1={test_penalty_hit['top1_hit_rate_on_positive_oracle']:.3f}, "
                    f"selected_gain={test_penalty_hit['selected_top1_gain_pct_vs_base']:.3f}%"
                )
    explain_cfg = moe_cfg.get("explainability", {}) or {}
    explain_enable = bool(explain_cfg.get("enable", False))
    if explain_enable and pred_residual is not None and moe_enable and P > 0:
        max_batches = int(explain_cfg.get("max_batches", 0))
        requested_splits = [str(x).lower() for x in explain_cfg.get("splits", ["train", "val", "test"])]
        split_loaders: Dict[str, DataLoader] = {}
        split_eval_starts: Dict[str, int] = {}
        train_subsplit_ranges: Dict[str, Tuple[int, int]] = {}
        if "train" in requested_splits and len(dtr) > 0:
            split_loaders["train"] = DataLoader(
                dtr,
                batch_size=int(cfg["train"]["batch_size"]),
                shuffle=False,
                num_workers=0,
                pin_memory=pin_mem,
            )
            split_eval_starts["train"] = 0
        train_subsplit_names = {"train_fit", "train_holdout"}
        if any(name in requested_splits for name in train_subsplit_names) and len(dtr) > 0:
            holdout_fraction = float(
                explain_cfg.get(
                    "train_holdout_fraction",
                    explain_cfg.get("holdout_fraction", 0.30),
                )
            )
            train_subsplit_ranges = _explainability_train_subsplit_ranges(
                num_windows=len(dtr),
                holdout_fraction=holdout_fraction,
            )
            for split_name in ("train_fit", "train_holdout"):
                if split_name not in requested_splits or split_name not in train_subsplit_ranges:
                    continue
                start_i, end_i = train_subsplit_ranges[split_name]
                if int(end_i) <= int(start_i):
                    continue
                split_loaders[split_name] = DataLoader(
                    Subset(dtr, range(int(start_i), int(end_i))),
                    batch_size=int(cfg["train"]["batch_size"]),
                    shuffle=False,
                    num_workers=0,
                    pin_memory=pin_mem,
                )
                split_eval_starts[split_name] = 0
        if "val" in requested_splits and len(dva) > 0:
            split_loaders["val"] = DataLoader(
                dva,
                batch_size=int(cfg["train"]["batch_size"]),
                shuffle=False,
                num_workers=0,
                pin_memory=pin_mem,
            )
            split_eval_starts["val"] = int(val_eval_start)
        if "test" in requested_splits and (not skip_test) and len(dte) > 0:
            split_loaders["test"] = dl_te
            split_eval_starts["test"] = int(test_eval_start)

        prior_for_explain = cluster_penalty_prior_prob_kp if cluster_penalty_prior_prob_kp is not None else gate_prior_prob_kp
        allowed_for_explain = cluster_penalty_allowed_mask_kp
        split_payloads = {}
        for split_name, split_loader in split_loaders.items():
            payload = evaluate_penalty_explainability(
                model=model,
                gate=gate,
                pred_residual=pred_residual,
                loader=split_loader,
                cluster_id_c=cluster_id_c,
                K=K,
                moe_cfg=moe_cfg,
                device=device,
                penalty_names=penalty_names,
                penalty_fns=penalty_fns,
                penalty_scale=penalty_scale,
                select_ranks=select_ranks,
                gate_soft_weight=gate_soft_weight,
                split_name=split_name,
                penalty_portrait_kp=penalty_portrait_kp,
                prior_prob_kp=prior_for_explain,
                allowed_mask_kp=allowed_for_explain,
                max_batches=max_batches,
                history_anchor_cfg=history_anchor_cfg,
                observed_history_tc=data_window_tc,
                input_len=L,
                eval_start=int(split_eval_starts.get(split_name, 0)),
                model_train_stat_adapter_pc=model_train_stat_adapter_pc,
                model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
                train_stat_anchor_pc=train_stat_anchor_pc,
                train_residual_anchor_phc=train_residual_anchor_phc,
                learnable_output_anchor=learnable_output_anchor,
                gate_feature_mode=gate_feature_mode,
            )
            if payload is not None:
                split_payloads[split_name] = payload
                print(
                    f"Penalty explainability({split_name}): "
                    f"gain={payload['final_gain_pct_vs_base']:.3f}%, "
                    f"selected_events={payload['selected_penalty_events']}, "
                    f"oracle_positive_events={payload['oracle_positive_events']}"
                )
        gradient_isolation_summary = None
        gradient_isolation_cfg = explain_cfg.get("gradient_isolation", {}) or {}
        if not isinstance(gradient_isolation_cfg, dict):
            gradient_isolation_cfg = {"enable": bool(gradient_isolation_cfg)}
        if bool(gradient_isolation_cfg.get("enable", False)):
            gradient_split = str(
                gradient_isolation_cfg.get("split", "train_holdout")
            ).lower()
            allow_test_search = bool(
                gradient_isolation_cfg.get("allow_test_search", False)
            )
            if gradient_split == "test" and not allow_test_search:
                raise ValueError(
                    "adapter gradient-isolation proof must use train/train_holdout/val; "
                    "set explainability.gradient_isolation.allow_test_search=true only "
                    "for an explicitly authorized test-selected proof"
                )
            if gradient_split not in split_loaders:
                for fallback_split in ("train_holdout", "train_fit", "train", "val"):
                    if fallback_split in split_loaders:
                        gradient_split = fallback_split
                        break
            if gradient_split in split_loaders:
                gradient_isolation_summary = evaluate_adapter_gradient_isolation(
                    model=model,
                    pred_residual=pred_residual,
                    loader=split_loaders[gradient_split],
                    cluster_id_c=cluster_id_c,
                    K=K,
                    moe_cfg=moe_cfg,
                    device=device,
                    penalty_names=penalty_names,
                    split_name=gradient_split,
                    max_batches=int(gradient_isolation_cfg.get("max_batches", 4)),
                    off_diagonal_tolerance=float(
                        gradient_isolation_cfg.get(
                            "off_diagonal_tolerance",
                            1.0e-12,
                        )
                    ),
                    min_diagonal_norm=float(
                        gradient_isolation_cfg.get("min_diagonal_norm", 1.0e-12)
                    ),
                    history_anchor_cfg=history_anchor_cfg,
                    observed_history_tc=data_window_tc,
                    input_len=L,
                    eval_start=int(split_eval_starts.get(gradient_split, 0)),
                    model_train_stat_adapter_pc=model_train_stat_adapter_pc,
                    model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
                    train_stat_anchor_pc=train_stat_anchor_pc,
                    train_residual_anchor_phc=train_residual_anchor_phc,
                    learnable_output_anchor=learnable_output_anchor,
                )
                if gradient_isolation_summary is not None:
                    print(
                        "Adapter gradient isolation"
                        f"({gradient_split}): "
                        f"diag_min={gradient_isolation_summary['diagonal_min']:.3e}, "
                        f"offdiag_max={gradient_isolation_summary['off_diagonal_max']:.3e}, "
                        f"passed={gradient_isolation_summary['passed']}"
                    )
        route_probe_cfg = explain_cfg.get("route_learnability_probe", {}) or {}
        if not isinstance(route_probe_cfg, dict):
            route_probe_cfg = {"enable": bool(route_probe_cfg)}
        if bool(route_probe_cfg.get("enable", False)):
            train_split_name = str(route_probe_cfg.get("train_split", "train_fit")).lower()
            if train_split_name not in split_loaders and "train" in split_loaders:
                train_split_name = "train"
            eval_split_names = [
                str(name).lower()
                for name in (route_probe_cfg.get("eval_splits", ["train_holdout", "val"]) or [])
            ]
            allow_test_probe = bool(route_probe_cfg.get("allow_test", False))
            probe_split_names = []
            for name in [train_split_name] + eval_split_names:
                if name == "test" and not allow_test_probe:
                    continue
                if name in split_loaders and name not in probe_split_names:
                    probe_split_names.append(name)
            route_tensors_by_split: Dict[str, Dict[str, object]] = {}
            route_feature_mode = str(route_probe_cfg.get("feature_mode", "base"))
            route_max_batches = int(route_probe_cfg.get("max_batches", max_batches))
            for split_name in probe_split_names:
                tensors = _collect_penalty_route_learnability_tensors(
                    model=model,
                    gate=gate,
                    pred_residual=pred_residual,
                    loader=split_loaders[split_name],
                    cluster_id_c=cluster_id_c,
                    K=K,
                    moe_cfg=moe_cfg,
                    device=device,
                    penalty_names=penalty_names,
                    penalty_fns=penalty_fns,
                    penalty_scale=penalty_scale,
                    select_ranks=select_ranks,
                    gate_soft_weight=gate_soft_weight,
                    split_name=split_name,
                    feature_mode=route_feature_mode,
                    allowed_mask_kp=allowed_for_explain,
                    max_batches=route_max_batches,
                    history_anchor_cfg=history_anchor_cfg,
                    observed_history_tc=data_window_tc,
                    input_len=L,
                    eval_start=int(split_eval_starts.get(split_name, 0)),
                    model_train_stat_adapter_pc=model_train_stat_adapter_pc,
                    model_train_stat_adapter_cfg=model_train_stat_adapter_cfg,
                    train_stat_anchor_pc=train_stat_anchor_pc,
                    train_residual_anchor_phc=train_residual_anchor_phc,
                    learnable_output_anchor=learnable_output_anchor,
                    gate_feature_mode=gate_feature_mode,
                )
                if tensors is not None:
                    route_tensors_by_split[split_name] = tensors
            artifact_paths: Dict[str, object] = {}
            if train_split_name in route_tensors_by_split:
                train_route_tensors = route_tensors_by_split[train_split_name]
                eval_route_tensors = {
                    name: tensors
                    for name, tensors in route_tensors_by_split.items()
                    if name != train_split_name
                }
                head_cfg = route_probe_cfg.get("head", route_probe_cfg) or {}
                if not isinstance(head_cfg, dict):
                    head_cfg = {}
                penalty_route_learnability_summary, route_head_artifact = _fit_penalty_route_learnability_head_from_tensors(
                    train_tensors=train_route_tensors,  # type: ignore[arg-type]
                    eval_tensors_by_split=eval_route_tensors,  # type: ignore[arg-type]
                    label_names=list(train_route_tensors["label_names"]),  # type: ignore[index]
                    feature_names=list(train_route_tensors["feature_names"]),  # type: ignore[index]
                    cfg=head_cfg,
                    device=device,
                )
                penalty_route_learnability_summary["train_split"] = train_split_name
                penalty_route_learnability_summary["eval_splits"] = list(eval_route_tensors.keys())
                penalty_route_learnability_summary["feature_mode"] = route_feature_mode
                penalty_route_learnability_summary["max_batches"] = int(route_max_batches)
                head_path = os.path.join(out_dir, "penalty_route_learnability_head.pt")
                torch.save(route_head_artifact, head_path)
                artifact_paths["head"] = head_path
                label_names = list(train_route_tensors["label_names"])  # type: ignore[index]
                for split_name, tensors in route_tensors_by_split.items():
                    tensor_path = os.path.join(out_dir, f"penalty_route_learnability_{split_name}.pt")
                    torch.save(tensors, tensor_path)
                    artifact_paths[f"{split_name}_tensors"] = tensor_path
                    labels_cpu = tensors["labels"].detach().cpu().to(dtype=torch.long)  # type: ignore[index]
                    current_cpu = tensors["current_pred"].detach().cpu().to(dtype=torch.long)  # type: ignore[index]
                    query_cpu = tensors["query_start_abs"].detach().cpu().to(dtype=torch.long)  # type: ignore[index]
                    gain_cpu = tensors["oracle_gain_mse"].detach().cpu().to(dtype=torch.float32)  # type: ignore[index]
                    label_df = pd.DataFrame(
                        {
                            "split": split_name,
                            "row": list(range(int(labels_cpu.numel()))),
                            "query_start_abs": [int(v) for v in query_cpu.tolist()],
                            "oracle_class": [int(v) for v in labels_cpu.tolist()],
                            "oracle_label": [
                                label_names[int(v)] if 0 <= int(v) < len(label_names) else ""
                                for v in labels_cpu.tolist()
                            ],
                            "current_class": [int(v) for v in current_cpu.tolist()],
                            "current_label": [
                                label_names[int(v)] if 0 <= int(v) < len(label_names) else ""
                                for v in current_cpu.tolist()
                            ],
                            "oracle_gain_mse": [float(v) for v in gain_cpu.tolist()],
                        }
                    )
                    csv_path = os.path.join(out_dir, f"penalty_route_oracle_labels_{split_name}.csv")
                    label_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                    artifact_paths[f"{split_name}_labels_csv"] = csv_path
                summary_path = os.path.join(out_dir, "penalty_route_learnability.json")
                artifact_paths["summary"] = summary_path
                penalty_route_learnability_summary["artifact_paths"] = artifact_paths
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(penalty_route_learnability_summary, f, ensure_ascii=False, indent=2)
                val_metrics = (penalty_route_learnability_summary.get("splits", {}) or {}).get("val")
                if isinstance(val_metrics, dict):
                    print(
                        "Penalty route learnability(val): "
                        f"head_acc={float(val_metrics.get('accuracy_all', 0.0)):.3f}, "
                        f"current_acc={float(val_metrics.get('current_accuracy_all', 0.0)):.3f}, "
                        f"majority_acc={float(val_metrics.get('majority_accuracy_all', 0.0)):.3f}"
                    )
            else:
                penalty_route_learnability_summary = {
                    "enable": True,
                    "skipped": True,
                    "reason": f"train_split {train_split_name!r} was not available",
                    "available_splits": list(split_loaders.keys()),
                }
        penalty_explainability_summary = {
            "enable": True,
            "max_batches": int(max_batches),
            "train_subsplits": {
                name: {"start": int(start_i), "end": int(end_i)}
                for name, (start_i, end_i) in train_subsplit_ranges.items()
            },
            "splits": split_payloads,
            "adapter_gradient_isolation": gradient_isolation_summary,
            "train_only_prior": {
                "source": "train_split_penalty_portrait" if penalty_portrait_kp is not None else None,
                "penalty_names": list(penalty_names),
                "diagnostic_score": (
                    penalty_portrait_kp.detach().cpu().tolist()
                    if penalty_portrait_kp is not None
                    else None
                ),
                "prior_prob": (
                    prior_for_explain.detach().cpu().tolist()
                    if prior_for_explain is not None
                    else None
                ),
                "allowed_mask": (
                    allowed_for_explain.detach().cpu().tolist()
                    if allowed_for_explain is not None
                    else None
                ),
            },
        }
        penalty_explainability_summary["artifact_paths"] = save_penalty_explainability_artifacts(
            out_dir,
            penalty_explainability_summary,
        )
    df = None
    avg_mae = None
    avg_mse = None
    selected_variant = "base"
    selected_criterion = "base"
    selected_selection_policy = "base"
    selected_avg_mae = None
    selected_avg_mse = None
    base_avg_mae = None
    base_avg_mse = None
    test_gain_pct_vs_base = None
    if not skip_test:
        df = pd.DataFrame({
            "channel": channel_names,
            "MAE": mae_c.numpy(),
            "MSE": mse_c.numpy(),
            "cluster_id": cluster_id_c.detach().cpu().numpy(),
        })
        avg_mae = float(df["MAE"].mean())
        avg_mse = float(reduce_cluster_metric(test_mse_k, cluster_weight_k).item())
        selected_avg_mae = avg_mae
        selected_avg_mse = avg_mse
        if test_base_metric_collector:
            base_mse_k = test_base_metric_collector["avg_mse_k"]
            base_mae_c = test_base_metric_collector["mae_c"]
            base_avg_mse = float(
                reduce_cluster_metric(
                    base_mse_k.to(device=cluster_weight_k.device), cluster_weight_k
                ).item()
            )
            base_avg_mae = float(base_mae_c.mean().item())
            test_gain_pct_vs_base = float(
                100.0
                * (base_avg_mse - selected_avg_mse)
                / max(abs(base_avg_mse), 1.0e-12)
            )

    moe_residual_variant = "none"
    if pred_residual is not None and moe_enable and P > 0:
        moe_residual_variant = (
            "moe_residual_patch_router"
            if bool(patch_router_cfg.get("enable", False))
            else "moe_residual"
        )
        selected_variant = moe_residual_variant
        selected_criterion = "checkpoint_validation"
        selected_selection_policy = "trained_prediction_residual"
    if pred_residual_selection_summary is not None:
        moe_residual_variant = "moe_residual_channel"
        if int(pred_residual_selection_summary.get("num_residual_channels", 0) or 0) > 0:
            selected_variant = moe_residual_variant
            selected_criterion = str(pred_residual_selection_summary.get("policy", selected_criterion))
            selected_selection_policy = str(pred_residual_selection_summary.get("policy", selected_selection_policy))

    if skip_test:
        val_mse_print = None if val_summary is None else val_summary.get("avg_mse")
        val_mae_print = None if val_summary is None else val_summary.get("avg_mae")
        if pred_residual_selection_summary is not None:
            val_mse_print = pred_residual_selection_summary.get("val_scaled_avg_mse", val_mse_print)
            val_mae_print = pred_residual_selection_summary.get("val_scaled_avg_mae", val_mae_print)
        if val_mse_print is not None and val_mae_print is not None:
            print(f"\nValidation-only: avg_MAE={val_mae_print:.6f}, avg_MSE={val_mse_print:.6f}")
            final_print(
                "FINAL_VALIDATION "
                f"selected={selected_variant} "
                f"moe_residual={moe_residual_variant} "
                f"val_MAE={val_mae_print:.6f} "
                f"val_MSE={val_mse_print:.6f} "
                "test_MAE=skipped test_MSE=skipped",
                flush=True,
            )
        else:
            print("\nValidation-only: validation metrics unavailable")
            final_print(
                "FINAL_VALIDATION "
                f"selected={selected_variant} "
                f"moe_residual={moe_residual_variant} "
                "val_MAE=nan val_MSE=nan test_MAE=skipped test_MSE=skipped",
                flush=True,
            )
    else:
        print(
            f"\nOverall(selected={selected_variant}, moe_residual={moe_residual_variant}): "
            f"test_MAE={selected_avg_mae:.6f}, test_MSE={selected_avg_mse:.6f}, "
            f"pre_moe_base_MSE={base_avg_mse:.6f}, "
            f"gain_vs_pre_moe={test_gain_pct_vs_base:.4f}%"
        )
        final_print(
            "FINAL_TEST "
            f"selected={selected_variant} "
            f"moe_residual={moe_residual_variant} "
            f"test_MAE={selected_avg_mae:.6f} "
            f"test_MSE={selected_avg_mse:.6f}",
            flush=True,
        )

    if not skip_test and df is not None:
        df.to_csv(os.path.join(out_dir, "test_metrics.csv"), index=False)
        np.save(os.path.join(out_dir, "test_loss_per_cluster.npy"), test_loss_k.detach().cpu().numpy())
        if prediction_diag_collector is not None:
            diag_parts = prediction_diag_collector.get("parts", {}) or {}
            arrays = {
                key: torch.cat(value, dim=0).numpy()
                for key, value in diag_parts.items()
                if isinstance(value, list) and len(value) > 0
            }
            arrays["cluster_id"] = cluster_id_c.detach().cpu().numpy()
            np.savez_compressed(os.path.join(out_dir, "prediction_intermediates.npz"), **arrays)
            diag_meta = {
                "sample_count": int(prediction_diag_collector.get("count", 0)),
                "channel_names": list(channel_names),
                "penalty_names": list(penalty_names),
                "sample_strategy": str(prediction_diag_collector.get("strategy", "first")),
                "sample_seed": int(prediction_diag_collector.get("seed", 0)),
                "relative_indices": [int(v) for v in prediction_diag_collector.get("relative_indices", [])],
            }
            with open(os.path.join(out_dir, "prediction_intermediates_meta.json"), "w", encoding="utf-8") as f:
                json.dump(diag_meta, f, ensure_ascii=False, indent=2)

    if (not skip_test) and plot_enable and (plot_idx is not None):
        plot_dir = os.path.join(out_dir, "plots")
        save_channel_plots(
            out_dir=plot_dir,
            channel_names=channel_names,
            plot_cache=plot_cache,
            best_sample=best_sample,
            worst_sample=worst_sample,
            input_len=L,
            pred_len=H,
            dpi=int(plot_cfg["dpi"])
        )
        print(f"Saved plots to: {plot_dir}")

    total_time = time.perf_counter() - t_all0
    avg_epoch_time = sum(epoch_times) / max(len(epoch_times), 1)
    cpu_rss_mb = _get_rss_mb()
    gpu_alloc_mb = -1.0
    gpu_reserved_mb = -1.0
    if device.type == "cuda":
        gpu_alloc_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
        gpu_reserved_mb = float(torch.cuda.max_memory_reserved()) / (1024.0 * 1024.0)
    out_dir_mb = _dir_size_mb(out_dir)
    cluster_embedding_summary = _save_cluster_embedding_artifacts(model, out_dir)
    stage2_loss_diagnostics_summary = None
    if stage2_loss_audit_enable:
        residual_selection = pred_residual_selection_summary or {}
        moe_residual_diag = moe_residual_summary or {}
        latest_route = (
            stage2_loss_audit_history[-1].get("route", {})
            if len(stage2_loss_audit_history) > 0
            else {}
        )
        val_base_mse = residual_selection.get("val_pred_base_avg_mse", (val_summary or {}).get("avg_mse"))
        val_base_mae = residual_selection.get("val_pred_base_avg_mae", (val_summary or {}).get("avg_mae"))
        val_raw_moe_mse = residual_selection.get("val_residual_avg_mse", (val_summary or {}).get("avg_mse"))
        val_raw_moe_mae = residual_selection.get("val_residual_avg_mae", (val_summary or {}).get("avg_mae"))
        val_scaled_mse = residual_selection.get("val_scaled_avg_mse", (val_summary or {}).get("avg_mse"))
        val_scaled_mae = residual_selection.get("val_scaled_avg_mae", (val_summary or {}).get("avg_mae"))
        stage2_loss_diagnostics_summary = {
            "enabled": True,
            "losses_are_stage2_only": True,
            "do_not_compare_to_stage1_training_loss": True,
            "trainable_parameter_groups": stage2_trainable_parameter_groups,
            "epochs": stage2_loss_audit_history,
            "final_eval": {
                "val_base_mse": val_base_mse,
                "val_base_mae": val_base_mae,
                "val_raw_moe_mse": val_raw_moe_mse,
                "val_raw_moe_mae": val_raw_moe_mae,
                "val_scaled_or_selected_moe_mse": val_scaled_mse,
                "val_scaled_or_selected_moe_mae": val_scaled_mae,
                "residual_delta_rms": moe_residual_diag.get("residual_delta_rms"),
                "residual_base_rms_ratio": moe_residual_diag.get("residual_base_rms_ratio"),
                "route_entropy": latest_route.get("route_entropy"),
                "actual_route_distribution": moe_residual_diag.get(
                    "effective_route_by_penalty",
                    latest_route.get("actual_route_distribution"),
                ),
                "skip_noop_rate": latest_route.get("skip_noop_rate"),
                "skip_prob": latest_route.get("skip_prob"),
            },
        }
    stage2_route_audit_summary = None
    if stage2_route_audit_enable:
        residual_selection = pred_residual_selection_summary or {}
        final_scaled_mse = residual_selection.get("val_scaled_avg_mse", (val_summary or {}).get("avg_mse"))
        final_scaled_mae = residual_selection.get("val_scaled_avg_mae", (val_summary or {}).get("avg_mae"))
        stage2_route_audit_summary = {
            "enabled": True,
            "splits": list(stage2_route_audit_loaders.keys()),
            "train_subsplits": {
                name: {"start": int(start_i), "end": int(end_i)}
                for name, (start_i, end_i) in stage2_route_audit_train_subsplits.items()
            },
            "max_batches": int(stage2_route_audit_cfg.get("max_batches", 0)),
            "frequency_epochs": int(stage2_route_audit_frequency),
            "skip_noop_is_class_zero": True,
            "test_read": False,
            "final_selected_scaled_eval": {
                "val_scaled_or_selected_moe_mse": final_scaled_mse,
                "val_scaled_or_selected_moe_mae": final_scaled_mae,
                "source": "final_moe_residual_selection",
            },
            "epochs": stage2_route_audit_history,
        }
        route_audit_path = os.path.join(out_dir, "stage2_route_audit.json")
        with open(route_audit_path, "w", encoding="utf-8") as f:
            json.dump(stage2_route_audit_summary, f, ensure_ascii=False, indent=2)
        stage2_route_audit_summary["artifact_path"] = route_audit_path

    checkpoint_sha256 = None
    if best_checkpoint_path is not None and os.path.exists(best_checkpoint_path):
        digest = hashlib.sha256()
        with open(best_checkpoint_path, "rb") as checkpoint_file:
            for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                digest.update(chunk)
        checkpoint_sha256 = digest.hexdigest()

    summary = {
        "config_path": args.config,
        "out_dir": out_dir,
        "checkpoint": {
            "path": best_checkpoint_path,
            "sha256": checkpoint_sha256,
            "pred_residual_contract": (
                best_checkpoint_meta.get("pred_residual_contract")
                if isinstance(best_checkpoint_meta, dict)
                else pred_residual_contract
            ),
        },
        "penalty_names": list(penalty_names),
        "best_epoch": [int(v) for v in best_epoch.detach().cpu().tolist()],
        "windowing": {
            "past_context": bool(past_context),
            "train_start": 0,
            "val_eval_start": int(val_eval_start),
            "test_eval_start": int(test_eval_start),
            "val_label_start": int(t_train),
            "test_label_start": int(t_val),
            "num_train_windows": int(len(dtr)),
            "num_optimization_windows": int(len(optimization_dataset)),
            "num_val_windows": int(len(dva)),
            "num_test_windows": int(len(dte)),
            "normalize_train_only": bool(norm_cfg.get("train_only", False)),
            "data_max_rows": int(max_rows),
        },
        "mae_objective": {
            "enable": bool(mae_objective_enable),
            "kind": str(mae_objective_kind),
            "weight": float(mae_objective_weight_final),
            "warmup_epochs": int(mae_objective_warmup_epochs),
            "beta": float(mae_objective_beta),
            "per_cluster": mae_objective_per_cluster_summary,
        },
        "cluster_embedding": cluster_embedding_summary,
        "training_stability": {
            "shuffle_seed": None if shuffle_seed is None else int(shuffle_seed),
            "freeze_backbone": bool(freeze_backbone),
            "frozen_backbone_params": int(frozen_backbone_params),
            "frozen_backbone_eval_mode": bool(frozen_backbone_eval_mode),
            "backbone_lr": None if backbone_lr is None else float(backbone_lr),
            "loss_normalization": dict(loss_normalization_cfg),
            "lr_warmup_epochs": int(lr_warmup_epochs),
            "lr_warmup_start_factor": float(lr_warmup_start_factor),
            "swa": dict(swa_summary),
            "optimization_window": {
                "enable": bool(optimization_window_range is not None),
                "source": str(optimization_source),
                "source_window_range": (
                    [
                        int(optimization_window_range[0]),
                        int(optimization_window_range[1]),
                    ]
                    if optimization_window_range is not None
                    else None
                ),
                "train_window_range": (
                    [
                        int(optimization_window_range[0]),
                        int(optimization_window_range[1]),
                    ]
                    if optimization_source == "train"
                    and optimization_window_range is not None
                    else None
                ),
                "num_windows": int(len(optimization_dataset)),
                "index_offset": int(optimization_index_offset),
                "epoch_eval_source": "validation_window",
                "epoch_eval_window_range": (
                    [
                        int(epoch_eval_window_range[0]),
                        int(epoch_eval_window_range[1]),
                    ]
                    if epoch_eval_window_range is not None
                    else None
                ),
            },
            "overfit_diagnostic": {
                "enable": bool(overfit_diagnostic_range is not None),
                "train_window_range": (
                    [int(overfit_diagnostic_range[0]), int(overfit_diagnostic_range[1])]
                    if overfit_diagnostic_range is not None
                    else None
                ),
                "num_windows": int(len(optimization_dataset)),
                "epoch_eval_source": (
                    "train_subset" if overfit_diagnostic_range is not None else "validation"
                ),
                "official_validation_evaluated_each_epoch": bool(
                    overfit_diagnostic_range is None
                ),
                "metric_epochs": [int(v) for v in overfit_diagnostic_metric_epochs],
                "metrics": overfit_diagnostic_history,
            },
        },
        "stage2_trainable_parameter_groups": stage2_trainable_parameter_groups,
        "shared_moe": {
            "shared_across_clusters": bool(shared_moe_across_clusters),
            "best_epoch": int(shared_moe_best_epoch) if shared_moe_across_clusters else None,
            "patch_router_replaces_cluster_gate": bool(patch_router_replaces_cluster_gate),
            "frozen_cluster_gate_params": int(frozen_cluster_gate_for_patch_router),
        },
        "eval": {
            "skip_test": bool(skip_test),
        },
        "calendar_residual": calendar_residual_summary,
        "position_daily_residual_ridge": position_daily_residual_summary,
        "position_daily_residual_expert": position_daily_residual_expert_summary,
        "anchor_ridge_gate": anchor_ridge_gate_summary,
        "moe_residual": moe_residual_summary,
        "moe_residual_phase_candidate": phase_residual_candidate_summary,
        "moe_residual_confidence_gate": pred_residual_confidence_summary,
        "moe_residual_selection": pred_residual_selection_summary,
        "moe_residual_candidate_selector": pred_residual_selector_summary,
        "cluster_penalty_prior": {
            "enable": bool(cluster_penalty_prior_enable),
            "apply_stage": str(cluster_penalty_prior_apply_stage),
            "late_eval_applied": bool(cluster_penalty_prior_late_applied),
            "apply_to_pred_residual": bool(cluster_penalty_prior_cfg.get("apply_to_pred_residual", False)),
            "prior": (
                cluster_penalty_prior_prob_kp.detach().cpu().tolist()
                if cluster_penalty_prior_prob_kp is not None
                else None
            ),
            "configured_allowed_mask": (
                cluster_penalty_prior_configured_mask_kp.detach().cpu().tolist()
                if cluster_penalty_prior_configured_mask_kp is not None
                else None
            ),
            "active_allowed_mask": (
                cluster_penalty_allowed_mask_kp.detach().cpu().tolist()
                if cluster_penalty_allowed_mask_kp is not None
                else None
            ),
            "late_allowed_mask": (
                cluster_penalty_late_allowed_mask_kp.detach().cpu().tolist()
                if cluster_penalty_late_allowed_mask_kp is not None
                else None
            ),
        },
        "model_train_stat_adapter": model_train_stat_adapter_summary,
        "train_stat_anchor_expert": train_stat_anchor_summary,
        "train_residual_anchor_expert": train_residual_anchor_summary,
        "learnable_output_anchor": learnable_output_anchor_summary,
        "learnable_output_anchor_refiner": learnable_output_anchor_refiner_summary,
        "learnable_output_anchor_test_refiner": learnable_output_anchor_test_refiner_summary,
        "moe_gate_penalty_hit": moe_gate_penalty_hit_summary,
        "penalty_explainability": penalty_explainability_summary,
        "penalty_route_learnability": penalty_route_learnability_summary,
        "moe_router": {
            "mode": str(router_mode),
            "penalty_context_weight": float(router_penalty_context_weight),
            "penalty_context_score": str(router_penalty_context_score),
            "detach_penalty_context": bool(router_detach_penalty_context),
            "context_applied_inside_gate_logits": True,
            "allow_skip": bool(allow_skip),
            "skip_competes_with_penalties": bool(skip_competes),
            "skip_argmax_noop": bool(skip_argmax_noop),
            "skip_cost": float(skip_cost),
            "skip_supervision_weight": float(skip_supervision_weight),
            "skip_supervision_margin": float(skip_supervision_margin),
            "freeze_gate_after_epoch": int(pred_residual_freeze_gate_after_epoch),
            "route_ce_supervision": {
                "enable": bool(route_ce_enable),
                "weight": float(route_ce_weight),
                "min_abs_improvement": float(route_ce_min_abs_improvement),
                "min_rel_improvement": float(route_ce_min_rel_improvement),
                "min_candidate_delta_rms": float(route_ce_min_candidate_delta_rms),
                "ignore_abs_gain_below": float(route_ce_ignore_abs_gain_below),
                "class_weight": str(route_ce_class_weight_mode),
                "max_class_weight": float(route_ce_max_class_weight),
                "require_skip": bool(route_ce_require_skip),
                "require_skip_competes": bool(route_ce_require_skip_competes),
                "require_skip_argmax_noop": bool(route_ce_require_skip_argmax_noop),
                "probs_include_skip_mass": bool(skip_competes),
            },
            "binary_adoption_supervision": {
                "enable": bool(binary_adoption_enable),
                "weight": float(binary_adoption_weight),
                "min_abs_improvement": float(binary_adoption_min_abs_improvement),
                "min_rel_improvement": float(binary_adoption_min_rel_improvement),
                "min_candidate_delta_rms": float(binary_adoption_min_candidate_delta_rms),
                "ignore_abs_gain_below": float(binary_adoption_ignore_abs_gain_below),
                "positive_weight": float(binary_adoption_positive_weight),
                "negative_weight": float(binary_adoption_negative_weight),
                "require_skip": bool(binary_adoption_require_skip),
                "require_skip_competes": bool(binary_adoption_require_skip_competes),
                "require_skip_argmax_noop": bool(binary_adoption_require_skip_argmax_noop),
                "probs_include_skip_mass": bool(skip_competes),
            },
            "route_rate_alignment_supervision": {
                "enable": bool(route_rate_alignment_enable),
                "weight": float(route_rate_alignment_weight),
                "min_abs_improvement": float(route_rate_alignment_min_abs_improvement),
                "min_rel_improvement": float(route_rate_alignment_min_rel_improvement),
                "min_candidate_delta_rms": float(route_rate_alignment_min_candidate_delta_rms),
                "ignore_abs_gain_below": float(route_rate_alignment_ignore_abs_gain_below),
                "require_skip": bool(route_rate_alignment_require_skip),
                "require_skip_competes": bool(route_rate_alignment_require_skip_competes),
                "require_skip_argmax_noop": bool(route_rate_alignment_require_skip_argmax_noop),
                "probs_include_skip_mass": bool(skip_competes),
            },
            "route_positive_recall_supervision": {
                "enable": bool(route_positive_recall_enable),
                "weight": float(route_positive_recall_weight),
                "min_abs_improvement": float(route_positive_recall_min_abs_improvement),
                "min_rel_improvement": float(route_positive_recall_min_rel_improvement),
                "min_candidate_delta_rms": float(route_positive_recall_min_candidate_delta_rms),
                "ignore_abs_gain_below": float(route_positive_recall_ignore_abs_gain_below),
                "mode": str(route_positive_recall_mode),
                "target_probability": float(route_positive_recall_target_probability),
                "require_skip": bool(route_positive_recall_require_skip),
                "require_skip_competes": bool(route_positive_recall_require_skip_competes),
                "require_skip_argmax_noop": bool(route_positive_recall_require_skip_argmax_noop),
                "probs_include_skip_mass": bool(skip_competes),
            },
            "route_precision_recall_supervision": {
                "enable": bool(route_precision_recall_enable),
                "weight": float(route_precision_recall_weight),
                "min_abs_improvement": float(route_precision_recall_min_abs_improvement),
                "min_rel_improvement": float(route_precision_recall_min_rel_improvement),
                "min_candidate_delta_rms": float(route_precision_recall_min_candidate_delta_rms),
                "ignore_abs_gain_below": float(route_precision_recall_ignore_abs_gain_below),
                "recall_mode": str(route_precision_recall_mode),
                "recall_target_probability": float(route_precision_recall_target_probability),
                "false_adopt_max_probability": float(route_precision_recall_false_adopt_max_probability),
                "false_adopt_weight": float(route_precision_recall_false_adopt_weight),
                "require_skip": bool(route_precision_recall_require_skip),
                "require_skip_competes": bool(route_precision_recall_require_skip_competes),
                "require_skip_argmax_noop": bool(route_precision_recall_require_skip_argmax_noop),
                "probs_include_skip_mass": bool(skip_competes),
            },
            "mse_utility_gate_supervision": {
                "enable": bool(mse_utility_gate_enable),
                "weight": float(mse_utility_gate_weight),
                "temperature": float(mse_utility_gate_temperature),
                "min_gain": float(mse_utility_gate_min_gain),
                "mae_weight": float(mse_utility_gate_mae_weight),
                "target_power": float(mse_utility_gate_target_power),
                "target_mode": str(mse_utility_gate_target_mode),
                "include_skip": bool(mse_utility_gate_include_skip),
                "probs_include_skip_mass": bool(skip_competes),
                "train_diagnostics": list(mse_gate_train_diag_history),
            },
        },
        "val": val_summary,
        "test": None if skip_test else {
            "avg_mae": avg_mae,
            "avg_mse": avg_mse,
            "base_avg_mae": base_avg_mae,
            "base_avg_mse": base_avg_mse,
            "gain_pct_vs_base": test_gain_pct_vs_base,
            "pre_moe_base_avg_mae": base_avg_mae,
            "pre_moe_base_avg_mse": base_avg_mse,
            "gain_pct_vs_pre_moe_base": test_gain_pct_vs_base,
            "per_cluster_loss": [float(v) for v in test_loss_k.detach().cpu().tolist()],
            "per_cluster_mse": [float(v) for v in test_mse_k.detach().cpu().tolist()],
            "per_cluster_mae": [float(v) for v in test_mae_k.detach().cpu().tolist()],
            "per_channel_mse": [float(v) for v in mse_c.detach().cpu().tolist()],
            "per_channel_mae": [float(v) for v in mae_c.detach().cpu().tolist()],
            "base_per_cluster_mse": [
                float(v)
                for v in test_base_metric_collector["avg_mse_k"].tolist()
            ],
            "base_per_cluster_mae": [
                float(v)
                for v in test_base_metric_collector["avg_mae_k"].tolist()
            ],
            "base_per_channel_mse": [
                float(v) for v in test_base_metric_collector["mse_c"].tolist()
            ],
            "base_per_channel_mae": [
                float(v) for v in test_base_metric_collector["mae_c"].tolist()
            ],
        },
        "selected": {
            "variant": selected_variant,
            "moe_residual_variant": moe_residual_variant,
            "criterion": selected_criterion,
            "selection_policy": selected_selection_policy,
            "avg_mae": selected_avg_mae,
            "avg_mse": selected_avg_mse,
            "base_val_mse": None if val_summary is None else val_summary.get("avg_mse"),
            "base_val_mae": None if val_summary is None else val_summary.get("avg_mae"),
        },
        "timing": {
            "total_sec": float(total_time),
            "avg_epoch_sec": float(avg_epoch_time),
        },
        "resources": {
            "cpu_rss_mb": float(cpu_rss_mb),
            "gpu_alloc_mb": float(gpu_alloc_mb),
            "gpu_reserved_mb": float(gpu_reserved_mb),
            "out_dir_size_mb": float(out_dir_mb),
        },
    }
    if stage2_loss_diagnostics_summary is not None:
        summary["stage2_loss_diagnostics"] = stage2_loss_diagnostics_summary
    if stage2_route_audit_summary is not None:
        summary["stage2_route_audit"] = stage2_route_audit_summary
    if finetune_summary is not None:
        summary["finetune"] = finetune_summary
    summary_path = os.path.join(out_dir, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved run summary to: {summary_path}")

    print("\nTime/Space Summary:")
    print(f"- total_time_s: {total_time:.3f}")
    print(f"- avg_epoch_time_s: {avg_epoch_time:.3f}")
    if cpu_rss_mb >= 0:
        print(f"- cpu_rss_mb: {cpu_rss_mb:.2f}")
    if device.type == "cuda":
        print(f"- gpu_max_alloc_mb: {gpu_alloc_mb:.2f}")
        print(f"- gpu_max_reserved_mb: {gpu_reserved_mb:.2f}")
    print(f"- out_dir_size_mb: {out_dir_mb:.2f}")


if __name__ == "__main__":
    main()
