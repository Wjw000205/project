import argparse
import copy
import math
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.windows import WindowTensorDataset, global_zscore, make_strict_windows
from src.models.cluster_predictor import build_cluster_predictor
from src.models.dynamic_lambda import ClusterwiseDynamicLambda
from src.models.learnable_lambda import ClusterwiseLearnableLambda
from src.models.moe_gate import ClusterwiseMoEGate, scatter_mean_bcf_to_bkf
from src.models.penalties import build_penalty_bank, normalize_penalties
from src.models.residual_moe import ClusterwisePredResidualMoE
from src.train import _compute_lambda_bkp, _select_rank_mask, extract_gate_features
from src.utils.cluster_memory import load_cluster_checkpoint, scatter_mean_bcl_to_bkl
from src.utils.clustering import cluster_channels_by_corr
from src.utils.pearson import pearson_corr_matrix


@dataclass
class DataContext:
    cfg: dict
    raw_df: pd.DataFrame
    date_col_name: str
    date_values: pd.Series
    channel_names: List[str]
    raw_data_tc: torch.Tensor
    norm_data_tc: torch.Tensor
    mean_c: torch.Tensor
    std_c: torch.Tensor
    cluster_id_c: torch.Tensor
    cluster_sizes: List[int]
    K: int
    L: int
    H: int
    t_train: int
    t_val: int
    xtr_norm: torch.Tensor
    ytr_norm: torch.Tensor
    xte_norm: torch.Tensor
    yte_norm: torch.Tensor
    xte_raw: torch.Tensor
    yte_raw: torch.Tensor


@dataclass
class EvalResult:
    yhat_raw: torch.Tensor
    mse_bc: torch.Tensor
    probs_bkp: Optional[torch.Tensor]
    mask_bkp: Optional[torch.Tensor]
    skip_bk: Optional[torch.Tensor]
    skip_prob_bk: Optional[torch.Tensor]
    lam_bkp: Optional[torch.Tensor]
    pen_bkp: Optional[torch.Tensor]
    contrib_bkp: Optional[torch.Tensor]
    penalty_names: List[str]
    skip_cost: float


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _safe_name(text: str) -> str:
    keep = []
    for ch in str(text):
        if ch.isalnum() or ch in {"_", "-"}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "item"


def expand_penalty_setting(raw_value, penalty_names: List[str], default_value, caster):
    if isinstance(raw_value, dict):
        base_default = raw_value.get("default", default_value)
        return [caster(raw_value.get(name, base_default)) for name in penalty_names]
    if isinstance(raw_value, (list, tuple)):
        if len(raw_value) != len(penalty_names):
            raise ValueError(f"Expected {len(penalty_names)} penalty values, got {len(raw_value)}")
        return [caster(v) for v in raw_value]
    value = default_value if raw_value is None else raw_value
    return [caster(value) for _ in penalty_names]


def build_run_config(
    base_cfg: dict,
    run_dir: Path,
    moe_enable: bool,
    epochs_override: Optional[int],
    disable_learnable_mse_weight: bool = False,
) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["exp"]["out_dir"] = str(run_dir)

    corr_cfg = cfg.setdefault("corr", {})
    corr_cfg["save_path"] = str(run_dir / "corr.npy")

    portrait_cfg = cfg.setdefault("portrait", {})
    portrait_cfg["out_dir"] = str(run_dir / "cluster_portraits")

    memory_cfg = cfg.setdefault("memory", {})
    memory_cfg["path"] = str(run_dir / "cluster_memory.pt")
    memory_cfg["checkpoint_path"] = str(run_dir / "best_checkpoint.pt")

    moe_cfg = cfg.setdefault("moe", {})
    moe_cfg["enable"] = bool(moe_enable)
    if disable_learnable_mse_weight:
        learn_mse_cfg = cfg.setdefault("train", {}).setdefault("learnable_mse_weight", {})
        learn_mse_cfg["enable"] = False

    if epochs_override is not None:
        cfg["train"]["epochs"] = int(epochs_override)

    return cfg


def run_training(repo_root: Path, config_path: Path, reuse_existing: bool) -> None:
    run_cfg = load_yaml(config_path)
    run_dir = Path(run_cfg["exp"]["out_dir"])
    checkpoint_path = run_dir / "best_checkpoint.pt"
    if checkpoint_path.exists() and reuse_existing:
        print(f"Reuse existing checkpoint: {checkpoint_path}")
        return
    cmd = [sys.executable, "-m", "src.train", "--config", str(config_path)]
    print(f"Run training: {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=repo_root)
    if completed.returncode != 0:
        raise RuntimeError(f"Training failed for {config_path} with return code {completed.returncode}")


def ensure_base_training(repo_root: Path, base_config_path: Path, reuse_existing: bool) -> Path:
    base_cfg = load_yaml(base_config_path)
    out_dir = Path(base_cfg["exp"]["out_dir"])
    checkpoint_path = out_dir / "best_checkpoint.pt"
    if checkpoint_path.exists() and reuse_existing:
        print(f"Reuse base moe_on checkpoint: {checkpoint_path}")
        return out_dir
    cmd = [sys.executable, "-m", "src.train", "--config", str(base_config_path)]
    print(f"Run base moe_on training: {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=repo_root)
    if completed.returncode != 0:
        raise RuntimeError(f"Training failed for base config {base_config_path} with return code {completed.returncode}")
    return out_dir


def prepare_data_context(cfg: dict) -> DataContext:
    csv_path = str(cfg["data"]["csv_path"])
    date_col = int(cfg["data"]["date_col"])
    raw_df = pd.read_csv(csv_path)
    max_rows = int(cfg.get("data", {}).get("max_rows", 0) or 0)
    if max_rows > 0:
        raw_df = raw_df.iloc[:max_rows].copy()
    date_col_name = raw_df.columns[date_col]
    date_values = pd.to_datetime(raw_df[date_col_name], errors="coerce")
    value_cols = [c for i, c in enumerate(raw_df.columns) if i != date_col]
    raw_values = raw_df[value_cols].to_numpy(dtype=np.float32)
    raw_data_tc = torch.tensor(raw_values, dtype=torch.float32)
    norm_data_tc = raw_data_tc.clone()

    tr = float(cfg["data"]["train_ratio"])
    vr = float(cfg["data"]["val_ratio"])
    te = float(cfg["data"]["test_ratio"])
    if abs(tr + vr + te - 1.0) > 1.0e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    T = int(raw_data_tc.shape[0])
    t_train = int(T * tr)
    t_val = int(T * (tr + vr))

    norm_cfg = cfg.get("normalize", {})
    mean_c = torch.zeros(raw_data_tc.shape[1], dtype=torch.float32)
    std_c = torch.ones(raw_data_tc.shape[1], dtype=torch.float32)
    if bool(norm_cfg.get("global_zscore", False)):
        if bool(norm_cfg.get("train_only", False)):
            train_seg = norm_data_tc[:t_train]
            mean_c = train_seg.mean(dim=0)
            std_c = train_seg.std(dim=0).clamp_min(1.0e-6)
            norm_data_tc = (norm_data_tc - mean_c.view(1, -1)) / std_c.view(1, -1)
        else:
            norm_data_tc, mean_c, std_c = global_zscore(norm_data_tc)

    cl = cfg["cluster"]
    method_norm = str(cl.get("method", "agglomerative")).lower()
    cluster_fit_tc = norm_data_tc[:t_train] if bool(cl.get("train_only", True)) else norm_data_tc
    if method_norm in {"random", "rand"}:
        corr_cc = torch.eye(norm_data_tc.shape[1], dtype=norm_data_tc.dtype)
    else:
        corr_cc = pearson_corr_matrix(cluster_fit_tc)
    cluster_id_c, _ = cluster_channels_by_corr(
        corr_cc=corr_cc,
        data_tc=norm_data_tc,
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
        random_state=cl.get("random_state", 0),
        min_cluster_size=int(cl["min_cluster_size"]),
        merge_small_clusters=bool(cl["merge_small_clusters"]),
        no_merge_if_channels_lt=int(cl["no_merge_if_channels_lt"]),
    )
    K = int(cluster_id_c.max().item() + 1)
    cluster_sizes = torch.bincount(cluster_id_c, minlength=K).tolist()

    L = int(cfg["window"]["input_len"])
    H = int(cfg["window"]["pred_len"])
    xtr_norm, ytr_norm = make_strict_windows(norm_data_tc, L, H, 0, t_train)
    xte_norm, yte_norm = make_strict_windows(norm_data_tc, L, H, t_val, T)
    xte_raw, yte_raw = make_strict_windows(raw_data_tc, L, H, t_val, T)
    if xte_norm.shape[0] == 0:
        raise ValueError("No test windows available")

    return DataContext(
        cfg=cfg,
        raw_df=raw_df,
        date_col_name=date_col_name,
        date_values=date_values,
        channel_names=value_cols,
        raw_data_tc=raw_data_tc,
        norm_data_tc=norm_data_tc,
        mean_c=mean_c,
        std_c=std_c,
        cluster_id_c=cluster_id_c,
        cluster_sizes=cluster_sizes,
        K=K,
        L=L,
        H=H,
        t_train=t_train,
        t_val=t_val,
        xtr_norm=xtr_norm,
        ytr_norm=ytr_norm,
        xte_norm=xte_norm,
        yte_norm=yte_norm,
        xte_raw=xte_raw,
        yte_raw=yte_raw,
    )


def compute_penalty_scale(
    xtr_norm: torch.Tensor,
    ytr_norm: torch.Tensor,
    penalty_names: List[str],
    penalty_fns: Dict[str, callable],
    pred_len: int,
    batch_size: int,
    device: torch.device,
    floor: float,
) -> torch.Tensor:
    P = len(penalty_names)
    if P == 0 or xtr_norm.shape[0] == 0:
        return torch.full((P,), floor, device=device)
    loader = DataLoader(WindowTensorDataset(xtr_norm, ytr_norm), batch_size=batch_size, shuffle=False, num_workers=0)
    sum_all = torch.zeros(P, device=device)
    sum_pos = torch.zeros(P, device=device)
    cnt_all = 0
    cnt_pos = torch.zeros(P, device=device)
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        last = x[..., -1:]
        yhat = last.expand(-1, -1, pred_len)
        pen_bcp = [penalty_fns[name](yhat, y) for name in penalty_names]
        pen_bcp = torch.stack(pen_bcp, dim=-1)
        pen_flat = pen_bcp.reshape(-1, P)
        sum_all += pen_flat.sum(dim=0)
        cnt_all += int(pen_flat.shape[0])
        pos = pen_flat > 0
        sum_pos += (pen_flat * pos).sum(dim=0)
        cnt_pos += pos.sum(dim=0)
    if cnt_all == 0:
        return torch.full((P,), floor, device=device)
    mean_all = sum_all / float(cnt_all)
    mean_pos = sum_pos / cnt_pos.clamp_min(1.0)
    scale = torch.where(cnt_pos > 0, mean_pos, mean_all)
    return scale.clamp_min(floor)


def build_base_lambda_kp(
    run_cfg: dict,
    meta: dict,
    penalty_names: List[str],
    K: int,
    device: torch.device,
    learnable_lambda: Optional[ClusterwiseLearnableLambda],
) -> torch.Tensor:
    P = len(penalty_names)
    if P == 0:
        return torch.zeros((K, 0), device=device)
    if learnable_lambda is not None:
        return learnable_lambda().detach()

    moe_cfg = meta.get("moe_cfg", run_cfg.get("moe", {}))
    epochs = int(run_cfg["train"]["epochs"])
    lambda_init_p = expand_penalty_setting(moe_cfg.get("lambda_init", 1.0), penalty_names, 1.0, float)
    lambda_min_p = expand_penalty_setting(moe_cfg.get("lambda_min", 0.0), penalty_names, 0.0, float)
    lambda_schedule_p = expand_penalty_setting(moe_cfg.get("lambda_schedule", "cosine"), penalty_names, "cosine", lambda v: str(v).lower())

    best_epoch = meta.get("best_epoch", None)
    if best_epoch is None:
        epoch_rows = [epochs for _ in range(K)]
    elif torch.is_tensor(best_epoch):
        epoch_rows = [int(v) for v in best_epoch.detach().cpu().tolist()]
    else:
        epoch_rows = [int(v) for v in best_epoch]

    rows = []
    for epoch_idx in epoch_rows:
        lam_p = []
        for p in range(P):
            lambda_max = lambda_init_p[p]
            lambda_min = lambda_min_p[p]
            lambda_schedule = lambda_schedule_p[p]
            if lambda_schedule in {"cosine", "cosineannealing"}:
                if epochs <= 1:
                    lam_p.append(lambda_max)
                else:
                    t = (epoch_idx - 1) / max(epochs - 1, 1)
                    lam_p.append(lambda_min + 0.5 * (lambda_max - lambda_min) * (1.0 + math.cos(math.pi * t)))
            else:
                lam_p.append(lambda_max)
        rows.append(torch.tensor(lam_p, dtype=torch.float32, device=device))
    return torch.stack(rows, dim=0)


def load_eval_modules(
    run_cfg: dict,
    checkpoint_path: Path,
    K: int,
    device: torch.device,
) -> dict:
    ckpt = load_cluster_checkpoint(str(checkpoint_path), device=device)
    meta = ckpt.get("meta", {})
    if len(meta) == 0:
        raise ValueError(f"Checkpoint meta missing: {checkpoint_path}")

    input_len = int(meta["input_len"])
    pred_len = int(meta["pred_len"])
    model_cfg = meta["model_cfg"]
    moe_cfg = meta.get("moe_cfg", {})
    penalty_names = list(meta.get("penalty_names", []))
    P = len(penalty_names)

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

    gate = None
    gate_state = ckpt.get("gate_state", None)
    if gate_state is not None and P > 0:
        gate_feat_dim = int(meta.get("gate_feat_dim", gate_state["W1.0"].shape[0]))
        gate_allow_skip = any(str(name).startswith("W_skip.") for name in gate_state.keys())
        gate = ClusterwiseMoEGate(
            num_clusters=K,
            feat_dim=gate_feat_dim,
            num_penalties=P,
            hidden_dim=int(moe_cfg.get("gate_hidden_dim", 64)),
            topk=int(moe_cfg.get("topk", 2)),
            allow_skip=gate_allow_skip,
            skip_init_bias=float(moe_cfg.get("skip_init_bias", -2.0)),
        ).to(device)
        gate.load_state_dict(gate_state)
        gate.temperature = float(moe_cfg.get("gate_temperature", 1.0))
        gate.noise_std = float(moe_cfg.get("gate_noise_std", 0.0))
        gate.logit_clip = float(moe_cfg.get("gate_logit_clip", 0.0))
        gate.prob_floor = float(moe_cfg.get("gate_prob_floor", 0.0))
        gate.eval()

    lambda_init_p = expand_penalty_setting(moe_cfg.get("lambda_init", 1.0), penalty_names, 1.0, float)
    lambda_min_p = expand_penalty_setting(moe_cfg.get("lambda_min", 0.0), penalty_names, 0.0, float)
    lambda_init_kp = torch.tensor(lambda_init_p, dtype=torch.float32, device=device).view(1, P).expand(K, P)
    lambda_min_kp = torch.tensor(lambda_min_p, dtype=torch.float32, device=device).view(1, P).expand(K, P)

    dynamic_lambda = None
    dyn_state = ckpt.get("dynamic_lambda_state", None)
    dyn_cfg = moe_cfg.get("dynamic_lambda", {})
    if dyn_state is not None and P > 0:
        dynamic_lambda = ClusterwiseDynamicLambda(
            num_clusters=K,
            feat_dim=int(meta.get("gate_feat_dim", 10)),
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
        dynamic_lambda.load_state_dict(dyn_state)
        dynamic_lambda.eval()

    learnable_lambda = None
    learn_state = ckpt.get("learnable_lambda_state", None)
    learn_cfg = moe_cfg.get("learnable_lambda", {})
    if learn_state is not None and P > 0:
        learnable_lambda = ClusterwiseLearnableLambda(
            init_lambda_kp=lambda_init_kp,
            lambda_min_kp=lambda_min_kp,
            share_floor=float(learn_cfg.get("share_floor", 0.0)),
        ).to(device)
        learnable_lambda.load_state_dict(learn_state)
        learnable_lambda.eval()

    pred_residual = None
    pred_residual_state = ckpt.get("pred_residual_state", None)
    pred_residual_cfg = moe_cfg.get("pred_side_residual", {}) or {}
    if pred_residual_state is not None and P > 0:
        pred_residual = ClusterwisePredResidualMoE(
            num_clusters=K,
            num_penalties=P,
            input_len=input_len,
            pred_len=pred_len,
            hidden_dim=int(pred_residual_cfg.get("corrector_hidden", 32)),
            init_alpha=float(pred_residual_cfg.get("init_alpha", -3.0)),
            alpha_scale=float(pred_residual_cfg.get("alpha_scale", 0.5)),
            use_y_base_input=bool(pred_residual_cfg.get("use_y_base_input", True)),
            feature_mode=str(pred_residual_cfg.get("feature_mode", "legacy")),
            residual_clip=float(pred_residual_cfg.get("residual_clip", 0.0)),
            intervention_enable=bool(pred_residual_cfg.get("intervention_enable", False)),
            intervention_init=float(pred_residual_cfg.get("intervention_init", -2.0)),
        ).to(device)
        pred_residual.load_state_dict(pred_residual_state)
        pred_residual.eval()

    base_lambda_kp = build_base_lambda_kp(run_cfg, meta, penalty_names, K, device, learnable_lambda)

    return {
        "ckpt": ckpt,
        "meta": meta,
        "model": model,
        "gate": gate,
        "dynamic_lambda": dynamic_lambda,
        "learnable_lambda": learnable_lambda,
        "pred_residual": pred_residual,
        "base_lambda_kp": base_lambda_kp,
        "lambda_min_kp": lambda_min_kp,
        "penalty_names": penalty_names,
        "moe_cfg": moe_cfg,
    }


def evaluate_run(
    context: DataContext,
    run_cfg: dict,
    run_dir: Path,
    device: torch.device,
    eval_batch_size: int,
) -> EvalResult:
    checkpoint_path = run_dir / "best_checkpoint.pt"
    bundle = load_eval_modules(run_cfg, checkpoint_path, context.K, device)
    model = bundle["model"]
    gate = bundle["gate"]
    dynamic_lambda = bundle["dynamic_lambda"]
    base_lambda_kp = bundle["base_lambda_kp"]
    lambda_min_kp = bundle["lambda_min_kp"]
    penalty_names = bundle["penalty_names"]
    moe_cfg = bundle["moe_cfg"]
    cluster_id_c_device = context.cluster_id_c.to(device)

    N, C, H = context.yte_raw.shape
    yhat_raw = torch.empty((N, C, H), dtype=torch.float32)
    mse_bc = torch.empty((N, C), dtype=torch.float32)

    probs_all = None
    mask_all = None
    skip_all = None
    skip_prob_all = None
    lam_all = None
    pen_all = None
    contrib_all = None

    P = len(penalty_names)
    use_moe = bool(moe_cfg.get("enable", True)) and (gate is not None) and P > 0
    use_skip = use_moe and bool(getattr(gate, "allow_skip", False))
    if use_moe:
        probs_all = torch.empty((N, context.K, P), dtype=torch.float32)
        mask_all = torch.empty((N, context.K, P), dtype=torch.float32)
        if use_skip:
            skip_all = torch.empty((N, context.K), dtype=torch.float32)
            skip_prob_all = torch.empty((N, context.K), dtype=torch.float32)
        lam_all = torch.empty((N, context.K, P), dtype=torch.float32)
        pen_all = torch.empty((N, context.K, P), dtype=torch.float32)
        contrib_all = torch.empty((N, context.K, P), dtype=torch.float32)

    penalty_scale = None
    penalty_fns = None
    if use_moe:
        penalty_fns = build_penalty_bank(penalty_names, jump_thr=float(context.cfg["penalties"]["jump_threshold"]))
        penalty_scale = compute_penalty_scale(
            xtr_norm=context.xtr_norm,
            ytr_norm=context.ytr_norm,
            penalty_names=penalty_names,
            penalty_fns=penalty_fns,
            pred_len=context.H,
            batch_size=eval_batch_size,
            device=device,
            floor=float(context.cfg["train"].get("penalty_scale_floor", 1.0e-3)),
        )

    loader = DataLoader(WindowTensorDataset(context.xte_norm, context.yte_norm), batch_size=eval_batch_size, shuffle=False, num_workers=0)
    mean_c = context.mean_c.view(1, -1, 1).to(device)
    std_c = context.std_c.view(1, -1, 1).to(device)

    with torch.no_grad():
        for x, y, idx in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            idx_cpu = idx.long()

            yhat = model(x, cluster_id_c_device)
            mse_batch = (yhat - y).pow(2).mean(dim=-1).detach().cpu()
            yhat_batch_raw = (yhat * std_c) + mean_c
            yhat_raw[idx_cpu] = yhat_batch_raw.detach().cpu()
            mse_bc[idx_cpu] = mse_batch

            if not use_moe:
                continue

            feat_bcf = extract_gate_features(x)
            feat_bkf = scatter_mean_bcf_to_bkf(feat_bcf, cluster_id_c_device, context.K)
            series_bkl = scatter_mean_bcl_to_bkl(x, cluster_id_c_device, context.K)
            mask_bkp, probs_bkp, skip_bk, skip_prob_bk = gate(feat_bkf, straight_through=False)
            select_ranks = moe_cfg.get("select_ranks", None)
            if select_ranks is not None:
                mask_bkp = _select_rank_mask(probs_bkp, [int(v) for v in select_ranks], straight_through=False)

            lam_bkp = _compute_lambda_bkp(
                base_lambda_kp=base_lambda_kp,
                feat_bkf=feat_bkf,
                series_bkl=series_bkl,
                dynamic_lambda=dynamic_lambda,
                lambda_min_kp=lambda_min_kp,
            )
            pen_bcp = torch.stack([penalty_fns[name](yhat, y) for name in penalty_names], dim=-1)
            pen_bcp = normalize_penalties(pen_bcp, scale=penalty_scale)
            pen_bkp = scatter_mean_bcf_to_bkf(pen_bcp, cluster_id_c_device, context.K)
            contrib_bkp = mask_bkp * lam_bkp * pen_bkp
            if use_skip:
                contrib_bkp = (1.0 - skip_bk.unsqueeze(-1)) * contrib_bkp

            probs_all[idx_cpu] = probs_bkp.detach().cpu()
            mask_all[idx_cpu] = mask_bkp.detach().cpu()
            if use_skip:
                skip_all[idx_cpu] = skip_bk.detach().cpu()
                skip_prob_all[idx_cpu] = skip_prob_bk.detach().cpu()
            lam_all[idx_cpu] = lam_bkp.detach().cpu()
            pen_all[idx_cpu] = pen_bkp.detach().cpu()
            contrib_all[idx_cpu] = contrib_bkp.detach().cpu()

    return EvalResult(
        yhat_raw=yhat_raw,
        mse_bc=mse_bc,
        probs_bkp=probs_all,
        mask_bkp=mask_all,
        skip_bk=skip_all,
        skip_prob_bk=skip_prob_all,
        lam_bkp=lam_all,
        pen_bkp=pen_all,
        contrib_bkp=contrib_all,
        penalty_names=penalty_names,
        skip_cost=float(moe_cfg.get("skip_cost", 0.0)) if use_skip else 0.0,
    )


def window_position_info(context: DataContext, window_idx: int) -> dict:
    start = context.t_val + int(window_idx)
    hist_start = start
    hist_end = start + context.L - 1
    pred_start = start + context.L
    pred_end = start + context.L + context.H - 1

    def _ts(i: int) -> str:
        if i < 0 or i >= len(context.date_values):
            return ""
        value = context.date_values.iloc[i]
        return "" if pd.isna(value) else str(value)

    return {
        "window_idx": int(window_idx),
        "hist_start_idx": hist_start,
        "hist_end_idx": hist_end,
        "pred_start_idx": pred_start,
        "pred_end_idx": pred_end,
        "hist_start_time": _ts(hist_start),
        "hist_end_time": _ts(hist_end),
        "pred_start_time": _ts(pred_start),
        "pred_end_time": _ts(pred_end),
    }


def active_penalty_text(result: EvalResult, window_idx: int, cluster_id: int, topn: int) -> str:
    if result.probs_bkp is None or result.mask_bkp is None:
        return "active penalties: none"
    probs = result.probs_bkp[window_idx, cluster_id]
    mask = result.mask_bkp[window_idx, cluster_id]
    skip = None if result.skip_bk is None else result.skip_bk[window_idx, cluster_id]
    skip_prob = None if result.skip_prob_bk is None else result.skip_prob_bk[window_idx, cluster_id]
    lam = None if result.lam_bkp is None else result.lam_bkp[window_idx, cluster_id]
    contrib = None if result.contrib_bkp is None else result.contrib_bkp[window_idx, cluster_id]

    prefix = None
    if skip is not None and skip_prob is not None:
        prefix = f"skip={'on' if float(skip.item()) > 0.5 else 'off'} p={float(skip_prob.item()):.3f}"
        if result.skip_cost > 0.0:
            prefix += f", cost={result.skip_cost:.3f}"
        if float(skip.item()) > 0.5:
            return prefix + "; penalties bypassed"

    selected = torch.nonzero(mask > 0.5, as_tuple=False).view(-1)
    if selected.numel() == 0:
        selected = torch.argsort(probs, descending=True)[: max(1, min(topn, probs.shape[0]))]
    else:
        order = torch.argsort(probs[selected], descending=True)
        selected = selected.index_select(0, order)

    parts = []
    for p in selected.tolist()[:topn]:
        text = f"{result.penalty_names[p]} p={float(probs[p].item()):.3f}"
        if lam is not None:
            text += f", lam={float(lam[p].item()):.3f}"
        if contrib is not None:
            text += f", c={float(contrib[p].item()):.3f}"
        parts.append(text)
    if len(parts) == 0:
        return prefix if prefix is not None else "active penalties: none"
    if prefix is not None:
        return prefix + "; active penalties: " + "; ".join(parts)
    return "active penalties: " + "; ".join(parts)


def plot_selected_window(
    context: DataContext,
    off_result: EvalResult,
    on_result: EvalResult,
    window_idx: int,
    rank: int,
    out_dir: Path,
    max_channels: int,
    dpi: int,
    active_topn: int,
    selection_mode: str = "best",
) -> List[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    channel_gain = off_result.mse_bc[window_idx] - on_result.mse_bc[window_idx]
    C = int(channel_gain.shape[0])
    top_channel_count = max(1, min(int(max_channels), C))
    mode = str(selection_mode).lower()
    descending = mode != "worst"
    channel_order = torch.argsort(channel_gain, descending=descending)[:top_channel_count].tolist()

    n_plot = len(channel_order)
    ncols = 1 if n_plot == 1 else (2 if n_plot <= 4 else 3)
    nrows = int(math.ceil(n_plot / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.0 * ncols, 3.8 * nrows + 1.2),
        squeeze=False,
    )
    axes_flat = axes.reshape(-1)

    x_raw = context.xte_raw[window_idx]
    y_raw = context.yte_raw[window_idx]
    off_pred = off_result.yhat_raw[window_idx]
    on_pred = on_result.yhat_raw[window_idx]
    t_hist = np.arange(context.L)
    t_pred = np.arange(context.L, context.L + context.H)
    pos_info = window_position_info(context, window_idx)
    details = []
    footer_lines: List[str] = []

    for ax, c in zip(axes_flat, channel_order):
        cluster_id = int(context.cluster_id_c[c].item())
        ax.plot(t_hist, x_raw[c].numpy(), color="#8f8f8f", linewidth=1.4, label="history")
        ax.plot(t_pred, y_raw[c].numpy(), color="#111111", linewidth=1.6, label="true")
        ax.plot(t_pred, off_pred[c].numpy(), color="#d95f02", linewidth=1.4, linestyle="--", label="moe_off")
        ax.plot(t_pred, on_pred[c].numpy(), color="#1b9e77", linewidth=1.4, label="moe_on")
        ax.axvline(context.L - 1, color="#b0b0b0", linewidth=1.0, linestyle=":")
        off_mse = float(off_result.mse_bc[window_idx, c].item())
        on_mse = float(on_result.mse_bc[window_idx, c].item())
        gain = off_mse - on_mse
        skip_active = None if on_result.skip_bk is None else float(on_result.skip_bk[window_idx, cluster_id].item())
        skip_prob = None if on_result.skip_prob_bk is None else float(on_result.skip_prob_bk[window_idx, cluster_id].item())
        penalty_text = active_penalty_text(on_result, window_idx, cluster_id, active_topn)
        ax.set_title(
            f"{context.channel_names[c]} | cluster={cluster_id} | "
            f"off={off_mse:.4f} on={on_mse:.4f} gain={gain:.4f}",
            fontsize=10,
        )
        footer_lines.append(f"{context.channel_names[c]} (cluster={cluster_id}): {penalty_text}")
        ax.grid(alpha=0.2)
        details.append({
            "subset": mode,
            "rank": rank,
            "window_idx": int(window_idx),
            "channel_idx": int(c),
            "channel": context.channel_names[c],
            "cluster_id": cluster_id,
            "mse_off": off_mse,
            "mse_on": on_mse,
            "mse_gain": gain,
            "skip_active": skip_active,
            "skip_prob": skip_prob,
            "skip_cost": on_result.skip_cost,
            "active_penalties": penalty_text,
            **pos_info,
        })

    for ax in axes_flat[n_plot:]:
        ax.axis("off")

    axes_flat[0].legend(loc="upper left", fontsize=9)
    mean_off = float(off_result.mse_bc[window_idx].mean().item())
    mean_on = float(on_result.mse_bc[window_idx].mean().item())
    mean_gain = mean_off - mean_on
    title_prefix = "Best" if descending else "Worst"
    fig.suptitle(
        f"{title_prefix} {rank} Window {window_idx} | "
        f"mean_mse_off={mean_off:.4f} mean_mse_on={mean_on:.4f} gain={mean_gain:.4f}\n"
        f"pred_range=[{pos_info['pred_start_idx']}, {pos_info['pred_end_idx']}] "
        f"{pos_info['pred_start_time']} -> {pos_info['pred_end_time']}",
        fontsize=13,
    )
    footer_width = 100 if ncols == 1 else 160
    footer_text = "\n".join(textwrap.fill(line, width=footer_width, subsequent_indent="  ") for line in footer_lines)
    footer_line_count = max(1, footer_text.count("\n") + 1)
    bottom_margin = min(0.34, 0.10 + 0.04 * footer_line_count)
    fig.tight_layout(rect=[0, bottom_margin, 1, 0.95])
    fig.text(
        0.02,
        0.02,
        footer_text,
        ha="left",
        va="bottom",
        fontsize=9,
    )
    fig.savefig(out_dir / f"{mode}_{rank:02d}_window_{window_idx:05d}.png", dpi=dpi)
    plt.close(fig)
    return details


def select_windows(
    off_result: EvalResult,
    on_result: EvalResult,
    topk: int,
    min_gap: int,
    descending: bool = True,
) -> torch.Tensor:
    gain = (off_result.mse_bc - on_result.mse_bc).mean(dim=1)
    order = torch.argsort(gain, descending=descending)
    selected: List[int] = []
    min_gap = max(1, int(min_gap))
    for idx in order.tolist():
        if any(abs(idx - kept) < min_gap for kept in selected):
            continue
        selected.append(int(idx))
        if len(selected) >= topk:
            break
    if len(selected) < topk:
        for idx in order.tolist():
            idx = int(idx)
            if idx in selected:
                continue
            selected.append(idx)
            if len(selected) >= topk:
                break
    return torch.tensor(selected[:topk], dtype=torch.long)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--eval-batch-size", type=int, default=256)
    ap.add_argument("--max-channels", type=int, default=1)
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--active-topn", type=int, default=2)
    ap.add_argument("--min-window-gap", type=int, default=None)
    ap.add_argument("--epochs-override", type=int, default=None)
    ap.add_argument("--reuse-moe-on", action="store_true")
    ap.add_argument("--reuse-existing", action="store_true")
    ap.add_argument(
        "--off-disable-learnable-mse-weight",
        action="store_true",
        help="Disable train.learnable_mse_weight only for the moe_off run.",
    )
    args = ap.parse_args()

    repo_root = REPO_ROOT
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (repo_root / config_path).resolve()
    base_cfg = load_yaml(config_path)

    compare_root = Path(args.out_dir) if args.out_dir is not None else (Path(base_cfg["exp"]["out_dir"]) / "moe_compare")
    if not compare_root.is_absolute():
        compare_root = (repo_root / compare_root).resolve()
    compare_root.mkdir(parents=True, exist_ok=True)

    runs_dir = compare_root / "runs"
    configs_dir = compare_root / "configs"
    plots_dir = compare_root / "plots"
    runs_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    for old_plot in plots_dir.glob("*.png"):
        old_plot.unlink()

    cfg_on = copy.deepcopy(base_cfg)
    cfg_on["exp"]["out_dir"] = str(runs_dir / "moe_on")
    corr_cfg_on = cfg_on.setdefault("corr", {})
    corr_cfg_on["save_path"] = str((runs_dir / "moe_on") / "corr.npy")
    portrait_cfg_on = cfg_on.setdefault("portrait", {})
    portrait_cfg_on["out_dir"] = str((runs_dir / "moe_on") / "cluster_portraits")
    memory_cfg_on = cfg_on.setdefault("memory", {})
    memory_cfg_on["path"] = str((runs_dir / "moe_on") / "cluster_memory.pt")
    memory_cfg_on["checkpoint_path"] = str((runs_dir / "moe_on") / "best_checkpoint.pt")
    if args.epochs_override is not None:
        cfg_on["train"]["epochs"] = int(args.epochs_override)
    cfg_off = build_run_config(
        base_cfg,
        runs_dir / "moe_off",
        moe_enable=False,
        epochs_override=args.epochs_override,
        disable_learnable_mse_weight=bool(args.off_disable_learnable_mse_weight),
    )
    cfg_on_path = configs_dir / "moe_on.yaml"
    cfg_off_path = configs_dir / "moe_off.yaml"
    dump_yaml(cfg_on_path, cfg_on)
    dump_yaml(cfg_off_path, cfg_off)

    if args.reuse_moe_on:
        moe_on_run_dir = ensure_base_training(repo_root, config_path, reuse_existing=args.reuse_existing)
    else:
        run_training(repo_root, cfg_on_path, reuse_existing=args.reuse_existing)
        moe_on_run_dir = Path(cfg_on["exp"]["out_dir"])
    run_training(repo_root, cfg_off_path, reuse_existing=args.reuse_existing)

    device_name = base_cfg["exp"].get("device", "cpu")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    context = prepare_data_context(base_cfg)

    off_result = evaluate_run(context, cfg_off, runs_dir / "moe_off", device, args.eval_batch_size)
    on_result = evaluate_run(context, cfg_on, moe_on_run_dir, device, args.eval_batch_size)

    min_window_gap = context.H if args.min_window_gap is None else int(args.min_window_gap)
    best_windows = select_windows(
        off_result,
        on_result,
        topk=max(1, int(args.topk)),
        min_gap=min_window_gap,
        descending=True,
    )
    worst_windows = select_windows(
        off_result,
        on_result,
        topk=max(1, int(args.topk)),
        min_gap=min_window_gap,
        descending=False,
    )

    def _render_window_set(window_ids: torch.Tensor, subset: str) -> tuple[list, list]:
        summary_rows = []
        detail_rows = []
        for rank, window_idx in enumerate(window_ids.tolist(), start=1):
            pos_info = window_position_info(context, window_idx)
            mean_off = float(off_result.mse_bc[window_idx].mean().item())
            mean_on = float(on_result.mse_bc[window_idx].mean().item())
            summary_rows.append({
                "subset": subset,
                "rank": rank,
                "window_idx": int(window_idx),
                "mean_mse_off": mean_off,
                "mean_mse_on": mean_on,
                "mean_mse_gain": mean_off - mean_on,
                **pos_info,
            })
            detail_rows.extend(
                plot_selected_window(
                    context=context,
                    off_result=off_result,
                    on_result=on_result,
                    window_idx=window_idx,
                    rank=rank,
                    out_dir=plots_dir,
                    max_channels=max(1, int(args.max_channels)),
                    dpi=int(args.dpi),
                    active_topn=max(1, int(args.active_topn)),
                    selection_mode=subset,
                )
            )
        return summary_rows, detail_rows

    best_summary_rows, best_detail_rows = _render_window_set(best_windows, "best")
    worst_summary_rows, worst_detail_rows = _render_window_set(worst_windows, "worst")

    pd.DataFrame(best_summary_rows).to_csv(compare_root / "best_windows.csv", index=False)
    pd.DataFrame(worst_summary_rows).to_csv(compare_root / "worst_windows.csv", index=False)
    pd.DataFrame(best_detail_rows).to_csv(compare_root / "best_window_channel_details.csv", index=False)
    pd.DataFrame(worst_detail_rows).to_csv(compare_root / "worst_window_channel_details.csv", index=False)
    pd.DataFrame(best_summary_rows).to_csv(compare_root / "top_windows.csv", index=False)
    pd.DataFrame(best_detail_rows).to_csv(compare_root / "top_window_channel_details.csv", index=False)

    print(f"Saved comparison configs to: {configs_dir}")
    print(f"Saved runs to: {runs_dir}")
    print(f"Saved best-window summary to: {compare_root / 'best_windows.csv'}")
    print(f"Saved worst-window summary to: {compare_root / 'worst_windows.csv'}")
    print(f"Saved best per-channel details to: {compare_root / 'best_window_channel_details.csv'}")
    print(f"Saved worst per-channel details to: {compare_root / 'worst_window_channel_details.csv'}")
    print(f"Saved plots to: {plots_dir}")


if __name__ == "__main__":
    main()
