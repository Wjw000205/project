from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.windows import WindowTensorDataset, global_zscore, make_label_range_windows, make_strict_windows
from src.models.cluster_predictor import build_cluster_predictor
from src.models.moe_gate import ClusterwiseMoEGate, scatter_mean_bcf_to_bkf
from src.models.residual_moe import ClusterwisePredResidualMoE
from src.train import GATE_FEATURE_NAMES, _select_rank_mask, extract_gate_features
from src.utils.cluster_memory import load_cluster_checkpoint
from src.utils.clustering import cluster_channels_by_corr
from src.utils.pearson import pearson_corr_matrix


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _normalize_and_cluster(cfg: dict) -> Tuple[pd.DataFrame, List[str], torch.Tensor, torch.Tensor, torch.Tensor, int, int, torch.Tensor]:
    csv_path = _resolve(str(cfg["data"]["csv_path"]))
    date_col = int(cfg["data"].get("date_col", 0))
    raw_df = pd.read_csv(csv_path)
    max_rows = int(cfg.get("data", {}).get("max_rows", 0) or 0)
    if max_rows > 0:
        raw_df = raw_df.iloc[:max_rows].copy()
    value_cols = [c for i, c in enumerate(raw_df.columns) if i != date_col]
    raw_tc = torch.tensor(raw_df[value_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
    T = int(raw_tc.shape[0])
    t_train = int(T * float(cfg["data"]["train_ratio"]))
    t_val = int(T * (float(cfg["data"]["train_ratio"]) + float(cfg["data"]["val_ratio"])))

    norm_tc = raw_tc.clone()
    norm_cfg = cfg.get("normalize", {}) or {}
    if bool(norm_cfg.get("global_zscore", False)):
        if bool(norm_cfg.get("train_only", False)):
            train_seg = norm_tc[:t_train]
            mean_c = train_seg.mean(dim=0)
            std_c = train_seg.std(dim=0).clamp_min(1.0e-6)
            norm_tc = (norm_tc - mean_c.view(1, -1)) / std_c.view(1, -1)
        else:
            norm_tc, _, _ = global_zscore(norm_tc)

    cl = cfg.get("cluster", {}) or {}
    fit_tc = norm_tc[:t_train] if bool(cl.get("train_only", True)) else norm_tc
    method = str(cl.get("method", "leader")).lower()
    corr_cc = torch.eye(norm_tc.shape[1], dtype=norm_tc.dtype) if method in {"random", "rand"} else pearson_corr_matrix(fit_tc)
    cluster_id_c, _ = cluster_channels_by_corr(
        corr_cc=corr_cc,
        data_tc=fit_tc,
        n_clusters=cl.get("n_clusters", None),
        distance_threshold=cl.get("distance_threshold", None),
        linkage=cl.get("linkage", "average"),
        method=cl.get("method", "leader"),
        kmeans_n_init=int(cl.get("kmeans_n_init", 10)),
        kmeans_max_iter=int(cl.get("kmeans_max_iter", 300)),
        spectral_affinity=cl.get("spectral_affinity", "corr"),
        rbf_gamma=float(cl.get("rbf_gamma", 1.0)),
        dbscan_eps=cl.get("dbscan_eps", None),
        dbscan_min_samples=int(cl.get("dbscan_min_samples", 5)),
        random_state=cl.get("random_state", 0),
        min_cluster_size=int(cl.get("min_cluster_size", 2)),
        merge_small_clusters=bool(cl.get("merge_small_clusters", True)),
        no_merge_if_channels_lt=int(cl.get("no_merge_if_channels_lt", 7)),
    )
    return raw_df, value_cols, raw_tc, norm_tc, cluster_id_c.long(), t_train, t_val, corr_cc


def _make_split_windows(cfg: dict, norm_tc: torch.Tensor, split: str, t_train: int, t_val: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
    L = int(cfg["window"]["input_len"])
    H = int(cfg["window"]["pred_len"])
    T = int(norm_tc.shape[0])
    past_context = bool(cfg.get("window", {}).get("past_context", False))
    if split == "train":
        x, y = make_strict_windows(norm_tc, L, H, 0, t_train)
        return x, y, 0
    if split == "val":
        if past_context:
            return make_label_range_windows(norm_tc, L, H, t_train, t_val)
        x, y = make_strict_windows(norm_tc, L, H, t_train, t_val)
        return x, y, t_train
    if split == "test":
        if past_context:
            return make_label_range_windows(norm_tc, L, H, t_val, T)
        x, y = make_strict_windows(norm_tc, L, H, t_val, T)
        return x, y, t_val
    raise ValueError(f"Unsupported split: {split}")


def _build_modules(cfg: dict, checkpoint_path: Path, device: torch.device):
    ckpt = load_cluster_checkpoint(str(checkpoint_path), device=device)
    meta = ckpt.get("meta", {}) or {}
    if not meta:
        raise ValueError(f"Checkpoint has no meta: {checkpoint_path}")
    cluster_id_c = meta["cluster_id_c"].to(device=device, dtype=torch.long)
    K = int(meta["K"])
    C = int(meta.get("num_channels", int(cluster_id_c.numel())))
    L = int(meta["input_len"])
    H = int(meta["pred_len"])
    model_cfg = meta.get("model_cfg", cfg.get("model", {}))
    moe_cfg = meta.get("moe_cfg", cfg.get("moe", {})) or {}
    penalty_names = list(meta.get("penalty_names", cfg.get("penalties", {}).get("enabled", [])))
    P = len(penalty_names)

    model = build_cluster_predictor(
        num_clusters=K,
        input_len=L,
        pred_len=H,
        model_cfg=model_cfg,
        num_channels=C,
        cluster_id_c=cluster_id_c,
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    gate_state = ckpt.get("gate_state", {})
    allow_skip = any(str(k).startswith("W_skip.") for k in gate_state.keys())
    gate = ClusterwiseMoEGate(
        num_clusters=K,
        feat_dim=int(meta.get("gate_feat_dim", len(GATE_FEATURE_NAMES))),
        num_penalties=P,
        hidden_dim=int(moe_cfg.get("gate_hidden_dim", 64)),
        topk=int(moe_cfg.get("topk", 1)),
        allow_skip=allow_skip,
        skip_init_bias=float(moe_cfg.get("skip_init_bias", -2.0)),
    ).to(device)
    gate.temperature = float(moe_cfg.get("gate_temperature", 1.0))
    gate.noise_std = 0.0
    gate.logit_clip = float(moe_cfg.get("gate_logit_clip", 0.0))
    gate.prob_floor = float(moe_cfg.get("gate_prob_floor", 0.0))
    gate.load_state_dict(gate_state, strict=True)
    gate.eval()

    pred_state = ckpt.get("pred_residual_state", None)
    if pred_state is None:
        raise ValueError("Checkpoint has no pred_residual_state; rerun training with moe.pred_side_residual.enable=true and memory.save_checkpoint=true.")
    pred_cfg = moe_cfg.get("pred_side_residual", {}) or {}
    pred_residual = ClusterwisePredResidualMoE(
        num_clusters=K,
        num_penalties=P,
        input_len=L,
        pred_len=H,
        hidden_dim=int(pred_cfg.get("corrector_hidden", 32)),
        init_alpha=float(pred_cfg.get("init_alpha", -3.0)),
        alpha_scale=float(pred_cfg.get("alpha_scale", 0.5)),
        use_y_base_input=bool(pred_cfg.get("use_y_base_input", True)),
        feature_mode=str(pred_cfg.get("feature_mode", "legacy")),
        residual_clip=float(pred_cfg.get("residual_clip", 0.0)),
        intervention_enable=bool(pred_cfg.get("intervention_enable", False)),
        intervention_init=float(pred_cfg.get("intervention_init", -2.0)),
    ).to(device)
    pred_residual.load_state_dict(pred_state, strict=True)
    pred_residual.eval()
    return {
        "model": model,
        "gate": gate,
        "pred_residual": pred_residual,
        "cluster_id_c": cluster_id_c,
        "K": K,
        "C": C,
        "L": L,
        "H": H,
        "P": P,
        "penalty_names": penalty_names,
        "moe_cfg": moe_cfg,
    }


def _selected_penalty_from_mask(mask_bkp: torch.Tensor, probs_bkp: torch.Tensor, cluster_id_c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Use the active route if available, and break ties by soft probability.
    channel_mask = mask_bkp[:, cluster_id_c, :]
    channel_probs = probs_bkp[:, cluster_id_c, :]
    score = channel_probs.masked_fill(channel_mask <= 0.0, -1.0)
    selected = score.argmax(dim=-1)
    selected_prob = channel_probs.gather(-1, selected.unsqueeze(-1)).squeeze(-1)
    return selected, selected_prob


def _feature_frame(
    features_bcf: torch.Tensor,
    y_base: torch.Tensor,
    x: torch.Tensor,
    channel_names: List[str],
    cluster_id_c: torch.Tensor,
    start_offset: int,
    batch_indices: torch.Tensor,
) -> pd.DataFrame:
    B, C, _ = features_bcf.shape
    rows = {}
    flat = features_bcf.detach().cpu().reshape(B * C, -1).numpy()
    for i, name in enumerate(GATE_FEATURE_NAMES):
        rows[f"feat_{name}"] = flat[:, i]

    hist_std = x.std(dim=-1).clamp_min(1.0e-6)
    rows["base_std_over_hist"] = (y_base.std(dim=-1) / hist_std).detach().cpu().reshape(-1).numpy()
    rows["base_shift_over_hist"] = ((y_base.mean(dim=-1) - x[..., -1]) / hist_std).detach().cpu().reshape(-1).numpy()
    rows["base_range_over_hist"] = ((y_base.amax(dim=-1) - y_base.amin(dim=-1)) / hist_std).detach().cpu().reshape(-1).numpy()
    rows["window_idx"] = np.repeat(batch_indices.detach().cpu().numpy() + int(start_offset), C)
    rows["channel"] = np.tile(channel_names, B)
    rows["channel_idx"] = np.tile(np.arange(C), B)
    rows["cluster"] = np.tile(cluster_id_c.detach().cpu().numpy(), B)
    return pd.DataFrame(rows)


def _collect_oracle_rows(
    cfg: dict,
    modules: dict,
    x_split: torch.Tensor,
    y_split: torch.Tensor,
    channel_names: List[str],
    split_start_offset: int,
    batch_size: int,
    device: torch.device,
    max_windows: int,
    eps: float,
) -> pd.DataFrame:
    model = modules["model"]
    gate = modules["gate"]
    pred_residual = modules["pred_residual"]
    cluster_id_c = modules["cluster_id_c"]
    K = int(modules["K"])
    P = int(modules["P"])
    penalty_names = list(modules["penalty_names"])
    select_ranks_raw = (modules["moe_cfg"] or {}).get("select_ranks", None)
    select_ranks = None if select_ranks_raw is None else [int(v) for v in select_ranks_raw]

    if max_windows > 0:
        x_split = x_split[:max_windows]
        y_split = y_split[:max_windows]
    loader = DataLoader(WindowTensorDataset(x_split, y_split), batch_size=batch_size, shuffle=False, num_workers=0)
    all_frames: List[pd.DataFrame] = []

    with torch.no_grad():
        for x, y, idx in loader:
            x = x.to(device)
            y = y.to(device)
            idx = idx.to(device=device, dtype=torch.long)
            y_base = model(x, cluster_id_c)
            base_mse = (y_base - y).pow(2).mean(dim=-1)

            feat_bcf = extract_gate_features(x)
            feat_bkf = scatter_mean_bcf_to_bkf(feat_bcf, cluster_id_c, K)
            mask_bkp, probs_bkp, skip_bk, skip_prob_bk = gate(feat_bkf, straight_through=False)
            if select_ranks is not None:
                mask_bkp = _select_rank_mask(probs_bkp, select_ranks, straight_through=False)

            actual = pred_residual(x, y_base, cluster_id_c, mask_bkp, skip_bk=skip_bk)
            gate_mse = (actual["y_final"] - y).pow(2).mean(dim=-1)

            mse_by_penalty = []
            for p in range(P):
                one = torch.zeros_like(mask_bkp)
                one[:, :, p] = 1.0
                no_skip = torch.zeros_like(skip_bk)
                out_p = pred_residual(x, y_base, cluster_id_c, one, skip_bk=no_skip)
                mse_p = (out_p["y_final"] - y).pow(2).mean(dim=-1)
                mse_by_penalty.append(mse_p)
            mse_bcp = torch.stack(mse_by_penalty, dim=-1)
            best_mse, best_idx = mse_bcp.min(dim=-1)
            best_gain = base_mse - best_mse
            best_positive = best_gain > float(eps)

            selected_idx, selected_prob = _selected_penalty_from_mask(mask_bkp, probs_bkp, cluster_id_c)
            selected_mse = mse_bcp.gather(-1, selected_idx.unsqueeze(-1)).squeeze(-1)
            selected_gain = base_mse - selected_mse

            frame = _feature_frame(feat_bcf, y_base, x, channel_names, cluster_id_c, split_start_offset, idx)
            B, C = base_mse.shape
            frame["base_mse"] = base_mse.detach().cpu().reshape(-1).numpy()
            frame["gate_mse"] = gate_mse.detach().cpu().reshape(-1).numpy()
            frame["gate_gain"] = (base_mse - gate_mse).detach().cpu().reshape(-1).numpy()
            frame["best_mse"] = best_mse.detach().cpu().reshape(-1).numpy()
            frame["best_gain"] = best_gain.detach().cpu().reshape(-1).numpy()
            frame["best_positive"] = best_positive.detach().cpu().reshape(-1).numpy().astype(bool)
            frame["best_penalty"] = [penalty_names[i] for i in best_idx.detach().cpu().reshape(-1).numpy()]
            frame["selected_penalty"] = [penalty_names[i] for i in selected_idx.detach().cpu().reshape(-1).numpy()]
            frame["selected_prob"] = selected_prob.detach().cpu().reshape(-1).numpy()
            frame["selected_gain"] = selected_gain.detach().cpu().reshape(-1).numpy()
            frame["selected_positive"] = (selected_gain.detach().cpu().reshape(-1).numpy() > float(eps))
            frame["gate_top1_hits_oracle"] = (selected_idx == best_idx).detach().cpu().reshape(-1).numpy().astype(bool)
            for p, name in enumerate(penalty_names):
                frame[f"mse_if_{name}"] = mse_bcp[..., p].detach().cpu().reshape(-1).numpy()
                frame[f"gain_if_{name}"] = (base_mse - mse_bcp[..., p]).detach().cpu().reshape(-1).numpy()
            all_frames.append(frame)
    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames, ignore_index=True)


def _summarize(df: pd.DataFrame, penalty_names: List[str]) -> Dict[str, object]:
    if df.empty:
        return {}
    base_mse = float(df["base_mse"].mean())
    gate_mse = float(df["gate_mse"].mean())
    oracle_mse = float(np.minimum(df["base_mse"].to_numpy(), df["best_mse"].to_numpy()).mean())
    selected_penalty_mse = float((df["base_mse"] - np.maximum(df["selected_gain"], 0.0)).mean())
    positive = df["best_positive"].to_numpy(dtype=bool)
    return {
        "rows": int(len(df)),
        "base_mse": base_mse,
        "gate_mse": gate_mse,
        "oracle_mse": oracle_mse,
        "selected_penalty_mse_if_skip_negative": selected_penalty_mse,
        "gate_gain_pct": 100.0 * (base_mse - gate_mse) / max(base_mse, 1.0e-12),
        "oracle_gain_pct": 100.0 * (base_mse - oracle_mse) / max(base_mse, 1.0e-12),
        "selected_penalty_gain_pct_if_skip_negative": 100.0 * (base_mse - selected_penalty_mse) / max(base_mse, 1.0e-12),
        "oracle_positive_rate": float(positive.mean()),
        "gate_positive_precision": float(df["selected_positive"].mean()),
        "gate_top1_hit_rate_all": float(df["gate_top1_hits_oracle"].mean()),
        "gate_top1_hit_rate_on_positive_oracle": float(df.loc[df["best_positive"], "gate_top1_hits_oracle"].mean()) if bool(positive.any()) else None,
        "penalty_names": penalty_names,
    }


def _write_summaries(df: pd.DataFrame, penalty_names: List[str], out_dir: Path) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    sample_path = out_dir / "oracle_samples.csv"
    df.to_csv(sample_path, index=False)
    paths["samples"] = sample_path

    summary = _summarize(df, penalty_names)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["summary"] = summary_path

    penalty_rows = []
    for name in penalty_names:
        sub = df[df["best_penalty"] == name]
        penalty_rows.append(
            {
                "penalty": name,
                "oracle_count": int(len(sub)),
                "oracle_share": float(len(sub) / max(len(df), 1)),
                "oracle_positive_count": int((sub["best_positive"]).sum()) if len(sub) else 0,
                "avg_best_gain_when_oracle": float(sub["best_gain"].mean()) if len(sub) else np.nan,
                "avg_gain_if_forced": float(df[f"gain_if_{name}"].mean()),
                "positive_rate_if_forced": float((df[f"gain_if_{name}"] > 0.0).mean()),
                "gate_selected_count": int((df["selected_penalty"] == name).sum()),
                "gate_selected_share": float((df["selected_penalty"] == name).mean()),
            }
        )
    penalty_df = pd.DataFrame(penalty_rows)
    penalty_path = out_dir / "penalty_oracle_summary.csv"
    penalty_df.to_csv(penalty_path, index=False)
    paths["penalty_summary"] = penalty_path

    confusion = pd.crosstab(df["best_penalty"], df["selected_penalty"], normalize="index")
    confusion = confusion.reindex(index=penalty_names, columns=penalty_names, fill_value=0.0)
    confusion_path = out_dir / "gate_vs_oracle_confusion.csv"
    confusion.to_csv(confusion_path)
    paths["confusion"] = confusion_path

    feature_cols = [c for c in df.columns if c.startswith("feat_") or c.startswith("base_")]
    bin_rows = []
    for col in feature_cols:
        values = df[col].replace([np.inf, -np.inf], np.nan)
        if values.nunique(dropna=True) < 4:
            continue
        try:
            bins = pd.qcut(values, q=4, duplicates="drop")
        except ValueError:
            continue
        tmp = df.assign(_bin=bins)
        for bin_name, sub in tmp.groupby("_bin", observed=True):
            row = {
                "feature": col,
                "bin": str(bin_name),
                "rows": int(len(sub)),
                "oracle_positive_rate": float(sub["best_positive"].mean()),
                "mean_best_gain": float(sub["best_gain"].mean()),
                "dominant_oracle_penalty": str(sub["best_penalty"].value_counts().idxmax()),
                "dominant_share": float(sub["best_penalty"].value_counts(normalize=True).max()),
            }
            for name in penalty_names:
                row[f"share_{name}"] = float((sub["best_penalty"] == name).mean())
            bin_rows.append(row)
    feature_bins = pd.DataFrame(bin_rows)
    feature_bins_path = out_dir / "feature_penalty_bins.csv"
    feature_bins.to_csv(feature_bins_path, index=False)
    paths["feature_bins"] = feature_bins_path

    rules = []
    global_gain = float(df["best_gain"].mean())
    for col in feature_cols:
        values = df[col].replace([np.inf, -np.inf], np.nan)
        qs = values.quantile([0.25, 0.5, 0.75]).dropna().unique()
        for threshold in qs:
            for op in [">=", "<="]:
                mask = values >= threshold if op == ">=" else values <= threshold
                sub = df[mask.fillna(False)]
                if len(sub) < max(50, int(0.02 * len(df))):
                    continue
                rules.append(
                    {
                        "feature": col,
                        "op": op,
                        "threshold": float(threshold),
                        "coverage": float(len(sub) / max(len(df), 1)),
                        "mean_best_gain": float(sub["best_gain"].mean()),
                        "gain_lift_vs_global": float(sub["best_gain"].mean() - global_gain),
                        "oracle_positive_rate": float(sub["best_positive"].mean()),
                        "dominant_oracle_penalty": str(sub["best_penalty"].value_counts().idxmax()),
                        "dominant_share": float(sub["best_penalty"].value_counts(normalize=True).max()),
                    }
                )
    rules_df = pd.DataFrame(rules).sort_values(["gain_lift_vs_global", "oracle_positive_rate"], ascending=False)
    rules_path = out_dir / "rule_candidates.csv"
    rules_df.to_csv(rules_path, index=False)
    paths["rules"] = rules_path

    try:
        fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=160)
        im = ax.imshow(confusion.to_numpy(), cmap="Blues", vmin=0.0, vmax=max(0.01, float(confusion.to_numpy().max())))
        ax.set_xticks(range(len(penalty_names)), penalty_names, rotation=35, ha="right")
        ax.set_yticks(range(len(penalty_names)), penalty_names)
        ax.set_xlabel("Gate selected")
        ax.set_ylabel("Oracle best")
        ax.set_title("Gate vs Oracle Penalty")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig_path = out_dir / "gate_vs_oracle_confusion.png"
        fig.savefig(fig_path)
        plt.close(fig)
        paths["confusion_png"] = fig_path
    except Exception:
        pass
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML used to train the checkpoint.")
    ap.add_argument("--checkpoint", default=None, help="Defaults to exp.out_dir/best_checkpoint.pt.")
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--eps", type=float, default=0.0)
    args = ap.parse_args()

    cfg_path = _resolve(args.config)
    cfg = _load_yaml(cfg_path)
    device = torch.device(args.device or cfg.get("exp", {}).get("device", "cuda:0"))
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    checkpoint_path = _resolve(args.checkpoint) if args.checkpoint else _resolve(Path(cfg["exp"]["out_dir"]) / "best_checkpoint.pt")
    out_dir = _resolve(args.out_dir) if args.out_dir else _resolve(Path(cfg["exp"]["out_dir"]) / f"penalty_gate_mining_{args.split}")
    _safe_mkdir(out_dir)

    _, channel_names, _, norm_tc, cluster_id_c_fit, t_train, t_val, _ = _normalize_and_cluster(cfg)
    x_split, y_split, split_start = _make_split_windows(cfg, norm_tc, args.split, t_train, t_val)
    modules = _build_modules(cfg, checkpoint_path, device)
    ckpt_cluster = modules["cluster_id_c"].detach().cpu()
    if ckpt_cluster.shape == cluster_id_c_fit.shape and not torch.equal(ckpt_cluster, cluster_id_c_fit):
        print("[warn] checkpoint cluster_id_c differs from config-refit cluster_id_c; using checkpoint cluster assignment.")
    batch_size = int(args.batch_size or cfg.get("train", {}).get("batch_size", 64))
    print(
        f"Mining split={args.split}, windows={len(x_split)}, channels={len(channel_names)}, "
        f"penalties={modules['penalty_names']}, checkpoint={checkpoint_path}"
    )
    df = _collect_oracle_rows(
        cfg=cfg,
        modules=modules,
        x_split=x_split,
        y_split=y_split,
        channel_names=channel_names,
        split_start_offset=split_start,
        batch_size=batch_size,
        device=device,
        max_windows=int(args.max_windows),
        eps=float(args.eps),
    )
    paths = _write_summaries(df, modules["penalty_names"], out_dir)
    summary = _summarize(df, modules["penalty_names"])
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
