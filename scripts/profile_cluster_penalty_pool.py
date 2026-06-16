from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from src.train import compute_channel_shape_features
from src.utils.clustering import cluster_channels_by_corr
from src.utils.pearson import pearson_corr_matrix


ROOT = Path(__file__).resolve().parents[1]


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def autocorr_by_channel(x: np.ndarray, lag: int) -> np.ndarray:
    if lag <= 0 or x.shape[0] <= lag + 1:
        return np.zeros(x.shape[1], dtype=np.float64)
    a = x[:-lag]
    b = x[lag:]
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    denom = np.sqrt((a * a).mean(axis=0) * (b * b).mean(axis=0))
    return np.where(denom > 1.0e-8, (a * b).mean(axis=0) / denom, 0.0)


def channel_feature_frame(train: np.ndarray, raw_train: np.ndarray, channel_names: list[str]) -> pd.DataFrame:
    d1 = np.diff(train, axis=0)
    d2 = np.diff(train, n=2, axis=0)
    std = train.std(axis=0)
    std = np.maximum(std, 1.0e-8)
    diff_std = d1.std(axis=0)
    d2_std = d2.std(axis=0)
    jump_thr = np.abs(d1).mean(axis=0) + 2.0 * d1.std(axis=0)
    t = np.linspace(-1.0, 1.0, train.shape[0], dtype=np.float64).reshape(-1, 1)
    centered = train - train.mean(axis=0, keepdims=True)
    trend = np.abs((centered * t).mean(axis=0) / max(float((t * t).mean()), 1.0e-8))
    return pd.DataFrame(
        {
            "channel": channel_names,
            "daily_ac": autocorr_by_channel(train, 288),
            "halfday_ac": autocorr_by_channel(train, 144),
            "weekly_ac": autocorr_by_channel(train, 2016),
            "diff_std": diff_std / std,
            "d2_std": d2_std / std,
            "curvature": d2_std / np.maximum(diff_std, 1.0e-8),
            "range_over_std": (train.max(axis=0) - train.min(axis=0)) / std,
            "jump_rate": (np.abs(d1) > jump_thr).mean(axis=0),
            "zero_rate": (raw_train == 0.0).mean(axis=0),
            "trend_abs": trend,
        }
    )


def minmax(values: pd.Series) -> pd.Series:
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo <= 1.0e-12:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - lo) / (hi - lo)


def recommend_penalties(cluster_rows: pd.DataFrame, topk: int) -> list[str]:
    row = cluster_rows.iloc[0]
    scores = {
        "seasonal_align": max(float(row["daily_score"]), float(row["weekly_score"])),
        "corr": 0.65 * float(row["daily_score"]) + 0.35 * float(row["weekly_score"]),
        "level": 0.55 * float(row["stable_score"]) + 0.25 * float(row["daily_score"]) + 0.20 * float(row["level_score"]),
        "range": float(row["range_score"]),
        "amp_under": 0.50 * float(row["range_score"]) + 0.35 * float(row["diff_score"]) + 0.15 * float(row["daily_score"]),
        "delta": float(row["diff_score"]),
        "d2_match": 0.60 * float(row["d2_score"]) + 0.40 * float(row["curvature_score"]),
        "jump": max(float(row["jump_score"]), float(row["zero_score"])),
        "trend": float(row["trend_score"]),
    }
    blocked = {"smooth", "jitter"}
    ranked = [name for name, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True) if name not in blocked]

    selected: list[str] = []
    # Keep at least one seasonal/shape key on PEMS-like data when it is strong.
    if scores["seasonal_align"] >= 0.45:
        selected.append("seasonal_align")
    elif scores["corr"] >= 0.45:
        selected.append("corr")
    # Add one dynamics key.
    dynamics = max(("delta", "d2_match", "jump"), key=lambda name: scores[name])
    if scores[dynamics] >= 0.35:
        selected.append(dynamics)
    # Add one amplitude/level key.
    amp = max(("range", "amp_under", "level"), key=lambda name: scores[name])
    if scores[amp] >= 0.35:
        selected.append(amp)
    for name in ranked:
        if len(selected) >= topk:
            break
        if name not in selected and scores[name] >= 0.25:
            selected.append(name)
    if not selected:
        selected = ranked[:topk]
    return selected[:topk]


def main() -> None:
    ap = argparse.ArgumentParser(description="Profile channel clusters and recommend a penalty allowlist per cluster.")
    ap.add_argument("--dataset", default="PEMS03")
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--config", default=None)
    ap.add_argument("--method", default="kmeans")
    ap.add_argument("--n-clusters", type=int, default=4)
    ap.add_argument("--distance-threshold", type=float, default=None)
    ap.add_argument("--feature-aware", action="store_true")
    ap.add_argument("--feature-weight", type=float, default=0.75)
    ap.add_argument("--acf-lags", default="1,12,24,96,144,288,2016")
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--out-dir", default="outputs/input96_mse_gate_cluster_moe_retrain_20260616_pems/penalty_profiles")
    args = ap.parse_args()

    cfg_path = Path(args.config) if args.config else ROOT / "configs" / f"{args.dataset}_H{args.horizon}.yaml"
    cfg = read_yaml(cfg_path)
    csv_path = ROOT / str(cfg.get("data", {}).get("csv_path", f"data/{args.dataset}.csv"))
    df = pd.read_csv(csv_path)
    date_col = cfg.get("data", {}).get("date_col", 0)
    values_df = df.drop(columns=[df.columns[int(date_col)]])
    channel_names = list(values_df.columns)
    raw = values_df.to_numpy(dtype=np.float64)
    train_ratio = float(cfg.get("data", {}).get("train_ratio", 0.7))
    train_end = int(raw.shape[0] * train_ratio)
    raw_train = raw[:train_end]
    mu = raw_train.mean(axis=0, keepdims=True)
    sigma = raw_train.std(axis=0, keepdims=True)
    sigma = np.where(sigma > 1.0e-8, sigma, 1.0)
    train = (raw_train - mu) / sigma

    data_tc = torch.tensor(train, dtype=torch.float32)
    corr_cc = pearson_corr_matrix(data_tc)
    extra = None
    if args.feature_aware:
        acf_lags = [int(x) for x in str(args.acf_lags).split(",") if str(x).strip()]
        extra = compute_channel_shape_features(data_tc, acf_lags=acf_lags)
    cluster_id_c, clusters = cluster_channels_by_corr(
        corr_cc=corr_cc,
        data_tc=data_tc,
        n_clusters=int(args.n_clusters) if args.n_clusters > 0 else None,
        distance_threshold=args.distance_threshold,
        method=args.method,
        kmeans_n_init=10,
        kmeans_max_iter=300,
        spectral_affinity="corr",
        random_state=2026,
        min_cluster_size=2,
        merge_small_clusters=True,
        no_merge_if_channels_lt=7,
        extra_features_cf=extra,
        feature_weight=float(args.feature_weight) if extra is not None else 0.0,
    )
    labels = cluster_id_c.detach().cpu().numpy().astype(int)
    features = channel_feature_frame(train, raw_train, channel_names)
    features["cluster_id"] = labels

    agg = (
        features.groupby("cluster_id")
        .agg(
            channels=("channel", "count"),
            daily_ac=("daily_ac", "median"),
            weekly_ac=("weekly_ac", "median"),
            halfday_ac=("halfday_ac", "median"),
            diff_std=("diff_std", "median"),
            d2_std=("d2_std", "median"),
            curvature=("curvature", "median"),
            range_over_std=("range_over_std", "median"),
            jump_rate=("jump_rate", "median"),
            zero_rate=("zero_rate", "median"),
            trend_abs=("trend_abs", "median"),
        )
        .reset_index()
        .sort_values("cluster_id")
    )
    agg["daily_score"] = minmax(agg["daily_ac"])
    agg["weekly_score"] = minmax(agg["weekly_ac"])
    agg["diff_score"] = minmax(agg["diff_std"])
    agg["d2_score"] = minmax(agg["d2_std"])
    agg["curvature_score"] = minmax(agg["curvature"])
    agg["range_score"] = minmax(agg["range_over_std"])
    agg["jump_score"] = minmax(agg["jump_rate"])
    agg["zero_score"] = minmax(agg["zero_rate"])
    agg["trend_score"] = minmax(agg["trend_abs"])
    agg["level_score"] = minmax(agg["range_over_std"] - agg["diff_std"])
    agg["stable_score"] = 1.0 - minmax(agg["diff_std"] + agg["jump_rate"] * 4.0)

    allowlist: list[list[str]] = []
    for _, one in agg.iterrows():
        rec = recommend_penalties(pd.DataFrame([one]), int(args.topk))
        allowlist.append(rec)
    agg["recommended_penalties"] = [";".join(x) for x in allowlist]

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dataset}_H{args.horizon}_{args.method}_k{args.n_clusters}"
    if args.feature_aware:
        stem += f"_feat{args.feature_weight:g}"
    profile_path = out_dir / f"{stem}_cluster_profile.csv"
    channels_path = out_dir / f"{stem}_channel_features.csv"
    allow_path = out_dir / f"{stem}_allowed_by_cluster.json"
    agg.to_csv(profile_path, index=False, encoding="utf-8-sig")
    features.to_csv(channels_path, index=False, encoding="utf-8-sig")
    allow_payload = {
        "dataset": args.dataset,
        "horizon": int(args.horizon),
        "method": args.method,
        "n_clusters": int(args.n_clusters),
        "feature_aware": bool(args.feature_aware),
        "feature_weight": float(args.feature_weight) if args.feature_aware else 0.0,
        "allowed_by_cluster": allowlist,
        "cluster_sizes": {str(k): len(v) for k, v in clusters.items()},
        "profile_csv": str(profile_path),
        "channel_features_csv": str(channels_path),
        "notes": "smooth and jitter are intentionally excluded because they are one-sided flattening regularizers.",
    }
    allow_path.write_text(json.dumps(allow_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(agg[["cluster_id", "channels", "daily_ac", "weekly_ac", "diff_std", "d2_std", "range_over_std", "jump_rate", "zero_rate", "recommended_penalties"]].to_string(index=False))
    print(f"profile_csv={profile_path}")
    print(f"allowed_by_cluster_json={allow_path}")


if __name__ == "__main__":
    main()
