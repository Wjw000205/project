from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.windows import WindowTensorDataset, make_label_range_windows, make_strict_windows
from src.models.cluster_predictor import build_cluster_predictor
from src.models.penalties import build_penalty_bank
from src.utils.cluster_memory import load_cluster_checkpoint


DEFAULT_PENALTIES = (
    "amp_under",
    "level",
    "delta",
    "diff_amp",
    "direction",
    "d2_match",
    "corr",
    "range",
    "trend",
    "jump",
    "seasonal_align",
)


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_data(cfg: dict[str, Any]) -> tuple[list[str], torch.Tensor, torch.Tensor, int, int]:
    csv_path = resolve(str(cfg["data"]["csv_path"]))
    date_col = int(cfg["data"].get("date_col", 0))
    raw_df = pd.read_csv(csv_path)
    max_rows = int(cfg.get("data", {}).get("max_rows", 0) or 0)
    if max_rows > 0:
        raw_df = raw_df.iloc[:max_rows].copy()
    value_cols = [col for i, col in enumerate(raw_df.columns) if i != date_col]
    raw_tc = torch.tensor(raw_df[value_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
    T = int(raw_tc.shape[0])
    train_ratio = float(cfg["data"].get("train_ratio", 0.7))
    val_ratio = float(cfg["data"].get("val_ratio", 0.1))
    t_train = int(T * train_ratio)
    t_val = int(T * (train_ratio + val_ratio))
    norm_tc = raw_tc.clone()
    norm_cfg = cfg.get("normalize", {}) or {}
    if bool(norm_cfg.get("global_zscore", False)):
        train_tc = norm_tc[:t_train]
        mean_c = train_tc.mean(dim=0)
        std_c = train_tc.std(dim=0).clamp_min(1.0e-6)
        norm_tc = (norm_tc - mean_c.view(1, -1)) / std_c.view(1, -1)
    return value_cols, raw_tc, norm_tc, t_train, t_val


def split_windows(
    cfg: dict[str, Any],
    norm_tc: torch.Tensor,
    *,
    split: str,
    t_train: int,
    t_val: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    L = int(cfg["window"]["input_len"])
    H = int(cfg["window"]["pred_len"])
    if split == "train":
        return make_strict_windows(norm_tc, L, H, 0, t_train)
    if split == "val":
        if bool(cfg.get("window", {}).get("past_context", False)):
            x, y, _ = make_label_range_windows(norm_tc, L, H, t_train, t_val)
            return x, y
        return make_strict_windows(norm_tc, L, H, t_train, t_val)
    raise ValueError("Only train and val are allowed. Test is intentionally unavailable for pool selection.")


def load_backbone_model(cfg: dict[str, Any], checkpoint: Path, device: torch.device):
    ckpt = load_cluster_checkpoint(str(checkpoint), device=device)
    meta = ckpt.get("meta", {}) or {}
    if not meta:
        raise ValueError(f"Checkpoint has no meta: {checkpoint}")
    cluster_id_c = meta["cluster_id_c"].to(device=device, dtype=torch.long)
    model = build_cluster_predictor(
        num_clusters=int(meta["K"]),
        input_len=int(meta["input_len"]),
        pred_len=int(meta["pred_len"]),
        model_cfg=meta.get("model_cfg", cfg.get("model", {})),
        num_channels=int(meta.get("num_channels", int(cluster_id_c.numel()))),
        cluster_id_c=cluster_id_c,
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, cluster_id_c, meta


@torch.no_grad()
def train_penalty_scale(
    model,
    source_cluster_id_c: torch.Tensor,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    penalty_names: list[str],
    penalty_fns: dict[str, Any],
    *,
    batch_size: int,
    device: torch.device,
    floor: float,
) -> np.ndarray:
    loader = DataLoader(WindowTensorDataset(x_train, y_train), batch_size=batch_size, shuffle=False, num_workers=0)
    sum_all = torch.zeros(len(penalty_names), device=device)
    sum_pos = torch.zeros(len(penalty_names), device=device)
    count_all = 0
    count_pos = torch.zeros(len(penalty_names), device=device)
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        y_base = model(x, source_cluster_id_c)
        pen = torch.stack([penalty_fns[name](y_base, y) for name in penalty_names], dim=-1)
        flat = pen.reshape(-1, len(penalty_names))
        sum_all += flat.sum(dim=0)
        count_all += int(flat.shape[0])
        pos = flat > 0
        sum_pos += (flat * pos).sum(dim=0)
        count_pos += pos.sum(dim=0)
    mean_all = sum_all / max(count_all, 1)
    mean_pos = sum_pos / count_pos.clamp_min(1.0)
    scale = torch.where(count_pos > 0, mean_pos, mean_all).clamp_min(float(floor))
    return scale.detach().cpu().numpy().astype(np.float64)


@torch.no_grad()
def val_residual_profile(
    model,
    source_cluster_id_c: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    penalty_names: list[str],
    penalty_fns: dict[str, Any],
    scale: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    max_windows: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_windows > 0:
        x_val = x_val[:max_windows]
        y_val = y_val[:max_windows]
    C = int(x_val.shape[1])
    P = len(penalty_names)
    sum_pen_cp = np.zeros((C, P), dtype=np.float64)
    sum_mse_c = np.zeros(C, dtype=np.float64)
    count = 0
    loader = DataLoader(WindowTensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False, num_workers=0)
    scale_t = torch.tensor(scale, dtype=torch.float32, device=device).view(1, 1, P)
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        y_base = model(x, source_cluster_id_c)
        base_mse = (y_base - y).pow(2).mean(dim=-1)
        pen = torch.stack([penalty_fns[name](y_base, y) for name in penalty_names], dim=-1)
        norm_pen = pen / scale_t.clamp_min(1.0e-8)
        sum_pen_cp += norm_pen.sum(dim=0).detach().cpu().numpy()
        sum_mse_c += base_mse.sum(dim=0).detach().cpu().numpy()
        count += int(x.shape[0])
    if count <= 0:
        raise ValueError("No validation windows available for residual profiling.")
    return sum_pen_cp / float(count), sum_mse_c / float(count)


def standardize_features(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return (x - mean) / np.maximum(std, 1.0e-8)


def remap_cluster_labels(labels: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    raw = np.asarray(labels, dtype=np.int64).reshape(-1)
    mapping = {int(label): idx for idx, label in enumerate(sorted(set(int(v) for v in raw.tolist())))}
    remapped = np.asarray([mapping[int(v)] for v in raw.tolist()], dtype=np.int64)
    return remapped, mapping


def select_allowed_penalties(
    scores: np.ndarray,
    penalty_names: list[str],
    *,
    topk: int,
    min_score: float,
    keep_ratio: float,
) -> list[str]:
    order = np.argsort(-scores)
    if order.size == 0:
        return []
    best = float(scores[order[0]])
    selected: list[str] = []
    for idx in order:
        score = float(scores[idx])
        if len(selected) >= int(topk):
            break
        if score < float(min_score) and score < best * float(keep_ratio):
            continue
        selected.append(penalty_names[int(idx)])
    if not selected:
        selected.append(penalty_names[int(order[0])])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Choose penalty pools from validation base-residual error modes.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default="outputs/input96_mse_gate_cluster_moe_retrain_20260616_pems/residual_penalty_profiles")
    parser.add_argument("--penalties", default=",".join(DEFAULT_PENALTIES))
    parser.add_argument("--n-clusters", type=int, default=4)
    parser.add_argument("--cluster-source", choices=["checkpoint", "residual_kmeans"], default="checkpoint")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.75)
    parser.add_argument("--keep-ratio", type=float, default=0.75)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-val-windows", type=int, default=0)
    parser.add_argument("--scale-floor", type=float, default=1.0e-6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    cfg_path = resolve(args.config)
    cfg = read_yaml(cfg_path)
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    penalty_names = [item.strip() for item in str(args.penalties).split(",") if item.strip()]
    if any(name in {"smooth", "jitter"} for name in penalty_names):
        raise ValueError("smooth/jitter are excluded: they are one-sided flattening regularizers.")

    channel_names, _, norm_tc, t_train, t_val = load_data(cfg)
    x_train, y_train = split_windows(cfg, norm_tc, split="train", t_train=t_train, t_val=t_val)
    x_val, y_val = split_windows(cfg, norm_tc, split="val", t_train=t_train, t_val=t_val)
    model, source_cluster_id_c, meta = load_backbone_model(cfg, resolve(args.checkpoint), device)
    if int(meta["input_len"]) != int(cfg["window"]["input_len"]) or int(meta["pred_len"]) != int(cfg["window"]["pred_len"]):
        raise ValueError("Checkpoint input/pred length does not match config.")

    penalty_fns = build_penalty_bank(penalty_names, jump_thr=float(cfg.get("penalties", {}).get("jump_threshold", 0.6)))
    scale = train_penalty_scale(
        model,
        source_cluster_id_c,
        x_train,
        y_train,
        penalty_names,
        penalty_fns,
        batch_size=int(args.batch_size),
        device=device,
        floor=float(args.scale_floor),
    )
    residual_cp, base_mse_c = val_residual_profile(
        model,
        source_cluster_id_c,
        x_val,
        y_val,
        penalty_names,
        penalty_fns,
        scale,
        batch_size=int(args.batch_size),
        device=device,
        max_windows=int(args.max_val_windows),
    )

    if str(args.cluster_source) == "checkpoint":
        labels, cluster_remap = remap_cluster_labels(source_cluster_id_c.detach().cpu().numpy())
        n_clusters = int(labels.max()) + 1
    else:
        features = standardize_features(np.concatenate([residual_cp, base_mse_c[:, None]], axis=1))
        n_clusters = max(1, min(int(args.n_clusters), residual_cp.shape[0]))
        labels = KMeans(n_clusters=n_clusters, n_init=20, random_state=2026).fit_predict(features).astype(int)
        cluster_remap = {}

    channel_rows = []
    for c, name in enumerate(channel_names):
        row = {"channel": name, "channel_idx": c, "cluster_id": int(labels[c]), "val_base_mse": float(base_mse_c[c])}
        for p, penalty in enumerate(penalty_names):
            row[f"val_norm_{penalty}"] = float(residual_cp[c, p])
        channel_rows.append(row)
    channel_df = pd.DataFrame(channel_rows)

    profile_rows = []
    allowed_by_cluster: list[list[str]] = []
    for k in range(n_clusters):
        mask = labels == k
        mean_scores = residual_cp[mask].mean(axis=0)
        allowed = select_allowed_penalties(
            mean_scores,
            penalty_names,
            topk=int(args.topk),
            min_score=float(args.min_score),
            keep_ratio=float(args.keep_ratio),
        )
        allowed_by_cluster.append(allowed)
        order = np.argsort(-mean_scores)
        row = {
            "cluster_id": k,
            "channels": int(mask.sum()),
            "val_base_mse": float(base_mse_c[mask].mean()),
            "recommended_penalties": ";".join(allowed),
            "top_penalty_1": penalty_names[int(order[0])],
            "top_score_1": float(mean_scores[int(order[0])]),
            "top_penalty_2": penalty_names[int(order[1])] if len(order) > 1 else "",
            "top_score_2": float(mean_scores[int(order[1])]) if len(order) > 1 else "",
            "top_penalty_3": penalty_names[int(order[2])] if len(order) > 2 else "",
            "top_score_3": float(mean_scores[int(order[2])]) if len(order) > 2 else "",
        }
        for p, penalty in enumerate(penalty_names):
            row[f"val_norm_{penalty}"] = float(mean_scores[p])
        profile_rows.append(row)
    profile_df = pd.DataFrame(profile_rows)

    union_penalties = [name for name in penalty_names if any(name in allowed for allowed in allowed_by_cluster)]
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = str(args.tag or f"{Path(args.config).stem}_residual_k{n_clusters}")
    profile_path = out_dir / f"{tag}_cluster_profile.csv"
    channel_path = out_dir / f"{tag}_channel_profile.csv"
    json_path = out_dir / f"{tag}_allowed_by_cluster.json"
    profile_df.to_csv(profile_path, index=False, encoding="utf-8-sig")
    channel_df.to_csv(channel_path, index=False, encoding="utf-8-sig")
    payload = {
        "test_used": False,
        "selection_split": "val",
        "scale_split": "train",
        "selection_target": "base_residual_penalty_modes",
        "config": str(cfg_path),
        "checkpoint": str(resolve(args.checkpoint)),
        "input_len": int(cfg["window"]["input_len"]),
        "pred_len": int(cfg["window"]["pred_len"]),
        "n_clusters": int(n_clusters),
        "cluster_source": str(args.cluster_source),
        "checkpoint_cluster_remap": {str(k): int(v) for k, v in cluster_remap.items()},
        "penalty_names_considered": penalty_names,
        "penalties_enabled": union_penalties,
        "allowed_by_cluster": allowed_by_cluster,
        "fixed_cluster_id": [int(v) for v in labels.tolist()],
        "cluster_profile_csv": str(profile_path),
        "channel_profile_csv": str(channel_path),
        "scale_by_penalty": {name: float(scale[i]) for i, name in enumerate(penalty_names)},
        "notes": (
            "Penalty pools are selected from validation base residual error modes. "
            "The test split is not evaluated or inspected by this script."
        ),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(profile_df[["cluster_id", "channels", "val_base_mse", "recommended_penalties", "top_penalty_1", "top_score_1", "top_penalty_2", "top_score_2", "top_penalty_3", "top_score_3"]].to_string(index=False))
    print(f"profile_csv={profile_path}")
    print(f"channel_profile_csv={channel_path}")
    print(f"allowed_by_cluster_json={json_path}")


if __name__ == "__main__":
    main()
