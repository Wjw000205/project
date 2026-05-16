from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.cluster_memory import (  # noqa: E402
    compute_cluster_prototypes,
    load_cluster_checkpoint,
    save_cluster_memory,
)


TARGETS = ["ETTh1", "ETTh2", "ETTm2"]
HORIZONS = [192, 336, 720]

FIELDS = [
    "status",
    "source",
    "target",
    "pred_len",
    "input_len",
    "source_config",
    "source_checkpoint",
    "source_memory",
    "source_test_mse",
    "source_test_mae",
    "source_val_mse",
    "source_val_mae",
    "target_config",
    "target_self_test_mse",
    "target_self_test_mae",
    "target_self_val_mse",
    "target_self_val_mae",
    "data_max_rows",
    "train_ratio",
    "val_ratio",
    "test_ratio",
    "normalize_train_only",
    "past_context",
    "resample_enable",
    "direct_mse",
    "direct_mae",
    "direct_route_uses_train_only",
    "direct_num_windows",
    "direct_eval_start",
    "direct_eval_label_start",
    "direct_eval_end",
    "val_route_mse",
    "val_route_mae",
    "val_route_selected_val_mse",
    "val_route_selected_val_mae",
    "val_route",
    "val_route_uses_train_only",
    "search_mode",
    "out_dir",
    "error",
]


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})


def run_cmd(cmd: list[str]) -> str:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout)
    return proc.stdout


def source_config_path(horizon: int) -> Path:
    return ROOT / "outputs" / "ett_global_h96_param_base" / "configs" / f"ETTm1_pred_{horizon}.yaml"


def source_run_dir(horizon: int) -> Path:
    return ROOT / "outputs" / "ett_global_h96_param_base" / "runs" / "ETTm1" / f"pred_{horizon}"


def target_config_path(target: str, horizon: int) -> Path:
    return ROOT / "outputs" / "ett_horizon_sweep" / "configs" / f"{target}_pred_{horizon}.yaml"


def target_run_summary_path(target: str, horizon: int) -> Path:
    return ROOT / "outputs" / "ett_horizon_sweep" / "runs" / target / f"pred_{horizon}" / "run_summary.json"


def data_frame_to_tensor(cfg: dict[str, Any]) -> tuple[torch.Tensor, list[str]]:
    data_cfg = cfg["data"]
    df = pd.read_csv(data_cfg["csv_path"])
    max_rows = int(data_cfg.get("max_rows", 0) or 0)
    if max_rows > 0:
        df = df.iloc[:max_rows].copy()
    date_col = df.columns[int(data_cfg.get("date_col", 0))]
    value_cols = [c for c in df.columns if c != date_col]
    values = df[value_cols].to_numpy(dtype="float32")
    return torch.tensor(values, dtype=torch.float32), value_cols


def normalized_source_data(cfg: dict[str, Any]) -> torch.Tensor:
    data_tc, _ = data_frame_to_tensor(cfg)
    tr = float(cfg["data"]["train_ratio"])
    vr = float(cfg["data"]["val_ratio"])
    te = float(cfg["data"]["test_ratio"])
    if abs(tr + vr + te - 1.0) > 1.0e-6:
        raise ValueError("Source split ratios must sum to 1.")
    t_train = int(data_tc.shape[0] * tr)
    norm_cfg = cfg.get("normalize", {}) or {}
    if bool(norm_cfg.get("global_zscore", True)):
        fit = data_tc[:t_train] if bool(norm_cfg.get("train_only", True)) else data_tc
        mean = fit.mean(dim=0, keepdim=True)
        std = fit.std(dim=0, keepdim=True).clamp_min(1.0e-6)
        data_tc = (data_tc - mean) / std
    return data_tc


def ensure_source_memory(horizon: int, out_root: Path) -> tuple[Path, Path, Path]:
    cfg_path = source_config_path(horizon)
    run_dir = source_run_dir(horizon)
    checkpoint_path = run_dir / "best_checkpoint.pt"
    summary_path = run_dir / "run_summary.json"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing source checkpoint: {checkpoint_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing source summary: {summary_path}")

    out_dir = out_root / "source" / f"ETTm1_pred_{horizon}"
    memory_path = out_dir / "cluster_memory.pt"
    if memory_path.exists():
        return checkpoint_path, summary_path, memory_path
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = read_yaml(cfg_path)
    data_tc, channel_names = data_frame_to_tensor(cfg)
    norm_tc = normalized_source_data(cfg)
    t_train = int(norm_tc.shape[0] * float(cfg["data"]["train_ratio"]))
    ckpt = load_cluster_checkpoint(str(checkpoint_path), device=torch.device("cpu"))
    cluster_id_c = ckpt["meta"]["cluster_id_c"].to(torch.long)
    prototypes_kt = compute_cluster_prototypes(norm_tc[:t_train], cluster_id_c)
    save_cluster_memory(
        str(memory_path),
        prototypes_kt,
        cluster_id_c,
        channel_names,
        meta={
            "kind": "train_segment_prototype_synthesized",
            "source_split": "train",
            "memory_len": int(t_train),
            "input_len": int(cfg["window"]["input_len"]),
            "pred_len": int(cfg["window"]["pred_len"]),
            "source_config": str(cfg_path),
            "source_checkpoint": str(checkpoint_path),
            "num_window_updates": 0,
        },
    )
    return checkpoint_path, summary_path, memory_path


def build_transfer_config(
    *,
    horizon: int,
    target: str,
    checkpoint_path: Path,
    summary_path: Path,
    memory_path: Path,
    out_dir: Path,
    device: str,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_cfg = read_yaml(source_config_path(horizon))
    target_cfg = read_yaml(target_config_path(target, horizon))
    ckpt = load_cluster_checkpoint(str(checkpoint_path), device=torch.device("cpu"))
    input_len = int(ckpt["meta"]["input_len"])
    pred_len = int(ckpt["meta"]["pred_len"])
    if pred_len != int(target_cfg["window"]["pred_len"]):
        raise ValueError(f"Target config horizon mismatch: {target} H{horizon}")

    cfg = {
        "exp": {
            "name": f"ETTm1_to_{target}_H{horizon}",
            "out_dir": str(out_dir / "direct_transfer"),
            "seed": 2026,
            "device": device,
        },
        "source": {
            "memory_path": str(memory_path),
            "checkpoint_path": str(checkpoint_path),
            "summary_path": str(summary_path),
            "csv_path": source_cfg["data"]["csv_path"],
            "date_col": source_cfg["data"].get("date_col", 0),
            "step_minutes": 15,
        },
        "data": dict(target_cfg["data"]),
        "window": {
            "input_len": input_len,
            "pred_len": pred_len,
            "past_context": bool(target_cfg.get("window", {}).get("past_context", False)),
        },
        "normalize": dict(target_cfg.get("normalize", {"global_zscore": True, "train_only": True})),
        "transfer": {
            "corr_mode": "cycle_template",
            "route_fit_scope": "train",
            "use_pred_residual": True,
            "phase_bins": 64,
            "phase_max_shift": None,
            "period_min": None,
            "period_max": None,
            "period_min_hours": 12,
            "period_max_hours": 168,
            "corr_align": "head",
            "corr_threshold": None,
            "fallback_mode": "hard",
            "fallback_topk": 2,
            "fallback_temp": 1.0,
            "resample": {
                "enable": target in {"ETTh1", "ETTh2"},
                "target_step_minutes": 15,
                "method": "linear",
            },
            "knn_hybrid": {
                "enable": False,
                "scope": "same_cluster",
                "bank_split": "train",
                "use_for_model_selection": False,
                "k": 16,
                "alpha": 0.1,
                "adaptive_alpha": "confidence",
                "confidence_floor": 0.0,
                "distance_sharpness": 1.0,
                "shape_bins": 24,
                "diff_bins": 12,
                "bank_stride": 4,
                "distance_weight": "inverse",
                "anchor_mode": "last",
            },
            "save_corr": True,
        },
        "eval": {
            "batch_size": batch_size,
            "split": "test",
        },
    }
    meta = {
        "input_len": input_len,
        "pred_len": pred_len,
        "resample_enable": cfg["transfer"]["resample"]["enable"],
    }
    return cfg, meta


def run_pair(
    *,
    horizon: int,
    target: str,
    out_root: Path,
    device: str,
    py: str,
    batch_size: int,
) -> dict[str, Any]:
    checkpoint_path, summary_path, memory_path = ensure_source_memory(horizon, out_root)
    pair_dir = out_root / f"ETTm1_to_{target}" / f"pred_{horizon}"
    cfg, meta = build_transfer_config(
        horizon=horizon,
        target=target,
        checkpoint_path=checkpoint_path,
        summary_path=summary_path,
        memory_path=memory_path,
        out_dir=pair_dir,
        device=device,
        batch_size=batch_size,
    )
    cfg_path = pair_dir / "base_config.yaml"
    write_yaml(cfg_path, cfg)

    source_summary = load_json(summary_path)
    target_summary = load_json(target_run_summary_path(target, horizon))
    row: dict[str, Any] = {
        "status": "ok",
        "source": "ETTm1",
        "target": target,
        "pred_len": horizon,
        "input_len": meta["input_len"],
        "source_config": str(source_config_path(horizon)),
        "source_checkpoint": str(checkpoint_path),
        "source_memory": str(memory_path),
        "source_test_mse": source_summary.get("test", {}).get("avg_mse", ""),
        "source_test_mae": source_summary.get("test", {}).get("avg_mae", ""),
        "source_val_mse": source_summary.get("val", {}).get("avg_mse", ""),
        "source_val_mae": source_summary.get("val", {}).get("avg_mae", ""),
        "target_config": str(target_config_path(target, horizon)),
        "target_self_test_mse": target_summary.get("test", {}).get("avg_mse", ""),
        "target_self_test_mae": target_summary.get("test", {}).get("avg_mae", ""),
        "target_self_val_mse": target_summary.get("val", {}).get("avg_mse", ""),
        "target_self_val_mae": target_summary.get("val", {}).get("avg_mae", ""),
        "data_max_rows": cfg.get("data", {}).get("max_rows", 0),
        "train_ratio": cfg["data"]["train_ratio"],
        "val_ratio": cfg["data"]["val_ratio"],
        "test_ratio": cfg["data"]["test_ratio"],
        "normalize_train_only": cfg.get("normalize", {}).get("train_only", ""),
        "past_context": cfg.get("window", {}).get("past_context", False),
        "resample_enable": meta["resample_enable"],
        "out_dir": str(pair_dir),
    }

    run_cmd([py, "-u", "-m", "src.transfer", "--config", str(cfg_path)])
    direct = load_json(pair_dir / "direct_transfer" / "transfer_summary.json")
    row.update(
        {
            "direct_mse": direct["avg_mse"],
            "direct_mae": direct["avg_mae"],
            "direct_route_uses_train_only": direct.get("route_uses_train_only", ""),
            "direct_num_windows": direct.get("num_eval_windows", ""),
            "direct_eval_start": direct.get("eval_start_index", ""),
            "direct_eval_label_start": direct.get("eval_label_start_index", ""),
            "direct_eval_end": direct.get("eval_end_index", ""),
        }
    )

    selection_dir = pair_dir / "val_loss_selection"
    run_cmd(
        [
            py,
            "-u",
            "scripts/run_ettm1_to_ettm2_val_loss_route_selection.py",
            "--config",
            str(cfg_path),
            "--out-root",
            str(selection_dir),
            "--device",
            device,
            "--batch-size",
            str(batch_size),
            "--python",
            py,
            "--search-mode",
            "greedy",
        ]
    )
    selected = load_json(selection_dir / "summary.json")
    selected_test = load_json(selection_dir / "selected_test_transfer" / "transfer_summary.json")
    row.update(
        {
            "val_route_mse": selected["selected_test_mse"],
            "val_route_mae": selected["selected_test_mae"],
            "val_route_selected_val_mse": selected["selected_val_mse"],
            "val_route_selected_val_mae": selected["selected_val_mae"],
            "val_route": json.dumps(selected["selected_route"]),
            "val_route_uses_train_only": selected_test.get("route_uses_train_only", ""),
            "search_mode": selected.get("search_mode", ""),
        }
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "ettm1_other_ett_horizon_transfer")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--python", type=Path, default=Path(sys.executable))
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    py = str(args.python)
    for horizon in HORIZONS:
        for target in TARGETS:
            print(f"=== ETTm1 -> {target} H{horizon} ===", flush=True)
            try:
                row = run_pair(
                    horizon=horizon,
                    target=target,
                    out_root=args.out_root,
                    device=args.device,
                    py=py,
                    batch_size=args.batch_size,
                )
            except Exception as exc:
                row = {
                    "status": "error",
                    "source": "ETTm1",
                    "target": target,
                    "pred_len": horizon,
                    "out_dir": str(args.out_root / f"ETTm1_to_{target}" / f"pred_{horizon}"),
                    "error": str(exc)[-4000:],
                }
            rows.append(row)
            write_rows(args.out_root / "transfer.csv", rows)
    write_rows(args.out_root / "transfer.csv", rows)
    print(f"Saved: {args.out_root / 'transfer.csv'}")


if __name__ == "__main__":
    main()
