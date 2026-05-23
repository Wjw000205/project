from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def deep_set_paths(cfg: dict[str, Any], out_dir: Path, name: str) -> None:
    cfg.setdefault("exp", {})["name"] = name
    cfg.setdefault("exp", {})["out_dir"] = str(out_dir)
    cfg.setdefault("corr", {})["save_path"] = str(out_dir / "corr.npy")
    cfg.setdefault("portrait", {})["out_dir"] = str(out_dir / "cluster_portraits")
    cfg.setdefault("knn_hybrid", {})["path"] = str(out_dir / "knn_shape_bank.pt")
    cfg.setdefault("memory", {})["path"] = str(out_dir / "cluster_memory.pt")
    cfg.setdefault("memory", {})["checkpoint_path"] = str(out_dir / "best_checkpoint.pt")


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def make_config(base_cfg: dict[str, Any], input_len: int, out_root: Path, device: str | None, sample_count: int) -> Path:
    cfg = json.loads(json.dumps(base_cfg))
    name = f"ETTh2_h96_input{input_len}_diag"
    out_dir = out_root / f"input{input_len}"
    cfg.setdefault("window", {})["input_len"] = int(input_len)
    cfg.setdefault("window", {})["pred_len"] = 96
    cfg.setdefault("eval", {})["skip_test"] = False
    cfg.setdefault("plot", {})["enable"] = False
    cfg.setdefault("portrait", {})["enable"] = False
    cfg.setdefault("knn_hybrid", {})["enable"] = False
    cfg.setdefault("calibration", {})["enable"] = False
    cfg.setdefault("memory", {})["enable"] = False
    cfg.setdefault("memory", {})["save_checkpoint"] = False
    cfg.setdefault("diagnostics", {})["save_prediction_intermediates"] = True
    cfg.setdefault("diagnostics", {})["prediction_sample_count"] = int(sample_count)
    if device:
        cfg.setdefault("exp", {})["device"] = str(device)
    deep_set_paths(cfg, out_dir, name)
    config_path = out_root / "configs" / f"input{input_len}.yaml"
    write_yaml(config_path, cfg)
    return config_path


def run_train(config_path: Path) -> int:
    out_dir = Path(load_yaml(config_path).get("exp", {}).get("out_dir"))
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("MOELOSS_PROGRESS_LEAVE", "0")
    with (out_dir / "stdout.log").open("w", encoding="utf-8") as stdout_f, (out_dir / "stderr.log").open("w", encoding="utf-8") as stderr_f:
        proc = subprocess.run(
            [sys.executable, "-m", "src.train", "--config", str(config_path)],
            cwd=str(REPO_ROOT),
            text=True,
            stdout=stdout_f,
            stderr=stderr_f,
            env=env,
        )
    return int(proc.returncode)


def read_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_channel_metrics(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "test_metrics.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def scalar_mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def scalar_mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def summarize_intermediates(run_dir: Path, out_path: Path) -> None:
    npz_path = run_dir / "prediction_intermediates.npz"
    meta_path = run_dir / "prediction_intermediates_meta.json"
    if not npz_path.exists():
        return
    z = np.load(npz_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    channels = meta.get("channel_names") or [f"ch{i}" for i in range(z["y_true"].shape[1])]
    y = z["y_true"]
    base = z["y_base"]
    raw = z["y_residual_raw"]
    final = z["y_final"]
    scale = z["residual_gate_scale"] if "residual_gate_scale" in z.files else np.full((y.shape[0], y.shape[1], 1), np.nan)
    rows = []
    for c, name in enumerate(channels):
        rows.append(
            {
                "channel": name,
                "base_mse": scalar_mse(base[:, c, :], y[:, c, :]),
                "raw_residual_mse": scalar_mse(raw[:, c, :], y[:, c, :]),
                "final_mse": scalar_mse(final[:, c, :], y[:, c, :]),
                "base_mae": scalar_mae(base[:, c, :], y[:, c, :]),
                "raw_residual_mae": scalar_mae(raw[:, c, :], y[:, c, :]),
                "final_mae": scalar_mae(final[:, c, :], y[:, c, :]),
                "raw_delta_rms": float(np.sqrt(np.mean((raw[:, c, :] - base[:, c, :]) ** 2))),
                "final_delta_rms": float(np.sqrt(np.mean((final[:, c, :] - base[:, c, :]) ** 2))),
                "mean_gate_scale": float(np.nanmean(scale[:, c, :])),
            }
        )
    write_csv(
        out_path,
        rows,
        [
            "channel",
            "base_mse",
            "raw_residual_mse",
            "final_mse",
            "base_mae",
            "raw_residual_mae",
            "final_mae",
            "raw_delta_rms",
            "final_delta_rms",
            "mean_gate_scale",
        ],
    )


def compare(out_root: Path) -> None:
    runs = {input_len: out_root / f"input{input_len}" for input_len in [336, 96]}
    summaries = {k: read_summary(v) for k, v in runs.items()}
    rows = []
    for input_len, summary in summaries.items():
        if not summary:
            continue
        selection = summary.get("moe_residual_selection") or {}
        hit = (summary.get("moe_gate_penalty_hit") or {}).get("test") or {}
        rows.append(
            {
                "input_len": input_len,
                "test_mse": (summary.get("test") or {}).get("avg_mse"),
                "test_mae": (summary.get("test") or {}).get("avg_mae"),
                "val_mse": (summary.get("val") or {}).get("avg_mse"),
                "per_cluster_test_mse": json.dumps((summary.get("test") or {}).get("per_cluster_mse", []), ensure_ascii=False),
                "best_epoch": json.dumps(summary.get("best_epoch", [])),
                "num_train_windows": (summary.get("windowing") or {}).get("num_train_windows"),
                "residual_channels": ",".join(selection.get("residual_channels", []) or []),
                "val_base_mse": selection.get("val_pred_base_avg_mse"),
                "val_raw_residual_mse": selection.get("val_residual_avg_mse"),
                "val_scaled_mse": selection.get("val_scaled_avg_mse"),
                "gate_test_top1": hit.get("top1_hit_rate_all"),
                "gate_test_positive_top1": hit.get("top1_hit_rate_on_positive_oracle"),
                "gate_test_selected_gain_pct": hit.get("selected_top1_gain_pct_vs_base"),
            }
        )
    write_csv(
        out_root / "summary_comparison.csv",
        rows,
        [
            "input_len",
            "test_mse",
            "test_mae",
            "val_mse",
            "per_cluster_test_mse",
            "best_epoch",
            "num_train_windows",
            "residual_channels",
            "val_base_mse",
            "val_raw_residual_mse",
            "val_scaled_mse",
            "gate_test_top1",
            "gate_test_positive_top1",
            "gate_test_selected_gain_pct",
        ],
    )

    channel_rows = []
    metrics = {k: read_channel_metrics(v) for k, v in runs.items()}
    by_channel = {}
    for input_len, items in metrics.items():
        for item in items:
            by_channel.setdefault(item["channel"], {})[input_len] = item
    for channel, item_by_len in by_channel.items():
        row = {"channel": channel}
        for input_len in [336, 96]:
            item = item_by_len.get(input_len, {})
            row[f"mse_{input_len}"] = item.get("MSE", "")
            row[f"mae_{input_len}"] = item.get("MAE", "")
            row[f"cluster_{input_len}"] = item.get("cluster_id", "")
        try:
            row["mse_delta_96_minus_336"] = float(row["mse_96"]) - float(row["mse_336"])
        except Exception:
            row["mse_delta_96_minus_336"] = ""
        channel_rows.append(row)
    write_csv(
        out_root / "per_channel_comparison.csv",
        channel_rows,
        ["channel", "mse_336", "mse_96", "mse_delta_96_minus_336", "mae_336", "mae_96", "cluster_336", "cluster_96"],
    )

    for input_len, run_dir in runs.items():
        summarize_intermediates(run_dir, out_root / f"prediction_intermediate_summary_input{input_len}.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare ETTh2 H96 input_len=336 vs 96, including prediction intermediates.")
    ap.add_argument("--base-config", default="outputs/ett_horizon_sweep/configs/ETTh2_pred_96.yaml")
    ap.add_argument("--out-root", default="outputs/input_len_diagnostics/ETTh2_H96")
    ap.add_argument("--device", default=None)
    ap.add_argument("--sample-count", type=int, default=64)
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    base_cfg = load_yaml(Path(args.base_config).resolve())
    config_paths = [make_config(base_cfg, input_len, out_root, args.device, args.sample_count) for input_len in [336, 96]]
    if args.run:
        for config_path in config_paths:
            print(f"Running {config_path}", flush=True)
            rc = run_train(config_path)
            print(f"Return code {rc}: {config_path}", flush=True)
            if rc != 0:
                raise SystemExit(rc)
    compare(out_root)
    print(f"Wrote diagnostics to {out_root}", flush=True)


if __name__ == "__main__":
    main()
