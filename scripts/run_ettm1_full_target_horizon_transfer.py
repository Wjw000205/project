from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_ettm1_other_ett_horizon_transfer import (  # noqa: E402
    data_frame_to_tensor,
    normalized_source_data,
    source_config_path,
    source_run_dir,
)
from src.utils.cluster_memory import (  # noqa: E402
    compute_cluster_prototypes,
    load_cluster_checkpoint,
    save_cluster_memory,
)


TARGETS = ["ETTh1", "ETTh2", "ETTm2", "weather", "traffic"]
HORIZONS = [96, 192, 336, 720]

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
    "target_pred_len_adjusted",
    "target_original_pred_len",
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
    "selected_policy",
    "selected_mse",
    "selected_mae",
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


def run_cmd(cmd: list[str]) -> None:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout[-5000:])


def h96_source_paths() -> tuple[Path, Path, Path]:
    cfg = read_yaml(ROOT / "configs" / "ETTm1ToETTm2.yaml")
    ckpt = Path(str(cfg["source"]["checkpoint_path"]))
    summary = Path(str(cfg["source"]["summary_path"]))
    memory = Path(str(cfg["source"]["memory_path"]))
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    if not summary.is_absolute():
        summary = ROOT / summary
    if not memory.is_absolute():
        memory = ROOT / memory
    return ckpt, summary, memory


def source_paths(horizon: int, out_root: Path) -> tuple[Path, Path, Path]:
    if horizon == 96:
        return h96_source_paths()
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


def source_cfg_path_for_horizon(horizon: int) -> Path:
    if horizon == 96:
        return ROOT / "outputs" / "ett_horizon_sweep" / "configs" / "ETTm1_pred_96.yaml"
    return source_config_path(horizon)


def target_cfg_path(target: str, horizon: int) -> Path:
    if target in {"ETTh1", "ETTh2", "ETTm2"}:
        if horizon == 96:
            # These are the final landed H96 configs for ETTh1/ETTh2/ETTm2.
            path = ROOT / "configs" / f"{target}.yaml"
            if path.exists():
                return path
        return ROOT / "outputs" / "ett_horizon_sweep" / "configs" / f"{target}_pred_{horizon}.yaml"
    return ROOT / "configs" / f"{target}.yaml"


def target_self_summary(target: str, horizon: int) -> dict[str, Any] | None:
    if target in {"ETTh1", "ETTh2", "ETTm2"}:
        if horizon == 96:
            path = ROOT / "outputs" / "ett_horizon_sweep" / "runs" / target / "pred_96" / "run_summary.json"
        else:
            path = ROOT / "outputs" / "ett_horizon_sweep" / "runs" / target / f"pred_{horizon}" / "run_summary.json"
        return load_json(path) if path.exists() else None
    return None


def existing_row(target: str, horizon: int) -> dict[str, Any] | None:
    if horizon == 96:
        path = ROOT / "outputs" / "aligned_h96_transfer_matrix" / "transfer.csv"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("source") == "ETTm1" and row.get("target") == target:
                    return normalize_existing_row(row, target, horizon)
        return None
    if target in {"ETTh1", "ETTh2", "ETTm2"}:
        path = ROOT / "outputs" / "ettm1_other_ett_horizon_transfer" / "transfer.csv"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("source") == "ETTm1" and row.get("target") == target and int(float(row.get("pred_len", -1))) == horizon:
                    return normalize_existing_row(row, target, horizon)
    return None


def normalize_existing_row(row: dict[str, Any], target: str, horizon: int) -> dict[str, Any]:
    out = {k: row.get(k, "") for k in FIELDS}
    out["status"] = row.get("status", "ok")
    out["source"] = "ETTm1"
    out["target"] = target
    out["pred_len"] = horizon
    out["input_len"] = row.get("input_len", 336)
    out["target_config"] = row.get("target_config", "")
    out["target_pred_len_adjusted"] = row.get("target_pred_len_adjusted", False)
    out["target_original_pred_len"] = row.get("target_original_pred_len", horizon)
    out["selected_policy"] = choose_policy(row)
    if out["selected_policy"] == "direct_train_only":
        out["selected_mse"] = row.get("direct_mse", "")
        out["selected_mae"] = row.get("direct_mae", "")
    else:
        out["selected_mse"] = row.get("val_route_mse", "")
        out["selected_mae"] = row.get("val_route_mae", "")
    if "target_self_test_mse" not in row or row.get("target_self_test_mse", "") == "":
        summary = target_self_summary(target, horizon)
        if summary is not None:
            out["target_self_test_mse"] = summary.get("test", {}).get("avg_mse", "")
            out["target_self_test_mae"] = summary.get("test", {}).get("avg_mae", "")
            out["target_self_val_mse"] = summary.get("val", {}).get("avg_mse", "")
            out["target_self_val_mae"] = summary.get("val", {}).get("avg_mae", "")
    return out


def choose_policy(row: dict[str, Any]) -> str:
    try:
        direct = float(row.get("direct_mse", "nan"))
        val = float(row.get("val_route_mse", "nan"))
    except Exception:
        return "val_route"
    if direct == direct and val == val and val > direct:
        return "direct_train_only"
    return "val_route"


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
    source_cfg_path = source_cfg_path_for_horizon(horizon)
    source_cfg = read_yaml(source_cfg_path)
    target_path = target_cfg_path(target, horizon)
    target_cfg = read_yaml(target_path)
    ckpt = load_cluster_checkpoint(str(checkpoint_path), device=torch.device("cpu"))
    input_len = int(ckpt["meta"]["input_len"])
    pred_len = int(ckpt["meta"]["pred_len"])
    if pred_len != int(horizon):
        raise ValueError(f"Source checkpoint horizon mismatch: expected H{horizon}, got H{pred_len}")

    data_cfg = dict(target_cfg["data"])
    window_cfg = dict(target_cfg.get("window", {}) or {})
    original_pred_len = int(window_cfg.get("pred_len", pred_len))

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
        "data": data_cfg,
        "window": {
            "input_len": input_len,
            "pred_len": pred_len,
            "past_context": bool(window_cfg.get("past_context", False)),
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
                "enable": target in {"ETTh1", "ETTh2", "weather", "traffic"},
                "target_step_minutes": 15,
                "method": "linear",
            },
            "knn_hybrid": {"enable": False},
            "save_corr": True,
        },
        "eval": {"batch_size": batch_size, "split": "test"},
    }
    meta = {
        "source_config": str(source_cfg_path),
        "target_config": str(target_path),
        "input_len": input_len,
        "pred_len": pred_len,
        "target_pred_len_adjusted": original_pred_len != pred_len,
        "target_original_pred_len": original_pred_len,
        "resample_enable": cfg["transfer"]["resample"]["enable"],
    }
    return cfg, meta


def run_pair(
    *,
    target: str,
    horizon: int,
    out_root: Path,
    device: str,
    py: str,
    batch_size: int,
    reuse_existing: bool,
) -> dict[str, Any]:
    if reuse_existing:
        existing = existing_row(target, horizon)
        if existing is not None:
            return existing

    checkpoint_path, summary_path, memory_path = source_paths(horizon, out_root)
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
    target_summary = target_self_summary(target, horizon) or {}
    row: dict[str, Any] = {
        "status": "ok",
        "source": "ETTm1",
        "target": target,
        "pred_len": horizon,
        "input_len": meta["input_len"],
        "source_config": meta["source_config"],
        "source_checkpoint": str(checkpoint_path),
        "source_memory": str(memory_path),
        "source_test_mse": source_summary.get("test", {}).get("avg_mse", ""),
        "source_test_mae": source_summary.get("test", {}).get("avg_mae", ""),
        "source_val_mse": source_summary.get("val", {}).get("avg_mse", ""),
        "source_val_mae": source_summary.get("val", {}).get("avg_mae", ""),
        "target_config": meta["target_config"],
        "target_self_test_mse": target_summary.get("test", {}).get("avg_mse", ""),
        "target_self_test_mae": target_summary.get("test", {}).get("avg_mae", ""),
        "target_self_val_mse": target_summary.get("val", {}).get("avg_mse", ""),
        "target_self_val_mae": target_summary.get("val", {}).get("avg_mae", ""),
        "target_pred_len_adjusted": meta["target_pred_len_adjusted"],
        "target_original_pred_len": meta["target_original_pred_len"],
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
            "auto",
            "--max-greedy-channels",
            "64",
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
    row["selected_policy"] = choose_policy(row)
    if row["selected_policy"] == "direct_train_only":
        row["selected_mse"] = row["direct_mse"]
        row["selected_mae"] = row["direct_mae"]
    else:
        row["selected_mse"] = row["val_route_mse"]
        row["selected_mae"] = row["val_route_mae"]
    return row


def plot_results(out_root: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception:
        return
    df = pd.DataFrame(rows)
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return
    ok["pred_len"] = ok["pred_len"].astype(int)
    ok["selected_mse"] = ok["selected_mse"].astype(float)
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for target, grp in ok.groupby("target"):
        grp = grp.sort_values("pred_len")
        ax.plot(grp["pred_len"], grp["selected_mse"], marker="o", label=target)
    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel("Selected transfer test MSE")
    ax.set_title("ETTm1 transfer across full targets and horizons")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_root / "full_target_horizon_transfer_mse.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "ettm1_full_target_horizon_transfer")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--python", type=Path, default=Path(sys.executable))
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--no-reuse-existing", action="store_true")
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        for target in TARGETS:
            print(f"=== ETTm1 -> {target} H{horizon} ===", flush=True)
            try:
                row = run_pair(
                    target=target,
                    horizon=horizon,
                    out_root=args.out_root,
                    device=args.device,
                    py=str(args.python),
                    batch_size=args.batch_size,
                    reuse_existing=not args.no_reuse_existing,
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
    plot_results(args.out_root, rows)
    print(args.out_root / "transfer.csv")


if __name__ == "__main__":
    main()
