from __future__ import annotations

import argparse
import copy
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

from src.utils.cluster_memory import compute_cluster_prototypes, load_cluster_checkpoint, save_cluster_memory  # noqa: E402


SOURCE = "ETTm1"
TARGETS = ["ETTh1", "ETTh2", "ETTm2"]
HORIZONS = [96, 192, 336, 720]
TARGET_STEP_MINUTES = 15

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
    "target_self_config",
    "target_self_test_mse",
    "target_self_test_mae",
    "zero_shot_mse",
    "zero_shot_mae",
    "zero_shot_route_uses_train_only",
    "zero_shot_cluster_id",
    "zero_shot_corr_mean",
    "finetune_lr",
    "finetune_epochs",
    "finetune_val_mse",
    "finetune_val_mae",
    "finetune_test_mse",
    "finetune_test_mae",
    "finetune_best_epoch",
    "finetune_vs_target_self_mse",
    "finetune_vs_target_self_mae",
    "finetune_gain_pct_vs_zero_shot",
    "target_self_gap_note",
    "config_path",
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


def run_cmd(cmd: list[str], log_path: Path | None = None) -> str:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout)
    return proc.stdout


def config_path(dataset: str, horizon: int) -> Path:
    path = ROOT / "configs" / f"{dataset}_H{horizon}.yaml"
    if path.exists():
        return path
    fallback = ROOT / "configs" / f"{dataset}.yaml"
    if not fallback.exists():
        raise FileNotFoundError(f"Missing config for {dataset} H{horizon}: {path}")
    return fallback


def best_results() -> dict[tuple[str, int], dict[str, Any]]:
    path = ROOT / "outputs" / "ett_horizon_specific_moe_tune" / "best_results.csv"
    out: dict[tuple[str, int], dict[str, Any]] = {}
    if not path.exists():
        return out
    for row in pd.read_csv(path).to_dict("records"):
        out[(str(row["dataset"]), int(row["horizon"]))] = row
    return out


def data_frame_to_tensor(cfg: dict[str, Any]) -> tuple[torch.Tensor, list[str]]:
    data_cfg = cfg["data"]
    df = pd.read_csv(ROOT / str(data_cfg["csv_path"]))
    max_rows = int(data_cfg.get("max_rows", 0) or 0)
    if max_rows > 0:
        df = df.iloc[:max_rows].copy()
    date_col = df.columns[int(data_cfg.get("date_col", 0))]
    value_cols = [c for c in df.columns if c != date_col]
    values = df[value_cols].to_numpy(dtype="float32")
    return torch.tensor(values, dtype=torch.float32), value_cols


def infer_step_minutes(df: pd.DataFrame, date_col: str) -> float:
    dt = pd.to_datetime(df[date_col])
    diffs = dt.diff().dropna()
    if diffs.empty:
        return 0.0
    step = diffs.mode().iloc[0]
    return float(step.total_seconds() / 60.0)


def resample_df(df: pd.DataFrame, date_col: str, target_step_min: int, method: str) -> pd.DataFrame:
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
    return out.reset_index()


def target_data_cfg_for_finetune(target_cfg: dict[str, Any], target: str, out_root: Path, resample_method: str) -> dict[str, Any]:
    data_cfg = copy.deepcopy(target_cfg["data"])
    if target not in {"ETTh1", "ETTh2"} or resample_method.lower() in {"none", "off", "false"}:
        return data_cfg
    raw_path = ROOT / str(data_cfg["csv_path"])
    raw_df = pd.read_csv(raw_path)
    max_rows = int(data_cfg.get("max_rows", 0) or 0)
    if max_rows > 0:
        raw_df = raw_df.iloc[:max_rows].copy()
    date_col = raw_df.columns[int(data_cfg.get("date_col", 0))]
    cur_step = infer_step_minutes(raw_df, date_col)
    if cur_step > 0 and int(round(cur_step)) == TARGET_STEP_MINUTES:
        return data_cfg
    prepared_path = out_root / "prepared_data" / f"{target}_{TARGET_STEP_MINUTES}min_{resample_method.lower()}.csv"
    if not prepared_path.exists():
        prepared_path.parent.mkdir(parents=True, exist_ok=True)
        resampled = resample_df(raw_df, date_col, TARGET_STEP_MINUTES, resample_method.lower())
        resampled.to_csv(prepared_path, index=False)
    data_cfg["csv_path"] = str(prepared_path)
    data_cfg["max_rows"] = 0
    return data_cfg


def normalized_train_data(cfg: dict[str, Any]) -> torch.Tensor:
    data_tc, _ = data_frame_to_tensor(cfg)
    t_train = int(data_tc.shape[0] * float(cfg["data"]["train_ratio"]))
    norm_cfg = cfg.get("normalize", {}) or {}
    if bool(norm_cfg.get("global_zscore", True)):
        fit = data_tc[:t_train] if bool(norm_cfg.get("train_only", True)) else data_tc
        mean = fit.mean(dim=0, keepdim=True)
        std = fit.std(dim=0, keepdim=True).clamp_min(1.0e-6)
        data_tc = (data_tc - mean) / std
    return data_tc[:t_train]


def source_run_dir(out_root: Path, horizon: int) -> Path:
    return out_root / "source" / f"{SOURCE}_H{horizon}"


def prepare_source_config(horizon: int, out_root: Path, device: str, source_epochs: int) -> Path:
    cfg = read_yaml(config_path(SOURCE, horizon))
    cfg = copy.deepcopy(cfg)
    out_dir = source_run_dir(out_root, horizon)
    cfg["exp"]["name"] = f"{SOURCE}_H{horizon}_current_transfer_source"
    cfg["exp"]["out_dir"] = str(out_dir)
    cfg["exp"]["device"] = device
    cfg["window"]["input_len"] = 336
    cfg["window"]["pred_len"] = int(horizon)
    cfg.setdefault("normalize", {})["train_only"] = True
    cfg.setdefault("cluster", {})["train_only"] = True
    cfg.setdefault("train", {})["epochs"] = int(source_epochs)
    cfg.setdefault("eval", {})["skip_test"] = False
    cfg["plot"] = {"enable": False}
    cfg["portrait"] = {"enable": False, "out_dir": str(out_dir / "cluster_portraits")}
    cfg.setdefault("knn_hybrid", {})["enable"] = False
    cfg["knn_hybrid"]["use_for_model_selection"] = False
    cfg["calibration"] = {"enable": False}
    cfg["memory"] = {
        "enable": False,
        "save_checkpoint": True,
        "path": str(out_dir / "cluster_memory.pt"),
        "checkpoint_path": str(out_dir / "best_checkpoint.pt"),
    }
    path = out_root / "configs" / "source" / f"{SOURCE}_H{horizon}_source.yaml"
    write_yaml(path, cfg)
    return path


def ensure_source(horizon: int, out_root: Path, device: str, py: str, source_epochs: int, rerun_source: bool) -> tuple[Path, Path, Path, dict[str, Any]]:
    cfg_path = prepare_source_config(horizon, out_root, device, source_epochs)
    cfg = read_yaml(cfg_path)
    out_dir = Path(cfg["exp"]["out_dir"])
    ckpt_path = out_dir / "best_checkpoint.pt"
    summary_path = out_dir / "run_summary.json"
    memory_path = out_dir / "cluster_memory.pt"
    if rerun_source or not ckpt_path.exists() or not summary_path.exists():
        print(f"[source] train {SOURCE} H{horizon}", flush=True)
        run_cmd([py, "-u", "-m", "src.train", "--config", str(cfg_path)], log_path=out_dir / "source_train.log")
    if not memory_path.exists():
        cfg = read_yaml(cfg_path)
        norm_train_tc = normalized_train_data(cfg)
        _, channel_names = data_frame_to_tensor(cfg)
        ckpt = load_cluster_checkpoint(str(ckpt_path), device=torch.device("cpu"))
        cluster_id_c = ckpt["meta"]["cluster_id_c"].to(torch.long)
        prototypes_kt = compute_cluster_prototypes(norm_train_tc, cluster_id_c)
        save_cluster_memory(
            str(memory_path),
            prototypes_kt,
            cluster_id_c,
            channel_names,
            meta={
                "kind": "current_source_train_prototype",
                "source_split": "train",
                "input_len": int(cfg["window"]["input_len"]),
                "pred_len": int(cfg["window"]["pred_len"]),
                "source_config": str(cfg_path),
                "source_checkpoint": str(ckpt_path),
            },
        )
    return cfg_path, ckpt_path, memory_path, load_json(summary_path)


def build_zero_config(
    *,
    source_cfg_path: Path,
    source_summary_path: Path,
    checkpoint_path: Path,
    memory_path: Path,
    target: str,
    horizon: int,
    out_root: Path,
    device: str,
    batch_size: int,
    resample_method: str,
) -> Path:
    source_cfg = read_yaml(source_cfg_path)
    target_cfg = read_yaml(config_path(target, horizon))
    out_dir = out_root / "zero_shot" / f"{SOURCE}_to_{target}" / f"pred_{horizon}"
    resample_enable = target in {"ETTh1", "ETTh2"} and resample_method.lower() not in {"none", "off", "false"}
    cfg = {
        "exp": {"name": f"{SOURCE}_to_{target}_H{horizon}_zero_shot", "out_dir": str(out_dir), "seed": 2026, "device": device},
        "source": {
            "memory_path": str(memory_path),
            "checkpoint_path": str(checkpoint_path),
            "summary_path": str(source_summary_path),
            "csv_path": source_cfg["data"]["csv_path"],
            "date_col": source_cfg["data"].get("date_col", 0),
            "step_minutes": TARGET_STEP_MINUTES,
        },
        "data": copy.deepcopy(target_cfg["data"]),
        "window": {"input_len": 336, "pred_len": int(horizon), "past_context": bool(target_cfg.get("window", {}).get("past_context", False))},
        "normalize": {"global_zscore": True, "train_only": True},
        "transfer": {
            "corr_mode": "cycle_template",
            "route_fit_scope": "train",
            "use_pred_residual": True,
            "phase_bins": 64,
            "period_min_hours": 12,
            "period_max_hours": 168,
            "corr_align": "head",
            "fallback_mode": "hard",
            "fallback_topk": 2,
            "fallback_temp": 1.0,
            "resample": {"enable": resample_enable, "target_step_minutes": TARGET_STEP_MINUTES, "method": resample_method.lower()},
            "knn_hybrid": {"enable": False, "scope": "same_cluster", "bank_split": "train", "use_for_model_selection": False},
            "save_corr": True,
        },
        "eval": {"batch_size": int(batch_size), "split": "test"},
    }
    path = out_root / "configs" / "zero_shot" / f"{SOURCE}_to_{target}_H{horizon}.yaml"
    write_yaml(path, cfg)
    return path


def build_finetune_config(
    *,
    source_cfg_path: Path,
    checkpoint_path: Path,
    memory_path: Path,
    fixed_cluster_id: list[int],
    target: str,
    horizon: int,
    out_root: Path,
    device: str,
    lr: float,
    epochs: int,
    batch_size: int,
    resample_method: str,
    load_gate: bool,
    load_dynamic_lambda: bool,
) -> Path:
    cfg = copy.deepcopy(read_yaml(source_cfg_path))
    target_cfg = read_yaml(config_path(target, horizon))
    out_dir = out_root / "runs" / f"{SOURCE}_to_{target}" / f"pred_{horizon}" / f"lr{lr:g}".replace(".", "p")
    data_cfg = target_data_cfg_for_finetune(target_cfg, target, out_root, resample_method)
    cfg["exp"] = {"name": f"{SOURCE}_to_{target}_H{horizon}_finetune", "out_dir": str(out_dir), "seed": 2026, "deterministic": True, "device": device}
    cfg["data"] = data_cfg
    cfg["window"]["input_len"] = 336
    cfg["window"]["pred_len"] = int(horizon)
    cfg["window"]["past_context"] = bool(target_cfg.get("window", {}).get("past_context", False))
    cfg["normalize"] = {"global_zscore": True, "train_only": True}
    cfg.setdefault("cluster", {})
    cfg["cluster"]["train_only"] = True
    cfg["cluster"]["fixed_cluster_id"] = [int(v) for v in fixed_cluster_id]
    cfg["corr"] = {"compute": True, "save_path": str(out_dir / "corr.npy")}
    cfg.setdefault("train", {})
    cfg["train"]["epochs"] = int(epochs)
    cfg["train"]["lr"] = float(lr)
    cfg["train"]["batch_size"] = int(batch_size)
    cfg["train"].setdefault("weight_decay", 0.0001)
    cfg["train"].setdefault("selection_metric", "val_mse")
    cfg.setdefault("early_stop", {})
    cfg["early_stop"]["patience"] = int(cfg["early_stop"].get("patience", 10))
    cfg["eval"] = {"skip_test": False}
    cfg["plot"] = {"enable": False}
    cfg["portrait"] = {"enable": False, "out_dir": str(out_dir / "cluster_portraits")}
    cfg.setdefault("knn_hybrid", {})["enable"] = False
    cfg["knn_hybrid"]["use_for_model_selection"] = False
    cfg["calibration"] = {"enable": False}
    cfg["memory"] = {
        "enable": False,
        "save_checkpoint": True,
        "path": str(out_dir / "cluster_memory.pt"),
        "checkpoint_path": str(out_dir / "best_checkpoint.pt"),
    }
    cfg["finetune"] = {
        "enable": True,
        "checkpoint_path": str(checkpoint_path),
        "memory_path": str(memory_path),
        "cluster_map": "index",
        "strict_window": True,
        "strict_model": True,
        "load_model": True,
        "load_gate": bool(load_gate),
        "load_dynamic_lambda": bool(load_dynamic_lambda),
        "load_learnable_lambda": True,
    }
    lr_tag = f"lr{lr:g}".replace(".", "p")
    path = out_root / "configs" / "finetune" / f"{SOURCE}_to_{target}_H{horizon}_{lr_tag}.yaml"
    write_yaml(path, cfg)
    return path


def run_one(args: argparse.Namespace, best: dict[tuple[str, int], dict[str, Any]], target: str, horizon: int, lr: float) -> dict[str, Any]:
    out_root: Path = args.out_root
    py = str(args.python)
    source_cfg, ckpt, memory, source_summary = ensure_source(horizon, out_root, args.device, py, args.source_epochs, args.rerun_source)
    source_summary_path = source_run_dir(out_root, horizon) / "run_summary.json"
    zero_cfg = build_zero_config(
        source_cfg_path=source_cfg,
        source_summary_path=source_summary_path,
        checkpoint_path=ckpt,
        memory_path=memory,
        target=target,
        horizon=horizon,
        out_root=out_root,
        device=args.device,
        batch_size=args.batch_size,
        resample_method=args.resample_method,
    )
    zero_out = Path(read_yaml(zero_cfg)["exp"]["out_dir"])
    zero_summary_path = zero_out / "transfer_summary.json"
    if args.rerun or not zero_summary_path.exists():
        print(f"[zero-shot] {SOURCE}->{target} H{horizon}", flush=True)
        run_cmd([py, "-u", "-m", "src.transfer", "--config", str(zero_cfg)], log_path=zero_out / "zero_shot.log")
    zero = load_json(zero_summary_path)
    fixed_cluster_id = [int(v) for v in zero["cluster_id"]]

    ft_cfg = build_finetune_config(
        source_cfg_path=source_cfg,
        checkpoint_path=ckpt,
        memory_path=memory,
        fixed_cluster_id=fixed_cluster_id,
        target=target,
        horizon=horizon,
        out_root=out_root,
        device=args.device,
        lr=lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        resample_method=args.resample_method,
        load_gate=not args.no_load_gate,
        load_dynamic_lambda=not args.no_load_dynamic_lambda,
    )
    ft_cfg_obj = read_yaml(ft_cfg)
    ft_out = Path(ft_cfg_obj["exp"]["out_dir"])
    ft_summary_path = ft_out / "run_summary.json"
    if args.rerun or not ft_summary_path.exists():
        print(f"[fine-tune] {SOURCE}->{target} H{horizon} lr={lr:g}", flush=True)
        run_cmd([py, "-u", "-m", "src.train", "--config", str(ft_cfg)], log_path=ft_out / "finetune.log")
    ft = load_json(ft_summary_path)

    target_best = best.get((target, horizon), {})
    t_mse = target_best.get("test_mse", "")
    t_mae = target_best.get("test_mae", "")
    ft_mse = ft.get("test", {}).get("avg_mse", "")
    ft_mae = ft.get("test", {}).get("avg_mae", "")
    row = {
        "status": "ok",
        "source": SOURCE,
        "target": target,
        "pred_len": horizon,
        "input_len": 336,
        "source_config": str(source_cfg),
        "source_checkpoint": str(ckpt),
        "source_memory": str(memory),
        "source_test_mse": source_summary.get("test", {}).get("avg_mse", ""),
        "source_test_mae": source_summary.get("test", {}).get("avg_mae", ""),
        "target_self_config": target_best.get("config_path", str(config_path(target, horizon))),
        "target_self_test_mse": t_mse,
        "target_self_test_mae": t_mae,
        "zero_shot_mse": zero.get("avg_mse", ""),
        "zero_shot_mae": zero.get("avg_mae", ""),
        "zero_shot_route_uses_train_only": zero.get("route_uses_train_only", ""),
        "zero_shot_cluster_id": json.dumps(fixed_cluster_id),
        "zero_shot_corr_mean": sum(float(v) for v in zero.get("corr_max", [])) / max(len(zero.get("corr_max", [])), 1),
        "finetune_lr": lr,
        "finetune_epochs": args.epochs,
        "finetune_val_mse": ft.get("val", {}).get("avg_mse", ""),
        "finetune_val_mae": ft.get("val", {}).get("avg_mae", ""),
        "finetune_test_mse": ft_mse,
        "finetune_test_mae": ft_mae,
        "finetune_best_epoch": json.dumps(ft.get("best_epoch", "")),
        "config_path": str(ft_cfg),
        "out_dir": str(ft_out),
    }
    try:
        row["finetune_vs_target_self_mse"] = float(ft_mse) - float(t_mse)
        row["finetune_vs_target_self_mae"] = float(ft_mae) - float(t_mae)
        if row["finetune_vs_target_self_mse"] > 0 or row["finetune_vs_target_self_mae"] > 0:
            row["target_self_gap_note"] = "finetune_worse_than_current_target_self"
    except Exception:
        pass
    try:
        zs = float(row["zero_shot_mse"])
        row["finetune_gain_pct_vs_zero_shot"] = (zs - float(ft_mse)) / zs * 100.0 if zs else ""
    except Exception:
        pass
    return row


def parse_csv_list(raw: str, cast=str) -> list[Any]:
    return [cast(v.strip()) for v in raw.split(",") if v.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "ettm1_current_full_horizon_transfer_finetune")
    ap.add_argument("--targets", type=str, default=",".join(TARGETS))
    ap.add_argument("--horizons", type=str, default=",".join(str(v) for v in HORIZONS))
    ap.add_argument("--lrs", type=str, default="0.0001")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--source-epochs", type=int, default=50)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--python", type=Path, default=Path(sys.executable))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--resample-method", type=str, default="last", choices=["last", "ffill", "linear", "mean", "none"])
    ap.add_argument("--no-load-gate", action="store_true")
    ap.add_argument("--no-load-dynamic-lambda", action="store_true")
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--rerun-source", action="store_true")
    args = ap.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    targets = parse_csv_list(args.targets, str)
    horizons = parse_csv_list(args.horizons, int)
    lrs = parse_csv_list(args.lrs, float)
    best = best_results()
    result_path = args.out_root / "transfer_finetune.csv"
    rows: list[dict[str, Any]] = []
    if result_path.exists() and not args.rerun:
        with result_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    done = {
        (r.get("target"), int(r.get("pred_len", -1)), str(r.get("finetune_lr")))
        for r in rows
        if r.get("status") == "ok"
    }
    for horizon in horizons:
        for target in targets:
            for lr in lrs:
                key = (target, horizon, str(lr))
                if key in done and not args.rerun:
                    print(f"[skip] {SOURCE}->{target} H{horizon} lr={lr:g}", flush=True)
                    continue
                try:
                    rows.append(run_one(args, best, target, horizon, lr))
                except Exception as exc:
                    rows.append({"status": "error", "source": SOURCE, "target": target, "pred_len": horizon, "finetune_lr": lr, "error": str(exc)[-4000:]})
                write_rows(result_path, rows)
    write_rows(result_path, rows)
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
