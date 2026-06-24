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

from src.utils.cluster_memory import (  # noqa: E402
    compute_cluster_prototypes,
    load_cluster_checkpoint,
    save_cluster_memory,
)


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
    "protocol",
    "source_config",
    "source_checkpoint",
    "source_memory",
    "source_test_mse",
    "source_test_mae",
    "source_val_mse",
    "source_val_mae",
    "target_self_config",
    "target_self_test_mse",
    "target_self_test_mae",
    "target_self_val_mse",
    "target_self_val_mae",
    "target_data_csv",
    "target_data_max_rows",
    "prepared_data_csv",
    "train_ratio",
    "val_ratio",
    "test_ratio",
    "normalize_train_only",
    "cluster_train_only",
    "resample_enable",
    "resample_method",
    "zero_shot_mse",
    "zero_shot_mae",
    "zero_shot_route_uses_train_only",
    "zero_shot_num_windows",
    "zero_shot_summary",
    "finetune_lr",
    "finetune_epochs",
    "finetune_patience",
    "finetune_load_model",
    "finetune_load_gate",
    "finetune_load_dynamic_lambda",
    "finetune_val_mse",
    "finetune_val_mae",
    "finetune_test_mse",
    "finetune_test_mae",
    "finetune_best_epoch",
    "finetune_total_sec",
    "finetune_avg_epoch_sec",
    "delta_mse_vs_zero_shot",
    "gain_mse_vs_zero_shot",
    "gain_pct_vs_zero_shot",
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


def source_config_path(horizon: int) -> Path:
    return ROOT / "outputs" / "ett_global_h96_param_base" / "configs" / f"{SOURCE}_pred_{horizon}.yaml"


def source_run_dir(horizon: int) -> Path:
    return ROOT / "outputs" / "ett_global_h96_param_base" / "runs" / SOURCE / f"pred_{horizon}"


def target_config_path(target: str, horizon: int) -> Path:
    return ROOT / "outputs" / "ett_horizon_sweep" / "configs" / f"{target}_pred_{horizon}.yaml"


def target_summary_path(target: str, horizon: int) -> Path:
    return ROOT / "outputs" / "ett_horizon_sweep" / "runs" / target / f"pred_{horizon}" / "run_summary.json"


def _infer_step_minutes(df: pd.DataFrame, date_col: str) -> float:
    dt = pd.to_datetime(df[date_col])
    diffs = dt.diff().dropna()
    if diffs.empty:
        return 0.0
    step = diffs.mode().iloc[0]
    return float(step.total_seconds() / 60.0)


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
    return out.reset_index()


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
    t_train = int(data_tc.shape[0] * float(cfg["data"]["train_ratio"]))
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
        raise FileNotFoundError(f"Missing source run_summary: {summary_path}")

    existing = run_dir / "cluster_memory.pt"
    if existing.exists():
        return checkpoint_path, summary_path, existing

    memory_path = out_root / "source_memory" / f"{SOURCE}_pred_{horizon}" / "cluster_memory.pt"
    if memory_path.exists():
        return checkpoint_path, summary_path, memory_path

    memory_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = read_yaml(cfg_path)
    norm_tc = normalized_source_data(cfg)
    t_train = int(norm_tc.shape[0] * float(cfg["data"]["train_ratio"]))
    _, channel_names = data_frame_to_tensor(cfg)
    ckpt = load_cluster_checkpoint(str(checkpoint_path), device=torch.device("cpu"))
    cluster_id_c = ckpt["meta"]["cluster_id_c"].to(torch.long)
    prototypes_kt = compute_cluster_prototypes(norm_tc[:t_train], cluster_id_c)
    save_cluster_memory(
        str(memory_path),
        prototypes_kt,
        cluster_id_c,
        channel_names,
        meta={
            "kind": "train_segment_prototype_synthesized_for_finetune",
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


def prepare_target_data(
    *,
    target: str,
    horizon: int,
    out_root: Path,
    resample_method: str,
) -> tuple[dict[str, Any], dict[str, Any], bool, Path | None]:
    target_cfg = read_yaml(target_config_path(target, horizon))
    data_cfg = copy.deepcopy(target_cfg["data"])
    data_cfg["train_ratio"] = 0.6
    data_cfg["val_ratio"] = 0.2
    data_cfg["test_ratio"] = 0.2
    data_cfg["date_col"] = int(data_cfg.get("date_col", 0))

    resample_enable = target in {"ETTh1", "ETTh2"} and resample_method.lower() not in {"none", "off", "false"}
    prepared_path: Path | None = None
    if resample_enable:
        raw_df = pd.read_csv(ROOT / str(data_cfg["csv_path"]))
        max_rows = int(data_cfg.get("max_rows", 0) or 0)
        if max_rows > 0:
            raw_df = raw_df.iloc[:max_rows].copy()
        date_col = raw_df.columns[int(data_cfg.get("date_col", 0))]
        cur_step = _infer_step_minutes(raw_df, date_col)
        if cur_step > 0 and int(round(cur_step)) == TARGET_STEP_MINUTES:
            resample_enable = False
        else:
            prepared_path = out_root / "prepared_data" / f"{target}_{TARGET_STEP_MINUTES}min_{resample_method}.csv"
            if not prepared_path.exists():
                prepared_path.parent.mkdir(parents=True, exist_ok=True)
                resampled = _resample_df(raw_df, date_col, TARGET_STEP_MINUTES, resample_method.lower())
                resampled.to_csv(prepared_path, index=False)
            data_cfg["csv_path"] = str(prepared_path)
            data_cfg["max_rows"] = 0

    meta = {
        "target_data_csv": target_cfg["data"]["csv_path"],
        "target_data_max_rows": target_cfg["data"].get("max_rows", 0),
        "prepared_data_csv": "" if prepared_path is None else str(prepared_path),
    }
    return data_cfg, target_cfg, resample_enable, prepared_path


def build_zero_shot_config(
    *,
    target: str,
    horizon: int,
    out_root: Path,
    device: str,
    batch_size: int,
    checkpoint_path: Path,
    source_summary_path: Path,
    memory_path: Path,
    resample_method: str,
) -> Path:
    target_cfg = read_yaml(target_config_path(target, horizon))
    source_cfg = read_yaml(source_config_path(horizon))
    ckpt = load_cluster_checkpoint(str(checkpoint_path), device=torch.device("cpu"))
    input_len = int(ckpt["meta"]["input_len"])
    pred_len = int(ckpt["meta"]["pred_len"])
    if pred_len != horizon:
        raise ValueError(f"Source checkpoint horizon mismatch: {checkpoint_path}")

    resample_enable = target in {"ETTh1", "ETTh2"} and resample_method.lower() not in {"none", "off", "false"}
    out_dir = out_root / "zero_shot" / f"{SOURCE}_to_{target}" / f"pred_{horizon}"
    cfg = {
        "exp": {
            "name": f"{SOURCE}_to_{target}_H{horizon}_zero_shot",
            "out_dir": str(out_dir),
            "seed": 2026,
            "device": device,
        },
        "source": {
            "memory_path": str(memory_path),
            "checkpoint_path": str(checkpoint_path),
            "summary_path": str(source_summary_path),
            "csv_path": source_cfg["data"]["csv_path"],
            "date_col": source_cfg["data"].get("date_col", 0),
            "step_minutes": TARGET_STEP_MINUTES,
        },
        "data": copy.deepcopy(target_cfg["data"]),
        "window": {
            "input_len": input_len,
            "pred_len": pred_len,
            "past_context": bool(target_cfg.get("window", {}).get("past_context", False)),
        },
        "normalize": {
            "global_zscore": True,
            "train_only": True,
        },
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
            "resample": {
                "enable": resample_enable,
                "target_step_minutes": TARGET_STEP_MINUTES,
                "method": resample_method.lower(),
            },
            "save_corr": True,
        },
        "eval": {
            "batch_size": batch_size,
            "split": "test",
        },
    }
    cfg_path = out_root / "configs" / "zero_shot" / f"{SOURCE}_to_{target}_pred_{horizon}.yaml"
    write_yaml(cfg_path, cfg)
    return cfg_path


def build_finetune_config(
    *,
    target: str,
    horizon: int,
    lr: float,
    epochs: int,
    out_root: Path,
    device: str,
    checkpoint_path: Path,
    memory_path: Path,
    data_cfg: dict[str, Any],
    target_cfg: dict[str, Any],
    load_gate: bool,
    load_dynamic_lambda: bool,
) -> Path:
    cfg = read_yaml(source_config_path(horizon))
    cfg = copy.deepcopy(cfg)
    tag = f"lr{lr:g}".replace(".", "p")
    out_dir = out_root / "runs" / f"{SOURCE}_to_{target}" / f"pred_{horizon}" / tag

    cfg["exp"] = {
        "name": f"{SOURCE}_to_{target}_H{horizon}_finetune_{tag}",
        "out_dir": str(out_dir),
        "seed": 2026,
        "deterministic": True,
        "device": device,
    }
    cfg["data"] = copy.deepcopy(data_cfg)
    cfg["window"]["input_len"] = 336
    cfg["window"]["pred_len"] = horizon
    cfg["window"]["past_context"] = bool(target_cfg.get("window", {}).get("past_context", False))
    cfg["normalize"] = {
        "global_zscore": True,
        "train_only": True,
    }
    cfg.setdefault("cluster", {})
    cfg["cluster"]["train_only"] = True
    cfg["corr"] = {
        "compute": True,
        "save_path": str(out_dir / "corr.npy"),
    }
    cfg.setdefault("train", {})
    cfg["train"]["epochs"] = int(epochs)
    cfg["train"]["lr"] = float(lr)
    cfg["train"].setdefault("weight_decay", 0.0001)
    cfg["train"].setdefault("batch_size", 64)
    cfg["train"].setdefault("selection_metric", "val_mse")
    cfg["train"]["penalty_warmup_epochs"] = min(int(cfg["train"].get("penalty_warmup_epochs", 10)), 5)
    cfg.setdefault("early_stop", {})
    cfg["early_stop"]["patience"] = int(cfg["early_stop"].get("patience", 10))
    cfg["early_stop"]["min_delta"] = float(cfg["early_stop"].get("min_delta", 1.0e-6))
    cfg["eval"] = {"skip_test": False}
    cfg["plot"] = {"enable": False}
    cfg["portrait"] = {
        "enable": False,
        "out_dir": str(out_dir / "cluster_portraits"),
    }
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
        "cluster_map": "corr",
        "corr_align": "head",
        "strict_window": True,
        "strict_model": True,
        "load_model": True,
        "load_gate": bool(load_gate),
        "load_dynamic_lambda": bool(load_dynamic_lambda),
        "load_learnable_lambda": True,
    }

    cfg_path = out_root / "configs" / "finetune" / f"{SOURCE}_to_{target}_pred_{horizon}_{tag}.yaml"
    write_yaml(cfg_path, cfg)
    return cfg_path


def read_target_self(target: str, horizon: int) -> dict[str, Any]:
    path = target_summary_path(target, horizon)
    if not path.exists():
        return {}
    return load_json(path)


def row_base(
    *,
    target: str,
    horizon: int,
    checkpoint_path: Path,
    source_summary_path: Path,
    memory_path: Path,
    target_cfg: dict[str, Any],
    target_data_meta: dict[str, Any],
    resample_enable: bool,
    resample_method: str,
) -> dict[str, Any]:
    source_summary = load_json(source_summary_path)
    target_summary = read_target_self(target, horizon)
    return {
        "status": "ok",
        "source": SOURCE,
        "target": target,
        "pred_len": horizon,
        "input_len": int(source_summary.get("windowing", {}).get("input_len", 336)),
        "source_config": str(source_config_path(horizon)),
        "source_checkpoint": str(checkpoint_path),
        "source_memory": str(memory_path),
        "source_test_mse": source_summary.get("test", {}).get("avg_mse", ""),
        "source_test_mae": source_summary.get("test", {}).get("avg_mae", ""),
        "source_val_mse": source_summary.get("val", {}).get("avg_mse", ""),
        "source_val_mae": source_summary.get("val", {}).get("avg_mae", ""),
        "target_self_config": str(target_config_path(target, horizon)),
        "target_self_test_mse": target_summary.get("test", {}).get("avg_mse", ""),
        "target_self_test_mae": target_summary.get("test", {}).get("avg_mae", ""),
        "target_self_val_mse": target_summary.get("val", {}).get("avg_mse", ""),
        "target_self_val_mae": target_summary.get("val", {}).get("avg_mae", ""),
        "target_data_csv": target_data_meta.get("target_data_csv", ""),
        "target_data_max_rows": target_data_meta.get("target_data_max_rows", ""),
        "prepared_data_csv": target_data_meta.get("prepared_data_csv", ""),
        "train_ratio": target_cfg["data"].get("train_ratio", 0.6),
        "val_ratio": target_cfg["data"].get("val_ratio", 0.2),
        "test_ratio": target_cfg["data"].get("test_ratio", 0.2),
        "normalize_train_only": True,
        "cluster_train_only": True,
        "resample_enable": resample_enable,
        "resample_method": resample_method,
    }


def run_one(
    *,
    target: str,
    horizon: int,
    lrs: list[float],
    epochs: int,
    out_root: Path,
    device: str,
    py: str,
    batch_size: int,
    resample_method: str,
    load_gate: bool,
    load_dynamic_lambda: bool,
    rerun: bool,
) -> list[dict[str, Any]]:
    checkpoint_path, source_summary_path, memory_path = ensure_source_memory(horizon, out_root)
    data_cfg, target_cfg, resample_enable, _prepared_path = prepare_target_data(
        target=target,
        horizon=horizon,
        out_root=out_root,
        resample_method=resample_method,
    )
    target_data_meta = {
        "target_data_csv": target_cfg["data"]["csv_path"],
        "target_data_max_rows": target_cfg["data"].get("max_rows", 0),
        "prepared_data_csv": data_cfg["csv_path"] if data_cfg["csv_path"] != target_cfg["data"]["csv_path"] else "",
    }

    base = row_base(
        target=target,
        horizon=horizon,
        checkpoint_path=checkpoint_path,
        source_summary_path=source_summary_path,
        memory_path=memory_path,
        target_cfg=target_cfg,
        target_data_meta=target_data_meta,
        resample_enable=resample_enable,
        resample_method=resample_method,
    )

    zero_cfg_path = build_zero_shot_config(
        target=target,
        horizon=horizon,
        out_root=out_root,
        device=device,
        batch_size=batch_size,
        checkpoint_path=checkpoint_path,
        source_summary_path=source_summary_path,
        memory_path=memory_path,
        resample_method=resample_method,
    )
    zero_summary_path = Path(read_yaml(zero_cfg_path)["exp"]["out_dir"]) / "transfer_summary.json"
    zero_summary_path = zero_summary_path if zero_summary_path.is_absolute() else ROOT / zero_summary_path
    if rerun or not zero_summary_path.exists():
        print(f"[zero-shot] {SOURCE}->{target} H{horizon}", flush=True)
        run_cmd(
            [py, "-u", "-m", "src.transfer", "--config", str(zero_cfg_path)],
            log_path=zero_summary_path.with_name("zero_shot.log"),
        )
    zero_summary = load_json(zero_summary_path)

    rows: list[dict[str, Any]] = []
    for lr in lrs:
        cfg_path = build_finetune_config(
            target=target,
            horizon=horizon,
            lr=lr,
            epochs=epochs,
            out_root=out_root,
            device=device,
            checkpoint_path=checkpoint_path,
            memory_path=memory_path,
            data_cfg=data_cfg,
            target_cfg=target_cfg,
            load_gate=load_gate,
            load_dynamic_lambda=load_dynamic_lambda,
        )
        cfg = read_yaml(cfg_path)
        out_dir = Path(cfg["exp"]["out_dir"])
        summary_path = out_dir / "run_summary.json"
        if rerun or not summary_path.exists():
            print(f"[fine-tune] {SOURCE}->{target} H{horizon} lr={lr:g}", flush=True)
            run_cmd(
                [py, "-u", "-m", "src.train", "--config", str(cfg_path)],
                log_path=out_dir / "finetune.log",
            )
        summary = load_json(summary_path)
        row = dict(base)
        row.update(
            {
                "zero_shot_mse": zero_summary.get("avg_mse", ""),
                "zero_shot_mae": zero_summary.get("avg_mae", ""),
                "zero_shot_route_uses_train_only": zero_summary.get("route_uses_train_only", ""),
                "zero_shot_num_windows": zero_summary.get("num_eval_windows", ""),
                "zero_shot_summary": str(zero_summary_path),
                "finetune_lr": lr,
                "finetune_epochs": epochs,
                "finetune_patience": cfg.get("early_stop", {}).get("patience", ""),
                "finetune_load_model": cfg.get("finetune", {}).get("load_model", ""),
                "finetune_load_gate": cfg.get("finetune", {}).get("load_gate", ""),
                "finetune_load_dynamic_lambda": cfg.get("finetune", {}).get("load_dynamic_lambda", ""),
                "finetune_val_mse": summary.get("val", {}).get("avg_mse", ""),
                "finetune_val_mae": summary.get("val", {}).get("avg_mae", ""),
                "finetune_test_mse": summary.get("test", {}).get("avg_mse", ""),
                "finetune_test_mae": summary.get("test", {}).get("avg_mae", ""),
                "finetune_best_epoch": json.dumps(summary.get("best_epoch", "")),
                "finetune_total_sec": summary.get("timing", {}).get("total_sec", ""),
                "finetune_avg_epoch_sec": summary.get("timing", {}).get("avg_epoch_sec", ""),
                "config_path": str(cfg_path),
                "out_dir": str(out_dir),
            }
        )
        try:
            zero_mse = float(row["zero_shot_mse"])
            ft_mse = float(row["finetune_test_mse"])
            row["delta_mse_vs_zero_shot"] = ft_mse - zero_mse
            row["gain_mse_vs_zero_shot"] = zero_mse - ft_mse
            row["gain_pct_vs_zero_shot"] = (zero_mse - ft_mse) / zero_mse * 100.0 if zero_mse != 0.0 else ""
        except Exception:
            pass
        rows.append(row)
    return rows


def parse_csv_list(raw: str, cast=str) -> list[Any]:
    return [cast(v.strip()) for v in raw.split(",") if v.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "ettm1_finetune_transfer")
    ap.add_argument("--targets", type=str, default=",".join(TARGETS))
    ap.add_argument("--horizons", type=str, default=",".join(str(v) for v in HORIZONS))
    ap.add_argument("--lrs", type=str, default="0.0001")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--python", type=Path, default=Path(sys.executable))
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--resample-method", type=str, default="last", choices=["last", "ffill", "linear", "mean", "none"])
    ap.add_argument("--no-load-gate", action="store_true")
    ap.add_argument("--no-load-dynamic-lambda", action="store_true")
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()

    targets = parse_csv_list(args.targets, str)
    horizons = parse_csv_list(args.horizons, int)
    lrs = parse_csv_list(args.lrs, float)
    args.out_root.mkdir(parents=True, exist_ok=True)
    result_path = args.out_root / "finetune_vs_zero_shot.csv"

    rows: list[dict[str, Any]] = []
    if result_path.exists() and not args.rerun:
        with result_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

    existing_keys = {
        (r.get("target"), int(r.get("pred_len", -1)), str(r.get("finetune_lr")))
        for r in rows
        if r.get("status") == "ok" and str(r.get("finetune_lr", "")) != ""
    }

    for horizon in horizons:
        for target in targets:
            for lr in lrs:
                key = (target, horizon, str(lr))
                if key in existing_keys and not args.rerun:
                    print(f"[skip] {SOURCE}->{target} H{horizon} lr={lr:g}", flush=True)
                    continue
                try:
                    new_rows = run_one(
                        target=target,
                        horizon=horizon,
                        lrs=[lr],
                        epochs=args.epochs,
                        out_root=args.out_root,
                        device=args.device,
                        py=str(args.python),
                        batch_size=args.batch_size,
                        resample_method=args.resample_method,
                        load_gate=not args.no_load_gate,
                        load_dynamic_lambda=not args.no_load_dynamic_lambda,
                        rerun=args.rerun,
                    )
                    rows.extend(new_rows)
                except Exception as exc:
                    rows.append(
                        {
                            "status": "error",
                            "source": SOURCE,
                            "target": target,
                            "pred_len": horizon,
                            "finetune_lr": lr,
                            "error": str(exc)[-4000:],
                        }
                    )
                write_rows(result_path, rows)

    write_rows(result_path, rows)
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
