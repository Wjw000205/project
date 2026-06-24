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


TARGET_STEP_MINUTES = 15


SOURCES = {
    "ETTm1": {
        "config": ROOT / "configs" / "ETTm1_H96.yaml",
        "targets": ["ETTh1", "ETTh2", "ETTm2"],
    },
    "ETTm2": {
        "config": ROOT / "configs" / "ETTm2_H96.yaml",
        "targets": ["ETTh1", "ETTh2", "ETTm1"],
    },
}

TARGET_SELF_H96 = {
    "ETTh1": (0.3578997254371643, 0.38686761260032654),
    "ETTh2": (0.272211492061615, 0.33122584223747253),
    "ETTm1": (0.29471489787101746, 0.3487132489681244),
    "ETTm2": (0.1645904928445816, 0.24671995639801025),
}

FINETUNE_FIELDS = [
    "status",
    "source",
    "target",
    "horizon",
    "input_len",
    "source_checkpoint",
    "source_memory",
    "source_self_mse",
    "source_self_mae",
    "target_self_mse",
    "target_self_mae",
    "zero_shot_mse",
    "zero_shot_mae",
    "zero_shot_route_uses_train_only",
    "zero_shot_cluster_id",
    "finetune_lr",
    "finetune_epochs",
    "finetune_val_mse",
    "finetune_val_mae",
    "finetune_test_mse",
    "finetune_test_mae",
    "finetune_best_epoch",
    "finetune_loaded_pred_residual",
    "finetune_gain_pct_vs_zero_shot_mse",
    "finetune_gain_pct_vs_zero_shot_mae",
    "finetune_gain_pct_vs_target_self_mse",
    "finetune_gain_pct_vs_target_self_mae",
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "status",
        "source",
        "target",
        "horizon",
        "input_len",
        "source_checkpoint",
        "source_memory",
        "source_self_mse",
        "source_self_mae",
        "target_self_mse",
        "target_self_mae",
        "transfer_mse",
        "transfer_mae",
        "gain_pct_vs_target_self_mse",
        "gain_pct_vs_target_self_mae",
        "route_uses_train_only",
        "zero_shot_cluster_id",
        "zero_shot_corr_mean",
        "config_path",
        "out_dir",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_finetune_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FINETUNE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FINETUNE_FIELDS})


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_lr(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def normalize_runtime_compatible_cfg(cfg: dict[str, Any]) -> None:
    pred_cfg = cfg.get("moe", {}).get("pred_side_residual")
    if not isinstance(pred_cfg, dict):
        return
    policy = str(pred_cfg.get("selection_policy", "")).lower()
    if policy == "val_mse_candidate_channel_guarded":
        pred_cfg["selection_policy"] = "val_mse_candidate_channel"


def _infer_step_minutes(df: pd.DataFrame, date_col: str) -> float:
    dt = pd.to_datetime(df[date_col])
    diffs = dt.diff().dropna()
    if diffs.empty:
        return 0.0
    step = diffs.mode().iloc[0]
    return float(step.total_seconds() / 60.0)


def _resample_df(df: pd.DataFrame, date_col: str, target_step_min: int, method: str) -> pd.DataFrame:
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


def template_config_path(source: str, target: str) -> Path:
    path = ROOT / "configs" / f"{source}To{target}.yaml"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def config_path(dataset: str, horizon: int = 96) -> Path:
    path = ROOT / "configs" / f"{dataset}_H{int(horizon)}.yaml"
    if path.exists():
        return path
    fallback = ROOT / "configs" / f"{dataset}.yaml"
    if not fallback.exists():
        raise FileNotFoundError(f"Missing config for {dataset} H{horizon}: {path}")
    return fallback


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


def target_data_cfg_for_finetune(
    target_cfg: dict[str, Any],
    target: str,
    out_root: Path,
    resample_method: str,
) -> dict[str, Any]:
    data_cfg = copy.deepcopy(target_cfg["data"])
    method = resample_method.lower()
    if target not in {"ETTh1", "ETTh2"} or method in {"none", "off", "false"}:
        return data_cfg
    raw_path = ROOT / str(data_cfg["csv_path"])
    raw_df = pd.read_csv(raw_path)
    max_rows = int(data_cfg.get("max_rows", 0) or 0)
    if max_rows > 0:
        raw_df = raw_df.iloc[:max_rows].copy()
    date_col = raw_df.columns[int(data_cfg.get("date_col", 0))]
    cur_step = _infer_step_minutes(raw_df, date_col)
    if cur_step > 0 and int(round(cur_step)) == TARGET_STEP_MINUTES:
        return data_cfg
    prepared_path = out_root / "prepared_data" / f"{target}_{TARGET_STEP_MINUTES}min_{method}.csv"
    if not prepared_path.exists():
        prepared_path.parent.mkdir(parents=True, exist_ok=True)
        _resample_df(raw_df, date_col, TARGET_STEP_MINUTES, method).to_csv(prepared_path, index=False)
    data_cfg["csv_path"] = str(prepared_path)
    data_cfg["max_rows"] = 0
    return data_cfg


def prepare_finetune_target_cfg(
    *,
    source: str,
    target: str,
    out_root: Path,
    resample_method: str = "last",
) -> dict[str, Any]:
    del source
    target_cfg = read_yaml(config_path(target, 96))
    data_cfg = target_data_cfg_for_finetune(target_cfg, target, out_root, resample_method)
    return {"data": data_cfg, "window": target_cfg.get("window", {})}


def export_source(
    *,
    source: str,
    out_root: Path,
    device: str,
    py: str,
    rerun_source: bool,
    source_epochs: int,
    source_config_path: Path | None = None,
) -> tuple[Path, Path, Path, Path]:
    source_config_path = Path(source_config_path) if source_config_path is not None else config_path(source, 96)
    source_cfg = read_yaml(source_config_path)
    source_run_dir = out_root / "source" / f"{source}_H96_legacy_aligned_export"
    export_cfg_path = out_root / "configs" / "source" / f"{source}_H96_legacy_aligned_export.yaml"
    cfg = copy.deepcopy(source_cfg)
    normalize_runtime_compatible_cfg(cfg)
    cfg.setdefault("exp", {})
    cfg["exp"]["name"] = f"{source}_H96_input96_legacy_aligned_transfer_source"
    cfg["exp"]["out_dir"] = str(source_run_dir)
    cfg["exp"]["device"] = device
    cfg.setdefault("window", {})
    cfg["window"]["input_len"] = 96
    cfg["window"]["pred_len"] = 96
    cfg.setdefault("normalize", {})["train_only"] = True
    cfg.setdefault("cluster", {})["train_only"] = True
    cfg.setdefault("train", {})["epochs"] = int(source_epochs)
    cfg.setdefault("memory", {})
    cfg["memory"]["enable"] = False
    cfg["memory"]["save_checkpoint"] = True
    cfg["memory"]["path"] = str(source_run_dir / "cluster_memory.pt")
    cfg["memory"]["checkpoint_path"] = str(source_run_dir / "best_checkpoint.pt")
    cfg.setdefault("eval", {})
    cfg["eval"]["skip_test"] = False
    cfg["plot"] = {"enable": False}
    cfg["portrait"] = {"enable": False, "out_dir": str(source_run_dir / "cluster_portraits")}
    write_yaml(export_cfg_path, cfg)

    checkpoint_path = source_run_dir / "best_checkpoint.pt"
    memory_path = source_run_dir / "cluster_memory.pt"
    summary_path = source_run_dir / "run_summary.json"
    if rerun_source or not (checkpoint_path.exists() and summary_path.exists()):
        print(f"[source] export {source} H96 input96 checkpoint+memory", flush=True)
        run_cmd(
            [py, "-u", "-m", "src.train", "--config", str(export_cfg_path)],
            log_path=source_run_dir / "source_export.log",
        )
    if rerun_source or not memory_path.exists():
        norm_train_tc = normalized_train_data(cfg)
        _, channel_names = data_frame_to_tensor(cfg)
        ckpt = load_cluster_checkpoint(str(checkpoint_path), device=torch.device("cpu"))
        cluster_id_c = ckpt["meta"]["cluster_id_c"].to(torch.long)
        prototypes_kt = compute_cluster_prototypes(norm_train_tc, cluster_id_c)
        save_cluster_memory(
            str(memory_path),
            prototypes_kt,
            cluster_id_c,
            channel_names,
            meta={
                "kind": "input96_legacy_aligned_source_train_prototype",
                "source_split": "train",
                "input_len": int(cfg["window"]["input_len"]),
                "pred_len": int(cfg["window"]["pred_len"]),
                "source_config": str(export_cfg_path),
                "source_checkpoint": str(checkpoint_path),
            },
        )
    return export_cfg_path, checkpoint_path, memory_path, summary_path


def make_finetune_config(
    *,
    source: str,
    target: str,
    source_cfg: dict[str, Any],
    target_cfg: dict[str, Any],
    source_checkpoint: Path,
    source_memory: Path,
    fixed_cluster_id: list[int],
    out_dir: Path,
    lr: float,
    epochs: int,
    batch_size: int,
    device: str,
    load_gate: bool = True,
    load_dynamic_lambda: bool = True,
) -> dict[str, Any]:
    cfg = copy.deepcopy(source_cfg)
    normalize_runtime_compatible_cfg(cfg)
    cfg["exp"] = {
        "name": f"input96_{source}_to_{target}_H96_finetune_{_fmt_lr(lr)}",
        "out_dir": str(out_dir),
        "seed": int(source_cfg.get("exp", {}).get("seed", 2026)),
        "deterministic": True,
        "device": device,
    }
    cfg["data"] = copy.deepcopy(target_cfg["data"])
    cfg.setdefault("window", {})
    cfg["window"]["input_len"] = 96
    cfg["window"]["pred_len"] = 96
    cfg["window"]["past_context"] = bool(target_cfg.get("window", {}).get("past_context", True))
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
    cfg["train"]["penalty_warmup_epochs"] = min(int(cfg["train"].get("penalty_warmup_epochs", 10)), 5)
    cfg.setdefault("early_stop", {})
    cfg["early_stop"]["patience"] = int(cfg["early_stop"].get("patience", 10))
    cfg["eval"] = {"skip_test": False}
    cfg["plot"] = {"enable": False}
    cfg["portrait"] = {"enable": False, "out_dir": str(out_dir / "cluster_portraits")}
    cfg["memory"] = {
        "enable": False,
        "save_checkpoint": True,
        "path": str(out_dir / "cluster_memory.pt"),
        "checkpoint_path": str(out_dir / "best_checkpoint.pt"),
    }
    cfg["finetune"] = {
        "enable": True,
        "checkpoint_path": str(source_checkpoint),
        "memory_path": str(source_memory),
        "cluster_map": "index",
        "strict_window": True,
        "strict_model": True,
        "load_model": True,
        "load_gate": bool(load_gate),
        "load_dynamic_lambda": bool(load_dynamic_lambda),
        "load_learnable_lambda": True,
    }
    return cfg


def select_best_finetune_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    best: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("status", "")).lower() != "ok":
            continue
        val_mse = _float_or_none(row.get("finetune_val_mse"))
        if val_mse is None:
            continue
        key = (str(row.get("source", "")), str(row.get("target", "")))
        current = best.get(key)
        if current is None or val_mse < current[0]:
            best[key] = (val_mse, row)
    return {key: row for key, (_, row) in best.items()}


def prepare_source(
    source: str,
    info: dict[str, Any],
    out_root: Path,
    device: str,
    py: str,
    rerun_source: bool,
    source_epochs: int,
) -> dict[str, Any]:
    source_cfg_path, checkpoint_path, memory_path, summary_path = export_source(
        source=source,
        out_root=out_root,
        device=device,
        py=py,
        rerun_source=rerun_source,
        source_epochs=source_epochs,
        source_config_path=info.get("config"),
    )
    cfg = read_yaml(source_cfg_path)
    ckpt = load_cluster_checkpoint(str(checkpoint_path), device=torch.device("cpu"))
    meta = ckpt.get("meta", {})
    if int(meta.get("input_len")) != 96 or int(meta.get("pred_len")) != 96:
        raise ValueError(f"{source} checkpoint is not input96/H96: {meta.get('input_len')}/{meta.get('pred_len')}")
    if not memory_path.exists():
        raise FileNotFoundError(f"Missing source memory: {memory_path}")
    summary = load_json(summary_path)
    return {
        "config": source_cfg_path,
        "checkpoint": checkpoint_path,
        "summary": summary_path,
        "memory": memory_path,
        "self_mse": summary.get("test", {}).get("avg_mse"),
        "self_mae": summary.get("test", {}).get("avg_mae"),
        "source_cfg": cfg,
    }


def build_transfer_config(
    *,
    source: str,
    target: str,
    source_info: dict[str, Any],
    out_root: Path,
    device: str,
    batch_size: int,
    resample_method: str,
) -> Path:
    target_cfg = read_yaml(config_path(target, 96))
    source_cfg = source_info["source_cfg"]
    out_dir = out_root / "zero_shot" / f"{source}_to_{target}" / "H96"
    method = resample_method.lower()
    resample_enable = target in {"ETTh1", "ETTh2"} and method not in {"none", "off", "false"}
    cfg = {
        "exp": {
            "name": f"input96_{source}_to_{target}_H96_transfer",
            "out_dir": str(out_dir),
            "seed": 2026,
            "device": device,
        },
        "source": {
            "memory_path": str(source_info["memory"]),
            "checkpoint_path": str(source_info["checkpoint"]),
            "summary_path": str(source_info["summary"]),
            "csv_path": source_cfg["data"]["csv_path"],
            "date_col": source_cfg["data"].get("date_col", 0),
            "step_minutes": TARGET_STEP_MINUTES,
        },
        "data": copy.deepcopy(target_cfg["data"]),
        "window": {
            "input_len": 96,
            "pred_len": 96,
            "past_context": bool(target_cfg.get("window", {}).get("past_context", False)),
        },
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
            "cluster_balance_repair": {
                "enable": True,
                "target_counts": "source",
                "min_unique_clusters": 2,
            },
            "resample": {
                "enable": resample_enable,
                "target_step_minutes": TARGET_STEP_MINUTES,
                "method": method,
            },
            "save_corr": True,
        },
        "eval": {"batch_size": int(batch_size), "split": "test"},
    }
    path = out_root / "configs" / f"{source}_to_{target}_H96_transfer.yaml"
    write_yaml(path, cfg)
    return path


def build_finetune_config_path(
    *,
    source: str,
    target: str,
    source_info: dict[str, Any],
    fixed_cluster_id: list[int],
    out_root: Path,
    device: str,
    lr: float,
    epochs: int,
    batch_size: int,
    resample_method: str,
    load_gate: bool,
    load_dynamic_lambda: bool,
) -> Path:
    target_cfg = prepare_finetune_target_cfg(
        source=source,
        target=target,
        out_root=out_root,
        resample_method=resample_method,
    )
    out_dir = out_root / "finetune" / f"{source}_to_{target}" / "H96" / f"lr{_fmt_lr(lr)}"
    cfg = make_finetune_config(
        source=source,
        target=target,
        source_cfg=source_info["source_cfg"],
        target_cfg=target_cfg,
        source_checkpoint=source_info["checkpoint"],
        source_memory=source_info["memory"],
        fixed_cluster_id=fixed_cluster_id,
        out_dir=out_dir,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        load_gate=load_gate,
        load_dynamic_lambda=load_dynamic_lambda,
    )
    path = out_root / "configs" / "finetune" / f"{source}_to_{target}_H96_lr{_fmt_lr(lr)}.yaml"
    write_yaml(path, cfg)
    return path


def run_cmd(cmd: list[str], log_path: Path) -> None:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout[-4000:])


def run_transfers(args: argparse.Namespace) -> list[dict[str, Any]]:
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"sources": {}, "configs": []}
    prepared = {
        source: prepare_source(
            source,
            info,
            out_root,
            device=args.device,
            py=str(args.python),
            rerun_source=args.rerun_source,
            source_epochs=args.source_epochs,
        )
        for source, info in SOURCES.items()
    }
    rows: list[dict[str, Any]] = []
    for source, source_info in prepared.items():
        manifest["sources"][source] = {
            "checkpoint": str(source_info["checkpoint"].relative_to(ROOT)),
            "summary": str(source_info["summary"].relative_to(ROOT)),
            "memory": str(source_info["memory"].relative_to(ROOT)),
            "source_self_mse": source_info["self_mse"],
            "source_self_mae": source_info["self_mae"],
        }
        for target in SOURCES[source]["targets"]:
            cfg_path = build_transfer_config(
                source=source,
                target=target,
                source_info=source_info,
                out_root=out_root,
                device=args.device,
                batch_size=args.batch_size,
                resample_method=args.resample_method,
            )
            manifest["configs"].append(str(cfg_path.relative_to(ROOT)))
            cfg = read_yaml(cfg_path)
            out_dir = Path(cfg["exp"]["out_dir"])
            summary_path = out_dir / "transfer_summary.json"
            row = {
                "status": "ok",
                "source": source,
                "target": target,
                "horizon": 96,
                "input_len": 96,
                "source_checkpoint": str(source_info["checkpoint"].relative_to(ROOT)),
                "source_memory": str(source_info["memory"].relative_to(ROOT)),
                "source_self_mse": source_info["self_mse"],
                "source_self_mae": source_info["self_mae"],
                "config_path": str(cfg_path.relative_to(ROOT)),
                "out_dir": str(out_dir),
            }
            try:
                if args.rerun or not summary_path.exists():
                    print(f"[transfer.py] {source}->{target} H96 input96", flush=True)
                    run_cmd(
                        [str(args.python), "-u", "-m", "src.transfer", "--config", str(cfg_path)],
                        log_path=out_dir / "transfer.log",
                    )
                summary = load_json(summary_path)
                transfer_mse = float(summary["avg_mse"])
                transfer_mae = float(summary["avg_mae"])
                target_mse, target_mae = TARGET_SELF_H96[target]
                row.update(
                    {
                        "target_self_mse": target_mse,
                        "target_self_mae": target_mae,
                        "transfer_mse": transfer_mse,
                        "transfer_mae": transfer_mae,
                        "gain_pct_vs_target_self_mse": (target_mse - transfer_mse) / target_mse * 100.0,
                        "gain_pct_vs_target_self_mae": (target_mae - transfer_mae) / target_mae * 100.0,
                        "route_uses_train_only": summary.get("route_uses_train_only"),
                        "zero_shot_cluster_id": json.dumps(summary.get("cluster_id", [])),
                        "zero_shot_corr_mean": summary.get("corr_mean", ""),
                    }
                )
            except Exception as exc:
                row["status"] = "error"
                row["error"] = str(exc)[-4000:]
            rows.append(row)
            write_csv(out_root / "input96_transfer_results.csv", rows)
    write_json(out_root / "manifest.json", manifest)
    write_csv(out_root / "input96_transfer_results.csv", rows)
    return rows


def _load_zero_cluster_id(zero_row: dict[str, Any]) -> list[int]:
    raw = zero_row.get("zero_shot_cluster_id", "")
    if raw:
        if isinstance(raw, str):
            return [int(v) for v in json.loads(raw)]
        return [int(v) for v in raw]
    out_dir = zero_row.get("out_dir")
    if not out_dir:
        raise ValueError("Missing zero-shot row/out_dir; cannot recover fixed_cluster_id")
    summary_path = Path(str(out_dir)) / "transfer_summary.json"
    summary = load_json(summary_path)
    return [int(v) for v in summary["cluster_id"]]


def run_finetunes(
    args: argparse.Namespace,
    prepared: dict[str, dict[str, Any]],
    zero_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    zero_by_pair = {(row.get("source"), row.get("target")): row for row in zero_rows}
    lrs = [float(v.strip()) for v in str(args.lrs).split(",") if v.strip()]
    rows: list[dict[str, Any]] = []
    result_path = args.out_root / "input96_transfer_finetune_results.csv"
    if result_path.exists() and not args.rerun_finetune:
        with result_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    done = {
        (row.get("source"), row.get("target"), str(row.get("finetune_lr")))
        for row in rows
        if row.get("status") == "ok"
    }
    for source, source_info in prepared.items():
        for target in SOURCES[source]["targets"]:
            zero = zero_by_pair.get((source, target), {})
            for lr in lrs:
                lr_s = f"{lr:g}"
                if (source, target, lr_s) in done and not args.rerun_finetune:
                    print(f"[skip finetune] {source}->{target} H96 lr={lr_s}", flush=True)
                    continue
                fixed_cluster_id = _load_zero_cluster_id(zero)
                cfg_path = build_finetune_config_path(
                    source=source,
                    target=target,
                    source_info=source_info,
                    fixed_cluster_id=fixed_cluster_id,
                    out_root=args.out_root,
                    device=args.device,
                    lr=lr,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    resample_method=args.resample_method,
                    load_gate=not args.no_load_gate,
                    load_dynamic_lambda=not args.no_load_dynamic_lambda,
                )
                cfg = read_yaml(cfg_path)
                out_dir = Path(cfg["exp"]["out_dir"])
                summary_path = out_dir / "run_summary.json"
                target_mse, target_mae = TARGET_SELF_H96[target]
                row = {
                    "status": "ok",
                    "source": source,
                    "target": target,
                    "horizon": 96,
                    "input_len": 96,
                    "source_checkpoint": str(source_info["checkpoint"].relative_to(ROOT)),
                    "source_memory": str(source_info["memory"].relative_to(ROOT)),
                    "source_self_mse": source_info["self_mse"],
                    "source_self_mae": source_info["self_mae"],
                    "target_self_mse": target_mse,
                    "target_self_mae": target_mae,
                    "zero_shot_mse": zero.get("transfer_mse", ""),
                    "zero_shot_mae": zero.get("transfer_mae", ""),
                    "zero_shot_route_uses_train_only": zero.get("route_uses_train_only", ""),
                    "zero_shot_cluster_id": json.dumps(fixed_cluster_id),
                    "finetune_lr": lr_s,
                    "finetune_epochs": int(args.epochs),
                    "config_path": str(cfg_path.relative_to(ROOT)),
                    "out_dir": str(out_dir),
                }
                try:
                    if args.rerun_finetune or not summary_path.exists():
                        print(f"[finetune] {source}->{target} H96 input96 lr={lr_s}", flush=True)
                        run_cmd(
                            [str(args.python), "-u", "-m", "src.train", "--config", str(cfg_path)],
                            log_path=out_dir / "finetune.log",
                        )
                    summary = load_json(summary_path)
                    ft_mse = float(summary["test"]["avg_mse"])
                    ft_mae = float(summary["test"]["avg_mae"])
                    zero_mse = _float_or_none(row.get("zero_shot_mse"))
                    zero_mae = _float_or_none(row.get("zero_shot_mae"))
                    row.update(
                        {
                            "finetune_val_mse": summary.get("val", {}).get("avg_mse", ""),
                            "finetune_val_mae": summary.get("val", {}).get("avg_mae", ""),
                            "finetune_test_mse": ft_mse,
                            "finetune_test_mae": ft_mae,
                            "finetune_best_epoch": json.dumps(summary.get("best_epoch", "")),
                            "finetune_loaded_pred_residual": summary.get("finetune", {}).get("loaded_pred_residual", ""),
                            "finetune_gain_pct_vs_target_self_mse": (target_mse - ft_mse) / target_mse * 100.0,
                            "finetune_gain_pct_vs_target_self_mae": (target_mae - ft_mae) / target_mae * 100.0,
                        }
                    )
                    if zero_mse is not None:
                        row["finetune_gain_pct_vs_zero_shot_mse"] = (zero_mse - ft_mse) / zero_mse * 100.0 if zero_mse else ""
                    if zero_mae is not None:
                        row["finetune_gain_pct_vs_zero_shot_mae"] = (zero_mae - ft_mae) / zero_mae * 100.0 if zero_mae else ""
                except Exception as exc:
                    row["status"] = "error"
                    row["error"] = str(exc)[-4000:]
                rows.append(row)
                write_finetune_csv(result_path, rows)
    write_finetune_csv(result_path, rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run H96/input96 ETT transfer.py rerun and summarize results.")
    parser.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "input96_transfer_legacy_aligned_rerun")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--rerun-source", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--rerun-finetune", action="store_true")
    parser.add_argument("--lrs", default="0.0001")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--source-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--resample-method", default="last", choices=["last", "ffill", "linear", "mean", "none"])
    parser.add_argument("--no-load-gate", action="store_true")
    parser.add_argument("--no-load-dynamic-lambda", action="store_true")
    args = parser.parse_args()
    rows = run_transfers(args)
    ft_rows: list[dict[str, Any]] = []
    if args.finetune:
        prepared = {
            source: prepare_source(
                source,
                info,
                args.out_root,
                device=args.device,
                py=str(args.python),
                rerun_source=False,
                source_epochs=args.source_epochs,
            )
            for source, info in SOURCES.items()
        }
        ft_rows = run_finetunes(args, prepared, rows)
    ok = sum(1 for row in rows if row.get("status") == "ok")
    print(f"Saved: {args.out_root / 'input96_transfer_results.csv'}")
    print(f"Rows: {len(rows)} ok={ok} error={len(rows)-ok}")
    if args.finetune:
        ft_ok = sum(1 for row in ft_rows if row.get("status") == "ok")
        print(f"Saved: {args.out_root / 'input96_transfer_finetune_results.csv'}")
        print(f"Finetune rows: {len(ft_rows)} ok={ft_ok} error={len(ft_rows)-ft_ok}")
        return 0 if ok == len(rows) and ft_ok == len(ft_rows) else 1
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
