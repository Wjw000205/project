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


SOURCES: dict[str, list[str]] = {
    "ETTm1": ["ETTh1", "ETTh2", "ETTm2"],
    "ETTm2": ["ETTh1", "ETTh2", "ETTm1"],
}
DEFAULT_HORIZONS = [96, 192, 336, 720]
TARGET_STEP_MINUTES = 15
QGWNT_LR = 5.0e-5
QGWNT_EPOCHS = 80

FIELDS = [
    "status",
    "source",
    "target",
    "horizon",
    "input_len",
    "reused_h96",
    "source_self_mse",
    "source_self_mae",
    "source_checkpoint",
    "source_memory",
    "zero_shot_eval_split",
    "zero_shot_mse",
    "zero_shot_mae",
    "zero_shot_route_uses_train_only",
    "zero_shot_cluster_id",
    "zero_shot_corr_mean",
    "val_raw_mse",
    "val_raw_mae",
    "val_base_mse",
    "val_base_mae",
    "val_scaled_mse",
    "val_scaled_mae",
    "test_mse",
    "test_mae",
    "best_epoch",
    "finetune_partial",
    "loaded_pred_residual",
    "val_summary",
    "test_summary",
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


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


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


def config_path(dataset: str, horizon: int) -> Path:
    path = ROOT / "configs" / f"{dataset}_H{int(horizon)}.yaml"
    if path.exists():
        return path
    fallback = ROOT / "configs" / f"{dataset}.yaml"
    if not fallback.exists():
        raise FileNotFoundError(f"Missing config for {dataset} H{horizon}: {path}")
    return fallback


def _fmt_lr(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def _parse_ints(raw: str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(raw, str):
        return [int(v.strip()) for v in raw.split(",") if v.strip()]
    return [int(v) for v in raw]


def normalize_runtime_compatible_cfg(cfg: dict[str, Any]) -> None:
    pred_cfg = cfg.get("moe", {}).get("pred_side_residual")
    if not isinstance(pred_cfg, dict):
        return
    policy = str(pred_cfg.get("selection_policy", "")).lower()
    if policy == "val_mse_candidate_channel_guarded":
        pred_cfg["selection_policy"] = "val_mse_candidate_channel"


def data_frame_to_tensor(cfg: dict[str, Any]) -> tuple[torch.Tensor, list[str]]:
    data_cfg = cfg["data"]
    path = Path(str(data_cfg["csv_path"]))
    if not path.is_absolute():
        path = ROOT / path
    df = pd.read_csv(path)
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
    raw_path = Path(str(data_cfg["csv_path"]))
    if not raw_path.is_absolute():
        raw_path = ROOT / raw_path
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


def make_source_config(
    *,
    source: str,
    horizon: int,
    out_root: Path,
    device: str,
    source_epochs: int,
) -> tuple[Path, dict[str, Any]]:
    cfg = copy.deepcopy(read_yaml(config_path(source, horizon)))
    normalize_runtime_compatible_cfg(cfg)
    out_dir = out_root / "source" / source / f"H{int(horizon)}"
    cfg.setdefault("exp", {})
    cfg["exp"]["name"] = f"input96_{source}_H{int(horizon)}_qgwnt_source"
    cfg["exp"]["out_dir"] = str(out_dir)
    cfg["exp"]["device"] = device
    cfg.setdefault("window", {})
    cfg["window"]["input_len"] = 96
    cfg["window"]["pred_len"] = int(horizon)
    cfg.setdefault("normalize", {})["train_only"] = True
    cfg.setdefault("cluster", {})["train_only"] = True
    cfg.setdefault("train", {})
    if int(source_epochs) > 0:
        cfg["train"]["epochs"] = int(source_epochs)
    cfg.setdefault("eval", {})["skip_test"] = False
    cfg["plot"] = {"enable": False}
    cfg["portrait"] = {"enable": False, "out_dir": str(out_dir / "cluster_portraits")}
    cfg["memory"] = {
        "enable": False,
        "save_checkpoint": True,
        "path": str(out_dir / "cluster_memory.pt"),
        "checkpoint_path": str(out_dir / "best_checkpoint.pt"),
    }
    path = out_root / "configs" / "source" / f"{source}_H{int(horizon)}_source.yaml"
    write_yaml(path, cfg)
    return path, cfg


def prepare_source(
    *,
    source: str,
    horizon: int,
    out_root: Path,
    device: str,
    py: str,
    source_epochs: int,
    rerun_source: bool,
) -> dict[str, Any]:
    cfg_path, cfg = make_source_config(
        source=source,
        horizon=horizon,
        out_root=out_root,
        device=device,
        source_epochs=source_epochs,
    )
    out_dir = Path(str(cfg["exp"]["out_dir"]))
    checkpoint_path = out_dir / "best_checkpoint.pt"
    memory_path = out_dir / "cluster_memory.pt"
    summary_path = out_dir / "run_summary.json"
    if rerun_source or not (checkpoint_path.exists() and summary_path.exists()):
        print(f"[source] {source} H{int(horizon)} input96", flush=True)
        run_cmd(
            [py, "-u", "-m", "src.train", "--config", str(cfg_path)],
            log_path=out_dir / "source.log",
        )
    if rerun_source or not memory_path.exists():
        norm_train_tc = normalized_train_data(cfg)
        _, channel_names = data_frame_to_tensor(cfg)
        ckpt = load_cluster_checkpoint(str(checkpoint_path), device=torch.device("cpu"))
        meta = ckpt.get("meta", {})
        if int(meta.get("input_len")) != 96 or int(meta.get("pred_len")) != int(horizon):
            raise ValueError(f"{source} H{horizon} checkpoint window mismatch: {meta.get('input_len')}/{meta.get('pred_len')}")
        cluster_id_c = meta["cluster_id_c"].to(torch.long)
        prototypes_kt = compute_cluster_prototypes(norm_train_tc, cluster_id_c)
        save_cluster_memory(
            str(memory_path),
            prototypes_kt,
            cluster_id_c,
            channel_names,
            meta={
                "kind": "input96_qgwnt_source_train_prototype",
                "source_split": "train",
                "input_len": 96,
                "pred_len": int(horizon),
                "source_config": str(cfg_path),
                "source_checkpoint": str(checkpoint_path),
            },
        )
    summary = load_json(summary_path)
    return {
        "config": cfg_path,
        "source_cfg": read_yaml(cfg_path),
        "checkpoint": checkpoint_path,
        "memory": memory_path,
        "summary": summary_path,
        "self_mse": summary.get("test", {}).get("avg_mse"),
        "self_mae": summary.get("test", {}).get("avg_mae"),
    }


def build_transfer_config(
    *,
    source: str,
    target: str,
    horizon: int,
    source_info: dict[str, Any],
    out_root: Path,
    device: str,
    batch_size: int,
    resample_method: str,
    eval_split: str,
) -> Path:
    target_cfg = read_yaml(config_path(target, horizon))
    source_cfg = source_info["source_cfg"]
    out_dir = out_root / "zero_shot" / f"{source}_to_{target}" / f"H{int(horizon)}"
    method = resample_method.lower()
    resample_enable = target in {"ETTh1", "ETTh2"} and method not in {"none", "off", "false"}
    cfg = {
        "exp": {
            "name": f"input96_{source}_to_{target}_H{int(horizon)}_transfer",
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
            "pred_len": int(horizon),
            "past_context": bool(target_cfg.get("window", {}).get("past_context", True)),
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
        "eval": {"batch_size": int(batch_size), "split": str(eval_split)},
    }
    path = out_root / "configs" / "transfer" / f"{source}_to_{target}_H{int(horizon)}_transfer.yaml"
    write_yaml(path, cfg)
    return path


def make_qgwnt_finetune_config(
    *,
    source: str,
    target: str,
    horizon: int,
    source_cfg: dict[str, Any],
    target_cfg: dict[str, Any],
    source_checkpoint: Path,
    source_memory: Path,
    fixed_cluster_id: list[int],
    out_dir: Path,
    device: str,
    batch_size: int,
    skip_test: bool,
) -> dict[str, Any]:
    cfg = copy.deepcopy(source_cfg)
    normalize_runtime_compatible_cfg(cfg)
    cfg["exp"] = {
        "name": f"input96_{source}_to_{target}_H{int(horizon)}_qgwnt_unfreeze_lr5e5_e80_{'valonly' if skip_test else 'testonce'}",
        "out_dir": str(out_dir),
        "seed": int(source_cfg.get("exp", {}).get("seed", 2026)),
        "deterministic": True,
        "device": device,
    }
    cfg["data"] = copy.deepcopy(target_cfg["data"])
    cfg.setdefault("window", {})
    cfg["window"]["input_len"] = 96
    cfg["window"]["pred_len"] = int(horizon)
    cfg["window"]["past_context"] = bool(target_cfg.get("window", {}).get("past_context", True))
    cfg["normalize"] = {"global_zscore": True, "train_only": True}
    cfg.setdefault("cluster", {})
    cfg["cluster"]["train_only"] = True
    cfg["cluster"]["fixed_cluster_id"] = [int(v) for v in fixed_cluster_id]
    cfg["corr"] = {"compute": True, "save_path": str(out_dir / "corr.npy")}
    cfg.setdefault("moe", {})
    cfg["moe"]["freeze_backbone"] = False
    cfg.setdefault("train", {})
    cfg["train"]["epochs"] = QGWNT_EPOCHS
    cfg["train"]["lr"] = QGWNT_LR
    cfg["train"]["batch_size"] = int(batch_size)
    cfg["train"].setdefault("weight_decay", 0.0001)
    cfg["train"].setdefault("selection_metric", "val_mse")
    cfg["train"]["penalty_warmup_epochs"] = min(int(cfg["train"].get("penalty_warmup_epochs", 10)), 5)
    cfg.setdefault("early_stop", {})
    cfg["early_stop"]["patience"] = int(cfg["early_stop"].get("patience", 10))
    cfg["eval"] = {"skip_test": bool(skip_test)}
    cfg["plot"] = {"enable": False}
    cfg["portrait"] = {"enable": False, "out_dir": str(out_dir / "cluster_portraits")}
    cfg["memory"] = {
        "enable": False,
        "save_checkpoint": True,
        "path": str(out_dir / "cluster_memory.pt"),
        "checkpoint_path": str(out_dir / "best_checkpoint.pt"),
    }
    finetune = {
        "enable": True,
        "checkpoint_path": str(source_checkpoint),
        "memory_path": str(source_memory),
        "cluster_map": "index",
        "strict_window": True,
        "strict_model": True,
        "load_model": True,
        "load_gate": True,
        "load_dynamic_lambda": True,
        "load_learnable_lambda": True,
    }
    if source == "ETTm1":
        finetune.update(
            {
                "partial_model_state": True,
                "load_pred_residual": True,
                "strict_pred_residual": False,
            }
        )
    cfg["finetune"] = finetune
    return cfg


def build_qgwnt_config_path(
    *,
    source: str,
    target: str,
    horizon: int,
    source_info: dict[str, Any],
    fixed_cluster_id: list[int],
    out_root: Path,
    device: str,
    batch_size: int,
    resample_method: str,
    skip_test: bool,
) -> Path:
    target_root_cfg = read_yaml(config_path(target, horizon))
    target_cfg = {
        "data": target_data_cfg_for_finetune(target_root_cfg, target, out_root, resample_method),
        "window": target_root_cfg.get("window", {}),
    }
    phase = "valonly" if skip_test else "testonce"
    out_dir = out_root / f"qgwnt_{phase}" / f"{source}_to_{target}" / f"H{int(horizon)}" / f"lr{_fmt_lr(QGWNT_LR)}"
    cfg = make_qgwnt_finetune_config(
        source=source,
        target=target,
        horizon=horizon,
        source_cfg=source_info["source_cfg"],
        target_cfg=target_cfg,
        source_checkpoint=source_info["checkpoint"],
        source_memory=source_info["memory"],
        fixed_cluster_id=fixed_cluster_id,
        out_dir=out_dir,
        device=device,
        batch_size=batch_size,
        skip_test=skip_test,
    )
    path = out_root / "configs" / "qgwnt" / f"{source}_to_{target}_H{int(horizon)}_{phase}.yaml"
    write_yaml(path, cfg)
    return path


def load_zero_cluster_id(zero_summary_path: Path) -> list[int]:
    summary = load_json(zero_summary_path)
    return [int(v) for v in summary["cluster_id"]]


def qgwnt_metrics_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    selection = summary.get("moe_residual_selection", {}) or {}
    out = {
        "val_raw_mse": summary.get("val", {}).get("avg_mse", ""),
        "val_raw_mae": summary.get("val", {}).get("avg_mae", ""),
        "val_base_mse": selection.get("val_pred_base_avg_mse", ""),
        "val_base_mae": selection.get("val_pred_base_avg_mae", ""),
        "val_scaled_mse": selection.get("val_scaled_avg_mse", ""),
        "val_scaled_mae": selection.get("val_scaled_avg_mae", ""),
        "best_epoch": json.dumps(summary.get("best_epoch", "")),
    }
    test = summary.get("test") or {}
    if test:
        out["test_mse"] = test.get("avg_mse", "")
        out["test_mae"] = test.get("avg_mae", "")
    finetune = summary.get("finetune", {}) or {}
    out["finetune_partial"] = finetune.get("partial_model_state", "")
    out["loaded_pred_residual"] = finetune.get("loaded_pred_residual", "")
    return out


def append_reused_h96_rows(rows: list[dict[str, Any]], reuse_root: Path) -> None:
    summary_csv = reuse_root / "qgwnt_other_pairs_summary.csv"
    if not summary_csv.exists():
        return
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            rows.append(
                {
                    "status": "reused",
                    "source": raw.get("source", ""),
                    "target": raw.get("target", ""),
                    "horizon": 96,
                    "input_len": 96,
                    "reused_h96": True,
                    "val_raw_mse": raw.get("val_raw_mse", ""),
                    "val_raw_mae": raw.get("val_raw_mae", ""),
                    "val_base_mse": raw.get("val_base_mse", ""),
                    "val_base_mae": raw.get("val_base_mae", ""),
                    "val_scaled_mse": raw.get("val_scaled_mse", ""),
                    "val_scaled_mae": raw.get("val_scaled_mae", ""),
                    "test_mse": raw.get("test_mse", ""),
                    "test_mae": raw.get("test_mae", ""),
                    "best_epoch": raw.get("best_epoch", ""),
                    "finetune_partial": raw.get("finetune_partial", ""),
                    "loaded_pred_residual": raw.get("loaded_pred_residual", ""),
                    "val_summary": raw.get("val_summary", ""),
                    "test_summary": raw.get("test_summary", ""),
                }
            )


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    def f4(v: Any) -> str:
        try:
            if v is None or v == "":
                return "null"
            return f"{float(v):.4f}"
        except (TypeError, ValueError):
            return str(v) if v not in (None, "") else "null"

    ok_rows = [r for r in rows if str(r.get("status", "")).lower() in {"ok", "reused"}]
    lines = [
        "# Input96 qgwnt full-horizon transfer summary",
        "",
        "Scope: input_len=96, horizons 96/192/336/720, transfer.py train-only route, qgwnt unfreeze (`freeze_backbone:false`, `lr=5e-5`, `epochs=80`, source gate kept). H96 rows are reused from the completed qgwnt probe when available.",
        "",
        "| Source | Target | H | Status | Val selected/scaled MSE/MAE | Test MSE/MAE | Route |",
        "|---|---|---:|---|---:|---:|---|",
    ]
    for row in sorted(ok_rows, key=lambda r: (str(r.get("source")), str(r.get("target")), int(r.get("horizon", 0) or 0))):
        route = row.get("zero_shot_cluster_id") or ("reused H96" if row.get("reused_h96") else "")
        lines.append(
            f"| {row.get('source')} | {row.get('target')} | {row.get('horizon')} | {row.get('status')} | "
            f"{f4(row.get('val_scaled_mse'))}/{f4(row.get('val_scaled_mae'))} | "
            f"{f4(row.get('test_mse'))}/{f4(row.get('test_mae'))} | {route} |"
        )
    error_rows = [r for r in rows if str(r.get("status", "")).lower() not in {"ok", "reused"}]
    if error_rows:
        lines.extend(["", "## Null/error rows", "", "| Source | Target | H | Error |", "|---|---|---:|---|"])
        for row in error_rows:
            error = str(row.get("error", "")).replace("\n", " ")[:500]
            lines.append(f"| {row.get('source')} | {row.get('target')} | {row.get('horizon')} | {error} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_full(args: argparse.Namespace) -> list[dict[str, Any]]:
    out_root: Path = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    horizons = _parse_ints(args.horizons)
    rows: list[dict[str, Any]] = []
    prepared: dict[tuple[str, int], dict[str, Any]] = {}

    if args.reuse_h96:
        append_reused_h96_rows(rows, args.reuse_h96_root)
        write_rows(out_root / "input96_qgwnt_full_horizon_results.csv", rows)

    run_horizons = [h for h in horizons if not (h == 96 and args.reuse_h96)]
    if args.phase in {"all", "source", "transfer", "valonly", "testonce"}:
        for source in SOURCES:
            for horizon in run_horizons:
                prepared[(source, horizon)] = prepare_source(
                    source=source,
                    horizon=horizon,
                    out_root=out_root,
                    device=args.device,
                    py=str(args.python),
                    source_epochs=args.source_epochs,
                    rerun_source=args.rerun_source,
                )

    if args.phase == "source":
        write_rows(out_root / "input96_qgwnt_full_horizon_results.csv", rows)
        return rows

    for source, targets in SOURCES.items():
        for target in targets:
            for horizon in run_horizons:
                row = {
                    "status": "ok",
                    "source": source,
                    "target": target,
                    "horizon": int(horizon),
                    "input_len": 96,
                    "reused_h96": False,
                }
                try:
                    source_info = prepared[(source, horizon)]
                    row.update(
                        {
                            "source_self_mse": source_info["self_mse"],
                            "source_self_mae": source_info["self_mae"],
                            "source_checkpoint": str(source_info["checkpoint"]),
                            "source_memory": str(source_info["memory"]),
                        }
                    )
                    transfer_cfg = build_transfer_config(
                        source=source,
                        target=target,
                        horizon=horizon,
                        source_info=source_info,
                        out_root=out_root,
                        device=args.device,
                        batch_size=args.batch_size,
                        resample_method=args.resample_method,
                        eval_split=args.transfer_eval_split,
                    )
                    transfer_out = Path(str(read_yaml(transfer_cfg)["exp"]["out_dir"]))
                    transfer_summary = transfer_out / "transfer_summary.json"
                    if args.phase in {"all", "transfer"} or args.rerun_transfer or not transfer_summary.exists():
                        print(f"[transfer.py] {source}->{target} H{int(horizon)} input96 split={args.transfer_eval_split}", flush=True)
                        run_cmd(
                            [str(args.python), "-u", "-m", "src.transfer", "--config", str(transfer_cfg)],
                            log_path=transfer_out / "transfer.log",
                        )
                    zero = load_json(transfer_summary)
                    fixed_cluster_id = [int(v) for v in zero["cluster_id"]]
                    row.update(
                        {
                            "zero_shot_eval_split": zero.get("eval_split", args.transfer_eval_split),
                            "zero_shot_mse": zero.get("avg_mse", ""),
                            "zero_shot_mae": zero.get("avg_mae", ""),
                            "zero_shot_route_uses_train_only": zero.get("route_uses_train_only", ""),
                            "zero_shot_cluster_id": json.dumps(fixed_cluster_id),
                            "zero_shot_corr_mean": zero.get("corr_mean", ""),
                        }
                    )
                    if args.phase == "transfer":
                        rows.append(row)
                        write_rows(out_root / "input96_qgwnt_full_horizon_results.csv", rows)
                        continue

                    val_cfg = build_qgwnt_config_path(
                        source=source,
                        target=target,
                        horizon=horizon,
                        source_info=source_info,
                        fixed_cluster_id=fixed_cluster_id,
                        out_root=out_root,
                        device=args.device,
                        batch_size=args.batch_size,
                        resample_method=args.resample_method,
                        skip_test=True,
                    )
                    val_out = Path(str(read_yaml(val_cfg)["exp"]["out_dir"]))
                    val_summary = val_out / "run_summary.json"
                    if args.phase in {"all", "valonly"} or args.rerun_valonly or not val_summary.exists():
                        print(f"[qgwnt val] {source}->{target} H{int(horizon)}", flush=True)
                        run_cmd(
                            [str(args.python), "-u", "-m", "src.train", "--config", str(val_cfg)],
                            log_path=val_out / "train.log",
                        )
                    val_data = load_json(val_summary)
                    row.update(qgwnt_metrics_from_summary(val_data))
                    row["val_summary"] = str(val_summary)
                    row["config_path"] = str(val_cfg)
                    row["out_dir"] = str(val_out)

                    if args.phase == "valonly":
                        rows.append(row)
                        write_rows(out_root / "input96_qgwnt_full_horizon_results.csv", rows)
                        continue

                    test_cfg = build_qgwnt_config_path(
                        source=source,
                        target=target,
                        horizon=horizon,
                        source_info=source_info,
                        fixed_cluster_id=fixed_cluster_id,
                        out_root=out_root,
                        device=args.device,
                        batch_size=args.batch_size,
                        resample_method=args.resample_method,
                        skip_test=False,
                    )
                    test_out = Path(str(read_yaml(test_cfg)["exp"]["out_dir"]))
                    test_summary = test_out / "run_summary.json"
                    if args.phase in {"all", "testonce"} or args.rerun_testonce or not test_summary.exists():
                        print(f"[qgwnt testonce] {source}->{target} H{int(horizon)}", flush=True)
                        run_cmd(
                            [str(args.python), "-u", "-m", "src.train", "--config", str(test_cfg)],
                            log_path=test_out / "train.log",
                        )
                    test_data = load_json(test_summary)
                    row.update(qgwnt_metrics_from_summary(test_data))
                    row["test_summary"] = str(test_summary)
                    row["config_path"] = str(test_cfg)
                    row["out_dir"] = str(test_out)
                except Exception as exc:
                    row["status"] = "error"
                    row["error"] = str(exc)[-4000:]
                rows.append(row)
                write_rows(out_root / "input96_qgwnt_full_horizon_results.csv", rows)
                write_markdown(out_root / "input96_qgwnt_full_horizon_summary.md", rows)
    write_rows(out_root / "input96_qgwnt_full_horizon_results.csv", rows)
    write_markdown(out_root / "input96_qgwnt_full_horizon_summary.md", rows)
    write_json(
        out_root / "manifest.json",
        {
            "input_len": 96,
            "horizons": horizons,
            "sources": SOURCES,
            "qgwnt": {"freeze_backbone": False, "lr": QGWNT_LR, "epochs": QGWNT_EPOCHS},
            "transfer_eval_split": args.transfer_eval_split,
            "reuse_h96": bool(args.reuse_h96),
        },
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run input96 qgwnt transfer for all ETT horizons.")
    parser.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "input96_transfer_qgwnt_full_horizon")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--horizons", default="96,192,336,720")
    parser.add_argument("--phase", choices=["all", "source", "transfer", "valonly", "testonce"], default="all")
    parser.add_argument(
        "--source-epochs",
        type=int,
        default=0,
        help="Override source train epochs only when positive; default preserves each horizon config.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--resample-method", default="last", choices=["last", "ffill", "linear", "mean", "none"])
    parser.add_argument("--transfer-eval-split", default="val", choices=["val", "test"])
    parser.add_argument("--reuse-h96", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-h96-root", type=Path, default=ROOT / "outputs" / "input96_transfer_qgwnt_probe")
    parser.add_argument("--rerun-source", action="store_true")
    parser.add_argument("--rerun-transfer", action="store_true")
    parser.add_argument("--rerun-valonly", action="store_true")
    parser.add_argument("--rerun-testonce", action="store_true")
    args = parser.parse_args()
    run_full(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
