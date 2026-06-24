import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.console_progress import PurpleProgressBar

DEFAULT_HORIZONS = [96, 192, 336, 720]
BANDWIDTH_HORIZONS = [24, 48, 96, 168]
BANDWIDTH_INPUT_LEN = 24
ETTH1_PAPER_NORM_PRED_LEN = 96
ETTH1_PAPER_NORM_MAX_ROWS = 14400
ETTH1_PAPER_NORM_INPUT_LEN = 336
TSL_EXTERNAL_SPLIT = {
    "electricity": (0.6999695863746959, 0.10006082725060828, 0.19996958637469586),
    "weather": (0.6999962046455139, 0.10000759070897222, 0.1999962046455139),
    "traffic": (0.7, 0.1, 0.2),
}
TSL_ETT_SPLIT = {
    "ETTh1": ("data/ETTh1.csv", 14400, 0.6, 0.2, 0.2),
    "ETTh2": ("data/ETTh2.csv", 14400, 0.6, 0.2, 0.2),
    "ETTm1": ("data/ETTm1.csv", 57600, 0.6, 0.2, 0.2),
    "ETTm2": ("data/ETTm2.csv", 57600, 0.6, 0.2, 0.2),
}
RESULT_FIELDS = [
    "status",
    "dataset",
    "pred_len",
    "selected_variant",
    "selected_test_avg_mae",
    "selected_test_avg_mse",
    "test_best_variant",
    "test_best_avg_mae",
    "test_best_avg_mse",
    "test_best_delta_mse_vs_base",
    "test_avg_mae",
    "test_avg_mse",
    "test_hybrid_avg_mae",
    "test_hybrid_avg_mse",
    "val_avg_mae",
    "val_avg_mse",
    "val_hybrid_avg_mae",
    "val_hybrid_avg_mse",
    "val_avg_loss",
    "val_hybrid_avg_loss",
    "val_hybrid_confidence",
    "val_hybrid_effective_alpha",
    "test_hybrid_confidence",
    "test_hybrid_effective_alpha",
    "data_csv",
    "base_config",
    "run_config",
    "out_dir",
    "best_epoch",
    "total_sec",
    "avg_epoch_sec",
    "wrapper_sec",
    "returncode",
    "error",
]


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, cfg: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=False, sort_keys=False)


def is_train_dataset_config(path: Path) -> bool:
    try:
        cfg = load_yaml(path)
    except Exception:
        return False
    required = {"exp", "data", "window", "normalize", "corr", "cluster", "model", "moe", "penalties", "train"}
    if not required.issubset(cfg.keys()):
        return False
    if "source" in cfg or "transfer" in cfg:
        return False
    data_csv = cfg.get("data", {}).get("csv_path")
    return bool(data_csv)


def discover_configs(config_dir: Path) -> List[Path]:
    return sorted(path for path in config_dir.glob("*.yaml") if is_train_dataset_config(path))


def config_dataset_name(path: Path, cfg: Dict[str, Any], *, prefer_stem: bool = False) -> str:
    if prefer_stem:
        return path.stem
    out_dir = str(cfg.get("exp", {}).get("out_dir", "")).strip().replace("\\", "/")
    if out_dir:
        name = Path(out_dir).name
        if name:
            return name
    return path.stem


def is_bandwidth_config(path: Path, dataset: str, cfg: Dict[str, Any]) -> bool:
    data_csv = str(cfg.get("data", {}).get("csv_path", ""))
    tokens = [dataset, path.stem, Path(data_csv).stem]
    return any("bandwidth" in token.lower() for token in tokens if token)


def horizons_for_config(
    path: Path,
    dataset: str,
    cfg: Dict[str, Any],
    requested_horizons: List[int] | None,
) -> List[int]:
    if requested_horizons is not None:
        return [int(h) for h in requested_horizons]
    if is_bandwidth_config(path, dataset, cfg):
        return BANDWIDTH_HORIZONS[:]
    return DEFAULT_HORIZONS[:]


def apply_etth1_paper_norm_96(cfg: Dict[str, Any]) -> None:
    cfg.setdefault("data", {})
    cfg["data"]["csv_path"] = "data/ETTh1.csv"
    cfg["data"]["date_col"] = 0
    cfg["data"]["max_rows"] = ETTH1_PAPER_NORM_MAX_ROWS
    cfg["data"]["train_ratio"] = 0.6
    cfg["data"]["val_ratio"] = 0.2
    cfg["data"]["test_ratio"] = 0.2

    cfg.setdefault("window", {})
    cfg["window"]["input_len"] = ETTH1_PAPER_NORM_INPUT_LEN
    cfg["window"]["pred_len"] = ETTH1_PAPER_NORM_PRED_LEN
    cfg["window"]["past_context"] = True

    cfg.setdefault("normalize", {})
    cfg["normalize"]["global_zscore"] = True
    cfg["normalize"]["train_only"] = False

    cfg.setdefault("model", {})
    cfg["model"]["predictor"] = "dlinear"
    cfg["model"]["hidden_dim"] = int(cfg["model"].get("hidden_dim", 128))
    cfg["model"]["dropout"] = 0.0
    cfg["model"]["dlinear_kernel_size"] = int(cfg["model"].get("dlinear_kernel_size", 25))
    cfg["model"]["channel_adapter"] = {
        "enable": False,
        "rank": 8,
        "init": "zero_delta",
        "scale": 1.0,
    }

def apply_tsl_alignment(cfg: Dict[str, Any], dataset: str) -> None:
    """Apply the data protocol used by the TSL-aligned main tables."""
    cfg.setdefault("data", {})
    cfg["data"]["date_col"] = 0
    data_stem = Path(str(cfg["data"].get("csv_path", ""))).stem
    dataset_key = dataset if dataset in TSL_ETT_SPLIT else data_stem
    if dataset_key in TSL_ETT_SPLIT:
        csv_path, max_rows, train_ratio, val_ratio, test_ratio = TSL_ETT_SPLIT[dataset_key]
        cfg["data"]["csv_path"] = csv_path
        cfg["data"]["max_rows"] = max_rows
        cfg["data"]["train_ratio"] = train_ratio
        cfg["data"]["val_ratio"] = val_ratio
        cfg["data"]["test_ratio"] = test_ratio
    else:
        key = dataset.lower()
        if key not in TSL_EXTERNAL_SPLIT:
            key = data_stem.lower()
        if key in TSL_EXTERNAL_SPLIT:
            train_ratio, val_ratio, test_ratio = TSL_EXTERNAL_SPLIT[key]
            cfg["data"]["max_rows"] = 0
            cfg["data"]["train_ratio"] = train_ratio
            cfg["data"]["val_ratio"] = val_ratio
            cfg["data"]["test_ratio"] = test_ratio

    cfg.setdefault("normalize", {})
    cfg["normalize"]["global_zscore"] = True
    cfg["normalize"]["train_only"] = True

    cfg.setdefault("cluster", {})
    cfg["cluster"]["train_only"] = True


def ensure_local_output_paths(
    cfg: Dict[str, Any],
    out_dir: Path,
    run_name: str,
    keep_artifacts: bool,
) -> None:
    cfg.setdefault("exp", {})
    cfg["exp"]["name"] = run_name
    cfg["exp"]["out_dir"] = str(out_dir)

    cfg.setdefault("corr", {})
    cfg["corr"]["save_path"] = str(out_dir / "corr.npy")

    cfg.setdefault("plot", {})
    if not keep_artifacts:
        cfg["plot"]["enable"] = False

    cfg.setdefault("portrait", {})
    cfg["portrait"]["out_dir"] = str(out_dir / "cluster_portraits")
    if not keep_artifacts:
        cfg["portrait"]["enable"] = False

    cfg.setdefault("memory", {})
    cfg["memory"]["path"] = str(out_dir / "cluster_memory.pt")
    cfg["memory"]["checkpoint_path"] = str(out_dir / "best_checkpoint.pt")
    if not keep_artifacts:
        cfg["memory"]["enable"] = False
        cfg["memory"]["save_checkpoint"] = False

def make_run_config(
    base_cfg: Dict[str, Any],
    dataset: str,
    pred_len: int,
    input_len: int | None,
    batch_size: int | None,
    out_root: Path,
    epochs: int | None,
    train_lr: float | None,
    train_weight_decay: float | None,
    early_patience: int | None,
    lr_scheduler_patience: int | None,
    lr_scheduler_factor: float | None,
    mae_objective_weight: float | None,
    device: str | None,
    keep_artifacts: bool,
    etth1_paper_norm_96: bool,
    tsl_align: bool,
) -> tuple[Dict[str, Any], Path]:
    cfg = copy.deepcopy(base_cfg)
    if tsl_align:
        apply_tsl_alignment(cfg, dataset)
    cfg.setdefault("window", {})
    if input_len is not None:
        cfg["window"]["input_len"] = int(input_len)
    cfg["window"]["pred_len"] = int(pred_len)
    if batch_size is not None:
        cfg.setdefault("train", {})
        cfg["train"]["batch_size"] = int(batch_size)
    if epochs is not None:
        cfg.setdefault("train", {})
        cfg["train"]["epochs"] = int(epochs)
    if train_lr is not None:
        cfg.setdefault("train", {})
        cfg["train"]["lr"] = float(train_lr)
    if train_weight_decay is not None:
        cfg.setdefault("train", {})
        cfg["train"]["weight_decay"] = float(train_weight_decay)
    if early_patience is not None:
        cfg.setdefault("early_stop", {})
        cfg["early_stop"]["patience"] = int(early_patience)
    if lr_scheduler_patience is not None:
        cfg.setdefault("train", {}).setdefault("lr_scheduler", {})
        cfg["train"]["lr_scheduler"]["patience"] = int(lr_scheduler_patience)
    if lr_scheduler_factor is not None:
        cfg.setdefault("train", {}).setdefault("lr_scheduler", {})
        cfg["train"]["lr_scheduler"]["factor"] = float(lr_scheduler_factor)
    if mae_objective_weight is not None:
        cfg.setdefault("train", {}).setdefault("mae_objective", {})
        cfg["train"]["mae_objective"]["weight"] = float(mae_objective_weight)
    if device:
        cfg.setdefault("exp", {})
        cfg["exp"]["device"] = device
    if etth1_paper_norm_96 and dataset == "ETTh1" and int(pred_len) == ETTH1_PAPER_NORM_PRED_LEN:
        apply_etth1_paper_norm_96(cfg)

    run_name = f"{dataset}_pred_{pred_len}"
    out_dir = out_root / dataset / f"pred_{pred_len}"
    ensure_local_output_paths(
        cfg,
        out_dir=out_dir,
        run_name=run_name,
        keep_artifacts=keep_artifacts,
    )
    return cfg, out_dir


def read_tail(path: Path, max_chars: int = 2000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:].strip()


def optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_selection_policy(value: Any) -> str:
    policy = str(value or "hybrid").lower()
    if policy == "val_mse":
        policy = "val_mse_margin"
    if policy not in {"hybrid", "val_mse_margin", "val_mae_guarded", "base"}:
        policy = "hybrid"
    return policy


def summary_to_row(
    summary_path: Path,
    *,
    status: str,
    dataset: str,
    pred_len: int,
    data_csv: str,
    base_config: Path,
    run_config: Path,
    out_dir: Path,
    wrapper_sec: float,
    returncode: int,
    selection_policy: str | None = None,
    error: str = "",
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "status": status,
        "dataset": dataset,
        "pred_len": int(pred_len),
        "data_csv": data_csv,
        "base_config": str(base_config),
        "run_config": str(run_config),
        "out_dir": str(out_dir),
        "test_avg_mae": "",
        "test_avg_mse": "",
        "val_avg_loss": "",
        "val_avg_mae": "",
        "val_avg_mse": "",
        "val_hybrid_avg_loss": "",
        "val_hybrid_avg_mae": "",
        "val_hybrid_avg_mse": "",
        "val_hybrid_confidence": "",
        "val_hybrid_effective_alpha": "",
        "test_hybrid_avg_mae": "",
        "test_hybrid_avg_mse": "",
        "test_hybrid_confidence": "",
        "test_hybrid_effective_alpha": "",
        "selected_variant": "base",
        "selected_test_avg_mae": "",
        "selected_test_avg_mse": "",
        "test_best_variant": "base",
        "test_best_avg_mae": "",
        "test_best_avg_mse": "",
        "test_best_delta_mse_vs_base": "",
        "best_epoch": "",
        "total_sec": "",
        "avg_epoch_sec": "",
        "wrapper_sec": wrapper_sec,
        "returncode": int(returncode),
        "error": error,
    }
    if not summary_path.exists():
        return row

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    val_info = summary.get("val") or {}
    val_hybrid_info = summary.get("val_hybrid") or {}
    val_hybrid_conf = summary.get("val_hybrid_confidence") or {}
    test_info = summary.get("test") or {}
    hybrid_info = summary.get("test_hybrid") or {}
    test_hybrid_conf = summary.get("test_hybrid_confidence") or {}
    selected_info = summary.get("selected") or {}
    timing = summary.get("timing") or {}
    row.update(
        {
            "test_avg_mae": test_info.get("avg_mae", ""),
            "test_avg_mse": test_info.get("avg_mse", ""),
            "val_avg_loss": val_info.get("avg_loss", ""),
            "val_avg_mae": val_info.get("avg_mae", ""),
            "val_avg_mse": val_info.get("avg_mse", ""),
            "val_hybrid_avg_loss": val_hybrid_info.get("avg_loss", ""),
            "val_hybrid_avg_mae": val_hybrid_info.get("avg_mae", ""),
            "val_hybrid_avg_mse": val_hybrid_info.get("avg_mse", ""),
            "val_hybrid_confidence": val_hybrid_conf.get("mean_confidence", ""),
            "val_hybrid_effective_alpha": val_hybrid_conf.get("mean_effective_alpha", ""),
            "test_hybrid_avg_mae": hybrid_info.get("avg_mae", ""),
            "test_hybrid_avg_mse": hybrid_info.get("avg_mse", ""),
            "test_hybrid_confidence": test_hybrid_conf.get("mean_confidence", ""),
            "test_hybrid_effective_alpha": test_hybrid_conf.get("mean_effective_alpha", ""),
            "best_epoch": ",".join(str(v) for v in summary.get("best_epoch", [])),
            "total_sec": timing.get("total_sec", ""),
            "avg_epoch_sec": timing.get("avg_epoch_sec", ""),
        }
    )
    selected_policy = normalize_selection_policy(
        selection_policy if selection_policy is not None else selected_info.get("selection_policy", "hybrid")
    )
    base_test_mae = optional_float(row["test_avg_mae"])
    base_test_mse = optional_float(row["test_avg_mse"])
    hybrid_test_mae = optional_float(row["test_hybrid_avg_mae"])
    hybrid_test_mse = optional_float(row["test_hybrid_avg_mse"])
    row["test_best_variant"] = "base"
    row["test_best_avg_mae"] = row["test_avg_mae"]
    row["test_best_avg_mse"] = row["test_avg_mse"]
    row["test_best_delta_mse_vs_base"] = 0.0 if base_test_mse is not None else ""
    if (
        base_test_mse is not None
        and hybrid_test_mse is not None
        and hybrid_test_mse < base_test_mse
        and hybrid_test_mae is not None
    ):
        row["test_best_variant"] = "hybrid"
        row["test_best_avg_mae"] = row["test_hybrid_avg_mae"]
        row["test_best_avg_mse"] = row["test_hybrid_avg_mse"]
        row["test_best_delta_mse_vs_base"] = hybrid_test_mse - base_test_mse

    row["selected_variant"] = "base"
    row["selected_test_avg_mae"] = row["test_avg_mae"]
    row["selected_test_avg_mse"] = row["test_avg_mse"]
    if selected_policy == "hybrid":
        if hybrid_test_mae is not None and hybrid_test_mse is not None:
            row["selected_variant"] = "hybrid"
            row["selected_test_avg_mae"] = row["test_hybrid_avg_mae"]
            row["selected_test_avg_mse"] = row["test_hybrid_avg_mse"]
    elif selected_policy in {"val_mse_margin", "val_mae_guarded"}:
        selected_variant = str(selected_info.get("variant", "")).lower()
        selected_mae = selected_info.get("avg_mae", "")
        selected_mse = selected_info.get("avg_mse", "")
        if selected_variant in {"base", "hybrid"} and selected_mae != "" and selected_mse != "":
            row["selected_variant"] = selected_variant
            row["selected_test_avg_mae"] = selected_mae
            row["selected_test_avg_mse"] = selected_mse
            return row
        if selected_policy == "val_mae_guarded":
            base_val_mae = optional_float(row["val_avg_mae"])
            hybrid_val_mae = optional_float(row["val_hybrid_avg_mae"])
            base_val_mse = optional_float(row["val_avg_mse"])
            hybrid_val_mse = optional_float(row["val_hybrid_avg_mse"])
            if (
                base_val_mae is not None
                and hybrid_val_mae is not None
                and base_val_mse is not None
                and hybrid_val_mse is not None
                and hybrid_val_mae < base_val_mae
                and (hybrid_val_mse - base_val_mse) <= 0.03 * max(abs(base_val_mse), 1.0e-12)
                and hybrid_test_mae is not None
                and hybrid_test_mse is not None
            ):
                row["selected_variant"] = "hybrid"
                row["selected_test_avg_mae"] = row["test_hybrid_avg_mae"]
                row["selected_test_avg_mse"] = row["test_hybrid_avg_mse"]
            return row
        base_val_mse = optional_float(row["val_avg_mse"])
        hybrid_val_mse = optional_float(row["val_hybrid_avg_mse"])
        hybrid_test_mae = optional_float(row["test_hybrid_avg_mae"])
        hybrid_test_mse = optional_float(row["test_hybrid_avg_mse"])
        if (
            base_val_mse is not None
            and hybrid_val_mse is not None
            and hybrid_val_mse < base_val_mse
            and hybrid_test_mae is not None
            and hybrid_test_mse is not None
        ):
            row["selected_variant"] = "hybrid"
            row["selected_test_avg_mae"] = row["test_hybrid_avg_mae"]
            row["selected_test_avg_mse"] = row["test_hybrid_avg_mse"]
    return row


def _norm_bool(value: Any) -> bool:
    return bool(value)


def _norm_str(value: Any, default: str = "") -> str:
    return str(default if value is None else value).strip().lower()


def _norm_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(default if value is None else value)
    except (TypeError, ValueError):
        return float(default)


def summary_matches_generated_config(summary_path: Path, cfg: Dict[str, Any]) -> bool:
    """Avoid reusing summaries produced by an older generated config."""
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    train_cfg = cfg.get("train", {}) or {}
    summary_mae = summary.get("mae_objective")
    cfg_mae = train_cfg.get("mae_objective", {}) or {}
    cfg_mae_enabled = _norm_bool(cfg_mae.get("enable", False))
    summary_mae_enabled = _norm_bool((summary_mae or {}).get("enable", False)) if summary_mae is not None else False
    if cfg_mae_enabled != summary_mae_enabled:
        return False
    if cfg_mae_enabled:
        if _norm_str((summary_mae or {}).get("kind", "l1"), "l1") != _norm_str(cfg_mae.get("kind", "l1"), "l1"):
            return False
        if abs(_norm_float((summary_mae or {}).get("weight", 0.0)) - _norm_float(cfg_mae.get("weight", 0.0))) > 1.0e-12:
            return False

    return True


def append_result(csv_path: Path, row: Dict[str, Any], append: bool) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists() and append
    mode = "a" if append else "w"
    with csv_path.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})


def normalize_result_row(row: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {field: row.get(field, "") for field in RESULT_FIELDS}
    if normalized.get("selected_variant", "") == "":
        normalized["selected_variant"] = "base"
    if normalized.get("selected_test_avg_mae", "") == "":
        normalized["selected_test_avg_mae"] = normalized.get("test_avg_mae", "")
    if normalized.get("selected_test_avg_mse", "") == "":
        normalized["selected_test_avg_mse"] = normalized.get("test_avg_mse", "")
    base_test_mse = optional_float(normalized.get("test_avg_mse", ""))
    hybrid_test_mse = optional_float(normalized.get("test_hybrid_avg_mse", ""))
    if normalized.get("test_best_variant", "") == "":
        normalized["test_best_variant"] = "base"
    if normalized.get("test_best_avg_mae", "") == "":
        normalized["test_best_avg_mae"] = normalized.get("test_avg_mae", "")
    if normalized.get("test_best_avg_mse", "") == "":
        normalized["test_best_avg_mse"] = normalized.get("test_avg_mse", "")
    if normalized.get("test_best_delta_mse_vs_base", "") == "":
        normalized["test_best_delta_mse_vs_base"] = 0.0 if base_test_mse is not None else ""
    if (
        base_test_mse is not None
        and hybrid_test_mse is not None
        and hybrid_test_mse < base_test_mse
    ):
        normalized["test_best_variant"] = "hybrid"
        normalized["test_best_avg_mae"] = normalized.get("test_hybrid_avg_mae", "")
        normalized["test_best_avg_mse"] = normalized.get("test_hybrid_avg_mse", "")
        normalized["test_best_delta_mse_vs_base"] = hybrid_test_mse - base_test_mse
    return normalized


def read_existing_results(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [normalize_result_row(dict(row)) for row in reader]


def write_all_results(csv_path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(normalize_result_row(row))


def write_wide_results(csv_path: Path, rows: List[Dict[str, Any]], horizons: List[int]) -> None:
    horizon_set = {int(v) for v in horizons}
    for row in rows:
        try:
            horizon_set.add(int(row.get("pred_len", 0)))
        except (TypeError, ValueError):
            continue
    horizons = sorted(v for v in horizon_set if v > 0)

    fields = ["dataset"]
    for horizon in horizons:
        fields.extend([
            f"selected_mae_{horizon}",
            f"selected_mse_{horizon}",
            f"test_best_mae_{horizon}",
            f"test_best_mse_{horizon}",
            f"base_mae_{horizon}",
            f"base_mse_{horizon}",
            f"hybrid_mae_{horizon}",
            f"hybrid_mse_{horizon}",
            f"mae_{horizon}",
            f"mse_{horizon}",
            f"variant_{horizon}",
            f"test_best_variant_{horizon}",
            f"status_{horizon}",
        ])
    fields.append("data_csv")
    for horizon in horizons:
        fields.extend([
            f"out_dir_{horizon}",
        ])

    grouped: Dict[str, Dict[str, Any]] = {}
    dataset_order: List[str] = []
    for row in rows:
        dataset = str(row["dataset"])
        if dataset not in grouped:
            dataset_order.append(dataset)
            grouped[dataset] = {"dataset": dataset, "data_csv": row.get("data_csv", "")}
        entry = grouped[dataset]
        horizon = int(row["pred_len"])
        entry[f"mae_{horizon}"] = row.get("selected_test_avg_mae", "")
        entry[f"mse_{horizon}"] = row.get("selected_test_avg_mse", "")
        entry[f"selected_mae_{horizon}"] = row.get("selected_test_avg_mae", "")
        entry[f"selected_mse_{horizon}"] = row.get("selected_test_avg_mse", "")
        entry[f"test_best_mae_{horizon}"] = row.get("test_best_avg_mae", "")
        entry[f"test_best_mse_{horizon}"] = row.get("test_best_avg_mse", "")
        entry[f"base_mae_{horizon}"] = row.get("test_avg_mae", "")
        entry[f"base_mse_{horizon}"] = row.get("test_avg_mse", "")
        entry[f"hybrid_mae_{horizon}"] = row.get("test_hybrid_avg_mae", "")
        entry[f"hybrid_mse_{horizon}"] = row.get("test_hybrid_avg_mse", "")
        entry[f"variant_{horizon}"] = row.get("selected_variant", "")
        entry[f"test_best_variant_{horizon}"] = row.get("test_best_variant", "")
        entry[f"status_{horizon}"] = row.get("status", "")
        entry[f"out_dir_{horizon}"] = row.get("out_dir", "")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for dataset in dataset_order:
            writer.writerow({field: grouped[dataset].get(field, "") for field in fields})


def result_key(row: Dict[str, Any]) -> tuple[str, int]:
    return str(row.get("dataset", "")), int(row.get("pred_len", 0))


def upsert_result_row(rows: List[Dict[str, Any]], row: Dict[str, Any]) -> None:
    key = result_key(row)
    normalized = normalize_result_row(row)
    for idx, existing in enumerate(rows):
        if result_key(existing) == key:
            rows[idx] = normalized
            return
    rows.append(normalized)


def run_train(config_path: Path, out_dir: Path, show_child_progress: bool) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = out_dir / "stdout.log"
    stderr_path = out_dir / "stderr.log"
    cmd = [sys.executable, "-m", "src.train", "--config", str(config_path)]
    env = os.environ.copy()
    env["MOELOSS_PROGRESS_LEAVE"] = "0"
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if show_child_progress:
        stdout_path.write_text(
            "stdout was streamed to the interactive console to show the child progress bar.\n",
            encoding="utf-8",
        )
        with stderr_path.open("w", encoding="utf-8") as stderr_f:
            completed = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                text=True,
                stderr=stderr_f,
                env=env,
            )
    else:
        with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
            completed = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                text=True,
                stdout=stdout_f,
                stderr=stderr_f,
                env=env,
            )
    return int(completed.returncode)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Serially run all dataset configs for multiple prediction horizons and collect MAE/MSE to CSV."
    )
    ap.add_argument("--config-dir", type=str, default="configs", help="Directory used for automatic config discovery.")
    ap.add_argument("--configs", type=str, nargs="*", default=None, help="Explicit config list. Overrides discovery.")
    ap.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Prediction horizons. Defaults to 96 192 336 720 for normal datasets; "
            "bandwidth configs default to 24 48 96 168."
        ),
    )
    ap.add_argument("--out-root", type=str, default="outputs/all_datasets_horizons")
    ap.add_argument("--results-csv", type=str, default=None)
    ap.add_argument("--wide-csv", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=None, help="Optional epoch override for every run.")
    ap.add_argument("--input-len", type=int, default=None, help="Optional input length override for every run.")
    ap.add_argument("--batch-size", type=int, default=None, help="Optional train batch size override for every run.")
    ap.add_argument("--train-lr", type=float, default=None, help="Optional train.lr override for every run.")
    ap.add_argument("--train-weight-decay", type=float, default=None, help="Optional train.weight_decay override for every run.")
    ap.add_argument("--early-patience", type=int, default=None, help="Optional early_stop.patience override for every run.")
    ap.add_argument("--lr-scheduler-patience", type=int, default=None, help="Optional train.lr_scheduler.patience override.")
    ap.add_argument("--lr-scheduler-factor", type=float, default=None, help="Optional train.lr_scheduler.factor override.")
    ap.add_argument("--mae-objective-weight", type=float, default=None, help="Optional train.mae_objective.weight override.")
    ap.add_argument("--device", type=str, default=None, help="Optional device override, e.g. cuda:0 or cpu.")
    ap.add_argument("--reuse-existing", action="store_true", help="Read an existing run_summary.json instead of rerunning.")
    ap.add_argument("--dry-run", action="store_true", help="Only write generated configs and planned CSV rows.")
    ap.add_argument("--append", action="store_true", help="Deprecated; results are updated by dataset/horizon by default.")
    ap.add_argument("--no-preserve-results", action="store_true", help="Rebuild results CSVs from only the runs in this invocation.")
    ap.add_argument("--stop-on-error", action="store_true", help="Stop after the first failed training run.")
    ap.add_argument("--keep-artifacts", action="store_true", help="Keep plot, portrait, memory and checkpoint outputs enabled.")
    ap.add_argument(
        "--tsl-align",
        action="store_true",
        help=(
            "Force the TSL-aligned data protocol: ETT uses the standard 0.6/0.2/0.2 splits "
            "with max_rows 14400/57600, Electricity/Weather/Traffic use 0.7/0.1/0.2, "
            "and normalization/cluster fitting use train only."
        ),
    )
    ap.add_argument(
        "--etth1-paper-norm-96",
        action="store_true",
        help=(
            "Opt in to the ETTh1 H=96 paper-normalization preset. "
            "By default ETTh1 batch runs inherit configs/ETTh1.yaml."
        ),
    )
    ap.add_argument(
        type=str,
        default=None,
        choices=["none", "agreement", "distance", "confidence", "distance_agreement"],
    )
    ap.add_argument(
        type=str,
        default=None,
        choices=["hybrid", "val_mse_margin", "val_mae_guarded", "val_mse", "base"],
    )
    ap.add_argument(
        type=float,
        default=None,
        help="Require this relative validation improvement before selecting hybrid, e.g. 0.01 for 1%%.",
    )
    ap.add_argument(
        type=float,
        default=None,
        help="Require this absolute validation improvement before selecting hybrid.",
    )
    ap.add_argument("--no-child-progress", action="store_true", help="Keep child training stdout in logs instead of showing its progress bar.")
    ap.add_argument(
        "--dataset-name-from-stem",
        action="store_true",
        help="Use each config filename stem as the dataset name instead of inferring it from exp.out_dir.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_root = resolve_path(args.out_root)
    results_csv = resolve_path(args.results_csv) if args.results_csv else out_root / "results.csv"
    wide_csv = resolve_path(args.wide_csv) if args.wide_csv else out_root / "results_wide.csv"
    config_paths = (
        [resolve_path(path) for path in args.configs]
        if args.configs is not None and len(args.configs) > 0
        else discover_configs(resolve_path(args.config_dir))
    )
    if not config_paths:
        raise SystemExit("No dataset training configs found.")

    if results_csv.exists() and not args.no_preserve_results:
        rows = read_existing_results(results_csv)
        # Normalize old headers before updating rows so long/wide CSVs use the
        # same selected_* columns and cannot drift by column position.
        write_all_results(results_csv, rows)
    else:
        rows = []
        if results_csv.exists() and args.no_preserve_results:
            results_csv.unlink()

    config_items = []
    for base_config in config_paths:
        base_cfg = load_yaml(base_config)
        dataset = config_dataset_name(base_config, base_cfg, prefer_stem=bool(args.dataset_name_from_stem))
        data_csv = str(base_cfg.get("data", {}).get("csv_path", ""))
        bandwidth_special = is_bandwidth_config(base_config, dataset, base_cfg)
        horizons = horizons_for_config(base_config, dataset, base_cfg, args.horizons)
        config_items.append((base_config, base_cfg, dataset, data_csv, bandwidth_special, horizons))

    total_runs = sum(len(item[-1]) for item in config_items)
    active_horizons = sorted({int(h) for item in config_items for h in item[-1]})
    run_idx = 0
    batch_progress = PurpleProgressBar(total=total_runs, label="Batch train", unit="run")
    show_child_progress = bool(sys.stdout.isatty() and not args.no_child_progress)
    for base_config, base_cfg, dataset, data_csv, bandwidth_special, horizons in config_items:
        for horizon in horizons:
            run_idx += 1
            input_len = args.input_len if args.input_len is not None else (BANDWIDTH_INPUT_LEN if bandwidth_special else None)
            keep_artifacts_for_run = bool(args.keep_artifacts or bandwidth_special)
            cfg, out_dir = make_run_config(
                base_cfg,
                dataset=dataset,
                pred_len=int(horizon),
                input_len=input_len,
                batch_size=args.batch_size,
                out_root=out_root,
                epochs=args.epochs,
                train_lr=args.train_lr,
                train_weight_decay=args.train_weight_decay,
                early_patience=args.early_patience,
                lr_scheduler_patience=args.lr_scheduler_patience,
                lr_scheduler_factor=args.lr_scheduler_factor,
                mae_objective_weight=args.mae_objective_weight,
                device=args.device,
                keep_artifacts=keep_artifacts_for_run,
                etth1_paper_norm_96=bool(args.etth1_paper_norm_96),
                tsl_align=bool(args.tsl_align),
            )
            config_path = out_root / "configs" / f"{dataset}_pred_{horizon}.yaml"
            write_yaml(config_path, cfg)
            summary_path = out_dir / "run_summary.json"

            if batch_progress.enabled:
                batch_progress.update(
                    run_idx - 1,
                    suffix=f"running {dataset} H={horizon}",
                    force=True,
                )
            else:
                print(f"[{run_idx}/{total_runs}] {dataset} pred_len={horizon}")
            t0 = time.perf_counter()
            if args.dry_run:
                row = summary_to_row(
                    summary_path,
                    status="prepared",
                    dataset=dataset,
                    pred_len=int(horizon),
                    data_csv=data_csv,
                    base_config=base_config,
                    run_config=config_path,
                    out_dir=out_dir,
                    wrapper_sec=0.0,
                    returncode=0,
                )
            elif args.reuse_existing and summary_matches_generated_config(summary_path, cfg):
                row = summary_to_row(
                    summary_path,
                    status="reused",
                    dataset=dataset,
                    pred_len=int(horizon),
                    data_csv=data_csv,
                    base_config=base_config,
                    run_config=config_path,
                    out_dir=out_dir,
                    wrapper_sec=0.0,
                    returncode=0,
                )
            else:
                returncode = run_train(config_path, out_dir, show_child_progress=show_child_progress)
                wrapper_sec = time.perf_counter() - t0
                error = ""
                status = "ok"
                if returncode != 0:
                    status = "failed"
                    error = read_tail(out_dir / "stderr.log") or read_tail(out_dir / "stdout.log")
                elif not summary_path.exists():
                    status = "failed"
                    error = "run_summary.json not found"

                row = summary_to_row(
                    summary_path,
                    status=status,
                    dataset=dataset,
                    pred_len=int(horizon),
                    data_csv=data_csv,
                    base_config=base_config,
                    run_config=config_path,
                    out_dir=out_dir,
                    wrapper_sec=wrapper_sec,
                    returncode=returncode,
                    error=error,
                )

            upsert_result_row(rows, row)
            write_all_results(results_csv, rows)
            write_wide_results(wide_csv, rows, active_horizons)
            progress_suffix = (
                f"{row['status']} {dataset} H={horizon} "
                f"selected={row.get('selected_variant', '')} "
                f"mae={row.get('selected_test_avg_mae', '')} "
                f"mse={row.get('selected_test_avg_mse', '')}"
            )
            if batch_progress.enabled:
                batch_progress.update(run_idx, suffix=progress_suffix, force=True)
            else:
                print(
                    f"  -> {row['status']} "
                    f"selected={row.get('selected_variant', '')} "
                    f"mae={row.get('selected_test_avg_mae', '')} "
                    f"mse={row.get('selected_test_avg_mse', '')} "
                    f"out={out_dir}"
                )
            if row["status"] == "failed" and args.stop_on_error:
                batch_progress.finish(current=run_idx, suffix=progress_suffix)
                raise SystemExit(f"Stopped after failed run: {dataset} pred_len={horizon}")

    batch_progress.finish(current=total_runs, suffix="done")
    write_all_results(results_csv, rows)
    write_wide_results(wide_csv, rows, active_horizons)
    print(f"Saved long results to: {results_csv}")
    print(f"Saved wide results to: {wide_csv}")


if __name__ == "__main__":
    main()
