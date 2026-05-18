from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[1]

DATASET = "ETTm1"
PRED_LEN = 96
INPUT_LEN = 336
TARGET_GAIN_PCT = 2.0

DEFAULT_PENALTIES = ("level", "delta", "d2_match", "diff_amp")

PENALTY_POOLS: list[tuple[str, tuple[str, ...]]] = [
    ("p_level_delta_d2_diff", ("level", "delta", "d2_match", "diff_amp")),
    ("p_level_delta_diff", ("level", "delta", "diff_amp")),
    ("p_level_delta", ("level", "delta")),
    ("p_level_d2_diff", ("level", "d2_match", "diff_amp")),
    ("p_delta_d2_diff", ("delta", "d2_match", "diff_amp")),
    ("p_level_range_delta_diff", ("level", "range", "delta", "diff_amp")),
    ("p_level_delta_trend_dir", ("level", "delta", "trend", "direction")),
    ("p_amp_delta_diff_dir", ("amp_under", "delta", "diff_amp", "direction")),
]

VALIDATION_FIELDS = [
    "status",
    "stage",
    "candidate",
    "budget",
    "rank",
    "penalties",
    "penalty_count",
    "lambda_init",
    "alpha_scale",
    "feature_mode",
    "residual_clip",
    "selection_policy",
    "gate_max_scale",
    "gate_init_scale",
    "gate_scale_reg",
    "gate_init_bias_level",
    "residual_gate_alpha",
    "gate_noise_std",
    "pred_aware",
    "penalty_ema",
    "off_val_mse",
    "on_val_mse",
    "zero_val_mse",
    "full_val_mse",
    "val_gain_mse",
    "val_gain_pct",
    "gimoe_val_gain_mse",
    "gimoe_val_gain_pct",
    "full_vs_zero_val_gain_mse",
    "full_vs_zero_val_gain_pct",
    "val_constraints_ok",
    "off_val_mae",
    "on_val_mae",
    "zero_val_mae",
    "full_val_mae",
    "on_val_scaled_mse",
    "on_val_scaled_mae",
    "off_best_epoch",
    "on_best_epoch",
    "zero_best_epoch",
    "full_best_epoch",
    "off_config",
    "on_config",
    "zero_config",
    "full_config",
    "off_out_dir",
    "on_out_dir",
    "zero_out_dir",
    "full_out_dir",
    "off_returncode",
    "on_returncode",
    "zero_returncode",
    "full_returncode",
    "wrapper_sec",
    "error",
]

FINAL_FIELDS = [
    "status",
    "candidate",
    "source_validation_rank",
    "penalties",
    "lambda_init",
    "alpha_scale",
    "feature_mode",
    "residual_clip",
    "selection_policy",
    "gate_max_scale",
    "gate_init_scale",
    "gate_scale_reg",
    "gate_init_bias_level",
    "residual_gate_alpha",
    "gate_noise_std",
    "pred_aware",
    "penalty_ema",
    "off_test_mse",
    "on_test_mse",
    "zero_test_mse",
    "full_test_mse",
    "test_gain_mse",
    "test_gain_pct",
    "gimoe_test_gain_mse",
    "gimoe_test_gain_pct",
    "full_vs_zero_test_gain_mse",
    "full_vs_zero_test_gain_pct",
    "off_test_mae",
    "on_test_mae",
    "zero_test_mae",
    "full_test_mae",
    "off_val_mse",
    "on_val_mse",
    "zero_val_mse",
    "full_val_mse",
    "off_best_epoch",
    "on_best_epoch",
    "zero_best_epoch",
    "full_best_epoch",
    "hit_target",
    "off_config",
    "on_config",
    "zero_config",
    "full_config",
    "off_out_dir",
    "on_out_dir",
    "zero_out_dir",
    "full_out_dir",
    "landed_config",
    "error",
]


@dataclass(frozen=True)
class Candidate:
    stage: str
    name: str
    penalties: tuple[str, ...] = DEFAULT_PENALTIES
    lambda_init: float = 0.1
    alpha_scale: float = 1.1
    feature_mode: str = "legacy"
    residual_clip: float = 0.0
    selection_policy: str = "val_mse_gate"
    gate_max_scale: float = 1.0
    gate_init_scale: float = 0.8
    gate_scale_reg: float = 1.0e-4
    gate_init_bias_level: float = 2.0
    residual_gate_alpha: float = 0.7
    gate_noise_std: float = 0.2
    pred_aware: bool = True
    penalty_ema: bool = True


def resolve(path: Path | str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (ROOT / p).resolve()


def slug(text: str) -> str:
    chars = []
    for ch in str(text):
        if ch.isalnum() or ch in {"_", "-"}:
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars).strip("_") or "candidate"


def ftag(value: float) -> str:
    return f"{float(value):.6g}".replace("-", "m").replace(".", "p")


def btag(value: bool) -> str:
    return "t" if bool(value) else "f"


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def set_run_paths(cfg: dict[str, Any], out_dir: Path, *, skip_test: bool) -> None:
    cfg.setdefault("exp", {})["out_dir"] = str(out_dir)
    cfg.setdefault("corr", {})["save_path"] = str(out_dir / "corr.npy")
    cfg.setdefault("plot", {})["enable"] = False
    cfg.setdefault("portrait", {})["enable"] = False
    cfg["portrait"]["out_dir"] = str(out_dir / "cluster_portraits")
    cfg.setdefault("knn_hybrid", {})["enable"] = False
    cfg["knn_hybrid"]["use_for_model_selection"] = False
    cfg["knn_hybrid"]["path"] = str(out_dir / "knn_shape_bank.pt")
    cfg.setdefault("calibration", {})["enable"] = False
    cfg["memory"] = {
        "enable": False,
        "save_checkpoint": False,
        "path": str(out_dir / "cluster_memory.pt"),
        "checkpoint_path": str(out_dir / "best_checkpoint.pt"),
    }
    cfg.setdefault("eval", {})["skip_test"] = bool(skip_test)


def force_fixed_protocol(cfg: dict[str, Any], *, device: str | None, epochs: int, batch_size: int | None) -> None:
    cfg.setdefault("exp", {})
    cfg["exp"]["name"] = cfg["exp"].get("name", "ETTm1_H96_gimoe")
    if device:
        cfg["exp"]["device"] = str(device)

    cfg["data"] = {
        "csv_path": "data/ETTm1.csv",
        "date_col": 0,
        "max_rows": 57600,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
    }
    cfg["window"] = {
        **(cfg.get("window", {}) or {}),
        "input_len": INPUT_LEN,
        "pred_len": PRED_LEN,
    }
    cfg.setdefault("normalize", {})["train_only"] = True
    cfg["normalize"]["global_zscore"] = True
    cfg.setdefault("cluster", {})["train_only"] = True

    cfg["model"] = {
        "predictor": "mlp",
        "hidden_dim": 256,
        "dropout": 0.2,
    }

    cfg.setdefault("train", {})["epochs"] = int(epochs)
    if batch_size is not None:
        cfg["train"]["batch_size"] = int(batch_size)


def configure_gimoe(
    cfg: dict[str, Any],
    cand: Candidate,
    *,
    enabled: bool,
    lambda_override: float | None = None,
    gate_epochs: int | None = None,
) -> None:
    penalties = list(cand.penalties)
    cfg.setdefault("penalties", {})["enabled"] = penalties
    lambda_value = float(cand.lambda_init if lambda_override is None else lambda_override)

    moe = cfg.setdefault("moe", {})
    moe["enable"] = bool(enabled)
    moe["topk"] = 1
    moe["freeze_lambda"] = False
    moe.setdefault("gate_hidden_dim", 32)
    moe.setdefault("min_k_for_extensions", 3)
    moe.setdefault("safeguard_hidden_dim", 64)
    moe["select_ranks"] = [1]
    moe["detach_penalty_grad"] = False
    moe["dynamic_lambda"] = {
        **(moe.get("dynamic_lambda", {}) or {}),
        "enable": False,
    }
    moe.setdefault("learnable_lambda", {})["enable"] = False
    moe["lambda_init"] = {name: lambda_value for name in penalties}
    moe["lambda_min"] = {name: 0.0 for name in penalties}
    moe["lambda_schedule"] = {name: "none" for name in penalties}

    residual = moe.setdefault("pred_side_residual", {})
    residual.update(
        {
            "enable": bool(enabled),
            "feature_mode": str(cand.feature_mode),
            "residual_clip": float(cand.residual_clip),
            "corrector_hidden": 32,
            "init_alpha": -3.0,
            "alpha_scale": float(cand.alpha_scale),
            "specialization_weight": 0.1,
            "norm_weight": 0.0,
            "use_y_base_input": True,
            "intervention_enable": False,
            "intervention_init": -2.0,
            "intervention_weight": 1.0e-3,
            "detach_routed_penalty_pred": False,
            "selection_policy": str(cand.selection_policy) if enabled else "none",
            "selection_min_abs_improvement": 0.0,
            "selection_min_rel_improvement": 0.0,
        }
    )
    gate = residual.setdefault("gate_calibrator", {})
    gate.update(
        {
            "loss": "mse",
            "selection_metric": "mse",
            "epochs": int(gate_epochs) if gate_epochs is not None else 30,
            "train_fraction": 0.7,
            "hidden_dim": 32,
            "batch_size": 256,
            "max_scale": float(cand.gate_max_scale),
            "init_scale": float(cand.gate_init_scale),
            "scale_reg": float(cand.gate_scale_reg),
        }
    )

    moe["gate_entropy_weight"] = 0.0
    moe["gate_balance_weight"] = 0.0
    moe["gate_route_on_penalty_only"] = True
    moe["router_mode"] = "learned"
    moe["router_penalty_context_weight"] = 0.0
    moe["router_detach_penalty_context"] = True
    moe["allow_skip"] = True
    moe["skip_cost"] = 0.15
    moe["skip_init_bias"] = -2.0
    moe["gate_temperature"] = 1.0
    moe["gate_noise_std"] = float(cand.gate_noise_std)
    moe["gate_init_bias"] = {
        "enable": True,
        "values": {
            "level": float(cand.gate_init_bias_level),
            "default": 0.0,
        },
    }
    moe["gate_soft_weight"] = 0.0
    moe["gate_prob_floor"] = 0.0
    moe["gate_entropy_target_frac"] = 0.7
    moe["residual_gate"] = {"enable": True, "alpha": float(cand.residual_gate_alpha)}
    moe["pred_aware"] = {
        "enable": bool(cand.pred_aware),
        "use_pred_features": bool(cand.pred_aware),
        "use_penalty_input": False,
    }
    moe["penalty_ema"] = {"enable": bool(cand.penalty_ema), "decay": 0.9}
    moe["sigmoid_branch"] = {
        **(moe.get("sigmoid_branch", {}) or {}),
        "enable": True,
        "gamma": 0.2,
        "init_bias": -2.0,
    }
    moe["gate_logit_clip"] = 5.0

    if not enabled:
        moe["dynamic_lambda"]["enable"] = False
        moe["pred_side_residual"]["enable"] = False
        moe.setdefault("learnable_lambda", {})["enable"] = False


def candidate_id(cand: Candidate) -> str:
    parts = [
        cand.stage,
        cand.name,
        "lam" + ftag(cand.lambda_init),
        "a" + ftag(cand.alpha_scale),
        cand.feature_mode,
        "clip" + ftag(cand.residual_clip),
        cand.selection_policy,
        "gmax" + ftag(cand.gate_max_scale),
        "ginit" + ftag(cand.gate_init_scale),
        "greg" + ftag(cand.gate_scale_reg),
        "gb" + ftag(cand.gate_init_bias_level),
        "rga" + ftag(cand.residual_gate_alpha),
        "noise" + ftag(cand.gate_noise_std),
        "pa" + btag(cand.pred_aware),
        "ema" + btag(cand.penalty_ema),
    ]
    full = slug("__".join(parts))
    digest = hashlib.sha1(full.encode("utf-8")).hexdigest()[:10]
    readable = slug(f"{cand.stage}_{cand.name}")[:44]
    return f"{readable}_{digest}"


def build_config(
    base_cfg: dict[str, Any],
    cand: Candidate,
    *,
    enabled: bool,
    role: str | None = None,
    lambda_override: float | None = None,
    out_dir: Path,
    skip_test: bool,
    device: str | None,
    epochs: int,
    batch_size: int | None,
    gate_epochs: int | None = None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    role_name = role or ("on" if enabled else "off")
    cfg.setdefault("exp", {})["name"] = f"{DATASET}_H96_{candidate_id(cand)}_{role_name}"
    force_fixed_protocol(cfg, device=device, epochs=epochs, batch_size=batch_size)
    set_run_paths(cfg, out_dir, skip_test=skip_test)
    configure_gimoe(cfg, cand, enabled=enabled, lambda_override=lambda_override, gate_epochs=gate_epochs)
    return cfg


def run_train(config_path: Path, out_dir: Path, *, reuse_existing: bool) -> tuple[int, float, str]:
    summary_path = out_dir / "run_summary.json"
    if reuse_existing and summary_path.exists():
        return 0, 0.0, ""
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = out_dir / "stdout.log"
    cmd = [sys.executable, "-u", "-m", "src.train", "--config", str(config_path)]
    start = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    wrapper_sec = time.perf_counter() - start
    error = ""
    if proc.returncode != 0:
        error = stdout_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    elif not summary_path.exists():
        error = "run_summary.json not found"
        return 1, wrapper_sec, error
    return int(proc.returncode), wrapper_sec, error


def validate_summary(summary: dict[str, Any], *, expect_skip_test: bool) -> list[str]:
    problems = []
    windowing = summary.get("windowing", {}) or {}
    if not bool(windowing.get("normalize_train_only", False)):
        problems.append("windowing.normalize_train_only is not true")
    if int(windowing.get("data_max_rows", 0)) != 57600:
        problems.append("windowing.data_max_rows is not 57600")
    if bool((summary.get("eval", {}) or {}).get("skip_test", None)) != bool(expect_skip_test):
        problems.append("eval.skip_test mismatch")
    if expect_skip_test and summary.get("test") is not None:
        problems.append("test metrics exist during validation search")
    return problems


def metric_block(summary: dict[str, Any], split: str) -> dict[str, Any]:
    block = summary.get(split, {}) or {}
    return {
        "mse": safe_float(block.get("avg_mse")),
        "mae": safe_float(block.get("avg_mae")),
    }


def val_scaled_block(summary: dict[str, Any]) -> dict[str, Any]:
    residual = summary.get("moe_residual_selection", {}) or {}
    return {
        "mse": safe_float(residual.get("val_scaled_avg_mse")),
        "mae": safe_float(residual.get("val_scaled_avg_mae")),
    }


def best_epoch_text(summary: dict[str, Any]) -> str:
    value = summary.get("best_epoch", "")
    if isinstance(value, list):
        return ",".join(str(v) for v in value)
    return str(value)


def row_from_candidate(
    *,
    cand: Candidate,
    budget: str,
    off_cfg_path: Path,
    on_cfg_path: Path,
    zero_cfg_path: Path,
    full_cfg_path: Path,
    off_out: Path,
    on_out: Path,
    zero_out: Path,
    full_out: Path,
    off_returncode: int,
    on_returncode: int,
    zero_returncode: int,
    full_returncode: int,
    wrapper_sec: float,
    error: str,
) -> dict[str, Any]:
    row = {
        "status": "error" if error else "ok",
        "stage": cand.stage,
        "candidate": candidate_id(cand),
        "budget": budget,
        "rank": "",
        "penalties": "|".join(cand.penalties),
        "penalty_count": len(cand.penalties),
        "lambda_init": cand.lambda_init,
        "alpha_scale": cand.alpha_scale,
        "feature_mode": cand.feature_mode,
        "residual_clip": cand.residual_clip,
        "selection_policy": cand.selection_policy,
        "gate_max_scale": cand.gate_max_scale,
        "gate_init_scale": cand.gate_init_scale,
        "gate_scale_reg": cand.gate_scale_reg,
        "gate_init_bias_level": cand.gate_init_bias_level,
        "residual_gate_alpha": cand.residual_gate_alpha,
        "gate_noise_std": cand.gate_noise_std,
        "pred_aware": cand.pred_aware,
        "penalty_ema": cand.penalty_ema,
        "off_config": str(off_cfg_path),
        "on_config": str(on_cfg_path),
        "zero_config": str(zero_cfg_path),
        "full_config": str(full_cfg_path),
        "off_out_dir": str(off_out),
        "on_out_dir": str(on_out),
        "zero_out_dir": str(zero_out),
        "full_out_dir": str(full_out),
        "off_returncode": off_returncode,
        "on_returncode": on_returncode,
        "zero_returncode": zero_returncode,
        "full_returncode": full_returncode,
        "wrapper_sec": wrapper_sec,
        "error": error,
    }
    if error:
        return row

    off_summary = read_json(off_out / "run_summary.json")
    zero_summary = read_json(zero_out / "run_summary.json")
    full_summary = read_json(full_out / "run_summary.json")
    problems = validate_summary(off_summary, expect_skip_test=True)
    problems.extend(validate_summary(zero_summary, expect_skip_test=True))
    problems.extend(validate_summary(full_summary, expect_skip_test=True))
    if problems:
        row["status"] = "error"
        row["error"] = "; ".join(problems)
        return row

    off_val = metric_block(off_summary, "val")
    zero_val = metric_block(zero_summary, "val")
    full_val = metric_block(full_summary, "val")
    full_scaled = val_scaled_block(full_summary)
    gimoe_gain = off_val["mse"] - full_val["mse"]
    gimoe_gain_pct = 100.0 * gimoe_gain / max(abs(off_val["mse"]), 1.0e-12)
    full_vs_zero_gain = zero_val["mse"] - full_val["mse"]
    full_vs_zero_gain_pct = 100.0 * full_vs_zero_gain / max(abs(zero_val["mse"]), 1.0e-12)
    row.update(
        {
            "off_val_mse": off_val["mse"],
            "on_val_mse": full_val["mse"],
            "zero_val_mse": zero_val["mse"],
            "full_val_mse": full_val["mse"],
            "val_gain_mse": gimoe_gain,
            "val_gain_pct": gimoe_gain_pct,
            "gimoe_val_gain_mse": gimoe_gain,
            "gimoe_val_gain_pct": gimoe_gain_pct,
            "full_vs_zero_val_gain_mse": full_vs_zero_gain,
            "full_vs_zero_val_gain_pct": full_vs_zero_gain_pct,
            "val_constraints_ok": bool(gimoe_gain_pct >= TARGET_GAIN_PCT),
            "off_val_mae": off_val["mae"],
            "on_val_mae": full_val["mae"],
            "zero_val_mae": zero_val["mae"],
            "full_val_mae": full_val["mae"],
            "on_val_scaled_mse": full_scaled["mse"],
            "on_val_scaled_mae": full_scaled["mae"],
            "off_best_epoch": best_epoch_text(off_summary),
            "on_best_epoch": best_epoch_text(full_summary),
            "zero_best_epoch": best_epoch_text(zero_summary),
            "full_best_epoch": best_epoch_text(full_summary),
        }
    )
    return row


def row_to_candidate(row: dict[str, Any], *, stage: str | None = None) -> Candidate:
    return Candidate(
        stage=stage or str(row["stage"]),
        name=str(row["candidate"]),
        penalties=tuple(p for p in str(row["penalties"]).split("|") if p),
        lambda_init=safe_float(row["lambda_init"], 0.1),
        alpha_scale=safe_float(row["alpha_scale"], 1.1),
        feature_mode=str(row["feature_mode"]),
        residual_clip=safe_float(row["residual_clip"], 0.0),
        selection_policy=str(row["selection_policy"]),
        gate_max_scale=safe_float(row["gate_max_scale"], 1.0),
        gate_init_scale=safe_float(row["gate_init_scale"], 0.8),
        gate_scale_reg=safe_float(row["gate_scale_reg"], 1.0e-4),
        gate_init_bias_level=safe_float(row["gate_init_bias_level"], 2.0),
        residual_gate_alpha=safe_float(row["residual_gate_alpha"], 0.7),
        gate_noise_std=safe_float(row["gate_noise_std"], 0.2),
        pred_aware=parse_bool(row["pred_aware"]),
        penalty_ema=parse_bool(row["penalty_ema"]),
    )


def validation_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, int]:
    return (
        -float(parse_bool(row.get("val_constraints_ok"))),
        -safe_float(row.get("gimoe_val_gain_pct", row.get("val_gain_pct")), -math.inf),
        -safe_float(row.get("full_vs_zero_val_gain_mse"), -math.inf),
        safe_float(row.get("full_val_mse", row.get("on_val_mse")), math.inf),
        int(float(row.get("penalty_count", 999))),
    )


def rank_validation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    ranked_ids = {id(r): idx for idx, r in enumerate(sorted(ok_rows, key=validation_sort_key), start=1)}
    for row in rows:
        row["rank"] = ranked_ids.get(id(row), "")
    return rows


def budget_values(budget: str) -> dict[str, Any]:
    if budget == "smoke":
        return {
            "stage1_names": [],
            "stage2_top_pools": 1,
            "stage3_top_candidates": 1,
            "max_stage2": 1,
            "max_stage3": 0,
            "validation_gate_epochs": 3,
            "stage2": {
                "alpha_scale": [1.1],
                "feature_mode": ["legacy"],
                "residual_clip": [0.0],
                "selection_policy": ["val_mse_gate"],
                "gate_max_scale": [1.0],
                "gate_init_scale": [0.8],
                "gate_scale_reg": [1.0e-4],
            },
            "stage3": {},
        }
    if budget == "local":
        return {
            "stage1_names": [
                "p_level_delta_d2_diff",
                "p_level_delta_diff",
                "p_level_delta",
                "p_level_delta_trend_dir",
            ],
            "stage2_top_pools": 2,
            "stage3_top_candidates": 2,
            "max_stage2": 4,
            "max_stage3": 4,
            "validation_gate_epochs": 8,
            "stage2": {
                "alpha_scale": [0.8, 1.1, 1.4],
                "feature_mode": ["legacy", "safe_augmented"],
                "residual_clip": [0.0],
                "selection_policy": ["val_mse_gate", "val_mse_scale"],
                "gate_max_scale": [0.75, 1.0],
                "gate_init_scale": [0.8],
                "gate_scale_reg": [1.0e-4],
            },
            "stage3": {
                "lambda_init": [0.05, 0.1, 0.15],
                "gate_init_bias_level": [1.0, 2.0, 3.0],
                "residual_gate_alpha": [0.7],
                "gate_noise_std": [0.0, 0.2],
                "pred_aware": [True, False],
                "penalty_ema": [True],
            },
        }
    if budget == "highgain":
        return {
            "stage1_names": [
                "p_level_delta_d2_diff",
                "p_level_delta_trend_dir",
            ],
            "stage2_top_pools": 3,
            "stage3_top_candidates": 3,
            "max_stage2": 8,
            "max_stage3": 18,
            "validation_gate_epochs": 8,
            "stage2": {
                "alpha_scale": [1.1, 1.3],
                "feature_mode": ["legacy"],
                "residual_clip": [0.0],
                "selection_policy": ["val_mse_gate"],
                "gate_max_scale": [0.75, 1.0, 1.25],
                "gate_init_scale": [0.8, 1.0],
                "gate_scale_reg": [1.0e-4, 5.0e-4],
            },
            "stage3": {
                "lambda_init": [0.005, 0.01, 0.02, 0.03, 0.05, 0.075],
                "gate_init_bias_level": [1.5, 2.0, 2.5],
                "residual_gate_alpha": [0.5, 0.7],
                "gate_noise_std": [0.0, 0.1, 0.2],
                "pred_aware": [True],
                "penalty_ema": [True, False],
            },
        }
    if budget == "compact":
        return {
            "stage1_names": None,
            "stage2_top_pools": 4,
            "stage3_top_candidates": 5,
            "max_stage2": 80,
            "max_stage3": 80,
            "validation_gate_epochs": 15,
            "stage2": {
                "alpha_scale": [0.8, 1.1, 1.4],
                "feature_mode": ["legacy", "safe_augmented"],
                "residual_clip": [0.0, 6.0],
                "selection_policy": ["val_mse_gate", "val_mse_scale"],
                "gate_max_scale": [0.75, 1.0],
                "gate_init_scale": [0.6, 0.8],
                "gate_scale_reg": [1.0e-4],
            },
            "stage3": {
                "lambda_init": [0.05, 0.075, 0.1, 0.15],
                "gate_init_bias_level": [1.0, 2.0, 3.0],
                "residual_gate_alpha": [0.5, 0.7, 0.9],
                "gate_noise_std": [0.0, 0.2],
                "pred_aware": [True, False],
                "penalty_ema": [True, False],
            },
        }
    return {
        "stage1_names": None,
        "stage2_top_pools": 4,
        "stage3_top_candidates": 5,
        "max_stage2": None,
        "max_stage3": None,
        "validation_gate_epochs": None,
        "stage2": {
            "alpha_scale": [0.6, 0.8, 1.1, 1.4, 1.8],
            "feature_mode": ["legacy", "safe_augmented"],
            "residual_clip": [0.0, 4.0, 6.0],
            "selection_policy": ["val_mse_gate", "val_mse_gate_guarded", "val_mse_scale"],
            "gate_max_scale": [0.5, 0.75, 1.0, 1.25],
            "gate_init_scale": [0.4, 0.6, 0.8, 1.0],
            "gate_scale_reg": [1.0e-5, 1.0e-4, 5.0e-4],
        },
        "stage3": {
            "lambda_init": [0.02, 0.05, 0.075, 0.1, 0.15, 0.2],
            "gate_init_bias_level": [0.0, 1.0, 2.0, 3.0],
            "residual_gate_alpha": [0.5, 0.7, 0.9],
            "gate_noise_std": [0.0, 0.1, 0.2],
            "pred_aware": [True, False],
            "penalty_ema": [True, False],
        },
    }


def take_evenly(candidates: list[Candidate], limit: int | None) -> list[Candidate]:
    if limit is None or len(candidates) <= limit:
        return candidates
    if limit <= 0:
        return []
    if limit == 1:
        return [candidates[0]]
    indexes = sorted({round(i * (len(candidates) - 1) / (limit - 1)) for i in range(limit)})
    return [candidates[i] for i in indexes]


def stage0_candidates() -> list[Candidate]:
    return [
        Candidate(
            stage="stage0",
            name="current_residual_full",
            penalties=DEFAULT_PENALTIES,
        )
    ]


def stage1_candidates() -> list[Candidate]:
    return [
        Candidate(stage="stage1", name=name, penalties=penalties)
        for name, penalties in PENALTY_POOLS
    ]


def filter_stage1(candidates: list[Candidate], budget_cfg: dict[str, Any]) -> list[Candidate]:
    names = budget_cfg.get("stage1_names")
    if names is None:
        return candidates
    allowed = {str(name) for name in names}
    return [cand for cand in candidates if cand.name in allowed]


def stage2_candidates(top_rows: list[dict[str, Any]], budget_cfg: dict[str, Any], limit: int | None) -> list[Candidate]:
    grid = budget_cfg["stage2"]
    out: list[Candidate] = []
    for row in top_rows:
        base = row_to_candidate(row, stage="stage2")
        for values in itertools.product(
            grid["alpha_scale"],
            grid["feature_mode"],
            grid["residual_clip"],
            grid["selection_policy"],
            grid["gate_max_scale"],
            grid["gate_init_scale"],
            grid["gate_scale_reg"],
        ):
            alpha, feature, clip, policy, max_scale, init_scale, scale_reg = values
            name = (
                f"{row['candidate']}__a{ftag(alpha)}__{feature}__clip{ftag(clip)}"
                f"__{policy}__gmax{ftag(max_scale)}__ginit{ftag(init_scale)}__greg{ftag(scale_reg)}"
            )
            out.append(
                replace(
                    base,
                    name=name,
                    alpha_scale=float(alpha),
                    feature_mode=str(feature),
                    residual_clip=float(clip),
                    selection_policy=str(policy),
                    gate_max_scale=float(max_scale),
                    gate_init_scale=float(init_scale),
                    gate_scale_reg=float(scale_reg),
                )
            )
    return take_evenly(out, limit)


def stage3_candidates(top_rows: list[dict[str, Any]], budget_cfg: dict[str, Any], limit: int | None) -> list[Candidate]:
    grid = budget_cfg["stage3"]
    if not grid:
        return []
    out: list[Candidate] = []
    for row in top_rows:
        base = row_to_candidate(row, stage="stage3")
        for values in itertools.product(
            grid["lambda_init"],
            grid["gate_init_bias_level"],
            grid["residual_gate_alpha"],
            grid["gate_noise_std"],
            grid["pred_aware"],
            grid["penalty_ema"],
        ):
            lam, gate_bias, residual_alpha, noise, pred_aware, penalty_ema = values
            name = (
                f"{row['candidate']}__lam{ftag(lam)}__gb{ftag(gate_bias)}"
                f"__rga{ftag(residual_alpha)}__noise{ftag(noise)}"
                f"__pa{btag(pred_aware)}__ema{btag(penalty_ema)}"
            )
            out.append(
                replace(
                    base,
                    name=name,
                    lambda_init=float(lam),
                    gate_init_bias_level=float(gate_bias),
                    residual_gate_alpha=float(residual_alpha),
                    gate_noise_std=float(noise),
                    pred_aware=bool(pred_aware),
                    penalty_ema=bool(penalty_ema),
                )
            )
    return take_evenly(out, limit)


def run_validation_candidate(
    cand: Candidate,
    *,
    base_cfg: dict[str, Any],
    out_root: Path,
    budget: str,
    device: str | None,
    epochs: int,
    batch_size: int | None,
    gate_epochs: int | None,
    reuse_existing: bool,
    dry_run: bool,
) -> dict[str, Any]:
    cid = candidate_id(cand)
    cfg_dir = out_root / "configs" / "validation" / cid
    run_dir = out_root / "runs" / "validation" / cid
    off_out = run_dir / "base_no_moe"
    zero_out = run_dir / "zero_lambda_residual"
    full_out = run_dir / "residual_full"
    on_out = full_out
    off_cfg_path = cfg_dir / "base_no_moe.yaml"
    zero_cfg_path = cfg_dir / "zero_lambda_residual.yaml"
    full_cfg_path = cfg_dir / "residual_full.yaml"
    on_cfg_path = full_cfg_path
    off_cfg = build_config(
        base_cfg,
        cand,
        enabled=False,
        role="base_no_moe",
        out_dir=off_out,
        skip_test=True,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        gate_epochs=gate_epochs,
    )
    zero_cfg = build_config(
        base_cfg,
        cand,
        enabled=True,
        role="zero_lambda_residual",
        lambda_override=0.0,
        out_dir=zero_out,
        skip_test=True,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        gate_epochs=gate_epochs,
    )
    full_cfg = build_config(
        base_cfg,
        cand,
        enabled=True,
        role="residual_full",
        out_dir=full_out,
        skip_test=True,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        gate_epochs=gate_epochs,
    )
    write_yaml(off_cfg_path, off_cfg)
    write_yaml(zero_cfg_path, zero_cfg)
    write_yaml(full_cfg_path, full_cfg)
    if dry_run:
        return {
            "status": "prepared",
            "stage": cand.stage,
            "candidate": cid,
            "budget": budget,
            "penalties": "|".join(cand.penalties),
            "penalty_count": len(cand.penalties),
            "lambda_init": cand.lambda_init,
            "alpha_scale": cand.alpha_scale,
            "feature_mode": cand.feature_mode,
            "residual_clip": cand.residual_clip,
            "selection_policy": cand.selection_policy,
            "gate_max_scale": cand.gate_max_scale,
            "gate_init_scale": cand.gate_init_scale,
            "gate_scale_reg": cand.gate_scale_reg,
            "gate_init_bias_level": cand.gate_init_bias_level,
            "residual_gate_alpha": cand.residual_gate_alpha,
            "gate_noise_std": cand.gate_noise_std,
            "pred_aware": cand.pred_aware,
            "penalty_ema": cand.penalty_ema,
            "off_config": str(off_cfg_path),
            "on_config": str(on_cfg_path),
            "zero_config": str(zero_cfg_path),
            "full_config": str(full_cfg_path),
            "off_out_dir": str(off_out),
            "on_out_dir": str(on_out),
            "zero_out_dir": str(zero_out),
            "full_out_dir": str(full_out),
        }
    off_code, off_sec, off_err = run_train(off_cfg_path, off_out, reuse_existing=reuse_existing)
    zero_code, zero_sec, zero_err = run_train(zero_cfg_path, zero_out, reuse_existing=reuse_existing)
    full_code, full_sec, full_err = run_train(full_cfg_path, full_out, reuse_existing=reuse_existing)
    error = off_err or zero_err or full_err
    if off_code != 0 and not error:
        error = f"base_no_moe failed with returncode={off_code}"
    if zero_code != 0 and not error:
        error = f"zero_lambda_residual failed with returncode={zero_code}"
    if full_code != 0 and not error:
        error = f"residual_full failed with returncode={full_code}"
    return row_from_candidate(
        cand=cand,
        budget=budget,
        off_cfg_path=off_cfg_path,
        on_cfg_path=on_cfg_path,
        zero_cfg_path=zero_cfg_path,
        full_cfg_path=full_cfg_path,
        off_out=off_out,
        on_out=on_out,
        zero_out=zero_out,
        full_out=full_out,
        off_returncode=off_code,
        on_returncode=full_code,
        zero_returncode=zero_code,
        full_returncode=full_code,
        wrapper_sec=off_sec + zero_sec + full_sec,
        error=error,
    )


def existing_keys(rows: Iterable[dict[str, Any]]) -> set[str]:
    return {str(r.get("candidate", "")) for r in rows if r.get("status") in {"ok", "prepared"}}


def run_validation_search(args: argparse.Namespace, base_cfg: dict[str, Any], out_root: Path) -> list[dict[str, Any]]:
    results_path = out_root / "validation_results.csv"
    rows = read_csv_rows(results_path) if args.reuse_existing else []
    done = existing_keys(rows)
    budget_cfg = budget_values(args.budget)

    def append_candidates(candidates: list[Candidate]) -> None:
        nonlocal rows, done
        for cand in candidates:
            cid = candidate_id(cand)
            if cid in done and args.reuse_existing:
                print(f"[reuse] {cand.stage} {cid}", flush=True)
                continue
            print(f"[run] {cand.stage} {cid}", flush=True)
            row = run_validation_candidate(
                cand,
                base_cfg=base_cfg,
                out_root=out_root,
                budget=args.budget,
                device=args.device,
                epochs=args.validation_epochs,
                batch_size=args.batch_size,
                gate_epochs=args.validation_gate_epochs,
                reuse_existing=args.reuse_existing,
                dry_run=args.dry_run,
            )
            rows = [r for r in rows if r.get("candidate") != cid]
            rows.append(row)
            rows = rank_validation_rows(rows)
            write_csv(results_path, rows, VALIDATION_FIELDS)
            done.add(cid)
            print(
                f"  -> {row.get('status')} gimoe_val_gain={row.get('gimoe_val_gain_pct', row.get('val_gain_pct', ''))} "
                f"full_vs_zero={row.get('full_vs_zero_val_gain_mse', '')} full_val={row.get('full_val_mse', row.get('on_val_mse', ''))}",
                flush=True,
            )

    append_candidates(stage0_candidates())
    if args.budget == "smoke":
        return rows

    append_candidates(filter_stage1(stage1_candidates(), budget_cfg))
    ok_rows = sorted([r for r in rows if r.get("status") == "ok"], key=validation_sort_key)
    stage1_ok = [r for r in ok_rows if str(r.get("stage")) in {"stage0", "stage1"}]
    top_pools = stage1_ok[: int(args.stage2_top_pools or budget_cfg["stage2_top_pools"])]
    max_stage2 = args.max_stage2 if args.max_stage2 is not None else budget_cfg["max_stage2"]
    append_candidates(stage2_candidates(top_pools, budget_cfg, max_stage2))

    ok_rows = sorted([r for r in rows if r.get("status") == "ok"], key=validation_sort_key)
    top_stage2 = ok_rows[: int(args.stage3_top_candidates or budget_cfg["stage3_top_candidates"])]
    max_stage3 = args.max_stage3 if args.max_stage3 is not None else budget_cfg["max_stage3"]
    append_candidates(stage3_candidates(top_stage2, budget_cfg, max_stage3))

    rows = rank_validation_rows(rows)
    write_csv(results_path, rows, VALIDATION_FIELDS)
    return rows


def best_validation_row(rows: list[dict[str, Any]], *, rank: int = 1) -> dict[str, Any] | None:
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        return None
    ranked = sorted(ok, key=validation_sort_key)
    idx = max(0, int(rank) - 1)
    if idx >= len(ranked):
        return None
    return ranked[idx]


def build_final_row(
    *,
    cand: Candidate,
    validation_row: dict[str, Any],
    off_cfg_path: Path,
    on_cfg_path: Path,
    zero_cfg_path: Path,
    full_cfg_path: Path,
    off_out: Path,
    on_out: Path,
    zero_out: Path,
    full_out: Path,
    landed_config: str,
    error: str,
) -> dict[str, Any]:
    row = {
        "status": "error" if error else "ok",
        "candidate": candidate_id(cand),
        "source_validation_rank": validation_row.get("rank", ""),
        "penalties": "|".join(cand.penalties),
        "lambda_init": cand.lambda_init,
        "alpha_scale": cand.alpha_scale,
        "feature_mode": cand.feature_mode,
        "residual_clip": cand.residual_clip,
        "selection_policy": cand.selection_policy,
        "gate_max_scale": cand.gate_max_scale,
        "gate_init_scale": cand.gate_init_scale,
        "gate_scale_reg": cand.gate_scale_reg,
        "gate_init_bias_level": cand.gate_init_bias_level,
        "residual_gate_alpha": cand.residual_gate_alpha,
        "gate_noise_std": cand.gate_noise_std,
        "pred_aware": cand.pred_aware,
        "penalty_ema": cand.penalty_ema,
        "off_config": str(off_cfg_path),
        "on_config": str(on_cfg_path),
        "zero_config": str(zero_cfg_path),
        "full_config": str(full_cfg_path),
        "off_out_dir": str(off_out),
        "on_out_dir": str(on_out),
        "zero_out_dir": str(zero_out),
        "full_out_dir": str(full_out),
        "landed_config": landed_config,
        "error": error,
    }
    if error:
        return row
    off_summary = read_json(off_out / "run_summary.json")
    zero_summary = read_json(zero_out / "run_summary.json")
    full_summary = read_json(full_out / "run_summary.json")
    problems = validate_summary(off_summary, expect_skip_test=False)
    problems.extend(validate_summary(zero_summary, expect_skip_test=False))
    problems.extend(validate_summary(full_summary, expect_skip_test=False))
    if problems:
        row["status"] = "error"
        row["error"] = "; ".join(problems)
        return row
    off_test = metric_block(off_summary, "test")
    zero_test = metric_block(zero_summary, "test")
    full_test = metric_block(full_summary, "test")
    off_val = metric_block(off_summary, "val")
    zero_val = metric_block(zero_summary, "val")
    full_val = metric_block(full_summary, "val")
    gimoe_gain = off_test["mse"] - full_test["mse"]
    gimoe_gain_pct = 100.0 * gimoe_gain / max(abs(off_test["mse"]), 1.0e-12)
    full_vs_zero_gain = zero_test["mse"] - full_test["mse"]
    full_vs_zero_gain_pct = 100.0 * full_vs_zero_gain / max(abs(zero_test["mse"]), 1.0e-12)
    row.update(
        {
            "off_test_mse": off_test["mse"],
            "on_test_mse": full_test["mse"],
            "zero_test_mse": zero_test["mse"],
            "full_test_mse": full_test["mse"],
            "test_gain_mse": gimoe_gain,
            "test_gain_pct": gimoe_gain_pct,
            "gimoe_test_gain_mse": gimoe_gain,
            "gimoe_test_gain_pct": gimoe_gain_pct,
            "full_vs_zero_test_gain_mse": full_vs_zero_gain,
            "full_vs_zero_test_gain_pct": full_vs_zero_gain_pct,
            "off_test_mae": off_test["mae"],
            "on_test_mae": full_test["mae"],
            "zero_test_mae": zero_test["mae"],
            "full_test_mae": full_test["mae"],
            "off_val_mse": off_val["mse"],
            "on_val_mse": full_val["mse"],
            "zero_val_mse": zero_val["mse"],
            "full_val_mse": full_val["mse"],
            "off_best_epoch": best_epoch_text(off_summary),
            "on_best_epoch": best_epoch_text(full_summary),
            "zero_best_epoch": best_epoch_text(zero_summary),
            "full_best_epoch": best_epoch_text(full_summary),
            "hit_target": bool(gimoe_gain_pct >= TARGET_GAIN_PCT),
        }
    )
    return row


def run_final_test(
    args: argparse.Namespace,
    base_cfg: dict[str, Any],
    out_root: Path,
    validation_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    selected = best_validation_row(validation_rows, rank=args.final_rank)
    if selected is None:
        print(f"[final] no validation row available for rank={args.final_rank}", flush=True)
        return None
    cand = row_to_candidate(selected, stage="final")
    cid = candidate_id(cand)
    cfg_dir = out_root / "configs" / "final" / cid
    run_dir = out_root / "runs" / "final" / cid
    off_out = run_dir / "base_no_moe"
    zero_out = run_dir / "zero_lambda_residual"
    full_out = run_dir / "residual_full"
    on_out = full_out
    off_cfg_path = cfg_dir / "base_no_moe.yaml"
    zero_cfg_path = cfg_dir / "zero_lambda_residual.yaml"
    full_cfg_path = cfg_dir / "residual_full.yaml"
    on_cfg_path = full_cfg_path
    off_cfg = build_config(
        base_cfg,
        cand,
        enabled=False,
        role="base_no_moe",
        out_dir=off_out,
        skip_test=False,
        device=args.device,
        epochs=args.final_eval_epochs,
        batch_size=args.batch_size,
        gate_epochs=args.final_gate_epochs,
    )
    zero_cfg = build_config(
        base_cfg,
        cand,
        enabled=True,
        role="zero_lambda_residual",
        lambda_override=0.0,
        out_dir=zero_out,
        skip_test=False,
        device=args.device,
        epochs=args.final_eval_epochs,
        batch_size=args.batch_size,
        gate_epochs=args.final_gate_epochs,
    )
    full_cfg = build_config(
        base_cfg,
        cand,
        enabled=True,
        role="residual_full",
        out_dir=full_out,
        skip_test=False,
        device=args.device,
        epochs=args.final_eval_epochs,
        batch_size=args.batch_size,
        gate_epochs=args.final_gate_epochs,
    )
    write_yaml(off_cfg_path, off_cfg)
    write_yaml(zero_cfg_path, zero_cfg)
    write_yaml(full_cfg_path, full_cfg)
    if args.dry_run:
        row = build_final_row(
            cand=cand,
            validation_row=selected,
            off_cfg_path=off_cfg_path,
            on_cfg_path=on_cfg_path,
            zero_cfg_path=zero_cfg_path,
            full_cfg_path=full_cfg_path,
            off_out=off_out,
            on_out=on_out,
            zero_out=zero_out,
            full_out=full_out,
            landed_config="",
            error="dry-run: final test not executed",
        )
    else:
        off_code, _off_sec, off_err = run_train(off_cfg_path, off_out, reuse_existing=args.reuse_existing)
        zero_code, _zero_sec, zero_err = run_train(zero_cfg_path, zero_out, reuse_existing=args.reuse_existing)
        full_code, _full_sec, full_err = run_train(full_cfg_path, full_out, reuse_existing=args.reuse_existing)
        error = off_err or zero_err or full_err
        if off_code != 0 and not error:
            error = f"base_no_moe failed with returncode={off_code}"
        if zero_code != 0 and not error:
            error = f"zero_lambda_residual failed with returncode={zero_code}"
        if full_code != 0 and not error:
            error = f"residual_full failed with returncode={full_code}"
        row = build_final_row(
            cand=cand,
            validation_row=selected,
            off_cfg_path=off_cfg_path,
            on_cfg_path=on_cfg_path,
            zero_cfg_path=zero_cfg_path,
            full_cfg_path=full_cfg_path,
            off_out=off_out,
            on_out=on_out,
            zero_out=zero_out,
            full_out=full_out,
            landed_config="",
            error=error,
        )
    final_path = out_root / "final_test_results.csv"
    rows = read_csv_rows(final_path)
    rows = [r for r in rows if r.get("candidate") != cid]
    rows.append(row)
    write_csv(final_path, rows, FINAL_FIELDS)
    if row.get("status") == "ok":
        print(
            f"[final] off={row['off_test_mse']:.6f} zero={row['zero_test_mse']:.6f} "
            f"full={row['full_test_mse']:.6f} gimoe_gain={row['test_gain_pct']:.3f}% "
            f"full_vs_zero={row['full_vs_zero_test_gain_mse']:.6f}",
            flush=True,
        )
        if bool(row.get("hit_target")) and args.land_config:
            landed = land_config(args.base_config, full_cfg, cid)
            row["landed_config"] = str(landed)
            rows = [r for r in rows if r.get("candidate") != cid]
            rows.append(row)
            write_csv(final_path, rows, FINAL_FIELDS)
    return row


def land_config(base_config: Path, cfg: dict[str, Any], cid: str) -> Path:
    target = resolve(base_config)
    landed = copy.deepcopy(cfg)
    out_dir = Path("outputs") / "ETTm1_h96_gimoe_gain_best"
    landed.setdefault("exp", {})["name"] = "ETTm1_H96_gimoe_gain_best"
    landed["exp"]["out_dir"] = str(out_dir)
    landed.setdefault("corr", {})["save_path"] = str(out_dir / "corr.npy")
    landed.setdefault("portrait", {})["out_dir"] = str(out_dir / "cluster_portraits")
    landed.setdefault("knn_hybrid", {})["path"] = str(out_dir / "knn_shape_bank.pt")
    landed["memory"] = {
        "enable": False,
        "save_checkpoint": False,
        "path": str(out_dir / "cluster_memory.pt"),
        "checkpoint_path": str(out_dir / "best_checkpoint.pt"),
    }
    landed.setdefault("eval", {})["skip_test"] = False
    target.write_text(yaml.safe_dump(landed, sort_keys=False, allow_unicode=False), encoding="utf-8")
    print(f"[landed] {cid} -> {target}", flush=True)
    return target


def validate_generated_yaml(out_root: Path) -> int:
    count = 0
    for path in (out_root / "configs").rglob("*.yaml"):
        cfg = read_yaml(path)
        problems = []
        if str(cfg.get("data", {}).get("csv_path")) != "data/ETTm1.csv":
            problems.append("data.csv_path is not data/ETTm1.csv")
        if int(cfg.get("data", {}).get("max_rows", -1)) != 57600:
            problems.append("data.max_rows is not 57600")
        if int(cfg.get("window", {}).get("input_len", -1)) != INPUT_LEN:
            problems.append("window.input_len is not 336")
        if int(cfg.get("window", {}).get("pred_len", -1)) != PRED_LEN:
            problems.append("window.pred_len is not 96")
        model = cfg.get("model", {}) or {}
        if model.get("predictor") != "mlp" or int(model.get("hidden_dim", -1)) != 256 or float(model.get("dropout", -1.0)) != 0.2:
            problems.append("model protocol is not fixed to mlp/256/0.2")
        if not bool(cfg.get("normalize", {}).get("train_only", False)):
            problems.append("normalize.train_only is not true")
        if not bool(cfg.get("cluster", {}).get("train_only", False)):
            problems.append("cluster.train_only is not true")
        if bool(cfg.get("knn_hybrid", {}).get("enable", True)):
            problems.append("knn_hybrid.enable is not false")
        if bool(cfg.get("calibration", {}).get("enable", True)):
            problems.append("calibration.enable is not false")
        if bool(cfg.get("moe", {}).get("dynamic_lambda", {}).get("enable", False)):
            problems.append("moe.dynamic_lambda.enable is not false")
        if problems:
            raise RuntimeError(f"Generated YAML protocol check failed for {path}: {'; '.join(problems)}")
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="ETTm1 H96 validation-selected GIMoE gain search.")
    parser.add_argument("--base-config", type=Path, default=ROOT / "configs" / "ETTm1.yaml")
    parser.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "ettm1_h96_gimoe_gain_search")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--budget", choices=["smoke", "local", "highgain", "compact", "full"], default="local")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--val-epochs", type=int, default=None)
    parser.add_argument("--final-epochs", type=int, default=None)
    parser.add_argument("--val-gate-epochs", type=int, default=None)
    parser.add_argument("--final-gate-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--stage2-top-pools", type=int, default=None)
    parser.add_argument("--stage3-top-candidates", type=int, default=None)
    parser.add_argument("--max-stage2", type=int, default=None)
    parser.add_argument("--max-stage3", type=int, default=None)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-final", action="store_true")
    parser.add_argument("--final-only", action="store_true")
    parser.add_argument("--final-rank", type=int, default=1)
    parser.add_argument("--land-config", action="store_true")
    args = parser.parse_args()
    args.validation_epochs = int(args.val_epochs if args.val_epochs is not None else args.epochs)
    args.final_eval_epochs = int(args.final_epochs if args.final_epochs is not None else args.epochs)
    default_budget_cfg = budget_values(args.budget)
    args.validation_gate_epochs = (
        int(args.val_gate_epochs)
        if args.val_gate_epochs is not None
        else default_budget_cfg.get("validation_gate_epochs")
    )

    base_path = resolve(args.base_config)
    out_root = resolve(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    base_cfg = read_yaml(base_path)

    if args.final_only:
        validation_rows = read_csv_rows(out_root / "validation_results.csv")
    else:
        validation_rows = run_validation_search(args, base_cfg, out_root)
    yaml_count = validate_generated_yaml(out_root)
    print(f"[yaml] parsed {yaml_count} generated YAML files", flush=True)

    if args.run_final or args.final_only:
        run_final_test(args, base_cfg, out_root, validation_rows)


if __name__ == "__main__":
    main()
