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
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
TARGET_GAIN_PCT = 5.0
INPUT_LEN = 336
PRED_LEN = 96

FIELDS = [
    "status",
    "dataset",
    "candidate",
    "penalties",
    "lambda_init",
    "alpha_scale",
    "feature_mode",
    "residual_clip",
    "selection_policy",
    "selection_min_rel_improvement",
    "gate_max_scale",
    "gate_init_scale",
    "gate_scale_reg",
    "gate_train_fraction",
    "gate_init_bias_enable",
    "gate_init_bias_level",
    "gate_noise_std",
    "pred_aware",
    "penalty_ema",
    "penalty_guard_allow_multi",
    "channel_guard_enable",
    "model_hidden_dim",
    "model_dropout",
    "mse_weight",
    "mae_weight",
    "base_test_mse",
    "zero_test_mse",
    "full_test_mse",
    "zero_gain_pct",
    "full_gain_pct",
    "full_vs_zero_mse",
    "hit_target",
    "base_test_mae",
    "zero_test_mae",
    "full_test_mae",
    "base_val_mse",
    "zero_val_mse",
    "full_val_mse",
    "base_best_epoch",
    "zero_best_epoch",
    "full_best_epoch",
    "base_out_dir",
    "zero_out_dir",
    "full_out_dir",
    "zero_config",
    "full_config",
    "wrapper_sec",
    "error",
]


@dataclass(frozen=True)
class Candidate:
    name: str
    penalties: tuple[str, ...]
    lambda_init: float
    alpha_scale: float
    feature_mode: str = "legacy"
    residual_clip: float = 0.0
    selection_policy: str = "val_mse_gate"
    selection_min_rel_improvement: float = 0.0
    gate_max_scale: float = 1.0
    gate_init_scale: float = 0.6
    gate_scale_reg: float = 5.0e-4
    gate_train_fraction: float = 0.7
    gate_init_bias_enable: bool = True
    gate_init_bias_level: float = 2.0
    gate_noise_std: float = 0.2
    pred_aware: bool = True
    penalty_ema: bool = True
    penalty_guard_allow_multi: bool = True
    channel_guard_enable: bool = True
    model_hidden_dim: int | None = None
    model_dropout: float | None = None
    mse_weight: float | None = None
    mae_weight: float | None = None


def resolve(path: Path | str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p).resolve()


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def read_csv(path: Path) -> list[dict[str, Any]]:
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


def metric(summary: dict[str, Any], split: str, key: str) -> float:
    return safe_float((summary.get(split, {}) or {}).get(key))


def best_epoch(summary: dict[str, Any]) -> str:
    value = summary.get("best_epoch", "")
    if isinstance(value, list):
        return ",".join(str(v) for v in value)
    return str(value)


def tag(text: str) -> str:
    out = []
    for ch in text:
        out.append(ch if ch.isalnum() or ch in {"_", "-"} else "_")
    return "".join(out).strip("_") or "candidate"


def cid(cand: Candidate) -> str:
    raw = "|".join(
        [
            cand.name,
            ",".join(cand.penalties),
            str(cand.lambda_init),
            str(cand.alpha_scale),
            cand.feature_mode,
            str(cand.residual_clip),
            cand.selection_policy,
            str(cand.selection_min_rel_improvement),
            str(cand.gate_max_scale),
            str(cand.gate_init_scale),
            str(cand.gate_scale_reg),
            str(cand.gate_train_fraction),
            str(cand.gate_init_bias_enable),
            str(cand.gate_init_bias_level),
            str(cand.gate_noise_std),
            str(cand.pred_aware),
            str(cand.penalty_ema),
            str(cand.penalty_guard_allow_multi),
            str(cand.channel_guard_enable),
            str(cand.model_hidden_dim),
            str(cand.model_dropout),
            str(cand.mse_weight),
            str(cand.mae_weight),
        ]
    )
    return f"{tag(cand.name)[:46]}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:10]}"


def fixed_protocol(cfg: dict[str, Any], *, out_dir: Path, device: str | None, epochs: int | None) -> None:
    if device:
        cfg.setdefault("exp", {})["device"] = device
    cfg.setdefault("exp", {})["out_dir"] = str(out_dir)
    cfg.setdefault("window", {})["input_len"] = INPUT_LEN
    cfg["window"]["pred_len"] = PRED_LEN
    cfg.setdefault("normalize", {})["train_only"] = True
    cfg.setdefault("cluster", {})["train_only"] = True
    if epochs is not None:
        cfg.setdefault("train", {})["epochs"] = int(epochs)
    cfg.setdefault("plot", {})["enable"] = False
    cfg.setdefault("portrait", {})["enable"] = False
    cfg["portrait"]["out_dir"] = str(out_dir / "cluster_portraits")
    cfg.setdefault("corr", {})["save_path"] = str(out_dir / "corr.npy")
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
    cfg.setdefault("eval", {})["skip_test"] = False


def apply_base_capacity(cfg: dict[str, Any], cand: Candidate | None) -> None:
    if cand is None:
        return
    if cand.model_hidden_dim is not None:
        cfg.setdefault("model", {})["hidden_dim"] = int(cand.model_hidden_dim)
    if cand.model_dropout is not None:
        cfg.setdefault("model", {})["dropout"] = float(cand.model_dropout)
    if cand.mse_weight is not None:
        cfg.setdefault("train", {})["mse_weight"] = float(cand.mse_weight)
    if cand.mae_weight is not None:
        mae_cfg = cfg.setdefault("train", {}).setdefault("mae_objective", {})
        mae_cfg["enable"] = True
        mae_cfg["kind"] = str(mae_cfg.get("kind", "l1"))
        mae_cfg["weight"] = float(cand.mae_weight)


def configure_base(cfg: dict[str, Any]) -> None:
    moe = cfg.setdefault("moe", {})
    moe["enable"] = False
    moe.setdefault("dynamic_lambda", {})["enable"] = False
    moe.setdefault("pred_side_residual", {})["enable"] = False
    moe.setdefault("learnable_lambda", {})["enable"] = False


def configure_moe(cfg: dict[str, Any], cand: Candidate, *, lambda_value: float) -> None:
    penalties = list(cand.penalties)
    cfg.setdefault("penalties", {})["enabled"] = penalties
    moe = cfg.setdefault("moe", {})
    moe["enable"] = True
    moe["topk"] = 1
    moe["freeze_lambda"] = False
    moe["detach_penalty_grad"] = False
    moe.setdefault("dynamic_lambda", {})["enable"] = False
    moe.setdefault("learnable_lambda", {})["enable"] = False
    moe["lambda_init"] = {name: float(lambda_value) for name in penalties}
    moe["lambda_min"] = {name: 0.0 for name in penalties}
    moe["lambda_schedule"] = {name: "none" for name in penalties}

    residual = moe.setdefault("pred_side_residual", {})
    residual.update(
        {
            "enable": True,
            "feature_mode": cand.feature_mode,
            "residual_clip": float(cand.residual_clip),
            "corrector_hidden": int(residual.get("corrector_hidden", 32)),
            "init_alpha": float(residual.get("init_alpha", -3.0)),
            "alpha_scale": float(cand.alpha_scale),
            "specialization_weight": float(residual.get("specialization_weight", 0.1)),
            "norm_weight": float(residual.get("norm_weight", 0.0)),
            "use_y_base_input": True,
            "intervention_enable": False,
            "intervention_init": -2.0,
            "intervention_weight": 1.0e-3,
            "detach_routed_penalty_pred": False,
            "selection_policy": cand.selection_policy,
            "selection_min_abs_improvement": 0.0,
            "selection_min_rel_improvement": float(cand.selection_min_rel_improvement),
            "penalty_guard": {
                "enable": True,
                "metric": "mse",
                "allow_multi": bool(cand.penalty_guard_allow_multi),
                "min_abs_improvement": 0.0,
                "min_rel_improvement": float(cand.selection_min_rel_improvement),
            },
            "channel_guard": {"enable": bool(cand.channel_guard_enable)},
            "validation_guard": {
                "enable": True,
                "select_fraction": 0.5,
                "min_abs_improvement": 0.0,
                "min_rel_improvement": max(0.002, float(cand.selection_min_rel_improvement)),
            },
            "diagnostics": {"enable": True},
        }
    )
    gate = residual.setdefault("gate_calibrator", {})
    gate.update(
        {
            "loss": "mse",
            "selection_metric": "mse",
            "epochs": int(gate.get("epochs", 30)),
            "train_fraction": float(cand.gate_train_fraction),
            "hidden_dim": int(gate.get("hidden_dim", 32)),
            "batch_size": int(gate.get("batch_size", 256)),
            "max_scale": float(cand.gate_max_scale),
            "init_scale": float(cand.gate_init_scale),
            "scale_reg": float(cand.gate_scale_reg),
            "scale_mode": "sigmoid",
            "standardize_features": True,
        }
    )

    moe["gate_route_on_penalty_only"] = True
    moe["router_mode"] = "learned"
    moe["router_penalty_context_weight"] = 0.0
    moe["router_detach_penalty_context"] = True
    moe["allow_skip"] = True
    moe["skip_cost"] = 0.15
    moe["skip_init_bias"] = -2.0
    moe["gate_temperature"] = 1.0
    moe["gate_noise_std"] = float(cand.gate_noise_std)
    moe["gate_soft_weight"] = 0.0
    moe["gate_prob_floor"] = 0.0
    moe["gate_entropy_weight"] = 0.0
    moe["gate_balance_weight"] = 0.0
    moe["gate_init_bias"] = {
        "enable": bool(cand.gate_init_bias_enable),
        "values": {"level": float(cand.gate_init_bias_level), "default": 0.0},
    }
    moe["residual_gate"] = {"enable": True, "alpha": 0.7}
    moe["pred_aware"] = {
        "enable": bool(cand.pred_aware),
        "use_pred_features": bool(cand.pred_aware),
        "use_penalty_input": False,
    }
    moe["penalty_ema"] = {"enable": bool(cand.penalty_ema), "decay": 0.9}


def build_config(
    base_cfg: dict[str, Any],
    *,
    dataset: str,
    cand: Candidate | None,
    role: str,
    out_dir: Path,
    device: str | None,
    epochs: int | None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("exp", {})["name"] = f"{dataset}_H96_{role if cand is None else cid(cand) + '_' + role}"
    fixed_protocol(cfg, out_dir=out_dir, device=device, epochs=epochs)
    apply_base_capacity(cfg, cand)
    if role == "base_no_moe":
        configure_base(cfg)
    elif role == "zero_lambda_residual":
        assert cand is not None
        configure_moe(cfg, cand, lambda_value=0.0)
    elif role == "residual_full":
        assert cand is not None
        configure_moe(cfg, cand, lambda_value=cand.lambda_init)
    else:
        raise ValueError(role)
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
        proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=f, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        return int(proc.returncode), elapsed, stdout_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    if not summary_path.exists():
        return 1, elapsed, "run_summary.json not found"
    return 0, elapsed, ""


def candidate_grid(dataset: str, base_penalties: tuple[str, ...], budget: str) -> list[Candidate]:
    common_pools = [
        base_penalties,
        ("range", "trend", "direction"),
        ("range", "delta", "trend"),
        ("level", "range", "trend", "direction"),
        ("amp_under", "delta", "diff_amp", "direction"),
        ("level", "delta", "d2_match", "diff_amp"),
        ("level", "delta", "diff_amp"),
        ("level", "delta", "trend", "direction"),
        ("delta", "trend", "direction"),
    ]
    if dataset == "ETTh2":
        common_pools.insert(1, ("jump", "amp_under", "level", "delta"))
    seen = set()
    pools = []
    for pool in common_pools:
        key = tuple(pool)
        if key not in seen:
            seen.add(key)
            pools.append(key)

    if budget == "capacity":
        pools = [
            base_penalties,
            ("delta", "trend", "direction"),
            ("level", "delta", "d2_match", "diff_amp"),
        ]
        presets = [
            {
                "lambda_init": 0.01,
                "alpha_scale": 1.4,
                "selection_policy": "val_mse_gate",
                "gate_max_scale": 1.25,
                "gate_init_scale": 0.8,
                "gate_init_bias_enable": True,
                "gate_init_bias_level": 1.0,
                "gate_noise_std": 0.1,
                "penalty_ema": False,
                "model_hidden_dim": 64,
                "model_dropout": 0.35,
                "mse_weight": 0.7,
                "mae_weight": 0.5,
            },
            {
                "lambda_init": 0.02,
                "alpha_scale": 1.8,
                "selection_policy": "val_mse_gate",
                "gate_max_scale": 1.5,
                "gate_init_scale": 1.0,
                "gate_init_bias_enable": True,
                "gate_init_bias_level": 1.5,
                "gate_noise_std": 0.2,
                "penalty_ema": True,
                "model_hidden_dim": 96,
                "model_dropout": 0.3,
                "mse_weight": 0.8,
                "mae_weight": 0.4,
            },
            {
                "lambda_init": 0.03,
                "alpha_scale": 2.0,
                "selection_policy": "val_mse_scale",
                "gate_max_scale": 1.5,
                "gate_init_scale": 1.0,
                "gate_init_bias_enable": False,
                "gate_noise_std": 0.1,
                "penalty_ema": False,
                "model_hidden_dim": 128,
                "model_dropout": 0.35,
                "mse_weight": 0.7,
                "mae_weight": 0.6,
            },
            {
                "lambda_init": 0.05,
                "alpha_scale": 2.2,
                "selection_policy": "val_mse_gate",
                "gate_max_scale": 1.75,
                "gate_init_scale": 1.2,
                "gate_init_bias_enable": True,
                "gate_init_bias_level": 2.0,
                "gate_noise_std": 0.2,
                "penalty_ema": True,
                "model_hidden_dim": 64,
                "model_dropout": 0.45,
                "mse_weight": 0.6,
                "mae_weight": 0.6,
            },
        ]
    elif budget == "focus":
        pools = [base_penalties]
        presets = [
            {
                "lambda_init": 0.01,
                "alpha_scale": 1.4,
                "selection_policy": "val_mse_scale",
                "selection_min_rel_improvement": 0.0,
                "gate_max_scale": 1.25,
                "gate_init_scale": 0.8,
                "gate_init_bias_enable": True,
                "gate_init_bias_level": 1.5,
                "gate_noise_std": 0.2,
                "penalty_ema": True,
            },
            {
                "lambda_init": 0.015,
                "alpha_scale": 1.5,
                "selection_policy": "val_mse_scale",
                "selection_min_rel_improvement": 0.0,
                "gate_max_scale": 1.5,
                "gate_init_scale": 1.0,
                "gate_init_bias_enable": True,
                "gate_init_bias_level": 1.5,
                "gate_noise_std": 0.2,
                "penalty_ema": True,
            },
            {
                "lambda_init": 0.02,
                "alpha_scale": 1.6,
                "selection_policy": "val_mse_gate",
                "selection_min_rel_improvement": 0.0,
                "gate_max_scale": 1.25,
                "gate_init_scale": 0.8,
                "gate_init_bias_enable": True,
                "gate_init_bias_level": 1.0,
                "gate_noise_std": 0.2,
                "penalty_ema": False,
            },
            {
                "lambda_init": 0.02,
                "alpha_scale": 1.8,
                "selection_policy": "val_mse_gate",
                "selection_min_rel_improvement": 0.0,
                "gate_max_scale": 1.5,
                "gate_init_scale": 1.0,
                "gate_init_bias_enable": False,
                "gate_noise_std": 0.1,
                "penalty_ema": False,
            },
            {
                "lambda_init": 0.03,
                "alpha_scale": 1.8,
                "selection_policy": "val_mse_scale",
                "selection_min_rel_improvement": 0.0,
                "gate_max_scale": 1.5,
                "gate_init_scale": 1.0,
                "gate_init_bias_enable": True,
                "gate_init_bias_level": 2.0,
                "gate_noise_std": 0.2,
                "penalty_ema": False,
            },
        ]
    elif budget == "guard":
        pools = [
            base_penalties,
            ("delta", "trend", "direction"),
            ("level", "delta", "d2_match", "diff_amp"),
            ("range", "trend", "direction"),
        ]
        presets = [
            {
                "lambda_init": 0.0,
                "alpha_scale": 1.4,
                "selection_policy": "val_mse_gate",
                "selection_min_rel_improvement": 0.0,
                "gate_max_scale": 1.5,
                "gate_init_scale": 1.0,
                "gate_scale_reg": 1.0e-5,
                "gate_train_fraction": 0.85,
                "gate_init_bias_enable": True,
                "gate_init_bias_level": 1.0,
                "gate_noise_std": 0.0,
                "penalty_ema": False,
                "penalty_guard_allow_multi": False,
                "channel_guard_enable": True,
            },
            {
                "lambda_init": 0.005,
                "alpha_scale": 1.8,
                "selection_policy": "val_mse_gate",
                "selection_min_rel_improvement": 0.0,
                "gate_max_scale": 1.75,
                "gate_init_scale": 1.0,
                "gate_scale_reg": 1.0e-5,
                "gate_train_fraction": 0.85,
                "gate_init_bias_enable": True,
                "gate_init_bias_level": 1.0,
                "gate_noise_std": 0.0,
                "penalty_ema": False,
                "penalty_guard_allow_multi": False,
                "channel_guard_enable": True,
            },
            {
                "lambda_init": 0.01,
                "alpha_scale": 2.2,
                "selection_policy": "val_mse_gate",
                "selection_min_rel_improvement": 0.0,
                "gate_max_scale": 2.0,
                "gate_init_scale": 1.2,
                "gate_scale_reg": 1.0e-5,
                "gate_train_fraction": 0.85,
                "gate_init_bias_enable": True,
                "gate_init_bias_level": 1.5,
                "gate_noise_std": 0.0,
                "penalty_ema": False,
                "penalty_guard_allow_multi": True,
                "channel_guard_enable": True,
            },
            {
                "lambda_init": 0.02,
                "alpha_scale": 2.5,
                "selection_policy": "val_mse_scale",
                "selection_min_rel_improvement": 0.0,
                "gate_max_scale": 2.0,
                "gate_init_scale": 1.2,
                "gate_scale_reg": 1.0e-5,
                "gate_train_fraction": 0.85,
                "gate_init_bias_enable": True,
                "gate_init_bias_level": 1.5,
                "gate_noise_std": 0.1,
                "penalty_ema": False,
                "penalty_guard_allow_multi": False,
                "channel_guard_enable": True,
            },
        ]
    elif budget == "smoke":
        pools = pools[:1]
        presets = [
            {
                "lambda_init": 0.005,
                "alpha_scale": 0.8,
                "selection_policy": "val_mse_gate_guarded",
                "selection_min_rel_improvement": 0.001,
                "gate_max_scale": 0.5,
                "gate_init_scale": 0.4,
                "gate_init_bias_enable": False,
                "gate_noise_std": 0.0,
                "penalty_ema": False,
            }
        ]
    else:
        presets = [
            {
                "lambda_init": 0.0,
                "alpha_scale": 0.8,
                "selection_policy": "val_mse_gate",
                "selection_min_rel_improvement": 0.0,
                "gate_max_scale": 0.5,
                "gate_init_scale": 0.4,
                "gate_init_bias_enable": False,
                "gate_noise_std": 0.0,
                "penalty_ema": False,
            },
            {
                "lambda_init": 0.001,
                "alpha_scale": 0.5,
                "selection_policy": "val_mse_gate_guarded",
                "selection_min_rel_improvement": 0.001,
                "gate_max_scale": 0.25,
                "gate_init_scale": 0.25,
                "gate_init_bias_enable": False,
                "gate_noise_std": 0.0,
                "penalty_ema": False,
            },
            {
                "lambda_init": 0.002,
                "alpha_scale": 1.1,
                "selection_policy": "val_mse_gate",
                "selection_min_rel_improvement": 0.0,
                "gate_max_scale": 1.0,
                "gate_init_scale": 0.6,
                "gate_init_bias_enable": True,
                "gate_init_bias_level": 1.0,
                "gate_noise_std": 0.1,
                "penalty_ema": False,
            },
            {
                "lambda_init": 0.003,
                "alpha_scale": 0.8,
                "selection_policy": "val_mse_gate_guarded",
                "selection_min_rel_improvement": 0.0005,
                "gate_max_scale": 0.5,
                "gate_init_scale": 0.4,
                "gate_init_bias_enable": False,
                "gate_noise_std": 0.0,
                "penalty_ema": False,
            },
            {
                "lambda_init": 0.005,
                "alpha_scale": 1.0,
                "selection_policy": "val_mse_gate",
                "selection_min_rel_improvement": 0.0,
                "gate_max_scale": 0.75,
                "gate_init_scale": 0.6,
                "gate_init_bias_enable": True,
                "gate_init_bias_level": 1.0,
                "gate_noise_std": 0.1,
                "penalty_ema": False,
            },
            {
                "lambda_init": 0.01,
                "alpha_scale": 1.2,
                "selection_policy": "val_mse_scale",
                "selection_min_rel_improvement": 0.0,
                "gate_max_scale": 1.0,
                "gate_init_scale": 0.6,
                "gate_init_bias_enable": True,
                "gate_init_bias_level": 1.5,
                "gate_noise_std": 0.2,
                "penalty_ema": True,
            },
        ]

    candidates: list[Candidate] = []
    for pool, preset in itertools.product(pools, presets):
        name = f"p{'_'.join(pool)}_lam{preset['lambda_init']}_a{preset['alpha_scale']}"
        candidates.append(Candidate(name=name, penalties=tuple(pool), **preset))
    return candidates


def run_dataset(args: argparse.Namespace, dataset: str, base_cfg_path: Path, out_root: Path) -> list[dict[str, Any]]:
    base_cfg = read_yaml(base_cfg_path)
    base_penalties = tuple((base_cfg.get("penalties", {}) or {}).get("enabled", []))
    if not base_penalties:
        base_penalties = tuple((base_cfg.get("moe", {}) or {}).get("lambda_init", {}).keys())
    if not base_penalties:
        raise ValueError(f"No base penalties found for {dataset}")
    candidates = candidate_grid(dataset, base_penalties, args.budget)
    if args.max_candidates is not None:
        candidates = candidates[: int(args.max_candidates)]

    rows_path = out_root / "results.csv"
    rows = read_csv(rows_path) if args.reuse_existing else []
    done = {(r.get("dataset"), r.get("candidate")) for r in rows if r.get("status") == "ok"}

    shared_base = args.budget != "capacity"
    base_out = out_root / "runs" / dataset / "base_no_moe"
    base_summary: dict[str, Any] | None = None
    base_sec = 0.0
    if shared_base:
        base_cfg_out = out_root / "configs" / dataset / "base_no_moe.yaml"
        base_run_cfg = build_config(
            base_cfg,
            dataset=dataset,
            cand=None,
            role="base_no_moe",
            out_dir=base_out,
            device=args.device,
            epochs=args.epochs,
        )
        write_yaml(base_cfg_out, base_run_cfg)
        base_code, base_sec, base_err = run_train(base_cfg_out, base_out, reuse_existing=args.reuse_existing)
        if base_code != 0:
            raise RuntimeError(f"{dataset} base failed: {base_err}")
        base_summary = read_json(base_out / "run_summary.json")

    for cand in candidates:
        key = (dataset, cid(cand))
        if key in done and args.reuse_existing:
            print(f"[skip] {dataset} {cid(cand)}", flush=True)
            continue
        run_dir = out_root / "runs" / dataset / cid(cand)
        cfg_dir = out_root / "configs" / dataset / cid(cand)
        if shared_base:
            cand_base_out = base_out
            cand_base_summary = base_summary
            cand_base_sec = base_sec
        else:
            cand_base_out = run_dir / "base_no_moe"
            cand_base_cfg_path = cfg_dir / "base_no_moe.yaml"
            cand_base_cfg = build_config(
                base_cfg,
                dataset=dataset,
                cand=cand,
                role="base_no_moe",
                out_dir=cand_base_out,
                device=args.device,
                epochs=args.epochs,
            )
            write_yaml(cand_base_cfg_path, cand_base_cfg)
            base_code, cand_base_sec, base_err = run_train(
                cand_base_cfg_path,
                cand_base_out,
                reuse_existing=args.reuse_existing,
            )
            if base_code != 0:
                row = {
                    "status": "error",
                    "dataset": dataset,
                    "candidate": cid(cand),
                    "penalties": "|".join(cand.penalties),
                    "base_out_dir": str(cand_base_out),
                    "wrapper_sec": cand_base_sec,
                    "error": f"base_no_moe failed: {base_err}",
                }
                rows = [r for r in rows if not (r.get("dataset") == dataset and r.get("candidate") == cid(cand))]
                rows.append(row)
                write_csv(rows_path, rows)
                continue
            cand_base_summary = read_json(cand_base_out / "run_summary.json")
        zero_out = run_dir / "zero_lambda_residual"
        full_out = run_dir / "residual_full"
        zero_cfg_path = cfg_dir / "zero_lambda_residual.yaml"
        full_cfg_path = cfg_dir / "residual_full.yaml"
        zero_cfg = build_config(
            base_cfg,
            dataset=dataset,
            cand=cand,
            role="zero_lambda_residual",
            out_dir=zero_out,
            device=args.device,
            epochs=args.epochs,
        )
        full_cfg = build_config(
            base_cfg,
            dataset=dataset,
            cand=cand,
            role="residual_full",
            out_dir=full_out,
            device=args.device,
            epochs=args.epochs,
        )
        write_yaml(zero_cfg_path, zero_cfg)
        write_yaml(full_cfg_path, full_cfg)
        print(f"[run] {dataset} {cid(cand)}", flush=True)
        zero_code, zero_sec, zero_err = run_train(zero_cfg_path, zero_out, reuse_existing=args.reuse_existing)
        full_code, full_sec, full_err = run_train(full_cfg_path, full_out, reuse_existing=args.reuse_existing)
        error = zero_err or full_err
        row: dict[str, Any] = {
            "status": "error" if error else "ok",
            "dataset": dataset,
            "candidate": cid(cand),
            "penalties": "|".join(cand.penalties),
            "lambda_init": cand.lambda_init,
            "alpha_scale": cand.alpha_scale,
            "feature_mode": cand.feature_mode,
            "residual_clip": cand.residual_clip,
            "selection_policy": cand.selection_policy,
            "selection_min_rel_improvement": cand.selection_min_rel_improvement,
            "gate_max_scale": cand.gate_max_scale,
            "gate_init_scale": cand.gate_init_scale,
            "gate_scale_reg": cand.gate_scale_reg,
            "gate_train_fraction": cand.gate_train_fraction,
            "gate_init_bias_enable": cand.gate_init_bias_enable,
            "gate_init_bias_level": cand.gate_init_bias_level,
            "gate_noise_std": cand.gate_noise_std,
            "pred_aware": cand.pred_aware,
            "penalty_ema": cand.penalty_ema,
            "penalty_guard_allow_multi": cand.penalty_guard_allow_multi,
            "channel_guard_enable": cand.channel_guard_enable,
            "model_hidden_dim": cand.model_hidden_dim,
            "model_dropout": cand.model_dropout,
            "mse_weight": cand.mse_weight,
            "mae_weight": cand.mae_weight,
            "base_out_dir": str(cand_base_out),
            "zero_out_dir": str(zero_out),
            "full_out_dir": str(full_out),
            "zero_config": str(zero_cfg_path),
            "full_config": str(full_cfg_path),
            "wrapper_sec": cand_base_sec + zero_sec + full_sec,
            "error": error,
        }
        if not error:
            zero_summary = read_json(zero_out / "run_summary.json")
            full_summary = read_json(full_out / "run_summary.json")
            assert cand_base_summary is not None
            base_mse = metric(cand_base_summary, "test", "avg_mse")
            zero_mse = metric(zero_summary, "test", "avg_mse")
            full_mse = metric(full_summary, "test", "avg_mse")
            zero_gain = 100.0 * (base_mse - zero_mse) / max(abs(base_mse), 1.0e-12)
            full_gain = 100.0 * (base_mse - full_mse) / max(abs(base_mse), 1.0e-12)
            full_vs_zero = zero_mse - full_mse
            row.update(
                {
                    "base_test_mse": base_mse,
                    "zero_test_mse": zero_mse,
                    "full_test_mse": full_mse,
                    "zero_gain_pct": zero_gain,
                    "full_gain_pct": full_gain,
                    "full_vs_zero_mse": full_vs_zero,
                    "hit_target": bool(zero_gain > 0.0 and full_vs_zero > 0.0 and full_gain >= TARGET_GAIN_PCT),
                    "base_test_mae": metric(cand_base_summary, "test", "avg_mae"),
                    "zero_test_mae": metric(zero_summary, "test", "avg_mae"),
                    "full_test_mae": metric(full_summary, "test", "avg_mae"),
                    "base_val_mse": metric(cand_base_summary, "val", "avg_mse"),
                    "zero_val_mse": metric(zero_summary, "val", "avg_mse"),
                    "full_val_mse": metric(full_summary, "val", "avg_mse"),
                    "base_best_epoch": best_epoch(cand_base_summary),
                    "zero_best_epoch": best_epoch(zero_summary),
                    "full_best_epoch": best_epoch(full_summary),
                }
            )
        rows = [r for r in rows if not (r.get("dataset") == dataset and r.get("candidate") == cid(cand))]
        rows.append(row)
        write_csv(rows_path, rows)
        print(
            f"  -> {row['status']} zero_gain={row.get('zero_gain_pct', '')} "
            f"full_gain={row.get('full_gain_pct', '')} full_vs_zero={row.get('full_vs_zero_mse', '')}",
            flush=True,
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "etth_h96_moe_constraint_search")
    parser.add_argument("--datasets", type=str, default="ETTh1,ETTh2")
    parser.add_argument("--budget", choices=["smoke", "local", "focus", "guard", "capacity"], default="local")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    out_root = resolve(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    config_map = {
        "ETTh1": ROOT / "configs" / "ETTh1.yaml",
        "ETTh2": ROOT / "configs" / "ETTh2.yaml",
    }
    for dataset in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        if dataset not in config_map:
            raise ValueError(f"Unsupported dataset: {dataset}")
        run_dataset(args, dataset, config_map[dataset], out_root)
    print(f"Saved: {out_root / 'results.csv'}")


if __name__ == "__main__":
    main()
