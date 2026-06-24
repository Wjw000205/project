import argparse
import copy
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PENALTY_DEFAULTS = {
    "amp": 0.1,
    "level": 0.1,
    "amp_under": 0.1,
    "range": 0.03,
    "trend": 0.05,
    "delta": 0.1,
    "d2_match": 0.1,
    "diff_amp": 0.1,
    "direction": 0.1,
    "jump": 0.1,
    "corr": 0.1,
    "jitter": 0.1,
    "smooth": 0.1,
}


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Path, cfg: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)


def deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)
    return dst


def slug(text: str, max_len: int = 100) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
    while "__" in safe:
        safe = safe.replace("__", "_")
    safe = safe.strip("_") or "candidate"
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    keep = max(8, max_len - len(digest) - 1)
    return f"{safe[:keep].rstrip('_')}_{digest}"


def ftag(value: Any) -> str:
    if isinstance(value, str):
        return value.replace(".", "p").replace("-", "m")
    number = float(value)
    text = f"{number:.0e}" if 0 < abs(number) < 0.001 else f"{number:g}"
    return text.replace("+", "").replace("-", "m").replace(".", "p")


def lambda_map(names: Sequence[str], scale: float) -> Dict[str, float]:
    return {name: float(PENALTY_DEFAULTS.get(name, 0.1) * scale) for name in names}


def zero_map(names: Sequence[str]) -> Dict[str, float]:
    return {name: 0.0 for name in names}


def none_schedule(names: Sequence[str]) -> Dict[str, str]:
    return {name: "none" for name in names}


def disable_leaky_or_external_paths(cfg: Dict[str, Any], out_dir: Path) -> None:
    cfg.setdefault("normalize", {})
    cfg["normalize"]["global_zscore"] = True
    cfg["normalize"]["train_only"] = True

    cfg.setdefault("cluster", {})
    cfg["cluster"]["train_only"] = True

    cfg.setdefault("corr", {})
    cfg["corr"]["save_path"] = str(out_dir / "corr.npy")

    cfg.setdefault("plot", {})
    cfg["plot"]["enable"] = False

    cfg.setdefault("portrait", {})
    cfg["portrait"]["enable"] = False
    cfg["portrait"]["out_dir"] = str(out_dir / "cluster_portraits")


    cfg.setdefault("eval", {})
    cfg["eval"]["skip_test"] = False


    cfg.setdefault("memory", {})
    cfg["memory"]["enable"] = False
    cfg["memory"]["save_checkpoint"] = False
    cfg["memory"]["path"] = str(out_dir / "cluster_memory.pt")
    cfg["memory"]["checkpoint_path"] = str(out_dir / "best_checkpoint.pt")


def fixed_h96_base(cfg: Dict[str, Any], out_dir: Path, device: Optional[str]) -> None:
    cfg.setdefault("exp", {})
    cfg["exp"]["name"] = out_dir.name
    cfg["exp"]["out_dir"] = str(out_dir)
    if device:
        cfg["exp"]["device"] = device
    cfg["exp"]["seed"] = int(cfg["exp"].get("seed", 2026))
    cfg["exp"]["deterministic"] = True

    cfg.setdefault("data", {})
    cfg["data"]["csv_path"] = "data/ETTh1.csv"
    cfg["data"]["date_col"] = int(cfg["data"].get("date_col", 0))
    cfg["data"]["max_rows"] = 14400
    cfg["data"]["train_ratio"] = 0.6
    cfg["data"]["val_ratio"] = 0.2
    cfg["data"]["test_ratio"] = 0.2

    cfg.setdefault("window", {})
    cfg["window"]["input_len"] = int(cfg["window"].get("input_len", 336))
    cfg["window"]["pred_len"] = 96
    cfg["window"]["past_context"] = True

    cfg.setdefault("model", {})
    cfg["model"]["predictor"] = "dlinear"
    cfg["model"]["dlinear_kernel_size"] = int(cfg["model"].get("dlinear_kernel_size", 25))

    cfg.setdefault("train", {})
    cfg["train"]["epochs"] = int(cfg["train"].get("epochs", 100))
    cfg["train"]["batch_size"] = int(cfg["train"].get("batch_size", 64))
    cfg["train"]["lr"] = float(cfg["train"].get("lr", 1.0e-3))
    cfg["train"]["mse_weight"] = float(cfg["train"].get("mse_weight", 0.9))
    cfg["train"]["selection_metric"] = str(cfg["train"].get("selection_metric", "val_mse"))
    cfg["train"]["weight_decay"] = float(cfg["train"].get("weight_decay", 2.0e-4))
    cfg["train"]["penalty_warmup_epochs"] = int(cfg["train"].get("penalty_warmup_epochs", 15))
    cfg["train"].setdefault("mae_objective", {})
    cfg["train"]["mae_objective"]["enable"] = bool(cfg["train"]["mae_objective"].get("enable", True))
    cfg["train"]["mae_objective"]["kind"] = str(cfg["train"]["mae_objective"].get("kind", "l1"))
    cfg["train"]["mae_objective"]["weight"] = float(cfg["train"]["mae_objective"].get("weight", 0.4))
    cfg["train"]["mae_objective"]["warmup_epochs"] = int(cfg["train"]["mae_objective"].get("warmup_epochs", 5))

    disable_leaky_or_external_paths(cfg, out_dir)


def moe_variant(
    penalties: Sequence[str],
    lambda_scale: float,
    alpha_scale: float,
    selection_scale_max: float,
    selection_scale_steps: int = 31,
    specialization_weight: float = 0.1,
    norm_weight: float = 0.0,
    gate_max_scale: float = 0.1,
    scale_reg: float = 5.0e-4,
    residual_clip: float = 0.0,
    feature_mode: str = "safe_augmented",
    dynamic_enable: bool = True,
    gate_entropy_weight: float = 0.0,
    gate_balance_weight: float = 0.0,
) -> Dict[str, Any]:
    penalties = tuple(penalties)
    return {
        "penalties": {"enabled": list(penalties)},
        "moe": {
            "enable": True,
            "topk": 1,
            "freeze_lambda": False,
            "gate_hidden_dim": 32,
            "select_ranks": [1],
            "detach_penalty_grad": True,
            "lambda_init": lambda_map(penalties, lambda_scale),
            "lambda_min": zero_map(penalties),
            "lambda_schedule": none_schedule(penalties),
            "pred_side_residual": {
                "enable": True,
                "feature_mode": feature_mode,
                "residual_clip": float(residual_clip),
                "corrector_hidden": 32,
                "init_alpha": -3.0,
                "alpha_scale": float(alpha_scale),
                "specialization_weight": float(specialization_weight),
                "norm_weight": float(norm_weight),
                "use_y_base_input": True,
                "intervention_enable": False,
                "intervention_init": -2.0,
                "intervention_weight": 1.0e-3,
                "detach_routed_penalty_pred": False,
                "selection_policy": "val_mse_scale",
                "selection_min_abs_improvement": 0.0,
                "selection_min_rel_improvement": 0.0,
                "selection_scale_min": 0.0,
                "selection_scale_max": float(selection_scale_max),
                "selection_scale_steps": int(selection_scale_steps),
            },
            "dynamic_lambda": {
                "enable": bool(dynamic_enable),
                "mode": "multiscale",
                "hidden_dim": 32,
                "segment_bins": [4, 8],
                "max_factor": 1.5,
                "mix": 0.6,
                "dropout": 0.0,
                "reg_weight": 1.0e-4,
            },
            "learnable_lambda": {"enable": False},
            "gate_entropy_weight": float(gate_entropy_weight),
            "gate_balance_weight": float(gate_balance_weight),
            "gate_route_on_penalty_only": True,
            "router_mode": "learned",
            "router_penalty_context_weight": 0.0,
            "router_detach_penalty_context": True,
            "allow_skip": True,
            "skip_cost": 0.15,
            "skip_init_bias": -2.0,
            "gate_temperature": 1.0,
            "gate_noise_std": 0.2,
            "gate_init_bias": {"enable": False, "values": {"default": 0.0}},
            "gate_soft_weight": 0.0,
            "gate_prob_floor": 0.0,
            "gate_entropy_target_frac": 0.7,
            "gate_logit_clip": 5.0,
        },
        "train": {"penalty_warmup_epochs": 15},
    }


def off_variant(penalties: Sequence[str]) -> Dict[str, Any]:
    return {
        "penalties": {"enabled": list(penalties)},
        "moe": {
            "enable": False,
            "dynamic_lambda": {"enable": False},
            "learnable_lambda": {"enable": False},
            "pred_side_residual": {"enable": False, "selection_policy": "none"},
            "gate_entropy_weight": 0.0,
            "gate_balance_weight": 0.0,
        },
    }


def base_specs() -> List[Tuple[str, Dict[str, Any]]]:
    specs: List[Tuple[str, Dict[str, Any]]] = []

    def add(name: str, patch: Dict[str, Any]) -> None:
        specs.append((name, patch))

    add("current_valmse", {})
    add("select_valmae", {"train": {"selection_metric": "val_mae"}})
    for input_len in [192, 256, 512, 720]:
        add(f"input{input_len}", {"window": {"input_len": input_len}})
    for kernel in [13, 17, 21, 31, 37, 49]:
        add(f"kernel{kernel}", {"model": {"dlinear_kernel_size": kernel}})
    for weight in [0.0, 0.2, 0.6, 0.8]:
        add(
            f"mae{ftag(weight)}",
            {"train": {"mae_objective": {"enable": weight > 0.0, "weight": weight}}},
        )
    for weight_decay in [0.0, 1.0e-4, 5.0e-4, 1.0e-3]:
        add(f"wd{ftag(weight_decay)}", {"train": {"weight_decay": weight_decay}})
    for batch_size in [32, 128]:
        add(f"batch{batch_size}", {"train": {"batch_size": batch_size}})
    for lr in [5.0e-4, 8.0e-4, 1.2e-3]:
        add(f"lr{ftag(lr)}", {"train": {"lr": lr}})
    add("input512_kernel37", {"window": {"input_len": 512}, "model": {"dlinear_kernel_size": 37}})
    add("input512_kernel49", {"window": {"input_len": 512}, "model": {"dlinear_kernel_size": 49}})
    add("input720_kernel49", {"window": {"input_len": 720}, "model": {"dlinear_kernel_size": 49}})
    add(
        "valmae_input512_kernel37",
        {
            "window": {"input_len": 512},
            "model": {"dlinear_kernel_size": 37},
            "train": {"selection_metric": "val_mae"},
        },
    )
    add(
        "valmae_kernel31_mae02",
        {
            "model": {"dlinear_kernel_size": 31},
            "train": {"selection_metric": "val_mae", "mae_objective": {"enable": True, "weight": 0.2}},
        },
    )
    for kernel in [13, 17, 21, 49]:
        add(
            f"kernel{kernel}_wd0p001",
            {"model": {"dlinear_kernel_size": kernel}, "train": {"weight_decay": 1.0e-3}},
        )
    for kernel in [13, 21, 49]:
        add(
            f"kernel{kernel}_wd5em04",
            {"model": {"dlinear_kernel_size": kernel}, "train": {"weight_decay": 5.0e-4}},
        )
    return specs


def moe_specs(names: Sequence[str]) -> List[Tuple[str, Sequence[str], Dict[str, Any]]]:
    available: Dict[str, Tuple[Sequence[str], Dict[str, Any]]] = {
        "best": (
            ("delta", "trend", "direction"),
            moe_variant(
                penalties=("delta", "trend", "direction"),
                lambda_scale=0.02,
                alpha_scale=0.35,
                selection_scale_max=0.8,
            ),
        ),
        "best_a45": (
            ("delta", "trend", "direction"),
            moe_variant(
                penalties=("delta", "trend", "direction"),
                lambda_scale=0.02,
                alpha_scale=0.45,
                selection_scale_max=0.9,
            ),
        ),
        "best_reg": (
            ("delta", "trend", "direction"),
            moe_variant(
                penalties=("delta", "trend", "direction"),
                lambda_scale=0.02,
                alpha_scale=0.35,
                selection_scale_max=0.8,
                specialization_weight=0.05,
                norm_weight=1.0e-4,
            ),
        ),
        "multi5": (
            ("level", "range", "delta", "trend", "direction"),
            moe_variant(
                penalties=("level", "range", "delta", "trend", "direction"),
                lambda_scale=0.015,
                alpha_scale=0.35,
                selection_scale_max=0.8,
                gate_balance_weight=0.01,
            ),
        ),
        "legacy": (
            ("delta", "trend", "direction"),
            moe_variant(
                penalties=("delta", "trend", "direction"),
                lambda_scale=0.02,
                alpha_scale=0.35,
                selection_scale_max=0.8,
                feature_mode="legacy",
            ),
        ),
        "lrt_dir": (
            ("level", "range", "trend", "direction"),
            moe_variant(
                penalties=("level", "range", "trend", "direction"),
                lambda_scale=0.015,
                alpha_scale=0.35,
                selection_scale_max=0.8,
                gate_balance_weight=0.01,
            ),
        ),
        "ampdiff_dir": (
            ("amp_under", "delta", "diff_amp", "direction"),
            moe_variant(
                penalties=("amp_under", "delta", "diff_amp", "direction"),
                lambda_scale=0.015,
                alpha_scale=0.35,
                selection_scale_max=0.8,
                gate_balance_weight=0.01,
            ),
        ),
        "shape5": (
            ("level", "range", "delta", "d2_match", "diff_amp"),
            moe_variant(
                penalties=("level", "range", "delta", "d2_match", "diff_amp"),
                lambda_scale=0.012,
                alpha_scale=0.35,
                selection_scale_max=0.8,
                gate_balance_weight=0.01,
            ),
        ),
        "corr_dtd": (
            ("corr", "delta", "trend", "direction"),
            moe_variant(
                penalties=("corr", "delta", "trend", "direction"),
                lambda_scale=0.01,
                alpha_scale=0.35,
                selection_scale_max=0.8,
                gate_balance_weight=0.01,
            ),
        ),
        "range_dtd": (
            ("range", "delta", "trend", "direction"),
            moe_variant(
                penalties=("range", "delta", "trend", "direction"),
                lambda_scale=0.015,
                alpha_scale=0.35,
                selection_scale_max=0.8,
                gate_balance_weight=0.01,
            ),
        ),
        "lt_dir": (
            ("level", "trend", "direction"),
            moe_variant(
                penalties=("level", "trend", "direction"),
                lambda_scale=0.02,
                alpha_scale=0.35,
                selection_scale_max=0.8,
            ),
        ),
        "td": (
            ("trend", "direction"),
            moe_variant(
                penalties=("trend", "direction"),
                lambda_scale=0.02,
                alpha_scale=0.35,
                selection_scale_max=0.8,
            ),
        ),
        "dd": (
            ("delta", "direction"),
            moe_variant(
                penalties=("delta", "direction"),
                lambda_scale=0.02,
                alpha_scale=0.35,
                selection_scale_max=0.8,
            ),
        ),
        "ld": (
            ("level", "delta"),
            moe_variant(
                penalties=("level", "delta"),
                lambda_scale=0.02,
                alpha_scale=0.35,
                selection_scale_max=0.8,
            ),
        ),
        "ldt": (
            ("level", "delta", "trend"),
            moe_variant(
                penalties=("level", "delta", "trend"),
                lambda_scale=0.02,
                alpha_scale=0.35,
                selection_scale_max=0.8,
            ),
        ),
        "ld_dir": (
            ("level", "delta", "direction"),
            moe_variant(
                penalties=("level", "delta", "direction"),
                lambda_scale=0.02,
                alpha_scale=0.35,
                selection_scale_max=0.8,
            ),
        ),
        "ld_d2": (
            ("level", "delta", "d2_match"),
            moe_variant(
                penalties=("level", "delta", "d2_match"),
                lambda_scale=0.02,
                alpha_scale=0.35,
                selection_scale_max=0.8,
            ),
        ),
        "shape5_nobal": (
            ("level", "range", "delta", "d2_match", "diff_amp"),
            moe_variant(
                penalties=("level", "range", "delta", "d2_match", "diff_amp"),
                lambda_scale=0.012,
                alpha_scale=0.35,
                selection_scale_max=0.8,
                gate_balance_weight=0.0,
            ),
        ),
        "shape5_a25": (
            ("level", "range", "delta", "d2_match", "diff_amp"),
            moe_variant(
                penalties=("level", "range", "delta", "d2_match", "diff_amp"),
                lambda_scale=0.012,
                alpha_scale=0.25,
                selection_scale_max=0.8,
                gate_balance_weight=0.01,
            ),
        ),
        "shape5_a45": (
            ("level", "range", "delta", "d2_match", "diff_amp"),
            moe_variant(
                penalties=("level", "range", "delta", "d2_match", "diff_amp"),
                lambda_scale=0.012,
                alpha_scale=0.45,
                selection_scale_max=0.8,
                gate_balance_weight=0.01,
            ),
        ),
        "shape5_s06": (
            ("level", "range", "delta", "d2_match", "diff_amp"),
            moe_variant(
                penalties=("level", "range", "delta", "d2_match", "diff_amp"),
                lambda_scale=0.012,
                alpha_scale=0.35,
                selection_scale_max=0.6,
                gate_balance_weight=0.01,
            ),
        ),
        "shape5_s10": (
            ("level", "range", "delta", "d2_match", "diff_amp"),
            moe_variant(
                penalties=("level", "range", "delta", "d2_match", "diff_amp"),
                lambda_scale=0.012,
                alpha_scale=0.35,
                selection_scale_max=1.0,
                gate_balance_weight=0.01,
            ),
        ),
        "shape5_lam006": (
            ("level", "range", "delta", "d2_match", "diff_amp"),
            moe_variant(
                penalties=("level", "range", "delta", "d2_match", "diff_amp"),
                lambda_scale=0.006,
                alpha_scale=0.35,
                selection_scale_max=0.8,
                gate_balance_weight=0.01,
            ),
        ),
        "shape5_lam02": (
            ("level", "range", "delta", "d2_match", "diff_amp"),
            moe_variant(
                penalties=("level", "range", "delta", "d2_match", "diff_amp"),
                lambda_scale=0.02,
                alpha_scale=0.35,
                selection_scale_max=0.8,
                gate_balance_weight=0.01,
            ),
        ),
    }
    specs: List[Tuple[str, Sequence[str], Dict[str, Any]]] = []
    for name in names:
        if name not in available:
            raise ValueError(f"Unknown MoE variant '{name}'. Available: {', '.join(sorted(available))}")
        penalties, patch = available[name]
        specs.append((name, penalties, patch))
    return specs


def build_config(
    base_cfg: Dict[str, Any],
    base_patch: Dict[str, Any],
    moe_patch: Dict[str, Any],
    out_dir: Path,
    device: Optional[str],
    epochs: Optional[int],
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    fixed_h96_base(cfg, out_dir, device)
    deep_update(cfg, base_patch)
    deep_update(cfg, moe_patch)
    if epochs is not None:
        cfg.setdefault("train", {})["epochs"] = int(epochs)
    cfg["exp"]["name"] = out_dir.name
    cfg["exp"]["out_dir"] = str(out_dir)
    disable_leaky_or_external_paths(cfg, out_dir)
    return cfg


def summary_path(config_path: Path) -> Path:
    cfg = load_yaml(config_path)
    return resolve(str(cfg["exp"]["out_dir"])) / "run_summary.json"


def run_train(config_path: Path, python_exe: str, reuse_existing: bool) -> int:
    sp = summary_path(config_path)
    if reuse_existing and sp.exists():
        print(f"[reuse] {sp}", flush=True)
        return 0
    cmd = [python_exe, "-m", "src.train", "--config", str(config_path)]
    print("[run] " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode


def load_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sf(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":")) if value is not None else ""


def result_row(base_name: str, moe_name: str, off_cfg: Path, on_cfg: Path) -> Dict[str, Any]:
    off = load_summary(summary_path(off_cfg))
    on = load_summary(summary_path(on_cfg))
    off_test = off.get("test") or {}
    on_test = on.get("test") or {}
    off_mse = sf(off_test.get("avg_mse"))
    on_mse = sf(on_test.get("avg_mse"))
    gain = off_mse - on_mse if not math.isnan(off_mse) and not math.isnan(on_mse) else float("nan")
    on_window = on.get("windowing") or {}
    on_residual = on.get("moe_residual") or {}
    on_select = on.get("moe_residual_selection") or {}
    on_cfg_data = load_yaml(on_cfg)
    base_cfg_data = load_yaml(off_cfg)
    return {
        "base": base_name,
        "moe_variant": moe_name,
        "target_hit": bool(on_mse < 0.360 and gain > 0.0) if not math.isnan(on_mse) and not math.isnan(gain) else False,
        "moe_positive": bool(gain > 0.0) if not math.isnan(gain) else False,
        "off_mse": off_mse,
        "on_mse": on_mse,
        "gain_mse": gain,
        "off_mae": sf(off_test.get("avg_mae")),
        "on_mae": sf(on_test.get("avg_mae")),
        "input_len": (on_cfg_data.get("window") or {}).get("input_len", ""),
        "kernel": (on_cfg_data.get("model") or {}).get("dlinear_kernel_size", ""),
        "selection_metric": (on_cfg_data.get("train") or {}).get("selection_metric", ""),
        "mae_weight": ((on_cfg_data.get("train") or {}).get("mae_objective") or {}).get("weight", ""),
        "weight_decay": (on_cfg_data.get("train") or {}).get("weight_decay", ""),
        "batch_size": (on_cfg_data.get("train") or {}).get("batch_size", ""),
        "lr": (on_cfg_data.get("train") or {}).get("lr", ""),
        "penalties": compact_json((on_cfg_data.get("penalties") or {}).get("enabled", [])),
        "alpha_scale": ((on_cfg_data.get("moe") or {}).get("pred_side_residual") or {}).get("alpha_scale", ""),
        "selection_scale_max": ((on_cfg_data.get("moe") or {}).get("pred_side_residual") or {}).get("selection_scale_max", ""),
        "residual_enabled": on_residual.get("enabled", ""),
        "residual_channels": compact_json(on_select.get("residual_channels")),
        "route": compact_json(on_residual.get("effective_route_by_penalty")),
        "num_test_windows": on_window.get("num_test_windows", ""),
        "normalize_train_only": on_window.get("normalize_train_only", ""),
        "off_config": str(off_cfg),
        "on_config": str(on_cfg),
        "off_summary": str(summary_path(off_cfg)),
        "on_summary": str(summary_path(on_cfg)),
        "off_out_dir": str(resolve(str((base_cfg_data.get("exp") or {}).get("out_dir", "")))),
        "on_out_dir": str(resolve(str((on_cfg_data.get("exp") or {}).get("out_dir", "")))),
    }


def write_results(out_root: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    rows.sort(
        key=lambda r: (
            1 if math.isnan(sf(r.get("on_mse"))) else 0,
            sf(r.get("on_mse")) if not math.isnan(sf(r.get("on_mse"))) else float("inf"),
            str(r.get("base", "")),
            str(r.get("moe_variant", "")),
        )
    )
    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "search_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    positives = [r for r in rows if sf(r.get("gain_mse")) > 0.0]
    hits = [r for r in positives if sf(r.get("on_mse")) < 0.360]
    with (out_root / "search_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_by_on_mse": rows[0],
                "best_positive_moe": min(positives, key=lambda r: sf(r["on_mse"])) if positives else None,
                "target_hits": hits,
                "num_rows": len(rows),
                "num_positive_moe": len(positives),
            },
            f,
            indent=2,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/ETTh1.yaml")
    parser.add_argument("--out-root", default="outputs/etth1_96_moe_broad_pair_search")
    parser.add_argument("--device", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--base-limit", type=int, default=0)
    parser.add_argument("--base-start", type=int, default=0)
    parser.add_argument("--base-names", default="")
    parser.add_argument("--moe-variants", default="best")
    parser.add_argument("--exclude-input-len-changes", action="store_true")
    parser.add_argument("--off-per-variant", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    base_cfg = load_yaml(resolve(args.base_config))
    out_root = resolve(args.out_root)
    cfg_root = out_root / "configs"
    run_root = out_root / "runs"
    variants = moe_specs([name.strip() for name in args.moe_variants.split(",") if name.strip()])

    bases = base_specs()
    if args.exclude_input_len_changes:
        bases = [
            (name, patch)
            for name, patch in bases
            if "input_len" not in (patch.get("window") or {}) and not name.startswith("input")
        ]
    if args.base_names:
        wanted = {name.strip() for name in args.base_names.split(",") if name.strip()}
        bases = [(name, patch) for name, patch in bases if name in wanted]
    if args.base_start > 0:
        bases = bases[args.base_start :]
    if args.base_limit > 0:
        bases = bases[: args.base_limit]

    rows: List[Dict[str, Any]] = []
    existing_csv = out_root / "search_results.csv"
    if existing_csv.exists():
        with existing_csv.open("r", encoding="utf-8", newline="") as f:
            rows.extend(csv.DictReader(f))

    for base_index, (base_name, base_patch) in enumerate(bases, start=args.base_start + 1):
        base_slug = slug(f"{base_index:02d}_{base_name}")
        off_cfg_path = cfg_root / base_slug / "off.yaml"
        off_dir = run_root / base_slug / "off"
        first_penalties = variants[0][1]
        if not args.off_per_variant:
            write_yaml(
                off_cfg_path,
                build_config(base_cfg, base_patch, off_variant(first_penalties), off_dir, args.device, args.epochs),
            )
            rc = run_train(off_cfg_path, args.python, args.reuse_existing)
            if rc != 0:
                return rc
            off_summary = load_summary(summary_path(off_cfg_path))
            off_mse = sf((off_summary.get("test") or {}).get("avg_mse"))
            print(f"[off] {base_name} mse={off_mse:.6f}", flush=True)

        for moe_name, _penalties, patch in variants:
            if args.off_per_variant:
                off_cfg_path = cfg_root / base_slug / f"off_{moe_name}.yaml"
                off_dir = run_root / base_slug / f"off_{moe_name}"
                write_yaml(
                    off_cfg_path,
                    build_config(base_cfg, base_patch, off_variant(_penalties), off_dir, args.device, args.epochs),
                )
                rc = run_train(off_cfg_path, args.python, args.reuse_existing)
                if rc != 0:
                    return rc
                off_summary = load_summary(summary_path(off_cfg_path))
                off_mse = sf((off_summary.get("test") or {}).get("avg_mse"))
                print(f"[off] {base_name}/{moe_name} mse={off_mse:.6f}", flush=True)
            on_slug = slug(f"{base_index:02d}_{base_name}_{moe_name}")
            on_cfg_path = cfg_root / base_slug / f"{moe_name}.yaml"
            on_dir = run_root / base_slug / on_slug
            write_yaml(on_cfg_path, build_config(base_cfg, base_patch, patch, on_dir, args.device, args.epochs))
            rc = run_train(on_cfg_path, args.python, args.reuse_existing)
            if rc != 0:
                return rc
            row = result_row(base_name, moe_name, off_cfg_path, on_cfg_path)
            rows = [
                r
                for r in rows
                if not (r.get("base") == base_name and r.get("moe_variant") == moe_name)
            ]
            rows.append(row)
            write_results(out_root, rows)
            print(
                f"[on] {base_name}/{moe_name} mse={row['on_mse']:.6f} "
                f"gain={row['gain_mse']:.6f} hit={row['target_hit']}",
                flush=True,
            )

    write_results(out_root, rows)
    print(f"[done] wrote {out_root / 'search_results.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
