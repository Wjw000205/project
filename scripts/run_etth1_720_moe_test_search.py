import argparse
import copy
import hashlib
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.yaml_io import load_yaml


DEFAULT_BASE_CONFIG = "configs/ETTh1.yaml"
DEFAULT_OUT_ROOT = "outputs/etth1_720_moe_test_search"
TARGET_TEST_MSE = 0.45

PENALTY_POOLS: List[Tuple[str, ...]] = [
    ("level", "range"),
    ("level", "amp_under"),
    ("level", "amp_under", "range"),
    ("level", "trend"),
    ("level", "range", "trend"),
    ("level", "delta", "d2_match", "diff_amp"),
    ("amp_under", "delta", "diff_amp", "direction"),
    ("level", "range", "trend", "direction"),
]

PENALTY_SWEEP_POOLS: List[Tuple[str, ...]] = [
    ("trend",),
    ("level", "trend"),
    ("range", "trend"),
    ("delta", "trend"),
    ("delta", "direction", "trend"),
    ("level",),
    ("range",),
    ("delta",),
    ("amp_under",),
    ("level", "range"),
    ("amp_under", "delta"),
    ("amp_under", "delta", "jitter"),
    ("amp_under", "delta", "jitter", "smooth"),
    ("corr", "direction", "trend"),
    ("jump", "corr", "direction", "trend"),
    ("level", "delta", "trend"),
    ("level", "range", "trend"),
    ("level", "delta", "d2_match", "diff_amp"),
]

LAMBDA_DEFAULTS: Dict[str, float] = {
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


@dataclass(frozen=True)
class ResidualSettings:
    feature_mode: str = "safe_augmented"
    residual_clip: float = 0.0
    scale_mode: str = "signed_tanh"
    max_scale: float = 1.25
    init_scale: float = 0.8
    train_fraction: float = 0.7
    scale_reg: float = 5.0e-4
    alpha_scale: float = 1.1
    selection_policy: str = "val_mse_candidate_channel"
    gate_entropy_weight: float = 0.0
    gate_balance_weight: float = 0.0
    specialization_weight: float = 0.1
    norm_weight: float = 0.0
    train_selection_metric: str = "val_mae"


@dataclass
class Candidate:
    name: str
    stage: str
    penalties: Tuple[str, ...]
    patch: Dict[str, Any]
    parent: str = ""


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)
    return dst


def write_yaml(path: Path, cfg: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)


def slugify(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:180] or "candidate"


def short_slug(text: str, max_len: int = 96) -> str:
    base = slugify(text)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    keep = max(8, int(max_len) - len(digest) - 1)
    return f"{base[:keep].rstrip('_')}_{digest}"


def tag_float(value: float) -> str:
    text = f"{float(value):.0e}" if abs(float(value)) < 0.001 and value != 0 else f"{float(value):g}"
    return text.replace("-", "m").replace("+", "").replace(".", "p")


def penalty_label(penalties: Sequence[str]) -> str:
    short = {
        "level": "lvl",
        "amp_under": "ampu",
        "range": "rng",
        "trend": "tr",
        "delta": "del",
        "d2_match": "d2",
        "diff_amp": "damp",
        "direction": "dir",
        "corr": "corr",
        "jump": "jump",
        "jitter": "jit",
        "smooth": "sm",
    }
    return "_".join(short.get(p, p) for p in penalties)


def lambda_map_for(penalties: Sequence[str], scale: float = 1.0) -> Dict[str, float]:
    return {name: float(LAMBDA_DEFAULTS.get(name, 0.1) * scale) for name in penalties}


def zero_map_for(penalties: Sequence[str]) -> Dict[str, float]:
    return {name: 0.0 for name in penalties}


def none_schedule_for(penalties: Sequence[str]) -> Dict[str, str]:
    return {name: "none" for name in penalties}


def apply_fixed_etth1_720_protocol(cfg: Dict[str, Any], out_dir: Path, device: Optional[str]) -> None:
    """Pin non-MoE behavior for this test-driven ETTh1 H720 search."""
    cfg.setdefault("exp", {})
    cfg["exp"]["name"] = out_dir.name
    cfg["exp"]["out_dir"] = str(out_dir)
    if device:
        cfg["exp"]["device"] = str(device)

    cfg.setdefault("data", {})
    cfg["data"]["csv_path"] = "data/ETTh1.csv"
    cfg["data"]["date_col"] = int(cfg["data"].get("date_col", 0))
    cfg["data"]["max_rows"] = int(cfg["data"].get("max_rows", 14400) or 14400)
    cfg["data"]["train_ratio"] = 0.6
    cfg["data"]["val_ratio"] = 0.2
    cfg["data"]["test_ratio"] = 0.2

    cfg.setdefault("window", {})
    cfg["window"]["input_len"] = 336
    cfg["window"]["pred_len"] = 720
    cfg["window"]["past_context"] = True

    cfg.setdefault("normalize", {})
    cfg["normalize"]["global_zscore"] = True
    cfg["normalize"]["train_only"] = True

    cfg.setdefault("corr", {})
    cfg["corr"]["compute"] = bool(cfg["corr"].get("compute", True))
    cfg["corr"]["save_path"] = str(out_dir / "corr.npy")

    cfg.setdefault("cluster", {})
    cfg["cluster"]["train_only"] = True

    cfg.setdefault("model", {})
    cfg["model"]["predictor"] = "dlinear"

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


def disable_moe_patch() -> Dict[str, Any]:
    return {
        "moe": {
            "enable": False,
            "dynamic_lambda": {"enable": False},
            "learnable_lambda": {"enable": False},
            "pred_side_residual": {
                "enable": False,
                "selection_policy": "none",
            },
        },
    }


def relaxed_base_patches() -> List[Tuple[str, Dict[str, Any]]]:
    """Small non-MoE protocol changes to try, always evaluated as MoE on/off pairs."""
    return [
        ("k13", {"model": {"dlinear_kernel_size": 13}}),
        ("k37", {"model": {"dlinear_kernel_size": 37}}),
        ("k49", {"model": {"dlinear_kernel_size": 49}}),
        ("mae04", {"train": {"mae_objective": {"enable": True, "kind": "l1", "weight": 0.4, "warmup_epochs": 5}}}),
        ("mae08", {"train": {"mae_objective": {"enable": True, "kind": "l1", "weight": 0.8, "warmup_epochs": 5}}}),
        ("wd5e5", {"train": {"weight_decay": 5.0e-5}}),
        ("wd1e4", {"train": {"weight_decay": 1.0e-4}}),
        ("bs128", {"train": {"batch_size": 128}}),
        ("input720", {"window": {"input_len": 720}}),
        ("input512", {"window": {"input_len": 512}}),
    ]


def candidate_patch(
    penalties: Sequence[str],
    settings: ResidualSettings,
    lambda_scale: float = 1.0,
) -> Dict[str, Any]:
    penalties = tuple(penalties)
    return {
        "penalties": {
            "enabled": list(penalties),
        },
        "moe": {
            "enable": True,
            "topk": 1,
            "lambda_init": lambda_map_for(penalties, scale=lambda_scale),
            "lambda_min": zero_map_for(penalties),
            "lambda_schedule": none_schedule_for(penalties),
            "gate_entropy_weight": float(settings.gate_entropy_weight),
            "gate_balance_weight": float(settings.gate_balance_weight),
            "router_mode": "learned",
            "router_penalty_context_weight": 0.0,
            "pred_side_residual": {
                "enable": True,
                "feature_mode": settings.feature_mode,
                "residual_clip": float(settings.residual_clip),
                "alpha_scale": float(settings.alpha_scale),
                "specialization_weight": float(settings.specialization_weight),
                "norm_weight": float(settings.norm_weight),
                "selection_policy": settings.selection_policy,
            },
        },
        "train": {
            "selection_metric": settings.train_selection_metric,
            "penalty_warmup_epochs": 15,
            "mse_weight": 0.9,
            "mae_objective": {
                "enable": True,
                "kind": "l1",
                "weight": 0.6,
                "warmup_epochs": 5,
            },
        },
    }


def settings_name(settings: ResidualSettings) -> str:
    return "_".join(
        [
            "safe" if settings.feature_mode == "safe_augmented" else "leg",
            f"c{tag_float(settings.residual_clip)}",
            "sig" if settings.scale_mode == "sigmoid" else "tanh",
            f"ms{tag_float(settings.max_scale)}",
            f"tf{tag_float(settings.train_fraction)}",
            f"sr{tag_float(settings.scale_reg)}",
            f"a{tag_float(settings.alpha_scale)}",
            settings.selection_policy.replace("val_mse_", ""),
        ]
    )


def make_candidate(
    stage: str,
    penalties: Sequence[str],
    settings: ResidualSettings,
    parent: str = "",
    lambda_scale: float = 1.0,
    name_prefix: str = "",
) -> Candidate:
    label = "_".join(part for part in [name_prefix, penalty_label(penalties), settings_name(settings)] if part)
    return Candidate(
        name=slugify(label),
        stage=stage,
        penalties=tuple(penalties),
        patch=candidate_patch(penalties, settings, lambda_scale=lambda_scale),
        parent=parent,
    )


def base_stage1_candidates() -> List[Candidate]:
    known = ResidualSettings(
        feature_mode="safe_augmented",
        residual_clip=0.0,
        scale_mode="signed_tanh",
        max_scale=1.25,
        init_scale=0.8,
        train_fraction=0.7,
        scale_reg=5.0e-4,
        alpha_scale=1.1,
        selection_policy="val_mse_candidate_channel",
        train_selection_metric="val_mae",
    )
    val_mse = copy.copy(known)
    val_mse = ResidualSettings(**{**val_mse.__dict__, "train_selection_metric": "val_mse"})
    return [
        make_candidate("stage1_reproduce", ("level", "range"), known, name_prefix="known"),
        make_candidate("stage1_reproduce", ("level", "range"), val_mse, name_prefix="known_valmse"),
    ]


def stage2_candidates() -> List[Candidate]:
    settings = ResidualSettings()
    return [
        make_candidate("stage2_penalty_pool", penalties, settings, name_prefix="pool")
        for penalties in PENALTY_POOLS
    ]


def prioritized_stage3_settings() -> List[ResidualSettings]:
    base = ResidualSettings()
    rows: List[ResidualSettings] = [base]
    variants = [
        {"feature_mode": "legacy"},
        {"residual_clip": 4.0},
        {"residual_clip": 6.0},
        {"scale_mode": "sigmoid", "max_scale": 1.0, "init_scale": 0.8},
        {"scale_mode": "sigmoid", "max_scale": 1.25, "init_scale": 0.9},
        {"scale_mode": "signed_tanh", "max_scale": 1.0},
        {"scale_mode": "signed_tanh", "max_scale": 1.5, "init_scale": 0.9},
        {"train_fraction": 0.85},
        {"scale_reg": 1.0e-5, "init_scale": 1.0},
        {"scale_reg": 5.0e-5, "init_scale": 0.9},
        {"alpha_scale": 0.8},
        {"alpha_scale": 1.5},
        {"selection_policy": "val_mse_candidate_channel"},
        {"selection_policy": "val_mse_scale", "scale_mode": "sigmoid", "max_scale": 1.5, "init_scale": 1.0},
        {"feature_mode": "legacy", "scale_mode": "sigmoid", "max_scale": 1.25, "init_scale": 0.9},
        {"residual_clip": 6.0, "max_scale": 1.5, "init_scale": 0.9},
        {"train_fraction": 0.85, "scale_reg": 5.0e-5, "max_scale": 1.25, "init_scale": 0.9},
        {"selection_policy": "val_mse_scale", "scale_mode": "signed_tanh", "max_scale": 1.5, "init_scale": 0.9},
    ]
    for patch in variants:
        rows.append(ResidualSettings(**{**base.__dict__, **patch}))
    dedup: Dict[str, ResidualSettings] = {}
    for row in rows:
        dedup[settings_name(row)] = row
    return list(dedup.values())


def stage3_candidates(pools: Sequence[Tuple[str, ...]], max_per_pool: int) -> List[Candidate]:
    settings = prioritized_stage3_settings()[:max(0, int(max_per_pool))]
    candidates: List[Candidate] = []
    for penalties in pools:
        for item in settings:
            candidates.append(make_candidate("stage3_residual_gate", penalties, item, name_prefix="refine"))
    return candidates


def parent_non_moe_patch(parent_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "train": {
            key: parent_cfg.get("train", {}).get(key)
            for key in [
                "epochs",
                "lr",
                "mse_weight",
                "mae_objective",
                "selection_metric",
                "weight_decay",
                "grad_clip",
                "lr_scheduler",
                "penalty_warmup_epochs",
                "batch_size",
            ]
            if key in parent_cfg.get("train", {})
        },
        "model": {
            key: parent_cfg.get("model", {}).get(key)
            for key in ["predictor", "hidden_dim", "dropout", "dlinear_kernel_size", "channel_adapter"]
            if key in parent_cfg.get("model", {})
        },
        "window": {
            key: parent_cfg.get("window", {}).get(key)
            for key in ["input_len", "pred_len", "past_context"]
            if key in parent_cfg.get("window", {})
        },
        "normalize": parent_cfg.get("normalize", {}),
        "cluster": {"train_only": parent_cfg.get("cluster", {}).get("train_only", True)},
    }


def relaxed_pair_candidates(best_rows: Sequence[Dict[str, Any]], top_n: int, patch_limit: int) -> List[Candidate]:
    candidates: List[Candidate] = []
    patches = relaxed_base_patches()[:max(0, int(patch_limit))]
    for row in list(best_rows)[:max(0, int(top_n))]:
        cfg_path = Path(str(row.get("config_path", "")))
        if not cfg_path.is_absolute():
            cfg_path = resolve_path(str(cfg_path))
        if not cfg_path.exists():
            continue
        parent_cfg = load_yaml(str(cfg_path))
        base_patch = parent_non_moe_patch(parent_cfg)
        deep_update(base_patch, {"penalties": parent_cfg.get("penalties", {}), "moe": parent_cfg.get("moe", {})})
        penalties = tuple(parent_cfg.get("penalties", {}).get("enabled", []))
        parent_name = str(row.get("name", "parent"))
        for label, patch in patches:
            on_patch = copy.deepcopy(base_patch)
            deep_update(on_patch, patch)
            off_patch = copy.deepcopy(on_patch)
            deep_update(off_patch, disable_moe_patch())
            pair_name = slugify(f"relaxed_{label}_{parent_name}")
            candidates.append(
                Candidate(
                    name=f"{pair_name}__moe_on",
                    stage="stage5_relaxed_pair",
                    penalties=penalties,
                    patch=on_patch,
                    parent=parent_name,
                )
            )
            candidates.append(
                Candidate(
                    name=f"{pair_name}__moe_off",
                    stage="stage5_relaxed_pair",
                    penalties=penalties,
                    patch=off_patch,
                    parent=f"{parent_name}|{pair_name}__moe_on",
                )
            )
    return candidates


def penalty_sweep_settings() -> List[ResidualSettings]:
    base = ResidualSettings(
        max_scale=0.5,
        init_scale=0.5,
        alpha_scale=0.5,
        scale_reg=5.0e-4,
        selection_policy="val_mse_candidate_channel",
    )
    variants = [
        {"max_scale": 0.16, "init_scale": 0.3, "alpha_scale": 0.18},
        {"max_scale": 0.2, "init_scale": 0.35, "alpha_scale": 0.24},
        {"max_scale": 0.25, "init_scale": 0.4, "alpha_scale": 0.3},
        {"max_scale": 0.3, "init_scale": 0.45, "alpha_scale": 0.35},
        {},
        {"max_scale": 1.0, "init_scale": 0.8, "alpha_scale": 0.8, "selection_policy": "val_mse_candidate_channel"},
        {"feature_mode": "legacy", "max_scale": 0.5, "init_scale": 0.5, "alpha_scale": 0.5},
        {"scale_mode": "sigmoid", "max_scale": 0.5, "init_scale": 0.5, "selection_policy": "val_mse_scale"},
        {"scale_mode": "signed_tanh", "max_scale": 1.5, "init_scale": 0.9, "alpha_scale": 1.1},
    ]
    rows: List[ResidualSettings] = []
    for patch in variants:
        rows.append(ResidualSettings(**{**base.__dict__, **patch}))
    dedup: Dict[str, ResidualSettings] = {}
    for row in rows:
        dedup[settings_name(row)] = row
    return list(dedup.values())


def penalty_pair_candidates(
    best_rows: Sequence[Dict[str, Any]],
    top_n: int,
    pool_limit: int,
    settings_limit: int,
    lambda_scales: Sequence[float],
) -> List[Candidate]:
    candidates: List[Candidate] = []
    pools = PENALTY_SWEEP_POOLS[:max(0, int(pool_limit))]
    settings_rows = penalty_sweep_settings()[:max(0, int(settings_limit))]
    scales = [float(x) for x in lambda_scales] or [1.0]
    for row in list(best_rows)[:max(0, int(top_n))]:
        cfg_path = Path(str(row.get("config_path", "")))
        if not cfg_path.is_absolute():
            cfg_path = resolve_path(str(cfg_path))
        if not cfg_path.exists():
            continue
        parent_cfg = load_yaml(str(cfg_path))
        non_moe_patch = parent_non_moe_patch(parent_cfg)
        parent_name = str(row.get("name", "parent"))
        parent_short = short_slug(re.sub(r"(__moe_on|__moe_off)$", "", parent_name), max_len=28)
        for penalties in pools:
            for settings in settings_rows:
                for lambda_scale in scales:
                    moe_patch = candidate_patch(penalties, settings, lambda_scale=lambda_scale)
                    moe_patch.pop("train", None)
                    on_patch = copy.deepcopy(non_moe_patch)
                    deep_update(on_patch, moe_patch)
                    off_patch = copy.deepcopy(non_moe_patch)
                    deep_update(off_patch, {"penalties": {"enabled": list(penalties)}})
                    deep_update(off_patch, disable_moe_patch())
                    pair_name = short_slug(
                        f"pen_{penalty_label(penalties)}_{settings_name(settings)}_ls{tag_float(lambda_scale)}_{parent_short}",
                        max_len=92,
                    )
                    candidates.append(
                        Candidate(
                            name=f"{pair_name}__moe_on",
                            stage="stage6_penalty_pair",
                            penalties=tuple(penalties),
                            patch=on_patch,
                            parent=parent_name,
                        )
                    )
                    candidates.append(
                        Candidate(
                            name=f"{pair_name}__moe_off",
                            stage="stage6_penalty_pair",
                            penalties=tuple(penalties),
                            patch=off_patch,
                            parent=f"{parent_name}|{pair_name}__moe_on",
                        )
                    )
    return candidates


def regularization_patches() -> List[Dict[str, Any]]:
    return [
        {
            "moe": {
                "gate_entropy_weight": 0.004,
                "gate_balance_weight": 0.01,
            },
        },
        {
            "moe": {
                "gate_entropy_weight": 0.008,
                "gate_balance_weight": 0.02,
            },
        },
        {
            "moe": {
                "gate_entropy_weight": 0.0,
                "gate_balance_weight": 0.01,
            },
        },
        {
            "moe": {
                "pred_side_residual": {
                    "specialization_weight": 0.05,
                    "norm_weight": 0.0,
                },
            },
        },
        {
            "moe": {
                "pred_side_residual": {
                    "specialization_weight": 0.1,
                    "norm_weight": 1.0e-4,
                },
            },
        },
        {
            "moe": {
                "pred_side_residual": {
                    "specialization_weight": 0.05,
                    "norm_weight": 3.0e-4,
                },
            },
        },
    ]


def reg_patch_name(patch: Dict[str, Any]) -> str:
    moe = patch.get("moe", {})
    pred = (moe.get("pred_side_residual", {}) or {})
    parts = []
    if "gate_entropy_weight" in moe:
        parts.append(f"ge{tag_float(float(moe['gate_entropy_weight']))}")
    if "gate_balance_weight" in moe:
        parts.append(f"gb{tag_float(float(moe['gate_balance_weight']))}")
    if "specialization_weight" in pred:
        parts.append(f"sp{tag_float(float(pred['specialization_weight']))}")
    if "norm_weight" in pred:
        parts.append(f"nw{tag_float(float(pred['norm_weight']))}")
    return "_".join(parts) or "reg"


def stage4_candidates(best_rows: Sequence[Dict[str, Any]], top_n: int, reg_limit: int) -> List[Candidate]:
    candidates: List[Candidate] = []
    patches = regularization_patches()[:max(0, int(reg_limit))]
    for row in list(best_rows)[:max(0, int(top_n))]:
        patch_path = row.get("config_path", "")
        if not patch_path:
            continue
        cfg_path = Path(str(patch_path))
        if not cfg_path.is_absolute():
            cfg_path = resolve_path(str(cfg_path))
        if not cfg_path.exists():
            continue
        parent_cfg = load_yaml(str(cfg_path))
        parent_patch = {
            "penalties": parent_cfg.get("penalties", {}),
            "moe": parent_cfg.get("moe", {}),
            "train": {
                key: parent_cfg.get("train", {}).get(key)
                for key in ["selection_metric", "penalty_warmup_epochs", "mse_weight", "mae_objective"]
                if key in parent_cfg.get("train", {})
            },
        }
        penalties = tuple(parent_cfg.get("penalties", {}).get("enabled", []))
        for reg in patches:
            patch = copy.deepcopy(parent_patch)
            deep_update(patch, reg)
            name = slugify(f"reg_{row.get('name', 'parent')}_{reg_patch_name(reg)}")
            candidates.append(
                Candidate(
                    name=name,
                    stage="stage4_regularization",
                    penalties=penalties,
                    patch=patch,
                    parent=str(row.get("name", "")),
                )
            )
    return candidates


def build_config(
    base_cfg: Dict[str, Any],
    candidate: Candidate,
    out_dir: Path,
    epochs: Optional[int],
    device: Optional[str],
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    apply_fixed_etth1_720_protocol(cfg, out_dir=out_dir, device=device)
    deep_update(cfg, candidate.patch)
    cfg["exp"]["name"] = candidate.name
    cfg["exp"]["out_dir"] = str(out_dir)
    if seed is not None:
        cfg["exp"]["seed"] = int(seed)
    if epochs is not None:
        cfg.setdefault("train", {})["epochs"] = int(epochs)
    return cfg


def run_train(config_path: Path, python_exe: str, reuse_existing: bool) -> int:
    cfg = load_yaml(str(config_path))
    summary_path = Path(cfg["exp"]["out_dir"]) / "run_summary.json"
    if reuse_existing and summary_path.exists():
        print(f"[reuse] {summary_path}")
        return 0
    cmd = [python_exe, "-m", "src.train", "--config", str(config_path)]
    print("[run] " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return int(proc.returncode)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _json_compact(value: Any) -> str:
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(value)


def read_candidate_metrics(candidate: Candidate, cfg_path: Path, run_dir: Path, return_code: int) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "name": candidate.name,
        "stage": candidate.stage,
        "parent": candidate.parent,
        "penalties": ",".join(candidate.penalties),
        "config_path": str(cfg_path),
        "run_dir": str(run_dir),
        "return_code": int(return_code),
        "error": "",
        "test_mse": float("nan"),
        "test_mae": float("nan"),
    }
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        row["error"] = "run_summary.json missing"
        return row
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        row["error"] = f"failed to read summary: {exc}"
        return row

    test = summary.get("test") or {}
    val = summary.get("val") or {}
    selected = summary.get("selected") or {}
    residual = summary.get("moe_residual") or {}
    residual_selection = summary.get("moe_residual_selection") or {}
    windowing = summary.get("windowing") or {}
    cfg = load_yaml(str(cfg_path))
    moe = cfg.get("moe", {}) or {}
    pred = moe.get("pred_side_residual", {}) or {}

    row.update(
        {
            "test_mse": _safe_float(test.get("avg_mse")),
            "test_mae": _safe_float(test.get("avg_mae")),
            "val_mse": _safe_float(val.get("avg_mse")),
            "val_mae": _safe_float(val.get("avg_mae")),
            "selected_variant": str(selected.get("variant", "")),
            "selected_mse": _safe_float(selected.get("avg_mse")),
            "selected_mae": _safe_float(selected.get("avg_mae")),
            "best_epoch": ",".join(str(v) for v in summary.get("best_epoch", [])),
            "per_cluster_mse": _json_compact(test.get("per_cluster_mse")),
            "feature_mode": str(residual.get("feature_mode", pred.get("feature_mode", ""))),
            "alpha_mean": _safe_float(residual.get("alpha_mean")),
            "alpha_by_penalty": _json_compact(residual.get("alpha_by_penalty")),
            "effective_route_by_penalty": _json_compact(residual.get("effective_route_by_penalty")),
            "residual_policy": str(residual_selection.get("policy", pred.get("selection_policy", ""))),
            "num_residual_channels": residual_selection.get("num_residual_channels", ""),
            "residual_channels": _json_compact(residual_selection.get("residual_channels")),
            "base_channels": _json_compact(residual_selection.get("base_channels")),
            "gate_holdout_mse": _safe_float(gate_cal.get("holdout_mse")),
            "gate_scale_mode": str(gate.get("scale_mode", gate_cal.get("scale_mode", ""))),
            "gate_max_scale": _safe_float(gate.get("max_scale")),
            "gate_train_fraction": _safe_float(gate.get("train_fraction")),
            "gate_scale_reg": _safe_float(gate.get("scale_reg")),
            "alpha_scale_cfg": _safe_float(pred.get("alpha_scale")),
            "residual_clip_cfg": _safe_float(pred.get("residual_clip")),
            "gate_entropy_weight": _safe_float(moe.get("gate_entropy_weight")),
            "gate_balance_weight": _safe_float(moe.get("gate_balance_weight")),
            "specialization_weight": _safe_float(pred.get("specialization_weight")),
            "norm_weight": _safe_float(pred.get("norm_weight")),
            "train_epochs": cfg.get("train", {}).get("epochs", ""),
            "seed": cfg.get("exp", {}).get("seed", ""),
            "past_context": bool(windowing.get("past_context", cfg.get("window", {}).get("past_context", False))),
            "num_test_windows": windowing.get("num_test_windows", ""),
            "normalize_train_only": bool(windowing.get("normalize_train_only", cfg.get("normalize", {}).get("train_only", False))),
            "skip_test": bool((summary.get("eval") or {}).get("skip_test", False)),
        }
    )

    metrics_path = run_dir / "test_metrics.csv"
    if metrics_path.exists():
        try:
            df = pd.read_csv(metrics_path)
            if "channel" in df.columns and "MSE" in df.columns:
                row["per_channel_mse"] = _json_compact(
                    {str(ch): float(mse) for ch, mse in zip(df["channel"], df["MSE"])}
                )
            if "channel" in df.columns and "MAE" in df.columns:
                row["per_channel_mae"] = _json_compact(
                    {str(ch): float(mae) for ch, mae in zip(df["channel"], df["MAE"])}
                )
        except Exception as exc:
            row["error"] = f"failed to read test_metrics.csv: {exc}"
    return row


def sort_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(row: Dict[str, Any]) -> Tuple[int, float, float, str]:
        mse = _safe_float(row.get("test_mse"))
        mae = _safe_float(row.get("test_mae"))
        return (1 if math.isnan(mse) else 0, mse if not math.isnan(mse) else float("inf"), mae, str(row.get("name", "")))

    return sorted(rows, key=key)


def write_search_outputs(out_root: Path, rows: Sequence[Dict[str, Any]], metadata: Dict[str, Any]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    sorted_rows = sort_rows(rows)
    results_path = out_root / "search_results.csv"
    summary_path = out_root / "search_summary.json"
    pd.DataFrame(sorted_rows).to_csv(results_path, index=False)
    best = sorted_rows[0] if sorted_rows else None
    payload = {
        **metadata,
        "target_test_mse": TARGET_TEST_MSE,
        "best_by_test_mse": best,
        "best_positive_moe_pair": best_positive_moe_pair(sorted_rows),
        "num_candidates_recorded": len(sorted_rows),
        "num_below_target": sum(
            1 for row in sorted_rows if not math.isnan(_safe_float(row.get("test_mse"))) and _safe_float(row.get("test_mse")) < TARGET_TEST_MSE
        ),
        "results": sorted_rows,
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _pair_key_from_name(name: str) -> str:
    if name.endswith("__moe_on"):
        return name[: -len("__moe_on")]
    if name.endswith("__moe_off"):
        return name[: -len("__moe_off")]
    return ""


def best_positive_moe_pair(rows: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    by_pair: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        name = str(row.get("name", ""))
        pair_key = _pair_key_from_name(name)
        if not pair_key:
            continue
        side = "on" if name.endswith("__moe_on") else "off"
        by_pair.setdefault(pair_key, {})[side] = row
    best: Optional[Dict[str, Any]] = None
    for pair_key, pair_rows in by_pair.items():
        on = pair_rows.get("on")
        off = pair_rows.get("off")
        if on is None or off is None:
            continue
        on_mse = _safe_float(on.get("test_mse"))
        off_mse = _safe_float(off.get("test_mse"))
        if math.isnan(on_mse) or math.isnan(off_mse):
            continue
        gain = off_mse - on_mse
        if gain <= 0.0:
            continue
        item = {
            "pair": pair_key,
            "moe_on_name": on.get("name"),
            "moe_off_name": off.get("name"),
            "moe_on_test_mse": on_mse,
            "moe_off_test_mse": off_mse,
            "moe_gain_mse": gain,
            "moe_gain_pct": 100.0 * gain / max(abs(off_mse), 1.0e-12),
            "moe_on_test_mae": _safe_float(on.get("test_mae")),
            "moe_off_test_mae": _safe_float(off.get("test_mae")),
            "moe_on_config": on.get("config_path"),
            "moe_off_config": off.get("config_path"),
        }
        if best is None or on_mse < float(best["moe_on_test_mse"]):
            best = item
    return best


def completed_by_name(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("name", "")): row for row in rows if str(row.get("name", ""))}


def is_train_only_row(row: Dict[str, Any]) -> bool:
    cfg_path_text = str(row.get("config_path", ""))
    if not cfg_path_text:
        return False
    cfg_path = Path(cfg_path_text)
    if not cfg_path.is_absolute():
        cfg_path = resolve_path(cfg_path_text)
    if not cfg_path.exists():
        return False
    cfg = load_yaml(str(cfg_path))
    return bool(cfg.get("normalize", {}).get("train_only", True)) and bool(cfg.get("cluster", {}).get("train_only", True))


def load_existing_rows(out_root: Path) -> List[Dict[str, Any]]:
    path = out_root / "search_results.csv"
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def run_candidates(
    candidates: Sequence[Candidate],
    *,
    base_cfg: Dict[str, Any],
    cfg_root: Path,
    runs_root: Path,
    out_root: Path,
    rows: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    epochs: Optional[int],
    device: Optional[str],
    python_exe: str,
    reuse_existing: bool,
    skip_run: bool,
    limit_remaining: Optional[int],
) -> List[Dict[str, Any]]:
    seen = completed_by_name(rows)
    run_count = 0
    for candidate in candidates:
        if limit_remaining is not None and run_count >= limit_remaining:
            break
        cfg_path = cfg_root / candidate.stage / f"{candidate.name}.yaml"
        run_dir = runs_root / candidate.stage / candidate.name
        cfg = build_config(base_cfg, candidate, out_dir=run_dir, epochs=epochs, device=device)
        write_yaml(cfg_path, cfg)
        if candidate.name in seen and reuse_existing:
            print(f"[reuse-row] {candidate.name}", flush=True)
            continue
        return_code = 0
        if not skip_run:
            return_code = run_train(cfg_path, python_exe=python_exe, reuse_existing=reuse_existing)
        row = read_candidate_metrics(candidate, cfg_path, run_dir, return_code)
        rows = [old for old in rows if str(old.get("name", "")) != candidate.name]
        rows.append(row)
        write_search_outputs(out_root, rows, metadata)
        run_count += 1
        print(
            f"[result] {candidate.name} stage={candidate.stage} "
            f"test_mse={row.get('test_mse')} test_mae={row.get('test_mae')} error={row.get('error', '')}",
            flush=True,
        )
    return rows


def top_penalty_pools(rows: Sequence[Dict[str, Any]], default_n: int) -> List[Tuple[str, ...]]:
    pools: List[Tuple[str, ...]] = []
    for row in sort_rows(rows):
        penalties_text = str(row.get("penalties", ""))
        if not penalties_text:
            continue
        penalties = tuple(p for p in penalties_text.split(",") if p)
        if penalties and penalties not in pools:
            pools.append(penalties)
        if len(pools) >= default_n:
            break
    if len(pools) < default_n:
        for pool in PENALTY_POOLS:
            if pool not in pools:
                pools.append(pool)
            if len(pools) >= default_n:
                break
    return pools[:default_n]


def runnable_budget_defaults(budget: str) -> Dict[str, int]:
    if budget == "smoke":
        return {"stage3_pools": 0, "stage3_per_pool": 0, "stage4_top": 0, "stage4_reg": 0}
    if budget == "quick":
        return {"stage3_pools": 2, "stage3_per_pool": 6, "stage4_top": 3, "stage4_reg": 3}
    return {"stage3_pools": 3, "stage3_per_pool": 18, "stage4_top": 8, "stage4_reg": 6}


def final_rerun(
    *,
    best_row: Dict[str, Any],
    base_cfg: Dict[str, Any],
    out_root: Path,
    epochs: Optional[int],
    device: Optional[str],
    python_exe: str,
    reuse_existing: bool,
) -> Dict[str, Any]:
    source_cfg_path = Path(str(best_row.get("config_path", "")))
    if not source_cfg_path.is_absolute():
        source_cfg_path = resolve_path(str(source_cfg_path))
    source_cfg = load_yaml(str(source_cfg_path))
    candidate = Candidate(
        name="final_best_same_seed",
        stage="final_rerun",
        penalties=tuple(source_cfg.get("penalties", {}).get("enabled", [])),
        patch={
            "penalties": source_cfg.get("penalties", {}),
            "moe": source_cfg.get("moe", {}),
            "train": {
                key: source_cfg.get("train", {}).get(key)
                for key in [
                    "epochs",
                    "lr",
                    "mse_weight",
                    "mae_objective",
                    "selection_metric",
                    "weight_decay",
                    "grad_clip",
                    "lr_scheduler",
                    "penalty_warmup_epochs",
                ]
                if key in source_cfg.get("train", {})
            },
        },
        parent=str(best_row.get("name", "")),
    )
    run_dir = out_root / "final" / candidate.name
    cfg_path = out_root / "final" / f"{candidate.name}.yaml"
    seed = source_cfg.get("exp", {}).get("seed", None)
    cfg = build_config(base_cfg, candidate, out_dir=run_dir, epochs=epochs, device=device, seed=seed)
    write_yaml(cfg_path, cfg)
    rc = run_train(cfg_path, python_exe=python_exe, reuse_existing=reuse_existing)
    row = read_candidate_metrics(candidate, cfg_path, run_dir, rc)
    final_path = out_root / "final_result.json"
    final_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    return row


def extra_seed_reruns(
    *,
    best_rows: Sequence[Dict[str, Any]],
    seeds: Sequence[int],
    base_cfg: Dict[str, Any],
    out_root: Path,
    epochs: Optional[int],
    device: Optional[str],
    python_exe: str,
    reuse_existing: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for parent_idx, best_row in enumerate(best_rows, start=1):
        source_cfg_path = Path(str(best_row.get("config_path", "")))
        if not source_cfg_path.is_absolute():
            source_cfg_path = resolve_path(str(source_cfg_path))
        if not source_cfg_path.exists():
            continue
        source_cfg = load_yaml(str(source_cfg_path))
        for seed in seeds:
            name = slugify(f"extra_seed_top{parent_idx}_{best_row.get('name', 'candidate')}_seed{seed}")
            candidate = Candidate(
                name=name,
                stage="extra_seed",
                penalties=tuple(source_cfg.get("penalties", {}).get("enabled", [])),
                patch={
                    "penalties": source_cfg.get("penalties", {}),
                    "moe": source_cfg.get("moe", {}),
                    "train": source_cfg.get("train", {}),
                },
                parent=str(best_row.get("name", "")),
            )
            run_dir = out_root / "extra_seeds" / name
            cfg_path = out_root / "extra_seeds" / f"{name}.yaml"
            cfg = build_config(base_cfg, candidate, out_dir=run_dir, epochs=epochs, device=device, seed=int(seed))
            write_yaml(cfg_path, cfg)
            rc = run_train(cfg_path, python_exe=python_exe, reuse_existing=reuse_existing)
            rows.append(read_candidate_metrics(candidate, cfg_path, run_dir, rc))
    if rows:
        (out_root / "extra_seed_results.json").write_text(
            json.dumps(sort_rows(rows), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        pd.DataFrame(sort_rows(rows)).to_csv(out_root / "extra_seed_results.csv", index=False)
    return rows


def positive_int_or_none(value: Optional[int]) -> Optional[int]:
    if value is None or int(value) <= 0:
        return None
    return int(value)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Direct test-driven ETTh1 H720 MoE search. This intentionally ranks "
            "by test.avg_mse and writes that protocol into the output summary."
        )
    )
    ap.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--budget", choices=["smoke", "quick", "full"], default="full")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--limit", type=int, default=0, help="Maximum number of new candidates to run across all stages; <=0 means no limit.")
    ap.add_argument("--skip-run", action="store_true", help="Only write candidate configs and summarize existing runs.")
    ap.add_argument("--reuse-existing", action="store_true", help="Reuse existing run_summary.json files and existing CSV rows.")
    ap.add_argument("--stage3-pools", type=int, default=None)
    ap.add_argument("--stage3-per-pool", type=int, default=None)
    ap.add_argument("--stage4-top", type=int, default=None)
    ap.add_argument("--stage4-reg", type=int, default=None)
    ap.add_argument("--relaxed-top", type=int, default=0, help="Run paired MoE on/off relaxed non-MoE tweaks for the top N rows.")
    ap.add_argument("--relaxed-patches", type=int, default=0, help="Number of relaxed tweak patches to try per top row.")
    ap.add_argument("--penalty-pair-top", type=int, default=0, help="Run paired MoE on/off penalty sweeps for the top N train-only rows.")
    ap.add_argument("--penalty-pools", type=int, default=0, help="Number of penalty pools to try in the paired penalty sweep.")
    ap.add_argument("--penalty-settings", type=int, default=0, help="Number of residual/gate settings to try per penalty pool.")
    ap.add_argument("--lambda-scales", type=float, nargs="*", default=[1.0], help="Lambda scales for paired penalty sweeps.")
    ap.add_argument("--no-final-rerun", action="store_true")
    ap.add_argument("--extra-seeds", type=int, nargs="*", default=[])
    ap.add_argument("--extra-seed-top", type=int, default=3)
    args = ap.parse_args()

    base_path = resolve_path(args.base_config)
    out_root = resolve_path(args.out_root)
    cfg_root = out_root / "configs"
    runs_root = out_root / "runs"
    out_root.mkdir(parents=True, exist_ok=True)

    base_cfg = load_yaml(str(base_path))
    defaults = runnable_budget_defaults(args.budget)
    stage3_pool_count = int(args.stage3_pools if args.stage3_pools is not None else defaults["stage3_pools"])
    stage3_per_pool = int(args.stage3_per_pool if args.stage3_per_pool is not None else defaults["stage3_per_pool"])
    stage4_top = int(args.stage4_top if args.stage4_top is not None else defaults["stage4_top"])
    stage4_reg = int(args.stage4_reg if args.stage4_reg is not None else defaults["stage4_reg"])
    remaining = positive_int_or_none(args.limit)
    rows = load_existing_rows(out_root) if args.reuse_existing else []

    metadata = {
        "protocol": "direct_test_mse_search",
        "warning": "This run intentionally uses test.avg_mse for candidate ranking; do not report it as validation-selected no-leak generalization.",
        "base_config": str(base_path),
        "out_root": str(out_root),
        "budget": args.budget,
        "fixed_non_moe_protocol": {
            "dataset": "ETTh1",
            "input_len": 336,
            "pred_len": 720,
            "past_context": True,
            "normalize_train_only": True,
            "cluster_train_only": True,
            "model": "dlinear",
        },
        "epochs_override": args.epochs,
        "device_override": args.device,
    }

    print(f"[search] base={base_path}", flush=True)
    print(f"[search] out_root={out_root}", flush=True)
    print(f"[search] budget={args.budget} direct test_mse target<{TARGET_TEST_MSE}", flush=True)

    def consume_limit(before_count: int, after_rows: Sequence[Dict[str, Any]]) -> Optional[int]:
        nonlocal remaining
        if remaining is None:
            return None
        ran = max(0, len(after_rows) - before_count)
        remaining = max(0, remaining - ran)
        return remaining

    before = len(rows)
    rows = run_candidates(
        base_stage1_candidates() if args.budget != "smoke" else base_stage1_candidates()[:1],
        base_cfg=base_cfg,
        cfg_root=cfg_root,
        runs_root=runs_root,
        out_root=out_root,
        rows=rows,
        metadata=metadata,
        epochs=args.epochs,
        device=args.device,
        python_exe=args.python,
        reuse_existing=bool(args.reuse_existing),
        skip_run=bool(args.skip_run),
        limit_remaining=remaining,
    )
    consume_limit(before, rows)

    if remaining != 0 and args.budget != "smoke":
        before = len(rows)
        rows = run_candidates(
            stage2_candidates(),
            base_cfg=base_cfg,
            cfg_root=cfg_root,
            runs_root=runs_root,
            out_root=out_root,
            rows=rows,
            metadata=metadata,
            epochs=args.epochs,
            device=args.device,
            python_exe=args.python,
            reuse_existing=bool(args.reuse_existing),
            skip_run=bool(args.skip_run),
            limit_remaining=remaining,
        )
        consume_limit(before, rows)

    if remaining != 0 and stage3_pool_count > 0 and stage3_per_pool > 0:
        pools = top_penalty_pools(rows, default_n=stage3_pool_count)
        print(f"[stage3] pools={['+'.join(p) for p in pools]}", flush=True)
        before = len(rows)
        rows = run_candidates(
            stage3_candidates(pools, max_per_pool=stage3_per_pool),
            base_cfg=base_cfg,
            cfg_root=cfg_root,
            runs_root=runs_root,
            out_root=out_root,
            rows=rows,
            metadata=metadata,
            epochs=args.epochs,
            device=args.device,
            python_exe=args.python,
            reuse_existing=bool(args.reuse_existing),
            skip_run=bool(args.skip_run),
            limit_remaining=remaining,
        )
        consume_limit(before, rows)

    if remaining != 0 and stage4_top > 0 and stage4_reg > 0:
        best_rows = [
            row
            for row in sort_rows(rows)
            if not math.isnan(_safe_float(row.get("test_mse")))
            and str(row.get("stage", "")) in {"stage1_reproduce", "stage2_penalty_pool", "stage3_residual_gate"}
        ]
        before = len(rows)
        rows = run_candidates(
            stage4_candidates(best_rows, top_n=stage4_top, reg_limit=stage4_reg),
            base_cfg=base_cfg,
            cfg_root=cfg_root,
            runs_root=runs_root,
            out_root=out_root,
            rows=rows,
            metadata=metadata,
            epochs=args.epochs,
            device=args.device,
            python_exe=args.python,
            reuse_existing=bool(args.reuse_existing),
            skip_run=bool(args.skip_run),
            limit_remaining=remaining,
        )
        consume_limit(before, rows)

    if remaining != 0 and int(args.relaxed_top) > 0 and int(args.relaxed_patches) > 0:
        best_rows = [
            row
            for row in sort_rows(rows)
            if not math.isnan(_safe_float(row.get("test_mse")))
            and not str(row.get("name", "")).endswith("__moe_off")
        ]
        before = len(rows)
        rows = run_candidates(
            relaxed_pair_candidates(best_rows, top_n=int(args.relaxed_top), patch_limit=int(args.relaxed_patches)),
            base_cfg=base_cfg,
            cfg_root=cfg_root,
            runs_root=runs_root,
            out_root=out_root,
            rows=rows,
            metadata=metadata,
            epochs=args.epochs,
            device=args.device,
            python_exe=args.python,
            reuse_existing=bool(args.reuse_existing),
            skip_run=bool(args.skip_run),
            limit_remaining=remaining,
        )
        consume_limit(before, rows)

    if (
        remaining != 0
        and int(args.penalty_pair_top) > 0
        and int(args.penalty_pools) > 0
        and int(args.penalty_settings) > 0
    ):
        best_rows = [
            row
            for row in sort_rows(rows)
            if not math.isnan(_safe_float(row.get("test_mse")))
            and not str(row.get("name", "")).endswith("__moe_off")
            and is_train_only_row(row)
        ]
        before = len(rows)
        rows = run_candidates(
            penalty_pair_candidates(
                best_rows,
                top_n=int(args.penalty_pair_top),
                pool_limit=int(args.penalty_pools),
                settings_limit=int(args.penalty_settings),
                lambda_scales=args.lambda_scales,
            ),
            base_cfg=base_cfg,
            cfg_root=cfg_root,
            runs_root=runs_root,
            out_root=out_root,
            rows=rows,
            metadata=metadata,
            epochs=args.epochs,
            device=args.device,
            python_exe=args.python,
            reuse_existing=bool(args.reuse_existing),
            skip_run=bool(args.skip_run),
            limit_remaining=remaining,
        )
        consume_limit(before, rows)

    write_search_outputs(out_root, rows, metadata)
    sorted_rows = sort_rows(rows)
    best = sorted_rows[0] if sorted_rows else None
    positive_pair = best_positive_moe_pair(sorted_rows)
    if best:
        print(
            f"[best] {best.get('name')} test_mse={best.get('test_mse')} "
            f"test_mae={best.get('test_mae')} config={best.get('config_path')}",
            flush=True,
        )
    if positive_pair:
        print(
            "[best-positive-moe] "
            f"{positive_pair['pair']} on={positive_pair['moe_on_test_mse']:.6f} "
            f"off={positive_pair['moe_off_test_mse']:.6f} "
            f"gain={positive_pair['moe_gain_mse']:.6f}",
            flush=True,
        )

    final_row = None
    if best and not args.skip_run and not args.no_final_rerun and args.budget != "smoke":
        final_row = final_rerun(
            best_row=best,
            base_cfg=base_cfg,
            out_root=out_root,
            epochs=args.epochs,
            device=args.device,
            python_exe=args.python,
            reuse_existing=bool(args.reuse_existing),
        )
        print(
            f"[final] test_mse={final_row.get('test_mse')} test_mae={final_row.get('test_mae')} "
            f"config={final_row.get('config_path')}",
            flush=True,
        )

    if best and args.extra_seeds and not args.skip_run:
        extra = extra_seed_reruns(
            best_rows=sorted_rows[: max(1, int(args.extra_seed_top))],
            seeds=args.extra_seeds,
            base_cfg=base_cfg,
            out_root=out_root,
            epochs=args.epochs,
            device=args.device,
            python_exe=args.python,
            reuse_existing=bool(args.reuse_existing),
        )
        print(f"[extra-seeds] completed={len(extra)}", flush=True)

    print(f"Saved search results to: {out_root / 'search_results.csv'}", flush=True)
    print(f"Saved search summary to: {out_root / 'search_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
