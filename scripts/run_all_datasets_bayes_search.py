from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

try:
    import numpy as np
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel
except Exception:  # pragma: no cover - handled at runtime for server portability
    np = None
    GaussianProcessRegressor = None
    Matern = None
    WhiteKernel = None


ROOT = Path(__file__).resolve().parents[1]

DATASET_CONFIGS = {
    "ETTh1": "ETTh1",
    "ETTh2": "ETTh2",
    "ETTm1": "ETTm1",
    "ETTm2": "ETTm2",
    "electricity": "electricity",
    "weather": "weather",
    "traffic": "traffic",
}

PENALTY_POOLS: dict[str, list[str]] = {
    "lddf": ["level", "delta", "d2_match", "diff_amp"],
    "ldf": ["level", "delta", "diff_amp"],
    "ld": ["level", "delta"],
    "range_ldf": ["level", "range", "delta", "diff_amp"],
    "trend_dir": ["delta", "trend", "direction"],
    "amp_only": ["amp_under"],
    "amp_diff": ["amp_under", "diff_amp"],
    "amp_delta": ["amp_under", "delta"],
    "amp_delta_diff": ["amp_under", "delta", "diff_amp"],
    "amp_dir": ["amp_under", "delta", "diff_amp", "direction"],
    "amp_level_dir": ["amp_under", "level", "delta", "diff_amp", "direction"],
    "amp_range_dir": ["amp_under", "range", "delta", "diff_amp", "direction"],
    "corr_trend": ["corr", "delta", "trend", "direction"],
}

PREDICTORS = [
    "mlp",
    "channel_head_mlp",
    "context_channel_head_mlp",
]

LAMBDA_PROFILES = ["flat", "amp_heavy", "diff_heavy", "level_heavy", "trend_heavy"]
RESIDUAL_MODES = ["none", "gated"]
FEATURE_MODES = ["legacy", "safe_augmented"]

FIELDS = [
    "status",
    "phase",
    "dataset",
    "horizon",
    "trial",
    "candidate_id",
    "objective",
    "val_mse",
    "val_mae",
    "test_mse",
    "test_mae",
    "selected_variant",
    "selected_mse",
    "selected_mae",
    "best_epoch",
    "penalty_pool",
    "penalties",
    "predictor",
    "hidden_dim",
    "dropout",
    "lambda_scale",
    "lambda_profile",
    "lambda_values",
    "lr",
    "weight_decay",
    "warmup",
    "batch_size",
    "distance_threshold",
    "merge_small_clusters",
    "feature_aware_weight",
    "gate_temperature",
    "gate_noise_std",
    "skip_cost",
    "prior_topk",
    "prior_strength",
    "residual_mode",
    "residual_alpha_scale",
    "residual_feature_mode",
    "epochs",
    "early_patience",
    "config_path",
    "out_dir",
    "returncode",
    "total_sec",
    "avg_epoch_sec",
    "error",
]


@dataclass(frozen=True)
class Candidate:
    penalty_pool: str
    predictor: str
    hidden_dim: int
    dropout: float
    lambda_scale: float
    lambda_profile: str
    lr: float
    weight_decay: float
    warmup: int
    batch_size: int
    distance_threshold: float
    merge_small_clusters: bool
    feature_aware_weight: float
    gate_temperature: float
    gate_noise_std: float
    skip_cost: float
    prior_topk: int
    prior_strength: float
    residual_mode: str
    residual_alpha_scale: float
    residual_feature_mode: str

    def key(self) -> tuple[Any, ...]:
        return (
            self.penalty_pool,
            self.predictor,
            self.hidden_dim,
            round(self.dropout, 4),
            round(self.lambda_scale, 6),
            self.lambda_profile,
            round(self.lr, 8),
            round(self.weight_decay, 8),
            self.warmup,
            self.batch_size,
            round(self.distance_threshold, 4),
            self.merge_small_clusters,
            round(self.feature_aware_weight, 4),
            round(self.gate_temperature, 4),
            round(self.gate_noise_std, 4),
            round(self.skip_cost, 4),
            self.prior_topk,
            round(self.prior_strength, 4),
            self.residual_mode,
            round(self.residual_alpha_scale, 4),
            self.residual_feature_mode,
        )

    def short_id(self) -> str:
        parts = [
            self.penalty_pool,
            self.predictor.replace("_", ""),
            f"h{self.hidden_dim}",
            f"do{self.dropout:.3g}",
            f"l{self.lambda_scale:.3g}",
            self.lambda_profile,
            f"lr{self.lr:.2g}",
            f"wd{self.weight_decay:.1g}",
            f"wu{self.warmup}",
            f"bs{self.batch_size}",
            f"dt{self.distance_threshold:.2g}",
            f"fa{self.feature_aware_weight:.2g}",
            f"gt{self.gate_temperature:.2g}",
            f"gn{self.gate_noise_std:.2g}",
            f"sk{self.skip_cost:.2g}",
            f"pk{self.prior_topk}",
            f"ps{self.prior_strength:.2g}",
            self.residual_mode,
            f"ra{self.residual_alpha_scale:.2g}",
            self.residual_feature_mode,
        ]
        raw = "_".join(parts)
        return raw.replace(".", "p").replace("-", "m")


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


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def safe_float(value: Any, default: float = math.inf) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        if math.isnan(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def expand_datasets(raw: list[str]) -> list[str]:
    if any(v.lower() == "all" for v in raw):
        return list(DATASET_CONFIGS)
    out = []
    for item in raw:
        if item not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset '{item}'. Use one of {sorted(DATASET_CONFIGS)} or all.")
        out.append(item)
    return out


def base_config_path(dataset: str, horizon: int) -> Path:
    stem = DATASET_CONFIGS[dataset]
    specific = ROOT / "configs" / f"{stem}_H{int(horizon)}.yaml"
    if specific.exists():
        return specific
    return ROOT / "configs" / f"{stem}.yaml"


def dataset_batch_choices(dataset: str, large_gpu: bool) -> list[int]:
    if dataset == "traffic":
        return [4, 8, 16, 24] if large_gpu else [4, 8]
    if dataset == "electricity":
        return [32, 64, 96, 128] if large_gpu else [16, 32, 64]
    if dataset in {"weather", "ETTm1", "ETTm2"}:
        return [32, 64, 128, 192] if large_gpu else [32, 64, 128]
    return [16, 32, 64, 96] if large_gpu else [16, 32, 64]


def weighted_lambdas(pool: str, scale: float, profile: str) -> dict[str, float]:
    penalties = PENALTY_POOLS[pool]
    weights = {p: 1.0 for p in penalties}
    if profile == "amp_heavy":
        for key in ("amp_under", "range"):
            if key in weights:
                weights[key] = 2.0
        for key in ("level", "trend"):
            if key in weights:
                weights[key] = 0.6
    elif profile == "diff_heavy":
        for key in ("diff_amp", "d2_match", "delta"):
            if key in weights:
                weights[key] = 2.0
        if "amp_under" in weights:
            weights["amp_under"] = 0.7
    elif profile == "level_heavy":
        for key in ("level", "range"):
            if key in weights:
                weights[key] = 2.0
        for key in ("direction", "trend"):
            if key in weights:
                weights[key] = 0.7
    elif profile == "trend_heavy":
        for key in ("trend", "direction", "delta"):
            if key in weights:
                weights[key] = 2.0
        if "level" in weights:
            weights["level"] = 0.7
    return {p: float(scale * weights[p]) for p in penalties}


def sample_candidate(
    rng: random.Random,
    *,
    dataset: str,
    predictors: list[str],
    large_gpu: bool,
    residual_modes: list[str],
) -> Candidate:
    pool_names = list(PENALTY_POOLS)
    hidden_choices = [64, 96, 128, 160, 192, 256]
    if dataset == "traffic":
        hidden_choices = [96, 128, 160, 192, 256]
    if dataset in {"ETTh1", "ETTh2"}:
        hidden_choices = [64, 96, 128, 160, 192]
    residual_mode = rng.choice(residual_modes)
    return Candidate(
        penalty_pool=rng.choice(pool_names),
        predictor=rng.choice(predictors),
        hidden_dim=rng.choice(hidden_choices),
        dropout=rng.uniform(0.0, 0.3),
        lambda_scale=10 ** rng.uniform(math.log10(0.003), math.log10(0.08)),
        lambda_profile=rng.choice(LAMBDA_PROFILES),
        lr=10 ** rng.uniform(math.log10(3.0e-4), math.log10(2.0e-3)),
        weight_decay=10 ** rng.uniform(math.log10(1.0e-5), math.log10(1.0e-3)),
        warmup=rng.choice([0, 2, 3, 5, 8]),
        batch_size=rng.choice(dataset_batch_choices(dataset, large_gpu)),
        distance_threshold=rng.uniform(0.45, 0.85),
        merge_small_clusters=rng.choice([True, False]),
        feature_aware_weight=rng.choice([0.0, 0.05, 0.1, 0.2, 0.35]),
        gate_temperature=rng.uniform(0.8, 1.8),
        gate_noise_std=rng.uniform(0.0, 0.3),
        skip_cost=rng.uniform(0.05, 0.35),
        prior_topk=rng.choice([0, 1, 2]),
        prior_strength=rng.uniform(0.0, 2.5),
        residual_mode=residual_mode,
        residual_alpha_scale=rng.uniform(0.25, 1.5),
        residual_feature_mode=rng.choice(FEATURE_MODES),
    )


def candidate_to_features(candidates: list[Candidate]) -> Any:
    if np is None:
        raise RuntimeError("numpy is required for Bayesian candidate encoding.")
    cat_values = {
        "penalty_pool": list(PENALTY_POOLS),
        "predictor": PREDICTORS,
        "lambda_profile": LAMBDA_PROFILES,
        "warmup": [0, 2, 3, 5, 8],
        "prior_topk": [0, 1, 2],
        "residual_mode": RESIDUAL_MODES,
        "residual_feature_mode": FEATURE_MODES,
        "batch_size": sorted({4, 8, 16, 24, 32, 64, 96, 128, 192}),
    }

    rows = []
    for c in candidates:
        row: list[float] = [
            float(c.hidden_dim) / 256.0,
            float(c.dropout),
            math.log10(float(c.lambda_scale)),
            math.log10(float(c.lr)),
            math.log10(float(c.weight_decay)),
            float(c.distance_threshold),
            1.0 if c.merge_small_clusters else 0.0,
            float(c.feature_aware_weight),
            float(c.gate_temperature),
            float(c.gate_noise_std),
            float(c.skip_cost),
            float(c.prior_strength),
            float(c.residual_alpha_scale),
        ]
        for attr, values in cat_values.items():
            value = getattr(c, attr)
            row.extend(1.0 if value == v else 0.0 for v in values)
        rows.append(row)
    return np.asarray(rows, dtype=float)


def normal_pdf(x: Any) -> Any:
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def normal_cdf(x: Any) -> Any:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def choose_next_candidate(
    *,
    rng: random.Random,
    dataset: str,
    predictors: list[str],
    large_gpu: bool,
    residual_modes: list[str],
    observed: list[tuple[Candidate, float]],
    tried: set[tuple[Any, ...]],
    pool_size: int,
) -> Candidate:
    if GaussianProcessRegressor is None or np is None or len(observed) < 4:
        for _ in range(max(1000, pool_size)):
            cand = sample_candidate(
                rng,
                dataset=dataset,
                predictors=predictors,
                large_gpu=large_gpu,
                residual_modes=residual_modes,
            )
            if cand.key() not in tried:
                return cand
        return sample_candidate(
            rng,
            dataset=dataset,
            predictors=predictors,
            large_gpu=large_gpu,
            residual_modes=residual_modes,
        )

    unique_pool: list[Candidate] = []
    seen = set(tried)
    attempts = 0
    while len(unique_pool) < pool_size and attempts < pool_size * 20:
        attempts += 1
        cand = sample_candidate(
            rng,
            dataset=dataset,
            predictors=predictors,
            large_gpu=large_gpu,
            residual_modes=residual_modes,
        )
        if cand.key() in seen:
            continue
        seen.add(cand.key())
        unique_pool.append(cand)
    if not unique_pool:
        return sample_candidate(
            rng,
            dataset=dataset,
            predictors=predictors,
            large_gpu=large_gpu,
            residual_modes=residual_modes,
        )

    train_cands = [c for c, _ in observed]
    y = np.asarray([v for _, v in observed], dtype=float)
    x_train = candidate_to_features(train_cands)
    x_pool = candidate_to_features(unique_pool)
    try:
        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1.0e-5)
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=rng.randint(0, 2**31 - 1))
        gp.fit(x_train, y)
        mu, std = gp.predict(x_pool, return_std=True)
        best = float(np.min(y))
        std = np.maximum(std, 1.0e-9)
        improvement = best - mu
        z = improvement / std
        ei = improvement * normal_cdf(z) + std * normal_pdf(z)
        idx = int(np.argmax(ei))
        return unique_pool[idx]
    except Exception:
        return rng.choice(unique_pool)


def configure(
    base_cfg: dict[str, Any],
    *,
    dataset: str,
    horizon: int,
    input_len: int,
    cand: Candidate,
    out_dir: Path,
    device: str,
    epochs: int,
    early_patience: int,
    skip_test: bool,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("exp", {})
    cfg["exp"].update(
        {
            "name": f"{dataset}_H{horizon}_{cand.short_id()}",
            "out_dir": str(out_dir),
            "device": device,
            "deterministic": bool(cfg.get("exp", {}).get("deterministic", True)),
        }
    )
    cfg.setdefault("window", {})
    cfg["window"]["input_len"] = int(input_len)
    cfg["window"]["pred_len"] = int(horizon)
    cfg["window"]["past_context"] = bool(cfg["window"].get("past_context", True))

    cfg.setdefault("normalize", {})
    cfg["normalize"]["global_zscore"] = True
    cfg["normalize"]["train_only"] = True

    cfg.setdefault("corr", {})
    cfg["corr"]["compute"] = True
    cfg["corr"]["save_path"] = str(out_dir / "corr.npy")

    cl = cfg.setdefault("cluster", {})
    cl["method"] = str(cl.get("method", "leader"))
    cl["distance_threshold"] = float(cand.distance_threshold)
    cl["min_cluster_size"] = int(cl.get("min_cluster_size", 2))
    cl["merge_small_clusters"] = bool(cand.merge_small_clusters)
    cl["no_merge_if_channels_lt"] = int(cl.get("no_merge_if_channels_lt", 7))
    cl["train_only"] = True
    if cand.feature_aware_weight > 0.0:
        cl["feature_aware"] = {
            "enable": True,
            "weight": float(cand.feature_aware_weight),
            "acf_lags": [1, 24, 96],
        }
    else:
        cl["feature_aware"] = {"enable": False, "weight": 0.0, "acf_lags": [1, 24, 96]}

    model = cfg.setdefault("model", {})
    model["predictor"] = cand.predictor
    model["hidden_dim"] = int(cand.hidden_dim)
    model["dropout"] = float(cand.dropout)
    model.setdefault("context_channel_head_include_delta", True)
    model.setdefault("channel_head_residual", True)
    model.setdefault("context_channel_head_residual", True)

    penalties = PENALTY_POOLS[cand.penalty_pool]
    cfg.setdefault("penalties", {})
    cfg["penalties"]["enabled"] = list(penalties)
    cfg["penalties"]["jump_threshold"] = float(cfg["penalties"].get("jump_threshold", 0.6))

    lambda_values = weighted_lambdas(cand.penalty_pool, cand.lambda_scale, cand.lambda_profile)
    moe = cfg.setdefault("moe", {})
    moe["enable"] = True
    moe["topk"] = min(int(moe.get("topk", 1)), max(1, len(penalties)))
    moe["select_ranks"] = [1]
    moe["detach_penalty_grad"] = False
    moe["lambda_init"] = lambda_values
    moe["lambda_min"] = {p: 0.0 for p in penalties}
    moe["lambda_schedule"] = {p: "none" for p in penalties}
    moe["gate_entropy_weight"] = 0.0
    moe["gate_balance_weight"] = 0.0
    moe["gate_route_on_penalty_only"] = True
    moe["router_mode"] = "learned"
    moe["router_penalty_context_weight"] = 0.0
    moe["router_detach_penalty_context"] = True
    moe["allow_skip"] = True
    moe["skip_cost"] = float(cand.skip_cost)
    moe["skip_init_bias"] = float(moe.get("skip_init_bias", -2.0))
    moe["gate_temperature"] = float(cand.gate_temperature)
    moe["gate_noise_std"] = float(cand.gate_noise_std)
    moe["gate_logit_clip"] = float(moe.get("gate_logit_clip", 5.0))
    moe.setdefault("dynamic_lambda", {})["enable"] = False
    moe.setdefault("learnable_lambda", {})["enable"] = False
    moe["cluster_penalty_prior"] = {
        "enable": bool(cand.prior_topk > 0 and cand.prior_strength > 0.0),
        "topk": int(cand.prior_topk),
        "hard_topk": True,
        "logit_strength": float(cand.prior_strength),
        "temperature": 1.0,
        "smoothing": 0.02,
        "use_normalized_penalty": True,
        "use_as_balance_target": False,
    }

    res = moe.setdefault("pred_side_residual", {})
    if cand.residual_mode == "none":
        res["enable"] = False
    else:
        res.update(
            {
                "enable": True,
                "corrector_hidden": max(16, int(cand.hidden_dim // 4)),
                "alpha_scale": float(cand.residual_alpha_scale),
                "selection_policy": "val_mse_gate_guarded",
                "feature_mode": cand.residual_feature_mode,
                "residual_clip": 4.0,
                "specialization_weight": 0.05,
                "norm_weight": 1.0e-4,
                "intervention_weight": 1.0e-3,
                "penalty_selector_enable": True,
                "selector_temperature": 1.0,
                "selector_use_cluster_context": True,
                "fusion_gate_enable": True,
                "fusion_init": -0.5,
                "fusion_use_cluster_context": True,
            }
        )
        res["channel_expert_adapters"] = {
            "enable": True,
            "mode": "merged_singletons",
            "mode_type": "override",
        }
        moe["channel_penalty_prior"] = {
            "enable": bool(cand.prior_topk > 0),
            "topk": int(max(1, cand.prior_topk)),
            "hard_topk": True,
            "temperature": 1.0,
            "smoothing": 0.02,
            "use_normalized_penalty": True,
        }

    train = cfg.setdefault("train", {})
    train["epochs"] = int(epochs)
    train["batch_size"] = int(cand.batch_size)
    train["lr"] = float(cand.lr)
    train["weight_decay"] = float(cand.weight_decay)
    train["selection_metric"] = "val_mse"
    train["penalty_warmup_epochs"] = int(cand.warmup)
    train.setdefault("mse_weight", 0.9)
    sched = train.setdefault("lr_scheduler", {})
    sched["name"] = "plateau"
    sched["factor"] = float(sched.get("factor", 0.5))
    sched["patience"] = min(int(sched.get("patience", 5)), max(3, early_patience // 2))
    sched["min_lr"] = float(sched.get("min_lr", 1.0e-6))

    cfg.setdefault("early_stop", {})
    cfg["early_stop"]["patience"] = int(early_patience)
    cfg["early_stop"]["min_delta"] = float(cfg["early_stop"].get("min_delta", 1.0e-6))

    cfg["eval"] = {"skip_test": bool(skip_test)}
    cfg["plot"] = {"enable": False}
    cfg["portrait"] = {"enable": False, "out_dir": str(out_dir / "cluster_portraits")}
    cfg["knn_hybrid"] = copy.deepcopy(cfg.get("knn_hybrid", {}))
    cfg["knn_hybrid"]["enable"] = False
    cfg["knn_hybrid"]["use_for_model_selection"] = False
    cfg["knn_hybrid"]["path"] = str(out_dir / "knn_shape_bank.pt")
    cfg["calibration"] = {"enable": False}
    cfg["memory"] = {
        "enable": False,
        "save_checkpoint": False,
        "path": str(out_dir / "cluster_memory.pt"),
        "checkpoint_path": str(out_dir / "best_checkpoint.pt"),
    }
    return cfg


def run_train(py: str, cfg_path: Path, out_dir: Path) -> tuple[int, str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    proc = subprocess.run(
        [py, "-u", "-m", "src.train", "--config", str(cfg_path)],
        cwd=str(ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.perf_counter() - t0
    (out_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8", errors="replace")
    return int(proc.returncode), proc.stdout, elapsed


def row_from_run(
    *,
    phase: str,
    dataset: str,
    horizon: int,
    trial: int,
    cand: Candidate,
    cfg_path: Path,
    out_dir: Path,
    epochs: int,
    early_patience: int,
    returncode: int,
    output: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "status": "ok" if returncode == 0 else ("oom" if "out of memory" in output.lower() else "error"),
        "phase": phase,
        "dataset": dataset,
        "horizon": int(horizon),
        "trial": int(trial),
        "candidate_id": cand.short_id(),
        "penalty_pool": cand.penalty_pool,
        "penalties": "|".join(PENALTY_POOLS[cand.penalty_pool]),
        "predictor": cand.predictor,
        "hidden_dim": cand.hidden_dim,
        "dropout": cand.dropout,
        "lambda_scale": cand.lambda_scale,
        "lambda_profile": cand.lambda_profile,
        "lambda_values": json.dumps(weighted_lambdas(cand.penalty_pool, cand.lambda_scale, cand.lambda_profile), sort_keys=True),
        "lr": cand.lr,
        "weight_decay": cand.weight_decay,
        "warmup": cand.warmup,
        "batch_size": cand.batch_size,
        "distance_threshold": cand.distance_threshold,
        "merge_small_clusters": cand.merge_small_clusters,
        "feature_aware_weight": cand.feature_aware_weight,
        "gate_temperature": cand.gate_temperature,
        "gate_noise_std": cand.gate_noise_std,
        "skip_cost": cand.skip_cost,
        "prior_topk": cand.prior_topk,
        "prior_strength": cand.prior_strength,
        "residual_mode": cand.residual_mode,
        "residual_alpha_scale": cand.residual_alpha_scale,
        "residual_feature_mode": cand.residual_feature_mode,
        "epochs": epochs,
        "early_patience": early_patience,
        "config_path": str(cfg_path),
        "out_dir": str(out_dir),
        "returncode": returncode,
        "error": "" if returncode == 0 else output[-3000:],
    }
    summary_path = out_dir / "run_summary.json"
    if not summary_path.exists():
        row["status"] = "error" if returncode == 0 else row["status"]
        row["error"] = row.get("error") or f"Missing run_summary.json: {summary_path}"
        row["objective"] = math.inf
        return row
    summary = read_json(summary_path)
    val = summary.get("val") or {}
    test = summary.get("test") or {}
    selected = summary.get("selected") or {}
    row.update(
        {
            "val_mse": val.get("avg_mse", ""),
            "val_mae": val.get("avg_mae", ""),
            "test_mse": test.get("avg_mse", ""),
            "test_mae": test.get("avg_mae", ""),
            "selected_variant": selected.get("variant", ""),
            "selected_mse": selected.get("avg_mse", ""),
            "selected_mae": selected.get("avg_mae", ""),
            "best_epoch": json.dumps(summary.get("best_epoch", "")),
            "total_sec": summary.get("timing", {}).get("total_sec", ""),
            "avg_epoch_sec": summary.get("timing", {}).get("avg_epoch_sec", ""),
        }
    )
    row["objective"] = safe_float(row.get("val_mse"))
    return row


def run_candidate(
    *,
    dataset: str,
    horizon: int,
    trial: int,
    cand: Candidate,
    phase: str,
    out_root: Path,
    py: str,
    device: str,
    input_len: int,
    epochs: int,
    early_patience: int,
    skip_test: bool,
    rerun: bool,
) -> dict[str, Any]:
    out_dir = out_root / ("final_runs" if phase == "final" else "runs") / dataset / f"H{horizon}" / f"trial_{trial:04d}_{cand.short_id()}"
    cfg_path = out_root / ("final_configs" if phase == "final" else "configs") / dataset / f"H{horizon}" / f"trial_{trial:04d}_{cand.short_id()}.yaml"
    base_cfg = read_yaml(base_config_path(dataset, horizon))
    cfg = configure(
        base_cfg,
        dataset=dataset,
        horizon=horizon,
        input_len=input_len,
        cand=cand,
        out_dir=out_dir,
        device=device,
        epochs=epochs,
        early_patience=early_patience,
        skip_test=skip_test,
    )
    write_yaml(cfg_path, cfg)
    summary_path = out_dir / "run_summary.json"
    if summary_path.exists() and not rerun:
        return row_from_run(
            phase=phase,
            dataset=dataset,
            horizon=horizon,
            trial=trial,
            cand=cand,
            cfg_path=cfg_path,
            out_dir=out_dir,
            epochs=epochs,
            early_patience=early_patience,
            returncode=0,
            output="reused",
        )
    returncode, output, _ = run_train(py, cfg_path, out_dir)
    return row_from_run(
        phase=phase,
        dataset=dataset,
        horizon=horizon,
        trial=trial,
        cand=cand,
        cfg_path=cfg_path,
        out_dir=out_dir,
        epochs=epochs,
        early_patience=early_patience,
        returncode=returncode,
        output=output,
    )


def observed_for_task(rows: list[dict[str, Any]], dataset: str, horizon: int) -> list[tuple[Candidate, float]]:
    out: list[tuple[Candidate, float]] = []
    for row in rows:
        if row.get("phase") != "search" or row.get("status") != "ok":
            continue
        if row.get("dataset") != dataset or int(row.get("horizon", -1)) != int(horizon):
            continue
        try:
            cand = Candidate(
                penalty_pool=str(row["penalty_pool"]),
                predictor=str(row["predictor"]),
                hidden_dim=int(float(row["hidden_dim"])),
                dropout=float(row["dropout"]),
                lambda_scale=float(row["lambda_scale"]),
                lambda_profile=str(row["lambda_profile"]),
                lr=float(row["lr"]),
                weight_decay=float(row["weight_decay"]),
                warmup=int(float(row["warmup"])),
                batch_size=int(float(row["batch_size"])),
                distance_threshold=float(row["distance_threshold"]),
                merge_small_clusters=str(row["merge_small_clusters"]).lower() in {"true", "1", "yes"},
                feature_aware_weight=float(row["feature_aware_weight"]),
                gate_temperature=float(row["gate_temperature"]),
                gate_noise_std=float(row["gate_noise_std"]),
                skip_cost=float(row["skip_cost"]),
                prior_topk=int(float(row["prior_topk"])),
                prior_strength=float(row["prior_strength"]),
                residual_mode=str(row["residual_mode"]),
                residual_alpha_scale=float(row["residual_alpha_scale"]),
                residual_feature_mode=str(row["residual_feature_mode"]),
            )
            value = safe_float(row.get("objective", row.get("val_mse")))
        except Exception:
            continue
        if math.isfinite(value):
            out.append((cand, value))
    return out


def write_best_by_task(path: Path, rows: list[dict[str, Any]]) -> None:
    best: list[dict[str, Any]] = []
    tasks = sorted({(r.get("dataset"), int(r.get("horizon", -1))) for r in rows if r.get("status") == "ok" and r.get("phase") == "search"})
    for dataset, horizon in tasks:
        subset = [
            r for r in rows
            if r.get("phase") == "search"
            and r.get("status") == "ok"
            and r.get("dataset") == dataset
            and int(r.get("horizon", -1)) == horizon
        ]
        if not subset:
            continue
        best.append(min(subset, key=lambda r: safe_float(r.get("objective", r.get("val_mse")))))
    write_rows(path, best)


def _search_points(rows: list[dict[str, Any]], dataset: str, horizon: int) -> list[tuple[int, float, str]]:
    points: list[tuple[int, float, str]] = []
    for row in rows:
        if row.get("phase") != "search" or row.get("status") != "ok":
            continue
        if row.get("dataset") != dataset or int(row.get("horizon", -1)) != int(horizon):
            continue
        value = safe_float(row.get("val_mse", row.get("objective")))
        if not math.isfinite(value):
            continue
        try:
            trial = int(row.get("trial", 0))
        except (TypeError, ValueError):
            trial = len(points)
        points.append((trial, value, str(row.get("candidate_id", ""))))
    points.sort(key=lambda item: item[0])
    return points


def _best_so_far(values: list[float]) -> list[float]:
    best = math.inf
    out: list[float] = []
    for value in values:
        best = min(best, value)
        out.append(best)
    return out


def plot_search_progress(out_root: Path, rows: list[dict[str, Any]]) -> None:
    """Refresh validation-search plots after each completed trial.

    Plotting is intentionally best-effort. A plotting failure must not stop a
    long search job on the server.
    """
    plot_dir = out_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on server env
        (plot_dir / "plot_error.txt").write_text(
            f"matplotlib import failed; search continues without plots.\n{exc}\n",
            encoding="utf-8",
        )
        return

    tasks = sorted(
        {
            (str(r.get("dataset")), int(r.get("horizon", -1)))
            for r in rows
            if r.get("phase") == "search" and r.get("status") == "ok"
        }
    )
    summary_items: list[tuple[str, list[int], list[float]]] = []
    for dataset, horizon in tasks:
        points = _search_points(rows, dataset, horizon)
        if not points:
            continue
        trials = [p[0] for p in points]
        vals = [p[1] for p in points]
        best_vals = _best_so_far(vals)
        summary_items.append((f"{dataset} H{horizon}", trials, best_vals))

        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
        ax.scatter(trials, vals, s=24, alpha=0.45, color="#8aa0b8", label="trial val MSE")
        ax.plot(trials, best_vals, color="#c0392b", linewidth=2.0, marker="o", markersize=3.5, label="best-so-far")
        best_idx = min(range(len(vals)), key=lambda i: vals[i])
        ax.scatter([trials[best_idx]], [vals[best_idx]], s=60, color="#1f7a3a", zorder=5, label="current best")
        ax.set_title(f"{dataset} H={horizon} validation search")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Validation MSE")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{dataset}_H{horizon}_val_progress.png")
        plt.close(fig)

    if summary_items:
        fig, ax = plt.subplots(figsize=(8.8, 5.0), dpi=150)
        for label, trials, best_vals in summary_items:
            ax.plot(trials, best_vals, linewidth=1.8, marker="o", markersize=3.0, label=label)
        ax.set_title("Validation best-so-far by task")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Best validation MSE so far")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend(frameon=False, fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(plot_dir / "all_tasks_val_best_so_far.png")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Bayesian-style val search for all datasets and horizons.")
    ap.add_argument("--datasets", nargs="+", default=["all"], help="Datasets or 'all'.")
    ap.add_argument("--horizons", nargs="+", type=int, default=[96], help="Prediction horizons to tune.")
    ap.add_argument("--input-len", type=int, default=336)
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "all_datasets_bayes_search")
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--trials", type=int, default=24, help="Trials per dataset/horizon.")
    ap.add_argument("--init-trials", type=int, default=8, help="Initial random trials before GP EI selection.")
    ap.add_argument("--candidate-pool-size", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=50, help="Default epoch budget. Used by search/final unless overridden.")
    ap.add_argument("--early-patience", type=int, default=10, help="Default early-stop patience. Used by search/final unless overridden.")
    ap.add_argument("--search-epochs", type=int, default=None, help="Epoch budget for search trials; defaults to --epochs.")
    ap.add_argument("--search-early-patience", type=int, default=None, help="Early-stop patience for search trials; defaults to --early-patience.")
    ap.add_argument("--final-epochs", type=int, default=None, help="Epoch budget for final test reruns; defaults to --epochs.")
    ap.add_argument("--final-early-patience", type=int, default=None, help="Early-stop patience for final test reruns; defaults to --early-patience.")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--predictors", nargs="+", default=["mlp"], choices=PREDICTORS)
    ap.add_argument("--residual-modes", nargs="+", default=["none", "gated"], choices=RESIDUAL_MODES)
    ap.add_argument("--large-gpu", action="store_true", help="Search larger batch sizes; intended for ~80GB GPUs.")
    ap.add_argument("--final-test-top-k", type=int, default=1, help="Rerun top-k val configs per task with eval.skip_test=false.")
    ap.add_argument("--plot-every", type=int, default=1, help="Refresh plots every N completed search trials. Use 0 to disable.")
    ap.add_argument("--no-plots", action="store_true", help="Disable incremental validation progress plots.")
    ap.add_argument("--search-only", action="store_true", help="Run validation search only; skip final test reruns.")
    ap.add_argument("--final-only", action="store_true", help="Skip search and rerun final tests from existing search results.")
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if GaussianProcessRegressor is None or np is None:
        print("WARNING: sklearn/numpy not available; script will fall back to random search.", flush=True)
    datasets = expand_datasets(args.datasets)
    horizons = [int(h) for h in args.horizons]
    search_epochs = int(args.search_epochs if args.search_epochs is not None else args.epochs)
    search_early_patience = int(
        args.search_early_patience if args.search_early_patience is not None else args.early_patience
    )
    final_epochs = int(args.final_epochs if args.final_epochs is not None else args.epochs)
    final_early_patience = int(
        args.final_early_patience if args.final_early_patience is not None else args.early_patience
    )
    plot_every = 0 if bool(args.no_plots) else max(0, int(args.plot_every))
    rng = random.Random(int(args.seed))
    args.out_root.mkdir(parents=True, exist_ok=True)
    result_path = args.out_root / "results.csv"
    final_path = args.out_root / "final_results.csv"
    best_path = args.out_root / "best_by_task.csv"
    rows = read_rows(result_path)

    if not args.final_only:
        for dataset in datasets:
            for horizon in horizons:
                task_rows = [
                    r for r in rows
                    if r.get("phase") == "search"
                    and r.get("dataset") == dataset
                    and int(r.get("horizon", -1)) == int(horizon)
                ]
                tried = {
                    c.key()
                    for c, _ in observed_for_task(rows, dataset, horizon)
                }
                print(f"\n=== Search {dataset} H{horizon}: {len(task_rows)}/{args.trials} trials ===", flush=True)
                while len(task_rows) < int(args.trials):
                    trial = len(task_rows)
                    observed = observed_for_task(rows, dataset, horizon)
                    if len(observed) < int(args.init_trials):
                        cand = sample_candidate(
                            rng,
                            dataset=dataset,
                            predictors=list(args.predictors),
                            large_gpu=bool(args.large_gpu),
                            residual_modes=list(args.residual_modes),
                        )
                        for _ in range(1000):
                            if cand.key() not in tried:
                                break
                            cand = sample_candidate(
                                rng,
                                dataset=dataset,
                                predictors=list(args.predictors),
                                large_gpu=bool(args.large_gpu),
                                residual_modes=list(args.residual_modes),
                            )
                    else:
                        cand = choose_next_candidate(
                            rng=rng,
                            dataset=dataset,
                            predictors=list(args.predictors),
                            large_gpu=bool(args.large_gpu),
                            residual_modes=list(args.residual_modes),
                            observed=observed,
                            tried=tried,
                            pool_size=int(args.candidate_pool_size),
                        )
                    tried.add(cand.key())
                    print(f"[trial {trial:04d}] {dataset} H{horizon} {cand.short_id()}", flush=True)
                    if args.dry_run:
                        row = {
                            "status": "planned",
                            "phase": "search",
                            "dataset": dataset,
                            "horizon": horizon,
                            "trial": trial,
                            "candidate_id": cand.short_id(),
                        }
                    else:
                        row = run_candidate(
                            dataset=dataset,
                            horizon=horizon,
                            trial=trial,
                            cand=cand,
                            phase="search",
                            out_root=args.out_root,
                            py=str(args.python),
                            device=str(args.device),
                            input_len=int(args.input_len),
                            epochs=search_epochs,
                            early_patience=search_early_patience,
                            skip_test=True,
                            rerun=bool(args.rerun),
                        )
                    rows = [
                        r for r in rows
                        if not (
                            r.get("phase") == "search"
                            and r.get("dataset") == dataset
                            and int(r.get("horizon", -1)) == int(horizon)
                            and int(r.get("trial", -1)) == int(trial)
                        )
                    ]
                    rows.append(row)
                    write_rows(result_path, rows)
                    write_best_by_task(best_path, rows)
                    completed_for_task = len(
                        [
                            r for r in rows
                            if r.get("phase") == "search"
                            and r.get("dataset") == dataset
                            and int(r.get("horizon", -1)) == int(horizon)
                        ]
                    )
                    if plot_every > 0 and (completed_for_task % plot_every == 0 or completed_for_task == int(args.trials)):
                        plot_search_progress(args.out_root, rows)
                    print(f"  -> {row.get('status')} val={row.get('val_mse', '')} obj={row.get('objective', '')}", flush=True)
                    task_rows = [
                        r for r in rows
                        if r.get("phase") == "search"
                        and r.get("dataset") == dataset
                        and int(r.get("horizon", -1)) == int(horizon)
                    ]

    if int(args.final_test_top_k) > 0 and not args.dry_run and not args.search_only:
        final_rows = read_rows(final_path)
        for dataset in datasets:
            for horizon in horizons:
                subset = [
                    r for r in rows
                    if r.get("phase") == "search"
                    and r.get("status") == "ok"
                    and r.get("dataset") == dataset
                    and int(r.get("horizon", -1)) == int(horizon)
                ]
                subset.sort(key=lambda r: safe_float(r.get("objective", r.get("val_mse"))))
                for rank, src in enumerate(subset[: int(args.final_test_top_k)]):
                    cand = Candidate(
                        penalty_pool=str(src["penalty_pool"]),
                        predictor=str(src["predictor"]),
                        hidden_dim=int(float(src["hidden_dim"])),
                        dropout=float(src["dropout"]),
                        lambda_scale=float(src["lambda_scale"]),
                        lambda_profile=str(src["lambda_profile"]),
                        lr=float(src["lr"]),
                        weight_decay=float(src["weight_decay"]),
                        warmup=int(float(src["warmup"])),
                        batch_size=int(float(src["batch_size"])),
                        distance_threshold=float(src["distance_threshold"]),
                        merge_small_clusters=str(src["merge_small_clusters"]).lower() in {"true", "1", "yes"},
                        feature_aware_weight=float(src["feature_aware_weight"]),
                        gate_temperature=float(src["gate_temperature"]),
                        gate_noise_std=float(src["gate_noise_std"]),
                        skip_cost=float(src["skip_cost"]),
                        prior_topk=int(float(src["prior_topk"])),
                        prior_strength=float(src["prior_strength"]),
                        residual_mode=str(src["residual_mode"]),
                        residual_alpha_scale=float(src["residual_alpha_scale"]),
                        residual_feature_mode=str(src["residual_feature_mode"]),
                    )
                    trial = int(src.get("trial", rank))
                    print(f"\n[final {rank}] {dataset} H{horizon} trial={trial} {cand.short_id()}", flush=True)
                    row = run_candidate(
                        dataset=dataset,
                        horizon=horizon,
                        trial=trial,
                        cand=cand,
                        phase="final",
                        out_root=args.out_root,
                        py=str(args.python),
                        device=str(args.device),
                        input_len=int(args.input_len),
                        epochs=final_epochs,
                        early_patience=final_early_patience,
                        skip_test=False,
                        rerun=bool(args.rerun),
                    )
                    final_rows = [
                        r for r in final_rows
                        if not (
                            r.get("phase") == "final"
                            and r.get("dataset") == dataset
                            and int(r.get("horizon", -1)) == int(horizon)
                            and int(r.get("trial", -1)) == int(trial)
                        )
                    ]
                    final_rows.append(row)
                    write_rows(final_path, final_rows)
                    if plot_every > 0:
                        plot_search_progress(args.out_root, rows)
                    print(f"  -> {row.get('status')} val={row.get('val_mse', '')} test={row.get('test_mse', '')}", flush=True)

    print(f"\nSaved search results: {result_path}", flush=True)
    print(f"Saved best-by-task: {best_path}", flush=True)
    if int(args.final_test_top_k) > 0:
        print(f"Saved final test results: {final_path}", flush=True)


if __name__ == "__main__":
    main()
