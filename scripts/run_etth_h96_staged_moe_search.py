from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
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
    "gate_max_scale",
    "gate_init_scale",
    "gate_scale_reg",
    "gate_train_fraction",
    "penalty_guard_allow_multi",
    "base_test_mse",
    "zero_test_mse",
    "full_test_mse",
    "zero_gain_pct",
    "full_gain_pct",
    "full_vs_zero_mse",
    "base_test_mae",
    "zero_test_mae",
    "full_test_mae",
    "base_val_mse",
    "zero_val_mse",
    "full_val_mse",
    "base_checkpoint",
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
    gate_max_scale: float = 1.0
    gate_init_scale: float = 0.8
    gate_scale_reg: float = 1.0e-5
    gate_train_fraction: float = 0.85
    penalty_guard_allow_multi: bool = False


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
            str(cand.gate_max_scale),
            str(cand.gate_init_scale),
            str(cand.gate_scale_reg),
            str(cand.gate_train_fraction),
            str(cand.penalty_guard_allow_multi),
        ]
    )
    slug = cand.name.replace(".", "_").replace("|", "_").replace(",", "_")
    return f"{slug[:44]}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:10]}"


def fixed_protocol(cfg: dict[str, Any], out_dir: Path, device: str | None, epochs: int | None) -> None:
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
    cfg.setdefault("eval", {})["skip_test"] = False


def configure_base(cfg: dict[str, Any], checkpoint_path: Path) -> None:
    moe = cfg.setdefault("moe", {})
    moe["enable"] = False
    moe.setdefault("dynamic_lambda", {})["enable"] = False
    moe.setdefault("pred_side_residual", {})["enable"] = False
    moe.setdefault("learnable_lambda", {})["enable"] = False
    cfg["memory"] = {
        "enable": False,
        "save_checkpoint": True,
        "path": str(checkpoint_path.with_name("cluster_memory.pt")),
        "checkpoint_path": str(checkpoint_path),
    }
    cfg.pop("finetune", None)


def configure_staged_moe(
    cfg: dict[str, Any],
    cand: Candidate,
    *,
    lambda_value: float,
    checkpoint_path: Path,
) -> None:
    penalties = list(cand.penalties)
    cfg.setdefault("penalties", {})["enabled"] = penalties
    cfg["finetune"] = {
        "enable": True,
        "checkpoint_path": str(checkpoint_path),
        "cluster_map": "index",
        "strict_window": True,
        "strict_model": True,
        "load_model": True,
        "load_gate": False,
        "load_dynamic_lambda": False,
        "load_learnable_lambda": False,
        "freeze_model": True,
    }
    cfg["memory"] = {
        "enable": False,
        "save_checkpoint": False,
        "path": str(checkpoint_path.with_name("cluster_memory.pt")),
        "checkpoint_path": str(checkpoint_path),
    }

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
            "base_anchor_weight": 0.0,
            "use_y_base_input": True,
            "intervention_enable": False,
            "intervention_init": -2.0,
            "intervention_weight": 1.0e-3,
            "detach_routed_penalty_pred": False,
            "selection_policy": cand.selection_policy,
            "selection_min_abs_improvement": 0.0,
            "selection_min_rel_improvement": 0.0,
            "penalty_guard": {
                "enable": False,
                "metric": "mse",
                "allow_multi": bool(cand.penalty_guard_allow_multi),
                "min_abs_improvement": 0.0,
                "min_rel_improvement": 0.0,
            },
            "channel_guard": {"enable": False},
            "validation_guard": {
                "enable": False,
                "select_fraction": 0.5,
                "min_abs_improvement": 0.0,
                "min_rel_improvement": 0.002,
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
    moe["gate_noise_std"] = 0.0
    moe["gate_soft_weight"] = 0.0
    moe["gate_prob_floor"] = 0.0
    moe["gate_entropy_weight"] = 0.0
    moe["gate_balance_weight"] = 0.0
    moe["gate_init_bias"] = {"enable": True, "values": {"level": 1.0, "default": 0.0}}
    moe["residual_gate"] = {"enable": True, "alpha": 0.7}
    moe["pred_aware"] = {"enable": True, "use_pred_features": True, "use_penalty_input": False}
    moe["penalty_ema"] = {"enable": False, "decay": 0.9}


def run_train(config_path: Path, out_dir: Path, reuse_existing: bool) -> tuple[int, float, str]:
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


def candidate_grid(base_penalties: tuple[str, ...]) -> list[Candidate]:
    pools = [
        base_penalties,
        ("delta", "trend", "direction"),
        ("level", "delta", "d2_match", "diff_amp"),
    ]
    presets = [
        {"lambda_init": 0.0, "alpha_scale": 0.8, "selection_policy": "none", "gate_max_scale": 1.0, "gate_init_scale": 0.8, "penalty_guard_allow_multi": False},
        {"lambda_init": 0.0, "alpha_scale": 1.4, "selection_policy": "val_mse_gate", "gate_max_scale": 1.5, "gate_init_scale": 1.0, "penalty_guard_allow_multi": False},
        {"lambda_init": 0.005, "alpha_scale": 1.8, "selection_policy": "val_mse_gate", "gate_max_scale": 1.75, "gate_init_scale": 1.0, "penalty_guard_allow_multi": False},
        {"lambda_init": 0.01, "alpha_scale": 2.2, "selection_policy": "val_mse_gate", "gate_max_scale": 2.0, "gate_init_scale": 1.2, "penalty_guard_allow_multi": True},
        {"lambda_init": 0.02, "alpha_scale": 2.5, "selection_policy": "val_mse_scale", "gate_max_scale": 2.0, "gate_init_scale": 1.2, "penalty_guard_allow_multi": False},
    ]
    cands: list[Candidate] = []
    seen = set()
    for pool in pools:
        if tuple(pool) in seen:
            continue
        seen.add(tuple(pool))
        for preset in presets:
            name = f"p{'_'.join(pool)}_lam{preset['lambda_init']}_a{preset['alpha_scale']}"
            cands.append(Candidate(name=name, penalties=tuple(pool), **preset))
    return cands


def run_dataset(args: argparse.Namespace, dataset: str, base_cfg_path: Path, out_root: Path) -> None:
    base_cfg = read_yaml(base_cfg_path)
    base_penalties = tuple((base_cfg.get("penalties", {}) or {}).get("enabled", []))
    if len(base_penalties) == 0:
        base_penalties = tuple((base_cfg.get("moe", {}) or {}).get("lambda_init", {}).keys())
    if len(base_penalties) == 0:
        raise ValueError(f"No penalties found in {base_cfg_path}")

    rows_path = out_root / "results.csv"
    rows = read_csv(rows_path) if args.reuse_existing else []
    done = {(r.get("dataset"), r.get("candidate")) for r in rows if r.get("status") == "ok"}
    cfg_root = out_root / "configs" / dataset
    run_root = out_root / "runs" / dataset

    base_out = run_root / "base_no_moe"
    base_ckpt = base_out / "base_checkpoint.pt"
    base_cfg_run = copy.deepcopy(base_cfg)
    fixed_protocol(base_cfg_run, base_out, args.device, args.base_epochs or args.epochs)
    configure_base(base_cfg_run, base_ckpt)
    base_cfg_path_run = cfg_root / "base_no_moe.yaml"
    write_yaml(base_cfg_path_run, base_cfg_run)
    print(f"[base] {dataset}", flush=True)
    base_code, base_sec, base_err = run_train(base_cfg_path_run, base_out, args.reuse_existing)
    if base_code != 0:
        raise RuntimeError(base_err)
    if not base_ckpt.exists():
        raise RuntimeError(f"Base checkpoint not found: {base_ckpt}")
    base_summary = read_json(base_out / "run_summary.json")

    candidates = candidate_grid(base_penalties)
    if args.max_candidates is not None:
        candidates = candidates[: max(0, int(args.max_candidates))]

    for cand in candidates:
        key = (dataset, cid(cand))
        if args.reuse_existing and key in done:
            print(f"[skip] {dataset} {cid(cand)}", flush=True)
            continue
        cand_dir = run_root / cid(cand)
        cfg_dir = cfg_root / cid(cand)
        zero_out = cand_dir / "staged_zero_lambda"
        full_out = cand_dir / "staged_full"
        zero_cfg_path = cfg_dir / "staged_zero_lambda.yaml"
        full_cfg_path = cfg_dir / "staged_full.yaml"

        zero_cfg = copy.deepcopy(base_cfg)
        fixed_protocol(zero_cfg, zero_out, args.device, args.epochs)
        configure_staged_moe(zero_cfg, cand, lambda_value=0.0, checkpoint_path=base_ckpt)
        full_cfg = copy.deepcopy(base_cfg)
        fixed_protocol(full_cfg, full_out, args.device, args.epochs)
        configure_staged_moe(full_cfg, cand, lambda_value=float(cand.lambda_init), checkpoint_path=base_ckpt)
        write_yaml(zero_cfg_path, zero_cfg)
        write_yaml(full_cfg_path, full_cfg)

        print(f"[run] {dataset} {cid(cand)}", flush=True)
        zero_code, zero_sec, zero_err = run_train(zero_cfg_path, zero_out, args.reuse_existing)
        full_code, full_sec, full_err = run_train(full_cfg_path, full_out, args.reuse_existing)
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
            "gate_max_scale": cand.gate_max_scale,
            "gate_init_scale": cand.gate_init_scale,
            "gate_scale_reg": cand.gate_scale_reg,
            "gate_train_fraction": cand.gate_train_fraction,
            "penalty_guard_allow_multi": cand.penalty_guard_allow_multi,
            "base_checkpoint": str(base_ckpt),
            "base_out_dir": str(base_out),
            "zero_out_dir": str(zero_out),
            "full_out_dir": str(full_out),
            "zero_config": str(zero_cfg_path),
            "full_config": str(full_cfg_path),
            "wrapper_sec": base_sec + zero_sec + full_sec,
            "error": error,
        }
        if not error:
            zero_summary = read_json(zero_out / "run_summary.json")
            full_summary = read_json(full_out / "run_summary.json")
            base_mse = metric(base_summary, "test", "avg_mse")
            zero_mse = metric(zero_summary, "test", "avg_mse")
            full_mse = metric(full_summary, "test", "avg_mse")
            row.update(
                {
                    "base_test_mse": base_mse,
                    "zero_test_mse": zero_mse,
                    "full_test_mse": full_mse,
                    "zero_gain_pct": 100.0 * (base_mse - zero_mse) / max(abs(base_mse), 1.0e-12),
                    "full_gain_pct": 100.0 * (base_mse - full_mse) / max(abs(base_mse), 1.0e-12),
                    "full_vs_zero_mse": zero_mse - full_mse,
                    "base_test_mae": metric(base_summary, "test", "avg_mae"),
                    "zero_test_mae": metric(zero_summary, "test", "avg_mae"),
                    "full_test_mae": metric(full_summary, "test", "avg_mae"),
                    "base_val_mse": metric(base_summary, "val", "avg_mse"),
                    "zero_val_mse": metric(zero_summary, "val", "avg_mse"),
                    "full_val_mse": metric(full_summary, "val", "avg_mse"),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "etth_h96_staged_moe_search")
    parser.add_argument("--datasets", type=str, default="ETTh1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--base-epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    out_root = resolve(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    config_map = {
        "ETTh1": ROOT / "configs" / "ETTh1.yaml",
        "ETTh2": ROOT / "configs" / "ETTh2.yaml",
        "ETTm1": ROOT / "configs" / "ETTm1.yaml",
    }
    for dataset in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        if dataset not in config_map:
            raise ValueError(f"Unsupported dataset: {dataset}")
        run_dataset(args, dataset, config_map[dataset], out_root)
    print(f"Saved: {out_root / 'results.csv'}")


if __name__ == "__main__":
    main()
