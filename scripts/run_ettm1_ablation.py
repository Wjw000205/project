from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]

FIELDS = [
    "status",
    "horizon",
    "variant",
    "description",
    "moe_enable",
    "dynamic_lambda_enable",
    "pred_side_residual_enable",
    "penalties",
    "epochs",
    "batch_size",
    "val_mse",
    "val_mae",
    "test_mse",
    "test_mae",
    "selected_variant",
    "selected_mse",
    "selected_mae",
    "best_epoch",
    "total_sec",
    "avg_epoch_sec",
    "wrapper_sec",
    "config_path",
    "out_dir",
    "returncode",
    "error",
]


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def variant_specs(suite: str, full_penalties: list[str]) -> list[dict[str, Any]]:
    current = [
        {
            "name": "base_no_moe",
            "description": "MLP base only; MoE, dynamic lambda, and penalty-keyed residual experts disabled",
            "moe": False,
            "dynamic": False,
            "residual": False,
            "penalties": full_penalties,
        },
        {
            "name": "residual_no_penalty_loss",
            "description": "Penalty-keyed residual experts enabled, but explicit routed penalty loss disabled by zero lambdas",
            "moe": True,
            "dynamic": False,
            "residual": True,
            "penalties": full_penalties,
            "lambda_override": 0.0,
        },
        {
            "name": "residual_full",
            "description": "Current architecture: penalty-keyed residual experts with fixed penalty lambdas; dynamic lambda disabled",
            "moe": True,
            "dynamic": False,
            "residual": True,
            "penalties": full_penalties,
        },
        {
            "name": "residual_no_gate_bias",
            "description": "Current architecture without gate_init_bias",
            "moe": True,
            "dynamic": False,
            "residual": True,
            "penalties": full_penalties,
            "gate_init_bias": False,
        },
        {
            "name": "residual_no_pred_aware",
            "description": "Current architecture without pred-aware routing features",
            "moe": True,
            "dynamic": False,
            "residual": True,
            "penalties": full_penalties,
            "pred_aware": False,
        },
        {
            "name": "residual_no_penalty_ema",
            "description": "Current architecture without penalty EMA smoothing",
            "moe": True,
            "dynamic": False,
            "residual": True,
            "penalties": full_penalties,
            "penalty_ema": False,
        },
    ]
    penalty = [
        {
            "name": "penalty_level",
            "description": "Current residual architecture with level only",
            "moe": True,
            "dynamic": False,
            "residual": True,
            "penalties": ["level"],
        },
        {
            "name": "penalty_level_delta",
            "description": "Current residual architecture with level and delta",
            "moe": True,
            "dynamic": False,
            "residual": True,
            "penalties": ["level", "delta"],
        },
        {
            "name": "penalty_level_delta_d2",
            "description": "Current residual architecture with level, delta, d2_match",
            "moe": True,
            "dynamic": False,
            "residual": True,
            "penalties": ["level", "delta", "d2_match"],
        },
        {
            "name": "penalty_level_delta_diff",
            "description": "Current residual architecture with level, delta, diff_amp",
            "moe": True,
            "dynamic": False,
            "residual": True,
            "penalties": ["level", "delta", "diff_amp"],
        },
        {
            "name": "penalty_no_level",
            "description": "Current residual architecture without level",
            "moe": True,
            "dynamic": False,
            "residual": True,
            "penalties": ["delta", "d2_match", "diff_amp"],
        },
        {
            "name": "penalty_no_delta",
            "description": "Current residual architecture without delta",
            "moe": True,
            "dynamic": False,
            "residual": True,
            "penalties": ["level", "d2_match", "diff_amp"],
        },
        {
            "name": "penalty_no_d2",
            "description": "Current residual architecture without d2_match",
            "moe": True,
            "dynamic": False,
            "residual": True,
            "penalties": ["level", "delta", "diff_amp"],
        },
        {
            "name": "penalty_no_diff",
            "description": "Current residual architecture without diff_amp",
            "moe": True,
            "dynamic": False,
            "residual": True,
            "penalties": ["level", "delta", "d2_match"],
        },
    ]
    negative_control = [
        {
            "name": "legacy_penalty_loss_only",
            "description": "Negative control: routed penalty loss without prediction-side residual experts",
            "moe": True,
            "dynamic": False,
            "residual": False,
            "penalties": full_penalties,
        }
    ]
    if suite == "current":
        return current
    if suite == "penalty":
        return [current[2]] + penalty
    if suite == "negative":
        return [current[0], current[2]] + negative_control
    if suite == "full":
        return current + penalty + negative_control
    raise ValueError(f"Unknown suite: {suite}")


def configure_paths(cfg: dict[str, Any], out_dir: Path) -> None:
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
    cfg.setdefault("eval", {})["skip_test"] = False


def apply_variant(cfg: dict[str, Any], spec: dict[str, Any]) -> None:
    penalties = list(spec["penalties"])
    cfg.setdefault("penalties", {})["enabled"] = penalties

    moe = cfg.setdefault("moe", {})
    moe["enable"] = bool(spec["moe"])
    moe.setdefault("dynamic_lambda", {})["enable"] = bool(spec["dynamic"]) and bool(spec["moe"])
    moe.setdefault("pred_side_residual", {})["enable"] = bool(spec["residual"]) and bool(spec["moe"])
    moe.setdefault("learnable_lambda", {})["enable"] = False

    # Keep lambda scale consistent with the landed ETTm1 config.
    original = moe.get("lambda_init", {}) or {}
    default_lambda = float(next(iter(original.values()), 0.1))
    if "lambda_override" in spec:
        default_lambda = float(spec["lambda_override"])
    moe["lambda_init"] = {name: float(original.get(name, default_lambda)) for name in penalties}
    if "lambda_override" in spec:
        moe["lambda_init"] = {name: float(spec["lambda_override"]) for name in penalties}
    moe["lambda_min"] = {name: 0.0 for name in penalties}
    moe["lambda_schedule"] = {name: "none" for name in penalties}

    if "gate_init_bias" in spec:
        moe.setdefault("gate_init_bias", {})["enable"] = bool(spec["gate_init_bias"])
    if "pred_aware" in spec:
        moe.setdefault("pred_aware", {})["enable"] = bool(spec["pred_aware"])
    if "penalty_ema" in spec:
        moe.setdefault("penalty_ema", {})["enable"] = bool(spec["penalty_ema"])


def build_config(
    base: dict[str, Any],
    *,
    spec: dict[str, Any],
    horizon: int,
    out_dir: Path,
    epochs: int | None,
    batch_size: int | None,
    device: str | None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    csv_name = Path(str(cfg.get("data", {}).get("csv_path", "ETT.csv"))).stem
    cfg.setdefault("exp", {})["name"] = f"{csv_name}_H{horizon}_{spec['name']}"
    if device:
        cfg["exp"]["device"] = str(device)
    cfg.setdefault("window", {})["input_len"] = int(cfg["window"].get("input_len", 336))
    cfg["window"]["pred_len"] = int(horizon)
    cfg.setdefault("normalize", {})["train_only"] = True
    cfg.setdefault("cluster", {})["train_only"] = True
    if epochs is not None:
        cfg.setdefault("train", {})["epochs"] = int(epochs)
    if batch_size is not None:
        cfg.setdefault("train", {})["batch_size"] = int(batch_size)
    configure_paths(cfg, out_dir)
    apply_variant(cfg, spec)
    return cfg


def run_config(config_path: Path, out_dir: Path) -> tuple[int, str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", "-m", "src.train", "--config", str(config_path)]
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    wrapper_sec = time.perf_counter() - start
    (out_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    return int(proc.returncode), proc.stdout, wrapper_sec


def row_from_summary(
    *,
    status: str,
    horizon: int,
    spec: dict[str, Any],
    cfg: dict[str, Any],
    config_path: Path,
    out_dir: Path,
    returncode: int,
    wrapper_sec: float,
    error: str = "",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "status": status,
        "horizon": horizon,
        "variant": spec["name"],
        "description": spec["description"],
        "moe_enable": cfg.get("moe", {}).get("enable", ""),
        "dynamic_lambda_enable": cfg.get("moe", {}).get("dynamic_lambda", {}).get("enable", ""),
        "pred_side_residual_enable": cfg.get("moe", {}).get("pred_side_residual", {}).get("enable", ""),
        "penalties": "|".join(cfg.get("penalties", {}).get("enabled", [])),
        "epochs": cfg.get("train", {}).get("epochs", ""),
        "batch_size": cfg.get("train", {}).get("batch_size", ""),
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "returncode": returncode,
        "wrapper_sec": wrapper_sec,
        "error": error,
    }
    summary_path = out_dir / "run_summary.json"
    if not summary_path.exists():
        return row
    summary = load_summary(summary_path)
    val = summary.get("val", {}) or {}
    test = summary.get("test", {}) or {}
    selected = summary.get("selected", {}) or {}
    timing = summary.get("timing", {}) or {}
    row.update(
        {
            "val_mse": val.get("avg_mse", ""),
            "val_mae": val.get("avg_mae", ""),
            "test_mse": test.get("avg_mse", ""),
            "test_mae": test.get("avg_mae", ""),
            "selected_variant": selected.get("variant", "base"),
            "selected_mse": selected.get("avg_mse", test.get("avg_mse", "")),
            "selected_mae": selected.get("avg_mae", test.get("avg_mae", "")),
            "best_epoch": json.dumps(summary.get("best_epoch", "")),
            "total_sec": timing.get("total_sec", ""),
            "avg_epoch_sec": timing.get("avg_epoch_sec", ""),
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, default=ROOT / "configs" / "ETTm1.yaml")
    parser.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "ettm1_ablation")
    parser.add_argument("--horizons", type=str, default="96")
    parser.add_argument("--suite", choices=["current", "penalty", "negative", "full"], default="current")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_path = args.base_config if args.base_config.is_absolute() else ROOT / args.base_config
    out_root = args.out_root if args.out_root.is_absolute() else ROOT / args.out_root
    horizons = [int(v.strip()) for v in args.horizons.split(",") if v.strip()]
    base_cfg = read_yaml(base_path)
    base_penalties = list((base_cfg.get("penalties", {}) or {}).get("enabled", []))
    if not base_penalties:
        base_penalties = list((base_cfg.get("moe", {}) or {}).get("lambda_init", {}).keys())
    if not base_penalties:
        raise ValueError(f"No penalties found in {base_path}")
    specs = variant_specs(args.suite, base_penalties)
    rows: list[dict[str, Any]] = []
    results_path = out_root / "results.csv"
    if results_path.exists() and args.reuse_existing:
        with results_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    done = {(str(r.get("horizon")), str(r.get("variant"))) for r in rows if r.get("status") == "ok"}

    for horizon in horizons:
        for spec in specs:
            key = (str(horizon), str(spec["name"]))
            out_dir = out_root / "runs" / f"H{horizon}" / spec["name"]
            config_path = out_root / "configs" / f"H{horizon}_{spec['name']}.yaml"
            cfg = build_config(
                base_cfg,
                spec=spec,
                horizon=horizon,
                out_dir=out_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
            )
            write_yaml(config_path, cfg)
            if key in done and args.reuse_existing:
                print(f"[skip] H{horizon} {spec['name']}", flush=True)
                continue
            print(f"[run] H{horizon} {spec['name']}", flush=True)
            if args.dry_run:
                row = row_from_summary(
                    status="prepared",
                    horizon=horizon,
                    spec=spec,
                    cfg=cfg,
                    config_path=config_path,
                    out_dir=out_dir,
                    returncode=0,
                    wrapper_sec=0.0,
                )
            else:
                returncode, output, wrapper_sec = run_config(config_path, out_dir)
                status = "ok" if returncode == 0 and (out_dir / "run_summary.json").exists() else "error"
                row = row_from_summary(
                    status=status,
                    horizon=horizon,
                    spec=spec,
                    cfg=cfg,
                    config_path=config_path,
                    out_dir=out_dir,
                    returncode=returncode,
                    wrapper_sec=wrapper_sec,
                    error="" if status == "ok" else output[-4000:],
                )
            rows = [r for r in rows if (str(r.get("horizon")), str(r.get("variant"))) != key]
            rows.append(row)
            write_rows(results_path, rows)
            print(
                f"  -> {row['status']} val={row.get('val_mse', '')} test={row.get('test_mse', '')}",
                flush=True,
            )
    write_rows(results_path, rows)
    print(f"Saved: {results_path}")


if __name__ == "__main__":
    main()
