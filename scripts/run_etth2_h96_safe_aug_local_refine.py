from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = (
    ROOT
    / "outputs"
    / "codex_table_target_20260614"
    / "etth2_h96_safe_aug_mae_refine1"
    / "configs"
    / "ETTh2"
    / "H96"
    / "expert_probe"
    / "gate_mae_alpha1p2_clip3.yaml"
)
BASELINE_MSE = 0.2849876563610394
BASELINE_MAE = 0.341655


FIELDS = [
    "variant",
    "status",
    "test_mse",
    "test_mae",
    "val_mse",
    "val_mae",
    "mse_gain_pct",
    "mae_gain_pct",
    "alpha_scale",
    "residual_clip",
    "gate_max_scale",
    "gate_init_scale",
    "gate_scale_reg",
    "gate_train_fraction",
    "mse_weight",
    "mae_weight",
    "config_path",
    "out_dir",
    "returncode",
    "error",
    "total_sec",
]


VARIANTS: list[dict[str, Any]] = [
    {
        "name": "alpha1p20_clip3_ms1p55_reg1e4_tf075_mae07",
        "alpha_scale": 1.20,
        "residual_clip": 3.0,
        "gate_max_scale": 1.55,
        "gate_init_scale": 0.40,
        "gate_scale_reg": 1.0e-4,
        "gate_train_fraction": 0.75,
        "selection_scale_max": 1.55,
        "mse_weight": 0.75,
        "mae_weight": 0.70,
    },
    {
        "name": "alpha1p25_clip3_ms1p55_reg2e4_tf070_mae06",
        "alpha_scale": 1.25,
        "residual_clip": 3.0,
        "gate_max_scale": 1.55,
        "gate_init_scale": 0.40,
        "gate_scale_reg": 2.0e-4,
        "gate_train_fraction": 0.70,
        "selection_scale_max": 1.55,
        "mse_weight": 0.80,
        "mae_weight": 0.60,
    },
    {
        "name": "alpha1p20_clip2p5_ms1p50_reg15e5_tf075_mae08",
        "alpha_scale": 1.20,
        "residual_clip": 2.5,
        "gate_max_scale": 1.50,
        "gate_init_scale": 0.40,
        "gate_scale_reg": 1.5e-4,
        "gate_train_fraction": 0.75,
        "selection_scale_max": 1.50,
        "mse_weight": 0.70,
        "mae_weight": 0.80,
    },
    {
        "name": "alpha1p30_clip3p5_ms1p65_reg2e4_tf070_mae07",
        "alpha_scale": 1.30,
        "residual_clip": 3.5,
        "gate_max_scale": 1.65,
        "gate_init_scale": 0.45,
        "gate_scale_reg": 2.0e-4,
        "gate_train_fraction": 0.70,
        "selection_scale_max": 1.65,
        "mse_weight": 0.75,
        "mae_weight": 0.70,
    },
]


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} did not contain a mapping")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def configure(base: dict[str, Any], variant: dict[str, Any], out_root: Path, device: str) -> dict[str, Any]:
    cfg = json.loads(json.dumps(base))
    name = str(variant["name"])
    out_dir = out_root / "runs" / "ETTh2" / "H96" / "expert_probe" / name
    cfg.setdefault("exp", {})["name"] = f"ETTh2_input96_H96_expert_probe_{name}"
    cfg["exp"]["out_dir"] = str(out_dir)
    cfg["exp"]["device"] = device
    cfg.setdefault("corr", {})["save_path"] = str(out_dir / "corr.npy")
    cfg.setdefault("portrait", {})["out_dir"] = str(out_dir / "cluster_portraits")
    cfg.setdefault("knn_hybrid", {})["path"] = str(out_dir / "knn_shape_bank.pt")
    cfg.setdefault("memory", {})["path"] = str(out_dir / "cluster_memory.pt")
    cfg["memory"]["checkpoint_path"] = str(out_dir / "best_checkpoint.pt")

    residual = cfg["moe"]["pred_side_residual"]
    residual["alpha_scale"] = float(variant["alpha_scale"])
    residual["residual_clip"] = float(variant["residual_clip"])
    residual["selection_scale_max"] = float(variant["selection_scale_max"])

    gate = residual.setdefault("gate_calibrator", {})
    gate["max_scale"] = float(variant["gate_max_scale"])
    gate["init_scale"] = float(variant["gate_init_scale"])
    gate["scale_reg"] = float(variant["gate_scale_reg"])
    gate["train_fraction"] = float(variant["gate_train_fraction"])

    train = cfg.setdefault("train", {})
    train["mse_weight"] = float(variant["mse_weight"])
    train.setdefault("mae_objective", {})["weight"] = float(variant["mae_weight"])
    return cfg


def run_train(python_exe: str, cfg_path: Path, out_dir: Path) -> tuple[int, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(
        [python_exe, "-u", "-m", "src.train", "--config", str(cfg_path)],
        cwd=ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (out_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8", errors="replace")
    return int(proc.returncode), proc.stdout


def summarize(variant: dict[str, Any], cfg_path: Path, out_dir: Path, returncode: int, output: str) -> dict[str, Any]:
    row = {
        "variant": variant["name"],
        "status": "ok" if returncode == 0 else "error",
        "config_path": str(cfg_path),
        "out_dir": str(out_dir),
        "returncode": returncode,
        "error": "" if returncode == 0 else output[-3000:],
        "alpha_scale": variant["alpha_scale"],
        "residual_clip": variant["residual_clip"],
        "gate_max_scale": variant["gate_max_scale"],
        "gate_init_scale": variant["gate_init_scale"],
        "gate_scale_reg": variant["gate_scale_reg"],
        "gate_train_fraction": variant["gate_train_fraction"],
        "mse_weight": variant["mse_weight"],
        "mae_weight": variant["mae_weight"],
    }
    summary_path = out_dir / "run_summary.json"
    if not summary_path.exists():
        row["status"] = "error"
        row["error"] = row["error"] or "run_summary.json missing"
        return row
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    test = summary.get("test") or {}
    val = summary.get("val") or {}
    timing = summary.get("timing") or {}
    test_mse = test.get("avg_mse", "")
    test_mae = test.get("avg_mae", "")
    row.update(
        {
            "test_mse": test_mse,
            "test_mae": test_mae,
            "val_mse": val.get("avg_mse", ""),
            "val_mae": val.get("avg_mae", ""),
            "total_sec": timing.get("total_sec", ""),
        }
    )
    if test_mse != "":
        row["mse_gain_pct"] = (BASELINE_MSE - float(test_mse)) / BASELINE_MSE * 100.0
    if test_mae != "":
        row["mae_gain_pct"] = (BASELINE_MAE - float(test_mae)) / BASELINE_MAE * 100.0
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Local ETTh2 H96 safe-augmented MAE residual refinement.")
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "codex_table_target_20260615_etth2_h96_safe_aug_mae_refine3")
    ap.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    ap.add_argument("--variants", nargs="+", default=[v["name"] for v in VARIANTS])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = read_yaml(args.base_config)
    selected = [v for v in VARIANTS if v["name"] in set(args.variants)]
    if not selected:
        raise ValueError("No matching variants selected.")

    out_root = args.out_root
    rows: list[dict[str, Any]] = []
    for variant in selected:
        cfg = configure(base, variant, out_root, args.device)
        name = str(variant["name"])
        cfg_path = out_root / "configs" / "ETTh2" / "H96" / "expert_probe" / f"{name}.yaml"
        out_dir = out_root / "runs" / "ETTh2" / "H96" / "expert_probe" / name
        write_yaml(cfg_path, cfg)
        print(f"[run] {name}", flush=True)
        if args.dry_run:
            row = {
                "variant": name,
                "status": "planned",
                "config_path": str(cfg_path),
                "out_dir": str(out_dir),
                **{key: variant[key] for key in variant if key != "name"},
            }
        else:
            returncode, output = run_train(args.python, cfg_path, out_dir)
            row = summarize(variant, cfg_path, out_dir, returncode, output)
        rows.append(row)
        write_rows(out_root / "results.csv", rows)
        print(
            f"[{row['status']}] {name} test={row.get('test_mse')}/{row.get('test_mae')} "
            f"gain={row.get('mse_gain_pct')}/{row.get('mae_gain_pct')}",
            flush=True,
        )
    write_rows(out_root / "results.csv", rows)


if __name__ == "__main__":
    main()
