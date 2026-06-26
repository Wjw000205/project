from __future__ import annotations

import argparse
import copy
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
    / "weather_h96_p144_refine"
    / "configs"
    / "p144_stat025_resid060_mae.yaml"
)

VARIANTS: dict[str, dict[str, Any]] = {
    "p144_stat010_resid020_seg8_mae": {"stat": 0.10, "resid": 0.20, "segments": 8},
    "p144_stat015_resid030_seg8_mae": {"stat": 0.15, "resid": 0.30, "segments": 8},
    "p144_stat015_resid035_seg8_mae": {"stat": 0.15, "resid": 0.35, "segments": 8},
    "p144_stat015_resid040_seg8_mae": {"stat": 0.15, "resid": 0.40, "segments": 8},
    "p144_stat020_resid030_seg8_mae": {"stat": 0.20, "resid": 0.30, "segments": 8},
    "p144_stat020_resid035_seg8_mae": {"stat": 0.20, "resid": 0.35, "segments": 8},
    "p144_stat020_resid040_seg4_mae": {"stat": 0.20, "resid": 0.40, "segments": 4},
    "p144_stat020_resid040_seg8_mae": {"stat": 0.20, "resid": 0.40, "segments": 8},
    "p144_stat020_resid050_seg8_mae": {"stat": 0.20, "resid": 0.50, "segments": 8},
    "p144_stat025_resid040_seg8_mae": {"stat": 0.25, "resid": 0.40, "segments": 8},
    "p144_stat025_resid050_seg8_mae": {"stat": 0.25, "resid": 0.50, "segments": 8},
    "p144_stat030_resid050_seg8_mae": {"stat": 0.30, "resid": 0.50, "segments": 8},
    "p144_stat025_resid060_seg4_mae": {"stat": 0.25, "resid": 0.60, "segments": 4},
    "p144_stat025_resid060_seg12_mae": {"stat": 0.25, "resid": 0.60, "segments": 12},
}

FIELDS = [
    "variant",
    "status",
    "test_mse",
    "test_mae",
    "val_mse",
    "val_mae",
    "target_mse_gap",
    "target_mae_gap",
    "total_sec",
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


def localize_paths(cfg: dict[str, Any], *, out_dir: Path, variant: str, device: str) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("exp", {})
    cfg["exp"]["name"] = f"weather_h96_{variant}"
    cfg["exp"]["out_dir"] = str(out_dir)
    cfg["exp"]["device"] = device
    cfg.setdefault("corr", {})["save_path"] = str(out_dir / "corr.npy")
    cfg.setdefault("portrait", {})["out_dir"] = str(out_dir / "cluster_portraits")
    cfg.setdefault("memory", {})["path"] = str(out_dir / "cluster_memory.pt")
    cfg.setdefault("memory", {})["checkpoint_path"] = str(out_dir / "best_checkpoint.pt")
    cfg.setdefault("eval", {})["skip_test"] = False
    return cfg


def apply_variant(cfg: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("window", {})["input_len"] = 96
    cfg.setdefault("window", {})["pred_len"] = 96
    cfg.setdefault("moe", {})["freeze_backbone"] = True
    cfg.setdefault("train", {})["epochs"] = 1
    cfg["train"]["lr"] = 0.0
    cfg["train"]["selection_metric"] = "val_mae"
    cfg["train"].setdefault("mae_objective", {}).update(
        {"enable": True, "kind": "l1", "weight": 2.0, "warmup_epochs": 0}
    )

    stat_sel = cfg["moe"]["train_stat_anchor_expert"]["scale_selection"]
    stat_sel.update({"enable": True, "metric": "mae", "max_scale": spec["stat"], "steps": 17})

    resid_sel = cfg["moe"]["train_residual_anchor_expert"]["scale_selection"]
    resid_sel.update(
        {
            "enable": True,
            "metric": "mae",
            "max_scale": spec["resid"],
            "steps": 25,
            "horizon_segments": spec["segments"],
        }
    )
    return cfg


def run_train(py: str, cfg_path: Path, out_dir: Path) -> tuple[int, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(
        [py, "-u", "-m", "src.train", "--config", str(cfg_path)],
        cwd=ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8", errors="replace")
    return int(proc.returncode), proc.stdout


def summarize(variant: str, cfg_path: Path, out_dir: Path, returncode: int, output: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "variant": variant,
        "status": "ok" if returncode == 0 else "error",
        "config_path": str(cfg_path),
        "out_dir": str(out_dir),
        "returncode": returncode,
        "error": "" if returncode == 0 else output[-3000:],
    }
    summary_path = out_dir / "run_summary.json"
    if not summary_path.exists():
        row["status"] = "error"
        row["error"] = row.get("error") or "run_summary.json missing"
        return row
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    val = summary.get("val") or {}
    test = summary.get("test") or {}
    timing = summary.get("timing") or {}
    test_mse = test.get("avg_mse", "")
    test_mae = test.get("avg_mae", "")
    row.update(
        {
            "val_mse": val.get("avg_mse", ""),
            "val_mae": val.get("avg_mae", ""),
            "test_mse": test_mse,
            "test_mae": test_mae,
            "target_mse_gap": float(test_mse) - 0.153 if test_mse != "" else "",
            "target_mae_gap": float(test_mae) - 0.190 if test_mae != "" else "",
            "total_sec": timing.get("total_sec", ""),
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine Weather H96 period-144 MAE prior anchors.")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "outputs" / "codex_table_target_20260615_weather_h96_p144_mae_prior_refine2",
    )
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    base = read_yaml(BASE_CONFIG)
    rows: list[dict[str, Any]] = []
    for variant in args.variants:
        spec = VARIANTS[variant]
        out_dir = args.out_root / "runs" / variant
        cfg_path = args.out_root / "configs" / f"{variant}.yaml"
        cfg = localize_paths(apply_variant(base, spec), out_dir=out_dir, variant=variant, device=args.device)
        write_yaml(cfg_path, cfg)
        summary_path = out_dir / "run_summary.json"
        if summary_path.exists() and not args.rerun:
            print(f"[reuse] {variant}", flush=True)
            row = summarize(variant, cfg_path, out_dir, 0, "")
        else:
            print(f"[run] {variant}", flush=True)
            returncode, output = run_train(args.python, cfg_path, out_dir)
            row = summarize(variant, cfg_path, out_dir, returncode, output)
        rows.append(row)
        write_rows(args.out_root / "results.csv", rows)
    write_rows(args.out_root / "results.csv", rows)


if __name__ == "__main__":
    main()
