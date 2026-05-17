from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]

TARGETS = {
    "ETTh1": {
        "config": ROOT / "configs" / "ETTm1ToETTh1.yaml",
        "csv": "data/ETTh1.csv",
        "batch_size": 512,
    },
    "ETTh2": {
        "config": ROOT / "configs" / "ETTm1ToETTh2.yaml",
        "csv": "data/ETTh2.csv",
        "batch_size": 512,
    },
    "ETTm2": {
        "config": ROOT / "configs" / "ETTm1ToETTm2.yaml",
        "csv": "data/ETTm2.csv",
        "batch_size": 512,
    },
    "Weather": {
        "config": None,
        "csv": "data/weather.csv",
        "batch_size": 256,
    },
    "Traffic": {
        "config": ROOT / "configs" / "ETTm1ToTraffic.yaml",
        "csv": "data/traffic.csv",
        "batch_size": 32,
    },
}

FIELDS = [
    "status",
    "target",
    "config",
    "out_dir",
    "search_mode",
    "selected_val_mse",
    "selected_val_mae",
    "selected_test_mse",
    "selected_test_mae",
    "selected_route_counts",
    "selected_route_length",
    "selected_test_config",
    "error",
]


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})


def make_weather_config(base_path: Path, out_path: Path) -> Path:
    cfg = read_yaml(base_path)
    cfg.setdefault("exp", {})
    cfg["exp"]["name"] = "ETTm1_to_Weather"
    cfg["exp"]["out_dir"] = "outputs/ETTm1ToWeather"
    cfg.setdefault("data", {})
    cfg["data"]["csv_path"] = "data/weather.csv"
    cfg["data"]["date_col"] = 0
    cfg["data"]["train_ratio"] = 0.7
    cfg["data"]["val_ratio"] = 0.1
    cfg["data"]["test_ratio"] = 0.2
    cfg.setdefault("window", {})
    cfg["window"]["input_len"] = 336
    cfg["window"]["pred_len"] = 96
    cfg.setdefault("normalize", {})
    cfg["normalize"]["global_zscore"] = True
    cfg["normalize"]["train_only"] = True
    transfer = cfg.setdefault("transfer", {})
    transfer["route_fit_scope"] = "train"
    transfer["use_pred_residual"] = True
    transfer["corr_mode"] = "cycle_template"
    transfer["phase_bins"] = 64
    transfer["phase_max_shift"] = None
    transfer["period_min_hours"] = 12
    transfer["period_max_hours"] = 168
    transfer["corr_align"] = "head"
    transfer.setdefault("resample", {})
    transfer["resample"].update({"enable": True, "target_step_minutes": 15, "method": "linear"})
    transfer.setdefault("knn_hybrid", {})["enable"] = False
    transfer["save_corr"] = True
    cfg.setdefault("eval", {})["batch_size"] = 64
    write_yaml(out_path, cfg)
    return out_path


def run_selection(config_path: Path, out_dir: Path, batch_size: int, python: Path | None) -> tuple[int, str]:
    py = str(python) if python is not None else sys.executable
    cmd = [
        py,
        "-u",
        "scripts/run_ettm1_to_ettm2_val_loss_route_selection.py",
        "--config",
        str(config_path),
        "--out-root",
        str(out_dir),
        "--batch-size",
        str(batch_size),
        "--search-mode",
        "auto",
        "--max-greedy-channels",
        "64",
    ]
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.returncode, proc.stdout


def collect_row(target: str, config_path: Path, out_dir: Path, status: str, error: str = "") -> dict[str, Any]:
    row: dict[str, Any] = {
        "status": status,
        "target": target,
        "config": str(config_path),
        "out_dir": str(out_dir),
        "error": error,
    }
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        route = [int(v) for v in summary.get("selected_route", [])]
        row.update(
            {
                "search_mode": summary.get("search_mode"),
                "selected_val_mse": summary.get("selected_val_mse"),
                "selected_val_mae": summary.get("selected_val_mae"),
                "selected_test_mse": summary.get("selected_test_mse"),
                "selected_test_mae": summary.get("selected_test_mae"),
                "selected_route_counts": json.dumps(dict(sorted(Counter(route).items())), ensure_ascii=False),
                "selected_route_length": len(route),
                "selected_test_config": summary.get("selected_test_config"),
            }
        )
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "ettm1_to_five_targets_val_loss_transfer")
    ap.add_argument("--python", type=Path, default=None)
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    weather_config = make_weather_config(
        ROOT / "configs" / "ETTm1ToETTm2.yaml",
        args.out_root / "configs" / "ETTm1ToWeather.yaml",
    )
    for target, spec in TARGETS.items():
        config_path = Path(spec["config"]) if spec["config"] is not None else weather_config
        out_dir = args.out_root / target
        print(f"[run] ETTm1 -> {target}")
        code, output = run_selection(config_path, out_dir, int(spec["batch_size"]), args.python)
        (out_dir / "runner.log").parent.mkdir(parents=True, exist_ok=True)
        (out_dir / "runner.log").write_text(output, encoding="utf-8")
        if code == 0:
            row = collect_row(target, config_path, out_dir, "ok")
            print(
                f"[ok] {target} test_mse={row.get('selected_test_mse')} "
                f"test_mae={row.get('selected_test_mae')}"
            )
        else:
            row = collect_row(target, config_path, out_dir, "failed", output[-3000:])
            print(f"[failed] {target}")
        rows.append(row)
        write_rows(args.out_root / "transfer.csv", rows)
    write_rows(args.out_root / "transfer.csv", rows)


if __name__ == "__main__":
    main()
