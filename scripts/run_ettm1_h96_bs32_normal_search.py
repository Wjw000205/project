from __future__ import annotations

import argparse
import csv
import copy
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=False), encoding="utf-8")


def set_run_paths(cfg: dict[str, Any], out_dir: Path, name: str) -> None:
    cfg.setdefault("exp", {})
    cfg["exp"]["name"] = name
    cfg["exp"]["out_dir"] = str(out_dir)

    cfg.setdefault("corr", {})
    cfg["corr"]["save_path"] = str(out_dir / "corr.npy")

    cfg.setdefault("plot", {})
    cfg["plot"]["enable"] = False

    cfg.setdefault("portrait", {})
    cfg["portrait"]["enable"] = False
    cfg["portrait"]["out_dir"] = str(out_dir / "cluster_portraits")

    cfg.setdefault("memory", {})
    cfg["memory"]["enable"] = False
    cfg["memory"]["save_checkpoint"] = False
    cfg["memory"]["path"] = str(out_dir / "cluster_memory.pt")
    cfg["memory"]["checkpoint_path"] = str(out_dir / "best_checkpoint.pt")



def candidates() -> list[dict[str, Any]]:
    return [
        {
            "name": "lr7e4_pat5",
            "lr": 7.0e-4,
            "weight_decay": 1.0e-4,
            "epochs": 120,
            "early_patience": 14,
            "sched_patience": 5,
            "sched_factor": 0.5,
            "mae_weight": 0.6,
        },
        {
            "name": "lr5e4_pat5",
            "lr": 5.0e-4,
            "weight_decay": 1.0e-4,
            "epochs": 140,
            "early_patience": 16,
            "sched_patience": 5,
            "sched_factor": 0.5,
            "mae_weight": 0.6,
        },
        {
            "name": "lr7e4_wd5e5",
            "lr": 7.0e-4,
            "weight_decay": 5.0e-5,
            "epochs": 120,
            "early_patience": 14,
            "sched_patience": 5,
            "sched_factor": 0.5,
            "mae_weight": 0.6,
        },
        {
            "name": "lr3e4_pat6",
            "lr": 3.0e-4,
            "weight_decay": 1.0e-4,
            "epochs": 160,
            "early_patience": 18,
            "sched_patience": 6,
            "sched_factor": 0.5,
            "mae_weight": 0.6,
        },
        {
            "name": "lr7e4_mae05",
            "lr": 7.0e-4,
            "weight_decay": 1.0e-4,
            "epochs": 120,
            "early_patience": 14,
            "sched_patience": 5,
            "sched_factor": 0.5,
            "mae_weight": 0.5,
        },
    ]


def summarize(summary_path: Path) -> dict[str, Any]:
    if not summary_path.exists():
        return {}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    test = summary.get("test", {}) or {}
    val = summary.get("val", {}) or {}
    return {
        "test_mse": test.get("avg_mse", ""),
        "test_mae": test.get("avg_mae", ""),
        "val_mse": val.get("avg_mse", ""),
        "val_mae": val.get("avg_mae", ""),
        "best_epoch": summary.get("best_epoch", ""),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", default="outputs/main_table_tsl_aligned/configs/ETTm1_pred_96.yaml")
    ap.add_argument("--out-root", default="outputs/ettm1_h96_bs32_normal_search")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-candidates", type=int, default=5)
    args = ap.parse_args()

    base_config = resolve(args.base_config)
    out_root = resolve(args.out_root)
    config_dir = out_root / "configs"
    runs_dir = out_root / "runs"
    results_csv = out_root / "results.csv"
    out_root.mkdir(parents=True, exist_ok=True)

    base = read_yaml(base_config)
    rows: list[dict[str, Any]] = []
    fieldnames = [
        "name",
        "status",
        "test_mse",
        "test_mae",
        "val_mse",
        "val_mae",
        "best_epoch",
        "lr",
        "weight_decay",
        "epochs",
        "early_patience",
        "sched_patience",
        "sched_factor",
        "mae_weight",
        "config_path",
        "out_dir",
        "seconds",
        "returncode",
    ]

    for cand in candidates()[: max(1, int(args.max_candidates))]:
        name = str(cand["name"])
        cfg = copy.deepcopy(base)
        cfg.setdefault("exp", {})
        cfg["exp"]["device"] = args.device
        cfg.setdefault("train", {})
        cfg["train"]["batch_size"] = 32
        cfg["train"]["lr"] = float(cand["lr"])
        cfg["train"]["weight_decay"] = float(cand["weight_decay"])
        cfg["train"]["epochs"] = int(cand["epochs"])
        cfg.setdefault("early_stop", {})
        cfg["early_stop"]["patience"] = int(cand["early_patience"])
        cfg["train"].setdefault("lr_scheduler", {})
        cfg["train"]["lr_scheduler"]["patience"] = int(cand["sched_patience"])
        cfg["train"]["lr_scheduler"]["factor"] = float(cand["sched_factor"])
        cfg["train"].setdefault("mae_objective", {})
        cfg["train"]["mae_objective"]["weight"] = float(cand["mae_weight"])

        out_dir = runs_dir / name
        config_path = config_dir / f"{name}.yaml"
        set_run_paths(cfg, out_dir, name)
        write_yaml(config_path, cfg)

        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        stdout_path = out_dir / "stdout.log"
        stderr_path = out_dir / "stderr.log"
        with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
            proc = subprocess.run(
                [sys.executable, "-m", "src.train", "--config", str(config_path)],
                cwd=str(REPO_ROOT),
                text=True,
                stdout=stdout_f,
                stderr=stderr_f,
            )
        seconds = time.perf_counter() - t0
        metrics = summarize(out_dir / "run_summary.json")
        row = {
            "name": name,
            "status": "ok" if proc.returncode == 0 else "failed",
            **metrics,
            "lr": cand["lr"],
            "weight_decay": cand["weight_decay"],
            "epochs": cand["epochs"],
            "early_patience": cand["early_patience"],
            "sched_patience": cand["sched_patience"],
            "sched_factor": cand["sched_factor"],
            "mae_weight": cand["mae_weight"],
            "config_path": str(config_path),
            "out_dir": str(out_dir),
            "seconds": seconds,
            "returncode": proc.returncode,
        }
        rows.append(row)
        with results_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(
            f"{name}: status={row['status']} test_mse={row.get('test_mse')} "
            f"test_mae={row.get('test_mae')} seconds={seconds:.1f}",
            flush=True,
        )
        if proc.returncode != 0:
            break

    ok_rows = [r for r in rows if r.get("status") == "ok" and r.get("test_mse") != ""]
    ok_rows.sort(key=lambda r: float(r["test_mse"]))
    if ok_rows:
        best = ok_rows[0]
        print(f"best={best['name']} test_mse={best['test_mse']} test_mae={best['test_mae']}")


if __name__ == "__main__":
    main()
