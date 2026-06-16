from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


FIELDS = [
    "status",
    "label",
    "shrink",
    "base_config",
    "config_path",
    "out_dir",
    "baseline_mse",
    "baseline_mae",
    "test_mse",
    "test_mae",
    "val_mse",
    "val_mae",
    "mse_delta",
    "mae_delta",
    "mse_delta_pct",
    "mae_delta_pct",
    "returncode",
    "total_sec",
    "error",
]


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=False, sort_keys=False)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def as_float(value: Any) -> float | None:
    try:
        if value in {"", None}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def shrink_tag(shrink: float) -> str:
    return f"s{str(float(shrink)).replace('.', 'p').replace('-', 'm')}"


def localize_paths(cfg: dict[str, Any], out_dir: Path) -> None:
    cfg.setdefault("exp", {})["out_dir"] = str(out_dir)
    cfg.setdefault("corr", {})["save_path"] = str(out_dir / "corr.npy")
    cfg.setdefault("portrait", {})["enable"] = False
    cfg["portrait"]["out_dir"] = str(out_dir / "cluster_portraits")
    cfg.setdefault("knn_hybrid", {})["path"] = str(out_dir / "knn_shape_bank.pt")
    cfg.setdefault("memory", {})["enable"] = False
    cfg["memory"]["save_checkpoint"] = False
    cfg["memory"]["path"] = str(out_dir / "cluster_memory.pt")
    cfg["memory"]["checkpoint_path"] = str(out_dir / "best_checkpoint.pt")
    cfg.setdefault("plot", {})["enable"] = False


def baseline_from_config(base_cfg: dict[str, Any]) -> tuple[float | None, float | None]:
    out_dir = resolve(str((base_cfg.get("exp", {}) or {}).get("out_dir", "")))
    summary = load_json(out_dir / "run_summary.json")
    test = summary.get("test", {}) or {}
    selected = summary.get("selected", {}) or {}
    mse = as_float(selected.get("avg_mse", test.get("avg_mse")))
    mae = as_float(selected.get("avg_mae", test.get("avg_mae")))
    return mse, mae


def build_config(
    base_cfg: dict[str, Any],
    *,
    label: str,
    shrink: float,
    out_root: Path,
    epochs_override: int | None,
    batch_size_override: int | None,
    device: str,
) -> tuple[dict[str, Any], Path, Path]:
    cfg = dict(base_cfg)
    cfg = yaml.safe_load(yaml.safe_dump(cfg, allow_unicode=False, sort_keys=False)) or {}
    tag = shrink_tag(shrink)
    out_dir = out_root / "runs" / label / tag
    config_path = out_root / "configs" / label / f"{tag}.yaml"
    cfg.setdefault("exp", {})["name"] = f"{label}_calibration_{tag}"
    cfg["exp"]["device"] = device
    localize_paths(cfg, out_dir)
    cfg["calibration"] = {
        "enable": True,
        "method": "median",
        "shrink": float(shrink),
        "max_abs": float((base_cfg.get("calibration", {}) or {}).get("max_abs", 0.0) or 0.0),
    }
    if epochs_override is not None:
        cfg.setdefault("train", {})["epochs"] = int(epochs_override)
    if batch_size_override is not None:
        cfg.setdefault("train", {})["batch_size"] = int(batch_size_override)
    return cfg, config_path, out_dir


def run_one(config_path: Path, out_dir: Path, reuse_existing: bool) -> tuple[int, float, str]:
    if reuse_existing and (out_dir / "run_summary.json").exists():
        return 0, 0.0, ""
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = out_dir / "stdout.log"
    stderr_path = out_dir / "stderr.log"
    cmd = [sys.executable, "-u", "-m", "src.train", "--config", str(config_path)]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    start = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
        completed = subprocess.run(cmd, cwd=str(ROOT), stdout=stdout_f, stderr=stderr_f, text=True, env=env)
    total_sec = time.perf_counter() - start
    error = ""
    if completed.returncode != 0:
        error = stderr_path.read_text(encoding="utf-8", errors="replace")[-3000:]
    return int(completed.returncode), total_sec, error


def row_from_summary(
    *,
    label: str,
    shrink: float,
    base_config: Path,
    config_path: Path,
    out_dir: Path,
    baseline_mse: float | None,
    baseline_mae: float | None,
    returncode: int,
    total_sec: float,
    error: str,
) -> dict[str, Any]:
    summary = load_json(out_dir / "run_summary.json")
    test = summary.get("test", {}) or {}
    val = summary.get("val", {}) or {}
    selected = summary.get("selected", {}) or {}
    test_mse = as_float(selected.get("avg_mse", test.get("avg_mse")))
    test_mae = as_float(selected.get("avg_mae", test.get("avg_mae")))
    val_mse = as_float(selected.get("base_val_mse", val.get("avg_mse")))
    val_mae = as_float(selected.get("base_val_mae", val.get("avg_mae")))
    status = "ok" if returncode == 0 and test_mse is not None and test_mae is not None else "failed"
    mse_delta = None if baseline_mse is None or test_mse is None else test_mse - baseline_mse
    mae_delta = None if baseline_mae is None or test_mae is None else test_mae - baseline_mae
    return {
        "status": status,
        "label": label,
        "shrink": shrink,
        "base_config": str(base_config),
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "baseline_mse": "" if baseline_mse is None else baseline_mse,
        "baseline_mae": "" if baseline_mae is None else baseline_mae,
        "test_mse": "" if test_mse is None else test_mse,
        "test_mae": "" if test_mae is None else test_mae,
        "val_mse": "" if val_mse is None else val_mse,
        "val_mae": "" if val_mae is None else val_mae,
        "mse_delta": "" if mse_delta is None else mse_delta,
        "mae_delta": "" if mae_delta is None else mae_delta,
        "mse_delta_pct": "" if mse_delta is None or not baseline_mse else mse_delta / baseline_mse * 100.0,
        "mae_delta_pct": "" if mae_delta is None or not baseline_mae else mae_delta / baseline_mae * 100.0,
        "returncode": returncode,
        "total_sec": round(total_sec, 3),
        "error": error,
    }


def build_sweep_config(
    base_cfg: dict[str, Any],
    *,
    label: str,
    shrinks: list[float],
    out_root: Path,
    epochs_override: int | None,
    batch_size_override: int | None,
    device: str,
    per_channel: bool = False,
    per_channel_grid: list[float] | None = None,
    per_channel_max_rel_mse: float = 0.01,
) -> tuple[dict[str, Any], Path, Path]:
    """One config that trains once and evaluates every shrink in-process."""
    cfg = yaml.safe_load(yaml.safe_dump(dict(base_cfg), allow_unicode=False, sort_keys=False)) or {}
    tag = "sweep"
    out_dir = out_root / "runs" / label / tag
    config_path = out_root / "configs" / label / f"{tag}.yaml"
    cfg.setdefault("exp", {})["name"] = f"{label}_calibration_{tag}"
    cfg["exp"]["device"] = device
    localize_paths(cfg, out_dir)
    cfg["calibration"] = {
        "enable": True,
        "method": "median",
        "shrink": float(shrinks[0]),
        "max_abs": float((base_cfg.get("calibration", {}) or {}).get("max_abs", 0.0) or 0.0),
        "shrink_sweep": [float(s) for s in shrinks],
    }
    if per_channel:
        cfg["calibration"]["per_channel_shrink"] = {
            "enable": True,
            "grid": [float(g) for g in (per_channel_grid or [0.0, 0.3, 0.5, 0.7, 0.85, 1.0])],
            "max_rel_mse_regression": float(per_channel_max_rel_mse),
        }
    if epochs_override is not None:
        cfg.setdefault("train", {})["epochs"] = int(epochs_override)
    if batch_size_override is not None:
        cfg.setdefault("train", {})["batch_size"] = int(batch_size_override)
    return cfg, config_path, out_dir


def rows_from_sweep_summary(
    *,
    label: str,
    base_config: Path,
    config_path: Path,
    out_dir: Path,
    baseline_mse: float | None,
    baseline_mae: float | None,
    returncode: int,
    total_sec: float,
    error: str,
) -> list[dict[str, Any]]:
    summary = load_json(out_dir / "run_summary.json")
    sweep = ((summary.get("calibration", {}) or {}).get("shrink_sweep", []) or [])
    rows: list[dict[str, Any]] = []
    for entry in sweep:
        test_mse = as_float(entry.get("test_mse"))
        test_mae = as_float(entry.get("test_mae"))
        mse_delta = None if baseline_mse is None or test_mse is None else test_mse - baseline_mse
        mae_delta = None if baseline_mae is None or test_mae is None else test_mae - baseline_mae
        rows.append({
            "status": "ok" if returncode == 0 and test_mse is not None else "failed",
            "label": label,
            "shrink": as_float(entry.get("shrink")),
            "base_config": str(base_config),
            "config_path": str(config_path),
            "out_dir": str(out_dir),
            "baseline_mse": "" if baseline_mse is None else baseline_mse,
            "baseline_mae": "" if baseline_mae is None else baseline_mae,
            "test_mse": "" if test_mse is None else test_mse,
            "test_mae": "" if test_mae is None else test_mae,
            "val_mse": entry.get("val_mse", ""),
            "val_mae": entry.get("val_mae", ""),
            "mse_delta": "" if mse_delta is None else mse_delta,
            "mae_delta": "" if mae_delta is None else mae_delta,
            "mse_delta_pct": "" if mse_delta is None or not baseline_mse else mse_delta / baseline_mse * 100.0,
            "mae_delta_pct": "" if mae_delta is None or not baseline_mae else mae_delta / baseline_mae * 100.0,
            "returncode": returncode,
            "total_sec": round(total_sec, 3),
            "error": error,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe median calibration shrink values for an existing config.")
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-root", default="outputs/calibration_shrink_probe_20260616")
    ap.add_argument("--shrinks", nargs="*", type=float, default=[0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs-override", type=int, default=None)
    ap.add_argument("--batch-size-override", type=int, default=None)
    ap.add_argument("--reuse-existing", action="store_true")
    ap.add_argument(
        "--single-run",
        action="store_true",
        help="Train once and evaluate all shrinks in-process via calibration.shrink_sweep (no retraining).",
    )
    ap.add_argument(
        "--per-channel",
        action="store_true",
        help="Also evaluate per-channel guarded shrink (each channel picks its own shrink under an MSE-regression cap).",
    )
    ap.add_argument("--per-channel-grid", nargs="*", type=float, default=[0.0, 0.3, 0.5, 0.7, 0.85, 1.0])
    ap.add_argument("--per-channel-max-rel-mse", type=float, default=0.01)
    args = ap.parse_args()

    base_config = resolve(args.base_config)
    out_root = resolve(args.out_root)
    base_cfg = load_yaml(base_config)
    baseline_mse, baseline_mae = baseline_from_config(base_cfg)
    results_path = out_root / "results.csv"
    rows = read_rows(results_path)
    by_key = {(row.get("label", ""), str(row.get("shrink", ""))): row for row in rows}

    if args.single_run:
        cfg, config_path, out_dir = build_sweep_config(
            base_cfg,
            label=str(args.label),
            shrinks=[float(s) for s in args.shrinks],
            out_root=out_root,
            epochs_override=args.epochs_override,
            batch_size_override=args.batch_size_override,
            device=str(args.device),
            per_channel=bool(args.per_channel),
            per_channel_grid=[float(g) for g in args.per_channel_grid],
            per_channel_max_rel_mse=float(args.per_channel_max_rel_mse),
        )
        write_yaml(config_path, cfg)
        returncode, total_sec, error = run_one(config_path, out_dir, bool(args.reuse_existing))
        sweep_rows = rows_from_sweep_summary(
            label=str(args.label),
            base_config=base_config,
            config_path=config_path,
            out_dir=out_dir,
            baseline_mse=baseline_mse,
            baseline_mae=baseline_mae,
            returncode=returncode,
            total_sec=total_sec,
            error=error,
        )
        for row in sweep_rows:
            by_key[(str(args.label), str(float(row["shrink"])))] = row
            print(json.dumps({k: row.get(k) for k in ["status", "label", "shrink", "val_mae", "test_mse", "test_mae", "mse_delta_pct", "mae_delta_pct"]}, ensure_ascii=False), flush=True)
        rows = list(by_key.values())
        rows.sort(key=lambda item: (item.get("label", ""), float(item.get("shrink", 0.0) or 0.0)))
        write_rows(results_path, rows)
        print(f"Wrote: {results_path} (single-run, total_sec={round(total_sec, 1)})", flush=True)
        return

    for shrink in args.shrinks:
        cfg, config_path, out_dir = build_config(
            base_cfg,
            label=str(args.label),
            shrink=float(shrink),
            out_root=out_root,
            epochs_override=args.epochs_override,
            batch_size_override=args.batch_size_override,
            device=str(args.device),
        )
        write_yaml(config_path, cfg)
        returncode, total_sec, error = run_one(config_path, out_dir, bool(args.reuse_existing))
        row = row_from_summary(
            label=str(args.label),
            shrink=float(shrink),
            base_config=base_config,
            config_path=config_path,
            out_dir=out_dir,
            baseline_mse=baseline_mse,
            baseline_mae=baseline_mae,
            returncode=returncode,
            total_sec=total_sec,
            error=error,
        )
        by_key[(str(args.label), str(float(shrink)))] = row
        rows = list(by_key.values())
        rows.sort(key=lambda item: (item.get("label", ""), float(item.get("shrink", 0.0) or 0.0)))
        write_rows(results_path, rows)
        print(json.dumps({k: row.get(k) for k in ["status", "label", "shrink", "test_mse", "test_mae", "mse_delta_pct", "mae_delta_pct"]}, ensure_ascii=False), flush=True)

    print(f"Wrote: {results_path}", flush=True)


if __name__ == "__main__":
    main()
