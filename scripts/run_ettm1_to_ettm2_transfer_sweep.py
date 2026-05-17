from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]

FIELDS = [
    "status",
    "name",
    "source",
    "corr_mode",
    "route_fit_scope",
    "use_pred_residual",
    "corr_align",
    "corr_max_lag",
    "phase_bins",
    "period_min_hours",
    "period_max_hours",
    "phase_max_shift",
    "corr_threshold",
    "fallback_mode",
    "fallback_topk",
    "mse",
    "mae",
    "hu_fl_mse",
    "hu_fl_mae",
    "route_counts",
    "corr_mean",
    "corr_min",
    "out_dir",
    "config",
    "error",
]


SOURCE_REFS = {
    "thr0p7": {
        "memory_path": "outputs/ettm1_h96_transfer_no_leak/source/cluster_memory.pt",
        "checkpoint_path": "outputs/ettm1_val_refinement_base/runs/ETTm1/pred_96/best_checkpoint.pt",
        "summary_path": "outputs/ettm1_val_refinement_base/runs/ETTm1/pred_96/run_summary.json",
    },
    "thr0p5": {
        "memory_path": "outputs/ettm1_threshold_transfer_to_ettm2/runs/thr_0p5/source/cluster_memory.pt",
        "checkpoint_path": "outputs/ettm1_threshold_transfer_to_ettm2/runs/thr_0p5/source/best_checkpoint.pt",
        "summary_path": "outputs/ettm1_threshold_transfer_to_ettm2/runs/thr_0p5/source/run_summary.json",
    },
}


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def command_python(args: argparse.Namespace) -> list[str]:
    if args.python:
        return [str(args.python)]
    return [sys.executable]


def route_stats(path: Path) -> tuple[str, float | None, float | None]:
    counts: dict[int, int] = {}
    vals: list[float] = []
    if not path.exists():
        return "", None, None
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = int(float(row["cluster_id"]))
            counts[cid] = counts.get(cid, 0) + 1
            if row.get("corr_max", "") != "":
                vals.append(float(row["corr_max"]))
    counts = dict(sorted(counts.items(), key=lambda kv: kv[0]))
    return (
        json.dumps(counts, ensure_ascii=False),
        sum(vals) / len(vals) if vals else None,
        min(vals) if vals else None,
    )


def hufl_metrics(path: Path) -> tuple[float | None, float | None]:
    if not path.exists():
        return None, None
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("channel") == "HUFL":
                return float(row["MSE"]), float(row["MAE"])
    return None, None


def make_variants() -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    # Full residual route-only variants.
    for align in ["head", "tail"]:
        for min_h, max_h in [(12, 168), (24, 24), (24, 168), (12, 48), (48, 168)]:
            variants.append(
                {
                    "source": "thr0p7",
                    "name": f"cycle_res_{align}_{min_h}_{max_h}",
                    "corr_mode": "cycle_template",
                    "use_pred_residual": True,
                    "corr_align": align,
                    "phase_bins": 64,
                    "period_min_hours": min_h,
                    "period_max_hours": max_h,
                    "phase_max_shift": None,
                    "corr_max_lag": 0,
                    "route_fit_scope": "train",
                }
            )
    for bins in [24, 48, 96, 128]:
        variants.append(
            {
                "source": "thr0p7",
                "name": f"cycle_res_bins{bins}",
                "corr_mode": "cycle_template",
                "use_pred_residual": True,
                "corr_align": "head",
                "phase_bins": bins,
                "period_min_hours": 12,
                "period_max_hours": 168,
                "phase_max_shift": None,
                "corr_max_lag": 0,
                "route_fit_scope": "train",
            }
        )
    for lag in [0, 24, 48, 96]:
        variants.append(
            {
                "source": "thr0p7",
                "name": f"pearson_res_lag{lag}",
                "corr_mode": "pearson",
                "use_pred_residual": True,
                "corr_align": "head",
                "phase_bins": 64,
                "period_min_hours": None,
                "period_max_hours": None,
                "phase_max_shift": None,
                "corr_max_lag": lag,
                "route_fit_scope": "train",
            }
        )
    # Base-only variants, including soft fallback, to diagnose residual transfer.
    for corr_mode in ["cycle_template", "pearson"]:
        for soft in [False, True]:
            variants.append(
                {
                    "source": "thr0p7",
                    "name": f"{corr_mode}_base_{'soft' if soft else 'hard'}",
                    "corr_mode": corr_mode,
                    "use_pred_residual": False,
                    "corr_align": "head",
                    "phase_bins": 64,
                    "period_min_hours": 12 if corr_mode == "cycle_template" else None,
                    "period_max_hours": 168 if corr_mode == "cycle_template" else None,
                    "phase_max_shift": None,
                    "corr_max_lag": 0,
                    "corr_threshold": 0.75 if soft else None,
                    "fallback_mode": "soft" if soft else "hard",
                    "fallback_topk": 2,
                    "route_fit_scope": "train",
                }
            )
    # Known threshold source comparison.
    variants.append(
        {
            "source": "thr0p5",
            "name": "thr0p5_cycle_res_default",
            "corr_mode": "cycle_template",
            "use_pred_residual": True,
            "corr_align": "head",
            "phase_bins": 64,
            "period_min_hours": 12,
            "period_max_hours": 168,
            "phase_max_shift": None,
            "corr_max_lag": 0,
            "route_fit_scope": "train",
        }
    )
    return variants


def patch_cfg(base: dict[str, Any], variant: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    source_ref = SOURCE_REFS[variant["source"]]
    cfg["exp"]["name"] = f"ETTm1_to_ETTm2_{variant['name']}"
    cfg["exp"]["out_dir"] = str(out_dir)
    cfg["source"].update(source_ref)
    transfer = cfg.setdefault("transfer", {})
    for key in [
        "corr_mode",
        "route_fit_scope",
        "use_pred_residual",
        "corr_align",
        "corr_max_lag",
        "phase_bins",
        "phase_max_shift",
        "period_min_hours",
        "period_max_hours",
        "corr_threshold",
        "fallback_mode",
        "fallback_topk",
    ]:
        if key in variant:
            transfer[key] = variant[key]
    transfer["save_corr"] = True
    transfer.setdefault("knn_hybrid", {})["enable"] = False
    transfer.setdefault("resample", {})["enable"] = False
    cfg["normalize"]["train_only"] = True
    return cfg


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", type=Path, default=ROOT / "configs" / "ETTm1ToETTm2.yaml")
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "ettm1_to_ettm2_transfer_sweep")
    ap.add_argument("--python", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    base = read_yaml(args.base_config)
    py = command_python(args)
    rows: list[dict[str, Any]] = []
    results_path = args.out_root / "transfer_sweep.csv"
    if results_path.exists() and not args.force:
        with results_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    done = {row.get("name") for row in rows if row.get("status") == "ok"}

    for variant in make_variants():
        if variant["name"] in done and not args.force:
            continue
        out_dir = args.out_root / "runs" / variant["name"]
        cfg_path = args.out_root / "configs" / f"{variant['name']}.yaml"
        row = {
            "status": "pending",
            "name": variant["name"],
            "source": variant["source"],
            "corr_mode": variant.get("corr_mode"),
            "route_fit_scope": variant.get("route_fit_scope", "train"),
            "use_pred_residual": variant.get("use_pred_residual", True),
            "corr_align": variant.get("corr_align", "head"),
            "corr_max_lag": variant.get("corr_max_lag", 0),
            "phase_bins": variant.get("phase_bins"),
            "period_min_hours": variant.get("period_min_hours"),
            "period_max_hours": variant.get("period_max_hours"),
            "phase_max_shift": variant.get("phase_max_shift"),
            "corr_threshold": variant.get("corr_threshold"),
            "fallback_mode": variant.get("fallback_mode", "hard"),
            "fallback_topk": variant.get("fallback_topk", 2),
            "out_dir": str(out_dir),
            "config": str(cfg_path),
            "error": "",
        }
        try:
            cfg = patch_cfg(base, variant, out_dir)
            write_yaml(cfg_path, cfg)
            print(f"[transfer] {variant['name']}")
            summary_path = out_dir / "transfer_summary.json"
            if args.force or not summary_path.exists():
                proc = subprocess.run(py + ["-m", "src.transfer", "--config", str(cfg_path)], cwd=str(ROOT))
                if proc.returncode != 0:
                    raise RuntimeError(f"src.transfer returncode={proc.returncode}")
            summary = read_json(summary_path)
            route_counts, corr_mean, corr_min = route_stats(out_dir / "cluster_assignment.csv")
            hufl_mse, hufl_mae = hufl_metrics(out_dir / "test_metrics.csv")
            row.update(
                {
                    "status": "ok",
                    "mse": summary.get("avg_mse"),
                    "mae": summary.get("avg_mae"),
                    "hu_fl_mse": hufl_mse,
                    "hu_fl_mae": hufl_mae,
                    "route_counts": route_counts,
                    "corr_mean": corr_mean,
                    "corr_min": corr_min,
                }
            )
        except Exception as exc:
            row["status"] = "error"
            row["error"] = repr(exc)
        rows = [r for r in rows if r.get("name") != variant["name"]]
        rows.append(row)
        write_rows(results_path, rows)

    rows_ok = [r for r in rows if r.get("status") == "ok" and r.get("mse") not in {None, ""}]
    rows_ok.sort(key=lambda r: float(r["mse"]))
    write_rows(args.out_root / "transfer_sweep_ranked.csv", rows_ok)
    if rows_ok:
        best = rows_ok[0]
        print(f"Best: {best['name']} mse={float(best['mse']):.6f} mae={float(best['mae']):.6f}")


if __name__ == "__main__":
    main()
