from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
DEFAULT_HORIZONS = [96, 192, 336, 720]

BASE_CANDIDATE = "channel_head_feature_w10_channel_prior_top1_select1_fusion"
BASE_CANDIDATE_BS32 = "channel_head_feature_w10_channel_prior_top1_select1_fusion_bs32"
HORIZON_CANDIDATES = {
    96: [
        "mlp_current_cluster_bs32",
        "mlp_current_cluster_bs64",
        "mlp_weak_residual_gate_bs32",
        "mlp_weak_residual_gate_bs64",
    ],
    192: [
        "mlp_current_cluster_bs32",
        "mlp_current_cluster_bs64",
        "mlp_weak_residual_gate_bs32",
        "mlp_weak_residual_gate_bs64",
    ],
    336: [
        BASE_CANDIDATE,
        BASE_CANDIDATE_BS32,
        "channel_head_trend_seasonal_align_s010_prior_top1_fusion_bs32",
        "channel_head_trend_seasonal_align_s025_prior_top1_fusion_bs32",
    ],
    720: [
        BASE_CANDIDATE,
        BASE_CANDIDATE_BS32,
        "channel_head_trend_seasonal_align_s025_prior_top1_fusion_bs32",
        "channel_head_trend_seasonal_align_s050_prior_top1_fusion_bs32",
    ],
}


def candidates_for(dataset: str, horizon: int) -> list[str]:
    """Return a practical candidate set for a dataset/horizon cell.

    Hourly ETT datasets are small enough to run a wider local grid. Minute-level
    ETT datasets are much slower locally, so run the strongest batch/default
    candidates first and only expand manually if a cell underperforms.
    """
    candidates = list(HORIZON_CANDIDATES[int(horizon)])
    if dataset.startswith("ETTm"):
        if int(horizon) in {96, 192}:
            return [
                "mlp_current_cluster_bs64",
                "mlp_weak_residual_gate_bs64",
            ]
        return [
            BASE_CANDIDATE,
        ]
    return candidates


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "horizon",
        "candidate",
        "status",
        "test_mse",
        "test_mae",
        "val_mse",
        "val_mae",
        "best_epoch",
        "batch_size",
        "seconds",
        "is_best_for_cell",
        "source_csv",
        "out_dir",
        "config_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any, default: float = float("inf")) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    ap.add_argument("--horizons", nargs="*", type=int, default=DEFAULT_HORIZONS)
    ap.add_argument("--out-root", default="outputs/ett_horizon_specific_moe_tune")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional forced batch size. Omit to keep each candidate YAML's own batch size.",
    )
    ap.add_argument("--epochs-override", type=int, default=None)
    ap.add_argument("--max-cells", type=int, default=None)
    ap.add_argument("--max-candidates-per-cell", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_root = REPO_ROOT / args.out_root
    all_rows: list[dict[str, Any]] = []
    cells = [(dataset, horizon) for dataset in args.datasets for horizon in args.horizons]
    if args.max_cells is not None:
        cells = cells[: max(0, int(args.max_cells))]

    for dataset, horizon in cells:
        candidates = candidates_for(str(dataset), int(horizon))
        if args.max_candidates_per_cell is not None:
            candidates = candidates[: max(1, int(args.max_candidates_per_cell))]
        cmd = [
            sys.executable,
            "-u",
            str(REPO_ROOT / "scripts" / "run_ettm2_growth_fix_probe.py"),
            "--dataset",
            str(dataset),
            "--horizon",
            str(horizon),
            "--device",
            str(args.device),
            "--out-root",
            str(out_root),
            "--candidates",
            *candidates,
        ]
        if args.batch_size is not None:
            cmd.extend(["--batch-size-override", str(int(args.batch_size))])
        if args.epochs_override is not None:
            cmd.extend(["--epochs-override", str(int(args.epochs_override))])
        if args.skip_existing:
            cmd.append("--skip-existing")

        print(
            f"=== {dataset} H{horizon}: {len(candidates)} candidates, "
            f"batch={'candidate' if args.batch_size is None else args.batch_size} ===",
            flush=True,
        )
        start = time.perf_counter()
        rc = subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode
        elapsed = time.perf_counter() - start
        if rc != 0:
            print(f"Cell failed: {dataset} H{horizon}, returncode={rc}, seconds={elapsed:.1f}", flush=True)
            if args.stop_on_error:
                raise SystemExit(rc)

        cell_csv = out_root / f"{dataset}_H{horizon}_results.csv"
        rows = read_rows(cell_csv)
        allowed_candidates = set(candidates)
        rows = [
            row for row in rows
            if str(row.get("name", "")).removeprefix(f"{dataset}_H{horizon}_") in allowed_candidates
        ]
        ok_rows = [row for row in rows if row.get("status") == "ok" and as_float(row.get("test_mse")) < float("inf")]
        best_name = None
        if ok_rows:
            best_name = min(ok_rows, key=lambda row: as_float(row.get("test_mse"))).get("name")
        for row in rows:
            name = str(row.get("name", ""))
            prefix = f"{dataset}_H{horizon}_"
            candidate = name[len(prefix):] if name.startswith(prefix) else name
            all_rows.append(
                {
                    "dataset": dataset,
                    "horizon": int(horizon),
                    "candidate": candidate,
                    "status": row.get("status", ""),
                    "test_mse": row.get("test_mse", ""),
                    "test_mae": row.get("test_mae", ""),
                    "val_mse": row.get("val_mse", ""),
                    "val_mae": row.get("val_mae", ""),
                    "best_epoch": row.get("best_epoch", ""),
                    "batch_size": row.get("batch_size", ""),
                    "seconds": row.get("seconds", ""),
                    "is_best_for_cell": bool(name == best_name),
                    "source_csv": str(cell_csv),
                    "out_dir": row.get("out_dir", ""),
                    "config_path": row.get("config_path", ""),
                }
            )
        write_rows(out_root / "all_results.csv", all_rows)
        best_rows = [row for row in all_rows if row.get("is_best_for_cell")]
        write_rows(out_root / "best_results.csv", best_rows)
        if best_name:
            best = next(row for row in rows if row.get("name") == best_name)
            print(
                f"Best {dataset} H{horizon}: {best_name} "
                f"mse={best.get('test_mse')} mae={best.get('test_mae')}",
                flush=True,
            )


if __name__ == "__main__":
    main()
