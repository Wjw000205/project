from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from run_cmapss_csv_tune_transfer import (
    CANDIDATES_FULL,
    DATASETS,
    OUT_DATA_DIR,
    RAW_DIR,
    ROOT,
    convert_cmapss,
    make_train_cfg,
    make_transfer_cfg,
    read_json,
    route_stats,
    write_csv,
    write_yaml,
)


PAIRS = [
    ("FD003", "FD002", "pearson"),
    ("FD004", "FD003", "pearson"),
    ("FD004", "FD003", "cycle_template"),
]

BEST_CANDIDATE = {
    "FD001": "range_trend_thr0p5_h128",
    "FD002": "shape_thr0p7_h128",
    "FD003": "shape_thr0p7_h128",
    "FD004": "shape_thr0p3_h256",
}

DETAIL_FIELDS = [
    "status",
    "seed",
    "source",
    "target",
    "corr_mode",
    "candidate",
    "multi_source_test_mse",
    "single_source_test_mse",
    "multi_transfer_mse",
    "single_transfer_mse",
    "single_minus_multi_mse",
    "multi_transfer_mae",
    "single_transfer_mae",
    "multi_route_counts",
    "single_route_counts",
    "multi_corr_mean",
    "single_corr_mean",
    "multi_source_out_dir",
    "single_source_out_dir",
    "multi_transfer_out_dir",
    "single_transfer_out_dir",
    "error",
]

SUMMARY_FIELDS = [
    "source",
    "target",
    "corr_mode",
    "n",
    "mean_single_minus_multi_mse",
    "std_single_minus_multi_mse",
    "win_rate_multi",
    "mean_multi_transfer_mse",
    "mean_single_transfer_mse",
    "mean_multi_transfer_mae",
    "mean_single_transfer_mae",
    "seeds",
]


def candidate_by_name(name: str) -> dict[str, Any]:
    for cand in CANDIDATES_FULL:
        if cand["name"] == name:
            return dict(cand)
    raise KeyError(name)


def command_python(args: argparse.Namespace) -> list[str]:
    if args.python:
        return [str(args.python)]
    return [sys.executable]


def run_cmd(cmd: list[str], *, cwd: Path, reuse_path: Path | None = None) -> tuple[int, float]:
    if reuse_path is not None and reuse_path.exists():
        return 0, 0.0
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(cwd))
    return int(proc.returncode), float(time.perf_counter() - start)


def patch_seed(cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    cfg["exp"]["seed"] = int(seed)
    if "cluster" in cfg:
        cfg["cluster"]["random_state"] = int(seed)
    return cfg


def test_mse(summary: dict[str, Any]) -> float:
    return float(summary.get("test", {}).get("avg_mse"))


def test_mae(summary: dict[str, Any]) -> float:
    return float(summary.get("test", {}).get("avg_mae"))


def train_source(
    *,
    args: argparse.Namespace,
    py: list[str],
    manifest: dict[str, Any],
    dataset: str,
    seed: int,
    single: bool,
) -> Path:
    candidate = candidate_by_name(BEST_CANDIDATE[dataset])
    if single:
        candidate["name"] = f"single_{candidate['name']}"
        candidate["distance_threshold"] = 2.0
        source_kind = "single"
    else:
        source_kind = "multi"
    out_dir = (
        args.out_root
        / "runs"
        / f"seed_{seed}"
        / source_kind
        / dataset
        / candidate["name"]
    )
    cfg_path = (
        args.out_root
        / "configs"
        / f"seed_{seed}"
        / source_kind
        / dataset
        / f"{candidate['name']}.yaml"
    )
    cfg = make_train_cfg(
        dataset=dataset,
        series_csv=Path(manifest["series_csvs"][dataset]),
        out_dir=out_dir,
        candidate=candidate,
        input_len=args.input_len,
        pred_len=args.pred_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
    patch_seed(cfg, seed)
    write_yaml(cfg_path, cfg)
    print(f"[train] seed={seed} {source_kind} {dataset} {candidate['name']}")
    code, _ = run_cmd(
        py + ["-m", "src.train", "--config", str(cfg_path)],
        cwd=ROOT,
        reuse_path=None if args.force else out_dir / "run_summary.json",
    )
    if code != 0:
        raise RuntimeError(f"train failed: seed={seed} {source_kind} {dataset} returncode={code}")
    return out_dir


def run_transfer(
    *,
    args: argparse.Namespace,
    py: list[str],
    manifest: dict[str, Any],
    source: str,
    target: str,
    corr_mode: str,
    seed: int,
    source_run: Path,
    single: bool,
) -> Path:
    source_kind = "single" if single else "multi"
    out_dir = (
        args.out_root
        / "runs"
        / f"seed_{seed}"
        / f"{source_kind}_transfer"
        / corr_mode
        / f"{source}_to_{target}"
    )
    cfg_path = (
        args.out_root
        / "configs"
        / f"seed_{seed}"
        / f"{source_kind}_transfer"
        / corr_mode
        / f"{source}_to_{target}.yaml"
    )
    cfg = make_transfer_cfg(
        source=source,
        target=target,
        source_run=source_run,
        source_csv=Path(manifest["series_csvs"][source]),
        target_csv=Path(manifest["series_csvs"][target]),
        out_dir=out_dir,
        input_len=args.input_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        device=args.device,
        corr_mode=corr_mode,
        period_min=50,
        period_max=350,
    )
    patch_seed(cfg, seed)
    write_yaml(cfg_path, cfg)
    print(f"[transfer] seed={seed} {source_kind} {corr_mode} {source}->{target}")
    code, _ = run_cmd(
        py + ["-m", "src.transfer", "--config", str(cfg_path)],
        cwd=ROOT,
        reuse_path=None if args.force else out_dir / "transfer_summary.json",
    )
    if code != 0:
        raise RuntimeError(
            f"transfer failed: seed={seed} {source_kind} {corr_mode} {source}->{target} returncode={code}"
        )
    return out_dir


def read_existing(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok = [r for r in rows if r.get("status") == "ok"]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in ok:
        key = (str(row["source"]), str(row["target"]), str(row["corr_mode"]))
        grouped.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (source, target, corr_mode), vals in sorted(grouped.items()):
        gains = [float(r["single_minus_multi_mse"]) for r in vals]
        multi_mse = [float(r["multi_transfer_mse"]) for r in vals]
        single_mse = [float(r["single_transfer_mse"]) for r in vals]
        multi_mae = [float(r["multi_transfer_mae"]) for r in vals]
        single_mae = [float(r["single_transfer_mae"]) for r in vals]
        out.append(
            {
                "source": source,
                "target": target,
                "corr_mode": corr_mode,
                "n": len(vals),
                "mean_single_minus_multi_mse": mean(gains),
                "std_single_minus_multi_mse": pstdev(gains) if len(gains) > 1 else 0.0,
                "win_rate_multi": sum(1 for g in gains if g > 0.0) / len(gains),
                "mean_multi_transfer_mse": mean(multi_mse),
                "mean_single_transfer_mse": mean(single_mse),
                "mean_multi_transfer_mae": mean(multi_mae),
                "mean_single_transfer_mae": mean(single_mae),
                "seeds": ",".join(str(r["seed"]) for r in vals),
            }
        )
    out.sort(key=lambda r: float(r["mean_single_minus_multi_mse"]), reverse=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "multicluster_seed_check")
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--data-out", type=Path, default=OUT_DATA_DIR)
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    ap.add_argument("--pairs", nargs="+", default=[":".join(p) for p in PAIRS])
    ap.add_argument("--input-len", type=int, default=96)
    ap.add_argument("--pred-len", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--python", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    manifest = convert_cmapss(args.raw_dir, args.data_out, force=False)
    py = command_python(args)
    detail_csv = args.out_root / "seed_detail.csv"
    detail_rows: list[dict[str, Any]] = read_existing(detail_csv)
    completed = {
        (str(r.get("seed")), r.get("source"), r.get("target"), r.get("corr_mode"))
        for r in detail_rows
        if r.get("status") == "ok"
    }
    parsed_pairs: list[tuple[str, str, str]] = []
    for item in args.pairs:
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"Pair must be SOURCE:TARGET:CORR_MODE, got {item}")
        source, target, corr_mode = parts
        if source not in DATASETS or target not in DATASETS:
            raise ValueError(f"Unknown dataset in pair {item}")
        parsed_pairs.append((source, target, corr_mode))

    needed_sources = sorted({source for source, _, _ in parsed_pairs})
    for seed in args.seeds:
        source_runs: dict[tuple[str, bool], Path] = {}
        for source in needed_sources:
            source_runs[(source, False)] = train_source(
                args=args,
                py=py,
                manifest=manifest,
                dataset=source,
                seed=seed,
                single=False,
            )
            source_runs[(source, True)] = train_source(
                args=args,
                py=py,
                manifest=manifest,
                dataset=source,
                seed=seed,
                single=True,
            )

        for source, target, corr_mode in parsed_pairs:
            key = (str(seed), source, target, corr_mode)
            if key in completed and not args.force:
                continue
            row: dict[str, Any] = {
                "status": "pending",
                "seed": seed,
                "source": source,
                "target": target,
                "corr_mode": corr_mode,
                "candidate": BEST_CANDIDATE[source],
                "multi_source_out_dir": str(source_runs[(source, False)]),
                "single_source_out_dir": str(source_runs[(source, True)]),
                "error": "",
            }
            try:
                multi_run = source_runs[(source, False)]
                single_run = source_runs[(source, True)]
                multi_transfer = run_transfer(
                    args=args,
                    py=py,
                    manifest=manifest,
                    source=source,
                    target=target,
                    corr_mode=corr_mode,
                    seed=seed,
                    source_run=multi_run,
                    single=False,
                )
                single_transfer = run_transfer(
                    args=args,
                    py=py,
                    manifest=manifest,
                    source=source,
                    target=target,
                    corr_mode=corr_mode,
                    seed=seed,
                    source_run=single_run,
                    single=True,
                )
                multi_source_summary = read_json(multi_run / "run_summary.json")
                single_source_summary = read_json(single_run / "run_summary.json")
                multi_summary = read_json(multi_transfer / "transfer_summary.json")
                single_summary = read_json(single_transfer / "transfer_summary.json")
                _, multi_route_counts, multi_corr_mean, _ = route_stats(multi_transfer / "cluster_assignment.csv")
                _, single_route_counts, single_corr_mean, _ = route_stats(single_transfer / "cluster_assignment.csv")
                multi_mse = float(multi_summary["avg_mse"])
                single_mse = float(single_summary["avg_mse"])
                row.update(
                    {
                        "status": "ok",
                        "multi_source_test_mse": test_mse(multi_source_summary),
                        "single_source_test_mse": test_mse(single_source_summary),
                        "multi_transfer_mse": multi_mse,
                        "single_transfer_mse": single_mse,
                        "single_minus_multi_mse": single_mse - multi_mse,
                        "multi_transfer_mae": float(multi_summary["avg_mae"]),
                        "single_transfer_mae": float(single_summary["avg_mae"]),
                        "multi_route_counts": multi_route_counts,
                        "single_route_counts": single_route_counts,
                        "multi_corr_mean": multi_corr_mean,
                        "single_corr_mean": single_corr_mean,
                        "multi_transfer_out_dir": str(multi_transfer),
                        "single_transfer_out_dir": str(single_transfer),
                    }
                )
            except Exception as exc:
                row["status"] = "error"
                row["error"] = repr(exc)
            detail_rows = [
                r
                for r in detail_rows
                if (str(r.get("seed")), r.get("source"), r.get("target"), r.get("corr_mode")) != key
            ]
            detail_rows.append(row)
            write_csv(detail_csv, detail_rows, DETAIL_FIELDS)
            write_csv(args.out_root / "seed_summary.csv", summarize(detail_rows), SUMMARY_FIELDS)

    write_csv(detail_csv, detail_rows, DETAIL_FIELDS)
    summary = summarize(detail_rows)
    write_csv(args.out_root / "seed_summary.csv", summary, SUMMARY_FIELDS)
    print(f"Saved seed detail to: {detail_csv}")
    print(f"Saved seed summary to: {args.out_root / 'seed_summary.csv'}")
    if summary:
        best = summary[0]
        print(
            "Best mean case: "
            f"{best['source']}->{best['target']} {best['corr_mode']} "
            f"mean_gain={float(best['mean_single_minus_multi_mse']):.6f}, "
            f"win_rate={float(best['win_rate_multi']):.2f}"
        )


if __name__ == "__main__":
    main()
