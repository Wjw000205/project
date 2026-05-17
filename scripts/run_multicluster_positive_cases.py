from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from run_cmapss_csv_tune_transfer import (
    CANDIDATES_COMPACT,
    CANDIDATES_FULL,
    DATASETS,
    OUT_DATA_DIR,
    RAW_DIR,
    ROOT,
    cluster_counts_from_memory,
    convert_cmapss,
    make_train_cfg,
    make_transfer_cfg,
    metric,
    read_json,
    route_stats,
    write_csv,
    write_yaml,
)


FIELDS = [
    "status",
    "source",
    "target",
    "corr_mode",
    "multi_candidate",
    "single_candidate",
    "target_reference_candidate",
    "multi_source_clusters",
    "single_source_clusters",
    "multi_transfer_mse",
    "single_transfer_mse",
    "multi_minus_single_mse",
    "single_minus_multi_mse",
    "multi_transfer_mae",
    "single_transfer_mae",
    "target_reference_mse",
    "target_reference_mae",
    "multi_gain_vs_target",
    "single_gain_vs_target",
    "single_target_route_clusters",
    "single_target_route_counts",
    "single_target_corr_mean",
    "single_target_corr_min",
    "multi_transfer_out_dir",
    "single_source_out_dir",
    "single_transfer_out_dir",
    "error",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def candidate_by_name(name: str) -> dict[str, Any]:
    for cand in CANDIDATES_FULL:
        if cand["name"] == name:
            return dict(cand)
    for cand in CANDIDATES_COMPACT:
        if cand["name"] == name:
            return dict(cand)
    raise KeyError(f"Unknown candidate: {name}")


def best_by_dataset(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv_rows(path)
    return {str(row["dataset"]): row for row in rows if row.get("status") == "ok"}


def transfers_by_pair(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv_rows(path)
    return {
        (str(row["source"]), str(row["target"])): row
        for row in rows
        if row.get("status") == "ok"
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-out", type=Path, default=OUT_DATA_DIR)
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--multi-root", type=Path, default=ROOT / "outputs" / "cmapss_val_search_transfer")
    ap.add_argument("--cycle-root", type=Path, default=ROOT / "outputs" / "cmapss_val_search_transfer_cycle")
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "multicluster_route_positive_cases")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--corr-modes", nargs="+", choices=["pearson", "cycle_template"], default=["pearson", "cycle_template"])
    ap.add_argument("--input-len", type=int, default=96)
    ap.add_argument("--pred-len", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--python", type=Path, default=None)
    ap.add_argument("--force-train", action="store_true")
    ap.add_argument("--force-transfer", action="store_true")
    args = ap.parse_args()

    manifest = convert_cmapss(args.raw_dir, args.data_out, force=False)
    best = best_by_dataset(args.multi_root / "best_by_dataset.csv")
    multi_pearson = transfers_by_pair(args.multi_root / "transfer.csv")
    multi_cycle = transfers_by_pair(args.cycle_root / "transfer.csv")
    py = command_python(args)

    datasets = [d for d in args.datasets if d in DATASETS and d in best]
    cfg_root = args.out_root / "configs"
    run_root = args.out_root / "runs"
    rows: list[dict[str, Any]] = []
    out_csv = args.out_root / "single_vs_multi.csv"
    if out_csv.exists() and not (args.force_train or args.force_transfer):
        rows = read_csv_rows(out_csv)

    completed = {
        (r.get("source"), r.get("target"), r.get("corr_mode"))
        for r in rows
        if r.get("status") == "ok"
    }

    single_source_runs: dict[str, Path] = {}
    single_source_summaries: dict[str, dict[str, Any]] = {}
    single_source_clusters: dict[str, int | None] = {}

    for source in datasets:
        source_best = best[source]
        source_candidate_name = str(source_best["candidate"])
        candidate = candidate_by_name(source_candidate_name)
        candidate["name"] = f"single_{source_candidate_name}"
        candidate["distance_threshold"] = 2.0
        source_csv = Path(manifest["series_csvs"][source])
        out_dir = run_root / "single_source" / source / candidate["name"]
        cfg_path = cfg_root / "single_source" / source / f"{candidate['name']}.yaml"
        cfg = make_train_cfg(
            dataset=source,
            series_csv=source_csv,
            out_dir=out_dir,
            candidate=candidate,
            input_len=args.input_len,
            pred_len=args.pred_len,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
        )
        write_yaml(cfg_path, cfg)
        print(f"[single-source] {source} {candidate['name']}")
        code, _ = run_cmd(
            py + ["-m", "src.train", "--config", str(cfg_path)],
            cwd=ROOT,
            reuse_path=None if args.force_train else out_dir / "run_summary.json",
        )
        if code != 0:
            raise RuntimeError(f"single source train failed for {source}: returncode={code}")
        single_source_runs[source] = out_dir
        single_source_summaries[source] = read_json(out_dir / "run_summary.json")
        single_k, _ = cluster_counts_from_memory(out_dir / "cluster_memory.pt")
        single_source_clusters[source] = single_k

    for source in datasets:
        for target in datasets:
            if source == target:
                continue
            for corr_mode in args.corr_modes:
                key = (source, target, corr_mode)
                if key in completed and not args.force_transfer:
                    continue
                multi_row = (multi_cycle if corr_mode == "cycle_template" else multi_pearson).get((source, target))
                if multi_row is None:
                    continue
                target_best = best[target]
                source_best = best[source]
                single_run = single_source_runs[source]
                target_csv = Path(manifest["series_csvs"][target])
                source_csv = Path(manifest["series_csvs"][source])
                out_dir = run_root / "single_transfer" / corr_mode / f"{source}_to_{target}"
                cfg_path = cfg_root / "single_transfer" / corr_mode / f"{source}_to_{target}.yaml"
                cfg = make_transfer_cfg(
                    source=source,
                    target=target,
                    source_run=single_run,
                    source_csv=source_csv,
                    target_csv=target_csv,
                    out_dir=out_dir,
                    input_len=args.input_len,
                    pred_len=args.pred_len,
                    batch_size=args.batch_size,
                    device=args.device,
                    corr_mode=corr_mode,
                    period_min=50,
                    period_max=350,
                )
                write_yaml(cfg_path, cfg)
                row: dict[str, Any] = {
                    "status": "pending",
                    "source": source,
                    "target": target,
                    "corr_mode": corr_mode,
                    "multi_candidate": source_best.get("candidate"),
                    "single_candidate": f"single_{source_best.get('candidate')}",
                    "target_reference_candidate": target_best.get("candidate"),
                    "multi_source_clusters": source_best.get("clusters"),
                    "single_source_clusters": single_source_clusters.get(source),
                    "multi_transfer_mse": multi_row.get("transfer_mse"),
                    "multi_transfer_mae": multi_row.get("transfer_mae"),
                    "target_reference_mse": target_best.get("test_mse"),
                    "target_reference_mae": target_best.get("test_mae"),
                    "multi_transfer_out_dir": multi_row.get("transfer_out_dir"),
                    "single_source_out_dir": str(single_run),
                    "single_transfer_out_dir": str(out_dir),
                    "error": "",
                }
                try:
                    print(f"[single-transfer] {corr_mode} {source} -> {target}")
                    code, elapsed = run_cmd(
                        py + ["-m", "src.transfer", "--config", str(cfg_path)],
                        cwd=ROOT,
                        reuse_path=None if args.force_transfer else out_dir / "transfer_summary.json",
                    )
                    if code != 0:
                        row["status"] = "error"
                        row["error"] = f"src.transfer returncode={code}"
                    else:
                        summary = read_json(out_dir / "transfer_summary.json")
                        route_k, route_counts, corr_mean, corr_min = route_stats(out_dir / "cluster_assignment.csv")
                        multi_mse = float(multi_row["transfer_mse"])
                        single_mse = float(summary["avg_mse"])
                        target_mse = float(target_best["test_mse"])
                        row.update(
                            {
                                "status": "ok",
                                "single_transfer_mse": single_mse,
                                "single_transfer_mae": float(summary["avg_mae"]),
                                "multi_minus_single_mse": multi_mse - single_mse,
                                "single_minus_multi_mse": single_mse - multi_mse,
                                "multi_gain_vs_target": target_mse - multi_mse,
                                "single_gain_vs_target": target_mse - single_mse,
                                "single_target_route_clusters": route_k,
                                "single_target_route_counts": route_counts,
                                "single_target_corr_mean": corr_mean,
                                "single_target_corr_min": corr_min,
                            }
                        )
                except Exception as exc:
                    row["status"] = "error"
                    row["error"] = repr(exc)
                rows = [
                    r
                    for r in rows
                    if (r.get("source"), r.get("target"), r.get("corr_mode")) != key
                ]
                rows.append(row)
                write_csv(out_csv, rows, FIELDS)

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    positives = [
        r
        for r in ok_rows
        if r.get("single_minus_multi_mse") not in {None, ""}
        and float(r["single_minus_multi_mse"]) > 0.0
    ]
    positives.sort(key=lambda r: float(r["single_minus_multi_mse"]), reverse=True)
    write_csv(args.out_root / "positive_cases.csv", positives, FIELDS)
    print(f"Saved comparison to: {out_csv}")
    print(f"Saved positive cases to: {args.out_root / 'positive_cases.csv'}")
    if positives:
        best_case = positives[0]
        print(
            "Best positive case: "
            f"{best_case['corr_mode']} {best_case['source']}->{best_case['target']} "
            f"multi={float(best_case['multi_transfer_mse']):.6f}, "
            f"single={float(best_case['single_transfer_mse']):.6f}, "
            f"gain={float(best_case['single_minus_multi_mse']):.6f}"
        )


if __name__ == "__main__":
    main()
