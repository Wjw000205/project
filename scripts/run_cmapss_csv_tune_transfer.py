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

import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "CMAPSSData"
OUT_DATA_DIR = ROOT / "data" / "CMAPSS"
DATASETS = ["FD001", "FD002", "FD003", "FD004"]

COLS = (
    ["unit", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

CSV_FIELDS_VAL = [
    "status",
    "dataset",
    "candidate",
    "input_len",
    "pred_len",
    "hidden_dim",
    "dropout",
    "distance_threshold",
    "penalties",
    "val_mse",
    "val_mae",
    "test_mse",
    "test_mae",
    "best_epoch",
    "clusters",
    "cluster_counts",
    "out_dir",
    "config",
    "elapsed_sec",
    "error",
]

CSV_FIELDS_TRANSFER = [
    "status",
    "source",
    "target",
    "source_candidate",
    "target_reference_candidate",
    "pred_len",
    "source_test_mse",
    "source_test_mae",
    "target_reference_test_mse",
    "target_reference_test_mae",
    "transfer_mse",
    "transfer_mae",
    "delta_mse_vs_target_reference",
    "gain_mse_vs_target_reference",
    "route_fit_scope",
    "normalize_train_only",
    "route_uses_train_only",
    "eval_uses_test_only",
    "predictor_variant",
    "corr_mode",
    "target_route_clusters",
    "target_route_counts",
    "target_corr_mean",
    "target_corr_min",
    "source_out_dir",
    "transfer_out_dir",
    "config",
    "elapsed_sec",
    "error",
]


CANDIDATES_COMPACT: list[dict[str, Any]] = [
    {
        "name": "shape_thr0p7_h128",
        "hidden_dim": 128,
        "dropout": 0.1,
        "distance_threshold": 0.7,
        "penalties": ["level", "delta", "d2_match", "diff_amp"],
        "alpha_scale": 0.8,
        "lambda_scale": 0.5,
    },
    {
        "name": "range_trend_thr0p5_h128",
        "hidden_dim": 128,
        "dropout": 0.1,
        "distance_threshold": 0.5,
        "penalties": ["level", "range", "trend", "direction"],
        "alpha_scale": 0.8,
        "lambda_scale": 0.5,
    },
    {
        "name": "shape_thr0p3_h256",
        "hidden_dim": 256,
        "dropout": 0.2,
        "distance_threshold": 0.3,
        "penalties": ["level", "delta", "d2_match", "diff_amp"],
        "alpha_scale": 1.1,
        "lambda_scale": 1.0,
    },
]

CANDIDATES_FULL: list[dict[str, Any]] = CANDIDATES_COMPACT + [
    {
        "name": "vol_dir_thr0p5_h128",
        "hidden_dim": 128,
        "dropout": 0.1,
        "distance_threshold": 0.5,
        "penalties": ["amp_under", "delta", "diff_amp", "direction"],
        "alpha_scale": 0.8,
        "lambda_scale": 0.7,
    },
    {
        "name": "trend_dir_thr0p7_h256",
        "hidden_dim": 256,
        "dropout": 0.2,
        "distance_threshold": 0.7,
        "penalties": ["delta", "trend", "direction"],
        "alpha_scale": 1.0,
        "lambda_scale": 0.7,
    },
]


def read_cmapss_txt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS, engine="python")
    return df


def convert_cmapss(raw_dir: Path, out_dir: Path, *, force: bool = False) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    series_dir = out_dir / "series"
    series_dir.mkdir(parents=True, exist_ok=True)
    train_csv = out_dir / "train.csv"
    test_csv = out_dir / "test.csv"
    rul_csv = out_dir / "rul.csv"
    manifest_path = out_dir / "manifest.json"
    if train_csv.exists() and test_csv.exists() and rul_csv.exists() and not force:
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as f:
                return json.load(f)

    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    rul_parts: list[pd.DataFrame] = []
    series_paths: dict[str, str] = {}
    for name in DATASETS:
        train = read_cmapss_txt(raw_dir / f"train_{name}.txt")
        test = read_cmapss_txt(raw_dir / f"test_{name}.txt")
        rul = pd.read_csv(raw_dir / f"RUL_{name}.txt", sep=r"\s+", header=None, names=["final_rul"], engine="python")
        rul.insert(0, "unit", range(1, len(rul) + 1))
        rul.insert(0, "dataset", name)

        train_max_cycle = train.groupby("unit")["cycle"].transform("max")
        train.insert(0, "dataset", name)
        train["rul"] = train_max_cycle - train["cycle"]

        test_max_cycle = test.groupby("unit")["cycle"].transform("max")
        test = test.merge(rul[["unit", "final_rul"]], on="unit", how="left")
        test.insert(0, "dataset", name)
        test["rul"] = test_max_cycle - test["cycle"] + test["final_rul"]

        train_parts.append(train)
        test_parts.append(test)
        rul_parts.append(rul)

        value_cols = [c for c in COLS if c not in {"unit", "cycle"}]
        wide = train[value_cols].copy()
        wide.insert(0, "date", range(len(wide)))
        series_path = series_dir / f"{name}.csv"
        wide.to_csv(series_path, index=False)
        series_paths[name] = str(series_path)

    train_all = pd.concat(train_parts, ignore_index=True)
    test_all = pd.concat(test_parts, ignore_index=True)
    rul_all = pd.concat(rul_parts, ignore_index=True)
    train_all.to_csv(train_csv, index=False)
    test_all.to_csv(test_csv, index=False)
    rul_all.to_csv(rul_csv, index=False)

    manifest = {
        "readable_csvs": {
            "train": str(train_csv),
            "test": str(test_csv),
            "rul": str(rul_csv),
        },
        "series_csvs": series_paths,
        "rows": {
            "train": int(len(train_all)),
            "test": int(len(test_all)),
            "rul": int(len(rul_all)),
        },
        "channels_per_series": int(len(value_cols)),
        "series_source": "train trajectories concatenated by unit/cycle; date is a synthetic row index",
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def metric(summary: dict[str, Any], split: str, key: str) -> float | None:
    obj = summary.get(split, {}) or {}
    if key in obj:
        return float(obj[key])
    return None


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


def cluster_counts_from_memory(path: Path) -> tuple[int | None, str]:
    if not path.exists():
        return None, ""
    payload = torch.load(path, map_location="cpu")
    cluster_id = payload.get("cluster_id_c")
    if cluster_id is None:
        return None, ""
    counts: dict[int, int] = {}
    for value in cluster_id.detach().cpu().tolist():
        key = int(value)
        counts[key] = counts.get(key, 0) + 1
    counts = dict(sorted(counts.items(), key=lambda item: item[0]))
    return len(counts), json.dumps(counts, ensure_ascii=False)


def route_stats(path: Path) -> tuple[int | None, str, float | None, float | None]:
    if not path.exists():
        return None, "", None, None
    counts: dict[int, int] = {}
    corr_values: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = int(float(row["cluster_id"]))
            counts[cid] = counts.get(cid, 0) + 1
            if row.get("corr_max", "") != "":
                corr_values.append(float(row["corr_max"]))
    counts = dict(sorted(counts.items(), key=lambda item: item[0]))
    corr_mean = sum(corr_values) / len(corr_values) if corr_values else None
    corr_min = min(corr_values) if corr_values else None
    return len(counts), json.dumps(counts, ensure_ascii=False), corr_mean, corr_min


def make_train_cfg(
    *,
    dataset: str,
    series_csv: Path,
    out_dir: Path,
    candidate: dict[str, Any],
    input_len: int,
    pred_len: int,
    epochs: int,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    penalties = list(candidate["penalties"])
    lambda_scale = float(candidate.get("lambda_scale", 1.0))
    lambda_init = {name: 0.1 * lambda_scale for name in penalties}
    cfg: dict[str, Any] = {
        "exp": {
            "name": f"CMAPSS_{dataset}_{candidate['name']}",
            "out_dir": str(out_dir),
            "seed": 2026,
            "deterministic": True,
            "device": device,
        },
        "data": {
            "csv_path": str(series_csv),
            "date_col": 0,
            "train_ratio": 0.7,
            "val_ratio": 0.1,
            "test_ratio": 0.2,
        },
        "window": {"input_len": int(input_len), "pred_len": int(pred_len)},
        "normalize": {"global_zscore": True, "train_only": True},
        "corr": {"compute": True, "save_path": str(out_dir / "corr.npy")},
        "cluster": {
            "method": "leader",
            "n_clusters": 3,
            "distance_threshold": float(candidate["distance_threshold"]),
            "linkage": "average",
            "kmeans_n_init": 10,
            "kmeans_max_iter": 300,
            "spectral_affinity": "corr",
            "rbf_gamma": 1.0,
            "dbscan_eps": 0.7,
            "dbscan_min_samples": 2,
            "random_state": 2026,
            "min_cluster_size": 2,
            "merge_small_clusters": True,
            "no_merge_if_channels_lt": 7,
            "train_only": True,
        },
        "model": {
            "predictor": "mlp",
            "hidden_dim": int(candidate["hidden_dim"]),
            "dropout": float(candidate["dropout"]),
        },
        "moe": {
            "enable": True,
            "topk": 1,
            "freeze_lambda": False,
            "gate_hidden_dim": 32,
            "min_k_for_extensions": 3,
            "safeguard_hidden_dim": 64,
            "select_ranks": [1],
            "detach_penalty_grad": False,
            "lambda_init": lambda_init,
            "lambda_min": {name: 0.0 for name in penalties},
            "lambda_schedule": {name: "none" for name in penalties},
            "pred_side_residual": {
                "enable": True,
                "feature_mode": "legacy",
                "residual_clip": 0.0,
                "corrector_hidden": 32,
                "init_alpha": -3.0,
                "alpha_scale": float(candidate.get("alpha_scale", 1.0)),
                "specialization_weight": 0.1,
                "norm_weight": 0.0,
                "use_y_base_input": True,
                "intervention_enable": False,
                "intervention_init": -2.0,
                "intervention_weight": 1.0e-3,
                "detach_routed_penalty_pred": False,
                "selection_policy": "val_mse_gate",
                "selection_min_abs_improvement": 0.0,
                "selection_min_rel_improvement": 0.0,
                "gate_calibrator": {
                    "loss": "mse",
                    "selection_metric": "mse",
                    "epochs": 20,
                    "train_fraction": 0.7,
                    "hidden_dim": 32,
                    "batch_size": 256,
                    "max_scale": 1.0,
                    "init_scale": 0.8,
                    "scale_reg": 1.0e-4,
                },
            },
            "dynamic_lambda": {
                "enable": True,
                "mode": "multiscale",
                "hidden_dim": 32,
                "segment_bins": [4, 8],
                "max_factor": 1.5,
                "mix": 0.6,
                "dropout": 0.0,
                "reg_weight": 1.0e-4,
            },
            "learnable_lambda": {
                "enable": False,
                "reg_weight": 0.01,
                "share_floor": 0.05,
                "bilevel": {
                    "enable": True,
                    "optimize_gate": False,
                    "outer_lr": 0.005,
                    "inner_lr": 0.001,
                    "val_metric": "val_mse",
                    "steps_per_epoch": 10,
                },
            },
            "gate_entropy_weight": 0.0,
            "gate_balance_weight": 0.0,
            "gate_route_on_penalty_only": True,
            "router_mode": "learned",
            "router_penalty_context_weight": 0.0,
            "router_detach_penalty_context": True,
            "allow_skip": True,
            "skip_cost": 0.15,
            "skip_init_bias": -2.0,
            "gate_temperature": 1.0,
            "gate_noise_std": 0.2,
            "gate_init_bias": {
                "enable": True,
                "values": {"level": 2.0, "default": 0.0},
            },
            "gate_soft_weight": 0.0,
            "gate_prob_floor": 0.0,
            "gate_entropy_target_frac": 0.7,
            "residual_gate": {"enable": True, "alpha": 0.7},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
            "gate_logit_clip": 5.0,
        },
        "penalties": {"enabled": penalties, "jump_threshold": 0.6},
        "train": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": 0.001,
            "mse_weight": 0.9,
            "selection_metric": "val_mse",
            "weight_decay": 1.0e-4,
            "grad_clip": 1.0,
            "penalty_warmup_epochs": min(10, max(1, int(epochs) // 4)),
            "mae_objective": {
                "enable": True,
                "kind": "l1",
                "weight": 0.6,
                "warmup_epochs": 5,
            },
            "lr_scheduler": {
                "name": "plateau",
                "factor": 0.5,
                "patience": 3,
                "min_lr": 1.0e-6,
            },
        },
        "early_stop": {"patience": min(8, max(3, int(epochs) // 3)), "min_delta": 1.0e-6},
        "knn_hybrid": {"enable": False, "use_for_model_selection": False},
        "eval": {"skip_test": False},
        "plot": {"enable": False},
        "portrait": {"enable": False},
        "memory": {
            "enable": True,
            "path": str(out_dir / "cluster_memory.pt"),
            "save_checkpoint": True,
            "checkpoint_path": str(out_dir / "best_checkpoint.pt"),
        },
    }
    return cfg


def make_transfer_cfg(
    *,
    source: str,
    target: str,
    source_run: Path,
    source_csv: Path,
    target_csv: Path,
    out_dir: Path,
    input_len: int,
    pred_len: int,
    batch_size: int,
    device: str,
    corr_mode: str,
    period_min: int | None,
    period_max: int | None,
) -> dict[str, Any]:
    transfer_cfg: dict[str, Any] = {
        "corr_mode": corr_mode,
        "route_fit_scope": "train",
        "use_pred_residual": True,
        "corr_align": "head",
        "corr_max_lag": 0,
        "corr_threshold": None,
        "fallback_mode": "hard",
        "fallback_topk": 2,
        "fallback_temp": 1.0,
        "resample": {"enable": False, "target_step_minutes": None, "method": "linear"},
        "knn_hybrid": {
            "enable": False,
            "scope": "same_cluster",
            "bank_split": "train",
            "use_for_model_selection": False,
            "k": 16,
            "alpha": 0.1,
            "adaptive_alpha": "confidence",
            "confidence_floor": 0.0,
            "distance_sharpness": 1.0,
            "shape_bins": 24,
            "diff_bins": 12,
            "bank_stride": 4,
            "distance_weight": "inverse",
            "anchor_mode": "last",
        },
        "save_corr": True,
    }
    if corr_mode in {"cycle", "cycle_template", "phase", "phase_template"}:
        transfer_cfg.update(
            {
                "phase_bins": 64,
                "phase_max_shift": None,
                "period_min": period_min,
                "period_max": period_max,
                "period_min_hours": None,
                "period_max_hours": None,
            }
        )
    return {
        "exp": {
            "name": f"CMAPSS_{source}_to_{target}",
            "out_dir": str(out_dir),
            "seed": 2026,
            "deterministic": True,
            "device": device,
        },
        "source": {
            "memory_path": str(source_run / "cluster_memory.pt"),
            "checkpoint_path": str(source_run / "best_checkpoint.pt"),
            "summary_path": str(source_run / "run_summary.json"),
            "csv_path": str(source_csv),
            "date_col": 0,
        },
        "data": {
            "csv_path": str(target_csv),
            "date_col": 0,
            "train_ratio": 0.7,
            "val_ratio": 0.1,
            "test_ratio": 0.2,
        },
        "window": {"input_len": int(input_len), "pred_len": int(pred_len)},
        "normalize": {"global_zscore": True, "train_only": True},
        "transfer": transfer_cfg,
        "eval": {"batch_size": int(batch_size)},
    }


def best_rows_by_dataset(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    ok_rows = [r for r in rows if r.get("status") == "ok" and r.get("val_mse") not in {None, ""}]
    for row in ok_rows:
        dataset = str(row["dataset"])
        if dataset not in best or float(row["val_mse"]) < float(best[dataset]["val_mse"]):
            best[dataset] = row
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--data-out", type=Path, default=OUT_DATA_DIR)
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "cmapss_val_search_transfer")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--budget", choices=["compact", "full"], default="compact")
    ap.add_argument("--input-len", type=int, default=96)
    ap.add_argument("--pred-len", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--python", type=Path, default=None)
    ap.add_argument("--force-convert", action="store_true")
    ap.add_argument("--force-train", action="store_true")
    ap.add_argument("--force-transfer", action="store_true")
    ap.add_argument("--skip-transfer", action="store_true")
    ap.add_argument("--transfer-corr-mode", choices=["pearson", "cycle_template"], default="pearson")
    ap.add_argument("--transfer-period-min", type=int, default=50)
    ap.add_argument("--transfer-period-max", type=int, default=350)
    args = ap.parse_args()

    manifest = convert_cmapss(args.raw_dir, args.data_out, force=args.force_convert)
    candidates = CANDIDATES_FULL if args.budget == "full" else CANDIDATES_COMPACT
    datasets = [d for d in args.datasets if d in DATASETS]
    py = command_python(args)

    cfg_root = args.out_root / "configs"
    run_root = args.out_root / "runs"
    val_rows: list[dict[str, Any]] = []
    existing_val_csv = args.out_root / "val_search.csv"
    if existing_val_csv.exists() and not args.force_train:
        with existing_val_csv.open("r", encoding="utf-8") as f:
            val_rows = list(csv.DictReader(f))

    completed_keys = {(r.get("dataset"), r.get("candidate")) for r in val_rows if r.get("status") == "ok"}
    for dataset in datasets:
        series_csv = Path(manifest["series_csvs"][dataset])
        for candidate in candidates:
            key = (dataset, candidate["name"])
            if key in completed_keys and not args.force_train:
                continue
            out_dir = run_root / "val_search" / dataset / candidate["name"]
            cfg_path = cfg_root / "val_search" / dataset / f"{candidate['name']}.yaml"
            cfg = make_train_cfg(
                dataset=dataset,
                series_csv=series_csv,
                out_dir=out_dir,
                candidate=candidate,
                input_len=args.input_len,
                pred_len=args.pred_len,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
            )
            write_yaml(cfg_path, cfg)
            row: dict[str, Any] = {
                "status": "pending",
                "dataset": dataset,
                "candidate": candidate["name"],
                "input_len": args.input_len,
                "pred_len": args.pred_len,
                "hidden_dim": candidate["hidden_dim"],
                "dropout": candidate["dropout"],
                "distance_threshold": candidate["distance_threshold"],
                "penalties": ",".join(candidate["penalties"]),
                "out_dir": str(out_dir),
                "config": str(cfg_path),
                "error": "",
            }
            try:
                print(f"[val] {dataset} {candidate['name']}")
                code, elapsed = run_cmd(
                    py + ["-m", "src.train", "--config", str(cfg_path)],
                    cwd=ROOT,
                    reuse_path=None if args.force_train else out_dir / "run_summary.json",
                )
                row["elapsed_sec"] = elapsed
                if code != 0:
                    row["status"] = "error"
                    row["error"] = f"src.train returncode={code}"
                else:
                    summary = read_json(out_dir / "run_summary.json")
                    clusters, counts = cluster_counts_from_memory(out_dir / "cluster_memory.pt")
                    row.update(
                        {
                            "status": "ok",
                            "val_mse": metric(summary, "val", "avg_mse"),
                            "val_mae": metric(summary, "val", "avg_mae"),
                            "test_mse": metric(summary, "test", "avg_mse"),
                            "test_mae": metric(summary, "test", "avg_mae"),
                            "best_epoch": json.dumps(summary.get("best_epoch", []), ensure_ascii=False),
                            "clusters": clusters,
                            "cluster_counts": counts,
                        }
                    )
            except Exception as exc:
                row["status"] = "error"
                row["error"] = repr(exc)
            val_rows = [r for r in val_rows if (r.get("dataset"), r.get("candidate")) != key]
            val_rows.append(row)
            write_csv(args.out_root / "val_search.csv", val_rows, CSV_FIELDS_VAL)

    best = best_rows_by_dataset(val_rows)
    best_rows = list(best.values())
    write_csv(args.out_root / "best_by_dataset.csv", best_rows, CSV_FIELDS_VAL)
    with (args.out_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "data_manifest": manifest,
                "budget": args.budget,
                "input_len": args.input_len,
                "pred_len": args.pred_len,
                "epochs": args.epochs,
                "best_by_dataset": {k: v for k, v in best.items()},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    if args.skip_transfer:
        print(f"Saved val search to: {args.out_root / 'val_search.csv'}")
        return

    transfer_rows: list[dict[str, Any]] = []
    transfer_csv = args.out_root / "transfer.csv"
    if transfer_csv.exists() and not args.force_transfer:
        with transfer_csv.open("r", encoding="utf-8") as f:
            transfer_rows = list(csv.DictReader(f))
    completed_pairs = {(r.get("source"), r.get("target")) for r in transfer_rows if r.get("status") == "ok"}

    for source in datasets:
        if source not in best:
            continue
        source_best = best[source]
        source_run = Path(str(source_best["out_dir"]))
        source_csv = Path(manifest["series_csvs"][source])
        source_summary = read_json(source_run / "run_summary.json")
        for target in datasets:
            if target == source or target not in best:
                continue
            pair_key = (source, target)
            if pair_key in completed_pairs and not args.force_transfer:
                continue
            target_best = best[target]
            target_csv = Path(manifest["series_csvs"][target])
            out_dir = run_root / "transfer" / f"{source}_to_{target}"
            cfg_path = cfg_root / "transfer" / f"{source}_to_{target}.yaml"
            cfg = make_transfer_cfg(
                source=source,
                target=target,
                source_run=source_run,
                source_csv=source_csv,
                target_csv=target_csv,
                out_dir=out_dir,
                input_len=args.input_len,
                pred_len=args.pred_len,
                batch_size=args.batch_size,
                device=args.device,
                corr_mode=args.transfer_corr_mode,
                period_min=args.transfer_period_min,
                period_max=args.transfer_period_max,
            )
            write_yaml(cfg_path, cfg)
            row: dict[str, Any] = {
                "status": "pending",
                "source": source,
                "target": target,
                "source_candidate": source_best.get("candidate"),
                "target_reference_candidate": target_best.get("candidate"),
                "pred_len": args.pred_len,
                "source_test_mse": metric(source_summary, "test", "avg_mse"),
                "source_test_mae": metric(source_summary, "test", "avg_mae"),
                "target_reference_test_mse": target_best.get("test_mse"),
                "target_reference_test_mae": target_best.get("test_mae"),
                "source_out_dir": str(source_run),
                "transfer_out_dir": str(out_dir),
                "config": str(cfg_path),
                "error": "",
            }
            try:
                print(f"[transfer] {source} -> {target}")
                code, elapsed = run_cmd(
                    py + ["-m", "src.transfer", "--config", str(cfg_path)],
                    cwd=ROOT,
                    reuse_path=None if args.force_transfer else out_dir / "transfer_summary.json",
                )
                row["elapsed_sec"] = elapsed
                if code != 0:
                    row["status"] = "error"
                    row["error"] = f"src.transfer returncode={code}"
                else:
                    summary = read_json(out_dir / "transfer_summary.json")
                    route_k, route_counts, corr_mean, corr_min = route_stats(out_dir / "cluster_assignment.csv")
                    transfer_mse = float(summary["avg_mse"])
                    transfer_mae = float(summary["avg_mae"])
                    target_mse = float(target_best["test_mse"])
                    row.update(
                        {
                            "status": "ok",
                            "transfer_mse": transfer_mse,
                            "transfer_mae": transfer_mae,
                            "delta_mse_vs_target_reference": transfer_mse - target_mse,
                            "gain_mse_vs_target_reference": target_mse - transfer_mse,
                            "route_fit_scope": summary.get("route_fit_scope"),
                            "normalize_train_only": summary.get("normalize_train_only"),
                            "route_uses_train_only": summary.get("route_uses_train_only"),
                            "eval_uses_test_only": summary.get("eval_uses_test_only"),
                            "predictor_variant": summary.get("predictor_variant"),
                            "corr_mode": summary.get("corr_mode"),
                            "target_route_clusters": route_k,
                            "target_route_counts": route_counts,
                            "target_corr_mean": corr_mean,
                            "target_corr_min": corr_min,
                        }
                    )
            except Exception as exc:
                row["status"] = "error"
                row["error"] = repr(exc)
            transfer_rows = [r for r in transfer_rows if (r.get("source"), r.get("target")) != pair_key]
            transfer_rows.append(row)
            write_csv(transfer_csv, transfer_rows, CSV_FIELDS_TRANSFER)

    write_csv(transfer_csv, transfer_rows, CSV_FIELDS_TRANSFER)
    print(f"Saved readable CSV manifest to: {args.data_out / 'manifest.json'}")
    print(f"Saved val search to: {args.out_root / 'val_search.csv'}")
    print(f"Saved best configs to: {args.out_root / 'best_by_dataset.csv'}")
    print(f"Saved transfer results to: {transfer_csv}")


if __name__ == "__main__":
    main()
