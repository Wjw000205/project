from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


PENALTY_CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "shape_l1",
        "penalties": ["level", "delta", "d2_match", "diff_amp"],
        "alpha_scale": 1.1,
        "lambda_scale": 1.0,
        "gate_balance_weight": 0.0,
    },
    {
        "name": "trend_dir",
        "penalties": ["trend", "direction"],
        "alpha_scale": 0.8,
        "lambda_scale": 1.0,
        "gate_balance_weight": 0.0,
    },
    {
        "name": "level_range_trend",
        "penalties": ["level", "range", "trend", "direction"],
        "alpha_scale": 0.8,
        "lambda_scale": 1.0,
        "gate_balance_weight": 0.01,
    },
    {
        "name": "vol_dir",
        "penalties": ["amp_under", "delta", "diff_amp", "direction"],
        "alpha_scale": 0.8,
        "lambda_scale": 1.0,
        "gate_balance_weight": 0.01,
    },
]


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def nested_metric(summary: dict[str, Any], split: str, key: str) -> float | None:
    obj = summary.get(split, {}) or {}
    if key in obj:
        return float(obj[key])
    return None


def convert_pems_npz(
    npz_path: Path,
    csv_path: Path,
    *,
    feature_idx: int = 0,
    channel_limit: int = 0,
    start: str = "2018-01-01 00:00:00",
    freq: str = "5min",
) -> dict[str, Any]:
    z = np.load(npz_path)
    if "data" not in z.files:
        raise ValueError(f"{npz_path} does not contain a 'data' array. Keys={z.files}")
    arr = z["data"]
    if arr.ndim != 3:
        raise ValueError(f"Expected [T,N,F] array, got shape={arr.shape}")
    if feature_idx < 0 or feature_idx >= arr.shape[2]:
        raise ValueError(f"feature_idx={feature_idx} out of range for shape={arr.shape}")
    values = arr[:, :, feature_idx].astype(np.float32)
    if channel_limit > 0:
        values = values[:, : min(int(channel_limit), values.shape[1])]
    dates = pd.date_range(start=start, periods=values.shape[0], freq=freq)
    columns = [f"sensor_{i:03d}" for i in range(values.shape[1])]
    df = pd.DataFrame(values, columns=columns)
    df.insert(0, "date", dates)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return {
        "npz": str(npz_path),
        "csv": str(csv_path),
        "shape": list(arr.shape),
        "feature_idx": int(feature_idx),
        "rows": int(values.shape[0]),
        "channels": int(values.shape[1]),
        "freq": freq,
    }


def maybe_convert(
    npz_path: Path,
    csv_path: Path,
    feature_idx: int,
    channel_limit: int,
    force: bool,
) -> dict[str, Any]:
    if csv_path.exists() and not force:
        df = pd.read_csv(csv_path, nrows=2)
        full_cols = pd.read_csv(csv_path, nrows=0).columns
        return {
            "npz": str(npz_path),
            "csv": str(csv_path),
            "shape": None,
            "feature_idx": int(feature_idx),
            "rows": None,
            "channels": int(len(full_cols) - 1),
            "freq": "existing",
        }
    return convert_pems_npz(npz_path, csv_path, feature_idx=feature_idx, channel_limit=channel_limit)


def base_train_cfg(
    *,
    name: str,
    csv_path: Path,
    out_dir: Path,
    input_len: int,
    pred_len: int,
    max_rows: int,
    epochs: int,
    batch_size: int,
    device: str,
    candidate: dict[str, Any],
    cluster_method: str = "leader",
    n_clusters: int = 3,
    distance_threshold: float | None = 0.7,
) -> dict[str, Any]:
    penalties = list(candidate["penalties"])
    lambda_scale = float(candidate.get("lambda_scale", 1.0))
    lambda_init = {p: 0.1 * lambda_scale for p in penalties}
    return {
        "exp": {
            "name": name,
            "out_dir": str(out_dir),
            "seed": 2026,
            "deterministic": True,
            "device": device,
        },
        "data": {
            "csv_path": str(csv_path),
            "date_col": 0,
            "max_rows": int(max_rows),
            "train_ratio": 0.7,
            "val_ratio": 0.1,
            "test_ratio": 0.2,
        },
        "window": {"input_len": int(input_len), "pred_len": int(pred_len)},
        "normalize": {"global_zscore": True, "train_only": True},
        "corr": {"compute": True, "save_path": str(out_dir / "corr.npy")},
        "cluster": {
            "method": str(cluster_method),
            "n_clusters": int(n_clusters),
            "distance_threshold": distance_threshold,
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
        "model": {"predictor": "mlp", "hidden_dim": 256, "dropout": 0.2},
        "moe": {
            "enable": True,
            "topk": 1,
            "freeze_lambda": False,
            "gate_hidden_dim": 32,
            "select_ranks": [1],
            "detach_penalty_grad": False,
            "lambda_init": lambda_init,
            "lambda_min": {p: 0.0 for p in penalties},
            "lambda_schedule": {p: "none" for p in penalties},
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
                "selection_policy": "val_mse_candidate_channel",
                "selection_min_abs_improvement": 0.0,
                "selection_min_rel_improvement": 0.0,
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
            "gate_balance_weight": float(candidate.get("gate_balance_weight", 0.0)),
            "gate_route_on_penalty_only": True,
            "router_mode": "learned",
            "router_penalty_context_weight": 0.0,
            "router_detach_penalty_context": True,
            "allow_skip": True,
            "skip_cost": 0.15,
            "skip_init_bias": -2.0,
            "gate_temperature": 1.0,
            "gate_noise_std": 0.2,
            "gate_soft_weight": 0.0,
            "gate_prob_floor": 0.0,
            "gate_entropy_target_frac": 0.7,
            "residual_gate": {"enable": True, "alpha": 0.7},
            "pred_aware": {
                "enable": True,
                "use_pred_features": True,
                "use_penalty_input": False,
            },
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
                "warmup_epochs": min(5, max(1, int(epochs) // 3)),
            },
            "lr_scheduler": {
                "name": "plateau",
                "factor": 0.5,
                "patience": 3,
                "min_lr": 1.0e-6,
            },
        },
        "early_stop": {"patience": min(8, max(3, int(epochs) // 3)), "min_delta": 1.0e-6},
        "plot": {"enable": False},
        "portrait": {"enable": False},
        "eval": {"skip_test": False},
        "memory": {
            "enable": True,
            "save_checkpoint": True,
            "path": str(out_dir / "cluster_memory.pt"),
            "checkpoint_path": str(out_dir / "best_checkpoint.pt"),
        },
    }


def transfer_cfg(
    *,
    source_name: str,
    target_name: str,
    source_csv: Path,
    target_csv: Path,
    source_run: Path,
    out_dir: Path,
    input_len: int,
    pred_len: int,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    return {
        "exp": {
            "name": f"{source_name}_to_{target_name}",
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
            "step_minutes": 5,
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
        "transfer": {
            "corr_mode": "cycle_template",
            "route_fit_scope": "train",
            "use_pred_residual": True,
            "phase_bins": 64,
            "phase_max_shift": None,
            "period_min": None,
            "period_max": None,
            "period_min_hours": 12,
            "period_max_hours": 168,
            "corr_align": "head",
            "corr_threshold": None,
            "fallback_mode": "hard",
            "fallback_topk": 2,
            "fallback_temp": 1.0,
            "resample": {
                "enable": False,
                "target_step_minutes": 5,
                "method": "linear",
            },
            "save_corr": True,
        },
        "eval": {"batch_size": int(batch_size)},
    }


def run_cmd(cmd: list[str], *, reuse_path: Path | None = None) -> tuple[int, float]:
    if reuse_path is not None and reuse_path.exists():
        return 0, 0.0
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    return int(proc.returncode), time.perf_counter() - start


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def assignment_stats(out_dir: Path) -> dict[str, Any]:
    path = out_dir / "cluster_assignment.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    counts = df["cluster_id"].value_counts().sort_index().to_dict()
    return {
        "cluster_counts": json.dumps({int(k): int(v) for k, v in counts.items()}),
        "corr_max_mean": float(df["corr_max"].mean()),
        "corr_max_min": float(df["corr_max"].min()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pems04-npz", default=str(ROOT / "data" / "pems04.npz"))
    ap.add_argument("--pems08-npz", default=str(ROOT / "data" / "pems08.npz"))
    ap.add_argument("--out-root", default=str(ROOT / "outputs" / "pems04_pems08_val_transfer"))
    ap.add_argument("--feature-idx", type=int, default=0)
    ap.add_argument(
        "--channel-limit",
        type=int,
        default=0,
        help="Keep only the first N sensors in converted CSVs. 0 keeps all sensors.",
    )
    ap.add_argument("--input-len", type=int, default=336)
    ap.add_argument("--pred-len", type=int, default=96)
    ap.add_argument("--max-rows", type=int, default=12000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--budget", choices=["smoke", "compact", "full"], default="compact")
    ap.add_argument("--source", choices=["PEMS04", "PEMS08"], default="PEMS04")
    ap.add_argument("--target", choices=["PEMS04", "PEMS08"], default="PEMS08")
    ap.add_argument("--cluster-method", default="leader", choices=["leader", "kmeans", "spectral", "agglomerative", "random"])
    ap.add_argument("--n-clusters", type=int, default=3)
    ap.add_argument("--distance-threshold", type=float, default=0.7)
    ap.add_argument(
        "--only-candidate",
        default="",
        help="Run only one named candidate, e.g. level_range_trend.",
    )
    ap.add_argument("--reuse-existing", action="store_true")
    ap.add_argument("--force-convert", action="store_true")
    ap.add_argument("--skip-target-base", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    cfg_dir = out_root / "configs"
    run_dir = out_root / "runs"
    data_dir = out_root / "data"
    result_path = out_root / "search_results.csv"
    transfer_result_path = out_root / "transfer.csv"

    channel_tag = f"_c{int(args.channel_limit)}" if int(args.channel_limit) > 0 else ""
    pems04_csv = data_dir / f"PEMS04{channel_tag}.csv"
    pems08_csv = data_dir / f"PEMS08{channel_tag}.csv"
    conversions = [
        maybe_convert(
            Path(args.pems04_npz),
            pems04_csv,
            int(args.feature_idx),
            int(args.channel_limit),
            bool(args.force_convert),
        ),
        maybe_convert(
            Path(args.pems08_npz),
            pems08_csv,
            int(args.feature_idx),
            int(args.channel_limit),
            bool(args.force_convert),
        ),
    ]
    with (out_root / "conversion_summary.json").open("w", encoding="utf-8") as f:
        json.dump(conversions, f, indent=2)

    if args.source == args.target:
        raise ValueError("--source and --target must differ.")
    csv_by_name = {"PEMS04": pems04_csv, "PEMS08": pems08_csv}
    source_csv = csv_by_name[str(args.source)]
    target_csv = csv_by_name[str(args.target)]

    if args.budget == "smoke":
        candidates = PENALTY_CANDIDATES[:1]
        epochs = 1
    elif args.budget == "compact":
        candidates = PENALTY_CANDIDATES
        epochs = int(args.epochs)
    else:
        candidates = []
        for cand in PENALTY_CANDIDATES:
            for scale in [0.5, 1.0, 2.0]:
                cur = dict(cand)
                cur["name"] = f"{cand['name']}_ls{str(scale).replace('.', 'p')}"
                cur["lambda_scale"] = scale
                candidates.append(cur)
        epochs = int(args.epochs)
    only_candidate = str(args.only_candidate).strip()
    if only_candidate:
        candidates = [c for c in candidates if str(c["name"]) == only_candidate]
        if not candidates:
            raise ValueError(
                f"Unknown --only-candidate={only_candidate!r}. "
                f"Available: {[c['name'] for c in PENALTY_CANDIDATES]}"
            )

    rows: list[dict[str, Any]] = []
    fields = [
        "status",
        "candidate",
        "penalties",
        "lambda_scale",
        "alpha_scale",
        "gate_balance_weight",
        "val_mse",
        "val_mae",
        "test_mse_ref",
        "test_mae_ref",
        "best_epoch",
        "out_dir",
        "config",
        "seconds",
        "error",
    ]
    for cand in candidates:
        name = str(cand["name"])
        cand_out = run_dir / f"{str(args.source).lower()}_search" / name
        cfg_path = cfg_dir / f"{str(args.source).lower()}_search" / f"{name}.yaml"
        cfg = base_train_cfg(
            name=f"{args.source}_{name}",
            csv_path=source_csv,
            out_dir=cand_out,
            input_len=int(args.input_len),
            pred_len=int(args.pred_len),
            max_rows=int(args.max_rows),
            epochs=epochs,
            batch_size=int(args.batch_size),
            device=str(args.device),
            candidate=cand,
            cluster_method=str(args.cluster_method),
            n_clusters=int(args.n_clusters),
            distance_threshold=(
                None
                if str(args.cluster_method).lower() in {"kmeans", "spectral", "random"}
                else float(args.distance_threshold)
            ),
        )
        write_yaml(cfg_path, cfg)
        row: dict[str, Any] = {
            "status": "ok",
            "candidate": name,
            "penalties": json.dumps(cand["penalties"]),
            "lambda_scale": cand.get("lambda_scale", 1.0),
            "alpha_scale": cand.get("alpha_scale", ""),
            "gate_balance_weight": cand.get("gate_balance_weight", ""),
            "out_dir": str(cand_out),
            "config": str(cfg_path),
        }
        try:
            rc, sec = run_cmd(
                [sys.executable, "-m", "src.train", "--config", str(cfg_path)],
                reuse_path=(cand_out / "run_summary.json") if args.reuse_existing else None,
            )
            row["seconds"] = sec
            if rc != 0:
                raise RuntimeError(f"train failed with code {rc}")
            summary = load_summary(cand_out / "run_summary.json")
            row["val_mse"] = nested_metric(summary, "val", "avg_mse")
            row["val_mae"] = nested_metric(summary, "val", "avg_mae")
            row["test_mse_ref"] = nested_metric(summary, "test", "avg_mse")
            row["test_mae_ref"] = nested_metric(summary, "test", "avg_mae")
            row["best_epoch"] = json.dumps(summary.get("best_epoch", []))
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
        rows.append(row)
        write_csv(result_path, rows, fields)

    valid = [r for r in rows if r.get("status") == "ok" and r.get("val_mse") not in ("", None)]
    if not valid:
        raise RuntimeError("No successful PEMS04 val-search candidate.")
    best = min(valid, key=lambda r: float(r["val_mse"]))
    best_name = str(best["candidate"])
    best_candidate = next(c for c in candidates if c["name"] == best_name)
    best_source_run = Path(str(best["out_dir"]))
    final_source_run = run_dir / f"{args.source}_best_source"
    if final_source_run.exists() and not args.reuse_existing:
        shutil.rmtree(final_source_run)
    if not final_source_run.exists():
        shutil.copytree(best_source_run, final_source_run)

    transfer_out = run_dir / f"{args.source}_to_{args.target}_transfer"
    transfer_cfg_path = cfg_dir / f"{args.source}_to_{args.target}_transfer.yaml"
    write_yaml(
        transfer_cfg_path,
        transfer_cfg(
            source_name=str(args.source),
            target_name=str(args.target),
            source_csv=source_csv,
            target_csv=target_csv,
            source_run=final_source_run,
            out_dir=transfer_out,
            input_len=int(args.input_len),
            pred_len=int(args.pred_len),
            device=str(args.device),
            batch_size=int(args.batch_size),
        ),
    )

    transfer_rows: list[dict[str, Any]] = []
    transfer_fields = [
        "status",
        "source",
        "target",
        "selected_candidate",
        "selected_val_mse",
        "selected_val_mae",
        "selected_source_test_mse_ref",
        "target_base_mse",
        "target_base_mae",
        "transfer_mse",
        "transfer_mae",
        "transfer_gain_mse_vs_target_base",
        "transfer_gain_mae_vs_target_base",
        "route_fit_scope",
        "normalize_train_only",
        "route_uses_train_only",
        "eval_uses_test_only",
        "predictor_variant",
        "penalty_names",
        "cluster_counts",
        "corr_max_mean",
        "corr_max_min",
        "source_run",
        "transfer_run",
        "target_base_run",
        "error",
    ]
    trow: dict[str, Any] = {
        "status": "ok",
        "source": str(args.source),
        "target": str(args.target),
        "selected_candidate": best_name,
        "selected_val_mse": best.get("val_mse"),
        "selected_val_mae": best.get("val_mae"),
        "selected_source_test_mse_ref": best.get("test_mse_ref"),
        "source_run": str(final_source_run),
        "transfer_run": str(transfer_out),
    }
    try:
        target_base_out = None
        if not args.skip_target_base:
            target_base_out = run_dir / f"{args.target}_target_base"
            target_base_cfg_path = cfg_dir / f"{args.target}_target_base.yaml"
            write_yaml(
                target_base_cfg_path,
                base_train_cfg(
                    name=f"{args.target}_target_base_{best_name}",
                    csv_path=target_csv,
                    out_dir=target_base_out,
                    input_len=int(args.input_len),
                    pred_len=int(args.pred_len),
                    max_rows=int(args.max_rows),
                    epochs=epochs,
                    batch_size=int(args.batch_size),
                    device=str(args.device),
                    candidate=best_candidate,
                    cluster_method=str(args.cluster_method),
                    n_clusters=int(args.n_clusters),
                    distance_threshold=(
                        None
                        if str(args.cluster_method).lower() in {"kmeans", "spectral", "random"}
                        else float(args.distance_threshold)
                    ),
                ),
            )
            rc, _ = run_cmd(
                [sys.executable, "-m", "src.train", "--config", str(target_base_cfg_path)],
                reuse_path=(target_base_out / "run_summary.json") if args.reuse_existing else None,
            )
            if rc != 0:
                raise RuntimeError(f"target base failed with code {rc}")
            target_summary = load_summary(target_base_out / "run_summary.json")
            trow["target_base_mse"] = nested_metric(target_summary, "test", "avg_mse")
            trow["target_base_mae"] = nested_metric(target_summary, "test", "avg_mae")
            trow["target_base_run"] = str(target_base_out)

        rc, _ = run_cmd(
            [sys.executable, "-m", "src.transfer", "--config", str(transfer_cfg_path)],
            reuse_path=(transfer_out / "transfer_summary.json") if args.reuse_existing else None,
        )
        if rc != 0:
            raise RuntimeError(f"transfer failed with code {rc}")
        ts = load_summary(transfer_out / "transfer_summary.json")
        trow["transfer_mse"] = ts.get("avg_mse")
        trow["transfer_mae"] = ts.get("avg_mae")
        trow["route_fit_scope"] = ts.get("route_fit_scope")
        trow["normalize_train_only"] = ts.get("normalize_train_only")
        trow["route_uses_train_only"] = ts.get("route_uses_train_only")
        trow["eval_uses_test_only"] = ts.get("eval_uses_test_only")
        trow["predictor_variant"] = ts.get("predictor_variant")
        trow["penalty_names"] = json.dumps(ts.get("penalty_names", []))
        trow.update(assignment_stats(transfer_out))
        if trow.get("target_base_mse") not in ("", None):
            trow["transfer_gain_mse_vs_target_base"] = float(trow["target_base_mse"]) - float(trow["transfer_mse"])
        if trow.get("target_base_mae") not in ("", None):
            trow["transfer_gain_mae_vs_target_base"] = float(trow["target_base_mae"]) - float(trow["transfer_mae"])
    except Exception as exc:
        trow["status"] = "failed"
        trow["error"] = str(exc)
    transfer_rows.append(trow)
    write_csv(transfer_result_path, transfer_rows, transfer_fields)

    print(f"Saved converted CSVs: {pems04_csv}, {pems08_csv}")
    print(f"Saved search results: {result_path}")
    print(f"Saved transfer results: {transfer_result_path}")


if __name__ == "__main__":
    main()
