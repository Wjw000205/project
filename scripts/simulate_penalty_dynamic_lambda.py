import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.yaml_io import load_yaml


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def deep_update(target: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = value
    return target


def ensure_local_paths(cfg: Dict[str, Any], out_dir: Path) -> None:
    cfg.setdefault("exp", {})
    cfg["exp"]["out_dir"] = str(out_dir)
    cfg.setdefault("corr", {})
    cfg["corr"]["save_path"] = str(out_dir / "corr.npy")
    cfg.setdefault("plot", {})
    cfg["plot"]["enable"] = False
    cfg.setdefault("portrait", {})
    cfg["portrait"]["enable"] = False
    cfg["portrait"]["out_dir"] = str(out_dir / "cluster_portraits")
    cfg.setdefault("memory", {})
    cfg["memory"]["path"] = str(out_dir / "cluster_memory.pt")
    cfg["memory"]["checkpoint_path"] = str(out_dir / "best_checkpoint.pt")


def read_skip_stats(csv_path: Path) -> Dict[str, float]:
    if not csv_path.exists():
        return {
            "avg_skip_active": 0.0,
            "avg_skip_prob": 0.0,
            "avg_top_penalty_prob": 0.0,
        }
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    skip_rows = [r for r in rows if r["penalty"] == "skip"]
    penalty_rows = [r for r in rows if r["penalty"] != "skip"]
    avg_skip_active = 0.0
    avg_skip_prob = 0.0
    if skip_rows:
        avg_skip_active = sum(float(r["avg_skip_active"]) for r in skip_rows) / len(skip_rows)
        avg_skip_prob = sum(float(r["avg_prob"]) for r in skip_rows) / len(skip_rows)
    top_by_cluster: Dict[int, float] = {}
    for row in penalty_rows:
        cid = int(row["cluster_id"])
        prob = float(row["avg_prob"])
        top_by_cluster[cid] = max(prob, top_by_cluster.get(cid, float("-inf")))
    avg_top_penalty_prob = 0.0 if not top_by_cluster else (sum(top_by_cluster.values()) / len(top_by_cluster))
    return {
        "avg_skip_active": avg_skip_active,
        "avg_skip_prob": avg_skip_prob,
        "avg_top_penalty_prob": avg_top_penalty_prob,
    }


def read_lambda_stats(csv_path: Path) -> Dict[str, float]:
    if not csv_path.exists():
        return {
            "avg_lambda_std": 0.0,
            "max_lambda_std": 0.0,
            "avg_lambda_mean": 0.0,
        }
    df = pd.read_csv(csv_path)
    if df.shape[0] == 0:
        return {
            "avg_lambda_std": 0.0,
            "max_lambda_std": 0.0,
            "avg_lambda_mean": 0.0,
        }
    return {
        "avg_lambda_std": float(df["lambda_std"].mean()),
        "max_lambda_std": float(df["lambda_std"].max()),
        "avg_lambda_mean": float(df["lambda_mean"].mean()),
    }


def read_run_metrics(run_dir: Path, label: str) -> Dict[str, Any]:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing run summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stats = read_skip_stats(run_dir / "cluster_penalty_probs.csv")
    lambda_stats = read_lambda_stats(run_dir / "cluster_lambda_stats.csv")
    penalty_names = summary.get("penalty_names", [])
    row = {
        "experiment": label,
        "run_dir": str(run_dir),
        "penalties": ",".join(penalty_names),
        "penalty_count": int(len(penalty_names)),
        "val_mse": float(summary.get("val", {}).get("avg_mse", float("nan"))),
        "test_mse": float(summary.get("test", {}).get("avg_mse", float("nan"))),
        "test_mae": float(summary.get("test", {}).get("avg_mae", float("nan"))),
        "best_epoch": ",".join(str(v) for v in summary.get("best_epoch", [])),
    }
    row.update(stats)
    row.update(lambda_stats)
    return row


def write_config(path: Path, cfg: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=False, sort_keys=False)


def run_train(config_path: Path) -> None:
    cmd = [sys.executable, "-m", "src.train", "--config", str(config_path)]
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-config",
        type=str,
        default="outputs/ETTm1/moe_ablation_20260420/configs/top1_route_skip_cost_0p15.yaml",
    )
    ap.add_argument(
        "--baseline-run-dir",
        type=str,
        default="outputs/ETTm1/moe_ablation_20260420/top1_route_skip_cost_0p15",
    )
    ap.add_argument("--out-root", type=str, default="outputs/ETTm1/penalty_lambda_ablation_20260420")
    ap.add_argument("--skip-run", action="store_true")
    args = ap.parse_args()

    base_config_path = resolve_path(args.base_config)
    baseline_run_dir = resolve_path(args.baseline_run_dir)
    out_root = resolve_path(args.out_root)
    cfg_root = out_root / "configs"
    out_root.mkdir(parents=True, exist_ok=True)

    base_cfg = load_yaml(str(base_config_path))

    experiments: List[Dict[str, Any]] = [
        {
            "name": "add_amp",
            "patch": {
                "moe": {
                    "lambda_init": {
                        "amp": 0.05,
                        "jump": 0.1,
                        "smooth": 0.1,
                        "level": 0.1,
                        "delta": 0.1,
                    },
                },
                "penalties": {
                    "enabled": ["amp", "jump", "smooth", "level", "delta"],
                },
            },
        },
        {
            "name": "add_amp_jitter",
            "patch": {
                "moe": {
                    "lambda_init": {
                        "amp": 0.05,
                        "jitter": 0.05,
                        "jump": 0.1,
                        "smooth": 0.1,
                        "level": 0.1,
                        "delta": 0.1,
                    },
                },
                "penalties": {
                    "enabled": ["amp", "jitter", "jump", "smooth", "level", "delta"],
                },
            },
        },
        {
            "name": "dynamic_multiscale_strong",
            "patch": {
                "moe": {
                    "dynamic_lambda": {
                        "enable": True,
                        "mode": "multiscale",
                        "hidden_dim": 64,
                        "segment_bins": [4, 8, 16],
                        "max_factor": 3.0,
                        "mix": 1.0,
                        "dropout": 0.0,
                        "reg_weight": 1.0e-5,
                    },
                },
            },
        },
        {
            "name": "add_amp_jitter_dynamic_multiscale_strong",
            "patch": {
                "moe": {
                    "lambda_init": {
                        "amp": 0.05,
                        "jitter": 0.05,
                        "jump": 0.1,
                        "smooth": 0.1,
                        "level": 0.1,
                        "delta": 0.1,
                    },
                    "dynamic_lambda": {
                        "enable": True,
                        "mode": "multiscale",
                        "hidden_dim": 64,
                        "segment_bins": [4, 8, 16],
                        "max_factor": 3.0,
                        "mix": 1.0,
                        "dropout": 0.0,
                        "reg_weight": 1.0e-5,
                    },
                },
                "penalties": {
                    "enabled": ["amp", "jitter", "jump", "smooth", "level", "delta"],
                },
            },
        },
        {
            "name": "dynamic_liquid",
            "patch": {
                "moe": {
                    "dynamic_lambda": {
                        "enable": True,
                        "mode": "liquid",
                        "hidden_dim": 48,
                        "max_factor": 3.0,
                        "mix": 1.0,
                        "dropout": 0.0,
                        "reg_weight": 1.0e-5,
                        "tau_min": 1.0,
                        "tau_max": 8.0,
                        "series_downsample_len": 64,
                    },
                },
            },
        },
    ]

    results: List[Dict[str, Any]] = [read_run_metrics(baseline_run_dir, label="baseline_top1_route_skip_cost_0p15")]

    for exp in experiments:
        name = str(exp["name"])
        out_dir = out_root / name
        cfg = copy.deepcopy(base_cfg)
        ensure_local_paths(cfg, out_dir)
        cfg.setdefault("exp", {})
        cfg["exp"]["name"] = f"penalty_lambda_ablation_{name}"
        deep_update(cfg, exp["patch"])
        cfg_path = cfg_root / f"{name}.yaml"
        write_config(cfg_path, cfg)
        print(f"[prepare] {name}: {cfg_path}")
        if not args.skip_run:
            print(f"[run] {name}")
            run_train(cfg_path)
        summary_path = out_dir / "run_summary.json"
        if summary_path.exists():
            results.append(read_run_metrics(out_dir, label=name))
        else:
            print(f"[skip] missing summary for {name}: {summary_path}")

    if len(results) == 0:
        print("Prepared configs only. Training was skipped.")
        return

    results_df = pd.DataFrame(results).sort_values(["test_mse", "val_mse", "experiment"]).reset_index(drop=True)
    results_path = out_root / "results.csv"
    summary_path = out_root / "summary.json"
    results_df.to_csv(results_path, index=False)
    summary = {
        "base_config": str(base_config_path),
        "baseline_run_dir": str(baseline_run_dir),
        "out_root": str(out_root),
        "experiments": [row["experiment"] for row in results],
        "best_by_test": results_df.iloc[0].to_dict() if results_df.shape[0] > 0 else None,
        "best_by_val": results_df.sort_values(["val_mse", "test_mse"]).iloc[0].to_dict() if results_df.shape[0] > 0 else None,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved results to: {results_path}")
    print(f"Saved summary to: {summary_path}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
