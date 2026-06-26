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


def ensure_local_paths(cfg: Dict[str, Any], out_dir: Path, seed: int, run_name: str) -> None:
    cfg.setdefault("exp", {})
    cfg["exp"]["name"] = run_name
    cfg["exp"]["out_dir"] = str(out_dir)
    cfg["exp"]["seed"] = int(seed)
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


def read_run_metrics(run_dir: Path, variant: str, seed: int) -> Dict[str, Any]:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing run summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    row = {
        "variant": variant,
        "seed": int(seed),
        "run_dir": str(run_dir),
        "val_mse": float(summary.get("val", {}).get("avg_mse", float("nan"))),
        "test_mse": float(summary.get("test", {}).get("avg_mse", float("nan"))),
        "test_mae": float(summary.get("test", {}).get("avg_mae", float("nan"))),
        "best_epoch": ",".join(str(v) for v in summary.get("best_epoch", [])),
    }
    row.update(read_skip_stats(run_dir / "cluster_penalty_probs.csv"))
    return row


def write_config(path: Path, cfg: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=False, sort_keys=False)


def run_train(config_path: Path) -> None:
    cmd = [sys.executable, "-m", "src.train", "--config", str(config_path)]
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def build_variants() -> List[Dict[str, Any]]:
    return [
        {"name": "default", "patch": {}},
        {"name": "patience10", "patch": {"early_stop": {"patience": 10}}},
        {"name": "sched5", "patch": {"train": {"lr_scheduler": {"patience": 5}}}},
        {"name": "warmup15", "patch": {"train": {"penalty_warmup_epochs": 15}}},
        {
            "name": "patience10_sched5",
            "patch": {"early_stop": {"patience": 10}, "train": {"lr_scheduler": {"patience": 5}}},
        },
        {
            "name": "patience10_warmup15",
            "patch": {"early_stop": {"patience": 10}, "train": {"penalty_warmup_epochs": 15}},
        },
        {
            "name": "patience10_sched5_warmup15",
            "patch": {
                "early_stop": {"patience": 10},
                "train": {"lr_scheduler": {"patience": 5}, "penalty_warmup_epochs": 15},
            },
        },
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-config",
        type=str,
        default="outputs/ETTm1/moe_router_refine_20260420/configs/context_w2_balance_0p01.yaml",
    )
    ap.add_argument("--out-root", type=str, default="outputs/ETTm1/moe_stability_20260420")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2025, 2026])
    ap.add_argument("--skip-run", action="store_true")
    args = ap.parse_args()

    base_config_path = resolve_path(args.base_config)
    out_root = resolve_path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cfg_root = out_root / "configs"

    base_cfg = load_yaml(str(base_config_path))
    variants = build_variants()
    results: List[Dict[str, Any]] = []

    for seed in args.seeds:
        for variant in variants:
            variant_name = variant["name"]
            run_name = f"{variant_name}_seed_{seed}"
            out_dir = out_root / run_name
            cfg = copy.deepcopy(base_cfg)
            deep_update(cfg, variant["patch"])
            ensure_local_paths(cfg, out_dir=out_dir, seed=seed, run_name=f"moe_stability_{run_name}")
            cfg_path = cfg_root / f"{run_name}.yaml"
            write_config(cfg_path, cfg)
            print(f"[prepare] {run_name}: {cfg_path}")
            if not args.skip_run:
                print(f"[run] {run_name}")
                run_train(cfg_path)
            summary_path = out_dir / "run_summary.json"
            if summary_path.exists():
                results.append(read_run_metrics(out_dir, variant=variant_name, seed=seed))
            else:
                print(f"[skip] missing summary for {run_name}: {summary_path}")

    if len(results) == 0:
        print("Prepared configs only. Training was skipped.")
        return

    results_df = pd.DataFrame(results).sort_values(["variant", "seed"]).reset_index(drop=True)
    agg_df = (
        results_df.groupby("variant", as_index=False)
        .agg(
            seed_count=("seed", "count"),
            val_mse_mean=("val_mse", "mean"),
            val_mse_std=("val_mse", "std"),
            test_mse_mean=("test_mse", "mean"),
            test_mse_std=("test_mse", "std"),
            test_mae_mean=("test_mae", "mean"),
            test_mae_std=("test_mae", "std"),
            avg_top_penalty_prob_mean=("avg_top_penalty_prob", "mean"),
            avg_skip_prob_mean=("avg_skip_prob", "mean"),
        )
        .sort_values(["test_mse_mean", "val_mse_mean"])
        .reset_index(drop=True)
    )

    per_seed_best = (
        results_df.sort_values(["seed", "test_mse", "val_mse"])
        .groupby("seed", as_index=False)
        .first()
        .to_dict(orient="records")
    )

    summary = {
        "base_config": str(base_config_path),
        "out_root": str(out_root),
        "seeds": list(args.seeds),
        "variants": [v["name"] for v in variants],
        "best_by_mean_test": agg_df.iloc[0].to_dict() if agg_df.shape[0] > 0 else None,
        "per_seed_best": per_seed_best,
    }

    results_path = out_root / "results.csv"
    agg_path = out_root / "aggregate.csv"
    summary_path = out_root / "summary.json"
    results_df.to_csv(results_path, index=False)
    agg_df.to_csv(agg_path, index=False)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved results to: {results_path}")
    print(f"Saved aggregate to: {agg_path}")
    print(f"Saved summary to: {summary_path}")
    print(results_df.to_string(index=False))
    print(agg_df.to_string(index=False))


if __name__ == "__main__":
    main()
