import argparse
import copy
import csv
import json
import os
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


def ensure_local_paths(cfg: Dict[str, Any], out_dir: Path, run_name: str) -> None:
    cfg.setdefault("exp", {})
    cfg["exp"]["name"] = run_name
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


def write_config(path: Path, cfg: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=False, sort_keys=False)


def run_train(config_path: Path) -> None:
    cmd = [sys.executable, "-m", "src.train", "--config", str(config_path)]
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True, env=env)


def read_run_metrics(run_dir: Path, name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing run summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    moe_cfg = cfg.get("moe", {})
    row = {
        "experiment": name,
        "run_dir": str(run_dir),
        "val_mse": float(summary.get("val", {}).get("avg_mse", float("nan"))),
        "test_mse": float(summary.get("test", {}).get("avg_mse", float("nan"))),
        "test_mae": float(summary.get("test", {}).get("avg_mae", float("nan"))),
        "best_epoch": ",".join(str(v) for v in summary.get("best_epoch", [])),
        "seed": int(cfg.get("exp", {}).get("seed", 0)),
        "deterministic": bool(cfg.get("exp", {}).get("deterministic", False)),
        "gate_hidden_dim": int(moe_cfg.get("gate_hidden_dim", 0)),
        "gate_noise_std": float(moe_cfg.get("gate_noise_std", 0.0)),
        "gate_temperature": float(moe_cfg.get("gate_temperature", 0.0)),
        "gate_entropy_weight": float(moe_cfg.get("gate_entropy_weight", 0.0)),
        "gate_balance_weight": float(moe_cfg.get("gate_balance_weight", 0.0)),
        "router_weight": float(moe_cfg.get("router_penalty_context_weight", 0.0)),
        "router_detach_penalty_context": bool(moe_cfg.get("router_detach_penalty_context", True)),
        "penalties": ",".join(cfg.get("penalties", {}).get("enabled", [])),
    }
    row.update(read_skip_stats(run_dir / "cluster_penalty_probs.csv"))
    return row


def build_variants() -> List[Dict[str, Any]]:
    return [
        {"name": "baseline", "patch": {}},
        {
            "name": "gate_h128",
            "patch": {"moe": {"gate_hidden_dim": 128}},
        },
        {
            "name": "gate_h128_router_w0p5",
            "patch": {
                "moe": {
                    "gate_hidden_dim": 128,
                    "router_penalty_context_weight": 0.5,
                }
            },
        },
        {
            "name": "gate_h128_router_w0p25",
            "patch": {
                "moe": {
                    "gate_hidden_dim": 128,
                    "router_penalty_context_weight": 0.25,
                }
            },
        },
        {
            "name": "gate_h128_router_w0p75",
            "patch": {
                "moe": {
                    "gate_hidden_dim": 128,
                    "router_penalty_context_weight": 0.75,
                }
            },
        },
        {
            "name": "gate_h128_router_w1",
            "patch": {
                "moe": {
                    "gate_hidden_dim": 128,
                    "router_penalty_context_weight": 1.0,
                }
            },
        },
        {
            "name": "gate_h128_router_w1p1",
            "patch": {
                "moe": {
                    "gate_hidden_dim": 128,
                    "router_penalty_context_weight": 1.1,
                }
            },
        },
        {
            "name": "gate_h128_router_w1p05",
            "patch": {
                "moe": {
                    "gate_hidden_dim": 128,
                    "router_penalty_context_weight": 1.05,
                }
            },
        },
        {
            "name": "gate_h128_router_w1p125",
            "patch": {
                "moe": {
                    "gate_hidden_dim": 128,
                    "router_penalty_context_weight": 1.125,
                }
            },
        },
        {
            "name": "gate_h128_router_w1p15",
            "patch": {
                "moe": {
                    "gate_hidden_dim": 128,
                    "router_penalty_context_weight": 1.15,
                }
            },
        },
        {
            "name": "gate_h128_router_w1p25",
            "patch": {
                "moe": {
                    "gate_hidden_dim": 128,
                    "router_penalty_context_weight": 1.25,
                }
            },
        },
        {
            "name": "gate_h128_router_w1p5",
            "patch": {
                "moe": {
                    "gate_hidden_dim": 128,
                    "router_penalty_context_weight": 1.5,
                }
            },
        },
        {
            "name": "gate_h128_router_w2p5",
            "patch": {
                "moe": {
                    "gate_hidden_dim": 128,
                    "router_penalty_context_weight": 2.5,
                }
            },
        },
        {
            "name": "gate_h128_learned_router",
            "patch": {
                "moe": {
                    "gate_hidden_dim": 128,
                    "router_mode": "learned",
                    "router_penalty_context_weight": 0.0,
                }
            },
        },
        {
            "name": "gate_h128_noise0p1_temp1p0",
            "patch": {
                "moe": {
                    "gate_hidden_dim": 128,
                    "gate_noise_std": 0.1,
                    "gate_temperature": 1.0,
                }
            },
        },
        {
            "name": "gate_h128_entropy_0p03",
            "patch": {
                "moe": {
                    "gate_hidden_dim": 128,
                    "gate_entropy_weight": 0.03,
                }
            },
        },
        {
            "name": "gate_h128_entropy_0p01",
            "patch": {
                "moe": {
                    "gate_hidden_dim": 128,
                    "gate_entropy_weight": 0.01,
                }
            },
        },
        {
            "name": "gate_h128_warmup_12",
            "patch": {
                "moe": {"gate_hidden_dim": 128},
                "train": {"penalty_warmup_epochs": 12},
            },
        },
        {
            "name": "gate_h128_no_skip",
            "patch": {
                "moe": {
                    "gate_hidden_dim": 128,
                    "allow_skip": False,
                }
            },
        },
        {
            "name": "gate_h128_noise0_temp1p2",
            "patch": {
                "moe": {
                    "gate_hidden_dim": 128,
                    "gate_noise_std": 0.0,
                    "gate_temperature": 1.2,
                }
            },
        },
        {
            "name": "gate_h128_level_delta",
            "patch": {
                "moe": {
                    "gate_hidden_dim": 128,
                    "lambda_init": {
                        "level": 0.1,
                        "delta": 0.1,
                    },
                },
                "penalties": {"enabled": ["level", "delta"]},
            },
        },
        {
            "name": "gate_h128_balance_0",
            "patch": {
                "moe": {
                    "gate_hidden_dim": 128,
                    "gate_balance_weight": 0.0,
                }
            },
        },
        {
            "name": "gate_h128_entropy_0p03_balance_0",
            "patch": {
                "moe": {
                    "gate_hidden_dim": 128,
                    "gate_entropy_weight": 0.03,
                    "gate_balance_weight": 0.0,
                }
            },
        },
        {
            "name": "router_no_detach",
            "patch": {"moe": {"router_detach_penalty_context": False}},
        },
        {
            "name": "entropy_0p03",
            "patch": {"moe": {"gate_entropy_weight": 0.03}},
        },
        {
            "name": "balance_0",
            "patch": {"moe": {"gate_balance_weight": 0.0}},
        },
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
                    }
                },
                "penalties": {"enabled": ["amp", "jump", "smooth", "level", "delta"]},
            },
        },
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", type=str, default="configs/ETTm1.yaml")
    ap.add_argument("--out-root", type=str, default="outputs/ETTm1/moe_optimize_20260420")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--skip-run", action="store_true")
    args = ap.parse_args()

    base_config_path = resolve_path(args.base_config)
    out_root = resolve_path(args.out_root)
    cfg_root = out_root / "configs"
    out_root.mkdir(parents=True, exist_ok=True)

    base_cfg = load_yaml(str(base_config_path))
    variants = build_variants()
    if args.only:
        allow = set(args.only)
        variants = [variant for variant in variants if variant["name"] in allow]
    results: List[Dict[str, Any]] = []

    for variant in variants:
        name = str(variant["name"])
        out_dir = out_root / name
        cfg = copy.deepcopy(base_cfg)
        ensure_local_paths(cfg, out_dir=out_dir, run_name=f"moe_optimize_{name}")
        deep_update(cfg, variant["patch"])
        cfg_path = cfg_root / f"{name}.yaml"
        write_config(cfg_path, cfg)
        print(f"[prepare] {name}: {cfg_path}")
        if not args.skip_run:
            print(f"[run] {name}")
            run_train(cfg_path)
        if (out_dir / "run_summary.json").exists():
            results.append(read_run_metrics(out_dir, name=name, cfg=cfg))
        else:
            print(f"[skip] missing summary for {name}: {out_dir / 'run_summary.json'}")

    if len(results) == 0:
        print("Prepared configs only. Training was skipped.")
        return

    results_df = pd.DataFrame(results).sort_values(["test_mse", "val_mse", "experiment"]).reset_index(drop=True)
    results_path = out_root / "results.csv"
    summary_path = out_root / "summary.json"
    results_df.to_csv(results_path, index=False)
    summary = {
        "base_config": str(base_config_path),
        "out_root": str(out_root),
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
