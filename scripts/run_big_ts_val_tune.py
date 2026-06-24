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

DATASETS = ["electricity", "traffic", "weather"]
HORIZONS = [96]

FIELDS = [
    "status",
    "dataset",
    "pred_len",
    "candidate_id",
    "penalties",
    "lambda_scale",
    "alpha_scale",
    "hidden_dim",
    "dropout",
    "batch_size",
    "dynamic_lambda",
    "residual_enable",
    "epochs",
    "max_rows",
    "normalize_train_only",
    "cluster_train_only",
    "skip_test",
    "plot_enable",
    "portrait_enable",
    "val_mse",
    "val_mae",
    "best_epoch",
    "total_sec",
    "avg_epoch_sec",
    "config_path",
    "out_dir",
    "returncode",
    "error",
]


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})


def run_cmd(cmd: list[str], log_path: Path) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout, encoding="utf-8")
    return proc.returncode, proc.stdout


def parse_list(raw: str, cast=str) -> list[Any]:
    return [cast(v.strip()) for v in raw.split(",") if v.strip()]


def candidate_grid(budget: str) -> list[dict[str, Any]]:
    penalties = {
        "lddf": ["level", "delta", "d2_match", "diff_amp"],
        "trend": ["delta", "trend", "direction"],
        "range_trend": ["level", "range", "trend", "direction"],
        "amp_dir": ["amp_under", "delta", "diff_amp", "direction"],
    }
    if budget == "smoke":
        return [
            {
                "id": "smoke_lddf_h64_l003_a06",
                "penalties": penalties["lddf"],
                "lambda_scale": 0.03,
                "alpha_scale": 0.6,
                "hidden_dim": 64,
                "dropout": 0.2,
                "dynamic_lambda": False,
                "residual_enable": False,
            }
        ]
    if budget == "compact":
        return [
            {
                "id": "lddf_h64_l003_a06",
                "penalties": penalties["lddf"],
                "lambda_scale": 0.03,
                "alpha_scale": 0.6,
                "hidden_dim": 64,
                "dropout": 0.2,
                "dynamic_lambda": False,
                "residual_enable": False,
            },
            {
                "id": "trend_h64_l003_a06",
                "penalties": penalties["trend"],
                "lambda_scale": 0.03,
                "alpha_scale": 0.6,
                "hidden_dim": 64,
                "dropout": 0.2,
                "dynamic_lambda": False,
                "residual_enable": False,
            },
            {
                "id": "range_trend_h64_l001_a035",
                "penalties": penalties["range_trend"],
                "lambda_scale": 0.01,
                "alpha_scale": 0.35,
                "hidden_dim": 64,
                "dropout": 0.3,
                "dynamic_lambda": False,
                "residual_enable": False,
            },
            {
                "id": "amp_dir_h128_l003_a06",
                "penalties": penalties["amp_dir"],
                "lambda_scale": 0.03,
                "alpha_scale": 0.6,
                "hidden_dim": 128,
                "dropout": 0.2,
                "dynamic_lambda": False,
                "residual_enable": False,
            },
            {
                "id": "lddf_h128_l01_dyn",
                "penalties": penalties["lddf"],
                "lambda_scale": 0.1,
                "alpha_scale": 0.6,
                "hidden_dim": 128,
                "dropout": 0.2,
                "dynamic_lambda": True,
                "residual_enable": False,
            },
        ]
    if budget == "refine":
        return [
            {
                "id": "lddf_h64_l001_a06",
                "penalties": penalties["lddf"],
                "lambda_scale": 0.01,
                "alpha_scale": 0.6,
                "hidden_dim": 64,
                "dropout": 0.2,
                "dynamic_lambda": False,
                "residual_enable": False,
            },
            {
                "id": "lddf_h64_l003_a06",
                "penalties": penalties["lddf"],
                "lambda_scale": 0.03,
                "alpha_scale": 0.6,
                "hidden_dim": 64,
                "dropout": 0.2,
                "dynamic_lambda": False,
                "residual_enable": False,
            },
            {
                "id": "lddf_h64_l006_a06",
                "penalties": penalties["lddf"],
                "lambda_scale": 0.06,
                "alpha_scale": 0.6,
                "hidden_dim": 64,
                "dropout": 0.2,
                "dynamic_lambda": False,
                "residual_enable": False,
            },
            {
                "id": "lddf_h64_l003_do01",
                "penalties": penalties["lddf"],
                "lambda_scale": 0.03,
                "alpha_scale": 0.6,
                "hidden_dim": 64,
                "dropout": 0.1,
                "dynamic_lambda": False,
                "residual_enable": False,
            },
            {
                "id": "lddf_h128_l003_a06",
                "penalties": penalties["lddf"],
                "lambda_scale": 0.03,
                "alpha_scale": 0.6,
                "hidden_dim": 128,
                "dropout": 0.2,
                "dynamic_lambda": False,
                "residual_enable": False,
            },
            {
                "id": "lddf_h192_l003_a06",
                "penalties": penalties["lddf"],
                "lambda_scale": 0.03,
                "alpha_scale": 0.6,
                "hidden_dim": 192,
                "dropout": 0.2,
                "dynamic_lambda": False,
                "residual_enable": False,
            },
            {
                "id": "amp_dir_h128_l001_a06",
                "penalties": penalties["amp_dir"],
                "lambda_scale": 0.01,
                "alpha_scale": 0.6,
                "hidden_dim": 128,
                "dropout": 0.2,
                "dynamic_lambda": False,
                "residual_enable": False,
            },
            {
                "id": "amp_dir_h128_l003_a06",
                "penalties": penalties["amp_dir"],
                "lambda_scale": 0.03,
                "alpha_scale": 0.6,
                "hidden_dim": 128,
                "dropout": 0.2,
                "dynamic_lambda": False,
                "residual_enable": False,
            },
            {
                "id": "amp_dir_h128_l006_a06",
                "penalties": penalties["amp_dir"],
                "lambda_scale": 0.06,
                "alpha_scale": 0.6,
                "hidden_dim": 128,
                "dropout": 0.2,
                "dynamic_lambda": False,
                "residual_enable": False,
            },
            {
                "id": "amp_dir_h128_l01_a06",
                "penalties": penalties["amp_dir"],
                "lambda_scale": 0.1,
                "alpha_scale": 0.6,
                "hidden_dim": 128,
                "dropout": 0.2,
                "dynamic_lambda": False,
                "residual_enable": False,
            },
            {
                "id": "amp_dir_h128_l003_do01",
                "penalties": penalties["amp_dir"],
                "lambda_scale": 0.03,
                "alpha_scale": 0.6,
                "hidden_dim": 128,
                "dropout": 0.1,
                "dynamic_lambda": False,
                "residual_enable": False,
            },
            {
                "id": "amp_dir_h192_l003_a06",
                "penalties": penalties["amp_dir"],
                "lambda_scale": 0.03,
                "alpha_scale": 0.6,
                "hidden_dim": 192,
                "dropout": 0.2,
                "dynamic_lambda": False,
                "residual_enable": False,
            },
            {
                "id": "lddf_h128_l003_dyn",
                "penalties": penalties["lddf"],
                "lambda_scale": 0.03,
                "alpha_scale": 0.6,
                "hidden_dim": 128,
                "dropout": 0.2,
                "dynamic_lambda": True,
                "residual_enable": False,
            },
            {
                "id": "lddf_h128_l006_dyn",
                "penalties": penalties["lddf"],
                "lambda_scale": 0.06,
                "alpha_scale": 0.6,
                "hidden_dim": 128,
                "dropout": 0.2,
                "dynamic_lambda": True,
                "residual_enable": False,
            },
        ]
    grid: list[dict[str, Any]] = []
    for name, pool in penalties.items():
        for hidden_dim in [64, 128, 256]:
            for lambda_scale in [0.01, 0.03, 0.1]:
                grid.append(
                    {
                        "id": f"{name}_h{hidden_dim}_l{str(lambda_scale).replace('.', 'p')}",
                        "penalties": pool,
                        "lambda_scale": lambda_scale,
                        "alpha_scale": 0.6,
                        "hidden_dim": hidden_dim,
                        "dropout": 0.2,
                        "dynamic_lambda": hidden_dim >= 128,
                        "residual_enable": False,
                    }
                )
    return grid


def apply_candidate(
    base: dict[str, Any],
    *,
    dataset: str,
    horizon: int,
    cand: dict[str, Any],
    out_dir: Path,
    batch_size: int,
    epochs: int,
    max_rows: int,
    device: str,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg["exp"] = {
        "name": f"{dataset}_H{horizon}_{cand['id']}",
        "out_dir": str(out_dir),
        "seed": int(base.get("exp", {}).get("seed", 2026)),
        "deterministic": True,
        "device": device,
    }
    cfg.setdefault("data", {})
    cfg["data"]["max_rows"] = int(max_rows)
    cfg["data"]["train_ratio"] = float(cfg["data"].get("train_ratio", 0.7))
    cfg["data"]["val_ratio"] = float(cfg["data"].get("val_ratio", 0.1))
    cfg["data"]["test_ratio"] = float(cfg["data"].get("test_ratio", 0.2))
    cfg.setdefault("window", {})
    cfg["window"]["input_len"] = int(cfg["window"].get("input_len", 336))
    cfg["window"]["pred_len"] = int(horizon)
    cfg.setdefault("normalize", {})
    cfg["normalize"]["global_zscore"] = True
    cfg["normalize"]["train_only"] = True
    cfg.setdefault("cluster", {})
    cfg["cluster"]["train_only"] = True
    cfg.setdefault("corr", {})
    cfg["corr"]["compute"] = True
    cfg["corr"]["save_path"] = str(out_dir / "corr.npy")
    cfg.setdefault("model", {})
    cfg["model"]["predictor"] = "mlp"
    cfg["model"]["hidden_dim"] = int(cand["hidden_dim"])
    cfg["model"]["dropout"] = float(cand["dropout"])
    cfg.setdefault("penalties", {})
    cfg["penalties"]["enabled"] = list(cand["penalties"])
    cfg["penalties"].setdefault("jump_threshold", 0.6)

    moe = cfg.setdefault("moe", {})
    moe["enable"] = True
    moe["gate_hidden_dim"] = min(int(moe.get("gate_hidden_dim", 32)), 32)
    moe["gate_entropy_weight"] = 0.0
    moe["gate_balance_weight"] = 0.0
    moe["router_mode"] = "learned"
    moe["router_penalty_context_weight"] = 0.0
    moe["detach_penalty_grad"] = False
    moe["lambda_init"] = {p: float(cand["lambda_scale"]) for p in cand["penalties"]}
    moe["lambda_min"] = {p: 0.0 for p in cand["penalties"]}
    moe["lambda_schedule"] = {p: "none" for p in cand["penalties"]}
    dyn = moe.setdefault("dynamic_lambda", {})
    dyn["enable"] = bool(cand["dynamic_lambda"])
    dyn["hidden_dim"] = min(int(dyn.get("hidden_dim", 32)), 32)
    dyn["dropout"] = 0.0
    res = moe.setdefault("pred_side_residual", {})
    res["enable"] = bool(cand["residual_enable"])
    res["corrector_hidden"] = 16
    res["alpha_scale"] = float(cand["alpha_scale"])
    res["selection_policy"] = "val_mse_candidate_channel"
    cal["epochs"] = 10
    cal["batch_size"] = 128

    train = cfg.setdefault("train", {})
    train["epochs"] = int(epochs)
    train["batch_size"] = int(batch_size)
    train["selection_metric"] = "val_mse"
    train["lr"] = float(train.get("lr", 0.001))
    train["weight_decay"] = float(train.get("weight_decay", 0.0001))
    train["penalty_warmup_epochs"] = min(int(train.get("penalty_warmup_epochs", 10)), 5)
    sched = train.setdefault("lr_scheduler", {})
    sched["patience"] = min(int(sched.get("patience", 3)), 3)
    cfg.setdefault("early_stop", {})
    cfg["early_stop"]["patience"] = min(int(cfg["early_stop"].get("patience", 10)), 5)
    cfg["early_stop"]["min_delta"] = float(cfg["early_stop"].get("min_delta", 1.0e-6))
    cfg["eval"] = {"skip_test": True}
    cfg["plot"] = {"enable": False}
    cfg["portrait"] = {"enable": False, "out_dir": str(out_dir / "cluster_portraits")}
    cfg["memory"] = {
        "enable": False,
        "save_checkpoint": False,
        "path": str(out_dir / "cluster_memory.pt"),
        "checkpoint_path": str(out_dir / "best_checkpoint.pt"),
    }
    return cfg


def run_candidate(
    *,
    dataset: str,
    horizon: int,
    cand: dict[str, Any],
    out_root: Path,
    py: str,
    batch_size: int,
    epochs: int,
    max_rows: int,
    device: str,
    rerun: bool,
) -> dict[str, Any]:
    base = read_yaml(ROOT / "configs" / f"{dataset}.yaml")
    out_dir = out_root / "runs" / dataset / f"pred_{horizon}" / cand["id"]
    cfg_path = out_root / "configs" / dataset / f"pred_{horizon}_{cand['id']}.yaml"
    cfg = apply_candidate(
        base,
        dataset=dataset,
        horizon=horizon,
        cand=cand,
        out_dir=out_dir,
        batch_size=batch_size,
        epochs=epochs,
        max_rows=max_rows,
        device=device,
    )
    write_yaml(cfg_path, cfg)
    summary_path = out_dir / "run_summary.json"
    log_path = out_dir / "stdout.log"
    returncode = 0
    output = ""
    if rerun or not summary_path.exists():
        returncode, output = run_cmd([py, "-u", "-m", "src.train", "--config", str(cfg_path)], log_path)

    row = {
        "status": "ok",
        "dataset": dataset,
        "pred_len": horizon,
        "candidate_id": cand["id"],
        "penalties": "|".join(cand["penalties"]),
        "lambda_scale": cand["lambda_scale"],
        "alpha_scale": cand["alpha_scale"],
        "hidden_dim": cand["hidden_dim"],
        "dropout": cand["dropout"],
        "batch_size": batch_size,
        "dynamic_lambda": cand["dynamic_lambda"],
        "residual_enable": cand["residual_enable"],
        "epochs": epochs,
        "max_rows": max_rows,
        "normalize_train_only": True,
        "cluster_train_only": True,
        "skip_test": True,
        "plot_enable": False,
        "portrait_enable": False,
        "config_path": str(cfg_path),
        "out_dir": str(out_dir),
        "returncode": returncode,
    }
    if returncode != 0:
        err = output[-4000:]
        row["status"] = "oom" if "out of memory" in output.lower() else "error"
        row["error"] = err
        return row
    if not summary_path.exists():
        row["status"] = "error"
        row["error"] = f"Missing run_summary.json: {summary_path}"
        return row
    summary = load_json(summary_path)
    row.update(
        {
            "val_mse": summary.get("val", {}).get("avg_mse", ""),
            "val_mae": summary.get("val", {}).get("avg_mae", ""),
            "best_epoch": json.dumps(summary.get("best_epoch", "")),
            "total_sec": summary.get("timing", {}).get("total_sec", ""),
            "avg_epoch_sec": summary.get("timing", {}).get("avg_epoch_sec", ""),
        }
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "electricity_traffic_val_tune")
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS))
    ap.add_argument("--horizons", type=str, default=",".join(str(v) for v in HORIZONS))
    ap.add_argument("--budget", type=str, default="compact", choices=["smoke", "compact", "refine", "full"])
    ap.add_argument("--candidate-ids", type=str, default="")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--python", type=Path, default=Path(sys.executable))
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()

    datasets = parse_list(args.datasets, str)
    horizons = parse_list(args.horizons, int)
    candidates = candidate_grid(args.budget)
    if args.candidate_ids.strip():
        keep = set(parse_list(args.candidate_ids, str))
        candidates = [c for c in candidates if c["id"] in keep]
        if not candidates:
            raise ValueError(f"No candidates matched --candidate-ids={args.candidate_ids!r}")
    args.out_root.mkdir(parents=True, exist_ok=True)
    result_path = args.out_root / "results.csv"
    rows: list[dict[str, Any]] = []
    if result_path.exists() and not args.rerun:
        with result_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    done = {
        (r.get("dataset"), int(r.get("pred_len", -1)), r.get("candidate_id"))
        for r in rows
        if r.get("status") == "ok"
    }
    py = str(args.python)
    for horizon in horizons:
        for dataset in datasets:
            for cand in candidates:
                key = (dataset, horizon, cand["id"])
                if key in done and not args.rerun:
                    print(f"[skip] {dataset} H{horizon} {cand['id']}", flush=True)
                    continue
                print(f"[run] {dataset} H{horizon} {cand['id']}", flush=True)
                try:
                    row = run_candidate(
                        dataset=dataset,
                        horizon=horizon,
                        cand=cand,
                        out_root=args.out_root,
                        py=py,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        max_rows=args.max_rows,
                        device=args.device,
                        rerun=args.rerun,
                    )
                except Exception as exc:
                    row = {
                        "status": "error",
                        "dataset": dataset,
                        "pred_len": horizon,
                        "candidate_id": cand["id"],
                        "error": str(exc)[-4000:],
                    }
                rows.append(row)
                write_rows(result_path, rows)
    write_rows(result_path, rows)
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
