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
DATASETS = ("weather", "traffic")

FIELDS = [
    "status",
    "dataset",
    "candidate_id",
    "penalties",
    "hidden_dim",
    "dropout",
    "lambda_values",
    "lr",
    "weight_decay",
    "warmup",
    "batch_size",
    "epochs",
    "distance_threshold",
    "val_mse",
    "val_mae",
    "test_mse",
    "test_mae",
    "best_epoch",
    "total_sec",
    "avg_epoch_sec",
    "config_path",
    "out_dir",
    "returncode",
    "error",
]


PENALTY_POOLS: dict[str, list[str]] = {
    "amp_dir": ["amp_under", "delta", "diff_amp", "direction"],
    "lddf": ["level", "delta", "d2_match", "diff_amp"],
    "ldf": ["level", "delta", "diff_amp"],
    "amp_delta_diff": ["amp_under", "delta", "diff_amp"],
    "amp_range_dir": ["amp_under", "range", "delta", "diff_amp", "direction"],
}


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def safe_float(value: Any, default: float = 999.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def candidate(
    pool: str,
    *,
    hidden_dim: int,
    dropout: float,
    lambda_values: dict[str, float] | float,
    lr: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
    warmup: int = 3,
    distance_threshold: float = 0.7,
    tag: str | None = None,
) -> dict[str, Any]:
    penalties = PENALTY_POOLS[pool]
    if isinstance(lambda_values, dict):
        lambdas = {p: float(lambda_values.get(p, 0.0)) for p in penalties}
        lam_tag = tag or "_".join(f"{p[:2]}{v:g}" for p, v in lambdas.items()).replace(".", "p")
    else:
        lambdas = {p: float(lambda_values) for p in penalties}
        lam_tag = tag or f"l{float(lambda_values):g}".replace(".", "p")
    drop_tag = f"do{dropout:g}".replace(".", "p")
    lr_tag = f"lr{lr:g}".replace(".", "p").replace("-", "m")
    wd_tag = f"wd{weight_decay:g}".replace(".", "p").replace("-", "m")
    dt_tag = f"dt{distance_threshold:g}".replace(".", "p")
    return {
        "id": f"{pool}_h{hidden_dim}_{drop_tag}_{lam_tag}_wu{warmup}_{lr_tag}_{wd_tag}_{dt_tag}",
        "pool": pool,
        "penalties": penalties,
        "hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
        "lambda_values": lambdas,
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "warmup": int(warmup),
        "distance_threshold": float(distance_threshold),
    }


def default_candidates(dataset: str) -> list[dict[str, Any]]:
    if dataset == "weather":
        return [
            candidate("amp_dir", hidden_dim=128, dropout=0.2, lambda_values=0.01),
            candidate("amp_dir", hidden_dim=128, dropout=0.15, lambda_values=0.01),
            candidate("amp_dir", hidden_dim=128, dropout=0.1, lambda_values=0.01),
            candidate("amp_dir", hidden_dim=160, dropout=0.15, lambda_values=0.01),
            candidate("amp_dir", hidden_dim=128, dropout=0.2, lambda_values=0.005),
            candidate("amp_dir", hidden_dim=128, dropout=0.2, lambda_values=0.02),
            candidate("lddf", hidden_dim=96, dropout=0.2, lambda_values=0.02),
            candidate("amp_range_dir", hidden_dim=128, dropout=0.15, lambda_values=0.01),
        ]
    if dataset == "traffic":
        return [
            candidate("amp_dir", hidden_dim=192, dropout=0.2, lambda_values=0.03),
            candidate("amp_dir", hidden_dim=192, dropout=0.1, lambda_values=0.03),
            candidate("amp_dir", hidden_dim=256, dropout=0.2, lambda_values=0.03),
            candidate("amp_dir", hidden_dim=192, dropout=0.2, lambda_values=0.02),
            candidate("amp_dir", hidden_dim=192, dropout=0.2, lambda_values=0.05),
            candidate("amp_delta_diff", hidden_dim=192, dropout=0.2, lambda_values=0.03),
            candidate(
                "amp_dir",
                hidden_dim=192,
                dropout=0.2,
                lambda_values={"amp_under": 0.02, "delta": 0.03, "diff_amp": 0.05, "direction": 0.02},
                tag="au02_d03_df05_dir02",
            ),
            candidate("lddf", hidden_dim=192, dropout=0.2, lambda_values=0.03),
        ]
    raise ValueError(f"Unknown dataset: {dataset}")


def configure(
    base_cfg: dict[str, Any],
    *,
    dataset: str,
    cand: dict[str, Any],
    out_dir: Path,
    device: str,
    batch_size: int,
    epochs: int,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg["exp"] = {
        "name": f"{dataset}_H96_{cand['id']}",
        "out_dir": str(out_dir),
        "seed": int(base_cfg.get("exp", {}).get("seed", 2026)),
        "deterministic": True,
        "device": device,
    }
    cfg.setdefault("window", {})["input_len"] = 336
    cfg["window"]["pred_len"] = 96
    cfg["window"]["past_context"] = bool(cfg["window"].get("past_context", True))
    cfg.setdefault("normalize", {})["global_zscore"] = True
    cfg["normalize"]["train_only"] = True
    cfg.setdefault("cluster", {})["train_only"] = True
    cfg["cluster"]["method"] = "leader"
    cfg["cluster"]["distance_threshold"] = float(cand["distance_threshold"])
    cfg.setdefault("corr", {})["compute"] = True
    cfg["corr"]["save_path"] = str(out_dir / "corr.npy")
    cfg.setdefault("model", {})["predictor"] = "mlp"
    cfg["model"]["hidden_dim"] = int(cand["hidden_dim"])
    cfg["model"]["dropout"] = float(cand["dropout"])

    penalties = list(cand["penalties"])
    cfg.setdefault("penalties", {})["enabled"] = penalties
    cfg["penalties"].setdefault("jump_threshold", 0.6)
    moe = cfg.setdefault("moe", {})
    moe["enable"] = True
    moe["detach_penalty_grad"] = False
    moe["topk"] = int(moe.get("topk", 1))
    moe["select_ranks"] = [1]
    moe["lambda_init"] = {p: float(cand["lambda_values"][p]) for p in penalties}
    moe["lambda_min"] = {p: 0.0 for p in penalties}
    moe["lambda_schedule"] = {p: "none" for p in penalties}
    moe["gate_entropy_weight"] = 0.0
    moe["gate_balance_weight"] = 0.0
    moe["router_mode"] = "learned"
    moe["router_penalty_context_weight"] = 0.0
    moe["gate_temperature"] = float(moe.get("gate_temperature", 1.2))
    moe["gate_noise_std"] = float(moe.get("gate_noise_std", 0.2))
    dyn = moe.setdefault("dynamic_lambda", {})
    dyn["enable"] = False
    res = moe.setdefault("pred_side_residual", {})
    res["enable"] = False

    train = cfg.setdefault("train", {})
    train["epochs"] = int(epochs)
    train["batch_size"] = int(batch_size)
    train["lr"] = float(cand["lr"])
    train["weight_decay"] = float(cand["weight_decay"])
    train["selection_metric"] = "val_mse"
    train["penalty_warmup_epochs"] = int(cand["warmup"])
    train.setdefault("mse_weight", 0.9)
    sched = train.setdefault("lr_scheduler", {})
    sched["name"] = "plateau"
    sched["factor"] = float(sched.get("factor", 0.5))
    sched["patience"] = min(int(sched.get("patience", 3)), 3)
    sched["min_lr"] = float(sched.get("min_lr", 1.0e-6))
    cfg.setdefault("early_stop", {})["patience"] = min(int(cfg["early_stop"].get("patience", 5)), 5)
    cfg["early_stop"]["min_delta"] = float(cfg["early_stop"].get("min_delta", 1.0e-6))
    cfg["eval"] = {"skip_test": False}
    cfg["plot"] = {"enable": False}
    cfg["portrait"] = {"enable": False, "out_dir": str(out_dir / "cluster_portraits")}
    cfg["memory"] = {
        "enable": False,
        "save_checkpoint": False,
        "path": str(out_dir / "cluster_memory.pt"),
        "checkpoint_path": str(out_dir / "best_checkpoint.pt"),
    }
    return cfg


def run_cmd(cmd: list[str], *, cwd: Path, log_path: Path) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    return proc.returncode, proc.stdout


def run_one(
    *,
    dataset: str,
    cand: dict[str, Any],
    out_root: Path,
    py: str,
    device: str,
    batch_size: int,
    epochs: int,
    rerun: bool,
) -> dict[str, Any]:
    out_dir = out_root / "runs" / dataset / cand["id"]
    cfg_path = out_root / "configs" / dataset / f"{cand['id']}.yaml"
    cfg = configure(
        read_yaml(ROOT / "configs" / f"{dataset}.yaml"),
        dataset=dataset,
        cand=cand,
        out_dir=out_dir,
        device=device,
        batch_size=batch_size,
        epochs=epochs,
    )
    write_yaml(cfg_path, cfg)
    summary_path = out_dir / "run_summary.json"
    output = "reused"
    returncode = 0
    if rerun or not summary_path.exists():
        returncode, output = run_cmd(
            [py, "-u", "-m", "src.train", "--config", str(cfg_path)],
            cwd=ROOT,
            log_path=out_dir / "stdout.log",
        )
    row: dict[str, Any] = {
        "status": "ok" if returncode == 0 else ("oom" if "out of memory" in output.lower() else "error"),
        "dataset": dataset,
        "candidate_id": cand["id"],
        "penalties": "|".join(cand["penalties"]),
        "hidden_dim": cand["hidden_dim"],
        "dropout": cand["dropout"],
        "lambda_values": json.dumps(cand["lambda_values"], sort_keys=True),
        "lr": cand["lr"],
        "weight_decay": cand["weight_decay"],
        "warmup": cand["warmup"],
        "batch_size": batch_size,
        "epochs": epochs,
        "distance_threshold": cand["distance_threshold"],
        "config_path": str(cfg_path),
        "out_dir": str(out_dir),
        "returncode": returncode,
        "error": "" if returncode == 0 else output[-3000:],
    }
    if not summary_path.exists():
        if returncode == 0:
            row["status"] = "error"
            row["error"] = f"Missing run_summary.json: {summary_path}"
        return row
    summary = read_json(summary_path)
    row.update(
        {
            "val_mse": summary.get("val", {}).get("avg_mse", ""),
            "val_mae": summary.get("val", {}).get("avg_mae", ""),
            "test_mse": summary.get("test", {}).get("avg_mse", ""),
            "test_mae": summary.get("test", {}).get("avg_mae", ""),
            "best_epoch": json.dumps(summary.get("best_epoch", "")),
            "total_sec": summary.get("timing", {}).get("total_sec", ""),
            "avg_epoch_sec": summary.get("timing", {}).get("avg_epoch_sec", ""),
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "weather_traffic_h96_param_tune")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--candidate-ids", nargs="*", default=None)
    parser.add_argument("--weather-epochs", type=int, default=30)
    parser.add_argument("--traffic-epochs", type=int, default=16)
    parser.add_argument("--weather-batch-size", type=int, default=64)
    parser.add_argument("--traffic-batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    result_path = args.out_root / "results.csv"
    rows = read_rows(result_path)
    done = {(r.get("dataset"), r.get("candidate_id")) for r in rows if r.get("status") == "ok"}
    for dataset in args.datasets:
        candidates = default_candidates(dataset)
        if args.candidate_ids:
            keep = set(args.candidate_ids)
            candidates = [c for c in candidates if c["id"] in keep]
        epochs = args.weather_epochs if dataset == "weather" else args.traffic_epochs
        batch_size = args.weather_batch_size if dataset == "weather" else args.traffic_batch_size
        for cand in candidates:
            if (dataset, cand["id"]) in done and not args.rerun:
                print(f"[skip] {dataset} {cand['id']}", flush=True)
                continue
            print(f"[run] {dataset} {cand['id']}", flush=True)
            try:
                row = run_one(
                    dataset=dataset,
                    cand=cand,
                    out_root=args.out_root,
                    py=str(args.python),
                    device=args.device,
                    batch_size=batch_size,
                    epochs=epochs,
                    rerun=args.rerun,
                )
            except Exception as exc:
                row = {
                    "status": "error",
                    "dataset": dataset,
                    "candidate_id": cand["id"],
                    "error": str(exc)[-3000:],
                }
            rows = [r for r in rows if not (r.get("dataset") == dataset and r.get("candidate_id") == cand["id"])]
            rows.append(row)
            write_rows(result_path, rows)
            print(f"  -> {row.get('status')} val={row.get('val_mse', '')} test={row.get('test_mse', '')}", flush=True)

    write_rows(result_path, rows)
    ranked = sorted(
        [r for r in rows if r.get("status") == "ok"],
        key=lambda r: (str(r.get("dataset")), safe_float(r.get("test_mse")), safe_float(r.get("val_mse"))),
    )
    print(f"Saved: {result_path}")
    for row in ranked[:20]:
        print(
            f"best {row['dataset']} {row['candidate_id']} "
            f"test={row.get('test_mse')} val={row.get('val_mse')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
