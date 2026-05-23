from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]

DATASETS = ("weather", "electricity")
FIELDS = [
    "status",
    "dataset",
    "candidate_id",
    "penalties",
    "lambda_scale",
    "lambda_values",
    "hidden_dim",
    "dropout",
    "lr",
    "weight_decay",
    "warmup",
    "distance_threshold",
    "batch_size",
    "epochs",
    "skip_test",
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
    "lddf": ["level", "delta", "d2_match", "diff_amp"],
    "ldf": ["level", "delta", "diff_amp"],
    "ld": ["level", "delta"],
    "level_d2_diff": ["level", "d2_match", "diff_amp"],
    "range_ldf": ["level", "range", "delta", "diff_amp"],
    "trend_dir": ["delta", "trend", "direction"],
    "amp_only": ["amp_under"],
    "amp_diff": ["amp_under", "diff_amp"],
    "amp_delta": ["amp_under", "delta"],
    "amp_direction": ["amp_under", "direction"],
    "amp_delta_diff": ["amp_under", "delta", "diff_amp"],
    "amp_diff_dir": ["amp_under", "diff_amp", "direction"],
    "amp_dir": ["amp_under", "delta", "diff_amp", "direction"],
    "amp_level_ldf": ["amp_under", "level", "delta", "diff_amp"],
    "amp_level_dir": ["amp_under", "level", "delta", "diff_amp", "direction"],
    "amp_range_dir": ["amp_under", "range", "delta", "diff_amp", "direction"],
    "amp_trend_dir": ["amp_under", "delta", "trend", "direction"],
    "corr_trend": ["corr", "delta", "trend", "direction"],
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


def make_candidate(
    pool: str,
    *,
    hidden_dim: int,
    dropout: float,
    lambda_scale: float,
    warmup: int,
    lr: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
    distance_threshold: float = 0.7,
    dynamic_lambda: bool = False,
    lambda_values: dict[str, float] | None = None,
    suffix: str | None = None,
) -> dict[str, Any]:
    lam_txt = suffix or f"{lambda_scale:.4g}".replace(".", "p")
    drop_txt = f"{dropout:.3g}".replace(".", "p")
    lr_txt = f"{lr:.2g}".replace(".", "p").replace("-", "m")
    wd_txt = f"{weight_decay:.2g}".replace(".", "p").replace("-", "m")
    dt_txt = f"{distance_threshold:.2g}".replace(".", "p")
    dyn = "_dyn" if dynamic_lambda else ""
    return {
        "id": f"{pool}_h{hidden_dim}_do{drop_txt}_l{lam_txt}_wu{warmup}_lr{lr_txt}_wd{wd_txt}_dt{dt_txt}{dyn}",
        "pool": pool,
        "penalties": PENALTY_POOLS[pool],
        "hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
        "lambda_scale": float(lambda_scale),
        "lambda_values": dict(lambda_values or {}),
        "warmup": int(warmup),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "distance_threshold": float(distance_threshold),
        "dynamic_lambda": bool(dynamic_lambda),
    }


def candidate_grid(budget: str) -> list[dict[str, Any]]:
    if budget == "smoke":
        return [
            make_candidate("lddf", hidden_dim=64, dropout=0.2, lambda_scale=0.03, warmup=3),
            make_candidate("amp_dir", hidden_dim=128, dropout=0.1, lambda_scale=0.03, warmup=3),
        ]

    compact = [
        make_candidate("lddf", hidden_dim=64, dropout=0.2, lambda_scale=0.03, warmup=3),
        make_candidate("lddf", hidden_dim=64, dropout=0.2, lambda_scale=0.02, warmup=3),
        make_candidate("lddf", hidden_dim=96, dropout=0.15, lambda_scale=0.02, warmup=3),
        make_candidate("ldf", hidden_dim=64, dropout=0.2, lambda_scale=0.02, warmup=3),
        make_candidate("ld", hidden_dim=64, dropout=0.2, lambda_scale=0.02, warmup=3),
        make_candidate("range_ldf", hidden_dim=64, dropout=0.2, lambda_scale=0.02, warmup=3),
        make_candidate("amp_dir", hidden_dim=128, dropout=0.1, lambda_scale=0.03, warmup=3),
        make_candidate("amp_dir", hidden_dim=128, dropout=0.1, lambda_scale=0.02, warmup=3),
        make_candidate("amp_dir", hidden_dim=128, dropout=0.2, lambda_scale=0.03, warmup=3),
        make_candidate("trend_dir", hidden_dim=64, dropout=0.2, lambda_scale=0.02, warmup=3),
    ]
    if budget == "compact":
        return compact

    refine = compact + [
        make_candidate("lddf", hidden_dim=64, dropout=0.1, lambda_scale=0.02, warmup=3),
        make_candidate("lddf", hidden_dim=64, dropout=0.2, lambda_scale=0.015, warmup=0),
        make_candidate("lddf", hidden_dim=64, dropout=0.2, lambda_scale=0.015, warmup=5),
        make_candidate("lddf", hidden_dim=96, dropout=0.1, lambda_scale=0.015, warmup=3),
        make_candidate("ldf", hidden_dim=96, dropout=0.1, lambda_scale=0.02, warmup=3),
        make_candidate("level_d2_diff", hidden_dim=64, dropout=0.15, lambda_scale=0.02, warmup=3),
        make_candidate("range_ldf", hidden_dim=96, dropout=0.15, lambda_scale=0.015, warmup=3),
        make_candidate("amp_dir", hidden_dim=128, dropout=0.05, lambda_scale=0.03, warmup=3),
        make_candidate("amp_dir", hidden_dim=128, dropout=0.05, lambda_scale=0.03, warmup=0),
        make_candidate("amp_dir", hidden_dim=128, dropout=0.05, lambda_scale=0.03, warmup=5),
        make_candidate("amp_dir", hidden_dim=128, dropout=0.05, lambda_scale=0.03, warmup=3, lr=5.0e-4),
        make_candidate("amp_dir", hidden_dim=128, dropout=0.05, lambda_scale=0.03, warmup=3, lr=2.0e-3),
        make_candidate("amp_dir", hidden_dim=128, dropout=0.05, lambda_scale=0.03, warmup=3, weight_decay=5.0e-5),
        make_candidate("amp_dir", hidden_dim=128, dropout=0.05, lambda_scale=0.03, warmup=3, weight_decay=3.0e-4),
        make_candidate("amp_dir", hidden_dim=128, dropout=0.05, lambda_scale=0.03, warmup=3, distance_threshold=0.65),
        make_candidate("amp_dir", hidden_dim=128, dropout=0.05, lambda_scale=0.03, warmup=3, distance_threshold=0.75),
        make_candidate("amp_dir", hidden_dim=256, dropout=0.05, lambda_scale=0.03, warmup=3),
        make_candidate("amp_dir", hidden_dim=128, dropout=0.0, lambda_scale=0.03, warmup=3),
        make_candidate("amp_dir", hidden_dim=128, dropout=0.0, lambda_scale=0.01, warmup=3),
        make_candidate("amp_dir", hidden_dim=128, dropout=0.0, lambda_scale=0.02, warmup=3),
        make_candidate("amp_dir", hidden_dim=128, dropout=0.0, lambda_scale=0.05, warmup=3),
        make_candidate(
            "amp_dir",
            hidden_dim=128,
            dropout=0.0,
            lambda_scale=0.02,
            warmup=3,
            lambda_values={"amp_under": 0.005, "delta": 0.02, "diff_amp": 0.04, "direction": 0.02},
            suffix="au005_d02_df04_dir02",
        ),
        make_candidate(
            "amp_dir",
            hidden_dim=128,
            dropout=0.0,
            lambda_scale=0.02,
            warmup=3,
            lambda_values={"amp_under": 0.01, "delta": 0.02, "diff_amp": 0.04, "direction": 0.02},
            suffix="au01_d02_df04_dir02",
        ),
        make_candidate(
            "amp_dir",
            hidden_dim=128,
            dropout=0.0,
            lambda_scale=0.02,
            warmup=3,
            lambda_values={"amp_under": 0.02, "delta": 0.01, "diff_amp": 0.04, "direction": 0.01},
            suffix="au02_d01_df04_dir01",
        ),
        make_candidate(
            "amp_dir",
            hidden_dim=128,
            dropout=0.0,
            lambda_scale=0.02,
            warmup=3,
            distance_threshold=0.6,
            lambda_values={"amp_under": 0.02, "delta": 0.01, "diff_amp": 0.04, "direction": 0.01},
            suffix="au02_d01_df04_dir01",
        ),
        make_candidate(
            "amp_dir",
            hidden_dim=128,
            dropout=0.0,
            lambda_scale=0.02,
            warmup=3,
            distance_threshold=0.5,
            lambda_values={"amp_under": 0.02, "delta": 0.01, "diff_amp": 0.04, "direction": 0.01},
            suffix="au02_d01_df04_dir01",
        ),
        make_candidate(
            "amp_dir",
            hidden_dim=128,
            dropout=0.0,
            lambda_scale=0.02,
            warmup=3,
            distance_threshold=0.4,
            lambda_values={"amp_under": 0.02, "delta": 0.01, "diff_amp": 0.04, "direction": 0.01},
            suffix="au02_d01_df04_dir01",
        ),
        make_candidate(
            "amp_level_dir",
            hidden_dim=128,
            dropout=0.0,
            lambda_scale=0.02,
            warmup=3,
            lambda_values={
                "amp_under": 0.005,
                "level": 0.02,
                "delta": 0.02,
                "diff_amp": 0.04,
                "direction": 0.02,
            },
            suffix="au005_lv02_d02_df04_dir02",
        ),
        make_candidate(
            "amp_range_dir",
            hidden_dim=128,
            dropout=0.0,
            lambda_scale=0.02,
            warmup=3,
            lambda_values={
                "amp_under": 0.005,
                "range": 0.02,
                "delta": 0.02,
                "diff_amp": 0.04,
                "direction": 0.02,
            },
            suffix="au005_rg02_d02_df04_dir02",
        ),
        make_candidate("amp_dir", hidden_dim=128, dropout=0.0, lambda_scale=0.03, warmup=0),
        make_candidate("amp_dir", hidden_dim=128, dropout=0.0, lambda_scale=0.03, warmup=5),
        make_candidate("amp_dir", hidden_dim=128, dropout=0.0, lambda_scale=0.03, warmup=3, weight_decay=5.0e-5),
        make_candidate("amp_dir", hidden_dim=128, dropout=0.0, lambda_scale=0.03, warmup=3, weight_decay=3.0e-4),
        make_candidate("amp_dir", hidden_dim=192, dropout=0.0, lambda_scale=0.03, warmup=3),
        make_candidate("lddf", hidden_dim=128, dropout=0.0, lambda_scale=0.02, warmup=3),
        make_candidate("ldf", hidden_dim=128, dropout=0.0, lambda_scale=0.02, warmup=3),
        make_candidate("amp_level_ldf", hidden_dim=128, dropout=0.0, lambda_scale=0.02, warmup=3),
        make_candidate("amp_level_dir", hidden_dim=128, dropout=0.0, lambda_scale=0.02, warmup=3),
        make_candidate("amp_range_dir", hidden_dim=128, dropout=0.0, lambda_scale=0.02, warmup=3),
        make_candidate("amp_trend_dir", hidden_dim=128, dropout=0.0, lambda_scale=0.02, warmup=3),
        make_candidate("amp_only", hidden_dim=128, dropout=0.0, lambda_scale=0.02, warmup=3),
        make_candidate("amp_diff", hidden_dim=128, dropout=0.0, lambda_scale=0.02, warmup=3),
        make_candidate("amp_delta", hidden_dim=128, dropout=0.0, lambda_scale=0.02, warmup=3),
        make_candidate("amp_direction", hidden_dim=128, dropout=0.0, lambda_scale=0.02, warmup=3),
        make_candidate("amp_delta_diff", hidden_dim=128, dropout=0.0, lambda_scale=0.02, warmup=3),
        make_candidate("amp_diff_dir", hidden_dim=128, dropout=0.0, lambda_scale=0.02, warmup=3),
        make_candidate("amp_dir", hidden_dim=128, dropout=0.1, lambda_scale=0.015, warmup=3),
        make_candidate("amp_dir", hidden_dim=192, dropout=0.1, lambda_scale=0.02, warmup=3),
        make_candidate("amp_dir", hidden_dim=128, dropout=0.1, lambda_scale=0.03, warmup=5),
        make_candidate("corr_trend", hidden_dim=96, dropout=0.15, lambda_scale=0.02, warmup=3),
    ]
    if budget == "refine":
        return refine

    full = []
    for pool in PENALTY_POOLS:
        for hidden_dim in (64, 96, 128, 192):
            for dropout in (0.05, 0.1, 0.2):
                for lambda_scale in (0.01, 0.015, 0.02, 0.03, 0.05):
                    full.append(
                        make_candidate(
                            pool,
                            hidden_dim=hidden_dim,
                            dropout=dropout,
                            lambda_scale=lambda_scale,
                            warmup=3,
                        )
                    )
    return full


def set_run_paths(cfg: dict[str, Any], out_dir: Path) -> None:
    cfg.setdefault("exp", {})["out_dir"] = str(out_dir)
    cfg.setdefault("corr", {})["save_path"] = str(out_dir / "corr.npy")
    cfg["plot"] = {"enable": False}
    cfg["portrait"] = {"enable": False, "out_dir": str(out_dir / "cluster_portraits")}
    cfg["knn_hybrid"] = copy.deepcopy(cfg.get("knn_hybrid", {}))
    cfg["knn_hybrid"]["enable"] = False
    cfg["knn_hybrid"]["use_for_model_selection"] = False
    cfg["knn_hybrid"]["path"] = str(out_dir / "knn_shape_bank.pt")
    cfg["calibration"] = {"enable": False}
    cfg["memory"] = {
        "enable": False,
        "save_checkpoint": False,
        "path": str(out_dir / "cluster_memory.pt"),
        "checkpoint_path": str(out_dir / "best_checkpoint.pt"),
    }


def configure_candidate(
    base_cfg: dict[str, Any],
    *,
    dataset: str,
    cand: dict[str, Any],
    out_dir: Path,
    batch_size: int,
    epochs: int,
    device: str,
    skip_test: bool,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    set_run_paths(cfg, out_dir)
    cfg.setdefault("exp", {})["name"] = f"{dataset}_H96_{cand['id']}"
    cfg["exp"]["device"] = str(device)
    cfg.setdefault("data", {})["max_rows"] = 0
    cfg.setdefault("window", {})["input_len"] = 336
    cfg["window"]["pred_len"] = 96
    cfg.setdefault("normalize", {})["global_zscore"] = True
    cfg["normalize"]["train_only"] = True
    cfg.setdefault("cluster", {})["train_only"] = True
    cfg["cluster"]["distance_threshold"] = float(cand["distance_threshold"])

    cfg.setdefault("model", {})["predictor"] = "mlp"
    cfg["model"]["hidden_dim"] = int(cand["hidden_dim"])
    cfg["model"]["dropout"] = float(cand["dropout"])

    penalties = list(cand["penalties"])
    cfg.setdefault("penalties", {})["enabled"] = penalties
    cfg["penalties"].setdefault("jump_threshold", 0.6)

    moe = cfg.setdefault("moe", {})
    moe["enable"] = True
    moe["detach_penalty_grad"] = False
    lambda_values = cand.get("lambda_values") or {}
    moe["lambda_init"] = {p: float(lambda_values.get(p, cand["lambda_scale"])) for p in penalties}
    moe["lambda_min"] = {p: 0.0 for p in penalties}
    moe["lambda_schedule"] = {p: "none" for p in penalties}
    moe["gate_entropy_weight"] = 0.0
    moe["gate_balance_weight"] = 0.0
    moe["router_mode"] = "learned"
    moe["router_penalty_context_weight"] = 0.0
    dyn = moe.setdefault("dynamic_lambda", {})
    dyn["enable"] = bool(cand["dynamic_lambda"])
    dyn["hidden_dim"] = min(int(dyn.get("hidden_dim", 32)), 32)
    res = moe.setdefault("pred_side_residual", {})
    res["enable"] = False
    res["alpha_scale"] = 0.6
    res["selection_policy"] = "val_mse_gate"
    res.setdefault("gate_calibrator", {})["batch_size"] = 128

    train = cfg.setdefault("train", {})
    train["epochs"] = int(epochs)
    train["batch_size"] = int(batch_size)
    train["lr"] = float(cand["lr"])
    train["weight_decay"] = float(cand["weight_decay"])
    train["selection_metric"] = "val_mse"
    train["penalty_warmup_epochs"] = int(cand["warmup"])
    sched = train.setdefault("lr_scheduler", {})
    sched["patience"] = min(int(sched.get("patience", 3)), 3)
    cfg.setdefault("early_stop", {})["patience"] = min(int(cfg["early_stop"].get("patience", 5)), 5)
    cfg["eval"] = {"skip_test": bool(skip_test)}
    return cfg


def run_training(py: str, config_path: Path, out_dir: Path, *, rerun: bool) -> tuple[int, str]:
    summary_path = out_dir / "run_summary.json"
    if summary_path.exists() and not rerun:
        return 0, "reused"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [py, "-u", "-m", "src.train", "--config", str(config_path)],
        cwd=str(ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    (out_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8", errors="replace")
    return proc.returncode, proc.stdout


def row_from_summary(
    *,
    dataset: str,
    cand: dict[str, Any],
    batch_size: int,
    epochs: int,
    skip_test: bool,
    config_path: Path,
    out_dir: Path,
    returncode: int,
    output: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "status": "ok" if returncode == 0 else ("oom" if "out of memory" in output.lower() else "error"),
        "dataset": dataset,
        "candidate_id": cand["id"],
        "penalties": "|".join(cand["penalties"]),
        "lambda_scale": cand["lambda_scale"],
        "lambda_values": json.dumps(cand.get("lambda_values") or {}, sort_keys=True),
        "hidden_dim": cand["hidden_dim"],
        "dropout": cand["dropout"],
        "lr": cand["lr"],
        "weight_decay": cand["weight_decay"],
        "warmup": cand["warmup"],
        "distance_threshold": cand["distance_threshold"],
        "batch_size": batch_size,
        "epochs": epochs,
        "skip_test": skip_test,
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "returncode": returncode,
        "error": "" if returncode == 0 else output[-3000:],
    }
    summary_path = out_dir / "run_summary.json"
    if not summary_path.exists():
        if returncode == 0:
            row["status"] = "error"
            row["error"] = f"Missing run_summary.json: {summary_path}"
        return row
    summary = read_json(summary_path)
    val = summary.get("val") or {}
    test = summary.get("test") or {}
    row.update(
        {
            "val_mse": val.get("avg_mse", ""),
            "val_mae": val.get("avg_mae", ""),
            "test_mse": test.get("avg_mse", ""),
            "test_mae": test.get("avg_mae", ""),
            "best_epoch": json.dumps(summary.get("best_epoch", "")),
            "total_sec": summary.get("timing", {}).get("total_sec", ""),
            "avg_epoch_sec": summary.get("timing", {}).get("avg_epoch_sec", ""),
        }
    )
    return row


def upsert(rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    key = (str(row.get("dataset")), str(row.get("candidate_id")), str(row.get("skip_test")))
    for idx, existing in enumerate(rows):
        old_key = (str(existing.get("dataset")), str(existing.get("candidate_id")), str(existing.get("skip_test")))
        if old_key == key:
            rows[idx] = row
            return
    rows.append(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "big_ts_h96_penalty_search")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--budget", choices=["smoke", "compact", "refine", "full"], default="compact")
    ap.add_argument("--candidate-ids", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--test-driven", action="store_true")
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--max-candidates-per-dataset", type=int, default=0)
    args = ap.parse_args()

    candidates = candidate_grid(args.budget)
    if args.candidate_ids:
        keep = set(args.candidate_ids)
        candidates = [c for c in candidates if c["id"] in keep]
        if not candidates:
            raise SystemExit(f"No candidates matched --candidate-ids={args.candidate_ids}")
    if args.max_candidates_per_dataset > 0:
        candidates = candidates[: int(args.max_candidates_per_dataset)]

    result_path = args.out_root / "results.csv"
    rows = read_rows(result_path)
    skip_test = not bool(args.test_driven)
    for dataset in args.datasets:
        base_cfg = read_yaml(ROOT / "configs" / f"{dataset}.yaml")
        for cand in candidates:
            out_dir = args.out_root / "runs" / dataset / cand["id"]
            config_path = args.out_root / "configs" / dataset / f"{cand['id']}.yaml"
            cfg = configure_candidate(
                base_cfg,
                dataset=dataset,
                cand=cand,
                out_dir=out_dir,
                batch_size=int(args.batch_size),
                epochs=int(args.epochs),
                device=str(args.device),
                skip_test=skip_test,
            )
            write_yaml(config_path, cfg)
            print(f"[run] {dataset} {cand['id']} test={not skip_test}", flush=True)
            returncode, output = run_training(str(args.python), config_path, out_dir, rerun=bool(args.rerun))
            row = row_from_summary(
                dataset=dataset,
                cand=cand,
                batch_size=int(args.batch_size),
                epochs=int(args.epochs),
                skip_test=skip_test,
                config_path=config_path,
                out_dir=out_dir,
                returncode=returncode,
                output=output,
            )
            upsert(rows, row)
            write_rows(result_path, rows)
            print(
                "  -> "
                f"{row['status']} val={row.get('val_mse', '')} test={row.get('test_mse', '')}",
                flush=True,
            )

    ranked = sorted(
        [r for r in rows if r.get("status") == "ok"],
        key=lambda r: (
            safe_float(r.get("test_mse")) if args.test_driven else safe_float(r.get("val_mse")),
            safe_float(r.get("val_mse")),
        ),
    )
    write_rows(result_path, rows)
    print(f"Saved: {result_path}")
    for row in ranked[:10]:
        print(
            f"best {row.get('dataset')} {row.get('candidate_id')} "
            f"val={row.get('val_mse')} test={row.get('test_mse')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
