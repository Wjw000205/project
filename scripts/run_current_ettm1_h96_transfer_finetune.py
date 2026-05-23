from __future__ import annotations

import argparse
import copy
import csv
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_ettm1_finetune_transfer as base_transfer  # noqa: E402


SOURCE = "ETTm1"
TARGETS = ["ETTh1", "ETTh2", "ETTm2"]
HORIZON = 96


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)


def read_best_rows(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    rows: dict[tuple[str, int], dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("status")) == "ok" and str(row.get("is_best_for_cell")) == "True":
                rows[(str(row["dataset"]), int(row["horizon"]))] = row
    return rows


def run_cmd(cmd: list[str], *, cwd: Path, log_path: Path | None = None) -> None:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout)


def prepare_source_checkpoint(
    *,
    best_rows: dict[tuple[str, int], dict[str, str]],
    out_root: Path,
    device: str,
    py: str,
    rerun_source: bool,
) -> tuple[Path, Path]:
    source_row = best_rows[(SOURCE, HORIZON)]
    source_cfg_path = Path(source_row["config_path"])
    source_cfg = read_yaml(source_cfg_path)

    source_run_dir = out_root / "source" / f"{SOURCE}_H{HORIZON}_current"
    cfg_path = out_root / "configs" / "source" / f"{SOURCE}_H{HORIZON}_current_source.yaml"
    cfg = copy.deepcopy(source_cfg)
    cfg["exp"] = dict(cfg.get("exp", {}))
    cfg["exp"]["name"] = f"{SOURCE}_H{HORIZON}_current_transfer_source"
    cfg["exp"]["out_dir"] = str(source_run_dir)
    cfg["exp"]["seed"] = 2026
    cfg["exp"]["device"] = device
    cfg.setdefault("memory", {})
    cfg["memory"]["enable"] = True
    cfg["memory"]["save_checkpoint"] = True
    cfg["memory"]["path"] = str(source_run_dir / "cluster_memory.pt")
    cfg["memory"]["checkpoint_path"] = str(source_run_dir / "best_checkpoint.pt")
    cfg.setdefault("eval", {})
    cfg["eval"]["skip_test"] = False
    cfg.setdefault("knn_hybrid", {})
    cfg["knn_hybrid"]["enable"] = False
    cfg["knn_hybrid"]["use_for_model_selection"] = False
    cfg["calibration"] = {"enable": False}
    write_yaml(cfg_path, cfg)

    checkpoint_path = source_run_dir / "best_checkpoint.pt"
    memory_path = source_run_dir / "cluster_memory.pt"
    summary_path = source_run_dir / "run_summary.json"
    if rerun_source or not (checkpoint_path.exists() and memory_path.exists() and summary_path.exists()):
        print(f"[source] training {SOURCE} H{HORIZON} with checkpoint export", flush=True)
        run_cmd(
            [py, "-u", "-m", "src.train", "--config", str(cfg_path)],
            cwd=ROOT,
            log_path=source_run_dir / "source_train.log",
        )
    return cfg_path, source_run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-results", type=Path, default=ROOT / "outputs" / "ett_horizon_specific_moe_tune" / "best_results.csv")
    parser.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "current_module_transfer_rerun")
    parser.add_argument("--targets", type=str, default=",".join(TARGETS))
    parser.add_argument("--lrs", type=str, default="0.0001")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--resample-method", type=str, default="last", choices=["last", "ffill", "linear", "mean", "none"])
    parser.add_argument("--rerun-source", action="store_true")
    parser.add_argument("--rerun-transfer", action="store_true")
    args = parser.parse_args()

    best_rows = read_best_rows(args.best_results)
    missing = [(ds, HORIZON) for ds in [SOURCE, *TARGETS] if (ds, HORIZON) not in best_rows]
    if missing:
        raise FileNotFoundError(f"Missing best H=96 rows: {missing}")

    source_cfg_path, source_run_dir = prepare_source_checkpoint(
        best_rows=best_rows,
        out_root=args.out_root,
        device=args.device,
        py=str(args.python),
        rerun_source=args.rerun_source,
    )

    def source_config_path(horizon: int) -> Path:
        if horizon != HORIZON:
            raise ValueError(f"This wrapper only supports H={HORIZON}.")
        return source_cfg_path

    def source_run_dir_fn(horizon: int) -> Path:
        if horizon != HORIZON:
            raise ValueError(f"This wrapper only supports H={HORIZON}.")
        return source_run_dir

    def target_config_path(target: str, horizon: int) -> Path:
        if horizon != HORIZON:
            raise ValueError(f"This wrapper only supports H={HORIZON}.")
        return Path(best_rows[(target, HORIZON)]["config_path"])

    def target_summary_path(target: str, horizon: int) -> Path:
        if horizon != HORIZON:
            raise ValueError(f"This wrapper only supports H={HORIZON}.")
        return Path(best_rows[(target, HORIZON)]["out_dir"]) / "run_summary.json"

    base_transfer.source_config_path = source_config_path
    base_transfer.source_run_dir = source_run_dir_fn
    base_transfer.target_config_path = target_config_path
    base_transfer.target_summary_path = target_summary_path

    targets = [v.strip() for v in args.targets.split(",") if v.strip()]
    lrs = [float(v.strip()) for v in args.lrs.split(",") if v.strip()]
    result_path = args.out_root / "finetune_vs_zero_shot_h96.csv"
    rows: list[dict[str, Any]] = []
    if result_path.exists() and not args.rerun_transfer:
        with result_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    existing = {
        (row.get("target"), int(row.get("pred_len", -1)), str(row.get("finetune_lr")))
        for row in rows
        if row.get("status") == "ok" and row.get("finetune_lr") not in {None, ""}
    }

    for target in targets:
        for lr in lrs:
            key = (target, HORIZON, str(lr))
            if key in existing and not args.rerun_transfer:
                print(f"[skip] {SOURCE}->{target} H{HORIZON} lr={lr:g}", flush=True)
                continue
            try:
                new_rows = base_transfer.run_one(
                    target=target,
                    horizon=HORIZON,
                    lrs=[lr],
                    epochs=args.epochs,
                    out_root=args.out_root,
                    device=args.device,
                    py=str(args.python),
                    batch_size=args.batch_size,
                    resample_method=args.resample_method,
                    load_gate=True,
                    load_dynamic_lambda=True,
                    rerun=args.rerun_transfer,
                )
                rows.extend(new_rows)
            except Exception as exc:
                rows.append(
                    {
                        "status": "error",
                        "source": SOURCE,
                        "target": target,
                        "pred_len": HORIZON,
                        "finetune_lr": lr,
                        "error": str(exc)[-4000:],
                    }
                )
            base_transfer.write_rows(result_path, rows)

    base_transfer.write_rows(result_path, rows)
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
