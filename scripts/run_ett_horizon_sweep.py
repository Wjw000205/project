import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
DEFAULT_HORIZONS = [96, 192, 336, 720]
DEFAULT_CONFIGS = {
    "ETTh1": "configs/ETTh1.yaml",
    "ETTh2": "configs/ETTh2.yaml",
    "ETTm1": "configs/ETTm1.yaml",
    "ETTm2": "configs/ETTm2.yaml",
}

CSV_FIELDS = [
    "status",
    "dataset",
    "pred_len",
    "input_len",
    "data_csv",
    "base_config",
    "run_config",
    "out_dir",
    "base_test_mse",
    "base_test_mae",
    "hybrid_test_mse",
    "hybrid_test_mae",
    "hybrid_delta_mse_vs_base",
    "hybrid_gain_mse_vs_base",
    "hybrid_delta_mae_vs_base",
    "hybrid_gain_mae_vs_base",
    "selected_variant",
    "selected_test_mse",
    "selected_test_mae",
    "val_base_mse",
    "val_base_mae",
    "val_hybrid_mse",
    "val_hybrid_mae",
    "selected_policy",
    "moe_residual_variant",
    "best_epoch",
    "total_sec",
    "avg_epoch_sec",
    "wrapper_sec",
    "returncode",
    "error",
]


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=False, sort_keys=False)


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_tail(path: Path, max_chars: int = 2000) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[-max_chars:].strip()


def configure_paths(
    cfg: dict[str, Any],
    *,
    dataset: str,
    pred_len: int,
    out_dir: Path,
    keep_artifacts: bool,
    save_checkpoint: bool,
) -> None:
    run_name = f"{dataset}_pred_{pred_len}"
    cfg.setdefault("exp", {})
    cfg["exp"]["name"] = run_name
    cfg["exp"]["out_dir"] = str(out_dir)

    cfg.setdefault("corr", {})
    cfg["corr"]["save_path"] = str(out_dir / "corr.npy")

    cfg.setdefault("plot", {})
    cfg.setdefault("portrait", {})
    if not keep_artifacts:
        cfg["plot"]["enable"] = False
        cfg["portrait"]["enable"] = False
    cfg["portrait"]["out_dir"] = str(out_dir / "cluster_portraits")

    cfg.setdefault("memory", {})
    cfg["memory"]["path"] = str(out_dir / "cluster_memory.pt")
    cfg["memory"]["checkpoint_path"] = str(out_dir / "best_checkpoint.pt")
    if not keep_artifacts:
        cfg["memory"]["enable"] = False
        cfg["memory"]["save_checkpoint"] = False
    if save_checkpoint:
        cfg["memory"]["save_checkpoint"] = True

    cfg.setdefault("knn_hybrid", {})
    cfg["knn_hybrid"]["path"] = str(out_dir / "knn_shape_bank.pt")


def configure_hybrid(
    cfg: dict[str, Any],
    *,
    enable_hybrid: bool,
    selection_policy: str,
    adaptive_alpha: str | None,
) -> None:
    knn = cfg.setdefault("knn_hybrid", {})
    knn["enable"] = bool(enable_hybrid)
    knn["use_for_model_selection"] = False
    knn.setdefault("mode", "rolling")
    knn.setdefault("bank_split", "history")
    knn.setdefault("feature_mode", "joint")
    knn.setdefault("template_mode", "residual")
    knn.setdefault("selection_min_rel_improvement", 0.0)
    knn.setdefault("selection_max_rel_mse_regression", 0.03)
    knn.setdefault("distance_weight", "inverse")
    knn.setdefault("anchor_mode", "last")
    knn.setdefault("bank_chunk_size", 8192)
    knn["selection_policy"] = selection_policy
    if adaptive_alpha is not None:
        knn["adaptive_alpha"] = adaptive_alpha


def make_config(
    base_cfg: dict[str, Any],
    *,
    dataset: str,
    base_config: Path,
    pred_len: int,
    input_len: int | None,
    out_root: Path,
    epochs: int | None,
    device: str | None,
    keep_artifacts: bool,
    save_checkpoint: bool,
    enable_hybrid: bool,
    hybrid_selection_policy: str,
    hybrid_adaptive_alpha: str | None,
) -> tuple[dict[str, Any], Path, Path]:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("window", {})
    if input_len is not None:
        cfg["window"]["input_len"] = int(input_len)
    cfg["window"]["pred_len"] = int(pred_len)

    cfg.setdefault("eval", {})
    cfg["eval"]["skip_test"] = False

    if epochs is not None:
        cfg.setdefault("train", {})
        cfg["train"]["epochs"] = int(epochs)
    if device:
        cfg.setdefault("exp", {})
        cfg["exp"]["device"] = device

    out_dir = out_root / "runs" / dataset / f"pred_{pred_len}"
    config_path = out_root / "configs" / f"{dataset}_pred_{pred_len}.yaml"
    configure_paths(
        cfg,
        dataset=dataset,
        pred_len=pred_len,
        out_dir=out_dir,
        keep_artifacts=keep_artifacts,
        save_checkpoint=save_checkpoint,
    )
    configure_hybrid(
        cfg,
        enable_hybrid=enable_hybrid,
        selection_policy=hybrid_selection_policy,
        adaptive_alpha=hybrid_adaptive_alpha,
    )
    return cfg, config_path, out_dir


def run_train(config_path: Path, out_dir: Path, show_stdout: bool) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = out_dir / "stdout.log"
    stderr_path = out_dir / "stderr.log"
    cmd = [sys.executable, "-m", "src.train", "--config", str(config_path)]
    env = os.environ.copy()
    env.setdefault("MOELOSS_PROGRESS_LEAVE", "0")
    if show_stdout:
        stdout_path.write_text("stdout was streamed to console.\n", encoding="utf-8")
        with stderr_path.open("w", encoding="utf-8") as stderr_f:
            completed = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, stderr=stderr_f, env=env)
    else:
        with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
            completed = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                text=True,
                stdout=stdout_f,
                stderr=stderr_f,
                env=env,
            )
    return int(completed.returncode)


def empty_row(
    *,
    status: str,
    dataset: str,
    pred_len: int,
    input_len: int,
    data_csv: str,
    base_config: Path,
    run_config: Path,
    out_dir: Path,
    wrapper_sec: float,
    returncode: int,
    error: str = "",
) -> dict[str, Any]:
    row = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "status": status,
            "dataset": dataset,
            "pred_len": int(pred_len),
            "input_len": int(input_len),
            "data_csv": data_csv,
            "base_config": str(base_config),
            "run_config": str(run_config),
            "out_dir": str(out_dir),
            "wrapper_sec": wrapper_sec,
            "returncode": int(returncode),
            "error": error,
        }
    )
    return row


def summary_to_row(
    summary_path: Path,
    *,
    status: str,
    dataset: str,
    pred_len: int,
    input_len: int,
    data_csv: str,
    base_config: Path,
    run_config: Path,
    out_dir: Path,
    wrapper_sec: float,
    returncode: int,
    error: str = "",
) -> dict[str, Any]:
    row = empty_row(
        status=status,
        dataset=dataset,
        pred_len=pred_len,
        input_len=input_len,
        data_csv=data_csv,
        base_config=base_config,
        run_config=run_config,
        out_dir=out_dir,
        wrapper_sec=wrapper_sec,
        returncode=returncode,
        error=error,
    )
    if not summary_path.exists():
        return row

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    test = summary.get("test") or {}
    hybrid = summary.get("test_hybrid") or {}
    val = summary.get("val") or {}
    val_hybrid = summary.get("val_hybrid") or {}
    selected = summary.get("selected") or {}
    timing = summary.get("timing") or {}

    base_mse = as_float(test.get("avg_mse"))
    base_mae = as_float(test.get("avg_mae"))
    hybrid_mse = as_float(hybrid.get("avg_mse"))
    hybrid_mae = as_float(hybrid.get("avg_mae"))
    row.update(
        {
            "base_test_mse": test.get("avg_mse", ""),
            "base_test_mae": test.get("avg_mae", ""),
            "hybrid_test_mse": hybrid.get("avg_mse", ""),
            "hybrid_test_mae": hybrid.get("avg_mae", ""),
            "selected_variant": selected.get("variant", ""),
            "selected_test_mse": selected.get("avg_mse", ""),
            "selected_test_mae": selected.get("avg_mae", ""),
            "val_base_mse": val.get("avg_mse", ""),
            "val_base_mae": val.get("avg_mae", ""),
            "val_hybrid_mse": val_hybrid.get("avg_mse", ""),
            "val_hybrid_mae": val_hybrid.get("avg_mae", ""),
            "selected_policy": selected.get("selection_policy", ""),
            "moe_residual_variant": selected.get("moe_residual_variant", ""),
            "best_epoch": ",".join(str(v) for v in summary.get("best_epoch", [])),
            "total_sec": timing.get("total_sec", ""),
            "avg_epoch_sec": timing.get("avg_epoch_sec", ""),
        }
    )
    if base_mse is not None and hybrid_mse is not None:
        row["hybrid_delta_mse_vs_base"] = hybrid_mse - base_mse
        row["hybrid_gain_mse_vs_base"] = base_mse - hybrid_mse
    if base_mae is not None and hybrid_mae is not None:
        row["hybrid_delta_mae_vs_base"] = hybrid_mae - base_mae
        row["hybrid_gain_mae_vs_base"] = base_mae - hybrid_mae
    if row["selected_variant"] == "":
        row["selected_variant"] = "base"
        row["selected_test_mse"] = row["base_test_mse"]
        row["selected_test_mae"] = row["base_test_mae"]
    return row


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def upsert_row(rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    key = (str(row["dataset"]), str(row["pred_len"]))
    for idx, existing in enumerate(rows):
        if (str(existing.get("dataset", "")), str(existing.get("pred_len", ""))) == key:
            rows[idx] = row
            return
    rows.append(row)


def write_wide(path: Path, rows: list[dict[str, Any]], horizons: list[int]) -> None:
    fields = ["dataset"]
    for horizon in horizons:
        fields.extend(
            [
                f"base_mse_{horizon}",
                f"base_mae_{horizon}",
                f"hybrid_mse_{horizon}",
                f"hybrid_mae_{horizon}",
                f"hybrid_gain_mse_{horizon}",
                f"hybrid_gain_mae_{horizon}",
                f"selected_variant_{horizon}",
                f"selected_mse_{horizon}",
                f"selected_mae_{horizon}",
                f"status_{horizon}",
            ]
        )
    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        dataset = str(row.get("dataset", ""))
        if dataset not in grouped:
            grouped[dataset] = {"dataset": dataset}
            order.append(dataset)
        horizon = row.get("pred_len", "")
        entry = grouped[dataset]
        entry[f"base_mse_{horizon}"] = row.get("base_test_mse", "")
        entry[f"base_mae_{horizon}"] = row.get("base_test_mae", "")
        entry[f"hybrid_mse_{horizon}"] = row.get("hybrid_test_mse", "")
        entry[f"hybrid_mae_{horizon}"] = row.get("hybrid_test_mae", "")
        entry[f"hybrid_gain_mse_{horizon}"] = row.get("hybrid_gain_mse_vs_base", "")
        entry[f"hybrid_gain_mae_{horizon}"] = row.get("hybrid_gain_mae_vs_base", "")
        entry[f"selected_variant_{horizon}"] = row.get("selected_variant", "")
        entry[f"selected_mse_{horizon}"] = row.get("selected_test_mse", "")
        entry[f"selected_mae_{horizon}"] = row.get("selected_test_mae", "")
        entry[f"status_{horizon}"] = row.get("status", "")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for dataset in order:
            writer.writerow({field: grouped[dataset].get(field, "") for field in fields})


def summary_reusable(summary_path: Path, config_path: Path) -> bool:
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return str(summary.get("config_path", "")).replace("/", "\\") == str(config_path).replace("/", "\\")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run ETTh1/ETTh2/ETTm1/ETTm2 over 96/192/336/720 and write base+hybrid metrics to CSV."
    )
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, choices=DEFAULT_DATASETS)
    ap.add_argument("--horizons", nargs="+", type=int, default=DEFAULT_HORIZONS)
    ap.add_argument("--out-root", default="outputs/ett_horizon_sweep")
    ap.add_argument("--results-csv", default=None)
    ap.add_argument("--wide-csv", default=None)
    ap.add_argument("--input-len", type=int, default=None, help="Override input_len for every generated run.")
    ap.add_argument("--epochs", type=int, default=None, help="Override training epochs for every generated run.")
    ap.add_argument("--device", default=None, help="Override exp.device, e.g. cuda:0 or cpu.")
    ap.add_argument("--reuse-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--keep-artifacts", action="store_true")
    ap.add_argument("--save-checkpoint", action="store_true", help="Force memory.save_checkpoint=true in generated configs.")
    ap.add_argument("--no-hybrid", action="store_true", help="Do not force knn_hybrid.enable=true.")
    ap.add_argument(
        "--hybrid-selection-policy",
        default="val_mae_guarded",
        choices=["hybrid", "val_mse_margin", "val_mae_guarded", "val_mse", "base"],
    )
    ap.add_argument(
        "--hybrid-adaptive-alpha",
        default=None,
        choices=["none", "agreement", "distance", "confidence", "distance_agreement"],
    )
    ap.add_argument("--show-child-stdout", action="store_true")
    ap.add_argument("--no-preserve-results", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_root = resolve_path(args.out_root)
    results_csv = resolve_path(args.results_csv) if args.results_csv else out_root / "results.csv"
    wide_csv = resolve_path(args.wide_csv) if args.wide_csv else out_root / "results_wide.csv"

    rows = [] if args.no_preserve_results else read_rows(results_csv)
    total = len(args.datasets) * len(args.horizons)
    run_idx = 0
    for dataset in args.datasets:
        base_config = resolve_path(DEFAULT_CONFIGS[dataset])
        base_cfg = load_yaml(base_config)
        data_csv = str(base_cfg.get("data", {}).get("csv_path", ""))
        inherited_input_len = int(base_cfg.get("window", {}).get("input_len", 0))
        input_len = int(args.input_len) if args.input_len is not None else inherited_input_len

        for pred_len in args.horizons:
            run_idx += 1
            cfg, config_path, out_dir = make_config(
                base_cfg,
                dataset=dataset,
                base_config=base_config,
                pred_len=int(pred_len),
                input_len=input_len,
                out_root=out_root,
                epochs=args.epochs,
                device=args.device,
                keep_artifacts=bool(args.keep_artifacts),
                save_checkpoint=bool(args.save_checkpoint),
                enable_hybrid=not bool(args.no_hybrid),
                hybrid_selection_policy=args.hybrid_selection_policy,
                hybrid_adaptive_alpha=args.hybrid_adaptive_alpha,
            )
            write_yaml(config_path, cfg)
            summary_path = out_dir / "run_summary.json"
            print(f"[{run_idx}/{total}] {dataset} pred_len={pred_len} config={config_path}")

            t0 = time.perf_counter()
            if args.dry_run:
                row = empty_row(
                    status="prepared",
                    dataset=dataset,
                    pred_len=int(pred_len),
                    input_len=input_len,
                    data_csv=data_csv,
                    base_config=base_config,
                    run_config=config_path,
                    out_dir=out_dir,
                    wrapper_sec=0.0,
                    returncode=0,
                )
            elif args.reuse_existing and summary_reusable(summary_path, config_path):
                row = summary_to_row(
                    summary_path,
                    status="reused",
                    dataset=dataset,
                    pred_len=int(pred_len),
                    input_len=input_len,
                    data_csv=data_csv,
                    base_config=base_config,
                    run_config=config_path,
                    out_dir=out_dir,
                    wrapper_sec=0.0,
                    returncode=0,
                )
            else:
                returncode = run_train(config_path, out_dir, show_stdout=bool(args.show_child_stdout))
                wrapper_sec = time.perf_counter() - t0
                status = "ok"
                error = ""
                if returncode != 0:
                    status = "failed"
                    error = read_tail(out_dir / "stderr.log") or read_tail(out_dir / "stdout.log")
                elif not summary_path.exists():
                    status = "failed"
                    error = "run_summary.json not found"
                row = summary_to_row(
                    summary_path,
                    status=status,
                    dataset=dataset,
                    pred_len=int(pred_len),
                    input_len=input_len,
                    data_csv=data_csv,
                    base_config=base_config,
                    run_config=config_path,
                    out_dir=out_dir,
                    wrapper_sec=wrapper_sec,
                    returncode=returncode,
                    error=error,
                )

            upsert_row(rows, row)
            write_rows(results_csv, rows)
            write_wide(wide_csv, rows, [int(h) for h in args.horizons])
            print(
                "  -> "
                f"{row['status']} base_mse={row.get('base_test_mse', '')} "
                f"hybrid_mse={row.get('hybrid_test_mse', '')} "
                f"selected={row.get('selected_variant', '')} "
                f"out={out_dir}"
            )
            if row["status"] == "failed" and args.stop_on_error:
                raise SystemExit(f"Stopped after failed run: {dataset} pred_len={pred_len}")

    write_rows(results_csv, rows)
    write_wide(wide_csv, rows, [int(h) for h in args.horizons])
    print(f"Saved long CSV: {results_csv}")
    print(f"Saved wide CSV: {wide_csv}")


if __name__ == "__main__":
    main()
