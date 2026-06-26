from __future__ import annotations

import argparse
import csv
import copy
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


REFERENCE = {
    "tag": "reference_threshold_0p7",
    "threshold": 0.7,
    "source_summary": ROOT
    / "outputs"
    / "ettm1_val_refinement_base"
    / "runs"
    / "ETTm1"
    / "pred_96"
    / "run_summary.json",
    "source_memory": ROOT / "outputs" / "ettm1_h96_transfer_no_leak" / "source" / "cluster_memory.pt",
    "transfer_summary": ROOT / "outputs" / "ETTm1ToETTm2" / "transfer_summary.json",
}


CSV_FIELDS = [
    "status",
    "tag",
    "distance_threshold",
    "corr_threshold",
    "source_clusters",
    "source_cluster_counts",
    "target_route_clusters",
    "target_route_counts",
    "target_corr_mean",
    "target_corr_min",
    "source_val_mse",
    "source_val_mae",
    "source_test_mse",
    "source_test_mae",
    "transfer_mse",
    "transfer_mae",
    "transfer_route_fit_scope",
    "normalize_train_only",
    "route_uses_train_only",
    "eval_uses_test_only",
    "source_out_dir",
    "transfer_out_dir",
    "source_config",
    "transfer_config",
    "error",
]


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def command_prefix(args: argparse.Namespace) -> list[str]:
    if args.conda_env:
        return [str(args.conda_exe), "run", "-n", str(args.conda_env), "python"]
    return [sys.executable]


def run_cmd(cmd: list[str], cwd: Path) -> float:
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=str(cwd), check=True)
    return float(time.perf_counter() - t0)


def scalar(summary: dict[str, Any], split: str, key: str) -> float | None:
    obj = summary.get(split, {}) or {}
    if key not in obj:
        return None
    return float(obj[key])


def threshold_tag(threshold: float) -> str:
    text = f"{threshold:.4g}".replace(".", "p").replace("-", "m")
    return f"thr_{text}"


def cluster_counts_from_memory(path: Path) -> tuple[int | None, str]:
    if not path.exists():
        return None, ""
    import torch

    payload = torch.load(path, map_location="cpu")
    cluster_id = payload.get("cluster_id_c")
    if cluster_id is None:
        return None, ""
    vals = [int(v) for v in cluster_id.detach().cpu().tolist()]
    counts: dict[int, int] = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    ordered = dict(sorted(counts.items(), key=lambda kv: kv[0]))
    return len(ordered), json.dumps(ordered, ensure_ascii=False)


def route_stats(path: Path) -> tuple[int | None, str, float | None, float | None]:
    if not path.exists():
        return None, "", None, None
    counts: dict[int, int] = {}
    corr_values: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = int(float(row["cluster_id"]))
            counts[cid] = counts.get(cid, 0) + 1
            if "corr_max" in row and row["corr_max"] != "":
                corr_values.append(float(row["corr_max"]))
    mean_corr = sum(corr_values) / len(corr_values) if corr_values else None
    min_corr = min(corr_values) if corr_values else None
    ordered = dict(sorted(counts.items(), key=lambda kv: kv[0]))
    return len(ordered), json.dumps(ordered, ensure_ascii=False), mean_corr, min_corr


def make_source_cfg(
    base_cfg: dict[str, Any],
    *,
    tag: str,
    threshold: float,
    out_dir: Path,
    epochs: int,
    device: str,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("exp", {})
    cfg["exp"]["name"] = f"ETTm1_H96_threshold_{tag}"
    cfg["exp"]["out_dir"] = str(out_dir)
    cfg["exp"]["device"] = device
    cfg["exp"]["seed"] = int(cfg["exp"].get("seed", 2026))
    cfg["exp"]["deterministic"] = bool(cfg["exp"].get("deterministic", True))

    cfg.setdefault("data", {})
    cfg["data"]["csv_path"] = "data/ETTm1.csv"
    cfg["data"]["date_col"] = 0
    cfg["data"]["max_rows"] = 57600
    cfg["data"]["train_ratio"] = 0.6
    cfg["data"]["val_ratio"] = 0.2
    cfg["data"]["test_ratio"] = 0.2

    cfg.setdefault("window", {})
    cfg["window"]["input_len"] = 336
    cfg["window"]["pred_len"] = 96

    cfg.setdefault("normalize", {})
    cfg["normalize"]["global_zscore"] = True
    cfg["normalize"]["train_only"] = True

    cfg.setdefault("corr", {})
    cfg["corr"]["compute"] = True
    cfg["corr"]["save_path"] = str(out_dir / "corr.npy")

    cfg.setdefault("cluster", {})
    cfg["cluster"]["method"] = "leader"
    cfg["cluster"]["distance_threshold"] = float(threshold)
    cfg["cluster"]["n_clusters"] = 3
    cfg["cluster"]["train_only"] = True
    cfg["cluster"]["merge_small_clusters"] = True
    cfg["cluster"]["min_cluster_size"] = 2
    cfg["cluster"]["no_merge_if_channels_lt"] = 7

    cfg.setdefault("model", {})
    cfg["model"]["predictor"] = "mlp"
    cfg["model"]["hidden_dim"] = 256
    cfg["model"]["dropout"] = 0.2

    cfg.setdefault("train", {})
    cfg["train"]["epochs"] = int(epochs)
    cfg["train"]["batch_size"] = int(cfg["train"].get("batch_size", 64))


    cfg.setdefault("eval", {})
    cfg["eval"]["skip_test"] = False

    cfg["memory"] = {
        "enable": True,
        "path": str(out_dir / "cluster_memory.pt"),
        "save_checkpoint": True,
        "checkpoint_path": str(out_dir / "best_checkpoint.pt"),
    }
    return cfg


def make_transfer_cfg(
    base_cfg: dict[str, Any],
    *,
    tag: str,
    source_run: Path,
    out_dir: Path,
    device: str,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("exp", {})
    cfg["exp"]["name"] = f"ETTm1_to_ETTm2_{tag}"
    cfg["exp"]["out_dir"] = str(out_dir)
    cfg["exp"]["device"] = device
    cfg["exp"]["seed"] = int(cfg["exp"].get("seed", 2026))

    cfg["source"] = {
        "memory_path": str(source_run / "cluster_memory.pt"),
        "checkpoint_path": str(source_run / "best_checkpoint.pt"),
        "summary_path": str(source_run / "run_summary.json"),
        "csv_path": "data/ETTm1.csv",
        "date_col": 0,
        "step_minutes": 15,
    }

    cfg.setdefault("data", {})
    cfg["data"]["csv_path"] = "data/ETTm2.csv"
    cfg["data"]["date_col"] = 0
    cfg["data"]["train_ratio"] = 0.7
    cfg["data"]["val_ratio"] = 0.1
    cfg["data"]["test_ratio"] = 0.2

    cfg.setdefault("window", {})
    cfg["window"]["input_len"] = 336
    cfg["window"]["pred_len"] = 96

    cfg.setdefault("normalize", {})
    cfg["normalize"]["global_zscore"] = True
    cfg["normalize"]["train_only"] = True

    cfg.setdefault("transfer", {})
    cfg["transfer"]["corr_mode"] = "cycle_template"
    cfg["transfer"]["route_fit_scope"] = "train"
    cfg["transfer"]["use_pred_residual"] = True
    cfg["transfer"]["phase_bins"] = int(cfg["transfer"].get("phase_bins", 64))
    cfg["transfer"]["period_min_hours"] = cfg["transfer"].get("period_min_hours", 12)
    cfg["transfer"]["period_max_hours"] = cfg["transfer"].get("period_max_hours", 168)
    cfg["transfer"]["corr_align"] = "head"
    cfg["transfer"]["corr_max_lag"] = 0
    cfg["transfer"]["save_corr"] = True
    cfg["transfer"].setdefault("resample", {})
    cfg["transfer"]["resample"]["enable"] = False

    cfg.setdefault("eval", {})
    cfg["eval"]["batch_size"] = int(cfg["eval"].get("batch_size", 64))
    return cfg


def reference_row() -> dict[str, Any] | None:
    if not REFERENCE["transfer_summary"].exists():
        return None
    source_summary = read_json(REFERENCE["source_summary"])
    transfer_summary = read_json(REFERENCE["transfer_summary"])
    source_k, source_counts = cluster_counts_from_memory(REFERENCE["source_memory"])
    assign_path = REFERENCE["transfer_summary"].parent / "cluster_assignment.csv"
    route_k, route_counts, corr_mean, corr_min = route_stats(assign_path)
    return {
        "status": "reference",
        "tag": REFERENCE["tag"],
        "distance_threshold": REFERENCE["threshold"],
        "corr_threshold": 1.0 - float(REFERENCE["threshold"]),
        "source_clusters": source_k,
        "source_cluster_counts": source_counts,
        "target_route_clusters": route_k,
        "target_route_counts": route_counts,
        "target_corr_mean": corr_mean,
        "target_corr_min": corr_min,
        "source_val_mse": scalar(source_summary, "val", "avg_mse"),
        "source_val_mae": scalar(source_summary, "val", "avg_mae"),
        "source_test_mse": scalar(source_summary, "test", "avg_mse"),
        "source_test_mae": scalar(source_summary, "test", "avg_mae"),
        "transfer_mse": transfer_summary.get("avg_mse"),
        "transfer_mae": transfer_summary.get("avg_mae"),
        "transfer_route_fit_scope": transfer_summary.get("route_fit_scope"),
        "normalize_train_only": transfer_summary.get("normalize_train_only"),
        "route_uses_train_only": transfer_summary.get("route_uses_train_only"),
        "eval_uses_test_only": transfer_summary.get("eval_uses_test_only"),
        "source_out_dir": str(REFERENCE["source_summary"].parent),
        "transfer_out_dir": str(REFERENCE["transfer_summary"].parent),
        "source_config": "",
        "transfer_config": "configs/ETTm1ToETTm2.yaml",
        "error": "",
    }


def write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", type=Path, default=ROOT / "configs" / "ETTm1.yaml")
    ap.add_argument("--transfer-template", type=Path, default=ROOT / "configs" / "ETTm1ToETTm2.yaml")
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "ettm1_threshold_transfer_to_ettm2")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.3, 0.05])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--conda-exe", type=Path, default=Path(r"F:\Anaconda3\Scripts\conda.exe"))
    ap.add_argument("--conda-env", type=str, default="my_fram")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-reference", action="store_true")
    args = ap.parse_args()

    out_root = args.out_root
    cfg_root = out_root / "configs"
    run_root = out_root / "runs"
    rows: list[dict[str, Any]] = []
    if not args.skip_reference:
        ref = reference_row()
        if ref is not None:
            rows.append(ref)

    base_cfg = read_yaml(args.base_config)
    transfer_template = read_yaml(args.transfer_template)
    prefix = command_prefix(args)

    for threshold in args.thresholds:
        tag = threshold_tag(float(threshold))
        source_run = run_root / tag / "source"
        transfer_run = run_root / tag / "transfer_ETTm2"
        source_cfg_path = cfg_root / f"{tag}_source_ETTm1.yaml"
        transfer_cfg_path = cfg_root / f"{tag}_transfer_ETTm2.yaml"
        row: dict[str, Any] = {
            "status": "pending",
            "tag": tag,
            "distance_threshold": float(threshold),
            "corr_threshold": 1.0 - float(threshold),
            "source_out_dir": str(source_run),
            "transfer_out_dir": str(transfer_run),
            "source_config": str(source_cfg_path),
            "transfer_config": str(transfer_cfg_path),
            "error": "",
        }
        try:
            source_cfg = make_source_cfg(
                base_cfg,
                tag=tag,
                threshold=float(threshold),
                out_dir=source_run,
                epochs=args.epochs,
                device=args.device,
            )
            transfer_cfg = make_transfer_cfg(
                transfer_template,
                tag=tag,
                source_run=source_run,
                out_dir=transfer_run,
                device=args.device,
            )
            write_yaml(source_cfg_path, source_cfg)
            write_yaml(transfer_cfg_path, transfer_cfg)

            source_summary_path = source_run / "run_summary.json"
            transfer_summary_path = transfer_run / "transfer_summary.json"
            if args.force or not source_summary_path.exists():
                print(f"[{tag}] training ETTm1 source with distance_threshold={threshold}")
                run_cmd(prefix + ["-m", "src.train", "--config", str(source_cfg_path)], ROOT)
            else:
                print(f"[{tag}] reuse source summary: {source_summary_path}")
            if args.force or not transfer_summary_path.exists():
                print(f"[{tag}] transfer ETTm1 -> ETTm2")
                run_cmd(prefix + ["-m", "src.transfer", "--config", str(transfer_cfg_path)], ROOT)
            else:
                print(f"[{tag}] reuse transfer summary: {transfer_summary_path}")

            source_summary = read_json(source_summary_path)
            transfer_summary = read_json(transfer_summary_path)
            source_k, source_counts = cluster_counts_from_memory(source_run / "cluster_memory.pt")
            route_k, route_counts, corr_mean, corr_min = route_stats(transfer_run / "cluster_assignment.csv")
            row.update(
                {
                    "status": "ok",
                    "source_clusters": source_k,
                    "source_cluster_counts": source_counts,
                    "target_route_clusters": route_k,
                    "target_route_counts": route_counts,
                    "target_corr_mean": corr_mean,
                    "target_corr_min": corr_min,
                    "source_val_mse": scalar(source_summary, "val", "avg_mse"),
                    "source_val_mae": scalar(source_summary, "val", "avg_mae"),
                    "source_test_mse": scalar(source_summary, "test", "avg_mse"),
                    "source_test_mae": scalar(source_summary, "test", "avg_mae"),
                    "transfer_mse": transfer_summary.get("avg_mse"),
                    "transfer_mae": transfer_summary.get("avg_mae"),
                    "transfer_route_fit_scope": transfer_summary.get("route_fit_scope"),
                    "normalize_train_only": transfer_summary.get("normalize_train_only"),
                    "route_uses_train_only": transfer_summary.get("route_uses_train_only"),
                    "eval_uses_test_only": transfer_summary.get("eval_uses_test_only"),
                }
            )
        except Exception as exc:
            row["status"] = "error"
            row["error"] = repr(exc)
            rows.append(row)
            write_results(out_root / "threshold_transfer_results.csv", rows)
            raise
        rows.append(row)
        write_results(out_root / "threshold_transfer_results.csv", rows)

    write_results(out_root / "threshold_transfer_results.csv", rows)
    ok_rows = [r for r in rows if r.get("status") in {"ok", "reference"} and r.get("transfer_mse") not in {None, ""}]
    if ok_rows:
        best = min(ok_rows, key=lambda r: float(r["transfer_mse"]))
        with (out_root / "best_threshold.json").open("w", encoding="utf-8") as f:
            json.dump(best, f, ensure_ascii=False, indent=2)
        print(
            f"Best transfer MSE: {float(best['transfer_mse']):.6f} "
            f"({best['tag']}, threshold={best['distance_threshold']})"
        )
    print(f"Saved results to: {out_root / 'threshold_transfer_results.csv'}")


if __name__ == "__main__":
    main()
