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


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)


def read_summary(run_dir: Path) -> dict[str, Any]:
    with (run_dir / "run_summary.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def best_config_path_for(dataset: str, horizon: int, results_csv: Path) -> Path:
    csv_path = results_csv if results_csv.is_absolute() else ROOT / results_csv
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = [
            row
            for row in csv.DictReader(f)
            if row.get("dataset") == dataset
            and int(float(row.get("horizon", "nan"))) == int(horizon)
            and row.get("status", "ok") == "ok"
        ]
    if not rows:
        raise ValueError(f"No best-result row for dataset={dataset}, horizon={horizon} in {csv_path}")
    best_rows = [row for row in rows if str(row.get("is_best_for_cell", "")).lower() == "true"]
    row = best_rows[0] if best_rows else min(rows, key=lambda r: float(r.get("test_mse", "inf")))
    path = Path(row["config_path"])
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def normalize_cfg(
    cfg: dict[str, Any],
    *,
    dataset: str,
    variant: str,
    run_dir: Path,
    device: str | None,
    input_len: int,
    pred_len: int,
    epochs: int | None,
    batch_size: int | None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("exp", {})["out_dir"] = str(run_dir)
    cfg["exp"]["name"] = f"cluster_ablation_{dataset}_{variant}"
    if device:
        cfg["exp"]["device"] = str(device)

    cfg.setdefault("window", {})["input_len"] = int(input_len)
    cfg["window"]["pred_len"] = int(pred_len)
    cfg.setdefault("normalize", {})["train_only"] = True
    cfg.setdefault("cluster", {})["train_only"] = True
    cfg.setdefault("corr", {})["save_path"] = str(run_dir / "corr.npy")

    cfg.setdefault("plot", {})["enable"] = False
    cfg.setdefault("portrait", {})["enable"] = False
    cfg["portrait"]["out_dir"] = str(run_dir / "cluster_portraits")
    cfg.setdefault("eval", {})["skip_test"] = False
    cfg["eval"]["save_predictions"] = False
    cfg["memory"] = {
        "enable": False,
        "save_checkpoint": False,
        "path": str(run_dir / "cluster_memory.pt"),
        "checkpoint_path": str(run_dir / "best_checkpoint.pt"),
    }

    if epochs is not None:
        cfg.setdefault("train", {})["epochs"] = int(epochs)
        if "epochs" in gate_cal:
            gate_cal["epochs"] = min(int(gate_cal["epochs"]), int(epochs))
    if batch_size is not None:
        cfg.setdefault("train", {})["batch_size"] = int(batch_size)
    return cfg


def patch_cluster(cfg: dict[str, Any], variant: str, *, k: int, threshold: float) -> None:
    cl = cfg.setdefault("cluster", {})
    variant = variant.lower()

    if variant == "main_best":
        return

    # Keep train-only clustering and all non-cluster fields fixed.  Merge is
    # disabled for method ablations so the requested algorithm is actually
    # responsible for the partition rather than post-hoc singleton pooling.
    cl["train_only"] = True
    cl["merge_small_clusters"] = False
    cl["min_cluster_size"] = 1
    cl["no_merge_if_channels_lt"] = 999
    cl["singleton_merge_strategy"] = "keep"

    if variant == "single_head":
        cl["method"] = "agglomerative"
        cl["n_clusters"] = 1
        cl["distance_threshold"] = None
        return
    if variant == "leader":
        cl["method"] = "leader"
        cl["n_clusters"] = int(k)
        cl["distance_threshold"] = float(threshold)
        return
    if variant in {"agglomerative", "agglo"}:
        cl["method"] = "agglomerative"
        cl["n_clusters"] = int(k)
        cl["distance_threshold"] = None
        cl["linkage"] = cl.get("linkage", "average")
        return
    if variant == "kmeans":
        cl["method"] = "kmeans"
        cl["n_clusters"] = int(k)
        cl["distance_threshold"] = None
        cl["kmeans_n_init"] = int(cl.get("kmeans_n_init", 20))
        cl["kmeans_max_iter"] = int(cl.get("kmeans_max_iter", 300))
        return
    if variant == "spectral":
        cl["method"] = "spectral"
        cl["n_clusters"] = int(k)
        cl["distance_threshold"] = None
        cl["spectral_affinity"] = cl.get("spectral_affinity", "corr")
        return
    if variant == "random":
        cl["method"] = "random"
        cl["n_clusters"] = int(k)
        cl["distance_threshold"] = None
        return
    raise ValueError(f"Unknown cluster ablation variant: {variant}")


def extract_cluster_sizes(summary: dict[str, Any]) -> tuple[int | str, str]:
    """Read cluster ids from explainability per-channel blocks when available."""
    found: list[int] = []

    def visit(value: Any) -> None:
        nonlocal found
        if found:
            return
        if isinstance(value, dict):
            per_channel = value.get("per_channel")
            if isinstance(per_channel, list) and per_channel:
                ids = [item.get("cluster_id") for item in per_channel if isinstance(item, dict)]
                if ids and all(isinstance(x, int) for x in ids):
                    found = [int(x) for x in ids]
                    return
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(summary)
    if not found:
        return "", ""
    max_id = max(found)
    sizes = [0 for _ in range(max_id + 1)]
    for cluster_id in found:
        sizes[cluster_id] += 1
    return len(sizes), ",".join(f"{idx}:{size}" for idx, size in enumerate(sizes))


def run_train(config_path: Path, *, python: str, reuse_existing: bool) -> int:
    cfg = read_yaml(config_path)
    run_dir = Path(cfg["exp"]["out_dir"])
    if reuse_existing and (run_dir / "run_summary.json").exists():
        print(f"[reuse] {run_dir}", flush=True)
        return 0
    cmd = [python, "-u", "-m", "src.train", "--config", str(config_path)]
    print(f"[run] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT))
    return int(proc.returncode)


def make_row(
    *,
    dataset: str,
    variant: str,
    cfg: dict[str, Any],
    config_path: Path,
    run_dir: Path,
    returncode: int,
) -> dict[str, Any]:
    cluster_cfg = cfg.get("cluster", {}) or {}
    row: dict[str, Any] = {
        "dataset": dataset,
        "variant": variant,
        "status": "ok" if returncode == 0 and (run_dir / "run_summary.json").exists() else "failed",
        "cluster_method": cluster_cfg.get("method", ""),
        "requested_n_clusters": cluster_cfg.get("n_clusters", ""),
        "distance_threshold": cluster_cfg.get("distance_threshold", ""),
        "merge_small_clusters": cluster_cfg.get("merge_small_clusters", ""),
        "min_cluster_size": cluster_cfg.get("min_cluster_size", ""),
        "single_head": variant == "single_head",
        "config_path": str(config_path),
        "out_dir": str(run_dir),
        "returncode": returncode,
    }
    if (run_dir / "run_summary.json").exists():
        summary = read_summary(run_dir)
        cluster_count, cluster_sizes = extract_cluster_sizes(summary)
        row.update(
            {
                "input_len": (summary.get("windowing", {}) or {}).get("input_len", cfg.get("window", {}).get("input_len", "")),
                "pred_len": (summary.get("windowing", {}) or {}).get("pred_len", cfg.get("window", {}).get("pred_len", "")),
                "test_mse": (summary.get("test", {}) or {}).get("avg_mse", ""),
                "test_mae": (summary.get("test", {}) or {}).get("avg_mae", ""),
                "val_mse": (summary.get("val", {}) or {}).get("avg_mse", ""),
                "val_mae": (summary.get("val", {}) or {}).get("avg_mae", ""),
                "best_epoch": ",".join(str(v) for v in summary.get("best_epoch", [])),
                "penalties": ",".join(str(v) for v in summary.get("penalty_names", [])),
                "actual_n_clusters": cluster_count,
                "cluster_sizes": cluster_sizes,
            }
        )
    return row


def add_gains(rows: list[dict[str, Any]]) -> None:
    by_dataset: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("status") == "ok" and row.get("variant") == "main_best":
            by_dataset[str(row["dataset"])] = row
    for row in rows:
        ref = by_dataset.get(str(row.get("dataset")))
        if not ref or row.get("status") != "ok":
            continue
        ref_mse = float(ref["test_mse"])
        cur_mse = float(row["test_mse"])
        row["delta_mse_vs_main"] = cur_mse - ref_mse
        row["gain_pct_vs_main"] = (ref_mse - cur_mse) / ref_mse * 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["ETTh1", "ETTh2", "ETTm1", "ETTm2"])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["main_best", "single_head", "agglomerative", "kmeans", "spectral", "random"],
    )
    parser.add_argument("--base-results-csv", type=Path, default=ROOT / "outputs" / "ett_horizon_specific_moe_tune" / "best_results.csv")
    parser.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "cluster_ablation_h96")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--input-len", type=int, default=336)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--default-k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for dataset in args.datasets:
        source_path = best_config_path_for(dataset, args.pred_len, args.base_results_csv)
        source_cfg = read_yaml(source_path)
        source_k = int((source_cfg.get("cluster", {}) or {}).get("n_clusters") or args.default_k)
        print(f"[dataset] {dataset} from {source_path}", flush=True)
        for variant in args.variants:
            run_dir = args.out_root / "runs" / dataset / variant
            cfg = normalize_cfg(
                source_cfg,
                dataset=dataset,
                variant=variant,
                run_dir=run_dir,
                device=args.device,
                input_len=args.input_len,
                pred_len=args.pred_len,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
            patch_cluster(cfg, variant, k=source_k, threshold=args.threshold)
            config_path = args.out_root / "configs" / dataset / f"{variant}.yaml"
            write_yaml(config_path, cfg)
            returncode = run_train(config_path, python=str(args.python), reuse_existing=args.reuse_existing)
            rows.append(
                make_row(
                    dataset=dataset,
                    variant=variant,
                    cfg=cfg,
                    config_path=config_path,
                    run_dir=run_dir,
                    returncode=returncode,
                )
            )
            add_gains(rows)
            write_csv(args.out_root / "cluster_ablation_results.csv", rows, output_fields())

    add_gains(rows)
    write_csv(args.out_root / "cluster_ablation_results.csv", rows, output_fields())
    print(f"Saved cluster ablation results to {args.out_root / 'cluster_ablation_results.csv'}", flush=True)


def output_fields() -> list[str]:
    return [
        "dataset",
        "variant",
        "status",
        "input_len",
        "pred_len",
        "cluster_method",
        "requested_n_clusters",
        "actual_n_clusters",
        "cluster_sizes",
        "distance_threshold",
        "merge_small_clusters",
        "min_cluster_size",
        "single_head",
        "test_mse",
        "test_mae",
        "val_mse",
        "val_mae",
        "best_epoch",
        "penalties",
        "delta_mse_vs_main",
        "gain_pct_vs_main",
        "config_path",
        "out_dir",
        "returncode",
    ]


if __name__ == "__main__":
    main()
