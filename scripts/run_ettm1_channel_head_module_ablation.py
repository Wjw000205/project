from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rerun_cluster_ablation import extract_cluster_sizes, output_fields
from scripts.rerun_current_module_ablation import (
    ROOT,
    add_gain,
    annotate_moe_fields,
    deep_update,
    disable_moe_patch,
    fixed_lambda_patch,
    make_run,
    metric_row,
    penalty_loss_only_patch,
    read_yaml,
    write_csv,
    zero_lambda_patch,
)


MODULE_FIELDS = [
    "dataset",
    "group",
    "label",
    "status",
    "input_len",
    "pred_len",
    "cluster_method",
    "requested_n_clusters",
    "actual_n_clusters",
    "cluster_sizes",
    "test_mse",
    "test_mae",
    "val_mse",
    "val_mae",
    "best_epoch",
    "penalties",
    "config_path",
    "out_dir",
    "returncode",
    "gain_vs_ref_pct",
    "moe_enable",
    "dynamic_lambda_enable",
    "pred_side_residual_enable",
    "cluster_penalty_prior_enable",
    "cluster_penalty_prior_topk",
    "penalty_selector_enable",
    "fusion_gate_enable",
    "allow_skip",
]


def singleton_cluster_patch(num_channels: int, seed: int) -> dict[str, Any]:
    # A fixed one-to-one channel assignment is the clearest "no clustering" path:
    # the model still builds cluster-specific heads/MoEs, but each cluster has one channel.
    return {
        "cluster": {
            "method": "agglomerative",
            "n_clusters": int(num_channels),
            "distance_threshold": None,
            "merge_small_clusters": False,
            "min_cluster_size": 1,
            "no_merge_if_channels_lt": 999,
            "singleton_merge_strategy": "keep",
            "train_only": True,
            "random_state": int(seed),
            "fixed_cluster_id": list(range(int(num_channels))),
        }
    }


def module_specs(base_cfg: dict[str, Any], labels: list[str]) -> list[tuple[str, dict[str, Any]]]:
    all_specs = [
        ("moe_off", disable_moe_patch()),
        ("zero_lambda_residual", zero_lambda_patch(base_cfg)),
        ("fixed_lambda_residual", fixed_lambda_patch()),
        ("penalty_loss_only", penalty_loss_only_patch()),
        ("full_current", {}),
    ]
    known = {label for label, _ in all_specs}
    unknown = [label for label in labels if label not in known]
    if unknown:
        raise ValueError(f"Unknown module labels: {unknown}; choose from {sorted(known)}")
    wanted = set(labels)
    return [(label, patch) for label, patch in all_specs if label in wanted]


def add_cluster_info(row: dict[str, Any]) -> None:
    cfg = read_yaml(Path(row["config_path"]))
    cluster_cfg = cfg.get("cluster", {}) or {}
    row["cluster_method"] = "fixed_singleton"
    row["requested_n_clusters"] = cluster_cfg.get("n_clusters", "")
    row["actual_n_clusters"] = ""
    row["cluster_sizes"] = ""
    run_dir = Path(row["out_dir"])
    summary_path = run_dir / "run_summary.json"
    if summary_path.exists():
        import json

        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        count, sizes = extract_cluster_sizes(summary)
        row["actual_n_clusters"] = count
        row["cluster_sizes"] = sizes


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def clustered_main_row(dataset: str, base_config_path: Path) -> dict[str, Any]:
    cfg = read_yaml(base_config_path)
    run_dir = Path(cfg["exp"]["out_dir"])
    row = metric_row(
        dataset=dataset,
        group="module",
        label="full_current",
        config_path=base_config_path,
        run_dir=run_dir,
        returncode=0,
    )
    cluster_cfg = cfg.get("cluster", {}) or {}
    row["variant"] = "main_best"
    row["cluster_method"] = cluster_cfg.get("method", "")
    row["requested_n_clusters"] = cluster_cfg.get("n_clusters", "")
    row["distance_threshold"] = cluster_cfg.get("distance_threshold", "")
    row["merge_small_clusters"] = cluster_cfg.get("merge_small_clusters", "")
    row["min_cluster_size"] = cluster_cfg.get("min_cluster_size", "")
    row["single_head"] = False
    row["delta_mse_vs_main"] = 0.0
    row["gain_pct_vs_main"] = 0.0
    add_cluster_info_for_cluster_row(row, cfg)
    return row


def add_cluster_info_for_cluster_row(row: dict[str, Any], cfg: dict[str, Any] | None = None) -> None:
    if cfg is None:
        cfg = read_yaml(Path(row["config_path"]))
    summary_path = Path(row["out_dir"]) / "run_summary.json"
    if not summary_path.exists():
        return
    import json

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    count, sizes = extract_cluster_sizes(summary)
    row["actual_n_clusters"] = count
    row["cluster_sizes"] = sizes


def to_cluster_row(module_row: dict[str, Any], ref_mse: float) -> dict[str, Any]:
    cfg = read_yaml(Path(module_row["config_path"]))
    cluster_cfg = cfg.get("cluster", {}) or {}
    row = {
        "dataset": module_row.get("dataset", "ETTm1"),
        "variant": f"channel_head_{module_row['label']}",
        "status": module_row.get("status", ""),
        "input_len": module_row.get("input_len", ""),
        "pred_len": module_row.get("pred_len", ""),
        "cluster_method": "fixed_singleton",
        "requested_n_clusters": cluster_cfg.get("n_clusters", ""),
        "actual_n_clusters": module_row.get("actual_n_clusters", ""),
        "cluster_sizes": module_row.get("cluster_sizes", ""),
        "distance_threshold": "",
        "merge_small_clusters": cluster_cfg.get("merge_small_clusters", ""),
        "min_cluster_size": cluster_cfg.get("min_cluster_size", ""),
        "single_head": False,
        "test_mse": module_row.get("test_mse", ""),
        "test_mae": module_row.get("test_mae", ""),
        "val_mse": module_row.get("val_mse", ""),
        "val_mae": module_row.get("val_mae", ""),
        "best_epoch": module_row.get("best_epoch", ""),
        "penalties": module_row.get("penalties", ""),
        "config_path": module_row.get("config_path", ""),
        "out_dir": module_row.get("out_dir", ""),
        "returncode": module_row.get("returncode", ""),
    }
    if row["status"] == "ok" and row["test_mse"] != "":
        cur_mse = float(row["test_mse"])
        row["delta_mse_vs_main"] = cur_mse - ref_mse
        row["gain_pct_vs_main"] = (ref_mse - cur_mse) / ref_mse * 100.0
    return row


def run_module_ablation(args: argparse.Namespace) -> list[dict[str, Any]]:
    base_cfg = read_yaml(args.base_config)
    base_cfg = copy.deepcopy(base_cfg)
    deep_update(base_cfg, singleton_cluster_patch(args.num_channels, args.seed))

    rows: list[dict[str, Any]] = []
    for label, patch in module_specs(base_cfg, args.labels):
        row = make_run(
            base_cfg,
            dataset=args.dataset,
            out_root=args.out_root,
            group="channel_head_module",
            label=label,
            patch=patch,
            device=args.device,
            pred_len=args.pred_len,
            input_len=args.input_len,
            epochs=args.epochs,
            batch_size=args.batch_size,
            reuse_existing=args.reuse_existing,
            python=str(args.python),
        )
        annotate_moe_fields(row)
        add_cluster_info(row)
        rows.append(row)
        add_gain(rows, ref_label="moe_off")
        write_csv(args.out_root / "module_results.csv", rows, MODULE_FIELDS)
    add_gain(rows, ref_label="moe_off")
    write_csv(args.out_root / "module_results.csv", rows, MODULE_FIELDS)
    return rows


def write_merged_cluster_table(args: argparse.Namespace, module_rows: list[dict[str, Any]]) -> Path:
    cluster_rows = read_csv_rows(args.cluster_table)
    new_variants = {"main_best"} | {f"channel_head_{row['label']}" for row in module_rows}
    kept_rows = [
        row
        for row in cluster_rows
        if not (row.get("dataset") == args.dataset and row.get("variant") in new_variants)
    ]
    main = clustered_main_row(args.dataset, args.base_config)
    ref_mse = float(main["test_mse"])
    merged = kept_rows + [main] + [to_cluster_row(row, ref_mse) for row in module_rows]
    out_path = args.cluster_table.parent / "cluster_ablation_results_with_channel_head.csv"
    write_csv(out_path, merged, output_fields())
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        type=Path,
        default=ROOT / "outputs" / "current_module_ablation_rerun_bestbase" / "configs" / "ETTm1" / "module" / "full_current.yaml",
    )
    parser.add_argument("--dataset", type=str, default="ETTm1")
    parser.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "ettm1_channel_head_module_ablation")
    parser.add_argument("--cluster-table", type=Path, default=ROOT / "outputs" / "cluster_ablation_h96" / "cluster_ablation_results.csv")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--input-len", type=int, default=336)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-channels", type=int, default=7)
    parser.add_argument("--labels", nargs="+", default=["full_current"])
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    args.base_config = args.base_config if args.base_config.is_absolute() else ROOT / args.base_config
    args.out_root = args.out_root if args.out_root.is_absolute() else ROOT / args.out_root
    args.cluster_table = args.cluster_table if args.cluster_table.is_absolute() else ROOT / args.cluster_table

    rows = run_module_ablation(args)
    merged_path = write_merged_cluster_table(args, rows)
    print(f"Saved module results to {args.out_root / 'module_results.csv'}", flush=True)
    print(f"Saved merged cluster table to {merged_path}", flush=True)


if __name__ == "__main__":
    main()
