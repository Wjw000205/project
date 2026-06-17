from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from run_input96_mse_gate_cluster_moe_retrain import (
    infer_warm_start_checkpoint,
    load_yaml,
    resolve,
)


ROOT = Path(__file__).resolve().parents[1]


def checkpoint_for(row: dict[str, str]) -> Path | None:
    cfg_path = resolve(row["moe_config"])
    cfg = load_yaml(cfg_path)
    ft_cfg = cfg.get("finetune", {}) or {}
    if bool(ft_cfg.get("enable", False)) and str(ft_cfg.get("checkpoint_path", "")).strip():
        path = resolve(str(ft_cfg["checkpoint_path"]))
        return path if path.exists() else None
    path = infer_warm_start_checkpoint(cfg, row)
    return path if path is not None and path.exists() else None


def run_cmd(cmd: list[str]) -> int:
    print(" ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete PEMS cluster-aware residual-profile penalty pool val runs.")
    parser.add_argument("--summary-csv", default=str(ROOT / "outputs/codex_table_target_20260614/input96_global_paired_backbone_moe_summary.csv"))
    parser.add_argument("--out-root", default=str(ROOT / "outputs/clusteraware_penalty_pool_completion_20260617/pems_k4_val"))
    parser.add_argument("--profile-dir", default=str(ROOT / "outputs/clusteraware_penalty_pool_completion_20260617/pems_residual_profiles_k4"))
    parser.add_argument("--datasets", nargs="*", default=["PEMS03", "PEMS04", "PEMS07", "PEMS08"])
    parser.add_argument("--horizons", nargs="*", type=int, default=[12, 24, 48, 96])
    parser.add_argument("--variant", default="pems_residual_profile_pool")
    parser.add_argument("--n-clusters", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    summary_csv = resolve(args.summary_csv)
    out_root = resolve(args.out_root)
    profile_dir = resolve(args.profile_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)

    dataset_filter = {item.upper() for item in args.datasets}
    horizon_filter = {int(item) for item in args.horizons}
    rows: list[dict[str, str]] = []
    with summary_csv.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            dataset = str(row.get("dataset", "")).upper()
            horizon = int(row.get("horizon", "0") or 0)
            if dataset in dataset_filter and horizon in horizon_filter:
                rows.append(row)

    completed: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for row in rows:
        dataset = str(row["dataset"])
        horizon = int(row["horizon"])
        checkpoint = checkpoint_for(row)
        if checkpoint is None:
            print(f"[skip] {dataset} H{horizon}: no compatible checkpoint", flush=True)
            skipped.append({"dataset": dataset, "horizon": horizon, "reason": "no compatible checkpoint"})
            continue

        tag = f"{dataset}_H{horizon}_residk{int(args.n_clusters)}"
        profile_json = profile_dir / f"{tag}_allowed_by_cluster.json"
        if (not profile_json.exists()) or (not args.reuse_existing):
            profile_cmd = [
                sys.executable,
                "scripts/profile_base_residual_penalty_pool.py",
                "--config",
                row["moe_config"],
                "--checkpoint",
                str(checkpoint),
                "--out-dir",
                str(profile_dir),
                "--penalties",
                "amp_under,level,delta,diff_amp,direction,d2_match,corr,range,trend,jump,seasonal_align",
                "--n-clusters",
                str(int(args.n_clusters)),
                "--cluster-source",
                "residual_kmeans",
                "--topk",
                "3",
                "--min-score",
                "0.75",
                "--keep-ratio",
                "0.75",
                "--batch-size",
                "64",
                "--device",
                str(args.device),
                "--tag",
                tag,
            ]
            print(f"[profile] {dataset} H{horizon}", flush=True)
            rc = run_cmd(profile_cmd)
            if rc != 0:
                skipped.append({"dataset": dataset, "horizon": horizon, "reason": f"profile rc={rc}"})
                continue
        if not profile_json.exists():
            skipped.append({"dataset": dataset, "horizon": horizon, "reason": "profile json missing"})
            continue

        val_cmd = [
            sys.executable,
            "scripts/run_input96_mse_gate_cluster_moe_retrain.py",
            "--summary-csv",
            str(summary_csv),
            "--out-root",
            str(out_root),
            "--datasets",
            dataset,
            "--horizons",
            str(horizon),
            "--variants",
            str(args.variant),
            "--device",
            str(args.device),
            "--skip-test",
            "--reuse-existing",
            "--residual-profile-json",
            str(profile_json),
        ]
        print(f"[val] {dataset} H{horizon}", flush=True)
        rc = run_cmd(val_cmd)
        completed.append(
            {
                "dataset": dataset,
                "horizon": horizon,
                "checkpoint": str(checkpoint),
                "profile": str(profile_json),
                "variant": str(args.variant),
                "val_rc": int(rc),
            }
        )

    status = {"completed": completed, "skipped": skipped}
    status_path = out_root / f"pems_k{int(args.n_clusters)}_batch_status.json"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"Wrote: {status_path}", flush=True)


if __name__ == "__main__":
    main()
