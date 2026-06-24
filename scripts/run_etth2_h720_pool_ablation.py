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


BASE_CONFIG = (
    ROOT
    / "outputs"
    / "cluster_penalty_prior_probe"
    / "configs"
    / "ETTh2_H720_channel_head_feature_w10_channel_prior_top1_select1_fusion_bs32.yaml"
)


POOL_VARIANTS: list[dict[str, Any]] = [
    {
        "label": "current_jald_top1",
        "penalties": ["jump", "amp_under", "level", "delta"],
        "cluster_topk": 1,
        "channel_topk": 1,
        "select_ranks": [1],
    },
    {
        "label": "current_jald_top2",
        "penalties": ["jump", "amp_under", "level", "delta"],
        "cluster_topk": 2,
        "channel_topk": 2,
        "select_ranks": [1, 2],
    },
    {
        "label": "no_level_jadd_top1",
        "penalties": ["jump", "amp_under", "delta", "diff_amp"],
        "cluster_topk": 1,
        "channel_topk": 1,
        "select_ranks": [1],
    },
    {
        "label": "no_level_jadd_top2",
        "penalties": ["jump", "amp_under", "delta", "diff_amp"],
        "cluster_topk": 2,
        "channel_topk": 2,
        "select_ranks": [1, 2],
    },
    {
        "label": "amp_delta_top2",
        "penalties": ["amp_under", "delta"],
        "cluster_topk": 2,
        "channel_topk": 2,
        "select_ranks": [1, 2],
    },
    {
        "label": "amp_delta_diff_dir_top1",
        "penalties": ["amp_under", "delta", "diff_amp", "direction"],
        "cluster_topk": 1,
        "channel_topk": 1,
        "select_ranks": [1],
    },
    {
        "label": "lddf_top1",
        "penalties": ["level", "delta", "d2_match", "diff_amp"],
        "cluster_topk": 1,
        "channel_topk": 1,
        "select_ranks": [1],
    },
    {
        "label": "delta_trend_dir_top1",
        "penalties": ["delta", "trend", "direction"],
        "cluster_topk": 1,
        "channel_topk": 1,
        "select_ranks": [1],
    },
]


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def posix(path: Path) -> str:
    return str(path).replace("\\", "/")


def rel(path: Path) -> str:
    try:
        return posix(path.relative_to(ROOT))
    except ValueError:
        return posix(path)


def metric(summary: dict[str, Any], split: str, name: str) -> Any:
    block = summary.get(split, {})
    if isinstance(block, dict):
        return block.get(f"avg_{name}", block.get(name))
    return None


def set_path_fields(cfg: dict[str, Any], run_dir: Path) -> None:
    cfg["exp"]["out_dir"] = rel(run_dir)
    cfg["corr"]["save_path"] = rel(run_dir / "corr.npy")
    cfg["portrait"]["out_dir"] = rel(run_dir / "cluster_portraits")
    cfg["memory"]["path"] = rel(run_dir / "cluster_memory.pt")
    cfg["memory"]["checkpoint_path"] = rel(run_dir / "best_checkpoint.pt")


def set_penalty_pool(cfg: dict[str, Any], variant: dict[str, Any]) -> None:
    penalties = list(variant["penalties"])
    cfg["penalties"]["enabled"] = penalties
    cfg["moe"]["select_ranks"] = list(variant["select_ranks"])
    cfg["moe"]["lambda_init"] = {p: 0.15 for p in penalties}
    cfg["moe"]["lambda_min"] = {p: 0.0 for p in penalties}
    cfg["moe"]["lambda_schedule"] = {p: "none" for p in penalties}
    cfg["moe"]["cluster_penalty_prior"]["topk"] = int(variant["cluster_topk"])
    cfg["moe"]["cluster_penalty_prior"]["hard_topk"] = True
    cfg["moe"]["channel_penalty_prior"]["topk"] = int(variant["channel_topk"])
    cfg["moe"]["channel_penalty_prior"]["hard_topk"] = True


def make_config(base: dict[str, Any], variant: dict[str, Any], out_root: Path, device: str) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    label = variant["label"]
    run_dir = out_root / "runs" / label
    cfg["exp"]["name"] = f"ETTh2_H720_pool_{label}"
    cfg["exp"]["seed"] = 2026
    cfg["exp"]["deterministic"] = True
    cfg["exp"]["device"] = device
    set_path_fields(cfg, run_dir)
    set_penalty_pool(cfg, variant)
    return cfg


def load_explainability(run_dir: Path) -> dict[str, Any]:
    csv_path = run_dir / "penalty_explainability.csv"
    result: dict[str, Any] = {}
    if not csv_path.exists():
        return result
    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for split in ("val", "test"):
        split_rows = [r for r in rows if r.get("split") == split]
        if not split_rows:
            continue
        selected = [r for r in split_rows if float(r.get("selected_count") or 0.0) > 0.0]
        result[f"{split}_selected_penalties"] = ";".join(
            f"c{r.get('cluster_id')}:{r.get('penalty')}" for r in selected
        )
        result[f"{split}_selected_gain_mse_sum"] = sum(
            float(r.get("selected_mean_gain_mse") or 0.0) for r in selected
        )
        cluster_seen: set[str] = set()
        weighted_num = 0.0
        weighted_den = 0.0
        for r in split_rows:
            cid = r.get("cluster_id", "")
            if cid in cluster_seen:
                continue
            cluster_seen.add(cid)
            channels = float(r.get("cluster_channels") or 0.0)
            gain = float(r.get("cluster_final_gain_pct") or 0.0)
            weighted_num += channels * gain
            weighted_den += channels
        result[f"{split}_cluster_gain_pct_weighted"] = weighted_num / weighted_den if weighted_den else None
    return result


def summarize(label: str, cfg_path: Path, cfg: dict[str, Any], returncode: int) -> dict[str, Any]:
    run_dir = ROOT / cfg["exp"]["out_dir"]
    summary_path = run_dir / "run_summary.json"
    row: dict[str, Any] = {
        "label": label,
        "returncode": returncode,
        "penalties": ",".join(cfg["penalties"]["enabled"]),
        "cluster_topk": cfg["moe"]["cluster_penalty_prior"]["topk"],
        "channel_topk": cfg["moe"]["channel_penalty_prior"]["topk"],
        "select_ranks": json.dumps(cfg["moe"]["select_ranks"], ensure_ascii=False),
        "config_path": rel(cfg_path),
        "out_dir": cfg["exp"]["out_dir"],
        "summary_path": rel(summary_path),
    }
    if summary_path.exists():
        summary = read_json(summary_path)
        residual = summary.get("moe_residual", {}) if isinstance(summary.get("moe_residual"), dict) else {}
        selection = (
            summary.get("moe_residual_selection", {})
            if isinstance(summary.get("moe_residual_selection"), dict)
            else {}
        )
        row.update(
            {
                "val_mse": metric(summary, "val", "mse"),
                "val_mae": metric(summary, "val", "mae"),
                "test_mse": metric(summary, "test", "mse"),
                "test_mae": metric(summary, "test", "mae"),
                "best_epoch": summary.get("best_epoch"),
                "alpha_mean": residual.get("alpha_mean"),
                "residual_base_rms_ratio": residual.get("residual_base_rms_ratio"),
                "effective_route_by_penalty": json.dumps(
                    residual.get("effective_route_by_penalty", {}), ensure_ascii=False
                ),
                "val_pred_base_mse": selection.get("val_pred_base_avg_mse"),
                "val_residual_mse": selection.get("val_residual_avg_mse"),
                "val_scaled_mse": selection.get("val_scaled_avg_mse"),
            }
        )
        row.update(load_explainability(run_dir))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default=str(BASE_CONFIG))
    parser.add_argument("--out-root", default="outputs/etth2_h720_pool_ablation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0, help="0 means run all variants.")
    parser.add_argument("--labels", nargs="*", default=None)
    args = parser.parse_args()

    base = read_yaml(Path(args.base_config))
    out_root = ROOT / args.out_root
    cfg_dir = out_root / "configs"
    rows: list[dict[str, Any]] = []
    ran = 0
    label_filter = set(args.labels or [])

    for variant in POOL_VARIANTS:
        label = variant["label"]
        if label_filter and label not in label_filter:
            continue
        cfg = make_config(base, variant, out_root, args.device)
        cfg_path = cfg_dir / f"{label}.yaml"
        write_yaml(cfg_path, cfg)
        run_dir = ROOT / cfg["exp"]["out_dir"]
        summary_path = run_dir / "run_summary.json"
        returncode = 0
        if args.prepare_only:
            print(f"[prepared] {cfg_path}")
        elif args.reuse_existing and summary_path.exists():
            print(f"[reuse] {summary_path}")
        elif args.max_runs and ran >= args.max_runs:
            print(f"[prepared-after-max] {cfg_path}")
        else:
            print(f"[run] {label}: {cfg_path}", flush=True)
            completed = subprocess.run(
                [args.python, "-u", "-m", "src.train", "--config", str(cfg_path)],
                cwd=ROOT,
            )
            returncode = completed.returncode
            ran += 1
        rows.append(summarize(label, cfg_path, cfg, returncode))

    out_root.mkdir(parents=True, exist_ok=True)
    result_path = out_root / "pool_results.csv"
    fieldnames = [
        "label",
        "returncode",
        "penalties",
        "cluster_topk",
        "channel_topk",
        "select_ranks",
        "val_mse",
        "val_mae",
        "test_mse",
        "test_mae",
        "best_epoch",
        "alpha_mean",
        "residual_base_rms_ratio",
        "effective_route_by_penalty",
        "val_pred_base_mse",
        "val_residual_mse",
        "val_scaled_mse",
        "val_selected_penalties",
        "val_selected_gain_mse_sum",
        "val_cluster_gain_pct_weighted",
        "test_selected_penalties",
        "test_selected_gain_mse_sum",
        "test_cluster_gain_pct_weighted",
        "config_path",
        "summary_path",
        "out_dir",
    ]
    with result_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {result_path}")


if __name__ == "__main__":
    main()
