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


VARIANTS: list[dict[str, Any]] = [
    {
        "label": "current_guard_rel0",
        "penalties": ["jump", "amp_under", "level", "delta"],
        "selection_policy": "val_mse_gate_guarded",
        "min_rel": 0.0,
        "alpha_scale": 1.6,
        "fusion_init": -1.5,
        "topk": 1,
        "select_ranks": [1],
    },
    {
        "label": "current_guard_rel002",
        "penalties": ["jump", "amp_under", "level", "delta"],
        "selection_policy": "val_mse_gate_guarded",
        "min_rel": 0.002,
        "alpha_scale": 1.6,
        "fusion_init": -1.5,
        "topk": 1,
        "select_ranks": [1],
    },
    {
        "label": "current_guard_soft_alpha08",
        "penalties": ["jump", "amp_under", "level", "delta"],
        "selection_policy": "val_mse_gate_guarded",
        "min_rel": 0.0,
        "alpha_scale": 0.8,
        "fusion_init": -2.0,
        "topk": 1,
        "select_ranks": [1],
    },
    {
        "label": "current_guard_alpha06_fusion25",
        "penalties": ["jump", "amp_under", "level", "delta"],
        "selection_policy": "val_mse_gate_guarded",
        "min_rel": 0.0,
        "alpha_scale": 0.6,
        "fusion_init": -2.5,
        "topk": 1,
        "select_ranks": [1],
    },
    {
        "label": "current_guard_gate_max05",
        "penalties": ["jump", "amp_under", "level", "delta"],
        "selection_policy": "val_mse_gate_guarded",
        "min_rel": 0.0,
        "alpha_scale": 1.6,
        "fusion_init": -1.5,
        "topk": 1,
        "select_ranks": [1],
        "gate_max_scale": 0.5,
    },
    {
        "label": "current_val_scale_max05",
        "penalties": ["jump", "amp_under", "level", "delta"],
        "selection_policy": "val_mse_scale",
        "min_rel": 0.0,
        "alpha_scale": 1.6,
        "fusion_init": -1.5,
        "topk": 1,
        "select_ranks": [1],
        "selection_scale_max": 0.5,
        "selection_scale_steps": 6,
    },
    {
        "label": "current_soft_prior_guard",
        "penalties": ["jump", "amp_under", "level", "delta"],
        "selection_policy": "val_mse_gate_guarded",
        "min_rel": 0.0,
        "alpha_scale": 1.2,
        "fusion_init": -2.0,
        "topk": 0,
        "hard_topk": False,
        "logit_strength": 1.0,
        "select_ranks": [1],
    },
    {
        "label": "current_soft_prior_alpha06",
        "penalties": ["jump", "amp_under", "level", "delta"],
        "selection_policy": "val_mse_gate_guarded",
        "min_rel": 0.0,
        "alpha_scale": 0.6,
        "fusion_init": -2.5,
        "topk": 0,
        "hard_topk": False,
        "logit_strength": 1.0,
        "select_ranks": [1],
    },
    {
        "label": "lddf_soft_prior_guard",
        "penalties": ["level", "delta", "d2_match", "diff_amp"],
        "selection_policy": "val_mse_gate_guarded",
        "min_rel": 0.0,
        "alpha_scale": 1.2,
        "fusion_init": -2.0,
        "topk": 0,
        "hard_topk": False,
        "logit_strength": 1.0,
        "select_ranks": [1],
    },
    {
        "label": "lddf_soft_prior_alpha06",
        "penalties": ["level", "delta", "d2_match", "diff_amp"],
        "selection_policy": "val_mse_gate_guarded",
        "min_rel": 0.0,
        "alpha_scale": 0.6,
        "fusion_init": -2.5,
        "topk": 0,
        "hard_topk": False,
        "logit_strength": 1.0,
        "select_ranks": [1],
    },
    {
        "label": "lddf_soft_prior_scale05",
        "penalties": ["level", "delta", "d2_match", "diff_amp"],
        "selection_policy": "val_mse_scale",
        "min_rel": 0.0,
        "alpha_scale": 1.2,
        "fusion_init": -2.0,
        "topk": 0,
        "hard_topk": False,
        "logit_strength": 1.0,
        "select_ranks": [1],
        "selection_scale_max": 0.5,
        "selection_scale_steps": 6,
    },
    {
        "label": "lddf_soft_prior_scale025",
        "penalties": ["level", "delta", "d2_match", "diff_amp"],
        "selection_policy": "val_mse_scale",
        "min_rel": 0.0,
        "alpha_scale": 1.2,
        "fusion_init": -2.0,
        "topk": 0,
        "hard_topk": False,
        "logit_strength": 1.0,
        "select_ranks": [1],
        "selection_scale_max": 0.25,
        "selection_scale_steps": 6,
    },
    {
        "label": "lddf_soft_prior_gate_max05",
        "penalties": ["level", "delta", "d2_match", "diff_amp"],
        "selection_policy": "val_mse_gate_guarded",
        "min_rel": 0.0,
        "alpha_scale": 1.2,
        "fusion_init": -2.0,
        "topk": 0,
        "hard_topk": False,
        "logit_strength": 1.0,
        "select_ranks": [1],
        "gate_max_scale": 0.5,
    },
    {
        "label": "lddf_soft_prior_minrel02",
        "penalties": ["level", "delta", "d2_match", "diff_amp"],
        "selection_policy": "val_mse_gate_guarded",
        "min_rel": 0.02,
        "alpha_scale": 1.2,
        "fusion_init": -2.0,
        "topk": 0,
        "hard_topk": False,
        "logit_strength": 1.0,
        "select_ranks": [1],
    },
    {
        "label": "lddf_soft_prior_minrel05",
        "penalties": ["level", "delta", "d2_match", "diff_amp"],
        "selection_policy": "val_mse_gate_guarded",
        "min_rel": 0.05,
        "alpha_scale": 1.2,
        "fusion_init": -2.0,
        "topk": 0,
        "hard_topk": False,
        "logit_strength": 1.0,
        "select_ranks": [1],
    },
    {
        "label": "lddf_soft_prior_train_calib",
        "penalties": ["level", "delta", "d2_match", "diff_amp"],
        "selection_policy": "val_mse_gate_guarded",
        "min_rel": 0.0,
        "alpha_scale": 1.2,
        "fusion_init": -2.0,
        "topk": 0,
        "hard_topk": False,
        "logit_strength": 1.0,
        "select_ranks": [1],
        "gate_source_split": "train",
    },
    {
        "label": "lddf_soft_prior_train_calib_minrel02",
        "penalties": ["level", "delta", "d2_match", "diff_amp"],
        "selection_policy": "val_mse_gate_guarded",
        "min_rel": 0.02,
        "alpha_scale": 1.2,
        "fusion_init": -2.0,
        "topk": 0,
        "hard_topk": False,
        "logit_strength": 1.0,
        "select_ranks": [1],
        "gate_source_split": "train",
    },
    {
        "label": "delta_trend_soft_prior_guard",
        "penalties": ["delta", "trend", "direction"],
        "selection_policy": "val_mse_gate_guarded",
        "min_rel": 0.0,
        "alpha_scale": 1.2,
        "fusion_init": -2.0,
        "topk": 0,
        "hard_topk": False,
        "logit_strength": 1.0,
        "select_ranks": [1],
    },
    {
        "label": "current_val_scale",
        "penalties": ["jump", "amp_under", "level", "delta"],
        "selection_policy": "val_mse_scale",
        "min_rel": 0.0,
        "alpha_scale": 1.6,
        "fusion_init": -1.5,
        "topk": 1,
        "select_ranks": [1],
        "selection_scale_steps": 11,
    },
    {
        "label": "lddf_guard_rel0",
        "penalties": ["level", "delta", "d2_match", "diff_amp"],
        "selection_policy": "val_mse_gate_guarded",
        "min_rel": 0.0,
        "alpha_scale": 1.6,
        "fusion_init": -1.5,
        "topk": 1,
        "select_ranks": [1],
    },
    {
        "label": "delta_trend_guard_rel0",
        "penalties": ["delta", "trend", "direction"],
        "selection_policy": "val_mse_gate_guarded",
        "min_rel": 0.0,
        "alpha_scale": 1.6,
        "fusion_init": -1.5,
        "topk": 1,
        "select_ranks": [1],
    },
    {
        "label": "no_level_guard_rel0",
        "penalties": ["jump", "amp_under", "delta", "diff_amp"],
        "selection_policy": "val_mse_gate_guarded",
        "min_rel": 0.0,
        "alpha_scale": 1.6,
        "fusion_init": -1.5,
        "topk": 1,
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


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def metric(summary: dict[str, Any], split: str, name: str) -> Any:
    block = summary.get(split, {})
    if isinstance(block, dict):
        return block.get(f"avg_{name}", block.get(name))
    return None


def set_paths(cfg: dict[str, Any], run_dir: Path) -> None:
    cfg["exp"]["out_dir"] = rel(run_dir)
    cfg["corr"]["save_path"] = rel(run_dir / "corr.npy")
    cfg["portrait"]["out_dir"] = rel(run_dir / "cluster_portraits")
    cfg["knn_hybrid"]["path"] = rel(run_dir / "knn_shape_bank.pt")
    cfg["memory"]["path"] = rel(run_dir / "cluster_memory.pt")
    cfg["memory"]["checkpoint_path"] = rel(run_dir / "best_checkpoint.pt")


def apply_variant(cfg: dict[str, Any], variant: dict[str, Any]) -> None:
    penalties = list(variant["penalties"])
    cfg["penalties"]["enabled"] = penalties
    cfg["moe"]["select_ranks"] = list(variant["select_ranks"])
    cfg["moe"]["lambda_init"] = {p: 0.15 for p in penalties}
    cfg["moe"]["lambda_min"] = {p: 0.0 for p in penalties}
    cfg["moe"]["lambda_schedule"] = {p: "none" for p in penalties}
    cfg["moe"]["cluster_penalty_prior"]["topk"] = int(variant["topk"])
    cfg["moe"]["channel_penalty_prior"]["topk"] = int(variant["topk"])
    hard_topk = bool(variant.get("hard_topk", True))
    cfg["moe"]["cluster_penalty_prior"]["hard_topk"] = hard_topk
    cfg["moe"]["channel_penalty_prior"]["hard_topk"] = hard_topk
    if "logit_strength" in variant:
        cfg["moe"]["cluster_penalty_prior"]["logit_strength"] = float(variant["logit_strength"])
    pred_res = cfg["moe"]["pred_side_residual"]
    pred_res["selection_policy"] = str(variant["selection_policy"])
    pred_res["selection_min_rel_improvement"] = float(variant["min_rel"])
    pred_res["selection_min_abs_improvement"] = 0.0
    pred_res["alpha_scale"] = float(variant["alpha_scale"])
    pred_res["fusion_init"] = float(variant["fusion_init"])
    gate_calib = pred_res.get("gate_calibrator", {})
    if "gate_max_scale" in variant:
        gate_calib["max_scale"] = float(variant["gate_max_scale"])
    if "gate_scale_reg" in variant:
        gate_calib["scale_reg"] = float(variant["gate_scale_reg"])
    if "gate_source_split" in variant:
        gate_calib["source_split"] = str(variant["gate_source_split"])
    pred_res["gate_calibrator"] = gate_calib
    if "selection_scale_steps" in variant:
        pred_res["selection_scale_min"] = 0.0
        pred_res["selection_scale_max"] = float(variant.get("selection_scale_max", 1.0))
        pred_res["selection_scale_steps"] = int(variant["selection_scale_steps"])


def make_config(base: dict[str, Any], variant: dict[str, Any], out_root: Path, device: str) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    label = str(variant["label"])
    run_dir = out_root / "runs" / label
    cfg["exp"]["name"] = f"ETTh2_H720_guard_param_{label}"
    cfg["exp"]["device"] = device
    set_paths(cfg, run_dir)
    apply_variant(cfg, variant)
    return cfg


def summarize(label: str, cfg_path: Path, cfg: dict[str, Any], returncode: int) -> dict[str, Any]:
    run_dir = ROOT / cfg["exp"]["out_dir"]
    summary_path = run_dir / "run_summary.json"
    row: dict[str, Any] = {
        "label": label,
        "returncode": returncode,
        "penalties": ",".join(cfg["penalties"]["enabled"]),
        "selection_policy": cfg["moe"]["pred_side_residual"]["selection_policy"],
        "min_rel": cfg["moe"]["pred_side_residual"].get("selection_min_rel_improvement"),
        "alpha_scale": cfg["moe"]["pred_side_residual"].get("alpha_scale"),
        "fusion_init": cfg["moe"]["pred_side_residual"].get("fusion_init"),
        "config_path": rel(cfg_path),
        "out_dir": cfg["exp"]["out_dir"],
    }
    if not summary_path.exists():
        return row
    summary = read_json(summary_path)
    residual = summary.get("moe_residual", {}) if isinstance(summary.get("moe_residual"), dict) else {}
    selection = (
        summary.get("moe_residual_selection", {})
        if isinstance(summary.get("moe_residual_selection"), dict)
        else {}
    )
    hit = summary.get("moe_gate_penalty_hit", {}) if isinstance(summary.get("moe_gate_penalty_hit"), dict) else {}
    test_hit = hit.get("test", {}) if isinstance(hit.get("test"), dict) else {}
    val_hit = hit.get("val", {}) if isinstance(hit.get("val"), dict) else {}
    val_base = selection.get("val_pred_base_avg_mse")
    val_scaled = selection.get("val_scaled_avg_mse")
    val_residual = selection.get("val_residual_avg_mse")
    row.update(
        {
            "val_mse": metric(summary, "val", "mse"),
            "test_mse": metric(summary, "test", "mse"),
            "test_mae": metric(summary, "test", "mae"),
            "best_epoch": json.dumps(summary.get("best_epoch"), ensure_ascii=False),
            "residual_base_rms_ratio": residual.get("residual_base_rms_ratio"),
            "effective_route_by_penalty": json.dumps(residual.get("effective_route_by_penalty", {}), ensure_ascii=False),
            "num_residual_channels": selection.get("num_residual_channels"),
            "residual_channels": ",".join(selection.get("residual_channels", []) or []),
            "base_channels": ",".join(selection.get("base_channels", []) or []),
            "val_pred_base_mse": val_base,
            "val_residual_mse": val_residual,
            "val_scaled_mse": val_scaled,
            "val_scaled_gain_pct": (
                (float(val_base) - float(val_scaled)) / max(abs(float(val_base)), 1.0e-12) * 100.0
                if val_base is not None and val_scaled is not None
                else None
            ),
            "val_raw_residual_gain_pct": (
                (float(val_base) - float(val_residual)) / max(abs(float(val_base)), 1.0e-12) * 100.0
                if val_base is not None and val_residual is not None
                else None
            ),
            "test_oracle_gain_pct": test_hit.get("oracle_gain_pct_vs_base"),
            "test_selected_top1_gain_pct": test_hit.get("selected_top1_gain_pct_vs_base"),
            "test_hit_rate": test_hit.get("top1_hit_rate_all"),
            "val_oracle_gain_pct": val_hit.get("oracle_gain_pct_vs_base"),
            "val_selected_top1_gain_pct": val_hit.get("selected_top1_gain_pct_vs_base"),
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default=str(BASE_CONFIG))
    parser.add_argument("--out-root", default="outputs/etth2_h720_guard_param_search")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    base = read_yaml(Path(args.base_config))
    out_root = ROOT / args.out_root
    cfg_dir = out_root / "configs"
    label_filter = set(args.labels or [])
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        label = str(variant["label"])
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
        else:
            print(f"[run] {label}: {cfg_path}", flush=True)
            completed = subprocess.run([args.python, "-u", "-m", "src.train", "--config", str(cfg_path)], cwd=ROOT)
            returncode = completed.returncode
        rows.append(summarize(label, cfg_path, cfg, returncode))

    out_root.mkdir(parents=True, exist_ok=True)
    result_path = out_root / "guard_param_results.csv"
    fieldnames = [
        "label",
        "returncode",
        "penalties",
        "selection_policy",
        "min_rel",
        "alpha_scale",
        "fusion_init",
        "val_mse",
        "test_mse",
        "test_mae",
        "best_epoch",
        "residual_base_rms_ratio",
        "effective_route_by_penalty",
        "num_residual_channels",
        "residual_channels",
        "base_channels",
        "val_pred_base_mse",
        "val_residual_mse",
        "val_scaled_mse",
        "val_scaled_gain_pct",
        "val_raw_residual_gain_pct",
        "test_oracle_gain_pct",
        "test_selected_top1_gain_pct",
        "test_hit_rate",
        "val_oracle_gain_pct",
        "val_selected_top1_gain_pct",
        "config_path",
        "out_dir",
    ]
    with result_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {result_path}")


if __name__ == "__main__":
    main()
