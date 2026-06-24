from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compute_train_residual_penalty_portrait import (  # noqa: E402
    PENALTY_NAMES,
    CellSpec,
    run_cell,
)


OUT_ROOT = ROOT / "outputs" / "anchorless_moe_diagnostic"


CELL_SPECS: dict[str, dict[str, str]] = {
    "ETTh1_H96": {
        "dataset": "ETTh1",
        "full_config": "outputs/input96_mse_gate_cluster_moe_retrain_20260616/configs/ETTh1/H96/mse_gate_w002_softprior.yaml",
        "checkpoint_path": "outputs/fresh_input_len96_20260610_etth1_ettm1_backbone_probe/runs/ETTh1/H96/common_backbone_h96/mlp_h128_do0_wd1e4_mae04/best_checkpoint.pt",
    },
    "ETTm2_H96": {
        "dataset": "ETTm2",
        "full_config": "outputs/input96_mse_gate_cluster_moe_retrain_20260616/configs/ETTm2/H96/mse_gate_w002_top2.yaml",
        "checkpoint_path": "outputs/fresh_input_len96_20260610_ettm2_backbone_lowdrop/runs/ETTm2/H96/common_backbone_h96/channel_h256_do0_wd1e3_mae06/best_checkpoint.pt",
    },
}


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=False, sort_keys=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def disable_anchor_blocks(moe: dict[str, Any]) -> None:
    for key in ("history_anchor_expert", "train_stat_anchor_expert", "train_residual_anchor_expert"):
        block = copy.deepcopy(moe.get(key, {}) or {})
        block["enable"] = False
        moe[key] = block


def localize(cfg: dict[str, Any], out_dir: Path, run_name: str) -> None:
    cfg.setdefault("exp", {})["name"] = run_name
    cfg["exp"]["out_dir"] = str(out_dir)
    cfg.setdefault("corr", {})["save_path"] = str(out_dir / "corr.npy")
    cfg.setdefault("plot", {})["enable"] = False
    cfg.setdefault("portrait", {})["enable"] = False
    cfg["portrait"]["out_dir"] = str(out_dir / "cluster_portraits")
    cfg.setdefault("memory", {})["enable"] = False
    cfg["memory"]["save_checkpoint"] = False
    cfg["memory"]["path"] = str(out_dir / "cluster_memory.pt")
    cfg["memory"]["checkpoint_path"] = str(out_dir / "best_checkpoint.pt")
    cfg.setdefault("eval", {})["skip_test"] = True


def normalized_scores(payload: dict[str, Any]) -> np.ndarray:
    raw = np.asarray(payload["portrait_raw"], dtype=np.float64)
    global_mean = np.asarray(payload["penalty_global_mean"], dtype=np.float64)
    return raw / np.maximum(global_mean.reshape(1, -1), 1.0e-12)


def diagnostic_pools(payload: dict[str, Any]) -> dict[str, list[str]]:
    scores = normalized_scores(payload)
    mean_score = scores.mean(axis=0)
    corr = np.asarray(payload.get("penalty_mse_corr", []), dtype=np.float64)
    corr_cfg = payload.get("mse_corr_exclusion", {}) or {}
    max_abs_corr = corr_cfg.get("max_abs_corr", None)
    blocked: set[int] = set()
    if corr.shape == (len(PENALTY_NAMES),) and max_abs_corr is not None:
        threshold = float(max_abs_corr)
        blocked = {
            idx
            for idx, value in enumerate(corr)
            if np.isfinite(value) and abs(float(value)) > threshold
        }
    global_order = sorted(
        (idx for idx in range(len(PENALTY_NAMES)) if idx not in blocked),
        key=lambda idx: (-float(mean_score[idx]), idx),
    )
    global3 = [PENALTY_NAMES[idx] for idx in global_order[:3]]

    selected = payload["selected_pool_top3"]
    rank_stats: dict[str, dict[str, float]] = {}
    for names in selected.values():
        for rank, name in enumerate(names):
            stats = rank_stats.setdefault(str(name), {"count": 0.0, "rank_sum": 0.0})
            stats["count"] += 1.0
            stats["rank_sum"] += float(rank)
    union_order = sorted(
        rank_stats,
        key=lambda name: (
            -rank_stats[name]["count"],
            rank_stats[name]["rank_sum"] / max(rank_stats[name]["count"], 1.0),
            PENALTY_NAMES.index(name),
        ),
    )
    return {
        "diag_global3": global3,
        "diag_union": union_order,
        "truth10": list(PENALTY_NAMES),
    }


def compute_portraits(device_text: str, batch_size: int, reuse_existing: bool) -> dict[str, Any]:
    portrait_path = OUT_ROOT / "penalty_portrait_anchorless_targets.json"
    if reuse_existing and portrait_path.exists():
        return read_json(portrait_path)
    use_cuda = torch.cuda.is_available() and str(device_text).startswith("cuda")
    device = torch.device(device_text if use_cuda else "cpu")
    cells: dict[str, Any] = {}
    for cell, spec in CELL_SPECS.items():
        print(f"=== portrait {cell} train y_base_vs_y device={device} ===", flush=True)
        cells[cell] = run_cell(
            CellSpec(
                name=cell,
                config_path=spec["full_config"],
                checkpoint_path=spec["checkpoint_path"],
            ),
            penalty_names=list(PENALTY_NAMES),
            batch_size=batch_size if batch_size > 0 else None,
            device=device,
            materialize_windows=False,
        )
    payload = {
        "meta": {
            "input_len": 96,
            "split": "train",
            "target": "ETTh1/ETTm2 frozen backbone residual error for anchorless MoE penalty-pool diagnosis",
        },
        "penalty_names": list(PENALTY_NAMES),
        "cells": cells,
    }
    write_json(portrait_path, payload)
    return payload


def current_pool(cfg: dict[str, Any]) -> list[str]:
    return [str(v) for v in cfg.get("penalties", {}).get("enabled", [])]


def build_variant_config(
    cell: str,
    variant: str,
    base_cfg: dict[str, Any],
    penalties: list[str],
) -> tuple[Path, Path]:
    cfg = copy.deepcopy(base_cfg)
    out_dir = OUT_ROOT / "runs" / cell / variant
    config_path = OUT_ROOT / "configs" / cell / f"{variant}.yaml"
    localize(cfg, out_dir, f"ANCHORLESS_DIAG_{cell}_{variant}")
    cfg.setdefault("penalties", {})["enabled"] = list(penalties)
    train = cfg.setdefault("train", {})
    moe = cfg.setdefault("moe", {})
    pred = moe.setdefault("pred_side_residual", {})
    moe["enable"] = True
    moe["freeze_backbone"] = True
    pred["enable"] = True
    disable_anchor_blocks(moe)
    moe["lambda_init"] = {"default": 0.0}
    moe["lambda_min"] = {"default": 0.0}

    if variant.endswith("_e3"):
        train["epochs"] = 3
    elif variant.endswith("_e5"):
        train["epochs"] = 5
    else:
        train["epochs"] = 1

    if "alpha_hi" in variant:
        pred["init_alpha"] = -1.5
        pred["alpha_scale"] = 2.0
    if "alpha1" in variant:
        pred["init_alpha"] = 4.0
        pred["alpha_scale"] = 1.0
    if "noskip" in variant:
        moe["allow_skip"] = False
        moe["skip_cost"] = 0.0
        moe["skip_supervision_weight"] = 0.0
    if "routeall" in variant:
        moe["select_ranks"] = list(range(1, len(penalties) + 1))
    if "nospec" in variant:
        pred["specialization_weight"] = 0.0
        pred["intervention_weight"] = 0.0
    if "directloss" in variant:
        train["mse_weight"] = 1.0
        train.setdefault("mae_objective", {})["weight"] = 0.0
        moe.setdefault("mse_utility_gate_supervision", {})["weight"] = 0.0
    if "wd0" in variant:
        train["weight_decay"] = 0.0
    if "hidden64" in variant:
        pred["corrector_hidden"] = 64
    if "safeaug" in variant:
        pred["feature_mode"] = "safe_augmented"
    if "nogatebce" in variant:
        gate_cal["activation_bce_weight"] = 0.0
        gate_cal["activation_inactive_scale_weight"] = 0.0

    write_yaml(config_path, cfg)
    return config_path, out_dir


def run_config(config_path: Path, out_dir: Path, python_exe: str, reuse_existing: bool) -> dict[str, Any]:
    summary_path = out_dir / "run_summary.json"
    if reuse_existing and summary_path.exists():
        return {"status": "reused", "returncode": 0, "seconds": 0.0}
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [python_exe, "-u", "-m", "src.train", "--config", str(config_path)]
    stdout_path = out_dir / "stdout.log"
    stderr_path = out_dir / "stderr.log"
    start = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=stdout_f, stderr=stderr_f, text=True, env=env)
    seconds = time.perf_counter() - start
    if proc.returncode != 0:
        return {
            "status": "failed",
            "returncode": int(proc.returncode),
            "seconds": seconds,
            "stderr_tail": stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:],
        }
    return {"status": "ok", "returncode": 0, "seconds": seconds}


def metric_block(summary: dict[str, Any]) -> dict[str, Any]:
    val = summary.get("val") or {}
    residual = summary.get("moe_residual_selection") or {}
    return {
        "raw_val_mse": val.get("avg_mse"),
        "raw_val_mae": val.get("avg_mae"),
        "selected_val_mse": residual.get("val_scaled_avg_mse", val.get("avg_mse")),
        "selected_val_mae": residual.get("val_scaled_avg_mae", val.get("avg_mae")),
        "base_val_mse": residual.get("val_pred_base_avg_mse"),
        "base_val_mae": residual.get("val_pred_base_avg_mae"),
        "residual_val_mse": residual.get("val_residual_avg_mse"),
        "residual_val_mae": residual.get("val_residual_avg_mae"),
        "mean_scale": residual.get("mean_scale"),
        "num_residual_channels": residual.get("num_residual_channels"),
        "residual_channels": residual.get("residual_channels"),
        "penalty_names": summary.get("penalty_names"),
        "moe_residual": summary.get("moe_residual"),
    }


def summarize(manifest: dict[str, Any]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for cell, payload in manifest["cells"].items():
        variants: dict[str, Any] = {}
        baseline_mse = None
        baseline_mae = None
        for variant, paths in payload["variants"].items():
            summary_path = Path(paths["summary_path"])
            summary = read_json(summary_path)
            block = metric_block(summary)
            if variant == "current_e1":
                baseline_mse = float(block["selected_val_mse"])
                baseline_mae = float(block["selected_val_mae"])
            variants[variant] = {
                **paths,
                **block,
            }
        if baseline_mse is not None and baseline_mae is not None:
            for values in variants.values():
                mse = float(values["selected_val_mse"])
                mae = float(values["selected_val_mae"])
                values["delta_vs_current_e1_mse"] = mse - baseline_mse
                values["delta_vs_current_e1_mae"] = mae - baseline_mae
                values["reduction_vs_current_e1_mse_pct"] = (
                    (baseline_mse - mse) / baseline_mse * 100.0 if baseline_mse else 0.0
                )
                values["reduction_vs_current_e1_mae_pct"] = (
                    (baseline_mae - mae) / baseline_mae * 100.0 if baseline_mae else 0.0
                )
        best_by_mse = min(variants, key=lambda name: float(variants[name]["selected_val_mse"]))
        best_by_mae = min(variants, key=lambda name: float(variants[name]["selected_val_mae"]))
        cells[cell] = {
            "dataset": CELL_SPECS[cell]["dataset"],
            "variants": variants,
            "best_by_val_mse": best_by_mse,
            "best_by_val_mae": best_by_mae,
        }
    return {
        "meta": {
            "input_len": 96,
            "anchors": "disabled for all generated variants",
            "eval": "val-only; eval.skip_test=true for generated runs",
            "goal": "Diagnose whether anchorless MoE weakness is due to parameters/features or penalty pool.",
        },
        "cells": cells,
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Anchorless MoE Diagnostic",
        "",
        "All generated runs use `eval.skip_test: true`; metrics below are validation only.",
        "",
        "| cell | variant | penalties | val MSE | val MAE | MSE vs current_e1 | MAE vs current_e1 | residual raw MSE | channels |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cell, payload in summary["cells"].items():
        for variant, values in payload["variants"].items():
            penalties = ",".join(values.get("penalty_names") or [])
            lines.append(
                f"| {cell} | {variant} | {penalties} | "
                f"{float(values['selected_val_mse']):.6f} | {float(values['selected_val_mae']):.6f} | "
                f"{float(values.get('reduction_vs_current_e1_mse_pct', 0.0)):.3f}% | "
                f"{float(values.get('reduction_vs_current_e1_mae_pct', 0.0)):.3f}% | "
                f"{float(values['residual_val_mse']):.6f} | {values.get('num_residual_channels')} |"
            )
    lines.extend(["", "## Best", ""])
    for cell, payload in summary["cells"].items():
        lines.append(
            f"- {cell}: best val MSE = `{payload['best_by_val_mse']}`, "
            f"best val MAE = `{payload['best_by_val_mae']}`."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Anchor-off ETTh1/ETTm2 MoE parameter and penalty-pool diagnostic.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    portraits = compute_portraits(
        device_text=str(args.device),
        batch_size=int(args.batch_size),
        reuse_existing=bool(args.reuse_existing),
    )
    manifest: dict[str, Any] = {
        "meta": {
            "input_len": 96,
            "anchors": "disabled",
            "eval_skip_test": True,
            "portrait_path": str(OUT_ROOT / "penalty_portrait_anchorless_targets.json"),
        },
        "cells": {},
    }

    for cell, spec in CELL_SPECS.items():
        base_cfg = read_yaml(resolve(spec["full_config"]))
        pools = diagnostic_pools(portraits["cells"][cell])
        pools["current"] = current_pool(base_cfg)
        variant_penalties = {
            "current_e1": pools["current"],
            "single_alpha_hi_nospec_directloss_e5": pools["current"][:1],
            "single_alpha_hi_noskip_nospec_directloss_e5": pools["current"][:1],
            "single_alpha1_noskip_nospec_directloss_e5": pools["current"][:1],
            "single_alpha1_noskip_nospec_directloss_wd0_e5": pools["current"][:1],
            "current_e3": pools["current"],
            "current_alpha_hi_e1": pools["current"],
            "current_alpha_hi_e3": pools["current"],
            "current_alpha_hi_routeall_nospec_e5": pools["current"],
            "current_alpha_hi_routeall_nospec_directloss_e5": pools["current"],
            "current_alpha1_noskip_nospec_directloss_wd0_e5": pools["current"],
            "current_hidden64_e1": pools["current"],
            "current_safeaug_e1": pools["current"],
            "diag_global3_e1": pools["diag_global3"],
            "diag_global3_alpha_hi_e1": pools["diag_global3"],
            "diag_global3_alpha_hi_routeall_nospec_e5": pools["diag_global3"],
            "diag_global3_alpha_hi_routeall_nospec_directloss_e5": pools["diag_global3"],
            "diag_global3_alpha1_noskip_nospec_directloss_wd0_e5": pools["diag_global3"],
            "diag_union_e1": pools["diag_union"],
            "diag_union_alpha_hi_e1": pools["diag_union"],
            "truth10_e1": pools["truth10"],
            "truth10_alpha_hi_e1": pools["truth10"],
        }
        manifest["cells"][cell] = {
            "pools": pools,
            "variants": {},
        }
        for variant, penalties in variant_penalties.items():
            cfg_path, out_dir = build_variant_config(cell, variant, base_cfg, penalties)
            manifest["cells"][cell]["variants"][variant] = {
                "config_path": str(cfg_path),
                "out_dir": str(out_dir),
                "summary_path": str(out_dir / "run_summary.json"),
                "penalties": list(penalties),
            }

    write_json(OUT_ROOT / "anchorless_moe_diagnostic_manifest.json", manifest)
    if args.prepare_only:
        print(f"prepared {OUT_ROOT / 'anchorless_moe_diagnostic_manifest.json'}")
        return

    run_results: dict[str, Any] = {}
    for cell, payload in manifest["cells"].items():
        run_results[cell] = {}
        for variant, paths in payload["variants"].items():
            print(f"=== {cell} {variant} ===", flush=True)
            result = run_config(
                Path(paths["config_path"]),
                Path(paths["out_dir"]),
                str(args.python),
                reuse_existing=bool(args.reuse_existing),
            )
            run_results[cell][variant] = result
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if int(result.get("returncode", 1)) != 0:
                write_json(OUT_ROOT / "anchorless_moe_diagnostic_run_results.json", {"runs": run_results})
                raise SystemExit(f"Run failed: {cell} {variant}")

    write_json(OUT_ROOT / "anchorless_moe_diagnostic_run_results.json", {"runs": run_results})
    summary = summarize(manifest)
    write_json(OUT_ROOT / "anchorless_moe_diagnostic_summary.json", summary)
    write_markdown(summary, OUT_ROOT / "anchorless_moe_diagnostic_summary.md")
    print(f"summary {OUT_ROOT / 'anchorless_moe_diagnostic_summary.json'}")


if __name__ == "__main__":
    main()
