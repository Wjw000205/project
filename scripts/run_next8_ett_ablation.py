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

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "outputs" / "next8_ett_ablation"


ANCHOR_DEFAULTS = {
    "train_stat_anchor_expert": {
        "enable": True,
        "period": 96,
        "alpha": 0.0,
        "mode": "phase_mean",
        "reference": "last",
        "blend_target": "prediction",
        "scale_selection": {
            "enable": True,
            "metric": "mse",
            "max_scale": 0.2,
            "steps": 9,
        },
    },
    "train_residual_anchor_expert": {
        "enable": True,
        "period": 96,
        "alpha": 0.0,
        "blend_target": "prediction",
        "scale_selection": {
            "enable": True,
            "metric": "mse",
            "max_scale": 1.2,
            "steps": 49,
            "horizon_segments": 7,
        },
    },
}


CELL_SPECS: dict[str, dict[str, str]] = {
    "ETTm1_H96": {
        "dataset": "ETTm1",
        "horizon": "96",
        "backbone_summary": "outputs/fresh_input_len96_20260610_ettm1_seasonal_blend_m010_full/runs/ETTm1/H96/light_backbone/mlp_anchor_basis_seasblend_m010_h256_r16_wd1e4_mae06/run_summary.json",
        "full_config": "outputs/input96_mse_gate_cluster_moe_retrain_20260616_ettm1_h96_mlp/configs/ETTm1/H96/mse_gate_w002_strong_safe_mse.yaml",
        "full_summary": "outputs/input96_mse_gate_cluster_moe_retrain_20260616_ettm1_h96_mlp/runs/ETTm1/H96/mse_gate_w002_strong_safe_mse/run_summary.json",
    },
    "ETTm2_H96": {
        "dataset": "ETTm2",
        "horizon": "96",
        "backbone_summary": "outputs/fresh_input_len96_20260610_ettm2_backbone_lowdrop/runs/ETTm2/H96/common_backbone_h96/channel_h256_do0_wd1e3_mae06/run_summary.json",
        "full_config": "outputs/input96_mse_gate_cluster_moe_retrain_20260616/configs/ETTm2/H96/mse_gate_w002_top2.yaml",
        "full_summary": "outputs/input96_mse_gate_cluster_moe_retrain_20260616/runs/ETTm2/H96/mse_gate_w002_top2/run_summary.json",
    },
    "ETTh2_H96": {
        "dataset": "ETTh2",
        "horizon": "96",
        "backbone_summary": "outputs/fresh_input_len96_20260609_etth2_backbone_ckpt/runs/ETTh2/H96/common_backbone_h96/current_model/run_summary.json",
        "full_config": "outputs/codex_table_target_20260614/etth2_h96_safe_aug_mae_refine1/configs/ETTh2/H96/expert_probe/gate_mae_alpha1p2_clip3.yaml",
        "full_summary": "outputs/codex_table_target_20260614/etth2_h96_safe_aug_mae_refine1/runs/ETTh2/H96/expert_probe/gate_mae_alpha1p2_clip3/run_summary.json",
        "generate_full": "true",
        "reuse_full_as_moe_only": "true",
    },
    "ETTh1_H96": {
        "dataset": "ETTh1",
        "horizon": "96",
        "backbone_summary": "outputs/fresh_input_len96_20260610_etth1_ettm1_backbone_probe/runs/ETTh1/H96/common_backbone_h96/mlp_h128_do0_wd1e4_mae04/run_summary.json",
        "full_config": "outputs/input96_mse_gate_cluster_moe_retrain_20260616/configs/ETTh1/H96/mse_gate_w002_softprior.yaml",
        "full_summary": "outputs/input96_mse_gate_cluster_moe_retrain_20260616/runs/ETTh1/H96/mse_gate_w002_softprior/run_summary.json",
        "generate_full": "true",
    },
}


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=False, sort_keys=False)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def localize_paths(cfg: dict[str, Any], out_dir: Path, name: str) -> None:
    cfg.setdefault("exp", {})["name"] = name
    cfg["exp"]["out_dir"] = str(out_dir)
    cfg.setdefault("corr", {})["save_path"] = str(out_dir / "corr.npy")
    cfg.setdefault("plot", {})["enable"] = False
    cfg.setdefault("portrait", {})["enable"] = False
    cfg["portrait"]["out_dir"] = str(out_dir / "cluster_portraits")
    cfg.setdefault("memory", {})["enable"] = False
    cfg["memory"]["save_checkpoint"] = False
    cfg["memory"]["path"] = str(out_dir / "cluster_memory.pt")
    cfg["memory"]["checkpoint_path"] = str(out_dir / "best_checkpoint.pt")
    cfg.setdefault("eval", {})["skip_test"] = False


def ensure_anchors(moe: dict[str, Any]) -> None:
    for key, default in ANCHOR_DEFAULTS.items():
        current = copy.deepcopy(moe.get(key, {}) or {})
        merged = copy.deepcopy(default)
        deep_update(merged, current)
        merged["enable"] = True
        moe[key] = merged


def disable_anchors(moe: dict[str, Any]) -> None:
    for key, default in ANCHOR_DEFAULTS.items():
        current = copy.deepcopy(moe.get(key, {}) or {})
        merged = copy.deepcopy(default)
        deep_update(merged, current)
        merged["enable"] = False
        moe[key] = merged


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def build_stage_config(cell: str, stage: str, base_cfg: dict[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    cfg = copy.deepcopy(base_cfg)
    out_dir = OUT_ROOT / "runs" / cell / stage
    config_path = OUT_ROOT / "configs" / cell / f"{stage}.yaml"
    localize_paths(cfg, out_dir, f"NEXT8_{cell}_{stage}")
    moe = cfg.setdefault("moe", {})
    if stage == "anchors":
        moe["enable"] = True
        moe["freeze_backbone"] = True
        ensure_anchors(moe)
        moe.setdefault("pred_side_residual", {})["enable"] = False
    elif stage == "moe_only_no_anchors":
        moe["enable"] = True
        moe["freeze_backbone"] = True
        disable_anchors(moe)
        moe.setdefault("pred_side_residual", {})["enable"] = True
    elif stage == "full":
        moe["enable"] = True
        moe["freeze_backbone"] = True
        ensure_anchors(moe)
        moe.setdefault("pred_side_residual", {})["enable"] = True
    else:
        raise ValueError(f"Unsupported generated stage: {stage}")
    write_yaml(config_path, cfg)
    return config_path, out_dir, cfg


def run_config(config_path: Path, out_dir: Path, python_exe: str, reuse_existing: bool) -> dict[str, Any]:
    summary_path = out_dir / "run_summary.json"
    if reuse_existing and summary_path.exists():
        return {"status": "reused", "returncode": 0, "seconds": 0.0}
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = out_dir / "stdout.log"
    stderr_path = out_dir / "stderr.log"
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [python_exe, "-u", "-m", "src.train", "--config", str(config_path)]
    start = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=stdout_f, stderr=stderr_f, text=True, env=env)
    seconds = time.perf_counter() - start
    if proc.returncode != 0:
        tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        return {"status": "failed", "returncode": proc.returncode, "seconds": seconds, "stderr_tail": tail}
    return {"status": "ok", "returncode": 0, "seconds": seconds}


def metric_block(summary: dict[str, Any], stage: str) -> dict[str, Any]:
    val = summary.get("val") or {}
    test = summary.get("test") or {}
    residual = summary.get("moe_residual_selection") or {}
    if stage == "full" and residual:
        val_mse = residual.get("val_scaled_avg_mse", val.get("avg_mse"))
        val_mae = residual.get("val_scaled_avg_mae", val.get("avg_mae"))
    else:
        val_mse = val.get("avg_mse")
        val_mae = val.get("avg_mae")
    return {
        "val_mse": float(val_mse),
        "val_mae": float(val_mae),
        "test_mse": float(test.get("avg_mse")),
        "test_mae": float(test.get("avg_mae")),
        "raw_val_mse": float(val.get("avg_mse")) if val.get("avg_mse") is not None else None,
        "raw_val_mae": float(val.get("avg_mae")) if val.get("avg_mae") is not None else None,
        "moe_residual_selection": residual or None,
    }


def contribution(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for split in ("val", "test"):
        for metric in ("mse", "mae"):
            key = f"{split}_{metric}"
            a = float(before[key])
            b = float(after[key])
            out[f"{key}_delta"] = b - a
            out[f"{key}_reduction_pct"] = (a - b) / a * 100.0 if a != 0.0 else 0.0
    return out


def build_manifest(generated: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "meta": {
            "task": "NEXT-8 ETT H96 component ablation",
            "input_len": 96,
            "stages": {
                "a_backbone": "existing val-selected frozen backbone checkpoint run",
                "d_moe_only_no_anchors": "moe.enable=true, pred_side_residual enabled, train_stat/train_residual anchors disabled",
                "b_anchors": "moe.enable=true, train_stat/train_residual anchors enabled, pred_side_residual disabled",
                "c_full": "anchors plus pred_side_residual penalty-MoE gate",
            },
            "selection": "validation-selected; test metrics are read only from the selected stage run_summary",
        },
        "cells": generated,
    }


def summarize(generated: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for cell, spec in CELL_SPECS.items():
        generated_cell = generated[cell]
        backbone_summary = read_json(resolve(spec["backbone_summary"]))
        anchors_summary = read_json(Path(generated_cell["anchors"]["summary_path"]))
        if generated_cell["moe_only_no_anchors"].get("generated", False):
            moe_only_summary = read_json(Path(generated_cell["moe_only_no_anchors"]["summary_path"]))
        else:
            moe_only_summary = read_json(resolve(generated_cell["moe_only_no_anchors"]["summary_path"]))
        if generated_cell["full"].get("generated", False):
            full_summary = read_json(Path(generated_cell["full"]["summary_path"]))
        else:
            full_summary = read_json(resolve(spec["full_summary"]))
        stages = {
            "a_backbone": metric_block(backbone_summary, "backbone"),
            "d_moe_only_no_anchors": metric_block(moe_only_summary, "full"),
            "b_anchors": metric_block(anchors_summary, "anchors"),
            "c_full": metric_block(full_summary, "full"),
        }
        cells[cell] = {
            "dataset": spec["dataset"],
            "horizon": int(spec["horizon"]),
            "source_paths": {
                "backbone_summary": str(resolve(spec["backbone_summary"])),
                "moe_only_no_anchors_config": generated_cell["moe_only_no_anchors"]["config_path"],
                "moe_only_no_anchors_summary": generated_cell["moe_only_no_anchors"]["summary_path"],
                "anchors_config": generated_cell["anchors"]["config_path"],
                "anchors_summary": generated_cell["anchors"]["summary_path"],
                "full_config": generated_cell["full"]["config_path"],
                "full_summary": generated_cell["full"]["summary_path"],
            },
            "stages": stages,
            "contributions": {
                "moe_only_a_to_d": contribution(stages["a_backbone"], stages["d_moe_only_no_anchors"]),
                "anchors_a_to_b": contribution(stages["a_backbone"], stages["b_anchors"]),
                "penalty_moe_b_to_c": contribution(stages["b_anchors"], stages["c_full"]),
                "anchors_added_to_moe_d_to_c": contribution(stages["d_moe_only_no_anchors"], stages["c_full"]),
                "total_a_to_c": contribution(stages["a_backbone"], stages["c_full"]),
            },
        }
    return {
        "meta": {
            "input_len": 96,
            "cells": list(CELL_SPECS.keys()),
            "stage_order": ["a_backbone", "d_moe_only_no_anchors", "b_anchors", "c_full"],
            "metric_note": (
                "For pred-residual stages, val_mse/val_mae use "
                "moe_residual_selection.val_scaled_* when present; raw_val_* keeps run_summary.val."
            ),
        },
        "cells": cells,
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# NEXT-8 ETT H96 Ablation",
        "",
        "| cell | stage | val MSE | val MAE | test MSE | test MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cell, payload in summary["cells"].items():
        for stage in summary["meta"]["stage_order"]:
            row = payload["stages"][stage]
            lines.append(
                f"| {cell} | {stage} | {row['val_mse']:.6f} | {row['val_mae']:.6f} | "
                f"{row['test_mse']:.6f} | {row['test_mae']:.6f} |"
            )
    lines.extend(["", "## Contributions", ""])
    lines.append("| cell | component | val MSE red. | val MAE red. | test MSE red. | test MAE red. |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for cell, payload in summary["cells"].items():
        for name, contrib in payload["contributions"].items():
            lines.append(
                f"| {cell} | {name} | {contrib['val_mse_reduction_pct']:.3f}% | "
                f"{contrib['val_mae_reduction_pct']:.3f}% | {contrib['test_mse_reduction_pct']:.3f}% | "
                f"{contrib['test_mae_reduction_pct']:.3f}% |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run NEXT-8 ETT H96 backbone/anchor/full ablation.")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--reuse-existing", action="store_true")
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()

    generated: dict[str, dict[str, Any]] = {}
    for cell, spec in CELL_SPECS.items():
        base_cfg = load_yaml(resolve(spec["full_config"]))
        generate_full = str(spec.get("generate_full", "")).lower() == "true"
        reuse_full_as_moe_only = str(spec.get("reuse_full_as_moe_only", "")).lower() == "true"
        anchors_cfg_path, anchors_out_dir, _ = build_stage_config(cell, "anchors", base_cfg)
        moe_only_cfg_path, moe_only_out_dir, _ = build_stage_config(cell, "moe_only_no_anchors", base_cfg)
        generated[cell] = {
            "moe_only_no_anchors": {
                "config_path": str(moe_only_cfg_path),
                "out_dir": str(moe_only_out_dir),
                "summary_path": str(moe_only_out_dir / "run_summary.json"),
                "generated": True,
            },
            "anchors": {
                "config_path": str(anchors_cfg_path),
                "out_dir": str(anchors_out_dir),
                "summary_path": str(anchors_out_dir / "run_summary.json"),
            },
            "full": {
                "config_path": str(resolve(spec["full_config"])),
                "summary_path": str(resolve(spec["full_summary"])),
                "generated": False,
            },
        }
        if reuse_full_as_moe_only:
            generated[cell]["moe_only_no_anchors"] = {
                "config_path": str(resolve(spec["full_config"])),
                "summary_path": str(resolve(spec["full_summary"])),
                "generated": False,
            }
        if generate_full:
            full_cfg_path, full_out_dir, _ = build_stage_config(cell, "full", base_cfg)
            generated[cell]["full"] = {
                "config_path": str(full_cfg_path),
                "out_dir": str(full_out_dir),
                "summary_path": str(full_out_dir / "run_summary.json"),
                "generated": True,
            }

    write_json(OUT_ROOT / "next8_manifest.json", build_manifest(generated))
    if args.prepare_only:
        print(f"prepared {OUT_ROOT / 'next8_manifest.json'}")
        return

    run_results: dict[str, Any] = {}
    for cell, payload in generated.items():
        run_results.setdefault(cell, {})
        moe_only = payload["moe_only_no_anchors"]
        if moe_only.get("generated", False):
            print(f"=== {cell} moe_only_no_anchors ===", flush=True)
            run_results[cell]["moe_only_no_anchors"] = run_config(
                Path(moe_only["config_path"]),
                Path(moe_only["out_dir"]),
                args.python,
                reuse_existing=bool(args.reuse_existing),
            )
            print(json.dumps(run_results[cell]["moe_only_no_anchors"], ensure_ascii=False), flush=True)
        anchors = payload["anchors"]
        print(f"=== {cell} anchors ===", flush=True)
        run_results[cell]["anchors"] = run_config(
            Path(anchors["config_path"]),
            Path(anchors["out_dir"]),
            args.python,
            reuse_existing=bool(args.reuse_existing),
        )
        print(json.dumps(run_results[cell]["anchors"], ensure_ascii=False), flush=True)
        full = payload["full"]
        if full.get("generated", False):
            print(f"=== {cell} full ===", flush=True)
            run_results[cell]["full"] = run_config(
                Path(full["config_path"]),
                Path(full["out_dir"]),
                args.python,
                reuse_existing=bool(args.reuse_existing),
            )
            print(json.dumps(run_results[cell]["full"], ensure_ascii=False), flush=True)

    write_json(OUT_ROOT / "next8_run_results.json", {"runs": run_results})
    summary = summarize(generated)
    write_json(OUT_ROOT / "next8_summary.json", summary)
    write_markdown(summary, OUT_ROOT / "next8_summary.md")
    print(f"summary {OUT_ROOT / 'next8_summary.json'}")


if __name__ == "__main__":
    main()
