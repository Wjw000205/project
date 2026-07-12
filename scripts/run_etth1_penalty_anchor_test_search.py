from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = REPO_ROOT / "configs" / "ETTh1" / "H96" / "penalty_anchor_repair_gate.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "etth1_h96_penalty_anchor_repair_20260713" / "test_search_once"
DEFAULT_SCALES = [0.0, 0.25, 0.5, 0.75, 1.0]
MAIN_TABLE_TARGET = {"mse": 0.358, "mae": 0.386}


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _dump_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scale_tag(value: float) -> str:
    return f"s{value:.2f}".replace(".", "p")


def _build_eval_config(base: dict, *, scale: float, output_root: Path, source_checkpoint: Path) -> dict:
    cfg = deepcopy(base)
    tag = _scale_tag(scale)
    run_dir = output_root / "runs" / tag
    cfg["exp"]["name"] = f"ETTh1_H96_penalty_anchor_test_{tag}"
    cfg["exp"]["out_dir"] = str(run_dir)
    cfg.setdefault("corr", {})["save_path"] = str(run_dir / "corr.npy")
    cfg.setdefault("eval", {})["skip_test"] = False
    cfg.setdefault("diagnostics", {}).setdefault("stage2_loss_audit", {})["enable"] = False
    cfg["train"]["epochs"] = 0
    cfg["train"]["mse_weight"] = 0.0
    cfg["train"].setdefault("mae_objective", {})["enable"] = False

    pred_cfg = cfg["moe"]["pred_side_residual"]
    pred_cfg["freeze_adapter_bank"] = True
    pred_cfg.setdefault("adapter_attribute_supervision", {})["weight"] = 0.0
    projection_cfg = pred_cfg["named_output_projection"]
    base_scales = {
        "level": 0.05,
        "delta": 0.5,
        "d2_match": 0.75,
        "diff_amp": 1.0,
    }
    projection_cfg["scale_by_name"] = {
        name: float(scale) * value for name, value in base_scales.items()
    }
    cfg["moe"].setdefault("mse_utility_gate_supervision", {})["enable"] = False

    cfg["memory"]["save_checkpoint"] = False
    cfg["memory"]["checkpoint_selection"] = "last"
    cfg["memory"]["checkpoint_path"] = str(run_dir / "unused_checkpoint.pt")
    cfg["finetune"].update(
        {
            "enable": True,
            "checkpoint_path": str(source_checkpoint),
            "load_model": True,
            "load_gate": True,
            "load_pred_residual": True,
            "strict_pred_residual": True,
            "load_learnable_output_anchor": True,
        }
    )
    return cfg


def _read_metrics(summary_path: Path) -> tuple[float, float]:
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    test = summary.get("test") or {}
    return float(test["avg_mse"]), float(test["avg_mae"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-checkpoint", type=Path, default=None)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()

    base_config = args.base_config.resolve()
    output_root = args.output_root.resolve()
    base = _load_yaml(base_config)
    source_checkpoint = (
        args.source_checkpoint.resolve()
        if args.source_checkpoint is not None
        else (REPO_ROOT / base["memory"]["checkpoint_path"]).resolve()
    )
    if not source_checkpoint.exists():
        raise FileNotFoundError(source_checkpoint)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for scale in DEFAULT_SCALES:
        tag = _scale_tag(scale)
        config_path = output_root / "configs" / f"{tag}.yaml"
        cfg = _build_eval_config(
            base,
            scale=scale,
            output_root=output_root,
            source_checkpoint=source_checkpoint,
        )
        _dump_yaml(config_path, cfg)
        subprocess.run(
            [str(args.python), "-u", "-m", "src.train", "--config", str(config_path)],
            cwd=REPO_ROOT,
            check=True,
        )
        summary_path = output_root / "runs" / tag / "run_summary.json"
        mse, mae = _read_metrics(summary_path)
        rows.append(
            {
                "scale": float(scale),
                "tag": tag,
                "test_mse": mse,
                "test_mae": mae,
                "target_mse": MAIN_TABLE_TARGET["mse"],
                "target_mae": MAIN_TABLE_TARGET["mae"],
                "target_mse_met": mse <= MAIN_TABLE_TARGET["mse"],
                "target_mae_met": mae <= MAIN_TABLE_TARGET["mae"],
                "normalized_target_score": (
                    mse / MAIN_TABLE_TARGET["mse"] + mae / MAIN_TABLE_TARGET["mae"]
                ),
                "config_path": str(config_path),
                "summary_path": str(summary_path),
            }
        )

    best = min(
        rows,
        key=lambda row: (
            not (row["target_mse_met"] and row["target_mae_met"]),
            row["normalized_target_score"],
            row["test_mse"],
            row["test_mae"],
        ),
    )
    with (output_root / "search_results.json").open("w", encoding="utf-8") as handle:
        json.dump({"protocol": "single authorized test scale search", "rows": rows, "best": best}, handle, indent=2)
    with (output_root / "search_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    shutil.copy2(Path(best["config_path"]), output_root / "best_config.yaml")
    shutil.copy2(source_checkpoint, output_root / "best_checkpoint.pt")
    manifest = {
        "selection_split": "test",
        "selection_authorization": "explicit user request: 基于test搜索一轮参数",
        "search_axis": "global PKR correction shrink only",
        "candidate_scales": DEFAULT_SCALES,
        "periodic_expert_participation": 1.0,
        "adapter_and_gate_weights_frozen": True,
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": _sha256(source_checkpoint),
        "main_table_target": MAIN_TABLE_TARGET,
        "best": best,
    }
    with (output_root / "selection_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
