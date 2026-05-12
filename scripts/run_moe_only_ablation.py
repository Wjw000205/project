import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.yaml_io import load_yaml


DEFAULT_CONFIGS = [
    "configs/ETTm1.yaml",
    "configs/ETTm2.yaml",
    "configs/ETTh1.yaml",
    "configs/ETTh2.yaml",
]


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def safe_name(path: Path) -> str:
    return path.stem.replace(" ", "_")


def ensure_run_paths(cfg: Dict[str, Any], out_dir: Path) -> None:
    cfg.setdefault("exp", {})
    cfg["exp"]["out_dir"] = str(out_dir)
    cfg.setdefault("corr", {})
    cfg["corr"]["save_path"] = str(out_dir / "corr.npy")
    cfg.setdefault("plot", {})
    cfg["plot"]["enable"] = False
    cfg.setdefault("portrait", {})
    cfg["portrait"]["enable"] = False
    cfg["portrait"]["out_dir"] = str(out_dir / "cluster_portraits")
    cfg.setdefault("calibration", {})
    cfg["calibration"]["enable"] = False
    cfg.setdefault("knn_hybrid", {})
    cfg["knn_hybrid"]["enable"] = False
    cfg.setdefault("memory", {})
    cfg["memory"]["path"] = str(out_dir / "cluster_memory.pt")
    cfg["memory"]["checkpoint_path"] = str(out_dir / "best_checkpoint.pt")


def apply_residual_defaults(
    cfg: Dict[str, Any],
    enabled: bool,
    residual_selection_policy: str = None,
    residual_scale_steps: int = None,
) -> None:
    moe_cfg = cfg.setdefault("moe", {})
    moe_cfg["enable"] = bool(enabled)
    pred_cfg = moe_cfg.setdefault("pred_side_residual", {})
    pred_cfg["enable"] = bool(enabled)
    pred_cfg.setdefault("corrector_hidden", 32)
    pred_cfg.setdefault("init_alpha", -3.0)
    pred_cfg.setdefault("alpha_scale", 0.5)
    pred_cfg.setdefault("specialization_weight", 0.1)
    pred_cfg.setdefault("norm_weight", 1.0e-4)
    pred_cfg.setdefault("use_y_base_input", True)
    pred_cfg.setdefault("intervention_enable", False)
    pred_cfg.setdefault("intervention_init", -2.0)
    pred_cfg.setdefault("intervention_weight", 1.0e-3)
    pred_cfg.setdefault("detach_routed_penalty_pred", False)
    if residual_selection_policy is not None:
        pred_cfg["selection_policy"] = residual_selection_policy if enabled else "none"
    else:
        pred_cfg.setdefault("selection_policy", "val_mse_gate" if enabled else "none")
    pred_cfg.setdefault("selection_min_abs_improvement", 0.0)
    pred_cfg.setdefault("selection_min_rel_improvement", 0.0)
    gate_cfg = pred_cfg.setdefault("gate_calibrator", {})
    gate_cfg.setdefault("loss", "mse")
    gate_cfg.setdefault("selection_metric", "mse")
    gate_cfg.setdefault("epochs", 30)
    gate_cfg.setdefault("train_fraction", 0.7)
    gate_cfg.setdefault("hidden_dim", 32)
    gate_cfg.setdefault("batch_size", 256)
    gate_cfg.setdefault("max_scale", 1.0)
    gate_cfg.setdefault("init_scale", 0.8)
    gate_cfg.setdefault("scale_reg", 1.0e-4)
    if residual_scale_steps is not None:
        pred_cfg["selection_scale_steps"] = int(residual_scale_steps)
    if not enabled:
        dyn_cfg = moe_cfg.setdefault("dynamic_lambda", {})
        dyn_cfg["enable"] = False
        learn_cfg = moe_cfg.setdefault("learnable_lambda", {})
        learn_cfg["enable"] = False


def write_yaml(path: Path, cfg: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)


def run_train(config_path: Path, reuse_existing: bool) -> None:
    cfg = load_yaml(str(config_path))
    summary_path = Path(cfg["exp"]["out_dir"]) / "run_summary.json"
    if summary_path.exists() and reuse_existing:
        print(f"[reuse] {summary_path}")
        return
    cmd = [sys.executable, "-m", "src.train", "--config", str(config_path)]
    print("[run] " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def read_metrics(run_dir: Path) -> Dict[str, Any]:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing run summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    test = summary.get("test", {}) or {}
    selected = summary.get("selected", {}) or {}
    return {
        "test_mse": float(test.get("avg_mse", float("nan"))),
        "test_mae": float(test.get("avg_mae", float("nan"))),
        "selected_mse": float(selected.get("avg_mse", test.get("avg_mse", float("nan")))),
        "selected_mae": float(selected.get("avg_mae", test.get("avg_mae", float("nan")))),
        "selected_variant": str(selected.get("variant", "base")),
        "moe_residual": summary.get("moe_residual", {}),
        "best_epoch": ",".join(str(v) for v in summary.get("best_epoch", [])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    ap.add_argument("--out-root", default="outputs/moe_only_ablation")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--residual-selection-policy", default=None)
    ap.add_argument("--residual-scale-steps", type=int, default=None)
    ap.add_argument("--skip-run", action="store_true")
    ap.add_argument("--reuse-existing", action="store_true")
    args = ap.parse_args()

    out_root = resolve_path(args.out_root)
    cfg_root = out_root / "configs"
    runs_root = out_root / "runs"
    out_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for cfg_text in args.configs:
        base_path = resolve_path(cfg_text)
        base_cfg = load_yaml(str(base_path))
        name = safe_name(base_path)
        run_rows: Dict[str, Dict[str, Any]] = {}

        for label, enabled in [("moe_on", True), ("moe_off", False)]:
            cfg = copy.deepcopy(base_cfg)
            if args.epochs is not None:
                cfg.setdefault("train", {})["epochs"] = int(args.epochs)
            if args.device is not None:
                cfg.setdefault("exp", {})["device"] = str(args.device)
            run_dir = runs_root / name / label
            ensure_run_paths(cfg, run_dir)
            cfg.setdefault("exp", {})["name"] = f"{name}_{label}"
            apply_residual_defaults(
                cfg,
                enabled=enabled,
                residual_selection_policy=args.residual_selection_policy,
                residual_scale_steps=args.residual_scale_steps,
            )
            cfg_path = cfg_root / name / f"{label}.yaml"
            write_yaml(cfg_path, cfg)
            if not args.skip_run:
                run_train(cfg_path, reuse_existing=bool(args.reuse_existing))
            metrics = read_metrics(run_dir) if (run_dir / "run_summary.json").exists() else {}
            run_rows[label] = {"config_path": str(cfg_path), "run_dir": str(run_dir), **metrics}

        on_mse = float(run_rows["moe_on"].get("test_mse", float("nan")))
        off_mse = float(run_rows["moe_off"].get("test_mse", float("nan")))
        gain = off_mse - on_mse
        gain_pct = 100.0 * gain / max(abs(off_mse), 1.0e-12)
        rows.append(
            {
                "dataset": name,
                "config": str(base_path),
                "moe_on_run": run_rows["moe_on"].get("run_dir", ""),
                "moe_off_run": run_rows["moe_off"].get("run_dir", ""),
                "moe_on_test_mse": on_mse,
                "moe_off_test_mse": off_mse,
                "mse_gain": gain,
                "mse_gain_pct": gain_pct,
                "moe_on_test_mae": run_rows["moe_on"].get("test_mae", float("nan")),
                "moe_off_test_mae": run_rows["moe_off"].get("test_mae", float("nan")),
                "moe_on_best_epoch": run_rows["moe_on"].get("best_epoch", ""),
                "moe_off_best_epoch": run_rows["moe_off"].get("best_epoch", ""),
                "moe_on_config": run_rows["moe_on"].get("config_path", ""),
                "moe_off_config": run_rows["moe_off"].get("config_path", ""),
            }
        )

    results = pd.DataFrame(rows)
    results_path = out_root / "results.csv"
    summary_path = out_root / "summary.json"
    results.to_csv(results_path, index=False)
    best = None
    if not results.empty:
        best = results.sort_values(["mse_gain_pct", "mse_gain"], ascending=False).iloc[0].to_dict()
    summary = {
        "out_root": str(out_root),
        "configs": [str(resolve_path(c)) for c in args.configs],
        "best_by_gain_pct": best,
        "results": rows,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved results to: {results_path}")
    print(f"Saved summary to: {summary_path}")
    if not results.empty:
        print(results.to_string(index=False))


if __name__ == "__main__":
    main()
