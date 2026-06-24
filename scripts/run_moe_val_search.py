import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.yaml_io import load_yaml


DEFAULT_BASE_CONFIG = "outputs/moe_tuning_noleak_configs6/ETTm1_level_only.yaml"


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)
    return dst


def write_yaml(path: Path, cfg: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)


def variant_overrides() -> Dict[str, Dict[str, Any]]:
    augmented = {
        "moe": {
            "pred_side_residual": {
                "feature_mode": "safe_augmented",
                "residual_clip": 6.0,
            },
        },
    }
    return {
        "legacy_level_gate": {},
        "legacy_level_gate_guarded": {
            "moe": {
                "pred_side_residual": {
                    "selection_policy": "val_mse_candidate_channel",
                },
            },
        },
        "legacy_level_gate_signed": {
            "moe": {
                "pred_side_residual": {
                },
            },
        },
        "legacy_level_gate_signed_guarded": {
            "moe": {
                "pred_side_residual": {
                    "selection_policy": "val_mse_candidate_channel",
                },
            },
        },
        "level_amp_gate": {
            "moe": {
                "lambda_init": {
                    "level": 0.1,
                    "amp_under": 0.1,
                },
                "lambda_min": {
                    "level": 0.0,
                    "amp_under": 0.0,
                },
                "lambda_schedule": {
                    "level": "none",
                    "amp_under": "none",
                },
            },
            "penalties": {
                "enabled": ["level", "amp_under"],
            },
            "train": {
                "epochs": 20,
            },
        },
        "level_amp_gate_guarded": {
            "moe": {
                "lambda_init": {
                    "level": 0.1,
                    "amp_under": 0.1,
                },
                "lambda_min": {
                    "level": 0.0,
                    "amp_under": 0.0,
                },
                "lambda_schedule": {
                    "level": "none",
                    "amp_under": "none",
                },
                "pred_side_residual": {
                    "selection_policy": "val_mse_candidate_channel",
                },
            },
            "penalties": {
                "enabled": ["level", "amp_under"],
            },
            "train": {
                "epochs": 20,
            },
        },
        "level_amp_scale": {
            "moe": {
                "lambda_init": {
                    "level": 0.1,
                    "amp_under": 0.1,
                },
                "lambda_min": {
                    "level": 0.0,
                    "amp_under": 0.0,
                },
                "lambda_schedule": {
                    "level": "none",
                    "amp_under": "none",
                },
                "pred_side_residual": {
                    "selection_policy": "val_mse_scale",
                    "selection_scale_min": 0.0,
                    "selection_scale_max": 1.5,
                    "selection_scale_steps": 61,
                },
            },
            "penalties": {
                "enabled": ["level", "amp_under"],
            },
            "train": {
                "epochs": 20,
            },
        },
        "level_amp_gate_amp_std": {
            "moe": {
                "lambda_init": {
                    "level": 0.1,
                    "amp_under": 0.1,
                },
                "lambda_min": {
                    "level": 0.0,
                    "amp_under": 0.0,
                },
                "lambda_schedule": {
                    "level": "none",
                    "amp_under": "none",
                },
                "pred_side_residual": {
                },
            },
            "penalties": {
                "enabled": ["level", "amp_under"],
            },
            "train": {
                "epochs": 20,
            },
        },
        "level_amp_gate_amp_std_loose": {
            "moe": {
                "lambda_init": {
                    "level": 0.1,
                    "amp_under": 0.1,
                },
                "lambda_min": {
                    "level": 0.0,
                    "amp_under": 0.0,
                },
                "lambda_schedule": {
                    "level": "none",
                    "amp_under": "none",
                },
                "pred_side_residual": {
                },
            },
            "penalties": {
                "enabled": ["level", "amp_under"],
            },
            "train": {
                "epochs": 20,
            },
        },
        "level_amp_gate_amp_std_frac85": {
            "moe": {
                "lambda_init": {
                    "level": 0.1,
                    "amp_under": 0.1,
                },
                "lambda_min": {
                    "level": 0.0,
                    "amp_under": 0.0,
                },
                "lambda_schedule": {
                    "level": "none",
                    "amp_under": "none",
                },
                "pred_side_residual": {
                },
            },
            "penalties": {
                "enabled": ["level", "amp_under"],
            },
            "train": {
                "epochs": 20,
            },
        },
        "aug_level_amp_gate_amp_std": {
            "moe": {
                "lambda_init": {
                    "level": 0.1,
                    "amp_under": 0.1,
                },
                "lambda_min": {
                    "level": 0.0,
                    "amp_under": 0.0,
                },
                "lambda_schedule": {
                    "level": "none",
                    "amp_under": "none",
                },
                "pred_side_residual": {
                    "feature_mode": "safe_augmented",
                    "residual_clip": 6.0,
                },
            },
            "penalties": {
                "enabled": ["level", "amp_under"],
            },
            "train": {
                "epochs": 20,
            },
        },
        "level_amp_raw": {
            "moe": {
                "lambda_init": {
                    "level": 0.1,
                    "amp_under": 0.1,
                },
                "lambda_min": {
                    "level": 0.0,
                    "amp_under": 0.0,
                },
                "lambda_schedule": {
                    "level": "none",
                    "amp_under": "none",
                },
                "pred_side_residual": {
                    "selection_policy": "none",
                },
            },
            "penalties": {
                "enabled": ["level", "amp_under"],
            },
            "train": {
                "epochs": 20,
            },
        },
        "full_penalty_gate": {
            "moe": {
                "lambda_init": {
                    "amp_under": 0.1,
                    "delta": 0.1,
                    "jitter": 0.1,
                    "smooth": 0.1,
                },
                "lambda_min": {
                    "amp_under": 0.0,
                    "delta": 0.0,
                    "jitter": 0.0,
                    "smooth": 0.0,
                },
                "lambda_schedule": {
                    "amp_under": "none",
                    "delta": "none",
                    "jitter": "none",
                    "smooth": "none",
                },
                "router_mode": "penalty_context",
                "router_penalty_context_weight": 1.1,
            },
            "penalties": {
                "enabled": ["amp_under", "delta", "jitter", "smooth"],
            },
            "train": {
                "selection_metric": "val_mae",
            },
        },
        "aug_level_gate_std": augmented,
        "aug_level_gate_amp": deep_update(
            copy.deepcopy(augmented),
            {
                "moe": {
                    "pred_side_residual": {
                    },
                },
            },
        ),
        "aug_level_gate_reg": deep_update(
            copy.deepcopy(augmented),
            {
                "moe": {
                    "pred_side_residual": {
                        "specialization_weight": 0.05,
                        "norm_weight": 3.0e-4,
                    },
                },
            },
        ),
        "aug_level_range_gate": deep_update(
            copy.deepcopy(augmented),
            {
                "moe": {
                    "topk": 1,
                    "lambda_init": {
                        "level": 0.1,
                        "range": 0.03,
                    },
                },
                "penalties": {
                    "enabled": ["level", "range"],
                },
            },
        ),
        "aug_level_range_gate_noclip": deep_update(
            copy.deepcopy(augmented),
            {
                "moe": {
                    "topk": 1,
                    "lambda_init": {
                        "level": 0.1,
                        "range": 0.03,
                    },
                    "pred_side_residual": {
                        "residual_clip": 0.0,
                    },
                },
                "penalties": {
                    "enabled": ["level", "range"],
                },
            },
        ),
        "aug_level_range_guarded_noclip": deep_update(
            copy.deepcopy(augmented),
            {
                "moe": {
                    "topk": 1,
                    "lambda_init": {
                        "level": 0.1,
                        "range": 0.03,
                    },
                    "pred_side_residual": {
                        "residual_clip": 0.0,
                        "selection_policy": "val_mse_candidate_channel",
                    },
                },
                "penalties": {
                    "enabled": ["level", "range"],
                },
            },
        ),
        "aug_level_range_signed_noclip": deep_update(
            copy.deepcopy(augmented),
            {
                "moe": {
                    "topk": 1,
                    "lambda_init": {
                        "level": 0.1,
                        "range": 0.03,
                    },
                    "pred_side_residual": {
                        "residual_clip": 0.0,
                    },
                },
                "penalties": {
                    "enabled": ["level", "range"],
                },
            },
        ),
    }


def ensure_no_leak_paths(cfg: Dict[str, Any], out_dir: Path, skip_test: bool) -> None:
    cfg.setdefault("exp", {})
    cfg["exp"]["out_dir"] = str(out_dir)
    cfg.setdefault("eval", {})
    cfg["eval"]["skip_test"] = bool(skip_test)
    cfg.setdefault("normalize", {})
    cfg["normalize"]["train_only"] = True
    cfg.setdefault("cluster", {})
    cfg["cluster"]["train_only"] = True
    cfg.setdefault("corr", {})
    cfg["corr"]["save_path"] = str(out_dir / "corr.npy")
    cfg.setdefault("plot", {})
    cfg["plot"]["enable"] = False
    cfg.setdefault("portrait", {})
    cfg["portrait"]["enable"] = False
    cfg["portrait"]["out_dir"] = str(out_dir / "cluster_portraits")
    cfg.setdefault("memory", {})
    cfg["memory"]["enable"] = False
    cfg["memory"]["save_checkpoint"] = False
    cfg["memory"]["path"] = str(out_dir / "cluster_memory.pt")
    cfg["memory"]["checkpoint_path"] = str(out_dir / "best_checkpoint.pt")


def apply_moe_state(cfg: Dict[str, Any], enabled: bool) -> None:
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
    pred_cfg.setdefault("feature_mode", "legacy")
    pred_cfg.setdefault("residual_clip", 0.0)
    pred_cfg.setdefault("intervention_enable", False)
    pred_cfg.setdefault("intervention_init", -2.0)
    pred_cfg.setdefault("intervention_weight", 1.0e-3)
    pred_cfg.setdefault("detach_routed_penalty_pred", False)
    if enabled:
        pred_cfg.setdefault("selection_policy", "val_mse_candidate_channel")
    else:
        pred_cfg["selection_policy"] = "none"
    gate_cfg.setdefault("loss", "mse")
    gate_cfg.setdefault("selection_metric", "mse")
    gate_cfg.setdefault("epochs", 30)
    gate_cfg.setdefault("train_fraction", 0.7)
    gate_cfg.setdefault("hidden_dim", 32)
    gate_cfg.setdefault("batch_size", 256)
    gate_cfg.setdefault("max_scale", 1.0)
    gate_cfg.setdefault("init_scale", 0.8)
    gate_cfg.setdefault("scale_reg", 1.0e-4)
    if not enabled:
        moe_cfg.setdefault("dynamic_lambda", {})["enable"] = False
        moe_cfg.setdefault("learnable_lambda", {})["enable"] = False


def run_train(config_path: Path, reuse_existing: bool) -> None:
    cfg = load_yaml(str(config_path))
    summary_path = Path(cfg["exp"]["out_dir"]) / "run_summary.json"
    if reuse_existing and summary_path.exists():
        print(f"[reuse] {summary_path}")
        return
    cmd = [sys.executable, "-m", "src.train", "--config", str(config_path)]
    print("[run] " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def read_val_metrics(run_dir: Path, require_no_test: bool) -> Dict[str, Any]:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        return {}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if require_no_test and summary.get("test") is not None:
        raise RuntimeError(f"Leak guard failed: search run wrote test metrics: {summary_path}")
    eval_cfg = summary.get("eval", {}) or {}
    if require_no_test and not bool(eval_cfg.get("skip_test", False)):
        raise RuntimeError(f"Leak guard failed: eval.skip_test is not recorded true: {summary_path}")
    val = summary.get("val", {}) or {}
    residual = summary.get("moe_residual_selection", {}) or {}
    raw_val_mse = float(val.get("avg_mse", float("nan")))
    raw_val_mae = float(val.get("avg_mae", float("nan")))
    selected_val_mse = float(residual.get("val_scaled_avg_mse", raw_val_mse))
    selected_val_mae = float(residual.get("val_scaled_avg_mae", raw_val_mae))
    return {
        "val_mse": selected_val_mse,
        "val_mae": selected_val_mae,
        "raw_val_mse": raw_val_mse,
        "raw_val_mae": raw_val_mae,
        "best_epoch": ",".join(str(v) for v in summary.get("best_epoch", [])),
        "residual_policy": str(residual.get("policy", "")),
        "residual_val_mse": selected_val_mse,
        "gate_holdout_mse": float(gate_summary.get("holdout_mse", float("nan"))),
        "feature_mode": str((summary.get("moe_residual", {}) or {}).get("feature_mode", "")),
        "test_is_null": summary.get("test") is None,
        "normalize_train_only": bool((summary.get("windowing", {}) or {}).get("normalize_train_only", False)),
    }


def build_run_config(
    base_cfg: Dict[str, Any],
    variant_name: str,
    overrides: Dict[str, Any],
    enabled: bool,
    out_dir: Path,
    epochs: Optional[int],
    device: Optional[str],
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    deep_update(cfg, overrides)
    ensure_no_leak_paths(cfg, out_dir=out_dir, skip_test=True)
    apply_moe_state(cfg, enabled=enabled)
    cfg.setdefault("exp", {})["name"] = f"{variant_name}_{'moe_on' if enabled else 'moe_off'}"
    if epochs is not None:
        cfg.setdefault("train", {})["epochs"] = int(epochs)
    if device is not None:
        cfg.setdefault("exp", {})["device"] = str(device)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    ap.add_argument("--out-root", default="outputs/moe_val_search_ETTm1")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--skip-run", action="store_true")
    ap.add_argument("--reuse-existing", action="store_true")
    args = ap.parse_args()

    base_path = resolve_path(args.base_config)
    out_root = resolve_path(args.out_root)
    cfg_root = out_root / "configs"
    runs_root = out_root / "runs"
    out_root.mkdir(parents=True, exist_ok=True)
    base_cfg = load_yaml(str(base_path))
    all_variants = variant_overrides()
    variant_names = args.variants if args.variants else list(all_variants.keys())
    unknown = [name for name in variant_names if name not in all_variants]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Supported: {sorted(all_variants)}")

    rows: List[Dict[str, Any]] = []
    for variant_name in variant_names:
        overrides = all_variants[variant_name]
        run_rows: Dict[str, Dict[str, Any]] = {}
        for label, enabled in [("moe_on", True), ("moe_off", False)]:
            run_dir = runs_root / variant_name / label
            cfg = build_run_config(
                base_cfg=base_cfg,
                variant_name=variant_name,
                overrides=overrides,
                enabled=enabled,
                out_dir=run_dir,
                epochs=args.epochs,
                device=args.device,
            )
            cfg_path = cfg_root / variant_name / f"{label}.yaml"
            write_yaml(cfg_path, cfg)
            if not args.skip_run:
                run_train(cfg_path, reuse_existing=bool(args.reuse_existing))
            metrics = read_val_metrics(run_dir, require_no_test=True)
            run_rows[label] = {"config_path": str(cfg_path), "run_dir": str(run_dir), **metrics}

        on_mse = float(run_rows["moe_on"].get("val_mse", float("nan")))
        off_mse = float(run_rows["moe_off"].get("val_mse", float("nan")))
        gain = off_mse - on_mse
        gain_pct = 100.0 * gain / max(abs(off_mse), 1.0e-12)
        rows.append(
            {
                "variant": variant_name,
                "moe_on_val_mse": on_mse,
                "moe_off_val_mse": off_mse,
                "val_mse_gain": gain,
                "val_mse_gain_pct": gain_pct,
                "moe_on_val_mae": run_rows["moe_on"].get("val_mae", float("nan")),
                "moe_off_val_mae": run_rows["moe_off"].get("val_mae", float("nan")),
                "moe_on_raw_val_mse": run_rows["moe_on"].get("raw_val_mse", float("nan")),
                "moe_off_raw_val_mse": run_rows["moe_off"].get("raw_val_mse", float("nan")),
                "moe_on_best_epoch": run_rows["moe_on"].get("best_epoch", ""),
                "moe_off_best_epoch": run_rows["moe_off"].get("best_epoch", ""),
                "feature_mode": run_rows["moe_on"].get("feature_mode", ""),
                "gate_holdout_mse": run_rows["moe_on"].get("gate_holdout_mse", float("nan")),
                "test_is_null": bool(run_rows["moe_on"].get("test_is_null", False))
                and bool(run_rows["moe_off"].get("test_is_null", False)),
                "normalize_train_only": bool(run_rows["moe_on"].get("normalize_train_only", False))
                and bool(run_rows["moe_off"].get("normalize_train_only", False)),
                "moe_on_config": run_rows["moe_on"].get("config_path", ""),
                "moe_off_config": run_rows["moe_off"].get("config_path", ""),
            }
        )

    results = pd.DataFrame(rows)
    if not results.empty:
        results = results.sort_values(["val_mse_gain_pct", "val_mse_gain"], ascending=False).reset_index(drop=True)
    results_path = out_root / "validation_results.csv"
    summary_path = out_root / "validation_summary.json"
    results.to_csv(results_path, index=False)
    best = None if results.empty else results.iloc[0].to_dict()
    summary = {
        "base_config": str(base_path),
        "out_root": str(out_root),
        "skip_test": True,
        "selection_metric": "val_mse_gain_pct",
        "best_by_validation_gain": best,
        "results": rows,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved validation results to: {results_path}")
    print(f"Saved validation summary to: {summary_path}")
    if not results.empty:
        print(results.to_string(index=False))


if __name__ == "__main__":
    main()
