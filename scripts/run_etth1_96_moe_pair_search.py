import argparse
import copy
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

LAMBDA_DEFAULTS = {
    "amp": 0.1,
    "level": 0.1,
    "amp_under": 0.1,
    "range": 0.03,
    "trend": 0.05,
    "delta": 0.1,
    "d2_match": 0.1,
    "diff_amp": 0.1,
    "direction": 0.1,
    "jump": 0.1,
    "corr": 0.1,
    "jitter": 0.1,
    "smooth": 0.1,
}


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def write_yaml(path: Path, cfg: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)


def deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)
    return dst


def short_slug(text: str, max_len: int = 96) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
    while "__" in safe:
        safe = safe.replace("__", "_")
    safe = safe.strip("_") or "candidate"
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    keep = max(8, max_len - len(digest) - 1)
    return f"{safe[:keep].rstrip('_')}_{digest}"


def tag(value: float) -> str:
    text = f"{float(value):.0e}" if 0 < abs(float(value)) < 0.001 else f"{float(value):g}"
    return text.replace("-", "m").replace("+", "").replace(".", "p")


def lambda_map(penalties: Sequence[str], scale: float) -> Dict[str, float]:
    return {name: float(LAMBDA_DEFAULTS.get(name, 0.1) * scale) for name in penalties}


def zero_map(penalties: Sequence[str]) -> Dict[str, float]:
    return {name: 0.0 for name in penalties}


def none_schedule(penalties: Sequence[str]) -> Dict[str, str]:
    return {name: "none" for name in penalties}


def fixed_h96_protocol(cfg: Dict[str, Any], out_dir: Path, device: Optional[str]) -> None:
    cfg.setdefault("exp", {})
    cfg["exp"]["name"] = out_dir.name
    cfg["exp"]["out_dir"] = str(out_dir)
    if device:
        cfg["exp"]["device"] = device

    cfg.setdefault("data", {})
    cfg["data"]["csv_path"] = "data/ETTh1.csv"
    cfg["data"]["date_col"] = int(cfg["data"].get("date_col", 0))
    cfg["data"]["max_rows"] = int(cfg["data"].get("max_rows", 14400) or 14400)
    cfg["data"]["train_ratio"] = 0.6
    cfg["data"]["val_ratio"] = 0.2
    cfg["data"]["test_ratio"] = 0.2

    cfg.setdefault("window", {})
    cfg["window"]["input_len"] = 336
    cfg["window"]["pred_len"] = 96
    cfg["window"]["past_context"] = True

    cfg.setdefault("normalize", {})
    cfg["normalize"]["global_zscore"] = True
    cfg["normalize"]["train_only"] = True

    cfg.setdefault("corr", {})
    cfg["corr"]["compute"] = bool(cfg["corr"].get("compute", True))
    cfg["corr"]["save_path"] = str(out_dir / "corr.npy")

    cfg.setdefault("cluster", {})
    cfg["cluster"]["train_only"] = True

    cfg.setdefault("model", {})
    cfg["model"]["predictor"] = "dlinear"
    cfg["model"]["dlinear_kernel_size"] = int(cfg["model"].get("dlinear_kernel_size", 25) or 25)

    cfg.setdefault("plot", {})
    cfg["plot"]["enable"] = False
    cfg.setdefault("portrait", {})
    cfg["portrait"]["enable"] = False
    cfg["portrait"]["out_dir"] = str(out_dir / "cluster_portraits")

    cfg.setdefault("knn_hybrid", {})
    cfg["knn_hybrid"]["enable"] = False
    cfg["knn_hybrid"]["path"] = str(out_dir / "knn_shape_bank.pt")

    cfg.setdefault("eval", {})
    cfg["eval"]["skip_test"] = False
    cfg.setdefault("calibration", {})
    cfg["calibration"]["enable"] = False
    cfg.setdefault("memory", {})
    cfg["memory"]["enable"] = False
    cfg["memory"]["save_checkpoint"] = False
    cfg["memory"]["path"] = str(out_dir / "cluster_memory.pt")
    cfg["memory"]["checkpoint_path"] = str(out_dir / "best_checkpoint.pt")

    cfg.setdefault("train", {})
    cfg["train"]["selection_metric"] = "val_mae"
    cfg["train"]["mse_weight"] = 0.9
    cfg["train"]["penalty_warmup_epochs"] = 15
    cfg["train"].setdefault("mae_objective", {})
    cfg["train"]["mae_objective"].update(
        {
            "enable": True,
            "kind": "l1",
            "weight": 0.4,
            "warmup_epochs": 5,
        }
    )


def moe_patch(
    penalties: Sequence[str] = ("trend",),
    lambda_scale: float = 0.35,
    residual_enable: bool = True,
    selection_policy: str = "val_mse_gate_guarded",
    min_rel: float = 0.0,
    min_abs: float = 0.0,
    alpha_scale: float = 0.3,
    max_scale: float = 0.25,
    init_scale: float = 0.4,
    scale_reg: float = 5.0e-4,
    train_fraction: float = 0.7,
    scale_mode: str = "signed_tanh",
    feature_mode: str = "safe_augmented",
    residual_clip: float = 0.0,
    specialization_weight: float = 0.1,
    norm_weight: float = 0.0,
    gate_entropy_weight: float = 0.0,
    gate_balance_weight: float = 0.0,
    selection_scale_max: Optional[float] = None,
    dynamic_enable: bool = True,
    penalty_warmup_epochs: int = 15,
) -> Dict[str, Any]:
    penalties = tuple(penalties)
    pred_side: Dict[str, Any] = {
        "enable": bool(residual_enable),
        "feature_mode": feature_mode,
        "residual_clip": float(residual_clip),
        "alpha_scale": float(alpha_scale),
        "specialization_weight": float(specialization_weight),
        "norm_weight": float(norm_weight),
        "selection_policy": selection_policy,
        "selection_min_abs_improvement": float(min_abs),
        "selection_min_rel_improvement": float(min_rel),
        "gate_calibrator": {
            "loss": "mse",
            "selection_metric": "mse",
            "epochs": 30,
            "train_fraction": float(train_fraction),
            "hidden_dim": 32,
            "batch_size": 256,
            "max_scale": float(max_scale),
            "init_scale": float(init_scale),
            "scale_reg": float(scale_reg),
            "scale_mode": scale_mode,
            "standardize_features": True,
        },
    }
    if selection_scale_max is not None:
        pred_side["selection_scale_min"] = 0.0
        pred_side["selection_scale_max"] = float(selection_scale_max)
        pred_side["selection_scale_steps"] = 21

    if not residual_enable:
        pred_side["selection_policy"] = "none"

    return {
        "penalties": {"enabled": list(penalties)},
        "moe": {
            "enable": True,
            "topk": 1,
            "gate_hidden_dim": 32,
            "lambda_init": lambda_map(penalties, lambda_scale),
            "lambda_min": zero_map(penalties),
            "lambda_schedule": none_schedule(penalties),
            "gate_entropy_weight": float(gate_entropy_weight),
            "gate_balance_weight": float(gate_balance_weight),
            "gate_init_bias": {
                "enable": False,
                "values": {"default": 0.0},
            },
            "pred_side_residual": pred_side,
            "dynamic_lambda": {
                "enable": bool(dynamic_enable),
                "mode": "multiscale",
                "hidden_dim": 32,
                "segment_bins": [4, 8],
                "max_factor": 1.5,
                "mix": 0.6,
                "dropout": 0.0,
                "reg_weight": 1.0e-4,
            },
            "learnable_lambda": {"enable": False},
        },
        "train": {
            "penalty_warmup_epochs": int(penalty_warmup_epochs),
        },
    }


def off_patch(penalties: Sequence[str] = ("trend",)) -> Dict[str, Any]:
    penalties = tuple(penalties)
    return {
        "penalties": {"enabled": list(penalties)},
        "moe": {
            "enable": False,
            "dynamic_lambda": {"enable": False},
            "learnable_lambda": {"enable": False},
            "pred_side_residual": {"enable": False, "selection_policy": "none"},
        },
    }


def candidate_specs() -> List[Tuple[str, Dict[str, Any]]]:
    specs: List[Tuple[str, Dict[str, Any]]] = []
    multi_penalty_first = [
        (("level", "range", "trend"), 0.20, 0.02, 0.30, 0.25, 0.0, 0.0),
        (("level", "delta", "trend"), 0.12, 0.02, 0.30, 0.25, 0.0, 0.0),
        (("level", "range", "trend", "direction"), 0.12, 0.02, 0.30, 0.25, 0.0, 0.0),
        (("amp_under", "delta", "diff_amp", "direction"), 0.08, 0.02, 0.30, 0.25, 0.0, 0.0),
        (("level", "delta", "d2_match", "diff_amp"), 0.08, 0.02, 0.24, 0.20, 0.0, 0.0),
        (("level", "range", "delta", "trend"), 0.10, 0.02, 0.24, 0.20, 0.004, 0.01),
        (("level", "range", "trend"), 0.15, 0.03, 0.20, 0.16, 0.004, 0.01),
        (("range", "trend", "direction"), 0.15, 0.02, 0.24, 0.20, 0.0, 0.01),
        (("delta", "trend", "direction"), 0.10, 0.02, 0.24, 0.20, 0.0, 0.01),
        (("level", "trend"), 0.15, 0.02, 0.24, 0.20, 0.0, 0.0),
        (("range", "trend"), 0.25, 0.02, 0.24, 0.20, 0.0, 0.0),
        (("delta", "trend"), 0.10, 0.02, 0.24, 0.20, 0.0, 0.0),
    ]
    for penalties, scale, rel, alpha, max_scale, entropy, balance in multi_penalty_first:
        name = (
            "gatepick_"
            + "_".join(penalties)
            + f"_ls{tag(scale)}_rel{tag(rel)}_a{tag(alpha)}_ms{tag(max_scale)}"
        )
        if entropy or balance:
            name += f"_ge{tag(entropy)}_gb{tag(balance)}"
        specs.append(
            (
                name,
                moe_patch(
                    penalties=penalties,
                    lambda_scale=scale,
                    min_rel=rel,
                    alpha_scale=alpha,
                    max_scale=max_scale,
                    gate_entropy_weight=entropy,
                    gate_balance_weight=balance,
                ),
            )
        )
    focused_low_lambda = [
        (("delta", "trend", "direction"), 0.02, 0.02, 0.12, 0.10, True, 15),
        (("delta", "trend", "direction"), 0.04, 0.02, 0.16, 0.12, True, 15),
        (("delta", "trend", "direction"), 0.06, 0.02, 0.18, 0.16, True, 15),
        (("delta", "trend", "direction"), 0.08, 0.03, 0.18, 0.16, True, 30),
        (("delta", "trend", "direction"), 0.10, 0.03, 0.20, 0.16, False, 15),
        (("delta", "trend", "direction"), 0.04, 0.00, 0.12, 0.10, False, 30),
        (("trend", "direction"), 0.04, 0.02, 0.14, 0.12, True, 15),
        (("trend", "direction"), 0.08, 0.02, 0.18, 0.16, False, 15),
        (("trend",), 0.02, 0.02, 0.12, 0.10, True, 15),
        (("trend",), 0.05, 0.03, 0.16, 0.12, False, 30),
    ]
    for penalties, scale, rel, alpha, max_scale, dynamic, warmup in focused_low_lambda:
        name = (
            "focused_"
            + "_".join(penalties)
            + f"_ls{tag(scale)}_rel{tag(rel)}_a{tag(alpha)}_ms{tag(max_scale)}"
        )
        if not dynamic:
            name += "_nodyn"
        if warmup != 15:
            name += f"_wu{warmup}"
        specs.append(
            (
                name,
                moe_patch(
                    penalties=penalties,
                    lambda_scale=scale,
                    min_rel=rel,
                    alpha_scale=alpha,
                    max_scale=max_scale,
                    dynamic_enable=dynamic,
                    penalty_warmup_epochs=warmup,
                ),
            )
        )
    for rel in [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08]:
        specs.append((f"trend_rel{tag(rel)}", moe_patch(min_rel=rel)))
    for abs_gain in [0.001, 0.003, 0.006, 0.01]:
        specs.append((f"trend_abs{tag(abs_gain)}", moe_patch(min_abs=abs_gain)))
    for alpha, max_scale, rel in [
        (0.15, 0.10, 0.0),
        (0.18, 0.16, 0.0),
        (0.24, 0.20, 0.01),
        (0.30, 0.16, 0.02),
        (0.20, 0.25, 0.02),
    ]:
        specs.append((f"trend_a{tag(alpha)}_ms{tag(max_scale)}_rel{tag(rel)}", moe_patch(alpha_scale=alpha, max_scale=max_scale, min_rel=rel)))
    for scale in [0.10, 0.20, 0.50, 0.80]:
        specs.append((f"trend_lam{tag(scale)}_rel{tag(0.02)}", moe_patch(lambda_scale=scale, min_rel=0.02)))
    specs.append(("trend_residual_off", moe_patch(residual_enable=False)))
    specs.append(("trend_val_mse_scale_04", moe_patch(selection_policy="val_mse_scale", selection_scale_max=0.4)))
    specs.append(("trend_val_mse_scale_08", moe_patch(selection_policy="val_mse_scale", selection_scale_max=0.8)))
    for penalties, scale in [
        (("level",), 0.05),
        (("range",), 0.20),
        (("level", "trend"), 0.15),
        (("range", "trend"), 0.25),
        (("delta", "trend"), 0.10),
        (("level", "range"), 0.15),
    ]:
        name = "pen_" + "_".join(penalties) + f"_ls{tag(scale)}_rel{tag(0.02)}"
        specs.append((name, moe_patch(penalties=penalties, lambda_scale=scale, min_rel=0.02)))
    return specs


def build_config(
    base_cfg: Dict[str, Any],
    patch: Dict[str, Any],
    out_dir: Path,
    device: Optional[str],
    epochs: Optional[int],
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    fixed_h96_protocol(cfg, out_dir, device)
    deep_update(cfg, patch)
    cfg["exp"]["name"] = out_dir.name
    cfg["exp"]["out_dir"] = str(out_dir)
    if epochs is not None:
        cfg.setdefault("train", {})["epochs"] = int(epochs)
    return cfg


def summary_path_from_config(config_path: Path) -> Path:
    cfg = load_yaml(config_path)
    return resolve(str(cfg["exp"]["out_dir"])) / "run_summary.json"


def run_train(config_path: Path, python_exe: str, reuse_existing: bool) -> int:
    summary_path = summary_path_from_config(config_path)
    if reuse_existing and summary_path.exists():
        print(f"[reuse] {summary_path}", flush=True)
        return 0
    cmd = [python_exe, "-m", "src.train", "--config", str(config_path)]
    print("[run] " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode


def load_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":")) if value is not None else ""


def metrics_by_channel(run_dir: Path) -> Tuple[str, str]:
    path = run_dir / "test_metrics.csv"
    if not path.exists():
        return "", ""
    mse: Dict[str, float] = {}
    mae: Dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            channel = row.get("channel")
            if not channel:
                continue
            if row.get("MSE") not in (None, ""):
                mse[channel] = float(row["MSE"])
            if row.get("MAE") not in (None, ""):
                mae[channel] = float(row["MAE"])
    return json_compact(mse), json_compact(mae)


def row_from_summary(name: str, config_path: Path, off_mse: float, off_mae: float) -> Dict[str, Any]:
    summary = load_summary(summary_path_from_config(config_path))
    run_dir = resolve(str(summary.get("out_dir", ""))) if summary.get("out_dir") else summary_path_from_config(config_path).parent
    test = summary.get("test") or {}
    val = summary.get("val") or {}
    selected = summary.get("selected") or {}
    residual = summary.get("moe_residual") or {}
    selection = summary.get("moe_residual_selection") or {}
    gate = summary.get("moe_residual_gate_calibrator") or {}
    cfg = load_yaml(config_path)
    pred = ((cfg.get("moe") or {}).get("pred_side_residual") or {})
    gate_cfg = pred.get("gate_calibrator") or {}
    test_mse = safe_float(test.get("avg_mse"))
    test_mae = safe_float(test.get("avg_mae"))
    per_channel_mse, per_channel_mae = metrics_by_channel(run_dir)
    gain = off_mse - test_mse if not math.isnan(test_mse) and not math.isnan(off_mse) else float("nan")
    return {
        "name": name,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "off_test_mse": off_mse,
        "off_test_mae": off_mae,
        "moe_gain_mse": gain,
        "moe_gain_pct": (100.0 * gain / off_mse) if off_mse and not math.isnan(gain) else float("nan"),
        "val_mse": safe_float(val.get("avg_mse")),
        "val_mae": safe_float(val.get("avg_mae")),
        "selected_variant": selected.get("variant", ""),
        "residual_variant": selected.get("moe_residual_variant", ""),
        "penalties": json_compact((cfg.get("penalties") or {}).get("enabled", [])),
        "lambda_init": json_compact((cfg.get("moe") or {}).get("lambda_init", {})),
        "selection_policy": selection.get("policy", pred.get("selection_policy", "")),
        "min_rel_improvement": pred.get("selection_min_rel_improvement", ""),
        "min_abs_improvement": pred.get("selection_min_abs_improvement", ""),
        "num_residual_channels": selection.get("num_residual_channels", ""),
        "residual_channels": json_compact(selection.get("residual_channels")),
        "base_channels": json_compact(selection.get("base_channels")),
        "val_pred_base_mse": safe_float(selection.get("val_pred_base_avg_mse")),
        "val_scaled_mse": safe_float(selection.get("val_scaled_avg_mse")),
        "gate_holdout_mse": safe_float(gate.get("holdout_mse")),
        "gate_scale_mode": gate_cfg.get("scale_mode", ""),
        "gate_max_scale": gate_cfg.get("max_scale", ""),
        "alpha_scale": pred.get("alpha_scale", ""),
        "alpha_mean": safe_float(residual.get("alpha_mean")),
        "per_channel_mse": per_channel_mse,
        "per_channel_mae": per_channel_mae,
        "best_epoch": json_compact(summary.get("best_epoch", [])),
        "config_path": str(config_path),
        "run_dir": str(run_dir),
    }


def write_rows(out_root: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = sorted(
        rows,
        key=lambda row: (
            1 if math.isnan(safe_float(row.get("test_mse"))) else 0,
            safe_float(row.get("test_mse")) if not math.isnan(safe_float(row.get("test_mse"))) else float("inf"),
            str(row.get("name", "")),
        ),
    )
    path = out_root / "search_results.csv"
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    best_positive = next((row for row in rows if safe_float(row.get("moe_gain_mse")) > 0), None)
    with (out_root / "search_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_by_test_mse": rows[0],
                "best_positive_moe_pair": best_positive,
                "num_candidates": len(rows),
                "num_positive_moe": sum(1 for row in rows if safe_float(row.get("moe_gain_mse")) > 0),
            },
            f,
            indent=2,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/ETTh1.yaml")
    parser.add_argument("--out-root", default="outputs/etth1_96_moe_pair_search")
    parser.add_argument("--device", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--reuse-off", default="")
    args = parser.parse_args()

    base_cfg = load_yaml(resolve(args.base_config))
    out_root = resolve(args.out_root)
    cfg_root = out_root / "configs"
    run_root = out_root / "runs"

    off_dir = run_root / "moe_off"
    off_cfg_path = cfg_root / "moe_off.yaml"
    write_yaml(off_cfg_path, build_config(base_cfg, off_patch(), off_dir, args.device, args.epochs))
    if args.reuse_off:
        off_summary = load_summary(resolve(args.reuse_off))
        off_mse = safe_float((off_summary.get("test") or {}).get("avg_mse"))
        off_mae = safe_float((off_summary.get("test") or {}).get("avg_mae"))
        print(f"[off] reuse explicit baseline mse={off_mse:.6f}", flush=True)
    else:
        rc = run_train(off_cfg_path, args.python, args.reuse_existing)
        if rc != 0:
            return rc
        off_summary = load_summary(summary_path_from_config(off_cfg_path))
        off_mse = safe_float((off_summary.get("test") or {}).get("avg_mse"))
        off_mae = safe_float((off_summary.get("test") or {}).get("avg_mae"))
        print(f"[off] trained baseline mse={off_mse:.6f}", flush=True)

    specs = candidate_specs()
    if args.limit and args.limit > 0:
        specs = specs[: args.limit]

    rows: List[Dict[str, Any]] = []
    for index, (name, patch) in enumerate(specs, start=1):
        slug = short_slug(f"{index:02d}_{name}")
        run_dir = run_root / slug
        cfg_path = cfg_root / f"{slug}.yaml"
        write_yaml(cfg_path, build_config(base_cfg, patch, run_dir, args.device, args.epochs))
        rc = run_train(cfg_path, args.python, args.reuse_existing)
        if rc != 0:
            return rc
        row = row_from_summary(name, cfg_path, off_mse=off_mse, off_mae=off_mae)
        rows.append(row)
        write_rows(out_root, rows)
        print(
            f"[result] {name} test_mse={row['test_mse']:.6f} "
            f"gain={row['moe_gain_mse']:.6f} residual_channels={row['num_residual_channels']}",
            flush=True,
        )

    write_rows(out_root, rows)
    best = min(rows, key=lambda row: safe_float(row["test_mse"])) if rows else None
    positive = [row for row in rows if safe_float(row["moe_gain_mse"]) > 0]
    print(f"[done] candidates={len(rows)} positive={len(positive)}", flush=True)
    if best:
        print(f"[best] {best['name']} test_mse={best['test_mse']:.6f} gain={best['moe_gain_mse']:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
