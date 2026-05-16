import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_knn_shape_variants import make_loader, run_eval  # noqa: E402
from scripts.compare_moe_on_off import (  # noqa: E402
    compute_penalty_scale,
    load_eval_modules,
    load_yaml,
    prepare_data_context,
)
from src.data.windows import make_strict_windows  # noqa: E402
from src.models.penalties import build_penalty_bank  # noqa: E402
from src.utils.knn_shape import KNNShapeConfig, ShapeKNNHybrid, predict_bank_outputs  # noqa: E402


CSV_FIELDS = [
    "status",
    "stage",
    "stage_rank",
    "global_rank",
    "candidate",
    "mode",
    "scope",
    "bank_split",
    "bank_stride",
    "feature_mode",
    "template_mode",
    "distance_weight",
    "anchor_mode",
    "shape_bins",
    "diff_bins",
    "pred_shape_bins",
    "pred_diff_bins",
    "k",
    "alpha",
    "adaptive_alpha",
    "distance_sharpness",
    "confidence_floor",
    "val_avg_mse",
    "val_avg_mae",
    "val_delta_mse_vs_base",
    "val_gain_mse_vs_base",
    "val_delta_mae_vs_base",
    "val_gain_mae_vs_base",
    "val_confidence",
    "val_effective_alpha",
    "test_avg_mse",
    "test_avg_mae",
    "test_delta_mse_vs_base",
    "test_gain_mse_vs_base",
    "test_delta_mae_vs_base",
    "test_gain_mae_vs_base",
    "test_confidence",
    "test_effective_alpha",
    "eval_sec",
    "error",
]


@dataclass(frozen=True)
class Candidate:
    k: int
    alpha: float
    adaptive_alpha: str
    distance_sharpness: float
    confidence_floor: float


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def parse_int_list(text: str) -> list[int]:
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("Expected at least one integer.")
    return values


def parse_float_list(text: str) -> list[float]:
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("Expected at least one float.")
    return values


def parse_modes(text: str) -> list[str]:
    modes = []
    for item in str(text).split(","):
        item = item.strip().lower()
        if item:
            modes.append(item)
    if not modes:
        raise ValueError("Expected at least one adaptive alpha mode.")
    return modes


def fmt_float(value: float) -> str:
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p")


def candidate_name(c: Candidate) -> str:
    return (
        f"k{int(c.k)}_a{fmt_float(c.alpha)}_aa{c.adaptive_alpha}"
        f"_ds{fmt_float(c.distance_sharpness)}_cf{fmt_float(c.confidence_floor)}"
    )


def candidate_key(c: Candidate) -> tuple[Any, ...]:
    adaptive = str(c.adaptive_alpha).lower()
    if adaptive == "none":
        return (int(c.k), round(float(c.alpha), 8), adaptive, 1.0, 0.0)
    return (
        int(c.k),
        round(float(c.alpha), 8),
        adaptive,
        round(float(c.distance_sharpness), 8),
        round(float(c.confidence_floor), 8),
    )


def generate_candidates(
    *,
    k_values: list[int],
    alpha_values: list[float],
    adaptive_modes: list[str],
    sharpness_values: list[float],
    confidence_floor_values: list[float],
) -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[tuple[Any, ...]] = set()
    for k in k_values:
        for alpha in alpha_values:
            for adaptive in adaptive_modes:
                if adaptive == "none":
                    items = [(1.0, 0.0)]
                else:
                    items = [(s, f) for s in sharpness_values for f in confidence_floor_values]
                for sharpness, floor in items:
                    cand = Candidate(
                        k=max(1, int(k)),
                        alpha=max(0.0, float(alpha)),
                        adaptive_alpha=str(adaptive).lower(),
                        distance_sharpness=max(0.0, float(sharpness)),
                        confidence_floor=min(1.0, max(0.0, float(floor))),
                    )
                    key = candidate_key(cand)
                    if key not in seen:
                        seen.add(key)
                        candidates.append(cand)
    return candidates


def metric_field(metric: str) -> str:
    metric = str(metric).lower()
    if metric == "mae":
        return "val_avg_mae"
    return "val_avg_mse"


def sort_rows(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    field = metric_field(metric)

    def key(row: dict[str, Any]) -> tuple[float, float, str]:
        try:
            primary = float(row.get(field, "inf"))
        except (TypeError, ValueError):
            primary = float("inf")
        try:
            mse = float(row.get("val_avg_mse", "inf"))
        except (TypeError, ValueError):
            mse = float("inf")
        return primary, mse, str(row.get("candidate", ""))

    return sorted(rows, key=key)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def short_candidate(row: dict[str, Any]) -> str:
    if row.get("candidate") == "base":
        return "base"
    return (
        f"{row.get('stage', '')} "
        f"k={row.get('k', '')} "
        f"a={as_float(row.get('alpha', 0.0)):.2f} "
        f"{row.get('adaptive_alpha', '')}"
    )


def markdown_table(rows: list[dict[str, Any]], fields: list[str], limit: int) -> str:
    take = rows[: max(0, int(limit))]
    if not take:
        return "_No rows._"
    lines = []
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
    for row in take:
        vals = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                vals.append(f"{value:.6g}")
            else:
                num = as_float(value)
                if not math.isnan(num) and field not in {"k", "stage_rank", "global_rank"}:
                    vals.append(f"{num:.6g}")
                else:
                    vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_markdown_report(
    path: Path,
    *,
    base_row: dict[str, Any],
    coarse_sorted: list[dict[str, Any]],
    fine_sorted: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    metric: str,
) -> None:
    fields = [
        "global_rank",
        "stage",
        "candidate",
        "k",
        "alpha",
        "adaptive_alpha",
        "val_avg_mse",
        "val_gain_mse_vs_base",
        "test_avg_mse",
        "test_gain_mse_vs_base",
    ]
    text = "\n".join(
        [
            "# ETTm1 Val-Based KNN/Hybrid Refinement",
            "",
            f"Ranking metric: `val_{metric}`. Test metrics are reported only as reference.",
            "",
            "## Base",
            "",
            markdown_table([base_row], fields, 1),
            "",
            "## Best Coarse Rows",
            "",
            markdown_table(coarse_sorted, fields, 10),
            "",
            "## Best Fine Rows",
            "",
            markdown_table(fine_sorted, fields, 10),
            "",
            "## Overall Val-Ranked Rows",
            "",
            markdown_table(ranked, fields, 20),
            "",
            "## Figures",
            "",
            "- `refinement_trace.png`",
            "- `top_val_candidates.png`",
            "- `heatmap_*.png`",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def plot_refinement_trace(
    path: Path,
    *,
    base_row: dict[str, Any],
    coarse_sorted: list[dict[str, Any]],
    fine_sorted: list[dict[str, Any]],
) -> None:
    labels = ["base"]
    vals = [as_float(base_row.get("val_avg_mse"))]
    if coarse_sorted:
        labels.append("coarse best")
        vals.append(as_float(coarse_sorted[0].get("val_avg_mse")))
    if fine_sorted:
        labels.append("fine best")
        vals.append(as_float(fine_sorted[0].get("val_avg_mse")))

    plt.figure(figsize=(6.5, 4.2))
    plt.plot(labels, vals, marker="o", linewidth=2)
    for idx, val in enumerate(vals):
        if not math.isnan(val):
            plt.text(idx, val, f"{val:.6f}", ha="center", va="bottom", fontsize=9)
    plt.ylabel("Validation MSE")
    plt.title("Val-MSE Refinement Trace")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_top_candidates(path: Path, ranked: list[dict[str, Any]], limit: int = 12) -> None:
    rows = [r for r in ranked if r.get("candidate") != "base"][:limit]
    if not rows:
        return
    labels = [f"#{i + 1}" for i in range(len(rows))]
    val = [as_float(r.get("val_avg_mse")) for r in rows]
    test = [as_float(r.get("test_avg_mse")) for r in rows]

    plt.figure(figsize=(9.5, 4.8))
    plt.bar(labels, val, color="#4c78a8", label="val_mse")
    has_test = any(not math.isnan(v) for v in test)
    if has_test:
        plt.plot(labels, test, color="#f58518", marker="o", linewidth=2, label="test_mse reference")
    for idx, row in enumerate(rows):
        plt.text(idx, val[idx], f"k={row.get('k')}\na={as_float(row.get('alpha')):.2f}", ha="center", va="bottom", fontsize=8)
    plt.ylabel("MSE")
    plt.title("Top Val-Ranked KNN/Hybrid Candidates")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_heatmaps(out_dir: Path, rows: list[dict[str, Any]]) -> list[Path]:
    paths: list[Path] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("candidate") == "base" or row.get("status") != "ok":
            continue
        key = (str(row.get("stage", "")), str(row.get("adaptive_alpha", "")))
        grouped.setdefault(key, []).append(row)

    for (stage, adaptive), group in grouped.items():
        alphas = sorted({as_float(r.get("alpha")) for r in group if not math.isnan(as_float(r.get("alpha")))})
        ks = sorted({int(float(r.get("k"))) for r in group if str(r.get("k", "")).strip()})
        if len(alphas) < 2 or len(ks) < 2:
            continue
        grid = [[math.nan for _ in alphas] for _ in ks]
        for r in group:
            k = int(float(r.get("k")))
            alpha = as_float(r.get("alpha"))
            value = as_float(r.get("val_avg_mse"))
            if math.isnan(alpha) or math.isnan(value):
                continue
            yi = ks.index(k)
            xi = alphas.index(alpha)
            cur = grid[yi][xi]
            if math.isnan(cur) or value < cur:
                grid[yi][xi] = value
        plt.figure(figsize=(max(6.5, len(alphas) * 0.6), max(3.8, len(ks) * 0.45)))
        im = plt.imshow(grid, aspect="auto", cmap="viridis")
        plt.colorbar(im, label="val_mse")
        plt.xticks(range(len(alphas)), [f"{a:.2f}" for a in alphas], rotation=45, ha="right")
        plt.yticks(range(len(ks)), [str(k) for k in ks])
        plt.xlabel("alpha")
        plt.ylabel("k")
        plt.title(f"{stage} val_mse heatmap: adaptive={adaptive}")
        plt.tight_layout()
        path = out_dir / f"heatmap_{stage}_{adaptive}.png"
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(path)
    return paths


def make_knn_config(base_cfg: dict[str, Any], args: argparse.Namespace, cand: Candidate) -> KNNShapeConfig:
    knn_dict = dict(base_cfg.get("knn_hybrid", {}) or {})
    knn_dict.update(
        {
            "enable": True,
            "use_for_model_selection": False,
            "mode": str(args.mode),
            "scope": str(args.scope),
            "bank_split": str(args.bank_split),
            "bank_stride": int(args.bank_stride),
            "feature_mode": str(args.feature_mode),
            "template_mode": str(args.template_mode),
            "distance_weight": str(args.distance_weight),
            "anchor_mode": str(args.anchor_mode),
            "shape_bins": int(args.shape_bins),
            "diff_bins": int(args.diff_bins),
            "pred_shape_bins": int(args.pred_shape_bins),
            "pred_diff_bins": int(args.pred_diff_bins),
            "k": int(cand.k),
            "alpha": float(cand.alpha),
            "alpha_horizon_ref": 0,
            "alpha_horizon_power": 0.0,
            "adaptive_alpha": str(cand.adaptive_alpha),
            "distance_sharpness": float(cand.distance_sharpness),
            "confidence_floor": float(cand.confidence_floor),
        }
    )
    return KNNShapeConfig.from_dict(knn_dict)


def bank_signature(split: str, cfg: KNNShapeConfig) -> tuple[Any, ...]:
    return (
        split,
        cfg.mode,
        cfg.scope,
        cfg.bank_split,
        int(cfg.bank_stride),
        cfg.feature_mode,
        cfg.template_mode,
        int(cfg.shape_bins),
        int(cfg.diff_bins),
        int(cfg.pred_shape_bins),
        int(cfg.pred_diff_bins),
        cfg.anchor_mode,
    )


def make_bank_windows(context, split: str, knn_cfg: KNNShapeConfig):
    total_len = int(context.norm_data_tc.shape[0])
    if split == "val":
        if knn_cfg.bank_split == "train":
            starts = torch.arange(0, int(context.xtr_norm.shape[0]), dtype=torch.long)
            return context.xtr_norm, context.ytr_norm, starts, "train"
        x_bank, y_bank = make_strict_windows(context.norm_data_tc, context.L, context.H, 0, context.t_val)
        starts = torch.arange(0, int(x_bank.shape[0]), dtype=torch.long)
        return x_bank, y_bank, starts, "pre_val"

    if knn_cfg.bank_split == "train":
        starts = torch.arange(0, int(context.xtr_norm.shape[0]), dtype=torch.long)
        return context.xtr_norm, context.ytr_norm, starts, "train"
    if knn_cfg.bank_split == "history" and knn_cfg.mode == "rolling":
        x_bank, y_bank = make_strict_windows(context.norm_data_tc, context.L, context.H, 0, total_len)
        starts = torch.arange(0, int(x_bank.shape[0]), dtype=torch.long)
        return x_bank, y_bank, starts, "full_history"
    x_bank, y_bank = make_strict_windows(context.norm_data_tc, context.L, context.H, 0, context.t_val)
    starts = torch.arange(0, int(x_bank.shape[0]), dtype=torch.long)
    return x_bank, y_bank, starts, "pre_test"


def add_delta_fields(row: dict[str, Any], base_val: dict[str, Any], base_test: dict[str, Any] | None = None) -> None:
    val_mse = float(row["val_avg_mse"])
    val_mae = float(row["val_avg_mae"])
    base_val_mse = float(base_val["val_avg_mse"])
    base_val_mae = float(base_val["val_avg_mae"])
    row["val_delta_mse_vs_base"] = val_mse - base_val_mse
    row["val_gain_mse_vs_base"] = base_val_mse - val_mse
    row["val_delta_mae_vs_base"] = val_mae - base_val_mae
    row["val_gain_mae_vs_base"] = base_val_mae - val_mae
    if base_test is not None and row.get("test_avg_mse", "") != "":
        test_mse = float(row["test_avg_mse"])
        test_mae = float(row["test_avg_mae"])
        base_test_mse = float(base_test["test_avg_mse"])
        base_test_mae = float(base_test["test_avg_mae"])
        row["test_delta_mse_vs_base"] = test_mse - base_test_mse
        row["test_gain_mse_vs_base"] = base_test_mse - test_mse
        row["test_delta_mae_vs_base"] = test_mae - base_test_mae
        row["test_gain_mae_vs_base"] = base_test_mae - test_mae


def make_base_row(base_val: dict[str, Any], base_test: dict[str, Any] | None) -> dict[str, Any]:
    row = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "status": "ok",
            "stage": "base",
            "stage_rank": 0,
            "global_rank": 0,
            "candidate": "base",
            "k": 0,
            "alpha": 0.0,
            "adaptive_alpha": "",
            "val_avg_mse": base_val["val_avg_mse"],
            "val_avg_mae": base_val["val_avg_mae"],
            "val_delta_mse_vs_base": 0.0,
            "val_gain_mse_vs_base": 0.0,
            "val_delta_mae_vs_base": 0.0,
            "val_gain_mae_vs_base": 0.0,
        }
    )
    if base_test is not None:
        row.update(
            {
                "test_avg_mse": base_test["test_avg_mse"],
                "test_avg_mae": base_test["test_avg_mae"],
                "test_delta_mse_vs_base": 0.0,
                "test_gain_mse_vs_base": 0.0,
                "test_delta_mae_vs_base": 0.0,
                "test_gain_mae_vs_base": 0.0,
            }
        )
    return row


def load_summary_base_metrics(run_dir: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        return None, None
    try:
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None, None

    base_val = None
    val = summary.get("val") or {}
    if "avg_mse" in val and "avg_mae" in val:
        base_val = {
            "val_avg_mse": float(val["avg_mse"]),
            "val_avg_mae": float(val["avg_mae"]),
            "val_confidence": "",
            "val_effective_alpha": "",
        }

    base_test = None
    test = summary.get("test") or {}
    if "avg_mse" in test and "avg_mae" in test:
        base_test = {
            "test_avg_mse": float(test["avg_mse"]),
            "test_avg_mae": float(test["avg_mae"]),
            "test_confidence": "",
            "test_effective_alpha": "",
        }
    return base_val, base_test


def fine_candidates_from_best(
    best_rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    coarse_modes: list[str],
    seen: set[tuple[Any, ...]],
) -> list[Candidate]:
    candidates: list[Candidate] = []
    multipliers = parse_float_list(args.fine_k_multipliers)
    alpha_radius = float(args.fine_alpha_radius)
    alpha_step = float(args.fine_alpha_step)
    sharpness_mult = parse_float_list(args.fine_sharpness_multipliers)
    floor_offsets = parse_float_list(args.fine_confidence_floor_offsets)

    for row in best_rows:
        base_k = max(1, int(float(row["k"])))
        k_values = sorted({max(1, int(round(base_k * m))) for m in multipliers})
        base_alpha = float(row["alpha"])
        lo = max(0.0, base_alpha - alpha_radius)
        hi = min(float(args.max_alpha), base_alpha + alpha_radius)
        steps = max(1, int(round((hi - lo) / max(alpha_step, 1.0e-9))))
        alpha_values = sorted({round(lo + i * alpha_step, 8) for i in range(steps + 1)} | {round(base_alpha, 8)})
        alpha_values = [a for a in alpha_values if 0.0 <= a <= float(args.max_alpha)]

        adaptive = str(row["adaptive_alpha"]).lower()
        if args.fine_adaptive == "all":
            modes = coarse_modes
        elif args.fine_adaptive == "best_plus_none":
            modes = sorted({adaptive, "none"})
        else:
            modes = [adaptive]

        base_sharpness = float(row["distance_sharpness"] or 1.0)
        base_floor = float(row["confidence_floor"] or 0.0)
        for k in k_values:
            for alpha in alpha_values:
                for mode in modes:
                    if mode == "none":
                        raw = [Candidate(k, alpha, mode, 1.0, 0.0)]
                    else:
                        raw = [
                            Candidate(
                                k,
                                alpha,
                                mode,
                                max(0.0, base_sharpness * sm),
                                min(1.0, max(0.0, base_floor + fo)),
                            )
                            for sm in sharpness_mult
                            for fo in floor_offsets
                        ]
                    for cand in raw:
                        key = candidate_key(cand)
                        if key not in seen:
                            seen.add(key)
                            candidates.append(cand)
    return candidates


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Val-driven coarse-to-fine KNN hybrid parameter sensitivity search using an existing checkpoint."
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--eval-batch-size", type=int, default=None)
    ap.add_argument("--metric", choices=["mse", "mae"], default="mse")
    ap.add_argument("--eval-test-top", type=int, default=5)

    ap.add_argument("--mode", choices=["fixed", "rolling"], default="fixed")
    ap.add_argument("--scope", choices=["same_channel", "same_cluster"], default="same_channel")
    ap.add_argument("--bank-split", choices=["train", "pre_test", "history"], default="train")
    ap.add_argument("--bank-stride", type=int, default=2)
    ap.add_argument("--feature-mode", choices=["hist", "joint"], default="joint")
    ap.add_argument("--template-mode", choices=["future", "residual"], default="residual")
    ap.add_argument("--distance-weight", choices=["inverse", "uniform"], default="inverse")
    ap.add_argument("--anchor-mode", choices=["last", "mean"], default="last")
    ap.add_argument("--shape-bins", type=int, default=24)
    ap.add_argument("--diff-bins", type=int, default=12)
    ap.add_argument("--pred-shape-bins", type=int, default=16)
    ap.add_argument("--pred-diff-bins", type=int, default=8)

    ap.add_argument("--coarse-k", default="16,48,96,160")
    ap.add_argument("--coarse-alpha", default="0.0,0.3,0.6,1.0,1.4,1.8")
    ap.add_argument("--coarse-adaptive", default="none,confidence,distance_agreement")
    ap.add_argument("--coarse-sharpness", default="0.5,1.0,2.0")
    ap.add_argument("--coarse-confidence-floor", default="0.0,0.1")

    ap.add_argument("--fine-top-n", type=int, default=3)
    ap.add_argument("--fine-k-multipliers", default="0.5,0.75,1.0,1.25,1.5")
    ap.add_argument("--fine-alpha-radius", type=float, default=0.3)
    ap.add_argument("--fine-alpha-step", type=float, default=0.1)
    ap.add_argument("--fine-adaptive", choices=["same", "best_plus_none", "all"], default="same")
    ap.add_argument("--fine-sharpness-multipliers", default="0.5,1.0,2.0")
    ap.add_argument("--fine-confidence-floor-offsets", default="-0.1,0.0,0.1")
    ap.add_argument("--max-alpha", type=float, default=2.0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    cfg = load_yaml(config_path)
    run_dir = resolve_path(args.run_dir) if args.run_dir else resolve_path(cfg["exp"]["out_dir"])
    checkpoint_path = run_dir / "best_checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    out_dir = resolve_path(args.out_dir) if args.out_dir else (run_dir / "knn_val_refinement")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_csv = out_dir / "all_results.csv"
    coarse_csv = out_dir / "coarse_results.csv"
    fine_csv = out_dir / "fine_results.csv"
    trace_csv = out_dir / "refinement_trace.csv"

    device_name = args.device or cfg["exp"].get("device", "cpu")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    batch_size = int(args.eval_batch_size or cfg["train"]["batch_size"])

    print(f"Config: {config_path}")
    print(f"Run dir: {run_dir}")
    print(f"Out dir: {out_dir}")
    print(f"Device: {device}")
    print(f"Ranking metric: val_{args.metric}")

    context = prepare_data_context(cfg)
    bundle = load_eval_modules(cfg, checkpoint_path, context.K, device)
    model = bundle["model"]
    gate = bundle["gate"]
    dynamic_lambda = bundle["dynamic_lambda"]
    pred_residual = bundle.get("pred_residual")
    lambda_kp = bundle["base_lambda_kp"]
    lambda_min_kp = bundle["lambda_min_kp"]
    penalty_names = bundle["penalty_names"]

    cluster_id_c = context.cluster_id_c.to(device)
    cluster_sizes = torch.bincount(cluster_id_c, minlength=context.K).float().to(device)
    cluster_weight_k = cluster_sizes / cluster_sizes.sum().clamp_min(1.0)

    penalty_fns = build_penalty_bank(penalty_names, jump_thr=float(cfg["penalties"]["jump_threshold"]))
    penalty_scale = compute_penalty_scale(
        xtr_norm=context.xtr_norm,
        ytr_norm=context.ytr_norm,
        penalty_names=penalty_names,
        penalty_fns=penalty_fns,
        pred_len=context.H,
        batch_size=batch_size,
        device=device,
        floor=float(cfg["train"].get("penalty_scale_floor", 1.0e-3)),
    )

    xva, yva = make_strict_windows(context.norm_data_tc, context.L, context.H, context.t_train, context.t_val)
    val_loader = make_loader(xva, yva, batch_size)
    test_loader = make_loader(context.xte_norm, context.yte_norm, batch_size)

    raw_select_ranks = cfg.get("moe", {}).get("select_ranks", [1])
    select_ranks = [int(v) for v in raw_select_ranks]
    moe_cfg = cfg["moe"]
    gate_soft_weight = float(moe_cfg.get("gate_soft_weight", 0.0))

    common_eval_kwargs = {
        "model": model,
        "gate": gate,
        "lambda_kp": lambda_kp,
        "penalty_names": penalty_names,
        "penalty_fns": penalty_fns,
        "cluster_id_c": cluster_id_c,
        "K": context.K,
        "moe_cfg": moe_cfg,
        "device": device,
        "select_ranks": select_ranks,
        "channel_count": len(context.channel_names),
        "mse_weight": float(cfg["train"].get("mse_weight", 1.0)),
        "gate_entropy_weight": float(moe_cfg.get("gate_entropy_weight", 0.0)),
        "gate_balance_weight": float(moe_cfg.get("gate_balance_weight", 0.0)),
        "gate_soft_weight": gate_soft_weight,
        "gate_entropy_target_frac": float(moe_cfg.get("gate_entropy_target_frac", 0.0)),
        "penalty_scale": penalty_scale,
        "dynamic_lambda": dynamic_lambda,
        "lambda_min_kp": lambda_min_kp,
        "cluster_weight_k": cluster_weight_k,
        "pred_residual": pred_residual,
    }

    print("Evaluate base on val/test...")
    base_val, base_test = load_summary_base_metrics(run_dir)
    if base_val is None:
        base_val = run_eval(
            split_name="val",
            loader=val_loader,
            eval_start=context.t_train,
            hybrid=None,
            **common_eval_kwargs,
        )
    if base_test is None and int(args.eval_test_top) > 0:
        base_test = run_eval(
            split_name="test",
            loader=test_loader,
            eval_start=context.t_val,
            hybrid=None,
            **common_eval_kwargs,
        )

    bank_cache: dict[tuple[Any, ...], ShapeKNNHybrid] = {}
    base_pred_cache: dict[tuple[Any, ...], torch.Tensor | None] = {}

    def get_hybrid(split: str, knn_cfg: KNNShapeConfig) -> ShapeKNNHybrid:
        sig = bank_signature(split, knn_cfg)
        if sig not in bank_cache:
            x_bank, y_bank, starts, label = make_bank_windows(context, split, knn_cfg)
            base_bank_pred = None
            if knn_cfg.needs_base_bank_prediction():
                pred_key = (sig, int(x_bank.shape[0]), int(x_bank.shape[1]), int(x_bank.shape[2]))
                if pred_key not in base_pred_cache:
                    base_pred_cache[pred_key] = predict_bank_outputs(
                        model=model,
                        x_bank_ncl=x_bank,
                        cluster_id_c=cluster_id_c,
                        batch_size=max(batch_size, 64),
                        device=device,
                    )
                base_bank_pred = base_pred_cache[pred_key]
            print(
                f"Build {split} bank: {label}, mode={knn_cfg.mode}, scope={knn_cfg.scope}, "
                f"bank_split={knn_cfg.bank_split}, windows={int(x_bank.shape[0])}"
            )
            bank_cache[sig] = ShapeKNNHybrid.fit(
                x_bank_ncl=x_bank,
                y_bank_nch=y_bank,
                cluster_id_c=cluster_id_c,
                cfg=knn_cfg,
                start_offsets_n=starts,
                base_bank_pred_nch=base_bank_pred,
            )
        return ShapeKNNHybrid(cfg=knn_cfg, banks=bank_cache[sig].banks)

    def evaluate_candidate(cand: Candidate, stage: str) -> dict[str, Any]:
        row = {field: "" for field in CSV_FIELDS}
        row.update(
            {
                "status": "ok",
                "stage": stage,
                "candidate": candidate_name(cand),
                "k": int(cand.k),
                "alpha": float(cand.alpha),
                "adaptive_alpha": cand.adaptive_alpha,
                "distance_sharpness": float(cand.distance_sharpness),
                "confidence_floor": float(cand.confidence_floor),
            }
        )
        knn_cfg = make_knn_config(cfg, args, cand)
        row.update(
            {
                "mode": knn_cfg.mode,
                "scope": knn_cfg.scope,
                "bank_split": knn_cfg.bank_split,
                "bank_stride": int(knn_cfg.bank_stride),
                "feature_mode": knn_cfg.feature_mode,
                "template_mode": knn_cfg.template_mode,
                "distance_weight": knn_cfg.distance_weight,
                "anchor_mode": knn_cfg.anchor_mode,
                "shape_bins": int(knn_cfg.shape_bins),
                "diff_bins": int(knn_cfg.diff_bins),
                "pred_shape_bins": int(knn_cfg.pred_shape_bins),
                "pred_diff_bins": int(knn_cfg.pred_diff_bins),
            }
        )
        t0 = time.perf_counter()
        try:
            h_val = get_hybrid("val", knn_cfg)
            h_val.reset_confidence_stats()
            val = run_eval(
                split_name="val",
                loader=val_loader,
                eval_start=context.t_train,
                hybrid=h_val,
                **common_eval_kwargs,
            )
            row.update(
                {
                    "val_avg_mse": val["val_avg_mse"],
                    "val_avg_mae": val["val_avg_mae"],
                    "val_confidence": val.get("val_confidence", ""),
                    "val_effective_alpha": val.get("val_effective_alpha", ""),
                }
            )
            add_delta_fields(row, base_val, base_test)
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = repr(exc)
        row["eval_sec"] = time.perf_counter() - t0
        return row

    base_row = make_base_row(base_val, base_test)
    all_rows: list[dict[str, Any]] = [base_row]

    coarse_candidates = generate_candidates(
        k_values=parse_int_list(args.coarse_k),
        alpha_values=parse_float_list(args.coarse_alpha),
        adaptive_modes=parse_modes(args.coarse_adaptive),
        sharpness_values=parse_float_list(args.coarse_sharpness),
        confidence_floor_values=parse_float_list(args.coarse_confidence_floor),
    )
    print(f"Stage coarse candidates: {len(coarse_candidates)}")
    coarse_rows = []
    seen = {candidate_key(c) for c in coarse_candidates}
    for idx, cand in enumerate(coarse_candidates, start=1):
        print(f"[coarse {idx}/{len(coarse_candidates)}] {candidate_name(cand)}")
        row = evaluate_candidate(cand, "coarse")
        coarse_rows.append(row)
        all_rows.append(row)
        write_rows(all_csv, all_rows)

    coarse_sorted = [r for r in sort_rows(coarse_rows, args.metric) if r.get("status") == "ok"]
    for idx, row in enumerate(coarse_sorted, start=1):
        row["stage_rank"] = idx
    write_rows(coarse_csv, coarse_sorted)

    fine_seed_rows = coarse_sorted[: max(1, int(args.fine_top_n))]
    fine_candidates = fine_candidates_from_best(
        fine_seed_rows,
        args=args,
        coarse_modes=parse_modes(args.coarse_adaptive),
        seen=seen,
    )
    print(f"Stage fine candidates: {len(fine_candidates)}")
    fine_rows = []
    for idx, cand in enumerate(fine_candidates, start=1):
        print(f"[fine {idx}/{len(fine_candidates)}] {candidate_name(cand)}")
        row = evaluate_candidate(cand, "fine")
        fine_rows.append(row)
        all_rows.append(row)
        write_rows(all_csv, all_rows)

    fine_sorted = [r for r in sort_rows(fine_rows, args.metric) if r.get("status") == "ok"]
    for idx, row in enumerate(fine_sorted, start=1):
        row["stage_rank"] = idx
    write_rows(fine_csv, fine_sorted)

    ranked = [r for r in sort_rows([r for r in all_rows if r.get("status") == "ok"], args.metric)]
    for idx, row in enumerate(ranked, start=1):
        row["global_rank"] = idx

    top_for_test = [r for r in ranked if r.get("candidate") != "base"][: max(0, int(args.eval_test_top))]
    if base_test is not None and len(top_for_test) > 0:
        print(f"Evaluate test for top {len(top_for_test)} val candidates...")
        by_candidate = {row["candidate"]: row for row in all_rows}
        for row in top_for_test:
            cand = Candidate(
                k=int(float(row["k"])),
                alpha=float(row["alpha"]),
                adaptive_alpha=str(row["adaptive_alpha"]),
                distance_sharpness=float(row["distance_sharpness"] or 1.0),
                confidence_floor=float(row["confidence_floor"] or 0.0),
            )
            knn_cfg = make_knn_config(cfg, args, cand)
            h_test = get_hybrid("test", knn_cfg)
            h_test.reset_confidence_stats()
            test = run_eval(
                split_name="test",
                loader=test_loader,
                eval_start=context.t_val,
                hybrid=h_test,
                **common_eval_kwargs,
            )
            target = by_candidate[row["candidate"]]
            target.update(
                {
                    "test_avg_mse": test["test_avg_mse"],
                    "test_avg_mae": test["test_avg_mae"],
                    "test_confidence": test.get("test_confidence", ""),
                    "test_effective_alpha": test.get("test_effective_alpha", ""),
                }
            )
            add_delta_fields(target, base_val, base_test)
            write_rows(all_csv, all_rows)

    ranked = [r for r in sort_rows([r for r in all_rows if r.get("status") == "ok"], args.metric)]
    for idx, row in enumerate(ranked, start=1):
        row["global_rank"] = idx
    write_rows(all_csv, ranked + [r for r in all_rows if r.get("status") != "ok"])

    trace_rows = []
    if coarse_sorted:
        trace_rows.append(dict(coarse_sorted[0], stage="coarse_best"))
    if fine_sorted:
        trace_rows.append(dict(fine_sorted[0], stage="fine_best"))
    if ranked:
        trace_rows.append(dict(ranked[0], stage="overall_best"))
    write_rows(trace_csv, trace_rows)

    summary = {
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "ranking_metric": f"val_{args.metric}",
        "base_val": base_val,
        "base_test": base_test,
        "coarse_candidates": len(coarse_candidates),
        "fine_candidates": len(fine_candidates),
        "best_coarse": coarse_sorted[0] if coarse_sorted else None,
        "best_fine": fine_sorted[0] if fine_sorted else None,
        "best_overall": ranked[0] if ranked else None,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    write_markdown_report(
        out_dir / "summary.md",
        base_row=base_row,
        coarse_sorted=coarse_sorted,
        fine_sorted=fine_sorted,
        ranked=ranked,
        metric=str(args.metric),
    )
    plot_refinement_trace(
        out_dir / "refinement_trace.png",
        base_row=base_row,
        coarse_sorted=coarse_sorted,
        fine_sorted=fine_sorted,
    )
    plot_top_candidates(out_dir / "top_val_candidates.png", ranked, limit=12)
    heatmap_paths = plot_heatmaps(out_dir, ranked + [r for r in all_rows if r.get("status") == "ok"])

    print(f"Saved all results: {all_csv}")
    print(f"Saved coarse results: {coarse_csv}")
    print(f"Saved fine results: {fine_csv}")
    print(f"Saved trace: {trace_csv}")
    print(f"Saved markdown report: {out_dir / 'summary.md'}")
    print(f"Saved trace plot: {out_dir / 'refinement_trace.png'}")
    print(f"Saved top-candidate plot: {out_dir / 'top_val_candidates.png'}")
    for path in heatmap_paths:
        print(f"Saved heatmap: {path}")
    print("Top val-ranked rows:")
    for row in ranked[:10]:
        print(
            json.dumps(
                {
                    "rank": row.get("global_rank"),
                    "stage": row.get("stage"),
                    "candidate": row.get("candidate"),
                    "val_avg_mse": row.get("val_avg_mse"),
                    "val_avg_mae": row.get("val_avg_mae"),
                    "test_avg_mse": row.get("test_avg_mse", ""),
                    "test_avg_mae": row.get("test_avg_mae", ""),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
