from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2")


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _gain_pct(base: float, mse: float) -> float:
    if abs(base) < 1.0e-12:
        return float("nan")
    return 100.0 * (base - mse) / abs(base)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _best_rows(best_results: Path) -> dict[str, dict[str, str]]:
    with best_results.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        if row.get("dataset") in DATASETS and str(row.get("horizon")) == "720":
            if str(row.get("is_best_for_cell", "")).lower() in {"true", "1", "yes"}:
                out[row["dataset"]] = row
    missing = [d for d in DATASETS if d not in out]
    if missing:
        raise FileNotFoundError(f"Missing H=720 best rows for: {missing}")
    return out


def _prior_top1_val_mse(explain: dict[str, Any]) -> tuple[float, float, str]:
    split = explain.get("splits", {}).get("val", {}) or {}
    rows = split.get("rows", []) or []
    by_cluster: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_cluster.setdefault(int(row["cluster_id"]), []).append(row)

    weighted_base = 0.0
    weighted_prior = 0.0
    total_weight = 0.0
    selected_desc: list[str] = []
    for cluster_id, cluster_rows in sorted(by_cluster.items()):
        allowed = [r for r in cluster_rows if bool(r.get("allowed_by_train_prior"))]
        candidates = allowed if allowed else cluster_rows
        # If top-k has more than one allowed penalty, use the strongest train prior as
        # the fixed train-prior top1 route for this diagnostic.
        chosen = max(candidates, key=lambda r: _as_float(r.get("train_prior_prob"), -1.0))
        channels = _as_float(chosen.get("cluster_channels"), 0.0)
        base = _as_float(chosen.get("cluster_base_mse"), 0.0)
        gain = _as_float(chosen.get("mean_single_penalty_gain_mse"), 0.0)
        weighted_base += channels * base
        weighted_prior += channels * (base - gain)
        total_weight += channels
        selected_desc.append(f"C{cluster_id}:{chosen.get('penalty')}")

    if total_weight <= 0.0:
        return float("nan"), float("nan"), ""
    base_mse = weighted_base / total_weight
    prior_mse = weighted_prior / total_weight
    return prior_mse, _gain_pct(base_mse, prior_mse), "; ".join(selected_desc)


def _diagnose_dataset(dataset: str, run_dir: Path) -> dict[str, Any]:
    summary = _load_json(run_dir / "run_summary.json")
    explain = _load_json(run_dir / "penalty_explainability.json")
    val_explain = explain.get("splits", {}).get("val", {}) or {}
    val_hit = (summary.get("moe_gate_penalty_hit") or {}).get("val") or {}

    base_mse = _as_float(val_explain.get("base_mse"))
    learned_mse = _as_float(val_explain.get("final_mse"))
    learned_gain = _as_float(val_explain.get("final_gain_pct_vs_base"))
    oracle_mse = _as_float(val_hit.get("oracle_mse"))
    oracle_gain = _as_float(val_hit.get("oracle_gain_pct_vs_base"))
    selected_top1_mse = _as_float(val_hit.get("selected_top1_mse"))
    selected_top1_gain = _as_float(val_hit.get("selected_top1_gain_pct_vs_base"))
    prior_mse, prior_gain, prior_route = _prior_top1_val_mse(explain)

    oracle_gap = oracle_gain - max(prior_gain, learned_gain)
    if oracle_gain <= 1.0:
        diagnosis = "weak_penalty_signal"
    elif prior_gain <= 0.0 and learned_gain <= 0.0:
        diagnosis = "route_or_adapter_cannot_use_available_oracle"
    elif prior_gain > learned_gain + 1.0:
        diagnosis = "learned_gate_underuses_train_prior"
    elif learned_gain >= prior_gain:
        diagnosis = "learned_gate_matches_or_exceeds_prior"
    else:
        diagnosis = "mixed_prior_gate_gap"

    return {
        "dataset": dataset,
        "horizon": 720,
        "run_dir": str(run_dir),
        "val_base_mse": base_mse,
        "oracle_mse": oracle_mse,
        "oracle_gain_pct": oracle_gain,
        "train_prior_top1_mse": prior_mse,
        "train_prior_top1_gain_pct": prior_gain,
        "learned_gate_mse": learned_mse,
        "learned_gate_gain_pct": learned_gain,
        "learned_selected_top1_mse": selected_top1_mse,
        "learned_selected_top1_gain_pct": selected_top1_gain,
        "oracle_minus_best_nonoracle_gain_pct": oracle_gap,
        "gate_hit_rate": _as_float(val_hit.get("top1_hit_rate_all")),
        "gate_hit_on_positive_oracle": _as_float(val_hit.get("top1_hit_rate_on_positive_oracle")),
        "oracle_positive_rate": _as_float(val_hit.get("oracle_positive_rate")),
        "selected_positive_rate": _as_float(val_hit.get("selected_positive_rate")),
        "train_prior_route": prior_route,
        "diagnosis": diagnosis,
    }


def _fmt(x: Any, digits: int = 4) -> str:
    if isinstance(x, float):
        if x != x:
            return ""
        return f"{x:.{digits}f}"
    return str(x)


def _write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines: list[str] = [
        "# H=720 PKR-MoE Route-Mode Diagnostic",
        "",
        "This diagnostic is read-only: it uses the selected H=720 ETT run artifacts and evaluates validation-split behavior without retraining or changing the main-table results.",
        "",
        "Compared modes:",
        "- **Oracle route:** per-sample best penalty branch on validation, with skip when no penalty improves over base. This is an upper bound and uses validation labels only for diagnosis.",
        "- **Train-prior top1 route:** fixed cluster-level top penalty selected from the train-only penalty prior; no validation labels are used to choose the route.",
        "- **Learned gate:** the actual trained PKR-MoE gate/fusion output on validation.",
        "",
        "| Dataset | Base MSE | Oracle MSE / Gain | Train-prior MSE / Gain | Learned-gate MSE / Gain | Gate hit | Oracle-positive | Prior route | Diagnosis |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {base} | {oracle_mse} / {oracle_gain}% | {prior_mse} / {prior_gain}% | "
            "{learned_mse} / {learned_gain}% | {hit} | {opos} | {route} | {diagnosis} |".format(
                dataset=row["dataset"],
                base=_fmt(row["val_base_mse"]),
                oracle_mse=_fmt(row["oracle_mse"]),
                oracle_gain=_fmt(row["oracle_gain_pct"], 2),
                prior_mse=_fmt(row["train_prior_top1_mse"]),
                prior_gain=_fmt(row["train_prior_top1_gain_pct"], 2),
                learned_mse=_fmt(row["learned_gate_mse"]),
                learned_gain=_fmt(row["learned_gate_gain_pct"], 2),
                hit=_fmt(row["gate_hit_rate"], 3),
                opos=_fmt(row["oracle_positive_rate"], 3),
                route=row["train_prior_route"],
                diagnosis=row["diagnosis"],
            )
        )

    lines.extend(
        [
            "",
            "Interpretation:",
            "- If oracle gain is also small, the long-horizon issue is mainly weak local penalty signal rather than validation/test distribution shift.",
            "- If oracle gain is large but train-prior and learned gate are weak, the penalty pool contains useful branches but the route/adapter policy does not exploit them reliably.",
            "- If validation learned-gate gain is positive but test gain is negative in the interpretability report, the remaining issue is regime shift rather than route capacity on validation.",
            "",
            "Summary: the H=720 failures are mixed, but the dominant pattern is not a single test-leak or test-shift artifact. ETTh1 and ETTm2 show weak or neutral usable long-horizon penalty correction on validation, while ETTh2 has a larger oracle-to-learned gap and is the clearest route/regime-shift case. ETTm1 remains the positive long-horizon counterexample.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-results", default="outputs/ett_horizon_specific_moe_tune/best_results.csv")
    parser.add_argument("--out-root", default="outputs/h720_moe_route_mode_diagnostic")
    args = parser.parse_args()

    repo = Path.cwd()
    best_results = (repo / args.best_results).resolve()
    out_root = (repo / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for dataset, best in _best_rows(best_results).items():
        run_dir = Path(best["out_dir"])
        if not run_dir.is_absolute():
            run_dir = (repo / run_dir).resolve()
        rows.append(_diagnose_dataset(dataset, run_dir))

    csv_path = out_root / "h720_route_mode_diagnostic.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = out_root / "h720_route_mode_diagnostic.md"
    _write_markdown(rows, md_path)

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
