from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

PENALTY_ORDER = [
    "amp_under",
    "level",
    "delta",
    "direction",
    "corr",
    "range",
    "trend",
    "seasonal_align",
    "jump",
]


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in ("", None):
        return float(default)
    return float(value)


def add_once(items: list[str], name: str) -> None:
    if name not in items:
        items.append(name)


def select_penalties_for_cluster(
    row: dict[str, Any],
    *,
    max_penalties: int = 6,
    allow_jump: bool = False,
) -> tuple[list[str], list[str]]:
    std_ratio = as_float(row, "pred_to_y_std_ratio", 1.0)
    corr = as_float(row, "corr_pred_y", 1.0)
    mae = max(as_float(row, "mae", 0.0), 1.0e-8)
    bias = abs(as_float(row, "bias_mean_y_minus_pred", 0.0))
    median_bias = abs(as_float(row, "bias_median_y_minus_pred", 0.0))
    horizon_bias = as_float(row, "horizon_bias_abs_mean", 0.0)
    trend_slope = abs(as_float(row, "trend_slope_residual", 0.0))
    spike_miss = max(0.0, as_float(row, "spike_miss_mean", 0.0))
    spike_abs = as_float(row, "spike_abs_error", 0.0)

    selected: list[str] = []
    gap_tags: list[str] = []
    bias_level = max(bias, median_bias, horizon_bias)
    spike_heavy = spike_abs >= max(0.75, mae * 2.0) or spike_miss >= max(0.50, mae * 1.5)
    severe_event_collapse = std_ratio < 0.65 and spike_abs >= max(1.50, mae * 5.0)

    if (
        corr >= 0.90
        and 0.85 <= std_ratio <= 1.15
        and bias_level <= 0.05
        and spike_abs <= max(0.25, mae * 1.25)
    ):
        return ["corr"], ["well_learned"]

    if bias >= 0.05 or median_bias >= 0.05:
        gap_tags.append("level_bias")
        add_once(selected, "level")

    if horizon_bias >= 0.03:
        gap_tags.append("horizon_bias")
        add_once(selected, "seasonal_align")
        add_once(selected, "level")

    if trend_slope >= 0.001:
        gap_tags.append("trend_drift")
        add_once(selected, "trend")

    if std_ratio < 0.85:
        gap_tags.append("variance_compression")
        add_once(selected, "amp_under")
        if std_ratio < 0.75:
            add_once(selected, "range")
    elif std_ratio > 1.20:
        gap_tags.append("variance_overfit")
        add_once(selected, "range")

    if corr < 0.70:
        gap_tags.append("event_dominated_dynamic_gap" if severe_event_collapse else "dynamic_mismatch")
        if not severe_event_collapse:
            add_once(selected, "delta")
            add_once(selected, "direction")
        if std_ratio >= 0.85:
            add_once(selected, "corr")
    elif corr < 0.88 and std_ratio >= 0.80:
        gap_tags.append("correlation_residual")
        add_once(selected, "corr")

    if spike_heavy:
        gap_tags.append("event_miss")
        add_once(selected, "amp_under")
        add_once(selected, "range")
        add_once(selected, "seasonal_align")
        if allow_jump:
            add_once(selected, "jump")

    if not selected:
        gap_tags.append("weak_residual")
        add_once(selected, "corr")

    selected = [name for name in selected if name in PENALTY_ORDER]
    selected.sort(key=PENALTY_ORDER.index)
    return selected[: int(max_penalties)], gap_tags


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build_profile_payload(
    *,
    cluster_rows: list[dict[str, Any]],
    channel_rows: list[dict[str, Any]],
    config: Path,
    checkpoint: Path,
    cluster_diagnostic_csv: Path,
    channel_diagnostic_csv: Path,
    split: str,
    max_penalties: int,
    allow_jump: bool,
) -> dict[str, Any]:
    if split not in {"train", "val"}:
        raise ValueError("Only train/val diagnostics can be used for profile selection.")

    fixed_channels = sorted(channel_rows, key=lambda row: int(row["channel_idx"]))
    fixed_cluster_id = [int(row["cluster_id"]) for row in fixed_channels]
    max_cluster = max(int(row["cluster_id"]) for row in cluster_rows)
    allowed_by_cluster: list[list[str]] = [["corr"] for _ in range(max_cluster + 1)]
    gap_summary: list[dict[str, Any]] = []

    for row in sorted(cluster_rows, key=lambda item: int(item["cluster_id"])):
        cluster_id = int(row["cluster_id"])
        selected, gap_tags = select_penalties_for_cluster(
            row,
            max_penalties=max_penalties,
            allow_jump=allow_jump,
        )
        allowed_by_cluster[cluster_id] = selected
        gap_summary.append(
            {
                "cluster_id": cluster_id,
                "channels": int(float(row.get("channels", 0) or 0)),
                "gap_tags": gap_tags,
                "recommended_penalties": selected,
                "mse": as_float(row, "mse", 0.0),
                "mae": as_float(row, "mae", 0.0),
                "pred_to_y_std_ratio": as_float(row, "pred_to_y_std_ratio", 1.0),
                "corr_pred_y": as_float(row, "corr_pred_y", 1.0),
                "horizon_bias_abs_mean": as_float(row, "horizon_bias_abs_mean", 0.0),
                "spike_miss_mean": as_float(row, "spike_miss_mean", 0.0),
                "channels_list": row.get("channels_list", ""),
            }
        )

    penalties_enabled = [
        name
        for name in PENALTY_ORDER
        if name != "jump" or allow_jump
        if any(name in allowed for allowed in allowed_by_cluster)
    ]
    return {
        "test_used": False,
        "selection_split": split,
        "scale_split": "train",
        "selection_target": "backbone_gap_penalty_modes",
        "config": str(config),
        "checkpoint": str(checkpoint),
        "cluster_source": "checkpoint",
        "diagnostic_csv": str(cluster_diagnostic_csv),
        "channel_diagnostic_csv": str(channel_diagnostic_csv),
        "n_clusters": int(max_cluster + 1),
        "penalty_names_considered": PENALTY_ORDER,
        "penalties_enabled": penalties_enabled,
        "allowed_by_cluster": allowed_by_cluster,
        "fixed_cluster_id": fixed_cluster_id,
        "backbone_gap_summary": gap_summary,
        "notes": (
            "Penalty pools are selected from backbone residual gap diagnostics on train/val. "
            "This file does not use test labels or test predictions."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a penalty-pool JSON from backbone gap diagnostics.")
    parser.add_argument("--cluster-diagnostic-csv", required=True)
    parser.add_argument("--channel-diagnostic-csv", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--max-penalties", type=int, default=6)
    parser.add_argument("--allow-jump", action="store_true")
    args = parser.parse_args()

    cluster_csv = resolve(args.cluster_diagnostic_csv)
    channel_csv = resolve(args.channel_diagnostic_csv)
    payload = build_profile_payload(
        cluster_rows=read_csv_rows(cluster_csv),
        channel_rows=read_csv_rows(channel_csv),
        config=resolve(args.config),
        checkpoint=resolve(args.checkpoint),
        cluster_diagnostic_csv=cluster_csv,
        channel_diagnostic_csv=channel_csv,
        split=str(args.split),
        max_penalties=int(args.max_penalties),
        allow_jump=bool(args.allow_jump),
    )
    out_json = resolve(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in payload["backbone_gap_summary"]:
        print(
            f"cluster={item['cluster_id']} "
            f"tags={';'.join(item['gap_tags'])} "
            f"penalties={';'.join(item['recommended_penalties'])}"
        )
    print(f"backbone_gap_profile_json={out_json}")


if __name__ == "__main__":
    main()
