from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.transfer import _df_to_tensor  # noqa: E402
from src.utils.cluster_memory import assign_channels_by_cycle_template, load_cluster_memory  # noqa: E402


ROUTES = [
    ("head_12_168", "head", 12, 168),
    ("tail_48_168", "tail", 48, 168),
]


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_target(cfg: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, list[str], dict[str, tuple[int, int]]]:
    raw_df = pd.read_csv(cfg["data"]["csv_path"])
    date_col = raw_df.columns[int(cfg["data"].get("date_col", 0))]
    data_tc, channels = _df_to_tensor(raw_df, date_col)
    data_tc = data_tc.to(device)
    T = int(data_tc.shape[0])
    t_train = int(T * float(cfg["data"]["train_ratio"]))
    t_val = int(T * (float(cfg["data"]["train_ratio"]) + float(cfg["data"]["val_ratio"])))
    train = data_tc[:t_train]
    mean_c = train.mean(dim=0, keepdim=True)
    std_c = train.std(dim=0, keepdim=True).clamp_min(1.0e-6)
    data_tc = (data_tc - mean_c) / std_c
    return data_tc, channels, {
        "train": (0, t_train),
        "val": (t_train, t_val),
        "test": (t_val, T),
        "train_val": (0, t_val),
    }


def route_rows(
    *,
    data_tc: torch.Tensor,
    channels: list[str],
    prototypes_kt: torch.Tensor,
    ranges: dict[str, tuple[int, int]],
    step_minutes: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for route_name, align, min_h, max_h in ROUTES:
        period_min = int(round(min_h * 60.0 / step_minutes))
        period_max = int(round(max_h * 60.0 / step_minutes))
        for split, (start, end) in ranges.items():
            segment = data_tc[start:end].contiguous()
            cluster_id_c, corr_ck, best_tau_ck = assign_channels_by_cycle_template(
                segment,
                prototypes_kt,
                phase_bins=64,
                period_min=period_min,
                period_max=period_max,
                align=align,
                phase_max_shift=None,
            )
            for c, channel in enumerate(channels):
                cid = int(cluster_id_c[c].item())
                rows.append(
                    {
                        "route": route_name,
                        "split": split,
                        "channel": channel,
                        "cluster_id": cid,
                        "corr_max": float(corr_ck[c, cid].item()),
                        "best_tau": int(best_tau_ck[c, cid].item()),
                    }
                )
    return rows


def metric_compare_rows() -> list[dict[str, Any]]:
    pairs = [
        (
            "head_12_168",
            "val",
            ROOT / "outputs" / "ettm1_to_ettm2_val_route_selection" / "val_runs" / "cycle_res_head_12_168" / "val_metrics.csv",
        ),
        (
            "tail_48_168",
            "val",
            ROOT / "outputs" / "ettm1_to_ettm2_val_route_selection" / "val_runs" / "cycle_res_tail_48_168" / "val_metrics.csv",
        ),
        ("head_12_168", "test", ROOT / "outputs" / "ETTm1ToETTm2" / "test_metrics.csv"),
        (
            "tail_48_168",
            "test",
            ROOT / "outputs" / "ettm1_to_ettm2_transfer_sweep" / "runs" / "cycle_res_tail_48_168" / "test_metrics.csv",
        ),
    ]
    frames = []
    for route, split, path in pairs:
        df = pd.read_csv(path)
        df["route"] = route
        df["split"] = split
        frames.append(df[["route", "split", "channel", "MSE", "MAE", "cluster_id"]])
    all_df = pd.concat(frames, ignore_index=True)
    rows: list[dict[str, Any]] = []
    for split in ["val", "test"]:
        wide = all_df[all_df["split"] == split].pivot(index="channel", columns="route", values="MSE")
        for channel, row in wide.iterrows():
            rows.append(
                {
                    "split": split,
                    "channel": channel,
                    "head_12_168_mse": float(row["head_12_168"]),
                    "tail_48_168_mse": float(row["tail_48_168"]),
                    "tail_minus_head_mse": float(row["tail_48_168"] - row["head_12_168"]),
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "ETTm1ToETTm2.yaml")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "ettm1_to_ettm2_route_shift_diagnostics")
    args = ap.parse_args()
    cfg = read_yaml(args.config)
    device = torch.device(str(cfg.get("exp", {}).get("device", "cuda:0")))
    memory = load_cluster_memory(str(cfg["source"]["memory_path"]), device=device)
    data_tc, channels, ranges = load_target(cfg, device)
    rows = route_rows(
        data_tc=data_tc,
        channels=channels,
        prototypes_kt=memory["prototypes_kt"].to(device),
        ranges=ranges,
        step_minutes=int(cfg.get("source", {}).get("step_minutes", 15)),
    )
    write_rows(args.out_dir / "segment_route_assignments.csv", rows)
    write_rows(args.out_dir / "route_metric_compare.csv", metric_compare_rows())
    print(f"Wrote diagnostics to {args.out_dir}")


if __name__ == "__main__":
    main()
