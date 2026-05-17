from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_ettm1_to_ettm2_input_fallback_selection import static_route_from_train  # noqa: E402
from scripts.run_ettm1_to_ettm2_input_route_selection import load_context, read_yaml  # noqa: E402
from scripts.run_ettm1_to_ettm2_portrait_route_search import (  # noqa: E402
    channel_to_cluster_distance,
    load_source_train_ct,
    name_distance,
    portraits,
    standardized_pair_distance,
)
from scripts.run_ettm1_to_ettm2_val_loss_route_selection import evaluate_route  # noqa: E402
from src.utils.cluster_memory import assign_channels_by_cycle_template  # noqa: E402


HORIZONS = [192, 336]
RESULT_FIELDS = [
    "horizon",
    "candidate",
    "route",
    "route_source",
    "uses_val_input",
    "changed_channels_vs_static",
    "val_mse",
    "val_mae",
    "test_mse",
    "test_mae",
    "target_self_mse",
    "source_test_mse",
]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in RESULT_FIELDS})


def route_key(route: tuple[int, ...]) -> str:
    return json.dumps([int(v) for v in route])


def _range_ct(ctx: dict[str, Any], name: str) -> torch.Tensor:
    data_tc = ctx["data_tc"]
    t_train = int(ctx["t_train"])
    t_val = int(ctx["t_val"])
    if name == "train":
        seg = data_tc[:t_train]
    elif name == "val_input":
        seg = data_tc[t_train:t_val]
    elif name == "train_val":
        seg = data_tc[:t_val]
    else:
        raise ValueError(f"Unknown split portrait range: {name}")
    return seg.T.contiguous()


def corr_route_for_range(ctx: dict[str, Any], cfg: dict[str, Any], name: str) -> tuple[int, ...]:
    transfer_cfg = cfg.get("transfer", {}) or {}
    data_tc = ctx["data_tc"]
    t_train = int(ctx["t_train"])
    t_val = int(ctx["t_val"])
    if name == "train":
        route_data = data_tc[:t_train]
    elif name == "val_input":
        route_data = data_tc[t_train:t_val]
    elif name == "train_val":
        route_data = data_tc[:t_val]
    else:
        raise ValueError(f"Unknown corr range: {name}")
    step_min = 15.0
    period_min = transfer_cfg.get("period_min", None)
    period_max = transfer_cfg.get("period_max", None)
    if transfer_cfg.get("period_min_hours", None) is not None:
        period_min = int(round(float(transfer_cfg["period_min_hours"]) * 60.0 / step_min))
    if transfer_cfg.get("period_max_hours", None) is not None:
        period_max = int(round(float(transfer_cfg["period_max_hours"]) * 60.0 / step_min))
    _, corr_ck, _ = assign_channels_by_cycle_template(
        route_data,
        ctx["prototypes_kt"],
        phase_bins=int(transfer_cfg.get("phase_bins", 64)),
        period_min=period_min,
        period_max=period_max,
        align=str(transfer_cfg.get("corr_align", "head")),
        phase_max_shift=transfer_cfg.get("phase_max_shift", None),
    )
    return tuple(int(v) for v in torch.argmax(corr_ck, dim=1).detach().cpu().tolist())


def _portrait_distances(
    ctx: dict[str, Any],
    target_range: str,
) -> dict[str, torch.Tensor]:
    target_p = portraits(_range_ct(ctx, target_range))
    source_p = portraits(ctx["prototypes_kt"].contiguous())
    source_train_ct = load_source_train_ct(int(ctx["pred_len"]), ctx["data_tc"].device)
    source_ch_p = portraits(source_train_ct)
    source_cluster_c = ctx["meta"]["cluster_id_c"].to(device=ctx["data_tc"].device, dtype=torch.long)
    out = {
        "stat": standardized_pair_distance(target_p["stat"], source_p["stat"]),
        "dyn": standardized_pair_distance(target_p["dyn"], source_p["dyn"]),
        "cycle": standardized_pair_distance(target_p["cycle"], source_p["cycle"]),
        "ch_min_stat": channel_to_cluster_distance(
            target_p["stat"], source_ch_p["stat"], source_cluster_c, num_clusters=int(ctx["K"]), reduce="min"
        ),
        "ch_min_dyn": channel_to_cluster_distance(
            target_p["dyn"], source_ch_p["dyn"], source_cluster_c, num_clusters=int(ctx["K"]), reduce="min"
        ),
        "ch_min_cycle": channel_to_cluster_distance(
            target_p["cycle"], source_ch_p["cycle"], source_cluster_c, num_clusters=int(ctx["K"]), reduce="min"
        ),
        "name": name_distance(ctx),
    }
    return out


def route_from_portrait(
    ctx: dict[str, Any],
    cfg: dict[str, Any],
    *,
    target_range: str,
    weights: tuple[float, float, float, float, float],
    channel_level: bool,
) -> tuple[int, ...]:
    cw, sw, dw, cyw, nw = weights
    d = _portrait_distances(ctx, target_range)
    corr = torch.zeros_like(d["stat"])
    if cw:
        transfer_cfg = cfg.get("transfer", {}) or {}
        data_tc = ctx["data_tc"]
        t_train = int(ctx["t_train"])
        t_val = int(ctx["t_val"])
        if target_range == "train":
            route_data = data_tc[:t_train]
        elif target_range == "val_input":
            route_data = data_tc[t_train:t_val]
        else:
            route_data = data_tc[:t_val]
        step_min = 15.0
        period_min = transfer_cfg.get("period_min", None)
        period_max = transfer_cfg.get("period_max", None)
        if transfer_cfg.get("period_min_hours", None) is not None:
            period_min = int(round(float(transfer_cfg["period_min_hours"]) * 60.0 / step_min))
        if transfer_cfg.get("period_max_hours", None) is not None:
            period_max = int(round(float(transfer_cfg["period_max_hours"]) * 60.0 / step_min))
        _, corr, _ = assign_channels_by_cycle_template(
            route_data,
            ctx["prototypes_kt"],
            phase_bins=int(transfer_cfg.get("phase_bins", 64)),
            period_min=period_min,
            period_max=period_max,
            align=str(transfer_cfg.get("corr_align", "head")),
            phase_max_shift=transfer_cfg.get("phase_max_shift", None),
        )

    stat = d["ch_min_stat"] if channel_level else d["stat"]
    dyn = d["ch_min_dyn"] if channel_level else d["dyn"]
    cycle = d["ch_min_cycle"] if channel_level else d["cycle"]
    score = float(cw) * corr - float(sw) * stat - float(dw) * dyn - float(cyw) * cycle - float(nw) * d["name"]
    return tuple(int(v) for v in torch.argmax(score, dim=1).detach().cpu().tolist())


def stable_replace(
    base: tuple[int, ...],
    first: tuple[int, ...],
    second: tuple[int, ...],
    third: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    out = []
    for i, b in enumerate(base):
        vals = [first[i], second[i]]
        if third is not None:
            vals.append(third[i])
        counts = Counter(vals)
        winner, count = counts.most_common(1)[0]
        if count >= 2:
            out.append(int(winner))
        elif first[i] == second[i]:
            out.append(int(first[i]))
        else:
            out.append(int(b))
    return tuple(out)


def majority_vote(base: tuple[int, ...], routes: list[tuple[int, ...]]) -> tuple[int, ...]:
    out = []
    for i, b in enumerate(base):
        counts = Counter(route[i] for route in routes)
        winner, count = counts.most_common(1)[0]
        tied = sum(1 for _, cur_count in counts.items() if cur_count == count) > 1
        out.append(int(b if tied else winner))
    return tuple(out)


def add_candidate(
    rows: list[dict[str, Any]],
    seen: set[tuple[int, ...]],
    *,
    horizon: int,
    name: str,
    route: tuple[int, ...],
    route_source: str,
    uses_val_input: bool,
    base_route: tuple[int, ...],
    ctx: dict[str, Any],
    target_self_mse: float,
    source_test_mse: float,
    batch_size: int,
) -> None:
    if route in seen:
        return
    seen.add(route)
    val = evaluate_route(ctx, route, split="val", batch_size=batch_size)
    test = evaluate_route(ctx, route, split="test", batch_size=batch_size)
    rows.append(
        {
            "horizon": horizon,
            "candidate": name,
            "route": route_key(route),
            "route_source": route_source,
            "uses_val_input": uses_val_input,
            "changed_channels_vs_static": sum(1 for a, b in zip(route, base_route) if a != b),
            "val_mse": val["avg_mse"],
            "val_mae": val["avg_mae"],
            "test_mse": test["avg_mse"],
            "test_mae": test["avg_mae"],
            "target_self_mse": target_self_mse,
            "source_test_mse": source_test_mse,
        }
    )


def source_target_mse(horizon: int) -> tuple[float, float]:
    with (ROOT / "outputs" / "ett_global_h96_param_base" / "runs" / "ETTm1" / f"pred_{horizon}" / "run_summary.json").open(
        "r", encoding="utf-8"
    ) as f:
        src = json.load(f)
    with (ROOT / "outputs" / "ett_horizon_sweep" / "runs" / "ETTm2" / f"pred_{horizon}" / "run_summary.json").open(
        "r", encoding="utf-8"
    ) as f:
        tgt = json.load(f)
    return float(src["test"]["avg_mse"]), float(tgt["test"]["avg_mse"])


def write_summary(out_root: Path, rows: list[dict[str, Any]]) -> None:
    summary = []
    for horizon in sorted({int(r["horizon"]) for r in rows}):
        cur = [r for r in rows if int(r["horizon"]) == horizon]
        by_val = min(cur, key=lambda r: (float(r["val_mse"]), float(r["val_mae"])))
        by_test = min(cur, key=lambda r: (float(r["test_mse"]), float(r["test_mae"])))
        static = next(r for r in cur if r["candidate"] == "static_corr_train")
        summary.append(
            {
                "horizon": horizon,
                "static_corr_train": static,
                "selected_by_val": by_val,
                "best_test_diagnostic": by_test,
                "val_test_gap_selected_by_val": float(by_val["test_mse"]) - float(by_val["val_mse"]),
            }
        )
    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def try_write_plot(out_root: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    horizons = sorted({int(r["horizon"]) for r in rows})
    fig, axes = plt.subplots(1, len(horizons), figsize=(6.0 * len(horizons), 4.8), squeeze=False)
    for ax, horizon in zip(axes[0], horizons):
        cur = [r for r in rows if int(r["horizon"]) == horizon]
        xs = [float(r["val_mse"]) for r in cur]
        ys = [float(r["test_mse"]) for r in cur]
        colors = ["#1f77b4" if not bool(r["uses_val_input"]) else "#ff7f0e" for r in cur]
        ax.scatter(xs, ys, c=colors, s=55, alpha=0.85)
        for r, x, y in zip(cur, xs, ys):
            label = str(r["candidate"]).replace("_", "\n")
            ax.annotate(label, (x, y), fontsize=7, xytext=(4, 3), textcoords="offset points")
        target = float(cur[0]["target_self_mse"])
        ax.axhline(target, color="#555555", linestyle="--", linewidth=1.0, label="target self test")
        ax.set_title(f"ETTm1 -> ETTm2 H{horizon}")
        ax.set_xlabel("route validation MSE")
        ax.set_ylabel("transfer test MSE")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_root / "val_vs_test_route_candidates.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "ettm1_to_ettm2_stable_portrait_route_search")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        base_cfg_path = ROOT / "outputs" / "ettm1_other_ett_horizon_transfer" / "ETTm1_to_ETTm2" / f"pred_{horizon}" / "base_config.yaml"
        cfg = read_yaml(base_cfg_path)
        ctx = load_context(cfg, device)
        source_test_mse, target_self_mse = source_target_mse(horizon)

        base = tuple(int(v) for v in static_route_from_train(ctx, cfg).detach().cpu().tolist())
        corr_train = corr_route_for_range(ctx, cfg, "train")
        corr_val = corr_route_for_range(ctx, cfg, "val_input")
        corr_train_val = corr_route_for_range(ctx, cfg, "train_val")

        routes: dict[str, tuple[tuple[int, ...], str, bool]] = {
            "static_corr_train": (base, "corr:train", False),
            "corr_val_input": (corr_val, "corr:val_input", True),
            "corr_train_val": (corr_train_val, "corr:train_val", True),
            "corr_stable": (stable_replace(base, corr_train, corr_val, corr_train_val), "corr:stable_vote", True),
        }

        presets = {
            "portrait_all": ((0.0, 0.5, 0.5, 0.5, 0.0), False),
            "hybrid_light": ((1.0, 0.1, 0.1, 0.1, 0.0), False),
            "hybrid_channel_min_light": ((1.0, 0.1, 0.1, 0.1, 0.0), True),
            "hybrid_channel_min_med": ((1.0, 0.25, 0.25, 0.25, 0.0), True),
        }
        for name, (weights, channel_level) in presets.items():
            train_route = route_from_portrait(ctx, cfg, target_range="train", weights=weights, channel_level=channel_level)
            val_route = route_from_portrait(ctx, cfg, target_range="val_input", weights=weights, channel_level=channel_level)
            train_val_route = route_from_portrait(ctx, cfg, target_range="train_val", weights=weights, channel_level=channel_level)
            routes[f"{name}_train"] = (train_route, f"{name}:train", False)
            routes[f"{name}_val_input"] = (val_route, f"{name}:val_input", True)
            routes[f"{name}_train_val"] = (train_val_route, f"{name}:train_val", True)
            routes[f"{name}_stable"] = (
                stable_replace(base, train_route, val_route, train_val_route),
                f"{name}:stable_vote",
                True,
            )
            routes[f"{name}_majority_with_corr"] = (
                majority_vote(base, [base, corr_train_val, train_route, val_route, train_val_route]),
                f"{name}:majority_with_corr",
                True,
            )

        seen: set[tuple[int, ...]] = set()
        for name, (route, route_source, uses_val_input) in routes.items():
            add_candidate(
                rows,
                seen,
                horizon=horizon,
                name=name,
                route=route,
                route_source=route_source,
                uses_val_input=uses_val_input,
                base_route=base,
                ctx=ctx,
                target_self_mse=target_self_mse,
                source_test_mse=source_test_mse,
                batch_size=args.batch_size,
            )
            write_rows(args.out_root / "stable_portrait_route_results.csv", rows)

    write_summary(args.out_root, rows)
    try_write_plot(args.out_root, rows)
    print(args.out_root / "stable_portrait_route_results.csv")
    print(args.out_root / "summary.json")


if __name__ == "__main__":
    main()
