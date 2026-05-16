from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_ettm1_to_ettm2_input_route_selection import (  # noqa: E402
    input_route_scores,
    load_context,
    read_yaml,
)
from src.transfer import _predict_with_optional_residual  # noqa: E402
from src.utils.cluster_memory import assign_channels_by_corr, assign_channels_by_cycle_template  # noqa: E402
from src.utils.metrics import accumulate_channel_errors, mse_mae_from_sums  # noqa: E402


FIELDS = [
    "status",
    "name",
    "eval_split",
    "mse",
    "mae",
    "align",
    "score_len",
    "max_lag",
    "margin_threshold",
    "delta_threshold",
    "min_top_score",
    "switch_rate",
    "route_counts",
    "out_dir",
    "error",
]


def write_fallback_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})


def static_route_from_train(ctx: dict[str, Any], cfg: dict[str, Any]) -> torch.Tensor:
    transfer = cfg.get("transfer", {})
    data_tc = ctx["data_tc"]
    route_data = data_tc[: int(ctx["t_train"])].contiguous()
    corr_mode = str(transfer.get("corr_mode", "cycle_template")).lower()
    align = str(transfer.get("corr_align", "head")).lower()
    prototypes = ctx["prototypes_kt"]
    if corr_mode in {"cycle", "cycle_template", "phase", "phase_template"}:
        period_min = transfer.get("period_min", None)
        period_max = transfer.get("period_max", None)
        period_min_h = transfer.get("period_min_hours", None)
        period_max_h = transfer.get("period_max_hours", None)
        if period_min is not None:
            period_min = int(period_min)
        if period_max is not None:
            period_max = int(period_max)
        if period_min_h is not None or period_max_h is not None:
            # ETTm data is 15 minutes; keep this robust if date parsing is available in future cfg changes.
            step_min = float(cfg.get("source", {}).get("step_minutes", 15) or 15)
            if period_min_h is not None:
                period_min = int(round(float(period_min_h) * 60.0 / step_min))
            if period_max_h is not None:
                period_max = int(round(float(period_max_h) * 60.0 / step_min))
        cluster_id_c, _, _ = assign_channels_by_cycle_template(
            route_data,
            prototypes,
            phase_bins=int(transfer.get("phase_bins", 64)),
            period_min=period_min,
            period_max=period_max,
            align=align,
            phase_max_shift=transfer.get("phase_max_shift", None),
        )
        return cluster_id_c
    cluster_id_c, _ = assign_channels_by_corr(
        route_data,
        prototypes,
        align=align,
        max_lag=int(transfer.get("corr_max_lag", 0)),
    )
    return cluster_id_c


def make_variants() -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = [
        {
            "name": "static_only",
            "align": "tail",
            "score_len": 96,
            "max_lag": 24,
            "margin_threshold": None,
            "delta_threshold": None,
            "min_top_score": None,
        }
    ]
    route_specs = [
        ("tail", 96, 24),
        ("tail", 192, 96),
        ("tail", 192, 24),
    ]
    for align, score_len, max_lag in route_specs:
        for margin in [0.05, 0.10, 0.20, 0.30]:
            for delta in [0.05, 0.10, 0.15, 0.20]:
                variants.append(
                    {
                        "name": f"fb_{align}_l{score_len}_lag{max_lag}_m{str(margin).replace('.', 'p')}_d{str(delta).replace('.', 'p')}",
                        "align": align,
                        "score_len": score_len,
                        "max_lag": max_lag,
                        "margin_threshold": margin,
                        "delta_threshold": delta,
                        "min_top_score": None,
                    }
                )
    return variants


def predict_by_route_patterns(
    ctx: dict[str, Any],
    x: torch.Tensor,
    route_bc: torch.Tensor,
) -> torch.Tensor:
    B, C, _ = x.shape
    yhat = x.new_empty((B, C, int(ctx["pred_len"])))
    unique_routes, inverse = torch.unique(route_bc.to(torch.long), dim=0, return_inverse=True)
    for pattern_idx, cluster_id_c in enumerate(unique_routes):
        batch_idx = (inverse == pattern_idx).nonzero(as_tuple=False).view(-1)
        if batch_idx.numel() == 0:
            continue
        _, yhat_part = _predict_with_optional_residual(
            model=ctx["model"],
            gate=ctx["gate"],
            pred_residual=ctx["pred_residual"],
            x=x.index_select(0, batch_idx),
            cluster_id_c=cluster_id_c.to(device=x.device, dtype=torch.long),
            meta=ctx["meta"],
            residual_scale_c=ctx["residual_scale_c"],
        )
        yhat.index_copy_(0, batch_idx, yhat_part)
    return yhat


def evaluate_fallback(
    ctx: dict[str, Any],
    static_cluster_id_c: torch.Tensor,
    variant: dict[str, Any],
    eval_split: str,
    out_dir: Path,
    batch_size: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    L = int(ctx["input_len"])
    H = int(ctx["pred_len"])
    C = int(ctx["C"])
    K = int(ctx["K"])
    data_tc = ctx["data_tc"]
    eval_start, eval_end = (ctx["t_train"], ctx["t_val"]) if eval_split == "val" else (ctx["t_val"], ctx["T"])
    eval_seg = data_tc[eval_start:eval_end]
    n_windows = int(eval_seg.shape[0] - L - H + 1)
    if n_windows <= 0:
        raise ValueError(f"No {eval_split} windows available.")

    se_c = torch.zeros(C, device=data_tc.device)
    ae_c = torch.zeros(C, device=data_tc.device)
    route_count = torch.zeros(K, device=data_tc.device)
    switch_count = 0
    total_routes = 0
    static_bc_template = static_cluster_id_c.to(device=data_tc.device, dtype=torch.long).view(1, C)
    denom = 0
    with torch.no_grad():
        for start in range(0, n_windows, batch_size):
            end = min(start + batch_size, n_windows)
            xs = []
            ys = []
            for i in range(start, end):
                win = eval_seg[i : i + L + H]
                xs.append(win[:L].T)
                ys.append(win[L:].T)
            x = torch.stack(xs, dim=0)
            y = torch.stack(ys, dim=0)
            bsz = int(x.shape[0])
            route_bc = static_bc_template.expand(bsz, C).clone()
            if variant["name"] != "static_only":
                scores = input_route_scores(
                    x,
                    ctx["prototypes_kt"],
                    align=str(variant["align"]),
                    score_len=int(variant["score_len"]),
                    max_lag=int(variant["max_lag"]),
                )
                top2 = torch.topk(scores, k=min(2, K), dim=-1)
                top_score = top2.values[..., 0]
                second_score = top2.values[..., 1] if top2.values.shape[-1] > 1 else top2.values[..., 0]
                top_route = top2.indices[..., 0]
                static_score = scores.gather(2, route_bc.unsqueeze(-1)).squeeze(-1)
                margin = top_score - second_score
                delta = top_score - static_score
                switch = (
                    (top_route != route_bc)
                    & (margin >= float(variant["margin_threshold"]))
                    & (delta >= float(variant["delta_threshold"]))
                )
                if variant.get("min_top_score") is not None:
                    switch = switch & (top_score >= float(variant["min_top_score"]))
                route_bc = torch.where(switch, top_route, route_bc)
                switch_count += int(switch.sum().item())
            total_routes += bsz * C
            route_count += torch.bincount(route_bc.reshape(-1), minlength=K).to(route_count)
            yhat = predict_by_route_patterns(ctx, x, route_bc)
            accumulate_channel_errors(se_c, ae_c, yhat, y)
            denom += int(x.shape[0] * H)
    mse_c, mae_c = mse_mae_from_sums(se_c, ae_c, denom)
    avg_mse = float(mse_c.mean().item())
    avg_mae = float(mae_c.mean().item())
    summary = {
        "avg_mse": avg_mse,
        "avg_mae": avg_mae,
        "eval_split": eval_split,
        "route_mode": "static_with_input_confidence_fallback",
        "route_uses_future_y": False,
        "static_route": static_cluster_id_c.detach().cpu().tolist(),
        "route_counts": {str(k): int(v) for k, v in enumerate(route_count.detach().cpu().tolist())},
        "switch_rate": switch_count / max(total_routes, 1),
        "variant": dict(variant),
    }
    with (out_dir / "transfer_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def row_from_fallback_summary(
    variant: dict[str, Any],
    eval_split: str,
    summary: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    return {
        "status": "ok",
        "name": variant["name"],
        "eval_split": eval_split,
        "mse": summary["avg_mse"],
        "mae": summary["avg_mae"],
        "align": variant.get("align"),
        "score_len": variant.get("score_len"),
        "max_lag": variant.get("max_lag"),
        "margin_threshold": variant.get("margin_threshold"),
        "delta_threshold": variant.get("delta_threshold"),
        "min_top_score": variant.get("min_top_score"),
        "switch_rate": summary.get("switch_rate"),
        "route_counts": json.dumps(summary.get("route_counts", {}), ensure_ascii=False),
        "out_dir": str(out_dir),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "ETTm1ToETTm2.yaml")
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "ettm1_to_ettm2_input_fallback_selection")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    if args.device is not None:
        cfg.setdefault("exp", {})["device"] = args.device
    device = torch.device(str(cfg.get("exp", {}).get("device", "cuda:0")))
    ctx = load_context(cfg, device)
    static_cluster_id_c = static_route_from_train(ctx, cfg)

    rows: list[dict[str, Any]] = []
    for variant in make_variants():
        out_dir = args.out_root / "val_runs" / variant["name"]
        try:
            summary = evaluate_fallback(ctx, static_cluster_id_c, variant, "val", out_dir, args.batch_size)
            row = row_from_fallback_summary(variant, "val", summary, out_dir)
            print(
                f"[val ok] {variant['name']} mse={summary['avg_mse']:.6f} "
                f"mae={summary['avg_mae']:.6f} switch={summary['switch_rate']:.4f}"
            )
        except Exception as exc:
            row = {**{k: "" for k in FIELDS}, "status": "failed", "name": variant["name"], "eval_split": "val", "error": str(exc)}
            print(f"[val failed] {variant['name']}: {exc}")
        rows.append(row)
        write_fallback_rows(args.out_root / "val_results.csv", rows)

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    ok_rows.sort(key=lambda r: (float(r["mse"]), float(r["mae"])))
    write_fallback_rows(args.out_root / "val_results_ranked.csv", ok_rows)
    if not ok_rows:
        raise RuntimeError("No fallback val runs completed.")

    winner_name = ok_rows[0]["name"]
    winner = next(v for v in make_variants() if v["name"] == winner_name)
    test_out = args.out_root / "test_winner" / winner_name
    test_summary = evaluate_fallback(ctx, static_cluster_id_c, winner, "test", test_out, args.batch_size)
    test_row = row_from_fallback_summary(winner, "test", test_summary, test_out)
    write_fallback_rows(args.out_root / "selected_test.csv", [test_row])
    final = {
        "selection_metric": "val.avg_mse",
        "selected_name": winner_name,
        "selected_val_mse": ok_rows[0]["mse"],
        "selected_val_mae": ok_rows[0]["mae"],
        "selected_test_mse": test_summary["avg_mse"],
        "selected_test_mae": test_summary["avg_mae"],
        "selected_switch_rate": test_summary["switch_rate"],
        "static_route": static_cluster_id_c.detach().cpu().tolist(),
        "route_uses_future_y": False,
    }
    with (args.out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
