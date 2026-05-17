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

from scripts.run_ettm1_to_ettm2_input_fallback_selection import static_route_from_train  # noqa: E402
from scripts.run_ettm1_to_ettm2_input_route_selection import input_route_scores, load_context, read_yaml  # noqa: E402
from scripts.run_ettm1_to_ettm2_soft_cluster_matching import iter_batches, predict_all_heads, source_target_mse  # noqa: E402
from scripts.run_ettm1_to_ettm2_val_loss_route_selection import evaluate_route  # noqa: E402
from src.utils.metrics import accumulate_channel_errors, mse_mae_from_sums  # noqa: E402


HORIZONS = [192, 336]
CHANNELS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


RESULT_FIELDS = [
    "horizon",
    "candidate",
    "align",
    "score_len",
    "max_lag",
    "bins",
    "min_count",
    "val_mse",
    "val_mae",
    "test_mse",
    "test_mae",
    "target_self_mse",
    "source_test_mse",
    "route_summary",
]


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def variants() -> list[dict[str, Any]]:
    out = []
    for align, score_len, max_lag in [
        ("tail", 96, 24),
        ("tail", 192, 24),
        ("tail", 192, 96),
        ("tail", 336, 96),
        ("head", 96, 24),
        ("head", 192, 96),
    ]:
        if max_lag < score_len:
            out.append(
                {
                    "name": f"{align}_l{score_len}_lag{max_lag}",
                    "align": align,
                    "score_len": score_len,
                    "max_lag": max_lag,
                }
            )
    return out


def bin_specs() -> dict[str, list[float]]:
    return {
        "top_only": [],
        "coarse": [0.05, 0.15],
        "medium": [0.03, 0.08, 0.15, 0.30],
    }


def score_groups(scores_bck: torch.Tensor, thresholds: list[float]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k_count = int(scores_bck.shape[-1])
    top = torch.topk(scores_bck, k=min(2, k_count), dim=-1)
    top_idx = top.indices[..., 0]
    if top.values.shape[-1] > 1:
        margin = top.values[..., 0] - top.values[..., 1]
    else:
        margin = torch.full_like(top.values[..., 0], float("inf"))
    if thresholds:
        boundaries = torch.tensor(thresholds, device=scores_bck.device, dtype=scores_bck.dtype)
        bin_idx = torch.bucketize(margin, boundaries)
    else:
        bin_idx = torch.zeros_like(top_idx)
    return top_idx, bin_idx.to(torch.long), margin


def calibrate_router(
    ctx: dict[str, Any],
    variant: dict[str, Any],
    thresholds: list[float],
    static_route: tuple[int, ...],
    min_count: int,
    batch_size: int,
) -> dict[str, Any]:
    c_count = int(ctx["C"])
    k_count = int(ctx["K"])
    bins = len(thresholds) + 1
    groups = k_count * bins
    device = ctx["data_tc"].device
    err_sum = torch.zeros((c_count, groups, k_count), device=device, dtype=torch.float64)
    cnt = torch.zeros((c_count, groups), device=device, dtype=torch.float64)

    with torch.no_grad():
        for x, y in iter_batches(ctx, "val", batch_size):
            scores = input_route_scores(
                x,
                ctx["prototypes_kt"],
                align=str(variant["align"]),
                score_len=int(variant["score_len"]),
                max_lag=int(variant["max_lag"]),
            )
            top_idx, bin_idx, _ = score_groups(scores, thresholds)
            group_bc = top_idx * bins + bin_idx
            preds = predict_all_heads(ctx, x)
            err_bck = (preds - y.unsqueeze(2)).pow(2).mean(dim=-1).to(torch.float64)
            for c in range(c_count):
                group_c = group_bc[:, c]
                for g in range(groups):
                    mask = group_c == g
                    if bool(mask.any()):
                        cnt[c, g] += int(mask.sum().item())
                        err_sum[c, g] += err_bck[mask, c, :].sum(dim=0)

    head_map = torch.empty((c_count, groups), device=device, dtype=torch.long)
    fallback = torch.tensor(static_route, device=device, dtype=torch.long)
    avg_err = err_sum / cnt.clamp_min(1.0).unsqueeze(-1)
    for c in range(c_count):
        for g in range(groups):
            if cnt[c, g].item() >= int(min_count):
                head_map[c, g] = int(torch.argmin(avg_err[c, g]).item())
            else:
                head_map[c, g] = int(fallback[c].item())
    return {
        "head_map": head_map,
        "count": cnt,
        "avg_err": avg_err,
        "thresholds": thresholds,
        "bins": bins,
        "groups": groups,
        "variant": variant,
        "min_count": min_count,
        "static_route": static_route,
    }


def evaluate_router(ctx: dict[str, Any], router: dict[str, Any], split: str, batch_size: int) -> dict[str, Any]:
    c_count = int(ctx["C"])
    bins = int(router["bins"])
    variant = router["variant"]
    thresholds = list(router["thresholds"])
    head_map = router["head_map"]
    se_c = torch.zeros(c_count, device=ctx["data_tc"].device)
    ae_c = torch.zeros(c_count, device=ctx["data_tc"].device)
    route_count = torch.zeros(int(ctx["K"]), device=ctx["data_tc"].device)
    denom = 0
    with torch.no_grad():
        for x, y in iter_batches(ctx, split, batch_size):
            scores = input_route_scores(
                x,
                ctx["prototypes_kt"],
                align=str(variant["align"]),
                score_len=int(variant["score_len"]),
                max_lag=int(variant["max_lag"]),
            )
            top_idx, bin_idx, _ = score_groups(scores, thresholds)
            group_bc = top_idx * bins + bin_idx
            route_bc = torch.empty_like(group_bc)
            for c in range(c_count):
                route_bc[:, c] = head_map[c].index_select(0, group_bc[:, c])
            preds = predict_all_heads(ctx, x)
            yhat = preds.gather(2, route_bc.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, int(ctx["pred_len"]))).squeeze(2)
            accumulate_channel_errors(se_c, ae_c, yhat, y)
            route_count += torch.bincount(route_bc.reshape(-1), minlength=int(ctx["K"])).to(route_count)
            denom += int(x.shape[0] * y.shape[-1])
    mse_c, mae_c = mse_mae_from_sums(se_c, ae_c, denom)
    return {
        "avg_mse": float(mse_c.mean().item()),
        "avg_mae": float(mae_c.mean().item()),
        "mse_c": mse_c,
        "mae_c": mae_c,
        "route_counts": {str(k): int(v) for k, v in enumerate(route_count.detach().cpu().tolist())},
    }


def router_summary(router: dict[str, Any]) -> str:
    head_map = router["head_map"].detach().cpu()
    cnt = router["count"].detach().cpu()
    pieces = []
    for c, channel in enumerate(CHANNELS[: head_map.shape[0]]):
        used = []
        for g in range(head_map.shape[1]):
            if cnt[c, g].item() > 0:
                used.append(f"g{g}->h{int(head_map[c, g].item())}/n{int(cnt[c, g].item())}")
        pieces.append(f"{channel}:{'|'.join(used)}")
    return "; ".join(pieces)


def write_selected_map(path: Path, horizon: int, name: str, router: dict[str, Any]) -> None:
    rows = []
    head_map = router["head_map"].detach().cpu()
    cnt = router["count"].detach().cpu()
    avg_err = router["avg_err"].detach().cpu()
    for c, channel in enumerate(CHANNELS[: head_map.shape[0]]):
        for group in range(head_map.shape[1]):
            rows.append(
                {
                    "horizon": horizon,
                    "candidate": name,
                    "channel": channel,
                    "group": group,
                    "selected_head": int(head_map[c, group].item()),
                    "val_count": int(cnt[c, group].item()),
                    "val_err_head0": float(avg_err[c, group, 0].item()),
                    "val_err_head1": float(avg_err[c, group, 1].item()) if avg_err.shape[2] > 1 else "",
                    "val_err_head2": float(avg_err[c, group, 2].item()) if avg_err.shape[2] > 2 else "",
                }
            )
    fields = [
        "horizon",
        "candidate",
        "channel",
        "group",
        "selected_head",
        "val_count",
        "val_err_head0",
        "val_err_head1",
        "val_err_head2",
    ]
    write_rows(path, rows, fields)


def maybe_plot(out_root: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    for horizon in sorted({int(r["horizon"]) for r in rows}):
        cur = [r for r in rows if int(r["horizon"]) == horizon]
        labels = [str(r["candidate"]) for r in cur]
        vals = [float(r["test_mse"]) for r in cur]
        order = sorted(range(len(vals)), key=lambda i: vals[i])[:12]
        fig, ax = plt.subplots(figsize=(10, 4.6))
        ax.bar([labels[i] for i in order], [vals[i] for i in order])
        ax.set_ylabel("test MSE")
        ax.set_title(f"ETTm1 -> ETTm2 H{horizon} calibrated dynamic cluster matching")
        ax.tick_params(axis="x", labelrotation=60)
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_root / f"H{horizon}_dynamic_matching_top.png", dpi=180)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "ettm1_to_ettm2_val_calibrated_dynamic_matching")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        base_cfg_path = ROOT / "outputs" / "ettm1_other_ett_horizon_transfer" / "ETTm1_to_ETTm2" / f"pred_{horizon}" / "base_config.yaml"
        cfg = read_yaml(base_cfg_path)
        ctx = load_context(cfg, device)
        source_test_mse, target_self_mse = source_target_mse(horizon)
        static_route = tuple(int(v) for v in static_route_from_train(ctx, cfg).detach().cpu().tolist())
        static_test = evaluate_route(ctx, static_route, split="test", batch_size=args.batch_size)
        rows.append(
            {
                "horizon": horizon,
                "candidate": "hard_static_corr",
                "align": "",
                "score_len": "",
                "max_lag": "",
                "bins": "",
                "min_count": "",
                "val_mse": "",
                "val_mae": "",
                "test_mse": static_test["avg_mse"],
                "test_mae": static_test["avg_mae"],
                "target_self_mse": target_self_mse,
                "source_test_mse": source_test_mse,
                "route_summary": json.dumps(static_route),
            }
        )

        for variant in variants():
            for bin_name, thresholds in bin_specs().items():
                for min_count in [20, 100, 300]:
                    name = f"{variant['name']}_{bin_name}_n{min_count}"
                    router = calibrate_router(ctx, variant, thresholds, static_route, min_count, args.batch_size)
                    val = evaluate_router(ctx, router, "val", args.batch_size)
                    test = evaluate_router(ctx, router, "test", args.batch_size)
                    rows.append(
                        {
                            "horizon": horizon,
                            "candidate": name,
                            "align": variant["align"],
                            "score_len": variant["score_len"],
                            "max_lag": variant["max_lag"],
                            "bins": bin_name,
                            "min_count": min_count,
                            "val_mse": val["avg_mse"],
                            "val_mae": val["avg_mae"],
                            "test_mse": test["avg_mse"],
                            "test_mae": test["avg_mae"],
                            "target_self_mse": target_self_mse,
                            "source_test_mse": source_test_mse,
                            "route_summary": router_summary(router),
                        }
                    )
                    write_rows(args.out_root / "dynamic_matching_results.csv", rows, RESULT_FIELDS)

        cur = [r for r in rows if int(r["horizon"]) == horizon]
        dyn = [r for r in cur if str(r["candidate"]) != "hard_static_corr"]
        selected = min(dyn, key=lambda r: (float(r["val_mse"]), float(r["test_mse"])))
        best_test = min(cur, key=lambda r: float(r["test_mse"]))
        summary.append({"horizon": horizon, "selected_by_val": selected, "best_test_diagnostic": best_test})

        selected_variant_name, selected_bins, selected_n = None, None, None
        for variant in variants():
            for bin_name in bin_specs():
                prefix = f"{variant['name']}_{bin_name}_n"
                if str(selected["candidate"]).startswith(prefix):
                    selected_variant_name = variant
                    selected_bins = bin_name
                    selected_n = int(str(selected["candidate"]).split("_n")[-1])
        if selected_variant_name is not None and selected_bins is not None and selected_n is not None:
            router = calibrate_router(
                ctx,
                selected_variant_name,
                bin_specs()[selected_bins],
                static_route,
                selected_n,
                args.batch_size,
            )
            write_selected_map(args.out_root / f"H{horizon}_selected_router_map.csv", horizon, str(selected["candidate"]), router)

        with (args.out_root / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    maybe_plot(args.out_root, rows)
    print(args.out_root / "dynamic_matching_results.csv")
    print(args.out_root / "summary.json")


if __name__ == "__main__":
    main()
