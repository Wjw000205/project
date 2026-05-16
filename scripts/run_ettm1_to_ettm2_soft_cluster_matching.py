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
from scripts.run_ettm1_to_ettm2_input_route_selection import load_context, read_yaml  # noqa: E402
from scripts.run_ettm1_to_ettm2_val_loss_route_selection import evaluate_route  # noqa: E402
from src.transfer import _predict_with_optional_residual  # noqa: E402
from src.utils.metrics import accumulate_channel_errors, mse_mae_from_sums  # noqa: E402


HORIZONS = [192, 336]
CHANNELS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


def route_key(route: tuple[int, ...]) -> str:
    return json.dumps([int(v) for v in route])


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


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


def simplex_grid(k: int, step: float, device: torch.device) -> torch.Tensor:
    units = int(round(1.0 / float(step)))
    if abs(units * float(step) - 1.0) > 1.0e-8:
        raise ValueError("step must divide 1.0 exactly.")
    rows: list[list[float]] = []

    def rec(prefix: list[int], remain: int, depth: int) -> None:
        if depth == k - 1:
            rows.append([*prefix, remain])
            return
        for v in range(remain + 1):
            rec([*prefix, v], remain - v, depth + 1)

    rec([], units, 0)
    return torch.tensor(rows, device=device, dtype=torch.float32) / float(units)


def predict_all_heads(ctx: dict[str, Any], x: torch.Tensor) -> torch.Tensor:
    c_count = int(ctx["C"])
    preds = []
    for head in range(int(ctx["K"])):
        cluster_id_c = torch.full((c_count,), head, device=x.device, dtype=torch.long)
        _, yhat = _predict_with_optional_residual(
            model=ctx["model"],
            gate=ctx["gate"],
            pred_residual=ctx["pred_residual"],
            x=x,
            cluster_id_c=cluster_id_c,
            meta=ctx["meta"],
            residual_scale_c=ctx["residual_scale_c"],
        )
        preds.append(yhat)
    return torch.stack(preds, dim=2)


def iter_batches(ctx: dict[str, Any], split: str, batch_size: int):
    l_count = int(ctx["input_len"])
    h_count = int(ctx["pred_len"])
    data_tc = ctx["data_tc"]
    label_start, end_idx = (ctx["t_train"], ctx["t_val"]) if split == "val" else (ctx["t_val"], ctx["T"])
    start_idx = max(0, int(label_start) - l_count) if bool(ctx.get("past_context", False)) else int(label_start)
    eval_seg = data_tc[start_idx:end_idx]
    n_windows = int(eval_seg.shape[0] - l_count - h_count + 1)
    if n_windows <= 0:
        raise ValueError(f"No {split} windows available.")
    for start in range(0, n_windows, batch_size):
        stop = min(start + batch_size, n_windows)
        xs = []
        ys = []
        for i in range(start, stop):
            win = eval_seg[i : i + l_count + h_count]
            xs.append(win[:l_count].T)
            ys.append(win[l_count:].T)
        yield torch.stack(xs, dim=0), torch.stack(ys, dim=0)


def collect_quadratic_stats(ctx: dict[str, Any], split: str, batch_size: int) -> dict[str, torch.Tensor | int]:
    c_count = int(ctx["C"])
    k_count = int(ctx["K"])
    device = ctx["data_tc"].device
    a = torch.zeros((c_count, k_count, k_count), device=device, dtype=torch.float64)
    b = torch.zeros((c_count, k_count), device=device, dtype=torch.float64)
    y2 = torch.zeros((c_count,), device=device, dtype=torch.float64)
    denom = 0
    with torch.no_grad():
        for x, y in iter_batches(ctx, split, batch_size):
            preds = predict_all_heads(ctx, x).to(torch.float64)
            yd = y.to(torch.float64)
            a += torch.einsum("bckh,bclh->ckl", preds, preds)
            b += torch.einsum("bckh,bch->ck", preds, yd)
            y2 += yd.pow(2).sum(dim=(0, 2))
            denom += int(x.shape[0] * y.shape[-1])
    return {"A": a / float(denom), "b": b / float(denom), "y2": y2 / float(denom), "denom": denom}


def mse_for_weights(stats: dict[str, torch.Tensor | int], weights_ck: torch.Tensor) -> torch.Tensor:
    a = stats["A"].to(weights_ck.device, dtype=torch.float64)
    b = stats["b"].to(weights_ck.device, dtype=torch.float64)
    y2 = stats["y2"].to(weights_ck.device, dtype=torch.float64)
    w = weights_ck.to(torch.float64)
    return torch.einsum("ck,ckl,cl->c", w, a, w) - 2.0 * (w * b).sum(dim=1) + y2


def select_weights(
    stats: dict[str, torch.Tensor | int],
    grid_w: torch.Tensor,
    prior_ck: torch.Tensor,
    reg: float,
    entropy_bonus: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    a = stats["A"].to(grid_w.device, dtype=torch.float64)
    b = stats["b"].to(grid_w.device, dtype=torch.float64)
    y2 = stats["y2"].to(grid_w.device, dtype=torch.float64)
    grid = grid_w.to(torch.float64)
    prior = prior_ck.to(grid_w.device, dtype=torch.float64)
    # obj[c, n] = mse of grid point n for channel c.
    obj = torch.einsum("nk,ckl,nl->cn", grid, a, grid) - 2.0 * torch.einsum("nk,ck->cn", grid, b) + y2[:, None]
    if reg > 0.0:
        obj = obj + float(reg) * (grid[None, :, :] - prior[:, None, :]).pow(2).sum(dim=2)
    if entropy_bonus != 0.0:
        entropy = -(grid.clamp_min(1.0e-8) * grid.clamp_min(1.0e-8).log()).sum(dim=1)
        obj = obj - float(entropy_bonus) * entropy[None, :]
    idx = torch.argmin(obj, dim=1)
    chosen = grid_w.index_select(0, idx)
    mse_c = mse_for_weights(stats, chosen)
    return chosen.to(torch.float32), mse_c.to(torch.float32)


def onehot_prior(route: tuple[int, ...], k_count: int, device: torch.device, smooth: float) -> torch.Tensor:
    out = torch.full((len(route), k_count), float(smooth) / max(k_count - 1, 1), device=device, dtype=torch.float32)
    for c, head in enumerate(route):
        out[c, int(head)] = 1.0 - float(smooth)
    return out


def uniform_prior(c_count: int, k_count: int, device: torch.device) -> torch.Tensor:
    return torch.full((c_count, k_count), 1.0 / float(k_count), device=device, dtype=torch.float32)


def evaluate_soft(ctx: dict[str, Any], weights_ck: torch.Tensor, split: str, batch_size: int) -> dict[str, Any]:
    c_count = int(ctx["C"])
    se_c = torch.zeros(c_count, device=ctx["data_tc"].device)
    ae_c = torch.zeros(c_count, device=ctx["data_tc"].device)
    denom = 0
    with torch.no_grad():
        for x, y in iter_batches(ctx, split, batch_size):
            preds = predict_all_heads(ctx, x)
            yhat = torch.einsum("bckh,ck->bch", preds, weights_ck.to(preds.device, dtype=preds.dtype))
            accumulate_channel_errors(se_c, ae_c, yhat, y)
            denom += int(x.shape[0] * y.shape[-1])
    mse_c, mae_c = mse_mae_from_sums(se_c, ae_c, denom)
    return {
        "avg_mse": float(mse_c.mean().item()),
        "avg_mae": float(mae_c.mean().item()),
        "mse_c": mse_c,
        "mae_c": mae_c,
    }


def write_weight_table(path: Path, horizon: int, name: str, weights_ck: torch.Tensor, mse_c: torch.Tensor, mae_c: torch.Tensor) -> None:
    rows = []
    weights = weights_ck.detach().cpu()
    for c, channel in enumerate(CHANNELS[: weights.shape[0]]):
        row = {
            "horizon": horizon,
            "candidate": name,
            "channel": channel,
            "selected_head": int(torch.argmax(weights[c]).item()),
            "weight_head0": float(weights[c, 0].item()),
            "weight_head1": float(weights[c, 1].item()) if weights.shape[1] > 1 else "",
            "weight_head2": float(weights[c, 2].item()) if weights.shape[1] > 2 else "",
            "test_mse": float(mse_c[c].item()),
            "test_mae": float(mae_c[c].item()),
        }
        rows.append(row)
    fields = [
        "horizon",
        "candidate",
        "channel",
        "selected_head",
        "weight_head0",
        "weight_head1",
        "weight_head2",
        "test_mse",
        "test_mae",
    ]
    write_rows(path, rows, fields)


def maybe_plot(out_root: Path, rows: list[dict[str, Any]], best_weights: dict[int, torch.Tensor]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    for horizon, weights in best_weights.items():
        w = weights.detach().cpu()
        fig, ax = plt.subplots(figsize=(9.0, 4.2))
        bottom = torch.zeros(w.shape[0])
        labels = CHANNELS[: w.shape[0]]
        for head in range(w.shape[1]):
            vals = w[:, head]
            ax.bar(labels, vals, bottom=bottom, label=f"head {head}")
            bottom += vals
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("soft cluster weight")
        ax.set_title(f"ETTm1 -> ETTm2 H{horizon} validation-calibrated soft cluster matching")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(out_root / f"H{horizon}_soft_cluster_weights.png", dpi=180)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "ettm1_to_ettm2_soft_cluster_matching")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--grid-step", type=float, default=0.02)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.out_root.mkdir(parents=True, exist_ok=True)

    result_rows: list[dict[str, Any]] = []
    best_weights: dict[int, torch.Tensor] = {}
    all_weight_rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        base_cfg_path = ROOT / "outputs" / "ettm1_other_ett_horizon_transfer" / "ETTm1_to_ETTm2" / f"pred_{horizon}" / "base_config.yaml"
        cfg = read_yaml(base_cfg_path)
        ctx = load_context(cfg, device)
        source_test_mse, target_self_mse = source_target_mse(horizon)
        static_route = tuple(int(v) for v in static_route_from_train(ctx, cfg).detach().cpu().tolist())
        k_count = int(ctx["K"])
        c_count = int(ctx["C"])
        grid = simplex_grid(k_count, float(args.grid_step), device)
        val_stats = collect_quadratic_stats(ctx, "val", args.batch_size)

        hard_static = evaluate_route(ctx, static_route, split="test", batch_size=args.batch_size)
        hard_val_route = tuple(int(v) for v in torch.argmin(val_stats["A"].diagonal(dim1=1, dim2=2) - 2.0 * val_stats["b"], dim=1).detach().cpu().tolist())
        hard_val = evaluate_route(ctx, hard_val_route, split="test", batch_size=args.batch_size)
        result_rows.extend(
            [
                {
                    "horizon": horizon,
                    "candidate": "hard_static_corr",
                    "selection": "train-only cycle/corr",
                    "val_mse": "",
                    "test_mse": hard_static["avg_mse"],
                    "test_mae": hard_static["avg_mae"],
                    "target_self_mse": target_self_mse,
                    "source_test_mse": source_test_mse,
                    "route_or_weights": route_key(static_route),
                },
                {
                    "horizon": horizon,
                    "candidate": "hard_val_channel_oracle",
                    "selection": "per-channel validation loss",
                    "val_mse": "",
                    "test_mse": hard_val["avg_mse"],
                    "test_mae": hard_val["avg_mae"],
                    "target_self_mse": target_self_mse,
                    "source_test_mse": source_test_mse,
                    "route_or_weights": route_key(hard_val_route),
                },
            ]
        )

        priors = {
            "uniform": uniform_prior(c_count, k_count, device),
            "static_smooth005": onehot_prior(static_route, k_count, device, smooth=0.05),
            "static_smooth015": onehot_prior(static_route, k_count, device, smooth=0.15),
        }
        candidates: list[tuple[str, torch.Tensor, float, float]] = []
        for prior_name, prior in priors.items():
            for reg in [0.0, 1.0e-4, 1.0e-3, 1.0e-2, 5.0e-2, 1.0e-1]:
                for entropy_bonus in [0.0, 1.0e-4, 1.0e-3]:
                    name = f"soft_{prior_name}_reg{reg:g}_ent{entropy_bonus:g}"
                    weights, val_mse_c = select_weights(val_stats, grid, prior, reg, entropy_bonus)
                    candidates.append((name, weights, float(val_mse_c.mean().item()), float(reg)))

        seen_weight_keys: set[str] = set()
        for name, weights, val_mse, reg in candidates:
            key = json.dumps(weights.detach().cpu().round(decimals=4).tolist())
            if key in seen_weight_keys:
                continue
            seen_weight_keys.add(key)
            test = evaluate_soft(ctx, weights, "test", args.batch_size)
            result_rows.append(
                {
                    "horizon": horizon,
                    "candidate": name,
                    "selection": "validation soft convex cluster matching",
                    "val_mse": val_mse,
                    "test_mse": test["avg_mse"],
                    "test_mae": test["avg_mae"],
                    "target_self_mse": target_self_mse,
                    "source_test_mse": source_test_mse,
                    "route_or_weights": key,
                }
            )
            for c, channel in enumerate(CHANNELS[: c_count]):
                all_weight_rows.append(
                    {
                        "horizon": horizon,
                        "candidate": name,
                        "channel": channel,
                        "weight_head0": float(weights[c, 0].item()),
                        "weight_head1": float(weights[c, 1].item()) if k_count > 1 else "",
                        "weight_head2": float(weights[c, 2].item()) if k_count > 2 else "",
                        "test_mse": float(test["mse_c"][c].item()),
                        "test_mae": float(test["mae_c"][c].item()),
                    }
                )

        horizon_rows = [r for r in result_rows if int(r["horizon"]) == horizon and str(r["candidate"]).startswith("soft_")]
        best = min(horizon_rows, key=lambda r: (float(r["val_mse"]), float(r["test_mse"])))
        best_weight_tensor = torch.tensor(json.loads(best["route_or_weights"]), device=device, dtype=torch.float32)
        best_weights[horizon] = best_weight_tensor
        best_test = evaluate_soft(ctx, best_weight_tensor, "test", args.batch_size)
        write_weight_table(
            args.out_root / f"H{horizon}_selected_weights.csv",
            horizon,
            str(best["candidate"]),
            best_weight_tensor,
            best_test["mse_c"],
            best_test["mae_c"],
        )

        fields = [
            "horizon",
            "candidate",
            "selection",
            "val_mse",
            "test_mse",
            "test_mae",
            "target_self_mse",
            "source_test_mse",
            "route_or_weights",
        ]
        write_rows(args.out_root / "soft_cluster_results.csv", result_rows, fields)

    weight_fields = [
        "horizon",
        "candidate",
        "channel",
        "weight_head0",
        "weight_head1",
        "weight_head2",
        "test_mse",
        "test_mae",
    ]
    write_rows(args.out_root / "all_candidate_channel_weights.csv", all_weight_rows, weight_fields)
    summary = []
    for horizon in HORIZONS:
        cur = [r for r in result_rows if int(r["horizon"]) == horizon]
        best_by_val = min(
            [r for r in cur if str(r["candidate"]).startswith("soft_")],
            key=lambda r: (float(r["val_mse"]), float(r["test_mse"])),
        )
        best_by_test = min(cur, key=lambda r: float(r["test_mse"]))
        summary.append({"horizon": horizon, "selected_by_val": best_by_val, "best_test_diagnostic": best_by_test})
    with (args.out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    maybe_plot(args.out_root, result_rows, best_weights)
    print(args.out_root / "soft_cluster_results.csv")
    print(args.out_root / "summary.json")


if __name__ == "__main__":
    main()
