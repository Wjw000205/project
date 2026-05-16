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


HORIZONS = [192, 336]
CHANNELS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


def route_key(route: tuple[int, ...]) -> str:
    return json.dumps([int(v) for v in route])


def write_matrix_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "horizon",
        "split",
        "channel",
        "channel_index",
        "head",
        "mse",
        "mae",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def write_route_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "horizon",
        "route_name",
        "route",
        "val_mse",
        "val_mae",
        "test_mse",
        "test_mae",
        "target_self_mse",
        "source_test_mse",
    ]
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


def route_from_matrix(mse_ck: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(v) for v in torch.argmin(mse_ck, dim=1).detach().cpu().tolist())


def maybe_existing_val_route(horizon: int) -> tuple[int, ...] | None:
    transfer_csv = ROOT / "outputs" / "ettm1_other_ett_horizon_transfer" / "transfer.csv"
    if not transfer_csv.exists():
        return None
    with transfer_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("target") == "ETTm2" and int(float(row.get("pred_len", -1))) == horizon:
                raw = row.get("val_route", "")
                if raw:
                    return tuple(int(v) for v in json.loads(raw))
    return None


def evaluate_named_routes(
    ctx: dict[str, Any],
    *,
    horizon: int,
    batch_size: int,
    routes: dict[str, tuple[int, ...]],
    source_test_mse: float,
    target_self_mse: float,
) -> list[dict[str, Any]]:
    out = []
    seen: set[tuple[int, ...]] = set()
    for name, route in routes.items():
        if route in seen:
            continue
        seen.add(route)
        val = evaluate_route(ctx, route, split="val", batch_size=batch_size)
        test = evaluate_route(ctx, route, split="test", batch_size=batch_size)
        out.append(
            {
                "horizon": horizon,
                "route_name": name,
                "route": route_key(route),
                "val_mse": val["avg_mse"],
                "val_mae": val["avg_mae"],
                "test_mse": test["avg_mse"],
                "test_mae": test["avg_mae"],
                "target_self_mse": target_self_mse,
                "source_test_mse": source_test_mse,
            }
        )
    return out


def write_heatmaps(out_root: Path, matrix_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return
    horizons = sorted({int(r["horizon"]) for r in matrix_rows})
    for horizon in horizons:
        cur = [r for r in matrix_rows if int(r["horizon"]) == horizon]
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), squeeze=False)
        for ax, split in zip(axes[0], ["val", "test"]):
            rows = [r for r in cur if r["split"] == split]
            channels = sorted({int(r["channel_index"]) for r in rows})
            heads = sorted({int(r["head"]) for r in rows})
            mat = np.zeros((len(channels), len(heads)), dtype=float)
            for r in rows:
                mat[int(r["channel_index"]), int(r["head"])] = float(r["mse"])
            im = ax.imshow(mat, aspect="auto", cmap="viridis")
            ax.set_title(f"H{horizon} {split} MSE")
            ax.set_xticks(range(len(heads)), [f"head {h}" for h in heads])
            ax.set_yticks(range(len(channels)), CHANNELS[: len(channels)])
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=7, color="white")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_root / f"H{horizon}_head_mse_heatmap.png", dpi=180)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "ettm1_to_ettm2_route_head_matrix")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.out_root.mkdir(parents=True, exist_ok=True)

    matrix_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        base_cfg_path = ROOT / "outputs" / "ettm1_other_ett_horizon_transfer" / "ETTm1_to_ETTm2" / f"pred_{horizon}" / "base_config.yaml"
        cfg = read_yaml(base_cfg_path)
        ctx = load_context(cfg, device)
        source_test_mse, target_self_mse = source_target_mse(horizon)
        C = int(ctx["C"])
        K = int(ctx["K"])

        mse_by_split: dict[str, torch.Tensor] = {}
        mae_by_split: dict[str, torch.Tensor] = {}
        for split in ["val", "test"]:
            mse_ck = torch.zeros((C, K), device=device)
            mae_ck = torch.zeros((C, K), device=device)
            for head in range(K):
                route = tuple([head] * C)
                metrics = evaluate_route(ctx, route, split=split, batch_size=args.batch_size)
                mse_ck[:, head] = metrics["mse_c"]
                mae_ck[:, head] = metrics["mae_c"]
                for c, channel in enumerate(ctx["channel_names"]):
                    matrix_rows.append(
                        {
                            "horizon": horizon,
                            "split": split,
                            "channel": channel,
                            "channel_index": c,
                            "head": head,
                            "mse": float(metrics["mse_c"][c].item()),
                            "mae": float(metrics["mae_c"][c].item()),
                        }
                    )
            mse_by_split[split] = mse_ck
            mae_by_split[split] = mae_ck

        static_route = tuple(int(v) for v in static_route_from_train(ctx, cfg).detach().cpu().tolist())
        val_opt_route = route_from_matrix(mse_by_split["val"])
        test_opt_route = route_from_matrix(mse_by_split["test"])
        routes = {
            "static_corr_train": static_route,
            "channel_val_oracle": val_opt_route,
            "channel_test_oracle_diagnostic": test_opt_route,
        }
        old_val_route = maybe_existing_val_route(horizon)
        if old_val_route is not None:
            routes["existing_greedy_val_route"] = old_val_route
        route_rows.extend(
            evaluate_named_routes(
                ctx,
                horizon=horizon,
                batch_size=args.batch_size,
                routes=routes,
                source_test_mse=source_test_mse,
                target_self_mse=target_self_mse,
            )
        )
        summary.append(
            {
                "horizon": horizon,
                "static_route": route_key(static_route),
                "channel_val_oracle_route": route_key(val_opt_route),
                "channel_test_oracle_route": route_key(test_opt_route),
                "target_self_mse": target_self_mse,
                "source_test_mse": source_test_mse,
            }
        )
        write_matrix_rows(args.out_root / "head_mse_matrix.csv", matrix_rows)
        write_route_rows(args.out_root / "route_comparison.csv", route_rows)
        with (args.out_root / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    write_heatmaps(args.out_root, matrix_rows)
    print(args.out_root / "head_mse_matrix.csv")
    print(args.out_root / "route_comparison.csv")
    print(args.out_root / "summary.json")


if __name__ == "__main__":
    main()
