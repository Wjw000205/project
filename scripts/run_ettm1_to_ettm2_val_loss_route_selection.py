from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_ettm1_to_ettm2_input_route_selection import load_context, read_yaml  # noqa: E402
from scripts.run_ettm1_to_ettm2_input_fallback_selection import static_route_from_train  # noqa: E402
from src.transfer import _predict_with_optional_residual  # noqa: E402
from src.utils.metrics import accumulate_channel_errors, mse_mae_from_sums  # noqa: E402


TRACE_FIELDS = [
    "start_name",
    "step",
    "action",
    "route",
    "val_mse",
    "val_mae",
]


def write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def write_trace(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRACE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in TRACE_FIELDS})


def write_metrics(path: Path, channels: list[str], mse_c: torch.Tensor, mae_c: torch.Tensor, cluster_id_c: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "channel": channels,
            "MSE": mse_c.detach().cpu().numpy(),
            "MAE": mae_c.detach().cpu().numpy(),
            "cluster_id": cluster_id_c.detach().cpu().numpy(),
        }
    )
    df.to_csv(path, index=False)


def evaluate_route(
    ctx: dict[str, Any],
    route: tuple[int, ...],
    *,
    split: str,
    batch_size: int,
    out_metrics: Path | None = None,
) -> dict[str, Any]:
    L = int(ctx["input_len"])
    H = int(ctx["pred_len"])
    C = int(ctx["C"])
    data_tc = ctx["data_tc"]
    eval_label_start, eval_end = (ctx["t_train"], ctx["t_val"]) if split == "val" else (ctx["t_val"], ctx["T"])
    eval_start = max(0, int(eval_label_start) - L) if bool(ctx.get("past_context", False)) else int(eval_label_start)
    eval_seg = data_tc[eval_start:eval_end]
    n_windows = int(eval_seg.shape[0] - L - H + 1)
    if n_windows <= 0:
        raise ValueError(f"No {split} windows available.")
    cluster_id_c = torch.tensor(route, device=data_tc.device, dtype=torch.long)
    se_c = torch.zeros(C, device=data_tc.device)
    ae_c = torch.zeros(C, device=data_tc.device)
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
            _, yhat = _predict_with_optional_residual(
                model=ctx["model"],
                gate=ctx["gate"],
                pred_residual=ctx["pred_residual"],
                x=x,
                cluster_id_c=cluster_id_c,
                meta=ctx["meta"],
                residual_scale_c=ctx["residual_scale_c"],
            )
            accumulate_channel_errors(se_c, ae_c, yhat, y)
            denom += int(x.shape[0] * H)
    mse_c, mae_c = mse_mae_from_sums(se_c, ae_c, denom)
    if out_metrics is not None:
        write_metrics(out_metrics, ctx["channel_names"], mse_c, mae_c, cluster_id_c)
    return {
        "avg_mse": float(mse_c.mean().item()),
        "avg_mae": float(mae_c.mean().item()),
        "mse_c": mse_c,
        "mae_c": mae_c,
        "num_eval_windows": n_windows,
        "eval_start_index": int(eval_start),
        "eval_label_start_index": int(eval_label_start),
        "eval_end_index": int(eval_end),
    }


def greedy_search(
    ctx: dict[str, Any],
    starts: dict[str, tuple[int, ...]],
    *,
    batch_size: int,
    search_mode: str,
) -> tuple[str, tuple[int, ...], dict[str, Any], list[dict[str, Any]]]:
    K = int(ctx["K"])
    C = int(ctx["C"])
    cache: dict[tuple[int, ...], dict[str, Any]] = {}
    trace: list[dict[str, Any]] = []

    def eval_cached(route: tuple[int, ...]) -> dict[str, Any]:
        if route not in cache:
            cache[route] = evaluate_route(ctx, route, split="val", batch_size=batch_size)
        return cache[route]

    best_name = ""
    best_route: tuple[int, ...] | None = None
    best_metrics: dict[str, Any] | None = None
    for start_name, start_route in starts.items():
        route = tuple(int(v) for v in start_route)
        metrics = eval_cached(route)
        trace.append(
            {
                "start_name": start_name,
                "step": 0,
                "action": "start",
                "route": json.dumps(route),
                "val_mse": metrics["avg_mse"],
                "val_mae": metrics["avg_mae"],
            }
        )
        if search_mode == "greedy":
            step = 0
            improved = True
            while improved:
                improved = False
                current = eval_cached(route)
                local_best_route = route
                local_best_metrics = current
                local_best_action = ""
                for c in range(C):
                    old_k = route[c]
                    for k in range(K):
                        if k == old_k:
                            continue
                        candidate = list(route)
                        candidate[c] = k
                        candidate_t = tuple(candidate)
                        cand_metrics = eval_cached(candidate_t)
                        if cand_metrics["avg_mse"] + 1.0e-12 < local_best_metrics["avg_mse"]:
                            local_best_route = candidate_t
                            local_best_metrics = cand_metrics
                            local_best_action = f"channel_{c}:{old_k}->{k}"
                if local_best_route != route:
                    route = local_best_route
                    metrics = local_best_metrics
                    step += 1
                    improved = True
                    trace.append(
                        {
                            "start_name": start_name,
                            "step": step,
                            "action": local_best_action,
                            "route": json.dumps(route),
                            "val_mse": metrics["avg_mse"],
                            "val_mae": metrics["avg_mae"],
                        }
                    )
        final_metrics = eval_cached(route)
        if best_metrics is None or final_metrics["avg_mse"] < best_metrics["avg_mse"]:
            best_name = start_name
            best_route = route
            best_metrics = final_metrics
    if best_route is None or best_metrics is None:
        raise RuntimeError("Route search failed.")
    return best_name, best_route, best_metrics, trace


def make_fixed_cfg(base: dict[str, Any], route: tuple[int, ...], out_dir: Path, eval_split: str) -> dict[str, Any]:
    cfg = json.loads(json.dumps(base))
    cfg.setdefault("exp", {})["out_dir"] = str(out_dir)
    cfg.setdefault("transfer", {})["fixed_cluster_id"] = [int(v) for v in route]
    cfg["transfer"].setdefault("knn_hybrid", {})["enable"] = False
    cfg.setdefault("normalize", {})["train_only"] = True
    cfg.setdefault("eval", {})["split"] = eval_split
    cfg["eval"].setdefault("batch_size", 64)
    return cfg


def run_transfer(py: list[str], cfg_path: Path) -> str:
    proc = subprocess.run(
        [*py, "-u", "-m", "src.transfer", "--config", str(cfg_path)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout)
    return proc.stdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "ETTm1ToETTm2.yaml")
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "ettm1_to_ettm2_val_loss_route_selection")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--python", type=Path, default=None)
    ap.add_argument("--search-mode", choices=["auto", "greedy", "global"], default="auto")
    ap.add_argument("--max-greedy-channels", type=int, default=64)
    args = ap.parse_args()

    base = read_yaml(args.config)
    if args.device is not None:
        base.setdefault("exp", {})["device"] = args.device
    device = torch.device(str(base.get("exp", {}).get("device", "cuda:0")))
    ctx = load_context(base, device)
    corr_route = tuple(int(v) for v in static_route_from_train(ctx, base).detach().cpu().tolist())
    K = int(ctx["K"])
    C = int(ctx["C"])
    search_mode = args.search_mode
    if search_mode == "auto":
        search_mode = "greedy" if C <= int(args.max_greedy_channels) else "global"
    starts = {"corr_train": corr_route}
    for k in range(K):
        starts[f"all_{k}"] = tuple([k] * C)

    best_start, best_route, best_val, trace = greedy_search(
        ctx,
        starts,
        batch_size=args.batch_size,
        search_mode=search_mode,
    )
    args.out_root.mkdir(parents=True, exist_ok=True)
    write_trace(args.out_root / "search_trace.csv", trace)
    evaluate_route(
        ctx,
        best_route,
        split="val",
        batch_size=args.batch_size,
        out_metrics=args.out_root / "selected_val_metrics.csv",
    )

    py = [str(args.python)] if args.python else [sys.executable]
    val_cfg = make_fixed_cfg(base, best_route, args.out_root / "selected_val_transfer", "val")
    test_cfg = make_fixed_cfg(base, best_route, args.out_root / "selected_test_transfer", "test")
    val_cfg_path = args.out_root / "selected_val_config.yaml"
    test_cfg_path = args.out_root / "selected_test_config.yaml"
    write_yaml(val_cfg_path, val_cfg)
    write_yaml(test_cfg_path, test_cfg)
    run_transfer(py, val_cfg_path)
    run_transfer(py, test_cfg_path)
    with (args.out_root / "selected_test_transfer" / "transfer_summary.json").open("r", encoding="utf-8") as f:
        test_summary = json.load(f)
    summary = {
        "selection_metric": "val.avg_mse",
        "search": "greedy_channel_cluster_val_loss",
        "search_mode": search_mode,
        "start_used": best_start,
        "corr_train_route": list(corr_route),
        "selected_route": list(best_route),
        "selected_val_mse": best_val["avg_mse"],
        "selected_val_mae": best_val["avg_mae"],
        "selected_test_mse": test_summary["avg_mse"],
        "selected_test_mae": test_summary["avg_mae"],
        "selected_test_config": str(test_cfg_path),
        "selected_test_out_dir": str(args.out_root / "selected_test_transfer"),
    }
    with (args.out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
