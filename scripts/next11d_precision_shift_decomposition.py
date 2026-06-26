from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.next11c_route_accuracy_diagnostic import _build_anchor_artifacts, _restore_cluster_penalty_prior
from scripts.next11d_binary_adoption_refit import _to_jsonable
from scripts.next11d_candidate_utility_stability import _candidate_gain_by_channel_penalty
from scripts.next11d_channel_action_space_diagnostic import (
    _allowed_mask_by_channel,
    _channel_oracle_labels_from_gain,
    _normalize_requested_splits,
)
from scripts.next11d_fixed_candidate_router_refit import _read_data_for_cfg
from scripts.shape_prior_diagnostic import _build_modules, _compute_penalty_scale, _make_loaders
from src.models.penalties import build_penalty_bank
from src.train import _collect_pred_residual_selector_tensors
from src.utils.seed import set_seed
from src.utils.yaml_io import load_yaml


def _safe_rate(num: int, denom: int) -> float:
    return float(num) / max(int(denom), 1)


def _decomposition_rows(
    *,
    labels_bc: torch.Tensor,
    pred_bc: torch.Tensor,
    gain_bcp: torch.Tensor,
    penalty_names: List[str],
) -> List[Dict[str, object]]:
    if labels_bc.dim() != 2 or pred_bc.dim() != 2:
        raise ValueError("labels_bc and pred_bc must have shape [B,C].")
    if gain_bcp.dim() != 3:
        raise ValueError("gain_bcp must have shape [B,C,P].")
    B, C = [int(v) for v in labels_bc.shape]
    if tuple(pred_bc.shape) != (B, C) or tuple(gain_bcp.shape[:2]) != (B, C):
        raise ValueError("labels, predictions, and gains must share [B,C].")
    P = int(gain_bcp.shape[2])
    if len(penalty_names) != P:
        raise ValueError("penalty_names length must match gain penalty dimension.")
    labels = labels_bc.detach().cpu().to(dtype=torch.long)
    pred = pred_bc.detach().cpu().to(dtype=torch.long)
    gain = gain_bcp.detach().cpu().to(dtype=torch.float32)
    rows: List[Dict[str, object]] = []
    for channel in range(C):
        for penalty_idx, penalty in enumerate(penalty_names, start=1):
            mask = pred[:, channel] == penalty_idx
            pred_count = int(mask.sum().item())
            selected_gain = gain[:, channel, penalty_idx - 1][mask]
            label_values = labels[:, channel][mask]
            exact = label_values == penalty_idx
            any_positive = label_values > 0
            false_skip = label_values == 0
            wrong_penalty = any_positive & ~exact
            negative_gain = selected_gain <= 0.0
            rows.append(
                {
                    "channel": int(channel),
                    "penalty_idx": int(penalty_idx - 1),
                    "penalty": str(penalty),
                    "pred_count": pred_count,
                    "exact_count": int(exact.sum().item()) if pred_count else 0,
                    "any_positive_count": int(any_positive.sum().item()) if pred_count else 0,
                    "false_skip_apply_count": int(false_skip.sum().item()) if pred_count else 0,
                    "wrong_penalty_count": int(wrong_penalty.sum().item()) if pred_count else 0,
                    "negative_gain_count": int(negative_gain.sum().item()) if pred_count else 0,
                    "exact_precision": _safe_rate(int(exact.sum().item()), pred_count),
                    "any_positive_precision": _safe_rate(int(any_positive.sum().item()), pred_count),
                    "false_skip_apply_rate": _safe_rate(int(false_skip.sum().item()), pred_count),
                    "wrong_penalty_rate": _safe_rate(int(wrong_penalty.sum().item()), pred_count),
                    "negative_gain_rate": _safe_rate(int(negative_gain.sum().item()), pred_count),
                    "mean_gain": float(selected_gain.mean().item()) if pred_count else 0.0,
                    "positive_gain_rate": float((selected_gain > 0.0).to(dtype=torch.float32).mean().item())
                    if pred_count
                    else 0.0,
                }
            )
    return rows


def _row_key(row: Dict[str, object]) -> tuple[int, str]:
    return int(row["channel"]), str(row["penalty"])


def _shift_rows(
    *,
    holdout_rows: List[Dict[str, object]],
    val_rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    holdout_by_key = {_row_key(row): row for row in holdout_rows}
    shifts: List[Dict[str, object]] = []
    total_val_false_skip = sum(int(row.get("false_skip_apply_count", 0) or 0) for row in val_rows)
    total_val_negative = sum(int(row.get("negative_gain_count", 0) or 0) for row in val_rows)
    for val in val_rows:
        key = _row_key(val)
        holdout = holdout_by_key.get(key, {})
        row = {
            "channel": int(val["channel"]),
            "penalty_idx": int(val.get("penalty_idx", -1)),
            "penalty": str(val["penalty"]),
            "holdout_pred_count": int(holdout.get("pred_count", 0) or 0),
            "val_pred_count": int(val.get("pred_count", 0) or 0),
            "holdout_exact_precision": float(holdout.get("exact_precision", 0.0) or 0.0),
            "val_exact_precision": float(val.get("exact_precision", 0.0) or 0.0),
            "holdout_any_positive_precision": float(holdout.get("any_positive_precision", 0.0) or 0.0),
            "val_any_positive_precision": float(val.get("any_positive_precision", 0.0) or 0.0),
            "holdout_mean_gain": float(holdout.get("mean_gain", 0.0) or 0.0),
            "val_mean_gain": float(val.get("mean_gain", 0.0) or 0.0),
            "val_false_skip_apply_count": int(val.get("false_skip_apply_count", 0) or 0),
            "val_negative_gain_count": int(val.get("negative_gain_count", 0) or 0),
        }
        row["exact_precision_delta"] = float(row["val_exact_precision"] - row["holdout_exact_precision"])
        row["any_positive_precision_delta"] = float(
            row["val_any_positive_precision"] - row["holdout_any_positive_precision"]
        )
        row["mean_gain_delta"] = float(row["val_mean_gain"] - row["holdout_mean_gain"])
        row["val_false_skip_share"] = _safe_rate(int(row["val_false_skip_apply_count"]), total_val_false_skip)
        row["val_negative_gain_share"] = _safe_rate(int(row["val_negative_gain_count"]), total_val_negative)
        shifts.append(row)
    shifts.sort(
        key=lambda row: (
            float(row.get("val_negative_gain_share", 0.0) or 0.0)
            + float(row.get("val_false_skip_share", 0.0) or 0.0),
            int(row.get("val_pred_count", 0) or 0),
        ),
        reverse=True,
    )
    return shifts


def _classify_shift(shifts: List[Dict[str, object]]) -> Dict[str, object]:
    active = [row for row in shifts if int(row.get("val_pred_count", 0) or 0) > 0]
    if not active:
        return {
            "failure_layer": "selection/adoption policy",
            "decision": "precision_shift_all_skip_or_empty",
            "top_false_skip_share": 0.0,
            "top_negative_gain_share": 0.0,
        }
    top_false = max(float(row.get("val_false_skip_share", 0.0) or 0.0) for row in active)
    top_negative = max(float(row.get("val_negative_gain_share", 0.0) or 0.0) for row in active)
    worst_precision_delta = min(float(row.get("exact_precision_delta", 0.0) or 0.0) for row in active)
    worst_gain_delta = min(float(row.get("mean_gain_delta", 0.0) or 0.0) for row in active)
    if top_false >= 0.50 or top_negative >= 0.50:
        decision = "precision_shift_concentrated_by_channel_penalty"
    elif worst_precision_delta < -0.20 or worst_gain_delta < -0.001:
        decision = "precision_shift_diffuse_train_val_utility_shift"
    else:
        decision = "precision_shift_not_obvious"
    return {
        "failure_layer": "train-val utility shift",
        "decision": decision,
        "top_false_skip_share": float(top_false),
        "top_negative_gain_share": float(top_negative),
        "worst_exact_precision_delta": float(worst_precision_delta),
        "worst_mean_gain_delta": float(worst_gain_delta),
    }


def _markdown_report(payload: Dict[str, object]) -> str:
    lines = [
        "# NEXT-11d Precision Shift Decomposition",
        "",
        f"- precision_dir: `{payload['precision_dir']}`",
        f"- no_test_read: `{payload['no_test_read']}`",
        f"- failure_layer: `{payload['verdict']['failure_layer']}`",
        f"- decision: `{payload['verdict']['decision']}`",
        "",
        "## Top Shift Rows",
        "",
        "| channel | penalty | holdout n | val n | holdout precision | val precision | precision delta | holdout gain | val gain | gain delta | val false-skip share | val negative-gain share |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("shift_rows", [])[: int(payload.get("top_k", 12))]:
        lines.append(
            "| {channel} | {penalty} | {hn} | {vn} | {hp:.4f} | {vp:.4f} | {pd:.4f} | {hg:.6f} | {vg:.6f} | {gd:.6f} | {fs:.4f} | {ng:.4f} |".format(
                channel=int(row["channel"]),
                penalty=str(row["penalty"]),
                hn=int(row["holdout_pred_count"]),
                vn=int(row["val_pred_count"]),
                hp=float(row["holdout_exact_precision"]),
                vp=float(row["val_exact_precision"]),
                pd=float(row["exact_precision_delta"]),
                hg=float(row["holdout_mean_gain"]),
                vg=float(row["val_mean_gain"]),
                gd=float(row["mean_gain_delta"]),
                fs=float(row["val_false_skip_share"]),
                ng=float(row["val_negative_gain_share"]),
            )
        )
    return "\n".join(lines) + "\n"


def _load_precision_paths(precision_dir: Path, config: Optional[Path], checkpoint: Optional[Path]) -> tuple[Path, Path]:
    summary_path = precision_dir / "channel_precision_refit.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing precision summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config_path = Path(str(config or summary["config_path"]))
    checkpoint_path = Path(str(checkpoint or summary["checkpoint_path"]))
    return config_path, checkpoint_path


def run(args: argparse.Namespace) -> Dict[str, object]:
    precision_dir = Path(args.precision_dir)
    cfg_path, checkpoint_path = _load_precision_paths(precision_dir, args.config, args.checkpoint)
    splits = _normalize_requested_splits(args.splits)
    if "test" in splits:
        raise ValueError("precision shift decomposition refuses to read test.")
    cfg = load_yaml(str(cfg_path))
    cfg.setdefault("eval", {})["skip_test"] = True
    set_seed(int(cfg["exp"]["seed"]), deterministic=bool((cfg.get("exp", {}) or {}).get("deterministic", False)))
    requested_device = str(args.device or cfg.get("exp", {}).get("device", "cpu"))
    device = torch.device(requested_device if torch.cuda.is_available() and requested_device != "cpu" else "cpu")
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    model, gate, pred_residual, cluster_id_c, K, moe_cfg, penalty_names = _build_modules(cfg, checkpoint, device)
    data_tc = _read_data_for_cfg(cfg)
    batch_size = int(cfg.get("train", {}).get("batch_size", 64))
    data_window_tc, loaders, eval_starts, train_loader, window_meta = _make_loaders(cfg, data_tc, batch_size=batch_size)
    penalty_fns = build_penalty_bank(penalty_names, jump_thr=float(cfg.get("penalties", {}).get("jump_threshold", 0.6)))
    penalty_scale = _compute_penalty_scale(train_loader, penalty_names, penalty_fns, int(window_meta["H"]), device)
    anchor = _build_anchor_artifacts(
        cfg=cfg,
        checkpoint=checkpoint,
        model=model,
        cluster_id_c=cluster_id_c,
        data_tc=data_window_tc,
        train_loader=train_loader,
        window_meta=window_meta,
        device=device,
    )
    prior_summary = _restore_cluster_penalty_prior(
        gate=gate,
        cfg=cfg,
        moe_cfg=moe_cfg,
        train_loader=train_loader,
        penalty_names=penalty_names,
        penalty_fns=penalty_fns,
        penalty_scale=penalty_scale,
        cluster_id_c=cluster_id_c,
        K=int(K),
        H=int(window_meta["H"]),
        device=device,
    )
    allowed_mask_kp = None
    if prior_summary.get("allowed_mask") is not None:
        allowed_mask_kp = torch.as_tensor(prior_summary["allowed_mask"], dtype=torch.bool)
    artifact = torch.load(precision_dir / "channel_precision_refit.pt", map_location="cpu")
    predictions_by_split = artifact["predictions"]
    rows_by_split: Dict[str, List[Dict[str, object]]] = {}
    for split in splits:
        tensors = _collect_pred_residual_selector_tensors(
            model=model,
            pred_residual=pred_residual,
            loader=loaders[split],
            cluster_id_c=cluster_id_c,
            K=int(K),
            moe_cfg=moe_cfg,
            device=device,
            penalty_count=len(penalty_names),
            history_anchor_cfg=anchor["history_anchor_cfg"],
            observed_history_tc=data_window_tc,
            input_len=int(window_meta["L"]),
            eval_start=int(eval_starts[split]),
            model_train_stat_adapter_pc=anchor["model_train_stat_adapter_pc"],
            model_train_stat_adapter_cfg=anchor["model_train_stat_adapter_cfg"],
            train_stat_anchor_pc=anchor["train_stat_anchor_pc"],
            train_residual_anchor_phc=anchor["train_residual_anchor_phc"],
            candidate_feature_mode=str(args.candidate_feature_mode),
        )
        if tensors is None:
            raise RuntimeError(f"Could not collect candidate tensors for split {split}.")
        gain_bcp = _candidate_gain_by_channel_penalty(
            base_bch=tensors["base"],
            cand_bcpH=tensors["cand"],
            y_bch=tensors["y"],
        )
        allowed_cp = _allowed_mask_by_channel(
            allowed_mask_kp=allowed_mask_kp,
            cluster_id_c=cluster_id_c.detach().cpu(),
            channel_count=int(gain_bcp.shape[1]),
            penalty_count=int(gain_bcp.shape[2]),
        )
        labels_bc = _channel_oracle_labels_from_gain(
            gain_bcp,
            allowed_cp=allowed_cp,
            margin=float(args.margin),
        )
        pred_flat = predictions_by_split[split].detach().cpu().to(dtype=torch.long).view(-1)
        pred_bc = pred_flat.reshape_as(labels_bc)
        rows_by_split[split] = _decomposition_rows(
            labels_bc=labels_bc,
            pred_bc=pred_bc,
            gain_bcp=gain_bcp,
            penalty_names=list(penalty_names),
        )
    if "train_holdout" not in rows_by_split or "val" not in rows_by_split:
        raise ValueError("decomposition requires train_holdout and val splits.")
    shift_rows = _shift_rows(holdout_rows=rows_by_split["train_holdout"], val_rows=rows_by_split["val"])
    payload = {
        "precision_dir": str(precision_dir),
        "config_path": str(cfg_path),
        "checkpoint_path": str(checkpoint_path),
        "out_dir": str(args.out_dir),
        "device": str(device),
        "no_test_read": True,
        "splits_collected": splits,
        "penalty_names": list(penalty_names),
        "top_k": int(args.top_k),
        "rows_by_split": rows_by_split,
        "shift_rows": shift_rows,
        "verdict": _classify_shift(shift_rows),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "precision_shift_decomposition.json").write_text(
        json.dumps(_to_jsonable(payload), indent=2),
        encoding="utf-8",
    )
    (out_dir / "precision_shift_decomposition.md").write_text(_markdown_report(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="NEXT-11d precision shift decomposition diagnostic.")
    parser.add_argument("--precision-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--splits", nargs="*", default=["train_holdout", "val"])
    parser.add_argument("--candidate-feature-mode", type=str, default="shape_proxy")
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()
    payload = run(args)
    verdict = payload["verdict"]
    print(
        "failure_layer={} decision={} top_false_skip_share={:.3f} top_negative_gain_share={:.3f} no_test_read=True".format(
            str(verdict["failure_layer"]),
            str(verdict["decision"]),
            float(verdict["top_false_skip_share"]),
            float(verdict["top_negative_gain_share"]),
        )
    )


if __name__ == "__main__":
    main()
