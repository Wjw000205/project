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

from scripts.next11c_route_accuracy_diagnostic import (  # noqa: E402
    _build_anchor_artifacts,
    _make_loaders,
    _read_data_for_cfg,
    _restore_cluster_penalty_prior,
)
from scripts.shape_prior_diagnostic import (  # noqa: E402
    _build_modules,
    _collect_shape_samples,
    _compute_penalty_scale,
)
from src.models.penalties import build_penalty_bank  # noqa: E402
from src.train import _normalize_gate_feature_mode  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402
from src.utils.yaml_io import load_yaml  # noqa: E402


def _safe_div(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if float(den) > 0.0 else 0.0


def _normalize_requested_splits(raw_splits: Iterable[str]) -> List[str]:
    allowed = {"train_fit", "train_holdout", "val"}
    splits: List[str] = []
    for raw in raw_splits:
        split = str(raw).strip().lower()
        if split == "train":
            split = "train_fit"
        if split == "test":
            raise ValueError("selected penalty gain diagnostic refuses to read test.")
        if split not in allowed:
            raise ValueError(f"unsupported split {raw!r}; expected train_fit, train_holdout, or val.")
        if split not in splits:
            splits.append(split)
    return splits or ["train_fit", "train_holdout", "val"]


def _selected_penalty_gain_summary(
    *,
    split: str,
    gain_bkp: torch.Tensor,
    labels_bk: torch.Tensor,
    current_pred_bk: torch.Tensor,
    penalty_names: List[str],
    action_margin: float = 0.0,
) -> Dict[str, object]:
    if gain_bkp.dim() != 3:
        raise ValueError("gain_bkp must have shape [B,K,P].")
    B, K, P = [int(v) for v in gain_bkp.shape]
    if int(len(penalty_names)) != P:
        raise ValueError("penalty_names length must match gain_bkp penalty dimension.")
    labels = labels_bk.detach().cpu().to(dtype=torch.long)
    current = current_pred_bk.detach().cpu().to(dtype=torch.long)
    if tuple(labels.shape) != (B, K) or tuple(current.shape) != (B, K):
        raise ValueError("labels_bk and current_pred_bk must share shape [B,K] with gain_bkp.")

    gain = gain_bkp.detach().cpu().to(dtype=torch.float32)
    selected = current > 0
    skipped = current == 0
    valid_selected = selected & (current <= P)
    route_count = int(B * K)

    selected_gain = torch.zeros((B, K), dtype=torch.float32)
    if bool(valid_selected.any().item()):
        gather_idx = (current.clamp(min=1, max=P) - 1).unsqueeze(-1)
        gathered = gain.gather(dim=-1, index=gather_idx).squeeze(-1)
        selected_gain = torch.where(valid_selected, gathered, torch.zeros_like(gathered))

    oracle_positive = labels > 0
    correct_penalty = valid_selected & (current == labels)
    wrong_penalty = valid_selected & oracle_positive & (current != labels)
    false_adopt = valid_selected & (~oracle_positive)
    missed_positive = skipped & oracle_positive
    selected_count = int(valid_selected.sum().item())
    oracle_positive_count = int(oracle_positive.sum().item())
    margin = float(action_margin)

    selected_values = selected_gain[valid_selected]
    payload: Dict[str, object] = {
        "split": str(split),
        "route_count": route_count,
        "selected_count": selected_count,
        "selected_rate": _safe_div(selected_count, route_count),
        "skip_count": int(skipped.sum().item()),
        "skip_rate": _safe_div(int(skipped.sum().item()), route_count),
        "oracle_positive_count": oracle_positive_count,
        "oracle_positive_rate": _safe_div(oracle_positive_count, route_count),
        "selected_label_precision": _safe_div(int(correct_penalty.sum().item()), selected_count),
        "selected_wrong_penalty_rate": _safe_div(int(wrong_penalty.sum().item()), selected_count),
        "selected_false_adopt_rate": _safe_div(int(false_adopt.sum().item()), selected_count),
        "selected_gain_mean_on_selected": float(selected_values.mean().item()) if selected_count else 0.0,
        "selected_gain_sum_per_route": float(selected_gain.sum().item() / max(route_count, 1)),
        "selected_gain_positive_rate": float((selected_values > 0.0).to(dtype=torch.float32).mean().item())
        if selected_count
        else 0.0,
        "selected_gain_above_margin_rate": float((selected_values > margin).to(dtype=torch.float32).mean().item())
        if selected_count
        else 0.0,
        "selected_gain_nonpositive_count": int((valid_selected & (selected_gain <= 0.0)).sum().item()),
        "missed_positive_count": int(missed_positive.sum().item()),
        "missed_positive_rate_on_oracle_positive": _safe_div(int(missed_positive.sum().item()), oracle_positive_count),
        "missed_positive_best_gain_mean": 0.0,
        "per_penalty": {},
        "per_cluster": [],
    }
    if int(missed_positive.sum().item()) > 0:
        best_gain_bk = gain.max(dim=-1).values
        payload["missed_positive_best_gain_mean"] = float(best_gain_bk[missed_positive].mean().item())

    per_penalty: Dict[str, object] = {}
    for penalty_label, name in enumerate(penalty_names, start=1):
        mask = current == int(penalty_label)
        values = selected_gain[mask]
        count = int(mask.sum().item())
        per_penalty[str(name)] = {
            "selected_count": count,
            "selected_rate": _safe_div(count, route_count),
            "label_precision": _safe_div(int((mask & (labels == penalty_label)).sum().item()), count),
            "false_adopt_rate": _safe_div(int((mask & (labels == 0)).sum().item()), count),
            "gain_mean": float(values.mean().item()) if count else 0.0,
            "gain_positive_rate": float((values > 0.0).to(dtype=torch.float32).mean().item()) if count else 0.0,
            "gain_above_margin_rate": float((values > margin).to(dtype=torch.float32).mean().item()) if count else 0.0,
        }
    payload["per_penalty"] = per_penalty

    per_cluster: List[Dict[str, object]] = []
    for cluster in range(K):
        row: Dict[str, object] = {"cluster": int(cluster), "per_penalty": {}}
        for penalty_label, name in enumerate(penalty_names, start=1):
            mask = current[:, cluster] == int(penalty_label)
            values = selected_gain[:, cluster][mask]
            count = int(mask.sum().item())
            row["per_penalty"][str(name)] = {
                "selected_count": count,
                "selected_rate": _safe_div(count, B),
                "label_precision": _safe_div(int((mask & (labels[:, cluster] == penalty_label)).sum().item()), count),
                "false_adopt_rate": _safe_div(int((mask & (labels[:, cluster] == 0)).sum().item()), count),
                "gain_mean": float(values.mean().item()) if count else 0.0,
                "gain_positive_rate": float((values > 0.0).to(dtype=torch.float32).mean().item()) if count else 0.0,
                "gain_above_margin_rate": float((values > margin).to(dtype=torch.float32).mean().item())
                if count
                else 0.0,
            }
        per_cluster.append(row)
    payload["per_cluster"] = per_cluster
    return payload


def _load_route_tensors(route_dir: Path, split: str) -> Dict[str, torch.Tensor]:
    path = route_dir / f"route_tensors_{split}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    tensors = obj.get("tensors", obj)
    if "labels" not in tensors or "current_pred" not in tensors:
        raise ValueError(f"{path} does not contain route labels/current predictions.")
    return tensors


def run_selected_penalty_gain_diagnostic(
    *,
    config_path: Path,
    checkpoint_path: Path,
    route_dir: Path,
    out_dir: Path,
    splits: Iterable[str],
    device_arg: Optional[str],
    action_margin: float,
) -> Dict[str, object]:
    requested_splits = _normalize_requested_splits(splits)
    cfg = load_yaml(str(config_path))
    set_seed(int(cfg["exp"]["seed"]), deterministic=bool((cfg.get("exp", {}) or {}).get("deterministic", False)))
    requested_device = str(device_arg or cfg.get("exp", {}).get("device", "cpu"))
    device = torch.device(requested_device if torch.cuda.is_available() and requested_device != "cpu" else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model, gate, pred_residual, cluster_id_c, K, moe_cfg, penalty_names = _build_modules(cfg, checkpoint, device)
    data_tc, _ = _read_data_for_cfg(cfg)
    batch_size = int(cfg.get("train", {}).get("batch_size", 64))
    loaders, eval_starts, train_loader, window_meta = _make_loaders(cfg, data_tc, batch_size=batch_size)
    penalty_fns = build_penalty_bank(penalty_names, jump_thr=float(cfg.get("penalties", {}).get("jump_threshold", 0.6)))
    penalty_scale = _compute_penalty_scale(train_loader, penalty_names, penalty_fns, int(window_meta["H"]), device)
    anchor = _build_anchor_artifacts(
        cfg=cfg,
        checkpoint=checkpoint,
        model=model,
        cluster_id_c=cluster_id_c,
        data_tc=data_tc,
        train_loader=train_loader,
        window_meta=window_meta,
        device=device,
    )
    prior = _restore_cluster_penalty_prior(
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
    allowed = prior.get("allowed_mask")
    if allowed is None:
        allowed_mask_kp = torch.ones((int(K), len(penalty_names)), device=device, dtype=torch.bool)
    else:
        allowed_mask_kp = torch.as_tensor(allowed, device=device, dtype=torch.bool)
    gate_feature_mode = _normalize_gate_feature_mode(
        str(checkpoint["meta"].get("gate_feature_mode", moe_cfg.get("gate_feature_mode", "history")))
    )

    payload: Dict[str, object] = {
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "route_dir": str(route_dir),
        "device": str(device),
        "test_read": False,
        "requested_splits": requested_splits,
        "action_margin": float(action_margin),
        "penalty_names": list(penalty_names),
        "splits": {},
    }
    for split in requested_splits:
        shape_tensors, _ = _collect_shape_samples(
            split_name=split,
            loader=loaders[split],
            eval_start=int(eval_starts[split]),
            model=model,
            gate=gate,
            pred_residual=pred_residual,
            cluster_id_c=cluster_id_c,
            K=int(K),
            moe_cfg=moe_cfg,
            device=device,
            penalty_names=penalty_names,
            penalty_fns=penalty_fns,
            penalty_scale=penalty_scale,
            allowed_mask_kp=allowed_mask_kp,
            history_anchor_cfg=anchor["history_anchor_cfg"],
            observed_history_tc=data_tc,
            input_len=int(window_meta["L"]),
            model_train_stat_adapter_pc=anchor["model_train_stat_adapter_pc"],
            model_train_stat_adapter_cfg=anchor["model_train_stat_adapter_cfg"],
            train_stat_anchor_pc=anchor["train_stat_anchor_pc"],
            train_residual_anchor_phc=anchor["train_residual_anchor_phc"],
            gate_feature_mode=gate_feature_mode,
        )
        route_tensors = _load_route_tensors(route_dir, split)
        gain_bkp = shape_tensors["gains"].to(dtype=torch.float32)
        B, K_shape, _ = [int(v) for v in gain_bkp.shape]
        labels = route_tensors["labels"].detach().cpu().to(dtype=torch.long).view(B, K_shape)
        current = route_tensors["current_pred"].detach().cpu().to(dtype=torch.long).view(B, K_shape)
        payload["splits"][split] = _selected_penalty_gain_summary(
            split=split,
            gain_bkp=gain_bkp,
            labels_bk=labels,
            current_pred_bk=current,
            penalty_names=list(penalty_names),
            action_margin=float(action_margin),
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "selected_penalty_gain_diagnostic.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose MSE gain of actually selected penalty routes without reading test.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--route-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--splits", nargs="*", default=["train_fit", "train_holdout", "val"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--action-margin", type=float, default=0.0)
    args = parser.parse_args()
    payload = run_selected_penalty_gain_diagnostic(
        config_path=Path(args.config),
        checkpoint_path=Path(args.checkpoint),
        route_dir=Path(args.route_dir),
        out_dir=Path(args.out_dir),
        splits=args.splits,
        device_arg=args.device,
        action_margin=float(args.action_margin),
    )
    print(f"[selected-gain] wrote {Path(args.out_dir) / 'selected_penalty_gain_diagnostic.json'}")
    for split, row in (payload.get("splits", {}) or {}).items():
        print(
            "[selected-gain] {split}: selected_rate={rate:.4f} mean_gain={gain:.6g} "
            "positive_rate={pos:.4f} false_adopt={false:.4f}".format(
                split=split,
                rate=float(row.get("selected_rate", 0.0)),
                gain=float(row.get("selected_gain_mean_on_selected", 0.0)),
                pos=float(row.get("selected_gain_positive_rate", 0.0)),
                false=float(row.get("selected_false_adopt_rate", 0.0)),
            )
        )


if __name__ == "__main__":
    main()
