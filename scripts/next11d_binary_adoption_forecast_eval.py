from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.next11c_route_accuracy_diagnostic import _build_anchor_artifacts
from scripts.next11d_binary_adoption_refit import _forecast_metrics_from_route_predictions, _to_jsonable
from scripts.next11d_fixed_candidate_router_refit import _read_data_for_cfg
from scripts.shape_prior_diagnostic import _build_modules, _make_loaders
from src.train import _collect_pred_residual_selector_tensors
from src.utils.seed import set_seed
from src.utils.yaml_io import load_yaml


def _pct_delta(new: float, old: Optional[float]) -> Optional[float]:
    if old is None or abs(float(old)) <= 1.0e-12:
        return None
    return float(100.0 * (float(new) - float(old)) / abs(float(old)))


def _load_json(path: Optional[Path]) -> Dict[str, object]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _route_tensor_path(tensors_dir: Path, split: str) -> Path:
    suffix = "train_fit" if split == "train" else split
    return tensors_dir / f"fixed_candidate_route_tensors_{suffix}.pt"


def _route_pred_bk(pred_flat: torch.Tensor, B: int, K: int, *, name: str) -> torch.Tensor:
    pred = pred_flat.detach().cpu().to(dtype=torch.long).view(-1)
    expected = int(B) * int(K)
    if int(pred.numel()) != expected:
        raise ValueError(f"{name} route prediction length {int(pred.numel())} does not match B*K={expected}.")
    return pred.reshape(int(B), int(K))


def _oracle_channel_metrics(tensors: Dict[str, torch.Tensor]) -> Dict[str, object]:
    base = tensors["base"].detach().cpu().to(dtype=torch.float32)
    cand = tensors["cand"].detach().cpu().to(dtype=torch.float32)
    y = tensors["y"].detach().cpu().to(dtype=torch.float32)
    base_err_bc = (base - y).pow(2).mean(dim=-1)
    cand_err_bcp = (cand - y.unsqueeze(2)).pow(2).mean(dim=-1)
    cand_best_err_bc, cand_best_p_bc = cand_err_bcp.min(dim=-1)
    use_candidate_bc = cand_best_err_bc < base_err_bc
    selected = base.clone()
    for p in range(int(cand.shape[2])):
        mask = use_candidate_bc & (cand_best_p_bc == p)
        if bool(mask.any().item()):
            selected = torch.where(mask.unsqueeze(-1), cand[:, :, p, :], selected)
    base_mse = float((base - y).pow(2).mean().item())
    selected_mse = float((selected - y).pow(2).mean().item())
    base_mae = float((base - y).abs().mean().item())
    selected_mae = float((selected - y).abs().mean().item())
    return {
        "base_mse": base_mse,
        "base_mae": base_mae,
        "selected_mse": selected_mse,
        "selected_mae": selected_mae,
        "selected_gain_pct_vs_base": float(100.0 * (base_mse - selected_mse) / max(abs(base_mse), 1.0e-12)),
        "selected_mae_gain_pct_vs_base": float(100.0 * (base_mae - selected_mae) / max(abs(base_mae), 1.0e-12)),
        "candidate_use_rate_channel": float(use_candidate_bc.to(dtype=torch.float32).mean().item()),
    }


def _markdown_report(payload: Dict[str, object]) -> str:
    lines = [
        "# NEXT-11d Binary Adoption Forecast Eval",
        "",
        f"- config: `{payload['config_path']}`",
        f"- checkpoint: `{payload['checkpoint_path']}`",
        f"- route_dir: `{payload['route_dir']}`",
        f"- no_test_read: `{payload['no_test_read']}`",
        "",
        "## Split Forecast Metrics",
        "",
        "| split | route | mse | mae | gain vs base | mae gain vs base | skip/use |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    splits = payload.get("splits", {}) or {}
    for split, split_payload in splits.items():
        if not isinstance(split_payload, dict):
            continue
        for route_name in ("binary", "current", "label_oracle", "channel_oracle"):
            metrics = split_payload.get(route_name)
            if not isinstance(metrics, dict):
                continue
            skip_or_use_rate = metrics.get("skip_rate_cluster", metrics.get("candidate_use_rate_channel", 0.0))
            lines.append(
                "| {split} | {route} | {mse:.6f} | {mae:.6f} | {gain:.3f}% | {mae_gain:.3f}% | {skip:.3f} |".format(
                    split=split,
                    route=route_name,
                    mse=float(metrics.get("selected_mse", metrics.get("base_mse", 0.0))),
                    mae=float(metrics.get("selected_mae", metrics.get("base_mae", 0.0))),
                    gain=float(metrics.get("selected_gain_pct_vs_base", 0.0)),
                    mae_gain=float(metrics.get("selected_mae_gain_pct_vs_base", 0.0)),
                    skip=float(skip_or_use_rate),
                )
            )
    val = (splits.get("val", {}) or {}) if isinstance(splits, dict) else {}
    if isinstance(val, dict) and isinstance(val.get("reference_deltas"), dict):
        lines.extend(["", "## Val Reference Deltas", ""])
        for name, row in val["reference_deltas"].items():
            if isinstance(row, dict):
                lines.append(
                    "- {name}: mse_delta={mse}, mae_delta={mae}".format(
                        name=name,
                        mse=row.get("mse_delta_pct"),
                        mae=row.get("mae_delta_pct"),
                    )
                )
    return "\n".join(lines) + "\n"


def run_eval(args: argparse.Namespace) -> Dict[str, object]:
    cfg = load_yaml(str(args.config))
    cfg.setdefault("eval", {})
    cfg["eval"]["skip_test"] = True
    set_seed(int(cfg["exp"]["seed"]), deterministic=bool((cfg.get("exp", {}) or {}).get("deterministic", False)))
    requested_device = str(args.device or cfg.get("exp", {}).get("device", "cpu"))
    device = torch.device(requested_device if torch.cuda.is_available() and requested_device != "cpu" else "cpu")
    checkpoint = torch.load(str(args.checkpoint), map_location=device)
    model, _, pred_residual, cluster_id_c, K, moe_cfg, penalty_names = _build_modules(cfg, checkpoint, device)
    data_tc = _read_data_for_cfg(cfg)
    batch_size = int(cfg.get("train", {}).get("batch_size", 64))
    data_window_tc, loaders, eval_starts, train_loader, window_meta = _make_loaders(cfg, data_tc, batch_size=batch_size)
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

    route_dir = Path(args.route_dir)
    route_payload = _load_json(route_dir / "binary_adoption_refit.json")
    tensors_dir_raw = args.route_tensors_dir
    if tensors_dir_raw is None and route_payload:
        tensors_dir_raw = Path(str(route_payload.get("tensors_dir", "")))
    if tensors_dir_raw is None:
        raise ValueError("--route-tensors-dir is required when route_dir has no binary_adoption_refit.json.")
    tensors_dir = Path(tensors_dir_raw)
    binary_predictions = torch.load(route_dir / "binary_adoption_predictions.pt", map_location="cpu")

    split_map = {"train_fit": "train", "train_holdout": "train_holdout", "val": "val"}
    split_payloads: Dict[str, object] = {}
    for loader_split, pred_split in split_map.items():
        tensors = _collect_pred_residual_selector_tensors(
            model=model,
            pred_residual=pred_residual,
            loader=loaders[loader_split],
            cluster_id_c=cluster_id_c,
            K=int(K),
            moe_cfg=moe_cfg,
            device=device,
            penalty_count=len(penalty_names),
            history_anchor_cfg=anchor["history_anchor_cfg"],
            observed_history_tc=data_window_tc,
            input_len=int(window_meta["L"]),
            eval_start=int(eval_starts[loader_split]),
            model_train_stat_adapter_pc=anchor["model_train_stat_adapter_pc"],
            model_train_stat_adapter_cfg=anchor["model_train_stat_adapter_cfg"],
            train_stat_anchor_pc=anchor["train_stat_anchor_pc"],
            train_residual_anchor_phc=anchor["train_residual_anchor_phc"],
            candidate_feature_mode=str(route_payload.get("route_feature_mode", "shape_proxy") if route_payload else "shape_proxy"),
        )
        if tensors is None:
            raise RuntimeError(f"Could not collect candidate tensors for split {loader_split}.")
        B = int(tensors["base"].shape[0])
        route_tensors = torch.load(_route_tensor_path(tensors_dir, pred_split), map_location="cpu")
        binary_bk = _route_pred_bk(binary_predictions[pred_split], B, int(K), name=f"binary/{pred_split}")
        current_bk = _route_pred_bk(route_tensors["current_pred"], B, int(K), name=f"current/{pred_split}")
        label_bk = _route_pred_bk(route_tensors["labels"], B, int(K), name=f"labels/{pred_split}")
        split_payload = {
            "binary": _forecast_metrics_from_route_predictions(
                base_bch=tensors["base"],
                cand_bcpH=tensors["cand"],
                y_bch=tensors["y"],
                cluster_id_c=cluster_id_c.detach().cpu(),
                route_pred_bk=binary_bk,
            ),
            "current": _forecast_metrics_from_route_predictions(
                base_bch=tensors["base"],
                cand_bcpH=tensors["cand"],
                y_bch=tensors["y"],
                cluster_id_c=cluster_id_c.detach().cpu(),
                route_pred_bk=current_bk,
            ),
            "label_oracle": _forecast_metrics_from_route_predictions(
                base_bch=tensors["base"],
                cand_bcpH=tensors["cand"],
                y_bch=tensors["y"],
                cluster_id_c=cluster_id_c.detach().cpu(),
                route_pred_bk=label_bk,
            ),
            "channel_oracle": _oracle_channel_metrics(tensors),
        }
        split_payloads[pred_split] = split_payload

    references = {
        "current_route_ce": {
            "mse": None if args.reference_current_mse is None else float(args.reference_current_mse),
            "mae": None if args.reference_current_mae is None else float(args.reference_current_mae),
        },
        "trainnoskip": {
            "mse": None if args.reference_trainnoskip_mse is None else float(args.reference_trainnoskip_mse),
            "mae": None if args.reference_trainnoskip_mae is None else float(args.reference_trainnoskip_mae),
        },
        "anchored_base": {
            "mse": None if args.reference_base_mse is None else float(args.reference_base_mse),
            "mae": None if args.reference_base_mae is None else float(args.reference_base_mae),
        },
    }
    val_payload = split_payloads.get("val", {})
    if isinstance(val_payload, dict) and isinstance(val_payload.get("binary"), dict):
        binary_val = val_payload["binary"]
        val_payload["reference_deltas"] = {
            name: {
                "mse_delta_pct": _pct_delta(float(binary_val["selected_mse"]), ref.get("mse")),
                "mae_delta_pct": _pct_delta(float(binary_val["selected_mae"]), ref.get("mae")),
            }
            for name, ref in references.items()
            if ref.get("mse") is not None or ref.get("mae") is not None
        }

    payload = {
        "config_path": str(args.config),
        "checkpoint_path": str(args.checkpoint),
        "route_dir": str(route_dir),
        "route_tensors_dir": str(tensors_dir),
        "out_dir": str(args.out_dir),
        "device": str(device),
        "no_test_read": True,
        "splits_collected": list(split_map.keys()),
        "penalty_names": list(penalty_names),
        "references": references,
        "splits": split_payloads,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "binary_adoption_forecast_eval.json").write_text(
        json.dumps(_to_jsonable(payload), indent=2),
        encoding="utf-8",
    )
    (out_dir / "binary_adoption_forecast_eval.md").write_text(_markdown_report(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="NEXT-11d forecast eval for offline binary adoption routes.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--route-dir", type=Path, required=True)
    parser.add_argument("--route-tensors-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--reference-current-mse", type=float, default=None)
    parser.add_argument("--reference-current-mae", type=float, default=None)
    parser.add_argument("--reference-trainnoskip-mse", type=float, default=None)
    parser.add_argument("--reference-trainnoskip-mae", type=float, default=None)
    parser.add_argument("--reference-base-mse", type=float, default=None)
    parser.add_argument("--reference-base-mae", type=float, default=None)
    args = parser.parse_args()
    payload = run_eval(args)
    val_binary = payload["splits"]["val"]["binary"]  # type: ignore[index]
    print(
        "val_binary_mse={:.6f} val_binary_mae={:.6f} gain_vs_base={:.3f}% no_test_read={}".format(
            float(val_binary["selected_mse"]),
            float(val_binary["selected_mae"]),
            float(val_binary["selected_gain_pct_vs_base"]),
            bool(payload["no_test_read"]),
        )
    )


if __name__ == "__main__":
    main()
