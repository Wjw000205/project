from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.shape_prior_diagnostic import _build_modules, _compute_penalty_scale, _make_loaders
from scripts.next11c_route_accuracy_diagnostic import _build_anchor_artifacts, _restore_cluster_penalty_prior
from src.data.reader import read_csv_time_series
from src.models.penalties import build_penalty_bank
from src.train import (
    _build_gate_routing_features,
    _cluster_route_oracle_labels_from_candidates,
    _normalize_gate_feature_mode,
    _parameter_grad_l2_norm,
    _pred_residual_candidates_on_eval_path,
    _route_accuracy_summary_from_labels,
    _route_ce_loss_from_probs,
    _route_probs_with_skip_class,
    _router_penalty_context_from_history,
    apply_history_anchor_adapter,
    apply_train_stat_anchor_expert,
    apply_train_stat_input_centering,
)
from src.utils.seed import set_seed
from src.utils.yaml_io import load_yaml


def _read_data_for_cfg(cfg: Dict[str, object]) -> torch.Tensor:
    data_cfg = cfg["data"]
    data_tc, _ = read_csv_time_series(str(data_cfg["csv_path"]), date_col=int(data_cfg.get("date_col", 0)))
    return data_tc.detach().cpu()


def _route_prediction_summaries(
    *,
    labels_bk: torch.Tensor,
    probs_bkp: torch.Tensor,
    skip_prob_bk: Optional[torch.Tensor],
    skip_bk: Optional[torch.Tensor],
    mask_bkp: torch.Tensor,
    penalty_names: List[str],
    probs_include_skip_mass: bool,
) -> Dict[str, object]:
    label_names = ["skip"] + [str(name) for name in penalty_names]
    joint_pred_bk = _route_probs_with_skip_class(
        probs_bkp=probs_bkp,
        skip_prob_bk=skip_prob_bk,
        probs_include_skip_mass=probs_include_skip_mass,
    ).argmax(dim=-1)
    hard_pred_bk = (mask_bkp * probs_bkp).argmax(dim=-1).to(dtype=torch.long) + 1
    if skip_bk is not None:
        hard_pred_bk = torch.where(skip_bk > 0.5, torch.zeros_like(hard_pred_bk), hard_pred_bk)
    return {
        "joint_argmax": _route_accuracy_summary_from_labels(
            labels=labels_bk.reshape(-1),
            current_pred=joint_pred_bk.reshape(-1),
            label_names=label_names,
        ),
        "hard_route": _route_accuracy_summary_from_labels(
            labels=labels_bk.reshape(-1),
            current_pred=hard_pred_bk.reshape(-1),
            label_names=label_names,
        ),
    }


def _make_snapshot(
    *,
    step: int,
    loss_value: float,
    grad_norm: float,
    labels_bk: torch.Tensor,
    probs_bkp: torch.Tensor,
    skip_prob_bk: Optional[torch.Tensor],
    skip_bk: Optional[torch.Tensor],
    mask_bkp: torch.Tensor,
    penalty_names: List[str],
    probs_include_skip_mass: bool,
) -> Dict[str, object]:
    summaries = _route_prediction_summaries(
        labels_bk=labels_bk,
        probs_bkp=probs_bkp,
        skip_prob_bk=skip_prob_bk,
        skip_bk=skip_bk,
        mask_bkp=mask_bkp,
        penalty_names=penalty_names,
        probs_include_skip_mass=probs_include_skip_mass,
    )
    out = {
        "step": int(step),
        "loss": float(loss_value),
        "gate_grad_norm": float(grad_norm),
        "joint_argmax": summaries["joint_argmax"],
        "hard_route": summaries["hard_route"],
    }
    if skip_prob_bk is not None and int(skip_prob_bk.numel()) > 0:
        sp = skip_prob_bk.detach().cpu().reshape(-1).to(dtype=torch.float32)
        out["skip_prob"] = {
            "mean": float(sp.mean().item()),
            "max": float(sp.max().item()),
            "gt_0_5_rate": float((sp > 0.5).to(dtype=torch.float32).mean().item()),
        }
    return out


def run_probe(args: argparse.Namespace) -> Dict[str, object]:
    cfg = load_yaml(str(args.config))
    cfg.setdefault("eval", {})
    cfg["eval"]["skip_test"] = True
    set_seed(int(cfg["exp"]["seed"]), deterministic=bool((cfg.get("exp", {}) or {}).get("deterministic", False)))
    requested_device = str(args.device or cfg.get("exp", {}).get("device", "cpu"))
    device = torch.device(requested_device if torch.cuda.is_available() and requested_device != "cpu" else "cpu")
    checkpoint = torch.load(str(args.checkpoint), map_location=device)
    model, gate, pred_residual, cluster_id_c, K, moe_cfg, penalty_names = _build_modules(cfg, checkpoint, device)
    if bool(args.force_skip_competes) and bool(moe_cfg.get("allow_skip", False)):
        gate.skip_competes = True
        gate.skip_argmax_noop = True
        moe_cfg["skip_competes_with_penalties"] = True
        moe_cfg["skip_argmax_noop"] = True
    gate.noise_std = float(args.gate_noise_std)
    data_tc = _read_data_for_cfg(cfg)
    batch_size = int(cfg.get("train", {}).get("batch_size", 64))
    data_window_tc, loaders, eval_starts, train_loader, window_meta = _make_loaders(cfg, data_tc, batch_size=batch_size)
    del eval_starts
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
        allowed_mask_kp = torch.as_tensor(prior_summary["allowed_mask"], device=device, dtype=torch.bool)

    model.eval()
    pred_residual.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    for param in pred_residual.parameters():
        param.requires_grad_(False)
    for param in gate.parameters():
        param.requires_grad_(True)

    train_iter = iter(loaders["train_fit"])
    batch = None
    for _ in range(max(0, int(args.batch_index) + 1)):
        batch = next(train_iter)
    if batch is None:
        raise RuntimeError("train_fit loader produced no batch.")
    x, y, idx = batch
    x = x.to(device)
    y = y.to(device)
    idx = idx.to(device=device, dtype=torch.long)
    query_start_abs_b = idx
    cid_c = cluster_id_c.to(device=device, dtype=torch.long)
    gate_feature_mode = _normalize_gate_feature_mode(
        str(checkpoint["meta"].get("gate_feature_mode", moe_cfg.get("gate_feature_mode", "history")))
    )
    router_mode = str(moe_cfg.get("router_mode", "learned")).lower()
    router_penalty_context_weight = float(moe_cfg.get("router_penalty_context_weight", 0.0))
    router_detach_penalty_context = bool(moe_cfg.get("router_detach_penalty_context", True))
    router_penalty_context_score = str(moe_cfg.get("router_penalty_context_score", "high_violation")).lower()

    with torch.no_grad():
        x_model = apply_train_stat_input_centering(
            x,
            query_start_abs_b=query_start_abs_b,
            stat_anchor_pc=anchor["model_train_stat_adapter_pc"],
            cfg=anchor["model_train_stat_adapter_cfg"],
        )
        y_base_raw = model(x_model, cluster_id_c)
        y_base = apply_history_anchor_adapter(
            y_base_raw,
            base_pred_bch=y_base_raw,
            observed_history_tc=data_window_tc,
            query_start_abs_b=query_start_abs_b,
            input_len=int(window_meta["L"]),
            cfg=anchor["history_anchor_cfg"],
        )
        y_base = apply_train_stat_anchor_expert(
            y_base,
            base_pred_bch=y_base,
            x_bcl=x,
            query_start_abs_b=query_start_abs_b,
            input_len=int(window_meta["L"]),
            stat_anchor_pc=anchor["model_train_stat_adapter_pc"],
            cfg=anchor["model_train_stat_adapter_cfg"],
        )
        gate_feat_bkf = _build_gate_routing_features(x, y_base, cluster_id_c, int(K), mode=gate_feature_mode)
        route_pen_bkp = _router_penalty_context_from_history(
            x_bcl=x,
            yhat_base_bch=y_base,
            penalty_names=penalty_names,
            penalty_fns=penalty_fns,
            penalty_scale=penalty_scale,
            cluster_id_c=cluster_id_c,
            K=int(K),
        )
        mask_bkp, probs_bkp, skip_bk, skip_prob_bk = gate(
            gate_feat_bkf,
            straight_through=False,
            penalty_context_bkp=route_pen_bkp,
            penalty_context_mode=router_mode,
            penalty_context_weight=router_penalty_context_weight,
            penalty_context_detach=router_detach_penalty_context,
            penalty_context_score=router_penalty_context_score,
        )
        pred_out = pred_residual(
            x,
            y_base,
            cluster_id_c,
            mask_bkp,
            skip_bk=skip_bk if bool(moe_cfg.get("allow_skip", False)) else None,
        )
        y_base_final, cand_bcpH = _pred_residual_candidates_on_eval_path(
            y_base,
            pred_out,
            apply_output_anchors=True,
            x_bcl=x,
            query_start_abs_b=query_start_abs_b,
            input_len=int(window_meta["L"]),
            moe_cfg=moe_cfg,
            moe_enable=True,
            observed_history_tc=data_window_tc,
            train_stat_anchor_pc=anchor["train_stat_anchor_pc"],
            train_residual_anchor_phc=anchor["train_residual_anchor_phc"],
        )
        labels_bk = _cluster_route_oracle_labels_from_candidates(
            base_bch=y_base_final,
            cand_bcpH=cand_bcpH,
            y_bch=y,
            cluster_id_c=cid_c,
            K=int(K),
            allowed_mask_kp=allowed_mask_kp,
            min_abs_improvement=float(args.min_abs_improvement),
            min_rel_improvement=float(args.min_rel_improvement),
            min_candidate_delta_rms=float(args.min_candidate_delta_rms),
        )

    optimizer = torch.optim.Adam([p for p in gate.parameters() if p.requires_grad], lr=float(args.lr))
    history: List[Dict[str, object]] = []
    watch_steps = {0, 1, 2, 5, 10, 20, 50, 100, int(args.steps)}
    gate.train()
    for step in range(0, int(args.steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        mask_bkp, probs_bkp, skip_bk, skip_prob_bk = gate(
            gate_feat_bkf,
            straight_through=False,
            penalty_context_bkp=route_pen_bkp,
            penalty_context_mode=router_mode,
            penalty_context_weight=router_penalty_context_weight,
            penalty_context_detach=router_detach_penalty_context,
            penalty_context_score=router_penalty_context_score,
        )
        loss_bk = _route_ce_loss_from_probs(
            probs_bkp=probs_bkp,
            skip_prob_bk=skip_prob_bk if bool(moe_cfg.get("allow_skip", False)) else None,
            labels_bk=labels_bk,
            probs_include_skip_mass=bool(gate.skip_competes),
        )
        loss = loss_bk.mean()
        if step > 0:
            loss.backward()
            grad_norm = _parameter_grad_l2_norm(gate.parameters())
            optimizer.step()
        else:
            grad_norm = 0.0
        if step in watch_steps or step == int(args.steps):
            history.append(
                _make_snapshot(
                    step=step,
                    loss_value=float(loss.detach().cpu().item()),
                    grad_norm=float(grad_norm),
                    labels_bk=labels_bk,
                    probs_bkp=probs_bkp.detach(),
                    skip_prob_bk=None if skip_prob_bk is None else skip_prob_bk.detach(),
                    skip_bk=None if skip_bk is None else skip_bk.detach(),
                    mask_bkp=mask_bkp.detach(),
                    penalty_names=penalty_names,
                    probs_include_skip_mass=bool(gate.skip_competes),
                )
            )
    final = history[-1]
    oracle_skip_rate = float(final["joint_argmax"]["oracle_skip_rate"])
    final_joint_acc = float(final["joint_argmax"]["current_accuracy_all"])
    final_hard_skip = float(final["hard_route"]["actual_skip_rate"])
    pass_gate = bool(final_joint_acc >= float(args.pass_accuracy) and (oracle_skip_rate <= 0.0 or final_hard_skip > 0.0))
    payload = {
        "config_path": str(args.config),
        "checkpoint_path": str(args.checkpoint),
        "out_dir": str(args.out_dir),
        "device": str(device),
        "no_test_read": True,
        "probe": {
            "batch_source": "train_fit",
            "batch_index": int(args.batch_index),
            "steps": int(args.steps),
            "lr": float(args.lr),
            "pass_accuracy": float(args.pass_accuracy),
            "force_skip_competes": bool(args.force_skip_competes),
            "gate_noise_std": float(args.gate_noise_std),
            "min_abs_improvement": float(args.min_abs_improvement),
            "min_rel_improvement": float(args.min_rel_improvement),
            "min_candidate_delta_rms": float(args.min_candidate_delta_rms),
        },
        "route_context": {
            "penalty_names": penalty_names,
            "label_names": ["skip"] + [str(name) for name in penalty_names],
            "cluster_count": int(K),
            "allow_skip": bool(moe_cfg.get("allow_skip", False)),
            "skip_competes_with_penalties": bool(gate.skip_competes),
            "skip_argmax_noop": bool(getattr(gate, "skip_argmax_noop", False)),
            "topk": int(moe_cfg.get("topk", 1)),
            "allowed_mask": prior_summary.get("allowed_mask"),
            "prior_restored": prior_summary,
            "gate_feature_mode": gate_feature_mode,
        },
        "label_summary": _route_accuracy_summary_from_labels(
            labels=labels_bk.reshape(-1),
            current_pred=labels_bk.reshape(-1),
            label_names=["skip"] + [str(name) for name in penalty_names],
        ),
        "history": history,
        "verdict": {
            "one_batch_overfit_pass": pass_gate,
            "failure_layer_if_fail": "routing target mismatch",
            "note": "Pass requires joint-argmax route accuracy above threshold and nonzero hard skip when oracle skip exists.",
        },
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gate_overfit_probe.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = [
        "# NEXT-11d Gate One-Batch Overfit Probe",
        "",
        f"- config: `{args.config}`",
        f"- checkpoint: `{args.checkpoint}`",
        f"- no_test_read: `{payload['no_test_read']}`",
        f"- final joint accuracy: {final_joint_acc:.4f}",
        f"- final hard skip rate: {final_hard_skip:.4f}",
        f"- oracle skip rate: {oracle_skip_rate:.4f}",
        f"- pass: `{pass_gate}`",
        "",
        "## History",
        "",
        "| step | loss | joint_acc | hard_acc | oracle_skip | hard_skip | grad_norm |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in history:
        md.append(
            "| {step} | {loss:.6f} | {joint:.4f} | {hard:.4f} | {oracle_skip:.4f} | {hard_skip:.4f} | {grad:.6f} |".format(
                step=int(row["step"]),
                loss=float(row["loss"]),
                joint=float(row["joint_argmax"]["current_accuracy_all"]),
                hard=float(row["hard_route"]["current_accuracy_all"]),
                oracle_skip=float(row["joint_argmax"]["oracle_skip_rate"]),
                hard_skip=float(row["hard_route"]["actual_skip_rate"]),
                grad=float(row["gate_grad_norm"]),
            )
        )
    (out_dir / "gate_overfit_probe.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="NEXT-11d one-batch gate route overfit probe.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--pass-accuracy", type=float, default=0.90)
    parser.add_argument("--gate-noise-std", type=float, default=0.0)
    parser.add_argument("--min-abs-improvement", type=float, default=0.0)
    parser.add_argument("--min-rel-improvement", type=float, default=0.0)
    parser.add_argument("--min-candidate-delta-rms", type=float, default=0.0)
    parser.add_argument("--force-skip-competes", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    payload = run_probe(args)
    final = payload["history"][-1]
    print(
        "final_joint_acc={:.4f} hard_acc={:.4f} oracle_skip={:.4f} hard_skip={:.4f} pass={}".format(
            float(final["joint_argmax"]["current_accuracy_all"]),
            float(final["hard_route"]["current_accuracy_all"]),
            float(final["joint_argmax"]["oracle_skip_rate"]),
            float(final["hard_route"]["actual_skip_rate"]),
            bool(payload["verdict"]["one_batch_overfit_pass"]),
        )
    )


if __name__ == "__main__":
    main()
