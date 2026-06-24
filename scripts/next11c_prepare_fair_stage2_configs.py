import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import yaml


VARIANTS = ["a_backbone_eval", "b_anchors", "d_moe_only_no_anchors", "c_full"]


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _write_yaml(path: Path, cfg: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")


def _nested(cfg: Dict[str, Any], *keys: str) -> Dict[str, Any]:
    cur = cfg
    for key in keys:
        cur = cur.setdefault(key, {})
        if not isinstance(cur, dict):
            raise TypeError(f"Expected mapping at {'.'.join(keys)}")
    return cur


def _localize_paths(cfg: Dict[str, Any], run_dir: Path, *, save_checkpoint: bool, skip_test: bool) -> None:
    _nested(cfg, "exp")["out_dir"] = str(run_dir)
    _nested(cfg, "eval")["skip_test"] = bool(skip_test)
    memory = _nested(cfg, "memory")
    memory["save_checkpoint"] = bool(save_checkpoint)
    memory["checkpoint_path"] = str(run_dir / "best_checkpoint.pt")
    if isinstance(cfg.get("corr"), dict):
        cfg["corr"]["save_path"] = str(run_dir / "corr.pt")
    if isinstance(cfg.get("portrait"), dict):
        cfg["portrait"]["out_dir"] = str(run_dir / "cluster_portraits")
def _set_stage2_diagnostics(cfg: Dict[str, Any]) -> None:
    _nested(cfg, "diagnostics", "stage2_loss_audit")["enable"] = True


def _set_ablation_flags(cfg: Dict[str, Any], *, moe: bool, stat_anchor: bool, residual_anchor: bool, pred_residual: bool) -> None:
    moe_cfg = _nested(cfg, "moe")
    moe_cfg["enable"] = bool(moe)
    moe_cfg["freeze_backbone"] = True
    _nested(cfg, "moe", "train_stat_anchor_expert")["enable"] = bool(stat_anchor)
    _nested(cfg, "moe", "train_residual_anchor_expert")["enable"] = bool(residual_anchor)
    _nested(cfg, "moe", "pred_side_residual")["enable"] = bool(pred_residual)


def _set_eval_only_schedule(cfg: Dict[str, Any]) -> None:
    _nested(cfg, "train")["epochs"] = 0
    _nested(cfg, "train")["penalty_warmup_epochs"] = 0
    _nested(cfg, "train")["lr_warmup_epochs"] = 0
    lr_scheduler = _nested(cfg, "train").get("lr_scheduler")
    if isinstance(lr_scheduler, dict):
        lr_scheduler["warmup_epochs"] = 0
    _nested(cfg, "train")["model_selection_start_epoch"] = 1
    _nested(cfg, "early_stop")["patience"] = 1


def _set_fair_stage2_schedule(
    cfg: Dict[str, Any],
    *,
    epochs: int,
    patience: int,
    penalty_warmup_epochs: int,
) -> None:
    train = _nested(cfg, "train")
    train["epochs"] = int(epochs)
    train["penalty_warmup_epochs"] = int(penalty_warmup_epochs)
    train["lr_warmup_epochs"] = 0
    lr_scheduler = train.get("lr_scheduler")
    if isinstance(lr_scheduler, dict):
        lr_scheduler["warmup_epochs"] = 0
    warmup = int(train.get("penalty_warmup_epochs", 0) or 0)
    train["model_selection_start_epoch"] = max(1, warmup + 1)
    _nested(cfg, "early_stop")["patience"] = int(patience)


def _source_configs(source_root: Path, cell: str) -> Dict[str, Dict[str, Any]]:
    cell_root = source_root / cell
    no_anchor = _load_yaml(cell_root / "moe_only_no_anchors.yaml")
    anchors = _load_yaml(cell_root / "anchors.yaml")
    full_path = cell_root / "full.yaml"
    full = _load_yaml(full_path) if full_path.exists() else deepcopy(anchors)
    if not full_path.exists():
        full["moe"]["pred_side_residual"] = deepcopy(no_anchor["moe"]["pred_side_residual"])
    return {"no_anchor": no_anchor, "anchors": anchors, "full": full}


def _checkpoint_path(cfg: Dict[str, Any]) -> str:
    finetune = cfg.get("finetune", {}) or {}
    return str(finetune.get("checkpoint_path", ""))


def generate_cell(
    source_root: Path,
    out_root: Path,
    cell: str,
    *,
    stage2_epochs: int,
    patience: int,
    penalty_warmup_epochs: int,
    skip_test: bool,
) -> List[Dict[str, Any]]:
    src = _source_configs(source_root, cell)
    checkpoint_path = _checkpoint_path(src["no_anchor"])
    rows: List[Dict[str, Any]] = []
    variant_cfgs = {
        "a_backbone_eval": deepcopy(src["no_anchor"]),
        "b_anchors": deepcopy(src["anchors"]),
        "d_moe_only_no_anchors": deepcopy(src["no_anchor"]),
        "c_full": deepcopy(src["full"]),
    }
    flags = {
        "a_backbone_eval": dict(moe=False, stat_anchor=False, residual_anchor=False, pred_residual=False),
        "b_anchors": dict(moe=True, stat_anchor=True, residual_anchor=True, pred_residual=False),
        "d_moe_only_no_anchors": dict(moe=True, stat_anchor=False, residual_anchor=False, pred_residual=True),
        "c_full": dict(moe=True, stat_anchor=True, residual_anchor=True, pred_residual=True),
    }
    for variant in VARIANTS:
        cfg = variant_cfgs[variant]
        if checkpoint_path:
            _nested(cfg, "finetune")["checkpoint_path"] = checkpoint_path
        _set_ablation_flags(cfg, **flags[variant])
        _set_stage2_diagnostics(cfg)
        is_stage2_train = variant in {"d_moe_only_no_anchors", "c_full"}
        if is_stage2_train:
            _set_fair_stage2_schedule(
                cfg,
                epochs=stage2_epochs,
                patience=patience,
                penalty_warmup_epochs=penalty_warmup_epochs,
            )
        else:
            _set_eval_only_schedule(cfg)
        run_dir = out_root / "runs" / cell / variant
        _localize_paths(cfg, run_dir, save_checkpoint=is_stage2_train, skip_test=skip_test)
        config_path = out_root / "configs" / cell / f"{variant}.yaml"
        _write_yaml(config_path, cfg)
        rows.append(
            {
                "cell": cell,
                "variant": variant,
                "config_path": str(config_path),
                "run_dir": str(run_dir),
                "checkpoint_path": _checkpoint_path(cfg),
                "epochs": int(cfg["train"]["epochs"]),
                "patience": int(cfg["early_stop"]["patience"]),
                "model_selection_start_epoch": int(cfg["train"]["model_selection_start_epoch"]),
                "penalty_warmup_epochs": int(cfg["train"].get("penalty_warmup_epochs", 0) or 0),
                "skip_test": bool(cfg["eval"]["skip_test"]),
                "save_checkpoint": bool(cfg["memory"]["save_checkpoint"]),
                "moe_enable": bool(cfg["moe"]["enable"]),
                "train_stat_anchor": bool(cfg["moe"]["train_stat_anchor_expert"]["enable"]),
                "train_residual_anchor": bool(cfg["moe"]["train_residual_anchor_expert"]["enable"]),
                "pred_side_residual": bool(cfg["moe"]["pred_side_residual"]["enable"]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare NEXT-11c fair Stage-2 ablation configs.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--cells", nargs="+", default=["ETTm2_H96", "ETTh1_H96"])
    parser.add_argument("--stage2-epochs", type=int, default=20)
    parser.add_argument("--cell-stage2-epochs", nargs="*", default=[], help="Optional CELL=EPOCH overrides.")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--penalty-warmup-epochs", type=int, default=0)
    parser.add_argument("--read-test", action="store_true", help="Set eval.skip_test:false for the frozen test-once run.")
    args = parser.parse_args()

    cell_epoch_overrides: Dict[str, int] = {}
    for raw in args.cell_stage2_epochs:
        if "=" not in raw:
            raise ValueError(f"Expected CELL=EPOCH override, got {raw!r}")
        cell_name, epoch_s = raw.split("=", 1)
        cell_epoch_overrides[cell_name] = int(epoch_s)

    all_rows: List[Dict[str, Any]] = []
    for cell in args.cells:
        all_rows.extend(
            generate_cell(
                args.source_root,
                args.out_root,
                cell,
                stage2_epochs=cell_epoch_overrides.get(cell, args.stage2_epochs),
                patience=args.patience,
                penalty_warmup_epochs=args.penalty_warmup_epochs,
                skip_test=not args.read_test,
            )
        )
    by_cell: Dict[str, List[str]] = {}
    for row in all_rows:
        by_cell.setdefault(row["cell"], []).append(row["checkpoint_path"])
    for cell, paths in by_cell.items():
        if len(set(paths)) != 1:
            raise RuntimeError(f"Generated configs for {cell} do not share one checkpoint: {paths}")
    manifest_path = args.out_root / "fair_stage2_manifest.json"
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"rows": all_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
