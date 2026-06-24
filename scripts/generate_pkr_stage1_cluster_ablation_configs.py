from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "outputs" / "pkr_moe_cluster_learning_ablation"

SOURCE_CONFIGS = {
    "ETTh1": ROOT
    / "outputs"
    / "fresh_input_len96_20260610_etth1_ettm1_backbone_probe"
    / "configs"
    / "ETTh1"
    / "H96"
    / "common_backbone_h96"
    / "mlp_h128_do0_wd1e4_mae04.yaml",
    "ETTm1": ROOT
    / "outputs"
    / "fresh_input_len96_20260610_ettm1_seasonal_blend_m010_full"
    / "configs"
    / "ETTm1"
    / "H96"
    / "light_backbone"
    / "mlp_anchor_basis_seasblend_m010_h256_r16_wd1e4_mae06.yaml",
}

CLUSTER_EMBEDDING_DEFAULT = {
    "enable": False,
    "dim": 8,
    "mode": "film",
    "film_scale": 0.1,
    "init_std": 0.02,
    "film_init": "zero",
}

PER_CLUSTER_MAE_DEFAULT = {
    "enable": False,
    "diagnostic": "mean_median_gap",
    "source": "train_targets",
    "normalize": "std",
    "pivot": "median",
    "max_multiplier": 1.25,
    "min_multiplier": 1.0,
    "max_windows": 0,
    "artifact": "cluster_mae_weights.csv",
}

VARIANTS = {
    "baseline": (False, False),
    "A": (True, False),
    "B": (False, True),
    "A_B": (True, True),
}


def load_yaml(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, cfg: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)


def configure_variant(dataset: str, source_cfg: Dict[str, object], variant: str) -> Dict[str, object]:
    enable_a, enable_b = VARIANTS[variant]
    cfg = copy.deepcopy(source_cfg)
    run_dir = OUT_ROOT / "runs" / dataset / "H96" / variant

    cfg.setdefault("exp", {})["name"] = f"{dataset}_H96_stage1_cluster_learning_{variant}"
    cfg["exp"]["out_dir"] = str(run_dir)

    cfg.setdefault("corr", {})["save_path"] = str(run_dir / "corr.npy")
    cfg.setdefault("eval", {})["skip_test"] = True

    cfg.setdefault("moe", {})["enable"] = False
    cfg["moe"].pop("freeze_backbone", None)
    cfg.pop("finetune", None)

    cfg.setdefault("memory", {})["enable"] = False
    cfg["memory"]["save_checkpoint"] = True
    cfg["memory"]["path"] = str(run_dir / "cluster_memory.pt")
    cfg["memory"]["checkpoint_path"] = str(run_dir / "best_checkpoint.pt")

    cfg.setdefault("portrait", {})["out_dir"] = str(run_dir / "cluster_portraits")

    cluster_embedding = dict(CLUSTER_EMBEDDING_DEFAULT)
    cluster_embedding["enable"] = bool(enable_a)
    cfg.setdefault("model", {})["cluster_embedding"] = cluster_embedding

    train_cfg = cfg.setdefault("train", {})
    mae_cfg = train_cfg.setdefault("mae_objective", {})
    per_cluster = dict(PER_CLUSTER_MAE_DEFAULT)
    per_cluster["enable"] = bool(enable_b)
    mae_cfg["per_cluster"] = per_cluster
    return cfg


def main() -> None:
    written = []
    for dataset, source_path in SOURCE_CONFIGS.items():
        source_cfg = load_yaml(source_path)
        for variant in VARIANTS:
            cfg = configure_variant(dataset, source_cfg, variant)
            out_path = OUT_ROOT / "configs" / dataset / "H96" / f"{variant}.yaml"
            write_yaml(out_path, cfg)
            written.append(out_path)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
