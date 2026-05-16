from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.cluster_memory import load_cluster_checkpoint  # noqa: E402


FIELDNAMES = [
    "status",
    "source",
    "target",
    "source_summary",
    "source_test_mse",
    "source_test_mae",
    "source_val_mse",
    "source_val_mae",
    "target_config",
    "base_config",
    "target_pred_len_adjusted",
    "target_original_pred_len",
    "input_len",
    "pred_len",
    "data_max_rows",
    "train_ratio",
    "val_ratio",
    "test_ratio",
    "normalize_train_only",
    "past_context",
    "direct_mse",
    "direct_mae",
    "direct_route_uses_train_only",
    "direct_num_windows",
    "direct_eval_start",
    "direct_eval_label_start",
    "direct_eval_end",
    "val_route_mse",
    "val_route_mae",
    "val_route_selected_val_mse",
    "val_route_selected_val_mae",
    "val_route",
    "val_route_uses_train_only",
    "val_route_num_windows",
    "search_mode",
    "out_dir",
    "error",
]


SOURCE_CONFIGS = {
    "ETTm1": {
        "base": ROOT / "configs" / "ETTm1ToETTm2.yaml",
        "targets": ["ETTh1", "ETTh2", "ETTm2", "weather", "traffic"],
    },
    "ETTm2": {
        "base": ROOT / "configs" / "ETTm2ToETTm1.yaml",
        "targets": ["ETTh1", "ETTh2", "ETTm1", "weather", "traffic"],
    },
}


TARGET_CONFIGS = {
    "ETTh1": ROOT / "configs" / "ETTh1.yaml",
    "ETTh2": ROOT / "configs" / "ETTh2.yaml",
    "ETTm1": ROOT / "outputs" / "ett_horizon_sweep" / "configs" / "ETTm1_pred_96.yaml",
    "ETTm2": ROOT / "configs" / "ETTm2.yaml",
    "weather": ROOT / "configs" / "weather.yaml",
    "traffic": ROOT / "configs" / "traffic.yaml",
}


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})


def run_cmd(cmd: list[str]) -> str:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout)
    return proc.stdout


def checkpoint_window(source_cfg: dict[str, Any]) -> tuple[int, int]:
    ckpt = load_cluster_checkpoint(str(source_cfg["source"]["checkpoint_path"]), device="cpu")
    meta = ckpt["meta"]
    return int(meta["input_len"]), int(meta["pred_len"])


def source_metrics(source_cfg: dict[str, Any]) -> dict[str, Any]:
    summary_path = Path(str(source_cfg["source"]["summary_path"]))
    if not summary_path.is_absolute():
        summary_path = ROOT / summary_path
    summary = load_json(summary_path)
    return {
        "source_summary": str(summary_path),
        "source_test_mse": summary.get("test", {}).get("avg_mse", ""),
        "source_test_mae": summary.get("test", {}).get("avg_mae", ""),
        "source_val_mse": summary.get("val", {}).get("avg_mse", ""),
        "source_val_mae": summary.get("val", {}).get("avg_mae", ""),
    }


def build_aligned_config(
    *,
    source: str,
    target: str,
    out_dir: Path,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base = read_yaml(SOURCE_CONFIGS[source]["base"])
    target_cfg_path = TARGET_CONFIGS[target]
    target_cfg = read_yaml(target_cfg_path)
    ckpt_input_len, ckpt_pred_len = checkpoint_window(base)

    data_cfg = dict(target_cfg.get("data", {}) or {})
    window_cfg = dict(target_cfg.get("window", {}) or {})
    original_pred_len = int(window_cfg.get("pred_len", ckpt_pred_len))

    cfg = json.loads(json.dumps(base))
    cfg.setdefault("exp", {})["name"] = f"{source}_to_{target}_aligned_h96"
    cfg["exp"]["out_dir"] = str(out_dir / "direct_transfer")
    cfg["exp"]["device"] = device

    cfg["data"] = {
        "csv_path": data_cfg.get("csv_path", cfg.get("data", {}).get("csv_path")),
        "date_col": data_cfg.get("date_col", cfg.get("data", {}).get("date_col", 0)),
        "train_ratio": data_cfg.get("train_ratio", cfg.get("data", {}).get("train_ratio", 0.7)),
        "val_ratio": data_cfg.get("val_ratio", cfg.get("data", {}).get("val_ratio", 0.1)),
        "test_ratio": data_cfg.get("test_ratio", cfg.get("data", {}).get("test_ratio", 0.2)),
    }
    if "max_rows" in data_cfg:
        cfg["data"]["max_rows"] = data_cfg["max_rows"]

    cfg["window"] = {
        "input_len": ckpt_input_len,
        "pred_len": ckpt_pred_len,
    }
    if "past_context" in window_cfg:
        cfg["window"]["past_context"] = bool(window_cfg.get("past_context", False))

    cfg["normalize"] = dict(target_cfg.get("normalize", cfg.get("normalize", {})) or {})
    cfg.setdefault("transfer", {}).setdefault("knn_hybrid", {})["enable"] = False
    cfg.setdefault("eval", {})["batch_size"] = int(cfg.get("eval", {}).get("batch_size", 512))
    cfg["eval"]["split"] = "test"

    resample_cfg = cfg.setdefault("transfer", {}).setdefault("resample", {})
    if target in {"ETTh1", "ETTh2", "weather", "traffic"}:
        resample_cfg["enable"] = True
        resample_cfg["target_step_minutes"] = int(base.get("source", {}).get("step_minutes", 15))
        resample_cfg.setdefault("method", "linear")
    else:
        resample_cfg["enable"] = False

    meta = {
        "target_config": str(target_cfg_path),
        "target_pred_len_adjusted": original_pred_len != ckpt_pred_len,
        "target_original_pred_len": original_pred_len,
        "input_len": ckpt_input_len,
        "pred_len": ckpt_pred_len,
    }
    return cfg, meta


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_pair(
    *,
    source: str,
    target: str,
    out_root: Path,
    device: str,
    py: str,
    batch_size: int,
    max_greedy_channels: int,
) -> dict[str, Any]:
    pair_dir = out_root / f"{source}_to_{target}"
    cfg, meta = build_aligned_config(source=source, target=target, out_dir=pair_dir, device=device)
    cfg.setdefault("eval", {})["batch_size"] = batch_size
    cfg_path = pair_dir / "base_config.yaml"
    write_yaml(cfg_path, cfg)

    row: dict[str, Any] = {
        "status": "ok",
        "source": source,
        "target": target,
        **source_metrics(cfg),
        "target_config": meta["target_config"],
        "base_config": str(cfg_path),
        "target_pred_len_adjusted": meta["target_pred_len_adjusted"],
        "target_original_pred_len": meta["target_original_pred_len"],
        "input_len": meta["input_len"],
        "pred_len": meta["pred_len"],
        "data_max_rows": cfg.get("data", {}).get("max_rows", 0),
        "train_ratio": cfg["data"]["train_ratio"],
        "val_ratio": cfg["data"]["val_ratio"],
        "test_ratio": cfg["data"]["test_ratio"],
        "normalize_train_only": cfg.get("normalize", {}).get("train_only", ""),
        "past_context": cfg.get("window", {}).get("past_context", False),
        "out_dir": str(pair_dir),
    }

    run_cmd([py, "-u", "-m", "src.transfer", "--config", str(cfg_path)])
    direct = load_json(pair_dir / "direct_transfer" / "transfer_summary.json")
    row.update(
        {
            "direct_mse": direct["avg_mse"],
            "direct_mae": direct["avg_mae"],
            "direct_route_uses_train_only": direct.get("route_uses_train_only", ""),
            "direct_num_windows": direct.get("num_eval_windows", ""),
            "direct_eval_start": direct.get("eval_start_index", ""),
            "direct_eval_label_start": direct.get("eval_label_start_index", ""),
            "direct_eval_end": direct.get("eval_end_index", ""),
        }
    )

    selection_dir = pair_dir / "val_loss_selection"
    run_cmd(
        [
            py,
            "-u",
            "scripts/run_ettm1_to_ettm2_val_loss_route_selection.py",
            "--config",
            str(cfg_path),
            "--out-root",
            str(selection_dir),
            "--device",
            device,
            "--batch-size",
            str(batch_size),
            "--python",
            py,
            "--search-mode",
            "auto",
            "--max-greedy-channels",
            str(max_greedy_channels),
        ]
    )
    selected = load_json(selection_dir / "summary.json")
    selected_test = load_json(selection_dir / "selected_test_transfer" / "transfer_summary.json")
    row.update(
        {
            "val_route_mse": selected["selected_test_mse"],
            "val_route_mae": selected["selected_test_mae"],
            "val_route_selected_val_mse": selected["selected_val_mse"],
            "val_route_selected_val_mae": selected["selected_val_mae"],
            "val_route": json.dumps(selected["selected_route"]),
            "val_route_uses_train_only": selected_test.get("route_uses_train_only", ""),
            "val_route_num_windows": selected_test.get("num_eval_windows", ""),
            "search_mode": selected.get("search_mode", ""),
        }
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "aligned_h96_transfer_matrix")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--python", type=Path, default=Path(sys.executable))
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-greedy-channels", type=int, default=64)
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    py = str(args.python)
    for source, spec in SOURCE_CONFIGS.items():
        for target in spec["targets"]:
            print(f"=== {source} -> {target} ===", flush=True)
            try:
                row = run_pair(
                    source=source,
                    target=target,
                    out_root=args.out_root,
                    device=args.device,
                    py=py,
                    batch_size=args.batch_size,
                    max_greedy_channels=args.max_greedy_channels,
                )
            except Exception as exc:  # keep the matrix complete
                row = {
                    "status": "error",
                    "source": source,
                    "target": target,
                    "out_dir": str(args.out_root / f"{source}_to_{target}"),
                    "error": str(exc)[-4000:],
                }
            rows.append(row)
            write_rows(args.out_root / "transfer.csv", rows)
    write_rows(args.out_root / "transfer.csv", rows)
    print(f"Saved: {args.out_root / 'transfer.csv'}")


if __name__ == "__main__":
    main()
