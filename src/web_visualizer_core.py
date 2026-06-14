from __future__ import annotations

import csv
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml


DEFAULT_CONDA_BAT = r"F:\Anaconda3\condabin\conda.bat"
DEFAULT_CONDA_ENV = "my_fram"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
PROGRESS_RE = re.compile(
    r"(?P<current>\d+)\s*/\s*(?P<total>\d+)\s+"
    r"(?P<percent>\d+(?:\.\d+)?)%\s+.*?"
    r"epoch=(?P<epoch>\d+)\s*/\s*(?P<epochs>\d+)"
    r"(?:\s+batch=(?P<batch>\d+)\s*/\s*(?P<batches>\d+|\.{3}))?",
    re.IGNORECASE,
)
LOSS_RE = re.compile(r"loss=(?P<loss>-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", re.IGNORECASE)
SUMMARY_RE = re.compile(
    r"\[Epoch\s+(?P<epoch>\d+)\]\s+loss=(?P<loss>-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)"
    r"(?:\s*\|\s*val_loss=(?P<val_loss>-?\d+(?:\.\d+)?(?:e[+-]?\d+)?))?",
    re.IGNORECASE,
)


def build_training_command(
    config_path: str,
    *,
    conda_bat: str | None = None,
    env_name: str | None = None,
) -> list[str]:
    conda = conda_bat or os.environ.get("MOE_VIS_CONDA_BAT") or DEFAULT_CONDA_BAT
    env = env_name or os.environ.get("MOE_VIS_CONDA_ENV") or DEFAULT_CONDA_ENV
    return [
        str(conda),
        "run",
        "--no-capture-output",
        "-n",
        str(env),
        "python",
        "-m",
        "src.train",
        "--config",
        str(config_path),
    ]


def _clean_console_text(text: str) -> str:
    return ANSI_RE.sub("", text).replace("\r", "\n")


def clean_training_log_text(text: str) -> str:
    return _clean_console_text(text)


def _is_live_progress_line(line: str) -> bool:
    return bool(PROGRESS_RE.search(line))


def format_training_log_tail(log_text: str, *, max_lines: int = 160) -> str:
    clean = _clean_console_text(log_text)
    lines = []
    for raw_line in clean.splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        if _is_live_progress_line(line):
            continue
        lines.append(line)
    return "\n".join(lines[-int(max_lines) :])


def parse_training_progress(log_text: str) -> dict[str, Any]:
    clean = _clean_console_text(log_text)
    progress: dict[str, Any] = {}
    for raw_line in clean.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = PROGRESS_RE.search(line)
        if match:
            batch_current = int(match.group("batch")) if match.group("batch") else None
            batch_total = int(match.group("batches")) if (match.group("batches") or "").isdigit() else None
            global_total = int(match.group("total"))
            epoch_total = int(match.group("epochs"))
            if batch_current is not None and batch_total is None and epoch_total > 0:
                inferred = int(round(global_total / max(epoch_total, 1)))
                batch_total = inferred if inferred > 0 else None
            epoch_percent = None
            if batch_current is not None and batch_total:
                epoch_percent = round((batch_current / max(batch_total, 1)) * 100.0, 2)
            elif "validating" in line.lower():
                epoch_percent = 100.0
            loss_match = LOSS_RE.search(line)
            progress = {
                "phase": "validating" if "validating" in line.lower() else "training",
                "epoch_current": int(match.group("epoch")),
                "epoch_total": epoch_total,
                "batch_current": batch_current,
                "batch_total": batch_total,
                "epoch_percent": epoch_percent,
                "global_current": int(match.group("current")),
                "global_total": global_total,
                "global_percent": float(match.group("percent")),
                "loss": float(loss_match.group("loss")) if loss_match else None,
                "val_loss": None,
                "raw": line,
            }
            continue
        match = SUMMARY_RE.search(line)
        if match:
            progress = {
                "phase": "epoch_summary",
                "epoch_current": int(match.group("epoch")),
                "epoch_total": None,
                "batch_current": None,
                "batch_total": None,
                "epoch_percent": 100.0,
                "global_current": None,
                "global_total": None,
                "global_percent": None,
                "loss": float(match.group("loss")),
                "val_loss": float(match.group("val_loss")) if match.group("val_loss") else None,
                "raw": line,
            }
    return progress


def _repo_path(root: Path | str, rel_path: str | Path) -> Path:
    root_path = Path(root).resolve()
    raw = Path(str(rel_path).replace("\\", "/"))
    path = raw if raw.is_absolute() else root_path / raw
    path = path.resolve()
    if path != root_path and root_path not in path.parents:
        raise ValueError(f"path escapes repository root: {rel_path}")
    return path


def _rel_id(root: Path | str, path: Path) -> str:
    return path.resolve().relative_to(Path(root).resolve()).as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _dataset_from_cfg(cfg: dict[str, Any], fallback: str) -> str:
    csv_path = str((cfg.get("data") or {}).get("csv_path") or "")
    if csv_path:
        return Path(csv_path).stem
    return fallback


def _number_or_text(value: str) -> Any:
    text = value.strip()
    if text == "":
        return ""
    try:
        if any(ch in text.lower() for ch in [".", "e"]):
            number = float(text)
            return number if math.isfinite(number) else None
        return int(text)
    except ValueError:
        return text


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _read_optional_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [
            {key: _number_or_text(value) for key, value in row.items()}
            for row in csv.DictReader(f)
        ]


def discover_configs(root: Path | str) -> list[dict[str, Any]]:
    repo = Path(root).resolve()
    configs_dir = repo / "configs"
    if not configs_dir.exists():
        return []

    options: list[dict[str, Any]] = []
    for path in sorted(configs_dir.rglob("*.yaml")):
        try:
            cfg = _read_yaml(path)
        except Exception:
            continue
        exp = cfg.get("exp") or {}
        window = cfg.get("window") or {}
        train = cfg.get("train") or {}
        name = path.stem
        options.append(
            {
                "id": _rel_id(repo, path),
                "name": name,
                "dataset": _dataset_from_cfg(cfg, fallback=name.split("_")[0]),
                "input_len": int(window.get("input_len") or 0),
                "pred_len": int(window.get("pred_len") or 0),
                "epochs": int(train.get("epochs") or 0),
                "device": str(exp.get("device") or ""),
                "out_dir": str(exp.get("out_dir") or ""),
            }
        )
    return sorted(options, key=lambda item: (item["dataset"], item["pred_len"], item["name"]))


def _summary_metric(summary: dict[str, Any], metric: str) -> float | None:
    for section_name in ["selected", "test", "val"]:
        section = summary.get(section_name) or {}
        value = section.get(metric)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def _load_config_for_summary(root: Path, summary: dict[str, Any]) -> dict[str, Any]:
    config_path = summary.get("config_path")
    if not config_path:
        return {}
    try:
        path = _repo_path(root, str(config_path))
    except ValueError:
        return {}
    if not path.exists():
        return {}
    try:
        return _read_yaml(path)
    except Exception:
        return {}


def _iter_run_summaries(outputs: Path, max_runs: int) -> list[Path]:
    patterns = [
        "*/run_summary.json",
        "*/*/run_summary.json",
        "*/*/*/run_summary.json",
        "*/*/*/*/run_summary.json",
    ]
    found: dict[Path, tuple[bool, float]] = {}
    for pattern in patterns:
        for path in outputs.glob(pattern):
            try:
                run_dir = path.parent
                found[path.resolve()] = (
                    (run_dir / "prediction_intermediates.npz").exists(),
                    float(path.stat().st_mtime),
                )
            except OSError:
                continue
    ordered = sorted(found.items(), key=lambda item: item[1], reverse=True)
    return [path for path, _ in ordered[: int(max_runs)]]


def discover_runs(root: Path | str, max_runs: int = 200) -> list[dict[str, Any]]:
    repo = Path(root).resolve()
    outputs = repo / "outputs"
    if not outputs.exists():
        return []

    rows: list[dict[str, Any]] = []
    for summary_path in _iter_run_summaries(outputs, max_runs=max_runs):
        run_dir = summary_path.parent
        try:
            summary = _read_json(summary_path)
        except Exception:
            continue
        cfg = _load_config_for_summary(repo, summary)
        window = cfg.get("window") or {}
        run_id = _rel_id(repo, run_dir)
        rows.append(
            {
                "id": run_id,
                "label": run_dir.name,
                "config_path": str(summary.get("config_path") or ""),
                "dataset": _dataset_from_cfg(cfg, fallback=run_dir.name.split("_")[0]),
                "pred_len": int(window.get("pred_len") or 0),
                "avg_mse": _summary_metric(summary, "avg_mse"),
                "avg_mae": _summary_metric(summary, "avg_mae"),
                "has_prediction_intermediates": (run_dir / "prediction_intermediates.npz").exists(),
                "modified_time": float(summary_path.stat().st_mtime),
            }
        )
    rows.sort(
        key=lambda item: (bool(item["has_prediction_intermediates"]), item["modified_time"]),
        reverse=True,
    )
    return rows[: int(max_runs)]


def load_run_payload(root: Path | str, run_dir: str) -> dict[str, Any]:
    repo = Path(root).resolve()
    path = _repo_path(repo, run_dir)
    summary_path = path / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing run_summary.json: {run_dir}")

    meta_path = path / "prediction_intermediates_meta.json"
    prediction_meta = _read_json(meta_path) if meta_path.exists() else {}
    npz_path = path / "prediction_intermediates.npz"
    if npz_path.exists() and "sample_count" not in prediction_meta:
        with np.load(npz_path) as z:
            if "idx" in z.files:
                prediction_meta["sample_count"] = int(z["idx"].shape[0])

    return {
        "id": _rel_id(repo, path),
        "summary": _json_safe(_read_json(summary_path)),
        "cluster_penalty_rows": _read_optional_csv(path / "cluster_penalty_probs.csv"),
        "prediction_meta": _json_safe(prediction_meta),
        "has_prediction_intermediates": npz_path.exists(),
    }


def _array_at(z: np.lib.npyio.NpzFile, key: str, index: int) -> list[Any]:
    if key not in z.files:
        return []
    return _json_safe(z[key][index])


def load_prediction_sample(root: Path | str, run_dir: str, sample_index: int = 0) -> dict[str, Any]:
    repo = Path(root).resolve()
    path = _repo_path(repo, run_dir)
    npz_path = path / "prediction_intermediates.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"missing prediction_intermediates.npz: {run_dir}")

    meta_path = path / "prediction_intermediates_meta.json"
    meta = _read_json(meta_path) if meta_path.exists() else {}
    with np.load(npz_path) as z:
        if "idx" not in z.files:
            raise ValueError("prediction intermediates must contain idx")
        sample_count = int(z["idx"].shape[0])
        index = int(sample_index)
        if index < 0 or index >= sample_count:
            raise IndexError(f"sample_index {sample_index} outside 0..{sample_count - 1}")

        channel_count = int(z["y_final"].shape[1]) if "y_final" in z.files else 0
        channel_names = list(meta.get("channel_names") or [f"ch{c}" for c in range(channel_count)])
        if "gate_probs" in z.files:
            penalty_count = int(z["gate_probs"].shape[-1])
        else:
            penalty_count = len(meta.get("penalty_names") or [])
        penalty_names = list(meta.get("penalty_names") or [f"penalty_{p}" for p in range(penalty_count)])

        clusters = []
        if "gate_probs" in z.files:
            probs_kp = z["gate_probs"][index]
            mask_kp = z["gate_mask"][index] if "gate_mask" in z.files else np.zeros_like(probs_kp)
            skip_k = z["skip_prob"][index] if "skip_prob" in z.files else np.zeros(probs_kp.shape[0])
            for k in range(int(probs_kp.shape[0])):
                probs = [float(v) for v in probs_kp[k].tolist()]
                skip_prob = float(skip_k[k]) if k < int(skip_k.shape[0]) else 0.0
                participation = []
                for p, prob in enumerate(probs):
                    penalty_name = penalty_names[p] if p < len(penalty_names) else f"penalty_{p}"
                    selected_flag = bool(float(mask_kp[k, p]) > 0.0)
                    effective_strength = float(prob) * (1.0 - skip_prob) if selected_flag else 0.0
                    participation.append(
                        {
                            "penalty": penalty_name,
                            "selected": selected_flag,
                            "gate_prob": float(prob),
                            "effective_strength": round(effective_strength, 12),
                        }
                    )
                selected = [
                    penalty_names[p] if p < len(penalty_names) else f"penalty_{p}"
                    for p, value in enumerate(mask_kp[k].tolist())
                    if float(value) > 0.0
                ]
                top_idx = int(np.argmax(probs_kp[k])) if len(probs) > 0 else -1
                clusters.append(
                    {
                        "cluster": k,
                        "probabilities": [
                            {
                                "penalty": penalty_names[p] if p < len(penalty_names) else f"penalty_{p}",
                                "prob": float(prob),
                            }
                            for p, prob in enumerate(probs)
                        ],
                        "penalty_participation": participation,
                        "selected_penalties": selected,
                        "top_penalty": penalty_names[top_idx] if 0 <= top_idx < len(penalty_names) else "",
                        "skip_prob": skip_prob,
                    }
                )

        if "cluster_id" in z.files:
            channel_cluster_ids = [int(v) for v in z["cluster_id"].tolist()]
        else:
            channel_cluster_ids = []

        if "residual_gate_scale" in z.files:
            scale_arr = z["residual_gate_scale"][index]
            residual_gate_scale = [float(v) for v in np.asarray(scale_arr).reshape(scale_arr.shape[0], -1)[:, 0]]
        else:
            residual_gate_scale = []

        return {
            "idx": int(z["idx"][index]),
            "sample_index": index,
            "sample_count": sample_count,
            "channel_names": channel_names,
            "penalty_names": penalty_names,
            "channel_cluster_ids": channel_cluster_ids,
            "series": {
                "x": _array_at(z, "x", index),
                "y_true": _array_at(z, "y_true", index),
                "y_base": _array_at(z, "y_base", index),
                "y_residual_raw": _array_at(z, "y_residual_raw", index),
                "y_final": _array_at(z, "y_final", index),
            },
            "clusters": clusters,
            "residual_gate_scale": residual_gate_scale,
        }


def materialize_training_config(
    root: Path | str,
    config_id: str,
    *,
    run_id: str | None = None,
    pred_len: int | None = None,
    device: str | None = None,
    sample_count: int = 32,
) -> dict[str, str]:
    repo = Path(root).resolve()
    source_path = _repo_path(repo, config_id)
    cfg = _read_yaml(source_path)
    clean_run_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in (run_id or ""))
    if not clean_run_id:
        clean_run_id = f"run-{int(time.time())}"
    run_dir = repo / "outputs" / "web_visualizer" / "runs" / clean_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg.setdefault("exp", {})
    cfg["exp"]["name"] = f"web_{source_path.stem}_{clean_run_id}"
    cfg["exp"]["out_dir"] = run_dir.as_posix()
    cfg.setdefault("corr", {})
    cfg["corr"]["save_path"] = (run_dir / "corr.npy").as_posix()
    cfg.setdefault("portrait", {})
    cfg["portrait"]["out_dir"] = (run_dir / "cluster_portraits").as_posix()
    cfg.setdefault("knn_hybrid", {})
    cfg["knn_hybrid"]["path"] = (run_dir / "knn_shape_bank.pt").as_posix()
    cfg.setdefault("memory", {})
    cfg["memory"]["path"] = (run_dir / "cluster_memory.pt").as_posix()
    cfg["memory"]["checkpoint_path"] = (run_dir / "best_checkpoint.pt").as_posix()
    if device:
        cfg["exp"]["device"] = str(device)
    if pred_len is not None:
        cfg.setdefault("window", {})
        cfg["window"]["pred_len"] = int(pred_len)
    cfg.setdefault("eval", {})
    cfg["eval"]["skip_test"] = False
    cfg.setdefault("diagnostics", {})
    cfg["diagnostics"]["save_prediction_intermediates"] = True
    cfg["diagnostics"]["prediction_sample_count"] = int(sample_count)
    cfg["diagnostics"]["prediction_sample_strategy"] = "stratified_random"
    cfg["diagnostics"]["prediction_sample_seed"] = sum(
        (idx + 1) * ord(ch) for idx, ch in enumerate(clean_run_id)
    ) % 2147483647

    generated_path = run_dir / "config.yaml"
    with generated_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return {
        "config_path": generated_path.as_posix(),
        "run_dir": run_dir.as_posix(),
        "run_id": clean_run_id,
    }
