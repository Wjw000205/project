from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_REPO_URL = "git@github.com:jackyue1994/OLinear.git"
DEFAULT_REPO_ROOT = Path(r"F:\python project\OLinear")
DEFAULT_DATA_ROOT = Path(r"F:\Python program\MoELoss - 20260401\data")
DEFAULT_TARGET_MD = Path(
    r"F:\Python program\MoELoss - 20260401\outputs\experiment_excel_summary\paper_style_experiment_summary.md"
)
DEFAULT_TARGET_CSV = Path(
    r"F:\Python program\MoELoss - 20260401\outputs\experiment_excel_summary\olinear_len336_results.csv"
)
DEFAULT_PYTHON = Path(r"C:\Users\33932\.conda\envs\my_fram\python.exe")


EXPECTED_JOBS = [
    ("Electricity", 96),
    ("Electricity", 192),
    ("Electricity", 336),
    ("Electricity", 720),
    ("Weather", 96),
    ("Weather", 192),
    ("Weather", 336),
    ("Weather", 720),
    ("ETTh1", 96),
    ("ETTh1", 192),
    ("ETTh1", 336),
    ("ETTh1", 720),
    ("ETTh2", 96),
    ("ETTh2", 192),
    ("ETTh2", 336),
    ("ETTh2", 720),
    ("ETTm1", 96),
    ("ETTm1", 192),
    ("ETTm1", 336),
    ("ETTm1", 720),
    ("ETTm2", 96),
    ("ETTm2", 192),
    ("ETTm2", 336),
    ("ETTm2", 720),
    ("PEMS03", 12),
    ("PEMS03", 24),
    ("PEMS03", 48),
    ("PEMS03", 96),
    ("PEMS04", 12),
    ("PEMS04", 24),
    ("PEMS04", 48),
    ("PEMS04", 96),
    ("PEMS07", 12),
    ("PEMS07", 24),
    ("PEMS07", 48),
    ("PEMS07", 96),
    ("PEMS08", 12),
    ("PEMS08", 24),
    ("PEMS08", 48),
    ("PEMS08", 96),
]


DATASETS = {
    "Electricity": {
        "source_names": ["electricity.csv"],
        "root": Path("dataset/electricity"),
        "data_path": "electricity.csv",
        "prefix": "electricity",
        "ratio": 0.7,
        "lengths": [96, 192, 336, 720],
        "channel_file": "electricity_COV_channel_ratio0.70.npy",
        "pems_npz": False,
        "q_no_ratio": False,
    },
    "Weather": {
        "source_names": ["weather.csv"],
        "root": Path("dataset/weather"),
        "data_path": "weather.csv",
        "prefix": "weather",
        "ratio": 0.7,
        "lengths": [96, 192, 336, 720],
        "channel_file": "weather_COV_channel_ratio0.70.npy",
        "pems_npz": False,
        "q_no_ratio": False,
    },
    "ETTh1": {
        "source_names": ["ETTh1.csv"],
        "root": Path("dataset/ETT-small"),
        "data_path": "ETTh1.csv",
        "prefix": "ETTh1",
        "ratio": 0.6,
        "lengths": [96, 192, 336, 720],
        "channel_file": "ETTh1_COV_channel_ratio0.60.npy",
        "pems_npz": False,
        "q_no_ratio": False,
    },
    "ETTh2": {
        "source_names": ["ETTh2.csv"],
        "root": Path("dataset/ETT-small"),
        "data_path": "ETTh2.csv",
        "prefix": "ETTh2",
        "ratio": 0.6,
        "lengths": [96, 192, 336, 720],
        "channel_file": "ETTh2_COV_channel_ratio0.60.npy",
        "pems_npz": False,
        "q_no_ratio": False,
    },
    "ETTm1": {
        "source_names": ["ETTm1.csv"],
        "root": Path("dataset/ETT-small"),
        "data_path": "ETTm1.csv",
        "prefix": "ETTm1",
        "ratio": 0.6,
        "lengths": [96, 192, 336, 720],
        "channel_file": "ETTm1_COV_channel_ratio0.60.npy",
        "pems_npz": False,
        "q_no_ratio": False,
    },
    "ETTm2": {
        "source_names": ["ETTm2.csv"],
        "root": Path("dataset/ETT-small"),
        "data_path": "ETTm2.csv",
        "prefix": "ETTm2",
        "ratio": 0.6,
        "lengths": [96, 192, 336, 720],
        "channel_file": "ETTm2_COV_channel_ratio0.60.npy",
        "pems_npz": False,
        "q_no_ratio": False,
    },
    "PEMS03": {
        "source_names": ["PEMS03.npz", "PEMS03.csv"],
        "root": Path("dataset/PEMS"),
        "data_path": "PEMS03.npz",
        "prefix": "PEMS03",
        "ratio": 0.6,
        "lengths": [12, 24, 48, 96, 336],
        "channel_file": "PEMS03_COV_channel_ratio0.60.npy",
        "pems_npz": True,
        "q_no_ratio": False,
    },
    "PEMS04": {
        "source_names": ["PEMS04.npz", "PEMS04.csv"],
        "root": Path("dataset/PEMS"),
        "data_path": "PEMS04.npz",
        "prefix": "PEMS04",
        "ratio": 0.6,
        "lengths": [12, 24, 48, 96, 336],
        "channel_file": "PEMS04_COV_channel_ratio0.60.npy",
        "pems_npz": True,
        "q_no_ratio": False,
    },
    "PEMS07": {
        "source_names": ["PEMS07.npz", "PEMS07.csv"],
        "root": Path("dataset/PEMS"),
        "data_path": "PEMS07.npz",
        "prefix": "PEMS07",
        "ratio": 0.6,
        "lengths": [12, 24, 48, 96, 336],
        "channel_file": "PEMS07_COV_channel_ratio0.60.npy",
        "pems_npz": True,
        "q_no_ratio": True,
    },
    "PEMS08": {
        "source_names": ["PEMS08.npz", "PEMS08.csv"],
        "root": Path("dataset/PEMS"),
        "data_path": "PEMS08.npz",
        "prefix": "PEMS08",
        "ratio": 0.6,
        "lengths": [12, 24, 48, 96, 336],
        "channel_file": "PEMS08_COV_channel_ratio0.60.npy",
        "pems_npz": True,
        "q_no_ratio": False,
    },
}


@dataclass
class JobConfig:
    root_arg: str
    root_check: Path
    data_path: str
    q_mat_file: str
    q_out_mat_file: str
    q_channel_file: str | None
    model_id: str
    data_arg: str
    enc_in: int
    learning_rate: str
    e_layers: int
    train_epochs: int
    patience: int
    lradj: str
    dropout: str
    save_pdf: int
    extra_args: list[str]


@dataclass
class Result:
    dataset: str
    horizon: int
    mse: float | None
    mae: float | None
    status: str
    log_file: str


def parse_args() -> argparse.Namespace:
    cwd = Path.cwd()
    default_repo = cwd if (cwd / "run.py").exists() else DEFAULT_REPO_ROOT
    default_python = DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable)
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=default_repo)
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--python", type=Path, default=default_python)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--seq-len", type=int, default=336)
    parser.add_argument("--log-root", type=Path, default=Path("logs/orthoLinear/len336_table"))
    parser.add_argument("--target-md", type=Path, default=DEFAULT_TARGET_MD)
    parser.add_argument("--target-csv", type=Path, default=DEFAULT_TARGET_CSV)
    parser.add_argument("--start-job", type=int, default=1)
    parser.add_argument("--resume-job", type=int, default=0)
    parser.add_argument("--resume-epoch", type=int, default=0)
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--overwrite-data", action="store_true")
    parser.add_argument("--overwrite-q", action="store_true")
    parser.add_argument("--no-file-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--allow-partial-summary", action="store_true")
    parser.add_argument("--skip-md", action="store_true")
    return parser.parse_args()


def reexec_with_target_python(args: argparse.Namespace) -> None:
    target = args.python
    if not target.exists():
        raise FileNotFoundError(f"python not found: {target}")
    current = Path(sys.executable).resolve()
    wanted = target.resolve()
    if current == wanted or os.environ.get("OLINEAR_PIPELINE_REEXEC") == "1":
        return
    env = os.environ.copy()
    env["OLINEAR_PIPELINE_REEXEC"] = "1"
    subprocess.check_call([str(target), str(Path(__file__).resolve()), *sys.argv[1:]], env=env)
    raise SystemExit(0)


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=cwd, env=env)


def ensure_repo(repo_root: Path, repo_url: str) -> None:
    if not repo_root.exists():
        repo_root.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", repo_url, str(repo_root)])
    if not (repo_root / ".git").exists() or not (repo_root / "run.py").exists():
        raise RuntimeError(f"not an OLinear repository: {repo_root}")


def patch_requirements(repo_root: Path) -> None:
    req = repo_root / "requirements.txt"
    if not req.exists():
        return
    raw_lines = req.read_text(encoding="utf-8", errors="ignore").splitlines()
    output: list[str] = []
    seen: set[str] = set()
    for line in raw_lines:
        value = "PyWavelets" if line.strip().lower() == "pywt" else line.strip()
        if not value:
            continue
        key = value.lower().replace("_", "-")
        if key not in seen:
            output.append(value)
            seen.add(key)
    for value in ["PyWavelets", "patool", "et_xmlfile"]:
        key = value.lower().replace("_", "-")
        if key not in seen:
            output.append(value)
            seen.add(key)
    req.write_text("\n".join(output) + "\n", encoding="utf-8")


def ensure_dependencies(skip_install: bool) -> None:
    if skip_install:
        return
    packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("matplotlib", "matplotlib"),
        ("torch", "torch"),
        ("fvcore", "fvcore"),
        ("einops", "einops"),
        ("thop", "thop"),
        ("timm", "timm"),
        ("reformer-pytorch", "reformer_pytorch"),
        ("openpyxl", "openpyxl"),
        ("seaborn", "seaborn"),
        ("PyWavelets", "pywt"),
        ("patool", "patoolib"),
        ("et_xmlfile", "et_xmlfile"),
    ]
    missing = [package for package, module in packages if importlib.util.find_spec(module) is None]
    if missing:
        run([sys.executable, "-m", "pip", "install", *missing])
        importlib.invalidate_caches()


def find_source(data_root: Path, names: list[str]) -> Path:
    for name in names:
        direct = data_root / name
        if direct.exists():
            return direct
    lower = {name.lower() for name in names}
    for item in data_root.rglob("*"):
        if item.is_file() and item.name.lower() in lower:
            return item
    raise FileNotFoundError(f"missing source data in {data_root}: {names}")


def q_file_name(prefix: str, length: int, ratio: float, no_ratio: bool) -> str:
    if no_ratio:
        return f"{prefix}_{length}.npy"
    return f"{prefix}_{length}_ratio{ratio:.1f}.npy"


def prepare_data(repo_root: Path, data_root: Path, overwrite_data: bool, overwrite_q: bool) -> None:
    import numpy as np
    import pandas as pd

    def dct_matrix(size: int) -> np.ndarray:
        rows = np.arange(size, dtype=np.float64)[:, None]
        cols = np.arange(size, dtype=np.float64)[None, :]
        mat = np.cos(np.pi / size * (cols + 0.5) * rows)
        mat[0, :] *= np.sqrt(1.0 / size)
        mat[1:, :] *= np.sqrt(2.0 / size)
        return mat.astype(np.float32)

    def numeric_frame(csv_path: Path) -> pd.DataFrame:
        frame = pd.read_csv(csv_path)
        frame = frame.dropna(axis=1, how="all")
        drop_cols = [c for c in frame.columns if str(c).lower() in {"date", "datetime", "time"}]
        if drop_cols:
            frame = frame.drop(columns=drop_cols)
        return frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    def values_from_source(source_path: Path) -> np.ndarray:
        if source_path.suffix.lower() == ".npz":
            loaded = np.load(source_path)
            data = loaded["data"] if "data" in loaded.files else loaded[loaded.files[0]]
            if data.ndim == 3:
                data = data[:, :, 0]
            return np.asarray(data, dtype=np.float32)
        return numeric_frame(source_path).to_numpy(dtype=np.float32)

    if not data_root.exists():
        raise FileNotFoundError(f"data root not found: {data_root}")

    for name, meta in DATASETS.items():
        source = find_source(data_root, meta["source_names"])
        root = repo_root / meta["root"]
        root.mkdir(parents=True, exist_ok=True)

        if meta["pems_npz"]:
            npz_path = root / meta["data_path"]
            if overwrite_data or not npz_path.exists():
                if source.suffix.lower() == ".npz":
                    shutil.copy2(source, npz_path)
                else:
                    values = numeric_frame(source).to_numpy(dtype=np.float32)
                    np.savez_compressed(npz_path, data=values[:, :, None])
        else:
            target = root / meta["data_path"]
            if overwrite_data or not target.exists():
                shutil.copy2(source, target)

        for length in meta["lengths"]:
            q_path = root / q_file_name(meta["prefix"], length, meta["ratio"], meta["q_no_ratio"])
            if overwrite_q or not q_path.exists():
                np.save(q_path, dct_matrix(length))

        channel_path = root / meta["channel_file"]
        if overwrite_q or not channel_path.exists():
            values = values_from_source(source)
            train_end = max(1, int(values.shape[0] * meta["ratio"]))
            corr = np.corrcoef(values[:train_end].T)
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            np.save(channel_path, corr)
        print(f"prepared {name}: {source}", flush=True)


def config_for(dataset: str, horizon: int, seq_len: int) -> JobConfig:
    base = dict(
        data_arg="custom",
        learning_rate="5e-4",
        e_layers=3,
        train_epochs=30,
        patience=5,
        lradj="type1",
        dropout="0.0",
        save_pdf=0,
        extra_args=[],
    )

    def common(**overrides: object) -> dict[str, object]:
        merged = base.copy()
        merged.update(overrides)
        return merged

    if dataset == "Electricity":
        return JobConfig(
            root_arg="./dataset/electricity/",
            root_check=Path("dataset/electricity"),
            data_path="electricity.csv",
            q_mat_file=f"electricity_{seq_len}_ratio0.7.npy",
            q_out_mat_file=f"electricity_{horizon}_ratio0.7.npy",
            q_channel_file="electricity_COV_channel_ratio0.70.npy",
            model_id=f"ECL_OLinear_{seq_len}_{horizon}_no_var_corr",
            enc_in=321,
            **common(lradj="cosine", train_epochs=30 if horizon == 96 else 50),
        )
    if dataset == "Weather":
        return JobConfig(
            root_arg="./dataset/weather/",
            root_check=Path("dataset/weather"),
            data_path="weather.csv",
            q_mat_file=f"weather_{seq_len}_ratio0.7.npy",
            q_out_mat_file=f"weather_{horizon}_ratio0.7.npy",
            q_channel_file="weather_COV_channel_ratio0.70.npy",
            model_id=f"Weather_OLinear_{seq_len}_{horizon}_no_var_corr",
            enc_in=21,
            **common(
                learning_rate="1e-3",
                e_layers=2,
                lradj="type3",
                extra_args=["--linear_attention", "0"],
            ),
        )
    if dataset == "ETTh1":
        return JobConfig(
            root_arg="./dataset/ETT-small/",
            root_check=Path("dataset/ETT-small"),
            data_path="ETTh1.csv",
            q_mat_file=f"ETTh1_{seq_len}_ratio0.6.npy",
            q_out_mat_file=f"ETTh1_{horizon}_ratio0.6.npy",
            q_channel_file="ETTh1_COV_channel_ratio0.60.npy",
            model_id=f"ETTh1_OLinear_{seq_len}_{horizon}_no_var_corr",
            enc_in=7,
            **common(
                data_arg="ETTh1",
                learning_rate="1e-4" if horizon == 720 else "5e-4",
                e_layers=2,
                patience=8,
                dropout="0.2",
            ),
        )
    if dataset == "ETTh2":
        return JobConfig(
            root_arg="./dataset/ETT-small/",
            root_check=Path("dataset/ETT-small"),
            data_path="ETTh2.csv",
            q_mat_file=f"ETTh2_{seq_len}_ratio0.6.npy",
            q_out_mat_file=f"ETTh2_{horizon}_ratio0.6.npy",
            q_channel_file="ETTh2_COV_channel_ratio0.60.npy",
            model_id=f"ETTh2_OLinear_{seq_len}_{horizon}_no_var_corr",
            enc_in=7,
            **common(
                data_arg="ETTh2",
                learning_rate="2e-4",
                e_layers=3,
                patience=8,
                dropout="0.3" if horizon == 720 else "0.2",
            ),
        )
    if dataset == "ETTm1":
        return JobConfig(
            root_arg="./dataset/ETT-small/",
            root_check=Path("dataset/ETT-small"),
            data_path="ETTm1.csv",
            q_mat_file=f"ETTm1_{seq_len}_ratio0.6.npy",
            q_out_mat_file=f"ETTm1_{horizon}_ratio0.6.npy",
            q_channel_file="ETTm1_COV_channel_ratio0.60.npy",
            model_id=f"ETTm1_OLinear_{seq_len}_{horizon}_1e-4_no_var_corr",
            enc_in=7,
            **common(
                data_arg="ETTm1",
                learning_rate="1e-4",
                e_layers=2,
                patience=8,
                dropout="0.2" if horizon == 720 else "0.1",
            ),
        )
    if dataset == "ETTm2":
        return JobConfig(
            root_arg="./dataset/ETT-small/",
            root_check=Path("dataset/ETT-small"),
            data_path="ETTm2.csv",
            q_mat_file=f"ETTm2_{seq_len}_ratio0.6.npy",
            q_out_mat_file=f"ETTm2_{horizon}_ratio0.6.npy",
            q_channel_file="ETTm2_COV_channel_ratio0.60.npy",
            model_id=f"ETTm2_OLinear_{seq_len}_{horizon}_no_var_corr",
            enc_in=7,
            **common(
                data_arg="ETTm2",
                learning_rate="1e-4",
                e_layers=1,
                patience=8,
                dropout="0.3" if horizon == 720 else "0.2",
            ),
        )
    pems = {
        "PEMS03": (358, f"PEMS03_{seq_len}_ratio0.6.npy", f"PEMS03_{horizon}_ratio0.6.npy", "PEMS03_COV_channel_ratio0.60.npy", f"PEMS03_OLinear_{seq_len}_{horizon}_no_var_corr", 1),
        "PEMS04": (307, f"PEMS04_{seq_len}_ratio0.6.npy", f"PEMS04_{horizon}_ratio0.6.npy", None, f"PEMS04_OLinear_{seq_len}_{horizon}_no_var_corr", 1),
        "PEMS07": (883, f"PEMS07_{seq_len}.npy", f"PEMS07_{horizon}.npy", "PEMS07_COV_channel_ratio0.60.npy", f"orthoLinear_PEMS07_{seq_len}_{horizon}_no_var_corr", 0),
        "PEMS08": (170, f"PEMS08_{seq_len}_ratio0.6.npy", f"PEMS08_{horizon}_ratio0.6.npy", "PEMS08_COV_channel_ratio0.60.npy", f"PEMS08_OLinear_{seq_len}_{horizon}_no_var_corr", 0),
    }
    if dataset in pems:
        enc_in, q_mat, q_out, q_channel, model_id, save_pdf = pems[dataset]
        return JobConfig(
            root_arg="./dataset/PEMS/",
            root_check=Path("dataset/PEMS"),
            data_path=f"{dataset}.npz",
            q_mat_file=q_mat,
            q_out_mat_file=q_out,
            q_channel_file=q_channel,
            model_id=model_id,
            enc_in=enc_in,
            **common(data_arg="PEMS", train_epochs=40, patience=10, lradj="cosine", save_pdf=save_pdf),
        )
    raise KeyError(dataset)


def validate_files(repo_root: Path, cfg: JobConfig) -> None:
    required = [
        (cfg.root_check / cfg.data_path, "data file"),
        (cfg.root_check / cfg.q_mat_file, "input Q matrix"),
        (cfg.root_check / cfg.q_out_mat_file, "output Q matrix"),
    ]
    if cfg.q_channel_file:
        required.append((cfg.root_check / cfg.q_channel_file, "channel Q matrix"))
    for rel_path, label in required:
        path = repo_root / rel_path
        if not path.exists():
            raise FileNotFoundError(f"missing {label}: {path}")


def build_command(cfg: JobConfig, horizon: int, seq_len: int, resume_training: int, resume_epoch: int) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        "run.py",
        "--is_training",
        "1",
        "--root_path",
        cfg.root_arg,
        "--data_path",
        cfg.data_path,
        "--q_mat_file",
        cfg.q_mat_file,
        "--q_out_mat_file",
        cfg.q_out_mat_file,
    ]
    if cfg.q_channel_file:
        cmd.extend(["--q_channel_file", cfg.q_channel_file])
    cmd.extend(
        [
            "--model_id",
            cfg.model_id,
            "--model",
            "OLinear",
            "--data",
            cfg.data_arg,
            "--features",
            "M",
            "--seq_len",
            str(seq_len),
            "--pred_len",
            str(horizon),
            "--enc_in",
            str(cfg.enc_in),
            "--dec_in",
            str(cfg.enc_in),
            "--c_out",
            str(cfg.enc_in),
            "--des",
            "Exp",
            "--embed_size",
            "16",
            "--d_model",
            "512",
            "--d_ff",
            "512",
            "--batch_size",
            "32",
            "--learning_rate",
            cfg.learning_rate,
            "--itr",
            "1",
            "--e_layers",
            str(cfg.e_layers),
            "--lossfun_alpha",
            "0.5",
            "--test_batch_size",
            "16",
            "--test_mode",
            "0",
            "--CKA_flag",
            "0",
            "--fix_seed",
            "1",
            "--resume_training",
            str(resume_training),
            "--resume_epoch",
            str(resume_epoch),
            "--save_every_epoch",
            "0",
            "--use_revin",
            "1",
            "--use_norm",
            "1",
            "--send_mail",
            "0",
            "--save_pdf",
            str(cfg.save_pdf),
            "--train_epochs",
            str(cfg.train_epochs),
            "--patience",
            str(cfg.patience),
            "--lradj",
            cfg.lradj,
            "--loss_mode",
            "L1",
            "--train_ratio",
            "1.0",
            "--dropout",
            cfg.dropout,
            "--plot_mat_flag",
            "0",
        ]
    )
    cmd.extend(cfg.extra_args)
    cmd.extend(["--checkpoints", "./checkpoints"])
    return cmd


def parse_log(log_path: Path) -> tuple[float | None, float | None, str]:
    if not log_path.exists():
        return None, None, "missing"
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        return None, None, "empty"
    patterns = [
        (re.compile(r"global best_mse:\s*([0-9.]+),\s*best_mae:\s*([0-9.]+)", re.I), False),
        (re.compile(r"best_test_batch_size:.*?best_mse:\s*([0-9.]+),\s*best_mae:\s*([0-9.]+)", re.I), False),
        (re.compile(r"Of all stages, best stage:.*?best mse:\s*([0-9.]+),\s*best mae:\s*([0-9.]+)", re.I), False),
        (re.compile(r"final output mse:\s*([0-9.]+),\s*mae:\s*([0-9.]+)", re.I), False),
        (re.compile(r"mae:\s*([0-9.]+),\s*mse:\s*([0-9.]+)", re.I), True),
    ]
    matches: list[tuple[int, float, float]] = []
    for pattern, swapped in patterns:
        for match in pattern.finditer(text):
            first = float(match.group(1))
            second = float(match.group(2))
            mse, mae = (second, first) if swapped else (first, second)
            matches.append((match.start(), mse, mae))
    if matches:
        _, mse, mae = max(matches, key=lambda item: item[0])
        return mse, mae, "complete"
    if "Traceback" in text or "[ERROR]" in text:
        return None, None, "failed"
    return None, None, "running_or_incomplete"


def collect_results(log_dir: Path) -> list[Result]:
    results = []
    for dataset, horizon in EXPECTED_JOBS:
        log_path = log_dir / f"{dataset}_seq336_pred{horizon}.log"
        mse, mae, status = parse_log(log_path)
        results.append(Result(dataset, horizon, mse, mae, status, str(log_path)))
    return results


def write_results_csv(results: list[Result], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "horizon", "mse", "mae", "value", "status", "log_file", "updated_at"],
        )
        writer.writeheader()
        for result in results:
            value = ""
            if result.mse is not None and result.mae is not None:
                value = f"{result.mse:.3f} / {result.mae:.3f}"
            writer.writerow(
                {
                    "dataset": result.dataset,
                    "horizon": result.horizon,
                    "mse": "" if result.mse is None else f"{result.mse:.5f}",
                    "mae": "" if result.mae is None else f"{result.mae:.5f}",
                    "value": value,
                    "status": result.status,
                    "log_file": result.log_file,
                    "updated_at": timestamp,
                }
            )


def split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def build_markdown_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def update_table2b(markdown_path: Path, results: list[Result]) -> int:
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    table_marker = next((idx for idx, line in enumerate(lines) if line.startswith("Table 2b ")), None)
    if table_marker is None:
        raise RuntimeError("could not find Table 2b marker")
    header_idx = None
    for idx in range(table_marker + 1, len(lines)):
        if lines[idx].startswith("| Dataset | H |"):
            header_idx = idx
            break
    if header_idx is None:
        raise RuntimeError("could not find Table 2b header")
    end_idx = header_idx
    while end_idx < len(lines) and lines[end_idx].startswith("|"):
        end_idx += 1

    headers = split_markdown_row(lines[header_idx])
    if "Ours" not in headers:
        raise RuntimeError("Table 2b has no Ours column")
    had_olinear = "OLinear" in headers
    if had_olinear:
        olinear_idx = headers.index("OLinear")
    else:
        olinear_idx = headers.index("Ours") + 1
        headers.insert(olinear_idx, "OLinear")

    lookup = {
        (r.dataset, str(r.horizon)): f"{r.mse:.3f} / {r.mae:.3f}"
        for r in results
        if r.mse is not None and r.mae is not None
    }
    updated_rows = [
        build_markdown_row(headers),
        build_markdown_row(["---" if header == "Dataset" else "---:" for header in headers]),
    ]
    updated = 0
    for line in lines[header_idx + 2 : end_idx]:
        cells = split_markdown_row(line)
        if not had_olinear:
            cells.insert(olinear_idx, "")
        while len(cells) < len(headers):
            cells.append("")
        value = lookup.get((cells[0], cells[1]))
        if value is not None:
            cells[olinear_idx] = value
            updated += 1
        updated_rows.append(build_markdown_row(cells))
    lines[header_idx:end_idx] = updated_rows
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return updated


def summarize(
    repo_root: Path,
    log_root: Path,
    target_csv: Path,
    target_md: Path,
    allow_partial: bool,
    skip_md: bool,
) -> int:
    log_dir = repo_root / log_root
    csv_path = log_dir / "olinear_len336_results.csv"
    results = collect_results(log_dir)
    write_results_csv(results, csv_path)
    target_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(csv_path, target_csv)

    incomplete = [r for r in results if r.status != "complete"]
    if incomplete and not allow_partial:
        print(f"Wrote CSV, but {len(incomplete)} jobs are incomplete. Skipping markdown update.", flush=True)
        for result in incomplete[:10]:
            print(f"  {result.dataset} H={result.horizon}: {result.status}", flush=True)
        return 2
    if not skip_md:
        updated = update_table2b(target_md, results)
        print(f"Updated Table 2b OLinear cells: {updated}", flush=True)
    print(f"Wrote CSV: {csv_path}", flush=True)
    print(f"Copied CSV: {target_csv}", flush=True)
    return 0


def run_jobs(args: argparse.Namespace) -> int:
    repo_root = args.repo_root
    log_dir = repo_root / args.log_root
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    env["PYTHONIOENCODING"] = "utf-8"

    for idx, (dataset, horizon) in enumerate(EXPECTED_JOBS, start=1):
        if idx < args.start_job:
            print(f"[{idx}] skip {dataset} H={horizon}", flush=True)
            continue
        cfg = config_for(dataset, horizon, args.seq_len)
        if not args.no_file_check:
            validate_files(repo_root, cfg)
        resume_training = 1 if idx == args.resume_job and args.resume_epoch > 0 else 0
        resume_epoch = args.resume_epoch if resume_training else 0
        cmd = build_command(cfg, horizon, args.seq_len, resume_training, resume_epoch)
        log_path = log_dir / f"{dataset}_seq{args.seq_len}_pred{horizon}.log"
        print(f"[{idx}/{len(EXPECTED_JOBS)}] {dataset} seq_len={args.seq_len} pred_len={horizon}", flush=True)
        print(f"Log: {log_path}", flush=True)
        if args.dry_run:
            print(" ".join(cmd), flush=True)
            continue
        with log_path.open("ab") as handle:
            code = subprocess.call(cmd, cwd=repo_root, env=env, stdout=handle, stderr=subprocess.STDOUT)
        summarize(repo_root, args.log_root, args.target_csv, args.target_md, allow_partial=True, skip_md=True)
        if code != 0:
            print(f"[ERROR] {dataset} H={horizon} failed with exit code {code}", flush=True)
            return code
        print(f"[OK] {dataset} H={horizon}", flush=True)
    if args.dry_run:
        return 0
    return summarize(
        repo_root,
        args.log_root,
        args.target_csv,
        args.target_md,
        allow_partial=False,
        skip_md=args.skip_md,
    )


def main() -> int:
    args = parse_args()
    reexec_with_target_python(args)
    ensure_repo(args.repo_root, args.repo_url)
    patch_requirements(args.repo_root)
    ensure_dependencies(args.skip_install)

    if args.summary_only:
        return summarize(
            args.repo_root,
            args.log_root,
            args.target_csv,
            args.target_md,
            allow_partial=args.allow_partial_summary,
            skip_md=args.skip_md,
        )

    prepare_data(args.repo_root, args.data_root, args.overwrite_data, args.overwrite_q)
    if args.prepare_only:
        return 0
    return run_jobs(args)


if __name__ == "__main__":
    raise SystemExit(main())
