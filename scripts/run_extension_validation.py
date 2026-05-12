"""
跨 horizon + 跨数据集，验证 ABD_h32 配置 vs baseline 的稳健性。
串行跑多组实验，最后输出统一对比表。
"""
import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# 实验组合：(label, base_cfg, pred_len)
EXPERIMENTS = [
    # 长 horizon 验证（同 ETTh1）
    ("ETTh1_H192", "configs/ETTh1.yaml", 192),
    ("ETTh1_H336", "configs/ETTh1.yaml", 336),
    # 跨数据集 H=96
    ("ETTh2_H96",  "configs/ETTh2.yaml", 96),
    ("ETTm1_H96",  "configs/ETTm1.yaml", 96),
    ("ETTm2_H96",  "configs/ETTm2.yaml", 96),
    # ETTm1 长 horizon 验证（关键：ETTm1 H=96 已显示 test -1.56% 改进，验证长程稳健性）
    ("ETTm1_H192", "configs/ETTm1.yaml", 192),
    ("ETTm1_H336", "configs/ETTm1.yaml", 336),
]


def run_one(label: str, base_cfg: str, pred_len: int, epochs: int, out_root: str) -> dict:
    out_dir = os.path.join(out_root, label)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "compare.log")
    cmd = [
        sys.executable,
        "scripts/compare_moe_extensions.py",
        "--base", base_cfg,
        "--out", out_dir,
        "--epochs", str(epochs),
        "--pred_len", str(pred_len),
        "--variants", "baseline,ABD_h32",
    ]
    print(f"\n>>> [{label}] base={base_cfg} pred_len={pred_len} epochs={epochs}")
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), stdout=logf, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - t0
    rc = proc.returncode
    # 解析 compare_results.json
    rj_path = os.path.join(out_dir, "compare_results.json")
    summary = {"label": label, "base_cfg": base_cfg, "pred_len": pred_len, "rc": rc, "elapsed": elapsed}
    if os.path.isfile(rj_path):
        rj = json.load(open(rj_path, "r", encoding="utf-8"))
        summary["base"] = rj.get("baseline", {})
        summary["abd_h32"] = rj.get("ABD_h32", {})
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", type=str, default="outputs/_extension_validation")
    ap.add_argument("--experiments", type=str, default="all", help="逗号分隔的 label，或 'all'")
    args = ap.parse_args()

    out_root = str(PROJECT_ROOT / args.out)
    os.makedirs(out_root, exist_ok=True)

    if args.experiments == "all":
        chosen = EXPERIMENTS
    else:
        wants = set(s.strip() for s in args.experiments.split(","))
        chosen = [e for e in EXPERIMENTS if e[0] in wants]
        if not chosen:
            print(f"No experiments matched: {args.experiments}")
            sys.exit(1)

    all_summaries = []
    for label, base_cfg, pred_len in chosen:
        s = run_one(label, base_cfg, pred_len, args.epochs, out_root)
        all_summaries.append(s)
        if s.get("rc", 1) != 0:
            print(f"[fail] {label} rc={s['rc']}")
            continue
        b = s.get("base", {}); a = s.get("abd_h32", {})
        # 当某一变体跑挂时，其 dict 里只有 error 字段，没有 val_mse 等。改为防御式取值并跳过下一组对比
        b_ok = bool(b) and "val_mse" in b
        a_ok = bool(a) and "val_mse" in a
        if b_ok and a_ok:
            d_val_mse = (a["val_mse"] - b["val_mse"]) / b["val_mse"] * 100
            d_val_mae = (a["val_mae"] - b["val_mae"]) / b["val_mae"] * 100
            d_test_mse = (a["test_mse_base"] - b["test_mse_base"]) / b["test_mse_base"] * 100
            d_test_mae = (a["test_mae_base"] - b["test_mae_base"]) / b["test_mae_base"] * 100
            print(
                f"[done] {label} time={s['elapsed']:.0f}s | "
                f"Δval_mse={d_val_mse:+.2f}% Δval_mae={d_val_mae:+.2f}% "
                f"Δtest_mse={d_test_mse:+.2f}% Δtest_mae={d_test_mae:+.2f}%"
            )
        else:
            miss = []
            if not b_ok: miss.append("baseline")
            if not a_ok: miss.append("ABD_h32")
            print(f"[partial] {label} time={s['elapsed']:.0f}s | missing/failed variant: {','.join(miss)}")

    # 输出大表
    print("\n" + "=" * 110)
    header = (
        f"{'label':<15}{'base_val_mse':>14}{'abd_val_mse':>14}{'Δval_mse':>11}"
        f"{'base_test_mse':>15}{'abd_test_mse':>14}{'Δtest_mse':>12}{'Δtest_mae':>12}"
    )
    print(header)
    print("-" * 110)
    for s in all_summaries:
        if s.get("rc", 1) != 0:
            print(f"{s['label']:<15} FAILED rc={s['rc']}")
            continue
        b = s.get("base", {}); a = s.get("abd_h32", {})
        if not (b and a):
            print(f"{s['label']:<15} missing")
            continue
        d_val_mse = (a["val_mse"] - b["val_mse"]) / b["val_mse"] * 100
        d_test_mse = (a["test_mse_base"] - b["test_mse_base"]) / b["test_mse_base"] * 100
        d_test_mae = (a["test_mae_base"] - b["test_mae_base"]) / b["test_mae_base"] * 100
        print(
            f"{s['label']:<15}{b['val_mse']:>14.5f}{a['val_mse']:>14.5f}{d_val_mse:>+10.2f}%"
            f"{b['test_mse_base']:>15.5f}{a['test_mse_base']:>14.5f}{d_test_mse:>+11.2f}%{d_test_mae:>+11.2f}%"
        )
    print("=" * 110)
    # 保存
    with open(os.path.join(out_root, "validation_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {os.path.join(out_root, 'validation_results.json')}")


if __name__ == "__main__":
    main()
