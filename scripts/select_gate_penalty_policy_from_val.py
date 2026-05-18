from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _penalty_names(df: pd.DataFrame) -> List[str]:
    names = []
    for col in df.columns:
        if col.startswith("mse_if_"):
            names.append(col.removeprefix("mse_if_"))
    return names


def _feature_cols(df: pd.DataFrame) -> List[str]:
    # Deployable features only: input descriptors and base forecast descriptors.
    # Do not include target-dependent labels such as base_mse, gate_mse, best_gain,
    # selected_gain, or mse_if_*.
    cols = []
    for col in df.columns:
        if col.startswith("feat_"):
            cols.append(col)
        elif col in {"base_std_over_hist", "base_shift_over_hist", "base_range_over_hist"}:
            cols.append(col)
    return cols


def _action_mse(df: pd.DataFrame, action: str) -> np.ndarray:
    if action == "skip":
        return df["base_mse"].to_numpy(dtype=np.float64)
    if action == "gate":
        return df["gate_mse"].to_numpy(dtype=np.float64)
    col = f"mse_if_{action}"
    if col not in df.columns:
        raise KeyError(f"Missing action column: {col}")
    return df[col].to_numpy(dtype=np.float64)


def _rule_mask(df: pd.DataFrame, rule: Dict[str, object]) -> np.ndarray:
    values = df[str(rule["feature"])].to_numpy(dtype=np.float64)
    threshold = float(rule["threshold"])
    if str(rule["op"]) == ">=":
        return np.isfinite(values) & (values >= threshold)
    if str(rule["op"]) == "<=":
        return np.isfinite(values) & (values <= threshold)
    raise ValueError(f"Unsupported op: {rule['op']}")


def _apply_policy(df: pd.DataFrame, rules: List[Dict[str, object]]) -> Dict[str, np.ndarray]:
    mse = df["base_mse"].to_numpy(dtype=np.float64).copy()
    action = np.array(["skip"] * len(df), dtype=object)
    covered = np.zeros(len(df), dtype=bool)
    for rule in rules:
        mask = _rule_mask(df, rule) & (~covered)
        if not mask.any():
            continue
        rule_action = str(rule["action"])
        mse_action = _action_mse(df, rule_action)
        mse[mask] = mse_action[mask]
        action[mask] = rule_action
        covered[mask] = True
    return {"mse": mse, "action": action, "covered": covered}


def _metrics(df: pd.DataFrame, mse: np.ndarray) -> Dict[str, float]:
    base = float(df["base_mse"].mean())
    gate = float(df["gate_mse"].mean()) if "gate_mse" in df.columns else float("nan")
    policy = float(np.mean(mse))
    return {
        "base_mse": base,
        "gate_mse": gate,
        "policy_mse": policy,
        "policy_gain_pct_vs_base": 100.0 * (base - policy) / max(abs(base), 1.0e-12),
        "policy_gain_pct_vs_gate": 100.0 * (gate - policy) / max(abs(gate), 1.0e-12) if np.isfinite(gate) else float("nan"),
    }


def _candidate_rules(
    df: pd.DataFrame,
    features: List[str],
    actions: List[str],
    quantiles: List[float],
    min_coverage: float,
) -> List[Dict[str, object]]:
    rules: List[Dict[str, object]] = []
    n = max(len(df), 1)
    min_rows = int(np.ceil(float(min_coverage) * n))
    for feature in features:
        values = pd.to_numeric(df[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if values.nunique(dropna=True) < 4:
            continue
        thresholds = values.quantile(quantiles).dropna().unique()
        for threshold in thresholds:
            for op in [">=", "<="]:
                base_rule = {"feature": feature, "op": op, "threshold": float(threshold)}
                mask = _rule_mask(df, base_rule)
                if int(mask.sum()) < min_rows:
                    continue
                for action in actions:
                    rules.append({**base_rule, "action": action, "coverage": float(mask.mean())})
    return rules


def _greedy_search(
    val_df: pd.DataFrame,
    max_rules: int,
    min_coverage: float,
    min_improvement: float,
    quantiles: List[float],
) -> Dict[str, object]:
    features = _feature_cols(val_df)
    penalties = _penalty_names(val_df)
    actions = ["gate"] + penalties
    candidates = _candidate_rules(val_df, features, actions, quantiles, min_coverage)

    rules: List[Dict[str, object]] = []
    current = val_df["base_mse"].to_numpy(dtype=np.float64).copy()
    candidate_rows: List[Dict[str, object]] = []
    for step in range(1, int(max_rules) + 1):
        covered = _apply_policy(val_df, rules)["covered"] if rules else np.zeros(len(val_df), dtype=bool)
        current_mse = float(current.mean())
        best: Optional[Dict[str, object]] = None
        best_mse = current_mse
        for cand in candidates:
            mask = _rule_mask(val_df, cand) & (~covered)
            if not mask.any():
                continue
            trial = current.copy()
            cand_mse = _action_mse(val_df, str(cand["action"]))
            trial[mask] = cand_mse[mask]
            trial_mse = float(trial.mean())
            improvement = current_mse - trial_mse
            row = dict(cand)
            row.update(
                {
                    "step": step,
                    "remaining_coverage": float(mask.mean()),
                    "val_mse_before": current_mse,
                    "val_mse_after": trial_mse,
                    "val_improvement": improvement,
                    "val_improvement_pct_vs_base": 100.0
                    * (float(val_df["base_mse"].mean()) - trial_mse)
                    / max(abs(float(val_df["base_mse"].mean())), 1.0e-12),
                }
            )
            candidate_rows.append(row)
            if trial_mse < best_mse:
                best = row
                best_mse = trial_mse
        if best is None or (current_mse - best_mse) < float(min_improvement):
            break
        rule = {
            "feature": best["feature"],
            "op": best["op"],
            "threshold": float(best["threshold"]),
            "action": best["action"],
            "coverage": float(best["coverage"]),
            "selected_step": step,
            "val_mse_after": float(best["val_mse_after"]),
            "val_improvement": float(best["val_improvement"]),
        }
        rules.append(rule)
        current = _apply_policy(val_df, rules)["mse"]
    return {
        "rules": rules,
        "candidate_results": pd.DataFrame(candidate_rows).sort_values(
            ["step", "val_improvement"], ascending=[True, False]
        )
        if candidate_rows
        else pd.DataFrame(),
    }


def _write_assignments(path: Path, df: pd.DataFrame, policy: Dict[str, np.ndarray]) -> None:
    cols = ["window_idx", "channel", "channel_idx", "cluster", "base_mse", "gate_mse", "best_penalty", "selected_penalty"]
    keep = [c for c in cols if c in df.columns]
    out = df[keep].copy()
    out["policy_action"] = policy["action"]
    out["policy_mse"] = policy["mse"]
    out["policy_gain_vs_base"] = df["base_mse"].to_numpy(dtype=np.float64) - policy["mse"]
    out.to_csv(path, index=False)


def _plot_bar(path: Path, val_metrics: Dict[str, float], test_metrics: Optional[Dict[str, float]]) -> None:
    labels = ["base", "current_gate", "val_rule"]
    rows = [("val", val_metrics)]
    if test_metrics is not None:
        rows.append(("test", test_metrics))
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=160)
    for i, (name, metrics) in enumerate(rows):
        values = [metrics["base_mse"], metrics["gate_mse"], metrics["policy_mse"]]
        offset = (i - (len(rows) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, label=name)
    ax.set_xticks(x, labels)
    ax.set_ylabel("MSE")
    ax.set_title("Val-Selected Penalty Policy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-samples", required=True)
    ap.add_argument("--test-samples", default=None, help="Optional final evaluation only; never used for rule selection.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-rules", type=int, default=3)
    ap.add_argument("--min-coverage", type=float, default=0.05)
    ap.add_argument("--min-improvement", type=float, default=0.0)
    ap.add_argument("--quantiles", default="0.1,0.2,0.25,0.33,0.5,0.67,0.75,0.8,0.9")
    args = ap.parse_args()

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val_df = pd.read_csv(_resolve(args.val_samples))
    quantiles = [float(v) for v in str(args.quantiles).split(",") if str(v).strip()]

    search = _greedy_search(
        val_df=val_df,
        max_rules=int(args.max_rules),
        min_coverage=float(args.min_coverage),
        min_improvement=float(args.min_improvement),
        quantiles=quantiles,
    )
    rules = list(search["rules"])
    val_policy = _apply_policy(val_df, rules)
    val_metrics = _metrics(val_df, val_policy["mse"])

    candidate_path = out_dir / "candidate_rules.csv"
    search["candidate_results"].to_csv(candidate_path, index=False)
    _write_assignments(out_dir / "val_policy_assignments.csv", val_df, val_policy)

    test_metrics = None
    if args.test_samples:
        test_df = pd.read_csv(_resolve(args.test_samples))
        test_policy = _apply_policy(test_df, rules)
        test_metrics = _metrics(test_df, test_policy["mse"])
        _write_assignments(out_dir / "test_policy_assignments.csv", test_df, test_policy)

    policy = {
        "selection_protocol": "Rules are selected on validation samples only. Test samples are optional final evaluation and are not used for rule search.",
        "rules": rules,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "deployable_features": _feature_cols(val_df),
        "actions": ["gate"] + _penalty_names(val_df),
        "val_samples": str(_resolve(args.val_samples)),
        "test_samples": str(_resolve(args.test_samples)) if args.test_samples else None,
    }
    policy_path = out_dir / "policy.json"
    policy_path.write_text(json.dumps(policy, indent=2, ensure_ascii=False), encoding="utf-8")
    _plot_bar(out_dir / "policy_mse_bar.png", val_metrics, test_metrics)

    print(json.dumps(policy, indent=2, ensure_ascii=False))
    print(f"candidate_rules: {candidate_path}")
    print(f"policy: {policy_path}")
    print(f"plot: {out_dir / 'policy_mse_bar.png'}")


if __name__ == "__main__":
    main()
