import os
import argparse
import time
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .utils.yaml_io import load_yaml
from .utils.seed import set_seed
from .data.windows import global_zscore, make_strict_windows, WindowTensorDataset
from .utils.metrics import accumulate_channel_errors, mse_mae_from_sums
from .utils.cluster_memory import (
    load_cluster_memory,
    load_cluster_checkpoint,
    assign_channels_by_corr,
    assign_channels_by_cycle_template,
)
from .models.cluster_predictor import build_cluster_predictor
from .models.moe_gate import ClusterwiseMoEGate
from .utils.knn_shape import KNNShapeConfig, ShapeKNNHybrid, predict_bank_outputs


def _infer_step_minutes(df: pd.DataFrame, date_col: str) -> float:
    dt = pd.to_datetime(df[date_col])
    diffs = dt.diff().dropna()
    if diffs.empty:
        return 0.0
    step = diffs.mode().iloc[0]
    return float(step.total_seconds() / 60.0)


def _df_to_tensor(df: pd.DataFrame, date_col: str) -> tuple[torch.Tensor, list[str]]:
    cols = list(df.columns)
    value_cols = [c for c in cols if c != date_col]
    values = df[value_cols].to_numpy(dtype=np.float32)
    data = torch.tensor(values, dtype=torch.float32)
    return data, value_cols


def _resample_df(
    df: pd.DataFrame,
    date_col: str,
    target_step_min: int,
    method: str,
) -> pd.DataFrame:
    if target_step_min <= 0:
        return df
    rule = f"{int(target_step_min)}T"
    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col])
    tmp = tmp.set_index(date_col)
    if method in {"mean", "avg"}:
        out = tmp.resample(rule).mean()
    elif method in {"last", "ffill"}:
        out = tmp.resample(rule).last()
    else:
        out = tmp.resample(rule).interpolate("time").ffill().bfill()
    out = out.reset_index()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out_dir = cfg["exp"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    set_seed(
        int(cfg["exp"]["seed"]),
        deterministic=bool(cfg.get("exp", {}).get("deterministic", False)),
    )
    device = torch.device(cfg["exp"]["device"] if torch.cuda.is_available() else "cpu")

    source_cfg = cfg["source"]
    memory_path = str(source_cfg["memory_path"])
    ckpt_path = str(source_cfg["checkpoint_path"])

    memory = load_cluster_memory(memory_path, device=device)
    prototypes_kt = memory["prototypes_kt"]
    K = int(prototypes_kt.shape[0])

    ckpt = load_cluster_checkpoint(ckpt_path, device=device)
    meta = ckpt.get("meta", {})
    if len(meta) == 0:
        raise ValueError("Checkpoint meta is missing. Re-train with memory.save_checkpoint enabled.")

    input_len = int(meta["input_len"])
    pred_len = int(meta["pred_len"])
    model_cfg = meta["model_cfg"]
    moe_cfg = meta.get("moe_cfg", {})
    penalty_names = list(meta.get("penalty_names", []))

    win_cfg = cfg["window"]
    if int(win_cfg["input_len"]) != input_len or int(win_cfg["pred_len"]) != pred_len:
        raise ValueError("Window config does not match checkpoint input_len/pred_len.")

    model = build_cluster_predictor(
        num_clusters=K,
        input_len=input_len,
        pred_len=pred_len,
        model_cfg=model_cfg,
        num_channels=meta.get("num_channels", None),
        cluster_id_c=meta.get("cluster_id_c", None),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    gate_state = ckpt.get("gate_state", None)
    if gate_state is not None:
        gate_feat_dim = int(meta.get("gate_feat_dim", gate_state["W1.0"].shape[0]))
        gate_allow_skip = any(str(name).startswith("W_skip.") for name in gate_state.keys())
        gate = ClusterwiseMoEGate(
            num_clusters=K,
            feat_dim=gate_feat_dim,
            num_penalties=len(penalty_names),
            hidden_dim=int(moe_cfg.get("hidden_dim", 64)),
            topk=int(moe_cfg.get("topk", 2)),
            allow_skip=gate_allow_skip,
            skip_init_bias=float(moe_cfg.get("skip_init_bias", -2.0)),
        ).to(device)
        gate.load_state_dict(gate_state)
        gate.eval()

    data_csv = cfg["data"]["csv_path"]
    date_col_idx = int(cfg["data"]["date_col"])
    raw_df = pd.read_csv(data_csv)
    date_col = raw_df.columns[date_col_idx]

    transfer_cfg = cfg.get("transfer", {})
    resample_cfg = transfer_cfg.get("resample", {})
    if bool(resample_cfg.get("enable", False)):
        target_step_min = resample_cfg.get("target_step_minutes", None)
        if target_step_min is None:
            target_step_min = cfg.get("source", {}).get("step_minutes", None)
        if target_step_min is None:
            src_csv = cfg.get("source", {}).get("csv_path", None)
            src_date_col = int(cfg.get("source", {}).get("date_col", 0))
            if src_csv is not None:
                src_df = pd.read_csv(src_csv, nrows=2000)
                target_step_min = _infer_step_minutes(src_df, src_df.columns[src_date_col])
        target_step_min = int(target_step_min) if target_step_min is not None else 0
        method = resample_cfg.get("method", None)
        if method is None:
            cur_step = _infer_step_minutes(raw_df, date_col)
            if cur_step > 0 and target_step_min > cur_step:
                method = "mean"
            else:
                method = "linear"
        method = str(method).lower()
        raw_df = _resample_df(raw_df, date_col, target_step_min, method)

    data_tc, channel_names = _df_to_tensor(raw_df, date_col)
    data_tc = data_tc.to(device)

    T, C = data_tc.shape
    tr = float(cfg["data"]["train_ratio"])
    vr = float(cfg["data"]["val_ratio"])
    te = float(cfg["data"]["test_ratio"])
    assert abs(tr + vr + te - 1.0) < 1e-6

    t_train = int(T * tr)
    t_val = int(T * (tr + vr))

    norm_cfg = cfg.get("normalize", {})
    if bool(norm_cfg.get("global_zscore", False)):
        if norm_cfg.get("train_only", False):
            train_seg = data_tc[:t_train]
            mean_c = train_seg.mean(dim=0, keepdim=True)
            std_c = train_seg.std(dim=0, keepdim=True).clamp_min(1e-6)
            data_tc = (data_tc - mean_c) / std_c
        else:
            data_tc, _, _ = global_zscore(data_tc)

    align = str(transfer_cfg.get("corr_align", "head")).lower()
    corr_mode = str(transfer_cfg.get("corr_mode", "cycle_template")).lower()
    if corr_mode in {"cycle", "cycle_template", "phase", "phase_template"}:
        phase_bins = int(transfer_cfg.get("phase_bins", 64))
        period_min = transfer_cfg.get("period_min", None)
        period_max = transfer_cfg.get("period_max", None)
        period_min_h = transfer_cfg.get("period_min_hours", None)
        period_max_h = transfer_cfg.get("period_max_hours", None)
        phase_max_shift = transfer_cfg.get("phase_max_shift", None)
        period_min = int(period_min) if period_min is not None else None
        period_max = int(period_max) if period_max is not None else None
        if period_min_h is not None or period_max_h is not None:
            step_min = _infer_step_minutes(raw_df, date_col)
            if step_min <= 0:
                step_min = 60.0
            if period_min_h is not None:
                period_min = int(round(float(period_min_h) * 60.0 / step_min))
            if period_max_h is not None:
                period_max = int(round(float(period_max_h) * 60.0 / step_min))
        phase_max_shift = int(phase_max_shift) if phase_max_shift is not None else None
        cluster_id_c, corr_ck, best_tau_ck = assign_channels_by_cycle_template(
            data_tc,
            prototypes_kt,
            phase_bins=phase_bins,
            period_min=period_min,
            period_max=period_max,
            align=align,
            phase_max_shift=phase_max_shift,
        )
    else:
        max_lag = int(transfer_cfg.get("corr_max_lag", 0))
        cluster_id_c, corr_ck = assign_channels_by_corr(
            data_tc, prototypes_kt, align=align, max_lag=max_lag
        )
        best_tau_ck = None

    corr_max_t = corr_ck.max(dim=1).values
    corr_max = corr_max_t.detach().cpu().numpy()

    corr_threshold = transfer_cfg.get("corr_threshold", None)
    fallback_mode = str(transfer_cfg.get("fallback_mode", "hard")).lower()
    fallback_topk = int(transfer_cfg.get("fallback_topk", 2))
    fallback_temp = float(transfer_cfg.get("fallback_temp", 1.0))

    use_soft_assignment = False
    w_ck = None
    low_mask = None
    if corr_threshold is not None and fallback_mode in {"soft", "ensemble", "topk"}:
        threshold = float(corr_threshold)
        low_mask = corr_max_t < threshold
        if low_mask.any():
            use_soft_assignment = True
            k_sel = max(1, min(fallback_topk, K))
            topv, topi = torch.topk(corr_ck, k=k_sel, dim=1)
            temp = max(fallback_temp, 1.0e-6)
            weights = torch.softmax(topv / temp, dim=1)
            w_ck = torch.zeros((C, K), device=device, dtype=corr_ck.dtype)
            w_ck.scatter_(1, cluster_id_c.view(-1, 1), 1.0)
            w_ck_low = torch.zeros((int(low_mask.sum().item()), K), device=device, dtype=corr_ck.dtype)
            w_ck_low.scatter_(1, topi[low_mask], weights[low_mask])
            w_ck[low_mask] = w_ck_low
        else:
            w_ck = None
    cluster_sizes = torch.bincount(cluster_id_c, minlength=K).detach().cpu().tolist()
    print("Transfer cluster sizes: " + ", ".join(f"{k}:{n}" for k, n in enumerate(cluster_sizes)))
    for i, name in enumerate(channel_names):
        if best_tau_ck is not None:
            tau = int(best_tau_ck[i, int(cluster_id_c[i].item())].item())
            print(f"Channel {name}: cluster={int(cluster_id_c[i].item())}, corr={float(corr_max[i]):.4f}, tau={tau}")
        else:
            print(f"Channel {name}: cluster={int(cluster_id_c[i].item())}, corr={float(corr_max[i]):.4f}")
    if low_mask is not None:
        print(f"Low-corr channels: {int(low_mask.sum().item())}/{C} (threshold={float(corr_threshold):.3f})")

    L = input_len
    H = pred_len
    xte, yte = make_strict_windows(data_tc, L, H, t_val, T)
    if xte.shape[0] == 0:
        raise ValueError("No test windows available for transfer evaluation.")

    knn_cfg = KNNShapeConfig.from_dict(transfer_cfg.get("knn_hybrid", {})).resolved_for_horizon(H)
    knn_hybrid = None
    knn_predict_batch_size = int(cfg.get("eval", {}).get("batch_size", 64))
    if knn_cfg.enable:
        if knn_cfg.mode == "rolling" and knn_cfg.bank_split == "history":
            x_bank, y_bank = make_strict_windows(data_tc, L, H, 0, T)
        else:
            bank_end = t_train if knn_cfg.bank_split == "train" else t_val
            x_bank, y_bank = make_strict_windows(data_tc, L, H, 0, bank_end)
        if x_bank.shape[0] == 0:
            raise ValueError("KNN hybrid bank is empty. Increase available history or change knn_hybrid.bank_split.")
        base_bank_pred = None
        if knn_cfg.needs_base_bank_prediction():
            base_bank_pred = predict_bank_outputs(
                model=model,
                x_bank_ncl=x_bank,
                cluster_id_c=cluster_id_c,
                batch_size=max(knn_predict_batch_size, 64),
                device=device,
            )
        knn_hybrid = ShapeKNNHybrid.fit(
            x_bank_ncl=x_bank,
            y_bank_nch=y_bank,
            cluster_id_c=cluster_id_c,
            cfg=knn_cfg,
            base_bank_pred_nch=base_bank_pred,
        )
        info = knn_hybrid.describe()
        print(
            "KNN hybrid enabled: "
            f"mode={info['mode']}, "
            f"scope={info['scope']}, bank_split={info['bank_split']}, "
            f"k={info['k']}, alpha={info['alpha']:.3f}, "
            f"feature_mode={info['feature_mode']}, template_mode={info['template_mode']}, "
            f"bank_sizes={info['bank_sizes']}"
        )

    dte = WindowTensorDataset(xte, yte)
    eval_cfg = cfg.get("eval", {})
    bs = int(eval_cfg.get("batch_size", 64))
    dl_te = DataLoader(dte, batch_size=bs, shuffle=False, num_workers=0)

    se_c = torch.zeros(C, device=device)
    ae_c = torch.zeros(C, device=device)
    base_se_c = torch.zeros(C, device=device) if knn_hybrid is not None else None
    base_ae_c = torch.zeros(C, device=device) if knn_hybrid is not None else None
    denom = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for x, y, idx in dl_te:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            idx = idx.to(device=device, dtype=torch.long, non_blocking=True)
            if use_soft_assignment and w_ck is not None:
                yhat = torch.zeros((x.shape[0], C, H), device=device, dtype=x.dtype)
                for k in range(K):
                    cid_k = torch.full((C,), k, device=device, dtype=torch.long)
                    yhat_k = model(x, cid_k)
                    yhat = yhat + yhat_k * w_ck[:, k].view(1, C, 1)
            else:
                yhat = model(x, cluster_id_c)
            if knn_hybrid is not None:
                accumulate_channel_errors(base_se_c, base_ae_c, yhat, y)
                query_start_abs_b = t_val + idx
                yhat = knn_hybrid.hybridize_batch(x, yhat, cluster_id_c, query_start_abs_b=query_start_abs_b)
            accumulate_channel_errors(se_c, ae_c, yhat, y)
            denom += int(x.shape[0] * y.shape[2])

    mse_c, mae_c = mse_mae_from_sums(se_c, ae_c, denom)
    base_mse_c = None
    base_mae_c = None
    if knn_hybrid is not None:
        base_mse_c, base_mae_c = mse_mae_from_sums(base_se_c, base_ae_c, denom)

    df = pd.DataFrame({
        "channel": channel_names,
        "MAE": mae_c.detach().cpu().numpy(),
        "MSE": mse_c.detach().cpu().numpy(),
        "cluster_id": cluster_id_c.detach().cpu().numpy(),
        "corr_max": corr_max,
    })
    if base_mse_c is not None and base_mae_c is not None:
        df["MAE_base"] = base_mae_c.detach().cpu().numpy()
        df["MSE_base"] = base_mse_c.detach().cpu().numpy()
    df.to_csv(os.path.join(out_dir, "test_metrics.csv"), index=False)

    assign_payload = {
        "channel": channel_names,
        "cluster_id": cluster_id_c.detach().cpu().numpy(),
        "corr_max": corr_max,
    }
    if best_tau_ck is not None:
        tau_best = best_tau_ck[torch.arange(C, device=device), cluster_id_c].detach().cpu().numpy()
        assign_payload["best_tau"] = tau_best
    if low_mask is not None:
        assign_payload["low_corr"] = low_mask.detach().cpu().numpy()
    assign_df = pd.DataFrame(assign_payload)
    assign_df.to_csv(os.path.join(out_dir, "cluster_assignment.csv"), index=False)

    if bool(transfer_cfg.get("save_corr", True)):
        np.save(os.path.join(out_dir, "cluster_corr.npy"), corr_ck.detach().cpu().numpy())

    avg_mae = float(df["MAE"].mean())
    avg_mse = float(df["MSE"].mean())
    summary = {
        "avg_mae": avg_mae,
        "avg_mse": avg_mse,
        "elapsed_sec": float(time.perf_counter() - t0),
        "num_test_windows": int(xte.shape[0]),
        "num_channels": int(C),
    }
    if base_mse_c is not None and base_mae_c is not None:
        base_avg_mae = float(base_mae_c.mean().item())
        base_avg_mse = float(base_mse_c.mean().item())
        summary["base_avg_mae"] = base_avg_mae
        summary["base_avg_mse"] = base_avg_mse
        summary["knn_hybrid"] = knn_hybrid.describe()
        print(f"Transfer test base:   avg_MAE={base_avg_mae:.6f}, avg_MSE={base_avg_mse:.6f}")
        print(f"Transfer test hybrid: avg_MAE={avg_mae:.6f}, avg_MSE={avg_mse:.6f}")
    else:
        print(f"Transfer test: avg_MAE={avg_mae:.6f}, avg_MSE={avg_mse:.6f}")

    with open(os.path.join(out_dir, "transfer_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved transfer outputs to: {out_dir}")
    print(f"Elapsed: {summary['elapsed_sec']:.3f}s")


if __name__ == "__main__":
    main()
