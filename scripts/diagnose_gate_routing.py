"""
Diagnose gate routing 合理性：
对每个 cluster，比较 gate 实际选择 vs penalty magnitude 排序，看是否选对。

合理性判定：
1. gate 应该选 penalty magnitude 大的（"该方面错得多 → 加强惩罚"）
2. 算 gate 选择频率和 raw penalty magnitude 的 Spearman rank correlation
3. 对每个 cluster，输出 ranked_by_prob vs ranked_by_magnitude 对比表
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.reader import read_csv_time_series
from src.data.windows import global_zscore, make_label_range_windows, make_strict_windows, WindowTensorDataset
from src.utils.pearson import pearson_corr_matrix
from src.utils.clustering import cluster_channels_by_corr
from src.models.cluster_predictor import build_cluster_predictor
from src.models.moe_gate import ClusterwiseMoEGate, scatter_mean_bcf_to_bkf, scatter_mean_bc_to_bk
from src.models.penalties import build_penalty_bank, normalize_penalties
from src.train import extract_gate_features, extract_pred_features, get_pred_feature_dim, get_gate_feature_dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="best_checkpoint.pt 路径，默认根据 cfg.exp.out_dir 推断")
    ap.add_argument("--split", type=str, default="val", choices=["val", "test"])
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    device = torch.device(cfg["exp"].get("device", "cpu") if torch.cuda.is_available() else "cpu")

    # 数据加载
    data_tc, channel_names = read_csv_time_series(cfg["data"]["csv_path"], date_col=int(cfg["data"]["date_col"]))
    max_rows = int(cfg["data"].get("max_rows", 0) or 0)
    if max_rows > 0:
        data_tc = data_tc[:max_rows]
    data_tc = data_tc.to(device)
    T, C = data_tc.shape
    tr = float(cfg["data"]["train_ratio"])
    vr = float(cfg["data"]["val_ratio"])
    t_train = int(T * tr)
    t_val = int(T * (tr + vr))

    # Normalize（与训练一致）
    norm_cfg = cfg["normalize"]
    if norm_cfg["global_zscore"]:
        if norm_cfg.get("train_only", False):
            train_seg = data_tc[:t_train]
            mean_c = train_seg.mean(dim=0, keepdim=True)
            std_c = train_seg.std(dim=0, keepdim=True).clamp_min(1e-6)
            data_tc = (data_tc - mean_c) / std_c
        else:
            data_tc, _, _ = global_zscore(data_tc)

    # Cluster
    cl = cfg["cluster"]
    corr_cc = pearson_corr_matrix(data_tc)
    cluster_id_c, _ = cluster_channels_by_corr(
        corr_cc=corr_cc, data_tc=data_tc,
        n_clusters=cl.get("n_clusters"),
        distance_threshold=cl.get("distance_threshold"),
        linkage=cl.get("linkage", "average"),
        method=cl.get("method", "leader"),
        min_cluster_size=int(cl["min_cluster_size"]),
        merge_small_clusters=bool(cl["merge_small_clusters"]),
        no_merge_if_channels_lt=int(cl["no_merge_if_channels_lt"]),
    )
    K = int(cluster_id_c.max().item() + 1)

    # Windows
    L = int(cfg["window"]["input_len"])
    H = int(cfg["window"]["pred_len"])
    data_window_tc = data_tc.detach().cpu()
    if args.split == "val":
        x_split, y_split = make_strict_windows(data_window_tc, L, H, t_train, t_val)
    else:
        x_split, y_split = make_strict_windows(data_window_tc, L, H, t_val, T)

    # Penalty
    penalty_names = list(cfg["penalties"]["enabled"])
    P = len(penalty_names)
    penalty_fns = build_penalty_bank(penalty_names, jump_thr=float(cfg["penalties"]["jump_threshold"]))

    # Build model + gate（与训练一致）
    model = build_cluster_predictor(
        num_clusters=K, input_len=L, pred_len=H,
        model_cfg=cfg["model"], num_channels=C, cluster_id_c=cluster_id_c,
    ).to(device)

    moe_cfg = cfg["moe"]
    gate_feat_dim = get_gate_feature_dim()
    pred_aware_cfg = moe_cfg.get("pred_aware", {}) or {}
    pred_aware_enable = bool(pred_aware_cfg.get("enable", False))
    use_pred_features = bool(pred_aware_cfg.get("use_pred_features", True)) and pred_aware_enable
    use_penalty_input = bool(pred_aware_cfg.get("use_penalty_input", False)) and pred_aware_enable
    pred_feat_dim = get_pred_feature_dim() if use_pred_features else 0
    sigmoid_branch_cfg = moe_cfg.get("sigmoid_branch", {}) or {}
    penalty_ema_cfg = moe_cfg.get("penalty_ema", {}) or {}
    moe_min_k = int(moe_cfg.get("min_k_for_extensions", 3))

    # K-aware safeguard
    if K < moe_min_k:
        pred_aware_enable = False
        pred_feat_dim = 0
        use_pred_features = use_penalty_input = False
        gate_hidden_dim = int(moe_cfg.get("safeguard_hidden_dim", 64))
        sigmoid_branch_enable = False
        penalty_ema_enable = False
    else:
        gate_hidden_dim = int(moe_cfg.get("gate_hidden_dim", 64))
        sigmoid_branch_enable = bool(sigmoid_branch_cfg.get("enable", False))
        penalty_ema_enable = bool(penalty_ema_cfg.get("enable", False))

    gate = ClusterwiseMoEGate(
        num_clusters=K, feat_dim=gate_feat_dim, num_penalties=P,
        hidden_dim=gate_hidden_dim, topk=int(moe_cfg["topk"]),
        pred_feat_dim=pred_feat_dim,
        use_penalty_input=use_penalty_input,
        use_penalty_ema=penalty_ema_enable,
        penalty_ema_decay=float(penalty_ema_cfg.get("decay", 0.9)),
        enable_sigmoid_branch=sigmoid_branch_enable,
        sigmoid_gamma=float(sigmoid_branch_cfg.get("gamma", 0.2)),
        sigmoid_init_bias=float(sigmoid_branch_cfg.get("init_bias", -2.0)),
    ).to(device)

    # Load checkpoint
    ckpt_path = args.checkpoint or os.path.join(cfg["exp"]["out_dir"], "best_checkpoint.pt")
    print(f"Loading checkpoint: {ckpt_path}")
    if not os.path.isfile(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        return
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    if "gate_state" in ckpt:
        gate.load_state_dict(ckpt["gate_state"], strict=False)

    # 跑 split 收集统计
    model.eval(); gate.eval()
    bs = int(cfg["train"]["batch_size"])
    from torch.utils.data import DataLoader
    ds = WindowTensorDataset(x_split, y_split)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)

    sum_pen_kp = torch.zeros(K, P, device=device)
    sum_prob_kp = torch.zeros(K, P, device=device)
    cnt_k = torch.zeros(K, device=device)

    with torch.no_grad():
        for x, y, _ in dl:
            x = x.to(device); y = y.to(device)
            yhat = model(x, cluster_id_c)
            # 算 raw penalty (per-channel)，先 normalize
            pen_bcp = []
            for name in penalty_names:
                pen_bcp.append(penalty_fns[name](yhat, y))
            pen_bcp = torch.stack(pen_bcp, dim=-1)  # [B,C,P] raw
            pen_bcp_norm = normalize_penalties(pen_bcp, scale=None)  # 用本 batch scale
            pen_bkp = scatter_mean_bcf_to_bkf(pen_bcp_norm, cluster_id_c, K)  # [B,K,P]

            # gate forward
            feat_bcf = extract_gate_features(x)
            feat_bkf = scatter_mean_bcf_to_bkf(feat_bcf, cluster_id_c, K)
            pred_feat_bkf = None
            if gate.pred_feat_dim > 0:
                pred_feat_bcf = extract_pred_features(yhat)
                pred_feat_bkf = scatter_mean_bcf_to_bkf(pred_feat_bcf, cluster_id_c, K)
            _, probs_bkp, _, _ = gate(
                feat_bkf,
                pred_feat_bkf=pred_feat_bkf,
                pen_bkp=pen_bkp if gate.use_penalty_input else None,
                straight_through=False,
            )
            sum_pen_kp += pen_bkp.sum(dim=0)
            sum_prob_kp += probs_bkp.sum(dim=0)
            cnt_k += pen_bkp.shape[0]

    avg_pen_kp = (sum_pen_kp / cnt_k.view(K, 1).clamp_min(1.0)).cpu().numpy()
    avg_prob_kp = (sum_prob_kp / cnt_k.view(K, 1).clamp_min(1.0)).cpu().numpy()

    # 报告
    print("\n" + "=" * 88)
    print(f"Gate Routing Diagnostic on {args.split} split (split rows {args.split})")
    print("=" * 88)

    from scipy.stats import spearmanr  # may need fallback

    for k in range(K):
        print(f"\n>>> Cluster {k}")
        print(f"{'penalty':<14}{'avg_magnitude':>16}{'rank_by_mag':>14}{'avg_gate_prob':>16}{'rank_by_prob':>14}")
        print("-" * 76)
        magnitudes = avg_pen_kp[k]
        probs = avg_prob_kp[k]
        # rank by magnitude (大的排前)
        rank_mag = (-magnitudes).argsort().argsort() + 1
        rank_prob = (-probs).argsort().argsort() + 1
        for p in range(P):
            mark_mag = "★" if rank_mag[p] == 1 else " "
            mark_prob = "★" if rank_prob[p] == 1 else " "
            print(f"{penalty_names[p]:<14}{magnitudes[p]:>16.5f}{rank_mag[p]:>13}{mark_mag}{probs[p]:>16.4f}{rank_prob[p]:>13}{mark_prob}")
        # rank correlation
        # 没有 scipy fallback：手动算 Pearson on ranks
        rank_mag_arr = (-magnitudes).argsort().argsort()
        rank_prob_arr = (-probs).argsort().argsort()
        if P > 1:
            xm = rank_mag_arr - rank_mag_arr.mean()
            ym = rank_prob_arr - rank_prob_arr.mean()
            rho = (xm * ym).sum() / (np.sqrt((xm * xm).sum() * (ym * ym).sum()) + 1e-8)
            top1_match = "✓" if rank_mag[np.argmax(probs)] == 1 else "✗"
            print(f"\n  Rank correlation (Spearman-like): {rho:+.3f}  | gate top-1 = penalty top-1: {top1_match}")

    # 总体诊断
    print("\n" + "=" * 88)
    print("解释：")
    print("  - 若 gate top-1 = penalty top-1（★星位置一致）：gate 路由合理（选了量级最大的）")
    print("  - 若 ★ 不一致：gate 选错了 penalty（对应 multi-penalty 梯度冲突的根因）")
    print("  - rank correlation > 0.5：gate 的整体路由偏好与 penalty 量级正相关")
    print("=" * 88)


if __name__ == "__main__":
    main()
