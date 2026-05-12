"""
对比 MoE Penalty 模块扩展（A: Prediction-Aware / B: Sigmoid Branch / D: Penalty EMA）的效果。

跑 4 组配置：baseline / +A / +AD / +ABD，每组用相同 seed、相同 epochs，
仅 moe.pred_aware / moe.penalty_ema / moe.sigmoid_branch 开关不同。
"""
import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def deep_update(base: dict, patch: dict) -> dict:
    """递归 in-place 更新 base，patch 覆盖同名 key。"""
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


VARIANTS = {
    "baseline": {
        # 全部默认关闭
    },
    # ★ 真正干净的 baseline：显式禁用所有 spec_moe / v12 创新（用于严格 ablation）
    "true_baseline": {
        "moe": {
            "pred_side_residual": {"enable": False},
            "residual_gate": {"enable": False},
            "pred_aware": {"enable": False, "use_pred_features": False, "use_penalty_input": False},
            "penalty_ema": {"enable": False},
            "sigmoid_branch": {"enable": False},
            "gate_balance_weight": 0.01,
            "gate_entropy_weight": 0.02,
            "gate_temperature": 1.2,
            "gate_hidden_dim": 128,
            "gate_mixup": {"enable": False},
            "penalty_replay": {"enable": False},
            "oracle_supervision": {"enable": False},
            "lambda_init": {"jump": 0.1, "amp_under": 0.1, "level": 0.1, "delta": 0.1},
        },
        "penalties": {
            "enabled": ["jump", "amp_under", "level", "delta"],
            "jump_threshold": 0.6,
        },
    },
    "A": {
        "moe": {
            "pred_aware": {
                "enable": True,
                "use_pred_features": True,
                "use_penalty_input": True,
            },
        },
    },
    "AD": {
        "moe": {
            "pred_aware": {
                "enable": True,
                "use_pred_features": True,
                "use_penalty_input": True,
            },
            "penalty_ema": {"enable": True, "decay": 0.9},
        },
    },
    "ABD": {
        "moe": {
            "pred_aware": {
                "enable": True,
                "use_pred_features": True,
                "use_penalty_input": True,
            },
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.4, "init_bias": -1.5},
        },
    },
    # 安全变体：去掉 use_penalty_input（避免 val 过拟合）
    "A_safe": {
        "moe": {
            "pred_aware": {
                "enable": True,
                "use_pred_features": True,
                "use_penalty_input": False,
            },
        },
    },
    "AD_safe": {
        "moe": {
            "pred_aware": {
                "enable": True,
                "use_pred_features": True,
                "use_penalty_input": False,
            },
            "penalty_ema": {"enable": True, "decay": 0.9},
        },
    },
    "ABD_safe": {
        "moe": {
            "pred_aware": {
                "enable": True,
                "use_pred_features": True,
                "use_penalty_input": False,
            },
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.4, "init_bias": -1.5},
        },
    },
    # Compact 变体：在 ABD_safe 基础上减小扩展容量 + 加大正则，验证「过参数化导致 val 过拟合」假设
    "ABD_compact": {
        "moe": {
            "gate_hidden_dim": 32,   # 64 → 32，减半 gate 内部容量
            "pred_aware": {
                "enable": True,
                "use_pred_features": True,
                "use_penalty_input": False,
            },
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},  # gamma 0.4→0.2，bias 更强压制
        },
        "train": {
            "weight_decay": 0.001,   # 0.0002 → 0.001，5x 加大正则
        },
    },
    # Compact 变体 2：仅减容量，不动 wd（用于隔离两者效果）
    "ABD_h32": {
        "moe": {
            "gate_hidden_dim": 32,
            "pred_aware": {
                "enable": True,
                "use_pred_features": True,
                "use_penalty_input": False,
            },
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    # ABD_h32 + Gate-Side Mixup Consistency Regularization (方案 1)
    "ABD_h32_mixup": {
        "moe": {
            "gate_hidden_dim": 32,
            "pred_aware": {
                "enable": True,
                "use_pred_features": True,
                "use_penalty_input": False,
            },
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
            "gate_mixup": {"enable": True, "alpha": 0.4, "weight": 0.1, "mode": "kl"},
        },
    },
    # ABD_h32 + Penalty Replay Buffer (方案 2 — 实证有害，仅保留作反例)
    "ABD_h32_replay": {
        "moe": {
            "gate_hidden_dim": 32,
            "pred_aware": {
                "enable": True,
                "use_pred_features": True,
                "use_penalty_input": False,
            },
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
            "penalty_replay": {
                "enable": True,
                "capacity": 512,
                "sample_size": 32,
                "weight": 0.05,
                "mode": "kl",
                "warmup_batches": 32,
            },
        },
    },
    # 方案 1 强化版：自适应权重 + cosine 衰减
    "ABD_h32_mixup_v2": {
        "moe": {
            "gate_hidden_dim": 32,
            "pred_aware": {
                "enable": True,
                "use_pred_features": True,
                "use_penalty_input": False,
            },
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
            "gate_mixup": {
                "enable": True,
                "alpha": 0.4,
                "weight": 0.1,
                "mode": "kl",
                "auto_full_size": 15000,    # ≤15k → 全开
                "auto_zero_size": 50000,    # ≥50k → 自动关
                "decay_schedule": "cosine", # 训练后期减压
                "decay_end_factor": 0.2,    # 末期保留 20% 强度
            },
        },
    },
    # v5 系列：基于 GPT 建议的「让 gate 更软」三件套（temperature / entropy / balance）
    # 在 ETTm1（大数据，K=3，已有 ABD_h32 -1.56% 改进）上看是否能继续突破
    "ABD_h32_v5_temp": {
        "moe": {
            "gate_hidden_dim": 32,
            "gate_temperature": 2.0,    # 仅拉高 T，让 softmax 更平滑
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    "ABD_h32_v5_reg": {
        "moe": {
            "gate_hidden_dim": 32,
            "gate_entropy_weight": 0.01,  # 鼓励 gate 保持熵（不过度自信）
            "gate_balance_weight": 0.005, # 负载均衡
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    "ABD_h32_v5_combo": {
        "moe": {
            "gate_hidden_dim": 32,
            "gate_temperature": 2.0,
            "gate_entropy_weight": 0.01,
            "gate_balance_weight": 0.005,
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    # v6 系列：v5_reg 强度梯度（ETTm1 上 v5_reg 取得 -1.70% 突破，看更强是否继续推进）
    "ABD_h32_v6_reg_x2": {
        "moe": {
            "gate_hidden_dim": 32,
            "gate_entropy_weight": 0.02,    # 2x
            "gate_balance_weight": 0.01,    # 2x
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    "ABD_h32_v6_reg_x4": {
        "moe": {
            "gate_hidden_dim": 32,
            "gate_entropy_weight": 0.04,    # 4x
            "gate_balance_weight": 0.02,    # 4x
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    # v8 系列：v5_reg + 进阶尝试（细调 reg 强度 / dropout / mixup_lite）
    "ABD_h32_v8_reg_low": {
        "moe": {
            "gate_hidden_dim": 32,
            "gate_entropy_weight": 0.008,    # 0.8x of v5_reg
            "gate_balance_weight": 0.004,
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    "ABD_h32_v8_reg_high": {
        "moe": {
            "gate_hidden_dim": 32,
            "gate_entropy_weight": 0.012,    # 1.2x of v5_reg
            "gate_balance_weight": 0.006,
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    "ABD_h32_v8_dropout": {
        "moe": {
            "gate_hidden_dim": 32,
            "gate_entropy_weight": 0.01,
            "gate_balance_weight": 0.005,
            "gate_dropout": 0.1,             # GPT 思路 3
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    # v9 解耦：单独看 entropy / balance 哪个起主导作用
    "ABD_h32_v9_entropy_only": {
        "moe": {
            "gate_hidden_dim": 32,
            "gate_entropy_weight": 0.008,    # 仅 entropy
            "gate_balance_weight": 0.0,
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    "ABD_h32_v9_balance_only": {
        "moe": {
            "gate_hidden_dim": 32,
            "gate_entropy_weight": 0.0,
            "gate_balance_weight": 0.004,    # 仅 balance
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    # v10 系列：基于 GPT 诊断 — Residual Gate 防 routing collapse
    "ABD_h32_v10_residual_03": {
        "moe": {
            "gate_hidden_dim": 32,
            "gate_entropy_weight": 0.008,
            "gate_balance_weight": 0.004,
            "residual_gate": {"enable": True, "alpha": 0.3},  # gate 影响 30%
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    "ABD_h32_v10_residual_05": {
        "moe": {
            "gate_hidden_dim": 32,
            "gate_entropy_weight": 0.008,
            "gate_balance_weight": 0.004,
            "residual_gate": {"enable": True, "alpha": 0.5},  # gate 影响 50%
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    "ABD_h32_v10_residual_07": {
        "moe": {
            "gate_hidden_dim": 32,
            "gate_entropy_weight": 0.008,
            "gate_balance_weight": 0.004,
            "residual_gate": {"enable": True, "alpha": 0.7},  # gate 影响 70%
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    # v11 精扫 alpha 在 0.6-0.85 区间确定甜点
    "ABD_h32_v11_alpha_06": {
        "moe": {
            "gate_hidden_dim": 32,
            "gate_entropy_weight": 0.008,
            "gate_balance_weight": 0.004,
            "residual_gate": {"enable": True, "alpha": 0.6},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    "ABD_h32_v11_alpha_075": {
        "moe": {
            "gate_hidden_dim": 32,
            "gate_entropy_weight": 0.008,
            "gate_balance_weight": 0.004,
            "residual_gate": {"enable": True, "alpha": 0.75},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    "ABD_h32_v11_alpha_08": {
        "moe": {
            "gate_hidden_dim": 32,
            "gate_entropy_weight": 0.008,
            "gate_balance_weight": 0.004,
            "residual_gate": {"enable": True, "alpha": 0.8},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    "ABD_h32_v11_alpha_085": {
        "moe": {
            "gate_hidden_dim": 32,
            "gate_entropy_weight": 0.008,
            "gate_balance_weight": 0.004,
            "residual_gate": {"enable": True, "alpha": 0.85},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    # v12 系列：基于 GPT 诊断「raw gate 95% 塌缩」直接治理
    # 在 alpha=0.7 基础上 (residual gate) 加强 balance + 调高 temperature 直接让 raw gate 不塌缩
    "ABD_h32_v12_T2_b01": {
        "moe": {
            "gate_hidden_dim": 32,
            "gate_temperature": 2.0,
            "gate_entropy_weight": 0.008,
            "gate_balance_weight": 0.01,         # 比 v8_reg_low 0.004 加强 2.5x
            "residual_gate": {"enable": True, "alpha": 0.7},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    "ABD_h32_v12_T3_b01": {
        "moe": {
            "gate_hidden_dim": 32,
            "gate_temperature": 3.0,             # GPT 推荐 T=3
            "gate_entropy_weight": 0.008,
            "gate_balance_weight": 0.01,
            "residual_gate": {"enable": True, "alpha": 0.7},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    "ABD_h32_v12_T2_b02": {
        "moe": {
            "gate_hidden_dim": 32,
            "gate_temperature": 2.0,
            "gate_entropy_weight": 0.008,
            "gate_balance_weight": 0.02,         # 更强 balance
            "residual_gate": {"enable": True, "alpha": 0.7},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    # 仅加强 balance 不动 T（看 balance 单独贡献）
    # v13 系列：基于 v12_T1_b02 突破 (-2.90%) 进一步精扫 balance 强度
    "ABD_h32_v13_b015": {
        "moe": {
            "gate_hidden_dim": 32,
            "gate_temperature": 1.0,
            "gate_entropy_weight": 0.008,
            "gate_balance_weight": 0.015,
            "residual_gate": {"enable": True, "alpha": 0.7},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    "ABD_h32_v13_b03": {
        "moe": {
            "gate_hidden_dim": 32,
            "gate_temperature": 1.0,
            "gate_entropy_weight": 0.008,
            "gate_balance_weight": 0.03,
            "residual_gate": {"enable": True, "alpha": 0.7},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    "ABD_h32_v13_b05": {
        "moe": {
            "gate_hidden_dim": 32,
            "gate_temperature": 1.0,
            "gate_entropy_weight": 0.008,
            "gate_balance_weight": 0.05,
            "residual_gate": {"enable": True, "alpha": 0.7},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    # 同时也扫 entropy 看是否需要联动
    "ABD_h32_v13_b02_e02": {
        "moe": {
            "gate_hidden_dim": 32,
            "gate_temperature": 1.0,
            "gate_entropy_weight": 0.02,    # entropy 也加倍
            "gate_balance_weight": 0.02,
            "residual_gate": {"enable": True, "alpha": 0.7},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    # v14 系列：在 balance=0.02 下重扫 alpha，验证 GPT 假说
    # 「balance 强了 5x 后 alpha 甜点可能从 0.7 上移到 0.75/0.8」
    "ABD_h32_v14_alpha_065_b02": {
        "moe": {
            "gate_hidden_dim": 32,
            "gate_temperature": 1.0,
            "gate_entropy_weight": 0.008,
            "gate_balance_weight": 0.02,    # 与 v12_T1_b02 同
            "residual_gate": {"enable": True, "alpha": 0.65},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    "ABD_h32_v14_alpha_075_b02": {
        "moe": {
            "gate_hidden_dim": 32,
            "gate_temperature": 1.0,
            "gate_entropy_weight": 0.008,
            "gate_balance_weight": 0.02,
            "residual_gate": {"enable": True, "alpha": 0.75},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    "ABD_h32_v14_alpha_08_b02": {
        "moe": {
            "gate_hidden_dim": 32,
            "gate_temperature": 1.0,
            "gate_entropy_weight": 0.008,
            "gate_balance_weight": 0.02,
            "residual_gate": {"enable": True, "alpha": 0.8},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    "ABD_h32_v14_alpha_085_b02": {
        "moe": {
            "gate_hidden_dim": 32,
            "gate_temperature": 1.0,
            "gate_entropy_weight": 0.008,
            "gate_balance_weight": 0.02,
            "residual_gate": {"enable": True, "alpha": 0.85},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    # v15 系列：基于 v12_T1_b02 (-2.90%) 测 balance schedule 机制 (GPT v15 验证)
    # warmup: 早期不加 balance，给 expert 分化时间，再约束
    "ABD_h32_v15_warmup10": {
        "moe": {
            "gate_hidden_dim": 32, "gate_temperature": 1.0,
            "gate_entropy_weight": 0.008, "gate_balance_weight": 0.02,
            "gate_balance_schedule": {"enable": True, "warmup_ratio": 0.1, "rampup_ratio": 0.2},
            "residual_gate": {"enable": True, "alpha": 0.7},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    "ABD_h32_v15_warmup20": {
        "moe": {
            "gate_hidden_dim": 32, "gate_temperature": 1.0,
            "gate_entropy_weight": 0.008, "gate_balance_weight": 0.02,
            "gate_balance_schedule": {"enable": True, "warmup_ratio": 0.2, "rampup_ratio": 0.3},
            "residual_gate": {"enable": True, "alpha": 0.7},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    # decay: 早期强 balance 防早期塌缩，后期减压让 gate 自由学习
    "ABD_h32_v15_decay30": {
        "moe": {
            "gate_hidden_dim": 32, "gate_temperature": 1.0,
            "gate_entropy_weight": 0.008, "gate_balance_weight": 0.02,
            "gate_balance_schedule": {"enable": True, "decay_ratio": 0.3, "end_factor": 0.2},
            "residual_gate": {"enable": True, "alpha": 0.7},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    # warmup + decay: 完整 hat-shape schedule
    # v16: 让 decay 真正落在 best_epoch 之前，强制其影响选择
    # epochs=80 + decay 50%（让 best_epoch 进入 decay 区）
    "ABD_h32_v16_decay_early": {
        "moe": {
            "gate_hidden_dim": 32, "gate_temperature": 1.0,
            "gate_entropy_weight": 0.008, "gate_balance_weight": 0.02,
            "gate_balance_schedule": {
                "enable": True,
                "warmup_ratio": 0.0, "rampup_ratio": 0.0,
                "decay_ratio": 0.6, "end_factor": 0.2,  # 后 60% decay 到 0.2x
            },
            "residual_gate": {"enable": True, "alpha": 0.7},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    # v18 系列：简化版 oracle supervision —— KL(softmax(pen/τ) || gate_probs) 直接监督 gate
    "ABD_h32_v18_oracle_001": {
        "moe": {
            "gate_hidden_dim": 32, "gate_temperature": 1.0,
            "gate_entropy_weight": 0.008, "gate_balance_weight": 0.02,
            "oracle_supervision": {"enable": True, "weight": 0.01, "temperature": 1.0},
            "residual_gate": {"enable": True, "alpha": 0.7},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    "ABD_h32_v18_oracle_005": {
        "moe": {
            "gate_hidden_dim": 32, "gate_temperature": 1.0,
            "gate_entropy_weight": 0.008, "gate_balance_weight": 0.02,
            "oracle_supervision": {"enable": True, "weight": 0.05, "temperature": 1.0},
            "residual_gate": {"enable": True, "alpha": 0.7},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    "ABD_h32_v18_oracle_01": {
        "moe": {
            "gate_hidden_dim": 32, "gate_temperature": 1.0,
            "gate_entropy_weight": 0.008, "gate_balance_weight": 0.02,
            "oracle_supervision": {"enable": True, "weight": 0.1, "temperature": 1.0},
            "residual_gate": {"enable": True, "alpha": 0.7},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    # 反向：小 penalty 高权重（看 oracle 方向是否重要）
    "ABD_h32_v18_oracle_inv": {
        "moe": {
            "gate_hidden_dim": 32, "gate_temperature": 1.0,
            "gate_entropy_weight": 0.008, "gate_balance_weight": 0.02,
            "oracle_supervision": {"enable": True, "weight": 0.05, "temperature": 1.0, "invert": True},
            "residual_gate": {"enable": True, "alpha": 0.7},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    # 不带 balance loss + oracle（看 oracle 能否替代 balance 防塌缩）
    "ABD_h32_v18_oracle_only": {
        "moe": {
            "gate_hidden_dim": 32, "gate_temperature": 1.0,
            "gate_entropy_weight": 0.008,
            "gate_balance_weight": 0.0,
            "oracle_supervision": {"enable": True, "weight": 0.05, "temperature": 1.0},
            "residual_gate": {"enable": True, "alpha": 0.7},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    # === 假设验证：MoE 在权重大 / MSE 弱 / MAE 关 时是否真起作用 ===
    "v12_lambda_5x": {
        "moe": {
            "lambda_init": {"jump": 0.5, "amp_under": 0.5, "level": 0.5, "delta": 0.5},
        },
    },
    "v12_mse_low": {
        "train": {
            "mse_weight": 0.3,
        },
    },
    "v12_mae_off": {
        "train": {
            "mae_objective": {"enable": False, "weight": 0.0},
        },
    },
    "v12_lambda_5x_mae_off": {
        "moe": {
            "lambda_init": {"jump": 0.5, "amp_under": 0.5, "level": 0.5, "delta": 0.5},
        },
        "train": {
            "mae_objective": {"enable": False, "weight": 0.0},
        },
    },
    # === Penalty 类型互补性 ablation ===
    # 假设：gate 选 level/amp_under 是因为这俩与 MSE 同方向；启用纯互补 penalty 让 gate 必须选互补类型
    "v12_complement_only": {
        "moe": {
            "lambda_init": {
                "jump": 0.1, "corr": 0.1, "direction": 0.1,
                "trend": 0.1, "jitter": 0.1, "smooth": 0.1,
            },
            "lambda_min": {
                "jump": 0.0, "corr": 0.0, "direction": 0.0,
                "trend": 0.0, "jitter": 0.0, "smooth": 0.0,
            },
        },
        "penalties": {
            "enabled": ["jump", "corr", "direction", "trend", "jitter", "smooth"],
            "jump_threshold": 0.3,  # 调低让 jump 真正触发
        },
    },
    # 全 12 种 penalty 池，让 gate 自己选
    "v12_full_pool": {
        "moe": {
            "lambda_init": {
                "amp": 0.1, "amp_under": 0.1, "level": 0.1, "delta": 0.1,
                "jump": 0.1, "corr": 0.1, "direction": 0.1, "trend": 0.1,
                "jitter": 0.1, "smooth": 0.1, "diff_amp": 0.1, "range": 0.1,
            },
            "lambda_min": {
                "amp": 0.0, "amp_under": 0.0, "level": 0.0, "delta": 0.0,
                "jump": 0.0, "corr": 0.0, "direction": 0.0, "trend": 0.0,
                "jitter": 0.0, "smooth": 0.0, "diff_amp": 0.0, "range": 0.0,
            },
        },
        "penalties": {
            "enabled": ["amp", "amp_under", "level", "delta", "jump", "corr",
                       "direction", "trend", "jitter", "smooth", "diff_amp", "range"],
            "jump_threshold": 0.3,
        },
    },
    # 强烈鼓励 gate 选互补 penalty：互补 λ 大，重叠 λ 小
    "v12_boost_complement": {
        "moe": {
            "lambda_init": {
                # 互补 penalty 大权重（10x）
                "jump": 1.0, "corr": 1.0, "direction": 1.0, "trend": 1.0,
                # 重叠 penalty 小权重（保留少量监督）
                "amp_under": 0.02, "level": 0.02, "delta": 0.02,
            },
            "lambda_min": {
                "jump": 0.0, "corr": 0.0, "direction": 0.0, "trend": 0.0,
                "amp_under": 0.0, "level": 0.0, "delta": 0.0,
            },
        },
        "penalties": {
            "enabled": ["jump", "amp_under", "level", "delta", "corr", "direction", "trend"],
            "jump_threshold": 0.3,
        },
    },
    "v12_lambda_10x_mae_off_mse_low": {
        "moe": {
            "lambda_init": {"jump": 1.0, "amp_under": 1.0, "level": 1.0, "delta": 1.0},
        },
        "train": {
            "mse_weight": 0.3,
            "mae_objective": {"enable": False, "weight": 0.0},
        },
    },
    # === Penalty-Supervised Expert MoE (核心架构改造) ===
    # 每个 expert_p 被对应 penalty_p 单独监督，强制 specialization
    # 破除"参数共享 → 多 penalty 梯度冲突"的诅咒
    "spec_moe_w05": {
        "moe": {
            "pred_side_residual": {
                "enable": True, "corrector_hidden": 32, "init_alpha": -3.0,
                "specialization_weight": 0.5,
            },
        },
    },
    # ★★ Hard-select Correction MoE：base + 单一 correction（用户精确描述的设计）
    "spec_moe_hard_correction": {
        "moe": {
            "pred_side_residual": {
                "enable": True, "corrector_hidden": 32, "init_alpha": -3.0,
                "specialization_weight": 1.0,
                "select_mode": "hard",     # ★ 关键：hard select 不混合
            },
        },
        "penalties": {
            "enabled": ["amp_under", "delta", "jitter", "smooth"],
            "jump_threshold": 0.6,
        },
    },
    "spec_moe_hard_correction_h128": {
        "moe": {
            "pred_side_residual": {
                "enable": True, "corrector_hidden": 128, "init_alpha": -3.0,
                "specialization_weight": 1.0,
                "select_mode": "hard",
            },
        },
        "penalties": {
            "enabled": ["amp_under", "delta", "jitter", "smooth"],
            "jump_threshold": 0.6,
        },
    },
    "spec_moe_hard_correction_spec5": {
        "moe": {
            "pred_side_residual": {
                "enable": True, "corrector_hidden": 64, "init_alpha": -3.0,
                "specialization_weight": 5.0,   # 加强 specialization
                "select_mode": "hard",
            },
        },
        "penalties": {
            "enabled": ["amp_under", "delta", "jitter", "smooth"],
            "jump_threshold": 0.6,
        },
    },
    "spec_moe_w10": {
        "moe": {
            "pred_side_residual": {
                "enable": True, "corrector_hidden": 32, "init_alpha": -3.0,
                "specialization_weight": 1.0,
            },
        },
    },
    "spec_moe_w20": {
        "moe": {
            "pred_side_residual": {
                "enable": True, "corrector_hidden": 32, "init_alpha": -3.0,
                "specialization_weight": 2.0,
            },
        },
    },
    # spec_moe + 互补 penalty 池：强制 expert 学 MSE 看不到的方向
    "spec_moe_w10_complement": {
        "moe": {
            "pred_side_residual": {
                "enable": True, "corrector_hidden": 32, "init_alpha": -3.0,
                "specialization_weight": 1.0,
            },
            "lambda_init": {
                "jump": 0.1, "corr": 0.1, "direction": 0.1, "trend": 0.1,
            },
            "lambda_min": {
                "jump": 0.0, "corr": 0.0, "direction": 0.0, "trend": 0.0,
            },
        },
        "penalties": {
            "enabled": ["jump", "corr", "direction", "trend"],
            "jump_threshold": 0.3,
        },
    },
    # spec_moe + boost 互补 penalty 权重（pool 含全部，但互补的权重大）
    # ★★ Hard Switch MoE — 真正"不混合"的 penalty 路由（核心创新）
    # K × P 个完整 predictor，每个样本 hard-select 走单一路径
    "hard_switch_4pool": {
        "moe": {
            "hard_switch_moe": {"enable": True, "hidden_dim": 128, "dropout": 0.1},
            "pred_side_residual": {"enable": False},
            "lambda_init": {"amp_under": 0.1, "delta": 0.1, "jitter": 0.1, "smooth": 0.1},
            "lambda_min": {"amp_under": 0.0, "delta": 0.0, "jitter": 0.0, "smooth": 0.0},
        },
        "penalties": {
            "enabled": ["amp_under", "delta", "jitter", "smooth"],
            "jump_threshold": 0.6,
        },
    },
    "hard_switch_8pool": {
        "moe": {
            "hard_switch_moe": {"enable": True, "hidden_dim": 128, "dropout": 0.1},
            "pred_side_residual": {"enable": False},
            "lambda_init": {
                "amp_under": 0.1, "level": 0.1, "delta": 0.1, "jump": 0.1,
                "corr": 0.1, "direction": 0.1, "jitter": 0.1, "smooth": 0.1,
            },
            "lambda_min": {
                "amp_under": 0.0, "level": 0.0, "delta": 0.0, "jump": 0.0,
                "corr": 0.0, "direction": 0.0, "jitter": 0.0, "smooth": 0.0,
            },
        },
        "penalties": {
            "enabled": ["amp_under", "level", "delta", "jump", "corr", "direction", "jitter", "smooth"],
            "jump_threshold": 0.3,
        },
    },
    # 加强 specialization：让 penalty 真正塑造 expert 参数
    "hard_switch_spec3": {
        "moe": {
            "hard_switch_moe": {"enable": True, "hidden_dim": 128, "dropout": 0.1},
            "pred_side_residual": {"enable": False, "specialization_weight": 3.0},
            "lambda_init": {"amp_under": 0.1, "delta": 0.1, "jitter": 0.1, "smooth": 0.1},
            "lambda_min": {"amp_under": 0.0, "delta": 0.0, "jitter": 0.0, "smooth": 0.0},
        },
        "penalties": {
            "enabled": ["amp_under", "delta", "jitter", "smooth"],
            "jump_threshold": 0.6,
        },
    },
    "hard_switch_spec5": {
        "moe": {
            "hard_switch_moe": {"enable": True, "hidden_dim": 128, "dropout": 0.1},
            "pred_side_residual": {"enable": False, "specialization_weight": 5.0},
            "lambda_init": {"amp_under": 0.1, "delta": 0.1, "jitter": 0.1, "smooth": 0.1},
            "lambda_min": {"amp_under": 0.0, "delta": 0.0, "jitter": 0.0, "smooth": 0.0},
        },
        "penalties": {
            "enabled": ["amp_under", "delta", "jitter", "smooth"],
            "jump_threshold": 0.6,
        },
    },
    # 容量更大的 hard switch
    "hard_switch_4pool_h256": {
        "moe": {
            "hard_switch_moe": {"enable": True, "hidden_dim": 256, "dropout": 0.1},
            "pred_side_residual": {"enable": False},
            "lambda_init": {"amp_under": 0.1, "delta": 0.1, "jitter": 0.1, "smooth": 0.1},
            "lambda_min": {"amp_under": 0.0, "delta": 0.0, "jitter": 0.0, "smooth": 0.0},
        },
        "penalties": {
            "enabled": ["amp_under", "delta", "jitter", "smooth"],
            "jump_threshold": 0.6,
        },
    },
    # ★ Input-driven Dynamic MoE：启用全部 12 种 penalty，让 gate per-sample 自适应选择
    # 这才是真正的 MoE：推理时不需要 val 选 pool，gate 看 x 直接路由
    "input_driven_full_pool": {
        "moe": {
            "pred_side_residual": {
                "enable": True, "corrector_hidden": 32, "init_alpha": -3.0,
                "specialization_weight": 1.0,
            },
            "topk": 2,  # 允许 gate 同时激活 2 个 expert，更有路由空间
            "lambda_init": {
                "amp": 0.1, "amp_under": 0.1, "level": 0.1, "delta": 0.1,
                "jump": 0.1, "corr": 0.1, "direction": 0.1, "trend": 0.1,
                "jitter": 0.1, "smooth": 0.1, "diff_amp": 0.1, "range": 0.1,
            },
            "lambda_min": {
                "amp": 0.0, "amp_under": 0.0, "level": 0.0, "delta": 0.0,
                "jump": 0.0, "corr": 0.0, "direction": 0.0, "trend": 0.0,
                "jitter": 0.0, "smooth": 0.0, "diff_amp": 0.0, "range": 0.0,
            },
        },
        "penalties": {
            "enabled": ["amp", "amp_under", "level", "delta", "jump", "corr",
                       "direction", "trend", "jitter", "smooth", "diff_amp", "range"],
            "jump_threshold": 0.3,
        },
    },
    # 中等池：8 个 penalty 留余地（去掉 amp、range、diff_amp 这些纯重叠的）
    "input_driven_8pool": {
        "moe": {
            "pred_side_residual": {
                "enable": True, "corrector_hidden": 32, "init_alpha": -3.0,
                "specialization_weight": 1.0,
            },
            "topk": 2,
            "lambda_init": {
                "amp_under": 0.1, "level": 0.1, "delta": 0.1, "jump": 0.1,
                "corr": 0.1, "direction": 0.1, "jitter": 0.1, "smooth": 0.1,
            },
            "lambda_min": {
                "amp_under": 0.0, "level": 0.0, "delta": 0.0, "jump": 0.0,
                "corr": 0.0, "direction": 0.0, "jitter": 0.0, "smooth": 0.0,
            },
        },
        "penalties": {
            "enabled": ["amp_under", "level", "delta", "jump", "corr", "direction", "jitter", "smooth"],
            "jump_threshold": 0.3,
        },
    },
    # 数据驱动 penalty 选择：删 jump/level (ETTm1 上无意义)，保留 amp_under/delta + 加 jitter/smooth
    "spec_moe_data_driven_2": {
        "moe": {
            "pred_side_residual": {
                "enable": True, "corrector_hidden": 32, "init_alpha": -3.0,
                "specialization_weight": 1.0,
            },
            "lambda_init": {"amp_under": 0.1, "delta": 0.1},
            "lambda_min": {"amp_under": 0.0, "delta": 0.0},
        },
        "penalties": {
            "enabled": ["amp_under", "delta"],
            "jump_threshold": 0.6,
        },
    },
    "spec_moe_data_driven_3": {
        "moe": {
            "pred_side_residual": {
                "enable": True, "corrector_hidden": 32, "init_alpha": -3.0,
                "specialization_weight": 1.0,
            },
            "lambda_init": {"amp_under": 0.1, "delta": 0.1, "jitter": 0.1},
            "lambda_min": {"amp_under": 0.0, "delta": 0.0, "jitter": 0.0},
        },
        "penalties": {
            "enabled": ["amp_under", "delta", "jitter"],
            "jump_threshold": 0.6,
        },
    },
    "spec_moe_data_driven_4": {
        "moe": {
            "pred_side_residual": {
                "enable": True, "corrector_hidden": 32, "init_alpha": -3.0,
                "specialization_weight": 1.0,
            },
            "lambda_init": {"amp_under": 0.1, "delta": 0.1, "jitter": 0.1, "smooth": 0.1},
            "lambda_min": {"amp_under": 0.0, "delta": 0.0, "jitter": 0.0, "smooth": 0.0},
        },
        "penalties": {
            "enabled": ["amp_under", "delta", "jitter", "smooth"],
            "jump_threshold": 0.6,
        },
    },
    # 只用 amp_under + jitter（一个幅度，一个平滑度）
    "spec_moe_amp_jitter": {
        "moe": {
            "pred_side_residual": {
                "enable": True, "corrector_hidden": 32, "init_alpha": -3.0,
                "specialization_weight": 1.0,
            },
            "lambda_init": {"amp_under": 0.1, "jitter": 0.1},
            "lambda_min": {"amp_under": 0.0, "jitter": 0.0},
        },
        "penalties": {
            "enabled": ["amp_under", "jitter"],
            "jump_threshold": 0.6,
        },
    },
    "spec_moe_w10_boost_complement": {
        "moe": {
            "pred_side_residual": {
                "enable": True, "corrector_hidden": 32, "init_alpha": -3.0,
                "specialization_weight": 1.0,
            },
            "lambda_init": {
                # 互补 penalty 强权重
                "jump": 0.5, "corr": 0.5, "direction": 0.5, "trend": 0.5,
                # MSE-重叠 penalty 弱权重
                "amp_under": 0.02, "level": 0.02, "delta": 0.02,
            },
            "lambda_min": {
                "jump": 0.0, "corr": 0.0, "direction": 0.0, "trend": 0.0,
                "amp_under": 0.0, "level": 0.0, "delta": 0.0,
            },
        },
        "penalties": {
            "enabled": ["jump", "amp_under", "level", "delta", "corr", "direction", "trend"],
            "jump_threshold": 0.3,
        },
    },
    "spec_moe_h64_w10": {
        "moe": {
            "pred_side_residual": {
                "enable": True, "corrector_hidden": 64, "init_alpha": -3.0,
                "specialization_weight": 1.0,
            },
        },
    },
    # === Penalty-aware Prediction-side MoE Residual (新设计) ===
    "pred_residual_h32": {
        "moe": {
            "pred_side_residual": {"enable": True, "corrector_hidden": 32, "init_alpha": -3.0},
        },
    },
    "pred_residual_h64": {
        "moe": {
            "pred_side_residual": {"enable": True, "corrector_hidden": 64, "init_alpha": -3.0},
        },
    },
    "pred_residual_h64_a1": {
        "moe": {
            "pred_side_residual": {"enable": True, "corrector_hidden": 64, "init_alpha": -1.0},
        },
    },
    # === Ablation: 测 MoE / KNN / 两者独立贡献 ===
    # 关掉 MoE Penalty（仅保留 cluster predictor + KNN hybrid + MAE objective）
    "ablation_no_moe": {
        "moe": {"enable": False},
    },
    # 关掉 KNN hybrid（仅保留 MoE Penalty + MAE）
    "ablation_no_knn": {
        "knn_hybrid": {"enable": False},
    },
    # 同时关掉 MoE 和 KNN（最干净的 base model）
    "ablation_no_moe_no_knn": {
        "moe": {"enable": False},
        "knn_hybrid": {"enable": False},
    },
    # === GPT v16: absolute_decay (epoch 起点 cosine decay) ===
    # 让 decay 真正落在 best_epoch=[16,21,28] 之前
    "ABD_h32_v16_dec_e8_to_005": {
        "moe": {
            "gate_hidden_dim": 32, "gate_temperature": 1.0,
            "gate_entropy_weight": 0.008, "gate_balance_weight": 0.02,
            "gate_balance_schedule": {
                "enable": True, "decay_start_epoch": 8, "decay_end_epoch": 32, "decay_end_value": 0.005,
            },
            "residual_gate": {"enable": True, "alpha": 0.7},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    "ABD_h32_v16_dec_e12_to_005": {
        "moe": {
            "gate_hidden_dim": 32, "gate_temperature": 1.0,
            "gate_entropy_weight": 0.008, "gate_balance_weight": 0.02,
            "gate_balance_schedule": {
                "enable": True, "decay_start_epoch": 12, "decay_end_epoch": 36, "decay_end_value": 0.005,
            },
            "residual_gate": {"enable": True, "alpha": 0.7},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    "ABD_h32_v16_dec_e16_to_005": {
        "moe": {
            "gate_hidden_dim": 32, "gate_temperature": 1.0,
            "gate_entropy_weight": 0.008, "gate_balance_weight": 0.02,
            "gate_balance_schedule": {
                "enable": True, "decay_start_epoch": 16, "decay_end_epoch": 40, "decay_end_value": 0.005,
            },
            "residual_gate": {"enable": True, "alpha": 0.7},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    "ABD_h32_v16_dec_e12_to_010": {
        "moe": {
            "gate_hidden_dim": 32, "gate_temperature": 1.0,
            "gate_entropy_weight": 0.008, "gate_balance_weight": 0.02,
            "gate_balance_schedule": {
                "enable": True, "decay_start_epoch": 12, "decay_end_epoch": 36, "decay_end_value": 0.010,
            },
            "residual_gate": {"enable": True, "alpha": 0.7},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    # === GPT v17: front_strong_late_weak (step decay) ===
    "ABD_h32_v17_front_b03_late_b005": {
        "moe": {
            "gate_hidden_dim": 32, "gate_temperature": 1.0,
            "gate_entropy_weight": 0.008,
            "gate_balance_weight": 0.03,                              # 前期更强
            "gate_balance_schedule": {
                "enable": True, "step_epoch": 12, "step_value_after": 0.005,
            },
            "residual_gate": {"enable": True, "alpha": 0.7},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    "ABD_h32_v17_front_b03_late_b01": {
        "moe": {
            "gate_hidden_dim": 32, "gate_temperature": 1.0,
            "gate_entropy_weight": 0.008, "gate_balance_weight": 0.03,
            "gate_balance_schedule": {
                "enable": True, "step_epoch": 12, "step_value_after": 0.01,
            },
            "residual_gate": {"enable": True, "alpha": 0.7},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    "ABD_h32_v17_front_b025_late_b005": {
        "moe": {
            "gate_hidden_dim": 32, "gate_temperature": 1.0,
            "gate_entropy_weight": 0.008, "gate_balance_weight": 0.025,
            "gate_balance_schedule": {
                "enable": True, "step_epoch": 12, "step_value_after": 0.005,
            },
            "residual_gate": {"enable": True, "alpha": 0.7},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    # v17 系列：GPT v13 — Dynamic Expert Bias (Loss-Free Balancing)
    # 给 logits 加 EMA bias：闲置 expert bias 上升，过用 expert bias 下降
    "ABD_h32_v17_dynbias_005": {
        "moe": {
            "gate_hidden_dim": 32, "gate_temperature": 1.0,
            "gate_entropy_weight": 0.008, "gate_balance_weight": 0.02,
            "dynamic_expert_bias": {"eta": 0.005, "clamp": 1.0},
            "residual_gate": {"enable": True, "alpha": 0.7},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    "ABD_h32_v17_dynbias_01": {
        "moe": {
            "gate_hidden_dim": 32, "gate_temperature": 1.0,
            "gate_entropy_weight": 0.008, "gate_balance_weight": 0.02,
            "dynamic_expert_bias": {"eta": 0.01, "clamp": 1.0},
            "residual_gate": {"enable": True, "alpha": 0.7},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    "ABD_h32_v17_dynbias_02": {
        "moe": {
            "gate_hidden_dim": 32, "gate_temperature": 1.0,
            "gate_entropy_weight": 0.008, "gate_balance_weight": 0.02,
            "dynamic_expert_bias": {"eta": 0.02, "clamp": 1.0},
            "residual_gate": {"enable": True, "alpha": 0.7},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    # 不带 balance loss 的 dynbias-only（看 dynbias 是否能替代 balance）
    "ABD_h32_v17_dynbias_only": {
        "moe": {
            "gate_hidden_dim": 32, "gate_temperature": 1.0,
            "gate_entropy_weight": 0.008,
            "gate_balance_weight": 0.0,
            "dynamic_expert_bias": {"eta": 0.01, "clamp": 1.0},
            "residual_gate": {"enable": True, "alpha": 0.7},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    "ABD_h32_v16_decay_aggressive": {
        "moe": {
            "gate_hidden_dim": 32, "gate_temperature": 1.0,
            "gate_entropy_weight": 0.008, "gate_balance_weight": 0.02,
            "gate_balance_schedule": {
                "enable": True,
                "warmup_ratio": 0.0, "rampup_ratio": 0.0,
                "decay_ratio": 0.8, "end_factor": 0.1,  # 后 80% decay 到 0.1x
            },
            "residual_gate": {"enable": True, "alpha": 0.7},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    "ABD_h32_v15_hat": {
        "moe": {
            "gate_hidden_dim": 32, "gate_temperature": 1.0,
            "gate_entropy_weight": 0.008, "gate_balance_weight": 0.02,
            "gate_balance_schedule": {
                "enable": True,
                "warmup_ratio": 0.15, "rampup_ratio": 0.2,
                "decay_ratio": 0.2, "end_factor": 0.5,
            },
            "residual_gate": {"enable": True, "alpha": 0.7},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    "ABD_h32_v12_T1_b02": {
        "moe": {
            "gate_hidden_dim": 32,
            "gate_temperature": 1.0,             # 不动 T
            "gate_entropy_weight": 0.008,
            "gate_balance_weight": 0.02,
            "residual_gate": {"enable": True, "alpha": 0.7},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    # v9 极小强度：进一步降低看是否还有微小提升
    "ABD_h32_v9_reg_xlow": {
        "moe": {
            "gate_hidden_dim": 32,
            "gate_entropy_weight": 0.005,
            "gate_balance_weight": 0.0025,
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    # v7：v5_reg (entropy+balance) + mixup_v3 (自适应+衰减) + K-aware safeguard 全合并
    # 目标：跨数据集统一最优 — ETTh2 safeguard、ETTh1 mixup+reg 双开、ETTm1 mixup 自动关 + reg 开
    "ABD_h32_v7_unified": {
        "moe": {
            "gate_hidden_dim": 32,
            "min_k_for_extensions": 3,
            "safeguard_hidden_dim": 64,
            "gate_entropy_weight": 0.01,    # v5_reg
            "gate_balance_weight": 0.005,    # v5_reg
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
            "gate_mixup": {
                "enable": True,
                "alpha": 0.4,
                "weight": 0.1,
                "mode": "kl",
                "auto_full_size": 12000,
                "auto_zero_size": 30000,
                "decay_schedule": "cosine",
                "decay_end_factor": 0.2,
            },
        },
    },
    # v3 综合强化版：K-aware 保护 + 阈值调严
    "ABD_h32_mixup_v3": {
        "moe": {
            "gate_hidden_dim": 32,
            "min_k_for_extensions": 3,      # 关键：K<3 时全部扩展自动禁用
            "pred_aware": {
                "enable": True,
                "use_pred_features": True,
                "use_penalty_input": False,
            },
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
            "gate_mixup": {
                "enable": True,
                "alpha": 0.4,
                "weight": 0.1,
                "mode": "kl",
                "auto_full_size": 12000,    # 调严：12k 以下全开
                "auto_zero_size": 30000,    # 关键：30k 以上完全关闭（让 ETTm1 完全恢复优势）
                "decay_schedule": "cosine",
                "decay_end_factor": 0.2,
            },
        },
    },
}


def make_config(base_cfg_path: str, variant: str, out_root: str, epochs: int, pred_len: int = 0, seed: int = 0) -> str:
    cfg = yaml.safe_load(open(base_cfg_path, "r", encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    deep_update(cfg, VARIANTS[variant])
    # 减少 epochs 控制时间
    if epochs > 0:
        cfg.setdefault("train", {})["epochs"] = epochs
    # 覆盖 horizon
    if pred_len > 0:
        cfg.setdefault("window", {})["pred_len"] = pred_len
    # 覆盖 seed（multi-seed 实验用）
    if seed > 0:
        cfg.setdefault("exp", {})["seed"] = int(seed)
    # 改输出目录
    cfg["exp"]["out_dir"] = os.path.join(out_root, variant)
    cfg["exp"]["name"] = f"compare_moe_ext_{variant}"
    # corr/portrait/knn 都重定向到子目录避免相互覆盖
    sub = cfg["exp"]["out_dir"]
    if "corr" in cfg and "save_path" in cfg["corr"]:
        cfg["corr"]["save_path"] = os.path.join(sub, "corr.npy")
    if "portrait" in cfg and "out_dir" in cfg["portrait"]:
        cfg["portrait"]["out_dir"] = os.path.join(sub, "cluster_portraits")
    if "knn_hybrid" in cfg and "path" in cfg["knn_hybrid"]:
        cfg["knn_hybrid"]["path"] = os.path.join(sub, "knn_shape_bank.pt")
    # 静默控制台
    cfg.setdefault("console", {})["quiet"] = True
    # 保存
    os.makedirs(sub, exist_ok=True)
    out_path = os.path.join(sub, "config.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return out_path


def run_variant(cfg_path: str, log_path: str) -> int:
    cmd = [sys.executable, "-m", "src.train", "--config", cfg_path]
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), stdout=logf, stderr=subprocess.STDOUT)
    return proc.returncode


def collect_metrics(out_dir: str) -> dict:
    summary_path = os.path.join(out_dir, "run_summary.json")
    if not os.path.isfile(summary_path):
        return {"error": "run_summary.json missing"}
    d = json.load(open(summary_path, "r", encoding="utf-8"))
    val = d.get("val", {})
    test = d.get("test", {})
    selected = d.get("selected", {})
    return {
        "val_mse": val.get("avg_mse"),
        "val_mae": val.get("avg_mae"),
        "test_mse_base": test.get("avg_mse"),
        "test_mae_base": test.get("avg_mae"),
        "selected_variant": selected.get("variant"),
        "selected_mse": selected.get("avg_mse"),
        "selected_mae": selected.get("avg_mae"),
        "best_epoch": d.get("best_epoch"),
        "elapsed": d.get("timing", {}).get("total_seconds") or d.get("timing", {}).get("total"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default="configs/ETTh1.yaml")
    ap.add_argument("--out", type=str, default="outputs/_compare_moe_ext")
    ap.add_argument("--epochs", type=int, default=30, help="缩短 epochs 控制实验时间")
    ap.add_argument("--pred_len", type=int, default=0, help=">0 时覆盖 base config 的 window.pred_len")
    ap.add_argument("--seed", type=int, default=0, help=">0 时覆盖 base config 的 exp.seed (multi-seed 实验用)")
    ap.add_argument("--variants", type=str, default="baseline,A,AD,ABD")
    args = ap.parse_args()

    base_path = str(PROJECT_ROOT / args.base) if not os.path.isabs(args.base) else args.base
    out_root = str(PROJECT_ROOT / args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(out_root, exist_ok=True)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    print(f"[compare] base={base_path}")
    print(f"[compare] out_root={out_root}")
    print(f"[compare] epochs={args.epochs}")
    print(f"[compare] variants={variants}")

    results = {}
    for v in variants:
        if v not in VARIANTS:
            print(f"[skip] unknown variant: {v}")
            continue
        cfg_path = make_config(base_path, v, out_root, args.epochs, pred_len=args.pred_len, seed=args.seed)
        log_path = os.path.join(out_root, v, "run.log")
        t0 = time.perf_counter()
        print(f"[run]  variant={v} ...")
        rc = run_variant(cfg_path, log_path)
        elapsed = time.perf_counter() - t0
        sub_out = os.path.join(out_root, v)
        metrics = collect_metrics(sub_out)
        metrics["return_code"] = rc
        metrics["wall_seconds"] = elapsed
        results[v] = metrics
        if rc != 0:
            print(f"[fail] variant={v} rc={rc} log={log_path}")
        else:
            print(
                f"[done] variant={v} time={elapsed:.1f}s "
                f"val_mse={metrics.get('val_mse'):.5f} val_mae={metrics.get('val_mae'):.5f} "
                f"test_mse={metrics.get('test_mse_base'):.5f} test_mae={metrics.get('test_mae_base'):.5f}"
            )

    # 输出对比表
    print("\n" + "=" * 90)
    print(f"{'variant':<10}{'val_mse':>12}{'val_mae':>12}{'test_mse':>12}{'test_mae':>12}{'sel_mse':>12}{'sel_mae':>12}{'sec':>8}")
    print("-" * 90)
    base = results.get("baseline", {})
    for v in variants:
        r = results.get(v, {})
        if "error" in r or r.get("return_code") != 0:
            print(f"{v:<10}  FAILED ({r.get('error') or r.get('return_code')})")
            continue
        val_mse = r.get("val_mse") or 0.0
        val_mae = r.get("val_mae") or 0.0
        test_mse = r.get("test_mse_base") or 0.0
        test_mae = r.get("test_mae_base") or 0.0
        sel_mse = r.get("selected_mse") or 0.0
        sel_mae = r.get("selected_mae") or 0.0
        sec = r.get("wall_seconds") or 0.0
        print(f"{v:<10}{val_mse:>12.5f}{val_mae:>12.5f}{test_mse:>12.5f}{test_mae:>12.5f}{sel_mse:>12.5f}{sel_mae:>12.5f}{sec:>8.1f}")
    # 相对 baseline 的改进
    if base.get("test_mse_base") and base.get("test_mae_base"):
        print("-" * 90)
        print(f"{'Δ vs base':<10}{'val_mse':>12}{'val_mae':>12}{'test_mse':>12}{'test_mae':>12}")
        for v in variants:
            if v == "baseline":
                continue
            r = results.get(v, {})
            if r.get("return_code") != 0:
                continue
            d_val_mse = (r["val_mse"] - base["val_mse"]) / base["val_mse"] * 100
            d_val_mae = (r["val_mae"] - base["val_mae"]) / base["val_mae"] * 100
            d_test_mse = (r["test_mse_base"] - base["test_mse_base"]) / base["test_mse_base"] * 100
            d_test_mae = (r["test_mae_base"] - base["test_mae_base"]) / base["test_mae_base"] * 100
            print(f"{v:<10}{d_val_mse:>+11.2f}%{d_val_mae:>+11.2f}%{d_test_mse:>+11.2f}%{d_test_mae:>+11.2f}%")
    print("=" * 90)
    # 保存 JSON
    with open(os.path.join(out_root, "compare_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {os.path.join(out_root, 'compare_results.json')}")


if __name__ == "__main__":
    main()
