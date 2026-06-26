"""
瀵规瘮 MoE Penalty 妯″潡鎵╁睍锛圓: Prediction-Aware / B: Sigmoid Branch / D: Penalty EMA锛夌殑鏁堟灉銆?

璺?4 缁勯厤缃細baseline / +A / +AD / +ABD锛屾瘡缁勭敤鐩稿悓 seed銆佺浉鍚?epochs锛?
浠?moe.pred_aware / moe.penalty_ema / moe.sigmoid_branch 寮€鍏充笉鍚屻€?
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
    """Recursive in-place update; patch values override matching keys."""
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


VARIANTS = {
    "baseline": {
        # 鍏ㄩ儴榛樿鍏抽棴
    },
    # 鈽?鐪熸骞插噣鐨?baseline锛氭樉寮忕鐢ㄦ墍鏈?spec_moe / v12 鍒涙柊锛堢敤浜庝弗鏍?ablation锛?
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
    # 瀹夊叏鍙樹綋锛氬幓鎺?use_penalty_input锛堥伩鍏?val 杩囨嫙鍚堬級
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
    # Compact 鍙樹綋锛氬湪 ABD_safe 鍩虹涓婂噺灏忔墿灞曞閲?+ 鍔犲ぇ姝ｅ垯锛岄獙璇併€岃繃鍙傛暟鍖栧鑷?val 杩囨嫙鍚堛€嶅亣璁?
    "ABD_compact": {
        "moe": {
            "gate_hidden_dim": 32,   # 64 鈫?32锛屽噺鍗?gate 鍐呴儴瀹归噺
            "pred_aware": {
                "enable": True,
                "use_pred_features": True,
                "use_penalty_input": False,
            },
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},  # gamma 0.4鈫?.2锛宐ias 鏇村己鍘嬪埗
        },
        "train": {
            "weight_decay": 0.001,   # 0.0002 鈫?0.001锛?x 鍔犲ぇ姝ｅ垯
        },
    },
    # Compact 鍙樹綋 2锛氫粎鍑忓閲忥紝涓嶅姩 wd锛堢敤浜庨殧绂讳袱鑰呮晥鏋滐級
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
    # ABD_h32 + Gate-Side Mixup Consistency Regularization (鏂规 1)
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
    # ABD_h32 + Penalty Replay Buffer (鏂规 2 鈥?瀹炶瘉鏈夊锛屼粎淇濈暀浣滃弽渚?
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
    # 鏂规 1 寮哄寲鐗堬細鑷€傚簲鏉冮噸 + cosine 琛板噺
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
                "auto_full_size": 15000,    # 鈮?5k 鈫?鍏ㄥ紑
                "auto_zero_size": 50000,    # 鈮?0k 鈫?鑷姩鍏?
                "decay_schedule": "cosine", # 璁粌鍚庢湡鍑忓帇
                "decay_end_factor": 0.2,    # 鏈湡淇濈暀 20% 寮哄害
            },
        },
    },
    # v5 绯诲垪锛氬熀浜?GPT 寤鸿鐨勩€岃 gate 鏇磋蒋銆嶄笁浠跺锛坱emperature / entropy / balance锛?
    # 鍦?ETTm1锛堝ぇ鏁版嵁锛孠=3锛屽凡鏈?ABD_h32 -1.56% 鏀硅繘锛変笂鐪嬫槸鍚﹁兘缁х画绐佺牬
    "ABD_h32_v5_temp": {
        "moe": {
            "gate_hidden_dim": 32,
            "gate_temperature": 2.0,    # 浠呮媺楂?T锛岃 softmax 鏇村钩婊?
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    "ABD_h32_v5_reg": {
        "moe": {
            "gate_hidden_dim": 32,
            "gate_entropy_weight": 0.01,  # 榧撳姳 gate 淇濇寔鐔碉紙涓嶈繃搴﹁嚜淇★級
            "gate_balance_weight": 0.005, # 璐熻浇鍧囪　
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
    # v6 绯诲垪锛歷5_reg 寮哄害姊害锛圗TTm1 涓?v5_reg 鍙栧緱 -1.70% 绐佺牬锛岀湅鏇村己鏄惁缁х画鎺ㄨ繘锛?
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
    # v8 绯诲垪锛歷5_reg + 杩涢樁灏濊瘯锛堢粏璋?reg 寮哄害 / dropout / mixup_lite锛?
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
            "gate_dropout": 0.1,             # GPT 鎬濊矾 3
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    # v9 瑙ｈ€︼細鍗曠嫭鐪?entropy / balance 鍝釜璧蜂富瀵间綔鐢?
    "ABD_h32_v9_entropy_only": {
        "moe": {
            "gate_hidden_dim": 32,
            "gate_entropy_weight": 0.008,    # 浠?entropy
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
            "gate_balance_weight": 0.004,    # 浠?balance
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    # v10 绯诲垪锛氬熀浜?GPT 璇婃柇 鈥?Residual Gate 闃?routing collapse
    "ABD_h32_v10_residual_03": {
        "moe": {
            "gate_hidden_dim": 32,
            "gate_entropy_weight": 0.008,
            "gate_balance_weight": 0.004,
            "residual_gate": {"enable": True, "alpha": 0.3},  # gate 褰卞搷 30%
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
            "residual_gate": {"enable": True, "alpha": 0.5},  # gate 褰卞搷 50%
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
            "residual_gate": {"enable": True, "alpha": 0.7},  # gate 褰卞搷 70%
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    # v11 绮炬壂 alpha 鍦?0.6-0.85 鍖洪棿纭畾鐢滅偣
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
    # v12 绯诲垪锛氬熀浜?GPT 璇婃柇銆宺aw gate 95% 濉岀缉銆嶇洿鎺ユ不鐞?
    # 鍦?alpha=0.7 鍩虹涓?(residual gate) 鍔犲己 balance + 璋冮珮 temperature 鐩存帴璁?raw gate 涓嶅缂?
    "ABD_h32_v12_T2_b01": {
        "moe": {
            "gate_hidden_dim": 32,
            "gate_temperature": 2.0,
            "gate_entropy_weight": 0.008,
            "gate_balance_weight": 0.01,         # 姣?v8_reg_low 0.004 鍔犲己 2.5x
            "residual_gate": {"enable": True, "alpha": 0.7},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    "ABD_h32_v12_T3_b01": {
        "moe": {
            "gate_hidden_dim": 32,
            "gate_temperature": 3.0,             # GPT 鎺ㄨ崘 T=3
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
            "gate_balance_weight": 0.02,         # 鏇村己 balance
            "residual_gate": {"enable": True, "alpha": 0.7},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    # 浠呭姞寮?balance 涓嶅姩 T锛堢湅 balance 鍗曠嫭璐＄尞锛?
    # v13 绯诲垪锛氬熀浜?v12_T1_b02 绐佺牬 (-2.90%) 杩涗竴姝ョ簿鎵?balance 寮哄害
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
    # 鍚屾椂涔熸壂 entropy 鐪嬫槸鍚﹂渶瑕佽仈鍔?
    "ABD_h32_v13_b02_e02": {
        "moe": {
            "gate_hidden_dim": 32,
            "gate_temperature": 1.0,
            "gate_entropy_weight": 0.02,    # entropy 涔熷姞鍊?
            "gate_balance_weight": 0.02,
            "residual_gate": {"enable": True, "alpha": 0.7},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    # v14 绯诲垪锛氬湪 balance=0.02 涓嬮噸鎵?alpha锛岄獙璇?GPT 鍋囪
    # 銆宐alance 寮轰簡 5x 鍚?alpha 鐢滅偣鍙兘浠?0.7 涓婄Щ鍒?0.75/0.8銆?
    "ABD_h32_v14_alpha_065_b02": {
        "moe": {
            "gate_hidden_dim": 32,
            "gate_temperature": 1.0,
            "gate_entropy_weight": 0.008,
            "gate_balance_weight": 0.02,    # 涓?v12_T1_b02 鍚?
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
    # v15 绯诲垪锛氬熀浜?v12_T1_b02 (-2.90%) 娴?balance schedule 鏈哄埗 (GPT v15 楠岃瘉)
    # warmup: 鏃╂湡涓嶅姞 balance锛岀粰 expert 鍒嗗寲鏃堕棿锛屽啀绾︽潫
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
    # decay: 鏃╂湡寮?balance 闃叉棭鏈熷缂╋紝鍚庢湡鍑忓帇璁?gate 鑷敱瀛︿範
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
    # warmup + decay: 瀹屾暣 hat-shape schedule
    # v16: 璁?decay 鐪熸钀藉湪 best_epoch 涔嬪墠锛屽己鍒跺叾褰卞搷閫夋嫨
    # epochs=80 + decay 50%锛堣 best_epoch 杩涘叆 decay 鍖猴級
    "ABD_h32_v16_decay_early": {
        "moe": {
            "gate_hidden_dim": 32, "gate_temperature": 1.0,
            "gate_entropy_weight": 0.008, "gate_balance_weight": 0.02,
            "gate_balance_schedule": {
                "enable": True,
                "warmup_ratio": 0.0, "rampup_ratio": 0.0,
                "decay_ratio": 0.6, "end_factor": 0.2,  # 鍚?60% decay 鍒?0.2x
            },
            "residual_gate": {"enable": True, "alpha": 0.7},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    # v18 绯诲垪锛氱畝鍖栫増 oracle supervision 鈥斺€?KL(softmax(pen/蟿) || gate_probs) 鐩存帴鐩戠潱 gate
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
    # 鍙嶅悜锛氬皬 penalty 楂樻潈閲嶏紙鐪?oracle 鏂瑰悜鏄惁閲嶈锛?
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
    # 涓嶅甫 balance loss + oracle锛堢湅 oracle 鑳藉惁鏇夸唬 balance 闃插缂╋級
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
    # === 鍋囪楠岃瘉锛歁oE 鍦ㄦ潈閲嶅ぇ / MSE 寮?/ MAE 鍏?鏃舵槸鍚︾湡璧蜂綔鐢?===
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
    # === Penalty 绫诲瀷浜掕ˉ鎬?ablation ===
    # 鍋囪锛歡ate 閫?level/amp_under 鏄洜涓鸿繖淇╀笌 MSE 鍚屾柟鍚戯紱鍚敤绾簰琛?penalty 璁?gate 蹇呴』閫変簰琛ョ被鍨?
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
            "jump_threshold": 0.3,  # 璋冧綆璁?jump 鐪熸瑙﹀彂
        },
    },
    # 鍏?12 绉?penalty 姹狅紝璁?gate 鑷繁閫?
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
    # 寮虹儓榧撳姳 gate 閫変簰琛?penalty锛氫簰琛?位 澶э紝閲嶅彔 位 灏?
    "v12_boost_complement": {
        "moe": {
            "lambda_init": {
                # 浜掕ˉ penalty 澶ф潈閲嶏紙10x锛?
                "jump": 1.0, "corr": 1.0, "direction": 1.0, "trend": 1.0,
                # 閲嶅彔 penalty 灏忔潈閲嶏紙淇濈暀灏戦噺鐩戠潱锛?
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
    # === Penalty-Supervised Expert MoE (鏍稿績鏋舵瀯鏀归€? ===
    # 姣忎釜 expert_p 琚搴?penalty_p 鍗曠嫭鐩戠潱锛屽己鍒?specialization
    # 鐮撮櫎"鍙傛暟鍏变韩 鈫?澶?penalty 姊害鍐茬獊"鐨勮瘏鍜?
    "spec_moe_w05": {
        "moe": {
            "pred_side_residual": {
                "enable": True, "corrector_hidden": 32, "init_alpha": -3.0,
                "specialization_weight": 0.5,
            },
        },
    },
    # 鈽呪槄 Hard-select Correction MoE锛歜ase + 鍗曚竴 correction锛堢敤鎴风簿纭弿杩扮殑璁捐锛?
    "spec_moe_hard_correction": {
        "moe": {
            "pred_side_residual": {
                "enable": True, "corrector_hidden": 32, "init_alpha": -3.0,
                "specialization_weight": 1.0,
                "select_mode": "hard",     # 鈽?鍏抽敭锛歨ard select 涓嶆贩鍚?
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
                "specialization_weight": 5.0,   # 鍔犲己 specialization
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
    # spec_moe + 浜掕ˉ penalty 姹狅細寮哄埗 expert 瀛?MSE 鐪嬩笉鍒扮殑鏂瑰悜
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
    # spec_moe + boost 浜掕ˉ penalty 鏉冮噸锛坧ool 鍚叏閮紝浣嗕簰琛ョ殑鏉冮噸澶э級
    # 鈽呪槄 Hard Switch MoE 鈥?鐪熸"涓嶆贩鍚?鐨?penalty 璺敱锛堟牳蹇冨垱鏂帮級
    # K 脳 P 涓畬鏁?predictor锛屾瘡涓牱鏈?hard-select 璧板崟涓€璺緞
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
    # 鍔犲己 specialization锛氳 penalty 鐪熸濉戦€?expert 鍙傛暟
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
    # 瀹归噺鏇村ぇ鐨?hard switch
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
    # 鈽?Input-driven Dynamic MoE锛氬惎鐢ㄥ叏閮?12 绉?penalty锛岃 gate per-sample 鑷€傚簲閫夋嫨
    # 杩欐墠鏄湡姝ｇ殑 MoE锛氭帹鐞嗘椂涓嶉渶瑕?val 閫?pool锛実ate 鐪?x 鐩存帴璺敱
    "input_driven_full_pool": {
        "moe": {
            "pred_side_residual": {
                "enable": True, "corrector_hidden": 32, "init_alpha": -3.0,
                "specialization_weight": 1.0,
            },
            "topk": 2,  # 鍏佽 gate 鍚屾椂婵€娲?2 涓?expert锛屾洿鏈夎矾鐢辩┖闂?
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
    # 涓瓑姹狅細8 涓?penalty 鐣欎綑鍦帮紙鍘绘帀 amp銆乺ange銆乨iff_amp 杩欎簺绾噸鍙犵殑锛?
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
    # 鏁版嵁椹卞姩 penalty 閫夋嫨锛氬垹 jump/level (ETTm1 涓婃棤鎰忎箟)锛屼繚鐣?amp_under/delta + 鍔?jitter/smooth
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
    # 鍙敤 amp_under + jitter锛堜竴涓箙搴︼紝涓€涓钩婊戝害锛?
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
                # 浜掕ˉ penalty 寮烘潈閲?
                "jump": 0.5, "corr": 0.5, "direction": 0.5, "trend": 0.5,
                # MSE-閲嶅彔 penalty 寮辨潈閲?
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
    # === Penalty-aware Prediction-side MoE Residual (鏂拌璁? ===
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
    "ablation_no_moe": {
        "moe": {"enable": False},
    },
    # === GPT v16: absolute_decay (epoch 璧风偣 cosine decay) ===
    # 璁?decay 鐪熸钀藉湪 best_epoch=[16,21,28] 涔嬪墠
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
            "gate_balance_weight": 0.03,                              # 鍓嶆湡鏇村己
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
    # v17 绯诲垪锛欸PT v13 鈥?Dynamic Expert Bias (Loss-Free Balancing)
    # 缁?logits 鍔?EMA bias锛氶棽缃?expert bias 涓婂崌锛岃繃鐢?expert bias 涓嬮檷
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
    # 涓嶅甫 balance loss 鐨?dynbias-only锛堢湅 dynbias 鏄惁鑳芥浛浠?balance锛?
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
                "decay_ratio": 0.8, "end_factor": 0.1,  # 鍚?80% decay 鍒?0.1x
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
            "gate_temperature": 1.0,             # 涓嶅姩 T
            "gate_entropy_weight": 0.008,
            "gate_balance_weight": 0.02,
            "residual_gate": {"enable": True, "alpha": 0.7},
            "pred_aware": {"enable": True, "use_pred_features": True, "use_penalty_input": False},
            "penalty_ema": {"enable": True, "decay": 0.9},
            "sigmoid_branch": {"enable": True, "gamma": 0.2, "init_bias": -2.0},
        },
    },
    # v9 鏋佸皬寮哄害锛氳繘涓€姝ラ檷浣庣湅鏄惁杩樻湁寰皬鎻愬崌
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
    # v7锛歷5_reg (entropy+balance) + mixup_v3 (鑷€傚簲+琛板噺) + K-aware safeguard 鍏ㄥ悎骞?
    # 鐩爣锛氳法鏁版嵁闆嗙粺涓€鏈€浼?鈥?ETTh2 safeguard銆丒TTh1 mixup+reg 鍙屽紑銆丒TTm1 mixup 鑷姩鍏?+ reg 寮€
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
    # v3 缁煎悎寮哄寲鐗堬細K-aware 淇濇姢 + 闃堝€艰皟涓?
    "ABD_h32_mixup_v3": {
        "moe": {
            "gate_hidden_dim": 32,
            "min_k_for_extensions": 3,      # 鍏抽敭锛欿<3 鏃跺叏閮ㄦ墿灞曡嚜鍔ㄧ鐢?
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
                "auto_full_size": 12000,    # 璋冧弗锛?2k 浠ヤ笅鍏ㄥ紑
                "auto_zero_size": 30000,    # 鍏抽敭锛?0k 浠ヤ笂瀹屽叏鍏抽棴锛堣 ETTm1 瀹屽叏鎭㈠浼樺娍锛?
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
    # 鍑忓皯 epochs 鎺у埗鏃堕棿
    if epochs > 0:
        cfg.setdefault("train", {})["epochs"] = epochs
    # 瑕嗙洊 horizon
    if pred_len > 0:
        cfg.setdefault("window", {})["pred_len"] = pred_len
    # 瑕嗙洊 seed锛坢ulti-seed 瀹為獙鐢級
    if seed > 0:
        cfg.setdefault("exp", {})["seed"] = int(seed)
    # 鏀硅緭鍑虹洰褰?
    cfg["exp"]["out_dir"] = os.path.join(out_root, variant)
    cfg["exp"]["name"] = f"compare_moe_ext_{variant}"
    # corr and portrait outputs are redirected to subdirectories
    sub = cfg["exp"]["out_dir"]
    if "corr" in cfg and "save_path" in cfg["corr"]:
        cfg["corr"]["save_path"] = os.path.join(sub, "corr.npy")
    if "portrait" in cfg and "out_dir" in cfg["portrait"]:
        cfg["portrait"]["out_dir"] = os.path.join(sub, "cluster_portraits")
    # 闈欓粯鎺у埗鍙?
    cfg.setdefault("console", {})["quiet"] = True
    # 淇濆瓨
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
    ap.add_argument("--epochs", type=int, default=30, help="缂╃煭 epochs 鎺у埗瀹為獙鏃堕棿")
    ap.add_argument("--pred_len", type=int, default=0, help=">0 鏃惰鐩?base config 鐨?window.pred_len")
    ap.add_argument("--seed", type=int, default=0, help=">0 鏃惰鐩?base config 鐨?exp.seed (multi-seed 瀹為獙鐢?")
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

    # 杈撳嚭瀵规瘮琛?
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
    # 鐩稿 baseline 鐨勬敼杩?
    if base.get("test_mse_base") and base.get("test_mae_base"):
        print("-" * 90)
        print(f"{'螖 vs base':<10}{'val_mse':>12}{'val_mae':>12}{'test_mse':>12}{'test_mae':>12}")
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
    # 淇濆瓨 JSON
    with open(os.path.join(out_root, "compare_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {os.path.join(out_root, 'compare_results.json')}")


if __name__ == "__main__":
    main()
