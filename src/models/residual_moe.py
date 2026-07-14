import math
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from ..utils.cluster_memory import scatter_mean_bcl_to_bkl


class ChannelPatchPenaltyRouter(nn.Module):
    """Route shared penalty experts from causal input patches."""

    def __init__(
        self,
        *,
        input_len: int,
        pred_len: int,
        num_penalties: int,
        num_channels: int = 0,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        cfg = cfg or {}
        self.L = int(input_len)
        self.H = int(pred_len)
        self.P = int(num_penalties)
        self.C = int(num_channels)
        self.patch_len = int(cfg.get("patch_len", 24))
        self.hidden_dim = int(cfg.get("hidden_dim", 32))
        self.topk = int(cfg.get("topk", 1))
        self.temperature = max(float(cfg.get("temperature", 1.0)), 1.0e-6)
        self.noise_std = max(float(cfg.get("noise_std", 0.0)), 0.0)
        self.allow_skip = bool(cfg.get("allow_skip", True))
        self.inference_route_mode = str(
            cfg.get("inference_route_mode", "hard")
        ).strip().lower()
        self.training_route_mode = str(
            cfg.get("training_route_mode", "straight_through")
        ).strip().lower()
        self.skip_init_bias = float(cfg.get("skip_init_bias", -2.0))
        self.feature_clip = max(float(cfg.get("feature_clip", 8.0)), 0.0)
        self.use_base_forecast = bool(cfg.get("use_base_forecast", False))
        if self.inference_route_mode not in {"hard", "soft"}:
            raise ValueError(
                "patch_router.inference_route_mode must be hard or soft."
            )
        if self.training_route_mode not in {"straight_through", "soft"}:
            raise ValueError(
                "patch_router.training_route_mode must be straight_through or soft."
            )
        self.use_full_history_features = bool(
            cfg.get("use_full_history_features", False)
        )
        self.use_channel_identity_features = bool(
            cfg.get("use_channel_identity_features", False)
        )
        self.time_phase_periods = sorted(
            {
                int(period)
                for period in (cfg.get("time_phase_periods", []) or [])
                if int(period) > 1
            }
        )
        self.lagged_delta_periods = sorted(
            {
                int(period)
                for period in (cfg.get("lagged_delta_periods", []) or [])
                if int(period) > 0
            }
        )
        if self.use_channel_identity_features and self.C <= 0:
            raise ValueError(
                "patch_router channel identity features require num_channels."
            )
        raw_fixed_penalty = cfg.get("fixed_penalty_index_by_channel", None)
        if raw_fixed_penalty is None:
            fixed_penalty_c = torch.empty(0, dtype=torch.long)
        else:
            fixed_penalty_c = torch.as_tensor(
                raw_fixed_penalty,
                dtype=torch.long,
            ).reshape(-1)
            if bool(((fixed_penalty_c < -1) | (fixed_penalty_c >= self.P)).any().item()):
                raise ValueError(
                    "patch_router fixed_penalty_index_by_channel values must be -1 "
                    f"or in [0,{self.P - 1}]."
                )
        self.register_buffer(
            "fixed_penalty_index_by_channel_c",
            fixed_penalty_c,
            persistent=False,
        )
        regime_context_cfg = cfg.get("regime_context", {}) or {}
        if not isinstance(regime_context_cfg, dict):
            regime_context_cfg = {"enable": bool(regime_context_cfg)}
        self.regime_context_enable = bool(regime_context_cfg.get("enable", False))
        raw_regime_lengths = regime_context_cfg.get(
            "lengths",
            [192, 384, 672],
        )
        self.regime_context_lengths = sorted(
            {
                int(value)
                for value in raw_regime_lengths
                if int(value) > 0
            }
        ) if self.regime_context_enable else []
        hierarchical_cfg = cfg.get("hierarchical_recall", {}) or {}
        if not isinstance(hierarchical_cfg, dict):
            hierarchical_cfg = {"enable": bool(hierarchical_cfg)}
        self.hierarchical_recall_enable = bool(hierarchical_cfg.get("enable", False))
        self.adopt_threshold = float(hierarchical_cfg.get("adopt_threshold", 0.5))
        self.adopt_init_bias = float(hierarchical_cfg.get("adopt_init_bias", 0.0))
        utility_verifier_cfg = hierarchical_cfg.get("utility_verifier", {}) or {}
        if not isinstance(utility_verifier_cfg, dict):
            utility_verifier_cfg = {"enable": bool(utility_verifier_cfg)}
        self.utility_verifier_enable = bool(utility_verifier_cfg.get("enable", False))
        self.utility_verifier_temperature = max(
            float(utility_verifier_cfg.get("temperature", 0.25)),
            1.0e-6,
        )
        expert_risk_cfg = hierarchical_cfg.get("expert_conditional_risk", {}) or {}
        if not isinstance(expert_risk_cfg, dict):
            expert_risk_cfg = {"enable": bool(expert_risk_cfg)}
        self.expert_conditional_risk_enable = bool(expert_risk_cfg.get("enable", False))
        dual_utility_cfg = expert_risk_cfg.get("dual_signed_utility", {}) or {}
        if not isinstance(dual_utility_cfg, dict):
            dual_utility_cfg = {"enable": bool(dual_utility_cfg)}
        self.expert_risk_dual_signed_utility_enable = bool(
            self.expert_conditional_risk_enable
            and dual_utility_cfg.get("enable", False)
        )
        analytic_residual_cfg = dual_utility_cfg.get(
            "analytic_residual",
            {},
        ) or {}
        if not isinstance(analytic_residual_cfg, dict):
            analytic_residual_cfg = {"enable": bool(analytic_residual_cfg)}
        self.expert_risk_analytic_residual_enable = bool(
            self.expert_risk_dual_signed_utility_enable
            and analytic_residual_cfg.get("enable", False)
        )
        self.expert_risk_analytic_residual_floor = max(
            float(analytic_residual_cfg.get("relative_floor", 0.05)),
            1.0e-6,
        )
        independent_activation_cfg = dual_utility_cfg.get(
            "independent_activation",
            {},
        ) or {}
        if not isinstance(independent_activation_cfg, dict):
            independent_activation_cfg = {
                "enable": bool(independent_activation_cfg)
            }
        self.expert_risk_independent_activation_enable = bool(
            self.expert_risk_dual_signed_utility_enable
            and independent_activation_cfg.get("enable", False)
        )
        compositional_periodic_cfg = cfg.get(
            "compositional_periodic_gate",
            {},
        ) or {}
        if not isinstance(compositional_periodic_cfg, dict):
            compositional_periodic_cfg = {
                "enable": bool(compositional_periodic_cfg)
            }
        self.compositional_periodic_gate_enable = bool(
            compositional_periodic_cfg.get("enable", False)
        )
        self.expert_risk_decoupled_encoder = bool(
            expert_risk_cfg.get("decoupled_encoder", True)
        )
        self.expert_risk_candidate_aware = bool(
            expert_risk_cfg.get("candidate_aware", True)
        )
        self.expert_risk_candidate_compatibility = bool(
            expert_risk_cfg.get("candidate_compatibility", False)
        )
        temporal_domain_cfg = expert_risk_cfg.get(
            "temporal_domain_ensemble",
            {},
        ) or {}
        if not isinstance(temporal_domain_cfg, dict):
            temporal_domain_cfg = {"enable": bool(temporal_domain_cfg)}
        self.expert_risk_temporal_domain_enable = bool(
            self.expert_conditional_risk_enable
            and temporal_domain_cfg.get("enable", False)
        )
        self.expert_risk_temporal_domain_count = max(
            2,
            int(temporal_domain_cfg.get("num_domains", 6)),
        )
        self.expert_risk_temporal_domain_train_windows = int(
            temporal_domain_cfg.get("train_window_count", 0)
        )
        self.expert_risk_temporal_domain_combine = str(
            temporal_domain_cfg.get("combine", "mean")
        ).strip().lower()
        if self.expert_risk_temporal_domain_enable:
            if self.expert_risk_temporal_domain_train_windows <= 0:
                raise ValueError(
                    "temporal_domain_ensemble.train_window_count must be positive."
                )
            if self.expert_risk_temporal_domain_combine != "mean":
                raise ValueError(
                    "temporal_domain_ensemble.combine currently supports only mean."
                )
        self.expert_risk_proposal_candidate_aware = bool(
            self.expert_conditional_risk_enable
            and expert_risk_cfg.get(
                "proposal_candidate_aware",
                self.expert_risk_candidate_aware,
            )
        )
        self.expert_risk_proposal_threshold = float(
            expert_risk_cfg.get("proposal_threshold", 0.5)
        )
        self.expert_risk_proposal_topk = int(expert_risk_cfg.get("proposal_topk", 2))
        self.expert_risk_proposal_rescue_enable = bool(
            expert_risk_cfg.get("proposal_rescue", False)
        )
        self.expert_risk_temperature = max(
            float(expert_risk_cfg.get("temperature", 0.25)),
            1.0e-6,
        )
        lower_quantile_cfg = expert_risk_cfg.get("lower_quantile", {}) or {}
        if not isinstance(lower_quantile_cfg, dict):
            lower_quantile_cfg = {"enable": bool(lower_quantile_cfg)}
        self.expert_risk_lower_quantile_enable = bool(
            lower_quantile_cfg.get("enable", False)
        )
        self.expert_risk_lower_quantile = float(
            lower_quantile_cfg.get("quantile", 0.2)
        )
        default_adoption_source = (
            "lower_quantile"
            if self.expert_risk_lower_quantile_enable
            else "expected_utility"
        )
        self.expert_risk_adoption_source = str(
            expert_risk_cfg.get("adoption_source", default_adoption_source)
        ).strip().lower()
        utility_veto_cfg = expert_risk_cfg.get("utility_veto", {}) or {}
        if not isinstance(utility_veto_cfg, dict):
            utility_veto_cfg = {"enable": bool(utility_veto_cfg)}
        self.expert_risk_utility_veto_enable = bool(
            utility_veto_cfg.get("enable", False)
        )
        self.expert_risk_utility_veto_detach_features = bool(
            utility_veto_cfg.get("detach_features", True)
        )
        default_adopt_threshold = (
            0.5
            if self.expert_risk_adoption_source in {
                "benefit_probability",
                "utility_veto",
            }
            else 0.0
        )
        temporal_calibration_cfg = expert_risk_cfg.get("temporal_calibration", {}) or {}
        if not isinstance(temporal_calibration_cfg, dict):
            temporal_calibration_cfg = {"enable": bool(temporal_calibration_cfg)}
        raw_adopt_threshold_by_penalty = expert_risk_cfg.get(
            "adopt_threshold_by_penalty",
            None,
        )
        self.expert_risk_per_penalty_threshold_enable = bool(
            self.expert_conditional_risk_enable
            and (
                raw_adopt_threshold_by_penalty is not None
                or temporal_calibration_cfg.get("per_penalty", False)
            )
        )
        if raw_adopt_threshold_by_penalty is None:
            adopt_threshold_by_penalty = torch.full(
                (self.P,),
                float(expert_risk_cfg.get("adopt_threshold", default_adopt_threshold)),
            )
        else:
            adopt_threshold_by_penalty = torch.as_tensor(
                raw_adopt_threshold_by_penalty,
                dtype=torch.float32,
            ).reshape(-1)
            if int(adopt_threshold_by_penalty.numel()) != self.P:
                raise ValueError(
                    "patch_router adopt_threshold_by_penalty must have one value "
                    f"per penalty ({self.P}), got {int(adopt_threshold_by_penalty.numel())}."
                )
        self.expert_risk_adopt_threshold = (
            nn.Parameter(
                torch.tensor(
                    float(
                        expert_risk_cfg.get(
                            "adopt_threshold",
                            default_adopt_threshold,
                        )
                    )
                ),
                requires_grad=False,
            )
            if self.expert_conditional_risk_enable
            else None
        )
        self.expert_risk_adopt_threshold_by_penalty = (
            nn.Parameter(adopt_threshold_by_penalty, requires_grad=False)
            if self.expert_risk_per_penalty_threshold_enable
            else None
        )
        pairwise_rank_cfg = expert_risk_cfg.get("pairwise_rank", {}) or {}
        if not isinstance(pairwise_rank_cfg, dict):
            pairwise_rank_cfg = {"enable": bool(pairwise_rank_cfg)}
        self.expert_risk_pairwise_rank_enable = bool(
            pairwise_rank_cfg.get("enable", False)
        )
        self.expert_risk_pairwise_detach_features = bool(
            pairwise_rank_cfg.get("detach_features", True)
        )
        self.expert_risk_pairwise_temperature = max(
            float(pairwise_rank_cfg.get("temperature", 1.0)),
            1.0e-6,
        )
        self.feature_source = "input_base" if self.use_base_forecast else "input_only"
        if self.use_full_history_features:
            self.feature_source += "_full_history"
        if self.use_channel_identity_features:
            self.feature_source += "_channel_id"
        if self.time_phase_periods:
            self.feature_source += "_time_phase"
        if self.lagged_delta_periods:
            self.feature_source += "_lagged_delta"
        self.short_history_mode = str(
            cfg.get("short_history_mode", "error")
        ).strip().lower()
        if self.patch_len <= 0:
            raise ValueError("patch_router.patch_len must be positive.")
        if self.H % self.patch_len != 0:
            raise ValueError("patch_router.patch_len must divide pred_len exactly.")
        if self.short_history_mode not in {"error", "cycle"}:
            raise ValueError(
                "patch_router.short_history_mode must be error or cycle."
            )
        if self.L < self.H and self.short_history_mode == "error":
            raise ValueError(
                "patch_router with input_len < pred_len requires "
                "short_history_mode=cycle."
            )
        if self.L < self.H and self.L < self.patch_len:
            raise ValueError(
                "patch_router cycle mode requires input_len >= patch_len."
            )
        if self.P <= 0:
            raise ValueError("patch_router requires at least one penalty expert.")
        if self.hidden_dim <= 0:
            raise ValueError("patch_router.hidden_dim must be positive.")
        if self.regime_context_enable and len(self.regime_context_lengths) == 0:
            raise ValueError("patch_router regime_context requires at least one positive length.")
        if self.lagged_delta_periods and not self.regime_context_enable:
            raise ValueError(
                "patch_router lagged_delta_periods requires regime_context.enable=true."
            )
        if (
            self.lagged_delta_periods
            and max(self.regime_context_lengths) < self.L + max(self.lagged_delta_periods)
        ):
            raise ValueError(
                "patch_router regime context must cover input_len plus the largest "
                "lagged delta period."
            )
        if self.hierarchical_recall_enable and not self.allow_skip:
            raise ValueError("patch_router hierarchical recall gate requires allow_skip=true.")
        if self.expert_conditional_risk_enable and not self.hierarchical_recall_enable:
            raise ValueError(
                "patch_router expert_conditional_risk requires hierarchical_recall.enable=true."
            )
        if (
            self.expert_risk_dual_signed_utility_enable
            and self.expert_risk_adoption_source != "expected_utility"
        ):
            raise ValueError(
                "patch_router dual_signed_utility requires "
                "expert_conditional_risk.adoption_source=expected_utility."
            )
        if (
            self.expert_risk_dual_signed_utility_enable
            and self.expert_risk_pairwise_rank_enable
        ):
            raise ValueError(
                "patch_router dual_signed_utility ranks candidates directly and "
                "cannot be combined with pairwise_rank."
            )
        if (
            self.compositional_periodic_gate_enable
            and not self.expert_risk_dual_signed_utility_enable
        ):
            raise ValueError(
                "patch_router compositional_periodic_gate requires "
                "expert_conditional_risk.dual_signed_utility.enable=true."
            )
        if self.compositional_periodic_gate_enable and self.topk != 1:
            raise ValueError(
                "patch_router compositional_periodic_gate requires topk=1 so "
                "each P+e action contains exactly one adapter."
            )
        if self.expert_risk_dual_signed_utility_enable:
            assert self.expert_risk_adopt_threshold is not None
            if abs(float(self.expert_risk_adopt_threshold.item())) > 1.0e-12:
                raise ValueError(
                    "patch_router dual_signed_utility fixes skip utility at zero; "
                    "expert_conditional_risk.adopt_threshold must be 0."
                )
            if (
                self.expert_risk_adopt_threshold_by_penalty is not None
                and bool(
                    (
                        self.expert_risk_adopt_threshold_by_penalty.abs()
                        > 1.0e-12
                    ).any().item()
                )
            ):
                raise ValueError(
                    "patch_router dual_signed_utility requires zero per-penalty "
                    "adopt thresholds."
                )
        if (
            self.expert_risk_independent_activation_enable
            and self.compositional_periodic_gate_enable
        ):
            raise ValueError(
                "patch_router independent adapter activation is incompatible with "
                "the single-action compositional periodic gate."
            )
        if self.expert_conditional_risk_enable and self.utility_verifier_enable:
            raise ValueError(
                "patch_router utility_verifier and expert_conditional_risk are mutually exclusive."
            )
        if not 0.0 < self.adopt_threshold < 1.0:
            raise ValueError("patch_router hierarchical adopt_threshold must be in (0,1).")
        if not 0.0 < self.expert_risk_proposal_threshold < 1.0:
            raise ValueError(
                "patch_router expert_conditional_risk proposal_threshold must be in (0,1)."
            )
        if self.expert_risk_proposal_topk <= 0:
            raise ValueError(
                "patch_router expert_conditional_risk proposal_topk must be positive."
            )
        if self.expert_risk_proposal_rescue_enable and self.expert_risk_proposal_topk != 2:
            raise ValueError(
                "patch_router proposal_rescue currently requires proposal_topk=2."
            )
        if not 0.0 < self.expert_risk_lower_quantile < 0.5:
            raise ValueError(
                "patch_router expert risk lower quantile must be in (0,0.5)."
            )
        if self.expert_risk_adoption_source not in {
            "benefit_probability",
            "expected_utility",
            "lower_quantile",
            "utility_veto",
        }:
            raise ValueError(
                "patch_router expert risk adoption_source must be benefit_probability, "
                "expected_utility, lower_quantile, or utility_veto."
            )
        if (
            self.expert_risk_adoption_source == "lower_quantile"
            and not self.expert_risk_lower_quantile_enable
        ):
            raise ValueError(
                "patch_router lower_quantile adoption_source requires lower_quantile.enable=true."
            )
        if (
            self.expert_risk_adoption_source == "utility_veto"
            and not self.expert_risk_utility_veto_enable
        ):
            raise ValueError(
                "patch_router utility_veto adoption_source requires utility_veto.enable=true."
            )
        if (
            self.expert_conditional_risk_enable
            and self.expert_risk_adoption_source
            in {"benefit_probability", "utility_veto"}
            and not 0.0 < float(self.expert_risk_adopt_threshold.item()) < 1.0
        ):
            raise ValueError(
                "patch_router probability adopt_threshold must be in (0,1)."
            )
        if (
            self.expert_risk_adopt_threshold_by_penalty is not None
            and self.expert_risk_adoption_source
            in {"benefit_probability", "utility_veto"}
            and not bool(
                (
                    (self.expert_risk_adopt_threshold_by_penalty > 0.0)
                    & (self.expert_risk_adopt_threshold_by_penalty < 1.0)
                ).all().item()
            )
        ):
            raise ValueError(
                "patch_router probability adopt_threshold_by_penalty values must be in (0,1)."
            )
        self.num_patches = self.H // self.patch_len
        self.history_patch_projection = "tail" if self.L >= self.H else "cycle"
        # Local patch shape plus level/scale/slope/diff/d2/endpoint context.
        self.feature_dim = self.patch_len + 6
        if self.use_base_forecast:
            self.feature_dim += self.patch_len + 3
        if self.use_full_history_features:
            self.feature_dim += self.L
        if self.use_channel_identity_features:
            self.feature_dim += self.C
        self.feature_dim += 2 * len(self.time_phase_periods)
        self.feature_dim += (self.patch_len + 6) * len(
            self.lagged_delta_periods
        )
        self.regime_feature_dim = 6 * len(self.regime_context_lengths)
        self.feature_dim += self.regime_feature_dim
        self.level_feature_index = self.patch_len
        output_dim = self.P if self.hierarchical_recall_enable else self.P + (1 if self.allow_skip else 0)
        self.W1 = nn.Parameter(torch.empty(self.feature_dim, self.hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(self.hidden_dim))
        self.W2 = nn.Parameter(torch.empty(self.hidden_dim, output_dim))
        self.b2 = nn.Parameter(torch.zeros(output_dim))
        self.W_adopt = (
            nn.Parameter(torch.empty(self.hidden_dim, 1))
            if self.hierarchical_recall_enable
            else None
        )
        self.b_adopt = (
            nn.Parameter(torch.full((1,), self.adopt_init_bias))
            if self.hierarchical_recall_enable
            else None
        )
        self.W_benefit = (
            nn.Parameter(torch.empty(self.hidden_dim, self.P))
            if self.hierarchical_recall_enable
            else None
        )
        self.b_benefit = (
            nn.Parameter(torch.zeros(self.P))
            if self.hierarchical_recall_enable
            else None
        )
        self.W_proposal1 = (
            nn.Parameter(torch.empty(self.feature_dim, self.hidden_dim))
            if self.expert_conditional_risk_enable and self.expert_risk_decoupled_encoder
            else None
        )
        self.b_proposal1 = (
            nn.Parameter(torch.zeros(self.hidden_dim))
            if self.expert_conditional_risk_enable and self.expert_risk_decoupled_encoder
            else None
        )
        self.W_risk_sign = (
            nn.Parameter(torch.empty(self.hidden_dim, self.P))
            if self.expert_conditional_risk_enable
            else None
        )
        self.b_risk_sign = (
            nn.Parameter(torch.zeros(self.P)) if self.expert_conditional_risk_enable else None
        )
        self.W_risk_sign_domain_delta = (
            nn.Parameter(
                torch.zeros(
                    self.expert_risk_temporal_domain_count,
                    self.hidden_dim,
                    self.P,
                )
            )
            if self.expert_risk_temporal_domain_enable
            else None
        )
        self.b_risk_sign_domain_delta = (
            nn.Parameter(
                torch.zeros(self.expert_risk_temporal_domain_count, self.P)
            )
            if self.expert_risk_temporal_domain_enable
            else None
        )
        self.W_risk_gain = (
            nn.Parameter(torch.empty(self.hidden_dim, self.P))
            if self.expert_conditional_risk_enable
            else None
        )
        self.b_risk_gain = (
            nn.Parameter(torch.zeros(self.P)) if self.expert_conditional_risk_enable else None
        )
        self.W_risk_cost = (
            nn.Parameter(torch.empty(self.hidden_dim, self.P))
            if self.expert_conditional_risk_enable
            else None
        )
        self.b_risk_cost = (
            nn.Parameter(torch.zeros(self.P)) if self.expert_conditional_risk_enable else None
        )
        self.W_risk_mse_utility = (
            nn.Parameter(torch.empty(self.hidden_dim, self.P))
            if self.expert_risk_dual_signed_utility_enable
            and not self.expert_risk_analytic_residual_enable
            else None
        )
        self.b_risk_mse_utility = (
            nn.Parameter(torch.zeros(self.P))
            if self.expert_risk_dual_signed_utility_enable
            and not self.expert_risk_analytic_residual_enable
            else None
        )
        self.W_risk_mae_utility = (
            nn.Parameter(torch.empty(self.hidden_dim, self.P))
            if self.expert_risk_dual_signed_utility_enable
            and not self.expert_risk_analytic_residual_enable
            else None
        )
        self.b_risk_mae_utility = (
            nn.Parameter(torch.zeros(self.P))
            if self.expert_risk_dual_signed_utility_enable
            and not self.expert_risk_analytic_residual_enable
            else None
        )
        self.W_predicted_residual = (
            nn.Parameter(torch.empty(self.hidden_dim, self.patch_len))
            if self.expert_risk_analytic_residual_enable
            else None
        )
        self.b_predicted_residual = (
            nn.Parameter(torch.zeros(self.patch_len))
            if self.expert_risk_analytic_residual_enable
            else None
        )
        self.W_periodic_mse_utility = (
            nn.Parameter(torch.empty(self.hidden_dim))
            if self.compositional_periodic_gate_enable
            else None
        )
        self.b_periodic_mse_utility = (
            nn.Parameter(torch.zeros(()))
            if self.compositional_periodic_gate_enable
            else None
        )
        self.W_periodic_mae_utility = (
            nn.Parameter(torch.empty(self.hidden_dim))
            if self.compositional_periodic_gate_enable
            else None
        )
        self.b_periodic_mae_utility = (
            nn.Parameter(torch.zeros(()))
            if self.compositional_periodic_gate_enable
            else None
        )
        self.W_risk_lower_quantile = (
            nn.Parameter(torch.empty(self.hidden_dim, self.P))
            if self.expert_conditional_risk_enable and self.expert_risk_lower_quantile_enable
            else None
        )
        self.b_risk_lower_quantile = (
            nn.Parameter(torch.zeros(self.P))
            if self.expert_conditional_risk_enable and self.expert_risk_lower_quantile_enable
            else None
        )
        self.W_risk_utility_veto = (
            nn.Parameter(torch.empty(self.hidden_dim, self.P))
            if self.expert_conditional_risk_enable
            and self.expert_risk_utility_veto_enable
            else None
        )
        self.b_risk_utility_veto = (
            nn.Parameter(torch.zeros(self.P))
            if self.expert_conditional_risk_enable
            and self.expert_risk_utility_veto_enable
            else None
        )
        self.W_pairwise_rank = (
            nn.Parameter(torch.empty(self.hidden_dim, self.P))
            if self.expert_conditional_risk_enable and self.expert_risk_pairwise_rank_enable
            else None
        )
        self.b_pairwise_rank = (
            nn.Parameter(torch.zeros(self.P))
            if self.expert_conditional_risk_enable and self.expert_risk_pairwise_rank_enable
            else None
        )
        self.candidate_feature_dim = (
            2 * self.patch_len + 19
            if self.expert_risk_candidate_compatibility
            else self.patch_len + 6
        )
        self.W_candidate = (
            nn.Parameter(torch.empty(self.candidate_feature_dim, self.hidden_dim))
            if self.expert_conditional_risk_enable and self.expert_risk_candidate_aware
            else None
        )
        self.b_candidate = (
            nn.Parameter(torch.zeros(self.hidden_dim))
            if self.expert_conditional_risk_enable and self.expert_risk_candidate_aware
            else None
        )
        self.penalty_embedding = (
            nn.Parameter(torch.empty(self.P, self.hidden_dim))
            if self.expert_conditional_risk_enable and self.expert_risk_candidate_aware
            else None
        )
        self.W_proposal_candidate = (
            nn.Parameter(torch.empty(self.candidate_feature_dim, self.hidden_dim))
            if (
                self.expert_conditional_risk_enable
                and self.expert_risk_candidate_aware
                and self.expert_risk_proposal_candidate_aware
            )
            else None
        )
        self.b_proposal_candidate = (
            nn.Parameter(torch.zeros(self.hidden_dim))
            if (
                self.expert_conditional_risk_enable
                and self.expert_risk_candidate_aware
                and self.expert_risk_proposal_candidate_aware
            )
            else None
        )
        self.proposal_penalty_embedding = (
            nn.Parameter(torch.empty(self.P, self.hidden_dim))
            if (
                self.expert_conditional_risk_enable
                and self.expert_risk_candidate_aware
                and self.expert_risk_proposal_candidate_aware
            )
            else None
        )
        self.W_proposal_rescue = (
            nn.Parameter(torch.empty(self.hidden_dim, self.P))
            if self.expert_conditional_risk_enable and self.expert_risk_proposal_rescue_enable
            else None
        )
        self.b_proposal_rescue = (
            nn.Parameter(torch.zeros(self.P))
            if self.expert_conditional_risk_enable and self.expert_risk_proposal_rescue_enable
            else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W1)
        nn.init.normal_(self.W2, mean=0.0, std=0.02)
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)
        if self.hierarchical_recall_enable:
            assert (
                self.W_adopt is not None
                and self.b_adopt is not None
                and self.W_benefit is not None
                and self.b_benefit is not None
            )
            nn.init.normal_(self.W_adopt, mean=0.0, std=0.02)
            nn.init.constant_(self.b_adopt, self.adopt_init_bias)
            nn.init.normal_(self.W_benefit, mean=0.0, std=0.02)
            nn.init.zeros_(self.b_benefit)
            if self.expert_conditional_risk_enable:
                assert (
                    self.W_risk_sign is not None
                    and self.b_risk_sign is not None
                    and self.W_risk_gain is not None
                    and self.b_risk_gain is not None
                    and self.W_risk_cost is not None
                    and self.b_risk_cost is not None
                )
                nn.init.normal_(self.W_risk_sign, mean=0.0, std=0.02)
                nn.init.zeros_(self.b_risk_sign)
                if self.expert_risk_temporal_domain_enable:
                    assert (
                        self.W_risk_sign_domain_delta is not None
                        and self.b_risk_sign_domain_delta is not None
                    )
                    nn.init.zeros_(self.W_risk_sign_domain_delta)
                    nn.init.zeros_(self.b_risk_sign_domain_delta)
                nn.init.normal_(self.W_risk_gain, mean=0.0, std=0.02)
                nn.init.constant_(self.b_risk_gain, -2.0)
                nn.init.normal_(self.W_risk_cost, mean=0.0, std=0.02)
                nn.init.constant_(self.b_risk_cost, -2.0)
                if self.expert_risk_dual_signed_utility_enable:
                    if self.expert_risk_analytic_residual_enable:
                        assert (
                            self.W_predicted_residual is not None
                            and self.b_predicted_residual is not None
                        )
                        # Zero predicted error makes every nonzero correction
                        # strictly worse, so the fresh structured gate skips.
                        nn.init.zeros_(self.W_predicted_residual)
                        nn.init.zeros_(self.b_predicted_residual)
                    else:
                        assert (
                            self.W_risk_mse_utility is not None
                            and self.b_risk_mse_utility is not None
                            and self.W_risk_mae_utility is not None
                            and self.b_risk_mae_utility is not None
                        )
                        # Zero is the exact utility of skip.  A fresh dual-utility
                        # gate must therefore be a bit-exact no-op before learning.
                        nn.init.zeros_(self.W_risk_mse_utility)
                        nn.init.zeros_(self.b_risk_mse_utility)
                        nn.init.zeros_(self.W_risk_mae_utility)
                        nn.init.zeros_(self.b_risk_mae_utility)
                    if self.compositional_periodic_gate_enable:
                        assert (
                            self.W_periodic_mse_utility is not None
                            and self.b_periodic_mse_utility is not None
                            and self.W_periodic_mae_utility is not None
                            and self.b_periodic_mae_utility is not None
                        )
                        # The delivered router always includes the periodic
                        # expert.  Zero initialization plus the explicit tie
                        # rule in forward preserves that exact epoch-0 action.
                        nn.init.zeros_(self.W_periodic_mse_utility)
                        nn.init.zeros_(self.b_periodic_mse_utility)
                        nn.init.zeros_(self.W_periodic_mae_utility)
                        nn.init.zeros_(self.b_periodic_mae_utility)
                if self.expert_risk_lower_quantile_enable:
                    assert (
                        self.W_risk_lower_quantile is not None
                        and self.b_risk_lower_quantile is not None
                    )
                    nn.init.normal_(self.W_risk_lower_quantile, mean=0.0, std=0.02)
                    nn.init.zeros_(self.b_risk_lower_quantile)
                if self.expert_risk_utility_veto_enable:
                    assert (
                        self.W_risk_utility_veto is not None
                        and self.b_risk_utility_veto is not None
                    )
                    nn.init.normal_(self.W_risk_utility_veto, mean=0.0, std=0.02)
                    nn.init.zeros_(self.b_risk_utility_veto)
                if self.expert_risk_pairwise_rank_enable:
                    assert self.W_pairwise_rank is not None and self.b_pairwise_rank is not None
                    nn.init.normal_(self.W_pairwise_rank, mean=0.0, std=0.02)
                    nn.init.zeros_(self.b_pairwise_rank)
                if self.expert_risk_candidate_aware:
                    assert (
                        self.W_candidate is not None
                        and self.b_candidate is not None
                        and self.penalty_embedding is not None
                    )
                    nn.init.xavier_uniform_(self.W_candidate)
                    nn.init.zeros_(self.b_candidate)
                    nn.init.normal_(self.penalty_embedding, mean=0.0, std=0.02)
                    if self.expert_risk_proposal_candidate_aware:
                        assert (
                            self.W_proposal_candidate is not None
                            and self.b_proposal_candidate is not None
                            and self.proposal_penalty_embedding is not None
                        )
                        nn.init.xavier_uniform_(self.W_proposal_candidate)
                        nn.init.zeros_(self.b_proposal_candidate)
                        nn.init.normal_(
                            self.proposal_penalty_embedding,
                            mean=0.0,
                            std=0.02,
                        )
                if self.expert_risk_proposal_rescue_enable:
                    assert (
                        self.W_proposal_rescue is not None
                        and self.b_proposal_rescue is not None
                    )
                    nn.init.normal_(self.W_proposal_rescue, mean=0.0, std=0.02)
                    nn.init.zeros_(self.b_proposal_rescue)
                if self.expert_risk_decoupled_encoder:
                    assert self.W_proposal1 is not None and self.b_proposal1 is not None
                    nn.init.xavier_uniform_(self.W_proposal1)
                    nn.init.zeros_(self.b_proposal1)
        elif self.allow_skip:
            with torch.no_grad():
                self.b2[0] = self.skip_init_bias

    def _regime_features(
        self,
        x_bcl: torch.Tensor,
        regime_context_bcl: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not self.regime_context_enable:
            return None
        if regime_context_bcl is None or regime_context_bcl.ndim != 3:
            raise ValueError(
                "patch_router regime_context requires causal history with shape [B,C,S]."
            )
        if tuple(regime_context_bcl.shape[:2]) != tuple(x_bcl.shape[:2]):
            raise ValueError("patch_router regime context batch/channel shape does not match input.")
        required = max(self.regime_context_lengths)
        if int(regime_context_bcl.shape[-1]) < required:
            raise ValueError(
                "patch_router regime context is shorter than the configured maximum: "
                f"got {int(regime_context_bcl.shape[-1])}, need {required}."
            )

        eps = 1.0e-6
        recent_mean = x_bcl.mean(dim=-1, keepdim=True)
        recent_std = x_bcl.std(dim=-1, unbiased=False, keepdim=True).clamp_min(eps)
        recent_range = (
            x_bcl.amax(dim=-1, keepdim=True) - x_bcl.amin(dim=-1, keepdim=True)
        ).clamp_min(eps)
        recent_d1 = x_bcl.diff(dim=-1)
        recent_mad1 = recent_d1.abs().mean(dim=-1, keepdim=True).clamp_min(eps)
        recent_mad2 = (
            recent_d1.diff(dim=-1).abs().mean(dim=-1, keepdim=True).clamp_min(eps)
            if int(x_bcl.shape[-1]) >= 3
            else torch.full_like(recent_mean, eps)
        )
        recent_endpoint = x_bcl[..., -1:] - x_bcl[..., :1]

        parts = []
        for context_len in self.regime_context_lengths:
            context = regime_context_bcl[..., -int(context_len) :]
            context_mean = context.mean(dim=-1, keepdim=True)
            context_std = context.std(dim=-1, unbiased=False, keepdim=True).clamp_min(eps)
            context_range = (
                context.amax(dim=-1, keepdim=True)
                - context.amin(dim=-1, keepdim=True)
            ).clamp_min(eps)
            context_d1 = context.diff(dim=-1)
            context_mad1 = context_d1.abs().mean(dim=-1, keepdim=True).clamp_min(eps)
            context_mad2 = (
                context_d1.diff(dim=-1).abs().mean(dim=-1, keepdim=True).clamp_min(eps)
                if int(context.shape[-1]) >= 3
                else torch.full_like(context_mean, eps)
            )
            scale_features = torch.cat(
                [
                    (recent_mean - context_mean) / context_std,
                    torch.log(recent_std / context_std),
                    torch.log(recent_range / context_range),
                    torch.log(recent_mad1 / context_mad1),
                    torch.log(recent_mad2 / context_mad2),
                    recent_endpoint / context_std,
                ],
                dim=-1,
            ).unsqueeze(2).expand(-1, -1, self.num_patches, -1)
            parts.append(scale_features)
        return torch.cat(parts, dim=-1)

    def _lagged_delta_features(
        self,
        x_bcl: torch.Tensor,
        regime_context_bcl: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Pointwise causal changes from prior same-phase input windows."""
        if not self.lagged_delta_periods:
            return None
        if regime_context_bcl is None or regime_context_bcl.ndim != 3:
            raise ValueError(
                "patch_router lagged delta features require causal regime history."
            )
        if tuple(regime_context_bcl.shape[:2]) != tuple(x_bcl.shape[:2]):
            raise ValueError(
                "patch_router lagged delta history batch/channel shape does not match input."
            )
        required = self.L + max(self.lagged_delta_periods)
        if int(regime_context_bcl.shape[-1]) < required:
            raise ValueError(
                "patch_router lagged delta history is too short: "
                f"got {int(regime_context_bcl.shape[-1])}, need {required}."
            )

        eps = 1.0e-6
        full_std = x_bcl.std(dim=-1, unbiased=False, keepdim=True).clamp_min(eps)
        time = torch.linspace(
            -1.0,
            1.0,
            steps=self.patch_len,
            device=x_bcl.device,
            dtype=x_bcl.dtype,
        ).view(1, 1, 1, self.patch_len)
        time_energy = time.square().mean().clamp_min(eps)
        parts = []
        for period in self.lagged_delta_periods:
            previous = regime_context_bcl[
                ...,
                -(self.L + int(period)) : -int(period),
            ]
            if int(previous.shape[-1]) != self.L:
                raise ValueError(
                    "patch_router lagged delta slice does not match input_len."
                )
            delta = (x_bcl - previous) / full_std
            delta_patch = self._history_patches(delta)
            mean = delta_patch.mean(dim=-1, keepdim=True)
            std = delta_patch.std(dim=-1, unbiased=False, keepdim=True)
            slope = ((delta_patch - mean) * time).mean(dim=-1, keepdim=True)
            slope = slope / time_energy
            endpoint = delta_patch[..., -1:] - delta_patch[..., :1]
            if self.patch_len >= 2:
                diff = delta_patch.diff(dim=-1)
                mad1 = diff.abs().mean(dim=-1, keepdim=True)
            else:
                diff = None
                mad1 = torch.zeros_like(mean)
            if diff is not None and self.patch_len >= 3:
                mad2 = diff.diff(dim=-1).abs().mean(dim=-1, keepdim=True)
            else:
                mad2 = torch.zeros_like(mean)
            parts.append(
                torch.cat(
                    [delta_patch, mean, std, slope, mad1, mad2, endpoint],
                    dim=-1,
                )
            )
        return torch.cat(parts, dim=-1)

    def _history_patches(self, x_bcl: torch.Tensor) -> torch.Tensor:
        """Align causal input patches to forecast patches without future labels."""
        if x_bcl.ndim != 3:
            raise ValueError("patch_router input must have shape [B,C,L].")
        batch, channels, observed_len = map(int, x_bcl.shape)
        if observed_len >= self.H:
            return x_bcl[..., -self.H :].reshape(
                batch,
                channels,
                self.num_patches,
                self.patch_len,
            )
        if self.short_history_mode != "cycle":
            raise ValueError(
                "patch_router input history is shorter than pred_len and "
                "short_history_mode is not cycle."
            )
        input_patch_count = observed_len // self.patch_len
        if input_patch_count <= 0:
            raise ValueError(
                "patch_router cycle mode requires at least one complete input patch."
            )
        usable_len = input_patch_count * self.patch_len
        input_patches = x_bcl[..., -usable_len:].reshape(
            batch,
            channels,
            input_patch_count,
            self.patch_len,
        )
        forecast_patch_index = torch.arange(
            self.num_patches,
            device=x_bcl.device,
        ).remainder(input_patch_count)
        return input_patches.index_select(2, forecast_patch_index)

    def _features(
        self,
        x_bcl: torch.Tensor,
        y_base_bch: Optional[torch.Tensor] = None,
        regime_context_bcl: Optional[torch.Tensor] = None,
        query_start_abs_b: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x_bcl.ndim != 3:
            raise ValueError("patch_router input must have shape [B,C,L].")
        eps = 1.0e-6
        batch, channels = int(x_bcl.shape[0]), int(x_bcl.shape[1])
        if self.use_channel_identity_features and channels != self.C:
            raise ValueError(
                "patch_router input channel count does not match channel identity size."
            )
        x_tail = self._history_patches(x_bcl)
        full_mean = x_bcl.mean(dim=-1, keepdim=True)
        full_std = x_bcl.std(dim=-1, unbiased=False, keepdim=True).clamp_min(eps)
        patch_mean = x_tail.mean(dim=-1, keepdim=True)
        patch_std = x_tail.std(dim=-1, unbiased=False, keepdim=True).clamp_min(eps)
        local_shape = (x_tail - patch_mean) / patch_std

        time = torch.linspace(
            -1.0,
            1.0,
            steps=self.patch_len,
            device=x_bcl.device,
            dtype=x_bcl.dtype,
        ).view(1, 1, 1, self.patch_len)
        time_energy = time.square().mean().clamp_min(eps)
        level = (patch_mean - full_mean.unsqueeze(2)) / full_std.unsqueeze(2)
        scale = patch_std / full_std.unsqueeze(2)
        slope = ((x_tail - patch_mean) * time).mean(dim=-1, keepdim=True)
        slope = slope / (time_energy * full_std.unsqueeze(2))
        endpoint = (x_tail[..., -1:] - x_tail[..., :1]) / full_std.unsqueeze(2)
        if self.patch_len >= 2:
            diff = x_tail.diff(dim=-1)
            mad1 = diff.abs().mean(dim=-1, keepdim=True) / full_std.unsqueeze(2)
        else:
            diff = None
            mad1 = torch.zeros_like(level)
        if diff is not None and self.patch_len >= 3:
            mad2 = diff.diff(dim=-1).abs().mean(dim=-1, keepdim=True) / full_std.unsqueeze(2)
        else:
            mad2 = torch.zeros_like(level)
        parts = [local_shape, level, scale, slope, mad1, mad2, endpoint]
        if self.use_full_history_features:
            full_shape = (x_bcl - full_mean) / full_std
            parts.append(
                full_shape.unsqueeze(2).expand(-1, -1, self.num_patches, -1)
            )
        if self.use_channel_identity_features:
            channel_id = torch.eye(
                self.C,
                device=x_bcl.device,
                dtype=x_bcl.dtype,
            ).view(1, self.C, 1, self.C)
            parts.append(
                channel_id.expand(batch, -1, self.num_patches, -1)
            )
        if self.time_phase_periods:
            if query_start_abs_b is None:
                raise ValueError(
                    "patch_router time phase features require query_start_abs_b."
                )
            query_start = query_start_abs_b.reshape(-1).to(
                device=x_bcl.device,
                dtype=x_bcl.dtype,
            )
            if int(query_start.numel()) != batch:
                raise ValueError(
                    "patch_router query_start_abs_b must match batch size."
                )
            patch_center = (
                query_start[:, None]
                + float(self.L)
                + torch.arange(
                    self.num_patches,
                    device=x_bcl.device,
                    dtype=x_bcl.dtype,
                )[None, :]
                * float(self.patch_len)
                + 0.5 * float(self.patch_len - 1)
            )
            phase_parts = []
            for period in self.time_phase_periods:
                angle = (
                    2.0
                    * torch.pi
                    * torch.remainder(patch_center, float(period))
                    / float(period)
                )
                phase_parts.extend([torch.sin(angle), torch.cos(angle)])
            phase_features = torch.stack(phase_parts, dim=-1).unsqueeze(1)
            parts.append(phase_features.expand(-1, channels, -1, -1))
        lagged_delta_features = self._lagged_delta_features(
            x_bcl,
            regime_context_bcl,
        )
        if lagged_delta_features is not None:
            parts.append(lagged_delta_features)
        if self.use_base_forecast:
            if y_base_bch is None or tuple(y_base_bch.shape) != (batch, channels, self.H):
                raise ValueError(
                    "patch_router.use_base_forecast requires y_base with shape [B,C,H]."
                )
            base_patch = y_base_bch.reshape(batch, channels, self.num_patches, self.patch_len)
            base_local = (base_patch - patch_mean) / patch_std
            base_mean_shift = (base_patch.mean(dim=-1, keepdim=True) - patch_mean) / full_std.unsqueeze(2)
            base_scale = base_patch.std(dim=-1, unbiased=False, keepdim=True) / full_std.unsqueeze(2)
            base_endpoint = (base_patch[..., -1:] - base_patch[..., :1]) / full_std.unsqueeze(2)
            parts.extend([base_local, base_mean_shift, base_scale, base_endpoint])
        regime_features = self._regime_features(x_bcl, regime_context_bcl)
        if regime_features is not None:
            parts.append(regime_features)
        features = torch.cat(parts, dim=-1)
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        if self.feature_clip > 0.0:
            features = features.clamp(-self.feature_clip, self.feature_clip)
        return features

    def _candidate_features(
        self,
        x_bcl: torch.Tensor,
        y_base_bch: torch.Tensor,
        candidate_delta_bcpH: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels = int(x_bcl.shape[0]), int(x_bcl.shape[1])
        if tuple(y_base_bch.shape) != (batch, channels, self.H):
            raise ValueError("patch_router y_base must have shape [B,C,H].")
        if tuple(candidate_delta_bcpH.shape) != (batch, channels, self.P, self.H):
            raise ValueError(
                "patch_router candidate_delta must have shape [B,C,P,H]."
            )
        eps = 1.0e-6
        full_std = x_bcl.std(dim=-1, unbiased=False).clamp_min(eps)
        delta = candidate_delta_bcpH.reshape(
            batch,
            channels,
            self.P,
            self.num_patches,
            self.patch_len,
        ).permute(0, 1, 3, 2, 4)
        scale = full_std[:, :, None, None, None]
        delta_scaled = delta / scale
        mean = delta_scaled.mean(dim=-1, keepdim=True)
        std = delta_scaled.std(dim=-1, unbiased=False, keepdim=True)
        time = torch.linspace(
            -1.0,
            1.0,
            steps=self.patch_len,
            device=x_bcl.device,
            dtype=x_bcl.dtype,
        ).view(1, 1, 1, 1, self.patch_len)
        slope = ((delta_scaled - mean) * time).mean(dim=-1, keepdim=True)
        slope = slope / time.square().mean().clamp_min(eps)
        endpoint = delta_scaled[..., -1:] - delta_scaled[..., :1]
        if self.patch_len >= 2:
            diff = delta_scaled.diff(dim=-1)
            mad1 = diff.abs().mean(dim=-1, keepdim=True)
        else:
            diff = None
            mad1 = torch.zeros_like(mean)
        if diff is not None and self.patch_len >= 3:
            mad2 = diff.diff(dim=-1).abs().mean(dim=-1, keepdim=True)
        else:
            mad2 = torch.zeros_like(mean)
        parts = [delta_scaled, mean, std, slope, mad1, mad2, endpoint]
        if self.expert_risk_candidate_compatibility:
            history = self._history_patches(x_bcl)
            candidate_full = y_base_bch.unsqueeze(2) + candidate_delta_bcpH
            candidate = candidate_full.reshape(
                batch,
                channels,
                self.P,
                self.num_patches,
                self.patch_len,
            ).permute(0, 1, 3, 2, 4)
            history_scaled = history.unsqueeze(3) / scale
            candidate_scaled = candidate / scale
            compatibility = candidate_scaled - history_scaled
            history_mean = history_scaled.mean(dim=-1, keepdim=True)
            candidate_mean = candidate_scaled.mean(dim=-1, keepdim=True)
            history_std = history_scaled.std(dim=-1, unbiased=False, keepdim=True)
            candidate_std = candidate_scaled.std(dim=-1, unbiased=False, keepdim=True)
            history_slope = (
                (history_scaled - history_mean) * time
            ).mean(dim=-1, keepdim=True) / time.square().mean().clamp_min(eps)
            candidate_slope = (
                (candidate_scaled - candidate_mean) * time
            ).mean(dim=-1, keepdim=True) / time.square().mean().clamp_min(eps)
            if self.patch_len >= 2:
                history_diff = history_scaled.diff(dim=-1)
                candidate_diff = candidate_scaled.diff(dim=-1)
                history_mad1 = history_diff.abs().mean(dim=-1, keepdim=True)
                candidate_mad1 = candidate_diff.abs().mean(dim=-1, keepdim=True)
            else:
                history_diff = None
                candidate_diff = None
                history_mad1 = torch.zeros_like(history_mean)
                candidate_mad1 = torch.zeros_like(candidate_mean)
            if history_diff is not None and candidate_diff is not None and self.patch_len >= 3:
                history_mad2 = history_diff.diff(dim=-1).abs().mean(dim=-1, keepdim=True)
                candidate_mad2 = candidate_diff.diff(dim=-1).abs().mean(dim=-1, keepdim=True)
            else:
                history_mad2 = torch.zeros_like(history_mean)
                candidate_mad2 = torch.zeros_like(candidate_mean)
            candidate_bcpqr = candidate_full.reshape(
                batch,
                channels,
                self.P,
                self.num_patches,
                self.patch_len,
            )
            previous_end = torch.cat(
                [
                    x_bcl[..., -1:].unsqueeze(2).expand(-1, -1, self.P, -1),
                    candidate_bcpqr[..., :-1, -1],
                ],
                dim=-1,
            ).permute(0, 1, 3, 2).unsqueeze(-1)
            boundary_jump = (
                candidate_bcpqr[..., 0].permute(0, 1, 3, 2).unsqueeze(-1)
                - previous_end
            ) / scale
            compatibility_scalars = [
                compatibility.square().mean(dim=-1, keepdim=True).sqrt(),
                candidate_mean - history_mean,
                candidate_std - history_std,
                candidate_slope - history_slope,
                candidate_mad1 - history_mad1,
                candidate_mad2 - history_mad2,
                boundary_jump,
            ]

            full_scale = full_std[:, :, None, None]
            full_delta = candidate_delta_bcpH / full_scale
            full_mean = full_delta.mean(dim=-1, keepdim=True)
            full_std_delta = full_delta.std(dim=-1, unbiased=False, keepdim=True)
            full_time = torch.linspace(
                -1.0,
                1.0,
                steps=self.H,
                device=x_bcl.device,
                dtype=x_bcl.dtype,
            ).view(1, 1, 1, self.H)
            full_slope = (
                (full_delta - full_mean) * full_time
            ).mean(dim=-1, keepdim=True) / full_time.square().mean().clamp_min(eps)
            if self.H >= 2:
                full_diff = full_delta.diff(dim=-1)
                full_mad1 = full_diff.abs().mean(dim=-1, keepdim=True)
            else:
                full_diff = None
                full_mad1 = torch.zeros_like(full_mean)
            if full_diff is not None and self.H >= 3:
                full_mad2 = full_diff.diff(dim=-1).abs().mean(dim=-1, keepdim=True)
            else:
                full_mad2 = torch.zeros_like(full_mean)
            full_range = full_delta.amax(dim=-1, keepdim=True) - full_delta.amin(
                dim=-1,
                keepdim=True,
            )
            full_summary = torch.cat(
                [
                    full_mean,
                    full_std_delta,
                    full_slope,
                    full_mad1,
                    full_mad2,
                    full_range,
                ],
                dim=-1,
            ).unsqueeze(2).expand(-1, -1, self.num_patches, -1, -1)
            parts.extend([compatibility, *compatibility_scalars, full_summary])
        features = torch.cat(parts, dim=-1)
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        if self.feature_clip > 0.0:
            features = features.clamp(-self.feature_clip, self.feature_clip)
        return features

    def forward(
        self,
        x_bcl: torch.Tensor,
        *,
        y_base_bch: Optional[torch.Tensor] = None,
        candidate_delta_bcpH: Optional[torch.Tensor] = None,
        regime_context_bcl: Optional[torch.Tensor] = None,
        query_start_abs_b: Optional[torch.Tensor] = None,
        straight_through: bool,
    ) -> Dict[str, torch.Tensor]:
        features = self._features(
            x_bcl,
            y_base_bch,
            regime_context_bcl=regime_context_bcl,
            query_start_abs_b=query_start_abs_b,
        )
        hidden = F.gelu(torch.einsum("bcqd,dm->bcqm", features, self.W1) + self.b1)
        logits = torch.einsum("bcqm,mp->bcqp", hidden, self.W2) + self.b2
        selected_risk_score = None
        selected_risk_benefit_prob = None
        risk_domain_std = None
        mse_utility_scores = None
        mae_utility_scores = None
        predicted_residual = None
        periodic_mse_utility_scores = None
        periodic_mae_utility_scores = None
        action_mse_utility_scores = None
        action_mae_utility_scores = None
        action_scores = None
        action_probs = None
        action_index = None
        periodic_route = None
        periodic_only = None
        backbone_route = None
        fixed_penalty_active_bcq = None
        if self.compositional_periodic_gate_enable:
            assert (
                self.W_periodic_mse_utility is not None
                and self.b_periodic_mse_utility is not None
                and self.W_periodic_mae_utility is not None
                and self.b_periodic_mae_utility is not None
            )
            periodic_mse_utility_scores = torch.tanh(
                torch.einsum(
                    "bcqm,m->bcq",
                    hidden,
                    self.W_periodic_mse_utility,
                )
                + self.b_periodic_mse_utility
            )
            periodic_mae_utility_scores = torch.tanh(
                torch.einsum(
                    "bcqm,m->bcq",
                    hidden,
                    self.W_periodic_mae_utility,
                )
                + self.b_periodic_mae_utility
            )
        if self.training and self.noise_std > 0.0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        if self.hierarchical_recall_enable:
            assert (
                self.W_adopt is not None
                and self.b_adopt is not None
                and self.W_benefit is not None
                and self.b_benefit is not None
            )
            proposal_hidden = hidden
            if self.W_proposal1 is not None and self.b_proposal1 is not None:
                proposal_hidden = F.gelu(
                    torch.einsum("bcqd,dm->bcqm", features, self.W_proposal1)
                    + self.b_proposal1
                )
            candidate_features = None
            if self.expert_conditional_risk_enable and self.expert_risk_candidate_aware:
                if candidate_delta_bcpH is None:
                    raise ValueError(
                        "candidate-aware patch risk gate requires candidate_delta_bcpH."
                    )
                candidate_features = self._candidate_features(
                    x_bcl,
                    y_base_bch,
                    candidate_delta_bcpH.detach(),
                )
            adopt_logits = (
                torch.einsum("bcqm,mr->bcqr", proposal_hidden, self.W_adopt).squeeze(-1)
                + self.b_adopt
            )
            proposal_expert_hidden = proposal_hidden.unsqueeze(3).expand(
                -1,
                -1,
                -1,
                self.P,
                -1,
            )
            if self.expert_risk_proposal_candidate_aware:
                if candidate_features is None:
                    raise ValueError(
                        "candidate-aware proposal gate requires candidate features."
                    )
                assert (
                    self.W_proposal_candidate is not None
                    and self.b_proposal_candidate is not None
                    and self.proposal_penalty_embedding is not None
                )
                proposal_candidate_hidden = (
                    torch.einsum(
                        "bcqpd,dm->bcqpm",
                        candidate_features,
                        self.W_proposal_candidate,
                    )
                    + self.b_proposal_candidate
                    + self.proposal_penalty_embedding.view(
                        1,
                        1,
                        1,
                        self.P,
                        self.hidden_dim,
                    )
                )
                proposal_expert_hidden = F.gelu(
                    proposal_hidden.unsqueeze(3) + proposal_candidate_hidden
                )
                benefit_logits = (
                    torch.einsum(
                        "bcqpm,mp->bcqp",
                        proposal_expert_hidden,
                        self.W_benefit,
                    )
                    + self.b_benefit
                )
            else:
                benefit_logits = (
                    torch.einsum("bcqm,mp->bcqp", proposal_hidden, self.W_benefit)
                    + self.b_benefit
                )
            if self.expert_risk_proposal_rescue_enable:
                assert (
                    self.W_proposal_rescue is not None
                    and self.b_proposal_rescue is not None
                )
                rescue_logits = (
                    torch.einsum(
                        "bcqpm,mp->bcqp",
                        proposal_expert_hidden,
                        self.W_proposal_rescue,
                    )
                    + self.b_proposal_rescue
                )
            else:
                rescue_logits = benefit_logits
            if self.training and self.noise_std > 0.0:
                adopt_logits = adopt_logits + torch.randn_like(adopt_logits) * self.noise_std
                benefit_logits = benefit_logits + torch.randn_like(benefit_logits) * self.noise_std
                if self.expert_risk_proposal_rescue_enable:
                    rescue_logits = rescue_logits + torch.randn_like(rescue_logits) * self.noise_std
            proposal_adopt_prob = torch.sigmoid(adopt_logits)
            penalty_benefit_probs = torch.sigmoid(benefit_logits / self.temperature)
            utility_scores = torch.tanh(logits / self.temperature)
            risk_benefit_prob = penalty_benefit_probs
            risk_positive_magnitude = utility_scores.clamp_min(0.0)
            risk_negative_magnitude = (-utility_scores).clamp_min(0.0)
            risk_lower_quantile_scores = utility_scores
            risk_utility_veto_prob = risk_benefit_prob
            pairwise_rank_scores = utility_scores
            proposal_mask = penalty_benefit_probs > self.expert_risk_proposal_threshold
            if self.expert_conditional_risk_enable:
                assert (
                    self.W_risk_sign is not None
                    and self.b_risk_sign is not None
                    and self.W_risk_gain is not None
                    and self.b_risk_gain is not None
                    and self.W_risk_cost is not None
                    and self.b_risk_cost is not None
                )
                risk_hidden = hidden.unsqueeze(3).expand(-1, -1, -1, self.P, -1)
                if self.expert_risk_candidate_aware:
                    assert candidate_features is not None
                    assert (
                        self.W_candidate is not None
                        and self.b_candidate is not None
                        and self.penalty_embedding is not None
                    )
                    candidate_hidden = (
                        torch.einsum("bcqpd,dm->bcqpm", candidate_features, self.W_candidate)
                        + self.b_candidate
                        + self.penalty_embedding.view(1, 1, 1, self.P, self.hidden_dim)
                    )
                    risk_hidden = F.gelu(risk_hidden + candidate_hidden)
                risk_sign_logits = (
                    torch.einsum("bcqpm,mp->bcqp", risk_hidden, self.W_risk_sign)
                    + self.b_risk_sign
                )
                if self.expert_risk_temporal_domain_enable:
                    assert (
                        self.W_risk_sign_domain_delta is not None
                        and self.b_risk_sign_domain_delta is not None
                    )
                    domain_sign_logits = (
                        risk_sign_logits.unsqueeze(0)
                        + torch.einsum(
                            "bcqpm,dmp->dbcqp",
                            risk_hidden,
                            self.W_risk_sign_domain_delta,
                        )
                        + self.b_risk_sign_domain_delta[:, None, None, None, :]
                    )
                    domain_risk_prob = torch.sigmoid(domain_sign_logits)
                    risk_domain_std = domain_risk_prob.std(
                        dim=0,
                        unbiased=False,
                    )
                    if self.training:
                        if query_start_abs_b is None:
                            raise ValueError(
                                "temporal domain risk training requires query indices."
                            )
                        query_index_b = query_start_abs_b.reshape(-1).to(
                            device=x_bcl.device,
                            dtype=torch.long,
                        )
                        if int(query_index_b.numel()) != int(x_bcl.shape[0]):
                            raise ValueError(
                                "temporal domain risk query indices must match batch size."
                            )
                        domain_id_b = torch.div(
                            query_index_b
                            * int(self.expert_risk_temporal_domain_count),
                            int(self.expert_risk_temporal_domain_train_windows),
                            rounding_mode="floor",
                        ).clamp(
                            0,
                            int(self.expert_risk_temporal_domain_count) - 1,
                        )
                        domain_risk_prob_bdcqp = domain_risk_prob.permute(
                            1,
                            0,
                            2,
                            3,
                            4,
                        )
                        risk_benefit_prob = domain_risk_prob_bdcqp[
                            torch.arange(
                                int(x_bcl.shape[0]),
                                device=x_bcl.device,
                            ),
                            domain_id_b,
                        ]
                    else:
                        risk_benefit_prob = domain_risk_prob.mean(dim=0)
                else:
                    risk_benefit_prob = torch.sigmoid(risk_sign_logits)
                risk_gain_logits = (
                    torch.einsum("bcqpm,mp->bcqp", risk_hidden, self.W_risk_gain)
                    + self.b_risk_gain
                )
                risk_cost_logits = (
                    torch.einsum("bcqpm,mp->bcqp", risk_hidden, self.W_risk_cost)
                    + self.b_risk_cost
                )
                risk_positive_magnitude = torch.sigmoid(risk_gain_logits)
                risk_negative_magnitude = torch.sigmoid(risk_cost_logits)
                utility_scores = (
                    risk_benefit_prob * risk_positive_magnitude
                    - (1.0 - risk_benefit_prob) * risk_negative_magnitude
                )
                if self.expert_risk_dual_signed_utility_enable:
                    if self.expert_risk_analytic_residual_enable:
                        assert (
                            self.W_predicted_residual is not None
                            and self.b_predicted_residual is not None
                            and candidate_delta_bcpH is not None
                        )
                        predicted_residual = (
                            torch.einsum(
                                "bcqm,mh->bcqh",
                                hidden,
                                self.W_predicted_residual,
                            )
                            + self.b_predicted_residual
                        )
                        input_scale = x_bcl.std(
                            dim=-1,
                            unbiased=False,
                        ).clamp_min(1.0e-6)
                        candidate_delta = candidate_delta_bcpH.reshape(
                            int(x_bcl.shape[0]),
                            int(x_bcl.shape[1]),
                            self.P,
                            self.num_patches,
                            self.patch_len,
                        ).permute(0, 1, 3, 2, 4)
                        candidate_delta = (
                            candidate_delta
                            / input_scale[:, :, None, None, None]
                        )
                        residual = predicted_residual.unsqueeze(3)
                        raw_mse_gain = (
                            2.0 * residual * candidate_delta
                            - candidate_delta.square()
                        ).mean(dim=-1)
                        mse_denom = (
                            predicted_residual.square().mean(
                                dim=-1,
                                keepdim=True,
                            )
                            + self.expert_risk_analytic_residual_floor
                        )
                        mse_utility_scores = torch.tanh(
                            raw_mse_gain / mse_denom
                        )
                        raw_mae_gain = (
                            residual.abs()
                            - (residual - candidate_delta).abs()
                        ).mean(dim=-1)
                        mae_denom = (
                            predicted_residual.abs().mean(
                                dim=-1,
                                keepdim=True,
                            )
                            + self.expert_risk_analytic_residual_floor
                        )
                        mae_utility_scores = torch.tanh(
                            raw_mae_gain / mae_denom
                        )
                    else:
                        assert (
                            self.W_risk_mse_utility is not None
                            and self.b_risk_mse_utility is not None
                            and self.W_risk_mae_utility is not None
                            and self.b_risk_mae_utility is not None
                        )
                        mse_utility_scores = torch.tanh(
                            torch.einsum(
                                "bcqpm,mp->bcqp",
                                risk_hidden,
                                self.W_risk_mse_utility,
                            )
                            + self.b_risk_mse_utility
                        )
                        mae_utility_scores = torch.tanh(
                            torch.einsum(
                                "bcqpm,mp->bcqp",
                                risk_hidden,
                                self.W_risk_mae_utility,
                            )
                            + self.b_risk_mae_utility
                        )
                    # Rank by the conservative dual-metric utility.  Because
                    # skip has fixed utility zero, the existing expected-
                    # utility adoption path rejects every non-positive score.
                    utility_scores = torch.minimum(
                        mse_utility_scores,
                        mae_utility_scores,
                    )
                    risk_benefit_prob = torch.sigmoid(
                        utility_scores / self.utility_verifier_temperature
                    )
                    risk_positive_magnitude = utility_scores.clamp_min(0.0)
                    risk_negative_magnitude = (-utility_scores).clamp_min(0.0)
                if self.expert_risk_utility_veto_enable:
                    assert (
                        self.W_risk_utility_veto is not None
                        and self.b_risk_utility_veto is not None
                    )
                    veto_hidden = (
                        risk_hidden.detach()
                        if self.expert_risk_utility_veto_detach_features
                        else risk_hidden
                    )
                    risk_utility_veto_prob = torch.sigmoid(
                        torch.einsum(
                            "bcqpm,mp->bcqp",
                            veto_hidden,
                            self.W_risk_utility_veto,
                        )
                        + self.b_risk_utility_veto
                    )
                else:
                    risk_utility_veto_prob = risk_benefit_prob
                if self.expert_risk_lower_quantile_enable:
                    assert (
                        self.W_risk_lower_quantile is not None
                        and self.b_risk_lower_quantile is not None
                    )
                    risk_lower_quantile_scores = torch.tanh(
                        torch.einsum(
                            "bcqpm,mp->bcqp",
                            risk_hidden,
                            self.W_risk_lower_quantile,
                        )
                        + self.b_risk_lower_quantile
                    )
                else:
                    risk_lower_quantile_scores = utility_scores
                if self.expert_risk_pairwise_rank_enable:
                    assert self.W_pairwise_rank is not None and self.b_pairwise_rank is not None
                    pairwise_hidden = (
                        risk_hidden.detach()
                        if self.expert_risk_pairwise_detach_features
                        else risk_hidden
                    )
                    pairwise_rank_scores = (
                        torch.einsum(
                            "bcqpm,mp->bcqp",
                            pairwise_hidden,
                            self.W_pairwise_rank,
                        )
                        + self.b_pairwise_rank
                    )
                else:
                    pairwise_rank_scores = utility_scores
                proposal_mask = torch.zeros_like(penalty_benefit_probs, dtype=torch.bool)
                if self.expert_risk_dual_signed_utility_enable:
                    # The utility heads score every candidate directly; an
                    # untrained proposal head must not silently remove one.
                    proposal_mask.fill_(True)
                elif self.expert_risk_proposal_rescue_enable:
                    primary_idx = benefit_logits.argmax(dim=-1, keepdim=True)
                    proposal_mask.scatter_(-1, primary_idx, True)
                    rescue_rank_logits = rescue_logits.masked_fill(proposal_mask, -1.0e4)
                    rescue_idx = rescue_rank_logits.argmax(dim=-1, keepdim=True)
                    proposal_mask.scatter_(-1, rescue_idx, True)
                else:
                    proposal_topk = min(self.expert_risk_proposal_topk, self.P)
                    proposal_idx = penalty_benefit_probs.topk(
                        k=proposal_topk,
                        dim=-1,
                    ).indices
                    proposal_mask.scatter_(-1, proposal_idx, True)
                rank_temperature = (
                    self.expert_risk_pairwise_temperature
                    if self.expert_risk_pairwise_rank_enable
                    else self.expert_risk_temperature
                )
                combined_rank_logits = (
                    pairwise_rank_scores / rank_temperature
                ).masked_fill(~proposal_mask, -1.0e4)
                penalty_conditional_probs = torch.softmax(combined_rank_logits, dim=-1)
                selected_penalty = combined_rank_logits.argmax(dim=-1, keepdim=True)
                if int(self.fixed_penalty_index_by_channel_c.numel()) > 0:
                    if int(self.fixed_penalty_index_by_channel_c.numel()) != int(x_bcl.shape[1]):
                        raise ValueError(
                            "patch_router fixed_penalty_index_by_channel length must match "
                            f"the channel count ({int(x_bcl.shape[1])})."
                        )
                    fixed_penalty_c = self.fixed_penalty_index_by_channel_c.to(
                        device=x_bcl.device,
                    )
                    fixed_penalty_active_bcq = (
                        fixed_penalty_c >= 0
                    ).view(1, -1, 1).expand(
                        int(x_bcl.shape[0]),
                        -1,
                        self.num_patches,
                    )
                    selected_penalty = fixed_penalty_c.clamp_min(0).view(
                        1,
                        -1,
                        1,
                        1,
                    ).expand(
                        int(x_bcl.shape[0]),
                        -1,
                        self.num_patches,
                        1,
                    )
                    penalty_conditional_probs = torch.zeros_like(
                        penalty_conditional_probs
                    ).scatter(
                        -1,
                        selected_penalty,
                        1.0,
                    )
                selected_utility = utility_scores.gather(-1, selected_penalty).squeeze(-1)
                selected_lower_quantile = risk_lower_quantile_scores.gather(
                    -1,
                    selected_penalty,
                ).squeeze(-1)
                selected_risk_benefit_prob = risk_benefit_prob.gather(
                    -1,
                    selected_penalty,
                ).squeeze(-1)
                assert self.expert_risk_adopt_threshold is not None
                if self.expert_risk_adoption_source == "benefit_probability":
                    selected_risk_score = selected_risk_benefit_prob
                    adopt_prob = selected_risk_benefit_prob
                elif self.expert_risk_adoption_source == "expected_utility":
                    selected_risk_score = selected_utility
                    adopt_prob = torch.sigmoid(
                        selected_utility / self.utility_verifier_temperature
                    )
                elif self.expert_risk_adoption_source == "utility_veto":
                    selected_risk_score = risk_utility_veto_prob.gather(
                        -1,
                        selected_penalty,
                    ).squeeze(-1)
                    adopt_prob = selected_risk_score
                else:
                    selected_risk_score = selected_lower_quantile
                    adopt_prob = torch.sigmoid(
                        selected_lower_quantile / self.utility_verifier_temperature
                    )
                if self.expert_risk_adopt_threshold_by_penalty is not None:
                    selected_risk_threshold = self.expert_risk_adopt_threshold_by_penalty[
                        selected_penalty.squeeze(-1)
                    ]
                else:
                    selected_risk_threshold = self.expert_risk_adopt_threshold
                utility_rejected = selected_risk_score <= selected_risk_threshold
                if fixed_penalty_active_bcq is not None:
                    utility_rejected = utility_rejected | (~fixed_penalty_active_bcq)
            else:
                combined_rank_logits = (
                    utility_scores + penalty_benefit_probs.clamp_min(1.0e-8).log()
                )
                penalty_conditional_probs = torch.softmax(combined_rank_logits, dim=-1)
            if self.utility_verifier_enable and not self.expert_conditional_risk_enable:
                max_utility = utility_scores.max(dim=-1).values
                adopt_prob = torch.sigmoid(
                    adopt_logits + max_utility / self.utility_verifier_temperature
                )
                utility_rejected = max_utility <= 0.0
            elif not self.expert_conditional_risk_enable:
                adopt_prob = proposal_adopt_prob
                utility_rejected = torch.zeros_like(adopt_prob, dtype=torch.bool)
            skip_prob = 1.0 - adopt_prob
            penalty_probs = adopt_prob.unsqueeze(-1) * penalty_conditional_probs
            skip_hard = (
                utility_rejected
                if self.expert_conditional_risk_enable
                else (proposal_adopt_prob < self.adopt_threshold) | utility_rejected
            ).to(dtype=adopt_prob.dtype)
        else:
            route_probs = torch.softmax(logits / self.temperature, dim=-1)
            if self.allow_skip:
                skip_prob = route_probs[..., 0]
                penalty_probs = route_probs[..., 1:]
                skip_hard = (route_probs.argmax(dim=-1) == 0).to(dtype=route_probs.dtype)
            else:
                skip_prob = torch.zeros_like(route_probs[..., 0])
                penalty_probs = route_probs
                skip_hard = torch.zeros_like(skip_prob)
            adopt_prob = 1.0 - skip_prob
            proposal_adopt_prob = adopt_prob
            penalty_conditional_probs = penalty_probs / adopt_prob.unsqueeze(-1).clamp_min(1.0e-8)
            penalty_benefit_probs = penalty_conditional_probs
            utility_scores = logits[..., -self.P :] / self.temperature
            risk_benefit_prob = penalty_benefit_probs
            risk_positive_magnitude = utility_scores.clamp_min(0.0)
            risk_negative_magnitude = (-utility_scores).clamp_min(0.0)
            risk_lower_quantile_scores = utility_scores
            risk_utility_veto_prob = risk_benefit_prob
            pairwise_rank_scores = utility_scores
            proposal_mask = penalty_benefit_probs > self.expert_risk_proposal_threshold

        if self.expert_risk_independent_activation_enable:
            activation_prob = torch.sigmoid(
                utility_scores / self.utility_verifier_temperature
            )
            penalty_probs = activation_prob
            skip_prob = (1.0 - activation_prob).prod(dim=-1)
            adopt_prob = 1.0 - skip_prob
            proposal_adopt_prob = adopt_prob
            skip_hard = (~(utility_scores > 0.0).any(dim=-1)).to(
                dtype=activation_prob.dtype
            )
            penalty_conditional_probs = torch.softmax(
                utility_scores / self.expert_risk_temperature,
                dim=-1,
            )

        selected_penalty_index = penalty_conditional_probs.argmax(dim=-1)
        if self.compositional_periodic_gate_enable:
            assert (
                mse_utility_scores is not None
                and mae_utility_scores is not None
                and periodic_mse_utility_scores is not None
                and periodic_mae_utility_scores is not None
            )
            zero_utility = torch.zeros_like(periodic_mse_utility_scores)
            combined_mse_utility = (
                periodic_mse_utility_scores.unsqueeze(-1)
                + mse_utility_scores
            )
            combined_mae_utility = (
                periodic_mae_utility_scores.unsqueeze(-1)
                + mae_utility_scores
            )
            action_mse_utility_scores = torch.cat(
                [
                    zero_utility.unsqueeze(-1),
                    periodic_mse_utility_scores.unsqueeze(-1),
                    combined_mse_utility,
                ],
                dim=-1,
            )
            action_mae_utility_scores = torch.cat(
                [
                    zero_utility.unsqueeze(-1),
                    periodic_mae_utility_scores.unsqueeze(-1),
                    combined_mae_utility,
                ],
                dim=-1,
            )
            action_scores = torch.minimum(
                action_mse_utility_scores,
                action_mae_utility_scores,
            )
            action_probs = torch.softmax(
                action_scores / self.temperature,
                dim=-1,
            )

            combined_scores = action_scores[..., 2:]
            if fixed_penalty_active_bcq is not None:
                fixed_candidate_mask = F.one_hot(
                    selected_penalty_index,
                    num_classes=self.P,
                ).to(dtype=torch.bool)
                fixed_candidate_mask = (
                    fixed_candidate_mask
                    & fixed_penalty_active_bcq.unsqueeze(-1)
                )
                combined_scores_for_selection = combined_scores.masked_fill(
                    ~fixed_candidate_mask,
                    float("-inf"),
                )
            else:
                combined_scores_for_selection = combined_scores
            best_combined_score, selected_penalty_index = (
                combined_scores_for_selection.max(dim=-1)
            )
            periodic_score = action_scores[..., 1]

            # Preserve the delivered always-periodic gate at epoch 0.  The
            # periodic utility heads are zero-initialized, so strict `>` keeps
            # P on a P/P+e tie while `>= 0` keeps P on a B/P tie.  With U_P=0,
            # this exactly reproduces the old U_e > 0 adapter boundary.
            use_combined = best_combined_score > periodic_score
            non_backbone_score = torch.where(
                use_combined,
                best_combined_score,
                periodic_score,
            )
            non_backbone_action = torch.where(
                use_combined,
                selected_penalty_index + 2,
                torch.ones_like(selected_penalty_index),
            )
            action_index = torch.where(
                non_backbone_score >= 0.0,
                non_backbone_action,
                torch.zeros_like(non_backbone_action),
            )
            backbone_route = action_index == 0
            periodic_only = action_index == 1
            periodic_route = ~backbone_route
            selected_risk_score = combined_scores.gather(
                dim=-1,
                index=selected_penalty_index.unsqueeze(-1),
            ).squeeze(-1)
            selected_risk_benefit_prob = risk_benefit_prob.gather(
                dim=-1,
                index=selected_penalty_index.unsqueeze(-1),
            ).squeeze(-1)
        if risk_domain_std is None:
            risk_domain_std = torch.zeros_like(risk_benefit_prob)
        if selected_risk_benefit_prob is None:
            selected_risk_benefit_prob = risk_benefit_prob.gather(
                dim=-1,
                index=selected_penalty_index.unsqueeze(-1),
            ).squeeze(-1)
        if selected_risk_score is None:
            selected_risk_score = risk_lower_quantile_scores.gather(
                dim=-1,
                index=selected_penalty_index.unsqueeze(-1),
            ).squeeze(-1)

        if self.expert_risk_independent_activation_enable:
            hard = (utility_scores > 0.0).to(dtype=penalty_probs.dtype)
        else:
            topk = max(1, min(self.topk, self.P))
            top_idx = penalty_probs.topk(k=topk, dim=-1).indices
            hard = torch.zeros_like(penalty_probs)
            hard.scatter_(-1, top_idx, 1.0)
            hard = hard * (1.0 - skip_hard.unsqueeze(-1))
        if self.compositional_periodic_gate_enable:
            assert action_index is not None
            hard = F.one_hot(
                (action_index - 2).clamp_min(0),
                num_classes=self.P,
            ).to(dtype=penalty_probs.dtype)
            combined_action = action_index >= 2
            hard = hard * combined_action.unsqueeze(-1).to(dtype=hard.dtype)
            skip_hard = (~combined_action).to(dtype=penalty_probs.dtype)
        use_soft_route = (
            (straight_through and self.training_route_mode == "soft")
            or ((not straight_through) and self.inference_route_mode == "soft")
        )
        if self.compositional_periodic_gate_enable and use_soft_route:
            assert action_probs is not None
            patch_route = action_probs[..., 2:]
            patch_skip = action_probs[..., :2].sum(dim=-1)
            penalty_probs = patch_route
            skip_prob = patch_skip
            adopt_prob = patch_route.sum(dim=-1)
        elif straight_through:
            patch_route = hard - penalty_probs.detach() + penalty_probs
            patch_skip = skip_hard - skip_prob.detach() + skip_prob
        elif use_soft_route:
            # Deterministic input-conditioned mixture.  Penalty probabilities
            # already include adoption mass, so the omitted mass is the exact
            # no-op/skip contribution and requires no residual branch.
            patch_route = penalty_probs
            patch_skip = skip_prob
        else:
            patch_route = hard
            patch_skip = skip_hard

        route_bcpq = patch_route.permute(0, 1, 3, 2).contiguous()
        route_bcph = route_bcpq.unsqueeze(-1).expand(
            -1,
            -1,
            -1,
            -1,
            self.patch_len,
        ).reshape(*route_bcpq.shape[:3], self.H)
        result = {
            "patch_route_bcph": route_bcph,
            "patch_probs_bcqp": penalty_probs,
            "patch_skip_bcq": patch_skip,
            "patch_skip_prob_bcq": skip_prob,
            "patch_adopt_prob_bcq": adopt_prob,
            "patch_fixed_penalty_active_bcq": (
                fixed_penalty_active_bcq
                if fixed_penalty_active_bcq is not None
                else torch.ones_like(adopt_prob, dtype=torch.bool)
            ),
            "patch_proposal_adopt_prob_bcq": proposal_adopt_prob,
            "patch_penalty_conditional_probs_bcqp": penalty_conditional_probs,
            "patch_penalty_benefit_probs_bcqp": penalty_benefit_probs,
            "patch_penalty_proposal_logits_bcqp": benefit_logits if self.hierarchical_recall_enable else logits[..., -self.P :],
            "patch_penalty_proposal_rescue_logits_bcqp": rescue_logits if self.hierarchical_recall_enable else logits[..., -self.P :],
            "patch_penalty_utility_scores_bcqp": utility_scores,
            "patch_penalty_risk_benefit_probs_bcqp": risk_benefit_prob,
            "patch_penalty_risk_domain_std_bcqp": risk_domain_std,
            "patch_penalty_risk_positive_magnitude_bcqp": risk_positive_magnitude,
            "patch_penalty_risk_negative_magnitude_bcqp": risk_negative_magnitude,
            "patch_penalty_risk_lower_quantile_scores_bcqp": risk_lower_quantile_scores,
            "patch_penalty_risk_utility_veto_probs_bcqp": risk_utility_veto_prob,
            "patch_penalty_pairwise_rank_scores_bcqp": pairwise_rank_scores,
            "patch_penalty_proposal_mask_bcqp": proposal_mask,
            "patch_selected_penalty_index_bcq": selected_penalty_index,
            "patch_selected_risk_score_bcq": selected_risk_score,
            "patch_selected_risk_benefit_prob_bcq": selected_risk_benefit_prob,
            "patch_selected_risk_domain_std_bcq": risk_domain_std.gather(
                dim=-1,
                index=selected_penalty_index.unsqueeze(-1),
            ).squeeze(-1),
            "patch_selected_utility_veto_prob_bcq": risk_utility_veto_prob.gather(
                dim=-1,
                index=selected_penalty_index.unsqueeze(-1),
            ).squeeze(-1),
        }
        if self.expert_risk_dual_signed_utility_enable:
            assert mse_utility_scores is not None and mae_utility_scores is not None
            result.update(
                {
                    "patch_penalty_mse_utility_scores_bcqp": mse_utility_scores,
                    "patch_penalty_mae_utility_scores_bcqp": mae_utility_scores,
                }
            )
            if predicted_residual is not None:
                result["patch_predicted_residual_bcqh"] = predicted_residual
        if self.compositional_periodic_gate_enable:
            assert (
                periodic_mse_utility_scores is not None
                and periodic_mae_utility_scores is not None
                and action_mse_utility_scores is not None
                and action_mae_utility_scores is not None
                and action_scores is not None
                and action_index is not None
                and periodic_route is not None
                and periodic_only is not None
                and backbone_route is not None
            )
            periodic_route_bch = periodic_route.unsqueeze(-1).expand(
                -1,
                -1,
                -1,
                self.patch_len,
            ).reshape(*periodic_route.shape[:2], self.H)
            periodic_route_output = periodic_route.to(dtype=patch_route.dtype)
            if use_soft_route:
                assert action_probs is not None
                periodic_route_weight = action_probs[..., 1:].sum(dim=-1)
                periodic_route_bch = periodic_route_weight.unsqueeze(-1).expand(
                    -1,
                    -1,
                    -1,
                    self.patch_len,
                ).reshape(*periodic_route_weight.shape[:2], self.H)
                periodic_route_output = periodic_route_weight.to(
                    dtype=patch_route.dtype
                )
            result.update(
                {
                    "patch_periodic_mse_utility_scores_bcq": (
                        periodic_mse_utility_scores
                    ),
                    "patch_periodic_mae_utility_scores_bcq": (
                        periodic_mae_utility_scores
                    ),
                    "patch_action_mse_utility_scores_bcqa": (
                        action_mse_utility_scores
                    ),
                    "patch_action_mae_utility_scores_bcqa": (
                        action_mae_utility_scores
                    ),
                    "patch_action_scores_bcqa": action_scores,
                    "patch_action_probs_bcqa": action_probs,
                    "patch_action_index_bcq": action_index,
                    "patch_periodic_route_bcq": periodic_route_output,
                    "patch_periodic_route_bch": periodic_route_bch.to(
                        dtype=patch_route.dtype
                    ),
                    "patch_periodic_only_bcq": periodic_only,
                    "patch_backbone_route_bcq": backbone_route,
                }
            )
        return result

    def set_expert_risk_adopt_threshold(self, threshold: float) -> None:
        if self.expert_risk_adopt_threshold is None:
            raise ValueError("expert risk adopt threshold requires expert_conditional_risk.")
        with torch.no_grad():
            self.expert_risk_adopt_threshold.fill_(float(threshold))

    def set_expert_risk_adopt_threshold_by_penalty(
        self,
        threshold_by_penalty: torch.Tensor | List[float],
    ) -> None:
        if self.expert_risk_adopt_threshold_by_penalty is None:
            raise ValueError(
                "per-penalty expert risk thresholds require "
                "temporal_calibration.per_penalty=true or adopt_threshold_by_penalty."
            )
        threshold = torch.as_tensor(
            threshold_by_penalty,
            dtype=self.expert_risk_adopt_threshold_by_penalty.dtype,
            device=self.expert_risk_adopt_threshold_by_penalty.device,
        ).reshape(-1)
        if int(threshold.numel()) != self.P:
            raise ValueError(
                f"expected {self.P} per-penalty thresholds, got {int(threshold.numel())}."
            )
        with torch.no_grad():
            self.expert_risk_adopt_threshold_by_penalty.copy_(threshold)


class ClusterwisePredResidualMoE(nn.Module):
    """
    Cluster-wise, penalty-keyed residual experts for prediction-side MoE.

    Each cluster owns P independent residual MLPs. Routing still happens at the
    cluster level through the existing gate; this module expands the selected
    cluster mask back to channels and adds the selected residual branches to the
    base forecast.
    """

    def __init__(
        self,
        num_clusters: int,
        num_penalties: int,
        input_len: int,
        pred_len: int,
        hidden_dim: int = 32,
        init_alpha: float = -3.0,
        alpha_scale: float = 0.5,
        use_y_base_input: bool = True,
        use_channel_identity_features: bool = False,
        feature_mode: str = "legacy",
        residual_clip: float = 0.0,
        intervention_enable: bool = True,
        intervention_init: float = -2.0,
        penalty_selector_enable: bool = False,
        selector_temperature: float = 1.0,
        selector_use_cluster_context: bool = True,
        fusion_gate_enable: bool = False,
        fusion_init: float = 0.0,
        fusion_use_cluster_context: bool = True,
        num_channels: int = 0,
        channel_expert_mask_c: Optional[torch.Tensor] = None,
        channel_expert_cluster_id_c: Optional[torch.Tensor] = None,
        channel_expert_mode: str = "override",
        penalty_names: Optional[List[str]] = None,
        seasonal_anchor_names: Optional[List[str]] = None,
        seasonal_anchor_period: int = 96,
        seasonal_anchor_num_periods: int = 1,
        seasonal_anchor_scale: float = 1.0,
        phase_residual_candidate_names: Optional[List[str]] = None,
        phase_residual_candidate_scale: float = 1.0,
        shared_across_clusters: bool = False,
        patch_router_cfg: Optional[Dict[str, Any]] = None,
        named_output_projection_enable: bool = False,
        named_output_projection_fixed_alpha: bool = False,
        named_output_projection_scale_by_name: Optional[Dict[str, float]] = None,
        named_output_projection_carrier_names: Optional[List[str]] = None,
        named_output_projection_patch_len: Optional[int] = None,
        periodic_anchor_expert_enable: bool = False,
        periodic_anchor_expert_scale: float = 1.0,
        position_daily_residual_expert_enable: bool = False,
        position_daily_residual_period: int = 96,
        position_daily_residual_harmonics: int = 4,
        anchor_ridge_gate_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.K = int(num_clusters)
        self.shared_across_clusters = bool(shared_across_clusters)
        self.param_K = 1 if self.shared_across_clusters else self.K
        self.P = int(num_penalties)
        self.L = int(input_len)
        self.H = int(pred_len)
        patch_router_cfg = patch_router_cfg or {}
        self.patch_router_enable = bool(patch_router_cfg.get("enable", False))
        if self.patch_router_enable and not self.shared_across_clusters:
            raise ValueError("patch_router requires shared_across_clusters=true.")
        self.patch_router = (
            ChannelPatchPenaltyRouter(
                input_len=self.L,
                pred_len=self.H,
                num_penalties=self.P,
                num_channels=int(num_channels or 0),
                cfg=patch_router_cfg,
            )
            if self.patch_router_enable
            else None
        )
        raw_patch_candidate_scale = patch_router_cfg.get(
            "candidate_scale_by_channel",
            None,
        )
        if raw_patch_candidate_scale is None:
            patch_candidate_scale_c = torch.empty(0, dtype=torch.float32)
        else:
            patch_candidate_scale_c = torch.as_tensor(
                raw_patch_candidate_scale,
                dtype=torch.float32,
            ).reshape(-1)
            if int(patch_candidate_scale_c.numel()) != int(num_channels or 0):
                raise ValueError(
                    "patch_router candidate_scale_by_channel length must match "
                    f"num_channels ({int(num_channels or 0)})."
                )
            if bool((patch_candidate_scale_c < 0.0).any().item()):
                raise ValueError(
                    "patch_router candidate_scale_by_channel values must be nonnegative."
                )
        self.register_buffer(
            "patch_candidate_scale_c",
            patch_candidate_scale_c,
            persistent=False,
        )
        raw_application_scale = patch_router_cfg.get(
            "application_scale_by_penalty",
            None,
        )
        if raw_application_scale is None:
            patch_application_scale_p = torch.empty(0, dtype=torch.float32)
        else:
            patch_application_scale_p = torch.as_tensor(
                raw_application_scale,
                dtype=torch.float32,
            ).reshape(-1)
            if int(patch_application_scale_p.numel()) != self.P:
                raise ValueError(
                    "patch_router application_scale_by_penalty length must match "
                    f"num_penalties ({self.P})."
                )
            if bool((patch_application_scale_p < 0.0).any().item()):
                raise ValueError(
                    "patch_router application_scale_by_penalty values must be nonnegative."
                )
        self.register_buffer(
            "patch_application_scale_p",
            patch_application_scale_p,
            persistent=False,
        )
        self.hidden_dim = int(hidden_dim)
        self.alpha_scale = float(alpha_scale)
        self.residual_clip = float(max(0.0, residual_clip))
        self.named_output_projection_enable = bool(named_output_projection_enable)
        self.named_output_projection_fixed_alpha = bool(named_output_projection_fixed_alpha)
        self.named_output_projection_carrier_names = frozenset(
            str(name) for name in (named_output_projection_carrier_names or [])
        )
        self.named_output_projection_patch_len = int(
            named_output_projection_patch_len or 0
        )
        if self.named_output_projection_patch_len < 0:
            raise ValueError("named_output_projection.patch_len must be nonnegative.")
        if (
            self.named_output_projection_patch_len > 0
            and self.H % self.named_output_projection_patch_len != 0
        ):
            raise ValueError(
                "named_output_projection.patch_len must divide pred_len exactly."
            )
        self.periodic_anchor_expert_enable = bool(periodic_anchor_expert_enable)
        self.periodic_anchor_expert_scale = float(periodic_anchor_expert_scale)
        self.position_daily_residual_expert_enable = bool(
            position_daily_residual_expert_enable
        )
        self.position_daily_residual_period = max(
            1, int(position_daily_residual_period)
        )
        self.position_daily_residual_harmonics = max(
            1, int(position_daily_residual_harmonics)
        )
        anchor_ridge_gate_cfg = anchor_ridge_gate_cfg or {}
        self.anchor_ridge_gate_enable = bool(
            anchor_ridge_gate_cfg.get("enable", False)
        )
        self.anchor_ridge_gate_hidden_dim = max(
            1, int(anchor_ridge_gate_cfg.get("hidden_dim", 16))
        )
        self.register_buffer(
            "position_daily_residual_coef_cfh",
            torch.empty(0, dtype=torch.float32),
            persistent=False,
        )
        if (
            self.patch_router is not None
            and self.patch_router.compositional_periodic_gate_enable
            and not self.periodic_anchor_expert_enable
        ):
            raise ValueError(
                "patch_router compositional_periodic_gate requires "
                "periodic_anchor_expert_enable=true."
            )
        self.use_y_base_input = bool(use_y_base_input)
        self.use_channel_identity_features = bool(use_channel_identity_features)
        self.feature_mode = str(feature_mode).lower()
        if self.feature_mode not in {"legacy", "safe_augmented"}:
            raise ValueError(
                "moe.pred_side_residual.feature_mode must be 'legacy' or 'safe_augmented'."
            )
        self.intervention_enable = bool(intervention_enable)
        self.intervention_init = float(intervention_init)
        self.penalty_selector_enable = bool(penalty_selector_enable)
        self.selector_temperature = max(float(selector_temperature), 1.0e-3)
        self.selector_use_cluster_context = bool(selector_use_cluster_context)
        self.fusion_gate_enable = bool(fusion_gate_enable)
        self.fusion_init = float(fusion_init)
        self.fusion_use_cluster_context = bool(fusion_use_cluster_context)
        self.C_channel = int(num_channels or 0)
        if channel_expert_mask_c is not None:
            mask = channel_expert_mask_c.detach().to(dtype=torch.bool).view(-1)
            self.C_channel = int(mask.numel())
        else:
            mask = torch.zeros(self.C_channel, dtype=torch.bool)
        if channel_expert_cluster_id_c is not None:
            parent = channel_expert_cluster_id_c.detach().to(dtype=torch.long).view(-1)
        else:
            parent = torch.zeros(self.C_channel, dtype=torch.long)
        if int(parent.numel()) != self.C_channel:
            raise ValueError(
                "channel_expert_cluster_id_c must have one entry per channel, "
                f"got {int(parent.numel())} vs {self.C_channel}"
            )
        self.channel_expert_enable = bool(mask.any().item())
        if self.shared_across_clusters and self.channel_expert_enable:
            raise ValueError("shared_across_clusters does not support channel_expert_adapters.")
        self.channel_expert_mode = str(channel_expert_mode or "override").lower()
        if self.channel_expert_mode not in {"override", "delta"}:
            raise ValueError("channel_expert_mode must be 'override' or 'delta'.")
        if penalty_names is None:
            names = [str(i) for i in range(self.P)]
        else:
            names = [str(name) for name in penalty_names]
            if len(names) != self.P:
                raise ValueError(f"penalty_names must have {self.P} entries, got {len(names)}")
        self.penalty_names = names
        projection_scale_by_name = named_output_projection_scale_by_name or {}
        projection_scale_p = torch.tensor(
            [float(projection_scale_by_name.get(name, 1.0)) for name in self.penalty_names],
            dtype=torch.float32,
        )
        anchor_name_set = {str(name) for name in (seasonal_anchor_names or [])}
        self.seasonal_anchor_period = max(int(seasonal_anchor_period), 1)
        self.seasonal_anchor_num_periods = max(int(seasonal_anchor_num_periods), 1)
        self.seasonal_anchor_scale = float(seasonal_anchor_scale)
        phase_residual_name_set = {str(name) for name in (phase_residual_candidate_names or [])}
        self.phase_residual_candidate_scale = float(phase_residual_candidate_scale)
        seasonal_mask = torch.tensor(
            [name in anchor_name_set for name in self.penalty_names],
            dtype=torch.float32,
        )
        phase_residual_mask = torch.tensor(
            [name in phase_residual_name_set for name in self.penalty_names],
            dtype=torch.float32,
        )
        seasonal_index = torch.zeros(
            self.H,
            self.seasonal_anchor_num_periods,
            dtype=torch.long,
        )
        seasonal_valid = torch.zeros(
            self.H,
            self.seasonal_anchor_num_periods,
            dtype=torch.bool,
        )
        for h in range(self.H):
            phase = h % self.seasonal_anchor_period
            for lag in range(1, self.seasonal_anchor_num_periods + 1):
                idx = self.L - lag * self.seasonal_anchor_period + phase
                if 0 <= idx < self.L:
                    seasonal_index[h, lag - 1] = int(idx)
                    seasonal_valid[h, lag - 1] = True
        if self.feature_mode == "legacy":
            input_dim = self.L + (self.H if self.use_y_base_input else 0)
        else:
            input_dim = self.L + self.H + 10 + (2 * self.H if self.use_y_base_input else 0)
        if self.use_channel_identity_features:
            if self.C_channel <= 0:
                raise ValueError(
                    "channel identity features require num_channels to be positive."
                )
            input_dim += self.C_channel
        self.input_dim = int(input_dim)
        self.selector_input_dim = self.input_dim * (3 if self.selector_use_cluster_context else 1)
        self.fusion_input_dim = self.input_dim * (3 if self.fusion_use_cluster_context else 1)
        if self.anchor_ridge_gate_enable:
            self.anchor_ridge_gate = nn.Sequential(
                nn.Linear(self.input_dim, self.anchor_ridge_gate_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.anchor_ridge_gate_hidden_dim, 2),
            )
            nn.init.xavier_uniform_(self.anchor_ridge_gate[0].weight)
            nn.init.zeros_(self.anchor_ridge_gate[0].bias)
            nn.init.zeros_(self.anchor_ridge_gate[2].weight)
            nn.init.constant_(self.anchor_ridge_gate[2].bias, 4.0)
        else:
            self.anchor_ridge_gate = None
        self.register_buffer(
            "anchor_ridge_gate_feature_mean_d",
            torch.empty(0, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "anchor_ridge_gate_feature_std_d",
            torch.empty(0, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "anchor_ridge_gate_fitted",
            torch.tensor(False, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "anchor_ridge_gate_threshold_2",
            torch.full((2,), 0.5, dtype=torch.float32),
            persistent=False,
        )

        # Use one real Parameter object per (cluster, penalty) expert. Keeping
        # penalty experts physically separate makes gradient isolation and
        # diagnostics unambiguous.
        num_experts = self.param_K * self.P
        self.W1 = nn.ParameterList(
            [nn.Parameter(torch.empty(self.input_dim, self.hidden_dim)) for _ in range(num_experts)]
        )
        self.b1 = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.hidden_dim)) for _ in range(num_experts)]
        )
        self.W2 = nn.ParameterList(
            [nn.Parameter(torch.empty(self.hidden_dim, self.H)) for _ in range(num_experts)]
        )
        self.b2 = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.H)) for _ in range(num_experts)]
        )
        self.log_alpha = nn.ParameterList(
            [nn.Parameter(torch.tensor(float(init_alpha))) for _ in range(num_experts)]
        )
        self.W_gate = nn.ParameterList(
            [nn.Parameter(torch.empty(self.hidden_dim)) for _ in range(num_experts)]
        )
        self.b_gate = nn.ParameterList(
            [nn.Parameter(torch.tensor(float(intervention_init))) for _ in range(num_experts)]
        )
        if self.penalty_selector_enable:
            self.W_selector = nn.ParameterList(
                [nn.Parameter(torch.empty(self.selector_input_dim, self.P)) for _ in range(self.param_K)]
            )
            self.b_selector = nn.ParameterList(
                [nn.Parameter(torch.zeros(self.P)) for _ in range(self.param_K)]
            )
        else:
            self.W_selector = nn.ParameterList()
            self.b_selector = nn.ParameterList()
        if self.fusion_gate_enable:
            self.W_fusion = nn.ParameterList(
                [nn.Parameter(torch.empty(self.fusion_input_dim)) for _ in range(self.param_K)]
            )
            self.b_fusion = nn.ParameterList(
                [nn.Parameter(torch.tensor(float(fusion_init))) for _ in range(self.param_K)]
            )
        else:
            self.W_fusion = nn.ParameterList()
            self.b_fusion = nn.ParameterList()
        if self.channel_expert_enable:
            num_channel_experts = self.C_channel * self.P
            self.channel_W1 = nn.ParameterList(
                [nn.Parameter(torch.empty(self.input_dim, self.hidden_dim)) for _ in range(num_channel_experts)]
            )
            self.channel_b1 = nn.ParameterList(
                [nn.Parameter(torch.zeros(self.hidden_dim)) for _ in range(num_channel_experts)]
            )
            self.channel_W2 = nn.ParameterList(
                [nn.Parameter(torch.empty(self.hidden_dim, self.H)) for _ in range(num_channel_experts)]
            )
            self.channel_b2 = nn.ParameterList(
                [nn.Parameter(torch.zeros(self.H)) for _ in range(num_channel_experts)]
            )
            self.channel_log_alpha = nn.ParameterList(
                [nn.Parameter(torch.tensor(float(init_alpha))) for _ in range(num_channel_experts)]
            )
            self.channel_W_gate = nn.ParameterList(
                [nn.Parameter(torch.empty(self.hidden_dim)) for _ in range(num_channel_experts)]
            )
            self.channel_b_gate = nn.ParameterList(
                [nn.Parameter(torch.tensor(float(intervention_init))) for _ in range(num_channel_experts)]
            )
        else:
            self.channel_W1 = nn.ParameterList()
            self.channel_b1 = nn.ParameterList()
            self.channel_W2 = nn.ParameterList()
            self.channel_b2 = nn.ParameterList()
            self.channel_log_alpha = nn.ParameterList()
            self.channel_W_gate = nn.ParameterList()
            self.channel_b_gate = nn.ParameterList()
        self.register_buffer("channel_expert_mask_c", mask, persistent=False)
        self.register_buffer("channel_expert_cluster_id_c", parent, persistent=False)
        self.register_buffer("channel_penalty_allowed_mask_cp", torch.empty(0), persistent=False)
        self.register_buffer("seasonal_anchor_mask_p", seasonal_mask, persistent=False)
        self.register_buffer("named_output_projection_scale_p", projection_scale_p, persistent=False)
        self.register_buffer("seasonal_anchor_index_hp", seasonal_index, persistent=False)
        self.register_buffer("seasonal_anchor_valid_hp", seasonal_valid, persistent=False)
        self.register_buffer("phase_residual_candidate_mask_p", phase_residual_mask, persistent=False)
        self.register_buffer("phase_residual_candidate_table_phc", torch.empty(0), persistent=False)
        self.register_buffer("confidence_threshold_kp", torch.empty(0), persistent=False)
        self.register_buffer("confidence_skip_threshold_k", torch.empty(0), persistent=False)
        self.register_buffer("patch_router_observed_history_tc", torch.empty(0), persistent=False)
        self.confidence_gate_enable = False
        self.reset_parameters()

    def _idx(self, k: int, p: int) -> int:
        return self._param_cluster(k) * self.P + int(p)

    def _param_cluster(self, k: int) -> int:
        return 0 if self.shared_across_clusters else int(k)

    def _stack_expert_params(self, params: nn.ParameterList) -> torch.Tensor:
        stacked = torch.stack(list(params), dim=0).reshape(self.param_K, self.P, *params[0].shape)
        if self.shared_across_clusters and self.K != 1:
            return stacked.expand(self.K, self.P, *stacked.shape[2:])
        return stacked

    def _stack_cluster_params(self, params: nn.ParameterList) -> torch.Tensor:
        stacked = torch.stack(list(params), dim=0)
        if self.shared_across_clusters and self.K != 1:
            return stacked.expand(self.K, *stacked.shape[1:])
        return stacked

    def _ch_idx(self, c: int, p: int) -> int:
        return int(c) * self.P + int(p)

    def set_channel_penalty_allowed_mask(self, mask_cp: Optional[torch.Tensor]) -> None:
        if mask_cp is None or int(mask_cp.numel()) == 0:
            self.channel_penalty_allowed_mask_cp = torch.empty(0, device=self.channel_penalty_allowed_mask_cp.device)
            return
        if mask_cp.ndim != 2 or int(mask_cp.shape[1]) != self.P:
            raise ValueError(
                f"channel penalty mask must have shape [C,{self.P}], got {tuple(mask_cp.shape)}"
            )
        self.channel_penalty_allowed_mask_cp = mask_cp.detach().to(dtype=torch.float32)

    def set_allowed_penalty_mask(self, mask_cp: Optional[torch.Tensor]) -> None:
        self.set_channel_penalty_allowed_mask(mask_cp)

    def set_phase_residual_candidate_table(self, table_phc: Optional[torch.Tensor]) -> None:
        device = self.phase_residual_candidate_mask_p.device
        if table_phc is None or int(table_phc.numel()) == 0:
            self.phase_residual_candidate_table_phc = torch.empty(0, device=device)
            return
        table = table_phc.detach().to(device=device, dtype=torch.float32)
        if table.ndim != 3 or int(table.shape[1]) != self.H:
            raise ValueError(
                "phase residual candidate table must have shape [period,H,C], "
                f"got {tuple(table.shape)} with H={self.H}"
            )
        self.phase_residual_candidate_table_phc = table

    def set_position_daily_residual_expert(
        self,
        coef_cfh: Optional[torch.Tensor],
        *,
        period: Optional[int] = None,
        harmonics: Optional[int] = None,
    ) -> None:
        if period is not None:
            self.position_daily_residual_period = max(1, int(period))
        if harmonics is not None:
            self.position_daily_residual_harmonics = max(1, int(harmonics))
        reference = next(self.parameters())
        if coef_cfh is None or int(coef_cfh.numel()) == 0:
            self.position_daily_residual_coef_cfh = torch.empty(
                0, device=reference.device, dtype=reference.dtype
            )
            return
        coef = coef_cfh.detach().to(
            device=reference.device, dtype=reference.dtype
        )
        expected_features = 1 + 2 * self.position_daily_residual_harmonics
        expected_shape = (self.C_channel, expected_features, self.H)
        if tuple(coef.shape) != expected_shape:
            raise ValueError(
                "position daily residual coefficients must have shape "
                f"{expected_shape}, got {tuple(coef.shape)}."
            )
        self.position_daily_residual_coef_cfh = coef

    def _position_daily_residual_expert_branch(
        self,
        query_start_abs_b: Optional[torch.Tensor],
        reference_bch: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not self.position_daily_residual_expert_enable
            or int(self.position_daily_residual_coef_cfh.numel()) == 0
        ):
            return torch.zeros_like(reference_bch)
        if query_start_abs_b is None:
            raise ValueError(
                "query_start_abs_b is required by position_daily_residual_expert."
            )
        origin = query_start_abs_b.to(
            device=reference_bch.device, dtype=reference_bch.dtype
        ) + float(self.L)
        angle = 2.0 * math.pi * origin / float(
            self.position_daily_residual_period
        )
        features = [torch.ones_like(angle)]
        for harmonic in range(1, self.position_daily_residual_harmonics + 1):
            features.extend(
                [
                    torch.sin(float(harmonic) * angle),
                    torch.cos(float(harmonic) * angle),
                ]
            )
        feature_bf = torch.stack(features, dim=-1)
        coef_cfh = self.position_daily_residual_coef_cfh.to(
            device=reference_bch.device, dtype=reference_bch.dtype
        )
        return torch.einsum("bf,cfh->bch", feature_bf, coef_cfh)

    def set_anchor_ridge_gate_normalization(
        self,
        feature_mean_d: torch.Tensor,
        feature_std_d: torch.Tensor,
        *,
        fitted: bool,
    ) -> None:
        if not self.anchor_ridge_gate_enable or self.anchor_ridge_gate is None:
            raise ValueError("anchor_ridge_gate is not enabled.")
        reference = next(self.anchor_ridge_gate.parameters())
        mean = feature_mean_d.detach().reshape(-1).to(
            device=reference.device,
            dtype=reference.dtype,
        )
        std = feature_std_d.detach().reshape(-1).to(
            device=reference.device,
            dtype=reference.dtype,
        )
        if int(mean.numel()) != self.input_dim or int(std.numel()) != self.input_dim:
            raise ValueError(
                "anchor_ridge_gate normalization must match input_dim "
                f"{self.input_dim}, got {int(mean.numel())}/{int(std.numel())}."
            )
        self.anchor_ridge_gate_feature_mean_d = mean
        self.anchor_ridge_gate_feature_std_d = std.clamp_min(1.0e-6)
        self.anchor_ridge_gate_fitted.fill_(bool(fitted))

    def anchor_ridge_gate_weights_from_features(
        self,
        feature_bcd: torch.Tensor,
        *,
        hard: Optional[bool] = None,
    ) -> torch.Tensor:
        if (
            not self.anchor_ridge_gate_enable
            or self.anchor_ridge_gate is None
            or not bool(self.anchor_ridge_gate_fitted.item())
        ):
            return torch.ones(
                *feature_bcd.shape[:2],
                2,
                device=feature_bcd.device,
                dtype=feature_bcd.dtype,
            )
        if (
            int(self.anchor_ridge_gate_feature_mean_d.numel()) != self.input_dim
            or int(self.anchor_ridge_gate_feature_std_d.numel()) != self.input_dim
        ):
            raise RuntimeError("anchor_ridge_gate is fitted without normalization.")
        mean = self.anchor_ridge_gate_feature_mean_d.to(
            device=feature_bcd.device,
            dtype=feature_bcd.dtype,
        )
        std = self.anchor_ridge_gate_feature_std_d.to(
            device=feature_bcd.device,
            dtype=feature_bcd.dtype,
        )
        normalized = (feature_bcd - mean.view(1, 1, -1)) / std.view(1, 1, -1)
        probability = torch.sigmoid(self.anchor_ridge_gate(normalized))
        use_hard = (not self.training) if hard is None else bool(hard)
        if not use_hard:
            return probability
        threshold = self.anchor_ridge_gate_threshold_2.to(
            device=probability.device,
            dtype=probability.dtype,
        )
        return (probability >= threshold.view(1, 1, 2)).to(
            dtype=probability.dtype
        )

    def set_anchor_ridge_gate_thresholds(
        self,
        anchor_threshold: float,
        ridge_threshold: float,
    ) -> None:
        threshold = torch.tensor(
            [float(anchor_threshold), float(ridge_threshold)],
            device=self.anchor_ridge_gate_threshold_2.device,
            dtype=self.anchor_ridge_gate_threshold_2.dtype,
        ).clamp(0.0, 1.0)
        self.anchor_ridge_gate_threshold_2.copy_(threshold)

    def set_patch_router_observed_history(self, observed_history_tc: torch.Tensor) -> None:
        if self.patch_router is None or not self.patch_router.regime_context_enable:
            self.patch_router_observed_history_tc = torch.empty(
                0,
                device=next(self.parameters()).device,
            )
            return
        if observed_history_tc.ndim != 2:
            raise ValueError("patch router observed history must have shape [T,C].")
        reference = next(self.parameters())
        self.patch_router_observed_history_tc = observed_history_tc.detach().to(
            device=reference.device,
            dtype=reference.dtype,
        )

    def _patch_router_regime_context(
        self,
        x_bcl: torch.Tensor,
        query_start_abs_b: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if self.patch_router is None or not self.patch_router.regime_context_enable:
            return None
        if query_start_abs_b is None:
            raise ValueError(
                "query_start_abs_b is required when patch_router.regime_context is enabled."
            )
        history = self.patch_router_observed_history_tc
        if history.ndim != 2 or int(history.numel()) == 0:
            raise ValueError("patch router regime context history has not been initialized.")
        if int(history.shape[1]) != int(x_bcl.shape[1]):
            raise ValueError("patch router regime context channel count does not match input.")
        context_len = max(self.patch_router.regime_context_lengths)
        query_start = torch.as_tensor(
            query_start_abs_b,
            device=history.device,
            dtype=torch.long,
        ).reshape(-1)
        if int(query_start.numel()) != int(x_bcl.shape[0]):
            raise ValueError("patch router regime context query count does not match batch size.")
        forecast_origin = query_start + int(self.L)
        if bool((forecast_origin > int(history.shape[0])).any().item()):
            raise ValueError("patch router regime context query exceeds observed history.")
        offsets = torch.arange(
            -int(context_len),
            0,
            device=history.device,
            dtype=torch.long,
        )
        indices = (forecast_origin.unsqueeze(-1) + offsets.unsqueeze(0)).clamp(
            0,
            int(history.shape[0]) - 1,
        )
        context = history.index_select(0, indices.reshape(-1)).reshape(
            int(x_bcl.shape[0]),
            int(context_len),
            int(history.shape[1]),
        )
        return context.permute(0, 2, 1).to(device=x_bcl.device, dtype=x_bcl.dtype)

    def set_confidence_gate(
        self,
        penalty_threshold_kp: Optional[torch.Tensor] = None,
        skip_threshold_k: Optional[torch.Tensor] = None,
        enable: bool = True,
    ) -> None:
        self.confidence_gate_enable = bool(enable)
        device = self.seasonal_anchor_mask_p.device
        if penalty_threshold_kp is None:
            self.confidence_threshold_kp = torch.empty(0, device=device)
        else:
            threshold = penalty_threshold_kp.detach().to(device=device, dtype=torch.float32)
            if threshold.shape != (self.K, self.P):
                raise ValueError(
                    "confidence penalty threshold must have shape "
                    f"[{self.K},{self.P}], got {tuple(threshold.shape)}"
                )
            self.confidence_threshold_kp = threshold.clamp_min(0.0)
        if skip_threshold_k is None:
            self.confidence_skip_threshold_k = torch.empty(0, device=device)
        else:
            skip_threshold = skip_threshold_k.detach().to(device=device, dtype=torch.float32).view(-1)
            if int(skip_threshold.numel()) != self.K:
                raise ValueError(
                    f"confidence skip threshold must have length {self.K}, got {int(skip_threshold.numel())}"
                )
            self.confidence_skip_threshold_k = skip_threshold.clamp(0.0, 1.0)

    def reset_parameters(self):
        for w in self.W1:
            nn.init.xavier_uniform_(w)
        for w in self.W2:
            nn.init.zeros_(w)
        for b in self.b2:
            nn.init.zeros_(b)
        for w in self.W_gate:
            nn.init.zeros_(w)
        for w in self.W_selector:
            nn.init.zeros_(w)
        for w in self.W_fusion:
            nn.init.zeros_(w)
        for w in self.channel_W1:
            nn.init.xavier_uniform_(w)
        for w in self.channel_W2:
            nn.init.zeros_(w)
        for w in self.channel_W_gate:
            nn.init.zeros_(w)

    def _history_proxy_forecast(self, x_bcl: torch.Tensor) -> torch.Tensor:
        if self.L >= self.H:
            return x_bcl[..., -self.H:]
        pad = x_bcl[..., -1:].expand(*x_bcl.shape[:-1], self.H - self.L)
        return torch.cat([x_bcl, pad], dim=-1)

    def _seasonal_anchor_forecast(self, x_bcl: torch.Tensor) -> torch.Tensor:
        """Repeat same-phase observations from input history without target access."""
        idx = self.seasonal_anchor_index_hp.to(device=x_bcl.device)
        valid = self.seasonal_anchor_valid_hp.to(device=x_bcl.device)
        if idx.numel() == 0:
            return x_bcl[..., -1:].expand(*x_bcl.shape[:2], self.H)
        values = x_bcl.index_select(dim=-1, index=idx.reshape(-1)).reshape(
            *x_bcl.shape[:2],
            self.H,
            self.seasonal_anchor_num_periods,
        )
        valid_f = valid.to(dtype=x_bcl.dtype)
        counts = valid_f.sum(dim=-1).clamp_min(1.0)
        anchors = (values * valid_f.view(1, 1, self.H, self.seasonal_anchor_num_periods)).sum(dim=-1)
        anchors = anchors / counts.view(1, 1, self.H)
        fallback = x_bcl[..., -1:].expand_as(anchors)
        has_anchor = valid.any(dim=-1).view(1, 1, self.H)
        return torch.where(has_anchor, anchors, fallback)

    def _phase_residual_candidate_forecast(
        self,
        query_start_abs_b: torch.Tensor,
        *,
        channel_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        table = self.phase_residual_candidate_table_phc
        if int(table.numel()) == 0:
            raise ValueError("phase residual candidate table is required when phase_residual_candidate_names is non-empty.")
        if table.ndim != 3 or int(table.shape[1]) != self.H or int(table.shape[2]) != int(channel_count):
            raise ValueError(
                "phase residual candidate table must have shape [period,H,C], "
                f"got {tuple(table.shape)} for H={self.H}, C={int(channel_count)}"
            )
        phases_b = (query_start_abs_b.detach().to(device=device, dtype=torch.long).reshape(-1) + self.L) % int(table.shape[0])
        if int(phases_b.numel()) == 0:
            return table.new_zeros((0, int(channel_count), self.H), dtype=dtype, device=device)
        residual_bhc = table.to(device=device, dtype=dtype).index_select(0, phases_b)
        return residual_bhc.permute(0, 2, 1).contiguous()

    def _safe_augmented_features(self, x_bcl: torch.Tensor, y_base_bch: torch.Tensor) -> torch.Tensor:
        eps = 1.0e-6
        last = x_bcl[..., -1:]
        x_centered = x_bcl - last
        proxy = self._history_proxy_forecast(x_bcl)
        proxy_centered = proxy - last

        hist_mean = x_bcl.mean(dim=-1)
        hist_std = x_bcl.std(dim=-1, unbiased=False).clamp_min(eps)
        hist_range = (x_bcl.amax(dim=-1) - x_bcl.amin(dim=-1)) / hist_std
        t_l = torch.linspace(-1.0, 1.0, steps=self.L, device=x_bcl.device, dtype=x_bcl.dtype).view(1, 1, -1)
        hist_slope = ((x_bcl - hist_mean.unsqueeze(-1)) * t_l).mean(dim=-1) / t_l.pow(2).mean().clamp_min(eps)
        hist_slope = hist_slope / hist_std
        if self.L >= 2:
            d1 = x_bcl[..., 1:] - x_bcl[..., :-1]
            recent_delta = d1[..., -1] / hist_std
            mad1 = d1.abs().mean(dim=-1) / hist_std
        else:
            recent_delta = torch.zeros_like(hist_mean)
            mad1 = torch.zeros_like(hist_mean)
            d1 = None
        if self.L >= 3 and d1 is not None:
            d2 = x_bcl[..., 2:] - 2.0 * x_bcl[..., 1:-1] + x_bcl[..., :-2]
            mad2 = d2.abs().mean(dim=-1) / hist_std
        else:
            mad2 = torch.zeros_like(hist_mean)
        proxy_std = proxy.std(dim=-1, unbiased=False) / hist_std

        if self.use_y_base_input:
            y_centered = y_base_bch - last
            base_minus_proxy = y_base_bch - proxy
            base_std = y_base_bch.std(dim=-1, unbiased=False) / hist_std
            base_shift = (y_base_bch.mean(dim=-1) - last.squeeze(-1)) / hist_std
        else:
            y_centered = None
            base_minus_proxy = None
            base_std = torch.zeros_like(hist_mean)
            base_shift = torch.zeros_like(hist_mean)

        scalar = torch.stack(
            [
                (hist_mean - last.squeeze(-1)) / hist_std,
                hist_std.log(),
                hist_range,
                hist_slope,
                recent_delta,
                mad1,
                mad2,
                proxy_std,
                base_std,
                base_shift,
            ],
            dim=-1,
        )
        parts = [x_centered, proxy_centered, scalar]
        if self.use_y_base_input:
            parts.extend([y_centered, base_minus_proxy])
        return torch.cat(parts, dim=-1)

    def _input_features(self, x_bcl: torch.Tensor, y_base_bch: torch.Tensor) -> torch.Tensor:
        last = x_bcl[..., -1:]
        x_centered = x_bcl - last
        if self.feature_mode == "safe_augmented":
            features = self._safe_augmented_features(x_bcl, y_base_bch)
        elif not self.use_y_base_input:
            features = x_centered
        else:
            y_centered = y_base_bch - last
            features = torch.cat([x_centered, y_centered], dim=-1)
        if self.use_channel_identity_features:
            channel_identity = torch.eye(
                int(x_bcl.shape[1]),
                device=x_bcl.device,
                dtype=x_bcl.dtype,
            ).unsqueeze(0).expand(int(x_bcl.shape[0]), -1, -1)
            features = torch.cat([features, channel_identity], dim=-1)
        return features

    @staticmethod
    def _remove_affine_component(values_bch: torch.Tensor) -> torch.Tensor:
        """Remove the constant and linear null-space of a second difference."""
        centered = values_bch - values_bch.mean(dim=-1, keepdim=True)
        if int(values_bch.shape[-1]) <= 1:
            return centered
        trend_h = torch.linspace(
            -1.0,
            1.0,
            int(values_bch.shape[-1]),
            device=values_bch.device,
            dtype=values_bch.dtype,
        )
        trend_h = trend_h - trend_h.mean()
        denom = trend_h.pow(2).sum().clamp_min(1.0e-12)
        coef_bc = (centered * trend_h.view(1, 1, -1)).sum(dim=-1, keepdim=True) / denom
        return centered - coef_bc * trend_h.view(1, 1, -1)

    def _project_named_segment(
        self,
        raw_bch: torch.Tensor,
        base_bch: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        """Map a free residual onto the correction space named by ``name``."""
        if name == "level":
            projected = raw_bch.mean(dim=-1, keepdim=True).expand_as(raw_bch)
        elif name == "delta":
            if name in self.named_output_projection_carrier_names:
                carrier = base_bch - base_bch.mean(dim=-1, keepdim=True)
                projected = self._project_onto_bounded_carrier(raw_bch, carrier)
            else:
                projected = raw_bch - raw_bch.mean(dim=-1, keepdim=True)
        elif name == "d2_match":
            if name in self.named_output_projection_carrier_names:
                carrier = self._remove_affine_component(base_bch)
                projected = self._project_onto_bounded_carrier(raw_bch, carrier)
            else:
                projected = self._remove_affine_component(raw_bch)
        elif name in {"diff_amp", "amp", "amp_under"}:
            # Amplitude experts may only rescale the current centered shape.
            # A tanh-bounded coefficient gives a deployed factor in [0.5, 1.5].
            carrier = base_bch - base_bch.mean(dim=-1, keepdim=True)
            projected = self._project_onto_bounded_carrier(raw_bch, carrier)
        else:
            projected = raw_bch

        # Keep residual_clip as a true bound without destroying the projection
        # invariants through a second pointwise tanh.
        if self.residual_clip > 0.0:
            max_abs = projected.abs().amax(dim=-1, keepdim=True)
            scale = torch.clamp(float(self.residual_clip) / max_abs.clamp_min(1.0e-12), max=1.0)
            projected = projected * scale
        return projected

    @staticmethod
    def _project_onto_bounded_carrier(
        raw_bch: torch.Tensor,
        carrier_bch: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce a free residual to one bounded structural scale coefficient."""
        denom = carrier_bch.pow(2).sum(dim=-1, keepdim=True)
        raw_coef = (
            (raw_bch * carrier_bch).sum(dim=-1, keepdim=True)
            / denom.clamp_min(1.0e-12)
        )
        coef = 0.5 * torch.tanh(raw_coef / 0.5)
        projected = coef * carrier_bch
        return torch.where(denom > 1.0e-12, projected, torch.zeros_like(projected))

    def _project_named_residuals(
        self,
        residuals_bcph: torch.Tensor,
        base_bch: torch.Tensor,
    ) -> torch.Tensor:
        if not self.named_output_projection_enable:
            return residuals_bcph
        patch_len = self.named_output_projection_patch_len
        if patch_len <= 0 and self.patch_router is not None:
            patch_len = max(1, int(self.patch_router.patch_len))
        if patch_len <= 0:
            patch_len = self.H
        projected_p = []
        for p, name in enumerate(self.penalty_names):
            pieces = []
            for start in range(0, self.H, patch_len):
                end = min(start + patch_len, self.H)
                pieces.append(
                    self._project_named_segment(
                        residuals_bcph[:, :, p, start:end],
                        base_bch[:, :, start:end],
                        name,
                    )
                )
            projected_p.append(torch.cat(pieces, dim=-1))
        return torch.stack(projected_p, dim=2)

    def _cluster_context_features(
        self,
        feat_bcd: torch.Tensor,
        cluster_id_c: torch.Tensor,
        use_cluster_context: bool,
    ) -> torch.Tensor:
        if not use_cluster_context:
            return feat_bcd
        cluster_mean_bkd = scatter_mean_bcl_to_bkl(feat_bcd, cluster_id_c, self.K)
        cluster_mean_bcd = cluster_mean_bkd.index_select(1, cluster_id_c)
        return torch.cat([feat_bcd, cluster_mean_bcd, feat_bcd - cluster_mean_bcd], dim=-1)

    def forward(
        self,
        x_bcl: torch.Tensor,
        y_base_bch: torch.Tensor,
        cluster_id_c: torch.Tensor,
        mask_bkp: torch.Tensor,
        skip_bk: Optional[torch.Tensor] = None,
        query_start_abs_b: Optional[torch.Tensor] = None,
        fixed_expert_delta_bch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
          y_final: [B,C,H]
          residuals: [B,C,P,H]
          branches: [B,C,P,H]
          route_bcp: [B,C,P] after optional skip suppression
          intervention_bcp: [B,C,P] target-free expert intervention gate
          effective_route_bcp: [B,C,P] route_bcp * intervention_bcp
          alpha_cp: [C,P]
        """
        if self.periodic_anchor_expert_enable:
            if fixed_expert_delta_bch is None:
                raise ValueError(
                    "fixed_expert_delta_bch is required when periodic_anchor_expert is enabled."
                )
            if fixed_expert_delta_bch.shape != y_base_bch.shape:
                raise ValueError(
                    "fixed_expert_delta_bch must match y_base_bch, got "
                    f"{tuple(fixed_expert_delta_bch.shape)} vs {tuple(y_base_bch.shape)}."
                )
            periodic_expert_branch_bch = (
                float(self.periodic_anchor_expert_scale)
                * fixed_expert_delta_bch.to(device=y_base_bch.device, dtype=y_base_bch.dtype)
            )
        else:
            periodic_expert_branch_bch = torch.zeros_like(y_base_bch)
        position_daily_expert_branch_bch = (
            self._position_daily_residual_expert_branch(
                query_start_abs_b,
                y_base_bch,
            )
        )
        anchor_ridge_gate_features_bcd = self._input_features(
            x_bcl,
            y_base_bch,
        )
        anchor_ridge_gate_weights_bc2 = (
            self.anchor_ridge_gate_weights_from_features(
                anchor_ridge_gate_features_bcd
            )
        )
        anchor_gate_weight_bc = anchor_ridge_gate_weights_bc2[..., 0]
        ridge_gate_weight_bc = anchor_ridge_gate_weights_bc2[..., 1]
        candidate_base_bch = y_base_bch + periodic_expert_branch_bch

        if self.P <= 0:
            zero_res = y_base_bch.new_zeros((*y_base_bch.shape[:2], 0, y_base_bch.shape[-1]))
            zero_route = y_base_bch.new_zeros((*y_base_bch.shape[:2], 0))
            return {
                "y_final": (
                    y_base_bch
                    + anchor_gate_weight_bc.unsqueeze(-1)
                    * periodic_expert_branch_bch
                    + ridge_gate_weight_bc.unsqueeze(-1)
                    * position_daily_expert_branch_bch
                ),
                "residuals": zero_res,
                "branches": zero_res,
                "route_bcp": zero_route,
                "intervention_bcp": zero_route,
                "effective_route_bcp": zero_route,
                "alpha_cp": y_base_bch.new_zeros((y_base_bch.shape[1], 0)),
                "candidate_base_bch": candidate_base_bch,
                "periodic_expert_branch_bch": periodic_expert_branch_bch,
                "periodic_expert_route_bc": torch.ones_like(y_base_bch[..., 0]),
                "position_daily_residual_expert_branch_bch": position_daily_expert_branch_bch,
                "anchor_ridge_gate_features_bcd": anchor_ridge_gate_features_bcd,
                "anchor_ridge_gate_weights_bc2": anchor_ridge_gate_weights_bc2,
            }

        feat_bcd = self._input_features(x_bcl, candidate_base_bch)
        cluster_id_c = cluster_id_c.to(device=x_bcl.device, dtype=torch.long)

        W1_kpdm = self._stack_expert_params(self.W1)
        b1_kpm = self._stack_expert_params(self.b1)
        W2_kpmh = self._stack_expert_params(self.W2)
        b2_kph = self._stack_expert_params(self.b2)
        Wg_kpm = self._stack_expert_params(self.W_gate)
        bg_kp = self._stack_expert_params(self.b_gate)
        W1 = W1_kpdm.index_select(0, cluster_id_c)  # [C,P,D,M]
        b1 = b1_kpm.index_select(0, cluster_id_c)  # [C,P,M]
        W2 = W2_kpmh.index_select(0, cluster_id_c)  # [C,P,M,H]
        b2 = b2_kph.index_select(0, cluster_id_c)  # [C,P,H]
        Wg = Wg_kpm.index_select(0, cluster_id_c)  # [C,P,M]
        bg = bg_kp.index_select(0, cluster_id_c)  # [C,P]

        h = torch.einsum("bcd,cpdm->bcpm", feat_bcd, W1) + b1.unsqueeze(0)
        h = F.gelu(h)
        residuals = torch.einsum("bcpm,cpmh->bcph", h, W2) + b2.unsqueeze(0)
        if (
            self.seasonal_anchor_scale != 0.0
            and self.seasonal_anchor_mask_p.numel() == self.P
            and bool((self.seasonal_anchor_mask_p > 0).any().item())
        ):
            seasonal_anchor = self._seasonal_anchor_forecast(x_bcl)
            anchor_residual = seasonal_anchor - candidate_base_bch
            mask_p = self.seasonal_anchor_mask_p.to(device=x_bcl.device, dtype=residuals.dtype)
            residuals = residuals + (
                float(self.seasonal_anchor_scale)
                * mask_p.view(1, 1, self.P, 1)
                * anchor_residual.unsqueeze(2)
            )
        if (
            self.phase_residual_candidate_scale != 0.0
            and self.phase_residual_candidate_mask_p.numel() == self.P
            and bool((self.phase_residual_candidate_mask_p > 0).any().item())
        ):
            if query_start_abs_b is None:
                raise ValueError("query_start_abs_b is required when phase_residual_candidate_names is non-empty.")
            phase_residual = self._phase_residual_candidate_forecast(
                query_start_abs_b,
                channel_count=int(y_base_bch.shape[1]),
                device=x_bcl.device,
                dtype=residuals.dtype,
            )
            mask_p = self.phase_residual_candidate_mask_p.to(device=x_bcl.device, dtype=residuals.dtype)
            residuals = residuals + (
                float(self.phase_residual_candidate_scale)
                * mask_p.view(1, 1, self.P, 1)
                * phase_residual.unsqueeze(2)
            )
        if self.channel_expert_enable:
            if self.C_channel != int(feat_bcd.shape[1]):
                raise ValueError(
                    f"channel expert adapters expected {self.C_channel} channels, got {int(feat_bcd.shape[1])}"
                )
            ch_W1 = torch.stack(list(self.channel_W1), dim=0).reshape(
                self.C_channel, self.P, self.input_dim, self.hidden_dim
            )
            ch_b1 = torch.stack(list(self.channel_b1), dim=0).reshape(self.C_channel, self.P, self.hidden_dim)
            ch_W2 = torch.stack(list(self.channel_W2), dim=0).reshape(
                self.C_channel, self.P, self.hidden_dim, self.H
            )
            ch_b2 = torch.stack(list(self.channel_b2), dim=0).reshape(self.C_channel, self.P, self.H)
            h_ch = torch.einsum("bcd,cpdm->bcpm", feat_bcd, ch_W1) + ch_b1.unsqueeze(0)
            h_ch = F.gelu(h_ch)
            residuals_ch = torch.einsum("bcpm,cpmh->bcph", h_ch, ch_W2) + ch_b2.unsqueeze(0)
            ch_mask_bcpm = self.channel_expert_mask_c.to(device=x_bcl.device).view(1, -1, 1, 1)
            if self.channel_expert_mode == "delta":
                residuals = residuals + ch_mask_bcpm.expand_as(residuals) * residuals_ch
            else:
                h = torch.where(ch_mask_bcpm, h_ch, h)
                residuals = torch.where(ch_mask_bcpm.expand_as(residuals), residuals_ch, residuals)
        if self.residual_clip > 0.0:
            clip = float(self.residual_clip)
            residuals = clip * torch.tanh(residuals / clip)
        residuals = self._project_named_residuals(residuals, candidate_base_bch)

        alpha_cp = self.alpha_values().index_select(0, cluster_id_c)  # [C,P]
        if self.channel_expert_enable:
            ch_alpha_cp = self.alpha_scale * torch.sigmoid(
                torch.stack(list(self.channel_log_alpha), dim=0).reshape(self.C_channel, self.P)
            )
            ch_mask_cp = self.channel_expert_mask_c.to(device=x_bcl.device).view(-1, 1)
            alpha_cp = torch.where(ch_mask_cp, ch_alpha_cp, alpha_cp)
        if self.named_output_projection_enable and self.named_output_projection_fixed_alpha:
            alpha_cp = self.named_output_projection_scale_p.to(
                device=x_bcl.device,
                dtype=residuals.dtype,
            ).view(1, self.P).expand(int(residuals.shape[1]), self.P)
        patch_router_out: Optional[Dict[str, torch.Tensor]] = None
        route_bcph: Optional[torch.Tensor] = None
        if self.patch_router is not None:
            patch_candidate_scale_bc = None
            if int(self.patch_candidate_scale_c.numel()) > 0:
                patch_candidate_scale_bc = self.patch_candidate_scale_c.to(
                    device=x_bcl.device,
                    dtype=residuals.dtype,
                ).view(1, -1)
            candidate_delta_bcpH = alpha_cp.unsqueeze(0).unsqueeze(-1) * residuals
            if patch_candidate_scale_bc is not None:
                candidate_delta_bcpH = (
                    candidate_delta_bcpH
                    * patch_candidate_scale_bc.unsqueeze(-1).unsqueeze(-1)
                )
            regime_context_bcl = self._patch_router_regime_context(
                x_bcl,
                query_start_abs_b,
            )
            patch_router_out = self.patch_router(
                x_bcl,
                y_base_bch=candidate_base_bch,
                candidate_delta_bcpH=candidate_delta_bcpH,
                regime_context_bcl=regime_context_bcl,
                query_start_abs_b=query_start_abs_b,
                straight_through=self.training,
            )
            route_bcph = patch_router_out["patch_route_bcph"]
            route_bcp = route_bcph.mean(dim=-1)
        else:
            route_bcp = mask_bkp[:, cluster_id_c, :]
            if skip_bk is not None:
                route_bcp = route_bcp * (1.0 - skip_bk[:, cluster_id_c].unsqueeze(-1))
        if self.channel_penalty_allowed_mask_cp.numel() > 0:
            channel_mask_cp = self.channel_penalty_allowed_mask_cp.to(device=x_bcl.device, dtype=route_bcp.dtype)
            if channel_mask_cp.shape != route_bcp.shape[1:]:
                raise ValueError(
                    "channel penalty mask shape must match [C,P], "
                    f"got {tuple(channel_mask_cp.shape)} vs {tuple(route_bcp.shape[1:])}"
                )
            route_bcp = route_bcp * channel_mask_cp.unsqueeze(0)
            if route_bcph is not None:
                route_bcph = route_bcph * channel_mask_cp.unsqueeze(0).unsqueeze(-1)
        if self.intervention_enable:
            gate_logits = torch.einsum("bcpm,cpm->bcp", h, Wg) + bg.unsqueeze(0)
            if self.channel_expert_enable:
                ch_Wg = torch.stack(list(self.channel_W_gate), dim=0).reshape(
                    self.C_channel, self.P, self.hidden_dim
                )
                ch_bg = torch.stack(list(self.channel_b_gate), dim=0).reshape(self.C_channel, self.P)
                gate_logits_ch = torch.einsum("bcpm,cpm->bcp", h, ch_Wg) + ch_bg.unsqueeze(0)
                ch_mask_bcp = self.channel_expert_mask_c.to(device=x_bcl.device).view(1, -1, 1)
                if self.channel_expert_mode == "delta":
                    gate_logits = gate_logits + ch_mask_bcp * gate_logits_ch
                else:
                    gate_logits = torch.where(ch_mask_bcp, gate_logits_ch, gate_logits)
            intervention_bcp = torch.sigmoid(gate_logits)
        else:
            intervention_bcp = torch.ones_like(route_bcp)
        if self.penalty_selector_enable:
            selector_feat = self._cluster_context_features(
                feat_bcd,
                cluster_id_c,
                self.selector_use_cluster_context,
            )
            Ws = self._stack_cluster_params(self.W_selector).index_select(0, cluster_id_c)
            bs = self._stack_cluster_params(self.b_selector).index_select(0, cluster_id_c)
            selector_logits = torch.einsum("bcd,cdp->bcp", selector_feat, Ws) + bs.unsqueeze(0)
            selector_bcp = torch.sigmoid(selector_logits / self.selector_temperature)
        else:
            selector_bcp = torch.ones_like(route_bcp)
        confidence_active_bcp = torch.ones_like(route_bcp)
        skip_confidence_bc = torch.zeros_like(route_bcp[..., 0])
        if bool(self.confidence_gate_enable):
            if int(self.confidence_threshold_kp.numel()) > 0:
                threshold_cp = self.confidence_threshold_kp.to(
                    device=x_bcl.device,
                    dtype=intervention_bcp.dtype,
                ).index_select(0, cluster_id_c)
                confidence_active_bcp = (intervention_bcp >= threshold_cp.unsqueeze(0)).to(dtype=route_bcp.dtype)
            if int(self.confidence_skip_threshold_k.numel()) > 0:
                selected_conf_bcp = torch.where(
                    route_bcp > 0.0,
                    intervention_bcp,
                    torch.zeros_like(intervention_bcp),
                )
                max_conf_bc = selected_conf_bcp.max(dim=-1).values
                skip_confidence_bc = 1.0 - max_conf_bc
                skip_threshold_c = self.confidence_skip_threshold_k.to(
                    device=x_bcl.device,
                    dtype=skip_confidence_bc.dtype,
                ).index_select(0, cluster_id_c)
                skip_active_bc = skip_confidence_bc >= skip_threshold_c.unsqueeze(0)
                confidence_active_bcp = confidence_active_bcp * (~skip_active_bc).unsqueeze(-1).to(dtype=route_bcp.dtype)
        route_gate_bcp = intervention_bcp * selector_bcp * confidence_active_bcp
        effective_route_bcph = None
        if route_bcph is not None:
            effective_route_bcph = route_bcph * route_gate_bcp.unsqueeze(-1)
            scale_bcph = effective_route_bcph * alpha_cp.unsqueeze(0).unsqueeze(-1)
            if int(self.patch_application_scale_p.numel()) > 0:
                scale_bcph = scale_bcph * self.patch_application_scale_p.to(
                    device=scale_bcph.device,
                    dtype=scale_bcph.dtype,
                ).view(1, 1, -1, 1)
            if int(self.patch_candidate_scale_c.numel()) > 0:
                scale_bcph = scale_bcph * self.patch_candidate_scale_c.to(
                    device=x_bcl.device,
                    dtype=scale_bcph.dtype,
                ).view(1, -1, 1, 1)
            branches = scale_bcph * residuals
            effective_route_bcp = effective_route_bcph.mean(dim=-1)
        else:
            effective_route_bcp = route_bcp * route_gate_bcp
            scale_bcp = effective_route_bcp * alpha_cp.unsqueeze(0)
            branches = scale_bcp.unsqueeze(-1) * residuals
        branch_sum_bch = branches.sum(dim=2)
        if self.fusion_gate_enable:
            fusion_feat = self._cluster_context_features(
                feat_bcd,
                cluster_id_c,
                self.fusion_use_cluster_context,
            )
            Wf = self._stack_cluster_params(self.W_fusion).index_select(0, cluster_id_c)
            bf = self._stack_cluster_params(self.b_fusion).index_select(0, cluster_id_c)
            fusion_bc = torch.sigmoid(torch.einsum("bcd,cd->bc", fusion_feat, Wf) + bf.unsqueeze(0))
        else:
            fusion_bc = torch.ones_like(route_bcp[..., 0])
        compositional_periodic_active = (
            patch_router_out is not None
            and self.patch_router is not None
            and self.patch_router.compositional_periodic_gate_enable
        )
        if compositional_periodic_active:
            periodic_expert_route_bch = patch_router_out[
                "patch_periodic_route_bch"
            ].to(device=y_base_bch.device, dtype=y_base_bch.dtype)
            periodic_expert_route_bch = (
                periodic_expert_route_bch
                * anchor_gate_weight_bc.unsqueeze(-1)
            )
            selected_base_bch = (
                y_base_bch
                + periodic_expert_route_bch * periodic_expert_branch_bch
            )
            periodic_expert_route_bc = periodic_expert_route_bch.mean(dim=-1)
        else:
            selected_base_bch = (
                y_base_bch
                + anchor_gate_weight_bc.unsqueeze(-1)
                * periodic_expert_branch_bch
            )
            periodic_expert_route_bc = anchor_gate_weight_bc
        y_final = (
            selected_base_bch
            + fusion_bc.unsqueeze(-1) * branch_sum_bch
            + ridge_gate_weight_bc.unsqueeze(-1)
            * position_daily_expert_branch_bch
        )

        result = {
            "y_final": y_final,
            "residuals": residuals,
            "branches": branches,
            "route_bcp": route_bcp,
            "intervention_bcp": intervention_bcp,
            "selector_bcp": selector_bcp,
            "confidence_active_bcp": confidence_active_bcp,
            "skip_confidence_bc": skip_confidence_bc,
            "effective_route_bcp": effective_route_bcp,
            "fusion_bc": fusion_bc,
            "alpha_cp": alpha_cp,
            "candidate_base_bch": candidate_base_bch,
            "periodic_expert_branch_bch": periodic_expert_branch_bch,
            "periodic_expert_route_bc": periodic_expert_route_bc,
            "position_daily_residual_expert_branch_bch": position_daily_expert_branch_bch,
            "anchor_ridge_gate_features_bcd": anchor_ridge_gate_features_bcd,
            "anchor_ridge_gate_weights_bc2": anchor_ridge_gate_weights_bc2,
        }
        if patch_router_out is not None:
            result.update(patch_router_out)
            result["effective_route_bcph"] = effective_route_bcph
            result["patch_candidate_scale_c"] = self.patch_candidate_scale_c
            result["patch_application_scale_p"] = self.patch_application_scale_p
            if compositional_periodic_active:
                result["selected_base_bch"] = selected_base_bch
                result["periodic_expert_route_bch"] = periodic_expert_route_bch
        return result

    def alpha_values(self) -> torch.Tensor:
        alpha = self.alpha_scale * torch.sigmoid(
            torch.stack(list(self.log_alpha), dim=0).reshape(self.param_K, self.P)
        )
        if self.shared_across_clusters and self.K != 1:
            return alpha.expand(self.K, self.P)
        return alpha

    def get_cluster_params(self, k: int) -> List[nn.Parameter]:
        if self.shared_across_clusters and int(k) != 0:
            return []
        params: List[nn.Parameter] = []
        for p in range(self.P):
            idx = self._idx(k, p)
            params.extend([
                self.W1[idx],
                self.b1[idx],
                self.W2[idx],
                self.b2[idx],
                self.log_alpha[idx],
                self.W_gate[idx],
                self.b_gate[idx],
            ])
        if self.penalty_selector_enable:
            idx = self._param_cluster(k)
            params.extend([self.W_selector[idx], self.b_selector[idx]])
        if self.fusion_gate_enable:
            idx = self._param_cluster(k)
            params.extend([self.W_fusion[idx], self.b_fusion[idx]])
        if self.patch_router is not None:
            params.extend(list(self.patch_router.parameters()))
        if self.channel_expert_enable and self.channel_expert_cluster_id_c.numel() > 0:
            idx = ((self.channel_expert_cluster_id_c == int(k)) & self.channel_expert_mask_c).nonzero(
                as_tuple=False
            ).view(-1)
            for c_t in idx:
                c = int(c_t.item())
                for p in range(self.P):
                    ch_idx = self._ch_idx(c, p)
                    params.extend([
                        self.channel_W1[ch_idx],
                        self.channel_b1[ch_idx],
                        self.channel_W2[ch_idx],
                        self.channel_b2[ch_idx],
                        self.channel_log_alpha[ch_idx],
                        self.channel_W_gate[ch_idx],
                        self.channel_b_gate[ch_idx],
                    ])
        return params

    def mask_cluster_grads(self, stopped_k: torch.Tensor):
        if self.shared_across_clusters:
            if not bool(stopped_k.detach().to(dtype=torch.bool).all().item()):
                return
            stopped_k = torch.ones((self.param_K,), dtype=torch.bool, device=stopped_k.device)
        for k in range(self.K):
            if not bool(stopped_k[k].item()):
                continue
            for param in self.get_cluster_params(k):
                if param.grad is not None:
                    param.grad.zero_()
            if self.shared_across_clusters:
                break

    def get_cluster_state(self, k: int) -> Dict[str, torch.Tensor]:
        state = {
            "W1": torch.stack([self.W1[self._idx(k, p)].detach().cpu() for p in range(self.P)], dim=0),
            "b1": torch.stack([self.b1[self._idx(k, p)].detach().cpu() for p in range(self.P)], dim=0),
            "W2": torch.stack([self.W2[self._idx(k, p)].detach().cpu() for p in range(self.P)], dim=0),
            "b2": torch.stack([self.b2[self._idx(k, p)].detach().cpu() for p in range(self.P)], dim=0),
            "log_alpha": torch.stack([self.log_alpha[self._idx(k, p)].detach().cpu() for p in range(self.P)], dim=0),
            "W_gate": torch.stack([self.W_gate[self._idx(k, p)].detach().cpu() for p in range(self.P)], dim=0),
            "b_gate": torch.stack([self.b_gate[self._idx(k, p)].detach().cpu() for p in range(self.P)], dim=0),
        }
        if self.channel_expert_enable and self.channel_expert_cluster_id_c.numel() > 0:
            idx = ((self.channel_expert_cluster_id_c == int(k)) & self.channel_expert_mask_c).nonzero(
                as_tuple=False
            ).view(-1)
            state["channel_idx"] = idx.detach().cpu()
            if idx.numel() > 0:
                state["channel_W1"] = torch.stack([
                    torch.stack([self.channel_W1[self._ch_idx(int(c.item()), p)].detach().cpu() for p in range(self.P)], dim=0)
                    for c in idx
                ], dim=0)
                state["channel_b1"] = torch.stack([
                    torch.stack([self.channel_b1[self._ch_idx(int(c.item()), p)].detach().cpu() for p in range(self.P)], dim=0)
                    for c in idx
                ], dim=0)
                state["channel_W2"] = torch.stack([
                    torch.stack([self.channel_W2[self._ch_idx(int(c.item()), p)].detach().cpu() for p in range(self.P)], dim=0)
                    for c in idx
                ], dim=0)
                state["channel_b2"] = torch.stack([
                    torch.stack([self.channel_b2[self._ch_idx(int(c.item()), p)].detach().cpu() for p in range(self.P)], dim=0)
                    for c in idx
                ], dim=0)
                state["channel_log_alpha"] = torch.stack([
                    torch.stack([self.channel_log_alpha[self._ch_idx(int(c.item()), p)].detach().cpu() for p in range(self.P)], dim=0)
                    for c in idx
                ], dim=0)
                state["channel_W_gate"] = torch.stack([
                    torch.stack([self.channel_W_gate[self._ch_idx(int(c.item()), p)].detach().cpu() for p in range(self.P)], dim=0)
                    for c in idx
                ], dim=0)
                state["channel_b_gate"] = torch.stack([
                    torch.stack([self.channel_b_gate[self._ch_idx(int(c.item()), p)].detach().cpu() for p in range(self.P)], dim=0)
                    for c in idx
                ], dim=0)
            else:
                state["channel_W1"] = torch.empty(0, self.P, self.input_dim, self.hidden_dim)
                state["channel_b1"] = torch.empty(0, self.P, self.hidden_dim)
                state["channel_W2"] = torch.empty(0, self.P, self.hidden_dim, self.H)
                state["channel_b2"] = torch.empty(0, self.P, self.H)
                state["channel_log_alpha"] = torch.empty(0, self.P)
                state["channel_W_gate"] = torch.empty(0, self.P, self.hidden_dim)
                state["channel_b_gate"] = torch.empty(0, self.P)
        if self.penalty_selector_enable:
            idx = self._param_cluster(k)
            state["W_selector"] = self.W_selector[idx].detach().cpu()
            state["b_selector"] = self.b_selector[idx].detach().cpu()
        if self.fusion_gate_enable:
            idx = self._param_cluster(k)
            state["W_fusion"] = self.W_fusion[idx].detach().cpu()
            state["b_fusion"] = self.b_fusion[idx].detach().cpu()
        if self.patch_router is not None:
            for name, param in self.patch_router.named_parameters():
                state[f"patch_router.{name}"] = param.detach().cpu().clone()
        return state

    def load_cluster_state(self, k: int, state: Dict[str, torch.Tensor]):
        for p in range(self.P):
            idx = self._idx(k, p)
            device = self.W1[idx].device
            self.W1[idx].data.copy_(state["W1"][p].to(device))
            self.b1[idx].data.copy_(state["b1"][p].to(device))
            self.W2[idx].data.copy_(state["W2"][p].to(device))
            self.b2[idx].data.copy_(state["b2"][p].to(device))
            self.log_alpha[idx].data.copy_(state["log_alpha"][p].to(device))
            if "W_gate" in state:
                self.W_gate[idx].data.copy_(state["W_gate"][p].to(device))
            if "b_gate" in state:
                self.b_gate[idx].data.copy_(state["b_gate"][p].to(device))
        if self.penalty_selector_enable and "W_selector" in state:
            idx = self._param_cluster(k)
            self.W_selector[idx].data.copy_(state["W_selector"].to(self.W_selector[idx].device))
        if self.penalty_selector_enable and "b_selector" in state:
            idx = self._param_cluster(k)
            self.b_selector[idx].data.copy_(state["b_selector"].to(self.b_selector[idx].device))
        if self.fusion_gate_enable and "W_fusion" in state:
            idx = self._param_cluster(k)
            self.W_fusion[idx].data.copy_(state["W_fusion"].to(self.W_fusion[idx].device))
        if self.fusion_gate_enable and "b_fusion" in state:
            idx = self._param_cluster(k)
            self.b_fusion[idx].data.copy_(state["b_fusion"].to(self.b_fusion[idx].device))
        if self.patch_router is not None:
            for name, param in self.patch_router.named_parameters():
                key = f"patch_router.{name}"
                if key in state:
                    param.data.copy_(state[key].to(param.device))
        if self.channel_expert_enable and "channel_idx" in state:
            saved_idx = state["channel_idx"].detach().cpu().to(dtype=torch.long)
            current_idx = ((self.channel_expert_cluster_id_c == int(k)) & self.channel_expert_mask_c).nonzero(
                as_tuple=False
            ).view(-1).detach().cpu()
            if saved_idx.numel() != current_idx.numel() or not torch.equal(saved_idx, current_idx):
                raise ValueError(f"channel expert adapter cluster {k} channel indices do not match checkpoint state.")
            for j, c_t in enumerate(current_idx):
                c = int(c_t.item())
                for p in range(self.P):
                    ch_idx = self._ch_idx(c, p)
                    self.channel_W1[ch_idx].data.copy_(state["channel_W1"][j, p].to(self.channel_W1[ch_idx].device))
                    self.channel_b1[ch_idx].data.copy_(state["channel_b1"][j, p].to(self.channel_b1[ch_idx].device))
                    self.channel_W2[ch_idx].data.copy_(state["channel_W2"][j, p].to(self.channel_W2[ch_idx].device))
                    self.channel_b2[ch_idx].data.copy_(state["channel_b2"][j, p].to(self.channel_b2[ch_idx].device))
                    self.channel_log_alpha[ch_idx].data.copy_(
                        state["channel_log_alpha"][j, p].to(self.channel_log_alpha[ch_idx].device)
                    )
                    if "channel_W_gate" in state:
                        self.channel_W_gate[ch_idx].data.copy_(
                            state["channel_W_gate"][j, p].to(self.channel_W_gate[ch_idx].device)
                        )
                    if "channel_b_gate" in state:
                        self.channel_b_gate[ch_idx].data.copy_(
                            state["channel_b_gate"][j, p].to(self.channel_b_gate[ch_idx].device)
                        )
