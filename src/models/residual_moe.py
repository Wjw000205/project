from typing import Dict, List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from ..utils.cluster_memory import scatter_mean_bcl_to_bkl


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
    ):
        super().__init__()
        self.K = int(num_clusters)
        self.P = int(num_penalties)
        self.L = int(input_len)
        self.H = int(pred_len)
        self.hidden_dim = int(hidden_dim)
        self.alpha_scale = float(alpha_scale)
        self.residual_clip = float(max(0.0, residual_clip))
        self.use_y_base_input = bool(use_y_base_input)
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
        anchor_name_set = {str(name) for name in (seasonal_anchor_names or [])}
        self.seasonal_anchor_period = max(int(seasonal_anchor_period), 1)
        self.seasonal_anchor_num_periods = max(int(seasonal_anchor_num_periods), 1)
        self.seasonal_anchor_scale = float(seasonal_anchor_scale)
        seasonal_mask = torch.tensor(
            [name in anchor_name_set for name in self.penalty_names],
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
        self.input_dim = int(input_dim)
        self.selector_input_dim = self.input_dim * (3 if self.selector_use_cluster_context else 1)
        self.fusion_input_dim = self.input_dim * (3 if self.fusion_use_cluster_context else 1)

        # Use one real Parameter object per (cluster, penalty) expert. Keeping
        # penalty experts physically separate makes gradient isolation and
        # diagnostics unambiguous.
        num_experts = self.K * self.P
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
                [nn.Parameter(torch.empty(self.selector_input_dim, self.P)) for _ in range(self.K)]
            )
            self.b_selector = nn.ParameterList(
                [nn.Parameter(torch.zeros(self.P)) for _ in range(self.K)]
            )
        else:
            self.W_selector = nn.ParameterList()
            self.b_selector = nn.ParameterList()
        if self.fusion_gate_enable:
            self.W_fusion = nn.ParameterList(
                [nn.Parameter(torch.empty(self.fusion_input_dim)) for _ in range(self.K)]
            )
            self.b_fusion = nn.ParameterList(
                [nn.Parameter(torch.tensor(float(fusion_init))) for _ in range(self.K)]
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
        self.register_buffer("seasonal_anchor_index_hp", seasonal_index, persistent=False)
        self.register_buffer("seasonal_anchor_valid_hp", seasonal_valid, persistent=False)
        self.reset_parameters()

    def _idx(self, k: int, p: int) -> int:
        return int(k) * self.P + int(p)

    def _ch_idx(self, c: int, p: int) -> int:
        return int(c) * self.P + int(p)

    def set_channel_penalty_allowed_mask(self, mask_cp: Optional[torch.Tensor]) -> None:
        if mask_cp is None:
            self.channel_penalty_allowed_mask_cp = torch.empty(0, device=self.channel_penalty_allowed_mask_cp.device)
            return
        if mask_cp.ndim != 2 or int(mask_cp.shape[1]) != self.P:
            raise ValueError(
                f"channel penalty mask must have shape [C,{self.P}], got {tuple(mask_cp.shape)}"
            )
        self.channel_penalty_allowed_mask_cp = mask_cp.detach().to(dtype=torch.float32)

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
            return self._safe_augmented_features(x_bcl, y_base_bch)
        if not self.use_y_base_input:
            return x_centered
        y_centered = y_base_bch - last
        return torch.cat([x_centered, y_centered], dim=-1)

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
        if self.P <= 0:
            zero_res = y_base_bch.new_zeros((*y_base_bch.shape[:2], 0, y_base_bch.shape[-1]))
            zero_route = y_base_bch.new_zeros((*y_base_bch.shape[:2], 0))
            return {
                "y_final": y_base_bch,
                "residuals": zero_res,
                "branches": zero_res,
                "route_bcp": zero_route,
                "intervention_bcp": zero_route,
                "effective_route_bcp": zero_route,
                "alpha_cp": y_base_bch.new_zeros((y_base_bch.shape[1], 0)),
            }

        feat_bcd = self._input_features(x_bcl, y_base_bch)
        cluster_id_c = cluster_id_c.to(device=x_bcl.device, dtype=torch.long)

        W1_kpdm = torch.stack(list(self.W1), dim=0).reshape(self.K, self.P, self.input_dim, self.hidden_dim)
        b1_kpm = torch.stack(list(self.b1), dim=0).reshape(self.K, self.P, self.hidden_dim)
        W2_kpmh = torch.stack(list(self.W2), dim=0).reshape(self.K, self.P, self.hidden_dim, self.H)
        b2_kph = torch.stack(list(self.b2), dim=0).reshape(self.K, self.P, self.H)
        Wg_kpm = torch.stack(list(self.W_gate), dim=0).reshape(self.K, self.P, self.hidden_dim)
        bg_kp = torch.stack(list(self.b_gate), dim=0).reshape(self.K, self.P)
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
            anchor_residual = seasonal_anchor - y_base_bch
            mask_p = self.seasonal_anchor_mask_p.to(device=x_bcl.device, dtype=residuals.dtype)
            residuals = residuals + (
                float(self.seasonal_anchor_scale)
                * mask_p.view(1, 1, self.P, 1)
                * anchor_residual.unsqueeze(2)
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

        alpha_cp = self.alpha_values().index_select(0, cluster_id_c)  # [C,P]
        if self.channel_expert_enable:
            ch_alpha_cp = self.alpha_scale * torch.sigmoid(
                torch.stack(list(self.channel_log_alpha), dim=0).reshape(self.C_channel, self.P)
            )
            ch_mask_cp = self.channel_expert_mask_c.to(device=x_bcl.device).view(-1, 1)
            alpha_cp = torch.where(ch_mask_cp, ch_alpha_cp, alpha_cp)
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
            Ws = torch.stack(list(self.W_selector), dim=0).index_select(0, cluster_id_c)
            bs = torch.stack(list(self.b_selector), dim=0).index_select(0, cluster_id_c)
            selector_logits = torch.einsum("bcd,cdp->bcp", selector_feat, Ws) + bs.unsqueeze(0)
            selector_bcp = torch.sigmoid(selector_logits / self.selector_temperature)
        else:
            selector_bcp = torch.ones_like(route_bcp)
        effective_route_bcp = route_bcp * intervention_bcp * selector_bcp
        scale_bcp = effective_route_bcp * alpha_cp.unsqueeze(0)
        branches = scale_bcp.unsqueeze(-1) * residuals
        branch_sum_bch = branches.sum(dim=2)
        if self.fusion_gate_enable:
            fusion_feat = self._cluster_context_features(
                feat_bcd,
                cluster_id_c,
                self.fusion_use_cluster_context,
            )
            Wf = torch.stack(list(self.W_fusion), dim=0).index_select(0, cluster_id_c)
            bf = torch.stack(list(self.b_fusion), dim=0).index_select(0, cluster_id_c)
            fusion_bc = torch.sigmoid(torch.einsum("bcd,cd->bc", fusion_feat, Wf) + bf.unsqueeze(0))
        else:
            fusion_bc = torch.ones_like(route_bcp[..., 0])
        y_final = y_base_bch + fusion_bc.unsqueeze(-1) * branch_sum_bch

        return {
            "y_final": y_final,
            "residuals": residuals,
            "branches": branches,
            "route_bcp": route_bcp,
            "intervention_bcp": intervention_bcp,
            "selector_bcp": selector_bcp,
            "effective_route_bcp": effective_route_bcp,
            "fusion_bc": fusion_bc,
            "alpha_cp": alpha_cp,
        }

    def alpha_values(self) -> torch.Tensor:
        return self.alpha_scale * torch.sigmoid(torch.stack(list(self.log_alpha), dim=0).reshape(self.K, self.P))

    def get_cluster_params(self, k: int) -> List[nn.Parameter]:
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
            params.extend([self.W_selector[k], self.b_selector[k]])
        if self.fusion_gate_enable:
            params.extend([self.W_fusion[k], self.b_fusion[k]])
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
        for k in range(self.K):
            if not bool(stopped_k[k].item()):
                continue
            for param in self.get_cluster_params(k):
                if param.grad is not None:
                    param.grad.zero_()

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
            state["W_selector"] = self.W_selector[k].detach().cpu()
            state["b_selector"] = self.b_selector[k].detach().cpu()
        if self.fusion_gate_enable:
            state["W_fusion"] = self.W_fusion[k].detach().cpu()
            state["b_fusion"] = self.b_fusion[k].detach().cpu()
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
            self.W_selector[k].data.copy_(state["W_selector"].to(self.W_selector[k].device))
        if self.penalty_selector_enable and "b_selector" in state:
            self.b_selector[k].data.copy_(state["b_selector"].to(self.b_selector[k].device))
        if self.fusion_gate_enable and "W_fusion" in state:
            self.W_fusion[k].data.copy_(state["W_fusion"].to(self.W_fusion[k].device))
        if self.fusion_gate_enable and "b_fusion" in state:
            self.b_fusion[k].data.copy_(state["b_fusion"].to(self.b_fusion[k].device))
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
