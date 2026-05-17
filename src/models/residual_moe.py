from typing import Dict, List, Optional

import torch
from torch import nn
from torch.nn import functional as F


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
        if self.feature_mode == "legacy":
            input_dim = self.L + (self.H if self.use_y_base_input else 0)
        else:
            input_dim = self.L + self.H + 10 + (2 * self.H if self.use_y_base_input else 0)
        self.input_dim = int(input_dim)

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
        self.reset_parameters()

    def _idx(self, k: int, p: int) -> int:
        return int(k) * self.P + int(p)

    def reset_parameters(self):
        for w in self.W1:
            nn.init.xavier_uniform_(w)
        for w in self.W2:
            nn.init.zeros_(w)
        for b in self.b2:
            nn.init.zeros_(b)
        for w in self.W_gate:
            nn.init.zeros_(w)

    def _history_proxy_forecast(self, x_bcl: torch.Tensor) -> torch.Tensor:
        if self.L >= self.H:
            return x_bcl[..., -self.H:]
        pad = x_bcl[..., -1:].expand(*x_bcl.shape[:-1], self.H - self.L)
        return torch.cat([x_bcl, pad], dim=-1)

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
        if self.residual_clip > 0.0:
            clip = float(self.residual_clip)
            residuals = clip * torch.tanh(residuals / clip)

        alpha_cp = self.alpha_values().index_select(0, cluster_id_c)  # [C,P]
        route_bcp = mask_bkp[:, cluster_id_c, :]
        if skip_bk is not None:
            route_bcp = route_bcp * (1.0 - skip_bk[:, cluster_id_c].unsqueeze(-1))
        if self.intervention_enable:
            gate_logits = torch.einsum("bcpm,cpm->bcp", h, Wg) + bg.unsqueeze(0)
            intervention_bcp = torch.sigmoid(gate_logits)
        else:
            intervention_bcp = torch.ones_like(route_bcp)
        effective_route_bcp = route_bcp * intervention_bcp
        scale_bcp = effective_route_bcp * alpha_cp.unsqueeze(0)
        branches = scale_bcp.unsqueeze(-1) * residuals
        y_final = y_base_bch + branches.sum(dim=2)

        return {
            "y_final": y_final,
            "residuals": residuals,
            "branches": branches,
            "route_bcp": route_bcp,
            "intervention_bcp": intervention_bcp,
            "effective_route_bcp": effective_route_bcp,
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
        return params

    def mask_cluster_grads(self, stopped_k: torch.Tensor):
        for k in range(self.K):
            if not bool(stopped_k[k].item()):
                continue
            for param in self.get_cluster_params(k):
                if param.grad is not None:
                    param.grad.zero_()

    def get_cluster_state(self, k: int) -> Dict[str, torch.Tensor]:
        return {
            "W1": torch.stack([self.W1[self._idx(k, p)].detach().cpu() for p in range(self.P)], dim=0),
            "b1": torch.stack([self.b1[self._idx(k, p)].detach().cpu() for p in range(self.P)], dim=0),
            "W2": torch.stack([self.W2[self._idx(k, p)].detach().cpu() for p in range(self.P)], dim=0),
            "b2": torch.stack([self.b2[self._idx(k, p)].detach().cpu() for p in range(self.P)], dim=0),
            "log_alpha": torch.stack([self.log_alpha[self._idx(k, p)].detach().cpu() for p in range(self.P)], dim=0),
            "W_gate": torch.stack([self.W_gate[self._idx(k, p)].detach().cpu() for p in range(self.P)], dim=0),
            "b_gate": torch.stack([self.b_gate[self._idx(k, p)].detach().cpu() for p in range(self.P)], dim=0),
        }

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
