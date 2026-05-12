from typing import Dict, Optional
import math
import torch
from torch import nn
from torch.nn import functional as F


class ClusterwiseDynamicLambda(nn.Module):
    """
    Learn an input-conditioned multiplicative scale for lambda.

    Modes:
    - mlp: legacy feature-only scaling
    - multiscale: lightweight multi-scale statistics + MLP
    - liquid: liquid-style recurrent scan over per-cluster input windows
    """

    def __init__(
        self,
        num_clusters: int,
        feat_dim: int,
        num_penalties: int,
        hidden_dim: int = 32,
        max_factor: float = 2.0,
        dropout: float = 0.0,
        mode: str = "multiscale",
        mix: float = 0.6,
        tau_min: float = 1.0,
        tau_max: float = 6.0,
        series_downsample_len: int = 32,
        segment_bins = (4, 8),
    ):
        super().__init__()
        self.K = num_clusters
        self.F = feat_dim
        self.P = num_penalties
        self.H = hidden_dim
        self.mode = str(mode).lower()
        self.segment_bins = tuple(max(int(v), 1) for v in segment_bins)
        self.max_factor = max(float(max_factor), 1.0)
        self.log_max_factor = math.log(self.max_factor)
        self.mix = float(max(0.0, min(mix, 1.0)))
        self.tau_min = max(float(tau_min), 1.0)
        self.tau_max = max(float(tau_max), self.tau_min)
        self.series_downsample_len = max(int(series_downsample_len), 1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._cluster_param_lists: Dict[str, nn.ParameterList] = {}

        if self.mode == "mlp":
            self._build_projector(self.F)
        elif self.mode == "multiscale":
            self._build_projector(self._multiscale_input_dim())
        elif self.mode == "liquid":
            self._build_liquid()
        else:
            raise ValueError(f"Unsupported dynamic lambda mode: {self.mode}")

        self.reset_parameters()

    def _register_cluster_params(self, name: str, params: nn.ParameterList):
        setattr(self, name, params)
        self._cluster_param_lists[name] = params

    def _build_projector(self, input_dim: int):
        self._register_cluster_params(
            "W1",
            nn.ParameterList([nn.Parameter(torch.empty(input_dim, self.H)) for _ in range(self.K)]),
        )
        self._register_cluster_params(
            "b1",
            nn.ParameterList([nn.Parameter(torch.zeros(self.H)) for _ in range(self.K)]),
        )
        self._register_cluster_params(
            "W2",
            nn.ParameterList([nn.Parameter(torch.zeros(self.H, self.P)) for _ in range(self.K)]),
        )
        self._register_cluster_params(
            "b2",
            nn.ParameterList([nn.Parameter(torch.zeros(self.P)) for _ in range(self.K)]),
        )

    def _multiscale_input_dim(self) -> int:
        series_feat_dim = 7 + sum((2 * bins) - 1 for bins in self.segment_bins)
        return self.F + series_feat_dim

    def _build_liquid(self):
        self._register_cluster_params(
            "W_ctx",
            nn.ParameterList([nn.Parameter(torch.empty(self.F, self.H)) for _ in range(self.K)]),
        )
        self._register_cluster_params(
            "b_ctx",
            nn.ParameterList([nn.Parameter(torch.zeros(self.H)) for _ in range(self.K)]),
        )
        self._register_cluster_params(
            "W_in",
            nn.ParameterList([nn.Parameter(torch.empty(self.H)) for _ in range(self.K)]),
        )
        self._register_cluster_params(
            "W_h",
            nn.ParameterList([nn.Parameter(torch.empty(self.H, self.H)) for _ in range(self.K)]),
        )
        self._register_cluster_params(
            "b_h",
            nn.ParameterList([nn.Parameter(torch.zeros(self.H)) for _ in range(self.K)]),
        )
        self._register_cluster_params(
            "W_tau_ctx",
            nn.ParameterList([nn.Parameter(torch.empty(self.F, self.H)) for _ in range(self.K)]),
        )
        self._register_cluster_params(
            "W_tau_in",
            nn.ParameterList([nn.Parameter(torch.empty(self.H)) for _ in range(self.K)]),
        )
        self._register_cluster_params(
            "b_tau",
            nn.ParameterList([nn.Parameter(torch.zeros(self.H)) for _ in range(self.K)]),
        )
        self._register_cluster_params(
            "W_out",
            nn.ParameterList([nn.Parameter(torch.zeros(self.H, self.P)) for _ in range(self.K)]),
        )
        self._register_cluster_params(
            "b_out",
            nn.ParameterList([nn.Parameter(torch.zeros(self.P)) for _ in range(self.K)]),
        )

    def reset_parameters(self):
        if self.mode in {"mlp", "multiscale"}:
            for w in self.W1:
                nn.init.xavier_uniform_(w)
            return

        for w in self.W_ctx:
            nn.init.xavier_uniform_(w)
        for w in self.W_h:
            nn.init.orthogonal_(w)
        for w in self.W_tau_ctx:
            nn.init.xavier_uniform_(w)
        for w in self.W_in:
            nn.init.normal_(w, mean=0.0, std=0.1)
        for w in self.W_tau_in:
            nn.init.normal_(w, mean=0.0, std=0.1)

    def _forward_mlp(self, feat_bkf: torch.Tensor) -> torch.Tensor:
        W1 = torch.stack(list(self.W1), dim=0)
        b1 = torch.stack(list(self.b1), dim=0)
        W2 = torch.stack(list(self.W2), dim=0)
        b2 = torch.stack(list(self.b2), dim=0)

        h = torch.einsum("bkf,kfh->bkh", feat_bkf, W1) + b1.unsqueeze(0)
        h = self.drop(self.act(h))
        raw = torch.einsum("bkh,khp->bkp", h, W2) + b2.unsqueeze(0)
        dyn_scale = torch.exp(self.log_max_factor * torch.tanh(raw))
        return (1.0 - self.mix) + self.mix * dyn_scale

    def _extract_multiscale_series_features(
        self,
        feat_bkf: torch.Tensor,
        series_bkl: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, K = feat_bkf.shape[:2]
        extra_dim = self._multiscale_input_dim() - self.F
        if series_bkl is None or series_bkl.shape[-1] == 0:
            return feat_bkf.new_zeros(B, K, extra_dim)

        x = series_bkl.to(dtype=feat_bkf.dtype)
        L = x.shape[-1]
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(1.0e-6)
        first = x[..., :1]
        last = x[..., -1:]
        value_range = x.amax(dim=-1, keepdim=True) - x.amin(dim=-1, keepdim=True)

        if L > 1:
            diff = x[..., 1:] - x[..., :-1]
            diff_abs_mean = diff.abs().mean(dim=-1, keepdim=True)
        else:
            diff_abs_mean = torch.zeros_like(mean)

        if L > 2:
            second_diff = x[..., 2:] - (2.0 * x[..., 1:-1]) + x[..., :-2]
            curvature_abs_mean = second_diff.abs().mean(dim=-1, keepdim=True)
        else:
            curvature_abs_mean = torch.zeros_like(mean)

        parts = [
            mean,
            std,
            (last - mean) / std,
            (last - first) / std,
            value_range / std,
            diff_abs_mean / std,
            curvature_abs_mean / std,
        ]

        for bins in self.segment_bins:
            pooled = F.adaptive_avg_pool1d(
                x.reshape(-1, 1, L),
                bins,
            ).reshape(B, K, bins)
            pooled = pooled - pooled.mean(dim=-1, keepdim=True)
            pooled = pooled / pooled.std(dim=-1, keepdim=True).clamp_min(1.0e-6)
            parts.append(pooled)
            if bins > 1:
                parts.append(pooled[..., 1:] - pooled[..., :-1])

        return torch.cat(parts, dim=-1)

    def _forward_multiscale(
        self,
        feat_bkf: torch.Tensor,
        series_bkl: Optional[torch.Tensor],
    ) -> torch.Tensor:
        W1 = torch.stack(list(self.W1), dim=0)
        b1 = torch.stack(list(self.b1), dim=0)
        W2 = torch.stack(list(self.W2), dim=0)
        b2 = torch.stack(list(self.b2), dim=0)

        series_feat = self._extract_multiscale_series_features(feat_bkf, series_bkl)
        combined = torch.cat([feat_bkf, series_feat], dim=-1)
        h = torch.einsum("bkf,kfh->bkh", combined, W1) + b1.unsqueeze(0)
        h = self.drop(self.act(h))
        raw = torch.einsum("bkh,khp->bkp", h, W2) + b2.unsqueeze(0)
        dyn_scale = torch.exp(self.log_max_factor * torch.tanh(raw))
        return (1.0 - self.mix) + self.mix * dyn_scale

    def _forward_liquid(self, feat_bkf: torch.Tensor, series_bkl: Optional[torch.Tensor]) -> torch.Tensor:
        W_ctx = torch.stack(list(self.W_ctx), dim=0)
        b_ctx = torch.stack(list(self.b_ctx), dim=0)
        W_in = torch.stack(list(self.W_in), dim=0)
        W_h = torch.stack(list(self.W_h), dim=0)
        b_h = torch.stack(list(self.b_h), dim=0)
        W_tau_ctx = torch.stack(list(self.W_tau_ctx), dim=0)
        W_tau_in = torch.stack(list(self.W_tau_in), dim=0)
        b_tau = torch.stack(list(self.b_tau), dim=0)
        W_out = torch.stack(list(self.W_out), dim=0)
        b_out = torch.stack(list(self.b_out), dim=0)

        ctx = torch.einsum("bkf,kfh->bkh", feat_bkf, W_ctx) + b_ctx.unsqueeze(0)
        ctx = self.drop(self.act(ctx))
        h = torch.tanh(ctx)

        if series_bkl is not None and series_bkl.shape[-1] > 0:
            x = series_bkl.to(dtype=feat_bkf.dtype)
            if x.shape[-1] > self.series_downsample_len:
                x = F.adaptive_avg_pool1d(
                    x.reshape(-1, 1, x.shape[-1]),
                    self.series_downsample_len,
                ).reshape(x.shape[0], x.shape[1], self.series_downsample_len)
            x = x - x.mean(dim=-1, keepdim=True)
            x = x / x.std(dim=-1, keepdim=True).clamp_min(1.0e-6)
            tau_base = torch.einsum("bkf,kfh->bkh", feat_bkf, W_tau_ctx) + b_tau.unsqueeze(0)

            # 预计算循环内与时间步无关的项，避免在 T 步中重复计算：
            # x_contrib_all[b,k,t,h] = x[b,k,t] * W_in[k,h]，形状 [B,K,T,H]
            # tau_logits_all[b,k,t,h] = tau_base[b,k,h] + x[b,k,t]*W_tau_in[k,h]，形状 [B,K,T,H]
            x_contrib_all = x.unsqueeze(-1) * W_in.unsqueeze(0).unsqueeze(2)   # [B,K,T,H]
            tau_logits_all = (
                tau_base.unsqueeze(2)
                + x.unsqueeze(-1) * W_tau_in.unsqueeze(0).unsqueeze(2)
            )  # [B,K,T,H]
            static_cand = ctx + b_h.unsqueeze(0)  # [B,K,H]，仅依赖输入特征

            tau_range = self.tau_max - self.tau_min
            for t in range(x.shape[-1]):
                cand = torch.tanh(
                    static_cand
                    + x_contrib_all[:, :, t]
                    + torch.einsum("bkh,khd->bkd", h, W_h)
                )
                tau = self.tau_min + tau_range * torch.sigmoid(tau_logits_all[:, :, t])
                h = h + (cand - h) / tau

        h = self.drop(h)
        raw = torch.einsum("bkh,khp->bkp", h, W_out) + b_out.unsqueeze(0)
        dyn_scale = torch.exp(self.log_max_factor * torch.tanh(raw))
        return (1.0 - self.mix) + self.mix * dyn_scale

    def forward(self, feat_bkf: torch.Tensor, series_bkl: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.mode == "mlp":
            return self._forward_mlp(feat_bkf)
        if self.mode == "multiscale":
            return self._forward_multiscale(feat_bkf, series_bkl)
        return self._forward_liquid(feat_bkf, series_bkl)

    def get_cluster_params(self, k: int):
        return [plist[k] for plist in self._cluster_param_lists.values()]

    def mask_cluster_grads(self, stopped_k: torch.Tensor):
        for k in range(self.K):
            if not bool(stopped_k[k].item()):
                continue
            for plist in self._cluster_param_lists.values():
                if plist[k].grad is not None:
                    plist[k].grad.zero_()

    def get_cluster_state(self, k: int) -> Dict[str, torch.Tensor]:
        return {
            name: plist[k].detach().cpu()
            for name, plist in self._cluster_param_lists.items()
        }

    def load_cluster_state(self, k: int, state: Dict[str, torch.Tensor]):
        for name, plist in self._cluster_param_lists.items():
            if name not in state:
                continue
            device = plist[k].device
            plist[k].data.copy_(state[name].to(device))
