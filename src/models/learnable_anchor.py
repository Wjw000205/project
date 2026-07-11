from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn


class ClusterwiseLearnableOutputAnchor(nn.Module):
    """
    Learn a small cluster/channel/horizon correction on top of static output anchors.

    Static anchor tables remain fixed train-derived priors. This module learns
    bounded deltas for the stat-anchor contribution, residual-anchor contribution,
    and an optional direct bias. Zero init is exactly equivalent to the static
    anchor path.
    """

    def __init__(
        self,
        *,
        num_clusters: int,
        num_channels: int,
        pred_len: int,
        cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()
        cfg = cfg or {}
        self.K = int(num_clusters)
        self.C = int(num_channels)
        self.H = int(pred_len)
        if self.K <= 0:
            raise ValueError("num_clusters must be positive for learnable_output_anchor.")
        if self.C <= 0:
            raise ValueError("num_channels must be positive for learnable_output_anchor.")
        if self.H <= 0:
            raise ValueError("pred_len must be positive for learnable_output_anchor.")

        self.max_scale_delta = float(cfg.get("max_scale_delta", 0.5))
        self.learn_stat_scale = bool(cfg.get("learn_stat_scale", True))
        self.learn_residual_scale = bool(cfg.get("learn_residual_scale", True))
        self.learn_bias = bool(cfg.get("learn_bias", False))
        self.max_bias = float(cfg.get("max_bias", 0.0))
        self.learn_history_trend = bool(
            cfg.get("learn_history_trend", cfg.get("history_trend_enable", False))
        )
        self.max_history_trend_delta = float(
            cfg.get("max_history_trend_delta", cfg.get("history_trend_max_delta", 0.1))
        )
        self.history_trend_window = max(
            0,
            int(cfg.get("history_trend_window", cfg.get("history_window", 0))),
        )
        self.history_trend_feature = self._normalize_history_trend_feature(
            str(cfg.get("history_trend_feature", "last_minus_mean"))
        )
        self.history_trend_projection = self._normalize_history_trend_projection(
            str(cfg.get("history_trend_projection", "linear"))
        )
        self.scale_temporal_basis_rank = max(
            0,
            int(
                cfg.get(
                    "scale_temporal_basis_rank",
                    cfg.get("temporal_basis_rank", cfg.get("temporal_rank", 0)),
                )
            ),
        )
        self.scale_parameterization = self._normalize_parameterization(
            str(cfg.get("scale_parameterization", cfg.get("parameterization", "channel")))
        )
        self.bias_parameterization = self._normalize_parameterization(
            str(cfg.get("bias_parameterization", self.scale_parameterization))
        )
        self.history_trend_parameterization = self._normalize_parameterization(
            str(cfg.get("history_trend_parameterization", self.scale_parameterization))
        )
        self.scale_shape = self._shape_for_parameterization(self.scale_parameterization)
        self.bias_shape = self._shape_for_parameterization(self.bias_parameterization)
        self.history_trend_shape = self._shape_for_parameterization(self.history_trend_parameterization)

        self.stat_scale_delta_raw = nn.ParameterList(
            [nn.Parameter(torch.zeros(*self.scale_shape)) for _ in range(self.K)]
        )
        self.residual_scale_delta_raw = nn.ParameterList(
            [nn.Parameter(torch.zeros(*self.scale_shape)) for _ in range(self.K)]
        )
        self.stat_scale_temporal_coef_raw = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.scale_shape[0], self.scale_temporal_basis_rank)) for _ in range(self.K)]
        )
        self.residual_scale_temporal_coef_raw = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.scale_shape[0], self.scale_temporal_basis_rank)) for _ in range(self.K)]
        )
        self.bias_raw = nn.ParameterList(
            [nn.Parameter(torch.zeros(*self.bias_shape)) for _ in range(self.K)]
        )
        self.history_trend_delta_raw = nn.ParameterList(
            [nn.Parameter(torch.zeros(*self.history_trend_shape)) for _ in range(self.K)]
        )
        self.register_buffer(
            "scale_temporal_basis_rh",
            self._build_temporal_basis(self.scale_temporal_basis_rank, self.H),
            persistent=True,
        )
        self.register_buffer(
            "history_trend_basis_h",
            self._build_history_trend_basis(self.history_trend_projection, self.H),
            persistent=True,
        )
        self.register_buffer("active_channel_mask_c", torch.ones(self.C, dtype=torch.float32), persistent=True)
        self.register_buffer(
            "active_channel_horizon_mask_ch",
            torch.ones(self.C, self.H, dtype=torch.float32),
            persistent=True,
        )

        if not self.learn_stat_scale:
            for param in self.stat_scale_delta_raw:
                param.requires_grad_(False)
            for param in self.stat_scale_temporal_coef_raw:
                param.requires_grad_(False)
        if not self.learn_residual_scale:
            for param in self.residual_scale_delta_raw:
                param.requires_grad_(False)
            for param in self.residual_scale_temporal_coef_raw:
                param.requires_grad_(False)
        if not self.learn_bias or self.max_bias == 0.0:
            for param in self.bias_raw:
                param.requires_grad_(False)
        if not self.learn_history_trend or self.max_history_trend_delta == 0.0:
            for param in self.history_trend_delta_raw:
                param.requires_grad_(False)
        if self.scale_temporal_basis_rank == 0:
            for param in self.stat_scale_temporal_coef_raw:
                param.requires_grad_(False)
            for param in self.residual_scale_temporal_coef_raw:
                param.requires_grad_(False)

    @staticmethod
    def _normalize_parameterization(value: str) -> str:
        value = str(value).lower().replace("-", "_")
        aliases = {
            "full": "channel_horizon",
            "channel_horizon": "channel_horizon",
            "channelwise_horizon": "channel_horizon",
            "channel": "channel",
            "channelwise": "channel",
            "horizon": "horizon",
            "horizonwise": "horizon",
            "scalar": "scalar",
            "cluster": "scalar",
            "cluster_scalar": "scalar",
        }
        if value not in aliases:
            raise ValueError(
                "learnable_output_anchor parameterization must be one of "
                "channel, channel_horizon, horizon, or scalar."
            )
        return aliases[value]

    @staticmethod
    def _normalize_history_trend_feature(value: str) -> str:
        value = str(value).lower().replace("-", "_")
        aliases = {
            "last_minus_mean": "last_minus_mean",
            "last_mean": "last_minus_mean",
            "demeaned_last": "last_minus_mean",
            "last_minus_first": "last_minus_first",
            "last_first": "last_minus_first",
            "recent_level": "recent_level",
            "recent_mean": "recent_level",
            "window_mean": "recent_level",
            "level": "recent_level",
            "mean_abs_diff": "mean_abs_diff",
            "mean_absolute_diff": "mean_abs_diff",
            "mean_abs_delta": "mean_abs_diff",
            "volatility": "mean_abs_diff",
            "recent_volatility": "mean_abs_diff",
            "recent_slope": "recent_slope",
            "slope": "recent_slope",
            "linear_slope": "recent_slope",
            "trend_slope": "recent_slope",
        }
        if value not in aliases:
            raise ValueError(
                "learnable_output_anchor history_trend_feature must be one of "
                "last_minus_mean, last_minus_first, recent_level, mean_abs_diff, or recent_slope."
            )
        return aliases[value]

    @staticmethod
    def _normalize_history_trend_projection(value: str) -> str:
        value = str(value).lower().replace("-", "_")
        aliases = {
            "linear": "linear",
            "ramp": "linear",
            "constant": "constant",
            "flat": "constant",
        }
        if value not in aliases:
            raise ValueError(
                "learnable_output_anchor history_trend_projection must be linear or constant."
            )
        return aliases[value]

    def _shape_for_parameterization(self, value: str) -> Tuple[int, int]:
        if value == "channel_horizon":
            return self.C, self.H
        if value == "channel":
            return self.C, 1
        if value == "horizon":
            return 1, self.H
        if value == "scalar":
            return 1, 1
        raise ValueError(f"Unsupported learnable_output_anchor parameterization: {value}")

    @staticmethod
    def _build_temporal_basis(rank: int, pred_len: int) -> torch.Tensor:
        rank = max(0, int(rank))
        pred_len = int(pred_len)
        if rank == 0:
            return torch.zeros(0, pred_len, dtype=torch.float32)
        h = torch.arange(pred_len, dtype=torch.float32).view(1, pred_len)
        r = torch.arange(1, rank + 1, dtype=torch.float32).view(rank, 1)
        basis = torch.cos(math.pi * (h + 0.5) * r / max(pred_len, 1))
        basis = basis - basis.mean(dim=1, keepdim=True)
        basis = basis / basis.abs().amax(dim=1, keepdim=True).clamp_min(1.0e-6)
        return basis

    @staticmethod
    def _build_history_trend_basis(projection: str, pred_len: int) -> torch.Tensor:
        pred_len = int(pred_len)
        if str(projection) == "constant":
            return torch.ones(pred_len, dtype=torch.float32)
        return torch.linspace(1.0 / max(pred_len, 1), 1.0, pred_len, dtype=torch.float32)

    def _select_by_channel_cluster(
        self,
        params: nn.ParameterList,
        cluster_id_c: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        cluster_id = cluster_id_c.to(device=device, dtype=torch.long).view(-1)
        if int(cluster_id.numel()) != self.C:
            raise ValueError(
                "learnable_output_anchor cluster_id_c length must match channel count "
                f"({int(cluster_id.numel())} != {self.C})."
            )
        if bool((cluster_id < 0).any()) or bool((cluster_id >= self.K).any()):
            raise ValueError("learnable_output_anchor cluster_id_c contains an out-of-range cluster id.")
        stacked_kch = torch.stack([param.to(device=device, dtype=dtype) for param in params], dim=0)
        channel_idx = torch.arange(self.C, device=device, dtype=torch.long)
        rows = int(stacked_kch.shape[1])
        horizon = int(stacked_kch.shape[2])
        if rows == self.C:
            selected_ch = stacked_kch[cluster_id, channel_idx, :]
        elif rows == 1:
            selected_ch = stacked_kch[cluster_id, 0, :]
        else:
            raise ValueError(
                "learnable_output_anchor parameter row count must be 1 or channel count "
                f"({rows} vs {self.C})."
            )
        return selected_ch

    def _expand_by_channel_cluster(
        self,
        params: nn.ParameterList,
        cluster_id_c: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        selected_ch = self._select_by_channel_cluster(
            params,
            cluster_id_c,
            device=device,
            dtype=dtype,
        )
        horizon = int(selected_ch.shape[1])
        if horizon == self.H:
            return selected_ch
        if horizon == 1:
            return selected_ch.expand(-1, self.H)
        raise ValueError(
            "learnable_output_anchor parameter horizon count must be 1 or pred_len "
            f"({horizon} vs {self.H})."
        )

    def _expand_temporal_scale(
        self,
        base_params: nn.ParameterList,
        temporal_coef_params: nn.ParameterList,
        cluster_id_c: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        scale_raw_ch = self._expand_by_channel_cluster(
            base_params,
            cluster_id_c,
            device=device,
            dtype=dtype,
        )
        if self.scale_temporal_basis_rank <= 0:
            return scale_raw_ch
        coef_cr = self._select_by_channel_cluster(
            temporal_coef_params,
            cluster_id_c,
            device=device,
            dtype=dtype,
        )
        basis_rh = self.scale_temporal_basis_rh.to(device=device, dtype=dtype)
        temporal_ch = torch.matmul(coef_cr, basis_rh)
        return scale_raw_ch + temporal_ch

    def _history_trend_feature_bcl(self, x_bcl: torch.Tensor) -> torch.Tensor:
        if x_bcl.ndim != 3:
            raise ValueError("learnable_output_anchor x_bcl must have shape [batch, channel, input_len].")
        if int(x_bcl.shape[1]) != self.C:
            raise ValueError(
                "learnable_output_anchor x_bcl channel count must match prediction channel count "
                f"({int(x_bcl.shape[1])} != {self.C})."
            )
        input_len = int(x_bcl.shape[2])
        if input_len <= 0:
            raise ValueError("learnable_output_anchor history trend requires a non-empty input window.")
        window = int(self.history_trend_window)
        if window <= 0 or window > input_len:
            window = input_len
        recent_bcw = x_bcl[:, :, -window:]
        last_bc = recent_bcw[:, :, -1]
        if self.history_trend_feature == "last_minus_first":
            return last_bc - recent_bcw[:, :, 0]
        if self.history_trend_feature == "recent_level":
            return recent_bcw.mean(dim=-1)
        if self.history_trend_feature == "mean_abs_diff":
            if int(recent_bcw.shape[-1]) <= 1:
                return torch.zeros_like(last_bc)
            return recent_bcw.diff(dim=-1).abs().mean(dim=-1)
        if self.history_trend_feature == "recent_slope":
            window_len = int(recent_bcw.shape[-1])
            if window_len <= 1:
                return torch.zeros_like(last_bc)
            t_w = torch.arange(window_len, device=recent_bcw.device, dtype=recent_bcw.dtype)
            t_w = t_w - t_w.mean()
            denom = t_w.square().sum().clamp_min(1.0e-12)
            centered_bcw = recent_bcw - recent_bcw.mean(dim=-1, keepdim=True)
            slope_bc = (centered_bcw * t_w.view(1, 1, window_len)).sum(dim=-1) / denom
            return slope_bc * float(window_len - 1)
        return last_bc - recent_bcw.mean(dim=-1)

    def forward(
        self,
        pred_bch: torch.Tensor,
        *,
        cluster_id_c: torch.Tensor,
        x_bcl: Optional[torch.Tensor] = None,
        stat_delta_bch: Optional[torch.Tensor] = None,
        residual_delta_bch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if pred_bch.ndim != 3:
            raise ValueError("learnable_output_anchor pred_bch must have shape [batch, channel, horizon].")
        if int(pred_bch.shape[1]) != self.C or int(pred_bch.shape[2]) != self.H:
            raise ValueError(
                "learnable_output_anchor prediction shape mismatch: "
                f"expected channel/horizon {(self.C, self.H)}, got {(int(pred_bch.shape[1]), int(pred_bch.shape[2]))}."
            )
        if stat_delta_bch is not None and stat_delta_bch.shape != pred_bch.shape:
            raise ValueError("learnable_output_anchor stat_delta_bch must match pred_bch shape.")
        if residual_delta_bch is not None and residual_delta_bch.shape != pred_bch.shape:
            raise ValueError("learnable_output_anchor residual_delta_bch must match pred_bch shape.")

        out = pred_bch
        if stat_delta_bch is not None and self.learn_stat_scale and self.max_scale_delta != 0.0:
            stat_scale_ch = self.max_scale_delta * torch.tanh(
                self._expand_temporal_scale(
                    self.stat_scale_delta_raw,
                    self.stat_scale_temporal_coef_raw,
                    cluster_id_c,
                    device=pred_bch.device,
                    dtype=pred_bch.dtype,
                )
            )
            out = out + stat_scale_ch.unsqueeze(0) * stat_delta_bch
        if residual_delta_bch is not None and self.learn_residual_scale and self.max_scale_delta != 0.0:
            residual_scale_ch = self.max_scale_delta * torch.tanh(
                self._expand_temporal_scale(
                    self.residual_scale_delta_raw,
                    self.residual_scale_temporal_coef_raw,
                    cluster_id_c,
                    device=pred_bch.device,
                    dtype=pred_bch.dtype,
                )
            )
            out = out + residual_scale_ch.unsqueeze(0) * residual_delta_bch
        if self.learn_bias and self.max_bias != 0.0:
            bias_ch = self.max_bias * torch.tanh(
                self._expand_by_channel_cluster(
                    self.bias_raw,
                    cluster_id_c,
                    device=pred_bch.device,
                    dtype=pred_bch.dtype,
                )
            )
            out = out + bias_ch.unsqueeze(0)
        if self.learn_history_trend and self.max_history_trend_delta != 0.0:
            if x_bcl is None:
                raise ValueError("learnable_output_anchor learn_history_trend requires x_bcl.")
            trend_bc = self._history_trend_feature_bcl(x_bcl.to(device=pred_bch.device, dtype=pred_bch.dtype))
            trend_delta_ch = self.max_history_trend_delta * torch.tanh(
                self._expand_by_channel_cluster(
                    self.history_trend_delta_raw,
                    cluster_id_c,
                    device=pred_bch.device,
                    dtype=pred_bch.dtype,
                )
            )
            trend_basis_h = self.history_trend_basis_h.to(device=pred_bch.device, dtype=pred_bch.dtype)
            trend_delta_ch = trend_delta_ch * trend_basis_h.view(1, self.H)
            out = out + trend_bc.unsqueeze(-1) * trend_delta_ch.unsqueeze(0)
        mask_ch = self.active_channel_mask_c.to(device=pred_bch.device, dtype=pred_bch.dtype).view(1, self.C, 1)
        mask_ch = mask_ch * self.active_channel_horizon_mask_ch.to(
            device=pred_bch.device,
            dtype=pred_bch.dtype,
        ).view(1, self.C, self.H)
        if bool((mask_ch < 1.0).any()):
            out = pred_bch + mask_ch * (out - pred_bch)
        return out

    def set_active_channel_mask(self, mask_c: torch.Tensor) -> None:
        mask = mask_c.detach().to(device=self.active_channel_mask_c.device, dtype=torch.float32).view(-1)
        if int(mask.numel()) != self.C:
            raise ValueError(
                "learnable_output_anchor active channel mask must match channel count "
                f"({int(mask.numel())} != {self.C})."
            )
        self.active_channel_mask_c.copy_(mask.clamp(0.0, 1.0))

    def clear_active_channel_mask(self) -> None:
        self.active_channel_mask_c.fill_(1.0)

    def set_active_channel_horizon_mask(self, mask_ch: torch.Tensor) -> None:
        mask = mask_ch.detach().to(
            device=self.active_channel_horizon_mask_ch.device,
            dtype=torch.float32,
        )
        if tuple(mask.shape) != (self.C, self.H):
            raise ValueError(
                "learnable_output_anchor active channel-horizon mask must match "
                f"(num_channels, pred_len) {(self.C, self.H)}, got {tuple(mask.shape)}."
            )
        self.active_channel_horizon_mask_ch.copy_(mask.clamp(0.0, 1.0))

    def clear_active_channel_horizon_mask(self) -> None:
        self.active_channel_horizon_mask_ch.fill_(1.0)

    def get_cluster_params(self, k: int) -> List[nn.Parameter]:
        k = int(k)
        params: List[nn.Parameter] = []
        for plist in (
            self.stat_scale_delta_raw,
            self.residual_scale_delta_raw,
            self.stat_scale_temporal_coef_raw,
            self.residual_scale_temporal_coef_raw,
            self.bias_raw,
            self.history_trend_delta_raw,
        ):
            param = plist[k]
            if param.requires_grad:
                params.append(param)
        return params

    def mask_cluster_grads(self, stopped_k: torch.Tensor) -> None:
        for k in range(self.K):
            if not bool(stopped_k[k].item()):
                continue
            for param in self.get_cluster_params(k):
                if param.grad is not None:
                    param.grad.zero_()

    def get_cluster_state(self, k: int) -> Dict[str, torch.Tensor]:
        k = int(k)
        return {
            "stat_scale_delta_raw": self.stat_scale_delta_raw[k].detach().cpu(),
            "residual_scale_delta_raw": self.residual_scale_delta_raw[k].detach().cpu(),
            "stat_scale_temporal_coef_raw": self.stat_scale_temporal_coef_raw[k].detach().cpu(),
            "residual_scale_temporal_coef_raw": self.residual_scale_temporal_coef_raw[k].detach().cpu(),
            "bias_raw": self.bias_raw[k].detach().cpu(),
            "history_trend_delta_raw": self.history_trend_delta_raw[k].detach().cpu(),
        }

    def load_cluster_state(self, k: int, state: Dict[str, torch.Tensor]) -> None:
        k = int(k)
        if "stat_scale_delta_raw" in state:
            self.stat_scale_delta_raw[k].data.copy_(state["stat_scale_delta_raw"].to(self.stat_scale_delta_raw[k].device))
        if "residual_scale_delta_raw" in state:
            self.residual_scale_delta_raw[k].data.copy_(
                state["residual_scale_delta_raw"].to(self.residual_scale_delta_raw[k].device)
            )
        if "stat_scale_temporal_coef_raw" in state:
            self.stat_scale_temporal_coef_raw[k].data.copy_(
                state["stat_scale_temporal_coef_raw"].to(self.stat_scale_temporal_coef_raw[k].device)
            )
        if "residual_scale_temporal_coef_raw" in state:
            self.residual_scale_temporal_coef_raw[k].data.copy_(
                state["residual_scale_temporal_coef_raw"].to(self.residual_scale_temporal_coef_raw[k].device)
            )
        if "bias_raw" in state:
            self.bias_raw[k].data.copy_(state["bias_raw"].to(self.bias_raw[k].device))
        if "history_trend_delta_raw" in state:
            self.history_trend_delta_raw[k].data.copy_(
                state["history_trend_delta_raw"].to(self.history_trend_delta_raw[k].device)
            )
