from typing import Dict, List, Optional
import torch
from torch import nn
from torch.nn import functional as F

from .cluster_mlp import ClusterwiseMLP
from ..utils.cluster_memory import scatter_mean_bcl_to_bkl


class _ClusterPredictorBase(nn.Module):
    def __init__(self, num_clusters: int):
        super().__init__()
        self.K = num_clusters

    def get_cluster_params(self, k: int) -> List[nn.Parameter]:
        raise NotImplementedError

    def mask_cluster_grads(self, stopped_k: torch.Tensor):
        for k in range(self.K):
            if not bool(stopped_k[k].item()):
                continue
            for p in self.get_cluster_params(k):
                if p.grad is not None:
                    p.grad.zero_()

    def get_cluster_state(self, k: int):
        raise NotImplementedError

    def load_cluster_state(self, k: int, state):
        raise NotImplementedError


class ClusterwiseRevIN(_ClusterPredictorBase):
    def __init__(self, base: _ClusterPredictorBase, eps: float = 1.0e-5):
        super().__init__(num_clusters=base.K)
        self.base = base
        self.eps = float(eps)

    def forward(self, x_bcl: torch.Tensor, cluster_id_c: torch.Tensor) -> torch.Tensor:
        mean = x_bcl.mean(dim=-1, keepdim=True)
        std = x_bcl.std(dim=-1, keepdim=True).clamp_min(self.eps)
        x_norm = (x_bcl - mean) / std
        y_norm = self.base(x_norm, cluster_id_c)
        return y_norm * std + mean

    def get_cluster_params(self, k: int) -> List[nn.Parameter]:
        return self.base.get_cluster_params(k)

    def get_cluster_state(self, k: int):
        return self.base.get_cluster_state(k)

    def load_cluster_state(self, k: int, state):
        self.base.load_cluster_state(k, state)


class ClusterwiseInputTail(_ClusterPredictorBase):
    """Expose a long input window to outer modules while feeding only its tail to the base predictor."""

    def __init__(self, base: _ClusterPredictorBase, input_len: int, tail_len: int):
        super().__init__(num_clusters=base.K)
        self.base = base
        self.L = int(input_len)
        self.tail_len = int(tail_len)
        self.H = int(base.H)
        if self.tail_len <= 0:
            raise ValueError("model.predictor_input_len must be positive.")
        if self.tail_len > self.L:
            raise ValueError(
                f"model.predictor_input_len={self.tail_len} cannot exceed window.input_len={self.L}."
            )

    def forward(self, x_bcl: torch.Tensor, cluster_id_c: torch.Tensor) -> torch.Tensor:
        if x_bcl.shape[-1] < self.tail_len:
            raise ValueError(
                f"Input length {int(x_bcl.shape[-1])} is shorter than predictor tail length {self.tail_len}."
            )
        return self.base(x_bcl[..., -self.tail_len :], cluster_id_c)

    def get_cluster_params(self, k: int) -> List[nn.Parameter]:
        return self.base.get_cluster_params(k)

    def get_cluster_state(self, k: int):
        return self.base.get_cluster_state(k)

    def load_cluster_state(self, k: int, state):
        self.base.load_cluster_state(k, state)


class ClusterwiseSeasonalResidual(_ClusterPredictorBase):
    def __init__(self, base: _ClusterPredictorBase, period: int, num_periods: int = 1):
        super().__init__(num_clusters=base.K)
        self.base = base
        self.period = max(int(period), 1)
        self.num_periods = max(int(num_periods), 1)
        self.L = int(base.L)
        self.H = int(base.H)

        idx_rows = []
        mask_rows = []
        for p in range(1, self.num_periods + 1):
            idx_h = []
            mask_h = []
            for h in range(self.H):
                idx = self.L - (p * self.period) + (h % self.period)
                valid = 0 <= idx < self.L
                idx_h.append(max(idx, 0))
                mask_h.append(1.0 if valid else 0.0)
            idx_rows.append(idx_h)
            mask_rows.append(mask_h)
        self.register_buffer("seasonal_idx_ph", torch.tensor(idx_rows, dtype=torch.long), persistent=False)
        self.register_buffer("seasonal_mask_ph", torch.tensor(mask_rows, dtype=torch.float32), persistent=False)

    def _seasonal_baseline(self, x_bcl: torch.Tensor) -> torch.Tensor:
        if self.seasonal_idx_ph.numel() == 0:
            return x_bcl.new_zeros((x_bcl.shape[0], x_bcl.shape[1], self.H))
        flat_idx = self.seasonal_idx_ph.reshape(-1)
        gathered = x_bcl.index_select(2, flat_idx)
        gathered = gathered.reshape(x_bcl.shape[0], x_bcl.shape[1], self.num_periods, self.H)
        mask = self.seasonal_mask_ph.view(1, 1, self.num_periods, self.H).to(device=x_bcl.device, dtype=x_bcl.dtype)
        denom = mask.sum(dim=2).clamp_min(1.0)
        return (gathered * mask).sum(dim=2) / denom

    def forward(self, x_bcl: torch.Tensor, cluster_id_c: torch.Tensor) -> torch.Tensor:
        baseline = self._seasonal_baseline(x_bcl)
        residual = self.base(x_bcl, cluster_id_c)
        return baseline + residual

    def get_cluster_params(self, k: int) -> List[nn.Parameter]:
        return self.base.get_cluster_params(k)

    def get_cluster_state(self, k: int):
        return self.base.get_cluster_state(k)

    def load_cluster_state(self, k: int, state):
        self.base.load_cluster_state(k, state)


class ClusterwiseSeasonalAnchor(ClusterwiseSeasonalResidual):
    def __init__(
        self,
        base: _ClusterPredictorBase,
        period: int,
        num_periods: int = 1,
        delta_scale: float = 1.0,
    ):
        super().__init__(base=base, period=period, num_periods=num_periods)
        self.delta_scale = float(delta_scale)

    def forward(self, x_bcl: torch.Tensor, cluster_id_c: torch.Tensor) -> torch.Tensor:
        seasonal = self._seasonal_baseline(x_bcl)
        base_pred = self.base(x_bcl, cluster_id_c)
        last = x_bcl[..., -1:]
        return seasonal + self.delta_scale * (base_pred - last)


class ClusterwiseRecursiveRollout(_ClusterPredictorBase):
    def __init__(self, base: _ClusterPredictorBase, pred_len: int):
        super().__init__(num_clusters=base.K)
        self.base = base
        self.L = int(base.L)
        self.chunk_len = int(base.H)
        self.H = int(pred_len)
        if self.chunk_len <= 0:
            raise ValueError("recursive_rollout requires a positive chunk length.")

    def forward(self, x_bcl: torch.Tensor, cluster_id_c: torch.Tensor) -> torch.Tensor:
        ctx = x_bcl
        chunks = []
        produced = 0
        while produced < self.H:
            y_chunk = self.base(ctx, cluster_id_c)
            take = min(self.chunk_len, self.H - produced)
            chunks.append(y_chunk[..., :take])
            produced += take
            ctx = torch.cat([ctx, y_chunk], dim=-1)[..., -self.L :]
        return torch.cat(chunks, dim=-1)

    def get_cluster_params(self, k: int) -> List[nn.Parameter]:
        return self.base.get_cluster_params(k)

    def get_cluster_state(self, k: int):
        return self.base.get_cluster_state(k)

    def load_cluster_state(self, k: int, state):
        self.base.load_cluster_state(k, state)


class ClusterwiseNLinear(_ClusterPredictorBase):
    def __init__(self, num_clusters: int, input_len: int, pred_len: int):
        super().__init__(num_clusters=num_clusters)
        self.L = input_len
        self.H = pred_len
        self.W = nn.ParameterList(
            [nn.Parameter(torch.empty(input_len, pred_len)) for _ in range(num_clusters)]
        )
        self.b = nn.ParameterList(
            [nn.Parameter(torch.zeros(pred_len)) for _ in range(num_clusters)]
        )
        self.reset_parameters()

    def reset_parameters(self):
        for w in self.W:
            nn.init.xavier_uniform_(w)

    def forward(self, x_bcl: torch.Tensor, cluster_id_c: torch.Tensor) -> torch.Tensor:
        last = x_bcl[..., -1:]
        x_centered = x_bcl - last
        W = torch.stack(list(self.W), dim=0).index_select(0, cluster_id_c)  # [C, L, H]
        b = torch.stack(list(self.b), dim=0).index_select(0, cluster_id_c)  # [C, H]
        y = torch.einsum("bcl,clh->bch", x_centered, W) + b.unsqueeze(0)
        return y + last

    def get_cluster_params(self, k: int) -> List[nn.Parameter]:
        return [self.W[k], self.b[k]]

    def get_cluster_state(self, k: int):
        return {
            "W": self.W[k].detach().cpu(),
            "b": self.b[k].detach().cpu(),
        }

    def load_cluster_state(self, k: int, state):
        device = self.W[k].device
        self.W[k].data.copy_(state["W"].to(device))
        self.b[k].data.copy_(state["b"].to(device))


class ClusterwiseContextMLP(_ClusterPredictorBase):
    def __init__(
        self,
        num_clusters: int,
        input_len: int,
        pred_len: int,
        hidden_dim: int,
        dropout: float = 0.0,
        include_delta: bool = True,
    ):
        super().__init__(num_clusters=num_clusters)
        self.L = input_len
        self.H = pred_len
        self.D = hidden_dim
        self.include_delta = bool(include_delta)
        self.input_dim = input_len * (3 if self.include_delta else 2)

        self.W1 = nn.ParameterList(
            [nn.Parameter(torch.empty(self.input_dim, hidden_dim)) for _ in range(num_clusters)]
        )
        self.b1 = nn.ParameterList(
            [nn.Parameter(torch.zeros(hidden_dim)) for _ in range(num_clusters)]
        )
        self.W2 = nn.ParameterList(
            [nn.Parameter(torch.empty(hidden_dim, pred_len)) for _ in range(num_clusters)]
        )
        self.b2 = nn.ParameterList(
            [nn.Parameter(torch.zeros(pred_len)) for _ in range(num_clusters)]
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        for w in self.W1:
            nn.init.xavier_uniform_(w)
        for w in self.W2:
            nn.init.xavier_uniform_(w)

    def forward(self, x_bcl: torch.Tensor, cluster_id_c: torch.Tensor) -> torch.Tensor:
        cluster_mean_bkl = scatter_mean_bcl_to_bkl(x_bcl, cluster_id_c, self.K)
        cluster_mean_bcl = cluster_mean_bkl.index_select(1, cluster_id_c)
        feat_parts = [x_bcl, cluster_mean_bcl]
        if self.include_delta:
            feat_parts.append(x_bcl - cluster_mean_bcl)
        feat_bcf = torch.cat(feat_parts, dim=-1)

        W1 = torch.stack(list(self.W1), dim=0).index_select(0, cluster_id_c)
        b1 = torch.stack(list(self.b1), dim=0).index_select(0, cluster_id_c)
        W2 = torch.stack(list(self.W2), dim=0).index_select(0, cluster_id_c)
        b2 = torch.stack(list(self.b2), dim=0).index_select(0, cluster_id_c)

        h = torch.einsum("bcl,cld->bcd", feat_bcf, W1) + b1.unsqueeze(0)
        h = self.drop(self.act(h))
        y = torch.einsum("bcd,cdh->bch", h, W2) + b2.unsqueeze(0)
        return y

    def get_cluster_params(self, k: int) -> List[nn.Parameter]:
        return [self.W1[k], self.b1[k], self.W2[k], self.b2[k]]

    def get_cluster_state(self, k: int):
        return {
            "W1": self.W1[k].detach().cpu(),
            "b1": self.b1[k].detach().cpu(),
            "W2": self.W2[k].detach().cpu(),
            "b2": self.b2[k].detach().cpu(),
        }

    def load_cluster_state(self, k: int, state):
        device = self.W1[k].device
        self.W1[k].data.copy_(state["W1"].to(device))
        self.b1[k].data.copy_(state["b1"].to(device))
        self.W2[k].data.copy_(state["W2"].to(device))
        self.b2[k].data.copy_(state["b2"].to(device))


class ClusterwiseSegmentMLP(_ClusterPredictorBase):
    def __init__(
        self,
        num_clusters: int,
        input_len: int,
        pred_len: int,
        hidden_dim: int,
        dropout: float = 0.0,
        chunk_len: int = 96,
    ):
        super().__init__(num_clusters=num_clusters)
        self.L = int(input_len)
        self.H = int(pred_len)
        self.D = int(hidden_dim)
        self.chunk_len = max(int(chunk_len), 1)
        self.num_segments = int((self.H + self.chunk_len - 1) // self.chunk_len)

        self.W1 = nn.ParameterList(
            [nn.Parameter(torch.empty(self.L, self.D)) for _ in range(num_clusters)]
        )
        self.b1 = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.D)) for _ in range(num_clusters)]
        )
        self.segment_emb = nn.ParameterList(
            [nn.Parameter(torch.empty(self.num_segments, self.D)) for _ in range(num_clusters)]
        )
        self.W2 = nn.ParameterList(
            [nn.Parameter(torch.empty(self.D, self.chunk_len)) for _ in range(num_clusters)]
        )
        self.b2 = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.num_segments, self.chunk_len)) for _ in range(num_clusters)]
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        for w in self.W1:
            nn.init.xavier_uniform_(w)
        for emb in self.segment_emb:
            nn.init.normal_(emb, mean=0.0, std=0.02)
        for w in self.W2:
            nn.init.xavier_uniform_(w)

    def forward(self, x_bcl: torch.Tensor, cluster_id_c: torch.Tensor) -> torch.Tensor:
        b, c, _ = x_bcl.shape
        y_bch = x_bcl.new_zeros((b, c, self.H))
        for k in range(self.K):
            idx = (cluster_id_c == k).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            x_bnl = x_bcl.index_select(1, idx)
            last = x_bnl[..., -1:]
            x_center = x_bnl - last
            h = torch.einsum("bnl,ld->bnd", x_center, self.W1[k]) + self.b1[k].view(1, 1, -1)
            h = self.drop(self.act(h))
            h_seg = h.unsqueeze(2) + self.segment_emb[k].view(1, 1, self.num_segments, self.D)
            chunks = torch.einsum("bnsd,dh->bnsh", h_seg, self.W2[k]) + self.b2[k].view(1, 1, self.num_segments, self.chunk_len)
            y_bnh = chunks.reshape(b, idx.numel(), self.num_segments * self.chunk_len)[..., : self.H] + last
            y_bch = y_bch.index_copy(1, idx, y_bnh)
        return y_bch

    def get_cluster_params(self, k: int) -> List[nn.Parameter]:
        return [self.W1[k], self.b1[k], self.segment_emb[k], self.W2[k], self.b2[k]]

    def get_cluster_state(self, k: int):
        return {
            "W1": self.W1[k].detach().cpu(),
            "b1": self.b1[k].detach().cpu(),
            "segment_emb": self.segment_emb[k].detach().cpu(),
            "W2": self.W2[k].detach().cpu(),
            "b2": self.b2[k].detach().cpu(),
        }

    def load_cluster_state(self, k: int, state):
        device = self.W1[k].device
        self.W1[k].data.copy_(state["W1"].to(device))
        self.b1[k].data.copy_(state["b1"].to(device))
        self.segment_emb[k].data.copy_(state["segment_emb"].to(device))
        self.W2[k].data.copy_(state["W2"].to(device))
        self.b2[k].data.copy_(state["b2"].to(device))


class ClusterwiseLongAnchorMLP(_ClusterPredictorBase):
    def __init__(
        self,
        num_clusters: int,
        input_len: int,
        pred_len: int,
        hidden_dim: int,
        dropout: float = 0.0,
        chunk_len: int = 96,
        anchor_points: int = 9,
        detail_scale: float = 0.5,
        residual: bool = True,
    ):
        super().__init__(num_clusters=num_clusters)
        self.L = int(input_len)
        self.H = int(pred_len)
        self.D = int(hidden_dim)
        self.chunk_len = max(int(chunk_len), 1)
        self.num_segments = int((self.H + self.chunk_len - 1) // self.chunk_len)
        self.anchor_points = max(int(anchor_points), 2)
        self.detail_scale = float(detail_scale)
        self.residual = bool(residual)

        self.W1 = nn.ParameterList(
            [nn.Parameter(torch.empty(self.L, self.D)) for _ in range(num_clusters)]
        )
        self.b1 = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.D)) for _ in range(num_clusters)]
        )
        self.anchor_emb = nn.ParameterList(
            [nn.Parameter(torch.empty(self.anchor_points, self.D)) for _ in range(num_clusters)]
        )
        self.W_anchor = nn.ParameterList(
            [nn.Parameter(torch.empty(self.D, 1)) for _ in range(num_clusters)]
        )
        self.b_anchor = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.anchor_points)) for _ in range(num_clusters)]
        )
        self.detail_emb = nn.ParameterList(
            [nn.Parameter(torch.empty(self.num_segments, self.D)) for _ in range(num_clusters)]
        )
        self.W_detail = nn.ParameterList(
            [nn.Parameter(torch.empty(self.D, self.chunk_len)) for _ in range(num_clusters)]
        )
        self.b_detail = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.num_segments, self.chunk_len)) for _ in range(num_clusters)]
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        for w in self.W1:
            nn.init.xavier_uniform_(w)
        for emb in self.anchor_emb:
            nn.init.normal_(emb, mean=0.0, std=0.02)
        for w in self.W_anchor:
            nn.init.xavier_uniform_(w)
        for emb in self.detail_emb:
            nn.init.normal_(emb, mean=0.0, std=0.02)
        for w in self.W_detail:
            nn.init.xavier_uniform_(w)

    def _interpolate_anchor(self, anchor_bna: torch.Tensor) -> torch.Tensor:
        b, n, a = anchor_bna.shape
        curve = F.interpolate(
            anchor_bna.reshape(b * n, 1, a),
            size=self.H,
            mode="linear",
            align_corners=True,
        )
        return curve.reshape(b, n, self.H)

    def forward(self, x_bcl: torch.Tensor, cluster_id_c: torch.Tensor) -> torch.Tensor:
        b, c, _ = x_bcl.shape
        y_bch = x_bcl.new_zeros((b, c, self.H))
        for k in range(self.K):
            idx = (cluster_id_c == k).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            x_bnl = x_bcl.index_select(1, idx)
            last = x_bnl[..., -1:]
            x_in = x_bnl - last if self.residual else x_bnl
            h = torch.einsum("bnl,ld->bnd", x_in, self.W1[k]) + self.b1[k].view(1, 1, -1)
            h = self.drop(self.act(h))

            h_anchor = h.unsqueeze(2) + self.anchor_emb[k].view(1, 1, self.anchor_points, self.D)
            anchor_bna = torch.einsum("bnad,do->bnao", h_anchor, self.W_anchor[k]).squeeze(-1)
            anchor_bna = anchor_bna + self.b_anchor[k].view(1, 1, self.anchor_points)
            anchor_bnh = self._interpolate_anchor(anchor_bna)

            h_detail = h.unsqueeze(2) + self.detail_emb[k].view(1, 1, self.num_segments, self.D)
            detail_bnsh = torch.einsum("bnsd,dh->bnsh", h_detail, self.W_detail[k])
            detail_bnsh = detail_bnsh + self.b_detail[k].view(1, 1, self.num_segments, self.chunk_len)
            detail_bnsh = detail_bnsh - detail_bnsh.mean(dim=-1, keepdim=True)
            detail_bnh = detail_bnsh.reshape(b, idx.numel(), self.num_segments * self.chunk_len)[..., : self.H]

            y_bnh = anchor_bnh + (self.detail_scale * detail_bnh)
            if self.residual:
                y_bnh = y_bnh + last
            y_bch = y_bch.index_copy(1, idx, y_bnh)
        return y_bch

    def get_cluster_params(self, k: int) -> List[nn.Parameter]:
        return [
            self.W1[k],
            self.b1[k],
            self.anchor_emb[k],
            self.W_anchor[k],
            self.b_anchor[k],
            self.detail_emb[k],
            self.W_detail[k],
            self.b_detail[k],
        ]

    def get_cluster_state(self, k: int):
        return {
            "W1": self.W1[k].detach().cpu(),
            "b1": self.b1[k].detach().cpu(),
            "anchor_emb": self.anchor_emb[k].detach().cpu(),
            "W_anchor": self.W_anchor[k].detach().cpu(),
            "b_anchor": self.b_anchor[k].detach().cpu(),
            "detail_emb": self.detail_emb[k].detach().cpu(),
            "W_detail": self.W_detail[k].detach().cpu(),
            "b_detail": self.b_detail[k].detach().cpu(),
        }

    def load_cluster_state(self, k: int, state):
        device = self.W1[k].device
        self.W1[k].data.copy_(state["W1"].to(device))
        self.b1[k].data.copy_(state["b1"].to(device))
        self.anchor_emb[k].data.copy_(state["anchor_emb"].to(device))
        self.W_anchor[k].data.copy_(state["W_anchor"].to(device))
        self.b_anchor[k].data.copy_(state["b_anchor"].to(device))
        self.detail_emb[k].data.copy_(state["detail_emb"].to(device))
        self.W_detail[k].data.copy_(state["W_detail"].to(device))
        self.b_detail[k].data.copy_(state["b_detail"].to(device))


class ClusterwiseChannelHeadMLP(_ClusterPredictorBase):
    def __init__(
        self,
        num_clusters: int,
        input_len: int,
        pred_len: int,
        hidden_dim: int,
        num_channels: int,
        cluster_id_c: torch.Tensor,
        dropout: float = 0.0,
        residual: bool = True,
        include_seasonal_profile: bool = False,
    ):
        super().__init__(num_clusters=num_clusters)
        self.L = int(input_len)
        self.H = int(pred_len)
        self.D = int(hidden_dim)
        self.C = int(num_channels)
        self.residual = bool(residual)
        if cluster_id_c.numel() != self.C:
            raise ValueError(
                f"channel_head_mlp expected cluster_id_c length {self.C}, got {int(cluster_id_c.numel())}."
            )
        self.register_buffer("cluster_id_c", cluster_id_c.detach().long().cpu(), persistent=False)

        self.W1 = nn.ParameterList(
            [nn.Parameter(torch.empty(self.L, self.D)) for _ in range(num_clusters)]
        )
        self.b1 = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.D)) for _ in range(num_clusters)]
        )
        self.W2 = nn.ParameterList(
            [nn.Parameter(torch.empty(self.D, self.H)) for _ in range(self.C)]
        )
        self.b2 = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.H)) for _ in range(self.C)]
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        for w in self.W1:
            nn.init.xavier_uniform_(w)
        for w in self.W2:
            nn.init.xavier_uniform_(w)

    def _cluster_channel_idx(self, k: int) -> torch.Tensor:
        return (self.cluster_id_c == int(k)).nonzero(as_tuple=False).view(-1)

    def forward(self, x_bcl: torch.Tensor, cluster_id_c: torch.Tensor) -> torch.Tensor:
        if cluster_id_c.numel() != self.C:
            raise ValueError(
                f"channel_head_mlp expected {self.C} channels, got {int(cluster_id_c.numel())}."
            )
        last = x_bcl[..., -1:]
        x_in = x_bcl - last if self.residual else x_bcl
        W1 = torch.stack(list(self.W1), dim=0).index_select(0, cluster_id_c)  # [C,L,D]
        b1 = torch.stack(list(self.b1), dim=0).index_select(0, cluster_id_c)  # [C,D]
        W2 = torch.stack(list(self.W2), dim=0)  # [C,D,H]
        b2 = torch.stack(list(self.b2), dim=0)  # [C,H]

        h = torch.einsum("bcl,cld->bcd", x_in, W1) + b1.unsqueeze(0)
        h = self.drop(self.act(h))
        y = torch.einsum("bcd,cdh->bch", h, W2) + b2.unsqueeze(0)
        return y + last if self.residual else y

    def get_cluster_params(self, k: int) -> List[nn.Parameter]:
        params: List[nn.Parameter] = [self.W1[k], self.b1[k]]
        idx = self._cluster_channel_idx(k)
        params.extend(self.W2[int(i.item())] for i in idx)
        params.extend(self.b2[int(i.item())] for i in idx)
        return params

    def get_cluster_state(self, k: int):
        idx = self._cluster_channel_idx(k)
        if idx.numel() > 0:
            w2 = torch.stack([self.W2[int(i.item())].detach().cpu() for i in idx], dim=0)
            b2 = torch.stack([self.b2[int(i.item())].detach().cpu() for i in idx], dim=0)
        else:
            w2 = torch.empty(0, self.D, self.H)
            b2 = torch.empty(0, self.H)
        return {
            "W1": self.W1[k].detach().cpu(),
            "b1": self.b1[k].detach().cpu(),
            "channel_idx": idx.detach().cpu(),
            "W2": w2,
            "b2": b2,
        }

    def load_cluster_state(self, k: int, state):
        device = self.W1[k].device
        self.W1[k].data.copy_(state["W1"].to(device))
        self.b1[k].data.copy_(state["b1"].to(device))
        idx = self._cluster_channel_idx(k)
        saved_idx = state.get("channel_idx", idx.detach().cpu())
        if saved_idx.numel() != idx.numel() or not torch.equal(saved_idx.cpu(), idx.detach().cpu()):
            raise ValueError(f"channel_head_mlp cluster {k} channel indices do not match checkpoint state.")
        w2 = state["W2"].to(device)
        b2 = state["b2"].to(device)
        for j, i in enumerate(idx):
            c = int(i.item())
            self.W2[c].data.copy_(w2[j])
            self.b2[c].data.copy_(b2[j])


class ClusterwiseContextChannelHeadMLP(_ClusterPredictorBase):
    def __init__(
        self,
        num_clusters: int,
        input_len: int,
        pred_len: int,
        hidden_dim: int,
        num_channels: int,
        cluster_id_c: torch.Tensor,
        dropout: float = 0.0,
        residual: bool = True,
        include_delta: bool = True,
    ):
        super().__init__(num_clusters=num_clusters)
        self.L = int(input_len)
        self.H = int(pred_len)
        self.D = int(hidden_dim)
        self.C = int(num_channels)
        self.residual = bool(residual)
        self.include_delta = bool(include_delta)
        self.input_dim = self.L * (3 if self.include_delta else 2)
        if cluster_id_c.numel() != self.C:
            raise ValueError(
                f"context_channel_head_mlp expected cluster_id_c length {self.C}, got {int(cluster_id_c.numel())}."
            )
        self.register_buffer("cluster_id_c", cluster_id_c.detach().long().cpu(), persistent=False)

        self.W1 = nn.ParameterList(
            [nn.Parameter(torch.empty(self.input_dim, self.D)) for _ in range(num_clusters)]
        )
        self.b1 = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.D)) for _ in range(num_clusters)]
        )
        self.W2 = nn.ParameterList(
            [nn.Parameter(torch.empty(self.D, self.H)) for _ in range(self.C)]
        )
        self.b2 = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.H)) for _ in range(self.C)]
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        for w in self.W1:
            nn.init.xavier_uniform_(w)
        for w in self.W2:
            nn.init.xavier_uniform_(w)

    def _cluster_channel_idx(self, k: int) -> torch.Tensor:
        return (self.cluster_id_c == int(k)).nonzero(as_tuple=False).view(-1)

    def forward(self, x_bcl: torch.Tensor, cluster_id_c: torch.Tensor) -> torch.Tensor:
        if cluster_id_c.numel() != self.C:
            raise ValueError(
                f"context_channel_head_mlp expected {self.C} channels, got {int(cluster_id_c.numel())}."
            )
        last = x_bcl[..., -1:]
        x_in = x_bcl - last if self.residual else x_bcl
        cluster_mean_bkl = scatter_mean_bcl_to_bkl(x_in, cluster_id_c, self.K)
        cluster_mean_bcl = cluster_mean_bkl.index_select(1, cluster_id_c)
        feat_parts = [x_in, cluster_mean_bcl]
        if self.include_delta:
            feat_parts.append(x_in - cluster_mean_bcl)
        feat_bcf = torch.cat(feat_parts, dim=-1)

        W1 = torch.stack(list(self.W1), dim=0).index_select(0, cluster_id_c)
        b1 = torch.stack(list(self.b1), dim=0).index_select(0, cluster_id_c)
        W2 = torch.stack(list(self.W2), dim=0)
        b2 = torch.stack(list(self.b2), dim=0)

        h = torch.einsum("bcl,cld->bcd", feat_bcf, W1) + b1.unsqueeze(0)
        h = self.drop(self.act(h))
        y = torch.einsum("bcd,cdh->bch", h, W2) + b2.unsqueeze(0)
        return y + last if self.residual else y

    def get_cluster_params(self, k: int) -> List[nn.Parameter]:
        params: List[nn.Parameter] = [self.W1[k], self.b1[k]]
        idx = self._cluster_channel_idx(k)
        params.extend(self.W2[int(i.item())] for i in idx)
        params.extend(self.b2[int(i.item())] for i in idx)
        return params

    def get_cluster_state(self, k: int):
        idx = self._cluster_channel_idx(k)
        if idx.numel() > 0:
            w2 = torch.stack([self.W2[int(i.item())].detach().cpu() for i in idx], dim=0)
            b2 = torch.stack([self.b2[int(i.item())].detach().cpu() for i in idx], dim=0)
        else:
            w2 = torch.empty(0, self.D, self.H)
            b2 = torch.empty(0, self.H)
        return {
            "W1": self.W1[k].detach().cpu(),
            "b1": self.b1[k].detach().cpu(),
            "channel_idx": idx.detach().cpu(),
            "W2": w2,
            "b2": b2,
        }

    def load_cluster_state(self, k: int, state):
        device = self.W1[k].device
        self.W1[k].data.copy_(state["W1"].to(device))
        self.b1[k].data.copy_(state["b1"].to(device))
        idx = self._cluster_channel_idx(k)
        saved_idx = state.get("channel_idx", idx.detach().cpu())
        if saved_idx.numel() != idx.numel() or not torch.equal(saved_idx.cpu(), idx.detach().cpu()):
            raise ValueError(f"context_channel_head_mlp cluster {k} channel indices do not match checkpoint state.")
        w2 = state["W2"].to(device)
        b2 = state["b2"].to(device)
        for j, i in enumerate(idx):
            c = int(i.item())
            self.W2[c].data.copy_(w2[j])
            self.b2[c].data.copy_(b2[j])


class ClusterwiseLongContextChannelHeadMLP(_ClusterPredictorBase):
    def __init__(
        self,
        num_clusters: int,
        input_len: int,
        tail_len: int,
        pred_len: int,
        hidden_dim: int,
        num_channels: int,
        cluster_id_c: torch.Tensor,
        dropout: float = 0.0,
        residual: bool = True,
        include_seasonal_profile: bool = False,
        output_mode: str = "direct",
        chunk_len: int = 96,
        anchor_points: Optional[int] = None,
        detail_scale: float = 0.25,
    ):
        super().__init__(num_clusters=num_clusters)
        self.L = int(input_len)
        self.tail_len = int(tail_len)
        self.H = int(pred_len)
        self.D = int(hidden_dim)
        self.C = int(num_channels)
        self.residual = bool(residual)
        self.include_seasonal_profile = bool(include_seasonal_profile)
        self.output_mode = str(output_mode or "direct").lower()
        if self.output_mode not in {"direct", "anchor"}:
            raise ValueError("long_context_channel_head_mlp output_mode must be 'direct' or 'anchor'.")
        self.chunk_len = max(int(chunk_len), 1)
        self.num_segments = int((self.H + self.chunk_len - 1) // self.chunk_len)
        default_anchor_points = int((self.H + self.chunk_len - 1) // self.chunk_len) + 1
        self.anchor_points = max(int(anchor_points or default_anchor_points), 2)
        self.detail_scale = float(detail_scale)
        self.context_dim = 10
        self.input_dim = self.tail_len + (self.tail_len if self.include_seasonal_profile else 0) + self.context_dim
        if self.tail_len <= 0:
            raise ValueError("long_context_channel_head_mlp requires positive predictor_input_len.")
        if self.tail_len > self.L:
            raise ValueError(
                f"predictor_input_len={self.tail_len} cannot exceed input_len={self.L}."
            )
        if cluster_id_c.numel() != self.C:
            raise ValueError(
                f"long_context_channel_head_mlp expected cluster_id_c length {self.C}, got {int(cluster_id_c.numel())}."
            )
        self.register_buffer("cluster_id_c", cluster_id_c.detach().long().cpu(), persistent=False)

        self.W1 = nn.ParameterList(
            [nn.Parameter(torch.empty(self.input_dim, self.D)) for _ in range(num_clusters)]
        )
        self.b1 = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.D)) for _ in range(num_clusters)]
        )
        if self.output_mode == "direct":
            self.W2 = nn.ParameterList(
                [nn.Parameter(torch.empty(self.D, self.H)) for _ in range(self.C)]
            )
            self.b2 = nn.ParameterList(
                [nn.Parameter(torch.zeros(self.H)) for _ in range(self.C)]
            )
            self.anchor_emb = nn.ParameterList()
            self.W_anchor = nn.ParameterList()
            self.b_anchor = nn.ParameterList()
            self.detail_emb = nn.ParameterList()
            self.W_detail = nn.ParameterList()
            self.b_detail = nn.ParameterList()
        else:
            self.W2 = nn.ParameterList()
            self.b2 = nn.ParameterList()
            self.anchor_emb = nn.ParameterList(
                [nn.Parameter(torch.empty(self.anchor_points, self.D)) for _ in range(num_clusters)]
            )
            self.W_anchor = nn.ParameterList(
                [nn.Parameter(torch.empty(self.D, 1)) for _ in range(self.C)]
            )
            self.b_anchor = nn.ParameterList(
                [nn.Parameter(torch.zeros(self.anchor_points)) for _ in range(self.C)]
            )
            self.detail_emb = nn.ParameterList(
                [nn.Parameter(torch.empty(self.num_segments, self.D)) for _ in range(num_clusters)]
            )
            self.W_detail = nn.ParameterList(
                [nn.Parameter(torch.empty(self.D, self.chunk_len)) for _ in range(self.C)]
            )
            self.b_detail = nn.ParameterList(
                [nn.Parameter(torch.zeros(self.num_segments, self.chunk_len)) for _ in range(self.C)]
            )
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        for w in self.W1:
            nn.init.xavier_uniform_(w)
        for w in self.W2:
            nn.init.xavier_uniform_(w)
        for emb in self.anchor_emb:
            nn.init.normal_(emb, mean=0.0, std=0.02)
        for w in self.W_anchor:
            nn.init.xavier_uniform_(w)
        for emb in self.detail_emb:
            nn.init.normal_(emb, mean=0.0, std=0.02)
        for w in self.W_detail:
            nn.init.xavier_uniform_(w)

    def _cluster_channel_idx(self, k: int) -> torch.Tensor:
        return (self.cluster_id_c == int(k)).nonzero(as_tuple=False).view(-1)

    def _lag_delta(self, x_bcl: torch.Tensor, lag: int, std_bc: torch.Tensor) -> torch.Tensor:
        idx = int(x_bcl.shape[-1]) - 1 - int(lag)
        if idx < 0:
            return torch.zeros_like(std_bc)
        return (x_bcl[..., -1] - x_bcl[..., idx]) / std_bc.clamp_min(1.0e-6)

    def _seasonal_acf(self, x_bcl: torch.Tensor, lag: int) -> torch.Tensor:
        lag = int(lag)
        if int(x_bcl.shape[-1]) <= lag:
            return x_bcl.new_zeros(x_bcl.shape[:2])
        a = x_bcl[..., lag:]
        b = x_bcl[..., :-lag]
        a = a - a.mean(dim=-1, keepdim=True)
        b = b - b.mean(dim=-1, keepdim=True)
        num = (a * b).mean(dim=-1)
        den = (a.pow(2).mean(dim=-1) * b.pow(2).mean(dim=-1)).sqrt()
        return num / den.clamp_min(1.0e-6)

    def _interpolate_anchor(self, anchor_bca: torch.Tensor) -> torch.Tensor:
        b, c, a = anchor_bca.shape
        curve = F.interpolate(
            anchor_bca.reshape(b * c, 1, a),
            size=self.H,
            mode="linear",
            align_corners=True,
        )
        return curve.reshape(b, c, self.H)

    def _context_features(self, x_bcl: torch.Tensor) -> torch.Tensor:
        eps = 1.0e-6
        B, C, L = x_bcl.shape
        tail = x_bcl[..., -self.tail_len :]
        last = x_bcl[..., -1:]
        if self.residual:
            tail_feat = tail - last
        else:
            tail_feat = tail
        parts = [tail_feat]

        if self.include_seasonal_profile:
            prev_cycles = max((L - self.tail_len) // self.tail_len, 0)
            if prev_cycles > 0:
                start = L - (prev_cycles + 1) * self.tail_len
                prev = x_bcl[..., start : L - self.tail_len]
                seasonal = prev.reshape(B, C, prev_cycles, self.tail_len).mean(dim=2)
                seasonal = seasonal - last if self.residual else seasonal
            else:
                seasonal = torch.zeros_like(tail_feat)
            parts.append(seasonal)

        mean = x_bcl.mean(dim=-1)
        std = x_bcl.std(dim=-1, unbiased=False).clamp_min(eps)
        range_over_std = (x_bcl.amax(dim=-1) - x_bcl.amin(dim=-1)) / std
        t_l = torch.linspace(-1.0, 1.0, steps=L, device=x_bcl.device, dtype=x_bcl.dtype).view(1, 1, L)
        slope = ((x_bcl - mean.unsqueeze(-1)) * t_l).mean(dim=-1) / t_l.pow(2).mean().clamp_min(eps)
        slope = slope / std
        mean_delta = (mean - last.squeeze(-1)) / std
        std_log = std.log()
        lag1 = self._lag_delta(x_bcl, self.tail_len, std)
        lag2 = self._lag_delta(x_bcl, 2 * self.tail_len, std)

        if L >= 2 * self.tail_len:
            prev = x_bcl[..., -2 * self.tail_len : -self.tail_len]
            prev_mean_delta = (tail.mean(dim=-1) - prev.mean(dim=-1)) / std
            prev_std_ratio = (
                tail.std(dim=-1, unbiased=False).clamp_min(eps)
                / prev.std(dim=-1, unbiased=False).clamp_min(eps)
            ).clamp(1.0e-3, 1.0e3).log()
        else:
            prev_mean_delta = torch.zeros_like(mean)
            prev_std_ratio = torch.zeros_like(mean)

        acf1 = self._seasonal_acf(x_bcl, self.tail_len)
        acf2 = self._seasonal_acf(x_bcl, 2 * self.tail_len)
        context = torch.stack(
            [
                mean_delta,
                std_log,
                range_over_std,
                slope,
                lag1,
                lag2,
                prev_mean_delta,
                prev_std_ratio,
                acf1,
                acf2,
            ],
            dim=-1,
        )
        context = torch.nan_to_num(context, nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0)
        parts.append(context)
        return torch.cat(parts, dim=-1)

    def forward(self, x_bcl: torch.Tensor, cluster_id_c: torch.Tensor) -> torch.Tensor:
        if cluster_id_c.numel() != self.C:
            raise ValueError(
                f"long_context_channel_head_mlp expected {self.C} channels, got {int(cluster_id_c.numel())}."
            )
        last = x_bcl[..., -1:]
        feat_bcd = self._context_features(x_bcl)
        W1 = torch.stack(list(self.W1), dim=0).index_select(0, cluster_id_c)
        b1 = torch.stack(list(self.b1), dim=0).index_select(0, cluster_id_c)

        h = torch.einsum("bcd,cdm->bcm", feat_bcd, W1) + b1.unsqueeze(0)
        h = self.drop(self.act(h))
        if self.output_mode == "direct":
            W2 = torch.stack(list(self.W2), dim=0)
            b2 = torch.stack(list(self.b2), dim=0)
            y = torch.einsum("bcm,cmh->bch", h, W2) + b2.unsqueeze(0)
        else:
            anchor_emb = torch.stack(list(self.anchor_emb), dim=0).index_select(0, cluster_id_c)
            W_anchor = torch.stack(list(self.W_anchor), dim=0)
            b_anchor = torch.stack(list(self.b_anchor), dim=0)
            h_anchor = h.unsqueeze(2) + anchor_emb.unsqueeze(0)
            anchor_bca = torch.einsum("bcam,cmo->bcao", h_anchor, W_anchor).squeeze(-1)
            anchor_bca = anchor_bca + b_anchor.unsqueeze(0)
            anchor_bch = self._interpolate_anchor(anchor_bca)

            detail_emb = torch.stack(list(self.detail_emb), dim=0).index_select(0, cluster_id_c)
            W_detail = torch.stack(list(self.W_detail), dim=0)
            b_detail = torch.stack(list(self.b_detail), dim=0)
            h_detail = h.unsqueeze(2) + detail_emb.unsqueeze(0)
            detail_bcsh = torch.einsum("bcsm,cmh->bcsh", h_detail, W_detail)
            detail_bcsh = detail_bcsh + b_detail.unsqueeze(0)
            detail_bcsh = detail_bcsh - detail_bcsh.mean(dim=-1, keepdim=True)
            detail_bch = detail_bcsh.reshape(h.shape[0], self.C, self.num_segments * self.chunk_len)[..., : self.H]
            y = anchor_bch + self.detail_scale * detail_bch
        return y + last if self.residual else y

    def get_cluster_params(self, k: int) -> List[nn.Parameter]:
        params: List[nn.Parameter] = [self.W1[k], self.b1[k]]
        idx = self._cluster_channel_idx(k)
        if self.output_mode == "direct":
            params.extend(self.W2[int(i.item())] for i in idx)
            params.extend(self.b2[int(i.item())] for i in idx)
        else:
            params.extend([self.anchor_emb[k], self.detail_emb[k]])
            params.extend(self.W_anchor[int(i.item())] for i in idx)
            params.extend(self.b_anchor[int(i.item())] for i in idx)
            params.extend(self.W_detail[int(i.item())] for i in idx)
            params.extend(self.b_detail[int(i.item())] for i in idx)
        return params

    def get_cluster_state(self, k: int):
        idx = self._cluster_channel_idx(k)
        state = {
            "W1": self.W1[k].detach().cpu(),
            "b1": self.b1[k].detach().cpu(),
            "channel_idx": idx.detach().cpu(),
            "output_mode": self.output_mode,
        }
        if self.output_mode == "direct":
            if idx.numel() > 0:
                state["W2"] = torch.stack([self.W2[int(i.item())].detach().cpu() for i in idx], dim=0)
                state["b2"] = torch.stack([self.b2[int(i.item())].detach().cpu() for i in idx], dim=0)
            else:
                state["W2"] = torch.empty(0, self.D, self.H)
                state["b2"] = torch.empty(0, self.H)
        else:
            state["anchor_emb"] = self.anchor_emb[k].detach().cpu()
            state["detail_emb"] = self.detail_emb[k].detach().cpu()
            if idx.numel() > 0:
                state["W_anchor"] = torch.stack([self.W_anchor[int(i.item())].detach().cpu() for i in idx], dim=0)
                state["b_anchor"] = torch.stack([self.b_anchor[int(i.item())].detach().cpu() for i in idx], dim=0)
                state["W_detail"] = torch.stack([self.W_detail[int(i.item())].detach().cpu() for i in idx], dim=0)
                state["b_detail"] = torch.stack([self.b_detail[int(i.item())].detach().cpu() for i in idx], dim=0)
            else:
                state["W_anchor"] = torch.empty(0, self.D, 1)
                state["b_anchor"] = torch.empty(0, self.anchor_points)
                state["W_detail"] = torch.empty(0, self.D, self.chunk_len)
                state["b_detail"] = torch.empty(0, self.num_segments, self.chunk_len)
        return state

    def load_cluster_state(self, k: int, state):
        device = self.W1[k].device
        self.W1[k].data.copy_(state["W1"].to(device))
        self.b1[k].data.copy_(state["b1"].to(device))
        idx = self._cluster_channel_idx(k)
        saved_idx = state.get("channel_idx", idx.detach().cpu())
        if saved_idx.numel() != idx.numel() or not torch.equal(saved_idx.cpu(), idx.detach().cpu()):
            raise ValueError(f"long_context_channel_head_mlp cluster {k} channel indices do not match checkpoint state.")
        if self.output_mode == "direct":
            w2 = state["W2"].to(device)
            b2 = state["b2"].to(device)
            for j, i in enumerate(idx):
                c = int(i.item())
                self.W2[c].data.copy_(w2[j])
                self.b2[c].data.copy_(b2[j])
        else:
            self.anchor_emb[k].data.copy_(state["anchor_emb"].to(device))
            self.detail_emb[k].data.copy_(state["detail_emb"].to(device))
            w_anchor = state["W_anchor"].to(device)
            b_anchor = state["b_anchor"].to(device)
            w_detail = state["W_detail"].to(device)
            b_detail = state["b_detail"].to(device)
            for j, i in enumerate(idx):
                c = int(i.item())
                self.W_anchor[c].data.copy_(w_anchor[j])
                self.b_anchor[c].data.copy_(b_anchor[j])
                self.W_detail[c].data.copy_(w_detail[j])
                self.b_detail[c].data.copy_(b_detail[j])


class ClusterwiseSeasonalityGatedChannelHeadMLP(_ClusterPredictorBase):
    def __init__(
        self,
        num_clusters: int,
        input_len: int,
        tail_len: int,
        pred_len: int,
        hidden_dim: int,
        num_channels: int,
        cluster_id_c: torch.Tensor,
        dropout: float = 0.0,
        residual: bool = True,
        include_seasonal_profile: bool = True,
        chunk_len: int = 96,
        anchor_points: Optional[int] = None,
        detail_scale: float = 0.25,
        mix_init: float = -2.0,
        gate_strength: float = 0.0,
        gate_threshold: float = 0.75,
    ):
        super().__init__(num_clusters=num_clusters)
        self.L = int(input_len)
        self.tail_len = int(tail_len)
        self.H = int(pred_len)
        self.C = int(num_channels)
        self.gate_strength = float(gate_strength)
        self.gate_threshold = float(gate_threshold)
        if cluster_id_c.numel() != self.C:
            raise ValueError(
                f"seasonality_gated_channel_head_mlp expected cluster_id_c length {self.C}, got {int(cluster_id_c.numel())}."
            )
        self.register_buffer("cluster_id_c", cluster_id_c.detach().long().cpu(), persistent=False)
        self.full_head = ClusterwiseChannelHeadMLP(
            num_clusters=num_clusters,
            input_len=input_len,
            pred_len=pred_len,
            hidden_dim=hidden_dim,
            num_channels=num_channels,
            cluster_id_c=cluster_id_c,
            dropout=dropout,
            residual=residual,
        )
        self.seasonal_head = ClusterwiseLongContextChannelHeadMLP(
            num_clusters=num_clusters,
            input_len=input_len,
            tail_len=tail_len,
            pred_len=pred_len,
            hidden_dim=hidden_dim,
            num_channels=num_channels,
            cluster_id_c=cluster_id_c,
            dropout=dropout,
            residual=residual,
            include_seasonal_profile=include_seasonal_profile,
            output_mode="anchor",
            chunk_len=chunk_len,
            anchor_points=anchor_points,
            detail_scale=detail_scale,
        )
        self.mix_logit_c = nn.Parameter(torch.full((self.C,), float(mix_init)))

    def _seasonality_score(self, x_bcl: torch.Tensor) -> torch.Tensor:
        if self.gate_strength == 0.0:
            return x_bcl.new_zeros(x_bcl.shape[:2])
        acf = self.seasonal_head._seasonal_acf(x_bcl, self.tail_len)
        acf = torch.nan_to_num(acf, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        return self.gate_strength * (acf - self.gate_threshold)

    def seasonal_mix(self, x_bcl: torch.Tensor) -> torch.Tensor:
        logits = self.mix_logit_c.to(device=x_bcl.device, dtype=x_bcl.dtype).view(1, self.C)
        logits = logits + self._seasonality_score(x_bcl)
        return torch.sigmoid(logits)

    def forward(self, x_bcl: torch.Tensor, cluster_id_c: torch.Tensor) -> torch.Tensor:
        y_full = self.full_head(x_bcl, cluster_id_c)
        y_seasonal = self.seasonal_head(x_bcl, cluster_id_c)
        mix = self.seasonal_mix(x_bcl).unsqueeze(-1)
        return y_full + mix * (y_seasonal - y_full)

    def _cluster_channel_idx(self, k: int) -> torch.Tensor:
        return (self.cluster_id_c == int(k)).nonzero(as_tuple=False).view(-1)

    def get_cluster_params(self, k: int) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        params.extend(self.full_head.get_cluster_params(k))
        params.extend(self.seasonal_head.get_cluster_params(k))
        params.append(self.mix_logit_c)
        return params

    def get_cluster_state(self, k: int):
        idx = self._cluster_channel_idx(k)
        return {
            "full_head": self.full_head.get_cluster_state(k),
            "seasonal_head": self.seasonal_head.get_cluster_state(k),
            "channel_idx": idx.detach().cpu(),
            "mix_logit": self.mix_logit_c.detach().cpu().index_select(0, idx.detach().cpu()),
        }

    def load_cluster_state(self, k: int, state):
        self.full_head.load_cluster_state(k, state["full_head"])
        self.seasonal_head.load_cluster_state(k, state["seasonal_head"])
        idx = self._cluster_channel_idx(k)
        saved_idx = state.get("channel_idx", idx.detach().cpu())
        if saved_idx.numel() != idx.numel() or not torch.equal(saved_idx.cpu(), idx.detach().cpu()):
            raise ValueError(f"seasonality_gated_channel_head_mlp cluster {k} channel indices do not match checkpoint state.")
        self.mix_logit_c.data.index_copy_(0, idx.to(self.mix_logit_c.device), state["mix_logit"].to(self.mix_logit_c.device))


class ClusterwiseAttnMLP(_ClusterPredictorBase):
    def __init__(
        self,
        num_clusters: int,
        input_len: int,
        pred_len: int,
        hidden_dim: int,
        dropout: float = 0.0,
        attn_dim: int = 64,
    ):
        super().__init__(num_clusters=num_clusters)
        self.L = input_len
        self.H = pred_len
        self.D = hidden_dim
        self.A = max(int(attn_dim), 8)

        self.Wq = nn.ParameterList(
            [nn.Parameter(torch.empty(input_len, self.A)) for _ in range(num_clusters)]
        )
        self.bq = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.A)) for _ in range(num_clusters)]
        )
        self.Wk = nn.ParameterList(
            [nn.Parameter(torch.empty(input_len, self.A)) for _ in range(num_clusters)]
        )
        self.bk = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.A)) for _ in range(num_clusters)]
        )
        self.W1 = nn.ParameterList(
            [nn.Parameter(torch.empty(input_len * 3, hidden_dim)) for _ in range(num_clusters)]
        )
        self.b1 = nn.ParameterList(
            [nn.Parameter(torch.zeros(hidden_dim)) for _ in range(num_clusters)]
        )
        self.W2 = nn.ParameterList(
            [nn.Parameter(torch.empty(hidden_dim, pred_len)) for _ in range(num_clusters)]
        )
        self.b2 = nn.ParameterList(
            [nn.Parameter(torch.zeros(pred_len)) for _ in range(num_clusters)]
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        for plist in [self.Wq, self.Wk, self.W1, self.W2]:
            for w in plist:
                nn.init.xavier_uniform_(w)

    def forward(self, x_bcl: torch.Tensor, cluster_id_c: torch.Tensor) -> torch.Tensor:
        b, c, _ = x_bcl.shape
        y_bch = x_bcl.new_zeros((b, c, self.H))
        scale = float(self.A) ** -0.5
        for k in range(self.K):
            idx = (cluster_id_c == k).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            x_bnl = x_bcl.index_select(1, idx)  # [B, N, L]
            last = x_bnl[..., -1:]
            x_center = x_bnl - last

            q = torch.einsum("bnl,la->bna", x_center, self.Wq[k]) + self.bq[k].view(1, 1, -1)
            kvec = torch.einsum("bnl,la->bna", x_center, self.Wk[k]) + self.bk[k].view(1, 1, -1)
            scores = torch.einsum("bna,bma->bnm", q, kvec) * scale
            attn = torch.softmax(scores, dim=-1)
            context = torch.einsum("bnm,bml->bnl", attn, x_center)

            feat = torch.cat([x_center, context, x_center - context], dim=-1)
            h = torch.einsum("bnf,fd->bnd", feat, self.W1[k]) + self.b1[k].view(1, 1, -1)
            h = self.drop(self.act(h))
            delta = torch.einsum("bnd,dh->bnh", h, self.W2[k]) + self.b2[k].view(1, 1, -1)
            y_bnh = delta + last
            y_bch = y_bch.index_copy(1, idx, y_bnh)
        return y_bch

    def get_cluster_params(self, k: int) -> List[nn.Parameter]:
        return [self.Wq[k], self.bq[k], self.Wk[k], self.bk[k], self.W1[k], self.b1[k], self.W2[k], self.b2[k]]

    def get_cluster_state(self, k: int):
        return {
            "Wq": self.Wq[k].detach().cpu(),
            "bq": self.bq[k].detach().cpu(),
            "Wk": self.Wk[k].detach().cpu(),
            "bk": self.bk[k].detach().cpu(),
            "W1": self.W1[k].detach().cpu(),
            "b1": self.b1[k].detach().cpu(),
            "W2": self.W2[k].detach().cpu(),
            "b2": self.b2[k].detach().cpu(),
        }

    def load_cluster_state(self, k: int, state):
        device = self.Wq[k].device
        self.Wq[k].data.copy_(state["Wq"].to(device))
        self.bq[k].data.copy_(state["bq"].to(device))
        self.Wk[k].data.copy_(state["Wk"].to(device))
        self.bk[k].data.copy_(state["bk"].to(device))
        self.W1[k].data.copy_(state["W1"].to(device))
        self.b1[k].data.copy_(state["b1"].to(device))
        self.W2[k].data.copy_(state["W2"].to(device))
        self.b2[k].data.copy_(state["b2"].to(device))


class _MovingAvg(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = max(int(kernel_size), 1)

    def forward(self, x_bcl: torch.Tensor) -> torch.Tensor:
        if self.kernel_size <= 1:
            return x_bcl
        pad_left = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size - 1 - pad_left
        x = x_bcl.reshape(-1, 1, x_bcl.shape[-1])
        x = F.pad(x, (pad_left, pad_right), mode="replicate")
        x = F.avg_pool1d(x, kernel_size=self.kernel_size, stride=1)
        return x.reshape_as(x_bcl)


class ClusterwiseDLinear(_ClusterPredictorBase):
    def __init__(self, num_clusters: int, input_len: int, pred_len: int, kernel_size: int = 25):
        super().__init__(num_clusters=num_clusters)
        self.L = input_len
        self.H = pred_len
        self.moving_avg = _MovingAvg(kernel_size=kernel_size)
        self.W_season = nn.ParameterList(
            [nn.Parameter(torch.empty(input_len, pred_len)) for _ in range(num_clusters)]
        )
        self.b_season = nn.ParameterList(
            [nn.Parameter(torch.zeros(pred_len)) for _ in range(num_clusters)]
        )
        self.W_trend = nn.ParameterList(
            [nn.Parameter(torch.empty(input_len, pred_len)) for _ in range(num_clusters)]
        )
        self.b_trend = nn.ParameterList(
            [nn.Parameter(torch.zeros(pred_len)) for _ in range(num_clusters)]
        )
        self.reset_parameters()

    def reset_parameters(self):
        for w in self.W_season:
            nn.init.xavier_uniform_(w)
        for w in self.W_trend:
            nn.init.xavier_uniform_(w)

    def forward(self, x_bcl: torch.Tensor, cluster_id_c: torch.Tensor) -> torch.Tensor:
        trend = self.moving_avg(x_bcl)
        season = x_bcl - trend
        W_season = torch.stack(list(self.W_season), dim=0).index_select(0, cluster_id_c)
        b_season = torch.stack(list(self.b_season), dim=0).index_select(0, cluster_id_c)
        W_trend = torch.stack(list(self.W_trend), dim=0).index_select(0, cluster_id_c)
        b_trend = torch.stack(list(self.b_trend), dim=0).index_select(0, cluster_id_c)
        y_season = torch.einsum("bcl,clh->bch", season, W_season) + b_season.unsqueeze(0)
        y_trend = torch.einsum("bcl,clh->bch", trend, W_trend) + b_trend.unsqueeze(0)
        return y_season + y_trend

    def get_cluster_params(self, k: int) -> List[nn.Parameter]:
        return [
            self.W_season[k],
            self.b_season[k],
            self.W_trend[k],
            self.b_trend[k],
        ]

    def get_cluster_state(self, k: int):
        return {
            "W_season": self.W_season[k].detach().cpu(),
            "b_season": self.b_season[k].detach().cpu(),
            "W_trend": self.W_trend[k].detach().cpu(),
            "b_trend": self.b_trend[k].detach().cpu(),
        }

    def load_cluster_state(self, k: int, state):
        device = self.W_season[k].device
        self.W_season[k].data.copy_(state["W_season"].to(device))
        self.b_season[k].data.copy_(state["b_season"].to(device))
        self.W_trend[k].data.copy_(state["W_trend"].to(device))
        self.b_trend[k].data.copy_(state["b_trend"].to(device))


class ClusterwiseChannelDLinear(_ClusterPredictorBase):
    def __init__(
        self,
        num_clusters: int,
        input_len: int,
        pred_len: int,
        num_channels: int,
        cluster_id_c: torch.Tensor,
        kernel_size: int = 25,
    ):
        super().__init__(num_clusters=num_clusters)
        self.L = int(input_len)
        self.H = int(pred_len)
        self.C = int(num_channels)
        if cluster_id_c.numel() != self.C:
            raise ValueError(
                f"channel_dlinear expected cluster_id_c length {self.C}, got {int(cluster_id_c.numel())}."
            )
        self.register_buffer("cluster_id_c", cluster_id_c.detach().long().cpu(), persistent=False)
        self.moving_avg = _MovingAvg(kernel_size=kernel_size)
        self.W_season = nn.ParameterList(
            [nn.Parameter(torch.empty(self.L, self.H)) for _ in range(self.C)]
        )
        self.b_season = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.H)) for _ in range(self.C)]
        )
        self.W_trend = nn.ParameterList(
            [nn.Parameter(torch.empty(self.L, self.H)) for _ in range(self.C)]
        )
        self.b_trend = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.H)) for _ in range(self.C)]
        )
        self.reset_parameters()

    def reset_parameters(self):
        for w in self.W_season:
            nn.init.xavier_uniform_(w)
        for w in self.W_trend:
            nn.init.xavier_uniform_(w)

    def _cluster_channel_idx(self, k: int) -> torch.Tensor:
        return (self.cluster_id_c == int(k)).nonzero(as_tuple=False).view(-1)

    def forward(self, x_bcl: torch.Tensor, cluster_id_c: torch.Tensor) -> torch.Tensor:
        if x_bcl.shape[1] != self.C:
            raise ValueError(f"channel_dlinear expected {self.C} channels, got {int(x_bcl.shape[1])}.")
        trend = self.moving_avg(x_bcl)
        season = x_bcl - trend
        W_season = torch.stack(list(self.W_season), dim=0)
        b_season = torch.stack(list(self.b_season), dim=0)
        W_trend = torch.stack(list(self.W_trend), dim=0)
        b_trend = torch.stack(list(self.b_trend), dim=0)
        y_season = torch.einsum("bcl,clh->bch", season, W_season) + b_season.unsqueeze(0)
        y_trend = torch.einsum("bcl,clh->bch", trend, W_trend) + b_trend.unsqueeze(0)
        return y_season + y_trend

    def get_cluster_params(self, k: int) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for i in self._cluster_channel_idx(k):
            c = int(i.item())
            params.extend([self.W_season[c], self.b_season[c], self.W_trend[c], self.b_trend[c]])
        return params

    def get_cluster_state(self, k: int):
        idx = self._cluster_channel_idx(k)
        if idx.numel() > 0:
            w_season = torch.stack([self.W_season[int(i.item())].detach().cpu() for i in idx], dim=0)
            b_season = torch.stack([self.b_season[int(i.item())].detach().cpu() for i in idx], dim=0)
            w_trend = torch.stack([self.W_trend[int(i.item())].detach().cpu() for i in idx], dim=0)
            b_trend = torch.stack([self.b_trend[int(i.item())].detach().cpu() for i in idx], dim=0)
        else:
            w_season = torch.empty(0, self.L, self.H)
            b_season = torch.empty(0, self.H)
            w_trend = torch.empty(0, self.L, self.H)
            b_trend = torch.empty(0, self.H)
        return {
            "channel_idx": idx.detach().cpu(),
            "W_season": w_season,
            "b_season": b_season,
            "W_trend": w_trend,
            "b_trend": b_trend,
        }

    def load_cluster_state(self, k: int, state):
        idx = self._cluster_channel_idx(k)
        saved_idx = state.get("channel_idx", idx.detach().cpu())
        if saved_idx.numel() != idx.numel() or not torch.equal(saved_idx.cpu(), idx.detach().cpu()):
            raise ValueError(f"channel_dlinear cluster {k} channel indices do not match checkpoint state.")
        device = self.W_season[0].device
        w_season = state["W_season"].to(device)
        b_season = state["b_season"].to(device)
        w_trend = state["W_trend"].to(device)
        b_trend = state["b_trend"].to(device)
        for j, i in enumerate(idx):
            c = int(i.item())
            self.W_season[c].data.copy_(w_season[j])
            self.b_season[c].data.copy_(b_season[j])
            self.W_trend[c].data.copy_(w_trend[j])
            self.b_trend[c].data.copy_(b_trend[j])


class ClusterwiseTemporalBasisAdapter(_ClusterPredictorBase):
    def __init__(
        self,
        base: _ClusterPredictorBase,
        rank: int = 16,
        scale: float = 1.0,
        init: str = "zero_delta",
        freeze_base: bool = False,
    ):
        super().__init__(num_clusters=base.K)
        self.base = base
        self.L = int(base.L)
        self.H = int(base.H)
        self.R = max(int(rank), 1)
        self.scale = float(scale)
        self.freeze_base = bool(freeze_base)
        if self.freeze_base:
            for p in self.base.parameters():
                p.requires_grad_(False)

        self.W = nn.ParameterList(
            [nn.Parameter(torch.empty(self.L, self.R)) for _ in range(self.K)]
        )
        self.b = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.R)) for _ in range(self.K)]
        )
        self.register_buffer("basis_rh", self._make_dct_basis(self.R, self.H), persistent=False)
        self.reset_parameters(init=str(init))

    @staticmethod
    def _make_dct_basis(rank: int, pred_len: int) -> torch.Tensor:
        t = torch.arange(pred_len, dtype=torch.float32).view(1, -1)
        r = torch.arange(rank, dtype=torch.float32).view(-1, 1)
        basis = torch.cos(torch.pi * (t + 0.5) * r / float(max(pred_len, 1)))
        basis = basis / basis.pow(2).mean(dim=1, keepdim=True).clamp_min(1.0e-8).sqrt()
        return basis

    def reset_parameters(self, init: str = "zero_delta"):
        init = str(init).lower()
        if init == "zero_delta":
            for w in self.W:
                nn.init.zeros_(w)
            for b in self.b:
                nn.init.zeros_(b)
            return
        for w in self.W:
            nn.init.xavier_uniform_(w)
        for b in self.b:
            nn.init.zeros_(b)

    def forward(self, x_bcl: torch.Tensor, cluster_id_c: torch.Tensor) -> torch.Tensor:
        base_y = self.base(x_bcl, cluster_id_c)
        last = x_bcl[..., -1:]
        x_center = x_bcl - last
        W = torch.stack(list(self.W), dim=0).index_select(0, cluster_id_c)
        b = torch.stack(list(self.b), dim=0).index_select(0, cluster_id_c)
        coeff_bcr = torch.einsum("bcl,clr->bcr", x_center, W) + b.unsqueeze(0)
        corr_bch = torch.einsum("bcr,rh->bch", coeff_bcr, self.basis_rh.to(dtype=x_bcl.dtype))
        return base_y + (self.scale * corr_bch)

    def get_cluster_params(self, k: int) -> List[nn.Parameter]:
        params = [] if self.freeze_base else self.base.get_cluster_params(k)
        return params + [self.W[k], self.b[k]]

    def get_cluster_state(self, k: int):
        return {
            "base": self.base.get_cluster_state(k),
            "W": self.W[k].detach().cpu(),
            "b": self.b[k].detach().cpu(),
        }

    def load_cluster_state(self, k: int, state):
        device = self.W[k].device
        if isinstance(state, dict) and "base" in state:
            self.base.load_cluster_state(k, state["base"])
            self.W[k].data.copy_(state["W"].to(device))
            self.b[k].data.copy_(state["b"].to(device))
        else:
            self.base.load_cluster_state(k, state)


class _PatchTSTExpert(nn.Module):
    def __init__(
        self,
        input_len: int,
        pred_len: int,
        patch_len: int,
        patch_stride: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.L = input_len
        self.H = pred_len
        self.patch_len = max(int(patch_len), 2)
        self.patch_stride = max(int(patch_stride), 1)
        self.num_patches = 1 + max(0, (self.L - self.patch_len) // self.patch_stride)
        if self.num_patches <= 0:
            raise ValueError("PatchTST requires input_len >= patch_len.")

        self.patch_proj = nn.Linear(self.patch_len, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(self.num_patches * d_model, pred_len)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.patch_proj.weight)
        nn.init.zeros_(self.patch_proj.bias)
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x_bli: torch.Tensor) -> torch.Tensor:
        x = x_bli.squeeze(-1)  # [B, L]
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp_min(1.0e-5)
        x = (x - mean) / std
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.patch_stride)  # [B, N, P]
        tok = self.patch_proj(patches) + self.pos_emb[:, :patches.shape[1], :]
        tok = self.encoder(tok)
        tok = self.drop(self.norm(tok))
        y = self.head(tok.reshape(tok.shape[0], -1))
        return y * std + mean


class ClusterwisePatchTST(_ClusterPredictorBase):
    def __init__(
        self,
        num_clusters: int,
        input_len: int,
        pred_len: int,
        d_model: int,
        dropout: float = 0.0,
        patch_len: int = 16,
        patch_stride: int = 8,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 256,
    ):
        super().__init__(num_clusters=num_clusters)
        self.L = input_len
        self.H = pred_len
        self.experts = nn.ModuleList(
            [
                _PatchTSTExpert(
                    input_len=input_len,
                    pred_len=pred_len,
                    patch_len=patch_len,
                    patch_stride=patch_stride,
                    d_model=d_model,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_clusters)
            ]
        )

    def forward(self, x_bcl: torch.Tensor, cluster_id_c: torch.Tensor) -> torch.Tensor:
        b, c, _ = x_bcl.shape
        y_bch = x_bcl.new_zeros((b, c, self.H))
        for k in range(self.K):
            idx = (cluster_id_c == k).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            x_bnl = x_bcl.index_select(1, idx)
            x_bnl = x_bnl.reshape(-1, self.L).unsqueeze(-1)
            y_bn = self.experts[k](x_bnl)
            y_bnh = y_bn.view(b, idx.numel(), self.H)
            y_bch = y_bch.index_copy(1, idx, y_bnh)
        return y_bch

    def get_cluster_params(self, k: int) -> List[nn.Parameter]:
        return list(self.experts[k].parameters())

    def get_cluster_state(self, k: int):
        return {n: t.detach().cpu() for n, t in self.experts[k].state_dict().items()}

    def load_cluster_state(self, k: int, state):
        device = next(self.experts[k].parameters()).device
        state_dev = {n: t.to(device) for n, t in state.items()}
        self.experts[k].load_state_dict(state_dev, strict=True)


class _NBEATSBlock(nn.Module):
    def __init__(self, input_len: int, pred_len: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        layers = []
        in_dim = input_len
        for _ in range(max(int(num_layers), 1)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.backcast_head = nn.Linear(hidden_dim, input_len)
        self.forecast_head = nn.Linear(hidden_dim, pred_len)
        self.reset_parameters()

    def reset_parameters(self):
        for mod in self.backbone:
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                nn.init.zeros_(mod.bias)
        nn.init.xavier_uniform_(self.backcast_head.weight)
        nn.init.zeros_(self.backcast_head.bias)
        nn.init.xavier_uniform_(self.forecast_head.weight)
        nn.init.zeros_(self.forecast_head.bias)

    def forward(self, x_bl: torch.Tensor):
        h = self.backbone(x_bl)
        backcast = self.backcast_head(h)
        forecast = self.forecast_head(h)
        return backcast, forecast


class _NBEATSExpert(nn.Module):
    def __init__(
        self,
        input_len: int,
        pred_len: int,
        hidden_dim: int,
        num_blocks: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                _NBEATSBlock(
                    input_len=input_len,
                    pred_len=pred_len,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                )
                for _ in range(max(int(num_blocks), 1))
            ]
        )

    def forward(self, x_bli: torch.Tensor) -> torch.Tensor:
        x = x_bli.squeeze(-1)
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp_min(1.0e-5)
        residual = (x - mean) / std
        forecast = torch.zeros((x.shape[0], self.blocks[0].forecast_head.out_features), device=x.device, dtype=x.dtype)
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast
        return forecast * std + mean


class ClusterwiseNBEATS(_ClusterPredictorBase):
    def __init__(
        self,
        num_clusters: int,
        input_len: int,
        pred_len: int,
        hidden_dim: int,
        dropout: float = 0.0,
        num_blocks: int = 4,
        num_layers: int = 4,
    ):
        super().__init__(num_clusters=num_clusters)
        self.L = input_len
        self.H = pred_len
        self.experts = nn.ModuleList(
            [
                _NBEATSExpert(
                    input_len=input_len,
                    pred_len=pred_len,
                    hidden_dim=hidden_dim,
                    num_blocks=num_blocks,
                    num_layers=num_layers,
                    dropout=dropout,
                )
                for _ in range(num_clusters)
            ]
        )

    def forward(self, x_bcl: torch.Tensor, cluster_id_c: torch.Tensor) -> torch.Tensor:
        b, c, _ = x_bcl.shape
        y_bch = x_bcl.new_zeros((b, c, self.H))
        for k in range(self.K):
            idx = (cluster_id_c == k).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            x_bnl = x_bcl.index_select(1, idx)
            x_bnl = x_bnl.reshape(-1, self.L).unsqueeze(-1)
            y_bn = self.experts[k](x_bnl)
            y_bnh = y_bn.view(b, idx.numel(), self.H)
            y_bch = y_bch.index_copy(1, idx, y_bnh)
        return y_bch

    def get_cluster_params(self, k: int) -> List[nn.Parameter]:
        return list(self.experts[k].parameters())

    def get_cluster_state(self, k: int):
        return {n: t.detach().cpu() for n, t in self.experts[k].state_dict().items()}

    def load_cluster_state(self, k: int, state):
        device = next(self.experts[k].parameters()).device
        state_dev = {n: t.to(device) for n, t in state.items()}
        self.experts[k].load_state_dict(state_dev, strict=True)


class _GRUExpert(nn.Module):
    def __init__(self, hidden_dim: int, pred_len: int, dropout: float, num_layers: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(hidden_dim, pred_len)

    def forward(self, x_bli: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x_bli)  # [B, L, D]
        h = self.drop(out[:, -1, :])  # [B, D]
        return self.head(h)  # [B, H]


class ClusterwiseGRU(_ClusterPredictorBase):
    def __init__(
        self,
        num_clusters: int,
        input_len: int,
        pred_len: int,
        hidden_dim: int,
        dropout: float = 0.0,
        num_layers: int = 1,
    ):
        super().__init__(num_clusters=num_clusters)
        self.L = input_len
        self.H = pred_len
        self.experts = nn.ModuleList(
            [
                _GRUExpert(
                    hidden_dim=hidden_dim,
                    pred_len=pred_len,
                    dropout=dropout,
                    num_layers=num_layers,
                )
                for _ in range(num_clusters)
            ]
        )

    def forward(self, x_bcl: torch.Tensor, cluster_id_c: torch.Tensor) -> torch.Tensor:
        b, c, _ = x_bcl.shape
        y_bch = x_bcl.new_zeros((b, c, self.H))
        for k in range(self.K):
            idx = (cluster_id_c == k).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            x_bnl = x_bcl.index_select(1, idx)  # [B,N,L]
            x_bnl = x_bnl.reshape(-1, self.L).unsqueeze(-1)  # [B*N,L,1]
            y_bn = self.experts[k](x_bnl)  # [B*N,H]
            y_bnh = y_bn.view(b, idx.numel(), self.H)  # [B,N,H]
            y_bch = y_bch.index_copy(1, idx, y_bnh)
        return y_bch

    def get_cluster_params(self, k: int) -> List[nn.Parameter]:
        return list(self.experts[k].parameters())

    def get_cluster_state(self, k: int):
        return {n: t.detach().cpu() for n, t in self.experts[k].state_dict().items()}

    def load_cluster_state(self, k: int, state):
        device = next(self.experts[k].parameters()).device
        state_dev = {n: t.to(device) for n, t in state.items()}
        self.experts[k].load_state_dict(state_dev, strict=True)


class _LSTMExpert(nn.Module):
    def __init__(self, hidden_dim: int, pred_len: int, dropout: float, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(hidden_dim, pred_len)

    def forward(self, x_bli: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x_bli)
        h = self.drop(out[:, -1, :])
        return self.head(h)


class ClusterwiseLSTM(_ClusterPredictorBase):
    def __init__(
        self,
        num_clusters: int,
        input_len: int,
        pred_len: int,
        hidden_dim: int,
        dropout: float = 0.0,
        num_layers: int = 1,
    ):
        super().__init__(num_clusters=num_clusters)
        self.L = input_len
        self.H = pred_len
        self.experts = nn.ModuleList(
            [
                _LSTMExpert(
                    hidden_dim=hidden_dim,
                    pred_len=pred_len,
                    dropout=dropout,
                    num_layers=num_layers,
                )
                for _ in range(num_clusters)
            ]
        )

    def forward(self, x_bcl: torch.Tensor, cluster_id_c: torch.Tensor) -> torch.Tensor:
        b, c, _ = x_bcl.shape
        y_bch = x_bcl.new_zeros((b, c, self.H))
        for k in range(self.K):
            idx = (cluster_id_c == k).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            x_bnl = x_bcl.index_select(1, idx)
            x_bnl = x_bnl.reshape(-1, self.L).unsqueeze(-1)
            y_bn = self.experts[k](x_bnl)
            y_bnh = y_bn.view(b, idx.numel(), self.H)
            y_bch = y_bch.index_copy(1, idx, y_bnh)
        return y_bch

    def get_cluster_params(self, k: int) -> List[nn.Parameter]:
        return list(self.experts[k].parameters())

    def get_cluster_state(self, k: int):
        return {n: t.detach().cpu() for n, t in self.experts[k].state_dict().items()}

    def load_cluster_state(self, k: int, state):
        device = next(self.experts[k].parameters()).device
        state_dev = {n: t.to(device) for n, t in state.items()}
        self.experts[k].load_state_dict(state_dev, strict=True)


class ClusterwiseChannelLSTMMixer(_ClusterPredictorBase):
    def __init__(
        self,
        num_clusters: int,
        input_len: int,
        pred_len: int,
        hidden_dim: int,
        num_channels: int,
        cluster_id_c: torch.Tensor,
        dropout: float = 0.0,
        lstm_num_layers: int = 1,
        lstm_hidden_dim: Optional[int] = None,
        mix_init: float = -2.0,
        hard_lstm_channels: Optional[List[int]] = None,
    ):
        super().__init__(num_clusters=num_clusters)
        self.L = int(input_len)
        self.H = int(pred_len)
        self.C = int(num_channels)
        if cluster_id_c.numel() != self.C:
            raise ValueError(
                f"channel_lstm_mixer expected cluster_id_c length {self.C}, got {int(cluster_id_c.numel())}."
            )
        self.register_buffer("cluster_id_c", cluster_id_c.detach().long().cpu(), persistent=False)
        self.channel_head = ClusterwiseChannelHeadMLP(
            num_clusters=num_clusters,
            input_len=input_len,
            pred_len=pred_len,
            hidden_dim=hidden_dim,
            num_channels=num_channels,
            cluster_id_c=cluster_id_c,
            dropout=dropout,
            residual=True,
        )
        lstm_base = ClusterwiseLSTM(
            num_clusters=num_clusters,
            input_len=input_len,
            pred_len=pred_len,
            hidden_dim=int(lstm_hidden_dim or hidden_dim),
            dropout=dropout,
            num_layers=lstm_num_layers,
        )
        self.lstm_revin = ClusterwiseRevIN(base=lstm_base)
        self.mix_logit_c = nn.Parameter(torch.full((self.C,), float(mix_init)))
        if hard_lstm_channels is None:
            hard_mask = torch.full((self.C,), float("nan"), dtype=torch.float32)
        else:
            hard_mask = torch.zeros((self.C,), dtype=torch.float32)
            for c in hard_lstm_channels:
                if 0 <= int(c) < self.C:
                    hard_mask[int(c)] = 1.0
        self.register_buffer("hard_lstm_mask_c", hard_mask, persistent=False)

    def forward(self, x_bcl: torch.Tensor, cluster_id_c: torch.Tensor) -> torch.Tensor:
        y_channel = self.channel_head(x_bcl, cluster_id_c)
        y_lstm = self.lstm_revin(x_bcl, cluster_id_c)
        if torch.isfinite(self.hard_lstm_mask_c).all():
            mix = self.hard_lstm_mask_c.to(device=x_bcl.device, dtype=x_bcl.dtype).view(1, self.C, 1)
        else:
            mix = torch.sigmoid(self.mix_logit_c.to(device=x_bcl.device, dtype=x_bcl.dtype)).view(1, self.C, 1)
        return y_channel + mix * (y_lstm - y_channel)

    def _cluster_channel_idx(self, k: int) -> torch.Tensor:
        return (self.cluster_id_c == int(k)).nonzero(as_tuple=False).view(-1)

    def get_cluster_params(self, k: int) -> List[nn.Parameter]:
        params = []
        params.extend(self.channel_head.get_cluster_params(k))
        params.extend(self.lstm_revin.get_cluster_params(k))
        params.append(self.mix_logit_c)
        return params

    def get_cluster_state(self, k: int):
        return {
            "channel_head": self.channel_head.get_cluster_state(k),
            "lstm_revin": self.lstm_revin.get_cluster_state(k),
            "mix_logit_c": self.mix_logit_c.detach().cpu(),
        }

    def load_cluster_state(self, k: int, state):
        self.channel_head.load_cluster_state(k, state["channel_head"])
        self.lstm_revin.load_cluster_state(k, state["lstm_revin"])
        self.mix_logit_c.data.copy_(state["mix_logit_c"].to(self.mix_logit_c.device))


class _Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


class _TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = _Chomp1d(padding)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = _Chomp1d(padding)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.out_act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop1(self.act1(self.chomp1(self.conv1(x))))
        out = self.drop2(self.act2(self.chomp2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.out_act(out + res)


class _TCNExpert(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        pred_len: int,
        dropout: float,
        levels: int,
        kernel_size: int,
        dilation_base: int,
    ):
        super().__init__()
        layers = []
        in_ch = 1
        for i in range(levels):
            dilation = int(dilation_base ** i)
            layers.append(
                _TemporalBlock(
                    in_ch=in_ch,
                    out_ch=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_ch = hidden_dim
        self.tcn = nn.Sequential(*layers)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(hidden_dim, pred_len)

    def forward(self, x_bli: torch.Tensor) -> torch.Tensor:
        x_bil = x_bli.transpose(1, 2)  # [B,1,L]
        h_bdl = self.tcn(x_bil)  # [B,D,L]
        h_bd = self.drop(h_bdl[:, :, -1])  # [B,D]
        return self.head(h_bd)  # [B,H]


class ClusterwiseTCN(_ClusterPredictorBase):
    def __init__(
        self,
        num_clusters: int,
        input_len: int,
        pred_len: int,
        hidden_dim: int,
        dropout: float = 0.0,
        levels: int = 2,
        kernel_size: int = 3,
        dilation_base: int = 2,
    ):
        super().__init__(num_clusters=num_clusters)
        self.L = input_len
        self.H = pred_len
        self.experts = nn.ModuleList(
            [
                _TCNExpert(
                    hidden_dim=hidden_dim,
                    pred_len=pred_len,
                    dropout=dropout,
                    levels=levels,
                    kernel_size=kernel_size,
                    dilation_base=dilation_base,
                )
                for _ in range(num_clusters)
            ]
        )

    def forward(self, x_bcl: torch.Tensor, cluster_id_c: torch.Tensor) -> torch.Tensor:
        b, c, _ = x_bcl.shape
        y_bch = x_bcl.new_zeros((b, c, self.H))
        for k in range(self.K):
            idx = (cluster_id_c == k).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            x_bnl = x_bcl.index_select(1, idx)  # [B,N,L]
            x_bnl = x_bnl.reshape(-1, self.L).unsqueeze(-1)  # [B*N,L,1]
            y_bn = self.experts[k](x_bnl)  # [B*N,H]
            y_bnh = y_bn.view(b, idx.numel(), self.H)  # [B,N,H]
            y_bch = y_bch.index_copy(1, idx, y_bnh)
        return y_bch

    def get_cluster_params(self, k: int) -> List[nn.Parameter]:
        return list(self.experts[k].parameters())

    def get_cluster_state(self, k: int):
        return {n: t.detach().cpu() for n, t in self.experts[k].state_dict().items()}

    def load_cluster_state(self, k: int, state):
        device = next(self.experts[k].parameters()).device
        state_dev = {n: t.to(device) for n, t in state.items()}
        self.experts[k].load_state_dict(state_dev, strict=True)


class ClusterwiseChannelResidualAdapter(_ClusterPredictorBase):
    def __init__(
        self,
        base: _ClusterPredictorBase,
        num_channels: int,
        cluster_id_c: torch.Tensor,
        rank: int,
        init: str = "zero_delta",
        scale: float = 1.0,
    ):
        super().__init__(num_clusters=base.K)
        self.base = base
        self.L = int(base.L)
        self.H = int(base.H)
        self.num_channels = int(num_channels)
        self.rank = max(int(rank), 1)
        self.scale = float(scale)

        cluster_id_c = cluster_id_c.detach().cpu().to(torch.long)
        if int(cluster_id_c.numel()) != self.num_channels:
            raise ValueError("channel_adapter requires cluster_id_c length to match num_channels.")
        self.register_buffer("cluster_id_c", cluster_id_c, persistent=False)

        self.channel_idx = []
        self.down = nn.ParameterList()
        self.up = nn.ParameterList()
        self.bias = nn.ParameterList()
        for k in range(self.K):
            idx = (cluster_id_c == k).nonzero(as_tuple=False).view(-1)
            self.register_buffer(f"channel_idx_{k}", idx, persistent=False)
            self.channel_idx.append(idx)
            n = int(idx.numel())
            self.down.append(nn.Parameter(torch.empty(n, self.L, self.rank)))
            self.up.append(nn.Parameter(torch.empty(n, self.rank, self.H)))
            self.bias.append(nn.Parameter(torch.zeros(n, self.H)))
        self.reset_parameters(init=init)

    def reset_parameters(self, init: str = "zero_delta") -> None:
        init = str(init).lower()
        for down, up, bias in zip(self.down, self.up, self.bias):
            if down.numel() > 0:
                nn.init.xavier_uniform_(down)
            if up.numel() > 0:
                if init == "zero_delta":
                    nn.init.zeros_(up)
                elif init == "xavier":
                    nn.init.xavier_uniform_(up)
                else:
                    raise ValueError(f"Unsupported channel_adapter.init='{init}'.")
            nn.init.zeros_(bias)

    def forward(self, x_bcl: torch.Tensor, cluster_id_c: torch.Tensor) -> torch.Tensor:
        y = self.base(x_bcl, cluster_id_c)
        if self.scale == 0.0:
            return y
        for k in range(self.K):
            idx = getattr(self, f"channel_idx_{k}").to(device=x_bcl.device)
            if idx.numel() == 0:
                continue
            x_bnl = x_bcl.index_select(1, idx)
            x_center = x_bnl - x_bnl[..., -1:]
            h_bnr = torch.einsum("bnl,nlr->bnr", x_center, self.down[k])
            delta_bnh = torch.einsum("bnr,nrh->bnh", h_bnr, self.up[k]) + self.bias[k].unsqueeze(0)
            y = y.index_copy(1, idx, y.index_select(1, idx) + self.scale * delta_bnh)
        return y

    def get_cluster_params(self, k: int) -> List[nn.Parameter]:
        return self.base.get_cluster_params(k) + [self.down[k], self.up[k], self.bias[k]]

    def get_cluster_state(self, k: int):
        return {
            "base": self.base.get_cluster_state(k),
            "down": self.down[k].detach().cpu(),
            "up": self.up[k].detach().cpu(),
            "bias": self.bias[k].detach().cpu(),
        }

    def load_cluster_state(self, k: int, state):
        self.base.load_cluster_state(k, state["base"])
        device = self.down[k].device
        self.down[k].data.copy_(state["down"].to(device))
        self.up[k].data.copy_(state["up"].to(device))
        self.bias[k].data.copy_(state["bias"].to(device))


def build_cluster_predictor(
    num_clusters: int,
    input_len: int,
    pred_len: int,
    model_cfg: Dict[str, object],
    num_channels: Optional[int] = None,
    cluster_id_c: Optional[torch.Tensor] = None,
) -> nn.Module:
    predictor = str(model_cfg.get("predictor", "mlp")).lower()
    force_revin = predictor in {"lstm_revin", "revin_lstm"}
    if force_revin:
        predictor = "lstm"
    internal_tail_predictor = predictor in {
        "long_context_channel_head_mlp",
        "lc_channel_head_mlp",
        "summary_channel_head_mlp",
        "long_context_anchor_channel_head_mlp",
        "lc_anchor_channel_head_mlp",
        "summary_anchor_channel_head_mlp",
        "seasonality_gated_channel_head_mlp",
        "seasonal_hybrid_channel_head_mlp",
    }
    hidden_dim = int(model_cfg["hidden_dim"])
    dropout = float(model_cfg.get("dropout", 0.0))
    recursive_enable = bool(model_cfg.get("recursive_rollout", False))
    recursive_chunk_len = int(model_cfg.get("recursive_chunk_len", 96))
    base_pred_len = recursive_chunk_len if recursive_enable else pred_len
    predictor_input_len_raw = model_cfg.get("predictor_input_len", model_cfg.get("base_input_len", input_len))
    predictor_input_len = int(predictor_input_len_raw)
    if predictor_input_len <= 0:
        raise ValueError("model.predictor_input_len must be positive.")
    if predictor_input_len > int(input_len):
        raise ValueError(
            f"model.predictor_input_len={predictor_input_len} cannot exceed window.input_len={int(input_len)}."
        )
    base_input_len = int(input_len) if internal_tail_predictor else predictor_input_len

    base_predictor: nn.Module
    if predictor in {"mlp", "cluster_mlp"}:
        base_predictor = ClusterwiseMLP(num_clusters, base_input_len, base_pred_len, hidden_dim, dropout)
    elif predictor in {"segment_mlp", "seg_mlp", "chunk_mlp"}:
        base_predictor = ClusterwiseSegmentMLP(
            num_clusters=num_clusters,
            input_len=base_input_len,
            pred_len=base_pred_len,
            hidden_dim=hidden_dim,
            dropout=dropout,
            chunk_len=int(model_cfg.get("segment_chunk_len", model_cfg.get("chunk_len", 96))),
        )
    elif predictor in {"long_anchor_mlp", "anchor_residual_mlp", "coarse_anchor_mlp"}:
        default_anchor_points = int((base_pred_len + 95) // 96) + 1
        base_predictor = ClusterwiseLongAnchorMLP(
            num_clusters=num_clusters,
            input_len=base_input_len,
            pred_len=base_pred_len,
            hidden_dim=hidden_dim,
            dropout=dropout,
            chunk_len=int(model_cfg.get("anchor_chunk_len", model_cfg.get("segment_chunk_len", 96))),
            anchor_points=int(model_cfg.get("anchor_points", default_anchor_points)),
            detail_scale=float(model_cfg.get("anchor_detail_scale", 0.5)),
            residual=bool(model_cfg.get("anchor_residual", True)),
        )
    elif predictor in {"channel_head_mlp", "channel_mlp"}:
        if num_channels is None or cluster_id_c is None:
            raise ValueError("model.predictor=channel_head_mlp requires num_channels and cluster_id_c.")
        base_predictor = ClusterwiseChannelHeadMLP(
            num_clusters=num_clusters,
            input_len=base_input_len,
            pred_len=base_pred_len,
            hidden_dim=hidden_dim,
            num_channels=int(num_channels),
            cluster_id_c=cluster_id_c,
            dropout=dropout,
            residual=bool(model_cfg.get("channel_head_residual", True)),
        )
    elif predictor in {"context_channel_head_mlp", "cluster_context_channel_head_mlp", "cc_channel_head_mlp"}:
        if num_channels is None or cluster_id_c is None:
            raise ValueError("model.predictor=context_channel_head_mlp requires num_channels and cluster_id_c.")
        base_predictor = ClusterwiseContextChannelHeadMLP(
            num_clusters=num_clusters,
            input_len=base_input_len,
            pred_len=base_pred_len,
            hidden_dim=hidden_dim,
            num_channels=int(num_channels),
            cluster_id_c=cluster_id_c,
            dropout=dropout,
            residual=bool(model_cfg.get("context_channel_head_residual", True)),
            include_delta=bool(model_cfg.get("context_channel_head_include_delta", True)),
        )
    elif predictor in {
        "long_context_channel_head_mlp",
        "lc_channel_head_mlp",
        "summary_channel_head_mlp",
        "long_context_anchor_channel_head_mlp",
        "lc_anchor_channel_head_mlp",
        "summary_anchor_channel_head_mlp",
    }:
        if num_channels is None or cluster_id_c is None:
            raise ValueError("model.predictor=long_context_channel_head_mlp requires num_channels and cluster_id_c.")
        output_mode = (
            "anchor"
            if predictor in {
                "long_context_anchor_channel_head_mlp",
                "lc_anchor_channel_head_mlp",
                "summary_anchor_channel_head_mlp",
            }
            else str(model_cfg.get("long_context_output_mode", "direct"))
        )
        base_predictor = ClusterwiseLongContextChannelHeadMLP(
            num_clusters=num_clusters,
            input_len=int(input_len),
            tail_len=predictor_input_len,
            pred_len=base_pred_len,
            hidden_dim=hidden_dim,
            num_channels=int(num_channels),
            cluster_id_c=cluster_id_c,
            dropout=dropout,
            residual=bool(model_cfg.get("long_context_channel_head_residual", True)),
            include_seasonal_profile=bool(model_cfg.get("long_context_include_seasonal_profile", False)),
            output_mode=output_mode,
            chunk_len=int(model_cfg.get("anchor_chunk_len", model_cfg.get("segment_chunk_len", 96))),
            anchor_points=model_cfg.get("anchor_points", None),
            detail_scale=float(model_cfg.get("anchor_detail_scale", 0.25)),
        )
    elif predictor in {"seasonality_gated_channel_head_mlp", "seasonal_hybrid_channel_head_mlp"}:
        if num_channels is None or cluster_id_c is None:
            raise ValueError("model.predictor=seasonality_gated_channel_head_mlp requires num_channels and cluster_id_c.")
        base_predictor = ClusterwiseSeasonalityGatedChannelHeadMLP(
            num_clusters=num_clusters,
            input_len=int(input_len),
            tail_len=predictor_input_len,
            pred_len=base_pred_len,
            hidden_dim=hidden_dim,
            num_channels=int(num_channels),
            cluster_id_c=cluster_id_c,
            dropout=dropout,
            residual=bool(model_cfg.get("seasonal_hybrid_residual", True)),
            include_seasonal_profile=bool(model_cfg.get("long_context_include_seasonal_profile", True)),
            chunk_len=int(model_cfg.get("anchor_chunk_len", model_cfg.get("segment_chunk_len", 96))),
            anchor_points=model_cfg.get("anchor_points", None),
            detail_scale=float(model_cfg.get("anchor_detail_scale", 0.25)),
            mix_init=float(model_cfg.get("seasonal_mix_init", -2.0)),
            gate_strength=float(model_cfg.get("seasonal_gate_strength", 0.0)),
            gate_threshold=float(model_cfg.get("seasonal_gate_threshold", 0.75)),
        )
    elif predictor == "attn_mlp":
        base_predictor = ClusterwiseAttnMLP(
            num_clusters=num_clusters,
            input_len=base_input_len,
            pred_len=base_pred_len,
            hidden_dim=hidden_dim,
            dropout=dropout,
            attn_dim=int(model_cfg.get("attn_dim", max(hidden_dim // 4, 32))),
        )
    elif predictor == "context_mlp":
        base_predictor = ClusterwiseContextMLP(
            num_clusters=num_clusters,
            input_len=base_input_len,
            pred_len=base_pred_len,
            hidden_dim=hidden_dim,
            dropout=dropout,
            include_delta=bool(model_cfg.get("context_include_delta", True)),
        )
    elif predictor == "nlinear":
        base_predictor = ClusterwiseNLinear(
            num_clusters=num_clusters,
            input_len=base_input_len,
            pred_len=base_pred_len,
        )
    elif predictor == "dlinear":
        base_predictor = ClusterwiseDLinear(
            num_clusters=num_clusters,
            input_len=base_input_len,
            pred_len=base_pred_len,
            kernel_size=int(model_cfg.get("dlinear_kernel_size", 25)),
        )
    elif predictor in {"channel_dlinear", "cdlinear"}:
        if num_channels is None or cluster_id_c is None:
            raise ValueError("model.predictor=channel_dlinear requires num_channels and cluster_id_c.")
        base_predictor = ClusterwiseChannelDLinear(
            num_clusters=num_clusters,
            input_len=base_input_len,
            pred_len=base_pred_len,
            num_channels=int(num_channels),
            cluster_id_c=cluster_id_c,
            kernel_size=int(model_cfg.get("dlinear_kernel_size", 25)),
        )
    elif predictor in {"patchtst", "patch_transformer"}:
        base_predictor = ClusterwisePatchTST(
            num_clusters=num_clusters,
            input_len=base_input_len,
            pred_len=base_pred_len,
            d_model=int(model_cfg.get("patch_d_model", hidden_dim)),
            dropout=dropout,
            patch_len=int(model_cfg.get("patch_len", 16)),
            patch_stride=int(model_cfg.get("patch_stride", 8)),
            num_layers=int(model_cfg.get("patch_num_layers", 2)),
            num_heads=int(model_cfg.get("patch_num_heads", 4)),
            ff_dim=int(model_cfg.get("patch_ff_dim", max(hidden_dim * 2, 128))),
        )
    elif predictor == "nbeats":
        base_predictor = ClusterwiseNBEATS(
            num_clusters=num_clusters,
            input_len=base_input_len,
            pred_len=base_pred_len,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_blocks=int(model_cfg.get("nbeats_num_blocks", 4)),
            num_layers=int(model_cfg.get("nbeats_num_layers", 4)),
        )
    elif predictor == "gru":
        base_predictor = ClusterwiseGRU(
            num_clusters=num_clusters,
            input_len=base_input_len,
            pred_len=base_pred_len,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_layers=int(model_cfg.get("gru_num_layers", 1)),
        )
    elif predictor == "lstm":
        base_predictor = ClusterwiseLSTM(
            num_clusters=num_clusters,
            input_len=base_input_len,
            pred_len=base_pred_len,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_layers=int(model_cfg.get("lstm_num_layers", 1)),
        )
    elif predictor in {"channel_lstm_mixer", "channel_lstm_moe", "backbone_moe"}:
        if num_channels is None or cluster_id_c is None:
            raise ValueError("model.predictor=channel_lstm_mixer requires num_channels and cluster_id_c.")
        base_predictor = ClusterwiseChannelLSTMMixer(
            num_clusters=num_clusters,
            input_len=base_input_len,
            pred_len=base_pred_len,
            hidden_dim=hidden_dim,
            num_channels=int(num_channels),
            cluster_id_c=cluster_id_c,
            dropout=dropout,
            lstm_num_layers=int(model_cfg.get("lstm_num_layers", 1)),
            lstm_hidden_dim=int(model_cfg.get("lstm_hidden_dim", hidden_dim)),
            mix_init=float(model_cfg.get("backbone_mix_init", model_cfg.get("mixer_lstm_init", -2.0))),
            hard_lstm_channels=list(model_cfg.get("backbone_lstm_channel_indices", []))
            if bool(model_cfg.get("backbone_hard_route", False))
            else None,
        )
    elif predictor == "tcn":
        base_predictor = ClusterwiseTCN(
            num_clusters=num_clusters,
            input_len=base_input_len,
            pred_len=base_pred_len,
            hidden_dim=hidden_dim,
            dropout=dropout,
            levels=int(model_cfg.get("tcn_levels", 2)),
            kernel_size=int(model_cfg.get("tcn_kernel_size", 3)),
            dilation_base=int(model_cfg.get("tcn_dilation_base", 2)),
        )
    else:
        raise ValueError(
            f"Unknown model.predictor='{predictor}'. Supported: mlp, segment_mlp, long_anchor_mlp, channel_head_mlp, context_channel_head_mlp, long_context_channel_head_mlp, long_context_anchor_channel_head_mlp, seasonality_gated_channel_head_mlp, attn_mlp, context_mlp, nlinear, dlinear, channel_dlinear, patchtst, nbeats, gru, lstm, lstm_revin, channel_lstm_mixer, tcn."
        )

    if force_revin or bool(model_cfg.get("revin", False)):
        base_predictor = ClusterwiseRevIN(
            base=base_predictor,
            eps=float(model_cfg.get("revin_eps", 1.0e-5)),
        )
    if predictor_input_len != int(input_len) and not internal_tail_predictor:
        base_predictor = ClusterwiseInputTail(
            base=base_predictor,
            input_len=int(input_len),
            tail_len=predictor_input_len,
        )
    if recursive_enable:
        base_predictor = ClusterwiseRecursiveRollout(base=base_predictor, pred_len=pred_len)
    if bool(model_cfg.get("seasonal_residual", False)):
        base_predictor = ClusterwiseSeasonalResidual(
            base=base_predictor,
            period=int(model_cfg.get("seasonal_period", 96)),
            num_periods=int(model_cfg.get("seasonal_num_periods", 1)),
        )
    if bool(model_cfg.get("seasonal_anchor", False)):
        base_predictor = ClusterwiseSeasonalAnchor(
            base=base_predictor,
            period=int(model_cfg.get("seasonal_anchor_period", model_cfg.get("seasonal_period", 96))),
            num_periods=int(model_cfg.get("seasonal_anchor_num_periods", model_cfg.get("seasonal_num_periods", 1))),
            delta_scale=float(model_cfg.get("seasonal_anchor_delta_scale", 1.0)),
        )
    basis_cfg = model_cfg.get("temporal_basis_adapter", {})
    if basis_cfg is None:
        basis_cfg = {}
    if bool(dict(basis_cfg).get("enable", False)):
        base_predictor = ClusterwiseTemporalBasisAdapter(
            base=base_predictor,
            rank=int(dict(basis_cfg).get("rank", 16)),
            scale=float(dict(basis_cfg).get("scale", 1.0)),
            init=str(dict(basis_cfg).get("init", "zero_delta")),
            freeze_base=bool(dict(basis_cfg).get("freeze_base", False)),
        )
    adapter_cfg = model_cfg.get("channel_adapter", {})
    if adapter_cfg is None:
        adapter_cfg = {}
    if bool(dict(adapter_cfg).get("enable", False)):
        if num_channels is None or cluster_id_c is None:
            raise ValueError("model.channel_adapter requires num_channels and cluster_id_c.")
        base_predictor = ClusterwiseChannelResidualAdapter(
            base=base_predictor,
            num_channels=int(num_channels),
            cluster_id_c=cluster_id_c,
            rank=int(dict(adapter_cfg).get("rank", 8)),
            init=str(dict(adapter_cfg).get("init", "zero_delta")),
            scale=float(dict(adapter_cfg).get("scale", 1.0)),
        )
    return base_predictor
