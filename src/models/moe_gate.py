from typing import Optional, Tuple
import torch
from torch import nn


def _cluster_assign_ck(cluster_id_c: torch.Tensor, K: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.nn.functional.one_hot(cluster_id_c.to(torch.long), num_classes=K).to(dtype=dtype)


def scatter_mean_bc_to_bk(x_bc: torch.Tensor, cluster_id_c: torch.Tensor, K: int) -> torch.Tensor:
    """
    x_bc: [B, C]
    -> [B, K] 对每个 batch 按 cluster 聚合均值
    """
    assign_ck = _cluster_assign_ck(cluster_id_c.to(device=x_bc.device), K, dtype=x_bc.dtype)
    out_bk = x_bc @ assign_ck
    cnt_k = assign_ck.sum(dim=0).clamp_min(1.0)
    return out_bk / cnt_k.view(1, K)

def scatter_mean_bcf_to_bkf(x_bcf: torch.Tensor, cluster_id_c: torch.Tensor, K: int) -> torch.Tensor:
    """
    x_bcf: [B, C, F] -> [B, K, F]
    """
    _, _, F = x_bcf.shape
    assign_ck = _cluster_assign_ck(cluster_id_c.to(device=x_bcf.device), K, dtype=x_bcf.dtype)
    out_bfk = torch.matmul(x_bcf.transpose(1, 2), assign_ck)
    cnt_k = assign_ck.sum(dim=0).clamp_min(1.0)
    return out_bfk.transpose(1, 2) / cnt_k.view(1, K, 1)

class ClusterwiseMoEGate(nn.Module):
    """
    MoE 选择惩罚项：每次最多激活 topk 个惩罚（默认2）。
    - 输入：按簇聚合后的特征 [B,K,F]
    - 输出：hard mask [B,K,P]（前向稀疏），以及 soft probs [B,K,P]
    训练时可用 straight-through：前向 hard，反向用 soft 近似。
    """
    def __init__(
        self,
        num_clusters: int,
        feat_dim: int,
        num_penalties: int,
        hidden_dim: int = 64,
        topk: int = 2,
        temperature: float = 1.0,
        noise_std: float = 0.0,
        logit_clip: float = 0.0,
        prob_floor: float = 0.0,
        allow_skip: bool = False,
        skip_init_bias: float = -2.0,
    ):
        super().__init__()
        self.K = num_clusters
        self.F = feat_dim
        self.P = num_penalties
        self.topk = topk
        self.temperature = float(temperature)
        self.noise_std = float(noise_std)
        self.logit_clip = float(logit_clip)
        self.allow_skip = bool(allow_skip)
        self.skip_init_bias = float(skip_init_bias)
        max_floor = (1.0 / max(self.P, 1)) - 1.0e-6
        self.prob_floor = float(max(0.0, min(prob_floor, max_floor)))
        prior_shape = (num_clusters, max(num_penalties, 1))
        prior_prob = torch.full(prior_shape, 1.0 / max(num_penalties, 1), dtype=torch.float32)
        self.register_buffer("penalty_prior_prob", prior_prob, persistent=False)
        self.register_buffer("penalty_prior_logits", prior_prob.clamp_min(1.0e-8).log(), persistent=False)
        self.register_buffer("penalty_allowed_mask", torch.ones_like(prior_prob), persistent=False)
        self.penalty_prior_strength = 0.0

        # 每个簇一个 gate（参数按簇存矩阵），避免 python 循环
        self.W1 = nn.ParameterList([
            nn.Parameter(torch.empty(feat_dim, hidden_dim)) for _ in range(num_clusters)
        ])
        self.b1 = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_dim)) for _ in range(num_clusters)
        ])
        self.W2 = nn.ParameterList([
            nn.Parameter(torch.empty(hidden_dim, num_penalties)) for _ in range(num_clusters)
        ])
        self.b2 = nn.ParameterList([
            nn.Parameter(torch.zeros(num_penalties)) for _ in range(num_clusters)
        ])
        if self.allow_skip:
            self.W_skip = nn.ParameterList([
                nn.Parameter(torch.empty(hidden_dim)) for _ in range(num_clusters)
            ])
            self.b_skip = nn.ParameterList([
                nn.Parameter(torch.full((), self.skip_init_bias)) for _ in range(num_clusters)
            ])
        else:
            self.W_skip = None
            self.b_skip = None
        self.act = nn.GELU()
        self.reset_parameters()

    def reset_parameters(self):
        for w in self.W1:
            nn.init.xavier_uniform_(w)
        for w in self.W2:
            nn.init.xavier_uniform_(w)
        if self.W_skip is not None:
            for w in self.W_skip:
                nn.init.normal_(w, mean=0.0, std=0.02)

    def set_penalty_prior(self, prior_kp: Optional[torch.Tensor], strength: float = 0.0):
        self.penalty_prior_strength = float(max(strength, 0.0))
        if self.P <= 0 or prior_kp is None or prior_kp.numel() == 0:
            self.penalty_prior_prob.fill_(1.0 / max(self.P, 1))
            self.penalty_prior_logits.copy_(self.penalty_prior_prob.clamp_min(1.0e-8).log())
            return
        prior = prior_kp.to(device=self.penalty_prior_prob.device, dtype=self.penalty_prior_prob.dtype)
        expected_shape = self.penalty_prior_prob[:, :self.P].shape
        if prior.shape != expected_shape:
            raise ValueError(f"Expected prior shape {tuple(expected_shape)}, got {tuple(prior.shape)}")
        prior = prior.clamp_min(1.0e-8)
        prior = prior / prior.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        self.penalty_prior_prob[:, :self.P].copy_(prior)
        self.penalty_prior_logits[:, :self.P].copy_(prior.log())

    def get_penalty_prior(self) -> Optional[torch.Tensor]:
        if self.P <= 0:
            return None
        return self.penalty_prior_prob[:, :self.P]

    def set_penalty_allowed_mask(self, allowed_kp: Optional[torch.Tensor]):
        if self.P <= 0 or allowed_kp is None or allowed_kp.numel() == 0:
            self.penalty_allowed_mask.fill_(1.0)
            return
        allowed = allowed_kp.to(device=self.penalty_allowed_mask.device, dtype=self.penalty_allowed_mask.dtype)
        expected_shape = self.penalty_allowed_mask[:, :self.P].shape
        if allowed.shape != expected_shape:
            raise ValueError(f"Expected allowed mask shape {tuple(expected_shape)}, got {tuple(allowed.shape)}")
        allowed = (allowed > 0).to(dtype=self.penalty_allowed_mask.dtype)
        empty = allowed.sum(dim=-1, keepdim=True) <= 0
        if bool(empty.any().item()):
            allowed = torch.where(empty, torch.ones_like(allowed), allowed)
        self.penalty_allowed_mask[:, :self.P].copy_(allowed)

    def forward(
        self,
        feat_bkf: torch.Tensor,
        straight_through: bool = True,
        penalty_context_bkp: Optional[torch.Tensor] = None,
        penalty_context_mode: str = "learned",
        penalty_context_weight: float = 0.0,
        penalty_context_detach: bool = True,
        penalty_context_score: str = "high_violation",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        feat_bkf: [B,K,F]
        返回：
          mask_bkp: [B,K,P] (hard in forward)
          probs_bkp: [B,K,P]
          skip_bk: [B,K]
          skip_prob_bk: [B,K]
        """
        W1 = torch.stack(list(self.W1), dim=0)
        b1 = torch.stack(list(self.b1), dim=0)
        W2 = torch.stack(list(self.W2), dim=0)
        b2 = torch.stack(list(self.b2), dim=0)
        h = torch.einsum("bkf,kfh->bkh", feat_bkf, W1) + b1.unsqueeze(0)  # [B,K,H]
        h = self.act(h)
        logits = torch.einsum("bkh,khp->bkp", h, W2) + b2.unsqueeze(0)    # [B,K,P]
        if self.P > 0 and self.penalty_prior_strength > 0.0:
            logits = logits + (self.penalty_prior_strength * self.penalty_prior_logits[:, :self.P].unsqueeze(0))
        allowed = None
        if self.P > 0:
            allowed = self.penalty_allowed_mask[:, :self.P].to(device=logits.device, dtype=torch.bool)
            logits = logits.masked_fill(~allowed.unsqueeze(0), -1.0e9)
        mode = str(penalty_context_mode).lower()
        if penalty_context_bkp is not None and penalty_context_bkp.numel() > 0 and mode != "learned":
            context = penalty_context_bkp.detach() if bool(penalty_context_detach) else penalty_context_bkp
            if context.shape != logits.shape:
                raise ValueError(f"Expected penalty_context_bkp shape {tuple(logits.shape)}, got {tuple(context.shape)}")
            context_logits = context.clamp_min(1.0e-8).log()
            score_mode = str(penalty_context_score).lower()
            if score_mode in {"low", "low_penalty", "low_violation", "min_penalty"}:
                context_logits = -context_logits
            elif score_mode not in {"high", "high_penalty", "high_violation", "max_penalty"}:
                raise ValueError(
                    "penalty_context_score must be high_violation or low_violation "
                    f"(got {penalty_context_score!r})."
                )
            weight = float(penalty_context_weight)
            if mode == "penalty_only":
                logits = (weight if weight > 0.0 else 1.0) * context_logits
            elif mode == "penalty_context":
                if weight > 0.0:
                    logits = logits + weight * context_logits
            else:
                raise ValueError(
                    "penalty_context_mode must be learned, penalty_context, or penalty_only "
                    f"(got {penalty_context_mode!r})."
                )
        if self.logit_clip > 0.0:
            logits = self.logit_clip * torch.tanh(logits / self.logit_clip)
        if self.training and self.noise_std > 0.0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        temp = max(self.temperature, 1.0e-6)
        probs = torch.softmax(logits / temp, dim=-1)
        if self.prob_floor > 0.0:
            probs = self.prob_floor + (1.0 - self.P * self.prob_floor) * probs

        k = min(self.topk, self.P)
        top_idx = probs.topk(k=k, dim=-1).indices  # [B,K,k]
        hard = torch.zeros_like(probs)
        hard.scatter_(-1, top_idx, 1.0)
        if allowed is not None:
            hard = hard * allowed.unsqueeze(0).to(dtype=hard.dtype)

        if straight_through:
            mask = hard - probs.detach() + probs
        else:
            mask = hard
        if self.allow_skip:
            W_skip = torch.stack(list(self.W_skip), dim=0)
            b_skip = torch.stack(list(self.b_skip), dim=0)
            skip_logits = torch.einsum("bkh,kh->bk", h, W_skip) + b_skip.unsqueeze(0)
            if self.logit_clip > 0.0:
                skip_logits = self.logit_clip * torch.tanh(skip_logits / self.logit_clip)
            if self.training and self.noise_std > 0.0:
                skip_logits = skip_logits + torch.randn_like(skip_logits) * self.noise_std
            skip_prob = torch.sigmoid(skip_logits)
            skip_hard = (skip_prob > 0.5).to(skip_prob.dtype)
            if straight_through:
                skip = skip_hard - skip_prob.detach() + skip_prob
            else:
                skip = skip_hard
        else:
            skip_prob = torch.zeros(mask.shape[:2], device=mask.device, dtype=mask.dtype)
            skip = torch.zeros_like(skip_prob)
        return mask, probs, skip, skip_prob

    def mask_cluster_grads(self, stopped_k: torch.Tensor):
        for k in range(self.K):
            if not bool(stopped_k[k].item()):
                continue
            if self.W1[k].grad is not None:
                self.W1[k].grad.zero_()
            if self.b1[k].grad is not None:
                self.b1[k].grad.zero_()
            if self.W2[k].grad is not None:
                self.W2[k].grad.zero_()
            if self.b2[k].grad is not None:
                self.b2[k].grad.zero_()
            if self.W_skip is not None and self.W_skip[k].grad is not None:
                self.W_skip[k].grad.zero_()
            if self.b_skip is not None and self.b_skip[k].grad is not None:
                self.b_skip[k].grad.zero_()

    def get_cluster_state(self, k: int):
        state = {
            "W1": self.W1[k].detach().cpu(),
            "b1": self.b1[k].detach().cpu(),
            "W2": self.W2[k].detach().cpu(),
            "b2": self.b2[k].detach().cpu(),
        }
        if self.W_skip is not None:
            state["W_skip"] = self.W_skip[k].detach().cpu()
            state["b_skip"] = self.b_skip[k].detach().cpu()
        return state

    def load_cluster_state(self, k: int, state):
        device = self.W1[k].device
        self.W1[k].data.copy_(state["W1"].to(device))
        self.b1[k].data.copy_(state["b1"].to(device))
        self.W2[k].data.copy_(state["W2"].to(device))
        self.b2[k].data.copy_(state["b2"].to(device))
        if self.W_skip is not None and ("W_skip" in state) and ("b_skip" in state):
            self.W_skip[k].data.copy_(state["W_skip"].to(device))
            self.b_skip[k].data.copy_(state["b_skip"].to(device))



