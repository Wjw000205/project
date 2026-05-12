from typing import Dict
import torch
from torch import nn
from torch.nn import functional as F


class ClusterwiseLearnableLambda(nn.Module):
    """
    Learn cluster-wise lambda allocation under a fixed per-cluster budget.
    The module reallocates budget across penalties instead of shrinking the
    chosen penalty toward zero to game the objective.
    Output: [K, P]
    """

    def __init__(
        self,
        init_lambda_kp: torch.Tensor,
        lambda_min_kp: torch.Tensor | None = None,
        share_floor: float = 0.0,
    ):
        super().__init__()
        if init_lambda_kp.dim() != 2:
            raise ValueError("init_lambda_kp must have shape [K, P]")
        init_lambda_kp = init_lambda_kp.detach().to(torch.float32)
        if lambda_min_kp is None:
            lambda_min_kp = torch.zeros_like(init_lambda_kp)
        else:
            lambda_min_kp = lambda_min_kp.detach().to(torch.float32)
        self.K, self.P = init_lambda_kp.shape
        floor = float(max(0.0, min(share_floor, (1.0 / max(self.P, 1)) - 1.0e-6)))
        self.share_floor = floor
        self.register_buffer("lambda_min_kp", lambda_min_kp)
        self.register_buffer("init_lambda_kp", init_lambda_kp)

        residual_budget_k = (init_lambda_kp.sum(dim=-1, keepdim=True) - lambda_min_kp.sum(dim=-1, keepdim=True)).clamp_min(1.0e-6)
        init_extra_kp = (init_lambda_kp - lambda_min_kp).clamp_min(0.0)
        init_share_kp = init_extra_kp / residual_budget_k
        init_share_kp = init_share_kp / init_share_kp.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)

        self.register_buffer("residual_budget_k", residual_budget_k)
        self.register_buffer("init_share_kp", init_share_kp)

        logits_kp = []
        if floor > 0.0:
            free_mass = max(1.0 - self.P * floor, 1.0e-6)
            free_share_kp = ((init_share_kp - floor) / free_mass).clamp_min(1.0e-6)
            free_share_kp = free_share_kp / free_share_kp.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
            logits_kp = free_share_kp.log()
        else:
            logits_kp = init_share_kp.clamp_min(1.0e-6).log()

        self.raw = nn.ParameterList([
            nn.Parameter(logits_kp[k].clone()) for k in range(self.K)
        ])

    def _share_kp(self) -> torch.Tensor:
        rows = []
        floor = self.share_floor
        free_mass = max(1.0 - self.P * floor, 0.0)
        for k in range(self.K):
            share = F.softmax(self.raw[k], dim=-1)
            if floor > 0.0:
                share = floor + free_mass * share
            rows.append(share)
        return torch.stack(rows, dim=0)

    def forward(self) -> torch.Tensor:
        share_kp = self._share_kp()
        return self.lambda_min_kp + self.residual_budget_k * share_kp

    def regularization(self) -> torch.Tensor:
        share_kp = self._share_kp().clamp_min(1.0e-8)
        target_share_kp = self.init_share_kp.clamp_min(1.0e-8)
        return (share_kp.log() - target_share_kp.log()).pow(2).mean(dim=-1)

    def mask_cluster_grads(self, stopped_k: torch.Tensor):
        for k in range(self.K):
            if not bool(stopped_k[k].item()):
                continue
            if self.raw[k].grad is not None:
                self.raw[k].grad.zero_()

    def get_cluster_state(self, k: int) -> Dict[str, torch.Tensor]:
        return {"raw": self.raw[k].detach().cpu()}

    def load_cluster_state(self, k: int, state: Dict[str, torch.Tensor]):
        device = self.raw[k].device
        self.raw[k].data.copy_(state["raw"].to(device))
