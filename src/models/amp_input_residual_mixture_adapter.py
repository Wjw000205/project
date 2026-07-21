"""Dynamic signed mixture over horizon-aligned input forecast residuals."""

from __future__ import annotations

import torch
from torch import nn

from src.models.amp_weekly_carrier_mixture_adapter import remove_affine_patch


class AmpInputResidualMixtureAdapter(nn.Module):
    """Predict coefficients, while the observed residuals supply the waveform."""

    def __init__(
        self,
        group_count: int = 56,
        input_patch_count: int = 8,
        patch_len: int = 12,
        group_embedding_width: int = 4,
        hidden_width: int = 64,
        bottleneck_width: int = 32,
        maximum_coefficient_abs: float = 4.0,
    ) -> None:
        super().__init__()
        if maximum_coefficient_abs <= 0.0:
            raise ValueError("maximum coefficient magnitude must be positive")
        self.group_count = int(group_count)
        self.input_patch_count = int(input_patch_count)
        self.patch_len = int(patch_len)
        self.maximum_coefficient_abs = float(maximum_coefficient_abs)
        self.group_embedding = nn.Embedding(group_count, group_embedding_width)
        input_width = input_patch_count * patch_len + group_embedding_width
        self.trunk = nn.Sequential(
            nn.Linear(input_width, hidden_width),
            nn.LayerNorm(hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, bottleneck_width),
            nn.GELU(),
        )
        self.coefficient = nn.Linear(bottleneck_width, input_patch_count)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.coefficient.weight)
        nn.init.zeros_(self.coefficient.bias)

    def forward(
        self, input_residual: torch.Tensor, group: torch.Tensor
    ) -> torch.Tensor:
        if input_residual.ndim != 3:
            raise ValueError("input residual must have shape [batch, patch, horizon]")
        batch = input_residual.shape[0]
        if input_residual.shape != (
            batch,
            self.input_patch_count,
            self.patch_len,
        ):
            raise ValueError("unexpected input residual shape")
        if group.shape != (batch,) or group.dtype != torch.long:
            raise ValueError("group must be a torch.long vector")
        feature = torch.cat(
            [input_residual.reshape(batch, -1), self.group_embedding(group)], dim=1
        )
        coefficient = self.maximum_coefficient_abs * torch.tanh(
            self.coefficient(self.trunk(feature))
        )
        correction = torch.einsum("bc,bcp->bp", coefficient, input_residual)
        return remove_affine_patch(correction)
