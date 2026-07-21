"""Light residual-to-residual Amp adapter."""

from __future__ import annotations

import torch
from torch import nn

from src.models.amp_weekly_carrier_mixture_adapter import remove_affine_patch


class AmpInputResidualAdapter(nn.Module):
    """Map aligned observed forecast errors to the current Amp correction."""

    def __init__(
        self,
        group_count: int = 56,
        input_patch_count: int = 8,
        patch_len: int = 12,
        group_embedding_width: int = 4,
        hidden_width: int = 64,
        bottleneck_width: int = 32,
    ) -> None:
        super().__init__()
        dimensions = (
            group_count,
            input_patch_count,
            patch_len,
            group_embedding_width,
            hidden_width,
            bottleneck_width,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("all input-residual adapter dimensions must be positive")
        self.group_count = int(group_count)
        self.input_patch_count = int(input_patch_count)
        self.patch_len = int(patch_len)
        self.group_embedding = nn.Embedding(group_count, group_embedding_width)
        input_width = input_patch_count * patch_len + group_embedding_width
        self.trunk = nn.Sequential(
            nn.Linear(input_width, hidden_width),
            nn.LayerNorm(hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, bottleneck_width),
            nn.GELU(),
        )
        self.correction = nn.Linear(bottleneck_width, patch_len)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

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
        if group.shape != (batch,):
            raise ValueError("group must have shape [batch]")
        if group.dtype != torch.long:
            raise ValueError("group must use torch.long dtype")
        feature = torch.cat(
            [input_residual.reshape(batch, -1), self.group_embedding(group)], dim=1
        )
        return remove_affine_patch(self.correction(self.trunk(feature)))
