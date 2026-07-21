"""Light shared full-window residual adapter for Amp correction."""

from __future__ import annotations

import torch
from torch import nn

from src.models.amp_weekly_carrier_mixture_adapter import remove_affine_patch


class AmpLowRankWindowAdapter(nn.Module):
    """Predict all channel/horizon residuals with shared low-rank weights."""

    def __init__(
        self,
        channel_count: int = 7,
        input_len: int = 96,
        pred_len: int = 96,
        patch_len: int = 12,
        channel_embedding_width: int = 4,
        hidden_width: int = 24,
    ) -> None:
        super().__init__()
        dimensions = (
            channel_count,
            input_len,
            pred_len,
            patch_len,
            channel_embedding_width,
            hidden_width,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("all window Amp adapter dimensions must be positive")
        if pred_len % patch_len != 0:
            raise ValueError("prediction length must be divisible by patch length")
        self.channel_count = int(channel_count)
        self.input_len = int(input_len)
        self.pred_len = int(pred_len)
        self.patch_len = int(patch_len)
        self.patch_count = self.pred_len // self.patch_len
        self.channel_embedding = nn.Embedding(
            self.channel_count, channel_embedding_width
        )
        feature_width = 2 * self.input_len + 2 * self.pred_len + channel_embedding_width
        self.trunk = nn.Sequential(
            nn.Linear(feature_width, hidden_width),
            nn.GELU(),
        )
        self.correction = nn.Linear(hidden_width, self.pred_len)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def forward(self, history: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or base.ndim != 3:
            raise ValueError("window Amp history/base must be rank three")
        batch = history.shape[0]
        if history.shape != (batch, self.channel_count, self.input_len):
            raise ValueError("unexpected window Amp history shape")
        if base.shape != (batch, self.channel_count, self.pred_len):
            raise ValueError("unexpected window Amp base shape")
        history_mean = history.mean(dim=1, keepdim=True).expand(-1, self.channel_count, -1)
        base_mean = base.mean(dim=1, keepdim=True).expand(-1, self.channel_count, -1)
        channel = torch.arange(self.channel_count, device=history.device)
        embedding = self.channel_embedding(channel).unsqueeze(0).expand(batch, -1, -1)
        features = torch.cat([history, base, history_mean, base_mean, embedding], dim=2)
        raw = self.correction(self.trunk(features))
        patches = raw.reshape(
            batch * self.channel_count * self.patch_count, self.patch_len
        )
        projected = remove_affine_patch(patches)
        return projected.reshape(batch, self.channel_count, self.pred_len)
