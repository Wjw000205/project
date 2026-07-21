"""Light multichannel joint-H96 mixture over aligned forecast residuals."""

from __future__ import annotations

import torch
from torch import nn

from src.models.amp_weekly_carrier_mixture_adapter import remove_affine_patch


class AmpH96MultichannelInputResidualMixtureAdapter(nn.Module):
    """Use synchronous residual context while mixing same-channel carriers."""

    def __init__(
        self,
        channel_count: int = 7,
        horizon_patch_count: int = 8,
        history_patch_count: int = 8,
        patch_len: int = 12,
        patch_width: int = 4,
        history_position_width: int = 2,
        channel_width: int = 4,
        horizon_width: int = 4,
        recurrent_width: int = 8,
        head_width: int = 24,
        maximum_coefficient_abs: float = 4.0,
    ) -> None:
        super().__init__()
        if maximum_coefficient_abs <= 0.0:
            raise ValueError("maximum coefficient magnitude must be positive")
        self.channel_count = int(channel_count)
        self.horizon_patch_count = int(horizon_patch_count)
        self.history_patch_count = int(history_patch_count)
        self.patch_len = int(patch_len)
        self.maximum_coefficient_abs = float(maximum_coefficient_abs)
        self.patch_encoder = nn.Sequential(
            nn.Linear(patch_len, patch_width),
            nn.LayerNorm(patch_width),
            nn.GELU(),
        )
        self.history_position = nn.Embedding(
            history_patch_count, history_position_width
        )
        self.channel_embedding = nn.Embedding(channel_count, channel_width)
        self.horizon_position = nn.Embedding(horizon_patch_count, horizon_width)
        recurrent_input = history_patch_count * (
            patch_width + history_position_width
        )
        self.horizon_encoder = nn.GRU(
            recurrent_input,
            recurrent_width,
            batch_first=True,
            bidirectional=True,
        )
        state_width = 2 * recurrent_width
        coefficient_input = (
            3 * state_width + channel_width + horizon_width
        )
        self.coefficient_head = nn.Sequential(
            nn.Linear(coefficient_input, head_width),
            nn.GELU(),
            nn.Linear(head_width, history_patch_count),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        final = self.coefficient_head[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, input_residual: torch.Tensor) -> torch.Tensor:
        if input_residual.ndim != 5:
            raise ValueError(
                "input residual must have shape [batch, channel, horizon_patch, history_patch, patch_len]"
            )
        batch = input_residual.shape[0]
        expected = (
            batch,
            self.channel_count,
            self.horizon_patch_count,
            self.history_patch_count,
            self.patch_len,
        )
        if input_residual.shape != expected:
            raise ValueError("unexpected multichannel H96 input residual shape")

        encoded = self.patch_encoder(input_residual)
        history_index = torch.arange(
            self.history_patch_count, device=input_residual.device
        )
        history_position = self.history_position(history_index).view(
            1, 1, 1, self.history_patch_count, -1
        )
        history_position = history_position.expand(
            batch,
            self.channel_count,
            self.horizon_patch_count,
            -1,
            -1,
        )
        encoded = torch.cat([encoded, history_position], dim=-1).flatten(3)
        encoded = encoded.reshape(
            batch * self.channel_count, self.horizon_patch_count, -1
        )
        horizon_state, _ = self.horizon_encoder(encoded)
        horizon_state = horizon_state.reshape(
            batch, self.channel_count, self.horizon_patch_count, -1
        )
        cross_mean = horizon_state.mean(dim=1, keepdim=True).expand_as(horizon_state)
        cross_std = horizon_state.std(dim=1, keepdim=True, unbiased=False).expand_as(
            horizon_state
        )
        channel_index = torch.arange(self.channel_count, device=input_residual.device)
        channel_state = self.channel_embedding(channel_index).view(
            1, self.channel_count, 1, -1
        )
        channel_state = channel_state.expand(
            batch, -1, self.horizon_patch_count, -1
        )
        horizon_index = torch.arange(
            self.horizon_patch_count, device=input_residual.device
        )
        horizon_position = self.horizon_position(horizon_index).view(
            1, 1, self.horizon_patch_count, -1
        )
        horizon_position = horizon_position.expand(
            batch, self.channel_count, -1, -1
        )
        feature = torch.cat(
            [
                horizon_state,
                cross_mean,
                cross_std,
                channel_state,
                horizon_position,
            ],
            dim=-1,
        )
        coefficient = self.maximum_coefficient_abs * torch.tanh(
            self.coefficient_head(feature)
        )
        correction = torch.einsum(
            "bcph,bcphk->bcpk", coefficient, input_residual
        ).reshape(batch * self.channel_count, -1)
        correction = remove_affine_patch(correction)
        return correction.reshape(batch, self.channel_count, -1)
