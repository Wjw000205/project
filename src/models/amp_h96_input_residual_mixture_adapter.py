"""Light joint-H96 mixture over aligned input forecast residuals."""

from __future__ import annotations

import torch
from torch import nn

from src.models.amp_weekly_carrier_mixture_adapter import remove_affine_patch


class AmpH96InputResidualMixtureAdapter(nn.Module):
    """Emit one globally non-affine H96 correction from an 8x8 carrier bank."""

    def __init__(
        self,
        channel_count: int = 7,
        horizon_patch_count: int = 8,
        history_patch_count: int = 8,
        patch_len: int = 12,
        patch_width: int = 8,
        position_width: int = 4,
        channel_width: int = 4,
        recurrent_width: int = 12,
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
        self.history_position = nn.Embedding(history_patch_count, position_width)
        self.horizon_position = nn.Embedding(horizon_patch_count, position_width)
        self.channel_embedding = nn.Embedding(channel_count, channel_width)
        recurrent_input = history_patch_count * (patch_width + position_width)
        self.horizon_encoder = nn.GRU(
            recurrent_input,
            recurrent_width,
            batch_first=True,
            bidirectional=True,
        )
        coefficient_input = 2 * recurrent_width + position_width + channel_width
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

    def predict_coefficients(
        self, input_residual: torch.Tensor, channel: torch.Tensor
    ) -> torch.Tensor:
        if input_residual.ndim != 4:
            raise ValueError(
                "input residual must have shape [batch, horizon_patch, history_patch, patch_len]"
            )
        batch = input_residual.shape[0]
        expected = (
            batch,
            self.horizon_patch_count,
            self.history_patch_count,
            self.patch_len,
        )
        if input_residual.shape != expected:
            raise ValueError("unexpected joint-H96 input residual shape")
        if channel.shape != (batch,) or channel.dtype != torch.long:
            raise ValueError("channel must be a torch.long vector")

        encoded = self.patch_encoder(input_residual)
        history_index = torch.arange(
            self.history_patch_count, device=input_residual.device
        )
        history_position = self.history_position(history_index).view(
            1, 1, self.history_patch_count, -1
        )
        history_position = history_position.expand(
            batch, self.horizon_patch_count, -1, -1
        )
        encoded = torch.cat([encoded, history_position], dim=-1).flatten(2)
        horizon_state, _ = self.horizon_encoder(encoded)

        horizon_index = torch.arange(
            self.horizon_patch_count, device=input_residual.device
        )
        horizon_position = self.horizon_position(horizon_index).view(
            1, self.horizon_patch_count, -1
        )
        horizon_position = horizon_position.expand(batch, -1, -1)
        channel_state = self.channel_embedding(channel).unsqueeze(1).expand(
            -1, self.horizon_patch_count, -1
        )
        coefficient_feature = torch.cat(
            [horizon_state, horizon_position, channel_state], dim=-1
        )
        return self.maximum_coefficient_abs * torch.tanh(
            self.coefficient_head(coefficient_feature)
        )

    def forward(self, input_residual: torch.Tensor, channel: torch.Tensor) -> torch.Tensor:
        coefficient = self.predict_coefficients(input_residual, channel)
        batch = input_residual.shape[0]
        correction = torch.einsum("bph,bphk->bpk", coefficient, input_residual)
        correction = correction.reshape(batch, -1)
        return remove_affine_patch(correction)
