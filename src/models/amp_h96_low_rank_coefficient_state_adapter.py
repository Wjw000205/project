"""Light H96 carrier mixture with a fixed low-rank coefficient state basis."""

from __future__ import annotations

import torch
from torch import nn

from src.models.amp_weekly_carrier_mixture_adapter import remove_affine_patch


class AmpH96LowRankCoefficientStateAdapter(nn.Module):
    """Predict a small latent state and decode it into 8x8 carrier weights."""

    def __init__(
        self,
        coefficient_basis: torch.Tensor,
        channel_count: int = 7,
        horizon_patch_count: int = 8,
        history_patch_count: int = 8,
        patch_len: int = 12,
        patch_width: int = 8,
        position_width: int = 4,
        channel_width: int = 4,
        recurrent_width: int = 12,
        head_width: int = 16,
        maximum_latent_abs: float = 12.0,
    ) -> None:
        super().__init__()
        if coefficient_basis.ndim != 3:
            raise ValueError("coefficient basis must have shape [rank, horizon, history]")
        if coefficient_basis.shape[1:] != (
            horizon_patch_count,
            history_patch_count,
        ):
            raise ValueError("coefficient basis dimensions do not match carrier bank")
        if maximum_latent_abs <= 0.0:
            raise ValueError("maximum latent magnitude must be positive")
        self.rank = int(coefficient_basis.shape[0])
        self.channel_count = int(channel_count)
        self.horizon_patch_count = int(horizon_patch_count)
        self.history_patch_count = int(history_patch_count)
        self.patch_len = int(patch_len)
        self.maximum_latent_abs = float(maximum_latent_abs)
        self.register_buffer(
            "coefficient_basis", coefficient_basis.detach().clone().float()
        )
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
        pooled_state = 4 * recurrent_width
        amplitude_state = horizon_patch_count + history_patch_count + 1
        head_input = pooled_state + channel_width + amplitude_state
        self.latent_head = nn.Sequential(
            nn.Linear(head_input, head_width),
            nn.GELU(),
            nn.Linear(head_width, self.rank),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        final = self.latent_head[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def _validate(
        self, input_residual: torch.Tensor, channel: torch.Tensor
    ) -> None:
        if input_residual.ndim != 4:
            raise ValueError(
                "input residual must have shape [batch, horizon_patch, history_patch, patch_len]"
            )
        batch = input_residual.shape[0]
        if input_residual.shape != (
            batch,
            self.horizon_patch_count,
            self.history_patch_count,
            self.patch_len,
        ):
            raise ValueError("unexpected joint-H96 input residual shape")
        if channel.shape != (batch,) or channel.dtype != torch.long:
            raise ValueError("channel must be a torch.long vector")

    def predict_latent(
        self, input_residual: torch.Tensor, channel: torch.Tensor
    ) -> torch.Tensor:
        self._validate(input_residual, channel)
        batch = input_residual.shape[0]
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
        pooled = torch.cat(
            [horizon_state.mean(dim=1), horizon_state.amax(dim=1)], dim=1
        )
        channel_state = self.channel_embedding(channel)
        patch_rms = torch.sqrt(input_residual.square().mean(dim=-1) + 1.0e-8)
        global_rms = torch.sqrt(
            input_residual.square().mean(dim=(1, 2, 3)) + 1.0e-8
        )
        horizon_rms = patch_rms.mean(dim=2)
        history_rms = patch_rms.mean(dim=1)
        amplitude = torch.cat(
            [
                torch.log1p(horizon_rms),
                torch.log1p(history_rms),
                torch.log1p(global_rms[:, None]),
            ],
            dim=1,
        )
        feature = torch.cat([pooled, channel_state, amplitude], dim=1)
        return self.maximum_latent_abs * torch.tanh(self.latent_head(feature))

    def predict_coefficients(
        self, input_residual: torch.Tensor, channel: torch.Tensor
    ) -> torch.Tensor:
        latent = self.predict_latent(input_residual, channel)
        return torch.einsum("br,rph->bph", latent, self.coefficient_basis)

    def forward(
        self, input_residual: torch.Tensor, channel: torch.Tensor
    ) -> torch.Tensor:
        coefficient = self.predict_coefficients(input_residual, channel)
        correction = torch.einsum(
            "bph,bphk->bpk", coefficient, input_residual
        ).reshape(input_residual.shape[0], -1)
        return remove_affine_patch(correction)
