"""Scale-aware light H96 mixture over aligned input forecast residuals."""

from __future__ import annotations

import torch

from src.models.amp_h96_input_residual_mixture_adapter import (
    AmpH96InputResidualMixtureAdapter,
)


class AmpH96ScaleAwareResidualMixtureAdapter(
    AmpH96InputResidualMixtureAdapter
):
    """Expose carrier magnitude to the coefficient head without a heavy body."""

    def __init__(self, *args, amplitude_feature_count: int = 2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if amplitude_feature_count != 2:
            raise ValueError("scale-aware adapter requires two amplitude features")
        old_head = self.coefficient_head
        old_first = old_head[0]
        head_width = old_first.out_features
        expanded_input = (
            old_first.in_features
            + amplitude_feature_count * self.history_patch_count
        )
        self.coefficient_head = torch.nn.Sequential(
            torch.nn.Linear(expanded_input, head_width),
            torch.nn.GELU(),
            torch.nn.Linear(head_width, self.history_patch_count),
        )
        for module in self.coefficient_head:
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.zeros_(module.bias)
        final = self.coefficient_head[-1]
        torch.nn.init.zeros_(final.weight)
        torch.nn.init.zeros_(final.bias)

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
        patch_rms = torch.sqrt(input_residual.square().mean(dim=-1) + 1.0e-8)
        global_rms = torch.sqrt(
            input_residual.square().mean(dim=(1, 2, 3), keepdim=True) + 1.0e-8
        ).reshape(batch, 1, 1)
        absolute_amplitude = torch.log1p(patch_rms)
        relative_amplitude = torch.log(
            torch.clamp(patch_rms / global_rms, min=1.0e-4, max=1.0e4)
        )
        amplitude_state = torch.cat(
            [absolute_amplitude, relative_amplitude], dim=-1
        )
        coefficient_feature = torch.cat(
            [
                horizon_state,
                horizon_position,
                channel_state,
                amplitude_state,
            ],
            dim=-1,
        )
        return self.maximum_coefficient_abs * torch.tanh(
            self.coefficient_head(coefficient_feature)
        )
