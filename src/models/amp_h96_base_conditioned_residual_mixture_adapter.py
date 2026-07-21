"""Light H96 residual mixture conditioned on the frozen base forecast shape."""

from __future__ import annotations

import torch

from src.models.amp_h96_scale_aware_residual_mixture_adapter import (
    AmpH96ScaleAwareResidualMixtureAdapter,
)
from src.models.amp_weekly_carrier_mixture_adapter import remove_affine_patch


class AmpH96BaseConditionedResidualMixtureAdapter(
    AmpH96ScaleAwareResidualMixtureAdapter
):
    """Mix aligned residual carriers using explicit residual and base state."""

    def __init__(self, *args, base_width: int = 4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if base_width <= 0:
            raise ValueError("base width must be positive")
        self.base_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.patch_len, base_width),
            torch.nn.LayerNorm(base_width),
            torch.nn.GELU(),
        )
        old_head = self.coefficient_head
        old_first = old_head[0]
        head_width = old_first.out_features
        self.coefficient_head = torch.nn.Sequential(
            torch.nn.Linear(old_first.in_features + base_width + 2, head_width),
            torch.nn.GELU(),
            torch.nn.Linear(head_width, self.history_patch_count),
        )
        for module in (self.base_encoder, self.coefficient_head):
            for child in module.modules():
                if isinstance(child, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(child.weight)
                    torch.nn.init.zeros_(child.bias)
        final = self.coefficient_head[-1]
        torch.nn.init.zeros_(final.weight)
        torch.nn.init.zeros_(final.bias)

    def predict_coefficients(
        self,
        input_residual: torch.Tensor,
        base_forecast: torch.Tensor,
        channel: torch.Tensor,
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
        if base_forecast.shape != (
            batch,
            self.horizon_patch_count * self.patch_len,
        ):
            raise ValueError("base forecast must be one H96 vector per row")
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

        residual_patch_rms = torch.sqrt(
            input_residual.square().mean(dim=-1) + 1.0e-8
        )
        residual_global_rms = torch.sqrt(
            input_residual.square().mean(dim=(1, 2, 3), keepdim=True) + 1.0e-8
        ).reshape(batch, 1, 1)
        residual_amplitude = torch.cat(
            [
                torch.log1p(residual_patch_rms),
                torch.log(
                    torch.clamp(
                        residual_patch_rms / residual_global_rms,
                        min=1.0e-4,
                        max=1.0e4,
                    )
                ),
            ],
            dim=-1,
        )

        base_patch = base_forecast.reshape(
            batch, self.horizon_patch_count, self.patch_len
        )
        base_encoded = self.base_encoder(base_patch)
        base_patch_rms = torch.sqrt(base_patch.square().mean(dim=-1) + 1.0e-8)
        base_global_rms = torch.sqrt(
            base_patch.square().mean(dim=(1, 2), keepdim=True) + 1.0e-8
        ).reshape(batch, 1)
        base_amplitude = torch.stack(
            [
                torch.log1p(base_patch_rms),
                torch.log(
                    torch.clamp(
                        base_patch_rms / base_global_rms,
                        min=1.0e-4,
                        max=1.0e4,
                    )
                ),
            ],
            dim=-1,
        )
        coefficient_feature = torch.cat(
            [
                horizon_state,
                horizon_position,
                channel_state,
                residual_amplitude,
                base_encoded,
                base_amplitude,
            ],
            dim=-1,
        )
        return self.maximum_coefficient_abs * torch.tanh(
            self.coefficient_head(coefficient_feature)
        )

    def forward(
        self,
        input_residual: torch.Tensor,
        base_forecast: torch.Tensor,
        channel: torch.Tensor,
    ) -> torch.Tensor:
        coefficient = self.predict_coefficients(
            input_residual, base_forecast, channel
        )
        correction = torch.einsum(
            "bph,bphk->bpk", coefficient, input_residual
        ).reshape(input_residual.shape[0], -1)
        return remove_affine_patch(correction)
