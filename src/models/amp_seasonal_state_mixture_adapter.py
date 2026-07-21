"""Light recurrent seasonal-state mixture for frozen-base Amp repair."""

from __future__ import annotations

import torch
from torch import nn

from src.models.amp_weekly_carrier_mixture_adapter import remove_affine_patch


class AmpSeasonalStateMixtureAdapter(nn.Module):
    """Encode ordered daily carriers and adjust four safe weekly experts."""

    def __init__(
        self,
        sequence_width: int = 12,
        context_width: int = 339,
        base_width: int = 12,
        expert_count: int = 4,
        hidden_width: int = 16,
        context_bottleneck: int = 12,
        base_bottleneck: int = 8,
        fusion_width: int = 16,
        maximum_coefficient_increment: float = 0.25,
    ) -> None:
        super().__init__()
        dimensions = (
            sequence_width,
            context_width,
            base_width,
            expert_count,
            hidden_width,
            context_bottleneck,
            base_bottleneck,
            fusion_width,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("all seasonal Amp dimensions must be positive")
        if maximum_coefficient_increment <= 0.0:
            raise ValueError("maximum coefficient increment must be positive")
        self.sequence_width = int(sequence_width)
        self.context_width = int(context_width)
        self.base_width = int(base_width)
        self.expert_count = int(expert_count)
        self.maximum_coefficient_increment = float(maximum_coefficient_increment)
        self.seasonal_dynamics = nn.GRU(
            input_size=self.sequence_width,
            hidden_size=hidden_width,
            batch_first=True,
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(self.context_width, context_bottleneck),
            nn.GELU(),
        )
        self.base_encoder = nn.Sequential(
            nn.Linear(self.base_width, base_bottleneck),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(
                hidden_width + context_bottleneck + base_bottleneck,
                fusion_width,
            ),
            nn.GELU(),
            nn.Linear(fusion_width, self.expert_count),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        final = self.fusion[-1]
        if not isinstance(final, nn.Linear):
            raise RuntimeError("unexpected seasonal Amp output module")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        seasonal_sequence: torch.Tensor,
        context: torch.Tensor,
        base: torch.Tensor,
        expert_shapes: torch.Tensor,
        baseline_coefficients: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if seasonal_sequence.ndim != 3 or expert_shapes.ndim != 3:
            raise ValueError("seasonal sequence/expert shapes must be rank three")
        if context.ndim != 2 or base.ndim != 2 or baseline_coefficients.ndim != 2:
            raise ValueError("seasonal Amp context/base/coefficients must be rank two")
        batch = seasonal_sequence.shape[0]
        if seasonal_sequence.shape[2] != self.sequence_width:
            raise ValueError("unexpected seasonal sequence width")
        if seasonal_sequence.shape[1] == 0:
            raise ValueError("seasonal sequence must be nonempty")
        if context.shape != (batch, self.context_width):
            raise ValueError("unexpected seasonal context shape")
        if base.shape != (batch, self.base_width):
            raise ValueError("unexpected seasonal base shape")
        if expert_shapes.shape[0] != batch or expert_shapes.shape[1] != self.expert_count:
            raise ValueError("unexpected seasonal expert shape")
        if baseline_coefficients.shape != (batch, self.expert_count):
            raise ValueError("unexpected seasonal baseline coefficient shape")
        _sequence, hidden = self.seasonal_dynamics(seasonal_sequence)
        state = torch.cat(
            [
                hidden[-1],
                self.context_encoder(context),
                self.base_encoder(base),
            ],
            dim=1,
        )
        increment = self.maximum_coefficient_increment * torch.tanh(
            self.fusion(state)
        )
        coefficients = baseline_coefficients + increment
        raw_correction = torch.einsum("be,bep->bp", coefficients, expert_shapes)
        correction = remove_affine_patch(raw_correction)
        return {
            "correction": correction,
            "coefficients": coefficients,
            "coefficient_increment": increment,
        }
