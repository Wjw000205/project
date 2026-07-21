"""Lightweight explicit temporal complex-phase state for the Amp adapter."""

from __future__ import annotations

import torch
from torch import nn

from src.models.amp_multi_phase_sign_head import project_phase_pairs_to_unit_ball


class AmpComplexPhaseStateAdapter(nn.Module):
    """Forecast confidence-bearing phase from a short complex state sequence."""

    def __init__(
        self,
        phase_state_width: int = 15,
        context_width: int = 339,
        base_state_width: int = 9,
        hidden_width: int = 12,
        context_bottleneck: int = 16,
        base_bottleneck: int = 8,
        fusion_width: int = 16,
        frequency_count: int = 3,
    ) -> None:
        super().__init__()
        dimensions = (
            phase_state_width,
            context_width,
            base_state_width,
            hidden_width,
            context_bottleneck,
            base_bottleneck,
            fusion_width,
            frequency_count,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("all complex-phase adapter dimensions must be positive")
        self.phase_state_width = int(phase_state_width)
        self.context_width = int(context_width)
        self.base_state_width = int(base_state_width)
        self.hidden_width = int(hidden_width)
        self.frequency_count = int(frequency_count)
        self.phase_dynamics = nn.GRU(
            input_size=self.phase_state_width,
            hidden_size=self.hidden_width,
            num_layers=1,
            batch_first=True,
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(self.context_width, context_bottleneck),
            nn.GELU(),
        )
        self.base_encoder = nn.Sequential(
            nn.Linear(self.base_state_width, base_bottleneck),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(
                self.hidden_width + context_bottleneck + base_bottleneck,
                fusion_width,
            ),
            nn.GELU(),
            nn.Linear(fusion_width, 2 * self.frequency_count),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        final = self.fusion[-1]
        if not isinstance(final, nn.Linear):
            raise RuntimeError("unexpected complex-phase output module")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        phase_state: torch.Tensor,
        context: torch.Tensor,
        base_state: torch.Tensor,
    ) -> torch.Tensor:
        if phase_state.ndim != 3:
            raise ValueError("complex phase state must be rank three")
        if context.ndim != 2 or base_state.ndim != 2:
            raise ValueError("complex phase context/base must be rank two")
        batch = phase_state.shape[0]
        if context.shape[0] != batch or base_state.shape[0] != batch:
            raise ValueError("complex phase batch sizes differ")
        if phase_state.shape[1] == 0:
            raise ValueError("complex phase sequence must be nonempty")
        if phase_state.shape[2] != self.phase_state_width:
            raise ValueError("unexpected complex phase state width")
        if context.shape[1] != self.context_width:
            raise ValueError("unexpected complex phase context width")
        if base_state.shape[1] != self.base_state_width:
            raise ValueError("unexpected complex phase base-state width")
        _sequence, hidden = self.phase_dynamics(phase_state)
        dynamics = hidden[-1]
        semantic = self.context_encoder(context)
        base = self.base_encoder(base_state)
        raw = self.fusion(torch.cat([dynamics, semantic, base], dim=1))
        return project_phase_pairs_to_unit_ball(raw)
