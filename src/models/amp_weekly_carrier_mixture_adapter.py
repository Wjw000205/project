"""Lightweight long-seasonal carrier mixture for frozen-base Amp repair."""

from __future__ import annotations

import torch
from torch import nn


def remove_affine_patch(values: torch.Tensor) -> torch.Tensor:
    """Remove patch mean and least-squares linear trend."""

    if values.ndim != 2:
        raise ValueError("Amp weekly correction must be rank two")
    centered = values - values.mean(dim=1, keepdim=True)
    time = torch.linspace(
        -1.0, 1.0, values.shape[1], dtype=values.dtype, device=values.device
    )
    time = time - time.mean()
    slope = (centered * time).sum(dim=1, keepdim=True) / time.square().sum()
    return centered - slope * time


class AmpWeeklyCarrierMixtureAdapter(nn.Module):
    """Predict small state-conditioned increments to a safe seasonal mixture."""

    def __init__(
        self,
        state_width: int = 531,
        carrier_count: int = 7,
        patch_len: int = 12,
        hidden_width: int = 24,
        maximum_coefficient_increment: float = 0.25,
    ) -> None:
        super().__init__()
        if min(state_width, carrier_count, patch_len, hidden_width) <= 0:
            raise ValueError("all weekly Amp adapter dimensions must be positive")
        if maximum_coefficient_increment <= 0.0:
            raise ValueError("maximum coefficient increment must be positive")
        self.state_width = int(state_width)
        self.carrier_count = int(carrier_count)
        self.patch_len = int(patch_len)
        self.maximum_coefficient_increment = float(
            maximum_coefficient_increment
        )
        self.coefficient_increment = nn.Sequential(
            nn.Linear(self.state_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, self.carrier_count),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        final = self.coefficient_increment[-1]
        if not isinstance(final, nn.Linear):
            raise RuntimeError("unexpected weekly Amp output module")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        state: torch.Tensor,
        carriers: torch.Tensor,
        baseline_coefficients: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if state.ndim != 2:
            raise ValueError("weekly Amp state must be rank two")
        if carriers.ndim != 3 or baseline_coefficients.ndim != 2:
            raise ValueError("weekly Amp carriers/coefficients have invalid rank")
        batch = state.shape[0]
        if state.shape != (batch, self.state_width):
            raise ValueError("unexpected weekly Amp state width")
        if carriers.shape != (batch, self.carrier_count, self.patch_len):
            raise ValueError("unexpected weekly Amp carrier shape")
        if baseline_coefficients.shape != (batch, self.carrier_count):
            raise ValueError("unexpected weekly Amp coefficient shape")
        increment = self.maximum_coefficient_increment * torch.tanh(
            self.coefficient_increment(state)
        )
        coefficients = baseline_coefficients + increment
        raw_correction = torch.einsum("bc,bcp->bp", coefficients, carriers)
        correction = remove_affine_patch(raw_correction)
        return {
            "correction": correction,
            "coefficients": coefficients,
            "coefficient_increment": increment,
        }
