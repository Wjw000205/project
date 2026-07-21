"""Dataset-independent direct Level residual state transition.

The physical history length is an external sequence-length hyperparameter.  It
never changes this module's learned shapes.  Every input token is built in the
Level coordinate (current target-free level displacement, its transition,
strictly matured Level feedback, and cross-channel summaries), and the output
is one direct signed Level residual coordinate.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class UniversalPeriodicLevelStateTransitionAdapter(nn.Module):
    """Fixed recurrent kernel for a variable number of physical periods."""

    feature_width = 6
    hidden_width = 16

    def __init__(self, max_abs_coordinate: float = 2.0) -> None:
        super().__init__()
        if not max_abs_coordinate > 0.0:
            raise ValueError("max_abs_coordinate must be positive")
        self.max_abs_coordinate = float(max_abs_coordinate)
        self.token_encoder = nn.Linear(self.feature_width, self.hidden_width)
        self.transition = nn.GRU(
            input_size=self.hidden_width,
            hidden_size=self.hidden_width,
            batch_first=True,
        )
        self.body = nn.Sequential(
            nn.Linear(self.hidden_width, self.hidden_width),
            nn.SiLU(),
            nn.Linear(self.hidden_width, 8),
            nn.SiLU(),
        )
        self.coordinate_head = nn.Linear(8, 1)

        nn.init.xavier_uniform_(self.token_encoder.weight)
        nn.init.zeros_(self.token_encoder.bias)
        for name, parameter in self.transition.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(parameter)
            else:
                nn.init.zeros_(parameter)
        for layer in (self.body[0], self.body[2]):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.coordinate_head.weight)
        nn.init.zeros_(self.coordinate_head.bias)

    def forward(
        self,
        level_state: torch.Tensor,
        output_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Emit a signed direct Level residual.

        Args:
            level_state: ``[rows, physical_periods, 6]`` target-free Level
                coordinate state, ordered oldest to newest.
            output_scale: ``[rows]`` positive causal unit conversion.
        """

        if level_state.ndim != 3 or level_state.shape[-1] != self.feature_width:
            raise ValueError("Level state must have shape [rows, periods, 6]")
        if level_state.shape[1] < 1:
            raise ValueError("at least one physical period is required")
        if output_scale.shape != level_state.shape[:1]:
            raise ValueError("Level output scale shape mismatch")
        if not torch.isfinite(level_state).all() or not torch.isfinite(output_scale).all():
            raise ValueError("Level state contains nonfinite values")
        if torch.any(output_scale <= 0.0):
            raise ValueError("Level output scale must be positive")

        encoded = F.silu(self.token_encoder(level_state))
        _sequence, final = self.transition(encoded)
        bounded = self.max_abs_coordinate * torch.tanh(
            self.coordinate_head(self.body(final[-1])).squeeze(-1)
        )
        return output_scale * bounded


__all__ = ["UniversalPeriodicLevelStateTransitionAdapter"]
