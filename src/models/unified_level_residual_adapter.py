"""Light Level adapter over semantic state and matured Level residual history."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class UnifiedLevelOutput:
    correction: torch.Tensor
    coefficient: torch.Tensor


class UnifiedLevelResidualAdapter(nn.Module):
    """Emit one constant residual correction without modifying the backbone."""

    def __init__(
        self,
        semantic_width: int,
        history_lags: int = 8,
        semantic_hidden: int = 48,
        history_hidden: int = 16,
        fusion_hidden: int = 48,
        max_abs_coefficient: float = 2.0,
    ) -> None:
        super().__init__()
        if semantic_width <= 0 or history_lags <= 0:
            raise ValueError("Level adapter widths must be positive")
        if min(semantic_hidden, history_hidden, fusion_hidden) <= 0:
            raise ValueError("Level adapter hidden widths must be positive")
        if max_abs_coefficient <= 0.0:
            raise ValueError("max_abs_coefficient must be positive")
        self.semantic_width = int(semantic_width)
        self.history_lags = int(history_lags)
        self.max_abs_coefficient = float(max_abs_coefficient)
        self.semantic_encoder = nn.Sequential(
            nn.Linear(self.semantic_width, semantic_hidden),
            nn.SiLU(),
        )
        self.history_projection = nn.Sequential(
            nn.Linear(1, history_hidden),
            nn.SiLU(),
        )
        self.history_encoder = nn.GRU(
            history_hidden, history_hidden, batch_first=True
        )
        self.fusion = nn.Sequential(
            nn.Linear(semantic_hidden + history_hidden, fusion_hidden),
            nn.SiLU(),
            nn.Linear(fusion_hidden, 1),
        )
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def predict_coefficient(
        self, semantic: torch.Tensor, matured_level: torch.Tensor
    ) -> torch.Tensor:
        if semantic.ndim != 2 or semantic.shape[1] != self.semantic_width:
            raise ValueError("semantic Level state shape mismatch")
        if matured_level.shape != (semantic.shape[0], self.history_lags):
            raise ValueError("matured Level history shape mismatch")
        semantic_state = self.semantic_encoder(semantic)
        history_token = self.history_projection(matured_level.unsqueeze(-1))
        _history_output, history_state = self.history_encoder(history_token)
        raw = self.fusion(
            torch.cat([semantic_state, history_state[-1]], dim=-1)
        ).squeeze(-1)
        return self.max_abs_coefficient * torch.tanh(
            raw / self.max_abs_coefficient
        )

    def forward(
        self,
        semantic: torch.Tensor,
        matured_level: torch.Tensor,
        horizon: int,
    ) -> UnifiedLevelOutput:
        if int(horizon) <= 0:
            raise ValueError("Level correction horizon must be positive")
        coefficient = self.predict_coefficient(semantic, matured_level)
        correction = coefficient[:, None].expand(-1, int(horizon))
        return UnifiedLevelOutput(correction=correction, coefficient=coefficient)


__all__ = ["UnifiedLevelOutput", "UnifiedLevelResidualAdapter"]
