"""Level-style multi-sign head for identifiable Amp phase coordinates."""

from __future__ import annotations

import torch
from torch import nn


class AmpMultiPhaseSignHead(nn.Module):
    """Predict one keep/flip logit for each orthogonal phase coordinate."""

    def __init__(
        self,
        waveform_width: int,
        context_width: int,
        coordinate_count: int = 6,
    ) -> None:
        super().__init__()
        if waveform_width <= 0 or context_width <= 0 or coordinate_count <= 0:
            raise ValueError("all Amp multi-phase dimensions must be positive")
        self.waveform_width = int(waveform_width)
        self.context_width = int(context_width)
        self.coordinate_count = int(coordinate_count)
        self.waveform_encoder = nn.Sequential(
            nn.Linear(self.waveform_width, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(self.context_width, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, self.coordinate_count),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        final = self.fusion[-1]
        if not isinstance(final, nn.Linear):
            raise RuntimeError("unexpected Amp multi-phase output module")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self, waveform_features: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        if waveform_features.ndim != 2 or context.ndim != 2:
            raise ValueError("Amp multi-phase inputs must be rank two")
        if waveform_features.shape[0] != context.shape[0]:
            raise ValueError("Amp multi-phase batch sizes differ")
        if waveform_features.shape[1] != self.waveform_width:
            raise ValueError("unexpected Amp multi-phase waveform width")
        if context.shape[1] != self.context_width:
            raise ValueError("unexpected Amp multi-phase context width")
        waveform = self.waveform_encoder(waveform_features)
        semantic = self.context_encoder(context)
        return self.fusion(torch.cat([waveform, semantic], dim=1))


def project_phase_pairs_to_unit_ball(raw: torch.Tensor) -> torch.Tensor:
    """Bound each final-axis pair while preserving zero as an exact no-op."""

    if raw.ndim != 2 or raw.shape[1] % 2 != 0:
        raise ValueError("continuous Amp phase output must contain coordinate pairs")
    pair = raw.reshape(raw.shape[0], raw.shape[1] // 2, 2)
    norm = torch.linalg.vector_norm(pair, dim=2, keepdim=True)
    return pair / torch.clamp(norm, min=1.0)


class AmpContinuousPhaseHead(AmpMultiPhaseSignHead):
    """Predict confidence-bearing continuous cosine/sine vectors."""

    def __init__(
        self,
        waveform_width: int,
        context_width: int,
        frequency_count: int = 3,
    ) -> None:
        if frequency_count <= 0:
            raise ValueError("Amp frequency count must be positive")
        self.frequency_count = int(frequency_count)
        super().__init__(
            waveform_width,
            context_width,
            coordinate_count=2 * self.frequency_count,
        )

    def forward(
        self, waveform_features: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        raw = super().forward(waveform_features, context)
        return project_phase_pairs_to_unit_ball(raw)
