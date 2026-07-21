"""Full-memory universal physical-period Shape adapter."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PeriodicShapeSequenceOutput:
    memory_weights: torch.Tensor
    seasonal_anchor: torch.Tensor
    residual: torch.Tensor
    action_strength: torch.Tensor
    raw_correction: torch.Tensor


class UniversalPeriodicShapeSequenceAdapter(nn.Module):
    """Decode Shape from 28 complete causal physical-period residual curves.

    The class has no dataset, native period, horizon, channel count, physical
    block count, or feature-width constructor fields.  Native clocks are
    converted parameter-free before this fixed canonical kernel is called.
    """

    query_width = 303
    memory_periods = 28
    canonical_steps = 96
    model_width = 64
    parameter_count = 109282

    def __init__(
        self,
        *,
        max_residual: float = 0.50,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if max_residual <= 0.0 or not 0.0 <= dropout < 1.0:
            raise ValueError("invalid universal Shape-sequence configuration")
        self.max_residual = float(max_residual)
        self.query_encoder = nn.Sequential(
            nn.Linear(self.query_width, self.model_width),
            nn.SiLU(),
            nn.Linear(self.model_width, self.model_width),
        )
        self.memory_encoder = nn.Sequential(
            nn.Linear(self.canonical_steps, self.model_width),
            nn.SiLU(),
            nn.Linear(self.model_width, self.model_width),
        )
        self.position_embedding = nn.Parameter(
            torch.empty(1 + self.memory_periods, self.model_width)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.model_width,
            nhead=4,
            dim_feedforward=128,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sequence_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.output_norm = nn.LayerNorm(self.model_width)
        self.memory_score = nn.Linear(self.model_width, 1)
        self.residual_head = nn.Linear(self.model_width, self.canonical_steps)
        self.action_head = nn.Linear(self.model_width, 1)

        for module in (self.query_encoder, self.memory_encoder):
            for layer_module in module:
                if isinstance(layer_module, nn.Linear):
                    nn.init.xavier_uniform_(layer_module.weight)
                    nn.init.zeros_(layer_module.bias)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        nn.init.zeros_(self.memory_score.weight)
        nn.init.zeros_(self.memory_score.bias)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        anchors: torch.Tensor,
    ) -> PeriodicShapeSequenceOutput:
        rows = int(query.shape[0]) if query.ndim == 2 else -1
        if query.shape != (rows, self.query_width):
            raise ValueError("Shape-sequence query shape mismatch")
        expected = (rows, self.memory_periods, self.canonical_steps)
        if tuple(memory.shape) != expected or tuple(anchors.shape) != expected:
            raise ValueError("Shape-sequence memory/anchor shape mismatch")
        query_token = self.query_encoder(query).unsqueeze(1)
        memory_token = self.memory_encoder(memory)
        token = torch.cat([query_token, memory_token], dim=1)
        token = token + self.position_embedding.to(dtype=token.dtype)
        encoded = self.output_norm(self.sequence_encoder(token))
        query_state = encoded[:, 0]
        memory_state = encoded[:, 1:]
        weights = torch.softmax(self.memory_score(memory_state).squeeze(2), dim=1)
        seasonal = torch.sum(weights[:, :, None] * anchors, dim=1)
        residual = self.max_residual * torch.tanh(
            self.residual_head(query_state) / self.max_residual
        )
        action = torch.tanh(self.action_head(query_state).squeeze(1))
        raw = action[:, None] * (seasonal + residual)
        return PeriodicShapeSequenceOutput(
            memory_weights=weights,
            seasonal_anchor=seasonal,
            residual=residual,
            action_strength=action,
            raw_correction=raw,
        )


class UniversalPeriodicShapeSequenceActionAdapter(
    UniversalPeriodicShapeSequenceAdapter
):
    """Same full-memory direction with a stronger isolated Shape action MLP."""

    parameter_count = 113442
    action_parameter_count = 4225

    def __init__(
        self,
        *,
        max_residual: float = 0.50,
        dropout: float = 0.05,
    ) -> None:
        super().__init__(max_residual=max_residual, dropout=dropout)
        self.action_head = nn.Sequential(
            nn.Linear(self.model_width, self.model_width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.model_width, 1),
        )
        nn.init.xavier_uniform_(self.action_head[0].weight)
        nn.init.zeros_(self.action_head[0].bias)
        nn.init.zeros_(self.action_head[3].weight)
        nn.init.zeros_(self.action_head[3].bias)


class UniversalPeriodicShapeMemoryActionAdapter(
    UniversalPeriodicShapeSequenceAdapter
):
    """Universal Shape with an action encoder that retains all causal memory.

    The waveform direction and signed applicability paths have distinct Shape-
    owned parameters. The latter sees the same fixed query and 28 canonical
    memories directly rather than a direction-optimized 64D bottleneck.
    """

    parameter_count = 212194
    action_parameter_count = 102977

    def __init__(
        self,
        *,
        max_residual: float = 0.50,
        dropout: float = 0.05,
    ) -> None:
        super().__init__(max_residual=max_residual, dropout=dropout)
        # Remove the base linear action instead of retaining unused parameters.
        self.action_head = nn.Identity()
        self.action_query_encoder = nn.Sequential(
            nn.Linear(self.query_width, self.model_width),
            nn.SiLU(),
            nn.Linear(self.model_width, self.model_width),
        )
        self.action_memory_encoder = nn.Sequential(
            nn.Linear(self.canonical_steps, self.model_width),
            nn.SiLU(),
            nn.Linear(self.model_width, self.model_width),
        )
        self.action_position_embedding = nn.Parameter(
            torch.empty(1 + self.memory_periods, self.model_width)
        )
        action_layer = nn.TransformerEncoderLayer(
            d_model=self.model_width,
            nhead=4,
            dim_feedforward=128,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.action_sequence_encoder = nn.TransformerEncoder(
            action_layer, num_layers=2
        )
        self.action_output_norm = nn.LayerNorm(self.model_width)
        self.action_output = nn.Linear(self.model_width, 1)

        for module in (self.action_query_encoder, self.action_memory_encoder):
            for layer_module in module:
                if isinstance(layer_module, nn.Linear):
                    nn.init.xavier_uniform_(layer_module.weight)
                    nn.init.zeros_(layer_module.bias)
        nn.init.normal_(self.action_position_embedding, mean=0.0, std=0.02)
        nn.init.zeros_(self.action_output.weight)
        nn.init.zeros_(self.action_output.bias)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        anchors: torch.Tensor,
    ) -> PeriodicShapeSequenceOutput:
        rows = int(query.shape[0]) if query.ndim == 2 else -1
        if query.shape != (rows, self.query_width):
            raise ValueError("Shape-sequence query shape mismatch")
        expected = (rows, self.memory_periods, self.canonical_steps)
        if tuple(memory.shape) != expected or tuple(anchors.shape) != expected:
            raise ValueError("Shape-sequence memory/anchor shape mismatch")

        direction_query = self.query_encoder(query).unsqueeze(1)
        direction_memory = self.memory_encoder(memory)
        direction_token = torch.cat([direction_query, direction_memory], dim=1)
        direction_token = direction_token + self.position_embedding.to(
            dtype=direction_token.dtype
        )
        direction_encoded = self.output_norm(
            self.sequence_encoder(direction_token)
        )
        direction_state = direction_encoded[:, 0]
        memory_state = direction_encoded[:, 1:]
        weights = torch.softmax(self.memory_score(memory_state).squeeze(2), dim=1)
        seasonal = torch.sum(weights[:, :, None] * anchors, dim=1)
        residual = self.max_residual * torch.tanh(
            self.residual_head(direction_state) / self.max_residual
        )

        action_query = self.action_query_encoder(query).unsqueeze(1)
        action_memory = self.action_memory_encoder(memory)
        action_token = torch.cat([action_query, action_memory], dim=1)
        action_token = action_token + self.action_position_embedding.to(
            dtype=action_token.dtype
        )
        action_state = self.action_output_norm(
            self.action_sequence_encoder(action_token)[:, 0]
        )
        action = torch.tanh(self.action_output(action_state).squeeze(1))
        raw = action[:, None] * (seasonal + residual)
        return PeriodicShapeSequenceOutput(
            memory_weights=weights,
            seasonal_anchor=seasonal,
            residual=residual,
            action_strength=action,
            raw_correction=raw,
        )


class UniversalPeriodicShapeSignedMemoryAdapter(
    UniversalPeriodicShapeSequenceAdapter
):
    """Predict a bounded signed mixture of 28 causal Shape memories."""

    parameter_count = 103042

    def __init__(
        self,
        *,
        max_residual: float = 0.50,
        dropout: float = 0.05,
    ) -> None:
        super().__init__(max_residual=max_residual, dropout=dropout)
        # The constrained decoder has neither the free waveform residual nor a
        # single global action.  Every memory gets its own signed coefficient.
        self.residual_head = nn.Identity()
        self.action_head = nn.Identity()
        self.memory_sign = nn.Linear(self.model_width, 1)
        nn.init.zeros_(self.memory_sign.weight)
        nn.init.zeros_(self.memory_sign.bias)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        anchors: torch.Tensor,
    ) -> PeriodicShapeSequenceOutput:
        rows = int(query.shape[0]) if query.ndim == 2 else -1
        if query.shape != (rows, self.query_width):
            raise ValueError("Shape-sequence query shape mismatch")
        expected = (rows, self.memory_periods, self.canonical_steps)
        if tuple(memory.shape) != expected or tuple(anchors.shape) != expected:
            raise ValueError("Shape-sequence memory/anchor shape mismatch")
        query_token = self.query_encoder(query).unsqueeze(1)
        memory_token = self.memory_encoder(memory)
        token = torch.cat([query_token, memory_token], dim=1)
        token = token + self.position_embedding.to(dtype=token.dtype)
        encoded = self.output_norm(self.sequence_encoder(token))
        memory_state = encoded[:, 1:]
        magnitude = torch.softmax(
            self.memory_score(memory_state).squeeze(2), dim=1
        )
        sign = torch.tanh(self.memory_sign(memory_state).squeeze(2))
        signed_weights = magnitude * sign
        correction = torch.sum(signed_weights[:, :, None] * anchors, dim=1)
        residual = torch.zeros_like(correction)
        action = torch.sum(torch.abs(signed_weights), dim=1)
        return PeriodicShapeSequenceOutput(
            memory_weights=signed_weights,
            seasonal_anchor=correction,
            residual=residual,
            action_strength=action,
            raw_correction=correction,
        )


class UniversalPeriodicShapePatchSignedMemoryAdapter(
    UniversalPeriodicShapeSequenceAdapter
):
    """Predict bounded signed memory weights independently on eight patches."""

    canonical_patches = 8
    patch_steps = 12
    parameter_count = 103952

    def __init__(
        self,
        *,
        max_residual: float = 0.50,
        dropout: float = 0.05,
    ) -> None:
        super().__init__(max_residual=max_residual, dropout=dropout)
        self.residual_head = nn.Identity()
        self.action_head = nn.Identity()
        self.memory_score = nn.Linear(self.model_width, self.canonical_patches)
        self.memory_sign = nn.Linear(self.model_width, self.canonical_patches)
        nn.init.zeros_(self.memory_score.weight)
        nn.init.zeros_(self.memory_score.bias)
        nn.init.zeros_(self.memory_sign.weight)
        nn.init.zeros_(self.memory_sign.bias)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        anchors: torch.Tensor,
    ) -> PeriodicShapeSequenceOutput:
        rows = int(query.shape[0]) if query.ndim == 2 else -1
        if query.shape != (rows, self.query_width):
            raise ValueError("Shape-sequence query shape mismatch")
        expected = (rows, self.memory_periods, self.canonical_steps)
        if tuple(memory.shape) != expected or tuple(anchors.shape) != expected:
            raise ValueError("Shape-sequence memory/anchor shape mismatch")
        query_token = self.query_encoder(query).unsqueeze(1)
        memory_token = self.memory_encoder(memory)
        token = torch.cat([query_token, memory_token], dim=1)
        token = token + self.position_embedding.to(dtype=token.dtype)
        memory_state = self.output_norm(self.sequence_encoder(token))[:, 1:]
        magnitude = torch.softmax(self.memory_score(memory_state), dim=1)
        sign = torch.tanh(self.memory_sign(memory_state))
        signed_weights = magnitude * sign
        anchor_patch = anchors.reshape(
            rows,
            self.memory_periods,
            self.canonical_patches,
            self.patch_steps,
        )
        correction = torch.sum(
            signed_weights[:, :, :, None] * anchor_patch, dim=1
        ).reshape(rows, self.canonical_steps)
        residual = torch.zeros_like(correction)
        action = torch.mean(torch.sum(torch.abs(signed_weights), dim=1), dim=1)
        return PeriodicShapeSequenceOutput(
            memory_weights=signed_weights,
            seasonal_anchor=correction,
            residual=residual,
            action_strength=action,
            raw_correction=correction,
        )


__all__ = [
    "PeriodicShapeSequenceOutput",
    "UniversalPeriodicShapeSequenceAdapter",
    "UniversalPeriodicShapeSequenceActionAdapter",
    "UniversalPeriodicShapeMemoryActionAdapter",
    "UniversalPeriodicShapeSignedMemoryAdapter",
    "UniversalPeriodicShapePatchSignedMemoryAdapter",
]
