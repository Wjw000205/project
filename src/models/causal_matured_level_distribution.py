"""Causal soft-sign and learned-magnitude Level-coordinate adapter.

The adapter consumes one target-free current query and exactly 28 ordered,
fully matured source records.  A source record contains the target-free query
state that was available at the source origin plus its now-observed Level
residual coordinate.  Current targets never enter the module.

The same state encoder is used for the current query and every source query.
One recurrent encoder is evaluated with the real matured coordinates and with
those coordinates replaced by zero.  Only the paired hidden-state difference
can reach the output heads, which prevents query, channel, period, or network
biases from manufacturing a static correction template.

The module emits one scalar coordinate for one native phase patch.  Geometry-
specific reconstruction is intentionally external: callers place
``origin x channel x patch`` on the batch axis, reshape the shared outputs,
and decode only through ``project_period_patch_level``.  The patch lattice is
an orthogonal output basis, not a fixed position template; the same weights
own every patch's independently inferred direction, amount, and confidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class CausalMaturedLevelDistributionOutput:
    """One direct Level coordinate and its auditable factorization."""

    coordinate: torch.Tensor
    sign_probability: torch.Tensor
    soft_sign: torch.Tensor
    confidence: torch.Tensor
    magnitude: torch.Tensor
    magnitude_envelope: torch.Tensor


class CausalMaturedLevelDistributionAdapter(nn.Module):
    """Predict one Level coordinate from an ordered matured distribution.

    Input memory order is fixed from oldest to newest.  The first
    ``query_width`` token fields have exactly the same semantics as ``query``;
    the final field is the source's fully matured Level residual coordinate.
    Dataset, horizon, channel, physical-period index, and any other expert are
    deliberately absent from both the constructor and parameter shapes.
    """

    query_width = 16
    token_width = 17
    memory_periods = 28
    hidden_width = 16
    magnitude_quantile = 0.95
    parameter_count = 2_960

    _value_feature_width = 5

    def __init__(self) -> None:
        super().__init__()

        # Current and historical target-free states use one shared encoder.
        self.shared_state_encoder = nn.Sequential(
            nn.Linear(self.query_width, self.hidden_width),
            nn.SiLU(),
            nn.Linear(self.hidden_width, self.hidden_width),
        )
        self.ordered_memory_encoder = nn.GRU(
            input_size=self.hidden_width + self._value_feature_width,
            hidden_size=self.hidden_width,
            batch_first=True,
        )

        # The paired difference is the only path to either output head.
        self.paired_projection = nn.Linear(
            2 * self.hidden_width, self.hidden_width, bias=False
        )
        self.sign_head = nn.Linear(self.hidden_width, 1, bias=False)
        self.magnitude_head = nn.Linear(self.hidden_width, 1, bias=False)

        # p=0.5 makes a newly constructed model an exact NOOP while leaving
        # the non-negative magnitude path available for joint auxiliary training.
        nn.init.zeros_(self.sign_head.weight)

        if sum(parameter.numel() for parameter in self.parameters()) != self.parameter_count:
            raise RuntimeError("causal matured Level parameter-count drift")

    @classmethod
    def _validate_inputs(
        cls,
        query: torch.Tensor,
        memory_tokens: torch.Tensor,
        memory_mask: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor]:
        if query.ndim != 2 or query.shape[1] != cls.query_width:
            raise ValueError(
                f"Level query must have shape [B,{cls.query_width}]"
            )
        batch = int(query.shape[0])
        expected = (batch, cls.memory_periods, cls.token_width)
        if tuple(memory_tokens.shape) != expected:
            raise ValueError(
                "Level memory tokens must have shape "
                f"[B,{cls.memory_periods},{cls.token_width}]"
            )
        if not query.is_floating_point() or not memory_tokens.is_floating_point():
            raise ValueError("Level query and memory tokens must be floating point")
        if query.device != memory_tokens.device:
            raise ValueError("Level query and memory tokens must share a device")
        if query.dtype != memory_tokens.dtype:
            raise ValueError("Level query and memory tokens must share a dtype")
        if not bool(torch.isfinite(query).all()) or not bool(
            torch.isfinite(memory_tokens).all()
        ):
            raise ValueError("Level query and memory tokens must be finite")

        if memory_mask is None:
            mask = torch.ones(
                batch,
                cls.memory_periods,
                dtype=torch.bool,
                device=query.device,
            )
        else:
            if tuple(memory_mask.shape) != (batch, cls.memory_periods):
                raise ValueError(
                    f"Level memory mask must have shape [B,{cls.memory_periods}]"
                )
            if memory_mask.device != query.device:
                raise ValueError("Level memory mask must share the query device")
            if memory_mask.dtype == torch.bool:
                mask = memory_mask
            elif memory_mask.is_floating_point() or memory_mask.dtype in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            ):
                if not bool(((memory_mask == 0) | (memory_mask == 1)).all()):
                    raise ValueError("Level memory mask must be binary")
                mask = memory_mask.to(dtype=torch.bool)
            else:
                raise ValueError("Level memory mask must be boolean or binary")
        return batch, mask

    @classmethod
    def causal_magnitude_envelope(
        cls, matured_level: torch.Tensor, memory_mask: torch.Tensor
    ) -> torch.Tensor:
        """Masked linear-interpolation q95 of causal absolute coordinates."""

        if matured_level.ndim != 2 or matured_level.shape[1] != cls.memory_periods:
            raise ValueError(
                f"matured Level values must have shape [B,{cls.memory_periods}]"
            )
        if tuple(memory_mask.shape) != tuple(matured_level.shape):
            raise ValueError("matured Level values and memory mask must align")
        mask = memory_mask.to(dtype=torch.bool)
        absolute = matured_level.abs()
        sentinel = torch.full_like(absolute, torch.finfo(absolute.dtype).max)
        ordered = torch.sort(torch.where(mask, absolute, sentinel), dim=1).values
        count = mask.sum(dim=1)
        position = (count - 1).clamp_min(0).to(dtype=absolute.dtype)
        position = position * cls.magnitude_quantile
        lower = torch.floor(position).to(dtype=torch.long)
        upper = torch.ceil(position).to(dtype=torch.long)
        fraction = position - lower.to(dtype=position.dtype)
        lower_value = ordered.gather(1, lower[:, None]).squeeze(1)
        upper_value = ordered.gather(1, upper[:, None]).squeeze(1)
        envelope = lower_value + fraction * (upper_value - lower_value)
        return torch.where(count > 0, envelope, torch.zeros_like(envelope))

    def _memory_inputs(
        self,
        memory_state: torch.Tensor,
        normalized_value: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = memory_mask.to(dtype=memory_state.dtype)
        normalized_value = normalized_value * mask
        memory_state = memory_state * mask.unsqueeze(2)
        recency = torch.linspace(
            1.0 / self.memory_periods,
            1.0,
            self.memory_periods,
            device=memory_state.device,
            dtype=memory_state.dtype,
        ).unsqueeze(0)
        recency = recency.expand(memory_state.shape[0], -1) * mask
        real_value = torch.stack(
            [
                normalized_value,
                normalized_value.abs(),
                torch.sign(normalized_value),
                mask,
                recency,
            ],
            dim=2,
        )
        zero_value = torch.stack(
            [
                torch.zeros_like(normalized_value),
                torch.zeros_like(normalized_value),
                torch.zeros_like(normalized_value),
                mask,
                recency,
            ],
            dim=2,
        )
        return (
            torch.cat([memory_state, real_value], dim=2),
            torch.cat([memory_state, zero_value], dim=2),
        )

    @staticmethod
    def normalize_matured_level(
        matured_level: torch.Tensor,
        magnitude_envelope: torch.Tensor,
    ) -> torch.Tensor:
        """Scale causal coordinates without erasing amplitudes above q95.

        ``asinh`` is linear near zero and logarithmic in the tails.  Unlike
        winsorization it remains one-to-one and unbounded, so a two-envelope
        residual is still distinguishable from a one-envelope residual while
        isolated historical spikes cannot numerically dominate the GRU.
        """

        if matured_level.ndim != 2 or magnitude_envelope.ndim != 1:
            raise ValueError("matured Level normalization shape mismatch")
        if matured_level.shape[0] != magnitude_envelope.shape[0]:
            raise ValueError("matured Level normalization batch mismatch")
        safe_envelope = magnitude_envelope.clamp_min(
            torch.finfo(matured_level.dtype).eps
        )
        normalized = torch.asinh(matured_level / safe_envelope[:, None])
        return torch.where(
            magnitude_envelope[:, None] > 0.0,
            normalized,
            torch.zeros_like(normalized),
        )

    def _paired_distribution_state(
        self,
        query: torch.Tensor,
        memory_tokens: torch.Tensor,
        memory_mask: torch.Tensor,
        magnitude_envelope: torch.Tensor,
    ) -> torch.Tensor:
        query_state = self.shared_state_encoder(query)
        memory_state = self.shared_state_encoder(
            memory_tokens[:, :, : self.query_width]
        )
        matured_level = memory_tokens[:, :, -1]
        normalized = self.normalize_matured_level(
            matured_level, magnitude_envelope
        )
        real_input, zero_input = self._memory_inputs(
            memory_state, normalized, memory_mask
        )
        initial = query_state.unsqueeze(0)
        _real_sequence, real_final = self.ordered_memory_encoder(
            real_input, initial
        )
        _zero_sequence, zero_final = self.ordered_memory_encoder(
            zero_input, initial
        )
        difference = real_final.squeeze(0) - zero_final.squeeze(0)
        gated = difference * torch.tanh(query_state)
        return torch.nn.functional.silu(
            self.paired_projection(torch.cat([difference, gated], dim=1))
        )

    def forward(
        self,
        query: torch.Tensor,
        memory_tokens: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
    ) -> CausalMaturedLevelDistributionOutput:
        _batch, mask = self._validate_inputs(query, memory_tokens, memory_mask)
        matured_level = memory_tokens[:, :, -1]
        envelope = self.causal_magnitude_envelope(matured_level, mask)
        hidden = self._paired_distribution_state(
            query, memory_tokens, mask, envelope
        )
        sign_logit = self.sign_head(hidden).squeeze(1)
        sign_probability = torch.sigmoid(sign_logit)
        soft_sign = torch.tanh(0.5 * sign_logit)
        confidence = soft_sign.abs()
        # The matured q95 is a causal unit, not an inference ceiling.  A
        # normalized softplus keeps magnitude non-negative and finite while
        # allowing the expert to recover more than one historical envelope
        # when the current state supports it.  Confidence, rather than an
        # arbitrary amplitude cap, suppresses uncertain actions.
        magnitude_factor = torch.nn.functional.softplus(
            self.magnitude_head(hidden).squeeze(1)
        ) / 0.6931471805599453
        magnitude = envelope * magnitude_factor
        coordinate = soft_sign * magnitude
        return CausalMaturedLevelDistributionOutput(
            coordinate=coordinate,
            sign_probability=sign_probability,
            soft_sign=soft_sign,
            confidence=confidence,
            magnitude=magnitude,
            magnitude_envelope=envelope,
        )


__all__ = [
    "CausalMaturedLevelDistributionAdapter",
    "CausalMaturedLevelDistributionOutput",
]
