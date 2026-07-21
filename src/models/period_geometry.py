"""Learned-parameter-free physical period geometry.

This module deliberately knows nothing about adapter kernels or datasets.
It describes only a native sampling clock.  Dataset bindings and the locked
P96/p12 kernel clock live in ``periodic_adapter_config.py``; deterministic
conversion between the two lives in ``periodic_adapter_io.py``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PeriodGeometry:
    """Native physical sampling geometry for one dataset."""

    period_steps: int
    patches_per_period: int = 8
    periods_per_week: int = 7

    def __post_init__(self) -> None:
        if self.period_steps <= 1:
            raise ValueError("period_steps must exceed one")
        if self.patches_per_period <= 0:
            raise ValueError("patches_per_period must be positive")
        if self.period_steps % self.patches_per_period != 0:
            raise ValueError("period_steps must be divisible by patches_per_period")
        if self.periods_per_week <= 0:
            raise ValueError("periods_per_week must be positive")

    @property
    def patch_steps(self) -> int:
        return self.period_steps // self.patches_per_period

    @property
    def week_steps(self) -> int:
        return self.period_steps * self.periods_per_week

    def patch_count(self, horizon: int) -> int:
        horizon = int(horizon)
        if horizon <= 0 or horizon % self.patch_steps != 0:
            raise ValueError("horizon must be a positive multiple of patch_steps")
        return horizon // self.patch_steps

    def period_count(self, horizon: int) -> int:
        horizon = int(horizon)
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        return (horizon + self.period_steps - 1) // self.period_steps


__all__ = ["PeriodGeometry"]
