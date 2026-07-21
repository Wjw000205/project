"""Dataset- and horizon-agnostic periodic adapter geometry.

The learned architecture owns one canonical eight-token period and a fixed
28-period residual memory.  A caller may provide the physical number of
samples in a period, but neither a dataset name nor a forecast horizon changes
the carrier recipe, learned layers, or residual spaces.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.models.period_geometry import PeriodGeometry


@dataclass(frozen=True)
class AdapterKernelGeometry:
    """The canonical learned-kernel clock, shared by every dataset/horizon."""

    period_steps: int = 96
    patch_steps: int = 12
    patches_per_period: int = 8
    carrier_periods: int = 28

    def __post_init__(self) -> None:
        if self.period_steps != self.patch_steps * self.patches_per_period:
            raise ValueError("kernel period/patch geometry is inconsistent")
        if self.carrier_periods < 2:
            raise ValueError("the universal carrier memory needs at least two periods")


LOCKED_KERNEL_GEOMETRY = AdapterKernelGeometry()


@dataclass(frozen=True)
class PeriodicAdapterConfig:
    """Only the physical sampling clock; all processing rules are derived."""

    native: PeriodGeometry
    kernel: AdapterKernelGeometry = LOCKED_KERNEL_GEOMETRY

    def __post_init__(self) -> None:
        if self.native.patches_per_period != self.kernel.patches_per_period:
            raise ValueError("native and kernel token counts must match")

    @property
    def carrier_period_ages(self) -> tuple[int, ...]:
        """Fixed complete-period ages, ordered from oldest to most recent."""

        return tuple(range(self.kernel.carrier_periods, 0, -1))

    @property
    def phase_vintage_offsets(self) -> tuple[int, ...]:
        """One physical period sampled at the canonical eight phase tokens."""

        return tuple(
            -self.native.period_steps + index * self.native.patch_steps
            for index in range(self.native.patches_per_period)
        )

    def kernel_horizon_steps(self, native_horizon: int) -> int:
        return self.native.patch_count(native_horizon) * self.kernel.patch_steps


ETTM1_NATIVE_GEOMETRY = PeriodGeometry(period_steps=96)
ELECTRICITY_NATIVE_GEOMETRY = PeriodGeometry(period_steps=24)

ETTM1_PERIODIC_ADAPTER_CONFIG = PeriodicAdapterConfig(
    native=ETTM1_NATIVE_GEOMETRY,
)

ELECTRICITY_PERIODIC_ADAPTER_CONFIG = PeriodicAdapterConfig(
    native=ELECTRICITY_NATIVE_GEOMETRY,
)


__all__ = [
    "AdapterKernelGeometry",
    "ELECTRICITY_NATIVE_GEOMETRY",
    "ELECTRICITY_PERIODIC_ADAPTER_CONFIG",
    "ETTM1_NATIVE_GEOMETRY",
    "ETTM1_PERIODIC_ADAPTER_CONFIG",
    "LOCKED_KERNEL_GEOMETRY",
    "PeriodicAdapterConfig",
]
