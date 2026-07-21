"""Differentiable, parameter-free native-clock/kernel-clock conversion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from src.models.periodic_adapter_config import PeriodicAdapterConfig


def causal_aligned_carrier_indices(
    origins: np.ndarray,
    patch_count: int,
    config: PeriodicAdapterConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the one universal causal carrier layout.

    Carrier ``k`` is one phase vintage from the preceding physical period.
    This is the exact normalized form of the accepted ETTm1 KNN state: output
    patch ``p`` retains lead ``p`` while its residual target is aligned to
    historical phase ``k``.  The eight-token formula is shared by every native
    clock.
    """

    origins = np.asarray(origins, dtype=np.int64)
    patch_count = int(patch_count)
    if origins.ndim != 1 or origins.size == 0 or patch_count <= 0:
        raise ValueError("carrier indices require nonempty origins and patches")
    geometry = config.native
    patch = np.arange(patch_count, dtype=np.int64)
    patch_offset = patch * geometry.patch_steps
    history_offset = np.asarray(config.phase_vintage_offsets, dtype=np.int64)
    source = (
        origins[:, None, None]
        + history_offset[None, None, :]
        - patch_offset[None, :, None]
    )
    patch_index = np.broadcast_to(patch[None, :, None], source.shape)
    return source, patch_index


def causal_period_memory_indices(
    origins: np.ndarray,
    patch_count: int,
    config: PeriodicAdapterConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the fixed 28-period fully matured residual memory.

    Carrier ``k`` is one complete historical physical period, with the carrier
    axis ordered oldest-to-recent.  For forecast patch ``p`` we retain the same
    forecast lead ``p`` in the historical residual stream, but move its origin
    back by both the carrier age and the physical-period segment containing
    ``p``.  Therefore the carrier target interval has fully matured before the
    current origin, including for multi-period forecasts.

    The formula is identical for all datasets.  Horizon only determines the
    natural number of output patches and never changes the memory length or a
    learned parameter.
    """

    origins = np.asarray(origins, dtype=np.int64)
    patch_count = int(patch_count)
    if origins.ndim != 1 or origins.size == 0 or patch_count <= 0:
        raise ValueError("carrier indices require nonempty origins and patches")
    geometry = config.native
    patch = np.arange(patch_count, dtype=np.int64)
    forecast_period = patch // geometry.patches_per_period
    age = np.asarray(config.carrier_period_ages, dtype=np.int64)
    source = origins[:, None, None] - (
        forecast_period[None, :, None] + age[None, None, :]
    ) * geometry.period_steps
    patch_index = np.broadcast_to(
        patch[None, :, None], source.shape
    )
    return source, patch_index


def resample_last_dimension(values: torch.Tensor, target_steps: int) -> torch.Tensor:
    """Linearly resample the final dimension with no learned parameter."""

    target_steps = int(target_steps)
    if values.ndim < 1 or target_steps <= 0:
        raise ValueError("resampling expects a tensor and positive target_steps")
    source_steps = int(values.shape[-1])
    if source_steps <= 0:
        raise ValueError("resampling source width must be positive")
    if source_steps == target_steps:
        return values
    prefix = values.shape[:-1]
    resized = F.interpolate(
        values.reshape(-1, 1, source_steps),
        size=target_steps,
        mode="linear",
        align_corners=True,
    )
    return resized.reshape(*prefix, target_steps)


def project_affine_free_blocks(
    values: torch.Tensor,
    block_steps: int,
) -> torch.Tensor:
    """Remove constant and linear coordinates independently per block."""

    if values.ndim < 2:
        raise ValueError("block projection expects [...,H]")
    block_steps = int(block_steps)
    if block_steps <= 1:
        raise ValueError("block_steps must exceed one")
    result = torch.empty_like(values)
    for left in range(0, int(values.shape[-1]), block_steps):
        right = min(left + block_steps, int(values.shape[-1]))
        block = values[..., left:right]
        basis = torch.linspace(
            -1.0,
            1.0,
            right - left,
            device=values.device,
            dtype=values.dtype,
        )
        basis = basis - basis.mean()
        centered = block - block.mean(dim=-1, keepdim=True)
        coefficient = (centered * basis).sum(dim=-1, keepdim=True)
        coefficient = coefficient / basis.square().sum().clamp_min(1.0e-12)
        result[..., left:right] = centered - coefficient * basis
    return result


@dataclass(frozen=True)
class PeriodicAdapterIO:
    """Stateless bridge around an unchanged P96/p12 learned kernel."""

    config: PeriodicAdapterConfig

    @property
    def is_kernel_clock(self) -> bool:
        native = self.config.native
        kernel = self.config.kernel
        return (
            native.period_steps == kernel.period_steps
            and native.patch_steps == kernel.patch_steps
        )

    def native_patches_to_kernel(self, values: torch.Tensor) -> torch.Tensor:
        if int(values.shape[-1]) != self.config.native.patch_steps:
            raise ValueError("native patch width drift")
        return resample_last_dimension(values, self.config.kernel.patch_steps)

    def kernel_patches_to_native(self, values: torch.Tensor) -> torch.Tensor:
        if int(values.shape[-1]) != self.config.kernel.patch_steps:
            raise ValueError("kernel patch width drift")
        return resample_last_dimension(values, self.config.native.patch_steps)

    def native_horizon_to_kernel(self, values: torch.Tensor) -> torch.Tensor:
        horizon = int(values.shape[-1])
        patch_count = self.config.native.patch_count(horizon)
        patches = values.reshape(
            *values.shape[:-1], patch_count, self.config.native.patch_steps
        )
        encoded = self.native_patches_to_kernel(patches)
        return encoded.reshape(
            *values.shape[:-1], patch_count * self.config.kernel.patch_steps
        )

    def native_patch_bank_to_kernel_period_blocks(
        self,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """Map ``[B,native_patch,...,native_step]`` to physical blocks.

        The returned layout is ``[B,physical_period,8,...,12]``.  The
        physical-period axis is an ordinary batch/instance axis: it never
        creates a new adapter or a horizon-specific parameter set.
        """

        if values.ndim < 3:
            raise ValueError("native patch bank must have at least three axes")
        patch_count = int(values.shape[1])
        patches_per_period = self.config.native.patches_per_period
        if patch_count <= 0 or patch_count % patches_per_period != 0:
            raise ValueError("native patch bank must contain complete physical periods")
        if int(values.shape[-1]) != self.config.native.patch_steps:
            raise ValueError("native patch-bank width drift")
        encoded = self.native_patches_to_kernel(values)
        period_count = patch_count // patches_per_period
        return encoded.reshape(
            int(values.shape[0]),
            period_count,
            patches_per_period,
            *values.shape[2:-1],
            self.config.kernel.patch_steps,
        )

    def native_horizon_to_kernel_period_blocks(
        self,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """Map ``[B,H_native]`` to ``[B,physical_period,96]``."""

        if values.ndim != 2:
            raise ValueError("native horizon values must have shape [B,H]")
        horizon = int(values.shape[1])
        patch_count = self.config.native.patch_count(horizon)
        if patch_count % self.config.native.patches_per_period != 0:
            raise ValueError("native horizon must contain complete physical periods")
        patches = values.reshape(
            int(values.shape[0]), patch_count, self.config.native.patch_steps
        )
        blocks = self.native_patch_bank_to_kernel_period_blocks(patches)
        return blocks.reshape(
            int(values.shape[0]),
            patch_count // self.config.native.patches_per_period,
            self.config.kernel.period_steps,
        )

    def kernel_period_blocks_to_native_horizon(
        self,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """Invert canonical ``[B,physical_period,96]`` blocks to native time."""

        if values.ndim != 3 or int(values.shape[2]) != self.config.kernel.period_steps:
            raise ValueError("kernel period blocks must have shape [B,S,96]")
        patches = values.reshape(
            int(values.shape[0]),
            int(values.shape[1]) * self.config.kernel.patches_per_period,
            self.config.kernel.patch_steps,
        )
        native = self.kernel_patches_to_native(patches)
        return native.reshape(int(values.shape[0]), -1)

    def kernel_horizon_to_native(self, values: torch.Tensor) -> torch.Tensor:
        kernel_patch = self.config.kernel.patch_steps
        horizon = int(values.shape[-1])
        if horizon <= 0 or horizon % kernel_patch != 0:
            raise ValueError("kernel horizon must be divisible by kernel patch width")
        patch_count = horizon // kernel_patch
        patches = values.reshape(*values.shape[:-1], patch_count, kernel_patch)
        decoded = self.kernel_patches_to_native(patches)
        return decoded.reshape(
            *values.shape[:-1], patch_count * self.config.native.patch_steps
        )

    def latest_period_to_kernel(self, history: torch.Tensor) -> torch.Tensor:
        native_period = self.config.native.period_steps
        if int(history.shape[-1]) < native_period:
            raise ValueError("history does not contain one complete native period")
        latest = history[..., -native_period:]
        return resample_last_dimension(latest, self.config.kernel.period_steps)

    def decode_amp_correction(self, kernel_correction: torch.Tensor) -> torch.Tensor:
        """Return native Amp output; ETTm1 is an exact identity operation."""

        if self.is_kernel_clock:
            return kernel_correction
        native = self.kernel_horizon_to_native(kernel_correction)
        return project_affine_free_blocks(
            native, self.config.native.period_steps
        )


__all__ = [
    "PeriodicAdapterIO",
    "causal_aligned_carrier_indices",
    "causal_period_memory_indices",
    "project_affine_free_blocks",
    "resample_last_dimension",
]
