"""Shared target-free feature construction for periodic penalty adapters.

Every function is dataset-name and forecast-horizon agnostic.  Physical
sampling enters only through :class:`PeriodGeometry`; all learned inputs are
mapped to the same canonical period/patch widths from ``PeriodicAdapterConfig``.
"""

from __future__ import annotations

import numpy as np

from src.models.period_geometry import PeriodGeometry
from src.models.periodic_adapter_config import PeriodicAdapterConfig
from src.models.periodic_adapter_io import causal_aligned_carrier_indices
from src.models.periodic_adapter_io import causal_period_memory_indices


def resample_numpy_last(values: np.ndarray, target_steps: int) -> np.ndarray:
    """Parameter-free align-corners linear resampling on the final axis."""

    values = np.asarray(values, dtype=np.float32)
    source_steps = int(values.shape[-1])
    target_steps = int(target_steps)
    if source_steps <= 0 or target_steps <= 0:
        raise ValueError("resampling widths must be positive")
    if source_steps == target_steps:
        return values
    position = np.linspace(0.0, source_steps - 1.0, target_steps)
    left = np.floor(position).astype(np.int64)
    right = np.minimum(left + 1, source_steps - 1)
    weight = position - left
    matrix = np.zeros((target_steps, source_steps), dtype=np.float32)
    row = np.arange(target_steps)
    matrix[row, left] += (1.0 - weight).astype(np.float32)
    matrix[row, right] += weight.astype(np.float32)
    return np.einsum("...s,ts->...t", values, matrix, optimize=True).astype(
        np.float32, copy=False
    )


def channel_descriptors(
    original: np.ndarray,
    normalized: np.ndarray,
    train_end: int,
    geometry: PeriodGeometry,
) -> np.ndarray:
    """Seven fixed distribution descriptors, shared by every channel set."""

    original = np.asarray(original)
    normalized = np.asarray(normalized)
    train_end = int(train_end)
    if original.shape != normalized.shape or original.ndim != 2:
        raise ValueError("channel descriptor arrays must share shape [T,C]")
    if not 0 < train_end <= original.shape[0]:
        raise ValueError("invalid channel descriptor training prefix")
    train_original = original[:train_end].astype(np.float64)
    train = normalized[:train_end].astype(np.float64)
    columns = [np.log(train_original.std(axis=0).clip(1.0e-6))]
    for lag in (1, geometry.period_steps, geometry.week_steps):
        if lag >= train_end:
            raise ValueError("training prefix is shorter than descriptor lag")
        columns.append((train[:-lag] * train[lag:]).mean(axis=0))
    columns.extend(
        [
            np.mean(train_original == 0.0, axis=0),
            np.diff(train, axis=0).std(axis=0),
            np.quantile(np.abs(train), 0.95, axis=0),
        ]
    )
    descriptor = np.stack(columns, axis=1)
    descriptor -= descriptor.mean(axis=0, keepdims=True)
    descriptor /= descriptor.std(axis=0, keepdims=True).clip(1.0e-6)
    if descriptor.shape[1] != 7 or not np.isfinite(descriptor).all():
        raise ValueError("invalid universal channel descriptors")
    return descriptor.astype(np.float32)


def materialize_aligned_carriers(
    stream: np.ndarray,
    origin0: int,
    origins: np.ndarray,
    channels: np.ndarray,
    patch_count: int,
    config: PeriodicAdapterConfig,
) -> np.ndarray:
    """Read the shared eight-phase KNN vintage bank from a residual stream."""

    origins = np.asarray(origins, dtype=np.int64)
    channels = np.asarray(channels, dtype=np.int64)
    if origins.shape != channels.shape:
        raise ValueError("carrier origin/channel rows must align")
    source, patch_index = causal_aligned_carrier_indices(
        origins, patch_count, config
    )
    source_index = source - int(origin0)
    if int(source_index.min()) < 0 or int(source_index.max()) >= stream.shape[0]:
        raise ValueError("carrier residual stream does not cover requested history")
    carrier = stream[source_index, channels[:, None, None], patch_index]
    expected = (
        origins.size,
        int(patch_count),
        config.native.patches_per_period,
        config.native.patch_steps,
    )
    if carrier.shape != expected:
        raise ValueError(f"carrier shape drift: {carrier.shape} != {expected}")
    return np.asarray(carrier, dtype=np.float32)


def materialize_period_memory(
    stream: np.ndarray,
    origin0: int,
    origins: np.ndarray,
    channels: np.ndarray,
    patch_count: int,
    config: PeriodicAdapterConfig,
) -> np.ndarray:
    """Read the shared 28-period fully matured residual memory."""

    origins = np.asarray(origins, dtype=np.int64)
    channels = np.asarray(channels, dtype=np.int64)
    if origins.shape != channels.shape:
        raise ValueError("period-memory origin/channel rows must align")
    source, patch_index = causal_period_memory_indices(
        origins, patch_count, config
    )
    source_index = source - int(origin0)
    if int(source_index.min()) < 0 or int(source_index.max()) >= stream.shape[0]:
        raise ValueError("period residual stream does not cover requested history")
    memory = stream[source_index, channels[:, None, None], patch_index]
    expected = (
        origins.size,
        int(patch_count),
        config.kernel.carrier_periods,
        config.native.patch_steps,
    )
    if memory.shape != expected:
        raise ValueError(f"period-memory shape drift: {memory.shape} != {expected}")
    return np.asarray(memory, dtype=np.float32)


def level_raw_features(
    normalized: np.ndarray,
    base: np.ndarray,
    residual: np.ndarray | None,
    origins: np.ndarray,
    channels: np.ndarray,
    patches: np.ndarray,
    config: PeriodicAdapterConfig,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Canonical Level inputs and shared patch-baseline-minus-Trend target."""

    geometry = config.native
    kernel = config.kernel
    origins = np.asarray(origins, dtype=np.int64)
    channels = np.asarray(channels, dtype=np.int64)
    patches = np.asarray(patches, dtype=np.int64)
    if not (origins.shape == channels.shape == patches.shape):
        raise ValueError("Level rows must align")
    native_history = normalized[
        origins[:, None]
        - geometry.period_steps
        + np.arange(geometry.period_steps, dtype=np.int64)[None],
        channels[:, None],
    ]
    path = resample_numpy_last(native_history, kernel.period_steps).astype(np.float64)
    mean = path.mean(axis=1, keepdims=True)
    std = path.std(axis=1, keepdims=True).clip(1.0e-6)
    last = path[:, -1:]
    hist_norm = (path - last) / std
    hist_delta = (path[:, -1:] - path[:, -2:-1]) / std
    time = np.linspace(-1.0, 1.0, kernel.period_steps)
    time -= time.mean()
    hist_slope = np.sum((path - mean) * time[None], axis=1, keepdims=True)
    hist_slope = hist_slope / np.sum(np.square(time)) / std
    history = np.concatenate(
        [hist_norm, (mean - last) / std, np.log(std), hist_delta, hist_slope],
        axis=1,
    )

    lead = patches[:, None] * geometry.patch_steps + np.arange(
        geometry.patch_steps, dtype=np.int64
    )[None]
    if int(lead.max()) >= base.shape[1]:
        raise ValueError("Level patch exceeds forecast horizon")
    base_native = np.take_along_axis(base, lead, axis=1)
    base_patch = resample_numpy_last(base_native, kernel.patch_steps).astype(np.float64)
    base_norm = (base_patch - last) / std
    slot = patches % geometry.patches_per_period
    position = np.linspace(-1.0, 1.0, geometry.patches_per_period)[slot, None]
    patch_feature = np.concatenate(
        [
            base_norm,
            base_norm.mean(axis=1, keepdims=True),
            base_norm.std(axis=1, keepdims=True),
            base_norm[:, :1],
            base_norm[:, -1:],
            base_norm[:, -1:] - base_norm[:, :1],
            position,
        ],
        axis=1,
    )
    features = np.concatenate([history, patch_feature], axis=1).astype(np.float32)
    expected_width = kernel.period_steps + 4 + kernel.patch_steps + 6
    if features.shape != (origins.size, expected_width):
        raise ValueError("Level canonical raw width drift")

    target_coordinate = None
    if residual is not None:
        residual = np.asarray(residual)
        block = patches // geometry.patches_per_period
        target_coordinate = np.empty(origins.size, dtype=np.float64)
        for current in np.unique(block):
            use = block == current
            left = int(current) * geometry.period_steps
            right = min(left + geometry.period_steps, residual.shape[1])
            if right <= left:
                raise ValueError("Level target block exceeds the forecast horizon")
            target_coordinate[use] = residual[use, left:right].mean(axis=1)
        target_coordinate = target_coordinate.astype(np.float32)
    return features, target_coordinate


def level_semantic_features(
    raw: np.ndarray,
    config: PeriodicAdapterConfig,
) -> np.ndarray:
    """Scale summaries over fixed canonical fractions of one period."""

    raw = np.asarray(raw)
    kernel = config.kernel
    history = raw[:, : kernel.period_steps].astype(np.float64)
    base_start = kernel.period_steps + 4
    base = raw[:, base_start : base_start + kernel.patch_steps].astype(np.float64)
    blocks: list[np.ndarray] = []
    means = []
    widths = tuple(kernel.period_steps // divisor for divisor in (8, 4, 2, 1))
    for width in widths:
        values = history[:, -width:]
        mean = values.mean(axis=1)
        std = values.std(axis=1)
        time = np.linspace(-1.0, 1.0, width)
        time -= time.mean()
        slope = np.sum((values - mean[:, None]) * time[None], axis=1)
        slope /= np.sum(np.square(time))
        delta = np.diff(values, axis=1)
        coherence = np.abs(delta.sum(axis=1)) / np.maximum(
            np.abs(delta).sum(axis=1), 1.0e-12
        )
        sign = np.sign(delta)
        last_sign = sign[:, -1:]
        same = (sign[:, ::-1] == last_sign) & (last_sign != 0.0)
        run = np.cumprod(same.astype(np.int8), axis=1).sum(axis=1) / float(width - 1)
        blocks.extend(
            [mean[:, None], std[:, None], slope[:, None], coherence[:, None], run[:, None]]
        )
        means.append(mean)
    means_array = np.stack(means, axis=1)
    blocks.append(np.diff(means_array, axis=1) * -1.0)
    base_mean = base.mean(axis=1)
    blocks.append(
        np.column_stack(
            [
                base_mean,
                base.std(axis=1),
                base[:, 0],
                base[:, -1],
                base[:, -1] - base[:, 0],
                raw[:, -1],
            ]
        )
    )
    blocks.append(base_mean[:, None] - means_array)
    blocks.append(raw[:, kernel.period_steps + 1 : kernel.period_steps + 4])
    result = np.concatenate(blocks, axis=1).astype(np.float32)
    if result.shape[1] != 36:
        raise ValueError("Level semantic width drift")
    return result


def level_context_features(
    origins: np.ndarray,
    channels: np.ndarray,
    patches: np.ndarray,
    descriptors: np.ndarray,
    geometry: PeriodGeometry,
) -> np.ndarray:
    """Fixed descriptor/phase/lead context with no channel-count dependency."""

    origins = np.asarray(origins, dtype=np.int64)
    channels = np.asarray(channels, dtype=np.int64)
    patches = np.asarray(patches, dtype=np.int64)
    descriptor = np.asarray(descriptors, dtype=np.float32)[channels]
    if descriptor.shape[1] != 7:
        raise ValueError("universal channel descriptor width must be seven")
    slot = patches % geometry.patches_per_period
    slot_onehot = np.eye(geometry.patches_per_period, dtype=np.float32)[slot]
    center = origins + patches * geometry.patch_steps + 0.5 * (geometry.patch_steps - 1)
    angle = 2.0 * np.pi * center / geometry.period_steps
    fourier = np.stack(
        [
            function(harmonic * angle)
            for harmonic in range(1, 5)
            for function in (np.sin, np.cos)
        ],
        axis=1,
    ).astype(np.float32)
    descriptor_interaction = (descriptor[:, :, None] * fourier[:, None]).reshape(
        origins.size, -1
    )
    slot_interaction = (slot_onehot[:, :, None] * fourier[:, None]).reshape(
        origins.size, -1
    )
    lead_periods = (
        patches * geometry.patch_steps + 0.5 * (geometry.patch_steps - 1)
    ) / geometry.period_steps
    weekly = 2.0 * np.pi * lead_periods / geometry.periods_per_week
    physical = np.stack(
        [lead_periods, np.log1p(lead_periods), np.sin(weekly), np.cos(weekly)],
        axis=1,
    ).astype(np.float32)
    result = np.concatenate(
        [
            descriptor,
            slot_onehot,
            fourier,
            descriptor_interaction,
            slot_interaction,
            physical,
        ],
        axis=1,
    ).astype(np.float32)
    if result.shape[1] != 147:
        raise ValueError("Level context width drift")
    return result


def level_feedback_features(
    stream: np.ndarray,
    origin0: int,
    origins: np.ndarray,
    channels: np.ndarray,
    patches: np.ndarray,
    descriptors: np.ndarray,
    geometry: PeriodGeometry,
) -> np.ndarray:
    """Seven matured same-phase residual profiles for every physical clock."""

    origins = np.asarray(origins, dtype=np.int64)
    channels = np.asarray(channels, dtype=np.int64)
    patches = np.asarray(patches, dtype=np.int64)
    slot_count = geometry.patches_per_period
    segment = patches // slot_count
    slot = patches % slot_count
    profiles = np.zeros((origins.size, 7, slot_count), dtype=np.float64)
    available = np.zeros_like(profiles, dtype=bool)
    for age in range(7):
        source = origins - (segment + age + 1) * geometry.period_steps
        source_index = source - int(origin0)
        if int(source_index.min()) < 0 or int(source_index.max()) >= stream.shape[0]:
            raise ValueError("Level feedback residual stream underflow")
        for current_slot in range(slot_count):
            source_patch = segment * slot_count + current_slot
            valid = source_patch < stream.shape[2]
            if np.any(valid):
                profiles[valid, age, current_slot] = stream[
                    source_index[valid], channels[valid], source_patch[valid]
                ].mean(axis=1)
                available[valid, age, current_slot] = True
    row = np.arange(origins.size)
    ages = np.arange(7)
    same = profiles[row[:, None], ages[None], slot[:, None]]
    same_available = available[row[:, None], ages[None], slot[:, None]]
    count = same_available.sum(axis=1).astype(np.float64)
    denominator = np.maximum(count, 1.0)
    mean7 = (same * same_available).sum(axis=1) / denominator
    centered = (same - mean7[:, None]) * same_available
    std7 = np.sqrt(np.square(centered).sum(axis=1) / denominator)
    abs_mean7 = (np.abs(same) * same_available).sum(axis=1) / denominator
    sign_mean7 = (np.sign(same) * same_available).sum(axis=1) / denominator
    latest = same[:, 0]
    oldest_index = np.maximum(count.astype(np.int64) - 1, 0)
    oldest = same[row, oldest_index]
    slope = np.where(count >= 2, (latest - oldest) / np.maximum(count - 1, 1), 0.0)
    mean_profile3 = (profiles[:, :3] * available[:, :3]).sum(axis=1) / np.maximum(
        available[:, :3].sum(axis=1), 1
    )
    lag1 = profiles[:, 0]
    patch_axis = np.arange(slot_count, dtype=np.float64)
    patch_centered = patch_axis - patch_axis.mean()
    lag1_count = np.maximum(available[:, 0].sum(axis=1), 1)
    lag1_mean = (lag1 * available[:, 0]).sum(axis=1) / lag1_count
    lag1_std = np.sqrt(
        np.square((lag1 - lag1_mean[:, None]) * available[:, 0]).sum(axis=1)
        / lag1_count
    )
    patch_slope = np.sum(lag1 * patch_centered[None], axis=1) / np.sum(
        np.square(patch_centered)
    )
    descriptor = np.asarray(descriptors, dtype=np.float64)[channels]
    slot_onehot = np.eye(slot_count, dtype=np.float64)[slot]
    interactions = np.concatenate(
        [
            latest[:, None] * descriptor,
            mean7[:, None] * descriptor,
            latest[:, None] * slot_onehot,
            mean7[:, None] * slot_onehot,
        ],
        axis=1,
    )
    result = np.concatenate(
        [
            lag1,
            same[:, [0, 1, 2, 6]],
            mean_profile3,
            np.column_stack([mean7, std7, abs_mean7, sign_mean7, slope, latest - mean7]),
            np.column_stack([lag1_mean, lag1_std, patch_slope, latest - lag1_mean]),
            (count / 7.0)[:, None],
            same_available[:, [0, 1, 2, 6]].astype(np.float64),
            interactions,
        ],
        axis=1,
    ).astype(np.float32)
    if result.shape[1] != 65 or not np.isfinite(result).all():
        raise ValueError("invalid Level feedback features")
    return result


def level_partial_maturity_features(
    stream: np.ndarray,
    origin0: int,
    origins: np.ndarray,
    channels: np.ndarray,
    period_instances: np.ndarray,
    geometry: PeriodGeometry,
) -> np.ndarray:
    """Fresh target-free Level bias from eight partially matured forecasts.

    For physical-period instance ``s`` and canonical prefix length ``k`` in
    ``1..8``, the source forecast starts at
    ``origin - s*period_steps - k*patch_steps``.  Reading the first ``k``
    patches of source period ``s`` therefore ends exactly at ``origin - 1``.
    Every value is observable at execution while remaining aligned to the same
    forecast-lead period as the current Level coordinate.

    P96/p12 and P24/p3 both map parameter-free to eight prefix means.  The
    returned 24D state contains those means, their ordered increments, and
    eight fixed trajectory summaries; it has no dataset or horizon field.
    """

    stream = np.asarray(stream)
    origins = np.asarray(origins, dtype=np.int64)
    channels = np.asarray(channels, dtype=np.int64)
    period_instances = np.asarray(period_instances, dtype=np.int64)
    if stream.ndim != 4:
        raise ValueError("Level partial-maturity stream must be [O,C,P,S]")
    if not (origins.shape == channels.shape == period_instances.shape):
        raise ValueError("Level partial-maturity row axes must align")
    if origins.ndim != 1:
        raise ValueError("Level partial-maturity rows must be one-dimensional")
    slots = int(geometry.patches_per_period)
    if slots != 8:
        raise ValueError("Level partial-maturity state requires eight patches per period")
    if origins.size == 0:
        return np.empty((0, 24), dtype=np.float32)
    if int(channels.min()) < 0 or int(channels.max()) >= stream.shape[1]:
        raise ValueError("Level partial-maturity channel is out of range")
    if int(period_instances.min()) < 0:
        raise ValueError("Level partial-maturity period instance is negative")
    patch_left = period_instances * slots
    if int((patch_left + slots).max()) > stream.shape[2]:
        raise ValueError("Level partial-maturity period exceeds residual stream")

    prefix = np.empty((origins.size, slots), dtype=np.float64)
    row = np.arange(origins.size)[:, None]
    for index, length in enumerate(range(1, slots + 1)):
        source = (
            origins
            - period_instances * int(geometry.period_steps)
            - length * int(geometry.patch_steps)
        )
        source_index = source - int(origin0)
        if int(source_index.min()) < 0 or int(source_index.max()) >= stream.shape[0]:
            raise ValueError("Level partial-maturity residual stream underflow")
        patches = patch_left[:, None] + np.arange(length, dtype=np.int64)[None]
        values = stream[source_index[:, None], channels[:, None], patches]
        prefix[:, index] = np.asarray(values, dtype=np.float64).mean(axis=(1, 2))

    increment = np.empty_like(prefix)
    increment[:, 0] = prefix[:, 0]
    increment[:, 1:] = np.diff(prefix, axis=1)
    axis = np.arange(slots, dtype=np.float64)
    centered_axis = axis - axis.mean()
    slope = np.sum(
        (prefix - prefix.mean(axis=1, keepdims=True)) * centered_axis[None],
        axis=1,
    ) / np.sum(np.square(centered_axis))
    summary = np.column_stack(
        [
            prefix.mean(axis=1),
            prefix.std(axis=1),
            prefix.min(axis=1),
            prefix.max(axis=1),
            np.sign(prefix).mean(axis=1),
            slope,
            prefix[:, 0] - prefix[:, -1],
            prefix[:, :3].mean(axis=1),
        ]
    )
    result = np.concatenate([prefix, increment, summary], axis=1).astype(np.float32)
    if result.shape != (origins.size, 24) or not np.isfinite(result).all():
        raise ValueError("invalid Level partial-maturity features")
    return result


def level_forecast_revision_features(
    base_stream: np.ndarray,
    origin0: int,
    origins: np.ndarray,
    channels: np.ndarray,
    period_instances: np.ndarray,
    geometry: PeriodGeometry,
) -> np.ndarray:
    """Target-free Level state from seven overlapping forecast revisions.

    For vintage age ``k`` in ``1..7`` patches, compare the current forecast for
    physical-period instance ``s`` with the older forecast at ``origin-k*p``
    over their exactly overlapping absolute timestamps.  Only frozen backbone
    predictions are read; realized targets and residuals are absent.

    Both P96/p12 and P24/p3 expose seven canonical revisions.  Together with
    ordered increments and ten fixed summaries they form a universal 24D
    state independent of dataset, horizon, channel count, and period count.
    """

    base_stream = np.asarray(base_stream)
    origins = np.asarray(origins, dtype=np.int64)
    channels = np.asarray(channels, dtype=np.int64)
    period_instances = np.asarray(period_instances, dtype=np.int64)
    if base_stream.ndim != 3:
        raise ValueError("Level forecast-revision base must be [O,C,H]")
    if not (origins.shape == channels.shape == period_instances.shape):
        raise ValueError("Level forecast-revision row axes must align")
    if origins.ndim != 1:
        raise ValueError("Level forecast-revision rows must be one-dimensional")
    slots = int(geometry.patches_per_period)
    if slots != 8:
        raise ValueError("Level forecast-revision state requires eight patches per period")
    if origins.size == 0:
        return np.empty((0, 24), dtype=np.float32)
    if int(channels.min()) < 0 or int(channels.max()) >= base_stream.shape[1]:
        raise ValueError("Level forecast-revision channel is out of range")
    if int(period_instances.min()) < 0:
        raise ValueError("Level forecast-revision period instance is negative")
    period_left = period_instances * int(geometry.period_steps)
    period_right = period_left + int(geometry.period_steps)
    if int(period_right.max()) > base_stream.shape[2]:
        raise ValueError("Level forecast-revision period exceeds base horizon")

    current_index = origins - int(origin0)
    if int(current_index.min()) < 0 or int(current_index.max()) >= base_stream.shape[0]:
        raise ValueError("Level forecast-revision current origin is unavailable")
    revision = np.empty((origins.size, slots - 1), dtype=np.float64)
    row = np.arange(origins.size)[:, None]
    for index, age in enumerate(range(1, slots)):
        source_index = origins - age * int(geometry.patch_steps) - int(origin0)
        if int(source_index.min()) < 0 or int(source_index.max()) >= base_stream.shape[0]:
            raise ValueError("Level forecast-revision vintage underflow")
        overlap = int(geometry.period_steps) - age * int(geometry.patch_steps)
        offset = np.arange(overlap, dtype=np.int64)[None]
        current_lead = period_left[:, None] + offset
        source_lead = period_left[:, None] + age * int(geometry.patch_steps) + offset
        current = base_stream[
            current_index[:, None], channels[:, None], current_lead
        ]
        source = base_stream[
            source_index[:, None], channels[:, None], source_lead
        ]
        revision[:, index] = np.asarray(current - source, dtype=np.float64).mean(
            axis=1
        )

    increment = np.empty_like(revision)
    increment[:, 0] = revision[:, 0]
    increment[:, 1:] = np.diff(revision, axis=1)
    axis = np.arange(slots - 1, dtype=np.float64)
    centered_axis = axis - axis.mean()
    slope = np.sum(
        (revision - revision.mean(axis=1, keepdims=True)) * centered_axis[None],
        axis=1,
    ) / np.sum(np.square(centered_axis))
    absolute = np.abs(revision)
    summary = np.column_stack(
        [
            revision.mean(axis=1),
            revision.std(axis=1),
            revision.min(axis=1),
            revision.max(axis=1),
            np.sign(revision).mean(axis=1),
            slope,
            revision[:, 0] - revision[:, -1],
            revision[:, :3].mean(axis=1),
            absolute.mean(axis=1),
            np.abs(revision.sum(axis=1))
            / np.maximum(absolute.sum(axis=1), 1.0e-12),
        ]
    )
    result = np.concatenate([revision, increment, summary], axis=1).astype(np.float32)
    if result.shape != (origins.size, 24) or not np.isfinite(result).all():
        raise ValueError("invalid Level forecast-revision features")
    return result


def amp_forecast_revision_features(
    base_stream: np.ndarray,
    origin0: int,
    origins: np.ndarray,
    channels: np.ndarray,
    geometry: PeriodGeometry,
) -> np.ndarray:
    """Target-free amplitude state from seven aligned forecast vintages.

    Each vintage compares only amplitude-owned observables over the exact
    overlapping absolute timestamps: horizon standard deviation,
    first-difference standard deviation, and peak-to-trough range.  Seven
    signed revisions for each observable plus three aggregate stability terms
    form a fixed 24D contract for both P96/p12 and P24/p3 clocks.
    """

    base_stream = np.asarray(base_stream)
    origins = np.asarray(origins, dtype=np.int64)
    channels = np.asarray(channels, dtype=np.int64)
    if base_stream.ndim != 3 or base_stream.shape[2] != 96:
        raise ValueError("Amp forecast-revision base must be [O,C,96]")
    if origins.ndim != 1 or origins.shape != channels.shape:
        raise ValueError("Amp forecast-revision row axes must align")
    if int(geometry.patches_per_period) != 8:
        raise ValueError("Amp forecast-revision state requires eight patches per period")
    if origins.size == 0:
        return np.empty((0, 24), dtype=np.float32)
    if int(channels.min()) < 0 or int(channels.max()) >= base_stream.shape[1]:
        raise ValueError("Amp forecast-revision channel is out of range")
    current_index = origins - int(origin0)
    if int(current_index.min()) < 0 or int(current_index.max()) >= base_stream.shape[0]:
        raise ValueError("Amp forecast-revision current origin is unavailable")

    standard = np.empty((origins.size, 7), dtype=np.float64)
    difference_standard = np.empty_like(standard)
    extent = np.empty_like(standard)
    for index, age in enumerate(range(1, 8)):
        shift = age * int(geometry.patch_steps)
        source_index = origins - shift - int(origin0)
        if int(source_index.min()) < 0 or int(source_index.max()) >= base_stream.shape[0]:
            raise ValueError("Amp forecast-revision vintage underflow")
        overlap = 96 - shift
        if overlap <= 1:
            raise ValueError("Amp forecast-revision overlap is too short")
        lead = np.arange(overlap, dtype=np.int64)[None]
        current = np.asarray(
            base_stream[current_index[:, None], channels[:, None], lead],
            dtype=np.float64,
        )
        source = np.asarray(
            base_stream[
                source_index[:, None],
                channels[:, None],
                lead + shift,
            ],
            dtype=np.float64,
        )
        standard[:, index] = current.std(axis=1) - source.std(axis=1)
        difference_standard[:, index] = np.diff(current, axis=1).std(
            axis=1
        ) - np.diff(source, axis=1).std(axis=1)
        extent[:, index] = (
            current.max(axis=1)
            - current.min(axis=1)
            - source.max(axis=1)
            + source.min(axis=1)
        )
    summary = np.column_stack(
        [
            standard.mean(axis=1),
            difference_standard.mean(axis=1),
            extent.mean(axis=1),
        ]
    )
    result = np.concatenate(
        [standard, difference_standard, extent, summary], axis=1
    ).astype(np.float32)
    if result.shape != (origins.size, 24) or not np.isfinite(result).all():
        raise ValueError("invalid Amp forecast-revision features")
    return result


def amp_under_physical_scalar_state(
    raw: np.ndarray,
    base_stream: np.ndarray,
    origin0: int,
    origins: np.ndarray,
    channels: np.ndarray,
    geometry: PeriodGeometry,
) -> np.ndarray:
    """28 matured AmpUnder deficits plus 24D target-free revision context."""

    raw = np.asarray(raw)
    base_stream = np.asarray(base_stream)
    origins = np.asarray(origins, dtype=np.int64)
    channels = np.asarray(channels, dtype=np.int64)
    if raw.ndim != 2 or base_stream.ndim != 3 or base_stream.shape[2] != 96:
        raise ValueError("AmpUnder scalar state needs raw [T,C] and base [O,C,96]")
    if raw.shape[1] != base_stream.shape[1]:
        raise ValueError("AmpUnder scalar state channel count mismatch")
    if origins.ndim != 1 or origins.shape != channels.shape:
        raise ValueError("AmpUnder scalar state row axes must align")
    if origins.size == 0:
        return np.empty((0, 52), dtype=np.float32)

    memory = np.empty((origins.size, 28), dtype=np.float64)
    lead = np.arange(96, dtype=np.int64)[None]
    for age in range(28):
        source_origin = origins - 96 - age * int(geometry.period_steps)
        source_index = source_origin - int(origin0)
        if int(source_index.min()) < 0 or int(source_index.max()) >= base_stream.shape[0]:
            raise ValueError("AmpUnder scalar memory is unavailable")
        target_index = source_origin[:, None] + lead
        prediction = np.asarray(base_stream[source_index, channels], dtype=np.float64)
        target = np.asarray(raw[target_index, channels[:, None]], dtype=np.float64)
        memory[:, age] = np.maximum(
            target.std(axis=1, ddof=1) - prediction.std(axis=1, ddof=1), 0.0
        )
    revisions = amp_forecast_revision_features(
        base_stream, origin0, origins, channels, geometry
    ).astype(np.float64)
    result = np.concatenate([memory, revisions], axis=1).astype(np.float32)
    if result.shape != (origins.size, 52) or not np.isfinite(result).all():
        raise ValueError("invalid AmpUnder physical scalar state")
    return result


def delta_matured_residual_carrier(
    raw: np.ndarray,
    base_stream: np.ndarray,
    origin0: int,
    origins: np.ndarray,
    channels: np.ndarray,
    geometry: PeriodGeometry,
) -> np.ndarray:
    """Eight complete, fully matured D1 forecast-residual curves.

    The returned layout is ``[row,vintage,96]`` with a fixed zero at the first
    D1 coordinate.  The newest source forecast ends at ``origin-1``; seven
    older sources are spaced by one native patch.  This is the uncompressed
    Delta-only causal carrier used by the operator-sequence adapter.
    """

    raw = np.asarray(raw)
    base_stream = np.asarray(base_stream)
    origins = np.asarray(origins, dtype=np.int64)
    channels = np.asarray(channels, dtype=np.int64)
    if raw.ndim != 2:
        raise ValueError("Delta matured residual raw series must be [T,C]")
    if base_stream.ndim != 3 or base_stream.shape[2] != 96:
        raise ValueError("Delta matured residual base must be [O,C,96]")
    if raw.shape[1] != base_stream.shape[1]:
        raise ValueError("Delta matured residual channel count mismatch")
    if origins.ndim != 1 or origins.shape != channels.shape:
        raise ValueError("Delta matured residual row axes must align")
    if int(geometry.patches_per_period) != 8:
        raise ValueError("Delta matured residual state requires eight patches")
    if origins.size == 0:
        return np.empty((0, 8, 96), dtype=np.float32)
    if int(channels.min()) < 0 or int(channels.max()) >= raw.shape[1]:
        raise ValueError("Delta matured residual channel is out of range")

    carrier = np.empty((origins.size, 8, 96), dtype=np.float64)
    lead = np.arange(96, dtype=np.int64)[None]
    for age in range(8):
        source_origin = origins - 96 - age * int(geometry.patch_steps)
        source_index = source_origin - int(origin0)
        if int(source_index.min()) < 0 or int(source_index.max()) >= base_stream.shape[0]:
            raise ValueError("Delta matured residual source forecast is unavailable")
        target_index = source_origin[:, None] + lead
        if int(target_index.min()) < 0 or int(target_index.max()) >= raw.shape[0]:
            raise ValueError("Delta matured residual target timestamps are unavailable")
        prediction = np.asarray(
            base_stream[source_index, channels], dtype=np.float64
        )
        target = np.asarray(raw[target_index, channels[:, None]], dtype=np.float64)
        residual = target - prediction
        carrier[:, age] = np.diff(
            residual, axis=1, prepend=residual[:, :1]
        )
    result = carrier.astype(np.float32)
    if result.shape != (origins.size, 8, 96) or not np.isfinite(result).all():
        raise ValueError("invalid Delta matured residual carrier")
    return result


def delta_physical_period_residual_memory(
    raw: np.ndarray,
    base_stream: np.ndarray,
    origin0: int,
    origins: np.ndarray,
    channels: np.ndarray,
    geometry: PeriodGeometry,
) -> np.ndarray:
    """28 complete, same-phase, fully matured Delta D1 residual memories.

    Memory zero starts at ``origin-96`` and ends at ``origin-1``.  Older
    memories move back by one physical period, never by one patch, so all
    curves retain the current origin's physical phase on both P96 and P24
    clocks.  The 28-memory count matches the universal causal coverage rule.
    """

    raw = np.asarray(raw)
    base_stream = np.asarray(base_stream)
    origins = np.asarray(origins, dtype=np.int64)
    channels = np.asarray(channels, dtype=np.int64)
    if raw.ndim != 2 or base_stream.ndim != 3 or base_stream.shape[2] != 96:
        raise ValueError("Delta physical memory inputs must be raw [T,C] and base [O,C,96]")
    if raw.shape[1] != base_stream.shape[1]:
        raise ValueError("Delta physical memory channel count mismatch")
    if origins.ndim != 1 or origins.shape != channels.shape:
        raise ValueError("Delta physical memory row axes must align")
    if origins.size == 0:
        return np.empty((0, 28, 96), dtype=np.float32)
    if int(channels.min()) < 0 or int(channels.max()) >= raw.shape[1]:
        raise ValueError("Delta physical memory channel is out of range")

    memory = np.empty((origins.size, 28, 96), dtype=np.float64)
    lead = np.arange(96, dtype=np.int64)[None]
    for age in range(28):
        source_origin = origins - 96 - age * int(geometry.period_steps)
        source_index = source_origin - int(origin0)
        if int(source_index.min()) < 0 or int(source_index.max()) >= base_stream.shape[0]:
            raise ValueError("Delta physical residual memory is unavailable")
        target_index = source_origin[:, None] + lead
        if int(target_index.min()) < 0 or int(target_index.max()) >= raw.shape[0]:
            raise ValueError("Delta physical residual target timestamps are unavailable")
        prediction = np.asarray(base_stream[source_index, channels], dtype=np.float64)
        target = np.asarray(raw[target_index, channels[:, None]], dtype=np.float64)
        residual = target - prediction
        memory[:, age] = np.diff(residual, axis=1, prepend=residual[:, :1])
    result = memory.astype(np.float32)
    if result.shape != (origins.size, 28, 96) or not np.isfinite(result).all():
        raise ValueError("invalid Delta physical residual memory")
    return result


def delta_physical_key_value_memory(
    raw: np.ndarray,
    base_stream: np.ndarray,
    origin0: int,
    origins: np.ndarray,
    channels: np.ndarray,
    geometry: PeriodGeometry,
) -> np.ndarray:
    """28 same-phase Delta key/value memories as ``[N,28,3,96]``.

    Each key is the source forecast's own ``D1(history), D1(base)`` state and
    its paired value is the subsequently matured actual
    ``D1(target-base)`` residual.  Current queries therefore compare like with
    like before retrieving a residual value.
    """

    raw = np.asarray(raw)
    base_stream = np.asarray(base_stream)
    origins = np.asarray(origins, dtype=np.int64)
    channels = np.asarray(channels, dtype=np.int64)
    if raw.ndim != 2 or base_stream.ndim != 3 or base_stream.shape[2] != 96:
        raise ValueError("Delta key/value inputs must be raw [T,C] and base [O,C,96]")
    if raw.shape[1] != base_stream.shape[1]:
        raise ValueError("Delta key/value channel count mismatch")
    if origins.ndim != 1 or origins.shape != channels.shape:
        raise ValueError("Delta key/value row axes must align")
    if origins.size == 0:
        return np.empty((0, 28, 3, 96), dtype=np.float32)
    if int(channels.min()) < 0 or int(channels.max()) >= raw.shape[1]:
        raise ValueError("Delta key/value channel is out of range")

    memory = np.empty((origins.size, 28, 3, 96), dtype=np.float64)
    lead = np.arange(96, dtype=np.int64)[None]
    history_lag = np.arange(-96, 0, dtype=np.int64)[None]
    for age in range(28):
        source_origin = origins - 96 - age * int(geometry.period_steps)
        source_index = source_origin - int(origin0)
        if int(source_index.min()) < 0 or int(source_index.max()) >= base_stream.shape[0]:
            raise ValueError("Delta key/value forecast memory is unavailable")
        history_index = source_origin[:, None] + history_lag
        target_index = source_origin[:, None] + lead
        if int(history_index.min()) < 0 or int(target_index.max()) >= raw.shape[0]:
            raise ValueError("Delta key/value raw timestamps are unavailable")
        history = np.asarray(raw[history_index, channels[:, None]], dtype=np.float64)
        prediction = np.asarray(base_stream[source_index, channels], dtype=np.float64)
        target = np.asarray(raw[target_index, channels[:, None]], dtype=np.float64)
        residual = target - prediction
        for lane, values in enumerate((history, prediction, residual)):
            memory[:, age, lane] = np.diff(
                values, axis=1, prepend=values[:, :1]
            )
    result = memory.astype(np.float32)
    if result.shape != (origins.size, 28, 3, 96) or not np.isfinite(result).all():
        raise ValueError("invalid Delta physical key/value memory")
    return result


def delta_physical_penalty_component_memory(
    component_d1_stream: np.ndarray,
    origin0: int,
    origins: np.ndarray,
    channels: np.ndarray,
    geometry: PeriodGeometry,
) -> np.ndarray:
    """Gather 28 source-local, fully decomposed Delta-component memories.

    ``component_d1_stream`` already contains the source forecast's actual
    penalty-native Delta correction after its own target-free orthogonal basis
    and risk guard, expressed as padded first differences.  The newest source
    ends at ``origin-1`` and every older value moves back by one physical
    period.  Thus both position and numerical value match the residual space
    that the current Delta adapter is trained to emit.
    """

    component_d1_stream = np.asarray(component_d1_stream)
    origins = np.asarray(origins, dtype=np.int64)
    channels = np.asarray(channels, dtype=np.int64)
    if component_d1_stream.ndim != 3 or component_d1_stream.shape[2] != 96:
        raise ValueError("Delta component stream must be [O,C,96]")
    if origins.ndim != 1 or origins.shape != channels.shape:
        raise ValueError("Delta component-memory row axes must align")
    if origins.size == 0:
        return np.empty((0, 28, 96), dtype=np.float32)
    if int(channels.min()) < 0 or int(channels.max()) >= component_d1_stream.shape[1]:
        raise ValueError("Delta component-memory channel is out of range")

    source_origin = (
        origins[:, None]
        - 96
        - np.arange(28, dtype=np.int64)[None] * int(geometry.period_steps)
    )
    source_index = source_origin - int(origin0)
    if int(source_index.min()) < 0 or int(source_index.max()) >= component_d1_stream.shape[0]:
        raise ValueError("Delta component memory is unavailable")
    memory = component_d1_stream[source_index, channels[:, None]]
    result = np.asarray(memory, dtype=np.float32)
    if result.shape != (origins.size, 28, 96) or not np.isfinite(result).all():
        raise ValueError("invalid Delta penalty-component memory")
    return result


def delta_component_memory_locator_state(
    component_d1_stream: np.ndarray,
    base_stream: np.ndarray,
    origin0: int,
    origins: np.ndarray,
    channels: np.ndarray,
    geometry: PeriodGeometry,
) -> np.ndarray:
    """Clean Delta values plus target-free current-position locator state.

    Lanes ``0:28`` are the only residual values: source-local, orthogonally
    decomposed, risk-guarded Delta components. Lanes ``28:35`` contain seven
    aligned D1 backbone forecast revisions and lanes ``35:42`` their exact
    availability masks. Consumers must use the latter fourteen lanes only to
    suppress/localize the former values, never as residual output.
    """

    component_d1_stream = np.asarray(component_d1_stream)
    base_stream = np.asarray(base_stream)
    origins = np.asarray(origins, dtype=np.int64)
    channels = np.asarray(channels, dtype=np.int64)
    if component_d1_stream.ndim != 3 or component_d1_stream.shape[2] != 96:
        raise ValueError("Delta locator component stream must be [O,C,96]")
    if base_stream.ndim != 3 or base_stream.shape[2] != 96:
        raise ValueError("Delta locator backbone stream must be [O,C,96]")
    if component_d1_stream.shape[:2] != base_stream.shape[:2]:
        raise ValueError("Delta locator component/backbone streams must align")
    if origins.ndim != 1 or origins.shape != channels.shape:
        raise ValueError("Delta locator state row axes must align")
    if int(geometry.patches_per_period) != 8:
        raise ValueError("Delta locator state requires eight patches")
    if origins.size == 0:
        return np.empty((0, 42, 96), dtype=np.float32)

    state = np.zeros((origins.size, 42, 96), dtype=np.float32)
    state[:, :28] = delta_physical_penalty_component_memory(
        component_d1_stream, origin0, origins, channels, geometry
    )
    current_index = origins - int(origin0)
    if int(current_index.min()) < 0 or int(current_index.max()) >= base_stream.shape[0]:
        raise ValueError("Delta locator current forecast is unavailable")
    for index, age in enumerate(range(1, 8)):
        shift = age * int(geometry.patch_steps)
        overlap = 96 - shift
        source_index = current_index - shift
        if int(source_index.min()) < 0 or int(source_index.max()) >= base_stream.shape[0]:
            raise ValueError("Delta locator older forecast is unavailable")
        current = np.asarray(
            base_stream[current_index, channels, :overlap], dtype=np.float64
        )
        source = np.asarray(
            base_stream[source_index, channels, shift : shift + overlap],
            dtype=np.float64,
        )
        difference = current - source
        revision = np.diff(difference, axis=1, prepend=difference[:, :1])
        state[:, 28 + index, :overlap] = revision.astype(np.float32)
        state[:, 35 + index, :overlap] = 1.0
    if state.shape != (origins.size, 42, 96) or not np.isfinite(state).all():
        raise ValueError("invalid Delta component-memory locator state")
    return state


def direction_physical_period_residual_memory(
    raw: np.ndarray,
    base_stream: np.ndarray,
    origin0: int,
    origins: np.ndarray,
    channels: np.ndarray,
    geometry: PeriodGeometry,
) -> np.ndarray:
    """28 same-phase memories of the actual one-sided Direction residual."""

    raw = np.asarray(raw)
    base_stream = np.asarray(base_stream)
    origins = np.asarray(origins, dtype=np.int64)
    channels = np.asarray(channels, dtype=np.int64)
    if raw.ndim != 2 or base_stream.ndim != 3 or base_stream.shape[2] != 96:
        raise ValueError("Direction physical memory inputs must be raw [T,C] and base [O,C,96]")
    if raw.shape[1] != base_stream.shape[1]:
        raise ValueError("Direction physical memory channel count mismatch")
    if origins.ndim != 1 or origins.shape != channels.shape:
        raise ValueError("Direction physical memory row axes must align")
    if origins.size == 0:
        return np.empty((0, 28, 96), dtype=np.float32)
    if int(channels.min()) < 0 or int(channels.max()) >= raw.shape[1]:
        raise ValueError("Direction physical memory channel is out of range")

    memory = np.empty((origins.size, 28, 96), dtype=np.float64)
    lead = np.arange(96, dtype=np.int64)[None]
    for age in range(28):
        source_origin = origins - 96 - age * int(geometry.period_steps)
        source_index = source_origin - int(origin0)
        if int(source_index.min()) < 0 or int(source_index.max()) >= base_stream.shape[0]:
            raise ValueError("Direction physical residual memory is unavailable")
        target_index = source_origin[:, None] + lead
        prediction = np.asarray(base_stream[source_index, channels], dtype=np.float64)
        target = np.asarray(raw[target_index, channels[:, None]], dtype=np.float64)
        target_d1 = np.diff(target, axis=1)
        prediction_d1 = np.diff(prediction, axis=1)
        deficit = np.maximum(
            np.abs(target_d1) - prediction_d1 * np.sign(target_d1), 0.0
        )
        operator_residual = np.sign(target_d1) * deficit
        memory[:, age] = np.concatenate(
            [np.zeros((origins.size, 1)), operator_residual], axis=1
        )
    result = memory.astype(np.float32)
    if result.shape != (origins.size, 28, 96) or not np.isfinite(result).all():
        raise ValueError("invalid Direction physical residual memory")
    return result


def delta_matured_residual_features(
    raw: np.ndarray,
    base_stream: np.ndarray,
    origin0: int,
    origins: np.ndarray,
    channels: np.ndarray,
    geometry: PeriodGeometry,
) -> np.ndarray:
    """64D summary of the uncompressed Delta-only matured carrier.

    The newest source forecast ends at ``origin-1``; seven older sources are
    spaced by one native patch. Only signed and absolute p12 means of D1
    residuals are retained. Latest, mean-3, mean-8, and vintage standard
    deviation give a fixed 4x16 contract for both supported physical clocks.
    """

    origins = np.asarray(origins, dtype=np.int64)
    carrier = delta_matured_residual_carrier(
        raw, base_stream, origin0, origins, channels, geometry
    ).astype(np.float64)
    summaries = np.empty((origins.size, 8, 16), dtype=np.float64)
    for age in range(8):
        patches = carrier[:, age].reshape(origins.size, 8, 12)
        summaries[:, age, :8] = patches.mean(axis=2)
        summaries[:, age, 8:] = np.abs(patches).mean(axis=2)
    result = np.concatenate(
        [
            summaries[:, 0],
            summaries[:, :3].mean(axis=1),
            summaries.mean(axis=1),
            summaries.std(axis=1),
        ],
        axis=1,
    ).astype(np.float32)
    if result.shape != (origins.size, 64) or not np.isfinite(result).all():
        raise ValueError("invalid Delta matured residual features")
    return result


def diff_amp_matured_residual_features(
    raw: np.ndarray,
    base_stream: np.ndarray,
    origin0: int,
    origins: np.ndarray,
    channels: np.ndarray,
    geometry: PeriodGeometry,
) -> np.ndarray:
    """36D DiffAmp-only state from matured D1-volatility errors.

    Each of eight fully matured forecast vintages contributes the exact global
    D1 standard-deviation error used by ``penalty_diff_amp`` plus eight local
    padded-p12 D1 standard-deviation errors. Latest, mean-3, mean-8, and
    vintage standard deviation form the fixed 4x9 target-free contract.
    """

    raw = np.asarray(raw)
    base_stream = np.asarray(base_stream)
    origins = np.asarray(origins, dtype=np.int64)
    channels = np.asarray(channels, dtype=np.int64)
    if raw.ndim != 2 or base_stream.ndim != 3 or base_stream.shape[2] != 96:
        raise ValueError("DiffAmp matured inputs must be raw [T,C] and base [O,C,96]")
    if raw.shape[1] != base_stream.shape[1]:
        raise ValueError("DiffAmp matured channel count mismatch")
    if origins.ndim != 1 or origins.shape != channels.shape:
        raise ValueError("DiffAmp matured row axes must align")
    if int(geometry.patches_per_period) != 8:
        raise ValueError("DiffAmp matured state requires eight patches")
    if origins.size == 0:
        return np.empty((0, 36), dtype=np.float32)
    if int(channels.min()) < 0 or int(channels.max()) >= raw.shape[1]:
        raise ValueError("DiffAmp matured channel is out of range")

    summaries = np.empty((origins.size, 8, 9), dtype=np.float64)
    lead = np.arange(96, dtype=np.int64)[None]
    for age in range(8):
        source_origin = origins - 96 - age * int(geometry.patch_steps)
        source_index = source_origin - int(origin0)
        if int(source_index.min()) < 0 or int(source_index.max()) >= base_stream.shape[0]:
            raise ValueError("DiffAmp matured source forecast is unavailable")
        target_index = source_origin[:, None] + lead
        prediction = np.asarray(base_stream[source_index, channels], dtype=np.float64)
        target = np.asarray(raw[target_index, channels[:, None]], dtype=np.float64)
        prediction_d1 = np.diff(prediction, axis=1)
        target_d1 = np.diff(target, axis=1)
        summaries[:, age, 0] = target_d1.std(axis=1, ddof=1) - prediction_d1.std(
            axis=1, ddof=1
        )
        prediction_padded = np.concatenate(
            [np.zeros((origins.size, 1)), prediction_d1], axis=1
        ).reshape(origins.size, 8, 12)
        target_padded = np.concatenate(
            [np.zeros((origins.size, 1)), target_d1], axis=1
        ).reshape(origins.size, 8, 12)
        summaries[:, age, 1:] = target_padded.std(
            axis=2, ddof=0
        ) - prediction_padded.std(axis=2, ddof=0)
    result = np.concatenate(
        [
            summaries[:, 0],
            summaries[:, :3].mean(axis=1),
            summaries.mean(axis=1),
            summaries.std(axis=1),
        ],
        axis=1,
    ).astype(np.float32)
    if result.shape != (origins.size, 36) or not np.isfinite(result).all():
        raise ValueError("invalid DiffAmp matured residual features")
    return result


def amp_context_features(
    origins: np.ndarray,
    channels: np.ndarray,
    descriptors: np.ndarray,
    geometry: PeriodGeometry,
) -> np.ndarray:
    """Four normalized calendar features plus seven channel descriptors."""

    origins = np.asarray(origins, dtype=np.int64)
    channels = np.asarray(channels, dtype=np.int64)
    period = 2.0 * np.pi * (origins % geometry.period_steps) / geometry.period_steps
    week = 2.0 * np.pi * (origins % geometry.week_steps) / geometry.week_steps
    calendar = np.stack(
        [np.sin(period), np.cos(period), np.sin(week), np.cos(week)], axis=1
    ).astype(np.float32)
    result = np.concatenate(
        [calendar, np.asarray(descriptors, dtype=np.float32)[channels]], axis=1
    ).astype(np.float32)
    if result.shape[1] != 11:
        raise ValueError("Amp context width drift")
    return result


def period_level_features(
    raw: np.ndarray,
    semantic: np.ndarray,
    context: np.ndarray,
    feedback: np.ndarray,
    config: PeriodicAdapterConfig,
) -> np.ndarray:
    """Aggregate eight phase views into one fixed-width Level decision row.

    Inputs have shape ``[B,S,D]`` with ``S <= 8`` for one physical period.
    A partial final period is zero-padded and accompanied by an explicit mask,
    so the learned sign/utility architecture stays unchanged at every horizon.
    """

    raw = np.asarray(raw, dtype=np.float32)
    semantic = np.asarray(semantic, dtype=np.float32)
    context = np.asarray(context, dtype=np.float32)
    feedback = np.asarray(feedback, dtype=np.float32)
    if not (
        raw.ndim == semantic.ndim == context.ndim == feedback.ndim == 3
        and raw.shape[:2]
        == semantic.shape[:2]
        == context.shape[:2]
        == feedback.shape[:2]
    ):
        raise ValueError("period-Level feature views must align as [B,S,D]")
    batch, slots = raw.shape[:2]
    maximum = config.native.patches_per_period
    if not 0 < slots <= maximum:
        raise ValueError("invalid number of period-Level phase views")
    history_width = config.kernel.period_steps + 4
    patch_width = config.kernel.patch_steps + 6
    if raw.shape[2] != history_width + patch_width:
        raise ValueError("period-Level raw width drift")
    if semantic.shape[2] != 36 or context.shape[2] != 147 or feedback.shape[2] != 65:
        raise ValueError("period-Level auxiliary width drift")
    padded_patch = np.zeros((batch, maximum, patch_width), dtype=np.float32)
    padded_patch[:, :slots] = raw[:, :, history_width:]
    mask = np.zeros((batch, maximum), dtype=np.float32)
    mask[:, :slots] = 1.0
    result = np.concatenate(
        [
            raw[:, 0, :history_width],
            padded_patch.reshape(batch, -1),
            semantic.mean(axis=1),
            semantic.std(axis=1),
            context.mean(axis=1),
            feedback.mean(axis=1),
            feedback.std(axis=1),
            mask,
        ],
        axis=1,
    ).astype(np.float32)
    expected = history_width + maximum * patch_width + 2 * 36 + 147 + 2 * 65 + maximum
    if result.shape != (batch, expected) or not np.isfinite(result).all():
        raise ValueError("invalid period-Level aggregate features")
    return result


def period_level_ordered_sign_features(
    raw: np.ndarray,
    semantic: np.ndarray,
    context: np.ndarray,
    feedback: np.ndarray,
    config: PeriodicAdapterConfig,
) -> np.ndarray:
    """Preserve the fixed eight-view order for universal Level-sign fitting.

    This is the ordered counterpart of :func:`period_level_features`.  The
    canonical history, patch views, semantic moments, and context semantics are
    unchanged; only the 65D matured-feedback views are padded and retained in
    physical phase order instead of being reduced to featurewise mean/std.
    Both supported native clocks map parameter-free to exactly eight views.
    """

    raw = np.asarray(raw, dtype=np.float32)
    semantic = np.asarray(semantic, dtype=np.float32)
    context = np.asarray(context, dtype=np.float32)
    feedback = np.asarray(feedback, dtype=np.float32)
    if not (
        raw.ndim == semantic.ndim == context.ndim == feedback.ndim == 3
        and raw.shape[:2]
        == semantic.shape[:2]
        == context.shape[:2]
        == feedback.shape[:2]
    ):
        raise ValueError("ordered period-Level feature views must align as [B,S,D]")
    batch, slots = raw.shape[:2]
    maximum = config.native.patches_per_period
    if maximum != 8 or not 0 < slots <= maximum:
        raise ValueError("ordered period-Level sign requires at most eight views")
    history_width = config.kernel.period_steps + 4
    patch_width = config.kernel.patch_steps + 6
    if raw.shape[2] != history_width + patch_width:
        raise ValueError("ordered period-Level raw width drift")
    if semantic.shape[2] != 36 or context.shape[2] != 147 or feedback.shape[2] != 65:
        raise ValueError("ordered period-Level auxiliary width drift")

    padded_patch = np.zeros((batch, maximum, patch_width), dtype=np.float32)
    padded_patch[:, :slots] = raw[:, :, history_width:]
    padded_feedback = np.zeros((batch, maximum, 65), dtype=np.float32)
    padded_feedback[:, :slots] = feedback
    mask = np.zeros((batch, maximum), dtype=np.float32)
    mask[:, :slots] = 1.0
    result = np.concatenate(
        [
            raw[:, 0, :history_width],
            padded_patch.reshape(batch, -1),
            semantic.mean(axis=1),
            semantic.std(axis=1),
            context.mean(axis=1),
            padded_feedback.reshape(batch, -1),
            mask,
        ],
        axis=1,
    ).astype(np.float32)
    expected = history_width + maximum * patch_width + 2 * 36 + 147 + maximum * 65 + maximum
    if expected != 991 or result.shape != (batch, expected) or not np.isfinite(result).all():
        raise ValueError("invalid ordered period-Level sign features")
    return result


def period_level_ordered_core_sign_features(
    raw: np.ndarray,
    semantic: np.ndarray,
    context: np.ndarray,
    feedback: np.ndarray,
    config: PeriodicAdapterConfig,
) -> np.ndarray:
    """Add only the non-derived ordered maturity core to the stable 601D row.

    The retained 10D value per canonical phase is fixed by the existing 65D
    feedback contract: four same-phase ages ``{0,1,2,6}`` (columns 8:12) and
    six causal seven-age statistics (columns 20:26).  Repeated full profiles,
    channel-descriptor products, and slot one-hot interactions are deliberately
    not duplicated.  The result is always 601 + 8*10 = 681 dimensions.
    """

    raw = np.asarray(raw, dtype=np.float32)
    semantic = np.asarray(semantic, dtype=np.float32)
    context = np.asarray(context, dtype=np.float32)
    feedback = np.asarray(feedback, dtype=np.float32)
    if not (
        raw.ndim == semantic.ndim == context.ndim == feedback.ndim == 3
        and raw.shape[:2]
        == semantic.shape[:2]
        == context.shape[:2]
        == feedback.shape[:2]
    ):
        raise ValueError("core period-Level feature views must align as [B,S,D]")
    batch, slots = feedback.shape[:2]
    maximum = config.native.patches_per_period
    if maximum != 8 or not 0 < slots <= maximum or feedback.shape[2] != 65:
        raise ValueError("core period-Level sign requires up to eight 65D views")
    stable = period_level_features(raw, semantic, context, feedback, config)
    core = np.concatenate([feedback[:, :, 8:12], feedback[:, :, 20:26]], axis=2)
    padded_core = np.zeros((batch, maximum, 10), dtype=np.float32)
    padded_core[:, :slots] = core
    result = np.concatenate(
        [stable[:, :-maximum], padded_core.reshape(batch, -1), stable[:, -maximum:]],
        axis=1,
    ).astype(np.float32)
    if result.shape != (batch, 681) or not np.isfinite(result).all():
        raise ValueError("invalid core ordered period-Level sign features")
    return result


def _linear_coordinate(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] <= 1:
        raise ValueError("linear coordinate expects [B,L] with L > 1")
    basis = np.linspace(-1.0, 1.0, values.shape[1], dtype=np.float64)
    basis -= basis.mean()
    centered = values - values.mean(axis=1, keepdims=True)
    return np.sum(centered * basis[None], axis=1) / np.sum(np.square(basis))


def period_trend_features(
    history: np.ndarray,
    base_period: np.ndarray,
    origins: np.ndarray,
    channels: np.ndarray,
    period_instances: np.ndarray,
    descriptors: np.ndarray,
    config: PeriodicAdapterConfig,
    *,
    length_fraction: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the fixed 106D anchored-Trend contract for one physical period.

    The former seven-channel one-hot lane is replaced by the same universal 7D
    channel descriptors used by every periodic adapter.  Consequently ETTm1
    P96 and Electricity P24 expose identical feature groups and parameter
    counts without a channel-count-dependent layer.
    """

    history = np.asarray(history, dtype=np.float32)
    base_period = np.asarray(base_period, dtype=np.float32)
    origins = np.asarray(origins, dtype=np.int64)
    channels = np.asarray(channels, dtype=np.int64)
    period_instances = np.asarray(period_instances, dtype=np.int64)
    if not (
        history.ndim == base_period.ndim == 2
        and history.shape[0]
        == base_period.shape[0]
        == origins.size
        == channels.size
        == period_instances.size
    ):
        raise ValueError("period-Trend rows do not align")
    kernel_steps = config.kernel.period_steps
    path = resample_numpy_last(history, kernel_steps).astype(np.float64)
    base = resample_numpy_last(base_period, kernel_steps).astype(np.float64)
    last = path[:, -1]
    history_std = path.std(axis=1).clip(1.0e-6)
    normalized_history = (path - last[:, None]) / history_std[:, None]
    normalized_base = (base - last[:, None]) / history_std[:, None]
    input_slopes = np.stack(
        [
            _linear_coordinate(normalized_history[:, -width:])
            for width in (12, 24, 48, 96)
        ],
        axis=1,
    )
    split2 = np.array_split(np.arange(kernel_steps), 2)
    split4 = np.array_split(np.arange(kernel_steps), 4)
    base_parts = [_linear_coordinate(normalized_base)]
    base_parts.extend(_linear_coordinate(normalized_base[:, part]) for part in split2)
    base_parts.extend(_linear_coordinate(normalized_base[:, part]) for part in split4)
    base_parts_array = np.stack(base_parts, axis=1)
    endpoint = 0.5 * (normalized_base[:, -1] - normalized_base[:, 0])
    semantic_local = np.concatenate(
        [input_slopes, base_parts_array, endpoint[:, None]], axis=1
    )
    semantic_raw = semantic_local * history_std[:, None]
    contrast = semantic_raw[:, 4:5] - semantic_raw[:, 3:4]
    semantic = np.concatenate(
        [semantic_local, semantic_raw, contrast, np.log(history_std)[:, None]],
        axis=1,
    ).astype(np.float32)
    if semantic.shape[1] != 26:
        raise ValueError("period-Trend semantic width drift")

    descriptor = np.asarray(descriptors, dtype=np.float32)[channels]
    if descriptor.shape[1] != 7:
        raise ValueError("period-Trend descriptor width must be seven")
    if length_fraction is None:
        length_fraction = np.ones(origins.size, dtype=np.float64)
    else:
        length_fraction = np.asarray(length_fraction, dtype=np.float64)
    center = (
        origins.astype(np.float64)
        + period_instances * config.native.period_steps
        + 0.5 * (length_fraction * config.native.period_steps - 1.0)
    )
    angle = 2.0 * np.pi * center / config.native.period_steps
    fourier = np.stack(
        [
            function(harmonic * angle)
            for harmonic in range(1, 5)
            for function in (np.sin, np.cos)
        ],
        axis=1,
    ).astype(np.float32)
    interaction = (descriptor[:, :, None] * fourier[:, None]).reshape(
        origins.size, -1
    )
    calendar = np.concatenate([descriptor, fourier, interaction], axis=1).astype(
        np.float32
    )
    if calendar.shape[1] != 71:
        raise ValueError("period-Trend calendar width drift")
    absolute = np.stack([last, path.mean(axis=1), path[:, 0]], axis=1)
    lead_periods = period_instances.astype(np.float64) + 0.5 * length_fraction
    weekly = 2.0 * np.pi * lead_periods / config.native.periods_per_week
    physical = np.stack(
        [
            lead_periods,
            np.log1p(lead_periods),
            1.0 / (1.0 + lead_periods),
            np.sin(weekly),
            np.cos(weekly),
            length_fraction,
        ],
        axis=1,
    )
    state = np.concatenate([absolute, physical], axis=1).astype(np.float32)
    if state.shape[1] != 9:
        raise ValueError("period-Trend state width drift")
    features = np.concatenate([semantic, calendar, state], axis=1).astype(np.float32)
    if features.shape != (origins.size, 106) or not np.isfinite(features).all():
        raise ValueError("invalid universal period-Trend features")
    anchor_coordinate = (
        _linear_coordinate(path) - _linear_coordinate(base)
    ).astype(np.float32)
    return features, anchor_coordinate, history_std.astype(np.float32)


def _affine_free_canonical(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    shape = values.shape
    flat = values.reshape(-1, shape[-1])
    centered = flat - flat.mean(axis=1, keepdims=True)
    basis = np.linspace(-1.0, 1.0, shape[-1], dtype=np.float64)
    basis -= basis.mean()
    coefficient = np.sum(centered * basis[None], axis=1, keepdims=True)
    coefficient /= np.sum(np.square(basis))
    return (centered - coefficient * basis[None]).reshape(shape).astype(np.float32)


def period_shape_features(
    history: np.ndarray,
    base_period: np.ndarray,
    period_memory: np.ndarray,
    origins: np.ndarray,
    channels: np.ndarray,
    period_instances: np.ndarray,
    descriptors: np.ndarray,
    config: PeriodicAdapterConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed 591D seasonal-Shape features and four causal anchors.

    ``period_memory`` is ordered oldest-to-recent and contains exactly 28 fully
    matured physical-period residual curves.  All native clocks are converted
    parameter-free to the common canonical P96 representation.
    """

    history = np.asarray(history, dtype=np.float32)
    base_period = np.asarray(base_period, dtype=np.float32)
    period_memory = np.asarray(period_memory, dtype=np.float32)
    origins = np.asarray(origins, dtype=np.int64)
    channels = np.asarray(channels, dtype=np.int64)
    period_instances = np.asarray(period_instances, dtype=np.int64)
    rows = origins.size
    if not (
        history.ndim == base_period.ndim == 2
        and period_memory.ndim == 3
        and history.shape[0]
        == base_period.shape[0]
        == period_memory.shape[0]
        == rows
        == channels.size
        == period_instances.size
        and period_memory.shape[1] == config.kernel.carrier_periods
    ):
        raise ValueError("period-Shape rows or memory do not align")
    steps = config.kernel.period_steps
    history_kernel = resample_numpy_last(history, steps).astype(np.float64)
    base_kernel = resample_numpy_last(base_period, steps).astype(np.float64)
    memory_kernel = resample_numpy_last(period_memory, steps).astype(np.float64)
    last = history_kernel[:, -1:]
    history_std = history_kernel.std(axis=1, keepdims=True).clip(1.0e-6)
    recent7 = memory_kernel[:, -7:]
    blocks = [
        (history_kernel - last) / history_std,
        (base_kernel - last) / history_std,
        recent7.mean(axis=1) / history_std,
        recent7.std(axis=1) / history_std,
        memory_kernel.mean(axis=1) / history_std,
        memory_kernel.std(axis=1) / history_std,
    ]
    descriptor = np.asarray(descriptors, dtype=np.float32)[channels]
    if descriptor.shape[1] != 7:
        raise ValueError("period-Shape descriptor width must be seven")
    center = (
        origins.astype(np.float64)
        + period_instances * config.native.period_steps
        + 0.5 * (config.native.period_steps - 1.0)
    )
    angle = 2.0 * np.pi * center / config.native.period_steps
    fourier = np.stack(
        [
            function(harmonic * angle)
            for harmonic in range(1, 5)
            for function in (np.sin, np.cos)
        ],
        axis=1,
    ).astype(np.float32)
    features = np.concatenate(
        [*(block.astype(np.float32) for block in blocks), descriptor, fourier],
        axis=1,
    ).astype(np.float32)
    if features.shape != (rows, 591) or not np.isfinite(features).all():
        raise ValueError("invalid universal period-Shape features")
    anchors = np.stack(
        [
            memory_kernel[:, -1],
            memory_kernel[:, -3:].mean(axis=1),
            recent7.mean(axis=1),
            memory_kernel.mean(axis=1),
        ],
        axis=1,
    )
    anchors = _affine_free_canonical(anchors)
    if anchors.shape != (rows, 4, steps):
        raise ValueError("period-Shape anchor width drift")
    return features, anchors


def period_shape_sequence_inputs(
    history: np.ndarray,
    base_period: np.ndarray,
    period_memory: np.ndarray,
    amp_reference: np.ndarray,
    origins: np.ndarray,
    channels: np.ndarray,
    period_instances: np.ndarray,
    descriptors: np.ndarray,
    config: PeriodicAdapterConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the fixed full-memory inputs for universal Shape sequence models.

    Every one of the 28 fully matured physical-period residual curves is first
    projected into the current physical Level/Trend/Amp complement and only
    then mapped to canonical P96.  This preserves the universal decomposition
    instead of mixing source-coordinate coefficients across forecast origins.
    """

    history = np.asarray(history, dtype=np.float32)
    base_period = np.asarray(base_period, dtype=np.float32)
    period_memory = np.asarray(period_memory, dtype=np.float32)
    amp_reference = np.asarray(amp_reference, dtype=np.float32)
    origins = np.asarray(origins, dtype=np.int64)
    channels = np.asarray(channels, dtype=np.int64)
    period_instances = np.asarray(period_instances, dtype=np.int64)
    rows = origins.size
    carrier_periods = config.kernel.carrier_periods
    if not (
        history.ndim == base_period.ndim == amp_reference.ndim == 2
        and period_memory.ndim == 3
        and history.shape[0]
        == base_period.shape[0]
        == amp_reference.shape[0]
        == period_memory.shape[0]
        == rows
        == channels.size
        == period_instances.size
        and period_memory.shape[1] == carrier_periods
        and history.shape[1]
        == base_period.shape[1]
        == amp_reference.shape[1]
        == period_memory.shape[2]
        == config.native.period_steps
    ):
        raise ValueError("period-Shape sequence rows or clocks do not align")

    memory_envelope = _affine_free_canonical(period_memory)
    amp_envelope = _affine_free_canonical(amp_reference)
    denominator = np.square(amp_envelope).sum(axis=1, keepdims=True).clip(1.0e-12)
    amp_coefficient = np.sum(
        memory_envelope * amp_envelope[:, None], axis=2, keepdims=True
    ) / denominator[:, None]
    memory_shape_native = memory_envelope - amp_coefficient * amp_envelope[:, None]
    anchors = resample_numpy_last(
        memory_shape_native, config.kernel.period_steps
    ).astype(np.float32)

    history_kernel = resample_numpy_last(history, config.kernel.period_steps).astype(
        np.float64
    )
    base_kernel = resample_numpy_last(base_period, config.kernel.period_steps).astype(
        np.float64
    )
    amp_kernel = resample_numpy_last(amp_envelope, config.kernel.period_steps).astype(
        np.float64
    )
    last = history_kernel[:, -1:]
    history_std = history_kernel.std(axis=1, keepdims=True).clip(1.0e-6)
    normalized_memory = anchors.astype(np.float64) / history_std[:, None]

    descriptor = np.asarray(descriptors, dtype=np.float32)[channels]
    if descriptor.shape != (rows, 7):
        raise ValueError("period-Shape sequence descriptor width must be seven")
    center = (
        origins.astype(np.float64)
        + period_instances * config.native.period_steps
        + 0.5 * (config.native.period_steps - 1.0)
    )
    angle = 2.0 * np.pi * center / config.native.period_steps
    fourier = np.stack(
        [
            function(harmonic * angle)
            for harmonic in range(1, 5)
            for function in (np.sin, np.cos)
        ],
        axis=1,
    ).astype(np.float32)
    query = np.concatenate(
        [
            (history_kernel - last) / history_std,
            (base_kernel - last) / history_std,
            amp_kernel / history_std,
            descriptor,
            fourier,
        ],
        axis=1,
    ).astype(np.float32)
    if query.shape != (rows, 303):
        raise ValueError("period-Shape sequence query width drift")
    if normalized_memory.shape != (rows, carrier_periods, 96):
        raise ValueError("period-Shape sequence memory width drift")
    if anchors.shape != (rows, carrier_periods, 96):
        raise ValueError("period-Shape sequence anchor width drift")
    if not (
        np.isfinite(query).all()
        and np.isfinite(normalized_memory).all()
        and np.isfinite(anchors).all()
    ):
        raise ValueError("invalid universal period-Shape sequence inputs")
    return query, normalized_memory.astype(np.float32), anchors


__all__ = [
    "amp_context_features",
    "channel_descriptors",
    "level_context_features",
    "level_feedback_features",
    "level_raw_features",
    "level_semantic_features",
    "materialize_aligned_carriers",
    "period_level_features",
    "period_shape_features",
    "period_shape_sequence_inputs",
    "period_trend_features",
    "resample_numpy_last",
]
