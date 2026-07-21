"""Horizon-parameterized exact Level/Amp p12 prior and H96 veto Gate.

Only complete physical H96 blocks are routed.  Any incomplete horizon tail is
an exact no-op.  Level, Amp, and Trend tensors are treated as frozen expert
outputs; this module owns only the small routing model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn


BLOCK = 96
PATCH = 12
PATCHES = BLOCK // PATCH
CHANNELS = 7
CONTEXT = 96
BASE_INPUT_LANES = 17
PRIOR_INPUT_LANES = 2
STATE_INPUT_LANES = 8
INPUT_LANES = BASE_INPUT_LANES + PRIOR_INPUT_LANES + STATE_INPUT_LANES
CANONICAL_ACTIONS = np.asarray((0, 1, 4, 5), dtype=np.int64)
MASK_NAMES = ("keep", "remove_level", "remove_amp", "skip")


def require(condition: object, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def active_steps(horizon: int) -> int:
    require(horizon >= BLOCK, "horizon must contain a complete H96 block")
    return (int(horizon) // BLOCK) * BLOCK


def complete_blocks(horizon: int) -> int:
    return active_steps(horizon) // BLOCK


def _latest_expert_utility(
    source: dict[str, np.ndarray], expert: str, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origins = source["origin"]
    lookup = {int(origin): row for row, origin in enumerate(origins)}
    mse = np.zeros_like(source[expert], dtype=np.float32)
    mae = np.zeros_like(mse)
    available = np.zeros_like(mse)
    for patch_index in range(active_steps(horizon) // PATCH):
        delay = (patch_index + 1) * PATCH
        previous = np.asarray(
            [lookup.get(int(origin) - delay, -1) for origin in origins],
            dtype=np.int64,
        )
        valid = previous >= 0
        if not np.any(valid):
            continue
        current_row = np.flatnonzero(valid)
        previous_row = previous[valid]
        require(
            np.all(origins[previous_row] + delay == origins[current_row]),
            f"latest {expert} utility endpoint drift",
        )
        current = slice(patch_index * PATCH, (patch_index + 1) * PATCH)
        error = (
            source["target"][previous_row, :, current]
            - source["base"][previous_row, :, current]
        ).astype(np.float64)
        correction = source[expert][previous_row, :, current].astype(np.float64)
        remaining = error - correction
        current_mse = 1.0 - np.sum(np.square(remaining), axis=2) / np.maximum(
            np.sum(np.square(error), axis=2), 1.0e-8
        )
        current_mae = 1.0 - np.sum(np.abs(remaining), axis=2) / np.maximum(
            np.sum(np.abs(error), axis=2), 1.0e-8
        )
        mse[current_row, :, current] = np.clip(current_mse, -4.0, 4.0)[
            :, :, None
        ]
        mae[current_row, :, current] = np.clip(current_mae, -4.0, 4.0)[
            :, :, None
        ]
        available[current_row, :, current] = 1.0
    return mse, mae, available


def shape_flat_source(
    flat: dict[str, np.ndarray], horizon: int
) -> dict[str, np.ndarray]:
    """Validate and shape a dense origin/channel frozen-expert stream."""

    required = ("origin", "channel", "base", "target", "level", "trend", "amp")
    require(all(name in flat for name in required), "missing frozen-expert tensor")
    order = np.lexsort((flat["channel"], flat["origin"]))
    ordered = {name: np.asarray(flat[name])[order].copy() for name in required}
    origins = np.unique(ordered["origin"])
    require(origins.size > 0 and np.all(np.diff(origins) == 1), "origin gap")
    require(
        ordered["origin"].size == origins.size * CHANNELS,
        "origin/channel width drift",
    )
    require(
        np.array_equal(ordered["origin"], np.repeat(origins, CHANNELS)),
        "origin ordering drift",
    )
    require(
        np.array_equal(
            ordered["channel"], np.tile(np.arange(CHANNELS), origins.size)
        ),
        "channel ordering drift",
    )
    source: dict[str, np.ndarray] = {
        "origin": origins.astype(np.int64),
        "channel": ordered["channel"].reshape(origins.size, CHANNELS),
    }
    for name in ("base", "target", "level", "trend", "amp"):
        require(
            ordered[name].shape == (origins.size * CHANNELS, horizon),
            f"{name} horizon drift",
        )
        source[name] = ordered[name].reshape(origins.size, CHANNELS, horizon)

    lookup = {int(origin): row for row, origin in enumerate(origins)}
    previous = np.asarray(
        [lookup.get(int(origin) - CONTEXT, -1) for origin in origins],
        dtype=np.int64,
    )
    context_available = previous >= 0
    safe = np.maximum(previous, 0)
    source["context_available"] = context_available
    source["context_observed"] = source["target"][safe, :, :CONTEXT].copy()
    source["context_residual"] = (
        source["target"][safe, :, :CONTEXT]
        - source["base"][safe, :, :CONTEXT]
    )
    source["context_level"] = source["level"][safe, :, :CONTEXT].copy()
    source["context_amp"] = source["amp"][safe, :, :CONTEXT].copy()
    for name in (
        "context_observed",
        "context_residual",
        "context_level",
        "context_amp",
    ):
        source[name][~context_available] = 0.0
    require(
        np.all(
            origins[safe[context_available]] + CONTEXT
            == origins[context_available]
        ),
        "matured H96 context endpoint drift",
    )

    for expert, prefix in (("amp", "latest_patch"), ("level", "latest_level")):
        mse, mae, available = _latest_expert_utility(source, expert, horizon)
        source[f"{prefix}_mse_utility"] = mse
        source[f"{prefix}_mae_utility"] = mae
        source[f"{prefix}_available"] = available
    return source


def origin_mask(
    source: dict[str, np.ndarray],
    *,
    start: int | None = None,
    stop: int | None = None,
    stride: int | None = None,
) -> np.ndarray:
    origins = source["origin"]
    first_matured = int(origins.min()) + CONTEXT
    use = origins >= first_matured
    if start is not None:
        use &= origins >= int(start)
    if stop is not None:
        use &= origins <= int(stop)
    if stride is not None:
        require(int(stride) > 0, "nonpositive origin stride")
        use &= (origins - first_matured) % int(stride) == 0
    require(np.any(use), "empty source mask")
    return use


def flat_block(
    source: dict[str, np.ndarray], use: np.ndarray, horizon: int
) -> dict[str, np.ndarray]:
    count = int(np.sum(use))
    require(count > 0, "empty flat block")
    return {
        "origin": np.repeat(source["origin"][use], CHANNELS),
        "channel": np.tile(np.arange(CHANNELS), count),
        **{
            name: source[name][use].reshape(-1, horizon)
            for name in ("base", "target", "level", "trend", "amp")
        },
    }


def patchize(block: dict[str, np.ndarray], horizon: int) -> dict[str, np.ndarray]:
    rows = block["origin"].size
    blocks = complete_blocks(horizon)
    patches = blocks * PATCHES
    physical_block = np.tile(np.repeat(np.arange(blocks), PATCHES), rows)
    patch_in_block = np.tile(np.arange(PATCHES), rows * blocks)
    lead_start = physical_block * BLOCK + patch_in_block * PATCH
    result = {
        "origin": np.repeat(block["origin"], patches).astype(np.int64) + lead_start,
        "forecast_origin": np.repeat(block["origin"], patches).astype(np.int64),
        "channel": np.repeat(block["channel"], patches).astype(np.int64),
        "physical_block": physical_block.astype(np.int64),
        "patch_in_block": patch_in_block.astype(np.int64),
        "lead_start": lead_start.astype(np.int64),
    }
    stop = active_steps(horizon)
    for name in ("base", "target", "level", "trend", "amp"):
        result[name] = block[name][:, :stop].reshape(rows * patches, PATCH)
    return result


def cell_coordinates(
    patch: dict[str, np.ndarray], horizon: int
) -> tuple[np.ndarray, ...]:
    channel = patch["channel"].astype(np.int64)
    block = patch["physical_block"].astype(np.int64)
    coordinate = patch["patch_in_block"].astype(np.int64)
    target_phase = (patch["origin"].astype(np.int64) % 96) // PATCH
    require(np.all((0 <= channel) & (channel < CHANNELS)), "channel key drift")
    require(
        np.all((0 <= block) & (block < complete_blocks(horizon))),
        "physical block key drift",
    )
    require(np.all((0 <= coordinate) & (coordinate < PATCHES)), "patch key drift")
    require(np.all((0 <= target_phase) & (target_phase < PATCHES)), "phase key drift")
    return channel, block, coordinate, target_phase


def _action_correction(
    level: np.ndarray, amp: np.ndarray, action: int
) -> np.ndarray:
    correction = np.zeros_like(level, dtype=np.float64)
    if action in (1, 5):
        correction += level.astype(np.float64)
    if action in (4, 5):
        correction += amp.astype(np.float64)
    return correction


def domain_utility(block: dict[str, np.ndarray], horizon: int) -> dict[str, np.ndarray]:
    patch = patchize(block, horizon)
    coordinates = cell_coordinates(patch, horizon)
    residual = patch["target"].astype(np.float64) - patch["base"].astype(np.float64)
    base_mse = np.square(residual).mean(axis=1)
    base_mae = np.abs(residual).mean(axis=1)
    mse = np.zeros((residual.shape[0], CANONICAL_ACTIONS.size), dtype=np.float64)
    mae = np.zeros_like(mse)
    for column, action in enumerate(CANONICAL_ACTIONS[1:], 1):
        correction = _action_correction(
            patch["level"], patch["amp"], int(action)
        )
        remaining = residual - correction
        mse[:, column] = base_mse - np.square(remaining).mean(axis=1)
        mae[:, column] = base_mae - np.abs(remaining).mean(axis=1)
    shape = (CHANNELS, complete_blocks(horizon), PATCHES, PATCHES)
    support = np.zeros(shape, dtype=np.int64)
    mse_sum = np.zeros(shape + (CANONICAL_ACTIONS.size,), dtype=np.float64)
    mae_sum = np.zeros_like(mse_sum)
    base_mse_sum = np.zeros(shape, dtype=np.float64)
    base_mae_sum = np.zeros(shape, dtype=np.float64)
    np.add.at(support, coordinates, 1)
    np.add.at(mse_sum, coordinates, mse)
    np.add.at(mae_sum, coordinates, mae)
    np.add.at(base_mse_sum, coordinates, base_mse)
    np.add.at(base_mae_sum, coordinates, base_mae)
    require(np.all(support > 0), "routing table left unseen cells")
    return {
        "normalized_mse": mse_sum / np.maximum(base_mse_sum[..., None], 1.0e-12),
        "normalized_mae": mae_sum / np.maximum(base_mae_sum[..., None], 1.0e-12),
        "support": support,
    }


def fit_prior_table(domains: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    require(bool(domains), "empty prior domains")
    mse = np.stack([domain["normalized_mse"] for domain in domains])
    mae = np.stack([domain["normalized_mae"] for domain in domains])
    mean_mse = mse.mean(axis=0)
    mean_mae = mae.mean(axis=0)
    joint_score = np.minimum(mean_mse, mean_mae)
    winner = np.argmax(joint_score, axis=-1)
    best = np.take_along_axis(joint_score, winner[..., None], axis=-1)[..., 0]
    action = CANONICAL_ACTIONS[winner]
    action[best <= 0.0] = 0
    require(np.isin(action, CANONICAL_ACTIONS).all(), "invalid prior action")
    return {
        "action": action.astype(np.int64),
        "joint_score": joint_score,
        "mean_normalized_mse": mean_mse,
        "mean_normalized_mae": mean_mae,
        "best_score": best,
    }


def prior_actions(
    source: dict[str, np.ndarray],
    use: np.ndarray,
    table: dict[str, np.ndarray],
    horizon: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    block = flat_block(source, use, horizon)
    patch = patchize(block, horizon)
    action = table["action"][cell_coordinates(patch, horizon)]
    count = int(np.sum(use))
    shaped = action.reshape(
        count, CHANNELS, complete_blocks(horizon), PATCHES
    ).reshape(-1, PATCHES)
    require(np.isin(shaped, CANONICAL_ACTIONS).all(), "prior action drift")
    return shaped.astype(np.int64), block


def fit_scaler(
    pieces: list[tuple[dict[str, np.ndarray], np.ndarray]], horizon: int
) -> dict[str, np.ndarray]:
    stop = active_steps(horizon)
    base = np.concatenate([source["base"][use, :, :stop] for source, use in pieces]).astype(np.float64)
    level = np.concatenate([source["level"][use, :, :stop] for source, use in pieces]).astype(np.float64)
    amp = np.concatenate([source["amp"][use, :, :stop] for source, use in pieces]).astype(np.float64)
    residual = np.concatenate([source["context_residual"][use] for source, use in pieces]).astype(np.float64)
    available = np.concatenate([source["context_available"][use] for source, use in pieces])
    require(np.any(available), "no matured context for scaler")
    return {
        "base_mean": base.mean(axis=(0, 2)),
        "base_std": np.maximum(base.std(axis=(0, 2)), 1.0e-6),
        "level_std": np.maximum(level.std(axis=(0, 2)), 1.0e-6),
        "amp_std": np.maximum(amp.std(axis=(0, 2)), 1.0e-6),
        "residual_std": np.maximum(residual[available].std(axis=(0, 2)), 1.0e-6),
    }


def make_base_samples(
    source: dict[str, np.ndarray],
    use: np.ndarray,
    scaler: dict[str, np.ndarray],
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    count = int(np.sum(use))
    require(count > 0, "empty routing samples")
    blocks = complete_blocks(horizon)
    channel = np.broadcast_to(
        np.arange(CHANNELS)[None, :, None], (count, CHANNELS, blocks)
    ).reshape(-1)
    lead_block = np.broadcast_to(
        np.arange(blocks)[None, None, :], (count, CHANNELS, blocks)
    ).reshape(-1)

    def current(name: str) -> np.ndarray:
        return (
            source[name][use, :, : active_steps(horizon)]
            .reshape(count, CHANNELS, blocks, BLOCK)
            .reshape(-1, BLOCK)
            .astype(np.float32)
        )

    def context(name: str) -> np.ndarray:
        value = source[name][use].astype(np.float32)
        return np.repeat(value[:, :, None, :], blocks, axis=2).reshape(-1, BLOCK)

    base = current("base")
    target = current("target")
    level = current("level")
    amp = current("amp")
    mean = scaler["base_mean"][channel, None]
    base_std = scaler["base_std"][channel, None]
    level_std = scaler["level_std"][channel, None]
    amp_std = scaler["amp_std"][channel, None]
    residual_std = scaler["residual_std"][channel, None]
    available = np.broadcast_to(
        source["context_available"][use, None, None], (count, CHANNELS, blocks)
    ).reshape(-1)
    phase = (np.arange(BLOCK, dtype=np.float32) // PATCH) / float(PATCHES)
    x = np.stack(
        (
            (base - mean) / base_std,
            level / level_std,
            amp / amp_std,
            (context("context_observed") - mean) / base_std,
            context("context_residual") / residual_std,
            context("context_level") / level_std,
            context("context_amp") / amp_std,
            np.broadcast_to(available[:, None], base.shape),
            np.broadcast_to(np.sin(2.0 * np.pi * phase)[None], base.shape),
            np.broadcast_to(np.cos(2.0 * np.pi * phase)[None], base.shape),
            np.broadcast_to(np.log1p(lead_block)[:, None], base.shape),
            current("latest_patch_mse_utility"),
            current("latest_patch_mae_utility"),
            current("latest_patch_available"),
            current("latest_level_mse_utility"),
            current("latest_level_mae_utility"),
            current("latest_level_available"),
        ),
        axis=1,
    )
    x = np.clip(x, -8.0, 8.0).astype(np.float32)
    require(x.shape == (count * CHANNELS * blocks, BASE_INPUT_LANES, BLOCK), "base feature drift")
    require(np.isfinite(x).all(), "nonfinite routing features")
    metadata = {
        "origin": np.repeat(source["origin"][use], CHANNELS * blocks),
        "channel": channel.astype(np.int64),
        "lead_block": lead_block.astype(np.int64),
    }
    return x, (target - base).astype(np.float32), level, amp, metadata


def matured_prior_utility(
    source: dict[str, np.ndarray],
    use: np.ndarray,
    table: dict[str, np.ndarray],
    horizon: int,
    *,
    endpoint_lag: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require(endpoint_lag >= 0, "negative matured endpoint lag")
    origins = source["origin"]
    lookup = {int(origin): row for row, origin in enumerate(origins)}
    all_prior, _ = prior_actions(
        source, np.ones(origins.shape, dtype=bool), table, horizon
    )
    blocks = complete_blocks(horizon)
    action = all_prior.reshape(origins.size, CHANNELS, blocks, PATCHES)
    query_rows = np.flatnonzero(use)
    mse = np.zeros((query_rows.size, CHANNELS, blocks), dtype=np.float32)
    mae = np.zeros_like(mse)
    available = np.zeros_like(mse, dtype=bool)
    for physical_block in range(blocks):
        block_delay = (physical_block + 1) * BLOCK
        delay = endpoint_lag + block_delay
        previous = np.asarray(
            [lookup.get(int(origins[row]) - delay, -1) for row in query_rows],
            dtype=np.int64,
        )
        valid = previous >= 0
        if not np.any(valid):
            continue
        previous_row = previous[valid]
        current = slice(physical_block * BLOCK, (physical_block + 1) * BLOCK)
        error = (
            source["target"][previous_row, :, current]
            - source["base"][previous_row, :, current]
        ).astype(np.float64)
        prior = action[previous_row, :, physical_block]
        level_mask = np.repeat(np.isin(prior, (1, 5)), PATCH, axis=2)
        amp_mask = np.repeat(np.isin(prior, (4, 5)), PATCH, axis=2)
        correction = (
            level_mask * source["level"][previous_row, :, current]
            + amp_mask * source["amp"][previous_row, :, current]
        ).astype(np.float64)
        remaining = error - correction
        mse[valid, :, physical_block] = (
            1.0
            - np.sum(np.square(remaining), axis=2)
            / np.maximum(np.sum(np.square(error), axis=2), 1.0e-8)
        ).astype(np.float32)
        mae[valid, :, physical_block] = (
            1.0
            - np.sum(np.abs(remaining), axis=2)
            / np.maximum(np.sum(np.abs(error), axis=2), 1.0e-8)
        ).astype(np.float32)
        available[valid, :, physical_block] = True
        require(
            np.all(
                origins[previous_row] + block_delay
                == origins[query_rows[valid]] - endpoint_lag
            ),
            "matured prior endpoint drift",
        )
    return mse.reshape(-1), mae.reshape(-1), available.reshape(-1)


def matured_amp_marginal_utility(
    source: dict[str, np.ndarray],
    use: np.ndarray,
    table: dict[str, np.ndarray],
    horizon: int,
    *,
    endpoint_lag: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal utility of adding Amp to the exact prior, relative to no-Amp."""

    require(endpoint_lag >= 0, "negative matured Amp endpoint lag")
    origins = source["origin"]
    lookup = {int(origin): row for row, origin in enumerate(origins)}
    all_prior, _ = prior_actions(
        source, np.ones(origins.shape, dtype=bool), table, horizon
    )
    blocks = complete_blocks(horizon)
    action = all_prior.reshape(origins.size, CHANNELS, blocks, PATCHES)
    query_rows = np.flatnonzero(use)
    mse = np.zeros((query_rows.size, CHANNELS, blocks), dtype=np.float32)
    mae = np.zeros_like(mse)
    available = np.zeros_like(mse, dtype=bool)
    for physical_block in range(blocks):
        block_delay = (physical_block + 1) * BLOCK
        delay = endpoint_lag + block_delay
        previous = np.asarray(
            [lookup.get(int(origins[row]) - delay, -1) for row in query_rows],
            dtype=np.int64,
        )
        valid = previous >= 0
        if not np.any(valid):
            continue
        previous_row = previous[valid]
        current = slice(physical_block * BLOCK, (physical_block + 1) * BLOCK)
        error = (
            source["target"][previous_row, :, current]
            - source["base"][previous_row, :, current]
        ).astype(np.float64)
        prior = action[previous_row, :, physical_block]
        level_mask = np.repeat(np.isin(prior, (1, 5)), PATCH, axis=2)
        amp_mask = np.repeat(np.isin(prior, (4, 5)), PATCH, axis=2)
        no_amp = (
            level_mask * source["level"][previous_row, :, current]
        ).astype(np.float64)
        full = no_amp + (
            amp_mask * source["amp"][previous_row, :, current]
        ).astype(np.float64)
        remaining_no_amp = error - no_amp
        remaining_full = error - full
        mse[valid, :, physical_block] = (
            (
                np.sum(np.square(remaining_no_amp), axis=2)
                - np.sum(np.square(remaining_full), axis=2)
            )
            / np.maximum(np.sum(np.square(error), axis=2), 1.0e-8)
        ).astype(np.float32)
        mae[valid, :, physical_block] = (
            (
                np.sum(np.abs(remaining_no_amp), axis=2)
                - np.sum(np.abs(remaining_full), axis=2)
            )
            / np.maximum(np.sum(np.abs(error), axis=2), 1.0e-8)
        ).astype(np.float32)
        available[valid, :, physical_block] = True
        require(
            np.all(
                origins[previous_row] + block_delay
                == origins[query_rows[valid]] - endpoint_lag
            ),
            "matured Amp marginal endpoint drift",
        )
    return mse.reshape(-1), mae.reshape(-1), available.reshape(-1)


def make_samples(
    source: dict[str, np.ndarray],
    use: np.ndarray,
    scaler: dict[str, np.ndarray],
    table: dict[str, np.ndarray],
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    x, error, level, amp, metadata = make_base_samples(source, use, scaler, horizon)
    prior, block = prior_actions(source, use, table, horizon)
    level_patch = np.isin(prior, (1, 5)).astype(np.float32)
    amp_patch = np.isin(prior, (4, 5)).astype(np.float32)
    level_point = np.repeat(level_patch, PATCH, axis=1)
    amp_point = np.repeat(amp_patch, PATCH, axis=1)
    x[:, 1] *= level_point
    x[:, 2] *= amp_point
    x = np.concatenate((x, level_point[:, None], amp_point[:, None]), axis=1)
    proposed_level = level * level_point
    proposed_amp = amp * amp_point
    corrections = np.stack(
        (
            proposed_level + proposed_amp,
            proposed_amp,
            proposed_level,
            np.zeros_like(proposed_level),
        ),
        axis=1,
    )
    remaining = error[:, None, :].astype(np.float64) - corrections.astype(np.float64)
    mse_ratio = np.sum(np.square(remaining), axis=2) / np.maximum(
        np.sum(np.square(error.astype(np.float64)), axis=1)[:, None], 1.0e-8
    )
    mae_ratio = np.sum(np.abs(remaining), axis=2) / np.maximum(
        np.sum(np.abs(error.astype(np.float64)), axis=1)[:, None], 1.0e-8
    )
    cost = np.maximum(mse_ratio, mae_ratio) + 0.25 * (mse_ratio + mae_ratio)
    label = np.argmin(cost, axis=1).astype(np.int64)

    current_mse, current_mae, current_available = matured_prior_utility(
        source, use, table, horizon, endpoint_lag=0
    )
    previous_mse, previous_mae, previous_available = matured_prior_utility(
        source, use, table, horizon, endpoint_lag=BLOCK
    )
    both = current_available & previous_available
    delta_mse = np.zeros_like(current_mse)
    delta_mae = np.zeros_like(current_mae)
    delta_mse[both] = current_mse[both] - previous_mse[both]
    delta_mae[both] = current_mae[both] - previous_mae[both]
    state = np.stack(
        (
            np.clip(current_mse, -4.0, 4.0),
            np.clip(current_mae, -4.0, 4.0),
            current_available.astype(np.float32),
            np.clip(previous_mse, -4.0, 4.0),
            np.clip(previous_mae, -4.0, 4.0),
            previous_available.astype(np.float32),
            np.clip(delta_mse, -4.0, 4.0),
            np.clip(delta_mae, -4.0, 4.0),
        ),
        axis=1,
    ).astype(np.float32)
    state = np.broadcast_to(state[:, :, None], (state.shape[0], STATE_INPUT_LANES, BLOCK))
    x = np.concatenate((x, state), axis=1).astype(np.float32)
    require(x.shape[1:] == (INPUT_LANES, BLOCK), "matured feature drift")
    require(np.isfinite(x).all() and np.isfinite(cost).all(), "nonfinite Gate sample")
    aux = {
        "block": block,
        "prior_action": prior,
        "error": error.astype(np.float32),
        "proposed_level": proposed_level.astype(np.float32),
        "proposed_amp": proposed_amp.astype(np.float32),
        "mse_ratio": mse_ratio.astype(np.float32),
        "mae_ratio": mae_ratio.astype(np.float32),
        "current_matured_available": current_available,
        "previous_matured_available": previous_available,
    }
    return x, label, cost.astype(np.float32), metadata, aux


def apply_veto(prior: np.ndarray, mask: np.ndarray) -> np.ndarray:
    require(prior.shape[0] == mask.size, "veto/prior group drift")
    output = prior.copy()
    remove_level = np.isin(mask, (1, 3))[:, None]
    remove_amp = np.isin(mask, (2, 3))[:, None]
    output[remove_level & (output == 1)] = 0
    output[remove_level & (output == 5)] = 4
    output[remove_amp & (output == 4)] = 0
    output[remove_amp & (output == 5)] = 1
    require(np.isin(output, CANONICAL_ACTIONS).all(), "veto emitted invalid subset")
    return output


def assemble_correction(
    block: dict[str, np.ndarray], action: np.ndarray, horizon: int
) -> np.ndarray:
    patch = patchize(block, horizon)
    require(action.shape == (patch["origin"].size,), "action shape drift")
    local = np.zeros_like(patch["base"], dtype=np.float64)
    for value in CANONICAL_ACTIONS[1:]:
        use = action == int(value)
        if np.any(use):
            candidate = _action_correction(
                patch["level"], patch["amp"], int(value)
            )
            local[use] = candidate[use]
    correction = np.zeros((block["origin"].size, horizon), dtype=np.float64)
    correction[:, : active_steps(horizon)] = local.reshape(
        block["origin"].size, active_steps(horizon)
    )
    return correction


class MaturedPriorTrendBlockVetoGate(nn.Module):
    """Shared H96 encoder selecting one of four exact deletion masks."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(INPUT_LANES, 24, kernel_size=5, padding=2),
            nn.GroupNorm(4, 24),
            nn.GELU(),
            nn.Conv1d(24, 24, kernel_size=3, padding=1),
            nn.GroupNorm(4, 24),
            nn.GELU(),
        )
        self.summary = nn.Sequential(nn.Linear(INPUT_LANES * 2, 16), nn.GELU())
        self.head = nn.Linear(40, len(MASK_NAMES))
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x).mean(dim=2)
        summary = self.summary(torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1))
        return self.head(torch.cat((encoded, summary), dim=1))


class ContinuousP12StrengthGate(nn.Module):
    """Shared H96 encoder producing bounded Level/Amp strength per p12 patch.

    The module owns no expert parameters.  Its positive identity bias makes a
    newly initialized Gate approximately preserve the independently fitted
    p12 prior; training can then reduce either frozen expert continuously.
    """

    def __init__(self, input_lanes: int, identity_logit: float = 4.0) -> None:
        super().__init__()
        require(input_lanes > 0, "continuous Gate input width must be positive")
        self.input_lanes = int(input_lanes)
        self.encoder = nn.Sequential(
            nn.Conv1d(self.input_lanes, 24, kernel_size=5, padding=2),
            nn.GroupNorm(4, 24),
            nn.GELU(),
            nn.Conv1d(24, 24, kernel_size=3, padding=1),
            nn.GroupNorm(4, 24),
            nn.GELU(),
        )
        self.head = nn.Conv1d(24, 2, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, float(identity_logit))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        require(
            x.ndim == 3
            and x.shape[1] == self.input_lanes
            and x.shape[2] == BLOCK,
            "continuous Gate input shape drift",
        )
        encoded = self.encoder(x).reshape(x.shape[0], 24, PATCHES, PATCH)
        return torch.sigmoid(self.head(encoded.mean(dim=3)))


def assemble_group_correction(
    block: dict[str, np.ndarray], local: np.ndarray, horizon: int
) -> np.ndarray:
    """Assemble one continuous H96 correction per row/physical-block group."""

    rows = int(block["origin"].size)
    expected = rows * complete_blocks(horizon)
    require(local.shape == (expected, BLOCK), "continuous group correction drift")
    correction = np.zeros((rows, horizon), dtype=np.float64)
    correction[:, : active_steps(horizon)] = local.astype(np.float64).reshape(
        rows, active_steps(horizon)
    )
    return correction


def action_counts(action: np.ndarray) -> dict[str, int]:
    names = {0: "SKIP", 1: "Level", 4: "Amp", 5: "Level+Amp"}
    return {names[int(value)]: int(np.sum(action == value)) for value in CANONICAL_ACTIONS}


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: float(metrics[key])
        for key in (
            "base_mse",
            "base_mae",
            "candidate_mse",
            "candidate_mae",
            "mse_gain_pct",
            "mae_gain_pct",
        )
    } | {
        "quarters": [
            {
                key: float(quarter[key])
                for key in (
                    "base_mse",
                    "base_mae",
                    "candidate_mse",
                    "candidate_mae",
                    "mse_gain_pct",
                    "mae_gain_pct",
                )
            }
            for quarter in metrics["time_quarters"]
        ]
    }
