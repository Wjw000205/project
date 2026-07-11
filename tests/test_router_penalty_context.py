from __future__ import annotations

from collections import Counter

import torch

from src.train import _router_penalty_context_from_history


def _router_inputs():
    torch.manual_seed(2026)
    return {
        "x_bcl": torch.randn(3, 4, 8),
        "yhat_base_bch": torch.randn(3, 4, 6),
        "cluster_id_c": torch.tensor([0, 0, 1, 1], dtype=torch.long),
        "K": 2,
    }


def _counted_penalties(calls: Counter):
    penalties = {}
    for index, name in enumerate(("level", "delta", "d2", "amp"), start=1):
        def penalty(pred: torch.Tensor, ref: torch.Tensor, *, _name=name, _value=float(index)):
            calls[_name] += 1
            return (pred - ref).square().mean(dim=-1) + _value

        penalties[name] = penalty
    return penalties


def test_learned_router_skips_penalty_context_computation() -> None:
    calls = Counter()
    penalty_fns = _counted_penalties(calls)

    context = _router_penalty_context_from_history(
        **_router_inputs(),
        penalty_names=list(penalty_fns),
        penalty_fns=penalty_fns,
        penalty_scale=None,
        router_mode="learned",
    )

    assert calls == Counter()
    assert context.shape == (3, 2, 4)
    assert torch.count_nonzero(context).item() == 0


def test_penalty_context_mode_preserves_existing_context_values() -> None:
    calls = Counter()
    penalty_fns = _counted_penalties(calls)
    inputs = _router_inputs()

    legacy = _router_penalty_context_from_history(
        **inputs,
        penalty_names=list(penalty_fns),
        penalty_fns=penalty_fns,
        penalty_scale=None,
    )
    routed = _router_penalty_context_from_history(
        **inputs,
        penalty_names=list(penalty_fns),
        penalty_fns=penalty_fns,
        penalty_scale=None,
        router_mode="penalty_context",
    )

    assert calls == Counter({name: 2 for name in penalty_fns})
    assert torch.equal(routed, legacy)
