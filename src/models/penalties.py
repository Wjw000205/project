from typing import Dict, Callable
import torch


# -----------------------------------------------------------------------------
# Per-penalty primitives. All return [B, C] tensors so they can be aggregated
# per-cluster downstream by `scatter_mean_bcf_to_bkf`.
# -----------------------------------------------------------------------------

def penalty_amp(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Symmetric amplitude alignment: penalises any deviation between the
    predicted and ground-truth horizon-level std. Pulls predicted std toward
    the truth from either side.

    Returns: [B, C]
    """
    std_p = y_hat.std(dim=-1)
    std_t = y.std(dim=-1)
    return (std_p - std_t).pow(2)


def penalty_amp_under(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    One-sided amplitude alignment: penalises ONLY when the predicted
    horizon-level std is smaller than the truth's. This directly counters
    the long-horizon "flat prediction" failure mode where MSE pushes the
    model toward the conditional mean. Predictions with std >= truth's std
    incur no penalty here, so the model is not punished for being lively.

    Returns: [B, C]
    """
    std_p = y_hat.std(dim=-1)
    std_t = y.std(dim=-1)
    deficit = (std_t - std_p).clamp_min(0.0)
    return deficit.pow(2)


def penalty_jitter(y_hat: torch.Tensor) -> torch.Tensor:
    """
    Second-difference smoothness. Penalises curvature of the prediction.
    NOTE: This is a one-sided regulariser (does not look at y), so it can
    contribute to over-flattening at long horizons. Use with care.

    Returns: [B, C]
    """
    if y_hat.shape[-1] < 3:
        return torch.zeros(y_hat.shape[0], y_hat.shape[1], device=y_hat.device, dtype=y_hat.dtype)
    d2 = y_hat[..., 2:] - 2.0 * y_hat[..., 1:-1] + y_hat[..., :-2]   # [B,C,H-2]
    return d2.pow(2).mean(dim=-1)


def penalty_d2_match(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Pointwise second-difference matching. Analogous to `delta` (first-difference
    matching) but acts on curvature.

        penalty = ((d2(y_hat) - d2(y))**2).mean(-1)

    This replaces the broken one-sided `jitter` penalty:
      - `jitter(y_hat)` minimizes y_hat's own curvature, pulling predictions
        toward flat — wrong direction on data where truth itself is curvy.
      - `d2_match(y_hat, y)` is direction-correct: zero when y_hat matches y's
        curvature, regardless of whether truth is flat or jittery.

    Returns: [B, C]
    """
    H_p = y_hat.shape[-1]; H_t = y.shape[-1]
    if H_p < 3 or H_t < 3:
        return torch.zeros(y.shape[0], y.shape[1], device=y.device, dtype=y.dtype)
    d2_p = y_hat[..., 2:] - 2.0 * y_hat[..., 1:-1] + y_hat[..., :-2]
    d2_t = y[..., 2:] - 2.0 * y[..., 1:-1] + y[..., :-2]
    return (d2_p - d2_t).pow(2).mean(dim=-1)


def penalty_jump(y_hat: torch.Tensor, y: torch.Tensor, thr: float = 2.0) -> torch.Tensor:
    """
    Jump-aware first-difference matching. Where the truth has a |Δy| above
    `thr`, penalise the squared mismatch between predicted and true Δ.
    Elsewhere the penalty is 0. Anchors the model on real high-energy events.

    Returns: [B, C]
    """
    if y.shape[-1] < 2:
        return torch.zeros(y.shape[0], y.shape[1], device=y.device, dtype=y.dtype)
    dy_t = y[..., 1:] - y[..., :-1]
    dy_p = y_hat[..., 1:] - y_hat[..., :-1]
    mask = (dy_t.abs() > thr).to(y.dtype)
    return ((dy_p - dy_t).pow(2) * mask).mean(dim=-1)


def penalty_smooth(y_hat: torch.Tensor) -> torch.Tensor:
    """
    First-difference smoothness. Penalises any change in the prediction.
    NOTE: This is a one-sided regulariser (does not look at y) and is a
    primary driver of long-horizon flattening. Replaced by `amp_under` /
    `delta` in the default config; kept here for ablation studies.

    Returns: [B, C]
    """
    if y_hat.shape[-1] < 2:
        return torch.zeros(y_hat.shape[0], y_hat.shape[1], device=y_hat.device, dtype=y_hat.dtype)
    d1 = y_hat[..., 1:] - y_hat[..., :-1]
    return d1.pow(2).mean(dim=-1)


def penalty_level(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Mean/level alignment. Penalises a horizon-level bias in the prediction.

    Returns: [B, C]
    """
    mean_p = y_hat.mean(dim=-1)
    mean_t = y.mean(dim=-1)
    return (mean_p - mean_t).pow(2)


def penalty_delta(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    First-difference matching. Penalises mismatch between predicted and true
    Δy at every step. Truth-anchored (does NOT push toward zero variance).

    Returns: [B, C]
    """
    if y.shape[-1] < 2:
        return torch.zeros(y.shape[0], y.shape[1], device=y.device, dtype=y.dtype)
    dy_t = y[..., 1:] - y[..., :-1]
    dy_p = y_hat[..., 1:] - y_hat[..., :-1]
    return (dy_p - dy_t).pow(2).mean(dim=-1)


def penalty_corr(y_hat: torch.Tensor, y: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """
    Horizon-shape correlation loss. It focuses on whether the predicted
    sequence bends with the target, independent of absolute level.

    Returns: [B, C]
    """
    pred = y_hat - y_hat.mean(dim=-1, keepdim=True)
    target = y - y.mean(dim=-1, keepdim=True)
    denom = pred.pow(2).sum(dim=-1).sqrt() * target.pow(2).sum(dim=-1).sqrt()
    corr = (pred * target).sum(dim=-1) / denom.clamp_min(eps)
    return 1.0 - corr.clamp(-1.0, 1.0)


def penalty_diff_amp(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    First-difference volatility alignment. This complements horizon-level
    amplitude by matching how much step-to-step movement exists.

    Returns: [B, C]
    """
    if y.shape[-1] < 2:
        return torch.zeros(y.shape[0], y.shape[1], device=y.device, dtype=y.dtype)
    dy_t = y[..., 1:] - y[..., :-1]
    dy_p = y_hat[..., 1:] - y_hat[..., :-1]
    std_p = dy_p.std(dim=-1)
    std_t = dy_t.std(dim=-1)
    return (std_p - std_t).pow(2)


def penalty_direction(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Directional movement under-fit loss. It penalises flat or opposite-sign
    predicted deltas when the target moves, while not punishing overshoot in
    the correct direction.

    Returns: [B, C]
    """
    if y.shape[-1] < 2:
        return torch.zeros(y.shape[0], y.shape[1], device=y.device, dtype=y.dtype)
    dy_t = y[..., 1:] - y[..., :-1]
    dy_p = y_hat[..., 1:] - y_hat[..., :-1]
    required = dy_t.abs()
    projected = dy_p * dy_t.sign()
    return (required - projected).clamp_min(0.0).pow(2).mean(dim=-1)


def penalty_range(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Horizon peak-to-trough range alignment. This gives the router a coarse
    amplitude/extent signal that is less sensitive to mean level.

    Returns: [B, C]
    """
    range_p = y_hat.amax(dim=-1) - y_hat.amin(dim=-1)
    range_t = y.amax(dim=-1) - y.amin(dim=-1)
    return (range_p - range_t).pow(2)


def penalty_trend(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Endpoint trend alignment over the prediction horizon.

    Returns: [B, C]
    """
    if y.shape[-1] < 2:
        return torch.zeros(y.shape[0], y.shape[1], device=y.device, dtype=y.dtype)
    trend_p = y_hat[..., -1] - y_hat[..., 0]
    trend_t = y[..., -1] - y[..., 0]
    return (trend_p - trend_t).pow(2)


# -----------------------------------------------------------------------------
# Cross-penalty utilities.
# -----------------------------------------------------------------------------

def normalize_penalties(
    pen_bcp: torch.Tensor,
    scale: torch.Tensor = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Normalise penalties so that different types live on a comparable scale
    before being mixed by lambda. If `scale` is None, uses each penalty's
    batch-channel mean as its own scale (per-step normalisation).

    pen_bcp: [B, C, P] -> [B, C, P]
    scale:   [P] or [1, 1, P], optional fixed scale (e.g. running mean).
    """
    if pen_bcp.numel() == 0:
        return pen_bcp
    if scale is None:
        scale = pen_bcp.mean(dim=(0, 1), keepdim=True)
    if scale.dim() == 1:
        scale = scale.view(1, 1, -1)
    scale = scale.clamp_min(eps)
    return pen_bcp / scale


# Mapping from penalty name -> (callable, needs_truth).
# A small registry keeps `build_penalty_bank` declarative and avoids the
# closure pitfall where lambdas in a loop can capture the wrong variables.
_PENALTY_FACTORIES: Dict[str, Callable[[float], Callable[..., torch.Tensor]]] = {
    "amp":       lambda jump_thr: (lambda yhat, y: penalty_amp(yhat, y)),
    "amp_under": lambda jump_thr: (lambda yhat, y: penalty_amp_under(yhat, y)),
    "jitter":    lambda jump_thr: (lambda yhat, y: penalty_jitter(yhat)),
    "d2_match":  lambda jump_thr: (lambda yhat, y: penalty_d2_match(yhat, y)),
    "smooth":    lambda jump_thr: (lambda yhat, y: penalty_smooth(yhat)),
    "level":     lambda jump_thr: (lambda yhat, y: penalty_level(yhat, y)),
    "delta":     lambda jump_thr: (lambda yhat, y: penalty_delta(yhat, y)),
    "corr":      lambda jump_thr: (lambda yhat, y: penalty_corr(yhat, y)),
    "diff_amp":  lambda jump_thr: (lambda yhat, y: penalty_diff_amp(yhat, y)),
    "direction": lambda jump_thr: (lambda yhat, y: penalty_direction(yhat, y)),
    "range":     lambda jump_thr: (lambda yhat, y: penalty_range(yhat, y)),
    "trend":     lambda jump_thr: (lambda yhat, y: penalty_trend(yhat, y)),
    # `jump` needs jump_thr captured at build time:
    "jump":      lambda jump_thr: (lambda yhat, y, _t=float(jump_thr): penalty_jump(yhat, y, thr=_t)),
}


def supported_penalty_names() -> tuple:
    """Names recognised by `build_penalty_bank`. Useful for config validation."""
    return tuple(_PENALTY_FACTORIES.keys())


def build_penalty_bank(enabled: list, jump_thr: float) -> Dict[str, Callable]:
    """
    Build a name -> callable mapping for the requested penalties.
    Each callable has signature (y_hat, y) -> [B, C].

    Raises ValueError on an unknown name, with the supported list included.
    """
    bank: Dict[str, Callable] = {}
    for name in enabled:
        factory = _PENALTY_FACTORIES.get(name)
        if factory is None:
            raise ValueError(
                f"Unknown penalty: '{name}'. Supported: {sorted(_PENALTY_FACTORIES.keys())}"
            )
        bank[name] = factory(float(jump_thr))
    return bank


def build_penalty_compute(enabled: list, jump_thr: float) -> Callable:
    """
    返回一个统一函数 compute(yhat, y) -> [B, C, P]，按 `enabled` 顺序输出所有惩罚。
    与 build_penalty_bank 等价，但共享中间量（d1, d2, mean, std）减少重复张量运算。
    例如 enabled=[jump, smooth, level, delta] 时 d1_p 仅算 1 次而非 3 次。
    """
    enabled = list(enabled)
    P = len(enabled)
    if P == 0:
        def empty_compute(yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.zeros(yhat.shape[0], yhat.shape[1], 0, device=yhat.device, dtype=yhat.dtype)
        return empty_compute

    name_set = set(enabled)
    unknown = name_set - set(_PENALTY_FACTORIES.keys())
    if unknown:
        raise ValueError(
            f"Unknown penalty: {sorted(unknown)}. Supported: {sorted(_PENALTY_FACTORIES.keys())}"
        )

    needs_d1_p = bool(name_set & {"smooth", "jump", "delta", "diff_amp", "direction"})
    needs_d1_t = bool(name_set & {"jump", "delta", "diff_amp", "direction"})
    needs_d2_p = bool(name_set & {"jitter", "d2_match"})
    needs_d2_t = "d2_match" in name_set
    needs_std = bool(name_set & {"amp", "amp_under"})
    needs_mean = bool(name_set & {"level", "corr"})
    needs_centered = "corr" in name_set
    needs_range = "range" in name_set
    needs_trend = "trend" in name_set
    thr = float(jump_thr)

    def compute(yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        H_p = yhat.shape[-1]
        H_t = y.shape[-1]
        zero_bc = None  # 仅在需要时构造，避免无谓分配

        def _zero():
            nonlocal zero_bc
            if zero_bc is None:
                zero_bc = torch.zeros(yhat.shape[0], yhat.shape[1], device=yhat.device, dtype=yhat.dtype)
            return zero_bc

        d1_p = (yhat[..., 1:] - yhat[..., :-1]) if (needs_d1_p and H_p >= 2) else None
        d1_t = (y[..., 1:] - y[..., :-1]) if (needs_d1_t and H_t >= 2) else None
        d2_p = (yhat[..., 2:] - 2.0 * yhat[..., 1:-1] + yhat[..., :-2]) if (needs_d2_p and H_p >= 3) else None
        d2_t = (y[..., 2:] - 2.0 * y[..., 1:-1] + y[..., :-2]) if (needs_d2_t and H_t >= 3) else None
        std_p = yhat.std(dim=-1) if needs_std else None
        std_t = y.std(dim=-1) if needs_std else None
        mean_p = yhat.mean(dim=-1) if needs_mean else None
        mean_t = y.mean(dim=-1) if needs_mean else None
        if needs_centered:
            cp = yhat - (mean_p.unsqueeze(-1) if mean_p is not None else yhat.mean(dim=-1, keepdim=True))
            ct = y - (mean_t.unsqueeze(-1) if mean_t is not None else y.mean(dim=-1, keepdim=True))
        else:
            cp = ct = None

        results = []
        for name in enabled:
            if name == "amp":
                results.append((std_p - std_t).pow(2))
            elif name == "amp_under":
                results.append((std_t - std_p).clamp_min(0.0).pow(2))
            elif name == "jitter":
                results.append(d2_p.pow(2).mean(dim=-1) if d2_p is not None else _zero())
            elif name == "d2_match":
                if d2_p is None or d2_t is None:
                    results.append(_zero())
                else:
                    results.append((d2_p - d2_t).pow(2).mean(dim=-1))
            elif name == "smooth":
                results.append(d1_p.pow(2).mean(dim=-1) if d1_p is not None else _zero())
            elif name == "level":
                results.append((mean_p - mean_t).pow(2))
            elif name == "delta":
                if d1_p is None or d1_t is None:
                    results.append(_zero())
                else:
                    results.append((d1_p - d1_t).pow(2).mean(dim=-1))
            elif name == "jump":
                if d1_p is None or d1_t is None:
                    results.append(_zero())
                else:
                    mask = (d1_t.abs() > thr).to(d1_p.dtype)
                    results.append(((d1_p - d1_t).pow(2) * mask).mean(dim=-1))
            elif name == "corr":
                denom = cp.pow(2).sum(dim=-1).sqrt() * ct.pow(2).sum(dim=-1).sqrt()
                corr = (cp * ct).sum(dim=-1) / denom.clamp_min(1.0e-6)
                results.append(1.0 - corr.clamp(-1.0, 1.0))
            elif name == "diff_amp":
                if d1_p is None or d1_t is None:
                    results.append(_zero())
                else:
                    results.append((d1_p.std(dim=-1) - d1_t.std(dim=-1)).pow(2))
            elif name == "direction":
                if d1_p is None or d1_t is None:
                    results.append(_zero())
                else:
                    required = d1_t.abs()
                    projected = d1_p * d1_t.sign()
                    results.append((required - projected).clamp_min(0.0).pow(2).mean(dim=-1))
            elif name == "range":
                results.append((yhat.amax(dim=-1) - yhat.amin(dim=-1) - (y.amax(dim=-1) - y.amin(dim=-1))).pow(2))
            elif name == "trend":
                if H_p < 2 or H_t < 2:
                    results.append(_zero())
                else:
                    results.append(((yhat[..., -1] - yhat[..., 0]) - (y[..., -1] - y[..., 0])).pow(2))
            else:
                # 已在 enabled 校验阶段排除未知项；此处仅作防御
                raise ValueError(f"Unknown penalty: '{name}'")

        return torch.stack(results, dim=-1)

    return compute
