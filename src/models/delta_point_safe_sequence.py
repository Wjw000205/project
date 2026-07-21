"""Point-safe cumulative-Delta sequence adapter.

The learned model consumes only target-free causal state and emits one signed
coefficient for every cumulative p12 Delta coordinate.  Its coefficients are
decoded through an analytic, mean-free basis; consequently the model cannot
emit Level, a free waveform, or another expert's action.

The module also owns the no-hyperparameter convex QCQP used to construct
TRAIN-only supervision.  That target is the best point-residual correction in
the declared Delta span subject to not increasing first-difference error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


P12_STEPS = 12
PHYSICAL_PERIOD_STEPS = 24
STATE_LANES = 12


@dataclass(frozen=True)
class DeltaGeometry:
    """Deterministic cumulative-p12 basis geometry.

    ``raw_basis`` contains the locally meaningful cumulative-p12 lanes.
    ``orthonormal_basis`` contains the same span with row Gram ``H I``.
    ``qr_r`` maps raw column coordinates to the orthonormal coordinates.
    """

    raw_basis: torch.Tensor
    orthonormal_basis: torch.Tensor
    qr_r: torch.Tensor


def build_cumulative_p12_delta_geometry(
    horizon: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> DeltaGeometry:
    """Build the fixed, mean-free cumulative-p12 Delta span."""

    horizon = int(horizon)
    if horizon <= 1 or horizon % PHYSICAL_PERIOD_STEPS != 0:
        raise ValueError("Delta horizon must contain complete P24 periods")
    positions = torch.arange(horizon, device=device, dtype=dtype)
    lanes: list[torch.Tensor] = []
    for patch in range(horizon // P12_STEPS):
        left = patch * P12_STEPS
        lane = (positions - float(left) + 1.0).clamp(0.0, float(P12_STEPS))
        lanes.append(lane - lane.mean())
    raw = torch.stack(lanes, dim=0)
    q, r = torch.linalg.qr(raw.transpose(0, 1), mode="reduced")
    diagonal = torch.diagonal(r).abs()
    if bool(torch.any(diagonal <= 1.0e-10)):
        raise RuntimeError("cumulative-p12 Delta basis lost rank")
    orthonormal = q.transpose(0, 1) * math.sqrt(float(horizon))
    return DeltaGeometry(raw_basis=raw, orthonormal_basis=orthonormal, qr_r=r)


def raw_to_orthonormal_coordinates(
    raw_coordinates: torch.Tensor,
    geometry: DeltaGeometry,
) -> torch.Tensor:
    """Map local cumulative-p12 coordinates to the row-orthonormal basis."""

    if raw_coordinates.ndim != 2:
        raise ValueError("raw coordinates must be [N,K]")
    k, horizon = geometry.raw_basis.shape
    if raw_coordinates.shape[1] != k:
        raise ValueError("raw coordinate width does not match Delta geometry")
    r = geometry.qr_r.to(raw_coordinates)
    return raw_coordinates @ r.transpose(0, 1) / math.sqrt(float(horizon))


def orthonormal_to_raw_coordinates(
    coordinates: torch.Tensor,
    geometry: DeltaGeometry,
) -> torch.Tensor:
    """Map row-orthonormal coordinates to local cumulative-p12 coordinates."""

    if coordinates.ndim != 2:
        raise ValueError("orthonormal coordinates must be [N,K]")
    k, horizon = geometry.raw_basis.shape
    if coordinates.shape[1] != k:
        raise ValueError("coordinate width does not match Delta geometry")
    r = geometry.qr_r.to(coordinates)
    rhs = math.sqrt(float(horizon)) * coordinates.transpose(0, 1)
    return torch.linalg.solve_triangular(r, rhs, upper=True).transpose(0, 1)


def decode_raw_delta_coordinates(
    raw_coordinates: torch.Tensor,
    geometry: DeltaGeometry,
) -> torch.Tensor:
    """Decode local coordinates to one actual point-residual action."""

    return raw_coordinates @ geometry.raw_basis.to(raw_coordinates)


@torch.no_grad()
def point_safe_delta_qcqp(
    residual: torch.Tensor,
    geometry: DeltaGeometry,
    *,
    bisection_steps: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
    """Solve the point-optimal, first-difference-safe Delta projection.

    For each residual row ``r`` this solves

    ``min_z ||r - zB||^2`` subject to
    ``||D(r-zB)||^2 <= ||Dr||^2``.

    ``B`` is the fixed row-orthonormal Delta basis.  The unconstrained point
    projection is used whenever feasible.  Otherwise the one KKT multiplier
    is bracketed and bisected without a loss weight or scale grid.
    """

    if residual.ndim != 2:
        raise ValueError("residual must be [N,H]")
    basis = geometry.orthonormal_basis.to(device=residual.device, dtype=torch.float64)
    values = residual.to(dtype=torch.float64)
    n, horizon = values.shape
    k, basis_horizon = basis.shape
    if horizon != basis_horizon:
        raise ValueError("residual horizon does not match Delta geometry")
    if n == 0:
        return values.clone(), values.new_zeros((0, k)), {
            "rows": 0,
            "active_constraint_rows": 0,
            "numeric_noop_rows": 0,
            "max_point_risk_increase": 0.0,
            "max_delta_risk_increase": 0.0,
        }

    gram_error = torch.max(
        torch.abs(
            basis @ basis.transpose(0, 1)
            - float(horizon) * torch.eye(k, device=basis.device, dtype=basis.dtype)
        )
    )
    if float(gram_error) > 1.0e-7 * float(horizon):
        raise RuntimeError("Delta basis row Gram drift")

    d_basis = torch.diff(basis, dim=1).transpose(0, 1)  # [H-1,K]
    d_residual = torch.diff(values, dim=1)
    point_rhs = values @ basis.transpose(0, 1)
    operator_rhs = d_residual @ d_basis
    operator_gram = d_basis.transpose(0, 1) @ d_basis
    eigenvalues, eigenvectors = torch.linalg.eigh(operator_gram)
    if float(eigenvalues.min()) <= 1.0e-12:
        raise RuntimeError("Delta first-difference Gram lost rank")

    point_rhs_eigen = point_rhs @ eigenvectors
    unconstrained = point_rhs / float(horizon)
    unconstrained_eigen = point_rhs_eigen / float(horizon)
    operator_rhs_eigen = operator_rhs @ eigenvectors

    def constraint_value(eigen_coordinates: torch.Tensor) -> torch.Tensor:
        return torch.sum(
            eigenvalues.unsqueeze(0) * eigen_coordinates.square()
            - 2.0 * operator_rhs_eigen * eigen_coordinates,
            dim=1,
        )

    phi0 = constraint_value(unconstrained_eigen)
    active = phi0 > 0.0
    coordinates_eigen = unconstrained_eigen.clone()

    if bool(active.any()):
        br = point_rhs_eigen[active]
        gd = operator_rhs_eigen[active]
        eigen = eigenvalues.unsqueeze(0)

        def active_coordinates(tau: torch.Tensor) -> torch.Tensor:
            one_minus = 1.0 - tau
            return (
                one_minus.unsqueeze(1) * br + tau.unsqueeze(1) * gd
            ) / (
                one_minus.unsqueeze(1) * float(horizon)
                + tau.unsqueeze(1) * eigen
            )

        def active_phi(tau: torch.Tensor) -> torch.Tensor:
            current = active_coordinates(tau)
            return torch.sum(eigen * current.square() - 2.0 * gd * current, dim=1)

        lo = torch.zeros(int(active.sum()), device=values.device, dtype=torch.float64)
        hi = torch.ones_like(lo)
        if bool((active_phi(hi) > 1.0e-10).any()):
            raise RuntimeError("tau=1 is not feasible for point-safe Delta QCQP")
        for _ in range(int(bisection_steps)):
            mid = 0.5 * (lo + hi)
            positive = active_phi(mid) > 0.0
            lo = torch.where(positive, mid, lo)
            hi = torch.where(positive, hi, mid)
        coordinates_eigen[active] = active_coordinates(hi)

    coordinates = coordinates_eigen @ eigenvectors.transpose(0, 1)
    action = coordinates @ basis
    d_action = torch.diff(action, dim=1)
    point_dot = torch.sum(values * action, dim=1)
    point_energy = action.square().sum(dim=1)
    delta_dot = torch.sum(d_residual * d_action, dim=1)
    delta_energy = d_action.square().sum(dim=1)
    point_gain = 2.0 * point_dot - point_energy
    delta_gain = (
        2.0 * delta_dot - delta_energy
    )
    point_scale = values.square().sum(dim=1).clamp_min(1.0)
    delta_scale = d_residual.square().sum(dim=1).clamp_min(1.0)
    eps = torch.finfo(torch.float64).eps
    point_tolerance = 256.0 * eps * point_scale
    delta_tolerance = 256.0 * eps * delta_scale
    point_bad = point_gain < -point_tolerance
    delta_bad = delta_gain < -delta_tolerance
    if bool(point_bad.any() or delta_bad.any()):
        raise RuntimeError("point-safe Delta QCQP violated a risk constraint")
    numeric_shrink = (point_gain < 0.0) | (delta_gain < 0.0)
    if bool(numeric_shrink.any()):
        one = torch.ones_like(point_gain)
        point_limit = torch.where(
            point_energy > 0.0,
            2.0 * point_dot / point_energy.clamp_min(torch.finfo(torch.float64).tiny),
            one,
        )
        delta_limit = torch.where(
            delta_energy > 0.0,
            2.0 * delta_dot / delta_energy.clamp_min(torch.finfo(torch.float64).tiny),
            one,
        )
        safe_scale = torch.minimum(one, torch.minimum(point_limit, delta_limit)).clamp_min(0.0)
        safe_scale = torch.nextafter(safe_scale, torch.zeros_like(safe_scale))
        safe_scale = torch.where(numeric_shrink, safe_scale, one)
        coordinates = coordinates * safe_scale.unsqueeze(1)
        action = action * safe_scale.unsqueeze(1)
        d_action = torch.diff(action, dim=1)
        point_gain = 2.0 * torch.sum(values * action, dim=1) - action.square().sum(dim=1)
        delta_gain = (
            2.0 * torch.sum(d_residual * d_action, dim=1)
            - d_action.square().sum(dim=1)
        )
        if bool((point_gain < -point_tolerance).any() or (delta_gain < -delta_tolerance).any()):
            raise RuntimeError("numeric feasibility shrink failed for Delta QCQP")

    return action, coordinates, {
        "rows": int(n),
        "active_constraint_rows": int(active.sum()),
        "numeric_noop_rows": int((action.square().sum(dim=1) == 0.0).sum()),
        "numeric_shrink_rows": int(numeric_shrink.sum()),
        "max_point_risk_increase": float((-point_gain).clamp_min(0.0).max()),
        "max_delta_risk_increase": float((-delta_gain).clamp_min(0.0).max()),
        "basis_gram_max_abs_error": float(gram_error),
    }


class PointSafeCumulativeDeltaAdapter(nn.Module):
    """Shared p12 Conv + shared P24 GRU Delta-coordinate adapter.

    Horizon and channel are batch/sequence geometry only; neither appears in
    the parameter shapes.  The same GRU performs the forward and reverse
    sweeps over fully target-free forecast tokens available at the origin.
    """

    def __init__(self, coordinate_scale: float = 1.0) -> None:
        super().__init__()
        coordinate_scale = float(coordinate_scale)
        if not math.isfinite(coordinate_scale) or coordinate_scale <= 0.0:
            raise ValueError("coordinate_scale must be finite and positive")
        self.register_buffer(
            "coordinate_scale", torch.tensor(coordinate_scale, dtype=torch.float32)
        )
        self.patch_conv = nn.Sequential(
            nn.Conv1d(STATE_LANES, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1, groups=32),
            nn.SiLU(),
            nn.Conv1d(32, 32, kernel_size=1),
            nn.SiLU(),
        )
        self.period_mixer = nn.Sequential(
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
        )
        self.period_gru = nn.GRU(input_size=64, hidden_size=64, batch_first=True)
        self.coordinate_hidden = nn.Sequential(nn.Linear(192, 64), nn.SiLU())
        self.coordinate_head = nn.Linear(64, 2)
        nn.init.zeros_(self.coordinate_head.weight)
        nn.init.zeros_(self.coordinate_head.bias)

    def _encode_period_hidden(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 3 or state.shape[1] != STATE_LANES:
            raise ValueError("Delta state must be [B,12,H]")
        batch, _lanes, horizon = state.shape
        if horizon <= 0 or horizon % PHYSICAL_PERIOD_STEPS != 0:
            raise ValueError("Delta state horizon must contain complete P24 periods")
        p12_count = horizon // P12_STEPS
        patches = (
            state.reshape(batch, STATE_LANES, p12_count, P12_STEPS)
            .permute(0, 2, 1, 3)
            .reshape(batch * p12_count, STATE_LANES, P12_STEPS)
        )
        patch_embedding = self.patch_conv(patches).mean(dim=2)
        period_count = p12_count // 2
        period_tokens = patch_embedding.reshape(batch, period_count, 64)
        period_tokens = self.period_mixer(period_tokens)
        forward_state, _ = self.period_gru(period_tokens)
        backward_reverse, _ = self.period_gru(torch.flip(period_tokens, dims=(1,)))
        backward_state = torch.flip(backward_reverse, dims=(1,))
        hidden = self.coordinate_hidden(
            torch.cat([period_tokens, forward_state, backward_state], dim=2)
        )
        return hidden

    def encode_p12_patch_embeddings(self, state: torch.Tensor) -> torch.Tensor:
        """Expose the frozen proposal's local p12 embeddings.

        This deliberately duplicates the patch portion of
        :meth:`_encode_period_hidden` instead of routing the legacy forward path
        through a new helper.  Existing adapters and checkpoints therefore keep
        their original parameter names and numerical execution unchanged.
        """

        if state.ndim != 3 or state.shape[1] != STATE_LANES:
            raise ValueError("Delta state must be [B,12,H]")
        batch, _lanes, horizon = state.shape
        if horizon <= 0 or horizon % PHYSICAL_PERIOD_STEPS != 0:
            raise ValueError("Delta state horizon must contain complete P24 periods")
        p12_count = horizon // P12_STEPS
        patches = (
            state.reshape(batch, STATE_LANES, p12_count, P12_STEPS)
            .permute(0, 2, 1, 3)
            .reshape(batch * p12_count, STATE_LANES, P12_STEPS)
        )
        return self.patch_conv(patches).mean(dim=2).reshape(batch, p12_count, 32)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        hidden = self._encode_period_hidden(state)
        batch = state.shape[0]
        p12_count = state.shape[2] // P12_STEPS
        raw = self.coordinate_head(hidden).reshape(batch, p12_count)
        return raw * self.coordinate_scale.to(device=raw.device, dtype=raw.dtype)

    def decode(self, state: torch.Tensor, geometry: DeltaGeometry) -> torch.Tensor:
        raw_coordinates = self(state)
        return decode_raw_delta_coordinates(raw_coordinates, geometry)


class CausalNullPairedDeltaAdapter(PointSafeCumulativeDeltaAdapter):
    """Shared Delta mapper whose action is a causal-state paired difference.

    The reference state is constructed outside the network from fully causal
    past states.  Subtracting shared hidden representations before one
    bias-free coordinate head makes a common/static network template cancel
    exactly while keeping every output inside the fixed Delta coordinate span.
    """

    def __init__(self, coordinate_scale: float = 1.0) -> None:
        super().__init__(coordinate_scale=coordinate_scale)
        self.coordinate_head = nn.Linear(64, 2, bias=False)
        nn.init.zeros_(self.coordinate_head.weight)

    def forward(
        self, state: torch.Tensor, reference_state: torch.Tensor
    ) -> torch.Tensor:
        if state.shape != reference_state.shape:
            raise ValueError("current/reference Delta states must have equal shape")
        current_hidden = self._encode_period_hidden(state)
        reference_hidden = self._encode_period_hidden(reference_state)
        hidden_delta = current_hidden - reference_hidden
        batch = state.shape[0]
        p12_count = state.shape[2] // P12_STEPS
        raw = self.coordinate_head(hidden_delta).reshape(batch, p12_count)
        return raw * self.coordinate_scale.to(device=raw.device, dtype=raw.dtype)

    def decode(
        self,
        state: torch.Tensor,
        reference_state: torch.Tensor,
        geometry: DeltaGeometry,
    ) -> torch.Tensor:
        raw_coordinates = self(state, reference_state)
        return decode_raw_delta_coordinates(raw_coordinates, geometry)


class ClusterContextualSignedP12DeltaAdapter(nn.Module):
    """Content-positioned signed Delta proposal with dynamic cluster context.

    The adapter owns the complete signed proposal: every p12 token receives one
    unconstrained raw Delta coordinate.  Cluster labels are used only to pool
    simultaneous channel features at the current origin.  They never index an
    embedding, lookup table, parameter set, or fixed output template.

    All operations after the shared current/reference difference are
    bias-free.  Consequently equal current and reference states produce exact
    zero at *any* fitted weights.  The final signed head is also zero-initialized,
    so a newly constructed adapter is an exact NOOP for arbitrary inputs.

    Dataset, horizon, channel count, cluster count, and absolute p12 position
    affect tensor geometry only and do not occur in any parameter shape.
    """

    _HIDDEN_WIDTH = 32
    _DILATIONS = (1, 2, 4, 8)

    def __init__(self, coordinate_scale: float = 1.0) -> None:
        super().__init__()
        coordinate_scale = float(coordinate_scale)
        if not math.isfinite(coordinate_scale) or coordinate_scale <= 0.0:
            raise ValueError("coordinate_scale must be finite and positive")
        self.register_buffer(
            "coordinate_scale", torch.tensor(coordinate_scale, dtype=torch.float32)
        )

        # This encoder is shared by current and reference patches.  Its biases
        # cancel exactly when the two input states are equal.
        self.patch_encoder = nn.Sequential(
            nn.Conv1d(STATE_LANES, self._HIDDEN_WIDTH, kernel_size=3, padding=1),
            nn.GroupNorm(4, self._HIDDEN_WIDTH),
            nn.SiLU(),
            nn.Conv1d(
                self._HIDDEN_WIDTH,
                self._HIDDEN_WIDTH,
                kernel_size=3,
                padding=1,
                groups=self._HIDDEN_WIDTH,
            ),
            nn.SiLU(),
            nn.Conv1d(
                self._HIDDEN_WIDTH, self._HIDDEN_WIDTH, kernel_size=1
            ),
            nn.SiLU(),
        )

        # Mean state may modulate the paired difference, but it cannot create
        # an action by itself because it appears only in a product with that
        # difference.
        self.mean_gate = nn.Linear(
            self._HIDDEN_WIDTH, self._HIDDEN_WIDTH
        )
        self.paired_projection = nn.Linear(
            2 * self._HIDDEN_WIDTH, self._HIDDEN_WIDTH, bias=False
        )

        # Local state plus dynamic same-cluster token mean/RMS.  No fitted
        # cluster statistic is stored in the network.
        self.context_projection = nn.Linear(
            3 * self._HIDDEN_WIDTH, self._HIDDEN_WIDTH, bias=False
        )
        self.dilated_convs = nn.ModuleList(
            nn.Conv1d(
                self._HIDDEN_WIDTH,
                self._HIDDEN_WIDTH,
                kernel_size=3,
                dilation=dilation,
                bias=False,
            )
            for dilation in self._DILATIONS
        )

        # One shared signed head consumes local sequence, row, and dynamic
        # cluster-global summaries.  It has no bias or token index input.
        self.coordinate_hidden = nn.Linear(
            5 * self._HIDDEN_WIDTH, self._HIDDEN_WIDTH, bias=False
        )
        self.coordinate_head = nn.Linear(
            self._HIDDEN_WIDTH, 1, bias=False
        )
        nn.init.zeros_(self.coordinate_head.weight)

    @staticmethod
    def _validate_inputs(
        state: torch.Tensor,
        reference_state: torch.Tensor,
        cluster_ids: torch.Tensor,
    ) -> tuple[int, int, int, int]:
        if state.ndim != 4 or state.shape[2] != STATE_LANES:
            raise ValueError("Delta state must be [B,C,12,H]")
        if state.shape != reference_state.shape:
            raise ValueError("current/reference Delta states must have equal shape")
        batch, channels, _lanes, horizon = state.shape
        if batch <= 0 or channels <= 0:
            raise ValueError("batch and channel dimensions must be positive")
        if horizon <= 0 or horizon % PHYSICAL_PERIOD_STEPS != 0:
            raise ValueError("Delta state horizon must contain complete P24 periods")
        if cluster_ids.ndim != 1 or cluster_ids.shape[0] != channels:
            raise ValueError("cluster_ids must be [C]")
        if cluster_ids.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise ValueError("cluster_ids must contain integer labels")
        return batch, channels, horizon // P12_STEPS, horizon

    @staticmethod
    def _zero_at_zero_sqrt(values: torch.Tensor) -> torch.Tensor:
        """Nonnegative square root whose exact zero input stays exact zero."""

        epsilon = values.new_tensor(torch.finfo(values.dtype).eps)
        return torch.sqrt(values.clamp_min(0.0) + epsilon) - torch.sqrt(epsilon)

    @classmethod
    def _zero_at_zero_rms(
        cls, values: torch.Tensor, *, dim: int | tuple[int, ...]
    ) -> torch.Tensor:
        """RMS magnitude whose exact all-zero input maps back to exact zero."""

        return cls._zero_at_zero_sqrt(values.square().mean(dim=dim))

    def _encode_patches(
        self, state: torch.Tensor, *, p12_count: int
    ) -> torch.Tensor:
        batch, channels, _lanes, _horizon = state.shape
        patches = (
            state.reshape(
                batch, channels, STATE_LANES, p12_count, P12_STEPS
            )
            .permute(0, 1, 3, 2, 4)
            .reshape(
                batch * channels * p12_count, STATE_LANES, P12_STEPS
            )
        )
        return self.patch_encoder(patches).mean(dim=2).reshape(
            batch, channels, p12_count, self._HIDDEN_WIDTH
        )

    @staticmethod
    def _cluster_weights(
        cluster_ids: torch.Tensor, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        labels = cluster_ids.detach().to(device=device)
        membership = labels.unsqueeze(1).eq(labels.unsqueeze(0))
        weights = membership.to(dtype=dtype)
        return weights / weights.sum(dim=1, keepdim=True)

    def forward(
        self,
        state: torch.Tensor,
        reference_state: torch.Tensor,
        cluster_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, p12_count, _horizon = self._validate_inputs(
            state, reference_state, cluster_ids
        )
        current = self._encode_patches(state, p12_count=p12_count)
        reference = self._encode_patches(
            reference_state, p12_count=p12_count
        )
        difference = current - reference
        mean_state = 0.5 * (current + reference)
        gated_difference = difference * torch.tanh(self.mean_gate(mean_state))
        hidden = self.paired_projection(
            torch.cat([difference, gated_difference], dim=3)
        )

        cluster_weights = self._cluster_weights(
            cluster_ids, device=hidden.device, dtype=hidden.dtype
        )
        cluster_token_mean = torch.einsum(
            "cd,bdkh->bckh", cluster_weights, hidden
        )
        cluster_token_rms = self._zero_at_zero_sqrt(
            torch.einsum(
                "cd,bdkh->bckh", cluster_weights, hidden.square()
            )
        )
        hidden = torch.nn.functional.silu(
            self.context_projection(
                torch.cat(
                    [hidden, cluster_token_mean, cluster_token_rms], dim=3
                )
            )
        )

        # Replicate padding makes a constant token sequence remain constant;
        # the convolutions can learn from relative content without manufacturing
        # an absolute p12 template at the horizon boundaries.
        sequence = hidden.reshape(
            batch * channels, p12_count, self._HIDDEN_WIDTH
        ).transpose(1, 2)
        for convolution, dilation in zip(self.dilated_convs, self._DILATIONS):
            padded = torch.nn.functional.pad(
                sequence, (dilation, dilation), mode="replicate"
            )
            sequence = sequence + torch.nn.functional.silu(convolution(padded))
        hidden = sequence.transpose(1, 2).reshape(
            batch, channels, p12_count, self._HIDDEN_WIDTH
        )

        row_mean = hidden.mean(dim=2)
        row_rms = self._zero_at_zero_rms(hidden, dim=2)
        cluster_global_mean = torch.einsum(
            "cd,bdh->bch", cluster_weights, row_mean
        )
        cluster_global_rms = self._zero_at_zero_sqrt(
            torch.einsum(
                "cd,bdh->bch",
                cluster_weights,
                hidden.square().mean(dim=2),
            )
        )
        context = torch.cat(
            [
                hidden,
                row_mean.unsqueeze(2).expand(-1, -1, p12_count, -1),
                row_rms.unsqueeze(2).expand(-1, -1, p12_count, -1),
                cluster_global_mean.unsqueeze(2).expand(
                    -1, -1, p12_count, -1
                ),
                cluster_global_rms.unsqueeze(2).expand(
                    -1, -1, p12_count, -1
                ),
            ],
            dim=3,
        )
        signed_hidden = torch.nn.functional.silu(
            self.coordinate_hidden(context)
        )
        raw = self.coordinate_head(signed_hidden).squeeze(3)
        return raw * self.coordinate_scale.to(device=raw.device, dtype=raw.dtype)

    def decode(
        self,
        state: torch.Tensor,
        reference_state: torch.Tensor,
        cluster_ids: torch.Tensor,
        geometry: DeltaGeometry,
    ) -> torch.Tensor:
        raw = self(state, reference_state, cluster_ids)
        batch, channels, p12_count = raw.shape
        action = decode_raw_delta_coordinates(
            raw.reshape(batch * channels, p12_count), geometry
        )
        return action.reshape(batch, channels, action.shape[1])


class SharedP12DeltaConfidenceHead(nn.Module):
    """A position-shared suppressive confidence model for a frozen proposal.

    The 65 input values at each p12 token are the difference and mean of the
    frozen current/reference 32D patch embeddings plus the corresponding
    frozen signed proposal coordinate.  No channel, absolute-position, or
    horizon identifier is present.  The head owns 2,145 parameters regardless
    of the number of p12 tokens.

    Confidence is intentionally not another residual-magnitude head.  The
    proposal coordinate and proposal embeddings are detached by default, and
    the output can only retain a value in ``[0, 1]`` of that frozen signed
    proposal.
    """

    def __init__(self, proposal_coordinate_scale: float = 1.0) -> None:
        super().__init__()
        proposal_coordinate_scale = float(proposal_coordinate_scale)
        if (
            not math.isfinite(proposal_coordinate_scale)
            or proposal_coordinate_scale <= 0.0
        ):
            raise ValueError("proposal_coordinate_scale must be finite and positive")
        self.register_buffer(
            "proposal_coordinate_scale",
            torch.tensor(proposal_coordinate_scale, dtype=torch.float32),
        )
        self.hidden = nn.Linear(65, 32)
        self.output = nn.Linear(32, 1)

    def forward(
        self,
        current_patch_embeddings: torch.Tensor,
        reference_patch_embeddings: torch.Tensor,
        raw_proposal_coordinates: torch.Tensor,
        *,
        stop_gradient_proposal: bool = True,
    ) -> torch.Tensor:
        if current_patch_embeddings.ndim != 3:
            raise ValueError("current patch embeddings must be [B,K,32]")
        if current_patch_embeddings.shape != reference_patch_embeddings.shape:
            raise ValueError("current/reference patch embeddings must have equal shape")
        if current_patch_embeddings.shape[2] != 32:
            raise ValueError("current/reference patch embeddings must have width 32")
        expected_coordinates = current_patch_embeddings.shape[:2]
        if raw_proposal_coordinates.shape != expected_coordinates:
            raise ValueError("raw proposal coordinates must be [B,K]")

        current = current_patch_embeddings
        reference = reference_patch_embeddings
        proposal = raw_proposal_coordinates
        if stop_gradient_proposal:
            current = current.detach()
            reference = reference.detach()
            proposal = proposal.detach()

        scale = self.proposal_coordinate_scale.to(
            device=proposal.device, dtype=proposal.dtype
        )
        tokens = torch.cat(
            [
                current - reference,
                0.5 * (current + reference),
                (proposal / scale).unsqueeze(2),
            ],
            dim=2,
        )
        logits = self.output(torch.nn.functional.silu(self.hidden(tokens))).squeeze(2)
        return torch.clamp((logits + 3.0) / 6.0, min=0.0, max=1.0)


class ClusterContextualP12DeltaConfidenceHead(nn.Module):
    """Cluster-conditioned, position-free confidence for a frozen proposal.

    Every p12 confidence is inferred from shared token processing and dynamic
    statistics of the current origin.  Cluster labels define only which
    channels participate in those statistics: there are no cluster embeddings,
    cluster-owned weights, channel tables, or absolute-position parameters.
    Consequently relabeling clusters is a no-op and synchronously permuting
    channels and their labels simply permutes the output channels.

    The final row and local heads are zero-initialized.  Before fitting, every
    row is therefore exactly the position-uniform confidence ``0.5``.  All
    proposal-derived inputs are unconditionally detached; this model can only
    suppress the separately fitted signed proposal.
    """

    _TOKEN_WIDTH = 65
    _HIDDEN_WIDTH = 32
    _DILATIONS = (1, 2, 4, 8)

    def __init__(self, proposal_coordinate_scale: float = 1.0) -> None:
        super().__init__()
        proposal_coordinate_scale = float(proposal_coordinate_scale)
        if (
            not math.isfinite(proposal_coordinate_scale)
            or proposal_coordinate_scale <= 0.0
        ):
            raise ValueError("proposal_coordinate_scale must be finite and positive")
        self.register_buffer(
            "proposal_coordinate_scale",
            torch.tensor(proposal_coordinate_scale, dtype=torch.float32),
        )

        self.token_norm = nn.LayerNorm(self._TOKEN_WIDTH)
        # LayerNorm already supplies an affine offset.  Omitting the redundant
        # projection bias keeps the fixed architecture at 23,972 parameters.
        self.token_projection = nn.Linear(
            self._TOKEN_WIDTH, self._HIDDEN_WIDTH, bias=False
        )
        self.dilated_convs = nn.ModuleList(
            nn.Conv1d(
                self._HIDDEN_WIDTH,
                self._HIDDEN_WIDTH,
                kernel_size=3,
                dilation=dilation,
            )
            for dilation in self._DILATIONS
        )

        self.row_hidden = nn.Linear(4 * self._HIDDEN_WIDTH, self._HIDDEN_WIDTH)
        self.row_output = nn.Linear(self._HIDDEN_WIDTH, 1)
        self.local_hidden = nn.Linear(5 * self._HIDDEN_WIDTH, self._HIDDEN_WIDTH)
        self.local_output = nn.Linear(self._HIDDEN_WIDTH, 1)
        nn.init.zeros_(self.row_output.weight)
        nn.init.zeros_(self.row_output.bias)
        nn.init.zeros_(self.local_output.weight)
        nn.init.zeros_(self.local_output.bias)

    @staticmethod
    def _stable_rms(values: torch.Tensor, *, dim: int | tuple[int, ...]) -> torch.Tensor:
        floor = torch.finfo(values.dtype).eps
        return values.square().mean(dim=dim).clamp_min(floor).sqrt()

    @staticmethod
    def _validate_inputs(
        current_patch_embeddings: torch.Tensor,
        reference_patch_embeddings: torch.Tensor,
        raw_proposal_coordinates: torch.Tensor,
        cluster_ids: torch.Tensor,
    ) -> tuple[int, int, int]:
        if current_patch_embeddings.ndim != 4:
            raise ValueError("current patch embeddings must be [B,C,K,32]")
        if current_patch_embeddings.shape != reference_patch_embeddings.shape:
            raise ValueError("current/reference patch embeddings must have equal shape")
        batch, channels, p12_count, width = current_patch_embeddings.shape
        if width != 32:
            raise ValueError("current/reference patch embeddings must have width 32")
        if batch <= 0 or channels <= 0 or p12_count <= 0:
            raise ValueError("batch, channel, and p12 dimensions must be positive")
        if raw_proposal_coordinates.shape != (batch, channels, p12_count):
            raise ValueError("raw proposal coordinates must be [B,C,K]")
        if cluster_ids.ndim != 1 or cluster_ids.shape[0] != channels:
            raise ValueError("cluster_ids must be [C]")
        if cluster_ids.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise ValueError("cluster_ids must contain integer labels")
        return batch, channels, p12_count

    def forward(
        self,
        current_patch_embeddings: torch.Tensor,
        reference_patch_embeddings: torch.Tensor,
        raw_proposal_coordinates: torch.Tensor,
        cluster_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, p12_count = self._validate_inputs(
            current_patch_embeddings,
            reference_patch_embeddings,
            raw_proposal_coordinates,
            cluster_ids,
        )

        # These tensors are produced by the independently fitted proposal and
        # are never part of the confidence optimizer's gradient graph.
        current = current_patch_embeddings.detach()
        reference = reference_patch_embeddings.detach()
        proposal = raw_proposal_coordinates.detach()
        scale = self.proposal_coordinate_scale.to(
            device=proposal.device, dtype=proposal.dtype
        )
        tokens = torch.cat(
            [
                current - reference,
                0.5 * (current + reference),
                (proposal / scale).unsqueeze(3),
            ],
            dim=3,
        )
        hidden = torch.nn.functional.silu(
            self.token_projection(self.token_norm(tokens))
        )

        # Replicate padding prevents the boundary from acting as an implicit
        # absolute-position code: a constant token sequence stays constant.
        sequence = hidden.reshape(
            batch * channels, p12_count, self._HIDDEN_WIDTH
        ).transpose(1, 2)
        for convolution, dilation in zip(self.dilated_convs, self._DILATIONS):
            padded = torch.nn.functional.pad(
                sequence, (dilation, dilation), mode="replicate"
            )
            sequence = sequence + torch.nn.functional.silu(convolution(padded))
        hidden = sequence.transpose(1, 2).reshape(
            batch, channels, p12_count, self._HIDDEN_WIDTH
        )

        row_mean = hidden.mean(dim=2)
        row_rms = self._stable_rms(hidden - row_mean.unsqueeze(2), dim=2)
        row_context = torch.cat([row_mean, row_rms], dim=2)

        labels = cluster_ids.detach().to(device=hidden.device)
        membership = labels.unsqueeze(1).eq(labels.unsqueeze(0))
        cluster_weights = membership.to(dtype=hidden.dtype)
        cluster_weights = cluster_weights / cluster_weights.sum(
            dim=1, keepdim=True
        )

        cluster_token_mean = torch.einsum(
            "cd,bdkh->bckh", cluster_weights, hidden
        )
        cluster_token_second = torch.einsum(
            "cd,bdkh->bckh", cluster_weights, hidden.square()
        )
        cluster_token_variance = (
            cluster_token_second - cluster_token_mean.square()
        ).clamp_min(torch.finfo(hidden.dtype).eps)
        cluster_token_rms = cluster_token_variance.sqrt()
        cluster_token_context = torch.cat(
            [cluster_token_mean, cluster_token_rms], dim=3
        )

        cluster_global_mean = cluster_token_mean.mean(dim=2)
        cluster_global_rms = cluster_token_second.mean(dim=2).clamp_min(
            torch.finfo(hidden.dtype).eps
        ).sqrt()
        cluster_global_context = torch.cat(
            [cluster_global_mean, cluster_global_rms], dim=2
        )

        row_logits = self.row_output(
            torch.nn.functional.silu(
                self.row_hidden(
                    torch.cat([row_context, cluster_global_context], dim=2)
                )
            )
        ).squeeze(2)
        local_logits = self.local_output(
            torch.nn.functional.silu(
                self.local_hidden(
                    torch.cat(
                        [
                            hidden,
                            row_context.unsqueeze(2).expand(-1, -1, p12_count, -1),
                            cluster_token_context,
                        ],
                        dim=3,
                    )
                )
            )
        ).squeeze(3)
        centered_local_logits = local_logits - local_logits.mean(dim=2, keepdim=True)
        logits = row_logits.unsqueeze(2) + centered_local_logits
        return torch.clamp((logits + 3.0) / 6.0, min=0.0, max=1.0)


def decode_confidence_weighted_raw_delta_coordinates(
    raw_proposal_coordinates: torch.Tensor,
    confidence: torch.Tensor,
    geometry: DeltaGeometry,
    *,
    stop_gradient_proposal: bool = True,
) -> torch.Tensor:
    """Suppress and decode a frozen signed proposal in the exact Delta span."""

    if raw_proposal_coordinates.ndim != 2:
        raise ValueError("raw proposal coordinates must be [B,K]")
    if confidence.shape != raw_proposal_coordinates.shape:
        raise ValueError("confidence must match raw proposal coordinates")
    if not bool(torch.isfinite(confidence).all()):
        raise ValueError("confidence must be finite")
    if bool(torch.any(confidence < 0.0)) or bool(torch.any(confidence > 1.0)):
        raise ValueError("confidence must lie in [0,1]")
    proposal = (
        raw_proposal_coordinates.detach()
        if stop_gradient_proposal
        else raw_proposal_coordinates
    )
    return decode_raw_delta_coordinates(proposal * confidence, geometry)


def confidence_weighted_delta_action(
    confidence_head: SharedP12DeltaConfidenceHead,
    current_patch_embeddings: torch.Tensor,
    reference_patch_embeddings: torch.Tensor,
    raw_proposal_coordinates: torch.Tensor,
    geometry: DeltaGeometry,
    *,
    stop_gradient_proposal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the confidence-weighted Delta action and its local confidence."""

    confidence = confidence_head(
        current_patch_embeddings,
        reference_patch_embeddings,
        raw_proposal_coordinates,
        stop_gradient_proposal=stop_gradient_proposal,
    )
    action = decode_confidence_weighted_raw_delta_coordinates(
        raw_proposal_coordinates,
        confidence,
        geometry,
        stop_gradient_proposal=stop_gradient_proposal,
    )
    return action, confidence


__all__ = [
    "CausalNullPairedDeltaAdapter",
    "ClusterContextualSignedP12DeltaAdapter",
    "ClusterContextualP12DeltaConfidenceHead",
    "DeltaGeometry",
    "PointSafeCumulativeDeltaAdapter",
    "SharedP12DeltaConfidenceHead",
    "build_cumulative_p12_delta_geometry",
    "confidence_weighted_delta_action",
    "decode_confidence_weighted_raw_delta_coordinates",
    "decode_raw_delta_coordinates",
    "orthonormal_to_raw_coordinates",
    "point_safe_delta_qcqp",
    "raw_to_orthonormal_coordinates",
]
