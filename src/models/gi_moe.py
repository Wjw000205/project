"""
GI-MoE (Gradient-Isolated MoE Loss) — v1 Adapter + v2 Hidden-Block, side-by-side.

Both versions share a strict design discipline:
  - Single backward pass per step.
  - Gradient isolation via detach trick only — NO hooks, NO multi-backward,
    NO neuron-level gradient routing.
  - MSE/MAE updates the FULL model.
  - Each penalty_p only updates ITS OWN private parameters.

==========================================================================
v1: Adapter form
==========================================================================
  r_p = Adapter_p(h)              # Linear -> GELU -> Linear, out [B,C,H]
  g_p = sigmoid(Gate_p(h))        # same shape, [B,C,H]
  y_final = y_base + Σ_p g_p · r_p

  Penalty path:
    visible_p = g_p.detach() · r_p
    y_view_p  = y_final.detach() + visible_p - visible_p.detach()
    L_pen_p   = penalty_p(y_view_p, y).mean()

==========================================================================
v2: Hidden-Block form
==========================================================================
  z_shared        = SharedProj(h)                 # [B,C,shared_dim]
  z_p             = PrivateProj_p(h)              # [B,C,private_dim]
  a_p             = GELU(z_p) * sigmoid(mask_p)   # PenaltyGatedActivation
  r_p             = PrivateHead_p(a_p)            # [B,C,H]
  g_p             = sigmoid(Gate_p(h))            # [B,C,H]
  α_p             = sigmoid(log_alpha_p)          # scalar, init -3.0
  y_base          = BaseHead(z_shared)            # [B,C,H]
  y_final         = y_base + Σ_p g_p · α_p · r_p

  Penalty path:
    visible_p = (g_p · α_p).detach() · r_p
    y_view_p  = y_final.detach() + visible_p - visible_p.detach()
    L_pen_p   = penalty_p(y_view_p, y).mean()

  Extra reg:
    L_mask_budget = Σ_p (sigmoid(mask_p).mean() - target)^2

==========================================================================
Constraints (both versions):
  - No cluster, no dynamic prototype, no KNN, no neuron-mask gradient hook.
  - h: [B, C, hidden_dim]; r_p / g_p / y_final: [B, C, H].
  - g_p / α_p must be detached in penalty path.
  - y_final must be detached in penalty path.
==========================================================================
"""
from typing import Dict, List, Optional, Tuple, Callable
import torch
from torch import nn
import torch.nn.functional as F


# =========================================================================
# Base predictor (no cluster). Returns (y_base, h) with h=[B,C,D].
# =========================================================================
class ClusterMLPBaseWithFeatures(nn.Module):
    """
    Stronger base: wraps production ClusterwiseMLP (per-cluster two-layer MLP).
    Each channel routes through its cluster's own (W1, b1, W2, b2). Exposes
    encode/decode separately so HiddenBlockMoEHead can attach branches in h
    space while keeping the strong base.decode for y_base.

    forward(x, return_features=True) -> (y, h) with h [B,C,D].
    """

    def __init__(self, num_clusters: int, input_len: int, pred_len: int,
                 hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        from .cluster_mlp import ClusterwiseMLP
        self.K = int(num_clusters)
        self.L = int(input_len)
        self.H = int(pred_len)
        self.D = int(hidden_dim)
        self.inner = ClusterwiseMLP(self.K, self.L, self.H, self.D, dropout)
        # NLinear-style "subtract last" trick, applied OUTSIDE ClusterwiseMLP
        # to stay consistent with SimpleBasePredictor's behavior.
        self._last_anchor = None

    def encode(self, x_bcl: torch.Tensor, cluster_id_c: torch.Tensor) -> torch.Tensor:
        # NLinear: subtract last value before encoding.
        last = x_bcl[..., -1:]
        x_c = x_bcl - last
        m = self.inner
        W1 = torch.stack(list(m.W1), dim=0).index_select(0, cluster_id_c)   # [C,L,D]
        b1 = torch.stack(list(m.b1), dim=0).index_select(0, cluster_id_c)   # [C,D]
        h = torch.einsum("bcl,cld->bcd", x_c, W1) + b1.unsqueeze(0)
        h = m.drop(m.act(h))
        self._last_anchor = last
        return h

    def decode(self, h: torch.Tensor, cluster_id_c: torch.Tensor,
               detach_weights: bool = False) -> torch.Tensor:
        m = self.inner
        W2 = torch.stack(list(m.W2), dim=0).index_select(0, cluster_id_c)   # [C,D,H]
        b2 = torch.stack(list(m.b2), dim=0).index_select(0, cluster_id_c)   # [C,H]
        if detach_weights:
            W2 = W2.detach()
            b2 = b2.detach()
        y = torch.einsum("bcd,cdh->bch", h, W2) + b2.unsqueeze(0)
        if self._last_anchor is not None:
            y = y + self._last_anchor
        return y

    def forward(self, x_bcl: torch.Tensor, cluster_id_c: Optional[torch.Tensor] = None,
                return_features: bool = False):
        if cluster_id_c is None:
            # Fallback: route everything to cluster 0.
            cluster_id_c = torch.zeros(x_bcl.shape[1], dtype=torch.long, device=x_bcl.device)
        h = self.encode(x_bcl, cluster_id_c)
        y = self.decode(h, cluster_id_c)
        if return_features:
            return y, h
        return y


class SimpleBasePredictor(nn.Module):
    """
    Two-layer MLP base predictor (per-channel sharing, NLinear-style).

    x: [B, C, L]
    encode(x) -> h: [B, C, D]
    decode(h) -> y: [B, C, H]
    forward(x, return_features=True) -> (y, h)
    """

    def __init__(self, input_len: int, pred_len: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.L = int(input_len)
        self.H = int(pred_len)
        self.D = int(hidden_dim)
        self.W1 = nn.Linear(self.L, self.D)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.W2 = nn.Linear(self.D, self.H)

    def encode(self, x_bcl: torch.Tensor) -> torch.Tensor:
        # NLinear trick: subtract last value, predict residual, add back.
        last = x_bcl[..., -1:]
        x_c = x_bcl - last
        h = self.drop(self.act(self.W1(x_c)))
        # stash last so decode can add it back
        self._last_anchor = last
        return h

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        y = self.W2(h)
        if getattr(self, "_last_anchor", None) is not None:
            y = y + self._last_anchor
        return y

    def forward(self, x_bcl: torch.Tensor, return_features: bool = False):
        h = self.encode(x_bcl)
        y = self.decode(h)
        if return_features:
            return y, h
        return y


# =========================================================================
# v1 Adapter:  Linear -> GELU -> Linear, output [B,C,H].
# =========================================================================
class PenaltyAdapter(nn.Module):
    """
    Adapter_p: h -> r_p [B,C,H]
    Gate_p:    (h, y_base.detach()) -> g_p [B,C,1]  (sigmoid; per-sample scalar)
    """

    def __init__(self, hidden_dim: int, output_dim: int, adapter_dim: int = 32,
                 gate_init_bias: float = -2.0):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(hidden_dim, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, output_dim),
        )
        # G1: gate input = concat(h, y_base) so it can reason about base's prediction.
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + output_dim, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, 1),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, float(gate_init_bias))

    def forward(self, h: torch.Tensor, y_base_detached: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r = self.adapter(h)                                    # [B,C,H]
        gate_input = torch.cat([h, y_base_detached], dim=-1)   # [B,C,D+H]
        g = torch.sigmoid(self.gate(gate_input))               # [B,C,1]
        return r, g


class PenaltyAdapterBank(nn.Module):
    """Bank of v1 adapters keyed by penalty name. h-in, output-level r/g out."""

    def __init__(self, penalty_names: List[str], hidden_dim: int, output_dim: int,
                 adapter_dim: int = 32, gate_init_bias: float = -2.0):
        super().__init__()
        if len(penalty_names) == 0:
            raise ValueError("PenaltyAdapterBank requires at least one penalty.")
        self.penalty_names = list(penalty_names)
        self.adapters = nn.ModuleDict(
            {p: PenaltyAdapter(hidden_dim, output_dim, adapter_dim, gate_init_bias=gate_init_bias)
             for p in self.penalty_names}
        )

    def forward(self, h: torch.Tensor, y_base: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        residuals: Dict[str, torch.Tensor] = {}
        gates: Dict[str, torch.Tensor] = {}
        y_base_sg = y_base.detach()
        for p, adp in self.adapters.items():
            r, g = adp(h, y_base_sg)
            residuals[p] = r
            gates[p] = g
        return residuals, gates

    def mix(self, y_base: torch.Tensor, residuals: Dict[str, torch.Tensor],
            gates: Dict[str, torch.Tensor]) -> torch.Tensor:
        y_final = y_base
        for p in self.penalty_names:
            y_final = y_final + gates[p] * residuals[p]
        return y_final

    def params_of(self, penalty: str) -> List[nn.Parameter]:
        return list(self.adapters[penalty].adapter.parameters())

    def gate_params_of(self, penalty: str) -> List[nn.Parameter]:
        return list(self.adapters[penalty].gate.parameters())


def gi_moe_loss(
    y_base: torch.Tensor,
    y_final: torch.Tensor,
    y: torch.Tensor,
    residuals: Dict[str, torch.Tensor],
    gates: Dict[str, torch.Tensor],
    penalty_fns: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    lambda_pen: float = 0.1,
    lambda_p: Optional[Dict[str, float]] = None,
    lambda_norm: float = 1.0e-4,
    mae_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """v1 GI-MoE Loss — masked-visibility detach trick at output level."""
    info: Dict[str, torch.Tensor] = {}

    # L_main: full model update via MSE/MAE on y_final.
    L_main = F.mse_loss(y_final, y)
    if mae_weight and mae_weight > 0.0:
        L_main = L_main + float(mae_weight) * F.l1_loss(y_final, y)
    info["L_main"] = L_main.detach()

    # L_pen_p: gradient flows ONLY to r_p (adapter_p).
    y_final_sg = y_final.detach()
    L_pen_total = y_base.new_zeros(())
    for p_name, fn in penalty_fns.items():
        if p_name not in residuals:
            continue
        r_p = residuals[p_name]
        g_p_sg = gates[p_name].detach()
        visible_p = g_p_sg * r_p
        y_view_p = y_final_sg + visible_p - visible_p.detach()
        L_pen_p = fn(y_view_p, y).mean()
        w_p = 1.0 if (lambda_p is None) else float(lambda_p.get(p_name, 1.0))
        L_pen_total = L_pen_total + w_p * L_pen_p
        info[f"L_pen_{p_name}"] = L_pen_p.detach()
    info["L_pen_total"] = L_pen_total.detach()

    # L_norm: keep y_final near y_base (no penalty leak to base).
    L_norm = F.mse_loss(y_final, y_base.detach())
    info["L_norm"] = L_norm.detach()

    with torch.no_grad():
        info["y_base_mse"] = F.mse_loss(y_base, y).detach()
        info["y_final_mse"] = F.mse_loss(y_final, y).detach()

    L_total = L_main + float(lambda_pen) * L_pen_total + float(lambda_norm) * L_norm
    return L_total, info


# =========================================================================
# v2 Hidden-Block:  shared block + per-penalty private blocks,
#                   penalty-gated activation, learnable α.
# =========================================================================
class PenaltyGatedActivation(nn.Module):
    """
    h_p = GELU(z) * sigmoid(mask_param)

    mask_param: nn.Parameter [private_dim], init 0 -> sigmoid=0.5.
    Owned by the private block of one penalty; updates only when that penalty's
    masked-visibility view fires.
    """

    def __init__(self, private_dim: int, mask_init: float = 0.0):
        super().__init__()
        self.mask_param = nn.Parameter(torch.full((int(private_dim),), float(mask_init)))

    @property
    def mask_value(self) -> torch.Tensor:
        return torch.sigmoid(self.mask_param)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.gelu(z) * self.mask_value


class _PrivateBlock(nn.Module):
    """One penalty's private branch:
       PrivateProj -> PenaltyGatedActivation -> PrivateHead   [B,C,H]
    Plus a per-sample SCALAR gate (sigmoid output [B,C,1]).

    Gate is intentionally scalar per (sample, channel) — it answers ONE
    binary-like question 'does this sample need penalty p?'. The shape [B,C,1]
    broadcasts when multiplied with r_p [B,C,H], so gate controls open/close
    intensity but does NOT shape r_p per horizon step. This was the original
    per-sample-selectivity intent — previous [B,C,H] gate gave 96 independent
    decisions per sample, defeating the selectivity semantics.
    """

    def __init__(self, in_dim: int, pred_len: int, private_dim: int = 32,
                 dropout: float = 0.0, mask_init: float = 0.0,
                 gate_init_bias: float = -2.0):
        super().__init__()
        self.proj = nn.Linear(int(in_dim), int(private_dim))
        self.pga = PenaltyGatedActivation(int(private_dim), mask_init=mask_init)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(int(private_dim), int(pred_len))
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        # G1 fix: gate sees BOTH h AND y_base. Input dim = in_dim + pred_len.
        # Gate now has the info to reason: "given what base is going to
        # predict, do I need penalty p to correct it?"
        self.gate_lin = nn.Linear(int(in_dim) + int(pred_len), 1)
        nn.init.zeros_(self.gate_lin.weight)
        nn.init.constant_(self.gate_lin.bias, float(gate_init_bias))

    def forward(self, h: torch.Tensor, y_base_detached: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        h: [B,C,in_dim]
        y_base_detached: [B,C,H]  must already be detached by caller to avoid
                                  routing gate's gradient back through base.
        """
        z = self.proj(h)
        a = self.pga(z)
        a = self.drop(a)
        r = self.head(a)                                            # [B,C,H]
        gate_input = torch.cat([h, y_base_detached], dim=-1)        # [B,C,in_dim+H]
        g = torch.sigmoid(self.gate_lin(gate_input))                # [B,C,1]
        return {"r": r, "g": g, "z": z, "a": a}


class HiddenBlockMoEHead(nn.Module):
    """
    v2 head (FIXED, post-B1): takes base hidden h: [B,C,in_dim] and emits ONLY
    penalty-private branches. y_base is produced by base.decode(h) outside this
    head (so head does NOT duplicate / weaken the base predictor).

    Per penalty p emits:
        r_p [B,C,H]       (private branch output)
        g_p [B,C,H]       (sigmoid gate)
        mask_value_p      (scalar penalty mask via PGA)
    Plus a global scalar alpha_p = sigmoid(log_alpha_p).

    Caller composes:
        y_final = y_base + Σ_p g_p · α_p · r_p
    where y_base comes from base.decode(h) (the STRONG base predictor).

    `shared_dim` is now ignored (kept for backward config compatibility).
    """

    def __init__(self, in_dim: int, pred_len: int,
                 penalty_names: List[str],
                 shared_dim: int = 128,     # kept for cfg compat; unused.
                 private_dim: int = 32,
                 dropout: float = 0.0, mask_init: float = 0.0,
                 log_alpha_init: float = -3.0,
                 gate_init_bias: float = -2.0,
                 use_pga: bool = True):
        super().__init__()
        if len(penalty_names) == 0:
            raise ValueError("HiddenBlockMoEHead requires at least one penalty.")
        self.penalty_names = list(penalty_names)
        self.in_dim = int(in_dim)
        self.H = int(pred_len)
        self.private_dim = int(private_dim)
        self.use_pga = bool(use_pga)

        # Per-penalty private blocks ONLY. No shared_proj / base_head — y_base
        # comes from the real base predictor outside this head.
        self.private = nn.ModuleDict(
            {p: _PrivateBlock(in_dim=self.in_dim, pred_len=self.H,
                              private_dim=self.private_dim, dropout=dropout,
                              mask_init=mask_init, gate_init_bias=gate_init_bias)
             for p in self.penalty_names}
        )
        if not self.use_pga:
            for p in self.penalty_names:
                self.private[p].pga.mask_param.requires_grad = False
                self.private[p].pga.mask_param.data.fill_(20.0)

        # Per-penalty log_alpha scalar.
        self.log_alpha = nn.ParameterDict(
            {p: nn.Parameter(torch.tensor(float(log_alpha_init)))
             for p in self.penalty_names}
        )

        # --- Penalty EMA normalizer ---
        # Each penalty's raw value lives on a very different scale (trend ≈ 0.4,
        # delta ≈ 0.05, etc.). To let lambda_p purely encode IMPORTANCE rather
        # than scale, normalize by an EMA of the raw value first:
        #     L_pen_p_normalized = L_pen_p_raw / EMA[L_pen_p_raw]
        # After this, mean(L_pen_p_normalized) ≈ 1 across penalties, and
        # lambda_p={delta:1, trend:1} truly means equal importance.
        self.pen_ema_momentum = 0.99
        for p in self.penalty_names:
            self.register_buffer(f"_pen_ema_{p}", torch.tensor(1.0))

    def normalize_penalty(self, name: str, raw_value: torch.Tensor) -> torch.Tensor:
        """Divide raw penalty by its running EMA and update EMA (training only).

        Returns the NORMALIZED value (~1.0 in expectation).
        Caller should multiply by lambda_p for the final weighted contribution.
        """
        buf = getattr(self, f"_pen_ema_{name}")
        normalized = raw_value / buf.clamp_min(1e-6)
        if self.training:
            with torch.no_grad():
                new_buf = self.pen_ema_momentum * buf + (1.0 - self.pen_ema_momentum) * raw_value.detach()
                buf.copy_(new_buf)
        return normalized

    def forward(self, h: torch.Tensor,
                y_base: Optional[torch.Tensor] = None,
                last_anchor: Optional[torch.Tensor] = None,
                ) -> Dict[str, object]:
        """
        h:      [B, C, in_dim]   features from base.encode
        y_base: [B, C, H]        REQUIRED — pass base.decode(h) explicitly.
                                 Head no longer produces its own y_base.
        last_anchor: ignored (kept for backward call compat).

        Returns dict with y_base (passthrough), y_final, residuals, gates,
        alphas, branches, mask_values.
        """
        if y_base is None:
            raise ValueError(
                "HiddenBlockMoEHead.forward now requires y_base from base.decode(h). "
                "Head does NOT produce its own y_base."
            )

        residuals: Dict[str, torch.Tensor] = {}
        gates: Dict[str, torch.Tensor] = {}
        branches: Dict[str, torch.Tensor] = {}
        alphas: Dict[str, torch.Tensor] = {}
        mask_values: Dict[str, torch.Tensor] = {}

        # Detach y_base when feeding into gates so that gate gradient doesn't
        # propagate back through base predictor.
        y_base_sg = y_base.detach()
        y_final = y_base
        for p in self.penalty_names:
            out = self.private[p](h, y_base_sg)                     # G1: gate sees y_base
            r_p = out["r"]
            g_p = out["g"]
            alpha_p = torch.sigmoid(self.log_alpha[p])
            branch_p = g_p * alpha_p * r_p
            y_final = y_final + branch_p
            residuals[p] = r_p
            gates[p] = g_p
            branches[p] = branch_p
            alphas[p] = alpha_p
            mask_values[p] = self.private[p].pga.mask_value

        return {
            "y_base": y_base, "y_final": y_final,
            "residuals": residuals, "gates": gates,
            "branches": branches, "alphas": alphas,
            "mask_values": mask_values,
        }

    # --- introspection helpers for verify / logging ---
    def private_params(self, penalty: str) -> List[nn.Parameter]:
        blk = self.private[penalty]
        params: List[nn.Parameter] = []
        params += list(blk.proj.parameters())
        params += list(blk.head.parameters())
        params.append(blk.pga.mask_param)
        return params

    def gate_params(self, penalty: str) -> List[nn.Parameter]:
        return list(self.private[penalty].gate_lin.parameters())

    def alpha_param(self, penalty: str) -> nn.Parameter:
        return self.log_alpha[penalty]

    def shared_params(self) -> List[nn.Parameter]:
        # No shared params anymore — return empty for verify compat.
        return []


def gi_moe_loss_v2(
    y_base: torch.Tensor,
    y_final: torch.Tensor,
    y: torch.Tensor,
    residuals: Dict[str, torch.Tensor],
    gates: Dict[str, torch.Tensor],
    alphas: Dict[str, torch.Tensor],
    penalty_fns: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    mask_values: Optional[Dict[str, torch.Tensor]] = None,
    lambda_pen: float = 0.1,
    lambda_p: Optional[Dict[str, float]] = None,
    lambda_norm: float = 1.0e-4,
    mae_weight: float = 0.3,
    lambda_mask: float = 1.0e-4,
    mask_target: float = 0.5,
    # --- per-sample gate supervision via improve_p ---
    bce_gate_supervision: bool = False,
    lambda_gate: float = 0.0,
    bce_tau: float = 0.01,
    detach_gate_from_main: bool = False,
    # --- G3: bimodal entropy reg on gate (pushes g toward 0 or 1) ---
    lambda_gate_bimodal: float = 0.0,
    # --- penalty normalization via head's EMA tracker ---
    head: Optional["HiddenBlockMoEHead"] = None,
    normalize_penalties: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """v2 GI-MoE Loss. Output-level masked-visibility with gate*alpha both detached.

    Optional `bce_gate_supervision`: adds a per-sample supervision signal for the
    gate. For each penalty p and each (b, c):
        improve_p(b,c) = MSE(y_base, y)[b,c] - MSE(y_base + α_p · r_p, y)[b,c]   (stopgrad)
        q_p(b,c)        = sigmoid(improve_p / τ).detach()
        L_gate_p        = BCE(g_p.mean(-1), q_p)
    Gate then learns to fire HIGH on samples where penalty p actually reduces
    MSE, LOW elsewhere — explicit per-sample selectivity.

    `detach_gate_from_main=True` additionally prevents L_main from updating gate
    params (recomputes y_final with gates detached). Lets BCE be the gate's sole
    supervisor — cleaner separation but bigger structural change.
    """
    info: Dict[str, torch.Tensor] = {}

    # --- Optionally rebuild y_final_for_main with detached gates ---
    if detach_gate_from_main:
        y_main = y_base
        for p in residuals:
            y_main = y_main + gates[p].detach() * alphas[p] * residuals[p]
    else:
        y_main = y_final

    L_main = F.mse_loss(y_main, y)
    if mae_weight and mae_weight > 0.0:
        L_main = L_main + float(mae_weight) * F.l1_loss(y_main, y)
    info["L_main"] = L_main.detach()

    y_final_sg = y_final.detach()
    L_pen_total = y_base.new_zeros(())
    for p_name, fn in penalty_fns.items():
        if p_name not in residuals:
            continue
        r_p = residuals[p_name]
        g_alpha_sg = (gates[p_name] * alphas[p_name]).detach()
        visible_p = g_alpha_sg * r_p
        y_view_p = y_final_sg + visible_p - visible_p.detach()
        L_pen_p_raw = fn(y_view_p, y).mean()
        # Internal EMA normalization (scale-equalization). After this,
        # lambda_p purely encodes IMPORTANCE rather than scale.
        if normalize_penalties and head is not None:
            L_pen_p = head.normalize_penalty(p_name, L_pen_p_raw)
        else:
            L_pen_p = L_pen_p_raw
        w_p = 1.0 if (lambda_p is None) else float(lambda_p.get(p_name, 1.0))
        L_pen_total = L_pen_total + w_p * L_pen_p
        info[f"L_pen_{p_name}_raw"] = L_pen_p_raw.detach()
        info[f"L_pen_{p_name}"] = L_pen_p.detach()
    info["L_pen_total"] = L_pen_total.detach()

    L_norm = F.mse_loss(y_final, y_base.detach())
    info["L_norm"] = L_norm.detach()

    L_mask_budget = y_base.new_zeros(())
    if mask_values is not None and lambda_mask and lambda_mask > 0.0:
        for p_name, m in mask_values.items():
            L_mask_budget = L_mask_budget + (m.mean() - float(mask_target)).pow(2)
    info["L_mask_budget"] = L_mask_budget.detach()

    # --- L_gate: per-sample BCE supervision via improve_p ---
    L_gate_total = y_base.new_zeros(())
    if bce_gate_supervision and lambda_gate and lambda_gate > 0.0:
        with torch.no_grad():
            mse_base_bc = (y_base - y).pow(2).mean(dim=-1)                # [B, C]
        for p_name in residuals:
            r_p = residuals[p_name]
            alpha_p = alphas[p_name]
            # improve_p: per-sample MSE delta from adding ONLY penalty p's branch
            with torch.no_grad():
                y_with_p = y_base + alpha_p * r_p                          # [B,C,H]
                mse_with_bc = (y_with_p - y).pow(2).mean(dim=-1)           # [B, C]
                improve_p = (mse_base_bc - mse_with_bc).detach()           # [B, C]
                q_p = torch.sigmoid(improve_p / float(bce_tau)).detach()   # [B, C] in (0,1)
            # Gate is now per-sample [B,C,1]; squeeze to match q_p [B,C].
            g_p_bc = gates[p_name].squeeze(-1).clamp(1e-6, 1.0 - 1e-6)      # [B, C]
            L_gate_p = F.binary_cross_entropy(g_p_bc, q_p)
            L_gate_total = L_gate_total + L_gate_p
            info[f"L_gate_{p_name}"] = L_gate_p.detach()
            info[f"q_mean_{p_name}"] = q_p.mean().detach()
    info["L_gate_total"] = L_gate_total.detach()

    with torch.no_grad():
        info["y_base_mse"] = F.mse_loss(y_base, y).detach()
        info["y_final_mse"] = F.mse_loss(y_final, y).detach()

    # --- L_gate_bimodal: push each gate toward 0 or 1 (binary decision) ---
    # Minimizing entropy of Bernoulli(g) pushes g toward 0 or 1.
    L_gate_bimodal = y_base.new_zeros(())
    if lambda_gate_bimodal and lambda_gate_bimodal > 0.0:
        for p_name in residuals:
            g = gates[p_name].clamp(1e-6, 1.0 - 1e-6)
            # Per-sample Bernoulli entropy: H = -g·log(g) - (1-g)·log(1-g) in [0, log 2].
            ent = -(g * g.log() + (1.0 - g) * (1.0 - g).log())
            L_gate_bimodal = L_gate_bimodal + ent.mean()
    info["L_gate_bimodal"] = L_gate_bimodal.detach()

    L_total = (L_main
               + float(lambda_pen) * L_pen_total
               + float(lambda_norm) * L_norm
               + float(lambda_mask) * L_mask_budget
               + float(lambda_gate) * L_gate_total
               + float(lambda_gate_bimodal) * L_gate_bimodal)
    return L_total, info


# =========================================================================
# Ablation losses (output-level penalty WITHOUT GI for comparison).
# =========================================================================
def loss_mse_only(y_final: torch.Tensor, y: torch.Tensor,
                  mae_weight: float = 0.0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    L = F.mse_loss(y_final, y)
    if mae_weight and mae_weight > 0.0:
        L = L + float(mae_weight) * F.l1_loss(y_final, y)
    return L, {"L_main": L.detach()}


def loss_ordinary_penalty(
    y_final: torch.Tensor, y: torch.Tensor,
    penalty_fns: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    lambda_pen: float = 0.1,
    lambda_p: Optional[Dict[str, float]] = None,
    mae_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Negative control: penalty acts directly on y_final, no isolation."""
    L_main = F.mse_loss(y_final, y)
    if mae_weight and mae_weight > 0.0:
        L_main = L_main + float(mae_weight) * F.l1_loss(y_final, y)
    L_pen_total = y_final.new_zeros(())
    info: Dict[str, torch.Tensor] = {"L_main": L_main.detach()}
    for p_name, fn in penalty_fns.items():
        L_pen_p = fn(y_final, y).mean()
        w_p = 1.0 if (lambda_p is None) else float(lambda_p.get(p_name, 1.0))
        L_pen_total = L_pen_total + w_p * L_pen_p
        info[f"L_pen_{p_name}"] = L_pen_p.detach()
    info["L_pen_total"] = L_pen_total.detach()
    return L_main + float(lambda_pen) * L_pen_total, info


# =========================================================================
# Grad-isolation verifiers.
# =========================================================================
def _grad_norm(params: List[nn.Parameter]) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum().item())
    return total ** 0.5


def verify_gi_moe_grad_isolation(
    base_model: nn.Module,
    bank: PenaltyAdapterBank,
    x: torch.Tensor,
    y: torch.Tensor,
    penalty_fns: Dict[str, Callable],
    target_penalty: str,
    tol: float = 1.0e-8,
    verbose: bool = True,
) -> Dict[str, float]:
    """v1: only adapter_{target_penalty} should receive gradient."""
    if target_penalty not in bank.penalty_names:
        raise ValueError(f"target_penalty {target_penalty!r} not in bank.")
    base_model.zero_grad(set_to_none=True)
    bank.zero_grad(set_to_none=True)

    y_base, h = base_model(x, return_features=True)
    residuals, gates = bank(h, y_base)
    y_final = bank.mix(y_base, residuals, gates)

    y_final_sg = y_final.detach()
    r_p = residuals[target_penalty]
    g_p_sg = gates[target_penalty].detach()
    visible_p = g_p_sg * r_p
    y_view_p = y_final_sg + visible_p - visible_p.detach()
    L_pen_p = penalty_fns[target_penalty](y_view_p, y).mean()

    L_pen_p.backward()

    report: Dict[str, float] = {}
    report[f"adapter_{target_penalty}"] = _grad_norm(bank.params_of(target_penalty))
    for p in bank.penalty_names:
        if p != target_penalty:
            report[f"adapter_{p}"] = _grad_norm(bank.params_of(p))
        report[f"gate_{p}"] = _grad_norm(bank.gate_params_of(p))
    report["base_model"] = _grad_norm(list(base_model.parameters()))

    if verbose:
        print(f"[verify v1] target={target_penalty}")
        for k, v in report.items():
            ok = (k == f"adapter_{target_penalty}" and v > tol) or \
                 (k != f"adapter_{target_penalty}" and v <= tol)
            print(f"  {'OK ' if ok else '!! '}{k:32s} grad_norm={v:.3e}")

    if report[f"adapter_{target_penalty}"] <= tol:
        raise AssertionError(f"v1 verify FAIL: adapter_{target_penalty} has zero grad.")
    for k, v in report.items():
        if k != f"adapter_{target_penalty}" and v > tol:
            raise AssertionError(f"v1 verify FAIL: leak to {k} ({v:.3e}).")
    return report


def verify_gi_moe_v2_grad_isolation(
    base_model: nn.Module,
    head: HiddenBlockMoEHead,
    x: torch.Tensor,
    y: torch.Tensor,
    penalty_fns: Dict[str, Callable],
    target_penalty: str,
    tol: float = 1.0e-8,
    verbose: bool = True,
) -> Dict[str, float]:
    """v2: only private_{target_penalty} (proj, head, mask_param) should receive gradient."""
    if target_penalty not in head.penalty_names:
        raise ValueError(f"target_penalty {target_penalty!r} not in head.")
    base_model.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)

    y_base, h = base_model(x, return_features=True)
    out = head(h, y_base=y_base)
    y_base = out["y_base"]; y_final = out["y_final"]
    residuals = out["residuals"]; gates = out["gates"]; alphas = out["alphas"]

    y_final_sg = y_final.detach()
    r_p = residuals[target_penalty]
    g_alpha_sg = (gates[target_penalty] * alphas[target_penalty]).detach()
    visible_p = g_alpha_sg * r_p
    y_view_p = y_final_sg + visible_p - visible_p.detach()
    L_pen_p = penalty_fns[target_penalty](y_view_p, y).mean()

    L_pen_p.backward()

    report: Dict[str, float] = {}
    report[f"private_{target_penalty}"] = _grad_norm(head.private_params(target_penalty))
    for p in head.penalty_names:
        if p != target_penalty:
            report[f"private_{p}"] = _grad_norm(head.private_params(p))
        report[f"gate_{p}"] = _grad_norm(head.gate_params(p))
        ap = head.alpha_param(p)
        report[f"alpha_{p}"] = float(ap.grad.detach().pow(2).sum().item() ** 0.5) if ap.grad is not None else 0.0
    report["shared_block"] = _grad_norm(head.shared_params())
    report["base_model"] = _grad_norm(list(base_model.parameters()))

    if verbose:
        print(f"[verify v2] target={target_penalty}")
        for k, v in report.items():
            ok = (k == f"private_{target_penalty}" and v > tol) or \
                 (k != f"private_{target_penalty}" and v <= tol)
            print(f"  {'OK ' if ok else '!! '}{k:32s} grad_norm={v:.3e}")

    if report[f"private_{target_penalty}"] <= tol:
        raise AssertionError(f"v2 verify FAIL: private_{target_penalty} has zero grad.")
    for k, v in report.items():
        if k != f"private_{target_penalty}" and v > tol:
            raise AssertionError(f"v2 verify FAIL: leak to {k} ({v:.3e}).")
    return report
