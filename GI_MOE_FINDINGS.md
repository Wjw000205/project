# GI-MoE Loss — Findings Summary

> Single-file research log capturing the experimental conclusions and
> mechanistic understanding reached on this codebase.
> Date: 2026-05-11.

## TL;DR

Given a sufficiently strong time-series predictor (production-grade
`ClusterwiseMLP K=3`) trained with MSE+MAE, the GI-MoE Loss framework — penalty-
private branches + per-sample gates + masked-visibility gradient isolation —
**does not improve test MSE** on the canonical ETT and Weather benchmarks. The
mechanism is sound (verify-grad passes, gates show per-sample selectivity, gate
becomes per-sample bimodal under G3), but the strong base predictor already
captures the structural failure modes that penalty supervision is designed to
correct.

On a **weaker** base (single shared MLP without per-cluster parameters), GI
provides a small but reproducible improvement (~−0.006 test_mse, paired 3/3
seeds). This is "rescuing weak base" rather than "structural enhancement of
strong base".

## Final Numbers (3-seed mean ± paired std)

| Dataset | Base form | moe_off | GI | Paired Δ |
|---|---|---|---|---|
| ETTm1 H=96 | SimpleBasePredictor K=1 | 0.3055 | 0.2996 | **−0.0059 ± 0.0025** ✓ |
| ETTm1 H=96 | ClusterwiseMLP K=3 | 0.2983 | 0.3078 | +0.0095 ± 0.0032 ✗ |
| ETTh1 H=720 | ClusterwiseMLP K=3 | 0.6799 | 0.6851 | +0.0052 ± 0.0072 ✗ |
| Weather H=192 | ClusterwiseMLP K=4 | 0.1937 | 0.1935 | −0.0002 ± 0.0023 ≈ |

## Architecture Iterations (chronological)

1. **v1 Adapter** (`PenaltyAdapter` MLP outputting [B,C,H], output-level mix).
   Worked on weak base, unstable on strong base (r_p divergence).
2. **v2 LoRA on h** (rank-r perturbation of base's hidden state).
   Constrained to W2's column space; gave similar weak-base gain.
3. **v3 W2-LoRA per-cluster** (rank-r perturbation of per-cluster W2).
   Most flexible but did not improve on strong base.
4. **v2 Hidden-Block** (chosen canonical form):
   - Shared block produced y_base internally — **B1 fix**: replace with
     `base.decode(h)` so head only outputs branches.
   - Gate was `[B,C,H]` per-step — **B2 fix**: reshape to `[B,C,1]` per-sample
     scalar gate. gate_std jumps from 0.15 to 0.35.
   - Gate input was `h` only — **G1**: concat `(h, y_base.detach())` so gate
     can reason about base's prediction.
   - **G3** bimodal entropy reg pushes per-sample gate toward 0 or 1 (+0.007
     gain on weak base, but interacts badly with low penalty scale).

## Key Mechanism Findings

### Gate dynamics
- L_main decides **whether gate opens** (does branches improve y_final?).
- L_pen decides **what direction branches learn** (which penalty's metric to
  fit).
- On strong base, L_main says "branches add noise" → gate closes regardless
  of penalty pressure. Fix B (G3=0) + Fix A (split grad clip) prevents
  pathological closure, but gates still settle at moderate-to-low values
  because branches genuinely don't help MSE.

### Penalty selection
- Diagnostic probe (per-cluster `rel_gap`) identified `delta`, `direction`,
  `trend` as direction-correct, high-room penalties on ETTm1/ETTh1.
- Original `jitter` and `smooth` were reverse-direction (minimize-only,
  pulling toward flat) — confirmed by probe and ablation.
- New penalty `d2_match` added (truth-matching second-difference) —
  rel_gap=1.0 but interchangeable with `direction` in 3-pen pool.
- **Pool size 3 is the sweet spot**; 2 too few (0.306), 4 over-saturated
  (0.299), gate auto-closes redundant penalties.

### Penalty scale and lambda_p
- **Normalize (EMA scale division) DID NOT help**: raw scale variation
  encodes natural curriculum (penalty value decreases as base improves).
- Manual lambda_p tuning is fragile — combined with G3, deviating from
  `{1,1,1}` triggers gate collapse.
- Conclusion: rely on natural scale + raw `lambda_p={1,1,1}` for current
  3-pen pool.

### Strong vs Weak base contrast
- SimpleBasePredictor K=1 base test_mse=0.3055. GI delivers −0.006.
- ClusterwiseMLP K=3 base test_mse=0.2983. GI delivers +0.0095 (loses).
- The "0.0072" delta between K=1 and K=3 base is what cluster-specific
  parameters give; GI's "win" on K=1 is just providing some fraction of
  that same capacity via adapters.

## What the Project's Code Now Contains

### Canonical files
- `configs/gi_moe_ETTm1.yaml` — annotated canonical config (SimpleBasePredictor
  K=1, used for the weak-base GI win documented above).
- `configs/gi_moe_ETTm1_clusterbase_v2.yaml` — strong-base config (cluster_mlp
  K=3, split grad clip, G3=0). GI loses on this but it represents the
  honest production-base configuration.
- `configs/gi_moe_ETTh1_clusterbase.yaml`,
  `configs/gi_moe_weather.yaml` — equivalents for ETTh1 and Weather.

### Source modules
- `src/models/gi_moe.py` — v1 + v2 implementations side-by-side, gate fixes
  (B1+B2+G1+G3), EMA penalty normalization (off by default), BCE-improve
  gate supervision (optional).
- `src/models/penalties.py` — added `penalty_d2_match` for truth-matching
  second-difference.

### Diagnostic tools
- `scripts/diagnose_predictions.py` — per-sample / global Q1-Q5 diagnostic.
- `scripts/diagnose_cluster_gi.py` — per-cluster MoE / penalty / gate
  diagnostic comparing moe_off vs GI.
- `scripts/probe_penalty_per_cluster.py` — per-cluster penalty rel_gap probe.
- `scripts/probe_penalty_effectivity.py` — global penalty rel_gap probe.

### Runner
- `scripts/run_gi_moe.py` — 8-mode unified runner with `--ablation-all`,
  `--seed`, `--verify-grad`. Cluster pipeline auto-fits when
  `model.predictor: cluster_mlp`. Split grad clip per module.

## Honest Stop Point

The GI-MoE Loss design space has been thoroughly explored on this codebase:
- 4 architectural iterations
- 7+ different lambda_p / penalty / gate / clip configurations
- 3 datasets × 3-seed verifications
- Per-cluster, per-penalty, per-sample diagnostics

The honest, statistically-significant conclusion is: **MoE-style penalty-
private adapters cannot improve over production-grade ClusterwiseMLP K=3 on
ETT/Weather forecasting under MSE evaluation**. Future directions that might
unlock the framework's potential:

1. **Distributional evaluation** — evaluate models on predictive
   distribution metrics (CRPS, energy score) rather than MSE. GI's penalties
   describe distributional properties, so a distributional metric might
   credit them properly.
2. **Larger base predictor** that genuinely fails on structural modes
   (transformer-class with capacity to overfit train MSE but not generalize
   on amplitude/trend). Test if GI provides regularization signal.
3. **Multi-horizon supervision** — penalty applied differently across
   horizon positions, since base's failure mode varies with horizon depth.
## 2026-05-12 MoE Residual Follow-up

### Leakage fix
- The earlier ~18-20% ETTm1 MoE gain was invalid: routing used future `y`
  through penalty values. Routing now uses a target-free history proxy only.
- Channel clustering now fits on the train split by default.
- No-leak baseline after this fix: the original residual MoE only gave about
  +0.1% to +1.1% MSE on ETTm1, although MAE improved consistently.

### Prediction residual MoE changes
- Added `ClusterwisePredResidualMoE`: separate K x P residual experts. Penalty
  expert parameters are physically separate `ParameterList` entries, so
  different penalties do not share residual parameters.
- Added validation-time residual selection:
  - `val_mse_channel`: full residual only on channels where validation MSE
    improves.
  - `val_mse_scale`: per-channel scalar selected on validation.
  - `val_mse_gate`: a small target-free scale gate trained on validation
    features `(history, base forecast, residual candidate)`.
- `detach_routed_penalty_pred` and `intervention_enable` default to `false`;
  detach caused base-path collapse in ETTm1 experiments.

### Best no-leak ETTm1 result so far

MoE-only ablation protocol: KNN/calibration/plot/portrait disabled by
`scripts/run_moe_only_ablation.py`, compare same config `moe.enable=true` vs
`false`.

| Config | Epoch budget | moe_on MSE | moe_off MSE | Gain |
|---|---:|---:|---:|---:|
| `level_only + val_mse_gate` | 100, early-stop | 0.286161 | 0.293661 | 2.55% |
| `level_only + val_mse_channel` | 100, early-stop | 0.286764 | 0.293661 | 2.35% |
| `level_only + val_mse_scale(0..1.5)` | 100, early-stop | 0.286848 | 0.293661 | 2.32% |
| `level_only alpha_scale=1.0 + gate` | 100, early-stop | 0.289476 | 0.293661 | 1.42% |
| `amp_under/delta/jitter/smooth + gate` | 20 | 0.290515 | 0.293780 | 1.11% |

Best run directory:
`outputs/moe_only_ablation_ETTm1_level_only_gate_cuda100`.

### Current conclusion

The no-leak residual MoE is real and improves ETTm1 MSE, but the best verified
gain is 2.55%, still below the requested 3-5%. The useful penalty changed from
directional/amp penalties to `level`: ETTm1's remaining MSE is dominated by
horizon level/bias correction, especially large-error load channels plus OT.
