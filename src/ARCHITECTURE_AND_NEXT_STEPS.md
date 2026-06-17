# PKR-MoE — Architecture & Exploration Log (read this first if you're an agent continuing the work)

This file is the single source of truth for **what the model is**, **what has been
tried**, and **what to do next**. It is written so an agent can pick up and run
experiments without re-deriving context. Keep it updated when you finish an experiment.

## Agent operating rule

- Agents working in this repository must read this file before any non-trivial
  exploration, experiment, or code change.
- Keep this file updated after meaningful explorations: record reusable findings,
  verdicts, configs or commands, output paths, metrics when available, and the next
  recommended action.
- Do not churn this file for purely mechanical inspections that produce no durable
  project knowledge.

### Self-check rules (every experiment — don't just report absolute numbers)
1. **Always compare against the baseline's val**, not just absolute val. Read the
   original (pre-change) run's `val.{avg_mse,avg_mae}` and report the **Δ%**. An
   absolute number alone hides whether the change actually helped. (E.g. depth on
   PEMS H48/H96 shows up as −25% to −40% val vs the hid128/blocks0 baseline.)
2. **Backbone-first, stop early.** For any backbone change, run **backbone-alone**
   (`moe.enable: false`, `skip_test: true`) and check val vs baseline FIRST. If val
   doesn't clearly improve, **stop — do not attach MoE** (the MoE absorbs small
   gains; attaching wastes compute). Only attach MoE when backbone-alone val shows a
   real, structural improvement.
3. **Counter-intuitive signal → halt and record, do not self-decide.** If you see
   "val improves but test regresses", "the change makes it worse", a metric moving
   the opposite way than expected, or a result that contradicts a verdict in §6 —
   **stop, write it down here as an observation, and leave the call to the human**.
   Do not quietly pick the test-flattering option (that is leakage) or bury the anomaly.

Last major result: **PEMS H12-H96 depth rollout completed**: hid192 + 2 CCH
residual blocks gives large gains over the previous PKR-MoE table on every
PEMS H12/H24/H48/H96 cell and beats OLinear on MSE everywhere. MAE double-win
vs OLinear is not universal: clean on PEMS08 and partial on H12/H24, still
worse on PEMS04/07 and near-parity on PEMS03 (see Section 6). The open
hypothesis is that other datasets (Electricity especially) may also be
depth-limited (Section 7).

---

## 0. Environment & how to run

- Conda env: **`my_fram`** (`C:\Users\33932\.conda\envs\my_fram`, torch 2.5.1+cu121, CUDA).
  Activate with `source activate my_fram` before any run.
- Entrypoint: `python -m src.train --config <path-to.yaml>`
- Sweeps/drivers: `scripts/run_*.py`.
- The Bash tool's default shell lacks coreutils (no `grep`/`head`/`cat`) — use the
  Grep/Read/Glob tools or `python -c` for file inspection.
- Each run writes `run_summary.json` to `exp.out_dir` with `val.{avg_mse,avg_mae}`,
  `test.{avg_mse,avg_mae}`, `selected.*`. That's where you read results.

---

## 1. Architecture: two-stage

1. **Stage-1 backbone** (per-cluster predictor). Trained alone (`moe.enable: false`),
   then **frozen** and handed to stage 2 via `finetune.checkpoint_path`.
2. **Stage-2 PKR-MoE**: penalty-routed residual experts + gate + anchors + (optional)
   median calibration, trained on top of the **frozen** backbone (`moe.freeze_backbone: true`).

The backbone defines the central prediction; PKR-MoE adds gated residual corrections.
**Key consequence (proven, §6):** the MoE *absorbs small backbone changes* — a backbone
tweak only moves the final pipeline if it is **structural / large**. Micro-tweaks
(e.g. FiLM, ~0.2%) get washed out; depth (~25%) survives and the MoE still adds on top.

### Key files
- `src/train.py` — the whole training+eval pipeline (one giant function). Per-cluster
  Adam optimizers, per-cluster early-stop, eval loop, MoE wiring, calibration.
- `src/models/cluster_predictor.py` — `build_cluster_predictor(...)` + ~30 backbone
  variant classes (the "predictor zoo"). **This is where backbone architecture lives.**
- `src/models/cluster_mlp.py` — `ClusterwiseMLP` (the plain `mlp` predictor core).
- `src/models/moe_gate.py`, `src/models/residual_moe.py`, `src/models/penalties.py`,
  `src/models/gi_moe.py` — the MoE side (gate, residual experts, penalty functions).

---

## 2. Backbone (stage 1) — the predictor zoo

`build_cluster_predictor` (bottom of `cluster_predictor.py`, dispatch near line ~2693)
maps `model.predictor` → a class. Notable predictors:

| `model.predictor` | class | notes |
|---|---|---|
| `mlp` (default) | `ClusterwiseMLP` (`cluster_mlp.py`) | per-cluster 2-layer MLP, **channel-independent**, NLinear subtract-last. Used by ETT/ECL. |
| `context_channel_head_mlp` (a.k.a. **cch**) | `ClusterwiseContextChannelHeadMLP` | **cross-channel** + per-channel output heads. Used by **PEMS**. Has a DEPTH knob. |
| `channel_head_mlp`, `long_context_channel_head_mlp`, `seasonality_gated_channel_head_mlp` | … | other cross-channel variants |
| `attn_mlp`, `dlinear`, `channel_dlinear`, `patchtst`, `nbeats`, `tcn`, `gru`, `lstm`, `channel_lstm_mixer` | … | alternative backbones, all **time-domain** |

**There is NO frequency/spectral variant** (FITS/OLinear-style). That's the one
architectural family not implemented (see §7, low priority).

### 2a. Per-cluster optimizer constraint (IMPORTANT for any new param)
Training uses **one Adam optimizer per cluster**, optimizing only
`model.get_cluster_params(k)`. Any new learnable param MUST be **per-cluster** and
registered in `get_cluster_params(k)`, `mask_cluster_grads`, `get_cluster_state(k)`,
`load_cluster_state(k)` — otherwise it won't be trained / saved / frozen correctly.
(See how `cluster_embedding`/FiLM was integrated in `cluster_mlp.py` for the pattern.)

### 2b. The DEPTH knob (the high-value lever)
`context_channel_head_mlp` supports residual blocks (extra depth) via config:
```yaml
model:
  predictor: context_channel_head_mlp
  hidden_dim: 192
  context_channel_head_blocks: 2          # <-- DEPTH. default 0. THIS is the lever.
  context_channel_head_block_scale: 0.5   # optional
  context_channel_head_block_init: zero_out  # optional (zero_out | xavier)
```
Built at `cluster_predictor.py` ~line 2781. `ClusterwiseContextChannelHeadMLP`
(`_setup_context_residual_blocks`, `_apply_context_residual_blocks`) adds
per-cluster residual MLP blocks in hidden space. Zero new code needed.

---

## 3. Stage-2 PKR-MoE (brief)
- **Penalty pool** (`penalties.enabled`): shape-aware penalties (`amp_under, delta,
  diff_amp, direction, d2_match, level, trend, corr, range, ...`, see `penalties.py`).
  PEMS uses `[amp_under, delta, diff_amp, direction]` (volatility/spike-aware). The pool
  should be chosen by **train-residual diagnostics** ("对症", treat where the backbone
  fails), not blind tuning.
- **Gate** (`moe_gate.py`): per-cluster routing over penalties; `mse_utility_gate_supervision`.
- **Residual experts** (`residual_moe.py`): per `(cluster, penalty)` MLP producing gated residuals.
- **Anchors**: `train_stat_anchor_expert`, `train_residual_anchor_expert` (period-based,
  e.g. p288 for PEMS daily). Scale chosen by `scale_selection` (val-internal).
- **Calibration** (`calibration.*`, eval-only): median per-(channel,horizon) offset.
  `calibration.shrink_sweep: [...]` + `scripts/run_calibration_shrink_probe.py --single-run`
  evaluate many shrinks from ONE trained model. **Cheapest MAE lever, but ~exhausted on
  ETT (~0.6%). Keep OUT of the comparison table (val-label post-hoc; apply symmetrically
  to baselines if reported).**

---

## 4. How a run is wired (config layering)
- **Backbone-pretraining config**: `moe.enable: false`, backbone trainable, no
  `finetune` load. Produces a `best_checkpoint.pt`. Select by `train.selection_metric`.
- **MoE-stage config**: `moe.enable: true`, `moe.freeze_backbone: true`,
  `finetune: {enable: true, load_model: true, strict_model: true, checkpoint_path: <backbone ckpt>}`.
  The MoE config's `model:` section MUST match the backbone's architecture exactly
  (predictor, hidden_dim, `context_channel_head_blocks`, ...) or `strict_model` load fails.

Reference PEMS08-H96 configs:
- backbone: `outputs/codex_table_target_20260614/pems08_input96_backbone_cch_e36/configs/PEMS08/H96/final/pems_cch_h128_do0_l001_mse050_mae150_bs64_valmae.yaml`
  (predictor `context_channel_head_mlp`, hid 128, epochs 36, mse_w 0.5, mae_w 1.5,
   penalties `amp_under/delta/diff_amp/direction`, `selection_metric: val_mae`, lazy windows).
- MoE: `outputs/codex_table_target_20260614/pems08_input96_moe_from_cch_freeze_p288_h96/configs/PEMS08/H96/moe_activation/trainstatresid_mean_p288_stat020_resid120_seg4.yaml`.

---

## 5. Methodology discipline (MUST follow — non-negotiable)
1. **Select on val, read test once.** Never use test to decide which config/variant to keep.
   Adoption rule for an MAE-leaning change: keep val MSE regression ≤ ~0.3–0.5% while
   improving val MAE. (MSE is the primary metric.)
2. **Backbone changes: test backbone-ALONE first** (does the structural gain exist on val?),
   then attach MoE (does it survive the MoE?). A gain must be large to survive (§1).
3. **Default-OFF + bit-exact equivalence** for any code feature, so the comparison table's
   numbers are never disturbed. The table is the publishable floor; protect it.
4. New code features that are NULL so far (don't re-chase): cluster-embedding/FiLM,
   per-cluster MAE weight, full-pipeline-residual calibration, cross-channel-without-depth.

---

## 6. Exploration findings (verdicts)

| direction | verdict | detail |
|---|---|---|
| Median calibration (shrink sweep) | small, ~exhausted | ETT ~0.6% MAE, double-win but tiny; per-channel collapses on ETTm1. Post-hoc trick, keep out of table. |
| Cluster embedding (FiLM) + per-cluster MAE weight | **NULL** | backbone micro-tweak (~0.2%) absorbed by MoE; full pipeline slightly worse. |
| Cross-channel `context_channel_head` on **ECL** (blocks=0) | **worse** | val +0.87% / +1.19% vs plain mlp. Cross-channel *without depth* doesn't help ECL. |
| Cross-channel + depth on **ECL-H96** (`cch`, hid192, blocks=2) | **NULL** | backbone val 0.113001/0.210303 vs plain-mlp baseline 0.112892/0.208403 = +0.10%/+0.91%; not clearly better, stopped before MoE. |
| Backbone **width** on PEMS08-H96 (hid 192/256, blocks=0) | useless | ~0% to −3%. |
| **Backbone DEPTH on PEMS08-H96** (`context_channel_head_blocks` 1→2) | **HUGE WIN** | see below |

### The PEMS08-H96 depth result (the breakthrough)
Backbone-alone on PEMS08-H96, val-selected (OLinear target = test 0.173 / 0.236):

| variant | val mse/mae | test mse/mae | gap vs target |
|---|---|---|---|
| hid128 b0 (original) | 0.2566 / 0.3306 | 0.2206 / 0.3255 | +27% / +38% |
| hid256 b0 (width) | 0.2480 / 0.3241 | 0.2134 / 0.3195 | +23% / +35% |
| hid192 **b1** | 0.2006 / 0.2758 | 0.1589 / 0.2677 | −8% / +13% |
| hid192 **b2** (val-best) | **0.1658 / 0.2449** | 0.1255 / 0.2305 | −27% / −2% |
| hid256 b2 | 0.1669 / 0.2460 | 0.1248 / 0.2300 | −28% / −3% |

**FULL pipeline (deep backbone + MoE):**
| | test mse/mae | gap vs target |
|---|---|---|
| original full pipeline | 0.1753 / 0.2890 | +1.3% / +22.5% (a loss, esp. MAE) |
| **MoE on hid192 b2** ⭐ | **0.1176 / 0.2247** | **−32.0% / −4.8% (clean double-win)** |

Takeaways: (1) **depth (residual blocks), not width**, is the lever; (2) the deep
backbone alone already beats target; (3) **the MoE still adds on top** of the deep
backbone (0.1255→0.1176 MSE), i.e. structural gains survive the MoE; (4) zero new code —
`context_channel_head_blocks` already existed. Run dir:
`outputs/pems08_h96_backbone_capacity/`. Best config:
`outputs/pems08_h96_backbone_capacity/configs/MOE_on_hid192_b2.yaml`.

Contrast: on ETT/ECL (plain `mlp` predictor) width/depth/cross-channel did NOT help —
plain MLP is saturated there. The depth win is specific to the **PEMS cch regime / long
horizon**, where the backbone was genuinely under-capacity.

---

### PEMS H48/H96 depth rollout (2026-06-17)

Run root: `outputs/pems_depth_rollout/` for new cells and PEMS03/04-H96;
PEMS08-H96 remains in `outputs/pems08_h96_backbone_capacity/`. Summary artifact:
`outputs/pems_depth_rollout/depth_rollout_summary.md`.

Protocol:
- Fixed variant: `context_channel_head_mlp`, `hidden_dim: 192`,
  `context_channel_head_blocks: 2`.
- New backbone runs for H48 and PEMS07-H96 used `eval.skip_test: true`; final
  frozen-MoE runs used `eval.skip_test: false`, so test was read once for the
  final selected pipeline in those cells.
- Existing PEMS03/04-H96 verification had already run b1/b2 + MoE. Closeout:
  b2 beat b1 on val for both cells, and MoE added on top of b2.

PEMS03/04-H96 closeout:

| cell | b1 backbone val/test | b2 backbone val/test | MoE on b2 val/test |
|---|---|---|---|
| PEMS03-H96 | 0.118016/0.239594; 0.171165/0.279898 | 0.106705/0.224047; 0.155506/0.261831 | 0.096851/0.215122; 0.137343/0.248539 |
| PEMS04-H96 | 0.103811/0.217575; 0.126884/0.239291 | 0.099142/0.211657; 0.120781/0.231630 | 0.089962/0.202039; 0.115193/0.226424 |

Full PEMS H48/H96 depth+MoE rollout:

| cell | backbone val mse/mae | MoE val mse/mae | MoE test mse/mae | vs old PKR | vs OLinear |
|---|---|---|---|---|---|
| PEMS03-H48 | 0.087627/0.201169 | 0.080474/0.194061 | 0.102999/0.212664 | -18.25%/-11.39% | -0.96%/+1.27% |
| PEMS03-H96 | 0.106705/0.224047 | 0.096851/0.215122 | 0.137343/0.248539 | -18.73%/-11.87% | -1.90%/+0.62% |
| PEMS04-H48 | 0.083498/0.192448 | 0.077301/0.184888 | 0.090328/0.197310 | -18.62%/-11.12% | -4.92%/+0.16% |
| PEMS04-H96 | 0.099142/0.211657 | 0.089962/0.202039 | 0.115193/0.226424 | -24.71%/-14.88% | -5.58%/+0.19% |
| PEMS07-H48 | 0.075877/0.178813 | 0.070712/0.173422 | 0.079334/0.179501 | -26.54%/-17.66% | -5.55%/+4.97% |
| PEMS07-H96 | 0.103624/0.212163 | 0.094327/0.203542 | 0.107024/0.209807 | -30.95%/-20.23% | -0.90%/+7.04% |
| PEMS08-H48 | 0.120436/0.211946 | 0.114947/0.207302 | 0.095284/0.201918 | -21.90%/-15.52% | -22.53%/-1.02% |
| PEMS08-H96 | 0.165827/0.244867 | 0.155276/0.236368 | 0.117636/0.224670 | -32.78%/-22.26% | -32.00%/-4.80% |

Verdict:
- NEXT-1 is confirmed. Depth was a structural capacity bottleneck for PEMS
  long horizons; the MoE still adds on top of the deeper backbone on every cell.
- Compared with the old PKR-MoE table, every H48/H96 PEMS cell improves by
  roughly 18-33% MSE and 11-22% MAE.
- Compared with OLinear, depth+MoE wins MSE on every H48/H96 PEMS cell. MAE is
  still slightly worse on PEMS03/04/07 (small on PEMS03/04, larger on PEMS07)
  and wins cleanly on PEMS08. Do not overclaim a full double-win outside PEMS08.

### PEMS H12/H24 depth add-on (2026-06-17)

Run root: `outputs/pems_depth_rollout/`. Summary artifact:
`outputs/pems_depth_rollout/depth_rollout_h12_h24_summary.md`.

Protocol:
- Fixed variant: `context_channel_head_mlp`, `hidden_dim: 192`,
  `context_channel_head_blocks: 2`.
- All H12/H24 backbone runs used `eval.skip_test: true`.
- Every H12/H24 backbone improved validation versus the original hid128/b0
  baseline, so all eight cells passed the val gate before MoE attachment.
- Final frozen-MoE runs used `eval.skip_test: false`; test was read once for the
  selected final pipeline.

Full PEMS H12/H24 depth+MoE rollout:

| cell | baseline val | backbone val | backbone val vs baseline | MoE val | MoE test | vs old PKR | vs OLinear |
|---|---|---|---|---|---|---|---|
| PEMS03-H12 | 0.058622/0.163587 | 0.052054/0.154133 | -11.20%/-5.78% | 0.050278/0.151907 | 0.057299/0.157899 | -4.50%/-2.53% | -4.50%/-0.69% |
| PEMS03-H24 | 0.078019/0.190583 | 0.064480/0.172499 | -17.35%/-9.49% | 0.060963/0.168505 | 0.074285/0.180074 | -7.14%/-4.72% | -4.76%/+0.60% |
| PEMS04-H12 | 0.070908/0.175526 | 0.064865/0.165997 | -8.52%/-5.43% | 0.061917/0.161797 | 0.065986/0.165131 | -4.37%/-3.43% | -2.96%/+1.31% |
| PEMS04-H24 | 0.084692/0.193161 | 0.071652/0.176322 | -15.40%/-8.72% | 0.067908/0.171137 | 0.075631/0.178133 | -9.96%/-6.74% | -4.26%/+1.21% |
| PEMS07-H12 | 0.058698/0.156973 | 0.051634/0.144072 | -12.03%/-8.22% | 0.049442/0.141318 | 0.051963/0.144964 | -7.21%/-4.63% | -0.07%/+5.05% |
| PEMS07-H24 | 0.080284/0.187593 | 0.061168/0.158795 | -23.81%/-15.35% | 0.057945/0.154888 | 0.062779/0.159843 | -14.00%/-9.69% | -3.42%/+5.86% |
| PEMS08-H12 | 0.074081/0.174550 | 0.067802/0.163914 | -8.48%/-6.09% | 0.065867/0.161233 | 0.060437/0.158729 | -5.57%/-4.95% | -11.12%/-0.17% |
| PEMS08-H24 | 0.102461/0.206497 | 0.085904/0.181259 | -16.16%/-12.22% | 0.083237/0.178602 | 0.073864/0.175320 | -12.07%/-9.63% | -17.01%/-1.51% |

Verdict:
- H12/H24 confirm the same PEMS CCH depth bottleneck: backbone-alone val gains
  are large enough to survive the frozen MoE stage.
- Compared with the old PKR-MoE table, every H12/H24 PEMS cell improves
  (roughly 4.5-14.0% MSE and 2.5-9.7% MAE).
- Compared with OLinear, MSE wins on every H12/H24 cell. MAE wins on
  PEMS03-H12 and PEMS08-H12/H24, is near parity on PEMS03-H24, and remains
  worse on PEMS04/PEMS07. Do not overclaim universal MAE double-win.

### ECL H96 cch+depth probe (2026-06-17)

Run root: `outputs/ecl_depth_probe/`. Summary artifact:
`outputs/ecl_depth_probe/ecl_h96_cch_h192_b2_backbone_summary.md`.

Protocol:
- Cloned `outputs/e_h96_alpha095_final_probe/configs/electricity/H96/final/electric_h96_centerres_h256_a095_wd0_bs128.yaml`.
- Changed backbone to `context_channel_head_mlp`, `hidden_dim: 192`,
  `context_channel_head_blocks: 2`.
- Used `moe.enable: false`, `eval.skip_test: true`; test was not read.
- Localized `exp.out_dir`, `corr.save_path`, `portrait.out_dir`,
  `knn_hybrid.path`, `memory.path`, and `memory.checkpoint_path` under
  `outputs/ecl_depth_probe/`; `memory.save_checkpoint: true`.

Result:
- Plain-mlp ECL-H96 baseline val: 0.112892 / 0.208403.
- cch+blocks2 hid192 backbone val: 0.113001 / 0.210303.
- Delta vs baseline: +0.10% / +0.91%.

Verdict:
- ECL cch+blocks2 backbone val = 0.113001/0.210303 vs 0.1129/0.2084,
  not clearly improved, so ECL remains conceded.
- Stop at backbone; do not attach MoE.

## 7. Open directions / NEXT STEPS (do these)

GPU is **serial** (one job at a time) — order the queue by certainty × value:
cheap/certain consolidation first, exploratory probes after. Discipline §5 always applies.

### NEXT-1 — ✅ DONE (2026-06-17): PEMS depth rollout, all 16 cells
Recipe `context_channel_head_mlp` + `hidden_dim:192` + `context_channel_head_blocks:2`,
val-selected, MoE attached on top. Full numbers in §6; runs in `outputs/pems_depth_rollout/`
(+ PEMS08-H96 in `outputs/pems08_h96_backbone_capacity/`). **Final verdict:**
- depth was a structural bottleneck for PEMS at **every** horizon (H12/24/48/96);
- the MoE still adds on top of the deeper backbone on every cell (gains survive);
- **MSE beats OLinear on all 16 PEMS cells**;
- **MAE = first-or-second**: clean win on PEMS08 (all H) + PEMS03-H12; near-parity on
  PEMS03/04; still behind on PEMS07 (worst, +5–7%).
- **Decision (user): good enough — SHIP, do NOT per-cell tune.** The uniform recipe is a
  strength; over-tuning risks overfit/leakage. H48/H96 audited clean (only caveat:
  H48 & PEMS07-H96 ran b2 only, no per-cell b1 re-check — acceptable).

### NEXT-2 (done 2026-06-17): integrate depth results into the comparison table
The clean PEMS depth wins have been integrated into the publishable table.
- Updated the **PEMS rows** of
  `outputs/codex_table_target_20260614/input96_olinear_filtered_comparison.md` with the
  depth+MoE **test** numbers from `outputs/pems_depth_rollout/depth_rollout_summary.md` +
  `depth_rollout_h12_h24_summary.md` (PEMS08-H96 from `outputs/pems08_h96_backbone_capacity/`).
- Use only val-selected / test-read-once numbers. **Do NOT add calibration to table numbers**
  (val-label post-hoc; only allowed if applied symmetrically to all baselines).
- Re-tallied 1st counts; non-PEMS rows were left untouched.

### NEXT-3 (done 2026-06-17): ECL depth probe
ECL uses plain `mlp` (channel-independent, no depth knob). Its MAE winners (OLinear/FITS)
use cross-channel + spectral, so we tested **cross-channel + depth together**:
- Backbone-alone `context_channel_head_mlp`, `hidden_dim: 192`,
  `context_channel_head_blocks: 2`.
- Result: val 0.113001 / 0.210303 vs plain-mlp baseline 0.112892 / 0.208403
  (`+0.10%/+0.91%`), i.e. not clearly improved.
- Decision: stopped before MoE. ECL stays conceded for now ("time-domain vs spectral"
  limitation remains the working explanation).

### NEXT-4 — probe: ETT long-horizon depth (expect likely NULL)
Hypothesis: ETT long horizons (H720/H336) might also benefit from depth.
- **Avoid the confound:** ETTh1-H96/ETTm1-H96 are plain `mlp` (no depth knob). Do NOT switch
  a plain-mlp cell to `context_channel_head_mlp` just for depth — that also injects cross-channel
  (which hurt ECL) and confounds. Instead either (a) use ETT cells **already on a cch/channel
  backbone** (config-only depth via `context_channel_head_blocks`), or (b) add a clean
  residual-block depth option to `ClusterwiseMLP` (real code, per §2a).
- Test **H720/H336 first** (depth benefit grows with horizon). Backbone-alone val first.
- **Expectation: likely NULL** (ETT is where plain MLP is near-optimal). A null is a useful
  result (confirms ETT saturated), not a failure — record and move on.

### NEXT-5 — lower priority: spectral/FITS backbone
No frequency-domain variant exists. Build only if NEXT-3/NEXT-4 stall and a genuinely new
architectural family is wanted (rFFT → complex linear → irFFT as a new per-cluster variant
in `cluster_predictor.py`). Real code; uncertain it survives MoE.

### NEXT-6 — parallel dework: consolidate
- Data-driven ("对症") penalty pool from **train-residual** diagnostics (not blind tuning).
- Interpretability artifact: cluster shape diagnostic → chosen penalty → expert (a key
  differentiator vs black-box baselines).

---

## 8. Status snapshot (update me)
- ✅ **PEMS depth rollout COMPLETE — all 16 cells (H12/24/48/96 × 03/04/07/08).**
  Uniform recipe `cch + hid192 + blocks2 + MoE`, val-selected, test read once.
  **MSE beats OLinear on every cell; MAE first-or-second** (clean win on PEMS08).
  Runs: `outputs/pems_depth_rollout/` + `outputs/pems08_h96_backbone_capacity/`.
  Summaries: `depth_rollout_summary.md`, `depth_rollout_h12_h24_summary.md`. Audited clean.
- ✅ Decision: PEMS good enough — ship, **no further per-cell tuning**.
- ✅ **NEXT-2 done:** integrated depth PEMS numbers into the comparison table (bookkeeping only; no calibration in table; counts re-tallied).
- ✅ NEXT-3 done: ECL cch+blocks2 backbone val = 0.113001/0.210303 vs 0.1129/0.2084; not clearly improved, stopped before MoE, ECL remains conceded.
- ⬜ NEXT-4: ETT H720/H336 depth probe (avoid cch confound; expect likely null).
- ⬜ NEXT-6: 对症 penalty pool + interpretability artifact.
- GPU serial → suggested queue: **ETT probe**.
- Comparison table = publishable floor at
  `outputs/codex_table_target_20260614/input96_olinear_filtered_comparison.md`; PEMS rows
  now contain the clean hid192+b2 depth re-runs; red/blue top2 highlighting was
  re-audited, a `Top2 Count` row was added, and TimeKAN(2025) screenshot values
  were inserted after OLinear where available. Leave other rows untouched.
