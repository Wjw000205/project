# Current Module Ablation and Transfer Rerun Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-run ablation and transfer experiments using the current PKR-MoE implementation and current dataset YAML configurations.

**Architecture:** Keep old outputs as historical records. Generate fresh configs from the current `configs/*.yaml`, force train-only normalization/clustering, disable KNN/calibration, and write all new runs under separate `outputs/current_module_*` roots.

**Tech Stack:** Python, PyTorch training entrypoint `src.train`, transfer entrypoint `src.transfer`, YAML configs, CSV summaries.

---

### Task 1: Current-Module Ablation Runner

**Files:**
- Create: `scripts/rerun_current_module_ablation.py`
- Output: `outputs/current_module_ablation_rerun/`

- [ ] Create a runner that reads `configs/{dataset}.yaml`, overrides `window.input_len=336`, `window.pred_len=96`, sets `normalize.train_only=true`, `cluster.train_only=true`, disables `knn_hybrid` and `calibration`, and writes generated YAMLs under the output root.
- [ ] Implement module ablations: `moe_off`, `zero_lambda_residual`, `fixed_lambda_residual`, `penalty_loss_only`, `full_current`.
- [ ] Implement detach ablations: `no_detach`, `detach_penalty_grad`, `detach_routed_penalty_pred`, `detach_both`.
- [ ] Implement backbone ablations: `mlp`, `nlinear`, `dlinear_k25`, `dlinear_k13` as paired MoE-on/off runs.
- [ ] Implement seed reruns for current MoE-on.
- [ ] Verify with `python -m py_compile scripts\rerun_current_module_ablation.py`.
- [ ] Smoke run one dataset with `--epochs 1`.

### Task 2: Current-Module Transfer Runner

**Files:**
- Create: `scripts/rerun_current_module_transfer.py`
- Output: `outputs/current_module_transfer_rerun/`

- [ ] Train current source checkpoints with memory/checkpoint saving enabled for ETTm1 H96 and ETTm2 H96.
- [ ] Generate transfer configs from current source checkpoints and current target configs.
- [ ] Run direct transfer and validation-route transfer with train-only route fitting.
- [ ] Summarize `source_test`, `target_self`, `direct_transfer`, `val_route_transfer`, and route metadata to CSV.
- [ ] Verify with `python -m py_compile scripts\rerun_current_module_transfer.py`.
- [ ] Smoke run one source-target pair.

### Task 3: Paper Summary Refresh

**Files:**
- Modify: `outputs/experiment_excel_summary/paper_style_experiment_summary.md`
- Create: `outputs/current_module_ablation_rerun/current_module_ablation_report.md`
- Create: `outputs/current_module_transfer_rerun/current_module_transfer_report.md`

- [ ] Mark old ablation/transfer tables as historical if their source is not the current-module rerun.
- [ ] Insert current-module ablation tables after reruns complete.
- [ ] Insert current-module transfer tables after reruns complete.
- [ ] Keep wording conservative: current-module rerun, validation-ranked route selection, test reported as final evaluation.
