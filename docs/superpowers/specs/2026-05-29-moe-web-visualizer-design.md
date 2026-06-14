# MoE Web Visualizer Design

## Goal

Build a first visible version of a local web page for PKR-MoE experiments. The page should let a user select dataset configs or existing runs, start a training job, and inspect MoE participation and prediction replay when diagnostic files are available.

## Scope

The first version prioritizes visibility over full orchestration. It will:

- List YAML configs from `configs/`.
- List existing run directories under `outputs/` that contain `run_summary.json`.
- Show run metrics, penalty names, residual MoE summary, and gate participation summaries from existing result files.
- Load `prediction_intermediates.npz` and `prediction_intermediates_meta.json` for one run and replay a selected sample with base, raw residual, final, and true prediction curves.
- Start a training subprocess from the browser by materializing a temporary config under `outputs/web_visualizer/runs/<run_id>/config.yaml`.

It will not add per-batch instrumentation to `src.train` in this first version. During training, the page will show process status and output directory; detailed MoE replay appears after `src.train` writes diagnostics.

## Architecture

`src/web_visualizer_core.py` will contain testable filesystem and payload helpers: config discovery, run discovery, diagnostic loading, JSON-safe conversion, and training config materialization.

`src/web_visualizer.py` will be a small standard-library HTTP server. It will serve one HTML page and JSON endpoints. The frontend will use native JavaScript and canvas/SVG so the first version does not require a Node or FastAPI setup.

## Data Flow

1. Browser requests `/api/configs` and `/api/runs`.
2. User selects an existing run; browser requests `/api/run?run_dir=...`.
3. Backend reads `run_summary.json`, `cluster_penalty_probs.csv`, and optional prediction intermediates metadata.
4. User chooses a sample; browser requests `/api/prediction?run_dir=...&sample=0`.
5. Backend returns one JSON-safe sample with time-series arrays and gate matrices.
6. User starts training; browser posts to `/api/train/start`, backend writes a temporary config and spawns `python -m src.train --config <config>`.

## Testing

Focused unit tests will cover:

- Config discovery reads dataset and horizon labels.
- Run discovery finds summaries and marks whether prediction replay is available.
- Prediction payload extraction converts a `.npz` sample into JSON-safe arrays and MoE gate summaries.
- Training config materialization overrides horizon, diagnostics, output directory, and device safely.
