# MoE Web Visualizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local browser dashboard that makes PKR-MoE training outputs and inference participation visible.

**Architecture:** Add a pure helper module for filesystem/config/result payloads and a small standard-library HTTP server for the UI. Keep `src.train` unchanged for the first version and read its existing output files.

**Tech Stack:** Python standard library HTTP server, PyYAML via existing dependency, NumPy/Pandas for existing diagnostic files, native browser JavaScript canvas/SVG.

---

### Task 1: Core Data Helpers

**Files:**
- Create: `src/web_visualizer_core.py`
- Test: `tests/test_web_visualizer_core.py`

- [ ] Write failing tests for config discovery, run discovery, prediction sample extraction, and training config materialization.
- [ ] Run `python -m pytest tests/test_web_visualizer_core.py -q` and confirm the tests fail because the module does not exist.
- [ ] Implement the helper functions in `src/web_visualizer_core.py`.
- [ ] Re-run `python -m pytest tests/test_web_visualizer_core.py -q` and confirm the tests pass.

### Task 2: Local Web Server

**Files:**
- Create: `src/web_visualizer.py`
- Test: `tests/test_web_visualizer_core.py`

- [ ] Add the HTTP server with endpoints for `/`, `/api/configs`, `/api/runs`, `/api/run`, `/api/prediction`, `/api/train/start`, and `/api/train/status`.
- [ ] Keep training process state in memory and write generated configs under `outputs/web_visualizer/runs/`.
- [ ] Serve an HTML page that displays config selection, run selection, metrics, MoE participation tables, gate heatmaps, and prediction replay curves.
- [ ] Run the core tests again to verify the server imports do not break helper behavior.

### Task 3: Browser Verification

**Files:**
- Modify: none unless verification exposes defects.

- [ ] Start the server with `python -m src.web_visualizer --port 8765`.
- [ ] Open `http://127.0.0.1:8765` in the in-app browser.
- [ ] Verify the page lists configs and existing runs.
- [ ] Select a run with prediction intermediates and verify curves and MoE gate tables render.
