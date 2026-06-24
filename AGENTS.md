# Project Agent Instructions

## Local Shell Access

- On Windows, `cmd.exe` is available through `%ComSpec%`.
- When a task needs Windows CMD semantics, run commands as `cmd /c <command>`.
- Keep using PowerShell only for PowerShell-specific commands or when it is clearly the better fit.

## Required Project Context

- At the start of any non-trivial task in this repository, read `src/ARCHITECTURE_AND_NEXT_STEPS.md` before exploring or editing code.
- Treat `src/ARCHITECTURE_AND_NEXT_STEPS.md` as the current architecture map, experiment log, and next-step source of truth.
- After any meaningful exploration, experiment, run analysis, architectural discovery, or decision that would help the next agent avoid re-deriving context, update `src/ARCHITECTURE_AND_NEXT_STEPS.md`.
- Record durable facts: what was tried, relevant commands or configs, output paths, validation/test metrics when available, verdicts, and the next recommended action.
- Do not update the log for purely mechanical or trivial inspections that produce no reusable finding.

## Required Experiment Loop

- For PKR-MoE repair, routing, penalty, adapter, gate, anchor, or ablation work, read this section and the self-check rules in `src/ARCHITECTURE_AND_NEXT_STEPS.md` before starting each task.
- Follow this loop for every non-trivial exploration:
  1. Explore the current evidence and implementation first.
  2. State the concrete hypothesis and the observable that would confirm or refute it.
  3. Run one controlled diagnostic or experiment at a time, using val-only unless the existing validation rule allows a single test read.
  4. If the result is weak, bad, or surprising, stop and analyze where the failure occurred before changing anything else. Classify the cause as precisely as possible: data/target mismatch, routing target, gate expressivity, adapter candidate quality, skip/no-op behavior, train-val shift, selection policy, optimizer/regularization, or eval-path wiring.
  5. Only after that analysis, choose the next smallest fix or diagnostic.
  6. Record the finding and next action in `src/ARCHITECTURE_AND_NEXT_STEPS.md`.
- Do not blindly stack config changes or try variants just to see what happens. A failed run must produce a diagnosis before the next run.
