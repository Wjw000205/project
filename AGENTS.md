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
