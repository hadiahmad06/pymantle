# Workstream C — Dispatch

**Status:** not started — blocked on Workstream A gate.

## Goal
Route an op call to the right kernel implementation, cheaply.

## Scope
- Static dtype x device x requires-grad dispatch.
- Deliberately dumb/minimal — PyTorch's dispatcher is general-purpose and pays a per-op cost
  for that generality; avoiding that cost is a feature here, not a gap.

## Interface
- *Exposes:* op registration macro.
- *Consumes:* Workstream B.
