# Workstream D — Autodiff

**Status:** not started — blocked on Workstream A gate.

## Goal
A differentiation engine shaped around SNN structure rather than a generic reverse-mode tape.

## Scope
- Tape where the time loop is a single differentiable primitive, not a sequence of per-step ops.
- Per-neuron-model recompute policy for surrogate gradients (recompute vs. store, hardcoded
  per model rather than generic autograd saving everything).
- Investigate forward-mode and online learning rules (e-prop, OSTL) as first-class alternatives —
  they suit SNNs and don't fit a reverse tape; this is a genuine differentiator, not a nice-to-have.

## Interface
- *Exposes:* `Node`, `backward()`.
- *Consumes:* Workstreams B, C.
