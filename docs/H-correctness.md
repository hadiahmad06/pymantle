# Workstream H — Correctness

**Status:** not started — blocked on Workstream A gate.

## Goal
Guarantee numerical correctness across the whole stack. Silent bugs in surrogate gradients
waste months of a researcher's downstream time — this is a hard requirement, not polish.

## Scope
- Finite-difference gradcheck.
- Bitwise-determinism mode.
- CPU-vs-CUDA parity suite (every CUDA kernel ships with a CPU reference and a test asserting
  they agree — see `CLAUDE.md` §3 rules, no exceptions).
- Fixed-seed cross-validation against SpikingJelly.

## Interface
- *Consumes:* everything.
