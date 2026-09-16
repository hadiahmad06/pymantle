# Workstream I — Compiler

**Status:** not started — decision pending. Do not drift into this without a logged decision.

## Goal
Evaluate an AOT compiler path for SNN models, which have static shapes and static control flow.

## Scope (if pursued)
- Domain IR with `Neuron` / `Synapse` / `TimeLoop` as first-class nodes.
- Fusion over the time axis.
- Codegen to CUDA C or PTX.
- Lean on MLIR rather than rolling custom compiler infra.

## Why this is gated
Eager execution (Workstream E) gives a 2x on architectures we hand-wrote kernels for; a compiler
gives a 2x on architectures users invent — but it's a real fork in the road. **The choice to
pursue this must be logged in `DECISIONS.md` before any code lands here.**

## Interface
- Not yet defined — depends on the decision.
