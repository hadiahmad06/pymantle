# Workstream E — Kernels

**Status:** not started — blocked on Workstream A gate.

## Goal
The actual speed: fused, persistent neuron-update kernels.

## Scope
- Fused multi-step neuron update, forward and backward: LIF, adaptive/ALIF, PLIF, Izhikevich.
- Persistent across `T` (state held in registers, not round-tripped to HBM each step).
- CUDA Graphs to collapse the static launch sequence.
- GEMM delegates to cuBLAS/CUTLASS — not reimplemented here.

## Interface
- *Exposes:* kernel registry.
- *Consumes:* Workstreams B, D.

## Watch
- Warp divergence in reset logic.
- Register pressure from the persistent design capping occupancy — measure both, don't assume.
