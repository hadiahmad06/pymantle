# Workstream A — Profiling and Baselines

**Status:** not started — this is the gate; no core work (B–I) begins until this produces numbers.

## Goal
Produce the wall-clock breakdown (GEMM / elementwise neuron update / kernel launch gaps /
allocator) for 3–4 representative SNN architectures under Nsight Systems, so we know whether
the speed thesis in `CLAUDE.md` §1 holds.

## Scope
- Harness that runs reference models under Nsight Systems and parses the traces.
- Models: small MLP-SNN on MNIST/N-MNIST, a spiking CNN, one recurrent/adaptive-LIF net,
  one with large T (>=100).
- Also measure: what `torch.compile`/Inductor already fuses, where SpikingJelly's multi-step
  CuPy kernels plateau, and actual firing rates (sparsity only pays off around 1–5% density).

## Interface
- *Produces:* `bench/FINDINGS.md` — the numbers every later design decision cites.
- *Consumes:* nothing.

## Notes
See `CLAUDE.md` §2 for the exact decision thresholds (>60% GEMM → reframe around memory/ergonomics;
>50% elementwise+launch → thesis confirmed).
