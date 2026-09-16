# PyMantle

A from-scratch deep learning framework for spiking neural networks. C++/CUDA core, thin Python
binding. Not a PyTorch wrapper, not a PyTorch fork.

---

## 1. What this project is

PyMantle trains SNNs faster than a general-purpose tensor framework can, by exploiting four
properties of SNN workloads that PyTorch's design cannot express:

1. **The time loop is the unit of work.** A forward pass is `for t in range(T): for layer: update(state, x[t])`.
   PyTorch launches `T x L x ~8` kernels and round-trips membrane potential to HBM every one.
   A persistent kernel holding `V` in registers across all `T` steps removes both the launch
   overhead and the memory traffic. The neuron update is pure elementwise arithmetic and 100%
   bandwidth-bound, so this is where the multiples are.
2. **Spikes are one bit stored as thirty-two.** BPTT saves activations for every timestep:
   memory is `O(T x N)` and the saved value is binary. Bit-packing is a 32x cut on the dominant
   memory term, which converts into longer sequences and larger batches.
3. **Surrogate gradients are recomputable.** Backward needs `sigma'(V - theta)`, an elementwise
   function of a recoverable value. Generic autograd must save it; a domain-specific engine can
   hardcode recompute-vs-store per neuron model.
4. **Binary activations change what a matmul is.** `spikes @ W` is masked column accumulation —
   operand values are never loaded. (Treat this as the least proven of the four. See §2.)

Secondary goal: an IR that natively represents "stateful neuron, discrete time" is the right
substrate for compiling to neuromorphic targets (Loihi, SpiNNaker, Akida). PyTorch's IR is not.

### Non-goals

- General-purpose tensor computing. Generality is the thing that makes PyTorch slow here.
- Beating cuBLAS/CUTLASS at GEMM. We will not. Delegate dense matmul from day one and never
  revisit it without a profile that demands it.
- CPU performance. CPU exists only as a correctness reference and for small-scale dev.
- Training ANNs, inference serving, mobile, quantization, ONNX, distributed — none of it, yet.

---

## 2. The gate: profile before building

**No core work starts until Workstream A produces a number.** Amdahl's law decides whether this
project is a speed framework or a research framework, and we need to know which one we are
writing before we commit years to it.

The number we need: for 3-4 representative SNN architectures, the wall-clock breakdown of
GEMM / elementwise neuron update / kernel launch gaps / allocator, measured under Nsight Systems.

- If GEMM dominates (>60%), the speed thesis is weak. Say so in the README, reframe around
  memory (thesis 2) and ergonomics, and scale the ambition down.
- If elementwise + launch gaps dominate (>50%), the thesis is strong and §1.1 and §1.3 are
  the whole product.

Also measure, in the same pass:
- What `torch.compile` / Inductor already fuses for free. That is our real baseline, not eager.
- Where SpikingJelly's multi-step CuPy kernels plateau and why. They already do a version of
  thesis 1; we need to know exactly what they leave on the table.
- Actual firing rates. Unstructured sparsity does not pay on a GPU until density is roughly
  1-5%. If our models fire at 15%, thesis 4 is dead and we stop designing around it.

Record all of this in `bench/FINDINGS.md`. Every later design decision cites it.

---

## 3. How to work on this repo

This brief is deliberately non-linear. The workstreams in §4 have defined interfaces between
them and can be developed in almost any order. When starting a session:

- **Ask which workstream, or propose one and wait.** Don't assume the next item in the list.
- **Read `DECISIONS.md` first.** It is the append-only log of architectural choices and their
  rationale. If a task contradicts a logged decision, stop and raise it rather than quietly
  doing it the new way.
- **Open questions live in `QUESTIONS.md`.** When you hit a fork that needs a human, append it
  there with the options and tradeoffs rather than picking one and moving on.
- Stubs are fine and encouraged. A workstream that needs an allocator can code against the
  allocator interface before the allocator exists. Mark stubs `// PYMANTLE-STUB:` so they grep.

### Rules

- **PyTorch is BSD-3.** Reading `c10/cuda/CUDACachingAllocator.cpp`, the dispatcher, and the
  autograd engine for reference is fair game and encouraged. Copying code is not — attribute
  in comments where a design is derived.
- **Every CUDA kernel ships with a CPU reference implementation** and a test asserting they
  agree. No exceptions. Silent numerical bugs in surrogate gradients will waste months of
  a researcher's time downstream.
- **Every performance claim ships with a reproducible benchmark script** in `bench/`, pinned
  seed, reported hardware, and a comparison against both SpikingJelly and `torch.compile`.
  A framework's entire credibility is one number; we do not publish estimates.
- **No new dependencies without asking.** Current allowed set: CUDA Toolkit, cuBLAS/CUTLASS,
  nanobind, CMake, pytest, Google Test.
- Don't scope-creep into features listed under Non-goals, however small they look.

---

## 4. Workstreams

Independent enough to attack in parallel. Each lists its interface contract with the others.

### A — Profiling and baselines  *(gate; do this first)*
Harness that runs reference models under Nsight, parses the traces, emits the §2 breakdown.
Models: small MLP-SNN on MNIST/N-MNIST, a spiking CNN, one recurrent/adaptive-LIF net, one with
large T (>=100). Outputs `bench/FINDINGS.md`.
*Produces:* the numbers everything else cites. *Consumes:* nothing.

### B — Core tensor and memory
Strided tensor, dtype system **including a real bit-packed 1-bit dtype** (not a bool alias),
device abstraction, stream/event handling, and a stream-aware caching allocator with
fragmentation control. The allocator is a large chunk of why PyTorch is fast; budget for it.
*Exposes:* `Tensor`, `Allocator`, `Stream`. *Consumes:* nothing.

### C — Dispatch
Static dtype x device x requires-grad dispatch. Keep it dumb. PyTorch's dispatcher is general
and pays for it per-op; not paying that is a feature.
*Consumes:* B. *Exposes:* op registration macro.

### D — Autodiff
Tape where **the time loop is a single differentiable primitive**, not a sequence of ops.
Per-neuron-model recompute policy for surrogate gradients. Investigate forward-mode and online
rules (e-prop, OSTL) as first-class alternatives — they suit SNNs and don't fit a reverse tape,
and supporting them is a genuine differentiator.
*Consumes:* B, C. *Exposes:* `Node`, `backward()`.

### E — Kernels
Fused multi-step neuron update, forward and backward: LIF, adaptive/ALIF, PLIF, Izhikevich.
Persistent across `T`. Then CUDA Graphs to collapse the static launch sequence. GEMM delegates
to cuBLAS/CUTLASS.
Watch: warp divergence in reset logic; register pressure from the persistent design capping
occupancy — measure both, don't assume.
*Consumes:* B, D. *Exposes:* kernel registry.

### F — Python layer and interop
nanobind, GIL discipline, buffer protocol, and **`__dlpack__` early**. Zero-copy interop with
PyTorch/JAX/CuPy is the escape hatch that lets us borrow their DataLoaders, optimizers and
plotting during bring-up instead of rebuilding a decade of ecosystem.
*Consumes:* B. *Exposes:* the `pymantle` module.

### G — Build and release
CMake, manylinux wheels, C++11 ABI, fat binaries with PTX fallback across sm_70 to sm_100+,
CUDA-version x Python-version CI matrix, sccache to hold builds under an hour.
This is the workstream that kills projects. Treat it as load-bearing, not chores.
*Consumes:* everything. *Exposes:* `pip install pymantle`.

### H — Correctness
Finite-difference gradcheck, bitwise-determinism mode, CPU-vs-CUDA parity suite,
fixed-seed cross-validation against SpikingJelly.
*Consumes:* everything.

### I — Compiler  *(decide, don't drift into)*
SNN models have static shapes and static control flow, so an AOT compiler is viable: a domain IR
with `Neuron` / `Synapse` / `TimeLoop` as first-class nodes, fusion over the time axis, codegen
to CUDA C or PTX. Lean on MLIR rather than rolling our own infra.
Eager gives a 2x on architectures we hand-wrote kernels for; a compiler gives a 2x on
architectures users invent. **This is a real fork in the road — log the choice in `DECISIONS.md`
before any code lands here.**

---

## 5. Milestone sequence (the linear spine, when in doubt)

1. Workstream A completes. Findings logged. Thesis confirmed or reframed.
2. CPU-only tensor + autograd + one LIF neuron that trains MNIST to expected accuracy.
3. DLPack interop, so PyTorch's data pipeline is available.
4. One fused CUDA multi-step LIF kernel. Benchmarked against SpikingJelly and `torch.compile`.
5. Bit-packed spike storage. Report the memory reduction and the max-`T` improvement.
6. CUDA Graphs.
7. Remaining neuron models.
8. Public release — **not before step 4's number is real and reproducible.**

Realistically multi-year to usability for a small team. Plan accordingly; don't let anything
get announced early.

<!-- pipeline:start -->
## Pipeline

This repo is wired into the personal dev pipeline (Slack/Linear/GitHub, service-agnostic scaffold).

- Allowed remote: hadiahmad06/pymantle
- Feature branches: `feature/<slug>` — create with a worktree, not a plain checkout:
  `git worktree add .pipeline/worktrees/<slug> -b feature/<slug>`
- Every commit you make must be prefixed `[claude]`
- Ticket context: `rg <id> .pipeline/tickets/`. If missing, fetch it first:
  `.pipeline/adapters/linear/fetch.sh <id> > .pipeline/tickets/<id>.md`
- Prior critical context humans flagged for you: `rg "CRITICAL" .pipeline/slack.md`
- Open PRs with `gh pr create`. Never merge unless explicitly asked.
- Never push directly to main/master — a hook enforces this, but don't attempt it.
<!-- pipeline:end -->
