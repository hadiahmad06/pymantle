# Workstream F — Python Layer and Interop

**Status:** not started — blocked on Workstream A gate.

## Goal
Make the C++/CUDA core usable from Python, and borrow the PyTorch/JAX/CuPy ecosystem during
bring-up instead of rebuilding it.

## Scope
- nanobind bindings.
- GIL discipline.
- Buffer protocol support.
- `__dlpack__` early — zero-copy interop with PyTorch/JAX/CuPy is the escape hatch that lets us
  use their DataLoaders, optimizers, and plotting tools without adopting their tensor framework.

## Interface
- *Exposes:* the `pymantle` module.
- *Consumes:* Workstream B.
