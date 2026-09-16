# Workstream G — Build and Release

**Status:** not started — blocked on Workstream A gate.

## Goal
Make `pip install pymantle` work reliably across environments. Treat as load-bearing, not chores —
this is the workstream that kills projects if neglected.

## Scope
- CMake build system.
- manylinux wheels.
- C++11 ABI.
- Fat binaries with PTX fallback across sm_70 to sm_100+.
- CUDA-version x Python-version CI matrix.
- sccache to hold builds under an hour.

## Interface
- *Exposes:* `pip install pymantle`.
- *Consumes:* everything.
