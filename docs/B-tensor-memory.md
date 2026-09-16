# Workstream B — Core Tensor and Memory

**Status:** not started — blocked on Workstream A gate.

## Goal
The foundational data structures every other workstream builds on.

## Scope
- Strided tensor type.
- Dtype system, including a real bit-packed 1-bit dtype (not a bool alias).
- Device abstraction.
- Stream/event handling.
- Stream-aware caching allocator with fragmentation control.

## Interface
- *Exposes:* `Tensor`, `Allocator`, `Stream`.
- *Consumes:* nothing.

## Notes
The allocator is a large chunk of why PyTorch is fast — budget real time for it, not a
throwaway bump allocator. Reading PyTorch's `c10/cuda/CUDACachingAllocator.cpp` for reference
is fine (BSD-3); copying code is not.
