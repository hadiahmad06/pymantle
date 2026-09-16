# bench/

Workstream A: the gate. Nothing in `CLAUDE.md` §4 (B–I) starts until this produces a real
number. See `CLAUDE.md` §2 for the exact decision thresholds.

## Setup (on the GPU machine, disk-space-conscious)

Two things eat disk for no reason if you install the obvious way: torch's prebuilt CUDA wheel
duplicates the CUDA runtime libs your machine already has via `nvcc`, and prebuilt `cupy-cudaXXx`
bundles a *second* private copy of cuBLAS/cuDNN/NCCL/cuTENSOR (easily 1-2GB) on top of that.
torch's own bundling is unavoidable through pip (building torch from source to avoid it costs far
more disk mid-build than it saves), but cupy's is avoidable — its source distribution builds
against the CUDA toolkit already on `PATH` instead of shipping its own copy.

```bash
# 1. torch: pick the newest cuXXX tag your driver supports. The wheel's bundled
#    CUDA runtime is independent of `nvcc --version` -- nvcc's version only
#    matters for step 4 below. Check available tags at
#    https://download.pytorch.org/whl/torch/
python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126

# 2. small leaf deps (numpy, tqdm -- everything else is hand-rolled to avoid
#    pulling in torchvision/matplotlib, see data.py)
python -m pip install --no-cache-dir -r requirements.txt

# 3. spikingjelly, --no-deps so it doesn't drag in its own torchvision/matplotlib
python -m pip install --no-cache-dir --no-deps spikingjelly

# 4. cupy: source build against this machine's CUDA toolkit, not the prebuilt
#    cupy-cudaXXx wheel. Needs a C/C++ toolchain; picks up nvcc from PATH.
python -m pip install --no-cache-dir cupy

# 5. reclaim pip's own wheel cache (it keeps a copy of everything downloaded above)
python -m pip cache purge
```

`nsys` (Nsight Systems CLI) must also be on `PATH` for `profile_kernels.sh` — it ships with the
CUDA Toolkit, so it should already be there alongside `nvcc`.

Sanity-check after install: `du -sh $(python -c "import torch,os;print(os.path.dirname(torch.__file__))")`
and `python -c "import cupy; cupy.show_config()"` (should list the system CUDA install, not a
bundled one, and there should be no `nvidia_cudnn_cu12`/`nvidia_cublas_cu12`/etc. wheels of cupy's
own sitting in `.venv/lib/*/site-packages/` next to torch's).

## Quick timing (wall/GPU time, launch-gap fraction)

```bash
python train.py --framework torch --model mlp --T 20 --batch-size 128
python train.py --framework sj    --model mlp --T 20 --batch-size 128
```

Each prints one JSON line: wall time, GPU time, the gap between them (a first-order proxy for
launch/dispatch overhead), samples/sec, and the GPU model (`torch.cuda.get_device_name`).
Timing starts only after `--warmup` steps have already run, so process/CUDA-context startup
never enters the measured window.

## Sweep T and batch size

```bash
python sweep.py --model mlp --T-values 4,20,100,500 --bs-values 32,128,512 --out results/mlp_sweep.csv
```

Runs every (T, batch_size) pair for both frameworks and reports the SpikingJelly/PyTorch
speedup ratio and each framework's launch-gap fraction at each point, so we can see how the
ratio moves with problem size (thesis: smaller T / smaller batch should favor SpikingJelly's
fused kernels more, since launch overhead is a bigger fraction of a smaller total).

## Proper kernel-category breakdown (GEMM / elementwise / launch gaps / allocator)

The wall-vs-GPU gap in `train.py` is a cheap proxy. For the real §2 breakdown, profile under
Nsight Systems:

```bash
./profile_kernels.sh torch mlp --T 20 --batch-size 128
python parse_profile.py profiles/torch_mlp_<timestamp>
```

This buckets kernel time (GEMM vs. elementwise, by name-substring match against cuBLAS/CUTLASS
vs. TensorIterator pointwise kernel names) and CUDA API time (`cudaLaunchKernel` as launch
overhead, `cudaMalloc`/`cudaFree` as allocator). It's a heuristic first pass -- cross-check
anything surprising by opening the `.nsys-rep` in the Nsight Systems GUI before it goes into
`FINDINGS.md`.

## Models

- `mlp` — 2-layer feedforward LIF-SNN.
- `conv` — 2-conv-layer spiking CNN (MNIST-shaped input).
- `rnn` — single recurrent ALIF layer (hidden spikes feed back through a recurrent weight
  matrix each timestep; adaptive threshold).

Each has a plain-PyTorch reference (`models.py`, naive per-timestep Python loop) and a
SpikingJelly multi-step equivalent (`models_sj.py`, fused CuPy backend where available).

## Recording results

Every number that goes into `FINDINGS.md` must record: GPU model, framework + version, T,
batch size, and whether data was `synthetic` or `mnist`. Launch overhead as a fraction of
total time gets *worse* on faster hardware (GEMM gets cheaper, fixed per-kernel launch cost
doesn't) — so the same model can look GEMM-bound on an A100 and launch-bound on an H100. The
GPU model is not a footnote here, it's load-bearing.
