# bench/

Workstream A: the gate. Nothing in `CLAUDE.md` §4 (B–I) starts until this produces a real
number. See `CLAUDE.md` §2 for the exact decision thresholds.

## Setup (on the GPU machine)

```bash
pip install -r requirements.txt
# nsys (Nsight Systems CLI) must also be on PATH for profile_kernels.sh -- it
# ships with the CUDA Toolkit / is installable standalone from NVIDIA.
```

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
