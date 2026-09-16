"""Train one SNN config and report timing.

Usage:
    python train.py --framework torch --model mlp --T 20 --batch-size 128
    python train.py --framework sj    --model conv --T 50 --batch-size 64 --data mnist

Timing methodology: the clock starts only after `--warmup` full train steps
have already run and torch.cuda.synchronize()'d. That absorbs CUDA context
init, cuDNN/cuBLAS handle creation, kernel autotune/JIT, and first-touch
allocator growth into an unmeasured region, so the measured window is
steady-state training only.

Reports both wall time (host-observed, includes launch/dispatch gaps not
hidden by async execution) and GPU time (cuda-event, device-busy only). The
gap between them is a first-order proxy for kernel-launch overhead -- the
same quantity Workstream A's Nsight profiling breaks down properly.
"""

import argparse
import json
import sys

import torch
import torch.nn as nn

from data import synthetic_batch, mnist_rate_coded
from timing import Timer, warmup, gpu_name

IN_SHAPES = {
    "mlp": (784,),
    "conv": (1, 28, 28),
    "rnn": (784,),
}


def build_model(framework, model_name, device):
    if framework == "torch":
        from models import MODELS

        model = MODELS[model_name]()
    elif framework == "sj":
        from models_sj import MODELS_SJ

        model = MODELS_SJ[model_name]()
    else:
        raise ValueError(framework)
    return model.to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--framework", choices=["torch", "sj"], required=True)
    p.add_argument("--model", choices=["mlp", "conv", "rnn"], required=True)
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--data", choices=["synthetic", "mnist"], default="synthetic")
    p.add_argument("--steps", type=int, default=50, help="measured steps")
    p.add_argument("--warmup", type=int, default=20, help="discarded steps before timing starts")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", default=None, help="write JSON result here in addition to stdout")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("warning: no CUDA device, results are meaningless for this benchmark", file=sys.stderr)

    torch.manual_seed(0)
    model = build_model(args.framework, args.model, device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    in_shape = IN_SHAPES[args.model]
    if args.data == "mnist":
        flatten = args.model != "conv"
        make_batch = mnist_rate_coded(args.T, args.batch_size, device, flatten=flatten)
    else:
        def make_batch():
            x = synthetic_batch(args.T, args.batch_size, in_shape, device)
            y = torch.randint(0, 10, (args.batch_size,), device=device)
            return x, y

    def step():
        x, y = make_batch()
        out_spikes = model(x)  # [T, B, out_dim]
        logits = out_spikes.mean(dim=0)  # rate-coded readout
        loss = loss_fn(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    warmup(step, n=args.warmup)

    timer = Timer()
    timer.start()
    for _ in range(args.steps):
        step()
    wall_s, gpu_s = timer.stop()

    n_samples = args.steps * args.batch_size
    result = {
        "framework": args.framework,
        "model": args.model,
        "T": args.T,
        "batch_size": args.batch_size,
        "data": args.data,
        "steps": args.steps,
        "warmup": args.warmup,
        "gpu": gpu_name(),
        "wall_s": wall_s,
        "gpu_s": gpu_s,
        "launch_gap_s": max(wall_s - gpu_s, 0.0),
        "launch_gap_frac": max(wall_s - gpu_s, 0.0) / wall_s if wall_s > 0 else None,
        "samples_per_s": n_samples / wall_s if wall_s > 0 else None,
    }

    line = json.dumps(result)
    print(line)
    if args.out:
        with open(args.out, "a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
