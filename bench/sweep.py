"""Sweep T and batch size across both frameworks for one model, and report
how the PyTorch/SpikingJelly ratio and each framework's launch-gap fraction
move with problem size.

Usage:
    python sweep.py --model mlp --out results/mlp_sweep.csv
    python sweep.py --model mlp --T-values 4,20,100,500 --bs-values 32,128,512
"""

import argparse
import csv
import json
import os
import subprocess
import sys


def run_one(framework, model, T, batch_size, data, steps, warmup):
    cmd = [
        sys.executable, "train.py",
        "--framework", framework,
        "--model", model,
        "--T", str(T),
        "--batch-size", str(batch_size),
        "--data", data,
        "--steps", str(steps),
        "--warmup", str(warmup),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["mlp", "conv", "rnn"], required=True)
    p.add_argument("--T-values", default="4,20,100,500")
    p.add_argument("--bs-values", default="32,128,512")
    p.add_argument("--data", choices=["synthetic", "mnist"], default="synthetic")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    T_values = [int(v) for v in args.T_values.split(",")]
    bs_values = [int(v) for v in args.bs_values.split(",")]

    rows = []
    for T in T_values:
        for bs in bs_values:
            row = {"model": args.model, "T": T, "batch_size": bs}
            for framework in ("torch", "sj"):
                r = run_one(framework, args.model, T, bs, args.data, args.steps, args.warmup)
                row[f"{framework}_wall_s"] = r["wall_s"]
                row[f"{framework}_gpu_s"] = r["gpu_s"]
                row[f"{framework}_launch_gap_frac"] = r["launch_gap_frac"]
                row["gpu"] = r["gpu"]
            row["sj_speedup"] = row["torch_wall_s"] / row["sj_wall_s"]
            print(row)
            rows.append(row)

    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
