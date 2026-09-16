"""Bucket an nsys CSV summary (from profile_kernels.sh) into the four
numbers CLAUDE.md Sec.2 asks for: GEMM / elementwise neuron update /
kernel launch gaps / allocator, as a fraction of total wall time.

This is a heuristic first pass, not the final word: kernel-name substring
matching for GEMM/elementwise, and CUDA API summary time for launch
overhead and allocator calls. Cross-check anything surprising by opening
the .nsys-rep in the Nsight Systems GUI before writing it into FINDINGS.md.

Usage:
    python parse_profile.py profiles/torch_mlp_1234567
    (reads profiles/torch_mlp_1234567_cuda_gpu_kern_sum.csv and
     profiles/torch_mlp_1234567_cuda_api_sum.csv)
"""

import csv
import sys

GEMM_MARKERS = ["gemm", "cutlass", "sgemm", "hgemm", "cublas", "implicit_convolve", "wgrad", "dgrad"]
ELEMENTWISE_MARKERS = [
    "elementwise_kernel", "vectorized_elementwise", "unrolled_elementwise",
    "CUDAFunctor", "fill_kernel", "copy_kernel", "reduce_kernel",
]
ALLOCATOR_MARKERS = ["cudamalloc", "cudafree", "cudamallocasync", "cudafreeasync"]
LAUNCH_MARKERS = ["cudalaunchkernel", "cudalaunchcooperativekernel"]


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def find_col(row, *candidates):
    for c in candidates:
        if c in row:
            return c
    raise KeyError(f"none of {candidates} found in columns {list(row.keys())}")


def bucket_kernels(rows):
    if not rows:
        return {"gemm_ns": 0, "elementwise_ns": 0, "other_kernel_ns": 0}
    name_col = find_col(rows[0], "Name")
    time_col = find_col(rows[0], "Total Time (ns)", "Total Time")
    gemm = elementwise = other = 0
    for r in rows:
        name = r[name_col].lower()
        t = int(float(r[time_col]))
        if any(m in name for m in GEMM_MARKERS):
            gemm += t
        elif any(m in name for m in ELEMENTWISE_MARKERS):
            elementwise += t
        else:
            other += t
    return {"gemm_ns": gemm, "elementwise_ns": elementwise, "other_kernel_ns": other}


def bucket_api(rows):
    if not rows:
        return {"launch_ns": 0, "allocator_ns": 0, "other_api_ns": 0}
    name_col = find_col(rows[0], "Name")
    time_col = find_col(rows[0], "Total Time (ns)", "Total Time")
    launch = allocator = other = 0
    for r in rows:
        name = r[name_col].lower()
        t = int(float(r[time_col]))
        if any(m in name for m in LAUNCH_MARKERS):
            launch += t
        elif any(m in name for m in ALLOCATOR_MARKERS):
            allocator += t
        else:
            other += t
    return {"launch_ns": launch, "allocator_ns": allocator, "other_api_ns": other}


def main():
    prefix = sys.argv[1]
    kern_rows = load_csv(f"{prefix}_cuda_gpu_kern_sum.csv")
    api_rows = load_csv(f"{prefix}_cuda_api_sum.csv")

    kern = bucket_kernels(kern_rows)
    api = bucket_api(api_rows)

    total_ns = sum(kern.values()) + api["launch_ns"] + api["allocator_ns"]
    if total_ns == 0:
        print("no time recorded -- check the profile ran and captured steps")
        return

    print(f"{'category':<20}{'ns':>15}{'% of total':>12}")
    for label, ns in [
        ("gemm", kern["gemm_ns"]),
        ("elementwise", kern["elementwise_ns"]),
        ("other kernels", kern["other_kernel_ns"]),
        ("launch gaps", api["launch_ns"]),
        ("allocator", api["allocator_ns"]),
    ]:
        print(f"{label:<20}{ns:>15}{100 * ns / total_ns:>11.1f}%")


if __name__ == "__main__":
    main()
