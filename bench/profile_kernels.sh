#!/usr/bin/env bash
# Profile one training config under Nsight Systems and dump CSV summaries
# for parse_profile.py to bucket into GEMM / elementwise / launch-gap /
# allocator time (the CLAUDE.md Sec.2 breakdown).
#
# Usage: profile_kernels.sh <framework: torch|sj> <model: mlp|conv|rnn> [extra train.py args...]
set -euo pipefail

FRAMEWORK="$1"; MODEL="$2"; shift 2
OUTDIR="profiles"
mkdir -p "$OUTDIR"
NAME="${FRAMEWORK}_${MODEL}_$(date +%s)"

nsys profile \
  --trace=cuda,nvtx,osrt \
  --output "$OUTDIR/$NAME" \
  --force-overwrite=true \
  python train.py --framework "$FRAMEWORK" --model "$MODEL" --steps 30 --warmup 20 "$@"

nsys stats \
  --report cuda_gpu_kern_sum,cuda_api_sum \
  --format csv \
  --output "$OUTDIR/$NAME" \
  "$OUTDIR/$NAME.nsys-rep"

echo "wrote $OUTDIR/${NAME}_cuda_gpu_kern_sum.csv and $OUTDIR/${NAME}_cuda_api_sum.csv"
echo "parse with: python parse_profile.py $OUTDIR/$NAME"
