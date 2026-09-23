#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
NVCC=${NVCC:-/usr/local/cuda-12.2/bin/nvcc}
"$NVCC" -O3 -std=c++17 -arch=sm_86 -Xcompiler -fPIC -shared \
  cuda_preprocess/preprocess.cu spmm_accel.cu -lcusparse -o cuda_preprocess/libaccel_preprocess.so
