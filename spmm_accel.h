#pragma once
#include <cuda_runtime.h>

extern "C" {
// Includes output zeroing. Returns a CUDA error code, not ag_error().
int ag_spmm_mapped(const int *meta, int blocks, const int *perm,
    const int *original_ptr, const int *sorted_ptr, const int *idx, const float *val,
    const float *x, float *y, int n, int cols, int original_output, cudaStream_t stream);
}
