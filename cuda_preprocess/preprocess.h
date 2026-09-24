#pragma once
#include <cuda_runtime.h>

// Int32 CSR; FP32 SpMM. The caller owns buffers and selects the device/stream.
// A workspace must not be used concurrently; destroy after stream completion.
extern "C" {
const char *ag_error();
void *ag_create(int n);
void ag_destroy(void *workspace);
// Stable degree sort: perm[sorted_row] = original_row, plus virtual CSR offsets.
// No edge indices or values are read, copied, or allocated.
int ag_mapping(void *workspace, const int *ptr, int *perm, int *sorted_ptr,
               cudaStream_t stream);
// Input offsets are degree-sorted. Allocate at least max(1, n + nnz/384)
// int4 records and one device int for total. Valid metadata is out[0:*total].
int ag_partition(void *workspace, const int *ptr, int4 *out, int *total,
                 cudaStream_t stream);
const char *ag_cusparse_error();
void *ag_cusparse_create(int n, int nnz, int cols, int *ptr, int *idx,
                         float *val, float *x, float *y);
int ag_cusparse_run(void *handle, float *x, float *y, cudaStream_t stream);
void ag_cusparse_destroy(void *handle);
}
