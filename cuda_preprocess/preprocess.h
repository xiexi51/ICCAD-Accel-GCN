#pragma once
#include <cuda_runtime.h>

// Device selection is the caller's responsibility. Calls use the supplied stream.
// n, nnz and every CSR offset must fit int32. Inputs must be valid CSR.
// Do not share a workspace between concurrent streams. Destroy after completion.
extern "C" {
const char *ag_error();
void *ag_create(int n);
void ag_destroy(void *workspace);
// Allocate at least n + floor(nnz/384) int4 records (at least one if empty).
// total is a single device int; valid output is out[0:*total].
int ag_partition(void *workspace, const int *ptr, int4 *out, int *total, cudaStream_t stream);
// perm[new_row] = original_row. Column IDs are not renumbered.
// Values may both be null for unweighted graphs. Outputs must not alias inputs.
// If newidx is null, only perm/newptr are generated; idx/val may be null too.
int ag_reorder(void *workspace, const int *ptr, const int *idx, const float *val,
               int *perm, int *newptr, int *newidx, float *newval, cudaStream_t stream);
int ag_spmm(const int *meta, int blocks, const int *idx, const float *val,
            const float *x, float *y, int n, int nnz, int cols, cudaStream_t stream);
int ag_spmm_mapped(const int *meta, int blocks, const int *perm,
    const int *original_ptr, const int *sorted_ptr, const int *idx, const float *val,
    const float *x, float *y, int n, int cols, int original_output, cudaStream_t stream);
}
