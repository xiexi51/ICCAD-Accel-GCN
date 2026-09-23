# CUDA preprocessing

CUDA preprocessing for ICCAD-Accel-GCN, preserving the original Python script
and SpMM kernel. Measurements are summarized in [RESULTS.md](RESULTS.md), with
raw samples in [../results](../results).

The variant that keeps edge arrays in place and its comparison results are in
[MAPPING_RESULTS.md](MAPPING_RESULTS.md). Mapping-only preprocessing, Accel,
and cuSPARSE results for Collab (GPU 2, 128 columns) are in
[COLLAB_RESULTS.md](COLLAB_RESULTS.md). The same commands support other
available official graphs:

```bash
PYTHON=/home/xix22010/anaconda3/envs/torch2/bin/python
"$PYTHON" cuda_preprocess/fetch_graph.py collab
CUDA_VISIBLE_DEVICES=2 "$PYTHON" cuda_preprocess/graph_benchmark.py --graph collab --cols 128
```

```python
from backend import Preprocessor, spmm_mapped

workspace = Preprocessor(n, nnz, mapping_only=True)  # No nnz-sized output allocation.
workspace.mapping(original_rowptr)  # Does not read indices or values.
blocks = workspace.count.item()
spmm_mapped(workspace.meta, blocks, workspace.perm, original_rowptr,
            workspace.rowptr, original_indices, original_values, x, out,
            original_output=True)  # Original node order; False selects degree-sorted order.
workspace.close()
```

Mapping-only mode generates a stable row permutation, virtual sorted rowptr,
and block4 records matching the original script. Outside the neighbor loop,
SpMM uses `original_row = perm[sorted_row]` and
`original_rowptr[original_row]` to locate the original edges. High-degree
segments additionally use `block_loc - sorted_rowptr[block_row]`. No per-edge
mapping or edge-index/weight copy is required. Direct original-order output
writes each row to `original_row`, avoiding a separate inverse-permutation gather.

Reproduce the comparison with identical stable sorting and 15 randomized
sample groups, each containing 10 calls per variant:

```bash
CUDA_VISIBLE_DEVICES=2 /home/xix22010/anaconda3/envs/torch2/bin/python \
  cuda_preprocess/mapping_benchmark.py
```

## Reproduction

The recorded experiments use an existing local environment without installing
dependencies. All CUDA experiments use physical GPU 2, visible as `cuda:0`
inside the process.

```bash
cd /home/xix22010/py_projects3/accel_gcn
export CUDA_VISIBLE_DEVICES=2
PYTHON=/home/xix22010/anaconda3/envs/torch2/bin/python

# Reddit from the official 18graphs.tar.gz is already in graphs/.
# If missing, run: bash cuda_preprocess/prepare_data.sh
bash cuda_preprocess/build.sh
"$PYTHON" cuda_preprocess/validate.py
"$PYTHON" cuda_preprocess/benchmark.py
"$PYTHON" cuda_preprocess/extra_timings.py
"$PYTHON" cuda_preprocess/training.py --epochs 20
"$PYTHON" cuda_preprocess/report.py

# Optional CUDA memory checking.
/usr/local/cuda-12.2/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
  "$PYTHON" cuda_preprocess/validate.py
```

Compilation requires C++17, CUDA/CUB, and cuSPARSE; defaults are nvcc 12.2 and
sm_86. Python handles allocation, dispatch, validation, and timing. Sorting,
scans, CSR materialization, and metadata generation execute in CUDA/CUB.
`backend.py` calls the shared library through ctypes, without a compiled PyTorch
extension. C++ callers can use [preprocess.h](preprocess.h) directly.

## Outputs and algorithms

`Preprocessor.partition(rowptr)` matches the computational portion of the
original `block_level_partition.py`. Input is the GPU int32 array corresponding
to `.new_indptr`; edge indices, SciPy CSR construction, and unit-weight arrays
are unnecessary. Output is a GPU int32 array of shape `[capacity, 4]` containing
`degree, row_begin, loc_begin, info`; GPU `count[0]` holds the valid length.

The algorithm marks consecutive equal-degree runs, computes their start indices
with prefix-max, counts blocks per row, obtains output offsets with an exclusive
sum, and emits int4 records in parallel. Zero-degree rows emit no blocks;
degrees above 384 are split into 384-edge segments. **Degrees 192–384 use
`warp_nz=32, block_rows=1`**, matching the original table ending at degree 191.

`Preprocessor.full(rowptr, indices, values, out_values)` also computes degrees,
uses CUB stable radix sorting for the permutation, scans new row offsets, copies
edge indices and optional FP32 weights row by row, and generates metadata.
Only rows are reordered; column IDs are unchanged, with
`perm[new_row] = original_row`. Outputs are `rowptr, indices, perm, meta, count`.
Materializing full CSR accesses edges and therefore costs O(N+E), not only O(N).

```python
from backend import Preprocessor

# ptr/idx: contiguous CUDA int32. Offsets and nnz must fit int32.
workspace = Preprocessor(ptr.numel() - 1, idx.numel())
meta, device_count = workspace.partition(ptr)
blocks = device_count.item()  # Synchronize if the launch grid is needed on CPU.
# meta[:blocks] can be passed directly to Accel-GCN without writing a .block4 file.
workspace.close()
```

Calls enqueue on PyTorch's current stream. The hot path performs no allocations
or CPU synchronization. Keep the workspace alive until GPU operations complete;
do not reuse it concurrently across streams. Repeated calls overwrite previous
outputs. C API callers must ensure sufficient capacity, correct dtypes, valid
CSR, and non-overlapping inputs/outputs.

The original repository does not provide the sorting script that produced
`.new_indices`. The new full mode uses stable sorting as described in the paper.
Its edge ordering is **not byte-identical to the downloaded `.new_indices`**,
but `.new_indptr` and `.block4` are identical. Each row has been checked against
the corresponding stable permutation of original CSR. Training restores node
order with an inverse permutation after SpMM and uses an explicit transpose
for backward, preserving mathematical semantics.

## Validation and timing conventions

- Reddit metadata matches both the unmodified Python script's output and the
  shipped file, element by element in int32.
- Thirteen test cases cover empty graphs, all-zero degrees, partial groups,
  strategy boundaries, high degrees, unsorted inputs, and degree ties. They
  validate full sorting and SpMM with 32/41/128 columns.
- Every Reddit SpMM output element is compared with cuSPARSE. Training forward
  and transposed backward are also validated element by element.
- CUDA event times use five warmups and the median of seven groups. Each group
  contains 100 metadata calls, 10 full preprocessing calls, or 20 SpMM calls.
- Single-call wall time includes host launch and synchronization. H2D uses
  pageable CPU memory and preallocated GPU buffers. Disk I/O is excluded.
- The original SpMM uses atomicAdd; the wrapper clears output before every call,
  and this cost is included in SpMM timing.
- Training uses real Reddit topology, synthetic features/labels, standard
  self-loops and symmetric degree normalization, two layers (602→128→41), ReLU,
  dropout 0.5, Adam, FP32, and disabled TF32. Five warmup epochs precede 20 measured
  epochs. The 300-epoch time is a linear extrapolation, not an accuracy or
  convergence experiment.
- The original graph has two asymmetric entries; Aᵀ is explicitly constructed
  for backward. Training time includes four SpMM calls (128, 41, 41, 128 columns),
  dense GEMM, nonlinearities, loss, gradients, Adam, and node-order restoration.
  Graph construction, normalization, transpose construction, validation, and
  checkpoints are excluded.

Data source: [original README](https://github.com/xiexi51/ICCAD-Accel-GCN).
Algorithm background: [Accel-GCN paper](https://xiexi51.github.io/assets/pdf/AccelGCN.pdf).
Based on upstream commit `8c27e74289f1848afddd637bfe7c6a54f6781dcd`.
File checksums are recorded in `results/data_sha256.txt`.
