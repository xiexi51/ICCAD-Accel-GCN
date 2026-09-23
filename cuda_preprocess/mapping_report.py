"""Render mapping measurements after both independent runs have completed."""
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
a=json.loads((root/'results/mapping_benchmark.json').read_text())
b=json.loads((root/'results/mapping_benchmark_repeat.json').read_text())
assert a['all_variants_match_cusparse'] and b['all_variants_match_cusparse']
assert 'ERROR SUMMARY: 0 errors' in (root/'results/mapping_sanitizer.log').read_text()
ms=lambda r,k:r['spmm'][k]['median_ms']
pre=lambda r,k:r['preprocessing'][k]['median_ms']
p,m=ms(a,'physical_sorted'),ms(a,'mapping_sorted')
delta=m-p
rows='\n'.join(f"| {label} | {ms(a,key):.4f} ms | {ms(b,key):.4f} ms |" for key,label in [
    ('physical_sorted','Materialized CSR, original SpMM kernel'),
    ('mapping_sorted','Original CSR, row-mapping kernel'),
    ('identity_mapping_sorted','Materialized CSR, same mapping kernel + identity mapping')])
text=f'''# Row mapping only: Reddit SpMM comparison

**Edge indices and weights do not need to move. The row-mapping implementation passes correctness checks. No practically meaningful SpMM performance loss was observed on Reddit with 128 columns on GPU 2.**

Hardware and input match the previous experiment: RTX A6000 48 GB, physical GPU 2, 232,965 nodes / 114,615,891 edges, FP32. Original edge weights are all one, but all variants still load FP32 values, preserving the original computation. The CUDA implementation is `spmm_kernel_mapped` in `preprocess.cu`; upstream `spmm_accel.cu` is unchanged.

## Single SpMM

The comparison uses **the same stable row permutation, metadata, input X, and degree-sorted output order**. Every call includes output zeroing. Input and output buffers are preallocated and GPU-resident.

| Implementation | First-run median | Independent second-run median |
|---|---:|---:|
{rows}

In the first run, mapping differs from materialized CSR by **{delta*1000:.2f} µs**, or **{(m/p-1)*100:+.4f}%**. The second-run difference is **{(ms(b,'mapping_sorted')/ms(b,'physical_sorted')-1)*100:+.4f}%**. These small differences indicate essentially equal performance, not a clear mapping slowdown.

Each run has 15 sample groups and 10 calls per variant per group. Variant order is randomized within each group to reduce temperature/frequency drift bias. CUDA events measure time after five warmups per variant. First-run paired differences range from {a['paired_mapping_penalty_percent']['min']:+.4f}% to {a['paired_mapping_penalty_percent']['max']:+.4f}%. Identity mapping is an auxiliary control: the same indirect-addressing kernel operates on materialized CSR. It helps assess addressing and layout together; subtracting two timings is not a rigorous hardware-overhead decomposition.

The earlier 68.20 ms result used downloaded `.new_indices`. Both variants here use the same **stable sorting**, so the concurrent comparison above is the basis for the conclusion.

## Preprocessing

| Path, starting from original CSR rowptr | First-run GPU time | Second-run GPU time |
|---|---:|---:|
| Stable sort + row mapping + virtual rowptr + block4, no edge copying | **{pre(a,'mapping_only_gpu'):.4f} ms** | {pre(b,'mapping_only_gpu'):.4f} ms |
| Stable sort + edge-index copy + block4 | {pre(a,'physical_indices_gpu'):.4f} ms | {pre(b,'physical_indices_gpu'):.4f} ms |
| Stable sort + edge-index/weight copy + block4 | {pre(a,'physical_indices_and_weights_gpu'):.4f} ms | {pre(b,'physical_indices_and_weights_gpu'):.4f} ms |

In the first run, full preprocessing without edge copying costs **{pre(a,'mapping_only_gpu')/m*100:.3f}%** of one SpMM and is **{pre(a,'physical_indices_gpu')/pre(a,'mapping_only_gpu'):.1f}×** faster than preprocessing with an edge-index copy. Host launch + GPU + synchronization takes {pre(a,'mapping_only_wall'):.4f} ms; including retrieval of the block count takes {pre(a,'mapping_only_wall_with_host_count'):.4f} ms. Disk I/O, H2D, and workspace allocation are excluded.

This scope differs from the earlier 0.0417 ms result: that result generated block4 from **already sorted rowptr**, whereas this measurement starts from **original rowptr** and includes degree computation and stable sorting.

Mapping-only preprocessing allocates no nnz-sized output buffer, avoiding **{a['avoided_index_copy_mib']:.2f} MiB** of copied indices. A general weighted graph also avoids the same-sized weight copy, saving **{a['avoided_index_and_weight_copy_mib']:.2f} MiB** in total. The row permutation and virtual rowptr each use approximately {a['row_mapping_mib']:.3f} MiB, plus block metadata and O(N) sort/scan workspace. No per-edge mapping is needed.

## Direct output in original node order

Training commonly requires original node order. The new kernel writes directly to `perm[sorted_row]` without a separate gather:

| Original-order output | First run | Second run |
|---|---:|---:|
| Original kernel on materialized CSR + inverse-permutation gather | {ms(a,'physical_plus_gather'):.4f} ms | {ms(b,'physical_plus_gather'):.4f} ms |
| Row-mapping kernel, direct original-node output | {ms(a,'mapping_original'):.4f} ms | {ms(b,'mapping_original'):.4f} ms |

The first-run reduction is {(ms(a,'physical_plus_gather')-ms(a,'mapping_original')):.4f} ms ({(1-ms(a,'mapping_original')/ms(a,'physical_plus_gather'))*100:.3f}%). This measures one SpMM plus output reordering; full training was not remeasured in this experiment.

## Addressing changes

```cpp
original_row = perm[sorted_row];
edge_begin = original_rowptr[original_row];
if (degree > 384)
    edge_begin += block_loc - sorted_rowptr[block_row_begin];

// Traverse the contiguous neighbor array in original CSR.
nz = edge_begin + warp_local_col + j;
acc += original_values[nz] * X[original_indices[nz], feature];
```

The `perm → original_rowptr` lookup occurs outside the neighbor loop, once per logical row/warp work segment rather than once per edge. High-degree rows retain the original metadata partitioning. Different rows within a block no longer require adjacent edge storage; each individual row's neighbors remain contiguous. Output can use either original node order or degree-sorted order.

## Correctness and reproduction

- Mapping-only permutations, virtual rowptr, and block4 match the materialized variant exactly; Reddit block4 also matches the shipped file.
- Thirteen boundary/random cases cover zero degrees, the 192/384 strategy boundaries, large rows spanning blocks, random weights, and 32/41/128 columns.
- Every output element in all five timed Reddit paths passed comparison with cuSPARSE.
- CUDA Compute Sanitizer memcheck: **0 errors**, recorded in `results/mapping_sanitizer.log`.

```bash
cd /home/xix22010/py_projects3/accel_gcn
export CUDA_VISIBLE_DEVICES=2
PYTHON=/home/xix22010/anaconda3/envs/torch2/bin/python
bash cuda_preprocess/build.sh
"$PYTHON" cuda_preprocess/validate.py
"$PYTHON" cuda_preprocess/mapping_benchmark.py
"$PYTHON" cuda_preprocess/mapping_benchmark.py --output results/mapping_benchmark_repeat.json
/usr/local/cuda-12.2/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \\
  "$PYTHON" cuda_preprocess/validate.py > results/mapping_sanitizer.log 2>&1
"$PYTHON" cuda_preprocess/mapping_report.py
```

API examples: [README.md](README.md). Raw samples: [first run](../results/mapping_benchmark.json), [second run](../results/mapping_benchmark_repeat.json).
'''
(root/'cuda_preprocess/MAPPING_RESULTS.md').write_text(text)
print(text)
