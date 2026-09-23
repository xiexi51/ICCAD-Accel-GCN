"""Render the recorded measurements without rerunning GPU work."""
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
b=json.loads((root/'results/benchmark.json').read_text())
t=json.loads((root/'results/training.json').read_text())
x=json.loads((root/'results/extra_timings.json').read_text())
m=lambda key:b['timing'][key]['median_ms']
s=m('spmm_accel_128_zero_included_gpu')
p=m('partition_gpu')
f=m('full_reorder_indices_values_and_partition_gpu')
train=t['accel']['estimated_300_epochs_s']
text=f'''# Reddit CUDA preprocessing measurements

Measured on 2026-09-19, physical **GPU 2: RTX A6000 48 GB**, UUID `GPU-38e64350-832b-f88f-09db-b9408f6a4514`. Original graph: **{b['graph']['n']:,} nodes, {b['graph']['nnz']:,} nonzeros**. SpMM: **128 columns, FP32**.

**CUDA preprocessing matching the original Python script's scope takes {p:.4f} ms. Full preprocessing from original CSR, including stable sorting and copying edge indices/FP32 weights, takes {f:.3f} ms. One Accel-GCN SpMM takes {s:.3f} ms.**

## Main results

| Operation | Median time | Relative to one Accel SpMM |
|---|---:|---:|
| CUDA `.new_indptr → .block4`, GPU-resident input/output | {p:.4f} ms | {p/s*100:.4f}% |
| Full CUDA preprocessing: sorting + index copy + metadata | {m('full_reorder_indices_and_partition_gpu'):.3f} ms | {m('full_reorder_indices_and_partition_gpu')/s*100:.2f}% |
| Full CUDA preprocessing, also copying FP32 weights | {f:.3f} ms | {f/s*100:.2f}% |
| Accel-GCN SpMM, including required output zeroing | {s:.3f} ms | 100% |
| cuSPARSE SpMM, original CSR | {m('spmm_cusparse_original_128_gpu'):.3f} ms | — |
| cuSPARSE SpMM, the same sorted CSR | {m('spmm_cusparse_sorted_128_gpu'):.3f} ms | — |

This comparison focuses on preprocessing cost. On this configuration, cuSPARSE with sorted CSR is slightly faster than the Accel kernel; these results do not imply that Accel outperforms every cuSPARSE configuration.

## Comparison with 300 training epochs

A full-batch, two-layer **602→128→41** GCN uses the real Reddit topology, random features/labels, standard self-loops and `D^-1/2 (A+I) D^-1/2` normalization, ReLU, dropout=0.5, cross-entropy, Adam, and FP32 with TF32 disabled. The training graph has {t['training_nnz']:,} nonzeros. Each epoch includes forward/backward, four SpMM calls (128, 41, 41, 128 columns), dense matrix multiplication, and optimizer steps. The Accel path includes a gather restoring node order.

| Implementation | Wall time per epoch | Extrapolated 300 epochs | Peak PyTorch allocation |
|---|---:|---:|---:|
| Accel-GCN | {t['accel']['median_ms']:.2f} ms | **{train:.2f} s** | {t['accel']['peak_torch_allocated_gib']:.2f} GiB |
| cuSPARSE, original node order | {t['cusparse']['median_ms']:.2f} ms | **{t['cusparse']['estimated_300_epochs_s']:.2f} s** | {t['cusparse']['peak_torch_allocated_gib']:.2f} GiB |

After five warmup epochs, {t['epochs_measured']} epochs were measured; the median was multiplied by 300. **The experiment did not run 300 epochs or evaluate accuracy/convergence with real features and labels.** It excludes downloads, disk I/O, normalization, self-loop/transpose construction, checkpoints, and extra validation. Memory includes resident A/Aᵀ, sorted copies, and work buffers; PyTorch statistics exclude the CUDA context and small direct library allocations. The workload fits the 48 GB GPU.

One full preprocessing pass ({f:.3f} ms) is **{f/(train*1000)*100:.4f}%** of the estimated {train:.2f} s training time. Backward explicitly uses Aᵀ, whose full CUDA preprocessing takes {t['transpose_full_preprocess_gpu']['median_ms']:.3f} ms. Preprocessing both A and Aᵀ once totals approximately {f+t['transpose_full_preprocess_gpu']['median_ms']:.3f} ms, or **{(f+t['transpose_full_preprocess_gpu']['median_ms'])/(train*1000)*100:.4f}%** of 300 epochs. Static graphs reuse metadata across epochs.

## Transfers, allocations, and CPU baselines

| Operation | Wall time |
|---|---:|
| Metadata hot path: host launch + GPU + synchronization | {m('partition_wall'):.4f} ms |
| Metadata + retrieving block count on the CPU | {x['partition_plus_host_count_wall']['median_ms']:.4f} ms |
| Uploading rowptr from CPU + metadata generation | {m('partition_with_rowptr_h2d_wall'):.4f} ms |
| First metadata call in this process, including workspace allocation | {m('partition_first_with_workspace_wall'):.3f} ms |
| Full weighted preprocessing, fresh workspace, GPU-resident input | {x['full_weighted_fresh_workspace_resident_input_wall']['median_ms']:.3f} ms |
| Full unweighted preprocessing + original CSR upload | {x['full_unweighted_with_pageable_csr_h2d_wall']['median_ms']:.3f} ms |
| Full weighted preprocessing + original CSR/weight upload | {x['full_weighted_with_pageable_csr_h2d_wall']['median_ms']:.3f} ms |
| Unmodified Python script, including file I/O and SciPy/NumPy allocations | {m('python_original_script_io_included'):.2f} ms |
| Equivalent Python reference, compute only; not a line-by-line profile of the original script | {m('python_reference_compute_only'):.2f} ms |

H2D uses pageable CPU memory and preallocated output buffers; disk I/O is excluded. Fresh-workspace tests use an already warmed CUDA runtime and allocator. Input upload also initializes CUDA before the first metadata call, so that measurement is not process-start-to-result latency. CPU script I/O and GPU-resident measurements have different scopes; their ratio is not a pure compute speedup.

## Correctness and implementation scope

- GPU output contains **{b['validation']['block_count']:,} int4 records**, **byte-identical** to the unmodified script and shipped `.block4`. SHA-256: `{b['validation']['metadata_sha256']}`.
- Full stable sorting was checked row by row against original CSR, and the new rowptr matches the downloaded file. Source code for the downloaded `.new_indices` ordering is unavailable. The new stable sort does not produce byte-identical edge ordering, but produces identical `.block4` and correctly restores node order.
- The original degree=192 strategy boundary, zero-degree skipping, and degree>384 splitting rules are preserved.
- All 13 boundary/random tests passed. CUDA Compute Sanitizer memcheck reported **0 errors**. Every Reddit SpMM output element passed comparison with cuSPARSE; maximum absolute error before normalization was {b['validation']['reddit_spmm_max_abs_error']:.8f}. Training forward/backward also passed numerical comparisons.
- The original graph has {t['asymmetric_entries']} asymmetric entries, so A=Aᵀ cannot be assumed. Backward constructs and preprocesses Aᵀ separately.
- Metadata generation requires only rowptr, with results and block count retained on GPU. O(E) edge copying for materialized CSR is reported separately from O(N + block_count) metadata generation.

CUDA event timing uses five warmups and the median of seven groups: 100 metadata calls, 10 full preprocessing calls, or 20 SpMM calls per group. No other compute process used GPU 2. Compilation: nvcc 12.2.140 / sm_86; PyTorch {b['environment']['torch']}. The dynamically loaded cuSPARSE version query returns {x['library_versions']['cusparseGetVersion']} from the existing PyTorch environment; these results must not be labeled cuSPARSE 12.2 measurements.

Implementation and commands: [README.md](README.md). Raw samples: [benchmark.json](../results/benchmark.json), [extra_timings.json](../results/extra_timings.json), [training.json](../results/training.json). Sources: [original repository](https://github.com/xiexi51/ICCAD-Accel-GCN), [paper](https://xiexi51.github.io/assets/pdf/AccelGCN.pdf).
'''
(root/'cuda_preprocess/RESULTS.md').write_text(text)
print(text)
