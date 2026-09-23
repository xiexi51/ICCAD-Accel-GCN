# Reddit CUDA preprocessing measurements

Measured on 2026-09-19, physical **GPU 2: RTX A6000 48 GB**, UUID `GPU-38e64350-832b-f88f-09db-b9408f6a4514`. Original graph: **232,965 nodes, 114,615,891 nonzeros**. SpMM: **128 columns, FP32**.

**CUDA preprocessing matching the original Python script's scope takes 0.0417 ms. Full preprocessing from original CSR, including stable sorting and copying edge indices/FP32 weights, takes 3.878 ms. One Accel-GCN SpMM takes 68.198 ms.**

## Main results

| Operation | Median time | Relative to one Accel SpMM |
|---|---:|---:|
| CUDA `.new_indptr → .block4`, GPU-resident input/output | 0.0417 ms | 0.0612% |
| Full CUDA preprocessing: sorting + index copy + metadata | 1.907 ms | 2.80% |
| Full CUDA preprocessing, also copying FP32 weights | 3.878 ms | 5.69% |
| Accel-GCN SpMM, including required output zeroing | 68.198 ms | 100% |
| cuSPARSE SpMM, original CSR | 72.717 ms | — |
| cuSPARSE SpMM, the same sorted CSR | 67.268 ms | — |

This comparison focuses on preprocessing cost. On this configuration, cuSPARSE with sorted CSR is slightly faster than the Accel kernel; these results do not imply that Accel outperforms every cuSPARSE configuration.

## Comparison with 300 training epochs

A full-batch, two-layer **602→128→41** GCN uses the real Reddit topology, random features/labels, standard self-loops and `D^-1/2 (A+I) D^-1/2` normalization, ReLU, dropout=0.5, cross-entropy, Adam, and FP32 with TF32 disabled. The training graph has 114,848,856 nonzeros. Each epoch includes forward/backward, four SpMM calls (128, 41, 41, 128 columns), dense matrix multiplication, and optimizer steps. The Accel path includes a gather restoring node order.

| Implementation | Wall time per epoch | Extrapolated 300 epochs | Peak PyTorch allocation |
|---|---:|---:|---:|
| Accel-GCN | 196.99 ms | **59.10 s** | 5.09 GiB |
| cuSPARSE, original node order | 209.65 ms | **62.89 s** | 5.09 GiB |

After five warmup epochs, 20 epochs were measured; the median was multiplied by 300. **The experiment did not run 300 epochs or evaluate accuracy/convergence with real features and labels.** It excludes downloads, disk I/O, normalization, self-loop/transpose construction, checkpoints, and extra validation. Memory includes resident A/Aᵀ, sorted copies, and work buffers; PyTorch statistics exclude the CUDA context and small direct library allocations. The workload fits the 48 GB GPU.

One full preprocessing pass (3.878 ms) is **0.0066%** of the estimated 59.10 s training time. Backward explicitly uses Aᵀ, whose full CUDA preprocessing takes 3.902 ms. Preprocessing both A and Aᵀ once totals approximately 7.780 ms, or **0.0132%** of 300 epochs. Static graphs reuse metadata across epochs.

## Transfers, allocations, and CPU baselines

| Operation | Wall time |
|---|---:|
| Metadata hot path: host launch + GPU + synchronization | 0.0546 ms |
| Metadata + retrieving block count on the CPU | 0.0746 ms |
| Uploading rowptr from CPU + metadata generation | 0.1591 ms |
| First metadata call in this process, including workspace allocation | 3.293 ms |
| Full weighted preprocessing, fresh workspace, GPU-resident input | 4.200 ms |
| Full unweighted preprocessing + original CSR upload | 22.763 ms |
| Full weighted preprocessing + original CSR/weight upload | 47.743 ms |
| Unmodified Python script, including file I/O and SciPy/NumPy allocations | 2330.97 ms |
| Equivalent Python reference, compute only; not a line-by-line profile of the original script | 344.82 ms |

H2D uses pageable CPU memory and preallocated output buffers; disk I/O is excluded. Fresh-workspace tests use an already warmed CUDA runtime and allocator. Input upload also initializes CUDA before the first metadata call, so that measurement is not process-start-to-result latency. CPU script I/O and GPU-resident measurements have different scopes; their ratio is not a pure compute speedup.

## Correctness and implementation scope

- GPU output contains **363,632 int4 records**, **byte-identical** to the unmodified script and shipped `.block4`. SHA-256: `6396bf7741f8b7a1a129c2f09185bad4aaee6f27e96ebe5adc5a2b19c8713fb7`.
- Full stable sorting was checked row by row against original CSR, and the new rowptr matches the downloaded file. Source code for the downloaded `.new_indices` ordering is unavailable. The new stable sort does not produce byte-identical edge ordering, but produces identical `.block4` and correctly restores node order.
- The original degree=192 strategy boundary, zero-degree skipping, and degree>384 splitting rules are preserved.
- All 13 boundary/random tests passed. CUDA Compute Sanitizer memcheck reported **0 errors**. Every Reddit SpMM output element passed comparison with cuSPARSE; maximum absolute error before normalization was 0.00057983. Training forward/backward also passed numerical comparisons.
- The original graph has 2 asymmetric entries, so A=Aᵀ cannot be assumed. Backward constructs and preprocesses Aᵀ separately.
- Metadata generation requires only rowptr, with results and block count retained on GPU. O(E) edge copying for materialized CSR is reported separately from O(N + block_count) metadata generation.

CUDA event timing uses five warmups and the median of seven groups: 100 metadata calls, 10 full preprocessing calls, or 20 SpMM calls per group. No other compute process used GPU 2. Compilation: nvcc 12.2.140 / sm_86; PyTorch 2.2.1+cu121. The dynamically loaded cuSPARSE version query returns 12100 from the existing PyTorch environment; these results must not be labeled cuSPARSE 12.2 measurements.

Implementation and commands: [README.md](README.md). Raw samples: [benchmark.json](../results/benchmark.json), [extra_timings.json](../results/extra_timings.json), [training.json](../results/training.json). Sources: [original repository](https://github.com/xiexi51/ICCAD-Accel-GCN), [paper](https://xiexi51.github.io/assets/pdf/AccelGCN.pdf).
