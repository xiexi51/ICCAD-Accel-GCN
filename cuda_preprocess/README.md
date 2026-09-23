# CUDA preprocessing

`main.cu` calls the CUDA implementation directly. No Python runtime, CPU
preprocessing implementation, or offline metadata generation is required.
The main benchmark runs mapped Accel-GCN SpMM and cuSPARSE on the original CSR.
Legacy GNNAdvisor source is retained separately and is not part of this target.

## Pipeline

1. CUDA kernels calculate row degrees and initialize row IDs.
2. A stable CUB radix sort produces `perm[sorted_row] = original_row`.
3. A prefix scan produces virtual degree-sorted CSR row offsets.
4. Run detection, block counts, a prefix scan, and emission generate `int4`
   block records `[degree, row_begin, loc_begin, info]`.

Only mapping and scheduling metadata are generated. Edge indices and weights
remain in their original GPU buffers; there is no reordered edge allocation or
copy. SpMM resolves the original row outside the neighbor loop and writes the
output in original node order. Large rows use virtual offsets to locate their
384-edge segments. Output zeroing is included in each SpMM call.

Metadata retains the original partition semantics: stable ordering for equal
degrees, one row per block for degrees 192–384, 384-edge segments above 384,
and no blocks for zero-degree rows.

## Generation and caching

Default, or `--metadata-generate`: generate metadata with CUDA once per graph
per executable invocation, and reuse it across feature widths and SpMM calls.
No metadata file is read or written.

`--metadata-cache DIR`: load `DIR/GRAPH.agmeta` when valid. On a missing,
outdated, truncated, or checksum-invalid entry, generate on the GPU and save
it with an atomic rename. The last metadata mode option on the command line
wins. An inaccessible cache directory is reported as an error.

The cache stores the permutation, virtual row offsets, and block records.
Its header contains a format/algorithm version, node and edge counts, a
fingerprint of the original row offsets, the block count, payload size, and a
payload checksum. Changed row offsets invalidate the entry. Column IDs and
weights do not affect this metadata and are not cached. Cache files use native
little-endian int32/uint64 data; they are local generated artifacts, not a
portable or untrusted interchange format.

`metadata_setup_ms` is single-run wall time, including workspace allocation,
CUDA work and synchronization; cache mode also includes fingerprinting and
file transfers. It excludes the common CSR input upload and metadata output
buffer allocation. First-use CUDA module initialization can affect this time.
It is not the warmed preprocessing microbenchmark reported on the test branch.

`accel_ms` and `cusparse_ms` are mean CUDA event times after 20 warmup calls,
using 100 timed calls by default. Override with `--warmup` and `--iterations`.
Metadata setup, descriptor construction, allocations and validation are outside
the SpMM interval. Accel-GCN includes required output zeroing; cuSPARSE uses
`beta=0`. Both operate on FP32 features and unit edge weights in this driver.

## Native interface and constraints

`preprocess.h` declares the CUDA C interface. `ag_create` allocates a reusable
workspace; `ag_mapping` builds the mapping and virtual row offsets;
`ag_partition` builds block metadata. Calls enqueue on the supplied stream.
The caller obtains the block count with a device-to-host copy before SpMM.
Do not share a workspace between concurrent streams or destroy it before work
completes. `metadata.h` provides the executable's ownership and cache handling.

CSR indices and offsets must fit int32. The executable checks CSR offsets and
column bounds; dense element offsets must also fit int32. The default target
architecture is `sm_86`, configurable through CMake. CUDA/CUB and cuSPARSE are
required. The native executable has no Python dependency. Cache integration
tests are retained on the test branch.
