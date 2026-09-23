# collab: mapping-only preprocessing and SpMM

GPU 2: NVIDIA RTX A6000, FP32, 128 columns. Original CSR from the official 18graphs.tar.gz: 235,868 nodes, 2,358,104 nonzeros, maximum degree 671.

| Operation | Median time |
|---|---:|
| Full mapping-only preprocessing: degree + stable sort + virtual rowptr + block4 | **0.0959 ms** |
| Accel-GCN SpMM: row mapping, direct original-node output | **1.5151 ms** |
| cuSPARSE SpMM: original CSR, original-node output | **1.8194 ms** |

Both main SpMM variants compute the same A×X and return the same node order. Accel includes required output zeroing; cuSPARSE uses beta=0. Preprocessing costs **6.33%** of one Accel SpMM. The cuSPARSE/Accel time ratio is **1.201×** on this configuration.

Preprocessing neither copies nor allocates duplicate edge indices or weights. It includes stable sorting, unlike block4 generation from already sorted rowptr. Inputs and outputs are GPU-resident; allocation, disk I/O, and H2D are excluded. Host launch + GPU + synchronization takes 0.1101 ms; including retrieval of the block count takes 0.1277 ms.

Additional controls with the same stable degree-sorted output order:

| Operation | Median time |
|---|---:|
| Accel, row mapping, sorted-row output | 1.5081 ms |
| Accel, materialized CSR, original kernel | 1.4558 ms |
| cuSPARSE, materialized CSR | 1.6153 ms |

The time difference between the two Accel variants with identical sorted output order is +3.59%. The control CSR is materialized separately before timing and is not part of mapping-only preprocessing.

The recorded cuSPARSE version query is **12100 (12.1.0 in this experiment)**; the algorithm is `CUSPARSE_SPMM_ALG_DEFAULT`. Each SpMM variant uses 20 warmups, followed by 15 groups of 100 calls, with randomized variant order within each group. CUDA events measure time; the report gives the median of group means. Preprocessing uses five warmups and seven groups of 100 calls. Edge weights are all one, but implementations still load FP32 values. X is a random dense matrix with a fixed seed.

Correctness: the permutation and virtual rowptr match Python stable sorting. All 21,741 int4 metadata records match the Python reference and shipped file exactly. Every output element in all five SpMM variants passed comparison with cuSPARSE.

```bash
cd /home/xix22010/py_projects3/accel_gcn
PYTHON=/home/xix22010/anaconda3/envs/torch2/bin/python
"$PYTHON" cuda_preprocess/fetch_graph.py collab
CUDA_VISIBLE_DEVICES=2 "$PYTHON" cuda_preprocess/graph_benchmark.py --graph collab --cols 128
```

Raw samples and checksums: [JSON](../results/collab_128_benchmark.json). This experiment measures preprocessing and SpMM, not training.
