"""Graph-selectable mapping preprocessing / Accel / cuSPARSE comparison."""
import argparse
import ctypes as C
import hashlib
import json
import os
import random
import numpy as np
import torch
from backend import Preprocessor, CuSparse, spmm, spmm_mapped, lib
from benchmark import ROOT, bench, wall_bench, stats
from reference import partition

parser=argparse.ArgumentParser()
parser.add_argument('--graph',default='collab')
parser.add_argument('--cols',type=int,default=128)
args=parser.parse_args()
torch.manual_seed(123); torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32=False
files=[ROOT/'graphs'/f'{args.graph}.graph.{suffix}' for suffix in ('ptrdump','edgedump')]
hp,hi=[np.fromfile(f,np.int32) for f in files]
n,e=len(hp)-1,len(hi)
assert hp[0]==0 and hp[-1]==e and np.all(np.diff(hp)>=0)
assert np.all((hi>=0)&(hi<n))
p,i=torch.from_numpy(hp).cuda(),torch.from_numpy(hi).cuda()
v=torch.ones(e,device='cuda')
m=Preprocessor(n,e,mapping_only=True); m.mapping(p)
blocks=m.count.item()
assert m.indices is None
perm=np.argsort(np.diff(hp),kind='stable').astype(np.int32)
expected_ptr=np.r_[0,np.cumsum(np.diff(hp)[perm])].astype(np.int32)
np.testing.assert_array_equal(m.perm.cpu().numpy(),perm)
np.testing.assert_array_equal(m.rowptr.cpu().numpy(),expected_ptr)
meta=m.meta[:blocks].cpu().numpy()
np.testing.assert_array_equal(meta,partition(expected_ptr))
shipped=np.fromfile(ROOT/'block_level_meta'/f'{args.graph}.block4',np.int32).reshape(-1,4)
np.testing.assert_array_equal(meta,shipped)

# Physical layout is only an additional control, outside mapping preprocessing.
physical=Preprocessor(n,e); sv=torch.empty_like(v)
physical.full(p,i,v,sv)
x=torch.randn(n,args.cols,device='cuda'); y=torch.empty_like(x); reference=torch.empty_like(x)
cs=CuSparse(p,i,v,x,y)
cs_sorted=CuSparse(physical.rowptr,physical.indices,sv,x,y)
cs.run(x,reference)
sorted_reference=reference[m.perm.long()]
variants={
    'accel_mapping_original_output':lambda:spmm_mapped(m.meta,blocks,m.perm,p,m.rowptr,i,v,x,y,original_output=True),
    'accel_mapping_sorted_output':lambda:spmm_mapped(m.meta,blocks,m.perm,p,m.rowptr,i,v,x,y),
    'accel_physical_sorted_output':lambda:spmm(physical.meta,blocks,physical.indices,sv,x,y),
    'cusparse_original':lambda:cs.run(x,y),
    'cusparse_sorted':lambda:cs_sorted.run(x,y)}
errors={}
for name,fn in variants.items():
    ref=sorted_reference if 'sorted' in name else reference
    actual=fn()
    torch.testing.assert_close(actual,ref,rtol=5e-4,atol=5e-3)
    errors[name]=float((actual-ref).abs().max())
    print('validated',name,'max_abs_error',errors[name],flush=True)
version=C.c_int(); version_handle=C.c_void_p()
assert lib.cusparseCreate(C.byref(version_handle))==0
assert lib.cusparseGetVersion(version_handle,C.byref(version))==0
assert lib.cusparseDestroy(version_handle)==0
result={'graph':args.graph,'n':n,'nnz':e,'cols':args.cols,'max_degree':int(np.diff(hp).max()),
        'gpu':torch.cuda.get_device_name(),'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'dtype':'float32','edge_values':'ones, explicitly loaded','cusparse_version':version.value,
        'cusparse_algorithm':'CUSPARSE_SPMM_ALG_DEFAULT','blocks':blocks,
        'metadata_matches_python_reference_and_shipped_file':True,'mapping_no_edge_allocation':True,
        'max_abs_errors':errors,'all_variants_validated':True,
        'input_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
def save():
    (ROOT/'results'/f'{args.graph}_{args.cols}_benchmark.json').write_text(json.dumps(result,indent=2))
for fn in variants.values():
    for _ in range(20):fn()
torch.cuda.synchronize()
result['preprocessing']={
    'mapping_only_gpu':bench(lambda:m.mapping(p),repeat=100),
    'mapping_only_wall':wall_bench(lambda:m.mapping(p)),
    'mapping_plus_host_count_wall':wall_bench(lambda:(m.mapping(p),m.count.item()))}
save()
print('mapping_only_gpu',result['preprocessing']['mapping_only_gpu']['median_ms'],flush=True)
samples={name:[] for name in variants}
rng=random.Random(456); orders=[]
for sample in range(15):
    order=list(variants); rng.shuffle(order); orders.append(order)
    for name in order:
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):variants[name]()
        end.record(); end.synchronize()
        samples[name].append(start.elapsed_time(end)/100)
result['spmm']={name:stats(times) for name,times in samples.items()}
result['measurement']={'warmup_per_variant':20,'sample_groups':15,'calls_per_group':100,
                       'orders':orders,'output_zero_in_accel':True,
                       'input_output_resident':True,'excludes':'allocation, I/O, H2D, CSR materialization'}
save()
for name,measurement in result['spmm'].items():
    print(name,measurement['median_ms'],flush=True)

ms=lambda name:result['spmm'][name]['median_ms']
prep=result['preprocessing']['mapping_only_gpu']['median_ms']
report=f'''# {args.graph}: mapping-only preprocessing and SpMM

GPU 2: {result['gpu']}, FP32, {args.cols} columns. Original CSR from the official 18graphs.tar.gz: {n:,} nodes, {e:,} nonzeros, maximum degree {result['max_degree']}.

| Operation | Median time |
|---|---:|
| Full mapping-only preprocessing: degree + stable sort + virtual rowptr + block4 | **{prep:.4f} ms** |
| Accel-GCN SpMM: row mapping, direct original-node output | **{ms('accel_mapping_original_output'):.4f} ms** |
| cuSPARSE SpMM: original CSR, original-node output | **{ms('cusparse_original'):.4f} ms** |

Both main SpMM variants compute the same A×X and return the same node order. Accel includes required output zeroing; cuSPARSE uses beta=0. Preprocessing costs **{prep/ms('accel_mapping_original_output')*100:.2f}%** of one Accel SpMM. The cuSPARSE/Accel time ratio is **{ms('cusparse_original')/ms('accel_mapping_original_output'):.3f}×** on this configuration.

Preprocessing neither copies nor allocates duplicate edge indices or weights. It includes stable sorting, unlike block4 generation from already sorted rowptr. Inputs and outputs are GPU-resident; allocation, disk I/O, and H2D are excluded. Host launch + GPU + synchronization takes {result['preprocessing']['mapping_only_wall']['median_ms']:.4f} ms; including retrieval of the block count takes {result['preprocessing']['mapping_plus_host_count_wall']['median_ms']:.4f} ms.

Additional controls with the same stable degree-sorted output order:

| Operation | Median time |
|---|---:|
| Accel, row mapping, sorted-row output | {ms('accel_mapping_sorted_output'):.4f} ms |
| Accel, materialized CSR, original kernel | {ms('accel_physical_sorted_output'):.4f} ms |
| cuSPARSE, materialized CSR | {ms('cusparse_sorted'):.4f} ms |

The time difference between the two Accel variants with identical sorted output order is {(ms('accel_mapping_sorted_output')/ms('accel_physical_sorted_output')-1)*100:+.2f}%. The control CSR is materialized separately before timing and is not part of mapping-only preprocessing.

The recorded cuSPARSE version query is **{version.value} (12.1.0 in this experiment)**; the algorithm is `CUSPARSE_SPMM_ALG_DEFAULT`. Each SpMM variant uses 20 warmups, followed by 15 groups of 100 calls, with randomized variant order within each group. CUDA events measure time; the report gives the median of group means. Preprocessing uses five warmups and seven groups of 100 calls. Edge weights are all one, but implementations still load FP32 values. X is a random dense matrix with a fixed seed.

Correctness: the permutation and virtual rowptr match Python stable sorting. All {blocks:,} int4 metadata records match the Python reference and shipped file exactly. Every output element in all five SpMM variants passed comparison with cuSPARSE.

```bash
cd /home/xix22010/py_projects3/accel_gcn
PYTHON=/home/xix22010/anaconda3/envs/torch2/bin/python
"$PYTHON" cuda_preprocess/fetch_graph.py {args.graph}
CUDA_VISIBLE_DEVICES=2 "$PYTHON" cuda_preprocess/graph_benchmark.py --graph {args.graph} --cols {args.cols}
```

Raw samples and checksums: [JSON](../results/{args.graph}_{args.cols}_benchmark.json). This experiment measures preprocessing and SpMM, not training.
'''
(ROOT/'cuda_preprocess'/f'{args.graph.upper()}_RESULTS.md').write_text(report)
cs.close(); cs_sorted.close(); m.close(); physical.close()
