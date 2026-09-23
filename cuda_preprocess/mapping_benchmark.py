"""Paired comparison: same stable row order, math, launch and output clearing."""
import json
import argparse
import os
import random
import torch
import numpy as np
from backend import Preprocessor, CuSparse, spmm, spmm_mapped
from benchmark import ROOT, bench, wall_bench, stats

parser=argparse.ArgumentParser()
parser.add_argument('--output',default='results/mapping_benchmark.json')
args=parser.parse_args()
torch.manual_seed(123)
torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32=False
hp=np.fromfile(ROOT/'graphs/reddit.dgl.graph.ptrdump',np.int32)
hi=np.fromfile(ROOT/'graphs/reddit.dgl.graph.edgedump',np.int32)
n,e=len(hp)-1,len(hi)
p,i=torch.from_numpy(hp).cuda(),torch.from_numpy(hi).cuda()
values=torch.ones(e,device='cuda'); sorted_values=torch.empty_like(values)
physical=Preprocessor(n,e)
mapped=Preprocessor(n,e,mapping_only=True)
physical.full(p,i,values,sorted_values)
mapped.mapping(p)
count=mapped.count.item()
assert mapped.indices is None and physical.count.item()==count
torch.testing.assert_close(mapped.perm,physical.perm,rtol=0,atol=0)
torch.testing.assert_close(mapped.rowptr,physical.rowptr,rtol=0,atol=0)
torch.testing.assert_close(mapped.meta[:count],physical.meta[:count],rtol=0,atol=0)
shipped=np.fromfile(ROOT/'block_level_meta/reddit.dgl.block4',np.int32).reshape(-1,4)
np.testing.assert_array_equal(mapped.meta[:count].cpu().numpy(),shipped)
identity=torch.arange(n,device='cuda',dtype=torch.int32)
inverse=torch.argsort(mapped.perm.long())
x=torch.randn(n,128,device='cuda'); out=torch.empty_like(x); gathered=torch.empty_like(x)
result={'gpu':torch.cuda.get_device_name(),'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'n':n,'nnz':e,'cols':128,'dtype':'float32','edge_values':'ones, explicitly loaded in every variant',
        'comparison':'identical stable permutation, edge loop, block partition, 384 threads and shared memory',
        'output_zero_included':True,'metadata_exact':True,'mapping_allocates_no_edge_array':True,
        'avoided_index_copy_mib':e*4/2**20,'avoided_index_and_weight_copy_mib':e*8/2**20,
        'row_mapping_mib':n*4/2**20,'virtual_rowptr_mib':(n+1)*4/2**20}
def save():
    (ROOT/args.output).write_text(json.dumps(result,indent=2))

def physical_sorted():
    return spmm(physical.meta,count,physical.indices,sorted_values,x,out)
def mapped_sorted():
    return spmm_mapped(mapped.meta,count,mapped.perm,p,mapped.rowptr,i,values,x,out)
def identity_sorted():
    # Same mapped kernel, but already-materialized CSR and identity row mapping.
    return spmm_mapped(physical.meta,count,identity,physical.rowptr,physical.rowptr,
                       physical.indices,sorted_values,x,out)
def mapped_original():
    return spmm_mapped(mapped.meta,count,mapped.perm,p,mapped.rowptr,i,values,x,out,original_output=True)
def physical_original():
    physical_sorted()
    torch.index_select(out,0,inverse,out=gathered)
    return gathered

cs=CuSparse(p,i,values,x,out)
reference=torch.empty_like(x); cs.run(x,reference)
sorted_reference=reference[mapped.perm.long()]
for name,fn,ref in [('physical_sorted',physical_sorted,sorted_reference),
                    ('mapping_sorted',mapped_sorted,sorted_reference),
                    ('identity_mapping_sorted',identity_sorted,sorted_reference),
                    ('mapping_original',mapped_original,reference),
                    ('physical_plus_gather',physical_original,reference)]:
    actual=fn()
    torch.testing.assert_close(actual,ref,rtol=5e-4,atol=5e-3)
    print('validated',name,flush=True)
result['all_variants_match_cusparse']=True
result['preprocessing']={
    'mapping_only_gpu':bench(lambda:mapped.mapping(p),repeat=100),
    'mapping_only_wall':wall_bench(lambda:mapped.mapping(p)),
    'mapping_only_wall_with_host_count':wall_bench(lambda:(mapped.mapping(p),mapped.count.item())),
    'physical_indices_gpu':bench(lambda:physical.full(p,i),repeat=10),
    'physical_indices_and_weights_gpu':bench(lambda:physical.full(p,i,values,sorted_values),repeat=10)}
save()
print('preprocessing',result['preprocessing'],flush=True)

variants={'physical_sorted':physical_sorted,'mapping_sorted':mapped_sorted,
          'identity_mapping_sorted':identity_sorted,'mapping_original':mapped_original,
          'physical_plus_gather':physical_original}
for fn in variants.values():
    for _ in range(5):fn()
torch.cuda.synchronize()
# Paired, shuffled ordering controls drift in clocks/temperature.
samples={k:[] for k in variants}; orders=[]
rng=random.Random(321)
for sample in range(15):
    order=list(variants); rng.shuffle(order); orders.append(order)
    for name in order:
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10): variants[name]()
        end.record(); end.synchronize()
        samples[name].append(start.elapsed_time(end)/10)
result['spmm']={k:stats(v) for k,v in samples.items()}
result['sample_orders']=orders
result['repeats_per_sample']=10
result['paired_mapping_penalty_percent']=stats([
    (m/p-1)*100 for m,p in zip(samples['mapping_sorted'],samples['physical_sorted'])])
# Name the units correctly; the generic stats helper normally reports ms.
result['paired_mapping_penalty_percent']={k.replace('_ms',''):v for k,v in result['paired_mapping_penalty_percent'].items()}
save()
for name,value in result['spmm'].items():print(name,value['median_ms'],flush=True)
print('paired_mapping_penalty_percent',result['paired_mapping_penalty_percent'],flush=True)
cs.close(); mapped.close(); physical.close()
