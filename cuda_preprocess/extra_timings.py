"""Supplement resident-GPU times with host-transfer and fresh-workspace costs."""
import ctypes as C
import json
import time
import numpy as np
import torch
from backend import Preprocessor, lib
from benchmark import ROOT, wall_bench, stats

torch.set_num_threads(8)
hp=torch.from_numpy(np.fromfile(ROOT/'graphs/reddit.dgl.graph.ptrdump',np.int32))
hi=torch.from_numpy(np.fromfile(ROOT/'graphs/reddit.dgl.graph.edgedump',np.int32))
p,i=hp.cuda(),hi.cuda()
n,e=len(p)-1,len(i)
v=torch.ones(e,device='cuda'); ov=torch.empty_like(v)
w=Preprocessor(n,e)
w.full(p,i,v,ov); torch.cuda.synchronize()
result={}
result['partition_plus_host_count_wall']=wall_bench(lambda:(w.partition(w.rowptr),w.count.item()),10)
result['full_unweighted_with_pageable_csr_h2d_wall']=wall_bench(lambda:(p.copy_(hp),i.copy_(hi),w.full(p,i)),5)
hv=torch.ones(e)
result['full_weighted_with_pageable_csr_h2d_wall']=wall_bench(lambda:(p.copy_(hp),i.copy_(hi),v.copy_(hv),w.full(p,i,v,ov)),5)
times=[]
for _ in range(10):
    torch.cuda.synchronize(); t=time.perf_counter()
    fresh=Preprocessor(n,e); fresh.full(p,i,v,ov); count=fresh.count.item()
    times.append((time.perf_counter()-t)*1000)
    fresh.close(); del fresh
result['full_weighted_fresh_workspace_resident_input_wall']=stats(times)
versions={'build_cuda_toolkit':'12.2.140 (nvcc default static cudart)'}
handle=C.c_void_p(); lib.cusparseCreate(C.byref(handle))
value=C.c_int(); lib.cusparseGetVersion(handle,C.byref(value)); versions['cusparseGetVersion']=value.value
lib.cusparseDestroy(handle)
result['library_versions']=versions
print(json.dumps(result,indent=2))
(ROOT/'results/extra_timings.json').write_text(json.dumps(result,indent=2))
w.close()
