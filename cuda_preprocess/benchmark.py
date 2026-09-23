"""Run on physical GPU 2: CUDA_VISIBLE_DEVICES=2 python cuda_preprocess/benchmark.py."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import tempfile
import time
import numpy as np
import torch
from backend import Preprocessor, CuSparse, spmm
from reference import partition

ROOT = Path(__file__).resolve().parents[1]

def stats(samples):
    return dict(median_ms=float(np.median(samples)), min_ms=float(np.min(samples)),
                max_ms=float(np.max(samples)), samples_ms=samples)

def bench(fn, repeat=20, samples=7):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(samples):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            fn()
        end.record(); end.synchronize()
        times.append(start.elapsed_time(end)/repeat)
    return stats(times)

def wall_bench(fn, repeat=10):
    times=[]
    for _ in range(repeat):
        torch.cuda.synchronize(); t=time.perf_counter()
        fn(); torch.cuda.synchronize()
        times.append((time.perf_counter()-t)*1000)
    return stats(times)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results/benchmark.json')
    args=parser.parse_args()
    torch.manual_seed(123)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    result={'environment':{'gpu':torch.cuda.get_device_name(), 'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                           'torch':torch.__version__, 'torch_cuda':torch.version.cuda, 'tf32':False}, 'timing':{}}
    def save():
        (ROOT/args.output).write_text(json.dumps(result,indent=2))
    def log(key,value):
        result['timing'][key]=value; save()
        print(key, value['median_ms'] if 'median_ms' in value else value, flush=True)
    def load(ext):
        return np.fromfile(ROOT/'graphs'/('reddit.dgl.'+ext),dtype=np.int32)
    hp,hi,hnp,hni=load('graph.ptrdump'),load('graph.edgedump'),load('new_indptr'),load('new_indices')
    n,e=len(hp)-1,len(hi)
    assert hp[0]==0 and hp[-1]==e and hnp[-1]==e
    result['graph']={'n':n,'nnz':e,'col':128,'max_degree':int(np.diff(hp).max()),
                     'self_loops':None,'source':'upstream README 18graphs.tar.gz'}
    print('Graph',result['graph'],flush=True)
    p,i,np_,ni=[torch.from_numpy(a).cuda() for a in (hp,hi,hnp,hni)]
    torch.cuda.synchronize()
    t=time.perf_counter(); w=Preprocessor(n,e); w.partition(np_); count=w.count.item()
    log('partition_first_with_workspace_wall',dict(median_ms=(time.perf_counter()-t)*1000))
    meta=w.meta[:count].cpu().numpy()
    upstream=np.fromfile(ROOT/'block_level_meta/reddit.dgl.block4',np.int32).reshape(-1,4)
    np.testing.assert_array_equal(meta,upstream)
    # Execute the actual unmodified upstream script in an isolated directory.
    with tempfile.TemporaryDirectory() as temp:
        temp=Path(temp); (temp/'graphs').mkdir()
        for ext in ('config','new_indptr','new_indices'):
            (temp/'graphs'/('reddit.dgl.'+ext)).symlink_to(ROOT/'graphs'/('reddit.dgl.'+ext))
        cwd=Path.cwd()
        try:
            os.chdir(temp); t=time.perf_counter()
            runpy.run_path(str(ROOT/'block_level_partition.py'),run_name='__main__')
            script_ms=(time.perf_counter()-t)*1000
        finally:
            os.chdir(cwd)
        actual=np.fromfile(temp/'block_level_meta/reddit.dgl.block4',np.int32).reshape(-1,4)
        np.testing.assert_array_equal(meta,actual)
    result['validation']={'metadata_matches_original_script':True,'metadata_matches_shipped_file':True,
                          'block_count':count,'metadata_sha256':hashlib.sha256(meta.tobytes()).hexdigest()}
    log('python_original_script_io_included',dict(median_ms=script_ms))
    times=[]
    for _ in range(3):
        t=time.perf_counter(); ref=partition(hnp); times.append((time.perf_counter()-t)*1000)
    np.testing.assert_array_equal(meta,ref)
    log('python_reference_compute_only',stats(times))
    log('partition_gpu',bench(lambda:w.partition(np_),repeat=100))
    log('partition_wall',wall_bench(lambda:w.partition(np_)))
    # Pageable host -> preallocated device; no disk read or allocation in this timing.
    cpu_ptr=torch.from_numpy(hnp)
    log('partition_with_rowptr_h2d_wall',wall_bench(lambda:(np_.copy_(cpu_ptr),w.partition(np_))))
    log('full_reorder_indices_and_partition_gpu',bench(lambda:w.full(p,i),repeat=10))
    log('full_reorder_indices_and_partition_wall',wall_bench(lambda:w.full(p,i)))
    values=torch.ones(e,device='cuda'); newvalues=torch.empty_like(values)
    log('full_reorder_indices_values_and_partition_gpu',bench(lambda:w.full(p,i,values,newvalues),repeat=10))
    log('full_reorder_indices_values_and_partition_wall',wall_bench(lambda:w.full(p,i,values,newvalues)))
    stable=np.argsort(np.diff(hp),kind='stable').astype(np.int32)
    np.testing.assert_array_equal(w.perm.cpu().numpy(),stable)
    np.testing.assert_array_equal(w.rowptr.cpu().numpy(),hnp)
    np.testing.assert_array_equal(w.meta[:w.count.item()].cpu().numpy(),meta)
    # Compare exact edge order with original CSR in bounded chunks.
    newidx=w.indices.cpu().numpy()
    for start in range(0,n,4096):
        end=min(n,start+4096)
        expected=np.concatenate([hi[hp[r]:hp[r+1]] for r in stable[start:end]])
        np.testing.assert_array_equal(newidx[hnp[start]:hnp[end]],expected)
    result['validation']['full_reorder_exact_stable']=True
    result['validation']['stable_indices_match_shipped_new_indices']=bool(np.array_equal(newidx,hni))
    save()
    x=torch.randn(n,128,device='cuda'); y=torch.empty_like(x); ref_y=torch.empty_like(x)
    # Use shipped row-reordered CSR for an exact comparison with the original experiment.
    w.partition(np_)
    cs_sorted=CuSparse(np_,ni,values,x,ref_y)
    cs_original=CuSparse(p,i,values,x,ref_y)
    spmm(w.meta,count,ni,values,x,y); cs_sorted.run(x,ref_y)
    torch.testing.assert_close(y,ref_y,rtol=5e-4,atol=5e-3)
    result['validation']['reddit_spmm_all_elements_close']=True
    result['validation']['reddit_spmm_max_abs_error']=float((y-ref_y).abs().max())
    log('spmm_accel_128_zero_included_gpu',bench(lambda:spmm(w.meta,count,ni,values,x,y)))
    log('spmm_cusparse_original_128_gpu',bench(lambda:cs_original.run(x,ref_y)))
    log('spmm_cusparse_sorted_128_gpu',bench(lambda:cs_sorted.run(x,ref_y)))
    log('spmm_output_zero_only_gpu',bench(lambda:y.zero_(),repeat=100))
    cs_sorted.close(); cs_original.close()
    w.close(); save()


if __name__=='__main__':
    main()
