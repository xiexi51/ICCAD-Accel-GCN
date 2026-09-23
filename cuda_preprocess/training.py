"""Short full-batch GCN training timing; uses an explicit transpose for backward."""
import argparse
import gc
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix, eye
from backend import Preprocessor, CuSparse, spmm
from benchmark import ROOT, stats, bench

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=20)
    args=parser.parse_args()
    torch.manual_seed(123); torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    hp=np.fromfile(ROOT/'graphs/reddit.dgl.graph.ptrdump',np.int32)
    hi=np.fromfile(ROOT/'graphs/reddit.dgl.graph.edgedump',np.int32)
    n,e=len(hp)-1,len(hi)
    result={'architecture':[602,128,41],'epochs_measured':args.epochs,'warmup_epochs':5,
            'target_epochs':300,'data':'real Reddit adjacency; synthetic features and labels',
            'full_batch':True,'train_nodes':int(n*0.66),'dropout':0.5,'optimizer':'Adam',
            'dtype':'float32','tf32':False,'four_spmm_widths_per_epoch':[128,41,41,128],
            'normalization':'D^-1/2 (A+I) D^-1/2; add standard GCN self-loops',
            'excludes':'dataset I/O, validation, checkpointing; 300-epoch time is extrapolated'}
    def save():
        (ROOT/'results/training.json').write_text(json.dumps(result,indent=2))
    print('Constructing explicit transpose...',flush=True)
    a=csr_matrix((np.ones(e,np.float32),hi,hp),shape=(n,n))
    result['input_self_loops']=int(np.count_nonzero(a.diagonal()))
    a=a+eye(n,format='csr',dtype=np.float32)
    e=a.nnz
    result['training_nnz']=int(e)
    at=a.T.tocsr()
    result['asymmetric_entries']=int((a!=at).nnz)
    result['self_loops']=int(np.count_nonzero(a.diagonal()))
    print('asymmetric entries:',result['asymmetric_entries'],flush=True)
    invdeg=np.maximum(np.diff(a.indptr),1).astype(np.float32)**-0.5
    # Normalize A and A^T consistently using A's degree vector; do not assume symmetry.
    for matrix in (a,at):
        for begin in range(0,n,4096):
            end=min(n,begin+4096); lo,hi_=matrix.indptr[begin],matrix.indptr[end]
            row=np.repeat(np.arange(begin,end),np.diff(matrix.indptr[begin:end+1]))
            matrix.data[lo:hi_]=invdeg[row]*invdeg[matrix.indices[lo:hi_]]
    # a and at have independent data buffers.
    workspaces=[]; data=[]; inverse=[]; counts=[]; sparse_handles=[]
    for matrix in (a,at):
        p=torch.from_numpy(matrix.indptr.astype(np.int32)).cuda()
        i=torch.from_numpy(matrix.indices.astype(np.int32)).cuda()
        v=torch.from_numpy(matrix.data).cuda(); sv=torch.empty_like(v)
        w=Preprocessor(n,e); w.full(p,i,v,sv)
        workspaces.append(w); data.append((p,i,v,sv)); inverse.append(torch.argsort(w.perm.long()))
        counts.append(w.count.item())
        handles={}
        for dim in (128,41):
            xx=torch.empty(n,dim,device='cuda'); yy=torch.empty_like(xx)
            handles[dim]=CuSparse(p,i,v,xx,yy)
        sparse_handles.append(handles)
    result['transpose_full_preprocess_gpu']=bench(lambda:workspaces[1].full(*data[1][:3],data[1][3]),repeat=10)
    del a,at; gc.collect()
    def aggregate(x,method,transpose=False):
        direction=int(transpose); x=x.contiguous(); out=torch.empty_like(x)
        if method=='accel':
            w=workspaces[direction]
            spmm(w.meta,counts[direction],w.indices,data[direction][3],x,out)
            return out[inverse[direction]]
        return sparse_handles[direction][x.shape[1]].run(x,out)
    class Aggregate(torch.autograd.Function):
        @staticmethod
        def forward(ctx,x,method):
            ctx.method=method
            return aggregate(x,method)
        @staticmethod
        def backward(ctx,grad):
            return aggregate(grad,ctx.method,transpose=True),None
    print('Checking forward and backward against cuSPARSE...',flush=True)
    for dim in (41,128):
        test=torch.randn(n,dim,device='cuda',requires_grad=True)
        aa=Aggregate.apply(test,'accel'); cc=Aggregate.apply(test,'cusparse')
        torch.testing.assert_close(aa,cc,rtol=5e-4,atol=5e-5)
        g=torch.randn_like(test)
        ga,=torch.autograd.grad(aa,test,g); gc_,=torch.autograd.grad(cc,test,g)
        torch.testing.assert_close(ga,gc_,rtol=5e-4,atol=5e-5)
    del test,aa,cc,g,ga,gc_
    result['forward_backward_close']=True
    features=torch.randn(n,602,device='cuda')
    labels=torch.randint(41,(n,),device='cuda')
    train_idx=torch.arange(int(n*0.66),device='cuda')
    for method in ('accel','cusparse'):
        torch.manual_seed(456)
        l1=torch.nn.Linear(602,128,bias=False,device='cuda')
        l2=torch.nn.Linear(128,41,bias=False,device='cuda')
        optimizer=torch.optim.Adam(list(l1.parameters())+list(l2.parameters()),lr=0.01,weight_decay=5e-4)
        def step():
            optimizer.zero_grad(set_to_none=True)
            h=F.dropout(features,p=0.5,training=True)
            h=F.relu(Aggregate.apply(l1(h),method))
            h=F.dropout(h,p=0.5,training=True)
            logits=Aggregate.apply(l2(h),method)
            loss=F.cross_entropy(logits[train_idx],labels[train_idx])
            loss.backward(); optimizer.step()
            return loss
        for _ in range(5): step()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        times=[]; gpu_times=[]
        for _ in range(args.epochs):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            t=time.perf_counter(); start.record(); loss=step(); end.record(); end.synchronize()
            times.append((time.perf_counter()-t)*1000); gpu_times.append(start.elapsed_time(end))
        measured=stats(times)
        measured.update(estimated_300_epochs_s=measured['median_ms']*0.3,
                        mean_based_300_epochs_s=float(np.mean(times))*0.3,
                        peak_torch_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                        final_loss=float(loss),gpu_event=stats(gpu_times))
        result[method]=measured; save()
        print(method,measured,flush=True)
        del optimizer,l1,l2
    for handles in sparse_handles:
        for handle in handles.values(): handle.close()
    for w in workspaces: w.close()
    save()

if __name__=='__main__':
    main()
