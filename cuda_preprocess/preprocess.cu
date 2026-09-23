#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cusparse.h>
#include <algorithm>
#include <stdexcept>
#include <string>

// All pointers are device pointers; the caller owns outputs and the stream.
// CSR uses int32 indices/offsets, as in the upstream repository.
static thread_local std::string error;
#define CK(x) do { auto e = (x); if(e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while(0)
#define API_BEGIN try {
#define API_END } catch(const std::exception& e) { error=e.what(); return -1; } return 0;

struct Workspace {
    int n;
    int *degree, *sorted_degree, *identity, *heads, *runs, *counts, *offsets;
    void *temp;
    size_t bytes;
    explicit Workspace(int size): n(size), temp(nullptr), bytes(0) {
        CK(cudaMalloc(&degree, size_t(7)*(n+1)*sizeof(int)));
        sorted_degree=degree+n+1; identity=sorted_degree+n+1;
        heads=identity+n+1; runs=heads+n+1; counts=runs+n+1; offsets=counts+n+1;
        size_t b=0;
        CK(cub::DeviceScan::InclusiveScan(nullptr,b,heads,runs,cub::Max(),n)); bytes=std::max(bytes,b);
        CK(cub::DeviceScan::ExclusiveSum(nullptr,b,counts,offsets,n+1)); bytes=std::max(bytes,b);
        CK(cub::DeviceRadixSort::SortPairs(nullptr,b,degree,sorted_degree,identity,heads,n)); bytes=std::max(bytes,b);
        CK(cudaMalloc(&temp,bytes));
    }
    ~Workspace() { cudaFree(temp); cudaFree(degree); }
};

__device__ int block_rows(int d) {
    // Intentionally match Python's table ending at degree 191, not 383.
    if(d>=192) return 1;
    return d<=32 ? 12 : d<=64 ? 6 : d<=96 ? 4 : d<=128 ? 3 : 2;
}
__device__ int warp_nz(int d) {
    if(d>=192) return 32;
    int f=12/block_rows(d);
    return (d+f-1)/f;
}
__global__ void degrees(const int *ptr,int *d,int *id,int n) {
    int r=blockIdx.x*blockDim.x+threadIdx.x;
    if(r<n) { d[r]=ptr[r+1]-ptr[r]; id[r]=r; }
    if(r==n) d[n]=0;
}
__global__ void run_heads(const int *ptr,int *heads,int n) {
    int r=blockIdx.x*blockDim.x+threadIdx.x;
    if(r<n) {
        int d=ptr[r+1]-ptr[r];
        heads[r]=(r==0 || d!=ptr[r]-ptr[r-1]) ? r : 0;
    }
}
__global__ void count_blocks(const int *ptr,const int *runs,int *counts,int n) {
    int r=blockIdx.x*blockDim.x+threadIdx.x;
    if(r<n) {
        int d=ptr[r+1]-ptr[r];
        counts[r]= d==0 ? 0 : d>384 ? (d+383)/384 : ((r-runs[r])%block_rows(d)==0);
    }
    if(r==n) counts[n]=0;
}
__global__ void emit(const int *ptr,const int *counts,const int *offsets,int4 *out,int *total,int n) {
    int r=blockIdx.x*blockDim.x+threadIdx.x;
    if(r==n) *total=offsets[n];
    if(r>=n || !counts[r]) return;
    int d=ptr[r+1]-ptr[r], loc=ptr[r], at=offsets[r];
    if(d>384) {
        for(int k=0;k<counts[r];++k) out[at+k]=make_int4(d,r,loc+384*k,min(384,d-384*k));
    } else {
        int nr=1, limit=block_rows(d);
        while(nr<limit && r+nr<n && ptr[r+nr+1]-ptr[r+nr]==d) ++nr;
        out[at]=make_int4(d,r,loc,(warp_nz(d)<<16)|nr);
    }
}
__global__ void copy_rows(const int *ptr,const int *idx,const float *val,
                          const int *perm,const int *newptr,int *newidx,float *newval,int n) {
    int r=(blockIdx.x*blockDim.x+threadIdx.x)/32, lane=threadIdx.x%32;
    if(r>=n) return;
    int old=perm[r], begin=ptr[old], len=ptr[old+1]-begin;
    for(int j=lane;j<len;j+=32) {
        newidx[newptr[r]+j]=idx[begin+j];
        if(val && newval) newval[newptr[r]+j]=val[begin+j];
    }
}
extern "C" const char *ag_error() { return error.c_str(); }
extern "C" void *ag_create(int n) {
    try { if(n<0) throw std::runtime_error("negative n"); return new Workspace(n); }
    catch(const std::exception& e) { error=e.what(); return nullptr; }
}
extern "C" void ag_destroy(void *p) { delete static_cast<Workspace*>(p); }
extern "C" int ag_partition(void *p,const int *ptr,int4 *out,int *total,cudaStream_t stream) {
    API_BEGIN
    auto &w=*static_cast<Workspace*>(p); int n=w.n, grid=(n+256)/256;
    run_heads<<<grid,256,0,stream>>>(ptr,w.heads,n);
    CK(cub::DeviceScan::InclusiveScan(w.temp,w.bytes,w.heads,w.runs,cub::Max(),n,stream));
    count_blocks<<<grid,256,0,stream>>>(ptr,w.runs,w.counts,n);
    CK(cub::DeviceScan::ExclusiveSum(w.temp,w.bytes,w.counts,w.offsets,n+1,stream));
    emit<<<grid,256,0,stream>>>(ptr,w.counts,w.offsets,out,total,n);
    CK(cudaGetLastError());
    API_END
}
extern "C" int ag_reorder(void *p,const int *ptr,const int *idx,const float *val,
                           int *perm,int *newptr,int *newidx,float *newval,cudaStream_t stream) {
    API_BEGIN
    auto &w=*static_cast<Workspace*>(p); int n=w.n;
    degrees<<<(n+256)/256,256,0,stream>>>(ptr,w.degree,w.identity,n);
    // CUB radix sort is stable: ties retain original row order.
    CK(cub::DeviceRadixSort::SortPairs(w.temp,w.bytes,w.degree,w.sorted_degree,w.identity,perm,n,0,32,stream));
    CK(cudaMemsetAsync(w.sorted_degree+n,0,sizeof(int),stream));
    CK(cub::DeviceScan::ExclusiveSum(w.temp,w.bytes,w.sorted_degree,newptr,n+1,stream));
    // Mapping-only mode does not read or allocate any edge-sized arrays.
    if(n && newidx) copy_rows<<<(n+7)/8,256,0,stream>>>(ptr,idx,val,perm,newptr,newidx,newval,n);
    CK(cudaGetLastError());
    API_END
}

// Same work assignment, accumulation, shared memory and launch as upstream.
// Only edge addressing changes. Row mapping is resolved outside the edge loop.
template<bool OriginalOutput>
__global__ void spmm_kernel_mapped(const int4 *meta,const int *perm,
    const int *original_ptr,const int *sorted_ptr,const int *idx,const float *val,
    const float *vin,float *vout,int cols) {
    int4 b=meta[blockIdx.x];
    int degree=b.x, begin=b.y, loc=b.z, info=b.w;
    int nrows=degree<=384 ? info&65535 : 1;
    int wnz=degree<=384 ? info>>16 : 32;
    int rownz=degree<=384 ? degree : info;
    int round_dim=((cols+31)/32)*32;
    int warps_per_row=(rownz+wnz-1)/wnz;
    extern __shared__ float cache[];
#pragma unroll
    for(int ext=0;ext<(cols+31)/32;++ext) {
        int lane=(threadIdx.x+ext*blockDim.x)%round_dim;
        if(lane>=cols) return;
        int wid=(threadIdx.x+ext*blockDim.x)/round_dim;
        int tid=wid*round_dim+lane;
        int localrow=wid/warps_per_row, localcol=wid%warps_per_row*wnz;
        if(localrow>=nrows) return;
        int sortedrow=begin+localrow;
        int originalrow=__ldg(perm+sortedrow);
        int edge_begin=__ldg(original_ptr+originalrow);
        // Large rows have multiple blocks; loc encodes the segment's offset
        // in the virtual sorted CSR, not an address into original indices.
        if(degree>384) edge_begin+=loc-__ldg(sorted_ptr+begin);
#pragma unroll
        for(int j=0;j<wnz;++j) {
            if(j+localcol>=rownz) break;
            if(j==0) cache[tid]=0;
            int nz=edge_begin+localcol+j;
            float left=__ldg(val+nz);
            float right=vin[__ldg(idx+nz)*cols+lane];
            cache[tid]+=left*right;
        }
        int outputrow=OriginalOutput ? originalrow : sortedrow;
        if(warps_per_row>1 || degree>384)
            atomicAdd(vout+outputrow*cols+lane,cache[tid]);
        else vout[outputrow*cols+lane]=cache[tid];
    }
}
extern "C" int ag_spmm_mapped(const int *meta,int blocks,const int *perm,
    const int *original_ptr,const int *sorted_ptr,const int *idx,const float *val,
    const float *x,float *y,int n,int cols,int original_output,cudaStream_t stream) {
    API_BEGIN
    CK(cudaMemsetAsync(y,0,size_t(n)*cols*sizeof(float),stream));
    size_t shared=12*((cols+31)/32)*32*sizeof(float);
    if(blocks) {
        if(original_output)
            spmm_kernel_mapped<true><<<blocks,384,shared,stream>>>(reinterpret_cast<const int4*>(meta),perm,original_ptr,sorted_ptr,idx,val,x,y,cols);
        else
            spmm_kernel_mapped<false><<<blocks,384,shared,stream>>>(reinterpret_cast<const int4*>(meta),perm,original_ptr,sorted_ptr,idx,val,x,y,cols);
    }
    CK(cudaGetLastError());
    API_END
}

// Link the upstream kernel unchanged; fix its required output initialization here.
__global__ void spmm_kernel_accel(const int*,const int*,const int*,const float*,const float*,float*,int,int,int,const float*);
extern "C" int ag_spmm(const int *meta,int blocks,const int *idx,const float *val,
                        const float *x,float *y,int n,int e,int cols,cudaStream_t stream) {
    API_BEGIN
    CK(cudaMemsetAsync(y,0,size_t(n)*cols*sizeof(float),stream));
    if(blocks) spmm_kernel_accel<<<blocks,384,12*((cols+31)/32)*32*sizeof(float),stream>>>(meta,nullptr,idx,val,x,y,n,e,cols,nullptr);
    CK(cudaGetLastError());
    API_END
}

#define SP(x) do { auto sparse_status=(x); if(sparse_status!=CUSPARSE_STATUS_SUCCESS) throw std::runtime_error(cusparseGetErrorString(sparse_status)); } while(0)
struct SparseMM {
    cusparseHandle_t h;
    cusparseSpMatDescr_t a;
    cusparseDnMatDescr_t b,c;
    void *buffer=nullptr;
    float alpha=1, beta=0;
    SparseMM(int n,int e,int cols,int *ptr,int *idx,float *val,float *x,float *y) {
        SP(cusparseCreate(&h));
        SP(cusparseCreateCsr(&a,n,n,e,ptr,idx,val,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F));
        SP(cusparseCreateDnMat(&b,n,cols,cols,x,CUDA_R_32F,CUSPARSE_ORDER_ROW));
        SP(cusparseCreateDnMat(&c,n,cols,cols,y,CUDA_R_32F,CUSPARSE_ORDER_ROW));
        size_t bytes=0;
        SP(cusparseSpMM_bufferSize(h,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,a,b,&beta,c,CUDA_R_32F,CUSPARSE_SPMM_ALG_DEFAULT,&bytes));
        if(bytes) CK(cudaMalloc(&buffer,bytes));
    }
    ~SparseMM() { cudaFree(buffer); cusparseDestroyDnMat(b); cusparseDestroyDnMat(c); cusparseDestroySpMat(a); cusparseDestroy(h); }
};
extern "C" void *ag_cusparse_create(int n,int e,int cols,int *ptr,int *idx,float *val,float *x,float *y) {
    try { return new SparseMM(n,e,cols,ptr,idx,val,x,y); }
    catch(const std::exception& e) { error=e.what(); return nullptr; }
}
extern "C" int ag_cusparse_run(void *p,float *x,float *y,cudaStream_t stream) {
    API_BEGIN
    auto &s=*static_cast<SparseMM*>(p);
    SP(cusparseSetStream(s.h,stream));
    SP(cusparseDnMatSetValues(s.b,x)); SP(cusparseDnMatSetValues(s.c,y));
    SP(cusparseSpMM(s.h,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,&s.alpha,s.a,s.b,&s.beta,s.c,CUDA_R_32F,CUSPARSE_SPMM_ALG_DEFAULT,s.buffer));
    API_END
}
extern "C" void ag_cusparse_destroy(void *p) { delete static_cast<SparseMM*>(p); }
