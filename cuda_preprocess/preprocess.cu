#include <cuda_runtime.h>
#include <cub/cub.cuh>
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
extern "C" int ag_mapping(void *p,const int *ptr,int *perm,int *newptr,cudaStream_t stream) {
    API_BEGIN
    auto &w=*static_cast<Workspace*>(p); int n=w.n;
    degrees<<<(n+256)/256,256,0,stream>>>(ptr,w.degree,w.identity,n);
    // CUB radix sort is stable: ties retain original row order.
    CK(cub::DeviceRadixSort::SortPairs(w.temp,w.bytes,w.degree,w.sorted_degree,w.identity,perm,n,0,32,stream));
    CK(cudaMemsetAsync(w.sorted_degree+n,0,sizeof(int),stream));
    CK(cub::DeviceScan::ExclusiveSum(w.temp,w.bytes,w.sorted_degree,newptr,n+1,stream));
    CK(cudaGetLastError());
    API_END
}
