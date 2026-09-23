#include "spmm_accel.h"

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
    auto status = cudaMemsetAsync(y,0,size_t(n)*cols*sizeof(float),stream);
    if(status != cudaSuccess) return static_cast<int>(status);
    size_t shared=12*((cols+31)/32)*32*sizeof(float);
    if(blocks) {
        if(original_output)
            spmm_kernel_mapped<true><<<blocks,384,shared,stream>>>(reinterpret_cast<const int4*>(meta),perm,original_ptr,sorted_ptr,idx,val,x,y,cols);
        else
            spmm_kernel_mapped<false><<<blocks,384,shared,stream>>>(reinterpret_cast<const int4*>(meta),perm,original_ptr,sorted_ptr,idx,val,x,y,cols);
    }
    return static_cast<int>(cudaGetLastError());
}
