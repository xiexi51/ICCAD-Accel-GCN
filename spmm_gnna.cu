#include "spmm_gnna.h"
#include "data.h"
#include <string>
#include <vector>
using namespace std;

extern string base_dir, graph;

const int WARPSIZE = 32;
const int WARP_PER_BLOCK = 4;
const int NZ_PER_WARP = 4;

__device__ inline void atomicAdd_F(float *address, float value)
{
    float old = value;
    while ((old = atomicExch(address, atomicExch(address, 0.0f) + old)) != 0.0f)
        ;
}

vector<vector<int>> build_part(int partSize, vector<int> indptr)
{
    int num_nodes = indptr.size() - 1;
    int degree, thisNumParts, numParts = 0;

    for (int i = 0; i < num_nodes; i++)
    {
        degree = indptr[i + 1] - indptr[i];
        if (degree % partSize == 0)
            thisNumParts = degree / partSize;
        else
            thisNumParts = degree / partSize + 1;
        numParts += thisNumParts;
    }

    auto partPtr = vector<int>(numParts + 1, 0);
    auto part2Node = vector<int>(numParts, 0);

    int part_counter = 0;
    for (int i = 0; i < num_nodes; i++)
    {
        int degree = indptr[i + 1] - indptr[i];
        if (degree % partSize == 0)
            thisNumParts = degree / partSize;
        else
            thisNumParts = degree / partSize + 1;

        for (int pid = 0; pid < thisNumParts; pid++)
        {
            int partBeg = indptr[i] + pid * partSize;
            int partEnd = partBeg + partSize < indptr[i + 1] ? partBeg + partSize : indptr[i + 1];
            partPtr[part_counter] = partBeg;
            part2Node[part_counter++] = i;
            if (i == num_nodes - 1 && partEnd == indptr[i + 1])
                partPtr[part_counter] = partEnd;
        }
    }
    return {partPtr, part2Node};
}

__global__ void SAG_cuda_kernel(
    float *output,
    float *input,
    int *row_pointers,
    int *column_index,
    float *degrees,
    int *part_pointers,
    int *part2Node,
    const int num_nodes,
    const int dim,
    const int num_parts,
    const int partSize,
    const int dimWorker,
    const int warpPerBlock)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x; // global thread-id
    int warpId = tid / WARPSIZE;                     // global warp-id
    int block_warpId = threadIdx.x / WARPSIZE;       // block warp-id
    int laneid = threadIdx.x % WARPSIZE;             // warp thread-id -- laneid

    extern __shared__ int part_meta[];                                     // part information.
    // __shared__ int part_meta[2048];
    int *partial_ids = part_meta;                                          // caching ids
    float *partial_results = (float *)&part_meta[partSize * warpPerBlock]; // caching partial results.

    if (warpId < num_parts)
    {

        int srcId = part2Node[warpId];           // aggregated source node
        int partBeg = part_pointers[warpId];     // partitioning pointer start
        int partEnd = part_pointers[warpId + 1]; // part pointer end

        // Cache the part neighbors.
        const int pindex_base = block_warpId * partSize;
        // #pragma unroll
        for (int nidx = partBeg + laneid; nidx < partEnd; nidx += dimWorker)
        {
            // printf("1--pindex_base: %d, laneid: %d\n", pindex_base, laneid);
            partial_ids[pindex_base + nidx - partBeg] = column_index[nidx];
            // if(partial_ids[pindex_base + laneid]  >= num_nodes || partial_ids[pindex_base + laneid]  < 0) printf("---- partial_ids: %d\n", partial_ids[pindex_base + laneid] );
        }

        __syncwarp();

        // Neighbor aggregation within each part
        const int presult_base = block_warpId * dim;
        for (int nIdx = 0; nIdx < partEnd - partBeg; nIdx++)
        {
            // if (laneid == 0) printf("2--pindex_base: %d, nIdx: %d\n", pindex_base, nIdx);
            int nid = partial_ids[pindex_base + nIdx];
            // if(nid >= num_nodes || nid < 0) printf("Error nid: %d\n", nid);

            // Initialize shared memory for partial results
            if (nIdx == 0)
                if (laneid < dimWorker)
#pragma unroll
                    for (int d = laneid; d < dim; d += dimWorker)
                    {
                        partial_results[presult_base + d] = 0.0f;
                    }

            if (laneid < dimWorker)
#pragma unroll
                for (int d = laneid; d < dim; d += dimWorker)
                {
                    partial_results[presult_base + d] += input[nid * dim + d];
                }
        }

        // output the result to global memory from the shared memory
        if (laneid < dimWorker)
#pragma unroll
            for (int d = laneid; d < dim; d += dimWorker)
            {
                atomicAdd_F((float *)&output[srcId * dim + d], partial_results[presult_base + d]);
            }
    }
}

void SPMM_GNNA::run(int dim)
{
    SAG_cuda_kernel<<<grid, block, shared_memory>>>(vout, vin, ptr, idx, 0, partPtr, part2Node, num_v, dim, num_parts, partSize, 32, warpPerBlock);
}

double SPMM_GNNA::do_test(bool timing, int dim)
{
    partSize = num_e / num_v;
    if(partSize < 1){
        partSize = 1;
    }
    
    vector<int> indptr_vec((int*)ptr, (int*)ptr + num_v + 1);
    vector<vector<int>> parts = build_part(partSize, indptr_vec);

    warpPerBlock = 12;
    num_parts = parts[1].size();
    block = warpPerBlock * WARPSIZE;
    grid = (num_parts * WARPSIZE + block - 1) / block;
    
    shared_memory = partSize * warpPerBlock * sizeof(int) + warpPerBlock * dim * sizeof(float);

    cudaMallocManaged(&partPtr, parts[0].size() * sizeof(int));
    cudaMallocManaged(&part2Node, parts[1].size() * sizeof(int));

    copy(parts[0].begin(), parts[0].end(), partPtr);
    copy(parts[1].begin(), parts[1].end(), part2Node);

    double ret = timing_body(timing, dim);

    cudaFree(partPtr);
    cudaFree(part2Node);

    return ret;
}

