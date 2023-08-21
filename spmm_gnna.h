#pragma once
#include "spmm_base.h"

class SPMM_GNNA : public SPMM_BASE
{
public:
    using SPMM_BASE::SPMM_BASE;

protected:
    int *partPtr, *part2Node;
    int shared_memory;
    int partSize, num_parts, warpPerBlock;
    int block, grid;

public:
    double do_test(bool timing, int dim);
protected:
    void run(int dim);

};