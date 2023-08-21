#pragma once
#include "spmm_base.h"

class SPMM_ACCEL : public SPMM_BASE
{
public:
    using SPMM_BASE::SPMM_BASE;

    float *vout_ref;

protected:
    int *coo_row, *_block4;
    

public:
    double do_test(bool timing, int dim);
protected:
    void run(int dim);

};

