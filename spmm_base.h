#pragma once
#include "util.h"
#include <string>
using namespace std;

class SPMM_BASE
{
public:
    SPMM_BASE(string _graph, int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int num_e, int dim)
    {
        this->_graph = _graph;
        this->ptr = ptr;
        this->idx = idx;
        this->val = val;
        this->vin = vin;
        this->vout = vout;
        this->num_v = num_v;
        this->num_e = num_e;
        this->dim = dim;
    }
    ~SPMM_BASE() {}

    virtual double do_test(bool timing, int dim) = 0;

protected:
    string _graph;
    int *ptr, *idx;
    float *val, *vin, *vout;
    int num_v, num_e, dim;
    dim3 grid, block;

protected:
    virtual void run(int dim) = 0;
    double timing_body(bool timing, int dim)
    {
        double ret = 0;
        if (!timing)
        {
            run(dim);
            cudaDeviceSynchronize();
        }
        else
        {
            int times = 4;
            // warmup
            for (int i = 0; i < times; i++)
            {
                run(dim);
            }
            cudaDeviceSynchronize();
            double measured_time = 0;
            for (int i = 0; i < times; i++)
            {
                timestamp(t0);
                run(dim);
                cudaDeviceSynchronize();
                timestamp(t1);
                measured_time += getDuration(t0, t1);
            }
            ret = measured_time / times;
        }
        return ret;
    }
    
};