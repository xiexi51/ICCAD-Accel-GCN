#pragma once
#include <cusparse.h>

double spmm_cusparse(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int num_e, int dim, int times);

double spmm_cusparse_coo(int *row, int *idx, float *val, float *vin, float *vout, int num_v, int num_e, int dim, int times);