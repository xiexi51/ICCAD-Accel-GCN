#include "spmm_cusparse.h"
#include "util.h"
#include <iostream>
using namespace std;

double spmm_cusparse(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int num_e, int dim, int times)
{
    cusparseHandle_t handle;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    float alpha = 1.0;
    // float beta = 1.0;
    float beta = 0.0;
    float *buf = NULL;
    cusparseCreate(&handle);
    if (ptr == NULL)
    {
        cout << "ptr is null !!!!" << endl;
    }
    cusparseCreateCsr(&matA, num_v, num_v, num_e, ptr, idx, val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnMat(&matB, num_v, dim, dim, vin, CUDA_R_32F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&matC, num_v, dim, dim, vout, CUDA_R_32F, CUSPARSE_ORDER_ROW);
    size_t bufferSize = 0;
    cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);
    cudaMallocManaged(&buf, bufferSize);

    double ret = 0;
    if (times == 0)
    {
        cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, buf);
        cudaDeviceSynchronize();
    }
    else
    {
        times = 10;
        // warmup
        for (int i = 0; i < times; i++)
        {
            cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, buf);
        }
        cudaDeviceSynchronize();
        double measured_time = 0;
        for (int i = 0; i < times; i++)
        {
            timestamp(t0);
            cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, buf);
            cudaDeviceSynchronize();
            timestamp(t1);
            measured_time += getDuration(t0, t1);
        }
        ret = measured_time / times;
    }

    cusparseDnMatGetValues(matC, (void **)&vout);
    cusparseDestroy(handle);
    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cudaFree(buf);

    return ret;
}


double spmm_cusparse_coo(int *row, int *idx, float *val, float *vin, float *vout, int num_v, int num_e, int dim, int times)
{
    cusparseHandle_t handle;
    
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    float alpha = 1.0;
    // float beta = 1.0;
    float beta = 0.0;
    float *buf = NULL;
    cusparseCreate(&handle);
    
    cusparseCreateCoo(&matA, num_v, num_v, num_e, row, idx, val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnMat(&matB, num_v, dim, dim, vin, CUDA_R_32F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&matC, num_v, dim, dim, vout, CUDA_R_32F, CUSPARSE_ORDER_ROW);
    size_t bufferSize = 0;
    cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);
    cudaMallocManaged(&buf, bufferSize);

    double ret = 0;
    if (times == 0)
    {
        cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, buf);
        cudaDeviceSynchronize();
    }
    else
    {
        times = 10;
        // warmup
        for (int i = 0; i < times; i++)
        {
            cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, buf);
        }
        cudaDeviceSynchronize();
        double measured_time = 0;
        for (int i = 0; i < times; i++)
        {
            timestamp(t0);
            cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, buf);
            cudaDeviceSynchronize();
            timestamp(t1);
            measured_time += getDuration(t0, t1);
        }
        ret = measured_time / times;
    }

    cusparseDnMatGetValues(matC, (void **)&vout);
    cusparseDestroy(handle);
    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cudaFree(buf);

    return ret;
}