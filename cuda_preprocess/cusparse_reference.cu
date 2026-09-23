#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdexcept>
#include <string>
static thread_local std::string error;
#define CK(x) do { auto e=(x); if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while(0)
#define API_BEGIN try {
#define API_END } catch(const std::exception& e) { error=e.what(); return -1; } return 0;
extern "C" const char *ag_cusparse_error() { return error.c_str(); }
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
