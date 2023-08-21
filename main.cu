#include <iostream>
#include "data.h"
#include "spmm_accel.h"
#include "spmm_gnna.h"
#include "spmm_cusparse.h"
#include <random>
#include <algorithm>
#include <filesystem>

string base_dir = "../graphs/";

int dim_min = 16, dim_max = 128, interval = 1;

int total_file_cnt, current_file_cnt;

using namespace std;

double check_err(float *out, float *out_ref, int len, bool &has_err)
{
    double err_sum = 0;
    bool show = 1;

    has_err = 0;

    for (int i = 0; i < len; i++)
    {
        double err = abs(out[i] - out_ref[i]);
        err_sum += err;
        // if (err_sum / (v_num * dim) >= 0.001 && show)
        // {
        //     show = 0;
        //     cout << "fail begin at " << i/32 << endl;
        // }
        if (err > 0.1 && has_err == 0)
        {
            has_err = 1;
            cout << "err at " << i << endl;
        }
    }
    cout << "err sum = " << err_sum << "  ";
    if (err_sum / len < 0.001)
    // if(!has_err)
    {
        cout << "validation pass!" << endl;
    }
    else
    {
        cout << "validation fail!" << endl;
    }
    return err_sum;
}

void test_graph(string graph, int spec_dim, bool checking, bool timing)
{
    if (spec_dim > 0)
    {
        dim_min = spec_dim;
        dim_max = spec_dim;
    }

    int *cu_indptr, *cu_indices, *cu_indptr_new, *cu_indices_new, *cu_coo_row;
    int v_num = cuda_read_array(&cu_indptr_new, base_dir + graph + ".new_indptr") - 1;
    int e_num = cuda_read_array(&cu_indices_new, base_dir + graph + ".new_indices");
    cuda_read_array(&cu_indptr, base_dir + graph + ".graph.ptrdump");
    cuda_read_array(&cu_indices, base_dir + graph + ".graph.edgedump");

    cudaMallocManaged(&cu_coo_row, e_num * sizeof(int));
    {
        int k = 0;
        for (int i = 0; i < v_num; i++)
        {
            for (int j = 0; j < cu_indptr[i + 1] - cu_indptr[i]; j++)
            {
                cu_coo_row[k++] = i;
            }
        }
    }

    // cout << "graph = " << graph << " v_num = " << v_num << " e_num = " << e_num << endl;
    float *cu_val;
    cudaMallocManaged(&cu_val, e_num * sizeof(float));

    float *cu_vin, *cu_vout_new, *cu_vout_ref, *cu_vout_gnna, *cu_vout_ref_new;
    cudaMallocManaged(&cu_vin, v_num * dim_max * sizeof(float));
    cudaMallocManaged(&cu_vout_new, v_num * dim_max * sizeof(float));
    cudaMallocManaged(&cu_vout_gnna, v_num * dim_max * sizeof(float));
    cudaMallocManaged(&cu_vout_ref, v_num * dim_max * sizeof(float));
    cudaMallocManaged(&cu_vout_ref_new, v_num * dim_max * sizeof(float));

    default_random_engine engine;
    engine.seed(123);

    uniform_real_distribution<float> rd(0, 1);

    int spm_value_mode = 3;
    switch (spm_value_mode)
    {
    case 1:
        generate(cu_val, cu_val + e_num, [&]()
                 { return rd(engine); });
        generate(cu_vin, cu_vin + v_num * dim_max, [&]()
                 { return rd(engine); });
        break;
    case 2: 
        for (int i = 0; i < e_num; i++)
        {
            cu_val[i] = 1;
        }
        for (int i = 0; i < v_num * dim_max; i++)
        {
            cu_vin[i] = 0.01 * i;
        }
        break;
    case 3: // GNNAdvisor requires all values of the sparse matrix to be 1
        for (int i = 0; i < e_num; i++)
        {
            cu_val[i] = 1;
        }
        generate(cu_vin, cu_vin + v_num * dim_max, [&]()
                 { return rd(engine); });
        break;

    default:
        break;
    }

    fill(cu_vout_gnna, cu_vout_gnna + v_num * dim_max, 0);
    fill(cu_vout_ref, cu_vout_ref + v_num * dim_max, 0);
    fill(cu_vout_ref_new, cu_vout_ref_new + v_num * dim_max, 0);

    SPMM_ACCEL accel(graph, cu_indptr_new, cu_indices_new, cu_val, cu_vin, cu_vout_new, v_num, e_num, dim_max);
    SPMM_GNNA gnna(graph, cu_indptr, cu_indices, cu_val, cu_vin, cu_vout_gnna, v_num, e_num, dim_max);

#define CHECK
#define TIMING

    for (int dim = dim_min; dim <= dim_max; dim += interval)
    {
        // cout << "dim = " << dim << endl;

        if(checking){
            cout << "CHECKING ..." << endl;
            spmm_cusparse(cu_indptr, cu_indices, cu_val, cu_vin, cu_vout_ref, v_num, e_num, dim, 0);
            spmm_cusparse(cu_indptr_new, cu_indices_new, cu_val, cu_vin, cu_vout_ref_new, v_num, e_num, dim, 0);
            
            accel.do_test(false, dim);
            gnna.do_test(false, dim);

            bool has_err = 0;
            
            cout << "checking accel" << endl;
            check_err(cu_vout_new, cu_vout_ref_new, v_num * dim, has_err);
            
            cout << "checking gnna" << endl;
            check_err(cu_vout_gnna, cu_vout_ref, v_num * dim, has_err);
        }

        if(timing){
            cout << "TIMING ..." << endl;
            double timing_mul = 1000000;
            string outstr = to_string(current_file_cnt) + "/" + to_string(total_file_cnt) + " " + graph + " " + to_string(dim);
            double t_cusparse = spmm_cusparse(cu_indptr, cu_indices, cu_val, cu_vin, cu_vout_ref, v_num, e_num, dim, 10);
            cout << outstr << " cusparse " << t_cusparse * timing_mul << endl;
            double t_gnna = gnna.do_test(true, dim);
            cout << outstr << " gnna " << t_gnna * timing_mul << endl;
            double t_accel = accel.do_test(true, dim);
            cout << outstr << " accel " << t_accel * timing_mul << endl;
        }
    }

    cudaFree(cu_indptr);
    cudaFree(cu_indices);
    cudaFree(cu_coo_row);
    cudaFree(cu_indptr_new);
    cudaFree(cu_indices_new);
    cudaFree(cu_val);
    cudaFree(cu_vin);
    cudaFree(cu_vout_new);
    cudaFree(cu_vout_gnna);
    cudaFree(cu_vout_ref);
    cudaFree(cu_vout_ref_new);
}

int main(int argc, char *argv[])
{
    if (argc > 2)  // usage: spmm_test artist 60
    {
        string arg_graph(argv[1]);
        int dim = atoi(argv[2]);
        cout << "dir = " << base_dir << endl;

        bool checking = true, timing = true;
        test_graph(arg_graph, dim, checking, timing);
    }
    else  // usage: spmm_test
    {
        string extension = ".config";

        total_file_cnt = 0;
        for (const auto &file : filesystem::directory_iterator(base_dir))
        {
            if (file.path().extension() == extension)
            {
                total_file_cnt++;
            }
        }

        current_file_cnt = 0;
        for (const auto &file : filesystem::directory_iterator(base_dir))
        {
            if (file.path().extension() == extension)
            {
                current_file_cnt++;
                string graph = file.path().stem().string();
                test_graph(graph, 0, false, true);
                cudaDeviceSynchronize();
            }
        }
    }

    return 0;
}