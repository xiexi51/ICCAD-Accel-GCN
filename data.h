#pragma once
#include <fstream>
#include <string>
using namespace std;

template <typename scalar_t>
int cuda_read_array(scalar_t **arr, string file)
{
    std::ifstream input(file, ios::in | ios::binary);
    input.seekg(0, input.end);
    int length = input.tellg();
    input.seekg(0, input.beg);
    int cnt = length / sizeof(scalar_t);
    // *arr = new scalar_t[cnt];
    cudaMallocManaged(arr, cnt * sizeof(scalar_t));

    input.read((char *)*arr, length);
    input.close();
    // *arr = reinterpret_cast<float *>(&buffer[0]);
    return cnt;
}
template <typename scalar_t>
int read_array(scalar_t **arr, string file)
{
    std::ifstream input(file, ios::in | ios::binary);
    input.seekg(0, input.end);
    int length = input.tellg();
    input.seekg(0, input.beg);
    int cnt = length / sizeof(scalar_t);
    *arr = new scalar_t[cnt];
    input.read((char *)*arr, length);
    input.close();
    return cnt;
}