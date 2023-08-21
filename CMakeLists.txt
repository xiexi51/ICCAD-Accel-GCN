cmake_minimum_required(VERSION 3.5)
project (spmm_test C CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME} main.cu spmm_accel.cu spmm_cusparse.cu spmm_gnna.cu)

target_link_libraries(${PROJECT_NAME}  cusparse)
