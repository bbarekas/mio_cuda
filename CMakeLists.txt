cmake_minimum_required(VERSION 3.16)
project(cuda_t1 CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(MAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin/nvcc.exe")

add_executable(cuda_t1 main.cu)

# find_package(benchmark REQUIRED)
# add_subdirectory(../benchmark)


set_target_properties(cuda_t1 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

# target_link_libraries(cuda_t1 PRIVATE benchmark::benchmark)
