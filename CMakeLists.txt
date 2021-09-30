cmake_minimum_required(VERSION 3.16)
project(mio_cuda CUDA)

set(CMAKE_CUDA_STANDARD 14)

# set(MAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin/nvcc.exe")

add_executable(mio_cuda main.cu)

find_package(benchmark REQUIRED)


set_target_properties(mio_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(mio_cuda PRIVATE benchmark::benchmark)
