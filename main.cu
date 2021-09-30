#include <iostream>
#include <math.h>
#include <chrono>
//#include <benchmark/benchmark.h>
#include <cuda_profiler_api.h>

// CUDA Kernel function to add the elements of two arrays on the GPU
__global__
void cuda_add(int n, float *x, float *y)
{

    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}


// function to add the elements of two arrays
void add(int n, float *x, float *y)
{
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

int native_main(void)
{
    int N = 1<<20; // 1M elements

    float *x = new float[N];
    float *y = new float[N];

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Run kernel on 1M elements on the CPU
    add(N, x, y);

    auto finish = std::chrono::high_resolution_clock::now();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;
    std::cout << "T: " << std::chrono::duration_cast<std::chrono::microseconds>(finish-start).count() << "ns\n";

    // Free memory
    delete [] x;
    delete [] y;

    return 0;
}

int cuda_main(void)
{
    int N = 1<<20; // 1M elements

    float *x;
    float *y;

    cudaProfilerStart();

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Run kernel on 1M elements on the CPU
    cuda_add<<<1, 1>>>(N, x, y);

    //auto finish = std::chrono::high_resolution_clock::now();

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    auto finish = std::chrono::high_resolution_clock::now();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "CUDA - Max error: " << maxError << std::endl;
    std::cout << "T: " << std::chrono::duration_cast<std::chrono::microseconds>(finish-start).count() << "ns\n";

    // Free memory
    cudaFree(x);
    cudaFree(y);

    cudaDeviceReset();
    cudaProfilerStop();

    return 0;
}

#if 0

static void bench_native_main(benchmark::State &state) {
    for (auto _ : state)
        native_main();
}
BENCHMARK(bench_native_main)->UseRealTime()->Unit(benchmark::kMillisecond);

static void bench_cuda_main(benchmark::State &state) {
    for (auto _ : state)
        cuda_main();
}
BENCHMARK(bench_cuda_main)->UseRealTime()->Unit(benchmark::kMillisecond);


// Run the benchmark
BENCHMARK_MAIN();

#else

//
int main() {
    std::cout << "Hello, World!" << std::endl;

    // Native version
    //native_main();

    // CUDA version
    cuda_main();

    return 0;
}

#endif