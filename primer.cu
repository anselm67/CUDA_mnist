#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <fstream>

__global__ void matrixAdd(int n, const float* A, const float* B, float* C) {
    int i  = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void check_error(int err) {
    if (err != cudaSuccess) {
        std::cerr << "Error: " << cudaGetErrorString((cudaError_t) err) << std::endl;
        exit(1);
    }
}


int max_threads() {
    int deviceId = 0; // Assuming you want to query the first device
    cudaDeviceProp deviceProp;
    cudaError_t error_id = cudaGetDeviceProperties(&deviceProp, deviceId);

    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error_id, __LINE__);
        printf("cudaGetDeviceProperties error: %s.\n", cudaGetErrorString(error_id));
        exit(EXIT_FAILURE);
    }

    return  deviceProp.maxThreadsPerBlock;
}

std::pair<float, float> measure(int n) {
    const int SIZE = n * sizeof(float);

    float *h_A, *h_B, *h_C;
    h_A = new float[n];
    h_B = new float[n];
    h_C = new float[n];
    for (int i = 0; i < n; i++) {
        h_A[i] = i;
        h_B[i] = n - i;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, SIZE);
    cudaMalloc((void**)&d_B, SIZE);
    cudaMalloc((void**)&d_C, SIZE);

    cudaMemcpy(d_A, h_A, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, SIZE, cudaMemcpyHostToDevice);

    int BLOCK_SIZE = max_threads();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop); 

    cudaEventRecord(start, 0);
    matrixAdd<<<ceil(n / (float) BLOCK_SIZE), BLOCK_SIZE>>>(n, d_A, d_B, d_C);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++) {
        h_C[i] = h_A[i] + h_B[i];
    }
    auto stop_cpu = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_cpu - start_cpu);

    cudaDeviceSynchronize();
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete [] h_A;
    delete [] h_B;
    delete [] h_C;

    return std::make_pair(duration.count(), milliseconds * 1000.0f);
}


int main() {
    std::ofstream outfile("results.csv");
    outfile << "N,CPU Time (us),GPU Time (ns)" << std::endl;

    for (int power = 3; power <= 26; power++) {
        int n = pow(2, power);
        std::pair<float, float> results = measure(n);
        float cpu_time = results.first;
        float gpu_time = results.second;
        outfile << n << "," << cpu_time << "," << gpu_time << std::endl;
        std::cout << "N=" << n << ", CPU Time=" << cpu_time << " ns, GPU Time=" << gpu_time << " ns" << std::endl;
    }

    outfile.close();
    std::cout << "Results saved to results.csv" << std::endl;

    return 0;
}
