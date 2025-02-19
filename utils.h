#ifndef UTILS_H
#define UTILS_H

#include <cstdio>
#include <tuple>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#define gpu_check() { gpu_assert(cudaPeekAtLastError(), __FILE__, __LINE__); }

inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            exit(code);
        }
   }
}

inline std::pair<dim3,dim3> get_grid2d(int width, int height, int threadsPerBlock) {
    return std::pair(
        dim3((width + threadsPerBlock - 1) / threadsPerBlock, (height + threadsPerBlock - 1) / threadsPerBlock),
        dim3(threadsPerBlock, threadsPerBlock)
    );
}

inline std::pair<dim3, dim3> get_grid1d(int height, int threadsPerBlock) {
    return std::pair(
        dim3((height + threadsPerBlock - 1) / threadsPerBlock),
        dim3(threadsPerBlock)
    );
}

void print_matrix(int width, int height, float* matrix, std::string title, bool copy = false);

int argmax(const float *arr, int size);

class Timer {
    public:
        Timer(const std::string& name) : name_(name), start_(std::chrono::high_resolution_clock::now()) {}
    
        ~Timer() {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
            std::cout << name_ << ": " << duration << " us" << std::endl;
        }
    
    private:
        std::string name_;
        std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

#endif // UTILS_H