
#include <curand_kernel.h>
#include <string>
#include <c++/12/bits/chrono.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <c++/12/iomanip>
#include <tuple>

#include "utils.h"

static float *cudaCopy(float *src, int size) {
    float *d_dst;
    cudaMalloc(&d_dst, size * sizeof(float));
    cudaMemcpy(d_dst, src, size * sizeof(float), cudaMemcpyHostToDevice);
    return d_dst;
}

static void check_equals(const std::string& caller, float *matrix, int size, std::vector<float> expected, float epsilon = 1e-4) {
    for (int i = 0; i < size; i++) {
        if (fabs(matrix[i] - expected[i]) > epsilon) {
            std::cerr << "Error in " << caller << ": Expected " << expected[i] << " but got " << matrix[i] << std::endl;
            return;
        }
    }
    std::cout << "Success in " << caller << ": All values match within epsilon " << epsilon << "." << std::endl;
}

__global__ void linear(
    int batch_size, int dim_in, int dim_out,
    float *input,       // (Cin, B)
    float *weights,     // (Cout, Cin)
    float *biases,      // (Cout)
    float *output       // (Cout, B)
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < batch_size && col < dim_out) {
        output[row * dim_out + col] = biases[col];
        for (int i = 0; i < dim_in; i++) {
            output[row * dim_out + col] += weights[i * dim_out + col] * input[row * dim_in + i];
        }
    }
}

__global__ void linear_backward(
    int batch_size, int dim_in, int dim_out,
    float *weights,     // (Cout, Cin)
    float *biases,      // (Cout)
    float *d_in,        
    float *d_out
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < batch_size && col < dim_out) {
        float dl = 0.f;
        for (int i = 0; i < dim_in; i++) {
            float w = weights[i * dim_out + col];
            dl += w * d_in[row * dim_in + i];
        }
        d_out[row * dim_out + col] = dl;
    }
}

__global__ void linear_update(
    int width, int height, int batch_size,
    float lr, float *weights, float *biases,
    float *activations, float *d_l
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < height && col < width) {
        float dw = 0.f, db = 0.f;
        for (int i = 0; i < batch_size; i++) {
            float act = activations[i * height + row];
            float dl = d_l[i * width + col];
            dw += act * dl;
            db += dl;
        }
        weights[row * width + col] -= lr * dw / batch_size;
        biases[col] -= lr * db / batch_size;
    }
}

void test_linear() {
    const int batch_size = 4;
    float input[12] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };    // 4 x 3
    float weights[6] = { 1, 1, 1, 1, 1, 1 };        // 3 x 2
    float biases[2] = { 1, 1 };                     // 2
    float output[8] = { 0, } ;                        // 4 x 2

    float *d_input = cudaCopy(input, 4 * 3 * sizeof(float));
    float *d_weights = cudaCopy(weights, 4 * 2 * sizeof(float));
    float *d_biases = cudaCopy(biases, 2 * sizeof(float));
    float *d_output = cudaCopy(output, 4 * 2 * sizeof(float));

    auto [dimGrid, dimBlock] = get_grid2d(3, 4, 16);
    linear<<<dimGrid, dimBlock>>>(batch_size, 3, 2, d_input, d_weights, d_biases, d_output);

    cudaMemcpy(output, d_output, 4 * 2 * sizeof(float), cudaMemcpyDeviceToHost);
    
    check_equals(__func__, output, 4 * 2, { 7, 7, 16, 16, 25, 25, 34, 34 });
}

__global__ void relu(
    int width, int height, 
    float *input,    // (C, B)
    float *output    // (C, B)
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < height && col < width) {
        float activation = input[row * width + col];
        output[row * width + col] = activation > 0.f ? activation : 0.f;
    }
}

__global__ void relu_backward(
    int width, int height,
    float *a, float *d_in, float *d_out
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < height && col < width) {
        float act = a[row * width + col];
        d_out[row * width + col] = act > 0.f ? d_in[row * width + col] : 0.f;
    }
}

void test_relu() {
    int width = 4;
    int height = 2;     // batch dimension.

    float input[8] = { 1, -1, 2, -2, 3, -3, 4, -4 };
    float output[8] = { 0 };

    float *d_input = cudaCopy(input, 8);
    float *d_output = cudaCopy(output, 8);

    auto [dimGrid, dimBlock] = get_grid2d(3, 4, 16);
    relu<<<dimGrid, dimBlock>>>(width, height, d_input, d_output);

    cudaMemcpy(output, d_output, 8 * sizeof(float), cudaMemcpyDeviceToHost);
    check_equals(__func__, output, 8, { 1, 0, 2, 0, 3, 0, 4, 0 });
}

__global__ void softmax(
    int width, int height,
    float *input,   // (C, B)
    float *output   // (C, B)
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < height && col < width) {
        float maxval = input[row * width];
        for (int i = 1; i < width; i++) {
            maxval = max(maxval, input[row * width + i]);
        }
        float divisor = 0;
        for (int i = 0; i < width; i++) {
            divisor += exp(input[row * width + i] - maxval);
        }
        output[row * width + col] = exp(input[row * width + col] - maxval) / divisor;
    }
}

void test_softmax() {
    int width = 2;
    int height = 4;     // batch dimension.

    float input[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    float output[8] = { 0 };

    float *d_input = cudaCopy(input, 8);
    float *d_output = cudaCopy(output, 8);

    auto [dimGrid, dimBlock] = get_grid2d(2, 4, 16);
    softmax<<<dimGrid, dimBlock>>>(width, height, d_input, d_output);

    cudaMemcpy(output, d_output, 8 * sizeof(float), cudaMemcpyDeviceToHost);
    check_equals(__func__, output, 8, { 0.2689, 0.7311, 0.2689, 0.7311, 0.2689, 0.7311, 0.2689, 0.7311 });
}

__global__ void cross_entropy(
    int width, int height,
    float *yhat,    // predicted probabilities (10, B)
    float *y,       // one-hot encoded labels (10, B)
    float *output   // cross entropy loss returned.
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < height) {
        float loss = 0;
        for (int i = 0; i < width; i++) {
            loss -= y[index * width + i] * log(max(1e-6, yhat[index * width + i]));
        }
        output[index] = loss;
    }
}

__global__ void cross_entropy_backward(
    int width, int height,
    float *yhat,    // predicted probabilities (10, B)
    float *y,       // one-hot encoded labels (10, B)
    float *output   // gradient return.
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < height && col < width) {
        output[row * width + col] = yhat[row * width + col] - y[row * width + col];
    }
}

void test_cross_entropy() {
    int width = 10;
    int height = 2;     // batch dimension.

    float yhat[20] = {
         1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    };
    float y[20] = { 
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    };
    float output[2] = { 0 };

    float *d_yhat = cudaCopy(yhat, 20);
    float *d_y = cudaCopy(y, 20);
    float *d_output = cudaCopy(output, 2);

    auto [dimGrid, dimBlock] = get_grid2d(10, 2, 16);
    cross_entropy<<<dimGrid, dimBlock>>>(width, height, d_yhat, d_y, d_output);

    cudaMemcpy(output, d_output, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    check_equals(__func__, output, 2, { 0.0, 0.6931 });

}

__global__ void init_rand(int width, int height, float *mat) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < height && col < width) {
        curandState state;
        curand_init(42, row * width + col, 0, &state);
        mat[row * width + col] = curand_normal(&state) * sqrtf(2.0 / height);
    }
}

void init_linear(float *w, float *b, int width, int height, int blockSize) {
    auto [dimGrid, dimBlock] = get_grid2d(width, height, blockSize);
    init_rand<<<dimGrid, dimBlock>>>(width, height, w);

    std::tie(dimGrid, dimBlock) = get_grid1d(height, blockSize);
    init_rand<<<dimGrid, dimBlock>>>(1, height, b);
}


