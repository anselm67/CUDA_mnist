#ifndef KERNELS_H
#define KERNELS_H
#include <vector>
#include <cuda_runtime.h>
#include <string>
#include <fstream>


// Declare your CUDA kernels here
__global__ void linear(int batch_size, int dim_in, int dim_out, float *input, float *weights, float *biases, float *output);
__global__ void relu(int width, int height, float *input, float *output);
__global__ void softmax(int width, int height, float *input, float *output);
__global__ void cross_entropy(int width, int height, float *predictions, float *labels, float *loss);
__global__ void cross_entropy_backward(int width, int height, float *predictions, float *labels, float *gradients);
__global__ void linear_backward(int batch_size, int dim_out, int dim_in, float *weights, float *biases, float *d_out, float *d_in);
__global__ void relu_backward(int width, int height, float *input, float *d_out, float *d_in);
__global__ void linear_update(int dim_out, int dim_in, int batch_size, float lr, float *weights, float *biases, float *activations, float *d_out);

__global__ void init_rand(int width, int height, float *mat);

void test_linear();
void test_relu();
void test_softmax();
void test_cross_entropy();

#endif // KERNELS_H