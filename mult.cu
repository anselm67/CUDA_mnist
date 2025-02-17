#include <stdio.h>
#include <iostream>

void check_error(int err)
{
    if (err != cudaSuccess)
    {
        std::cerr << "Error: " << cudaGetErrorString((cudaError_t)err) << std::endl;
        exit(1);
    }
}

__global__ void matmul_elem(int n, float *a, float *b, float *c)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (column < n && row < n)
    {
        float dot_product = 0;
        for (int i = 0; i < n; i++)
        {
            dot_product += a[row * n + i] * b[i * n + column];
        }
        c[row * n + column] = dot_product;
    }
}

int mult()
{
    float h_A[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    float h_B[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    float h_C[3][3] = {0};

    float *d_A, *d_B, *d_C;
    int size = 3 * 3 * sizeof(float);

    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(3, 3);
    dim3 blocksPerGrid(1, 1);
    matmul_elem<<<blocksPerGrid, threadsPerBlock>>>(3, d_A, d_B, d_C);
    check_error(cudaGetLastError());

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            printf("%f ", h_C[i][j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

__global__ void broadcast_elem(int n, float *a, float *b, float *c, float *d)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < n && y < n && z < n) {
        int index = x * n * n + y * n + z;
        d[index] = a[index] + b[y * n + z] + c[z];
    }
}

int broadcast()
{
    const int n = 3;
    float h_A[n][n][n] = {
        {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
        {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
        {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
    };
    float h_B[n][n] = {{2, 2, 2}, {2, 2, 2}, {2, 2, 2}};
    float h_C[n] = {3, 3, 3};
    float h_D[n][n][n] = {0};

    float *d_A, *d_B, *d_C, *d_D;

    cudaMalloc((void **)&d_A, n * n * n * sizeof(float));
    cudaMalloc((void **)&d_B, n * n * sizeof(float));
    cudaMalloc((void **)&d_C, n * sizeof(float));
    cudaMalloc((void **)&d_D, n * n * n * sizeof(float));

    int err;
    err = cudaMemcpy(d_A, h_A, n * n * n * sizeof(float), cudaMemcpyHostToDevice);
    check_error(err);
    err = cudaMemcpy(d_B, h_B, n * n * sizeof(float), cudaMemcpyHostToDevice);
    check_error(err);
    err = cudaMemcpy(d_C, h_C, n * sizeof(float), cudaMemcpyHostToDevice);
    check_error(err);

    dim3 threadsPerBlock(n, n, n);
    dim3 blocksPerGrid(1, 1, 1);
    broadcast_elem<<<blocksPerGrid, threadsPerBlock>>>(3, d_A, d_B, d_C, d_D);
    check_error(cudaGetLastError());

    cudaMemcpy(h_D, d_D, sizeof(h_D), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                printf("%f ", h_D[i][j][k]);
            }
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);

    return 0;
}

int main()
{
    // mult();
    broadcast();
    return 0;
}