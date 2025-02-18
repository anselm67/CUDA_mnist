
#include <curand_kernel.h>
#include <string>
#include <c++/12/bits/chrono.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <c++/12/iomanip>
#include <tuple>

static const int input_size = 784;
static const int label_size = 10;
const int train_length = 60000;
const int test_length = 10000;

#define gpu_check() { gpu_assert(cudaPeekAtLastError(), __FILE__, __LINE__); }

inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            exit(code);
        }
   }
}

inline std::pair<dim3,dim3> get_grid2d(int width, int height, int block_size) {
    return std::pair(
        dim3(ceil(width / (float) block_size), ceil(height / (float) block_size), 1),
        dim3(block_size, block_size, 1)
    );
}

inline std::pair<dim3, dim3> get_grid1d(int height, int block_size) {
    return std::pair(
        dim3(ceil(height / (float) block_size), 1, 1),
        dim3(block_size, 1, 1)
    );
}


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

int parse_csv_line(const std::string& line, std::vector<int> &result) {
    std::stringstream ss(line);
    std::string token;

    while (std::getline(ss, token, ',')) {
        try {
            result.push_back(std::stoi(token));
        } catch (const std::invalid_argument& e) {
            std::cerr << "Invalid argument: " << token << std::endl;
            // Handle the error as needed (e.g., skip the token, return an empty vector, etc.)
        } catch (const std::out_of_range& e) {
            std::cerr << "Out of range: " << token << std::endl;
            // Handle the error as needed
        }
    }

    return result.size();
}

void read_mnist(std::ifstream& fin, int start, int length, float* x, float* y) {
    std::string line;
    std::vector<char> buffer(4096);
    std::vector<int> fields(1024);

    for (int i = start; i < start + length; ++i)
    {
        if (!std::getline(fin, line)) {
            throw std::runtime_error("Unexpected end of file.");
        }

        fields.clear();
        if ( parse_csv_line(line, fields) != 1 + input_size ) {
            throw std::runtime_error("Failed to read pixel values");
        }

        memset(y + label_size * i, 0, label_size * sizeof(float));
        y[label_size * i + fields[0]] = 1.0f;

        float* x_row = x + i * input_size;
        for (int j = 0; j < input_size; ++j) {
            x_row[j] = ((float) fields[1+j]) / 255.0f;
        }
    }
}

void print_matrix(int width, int height, float* matrix, std::string title) {
    std::cout<<title<<std::endl;
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            std::cout <<std::fixed << std::setprecision(2) << matrix[i*width+j] << ", ";
        }
        std::cout<<std::endl;
    }
}

float *cudaCopy(float *src, int size) {
    float *d_dst;
    cudaMalloc(&d_dst, size * sizeof(float));
    cudaMemcpy(d_dst, src, size * sizeof(float), cudaMemcpyHostToDevice);
    return d_dst;
}

void check_equals(const std::string& caller, float *matrix, int size, std::vector<float> expected, float epsilon = 1e-4) {
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
        output[row * width + col] = activation > 0 ? activation : 0;
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
    init_rand<<<dimGrid, dimBlock>>>(height, 1, b);
}

void train(const float *mnist_train_x, const float *mnist_train_y) {
    const int batch_size = 64;
    const int blockSize = 16;

    int size1 = 300;
    int size2 = 100;
    int size3 = 10;



    float *input, *y;
    cudaMalloc(&input, input_size * batch_size * sizeof(float));
    cudaMalloc(&y, label_size * batch_size * sizeof(float));

    float *w1, *b1, *d_l1;
    cudaMalloc(&w1, size1 * input_size * sizeof(float));
    cudaMalloc(&b1, size1 * sizeof(float));
    cudaMalloc(&d_l1, size1*batch_size*sizeof(float));
    init_linear(w1, b1, size1, input_size, blockSize);

    float *w2, *b2, *d_l2;
    cudaMalloc(&w2, size2 * size1 * sizeof(float));
    cudaMalloc(&b2, size2 * sizeof(float));
    cudaMalloc(&d_l2, size2*batch_size*sizeof(float));
    init_linear(w2, b2, size2, size1, blockSize);

    float *w3, *b3, *d_l3;
    cudaMalloc(&w3, size3 * size2 * sizeof(float));
    cudaMalloc(&b3, size3 * sizeof(float));
    cudaMalloc(&d_l3, size3*batch_size*sizeof(float));
    init_linear(w3, b3, size3, size2, blockSize);

    float *x1, *a1, *x2, *a2, *x3, *a3;
    cudaMalloc(&x1, batch_size * size1 * sizeof(float));
    cudaMalloc(&a1, batch_size * size1 * sizeof(float));

    cudaMalloc(&x2, batch_size * size2 * sizeof(float));
    cudaMalloc(&a2, batch_size * size2 * sizeof(float));

    cudaMalloc(&x3, batch_size * size3 * sizeof(float));
    cudaMalloc(&a3, batch_size * size3 * sizeof(float));

    float *loss;
    cudaMalloc(&loss, batch_size * sizeof(float));

    for (int epoch = 0; epoch < 1; epoch++) {
        for (int batch = 0; batch < train_length / batch_size; batch++) {
            cudaMemcpy(input, &mnist_train_x[batch*batch_size*input_size], 
                batch_size * input_size * sizeof(float), 
                cudaMemcpyHostToDevice);
            cudaMemcpy(y, &mnist_train_y[batch*batch_size*label_size],
                batch_size * label_size * sizeof(float), 
                cudaMemcpyHostToDevice);
            gpu_check();

            cudaDeviceSynchronize();
            gpu_check();

            auto [dimGrid, dimBlock] = get_grid2d(size1, batch_size, blockSize);
            linear<<<dimGrid, dimBlock>>>(batch_size, input_size, size1, input, w1, b1, x1);
            relu<<<dimGrid, dimBlock>>>(batch_size, size1, x1, a1);

            std::tie(dimGrid, dimBlock) = get_grid2d(size2, batch_size, blockSize);
            linear<<<dim3(1, 1, 1), dimBlock>>>(batch_size, size1, size2, a1, w2, b2, x2);
            relu<<<dimGrid, dimBlock>>>(batch_size, size2, x2, a2);

            std::tie(dimGrid, dimBlock) = get_grid2d(size3, batch_size, blockSize);
            linear<<<dimGrid, dimBlock>>>(batch_size, size2, size3, a2, w3, b3, x3);
            softmax<<<dimGrid, dimBlock>>>(batch_size, size3, x3, a3);

            std::tie(dimGrid, dimBlock) = get_grid1d(size3, blockSize);
            cross_entropy<<<dimGrid, dimBlock>>>(size3, batch_size, a3, y, loss);

            cudaDeviceSynchronize();
            gpu_check();

        }
    }
}

void run_main() {

    float *mnist_train_x = new float[input_size * train_length];
    float *mnist_train_y = new float[label_size * train_length];

    std::ifstream fin("/home/anselm/datasets/mnist/mnist_train.csv");
    if (!fin.is_open()) {
        std::cerr << "Error: Could not open the file mnist_train.csv" << std::endl;
        return;
    }
    read_mnist(fin, 0, train_length, mnist_train_x, mnist_train_y);
    fin.close();

    // print_matrix(28, 28, mnist_train_x, "MNIST Train X");
    train(mnist_train_x, mnist_train_y);

}

void test_main() {
    test_linear();
    test_relu();
    test_softmax();
    test_cross_entropy();
}

int main() {
    test_main();
}