
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include <iostream>
#include <fstream>

#include "mnist.h"
#include "kernels.h"
#include "utils.h"
#include "dataset.h"
#include <iomanip>

void NN::init_linear(float *w, float *b, int width, int height, int blockSize) {
    auto [dimGrid, dimBlock] = get_grid2d(width, height, blockSize);
    init_rand<<<dimGrid, dimBlock>>>(width, height, w);

    std::tie(dimGrid, dimBlock) = get_grid1d(height, blockSize);
    init_rand<<<dimGrid, dimBlock>>>(1, height, b);
}

void NN::init() {
    cudaMalloc(&w1, size1 * input_size * sizeof(float));
    cudaMalloc(&b1, size1 * sizeof(float));
    cudaMalloc(&d_l1, size1*batch_size*sizeof(float));
    this->init_linear(w1, b1, size1, input_size, threadsPerBlock);

    cudaMalloc(&w2, size2 * size1 * sizeof(float));
    cudaMalloc(&b2, size2 * sizeof(float));
    cudaMalloc(&d_l2, size2*batch_size*sizeof(float));
    init_linear(w2, b2, size2, size1, threadsPerBlock);

    cudaMalloc(&w3, size3 * size2 * sizeof(float));
    cudaMalloc(&b3, size3 * sizeof(float));
    cudaMalloc(&d_l3, size3*batch_size*sizeof(float));
    init_linear(w3, b3, size3, size2, threadsPerBlock);

    cudaMalloc(&x1, batch_size * size1 * sizeof(float));
    cudaMalloc(&a1, batch_size * size1 * sizeof(float));

    cudaMalloc(&x2, batch_size * size2 * sizeof(float));
    cudaMalloc(&a2, batch_size * size2 * sizeof(float));

    cudaMalloc(&x3, batch_size * size3 * sizeof(float));
    cudaMalloc(&logits, batch_size * size3 * sizeof(float));

    cudaMalloc(&d_input, batch_size * input_size * sizeof(float));
    cudaMalloc(&d_y, batch_size * label_size * sizeof(float));
    cudaMalloc(&d_loss, batch_size * sizeof(float));

    gpu_check();
}

void NN::forward(float *h_input, float *h_y, bool do_loss) {
    cudaMemcpy(d_input, h_input, 
        batch_size * input_size * sizeof(float), 
        cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y,
        batch_size * label_size * sizeof(float), 
        cudaMemcpyHostToDevice);
    gpu_check();

    auto [blocks, threads] = get_grid2d(size1, batch_size, threadsPerBlock);
    linear<<<blocks, threads>>>(batch_size, input_size, size1, d_input, w1, b1, x1);
    relu<<<blocks, threads>>>(size1, batch_size, x1, a1);

    std::tie(blocks, threads) = get_grid2d(size2, batch_size, threadsPerBlock);
    linear<<<blocks, threads>>>(batch_size, size1, size2, a1, w2, b2, x2);
    relu<<<blocks, threads>>>(size2, batch_size, x2, a2);

    std::tie(blocks, threads) = get_grid2d(size3, batch_size, threadsPerBlock);
    linear<<<blocks, threads>>>(batch_size, size2, size3, a2, w3, b3, x3);
    softmax<<<blocks, threads>>>(size3, batch_size, x3, logits);

    if (do_loss) {
        std::tie(blocks, threads) = get_grid1d(size3, threadsPerBlock);
        cross_entropy<<<blocks, threads>>>(size3, batch_size, logits, d_y, d_loss);
    }

    gpu_check();
}

void NN::backward() {
    auto [blocks, threads] = get_grid2d(size3, batch_size, threadsPerBlock);
    cross_entropy_backward<<<blocks, threads>>>(size3, batch_size, logits, d_y, d_l3);

    std::tie(blocks, threads) = get_grid2d(size2, batch_size, threadsPerBlock);
    linear_backward<<<blocks, threads>>>(batch_size, size3, size2, w3, b3, d_l3, d_l2);
    relu_backward<<<blocks, threads>>>(size2, batch_size, a2, d_l2, d_l2);

    std::tie(blocks, threads) = get_grid2d(size1, batch_size, threadsPerBlock);
    linear_backward<<<blocks, threads>>>(batch_size, size2, size1, w2, b2, d_l2, d_l1);
    relu_backward<<<blocks, threads>>>(size1, batch_size, a1, d_l1, d_l1);
    gpu_check();
}

void NN::update() {
    auto [blocks, threads] = get_grid2d(size3, size2, threadsPerBlock);
    linear_update<<<blocks, threads>>>(size3, size2, batch_size, lr, w3, b3, a2, d_l3);

    std::tie(blocks, threads) = get_grid2d(size2, size1, threadsPerBlock);
    linear_update<<<blocks, threads>>>(size2, size1, batch_size, lr, w2, b2, a1, d_l2);

    std::tie(blocks, threads) = get_grid2d(size1, input_size, threadsPerBlock);
    linear_update<<<blocks, threads>>>(size1, input_size, batch_size, lr, w1, b1, d_input, d_l1);            
    gpu_check();
}

void NN::train(int epochs, const MNIST &train_dataset, const MNIST &test_dataset) {
    float *h_input = train_dataset.input, *h_y = train_dataset.labels;
    float h_loss[batch_size] = { 0 };
    float h_logits[size3 * batch_size] = { 0 };

    for (int epoch = 0; epoch < epochs; epoch++) {
        auto start_time = std::chrono::high_resolution_clock::now();
        int total = 0;
        int correct = 0;
        float cum_loss = 0;

        for (int batch = 0; batch < train_dataset.size / batch_size; batch++) {
            total += batch_size;

            this->forward(
                h_input + batch * batch_size * input_size, 
                h_y + batch * batch_size * label_size, 
                true        // Request loss.
            );
            // Computes loss.
            cudaMemcpy(&h_loss, d_loss, batch_size * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_logits, logits, size3 * batch_size * sizeof(float), cudaMemcpyDeviceToHost);

            for (int i = 0; i < batch_size; i++) {
                int offset = batch * batch_size * label_size + i * label_size;
                int predicted_class = argmax(h_logits + i * label_size, label_size);
                int true_class = argmax(h_y + offset, label_size);
            
                if (predicted_class == true_class) {
                    correct++;
                }
                cum_loss += h_loss[i];
            }
            
            this->backward();
            this->update();
        }

        auto stop_time = std::chrono::high_resolution_clock::now();
        float epoch_time = std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(stop_time - start_time).count();
        float valid_accuracy = this->valid(test_dataset);
        std::cout << std::fixed << std::setprecision(3)
            << "Epoch " << epoch << ", " << epoch_time << "ms:"
            << " accuracy: " << (float) correct / total 
            << " valid-accuracy: " << valid_accuracy
            << ", loss: " << cum_loss << std::endl;
    }
}

float NN::valid(const MNIST& dataset) {
    float *h_input = dataset.input, *h_y = dataset.labels;
    float h_logits[size3 * batch_size] = { 0 };

    int total = 0;
    int correct = 0;

    for (int batch = 0; batch < dataset.size / batch_size; batch++) {
        total += batch_size;

        this->forward(
            h_input + batch * batch_size * input_size, 
            h_y + batch * batch_size * label_size, 
            false        // No loss necessary.
        );
        cudaMemcpy(&h_logits, logits, size3 * batch_size * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < batch_size; i++) {
            int offset = batch * batch_size * label_size + i * label_size;
            int predicted_class = argmax(h_logits + i * label_size, label_size);
            int true_class = argmax(h_y + offset, label_size);
        
            if (predicted_class == true_class) {
                correct++;
            }
        }
    }

    return (float) correct / total;
}

void nn_main() {

    NN nn;
    nn.init();

    MNIST train_dataset(mnist_train_length);
    MNIST test_dataset(mnist_test_length);
    read_mnist("/home/anselm/datasets/mnist/mnist_train.csv", train_dataset);
    read_mnist("/home/anselm/datasets/mnist/mnist_test.csv", test_dataset);
    

    nn.train(60, train_dataset, test_dataset);
}

void test_main() {
    test_linear();
    test_relu();
    test_softmax();
    test_cross_entropy();
}

int main() {
    nn_main();
}