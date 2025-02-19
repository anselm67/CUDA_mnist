#ifndef DATASET_H
#define DATASET_H

#include <string>

static const int mnist_input_size = 784;
static const int mnist_label_size = 10;
static const int mnist_train_length = 60000;
static const int mnist_test_length = 10000;

class MNIST {
    public:
    
    MNIST(int size) : size(size) {
        input = new float[size * mnist_input_size];
        labels = new float[size * mnist_label_size];
    }
    
    ~MNIST() {
        delete[] input;
        delete[] labels;
    }

    int size;
    float *input;
    float *labels;
};

bool read_mnist(const std::string& filename, MNIST& mnist);

#endif // DATASET_H