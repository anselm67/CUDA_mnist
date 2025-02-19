#ifndef DATASET_H
#define DATASET_H

#include <string>

static const int mnist_input_size = 784;
static const int mnist_label_size = 10;
static const int mnist_train_length = 60000;
static const int mnist_test_length = 10000;

void read_mnist(std::ifstream& fin, int start, int length, float* x, float* y);

#endif // DATASET_H