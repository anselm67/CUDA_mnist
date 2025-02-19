#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <cstring>

#include "dataset.h"

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

bool read_mnist(const std::string& filename, MNIST& mnist) {
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        throw std::runtime_error(filename + " not found.");
    }

    std::string line;
    std::vector<char> buffer(4096);
    std::vector<int> fields(1024);
    int rows = 0;
    while ( std::getline(fin, line) ) {
        fields.clear();

        if ( parse_csv_line(line, fields) != 1 + mnist_input_size ) {
            throw std::runtime_error("Failed to read pixel values");
        }

        memset(mnist.labels + mnist_label_size * rows, 0, mnist_label_size * sizeof(float));
        mnist.labels[mnist_label_size * rows + fields[0]] = 1.0f;

        float* x_row = mnist.input + rows * mnist_input_size;
        for (int j = 0; j < mnist_input_size; ++j) {
            x_row[j] = ((float) fields[1+j]) / 255.0f;
        }
        rows++;
    }
    fin.close();
    return mnist.size == rows;
}
