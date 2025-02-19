#include "utils.h"
#include <iomanip>

int argmax(const float *v, int size) {
    int max_index = 0;
    float max_value = v[0];
    for (int i = 1; i < size; i++) {
        if (v[i] > max_value) {
            max_value = v[i];
            max_index = i;
        }
    }
    return max_index;
}

void print_matrix(int width, int height, float* matrix, std::string title, bool copy) {
    if (copy) {
        float *h_matrix = new float[width * height];
        cudaMemcpy(h_matrix, matrix, width * height * sizeof(float), cudaMemcpyDeviceToHost);
        print_matrix(width, height, h_matrix, title);
        delete [] h_matrix;
    } else {
        std::cout<<title<<std::endl;
        for(int i = 0; i < height; i++) {
            for(int j = 0; j < width; j++) {
                std::cout <<std::fixed << std::setprecision(3) << matrix[i*width+j] << ", ";
                if (std::isnan(matrix[i*width+j]) ) {
                    std::cerr << "NaN value at " << i << ", " << j << std::endl;
                }
            }
            std::cout<<std::endl;
        }
    }
}
