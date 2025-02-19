#ifndef MNIST_H
#define MNIST_H 

#include "dataset.h"

class NN {
    public:
        const int input_size = mnist_input_size;
        const int label_size = mnist_label_size;
        
        void init();
        void forward(float *h_input, float *h_y, bool do_loss=false);
        void backward();
        void update();
        void train(int length, float *h_input, float *h_y, int epochs);

    private:
        void init_linear(float *w, float *b, int width, int height, int blockSize);

        const int threadsPerBlock = 16;
        const int batch_size = 64;
        const int size1 = 300;    
        const int size2 = 100;
        const int size3 = 10;
        const float lr = 0.003f;

        float *w1, *b1, *d_l1;
        float *w2, *b2, *d_l2;
        float *w3, *b3, *d_l3;
        float *x1, *a1, *x2, *a2, *x3;
        float *logits;
        
        float *d_loss;
        float *d_input, *d_y;
};

void run_main();

#endif // MNIST_H