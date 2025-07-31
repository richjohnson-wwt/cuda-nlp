#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>

__global__ void softmax_kernel(float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    float max_val = -CUDART_INF_F;
    for (int i = 0; i < cols; ++i) {
        float val = input[row * cols + i];
        if (val > max_val) max_val = val;
    }

    float sum = 0.0f;
    for (int i = 0; i < cols; ++i) {
        float e = expf(input[row * cols + i] - max_val);
        output[row * cols + i] = e;
        sum += e;
    }

    for (int i = 0; i < cols; ++i) {
        output[row * cols + i] /= sum;
    }
}
