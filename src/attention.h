#pragma once

#include <cmath>
#include <iostream>
#include <vector>

#ifdef __CUDACC__
    #define HD __host__ __device__
#else
    #define HD
#endif

// Error checking macro for both CPU and GPU builds
#ifdef USE_CUDA
    #define CHECK_ERROR(call)                                                \
        do                                                                   \
        {                                                                    \
            cudaError_t err = call;                                          \
            if (err != cudaSuccess)                                          \
            {                                                                \
                std::cerr << "CUDA Error: " << cudaGetErrorString(err) << "\n"; \
                std::exit(EXIT_FAILURE);                                     \
            }                                                                \
        } while (0)
#else
    #define CHECK_ERROR(call) call
#endif

// Core mathematical operations that work on both CPU and GPU
HD inline float softmax_exp(float x, float max_val) {
    return expf(x - max_val);
}

HD inline float matrix_multiply_element(const float *A, const float *B, int row, int col, int K, int N) {
    float sum = 0.0f;
    for (int i = 0; i < K; ++i) {
        sum += A[row * K + i] * B[i * N + col];
    }
    return sum;
}

// Public interface functions (same signature for both CPU and GPU)
void matmul(const float *A, const float *B, float *C, int M, int N, int K);

void softmax_2d(float *matrix, int rows, int cols);

void self_attention_forward(
    const float *input,   // [seq_len × dim] - input embeddings
    const float *wq,      // [dim × dim] - query weight matrix
    const float *wk,      // [dim × dim] - key weight matrix  
    const float *wv,      // [dim × dim] - value weight matrix
    float *output,        // [seq_len × dim] - output embeddings
    int seq_len,          // sequence length
    int dim               // embedding dimension
);
