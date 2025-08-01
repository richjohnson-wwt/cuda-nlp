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

HD inline void my_softmax_exp(float *matrix, int row, int cols) {
    float max_val = -1e20f;
    for (int i = 0; i < cols; ++i) {
        float current_val = matrix[row * cols + i];
        max_val = (current_val > max_val) ? current_val : max_val;
    }
    
    // Compute exponentials and sum
    float sum = 0.0f;
    for (int i = 0; i < cols; ++i) {
        matrix[row * cols + i] = expf(matrix[row * cols + i] - max_val);
        sum += matrix[row * cols + i];
    }
    
    // Normalize
    for (int i = 0; i < cols; ++i) {
        matrix[row * cols + i] /= sum;
    }
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

// Backward compatibility wrapper (for existing main.cu)
void launch_matmul(const float *A, const float *B, float *C, int M, int N, int K);

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
