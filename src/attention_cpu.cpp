#include "attention.h"
#include <algorithm>
#include <cstring>

void matmul(const float *A, const float *B, float *C, int M, int N, int K) {
    // std::cout << "Running matrix multiplication on CPU" << std::endl;
    
    // A: M x K
    // B: K x N  
    // C: M x N
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            C[row * N + col] = matrix_multiply_element(A, B, row, col, K, N);
        }
    }
}

void softmax_2d(float *matrix, int rows, int cols) {
    // std::cout << "Running softmax on CPU" << std::endl;
    
    for (int row = 0; row < rows; ++row) {
        // Find max value in the row for numerical stability
        float max_val = -1e20f;
        for (int i = 0; i < cols; ++i) {
            max_val = std::max(max_val, matrix[row * cols + i]);
        }
        
        // Compute exponentials and sum
        float sum = 0.0f;
        for (int i = 0; i < cols; ++i) {
            matrix[row * cols + i] = softmax_exp(matrix[row * cols + i], max_val);
            sum += matrix[row * cols + i];
        }
        
        // Normalize
        for (int i = 0; i < cols; ++i) {
            matrix[row * cols + i] /= sum;
        }
    }
}

void self_attention_forward(
    const float *input,   // [seq_len × dim]
    const float *wq,      // [dim × dim]
    const float *wk,      // [dim × dim]
    const float *wv,      // [dim × dim]
    float *output,        // [seq_len × dim]
    int seq_len,
    int dim)
{
    // std::cout << "Running self-attention forward pass on CPU" << std::endl;
    
    // Allocate temporary matrices
    size_t mat_bytes = seq_len * dim * sizeof(float);
    size_t score_bytes = seq_len * seq_len * sizeof(float);
    
    float *Q = new float[seq_len * dim];
    float *K = new float[seq_len * dim];
    float *V = new float[seq_len * dim];
    float *scores = new float[seq_len * seq_len];
    float *attention = new float[seq_len * seq_len];
    
    // Compute Q = input × wq
    matmul(input, wq, Q, seq_len, dim, dim);
    
    // Compute K = input × wk  
    matmul(input, wk, K, seq_len, dim, dim);
    
    // Compute V = input × wv
    matmul(input, wv, V, seq_len, dim, dim);
    
    // Compute scores = Q × K^T
    // Note: We need to transpose K, so we swap dimensions in the matmul call
    // Q: [seq_len × dim], K^T: [dim × seq_len] -> scores: [seq_len × seq_len]
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < dim; ++k) {
                sum += Q[i * dim + k] * K[j * dim + k]; // K^T access pattern
            }
            scores[i * seq_len + j] = sum;
        }
    }
    
    // Scale scores
    float scale = 1.0f / sqrtf((float)dim);
    for (int i = 0; i < seq_len * seq_len; ++i) {
        scores[i] *= scale;
    }
    
    // Copy scores to attention matrix
    std::memcpy(attention, scores, score_bytes);
    
    // Apply softmax to attention weights
    softmax_2d(attention, seq_len, seq_len);
    
    // Compute output = attention × V
    matmul(attention, V, output, seq_len, dim, seq_len);
    
    // Cleanup
    delete[] Q;
    delete[] K;
    delete[] V;
    delete[] scores;
    delete[] attention;
}
