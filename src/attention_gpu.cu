#include "attention.h"
#include <cuda_runtime.h>

__global__ void matmul_kernel(const float *A, const float *B, float *C, int M, int N, int K)
{
    // A: M x K
    // B: K x N
    // C: M x N
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        C[row * N + col] = matrix_multiply_element(A, B, row, col, K, N);
    }
}

void matmul(const float *A, const float *B, float *C, int M, int N, int K)
{
    printf("Running matrix multiplication on GPU\n");
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    matmul_kernel<<<grid, block>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

__global__ void softmax_2d_kernel(float *matrix, int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows)
        return;

    float max_val = -1e20;
    for (int i = 0; i < cols; ++i)
        max_val = fmaxf(max_val, matrix[row * cols + i]);

    float sum = 0.0f;
    for (int i = 0; i < cols; ++i)
    {
        matrix[row * cols + i] = softmax_exp(matrix[row * cols + i], max_val);
        sum += matrix[row * cols + i];
    }

    for (int i = 0; i < cols; ++i)
        matrix[row * cols + i] /= sum;
}

void softmax_2d(float *matrix, int rows, int cols)
{
    printf("Running softmax on GPU\n");
    softmax_2d_kernel<<<rows, 1>>>(matrix, rows, cols);
    cudaDeviceSynchronize();
}

__global__ void compute_qkt_kernel(const float *Q, const float *K, float *scores, int seq_len, int dim)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < seq_len && j < seq_len) {
        float sum = 0.0f;
        for (int k = 0; k < dim; ++k) {
            sum += Q[i * dim + k] * K[j * dim + k];
        }
        scores[i * seq_len + j] = sum;
    }
}

__global__ void scale_kernel(const float *scores, float *attention, float scale, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        attention[idx] = scores[idx] * scale;
    }
}

void self_attention_forward(
    const float *input,   // [seq_len × dim] - input embeddings  
    const float *wq,      // [dim × dim] - query weight matrix
    const float *wk,      // [dim × dim] - key weight matrix
    const float *wv,      // [dim × dim] - value weight matrix
    float *output,        // [seq_len × dim] - output embeddings
    int seq_len,          // sequence length
    int dim)              // embedding dimension
{
    printf("Running self-attention forward pass on GPU\n");
    
    float *d_Q, *d_K, *d_V;
    float *d_scores, *d_attention;

    size_t mat_bytes = seq_len * dim * sizeof(float);
    size_t score_bytes = seq_len * seq_len * sizeof(float);

    // Allocate GPU memory
    CHECK_ERROR(cudaMalloc(&d_Q, mat_bytes));
    CHECK_ERROR(cudaMalloc(&d_K, mat_bytes));
    CHECK_ERROR(cudaMalloc(&d_V, mat_bytes));
    CHECK_ERROR(cudaMalloc(&d_scores, score_bytes));
    CHECK_ERROR(cudaMalloc(&d_attention, score_bytes));

    dim3 block(16, 16);
    dim3 grid((dim + block.x - 1) / block.x, (seq_len + block.y - 1) / block.y);

    // Compute Q = input × wq
    matmul_kernel<<<grid, block>>>(input, wq, d_Q, seq_len, dim, dim);
    
    // Compute K = input × wk  
    matmul_kernel<<<grid, block>>>(input, wk, d_K, seq_len, dim, dim);
    
    // Compute V = input × wv
    matmul_kernel<<<grid, block>>>(input, wv, d_V, seq_len, dim, dim);
    cudaDeviceSynchronize();

    // Compute Q × K^T → scores
    dim3 score_grid((seq_len + block.x - 1) / block.x, (seq_len + block.y - 1) / block.y);
    
    // Launch the QK^T computation
    compute_qkt_kernel<<<score_grid, block>>>(d_Q, d_K, d_scores, seq_len, dim);
    cudaDeviceSynchronize();

    // Scale scores
    float scale = 1.0f / sqrtf((float)dim);
    
    int total_elements = seq_len * seq_len;
    int scale_threads = 256;
    int scale_blocks = (total_elements + scale_threads - 1) / scale_threads;
    scale_kernel<<<scale_blocks, scale_threads>>>(d_scores, d_attention, scale, total_elements);
    cudaDeviceSynchronize();

    // Apply softmax to attention weights
    softmax_2d_kernel<<<seq_len, 1>>>(d_attention, seq_len, seq_len);
    cudaDeviceSynchronize();

    // Compute output = attention × V
    matmul_kernel<<<grid, block>>>(d_attention, d_V, output, seq_len, dim, seq_len);
    cudaDeviceSynchronize();

    // Cleanup
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_scores);
    cudaFree(d_attention);
}
