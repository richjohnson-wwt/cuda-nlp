#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call)                                                    \
    do                                                                      \
    {                                                                       \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess)                                             \
        {                                                                   \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << "\n"; \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

__global__ void matmul(const float *A, const float *B, float *C, int M, int N, int K)
{
    // A: M x K
    // B: K x N
    // C: M x N
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i)
            sum += A[row * K + i] * B[i * N + col];
        C[row * N + col] = sum;
    }
}

void launch_matmul(const float *A, const float *B, float *C,
                   int M, int N, int K)
{
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    matmul<<<grid, block>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

__global__ void softmax_2d(float *matrix, int rows, int cols)
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
        matrix[row * cols + i] = __expf(matrix[row * cols + i] - max_val);
        sum += matrix[row * cols + i];
    }

    for (int i = 0; i < cols; ++i)
        matrix[row * cols + i] /= sum;
}

void self_attention_forward(
    const float *d_input, // [seq_len × dim]
    const float *d_wq,    // [dim × dim]
    const float *d_wk,
    const float *d_wv,
    float *d_output, // [seq_len × dim]
    int seq_len,
    int dim)
{
    float *d_Q, *d_K, *d_V;
    float *d_scores, *d_attention;

    size_t mat_bytes = seq_len * dim * sizeof(float);
    size_t score_bytes = seq_len * seq_len * sizeof(float);

    cudaError_t err;
    err = cudaMalloc(&d_Q, mat_bytes);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaMalloc d_Q failed\n";
        std::exit(1);
    }

    err = cudaMalloc(&d_K, mat_bytes);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaMalloc d_K failed\n";
        std::exit(1);
    }

    err = cudaMalloc(&d_V, mat_bytes);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaMalloc d_V failed\n";
        std::exit(1);
    }

    err = cudaMalloc(&d_scores, score_bytes);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaMalloc d_scores failed\n";
        std::exit(1);
    }

    err = cudaMalloc(&d_attention, score_bytes);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaMalloc d_attention failed\n";
        std::exit(1);
    }

    dim3 block(16, 16);
    dim3 grid((dim + block.x - 1) / block.x, (seq_len + block.y - 1) / block.y);

    matmul<<<grid, block>>>(d_input, d_wq, d_Q, seq_len, dim, dim);
    matmul<<<grid, block>>>(d_input, d_wk, d_K, seq_len, dim, dim);
    matmul<<<grid, block>>>(d_input, d_wv, d_V, seq_len, dim, dim);
    cudaDeviceSynchronize(); // <-- Sync here!

    // Q × Kᵀ → scores
    matmul<<<grid, block>>>(d_Q, d_K, d_scores, seq_len, seq_len, dim);
    cudaDeviceSynchronize();

    // scale scores
    float scale = 1.0f / sqrtf((float)dim);
    CHECK_CUDA(cudaMemcpy(d_attention, d_scores, score_bytes, cudaMemcpyDeviceToDevice));
    cudaDeviceSynchronize();

    // In-place scale kernel (can also do on host, for now just assume it's applied)
    // Could write a kernel for this — skip for now.

    // softmax on rows
    softmax_2d<<<seq_len, 1>>>(d_attention, seq_len, seq_len);
    cudaDeviceSynchronize();

    // attention × V → output
    matmul<<<grid, block>>>(d_attention, d_V, d_output, seq_len, dim, seq_len);
    cudaDeviceSynchronize();

    // cleanup
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_scores);
    cudaFree(d_attention);
}

void run_inference_on_gpu(const std::vector<float> &h_input, int seq_len, int dim,
                          std::vector<float> &logits, const std::vector<float> &wout)
{
    int vocab_size = logits.size();

    float *d_input, *d_logits, *d_wq, *d_wk, *d_wv, *d_wout;
    size_t input_size = seq_len * dim * sizeof(float);
    size_t logits_size = vocab_size * sizeof(float);
    size_t w_size = dim * dim * sizeof(float);
    size_t wout_size = dim * vocab_size * sizeof(float);

    // Allocate memory
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_logits, logits_size);
    cudaMalloc(&d_wq, w_size);
    cudaMalloc(&d_wk, w_size);
    cudaMalloc(&d_wv, w_size);
    cudaMalloc(&d_wout, wout_size);

    // Initialize weights (you can reuse trained weights from main)
    std::vector<float> wq(dim * dim), wk(dim * dim), wv(dim * dim);
    // Fill these from training weights...

    // Copy to device
    cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_wq, wq.data(), w_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_wk, wk.data(), w_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_wv, wv.data(), w_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_wout, wout.data(), wout_size, cudaMemcpyHostToDevice);

    // Call kernel
    self_attention_forward(d_input, d_wq, d_wk, d_wv, d_wout, seq_len, dim);

    // Copy logits back
    cudaMemcpy(logits.data(), d_logits, logits_size, cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_input);
    cudaFree(d_logits);
    cudaFree(d_wq);
    cudaFree(d_wk);
    cudaFree(d_wv);
    cudaFree(d_wout);
}
