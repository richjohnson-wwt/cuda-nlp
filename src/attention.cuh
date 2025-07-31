#pragma once

__global__ void matmul(const float *A, const float *B, float *C,
                       int M, int N, int K);

void launch_matmul(const float *A, const float *B, float *C,
                   int M, int N, int K);

void self_attention_forward(
    const float *d_input, const float *d_wq,
    const float *d_wk, const float *d_wv,
    float *d_output, int seq_len, int dim);