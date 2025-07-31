#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <cmath>


extern void self_attention_forward(const float*, const float*, const float*, const float*, float*, int, int);

TEST_CASE("Self-attention sanity test", "[attention]") {
    const int seq_len = 4;
    const int dim = 8;

    std::vector<float> input(seq_len * dim, 0.1f);
    std::vector<float> wq(dim * dim, 0.01f);
    std::vector<float> wk(dim * dim, 0.02f);
    std::vector<float> wv(dim * dim, 0.03f);
    std::vector<float> output(seq_len * dim, 0.0f);

    float *d_input, *d_wq, *d_wk, *d_wv, *d_output;
    cudaMalloc(&d_input, input.size() * sizeof(float));
    cudaMalloc(&d_wq, wq.size() * sizeof(float));
    cudaMalloc(&d_wk, wk.size() * sizeof(float));
    cudaMalloc(&d_wv, wv.size() * sizeof(float));
    cudaMalloc(&d_output, output.size() * sizeof(float));

    cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wq, wq.data(), wq.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wk, wk.data(), wk.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wv, wv.data(), wv.size() * sizeof(float), cudaMemcpyHostToDevice);

    self_attention_forward(d_input, d_wq, d_wk, d_wv, d_output, seq_len, dim);
    cudaMemcpy(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Check a few outputs aren't zero
    for (int i = 0; i < dim; ++i)
        REQUIRE(std::abs(output[i]) > 0.0001f);

    cudaFree(d_input); cudaFree(d_wq); cudaFree(d_wk); cudaFree(d_wv); cudaFree(d_output);
}
