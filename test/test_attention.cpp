#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <cmath>
#include "../src/attention.h"

TEST_CASE("Self-attention sanity test", "[attention]") {
    const int seq_len = 4;
    const int dim = 8;

    std::vector<float> input(seq_len * dim, 0.1f);
    std::vector<float> wq(dim * dim, 0.01f);
    std::vector<float> wk(dim * dim, 0.02f);
    std::vector<float> wv(dim * dim, 0.03f);
    std::vector<float> output(seq_len * dim, 0.0f);

    // Call CPU attention function directly with host memory (no CUDA allocation needed)
    self_attention_forward(input.data(), wq.data(), wk.data(), wv.data(), 
                         output.data(), seq_len, dim);

    // Check a few outputs aren't zero
    for (int i = 0; i < dim; ++i)
        REQUIRE(std::abs(output[i]) > 0.0001f);
    
    // Additional check: ensure output values are reasonable (not NaN or inf)
    for (int i = 0; i < seq_len * dim; ++i) {
        REQUIRE(std::isfinite(output[i]));
    }
}
