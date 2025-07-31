#pragma once
#include <vector>

class PositionalEncoding {
public:
    static std::vector<std::vector<float>> generate(int seq_len, int dim);
    static std::vector<std::vector<float>> add_positional_encoding(const std::vector<std::vector<float>>& embeddings);
};
