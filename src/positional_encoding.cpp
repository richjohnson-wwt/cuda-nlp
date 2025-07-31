#include "positional_encoding.hpp"
#include <cmath>

std::vector<std::vector<float>> PositionalEncoding::generate(int seq_len, int dim) {
    std::vector<std::vector<float>> pe(seq_len, std::vector<float>(dim));

    for (int pos = 0; pos < seq_len; ++pos) {
        for (int i = 0; i < dim; ++i) {
            float angle = pos / std::pow(10000.0f, (2 * (i / 2)) / (float)dim);
            if (i % 2 == 0) {
                pe[pos][i] = std::sin(angle);
            } else {
                pe[pos][i] = std::cos(angle);
            }
        }
    }

    return pe;
}

std::vector<std::vector<float>> PositionalEncoding::add_positional_encoding(const std::vector<std::vector<float>>& embeddings) {
    int seq_len = embeddings.size();
    int dim = embeddings[0].size();

    std::vector<std::vector<float>> encoded = embeddings;

    for (int pos = 0; pos < seq_len; ++pos) {
        for (int i = 0; i < dim; ++i) {
            float angle = pos / std::pow(10000.0f, 2.0f * (i / 2) / dim);
            if (i % 2 == 0)
                encoded[pos][i] += std::sin(angle);
            else
                encoded[pos][i] += std::cos(angle);
        }
    }

    return encoded;
}