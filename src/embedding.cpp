#include "embedding.hpp"
#include <cstdlib>
#include <ctime>

Embedding::Embedding(int vocab_size, int embedding_dim)
    : embedding_matrix(vocab_size, std::vector<float>(embedding_dim)) {
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    for (int i = 0; i < vocab_size; ++i)
        for (int j = 0; j < embedding_dim; ++j)
            embedding_matrix[i][j] = ((float) std::rand() / RAND_MAX) * 0.1f;
}

std::vector<std::vector<float>> Embedding::lookup(const std::vector<int>& token_ids) const {
    std::vector<std::vector<float>> result;
    for (int id : token_ids) {
        result.push_back(embedding_matrix.at(id));
    }
    return result;
}
