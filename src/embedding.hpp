#pragma once
#include <vector>

class Embedding {
public:
    Embedding(int vocab_size, int embedding_dim);
    std::vector<std::vector<float>> lookup(const std::vector<int>& token_ids) const;
private:
    std::vector<std::vector<float>> embedding_matrix;
};
