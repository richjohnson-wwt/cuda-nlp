#pragma once
#include <vector>

class PositionalEncoding {
public:
    static std::vector<std::vector<float>> generate(int seq_len, int dim);
};
