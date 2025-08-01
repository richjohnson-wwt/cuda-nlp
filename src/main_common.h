#pragma once

#include <vector>

// Random normal initializer
std::vector<float> random_normal(int size, float mean = 0.0f, float stddev = 0.02f)
{
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(mean, stddev);
    std::vector<float> out(size);
    for (float &x : out)
        x = dist(gen);
    return out;
}

std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> result(logits.size());
    float max_logit = *std::max_element(logits.begin(), logits.end());

    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        result[i] = std::exp(logits[i] - max_logit);
        sum += result[i];
    }

    for (float& val : result)
        val /= sum;

    return result;
}