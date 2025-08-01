#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <sstream>

// Include your components
#include "tokenizer.hpp"
#include "embedding.hpp"
#include "positional_encoding.hpp"
#include "attention.cuh"
#include "interactive.h"
#include "main_common.h"


int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input sentence> <target word>\n";
        return 1;
    }

    std::string input = argv[1];
    std::string target = argv[2];

    std::cout << "Input: \"" << input << "\"\n";
    std::cout << "Target: \"" << target << "\"\n";

    Tokenizer tokenizer;
    // std::string input = "the cat sat on the";
    // std::string target = "mat";

    auto input_tokens = tokenizer.tokenize(input);    // [2, 0, 6, 8, 2]
    int target_token = tokenizer.tokenize(target)[0]; // e.g., 4

    const int seq_len = input_tokens.size();
    const int dim = 8;
    const int vocab_size = 18;
    const int epochs = 100;

    // Model weights
    std::vector<float> wq = random_normal(dim * dim);
    std::vector<float> wk = random_normal(dim * dim);
    std::vector<float> wv = random_normal(dim * dim);
    std::vector<float> wout = random_normal(dim * vocab_size);

    // Device buffers
    float *d_input, *d_wq, *d_wk, *d_wv, *d_output;
    float *d_wout, *d_logits;

    cudaMalloc(&d_input, seq_len * dim * sizeof(float));
    cudaMalloc(&d_wq, dim * dim * sizeof(float));
    cudaMalloc(&d_wk, dim * dim * sizeof(float));
    cudaMalloc(&d_wv, dim * dim * sizeof(float));
    cudaMalloc(&d_output, seq_len * dim * sizeof(float));

    cudaMalloc(&d_wout, dim * vocab_size * sizeof(float));
    cudaMalloc(&d_logits, seq_len * vocab_size * sizeof(float));

    Embedding embedding(vocab_size, dim);

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        // ----- Forward -----
        
        auto embedded = embedding.lookup(input_tokens);
        auto pos_enc = PositionalEncoding::generate(seq_len, dim);

        for (int i = 0; i < seq_len; ++i)
            for (int j = 0; j < dim; ++j)
                embedded[i][j] += pos_enc[i][j];

        std::vector<float> input_flat;
        for (auto &v : embedded)
            input_flat.insert(input_flat.end(), v.begin(), v.end());

        std::vector<float> context(seq_len * dim);
        std::vector<float> logits(seq_len * vocab_size);

        cudaMemcpy(d_input, input_flat.data(), input_flat.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_wq, wq.data(), wq.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_wk, wk.data(), wk.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_wv, wv.data(), wv.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_wout, wout.data(), wout.size() * sizeof(float), cudaMemcpyHostToDevice);

        self_attention_forward(d_input, d_wq, d_wk, d_wv, d_output, seq_len, dim);

        launch_matmul(d_output, d_wout, d_logits, seq_len, vocab_size, dim);
        cudaMemcpy(context.data(), d_output, context.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(logits.data(), d_logits, logits.size() * sizeof(float), cudaMemcpyDeviceToHost);

        // ----- Softmax + Loss -----
        int last = seq_len - 1;
        float max_logit = -1e9f;
        for (int i = 0; i < vocab_size; ++i)
            max_logit = std::max(max_logit, logits[last * vocab_size + i]);

        std::vector<float> probs(vocab_size);
        float sum = 0.0f;
        for (int i = 0; i < vocab_size; ++i)
        {
            probs[i] = std::exp(logits[last * vocab_size + i] - max_logit);
            sum += probs[i];
        }
        for (float &p : probs)
            p /= sum;

        float loss = -std::log(probs[target_token]);

        // ----- Gradients -----
        std::vector<float> dlogits(vocab_size);
        for (int i = 0; i < vocab_size; ++i)
            dlogits[i] = probs[i];
        dlogits[target_token] -= 1.0f;

        std::vector<float> grad_wout(dim * vocab_size, 0.0f);
        for (int i = 0; i < vocab_size; ++i)
        {
            for (int j = 0; j < dim; ++j)
            {
                grad_wout[j * vocab_size + i] = dlogits[i] * context[last * dim + j];
            }
        }

        // ----- SGD Update -----
        float lr = 0.5f;
        for (int i = 0; i < wout.size(); ++i)
            wout[i] -= lr * grad_wout[i];

        // Print progress
        int pred = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
        std::cout << "Epoch " << epoch
                  << " | Loss: " << loss
                  << " | Predicted token ID: " << pred
                  << " | Target ID: " << target_token << "\n";
    }

    interactive_loop(tokenizer, embedding, wout, dim, vocab_size);

    // ----- Cleanup -----
    cudaFree(d_input);
    cudaFree(d_wq);
    cudaFree(d_wk);
    cudaFree(d_wv);
    cudaFree(d_output);
    cudaFree(d_wout);
    cudaFree(d_logits);

    return 0;
}
