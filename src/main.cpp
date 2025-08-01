#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <sstream>

// Include your components
#include "tokenizer.hpp"
#include "embedding.hpp"
#include "positional_encoding.hpp"
#include "attention.h"
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
    auto input_tokens = tokenizer.tokenize(input);
    auto target_tokens = tokenizer.tokenize(target);

    if (target_tokens.empty()) {
        std::cerr << "Target word not found in vocabulary.\n";
        return 1;
    }
    int target_token = target_tokens[0];

    int seq_len = input_tokens.size();
    const int dim = 32;
    const int vocab_size = 18;
    const int epochs = 100;

    std::cout << "Running on CPU with seq_len=" << seq_len << ", dim=" << dim << std::endl;

    // Model weights
    std::vector<float> wq = random_normal(dim * dim);
    std::vector<float> wk = random_normal(dim * dim);
    std::vector<float> wv = random_normal(dim * dim);
    std::vector<float> wout = random_normal(dim * vocab_size);

    // Host buffers (no CUDA memory allocation needed)
    std::vector<float> input_flat(seq_len * dim);
    std::vector<float> output(seq_len * dim);
    std::vector<float> logits(seq_len * vocab_size);

    Embedding embedding(vocab_size, dim);

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        // ----- Forward -----
        
        auto embedded = embedding.lookup(input_tokens);
        auto pos_enc = PositionalEncoding::generate(seq_len, dim);

        for (int i = 0; i < seq_len; ++i)
            for (int j = 0; j < dim; ++j)
                embedded[i][j] += pos_enc[i][j];

        // Flatten embedded input
        input_flat.clear();
        for (auto &v : embedded)
            input_flat.insert(input_flat.end(), v.begin(), v.end());

        std::vector<float> context(seq_len * dim);

        // Call CPU attention function directly with host memory
        self_attention_forward(input_flat.data(), wq.data(), wk.data(), wv.data(), 
                             output.data(), seq_len, dim);

        // Matrix multiplication: output × wout → logits
        matmul(output.data(), wout.data(), logits.data(), seq_len, vocab_size, dim);

        // Copy output for context (no CUDA memcpy needed)
        context = output;

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

    // No CUDA cleanup needed for CPU version
    std::cout << "Training completed successfully on CPU!\n";
    return 0;
}
