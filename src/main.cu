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

void interactive_loop(const Tokenizer &tokenizer,
                      Embedding &embedding,
                      const std::vector<float> &wout,
                      int dim, int vocab_size)
{
    std::string command;
    while (true)
    {
        std::cout << "\nCommand (predict, similarity, exit): ";
        std::getline(std::cin, command);

        if (command == "exit")
            break;

        else if (command == "predict")
        {
            std::cout << "Enter input sentence: ";
            std::string input;
            std::getline(std::cin, input);

            auto token_ids = tokenizer.tokenize(input);
            int seq_len = token_ids.size();
            if (seq_len == 0)
            {
                std::cout << "Empty input.\n";
                continue;
            }

            auto embedded = embedding.lookup(token_ids);
            auto pos_encoded = PositionalEncoding::add_positional_encoding(embedded);

            std::vector<float> input_flat(seq_len * dim);
            for (int i = 0; i < seq_len; ++i)
                for (int j = 0; j < dim; ++j)
                    input_flat[i * dim + j] = pos_encoded[i][j];

            // Compute logits
            std::vector<float> logits(vocab_size, 0.0f);
            for (int i = 0; i < dim; ++i)
                for (int j = 0; j < vocab_size; ++j)
                    logits[j] += input_flat[(seq_len - 1) * dim + i] * wout[i * vocab_size + j];

            auto probs = softmax(logits);
            int pred_id = std::max_element(probs.begin(), probs.end()) - probs.begin();

            std::cout << "Predicted next token: \"" << tokenizer.detokenize({pred_id})
                      << "\" (prob = " << probs[pred_id] << ")\n";
        }

        else if (command == "similarity")
        {
            std::string word1, word2;
            std::cout << "Enter word 1: ";
            std::getline(std::cin, word1);
            std::cout << "Enter word 2: ";
            std::getline(std::cin, word2);

            std::vector<int> id1 = tokenizer.tokenize({word1});
            std::vector<int> id2 = tokenizer.tokenize({word2});

            if (id1[0] == tokenizer.unk_id() || id2[0] == tokenizer.unk_id())
            {
                std::cout << "One or both words not found in vocab.\n";
                continue;
            }

            auto vec1 = embedding.lookup(id1)[0];
            auto vec2 = embedding.lookup(id2)[0];

            float dot = 0, norm1 = 0, norm2 = 0;
            for (int i = 0; i < dim; ++i)
            {
                dot += vec1[i] * vec2[i];
                norm1 += vec1[i] * vec1[i];
                norm2 += vec2[i] * vec2[i];
            }

            float sim = dot / (std::sqrt(norm1) * std::sqrt(norm2));
            std::cout << "Cosine similarity: " << sim << "\n";
        }

        else
        {
            std::cout << "Unknown command.\n";
        }
    }

    std::cout << "Goodbye.\n";
}

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
