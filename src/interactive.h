#pragma once

#include "tokenizer.hpp"
#include "embedding.hpp"
#include "attention.h"
#include "main_common.h"

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