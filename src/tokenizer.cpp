#include "tokenizer.hpp"
#include <sstream>

Tokenizer::Tokenizer() {
    initialize_vocab();
}

void Tokenizer::initialize_vocab() {
    std::vector<std::string> vocab = {
        "the", "cat", "sat", "on", "mat", "dog", "ran", "to", "a", "feline", "rested", "rug", "[UNK]"
    };
    for (size_t i = 0; i < vocab.size(); ++i) {
        word_to_index[vocab[i]] = static_cast<int>(i);
        index_to_word[static_cast<int>(i)] = vocab[i];
    }
}

std::vector<int> Tokenizer::tokenize(const std::string& sentence) const {
    std::vector<int> tokens;
    std::istringstream iss(sentence);
    std::string word;
    while (iss >> word) {
        auto it = word_to_index.find(word);
        if (it != word_to_index.end())
            tokens.push_back(it->second);
        else
            tokens.push_back(word_to_index.at("[UNK]"));
    }
    return tokens;
}

std::string Tokenizer::detokenize(const std::vector<int>& tokens) const {
    std::ostringstream oss;
    for (int id : tokens) {
        auto it = index_to_word.find(id);
        if (it != index_to_word.end())
            oss << it->second << " ";
        else
            oss << "[UNK] ";
    }
    return oss.str();
}
