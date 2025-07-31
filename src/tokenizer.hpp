#pragma once
#include <string>
#include <vector>
#include <unordered_map>

class Tokenizer {
public:
    Tokenizer();
    std::vector<int> tokenize(const std::string& sentence) const;
    std::string detokenize(const std::vector<int>& tokens) const;
    int unk_id() const;

private:
    std::unordered_map<std::string, int> word_to_index;
    std::unordered_map<int, std::string> index_to_word;
    void initialize_vocab();
};
