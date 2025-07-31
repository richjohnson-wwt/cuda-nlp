#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include "../src/tokenizer.hpp"

TEST_CASE("Tokenizer roundtrip") {
    Tokenizer tokenizer;
    std::string sentence = "the cat sat on the mat";
    auto tokens = tokenizer.tokenize(sentence);
    REQUIRE(tokens.size() == 6);
    auto detok = tokenizer.detokenize(tokens);
    REQUIRE(detok.find("cat") != std::string::npos);
}
