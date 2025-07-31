#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>

#include "../src/embedding.hpp"

TEST_CASE("Embedding lookup") {
    Embedding emb(13, 4);
    std::vector<int> ids = {0, 1, 2};
    auto vectors = emb.lookup(ids);
    REQUIRE(vectors.size() == 3);
    REQUIRE(vectors[0].size() == 4);
}