

find_package(Catch2 REQUIRED)

add_executable(test_runner test_tokenizer.cpp test_embedding.cpp test_attention.cu ../src/attention.cu ../src/embedding.cpp ../src/tokenizer.cpp)
target_link_libraries(test_runner PRIVATE Catch2::Catch2WithMain)
set_property(TARGET test_runner PROPERTY CUDA_ARCHITECTURES OFF)

include(CTest)
include(Catch)
catch_discover_tests(test_runner)

add_test(NAME nlpTests COMMAND test_runner)