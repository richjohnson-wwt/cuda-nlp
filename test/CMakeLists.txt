

find_package(Catch2 REQUIRED)

if(USE_CUDA)
    set(TEST_SOURCES
        test_tokenizer.cpp
        test_embedding.cpp
        test_attention.cu
        ../src/attention_gpu.cu
        ../src/embedding.cpp
        ../src/tokenizer.cpp
    )
else()
    set(TEST_SOURCES
        test_tokenizer.cpp
        test_embedding.cpp
        test_attention.cpp
        ../src/attention_cpu.cpp
        ../src/embedding.cpp
        ../src/tokenizer.cpp
    )
endif()

add_executable(test_runner ${TEST_SOURCES})
target_link_libraries(test_runner PRIVATE Catch2::Catch2WithMain)
target_include_directories(test_runner PRIVATE ../src)

if(USE_CUDA)
    set_property(TARGET test_runner PROPERTY CUDA_ARCHITECTURES 89)
endif()

include(CTest)
include(Catch)
catch_discover_tests(test_runner)
