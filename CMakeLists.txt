cmake_minimum_required(VERSION 3.6)
project(cuda-nlp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

set(CMAKE_CUDA_ARCHITECTURES 89)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(src)


option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()