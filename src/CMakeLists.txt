

add_executable(main main.cu attention.cu tokenizer.cpp embedding.cpp positional_encoding.cpp)

set_property(TARGET main PROPERTY CUDA_ARCHITECTURES OFF)