if(USE_CUDA)
    set(SOURCES
        main.cu
        attention_gpu.cu
        tokenizer.cpp
        embedding.cpp
        positional_encoding.cpp
    )
else()
    set(SOURCES
        main.cpp
        attention_cpu.cpp
        tokenizer.cpp
        embedding.cpp
        positional_encoding.cpp
    )
endif()

add_executable(main ${SOURCES})
target_include_directories(main PRIVATE .)

if(USE_CUDA)
    set_property(TARGET main PROPERTY CUDA_ARCHITECTURES 89)
endif()