## Overview
This is a learning project that uses softmax, self-attention and matrix multiplication to train a little model to predict the next word in a sentence. It also explores a strategy for unit testing the code, using Conan package manager.

#### Intial Setup - Do every time a new VM is started

    uv venv
    source .venv/bin/activate
    uv pip install conan
    conan profile detect
    Install C++ and CMake Extensions in VSCode
    vi ~/.gitconfig

    [user]
        email = rich.johnson@wwt.com
        name = Rich Johnson


#### Debug Config

    conan install . --output-folder=build/Debug --build=missing --settings=build_type=Debug
    cd build/Debug 
    
    # All commands in build/Debug
    cmake ../.. -G "Unix Makefiles" -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Debug
    cmake --build .

    Train the model with (input sentence) (target):

    ./src/main "the dog is" "great"
    ./src/main "WWT is a great place to" "work"

    ./test/test_runner
    ctest
