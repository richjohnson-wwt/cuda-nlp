# Intial Setup - Do every time a new VM is started

    uv venv
    source .venv/bin/activate
    uv pip install conan
    conan profile detect
    Install C++ and CMake Extensions in VSCode
    vi ~/.gitconfig

    [user]
        email = rich.johnson@wwt.com
        name = Rich Johnson


# Debug Config

    conan install . --output-folder=build/debug_cpu --build=missing --settings=build_type=Debug
    conan install . --output-folder=build/debug_gpu --build=missing --settings=build_type=Debug
    cd build/debug_cpu 
    cd build/debug_gpu 
    
    # All commands in build/debug_cpu
    cmake ../.. -G "Unix Makefiles" -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Debug -DUSE_CUDA=OFF
    cmake ../.. -G "Unix Makefiles" -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Debug -DUSE_CUDA=ON
    cmake --build .

    Train the model with (input sentence) (target):

    ./src/main "the dog is" "great"
    ./src/main "WWT is a great place to" "work"

    ./test/test_runner
    ctest