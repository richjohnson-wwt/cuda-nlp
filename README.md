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

    conan install . --output-folder=build/Debug --build=missing --settings=build_type=Debug
    cd build/Debug 
    
    # All commands in build/Debug
    cmake ../.. -G "Unix Makefiles" -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Debug
    cmake --build .

    ./src/cuda-nlp
    ./test/test_nlp
    ctest