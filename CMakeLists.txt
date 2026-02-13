cmake_minimum_required(VERSION 3.20)
project(C++_NN_FRAMEWORK VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(OpenMP REQUIRED)


set(SOURCES
        src/main.cpp
        src/Matrix.cpp
        src/DenseLayer.cpp
        src/ActivationLayer.cpp
        src/NeuralNetwork.cpp
        src/SoftMaxLayer.cpp
        src/MSE.cpp
        src/IDXReader.cpp
        src/CrossEntropy.cpp
)

add_executable(C++_NN_FRAMEWORK ${SOURCES})


target_include_directories(C++_NN_FRAMEWORK PRIVATE include)


if(MSVC)
    target_compile_options(C++_NN_FRAMEWORK PRIVATE /O2 /openmp)
else()

    target_compile_options(C++_NN_FRAMEWORK PRIVATE -O3 -march=native -fopenmp)


    target_link_options(C++_NN_FRAMEWORK PRIVATE -fopenmp)
endif()


if(OpenMP_CXX_FOUND)
    target_link_libraries(C++_NN_FRAMEWORK PUBLIC OpenMP::OpenMP_CXX)
endif()