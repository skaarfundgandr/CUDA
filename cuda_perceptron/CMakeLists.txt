cmake_minimum_required(VERSION 4.2)
project(cuda_perceptron CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(cuda_perceptron main.cu)

set_target_properties(cuda_perceptron PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
