cmake_minimum_required(VERSION 4.2)
project(vecAdd CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(vecAdd main.cu)

set_target_properties(vecAdd PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
