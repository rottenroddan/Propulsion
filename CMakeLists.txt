cmake_minimum_required(VERSION 3.16)
project(Propulsion CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(Propulsion main.cu Propulsion.cuh)

add_compile_options(-O3)

set_target_properties(
        Propulsion
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)