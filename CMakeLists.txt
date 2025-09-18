cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(n_body LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS on)
set(CMAKE_CUDA_ARCHITECTURES "native")

add_subdirectory(src)
