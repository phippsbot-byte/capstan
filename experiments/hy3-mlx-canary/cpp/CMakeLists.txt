cmake_minimum_required(VERSION 3.20)
project(hy3_sidecar_io LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(hy3_sidecar_io hy3_sidecar_io.cpp)
target_compile_options(hy3_sidecar_io PRIVATE -Wall -Wextra -Wpedantic -O3)
