
cmake_minimum_required(VERSION 2.8 )
project(image)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-fPIC")
set(CMAKE_BUILD_TYPE "Debug")

add_subdirectory(examples)
add_subdirectory(src)
add_subdirectory(src_bin)

