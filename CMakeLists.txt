
cmake_minimum_required(VERSION 2.8 )
project(IMAGE)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-fPIC")
#set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_BUILD_TYPE "Release")

set(IMAGE_VERSION_MAJOR 1)
set(IMAGE_VERSION_MINOR 0)
set(IMAGE_VERSION_PATCH 0)
set(IMAGE_VERSION
        ${IMAGE_VERSION_MAJOR}.${IMAGE_VERSION_MINOR}.${IMAGE_VERSION_PATCH})

#if (WIN32 AND NOT CYGWN)
#    set()
#else()
#    set()
#endif ()

#option(USE_CUDA "手否使用cuda" ON)
#list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmakemodule)

find_package(CUDA QUIET)
if(CUDA_FOUND)
    set(CUDA_ENABLED true)
endif ()

configure_file(
        "${PROJECT_SOURCE_DIR}/config/IMAGEConfig.h.in"
        "include/IMAGE/IMAGEConfig.h"
)

configure_file(
        "${PROJECT_SOURCE_DIR}/config/IMAGEConfig.cmake.in"
        "${PROJECT_SOURCE_DIR}/cmakemodule/IMAGEConfig.cmake" @ONLY
)
set_property(TARGET ${IMAGEConfig.cmake} PROPERTY FOLDER cmakemodule)

configure_file(
        "${PROJECT_SOURCE_DIR}/config/IMAGEConfigVersion.cmake.in"
        "${PROJECT_SOURCE_DIR}/cmakemodule/IMAGEConfigVeision.cmake" @ONLY
)

set(MATH_DIR /home/as/MATH/cmakemodule)
find_package(MATH REQUIRED)
message(${MATH_FOUND})
message(${MATH_INCLUDE_DIR})
message(${MATH_LIBRARIES})

install(FILES IMAGEConfig.cmake
        DESTINATION share/IMAGE)
install(FILES IMAGEConfigVersion.cmake
        DESTINATION share/IMAGE)

add_subdirectory(examples)
add_subdirectory(src)
add_subdirectory(src_bin)

install(FILES IMAGEConfig.h
        DESTINATION include/IMAGE)