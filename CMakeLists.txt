
cmake_minimum_required(VERSION 2.8 )
project(IMAGE)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-fPIC")
set(CMAKE_BUILD_TYPE "Debug")

set(IMAGE_VERSION_MAJOR 1)
set(IMAGE_VERSION_MINOR 5)
set(IMAGE_VERSION_PATCH 1)
set(IMAGE_VERSION
        ${IMAGE_VERSION_MAJOR}.${IMAGE_VERSION_MINOR}.${IMAGE_VERSION_PATCH})

message("当前IMAGE版本为:${IMAGE_VERSION}")
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmakemodule)

configure_file(
        "${PROJECT_SOURCE_DIR}/config/CmakeHelper.cmake.in"
        "${PROJECT_SOURCE_DIR}/cmakemodule/CmakeHelper.cmake" @ONLY
)
include(cmakemodule/CmakeHelper.cmake)
#if (WIN32 AND NOT CYGWN)
#    set()
#else()
#    set()
#endif ()

#option(USE_CUDA "手否使用cuda" ON)
#list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmakemodule)


#以同步方式，在同一时间内执行同一条指令
option(SIMD_ENABLED "是否使用SIMD指令集进行优化" ON)
#用于共享内存并行系统的多处理器程序设计的一套指导性编译处理方案
option(OPENMP_ENABLED "是否使用OPenMP 进行并行优化" ON)

option(CUDA_ENABLED "是否使用CUDA加速" ON)

option(TESTS_ENABLED "是否编译测试二进制文件" OFF)

####################################################
######################TEST##########################
####################################################
set(TESTS_ENABLED ON)
if (TESTS_ENABLED)
    message(STATUS "允许编译测试文件")
    enable_testing()
else()
    message(STATUS "禁止编译测试文件")
endif ()

####################################################
######################CUDA##########################
####################################################
set(CUDA_MIN_VERSION "9.0")
if (CUDA_ENABLED)
    find_package(CUDA ${CUDA_MIN_VERSION} QUIET)
    if (CUDA_FOUND)
        message("CUDA已经安装")
    else(CUDA_FOUND)
        message(STATUS "未找到 CUDA")
        message(STATUS "  请安装不低于${CUDA_MIN_VERSION}的CUDA版本")
    endif (CUDA_FOUND)
endif ()

if (CUDA_FOUND)
    if (CUDA_ENABLED)
        add_definitions("-DCUDA_ENABLED")
        message(STATUS "使能CUDA支持（版本：，"
                "架构：）")
    endif ()
else()
    set(CUDA_FOUND OFF)
    message(STATUS "  禁用CUDA支持")
endif ()

####################################################
#####################BOOST##########################
####################################################

find_package(Boost REQUIRED COMPONENTS
        program_options
        filesystem
        graph
        regex
        system
        unit_test_framework
        )
if (Boost_FOUND)
    message(STATUS "  找到Boost")
else()
    message(STATUS "  未找到Boost")
    message(STATUS "  请安装Boost")
endif ()

####################################################
#######################SIMD#########################
####################################################

if (SIMD_ENABLED)
    message(STATUS "使能SIMD支持")
else()
    message(STATUS "禁用SIMD支持")
endif ()

####################################################
######################OpenMP########################
####################################################

if (OPENMP_ENABLED)
    message(STATUS "使能OPENMP支持")
    add_definitions("-DOPENMP_ENABLED")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(STATUS "禁用OPENMP支持")
endif ()

####################################################
######################MATH##########################
####################################################

#find_package(MATH REQUIRED)
#if (MATH_FOUND)
#    message(STATUS "  找到MATH，版本为${MATH_VERSION}")
#    message(STATUS "  头文件：${MATH_INCLUDE_DIR}")
#    message(STATUS "  库目录：${MATH_LIBRARY_DIR}")
#    message(STATUS "  库文件：MATH_LIBS ${MATH_LIBS}")
#    message(STATUS "  库文件：MATH_LIBRARY ${MATH_LIBRARY}")
#    message(STATUS "  库文件：MATH_LIBRARIES ${MATH_LIBRARIES}")
#else()
#    message(STATUS "  未找到MATH")
#    message(STATUS "  请安装MATH")
#endif ()
configure_file(
        "${PROJECT_SOURCE_DIR}/config/IMAGEConfig.h.in"
        "${PROJECT_SOURCE_DIR}/include/IMAGEConfig.h"
)

configure_file(
        "${PROJECT_SOURCE_DIR}/config/IMAGEConfig.cmake.in"
        "${PROJECT_SOURCE_DIR}/cmakemodule/IMAGEConfig.cmake" @ONLY
)
set_property(TARGET ${IMAGEConfig.cmake} PROPERTY FOLDER cmakemodule)

configure_file(
        "${PROJECT_SOURCE_DIR}/config/IMAGEConfigVersion.cmake.in"
        "${PROJECT_SOURCE_DIR}/cmakemodule/IMAGEConfigVersion.cmake" @ONLY
)

set(MATH_DIR /home/as/MATH/cmakemodule)
#find_package(MATH REQUIRED)
#message(${MATH_FOUND})
#message(${MATH_INCLUDE_DIR})
#message(${MATH_LIBRARIES})

install(FILES IMAGEConfig.cmake
        DESTINATION share/IMAGE)
install(FILES IMAGEConfigVersion.cmake
        DESTINATION share/IMAGE)

add_subdirectory(examples)
add_subdirectory(src)
add_subdirectory(TestCase)

install(FILES cmakemodule/IMAGEConfig.cmake
        DESTINATION share/IMAGE)
install(FILES cmakemodule/IMAGEConfigVersion.cmake
        DESTINATION share/IMAGE)
install(DIRECTORY include/IMAGE/
        DESTINATION include/IMAGE)

find_library (
        MATHLIB
        NAMES math function util # what to look for
        HINTS "/usr/local/lib/MATH/" # where to look
        NO_DEFAULT_PATH # do not search system default paths
)

# check if we found the library
message(STATUS "MATHLIB: [${MATHLIB}]")
#
#if (NOT MATHLIB)
#    message(SEND_ERROR "Did not find lib MATH")
#endif()
