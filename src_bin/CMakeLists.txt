include_directories(${MATH_INCLUDE_DIRS})
include_directories(../include)
link_directories(${MATH_LIBRARY_DIRS})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/examples_bin)

add_executable(cuda_main image_main_cuda.cpp)
target_link_libraries(cuda_main image image_cuda)