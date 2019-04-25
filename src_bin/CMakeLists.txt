include_directories(../../MATH/include)
include_directories(../include)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/examples_bin)

add_executable(cuda_main image_main_cuda.cpp)
target_link_libraries(cuda_main image image_cuda)