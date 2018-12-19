include_directories(../../MATH/include)
include_directories(../include)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/examples_bin)

add_executable(image_main image_main.cpp)
target_link_libraries(image_main image)