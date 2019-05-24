include_directories(${MATH_INCLUDE_DIRS})
include_directories(../include)
include_directories(../src)
link_directories(${MATH_LIBRARY_DIRS})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/examples_bin)

add_executable(image_main image_main.cpp)
target_link_libraries(image_main image)