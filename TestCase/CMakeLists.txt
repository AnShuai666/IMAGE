include_directories(../include)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

IMAGE_ADD_TEST(IMAGE_IO_TEST image_io_test.cpp)