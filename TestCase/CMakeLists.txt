include_directories(../include)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
link_directories(/home/doing/gitHub/MATH/lib)

IMAGE_ADD_TEST(IMAGE_IO_TEST image_io_test.cpp)
IMAGE_ADD_TEST(IMAGE_CLASS_TEST image_class_test.cpp)
IMAGE_ADD_TEST(IMAGE_PROCESS_TEST image_process_test.cpp)
#target_link_libraries(IMAGE_CLASS_TEST image)