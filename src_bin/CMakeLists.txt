include_directories(${MATH_INCLUDE_DIRS})
include_directories(../include)
link_directories(${MATH_LIBRARY_DIRS})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/examples_bin)

add_executable(cuda_main image_main_cuda.cpp)
target_link_libraries(cuda_main image image_cuda math util )

add_executable(fill_color fill_color.cpp)
target_link_libraries(fill_color image image_cuda)

add_executable(add_channels add_Channels.cpp)
target_link_libraries(add_channels image image_cuda)

add_executable(swap_channels swap_channels.cpp)
target_link_libraries(swap_channels image image_cuda)

add_executable(copy_channels copy_channels.cpp)
target_link_libraries(copy_channels image image_cuda math util)

add_executable(delete_channels delete_channel.cpp)
target_link_libraries(delete_channels image image_cuda math util)

add_executable(byte_to_float_image byte_to_float_image.cpp)
target_link_libraries(byte_to_float_image image image_cuda math util)
