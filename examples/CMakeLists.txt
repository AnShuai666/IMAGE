include_directories(../../MATH/include)
include_directories(../include)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/examples_bin)

add_executable(feature_detection feature_detection.cpp)
target_link_libraries(feature_detection image)