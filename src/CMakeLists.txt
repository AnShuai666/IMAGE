include_directories(../../MATH/include)
include_directories(../include)
include_directories(${MATH_INCLUDE_DIRS})
link_directories(${MATH_LIBRARY_DIRS})
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

find_package(PNG REQUIRED)
if (PNG_FOUND)
    message("PNG 找到: ${PNG_LIBRARIES}")
    include_directories(${PNG_INCLUDE_DIRS})
    add_definitions(${PNG_DEFINITIONS})
endif ()

find_package(JPEG REQUIRED)
if (JPEG_FOUND)
    message(JPEG 找到: ${JPEG_LIBRARIES})
    include_directories(${JPEG_INCLUDE_DIR})
endif ()

set(HEADERS
        ${PROJECT_SOURCE_DIR}/include/IMAGE/define.h
        ${PROJECT_SOURCE_DIR}/include/IMAGE/image.hpp
        ${PROJECT_SOURCE_DIR}/include/IMAGE/image_io.h
        ${PROJECT_SOURCE_DIR}/include/exception.h
        ${PROJECT_SOURCE_DIR}/include/mystring.h
        ${PROJECT_SOURCE_DIR}/include/IMAGE/image_process.hpp
        ${PROJECT_SOURCE_DIR}/include/timer.h
        ${PROJECT_SOURCE_DIR}/include/IMAGE/visualizer.hpp
        ${PROJECT_SOURCE_DIR}/include/IMAGE/nearest_neighbor.hpp
        ${PROJECT_SOURCE_DIR}/include/Features2d/features2d.h
        ${PROJECT_SOURCE_DIR}/include/IMAGE/matching.h
        ${PROJECT_SOURCE_DIR}/include/IMAGE/image_pyramid.h
        )

set(SOURCE_FILES
        Features2d/sift.cpp
        Features2d/ORB.cpp
        Features2d/feature2d.cpp
        define.cpp
        image_io.cpp
        image_process.cpp
        Features2d/Fast.cpp
        matching.cpp
        image_pyramid.cpp
        )

link_directories(/home/doing/gitHub/MATH/lib)

add_library(image SHARED ${HEADERS} ${SOURCE_FILES})

target_link_libraries(image ${PNG_LIBRARIES} ${JPEG_LIBRARIES} )
target_link_libraries(image matrix)

install(TARGETS image
        LIBRARY DESTINATION lib/IMAGE
        ARCHIVE DESTINATION lib/IMAGE
        RUNTIME DESTINATION bin/IMAGE
        )
if (CUDA_FOUND)
    add_subdirectory(cuda_src)
endif ()