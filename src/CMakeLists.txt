include_directories(../include)
include_directories(${MATH_INCLUDE_DIRS})
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
        ../include/IMAGE/define.h
        ../include/IMAGE/image.hpp
        ../include/IMAGE/image_io.h
        ../include/IMAGE/sift.hpp
        ../include/IMAGE/image_process.hpp
        ../include/IMAGE/visualizer.hpp
        ../include/IMAGE/nearest_neighbor.hpp
        ../include/IMAGE/matching.h
        ../include/IMAGE/image_pyramid.h

        )

set(SOURCE_FILES
        define.cpp
        image_io.cpp
        sift.cpp
        image_process.cpp
        matching.cpp
        image_pyramid.cpp
        )
add_library(image SHARED ${HEADERS} ${SOURCE_FILES}  )
target_link_libraries(image ${PNG_LIBRARIES} ${JPEG_LIBRARIES} )

if (CUDA_FOUND)
    add_subdirectory(cuda_src)
endif ()