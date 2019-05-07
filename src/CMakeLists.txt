include_directories(..)
include_directories(../../MATH/include)
include_directories(../include)
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
        ${PROJECT_SOURCE_DIR}/include/define.h
        ${PROJECT_SOURCE_DIR}/include/image.hpp
        ${PROJECT_SOURCE_DIR}/include/image_io.h
        ${PROJECT_SOURCE_DIR}/include/exception.h
        ${PROJECT_SOURCE_DIR}/include/mystring.h
        #${PROJECT_SOURCE_DIR}/include/sift.hpp
        ${PROJECT_SOURCE_DIR}/include/image_process.hpp
        ${PROJECT_SOURCE_DIR}/include/timer.h
        ${PROJECT_SOURCE_DIR}/include/visualizer.hpp
        ${PROJECT_SOURCE_DIR}/include/nearest_neighbor.hpp
        ${PROJECT_SOURCE_DIR}/include/features2d/Array.h
        ${PROJECT_SOURCE_DIR}/include/features2d/ptr.hpp
        ${PROJECT_SOURCE_DIR}/include/features2d/features2d.hpp
        )

set(SOURCE_FILES
        features2d/Array.cpp
        features2d/sift.cpp
        features2d/feature2d.cpp
        define.cpp
        image_io.cpp
        #sift.cpp
        image_process.cpp
        )

link_directories(/home/doing/gitHub/MATH/lib)

add_library(image SHARED ${HEADERS} ${SOURCE_FILES}  )

target_link_libraries(image ${PNG_LIBRARIES} ${JPEG_LIBRARIES} )
target_link_libraries(image matrix_LU)
if (CUDA_FOUND)
    add_subdirectory(cuda_src)
endif ()