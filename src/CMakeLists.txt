include_directories(..)
include_directories(../../MATH/include)
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
        ${PROJECT_SOURCE_DIR}/include/string.h
        ${PROJECT_SOURCE_DIR}/include/sift.hpp
        ${PROJECT_SOURCE_DIR}/include/image_process.hpp
        ${PROJECT_SOURCE_DIR}/include/timer.h
        )

set(SOURCE_FILES
        define.cpp
        image_io.cpp
        sift.cpp
        image_process.cpp
        ${PROJECT_SOURCE_DIR}/include/exception.h
        )
add_library(image SHARED ${HEADERS} ${SOURCE_FILES}  )
target_link_libraries(image ${PNG_LIBRARIES} ${JPEG_LIBRARIES})