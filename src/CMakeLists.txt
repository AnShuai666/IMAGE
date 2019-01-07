include_directories(../include)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

find_package(PNG REQUIRED)
if (PNG_FOUND)
    message("PNG FOUND: ${PNG_LIBRARIES}")
    include_directories(${PNG_INCLUDE_DIRS})
    add_definitions(${PNG_DEFINITIONS})
endif ()


set(HEADERS
        ${PROJECT_SOURCE_DIR}/include/define.h
        ${PROJECT_SOURCE_DIR}/include/image.hpp
        ../include/image_io.h
        ../include/exception.h ../include/string.h)

set(SOURCE_FILES
        image_io.cpp
        )
add_library(image SHARED ${HEADERS} ${SOURCE_FILES})