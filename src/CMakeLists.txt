include_directories(../../include)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

set(HEADERS
        ${PROJECT_SOURCE_DIR}/include/define.h
        ${PROJECT_SOURCE_DIR}/include/image.hpp
        )
set(SOURCE_FILES
        a.cpp
        )
add_library(image SHARED ${HEADERS} ${SOURCE_FILES})