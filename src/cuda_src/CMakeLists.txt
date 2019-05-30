include_directories(..)
include_directories(../../include)
include_directories(../../MATH/include)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)



set(HEADERS
        ../../include/IMAGE/define.h
        ../../include/IMAGE/cuda_include/process_cuda.h

        )

set(SOURCE_FILES
       ${PROJECT_SOURCE_DIR}/src/cuda_src/image_io.cu
        )
cuda_add_library(image_cuda SHARED ${HEADERS} ${SOURCE_FILES}  )
target_link_libraries(image_cuda ${PNG_LIBRARIES} ${JPEG_LIBRARIES})