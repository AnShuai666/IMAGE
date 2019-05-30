include_directories(..)
include_directories(../../include)

include_directories(${MATH_INCLUDE_DIRS})
include_directories(../include)
link_directories(${MATH_LIBRARY_DIRS})
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)



set(HEADERS
        ../../include/IMAGE/define.h
        ../../include/IMAGE/cuda_include/process_cuda.h
        ../../include/cuda_include/image_1.cuh
        ../../include/cuda_include/image_process_1.cuh
        )

set(SOURCE_FILES
        ${PROJECT_SOURCE_DIR}/src/cuda_src/image_process_1.cu
        ${PROJECT_SOURCE_DIR}/src/cuda_src/image.cu
        ${PROJECT_SOURCE_DIR}/src/cuda_src/image_process_11.cu
        ${PROJECT_SOURCE_DIR}/src/cuda_src/cuda_common.cu
        )
cuda_add_library(image_cuda SHARED ${HEADERS} ${SOURCE_FILES}  )
target_link_libraries(image_cuda ${PNG_LIBRARIES} ${JPEG_LIBRARIES} image math util)