include_directories(..)
include_directories(../../include)
include_directories(../../MATH/include)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)



set(HEADERS
        ${PROJECT_SOURCE_DIR}/include/define.h
        ${PROJECT_SOURCE_DIR}/include/cuda_include/process_cuda.h
        ${PROJECT_SOURCE_DIR}/include/cuda_include/image_process.cuh
        ${PROJECT_SOURCE_DIR}/include/cuda_include/sharedmem.cuh
        ${PROJECT_SOURCE_DIR}/include/cuda_include/image.cuh
        ${PROJECT_SOURCE_DIR}/include/cuda_include/common.cuh
        )

set(SOURCE_FILES
        ${PROJECT_SOURCE_DIR}/src/cuda_src/image_process.cu
        ${PROJECT_SOURCE_DIR}/src/cuda_src/image.cu
        )
cuda_add_library(image_cuda SHARED ${HEADERS} ${SOURCE_FILES}  )
target_link_libraries(image_cuda ${PNG_LIBRARIES} ${JPEG_LIBRARIES})