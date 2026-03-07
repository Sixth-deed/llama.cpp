find_package(CUDAToolkit)
enable_language(CUDA)
add_library(tensor_init_cuda STATIC
    init_tensor_cuda.cu
)

target_include_directories(tensor_init_cuda PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(tensor_init_cuda PUBLIC
    ggml::ggml
    CUDA::cudart   
    CUDA::curand
)

set_target_properties(tensor_init_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES "86"
    POSITION_INDEPENDENT_CODE ON
)