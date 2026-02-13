set(TARGET llama-simple)
add_executable(${TARGET} simple.cpp)
install(TARGETS ${TARGET} RUNTIME)

set(CUDA_PATH "/usr/local/cuda-13.1")
target_include_directories(${TARGET} PRIVATE ${CUDA_PATH}/include)
target_link_libraries(${TARGET} PRIVATE llama dl)

target_compile_features(${TARGET} PRIVATE cxx_std_17)
