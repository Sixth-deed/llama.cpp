set(TARGET pipo-alg-bench)
add_executable(${TARGET} pipo_alg_bench.cpp)
install(TARGETS ${TARGET} RUNTIME)

target_link_libraries(${TARGET} PRIVATE llama)
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_compile_features(${TARGET} PRIVATE cxx_std_17)
