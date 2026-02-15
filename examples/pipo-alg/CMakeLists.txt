set(TARGET pipo-alg)
add_executable(${TARGET} pipo_alg.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET}  PRIVATE common ggml)
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/vendor)
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_compile_features(${TARGET} PRIVATE cxx_std_17)
