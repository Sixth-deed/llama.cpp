set(TARGET llama-simple)
add_executable(${TARGET} simple.cpp)
install(TARGETS ${TARGET} RUNTIME)

target_link_libraries(${TARGET} PRIVATE llama)

target_compile_features(${TARGET} PRIVATE cxx_std_17)
