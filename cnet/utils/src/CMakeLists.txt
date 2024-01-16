project(${CMAKE_PROJECT_NAME})

add_library(${CNET_UTILS_LIB} STATIC
  raw_mat.cpp strassen.cpp 
)

target_include_directories(${CNET_UTILS_LIB}
  PUBLIC ${CNET_UTILS_INCLUDE_DIR}
)

# Compile to support AVX
target_compile_options(${CNET_UTILS_LIB}
  PRIVATE -mavx -march=native
)


find_package(OpenMP REQUIRED)
target_link_libraries(${CNET_UTILS_LIB}
  PRIVATE OpenMP::OpenMP_CXX
)

