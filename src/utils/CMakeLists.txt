project(${CMAKE_PROJECT_NAME})

set(CNET_UTILS_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/include)
set(CNET_UTILS_LIB cnet_utils)

add_library(${CNET_UTILS_LIB} STATIC
  raw_mat.cpp strassen.cpp 
)

# Compile to support AVX
target_compile_options(${CNET_UTILS_LIB}
  PRIVATE -mavx -march=native
)

find_package(OpenMP REQUIRED)
target_link_libraries(${CNET_UTILS_LIB}
  PRIVATE OpenMP::OpenMP_CXX
)

target_include_directories(${CNET_UTILS_LIB}
  PUBLIC ${CNET_UTILS_INCLUDE_DIR}
)

