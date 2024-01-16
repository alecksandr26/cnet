project(${CMAKE_PROJECT_NAME})

add_library(${CMAKE_PROJECT_NAME} SHARED
  mat.cpp variable.cpp activation.cpp loss.cpp backprop.cpp weights.cpp layers.cpp
)

target_link_libraries(${CMAKE_PROJECT_NAME}
  PRIVATE cnet_utils
)

# Compile to support AVX
target_compile_options(${CMAKE_PROJECT_NAME}
  PRIVATE -mavx -march=native
)

find_package(OpenMP REQUIRED)
target_link_libraries(${CMAKE_PROJECT_NAME}
  PRIVATE OpenMP::OpenMP_CXX
)

target_include_directories(${CMAKE_PROJECT_NAME}
  PUBLIC
  $<BUILD_INTERFACE:${INCLUDE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)


