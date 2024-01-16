project(${CMAKE_PROJECT_NAME})

set(INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/include)
set(CNET_UTILS_LIB cnet_utils)

add_library(${CMAKE_PROJECT_NAME} SHARED
  mat.cpp variable.cpp activation.cpp loss.cpp backprop.cpp weights.cpp layers.cpp
)

set(CNET_UTILS_LIB cnet_utils)
add_subdirectory(utils)
target_link_libraries(${CMAKE_PROJECT_NAME}
  PRIVATE ${CNET_UTILS_LIB}
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

install(TARGETS ${CMAKE_PROJECT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Installing the header files
install(DIRECTORY ${INCLUDE_DIR}/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/
  FILES_MATCHING PATTERN "*.hpp"
)


