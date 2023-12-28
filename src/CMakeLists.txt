project(${CMAKE_PROJECT_NAME})

file(GLOB SOURCES
  ${SRC_DIR}/*.cpp
)

add_library(${CMAKE_PROJECT_NAME} SHARED
  ${SOURCES}  
)

# Try to compile to suppoert AVX512F
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE -mavx -march=native)

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



