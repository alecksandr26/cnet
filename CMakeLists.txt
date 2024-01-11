# To build the build system debug purpuse
# cmake . -B build/

# To build the project for release version
# cmake . -B build/ -DCMAKE_BUILD_TYPE=Release

# To build the project with a local install directory
# cmake . -B build/ -DCMAKE_INSTALL_PREFIX=./install/

# To compile the project:
# `cd build/ && make `

cmake_minimum_required(VERSION 3.5)

project(cnet VERSION 1.0.0
  DESCRIPTION "ComplexNet is an artificial neural network framework designed to handle complex-val parameters."
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)

if(UNIX)
  set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra -pedantic -ggdb")
  set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG -ffast-math -march=native")
else()
  # Flags for Windows with MSVC
  set(CMAKE_CXX_FLAGS_DEBUG "/W3 /Od /Zi /DDEBUG")
  set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
endif()

if(DEFINED CMAKE_BUILD_TYPE AND CMAKE_BUILD_TYPE EQUAL Release)
  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE})
else()
  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG})
endif()

message("-- CXX Compiler Flags: ${CMAKE_CXX_FLAGS}")

include(GNUInstallDirs)

set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests)
set(EXAMPLE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/examples)

add_subdirectory(${SRC_DIR}/)
add_subdirectory(${INCLUDE_DIR}/)

# To build the tests
if(NOT DEFINED CMAKE_BUILD_TYPE OR NOT CMAKE_BUILD_TYPE STREQUAL "Release")
  enable_testing()

  add_subdirectory(${TEST_DIR}/)
  add_subdirectory(${EXAMPLE_DIR}/)
endif()


# To format the code 
file(GLOB SOURCES
  ${SRC_DIR}/*.cpp
)

file(GLOB HEADERS
  ${INCLUDE_DIR}/*.hpp
)

set(CLANG_FORMAT "clang-format")
set(FORMAT "format")
add_custom_target(${FORMAT})

add_custom_command(
  TARGET ${FORMAT}
  COMMAND ${CLANG_FORMAT} -i ${SOURCES} ${HEADERS}
  COMMENT "Running clang-format on source files: ${SOURCES} ${HEADERS}"
)

