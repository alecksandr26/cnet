# cmake -DCMAKE_BUILD_TYPE=DEBUG
cmake_minimum_required(VERSION 3.26)

project(cnet VERSION 0.0
  DESCRIPTION "ComplexNet is an artificial neural network framework designed to handle complex-valued parameters."
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(SRC src)

if (UNIX)
  set(CMAKE_CXX_FLAGS_DEBUG "-Wall -pedantic -ggdb")
elseif (WIN32)
  set(CMAKE_CXX_FLAGS_DEBUG "/GS /Wall /pedantic")
endif()

message("-- CXX Compiler Flags: ${CMAKE_CXX_FLAGS_DEBUG}")

include_directories(include)

set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Catch all the cpp files
file(GLOB SOURCES
  ${SRC}/*.cpp)



add_library(cnet STATIC
>>>>>>> feature-ann
  ${SOURCES}
)

set_target_properties(cnet
  PROPERTIES VERSION ${PROJECT_VERSION}
)

include(${CMAKE_CURRENT_SOURCE_DIR}/test/CMakeLists.txt)

