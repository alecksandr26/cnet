cmake_minimum_required(VERSION 3.26)

project(CNet VERSION 0.0)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

set(SOURCES
  src/cnet.cpp)

add_library(cnet SHARED
  ${SOURCES}
)

set_target_properties(cnet
  PROPERTIES VERSION ${PROJECT_VERSION}
)

install(TARGETS
  cnet LIBRARY DESTINATION
  lib
)

install(FILES
  include/cnet.h
  DESTINATION
  include
)

