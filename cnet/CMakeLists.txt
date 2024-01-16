project(${CMAKE_PROJECT_NAME})

set(INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/include)
set(SRC_DIR ${CMAKE_CURRENT_LIST_DIR}/src)
set(UTILS_LIB_DIR ${CMAKE_CURRENT_LIST_DIR}/utils)
set(TEST_DIR ${CMAKE_CURRENT_LIST_DIR}/tests)

# Add this module which contain utitlies for the different process in the framework

add_subdirectory(${UTILS_LIB_DIR})
add_subdirectory(${SRC_DIR})

# To install the framework
install(DIRECTORY ${INCLUDE_DIR}/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/
  FILES_MATCHING PATTERN "*.hpp"
)

install(TARGETS ${CMAKE_PROJECT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)


if(NOT DEFINED CMAKE_BUILD_TYPE OR NOT CMAKE_BUILD_TYPE STREQUAL "Release")
  add_subdirectory(${TEST_DIR}/)
endif()

