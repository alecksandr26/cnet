project(${CMAKE_PROJECT_NAME})

# include FetchContent Module and bring Google test
include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

# TO AVOID INSTALLING GTEST
# https://github.com/google/googletest/issues/3486
option(INSTALL_GTEST "Enable installation of googletest." OFF)


# Make it available
FetchContent_MakeAvailable(googletest)
add_library(GTest::GTest INTERFACE IMPORTED)


# Add the exectuables
add_executable(test_mat
  test_mat.cpp
)

# add_executable(test_afunc
#   test_afunc.cpp
# )

# add_executable(test_layer
#   test_layer.cpp
# )

# add_executable(test_model
#   test_model.cpp
# )

# Link the library
target_link_libraries(test_mat
  PRIVATE
  ${CMAKE_PROJECT_NAME}
  GTest::gtest_main
)

# target_link_libraries(test_afunc
#   PRIVATE
#   ${CMAKE_PROJECT_NAME}
#   GTest::gtest_main
# )

# target_link_libraries(test_layer
#   PRIVATE
#   ${CMAKE_PROJECT_NAME}
#   GTest::gtest_main
# )

# target_link_libraries(test_model
#   PRIVATE
#   ${CMAKE_PROJECT_NAME}
#   GTest::gtest_main
# )

include(GoogleTest)
gtest_discover_tests(test_mat)

# gtest_discover_tests(test_afunc)
# gtest_discover_tests(test_layer)
# gtest_discover_tests(test_model)

# include(CTest)

# add_executable(test_mat ${CMAKE_CURRENT_SOURCE_DIR}/test/test_mat.cpp)
# add_executable(test_afunc ${CMAKE_CURRENT_SOURCE_DIR}/test/test_afunc.cpp)
# add_executable(test_layer ${CMAKE_CURRENT_SOURCE_DIR}/test/test_layer.cpp)
# add_executable(test_ann ${CMAKE_CURRENT_SOURCE_DIR}/test/test_ann.cpp)

# target_link_libraries(test_mat cnet)
# target_link_libraries(test_afunc cnet)
# target_link_libraries(test_layer cnet)
# target_link_libraries(test_ann cnet)

# add_test(NAME test_mat
#   COMMAND valgrind
#   --error-exitcode=1
#   --leak-check=full
#   --track-origins=yes
#   -s
#   --show-leak-kinds=all
#   --max-stackframe=62179200 ./test_mat)

# add_test(NAME test_afunc
#   COMMAND valgrind
#   --error-exitcode=1
#   --leak-check=full
#   --track-origins=yes
#   -s
#   --show-leak-kinds=all
#   --max-stackframe=62179200 ./test_afunc)

# add_test(NAME test_layer
#   COMMAND valgrind
#   --error-exitcode=1
#   --leak-check=full
#   --track-origins=yes
#   -s
#   --show-leak-kinds=all
#   --max-stackframe=62179200 ./test_layer)

# add_test(NAME test_ann
#   COMMAND valgrind
#   --error-exitcode=1
#   --leak-check=full
#   --track-origins=yes
#   -s
#   --show-leak-kinds=all
#   --max-stackframe=62179200 ./test_ann)




