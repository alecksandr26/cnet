include(CTest)
enable_testing()

add_executable(test_mat ${CMAKE_CURRENT_SOURCE_DIR}/test/test_mat.cpp)
add_executable(test_afuncs ${CMAKE_CURRENT_SOURCE_DIR}/test/test_afuncs.cpp)
add_executable(test_layer ${CMAKE_CURRENT_SOURCE_DIR}/test/test_layer.cpp)

target_link_libraries(test_mat cnet)
target_link_libraries(test_afuncs cnet)
target_link_libraries(test_layer cnet)

add_test(NAME test_mat
  COMMAND valgrind
  --error-exitcode=1
  --leak-check=full
  --track-origins=yes
  -s
  --show-leak-kinds=all
  --max-stackframe=62179200 ./test_mat)

add_test(NAME test_afuncs
  COMMAND valgrind
  --error-exitcode=1
  --leak-check=full
  --track-origins=yes
  -s
  --show-leak-kinds=all
  --max-stackframe=62179200 ./test_afuncs)


add_test(NAME test_layer
  COMMAND valgrind
  --error-exitcode=1
  --leak-check=full
  --track-origins=yes
  -s
  --show-leak-kinds=all
  --max-stackframe=62179200 ./test_layer)




