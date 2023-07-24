include(CTest)
enable_testing()

add_executable(test_mat ${CMAKE_CURRENT_SOURCE_DIR}/test/test_mat.cpp)
add_executable(test_afunc ${CMAKE_CURRENT_SOURCE_DIR}/test/test_afunc.cpp)
add_executable(test_layer ${CMAKE_CURRENT_SOURCE_DIR}/test/test_layer.cpp)
add_executable(test_ann ${CMAKE_CURRENT_SOURCE_DIR}/test/test_ann.cpp)

target_link_libraries(test_mat cnet)
target_link_libraries(test_afunc cnet)
target_link_libraries(test_layer cnet)
target_link_libraries(test_ann cnet)

add_test(NAME test_mat
  COMMAND valgrind
  --error-exitcode=1
  --leak-check=full
  --track-origins=yes
  -s
  --show-leak-kinds=all
  --max-stackframe=62179200 ./test_mat)

add_test(NAME test_afunc
  COMMAND valgrind
  --error-exitcode=1
  --leak-check=full
  --track-origins=yes
  -s
  --show-leak-kinds=all
  --max-stackframe=62179200 ./test_afunc)

add_test(NAME test_layer
  COMMAND valgrind
  --error-exitcode=1
  --leak-check=full
  --track-origins=yes
  -s
  --show-leak-kinds=all
  --max-stackframe=62179200 ./test_layer)

add_test(NAME test_ann
  COMMAND valgrind
  --error-exitcode=1
  --leak-check=full
  --track-origins=yes
  -s
  --show-leak-kinds=all
  --max-stackframe=62179200 ./test_ann)




