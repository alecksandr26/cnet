project(${CMAKE_PROJECT_NAME})

add_executable(bench_my_lib
  bench_my_lib.cpp
)

add_executable(example_layer
  example_layer.cpp
)

add_executable(example_var
  example_var.cpp
)

target_link_libraries(bench_my_lib
  PRIVATE ${CMAKE_PROJECT_NAME}
)

target_link_libraries(example_var
  PRIVATE ${CMAKE_PROJECT_NAME}
)

target_link_libraries(example_layer
  PRIVATE ${CMAKE_PROJECT_NAME}
)

find_package(Python COMPONENTS Interpreter Development)
if(Python_Interpreter_FOUND)
  set(BENCH_PYTHON "bench_python")
  add_custom_target(${BENCH_PYTHON})

  add_custom_command(
    TARGET ${BENCH_PYTHON}
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/bench_python.py
    COMMENT "Running bench_python.py with ${Python_EXECUTABLE}"
  )

  set(NET_PYTHON "python_xor_net")
  add_custom_target(${NET_PYTHON})

  add_custom_command(
    TARGET ${NET_PYTHON}
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/python_xor_net.py
    COMMENT "Running python_xor_net.py with ${Python_EXECUTABLE}"
  )

  set(KERAS_NET_PYTHON "keras_xor_example")
  add_custom_target(${KERAS_NET_PYTHON})

  add_custom_command(
    TARGET ${KERAS_NET_PYTHON}
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/keras_xor_example.py
    COMMENT "Running keras_xor_example.py with ${Python_EXECUTABLE}"
  )
endif()

find_package(GSL)
if(GSL_FOUND)
  
  add_executable(bench_c
    bench_c.c
  )
  
  target_include_directories(bench_c PRIVATE ${GSL_INCLUDE_DIRS})
  target_link_libraries(bench_c PRIVATE ${GSL_LIBRARIES})
endif()

add_executable(c_xor_net
  c_xor_net.c
)

target_link_libraries(c_xor_net PRIVATE m)



