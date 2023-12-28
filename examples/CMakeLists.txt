project(${CMAKE_PROJECT_NAME})

add_executable(bench_my_lib
  bench_my_lib.cpp
)

target_link_libraries(bench_my_lib
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
endif()


find_package(GSL)
if(GSL_FOUND)
  
  add_executable(bench_c
    bench_c.c
  )

  target_include_directories(bench_c PRIVATE ${GSL_INCLUDE_DIRS})
  target_link_libraries(bench_c PRIVATE ${GSL_LIBRARIES})
endif()








