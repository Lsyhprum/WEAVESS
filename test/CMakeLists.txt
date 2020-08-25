set(CMAKE_CXX_STANDARD 11)

add_executable(index_test IndexTest.cpp)
target_link_libraries(index_test ${PROJECT_NAME})

