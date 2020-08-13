set(CMAKE_CXX_STANDARD 11)

add_executable(test test.cpp)
target_link_libraries(test ${PROJECT_NAME})
# -ltcmalloc