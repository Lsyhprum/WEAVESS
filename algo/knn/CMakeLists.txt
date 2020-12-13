set(CMAKE_CXX_STANDARD 11)


add_executable(gen_knn gen_knn.cpp)
target_link_libraries(gen_knn ${PROJECT_NAME})