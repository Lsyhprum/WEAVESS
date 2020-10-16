cmake_minimum_required(VERSION 3.17)
project(WEAVESS_dataset)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-fopenmp")

add_executable(WEAVESS_dataset main.cpp)