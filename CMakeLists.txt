cmake_minimum_required(VERSION 2.8)
project(WEAVESS)

set(CMAKE_CXX_STANDARD 11)

include_directories(${PROJECT_SOURCE_DIR}/include)

#OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "no OpenMP supprot")
endif()

#Boost
set(BOOST_ROOT F:/MinGW-W64/boost_1_73_0)
find_package(BOOST)
if(Boost_FOUND)
    message(Boost_INCLUDE_DIRS "${Boost_INCLUDE_DIRS}")
    message(Boost_LIBRARY_DIRS "${Boost_LIBRARY_DIRS}")
    include_directories(${Boost_INCLUDE_DIRS})
    link_directories(${Boost_LIBRARY_DIRS})
endif()

#Python
# 添加python.h头文件路径
include_directories(F:/Python/include)
# 添加python动态链接库文件
link_libraries(F:/Python/libs/python38.lib)

#add_definitions (-std=c++11 -O3 -lboost -march=native -Wall -DINFO)

add_subdirectory(src)
add_subdirectory(test)