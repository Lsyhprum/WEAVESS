cmake_minimum_required(VERSION 2.8)
# set cmake C and CXX compiler, only required for local Mac environment
#set(CMAKE_C_COMPILER "/usr/local/bin/gcc-7")
#set(CMAKE_CXX_COMPILER "/usr/local/bin/g++-7")
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
# windows
set(BOOST_ROOT G:/MinGW-w64/boost_1_73_0)
# macOS
#set(BOOST_ROOT /usr/local/Cellar/boost/1.74.0/)
find_package(BOOST)
if(Boost_FOUND)
    #message(Boost_INCLUDE_DIRS "${Boost_INCLUDE_DIRS}")
    #message(Boost_LIBRARY_DIRS "${Boost_LIBRARY_DIRS}")
    include_directories(${Boost_INCLUDE_DIRS})
    link_directories(${Boost_LIBRARY_DIRS})
endif()

#Python
#include_directories(F:/Python/include)
#link_libraries(F:/Python/libs/python38.lib)

#add_definitions (-std=c++11 -O3 -lboost -march=native -Wall -DINFO)

add_subdirectory(src)
add_subdirectory(test)