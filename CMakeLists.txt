cmake_minimum_required(VERSION 3.5)
project(2048)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Ofast -fopenmp")

add_executable(2048 main.cpp game.cpp agent.cpp)