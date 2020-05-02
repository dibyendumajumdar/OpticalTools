cmake_minimum_required(VERSION 3.16)
project(OpticalTools)

set(CMAKE_CXX_STANDARD 14)

add_executable(OptBenchParser OptBenchParser.cpp)
add_executable(RayTracer RayTracer.cpp)