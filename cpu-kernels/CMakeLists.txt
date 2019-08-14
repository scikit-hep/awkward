cmake_minimum_required(VERSION 2.8.9)
project (libawkward)

include_directories(include)

file(GLOB SOURCES "src/*.cpp")

add_library(awkward-cpu-kernels SHARED ${SOURCES})
