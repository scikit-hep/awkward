cmake_minimum_required(VERSION 2.8.12)
project (awkward)

include_directories(include)
add_subdirectory(pybind11)
file(GLOB CPU_KERNEL_SOURCES "src/cpu-kernels/*.cpp")
file(GLOB LIBAWKWARD_SOURCES "src/libawkward/*.cpp")

add_library(awkward-cpu-kernels SHARED ${CPU_KERNEL_SOURCES})

add_library(awkward SHARED ${LIBAWKWARD_SOURCES})
target_link_libraries(awkward PRIVATE awkward-cpu-kernels)

pybind11_add_module(layout src/pyawkward.cpp)
target_link_libraries(layout PRIVATE awkward)
