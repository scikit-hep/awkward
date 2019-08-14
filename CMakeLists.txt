cmake_minimum_required(VERSION 2.8.9)
project (libawkward)

include_directories(include)

file(GLOB CPU_KERNEL_SOURCES "src/cpu-kernels/*.cpp")
file(GLOB LIBAWKWARD_SOURCES "src/libawkward/*.cpp")

add_library(awkward-cpu-kernels SHARED ${CPU_KERNEL_SOURCES})
add_library(awkward             SHARED ${LIBAWKWARD_SOURCES})
target_link_libraries(awkward awkward-cpu-kernels)
