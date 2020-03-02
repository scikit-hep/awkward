# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

cmake_minimum_required(VERSION 3.1...3.16)
project(through-python LANGUAGES CXX)

include_directories(${CMAKE_PREFIX_PATH}/include)

add_subdirectory(../pybind11 cmake_pybind11)

find_library(CPU-KERNELS awkward-cpu-kernels HINTS ${CMAKE_PREFIX_PATH}/lib)
find_library(LIBAWKWARD awkward HINTS ${CMAKE_PREFIX_PATH}/lib)
message(STATUS "Libraries: ${LIBAWKWARD} ${CPU-KERNELS}")

pybind11_add_module(through-python through-python.cpp)
set_target_properties(through-python PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_link_libraries(through-python PRIVATE ${CPU-KERNELS} ${LIBAWKWARD})
