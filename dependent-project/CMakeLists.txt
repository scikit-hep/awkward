# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

cmake_minimum_required(VERSION 2.8.12.2)
project (dependent-project)

include_directories(include)

add_subdirectory(../pybind11 cmake_pybind11)

find_library(CPU-KERNELS awkward-cpu-kernels-static PATHS awkward1)
find_library(LIBAWKWARD awkward-static PATHS awkward1)
message(${LIBAWKWARD})

pybind11_add_module(dependent dependent.cpp)
set_target_properties(dependent PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_link_libraries(dependent PRIVATE ${CPU-KERNELS} ${LIBAWKWARD})
