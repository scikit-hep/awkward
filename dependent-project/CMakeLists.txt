# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

cmake_minimum_required(VERSION 2.8.12.2)
project (dependent-project)

include_directories(include)

add_subdirectory(pybind11)

pybind11_add_module(dependent dependent.cpp)
set_target_properties(dependent PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_link_libraries(dependent "awkward/libawkward-static.a")
