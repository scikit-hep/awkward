# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

cmake_minimum_required(VERSION 3.1...3.16)
project(dependent LANGUAGES CXX)

execute_process(COMMAND python -m awkward1.config --incdir OUTPUT_VARIABLE AWKWARD_INCLUDE)
execute_process(COMMAND python -m awkward1.config --libdir OUTPUT_VARIABLE AWKWARD_LIBRARIES)

message("AWKWARD_INCLUDE ${AWKWARD_INCLUDE}")
message("AWKWARD_LIBRARIES ${AWKWARD_LIBRARIES}")

include_directories(${AWKWARD_INCLUDE})

add_subdirectory(../pybind11 cmake_pybind11)

find_library(CPU-KERNELS awkward-cpu-kernels PATHS ${AWKWARD_LIBRARIES} REQUIRED)
find_library(LIBAWKWARD awkward PATHS ${AWKWARD_LIBRARIES} REQUIRED)
message(STATUS "Libraries: ${LIBAWKWARD} ${CPU-KERNELS}")

pybind11_add_module(dependent dependent.cpp)
set_target_properties(dependent PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_link_libraries(dependent PRIVATE ${CPU-KERNELS} ${LIBAWKWARD})
