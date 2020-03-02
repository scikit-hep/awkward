# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

cmake_minimum_required(VERSION 3.1...3.16)
project(dependent-project LANGUAGES CXX)

include_directories(${CMAKE_PREFIX_PATH}/include)

add_subdirectory(../pybind11 cmake_pybind11)

execute_process(COMMAND ${PYTHON_EXECUTABLE}
    -c "import pkg_resources; print(pkg_resources.resource_filename('awkward1',''))"
    OUTPUT_VARIABLE SUGGESTED_LOCATION
    ERROR_QUIET
    )
find_library(CPU-KERNELS awkward-cpu-kernels-static HINTS "${SUGGESTED_LOCATION}")
find_library(LIBAWKWARD awkward-static HINTS "${SUGGESTED_LOCATION}")
message(STATUS "Libraries: ${LIBAWKWARD} ${CPU-KERNELS}")

pybind11_add_module(dependent dependent.cpp)
set_target_properties(dependent PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_link_libraries(dependent PRIVATE ${CPU-KERNELS} ${LIBAWKWARD})
