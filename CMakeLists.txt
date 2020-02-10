# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

cmake_minimum_required(VERSION 2.8.12.2)
message("-- CMake version " ${CMAKE_VERSION})
project (awkward)
cmake_policy(SET CMP0063 NEW)

# This is a C++11 project, but Windows starts at C++14.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if (MSVC)
  set(PYBIND11_CPP_STANDARD /std:c++14)
else()
  set(PYBIND11_CPP_STANDARD -std=c++11)
endif()

# VERSION_INFO is the version stamp for everything (all C++, all Python).
file(READ "VERSION_INFO" VERSION_INFO)
string(STRIP ${VERSION_INFO} VERSION_INFO)
add_definitions(-DVERSION_INFO="${VERSION_INFO}")

set(CMAKE_MACOSX_RPATH 1)

# Three tiers: cpu-kernels (extern "C" interface), libawkward (C++), and Python modules.
file(GLOB CPU_KERNEL_SOURCES "src/cpu-kernels/*.cpp")
file(GLOB LIBAWKWARD_SOURCES "src/libawkward/*.cpp" "src/libawkward/array/*.cpp" "src/libawkward/fillable/*.cpp" "src/libawkward/type/*.cpp" "src/libawkward/io/*.cpp")
file(GLOB PYAWKWARD_SOURCES "src/python/*.cpp")
include_directories(include)

# C++ dependencies (header-only): RapidJSON and pybind11.
include_directories(rapidjson/include)
add_subdirectory(pybind11)

# Macro to add C++ tests (part of CMake build, distinct from pytests in Python).
enable_testing()
macro(addtest name filename)
  add_executable(${name} ${filename})
  target_link_libraries(${name} PRIVATE awkward-static awkward-cpu-kernels-static)
  set_target_properties(${name} PROPERTIES CXX_VISIBILITY_PRESET hidden)
  add_test(${name} ${name})
endmacro(addtest)

# First tier: cpu-kernels (object files, static library, and dynamic library).
add_library(awkward-cpu-kernels-objects OBJECT ${CPU_KERNEL_SOURCES})
set_target_properties(awkward-cpu-kernels-objects PROPERTIES POSITION_INDEPENDENT_CODE 1)
add_library(awkward-cpu-kernels-static STATIC $<TARGET_OBJECTS:awkward-cpu-kernels-objects>)
set_property(TARGET awkward-cpu-kernels-static PROPERTY POSITION_INDEPENDENT_CODE ON)
add_library(awkward-cpu-kernels        SHARED $<TARGET_OBJECTS:awkward-cpu-kernels-objects>)
set_target_properties(awkward-cpu-kernels-objects PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(awkward-cpu-kernels-static PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(awkward-cpu-kernels PROPERTIES CXX_VISIBILITY_PRESET hidden)

# Second tier: libawkward (object files, static library, and dynamic library).
add_library(awkward-objects OBJECT ${LIBAWKWARD_SOURCES})
set_target_properties(awkward-objects PROPERTIES POSITION_INDEPENDENT_CODE 1)
add_library(awkward-static STATIC $<TARGET_OBJECTS:awkward-objects>)
set_property(TARGET awkward-static PROPERTY POSITION_INDEPENDENT_CODE ON)
add_library(awkward        SHARED $<TARGET_OBJECTS:awkward-objects>)
target_link_libraries(awkward-static PRIVATE awkward-cpu-kernels-static)
target_link_libraries(awkward        PRIVATE awkward-cpu-kernels-static)
set_target_properties(awkward-objects PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(awkward-static PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(awkward PROPERTIES CXX_VISIBILITY_PRESET hidden)

# Tests for second tier (these do not depend on Python).
addtest(PR016 tests/test_PR016_finish_getitem_for_rawarray.cpp)
addtest(PR019 tests/test_PR019_use_json_library.cpp)
addtest(PR030 tests/test_PR030_recordarray_in_numba.cpp)

# Third tier: Python modules.
pybind11_add_module(layout ${PYAWKWARD_SOURCES})
set_target_properties(layout PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_link_libraries(layout PRIVATE awkward-static)

# Everything can be built with
#
#     mkdir build
#     cd build
#     cmake ..
#     make all
#     make CTEST_OUTPUT_ON_FAILURE=1 test    # optional: run C++ tests
#     cd ..
#
# or with
#
#     python setup.py build
#
# which additionally moves compiled libraries into the awkward1/ Python source tree.
