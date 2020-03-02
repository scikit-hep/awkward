# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

cmake_minimum_required(VERSION 3.10...3.16)
message(STATUS "CMake version ${CMAKE_VERSION}")
project(awkward LANGUAGES CXX)

# This is a C++11 project, but Windows starts at C++14.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if (MSVC)
  set(CMAKE_CXX_STANDARD 14)
endif()

# VERSION_INFO is the version stamp for everything (all C++, all Python).
file(READ "VERSION_INFO" VERSION_INFO)
string(STRIP ${VERSION_INFO} VERSION_INFO)
add_definitions(-DVERSION_INFO="${VERSION_INFO}")

# Three tiers: cpu-kernels (extern "C" interface), libawkward (C++), and Python modules.
file(GLOB CPU_KERNEL_SOURCES CONFIGURE_DEPENDS "src/cpu-kernels/*.cpp")
file(GLOB_RECURSE LIBAWKWARD_SOURCES CONFIGURE_DEPENDS "src/libawkward/*.cpp")
include_directories(include)

# C++ dependencies (header-only): RapidJSON and pybind11.
include_directories(rapidjson/include)

# Macro to add C++ tests (part of CMake build, distinct from pytests in Python).
include(CTest)

macro(addtest name filename)
  if (BUILD_TESTING)
    add_executable(${name} ${filename})
    target_link_libraries(${name} PRIVATE awkward-static awkward-cpu-kernels-static)
    set_target_properties(${name} PROPERTIES CXX_VISIBILITY_PRESET hidden)
    add_test(${name} ${name})
  endif()
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
addtest(test0016 tests/test_0016-finish-getitem-for-rawarray.cpp)
addtest(test0019 tests/test_0019-use-json-library.cpp)
addtest(test0030 tests/test_0030-recordarray-in-numba.cpp)

install(TARGETS awkward-static awkward awkward-cpu-kernels awkward-cpu-kernels-static LIBRARY DESTINATION "awkward1")

# Third tier: Python modules.

if (PYBUILD)
  add_subdirectory(pybind11)
  
  file(GLOB LAYOUT_SOURCES "src/python/layout/*.cpp")
  pybind11_add_module(layout src/python/layout.cpp ${LAYOUT_SOURCES})
  set_target_properties(layout PROPERTIES CXX_VISIBILITY_PRESET hidden)
  target_link_libraries(layout PRIVATE awkward-static)

  pybind11_add_module(types src/python/types.cpp)
  set_target_properties(types PROPERTIES CXX_VISIBILITY_PRESET hidden)
  target_link_libraries(types PRIVATE awkward-static)

  pybind11_add_module(_io src/python/_io.cpp)
  set_target_properties(_io PROPERTIES CXX_VISIBILITY_PRESET hidden)
  target_link_libraries(_io PRIVATE awkward-static)

  install(TARGETS awkward-cpu-kernels awkward-cpu-kernels-static awkward awkward-static layout types _io LIBRARY DESTINATION "awkward1")

endif()
