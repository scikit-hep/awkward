# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

cmake_minimum_required(VERSION 2.8.12.2)
project (awkward)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if (MSVC)
  set(PYBIND11_CPP_STANDARD /std:c++14)
else()
  set(PYBIND11_CPP_STANDARD -std=c++11)
endif()

file(READ "VERSION_INFO" VERSION_INFO)
string(STRIP ${VERSION_INFO} VERSION_INFO)
add_definitions(-DVERSION_INFO="${VERSION_INFO}")

set(CMAKE_MACOSX_RPATH 1)

file(GLOB CPU_KERNEL_SOURCES "src/cpu-kernels/*.cpp")
file(GLOB LIBAWKWARD_SOURCES "src/libawkward/*.cpp" "src/libawkward/array/*.cpp" "src/libawkward/fillable/*.cpp" "src/libawkward/type/*.cpp" "src/libawkward/io/*.cpp")
include_directories(include)
include_directories(rapidjson/include)

add_subdirectory(pybind11)

enable_testing()

macro(addtest name filename)
  add_executable(${name} ${filename})
  target_link_libraries(${name} PRIVATE awkward-static awkward-cpu-kernels-static)
  add_test(${name} ${name})
endmacro(addtest)

add_library(awkward-cpu-kernels-objects OBJECT ${CPU_KERNEL_SOURCES})
set_target_properties(awkward-cpu-kernels-objects PROPERTIES POSITION_INDEPENDENT_CODE 1)
add_library(awkward-cpu-kernels-static STATIC $<TARGET_OBJECTS:awkward-cpu-kernels-objects>)
add_library(awkward-cpu-kernels        SHARED $<TARGET_OBJECTS:awkward-cpu-kernels-objects>)

add_library(awkward-objects OBJECT ${LIBAWKWARD_SOURCES})
set_target_properties(awkward-objects PROPERTIES POSITION_INDEPENDENT_CODE 1)
add_library(awkward-static STATIC $<TARGET_OBJECTS:awkward-objects>)
add_library(awkward        SHARED $<TARGET_OBJECTS:awkward-objects>)
target_link_libraries(awkward-static PRIVATE awkward-cpu-kernels-static)
target_link_libraries(awkward        PRIVATE awkward-cpu-kernels-static)
addtest(PR016 tests/test_PR016_finish_getitem_for_rawarray.cpp)

pybind11_add_module(layout src/pyawkward.cpp)
set_target_properties(layout PROPERTIES CXX_VISIBILITY_PRESET default)
target_link_libraries(layout PRIVATE awkward-static)
