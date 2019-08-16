cmake_minimum_required(VERSION 2.8.12.2)
project (awkward)

file(READ "VERSION_INFO" VERSION_INFO)
string(STRIP ${VERSION_INFO} VERSION_INFO)
add_definitions(-DVERSION_INFO="${VERSION_INFO}")

set(CMAKE_MACOSX_RPATH 1)

file(GLOB CPU_KERNEL_SOURCES "src/cpu-kernels/*.cpp")
file(GLOB LIBAWKWARD_SOURCES "src/libawkward/*.cpp")
include_directories(include)

if (MSVC)
  set(PYBIND11_CPP_STANDARD /std:c++14)
else()
  set(PYBIND11_CPP_STANDARD -std=c++11)
endif()
add_subdirectory(pybind11)

enable_testing()

macro(addtest name filename)
  add_executable(${name} ${filename})
  target_link_libraries(${name} PRIVATE awkward-static awkward-cpu-kernels-static)
  add_test(${name} ${name})
endmacro(addtest)

add_library(awkward-cpu-kernels-objects OBJECT ${CPU_KERNEL_SOURCES})
set_property(TARGET awkward-cpu-kernels-objects PROPERTY POSITION_INDEPENDENT_CODE 1)
add_library(awkward-cpu-kernels-static STATIC $<TARGET_OBJECTS:awkward-cpu-kernels-objects>)
add_library(awkward-cpu-kernels        SHARED $<TARGET_OBJECTS:awkward-cpu-kernels-objects>)
addtest(test-dummy1 "tests/cpu-kernels/dummy1.cpp")

add_library(awkward-objects OBJECT ${LIBAWKWARD_SOURCES})
set_property(TARGET awkward-objects PROPERTY POSITION_INDEPENDENT_CODE 1)
add_library(awkward-static STATIC $<TARGET_OBJECTS:awkward-objects>)
add_library(awkward        SHARED $<TARGET_OBJECTS:awkward-objects>)
target_link_libraries(awkward-static PRIVATE awkward-cpu-kernels-static)
target_link_libraries(awkward        PRIVATE awkward-cpu-kernels-static)
addtest(test-dummy2 "tests/libawkward/dummy2.cpp")

pybind11_add_module(layout src/pyawkward.cpp)
target_link_libraries(layout PRIVATE awkward-static)
