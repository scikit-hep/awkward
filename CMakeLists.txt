cmake_minimum_required(VERSION 2.8.12)
project (awkward)

file(READ "VERSION_INFO" VERSION_INFO)
string(STRIP ${VERSION_INFO} VERSION_INFO)
add_definitions(-DVERSION_INFO="${VERSION_INFO}")

file(GLOB CPU_KERNEL_SOURCES "src/cpu-kernels/*.cpp")
file(GLOB LIBAWKWARD_SOURCES "src/libawkward/*.cpp")
include_directories(include)

if (MSVC)
  set(PYBIND11_CPP_STANDARD /std:c++14)
else()
  set(PYBIND11_CPP_STANDARD -std=c++14)
endif()
add_subdirectory(pybind11)

enable_testing()

macro(addtest name filename)
  add_executable(${name} ${filename})
  target_link_libraries(${name} PRIVATE awkward awkward-cpu-kernels)
  add_test(${name} ${name})
endmacro(addtest)

add_library(awkward-cpu-kernels STATIC ${CPU_KERNEL_SOURCES})
target_compile_features(awkward-cpu-kernels PRIVATE cxx_std_14)
addtest(test-dummy1 "tests/cpu-kernels/dummy1.cpp")

add_library(awkward STATIC ${LIBAWKWARD_SOURCES})
target_compile_features(awkward PRIVATE cxx_std_14)
target_link_libraries(awkward PRIVATE awkward-cpu-kernels)
addtest(test-dummy2 "tests/libawkward/dummy2.cpp")

pybind11_add_module(layout src/pyawkward.cpp)
target_compile_features(layout PRIVATE cxx_std_14)
target_link_libraries(layout PRIVATE awkward)
