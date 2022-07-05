# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

cmake_minimum_required(VERSION 3.14...3.18)

project(tests-cpp)

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

message(STATUS "CMake version ${CMAKE_VERSION}")
message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")

add_executable(test001 test-growable-buffer.cpp)
add_executable(test002 test-layout-builder.cpp)

enable_testing()
add_test(
  NAME test001
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests-cpp
  COMMAND test001)
add_test(
  NAME test002
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests-cpp
  COMMAND test002)
