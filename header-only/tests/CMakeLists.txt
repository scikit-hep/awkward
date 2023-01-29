# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

cmake_minimum_required(VERSION 3.15...3.24)

# Project must be near the top
project(header-only-tests LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# C++ dependencies (header-only): GrowableBuffer
add_library(awkward-parent INTERFACE)
target_include_directories(awkward-parent INTERFACE ../header-only)
target_compile_features(awkward-parent INTERFACE cxx_std_11)

macro(addtest name filename)
  add_executable(${name} ${filename})
  target_link_libraries(${name} PRIVATE awkward-static awkward-cpu-kernels-static)
  set_target_properties(${name} PROPERTIES CXX_VISIBILITY_PRESET hidden)
  set_target_properties(${name} PROPERTIES VISIBILITY_INLINES_HIDDEN ON)
  add_test(${name} ${name})
endmacro(addtest)

macro(addtest_nolibs name filename)
  target_compile_features(awkward-parent INTERFACE cxx_std_14)

  add_executable(${name} ${filename})
  target_link_libraries(${name} PRIVATE awkward-parent)
  set_target_properties(${name} PROPERTIES CXX_VISIBILITY_PRESET hidden)
  set_target_properties(${name} PROPERTIES VISIBILITY_INLINES_HIDDEN ON)
  add_test(${name} ${name})
endmacro(addtest_nolibs)

addtest_nolibs(test_1494-layout-builder test_1494-layout-builder.cpp)
addtest_nolibs(test_1542-growable-buffer test_1542-growable-buffer.cpp)
addtest_nolibs(test_1560-builder-options test_1560-builder-options.cpp)
