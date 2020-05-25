# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

cmake_minimum_required(VERSION 3.10...3.16)

# Project must be near the top
project(awkward LANGUAGES CXX)

message(STATUS "CMake version ${CMAKE_VERSION}")

set(SIMPLE_TYPES Debug Release)
if(CMAKE_BUILD_TYPE IN_LIST SIMPLE_TYPES)
  message(STATUS "${CMAKE_BUILD_TYPE}")
else()
  message(STATUS "CMAKE_BUILD_TYPE = " ${CMAKE_BUILD_TYPE})
endif()

# This is a C++11 project, but Windows starts at C++14.
if(MSVC)
  set(CMAKE_CXX_STANDARD 14)
else()
  set(CMAKE_CXX_STANDARD 11)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# VERSION_INFO is the version stamp for everything (all C++, all Python).
file(READ "VERSION_INFO" VERSION_INFO)
string(STRIP ${VERSION_INFO} VERSION_INFO)
add_definitions(-DVERSION_INFO="${VERSION_INFO}")

# Let CMake know the version too
project(awkward LANGUAGES CXX VERSION ${VERSION_INFO})

# Finding CUDA
option(BUILD_CUDA_KERNELS "Build GPU Kernels with CUDA" OFF)
if(BUILD_CUDA_KERNELS)
  # Release additional checks once CUDA Kernels if officially released
  # include(CheckLanguage)
  # check_language(CUDA)
  # if(CMAKE_CUDA_COMPILER)
  #   enable_language(CUDA)
  add_definitions( -DBUILD_CUDA_KERNELS )
  # else()
  #   message(FATAL_ERROR "NO CUDA COMPILER FOUND! PLEASE CHECK YOUR CUDA PATH, CMake will now exit")
  # endif()
endif()

# Three tiers: [cpu-kernels (extern "C" interface), cuda-kernels (extern "C" interface)], libawkward (C++), and Python modules.
file(GLOB CPU_KERNEL_SOURCES CONFIGURE_DEPENDS "src/cpu-kernels/*.cpp")
if(BUILD_CUDA_KERNELS)
  file(GLOB CUDA_KERNEL_SOURCES CONFIGURE_DEPENDS "src/cuda-kernels/*.cpp")
endif()
file(GLOB_RECURSE LIBAWKWARD_SOURCES CONFIGURE_DEPENDS "src/libawkward/*.cpp")

include_directories(include)

# C++ dependencies (header-only): RapidJSON and pybind11.
include_directories(rapidjson/include)

# Macro to add C++ tests (part of CMake build, distinct from pytests in Python).
include(CTest)

macro(addtest name filename)
  if(BUILD_TESTING)
    add_executable(${name} ${filename})
    target_link_libraries(${name} PRIVATE awkward-static awkward-cpu-kernels-static)
    set_target_properties(${name} PROPERTIES CXX_VISIBILITY_PRESET hidden)
    set_target_properties(${name} PROPERTIES VISIBILITY_INLINES_HIDDEN ON)
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
set_target_properties(awkward-cpu-kernels-objects PROPERTIES VISIBILITY_INLINES_HIDDEN ON)
set_target_properties(awkward-cpu-kernels-static PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(awkward-cpu-kernels-static PROPERTIES VISIBILITY_INLINES_HIDDEN ON)
set_target_properties(awkward-cpu-kernels PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(awkward-cpu-kernels PROPERTIES VISIBILITY_INLINES_HIDDEN ON)

#First tier: cuda-kernels, conditional compilation
if(BUILD_CUDA_KERNELS)
  add_library(awkward-cuda-kernels-objects OBJECT ${CUDA_KERNEL_SOURCES})
  set_target_properties(awkward-cuda-kernels-objects PROPERTIES POSITION_INDEPENDENT_CODE 1)
  add_library(awkward-cuda-kernels-static STATIC $<TARGET_OBJECTS:awkward-cuda-kernels-objects>)
  set_property(TARGET awkward-cuda-kernels-static PROPERTY POSITION_INDEPENDENT_CODE ON)
  add_library(awkward-cuda-kernels        SHARED $<TARGET_OBJECTS:awkward-cuda-kernels-objects>)
  set_target_properties(awkward-cuda-kernels-objects PROPERTIES CXX_VISIBILITY_PRESET hidden)
  set_target_properties(awkward-cuda-kernels-objects PROPERTIES VISIBILITY_INLINES_HIDDEN ON)
  set_target_properties(awkward-cuda-kernels-static PROPERTIES CXX_VISIBILITY_PRESET hidden)
  set_target_properties(awkward-cuda-kernels-static PROPERTIES VISIBILITY_INLINES_HIDDEN ON)
  set_target_properties(awkward-cuda-kernels PROPERTIES CXX_VISIBILITY_PRESET hidden)
  set_target_properties(awkward-cuda-kernels PROPERTIES VISIBILITY_INLINES_HIDDEN ON)
endif()

if (APPLE)
  set_target_properties(awkward-cpu-kernels PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "@loader_path")
  if(BUILD_CUDA_KERNELS)
    set_target_properties(awkward-cuda-kernels PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "@loader_path")
  endif()
else()
  set_target_properties(awkward-cpu-kernels PROPERTIES BUILD_RPATH_USE_ORIGIN TRUE)
  if(BUILD_CUDA_KERNELS)
    set_target_properties(awkward-cuda-kernels PROPERTIES BUILD_RPATH_USE_ORIGIN TRUE)
  endif()
endif()

# Second tier: libawkward (object files, static library, and dynamic library).
add_library(awkward-objects OBJECT ${LIBAWKWARD_SOURCES})
set_target_properties(awkward-objects PROPERTIES POSITION_INDEPENDENT_CODE 1)
if (${CMAKE_CXX_COMPILER_ID} MATCHES "^(|Apple)Clang$")
  # Avoid emitting vtables in the dependent libraries
  target_compile_options(awkward-objects PRIVATE -Werror=weak-vtables -Wweak-vtables)
endif()
add_library(awkward-static STATIC $<TARGET_OBJECTS:awkward-objects>)
set_property(TARGET awkward-static PROPERTY POSITION_INDEPENDENT_CODE ON)
add_library(awkward        SHARED $<TARGET_OBJECTS:awkward-objects>)
target_link_libraries(awkward-static PRIVATE awkward-cpu-kernels-static)
target_link_libraries(awkward        PRIVATE awkward-cpu-kernels-static)

if(BUILD_CUDA_KERNELS)
  target_link_libraries(awkward-static PRIVATE awkward-cuda-kernels-static)
  target_link_libraries(awkward        PRIVATE awkward-cuda-kernels-static)
endif()

set_target_properties(awkward-objects PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(awkward-objects PROPERTIES VISIBILITY_INLINES_HIDDEN ON)
set_target_properties(awkward-static PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(awkward-static PROPERTIES VISIBILITY_INLINES_HIDDEN ON)
set_target_properties(awkward PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(awkward PROPERTIES VISIBILITY_INLINES_HIDDEN ON)

if (APPLE)
  set_target_properties(awkward PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "@loader_path")
else()
  set_target_properties(awkward PROPERTIES BUILD_RPATH_USE_ORIGIN TRUE)
endif()

# Tests for second tier (these do not depend on Python).
addtest(test0016 tests/test_0016-finish-getitem-for-rawarray.cpp)
addtest(test0019 tests/test_0019-use-json-library.cpp)
addtest(test0030 tests/test_0030-recordarray-in-numba.cpp)

# Third tier: Python modules.
if (PYBUILD)
  add_subdirectory(pybind11)

  file(GLOB LAYOUT_SOURCES "src/python/*.cpp")
  pybind11_add_module(_ext ${LAYOUT_SOURCES})
  set_target_properties(_ext PROPERTIES CXX_VISIBILITY_PRESET hidden)
  set_target_properties(_ext PROPERTIES VISIBILITY_INLINES_HIDDEN ON)
  if (MSVC)
    target_link_libraries(_ext PRIVATE awkward-static)
  else()
    target_link_libraries(_ext PRIVATE awkward)
  endif()
  if (APPLE)
    set_target_properties(_ext PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "@loader_path")
  else()
    set_target_properties(_ext PROPERTIES BUILD_RPATH_USE_ORIGIN TRUE)
  endif()

  if(BUILD_CUDA_KERNELS)
    install(TARGETS awkward-static awkward awkward-cpu-kernels awkward-cpu-kernels-static awkward-cuda-kernels awkward-cuda-kernels-static
            _ext
            LIBRARY DESTINATION awkward1 ARCHIVE DESTINATION awkward1)
  else()
  install(TARGETS awkward-static awkward awkward-cpu-kernels awkward-cpu-kernels-static
          _ext
          LIBRARY DESTINATION awkward1 ARCHIVE DESTINATION awkward1)
  endif()

# Third tier: install without Python modules.
else()
  if(BUILD_CUDA_KERNELS)
    install(TARGETS awkward-static awkward awkward-cpu-kernels awkward-cpu-kernels-static awkward-cuda-kernels awkward-cuda-kernels-static
            LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
  else()
    install(TARGETS awkward-static awkward awkward-cpu-kernels awkward-cpu-kernels-static
            LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
  endif()
endif()
