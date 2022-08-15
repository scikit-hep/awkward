# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

cmake_minimum_required(VERSION 3.14...3.18)

# VERSION_INFO is the version stamp for everything (all C++, all Python).
file(READ "VERSION_INFO" VERSION_INFO)
string(STRIP ${VERSION_INFO} VERSION_INFO)
string(REPLACE "rc" "." VERSION_INFO_SIMPLE "${VERSION_INFO}")

# Project must be near the top
project(
  awkward
  LANGUAGES CXX
  VERSION ${VERSION_INFO_SIMPLE})

message(STATUS "CMake version ${CMAKE_VERSION}")
message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")

include(CMakeDependentOption)

# Defaults for properties in this directory (and below)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

if(APPLE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  set(CMAKE_INSTALL_RPATH "@loader_path")
else()
  set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
endif()

# Three tiers: [cpu-kernels (extern "C" interface), cuda-kernels (extern "C" interface)],
# libawkward (C++), and Python modules.
file(GLOB CPU_KERNEL_SOURCES CONFIGURE_DEPENDS "src/cpu-kernels/*.cpp")
file(GLOB_RECURSE LIBAWKWARD_SOURCES CONFIGURE_DEPENDS "src/libawkward/*.cpp")

# Shared properties
add_library(awkward-parent INTERFACE)
target_compile_definitions(awkward-parent INTERFACE VERSION_INFO="${VERSION_INFO}")
target_include_directories(awkward-parent INTERFACE include)
target_compile_features(awkward-parent INTERFACE cxx_std_11)

# C++ dependencies (header-only): RapidJSON and dlpack
target_include_directories(awkward-parent INTERFACE rapidjson/include dlpack/include)

# C++ dependencies (header-only): GrowableBuffer
target_include_directories(awkward-parent INTERFACE src/awkward/_v2/cpp-headers/)

# First tier: cpu-kernels (object files, static library, and dynamic library).
add_library(awkward-cpu-kernels-objects OBJECT ${CPU_KERNEL_SOURCES})
set_target_properties(awkward-cpu-kernels-objects PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(awkward-cpu-kernels-objects PUBLIC awkward-parent)

add_library(awkward-cpu-kernels-static STATIC $<TARGET_OBJECTS:awkward-cpu-kernels-objects>)
set_property(TARGET awkward-cpu-kernels-static PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(awkward-cpu-kernels-static PUBLIC awkward-parent)

add_library(awkward-cpu-kernels SHARED $<TARGET_OBJECTS:awkward-cpu-kernels-objects>)
target_link_libraries(awkward-cpu-kernels PUBLIC awkward-parent)

# Second tier: libawkward (object files, static library, and dynamic library).
add_library(awkward-objects OBJECT ${LIBAWKWARD_SOURCES})
set_target_properties(awkward-objects PROPERTIES POSITION_INDEPENDENT_CODE 1)
target_compile_definitions(awkward-objects PRIVATE LIBAWKWARD_EXPORT_SYMBOL=EXPORT_SYMBOL)
if(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
  # Avoid emitting vtables in the dependent libraries
  target_compile_options(
    awkward-objects
    PRIVATE -Werror=weak-vtables
            -Wweak-vtables
            -Wshorten-64-to-32
            -Wsign-compare
            -Wsign-conversion
            -Wshift-sign-overflow
            -Wreorder
            -Wrange-loop-analysis
            -Wconversion
            -Wunused)
endif()
target_link_libraries(awkward-objects PUBLIC awkward-parent)

add_library(awkward-static STATIC $<TARGET_OBJECTS:awkward-objects>)
set_property(TARGET awkward-static PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(awkward-static PRIVATE awkward-cpu-kernels-static ${CMAKE_DL_LIBS})
target_link_libraries(awkward-static PUBLIC awkward-parent)

add_library(awkward SHARED $<TARGET_OBJECTS:awkward-objects>)
target_link_libraries(awkward PRIVATE awkward-cpu-kernels-static ${CMAKE_DL_LIBS})
target_link_libraries(awkward PUBLIC awkward-parent)

# Tests Macro to add C++ tests (part of CMake build, distinct from pytests in Python).
include(CTest)

if(BUILD_TESTING)
  add_subdirectory(tests-cpp)
  add_subdirectory(tests)

  addtest_nolibs(test_1494-layout-builder tests-cpp/test_1494-layout-builder.cpp)
  addtest_nolibs(test_1542-growable-buffer tests-cpp/test_1542-growable-buffer.cpp)
  addtest(test_1542-array-builder tests-cpp/test_1542-array-builder.cpp)
  addtest_nolibs(test_1560-builder-options tests-cpp/test_1560-builder-options.cpp)

endif()

option(PYBUILD "Build Python modules")
cmake_dependent_option(AWKWARD_EXTERNAL_PYBIND11 "Build against an external pybind11" OFF
                       "PYBUILD" OFF)

# Third tier: Python modules.
if(PYBUILD)
  if(AWKWARD_EXTERNAL_PYBIND11)
    find_package(pybind11 CONFIG REQUIRED)
  else()
    add_subdirectory(pybind11)
  endif()

  file(GLOB LAYOUT_SOURCES "src/python/*.cpp")
  pybind11_add_module(_ext ${LAYOUT_SOURCES})
  target_link_libraries(_ext PRIVATE awkward-static)

  install(
    TARGETS awkward awkward-parent awkward-cpu-kernels _ext
    LIBRARY DESTINATION awkward
    ARCHIVE DESTINATION awkward)

  # Third tier: install without Python modules.
else()
  install(
    TARGETS awkward-static awkward awkward-parent awkward-cpu-kernels awkward-cpu-kernels-static
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
endif()
