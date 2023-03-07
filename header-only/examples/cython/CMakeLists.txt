cmake_minimum_required(VERSION 3.15...3.26)

project(demo)

# Download headers include(FetchContent) set(AWKWARD_VERSION "v2.1.0") FetchContent_Declare(
# awkward-headers URL
# https://github.com/scikit-hep/awkward/releases/download/${AWKWARD_VERSION}/header-only.zip )
# FetchContent_MakeAvailable(awkward-headers)
add_subdirectory(/home/angus/Git/awkward/header-only header-only EXCLUDE_FROM_ALL)

find_package(Cython REQUIRED)
find_package(Python COMPONENTS Interpreter Development.Module)
find_package(PythonExtensions REQUIRED)

# Build demo module
add_cython_target(_demo demo.pyx PY3 CXX)
add_python_extension(_demo SOURCES ${_demo} LINK_LIBRARIES awkward::layout-builder)

install(TARGETS _demo LIBRARY DESTINATION demo)
