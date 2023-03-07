cmake_minimum_required(VERSION 3.15...3.26)

project(demo)

# Download headers
include(FetchContent)
set(AWKWARD_VERSION "v2.0.10")
FetchContent_Declare(
  awkward-headers
  URL https://github.com/scikit-hep/awkward/releases/download/${AWKWARD_VERSION}/header-only.zip)
FetchContent_MakeAvailable(awkward-headers)

find_package(Cython REQUIRED)
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(PythonExtensions REQUIRED)
include(UsePythonExtensions)

# Build demo module
add_cython_target(_demo _demo.pyx PY3 CXX)
add_python_library(_demo MODULE SOURCES ${_demo} LINK_LIBRARIES awkward::layout-builder)
python_extension_module(_demo)

install(
  TARGETS _demo
  RUNTIME DESTINATION demo
  LIBRARY DESTINATION demo
  ARCHIVE DESTINATION demo)
