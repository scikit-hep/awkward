# BSD 3-Clause License; see
# https://github.com/scikit-hep/awkward/blob/main/LICENSE

cmake_minimum_required(VERSION 3.15...3.26)

project(demo LANGUAGES CXX C)

# Download headers
include(FetchContent)
set(AWKWARD_VERSION "v2.4.3")
FetchContent_Declare(
  awkward-headers
  URL https://github.com/scikit-hep/awkward/releases/download/${AWKWARD_VERSION}/header-only.zip
)
# Instead of using `FetchContent_MakeAvailable(awkward-headers)`, we manually
# load the target so that we can EXCLUDE_FROM_ALL
FetchContent_GetProperties(awkward-headers)
if(NOT awkward-headers_POPULATED)
  FetchContent_Populate(awkward-headers)
  add_subdirectory(${awkward-headers_SOURCE_DIR} ${awkward-headers_BINARY_DIR}
                   EXCLUDE_FROM_ALL)
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)
find_package(Cython REQUIRED)
find_package(PythonExtensions REQUIRED)

# Build demo module
add_cython_target(_demo _demo.pyx PY3 CXX)
add_python_library(
  _demo
  MODULE
  SOURCES
  ${_demo}
  demo_impl.cpp
  LINK_LIBRARIES
  awkward::layout-builder
  Python::NumPy
  INCLUDE_DIRECTORIES
  include)
python_extension_module(_demo)

install(
  TARGETS _demo
  RUNTIME DESTINATION demo
  LIBRARY DESTINATION demo
  ARCHIVE DESTINATION demo)
