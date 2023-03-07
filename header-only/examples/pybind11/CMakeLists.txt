cmake_minimum_required(VERSION 3.15...3.26)

project(
  ${SKBUILD_PROJECT_NAME}
  LANGUAGES CXX
  VERSION ${SKBUILD_PROJECT_VERSION})

# Download headers
include(FetchContent)
set(AWKWARD_VERSION "v2.0.10")
FetchContent_Declare(
  awkward-headers
  URL https://github.com/scikit-hep/awkward/releases/download/${AWKWARD_VERSION}/header-only.zip)
FetchContent_MakeAvailable(awkward-headers)

# Load interpreter & pybind11
find_package(Python COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG)

# Build demo module
pybind11_add_module(demo demo.cpp)
target_link_libraries(demo PRIVATE awkward::layout-builder)

# Install into wheel
install(TARGETS demo DESTINATION .)
