# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

cmake_minimum_required(VERSION 3.15...3.30)

# Project must be near the top
project(
  ${SKBUILD_PROJECT_NAME}
  LANGUAGES CXX
  VERSION ${SKBUILD_PROJECT_VERSION})

message(STATUS "CMake version ${CMAKE_VERSION}")
message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")

if(CMAKE_CXX_COMPILER_ID MATCHES AppleClang AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL
                                                15)
  if(CMAKE_VERSION VERSION_LESS 3.29)
    message(WARNING "CMake should be 3.29+ to support AppleClang 15+. Trying anyway.")
  endif()
  set(CMAKE_LINKER_TYPE APPLE_CLASSIC)
endif()

# Check for header-only libraries
if(NOT (EXISTS "header-only" AND EXISTS "include/awkward/kernels.h"))
  message(
    FATAL_ERROR
      "\
awkward-cpp relies upon generated and copied artefacts such as the header-only libraries and generated kernel headers. \
These could not be found, which indicates that\n\n\
    nox -s prepare\
\n\nwas skipped or failed. \
Please check https://github.com/scikit-hep/awkward#installation-for-developers to learn more about this process.\
")
endif()

# Setup the RPATH for built libraries
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
if(APPLE)
  set(CMAKE_INSTALL_RPATH "@loader_path")
else()
  set(CMAKE_INSTALL_RPATH "\$ORIGIN")
endif()

# Three tiers: [cpu-kernels (extern "C" interface), cuda-kernels (extern "C" interface)],
# libawkward (C++), and Python modules.
file(GLOB CPU_KERNEL_SOURCES CONFIGURE_DEPENDS "src/cpu-kernels/*.cpp")
file(GLOB_RECURSE LIBAWKWARD_SOURCES CONFIGURE_DEPENDS "src/libawkward/*.cpp")

# Shared properties
add_library(awkward-parent INTERFACE)
target_compile_definitions(awkward-parent INTERFACE VERSION_INFO="${SKBUILD_PROJECT_VERSION}")
target_include_directories(awkward-parent INTERFACE include)
target_compile_features(awkward-parent INTERFACE cxx_std_17)

# C++ dependencies (header-only): RapidJSON
target_include_directories(awkward-parent INTERFACE rapidjson/include)

# C++ dependencies (header-only): GrowableBuffer
add_subdirectory(header-only EXCLUDE_FROM_ALL)
target_link_libraries(awkward-parent INTERFACE awkward::growable-buffer)

# First tier: cpu-kernels
add_library(awkward-cpu-kernels SHARED ${CPU_KERNEL_SOURCES})
target_link_libraries(awkward-cpu-kernels PUBLIC awkward-parent)
set_target_properties(
  awkward-cpu-kernels
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             CXX_EXTENSIONS NO)

# Second tier: libawkward
add_library(awkward SHARED ${LIBAWKWARD_SOURCES})
if(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
  # Avoid emitting vtables in the dependent libraries
  target_compile_options(
    awkward
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
target_link_libraries(awkward PUBLIC awkward-parent)
set_target_properties(
  awkward
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             CXX_EXTENSIONS NO)

# Third tier: Python modules.
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Install python bindings
file(GLOB LAYOUT_SOURCES "src/python/*.cpp")
pybind11_add_module(_ext MODULE ${LAYOUT_SOURCES})
target_link_libraries(_ext PRIVATE awkward)
set_target_properties(
  _ext
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             CXX_EXTENSIONS NO)

# Install pure-python files
file(GLOB_RECURSE PYTHON_SOURCES "src/${SKBUILD_PROJECT_NAME}/*.py")

install(
  TARGETS awkward awkward-parent awkward-cpu-kernels _ext
  LIBRARY DESTINATION "${SKBUILD_PROJECT_NAME}/lib"
  RUNTIME DESTINATION "${SKBUILD_PROJECT_NAME}/lib"
  ARCHIVE DESTINATION "${SKBUILD_PROJECT_NAME}/lib")

install(FILES ${PYTHON_SOURCES} DESTINATION ${SKBUILD_PROJECT_NAME})
