# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

cmake_minimum_required(VERSION 3.15...3.24)

# Project must be near the top
project(
  ${SKBUILD_PROJECT_NAME}
  LANGUAGES CXX
  VERSION ${SKBUILD_PROJECT_VERSION})

message(STATUS "CMake version ${CMAKE_VERSION}")
message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")

# Defaults for properties in this directory (and below)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

if(APPLE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  set(CMAKE_INSTALL_RPATH "@loader_path")
elseif(EMSCRIPTEN)
  message(STATUS "Using Emscripten")
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
  # TODO: this can be removed with pyodide-build>=0.22.0a1
  set(CMAKE_STRIP "emstrip")
  string(APPEND CMAKE_CXX_FLAGS " -s DISABLE_EXCEPTION_CATCHING=0")
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

# C++ dependencies (header-only): RapidJSON
target_include_directories(awkward-parent INTERFACE rapidjson/include)

# C++ dependencies (header-only): GrowableBuffer
target_include_directories(awkward-parent INTERFACE header-only)

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

# Third tier: Python modules.
find_package(pybind11 CONFIG REQUIRED)

# Install python bindings
file(GLOB LAYOUT_SOURCES "src/python/*.cpp")
pybind11_add_module(_ext ${LAYOUT_SOURCES})
target_link_libraries(_ext PRIVATE awkward-static)

# Install pure-python files
file(GLOB_RECURSE PYTHON_SOURCES "src/${SKBUILD_PROJECT_NAME}/*.py")

install(
  TARGETS awkward awkward-parent awkward-cpu-kernels _ext
  LIBRARY DESTINATION "${SKBUILD_PROJECT_NAME}/lib"
  RUNTIME DESTINATION "${SKBUILD_PROJECT_NAME}/lib"
  ARCHIVE DESTINATION "${SKBUILD_PROJECT_NAME}/lib")

install(FILES ${PYTHON_SOURCES} DESTINATION ${SKBUILD_PROJECT_NAME})
