# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

cmake_minimum_required(VERSION 3.15...3.30)

if(NOT DEFINED SKBUILD)
  set(SKBUILD_PROJECT_NAME awkward_cpp)
  set(SKBUILD_PROJECT_VERSION 0.0.0)
endif()

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

# Optional: compile with -O2 instead of the default Release -O3. The flag is appended after the
# build-type flags, so the trailing -O2 overrides the earlier -O3 (GCC/Clang honour the last -O on
# the command line). Useful for checking that optimisation level doesn't change kernel results, and
# for faster builds. Enable with: python -m pip install -v ./awkward-cpp -C
# cmake.define.AWKWARD_O2=ON
option(AWKWARD_O2 "Compile with -O2 instead of the default -O3" OFF)
if(AWKWARD_O2)
  if(MSVC)
    add_compile_options(/O2)
  else()
    add_compile_options(-O2)
  endif()
  message(STATUS "AWKWARD_O2 enabled: compiling with -O2 (overrides the default -O3)")
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# Generated artefacts (header-only copies, kernels.h, kernel signature tables, and kernel tests)
# are produced from sources at the repository root. A git checkout has the sources but not the
# artefacts; an sdist ships the artefacts but not the sources.
option(AWKWARD_PREPARE
       "Generate kernel artefacts from the repository sources when they are available" ON)
set(AWKWARD_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
if(AWKWARD_PREPARE AND EXISTS "${AWKWARD_REPO_ROOT}/kernel-specification.yml")
  set(prepare_scripts "dev/copy-cpp-headers.py" "dev/generate-kernel-signatures.py")
  set(prepare_inputs "${AWKWARD_REPO_ROOT}/kernel-specification.yml")

  # The kernel tests are only shipped in the sdist, not the wheel; generating them additionally
  # needs numpy, which some cross-compiling build environments (pyodide) cannot provide.
  if(SKBUILD_STATE STREQUAL "sdist")
    list(APPEND prepare_scripts "dev/generate-tests.py")
    list(APPEND prepare_inputs "${AWKWARD_REPO_ROOT}/kernel-test-data.json")
  endif()

  # Rerun configure (and hence generation) when any generation input changes
  foreach(script IN LISTS prepare_scripts)
    list(APPEND prepare_inputs "${AWKWARD_REPO_ROOT}/${script}")
  endforeach()
  file(GLOB_RECURSE header_only_sources CONFIGURE_DEPENDS "${AWKWARD_REPO_ROOT}/header-only/*")
  list(APPEND prepare_inputs ${header_only_sources})
  set_property(
    DIRECTORY
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS ${prepare_inputs})

  # The scripts leave unchanged outputs untouched (content compare / mtime-preserving copies), so
  # running them on every configure doesn't trigger needless recompilation.
  foreach(script IN LISTS prepare_scripts)
    message(STATUS "Generating kernel artefacts: ${script}")
    execute_process(COMMAND "${Python_EXECUTABLE}" "${AWKWARD_REPO_ROOT}/${script}"
                    RESULT_VARIABLE prepare_result)
    if(NOT prepare_result EQUAL 0)
      message(
        FATAL_ERROR
          "${script} failed (exit code ${prepare_result}). Generating awkward-cpp's kernel \
artefacts requires 'pyyaml' (plus 'numpy' when generating the kernel tests for an sdist) in the \
build environment; these are installed automatically for isolated builds. Pass \
-DAWKWARD_PREPARE=OFF to skip generation if the artefacts already exist.")
    endif()
  endforeach()
endif()

# Check for generated artefacts (pre-generated in an sdist, created above in a git checkout)
if(NOT (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/header-only"
        AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/awkward/kernels.h"))
  message(
    FATAL_ERROR
      "\
awkward-cpp relies upon generated and copied artefacts such as the header-only libraries and generated kernel headers. \
These could not be found and could not be generated (generation requires the repository sources at '${AWKWARD_REPO_ROOT}'). \
Please check https://github.com/scikit-hep/awkward#installation-for-developers to learn more about this process.\
")
endif()

# Setup the RPATH for built libraries
if(APPLE)
  set(CMAKE_INSTALL_RPATH "@loader_path")
else()
  set(CMAKE_INSTALL_RPATH "\$ORIGIN")
endif()
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)

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

# WASM needs exceptions enabled
if(CMAKE_SYSTEM_NAME MATCHES Emscripten)
  set_property(
    TARGET awkward-parent
    APPEND
    PROPERTY INTERFACE_LINK_OPTIONS -fexceptions)
  set_property(
    TARGET awkward-parent
    APPEND
    PROPERTY INTERFACE_COMPILE_OPTIONS -fexceptions)
endif()

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

# This has to be passed explicitly to make Pyodide 0.28 happy
if(CMAKE_SYSTEM_NAME MATCHES Emscripten)
  target_link_options(_ext PRIVATE "-Wl,-rpath=\$ORIGIN")
endif()

# Install pure-python files
file(GLOB_RECURSE PYTHON_SOURCES "src/${SKBUILD_PROJECT_NAME}/*.py")

install(
  TARGETS awkward awkward-parent awkward-cpu-kernels _ext
  LIBRARY DESTINATION "${SKBUILD_PROJECT_NAME}/lib"
  RUNTIME DESTINATION "${SKBUILD_PROJECT_NAME}/lib"
  ARCHIVE DESTINATION "${SKBUILD_PROJECT_NAME}/lib")

install(FILES ${PYTHON_SOURCES} DESTINATION ${SKBUILD_PROJECT_NAME})
