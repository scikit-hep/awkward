# BSD 3-Clause License; see
# https://github.com/scikit-hep/awkward/blob/main/LICENSE

cmake_minimum_required(VERSION 3.15...3.24)

project(
  awkward-headers
  LANGUAGES CXX
  VERSION 1.0.0)

include(GNUInstallDirs)

# Build tests?
option(BUILD_TESTS OFF)

# Add aliases for `add_subdirectory`
set(NAMESPACE "awkward::")

# We want to specify different compile options, so we split the headers into
# "components"
macro(add_component name)
  add_library(${name} INTERFACE)
  add_library(${NAMESPACE}${name} ALIAS ${name})
  target_include_directories(
    ${name} INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${name}>
                      $<INSTALL_INTERFACE:include/${name}>)
  target_compile_features(${name} INTERFACE cxx_std_17)

  install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${name}/awkward"
          DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${name}")
endmacro(add_component)

# Component: Builder Options
add_component(builder-options)
# Component: Growable Buffer
add_component(growable-buffer)
target_link_libraries(growable-buffer INTERFACE builder-options)
# Component: Layout Builder
add_component(layout-builder)
target_link_libraries(layout-builder INTERFACE growable-buffer builder-options)

# Build test suite?
if(BUILD_TESTS)
  add_subdirectory(tests)
endif(BUILD_TESTS)

# Installation
install(
  TARGETS layout-builder growable-buffer builder-options
  EXPORT ${PROJECT_NAME}Targets
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(
  EXPORT ${PROJECT_NAME}Targets
  FILE ${PROJECT_NAME}Config.cmake
  NAMESPACE ${NAMESPACE}
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${PROJECT_NAME}ConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion)
