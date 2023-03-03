# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

cmake_minimum_required(VERSION 3.15...3.24)

project(
  awkward-headers
  LANGUAGES CXX
  VERSION 1.0.0)

include(GNUInstallDirs)

add_library(${PROJECT_NAME} INTERFACE)
set_target_properties(
  ${PROJECT_NAME}
  PROPERTIES CXX_STANDARD 11
             CXX_STANDARD_REQUIRED YES
             CXX_EXTENSIONS NO)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)
target_include_directories(
  ${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                            $<INSTALL_INTERFACE:include>)

# For `add_subdirectory`
set(NAMESPACE "awkward::")
add_library(${NAMESPACE}awkward-headers ALIAS ${PROJECT_NAME})

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/awkward"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(
  TARGETS ${PROJECT_NAME}
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
