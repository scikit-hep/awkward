# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

cmake_minimum_required(VERSION 3.15...3.24)

project(awkward-headers LANGUAGES CXX)

include(GNUInstallDirs)

add_library(${PROJECT_NAME} INTERFACE)
set_target_properties(
  ${PROJECT_NAME}
  PROPERTIES CXX_STANDARD 11
             CXX_STANDARD_REQUIRED YES
             CXX_EXTENSIONS NO)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

target_include_directories(
  ${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${PROJECT_NAME}/include>
                            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
# For `add_subdirectory`
add_library(awkward::awkward-headers ALIAS ${PROJECT_NAME})

install(
  TARGETS ${PROJECT_NAME}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
