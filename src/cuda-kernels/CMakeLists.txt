# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

cmake_minimum_required(VERSION 3.10...3.17)

# Project must be near the top
project(awkward_cuda_kernels LANGUAGES CXX)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
else()
  message(FATAL_ERROR "NO CUDA COMPILER FOUND! PLEASE CHECK YOUR CUDA PATH, CMake will now exit")
endif()

# We can avoid this by introducing this CMakeLists to the top level CMakeLists
include_directories(../../include)

file(GLOB CUDA_KERNEL_SOURCES CONFIGURE_DEPENDS "src/*.cu")

include(CTest)

add_library(awkward-cuda-kernels-objects OBJECT ${CUDA_KERNEL_SOURCES})
set_target_properties(awkward-cuda-kernels-objects PROPERTIES POSITION_INDEPENDENT_CODE 1)
add_library(awkward-cuda-kernels-static STATIC $<TARGET_OBJECTS:awkward-cuda-kernels-objects>)
set_property(TARGET awkward-cuda-kernels-static PROPERTY POSITION_INDEPENDENT_CODE ON)
add_library(awkward-cuda-kernels        SHARED $<TARGET_OBJECTS:awkward-cuda-kernels-objects>)
set_target_properties(awkward-cuda-kernels-objects PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(awkward-cuda-kernels-objects PROPERTIES VISIBILITY_INLINES_HIDDEN ON)
set_target_properties(awkward-cuda-kernels-static PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(awkward-cuda-kernels-static PROPERTIES VISIBILITY_INLINES_HIDDEN ON)
set_target_properties(awkward-cuda-kernels PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(awkward-cuda-kernels PROPERTIES VISIBILITY_INLINES_HIDDEN ON)

install(TARGETS awkward-cuda-kernels awkward-cuda-kernels-static)




