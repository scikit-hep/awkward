# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

cmake_minimum_required(VERSION 3.10...3.17)

project(cuda_kernels LANGUAGES CXX CUDA)

message(STATUS "CMake version " ${CMAKE_VERSION})

set(SIMPLE_TYPES Debug Release)
if(CMAKE_BUILD_TYPE IN_LIST SIMPLE_TYPES)
    message(STATUS "${CMAKE_BUILD_TYPE}")
else()
    message(STATUS "CMAKE_BUILD_TYPE = " ${CMAKE_BUILD_TYPE})
endif()

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_definitions( -DBUILD_CUDA_KERNELS )
else()
   message(FATAL_ERROR "NO CUDA COMPILER FOUND! PLEASE CHECK YOUR CUDA PATH, CMake will now exit")
endif()

# Is there a cleaner way to do this?
include_directories(../../include)

file(GLOB CUDA_KERNEL_SOURCES CONFIGURE_DEPENDS "src/identities.cpp")

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

install(TARGETS awkward-cuda-kernels awkward-cuda-kernels-static
        LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)