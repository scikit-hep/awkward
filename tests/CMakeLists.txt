
macro(addtest name filename)
  if(BUILD_TESTING)
    add_executable(${name} ${filename})
    target_link_libraries(${name} PRIVATE awkward-static awkward-cpu-kernels-static)
    set_target_properties(${name} PROPERTIES CXX_VISIBILITY_PRESET hidden)
    set_target_properties(${name} PROPERTIES VISIBILITY_INLINES_HIDDEN ON)
    add_test(${name} ${name})
  endif()
endmacro(addtest)

# Tests for second tier (these do not depend on Python).
addtest(test0016 test_0016-finish-getitem-for-rawarray.cpp)
addtest(test0019 test_0019-use-json-library.cpp)
addtest(test0030 test_0030-recordarray-in-numba.cpp)
addtest(test0074 test_0074-argsort-and-sort-rawarray.cpp)
addtest(test0769 test_0769-typed-array-builder.cpp)
