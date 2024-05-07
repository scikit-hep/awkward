# BSD 3-Clause License; see
# https://github.com/scikit-hep/awkward/blob/main/LICENSE

macro(addtest_nolibs name filename)
  add_executable(${name} ${filename})
  set_target_properties(
    ${name}
    PROPERTIES CXX_VISIBILITY_PRESET hidden
               VISIBILITY_INLINES_HIDDEN ON
               CXX_EXTENSIONS NO)
  add_test(${name} ${name})
endmacro(addtest_nolibs)

addtest_nolibs(test_1494-layout-builder test_1494-layout-builder.cpp)
addtest_nolibs(test_1542-growable-buffer test_1542-growable-buffer.cpp)
addtest_nolibs(test_1560-builder-options test_1560-builder-options.cpp)
addtest_nolibs(test_3091-layout-builder-is-valid
               test_3091-layout-builder-is-valid.cpp)

target_link_libraries(test_1494-layout-builder PRIVATE awkward::layout-builder)
target_link_libraries(test_1542-growable-buffer
                      PRIVATE awkward::growable-buffer)
target_link_libraries(test_1560-builder-options
                      PRIVATE awkward::builder-options)
target_link_libraries(test_3091-layout-builder-is-valid
                      PRIVATE awkward::layout-builder)
