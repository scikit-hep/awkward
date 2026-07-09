// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

// Regression tests for:
//   1. Union::form() emitting invalid JSON when parameters are set
//   2. Record<MAP,...>::form() failing to compile for non-std::map MAP types

#include "awkward/LayoutBuilder.h"

#include <cassert>
#include <string>
#include <unordered_map>

// ---- test 1: Union::form() with set_parameters ---------------------------

void
test_union_form_with_parameters() {
  using UnionBuilder =
      awkward::LayoutBuilder::Union<int8_t, int32_t,
                                    awkward::LayoutBuilder::Numpy<double>,
                                    awkward::LayoutBuilder::Numpy<int32_t>>;

  UnionBuilder builder;
  builder.set_parameters("\"__array__\": \"union_test\"");

  std::string form = builder.form();

  // Must contain no double-comma sequences (was the bug: "..., , " appeared)
  assert(form.find(",,") == std::string::npos);

  // parameters must appear before the outer form_key
  auto params_pos = form.find("\"parameters\"");
  assert(params_pos != std::string::npos);
  assert(params_pos < form.rfind("\"form_key\""));

  // Confirm proper separator: parameters block ends with "}, " before form_key
  assert(form.rfind("}, \"form_key\"") != std::string::npos);
}

// ---- test 2: Record<unordered_map,...>::form() compiles and is consistent --

void
test_record_unordered_map() {
  using UnorderedMap = std::unordered_map<std::size_t, std::string>;

  using Field0 = awkward::LayoutBuilder::Field<0, awkward::LayoutBuilder::Numpy<int32_t>>;
  using Field1 = awkward::LayoutBuilder::Field<1, awkward::LayoutBuilder::Numpy<double>>;

  // Build with std::unordered_map — this must compile (was the bug)
  awkward::LayoutBuilder::Record<UnorderedMap, Field0, Field1> rec(
      UnorderedMap({{0, "x"}, {1, "y"}}));
  std::string form = rec.form();

  assert(form.find(",,") == std::string::npos);
  assert(form.find("\"class\": \"RecordArray\"") != std::string::npos);
  assert(form.find("\"x\"") != std::string::npos);
  assert(form.find("\"y\"") != std::string::npos);
}

// ---- main -------------------------------------------------------------------

int
main(int /* argc */, char** /* argv */) {
  test_union_form_with_parameters();
  test_record_unordered_map();
  return 0;
}
