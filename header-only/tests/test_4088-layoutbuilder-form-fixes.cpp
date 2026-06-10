// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

// Regression tests for:
//   1. Union::form() emitting invalid JSON when parameters are set
//   2. Record<MAP,...>::form() failing to compile for non-std::map MAP types

#include "awkward/LayoutBuilder.h"

#include <cassert>
#include <string>
#include <unordered_map>

// ---- helpers ----------------------------------------------------------------

// Checks that the string contains no double-comma sequences.
static bool
no_double_comma(const std::string& s) {
  return s.find(",,") == std::string::npos;
}

// Checks that "parameters" appears in the string and comes before the *last*
// occurrence of "form_key" (which is the outer one in Union/Record forms).
// The outer form_key is always last because nested ones appear inside contents.
static bool
params_before_outer_form_key(const std::string& s) {
  auto params_pos = s.find("\"parameters\"");
  if (params_pos == std::string::npos) return false;
  auto outer_fk = s.rfind("\"form_key\"");
  if (outer_fk == std::string::npos) return false;
  return params_pos < outer_fk;
}

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
  assert(no_double_comma(form));

  // Must have "class": "UnionArray"
  assert(form.find("\"class\": \"UnionArray\"") != std::string::npos);

  // parameters must appear before the outer form_key
  assert(params_before_outer_form_key(form));

  // Must contain the parameter value we set
  assert(form.find("\"__array__\": \"union_test\"") != std::string::npos);

  // The outer form_key entry must be the very last key before the closing brace
  assert(form.find("\"form_key\": \"node0\" }") != std::string::npos);

  // Confirm proper separator: parameters block ends with "}, " before form_key
  assert(form.rfind("}, \"form_key\"") != std::string::npos);
}

// ---- test 2: Union::form() without parameters (unchanged behaviour) -------

void
test_union_form_without_parameters() {
  using UnionBuilder =
      awkward::LayoutBuilder::Union<int8_t, int64_t,
                                    awkward::LayoutBuilder::Numpy<float>,
                                    awkward::LayoutBuilder::Numpy<double>>;

  UnionBuilder builder;
  std::string form = builder.form();

  assert(no_double_comma(form));
  assert(form.find("\"class\": \"UnionArray\"") != std::string::npos);
  // No parameters key when none are set
  assert(form.find("\"parameters\"") == std::string::npos);
  assert(form.find("\"form_key\": \"node0\" }") != std::string::npos);
}

// ---- test 3: Record<unordered_map,...>::form() compiles and is consistent --

void
test_record_unordered_map() {
  using OrderedMap = std::map<std::size_t, std::string>;
  using UnorderedMap = std::unordered_map<std::size_t, std::string>;

  using Field0 = awkward::LayoutBuilder::Field<0, awkward::LayoutBuilder::Numpy<int32_t>>;
  using Field1 = awkward::LayoutBuilder::Field<1, awkward::LayoutBuilder::Numpy<double>>;

  // Build with std::map
  awkward::LayoutBuilder::Record<OrderedMap, Field0, Field1> ordered_rec(
      OrderedMap({{0, "x"}, {1, "y"}}));
  std::string ordered_form = ordered_rec.form();

  // Build with std::unordered_map — this must compile (was the bug)
  awkward::LayoutBuilder::Record<UnorderedMap, Field0, Field1> unordered_rec(
      UnorderedMap({{0, "x"}, {1, "y"}}));
  std::string unordered_form = unordered_rec.form();

  // Both must be valid JSON-like strings (no double commas)
  assert(no_double_comma(ordered_form));
  assert(no_double_comma(unordered_form));

  // Both must declare RecordArray
  assert(ordered_form.find("\"class\": \"RecordArray\"") != std::string::npos);
  assert(unordered_form.find("\"class\": \"RecordArray\"") != std::string::npos);

  // Both must contain the same field names
  assert(ordered_form.find("\"x\"") != std::string::npos);
  assert(ordered_form.find("\"y\"") != std::string::npos);
  assert(unordered_form.find("\"x\"") != std::string::npos);
  assert(unordered_form.find("\"y\"") != std::string::npos);
}

// ---- main -------------------------------------------------------------------

int
main(int /* argc */, char** /* argv */) {
  test_union_form_with_parameters();
  test_union_form_without_parameters();
  test_record_unordered_map();
  return 0;
}
